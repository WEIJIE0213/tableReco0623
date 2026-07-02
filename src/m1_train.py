#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M1 训练：Qwen2.5-VL-3B(LoRA) + 结构头（跨度/空标志），联合 LM 损失。

数据沿用内容一的 jsonl（{messages:[user,assistant], images:[path]}）。
结构目标直接从 assistant 的 HTML 解析（与 <td 锚点严格对齐，无需回查 label）。

单卡示例（先 smoke 跑通）：
  CUDA_VISIBLE_DEVICES=1 MAX_PIXELS=401408 python src/m1_train.py \
      --train data/jsonl/train.jsonl --val data/jsonl/val.jsonl \
      --out checkpoints/m1/smoke --max-steps 20 --log-steps 2 --grad-checkpoint

全量：
  CUDA_VISIBLE_DEVICES=1 MAX_PIXELS=401408 python src/m1_train.py \
      --train data/jsonl/train.jsonl --val data/jsonl/val.jsonl \
      --out checkpoints/m1/v1-$(date +%m%d-%H%M) --epochs 2 --grad-checkpoint
"""
import argparse, json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 容忍轻度截断的 PNG，避免单张坏图中断训练

from struct_targets import (targets_from_html, td_char_starts,
                            anchor_token_indices, locate_html_base)

DEFAULT_BASE = os.path.expanduser(
    "~/.cache/modelscope/hub/models/Qwen/Qwen2___5-VL-3B-Instruct")


class JsonlDS(Dataset):
    def __init__(self, path, limit=0):
        self.rows = open(path, encoding="utf-8").read().splitlines()
        if limit:
            self.rows = self.rows[:limit]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        s = json.loads(self.rows[i])
        return {
            "image": s["images"][0],
            "prompt": s["messages"][0]["content"].replace("<image>", "").strip(),
            "html": s["messages"][1]["content"],
        }


class Collator:
    """构建模型输入 + 每个 <td 的全局锚点 token 索引 + 结构目标（自校验）。"""
    def __init__(self, proc, max_new=None):
        self.proc = proc
        self.tok = proc.tokenizer
        self.tok.padding_side = "right"
        # 显式控制视觉 token 上限（不依赖 env 是否被 qwen_vl_utils 读取）
        self.max_pixels = int(os.environ.get("MAX_PIXELS", "401408"))
        self.min_pixels = int(os.environ.get("MIN_PIXELS", "50176"))

    def _encode(self, image, prompt, html):
        from qwen_vl_utils import process_vision_info
        user_msg = [{"role": "user", "content": [
            {"type": "image", "image": image,
             "max_pixels": self.max_pixels, "min_pixels": self.min_pixels},
            {"type": "text", "text": prompt}]}]
        full_msg = user_msg + [{"role": "assistant", "content": html}]

        prompt_text = self.proc.apply_chat_template(
            user_msg, tokenize=False, add_generation_prompt=True)
        full_text = self.proc.apply_chat_template(
            full_msg, tokenize=False, add_generation_prompt=False)

        image_inputs, _ = process_vision_info(user_msg)
        enc = self.proc(text=[full_text], images=image_inputs,
                        padding=False, return_tensors="pt")
        # prompt 长度（同图，用于 mask 标签）
        penc = self.proc(text=[prompt_text], images=image_inputs,
                         padding=False, return_tensors="pt")
        plen = penc["input_ids"].shape[1]

        input_ids = enc["input_ids"][0]
        labels = input_ids.clone()
        labels[:plen] = -100

        # 结构目标（从 html 解析，顺序 == <td 顺序）
        tgt = targets_from_html(html)

        # 锚点：assistant 区应与 html 单独分词一致
        html_ids = self.tok(html, add_special_tokens=False,
                            return_offsets_mapping=True)
        hid = torch.tensor(html_ids["input_ids"])
        anchors = []
        ok = (plen + hid.shape[0] <= input_ids.shape[0]) and \
             torch.equal(input_ids[plen:plen + hid.shape[0]], hid)
        if ok:
            local = anchor_token_indices(html_ids["offset_mapping"],
                                         td_char_starts(html))
            if None not in local and len(local) == tgt["n"]:
                anchors = [plen + t for t in local]
        # 校验失败 → 该样本只用 LM 损失
        if not anchors:
            tgt = {"rowspan_cls": [], "colspan_cls": [], "empty": [], "role": [], "n": 0}

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": enc["attention_mask"][0],
            "pixel_values": enc["pixel_values"],
            "image_grid_thw": enc["image_grid_thw"],
            "anchors": anchors,
            "rowspan_t": torch.tensor(tgt["rowspan_cls"], dtype=torch.long),
            "colspan_t": torch.tensor(tgt["colspan_cls"], dtype=torch.long),
            "empty_t": torch.tensor(tgt["empty"], dtype=torch.long),
            "role_t": torch.tensor(tgt["role"], dtype=torch.long),
        }

    def __call__(self, batch):
        items = []
        for b in batch:
            try:
                items.append(self._encode(b["image"], b["prompt"], b["html"]))
            except Exception as e:
                print(f"[skip] 坏样本 {b.get('image')}: {e!r}", flush=True)
        if not items:
            return None  # 整批都坏 → 训练循环跳过
        maxlen = max(it["input_ids"].shape[0] for it in items)
        pad_id = self.tok.pad_token_id or 0
        B = len(items)
        input_ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
        labels = torch.full((B, maxlen), -100, dtype=torch.long)
        attn = torch.zeros((B, maxlen), dtype=torch.long)
        for b, it in enumerate(items):
            L = it["input_ids"].shape[0]
            input_ids[b, :L] = it["input_ids"]
            labels[b, :L] = it["labels"]
            attn[b, :L] = it["attention_mask"]
        pixel_values = torch.cat([it["pixel_values"] for it in items], 0)
        image_grid_thw = torch.cat([it["image_grid_thw"] for it in items], 0)
        return {
            "input_ids": input_ids, "labels": labels, "attention_mask": attn,
            "pixel_values": pixel_values, "image_grid_thw": image_grid_thw,
            "anchors": [it["anchors"] for it in items],
            "rowspan_t": [it["rowspan_t"] for it in items],
            "colspan_t": [it["colspan_t"] for it in items],
            "empty_t": [it["empty_t"] for it in items],
            "role_t": [it["role_t"] for it in items],
        }


def move(batch, device):
    for k in ("input_ids", "labels", "attention_mask", "pixel_values", "image_grid_thw"):
        batch[k] = batch[k].to(device)
    return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--train", default="data/jsonl/train.jsonl")
    ap.add_argument("--val", default="data/jsonl/val.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=0, help=">0 时覆盖 epochs（smoke）")
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--lambda-span", type=float, default=0.5)
    ap.add_argument("--lambda-empty", type=float, default=0.5)
    ap.add_argument("--lambda-role", type=float, default=0.0)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--log-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=0, help=">0 时按步存")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()
    os.environ.setdefault("MAX_PIXELS", "401408")

    device = "cuda:0"
    proc = AutoProcessor.from_pretrained(args.base)

    from m1_model import StructVLM
    model = StructVLM(args.base, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
                      lora_dropout=args.lora_dropout, lambda_span=args.lambda_span,
                      lambda_empty=args.lambda_empty, lambda_role=args.lambda_role)
    model.to(device)
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.model.enable_input_require_grads()
    model.train()

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[m1] 可训练参数量: {n_train/1e6:.2f}M", flush=True)

    ds = JsonlDS(args.train, limit=args.limit)
    coll = Collator(proc)
    dl = DataLoader(ds, batch_size=args.bsz, shuffle=True, collate_fn=coll,
                    num_workers=args.num_workers, drop_last=False)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    steps_per_epoch = math.ceil(len(dl) / args.grad_accum)
    total_steps = args.max_steps if args.max_steps > 0 else int(steps_per_epoch * args.epochs)
    from transformers import get_cosine_schedule_with_warmup
    warmup = max(1, int(total_steps * args.warmup_ratio))
    sched = get_cosine_schedule_with_warmup(opt, warmup, total_steps)
    print(f"[m1] 样本 {len(ds)} | 每 epoch 优化步 {steps_per_epoch} | 计划步数 {total_steps} "
          f"| warmup {warmup} | lr {args.lr}", flush=True)

    step = 0
    micro = 0
    t0 = time.time()
    accum = {"lm": 0.0, "span": 0.0, "empty": 0.0, "anchor": 0}
    done = False
    opt.zero_grad(set_to_none=True)
    while not done:
        for batch in dl:
            if batch is None:
                continue
            batch = move(batch, device)
            out = model(**batch)
            loss = out["loss"] / args.grad_accum
            loss.backward()
            accum["lm"] += float(out["lm_loss"])
            accum["span"] += float(out["span_loss"])
            accum["empty"] += float(out["empty_loss"])
            accum["anchor"] += int(out["n_anchor"])
            micro += 1
            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_steps == 0:
                    g = args.grad_accum
                    dt = time.time() - t0
                    print(f"[step {step}/{total_steps}] loss(lm={accum['lm']/g:.4f} "
                          f"span={accum['span']/g:.4f} empty={accum['empty']/g:.4f}) "
                          f"anchors/micro={accum['anchor']/g:.1f} {dt:.1f}s", flush=True)
                accum = {"lm": 0.0, "span": 0.0, "empty": 0.0, "anchor": 0}
                if args.save_steps and step % args.save_steps == 0:
                    d = os.path.join(args.out, f"checkpoint-{step}")
                    model.save_pretrained(d); print(f"[m1] 保存 {d}", flush=True)
                if step >= total_steps:
                    done = True
                    break
    model.save_pretrained(args.out)
    print(f"[m1] 训练完成，保存 -> {args.out}  用时 {(time.time()-t0)/60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
