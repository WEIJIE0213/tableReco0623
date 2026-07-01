#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用训练好的 LoRA 模型在 test 集上批量推理，输出 {image, pred, gold} 的 jsonl。
独立用 transformers + peft（不依赖 swift CLI），稳定。

用法（项目根目录、tablereco 环境，单卡）：
  CUDA_VISIBLE_DEVICES=1 MAX_PIXELS=401408 python src/infer_test.py \
      --adapter checkpoints/stage1_lora/v0-XXXX/checkpoint-200
"""
import argparse, json, os
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 容忍轻度截断的 PNG

DEFAULT_BASE = os.path.expanduser(
    "~/.cache/modelscope/hub/models/Qwen/Qwen2___5-VL-3B-Instruct")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE, help="基座模型本地目录")
    ap.add_argument("--adapter", required=True, help="LoRA checkpoint 目录")
    ap.add_argument("--test", default="data/jsonl/test.jsonl")
    ap.add_argument("--out", default="data/eval/pred_test.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0, help=">0 时只跑前 N 条（调试）")
    ap.add_argument("--num-shards", type=int, default=1, help="多卡并行时的分片总数")
    ap.add_argument("--shard-id", type=int, default=0, help="本进程负责的分片编号(0..num_shards-1)")
    args = ap.parse_args()
    os.environ.setdefault("MAX_PIXELS", "401408")

    proc = AutoProcessor.from_pretrained(args.base)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="cuda:0")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    lines = open(args.test, encoding="utf-8").read().splitlines()
    if args.limit:
        lines = lines[:args.limit]
    if args.num_shards > 1:
        lines = [ln for k, ln in enumerate(lines) if k % args.num_shards == args.shard_id]
        print(f"[shard {args.shard_id}/{args.num_shards}] 负责 {len(lines)} 条", flush=True)

    n = 0
    with open(args.out, "w", encoding="utf-8") as w:
        for i, ln in enumerate(lines):
            s = json.loads(ln)
            img = s["images"][0]
            prompt = s["messages"][0]["content"].replace("<image>", "").strip()
            gold = s["messages"][1]["content"]
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt}]}]
            try:
                text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = proc(text=[text], images=image_inputs, videos=video_inputs,
                              padding=True, return_tensors="pt").to("cuda:0")
                with torch.no_grad():
                    gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                trimmed = gen[0][inputs.input_ids.shape[1]:]
                pred = proc.decode(trimmed, skip_special_tokens=True)
            except Exception as e:
                pred = ""
                print(f"[warn] {img} 推理失败: {e!r}")
            w.write(json.dumps({"image": img, "pred": pred, "gold": gold}, ensure_ascii=False) + "\n")
            w.flush()
            n += 1
            print(f"[{n}/{len(lines)}] {os.path.basename(img)}  pred_len={len(pred)}", flush=True)
    print("saved:", args.out)

if __name__ == "__main__":
    main()
