#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M1 结构分支模型：Qwen2.5-VL-3B (LoRA) + 轻量结构头。

思路：主分支照常自回归生成 HTML（LM 损失）。同时取解码器最后一层隐状态，
在每个 <td 的锚点 token 上并联：
  - 跨度头 span_head：预测 rowspan / colspan（各 MAX_SPAN 类）
  - 空标志头 empty_head：预测该单元格是否为空（2 类）
  - （可选）角色头 role_head：为后续 M2 表头层级预留
联合损失：L = L_html + λ_span·(L_rowspan+L_colspan)/2 + λ_empty·L_empty (+ λ_role·L_role)

不修改 Qwen 内部 forward，只消费其 hidden_states，工程风险低、可回退。
底座走 LoRA，结构头全量训练。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model

from struct_targets import MAX_SPAN, ROLE2ID


class StructHeads(nn.Module):
    """挂在隐状态上的轻量结构头（fp32，数值稳定）。"""
    def __init__(self, hidden_size, max_span=MAX_SPAN, n_roles=len(ROLE2ID), dropout=0.1):
        super().__init__()
        h = hidden_size
        self.proj = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Dropout(dropout))
        self.rowspan = nn.Linear(h, max_span)
        self.colspan = nn.Linear(h, max_span)
        self.empty = nn.Linear(h, 2)
        self.role = nn.Linear(h, n_roles)

    def forward(self, feats):  # feats: (M, H) fp32
        z = self.proj(feats)
        return {
            "rowspan": self.rowspan(z),
            "colspan": self.colspan(z),
            "empty": self.empty(z),
            "role": self.role(z),
        }


class StructVLM(nn.Module):
    def __init__(self, base_dir, lora_rank=16, lora_alpha=32, lora_dropout=0.05,
                 freeze_vit=True, lambda_span=0.5, lambda_empty=0.5, lambda_role=0.0,
                 dtype=torch.bfloat16):
        super().__init__()
        self.lambda_span = lambda_span
        self.lambda_empty = lambda_empty
        self.lambda_role = lambda_role

        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_dir, dtype=dtype)
        # LoRA 只挂在语言解码器的注意力/MLP 线性层上
        target = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
        lcfg = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                          target_modules=target, task_type="CAUSAL_LM")
        self.model = get_peft_model(base, lcfg)
        if freeze_vit:
            for n, p in self.model.named_parameters():
                if "visual" in n:
                    p.requires_grad = False

        hidden = base.config.hidden_size if hasattr(base.config, "hidden_size") \
            else base.config.text_config.hidden_size
        self.heads = StructHeads(hidden)  # fp32 头

    def gradient_checkpointing_enable(self, **kw):
        self.model.gradient_checkpointing_enable(**kw)

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                pixel_values=None, image_grid_thw=None,
                anchors=None, rowspan_t=None, colspan_t=None, empty_t=None, role_t=None,
                **kw):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         labels=labels, pixel_values=pixel_values,
                         image_grid_thw=image_grid_thw,
                         output_hidden_states=True, use_cache=False)
        lm_loss = out.loss
        hs = out.hidden_states[-1]  # (B, T, H)

        # 收集全 batch 的锚点特征与目标
        feats, r_t, c_t, e_t, ro_t = [], [], [], [], []
        if anchors is not None:
            for b, idxs in enumerate(anchors):
                if not idxs:
                    continue
                ii = torch.as_tensor(idxs, device=hs.device, dtype=torch.long)
                feats.append(hs[b].index_select(0, ii))
                r_t.append(rowspan_t[b]); c_t.append(colspan_t[b])
                e_t.append(empty_t[b]);   ro_t.append(role_t[b])

        zero = lm_loss.new_zeros(())
        span_loss = empty_loss = role_loss = zero
        n_anchor = 0
        if feats:
            feats = torch.cat(feats, 0).float()
            r_t = torch.cat(r_t, 0).to(hs.device)
            c_t = torch.cat(c_t, 0).to(hs.device)
            e_t = torch.cat(e_t, 0).to(hs.device)
            ro_t = torch.cat(ro_t, 0).to(hs.device)
            n_anchor = feats.shape[0]
            pred = self.heads(feats)
            span_loss = 0.5 * (F.cross_entropy(pred["rowspan"], r_t)
                               + F.cross_entropy(pred["colspan"], c_t))
            empty_loss = F.cross_entropy(pred["empty"], e_t)
            role_loss = F.cross_entropy(pred["role"], ro_t)

        total = lm_loss + self.lambda_span * span_loss + self.lambda_empty * empty_loss
        if self.lambda_role:
            total = total + self.lambda_role * role_loss

        return {
            "loss": total,
            "lm_loss": lm_loss.detach(),
            "span_loss": span_loss.detach(),
            "empty_loss": empty_loss.detach(),
            "role_loss": role_loss.detach(),
            "n_anchor": n_anchor,
        }

    def save_pretrained(self, out_dir):
        import os
        os.makedirs(out_dir, exist_ok=True)
        self.model.save_pretrained(out_dir)  # LoRA adapter
        torch.save(self.heads.state_dict(), os.path.join(out_dir, "struct_heads.pt"))

    @torch.no_grad()
    def load_heads(self, out_dir):
        import os
        p = os.path.join(out_dir, "struct_heads.pt")
        self.heads.load_state_dict(torch.load(p, map_location="cpu"))
