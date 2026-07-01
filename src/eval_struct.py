#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
细粒度结构评测：行/列数准确率、合并单元格 span F1、空单元格 F1、单元格内容准确率，
并按难度（边框 / 表头层数 / 宽表）分组。

输入：infer_test.py 的 {image, pred, gold} jsonl + 原始 labels 目录（取难度 meta 与金标结构）。
用法：
  python src/eval_struct.py --pred data/eval/pred_test.jsonl --labels-dir data/synth/serif/labels
"""
import argparse, json, os
from collections import defaultdict
from lxml import html as lhtml


def parse_table_html(html_str):
    """把 HTML 表格解析成 cells（含 row/col/rowspan/colspan/text/is_empty）与网格尺寸。"""
    if not html_str:
        return None
    try:
        root = lhtml.fromstring("<html><body>%s</body></html>" % html_str)
    except Exception:
        return None
    tabs = root.xpath("body//table")
    if not tabs:
        return None
    rows = tabs[0].xpath(".//tr")
    cells, occ = [], set()
    r = 0
    for tr in rows:
        c = 0
        for td in tr.xpath("./td|./th"):
            while (r, c) in occ:
                c += 1
            rs = int(td.get("rowspan", "1") or 1)
            cs = int(td.get("colspan", "1") or 1)
            text = (td.text_content() or "").strip()
            cells.append({"row": r, "col": c, "rowspan": rs, "colspan": cs,
                          "text": text, "is_empty": text == ""})
            for i in range(r, r + rs):
                for j in range(c, c + cs):
                    occ.add((i, j))
            c += cs
        r += 1
    n_cols = max((cl["col"] + cl["colspan"] for cl in cells), default=0)
    return {"n_rows": r, "n_cols": n_cols, "cells": cells}


def spans(cells):      # 合并单元格集合
    return {(c["row"], c["col"], c["rowspan"], c["colspan"]) for c in cells
            if c["rowspan"] > 1 or c["colspan"] > 1}

def empties(cells):    # 空单元格位置集合
    return {(c["row"], c["col"]) for c in cells if c["is_empty"]}

def text_map(cells):   # (row,col)->text
    return {(c["row"], c["col"]): c["text"] for c in cells}

def prf(pred_set, gold_set):
    tp = len(pred_set & gold_set)
    p = tp / len(pred_set) if pred_set else (1.0 if not gold_set else 0.0)
    r = tp / len(gold_set) if gold_set else (1.0 if not pred_set else 0.0)
    f = 2 * p * r / (p + r) if (p + r) else (1.0 if not gold_set and not pred_set else 0.0)
    return f


def bucket(meta, grid):
    b = meta.get("border", "na")
    border = "无线/弱线" if b in ("none", "weak", "minimal", "partial", "borderless") else "有线"
    depth = "表头≥3层" if meta.get("header_depth", 1) >= 3 else "表头≤2层"
    ncol = grid.get("n_cols", 0)
    width = "宽表(≥12列)" if ncol >= 12 else "非宽表"
    return [border, depth, width]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="data/eval/pred_test.jsonl")
    ap.add_argument("--labels-dir", default="data/synth/serif/labels")
    args = ap.parse_args()

    agg = defaultdict(lambda: defaultdict(float))
    cnt = defaultdict(int)

    def add(group, **kv):
        cnt[group] += 1
        for k, v in kv.items():
            agg[group][k] += v

    rows = [json.loads(x) for x in open(args.pred, encoding="utf-8")]
    miss = 0
    for r in rows:
        img = r["image"].replace("\\", "/")
        stem = os.path.splitext(os.path.basename(img))[0]
        # 优先从图片路径推导 label（/images/ -> /labels/），自动跨多个数据目录
        cand = []
        if "/images/" in img:
            cand.append(os.path.splitext(img.replace("/images/", "/labels/"))[0] + ".json")
        cand.append(os.path.join(args.labels_dir, stem + ".json"))
        lp = next((p for p in cand if os.path.exists(p)), None)
        if lp is None:
            miss += 1
            continue
        gold = json.load(open(lp, encoding="utf-8"))
        gcells = gold["cells"]
        gmeta = gold.get("meta", {})
        ggrid = {"n_rows": gold["grid"]["n_rows"], "n_cols": gold["grid"]["n_cols"]}
        pred = parse_table_html(r.get("pred", ""))
        if pred is None:
            metrics = dict(row_ok=0, col_ok=0, span_f1=0.0, empty_f1=0.0, cell_acc=0.0)
        else:
            row_ok = 1.0 if pred["n_rows"] == ggrid["n_rows"] else 0.0
            col_ok = 1.0 if pred["n_cols"] == ggrid["n_cols"] else 0.0
            span_f1 = prf(spans(pred["cells"]), spans(gcells))
            empty_f1 = prf(empties(pred["cells"]), empties(gcells))
            ptm, gtm = text_map(pred["cells"]), text_map(gcells)
            same = sum(1 for k, v in gtm.items() if ptm.get(k) == v)
            cell_acc = same / len(gtm) if gtm else 0.0
            metrics = dict(row_ok=row_ok, col_ok=col_ok, span_f1=span_f1,
                           empty_f1=empty_f1, cell_acc=cell_acc)
        add("总体", **metrics)
        for g in bucket(gmeta, ggrid):
            add(g, **metrics)

    def show(group):
        n = cnt[group]
        if not n:
            return
        a = agg[group]
        print(f"  [{group:<12}] n={n:<4} 行数Acc={a['row_ok']/n:.3f} 列数Acc={a['col_ok']/n:.3f} "
              f"合并spanF1={a['span_f1']/n:.3f} 空格F1={a['empty_f1']/n:.3f} 单元格内容Acc={a['cell_acc']/n:.3f}")

    print("=== 细粒度结构指标 ===")
    if miss:
        print(f"(注意：{miss} 条样本未找到对应 label，已跳过)")
    show("总体")
    print("--- 按难度分组 ---")
    for g in ["有线", "无线/弱线", "表头≤2层", "表头≥3层", "非宽表", "宽表(≥12列)"]:
        show(g)


if __name__ == "__main__":
    main()
