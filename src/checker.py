#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结构合法性校验器。
- 占用矩阵自检：无重叠、无越界、无漏覆盖。
- 表头层级边引用的 cell id 是否存在。
- 可作为独立质检，也供内容三的奖励复用（import check_label）。

用法：
    python src/checker.py --data-root data/synth/serif
"""
import argparse, glob, json, os

def check_label(d):
    """返回 (ok, issues:list[str])。"""
    issues = []
    R = d["grid"]["n_rows"]; C = d["grid"]["n_cols"]
    occ = [[0] * C for _ in range(R)]
    ids = set()
    for c in d["cells"]:
        ids.add(c.get("id"))
        r, co = c["row"], c["col"]; rs = c.get("rowspan", 1); cs = c.get("colspan", 1)
        if r < 0 or co < 0 or r + rs > R or co + cs > C:
            issues.append(f"cell {c.get('id')} 越界 r={r} c={co} rs={rs} cs={cs}")
            continue
        for i in range(r, r + rs):
            for j in range(co, co + cs):
                occ[i][j] += 1
    overlap = sum(1 for i in range(R) for j in range(C) if occ[i][j] > 1)
    uncov = sum(1 for i in range(R) for j in range(C) if occ[i][j] == 0)
    if overlap: issues.append(f"重叠单元格 {overlap} 处")
    if uncov: issues.append(f"未覆盖网格 {uncov} 处")
    # 表头层级边引用检查
    ht = d.get("header_tree", {})
    for e in ht.get("edges", []):
        if e.get("parent") not in ids or e.get("child") not in ids:
            issues.append(f"表头边引用了不存在的 cell: {e}")
    return (len(issues) == 0, issues)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/synth/serif")
    ap.add_argument("--show", type=int, default=10, help="最多展示多少条问题样本")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.data_root, "labels", "*.json")))
    bad = []
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        ok, issues = check_label(d)
        if not ok:
            bad.append((os.path.basename(f), issues))
    print(f"校验 {len(files)} 份，问题样本 {len(bad)} 份")
    for name, iss in bad[:args.show]:
        print("  ", name, iss)
    return 0 if not bad else 1

if __name__ == "__main__":
    raise SystemExit(main())
