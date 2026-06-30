#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内容一 数据构建：把逐图 JSON 标注转换成 ms-swift 训练用的 jsonl。

- 训练目标：表格 HTML（含 rowspan/colspan、空单元格），不含坐标。
- 原始 bbox 不改、不写入训练样本（保留在 labels/ 里给内容二用）。
- 同时做分层 train/val/test 划分。
- 图片路径写成相对项目根目录的相对路径，训练时从项目根启动即可解析。

用法（在项目根目录运行）：
    python src/build_dataset.py \
        --data-root data/synth/serif \
        --out data/jsonl \
        --rel-prefix data/synth/serif/images

产物：
    data/jsonl/train.jsonl  val.jsonl  test.jsonl
    data/splits/{train,val,test}.txt
"""
import argparse, glob, html, json, os, random
from collections import defaultdict

HEADER_ROLES = {"col_header", "row_header", "corner"}

def build_html(label):
    """从 cells 重建 HTML 表格字符串。"""
    R = label["grid"]["n_rows"]; C = label["grid"]["n_cols"]
    covered = [[False] * C for _ in range(R)]
    topleft = {(c["row"], c["col"]): c for c in label["cells"]}
    rows_html = []
    for r in range(R):
        cells_html = []
        for c in range(C):
            if covered[r][c]:
                continue
            cell = topleft.get((r, c))
            if cell is None:
                # 网格有洞（理论上不该发生）——补一个空 td 保证结构合法
                cells_html.append("<td></td>")
                covered[r][c] = True
                continue
            rs = cell.get("rowspan", 1); cs = cell.get("colspan", 1)
            for i in range(r, min(r + rs, R)):
                for j in range(c, min(c + cs, C)):
                    covered[i][j] = True
            attr = ""
            if rs > 1: attr += f' rowspan="{rs}"'
            if cs > 1: attr += f' colspan="{cs}"'
            text = "" if cell.get("is_empty") else html.escape(str(cell.get("text", "")))
            cells_html.append(f"<td{attr}>{text}</td>")
        rows_html.append("<tr>" + "".join(cells_html) + "</tr>")
    return "<table>" + "".join(rows_html) + "</table>"

PROMPT = "<image>\n请识别这张表格的完整结构和单元格内容，按 HTML 表格输出，合并单元格用 rowspan/colspan 表示，空单元格保留为空。"

def to_sample(label, img_relpath):
    return {
        "messages": [
            {"role": "user", "content": PROMPT},
            {"role": "assistant", "content": build_html(label)},
        ],
        "images": [img_relpath],
    }

def stratum_key(label):
    m = label.get("meta", {})
    border = m.get("border", "na")
    depth = m.get("header_depth", label["grid"]["n_rows"] and 1)
    ncol = label["grid"]["n_cols"]
    wbucket = "wide" if ncol >= 12 else ("mid" if ncol >= 7 else "narrow")
    return f"{border}|h{depth}|{wbucket}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", nargs="+", default=["data/synth/serif"],
                    help="一个或多个数据目录(各含 images/ 和 labels/)，会合并后统一划分")
    ap.add_argument("--out", default="data/jsonl")
    ap.add_argument("--ratios", default="0.8,0.1,0.1")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # 分层（多目录合并，每条样本记住自己所属目录的图片前缀）
    strata = defaultdict(list)
    for root in args.data_root:
        rel_prefix = os.path.join(root, "images")
        label_files = sorted(glob.glob(os.path.join(root, "labels", "*.json")))
        assert label_files, f"未找到标注：{root}/labels"
        for f in label_files:
            d = json.load(open(f, encoding="utf-8"))
            strata[stratum_key(d)].append((f, d, rel_prefix))
        print(f"读取 {root}: {len(label_files)} 条")

    r_train, r_val, r_test = [float(x) for x in args.ratios.split(",")]
    rng = random.Random(args.seed)
    split = {"train": [], "val": [], "test": []}
    for key, items in strata.items():
        rng.shuffle(items)
        n = len(items); n_tr = int(n * r_train); n_va = int(n * r_val)
        split["train"] += items[:n_tr]
        split["val"]   += items[n_tr:n_tr + n_va]
        split["test"]  += items[n_tr + n_va:]

    os.makedirs(args.out, exist_ok=True)
    os.makedirs("data/splits", exist_ok=True)
    counts = {}
    for name, items in split.items():
        jpath = os.path.join(args.out, f"{name}.jsonl")
        ids = []
        with open(jpath, "w", encoding="utf-8") as w:
            for f, d, rel_prefix in items:
                stem = os.path.splitext(os.path.basename(f))[0]
                img = f"{rel_prefix}/{d['image']}" if "image" in d else f"{rel_prefix}/{stem}.png"
                w.write(json.dumps(to_sample(d, img), ensure_ascii=False) + "\n")
                ids.append(stem)
        with open(os.path.join("data/splits", f"{name}.txt"), "w", encoding="utf-8") as w:
            w.write("\n".join(ids) + "\n")
        counts[name] = len(items)
    print("划分完成:", counts, "| 总计", sum(counts.values()))
    print("jsonl 输出:", args.out)

if __name__ == "__main__":
    main()
