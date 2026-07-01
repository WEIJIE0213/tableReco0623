#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M1 结构分支的监督目标抽取 + <td 锚点对齐。

核心保证：本模块产出的「有序单元格目标列表」与 build_dataset.build_html 生成的
HTML 里 <td 出现顺序严格一一对应（同一套阅读顺序 + 同样的越界裁剪）。
这样训练时就能把每个 <td 的解码器隐状态对上它的 (rowspan,colspan,is_empty,role) 目标。

用法（自测，指向任一含 labels/ 的数据目录）：
    python src/struct_targets.py --data-root data/synth/serif --show 5
"""
import argparse, glob, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_dataset import build_html  # 复用同一套 HTML 生成，避免顺序漂移

MAX_SPAN = 10  # rowspan/colspan 截断到 1..MAX_SPAN，做分类
ROLE2ID = {"data": 0, "col_header": 1, "row_header": 2, "corner": 3}


def ordered_cells(label):
    """按 build_html 完全一致的阅读顺序返回每个 <td 对应的结构信息。

    返回 list[dict]，长度 == build_html(label) 中 '<td' 的出现次数。
    对网格空洞（build_html 补 <td></td> 的位置）填充一个空 data 单元格。
    """
    R = label["grid"]["n_rows"]; C = label["grid"]["n_cols"]
    covered = [[False] * C for _ in range(R)]
    topleft = {(c["row"], c["col"]): c for c in label["cells"]}
    out = []
    for r in range(R):
        for c in range(C):
            if covered[r][c]:
                continue
            cell = topleft.get((r, c))
            if cell is None:  # 空洞，与 build_html 的 <td></td> 对齐
                out.append({"rowspan": 1, "colspan": 1, "is_empty": True,
                            "role": "data", "_hole": True})
                covered[r][c] = True
                continue
            rs = cell.get("rowspan", 1); cs = cell.get("colspan", 1)
            for i in range(r, min(r + rs, R)):
                for j in range(c, min(c + cs, C)):
                    covered[i][j] = True
            is_empty = cell.get("is_empty", str(cell.get("text", "")) == "")
            out.append({"rowspan": rs, "colspan": cs,
                        "is_empty": bool(is_empty),
                        "role": cell.get("role", "data") or "data"})
    return out


def cell_target_arrays(label):
    """把有序单元格转成训练用的整型标签数组。"""
    cells = ordered_cells(label)
    return {
        "rowspan_cls": [min(c["rowspan"], MAX_SPAN) - 1 for c in cells],
        "colspan_cls": [min(c["colspan"], MAX_SPAN) - 1 for c in cells],
        "empty": [1 if c["is_empty"] else 0 for c in cells],
        "role": [ROLE2ID.get(c["role"], 0) for c in cells],
        "n": len(cells),
    }


def targets_from_html(html_str):
    """直接从训练用的 HTML 解析每个 <td 的结构目标（与 <td 顺序一一对应）。

    build_html 已对文本做过 html.escape，单元格内不含裸 <，故非贪婪匹配安全。
    role 无法从 HTML 得到（M1 用不到，置 0；M2 再从 label 补）。
    返回 dict: rowspan_cls / colspan_cls / empty / role / n。
    """
    cells = re.findall(r"<td((?:\s+[a-zA-Z]+=\"[^\"]*\")*)\s*>(.*?)</td>", html_str, flags=re.S)
    rowspan_cls, colspan_cls, empty, role = [], [], [], []
    for attr, inner in cells:
        rs = re.search(r'rowspan="(\d+)"', attr)
        cs = re.search(r'colspan="(\d+)"', attr)
        rs = int(rs.group(1)) if rs else 1
        cs = int(cs.group(1)) if cs else 1
        rowspan_cls.append(min(rs, MAX_SPAN) - 1)
        colspan_cls.append(min(cs, MAX_SPAN) - 1)
        empty.append(1 if inner.strip() == "" else 0)
        role.append(0)
    return {"rowspan_cls": rowspan_cls, "colspan_cls": colspan_cls,
            "empty": empty, "role": role, "n": len(rowspan_cls)}


# ---- 锚点对齐（供 collator 使用） -------------------------------------------

def td_char_starts(text):
    """text 中每个 '<td' 起始字符位置（升序）。"""
    return [m.start() for m in re.finditer(r"<td", text)]


def locate_html_base(full_text, html_str):
    """assistant HTML 在整段 chat 文本里的起始字符偏移（找不到返回 -1）。"""
    return full_text.rfind(html_str)


def anchor_token_indices(offset_mapping, char_starts):
    """把每个 <td 的字符位置映射到覆盖它的 token 下标。

    offset_mapping: fast tokenizer 的 (start,end) 列表（对整段文本）。
    char_starts:    每个 <td 的绝对字符位置。
    返回等长 list[int]；映射不到的位置为 None（调用方应过滤）。
    """
    res = []
    for s in char_starts:
        idx = None
        for t, (st, en) in enumerate(offset_mapping):
            if en > st and st <= s < en:
                idx = t
                break
        res.append(idx)
    return res


# ---- 自测 -------------------------------------------------------------------

def _self_test(data_root, show):
    files = sorted(glob.glob(os.path.join(data_root, "labels", "*.json")))
    assert files, f"未找到标注：{data_root}/labels"
    n_ok = n_bad = 0
    shown = 0
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        html_str = build_html(d)
        n_td = html_str.count("<td")
        tgt = cell_target_arrays(d)
        ok = (n_td == tgt["n"])
        # 顺序一致性：<td 数 == 目标数；空标志与 HTML 里空 td 数也应吻合
        n_empty_html = len(re.findall(r"<td[^>]*></td>", html_str))
        n_empty_tgt = sum(tgt["empty"])
        if ok:
            n_ok += 1
        else:
            n_bad += 1
            if shown < show:
                print(f"  [MISMATCH] {os.path.basename(f)} n_td={n_td} n_tgt={tgt['n']}")
                shown += 1
        if shown < show and ok and n_empty_html != n_empty_tgt:
            print(f"  [空数不符] {os.path.basename(f)} html空={n_empty_html} tgt空={n_empty_tgt}"
                  f"（可能是 is_empty 标注与文本不一致，非致命）")
    print(f"自测 {len(files)} 份：锚点对齐 OK {n_ok} / MISMATCH {n_bad}")
    return 0 if n_bad == 0 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/synth/serif")
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()
    return _self_test(args.data_root, args.show)


if __name__ == "__main__":
    raise SystemExit(main())
