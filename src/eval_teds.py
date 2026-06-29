#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
表格 HTML 评测：TEDS、S-TEDS（结构-only）、完全匹配率。
输入：infer_test.py 产出的 {image, pred, gold} jsonl。

依赖：pip install apted distance lxml
用法：python src/eval_teds.py --pred data/eval/pred_test.jsonl
"""
import argparse, json
from collections import deque
import distance
from apted import APTED, Config
from apted.helpers import Tree
from lxml import html


class TableTree(Tree):
    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)

    def bracket(self):
        if self.tag == "td":
            result = '"tag": %s, "colspan": %d, "rowspan": %d, "text": %s' % (
                self.tag, self.colspan or 0, self.rowspan or 0, self.content)
        else:
            result = '"tag": %s' % self.tag
        for child in self.children:
            result += child.bracket()
        return "{{{}}}".format(result)


class CustomConfig(Config):
    @staticmethod
    def maximum(*sequences):
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        m = self.maximum(*sequences)
        return 0.0 if m == 0 else float(distance.levenshtein(*sequences)) / m

    def rename(self, node1, node2):
        if (node1.tag != node2.tag) or (node1.colspan != node2.colspan) or (node1.rowspan != node2.rowspan):
            return 1.0
        if node1.tag == "td":
            if node1.content or node2.content:
                return self.normalized_distance(node1.content or [], node2.content or [])
        return 0.0


class TEDS(object):
    def __init__(self, structure_only=False):
        self.structure_only = structure_only

    def tokenize(self, node, toks):
        toks.append("<%s>" % node.tag)
        if node.text is not None:
            toks += list(node.text)
        for n in node.getchildren():
            self.tokenize(n, toks)
        if node.tag != "unk":
            toks.append("</%s>" % node.tag)

    def load_html_tree(self, node, parent=None):
        if node.tag == "td":
            if self.structure_only:
                cell = []
            else:
                toks = []
                self.tokenize(node, toks)
                cell = toks[1:-1]
            new_node = TableTree(node.tag,
                                 int(node.attrib.get("colspan", "1")),
                                 int(node.attrib.get("rowspan", "1")),
                                 cell, *deque())
        else:
            new_node = TableTree(node.tag, None, None, None, *deque())
        if parent is not None:
            parent.children.append(new_node)
        if node.tag != "td":
            for n in node.getchildren():
                self.load_html_tree(n, new_node)
        if parent is None:
            return new_node

    def evaluate(self, pred, true):
        if (not pred) or (not true):
            return 0.0
        parser = html.HTMLParser(remove_comments=True, encoding="utf-8")
        try:
            p = html.fromstring("<html><body>%s</body></html>" % pred, parser=parser)
            t = html.fromstring("<html><body>%s</body></html>" % true, parser=parser)
        except Exception:
            return 0.0
        p_tab = p.xpath("body//table")
        t_tab = t.xpath("body//table")
        if not p_tab or not t_tab:
            return 0.0
        p_tab, t_tab = p_tab[0], t_tab[0]
        n_nodes = max(len(p_tab.xpath(".//*")), len(t_tab.xpath(".//*")))
        if n_nodes == 0:
            return 0.0
        tp = self.load_html_tree(p_tab)
        tt = self.load_html_tree(t_tab)
        dist = APTED(tp, tt, CustomConfig()).compute_edit_distance()
        return 1.0 - float(dist) / n_nodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="data/eval/pred_test.jsonl")
    args = ap.parse_args()
    teds = TEDS(structure_only=False)
    steds = TEDS(structure_only=True)
    rows = [json.loads(x) for x in open(args.pred, encoding="utf-8")]
    n = len(rows)
    t_sum = s_sum = exact = 0
    for r in rows:
        pred, gold = r.get("pred", ""), r.get("gold", "")
        t_sum += teds.evaluate(pred, gold)
        s_sum += steds.evaluate(pred, gold)
        if pred.strip() == gold.strip():
            exact += 1
    print(f"样本数: {n}")
    print(f"TEDS    (内容+结构): {t_sum / n:.4f}")
    print(f"S-TEDS  (仅结构)   : {s_sum / n:.4f}")
    print(f"完全匹配率         : {exact / n:.4f}")


if __name__ == "__main__":
    main()
