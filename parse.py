#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import List, Optional, Dict, Tuple, Any, Iterable

log = logging.getLogger(__name__)


# =========================
# CLI
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Probabilistic Earley Parser")

    parser.add_argument("grammar", type=Path)
    parser.add_argument("sentences", type=Path)

    parser.add_argument("-s", "--start_symbol", default="ROOT")
    parser.add_argument("--chart", action="store_true")
    parser.add_argument("--all-parses", action="store_true")

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="store_const",
                           dest="logging_level", const=logging.DEBUG)
    verbosity.add_argument("-q", "--quiet", action="store_const",
                           dest="logging_level", const=logging.WARNING)

    parser.set_defaults(logging_level=logging.INFO)
    return parser.parse_args()


# =========================
# Data Structures
# =========================
@dataclass(frozen=True)
class Rule:
    lhs: str
    rhs: Tuple[str, ...]
    weight: float


@dataclass(frozen=True)
class Item:
    rule: Rule
    dot: int
    start: int

    def next_symbol(self):
        return None if self.dot == len(self.rule.rhs) else self.rule.rhs[self.dot]

    def advance(self):
        return Item(self.rule, self.dot + 1, self.start)


# =========================
# Agenda
# =========================
class Agenda:
    def __init__(self):
        self.items: List[Item] = []
        self.index: Dict[Item, int] = {}
        self.weights: Dict[Item, float] = {}
        self.backptr: Dict[Item, Any] = {}
        self.all_backptr: Dict[Item, List] = {}
        self.ptr = 0

    def push(self, item, weight, bp=None):
        if item in self.index:
            if weight < self.weights[item]:
                self.weights[item] = weight
                self.backptr[item] = bp
            if bp is not None:
                self.all_backptr[item].append((weight, bp))
        else:
            self.index[item] = len(self.items)
            self.items.append(item)
            self.weights[item] = weight
            self.backptr[item] = bp
            self.all_backptr[item] = [(weight, bp)] if bp else []

    def pop(self):
        item = self.items[self.ptr]
        self.ptr += 1
        return item

    def __len__(self):
        return len(self.items) - self.ptr

    def all(self):
        return self.items


# =========================
# Grammar
# =========================
class Grammar:
    def __init__(self, start_symbol: str, file: Path):
        self.start = start_symbol
        self.rules: Dict[str, List[Rule]] = {}
        self.load(file)

    def load(self, file: Path):
        for line in open(file):
            line = line.split("#")[0].strip()
            if not line:
                continue

            prob, lhs, rhs = line.split("\t")
            prob = float(prob)
            weight = -math.log2(prob) if prob > 0 else float("inf")

            rule = Rule(lhs, tuple(rhs.split()), weight)
            self.rules.setdefault(lhs, []).append(rule)

    def expansions(self, lhs):
        return self.rules.get(lhs, [])

    def is_nonterminal(self, sym):
        return sym in self.rules


# =========================
# Earley Parser
# =========================
class EarleyParser:
    def __init__(self, tokens, grammar):
        self.tokens = tokens
        self.grammar = grammar
        self.chart = [Agenda() for _ in range(len(tokens) + 1)]
        self.profile = Counter()
        self._run()

    def _run(self):
        self._predict(self.grammar.start, 0)

        for i in range(len(self.chart)):
            col = self.chart[i]
            while col:
                item = col.pop()
                nxt = item.next_symbol()

                if nxt is None:
                    self._attach(item, i)
                elif self.grammar.is_nonterminal(nxt):
                    self._predict(nxt, i)
                else:
                    self._scan(item, i)

    # -----------------
    # Core ops
    # -----------------
    def _predict(self, nt, pos):
        for rule in self.grammar.expansions(nt):
            self.chart[pos].push(Item(rule, 0, pos), rule.weight)

    def _scan(self, item, pos):
        if pos < len(self.tokens) and self.tokens[pos] == item.next_symbol():
            new = item.advance()
            w = self.chart[pos].weights[item]
            bp = (item, pos, self.tokens[pos])
            self.chart[pos + 1].push(new, w, bp)

    def _attach(self, item, pos):
        start = item.start
        w_item = self.chart[pos].weights[item]

        for cust in self.chart[start].all():
            if cust.next_symbol() == item.rule.lhs:
                new = cust.advance()
                w = self.chart[start].weights[cust] + w_item
                bp = (cust, start, (item, pos))
                self.chart[pos].push(new, w, bp)

    # -----------------
    # Output
    # -----------------
    def accepted(self):
        return any(
            item.rule.lhs == self.grammar.start and
            item.start == 0 and
            item.next_symbol() is None
            for item in self.chart[-1].all()
        )

    def best_parse(self):
        best = None
        best_w = float("inf")

        for item in self.chart[-1].all():
            if item.rule.lhs == self.grammar.start and item.start == 0:
                w = self.chart[-1].weights[item]
                if w < best_w:
                    best = item
                    best_w = w

        if not best:
            return None

        return best_w, self._tree(best, len(self.tokens)), self._tree_spans(best, len(self.tokens))

    def all_parses(self):
        results = []

        for item in self.chart[-1].all():
            if item.rule.lhs == self.grammar.start and item.start == 0:
                results.extend(self._enumerate(item, len(self.tokens)))

        return sorted(results, key=lambda x: x[0])

    # -----------------
    # Tree building
    # -----------------
    def _tree(self, item, col):
        children = []
        while item.dot > 0:
            bp = self.chart[col].backptr.get(item)
            if not bp:
                break
            prev, prev_col, child = bp

            if isinstance(child, str):
                children.append(child)
            else:
                children.append(self._tree(child[0], child[1]))

            item, col = prev, prev_col

        children.reverse()
        return f"({item.rule.lhs} {' '.join(children)})"

    def _tree_spans(self, item, col):
        children = []
        cur_item = item
        cur_col = col

        while cur_item.dot > 0:
            bp = self.chart[cur_col].backptr.get(cur_item)
            if not bp:
                break

            prev_item, prev_col, child = bp

            if isinstance(child, str):
                children.append(child)
            else:
                child_item, child_col = child
                children.append(self._tree_spans(child_item, child_col))

            cur_item = prev_item
            cur_col = prev_col

        children.reverse()
        return f"({item.rule.lhs} [{item.start},{col}] {' '.join(children)})"

    def _enumerate(self, item, col):
        # same logic as original but simplified
        return [(self.chart[col].weights[item],
                 self._tree(item, col),
                 self._tree_spans(item, col))]

    def print_chart(self):
        for i, col in enumerate(self.chart):
            print(f"\n--- Column {i} ---")
            for item in col.all():
                print(item, col.weights[item])


# =========================
# Main
# =========================
def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)

    grammar = Grammar(args.start_symbol, args.grammar)

    for line in open(args.sentences):
        sent = line.strip()
        if not sent:
            continue

        parser = EarleyParser(sent.split(), grammar)

        if args.chart:
            parser.print_chart()

        if args.all_parses:
            results = parser.all_parses()
            if not results:
                print("NONE")
            else:
                for w, t, ts in results:
                    print(t)
                    print(ts)
                    print(w)
        else:
            res = parser.best_parse()
            if res:
                w, t, ts = res
                print(t)
                print(ts)
                print(w)
            else:
                print("NONE")


if __name__ == "__main__":
    main()