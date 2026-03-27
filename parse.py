#!/usr/bin/env python3
"""
Probabilistic Earley parser that reconstructs the highest-probability parse
of each given sentence under a PCFG.

Usage: ./parse.py foo.gr foo.sen

For each sentence, prints:
  - The best parse tree (in parenthesized notation), or NONE if no parse exists.
  - The same tree annotated with [start,end] spans.
  - The weight of the best parse (-log2 probability).

Flags:
  --all-parses   Print all valid parses (not just the best one).
  --chart        Print the Earley chart after parsing.
"""

from __future__ import annotations
import argparse
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import Counter as CounterType, Iterable, List, Optional, Dict, Tuple, Any


log = logging.getLogger(Path(__file__).stem)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "grammar", type=Path, help="Path to .gr file containing a PCFG"
    )
    parser.add_argument(
        "sentences", type=Path, help="Path to .sen file containing tokenized input sentences"
    )
    parser.add_argument(
        "-s", "--start_symbol", type=str, default="ROOT",
        help="Start symbol of the grammar (default: ROOT)"
    )
    parser.add_argument(
        "--progress", action="store_true", default=False,
        help="Display a progress bar (requires tqdm)"
    )
    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", dest="logging_level",
        action="store_const", const=logging.DEBUG
    )
    verbosity.add_argument(
        "-q", "--quiet", dest="logging_level",
        action="store_const", const=logging.WARNING
    )
    parser.add_argument(
        "--chart", action="store_true", default=False,
        help="Print the Earley chart after parsing"
    )
    parser.add_argument(
        "--all-parses", action="store_true", default=False,
        help="Print all valid parses, not just the best one"
    )
    return parser.parse_args()


# --- Data Structures ----------------------------------------------------------

@dataclass(frozen=True)
class Rule:
    """A grammar rule with a left-hand side, right-hand side, and weight (-log2 prob)."""
    lhs: str
    rhs: Tuple[str, ...]
    weight: float = 0.0

    def __repr__(self) -> str:
        return f"{self.lhs} -> {' '.join(self.rhs)}"


@dataclass(frozen=True)
class Item:
    """An Earley item: a dotted rule together with a start position.

    Frozen so it is hashable and can be used as a dictionary key.
    The item's identity (for duplicate detection) is determined by
    (rule, dot_position, start_position).
    """
    rule: Rule
    dot_position: int
    start_position: int

    def next_symbol(self) -> Optional[str]:
        """Return the symbol after the dot, or None if the dot is at the end."""
        if self.dot_position == len(self.rule.rhs):
            return None
        return self.rule.rhs[self.dot_position]

    def with_dot_advanced(self) -> Item:
        """Return a new Item with the dot moved one position to the right."""
        if self.next_symbol() is None:
            raise IndexError("Can't advance dot past end of rule")
        return Item(
            rule=self.rule,
            dot_position=self.dot_position + 1,
            start_position=self.start_position
        )

    def __repr__(self) -> str:
        DOT = "."
        rhs = list(self.rule.rhs)
        rhs.insert(self.dot_position, DOT)
        return f"({self.start_position}, {self.rule.lhs} -> {' '.join(rhs)})"


# --- Agenda -------------------------------------------------------------------

class Agenda:
    """An agenda (column) of Earley items with weights and backpointers
    for Viterbi (best-parse) tracking.

    Each item is pushed at most once into the ordered list _items.
    If an item is pushed again with a *lower* weight, its weight and
    backpointer are updated in-place (but it is not re-added to the queue).

    >>> a = Agenda()
    >>> r = Rule('X', ('a',), 1.0)
    >>> i1 = Item(r, 0, 0)
    >>> a.push(i1, 5.0)
    >>> len(a)
    1
    >>> a.push(i1, 3.0)   # better weight -> update
    >>> len(a)
    1
    >>> a.get_weight(i1)
    3.0
    >>> a.push(i1, 9.0)   # worse weight -> ignore
    >>> a.get_weight(i1)
    3.0
    """

    def __init__(self) -> None:
        self._items: List[Item] = []          # all items in push order
        self._index: Dict[Item, int] = {}     # item -> index in _items (O(1) lookup)
        self._next: int = 0                   # next item to pop
        self._weights: Dict[Item, float] = {}
        self._backpointers: Dict[Item, Any] = {}
        # For --all-parses: store ALL backpointers, not just the best
        self._all_backpointers: Dict[Item, List[Tuple[float, Any]]] = {}

    def __len__(self) -> int:
        """Number of items waiting to be popped."""
        return len(self._items) - self._next

    def push(self, item: Item, weight: float, backpointer: Any = None) -> None:
        """Add item, or update weight/backpointer if a strictly better weight is found.
        O(1) amortized thanks to dict-based duplicate detection."""
        if item in self._index:
            # Item already seen -- keep the better (lower) weight for Viterbi.
            if weight < self._weights[item]:
                self._weights[item] = weight
                self._backpointers[item] = backpointer
            # Always record the alternative backpointer for --all-parses
            if backpointer is not None:
                self._all_backpointers[item].append((weight, backpointer))
        else:
            self._items.append(item)
            self._index[item] = len(self._items) - 1
            self._weights[item] = weight
            self._backpointers[item] = backpointer
            if backpointer is not None:
                self._all_backpointers[item] = [(weight, backpointer)]
            else:
                self._all_backpointers[item] = []

    def pop(self) -> Item:
        """Dequeue the next unprocessed item (FIFO order)."""
        if len(self) == 0:
            raise IndexError("Agenda is empty")
        item = self._items[self._next]
        self._next += 1
        return item

    def all(self) -> Iterable[Item]:
        """All items ever pushed (including already-popped ones).
        Needed for attach: completed items must find their customers."""
        return self._items

    def get_weight(self, item: Item) -> float:
        return self._weights[item]

    def get_backpointer(self, item: Item) -> Any:
        return self._backpointers.get(item)

    def get_all_backpointers(self, item: Item) -> List[Tuple[float, Any]]:
        return self._all_backpointers.get(item, [])

    def __repr__(self) -> str:
        n = self._next
        return f"Agenda({self._items[:n]}; {self._items[n:]})"


# --- Grammar ------------------------------------------------------------------

class Grammar:
    """A weighted context-free grammar loaded from .gr files."""

    def __init__(self, start_symbol: str, *files: Path) -> None:
        self.start_symbol = start_symbol
        self._expansions: Dict[str, List[Rule]] = {}
        for file in files:
            self.add_rules_from_file(file)

    def add_rules_from_file(self, file: Path) -> None:
        """Load rules from a tab-delimited .gr file.
        Format per line: <probability>\\t<lhs>\\t<rhs>
        Weight = -log2(probability)."""
        with open(file, "r") as f:
            for line in f:
                line = line.split("#")[0].rstrip()  # strip comments & trailing whitespace
                if line == "":
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                prob_str, lhs, rhs_str = parts[0], parts[1], parts[2]
                prob = float(prob_str)
                rhs = tuple(rhs_str.split())
                if prob <= 0:
                    weight = float('inf')
                else:
                    weight = -math.log2(prob)
                rule = Rule(lhs=lhs, rhs=rhs, weight=weight)
                if lhs not in self._expansions:
                    self._expansions[lhs] = []
                self._expansions[lhs].append(rule)

    def expansions(self, lhs: str) -> Iterable[Rule]:
        """Return all rules that expand `lhs`, or empty list if none."""
        return self._expansions.get(lhs, [])

    def is_nonterminal(self, symbol: str) -> bool:
        """A symbol is a nonterminal iff it has at least one expansion rule."""
        return symbol in self._expansions


# --- Earley Chart Parser ------------------------------------------------------

class EarleyChart:
    """Probabilistic Earley chart parser with Viterbi (best-parse) tracking."""

    def __init__(self, tokens: List[str], grammar: Grammar, progress: bool = False) -> None:
        self.tokens = tokens
        self.grammar = grammar
        self.progress = progress
        self.profile: CounterType[str] = Counter()
        self.cols: List[Agenda]
        self._run_earley()  # fill the chart

    def accepted(self) -> bool:
        """Was the sentence accepted (i.e. does a complete ROOT item exist)?"""
        for item in self.cols[-1].all():
            if (item.rule.lhs == self.grammar.start_symbol
                    and item.next_symbol() is None
                    and item.start_position == 0):
                return True
        return False

    def get_best_parse(self) -> Optional[Tuple[float, str, str]]:
        """Return (weight, tree_string, tree_with_spans) for the best parse, or None."""
        best_item = None
        best_weight = float('inf')
        for item in self.cols[-1].all():
            if (item.rule.lhs == self.grammar.start_symbol
                    and item.next_symbol() is None
                    and item.start_position == 0):
                w = self.cols[-1].get_weight(item)
                if w < best_weight:
                    best_weight = w
                    best_item = item
        if best_item is None:
            return None
        tree = self._build_tree(best_item, len(self.tokens))
        tree_spans = self._build_tree_with_spans(best_item, len(self.tokens))
        return (best_weight, tree, tree_spans)

    def get_all_parses(self) -> List[Tuple[float, str, str]]:
        """Return all valid parses as list of (weight, tree, tree_with_spans).
        Uses the stored alternative backpointers to enumerate all derivations."""
        results = []
        for item in self.cols[-1].all():
            if (item.rule.lhs == self.grammar.start_symbol
                    and item.next_symbol() is None
                    and item.start_position == 0):
                # Enumerate all trees for this completed start-symbol item
                all_trees = self._enumerate_all_trees_for_item(item, len(self.tokens))
                for (weight, tree, tree_spans) in all_trees:
                    results.append((weight, tree, tree_spans))
        # Sort by weight (best first)
        results.sort(key=lambda x: x[0])
        return results

    # -- Tree reconstruction via backpointers ----------------------------------

    def _build_tree(self, item: Item, col_idx: int) -> str:
        """Reconstruct the best parse tree from Viterbi backpointers."""
        children: List[str] = []
        cur_item = item
        cur_col = col_idx

        while cur_item.dot_position > 0:
            bp = self.cols[cur_col].get_backpointer(cur_item)
            if bp is None:
                break
            prev_item, prev_col, child_info = bp

            if isinstance(child_info, str):
                children.append(child_info)
            else:
                attached_item, attached_col = child_info
                children.append(self._build_tree(attached_item, attached_col))

            cur_item = prev_item
            cur_col = prev_col

        children.reverse()
        return "(" + item.rule.lhs + " " + " ".join(children) + ")"

    def _build_tree_with_spans(self, item: Item, col_idx: int) -> str:
        """Reconstruct the best parse tree with [start,end] span annotations."""
        children: List[str] = []
        cur_item = item
        cur_col = col_idx

        while cur_item.dot_position > 0:
            bp = self.cols[cur_col].get_backpointer(cur_item)
            if bp is None:
                break
            prev_item, prev_col, child_info = bp

            if isinstance(child_info, str):
                children.append(child_info)
            else:
                attached_item, attached_col = child_info
                children.append(self._build_tree_with_spans(attached_item, attached_col))

            cur_item = prev_item
            cur_col = prev_col

        children.reverse()
        start = item.start_position
        end = col_idx
        return "(" + item.rule.lhs + " [" + str(start) + "," + str(end) + "] " + " ".join(children) + ")"

    def _enumerate_all_trees(self, item: Item, col_idx: int) -> List[Tuple[float, List[str], List[str]]]:
        """Enumerate all derivations for item ending at col_idx.
        Returns list of (weight, children_list, children_with_spans_list).
        Each entry represents one full derivation of all children of this item's rule."""
        if item.dot_position == 0:
            # Base case: no children consumed yet
            return [(0.0, [], [])]

        # Get ALL backpointers for this (item, col_idx) pair
        all_bp = self.cols[col_idx].get_all_backpointers(item)
        if not all_bp:
            bp = self.cols[col_idx].get_backpointer(item)
            if bp is None:
                return []
            all_bp = [(self.cols[col_idx].get_weight(item), bp)]

        results = []
        for bp_weight, bp in all_bp:
            prev_item, prev_col, child_info = bp

            if isinstance(child_info, str):
                # SCAN: the last child is a terminal
                child_str = child_info
                child_span_str = child_info
            else:
                # ATTACH: the last child is a completed nonterminal subtree
                attached_item, attached_col = child_info
                # Get all subtrees for the attached constituent
                sub_derivations = self._enumerate_all_trees_for_item(attached_item, attached_col)
                if not sub_derivations:
                    continue
                for (sub_w, sub_tree, sub_tree_spans) in sub_derivations:
                    # Recurse on prefix
                    prefix_derivations = self._enumerate_all_trees(prev_item, prev_col)
                    for (pw, prefix_children, prefix_span_children) in prefix_derivations:
                        results.append((
                            bp_weight,
                            prefix_children + [sub_tree],
                            prefix_span_children + [sub_tree_spans]
                        ))
                continue  # already added results for ATTACH case

            # For SCAN case: recurse on prefix
            prefix_derivations = self._enumerate_all_trees(prev_item, prev_col)
            for (pw, prefix_children, prefix_span_children) in prefix_derivations:
                results.append((
                    bp_weight,
                    prefix_children + [child_str],
                    prefix_span_children + [child_span_str]
                ))

        return results

    def _enumerate_all_trees_for_item(self, item: Item, col_idx: int) -> List[Tuple[float, str, str]]:
        """Enumerate all complete subtrees rooted at item ending at col_idx.
        Returns list of (weight, tree_string, tree_with_spans_string)."""
        derivations = self._enumerate_all_trees(item, col_idx)
        results = []
        start = item.start_position
        end = col_idx
        lhs = item.rule.lhs
        for (w, children, span_children) in derivations:
            tree = "(" + lhs + " " + " ".join(children) + ")"
            tree_spans = "(" + lhs + " [" + str(start) + "," + str(end) + "] " + " ".join(span_children) + ")"
            results.append((w, tree, tree_spans))
        # Deduplicate
        seen = set()
        unique = []
        for r in results:
            if r[1] not in seen:
                seen.add(r[1])
                unique.append(r)
        return unique

    # -- Core Earley algorithm -------------------------------------------------

    def _run_earley(self) -> None:
        """Fill in the Earley chart using Predict/Scan/Attach."""
        n = len(self.tokens)
        self.cols = [Agenda() for _ in range(n + 1)]

        # Seed: predict all expansions of the start symbol at position 0
        self._predict(self.grammar.start_symbol, 0)

        # Process columns left to right
        for i in range(n + 1):
            log.debug("")
            log.debug(f"=== Processing column {i} ===")
            column = self.cols[i]
            while column:   # while there are unprocessed items
                item = column.pop()
                next_sym = item.next_symbol()
                if next_sym is None:
                    # Completed item -> attach to customers
                    log.debug(f"  {item} => ATTACH")
                    self._attach(item, i)
                elif self.grammar.is_nonterminal(next_sym):
                    # Nonterminal after dot -> predict
                    log.debug(f"  {item} => PREDICT")
                    self._predict(next_sym, i)
                else:
                    # Terminal after dot -> scan
                    log.debug(f"  {item} => SCAN")
                    self._scan(item, i)

    def _predict(self, nonterminal: str, position: int) -> None:
        """Predict: for each rule A -> alpha expanding `nonterminal`,
        add the item (position, A -> .alpha) with weight = rule.weight."""
        for rule in self.grammar.expansions(nonterminal):
            new_item = Item(rule, dot_position=0, start_position=position)
            self.cols[position].push(new_item, weight=rule.weight, backpointer=None)
            log.debug(f"    Predicted: {new_item} (w={rule.weight:.4f})")
            self.profile["PREDICT"] += 1

    def _scan(self, item: Item, position: int) -> None:
        """Scan: if the terminal after the dot matches the current input word,
        advance the dot and place the new item in the next column.
        Weight is inherited from the parent item (scanning is free)."""
        if position < len(self.tokens) and self.tokens[position] == item.next_symbol():
            new_item = item.with_dot_advanced()
            weight = self.cols[position].get_weight(item)
            bp = (item, position, self.tokens[position])  # backpointer: scanned terminal
            self.cols[position + 1].push(new_item, weight=weight, backpointer=bp)
            log.debug(f"    Scanned: {new_item} in col {position+1} (w={weight:.4f})")
            self.profile["SCAN"] += 1

    def _attach(self, item: Item, position: int) -> None:
        """Attach (complete): for each customer in column[item.start_position]
        that is waiting for item.rule.lhs, advance the customer's dot.
        New weight = customer_weight + completed_weight (summing -log2 probs)."""
        mid = item.start_position
        completed_weight = self.cols[position].get_weight(item)

        for customer in self.cols[mid].all():
            if customer.next_symbol() == item.rule.lhs:
                new_item = customer.with_dot_advanced()
                customer_weight = self.cols[mid].get_weight(customer)
                new_weight = customer_weight + completed_weight
                bp = (customer, mid, (item, position))  # backpointer: attached constituent
                self.cols[position].push(new_item, weight=new_weight, backpointer=bp)
                log.debug(f"    Attached: {new_item} in col {position} (w={new_weight:.4f})")
                self.profile["ATTACH"] += 1

    # -- Chart printing --------------------------------------------------------

    def print_chart(self) -> None:
        """Print the Earley chart (all columns and their items)."""
        for i, col in enumerate(self.cols):
            if i == 0:
                print(f"\n--- Column {i} (before input) ---")
            elif i <= len(self.tokens):
                print(f"\n--- Column {i} (after '{self.tokens[i-1]}') ---")
            else:
                print(f"\n--- Column {i} ---")
            for item in col.all():
                w = col.get_weight(item)
                print(f"  {item}  [w={w:.4f}]")


# --- Main ---------------------------------------------------------------------

def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)
    grammar = Grammar(args.start_symbol, args.grammar)

    with open(args.sentences) as f:
        for sentence in f.readlines():
            sentence = sentence.strip()
            if sentence != "":   # skip blank lines
                log.debug("=" * 70)
                log.debug(f"Parsing: {sentence}")
                chart = EarleyChart(sentence.split(), grammar, progress=args.progress)
                if args.chart:
                    chart.print_chart()

                if args.all_parses:
                    # Print all valid parses
                    all_results = chart.get_all_parses()
                    if not all_results:
                        print("NONE")
                    else:
                        for weight, tree, tree_spans in all_results:
                            print(tree)
                            print(tree_spans)
                            print(f"{weight}")
                else:
                    # Print only the best parse
                    result = chart.get_best_parse()
                    if result is not None:
                        weight, tree, tree_spans = result
                        print(tree)
                        print(tree_spans)
                        print(f"{weight}")
                    else:
                        print("NONE")
                log.debug(f"Profile: {chart.profile}")


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=False)
    main()
