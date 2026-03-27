#!/usr/bin/env python3
"""
parse.py
A weighted probabilistic Earley chart parser for context-free grammars.

Grammar files (.gr) use the format:
    <probability> <LHS> <RHS token> [<RHS token> ...]

Sentence files (.sen) contain one sentence per line (space-separated tokens).

Usage:
    python parse.py grammar.gr sentences.sen [--chart] [--all-parses] [--spans] [--progress]
"""

import math
import argparse
from collections import defaultdict

# ---------------------------------------------------------------------------
# Optional progress-bar support
# ---------------------------------------------------------------------------
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        """Transparent fallback when tqdm is not installed."""
        return iterable


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class ProductionRule:
    """
    Represents a single weighted production rule of the form:
        LHS -> RHS[0] RHS[1] ...

    The rule weight is stored as the negative log-base-2 of the probability,
    so that summing weights corresponds to multiplying probabilities, and
    lower total weight means higher probability.

    Attributes:
        lhs     (str):   Left-hand side non-terminal symbol.
        rhs     (tuple): Ordered sequence of right-hand side symbols.
        weight  (float): Negative log2 of the rule probability.
    """

    def __init__(self, lhs: str, rhs: list, probability: float) -> None:
        """
        Initialise a production rule.

        Args:
            lhs:         Left-hand side non-terminal.
            rhs:         List of right-hand side symbols.
            probability: Rule probability in (0, 1].
        """
        self.lhs = lhs
        self.rhs = tuple(rhs)
        self.weight = -math.log2(probability)

    def __repr__(self) -> str:
        return f"{self.lhs} -> {' '.join(self.rhs)}"


class EarleyItem:
    """
    An Earley chart item (also called an *edge* or *state*).

    An item tracks how far through a production rule we have matched
    and at which position in the input the match started.

    Attributes:
        rule  (ProductionRule): The grammar rule associated with this item.
        dot   (int):            Index of the next symbol to match (0 = nothing matched yet).
        start (int):            Input position where this item's match began.
    """

    def __init__(self, rule: ProductionRule, dot: int, start: int) -> None:
        """
        Initialise an Earley item.

        Args:
            rule:  The production rule this item is tracking.
            dot:   Current dot position within the rule's RHS.
            start: Chart column where this item originated.
        """
        self.rule = rule
        self.dot = dot
        self.start = start

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def symbol_at_dot(self):
        """
        Return the next symbol to be matched, or None if the item is complete.

        Returns:
            str | None: The symbol immediately after the dot, or None.
        """
        if self.dot < len(self.rule.rhs):
            return self.rule.rhs[self.dot]
        return None

    def is_complete(self) -> bool:
        """Return True when the dot has passed all symbols in the RHS."""
        return self.dot == len(self.rule.rhs)

    def advance(self) -> "EarleyItem":
        """
        Produce a new item identical to this one but with the dot moved one
        position to the right.

        Returns:
            EarleyItem: A new item with dot incremented by 1.
        """
        return EarleyItem(self.rule, self.dot + 1, self.start)

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def __hash__(self) -> int:
        return hash((self.rule.lhs, self.rule.rhs, self.dot, self.start))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EarleyItem):
            return NotImplemented
        return (
            self.rule.lhs == other.rule.lhs
            and self.rule.rhs == other.rule.rhs
            and self.dot == other.dot
            and self.start == other.start
        )

    def __repr__(self) -> str:
        annotated = list(self.rule.rhs)
        annotated.insert(self.dot, "•")
        return f"[{self.rule.lhs} -> {' '.join(annotated)}, @{self.start}]"


class ChartColumn:
    """
    One column of the Earley chart, representing all items that end at a
    particular input position.

    Items are stored in arrival order and processed sequentially (a simple
    agenda / worklist pattern).  When an item is first inserted its best
    known weight is recorded; if a cheaper path to the same item is later
    discovered the stored weight is updated.

    Attributes:
        _queue          (list):         Ordered list of items (may contain duplicates
                                        when a weight update triggers re-processing).
        best_weight     (dict):         Maps EarleyItem -> current best weight.
        backpointers    (defaultdict):  Maps EarleyItem -> list of (predecessor, evidence) pairs
                                        for parse-forest reconstruction.
        _cursor         (int):          Index of the next unprocessed item in _queue.
        listeners       (defaultdict):  Maps symbol -> list of items whose dot is on that symbol,
                                        used to accelerate the Completer step.
    """

    def __init__(self) -> None:
        self._queue: list = []
        self.best_weight: dict = {}
        self.backpointers: defaultdict = defaultdict(list)
        self._cursor: int = 0
        self.listeners: defaultdict = defaultdict(list)

    def enqueue(self, item: EarleyItem, weight: float, backpointer) -> None:
        """
        Add an item to this column, recording its weight and backpointer.

        If the item is new it is appended to the processing queue and
        registered in the listener index.  If it already exists but the new
        weight is strictly better, the weight table is updated and the item
        is queued again so that downstream items can be re-evaluated.

        Args:
            item:         The Earley item to insert.
            weight:       Path weight (negative log probability) for this item.
            backpointer:  A (predecessor_item, evidence) pair for tree
                          reconstruction, or None for seed items.
        """
        # Record the backpointer (parse-forest edge) if not already stored.
        if backpointer is not None and backpointer not in self.backpointers[item]:
            self.backpointers[item].append(backpointer)

        if item not in self.best_weight:
            # First time we've seen this item.
            self.best_weight[item] = weight
            self._queue.append(item)

            # Register so the Completer can find items waiting on this symbol.
            pending_symbol = item.symbol_at_dot()
            if pending_symbol is not None:
                self.listeners[pending_symbol].append(item)
        else:
            # We've seen this item before; update weight only if improved.
            if weight < self.best_weight[item]:
                self.best_weight[item] = weight
                self._queue.append(item)

    def dequeue(self):
        """
        Return the next unprocessed item in arrival order, or None if the
        queue is exhausted.

        Returns:
            EarleyItem | None
        """
        if self._cursor < len(self._queue):
            item = self._queue[self._cursor]
            self._cursor += 1
            return item
        return None


# ---------------------------------------------------------------------------
# Grammar
# ---------------------------------------------------------------------------

class ContextFreeGrammar:
    """
    A probabilistic context-free grammar (PCFG) loaded from a text file.

    The grammar pre-computes the transitive closure of the *left-corner*
    relation for each non-terminal so that the Predictor step can skip rules
    whose left-most symbol can never derive the next input token.

    Attributes:
        rules           (defaultdict): Maps LHS symbol -> list of ProductionRule.
        nonterminals    (set):         All non-terminal symbols in the grammar.
        left_corner_map (defaultdict): Maps non-terminal -> set of all symbols
                                       (terminal or non-terminal) that can appear
                                       as the first token of any derivation.
    """

    def __init__(self, filepath: str) -> None:
        """
        Load a grammar from a file.

        Each non-blank, non-comment line must have the format:
            <probability>  <LHS>  <RHS_1>  [<RHS_2> ...]

        Lines beginning with '#' are treated as comments.

        Args:
            filepath: Path to the grammar (.gr) file.
        """
        self.rules: defaultdict = defaultdict(list)
        self.nonterminals: set = set()

        self._load_rules(filepath)
        self.left_corner_map: defaultdict = self._build_left_corner_closure()

    def _load_rules(self, filepath: str) -> None:
        """
        Parse the grammar file and populate self.rules and self.nonterminals.

        Args:
            filepath: Path to the grammar file.
        """
        with open(filepath, "r") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                tokens = line.split()
                probability = float(tokens[0])
                lhs = tokens[1]
                rhs = tokens[2:]
                rule = ProductionRule(lhs, rhs, probability)
                self.rules[lhs].append(rule)
                self.nonterminals.add(lhs)

    def _build_left_corner_closure(self) -> defaultdict:
        """
        Compute the reflexive-transitive closure of the left-corner relation.

        The left corner of a rule  A -> alpha  is the first symbol of alpha.
        The closure ensures that if  A ->* B ... ->* c ...  then  c  appears
        in the left-corner set of  A.

        Returns:
            defaultdict mapping each non-terminal to a frozenset-compatible
            set of reachable left-corner symbols.
        """
        lc: defaultdict = defaultdict(set)

        # Base case: direct left corners from each rule.
        for lhs, rule_list in self.rules.items():
            for rule in rule_list:
                if rule.rhs:
                    lc[lhs].add(rule.rhs[0])

        # Iterative closure: propagate through non-terminal left corners.
        converged = False
        while not converged:
            converged = True
            for lhs in list(self.rules.keys()):
                before = len(lc[lhs])
                extended = set(lc[lhs])
                for symbol in lc[lhs]:
                    if self.is_nonterminal(symbol):
                        extended.update(lc[symbol])
                if len(extended) > before:
                    lc[lhs] = extended
                    converged = False

        return lc

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def is_nonterminal(self, symbol: str) -> bool:
        """
        Return True if *symbol* is a non-terminal in this grammar.

        Args:
            symbol: The symbol to check.

        Returns:
            bool
        """
        return symbol in self.nonterminals

    def can_derive_leading_token(self, symbol: str, token: str) -> bool:
        """
        Return True if *symbol* can derive a string whose first token is *token*.

        This uses the pre-computed left-corner closure for efficiency.

        Args:
            symbol: A grammar symbol (terminal or non-terminal).
            token:  An input token to check against.

        Returns:
            bool
        """
        return symbol == token or token in self.left_corner_map[symbol]


# ---------------------------------------------------------------------------
# Parse-forest reconstruction
# ---------------------------------------------------------------------------

def _collect_trees(item: EarleyItem, chart: list, end_col: int, visiting: frozenset):
    """
    Recursively collect all parse trees rooted at *item* that span up to
    *end_col* in the chart.

    Uses a visited-set to guard against infinite recursion in cyclic grammars.

    Args:
        item:     A complete EarleyItem whose derivations we want to enumerate.
        chart:    The full Earley chart (list of ChartColumn objects).
        end_col:  The chart column index where this item ends.
        visiting: Frozenset of (item, end_col) pairs currently on the call stack.

    Returns:
        list of (tree_node dict, float) pairs, where each tree node has keys
        'lhs', 'start', 'end', and 'children'.
    """
    state_key = (item, end_col)
    if state_key in visiting:
        return []
    visiting = visiting | {state_key}

    derivations = _collect_derivations(item, chart, end_col, visiting)
    result = []
    for child_list, total_weight in derivations:
        node = {
            "lhs":      item.rule.lhs,
            "start":    item.start,
            "end":      end_col,
            "children": child_list,
        }
        result.append((node, total_weight))
    return result


def _collect_derivations(item: EarleyItem, chart: list, end_col: int, visiting: frozenset):
    """
    Enumerate all ways of deriving the children for *item* ending at *end_col*.

    Backpointers stored in the chart column are followed recursively.  Each
    backpointer is a pair:
      - (predecessor_item, terminal_string): the predecessor scanned a terminal.
      - (predecessor_item, completed_item):  the predecessor was advanced by a
                                              completed non-terminal item.

    Args:
        item:     The EarleyItem whose derivation history we trace.
        chart:    The full Earley chart.
        end_col:  Column where this item ends.
        visiting: Cycle-detection set.

    Returns:
        list of ([child, ...], weight) pairs.
    """
    # Dot at position 0 means no symbols have been matched yet;
    # the only "derivation" is the empty child list at rule weight.
    if item.dot == 0:
        return [([], item.rule.weight)]

    backpointers = chart[end_col].backpointers.get(item, [])
    all_derivations = []

    for predecessor, evidence in backpointers:
        if isinstance(evidence, str):
            # The dot advanced over a terminal token.
            terminal = evidence
            left_derivs = _collect_derivations(predecessor, chart, end_col - 1, visiting)
            for prev_children, prev_weight in left_derivs:
                all_derivations.append((prev_children + [terminal], prev_weight))
        else:
            # The dot advanced over a completed non-terminal sub-tree.
            completed_item = evidence
            subtree_start = completed_item.start
            right_trees = _collect_trees(completed_item, chart, end_col, visiting)
            left_derivs = _collect_derivations(predecessor, chart, subtree_start, visiting)
            for prev_children, prev_weight in left_derivs:
                for subtree, subtree_weight in right_trees:
                    combined_weight = prev_weight + subtree_weight
                    all_derivations.append((prev_children + [subtree], combined_weight))

    return all_derivations


def render_tree(node, include_spans: bool) -> str:
    """
    Convert a parse-tree node (or a terminal string) into a bracketed string.

    Args:
        node:          Either a terminal string or a dict with keys
                       'lhs', 'start', 'end', and 'children'.
        include_spans: When True, annotate each non-terminal with its
                       [start, end] span indices.

    Returns:
        str: A Penn-Treebank-style bracketed representation.
    """
    if isinstance(node, str):
        return node

    child_strs = " ".join(render_tree(child, include_spans) for child in node["children"])
    if include_spans:
        return f"({node['lhs']} [{node['start']},{node['end']}] {child_strs})"
    return f"({node['lhs']} {child_strs})"


# ---------------------------------------------------------------------------
# Core Earley parsing algorithm
# ---------------------------------------------------------------------------

def run_earley_parser(
    tokens: list,
    grammar: ContextFreeGrammar,
    dump_chart: bool = False,
    show_progress: bool = False,
) -> list:
    """
    Run the Earley parsing algorithm on a list of tokens.

    The three standard Earley operations are:
    - **Predictor**: For an item expecting a non-terminal B at position i,
      seed column i with all rules for B whose left corner can derive token i.
    - **Scanner**: For an item expecting the terminal at position i, advance
      it into column i+1.
    - **Completer**: When an item is complete, advance all items in the
      originating column that were waiting for this non-terminal.

    Weights (negative log probabilities) are accumulated and the best weight
    per item is tracked; cheaper paths update the stored weight and trigger
    re-processing.

    Args:
        tokens:        Ordered list of input word tokens.
        grammar:       A loaded ContextFreeGrammar instance.
        dump_chart:    If True, print the full chart to stdout after parsing.
        show_progress: If True, display a tqdm progress bar over chart columns.

    Returns:
        A list of (tree_node, weight) tuples sorted by ascending weight
        (i.e. descending probability).  An empty list means the input is
        not in the grammar's language.
    """
    n = len(tokens)
    chart = [ChartColumn() for _ in range(n + 1)]

    # ---------- Initialisation: seed column 0 with ROOT rules ----------
    for rule in grammar.rules.get("ROOT", []):
        if n > 0 and not grammar.can_derive_leading_token(rule.rhs[0], tokens[0]):
            continue
        chart[0].enqueue(EarleyItem(rule, 0, 0), rule.weight, None)

    # ---------- Main loop over chart columns ----------
    column_range = tqdm(range(n + 1), desc="Parsing", leave=False) if show_progress else range(n + 1)

    for col_idx in column_range:
        already_predicted: set = set()

        while True:
            current_item = chart[col_idx].dequeue()
            if current_item is None:
                break

            pending = current_item.symbol_at_dot()
            item_weight = chart[col_idx].best_weight[current_item]

            if pending is None:
                # ---- Completer ----
                origin = current_item.start
                for waiting_item in chart[origin].listeners[current_item.rule.lhs]:
                    waiting_weight = chart[origin].best_weight[waiting_item]
                    combined = item_weight + waiting_weight
                    chart[col_idx].enqueue(
                        waiting_item.advance(),
                        combined,
                        (waiting_item, current_item),
                    )

            elif grammar.is_nonterminal(pending):
                # ---- Predictor ----
                if col_idx < n and pending not in already_predicted:
                    already_predicted.add(pending)
                    for candidate_rule in grammar.rules[pending]:
                        if grammar.can_derive_leading_token(candidate_rule.rhs[0], tokens[col_idx]):
                            chart[col_idx].enqueue(
                                EarleyItem(candidate_rule, 0, col_idx),
                                candidate_rule.weight,
                                None,
                            )

            else:
                # ---- Scanner ----
                if col_idx < n and tokens[col_idx] == pending:
                    chart[col_idx + 1].enqueue(
                        current_item.advance(),
                        item_weight,
                        (current_item, tokens[col_idx]),
                    )

    # ---------- Optional chart dump ----------
    if dump_chart:
        _print_chart(chart, tokens, n)

    # ---------- Collect complete ROOT parses from the final column ----------
    completed_parses = []
    for final_item in chart[n].best_weight:
        if (
            final_item.rule.lhs == "ROOT"
            and final_item.is_complete()
            and final_item.start == 0
        ):
            for tree_node, weight in _collect_trees(final_item, chart, n, frozenset()):
                completed_parses.append((tree_node, weight))

    completed_parses.sort(key=lambda pair: pair[1])
    return completed_parses


def _print_chart(chart: list, tokens: list, n: int) -> None:
    """
    Pretty-print the full Earley chart to stdout for debugging.

    Args:
        chart:  List of ChartColumn objects.
        tokens: The input token sequence.
        n:      Number of tokens (length of the input).
    """
    print(f"CHART  input: {' '.join(tokens)}")
    for col_idx in range(n + 1):
        print(f"  === Column {col_idx} ===")
        for item, weight in chart[col_idx].best_weight.items():
            marker = " ✓" if item.is_complete() else ""
            print(f"    {item}  w={weight:.4f}{marker}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Command-line interface for the probabilistic Earley parser.

    Reads a grammar file and a sentences file, parses each sentence, and
    prints either the single best parse or all parses, optionally annotated
    with span information.
    """
    cli = argparse.ArgumentParser(
        description="Weighted probabilistic Earley parser for context-free grammars."
    )
    cli.add_argument("grammar",   help="Path to the grammar file (.gr)")
    cli.add_argument("sentences", help="Path to the sentences file (.sen)")
    cli.add_argument(
        "--chart",
        action="store_true",
        help="Dump the full Earley chart to stdout after each parse.",
    )
    cli.add_argument(
        "--all-parses",
        action="store_true",
        help="Print every valid parse (useful for exploring ambiguity).",
    )
    cli.add_argument(
        "--spans",
        action="store_true",
        help="Annotate each non-terminal in the output tree with its [start,end] span.",
    )
    cli.add_argument(
        "--progress",
        action="store_true",
        help="Show a per-column progress bar (requires tqdm).",
    )
    args = cli.parse_args()

    grammar = ContextFreeGrammar(args.grammar)

    with open(args.sentences, "r") as sentence_file:
        for raw_line in sentence_file:
            sentence = raw_line.strip()
            if not sentence:
                continue

            tokens = sentence.split()
            parses = run_earley_parser(
                tokens,
                grammar,
                dump_chart=args.chart,
                show_progress=args.progress,
            )

            if args.all_parses:
                # ------ Print every parse found ------
                print(f"--- All parses for: {sentence} ---")
                if not parses:
                    print("NONE")
                for rank, (tree_node, weight) in enumerate(parses, start=1):
                    probability = 2 ** (-weight)
                    print(f"Parse {rank}: probability={probability:.6f}  weight={weight:.6f}")
                    print(render_tree(tree_node, include_spans=args.spans))
                print(f"--- Total: {len(parses)} parse(s) ---")

            else:
                # ------ Print only the highest-probability parse ------
                if not parses:
                    print("NONE")
                else:
                    best_tree, best_weight = parses[0]
                    print(render_tree(best_tree, include_spans=False))
                    if args.spans:
                        print(render_tree(best_tree, include_spans=True))
                    print(best_weight)


if __name__ == "__main__":
    main()