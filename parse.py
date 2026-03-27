#!/usr/bin/env python3
import sys
import math
import argparse
from collections import defaultdict

# Try to import tqdm for the progress bar, fallback to standard iterator if missing
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

class Rule:
    def __init__(self, lhs, rhs, prob):
        self.lhs = lhs
        self.rhs = tuple(rhs)
        self.weight = -math.log2(prob)

    def __repr__(self):
        return f"{self.lhs} -> {' '.join(self.rhs)}"

class Item:
    def __init__(self, rule, dot, start):
        self.rule = rule
        self.dot = dot
        self.start = start

    def next_symbol(self):
        if self.dot < len(self.rule.rhs):
            return self.rule.rhs[self.dot]
        return None

    def advance(self):
        return Item(self.rule, self.dot + 1, self.start)

    def is_complete(self):
        return self.dot == len(self.rule.rhs)

    def __hash__(self):
        return hash((self.rule.lhs, self.rule.rhs, self.dot, self.start))

    def __eq__(self, other):
        return (self.rule.lhs == other.rule.lhs and 
                self.rule.rhs == other.rule.rhs and 
                self.dot == other.dot and 
                self.start == other.start)

    def __repr__(self):
        rhs_list = list(self.rule.rhs)
        rhs_list.insert(self.dot, ".")
        return f"[{self.rule.lhs} -> {' '.join(rhs_list)}, {self.start}]"

class Agenda:
    def __init__(self):
        self.items = []
        self.item_dict = {}  
        self.all_bps = defaultdict(list) 
        self.processing_idx = 0
        self.waiting_for = defaultdict(list)

    def push(self, item, weight, bp):
        if bp is not None and bp not in self.all_bps[item]:
            self.all_bps[item].append(bp)
            
        if item not in self.item_dict:
            self.item_dict[item] = weight
            self.items.append(item)
            
            sym = item.next_symbol()
            if sym is not None:
                self.waiting_for[sym].append(item)
        else:
            if weight < self.item_dict[item]:
                self.item_dict[item] = weight
                self.items.append(item)

    def pop(self):
        if self.processing_idx < len(self.items):
            item = self.items[self.processing_idx]
            self.processing_idx += 1
            return item
        return None

class Grammar:
    def __init__(self, filepath):
        self.rules = defaultdict(list)
        self.non_terminals = set()
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts or line.startswith('#'):
                    continue
                prob = float(parts[0])
                lhs = parts[1]
                rhs = parts[2:]
                rule = Rule(lhs, rhs, prob)
                self.rules[lhs].append(rule)
                self.non_terminals.add(lhs)
                
        self.compute_left_corners()

    def is_nonterminal(self, symbol):
        return symbol in self.non_terminals

    def compute_left_corners(self):
        self.left_corners = defaultdict(set)
        for lhs, rules in self.rules.items():
            for r in rules:
                if r.rhs:
                    self.left_corners[lhs].add(r.rhs[0])

        changed = True
        while changed:
            changed = False
            for lhs in list(self.rules.keys()):
                current_size = len(self.left_corners[lhs])
                new_corners = set(self.left_corners[lhs])
                for sym in self.left_corners[lhs]:
                    if self.is_nonterminal(sym):
                        new_corners.update(self.left_corners[sym])
                if len(new_corners) > current_size:
                    self.left_corners[lhs] = new_corners
                    changed = True

    def can_start_with(self, top_symbol, target_word):
        if top_symbol == target_word:
            return True
        return target_word in self.left_corners[top_symbol]

# --- TREE RECONSTRUCTION ENGINE ---

def get_all_trees_structured(item, chart, end, visited=None):
    if visited is None: visited = set()
    state = (item, end)
    if state in visited: return [] 
    visited = visited | {state}
    
    derivs = get_all_derivations(item, chart, end, visited)
    trees = []
    for children, w in derivs:
        trees.append(({
            'lhs': item.rule.lhs,
            'start': item.start,
            'end': end,
            'children': children
        }, w))
    return trees

def get_all_derivations(item, chart, end, visited):
    if item.dot == 0:
        return [([], item.rule.weight)]
        
    bps = chart[end].all_bps.get(item, [])
    results = []
    for bp in bps:
        left_item, right_node = bp
        
        if isinstance(right_node, str):
            symbol = right_node
            left_derivs = get_all_derivations(left_item, chart, end - 1, visited)
            for left_children, left_w in left_derivs:
                results.append((left_children + [symbol], left_w))
        else:
            mid = right_node.start
            right_trees = get_all_trees_structured(right_node, chart, end, visited)
            left_derivs = get_all_derivations(left_item, chart, mid, visited)
            for left_children, left_w in left_derivs:
                for right_tree, right_w in right_trees:
                    results.append((left_children + [right_tree], left_w + right_w))
    return results

def format_tree(node, with_spans):
    if isinstance(node, str): return node
    children_strs = [format_tree(c, with_spans) for c in node['children']]
    children_joined = " ".join(children_strs)
    if with_spans:
        return f"({node['lhs']} [{node['start']},{node['end']}] {children_joined})"
    else:
        return f"({node['lhs']} {children_joined})"

# --- EARLEY PARSER CORE ---

def parse_sentence(words, grammar, print_chart=False, show_progress=False):
    n = len(words)
    chart = [Agenda() for _ in range(n + 1)]
    
    start_rules = grammar.rules.get('ROOT', [])
    for rule in start_rules:
        if n > 0 and not grammar.can_start_with(rule.rhs[0], words[0]):
            continue
        chart[0].push(Item(rule, 0, 0), rule.weight, None)

    col_iterator = tqdm(range(n + 1), desc="Parsing", leave=False) if show_progress else range(n + 1)

    for i in col_iterator:
        predicted_nonterminals = set()

        while True:
            item = chart[i].pop()
            if not item:
                break

            next_sym = item.next_symbol()
            current_weight = chart[i].item_dict[item]

            if next_sym is None:
                mid = item.start
                for customer in chart[mid].waiting_for[item.rule.lhs]:
                    cust_weight = chart[mid].item_dict[customer]
                    new_weight = current_weight + cust_weight
                    chart[i].push(customer.advance(), new_weight, (customer, item))
            
            elif grammar.is_nonterminal(next_sym):
                if i < n:
                    if next_sym not in predicted_nonterminals:
                        predicted_nonterminals.add(next_sym)
                        for rule in grammar.rules[next_sym]:
                            if grammar.can_start_with(rule.rhs[0], words[i]):
                                chart[i].push(Item(rule, 0, i), rule.weight, None)
            
            else:
                if i < n and words[i] == next_sym:
                    chart[i + 1].push(item.advance(), current_weight, (item, words[i]))

    if print_chart:
        print(f"CHART for: {' '.join(words)}")
        for j in range(n + 1):
            print(f"  === Column {j} ===")
            for item in chart[j].item_dict:
                w = chart[j].item_dict[item]
                comp = " ✓" if item.is_complete() else ""
                print(f"    {item} w={w:.4f}{comp}")
        print()

    all_parses = []
    for item in chart[n].items:
        if item.rule.lhs == 'ROOT' and item.is_complete() and item.start == 0:
            trees_with_weights = get_all_trees_structured(item, chart, n)
            for tree_node, weight in trees_with_weights:
                all_parses.append((tree_node, weight))
    
    all_parses.sort(key=lambda x: x[1])
    return all_parses

def main():
    parser = argparse.ArgumentParser(description="Optimized Probabilistic Earley parser")
    parser.add_argument("grammar", help="Path to .gr file")
    parser.add_argument("sentences", help="Path to .sen file")
    
    # User-requested flags
    parser.add_argument("--chart", action="store_true", help="Print the Earley chart")
    parser.add_argument("--all-parses", action="store_true", help="Print all parses (ambiguities)")
    parser.add_argument("--spans", action="store_true", help="Include spanned trees in the output")
    parser.add_argument("--progress", action="store_true", help="Show progress bar")
    args = parser.parse_args()
    
    grammar = Grammar(args.grammar)
    
    with open(args.sentences, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            words = line.split()
            
            all_parses = parse_sentence(words, grammar, print_chart=args.chart, show_progress=args.progress)
            
            if args.all_parses:
                # Print every valid parse found
                print(f"--- All parses for: {' '.join(words)} ---")
                if not all_parses:
                    print("NONE")
                for i, (tree_node, w) in enumerate(all_parses, 1):
                    prob = 2 ** (-w)
                    print(f"Parse {i}: probability = {prob:.6f}, weight = {w}")
                    if args.spans:
                        print(format_tree(tree_node, with_spans=True))
                    else:
                        print(format_tree(tree_node, with_spans=False))
                print(f"--- Total: {len(all_parses)} parse(s) ---")
                
            else:
                # Default behavior: Just the best parse
                if not all_parses:
                    print("NONE")
                else:
                    best_tree_node, best_w = all_parses[0]
                    # Always print the unspanned tree first
                    print(format_tree(best_tree_node, with_spans=False))
                    # Only print the spanned tree if the flag was passed
                    if args.spans:
                        print(format_tree(best_tree_node, with_spans=True))
                    # Always print the score/weight
                    print(best_w)

if __name__ == "__main__":
    main()