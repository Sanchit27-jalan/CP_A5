# CP_A5 Parser

This project contains `parse.py`, a weighted probabilistic Earley parser for context-free grammars.

## How to run

From the project root:

```bash
python3 parse.py <grammar_file.gr> <sentences_file.sen> [options]
```

Example:

```bash
python3 parse.py arith.gr arith.sen
```

## Arguments

### Required positional arguments

- `grammar` : Path to grammar file (`.gr`)
- `sentences` : Path to sentence file (`.sen`)

### Optional flags

- `--chart` : Print the full Earley chart after each parse
- `--all-parses` : Print all valid parses (instead of only the best parse)
- `--spans` : Include `[start,end]` span annotations in output trees
- `--progress` : Show per-column progress bar (`tqdm` if installed)

## More examples

Best parse only:

```bash
python3 parse.py timeflies.gr timeflies.sen
```

All parses with spans:

```bash
python3 parse.py papa.gr papa.sen --all-parses --spans
```

Debug chart + progress:

```bash
python3 parse.py soldier.gr soldier.sen --chart --progress
```
