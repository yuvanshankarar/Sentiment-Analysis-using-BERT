# Sentiment Analysis CLI — project.py

**Date:** May 24, 2025

**Description:** A command-line sentiment analysis tool that classifies input
text as POSITIVE or NEGATIVE using a pre-trained DistilBERT model hosted on
Hugging Face. Supports inline text arguments and batch processing from a file.

---

## Table of Contents

1. Summary & Dependencies
2. Full Source Code
3. Function / Class Reference
4. Usage
5. Appendix — Markdown Source

---

## 1. Summary & Dependencies

`project.py` is a lightweight CLI wrapper around Hugging Face Transformers'
`pipeline` API. It loads the `distilbert-base-uncased-finetuned-sst-2-english`
model on first use (lazy singleton), classifies one or more text strings, and
prints label + confidence score to stdout.

**Dependencies:**

| Package | Role | Notes |
|---|---|---|
| `transformers` | Core ML pipeline & tokeniser | `pip install transformers` |
| `torch` / `tensorflow` | Backend tensor engine | One required |
| `argparse` | CLI argument parsing | Python stdlib |
| `pathlib` | File-path abstraction | Python stdlib |
| `os` | Environment variable access | Python stdlib |

---

## 2. Full Source Code

```python
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

from transformers import Pipeline, pipeline

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
HF_TOKEN_ENV = "HF_TOKEN"

_classifier: Pipeline | None = None


def get_classifier() -> Pipeline:
    global _classifier
    if _classifier is None:
        auth_token = os.environ.get(HF_TOKEN_ENV)
        pipeline_kwargs = {
            "model": MODEL_NAME,
            "tokenizer": MODEL_NAME,
            "return_all_scores": False,
        }
        if auth_token:
            pipeline_kwargs["use_auth_token"] = auth_token

        _classifier = pipeline(
            "sentiment-analysis",
            **pipeline_kwargs,
        )
    return _classifier


def predict_sentiment(texts: Iterable[str]) -> list[dict[str, object]]:
    classifier = get_classifier()
    return classifier(list(texts), truncation=True)


def format_prediction(text: str, prediction: dict[str, object]) -> str:
    label = prediction.get("label", "UNKNOWN")
    score = prediction.get("score", 0.0)
    return f"{label} ({score:.4f}) - {text}"


def load_texts_from_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sentiment analysis using a pretrained BERT-based model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--text",
        "-t",
        nargs="+",
        help="One or more text strings to analyze.",
    )
    parser.add_argument(
        "--input-file",
        "-i",
        type=Path,
        help="Path to a UTF-8 text file with one sentence per line.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    texts: list[str] = []
    if args.text:
        texts.extend(args.text)

    if args.input_file:
        if not args.input_file.exists():
            raise SystemExit(f"Error: input file does not exist: {args.input_file}")
        texts_from_file = load_texts_from_file(args.input_file)
        texts.extend(texts_from_file)

    if not texts:
        parser.print_help()
        print("\nExample:\n  python project.py -t \"I love this movie\" \"This is terrible\"")
        raise SystemExit(0)

    predictions = predict_sentiment(texts)
    for text, prediction in zip(texts, predictions):
        print(format_prediction(text, prediction))


if __name__ == "__main__":
    main()
```

---

## 3. Function / Class Reference

### `get_classifier() -> Pipeline`

Implements a lazy singleton pattern for the HuggingFace pipeline. On the first
call it reads the `HF_TOKEN` environment variable; if present, the token is
forwarded as `use_auth_token` to support private or gated models. The pipeline
is configured for sentiment-analysis with `return_all_scores=False` so only
the winning label is returned. Subsequent calls skip initialisation and return
the cached object, making repeated inference cheap.

### `predict_sentiment(texts: Iterable[str]) -> list[dict[str, object]]`

Accepts any iterable of strings and materialises it into a list before
forwarding to the pipeline. Token truncation is always enabled via
`truncation=True`, preventing runtime errors on inputs longer than the model's
512-token context window. Returns a list of dicts, one per input, each
containing `label` (str) and `score` (float).

### `format_prediction(text: str, prediction: dict) -> str`

A pure formatting function with no side effects. Extracts `label` and `score`
from the prediction dict with safe defaults so it never raises on malformed
pipeline output. The score is rendered to four decimal places, giving callers
a stable, machine-parseable output format convenient for downstream piping.

### `load_texts_from_file(path: Path) -> list[str]`

Reads the entire file at `path` as UTF-8, splits on newlines, strips whitespace
from every line, and discards blank lines. Returns an empty list for files that
are empty or contain only whitespace. Delegates to `Path.read_text`, which
guarantees the file handle is always closed.

### `build_parser() -> argparse.ArgumentParser`

Constructs and returns the argument parser used by `main()`. Separating parser
construction into its own function makes the CLI interface independently
testable. The `--text` flag uses `nargs='+'` to accept multiple strings in one
invocation; `--input-file` accepts a `Path` object validated at runtime.

### `main() -> None`

The script's entry point. Orchestrates argument parsing, input aggregation from
both CLI text and file sources, and output formatting. If neither source
provides any text, prints the help message plus a quick example and exits with
code 0. File existence is validated before reading for clear error messages.

---

## 4. Usage

### Installation

```bash
pip install transformers torch
# or with TensorFlow:
pip install transformers tensorflow
```

### Run examples

```bash
# Single inline text
python project.py -t "I love this movie!"

# Multiple strings
python project.py -t "Great film!" "Absolutely terrible."

# From file (one sentence per line)
python project.py -i reviews.txt

# Combined
python project.py -t "Quick test" -i reviews.txt

# Private / gated model
HF_TOKEN=hf_xxxx python project.py -t "Hello"

# Show help
python project.py --help
```

### Expected output

```
POSITIVE (0.9998) - I love this movie!
NEGATIVE (0.9994) - Absolutely terrible.
```

> **Note:** `HF_TOKEN` is optional. The default public model requires no token.

---

## 5. Appendix — Markdown Source

This document **is** the Markdown source. Save this file as `project.md` and
convert to PDF with:

```bash
pandoc project.md -o project_documentation.pdf \
  --pdf-engine=xelatex \
  -V geometry:margin=1in \
  --highlight-style=tango
```
