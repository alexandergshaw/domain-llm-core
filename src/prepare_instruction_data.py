"""prepare_instruction_data.py – Convert instruction-tuning examples to training text.

Reads a JSONL file where each record has the fields:
    instruction, input, output, source_file, category

Formats every valid record as:

    <bos>
    ### Instruction:
    <instruction text>

    ### Input:
    <input text>

    ### Response:
    <output text>
    <eos>

Skips records with missing or empty required fields, printing a warning for each.
Prints a summary table of example counts grouped by category.
Writes all formatted records to the output file (default: data/train.txt).

Usage
-----
    python src/prepare_instruction_data.py \\
        --examples-file examples/instruction_examples.jsonl \\
        --output-file data/train.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_FIELDS: tuple[str, ...] = ("instruction", "input", "output", "source_file", "category")

BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_record(record: dict) -> str:
    """Return a formatted training string for a single instruction example.

    Format::

        <bos>
        ### Instruction:
        {instruction}

        ### Input:
        {input}

        ### Response:
        {output}
        <eos>
    """
    return (
        f"{BOS_TOKEN}\n"
        f"### Instruction:\n{record['instruction'].strip()}\n\n"
        f"### Input:\n{record['input'].strip()}\n\n"
        f"### Response:\n{record['output'].strip()}\n"
        f"{EOS_TOKEN}"
    )


# Fields that must be present AND non-empty.
# `input` is intentionally excluded: many instruction-tuning examples have no
# additional context, so an empty string is perfectly valid there.
_NON_EMPTY_FIELDS: tuple[str, ...] = ("instruction", "output", "source_file", "category")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_record(record: dict, line_number: int) -> list[str]:
    """Return a list of validation error messages for *record*.

    An empty list means the record is valid.

    All fields in REQUIRED_FIELDS must be present and must be strings.
    Fields in ``_NON_EMPTY_FIELDS`` must additionally be non-empty.
    The ``input`` field may be an empty string (no additional context needed).
    """
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in record:
            errors.append(f"missing field '{field}'")
        elif not isinstance(record[field], str):
            errors.append(f"field '{field}' must be a string, got {type(record[field]).__name__}")
        elif field in _NON_EMPTY_FIELDS and record[field].strip() == "":
            errors.append(f"field '{field}' is empty")
    return errors


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_file(examples_path: Path, output_path: Path) -> None:
    """Read *examples_path*, format valid records, write to *output_path*."""

    if not examples_path.exists():
        print(
            f"ERROR: Examples file not found: '{examples_path}'.\n"
            f"  Create the file or pass --examples-file to specify a different path.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    formatted_records: list[str] = []
    category_counts: Counter = Counter()
    total_rows = 0
    skipped_rows = 0

    with examples_path.open(encoding="utf-8") as fh:
        for line_number, raw_line in enumerate(fh, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue  # skip blank lines silently

            total_rows += 1

            # Parse JSON
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                warnings.warn(
                    f"Line {line_number}: JSON parse error – {exc}. Skipping.",
                    stacklevel=1,
                )
                skipped_rows += 1
                continue

            if not isinstance(record, dict):
                warnings.warn(
                    f"Line {line_number}: expected a JSON object, got {type(record).__name__}. Skipping.",
                    stacklevel=1,
                )
                skipped_rows += 1
                continue

            # Validate required fields
            errors = validate_record(record, line_number)
            if errors:
                error_str = "; ".join(errors)
                warnings.warn(
                    f"Line {line_number}: invalid record ({error_str}). Skipping.",
                    stacklevel=1,
                )
                skipped_rows += 1
                continue

            # Record is valid
            formatted_records.append(format_record(record))
            category_counts[record["category"].strip()] += 1

    # Write output
    with output_path.open("w", encoding="utf-8") as fh:
        fh.write("\n\n".join(formatted_records))
        if formatted_records:
            fh.write("\n")  # trailing newline

    # Print summary
    valid_count = len(formatted_records)
    print(f"\nProcessed {total_rows} rows from '{examples_path}'")
    print(f"  Valid   : {valid_count}")
    print(f"  Skipped : {skipped_rows}")

    if category_counts:
        print("\nExamples by category:")
        max_cat_len = max(len(c) for c in category_counts)
        for category, count in sorted(category_counts.items()):
            print(f"  {category:<{max_cat_len}}  {count:>5}")

    print(f"\nOutput written to '{output_path}' ({valid_count} examples).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert instruction JSONL examples into a training text file."
    )
    parser.add_argument(
        "--examples-file",
        type=Path,
        default=Path("examples/instruction_examples.jsonl"),
        help="Path to the input JSONL file (default: examples/instruction_examples.jsonl).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/train.txt"),
        help="Path to the output training text file (default: data/train.txt).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    process_file(args.examples_file, args.output_file)


if __name__ == "__main__":
    main()
