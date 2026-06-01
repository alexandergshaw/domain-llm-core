"""tests/test_prepare_instruction_data.py – Tests for prepare_instruction_data.py."""

from __future__ import annotations

import json
import textwrap
import warnings
from pathlib import Path

import pytest

from src.prepare_instruction_data import (
    BOS_TOKEN,
    EOS_TOKEN,
    REQUIRED_FIELDS,
    _NON_EMPTY_FIELDS,
    format_record,
    process_file,
    validate_record,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_record(**kwargs) -> dict:
    """Return a fully valid record with optional field overrides."""
    base = {
        "instruction": "Summarize this.",
        "input": "Some text.",
        "output": "A summary.",
        "source_file": "test.txt",
        "category": "summarization",
    }
    base.update(kwargs)
    return base


def write_jsonl(path: Path, records: list) -> None:
    """Write a list of objects (or raw strings) to a JSONL file."""
    with path.open("w", encoding="utf-8") as fh:
        for item in records:
            if isinstance(item, str):
                fh.write(item + "\n")
            else:
                fh.write(json.dumps(item) + "\n")


# ---------------------------------------------------------------------------
# format_record
# ---------------------------------------------------------------------------

class TestFormatRecord:
    def test_contains_bos_and_eos(self):
        rec = make_record()
        text = format_record(rec)
        assert text.startswith(BOS_TOKEN)
        assert text.endswith(EOS_TOKEN)

    def test_contains_instruction_header(self):
        rec = make_record(instruction="Do something.")
        text = format_record(rec)
        assert "### Instruction:" in text
        assert "Do something." in text

    def test_contains_input_header(self):
        rec = make_record(input="My input.")
        text = format_record(rec)
        assert "### Input:" in text
        assert "My input." in text

    def test_contains_response_header(self):
        rec = make_record(output="My output.")
        text = format_record(rec)
        assert "### Response:" in text
        assert "My output." in text

    def test_strips_whitespace(self):
        rec = make_record(instruction="  Trim me.  ", input="  also trim  ", output="  trimmed  ")
        text = format_record(rec)
        assert "Trim me." in text
        assert "also trim" in text
        assert "trimmed" in text

    def test_full_format_order(self):
        rec = make_record(instruction="A", input="B", output="C")
        text = format_record(rec)
        bos_pos = text.index(BOS_TOKEN)
        instr_pos = text.index("### Instruction:")
        inp_pos = text.index("### Input:")
        resp_pos = text.index("### Response:")
        eos_pos = text.index(EOS_TOKEN)
        assert bos_pos < instr_pos < inp_pos < resp_pos < eos_pos


# ---------------------------------------------------------------------------
# validate_record
# ---------------------------------------------------------------------------

class TestValidateRecord:
    def test_valid_record_no_errors(self):
        errors = validate_record(make_record(), line_number=1)
        assert errors == []

    @pytest.mark.parametrize("field", REQUIRED_FIELDS)
    def test_missing_field(self, field):
        rec = make_record()
        del rec[field]
        errors = validate_record(rec, line_number=1)
        assert any(field in e for e in errors)

    @pytest.mark.parametrize("field", [f for f in REQUIRED_FIELDS if f != "input"])
    def test_empty_field(self, field):
        rec = make_record(**{field: "   "})
        errors = validate_record(rec, line_number=1)
        assert any(field in e for e in errors)

    def test_empty_input_is_valid(self):
        """An empty 'input' field is allowed (no additional context needed)."""
        rec = make_record(input="")
        errors = validate_record(rec, line_number=1)
        assert errors == []

    @pytest.mark.parametrize("field", REQUIRED_FIELDS)
    def test_non_string_field(self, field):
        rec = make_record(**{field: 123})
        errors = validate_record(rec, line_number=1)
        assert any(field in e for e in errors)


# ---------------------------------------------------------------------------
# process_file – happy path
# ---------------------------------------------------------------------------

class TestProcessFileHappyPath:
    def test_writes_output_file(self, tmp_path):
        examples = tmp_path / "examples.jsonl"
        output = tmp_path / "out.txt"
        write_jsonl(examples, [make_record()])
        process_file(examples, output)
        assert output.exists()

    def test_output_contains_formatted_record(self, tmp_path):
        examples = tmp_path / "examples.jsonl"
        output = tmp_path / "out.txt"
        rec = make_record(instruction="Explain X.", input="Context.", output="Answer.")
        write_jsonl(examples, [rec])
        process_file(examples, output)
        content = output.read_text(encoding="utf-8")
        assert BOS_TOKEN in content
        assert "### Instruction:" in content
        assert "Explain X." in content
        assert "### Input:" in content
        assert "Context." in content
        assert "### Response:" in content
        assert "Answer." in content
        assert EOS_TOKEN in content

    def test_multiple_records_separated(self, tmp_path):
        examples = tmp_path / "examples.jsonl"
        output = tmp_path / "out.txt"
        records = [make_record(instruction=f"Task {i}.") for i in range(3)]
        write_jsonl(examples, records)
        process_file(examples, output)
        content = output.read_text(encoding="utf-8")
        assert content.count(BOS_TOKEN) == 3
        assert content.count(EOS_TOKEN) == 3

    def test_creates_output_directory(self, tmp_path):
        examples = tmp_path / "examples.jsonl"
        output = tmp_path / "nested" / "dir" / "out.txt"
        write_jsonl(examples, [make_record()])
        process_file(examples, output)
        assert output.exists()

    def test_blank_lines_ignored(self, tmp_path):
        examples = tmp_path / "examples.jsonl"
        output = tmp_path / "out.txt"
        examples.write_text(
            "\n" + json.dumps(make_record()) + "\n\n",
            encoding="utf-8",
        )
        process_file(examples, output)
        content = output.read_text(encoding="utf-8")
        assert content.count(BOS_TOKEN) == 1

    def test_real_examples_jsonl(self, tmp_path):
        """Process the actual examples/instruction_examples.jsonl if it exists."""
        repo_root = Path(__file__).parent.parent
        real_examples = repo_root / "examples" / "instruction_examples.jsonl"
        if not real_examples.exists():
            pytest.skip("examples/instruction_examples.jsonl not found")
        output = tmp_path / "out.txt"
        process_file(real_examples, output)
        content = output.read_text(encoding="utf-8")
        # Should have at least one valid record
        assert BOS_TOKEN in content


# ---------------------------------------------------------------------------
# process_file – error / skip handling
# ---------------------------------------------------------------------------

class TestProcessFileSkipping:
    def test_skips_invalid_json(self, tmp_path):
        examples = tmp_path / "examples.jsonl"
        output = tmp_path / "out.txt"
        examples.write_text("not valid json\n", encoding="utf-8")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            process_file(examples, output)
        assert any("JSON parse error" in str(w.message) for w in caught)
        # Output file should exist but be effectively empty
        content = output.read_text(encoding="utf-8")
        assert BOS_TOKEN not in content

    def test_skips_record_missing_field(self, tmp_path):
        examples = tmp_path / "examples.jsonl"
        output = tmp_path / "out.txt"
        rec = make_record()
        del rec["output"]
        write_jsonl(examples, [rec])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            process_file(examples, output)
        assert any("invalid record" in str(w.message) for w in caught)
        content = output.read_text(encoding="utf-8")
        assert BOS_TOKEN not in content

    def test_skips_record_empty_field(self, tmp_path):
        examples = tmp_path / "examples.jsonl"
        output = tmp_path / "out.txt"
        rec = make_record(instruction="  ")
        write_jsonl(examples, [rec])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            process_file(examples, output)
        assert any("invalid record" in str(w.message) for w in caught)

    def test_valid_and_invalid_mixed(self, tmp_path):
        examples = tmp_path / "examples.jsonl"
        output = tmp_path / "out.txt"
        bad = make_record()
        del bad["category"]
        records = [make_record(instruction="Good 1."), bad, make_record(instruction="Good 2.")]
        write_jsonl(examples, records)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            process_file(examples, output)
        content = output.read_text(encoding="utf-8")
        assert content.count(BOS_TOKEN) == 2

    def test_missing_examples_file_exits(self, tmp_path):
        missing = tmp_path / "nonexistent.jsonl"
        output = tmp_path / "out.txt"
        with pytest.raises(SystemExit):
            process_file(missing, output)

    def test_non_object_json_line_skipped(self, tmp_path):
        examples = tmp_path / "examples.jsonl"
        output = tmp_path / "out.txt"
        examples.write_text('["a", "b", "c"]\n', encoding="utf-8")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            process_file(examples, output)
        assert any("JSON object" in str(w.message) for w in caught)

    def test_empty_input_not_skipped(self, tmp_path):
        """Records with an empty 'input' field should be included, not skipped."""
        examples = tmp_path / "examples.jsonl"
        output = tmp_path / "out.txt"
        rec = make_record(input="")
        write_jsonl(examples, [rec])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            process_file(examples, output)
        assert not any("invalid record" in str(w.message) for w in caught)
        content = output.read_text(encoding="utf-8")
        assert content.count(BOS_TOKEN) == 1
