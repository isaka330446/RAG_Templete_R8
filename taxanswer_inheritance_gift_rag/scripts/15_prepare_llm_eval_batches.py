from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from eval_common import read_jsonl, resolve_path, write_jsonl


DEFAULT_FIELDS = [
    "question_id",
    "prediction_id",
    "source_question_id",
    "retrieval_mode",
    "question",
    "expected_answer",
    "short_answer",
    "answer_type",
    "difficulty",
    "question_scope",
    "expected_sources",
    "expected_sources_json",
    "must_include_json",
    "must_not_include_json",
    "retrieval_keywords_json",
    "grading_rubric_json",
    "actual_answer",
    "sources_json",
    "retrieved_contexts_json",
    "source_refs_json",
    "expected_source_hit",
    "answer_source",
    "cache_hit",
    "elapsed_sec",
    "status",
    "error",
]

TEXT_LIMIT_FIELDS = {
    "expected_answer",
    "short_answer",
    "actual_answer",
    "error",
}

JSON_FIELD_NAMES = {
    "expected_sources_json",
    "must_include_json",
    "must_not_include_json",
    "retrieval_keywords_json",
    "grading_rubric_json",
    "sources_json",
    "retrieved_contexts_json",
    "source_refs_json",
}

SOURCE_TEXT_KEYS = {
    "child_text",
    "parent_text",
    "text",
    "snippet",
    "content",
}

def truncate_text(value: Any, limit: int) -> str:
    text = str(value or "")
    if limit <= 0 or len(text) <= limit:
        return text
    omitted = len(text) - limit
    return text[:limit].rstrip() + f"\n...[truncated {omitted} chars]"


def parse_jsonish(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    text = str(value or "").strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def compact_json_value(value: Any, *, max_context_chars: int) -> Any:
    parsed = parse_jsonish(value)
    if isinstance(parsed, list):
        return [compact_json_value(item, max_context_chars=max_context_chars) for item in parsed]
    if isinstance(parsed, dict):
        return {
            key: truncate_text(val, max_context_chars) if key in SOURCE_TEXT_KEYS else compact_json_value(val, max_context_chars=max_context_chars)
            for key, val in parsed.items()
        }
    if isinstance(parsed, str):
        return truncate_text(parsed, max_context_chars)
    return parsed


def compact_row(
    row: dict[str, Any],
    fields: list[str],
    *,
    max_context_chars: int,
    max_answer_chars: int,
    no_truncate: bool,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    context_limit = 0 if no_truncate else max_context_chars
    answer_limit = 0 if no_truncate else max_answer_chars
    for field in fields:
        if field not in row:
            continue
        value = row.get(field)
        if field in JSON_FIELD_NAMES:
            output[field] = compact_json_value(value, max_context_chars=context_limit)
        elif field in TEXT_LIMIT_FIELDS:
            output[field] = truncate_text(value, answer_limit)
        else:
            output[field] = value
    return output


def chunk_rows(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [rows[idx : idx + batch_size] for idx in range(0, len(rows), batch_size)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Split predictions.jsonl into compact LLM-evaluation batches.")
    parser.add_argument("--input", required=True, help="Path to predictions.jsonl.")
    parser.add_argument("--output-dir", default="", help="Defaults to <input_dir>/llm_eval_batches.")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-context-chars", type=int, default=1200)
    parser.add_argument("--max-answer-chars", type=int, default=2500)
    parser.add_argument("--no-truncate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    input_path = resolve_path(args.input, Path(args.input))
    output_dir = resolve_path(args.output_dir, input_path.parent / "llm_eval_batches") if args.output_dir else input_path.parent / "llm_eval_batches"
    rows = read_jsonl(input_path)
    compacted = [
        compact_row(
            row,
            DEFAULT_FIELDS,
            max_context_chars=args.max_context_chars,
            max_answer_chars=args.max_answer_chars,
            no_truncate=args.no_truncate,
        )
        for row in rows
    ]
    batches = chunk_rows(compacted, args.batch_size)

    manifest = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "total_rows": len(rows),
        "batch_size": args.batch_size,
        "batch_count": len(batches),
        "fields": DEFAULT_FIELDS,
        "max_context_chars": None if args.no_truncate else args.max_context_chars,
        "max_answer_chars": None if args.no_truncate else args.max_answer_chars,
    }
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, batch in enumerate(batches, start=1):
        batch_path = output_dir / f"batch_{idx:03d}.jsonl"
        write_jsonl(batch_path, batch)

    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**manifest, "rows_per_batch": [len(batch) for batch in batches], "estimated_batches": math.ceil(len(rows) / args.batch_size) if args.batch_size else 0}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
