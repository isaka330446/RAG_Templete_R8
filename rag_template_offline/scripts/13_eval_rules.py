from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval_common import read_jsonl, resolve_path, write_csv, write_jsonl


NO_ANSWER_MARKERS = [
    "確認できません",
    "該当する記載がありません",
    "根拠文書が見つかりません",
    "見つかりませんでした",
    "判断できません",
    "回答できません",
]


def parse_json_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw = value
    else:
        text = str(value or "").strip()
        if not text:
            return []
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            raw = [item.strip() for item in text.replace("、", "|").split("|")]
    return [str(item).strip() for item in raw if str(item).strip()]


def is_no_answer_row(row: dict[str, Any]) -> bool:
    return str(row.get("answer_type") or "").casefold() == "no_answer" or str(row.get("is_no_answer")).casefold() in {"1", "true", "yes"}


def score_row(row: dict[str, Any]) -> dict[str, Any]:
    answer = str(row.get("actual_answer") or "")
    must_include = parse_json_list(row.get("must_include_json"))
    must_not_include = parse_json_list(row.get("must_not_include_json"))
    include_hits = [term for term in must_include if term in answer]
    include_missing = [term for term in must_include if term not in answer]
    forbidden_hits = [term for term in must_not_include if term in answer]
    no_answer = is_no_answer_row(row)
    no_answer_ok = any(marker in answer for marker in NO_ANSWER_MARKERS) if no_answer else None
    include_score = len(include_hits) / len(must_include) if must_include else 1.0
    forbidden_score = 1.0 if not forbidden_hits else 0.0
    no_answer_score = 1.0 if no_answer_ok else 0.0 if no_answer else 1.0
    total_score = (include_score + forbidden_score + no_answer_score) / 3
    return {
        "question_id": row.get("question_id"),
        "question": row.get("question"),
        "answer_type": row.get("answer_type"),
        "status": row.get("status"),
        "include_score": include_score,
        "forbidden_score": forbidden_score,
        "no_answer_score": no_answer_score,
        "rule_score": total_score,
        "must_include_count": len(must_include),
        "must_include_hit_count": len(include_hits),
        "must_include_missing_json": json.dumps(include_missing, ensure_ascii=False),
        "must_not_include_hit_json": json.dumps(forbidden_hits, ensure_ascii=False),
        "is_no_answer": no_answer,
        "no_answer_ok": no_answer_ok,
        "expected_source_hit": row.get("expected_source_hit"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight rule-based evaluation over predictions.jsonl.")
    parser.add_argument("--input", required=True, help="predictions.jsonl path")
    parser.add_argument("--output-dir", default="", help="Defaults to the input file directory.")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    input_path = resolve_path(args.input, Path(args.input))
    output_dir = resolve_path(args.output_dir, input_path.parent) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [row for row in read_jsonl(input_path) if row.get("status") == "ok"]
    if args.limit:
        rows = rows[: args.limit]
    scores = [score_row(row) for row in rows]
    write_jsonl(output_dir / "rule_scores.jsonl", scores)
    write_csv(output_dir / "rule_scores.csv", scores)
    avg = sum(float(row.get("rule_score") or 0.0) for row in scores) / len(scores) if scores else 0.0
    source_hits = sum(1 for row in scores if str(row.get("expected_source_hit")).casefold() in {"1", "true", "yes"})
    summary = {
        "count": len(scores),
        "avg_rule_score": avg,
        "expected_source_hit_rate": source_hits / len(scores) if scores else 0.0,
        "no_answer_count": sum(1 for row in scores if row.get("is_no_answer")),
    }
    (output_dir / "rule_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
