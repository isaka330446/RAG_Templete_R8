from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parent.parent))

from api.config import get_required_url
from eval_common import read_jsonl, resolve_path, write_csv, write_jsonl


def metric_result(metric: Any, test_case: Any) -> dict[str, Any]:
    metric.measure(test_case)
    return {
        "score": getattr(metric, "score", None),
        "success": getattr(metric, "success", None),
        "reason": getattr(metric, "reason", ""),
    }


def is_no_answer_row(row: dict[str, Any]) -> bool:
    return str(row.get("answer_type") or "").casefold() == "no_answer" or str(row.get("is_no_answer")).casefold() in {"1", "true", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate prediction run with DeepEval.")
    parser.add_argument("--input", required=True, help="predictions.jsonl path")
    parser.add_argument("--output-dir", default="", help="Defaults to the input file directory.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--local-openai", action="store_true")
    parser.add_argument("--base-url", default=get_required_url("eval_llm_base_url"))
    parser.add_argument("--model", default=os.getenv("EVAL_LLM_MODEL", "local-eval-model"))
    parser.add_argument("--api-key", default=os.getenv("EVAL_LLM_API_KEY", "dummy"))
    parser.add_argument("--include-no-answer", action="store_true", help="Include answer_type=no_answer rows in DeepEval metrics.")
    args = parser.parse_args()

    os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
    os.environ.setdefault("CONFIDENT_AI_DISABLE_TELEMETRY", "1")
    if args.local_openai:
        os.environ["OPENAI_API_KEY"] = args.api_key
        os.environ["OPENAI_BASE_URL"] = args.base_url

    try:
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            ContextualRelevancyMetric,
            FaithfulnessMetric,
        )
        from deepeval.test_case import LLMTestCase
    except ImportError as exc:
        raise SystemExit(
            "DeepEval dependencies are not installed. Run: pip install -r requirements_eval.txt"
        ) from exc

    input_path = resolve_path(args.input, Path(args.input))
    output_dir = resolve_path(args.output_dir, input_path.parent) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [row for row in read_jsonl(input_path) if row.get("status") == "ok"]
    if not args.include_no_answer:
        rows = [row for row in rows if not is_no_answer_row(row)]
    if args.limit:
        rows = rows[: args.limit]

    metric_builders = {
        "answer_relevancy": lambda: AnswerRelevancyMetric(model=args.model),
        "faithfulness": lambda: FaithfulnessMetric(model=args.model),
        "contextual_precision": lambda: ContextualPrecisionMetric(model=args.model),
        "contextual_recall": lambda: ContextualRecallMetric(model=args.model),
        "contextual_relevancy": lambda: ContextualRelevancyMetric(model=args.model),
    }
    score_rows: list[dict[str, Any]] = []
    for row in rows:
        contexts = json.loads(row.get("retrieved_contexts_json") or "[]")
        test_case = LLMTestCase(
            input=row.get("question", ""),
            actual_output=row.get("actual_answer", ""),
            expected_output=row.get("expected_answer", ""),
            retrieval_context=contexts,
        )
        out = {
            "question_id": row.get("question_id"),
            "question": row.get("question"),
        }
        for name, builder in metric_builders.items():
            try:
                result = metric_result(builder(), test_case)
                out[f"{name}_score"] = result.get("score")
                out[f"{name}_success"] = result.get("success")
                out[f"{name}_reason"] = result.get("reason")
            except Exception as exc:
                out[f"{name}_error"] = str(exc)
        score_rows.append(out)

    write_jsonl(output_dir / "deepeval_scores.jsonl", score_rows)
    write_csv(output_dir / "deepeval_scores.csv", score_rows)
    summary = {"count": len(score_rows), "include_no_answer": args.include_no_answer}
    numeric_keys = sorted({key for row in score_rows for key, value in row.items() if key.endswith("_score") and isinstance(value, (int, float))})
    for key in numeric_keys:
        vals = [float(row[key]) for row in score_rows if isinstance(row.get(key), (int, float))]
        if vals:
            summary[f"avg_{key}"] = sum(vals) / len(vals)
    (output_dir / "deepeval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
