from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parent.parent))

from api.config import get_required_url, load_settings
from eval_common import read_jsonl, resolve_path, write_csv, write_jsonl
from ragas_compat import build_openai_compatible_ragas_models, ensure_ragas_import_compat, install_hint


SETTINGS = load_settings()
EVAL_LLM_SETTINGS = SETTINGS.get("eval_llm") or SETTINGS.get("alias_llm") or SETTINGS.get("llm") or {}
EMBEDDING_SETTINGS = SETTINGS.get("embedding") or {}


def is_no_answer_row(row: dict[str, Any]) -> bool:
    return str(row.get("answer_type") or "").casefold() == "no_answer" or str(row.get("is_no_answer")).casefold() in {"1", "true", "yes"}


def load_predictions(
    path: Path,
    limit: int = 0,
    include_no_answer: bool = False,
    retrieval_mode: str = "",
    include_retrieval_only: bool = False,
) -> list[dict[str, Any]]:
    rows = [row for row in read_jsonl(path) if row.get("status") == "ok"]
    if retrieval_mode:
        rows = [row for row in rows if str(row.get("retrieval_mode") or "").casefold() == retrieval_mode.casefold()]
    if not include_no_answer:
        rows = [row for row in rows if not is_no_answer_row(row)]
    if not include_retrieval_only:
        rows = [row for row in rows if str(row.get("actual_answer") or "").strip()]
    return rows[:limit] if limit else rows


def configure_openai_compatible_env(args: argparse.Namespace) -> None:
    if args.base_url:
        os.environ["OPENAI_BASE_URL"] = args.base_url
        os.environ["OPENAI_API_BASE"] = args.base_url
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    if args.model:
        os.environ["OPENAI_MODEL_NAME"] = args.model
    if args.embedding_base_url:
        os.environ["EVAL_EMBEDDING_BASE_URL"] = args.embedding_base_url
        os.environ["OPENAI_EMBEDDING_BASE_URL"] = args.embedding_base_url
    if args.embedding_model:
        os.environ["EVAL_EMBEDDING_MODEL"] = args.embedding_model
        os.environ["OPENAI_EMBEDDING_MODEL"] = args.embedding_model
    if args.embedding_api_key:
        os.environ["EVAL_EMBEDDING_API_KEY"] = args.embedding_api_key


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate prediction run with RAGAS.")
    parser.add_argument("--input", required=True, help="predictions.jsonl path")
    parser.add_argument("--output-dir", default="", help="Defaults to the input file directory.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--base-url", default=get_required_url("eval_llm_base_url"))
    parser.add_argument("--model", default=os.getenv("EVAL_LLM_MODEL") or EVAL_LLM_SETTINGS.get("model") or "")
    parser.add_argument("--api-key", default=os.getenv("EVAL_LLM_API_KEY") or EVAL_LLM_SETTINGS.get("api_key") or "dummy")
    parser.add_argument("--embedding-base-url", default=get_required_url("embedding_base_url"))
    parser.add_argument("--embedding-model", default=os.getenv("EVAL_EMBEDDING_MODEL") or EMBEDDING_SETTINGS.get("model") or "")
    parser.add_argument("--embedding-api-key", default=os.getenv("EVAL_EMBEDDING_API_KEY") or EMBEDDING_SETTINGS.get("api_key") or "dummy")
    parser.add_argument("--include-no-answer", action="store_true", help="Include answer_type=no_answer rows in RAGAS metrics.")
    parser.add_argument("--retrieval-mode", default="", help="Filter predictions by retrieval_mode: ask, vector, hybrid, hybrid_reranker.")
    parser.add_argument("--include-retrieval-only", action="store_true", help="Include rows without actual_answer. Useful only for retrieval-context experiments.")
    args = parser.parse_args()

    input_path = resolve_path(args.input, Path(args.input))
    output_dir = resolve_path(args.output_dir, input_path.parent) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        compat_status = ensure_ragas_import_compat()
        configure_openai_compatible_env(args)
        from datasets import Dataset
        from ragas import evaluate
        try:
            from ragas.metrics import (
                answer_correctness,
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )
        except ImportError:
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )
            answer_correctness = None
    except ImportError as exc:
        raise SystemExit(
            "RAGAS dependencies could not be imported.\n"
            f"{install_hint()}\n"
            f"Original import error: {exc}"
        ) from exc
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    rows = load_predictions(
        input_path,
        args.limit,
        include_no_answer=args.include_no_answer,
        retrieval_mode=args.retrieval_mode,
        include_retrieval_only=args.include_retrieval_only,
    )
    if not rows:
        summary = {
            "count": 0,
            "retrieval_mode": args.retrieval_mode,
            "include_no_answer": args.include_no_answer,
            "include_retrieval_only": args.include_retrieval_only,
            "message": "No rows matched the requested filters.",
        }
        write_jsonl(output_dir / "ragas_scores.jsonl", [])
        (output_dir / "ragas_scores.csv").write_text("", encoding="utf-8-sig")
        (output_dir / "ragas_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output_dir": str(output_dir), **summary}, ensure_ascii=False, indent=2))
        return
    records = []
    for row in rows:
        question = row.get("question", "")
        answer = row.get("actual_answer", "")
        contexts = json.loads(row.get("retrieved_contexts_json") or "[]")
        reference = row.get("expected_answer", "")
        records.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": reference,
            "reference": reference,
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts,
        })
    dataset = Dataset.from_list(records)
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    if answer_correctness is not None:
        metrics.append(answer_correctness)

    evaluator_models = build_openai_compatible_ragas_models(
        llm_base_url=args.base_url,
        llm_model=args.model,
        llm_api_key=args.api_key,
        embedding_base_url=args.embedding_base_url,
        embedding_model=args.embedding_model,
        embedding_api_key=args.embedding_api_key,
    )
    evaluate_kwargs: dict[str, Any] = {"metrics": metrics}
    evaluate_signature = inspect.signature(evaluate)
    if "llm" in evaluate_signature.parameters:
        evaluate_kwargs["llm"] = evaluator_models["llm"]
    if "embeddings" in evaluate_signature.parameters:
        evaluate_kwargs["embeddings"] = evaluator_models["embeddings"]
    result = evaluate(dataset, **evaluate_kwargs)
    try:
        df = result.to_pandas()
        score_rows = df.to_dict(orient="records")
    except Exception:
        score_rows = []
        for idx, record in enumerate(records):
            row = {"row_index": idx, "question": record["question"]}
            row.update(dict(result))
            score_rows.append(row)

    for idx, score in enumerate(score_rows):
        if idx < len(rows):
            score["question_id"] = rows[idx].get("question_id")
    write_jsonl(output_dir / "ragas_scores.jsonl", score_rows)
    write_csv(output_dir / "ragas_scores.csv", score_rows)
    summary = {
        "count": len(score_rows),
        "eval_llm": {
            "base_url": args.base_url,
            "model": args.model,
            "api_key_set": bool(args.api_key),
        },
        "eval_embedding": {
            "base_url": args.embedding_base_url,
            "model": args.embedding_model,
            "api_key_set": bool(args.embedding_api_key),
        },
        "include_no_answer": args.include_no_answer,
        "retrieval_mode": args.retrieval_mode,
        "include_retrieval_only": args.include_retrieval_only,
        "ragas_compat": {
            "vertexai_import_shim_installed": bool(compat_status.get("vertexai_import_shim_installed")),
            "dependency_status": compat_status.get("dependency_status", []),
            "wrapped_llm": bool(evaluator_models.get("wrapped_llm")),
            "wrapped_embeddings": bool(evaluator_models.get("wrapped_embeddings")),
        },
    }
    numeric_keys = sorted({key for row in score_rows for key, value in row.items() if isinstance(value, (int, float))})
    for key in numeric_keys:
        vals = [float(row[key]) for row in score_rows if isinstance(row.get(key), (int, float))]
        if vals:
            summary[f"avg_{key}"] = sum(vals) / len(vals)
    (output_dir / "ragas_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
