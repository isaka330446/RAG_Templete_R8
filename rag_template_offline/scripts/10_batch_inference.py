from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from eval_common import (
    DEFAULT_EVAL_CSV,
    DEFAULT_EVAL_JSONL,
    append_csv,
    append_jsonl,
    contexts_from_sources,
    create_run_dir,
    expected_source_hit,
    read_eval_dataset,
    read_jsonl,
    resolve_path,
    source_ref,
    split_multi,
    summarize_predictions,
    write_csv,
)
from api.config import rag_ask_url


def call_ask(api_url: str, row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "question": row.get("question") or "",
        "corpus_ids": split_multi(row.get("corpus_ids")) or None,
        "show_debug": bool(args.show_debug),
        "session_id": f"eval-{args.run_id or 'batch'}-{row.get('question_id')}",
    }
    if args.top_k:
        payload["top_k"] = args.top_k
    started = time.perf_counter()
    last_error = ""
    for attempt in range(max(1, args.retries + 1)):
        try:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            req = Request(api_url, data=body, headers={"Content-Type": "application/json; charset=utf-8"}, method="POST")
            with urlopen(req, timeout=args.timeout_sec) as res:
                data = json.loads(res.read().decode("utf-8"))
            elapsed = time.perf_counter() - started
            sources = data.get("sources") or []
            return build_prediction(row, data, elapsed, "")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code}: {detail or exc.reason}"
        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        if attempt < args.retries:
            time.sleep(min(2.0 * (attempt + 1), 10.0))
    return build_prediction(row, None, time.perf_counter() - started, last_error)


def build_prediction(row: dict[str, Any], data: dict[str, Any] | None, elapsed: float, error: str) -> dict[str, Any]:
    base = dict(row)
    base["question_id"] = str(base.get("question_id") or "")
    base["question"] = str(base.get("question") or "")
    base["expected_answer"] = str(base.get("expected_answer") or "")
    base["expected_sources"] = str(base.get("expected_sources") or "")
    base["elapsed_sec"] = round(elapsed, 3)
    base["executed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if data is None:
        base.update({
            "actual_answer": "",
            "sources_json": "[]",
            "retrieved_contexts_json": "[]",
            "source_refs_json": "[]",
            "expected_source_hit": False,
            "answer_source": "",
            "cache_hit": False,
            "qa_cache_id": "",
            "cache_similarity": "",
            "debug_json": "{}",
            "status": "error",
            "error": error,
        })
        return base
    sources = data.get("sources") or []
    source_refs = [source_ref(source) for source in sources]
    base.update({
        "actual_answer": data.get("answer", ""),
        "sources_json": json.dumps(sources, ensure_ascii=False),
        "retrieved_contexts_json": json.dumps(contexts_from_sources(sources), ensure_ascii=False),
        "source_refs_json": json.dumps(source_refs, ensure_ascii=False),
        "expected_source_hit": expected_source_hit(base.get("expected_sources"), sources),
        "answer_source": data.get("answer_source", ""),
        "cache_hit": bool(data.get("cache_hit", False)),
        "qa_cache_id": data.get("qa_cache_id", ""),
        "cache_similarity": data.get("cache_similarity", ""),
        "debug_json": json.dumps(data.get("debug") or {}, ensure_ascii=False),
        "status": "ok",
        "error": "",
    })
    return base


def completed_ids(path: Path) -> set[str]:
    return {str(row.get("question_id")) for row in read_jsonl(path) if row.get("status") == "ok"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch inference against the RAG /ask API.")
    default_input = DEFAULT_EVAL_JSONL if DEFAULT_EVAL_JSONL.exists() else DEFAULT_EVAL_CSV
    parser.add_argument("--input", default=str(default_input))
    parser.add_argument("--api", default=rag_ask_url())
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--show-debug", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout-sec", type=int, default=240)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--disable-search-tags", action="store_true", help="Mark this run as a SearchTag-disabled A/B run. Start the API with SEARCH_TAGS_ENABLED=false.")
    args = parser.parse_args()

    input_path = resolve_path(args.input, default_input)
    run_dir = create_run_dir(args.output_dir or None, args.run_id or None)
    predictions_jsonl = run_dir / "predictions.jsonl"
    predictions_csv = run_dir / "predictions.csv"
    manifest_path = run_dir / "run_manifest.json"

    rows = read_eval_dataset(input_path)
    rows = rows[max(0, args.start - 1):]
    if args.limit:
        rows = rows[: args.limit]
    if args.resume:
        done = completed_ids(predictions_jsonl)
        rows = [row for row in rows if str(row.get("question_id")) not in done]

    manifest = {
        "run_id": args.run_id or run_dir.name,
        "input": str(input_path),
        "api": args.api,
        "output_dir": str(run_dir),
        "limit": args.limit,
        "start": args.start,
        "top_k": args.top_k,
        "show_debug": args.show_debug,
        "workers": args.workers,
        "dry_run": args.dry_run,
        "disable_search_tags": args.disable_search_tags,
        "search_tags_note": "Start the RAG API with SEARCH_TAGS_ENABLED=false for an actual disabled run." if args.disable_search_tags else "",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"run_dir": str(run_dir), "selected": len(rows), "manifest": manifest}, ensure_ascii=False, indent=2))
        return

    results: list[dict[str, Any]] = []
    write_header = not predictions_csv.exists() or not args.resume
    if not args.resume:
        predictions_jsonl.write_text("", encoding="utf-8")
        predictions_csv.write_text("", encoding="utf-8-sig")

    if args.workers <= 1:
        for row in rows:
            result = call_ask(args.api, row, args)
            results.append(result)
            append_jsonl(predictions_jsonl, result)
            append_csv(predictions_csv, result, write_header=write_header)
            write_header = False
            print(f"{result.get('question_id')} {result.get('status')} {result.get('elapsed_sec')}s")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(call_ask, args.api, row, args) for row in rows]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                append_jsonl(predictions_jsonl, result)
                append_csv(predictions_csv, result, write_header=write_header)
                write_header = False
                print(f"{result.get('question_id')} {result.get('status')} {result.get('elapsed_sec')}s")

    all_rows = read_jsonl(predictions_jsonl)
    summary = summarize_predictions(all_rows)
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(predictions_csv, all_rows)
    print(json.dumps({"run_dir": str(run_dir), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
