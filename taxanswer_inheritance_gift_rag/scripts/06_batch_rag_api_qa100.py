# 100問評価セットをRAG APIへバッチ推論し、CSV/JSONLで結果を保存します。
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from api.config import rag_ask_url

DEFAULT_INPUT = BASE_DIR / "eval" / "qa_100_questions.csv"
DEFAULT_CSV_OUTPUT = BASE_DIR / "eval" / "qa_100_rag_results.csv"
DEFAULT_JSONL_OUTPUT = BASE_DIR / "eval" / "qa_100_rag_results.jsonl"


INPUT_COLUMNS = [
    "question_id",
    "question",
    "expected_answer",
    "expected_sources",
    "corpus_ids",
    "reference_scope",
    "trap_type",
    "style",
    "category",
    "difficulty",
    "memo",
]

OUTPUT_COLUMNS = INPUT_COLUMNS + [
    "actual_answer",
    "answer_source",
    "cache_hit",
    "qa_cache_id",
    "cache_similarity",
    "source_count",
    "actual_sources",
    "debug_json",
    "elapsed_sec",
    "status",
    "error",
    "executed_at",
]


def resolve_path(value: str | None, default: Path) -> Path:
    if not value:
        return default
    path = Path(value)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def read_questions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [col for col in INPUT_COLUMNS if col not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"input csv missing columns: {', '.join(missing)}")
        return [dict(row) for row in reader]


def split_corpus_ids(value: str) -> list[str] | None:
    corpus_ids = [item.strip() for item in str(value or "").split("|") if item.strip()]
    return corpus_ids or None


def summarize_sources(sources: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    for source in sources:
        parts = [
            str(source.get("corpus_id") or ""),
            str(source.get("title") or ""),
            str(source.get("source_file") or ""),
            str(source.get("heading_path") or ""),
        ]
        rows.append("::".join(parts))
    return " || ".join(rows)


def append_csv(path: Path, row: dict[str, Any], write_header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "ok" and row.get("question_id"):
                completed.add(row["question_id"])
    return completed


def call_ask_api(
    api_url: str,
    row: dict[str, str],
    top_k: int | None,
    show_debug: bool,
    timeout_sec: int,
    session_prefix: str,
) -> tuple[dict[str, Any] | None, float, str]:
    payload: dict[str, Any] = {
        "question": row["question"],
        "corpus_ids": split_corpus_ids(row.get("corpus_ids", "")),
        "show_debug": show_debug,
        "session_id": f"{session_prefix}-{row['question_id']}",
    }
    if top_k:
        payload["top_k"] = top_k

    started = time.perf_counter()
    try:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            api_url,
            data=body,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "application/json",
            },
            method="POST",
        )
        with urlopen(request, timeout=timeout_sec) as response:
            response_body = response.read().decode("utf-8")
        elapsed = time.perf_counter() - started
        return json.loads(response_body), elapsed, ""
    except HTTPError as exc:
        elapsed = time.perf_counter() - started
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            detail = ""
        return None, elapsed, f"HTTP {exc.code}: {detail or exc.reason}"
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        elapsed = time.perf_counter() - started
        return None, elapsed, str(exc)


def build_output_row(
    input_row: dict[str, str],
    data: dict[str, Any] | None,
    elapsed: float,
    error: str,
) -> dict[str, Any]:
    executed_at = datetime.now(timezone.utc).isoformat()
    output: dict[str, Any] = {col: input_row.get(col, "") for col in INPUT_COLUMNS}
    output["elapsed_sec"] = round(elapsed, 3)
    output["executed_at"] = executed_at

    if data is None:
        output.update(
            {
                "actual_answer": "",
                "answer_source": "",
                "cache_hit": "",
                "qa_cache_id": "",
                "cache_similarity": "",
                "source_count": 0,
                "actual_sources": "",
                "debug_json": "",
                "status": "error",
                "error": error,
            }
        )
        return output

    sources = data.get("sources") or []
    output.update(
        {
            "actual_answer": data.get("answer", ""),
            "answer_source": data.get("answer_source", ""),
            "cache_hit": data.get("cache_hit", False),
            "qa_cache_id": data.get("qa_cache_id", ""),
            "cache_similarity": data.get("cache_similarity", ""),
            "source_count": len(sources),
            "actual_sources": summarize_sources(sources),
            "debug_json": json.dumps(data.get("debug"), ensure_ascii=False) if data.get("debug") else "",
            "status": "ok",
            "error": "",
        }
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 100-question QA set against the local RAG API.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input QA CSV. Relative paths are project-root relative.")
    parser.add_argument("--output-csv", default=str(DEFAULT_CSV_OUTPUT), help="Output result CSV.")
    parser.add_argument("--output-jsonl", default=str(DEFAULT_JSONL_OUTPUT), help="Output raw JSONL.")
    parser.add_argument("--api", default=rag_ask_url(), help="RAG /ask endpoint URL.")
    parser.add_argument("--top-k", type=int, default=None, help="Override RAG top_k.")
    parser.add_argument("--timeout-sec", type=int, default=240, help="Request timeout seconds.")
    parser.add_argument("--sleep-sec", type=float, default=0.0, help="Sleep seconds between requests.")
    parser.add_argument("--limit", type=int, default=0, help="Only run the first N selected questions.")
    parser.add_argument("--start", type=int, default=1, help="1-based start row.")
    parser.add_argument("--resume", action="store_true", help="Skip question_ids already marked ok in output CSV.")
    parser.add_argument("--show-debug", action="store_true", help="Request debug info from API.")
    parser.add_argument("--session-prefix", default="qa100", help="Session id prefix.")
    args = parser.parse_args()

    input_path = resolve_path(args.input, DEFAULT_INPUT)
    output_csv = resolve_path(args.output_csv, DEFAULT_CSV_OUTPUT)
    output_jsonl = resolve_path(args.output_jsonl, DEFAULT_JSONL_OUTPUT)

    questions = read_questions(input_path)
    if args.start > 1:
        questions = questions[args.start - 1 :]
    if args.limit > 0:
        questions = questions[: args.limit]

    completed_ids = load_completed_ids(output_csv) if args.resume else set()
    write_header = not output_csv.exists() or not args.resume
    if output_csv.exists() and not args.resume:
        output_csv.unlink()
    if output_jsonl.exists() and not args.resume:
        output_jsonl.unlink()

    total = len(questions)
    ok_count = 0
    error_count = 0
    skipped_count = 0

    for idx, question in enumerate(questions, start=1):
        question_id = question["question_id"]
        if question_id in completed_ids:
            skipped_count += 1
            print(f"[{idx}/{total}] skip {question_id}")
            continue

        data, elapsed, error = call_ask_api(
            api_url=args.api,
            row=question,
            top_k=args.top_k,
            show_debug=args.show_debug,
            timeout_sec=args.timeout_sec,
            session_prefix=args.session_prefix,
        )
        output = build_output_row(question, data, elapsed, error)
        append_csv(output_csv, output, write_header=write_header)
        append_jsonl(output_jsonl, output)
        write_header = False

        if output["status"] == "ok":
            ok_count += 1
            print(f"[{idx}/{total}] ok {question_id} sources={output['source_count']} elapsed={output['elapsed_sec']}s")
        else:
            error_count += 1
            print(f"[{idx}/{total}] error {question_id}: {error}")

        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    print(
        f"done ok={ok_count} error={error_count} skipped={skipped_count} "
        f"csv={output_csv} jsonl={output_jsonl}"
    )


if __name__ == "__main__":
    main()
