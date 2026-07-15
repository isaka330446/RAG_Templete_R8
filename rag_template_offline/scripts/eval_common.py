from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_CSV = BASE_DIR / "eval" / "qa_100_questions.csv"
DEFAULT_EVAL_JSONL = BASE_DIR / "eval" / "golden" / "rag_golden_dataset.jsonl"
DEFAULT_RUNS_DIR = BASE_DIR / "eval" / "runs"


def utc_timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def resolve_path(value: str | None, default: Path) -> Path:
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else BASE_DIR / path


def split_multi(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [item.strip() for item in re.split(r"[|;,]", text) if item.strip()]


def read_eval_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
    for idx, row in enumerate(rows, start=1):
        row.setdefault("question_id", f"Q{idx:04d}")
        row.setdefault("question", row.get("input") or row.get("query") or "")
        row.setdefault("expected_answer", row.get("reference") or row.get("ground_truth") or "")
        row.setdefault("expected_sources", row.get("expected_source") or "")
        row.setdefault("corpus_ids", "")
    return rows


def json_text(value: Any) -> str:
    return json.dumps(value if value is not None else [], ensure_ascii=False)


def expected_source_candidates_from_json(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return value
    candidates: list[str] = []
    for source in value if isinstance(value, list) else [value]:
        if not isinstance(source, dict):
            text = str(source or "").strip()
            if text:
                candidates.append(text)
            continue
        source_path = str(source.get("source_path") or "").strip()
        if source_path:
            candidates.append(source_path)
            path = Path(source_path.replace("\\", "/"))
            candidates.extend([path.name, path.stem])
        for key in ("heading", "evidence_quote", "source_url", "document_code", "document_id"):
            text = str(source.get(key) or "").strip()
            if text:
                candidates.append(text)
    seen: set[str] = set()
    out: list[str] = []
    for item in candidates:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return " | ".join(out)


def normalize_eval_row(row: dict[str, Any], idx: int) -> dict[str, Any]:
    expected_sources = row.get("expected_sources", "")
    out = dict(row)
    out["question_id"] = str(row.get("id") or row.get("question_id") or f"Q{idx:04d}")
    out["question"] = str(row.get("question") or row.get("input") or row.get("query") or "")
    out["expected_answer"] = str(row.get("expected_answer") or row.get("reference") or row.get("ground_truth") or "")
    out["expected_sources"] = expected_source_candidates_from_json(expected_sources) or str(row.get("expected_source") or "")
    out["corpus_ids"] = str(row.get("corpus_ids") or "")
    out["answer_type"] = str(row.get("answer_type") or "")
    out["short_answer"] = str(row.get("short_answer") or "")
    out["difficulty"] = str(row.get("difficulty") or "")
    out["question_scope"] = str(row.get("question_scope") or "")
    out["expected_sources_json"] = json_text(expected_sources)
    out["must_include_json"] = json_text(row.get("must_include"))
    out["must_not_include_json"] = json_text(row.get("must_not_include"))
    out["retrieval_keywords_json"] = json_text(row.get("retrieval_keywords"))
    out["grading_rubric_json"] = json_text(row.get("grading_rubric"))
    out["notes"] = str(row.get("notes") or "")
    out["is_no_answer"] = out["answer_type"] == "no_answer"
    return out


def read_eval_jsonl(path: Path) -> list[dict[str, Any]]:
    return [normalize_eval_row(row, idx) for idx, row in enumerate(read_jsonl(path), start=1)]


def read_eval_dataset(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_eval_jsonl(path)
    return [normalize_eval_row(row, idx) for idx, row in enumerate(read_eval_csv(path), start=1)]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row: dict[str, Any], *, write_header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def source_ref(source: dict[str, Any]) -> str:
    parts = [
        source.get("corpus_id"),
        source.get("document_code"),
        source.get("title"),
        source.get("source_url"),
        source.get("source_file"),
        source.get("heading_path"),
        source.get("child_id"),
    ]
    return " ".join(str(part) for part in parts if part)


def expected_source_hit(expected_sources: Any, sources: list[dict[str, Any]]) -> bool:
    expected = split_multi(expected_sources)
    if not expected:
        return False
    haystack = "\n".join(source_ref(source) for source in sources)
    return any(item and item in haystack for item in expected)


def contexts_from_sources(sources: list[dict[str, Any]]) -> list[str]:
    contexts: list[str] = []
    for source in sources:
        text = source.get("child_text") or source.get("parent_text") or ""
        if text:
            contexts.append(str(text))
    return contexts


def create_run_dir(output_dir: str | None, run_id: str | None, suffix: str = "batch") -> Path:
    if output_dir:
        path = resolve_path(output_dir, DEFAULT_RUNS_DIR)
    else:
        path = DEFAULT_RUNS_DIR / f"{run_id or utc_timestamp()}_{suffix}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def summarize_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    errors = total - len(ok_rows)
    cache_hits = sum(1 for row in ok_rows if str(row.get("cache_hit")).lower() in {"1", "true", "yes"})
    expected_hits = sum(1 for row in ok_rows if row.get("expected_source_hit") is True or str(row.get("expected_source_hit")).lower() == "true")
    latencies = [float(row.get("elapsed_sec") or 0.0) for row in ok_rows]
    return {
        "total": total,
        "ok": len(ok_rows),
        "errors": errors,
        "expected_source_hit_rate": expected_hits / len(ok_rows) if ok_rows else 0.0,
        "cache_hit_rate": cache_hits / len(ok_rows) if ok_rows else 0.0,
        "avg_elapsed_sec": sum(latencies) / len(latencies) if latencies else 0.0,
    }
