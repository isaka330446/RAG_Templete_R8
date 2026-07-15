# ローカルLLMへ並列リクエストを投げ、子チャンクごとのSearchTagを生成します。
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv() -> None:
        return None


sys.path.append(str(Path(__file__).resolve().parent.parent))

from api.config import get_required_url, load_settings


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
CHILD_IN = BASE_DIR / "chunks" / "child_chunks.jsonl"
CHILD_OUT = BASE_DIR / "chunks" / "child_chunks_with_tags.jsonl"
PARTIAL_OUT = BASE_DIR / "chunks" / "child_chunks_with_tags.partial.jsonl"

SETTINGS = load_settings().get("llm", {})
LLM_ENDPOINT_BASE = get_required_url("llm_base_url")
LLM_API_KEY = os.getenv("LLM_API_KEY") or SETTINGS.get("api_key", "dummy")
LLM_MODEL = os.getenv("LLM_MODEL") or SETTINGS.get("model", "local-model")
LLM_TIMEOUT_SEC = int(SETTINGS.get("timeout_sec", 180))

DEFAULT_WORKERS = int(os.getenv("SEARCH_TAG_WORKERS", "8"))
DEFAULT_CHECKPOINT_EVERY = int(os.getenv("SEARCH_TAG_CHECKPOINT_EVERY", "25"))


TAG_PROMPT = """あなたはRAG検索品質を改善するためのSearchTag作成担当です。

以下の子チャンクに対して、検索に役立つ短いタグを日本語中心で10〜20個作成してください。

条件:
- 文書に明示されている概念・手続・様式・条件・対象者を優先する。
- ユーザーが質問で使いそうな言い換えも含める。
- 本文にない制度を勝手に追加しない。
- 出力はJSON配列のみ。
- 例: ["申請手続", "承認", "サンプル申請書"]

# 子チャンク
{chunk}
"""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def existing_by_child_id(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    return {str(row.get("child_id")): row for row in rows if row.get("child_id")}


def is_completed_tag_row(row: dict[str, Any]) -> bool:
    return "search_tags" in row and bool(row.get("search_text")) and not row.get("tag_error")


def clean_llm_json(content: str) -> str:
    content = content.strip()
    content = re.sub(r"^```json\s*", "", content)
    content = re.sub(r"^```\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    return content.strip()


def normalize_tags(value: Any, *, limit: int = 40) -> list[str]:
    if not isinstance(value, list):
        return []
    tags: list[str] = []
    seen: set[str] = set()
    for item in value:
        tag = re.sub(r"\s+", " ", str(item)).strip()
        if not tag or tag in seen:
            continue
        tags.append(tag)
        seen.add(tag)
        if len(tags) >= limit:
            break
    return tags


def call_llm(chunk: str, *, max_chunk_chars: int) -> list[str]:
    import requests

    url = f"{LLM_ENDPOINT_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {LLM_API_KEY}"}
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "JSONだけを返してください。"},
            {"role": "user", "content": TAG_PROMPT.format(chunk=chunk[:max_chunk_chars])},
        ],
        "temperature": 0.1,
        "max_tokens": 512,
    }
    res = requests.post(url, json=payload, headers=headers, timeout=LLM_TIMEOUT_SEC)
    res.raise_for_status()
    content = clean_llm_json(res.json()["choices"][0]["message"]["content"])

    try:
        return normalize_tags(json.loads(content))
    except Exception:
        return []


def build_search_text(row: dict[str, Any], tags: list[str]) -> str:
    return "\n".join(
        [
            str(row.get("title", "")),
            str(row.get("heading_path", "")),
            str(row.get("text", "")),
            " ".join(tags),
        ]
    ).strip()


def process(row: dict[str, Any], *, max_chunk_chars: int) -> dict[str, Any]:
    output = dict(row)
    tags = call_llm(str(output.get("text", "")), max_chunk_chars=max_chunk_chars)
    output["search_tags"] = tags
    output["search_text"] = build_search_text(output, tags)
    output.pop("tag_error", None)
    return output


def fallback_row(row: dict[str, Any], error: Exception) -> dict[str, Any]:
    output = dict(row)
    output["search_tags"] = []
    output["search_text"] = build_search_text(output, [])
    output["tag_error"] = str(error)
    return output


def write_jsonl(path: Path, source_rows: list[dict[str, Any]], rows_by_id: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for source_row in source_rows:
            child_id = str(source_row.get("child_id") or "")
            row = rows_by_id.get(child_id, source_row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_resume_rows(output_path: Path, partial_path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in [output_path, partial_path]:
        for child_id, row in existing_by_child_id(path).items():
            if is_completed_tag_row(row):
                rows[child_id] = row
    return rows


def run_parallel(
    *,
    source_rows: list[dict[str, Any]],
    output_path: Path,
    partial_path: Path,
    workers: int,
    checkpoint_every: int,
    max_chunk_chars: int,
    resume: bool,
) -> tuple[int, int, int]:
    rows_by_id = load_resume_rows(output_path, partial_path) if resume else {}
    pending = [row for row in source_rows if str(row.get("child_id") or "") not in rows_by_id]
    reused = len(source_rows) - len(pending)
    completed = 0
    failed = 0

    print(f"workers={workers} pending={len(pending)} reused={reused}")

    row_iter = iter(pending)
    futures = {}

    def submit_next(executor: ThreadPoolExecutor) -> bool:
        try:
            row = next(row_iter)
        except StopIteration:
            return False
        future = executor.submit(process, row, max_chunk_chars=max_chunk_chars)
        futures[future] = row
        return True

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for _ in range(workers):
            if not submit_next(executor):
                break

        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                source_row = futures.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = fallback_row(source_row, exc)
                    failed += 1
                rows_by_id[str(result.get("child_id") or "")] = result
                completed += 1

                if checkpoint_every > 0 and completed % checkpoint_every == 0:
                    write_jsonl(partial_path, source_rows, rows_by_id)
                    print(f"checkpoint completed={completed} failed={failed} -> {partial_path}")

                submit_next(executor)

    write_jsonl(output_path, source_rows, rows_by_id)
    if partial_path.exists():
        partial_path.unlink()
    return completed, reused, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SearchTags for child chunks with a local OpenAI-compatible LLM.")
    parser.add_argument("--input", default=str(CHILD_IN), help="Input child_chunks.jsonl")
    parser.add_argument("--output", default=str(CHILD_OUT), help="Output child_chunks_with_tags.jsonl")
    parser.add_argument("--partial-output", default=str(PARTIAL_OUT), help="Checkpoint JSONL path")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Concurrent LLM requests. Default: 8")
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY, help="Write checkpoint every N completed rows")
    parser.add_argument("--max-chunk-chars", type=int, default=5000, help="Maximum child chunk characters sent to the LLM")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N source rows")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing output/partial rows and regenerate all selected rows")
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.checkpoint_every < 0:
        parser.error("--checkpoint-every must be >= 0")
    if args.max_chunk_chars < 200:
        parser.error("--max-chunk-chars must be >= 200")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")

    rows = load_jsonl(Path(args.input))
    if args.limit is not None:
        rows = rows[: args.limit]

    completed, reused, failed = run_parallel(
        source_rows=rows,
        output_path=Path(args.output),
        partial_path=Path(args.partial_output),
        workers=args.workers,
        checkpoint_every=args.checkpoint_every,
        max_chunk_chars=args.max_chunk_chars,
        resume=not args.no_resume,
    )
    print(f"wrote: {args.output} rows={len(rows)} completed={completed} reused={reused} failed={failed}")


if __name__ == "__main__":
    main()
