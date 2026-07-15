# 管理者レビュー済みQA seedをEmbeddingし、SQLiteの承認済みQAキャッシュへ登録します。
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Iterable


BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

DEFAULT_INPUT = BASE_DIR / "data" / "approved_qa_seed" / "approved_qa_seed_100.jsonl"
APPROVED_STATUS = "approved_for_import"
CANDIDATE_STATUS = "candidate_needs_admin_review"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            item["_line_no"] = line_no
            items.append(item)
    return items


def iter_importable(
    items: Iterable[dict[str, Any]],
    *,
    accept_candidates: bool,
    limit: int | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    stats = {
        "loaded": 0,
        "missing_evidence": 0,
        "skipped_status": 0,
        "selected": 0,
    }
    selected: list[dict[str, Any]] = []
    for item in items:
        stats["loaded"] += 1
        if not item.get("evidence"):
            stats["missing_evidence"] += 1
            continue
        status = str(item.get("approval_status") or "")
        if status != APPROVED_STATUS and not (accept_candidates and status == CANDIDATE_STATUS):
            stats["skipped_status"] += 1
            continue
        selected.append(item)
        stats["selected"] += 1
        if limit is not None and len(selected) >= limit:
            break
    return selected, stats


def already_registered(sqlite_path: Path, question: str, corpus_version: str, index_version: str) -> bool:
    if not sqlite_path.exists():
        return False
    with sqlite3.connect(sqlite_path) as conn:
        row = conn.execute(
            """
            SELECT id
            FROM approved_qa
            WHERE status = 'approved'
              AND question = ?
              AND corpus_version = ?
              AND index_version = ?
            LIMIT 1
            """,
            (question, corpus_version, index_version),
        ).fetchone()
    return row is not None


def batch(items: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    size = max(1, size)
    for start in range(0, len(items), size):
        yield items[start : start + size]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Import reviewed approved-QA seed rows into the local SQLite answer cache. "
            "Dry-run is the default. Use --apply only after admin review."
        )
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Seed JSONL path")
    parser.add_argument("--apply", action="store_true", help="Write approved rows to SQLite")
    parser.add_argument(
        "--accept-candidates",
        action="store_true",
        help="Treat candidate_needs_admin_review rows as approved for this import",
    )
    parser.add_argument("--approved-by", default="", help="Admin/user name stored in approved_by")
    parser.add_argument("--corpus-version", default="", help="Override corpus_version for imported rows")
    parser.add_argument("--index-version", default="", help="Override index_version for imported rows")
    parser.add_argument("--limit", type=int, default=None, help="Maximum rows to import")
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size")
    parser.add_argument("--no-skip-duplicates", action="store_true", help="Register even if the same question/version exists")
    parser.add_argument("--generate-aliases", action="store_true", help="Generate LLM paraphrase aliases after QA import")
    parser.add_argument("--max-aliases-per-qa", type=int, default=8, help="Maximum generated aliases per QA")
    parser.add_argument("--alias-dry-run", action="store_true", help="Generate alias candidates but do not save them")
    args = parser.parse_args()

    if args.apply and not args.approved_by.strip():
        parser.error("--approved-by is required with --apply")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")

    items = load_jsonl(Path(args.input))
    selected, stats = iter_importable(items, accept_candidates=args.accept_candidates, limit=args.limit)

    print(f"loaded={stats['loaded']}")
    print(f"selected={stats['selected']}")
    print(f"missing_evidence={stats['missing_evidence']}")
    print(f"skipped_status={stats['skipped_status']}")
    if not args.apply:
        print("dry_run=true")
        print(f"importable_status={APPROVED_STATUS}")
        print("Use --apply --approved-by <name> after review.")
        print("If you intentionally reviewed the candidate seed as-is, add --accept-candidates.")
        return

    from api.answer_cache import (
        AnswerCacheStore,
        build_alias_generation_prompt,
        detect_alias_risk_flags,
        filter_alias_candidates,
        parse_alias_generation_response,
    )
    from api.config import load_settings
    from api.llm_client import OpenAICompatibleEmbedding, OpenAICompatibleLLM

    store = AnswerCacheStore()
    embedder = OpenAICompatibleEmbedding()
    alias_llm_settings = load_settings().get("alias_llm", {})
    llm = OpenAICompatibleLLM(
        api_key=alias_llm_settings.get("api_key"),
        model=alias_llm_settings.get("model"),
        url_key="alias_llm_base_url",
        temperature=float(alias_llm_settings.get("temperature", store.alias_generation_temperature)),
        max_tokens=int(alias_llm_settings.get("max_tokens", store.alias_generation_max_tokens)),
    )
    if alias_llm_settings.get("timeout_sec"):
        llm.timeout_sec = int(alias_llm_settings.get("timeout_sec"))

    inserted = 0
    duplicates = 0
    alias_created = 0
    alias_errors = 0
    for group in batch(selected, args.batch_size):
        pending: list[dict[str, Any]] = []
        for item in group:
            corpus_version = args.corpus_version.strip() or str(item.get("corpus_version") or store.corpus_version)
            index_version = args.index_version.strip() or str(item.get("index_version") or store.index_version)
            item["_import_corpus_version"] = corpus_version
            item["_import_index_version"] = index_version
            if not args.no_skip_duplicates and already_registered(
                store.sqlite_path,
                str(item.get("question") or ""),
                corpus_version,
                index_version,
            ):
                duplicates += 1
                continue
            pending.append(item)

        if not pending:
            continue

        embeddings = embedder.embed([str(item["question"]) for item in pending])
        for item, embedding in zip(pending, embeddings):
            qa_id = store.create_approved_qa(
                question=str(item["question"]),
                answer=str(item["answer"]),
                question_embedding=embedding,
                evidence=list(item.get("evidence") or []),
                corpus_version=str(item["_import_corpus_version"]),
                index_version=str(item["_import_index_version"]),
                approved_by=args.approved_by.strip(),
                memo=str(item.get("memo") or ""),
            )
            inserted += 1
            if args.generate_aliases:
                try:
                    evidence = list(item.get("evidence") or [])
                    prompt = build_alias_generation_prompt(
                        str(item["question"]),
                        str(item["answer"]),
                        evidence,
                        args.max_aliases_per_qa,
                    )
                    raw = llm.chat([
                        {"role": "system", "content": "Generate safe approved-QA aliases as JSON only."},
                        {"role": "user", "content": prompt},
                    ])
                    candidates = filter_alias_candidates(
                        parse_alias_generation_response(raw),
                        question=str(item["question"]),
                        max_aliases=args.max_aliases_per_qa,
                    )
                    if args.alias_dry_run:
                        print(f"alias_dry_run qa_id={qa_id} candidates={len(candidates)}")
                        continue
                    alias_embeddings = embedder.embed(candidates) if candidates else []
                    result = store.add_aliases(
                        qa_id,
                        [
                            {
                                "alias_text": alias,
                                "alias_type": "llm_paraphrase",
                                "status": "disabled" if detect_alias_risk_flags(str(item["question"]), alias, evidence) else store.generated_alias_default_status,
                                "risk_flags": detect_alias_risk_flags(str(item["question"]), alias, evidence),
                                "embedding": alias_embedding,
                                "generator_model": llm.model,
                                "memo": "generated by seed import",
                            }
                            for alias, alias_embedding in zip(candidates, alias_embeddings)
                        ],
                    )
                    alias_created += int(result.get("created_count") or 0)
                except Exception as exc:
                    alias_errors += 1
                    print(f"alias_error qa_id={qa_id} error={exc}")

    print(f"inserted={inserted}")
    print(f"duplicates_skipped={duplicates}")
    print(f"alias_created={alias_created}")
    print(f"alias_errors={alias_errors}")
    print(f"sqlite_path={store.sqlite_path}")


if __name__ == "__main__":
    main()
