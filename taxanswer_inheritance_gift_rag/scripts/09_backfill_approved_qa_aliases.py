from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))


def backup_sqlite(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"SQLite file not found: {path}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_suffix(path.suffix + f".bak_{timestamp}")
    shutil.copy2(path, backup_path)
    return backup_path


def inspect_sqlite_readonly(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "schema_migrated": False, "message": f"DB not found: {path}"}
    uri = f"file:{path.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        tables = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }
        result: dict[str, Any] = {
            "exists": True,
            "schema_migrated": "approved_qa_aliases" in tables,
            "tables": sorted(tables),
        }
        if "approved_qa" in tables:
            result["approved_qa_count"] = int(conn.execute("SELECT COUNT(*) FROM approved_qa").fetchone()[0])
        if "approved_qa_aliases" in tables:
            result["alias_count"] = int(conn.execute("SELECT COUNT(*) FROM approved_qa_aliases").fetchone()[0])
            result["active_alias_count"] = int(
                conn.execute("SELECT COUNT(*) FROM approved_qa_aliases WHERE status = 'active'").fetchone()[0]
            )
        return result


def build_alias_llm(store: Any) -> tuple[Any, dict[str, Any]]:
    from api.config import load_settings
    from api.llm_client import OpenAICompatibleLLM

    settings = load_settings()
    alias_llm_settings = settings.get("alias_llm", {}) or {}
    alias_configured = bool(alias_llm_settings.get("model"))
    llm = OpenAICompatibleLLM(
        api_key=alias_llm_settings.get("api_key"),
        model=alias_llm_settings.get("model"),
        url_key="alias_llm_base_url",
        temperature=float(alias_llm_settings.get("temperature", store.alias_generation_temperature)),
        max_tokens=int(alias_llm_settings.get("max_tokens", store.alias_generation_max_tokens)),
    )
    if alias_llm_settings.get("timeout_sec") is not None:
        llm.timeout_sec = int(alias_llm_settings["timeout_sec"])
    return llm, {
        "base_url": llm.base_url,
        "model": llm.model,
        "alias_llm_configured": alias_configured,
        "fallback_to_default_llm": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill approved QA aliases into logs/answer_cache.sqlite.")
    parser.add_argument("--check", action="store_true", help="Read-only DB/schema check. No DB file, DDL, or DML is created.")
    parser.add_argument("--migrate-schema", action="store_true", help="Create/alter alias schema explicitly.")
    parser.add_argument("--apply", action="store_true", help="Write data changes. Default is dry-run.")
    parser.add_argument("--dry-run", action="store_true", help="Preview only. This is the default when --apply is not set.")
    parser.add_argument("--backup", action="store_true", help="Create timestamped SQLite backup before write operations.")
    parser.add_argument("--ensure-original", action="store_true", help="Create missing original aliases.")
    parser.add_argument(
        "--generate-llm",
        "--generate-llm-aliases",
        dest="generate_llm_aliases",
        action="store_true",
        help="Generate LLM paraphrase aliases.",
    )
    parser.add_argument("--only-without-llm-aliases", action="store_true", help="Only target QA without active LLM aliases.")
    parser.add_argument("--limit", type=int, default=50, help="Maximum QA rows to process.")
    parser.add_argument("--max-aliases-per-qa", type=int, default=8, help="Maximum generated aliases per QA.")
    parser.add_argument("--status", choices=["active", "disabled"], default="", help="Status for generated LLM aliases.")
    parser.add_argument("--corpus-version", default="", help="Filter corpus_version.")
    parser.add_argument("--index-version", default="", help="Filter index_version.")
    args = parser.parse_args()

    from api.config import load_settings, project_path

    cache_settings = load_settings().get("answer_cache", {})
    sqlite_path = project_path(cache_settings.get("sqlite_path", "logs/answer_cache.sqlite"))

    if args.check:
        print(json.dumps({"sqlite_path": str(sqlite_path), **inspect_sqlite_readonly(sqlite_path)}, ensure_ascii=False, indent=2))
        return

    is_write_operation = args.apply or args.migrate_schema
    if is_write_operation and args.backup:
        if sqlite_path.exists():
            backup_path = backup_sqlite(sqlite_path)
        elif args.apply:
            raise FileNotFoundError(f"SQLite file not found: {sqlite_path}")
        else:
            backup_path = None
    else:
        backup_path = None
    if args.apply and not args.backup:
        parser.error("--backup is required with --apply to protect the operational SQLite file")
    if args.migrate_schema:
        from api.answer_cache import AnswerCacheStore

        store = AnswerCacheStore(auto_migrate=True)
        print(json.dumps({
            "dry_run": False,
            "migrated_schema": True,
            "sqlite_path": str(store.sqlite_path),
            "backup_path": str(backup_path) if backup_path else None,
            "alias_index": store.alias_index_status(),
        }, ensure_ascii=False, indent=2))
        return

    if not args.ensure_original and not args.generate_llm_aliases:
        parser.error("Specify --ensure-original and/or --generate-llm")

    dry_run = not args.apply
    readonly_status = inspect_sqlite_readonly(sqlite_path)
    if dry_run and not readonly_status.get("exists"):
        print(json.dumps({"dry_run": True, "sqlite_path": str(sqlite_path), **readonly_status}, ensure_ascii=False, indent=2))
        return
    if dry_run and not readonly_status.get("schema_migrated"):
        print(json.dumps({
            "dry_run": True,
            "sqlite_path": str(sqlite_path),
            **readonly_status,
            "next_step": "Run --migrate-schema --backup before data backfill.",
        }, ensure_ascii=False, indent=2))
        return

    from api.answer_cache import (
        AnswerCacheStore,
        build_alias_generation_prompt,
        detect_alias_risk_flags,
        filter_alias_candidates,
        normalize_qa_text,
        parse_alias_generation_response,
    )

    store = AnswerCacheStore(auto_migrate=bool(args.apply))

    result = store.backfill_aliases(
        ensure_original=args.ensure_original,
        generate_llm_aliases=args.generate_llm_aliases,
        only_without_llm_aliases=args.only_without_llm_aliases,
        limit=args.limit,
        dry_run=dry_run,
        corpus_version=args.corpus_version.strip() or None,
        index_version=args.index_version.strip() or None,
    )

    generated = 0
    created = 0
    skipped = 0
    errors = 0
    alias_llm_meta: dict[str, Any] | None = None
    if args.generate_llm_aliases:
        from api.llm_client import OpenAICompatibleEmbedding

        embedder = OpenAICompatibleEmbedding()
        llm, alias_llm_meta = build_alias_llm(store)
        for target in result.get("items", []):
            qa_id = int(target["qa_id"])
            qa = store.get_approved_qa(qa_id)
            if not qa:
                skipped += 1
                continue
            try:
                existing_normalized = {
                    str(alias.get("normalized_text") or "")
                    for alias in store.list_aliases(qa_id)
                    if alias.get("status") == "active"
                }
                prompt = build_alias_generation_prompt(
                    str(qa.get("question") or ""),
                    str(qa.get("answer") or ""),
                    list(qa.get("evidence") or []),
                    args.max_aliases_per_qa,
                )
                raw = llm.chat([
                    {"role": "system", "content": "Generate safe approved-QA aliases as JSON only."},
                    {"role": "user", "content": prompt},
                ])
                candidates = filter_alias_candidates(
                    parse_alias_generation_response(raw),
                    question=str(qa.get("question") or ""),
                    existing_normalized=existing_normalized,
                    max_aliases=args.max_aliases_per_qa,
                )
                generated += len(candidates)
                if dry_run:
                    rows = [
                        {
                            "alias_text": alias,
                            "normalized_text": normalize_qa_text(alias),
                            "risk_flags": detect_alias_risk_flags(str(qa.get("question") or ""), alias, list(qa.get("evidence") or [])),
                        }
                        for alias in candidates
                    ]
                    print(f"dry_run qa_id={qa_id} candidates={len(candidates)} {json.dumps(rows, ensure_ascii=False)}")
                    continue
                embeddings = embedder.embed(candidates) if candidates else []
                generated_status = args.status or store.generated_alias_default_status
                add_result = store.add_aliases(
                    qa_id,
                    [
                        {
                            "alias_text": alias,
                            "alias_type": "llm_paraphrase",
                            "status": "disabled" if detect_alias_risk_flags(str(qa.get("question") or ""), alias, list(qa.get("evidence") or [])) else generated_status,
                            "risk_flags": detect_alias_risk_flags(str(qa.get("question") or ""), alias, list(qa.get("evidence") or [])),
                            "embedding": embedding,
                            "generator_model": llm.model,
                            "memo": "generated by backfill script",
                        }
                        for alias, embedding in zip(candidates, embeddings)
                    ],
                )
                created += int(add_result.get("created_count") or 0)
                skipped += len(add_result.get("skipped") or [])
                errors += len(add_result.get("errors") or [])
            except Exception as exc:
                errors += 1
                print(f"error qa_id={qa_id} {exc}")

    summary = {
        "dry_run": dry_run,
        "sqlite_path": str(store.sqlite_path),
        "backup_path": str(backup_path) if backup_path else None,
        "target_count": len(result.get("items", [])),
        "would_insert": result.get("would_insert", 0),
        "would_update": result.get("would_update", 0),
        "would_disable": result.get("would_disable", 0),
        "original_created": result.get("original", {}).get("created", 0),
        "generated_candidates": generated,
        "alias_created": created,
        "skipped": skipped,
        "errors": errors,
        "alias_llm": alias_llm_meta,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
