# RAG API、管理者用API、ログ確認APIをFastAPIで公開します。
import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from api.config import get_cors_allow_origins, load_settings, project_path, rag_ask_url
from api.schemas import (
    QAAliasAddRequest,
    QAAliasBackfillRequest,
    QAAliasGenerateRequest,
    QAAliasUpdateRequest,
    QAMatchDebugRequest,
    ApprovedQARequest,
    ApprovedQAUpdateRequest,
    AskRequest,
    AskResponse,
    EvidenceSearchRequest,
    HallucinationReportRequest,
    LogTrendReportRequest,
    QASimilarRequest,
    QATestMatchRequest,
    ReportAnalysisUpdate,
    ReportStatusUpdate,
    SearchTagUpdateRequest,
)
from api.answer_cache import (
    build_alias_generation_prompt,
    detect_alias_risk_flags,
    filter_alias_candidates,
    normalize_qa_text,
    parse_alias_generation_response,
)
from api.llm_client import OpenAICompatibleLLM
from api.rag_engine import RAGEngine

app = FastAPI(title="RAG Template API", version="1.0.0")
settings = load_settings()
api_settings = settings.get("api", {})

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_allow_origins(),
    allow_credentials=bool(api_settings.get("allow_credentials", False)),
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = RAGEngine()


def _answer_cache_settings() -> dict:
    return load_settings().get("answer_cache", {})


def _embed_texts(texts: list[str]) -> list[list[float]]:
    return engine.retriever.embedding.embed(texts)


def _generate_aliases_for_qa(
    qa_id: int,
    *,
    max_aliases: int = 8,
    replace_existing_generated: bool = False,
    dry_run: bool = False,
    status: str | None = None,
) -> dict:
    qa = engine.answer_cache.get_approved_qa(qa_id)
    if qa is None:
        raise ValueError("approved QA not found")
    settings = _answer_cache_settings()
    max_aliases = max(1, min(int(max_aliases or settings.get("max_aliases_per_qa", 8)), 20))
    existing_normalized = {
        str(alias.get("normalized_text") or "")
        for alias in engine.answer_cache.list_aliases(qa_id)
        if alias.get("status") == "active"
    }
    alias_llm_settings = load_settings().get("alias_llm", {})
    llm = OpenAICompatibleLLM(
        api_key=alias_llm_settings.get("api_key"),
        model=alias_llm_settings.get("model"),
        temperature=float(alias_llm_settings.get("temperature", settings.get("alias_generation_temperature", 0.0))),
        max_tokens=int(alias_llm_settings.get("max_tokens", settings.get("alias_generation_max_tokens", 1200))),
        url_key="alias_llm_base_url",
    )
    if alias_llm_settings.get("timeout_sec"):
        llm.timeout_sec = int(alias_llm_settings.get("timeout_sec"))
    prompt = build_alias_generation_prompt(
        str(qa.get("question") or ""),
        str(qa.get("answer") or ""),
        list(qa.get("evidence") or []),
        max_aliases,
    )
    text = llm.chat([
        {"role": "system", "content": "あなたは承認済みQAの安全な別名質問をJSONだけで生成します。"},
        {"role": "user", "content": prompt},
    ])
    candidates = filter_alias_candidates(
        parse_alias_generation_response(text),
        question=str(qa.get("question") or ""),
        existing_normalized=existing_normalized,
        max_aliases=max_aliases,
    )
    candidate_rows = [
        {
            "alias_text": alias,
            "normalized_text": normalize_qa_text(alias),
            "risk_flags": detect_alias_risk_flags(
                str(qa.get("question") or ""),
                alias,
                list(qa.get("evidence") or []),
            ),
            "duplicate": False,
        }
        for alias in candidates
    ]
    if dry_run:
        return {
            "qa_id": qa_id,
            "question": qa.get("question"),
            "dry_run": True,
            "candidates": candidate_rows,
            "created": [],
            "old_aliases_kept": True,
            "disabled_old_count": 0,
            "inserted_count": 0,
            "errors": [],
            "raw": text,
        }
    if not candidates:
        return {
            "qa_id": qa_id,
            "dry_run": False,
            "candidates": [],
            "created": [],
            "old_aliases_kept": True,
            "disabled_old_count": 0,
            "inserted_count": 0,
            "errors": [{"error": "no alias candidates generated"}],
            "raw": text,
        }
    embeddings = _embed_texts(candidates) if candidates else []
    if len(embeddings) != len(candidates):
        return {
            "qa_id": qa_id,
            "dry_run": False,
            "candidates": candidate_rows,
            "created": [],
            "old_aliases_kept": True,
            "disabled_old_count": 0,
            "inserted_count": 0,
            "errors": [{"error": "embedding count mismatch"}],
            "raw": text,
        }
    alias_payload = [
        {
            "alias_text": alias,
            "alias_type": "llm_paraphrase",
            "status": status or engine.answer_cache.generated_alias_default_status,
            "embedding": embedding,
            "generator_model": llm.model,
            "memo": "generated by local LLM",
            "risk_flags": row.get("risk_flags", []),
        }
        for alias, embedding, row in zip(candidates, embeddings, candidate_rows)
    ]
    if replace_existing_generated:
        result = engine.answer_cache.replace_llm_aliases(
            qa_id,
            alias_payload,
            memo="replace generated aliases",
        )
    else:
        result = engine.answer_cache.add_aliases(qa_id, alias_payload)
        result.update({
            "old_aliases_kept": True,
            "disabled_old_count": 0,
            "inserted_count": int(result.get("created_count") or 0),
        })
    created_items = result.get("created") or []
    result.setdefault("created_count", len(created_items))
    result.setdefault("active_count", sum(1 for item in created_items if item.get("status") == "active"))
    result.setdefault("disabled_count", sum(1 for item in created_items if item.get("status") == "disabled"))
    result.setdefault("risk_count", sum(1 for item in created_items if item.get("risk_flags")))
    return {"qa_id": qa_id, "dry_run": False, "candidates": candidate_rows, **result}


def build_log_trend_report_prompt(dashboard: dict, days: int) -> str:
    report_data = {
        "集計期間日数": days,
        "KPI": dashboard.get("overview", {}),
        "日次質問数": dashboard.get("daily_questions", []),
        "直近24時間の時間別質問数": dashboard.get("hourly_questions", []),
        "No Hitと低信頼": dashboard.get("quality", {}),
        "子チャンクヒットランキング": dashboard.get("top_hit_chunks", [])[:20],
        "参照ファイルランキング": dashboard.get("top_source_files", [])[:20],
        "頻出質問ランキング": dashboard.get("top_questions", [])[:20],
    }
    return (
        "あなたはRAG運用改善と業務改善を支援する分析担当者です。\n"
        "以下のRAGチャットログ集計を読み、管理者向けに日本語で簡潔な傾向レポートを作成してください。\n"
        "単なる数字の説明ではなく、RAG改善、追加すべき文書、研修やマニュアル化すべき業務領域、次のアクションを分けて提案してください。\n"
        "根拠が薄い推測は断定せず、ログから読み取れる範囲で書いてください。\n\n"
        f"{json.dumps(report_data, ensure_ascii=False, indent=2)}"
    )


def file_info(relative_path: str) -> dict:
    path = project_path(relative_path)
    if not path.exists():
        return {"path": relative_path, "exists": False, "size_bytes": 0, "modified_at": None}
    modified_at = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
    return {
        "path": relative_path,
        "exists": True,
        "size_bytes": path.stat().st_size if path.is_file() else directory_size(path),
        "modified_at": modified_at,
    }


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def load_jsonl_rows(relative_path: str, limit: int | None = None) -> list[dict]:
    path = project_path(relative_path)
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def jsonl_summary(relative_path: str) -> dict:
    rows = load_jsonl_rows(relative_path)
    by_corpus = Counter(str(row.get("corpus_id") or "unknown") for row in rows)
    by_validity = Counter(_validity_status(row.get("valid_from"), row.get("valid_until")) for row in rows)
    tagged_count = sum(1 for row in rows if row.get("search_tags"))
    return {
        **file_info(relative_path),
        "row_count": len(rows),
        "tagged_count": tagged_count,
        "by_validity": dict(by_validity),
        "by_corpus": [{"corpus_id": corpus_id, "count": count} for corpus_id, count in by_corpus.most_common()],
    }


def corpus_status() -> list[dict]:
    config_path = project_path("config/corpus_settings.json")
    if not config_path.exists():
        return []
    data = json.loads(config_path.read_text(encoding="utf-8"))
    rows = []
    for corpus in data.get("corpora", []):
        markdown_dir = corpus.get("markdown_dir", "")
        markdown_path = project_path(markdown_dir) if markdown_dir else None
        markdown_files = list(markdown_path.rglob("*.md")) if markdown_path and markdown_path.exists() else []
        validity_counts = Counter()
        for md_path in markdown_files:
            metadata = _markdown_frontmatter(md_path)
            validity_counts[_validity_status(metadata.get("valid_from"), metadata.get("valid_until"))] += 1
        rows.append({
            "corpus_id": corpus.get("corpus_id"),
            "display_name": corpus.get("display_name"),
            "enabled": bool(corpus.get("enabled", True)),
            "priority": corpus.get("priority"),
            "markdown_dir": markdown_dir,
            "markdown_files": len(markdown_files),
            "validity": dict(validity_counts),
            "description": corpus.get("description", ""),
        })
    return rows


def _load_corpus_settings(*, required: bool = False) -> dict:
    config_path = project_path("config/corpus_settings.json")
    if not config_path.exists():
        if required:
            raise HTTPException(status_code=404, detail="config/corpus_settings.json not found")
        return {}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"failed to read corpus settings: {exc}") from exc
    if not isinstance(data.get("corpora"), list):
        if required:
            raise HTTPException(status_code=500, detail="corpus_settings.json must contain corpora list")
        return {}
    return data


def _write_corpus_settings(data: dict) -> None:
    config_path = project_path("config/corpus_settings.json")
    tmp_path = config_path.with_name(f"{config_path.name}.tmp")
    try:
        tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(config_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"failed to write corpus settings: {exc}") from exc


def default_enabled_corpus_ids() -> list[str] | None:
    data = _load_corpus_settings(required=False)
    corpora = data.get("corpora")
    if not isinstance(corpora, list) or not corpora:
        return None
    return [
        str(corpus.get("corpus_id"))
        for corpus in corpora
        if corpus.get("corpus_id") and bool(corpus.get("enabled", True))
    ]


def resolve_request_corpus_ids(corpus_ids: list[str] | None) -> list[str] | None:
    return corpus_ids if corpus_ids is not None else default_enabled_corpus_ids()


def update_corpus_enabled(corpus_id: str, enabled: bool) -> dict:
    data = _load_corpus_settings(required=True)
    for corpus in data.get("corpora", []):
        if str(corpus.get("corpus_id") or "") == str(corpus_id):
            corpus["enabled"] = bool(enabled)
            _write_corpus_settings(data)
            return dict(corpus)
    raise HTTPException(status_code=404, detail=f"corpus_id not found: {corpus_id}")


def _slugify(value: str, fallback: str = "document") -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    if not text:
        digest = hashlib.sha1(str(value or fallback).encode("utf-8")).hexdigest()[:10]
        text = f"{fallback}_{digest}"
    return text[:80].strip("._-") or fallback


def _strip_frontmatter(markdown_text: str) -> str:
    text = str(markdown_text or "").lstrip("\ufeff")
    if not text.startswith("---"):
        return text.strip()
    end = text.find("\n---", 3)
    if end == -1:
        return text.strip()
    return text[end + 4 :].lstrip()


def _yaml_line(key: str, value: object) -> str:
    return f"{key}: {json.dumps(str(value or ''), ensure_ascii=False)}"


def _normalize_date_field(payload: dict, key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        return ""
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"{key} must be YYYY-MM-DD") from exc


def _validity_status(valid_from: str | None, valid_until: str | None, today: date | None = None) -> str:
    today = today or date.today()
    start = None
    end = None
    try:
        if valid_from:
            start = date.fromisoformat(str(valid_from))
        if valid_until:
            end = date.fromisoformat(str(valid_until))
    except ValueError:
        return "invalid_period"
    if start and end and start > end:
        return "invalid_period"
    if start and today < start:
        return "not_started"
    if end and today > end:
        return "expired"
    if not start and not end:
        return "unbounded"
    return "current"


def _markdown_frontmatter(path: Path) -> dict[str, str]:
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return {}
    text = raw.lstrip("\ufeff")
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    metadata: dict[str, str] = {}
    for line in text[3:end].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        try:
            parsed = json.loads(value)
            value = str(parsed)
        except Exception:
            value = value.strip("\"'")
        metadata[key.strip()] = value
    return metadata


def _safe_project_path(relative_path: str) -> Path:
    root = project_path(".").resolve()
    target = project_path(relative_path).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"path is outside project: {relative_path}") from exc
    return target


def ensure_corpus_for_document(payload: dict) -> tuple[dict, bool]:
    corpus_id = str(payload.get("corpus_id") or "").strip()
    if not corpus_id:
        raise HTTPException(status_code=422, detail="corpus_id is required")

    data = _load_corpus_settings(required=True)
    corpora = data.setdefault("corpora", [])
    existing = None
    for corpus in corpora:
        if str(corpus.get("corpus_id") or "") == corpus_id:
            existing = corpus
            break

    created = existing is None
    corpus = existing if existing is not None else {"corpus_id": corpus_id}
    default_markdown_dir = f"data/markdown/{_slugify(corpus_id, 'corpus')}"
    markdown_dir = str(payload.get("markdown_dir") or corpus.get("markdown_dir") or default_markdown_dir).strip()
    if not markdown_dir:
        markdown_dir = default_markdown_dir
    _safe_project_path(markdown_dir).mkdir(parents=True, exist_ok=True)

    corpus["markdown_dir"] = markdown_dir
    corpus["display_name"] = str(payload.get("display_name") or corpus.get("display_name") or corpus_id).strip()
    corpus["description"] = str(payload.get("description") or corpus.get("description") or "").strip()
    corpus["enabled"] = bool(payload.get("enabled", corpus.get("enabled", True)))
    try:
        corpus["priority"] = int(payload.get("priority", corpus.get("priority", 100)))
    except Exception:
        corpus["priority"] = 100

    if created:
        corpora.append(corpus)
    _write_corpus_settings(data)
    return dict(corpus), created


def save_registered_markdown(payload: dict, corpus: dict) -> dict:
    markdown_text = _strip_frontmatter(str(payload.get("markdown_text") or payload.get("content") or ""))
    if not markdown_text:
        raise HTTPException(status_code=422, detail="markdown_text is required")

    title = str(payload.get("title") or "").strip()
    document_id = str(payload.get("document_id") or "").strip()
    filename_seed = str(payload.get("filename") or document_id or title or "document")
    filename = _slugify(Path(filename_seed).stem, "document") + ".md"
    markdown_dir = str(corpus.get("markdown_dir") or "").strip()
    if not markdown_dir:
        raise HTTPException(status_code=422, detail="markdown_dir is required")

    valid_from = _normalize_date_field(payload, "valid_from")
    valid_until = _normalize_date_field(payload, "valid_until")
    if valid_from and valid_until and date.fromisoformat(valid_from) > date.fromisoformat(valid_until):
        raise HTTPException(status_code=422, detail="valid_from must be before or equal to valid_until")

    output_path = _safe_project_path(markdown_dir) / filename
    metadata = {
        "title": title or filename_seed,
        "document_type": str(payload.get("document_type") or "manual").strip(),
        "document_id": document_id or Path(filename).stem,
        "source_url": str(payload.get("source_url") or "").strip(),
        "source_site": str(payload.get("source_site") or "").strip(),
        "valid_from": valid_from,
        "valid_until": valid_until,
        "registered_by": str(payload.get("registered_by") or "admin").strip(),
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    frontmatter = "\n".join(["---", *[_yaml_line(key, value) for key, value in metadata.items() if value], "---", ""])
    output_path.write_text(frontmatter + markdown_text.rstrip() + "\n", encoding="utf-8")
    return {
        "path": output_path.relative_to(project_path(".")).as_posix(),
        "filename": filename,
        "metadata": metadata,
    }


def run_maintenance_script(script_name: str, timeout_sec: int = 3600) -> dict:
    script_path = project_path(f"scripts/{script_name}")
    if not script_path.exists():
        raise HTTPException(status_code=500, detail=f"script not found: {script_name}")
    started = datetime.now(timezone.utc).isoformat()
    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(project_path(".")),
            text=True,
            capture_output=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail=f"{script_name} timed out after {timeout_sec} sec") from exc
    result = {
        "script": script_name,
        "started_at": started,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-12000:],
        "stderr": proc.stderr[-12000:],
    }
    if proc.returncode != 0:
        raise HTTPException(status_code=500, detail={"message": f"{script_name} failed", "result": result})
    return result


def register_markdown_document(payload: dict) -> dict:
    corpus, corpus_created = ensure_corpus_for_document(payload)
    saved = save_registered_markdown(payload, corpus)
    steps: list[dict] = []
    if bool(payload.get("run_chunking", True)):
        steps.append(run_maintenance_script("01_make_chunks.py"))
    if bool(payload.get("run_indexing", True)):
        steps.append(run_maintenance_script("03_build_index.py"))
    reload_result = None
    if steps:
        reload_result = engine.reload_retriever()
    return {
        "status": "ok",
        "corpus": corpus,
        "corpus_created": corpus_created,
        "document": saved,
        "steps": steps,
        "retriever_reload": reload_result,
        "index_status": admin_index_status(),
    }


def chunk_report_summary() -> dict:
    path = project_path("chunks/chunk_report.csv")
    if not path.exists():
        return {**file_info("chunks/chunk_report.csv"), "row_count": 0, "warning_count": 0, "by_corpus": [], "warnings": []}

    by_corpus: dict[str, dict] = {}
    warnings = []
    row_count = 0
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            row_count += 1
            corpus_id = row.get("corpus_id") or "unknown"
            stat = by_corpus.setdefault(corpus_id, {"corpus_id": corpus_id, "files": 0, "parent_count": 0, "child_count": 0})
            stat["files"] += 1
            stat["parent_count"] += int(float(row.get("parent_count") or 0))
            stat["child_count"] += int(float(row.get("child_count") or 0))
            warning = row.get("warnings")
            if warning:
                warnings.append({
                    "corpus_id": corpus_id,
                    "source_file": row.get("source_file"),
                    "title": row.get("title"),
                    "warnings": warning,
                })

    return {
        **file_info("chunks/chunk_report.csv"),
        "row_count": row_count,
        "warning_count": len(warnings),
        "by_corpus": list(by_corpus.values()),
        "warnings": warnings[:100],
    }


def form_catalog_status() -> dict:
    path = project_path("data/forms/form_catalog.csv")
    info = file_info("data/forms/form_catalog.csv")
    row_count = 0
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            row_count = sum(1 for _ in csv.DictReader(f))
    return {
        **info,
        "row_count": row_count,
        "configured": row_count > 0,
        "ui_state": "available" if row_count > 0 else "not_configured",
    }


def vector_collection_status() -> dict:
    status = {
        "collection_name": getattr(engine.retriever, "collection_name", None),
        "corpus_version": getattr(engine.retriever, "corpus_version", None),
        "index_version": getattr(engine.retriever, "index_version", None),
        "count": None,
        "error": None,
    }
    collection = getattr(engine.retriever, "collection", None)
    if collection is not None and hasattr(collection, "count"):
        try:
            status["count"] = collection.count()
        except Exception as exc:
            status["error"] = str(exc)
    release_manager = getattr(engine.retriever, "release_manager", None)
    if release_manager is not None:
        try:
            status["active_release"] = release_manager.get_active_release()
            status["release_count"] = len(release_manager.list_releases())
        except Exception as exc:
            status["release_error"] = str(exc)
    return status


SEARCH_TAG_FILE = "chunks/child_chunks_with_tags.jsonl"
BASE_CHILD_FILE = "chunks/child_chunks.jsonl"
SEARCH_TAG_REINDEX_WARNING = (
    "SearchTag JSONL and in-memory BM25 were updated. "
    "Run scripts/03_build_index.py to reflect the change in dense vector search."
)


def child_tag_read_source() -> tuple[Path, str]:
    tagged_path = project_path(SEARCH_TAG_FILE)
    if tagged_path.exists():
        return tagged_path, SEARCH_TAG_FILE
    return project_path(BASE_CHILD_FILE), BASE_CHILD_FILE


def load_child_tag_rows() -> tuple[list[dict], str]:
    path, relative_path = child_tag_read_source()
    if not path.exists():
        return [], relative_path
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows, relative_path


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def normalize_search_tags(raw_tags: list[str]) -> list[str]:
    tags = []
    seen = set()
    for raw_tag in raw_tags or []:
        for part in str(raw_tag).replace("\r", "\n").replace(",", "\n").replace("、", "\n").replace("，", "\n").split("\n"):
            tag = part.strip()
            if tag and tag not in seen:
                seen.add(tag)
                tags.append(tag)
    return tags[:100]


def build_child_search_text(row: dict) -> str:
    return "\n".join([
        str(row.get("title") or ""),
        str(row.get("heading_path") or ""),
        str(row.get("text") or row.get("child_text") or ""),
        " ".join(str(tag) for tag in row.get("search_tags") or []),
    ]).strip()


def search_tag_haystack(row: dict) -> str:
    values = [
        row.get("child_id"),
        row.get("parent_id"),
        row.get("corpus_id"),
        row.get("title"),
        row.get("heading_path"),
        row.get("source_file"),
        row.get("source_url"),
        row.get("text"),
        " ".join(str(tag) for tag in row.get("search_tags") or []),
    ]
    return "\n".join(str(value or "") for value in values).casefold()


def search_tag_summary(row: dict, relative_path: str) -> dict:
    text = str(row.get("text") or row.get("child_text") or "")
    return {
        "child_id": row.get("child_id"),
        "parent_id": row.get("parent_id"),
        "corpus_id": row.get("corpus_id"),
        "title": row.get("title"),
        "heading_path": row.get("heading_path"),
        "source_file": row.get("source_file"),
        "source_url": row.get("source_url"),
        "search_tags": row.get("search_tags") or [],
        "tag_count": len(row.get("search_tags") or []),
        "text_preview": text[:240] + ("..." if len(text) > 240 else ""),
        "editable_file": SEARCH_TAG_FILE,
        "source_file_path": relative_path,
    }


def find_child_tag_row(child_id: str) -> tuple[list[dict], int, dict | None, str]:
    rows, relative_path = load_child_tag_rows()
    for idx, row in enumerate(rows):
        if str(row.get("child_id") or "") == child_id:
            return rows, idx, row, relative_path
    return rows, -1, None, relative_path


@app.get("/health")
def health():
    return {
        "status": "ok",
        "logging": engine.log_store.status(),
        "answer_cache": engine.answer_cache.status(),
    }


@app.get("/health/deep")
def health_deep():
    logging_status = engine.log_store.status()
    answer_cache_status = engine.answer_cache.status()
    retriever_status = {
        "collection_name": getattr(engine.retriever, "collection_name", None),
        "corpus_version": getattr(engine.retriever, "corpus_version", None),
        "index_version": getattr(engine.retriever, "index_version", None),
    }
    alias_status = answer_cache_status.get("alias_index") or {}
    degraded = not logging_status.get("enabled") or not answer_cache_status.get("enabled") or bool(alias_status.get("error"))
    return {
        "status": "degraded" if degraded else "ok",
        "logging": logging_status,
        "answer_cache": answer_cache_status,
        "retriever": retriever_status,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    corpus_ids = resolve_request_corpus_ids(req.corpus_ids)
    result = engine.ask(
        question=req.question,
        corpus_ids=corpus_ids,
        top_k=req.top_k,
        show_debug=req.show_debug,
        session_id=req.session_id,
        history=[m.model_dump() for m in req.history],
    )
    return result


@app.post("/ask_stream")
def ask_stream(req: AskRequest):
    corpus_ids = resolve_request_corpus_ids(req.corpus_ids)

    def iter_events():
        try:
            for event in engine.ask_events(
                question=req.question,
                corpus_ids=corpus_ids,
                top_k=req.top_k,
                show_debug=req.show_debug,
                session_id=req.session_id,
                history=[m.model_dump() for m in req.history],
            ):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as exc:
            error_event = {
                "type": "error",
                "stage": "error",
                "message": str(exc),
                "session_id": req.session_id,
            }
            yield json.dumps(error_event, ensure_ascii=False) + "\n"

    return StreamingResponse(iter_events(), media_type="application/x-ndjson")


@app.get("/logs/recent")
def recent_logs(
    limit: int = Query(default=50, ge=1, le=500),
    include_sources: bool = Query(default=False),
):
    return {
        "logs": engine.log_store.recent(limit=limit, include_sources=include_sources),
        "logging": engine.log_store.status(),
    }


@app.get("/admin/logs/recent")
def admin_recent_logs(limit: int = Query(default=50, ge=1, le=500)):
    return {
        "logs": engine.log_store.recent(limit=limit, include_sources=True),
        "logging": engine.log_store.status(),
    }


@app.get("/admin/logs/status")
def admin_logs_status():
    return engine.log_store.status()


@app.get("/admin/logs/search")
def admin_search_logs(
    limit: int = Query(default=100, ge=1, le=500),
    days: int = Query(default=30, ge=1, le=365),
    query: str | None = Query(default=None),
    corpus_id: str | None = Query(default=None),
    source_state: str = Query(default="all", pattern="^(all|with_sources|no_sources|low_confidence)$"),
    low_score_threshold: float = Query(default=0.35, ge=0.0, le=1.0),
    answer_source: str | None = Query(default=None),
    qa_cache_id: int | None = Query(default=None),
    session_id: str | None = Query(default=None),
    log_id: int | None = Query(default=None),
    min_score: float | None = Query(default=None, ge=0.0, le=1.0),
    max_score: float | None = Query(default=None, ge=0.0, le=1.0),
):
    return {
        "logs": engine.log_store.search(
            limit=limit,
            days=days,
            query=query,
            corpus_id=corpus_id,
            source_state=source_state,
            low_score_threshold=low_score_threshold,
            answer_source=answer_source,
            qa_cache_id=qa_cache_id,
            session_id=session_id,
            log_id=log_id,
            min_score=min_score,
            max_score_filter=max_score,
        ),
        "logging": engine.log_store.status(),
    }


@app.get("/admin/logs/dashboard")
def admin_log_dashboard(
    sample_limit: int = Query(default=1000, ge=1, le=5000),
    top_n: int = Query(default=20, ge=1, le=100),
    days: int = Query(default=7, ge=1, le=90),
    low_score_threshold: float = Query(default=0.35, ge=0.0, le=1.0),
):
    dashboard = engine.log_store.dashboard(
        sample_limit=sample_limit,
        top_n=top_n,
        days=days,
        low_score_threshold=low_score_threshold,
    )
    dashboard["logging"] = engine.log_store.status()
    return dashboard


@app.post("/admin/logs/report")
def admin_log_trend_report(req: LogTrendReportRequest):
    dashboard = engine.log_store.dashboard(
        sample_limit=req.sample_limit,
        top_n=req.top_n,
        days=req.days,
        low_score_threshold=req.low_score_threshold,
    )
    messages = [
        {"role": "system", "content": "あなたはRAG運用ログを分析し、現場改善に使える示唆を返すアシスタントです。"},
        {"role": "user", "content": build_log_trend_report_prompt(dashboard, req.days)},
    ]
    try:
        report = engine.llm.chat(messages)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"log trend report generation failed: {exc}") from exc
    return {"report": report, "dashboard": dashboard}


@app.get("/admin/index/status")
def admin_index_status():
    settings_snapshot = load_settings()
    return {
        "corpora": corpus_status(),
        "chunks": {
            "parent_chunks": jsonl_summary("chunks/parent_chunks.jsonl"),
            "child_chunks": jsonl_summary("chunks/child_chunks.jsonl"),
            "child_chunks_with_tags": jsonl_summary("chunks/child_chunks_with_tags.jsonl"),
            "chunk_report": chunk_report_summary(),
        },
        "forms": form_catalog_status(),
        "storage": {
            "chunks": file_info("chunks"),
            "chroma": file_info(settings_snapshot.get("retrieval", {}).get("chroma_path") or "indexes/chroma"),
            "qdrant": file_info("indexes/qdrant"),
            "logs": file_info("logs"),
        },
        "retrieval_settings": settings_snapshot.get("retrieval", {}),
        "vector_db": settings_snapshot.get("vector_db", {}),
        "answer_cache": {
            "enabled": settings_snapshot.get("answer_cache", {}).get("enabled", True),
            "corpus_version": settings_snapshot.get("answer_cache", {}).get("corpus_version"),
            "index_version": settings_snapshot.get("answer_cache", {}).get("index_version"),
        },
        "vector_collection": vector_collection_status(),
    }


@app.post("/admin/index/corpora/{corpus_id}/enabled")
def admin_index_corpus_enabled(corpus_id: str, payload: dict):
    enabled = payload.get("enabled")
    if not isinstance(enabled, bool):
        raise HTTPException(status_code=422, detail="enabled must be a boolean")
    updated = update_corpus_enabled(corpus_id, enabled)
    return {
        "status": "ok",
        "updated": updated,
        "corpora": corpus_status(),
        "default_corpus_ids": default_enabled_corpus_ids(),
    }


@app.post("/admin/documents/register-markdown")
def admin_register_markdown_document(payload: dict):
    try:
        return register_markdown_document(payload)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"markdown document registration failed: {exc}") from exc


def count_status(summary: dict, status: str) -> int:
    for row in summary.get("by_status", []):
        if row.get("status") == status:
            return int(row.get("count") or 0)
    return 0


def build_index_summary(index_status: dict) -> dict:
    corpora = index_status.get("corpora", [])
    chunks = index_status.get("chunks", {})
    parent = chunks.get("parent_chunks", {})
    child = chunks.get("child_chunks", {})
    tagged = chunks.get("child_chunks_with_tags", {})
    report = chunks.get("chunk_report", {})
    child_count = int(child.get("row_count") or 0)
    tagged_count = int(tagged.get("tagged_count") or 0)
    return {
        "enabled_corpora": sum(1 for corpus in corpora if corpus.get("enabled")),
        "markdown_files": sum(int(corpus.get("markdown_files") or 0) for corpus in corpora),
        "parent_chunks": int(parent.get("row_count") or 0),
        "child_chunks": child_count,
        "tagged_chunks": tagged_count,
        "tag_coverage_rate": round(tagged_count / child_count, 4) if child_count else 0.0,
        "chunk_warnings": int(report.get("warning_count") or 0),
        "vector_count": index_status.get("vector_collection", {}).get("count"),
        "vector_mismatch": (
            index_status.get("vector_collection", {}).get("count") is not None
            and int(index_status.get("vector_collection", {}).get("count") or 0) != child_count
        ),
    }


def percent(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator) / float(denominator), 4) if denominator else 0.0


def priority_rank(priority: str) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(priority, 3)


def make_task(
    *,
    priority: str,
    task_type: str,
    title: str,
    question: str = "",
    count: int = 1,
    last_seen_at: str | None = None,
    related_corpus: str = "",
    max_score: float | None = None,
    reason_hint: str = "",
    suggested_action: str = "",
    links: dict | None = None,
    extra: dict | None = None,
) -> dict:
    task = {
        "priority": priority,
        "task_type": task_type,
        "title": title,
        "question": question,
        "count": count,
        "last_seen_at": last_seen_at,
        "related_corpus": related_corpus,
        "max_score": max_score,
        "reason_hint": reason_hint,
        "suggested_action": suggested_action,
        "links": links or {},
    }
    if extra:
        task.update(extra)
    return task


def build_improvement_tasks(
    *,
    dashboard: dict,
    open_reports: list[dict],
    tag_candidates: list[dict],
    chunk_warnings: list[dict],
    index_summary: dict,
    eval_status: dict,
    top_n: int,
) -> list[dict]:
    search_tag_settings = load_settings().get("search_tags", {})
    show_search_tag_candidates = bool(search_tag_settings.get("show_improvement_candidates", False))
    tasks: list[dict] = []
    for report in open_reports:
        tasks.append(make_task(
            priority="high",
            task_type="hallucination_report",
            title="未対応のハルシネーション疑い報告があります",
            question=str(report.get("question") or ""),
            count=1,
            last_seen_at=report.get("created_at"),
            reason_hint="ユーザーが回答品質に疑義を出しています",
            suggested_action="報告詳細で回答時の根拠を確認し、QA登録・SearchTag補強・文書追加のいずれかに分類してください。",
            links={"reports": f"/reports?report_id={report.get('id')}"},
        ))

    for log in dashboard.get("quality", {}).get("no_hit_logs", [])[:top_n]:
        tasks.append(make_task(
            priority="high",
            task_type="no_hit",
            title="根拠なし回答が発生しています",
            question=str(log.get("question") or log.get("question_preview") or ""),
            count=1,
            last_seen_at=log.get("created_at"),
            max_score=0.0,
            reason_hint="検索結果の根拠チャンクが0件です",
            suggested_action="対象外質問か、文書不足か、SearchTag不足かをログ詳細から確認してください。",
            links={"logs": f"/logs?source_state=no_sources&query={log.get('id', '')}"},
        ))

    for log in dashboard.get("quality", {}).get("low_confidence_logs", [])[:top_n]:
        tasks.append(make_task(
            priority="medium",
            task_type="low_confidence",
            title="低信頼の回答があります",
            question=str(log.get("question") or log.get("question_preview") or ""),
            count=1,
            last_seen_at=log.get("created_at"),
            max_score=log.get("max_score"),
            reason_hint="根拠はありますが最大スコアがしきい値未満です",
            suggested_action="正しい文書がヒットしているかを確認し、SearchTag補強またはQA候補化を検討してください。",
            links={"logs": f"/logs?source_state=low_confidence&query={log.get('id', '')}"},
        ))

    for log in dashboard.get("quality", {}).get("near_cache_miss_logs", [])[:top_n]:
        tasks.append(make_task(
            priority="medium",
            task_type="near_cache_miss",
            title="惜しい承認QAキャッシュミスがあります",
            question=str(log.get("question") or log.get("question_preview") or ""),
            count=1,
            last_seen_at=log.get("created_at"),
            max_score=log.get("cache_candidate_similarity"),
            reason_hint=f"候補QA#{log.get('cache_candidate_qa_id')} similarity={log.get('cache_candidate_similarity')} / reason={log.get('cache_miss_reason')}",
            suggested_action="候補QAにaliasを追加するか、match-debugで誤爆しないか確認してください。",
            links={
                "logs": f"/logs?log_id={log.get('id', '')}",
                "qa": f"/qa-cache?qa_id={log.get('cache_candidate_qa_id', '')}&question={log.get('question', '')}",
            },
            extra={
                "log_id": log.get("id"),
                "cache_candidate_qa_id": log.get("cache_candidate_qa_id"),
                "cache_candidate_similarity": log.get("cache_candidate_similarity"),
                "cache_miss_reason": log.get("cache_miss_reason"),
            },
        ))

    for row in dashboard.get("top_questions", [])[:top_n]:
        hits = int(row.get("hits") or 0)
        if hits >= 2:
            tasks.append(make_task(
                priority="medium",
                task_type="frequent_question",
                title="頻出質問です",
                question=str(row.get("question") or ""),
                count=hits,
                reason_hint="同じ質問が複数回出ています",
                suggested_action="回答が安定しているなら承認QA化し、揺れるなら評価セットへ追加してください。",
                links={"qa_new": f"/qa-cache?question={row.get('question', '')}"},
            ))

    if show_search_tag_candidates:
        for candidate in tag_candidates[:top_n]:
            tasks.append(make_task(
                priority="low",
                task_type="search_tag_candidate",
                title="SearchTag補強候補",
                count=int(candidate.get("tag_count") or 0),
                related_corpus=str(candidate.get("corpus_id") or ""),
                reason_hint="失敗/低信頼ログの語句に関連しそうなチャンクです",
                suggested_action="補助機能としてSearchTag編集を確認してください。通常は先にQA alias追加、承認QA化、文書不足確認を優先します。",
                links={"search_tags": f"/search-tags?child_id={candidate.get('child_id', '')}"},
            ))

    for warning in chunk_warnings[:top_n]:
        tasks.append(make_task(
            priority="medium",
            task_type="index_warning",
            title="チャンク生成警告があります",
            count=1,
            last_seen_at=None,
            related_corpus=str(warning.get("corpus_id") or ""),
            reason_hint=str(warning.get("warnings") or ""),
            suggested_action="chunk_report.csvを確認し、親子チャンク分割やMarkdown構造を見直してください。",
            links={"index": "/index-eval"},
        ))

    if index_summary.get("vector_mismatch"):
        tasks.append(make_task(
            priority="high",
            task_type="index_warning",
            title="子チャンク数とベクトル数が一致していません",
            count=1,
            reason_hint="検索対象のIndexが最新チャンクを反映していない可能性があります",
            suggested_action="scripts/03_build_index.pyを再実行し、Chroma collectionを更新してください。",
            links={"index": "/index-eval"},
        ))

    failed_count = int(eval_status.get("latest", {}).get("failed_count") or 0)
    if failed_count:
        tasks.append(make_task(
            priority="high",
            task_type="eval_failed",
            title="評価セットに失敗があります",
            count=failed_count,
            last_seen_at=eval_status.get("latest", {}).get("modified_at"),
            reason_hint="直近の評価結果にNG/エラー/根拠不足が含まれます",
            suggested_action="失敗質問を改善キューへ回し、QA登録・SearchTag補強・文書追加を検討してください。",
            links={"eval": "/index-eval#eval"},
        ))

    tasks.sort(key=lambda item: (priority_rank(item.get("priority", "")), -int(item.get("count") or 0), str(item.get("last_seen_at") or "")), reverse=False)
    return tasks[: max(top_n, 10)]


def latest_eval_result() -> dict:
    candidates = [
        "eval/qa_100_rag_results.csv",
        "eval/eval_results.csv",
    ]
    existing = [file_info(path) for path in candidates if project_path(path).exists()]
    if not existing:
        return {"exists": False, "path": None, "modified_at": None, "row_count": 0, "failed_count": 0, "items": []}
    latest = max(existing, key=lambda item: item.get("modified_at") or "")
    path = project_path(str(latest["path"]))
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
    failed = []
    for row in rows:
        status = str(row.get("status") or "").casefold()
        source_count = int(float(row.get("source_count") or 0)) if str(row.get("source_count") or "").replace(".", "", 1).isdigit() else None
        if status and status != "ok":
            failed.append(row)
            continue
        if source_count == 0:
            failed.append(row)
    return {
        **latest,
        "exists": True,
        "row_count": len(rows),
        "failed_count": len(failed),
        "ok_count": len(rows) - len(failed),
        "items": rows[:100],
        "failed_items": failed[:100],
    }


def eval_status_summary() -> dict:
    qa100 = file_info("eval/qa_100_questions.csv")
    golden = file_info("eval/golden/rag_golden_dataset.jsonl")
    latest = latest_eval_result()
    return {
        "qa_100_questions": qa100,
        "golden_jsonl": golden,
        "latest": latest,
        "commands": [
            "python scripts/10_batch_inference.py --input eval/golden/rag_golden_dataset.jsonl --show-debug",
            "python scripts/11_eval_ragas.py --input eval/runs/<run>/predictions.jsonl",
            "python scripts/12_eval_deepeval.py --input eval/runs/<run>/predictions.jsonl --local-openai",
            "python scripts/13_eval_rules.py --input eval/runs/<run>/predictions.jsonl",
            f"legacy: python scripts/06_batch_rag_api_qa100.py --api {rag_ask_url()}",
            f"legacy: python scripts/05_batch_eval.py --input eval/qa_100_questions.csv --api {rag_ask_url()}",
        ],
    }


@app.get("/admin/ops/summary")
def admin_ops_summary(
    sample_limit: int = Query(default=1000, ge=1, le=5000),
    top_n: int = Query(default=20, ge=1, le=100),
    days: int = Query(default=7, ge=1, le=90),
    low_score_threshold: float = Query(default=0.35, ge=0.0, le=1.0),
):
    dashboard = engine.log_store.dashboard(
        sample_limit=sample_limit,
        top_n=top_n,
        days=days,
        low_score_threshold=low_score_threshold,
    )
    report_summary = engine.answer_cache.report_summary(limit=top_n)
    qa_summary = engine.answer_cache.approved_summary(limit=top_n)
    alias_summary = qa_summary.get("aliases", {})
    index_status = admin_index_status()
    sample = dashboard.get("sample", {})
    quality = dashboard.get("quality", {})
    overview = dashboard.get("overview", {})
    sampled_questions = int(sample.get("sampled_questions") or 0)
    no_hit_count = int(quality.get("no_hit_count") or 0)
    low_confidence_count = int(quality.get("low_confidence_count") or 0)
    index_summary = build_index_summary(index_status)
    logging_status = engine.log_store.status()
    open_reports = count_status(report_summary, "open")
    alerts = []
    if not logging_status.get("enabled"):
        alerts.append({
            "level": "warning",
            "message": "チャットログDBが無効化されています",
            "action": "/admin/logs/status で last_error と db_path を確認してください",
        })
    if open_reports:
        alerts.append({"level": "error", "message": f"未対応報告が{open_reports}件あります", "action": "報告対応を確認してください"})
    if sampled_questions and percent(no_hit_count, sampled_questions) >= 0.1:
        alerts.append({"level": "warning", "message": "根拠なし率が高めです", "action": "No Hitログを改善キューで確認してください"})
    if sampled_questions and percent(low_confidence_count, sampled_questions) >= 0.2:
        alerts.append({"level": "warning", "message": "低信頼回答が増えています", "action": "SearchTag補強または文書追加を検討してください"})
    if index_summary.get("chunk_warnings"):
        alerts.append({"level": "warning", "message": "chunk_reportに警告があります", "action": "文書/評価画面でチャンク警告を確認してください"})
    if index_summary.get("vector_mismatch"):
        alerts.append({"level": "error", "message": "子チャンク数とベクトル数が一致していません", "action": "Indexを再構築してください"})
    health_status = "異常あり" if any(a["level"] == "error" for a in alerts) else "警告あり" if alerts else "OK"
    return {
        "dashboard": dashboard,
        "reports": report_summary,
        "qa_cache": qa_summary,
        "index": index_summary,
        "health": {
            "status": health_status,
            "alerts": alerts,
            "last_log_at": overview.get("last_log_at"),
            "logging": logging_status,
        },
        "kpis": {
            "questions_in_window": overview.get("window_questions", 0),
            "sessions_in_window": overview.get("window_sessions", 0),
            "no_hit_count": no_hit_count,
            "low_confidence_count": low_confidence_count,
            "no_hit_rate": percent(no_hit_count, sampled_questions),
            "low_confidence_rate": percent(low_confidence_count, sampled_questions),
            "open_reports": open_reports,
            "approved_qa": count_status(qa_summary, "approved"),
            "disabled_qa": count_status(qa_summary, "disabled"),
            "active_aliases": alias_summary.get("active", 0),
            "disabled_aliases": alias_summary.get("disabled", 0),
            "alias_missing_qa": alias_summary.get("original_missing_qa", 0),
            "llm_alias_missing_qa": alias_summary.get("llm_missing_qa", 0),
            "alias_index_count": (alias_summary.get("alias_index") or {}).get("count", 0),
            "qa_cache_hits": overview.get("window_cache_hits", 0),
            "qa_cache_hit_rate": percent(int(overview.get("window_cache_hits") or 0), int(overview.get("window_questions") or 0)),
            "rag_answer_rate": percent(int(overview.get("window_rag_answers") or 0), int(overview.get("window_questions") or 0)),
            "avg_max_score": overview.get("avg_max_score", 0.0),
            "error_count": overview.get("error_count", 0),
            "parent_chunks": index_summary.get("parent_chunks", 0),
            "child_chunks": index_summary.get("child_chunks", 0),
            "tag_coverage_rate": index_summary.get("tag_coverage_rate", 0.0),
            "chunk_warnings": index_summary.get("chunk_warnings", 0),
            "vector_count": index_summary.get("vector_count"),
        },
    }


@app.get("/admin/actions/improvement-candidates")
def admin_improvement_candidates(
    sample_limit: int = Query(default=1000, ge=1, le=5000),
    top_n: int = Query(default=20, ge=1, le=100),
    days: int = Query(default=7, ge=1, le=90),
    low_score_threshold: float = Query(default=0.35, ge=0.0, le=1.0),
):
    dashboard = engine.log_store.dashboard(
        sample_limit=sample_limit,
        top_n=top_n,
        days=days,
        low_score_threshold=low_score_threshold,
    )
    rows, relative_path = load_child_tag_rows()
    low_confidence_detail_logs = engine.log_store.search(
        limit=top_n,
        days=days,
        source_state="low_confidence",
        low_score_threshold=low_score_threshold,
    )
    failed_child_ids = {
        str(source.get("child_id") or "")
        for log in low_confidence_detail_logs
        for source in log.get("sources", [])
        if isinstance(source, dict) and source.get("child_id")
    }
    tag_candidates = []
    for row in rows:
        summary = search_tag_summary(row, relative_path)
        if str(row.get("child_id") or "") in failed_child_ids:
            summary["candidate_reason"] = "低信頼ログで実際に参照されたチャンクです"
            tag_candidates.append(summary)
        elif len(row.get("search_tags") or []) <= 2 and len(tag_candidates) < top_n:
            summary["candidate_reason"] = "SearchTagが少ないため、失敗ログ調査時の補助候補です"
            tag_candidates.append(summary)
        if len(tag_candidates) >= top_n:
            break
    chunk_report = chunk_report_summary()
    index_summary = build_index_summary(admin_index_status())
    eval_status = eval_status_summary()
    open_reports = engine.answer_cache.recent_reports(status="open", limit=top_n)
    tasks = build_improvement_tasks(
        dashboard=dashboard,
        open_reports=open_reports,
        tag_candidates=tag_candidates,
        chunk_warnings=chunk_report.get("warnings", [])[:top_n],
        index_summary=index_summary,
        eval_status=eval_status,
        top_n=top_n,
    )
    qa_summary = engine.answer_cache.approved_summary(limit=top_n)
    alias_summary = qa_summary.get("aliases", {})
    missing_original = int(alias_summary.get("original_missing_qa") or 0)
    missing_llm = int(alias_summary.get("llm_missing_qa") or 0)
    if missing_original:
        tasks.append(make_task(
            priority="high",
            task_type="qa_alias_missing",
            title="original alias未作成の承認QAがあります",
            count=missing_original,
            reason_hint="既存SQLiteにaliasテーブル導入前のQAが残っている可能性があります。",
            suggested_action="承認QA画面からoriginal alias補完を実行してください。",
            links={"qa": "/qa-cache"},
        ))
    if missing_llm:
        tasks.append(make_task(
            priority="low",
            task_type="qa_alias_missing",
            title="LLM alias未生成の承認QAがあります",
            count=missing_llm,
            reason_hint="承認QAの言い換え入口が少ない可能性があります。",
            suggested_action="1QAあたり5〜8個を目安にdry-run後、問題ないものだけ保存してください。",
            links={"qa": "/qa-cache"},
        ))
    tasks.sort(key=lambda item: (priority_rank(item.get("priority", "")), -int(item.get("count") or 0), str(item.get("last_seen_at") or "")), reverse=False)
    tasks = tasks[: max(top_n, 10)]
    return {
        "tasks": tasks,
        "no_hit_logs": dashboard.get("quality", {}).get("no_hit_logs", [])[:top_n],
        "low_confidence_logs": dashboard.get("quality", {}).get("low_confidence_logs", [])[:top_n],
        "frequent_questions": dashboard.get("top_questions", [])[:top_n],
        "frequent_chunks": dashboard.get("top_hit_chunks", [])[:top_n],
        "open_reports": open_reports,
        "search_tag_candidates": tag_candidates,
        "chunk_warnings": chunk_report.get("warnings", [])[:top_n],
        "eval_failed": eval_status.get("latest", {}).get("failed_items", [])[:top_n],
    }


@app.get("/admin/search-tags")
def admin_search_tags(
    query: str | None = Query(default=None),
    corpus_id: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
):
    rows, relative_path = load_child_tag_rows()
    query_text = (query or "").strip().casefold()
    corpus_filter = (corpus_id or "").strip()
    items = []
    total_matches = 0
    for row in rows:
        if corpus_filter and str(row.get("corpus_id") or "") != corpus_filter:
            continue
        if query_text and query_text not in search_tag_haystack(row):
            continue
        total_matches += 1
        if len(items) < limit:
            items.append(search_tag_summary(row, relative_path))
    return {
        "items": items,
        "total_matches": total_matches,
        "returned": len(items),
        "source_file": relative_path,
        "editable_file": SEARCH_TAG_FILE,
        "dense_reindex_required": True,
        "warning": SEARCH_TAG_REINDEX_WARNING,
    }


@app.get("/admin/search-tags/{child_id:path}")
def admin_search_tag_detail(child_id: str):
    _, _, row, relative_path = find_child_tag_row(child_id)
    if row is None:
        raise HTTPException(status_code=404, detail="child chunk not found")
    return {
        "item": row,
        "summary": search_tag_summary(row, relative_path),
        "source_file": relative_path,
        "editable_file": SEARCH_TAG_FILE,
        "dense_reindex_required": True,
        "warning": SEARCH_TAG_REINDEX_WARNING,
    }


@app.patch("/admin/search-tags/{child_id:path}")
def update_search_tags(child_id: str, req: SearchTagUpdateRequest):
    rows, idx, row, relative_path = find_child_tag_row(child_id)
    if row is None:
        raise HTTPException(status_code=404, detail="child chunk not found")

    updated = dict(row)
    updated["search_tags"] = normalize_search_tags(req.search_tags)
    updated["search_text"] = build_child_search_text(updated)
    rows[idx] = updated

    write_jsonl_atomic(project_path(SEARCH_TAG_FILE), rows)

    reload_result = None
    reload_error = None
    if req.reload_retriever:
        try:
            reload_result = engine.reload_retriever()
        except Exception as exc:
            reload_error = str(exc)

    return {
        "status": "ok",
        "item": updated,
        "summary": search_tag_summary(updated, SEARCH_TAG_FILE),
        "source_file": relative_path,
        "editable_file": SEARCH_TAG_FILE,
        "reload_result": reload_result,
        "reload_error": reload_error,
        "dense_reindex_required": True,
        "warning": SEARCH_TAG_REINDEX_WARNING,
    }


@app.post("/feedback/hallucination")
def report_hallucination(req: HallucinationReportRequest):
    try:
        report_id = engine.answer_cache.report_hallucination(
            question=req.question,
            answer=req.answer,
            session_id=req.session_id,
            log_id=req.log_id,
            comment=req.comment,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"report_id": report_id}


@app.get("/admin/reports")
def admin_reports(
    status: str | None = Query(default=None, pattern="^(open|resolved|ignored)$"),
    limit: int = Query(default=50, ge=1, le=500),
):
    return {"reports": engine.answer_cache.recent_reports(status=status, limit=limit)}


@app.get("/admin/reports/summary")
def admin_reports_summary(limit: int = Query(default=20, ge=1, le=100)):
    return engine.answer_cache.report_summary(limit=limit)


@app.get("/admin/reports/{report_id}")
def admin_report_detail(report_id: int):
    report = engine.answer_cache.get_report(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="hallucination report not found")
    related_logs = []
    if report.get("log_id"):
        related_logs = engine.log_store.search(limit=1, days=365, log_id=int(report["log_id"]))
    similar_logs = engine.log_store.search(limit=10, days=90, query=report.get("question"))
    return {
        "report": report,
        "log": related_logs[0] if related_logs else None,
        "similar_logs": [row for row in similar_logs if str(row.get("id")) != str(report.get("log_id"))],
    }


@app.patch("/admin/reports/{report_id}/status")
def update_report_status(report_id: int, req: ReportStatusUpdate):
    try:
        engine.answer_cache.update_report_status(report_id, req.status)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok"}


@app.post("/admin/reports/{report_id}/analysis")
def update_report_analysis(report_id: int, req: ReportAnalysisUpdate):
    try:
        item = engine.answer_cache.update_report_analysis(
            report_id,
            status=req.status,
            issue_type=req.issue_type,
            resolution_type=req.resolution_type,
            admin_memo=req.admin_memo,
            linked_child_id=req.linked_child_id,
            resolved_qa_id=req.resolved_qa_id,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"status": "ok", "report": item}


@app.post("/admin/evidence/search")
def admin_evidence_search(req: EvidenceSearchRequest):
    corpus_ids = resolve_request_corpus_ids(req.corpus_ids)
    sources = engine.retriever.search(req.query, corpus_ids=corpus_ids, top_k=req.top_k)
    return {"sources": sources, "corpus_ids": corpus_ids}


@app.post("/admin/qa-cache")
def create_approved_qa(req: ApprovedQARequest):
    qa_id: int | None = None
    try:
        question_embedding = engine.retriever.embedding.embed([req.question])[0]
        qa_id = engine.answer_cache.create_approved_qa(
            question=req.question,
            answer=req.answer,
            question_embedding=question_embedding,
            evidence=req.evidence,
            corpus_version=req.corpus_version,
            index_version=req.index_version,
            approved_by=req.approved_by,
            source_report_id=req.source_report_id,
            memo=req.memo,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"embedding or cache registration failed: {exc}") from exc

    alias_result: dict = {"manual": None, "generated": None}
    alias_generation_error = None
    try:
        manual_aliases = [str(item).strip() for item in req.alias_texts if str(item).strip()]
        if manual_aliases:
            embeddings = _embed_texts(manual_aliases)
            alias_result["manual"] = engine.answer_cache.add_aliases(
                qa_id,
                [
                    {
                        "alias_text": alias,
                        "alias_type": "admin_alias",
                        "status": "active",
                        "embedding": embedding,
                        "memo": "created with approved QA",
                    }
                    for alias, embedding in zip(manual_aliases, embeddings)
                ],
            )
        if req.generate_aliases and engine.answer_cache.enable_alias_generation:
            alias_result["generated"] = _generate_aliases_for_qa(
                qa_id,
                max_aliases=engine.answer_cache.max_aliases_per_qa,
                dry_run=False,
            )
    except Exception as exc:
        alias_generation_error = str(exc)
    return {"qa_id": qa_id, "alias_result": alias_result, "alias_generation_error": alias_generation_error}


@app.get("/admin/qa-cache")
def recent_approved_qa(
    limit: int = Query(default=50, ge=1, le=500),
    status: str = Query(default="all", pattern="^(approved|disabled|all)$"),
):
    try:
        items = engine.answer_cache.recent_approved(limit=limit, status=status)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"items": items}


@app.get("/admin/qa-cache/summary")
def admin_qa_cache_summary(limit: int = Query(default=20, ge=1, le=100)):
    return engine.answer_cache.approved_summary(limit=limit)


@app.post("/admin/qa-cache/backfill-aliases")
def admin_qa_cache_backfill_aliases(req: QAAliasBackfillRequest):
    result = engine.answer_cache.backfill_aliases(
        ensure_original=req.ensure_original,
        generate_llm_aliases=req.generate_llm_aliases,
        only_without_llm_aliases=req.only_without_llm_aliases,
        limit=req.limit,
        dry_run=req.dry_run,
        corpus_version=req.corpus_version,
        index_version=req.index_version,
    )
    if req.dry_run or not req.generate_llm_aliases:
        return result
    stats = {"generated": 0, "created": 0, "errors": 0, "items": []}
    for item in result.get("items", []):
        try:
            generated = _generate_aliases_for_qa(
                int(item["qa_id"]),
                max_aliases=req.max_aliases_per_qa,
                dry_run=False,
            )
            stats["generated"] += 1
            stats["created"] += int(generated.get("created_count") or generated.get("inserted_count") or 0)
            stats["items"].append(generated)
        except Exception as exc:
            stats["errors"] += 1
            stats["items"].append({"qa_id": item.get("qa_id"), "error": str(exc)})
    result["llm_alias_generation"] = stats
    return result


@app.post("/admin/qa-cache/reload-index")
def admin_qa_cache_reload_index():
    return {"status": "ok", "alias_index": engine.answer_cache.reload_alias_index()}


@app.post("/admin/qa-cache/match-debug")
def admin_qa_cache_match_debug(req: QAMatchDebugRequest):
    try:
        corpus_ids = resolve_request_corpus_ids(req.corpus_ids)
        return engine.debug_cache_match(
            req.question,
            corpus_ids=corpus_ids,
            corpus_version=req.corpus_version,
            index_version=req.index_version,
            top_n=req.top_n,
            threshold=req.threshold,
            apply_llm_intent_judge=req.apply_llm_intent_judge,
            include_disabled=req.include_disabled,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"QA match debug failed: {exc}") from exc


@app.get("/admin/qa-cache/alias-conflicts")
def admin_qa_cache_alias_conflicts(limit: int = Query(default=100, ge=1, le=500)):
    return {"conflicts": engine.answer_cache.alias_conflicts(limit=limit)}


@app.patch("/admin/qa-cache/aliases/{alias_id}")
def admin_qa_cache_update_alias(alias_id: int, req: QAAliasUpdateRequest):
    try:
        embedding = None
        if req.alias_text is not None:
            embedding = _embed_texts([req.alias_text])[0]
        alias = engine.answer_cache.update_alias(
            alias_id,
            alias_text=req.alias_text,
            status=req.status,
            embedding=embedding,
            memo=req.memo,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"alias update failed: {exc}") from exc
    return {"status": "ok", "alias": alias, "alias_index": engine.answer_cache.alias_index_status()}


@app.post("/admin/qa-cache/similar")
def admin_qa_cache_similar(req: QASimilarRequest):
    try:
        corpus_ids = resolve_request_corpus_ids(req.corpus_ids)
        question_embedding = _embed_texts([req.question])[0]
        matches = engine.answer_cache.similar_approved(
            question_embedding,
            corpus_ids=corpus_ids,
            corpus_version=req.corpus_version,
            index_version=req.index_version,
            top_n=req.top_n,
            threshold=req.threshold,
            include_disabled=req.include_disabled_qa,
            include_disabled_aliases=req.include_disabled_aliases,
            question=req.question,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"similar QA check failed: {exc}") from exc
    return {"matches": matches}


@app.post("/admin/qa-cache/test-match")
def admin_qa_cache_test_match(req: QATestMatchRequest):
    try:
        corpus_ids = resolve_request_corpus_ids(req.corpus_ids)
        question_embedding = _embed_texts([req.question])[0]
        debug = engine.answer_cache.match_debug(
            question_embedding,
            question=req.question,
            corpus_ids=corpus_ids,
            corpus_version=req.corpus_version,
            index_version=req.index_version,
            threshold=req.threshold,
            include_disabled_qa=req.include_disabled,
            include_disabled_aliases=req.include_disabled,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"QA test match failed: {exc}") from exc
    return {"match": debug.get("best") if debug.get("decision") == "hit" else None, "debug": debug}


@app.get("/admin/qa-cache/{qa_id}/aliases")
def admin_qa_cache_aliases(qa_id: int):
    if engine.answer_cache.get_approved_qa(qa_id) is None:
        raise HTTPException(status_code=404, detail="approved QA not found")
    return {
        "qa_id": qa_id,
        "aliases": engine.answer_cache.list_aliases(qa_id),
        "alias_index": engine.answer_cache.alias_index_status(),
    }


@app.post("/admin/qa-cache/{qa_id}/aliases")
def admin_qa_cache_add_aliases(qa_id: int, req: QAAliasAddRequest):
    aliases = [str(item).strip() for item in req.aliases if str(item).strip()]
    try:
        embeddings = _embed_texts(aliases) if aliases else []
        result = engine.answer_cache.add_aliases(
            qa_id,
            [
                {
                    "alias_text": alias,
                    "alias_type": req.alias_type,
                    "status": req.status,
                    "embedding": embedding,
                    "memo": req.memo,
                    "force_active_conflict": req.force_active_conflict,
                }
                for alias, embedding in zip(aliases, embeddings)
            ],
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"alias add failed: {exc}") from exc
    return {"status": "ok", **result, "alias_index": engine.answer_cache.alias_index_status()}


@app.post("/admin/qa-cache/{qa_id}/aliases/generate")
def admin_qa_cache_generate_aliases(qa_id: int, req: QAAliasGenerateRequest):
    try:
        return _generate_aliases_for_qa(
            qa_id,
            max_aliases=req.max_aliases,
            replace_existing_generated=req.replace_existing_generated,
            dry_run=req.dry_run,
            status=req.status,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"alias generation failed: {exc}") from exc


@app.post("/admin/qa-cache/{qa_id}/disable")
def disable_approved_qa(qa_id: int):
    try:
        item = engine.answer_cache.disable_approved_qa(qa_id, memo="管理画面から無効化")
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"status": "ok", "item": item}


@app.get("/admin/qa-cache/{qa_id}")
def get_approved_qa(qa_id: int):
    item = engine.answer_cache.get_approved_qa(qa_id)
    if item is None:
        raise HTTPException(status_code=404, detail="approved QA not found")
    return item


@app.patch("/admin/qa-cache/{qa_id}")
def update_approved_qa(qa_id: int, req: ApprovedQAUpdateRequest):
    try:
        question_embedding = engine.retriever.embedding.embed([req.question])[0]
        item = engine.answer_cache.update_approved_qa(
            qa_id,
            question=req.question,
            answer=req.answer,
            question_embedding=question_embedding,
            evidence=req.evidence,
            status=req.status,
            corpus_version=req.corpus_version or getattr(engine.retriever, "corpus_version", None),
            index_version=req.index_version or getattr(engine.retriever, "index_version", None),
            approved_by=req.approved_by,
            memo=req.memo,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"embedding or cache update failed: {exc}") from exc
    return item


@app.get("/admin/eval/status")
def admin_eval_status():
    return eval_status_summary()


@app.get("/admin/eval/latest")
def admin_eval_latest():
    latest = latest_eval_result()
    if not latest.get("exists"):
        raise HTTPException(status_code=404, detail="eval result not found")
    return latest
