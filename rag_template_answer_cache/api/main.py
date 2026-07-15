# 承認済みQAキャッシュ対応RAG APIと管理者用APIをFastAPIで公開します。
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from api.config import load_settings, project_path
from api.schemas import (
    ApprovedQARequest,
    ApprovedQAUpdateRequest,
    AskRequest,
    AskResponse,
    EvidenceSearchRequest,
    HallucinationReportRequest,
    LogTrendReportRequest,
    ReportStatusUpdate,
    SearchTagUpdateRequest,
)
from api.rag_engine import RAGEngine

app = FastAPI(title="RAG Template API", version="1.0.0")
settings = load_settings()
api_settings = settings.get("api", {})

app.add_middleware(
    CORSMiddleware,
    allow_origins=api_settings.get("cors_allow_origins", ["http://localhost:8501", "http://127.0.0.1:8501"]),
    allow_credentials=bool(api_settings.get("allow_credentials", False)),
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = RAGEngine()


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
    tagged_count = sum(1 for row in rows if row.get("search_tags"))
    return {
        **file_info(relative_path),
        "row_count": len(rows),
        "tagged_count": tagged_count,
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
        markdown_count = len(list(markdown_path.rglob("*.md"))) if markdown_path and markdown_path.exists() else 0
        rows.append({
            "corpus_id": corpus.get("corpus_id"),
            "display_name": corpus.get("display_name"),
            "enabled": bool(corpus.get("enabled", True)),
            "priority": corpus.get("priority"),
            "markdown_dir": markdown_dir,
            "markdown_files": markdown_count,
            "description": corpus.get("description", ""),
        })
    return rows


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
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    result = engine.ask(
        question=req.question,
        corpus_ids=req.corpus_ids,
        top_k=req.top_k,
        show_debug=req.show_debug,
        session_id=req.session_id,
        history=[m.model_dump() for m in req.history],
    )
    return result


@app.get("/logs/recent")
def recent_logs(
    limit: int = Query(default=50, ge=1, le=500),
    include_sources: bool = Query(default=False),
):
    return {"logs": engine.log_store.recent(limit=limit, include_sources=include_sources)}


@app.get("/admin/logs/recent")
def admin_recent_logs(limit: int = Query(default=50, ge=1, le=500)):
    return {"logs": engine.log_store.recent(limit=limit, include_sources=True)}


@app.get("/admin/logs/dashboard")
def admin_log_dashboard(
    sample_limit: int = Query(default=1000, ge=1, le=5000),
    top_n: int = Query(default=20, ge=1, le=100),
    days: int = Query(default=7, ge=1, le=90),
    low_score_threshold: float = Query(default=0.35, ge=0.0, le=1.0),
):
    return engine.log_store.dashboard(
        sample_limit=sample_limit,
        top_n=top_n,
        days=days,
        low_score_threshold=low_score_threshold,
    )


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
        "forms": file_info("data/forms/form_catalog.csv"),
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


@app.patch("/admin/reports/{report_id}/status")
def update_report_status(report_id: int, req: ReportStatusUpdate):
    try:
        engine.answer_cache.update_report_status(report_id, req.status)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok"}


@app.post("/admin/evidence/search")
def admin_evidence_search(req: EvidenceSearchRequest):
    sources = engine.retriever.search(req.query, corpus_ids=req.corpus_ids, top_k=req.top_k)
    return {"sources": sources}


@app.post("/admin/qa-cache")
def create_approved_qa(req: ApprovedQARequest):
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
    return {"qa_id": qa_id}


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
