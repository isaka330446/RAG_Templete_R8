# チャット質問、回答、根拠、デバッグ情報をSQLiteへ保存します。
import json
import sqlite3
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from api.config import load_settings, project_path


class SQLiteLogStore:
    def __init__(self):
        settings = load_settings().get("logging", {})
        self.enabled = bool(settings.get("enabled", True))
        self.required = bool(settings.get("required", True))
        self.db_path = project_path(settings.get("sqlite_path", "logs/rag_chat_logs.sqlite"))
        self.busy_timeout_ms = int(settings.get("busy_timeout_ms", 10000))
        self.connect_timeout_sec = int(settings.get("connect_timeout_sec", 30))
        self.configured_journal_mode = str(settings.get("journal_mode", "DELETE") or "DELETE").upper()
        self.startup_write_check = bool(settings.get("startup_write_check", True))
        self.max_source_preview_chars = int(settings.get("max_source_preview_chars", 500))
        self.store_full_source_text = bool(settings.get("store_full_source_text", False))
        self.last_error: str | None = None
        self.last_error_at: str | None = None
        self.journal_mode: str | None = None
        self.startup_check_result: str | None = None
        if self.enabled:
            try:
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
                self._init_db()
                if self.startup_write_check:
                    self._startup_write_check()
            except (sqlite3.Error, OSError) as exc:
                self._handle_error(exc, "SQLite startup check failed")

    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=self.connect_timeout_sec)
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            row = conn.execute(f"PRAGMA journal_mode={self.configured_journal_mode}").fetchone()
            self.journal_mode = str(row[0]).upper() if row else self.configured_journal_mode
        except sqlite3.Error:
            conn.close()
            raise
        return conn

    def _handle_error(self, exc: BaseException, context: str, *, raise_when_required: bool = True) -> None:
        self.last_error = str(exc)
        self.last_error_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        message = (
            f"{context}. db_path={self.db_path} "
            f"journal_mode={self.configured_journal_mode} error={exc}"
        )
        if self.required and raise_when_required:
            raise RuntimeError(message) from exc
        self.enabled = False
        print(f"[WARN] {message}")

    @staticmethod
    def _ensure_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        for name, definition in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")

    @staticmethod
    def _decode_json(value: Optional[str], default: Any) -> Any:
        if not value:
            return default
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default

    @staticmethod
    def _preview(value: Any, limit: int = 240) -> str:
        text = str(value or "").strip().replace("\r\n", "\n")
        if len(text) <= limit:
            return text
        return text[:limit] + "...[省略]"

    def compact_sources_for_log(self, sources: list[dict[str, Any]] | Any) -> list[dict[str, Any]]:
        if not isinstance(sources, list):
            return []
        compacted: list[dict[str, Any]] = []
        for source in sources:
            if not isinstance(source, dict):
                continue
            child_text = source.get("child_text") or source.get("text") or ""
            parent_text = source.get("parent_text") or ""
            item = {
                "corpus_id": source.get("corpus_id"),
                "parent_id": source.get("parent_id"),
                "child_id": source.get("child_id"),
                "source_file": source.get("source_file"),
                "source_url": source.get("source_url"),
                "title": source.get("title"),
                "heading_path": source.get("heading_path"),
                "document_code": source.get("document_code"),
                "document_type": source.get("document_type"),
                "document_id": source.get("document_id"),
                "score": source.get("score"),
                "child_text_preview": self._preview(child_text, self.max_source_preview_chars),
                "parent_text_preview": self._preview(parent_text, self.max_source_preview_chars),
            }
            if self.store_full_source_text:
                item["child_text"] = child_text
                item["parent_text"] = parent_text
            compacted.append(item)
        return compacted

    def _compact_debug_for_log(self, debug: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if not isinstance(debug, dict):
            return debug
        compact = dict(debug)
        for key in ["contexts", "context", "prompt", "messages", "parent_context", "retrieved_contexts"]:
            if key in compact:
                compact[key] = self._preview(compact.get(key), self.max_source_preview_chars)
        return compact

    @staticmethod
    def _safe_score(value: Any) -> float:
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _source_key(source: dict[str, Any]) -> str:
        corpus_id = str(source.get("corpus_id") or "")
        parent_id = str(source.get("parent_id") or "")
        child_id = str(source.get("child_id") or "")
        if child_id:
            return "|".join([corpus_id, parent_id, child_id])
        return "|".join([
            corpus_id,
            str(source.get("source_file") or ""),
            str(source.get("heading_path") or ""),
        ])

    def _fallback_metrics(
        self,
        *,
        sources: list[dict[str, Any]],
        debug: dict[str, Any],
        answer_source: Any = None,
        cache_hit: Any = None,
        qa_cache_id: Any = None,
        cache_similarity: Any = None,
        cache_candidate_qa_id: Any = None,
        cache_candidate_alias_id: Any = None,
        cache_candidate_similarity: Any = None,
        cache_miss_reason: Any = None,
        cache_match_method: Any = None,
        source_count: Any = None,
        max_score: Any = None,
    ) -> dict[str, Any]:
        scores = [self._safe_score(source.get("score")) for source in sources if isinstance(source, dict)]
        resolved_source_count = int(source_count) if source_count not in (None, "") else len(sources)
        resolved_max_score = self._safe_score(max_score) if max_score not in (None, "") else max(scores, default=0.0)
        resolved_answer_source = str(answer_source or debug.get("answer_source") or "rag")
        resolved_cache_hit = bool(cache_hit) if cache_hit is not None else resolved_answer_source == "approved_qa_cache"
        return {
            "answer_source": resolved_answer_source,
            "cache_hit": resolved_cache_hit,
            "qa_cache_id": qa_cache_id if qa_cache_id not in (None, "") else debug.get("qa_cache_id"),
            "cache_similarity": cache_similarity if cache_similarity not in (None, "") else debug.get("cache_similarity"),
            "cache_candidate_qa_id": cache_candidate_qa_id if cache_candidate_qa_id not in (None, "") else debug.get("cache_candidate_qa_id"),
            "cache_candidate_alias_id": cache_candidate_alias_id if cache_candidate_alias_id not in (None, "") else debug.get("cache_candidate_alias_id"),
            "cache_candidate_similarity": cache_candidate_similarity if cache_candidate_similarity not in (None, "") else debug.get("cache_candidate_similarity"),
            "cache_miss_reason": cache_miss_reason if cache_miss_reason not in (None, "") else debug.get("cache_miss_reason"),
            "cache_match_method": cache_match_method if cache_match_method not in (None, "") else debug.get("cache_match_method"),
            "source_count": resolved_source_count,
            "max_score": round(resolved_max_score, 6),
        }

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ask_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    corpus_ids_json TEXT,
                    top_k INTEGER,
                    history_json TEXT,
                    sources_json TEXT,
                    debug_json TEXT
                )
                """
            )
            self._ensure_columns(
                conn,
                "ask_logs",
                {
                    "answer_source": "TEXT",
                    "cache_hit": "INTEGER",
                    "qa_cache_id": "INTEGER",
                    "cache_similarity": "REAL",
                    "source_count": "INTEGER",
                    "max_score": "REAL",
                    "latency_ms": "INTEGER",
                    "error_type": "TEXT",
                    "cache_candidate_qa_id": "INTEGER",
                    "cache_candidate_alias_id": "INTEGER",
                    "cache_candidate_similarity": "REAL",
                    "cache_miss_reason": "TEXT",
                    "cache_match_method": "TEXT",
                },
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ask_logs_session ON ask_logs(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ask_logs_created_at ON ask_logs(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ask_logs_answer_source ON ask_logs(answer_source)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS app_startup_check (
                    id INTEGER PRIMARY KEY,
                    checked_at TEXT NOT NULL
                )
                """
            )

    def _startup_write_check(self) -> None:
        checked_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO app_startup_check (id, checked_at)
                VALUES (1, ?)
                """,
                (checked_at,),
            )
            conn.execute("DELETE FROM app_startup_check WHERE id = 1")
            conn.commit()
        self.startup_check_result = "ok"

    def log_ask(
        self,
        *,
        session_id: str,
        question: str,
        answer: str,
        corpus_ids: Optional[list[str]] = None,
        top_k: Optional[int] = None,
        history: Optional[list[dict[str, str]]] = None,
        sources: Optional[list[dict[str, Any]]] = None,
        debug: Optional[dict[str, Any]] = None,
        answer_source: str = "rag",
        cache_hit: bool = False,
        qa_cache_id: Optional[int] = None,
        cache_similarity: Optional[float] = None,
        cache_candidate_qa_id: Optional[int] = None,
        cache_candidate_alias_id: Optional[int] = None,
        cache_candidate_similarity: Optional[float] = None,
        cache_miss_reason: Optional[str] = None,
        cache_match_method: Optional[str] = None,
        latency_ms: Optional[int] = None,
        error_type: Optional[str] = None,
    ) -> Optional[int]:
        if not self.enabled:
            return None

        try:
            created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            source_rows = sources if isinstance(sources, list) else []
            compact_sources = self.compact_sources_for_log(source_rows)
            compact_debug = self._compact_debug_for_log(debug)
            debug_dict = debug if isinstance(debug, dict) else {}
            history_rows = history if isinstance(history, list) else []
            metrics = self._fallback_metrics(
                sources=source_rows,
                debug=debug_dict,
                answer_source=answer_source,
                cache_hit=cache_hit,
                qa_cache_id=qa_cache_id,
                cache_similarity=cache_similarity,
                cache_candidate_qa_id=cache_candidate_qa_id,
                cache_candidate_alias_id=cache_candidate_alias_id,
                cache_candidate_similarity=cache_candidate_similarity,
                cache_miss_reason=cache_miss_reason,
                cache_match_method=cache_match_method,
            )
            with self._connect() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO ask_logs (
                        created_at,
                        session_id,
                        question,
                        answer,
                        corpus_ids_json,
                        top_k,
                        history_json,
                        sources_json,
                        debug_json,
                        answer_source,
                        cache_hit,
                        qa_cache_id,
                        cache_similarity,
                        source_count,
                        max_score,
                        latency_ms,
                        error_type,
                        cache_candidate_qa_id,
                        cache_candidate_alias_id,
                        cache_candidate_similarity,
                        cache_miss_reason,
                        cache_match_method
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        created_at,
                        session_id,
                        question,
                        answer,
                        json.dumps(corpus_ids, ensure_ascii=False),
                        top_k,
                        json.dumps(history_rows, ensure_ascii=False),
                        json.dumps(compact_sources, ensure_ascii=False),
                        json.dumps(compact_debug, ensure_ascii=False),
                        metrics["answer_source"],
                        1 if metrics["cache_hit"] else 0,
                        metrics["qa_cache_id"],
                        metrics["cache_similarity"],
                        metrics["source_count"],
                        metrics["max_score"],
                        latency_ms,
                        error_type,
                        metrics["cache_candidate_qa_id"],
                        metrics["cache_candidate_alias_id"],
                        metrics["cache_candidate_similarity"],
                        metrics["cache_miss_reason"],
                        metrics["cache_match_method"],
                    ),
                )
                return int(cur.lastrowid)
        except (sqlite3.Error, OSError) as exc:
            self._handle_error(exc, "chat log write failed")
            return None

    def recent(self, limit: int = 50, include_sources: bool = False) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        limit = max(1, min(limit, 500))
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT *
                    FROM ask_logs
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        except (sqlite3.Error, OSError) as exc:
            self._handle_error(exc, "chat log recent read failed", raise_when_required=False)
            return []
        items = []
        for row in rows:
            item = dict(row)
            sources = self._decode_json(item.get("sources_json"), []) if include_sources else []
            debug = self._decode_json(item.get("debug_json"), {}) if include_sources else {}
            debug = debug if isinstance(debug, dict) else {}
            metrics = self._fallback_metrics(
                sources=sources if isinstance(sources, list) else [],
                debug=debug,
                answer_source=item.get("answer_source"),
                cache_hit=item.get("cache_hit"),
                qa_cache_id=item.get("qa_cache_id"),
                cache_similarity=item.get("cache_similarity"),
                cache_candidate_qa_id=item.get("cache_candidate_qa_id"),
                cache_candidate_alias_id=item.get("cache_candidate_alias_id"),
                cache_candidate_similarity=item.get("cache_candidate_similarity"),
                cache_miss_reason=item.get("cache_miss_reason"),
                cache_match_method=item.get("cache_match_method"),
                source_count=item.get("source_count"),
                max_score=item.get("max_score"),
            )
            item["corpus_ids"] = self._decode_json(item.get("corpus_ids_json"), [])
            item.update(metrics)
            item["question_preview"] = self._preview(item.get("question"), 160)
            item["answer_preview"] = self._preview(item.get("answer"), 240)
            if include_sources:
                item["sources"] = sources if isinstance(sources, list) else []
                item["debug"] = debug
            item.pop("sources_json", None)
            item.pop("debug_json", None)
            items.append(item)
        return items

    def search(
        self,
        *,
        limit: int = 100,
        days: int = 30,
        query: Optional[str] = None,
        corpus_id: Optional[str] = None,
        source_state: str = "all",
        low_score_threshold: float = 0.35,
        answer_source: Optional[str] = None,
        qa_cache_id: Optional[int] = None,
        session_id: Optional[str] = None,
        log_id: Optional[int] = None,
        min_score: Optional[float] = None,
        max_score_filter: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        limit = max(1, min(limit, 500))
        days = max(1, min(days, 365))
        low_score_threshold = max(0.0, min(float(low_score_threshold), 1.0))
        source_state = source_state if source_state in {"all", "with_sources", "no_sources", "low_confidence"} else "all"
        query_text = str(query or "").strip().casefold()
        corpus_filter = str(corpus_id or "").strip()
        answer_source_filter = str(answer_source or "").strip()
        session_filter = str(session_id or "").strip()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")

        try:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT *
                    FROM ask_logs
                    WHERE created_at >= ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (cutoff, min(5000, max(limit * 8, limit))),
                ).fetchall()
        except (sqlite3.Error, OSError) as exc:
            self._handle_error(exc, "chat log search read failed", raise_when_required=False)
            return []

        items = []
        for row in rows:
            item = dict(row)
            sources = self._decode_json(item.get("sources_json"), [])
            sources = sources if isinstance(sources, list) else []
            corpus_ids = self._decode_json(item.get("corpus_ids_json"), [])
            corpus_ids = corpus_ids if isinstance(corpus_ids, list) else []
            debug = self._decode_json(item.get("debug_json"), {})
            debug = debug if isinstance(debug, dict) else {}
            scores = [self._safe_score(source.get("score")) for source in sources if isinstance(source, dict)]
            fallback = self._fallback_metrics(
                sources=sources,
                debug=debug,
                answer_source=item.get("answer_source"),
                cache_hit=item.get("cache_hit"),
                qa_cache_id=item.get("qa_cache_id"),
                cache_similarity=item.get("cache_similarity"),
                cache_candidate_qa_id=item.get("cache_candidate_qa_id"),
                cache_candidate_alias_id=item.get("cache_candidate_alias_id"),
                cache_candidate_similarity=item.get("cache_candidate_similarity"),
                cache_miss_reason=item.get("cache_miss_reason"),
                cache_match_method=item.get("cache_match_method"),
                source_count=item.get("source_count"),
                max_score=item.get("max_score"),
            )
            max_score = float(fallback["max_score"])
            source_count = int(fallback["source_count"])

            haystack = "\n".join([
                str(item.get("question") or ""),
                str(item.get("answer") or ""),
                " ".join(str(value) for value in corpus_ids),
                "\n".join(str(source.get("heading_path") or source.get("source_file") or "") for source in sources if isinstance(source, dict)),
            ]).casefold()
            if log_id is not None and int(item.get("id") or 0) != int(log_id):
                continue
            if session_filter and session_filter != str(item.get("session_id") or ""):
                continue
            if answer_source_filter and answer_source_filter != str(fallback.get("answer_source") or ""):
                continue
            if qa_cache_id is not None and str(item.get("qa_cache_id") or fallback.get("qa_cache_id") or "") != str(qa_cache_id):
                continue
            if min_score is not None and max_score < float(min_score):
                continue
            if max_score_filter is not None and max_score > float(max_score_filter):
                continue
            if query_text and query_text not in haystack:
                continue
            if corpus_filter:
                source_corpus_ids = {str(source.get("corpus_id") or "") for source in sources if isinstance(source, dict)}
                request_corpus_ids = {str(value) for value in corpus_ids}
                if corpus_filter not in source_corpus_ids and corpus_filter not in request_corpus_ids:
                    continue
            if source_state == "with_sources" and source_count == 0:
                continue
            if source_state == "no_sources" and source_count > 0:
                continue
            if source_state == "low_confidence" and not (source_count > 0 and max_score < low_score_threshold):
                continue

            item["corpus_ids"] = corpus_ids
            item.update(fallback)
            item["quality_state"] = "no_sources" if source_count == 0 else "low_confidence" if max_score < low_score_threshold else "normal"
            item["question_preview"] = self._preview(item.get("question"), 180)
            item["answer_preview"] = self._preview(item.get("answer"), 260)
            item["debug"] = debug
            item["sources"] = sources
            item.pop("sources_json", None)
            item.pop("debug_json", None)
            items.append(item)
            if len(items) >= limit:
                break
        return items

    def _empty_dashboard(self, low_score_threshold: float = 0.35) -> dict[str, Any]:
        return {
            "overview": {
                "enabled": False,
                "total_questions": 0,
                "unique_sessions": 0,
                "window_questions": 0,
                "window_sessions": 0,
                "logs_with_sources": 0,
                "last_24h_questions": 0,
                "last_7d_questions": 0,
                "first_log_at": None,
                "last_log_at": None,
                "qa_cache_hits": 0,
                "rag_answers": 0,
                "error_count": 0,
                "avg_max_score": 0.0,
                "p50_latency_ms": None,
                "p95_latency_ms": None,
                "last_error": self.last_error,
                "last_error_at": self.last_error_at,
            },
            "sample": {"sample_limit": 0, "sampled_questions": 0, "sampled_no_source_questions": 0},
            "top_hit_chunks": [],
            "top_source_files": [],
            "top_questions": [],
            "daily_questions": [],
            "hourly_questions": [],
            "quality": {
                "low_score_threshold": low_score_threshold,
                "no_hit_count": 0,
                "low_confidence_count": 0,
                "no_hit_logs": [],
                "low_confidence_logs": [],
                "near_cache_miss_logs": [],
                "daily_quality": [],
            },
        }

    def dashboard(
        self,
        sample_limit: int = 1000,
        top_n: int = 20,
        days: int = 7,
        low_score_threshold: float = 0.35,
    ) -> dict[str, Any]:
        try:
            return self._dashboard_impl(
                sample_limit=sample_limit,
                top_n=top_n,
                days=days,
                low_score_threshold=low_score_threshold,
            )
        except (sqlite3.Error, OSError) as exc:
            self._handle_error(exc, "chat log dashboard read failed", raise_when_required=False)
            return self._empty_dashboard(low_score_threshold=low_score_threshold)

    def _dashboard_impl(
        self,
        sample_limit: int = 1000,
        top_n: int = 20,
        days: int = 7,
        low_score_threshold: float = 0.35,
    ) -> dict[str, Any]:
        if not self.enabled:
            return self._empty_dashboard(low_score_threshold=low_score_threshold)

        sample_limit = max(1, min(sample_limit, 5000))
        top_n = max(1, min(top_n, 100))
        days = max(1, min(days, 90))
        low_score_threshold = max(0.0, min(float(low_score_threshold), 1.0))
        now = datetime.now(timezone.utc)
        cutoff_24h = (now - timedelta(hours=24)).isoformat(timespec="seconds")
        cutoff_7d = (now - timedelta(days=7)).isoformat(timespec="seconds")
        cutoff_window = (now - timedelta(days=days)).isoformat(timespec="seconds")

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            overview_row = conn.execute(
                """
                SELECT
                    COUNT(*) AS total_questions,
                    COUNT(DISTINCT session_id) AS unique_sessions,
                    SUM(CASE WHEN sources_json IS NOT NULL AND sources_json NOT IN ('', '[]', 'null') THEN 1 ELSE 0 END) AS logs_with_sources,
                    SUM(CASE WHEN cache_hit = 1 THEN 1 ELSE 0 END) AS qa_cache_hits,
                    SUM(CASE WHEN answer_source IS NULL OR answer_source = 'rag' THEN 1 ELSE 0 END) AS rag_answers,
                    SUM(CASE WHEN error_type IS NOT NULL AND error_type != '' THEN 1 ELSE 0 END) AS error_count,
                    AVG(CASE WHEN max_score IS NOT NULL THEN max_score ELSE NULL END) AS avg_max_score,
                    MIN(created_at) AS first_log_at,
                    MAX(created_at) AS last_log_at
                FROM ask_logs
                """
            ).fetchone()
            last_24h = conn.execute("SELECT COUNT(*) FROM ask_logs WHERE created_at >= ?", (cutoff_24h,)).fetchone()[0]
            last_7d = conn.execute("SELECT COUNT(*) FROM ask_logs WHERE created_at >= ?", (cutoff_7d,)).fetchone()[0]
            window_questions = conn.execute("SELECT COUNT(*) FROM ask_logs WHERE created_at >= ?", (cutoff_window,)).fetchone()[0]
            window_sessions = conn.execute(
                "SELECT COUNT(DISTINCT session_id) FROM ask_logs WHERE created_at >= ?",
                (cutoff_window,),
            ).fetchone()[0]
            window_cache_hits = conn.execute(
                "SELECT COUNT(*) FROM ask_logs WHERE created_at >= ? AND cache_hit = 1",
                (cutoff_window,),
            ).fetchone()[0]
            window_rag_answers = conn.execute(
                "SELECT COUNT(*) FROM ask_logs WHERE created_at >= ? AND (answer_source IS NULL OR answer_source = 'rag')",
                (cutoff_window,),
            ).fetchone()[0]
            daily_rows = conn.execute(
                """
                SELECT substr(created_at, 1, 10) AS date, COUNT(*) AS questions
                FROM ask_logs
                WHERE created_at >= ?
                GROUP BY substr(created_at, 1, 10)
                ORDER BY date
                """,
                (cutoff_window,),
            ).fetchall()
            hourly_rows = conn.execute(
                """
                SELECT substr(created_at, 1, 13) || ':00' AS hour, COUNT(*) AS questions
                FROM ask_logs
                WHERE created_at >= ?
                GROUP BY substr(created_at, 1, 13)
                ORDER BY hour
                """,
                (cutoff_24h,),
            ).fetchall()
            rows = conn.execute(
                """
                SELECT id, created_at, session_id, question, answer, sources_json, debug_json,
                       answer_source, cache_hit, qa_cache_id, cache_similarity, source_count, max_score,
                       latency_ms, error_type,
                       cache_candidate_qa_id, cache_candidate_alias_id, cache_candidate_similarity,
                       cache_miss_reason, cache_match_method
                FROM ask_logs
                WHERE created_at >= ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (cutoff_window, sample_limit),
            ).fetchall()

        overview = dict(overview_row or {})
        overview.update({
            "enabled": True,
            "total_questions": int(overview.get("total_questions") or 0),
            "unique_sessions": int(overview.get("unique_sessions") or 0),
            "window_questions": int(window_questions or 0),
            "window_sessions": int(window_sessions or 0),
            "window_cache_hits": int(window_cache_hits or 0),
            "window_rag_answers": int(window_rag_answers or 0),
            "logs_with_sources": int(overview.get("logs_with_sources") or 0),
            "qa_cache_hits": int(overview.get("qa_cache_hits") or 0),
            "rag_answers": int(overview.get("rag_answers") or 0),
            "error_count": int(overview.get("error_count") or 0),
            "avg_max_score": round(float(overview.get("avg_max_score") or 0.0), 6),
            "last_24h_questions": int(last_24h or 0),
            "last_7d_questions": int(last_7d or 0),
        })
        daily_counts = {row["date"]: int(row["questions"] or 0) for row in daily_rows}
        daily_questions = []
        for offset in range(days - 1, -1, -1):
            day = (now - timedelta(days=offset)).date().isoformat()
            daily_questions.append({"date": day, "questions": daily_counts.get(day, 0)})
        hourly_questions = [{"hour": row["hour"], "questions": int(row["questions"] or 0)} for row in hourly_rows]

        chunk_stats: dict[str, dict[str, Any]] = {}
        source_file_stats: dict[str, dict[str, Any]] = {}
        question_counter: Counter[str] = Counter()
        sampled_no_source_questions = 0
        no_hit_logs: list[dict[str, Any]] = []
        low_confidence_logs: list[dict[str, Any]] = []
        low_confidence_candidates: list[dict[str, Any]] = []
        near_cache_miss_logs: list[dict[str, Any]] = []
        quality_by_date: dict[str, dict[str, Any]] = {}

        for row in rows:
            question = str(row["question"] or "").strip()
            if question:
                question_counter[question] += 1

            sources = self._decode_json(row["sources_json"], [])
            sources = sources if isinstance(sources, list) else []
            debug = self._decode_json(row["debug_json"], {})
            debug = debug if isinstance(debug, dict) else {}
            metrics = self._fallback_metrics(
                sources=sources,
                debug=debug,
                answer_source=row["answer_source"],
                cache_hit=row["cache_hit"],
                qa_cache_id=row["qa_cache_id"],
                cache_similarity=row["cache_similarity"],
                cache_candidate_qa_id=row["cache_candidate_qa_id"],
                cache_candidate_alias_id=row["cache_candidate_alias_id"],
                cache_candidate_similarity=row["cache_candidate_similarity"],
                cache_miss_reason=row["cache_miss_reason"],
                cache_match_method=row["cache_match_method"],
                source_count=row["source_count"],
                max_score=row["max_score"],
            )
            source_count = int(metrics["source_count"])
            max_score = float(metrics["max_score"])
            log_day = str(row["created_at"] or "")[:10]
            if log_day:
                quality = quality_by_date.setdefault(
                    log_day,
                    {"date": log_day, "questions": 0, "no_hit": 0, "low_confidence": 0},
                )
                quality["questions"] += 1

            log_summary = {
                "id": row["id"],
                "created_at": row["created_at"],
                "session_id": row["session_id"],
                "question": row["question"],
                "question_preview": self._preview(row["question"], 180),
                "answer_preview": self._preview(row["answer"], 220),
                "source_count": source_count,
                "max_score": round(max_score, 6),
                "answer_source": metrics["answer_source"],
                "cache_hit": metrics["cache_hit"],
                "qa_cache_id": metrics["qa_cache_id"],
                "cache_similarity": metrics["cache_similarity"],
                "cache_candidate_qa_id": metrics["cache_candidate_qa_id"],
                "cache_candidate_alias_id": metrics["cache_candidate_alias_id"],
                "cache_candidate_similarity": metrics["cache_candidate_similarity"],
                "cache_miss_reason": metrics["cache_miss_reason"],
                "cache_match_method": metrics["cache_match_method"],
                "debug_reason": debug.get("reason"),
            }

            if (
                not metrics["cache_hit"]
                and metrics.get("cache_candidate_similarity") is not None
                and str(metrics.get("cache_miss_reason") or "") in {"below_accept_threshold", "margin_too_small", "llm_judge_rejected"}
                and len(near_cache_miss_logs) < top_n
            ):
                near_cache_miss_logs.append(log_summary)

            if source_count == 0:
                sampled_no_source_questions += 1
                if log_day:
                    quality_by_date[log_day]["no_hit"] += 1
                if len(no_hit_logs) < top_n:
                    no_hit_logs.append(log_summary)
                continue

            if max_score < low_score_threshold:
                if log_day:
                    quality_by_date[log_day]["low_confidence"] += 1
                low_confidence_candidates.append(log_summary)

            for source in sources:
                if not isinstance(source, dict):
                    continue
                score = self._safe_score(source.get("score"))
                key = self._source_key(source)
                stat = chunk_stats.setdefault(
                    key,
                    {
                        "corpus_id": source.get("corpus_id"),
                        "parent_id": source.get("parent_id"),
                        "child_id": source.get("child_id"),
                        "title": source.get("title"),
                        "heading_path": source.get("heading_path"),
                        "source_file": source.get("source_file"),
                        "source_url": source.get("source_url"),
                        "document_type": source.get("document_type"),
                        "hits": 0,
                        "score_sum": 0.0,
                        "max_score": 0.0,
                        "last_hit_at": row["created_at"],
                        "sample_log_ids": [],
                        "child_text_preview": self._preview(source.get("child_text") or source.get("text"), 320),
                    },
                )
                stat["hits"] += 1
                stat["score_sum"] += score
                stat["max_score"] = max(float(stat.get("max_score") or 0.0), score)
                if str(row["created_at"]) > str(stat.get("last_hit_at") or ""):
                    stat["last_hit_at"] = row["created_at"]
                if len(stat["sample_log_ids"]) < 5:
                    stat["sample_log_ids"].append(row["id"])

                file_key = str(source.get("source_file") or source.get("source_url") or source.get("corpus_id") or "unknown")
                file_stat = source_file_stats.setdefault(
                    file_key,
                    {
                        "source_file": source.get("source_file"),
                        "source_url": source.get("source_url"),
                        "corpus_id": source.get("corpus_id"),
                        "hits": 0,
                        "last_hit_at": row["created_at"],
                    },
                )
                file_stat["hits"] += 1
                if str(row["created_at"]) > str(file_stat.get("last_hit_at") or ""):
                    file_stat["last_hit_at"] = row["created_at"]

        top_hit_chunks = sorted(chunk_stats.values(), key=lambda item: item["hits"], reverse=True)[:top_n]
        for rank, item in enumerate(top_hit_chunks, start=1):
            item["rank"] = rank
            item["avg_score"] = round(float(item.pop("score_sum", 0.0)) / max(1, int(item["hits"])), 6)
            item["max_score"] = round(float(item.get("max_score") or 0.0), 6)

        top_source_files = sorted(source_file_stats.values(), key=lambda item: item["hits"], reverse=True)[:top_n]
        for rank, item in enumerate(top_source_files, start=1):
            item["rank"] = rank
        low_confidence_logs = sorted(low_confidence_candidates, key=lambda item: item["max_score"])[:top_n]
        daily_quality = []
        for row in daily_questions:
            day = row["date"]
            quality = quality_by_date.get(day, {"date": day, "questions": 0, "no_hit": 0, "low_confidence": 0})
            daily_quality.append(quality)

        return {
            "overview": overview,
            "sample": {
                "days": days,
                "sample_limit": sample_limit,
                "sampled_questions": len(rows),
                "sampled_no_source_questions": sampled_no_source_questions,
            },
            "top_hit_chunks": top_hit_chunks,
            "top_source_files": top_source_files,
            "daily_questions": daily_questions,
            "hourly_questions": hourly_questions,
            "quality": {
                "low_score_threshold": low_score_threshold,
                "no_hit_count": sampled_no_source_questions,
                "low_confidence_count": len(low_confidence_candidates),
                "no_hit_logs": no_hit_logs,
                "low_confidence_logs": low_confidence_logs,
                "daily_quality": daily_quality,
                "near_cache_miss_logs": near_cache_miss_logs,
            },
            "top_questions": [
                {"rank": idx, "question": question, "hits": hits}
                for idx, (question, hits) in enumerate(question_counter.most_common(top_n), start=1)
            ],
        }

    def status(self) -> dict[str, Any]:
        exists = self.db_path.exists()
        status: dict[str, Any] = {
            "enabled": self.enabled,
            "required": self.required,
            "db_path": str(self.db_path),
            "exists": exists,
            "size_bytes": self.db_path.stat().st_size if exists else 0,
            "wal_exists": self.db_path.with_name(self.db_path.name + "-wal").exists(),
            "shm_exists": self.db_path.with_name(self.db_path.name + "-shm").exists(),
            "last_error": self.last_error,
            "last_error_at": self.last_error_at,
            "journal_mode": self.journal_mode or self.configured_journal_mode,
            "configured_journal_mode": self.configured_journal_mode,
            "startup_check_result": self.startup_check_result,
            "row_count": None,
            "writable": False,
        }
        if not self.enabled:
            return status
        try:
            with self._connect() as conn:
                row = conn.execute("SELECT COUNT(*) FROM ask_logs").fetchone()
                status["row_count"] = int(row[0]) if row else 0
                status["journal_mode"] = self.journal_mode
                status["writable"] = True
        except (sqlite3.Error, OSError) as exc:
            self._handle_error(exc, "chat log status check failed", raise_when_required=False)
            status["enabled"] = False
            status["last_error"] = self.last_error
            status["last_error_at"] = self.last_error_at
            status["writable"] = False
        return status
