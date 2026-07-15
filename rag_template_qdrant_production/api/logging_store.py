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
        self.db_path = project_path(settings.get("sqlite_path", "logs/rag_chat_logs.sqlite"))
        if self.enabled:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ask_logs_session ON ask_logs(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ask_logs_created_at ON ask_logs(created_at)")

    def log_ask(
        self,
        *,
        session_id: str,
        question: str,
        answer: str,
        corpus_ids: Optional[list[str]],
        top_k: Optional[int],
        history: list[dict[str, str]],
        sources: list[dict[str, Any]],
        debug: Optional[dict[str, Any]],
    ) -> Optional[int]:
        if not self.enabled:
            return None

        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
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
                    debug_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    session_id,
                    question,
                    answer,
                    json.dumps(corpus_ids, ensure_ascii=False),
                    top_k,
                    json.dumps(history, ensure_ascii=False),
                    json.dumps(sources, ensure_ascii=False),
                    json.dumps(debug, ensure_ascii=False),
                ),
            )
            return int(cur.lastrowid)

    def recent(self, limit: int = 50, include_sources: bool = False) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        limit = max(1, min(limit, 500))
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT
                    id,
                    created_at,
                    session_id,
                    question,
                    answer,
                    corpus_ids_json,
                    top_k,
                    sources_json,
                    debug_json
                FROM ask_logs
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        items = []
        for row in rows:
            item = dict(row)
            sources = self._decode_json(item.get("sources_json"), [])
            item["corpus_ids"] = self._decode_json(item.get("corpus_ids_json"), [])
            item["source_count"] = len(sources) if isinstance(sources, list) else 0
            item["question_preview"] = self._preview(item.get("question"), 160)
            item["answer_preview"] = self._preview(item.get("answer"), 240)
            if include_sources:
                item["sources"] = sources if isinstance(sources, list) else []
                item["debug"] = self._decode_json(item.get("debug_json"), None)
            item.pop("sources_json", None)
            item.pop("debug_json", None)
            items.append(item)
        return items

    def dashboard(
        self,
        sample_limit: int = 1000,
        top_n: int = 20,
        days: int = 7,
        low_score_threshold: float = 0.35,
    ) -> dict[str, Any]:
        if not self.enabled:
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
                    "daily_quality": [],
                },
            }

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
                SELECT id, created_at, session_id, question, answer, sources_json, debug_json
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
            "logs_with_sources": int(overview.get("logs_with_sources") or 0),
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
        quality_by_date: dict[str, dict[str, Any]] = {}

        for row in rows:
            question = str(row["question"] or "").strip()
            if question:
                question_counter[question] += 1

            sources = self._decode_json(row["sources_json"], [])
            sources = sources if isinstance(sources, list) else []
            debug = self._decode_json(row["debug_json"], {})
            debug = debug if isinstance(debug, dict) else {}
            source_count = len(sources)
            scores = [self._safe_score(source.get("score")) for source in sources if isinstance(source, dict)]
            max_score = max(scores, default=0.0)
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
                "debug_reason": debug.get("reason"),
            }

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
            },
            "top_questions": [
                {"rank": idx, "question": question, "hits": hits}
                for idx, (question, hits) in enumerate(question_counter.most_common(top_n), start=1)
            ],
        }
