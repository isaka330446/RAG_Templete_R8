# 管理者承認済みQAキャッシュとハルシネーション報告をSQLiteで管理します。
import json
import math
import sqlite3
from datetime import datetime, timezone
from typing import Any, Optional

from api.config import load_settings, project_path


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


VALID_QA_STATUSES = {"approved", "disabled"}


class AnswerCacheStore:
    def __init__(self):
        settings = load_settings()
        cache_settings = settings.get("answer_cache", {})
        self.enabled = bool(cache_settings.get("enabled", True))
        self.sqlite_path = project_path(cache_settings.get("sqlite_path", "logs/answer_cache.sqlite"))
        self.high_similarity_threshold = float(cache_settings.get("high_similarity_threshold", 0.88))
        self.corpus_version = str(cache_settings.get("corpus_version", "default"))
        self.index_version = str(cache_settings.get("index_version", "default"))
        if self.enabled:
            self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()

    def _connect(self):
        return sqlite3.connect(self.sqlite_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS approved_qa (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    question_embedding_json TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    corpus_version TEXT NOT NULL,
                    index_version TEXT NOT NULL,
                    approved_by TEXT,
                    source_report_id INTEGER,
                    memo TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS hallucination_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    session_id TEXT,
                    log_id INTEGER,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    comment TEXT,
                    resolved_qa_id INTEGER
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_approved_qa_status ON approved_qa(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_approved_qa_version ON approved_qa(corpus_version, index_version)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reports_status ON hallucination_reports(status)")

    def find_match(
        self,
        query_embedding: list[float],
        corpus_ids: Optional[list[str]] = None,
        corpus_version: Optional[str] = None,
        index_version: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> Optional[dict[str, Any]]:
        if not self.enabled:
            return None
        if corpus_ids is not None and len(corpus_ids) == 0:
            return None

        corpus_version = corpus_version or self.corpus_version
        index_version = index_version or self.index_version
        threshold = self.high_similarity_threshold if threshold is None else threshold

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT *
                FROM approved_qa
                WHERE status = 'approved'
                  AND corpus_version = ?
                  AND index_version = ?
                """,
                (corpus_version, index_version),
            ).fetchall()

        best = None
        allowed = set(corpus_ids) if corpus_ids is not None else None
        for row in rows:
            item = dict(row)
            evidence = json.loads(item.get("evidence_json") or "[]")
            if allowed is not None and evidence:
                evidence_corpus_ids = {str(e.get("corpus_id")) for e in evidence if e.get("corpus_id")}
                if evidence_corpus_ids and evidence_corpus_ids.isdisjoint(allowed):
                    continue
            embedding = json.loads(item["question_embedding_json"])
            similarity = cosine_similarity(query_embedding, embedding)
            if best is None or similarity > best["similarity"]:
                item["similarity"] = similarity
                item["evidence"] = evidence
                best = item

        if best and best["similarity"] >= threshold:
            return best
        return None

    def create_approved_qa(
        self,
        *,
        question: str,
        answer: str,
        question_embedding: list[float],
        evidence: list[dict[str, Any]],
        corpus_version: Optional[str] = None,
        index_version: Optional[str] = None,
        approved_by: str = "",
        source_report_id: Optional[int] = None,
        memo: str = "",
    ) -> int:
        if not self.enabled:
            raise RuntimeError("answer cache is disabled")

        timestamp = now_utc()
        corpus_version = corpus_version or self.corpus_version
        index_version = index_version or self.index_version
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO approved_qa (
                    created_at,
                    updated_at,
                    status,
                    question,
                    answer,
                    question_embedding_json,
                    evidence_json,
                    corpus_version,
                    index_version,
                    approved_by,
                    source_report_id,
                    memo
                )
                VALUES (?, ?, 'approved', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    timestamp,
                    question,
                    answer,
                    json.dumps(question_embedding, ensure_ascii=False),
                    json.dumps(evidence, ensure_ascii=False),
                    corpus_version,
                    index_version,
                    approved_by,
                    source_report_id,
                    memo,
                ),
            )
            qa_id = int(cur.lastrowid)
            if source_report_id:
                conn.execute(
                    """
                    UPDATE hallucination_reports
                    SET status = 'resolved',
                        resolved_qa_id = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (qa_id, timestamp, source_report_id),
            )
            return qa_id

    def _row_to_approved_qa(self, row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        evidence = json.loads(item.pop("evidence_json", "[]") or "[]")
        item.pop("question_embedding_json", None)
        item["evidence"] = evidence
        item["evidence_count"] = len(evidence) if isinstance(evidence, list) else 0
        return item

    def get_approved_qa(self, qa_id: int) -> Optional[dict[str, Any]]:
        if not self.enabled:
            return None
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT *
                FROM approved_qa
                WHERE id = ?
                """,
                (qa_id,),
            ).fetchone()
        return self._row_to_approved_qa(row) if row else None

    def recent_approved(self, limit: int = 50, status: Optional[str] = None) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        limit = max(1, min(limit, 500))
        sql = """
            SELECT *
            FROM approved_qa
        """
        params: list[Any] = []
        if status and status != "all":
            if status not in VALID_QA_STATUSES:
                raise ValueError("invalid status")
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_approved_qa(row) for row in rows]

    def update_approved_qa(
        self,
        qa_id: int,
        *,
        question: str,
        answer: str,
        question_embedding: list[float],
        evidence: list[dict[str, Any]],
        status: str,
        corpus_version: Optional[str] = None,
        index_version: Optional[str] = None,
        approved_by: str = "",
        memo: str = "",
    ) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("answer cache is disabled")
        if status not in VALID_QA_STATUSES:
            raise ValueError("invalid status")

        timestamp = now_utc()
        corpus_version = corpus_version or self.corpus_version
        index_version = index_version or self.index_version
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE approved_qa
                SET updated_at = ?,
                    status = ?,
                    question = ?,
                    answer = ?,
                    question_embedding_json = ?,
                    evidence_json = ?,
                    corpus_version = ?,
                    index_version = ?,
                    approved_by = ?,
                    memo = ?
                WHERE id = ?
                """,
                (
                    timestamp,
                    status,
                    question,
                    answer,
                    json.dumps(question_embedding, ensure_ascii=False),
                    json.dumps(evidence, ensure_ascii=False),
                    corpus_version,
                    index_version,
                    approved_by,
                    memo,
                    qa_id,
                ),
            )
            if cur.rowcount == 0:
                raise ValueError("approved QA not found")

        item = self.get_approved_qa(qa_id)
        if item is None:
            raise ValueError("approved QA not found")
        return item

    def report_hallucination(
        self,
        *,
        question: str,
        answer: str,
        session_id: Optional[str] = None,
        log_id: Optional[int] = None,
        comment: str = "",
    ) -> int:
        if not self.enabled:
            raise RuntimeError("answer cache is disabled")

        timestamp = now_utc()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO hallucination_reports (
                    created_at,
                    updated_at,
                    status,
                    session_id,
                    log_id,
                    question,
                    answer,
                    comment
                )
                VALUES (?, ?, 'open', ?, ?, ?, ?, ?)
                """,
                (timestamp, timestamp, session_id, log_id, question, answer, comment),
            )
            return int(cur.lastrowid)

    def recent_reports(self, status: Optional[str] = None, limit: int = 50) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        limit = max(1, min(limit, 500))
        sql = """
            SELECT id, created_at, updated_at, status, session_id, log_id,
                   question, answer, comment, resolved_qa_id
            FROM hallucination_reports
        """
        params: list[Any] = []
        if status:
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def update_report_status(self, report_id: int, status: str) -> None:
        if status not in {"open", "resolved", "ignored"}:
            raise ValueError("invalid status")
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE hallucination_reports
                SET status = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, now_utc(), report_id),
            )
