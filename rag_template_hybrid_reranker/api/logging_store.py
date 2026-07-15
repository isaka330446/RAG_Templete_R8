# チャット質問、回答、根拠、デバッグ情報をSQLiteへ保存します。
import json
import sqlite3
from datetime import datetime, timezone
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

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        limit = max(1, min(limit, 500))
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, created_at, session_id, question, answer, corpus_ids_json, top_k
                FROM ask_logs
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
