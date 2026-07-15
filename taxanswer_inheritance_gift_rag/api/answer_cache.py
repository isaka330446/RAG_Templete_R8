from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
import unicodedata
from datetime import datetime, timezone
from typing import Any, Optional

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal admin environments
    np = None

from api.config import load_settings, project_path


VALID_QA_STATUSES = {"approved", "disabled"}
VALID_ALIAS_STATUSES = {"active", "disabled"}
VALID_ALIAS_TYPES = {"original", "admin_alias", "llm_paraphrase", "normalized"}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_qa_text(text: str) -> str:
    """Lightweight normalization for approved-QA alias lookup."""
    value = unicodedata.normalize("NFKC", str(text or "")).strip().casefold()
    value = value.replace("\u3000", " ")
    value = re.sub(r"[‐-‒–—―ーｰ]+", "-", value)
    value = re.sub(r"\bno\s*\.?\s*([0-9]+)", r"no\1", value)
    value = re.sub(r"タックス\s*アンサー\s*([0-9]+)", r"タックスアンサー\1", value)
    value = re.sub(r"[ \t\r\n]+", " ", value)
    value = re.sub(r"\s*([。、,.!?！？:：;；/／()（）［］\[\]{}<>＜＞])\s*", r"\1", value)
    return value.strip()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def safe_json_loads(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def parse_json_object_from_llm(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a local LLM response.

    The cache path must fail closed, so invalid JSON returns an empty dict.
    """
    raw = str(text or "").strip()
    if not raw:
        return {}
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", raw):
        try:
            payload, _ = decoder.raw_decode(raw[match.start():])
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            continue
    return {}


def extract_taxanswer_numbers(text: str) -> set[str]:
    value = unicodedata.normalize("NFKC", str(text or ""))
    patterns = [
        r"No\.?\s*([0-9]{3,5})",
        r"タックス\s*アンサー\s*([0-9]{3,5})",
        r"コード\s*([0-9]{3,5})",
    ]
    found: set[str] = set()
    for pattern in patterns:
        found.update(re.findall(pattern, value, flags=re.IGNORECASE))
    return found


def detect_alias_risk_flags(question: str, alias_text: str, evidence: list[dict[str, Any]] | None = None) -> list[str]:
    """Small mechanical guardrails for generated alias candidates."""
    source = str(question or "")
    alias = str(alias_text or "")
    flags: list[str] = []
    if "相続税" in source and "贈与税" in alias and "贈与税" not in source:
        flags.append("tax_type_changed_to_gift_tax")
    if "贈与税" in source and "相続税" in alias and "相続税" not in source:
        flags.append("tax_type_changed_to_inheritance_tax")
    source_numbers = extract_taxanswer_numbers(source)
    for item in evidence or []:
        if isinstance(item, dict):
            source_numbers.update(extract_taxanswer_numbers(item.get("taxanswer_no") or ""))
            source_numbers.update(extract_taxanswer_numbers(item.get("title") or ""))
    alias_numbers = extract_taxanswer_numbers(alias)
    if source_numbers and alias_numbers and source_numbers.isdisjoint(alias_numbers):
        flags.append("taxanswer_no_changed")
    risky_words = ["個別", "具体的", "ケース", "判断", "令和", "年度", "改正"]
    if any(word in alias for word in risky_words) and not any(word in source for word in risky_words):
        flags.append("scope_or_time_condition_added")
    return flags


def evidence_titles(evidence: list[dict[str, Any]], limit: int = 8) -> str:
    titles: list[str] = []
    for item in evidence[:limit]:
        title = item.get("title") or item.get("heading_path") or item.get("source_file") or item.get("source_url")
        if title:
            titles.append(str(title))
    return "\n".join(f"- {title}" for title in titles)


def build_alias_generation_prompt(question: str, answer: str, evidence: list[dict[str, Any]], max_aliases: int) -> str:
    return f"""あなたは税務RAGの承認済みQAキャッシュを管理する補助者です。
次の承認済みQAと「同じ回答をそのまま返して安全」な日本語の別名質問だけを生成してください。

禁止:
- 税目を変えること
- 制度を変えること
- 条件を追加すること
- 個別判断が必要な質問にすること
- 回答範囲を広げること
- 年度や改正時点を勝手に追加すること
- 根拠にない論点を混ぜること

出力はJSONのみ。
形式:
{{"aliases": ["...", "..."]}}
最大 {max_aliases} 件。

正式質問:
{question}

承認済み回答:
{answer}

根拠の見出し・タイトル:
{evidence_titles(evidence)}
"""


def parse_alias_generation_response(text: str) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    payload = parse_json_object_from_llm(raw)
    aliases = payload.get("aliases") if isinstance(payload, dict) else None
    if isinstance(aliases, list):
        return [str(item) for item in aliases]

    recovered: list[str] = []
    for line in raw.splitlines():
        line = re.sub(r"^\s*[-*・\d.)）]+\s*", "", line).strip().strip('"').strip("'")
        if line:
            recovered.append(line)
    return recovered


def filter_alias_candidates(
    aliases: list[str],
    *,
    question: str,
    existing_normalized: Optional[set[str]] = None,
    max_aliases: int = 8,
    max_length: int = 120,
) -> list[str]:
    existing = set(existing_normalized or set())
    question_norm = normalize_qa_text(question)
    results: list[str] = []
    for alias in aliases:
        candidate = str(alias or "").strip()
        if not candidate or len(candidate) > max_length:
            continue
        normalized = normalize_qa_text(candidate)
        if not normalized or normalized == question_norm or normalized in existing:
            continue
        existing.add(normalized)
        results.append(candidate)
        if len(results) >= max_aliases:
            break
    return results


class AnswerCacheStore:
    def __init__(self, *, auto_migrate: bool = True, auto_ensure_original_aliases: bool = False):
        settings = load_settings()
        cache_settings = settings.get("answer_cache", {})
        self.enabled = bool(cache_settings.get("enabled", True))
        self.required = bool(cache_settings.get("required", True))
        self.sqlite_path = project_path(cache_settings.get("sqlite_path", "logs/answer_cache.sqlite"))
        self.busy_timeout_ms = int(cache_settings.get("busy_timeout_ms", 10000))
        self.connect_timeout_sec = int(cache_settings.get("connect_timeout_sec", 30))
        self.configured_journal_mode = str(cache_settings.get("journal_mode", "DELETE") or "DELETE").upper()
        self.startup_write_check = bool(cache_settings.get("startup_write_check", True))
        self.high_similarity_threshold = float(cache_settings.get("high_similarity_threshold", 0.88))
        self.semantic_accept_threshold = float(
            cache_settings.get("semantic_accept_threshold", self.high_similarity_threshold)
        )
        self.semantic_gray_threshold = float(cache_settings.get("semantic_gray_threshold", 0.82))
        self.margin_threshold = float(cache_settings.get("margin_threshold", 0.03))
        self.corpus_version = str(cache_settings.get("corpus_version", "default"))
        self.index_version = str(cache_settings.get("index_version", "default"))
        self.enable_alias_search = bool(cache_settings.get("enable_alias_search", True))
        self.enable_alias_generation = bool(cache_settings.get("enable_alias_generation", False))
        self.max_aliases_per_qa = int(cache_settings.get("max_aliases_per_qa", 8))
        self.alias_default_status = str(cache_settings.get("alias_default_status", "active"))
        self.generated_alias_default_status = str(cache_settings.get("generated_alias_default_status", "active"))
        self.enable_llm_intent_judge = bool(cache_settings.get("enable_llm_intent_judge", False))
        self.intent_judge_gray_min = float(cache_settings.get("intent_judge_gray_min", self.semantic_gray_threshold))
        self.alias_generation_temperature = float(cache_settings.get("alias_generation_temperature", 0.0))
        self.alias_generation_max_tokens = int(cache_settings.get("alias_generation_max_tokens", 1200))

        self._alias_matrix: Any = None
        self._alias_items: list[dict[str, Any]] = []
        self._alias_loaded_at: str | None = None
        self._alias_index_error: str | None = None
        self._alias_index_meta: dict[str, Any] = {}
        self._alias_lock = threading.RLock()
        self.journal_mode: str | None = None
        self.last_error: str | None = None
        self.last_error_at: str | None = None
        self.startup_check_result: str | None = None

        if self.enabled:
            try:
                if auto_migrate:
                    self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
                    self._init_db()
                    if self.startup_write_check:
                        self._startup_write_check()
                if auto_ensure_original_aliases:
                    self.ensure_original_aliases()
                elif auto_migrate:
                    self.reload_alias_index()
            except (sqlite3.Error, OSError) as exc:
                self._handle_startup_error(exc)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.sqlite_path, timeout=self.connect_timeout_sec)
        conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        conn.execute("PRAGMA foreign_keys=ON")
        row = conn.execute(f"PRAGMA journal_mode={self.configured_journal_mode}").fetchone()
        self.journal_mode = str(row[0]).upper() if row else self.configured_journal_mode
        return conn

    def _handle_startup_error(self, exc: BaseException) -> None:
        self.last_error = str(exc)
        self.last_error_at = now_utc()
        message = (
            f"SQLite startup check failed. db_path={self.sqlite_path} "
            f"journal_mode={self.configured_journal_mode} error={exc}"
        )
        if self.required:
            raise RuntimeError(message) from exc
        self.enabled = False
        print(f"[WARN] {message}")

    @staticmethod
    def _ensure_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        for name, definition in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")

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
                CREATE TABLE IF NOT EXISTS approved_qa_aliases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    qa_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    alias_text TEXT NOT NULL,
                    normalized_text TEXT,
                    alias_type TEXT NOT NULL DEFAULT 'manual',
                    embedding_json TEXT NOT NULL,
                    generator_model TEXT,
                    memo TEXT,
                    FOREIGN KEY (qa_id) REFERENCES approved_qa(id)
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
            self._ensure_columns(
                conn,
                "approved_qa",
                {
                    "needs_alias_review": "INTEGER DEFAULT 0",
                },
            )
            self._ensure_columns(
                conn,
                "approved_qa_aliases",
                {
                    "risk_flags_json": "TEXT",
                },
            )
            self._ensure_columns(
                conn,
                "hallucination_reports",
                {
                    "resolved_qa_id": "INTEGER",
                    "issue_type": "TEXT",
                    "resolution_type": "TEXT",
                    "admin_memo": "TEXT",
                    "resolved_at": "TEXT",
                    "linked_child_id": "TEXT",
                },
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_approved_qa_status ON approved_qa(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_approved_qa_version ON approved_qa(corpus_version, index_version)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reports_status ON hallucination_reports(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_approved_qa_aliases_qa_id ON approved_qa_aliases(qa_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_approved_qa_aliases_status ON approved_qa_aliases(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_approved_qa_aliases_type ON approved_qa_aliases(alias_type)")
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_approved_qa_alias_unique_active
                ON approved_qa_aliases(qa_id, normalized_text, alias_type)
                WHERE status = 'active'
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS app_startup_check (
                    id INTEGER PRIMARY KEY,
                    checked_at TEXT NOT NULL
                )
                """
            )

    def _startup_write_check(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO app_startup_check (id, checked_at)
                VALUES (1, ?)
                """,
                (now_utc(),),
            )
            conn.execute("DELETE FROM app_startup_check WHERE id = 1")
            conn.commit()
        self.startup_check_result = "ok"

    def _qa_exists_conn(self, conn: sqlite3.Connection, qa_id: int) -> bool:
        row = conn.execute("SELECT id FROM approved_qa WHERE id = ?", (qa_id,)).fetchone()
        return row is not None

    def _upsert_original_alias_conn(
        self,
        conn: sqlite3.Connection,
        *,
        qa_id: int,
        question: str,
        embedding_json: str,
        memo: str = "",
    ) -> bool:
        timestamp = now_utc()
        normalized = normalize_qa_text(question)
        row = conn.execute(
            """
            SELECT id
            FROM approved_qa_aliases
            WHERE qa_id = ? AND alias_type = 'original' AND status = 'active'
            ORDER BY id
            LIMIT 1
            """,
            (qa_id,),
        ).fetchone()
        if row:
            conn.execute(
                """
                UPDATE approved_qa_aliases
                SET updated_at = ?,
                    alias_text = ?,
                    normalized_text = ?,
                    embedding_json = ?,
                    memo = CASE WHEN ? != '' THEN ? ELSE memo END
                WHERE id = ?
                """,
                (timestamp, question, normalized, embedding_json, memo, memo, int(row[0])),
            )
            return False
        conn.execute(
            """
            INSERT OR IGNORE INTO approved_qa_aliases (
                qa_id, created_at, updated_at, status, alias_text, normalized_text,
                alias_type, embedding_json, generator_model, memo
            )
            VALUES (?, ?, ?, 'active', ?, ?, 'original', ?, NULL, ?)
            """,
            (qa_id, timestamp, timestamp, question, normalized, embedding_json, memo),
        )
        return True

    def _ensure_original_aliases_conn(self, conn: sqlite3.Connection) -> dict[str, int]:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT qa.id, qa.question, qa.question_embedding_json
            FROM approved_qa qa
            WHERE NOT EXISTS (
                SELECT 1
                FROM approved_qa_aliases a
                WHERE a.qa_id = qa.id
                  AND a.alias_type = 'original'
                  AND a.status = 'active'
            )
            """
        ).fetchall()
        created = 0
        skipped = 0
        for row in rows:
            embedding_json = str(row["question_embedding_json"] or "")
            if not embedding_json:
                skipped += 1
                continue
            if self._upsert_original_alias_conn(
                conn,
                qa_id=int(row["id"]),
                question=str(row["question"] or ""),
                embedding_json=embedding_json,
            ):
                created += 1
        return {"checked": len(rows), "created": created, "skipped": skipped}

    def ensure_original_aliases(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "created": 0, "skipped": 0}
        with self._connect() as conn:
            result = self._ensure_original_aliases_conn(conn)
        self.reload_alias_index()
        return {"enabled": True, **result, "alias_index": self.alias_index_status()}

    def _row_to_approved_qa(self, row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        evidence = safe_json_loads(item.pop("evidence_json", "[]"), [])
        item.pop("question_embedding_json", None)
        item["evidence"] = evidence if isinstance(evidence, list) else []
        item["evidence_count"] = len(item["evidence"])
        return item

    def _evidence_matches(self, evidence: Any, corpus_ids: Optional[list[str]]) -> bool:
        if corpus_ids is None:
            return True
        if len(corpus_ids) == 0:
            return False
        evidence_list = evidence if isinstance(evidence, list) else []
        if not evidence_list:
            return False
        allowed = {str(item) for item in corpus_ids}
        evidence_corpus_ids = {str(e.get("corpus_id")) for e in evidence_list if isinstance(e, dict) and e.get("corpus_id")}
        return bool(evidence_corpus_ids) and not evidence_corpus_ids.isdisjoint(allowed)

    @staticmethod
    def _normalize_vector(values: Any) -> Any:
        if np is None:
            try:
                vector = [float(value) for value in values]
            except (TypeError, ValueError):
                return None
            if not vector:
                return None
            norm = math.sqrt(sum(value * value for value in vector))
            if norm <= 0:
                return None
            return [value / norm for value in vector]
        try:
            vector = np.asarray(values, dtype=np.float32)
        except (TypeError, ValueError):
            return None
        if vector.ndim != 1 or vector.size == 0:
            return None
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            return None
        return vector / norm

    def reload_alias_index(self) -> dict[str, Any]:
        if not self.enabled:
            with self._alias_lock:
                self._alias_items = []
                self._alias_matrix = None
                self._alias_index_meta = {"enabled": False, "loaded_count": 0}
            return {"enabled": False, "count": 0}
        try:
            with self._connect() as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT
                        a.id AS alias_id,
                        a.qa_id,
                        a.alias_text,
                        a.normalized_text,
                        a.alias_type,
                        a.status AS alias_status,
                        a.updated_at AS alias_updated_at,
                        a.generator_model,
                        a.memo AS alias_memo,
                        a.risk_flags_json,
                        a.embedding_json,
                        qa.status AS qa_status,
                        qa.question,
                        qa.answer,
                        qa.evidence_json,
                        qa.corpus_version,
                        qa.index_version,
                        qa.approved_by,
                        qa.memo AS qa_memo
                    FROM approved_qa_aliases a
                    JOIN approved_qa qa ON qa.id = a.qa_id
                    WHERE qa.status = 'approved'
                      AND a.status = 'active'
                    ORDER BY a.id
                    """
                ).fetchall()
        except Exception as exc:
            self._alias_index_error = str(exc)
            with self._alias_lock:
                self._alias_items = []
                self._alias_matrix = None
                self._alias_index_meta = {"enabled": True, "loaded_count": 0, "error": str(exc)}
            return {"enabled": True, "count": 0, "error": str(exc)}

        vectors: list[Any] = []
        items: list[dict[str, Any]] = []
        dimensions: Optional[int] = None
        invalid_embedding_count = 0
        dimension_mismatch_count = 0
        for row in rows:
            embedding = safe_json_loads(row["embedding_json"], [])
            vector = self._normalize_vector(embedding)
            if vector is None:
                invalid_embedding_count += 1
                continue
            vector_size = len(vector) if np is None else int(vector.size)
            if dimensions is None:
                dimensions = vector_size
            if vector_size != dimensions:
                dimension_mismatch_count += 1
                continue
            evidence = safe_json_loads(row["evidence_json"], [])
            item = dict(row)
            item.pop("embedding_json", None)
            item.pop("evidence_json", None)
            risk_flags = safe_json_loads(item.pop("risk_flags_json", "[]"), [])
            item["id"] = int(item["qa_id"])
            item["qa_id"] = int(item["qa_id"])
            item["alias_id"] = int(item["alias_id"])
            item["evidence"] = evidence if isinstance(evidence, list) else []
            item["evidence_count"] = len(item["evidence"])
            item["risk_flags"] = risk_flags if isinstance(risk_flags, list) else []
            items.append(item)
            vectors.append(vector)

        meta = {
            "enabled": True,
            "count": len(items),
            "loaded_count": len(items),
            "skipped_count": invalid_embedding_count + dimension_mismatch_count,
            "embedding_dimension": dimensions,
            "dimension_mismatch_count": dimension_mismatch_count,
            "invalid_embedding_count": invalid_embedding_count,
            "last_reloaded_at": now_utc(),
        }
        with self._alias_lock:
            self._alias_items = items
            self._alias_matrix = np.vstack(vectors) if vectors and np is not None else vectors if vectors else None
            self._alias_loaded_at = str(meta["last_reloaded_at"])
            self._alias_index_error = None
            self._alias_index_meta = dict(meta)
        return meta

    def alias_index_status(self) -> dict[str, Any]:
        with self._alias_lock:
            matrix = self._alias_matrix
            dimension = None
            if np is not None and matrix is not None and getattr(matrix, "ndim", None) == 2:
                dimension = int(matrix.shape[1])
            elif isinstance(matrix, list) and matrix:
                dimension = len(matrix[0])
            meta = dict(self._alias_index_meta)
            meta.update({
                "enabled": self.enabled and self.enable_alias_search,
                "count": len(self._alias_items),
                "loaded_count": len(self._alias_items),
                "dimension": dimension,
                "embedding_dimension": meta.get("embedding_dimension", dimension),
                "loaded_at": self._alias_loaded_at,
                "last_reloaded_at": meta.get("last_reloaded_at", self._alias_loaded_at),
                "error": self._alias_index_error,
            })
            return meta

    def status(self) -> dict[str, Any]:
        exists = self.sqlite_path.exists()
        status: dict[str, Any] = {
            "enabled": self.enabled,
            "required": self.required,
            "db_path": str(self.sqlite_path),
            "exists": exists,
            "size_bytes": self.sqlite_path.stat().st_size if exists else 0,
            "wal_exists": self.sqlite_path.with_name(self.sqlite_path.name + "-wal").exists(),
            "shm_exists": self.sqlite_path.with_name(self.sqlite_path.name + "-shm").exists(),
            "journal_mode": self.journal_mode or self.configured_journal_mode,
            "configured_journal_mode": self.configured_journal_mode,
            "startup_check_result": self.startup_check_result,
            "last_error": self.last_error,
            "last_error_at": self.last_error_at,
            "alias_index": self.alias_index_status(),
        }
        if not self.enabled:
            return status
        try:
            with self._connect() as conn:
                row = conn.execute("SELECT COUNT(*) FROM approved_qa").fetchone()
                status["approved_qa_count"] = int(row[0]) if row else 0
        except (sqlite3.Error, OSError) as exc:
            status["last_error"] = str(exc)
            status["last_error_at"] = now_utc()
        return status

    def _ensure_alias_index_loaded(self) -> None:
        with self._alias_lock:
            needs_load = self._alias_matrix is None and not self._alias_items
        if needs_load:
            self.reload_alias_index()

    def _lookup_snapshot(
        self,
        *,
        include_disabled_qa: bool = False,
        include_disabled_aliases: bool = False,
    ) -> tuple[list[dict[str, Any]], Any]:
        if not include_disabled_qa and not include_disabled_aliases:
            self._ensure_alias_index_loaded()
            with self._alias_lock:
                return list(self._alias_items), self._alias_matrix.copy() if np is not None and self._alias_matrix is not None else list(self._alias_matrix or [])

        where = []
        if not include_disabled_qa:
            where.append("qa.status = 'approved'")
        if not include_disabled_aliases:
            where.append("a.status = 'active'")
        sql_where = "WHERE " + " AND ".join(where) if where else ""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""
                SELECT
                    a.id AS alias_id,
                    a.qa_id,
                    a.alias_text,
                    a.normalized_text,
                    a.alias_type,
                    a.status AS alias_status,
                    a.updated_at AS alias_updated_at,
                    a.generator_model,
                    a.memo AS alias_memo,
                    a.risk_flags_json,
                    a.embedding_json,
                    qa.status AS qa_status,
                    qa.question,
                    qa.answer,
                    qa.evidence_json,
                    qa.corpus_version,
                    qa.index_version,
                    qa.approved_by,
                    qa.memo AS qa_memo
                FROM approved_qa_aliases a
                JOIN approved_qa qa ON qa.id = a.qa_id
                {sql_where}
                ORDER BY a.id
                """
            ).fetchall()
        vectors: list[Any] = []
        items: list[dict[str, Any]] = []
        dimensions: Optional[int] = None
        for row in rows:
            embedding = safe_json_loads(row["embedding_json"], [])
            vector = self._normalize_vector(embedding)
            if vector is None:
                continue
            vector_size = len(vector) if np is None else int(vector.size)
            if dimensions is None:
                dimensions = vector_size
            if vector_size != dimensions:
                continue
            evidence = safe_json_loads(row["evidence_json"], [])
            risk_flags = safe_json_loads(row["risk_flags_json"], [])
            item = dict(row)
            item.pop("embedding_json", None)
            item.pop("evidence_json", None)
            item.pop("risk_flags_json", None)
            item["id"] = int(item["qa_id"])
            item["qa_id"] = int(item["qa_id"])
            item["alias_id"] = int(item["alias_id"])
            item["evidence"] = evidence if isinstance(evidence, list) else []
            item["evidence_count"] = len(item["evidence"])
            item["risk_flags"] = risk_flags if isinstance(risk_flags, list) else []
            items.append(item)
            vectors.append(vector)
        matrix = np.vstack(vectors) if vectors and np is not None else vectors if vectors else None
        return items, matrix

    def _candidate_public(self, item: dict[str, Any], similarity: float, margin: float = 0.0) -> dict[str, Any]:
        return {
            "id": item.get("qa_id"),
            "qa_id": item.get("qa_id"),
            "alias_id": item.get("alias_id"),
            "alias_text": item.get("alias_text"),
            "alias_type": item.get("alias_type"),
            "question": item.get("question"),
            "answer": item.get("answer"),
            "evidence": item.get("evidence") or [],
            "evidence_count": item.get("evidence_count") or 0,
            "similarity": round(float(similarity), 6),
            "semantic_score": round(float(similarity), 6),
            "lexical_score": None,
            "final_score": round(float(similarity), 6),
            "margin": round(float(margin), 6),
            "corpus_version": item.get("corpus_version"),
            "index_version": item.get("index_version"),
            "match_method": "original_semantic" if item.get("alias_type") == "original" else "alias_semantic",
            "matched_alias_id": item.get("alias_id"),
            "matched_alias_text": item.get("alias_text"),
            "matched_alias_type": item.get("alias_type"),
            "status": item.get("qa_status"),
            "alias_status": item.get("alias_status"),
            "risk_flags": item.get("risk_flags") or [],
        }

    def match_debug(
        self,
        query_embedding: list[float],
        *,
        question: str = "",
        corpus_ids: Optional[list[str]] = None,
        corpus_version: Optional[str] = None,
        index_version: Optional[str] = None,
        top_n: int = 10,
        threshold: Optional[float] = None,
        include_disabled_qa: bool = False,
        include_disabled_aliases: bool = False,
    ) -> dict[str, Any]:
        top_n = max(1, min(int(top_n or 10), 50))
        accept_threshold = float(threshold) if threshold is not None else self.semantic_accept_threshold
        corpus_version = corpus_version or self.corpus_version
        index_version = index_version or self.index_version
        normalized_question = normalize_qa_text(question)

        base = {
            "question": question,
            "normalized_question": normalized_question,
            "cache_lookup_query": question,
            "thresholds": {
                "semantic_accept_threshold": accept_threshold,
                "semantic_gray_threshold": self.semantic_gray_threshold,
                "margin_threshold": self.margin_threshold,
                "intent_judge_gray_min": self.intent_judge_gray_min,
            },
            "alias_index": self.alias_index_status(),
            "decision": "miss",
            "miss_reason": None,
            "best": None,
            "candidates": [],
            "exact_match": None,
        }
        if not self.enabled or not self.enable_alias_search:
            base["miss_reason"] = "disabled"
            return base

        items, matrix = self._lookup_snapshot(
            include_disabled_qa=include_disabled_qa,
            include_disabled_aliases=include_disabled_aliases,
        )
        if matrix is None or not items:
            base["miss_reason"] = "no_alias_index"
            return base

        exact_grouped: dict[int, dict[str, Any]] = {}
        for item in items:
            if str(item.get("corpus_version") or "") != str(corpus_version):
                continue
            if str(item.get("index_version") or "") != str(index_version):
                continue
            if not self._evidence_matches(item.get("evidence"), corpus_ids):
                continue
            if str(item.get("normalized_text") or "") == normalized_question and normalized_question:
                qa_id = int(item["qa_id"])
                exact_grouped.setdefault(qa_id, item)
        if exact_grouped:
            exact_candidates = [
                self._candidate_public(item, 1.0, 1.0 if i == 0 else 0.0)
                for i, item in enumerate(exact_grouped.values())
            ]
            for candidate in exact_candidates:
                candidate["match_method"] = "normalized_exact"
                candidate["semantic_score"] = None
                candidate["lexical_score"] = 1.0
                candidate["final_score"] = 1.0
                candidate["would_hit"] = len(exact_grouped) == 1
            best = exact_candidates[0]
            base["exact_match"] = {"count": len(exact_grouped), "normalized_text": normalized_question}
            base["best"] = best
            base["candidates"] = exact_candidates[:top_n]
            if len(exact_grouped) == 1:
                base["decision"] = "hit"
                base["miss_reason"] = None
            else:
                base["decision"] = "gray"
                base["miss_reason"] = "alias_conflict"
            return base

        query_vector = self._normalize_vector(query_embedding)
        if query_vector is None:
            base["miss_reason"] = "invalid_query_embedding"
            return base
        if np is None:
            if len(query_vector) != len(matrix[0]):
                base["miss_reason"] = "dimension_mismatch"
                return base
            scores = [sum(a * b for a, b in zip(vector, query_vector)) for vector in matrix]
            order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
        else:
            if query_vector.size != matrix.shape[1]:
                base["miss_reason"] = "dimension_mismatch"
                return base
            scores = matrix @ query_vector
            order = np.argsort(-scores)
        grouped: dict[int, dict[str, Any]] = {}
        filtered_any = False
        for idx in order:
            item = items[int(idx)]
            if str(item.get("corpus_version") or "") != str(corpus_version):
                continue
            if str(item.get("index_version") or "") != str(index_version):
                continue
            if not self._evidence_matches(item.get("evidence"), corpus_ids):
                continue
            filtered_any = True
            qa_id = int(item["qa_id"])
            similarity = float(scores[int(idx)])
            current = grouped.get(qa_id)
            if current is None or similarity > float(current["similarity"]):
                grouped[qa_id] = {"item": item, "similarity": similarity}
            if len(grouped) >= max(top_n * 4, top_n) and len(grouped) >= top_n:
                break

        if not grouped:
            base["miss_reason"] = "version_mismatch_or_corpus_filtered" if not filtered_any else "no_candidates"
            return base

        ranked = sorted(grouped.values(), key=lambda row: row["similarity"], reverse=True)
        best_row = ranked[0]
        second_row = ranked[1] if len(ranked) > 1 else None
        margin = float(best_row["similarity"]) - float(second_row["similarity"]) if second_row else 1.0
        candidates = [
            self._candidate_public(row["item"], float(row["similarity"]), margin if i == 0 else 0.0)
            for i, row in enumerate(ranked[:top_n])
        ]
        for i, candidate in enumerate(candidates):
            candidate["would_hit"] = (
                i == 0
                and float(candidate.get("similarity") or 0.0) >= accept_threshold
                and margin >= self.margin_threshold
            )

        best = candidates[0]
        best["margin"] = round(margin, 6)
        if float(best["similarity"]) < self.semantic_gray_threshold:
            decision = "miss"
            miss_reason = "below_gray_threshold"
        elif float(best["similarity"]) >= accept_threshold and margin >= self.margin_threshold:
            decision = "hit"
            miss_reason = None
        elif margin < self.margin_threshold:
            decision = "gray"
            miss_reason = "margin_too_small"
        else:
            decision = "gray"
            miss_reason = "below_accept_threshold"

        base.update({
            "decision": decision,
            "miss_reason": miss_reason,
            "best": best,
            "candidates": candidates,
        })
        return base

    def find_match(
        self,
        query_embedding: list[float],
        corpus_ids: Optional[list[str]] = None,
        corpus_version: Optional[str] = None,
        index_version: Optional[str] = None,
        threshold: Optional[float] = None,
        question: str = "",
    ) -> Optional[dict[str, Any]]:
        debug = self.match_debug(
            query_embedding,
            question=question,
            corpus_ids=corpus_ids,
            corpus_version=corpus_version,
            index_version=index_version,
            threshold=threshold,
        )
        if debug.get("decision") in {"hit", "llm_judge_hit"}:
            return debug.get("best")
        return None

    def similar_approved(
        self,
        query_embedding: list[float],
        *,
        corpus_ids: Optional[list[str]] = None,
        corpus_version: Optional[str] = None,
        index_version: Optional[str] = None,
        top_n: int = 5,
        threshold: float = 0.0,
        include_disabled: bool = False,
        include_disabled_aliases: bool = False,
        question: str = "",
    ) -> list[dict[str, Any]]:
        debug = self.match_debug(
            query_embedding,
            question=question,
            corpus_ids=corpus_ids,
            corpus_version=corpus_version,
            index_version=index_version,
            top_n=top_n,
            threshold=None,
            include_disabled_qa=include_disabled,
            include_disabled_aliases=include_disabled_aliases,
        )
        matches = [
            item for item in debug.get("candidates", [])
            if float(item.get("similarity") or 0.0) >= max(0.0, float(threshold))
        ]
        return matches[: max(1, min(top_n, 20))]

    def get_approved_qa(self, qa_id: int) -> Optional[dict[str, Any]]:
        if not self.enabled:
            return None
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM approved_qa WHERE id = ?", (qa_id,)).fetchone()
        return self._row_to_approved_qa(row) if row else None

    def recent_approved(self, limit: int = 50, status: Optional[str] = None) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        limit = max(1, min(limit, 500))
        sql = "SELECT * FROM approved_qa"
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

    def approved_summary(self, limit: int = 10) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "total": 0, "by_status": [], "by_version": [], "recent": []}
        limit = max(1, min(limit, 100))
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            status_rows = conn.execute(
                "SELECT status, COUNT(*) AS count FROM approved_qa GROUP BY status ORDER BY count DESC"
            ).fetchall()
            version_rows = conn.execute(
                """
                SELECT corpus_version, index_version, COUNT(*) AS count
                FROM approved_qa
                GROUP BY corpus_version, index_version
                ORDER BY count DESC
                LIMIT 20
                """
            ).fetchall()
            alias_status_rows = conn.execute(
                "SELECT status, COUNT(*) AS count FROM approved_qa_aliases GROUP BY status ORDER BY count DESC"
            ).fetchall()
            alias_type_rows = conn.execute(
                "SELECT alias_type, COUNT(*) AS count FROM approved_qa_aliases GROUP BY alias_type ORDER BY count DESC"
            ).fetchall()
            original_missing = conn.execute(
                """
                SELECT COUNT(*)
                FROM approved_qa qa
                WHERE qa.status = 'approved'
                  AND NOT EXISTS (
                      SELECT 1 FROM approved_qa_aliases a
                      WHERE a.qa_id = qa.id AND a.alias_type = 'original' AND a.status = 'active'
                  )
                """
            ).fetchone()[0]
            llm_missing = conn.execute(
                """
                SELECT COUNT(*)
                FROM approved_qa qa
                WHERE qa.status = 'approved'
                  AND NOT EXISTS (
                      SELECT 1 FROM approved_qa_aliases a
                      WHERE a.qa_id = qa.id AND a.alias_type = 'llm_paraphrase' AND a.status = 'active'
                  )
                """
            ).fetchone()[0]
            current_version_count = conn.execute(
                """
                SELECT COUNT(*)
                FROM approved_qa
                WHERE status = 'approved' AND corpus_version = ? AND index_version = ?
                """,
                (self.corpus_version, self.index_version),
            ).fetchone()[0]
            version_excluded_count = conn.execute(
                """
                SELECT COUNT(*)
                FROM approved_qa
                WHERE status = 'approved' AND NOT (corpus_version = ? AND index_version = ?)
                """,
                (self.corpus_version, self.index_version),
            ).fetchone()[0]
            evidence_missing_count = conn.execute(
                """
                SELECT COUNT(*)
                FROM approved_qa
                WHERE status = 'approved' AND (evidence_json IS NULL OR evidence_json = '' OR evidence_json = '[]')
                """
            ).fetchone()[0]
        by_status = [dict(row) for row in status_rows]
        by_alias_status = [dict(row) for row in alias_status_rows]
        active_aliases = sum(int(row.get("count") or 0) for row in by_alias_status if row.get("status") == "active")
        disabled_aliases = sum(int(row.get("count") or 0) for row in by_alias_status if row.get("status") == "disabled")
        return {
            "enabled": True,
            "total": sum(int(row.get("count") or 0) for row in by_status),
            "by_status": by_status,
            "by_version": [dict(row) for row in version_rows],
            "recent": self.recent_approved(limit=limit, status="all"),
            "aliases": {
                "active": active_aliases,
                "disabled": disabled_aliases,
                "by_status": by_alias_status,
                "by_type": [dict(row) for row in alias_type_rows],
                "original_missing_qa": int(original_missing or 0),
                "llm_missing_qa": int(llm_missing or 0),
                "alias_index": self.alias_index_status(),
                "current_corpus_version": self.corpus_version,
                "current_index_version": self.index_version,
                "current_version_approved_qa": int(current_version_count or 0),
                "version_excluded_approved_qa": int(version_excluded_count or 0),
                "evidence_missing_qa": int(evidence_missing_count or 0),
            },
        }

    def alias_conflicts(self, limit: int = 100) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        limit = max(1, min(int(limit or 100), 500))
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT normalized_text,
                       COUNT(DISTINCT qa_id) AS qa_count,
                       GROUP_CONCAT(DISTINCT qa_id) AS qa_ids,
                       GROUP_CONCAT(alias_text, ' | ') AS alias_texts
                FROM approved_qa_aliases
                WHERE status = 'active' AND COALESCE(normalized_text, '') != ''
                GROUP BY normalized_text
                HAVING COUNT(DISTINCT qa_id) > 1
                ORDER BY qa_count DESC, normalized_text
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

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
        embedding_json = json.dumps(question_embedding, ensure_ascii=False)
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO approved_qa (
                    created_at, updated_at, status, question, answer, question_embedding_json,
                    evidence_json, corpus_version, index_version, approved_by, source_report_id, memo
                )
                VALUES (?, ?, 'approved', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    timestamp,
                    question,
                    answer,
                    embedding_json,
                    json.dumps(evidence, ensure_ascii=False),
                    corpus_version,
                    index_version,
                    approved_by,
                    source_report_id,
                    memo,
                ),
            )
            qa_id = int(cur.lastrowid)
            self._upsert_original_alias_conn(conn, qa_id=qa_id, question=question, embedding_json=embedding_json)
            if source_report_id:
                conn.execute(
                    """
                    UPDATE hallucination_reports
                    SET status = 'resolved',
                        resolved_qa_id = ?,
                        resolution_type = COALESCE(resolution_type, 'qa_created'),
                        resolved_at = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (qa_id, timestamp, timestamp, source_report_id),
                )
        self.reload_alias_index()
        return qa_id

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
        embedding_json = json.dumps(question_embedding, ensure_ascii=False)
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            old_row = conn.execute("SELECT question, answer, evidence_json FROM approved_qa WHERE id = ?", (qa_id,)).fetchone()
            if old_row is None:
                raise ValueError("approved QA not found")
            old_evidence_json = str(old_row["evidence_json"] or "[]")
            new_evidence_json = json.dumps(evidence, ensure_ascii=False)
            content_changed = (
                str(old_row["question"] or "") != str(question or "")
                or str(old_row["answer"] or "") != str(answer or "")
                or old_evidence_json != new_evidence_json
            )
            cur = conn.execute(
                """
                UPDATE approved_qa
                SET updated_at = ?, status = ?, question = ?, answer = ?, question_embedding_json = ?,
                    evidence_json = ?, corpus_version = ?, index_version = ?, approved_by = ?, memo = ?,
                    needs_alias_review = ?
                WHERE id = ?
                """,
                (
                    timestamp,
                    status,
                    question,
                    answer,
                    embedding_json,
                    json.dumps(evidence, ensure_ascii=False),
                    corpus_version,
                    index_version,
                    approved_by,
                    memo,
                    1 if content_changed else 0,
                    qa_id,
                ),
            )
            if cur.rowcount == 0:
                raise ValueError("approved QA not found")
            self._upsert_original_alias_conn(conn, qa_id=qa_id, question=question, embedding_json=embedding_json)
            disabled_llm_aliases = 0
            if content_changed:
                disabled_llm_aliases = int(conn.execute(
                    """
                    UPDATE approved_qa_aliases
                    SET status = 'disabled',
                        updated_at = ?,
                        memo = CASE WHEN memo IS NULL OR memo = ''
                            THEN 'disabled because parent QA content changed'
                            ELSE memo || ' / disabled because parent QA content changed'
                        END
                    WHERE qa_id = ? AND alias_type = 'llm_paraphrase' AND status = 'active'
                    """,
                    (timestamp, qa_id),
                ).rowcount or 0)
        self.reload_alias_index()
        item = self.get_approved_qa(qa_id)
        if item is None:
            raise ValueError("approved QA not found")
        if content_changed:
            item["needs_alias_review"] = True
            item["alias_review_warning"] = "QA content changed; llm_paraphrase aliases were disabled and admin_alias should be reviewed."
            item["disabled_llm_aliases"] = disabled_llm_aliases
        return item

    def disable_approved_qa(self, qa_id: int, memo: str = "") -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("answer cache is disabled")
        timestamp = now_utc()
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE approved_qa
                SET status = 'disabled',
                    memo = CASE WHEN ? != '' THEN ? ELSE memo END,
                    updated_at = ?
                WHERE id = ?
                """,
                (memo, memo, timestamp, qa_id),
            )
            if cur.rowcount == 0:
                raise ValueError("approved QA not found")
        self.reload_alias_index()
        item = self.get_approved_qa(qa_id)
        if item is None:
            raise ValueError("approved QA not found")
        return item

    def list_aliases(self, qa_id: int) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, qa_id, created_at, updated_at, status, alias_text, normalized_text,
                       alias_type, generator_model, memo, risk_flags_json
                FROM approved_qa_aliases
                WHERE qa_id = ?
                ORDER BY status = 'active' DESC, alias_type, id
                """,
                (qa_id,),
            ).fetchall()
        items = []
        for row in rows:
            item = dict(row)
            risk_flags = safe_json_loads(item.pop("risk_flags_json", "[]"), [])
            item["risk_flags"] = risk_flags if isinstance(risk_flags, list) else []
            items.append(item)
        return items

    def add_aliases(self, qa_id: int, aliases: list[dict[str, Any]]) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("answer cache is disabled")
        timestamp = now_utc()
        created: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            qa_row = conn.execute("SELECT question, evidence_json FROM approved_qa WHERE id = ?", (qa_id,)).fetchone()
            if qa_row is None:
                raise ValueError("approved QA not found")
            qa_question = str(qa_row["question"] or "")
            qa_evidence = safe_json_loads(qa_row["evidence_json"], [])
            for alias in aliases:
                alias_text = str(alias.get("alias_text") or alias.get("text") or "").strip()
                alias_type = str(alias.get("alias_type") or "admin_alias")
                status = str(alias.get("status") or (self.generated_alias_default_status if alias_type == "llm_paraphrase" else self.alias_default_status) or "active")
                if alias_type not in VALID_ALIAS_TYPES:
                    errors.append({"alias_text": alias_text, "error": "invalid alias_type"})
                    continue
                if status not in VALID_ALIAS_STATUSES:
                    errors.append({"alias_text": alias_text, "error": "invalid status"})
                    continue
                embedding = alias.get("embedding")
                embedding_json = alias.get("embedding_json")
                if embedding_json is None:
                    if not isinstance(embedding, list):
                        errors.append({"alias_text": alias_text, "error": "embedding required"})
                        continue
                    embedding_json = json.dumps(embedding, ensure_ascii=False)
                if not alias_text:
                    skipped.append({"alias_text": alias_text, "reason": "empty"})
                    continue
                normalized = normalize_qa_text(alias_text)
                risk_flags = alias.get("risk_flags")
                if not isinstance(risk_flags, list):
                    risk_flags = detect_alias_risk_flags(qa_question, alias_text, qa_evidence if isinstance(qa_evidence, list) else [])
                requested_status = status
                force_active_conflict = bool(alias.get("force_active_conflict", False))
                conflict_rows = conn.execute(
                    """
                    SELECT DISTINCT qa_id
                    FROM approved_qa_aliases
                    WHERE normalized_text = ? AND qa_id != ? AND status = 'active'
                    """,
                    (normalized, qa_id),
                ).fetchall()
                if conflict_rows:
                    warnings.append({
                        "alias_text": alias_text,
                        "normalized_text": normalized,
                        "warning": "alias_conflict_across_qa",
                        "conflict_qa_ids": [int(row["qa_id"]) for row in conflict_rows],
                    })
                    if requested_status == "active" and not force_active_conflict:
                        status = "disabled"
                        if "alias_conflict_across_qa" not in risk_flags:
                            risk_flags.append("alias_conflict_across_qa")
                before = conn.total_changes
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO approved_qa_aliases (
                        qa_id, created_at, updated_at, status, alias_text, normalized_text,
                        alias_type, embedding_json, generator_model, memo, risk_flags_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        qa_id,
                        timestamp,
                        timestamp,
                        status,
                        alias_text,
                        normalized,
                        alias_type,
                        str(embedding_json),
                        alias.get("generator_model"),
                        alias.get("memo") or "",
                        json.dumps(risk_flags, ensure_ascii=False),
                    ),
                )
                if conn.total_changes == before:
                    skipped.append({"alias_text": alias_text, "reason": "duplicate"})
                else:
                    created.append({
                        "id": int(cur.lastrowid),
                        "alias_text": alias_text,
                        "normalized_text": normalized,
                        "alias_type": alias_type,
                        "status": status,
                        "risk_flags": risk_flags,
                    })
        if created:
            self.reload_alias_index()
        return {"qa_id": qa_id, "created": created, "skipped": skipped, "warnings": warnings, "errors": errors, "created_count": len(created)}

    def replace_llm_aliases(self, qa_id: int, aliases: list[dict[str, Any]], memo: str = "replace generated aliases") -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("answer cache is disabled")
        if not aliases:
            return {
                "qa_id": qa_id,
                "old_aliases_kept": True,
                "disabled_old_count": 0,
                "inserted_count": 0,
                "created": [],
                "errors": [{"error": "no new aliases"}],
            }
        timestamp = now_utc()
        prepared: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            qa_row = conn.execute("SELECT question, evidence_json FROM approved_qa WHERE id = ?", (qa_id,)).fetchone()
            if qa_row is None:
                raise ValueError("approved QA not found")
            qa_question = str(qa_row["question"] or "")
            qa_evidence = safe_json_loads(qa_row["evidence_json"], [])
            for alias in aliases:
                alias_text = str(alias.get("alias_text") or alias.get("text") or "").strip()
                embedding = alias.get("embedding")
                if not alias_text:
                    errors.append({"alias_text": alias_text, "error": "empty"})
                    continue
                if not isinstance(embedding, list):
                    errors.append({"alias_text": alias_text, "error": "embedding required"})
                    continue
                risk_flags = alias.get("risk_flags")
                if not isinstance(risk_flags, list):
                    risk_flags = detect_alias_risk_flags(qa_question, alias_text, qa_evidence if isinstance(qa_evidence, list) else [])
                normalized_text = normalize_qa_text(alias_text)
                item_status = str(alias.get("status") or self.generated_alias_default_status or "active")
                conflict_rows = conn.execute(
                    """
                    SELECT DISTINCT qa_id
                    FROM approved_qa_aliases
                    WHERE normalized_text = ? AND qa_id != ? AND status = 'active'
                    """,
                    (normalized_text, qa_id),
                ).fetchall()
                if conflict_rows and item_status == "active":
                    item_status = "disabled"
                    if "alias_conflict_across_qa" not in risk_flags:
                        risk_flags.append("alias_conflict_across_qa")
                prepared.append({
                    "alias_text": alias_text,
                    "normalized_text": normalized_text,
                    "status": item_status,
                    "embedding_json": json.dumps(embedding, ensure_ascii=False),
                    "generator_model": alias.get("generator_model"),
                    "memo": alias.get("memo") or memo,
                    "risk_flags": risk_flags,
                })
            if not prepared:
                return {
                    "qa_id": qa_id,
                    "old_aliases_kept": True,
                    "disabled_old_count": 0,
                    "inserted_count": 0,
                    "created": [],
                    "errors": errors or [{"error": "no valid aliases"}],
                }
            for item in prepared:
                if item["status"] not in VALID_ALIAS_STATUSES:
                    raise ValueError("invalid status")
            disabled_cur = conn.execute(
                """
                UPDATE approved_qa_aliases
                SET status = 'disabled',
                    updated_at = ?,
                    memo = CASE WHEN ? != '' THEN ? ELSE memo END
                WHERE qa_id = ? AND alias_type = 'llm_paraphrase' AND status = 'active'
                """,
                (timestamp, memo, memo, qa_id),
            )
            created: list[dict[str, Any]] = []
            for item in prepared:
                cur = conn.execute(
                    """
                    INSERT INTO approved_qa_aliases (
                        qa_id, created_at, updated_at, status, alias_text, normalized_text,
                        alias_type, embedding_json, generator_model, memo, risk_flags_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, 'llm_paraphrase', ?, ?, ?, ?)
                    """,
                    (
                        qa_id,
                        timestamp,
                        timestamp,
                        item["status"],
                        item["alias_text"],
                        item["normalized_text"],
                        item["embedding_json"],
                        item["generator_model"],
                        item["memo"],
                        json.dumps(item["risk_flags"], ensure_ascii=False),
                    ),
                )
                created.append({
                    "id": int(cur.lastrowid),
                    "alias_text": item["alias_text"],
                    "normalized_text": item["normalized_text"],
                    "alias_type": "llm_paraphrase",
                    "status": item["status"],
                    "risk_flags": item["risk_flags"],
                })
            disabled_old_count = int(disabled_cur.rowcount or 0)
        self.reload_alias_index()
        return {
            "qa_id": qa_id,
            "old_aliases_kept": False,
            "disabled_old_count": disabled_old_count,
            "inserted_count": len(created),
            "created": created,
            "errors": errors,
        }

    def update_alias(
        self,
        alias_id: int,
        *,
        alias_text: str | None = None,
        status: str | None = None,
        embedding: list[float] | None = None,
        memo: str = "",
    ) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("answer cache is disabled")
        timestamp = now_utc()
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM approved_qa_aliases WHERE id = ?", (alias_id,)).fetchone()
            if row is None:
                raise ValueError("alias not found")
            next_text = str(alias_text if alias_text is not None else row["alias_text"]).strip()
            next_status = str(status if status is not None else row["status"])
            if not next_text:
                raise ValueError("alias_text is required")
            if next_status not in VALID_ALIAS_STATUSES:
                raise ValueError("invalid status")
            text_changed = next_text != str(row["alias_text"])
            if text_changed and embedding is None:
                raise ValueError("embedding is required when alias_text changes")
            embedding_json = json.dumps(embedding, ensure_ascii=False) if embedding is not None else row["embedding_json"]
            risk_flags_json = row["risk_flags_json"] if "risk_flags_json" in row.keys() else None
            if text_changed:
                qa_row = conn.execute("SELECT question, evidence_json FROM approved_qa WHERE id = ?", (int(row["qa_id"]),)).fetchone()
                qa_evidence = safe_json_loads(qa_row["evidence_json"], []) if qa_row else []
                risk_flags = detect_alias_risk_flags(str(qa_row["question"] or "") if qa_row else "", next_text, qa_evidence if isinstance(qa_evidence, list) else [])
                risk_flags_json = json.dumps(risk_flags, ensure_ascii=False)
            risk_flags_for_update = safe_json_loads(risk_flags_json, []) if risk_flags_json else []
            normalized_next = normalize_qa_text(next_text)
            if next_status == "active":
                conflict_rows = conn.execute(
                    """
                    SELECT DISTINCT qa_id
                    FROM approved_qa_aliases
                    WHERE normalized_text = ? AND qa_id != ? AND status = 'active'
                    """,
                    (normalized_next, int(row["qa_id"])),
                ).fetchall()
                if conflict_rows:
                    next_status = "disabled"
                    if "alias_conflict_across_qa" not in risk_flags_for_update:
                        risk_flags_for_update.append("alias_conflict_across_qa")
                    risk_flags_json = json.dumps(risk_flags_for_update, ensure_ascii=False)
            conn.execute(
                """
                UPDATE approved_qa_aliases
                SET updated_at = ?, alias_text = ?, normalized_text = ?, status = ?,
                    embedding_json = ?, risk_flags_json = ?, memo = CASE WHEN ? != '' THEN ? ELSE memo END
                WHERE id = ?
                """,
                (timestamp, next_text, normalized_next, next_status, embedding_json, risk_flags_json, memo, memo, alias_id),
            )
        self.reload_alias_index()
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            updated = conn.execute(
                """
                SELECT id, qa_id, created_at, updated_at, status, alias_text, normalized_text,
                       alias_type, generator_model, memo, risk_flags_json
                FROM approved_qa_aliases
                WHERE id = ?
                """,
                (alias_id,),
            ).fetchone()
        if not updated:
            return {}
        item = dict(updated)
        risk_flags = safe_json_loads(item.pop("risk_flags_json", "[]"), [])
        item["risk_flags"] = risk_flags if isinstance(risk_flags, list) else []
        return item

    def disable_alias(self, alias_id: int, memo: str = "") -> dict[str, Any]:
        return self.update_alias(alias_id, status="disabled", memo=memo)

    def disable_aliases(self, qa_id: int, alias_type: str, memo: str = "") -> int:
        if alias_type not in VALID_ALIAS_TYPES:
            raise ValueError("invalid alias_type")
        timestamp = now_utc()
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE approved_qa_aliases
                SET status = 'disabled',
                    memo = CASE WHEN ? != '' THEN ? ELSE memo END,
                    updated_at = ?
                WHERE qa_id = ? AND alias_type = ? AND status = 'active'
                """,
                (memo, memo, timestamp, qa_id, alias_type),
            )
            count = int(cur.rowcount or 0)
        if count:
            self.reload_alias_index()
        return count

    def backfill_aliases(
        self,
        *,
        ensure_original: bool = True,
        generate_llm_aliases: bool = False,
        only_without_llm_aliases: bool = True,
        limit: int = 50,
        dry_run: bool = True,
        corpus_version: Optional[str] = None,
        index_version: Optional[str] = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        limit = max(1, min(int(limit or 50), 1000))
        corpus_version = corpus_version or self.corpus_version
        index_version = index_version or self.index_version
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            where = [
                "qa.status = 'approved'",
                "qa.corpus_version = ?",
                "qa.index_version = ?",
            ]
            params: list[Any] = [corpus_version, index_version]
            if only_without_llm_aliases and generate_llm_aliases:
                where.append(
                    """
                    NOT EXISTS (
                        SELECT 1 FROM approved_qa_aliases a
                        WHERE a.qa_id = qa.id AND a.alias_type = 'llm_paraphrase' AND a.status = 'active'
                    )
                    """
                )
            rows = conn.execute(
                f"""
                SELECT qa.*
                FROM approved_qa qa
                WHERE {' AND '.join(where)}
                ORDER BY qa.id
                LIMIT ?
                """,
                [*params, limit],
            ).fetchall()
            missing_original_rows = []
            for row in rows:
                original = conn.execute(
                    """
                    SELECT 1
                    FROM approved_qa_aliases
                    WHERE qa_id = ? AND alias_type = 'original' AND status = 'active'
                    LIMIT 1
                    """,
                    (int(row["id"]),),
                ).fetchone()
                if original is None:
                    missing_original_rows.append(row)
            missing_original_count = len(missing_original_rows)
            original_result = {"checked": len(missing_original_rows), "created": 0, "skipped": 0}
            if ensure_original and not dry_run:
                for row in missing_original_rows:
                    embedding_json = str(row["question_embedding_json"] or "")
                    if not embedding_json:
                        original_result["skipped"] += 1
                        continue
                    if self._upsert_original_alias_conn(
                        conn,
                        qa_id=int(row["id"]),
                        question=str(row["question"] or ""),
                        embedding_json=embedding_json,
                    ):
                        original_result["created"] += 1
                    else:
                        original_result["skipped"] += 1
        if not dry_run:
            self.reload_alias_index()
        return {
            "enabled": True,
            "dry_run": dry_run,
            "target_count": len(rows),
            "ensure_original": ensure_original,
            "generate_llm_aliases": generate_llm_aliases,
            "only_without_llm_aliases": only_without_llm_aliases,
            "would_insert": int(missing_original_count or 0) if ensure_original else 0,
            "would_update": 0,
            "would_disable": 0,
            "original": original_result,
            "items": [
                {"qa_id": int(row["id"]), "question": row["question"], "corpus_version": row["corpus_version"], "index_version": row["index_version"]}
                for row in rows
            ],
            "alias_index": self.alias_index_status(),
        }

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
                    created_at, updated_at, status, session_id, log_id, question, answer, comment
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
                   question, answer, comment, resolved_qa_id,
                   issue_type, resolution_type, admin_memo, resolved_at, linked_child_id
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

    def get_report(self, report_id: int) -> Optional[dict[str, Any]]:
        if not self.enabled:
            return None
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT id, created_at, updated_at, status, session_id, log_id,
                       question, answer, comment, resolved_qa_id,
                       issue_type, resolution_type, admin_memo, resolved_at, linked_child_id
                FROM hallucination_reports
                WHERE id = ?
                """,
                (report_id,),
            ).fetchone()
        return dict(row) if row else None

    def report_summary(self, limit: int = 10) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "total": 0, "by_status": [], "recent": [], "open": []}
        limit = max(1, min(limit, 100))
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            status_rows = conn.execute(
                "SELECT status, COUNT(*) AS count FROM hallucination_reports GROUP BY status ORDER BY count DESC"
            ).fetchall()
        by_status = [dict(row) for row in status_rows]
        return {
            "enabled": True,
            "total": sum(int(row.get("count") or 0) for row in by_status),
            "by_status": by_status,
            "recent": self.recent_reports(limit=limit),
            "open": self.recent_reports(status="open", limit=limit),
        }

    def update_report_status(self, report_id: int, status: str) -> None:
        if status not in {"open", "resolved", "ignored"}:
            raise ValueError("invalid status")
        timestamp = now_utc()
        resolved_at = timestamp if status in {"resolved", "ignored"} else None
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE hallucination_reports
                SET status = ?, resolved_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, resolved_at, timestamp, report_id),
            )

    def update_report_analysis(
        self,
        report_id: int,
        *,
        status: Optional[str] = None,
        issue_type: Optional[str] = None,
        resolution_type: Optional[str] = None,
        admin_memo: str = "",
        linked_child_id: Optional[str] = None,
        resolved_qa_id: Optional[int] = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("answer cache is disabled")
        if status is not None and status not in {"open", "resolved", "ignored"}:
            raise ValueError("invalid status")
        timestamp = now_utc()
        current = self.get_report(report_id)
        if current is None:
            raise ValueError("hallucination report not found")
        next_status = status or current.get("status") or "open"
        next_issue_type = issue_type if issue_type is not None else current.get("issue_type")
        next_resolution_type = resolution_type if resolution_type is not None else current.get("resolution_type")
        next_admin_memo = admin_memo if admin_memo else str(current.get("admin_memo") or "")
        next_linked_child_id = linked_child_id if linked_child_id is not None else current.get("linked_child_id")
        next_resolved_qa_id = resolved_qa_id if resolved_qa_id is not None else current.get("resolved_qa_id")
        resolved_at = current.get("resolved_at")
        if next_status in {"resolved", "ignored"} and not resolved_at:
            resolved_at = timestamp
        if next_status == "open":
            resolved_at = None
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE hallucination_reports
                SET status = ?, issue_type = ?, resolution_type = ?, admin_memo = ?,
                    linked_child_id = ?, resolved_qa_id = ?, resolved_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    next_status,
                    next_issue_type,
                    next_resolution_type,
                    next_admin_memo,
                    next_linked_child_id,
                    next_resolved_qa_id,
                    resolved_at,
                    timestamp,
                    report_id,
                ),
            )
        updated = self.get_report(report_id)
        if updated is None:
            raise ValueError("hallucination report not found")
        return updated
