from __future__ import annotations

import json
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_EVENT_PATH = BASE_DIR / "data" / "meeting_events" / "meeting_events.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


class MeetingEventStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else DEFAULT_EVENT_PATH

    def all(self) -> list[dict]:
        return self._dedupe(load_jsonl(self.path))

    def replace_all(self, rows: list[dict]) -> None:
        write_jsonl_atomic(self.path, self._dedupe(rows))

    def query(
        self,
        *,
        meeting_id: str | None = None,
        meeting_ids: list[str] | None = None,
        topic: str | None = None,
        event_type: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        status: str | None = None,
        owner: str | None = None,
        source_type: str | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        rows = self.all()
        id_set = set(meeting_ids or [])
        if meeting_id:
            id_set.add(meeting_id)

        topic_text = (topic or "").casefold().strip()
        status_text = (status or "").casefold().strip()
        owner_text = (owner or "").casefold().strip()
        source_type_text = (source_type or "").casefold().strip()

        filtered = []
        for row in rows:
            if id_set and str(row.get("meeting_id") or "") not in id_set:
                continue
            if event_type and str(row.get("event_type") or "") != event_type:
                continue
            if not _date_in_range(str(row.get("event_date") or row.get("meeting_date") or ""), date_from, date_to):
                continue
            if topic_text and topic_text not in _topic_haystack(row):
                continue
            if status_text and status_text not in str(row.get("status") or "").casefold():
                continue
            if owner_text and owner_text not in str(row.get("owner") or "").casefold():
                continue
            if source_type_text and not _has_source_type(row, source_type_text):
                continue
            filtered.append(row)

        filtered.sort(key=_event_sort_key)
        if limit is not None:
            return filtered[: max(0, int(limit))]
        return filtered

    def _dedupe(self, rows: list[dict]) -> list[dict]:
        seen = set()
        deduped = []
        for row in rows:
            event_id = str(row.get("event_id") or "")
            if not event_id:
                continue
            if event_id in seen:
                continue
            seen.add(event_id)
            deduped.append(row)
        deduped.sort(key=_event_sort_key)
        return deduped


def _event_sort_key(row: dict) -> tuple[str, str, str, str]:
    return (
        str(row.get("event_date") or ""),
        str(row.get("meeting_date") or ""),
        str(row.get("meeting_id") or ""),
        str(row.get("event_id") or ""),
    )


def _date_in_range(value: str, date_from: str | None, date_to: str | None) -> bool:
    if date_from and value and value < date_from:
        return False
    if date_to and value and value > date_to:
        return False
    return True


def _topic_haystack(row: dict[str, Any]) -> str:
    values = [
        row.get("topic"),
        row.get("subtopic"),
        row.get("event_summary"),
        row.get("meeting_name"),
    ]
    return "\n".join(str(value or "") for value in values).casefold()


def _has_source_type(row: dict, source_type: str) -> bool:
    refs = row.get("source_refs") or []
    return any(str(ref.get("source_type") or "").casefold() == source_type for ref in refs if isinstance(ref, dict))
