from __future__ import annotations

from typing import Any


def format_timeline_answer(events: list[dict]) -> str:
    if not events:
        return (
            "## 1. 時系列サマリ\n"
            "該当するMeetingEventは見つかりませんでした。\n\n"
            "## 2. 詳細タイムライン表\n"
            "| 日付 | 会議名 | 種別 | トピック | 内容 | 担当 | 期限 | 根拠 |\n"
            "|---|---|---|---|---|---|---|---|\n\n"
            "## 3. 現在の状態\n確認できません。\n\n"
            "## 4. 未完了事項・宿題\n確認できません。\n\n"
            "## 5. 根拠一覧\nなし"
        )

    table_rows = []
    for event in events:
        table_rows.append(
            "| {date} | {meeting} | {etype} | {topic} | {summary} | {owner} | {due} | {source} |".format(
                date=escape_table(event.get("event_date") or event.get("meeting_date") or ""),
                meeting=escape_table(event.get("meeting_name") or ""),
                etype=escape_table(event.get("event_type") or ""),
                topic=escape_table(event.get("topic") or ""),
                summary=escape_table(event.get("event_summary") or ""),
                owner=escape_table(event.get("owner") or ""),
                due=escape_table(event.get("due_date") or ""),
                source=escape_table(source_label(event)),
            )
        )

    current = current_state(events)
    open_items = [
        event for event in events
        if event.get("event_type") in {"action_item", "pending"} and not is_done(event.get("status"))
    ]
    open_text = "\n".join(
        f"- {event.get('event_date') or event.get('meeting_date')}: {event.get('event_summary')} "
        f"(担当: {event.get('owner') or '未記載'} / 期限: {event.get('due_date') or '未記載'})"
        for event in open_items
    ) or "確認できる未完了事項・宿題はありません。"
    evidence_text = "\n".join(f"- {source_label(event)}" for event in events) or "なし"

    return (
        f"## 1. 時系列サマリ\n"
        f"{len(events)}件のMeetingEventをevent_date、meeting_dateの昇順で整理しました。\n\n"
        f"## 2. 詳細タイムライン表\n"
        "| 日付 | 会議名 | 種別 | トピック | 内容 | 担当 | 期限 | 根拠 |\n"
        "|---|---|---|---|---|---|---|---|\n"
        + "\n".join(table_rows)
        + "\n\n"
        f"## 3. 現在の状態\n{current}\n\n"
        f"## 4. 未完了事項・宿題\n{open_text}\n\n"
        f"## 5. 根拠一覧\n{evidence_text}"
    )


def event_sources(events: list[dict]) -> list[dict]:
    sources = []
    for idx, event in enumerate(events, start=1):
        ref = first_source_ref(event)
        sources.append({
            "corpus_id": "meeting_events",
            "parent_id": str(event.get("event_id") or f"meeting_event_{idx}"),
            "child_id": f"{event.get('event_id') or idx}:event",
            "title": event.get("meeting_name"),
            "heading_path": f"{event.get('topic') or ''} > {event.get('event_type') or ''}",
            "child_text": event.get("event_summary") or "",
            "parent_text": "",
            "score": 1.0,
            "source_file": ref.get("source_file"),
            "search_tags": [],
            "forms": [],
            "document_type": ref.get("document_type"),
            "source_type": ref.get("source_type"),
            "meeting_id": event.get("meeting_id"),
            "meeting_name": event.get("meeting_name"),
            "meeting_date": event.get("meeting_date"),
            "agenda": ref.get("agenda"),
            "topic": event.get("topic"),
            "section_title": ref.get("section_title"),
            "slide_no": ref.get("slide_no"),
            "slide_title": ref.get("slide_title"),
        })
    return sources


def current_state(events: list[dict]) -> str:
    for event in reversed(events):
        if event.get("event_type") in {"decision", "change", "completion", "report"}:
            return (
                f"{event.get('event_date') or event.get('meeting_date')}時点では、"
                f"{event.get('event_summary')} と記録されています。"
            )
    last = events[-1]
    return f"最新イベントは {last.get('event_date') or last.get('meeting_date')} の {last.get('event_summary')} です。"


def escape_table(value: Any) -> str:
    text = str(value or "").replace("\n", " ")
    return text.replace("|", "\\|")


def is_done(status: Any) -> bool:
    text = str(status or "").strip().lower()
    return text in {"done", "closed", "completed", "完了", "対応済み", "終了"}


def first_source_ref(event: dict) -> dict:
    refs = event.get("source_refs") or []
    for ref in refs:
        if isinstance(ref, dict):
            return ref
    return {}


def source_label(event: dict) -> str:
    ref = first_source_ref(event)
    source_type = ref.get("source_type") or ""
    source_file = ref.get("source_file") or ""
    if source_type == "slide":
        return f"{source_file} / Slide {ref.get('slide_no')}: {ref.get('slide_title')}"
    if source_type == "minutes":
        return f"{source_file} / {ref.get('agenda') or ''} / {ref.get('section_title') or ''} / {ref.get('item_type') or ''}"
    return source_file or str(event.get("event_id") or "")
