from pathlib import Path
import argparse
import json
import sys


BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from api.meeting_event_store import MeetingEventStore


def source_label(event: dict) -> str:
    refs = event.get("source_refs") or []
    ref = refs[0] if refs and isinstance(refs[0], dict) else {}
    source_type = ref.get("source_type") or ""
    if source_type == "slide":
        return f"{ref.get('source_file')} / Slide {ref.get('slide_no')}: {ref.get('slide_title')}"
    if source_type == "minutes":
        return f"{ref.get('source_file')} / {ref.get('agenda') or ''} / {ref.get('section_title') or ''} / {ref.get('item_type') or ''}"
    return ref.get("source_file") or event.get("event_id") or ""


def print_markdown(events: list[dict]) -> None:
    print("| 日付 | 会議名 | 種別 | トピック | 内容 | 担当 | 期限 | 根拠 |")
    print("|---|---|---|---|---|---|---|---|")
    for event in events:
        values = [
            event.get("event_date") or event.get("meeting_date") or "",
            event.get("meeting_name") or "",
            event.get("event_type") or "",
            event.get("topic") or "",
            event.get("event_summary") or "",
            event.get("owner") or "",
            event.get("due_date") or "",
            source_label(event),
        ]
        print("| " + " | ".join(str(value).replace("\n", " ").replace("|", "\\|") for value in values) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Query MeetingEvent JSONL and print a sorted timeline.")
    parser.add_argument("--meeting-id")
    parser.add_argument("--topic")
    parser.add_argument("--event-type")
    parser.add_argument("--date-from")
    parser.add_argument("--date-to")
    parser.add_argument("--status")
    parser.add_argument("--owner")
    parser.add_argument("--source-type", choices=["slide", "minutes"])
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead of Markdown table.")
    args = parser.parse_args()

    events = MeetingEventStore().query(
        meeting_id=args.meeting_id,
        topic=args.topic,
        event_type=args.event_type,
        date_from=args.date_from,
        date_to=args.date_to,
        status=args.status,
        owner=args.owner,
        source_type=args.source_type,
        limit=args.limit,
    )
    if args.json:
        print(json.dumps({"events": events}, ensure_ascii=False, indent=2))
    else:
        print_markdown(events)


if __name__ == "__main__":
    main()
