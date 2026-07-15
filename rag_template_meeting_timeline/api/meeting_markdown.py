from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import re
from pathlib import Path
from typing import Any


MEETING_DOCUMENT_TYPES = {"meeting_slide_deck", "meeting_minutes"}
EVENT_TYPES = {
    "proposal",
    "discussion",
    "concern",
    "decision",
    "action_item",
    "change",
    "report",
    "completion",
    "pending",
    "rejection",
}
BLANK_VALUES = {"", '""', "''", "なし", "無し", "不明", "明記なし", "取得できない"}


def stable_id(*parts: str) -> str:
    raw = "||".join(str(part or "") for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class MeetingMeta:
    meeting_id: str
    meeting_name: str
    meeting_date: str
    fiscal_year: str | None
    department: str | None
    source_file: str
    document_type: str
    confidentiality: str | None


@dataclass
class SlideRecord:
    meeting_id: str
    meeting_name: str
    meeting_date: str
    source_file: str
    slide_no: int
    slide_title: str
    agenda: str | None
    section: str | None
    visible_text: str
    tables_markdown: str | None
    figures_and_charts: str | None
    visual_summary: str | None
    speaker_notes: str | None
    proposed_events: list[dict] = field(default_factory=list)
    search_tags: list[str] = field(default_factory=list)
    source_type: str = "slide"


@dataclass
class MinutesSectionRecord:
    meeting_id: str
    meeting_name: str
    meeting_date: str
    source_file: str
    agenda: str | None
    topic: str | None
    section_title: str
    explanation: str | None
    discussion: str | None
    decisions: list[dict] = field(default_factory=list)
    concerns: list[dict] = field(default_factory=list)
    action_items: list[dict] = field(default_factory=list)
    pending_items: list[dict] = field(default_factory=list)
    rejected_items: list[dict] = field(default_factory=list)
    search_tags: list[str] = field(default_factory=list)
    section_type: str | None = None
    source_type: str = "minutes"


@dataclass
class MeetingEvent:
    event_id: str
    meeting_id: str
    meeting_name: str
    meeting_date: str
    event_date: str
    topic: str
    subtopic: str | None
    event_type: str
    event_summary: str
    before_state: str | None
    after_state: str | None
    owner: str | None
    due_date: str | None
    status: str | None
    source_refs: list[dict] = field(default_factory=list)
    confidence: float | None = None


@dataclass
class ParsedMeetingMarkdown:
    meta: MeetingMeta | None
    slides: list[SlideRecord] = field(default_factory=list)
    minutes_sections: list[MinutesSectionRecord] = field(default_factory=list)
    events: list[MeetingEvent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    text = _strip_leading_markdown_comments(text)
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    end_index = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end_index = idx
            break
    if end_index is None:
        return {}, text

    meta: dict[str, str] = {}
    for line in lines[1:end_index]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, value = _split_key_value(line)
        if key:
            meta[key] = value
    return meta, "\n".join(lines[end_index + 1 :]).lstrip()


def document_type_from_markdown(text: str) -> str:
    meta, _ = parse_frontmatter(text)
    return str(meta.get("document_type") or "").strip()


def parse_meeting_markdown_file(path: Path) -> ParsedMeetingMarkdown:
    try:
        return parse_meeting_markdown_text(path.read_text(encoding="utf-8"), source_path=path)
    except Exception as exc:
        return ParsedMeetingMarkdown(meta=None, errors=[f"{path}: {exc}"])


def parse_meeting_markdown_text(text: str, source_path: Path | None = None) -> ParsedMeetingMarkdown:
    frontmatter, body = parse_frontmatter(text)
    document_type = str(frontmatter.get("document_type") or "").strip()
    source_name = str(source_path or frontmatter.get("source_file") or "")
    parsed = ParsedMeetingMarkdown(meta=None)

    if document_type not in MEETING_DOCUMENT_TYPES:
        parsed.errors.append(f"unsupported document_type: {document_type or '(missing)'}")
        return parsed

    meeting_date = str(frontmatter.get("meeting_date") or "").strip()
    if not meeting_date:
        parsed.errors.append(f"{source_name}: meeting_date is required")
        return parsed

    meeting_id = str(frontmatter.get("meeting_id") or "").strip()
    if not meeting_id:
        parsed.warnings.append(f"{source_name}: meeting_id is missing")

    meta = MeetingMeta(
        meeting_id=meeting_id,
        meeting_name=str(frontmatter.get("meeting_name") or frontmatter.get("deck_title") or "").strip(),
        meeting_date=meeting_date,
        fiscal_year=_none_if_blank(frontmatter.get("fiscal_year")),
        department=_none_if_blank(frontmatter.get("department")),
        source_file=str(frontmatter.get("source_file") or source_path or "").strip(),
        document_type=document_type,
        confidentiality=_none_if_blank(frontmatter.get("confidentiality")),
    )
    parsed.meta = meta

    if document_type == "meeting_slide_deck":
        parsed.slides = _parse_slides(body, meta)
        for slide in parsed.slides:
            parsed.events.extend(_events_from_slide(slide, document_type, parsed.warnings))
    else:
        parsed.minutes_sections = _parse_minutes_sections(body, meta)
        explicit_events = _parse_explicit_meeting_events(body, meta)
        if explicit_events:
            parsed.events.extend(explicit_events)
        else:
            for section in parsed.minutes_sections:
                parsed.events.extend(_events_from_minutes_section(section, document_type))

    if not parsed.slides and not parsed.minutes_sections:
        parsed.warnings.append(f"{source_name}: no meeting records parsed")
    return parsed


def build_meeting_chunks(
    parsed: ParsedMeetingMarkdown,
    corpus_id: str,
    markdown_file: str,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    parents: list[dict] = []
    children: list[dict] = []
    reports: list[dict] = []
    events = [asdict(event) for event in parsed.events]

    for slide in parsed.slides:
        parent, slide_children = _build_slide_chunks(slide, corpus_id, markdown_file)
        parents.append(parent)
        children.extend(slide_children)
        reports.append(_report_row(corpus_id, markdown_file, parent, slide_children, parsed))

    for section in parsed.minutes_sections:
        parent, section_children = _build_minutes_chunks(section, corpus_id, markdown_file)
        parents.append(parent)
        children.extend(section_children)
        reports.append(_report_row(corpus_id, markdown_file, parent, section_children, parsed))

    return parents, children, reports, events


def _parse_slides(body: str, meta: MeetingMeta) -> list[SlideRecord]:
    matches = list(re.finditer(r"(?m)^#\s+Slide\s+(\d+)\s*:?\s*(.*?)\s*$", body))
    slides: list[SlideRecord] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
        block = body[start:end].strip()
        sections = _h2_sections(block)
        slide_meta = _parse_key_values(sections.get("slide_metadata", ""))
        slide_no = _to_int(slide_meta.get("slide_no"), _to_int(match.group(1), idx + 1))
        slide_title = str(slide_meta.get("slide_title") or match.group(2) or f"Slide {slide_no}").strip()
        slides.append(
            SlideRecord(
                meeting_id=meta.meeting_id,
                meeting_name=meta.meeting_name,
                meeting_date=meta.meeting_date,
                source_file=meta.source_file,
                slide_no=slide_no,
                slide_title=slide_title,
                agenda=_none_if_blank(slide_meta.get("agenda")),
                section=_none_if_blank(slide_meta.get("section")),
                visible_text=sections.get("visible_text", "").strip(),
                tables_markdown=_none_if_blank(sections.get("tables")),
                figures_and_charts=_none_if_blank(sections.get("figures_and_charts")),
                visual_summary=_none_if_blank(sections.get("visual_summary")),
                speaker_notes=_none_if_blank(sections.get("speaker_notes")),
                proposed_events=_parse_list_dicts(sections.get("proposed_events", "")),
                search_tags=_parse_search_tags(sections.get("search_tags", "")),
            )
        )
    return slides


def _parse_minutes_sections(body: str, meta: MeetingMeta) -> list[MinutesSectionRecord]:
    matches = list(re.finditer(r"(?m)^#\s+(.+?)\s*$", body))
    records: list[MinutesSectionRecord] = []
    for idx, match in enumerate(matches):
        section_title = match.group(1).strip()
        normalized = section_title.strip().lower()
        if normalized in {"meeting_overview", "meeting_events"}:
            continue
        if not (section_title.startswith("議題") or section_title.startswith("Agenda") or "議題" in section_title):
            continue

        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
        block = body[start:end].strip()
        sections = _h2_sections(block)
        section_meta = _parse_key_values(sections.get("section_metadata", ""))
        records.append(
            MinutesSectionRecord(
                meeting_id=meta.meeting_id,
                meeting_name=meta.meeting_name,
                meeting_date=meta.meeting_date,
                source_file=meta.source_file,
                agenda=_none_if_blank(section_meta.get("agenda")),
                topic=_none_if_blank(section_meta.get("topic")),
                section_title=section_title,
                explanation=_none_if_blank(sections.get("explanation")),
                discussion=_none_if_blank(sections.get("discussion")),
                decisions=_parse_list_dicts(sections.get("decisions", "")),
                concerns=_parse_list_dicts(sections.get("concerns", "")),
                action_items=_parse_list_dicts(sections.get("action_items", "")),
                pending_items=_parse_list_dicts(sections.get("pending_items", "")),
                rejected_items=_parse_list_dicts(sections.get("rejected_items", "")),
                search_tags=_parse_search_tags(sections.get("search_tags", "")),
                section_type=_none_if_blank(section_meta.get("section_type")),
            )
        )
    return records


def _events_from_slide(slide: SlideRecord, document_type: str, warnings: list[str]) -> list[MeetingEvent]:
    events: list[MeetingEvent] = []
    for idx, item in enumerate(slide.proposed_events, start=1):
        raw_type = _clean_scalar(item.get("event_type") or item.get("type") or "proposal").lower()
        if raw_type in {"", "none"}:
            continue
        event_type = raw_type if raw_type in EVENT_TYPES else "proposal"
        if event_type == "decision":
            event_type = "proposal"
            warnings.append(
                f"{slide.source_file} slide {slide.slide_no}: proposed event was downgraded from decision to proposal"
            )
        summary = _first_value(item, ["summary", "event_summary", "proposal", "report", "text"])
        if not summary:
            continue
        topic = _first_value(item, ["topic", "agenda"]) or slide.agenda or slide.slide_title
        events.append(
            _make_event(
                meeting_id=slide.meeting_id,
                meeting_name=slide.meeting_name,
                meeting_date=slide.meeting_date,
                event_type=event_type,
                topic=topic,
                summary=summary,
                item=item,
                source_refs=[
                    {
                        "source_type": "slide",
                        "document_type": document_type,
                        "source_file": slide.source_file,
                        "slide_no": slide.slide_no,
                        "slide_title": slide.slide_title,
                    }
                ],
                fallback_key=f"slide:{slide.slide_no}:{idx}",
            )
        )
    return events


def _events_from_minutes_section(section: MinutesSectionRecord, document_type: str) -> list[MeetingEvent]:
    source_base = {
        "source_type": "minutes",
        "document_type": document_type,
        "source_file": section.source_file,
        "agenda": section.agenda,
        "section_title": section.section_title,
    }
    specs = [
        ("decisions", "decision", ["decision", "summary", "event_summary"]),
        ("concerns", "concern", ["concern", "summary", "event_summary"]),
        ("action_items", "action_item", ["action_item", "task", "summary", "event_summary"]),
        ("pending_items", "pending", ["pending_item", "pending", "summary", "event_summary"]),
        ("rejected_items", "rejection", ["rejected_item", "rejection", "summary", "event_summary"]),
    ]
    events: list[MeetingEvent] = []
    for field_name, event_type, summary_keys in specs:
        rows = getattr(section, field_name)
        for idx, item in enumerate(rows, start=1):
            summary = _first_value(item, summary_keys)
            if not summary:
                continue
            events.append(
                _make_event(
                    meeting_id=section.meeting_id,
                    meeting_name=section.meeting_name,
                    meeting_date=section.meeting_date,
                    event_type=event_type,
                    topic=_first_value(item, ["topic"]) or section.topic or section.agenda or section.section_title,
                    summary=summary,
                    item=item,
                    source_refs=[{**source_base, "item_type": field_name}],
                    fallback_key=f"{section.section_title}:{field_name}:{idx}",
                )
            )
    if section.discussion:
        events.append(
            _make_event(
                meeting_id=section.meeting_id,
                meeting_name=section.meeting_name,
                meeting_date=section.meeting_date,
                event_type="discussion",
                topic=section.topic or section.agenda or section.section_title,
                summary=_compact_text(section.discussion),
                item={},
                source_refs=[{**source_base, "item_type": "discussion"}],
                fallback_key=f"{section.section_title}:discussion",
            )
        )
    return events


def _parse_explicit_meeting_events(body: str, meta: MeetingMeta) -> list[MeetingEvent]:
    match = re.search(r"(?ms)^#\s+meeting_events\s*$([\s\S]*)", body)
    if not match:
        return []
    events: list[MeetingEvent] = []
    for idx, item in enumerate(_parse_list_dicts(match.group(1)), start=1):
        event_type = _clean_scalar(item.get("event_type") or "").lower()
        if event_type not in EVENT_TYPES:
            continue
        summary = _first_value(item, ["event_summary", "summary", "text"])
        if not summary:
            continue
        events.append(
            _make_event(
                meeting_id=meta.meeting_id,
                meeting_name=meta.meeting_name,
                meeting_date=meta.meeting_date,
                event_type=event_type,
                topic=_first_value(item, ["topic"]) or "",
                summary=summary,
                item=item,
                source_refs=[
                    {
                        "source_type": "minutes",
                        "document_type": meta.document_type,
                        "source_file": meta.source_file,
                        "section_title": _none_if_blank(item.get("source_section")),
                        "item_type": event_type,
                    }
                ],
                fallback_key=f"explicit:{idx}",
            )
        )
    return events


def _make_event(
    *,
    meeting_id: str,
    meeting_name: str,
    meeting_date: str,
    event_type: str,
    topic: str,
    summary: str,
    item: dict,
    source_refs: list[dict],
    fallback_key: str,
) -> MeetingEvent:
    event_date = _none_if_blank(item.get("event_date") or item.get("date")) or meeting_date
    confidence = _to_float(item.get("confidence"))
    event_id = stable_id(meeting_id, meeting_date, event_type, topic, summary, fallback_key)
    return MeetingEvent(
        event_id=event_id,
        meeting_id=meeting_id,
        meeting_name=meeting_name,
        meeting_date=meeting_date,
        event_date=event_date,
        topic=_clean_scalar(topic),
        subtopic=_none_if_blank(item.get("subtopic")),
        event_type=event_type,
        event_summary=_clean_scalar(summary),
        before_state=_none_if_blank(item.get("before_state")),
        after_state=_none_if_blank(item.get("after_state")),
        owner=_none_if_blank(item.get("owner") or item.get("担当")),
        due_date=_none_if_blank(item.get("due_date") or item.get("期限")),
        status=_none_if_blank(item.get("status") or item.get("状態")),
        source_refs=source_refs,
        confidence=confidence,
    )


def _build_slide_chunks(slide: SlideRecord, corpus_id: str, markdown_file: str) -> tuple[dict, list[dict]]:
    parent_id = stable_id(corpus_id, markdown_file, "slide", str(slide.slide_no), slide.slide_title)
    base = {
        "corpus_id": corpus_id,
        "document_type": "meeting_slide_deck",
        "source_type": "slide",
        "meeting_id": slide.meeting_id,
        "meeting_name": slide.meeting_name,
        "meeting_date": slide.meeting_date,
        "source_file": slide.source_file,
        "markdown_file": markdown_file,
        "slide_no": slide.slide_no,
        "slide_title": slide.slide_title,
        "agenda": slide.agenda,
        "section": slide.section,
    }
    heading_path = f"{slide.meeting_name} > Slide {slide.slide_no}: {slide.slide_title}".strip()
    parent_text = _labeled_text(
        [
            ("meeting_name", slide.meeting_name),
            ("meeting_date", slide.meeting_date),
            ("source_file", slide.source_file),
            ("slide_no", str(slide.slide_no)),
            ("slide_title", slide.slide_title),
            ("agenda", slide.agenda),
            ("section", slide.section),
            ("visible_text", slide.visible_text),
            ("tables_markdown", slide.tables_markdown),
            ("figures_and_charts", slide.figures_and_charts),
            ("visual_summary", slide.visual_summary),
            ("speaker_notes", slide.speaker_notes),
        ]
    )
    parent = {
        **base,
        "parent_id": parent_id,
        "title": f"{slide.meeting_name} Slide {slide.slide_no}: {slide.slide_title}".strip(),
        "heading_path": heading_path,
        "text": parent_text,
        "search_tags": slide.search_tags,
    }
    children = _build_children(
        parent_id=parent_id,
        base=base,
        title=parent["title"],
        heading_path=heading_path,
        fields=[
            ("slide_title", slide.slide_title),
            ("visible_text", slide.visible_text),
            ("tables_markdown", slide.tables_markdown),
            ("figures_and_charts", slide.figures_and_charts),
            ("visual_summary", slide.visual_summary),
            ("speaker_notes", slide.speaker_notes),
            ("proposed_events", _format_list(slide.proposed_events)),
            ("search_tags", "\n".join(f"- {tag}" for tag in slide.search_tags)),
        ],
        search_tags=slide.search_tags,
    )
    return parent, children


def _build_minutes_chunks(section: MinutesSectionRecord, corpus_id: str, markdown_file: str) -> tuple[dict, list[dict]]:
    parent_id = stable_id(corpus_id, markdown_file, "minutes", section.section_title, section.topic or "")
    base = {
        "corpus_id": corpus_id,
        "document_type": "meeting_minutes",
        "source_type": "minutes",
        "meeting_id": section.meeting_id,
        "meeting_name": section.meeting_name,
        "meeting_date": section.meeting_date,
        "source_file": section.source_file,
        "markdown_file": markdown_file,
        "agenda": section.agenda,
        "topic": section.topic,
        "section_title": section.section_title,
        "section_type": section.section_type,
    }
    heading_path = f"{section.meeting_name} > {section.section_title}".strip()
    parent_text = _labeled_text(
        [
            ("meeting_name", section.meeting_name),
            ("meeting_date", section.meeting_date),
            ("source_file", section.source_file),
            ("agenda", section.agenda),
            ("topic", section.topic),
            ("explanation", section.explanation),
            ("discussion", section.discussion),
            ("decisions", _format_list(section.decisions)),
            ("concerns", _format_list(section.concerns)),
            ("action_items", _format_list(section.action_items)),
            ("pending_items", _format_list(section.pending_items)),
            ("rejected_items", _format_list(section.rejected_items)),
        ]
    )
    parent = {
        **base,
        "parent_id": parent_id,
        "title": f"{section.meeting_name} {section.section_title}".strip(),
        "heading_path": heading_path,
        "text": parent_text,
        "search_tags": section.search_tags,
    }
    children = _build_children(
        parent_id=parent_id,
        base=base,
        title=parent["title"],
        heading_path=heading_path,
        fields=[
            ("explanation", section.explanation),
            ("discussion", section.discussion),
            ("decisions", _format_list(section.decisions)),
            ("concerns", _format_list(section.concerns)),
            ("action_items", _format_list(section.action_items)),
            ("pending_items", _format_list(section.pending_items)),
            ("rejected_items", _format_list(section.rejected_items)),
            ("search_tags", "\n".join(f"- {tag}" for tag in section.search_tags)),
        ],
        search_tags=section.search_tags,
    )
    return parent, children


def _build_children(
    *,
    parent_id: str,
    base: dict,
    title: str,
    heading_path: str,
    fields: list[tuple[str, str | None]],
    search_tags: list[str],
) -> list[dict]:
    children: list[dict] = []
    for idx, (content_type, value) in enumerate(fields, start=1):
        text = _clean_scalar(value)
        if _is_blank_value(text):
            continue
        child_text = f"{content_type}\n{text}"
        child_id = stable_id(parent_id, content_type, str(idx), child_text[:120])
        row = {
            **base,
            "parent_id": parent_id,
            "child_id": child_id,
            "title": title,
            "heading_path": f"{heading_path} > {content_type}",
            "content_type": content_type,
            "text": child_text,
            "search_tags": search_tags,
        }
        row["search_text"] = "\n".join([title, row["heading_path"], child_text, " ".join(search_tags)]).strip()
        children.append(row)
    return children


def _report_row(
    corpus_id: str,
    markdown_file: str,
    parent: dict,
    children: list[dict],
    parsed: ParsedMeetingMarkdown,
) -> dict:
    return {
        "corpus_id": corpus_id,
        "source_file": markdown_file,
        "title": parent.get("title", ""),
        "document_type": parent.get("document_type", ""),
        "source_type": parent.get("source_type", ""),
        "meeting_id": parent.get("meeting_id", ""),
        "parent_count": 1,
        "child_count": len(children),
        "warnings": "; ".join(parsed.warnings),
        "errors": "; ".join(parsed.errors),
    }


def _h2_sections(text: str) -> dict[str, str]:
    matches = list(re.finditer(r"(?m)^##\s+(.+?)\s*$", text))
    sections: dict[str, str] = {}
    for idx, match in enumerate(matches):
        key = _normalize_key(match.group(1))
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        sections[key] = text[start:end].strip()
    return sections


def _strip_leading_markdown_comments(text: str) -> str:
    lines = text.splitlines()
    idx = 0
    while idx < len(lines):
        stripped = lines[idx].strip()
        if not stripped:
            idx += 1
            continue
        if stripped.startswith("<!--") and stripped.endswith("-->"):
            idx += 1
            continue
        break
    return "\n".join(lines[idx:])


def _parse_key_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("- "):
            line = line[2:].strip()
        key, value = _split_key_value(line)
        if key:
            values[key] = value
    return values


def _parse_list_dicts(text: str) -> list[dict]:
    if _is_blank_value(text):
        return []
    rows: list[dict] = []
    current: dict[str, Any] | None = None
    for raw in text.splitlines():
        if not raw.strip():
            continue
        bullet = re.match(r"^\s*-\s*(.*)$", raw)
        if bullet:
            if current and not _is_empty_list_item(current):
                rows.append(current)
            current = {}
            first = bullet.group(1).strip()
            key, value = _split_key_value(first)
            if key:
                current[key] = value
            elif first:
                current["summary"] = first
            continue
        if current is None:
            continue
        key, value = _split_key_value(raw.strip())
        if key:
            current[key] = value
        else:
            current["text"] = "\n".join([current.get("text", ""), raw.strip()]).strip()
    if current and not _is_empty_list_item(current):
        rows.append(current)
    return rows


def _parse_search_tags(text: str) -> list[str]:
    if _is_blank_value(text):
        return []
    tags: list[str] = []
    seen = set()
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("- "):
            line = line[2:].strip()
        if not line:
            continue
        for part in re.split(r"[,、]", line):
            tag = part.strip()
            if tag and not _is_blank_value(tag) and tag not in seen:
                seen.add(tag)
                tags.append(tag)
    return tags


def _split_key_value(line: str) -> tuple[str | None, str]:
    if ":" not in line:
        return None, _clean_scalar(line)
    key, value = line.split(":", 1)
    key = key.strip()
    if not key:
        return None, _clean_scalar(value)
    return key, _clean_scalar(value)


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", value.strip().lower()).strip("_")


def _clean_scalar(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    return text


def _is_blank_value(value: Any) -> bool:
    text = _clean_scalar(value)
    return text in BLANK_VALUES


def _is_empty_list_item(item: dict) -> bool:
    meaningful = []
    for key, value in item.items():
        if key == "event_type" and _clean_scalar(value).lower() == "none":
            continue
        if not _is_blank_value(value):
            meaningful.append(value)
    return not meaningful


def _none_if_blank(value: Any) -> str | None:
    if value is None:
        return None
    text = _clean_scalar(value)
    if _is_blank_value(text):
        return None
    return text or None


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(_clean_scalar(value))
    except Exception:
        return default


def _to_float(value: Any) -> float | None:
    if value is None or _is_blank_value(value):
        return None
    try:
        return float(_clean_scalar(value))
    except Exception:
        return None


def _first_value(item: dict, keys: list[str]) -> str:
    for key in keys:
        value = item.get(key)
        if value is not None and not _is_blank_value(value):
            return _clean_scalar(value)
    return ""


def _compact_text(value: str, limit: int = 400) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit] + ("..." if len(text) > limit else "")


def _labeled_text(items: list[tuple[str, str | None]]) -> str:
    blocks = []
    for label, value in items:
        text = _clean_scalar(value)
        if not _is_blank_value(text):
            blocks.append(f"## {label}\n{text}")
    return "\n\n".join(blocks).strip()


def _format_list(rows: list[dict]) -> str:
    if not rows:
        return ""
    lines = []
    for row in rows:
        if not row or _is_empty_list_item(row):
            continue
        lines.append("- " + json.dumps(row, ensure_ascii=False, sort_keys=True))
    return "\n".join(lines)
