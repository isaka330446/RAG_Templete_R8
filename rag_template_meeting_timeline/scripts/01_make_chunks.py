from pathlib import Path
import csv
import json
import re
import hashlib
import sys
from typing import Any, List, Tuple


BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from api.meeting_event_store import MeetingEventStore
from api.meeting_markdown import (
    MEETING_DOCUMENT_TYPES,
    build_meeting_chunks,
    document_type_from_markdown,
    parse_meeting_markdown_file,
)


CONFIG_PATH = BASE_DIR / "config" / "corpus_settings.json"
CHUNK_DIR = BASE_DIR / "chunks"
PARENT_OUT = CHUNK_DIR / "parent_chunks.jsonl"
CHILD_OUT = CHUNK_DIR / "child_chunks.jsonl"
CHUNK_REPORT_OUT = CHUNK_DIR / "chunk_report.csv"
MEETING_EVENT_STORE = MeetingEventStore()


def stable_id(*parts: str) -> str:
    raw = "||".join(parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def load_corpora() -> List[dict]:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return [c for c in data.get("corpora", []) if c.get("enabled", True)]


def strip_yaml_frontmatter(text: str) -> str:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end + 4 :].lstrip()
    return text


def parse_heading(line: str) -> Tuple[int, str] | None:
    m = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
    if not m:
        return None
    return len(m.group(1)), m.group(2).strip()


def split_to_parent_sections(text: str) -> List[dict]:
    lines = text.splitlines()
    sections = []
    current = []
    heading_stack: List[Tuple[int, str]] = []
    current_heading_path = ""

    def flush():
        nonlocal current, current_heading_path
        body = "\n".join(current).strip()
        content_lines = [
            line for line in current
            if line.strip() and parse_heading(line) is None
        ]
        if body:
            if not content_lines:
                current = []
                return
            sections.append({
                "heading_path": current_heading_path,
                "text": body,
            })
        current = []

    for line in lines:
        h = parse_heading(line)
        if h and h[0] <= 3:
            flush()
            level, title = h
            heading_stack[:] = [(lv, t) for lv, t in heading_stack if lv < level]
            heading_stack.append((level, title))
            current_heading_path = " > ".join(t for _, t in heading_stack)
            current.append(line)
        else:
            current.append(line)
    flush()

    if not sections and text.strip():
        sections.append({"heading_path": "", "text": text.strip()})

    max_chars = 20000
    normalized = []
    for sec in sections:
        body = sec["text"]
        if len(body) <= max_chars:
            normalized.append(sec)
            continue
        paras = re.split(r"\n{2,}", body)
        buf = []
        idx = 1
        for p in paras:
            if sum(len(x) for x in buf) + len(p) > max_chars and buf:
                normalized.append({
                    "heading_path": f'{sec["heading_path"]} / part{idx}',
                    "text": "\n\n".join(buf).strip(),
                })
                idx += 1
                buf = []
            buf.append(p)
        if buf:
            normalized.append({
                "heading_path": f'{sec["heading_path"]} / part{idx}' if idx > 1 else sec["heading_path"],
                "text": "\n\n".join(buf).strip(),
            })
    return normalized


def split_child_chunks(parent_text: str, max_chars: int = 1200, overlap_chars: int = 150) -> List[str]:
    blocks = re.split(r"(?=\n#{1,6}\s+|\n第[0-9一二三四五六七八九十百]+条|\n\([0-9一二三四五六七八九十]+\))", "\n" + parent_text)
    blocks = [b.strip() for b in blocks if b.strip()]

    chunks = []
    buf = ""
    for b in blocks:
        if len(buf) + len(b) + 2 <= max_chars:
            buf = (buf + "\n\n" + b).strip()
        else:
            if buf:
                chunks.append(buf)
            if len(b) <= max_chars:
                buf = b
            else:
                start = 0
                while start < len(b):
                    end = start + max_chars
                    chunks.append(b[start:end])
                    start = max(0, end - overlap_chars)
                    if start >= len(b):
                        break
                buf = ""
    if buf:
        chunks.append(buf)

    return chunks


def build_generic_chunks(corpus_id: str, md_path: Path) -> tuple[list[dict], list[dict], dict]:
    raw = md_path.read_text(encoding="utf-8")
    text = strip_yaml_frontmatter(raw)
    rel = md_path.relative_to(BASE_DIR).as_posix()
    title = md_path.stem
    parents = []
    children = []

    parent_sections = split_to_parent_sections(text)
    for p_idx, sec in enumerate(parent_sections, start=1):
        parent_id = stable_id(corpus_id, rel, str(p_idx), sec["heading_path"])
        parent = {
            "parent_id": parent_id,
            "corpus_id": corpus_id,
            "title": title,
            "source_file": rel,
            "heading_path": sec["heading_path"],
            "text": sec["text"],
        }
        parents.append(parent)

        child_texts = split_child_chunks(sec["text"])
        for c_idx, child_text in enumerate(child_texts, start=1):
            child_id = stable_id(parent_id, str(c_idx), child_text[:80])
            children.append({
                "child_id": child_id,
                "parent_id": parent_id,
                "corpus_id": corpus_id,
                "title": title,
                "source_file": rel,
                "heading_path": sec["heading_path"],
                "text": child_text,
            })

    report = {
        "corpus_id": corpus_id,
        "source_file": rel,
        "title": title,
        "document_type": "markdown",
        "source_type": "",
        "meeting_id": "",
        "parent_count": len(parents),
        "child_count": len(children),
        "warnings": "",
        "errors": "",
    }
    return parents, children, report


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_chunk_report(rows: list[dict]) -> None:
    fieldnames = [
        "corpus_id",
        "source_file",
        "title",
        "document_type",
        "source_type",
        "meeting_id",
        "parent_count",
        "child_count",
        "warnings",
        "errors",
    ]
    with CHUNK_REPORT_OUT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def error_report(corpus_id: str, md_path: Path, errors: list[str], warnings: list[str] | None = None) -> dict:
    rel = md_path.relative_to(BASE_DIR).as_posix()
    return {
        "corpus_id": corpus_id,
        "source_file": rel,
        "title": md_path.stem,
        "document_type": "",
        "source_type": "",
        "meeting_id": "",
        "parent_count": 0,
        "child_count": 0,
        "warnings": "; ".join(warnings or []),
        "errors": "; ".join(errors),
    }


def main():
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    parents: list[dict[str, Any]] = []
    children: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    meeting_events: list[dict[str, Any]] = []

    for corpus in load_corpora():
        corpus_id = corpus["corpus_id"]
        md_dir = BASE_DIR / corpus["markdown_dir"]
        if not md_dir.exists():
            continue

        for md_path in sorted(md_dir.rglob("*.md")):
            raw = md_path.read_text(encoding="utf-8")
            document_type = document_type_from_markdown(raw)
            rel = md_path.relative_to(BASE_DIR).as_posix()

            if document_type in MEETING_DOCUMENT_TYPES:
                parsed = parse_meeting_markdown_file(md_path)
                if parsed.errors:
                    reports.append(error_report(corpus_id, md_path, parsed.errors, parsed.warnings))
                    print(f"meeting parse error: {rel} errors={len(parsed.errors)} warnings={len(parsed.warnings)}")
                    continue
                meeting_parents, meeting_children, meeting_reports, events = build_meeting_chunks(parsed, corpus_id, rel)
                parents.extend(meeting_parents)
                children.extend(meeting_children)
                reports.extend(meeting_reports)
                meeting_events.extend(events)
                if parsed.warnings:
                    print(f"meeting parse warning: {rel} warnings={len(parsed.warnings)}")
                continue

            generic_parents, generic_children, report = build_generic_chunks(corpus_id, md_path)
            parents.extend(generic_parents)
            children.extend(generic_children)
            reports.append(report)

    write_jsonl(PARENT_OUT, parents)
    write_jsonl(CHILD_OUT, children)
    write_chunk_report(reports)
    MEETING_EVENT_STORE.replace_all(meeting_events)

    print(f"parents: {len(parents)} -> {PARENT_OUT}")
    print(f"children: {len(children)} -> {CHILD_OUT}")
    print(f"chunk_report: {len(reports)} -> {CHUNK_REPORT_OUT}")
    print(f"meeting_events: {len(meeting_events)} -> {MEETING_EVENT_STORE.path}")


if __name__ == "__main__":
    main()
