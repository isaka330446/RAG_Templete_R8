# Markdown化済み文書から親チャンク、子チャンク、チャンク監査レポートを生成します。
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import hashlib
import json
import re
import sys
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from api.text_cleaning import clean_rag_text

CONFIG_PATH = BASE_DIR / "config" / "corpus_settings.json"
CHUNK_DIR = BASE_DIR / "chunks"
PARENT_OUT = CHUNK_DIR / "parent_chunks.jsonl"
CHILD_OUT = CHUNK_DIR / "child_chunks.jsonl"
CHUNK_REPORT_OUT = CHUNK_DIR / "chunk_report.csv"

# Parent chunks are the expansion context sent to the LLM.
# Child chunks are the small retrieval units stored in BM25/Chroma.
MAX_PARENT_CHARS = 12000
MAX_CHILD_CHARS = 900
CHILD_OVERLAP_CHARS = 120

CHUNKING_KEYS = {
    "parent_split_levels",
    "max_parent_chars",
    "min_parent_chars",
    "child_max_chars",
    "child_overlap_chars",
    "force_heading_split",
    "auto_descend",
    "chunking_mode",
}

DEFAULT_CHUNKING = {
    "parent_split_levels": [2],
    "max_parent_chars": MAX_PARENT_CHARS,
    "min_parent_chars": 0,
    "child_max_chars": MAX_CHILD_CHARS,
    "child_overlap_chars": CHILD_OVERLAP_CHARS,
    "force_heading_split": False,
    "auto_descend": False,
    "chunking_mode": "auto",
}

MARKER_COMMENT_RE = re.compile(r"<!--(?P<body>.*?)-->", re.DOTALL)
MARKER_KIND_RE = re.compile(r"\b(?P<kind>parent_chunk|child_chunk)\b\s*[:\uff1a]\s*(?P<id>[^|]+)", re.IGNORECASE)
MARKER_ATTR_RE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*[:\uff1a]\s*(?P<value>.*)")


def stable_id(*parts: str) -> str:
    raw = "||".join(parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def load_corpora() -> list[dict[str, Any]]:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return [c for c in data.get("corpora", []) if c.get("enabled", True)]


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def parse_int(value: Any, default: int, *, minimum: int = 0) -> int:
    try:
        parsed = int(str(value).strip())
    except Exception:
        return default
    return max(minimum, parsed)


def parse_heading_levels(value: Any) -> list[int]:
    if isinstance(value, list):
        raw_values = value
    else:
        raw_values = re.split(r"[,| ]+", str(value or ""))
    levels: list[int] = []
    for raw in raw_values:
        try:
            level = int(str(raw).strip())
        except Exception:
            continue
        if 1 <= level <= 6 and level not in levels:
            levels.append(level)
    return sorted(levels) or [2]


def build_chunking(corpus: dict[str, Any], metadata: dict[str, str]) -> dict[str, Any]:
    chunking = dict(DEFAULT_CHUNKING)
    corpus_chunking = corpus.get("chunking")
    if isinstance(corpus_chunking, dict):
        chunking.update(corpus_chunking)

    for key in CHUNKING_KEYS:
        if metadata.get(key):
            chunking[key] = metadata[key]

    chunking["parent_split_levels"] = parse_heading_levels(chunking.get("parent_split_levels"))
    chunking["max_parent_chars"] = parse_int(chunking.get("max_parent_chars"), MAX_PARENT_CHARS, minimum=500)
    chunking["min_parent_chars"] = parse_int(chunking.get("min_parent_chars"), 0, minimum=0)
    chunking["child_max_chars"] = parse_int(chunking.get("child_max_chars"), MAX_CHILD_CHARS, minimum=200)
    chunking["child_overlap_chars"] = parse_int(chunking.get("child_overlap_chars"), CHILD_OVERLAP_CHARS, minimum=0)
    chunking["force_heading_split"] = parse_bool(chunking.get("force_heading_split"))
    chunking["auto_descend"] = parse_bool(chunking.get("auto_descend"))
    mode = str(chunking.get("chunking_mode") or "auto").strip().lower()
    chunking["chunking_mode"] = mode if mode in {"auto", "heading", "html_comment"} else "auto"
    return chunking


def has_custom_chunking(corpus: dict[str, Any], metadata: dict[str, str]) -> bool:
    return isinstance(corpus.get("chunking"), dict) or any(key in metadata for key in CHUNKING_KEYS)


def parse_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    if not raw.startswith("---"):
        return {}, raw

    end = raw.find("\n---", 3)
    if end == -1:
        return {}, raw

    frontmatter_text = raw[3:end].strip()
    body = raw[end + 4 :].lstrip()
    metadata: dict[str, str] = {}
    for line in frontmatter_text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] == '"':
            value = value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        metadata[key.strip()] = value
    return metadata, body


def parse_heading(line: str) -> tuple[int, str] | None:
    match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
    if not match:
        return None
    return len(match.group(1)), match.group(2).strip()


def document_title(text: str, metadata: dict[str, str], fallback: str) -> str:
    if metadata.get("title"):
        return metadata["title"]
    for line in text.splitlines():
        heading = parse_heading(line)
        if heading and heading[0] == 1:
            return heading[1]
    return fallback


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    return value.strip()


def parse_chunk_markers(text: str) -> list[dict[str, Any]]:
    markers: list[dict[str, Any]] = []
    for match in MARKER_COMMENT_RE.finditer(text):
        body = match.group("body").strip()
        kind_match = MARKER_KIND_RE.search(body)
        if not kind_match:
            continue
        kind = kind_match.group("kind").lower()
        marker_id = _strip_quotes(kind_match.group("id"))
        attrs: dict[str, str] = {}
        for part in body.split("|"):
            attr_match = MARKER_ATTR_RE.match(part.strip())
            if not attr_match:
                continue
            key = attr_match.group("key").strip().lower()
            attrs[key] = _strip_quotes(attr_match.group("value"))
        markers.append(
            {
                "kind": kind,
                "marker_id": marker_id,
                "attrs": attrs,
                "start": match.start(),
                "end": match.end(),
                "raw": match.group(0),
            }
        )
    return markers


def marker_title(marker: dict[str, Any], fallback: str) -> str:
    attrs = marker.get("attrs") or {}
    return attrs.get("title") or marker.get("marker_id") or fallback


def marker_source(marker: dict[str, Any]) -> str:
    attrs = marker.get("attrs") or {}
    return attrs.get("source") or attrs.get("page") or ""


def split_to_marker_parent_sections(
    text: str,
    title: str,
    chunking: dict[str, Any],
    *,
    strip_html_tags: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    markers = parse_chunk_markers(text)
    parent_markers = [m for m in markers if m["kind"] == "parent_chunk"]
    child_markers = [m for m in markers if m["kind"] == "child_chunk"]
    diagnostics: dict[str, Any] = {
        "marker_parent_count": len(parent_markers),
        "marker_child_count": len(child_markers),
        "marker_warnings": [],
    }
    if not parent_markers:
        diagnostics["marker_warnings"].append("html_comment_markers_not_found")
        return [], diagnostics

    parent_sections: list[dict[str, Any]] = []
    preamble = clean_rag_text(text[: parent_markers[0]["start"]], strip_html_tags=strip_html_tags)
    if preamble:
        diagnostics["marker_warnings"].append("text_before_first_parent_marker_ignored")

    for idx, parent_marker in enumerate(parent_markers):
        next_parent_start = parent_markers[idx + 1]["start"] if idx + 1 < len(parent_markers) else len(text)
        segment = text[parent_marker["start"] : next_parent_start]
        parent_marker_id = parent_marker.get("marker_id", "")
        parent_title = marker_title(parent_marker, title)
        parent_heading = parent_title
        parent_text = clean_rag_text(segment, strip_html_tags=strip_html_tags)

        section_child_markers = [
            m for m in child_markers if parent_marker["start"] < m["start"] < next_parent_start
        ]
        child_sections: list[dict[str, Any]] = []
        for c_idx, child_marker in enumerate(section_child_markers):
            child_end = (
                section_child_markers[c_idx + 1]["start"]
                if c_idx + 1 < len(section_child_markers)
                else next_parent_start
            )
            attrs = child_marker.get("attrs") or {}
            declared_parent = attrs.get("parent", "")
            if declared_parent and parent_marker_id and declared_parent != parent_marker_id:
                diagnostics["marker_warnings"].append(
                    f"child_parent_mismatch:{child_marker.get('marker_id')}->{declared_parent}!={parent_marker_id}"
                )
            child_title = marker_title(child_marker, f"child{c_idx + 1}")
            child_heading = f"{parent_heading} > {child_title}" if parent_heading else child_title
            child_text = clean_rag_text(text[child_marker["start"] : child_end], strip_html_tags=strip_html_tags)
            if not child_text:
                diagnostics["marker_warnings"].append(f"empty_child_marker:{child_marker.get('marker_id')}")
                continue
            child_sections.append(
                {
                    "heading_path": child_heading,
                    "text": child_text,
                    "child_marker_id": child_marker.get("marker_id", ""),
                    "marker_source": marker_source(child_marker),
                    "chunking_method": "html_comment",
                }
            )

        if not parent_text:
            diagnostics["marker_warnings"].append(f"empty_parent_marker:{parent_marker_id}")
            continue
        parent_sections.append(
            {
                "heading_path": parent_heading,
                "text": parent_text,
                "parent_marker_id": parent_marker_id,
                "marker_source": marker_source(parent_marker),
                "chunking_method": "html_comment",
                "child_sections": child_sections,
            }
        )

    return parent_sections, diagnostics


def split_marker_child_sections(section: dict[str, Any], chunking: dict[str, Any]) -> list[dict[str, Any]]:
    child_sections = section.get("child_sections") or []
    if not child_sections:
        return split_child_chunks(section["text"], section["heading_path"], chunking)

    rows: list[dict[str, Any]] = []
    max_child_chars = chunking["child_max_chars"]
    overlap_chars = chunking["child_overlap_chars"]
    for child in child_sections:
        if len(child["text"]) <= max_child_chars:
            rows.append(child)
            continue
        for part in split_large_text(child["text"], child["heading_path"], max_child_chars, overlap_chars):
            rows.append(
                {
                    **part,
                    "child_marker_id": child.get("child_marker_id", ""),
                    "marker_source": child.get("marker_source", ""),
                    "chunking_method": "html_comment",
                }
            )
    return rows


def split_by_heading_levels(text: str, base_heading_path: str, levels: list[int]) -> list[dict[str, str]]:
    target_levels = set(levels)
    min_target_level = min(target_levels)
    lines = text.splitlines()
    prefix: list[str] = []
    sections: list[dict[str, str]] = []
    current: list[str] = []
    current_heading = base_heading_path
    seen_target_heading = False
    heading_stack: list[tuple[int, str]] = []

    def flush() -> None:
        nonlocal current
        body = "\n".join(current).strip()
        if body:
            section_text = "\n".join(prefix + [""] + current).strip() if prefix else body
            sections.append({"heading_path": current_heading, "text": section_text})
        current = []

    for line in lines:
        heading = parse_heading(line)
        if heading:
            level, heading_title = heading
            heading_stack[:] = [(lv, t) for lv, t in heading_stack if lv < level]
            heading_stack.append((level, heading_title))

            if level < min_target_level:
                if seen_target_heading:
                    flush()
                    seen_target_heading = False
                    prefix = [line]
                else:
                    prefix.append(line)
                continue

            if level in target_levels:
                path_titles = [t for lv, t in heading_stack if 1 < lv <= level]
                if seen_target_heading:
                    flush()
                else:
                    seen_target_heading = True
                current_heading = base_heading_path
                if path_titles:
                    current_heading = f"{base_heading_path} > {' > '.join(path_titles)}"
                current = [line]
                continue

        if seen_target_heading:
            current.append(line)
        else:
            prefix.append(line)

    if seen_target_heading:
        flush()
    else:
        body = "\n".join(lines).strip()
        if body:
            sections.append({"heading_path": base_heading_path, "text": body})

    return sections


def split_by_h2(text: str, title: str) -> list[dict[str, str]]:
    return split_by_heading_levels(text, title, [2])


def count_headings(text: str, levels: list[int]) -> int:
    target_levels = set(levels)
    count = 0
    for line in text.splitlines():
        heading = parse_heading(line)
        if heading and heading[0] in target_levels:
            count += 1
    return count


def split_large_text(text: str, heading_path: str, max_chars: int, overlap_chars: int) -> list[dict[str, str]]:
    if len(text) <= max_chars:
        return [{"heading_path": heading_path, "text": text}]

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    sections: list[dict[str, str]] = []
    buffer: list[str] = []
    part = 1

    def flush() -> None:
        nonlocal buffer, part
        body = "\n\n".join(buffer).strip()
        if body:
            suffix = f" / part{part}" if part > 1 else ""
            sections.append({"heading_path": f"{heading_path}{suffix}", "text": body})
            part += 1
        buffer = []

    for paragraph in paragraphs:
        current_size = sum(len(x) for x in buffer) + max(0, len(buffer) - 1) * 2
        if buffer and current_size + len(paragraph) + 2 > max_chars:
            flush()

        if len(paragraph) <= max_chars:
            buffer.append(paragraph)
            continue

        # Fallback for exceptionally long tables or paragraphs.
        flush()
        start = 0
        while start < len(paragraph):
            end = min(len(paragraph), start + max_chars)
            chunk = paragraph[start:end].strip()
            if chunk:
                sections.append({"heading_path": f"{heading_path} / part{part}", "text": chunk})
                part += 1
            if end >= len(paragraph):
                break
            start = max(0, end - overlap_chars)

    if buffer:
        flush()
    return sections


def split_oversize_parent(section: dict[str, str], chunking: dict[str, Any], start_level: int) -> list[dict[str, str]]:
    max_parent_chars = chunking["max_parent_chars"]
    if len(section["text"]) <= max_parent_chars:
        return [section]

    if chunking["auto_descend"]:
        for level in range(max(1, start_level), 7):
            nested = split_by_heading_levels(section["text"], section["heading_path"], [level])
            if len(nested) <= 1:
                continue
            rows: list[dict[str, str]] = []
            for nested_section in nested:
                rows.extend(split_oversize_parent(nested_section, chunking, level + 1))
            return rows

    return split_large_text(
        section["text"],
        section["heading_path"],
        max_parent_chars,
        chunking["child_overlap_chars"],
    )


def merge_small_sections(sections: list[dict[str, str]], min_chars: int, max_chars: int) -> list[dict[str, str]]:
    if min_chars <= 0 or len(sections) <= 1:
        return sections

    merged: list[dict[str, str]] = []
    buffer: dict[str, str] | None = None

    def append_buffer() -> None:
        nonlocal buffer
        if buffer:
            merged.append(buffer)
        buffer = None

    for section in sections:
        if buffer is None:
            buffer = dict(section)
            continue

        combined_text = f'{buffer["text"].rstrip()}\n\n{section["text"].strip()}'
        if len(buffer["text"]) < min_chars and len(combined_text) <= max_chars:
            buffer["text"] = combined_text
            buffer["heading_path"] = f'{buffer["heading_path"]} + {section["heading_path"]}'
        else:
            append_buffer()
            buffer = dict(section)

    append_buffer()
    return merged


def split_to_parent_sections(text: str, title: str, chunking: dict[str, Any]) -> list[dict[str, str]]:
    max_parent_chars = chunking["max_parent_chars"]
    levels = chunking["parent_split_levels"]
    should_split_by_heading = chunking["force_heading_split"] or len(text) > max_parent_chars

    if should_split_by_heading:
        sections = split_by_heading_levels(text, title, levels)
    else:
        sections = [{"heading_path": title, "text": text.strip()}]

    parent_sections: list[dict[str, str]] = []
    start_level = max(levels) + 1 if levels else 2
    for section in sections:
        parent_sections.extend(split_oversize_parent(section, chunking, start_level))
    return merge_small_sections(parent_sections, chunking["min_parent_chars"], max_parent_chars)


def split_to_parent_sections_legacy(text: str, title: str) -> list[dict[str, str]]:
    if len(text) <= MAX_PARENT_CHARS:
        return [{"heading_path": title, "text": text.strip()}]

    parent_sections: list[dict[str, str]] = []
    for section in split_by_h2(text, title):
        parent_sections.extend(split_large_text(section["text"], section["heading_path"], MAX_PARENT_CHARS, CHILD_OVERLAP_CHARS))
    return parent_sections


def iter_blocks(text: str, default_heading_path: str) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    heading_stack: list[tuple[int, str]] = []
    current: list[str] = []
    current_path = default_heading_path

    def heading_path() -> str:
        if not heading_stack:
            return default_heading_path
        return " > ".join(title for _, title in heading_stack)

    def flush() -> None:
        nonlocal current
        body = "\n".join(current).strip()
        if body:
            blocks.append({"heading_path": current_path, "text": body})
        current = []

    for line in text.splitlines():
        heading = parse_heading(line)
        if heading:
            flush()
            level, title = heading
            heading_stack[:] = [(lv, t) for lv, t in heading_stack if lv < level]
            heading_stack.append((level, title))
            current_path = heading_path()
            current = [line]
            continue

        if not line.strip():
            flush()
            continue

        if not current:
            current_path = heading_path()
        current.append(line)

    flush()
    return blocks


def split_long_block(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap_chars)
    return chunks


def split_child_chunks(parent_text: str, parent_heading_path: str, chunking: dict[str, Any]) -> list[dict[str, str]]:
    blocks = iter_blocks(parent_text, parent_heading_path)
    chunks: list[dict[str, str]] = []
    buffer: list[str] = []
    buffer_path = parent_heading_path
    max_child_chars = chunking["child_max_chars"]
    overlap_chars = chunking["child_overlap_chars"]

    def flush() -> None:
        nonlocal buffer, buffer_path
        body = "\n\n".join(buffer).strip()
        if body:
            chunks.append({"heading_path": buffer_path, "text": body})
        buffer = []
        buffer_path = parent_heading_path

    for block in blocks:
        block_text = block["text"]
        block_path = block["heading_path"]
        current_size = sum(len(x) for x in buffer) + max(0, len(buffer) - 1) * 2

        if buffer and current_size + len(block_text) + 2 > max_child_chars:
            flush()

        if len(block_text) <= max_child_chars:
            if not buffer:
                buffer_path = block_path
            buffer.append(block_text)
            continue

        flush()
        for chunk in split_long_block(block_text, max_child_chars, overlap_chars):
            chunks.append({"heading_path": block_path, "text": chunk})

    if buffer:
        flush()
    return chunks


def base_row(metadata: dict[str, str]) -> dict[str, str]:
    keys = [
        "document_code",
        "document_type",
        "document_id",
        "document_series",
        "tax_type",
        "category",
        "asset_tax_category",
        "source_url",
        "source_site",
        "pdf_path",
        "version_date",
        "valid_from",
        "valid_until",
    ]
    return {key: metadata[key] for key in keys if metadata.get(key)}


def chunking_label(chunking: dict[str, Any]) -> str:
    levels = ",".join(str(x) for x in chunking["parent_split_levels"])
    return (
        f"levels={levels};max_parent={chunking['max_parent_chars']};"
        f"min_parent={chunking['min_parent_chars']};child={chunking['child_max_chars']};"
        f"overlap={chunking['child_overlap_chars']};force={chunking['force_heading_split']};"
        f"auto_descend={chunking['auto_descend']};mode={chunking.get('chunking_mode', 'auto')}"
    )


def report_warnings(
    *,
    text: str,
    parent_sections: list[dict[str, str]],
    child_count: int,
    chunking: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    parent_count = len(parent_sections)
    parent_lengths = [len(section["text"]) for section in parent_sections]
    max_parent = max(parent_lengths, default=0)
    min_parent = min(parent_lengths, default=0)
    if max_parent > chunking["max_parent_chars"]:
        warnings.append("parent_over_max_chars")
    if chunking["min_parent_chars"] and min_parent < chunking["min_parent_chars"]:
        warnings.append("parent_under_min_chars")
    if parent_count and child_count and abs(parent_count - child_count) <= max(1, parent_count * 0.1):
        warnings.append("parent_child_counts_too_close")
    if len(text) > chunking["max_parent_chars"] and parent_count <= 1:
        warnings.append("large_document_not_split")
    if chunking["force_heading_split"] and count_headings(text, chunking["parent_split_levels"]) == 0:
        warnings.append("configured_headings_not_found")
    return warnings


def report_row(
    *,
    corpus_id: str,
    rel: str,
    title: str,
    text: str,
    parent_sections: list[dict[str, str]],
    child_count: int,
    chunking: dict[str, Any],
    chunking_method: str = "heading",
    marker_parent_count: int = 0,
    marker_child_count: int = 0,
    marker_warnings: list[str] | None = None,
) -> dict[str, Any]:
    parent_lengths = [len(section["text"]) for section in parent_sections]
    child_per_parent = round(child_count / len(parent_sections), 2) if parent_sections else 0
    warnings = report_warnings(
        text=text,
        parent_sections=parent_sections,
        child_count=child_count,
        chunking=chunking,
    )
    marker_warnings = marker_warnings or []
    return {
        "corpus_id": corpus_id,
        "source_file": rel,
        "title": title,
        "source_chars": len(text),
        "parent_count": len(parent_sections),
        "child_count": child_count,
        "children_per_parent": child_per_parent,
        "parent_min_chars": min(parent_lengths, default=0),
        "parent_max_chars": max(parent_lengths, default=0),
        "parent_avg_chars": round(sum(parent_lengths) / len(parent_lengths), 1) if parent_lengths else 0,
        "configured_heading_count": count_headings(text, chunking["parent_split_levels"]),
        "chunking": chunking_label(chunking),
        "warnings": "|".join(warnings + marker_warnings),
        "chunking_method": chunking_method,
        "marker_parent_count": marker_parent_count,
        "marker_child_count": marker_child_count,
        "marker_warnings": "|".join(marker_warnings),
    }


def write_chunk_report(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "corpus_id",
        "source_file",
        "title",
        "source_chars",
        "parent_count",
        "child_count",
        "children_per_parent",
        "parent_min_chars",
        "parent_max_chars",
        "parent_avg_chars",
        "configured_heading_count",
        "chunking",
        "warnings",
        "chunking_method",
        "marker_parent_count",
        "marker_child_count",
        "marker_warnings",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build parent/child chunks from Markdown sources.")
    parser.add_argument(
        "--html-tags",
        choices=["strip", "preserve"],
        default="strip",
        help="strip removes HTML tags from saved chunks; preserve keeps tags for chunking experiments.",
    )
    return parser.parse_args()


def main(*, html_tags: str = "strip") -> None:
    strip_html_tags = html_tags != "preserve"
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    parents = []
    children = []
    report_rows = []

    for corpus in load_corpora():
        corpus_id = corpus["corpus_id"]
        md_dir = BASE_DIR / corpus["markdown_dir"]
        if not md_dir.exists():
            continue

        for md_path in sorted(md_dir.rglob("*.md")):
            raw = md_path.read_text(encoding="utf-8")
            metadata, text = parse_frontmatter(raw)
            rel = md_path.relative_to(BASE_DIR).as_posix()
            title = document_title(text, metadata, md_path.stem)
            inherited = base_row(metadata)
            chunking = build_chunking(corpus, metadata)
            custom_chunking = has_custom_chunking(corpus, metadata)
            marker_diagnostics = {
                "marker_parent_count": 0,
                "marker_child_count": 0,
                "marker_warnings": [],
            }
            chunking_method = "heading"

            markers = parse_chunk_markers(text)
            parent_marker_count = sum(1 for marker in markers if marker["kind"] == "parent_chunk")
            mode = chunking.get("chunking_mode", "auto")
            use_marker_chunking = mode == "html_comment" or (mode == "auto" and parent_marker_count > 0)

            if use_marker_chunking:
                parent_sections, marker_diagnostics = split_to_marker_parent_sections(
                    text,
                    title,
                    chunking,
                    strip_html_tags=strip_html_tags,
                )
                if parent_sections:
                    chunking_method = "html_comment"
                else:
                    marker_diagnostics["marker_warnings"].append("fallback_to_heading_chunking")
                    if custom_chunking:
                        parent_sections = split_to_parent_sections(text, title, chunking)
                    else:
                        parent_sections = split_to_parent_sections_legacy(text, title)
            elif custom_chunking:
                parent_sections = split_to_parent_sections(text, title, chunking)
            else:
                parent_sections = split_to_parent_sections_legacy(text, title)

            doc_child_count = 0
            for p_idx, section in enumerate(parent_sections, start=1):
                parent_marker_id = section.get("parent_marker_id", "")
                marker_source_value = section.get("marker_source", "")
                parent_id = stable_id(corpus_id, rel, "parent", str(p_idx), parent_marker_id, section["heading_path"])
                parent = {
                    "parent_id": parent_id,
                    "corpus_id": corpus_id,
                    "title": title,
                    "source_file": rel,
                    "heading_path": section["heading_path"],
                    "text": clean_rag_text(section["text"], strip_html_tags=strip_html_tags),
                    **inherited,
                }
                if chunking_method == "html_comment":
                    parent.update(
                        {
                            "chunking_method": section.get("chunking_method", chunking_method),
                            "parent_marker_id": parent_marker_id,
                            "marker_source": marker_source_value,
                        }
                    )
                parents.append(parent)

                child_chunks = (
                    split_marker_child_sections(section, chunking)
                    if chunking_method == "html_comment"
                    else split_child_chunks(section["text"], section["heading_path"], chunking)
                )
                doc_child_count += len(child_chunks)
                for c_idx, child in enumerate(child_chunks, start=1):
                    child_text = clean_rag_text(child["text"], strip_html_tags=strip_html_tags)
                    child_id = stable_id(parent_id, "child", str(c_idx), child.get("child_marker_id", ""), child_text[:80])
                    child_row = {
                        "child_id": child_id,
                        "parent_id": parent_id,
                        "corpus_id": corpus_id,
                        "title": title,
                        "source_file": rel,
                        "heading_path": child["heading_path"],
                        "text": child_text,
                        **inherited,
                    }
                    if chunking_method == "html_comment":
                        child_row.update(
                            {
                                "chunking_method": child.get("chunking_method", chunking_method),
                                "parent_marker_id": parent_marker_id,
                                "child_marker_id": child.get("child_marker_id", ""),
                                "marker_source": child.get("marker_source", marker_source_value),
                            }
                        )
                    children.append(child_row)

            report_rows.append(
                report_row(
                    corpus_id=corpus_id,
                    rel=rel,
                    title=title,
                    text=text,
                    parent_sections=parent_sections,
                    child_count=doc_child_count,
                    chunking=chunking,
                    chunking_method=chunking_method,
                    marker_parent_count=int(marker_diagnostics.get("marker_parent_count") or 0),
                    marker_child_count=int(marker_diagnostics.get("marker_child_count") or 0),
                    marker_warnings=list(marker_diagnostics.get("marker_warnings") or []),
                )
            )

    with PARENT_OUT.open("w", encoding="utf-8") as f:
        for row in parents:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with CHILD_OUT.open("w", encoding="utf-8") as f:
        for row in children:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_chunk_report(CHUNK_REPORT_OUT, report_rows)

    ratio = round(len(children) / len(parents), 2) if parents else 0
    warning_count = sum(1 for row in report_rows if row["warnings"])
    print(f"parents: {len(parents)} -> {PARENT_OUT}")
    print(f"children: {len(children)} -> {CHILD_OUT}")
    print(f"children_per_parent: {ratio}")
    print(f"chunk_report: {CHUNK_REPORT_OUT} warnings={warning_count}")


if __name__ == "__main__":
    args = parse_args()
    main(html_tags=args.html_tags)
