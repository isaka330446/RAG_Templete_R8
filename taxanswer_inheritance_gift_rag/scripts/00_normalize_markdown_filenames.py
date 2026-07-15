from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
MARKDOWN_DIR = BASE_DIR / "data" / "markdown"
CHUNK_FILES = [
    BASE_DIR / "chunks" / "parent_chunks.jsonl",
    BASE_DIR / "chunks" / "child_chunks.jsonl",
    BASE_DIR / "chunks" / "child_chunks_with_tags.jsonl",
]
CHUNK_REPORT = BASE_DIR / "chunks" / "chunk_report.csv"


def parse_frontmatter(raw: str) -> tuple[dict[str, str], str, bool]:
    if not raw.startswith("---\n"):
        return {}, raw, False
    end = raw.find("\n---", 4)
    if end == -1:
        return {}, raw, False
    metadata: dict[str, str] = {}
    for line in raw[4:end].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] == '"':
            value = value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        metadata[key.strip()] = value
    return metadata, raw[end + 4 :].lstrip(), True


def yaml_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_frontmatter(metadata: dict[str, str], body: str) -> str:
    preferred = [
        "title",
        "source_url",
        "document_id",
        "document_type",
        "corpus_id",
        "tax_type",
        "source_site",
        "law_basis_date",
        "original_filename",
    ]
    keys = [key for key in preferred if key in metadata]
    keys.extend(key for key in metadata if key not in keys)
    lines = ["---"]
    lines.extend(f"{key}: {yaml_quote(str(metadata[key]))}" for key in keys)
    lines.append("---")
    return "\n".join(lines) + "\n\n" + body.lstrip()


def title_from_body(body: str, fallback: str) -> str:
    for line in body.splitlines():
        match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if match:
            return match.group(1).strip()
    return fallback


def safe_id(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = text.replace("/", "_").replace("\\", "_")
    text = re.sub(r"[^0-9a-zA-Z_\-.]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text


def prefix_for(path: Path, metadata: dict[str, str]) -> str:
    rel = path.relative_to(MARKDOWN_DIR).as_posix()
    corpus = rel.split("/", 1)[0]
    if corpus == "nta_sozoku_shitsugi":
        return "shitsugi"
    if corpus == "nta_sozoku_sochiho_tsutatsu":
        return "sochiho"
    if corpus == "nta_sozoku_hyoka_tsutatsu":
        return "hyoka"
    if corpus == "nta_sozoku_tsutatsu":
        return "tsutatsu"
    if corpus == "nta_sozoku_joho_zeikaishaku_pdf":
        return "joho"
    if corpus == "nta_taxanswer_asset_tax":
        return "taxanswer"
    return safe_id(corpus.replace("nta_", "")) or "doc"


def identifier_for(path: Path, metadata: dict[str, str]) -> str:
    document_id = safe_id(metadata.get("document_id", ""))
    if document_id:
        return document_id
    stem = path.stem
    leading = re.match(r"([0-9A-Za-z][0-9A-Za-z_\-.]*)(?:_|$)", stem)
    if leading:
        return safe_id(leading.group(1))
    digest = hashlib.sha1(path.relative_to(BASE_DIR).as_posix().encode("utf-8")).hexdigest()[:10]
    return digest


def planned_name(path: Path, metadata: dict[str, str], used: set[Path]) -> Path:
    prefix = prefix_for(path, metadata)
    ident = identifier_for(path, metadata)
    if ident == prefix or ident.startswith(f"{prefix}_"):
        base = f"{ident}.md"
    else:
        base = f"{prefix}_{ident}.md"
    target = path.with_name(base)
    counter = 2
    while target in used or (target.exists() and target != path):
        target = path.with_name(f"{prefix}_{ident}_{counter}.md")
        counter += 1
    used.add(target)
    return target


def iter_markdown() -> list[Path]:
    return sorted(MARKDOWN_DIR.rglob("*.md"))


def update_jsonl(path: Path, mapping: dict[str, str]) -> int:
    if not path.exists():
        return 0
    changed = 0
    lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        source_file = item.get("source_file")
        if source_file in mapping:
            item["source_file"] = mapping[source_file]
            changed += 1
        lines.append(json.dumps(item, ensure_ascii=False))
    if changed:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return changed


def update_chunk_report(path: Path, mapping: dict[str, str]) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []
    if "source_file" not in fieldnames:
        return 0
    changed = 0
    for row in rows:
        old = row.get("source_file")
        if old in mapping:
            row["source_file"] = mapping[old]
            changed += 1
    if changed:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return changed


def normalize_markdown_files(*, apply: bool) -> dict[str, Any]:
    used: set[Path] = set()
    plans: list[tuple[Path, Path, dict[str, str], str]] = []
    for path in iter_markdown():
        raw = path.read_text(encoding="utf-8")
        metadata, body, _has_frontmatter = parse_frontmatter(raw)
        metadata.setdefault("title", title_from_body(body, path.stem))
        metadata.setdefault("original_filename", path.name)
        target = planned_name(path, metadata, used)
        if target != path or "original_filename" not in parse_frontmatter(raw)[0]:
            plans.append((path, target, metadata, body))

    mapping: dict[str, str] = {}
    for source, target, metadata, body in plans:
        old_rel = source.relative_to(BASE_DIR).as_posix()
        new_rel = target.relative_to(BASE_DIR).as_posix()
        mapping[old_rel] = new_rel
        if not apply:
            continue
        source.write_text(render_frontmatter(metadata, body), encoding="utf-8")
        if target != source:
            source.replace(target)

    chunk_updates: dict[str, int] = {}
    if apply and mapping:
        for chunk_file in CHUNK_FILES:
            chunk_updates[chunk_file.relative_to(BASE_DIR).as_posix()] = update_jsonl(chunk_file, mapping)
        chunk_updates[CHUNK_REPORT.relative_to(BASE_DIR).as_posix()] = update_chunk_report(CHUNK_REPORT, mapping)

    max_name_len = 0
    max_path_len = 0
    for path in iter_markdown() if apply else [target for _, target, _, _ in plans] + [p for p in iter_markdown() if p not in {a for a, *_ in plans}]:
        max_name_len = max(max_name_len, len(path.name))
        max_path_len = max(max_path_len, len(str(path.resolve())))

    return {
        "apply": apply,
        "markdown_count": len(iter_markdown()),
        "planned_changes": len(plans),
        "mapping_count": len(mapping),
        "max_filename_length": max_name_len,
        "max_path_length": max_path_len,
        "chunk_updates": chunk_updates,
        "sample": [
            {
                "from": source.relative_to(BASE_DIR).as_posix(),
                "to": target.relative_to(BASE_DIR).as_posix(),
            }
            for source, target, *_ in plans[:10]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize Markdown filenames to short stable IDs.")
    parser.add_argument("--apply", action="store_true", help="Rename files and update source_file references.")
    args = parser.parse_args()
    print(json.dumps(normalize_markdown_files(apply=args.apply), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
