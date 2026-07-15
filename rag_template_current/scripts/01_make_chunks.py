# Markdown化済み文書から親チャンクと子チャンクを生成します。
from pathlib import Path
import json
import re
import hashlib
from typing import List, Dict, Any, Tuple


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "corpus_settings.json"
CHUNK_DIR = BASE_DIR / "chunks"
PARENT_OUT = CHUNK_DIR / "parent_chunks.jsonl"
CHILD_OUT = CHUNK_DIR / "child_chunks.jsonl"


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
    """
    親チャンク:
    - 原則として見出し単位
    - h1/h2/h3を親候補にする
    - 大きすぎる場合は文字数で補助分割
    """
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

    # 見出しがない文書
    if not sections and text.strip():
        sections.append({"heading_path": "", "text": text.strip()})

    # 長大親チャンクの補助分割
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
    """
    子チャンク:
    - 検索用
    - 見出し、条文、段落をなるべく維持
    - max_charsを超える場合だけ文字数で分割
    """
    blocks = re.split(r"(?=\n#{1,6}\s+|\n第[0-9０-９一二三四五六七八九十百]+条|\n\([0-9０-９]+\))", "\n" + parent_text)
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


def main():
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    parents = []
    children = []

    for corpus in load_corpora():
        corpus_id = corpus["corpus_id"]
        md_dir = BASE_DIR / corpus["markdown_dir"]
        if not md_dir.exists():
            continue

        for md_path in sorted(md_dir.rglob("*.md")):
            raw = md_path.read_text(encoding="utf-8")
            text = strip_yaml_frontmatter(raw)
            rel = md_path.relative_to(BASE_DIR).as_posix()
            title = md_path.stem

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

    with PARENT_OUT.open("w", encoding="utf-8") as f:
        for row in parents:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with CHILD_OUT.open("w", encoding="utf-8") as f:
        for row in children:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"parents: {len(parents)} -> {PARENT_OUT}")
    print(f"children: {len(children)} -> {CHILD_OUT}")


if __name__ == "__main__":
    main()
