from __future__ import annotations

import html
import json
import os
import re
import unicodedata
import uuid
from pathlib import Path
from urllib.parse import urlencode

import httpx
from fasthtml.common import fast_app, serve
from markdown_it import MarkdownIt
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles


BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_SETTINGS = BASE_DIR / "config" / "corpus_settings.json"
DEFAULT_API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000/ask")
DEFAULT_TOP_K = 8
EVIDENCE_UI_MODE = os.getenv("RAG_EVIDENCE_UI_MODE", "side").strip().lower()
EVIDENCE_UI_MODE = "popup" if EVIDENCE_UI_MODE in {"popup", "modal"} else "side"

app, rt = fast_app()
app.routes.insert(0, Mount("/static", app=StaticFiles(directory=BASE_DIR / "static"), name="static"))
MARKDOWN = MarkdownIt("commonmark", {"html": False, "breaks": False}).enable(["table", "strikethrough"])
FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")
CIRCLED_NUMBER_MAP = {
    char: str(index)
    for index, char in enumerate("①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳", start=1)
}
CIRCLED_NUMBER_PATTERN = "[" + "".join(CIRCLED_NUMBER_MAP) + "]"
EVIDENCE_PREFIX_PATTERN = r"(?:根拠(?:文書|資料)?|出典|参考|参照|引用|資料|ソース|source|Source|SOURCE)"
EVIDENCE_NUMBER_PATTERN = rf"(?:[0-9０-９]+|{CIRCLED_NUMBER_PATTERN})"
EVIDENCE_NUMBER_LIST_PATTERN = rf"{EVIDENCE_NUMBER_PATTERN}(?:\s*(?:[、,，/／・･]|や|と|及び|および|and)\s*{EVIDENCE_NUMBER_PATTERN})*"
EVIDENCE_TITLE_PATTERN = rf"(?P<prefix>{EVIDENCE_PREFIX_PATTERN})(?P<sep>\s*[：:]\s*)(?P<title>[^\r\n]+)"


def esc(value: object) -> str:
    return html.escape(str(value or ""), quote=True)


def text_block(value: object) -> str:
    return esc(value).replace("\n", "<br>")


def markdown_block(value: object) -> str:
    return MARKDOWN.render(str(value or ""))


def source_file_name(source: dict) -> str:
    value = str(source.get("source_file") or source.get("source_url") or "").strip()
    if not value:
        return ""
    if value.startswith(("http://", "https://")):
        return value
    return value.replace("\\", "/").split("/")[-1]


def source_url(session_id: str, message_index: int, source_index: int) -> str:
    return "/source?" + urlencode(
        {"session_id": session_id, "message_index": message_index, "source_index": source_index}
    )


def evidence_token(label: str, session_id: str, message_index: int, source_index: int) -> str:
    url = source_url(session_id, message_index, source_index)
    return (
        f'<a href="{esc(url)}" class="evidence-ref" data-evidence-url="{esc(url)}">'
        f"{esc(label)}</a>"
    )


def normalize_evidence_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"[\s\-_/#:：,，、。．.・･「」『』【】\[\]()（）<>＜＞\"'`*_]+", "", text)


def source_match_terms(source: dict) -> list[str]:
    terms = [
        source_title(source),
        source.get("heading_path"),
        source.get("title"),
        source_file_name(source),
        source.get("source_url"),
    ]
    file_name = source_file_name(source)
    if file_name:
        terms.append(Path(file_name).stem)
    normalized = []
    seen = set()
    for term in terms:
        norm = normalize_evidence_text(term)
        if norm and norm not in seen:
            normalized.append(norm)
            seen.add(norm)
    return normalized


def title_is_specific(norm: str) -> bool:
    return len(norm) >= 8 or bool(re.search(r"\d{3,}", norm))


def find_source_by_title(title: str, sources: list[dict]) -> int | None:
    target = normalize_evidence_text(title)
    if len(target) < 4:
        return None

    matches: list[tuple[int, int]] = []
    for idx, source in enumerate(sources):
        best_score = 0
        for term in source_match_terms(source):
            if len(term) < 4:
                continue
            if target == term:
                best_score = max(best_score, 3)
            elif title_is_specific(target) and target in term:
                best_score = max(best_score, 2)
            elif title_is_specific(term) and term in target:
                best_score = max(best_score, 1)
        if best_score:
            matches.append((best_score, idx))

    if not matches:
        return None
    best_score = max(score for score, _ in matches)
    winners = [idx for score, idx in matches if score == best_score]
    return winners[0] if len(winners) == 1 else None


def answer_markdown(content: object, session_id: str, message_index: int, sources: list[dict]) -> str:
    text = str(content or "")
    replacements: dict[str, str] = {}

    def token_for(label: str, source_number: int) -> str:
        token = f"EVIDENCEREFTOKEN{len(replacements)}END"
        replacements[token] = evidence_token(label, session_id, message_index, source_number - 1)
        return token

    def number_from(raw: str) -> int | None:
        if raw in CIRCLED_NUMBER_MAP:
            return int(CIRCLED_NUMBER_MAP[raw])
        try:
            return int(raw.translate(FULLWIDTH_DIGITS))
        except ValueError:
            return None

    def replace_prefixed_group(match: re.Match[str]) -> str:
        prefix = match.group("prefix")
        nums = match.group("nums")
        out = []
        last = 0
        first = True
        for num_match in re.finditer(EVIDENCE_NUMBER_PATTERN, nums):
            out.append(nums[last:num_match.start()])
            raw = num_match.group(0)
            source_number = number_from(raw)
            label = (prefix if first else "") + raw
            out.append(token_for(label, source_number) if source_number is not None and 1 <= source_number <= len(sources) else label)
            first = False
            last = num_match.end()
        out.append(nums[last:])
        return "".join(out)

    def replace_bracket_number(match: re.Match[str]) -> str:
        raw = match.group("num")
        source_number = number_from(raw)
        if source_number is None or not (1 <= source_number <= len(sources)):
            return match.group(0)
        return token_for(f"根拠{raw}", source_number)

    def replace_title_reference(match: re.Match[str]) -> str:
        title = match.group("title").strip()
        source_index = find_source_by_title(title, sources)
        if source_index is None:
            return match.group(0)
        return f"{match.group('prefix')}{match.group('sep')}{token_for(title, source_index + 1)}"

    linked = re.sub(
        rf"(?:[［\[\(（【]\s*)?(?P<prefix>{EVIDENCE_PREFIX_PATTERN})\s*(?:(?:[:：#]|No\.?|№)\s*)?(?P<nums>{EVIDENCE_NUMBER_LIST_PATTERN})(?:\s*[］\]\)）】])?",
        replace_prefixed_group,
        text,
    )
    linked = re.sub(rf"[［\[\(（【](?P<num>{EVIDENCE_NUMBER_PATTERN})[］\]\)）】]", replace_bracket_number, linked)
    linked = re.sub(EVIDENCE_TITLE_PATTERN, replace_title_reference, linked)
    rendered = markdown_block(linked)
    for token, html_value in replacements.items():
        rendered = rendered.replace(token, html_value)
    return rendered


def evidence_link_row(session_id: str, message_index: int, sources: list[dict]) -> str:
    if not sources:
        return ""
    links = "".join(evidence_token(f"根拠{idx + 1}", session_id, message_index, idx) for idx, _ in enumerate(sources))
    return f'<div class="answer-evidence-links"><span>根拠リンク</span>{links}</div>'


def short_text(value: object, limit: int = 78) -> str:
    text = str(value or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def api_base_url(ask_url: str) -> str:
    base = str(ask_url or DEFAULT_API_URL).rstrip("/")
    return base[: -len("/ask")] if base.endswith("/ask") else base


def load_corpora() -> list[dict]:
    if not CORPUS_SETTINGS.exists():
        return []
    data = json.loads(CORPUS_SETTINGS.read_text(encoding="utf-8"))
    corpora = [c for c in data.get("corpora", []) if c.get("enabled", True)]
    return sorted(corpora, key=lambda item: item.get("priority", 999))


def source_title(source: dict) -> str:
    return str(
        source.get("heading_path")
        or source.get("title")
        or source_file_name(source)
        or "根拠"
    )


def score_text(source: dict) -> str:
    score = source.get("score")
    return f"{score:.3f}" if isinstance(score, (int, float)) else "-"


def score_value(source: dict) -> float | None:
    score = source.get("score")
    return float(score) if isinstance(score, (int, float)) else None


def match_label(source: dict) -> tuple[str, str]:
    score = score_value(source)
    if score is None:
        return "補足", "support"
    if score >= 0.75:
        return "高一致", "high"
    if score >= 0.55:
        return "中一致", "medium"
    if score >= 0.35:
        return "補足", "support"
    return "要確認", "review"


def has_low_confidence(sources: list[dict]) -> bool:
    scores = [score_value(source) for source in sources]
    scores = [score for score in scores if score is not None]
    return bool(scores) and max(scores) < 0.35


def ms_icon(name: str, extra_class: str = "") -> str:
    class_name = f"ms-icon {extra_class}".strip()
    return f'<span class="{esc(class_name)}" aria-hidden="true">{esc(name)}</span>'


def source_summary_label(source: dict, limit: int = 42) -> str:
    title = source_title(source)
    slide = source.get("slide_no")
    section = source.get("section_title") or source.get("agenda") or source.get("topic")
    suffix = f" / Slide {slide}" if slide not in (None, "") else (f" / {section}" if section else "")
    return short_text(f"{title}{suffix}", limit)


class ConversationStore:
    def __init__(self) -> None:
        self.sessions: dict[str, list[dict]] = {}

    def ensure(self, session_id: str) -> list[dict]:
        return self.sessions.setdefault(session_id, [])

    def history_for_api(self, session_id: str) -> list[dict]:
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in self.ensure(session_id)
            if msg.get("role") in {"user", "assistant"} and msg.get("content")
        ]

    def source(self, session_id: str, message_index: int, source_index: int) -> dict | None:
        messages = self.ensure(session_id)
        if message_index < 0 or message_index >= len(messages):
            return None
        sources = messages[message_index].get("sources") or []
        if source_index < 0 or source_index >= len(sources):
            return None
        return sources[source_index]


store = ConversationStore()


MODERN_CSS = """
@font-face {
  font-family: "Noto Sans JP Local";
  src: url("/static/fonts/NotoSansJP-wght.ttf") format("truetype");
  font-weight: 100 900;
  font-style: normal;
  font-display: swap;
}
:root {
  color-scheme: light;
  --bg: #fbfbfd;
  --ink: #14161a;
  --muted: #667085;
  --panel: #ffffff;
  --soft: #f3f5f8;
  --line: #e4e7ec;
  --primary: #2563eb;
  --primary-dark: #1d4ed8;
  --violet: #7c3aed;
  --danger: #b42318;
  --shadow: 0 24px 80px rgba(15, 23, 42, 0.12);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  min-height: 100vh;
  background:
    radial-gradient(circle at 20% 0%, rgba(37,99,235,.08), transparent 28%),
    radial-gradient(circle at 88% 12%, rgba(124,58,237,.08), transparent 28%),
    var(--bg);
  color: var(--ink);
  font-family: "Noto Sans JP Local", "Noto Sans JP", "Noto Sans CJK JP", "Yu Gothic UI", "Meiryo", sans-serif;
}
button, input, textarea { font: inherit; }
a { color: inherit; text-decoration: none; }
.shell { min-height: 100vh; display: flex; flex-direction: column; }
.nav {
  height: 66px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 20px;
  padding: 0 24px;
}
.logo { display: flex; align-items: center; gap: 12px; font-weight: 800; }
.logo-mark {
  width: 34px;
  height: 34px;
  border-radius: 10px;
  display: grid;
  place-items: center;
  background: #111827;
  color: #fff;
}
.nav-sub { color: var(--muted); font-size: 13px; }
.nav-actions { display: flex; align-items: center; gap: 10px; }
.pill-link, .icon-btn, .send-btn, .source-chip, .mobile-menu-btn, .pin-btn, .settings-rail {
  border: 1px solid var(--line);
  background: rgba(255,255,255,.88);
  border-radius: 999px;
  padding: 9px 13px;
  cursor: pointer;
}
.mobile-menu-btn { display: none; }
.send-btn {
  background: var(--primary);
  border-color: var(--primary);
  color: #fff;
  font-weight: 800;
}
.send-btn:hover { background: var(--primary-dark); }
.workspace {
  width: min(1880px, calc(100vw - 32px));
  margin: 4px auto 28px;
  display: grid;
  grid-template-columns: 56px minmax(680px, 1fr) minmax(380px, 28vw);
  gap: 18px;
  align-items: stretch;
  transition: grid-template-columns .22s ease;
}
.workspace.sidebar-expanded,
.workspace.sidebar-pinned,
.workspace:has(.settings:hover),
.workspace:has(.settings:focus-within) {
  grid-template-columns: minmax(250px, 300px) minmax(680px, 1fr) minmax(380px, 28vw);
}
.workspace.evidence-popup-mode {
  grid-template-columns: 56px minmax(680px, 1fr);
}
.workspace.evidence-popup-mode.sidebar-expanded,
.workspace.evidence-popup-mode.sidebar-pinned,
.workspace.evidence-popup-mode:has(.settings:hover),
.workspace.evidence-popup-mode:has(.settings:focus-within) {
  grid-template-columns: minmax(250px, 300px) minmax(680px, 1fr);
}
.settings {
  position: sticky;
  top: 16px;
  min-height: calc(100vh - 98px);
  overflow: hidden;
  padding: 0;
  background: rgba(255,255,255,.86);
  border: 1px solid var(--line);
  border-radius: 22px;
  box-shadow: var(--shadow);
  transition: padding .18s ease, box-shadow .18s ease;
}
.workspace.sidebar-expanded .settings,
.workspace.sidebar-pinned .settings,
.workspace:has(.settings:hover) .settings,
.workspace:has(.settings:focus-within) .settings {
  padding: 18px;
}
.settings-rail {
  width: 100%;
  height: 100%;
  min-height: calc(100vh - 98px);
  display: grid;
  place-items: center;
  border: 0;
  border-radius: 22px;
  color: var(--primary-dark);
  font-weight: 900;
}
.rail-text {
  writing-mode: vertical-rl;
  letter-spacing: 0;
}
.settings-body {
  opacity: 0;
  visibility: hidden;
  transform: translateX(-10px);
  pointer-events: none;
  max-height: 0;
  overflow: hidden;
  transition: opacity .18s ease, transform .18s ease, visibility .18s ease;
}
.workspace.sidebar-expanded .settings-rail,
.workspace.sidebar-pinned .settings-rail,
.workspace:has(.settings:hover) .settings-rail,
.workspace:has(.settings:focus-within) .settings-rail {
  display: none;
}
.workspace.sidebar-expanded .settings-body,
.workspace.sidebar-pinned .settings-body,
.workspace:has(.settings:hover) .settings-body,
.workspace:has(.settings:focus-within) .settings-body {
  opacity: 1;
  visibility: visible;
  transform: translateX(0);
  pointer-events: auto;
  max-height: none;
  overflow: visible;
}
.settings-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 14px;
}
.settings-head h2 { margin: 0; font-size: 16px; }
.pin-btn {
  padding: 7px 10px;
  font-size: 12px;
  font-weight: 800;
}
.pin-btn[aria-pressed="true"] {
  background: var(--primary);
  border-color: var(--primary);
  color: #fff;
}
.sidebar-backdrop { display: none; }
.sidebar-field { display: grid; gap: 7px; margin-bottom: 14px; }
.sidebar-field label, .sidebar-label { color: var(--muted); font-size: 12px; font-weight: 800; }
.sidebar-range {
  display: grid;
  grid-template-columns: 1fr 36px;
  gap: 8px;
  align-items: center;
}
.sidebar-range output { color: var(--primary); font-weight: 900; }
.sidebar-corpus-list { display: grid; gap: 8px; margin: 8px 0 14px; }
.sidebar-corpus {
  display: grid;
  grid-template-columns: 22px 1fr;
  gap: 8px;
  padding: 9px;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: #fff;
}
.sidebar-corpus span { font-size: 13px; line-height: 1.45; }
.chat-panel {
  min-height: calc(100vh - 98px);
  background: rgba(255,255,255,.78);
  border: 1px solid rgba(228,231,236,.9);
  border-radius: 22px;
  box-shadow: var(--shadow);
  backdrop-filter: blur(14px);
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
.toolbar {
  padding: 16px 18px;
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 14px;
  border-bottom: 1px solid var(--line);
  background: rgba(255,255,255,.7);
}
.settings-row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.range-wrap {
  display: inline-grid;
  grid-template-columns: 94px 82px 32px;
  gap: 8px;
  align-items: center;
  padding: 8px 12px;
  border: 1px solid var(--line);
  border-radius: 999px;
  background: #fff;
  color: var(--muted);
  font-size: 13px;
}
.range-wrap output { color: var(--primary); font-weight: 800; }
.corpus-menu { position: relative; }
.corpus-menu summary {
  list-style: none;
  cursor: pointer;
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 10px 13px;
  background: #fff;
  color: var(--muted);
}
.corpus-menu summary::-webkit-details-marker { display: none; }
.corpus-pop {
  position: absolute;
  z-index: 20;
  top: 46px;
  left: 0;
  width: 360px;
  max-height: 360px;
  overflow: auto;
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 16px;
  box-shadow: var(--shadow);
  padding: 10px;
}
.corpus-check {
  display: grid;
  grid-template-columns: 22px 1fr;
  gap: 8px;
  padding: 9px;
  border-radius: 12px;
}
.corpus-check:hover { background: var(--soft); }
.corpus-check span { font-size: 13px; line-height: 1.45; }
.session { color: var(--muted); font-size: 12px; word-break: break-all; }
.messages {
  flex: 1;
  display: grid;
  align-content: start;
  gap: 20px;
  padding: 28px;
  overflow: auto;
  max-height: calc(100vh - 280px);
}
.empty {
  min-height: 360px;
  display: grid;
  place-items: center;
  text-align: center;
  color: var(--muted);
}
.empty strong {
  display: block;
  color: var(--ink);
  font-size: clamp(24px, 4vw, 42px);
  line-height: 1.2;
  margin-bottom: 10px;
}
.message { display: grid; gap: 10px; }
.message.user { justify-items: end; }
.bubble {
  max-width: min(1040px, 94%);
  border-radius: 18px;
  padding: 15px 18px;
  line-height: 1.8;
}
.message.user .bubble { max-width: min(720px, 78%); }
.message.user .bubble {
  background: #111827;
  color: #fff;
  border-bottom-right-radius: 6px;
}
.message.assistant .bubble {
  width: 100%;
  max-width: none;
  background: #fff;
  border: 1px solid var(--line);
  border-bottom-left-radius: 6px;
}
.markdown-body {
  max-width: none;
  overflow-x: auto;
}
.markdown-body > :first-child { margin-top: 0; }
.markdown-body > :last-child { margin-bottom: 0; }
.markdown-body h1, .markdown-body h2, .markdown-body h3 {
  line-height: 1.45;
  margin: 1.2em 0 .55em;
}
.markdown-body h1 { font-size: 1.42rem; }
.markdown-body h2 { font-size: 1.22rem; }
.markdown-body h3 { font-size: 1.06rem; }
.markdown-body p { margin: .55em 0; }
.markdown-body ul, .markdown-body ol { padding-left: 1.35em; margin: .55em 0; }
.markdown-body li { margin: .24em 0; }
.markdown-body table { width: 100%; border-collapse: collapse; margin: .8em 0; font-size: .94em; }
.markdown-body th, .markdown-body td { border: 1px solid var(--line); padding: 7px 8px; vertical-align: top; }
.markdown-body th { background: var(--soft); }
.markdown-body code {
  background: #eef2ff;
  border-radius: 5px;
  padding: 1px 5px;
}
.evidence-ref {
  display: inline-flex;
  align-items: center;
  min-height: 24px;
  padding: 1px 8px;
  border-radius: 999px;
  background: #eef2ff;
  color: var(--primary-dark);
  font-size: .9em;
  font-weight: 900;
  text-decoration: none;
  border: 1px solid rgba(37,99,235,.18);
}
.evidence-ref:hover { background: #dbeafe; }
.answer-evidence-links {
  margin-top: 16px;
  padding-top: 12px;
  border-top: 1px solid var(--line);
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}
.answer-evidence-links span {
  color: var(--muted);
  font-size: 12px;
  font-weight: 900;
}
.assistant-meta { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.badge {
  min-height: 24px;
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  padding: 2px 9px;
  background: #eef2ff;
  color: #3730a3;
  font-size: 12px;
  font-weight: 800;
}
.badge.cache { background: #ecfdf3; color: #067647; }
.source-strip { display: flex; flex-wrap: wrap; gap: 8px; }
.source-chip {
  padding: 7px 11px;
  font-size: 12px;
  color: var(--primary-dark);
}
.composer {
  padding: 16px;
  border-top: 1px solid var(--line);
  background: rgba(255,255,255,.84);
}
.composer-box {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 10px;
  align-items: end;
  max-width: min(1120px, 100%);
  margin: 0 auto;
}
.composer textarea {
  min-height: 86px;
  resize: vertical;
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 14px 16px;
  background: #fff;
}
.notice {
  margin: 14px 18px 0;
  padding: 10px 13px;
  border: 1px solid #fed7aa;
  background: #fff7ed;
  color: #9a3412;
  border-radius: 12px;
}
.drawer {
  background: rgba(255,255,255,.86);
  border: 1px solid var(--line);
  border-radius: 22px;
  box-shadow: var(--shadow);
  padding: 18px;
  max-height: calc(100vh - 98px);
  overflow: auto;
  position: sticky;
  top: 16px;
}
.evidence-modal-root { display: contents; }
.evidence-modal {
  position: fixed;
  inset: 0;
  z-index: 90;
  display: grid;
  place-items: center;
  padding: 24px;
}
.evidence-modal-backdrop {
  position: absolute;
  inset: 0;
  background: rgba(15, 23, 42, .38);
}
.evidence-modal-card {
  position: relative;
  width: min(960px, calc(100vw - 36px));
  max-height: calc(100vh - 48px);
  overflow: auto;
  padding: 20px;
  background: rgba(255,255,255,.98);
  border: 1px solid var(--line);
  border-radius: 22px;
  box-shadow: 0 24px 80px rgba(15, 23, 42, .24);
}
.modal-close {
  position: absolute;
  top: 14px;
  right: 14px;
  border: 1px solid var(--line);
  background: #fff;
  color: var(--ink);
  border-radius: 999px;
  width: 36px;
  height: 36px;
  cursor: pointer;
  font-weight: 900;
}
body.modal-open { overflow: hidden; }
.drawer h2 { margin: 0 0 10px; font-size: 16px; }
.drawer-empty {
  min-height: 300px;
  display: grid;
  place-items: center;
  text-align: center;
  color: var(--muted);
  border: 1px dashed var(--line);
  border-radius: 16px;
  background: rgba(243,245,248,.72);
}
.evidence-title { font-size: 17px; font-weight: 900; margin: 0 0 12px; }
.kv {
  display: grid;
  gap: 3px;
  padding: 8px 0;
  border-bottom: 1px solid #edf1f5;
}
.kv span { color: var(--muted); font-size: 12px; }
.kv strong { font-size: 13px; }
.chunk {
  margin-top: 14px;
  padding: 13px;
  border: 1px solid var(--line);
  border-radius: 16px;
  background: #fff;
  line-height: 1.75;
}
.chunk h3 { margin: 0 0 8px; color: var(--muted); font-size: 13px; }
details { margin-top: 12px; }
summary { cursor: pointer; color: var(--primary-dark); font-weight: 800; }
.report-box {
  max-width: 760px;
  padding: 10px;
  border: 1px solid var(--line);
  border-radius: 16px;
  background: rgba(255,255,255,.7);
}
.report-box textarea { width: 100%; min-height: 64px; border: 1px solid var(--line); border-radius: 12px; padding: 10px; }
.report-actions { margin-top: 8px; display: flex; justify-content: space-between; gap: 10px; align-items: center; }
.report-status { color: #067647; font-size: 13px; font-weight: 800; }
.error { color: var(--danger); }
.pending-answer .bubble { border-style: dashed; color: var(--muted); }
.loading-row { display: inline-flex; align-items: center; gap: 8px; font-weight: 800; }
.dots { display: inline-flex; gap: 4px; }
.dots i {
  width: 6px;
  height: 6px;
  border-radius: 999px;
  background: var(--primary);
  animation: dotPulse 1.1s infinite ease-in-out;
}
.dots i:nth-child(2) { animation-delay: .16s; }
.dots i:nth-child(3) { animation-delay: .32s; }
@keyframes dotPulse {
  0%, 80%, 100% { transform: translateY(0); opacity: .35; }
  40% { transform: translateY(-3px); opacity: 1; }
}
.busy .send-btn, .composer.is-submitting .send-btn { opacity: .62; pointer-events: none; }
@media (max-width: 1060px) {
  .workspace { grid-template-columns: 56px minmax(0, 1fr); }
  .workspace.sidebar-expanded,
  .workspace.sidebar-pinned,
  .workspace:has(.settings:hover),
  .workspace:has(.settings:focus-within) {
    grid-template-columns: 280px minmax(0, 1fr);
  }
  .workspace.evidence-popup-mode,
  .workspace.evidence-popup-mode.sidebar-expanded,
  .workspace.evidence-popup-mode.sidebar-pinned,
  .workspace.evidence-popup-mode:has(.settings:hover),
  .workspace.evidence-popup-mode:has(.settings:focus-within) {
    grid-template-columns: 56px minmax(0, 1fr);
  }
  .drawer { position: static; max-height: none; }
}
@media (max-width: 720px) {
  .nav { height: auto; padding: 14px; align-items: flex-start; flex-direction: column; }
  .nav-actions { width: 100%; justify-content: space-between; }
  .mobile-menu-btn { display: inline-flex; align-items: center; gap: 6px; }
  .workspace { width: calc(100vw - 18px); }
  .workspace,
  .workspace.sidebar-expanded,
  .workspace.sidebar-pinned,
  .workspace:has(.settings:hover),
  .workspace:has(.settings:focus-within) {
    grid-template-columns: 1fr;
  }
  .settings {
    position: fixed;
    z-index: 60;
    top: 0;
    left: 0;
    width: min(340px, calc(100vw - 42px));
    height: 100vh;
    max-height: none;
    border-radius: 0 18px 18px 0;
    transform: translateX(-104%);
    transition: transform .22s ease;
    padding: 18px;
  }
  .settings-rail { display: none; }
  .settings-body {
    opacity: 1;
    visibility: visible;
    transform: none;
    pointer-events: auto;
    max-height: none;
    overflow: auto;
    height: 100%;
  }
  .workspace.sidebar-mobile-open .settings { transform: translateX(0); }
  .sidebar-backdrop {
    position: fixed;
    inset: 0;
    z-index: 55;
    background: rgba(15, 23, 42, .36);
  }
  .workspace.sidebar-mobile-open .sidebar-backdrop { display: block; }
  .evidence-modal { padding: 12px; align-items: end; }
  .evidence-modal-card {
    width: 100%;
    max-height: min(82vh, 720px);
    border-radius: 18px 18px 0 0;
  }
  .toolbar { grid-template-columns: 1fr; }
  .composer-box { grid-template-columns: 1fr; }
  .messages { padding: 18px; max-height: none; }
  .corpus-pop { width: min(360px, calc(100vw - 36px)); }
}
"""

UX_REFRESH_CSS = """
@font-face {
  font-family: "Material Symbols Rounded";
  font-style: normal;
  font-weight: 400;
  src: url("/static/fonts/material-symbols-rounded.woff2") format("woff2");
  font-display: block;
}
:root {
  --bg: #f7f8fb;
  --surface: rgba(255, 255, 255, 0.84);
  --surface-strong: #ffffff;
  --text: #172033;
  --muted: #667085;
  --line: rgba(23, 32, 51, 0.10);
  --primary: #305cff;
  --primary-soft: rgba(48, 92, 255, 0.10);
  --accent: #00b8a9;
  --accent-soft: rgba(0, 184, 169, 0.10);
  --warning: #f59e0b;
  --danger: #ef4444;
  --shadow: 0 18px 60px rgba(15, 23, 42, 0.10);
  --radius-lg: 24px;
  --radius-md: 16px;
  --radius-sm: 12px;
}
body {
  background:
    linear-gradient(135deg, rgba(48, 92, 255, .12), transparent 30%),
    linear-gradient(315deg, rgba(0, 184, 169, .10), transparent 28%),
    var(--bg);
  color: var(--text);
}
.ms-icon {
  font-family: "Material Symbols Rounded";
  font-weight: normal;
  font-style: normal;
  font-size: 21px;
  line-height: 1;
  letter-spacing: 0;
  text-transform: none;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  white-space: nowrap;
  direction: ltr;
  -webkit-font-feature-settings: "liga";
  -webkit-font-smoothing: antialiased;
  font-feature-settings: "liga";
  font-variation-settings: "FILL" 0, "wght" 420, "GRAD" 0, "opsz" 24;
}
.ms-icon.filled { font-variation-settings: "FILL" 1, "wght" 520, "GRAD" 0, "opsz" 24; }
.ms-icon.small { font-size: 18px; }
.ms-icon.large { font-size: 28px; }
button:focus-visible, a:focus-visible, input:focus-visible, textarea:focus-visible, summary:focus-visible {
  outline: 3px solid rgba(48, 92, 255, .28);
  outline-offset: 3px;
}
.nav {
  height: 72px;
  padding: 0 clamp(18px, 3vw, 38px);
}
.logo-mark {
  background: linear-gradient(135deg, var(--primary), var(--accent));
  border-radius: 14px;
}
.nav-sub { color: var(--muted); }
.workspace {
  width: min(1760px, calc(100vw - 32px));
  grid-template-columns: 56px minmax(680px, 1fr) minmax(360px, 26vw);
  gap: 20px;
}
.workspace.sidebar-expanded,
.workspace.sidebar-pinned,
.workspace:has(.settings:hover),
.workspace:has(.settings:focus-within) {
  grid-template-columns: minmax(220px, 240px) minmax(680px, 1fr) minmax(360px, 26vw);
}
.workspace.evidence-popup-mode { grid-template-columns: 56px minmax(680px, 1fr); }
.workspace.evidence-popup-mode.sidebar-expanded,
.workspace.evidence-popup-mode.sidebar-pinned,
.workspace.evidence-popup-mode:has(.settings:hover),
.workspace.evidence-popup-mode:has(.settings:focus-within) {
  grid-template-columns: minmax(220px, 240px) minmax(680px, 1fr);
}
.settings {
  border-radius: var(--radius-lg);
  background: rgba(255,255,255,.72);
  backdrop-filter: blur(18px);
}
.settings-rail {
  border-radius: var(--radius-lg);
  color: var(--primary);
  background: rgba(255,255,255,.66);
}
.rail-text {
  writing-mode: vertical-rl;
  display: inline-flex;
  gap: 8px;
  align-items: center;
  letter-spacing: 0;
}
.dock-menu { display: grid; gap: 8px; margin: 10px 0 18px; }
.dock-link {
  width: 100%;
  min-height: 44px;
  border: 1px solid transparent;
  border-radius: 15px;
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 0 11px;
  color: var(--text);
  background: transparent;
  cursor: pointer;
  font-weight: 800;
}
.dock-link:hover, .dock-link:focus-visible {
  background: var(--primary-soft);
  border-color: rgba(48,92,255,.12);
  color: var(--primary);
}
.dock-link .dock-label {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.settings-head h2 { font-size: 15px; }
.pin-btn, .pill-link, .mobile-menu-btn, .send-btn, .source-chip, .answer-action, .template-card {
  min-height: 42px;
}
.sidebar-corpus {
  border-radius: 14px;
  background: rgba(255,255,255,.72);
}
.chat-panel {
  min-height: calc(100vh - 104px);
  border-radius: 28px;
  border: 1px solid rgba(255,255,255,.78);
  background: rgba(255,255,255,.70);
  box-shadow: var(--shadow);
}
.workspace-head {
  padding: 22px 24px 16px;
  border-bottom: 1px solid var(--line);
  background: rgba(255,255,255,.58);
}
.workspace-head-main {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 18px;
}
.workspace-kicker {
  color: var(--primary);
  font-size: 12px;
  font-weight: 900;
  letter-spacing: .04em;
}
.workspace-title { margin: 4px 0 4px; font-size: clamp(22px, 3vw, 34px); line-height: 1.25; }
.workspace-copy { margin: 0; color: var(--muted); line-height: 1.7; }
.dataset-badges { display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }
.dataset-badge {
  min-height: 30px;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 999px;
  color: var(--primary);
  background: var(--primary-soft);
  border: 1px solid rgba(48,92,255,.13);
  font-size: 12px;
  font-weight: 900;
}
.messages {
  gap: 22px;
  padding: 26px clamp(18px, 3vw, 42px);
  max-height: calc(100vh - 326px);
}
.empty {
  min-height: 430px;
  display: block;
  text-align: left;
  color: var(--text);
}
.empty-hero {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(300px, 440px);
  gap: 28px;
  align-items: center;
}
.empty-eyebrow { color: var(--primary); font-size: 13px; font-weight: 900; }
.empty h1 { margin: 8px 0 12px; font-size: clamp(28px, 4vw, 46px); line-height: 1.18; letter-spacing: 0; }
.empty p { margin: 0; color: var(--muted); line-height: 1.8; }
.template-grid { display: grid; gap: 10px; }
.template-card {
  border: 1px solid var(--line);
  border-radius: 18px;
  background: rgba(255,255,255,.82);
  box-shadow: 0 10px 30px rgba(15,23,42,.06);
  padding: 14px 15px;
  text-align: left;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  cursor: pointer;
  color: var(--text);
}
.template-card:hover { transform: translateY(-1px); border-color: rgba(48,92,255,.22); color: var(--primary); }
.message.user .bubble {
  border-radius: 20px 20px 6px 20px;
  background: #172033;
}
.message.assistant { justify-items: stretch; }
.message.assistant .bubble { border: 0; background: transparent; padding: 0; }
.answer-card {
  width: 100%;
  border: 1px solid rgba(255,255,255,.78);
  border-radius: 26px;
  background: rgba(255,255,255,.92);
  box-shadow: 0 16px 48px rgba(15,23,42,.08);
  padding: clamp(18px, 2.4vw, 28px);
}
.answer-card.no-hit { border-color: rgba(245,158,11,.32); background: rgba(255,251,235,.86); }
.answer-card.low-confidence { border-color: rgba(245,158,11,.24); }
.answer-card-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
  margin-bottom: 14px;
}
.answer-state {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  color: var(--accent);
  font-size: 12px;
  font-weight: 900;
}
.answer-note {
  color: var(--muted);
  font-size: 12px;
  line-height: 1.6;
  max-width: 520px;
}
.answer-body { line-height: 1.9; }
.source-strip {
  margin-top: 18px;
  padding-top: 14px;
  border-top: 1px solid var(--line);
  display: flex;
  flex-wrap: wrap;
  gap: 9px;
  align-items: center;
}
.source-strip::before {
  content: "根拠";
  color: var(--muted);
  font-size: 12px;
  font-weight: 900;
  margin-right: 2px;
}
.source-chip {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  max-width: min(360px, 100%);
  border-color: rgba(48,92,255,.18);
  background: var(--primary-soft);
  color: var(--primary);
  font-size: 12px;
  font-weight: 900;
}
.source-chip .chip-text {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.match-pill {
  border-radius: 999px;
  padding: 1px 7px;
  background: rgba(255,255,255,.72);
  color: var(--muted);
  font-size: 11px;
}
.match-high { color: #067647; }
.match-medium { color: #305cff; }
.match-support { color: #667085; }
.match-review { color: #b42318; }
.answer-actions {
  margin-top: 16px;
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
}
.answer-action {
  border: 1px solid var(--line);
  border-radius: 999px;
  background: #fff;
  color: var(--text);
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding: 9px 13px;
  font-weight: 900;
  cursor: pointer;
}
.answer-action.primary { background: var(--primary); color: #fff; border-color: var(--primary); }
.answer-action.report { color: var(--danger); }
.no-hit-panel {
  margin-top: 16px;
  border-radius: 18px;
  border: 1px solid rgba(245,158,11,.24);
  background: rgba(255,255,255,.72);
  padding: 16px;
}
.no-hit-panel h3 { margin: 0 0 8px; font-size: 16px; }
.no-hit-panel p { margin: 0 0 12px; color: var(--muted); line-height: 1.7; }
.no-hit-actions { display: flex; flex-wrap: wrap; gap: 8px; }
.composer {
  position: sticky;
  bottom: 0;
  padding: 16px clamp(16px, 3vw, 34px);
  background: linear-gradient(180deg, rgba(255,255,255,.50), rgba(255,255,255,.92));
  backdrop-filter: blur(14px);
}
.composer-box {
  grid-template-columns: 1fr auto;
  max-width: 1180px;
  border: 1px solid rgba(48,92,255,.14);
  border-radius: 24px;
  background: #fff;
  padding: 10px;
  box-shadow: 0 14px 42px rgba(15,23,42,.08);
}
.composer textarea {
  border: 0;
  min-height: 72px;
  max-height: 220px;
  padding: 12px 14px;
}
.composer textarea:focus { outline: none; }
.composer-help {
  display: flex;
  align-items: center;
  gap: 7px;
  color: var(--muted);
  font-size: 12px;
  padding: 0 4px 8px;
}
.send-btn {
  min-width: 148px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  border-radius: 18px;
  background: var(--primary);
}
.drawer {
  border-radius: 28px;
  background: rgba(255,255,255,.88);
  backdrop-filter: blur(16px);
}
.drawer-head, .modal-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 12px;
  margin-bottom: 12px;
}
.drawer-empty { border-radius: 20px; }
.evidence-modal {
  place-items: stretch end;
}
.evidence-modal-card {
  width: min(760px, calc(100vw - 32px));
  height: calc(100vh - 32px);
  max-height: calc(100vh - 32px);
  border-radius: 28px;
}
.report-box {
  margin-top: 14px;
  max-width: 820px;
  background: rgba(255,255,255,.72);
}
.report-box summary {
  min-height: 42px;
  display: inline-flex;
  align-items: center;
  gap: 7px;
  color: var(--danger);
}
.report-grid { display: grid; grid-template-columns: minmax(180px, 240px) 1fr; gap: 10px; margin-top: 10px; }
.report-grid select, .report-box textarea {
  width: 100%;
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 10px 12px;
  background: #fff;
}
.pending-answer .bubble { background: transparent; border: 0; }
.loading-card {
  border: 1px solid rgba(48,92,255,.14);
  border-radius: 24px;
  background: rgba(255,255,255,.90);
  padding: 18px;
  box-shadow: 0 14px 42px rgba(15,23,42,.08);
}
.loading-steps { display: grid; gap: 9px; margin-top: 12px; }
.loading-step {
  display: flex;
  align-items: center;
  gap: 9px;
  color: var(--muted);
  font-size: 13px;
  font-weight: 800;
}
.loading-step .step-dot {
  width: 9px;
  height: 9px;
  border-radius: 999px;
  background: var(--primary);
  animation: dotPulse 1.35s infinite ease-in-out;
}
.loading-step:nth-child(2) .step-dot { animation-delay: .18s; }
.loading-step:nth-child(3) .step-dot { animation-delay: .36s; }
.loading-step:nth-child(4) .step-dot { animation-delay: .54s; }
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: .01ms !important;
    animation-iteration-count: 1 !important;
    scroll-behavior: auto !important;
    transition-duration: .01ms !important;
  }
}
@media (max-width: 1060px) {
  .empty-hero { grid-template-columns: 1fr; }
}
@media (max-width: 720px) {
  .workspace,
  .workspace.evidence-popup-mode {
    width: calc(100vw - 14px);
    grid-template-columns: 1fr;
  }
  .workspace-head-main { display: grid; }
  .dataset-badges { justify-content: flex-start; }
  .messages { max-height: none; padding: 18px; }
  .answer-card { border-radius: 22px; }
  .answer-card-head { display: grid; }
  .report-grid { grid-template-columns: 1fr; }
  .composer-box { grid-template-columns: 1fr; }
  .send-btn { width: 100%; }
  .evidence-modal { align-items: end; padding: 10px; }
  .evidence-modal-card {
    width: 100%;
    height: min(82vh, 760px);
    border-radius: 24px 24px 0 0;
  }
}
"""


APP_JS = """
function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function showPendingMessage(form) {
  const messages = document.querySelector(".messages");
  const textarea = form.querySelector('textarea[name="question"]');
  if (!messages || !textarea) return;
  const question = textarea.value.trim();
  if (!question) return;
  messages.querySelector(".empty")?.remove();
  messages.insertAdjacentHTML("beforeend", `
    <section class="message user pending-user"><div class="bubble">${escapeHtml(question).replaceAll("\\n", "<br>")}</div></section>
    <section class="message assistant pending-answer" data-pending-answer>
      <div class="bubble loading-card">
        <div class="loading-row">
          <span class="ms-icon" aria-hidden="true">sync</span>
          <span>根拠付きで確認しています</span>
          <span class="dots" aria-hidden="true"><i></i><i></i><i></i></span>
        </div>
        <div class="loading-steps" aria-label="回答生成の進捗">
          <div class="loading-step"><span class="step-dot"></span><span>質問を解析しています</span></div>
          <div class="loading-step"><span class="step-dot"></span><span>関連文書を探しています</span></div>
          <div class="loading-step"><span class="step-dot"></span><span>根拠を確認しています</span></div>
          <div class="loading-step"><span class="step-dot"></span><span>回答を作成しています</span></div>
        </div>
      </div>
    </section>
  `);
  messages.scrollTop = messages.scrollHeight;
}

function setSubmitting(form, active) {
  const button = form.querySelector('button[type="submit"]');
  if (active) {
    form.classList.add("is-submitting");
    if (button) {
      button.dataset.originalText = button.textContent;
      button.textContent = "確認しています";
      button.disabled = true;
    }
    return;
  }
  form.classList.remove("is-submitting");
  if (button) {
    button.textContent = button.dataset.originalText || "根拠付きで確認";
    button.disabled = false;
  }
}

function showSubmitError(message) {
  const pending = document.querySelector("[data-pending-answer]");
  if (!pending) return;
  pending.outerHTML = `
    <section class="message assistant">
      <div class="answer-card no-hit">
        <div class="answer-state"><span class="ms-icon" aria-hidden="true">warning</span>送信に失敗しました</div>
        <p>RAG APIの起動状態を確認してください。</p>
        <div class="error">${escapeHtml(message)}</div>
      </div>
    </section>
  `;
}

function closeEvidenceModal() {
  const root = document.getElementById("evidence-modal-root");
  if (root) root.innerHTML = "";
  document.body.classList.remove("modal-open");
}

let sidebarHover = false;
let sidebarFocus = false;

function sidebarParts() {
  return {
    workspace: document.getElementById("workspace"),
    sidebar: document.getElementById("settings-sidebar"),
    mobileToggle: document.querySelector("[data-sidebar-mobile-toggle]")
  };
}

function syncSidebarState() {
  const { workspace, sidebar, mobileToggle } = sidebarParts();
  if (!workspace || !sidebar) return;
  const pinned = localStorage.getItem("ragSidebarPinned") === "1";
  workspace.classList.toggle("sidebar-pinned", pinned);
  workspace.classList.toggle("sidebar-expanded", pinned || sidebarHover || sidebarFocus);
  const pin = sidebar.querySelector("[data-sidebar-pin]");
  if (pin) {
    pin.setAttribute("aria-pressed", pinned ? "true" : "false");
    pin.textContent = pinned ? "固定中" : "ピン留め";
  }
  if (mobileToggle) {
    mobileToggle.setAttribute("aria-expanded", workspace.classList.contains("sidebar-mobile-open") ? "true" : "false");
  }
}

function closeMobileSidebar() {
  const { workspace, mobileToggle } = sidebarParts();
  if (!workspace) return;
  workspace.classList.remove("sidebar-mobile-open");
  if (mobileToggle) mobileToggle.setAttribute("aria-expanded", "false");
}

function bindSidebar() {
  const { workspace, sidebar, mobileToggle } = sidebarParts();
  if (!workspace || !sidebar) return;
  syncSidebarState();

  if (sidebar.dataset.sidebarBound !== "1") {
    sidebar.dataset.sidebarBound = "1";
    sidebar.addEventListener("mouseenter", () => { sidebarHover = true; syncSidebarState(); });
    sidebar.addEventListener("mouseleave", () => { sidebarHover = false; syncSidebarState(); });
    sidebar.addEventListener("focusin", () => { sidebarFocus = true; syncSidebarState(); });
    sidebar.addEventListener("focusout", () => {
      window.setTimeout(() => {
        sidebarFocus = sidebar.contains(document.activeElement);
        syncSidebarState();
      }, 0);
    });
    sidebar.querySelector("[data-sidebar-rail]")?.addEventListener("click", () => {
      syncSidebarState();
      sidebar.querySelector(".settings-body input, .settings-body button, .settings-body textarea")?.focus();
    });
  }

  const pin = sidebar.querySelector("[data-sidebar-pin]");
  if (pin && pin.dataset.sidebarBound !== "1") {
    pin.dataset.sidebarBound = "1";
    pin.addEventListener("click", () => {
      const next = localStorage.getItem("ragSidebarPinned") !== "1";
      localStorage.setItem("ragSidebarPinned", next ? "1" : "0");
      syncSidebarState();
    });
  }

  const backdrop = workspace.querySelector("[data-sidebar-backdrop]");
  if (backdrop && backdrop.dataset.sidebarBound !== "1") {
    backdrop.dataset.sidebarBound = "1";
    backdrop.addEventListener("click", closeMobileSidebar);
  }

  if (mobileToggle && mobileToggle.dataset.sidebarBound !== "1") {
    mobileToggle.dataset.sidebarBound = "1";
    mobileToggle.addEventListener("click", () => {
      const current = document.getElementById("workspace");
      const currentSidebar = document.getElementById("settings-sidebar");
      if (!current || !currentSidebar) return;
      const open = !current.classList.contains("sidebar-mobile-open");
      current.classList.toggle("sidebar-mobile-open", open);
      mobileToggle.setAttribute("aria-expanded", open ? "true" : "false");
      if (open) currentSidebar.focus({ preventScroll: true });
    });
  }
}

function activeComposer() {
  return document.querySelector("[data-chat-form]");
}

function fillComposer(question, submitNow) {
  const form = activeComposer();
  if (!form) return;
  const textarea = form.querySelector('textarea[name="question"]');
  if (!textarea) return;
  textarea.value = question || "";
  textarea.focus();
  textarea.dispatchEvent(new Event("input", { bubbles: true }));
  if (submitNow && textarea.value.trim()) {
    form.requestSubmit();
  }
}

function bindQuestionTemplates() {
  document.querySelectorAll("[data-fill-question]").forEach((button) => {
    if (button.dataset.bound === "1") return;
    button.dataset.bound = "1";
    button.addEventListener("click", () => fillComposer(button.dataset.fillQuestion || "", button.dataset.submitQuestion === "1"));
  });
}

function bindComposerShortcuts() {
  document.querySelectorAll("[data-chat-form] textarea[name='question']").forEach((textarea) => {
    if (textarea.dataset.shortcutBound === "1") return;
    textarea.dataset.shortcutBound = "1";
    textarea.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        textarea.form?.requestSubmit();
      }
    });
  });
}

function bindFastRag() {
  bindSidebar();
  bindQuestionTemplates();
  bindComposerShortcuts();

  document.querySelectorAll("[data-modal-close]").forEach((button) => {
    if (button.dataset.bound === "1") return;
    button.dataset.bound = "1";
    button.addEventListener("click", closeEvidenceModal);
  });

  document.querySelectorAll("[data-chat-form]").forEach((form) => {
    if (form.dataset.bound === "1") return;
    form.dataset.bound = "1";
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (form.classList.contains("is-submitting")) return;
      document.body.classList.add("busy");
      setSubmitting(form, true);
      showPendingMessage(form);
      const workspace = document.getElementById("workspace");
      try {
        const res = await fetch(form.action, { method: "POST", body: new FormData(form) });
        workspace.outerHTML = await res.text();
      } catch (err) {
        showSubmitError(String(err));
      } finally {
        document.body.classList.remove("busy");
        setSubmitting(form, false);
        bindFastRag();
      }
    });
  });

  document.querySelectorAll("[data-evidence-url]").forEach((button) => {
    if (button.dataset.bound === "1") return;
    button.dataset.bound = "1";
    button.addEventListener("click", async (event) => {
      event.preventDefault();
      const res = await fetch(button.dataset.evidenceUrl);
      const html = await res.text();
      const modalRoot = document.getElementById("evidence-modal-root");
      if (modalRoot) {
        modalRoot.innerHTML = html;
        document.body.classList.add("modal-open");
        bindFastRag();
        return;
      }
      const drawer = document.getElementById("evidence-drawer");
      if (drawer) drawer.outerHTML = html;
      bindFastRag();
    });
  });

  document.querySelectorAll("[data-report-form]").forEach((form) => {
    if (form.dataset.bound === "1") return;
    form.dataset.bound = "1";
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const target = document.getElementById(form.dataset.reportTarget);
      const res = await fetch(form.action, { method: "POST", body: new FormData(form) });
      target.outerHTML = await res.text();
    });
  });
}
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeEvidenceModal();
  if (event.key === "Escape") closeMobileSidebar();
});
document.addEventListener("DOMContentLoaded", bindFastRag);
"""


def selected_default(corpora: list[dict]) -> list[str]:
    return [str(c["corpus_id"]) for c in corpora]


def toolbar_html(api_url: str, selected: list[str], top_k: int, corpora: list[dict], form_id: str, session_id: str) -> str:
    selected_set = set(selected)
    corpus_items = []
    for corpus in corpora:
        corpus_id = str(corpus.get("corpus_id"))
        checked = "checked" if corpus_id in selected_set else ""
        corpus_items.append(
            f"""
            <label class="sidebar-corpus">
              <input type="checkbox" name="corpus_ids" value="{esc(corpus_id)}" form="{esc(form_id)}" {checked}>
              <span>{esc(corpus.get("display_name"))}</span>
            </label>
            """
        )
    dock_items = [
        ("add", "新しい質問", "/"),
        ("folder", "文書セット", "#document-sets"),
        ("history", "質問履歴", "#chat-history"),
        ("star", "質問例", "#question-templates"),
        ("help", "使い方", "#usage-guide"),
        ("tune", "設定", "#settings"),
    ]
    dock_html = "".join(
        f"""
        <a class="dock-link" href="{esc(href)}">
          {ms_icon(icon)}
          <span class="dock-label">{esc(label)}</span>
        </a>
        """
        for icon, label, href in dock_items
    )
    return f"""
    <aside id="settings-sidebar" class="settings" tabindex="-1">
      <button type="button" class="settings-rail" data-sidebar-rail aria-label="メニューを開く">
        <span class="rail-text">{ms_icon("chat")}<span>RAG</span></span>
      </button>
      <div class="settings-body">
        <div class="settings-head">
          <h2>Evidence Dock</h2>
          <button type="button" class="pin-btn" data-sidebar-pin aria-pressed="false" aria-label="Dockを固定">ピン留め</button>
        </div>
        <nav class="dock-menu" aria-label="ユーザーメニュー">
          {dock_html}
        </nav>
        <div class="sidebar-field">
          <label id="settings">根拠数</label>
          <div class="sidebar-range">
            <input type="range" name="top_k" min="3" max="15" value="{int(top_k)}" form="{esc(form_id)}" oninput="this.nextElementSibling.value=this.value">
            <output>{int(top_k)}</output>
          </div>
        </div>
        <div id="document-sets" class="sidebar-label">対象資料 {len(selected)}件</div>
        <div class="sidebar-corpus-list">{"".join(corpus_items)}</div>
        <p class="session">選択した文書セットと根拠数は、次の質問から反映されます。</p>
        <div id="usage-guide" class="sidebar-field">
          <span class="sidebar-label">使い方</span>
          <p class="session">質問すると、回答カードと根拠チップが表示されます。根拠チップを押すと本文を確認できます。</p>
        </div>
      </div>
    </aside>
    <div class="sidebar-backdrop" data-sidebar-backdrop></div>
    """


def answer_badge(message: dict) -> str:
    if message.get("cache_hit"):
        sim = message.get("cache_similarity")
        sim_text = " / 承認済み回答を再利用" if isinstance(sim, (int, float)) else ""
        return f'<span class="badge cache">{ms_icon("bookmark", "small")}承認済みQA{esc(sim_text)}</span>'
    answer_source = "時系列" if message.get("answer_source") == "meeting_timeline" else "RAG検索"
    return f'<span class="badge">{ms_icon("check", "small")}回答元: {esc(answer_source)}</span>'


def source_chips(session_id: str, message_index: int, sources: list[dict]) -> str:
    if not sources:
        return """
        <div class="no-hit-panel">
          <h3>十分な根拠が見つかりませんでした</h3>
          <p>対象文書が未登録、別表現で記載、または質問が広すぎる可能性があります。</p>
          <div class="no-hit-actions">
            <button type="button" class="answer-action" data-fill-question="もう少し具体的に、対象の会議名や論点を入れて確認したい">
              <span class="ms-icon" aria-hidden="true">edit</span>質問を言い換える
            </button>
            <button type="button" class="answer-action" data-fill-question="この内容について関連する資料や議事録を探して">
              <span class="ms-icon" aria-hidden="true">search</span>質問例を見る
            </button>
          </div>
        </div>
        """
    chips = []
    for source_index, source in enumerate(sources):
        title = source_title(source)
        url = source_url(session_id, message_index, source_index)
        label, label_class = match_label(source)
        chips.append(
            f"""
            <button type="button" class="source-chip" data-evidence-url="{esc(url)}" title="{esc(title)}" aria-label="根拠{source_index + 1}を開く">
              {ms_icon("article", "small")}
              <span class="chip-text">根拠{source_index + 1}: {esc(source_summary_label(source))}</span>
              <span class="match-pill match-{esc(label_class)}">{esc(label)}</span>
            </button>
            """
        )
    return f'<div class="source-strip">{"".join(chips)}</div>'


def report_box(session_id: str, message_index: int, api_url: str, message: dict) -> str:
    target_id = f"report-{message_index}"
    if message.get("reported"):
        return f'<div id="{target_id}" class="report-status">{ms_icon("check", "small")}報告を受け付けました。</div>'
    if not message.get("log_id"):
        return f'<div id="{target_id}"></div>'
    return f"""
    <details id="{target_id}" class="report-box">
      <summary>{ms_icon("report", "filled small")}この回答を報告</summary>
      <form action="/report" data-report-form data-report-target="{esc(target_id)}">
        <input type="hidden" name="session_id" value="{esc(session_id)}">
        <input type="hidden" name="message_index" value="{message_index}">
        <input type="hidden" name="api_url" value="{esc(api_url)}">
        <div class="report-grid">
          <select name="reason" aria-label="報告理由">
            <option value="根拠と違うかもしれない">根拠と違うかもしれない</option>
            <option value="回答が不十分">回答が不十分</option>
            <option value="探している情報ではない">探している情報ではない</option>
            <option value="その他">その他</option>
          </select>
          <textarea name="comment" placeholder="気になった点を短く書いてください"></textarea>
        </div>
        <div class="report-actions">
          <span class="session">報告は改善のために利用されます。</span>
          <button type="submit" class="answer-action report">{ms_icon("feedback", "small")}報告する</button>
        </div>
      </form>
    </details>
    """


def messages_html(session_id: str, api_url: str) -> str:
    messages = store.ensure(session_id)
    if not messages:
        return """
        <div class="empty">
          <div>
            <strong>税務文書を、会話で探す。</strong>
            <span>質問すると回答下に引用ボタンが並び、右側に根拠が開きます。</span>
          </div>
        </div>
        """
    rows = []
    for idx, message in enumerate(messages):
        role = message.get("role")
        if role == "user":
            rows.append(f'<section class="message user"><div class="bubble">{text_block(message.get("content"))}</div></section>')
            continue
        sources = message.get("sources") or []
        answer_html = answer_markdown(message.get("content"), session_id, idx, sources)
        rows.append(
            f"""
            <section class="message assistant">
              <div class="bubble markdown-body">{answer_html}{evidence_link_row(session_id, idx, sources)}</div>
              <div class="assistant-meta">{answer_badge(message)}</div>
              {source_chips(session_id, idx, sources)}
              {report_box(session_id, idx, api_url, message)}
            </section>
            """
        )
    return "".join(rows)


def question_templates_html() -> str:
    templates = [
        "この資料の要点を根拠付きで教えて",
        "前回会議からの変更点を整理して",
        "決定事項だけを時系列で教えて",
        "未完了の宿題と担当を確認したい",
        "この文書で決まっているルールを一覧にして",
    ]
    cards = "".join(
        f"""
        <button type="button" class="template-card" data-fill-question="{esc(question)}">
          <span>{esc(question)}</span>
          {ms_icon("arrow_forward", "small")}
        </button>
        """
        for question in templates
    )
    return f'<div id="question-templates" class="template-grid">{cards}</div>'


def answer_actions(session_id: str, message_index: int, sources: list[dict]) -> str:
    if not sources:
        return ""
    first_url = source_url(session_id, message_index, 0)
    return f"""
    <div class="answer-actions">
      <button type="button" class="answer-action primary" data-evidence-url="{esc(first_url)}">
        {ms_icon("link", "small")}根拠を見る
      </button>
      <span class="answer-note">登録文書を根拠に生成しています。重要な判断では根拠本文も確認してください。</span>
    </div>
    """


def messages_html_refreshed(session_id: str, api_url: str) -> str:
    messages = store.ensure(session_id)
    if not messages:
        return f"""
        <div class="empty">
          <div class="empty-hero">
            <div>
              <div class="empty-eyebrow">Evidence Workspace</div>
              <h1>今日は何を確認しますか？</h1>
              <p>会議資料や議事録を根拠に、要点・決定事項・変更点・未完了事項を確認できます。</p>
            </div>
            {question_templates_html()}
          </div>
        </div>
        """
    rows = []
    for idx, message in enumerate(messages):
        role = message.get("role")
        if role == "user":
            rows.append(f'<section class="message user"><div class="bubble">{text_block(message.get("content"))}</div></section>')
            continue
        sources = message.get("sources") or []
        answer_html = answer_markdown(message.get("content"), session_id, idx, sources)
        quality_class = "no-hit" if not sources else ("low-confidence" if has_low_confidence(sources) else "")
        rows.append(
            f"""
            <section class="message assistant">
              <article class="answer-card {quality_class}">
                <div class="answer-card-head">
                  <div class="answer-state">{ms_icon("check", "small")}回答を生成しました</div>
                  <div class="assistant-meta">{answer_badge(message)}</div>
                </div>
                <div class="markdown-body answer-body">{answer_html}{evidence_link_row(session_id, idx, sources)}</div>
                {source_chips(session_id, idx, sources)}
                {answer_actions(session_id, idx, sources)}
                {report_box(session_id, idx, api_url, message)}
              </article>
            </section>
            """
        )
    return "".join(rows)


def latest_source(session_id: str) -> dict | None:
    for message in reversed(store.ensure(session_id)):
        sources = message.get("sources") or []
        if sources:
            return sources[0]
    return None


def evidence_drawer(source: dict | None = None, source_index: int | None = None) -> str:
    if not source:
        return """
        <aside id="evidence-drawer" class="drawer">
          <h2>根拠</h2>
          <div class="drawer-empty">回答下の根拠ボタンを押すと、ここに参照チャンクが表示されます。</div>
        </aside>
        """
    metadata = [
        ("スコア", score_text(source)),
        ("参照元", source_file_name(source)),
        ("見出し", source.get("heading_path")),
    ]
    kvs = "".join(
        f'<div class="kv"><span>{esc(label)}</span><strong>{esc(value)}</strong></div>'
        for label, value in metadata
        if value not in (None, "")
    )
    parent = source.get("parent_text")
    parent_html = (
        f'<details><summary>親チャンク全文</summary><div class="chunk markdown-body">{markdown_block(parent)}</div></details>'
        if parent
        else ""
    )
    number = "" if source_index is None else f"根拠{source_index + 1}: "
    return f"""
    <aside id="evidence-drawer" class="drawer">
      <h2>根拠</h2>
      <p class="evidence-title">{esc(number)}{esc(source_title(source))}</p>
      {kvs}
      <div class="chunk">
        <h3>ヒットした子チャンク</h3>
        <div class="markdown-body">{markdown_block(source.get("child_text") or source.get("text") or "")}</div>
      </div>
      {parent_html}
    </aside>
    """


def evidence_modal(source: dict | None = None, source_index: int | None = None) -> str:
    if not source:
        return """
        <div class="evidence-modal" role="dialog" aria-modal="true" aria-label="根拠詳細">
          <button type="button" class="evidence-modal-backdrop" data-modal-close aria-label="閉じる"></button>
          <section class="evidence-modal-card">
            <button type="button" class="modal-close" data-modal-close aria-label="閉じる">×</button>
            <h2>根拠</h2>
            <div class="drawer-empty">根拠はありません。</div>
          </section>
        </div>
        """
    metadata = [
        ("スコア", score_text(source)),
        ("参照元", source_file_name(source)),
        ("見出し", source.get("heading_path")),
    ]
    kvs = "".join(
        f'<div class="kv"><span>{esc(label)}</span><strong>{esc(value)}</strong></div>'
        for label, value in metadata
        if value not in (None, "")
    )
    parent = source.get("parent_text")
    parent_html = (
        f'<details><summary>親チャンク全文</summary><div class="chunk markdown-body">{markdown_block(parent)}</div></details>'
        if parent
        else ""
    )
    number = "" if source_index is None else f"根拠{source_index + 1}: "
    return f"""
    <div class="evidence-modal" role="dialog" aria-modal="true" aria-label="根拠詳細">
      <button type="button" class="evidence-modal-backdrop" data-modal-close aria-label="閉じる"></button>
      <section class="evidence-modal-card">
        <button type="button" class="modal-close" data-modal-close aria-label="閉じる">×</button>
        <h2>根拠</h2>
        <p class="evidence-title">{esc(number)}{esc(source_title(source))}</p>
        {kvs}
        <div class="chunk">
          <h3>ヒットした子チャンク</h3>
          <div class="markdown-body">{markdown_block(source.get("child_text") or source.get("text") or "")}</div>
        </div>
        {parent_html}
      </section>
    </div>
    """


def evidence_detail_body(source: dict | None = None, source_index: int | None = None) -> str:
    if not source:
        return """
        <div class="drawer-head">
          <div>
            <h2>根拠詳細</h2>
            <p class="answer-note">根拠はありません。</p>
          </div>
        </div>
        <div class="drawer-empty">回答に紐づく根拠が見つかりませんでした。</div>
        """
    label, label_class = match_label(source)
    number = "" if source_index is None else f"根拠{source_index + 1}: "
    visible_meta = [
        ("一致度", label),
        ("参照元", source_file_name(source)),
        ("見出し", source.get("heading_path")),
        ("会議名", source.get("meeting_name")),
        ("会議日", source.get("meeting_date")),
        ("資料種別", source.get("source_type")),
        ("スライド", source.get("slide_no")),
        ("議題", source.get("agenda") or source.get("topic") or source.get("section_title")),
    ]
    kvs = "".join(
        f'<div class="kv"><span>{esc(key)}</span><strong>{esc(value)}</strong></div>'
        for key, value in visible_meta
        if value not in (None, "")
    )
    internal = [
        ("score", score_text(source)),
        ("corpus_id", source.get("corpus_id")),
        ("parent_id", source.get("parent_id")),
        ("child_id", source.get("child_id")),
    ]
    internal_kvs = "".join(
        f'<div class="kv"><span>{esc(key)}</span><strong>{esc(value)}</strong></div>'
        for key, value in internal
        if value not in (None, "")
    )
    parent = source.get("parent_text")
    parent_html = (
        f'<details><summary>親チャンク全文を見る</summary><div class="chunk markdown-body">{markdown_block(parent)}</div></details>'
        if parent
        else ""
    )
    internal_html = f'<details><summary>内部情報</summary>{internal_kvs}</details>' if internal_kvs else ""
    return f"""
    <div class="drawer-head">
      <div>
        <h2>根拠詳細</h2>
        <p class="evidence-title">{esc(number)}{esc(source_title(source))}</p>
      </div>
      <span class="match-pill match-{esc(label_class)}">{esc(label)}</span>
    </div>
    {kvs}
    <div class="chunk">
      <h3>ヒットした本文</h3>
      <div class="markdown-body">{markdown_block(source.get("child_text") or source.get("text") or "")}</div>
    </div>
    {parent_html}
    {internal_html}
    """


def evidence_drawer_refreshed(source: dict | None = None, source_index: int | None = None) -> str:
    return f'<aside id="evidence-drawer" class="drawer">{evidence_detail_body(source, source_index)}</aside>'


def evidence_modal_refreshed(source: dict | None = None, source_index: int | None = None) -> str:
    return f"""
    <div class="evidence-modal" role="dialog" aria-modal="true" aria-label="根拠詳細">
      <button type="button" class="evidence-modal-backdrop" data-modal-close aria-label="閉じる"></button>
      <section class="evidence-modal-card">
        <button type="button" class="modal-close" data-modal-close aria-label="閉じる">{ms_icon("close")}</button>
        {evidence_detail_body(source, source_index)}
      </section>
    </div>
    """


def evidence_response(source: dict | None = None, source_index: int | None = None) -> str:
    return evidence_modal_refreshed(source, source_index) if EVIDENCE_UI_MODE == "popup" else evidence_drawer_refreshed(source, source_index)


def workspace_html(
    *,
    session_id: str,
    api_url: str,
    selected: list[str],
    top_k: int,
    notice: str = "",
) -> str:
    corpora = load_corpora()
    form_id = f"chat-form-{session_id}"
    source = latest_source(session_id)
    mode_class = " evidence-popup-mode" if EVIDENCE_UI_MODE == "popup" else ""
    evidence_html = '<div id="evidence-modal-root" class="evidence-modal-root"></div>' if EVIDENCE_UI_MODE == "popup" else evidence_drawer_refreshed(source, 0 if source else None)
    selected_labels = [
        str(corpus.get("display_name") or corpus.get("corpus_id"))
        for corpus in corpora
        if str(corpus.get("corpus_id")) in set(selected)
    ]
    dataset_label = "、".join(selected_labels[:2]) + (" ほか" if len(selected_labels) > 2 else "")
    return f"""
    <main id="workspace" class="workspace{mode_class}">
      {toolbar_html(api_url, selected, top_k, corpora, form_id, session_id)}
      <section class="chat-panel">
        <header class="workspace-head">
          <div class="workspace-head-main">
            <div>
              <div class="workspace-kicker">RAG Search Console</div>
              <h1 class="workspace-title">業務文書を根拠付きで探索するAIチャット</h1>
              <p class="workspace-copy">質問に集中し、必要な時だけ根拠チップから本文を確認できます。</p>
            </div>
            <div class="dataset-badges" aria-label="現在の検索対象">
              <span class="dataset-badge">{ms_icon("folder", "small")}対象資料: {esc(dataset_label or "未選択")}</span>
              <span class="dataset-badge">{ms_icon("article", "small")}根拠: 最大{int(top_k)}件</span>
            </div>
          </div>
        </header>
        {'<div class="notice">' + esc(notice) + '</div>' if notice else ''}
        <div id="chat-history" class="messages">{messages_html_refreshed(session_id, api_url)}</div>
        <form id="{esc(form_id)}" action="/ask" data-chat-form class="composer">
          <input type="hidden" name="session_id" value="{esc(session_id)}">
          <input type="hidden" name="api_url" value="{esc(api_url)}">
          <div class="composer-help">{ms_icon("info", "small")}Enterで質問、Shift+Enterで改行。回答後は根拠チップから本文を確認できます。</div>
          <div class="composer-box">
            <textarea name="question" placeholder="会議資料・議事録・時系列の経緯について質問してください"></textarea>
            <button type="submit" class="send-btn">{ms_icon("send", "small")}根拠付きで確認</button>
          </div>
        </form>
      </section>
      {evidence_html}
    </main>
    """


def page_html(session_id: str, api_url: str, selected: list[str], top_k: int) -> str:
    return f"""
    <!doctype html>
    <html lang="ja">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Meeting Timeline RAG | Modern Chat</title>
      <style>{MODERN_CSS}{UX_REFRESH_CSS}</style>
    </head>
    <body>
      <div class="shell">
        <nav class="nav">
          <div>
            <div class="logo"><span class="logo-mark">R</span><span>Evidence Workspace</span></div>
            <div class="nav-sub">業務文書を根拠付きで探索できるAIチャット</div>
          </div>
          <div class="nav-actions">
            <button type="button" class="mobile-menu-btn" data-sidebar-mobile-toggle aria-controls="settings-sidebar" aria-expanded="false">{ms_icon("menu", "small")}メニュー</button>
            <a class="pill-link" href="/">{ms_icon("add", "small")}新しい質問を始める</a>
          </div>
        </nav>
        {workspace_html(session_id=session_id, api_url=api_url, selected=selected, top_k=top_k)}
      </div>
      <script>{APP_JS}</script>
    </body>
    </html>
    """


async def home(request: Request) -> HTMLResponse:
    session_id = request.query_params.get("session_id") or str(uuid.uuid4())
    corpora = load_corpora()
    selected = request.query_params.getlist("corpus_ids") or selected_default(corpora)
    top_k = int(request.query_params.get("top_k") or DEFAULT_TOP_K)
    api_url = request.query_params.get("api_url") or DEFAULT_API_URL
    store.ensure(session_id)
    return HTMLResponse(page_html(session_id, api_url, selected, top_k))


async def ask(request: Request) -> HTMLResponse:
    form = await request.form()
    session_id = str(form.get("session_id") or uuid.uuid4())
    api_url = str(form.get("api_url") or DEFAULT_API_URL)
    selected = [str(value) for value in form.getlist("corpus_ids") if str(value).strip()]
    top_k = int(form.get("top_k") or DEFAULT_TOP_K)
    question = str(form.get("question") or "").strip()
    store.ensure(session_id)

    if not selected:
        return HTMLResponse(workspace_html(session_id=session_id, api_url=api_url, selected=selected, top_k=top_k, notice="検索対象文書を1つ以上選択してください。"))
    if not question:
        return HTMLResponse(workspace_html(session_id=session_id, api_url=api_url, selected=selected, top_k=top_k, notice="質問を入力してください。"))

    history = store.history_for_api(session_id)
    store.ensure(session_id).append({"role": "user", "content": question})
    payload = {
        "question": question,
        "corpus_ids": selected,
        "top_k": top_k,
        "show_debug": False,
        "session_id": session_id,
        "history": history,
    }
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(api_url, json=payload)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        store.ensure(session_id).pop()
        return HTMLResponse(workspace_html(session_id=session_id, api_url=api_url, selected=selected, top_k=top_k, notice=f"API呼び出しに失敗しました: {exc}"))

    session_id = data.get("session_id") or session_id
    store.ensure(session_id).append(
        {
            "role": "assistant",
            "content": data.get("answer", ""),
            "sources": data.get("sources", []),
            "log_id": data.get("log_id"),
            "answer_source": data.get("answer_source"),
            "cache_hit": data.get("cache_hit", False),
            "qa_cache_id": data.get("qa_cache_id"),
            "cache_similarity": data.get("cache_similarity"),
            "question": question,
        }
    )
    return HTMLResponse(workspace_html(session_id=session_id, api_url=api_url, selected=selected, top_k=top_k))


async def source(request: Request) -> HTMLResponse:
    session_id = str(request.query_params.get("session_id") or "")
    message_index = int(request.query_params.get("message_index") or -1)
    source_index = int(request.query_params.get("source_index") or -1)
    item = store.source(session_id, message_index, source_index)
    return HTMLResponse(evidence_response(item, source_index if item else None))


async def report(request: Request) -> HTMLResponse:
    form = await request.form()
    session_id = str(form.get("session_id") or "")
    api_url = str(form.get("api_url") or DEFAULT_API_URL)
    message_index = int(form.get("message_index") or -1)
    comment = str(form.get("comment") or "")
    reason = str(form.get("reason") or "").strip()
    if reason:
        comment = f"理由: {reason}\n{comment}".strip()
    target_id = f"report-{message_index}"
    messages = store.ensure(session_id)
    if message_index < 0 or message_index >= len(messages):
        return HTMLResponse(f'<div id="{target_id}" class="error">対象の回答が見つかりません。</div>')
    message = messages[message_index]
    payload = {
        "question": message.get("question") or "",
        "answer": message.get("content") or "",
        "session_id": session_id,
        "log_id": message.get("log_id"),
        "comment": comment,
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(f"{api_base_url(api_url)}/feedback/hallucination", json=payload)
            response.raise_for_status()
            report_id = response.json().get("report_id")
    except Exception as exc:
        return HTMLResponse(f'<div id="{target_id}" class="error">報告に失敗しました: {esc(exc)}</div>')
    message["reported"] = True
    message["report_id"] = report_id
    return HTMLResponse(f'<div id="{target_id}" class="report-status">{ms_icon("check", "small")}報告を受け付けました。</div>')


async def health(_: Request) -> HTMLResponse:
    return HTMLResponse("ok")


rt("/", methods=["GET"])(home)
rt("/ask", methods=["POST"])(ask)
rt("/source", methods=["GET"])(source)
rt("/report", methods=["POST"])(report)
rt("/health", methods=["GET"])(health)


if __name__ == "__main__":
    serve()
