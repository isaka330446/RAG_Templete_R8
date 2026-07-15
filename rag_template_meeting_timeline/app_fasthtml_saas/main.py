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
    text = esc(value)
    return text.replace("\n", "<br>")


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


def short_text(value: object, limit: int = 72) -> str:
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


SAAS_CSS = """
@font-face {
  font-family: "Noto Sans JP Local";
  src: url("/static/fonts/NotoSansJP-wght.ttf") format("truetype");
  font-weight: 100 900;
  font-style: normal;
  font-display: swap;
}
:root {
  color-scheme: light;
  --bg: #f5f7fb;
  --panel: #ffffff;
  --panel-soft: #f9fbfd;
  --line: #dfe5ec;
  --ink: #1d2733;
  --muted: #6a7482;
  --primary: #0f766e;
  --primary-dark: #115e59;
  --accent: #2563eb;
  --danger: #b42318;
  --warn-bg: #fff7ed;
  --shadow: 0 16px 40px rgba(22, 34, 51, 0.09);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  min-height: 100vh;
  background: var(--bg);
  color: var(--ink);
  font-family: "Noto Sans JP Local", "Noto Sans JP", "Noto Sans CJK JP", "Yu Gothic UI", "Meiryo", sans-serif;
}
a { color: inherit; text-decoration: none; }
button, input, textarea, select { font: inherit; }
.app-frame { min-height: 100vh; display: flex; flex-direction: column; }
.topbar {
  height: 64px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 20px;
  padding: 0 24px;
  border-bottom: 1px solid var(--line);
  background: rgba(255,255,255,0.92);
  backdrop-filter: blur(10px);
  position: sticky;
  top: 0;
  z-index: 20;
}
.brand { display: flex; flex-direction: column; gap: 2px; }
.brand strong { font-size: 18px; letter-spacing: 0; }
.brand span { color: var(--muted); font-size: 12px; }
.top-actions { display: flex; align-items: center; gap: 10px; }
.ghost-link, .primary-btn, .secondary-btn, .source-btn, .mobile-menu-btn, .pin-btn, .settings-rail {
  border: 1px solid var(--line);
  background: #fff;
  color: var(--ink);
  border-radius: 8px;
  padding: 9px 12px;
  cursor: pointer;
}
.mobile-menu-btn { display: none; }
.primary-btn { background: var(--primary); border-color: var(--primary); color: #fff; font-weight: 700; }
.primary-btn:hover { background: var(--primary-dark); }
.secondary-btn:hover, .ghost-link:hover, .source-btn:hover, .mobile-menu-btn:hover, .pin-btn:hover, .settings-rail:hover {
  border-color: var(--primary);
  color: var(--primary-dark);
}
.workspace {
  width: min(1900px, calc(100vw - 32px));
  margin: 16px auto 24px;
  display: grid;
  grid-template-columns: 56px minmax(560px, 1.7fr) minmax(380px, .95fr);
  gap: 16px;
  align-items: start;
  transition: grid-template-columns .22s ease;
}
.workspace.sidebar-expanded,
.workspace.sidebar-pinned,
.workspace:has(.settings:hover),
.workspace:has(.settings:focus-within) {
  grid-template-columns: minmax(250px, 300px) minmax(560px, 1.7fr) minmax(380px, .95fr);
}
.workspace.evidence-popup-mode {
  grid-template-columns: 56px minmax(560px, 1fr);
}
.workspace.evidence-popup-mode.sidebar-expanded,
.workspace.evidence-popup-mode.sidebar-pinned,
.workspace.evidence-popup-mode:has(.settings:hover),
.workspace.evidence-popup-mode:has(.settings:focus-within) {
  grid-template-columns: minmax(250px, 300px) minmax(560px, 1fr);
}
.card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: var(--shadow);
}
.settings {
  padding: 0;
  position: sticky;
  top: 80px;
  min-height: 280px;
  overflow: hidden;
  transition: padding .18s ease, box-shadow .18s ease;
}
.workspace.sidebar-expanded .settings,
.workspace.sidebar-pinned .settings,
.workspace:has(.settings:hover) .settings,
.workspace:has(.settings:focus-within) .settings {
  padding: 16px;
}
.settings-rail {
  width: 100%;
  height: 100%;
  min-height: 280px;
  display: grid;
  place-items: center;
  border: 0;
  border-radius: 8px;
  background: var(--panel);
  color: var(--primary-dark);
  font-weight: 800;
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
  margin-bottom: 12px;
}
.settings-head h2 { margin: 0; }
.pin-btn {
  padding: 6px 9px;
  font-size: 12px;
  font-weight: 700;
}
.pin-btn[aria-pressed="true"] {
  background: var(--primary);
  border-color: var(--primary);
  color: #fff;
}
.sidebar-backdrop { display: none; }
.settings h2, .evidence h2 { font-size: 15px; margin: 0 0 12px; }
.field { display: grid; gap: 6px; margin-bottom: 14px; }
.field label, .corpus-title { color: var(--muted); font-size: 12px; font-weight: 700; }
.field input[type="text"], .field textarea {
  width: 100%;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px 11px;
  background: #fff;
}
.range-row { display: grid; grid-template-columns: 1fr 42px; gap: 10px; align-items: center; }
.range-row output { text-align: center; color: var(--primary); font-weight: 700; }
.corpus-list { display: grid; gap: 8px; margin: 8px 0 16px; }
.corpus-item {
  display: grid;
  grid-template-columns: 20px 1fr;
  gap: 8px;
  padding: 9px;
  background: var(--panel-soft);
  border: 1px solid var(--line);
  border-radius: 8px;
}
.corpus-item span { font-size: 13px; line-height: 1.4; }
.chat.card { min-height: calc(100vh - 104px); display: flex; flex-direction: column; overflow: hidden; }
.chat-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 14px 16px;
  border-bottom: 1px solid var(--line);
  background: var(--panel-soft);
}
.chat-header strong { font-size: 15px; }
.session { color: var(--muted); font-size: 12px; word-break: break-all; }
.messages { flex: 1; padding: 18px; display: grid; gap: 16px; overflow: auto; max-height: calc(100vh - 290px); }
.empty {
  min-height: 260px;
  display: grid;
  place-items: center;
  color: var(--muted);
  text-align: center;
  border: 1px dashed var(--line);
  border-radius: 8px;
  background: var(--panel-soft);
}
.message { display: grid; gap: 8px; }
.bubble {
  max-width: 92%;
  padding: 13px 15px;
  border-radius: 8px;
  line-height: 1.75;
  white-space: normal;
}
.message.user { justify-items: end; }
.message.user .bubble { max-width: min(720px, 78%); background: var(--primary); color: #fff; }
.message.assistant { justify-items: stretch; }
.message.assistant .bubble {
  width: 100%;
  max-width: none;
  background: #fff;
  border: 1px solid var(--line);
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
.markdown-body th { background: var(--panel-soft); }
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
  background: #e8f5f3;
  color: var(--primary-dark);
  font-size: .9em;
  font-weight: 800;
  text-decoration: none;
  border: 1px solid rgba(15,118,110,.18);
}
.evidence-ref:hover { background: #d7f0eb; }
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
  font-weight: 800;
}
.meta-row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.badge {
  display: inline-flex;
  align-items: center;
  min-height: 24px;
  padding: 2px 8px;
  border-radius: 999px;
  background: #e8f5f3;
  color: var(--primary-dark);
  font-size: 12px;
  font-weight: 700;
}
.badge.secondary { background: #eef2ff; color: #3730a3; }
.source-grid { display: flex; flex-wrap: wrap; gap: 8px; }
.source-btn { padding: 7px 10px; font-size: 12px; }
.composer {
  border-top: 1px solid var(--line);
  padding: 14px 16px;
  background: #fff;
}
.composer textarea {
  width: 100%;
  min-height: 86px;
  resize: vertical;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px;
}
.composer-actions { margin-top: 10px; display: flex; justify-content: space-between; gap: 12px; align-items: center; }
.notice {
  padding: 10px 12px;
  margin: 0 16px 12px;
  border: 1px solid #fed7aa;
  background: var(--warn-bg);
  color: #9a3412;
  border-radius: 8px;
}
.evidence {
  padding: 16px;
  position: sticky;
  top: 80px;
  max-height: calc(100vh - 104px);
  overflow: auto;
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
  padding: 18px;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 10px;
  box-shadow: 0 24px 80px rgba(15, 23, 42, .22);
}
.modal-close {
  position: absolute;
  top: 12px;
  right: 12px;
  border: 1px solid var(--line);
  background: #fff;
  color: var(--ink);
  border-radius: 999px;
  width: 34px;
  height: 34px;
  cursor: pointer;
  font-weight: 800;
}
body.modal-open { overflow: hidden; }
.evidence-empty {
  min-height: 220px;
  display: grid;
  place-items: center;
  text-align: center;
  color: var(--muted);
  border: 1px dashed var(--line);
  border-radius: 8px;
  background: var(--panel-soft);
}
.evidence-title { font-size: 16px; font-weight: 800; margin: 0 0 8px; }
.kv { display: grid; grid-template-columns: 112px 1fr; gap: 8px; padding: 5px 0; border-bottom: 1px solid #edf1f5; }
.kv span:first-child { color: var(--muted); font-size: 12px; }
.chunk {
  margin-top: 14px;
  padding: 12px;
  background: var(--panel-soft);
  border: 1px solid var(--line);
  border-radius: 8px;
  line-height: 1.7;
}
.chunk h3 { font-size: 13px; margin: 0 0 8px; color: var(--muted); }
details { margin-top: 12px; }
summary { cursor: pointer; color: var(--primary-dark); font-weight: 700; }
.report-box {
  margin-top: 8px;
  padding: 10px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel-soft);
}
.report-box textarea { width: 100%; min-height: 70px; border: 1px solid var(--line); border-radius: 8px; padding: 8px; }
.report-status { color: var(--primary-dark); font-size: 13px; font-weight: 700; }
.error { color: var(--danger); }
.pending-answer .bubble { border-style: dashed; color: var(--muted); }
.loading-row { display: inline-flex; align-items: center; gap: 8px; font-weight: 700; }
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
.busy .primary-btn, .composer.is-submitting .primary-btn { opacity: .65; pointer-events: none; }
@media (max-width: 1180px) {
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
  .evidence { grid-column: 1 / -1; position: static; max-height: none; }
}
@media (max-width: 820px) {
  .topbar { height: auto; align-items: flex-start; padding: 14px; flex-direction: column; }
  .top-actions { width: 100%; justify-content: space-between; }
  .mobile-menu-btn { display: inline-flex; align-items: center; gap: 6px; }
  .workspace { width: calc(100vw - 20px); grid-template-columns: 1fr; margin-top: 10px; }
  .workspace.sidebar-expanded,
  .workspace.sidebar-pinned,
  .workspace:has(.settings:hover),
  .workspace:has(.settings:focus-within) {
    grid-template-columns: 1fr;
  }
  .workspace.evidence-popup-mode,
  .workspace.evidence-popup-mode.sidebar-expanded,
  .workspace.evidence-popup-mode.sidebar-pinned,
  .workspace.evidence-popup-mode:has(.settings:hover),
  .workspace.evidence-popup-mode:has(.settings:focus-within) {
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
    border-radius: 0 12px 12px 0;
    transform: translateX(-104%);
    transition: transform .22s ease;
    padding: 16px;
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
    border-radius: 12px 12px 0 0;
  }
  .messages { max-height: none; }
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
      <div class="bubble">
        <div class="loading-row">
          <span>送信しました。回答生成中です</span>
          <span class="dots" aria-hidden="true"><i></i><i></i><i></i></span>
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
      button.textContent = "回答生成中";
      button.disabled = true;
    }
    return;
  }
  form.classList.remove("is-submitting");
  if (button) {
    button.textContent = button.dataset.originalText || "質問する";
    button.disabled = false;
  }
}

function showSubmitError(message) {
  const pending = document.querySelector("[data-pending-answer]");
  if (!pending) return;
  pending.outerHTML = `
    <section class="message assistant">
      <div class="bubble error">送信に失敗しました。RAG APIの起動状態を確認してください。<br>${escapeHtml(message)}</div>
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

function bindFastRag() {
  bindSidebar();

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
        const html = await res.text();
        workspace.outerHTML = html;
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
      const panel = document.getElementById("evidence-panel");
      if (panel) panel.outerHTML = html;
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


def controls_html(api_url: str, selected: list[str], top_k: int, corpora: list[dict], form_id: str) -> str:
    items = []
    selected_set = set(selected)
    for corpus in corpora:
        corpus_id = str(corpus.get("corpus_id"))
        checked = "checked" if corpus_id in selected_set else ""
        items.append(
            f"""
            <label class="corpus-item">
              <input type="checkbox" name="corpus_ids" value="{esc(corpus_id)}" form="{esc(form_id)}" {checked}>
              <span>{esc(corpus.get("display_name"))}</span>
            </label>
            """
        )
    return f"""
    <aside id="settings-sidebar" class="settings card" tabindex="-1">
      <button type="button" class="settings-rail" data-sidebar-rail aria-label="検索設定を開く">
        <span class="rail-text">検索設定</span>
      </button>
      <div class="settings-body">
        <div class="settings-head">
          <h2>検索設定</h2>
          <button type="button" class="pin-btn" data-sidebar-pin aria-pressed="false">ピン留め</button>
        </div>
        <div class="field">
          <label>根拠数</label>
          <div class="range-row">
            <input type="range" name="top_k" min="3" max="15" value="{int(top_k)}" form="{esc(form_id)}" oninput="this.nextElementSibling.value=this.value">
            <output>{int(top_k)}</output>
          </div>
        </div>
        <div class="corpus-title">検索対象文書</div>
        <div class="corpus-list">{"".join(items)}</div>
        <p class="session">文書を絞ると回答キャッシュも選択Corpusに合わせて判定されます。</p>
      </div>
    </aside>
    <div class="sidebar-backdrop" data-sidebar-backdrop></div>
    """


def answer_badges(message: dict) -> str:
    if message.get("cache_hit"):
        sim = message.get("cache_similarity")
        sim_text = f" 類似度 {sim:.3f}" if isinstance(sim, (int, float)) else ""
        return f'<span class="badge">承認済みQAキャッシュ{esc(sim_text)}</span>'
    source = message.get("answer_source") or "rag"
    return f'<span class="badge secondary">回答元: {esc(source)}</span>'


def source_buttons(session_id: str, message_index: int, sources: list[dict]) -> str:
    if not sources:
        return ""
    buttons = []
    for source_index, source in enumerate(sources):
        label = f"根拠{source_index + 1}"
        title = f"{source_title(source)} / スコア {score_text(source)}"
        url = source_url(session_id, message_index, source_index)
        buttons.append(
            f'<button type="button" class="source-btn" data-evidence-url="{esc(url)}" title="{esc(title)}">{esc(label)}</button>'
        )
    return f'<div class="source-grid">{"".join(buttons)}</div>'


def report_box(session_id: str, message_index: int, api_url: str, message: dict) -> str:
    target_id = f"report-{message_index}"
    if message.get("reported"):
        return f'<div id="{target_id}" class="report-status">報告済みです。report_id={esc(message.get("report_id"))}</div>'
    if not message.get("log_id"):
        return f'<div id="{target_id}"></div>'
    return f"""
    <div id="{target_id}" class="report-box">
      <form action="/report" data-report-form data-report-target="{esc(target_id)}">
        <input type="hidden" name="session_id" value="{esc(session_id)}">
        <input type="hidden" name="message_index" value="{message_index}">
        <input type="hidden" name="api_url" value="{esc(api_url)}">
        <textarea name="comment" placeholder="ハルシネーション疑いの理由や補足を入力"></textarea>
        <div class="composer-actions">
          <span class="session">回答に誤りがある場合は報告できます</span>
          <button type="submit" class="secondary-btn">報告する</button>
        </div>
      </form>
    </div>
    """


def messages_html(session_id: str, api_url: str) -> str:
    messages = store.ensure(session_id)
    if not messages:
        return """
        <div class="empty">
          <div>
            <strong>質問を入力すると、回答と根拠がここに表示されます。</strong><br>
            <span>右ペインで参照チャンクを確認しながら読み進められます。</span>
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
              <div class="meta-row">{answer_badges(message)}</div>
              {source_buttons(session_id, idx, sources)}
              {report_box(session_id, idx, api_url, message)}
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


def evidence_panel(source: dict | None = None, source_index: int | None = None) -> str:
    if not source:
        return """
        <aside id="evidence-panel" class="evidence card">
          <h2>根拠詳細</h2>
          <div class="evidence-empty">回答後に根拠ボタンを押すと、ここに詳細が表示されます。</div>
        </aside>
        """
    metadata = [
        ("スコア", score_text(source)),
        ("参照元", source_file_name(source)),
        ("見出し", source.get("heading_path")),
        ("文書種別", source.get("document_type")),
        ("参照種別", source.get("source_type")),
    ]
    kvs = "".join(
        f'<div class="kv"><span>{esc(label)}</span><strong>{esc(value)}</strong></div>'
        for label, value in metadata
        if value not in (None, "")
    )
    parent = source.get("parent_text")
    parent_html = (
        f"<details><summary>回答生成に使った親チャンク全文</summary><div class=\"chunk markdown-body\">{markdown_block(parent)}</div></details>"
        if parent
        else ""
    )
    forms = source.get("forms") or []
    form_html = ""
    if forms:
        form_rows = "".join(
            f'<li>{esc(form.get("form_name"))} / {esc(form.get("file_path"))}</li>'
            for form in forms
        )
        form_html = f"<details><summary>関連様式</summary><ul>{form_rows}</ul></details>"
    number = "" if source_index is None else f"根拠{source_index + 1}: "
    return f"""
    <aside id="evidence-panel" class="evidence card">
      <h2>根拠詳細</h2>
      <p class="evidence-title">{esc(number)}{esc(source_title(source))}</p>
      {kvs}
      <div class="chunk">
        <h3>ヒットした子チャンク</h3>
        <div class="markdown-body">{markdown_block(source.get("child_text") or source.get("text") or "")}</div>
      </div>
      {parent_html}
      {form_html}
    </aside>
    """


def evidence_modal(source: dict | None = None, source_index: int | None = None) -> str:
    if not source:
        return """
        <div class="evidence-modal" role="dialog" aria-modal="true" aria-label="根拠詳細">
          <button type="button" class="evidence-modal-backdrop" data-modal-close aria-label="閉じる"></button>
          <section class="evidence-modal-card">
            <button type="button" class="modal-close" data-modal-close aria-label="閉じる">×</button>
            <h2>根拠詳細</h2>
            <div class="evidence-empty">根拠はありません。</div>
          </section>
        </div>
        """
    metadata = [
        ("スコア", score_text(source)),
        ("参照元", source_file_name(source)),
        ("見出し", source.get("heading_path")),
        ("文書種別", source.get("document_type")),
        ("参照種別", source.get("source_type")),
    ]
    kvs = "".join(
        f'<div class="kv"><span>{esc(label)}</span><strong>{esc(value)}</strong></div>'
        for label, value in metadata
        if value not in (None, "")
    )
    parent = source.get("parent_text")
    parent_html = (
        f'<details><summary>回答生成に使った親チャンク全文</summary><div class="chunk markdown-body">{markdown_block(parent)}</div></details>'
        if parent
        else ""
    )
    number = "" if source_index is None else f"根拠{source_index + 1}: "
    return f"""
    <div class="evidence-modal" role="dialog" aria-modal="true" aria-label="根拠詳細">
      <button type="button" class="evidence-modal-backdrop" data-modal-close aria-label="閉じる"></button>
      <section class="evidence-modal-card">
        <button type="button" class="modal-close" data-modal-close aria-label="閉じる">×</button>
        <h2>根拠詳細</h2>
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


def evidence_response(source: dict | None = None, source_index: int | None = None) -> str:
    return evidence_modal(source, source_index) if EVIDENCE_UI_MODE == "popup" else evidence_panel(source, source_index)


def workspace_html(
    *,
    session_id: str,
    api_url: str,
    selected: list[str],
    top_k: int,
    notice: str = "",
) -> str:
    corpora = load_corpora()
    source = latest_source(session_id)
    form_id = f"chat-form-{session_id}"
    mode_class = " evidence-popup-mode" if EVIDENCE_UI_MODE == "popup" else ""
    evidence_html = '<div id="evidence-modal-root" class="evidence-modal-root"></div>' if EVIDENCE_UI_MODE == "popup" else evidence_panel(source, 0 if source else None)
    return f"""
    <main id="workspace" class="workspace{mode_class}">
      {controls_html(api_url, selected, top_k, corpora, form_id)}
      <section class="chat card">
        <div class="chat-header">
          <strong>Meeting Timeline RAGチャット</strong>
          <span class="session">根拠付きで回答します</span>
        </div>
        {'<div class="notice">' + esc(notice) + '</div>' if notice else ''}
        <div class="messages">{messages_html(session_id, api_url)}</div>
        <form id="{esc(form_id)}" action="/ask" data-chat-form class="composer">
          <input type="hidden" name="session_id" value="{esc(session_id)}">
          <input type="hidden" name="api_url" value="{esc(api_url)}">
          <textarea name="question" placeholder="会議資料・議事録・時系列の経緯について質問してください"></textarea>
          <div class="composer-actions">
            <span class="session">選択Corpus: {len(selected)}件 / 根拠数: {int(top_k)}</span>
            <button type="submit" class="primary-btn">質問する</button>
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
      <title>Meeting Timeline RAG | SaaS UI</title>
      <style>{SAAS_CSS}</style>
    </head>
    <body>
      <div class="app-frame">
        <header class="topbar">
          <div class="brand">
            <strong>Meeting Timeline RAG Workspace</strong>
            <span>回答と根拠を横並びで確認する業務SaaS風UI</span>
          </div>
          <div class="top-actions">
            <button type="button" class="mobile-menu-btn" data-sidebar-mobile-toggle aria-controls="settings-sidebar" aria-expanded="false">メニュー</button>
            <a class="ghost-link" href="/">新しい会話</a>
          </div>
        </header>
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
    return HTMLResponse(f'<div id="{target_id}" class="report-status">報告を受け付けました。report_id={esc(report_id)}</div>')


async def health(_: Request) -> HTMLResponse:
    return HTMLResponse("ok")


rt("/", methods=["GET"])(home)
rt("/ask", methods=["POST"])(ask)
rt("/source", methods=["GET"])(source)
rt("/report", methods=["POST"])(report)
rt("/health", methods=["GET"])(health)


if __name__ == "__main__":
    serve()
