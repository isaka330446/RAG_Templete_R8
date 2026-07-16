from __future__ import annotations

import html
import json
import re
import time
import unicodedata
import uuid
from pathlib import Path
from urllib.parse import urlencode

import httpx
from api.config import (
    get_required_url_value,
    get_url_number,
    rag_api_base_url,
    rag_ask_url,
    rag_ask_stream_url,
    runtime_rag_api_base_url,
)
from api.text_cleaning import clean_display_text
from fasthtml.common import fast_app
from markdown_it import MarkdownIt
from starlette.requests import Request
from starlette.responses import HTMLResponse, StreamingResponse
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles


BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_SETTINGS = BASE_DIR / "config" / "corpus_settings.json"
DEFAULT_TOP_K = 8

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
    return MARKDOWN.render(clean_display_text(value))


def source_file_name(source: dict) -> str:
    value = str(source.get("source_file") or source.get("source_url") or "").strip()
    if not value:
        return ""
    if value.startswith(("http" + "://", "https" + "://")):
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
        source.get("document_id"),
        source.get("document_series"),
        source.get("document_code"),
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


def source_direct_reference_terms(source: dict) -> list[str]:
    terms = [
        source_title(source),
        source.get("heading_path"),
        source.get("title"),
        source.get("document_id"),
        source.get("document_series"),
        source_file_name(source),
    ]
    file_name = source_file_name(source)
    if file_name:
        terms.append(Path(file_name).stem)
    document_code = str(source.get("document_code") or "").strip()
    if document_code:
        terms.extend(
            [
                f"文書コード {document_code}",
                f"文書コード{document_code}",
                f"Document {document_code}",
                f"Doc.{document_code}",
                f"Doc{document_code}",
                f"Document ID {document_code}",
                f"No.{document_code}",
                f"No{document_code}",
            ]
        )
    normalized_terms = []
    seen = set()
    for term in terms:
        text = str(term or "").strip()
        norm = normalize_evidence_text(text)
        if not text or norm in seen:
            continue
        if len(norm) < 5 and not re.fullmatch(r"\d{4,}", norm):
            continue
        normalized_terms.append(text)
        seen.add(norm)
    return normalized_terms


def unique_source_reference_terms(sources: list[dict]) -> list[tuple[str, int]]:
    owners: dict[str, tuple[str, set[int]]] = {}
    for source_index, source in enumerate(sources):
        for term in source_direct_reference_terms(source):
            norm = normalize_evidence_text(term)
            if not norm:
                continue
            display, indices = owners.setdefault(norm, (term, set()))
            indices.add(source_index)
    unique_terms = [
        (display, next(iter(indices)))
        for display, indices in owners.values()
        if len(indices) == 1
    ]
    return sorted(unique_terms, key=lambda item: len(item[0]), reverse=True)


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

    def replace_direct_reference(term: str, source_index: int, value: str) -> str:
        pattern = re.compile(re.escape(term))

        def repl(match: re.Match[str]) -> str:
            return token_for(match.group(0), source_index + 1)

        return pattern.sub(repl, value)

    linked = re.sub(
        rf"(?:[［\[\(（【]\s*)?(?P<prefix>{EVIDENCE_PREFIX_PATTERN})\s*(?:(?:[:：#]|No\.?|№)\s*)?(?P<nums>{EVIDENCE_NUMBER_LIST_PATTERN})(?:\s*[］\]\)）】])?",
        replace_prefixed_group,
        text,
    )
    linked = re.sub(rf"[［\[\(（【](?P<num>{EVIDENCE_NUMBER_PATTERN})[］\]\)）】]", replace_bracket_number, linked)
    linked = re.sub(EVIDENCE_TITLE_PATTERN, replace_title_reference, linked)
    for term, source_index in unique_source_reference_terms(sources):
        linked = replace_direct_reference(term, source_index, linked)
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
    return runtime_rag_api_base_url(str(ask_url or "").strip() or None)


def api_stream_url(ask_url: str) -> str:
    return f"{api_base_url(ask_url)}/ask_stream"


def stream_json(event: dict) -> str:
    return json.dumps(event, ensure_ascii=False) + "\n"


def load_corpora() -> list[dict]:
    if not CORPUS_SETTINGS.exists():
        return []
    data = json.loads(CORPUS_SETTINGS.read_text(encoding="utf-8"))
    corpora = [c for c in data.get("corpora", []) if c.get("enabled", True)]
    return sorted(corpora, key=lambda item: item.get("priority", 999))


def source_title(source: dict) -> str:
    return clean_display_text(
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

    def clear(self, session_id: str) -> None:
        # ユーザー画面では会話履歴を保持せず、直近の質問と回答だけを
        # 根拠表示・報告操作のため一時的に保持する。
        self.sessions[session_id] = []

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
  --bg: #f7f9ff;
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
    radial-gradient(ellipse at 20% 0%, rgba(196, 214, 255, .55), transparent 42%),
    radial-gradient(ellipse at 86% 10%, rgba(229, 218, 255, .50), transparent 38%),
    linear-gradient(180deg, rgba(255,255,255,.82), rgba(238,244,255,.70) 54%, rgba(249,251,255,.92)),
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
.logo { display: flex; align-items: center; gap: 12px; font-weight: 800; color: inherit; text-decoration: none; }
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
  place-items: start center;
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
.composer-actions-top {
  max-width: min(980px, 100%);
  margin: 0 auto 8px;
  display: flex;
  justify-content: flex-end;
}
.composer-new-chat {
  width: 58px;
  height: 58px;
  min-width: 58px;
  min-height: 58px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0;
  padding: 0;
  border: 1px solid rgba(23,32,51,.08);
  border-radius: 999px;
  background: rgba(244, 242, 243, .92);
  color: #101828;
  text-decoration: none;
  font-size: 13px;
  font-weight: 900;
  box-shadow: 0 12px 28px rgba(15,23,42,.08);
}
.composer-new-chat .ms-icon { font-size: 22px; }
.composer-new-chat:hover {
  background: rgba(255,255,255,.96);
  color: var(--primary);
  transform: translateY(-1px);
}
.composer-box {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 10px;
  align-items: end;
  max-width: min(980px, 100%);
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
  width: min(1320px, calc(100vw - 48px));
  height: min(88vh, 980px);
  max-height: calc(100vh - 48px);
  overflow: hidden;
  display: flex;
  flex-direction: column;
  padding: 20px;
  background: rgba(255,255,255,.98);
  border: 1px solid var(--line);
  border-radius: 22px;
  box-shadow: 0 24px 80px rgba(15, 23, 42, .24);
}
.evidence-detail-scroll {
  overflow: auto;
  flex: 1 1 auto;
  min-height: 0;
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
.evidence-empty {
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
.approve-box {
  margin-top: 10px;
  padding: 10px 12px;
  border: 1px solid rgba(48,92,255,.14);
  border-radius: 14px;
  background: rgba(48,92,255,.06);
}
.approve-box form {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}
.answer-action.approve {
  background: #fff;
  border-color: rgba(48,92,255,.28);
  color: var(--primary-dark);
}
.approve-status {
  margin-top: 10px;
  color: #067647;
  font-size: 13px;
  font-weight: 800;
}
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
  --bg: #f7f9ff;
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
    radial-gradient(ellipse at 18% 0%, rgba(196, 214, 255, .58), transparent 42%),
    radial-gradient(ellipse at 86% 12%, rgba(229, 218, 255, .52), transparent 38%),
    linear-gradient(180deg, rgba(255,255,255,.84), rgba(238,244,255,.72) 56%, rgba(249,251,255,.94)),
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
.shell:has(.workspace.evidence-popup-mode) .nav,
.shell:has(.workspace.is-empty) .nav {
  display: none;
}
.workspace.is-empty {
  width: 100vw;
  margin: 0;
  grid-template-columns: 56px minmax(0, 1fr);
  gap: 22px;
}
.workspace.is-empty .settings {
  top: 0;
  min-height: 100vh;
  border-radius: 0 28px 28px 0;
  border-left: 0;
  border-top: 0;
  border-bottom: 0;
  border-color: rgba(255,255,255,.72);
  background: rgba(255,255,255,.72);
  box-shadow: 10px 0 42px rgba(15,23,42,.08);
  padding: 0;
}
.workspace.is-empty.sidebar-expanded .settings,
.workspace.is-empty.sidebar-pinned .settings,
.workspace.is-empty:has(.settings:hover) .settings,
.workspace.is-empty:has(.settings:focus-within) .settings {
  padding: 18px;
}
.workspace.is-empty .settings-rail {
  display: flex;
  align-items: stretch;
  justify-content: flex-start;
  width: 100%;
  min-height: 100vh;
  padding: 26px 7px 16px;
  border-radius: 0 28px 28px 0;
  background: transparent;
}
.workspace.is-empty.sidebar-expanded .settings-rail,
.workspace.is-empty.sidebar-pinned .settings-rail,
.workspace.is-empty:has(.settings:hover) .settings-rail,
.workspace.is-empty:has(.settings:focus-within) .settings-rail {
  display: none;
}
.workspace.is-empty.sidebar-expanded .settings-body,
.workspace.is-empty.sidebar-pinned .settings-body,
.workspace.is-empty:has(.settings:hover) .settings-body,
.workspace.is-empty:has(.settings:focus-within) .settings-body {
  display: block;
}
.rail-icons {
  width: 100%;
  display: grid;
  align-content: start;
  gap: 14px;
}
.rail-icon {
  width: 42px;
  height: 42px;
  border-radius: 16px;
  display: grid;
  place-items: center;
  color: #5b6475;
}
.rail-icon:hover {
  background: rgba(48,92,255,.10);
  color: var(--primary);
}
.workspace.is-empty .chat-panel {
  min-height: calc(100vh - 96px);
  border: 0;
  background: transparent;
  box-shadow: none;
  backdrop-filter: none;
}
.workspace.is-empty .workspace-head {
  padding: 24px clamp(24px, 5vw, 86px) 8px;
  border-bottom: 0;
}
.workspace.is-empty .workspace-head-main {
  align-items: start;
}
.workspace.is-empty .workspace-kicker {
  display: none;
}
.workspace.is-empty .workspace-brand-lockup {
  display: flex;
  align-items: center;
  gap: 18px;
}
.workspace.is-empty .workspace-brand-mark {
  width: 54px;
  height: 54px;
  border-radius: 18px;
  display: grid;
  place-items: center;
  color: #305cff;
  background: rgba(255,255,255,.48);
  box-shadow: inset 0 0 0 1px rgba(48,92,255,.18), 0 14px 34px rgba(48,92,255,.12);
}
.workspace.is-empty .workspace-brand-mark .ms-icon {
  font-size: 34px;
}
.workspace.is-empty .workspace-title {
  margin: 0;
  font-size: clamp(28px, 3vw, 44px);
  line-height: 1.08;
  font-weight: 900;
  color: #172033;
}
.workspace.is-empty .workspace-title .brand-accent {
  background: linear-gradient(90deg, #1f5eff, #8a4bd8);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}
.workspace.is-empty .workspace-copy {
  margin-top: 8px;
  color: #172033;
  font-size: clamp(16px, 1.45vw, 24px);
  font-weight: 800;
}
.workspace.is-empty .dataset-badges {
  align-self: start;
  padding-top: 8px;
}
.workspace.is-empty .dataset-badge {
  min-height: 44px;
  padding: 9px 16px;
  border: 1px solid rgba(113, 128, 150, .22);
  border-radius: 999px;
  background: rgba(255,255,255,.66);
  box-shadow: 0 10px 24px rgba(15,23,42,.05);
}
.workspace.is-empty .messages {
  max-height: none;
  overflow: visible;
  padding: 34px clamp(24px, 6vw, 118px) 56px;
}
.empty {
  min-height: 560px;
  display: block;
  text-align: center;
  color: var(--text);
}
.empty-hero {
  width: min(1450px, 100%);
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 32px;
  align-items: center;
  justify-content: center;
}
.empty-hero-main {
  width: min(1050px, 100%);
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
}
.empty-eyebrow { display: none; }
.empty h1 {
  margin: 0 0 12px;
  font-size: clamp(30px, 3vw, 50px);
  line-height: 1.18;
  letter-spacing: 0;
  font-weight: 900;
  color: #172033;
}
.empty p {
  max-width: 760px;
  margin: 0 auto;
  color: #303847;
  line-height: 1.8;
  font-size: clamp(14px, 1.1vw, 18px);
  font-weight: 700;
}
.template-grid {
  width: min(1280px, 100%);
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 16px;
}
.template-card {
  border: 1px solid var(--line);
  border-radius: 20px;
  background: rgba(255,255,255,.78);
  box-shadow: 0 14px 32px rgba(15,23,42,.10);
  padding: 18px 18px;
  min-height: 78px;
  text-align: left;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  cursor: pointer;
  color: var(--text);
  font-size: 15px;
  font-weight: 850;
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
  border: 0;
  border-radius: 0;
  background: transparent;
  box-shadow: none;
  padding: 0;
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
.composer-actions-top {
  max-width: min(980px, 100%);
  margin: 0 auto 8px;
}
.composer-box {
  grid-template-columns: 1fr auto;
  max-width: min(980px, 100%);
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
.modal-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 12px;
  margin-bottom: 12px;
}
.evidence-empty { border-radius: 20px; }
.evidence-modal {
  place-items: center;
  padding: 24px;
}
.evidence-modal-card {
  width: min(1320px, calc(100vw - 48px));
  height: min(88vh, 980px);
  max-height: calc(100vh - 48px);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  border-radius: 28px;
}
.evidence-detail-scroll {
  overflow: auto;
  flex: 1 1 auto;
  min-height: 0;
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
.loading-panel {
  position: relative;
  overflow: hidden;
  padding: 14px;
  border-radius: 10px;
  background: rgba(255,255,255,.86);
  border: 1px solid rgba(48,92,255,.14);
}
.loading-panel::before {
  content: "";
  position: absolute;
  left: 14px;
  right: 14px;
  top: 0;
  height: 3px;
  border-radius: 999px;
  background:
    linear-gradient(90deg, var(--primary), #00b8a9) 0 0 / calc(42% + var(--loading-track-x, 0%)) 100% no-repeat,
    rgba(37,99,235,.12);
}
.loading-panel::after {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(105deg, transparent 0%, rgba(13,148,136,.10) 42%, rgba(37,99,235,.12) 50%, transparent 58%);
  transform: translateX(-100%);
  animation: loadingSweep 1.8s infinite ease-in-out;
  pointer-events: none;
}
.loading-steps {
  display: grid;
  gap: 8px;
  margin-top: 10px;
}
.loading-step {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--muted);
  font-size: 13px;
  font-weight: 700;
  animation: loadingStepFocus 4.8s infinite ease-in-out;
}
.loading-step .step-dot {
  width: 8px;
  height: 8px;
  border-radius: 999px;
  background: var(--primary);
  animation: dotPulse 1.1s infinite ease-in-out;
}
.loading-step:nth-child(1) { animation-delay: 0s; }
.loading-step:nth-child(2) { animation-delay: 1.2s; }
.loading-step:nth-child(3) { animation-delay: 2.4s; }
.loading-step:nth-child(4) { animation-delay: 3.6s; }
.loading-step:nth-child(2) .step-dot { animation-delay: .16s; }
.loading-step:nth-child(3) .step-dot { animation-delay: .32s; }
.loading-step:nth-child(4) .step-dot { animation-delay: .48s; }
.loading-current {
  position: relative;
  margin-top: 10px;
  padding: 9px 11px;
  border-radius: 999px;
  background: linear-gradient(90deg, rgba(48,92,255,.10), rgba(0,184,169,.12));
  color: var(--primary-dark);
  font-size: 13px;
  font-weight: 900;
  z-index: 1;
}
.loading-track {
  position: relative;
  height: 3px;
  margin-top: 10px;
  overflow: hidden;
  border-radius: 999px;
  background: rgba(37,99,235,.12);
  z-index: 1;
}
.loading-track::after {
  content: "";
  position: absolute;
  inset: 0;
  width: 42%;
  border-radius: inherit;
  background: linear-gradient(90deg, var(--primary), #00b8a9);
  transform: translateX(var(--loading-track-x, 0%));
  transition: transform .22s ease;
}
.loading-step {
  animation: none;
  opacity: .48;
  transition: color .18s ease, opacity .18s ease, transform .18s ease;
}
.loading-step.is-active {
  color: var(--primary-dark);
  opacity: 1;
  transform: translateX(5px);
}
.loading-step.is-complete {
  color: #067647;
  opacity: .82;
}
@keyframes loadingSweep {
  0% { transform: translateX(-100%); opacity: 0; }
  20%, 70% { opacity: 1; }
  100% { transform: translateX(100%); opacity: 0; }
}
@keyframes loadingStepFocus {
  0%, 20%, 100% { color: var(--muted); transform: translateX(0); opacity: .48; }
  8%, 14% { color: var(--primary-dark); transform: translateX(5px); opacity: 1; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: .01ms !important;
    animation-iteration-count: 1 !important;
    scroll-behavior: auto !important;
    transition-duration: .01ms !important;
  }
}
.pending-answer .dots i {
  animation-name: dotPulse !important;
  animation-duration: 1.1s !important;
  animation-timing-function: ease-in-out !important;
  animation-iteration-count: infinite !important;
}
.pending-answer .loading-step .step-dot {
  animation: none !important;
  opacity: .56;
  transform: translateY(0);
}
.pending-answer .loading-step.is-active .step-dot {
  animation-name: dotPulse !important;
  animation-duration: 1.1s !important;
  animation-timing-function: ease-in-out !important;
  animation-iteration-count: infinite !important;
  animation-delay: 0s !important;
}
.pending-answer .dots i:nth-child(2) { animation-delay: .16s !important; }
.pending-answer .dots i:nth-child(3) { animation-delay: .32s !important; }
.pending-answer .loading-panel::before {
  display: none;
  animation: none !important;
}
.pending-answer .loading-track::after {
  animation: loadingTrackRunner 1.45s ease-in-out infinite !important;
}
@keyframes loadingBarShimmer {
  from { background-position: -140px 0, 0 0, 0 0; }
  to { background-position: 100% 0, 0 0, 0 0; }
}
@keyframes loadingTrackRunner {
  0% { transform: translateX(calc(var(--loading-track-x, 0%) - 50%)); opacity: .34; }
  45% { opacity: 1; }
  100% { transform: translateX(calc(var(--loading-track-x, 0%) + 120%)); opacity: .34; }
}
.composer {
  border-top: 1px solid var(--line);
  padding: 16px;
  background: rgba(255,255,255,.84);
}
.composer-initial {
  width: 100%;
  max-width: 1040px;
  margin: 28px auto 0;
  padding: 0;
  border-top: 0;
  background: transparent;
}
.composer-docked {
  padding: 16px clamp(18px, 3vw, 42px) 22px;
  border-top: 0;
  border-radius: 0 0 26px 26px;
  background: transparent;
}
.hero-composer-row {
  width: min(1040px, 100%);
  margin: 0 auto;
  display: flex;
  align-items: stretch;
  justify-content: center;
  gap: 12px;
}
.chat-composer {
  flex: 1 1 auto;
  display: flex;
  align-items: stretch;
  gap: 10px;
}
.composer-initial .hero-composer-row {
  display: flex;
  align-items: center;
}
.composer-docked .hero-composer-row {
  align-items: center;
}
.composer-initial .chat-composer,
.composer-docked .chat-composer {
  min-height: 92px;
  align-items: center;
  gap: 12px;
  padding: 13px 14px 13px 28px;
  border: 2px solid rgba(48,92,255,.46);
  border-radius: 999px;
  background: rgba(255,255,255,.84);
  box-shadow: 0 22px 54px rgba(40, 68, 145, .16);
}
.composer-initial .chat-input,
.composer-docked .chat-input {
  min-height: 58px;
  padding: 16px 0;
  border: 0;
  background: transparent;
  font-size: 17px;
}
.composer-initial .chat-input:focus,
.composer-docked .chat-input:focus {
  outline: none;
}
.composer-initial .send-btn,
.composer-docked .send-btn {
  width: 58px;
  height: 58px;
  min-width: 58px;
  border-radius: 50%;
  padding: 0;
  display: inline-grid;
  place-items: center;
  box-shadow: 0 12px 30px rgba(48,92,255,.26);
}
.composer-initial .send-btn .send-text,
.composer-docked .send-btn .send-text {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
}
.composer-initial .composer-help,
.composer-docked .composer-help {
  display: none;
}
.chat-input {
  flex: 1 1 auto;
}
.composer-new-chat {
  flex: 0 0 58px;
  width: 58px;
  height: 58px;
  min-width: 58px;
  min-height: 58px;
  align-self: center;
  display: inline-grid;
  place-items: center;
  padding: 0;
  border-radius: 50%;
  white-space: nowrap;
}
.composer-new-chat .ms-icon { font-size: 22px; line-height: 1; }
.composer-help {
  width: min(920px, 100%);
  margin: 10px auto 0;
  justify-content: center;
}
.workspace.evidence-popup-mode {
  width: 100vw;
  margin: 0;
  grid-template-columns: 56px minmax(0, 1fr);
  gap: 22px;
}
.workspace.evidence-popup-mode.sidebar-expanded,
.workspace.evidence-popup-mode.sidebar-pinned,
.workspace.evidence-popup-mode:has(.settings:hover),
.workspace.evidence-popup-mode:has(.settings:focus-within) {
  grid-template-columns: minmax(220px, 240px) minmax(0, 1fr);
}
.workspace.evidence-popup-mode .settings {
  top: 0;
  min-height: 100vh;
  border-radius: 0 28px 28px 0;
  border-left: 0;
  border-top: 0;
  border-bottom: 0;
  border-color: rgba(255,255,255,.72);
  background: rgba(255,255,255,.72);
  box-shadow: 10px 0 42px rgba(15,23,42,.08);
  padding: 0;
}
.workspace.evidence-popup-mode.sidebar-expanded .settings,
.workspace.evidence-popup-mode.sidebar-pinned .settings,
.workspace.evidence-popup-mode:has(.settings:hover) .settings,
.workspace.evidence-popup-mode:has(.settings:focus-within) .settings {
  padding: 18px;
}
.workspace.evidence-popup-mode .settings-rail {
  display: flex;
  align-items: stretch;
  justify-content: flex-start;
  width: 100%;
  min-height: 100vh;
  padding: 26px 7px 16px;
  border-radius: 0 28px 28px 0;
  background: transparent;
}
.workspace.evidence-popup-mode.sidebar-expanded .settings-rail,
.workspace.evidence-popup-mode.sidebar-pinned .settings-rail,
.workspace.evidence-popup-mode:has(.settings:hover) .settings-rail,
.workspace.evidence-popup-mode:has(.settings:focus-within) .settings-rail {
  display: none;
}
.workspace.evidence-popup-mode .chat-panel {
  min-height: calc(100vh - 96px);
  border: 0;
  background: transparent;
  box-shadow: none;
  backdrop-filter: none;
  overflow: visible;
}
.workspace.evidence-popup-mode .workspace-head {
  padding: 24px clamp(24px, 5vw, 86px) 8px;
  border-bottom: 0;
  background: transparent;
}
.workspace.evidence-popup-mode .workspace-head-main {
  align-items: start;
}
.workspace.evidence-popup-mode .workspace-kicker {
  display: none;
}
.workspace.evidence-popup-mode .workspace-brand-lockup {
  display: flex;
  align-items: center;
  gap: 18px;
}
.workspace.evidence-popup-mode .workspace-brand-mark {
  width: 54px;
  height: 54px;
  border-radius: 18px;
  display: grid;
  place-items: center;
  color: #305cff;
  background: rgba(255,255,255,.48);
  box-shadow: inset 0 0 0 1px rgba(48,92,255,.18), 0 14px 34px rgba(48,92,255,.12);
}
.workspace.evidence-popup-mode .workspace-brand-mark .ms-icon {
  font-size: 34px;
}
.workspace.evidence-popup-mode .workspace-title {
  margin: 0;
  font-size: clamp(28px, 3vw, 44px);
  line-height: 1.08;
  font-weight: 900;
  color: #172033;
}
.workspace.evidence-popup-mode .workspace-title .brand-accent {
  background: linear-gradient(90deg, #1f5eff, #8a4bd8);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}
.workspace.evidence-popup-mode .workspace-copy {
  margin-top: 8px;
  color: #172033;
  font-size: clamp(16px, 1.45vw, 24px);
  font-weight: 800;
}
.workspace.evidence-popup-mode .dataset-badges {
  align-self: start;
  padding-top: 8px;
}
.workspace.evidence-popup-mode .dataset-badge {
  min-height: 44px;
  padding: 9px 16px;
  border: 1px solid rgba(113, 128, 150, .22);
  border-radius: 999px;
  background: rgba(255,255,255,.66);
  box-shadow: 0 10px 24px rgba(15,23,42,.05);
}
.workspace.evidence-popup-mode .messages {
  max-height: none;
  overflow: visible;
  padding: 26px clamp(24px, 6vw, 118px) 120px;
}
.workspace.evidence-popup-mode.is-empty .messages {
  padding: 34px clamp(24px, 6vw, 118px) 56px;
}
@media (max-width: 1060px) {
  .template-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
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
  .hero-composer-row,
  .chat-composer { flex-direction: column; }
  .workspace.is-empty { width: calc(100vw - 14px); margin: 0 auto; grid-template-columns: 1fr; }
  .workspace.is-empty .messages { padding: 18px; }
  .workspace.is-empty .workspace-head { padding: 18px; }
  .workspace.is-empty .workspace-brand-lockup { justify-content: center; text-align: left; }
  .workspace.is-empty .workspace-brand-mark { display: none; }
  .template-grid { grid-template-columns: 1fr; }
  .composer-new-chat { align-self: center; }
  .composer-initial .chat-composer,
  .composer-docked .chat-composer {
    min-height: auto;
    border-radius: 28px;
    padding: 14px;
  }
  .composer-initial .chat-input,
  .composer-docked .chat-input {
    width: 100%;
    min-height: 92px;
  }
  .composer-initial .send-btn,
  .composer-docked .send-btn {
    width: 100%;
    height: 50px;
    border-radius: 18px;
  }
  .composer-initial .send-btn .send-text,
  .composer-docked .send-btn .send-text {
    position: static;
    width: auto;
    height: auto;
    overflow: visible;
    clip: auto;
  }
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

const STREAM_STAGE_MAP = {
  accepted: { index: 0, message: "質問を受け付けました" },
  cache_lookup: { index: 0, message: "承認済みQAを確認しています" },
  cache_hit: { index: 3, message: "承認済みQAから回答を準備しています" },
  retrieval: { index: 1, message: "関連文書を探しています" },
  rerank: { index: 1, message: "検索結果を並べ替えています" },
  context: { index: 2, message: "根拠を確認しています" },
  llm: { index: 3, message: "回答を作成しています" },
  logging: { index: 3, message: "回答ログを保存しています" },
  done: { index: 3, message: "回答が完了しました" }
};

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
      <div class="bubble loading-panel" data-loading-panel>
        <div class="loading-row">
          <span>根拠付きで確認しています</span>
          <span class="dots" aria-hidden="true"><i></i><i></i><i></i></span>
        </div>
        <div class="loading-current" data-loading-current>質問を受け付けました</div>
        <div class="loading-track" data-loading-track></div>
        <div class="loading-steps" aria-label="回答生成の進捗">
          <div class="loading-step is-active"><span class="step-dot"></span><span>質問を解析しています</span></div>
          <div class="loading-step"><span class="step-dot"></span><span>関連文書を探しています</span></div>
          <div class="loading-step"><span class="step-dot"></span><span>根拠を確認しています</span></div>
          <div class="loading-step"><span class="step-dot"></span><span>回答を作成しています</span></div>
        </div>
      </div>
    </section>
  `);
  messages.scrollTop = messages.scrollHeight;
}

function updateLoadingStage(stage, message) {
  const panel = document.querySelector("[data-loading-panel]");
  if (!panel) return;
  const info = STREAM_STAGE_MAP[stage] || { index: 0, message: message || "処理しています" };
  const title = panel.querySelector(".loading-row span");
  if (title) title.textContent = message || info.message;
  const current = panel.querySelector("[data-loading-current]");
  if (current) current.textContent = message || info.message;
  const steps = Array.from(panel.querySelectorAll(".loading-step"));
  steps.forEach((step, index) => {
    step.classList.toggle("is-active", index === info.index);
    step.classList.toggle("is-complete", index < info.index);
  });
  panel.style.setProperty("--loading-track-x", `${Math.min(58, Math.max(0, info.index * 19))}%`);
}

async function consumeChatStream(response, workspace) {
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const contentType = response.headers.get("content-type") || "";
  if (!response.body || !contentType.includes("application/x-ndjson")) {
    workspace.outerHTML = await response.text();
    return;
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (!line.trim()) continue;
      const event = JSON.parse(line);
      if (event.type === "progress") {
        updateLoadingStage(event.stage, event.message);
      } else if (event.type === "done") {
        updateLoadingStage("done", event.message);
        if (event.html) workspace.outerHTML = event.html;
        return;
      } else if (event.type === "error") {
        throw new Error(event.message || "RAG API stream failed");
      }
    }
  }
  if (buffer.trim()) {
    const event = JSON.parse(buffer);
    if (event.type === "done" && event.html) workspace.outerHTML = event.html;
    if (event.type === "error") throw new Error(event.message || "RAG API stream failed");
  }
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

function controlsForChatForm(form, selector) {
  if (!form || !form.id) return [];
  return Array.from(document.querySelectorAll(selector)).filter((control) => control.dataset.chatFormId === form.id);
}

function syncChatSettings(form) {
  if (!form) return;
  const corpusHolder = form.querySelector("[data-form-corpus-hidden]");
  const corpusControls = controlsForChatForm(form, 'input[name="corpus_ids"]');
  if (corpusHolder && corpusControls.length) {
    while (corpusHolder.firstChild) corpusHolder.removeChild(corpusHolder.firstChild);
    corpusControls.filter((control) => control.checked).forEach((control) => {
      const hidden = document.createElement("input");
      hidden.type = "hidden";
      hidden.name = "corpus_ids";
      hidden.value = control.value;
      corpusHolder.appendChild(hidden);
    });
  }
  const topKHidden = form.querySelector("[data-form-top-k-hidden]");
  const topKControl = controlsForChatForm(form, 'input[name="top_k"]')[0];
  if (topKHidden && topKControl) topKHidden.value = topKControl.value;
}

function bindChatSettingSync() {
  document.querySelectorAll("[data-chat-form-id]").forEach((control) => {
    if (control.dataset.settingBound === "1") return;
    control.dataset.settingBound = "1";
    const handler = () => {
      const form = document.getElementById(control.dataset.chatFormId);
      syncChatSettings(form);
    };
    control.addEventListener("change", handler);
    control.addEventListener("input", handler);
  });
  document.querySelectorAll("[data-chat-form]").forEach(syncChatSettings);
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
  bindChatSettingSync();
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
      syncChatSettings(form);
      setSubmitting(form, true);
      showPendingMessage(form);
      const workspace = document.getElementById("workspace");
      try {
        const res = await fetch(form.action, { method: "POST", body: new FormData(form) });
        await consumeChatStream(res, workspace);
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

  document.querySelectorAll("[data-approve-form]").forEach((form) => {
    if (form.dataset.bound === "1") return;
    form.dataset.bound = "1";
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const target = document.getElementById(form.dataset.approveTarget);
      const button = form.querySelector('button[type="submit"]');
      if (button) {
        button.disabled = true;
        button.dataset.originalText = button.textContent;
        button.textContent = "登録しています";
      }
      try {
        const res = await fetch(form.action, { method: "POST", body: new FormData(form) });
        target.outerHTML = await res.text();
      } catch (err) {
        if (target) target.outerHTML = `<div id="${escapeHtml(form.dataset.approveTarget)}" class="error">登録に失敗しました。${escapeHtml(String(err))}</div>`;
      } finally {
        if (button) {
          button.disabled = false;
          button.textContent = button.dataset.originalText || "この回答を承認済みQAに登録";
        }
      }
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


def unique_selected(values: list[object]) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for value in values:
        corpus_id = str(value or "").strip()
        if not corpus_id or corpus_id in seen:
            continue
        selected.append(corpus_id)
        seen.add(corpus_id)
    return selected


def selected_from_form(form) -> list[str]:
    selected = unique_selected(list(form.getlist("corpus_ids")))
    if not selected and str(form.get("corpus_selection_rendered") or "") != "1":
        return selected_default(load_corpora())
    return selected


def toolbar_html(api_url: str, selected: list[str], top_k: int, corpora: list[dict], form_id: str, session_id: str) -> str:
    selected_set = set(selected)
    corpus_items = []
    for corpus in corpora:
        corpus_id = str(corpus.get("corpus_id"))
        checked = "checked" if corpus_id in selected_set else ""
        corpus_items.append(
            f"""
            <label class="sidebar-corpus">
              <input type="checkbox" name="corpus_ids" value="{esc(corpus_id)}" data-chat-form-id="{esc(form_id)}" {checked}>
              <span>{esc(corpus.get("display_name"))}</span>
            </label>
            """
        )
    dock_items = [
        ("edit", "新しいチャット", "/"),
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
    rail_html = "".join(
        f'<span class="rail-icon" title="{esc(label)}" aria-hidden="true">{ms_icon(icon)}</span>'
        for icon, label, _ in dock_items
    )
    return f"""
    <aside id="settings-sidebar" class="settings" tabindex="-1">
      <button type="button" class="settings-rail" data-sidebar-rail aria-label="メニューを開く">
        <span class="rail-icons">{rail_html}</span>
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
            <input type="range" name="top_k" min="3" max="15" value="{int(top_k)}" data-chat-form-id="{esc(form_id)}" oninput="this.nextElementSibling.value=this.value">
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
    answer_source = "RAG検索"
    return f'<span class="badge">{ms_icon("check", "small")}回答元: {esc(answer_source)}</span>'


def source_chips(session_id: str, message_index: int, sources: list[dict]) -> str:
    if not sources:
        return """
        <div class="no-hit-panel">
          <h3>十分な根拠が見つかりませんでした</h3>
          <p>対象文書が未登録、別表現で記載、または質問が広すぎる可能性があります。</p>
          <div class="no-hit-actions">
            <button type="button" class="answer-action" data-fill-question="もう少し具体的に、文書名・条件・対象を入れて確認したい">
              <span class="ms-icon" aria-hidden="true">edit</span>質問を言い換える
            </button>
            <button type="button" class="answer-action" data-fill-question="この内容について関連する登録文書を探して">
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


def approved_qa_box(session_id: str, message_index: int, api_url: str, message: dict) -> str:
    target_id = f"approve-qa-{message_index}"
    if message.get("cache_hit"):
        return f'<div id="{target_id}" class="approve-status">{ms_icon("bookmark", "small")}承認済みQAから回答しています。</div>'
    if message.get("qa_registered"):
        return f'<div id="{target_id}" class="approve-status">{ms_icon("check", "small")}承認済みQAに登録しました。QA ID: {esc(message.get("qa_id"))}</div>'
    if not message.get("question") or not message.get("content") or not message.get("sources"):
        return f'<div id="{target_id}"></div>'
    return f"""
    <div id="{target_id}" class="approve-box">
      <form action="/approve_qa" data-approve-form data-approve-target="{esc(target_id)}">
        <input type="hidden" name="session_id" value="{esc(session_id)}">
        <input type="hidden" name="message_index" value="{message_index}">
        <input type="hidden" name="api_url" value="{esc(api_url)}">
        <button type="submit" class="answer-action approve">{ms_icon("bookmark", "small")}この回答を承認済みQAに登録</button>
        <span class="answer-note">次回から同じ質問に近い内容は、LLMを使わず根拠付きで回答できます。</span>
      </form>
    </div>
    """


def messages_html(session_id: str, api_url: str) -> str:
    messages = store.ensure(session_id)
    if not messages:
        return """
        <div class="empty">
          <div>
            <strong>業務文書を、会話で探す。</strong>
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
              {approved_qa_box(session_id, idx, api_url, message)}
              {report_box(session_id, idx, api_url, message)}
            </section>
            """
        )
    return "".join(rows)


QUESTION_TEMPLATE_CACHE = {"expires_at": 0.0, "items": []}


def latest_qa_question_templates() -> list[str]:
    now = time.monotonic()
    cached = QUESTION_TEMPLATE_CACHE.get("items") or []
    if cached and float(QUESTION_TEMPLATE_CACHE.get("expires_at") or 0.0) > now:
        return list(cached)
    questions: list[str] = []
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(
                f"{rag_api_base_url().rstrip('/')}/admin/qa-cache",
                params={"status": "approved", "limit": 5},
            )
            response.raise_for_status()
            data = response.json()
        rows = data.get("items") if isinstance(data, dict) else data
        for item in rows or []:
            question = str((item or {}).get("question") or "").strip()
            if question and question not in questions:
                questions.append(question)
            if len(questions) >= 5:
                break
    except Exception:
        questions = []
    QUESTION_TEMPLATE_CACHE["items"] = list(questions)
    QUESTION_TEMPLATE_CACHE["expires_at"] = now + 30.0
    return questions


def question_templates_html() -> str:
    templates = latest_qa_question_templates() or [
        "登録済み文書に根拠がない質問への対応を教えて",
        "新しいMarkdown文書を追加した後の手順を教えて",
        "承認済みQAを登録する流れを教えて",
        "重要判断で根拠を確認する方法を教えて",
        "対象外質問への答え方を根拠付きで確認して",
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


def composer_html(form_id: str, session_id: str, selected: list[str], top_k: int, *, initial: bool = False) -> str:
    class_name = "composer composer-initial" if initial else "composer composer-docked"
    placeholder = "ここに業務文書に関する質問を入力してください（例：文書更新後の手順は？）" if initial else "登録済み文書について質問してください"
    send_text = '<span class="send-text">根拠付きで確認</span>'
    corpus_inputs = "".join(
        f'<input type="hidden" name="corpus_ids" value="{esc(corpus_id)}">'
        for corpus_id in unique_selected(list(selected))
    )
    new_chat_html = ""
    return f"""
    <form id="{esc(form_id)}" action="/ask_stream" data-chat-form class="{class_name}">
      <input type="hidden" name="session_id" value="{esc(session_id)}">
      <input type="hidden" name="corpus_selection_rendered" value="1">
      <input type="hidden" name="top_k" value="{int(top_k)}" data-form-top-k-hidden>
      <span data-form-corpus-hidden hidden>{corpus_inputs}</span>
      <div class="hero-composer-row">
        {new_chat_html}
        <div class="chat-composer">
          <textarea class="chat-input" name="question" placeholder="{esc(placeholder)}"></textarea>
          <button type="submit" class="send-btn" aria-label="根拠付きで確認">{ms_icon("send", "small")}{send_text}</button>
        </div>
      </div>
      <div class="composer-help">{ms_icon("info", "small")}Enterで質問、Shift+Enterで改行。回答後は根拠チップから本文を確認できます。</div>
    </form>
    """


def messages_html_refreshed(session_id: str, api_url: str, empty_composer_html: str = "") -> str:
    messages = store.ensure(session_id)
    if not messages:
        return f"""
        <div class="empty">
          <div class="empty-hero">
            <div class="empty-hero-main">
              <div class="empty-eyebrow">Evidence Workspace</div>
              <h1>今日は何を確認しますか？</h1>
              <p>登録済みの業務文書を横断的に、判断材料と根拠本文を確認できます。</p>
              {empty_composer_html}
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
                {approved_qa_box(session_id, idx, api_url, message)}
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


def evidence_detail_body(source: dict | None = None, source_index: int | None = None) -> str:
    if not source:
        return """
        <div class="modal-head">
          <div>
            <h2>根拠詳細</h2>
            <p class="answer-note">根拠はありません。</p>
          </div>
        </div>
        <div class="evidence-empty">回答に紐づく根拠が見つかりませんでした。</div>
        """
    label, label_class = match_label(source)
    number = "" if source_index is None else f"根拠{source_index + 1}: "
    visible_meta = [
        ("一致度", label),
        ("参照元", source_file_name(source)),
        ("資料名", source.get("title")),
        ("見出し", source.get("heading_path")),
        ("文書コード", source.get("document_code")),
        ("文書種別", source.get("source_type")),
        ("セクション", source.get("section_title")),
    ]
    kvs = "".join(
        f'<div class="kv"><span>{esc(key)}</span><strong>{esc(clean_display_text(value))}</strong></div>'
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
        f'<div class="kv"><span>{esc(key)}</span><strong>{esc(clean_display_text(value))}</strong></div>'
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
    <div class="modal-head">
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


def evidence_modal_refreshed(source: dict | None = None, source_index: int | None = None) -> str:
    return f"""
    <div class="evidence-modal" role="dialog" aria-modal="true" aria-label="根拠詳細">
      <button type="button" class="evidence-modal-backdrop" data-modal-close aria-label="閉じる"></button>
      <section class="evidence-modal-card">
        <button type="button" class="modal-close" data-modal-close aria-label="閉じる">{ms_icon("close")}</button>
        <div class="evidence-detail-scroll">{evidence_detail_body(source, source_index)}</div>
      </section>
    </div>
    """


def evidence_response(source: dict | None = None, source_index: int | None = None) -> str:
    return evidence_modal_refreshed(source, source_index)


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
    mode_class = " evidence-popup-mode"
    evidence_html = '<div id="evidence-modal-root" class="evidence-modal-root"></div>'
    selected_labels = [
        str(corpus.get("display_name") or corpus.get("corpus_id"))
        for corpus in corpora
        if str(corpus.get("corpus_id")) in set(selected)
    ]
    if selected_labels and len(selected_labels) == len(corpora):
        dataset_label = "全資料"
    else:
        dataset_label = "、".join(selected_labels[:2]) + (" ほか" if len(selected_labels) > 2 else "")
    is_empty = not store.ensure(session_id)
    empty_composer = composer_html(form_id, session_id, selected, top_k, initial=True) if is_empty else ""
    docked_composer = "" if is_empty else composer_html(form_id, session_id, selected, top_k, initial=False)
    empty_class = " is-empty" if is_empty else ""
    return f"""
    <main id="workspace" class="workspace{mode_class}{empty_class}">
      {toolbar_html(api_url, selected, top_k, corpora, form_id, session_id)}
      <section class="chat-panel">
        <header class="workspace-head">
          <div class="workspace-head-main">
            <div>
              <div class="workspace-kicker">RAG Search Console</div>
              <div class="workspace-brand-lockup">
                <span class="workspace-brand-mark">{ms_icon("sync")}</span>
                <div>
                  <h1 class="workspace-title">RAG <span class="brand-accent">Workspace</span></h1>
                  <p class="workspace-copy">業務文書を根拠付きで探索するAIチャット</p>
                </div>
              </div>
            </div>
            <div class="dataset-badges" aria-label="現在の検索対象">
              <span class="dataset-badge">{ms_icon("folder", "small")}対象資料 | {esc(dataset_label or "未選択")}</span>
              <span class="dataset-badge">{ms_icon("article", "small")}根拠 | {int(top_k)}件</span>
            </div>
          </div>
        </header>
        {'<div class="notice">' + esc(notice) + '</div>' if notice else ''}
        <div id="chat-history" class="messages">{messages_html_refreshed(session_id, api_url, empty_composer)}</div>
        {docked_composer}
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
      <title>RAG Template | Modern Chat</title>
      <style>{MODERN_CSS}{UX_REFRESH_CSS}</style>
    </head>
    <body>
      <div class="shell">
        <nav class="nav">
          <div>
            <a class="logo" href="/" aria-label="トップへ戻る / リロード"><span class="logo-mark">R</span><span>RAG Workspace</span></a>
            <div class="nav-sub">登録済み文書を根拠付きで探索できるAIチャット</div>
          </div>
          <div class="nav-actions">
            <button type="button" class="mobile-menu-btn" data-sidebar-mobile-toggle aria-controls="settings-sidebar" aria-expanded="false">{ms_icon("menu", "small")}メニュー</button>
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
    api_url = rag_ask_url()
    store.clear(session_id)
    return HTMLResponse(page_html(session_id, api_url, selected, top_k))


async def ask(request: Request) -> HTMLResponse:
    form = await request.form()
    session_id = str(form.get("session_id") or uuid.uuid4())
    api_url = rag_ask_url()
    selected = selected_from_form(form)
    top_k = int(form.get("top_k") or DEFAULT_TOP_K)
    question = str(form.get("question") or "").strip()
    store.clear(session_id)

    if not selected:
        return HTMLResponse(workspace_html(session_id=session_id, api_url=api_url, selected=selected, top_k=top_k, notice="検索対象文書を1つ以上選択してください。"))
    if not question:
        return HTMLResponse(workspace_html(session_id=session_id, api_url=api_url, selected=selected, top_k=top_k, notice="質問を入力してください。"))

    store.ensure(session_id).append({"role": "user", "content": question})
    payload = {
        "question": question,
        "corpus_ids": selected,
        "top_k": top_k,
        "show_debug": False,
        "session_id": session_id,
        "history": [],
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


async def ask_stream(request: Request) -> StreamingResponse:
    form = await request.form()
    session_id = str(form.get("session_id") or uuid.uuid4())
    api_url = rag_ask_url()
    selected = selected_from_form(form)
    top_k = int(form.get("top_k") or DEFAULT_TOP_K)
    question = str(form.get("question") or "").strip()
    store.clear(session_id)

    async def events():
        if not selected:
            html_value = workspace_html(
                session_id=session_id,
                api_url=api_url,
                selected=selected,
                top_k=top_k,
                notice="検索対象文書を1つ以上選択してください。",
            )
            yield stream_json({"type": "done", "stage": "done", "message": "入力を確認してください", "html": html_value})
            return
        if not question:
            html_value = workspace_html(
                session_id=session_id,
                api_url=api_url,
                selected=selected,
                top_k=top_k,
                notice="質問を入力してください。",
            )
            yield stream_json({"type": "done", "stage": "done", "message": "入力を確認してください", "html": html_value})
            return

        store.ensure(session_id).append({"role": "user", "content": question})
        payload = {
            "question": question,
            "corpus_ids": selected,
            "top_k": top_k,
            "show_debug": False,
            "session_id": session_id,
            "history": [],
        }
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                async with client.stream("POST", rag_ask_stream_url(), json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        event = json.loads(line)
                        if event.get("type") == "progress":
                            yield stream_json(event)
                            continue
                        if event.get("type") == "error":
                            store.ensure(session_id).pop()
                            yield stream_json(event)
                            return
                        if event.get("type") == "done":
                            data = event.get("data") or {}
                            final_session_id = data.get("session_id") or session_id
                            store.ensure(final_session_id).append(
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
                            html_value = workspace_html(
                                session_id=final_session_id,
                                api_url=api_url,
                                selected=selected,
                                top_k=top_k,
                            )
                            yield stream_json({
                                "type": "done",
                                "stage": "done",
                                "message": event.get("message") or "回答が完了しました",
                                "session_id": final_session_id,
                                "html": html_value,
                            })
                            return
            yield stream_json({"type": "error", "stage": "error", "message": "RAG API stream ended without a result."})
        except Exception as exc:
            messages = store.ensure(session_id)
            if messages and messages[-1].get("role") == "user" and messages[-1].get("content") == question:
                messages.pop()
            yield stream_json({"type": "error", "stage": "error", "message": str(exc), "session_id": session_id})

    return StreamingResponse(events(), media_type="application/x-ndjson")


async def source(request: Request) -> HTMLResponse:
    session_id = str(request.query_params.get("session_id") or "")
    message_index = int(request.query_params.get("message_index") or -1)
    source_index = int(request.query_params.get("source_index") or -1)
    item = store.source(session_id, message_index, source_index)
    return HTMLResponse(evidence_response(item, source_index if item else None))


async def report(request: Request) -> HTMLResponse:
    form = await request.form()
    session_id = str(form.get("session_id") or "")
    api_url = rag_ask_url()
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


async def approve_qa(request: Request) -> HTMLResponse:
    form = await request.form()
    session_id = str(form.get("session_id") or "")
    api_url = rag_ask_url()
    message_index = int(form.get("message_index") or -1)
    target_id = f"approve-qa-{message_index}"
    messages = store.ensure(session_id)
    if message_index < 0 or message_index >= len(messages):
        return HTMLResponse(f'<div id="{target_id}" class="error">登録対象の回答が見つかりません。</div>')
    message = messages[message_index]
    if message.get("cache_hit"):
        return HTMLResponse(f'<div id="{target_id}" class="approve-status">{ms_icon("bookmark", "small")}この回答は既に承認済みQA由来です。</div>')
    question = str(message.get("question") or "").strip()
    answer = str(message.get("content") or "").strip()
    evidence = message.get("sources") or []
    if not question or not answer or not evidence:
        return HTMLResponse(f'<div id="{target_id}" class="error">質問・回答・根拠が揃っていないため登録できません。</div>')
    payload = {
        "question": question,
        "answer": answer,
        "evidence": evidence,
        "approved_by": "chat_app",
        "memo": f"チャット画面から登録。session_id={session_id}, log_id={message.get('log_id')}",
    }
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(f"{api_base_url(api_url)}/admin/qa-cache", json=payload)
            response.raise_for_status()
            qa_id = response.json().get("qa_id")
    except Exception as exc:
        return HTMLResponse(f'<div id="{target_id}" class="error">承認済みQA登録に失敗しました: {esc(exc)}</div>')
    message["qa_registered"] = True
    message["qa_id"] = qa_id
    return HTMLResponse(f'<div id="{target_id}" class="approve-status">{ms_icon("check", "small")}承認済みQAに登録しました。QA ID: {esc(qa_id)}</div>')


async def health(_: Request) -> HTMLResponse:
    return HTMLResponse("ok")


rt("/", methods=["GET"])(home)
rt("/ask", methods=["POST"])(ask)
rt("/ask_stream", methods=["POST"])(ask_stream)
rt("/source", methods=["GET"])(source)
rt("/report", methods=["POST"])(report)
rt("/approve_qa", methods=["POST"])(approve_qa)
rt("/health", methods=["GET"])(health)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=get_required_url_value("frontend_bind_host"),
        port=get_url_number("frontend_bind_port"),
    )
