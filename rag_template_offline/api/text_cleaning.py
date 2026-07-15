from __future__ import annotations

import html
import re


HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
BR_TAG_RE = re.compile(r"<\s*br\s*/?\s*>", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>\n]+>")
WORK_LABEL_RE = re.compile(
    r"(^|\n)(?P<prefix>\s*#{1,6}\s*)?(?:\u89aa\u30c1\u30e3\u30f3\u30af\u5019\u88dc|\u5b50\u30c1\u30e3\u30f3\u30af\u5019\u88dc)\s*[:\uff1a]\s*"
)


def clean_rag_text(value: object, *, strip_html_tags: bool = True) -> str:
    """Remove conversion/control markup before indexing, display, or LLM context use."""
    text = str(value or "")
    if not text:
        return ""
    text = HTML_COMMENT_RE.sub("", text)
    text = text.replace("<!--", "").replace("-->", "")
    if strip_html_tags:
        text = BR_TAG_RE.sub("\n", text)
        text = HTML_TAG_RE.sub("", text)
    text = html.unescape(text)

    def replace_work_label(match: re.Match[str]) -> str:
        return f"{match.group(1)}{match.group('prefix') or ''}"

    text = WORK_LABEL_RE.sub(replace_work_label, text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_display_text(value: object) -> str:
    return clean_rag_text(value)
