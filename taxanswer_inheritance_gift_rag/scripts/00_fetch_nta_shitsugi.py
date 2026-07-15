# 国税庁の相続税・贈与税に関する質疑応答事例を取得してMarkdown化します。
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from api.config import get_nta_url


BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_ID = "nta_sozoku_shitsugi"
SOURCE_DIR = BASE_DIR / "data" / "sources"
MARKDOWN_DIR = BASE_DIR / "data" / "markdown" / CORPUS_ID
LINKS_CSV = SOURCE_DIR / "nta_sozoku_shitsugi_links.csv"
MANIFEST_JSON = SOURCE_DIR / "nta_sozoku_shitsugi_manifest.json"

START_URL = get_nta_url("shitsugi_start_url")
SCOPE_PREFIX = get_nta_url("shitsugi_scope_prefix")

REQUEST_HEADERS = {
    "User-Agent": "rag-template-nta-shitsugi-fetcher/1.0 (+offline-rag-prep)",
}

STOP_MARKERS = {
    "このページの先頭へ",
    "サイトマップ（コンテンツ一覧）",
    "サイトマップ",
}

SKIP_LINES = {
    "すべての機能をご利用いただくにはJavascriptを有効にしてください。",
    "ホーム",
    "法令等",
    "質疑応答事例",
}


@dataclass
class ShitsugiPage:
    corpus_id: str
    document_type: str
    tax_type: str
    url: str
    title: str = ""
    relative_url_path: str = ""
    markdown_path: str = ""
    fetch_status: str = "pending"
    error: str = ""


class LinkCollector(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href") or ""
        if href:
            self.links.append(urljoin(self.base_url, href).split("#")[0])


def normalize_space(value: str) -> str:
    value = unescape(value)
    value = value.replace("\u3000", " ")
    return re.sub(r"\s+", " ", value).strip()


def fetch_html(url: str, timeout_sec: int = 30) -> str:
    req = Request(url, headers=REQUEST_HEADERS)
    with urlopen(req, timeout=timeout_sec) as res:
        raw = res.read()
        content_type = res.headers.get("Content-Type", "")

    encoding = detect_encoding(raw, content_type)
    for candidate in [encoding, "shift_jis", "cp932", "utf-8"]:
        try:
            return raw.decode(candidate)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def detect_encoding(raw: bytes, content_type: str) -> str:
    header_match = re.search(r"charset=([A-Za-z0-9_\-]+)", content_type, flags=re.IGNORECASE)
    if header_match:
        return header_match.group(1)

    head = raw[:4096].decode("ascii", errors="ignore")
    meta_match = re.search(r"<meta[^>]+charset=[\"']?([A-Za-z0-9_\-]+)", head, flags=re.IGNORECASE)
    if meta_match:
        return meta_match.group(1)

    return "utf-8"


def is_scoped_html_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if parsed.netloc != "www.nta.go.jp":
        return False
    if not url.startswith(SCOPE_PREFIX):
        return False
    return parsed.path.endswith(".htm")


def collect_links(html: str, base_url: str) -> list[str]:
    parser = LinkCollector(base_url)
    parser.feed(html)
    seen: set[str] = set()
    links: list[str] = []
    for link in parser.links:
        if is_scoped_html_url(link) and link not in seen:
            seen.add(link)
            links.append(link)
    return links


def natural_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def relative_url_path(url: str) -> str:
    return url[len(SCOPE_PREFIX) :]


def is_content_page(url: str) -> bool:
    return relative_url_path(url) != "01.htm"


def crawl(timeout_sec: int, sleep_sec: float, max_pages: int) -> tuple[list[str], dict[str, str]]:
    queue = deque([START_URL])
    seen: set[str] = set()
    html_cache: dict[str, str] = {}

    while queue:
        url = queue.popleft()
        if url in seen:
            continue
        if max_pages and len(seen) >= max_pages:
            break
        seen.add(url)

        html = fetch_html(url, timeout_sec=timeout_sec)
        html_cache[url] = html
        for link in collect_links(html, url):
            if link not in seen:
                queue.append(link)
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    urls = sorted(seen, key=lambda u: natural_key(relative_url_path(u)))
    return urls, html_cache


def html_to_markdown(html: str) -> str:
    text = re.sub(
        r"(?is)<(script|style|noscript|svg|form|button|select|textarea)\b[^>]*>.*?</\1>",
        "\n",
        html,
    )
    text = re.sub(
        r"(?is)<img\b[^>]*\balt=[\"']([^\"']+)[\"'][^>]*>",
        lambda m: f"\n[画像: {normalize_space(m.group(1))}]\n",
        text,
    )
    for level in range(6, 0, -1):
        text = re.sub(rf"(?is)<h{level}\b[^>]*>", "\n" + ("#" * level) + " ", text)
        text = re.sub(rf"(?is)</h{level}>", "\n", text)
    text = re.sub(r"(?is)<br\s*/?>", "\n", text)
    text = re.sub(r"(?is)<li\b[^>]*>", "\n- ", text)
    text = re.sub(r"(?is)</li>", "\n", text)
    text = re.sub(r"(?is)<(th|td)\b[^>]*>", " ", text)
    text = re.sub(r"(?is)</(th|td)>", " | ", text)
    text = re.sub(r"(?is)</(p|div|section|article|main|tr|table|ul|ol|dl|dt|dd)>", "\n", text)
    text = re.sub(r"(?is)<[^>]+>", "", text)
    text = unescape(text)

    lines: list[str] = []
    previous_blank = True
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = normalize_space(raw_line)
        if line in SKIP_LINES:
            continue
        if not line:
            if not previous_blank:
                lines.append("")
            previous_blank = True
            continue
        lines.append(line)
        previous_blank = False
    return "\n".join(lines).strip()


def trim_to_main_content(markdown: str) -> str:
    lines = markdown.splitlines()
    start = 0
    first_h2 = None
    for idx, line in enumerate(lines):
        if re.match(r"^#\s+", line):
            start = idx
            break
        if first_h2 is None and re.match(r"^##\s+", line):
            first_h2 = idx
    else:
        if first_h2 is not None:
            start = first_h2

    trimmed: list[str] = []
    for line in lines[start:]:
        plain = line.lstrip("#").strip()
        if trimmed and plain in STOP_MARKERS:
            break
        trimmed.append(line)

    while trimmed and not trimmed[-1].strip():
        trimmed.pop()
    return "\n".join(trimmed).strip()


def extract_html_title(html: str) -> str:
    match = re.search(r"(?is)<title[^>]*>(.*?)</title>", html)
    if not match:
        return ""
    title = normalize_space(re.sub(r"<[^>]+>", "", match.group(1)))
    return re.sub(r"\s*[｜|]\s*国税庁\s*$", "", title).strip()


def extract_title(markdown: str, fallback: str) -> str:
    match = re.search(r"^#{1,2}\s+(.+)$", markdown, flags=re.MULTILINE)
    if match:
        title = normalize_space(match.group(1))
        if title:
            return title
    return fallback


def extract_law_basis_date(markdown: str) -> str:
    match = re.search(r"(令和[0-9０-９]+年[0-9０-９]+月[0-9０-９]+日現在)", markdown)
    if not match:
        return ""
    return normalize_space(match.group(1))


def yaml_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def safe_filename(relative_path: str, title: str) -> str:
    stem = re.sub(r"\.htm$", "", relative_path, flags=re.IGNORECASE)
    stem = re.sub(r"[\\/]+", "__", stem)
    stem = re.sub(r"[^0-9A-Za-z_\-.]+", "_", stem).strip("._")
    return f"shitsugi_{stem or 'unknown'}.md"


def markdown_path_for(page: ShitsugiPage) -> Path:
    return MARKDOWN_DIR / safe_filename(page.relative_url_path, page.title)


def render_markdown(page: ShitsugiPage, html: str) -> str:
    converted = trim_to_main_content(html_to_markdown(html))
    page.title = extract_title(converted, page.title or extract_html_title(html) or page.relative_url_path)
    body = re.sub(r"^#{1,2}\s+.+(?:\n+)?", "", converted, count=1).lstrip()
    document_id = re.sub(r"\.htm$", "", page.relative_url_path, flags=re.IGNORECASE)
    law_basis_date = extract_law_basis_date(converted)

    frontmatter = {
        "title": page.title,
        "corpus_id": page.corpus_id,
        "document_id": document_id,
        "document_type": page.document_type,
        "tax_type": page.tax_type,
        "source_url": page.url,
        "source_site": "国税庁",
    }
    if law_basis_date:
        frontmatter["law_basis_date"] = law_basis_date
    yaml_lines = ["---"]
    yaml_lines.extend(f"{key}: {yaml_quote(value)}" for key, value in frontmatter.items())
    yaml_lines.append("---")

    metadata_lines = [
        f"# {page.title}",
        "",
        f"- 文書種別: {page.document_type}",
        f"- 文書ID: {document_id}",
        f"- 税目: {page.tax_type}",
        f"- 出典URL: {page.url}",
    ]
    if law_basis_date:
        metadata_lines.append(f"- 法令基準日: {law_basis_date}")
    return "\n".join(yaml_lines + [""] + metadata_lines + ["", body]).strip() + "\n"


def write_links_csv(rows: Iterable[ShitsugiPage]) -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "corpus_id",
        "document_type",
        "tax_type",
        "title",
        "url",
        "relative_url_path",
        "markdown_path",
        "fetch_status",
        "error",
    ]
    with LINKS_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: getattr(row, key) for key in fieldnames})


def write_manifest(rows: list[ShitsugiPage], started_at: str, finished_at: str) -> None:
    statuses: dict[str, int] = {}
    for row in rows:
        statuses[row.fetch_status] = statuses.get(row.fetch_status, 0) + 1

    manifest = {
        "corpus_id": CORPUS_ID,
        "source_site": "国税庁",
        "document_type": "質疑応答事例",
        "source_url": START_URL,
        "started_at": started_at,
        "finished_at": finished_at,
        "target_count": len(rows),
        "markdown_count": sum(1 for row in rows if row.fetch_status == "ok"),
        "statuses": statuses,
        "outputs": {
            "links_csv": str(LINKS_CSV.relative_to(BASE_DIR)).replace("\\", "/"),
            "markdown_dir": str(MARKDOWN_DIR.relative_to(BASE_DIR)).replace("\\", "/"),
        },
        "targets": [asdict(row) for row in rows],
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch NTA inheritance/gift Q&A pages and convert them to Markdown.")
    parser.add_argument("--limit", type=int, default=0, help="Limit pages for smoke tests.")
    parser.add_argument("--max-pages", type=int, default=1000, help="Safety limit for crawler.")
    parser.add_argument("--sleep-sec", type=float, default=0.05, help="Sleep seconds between page fetches.")
    parser.add_argument("--timeout-sec", type=int, default=30, help="HTTP timeout seconds.")
    parser.add_argument("--dry-run", action="store_true", help="Only collect target links.")
    parser.add_argument("--clean", action="store_true", help="Remove existing Markdown output before writing.")
    args = parser.parse_args()

    started_at = datetime.now(timezone.utc).isoformat()
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    urls, html_cache = crawl(args.timeout_sec, args.sleep_sec, args.max_pages)
    urls = [url for url in urls if is_content_page(url)]
    if args.limit > 0:
        urls = urls[: args.limit]

    pages = [
        ShitsugiPage(
            corpus_id=CORPUS_ID,
            document_type="質疑応答事例",
            tax_type="相続税・贈与税",
            url=url,
            relative_url_path=relative_url_path(url),
        )
        for url in urls
    ]

    if args.dry_run:
        write_links_csv(pages)
        write_manifest(pages, started_at, datetime.now(timezone.utc).isoformat())
        print(f"collected shitsugi pages={len(pages)} -> {LINKS_CSV}")
        return

    if args.clean and MARKDOWN_DIR.exists():
        if not MARKDOWN_DIR.resolve().is_relative_to((BASE_DIR / "data" / "markdown").resolve()):
            raise RuntimeError(f"Refusing to clean outside markdown directory: {MARKDOWN_DIR}")
        shutil.rmtree(MARKDOWN_DIR)

    for idx, page in enumerate(pages, start=1):
        try:
            html = html_cache.get(page.url)
            if html is None:
                html = fetch_html(page.url, timeout_sec=args.timeout_sec)
            markdown = render_markdown(page, html)
            out_path = markdown_path_for(page)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(markdown, encoding="utf-8")
            page.markdown_path = str(out_path.relative_to(BASE_DIR)).replace("\\", "/")
            page.fetch_status = "ok"
            print(f"[{idx}/{len(pages)}] ok {page.relative_url_path} {page.title}")
        except Exception as exc:
            page.fetch_status = "error"
            page.error = str(exc)
            print(f"[{idx}/{len(pages)}] error {page.url}: {exc}")
        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    finished_at = datetime.now(timezone.utc).isoformat()
    write_links_csv(pages)
    write_manifest(pages, started_at, finished_at)
    ok_count = sum(1 for row in pages if row.fetch_status == "ok")
    print(f"fetched shitsugi markdown={ok_count}/{len(pages)}")
    print(f"links csv: {LINKS_CSV}")
    print(f"manifest: {MANIFEST_JSON}")


if __name__ == "__main__":
    main()
