# 国税庁タックスアンサーの資産税関連ページを取得してMarkdown化します。
from __future__ import annotations

import argparse
import csv
import json
import re
import time
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
CORPUS_ID = "nta_taxanswer_asset_tax"
SOURCE_DIR = BASE_DIR / "data" / "sources"
MARKDOWN_DIR = BASE_DIR / "data" / "markdown" / CORPUS_ID
LINKS_CSV = SOURCE_DIR / "nta_taxanswer_asset_tax_links.csv"
MANIFEST_JSON = SOURCE_DIR / "nta_taxanswer_asset_tax_manifest.json"

NTA_BASE_URL = get_nta_url("site_base_url")
INDEX_URL = get_nta_url("taxanswer_code_index_url")
BUNYA_URL = get_nta_url("taxanswer_asset_tax_url")

REQUEST_HEADERS = {
    "User-Agent": "rag-template-taxanswer-fetcher/1.0 (+offline-rag-prep)",
}

TARGET_MAJOR_SECTIONS = {
    "相続税",
    "贈与税",
    "財産の評価",
    "譲渡所得",
}

TAX_ORDER = {
    "相続税": 1,
    "贈与税": 2,
    "財産の評価": 3,
    "所得税": 4,
}

DIR_BY_ASSET_CATEGORY = {
    "相続税": "01_inheritance_tax",
    "贈与税": "02_gift_tax",
    "財産の評価": "03_property_valuation",
    "譲渡所得": "04_transfer_income",
    "相続・贈与関連所得税": "05_related_income_tax",
}

ARTICLE_RE = re.compile(r"/taxes/shiraberu/taxanswer/([^/?#]+)/([0-9-]+)\.htm$")
CODE_TITLE_RE = re.compile(r"^(?:No\.)?\s*([0-9-]+)\s+(.+)$")

STOP_MARKERS = {
    "お問い合わせ先",
    "このコンテンツはお役にたちましたか？",
    "ご協力ありがとうございました",
    "サイトマップ（コンテンツ一覧）",
}

SKIP_LINES = {
    "すべての機能をご利用いただくにはJavascriptを有効にしてください。",
    "このページの先頭へ",
    "はい いいえ",
}


@dataclass(frozen=True)
class RawLink:
    url: str
    text: str
    h2: str
    h3: str
    source_url: str


@dataclass
class TargetArticle:
    code: str
    title: str
    url: str
    tax_type: str
    category: str
    asset_tax_category: str
    source_index_url: str
    source_h2: str
    source_h3: str
    markdown_path: str = ""
    law_basis_date: str = ""
    fetch_status: str = "pending"
    error: str = ""


class TaxAnswerLinkParser(HTMLParser):
    def __init__(self, base_url: str, source_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.source_url = source_url
        self.current_h2 = ""
        self.current_h3 = ""
        self._heading_tag = ""
        self._heading_parts: list[str] = []
        self._link_href = ""
        self._link_parts: list[str] = []
        self.links: list[RawLink] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = dict(attrs)
        if tag in {"h2", "h3"}:
            self._heading_tag = tag
            self._heading_parts = []
        elif tag == "a":
            self._link_href = attr.get("href") or ""
            self._link_parts = []

    def handle_endtag(self, tag: str) -> None:
        if self._heading_tag == tag:
            text = normalize_space("".join(self._heading_parts))
            if tag == "h2":
                self.current_h2 = text
                self.current_h3 = ""
            elif tag == "h3":
                self.current_h3 = text
            self._heading_tag = ""
            self._heading_parts = []
        elif tag == "a" and self._link_href:
            text = normalize_space("".join(self._link_parts))
            if text:
                self.links.append(
                    RawLink(
                        url=urljoin(self.base_url, self._link_href).split("#")[0],
                        text=text,
                        h2=self.current_h2,
                        h3=self.current_h3,
                        source_url=self.source_url,
                    )
                )
            self._link_href = ""
            self._link_parts = []

    def handle_data(self, data: str) -> None:
        if self._heading_tag:
            self._heading_parts.append(data)
        if self._link_href:
            self._link_parts.append(data)


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
    try:
        return raw.decode(encoding)
    except UnicodeDecodeError:
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


def parse_links(source_url: str, html: str) -> list[RawLink]:
    parser = TaxAnswerLinkParser(NTA_BASE_URL, source_url)
    parser.feed(html)
    return parser.links


def parse_article_url(url: str) -> tuple[str, str] | None:
    path = urlparse(url).path
    match = ARTICLE_RE.search(path)
    if not match:
        return None
    return match.group(1), match.group(2)


def split_code_title(raw_text: str, fallback_code: str) -> tuple[str, str]:
    text = normalize_space(raw_text)
    match = CODE_TITLE_RE.match(text)
    if match:
        return match.group(1), match.group(2).strip()
    text = re.sub(rf"^(?:No\.)?\s*{re.escape(fallback_code)}\s*", "", text).strip()
    return fallback_code, text or f"No.{fallback_code}"


def classify_link(link: RawLink) -> TargetArticle | None:
    parsed = parse_article_url(link.url)
    if not parsed:
        return None

    _, code = parsed
    code, clean_title = split_code_title(link.text, code)
    major = link.h2
    minor = link.h3

    if major in {"相続税", "贈与税", "財産の評価"}:
        tax_type = major
        asset_category = major
    elif major == "譲渡所得":
        tax_type = "所得税"
        asset_category = "譲渡所得"
    elif link.source_url == BUNYA_URL and major == "所得税":
        tax_type = "所得税"
        asset_category = "相続・贈与関連所得税"
    else:
        return None

    return TargetArticle(
        code=code,
        title=clean_title,
        url=link.url,
        tax_type=tax_type,
        category=minor or major,
        asset_tax_category=asset_category,
        source_index_url=link.source_url,
        source_h2=major,
        source_h3=minor,
    )


def sort_key(article: TargetArticle) -> tuple[int, int, str]:
    numeric = int(re.sub(r"\D", "", article.code) or "0")
    return TAX_ORDER.get(article.tax_type, 99), numeric, article.code


def collect_targets(timeout_sec: int) -> list[TargetArticle]:
    targets_by_url: dict[str, TargetArticle] = {}
    for source_url in (BUNYA_URL, INDEX_URL):
        html = fetch_html(source_url, timeout_sec=timeout_sec)
        for link in parse_links(source_url, html):
            target = classify_link(link)
            if target and target.url not in targets_by_url:
                targets_by_url[target.url] = target

    return sorted(targets_by_url.values(), key=sort_key)


def html_fragment_to_markdown(html: str) -> str:
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


def trim_to_article(markdown: str) -> str:
    lines = markdown.splitlines()
    start = 0
    for idx, line in enumerate(lines):
        if re.match(r"^#\s+No\.[0-9-]+", line) or re.match(r"^#\s+", line):
            start = idx
            break

    trimmed: list[str] = []
    for line in lines[start:]:
        plain = line.lstrip("#").strip()
        if trimmed and (plain in STOP_MARKERS or plain.startswith("税の情報・手続・用紙")):
            break
        trimmed.append(line)

    while trimmed and not trimmed[-1].strip():
        trimmed.pop()
    return "\n".join(trimmed).strip()


def extract_title(markdown: str, article: TargetArticle) -> str:
    match = re.search(r"^#\s+(.+)$", markdown, flags=re.MULTILINE)
    if match:
        return normalize_space(match.group(1))
    return f"No.{article.code} {article.title}".strip()


def extract_law_basis_date(markdown: str) -> str:
    match = re.search(r"\[(令和[^\]\n]+現在法令等)\]", markdown)
    return normalize_space(match.group(1)) if match else ""


def remove_first_heading(markdown: str) -> str:
    return re.sub(r"^#\s+.+(?:\n+)?", "", markdown, count=1).lstrip()


def yaml_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def safe_filename(code: str, title: str) -> str:
    safe_code = re.sub(r"[^0-9A-Za-z_\-.]+", "_", str(code)).strip("._")
    return f"taxanswer_{safe_code or 'unknown'}.md"


def markdown_path_for(article: TargetArticle) -> Path:
    dirname = DIR_BY_ASSET_CATEGORY.get(article.asset_tax_category, "99_other")
    return MARKDOWN_DIR / dirname / safe_filename(article.code, article.title)


def render_markdown(article: TargetArticle, article_html: str) -> tuple[str, str]:
    converted = trim_to_article(html_fragment_to_markdown(article_html))
    page_title = extract_title(converted, article)
    law_basis_date = extract_law_basis_date(converted)
    body = remove_first_heading(converted)

    frontmatter = {
        "title": page_title,
        "taxanswer_no": article.code,
        "tax_type": article.tax_type,
        "category": article.category,
        "asset_tax_category": article.asset_tax_category,
        "source_url": article.url,
        "source_site": "国税庁",
        "corpus_id": CORPUS_ID,
        "law_basis_date": law_basis_date,
    }
    yaml_lines = ["---"]
    yaml_lines.extend(f"{key}: {yaml_quote(value)}" for key, value in frontmatter.items())
    yaml_lines.append("---")

    metadata_lines = [
        f"# {page_title}",
        "",
        f"- TaxAnswer No.: {article.code}",
        f"- 出典URL: {article.url}",
        f"- 税目: {article.tax_type}",
        f"- カテゴリ: {article.category}",
        f"- 資産税カテゴリ: {article.asset_tax_category}",
    ]
    if law_basis_date:
        metadata_lines.append(f"- 法令基準日: {law_basis_date}")

    markdown = "\n".join(yaml_lines + [""] + metadata_lines + ["", body]).strip() + "\n"
    return markdown, law_basis_date


def write_links_csv(rows: Iterable[TargetArticle]) -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "code",
        "title",
        "tax_type",
        "category",
        "asset_tax_category",
        "url",
        "source_index_url",
        "source_h2",
        "source_h3",
        "markdown_path",
        "law_basis_date",
        "fetch_status",
        "error",
    ]
    with LINKS_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: getattr(row, key) for key in fieldnames})


def write_manifest(rows: list[TargetArticle], started_at: str, finished_at: str) -> None:
    counts: dict[str, int] = {}
    statuses: dict[str, int] = {}
    for row in rows:
        counts[row.asset_tax_category] = counts.get(row.asset_tax_category, 0) + 1
        statuses[row.fetch_status] = statuses.get(row.fetch_status, 0) + 1

    manifest = {
        "corpus_id": CORPUS_ID,
        "source_site": "国税庁",
        "source_urls": [BUNYA_URL, INDEX_URL],
        "started_at": started_at,
        "finished_at": finished_at,
        "target_count": len(rows),
        "markdown_count": sum(1 for row in rows if row.fetch_status == "ok"),
        "counts_by_asset_tax_category": counts,
        "statuses": statuses,
        "outputs": {
            "links_csv": str(LINKS_CSV.relative_to(BASE_DIR)).replace("\\", "/"),
            "markdown_dir": str(MARKDOWN_DIR.relative_to(BASE_DIR)).replace("\\", "/"),
        },
        "targets": [asdict(row) for row in rows],
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch NTA TaxAnswer pages and convert them to Markdown.")
    parser.add_argument("--limit", type=int, default=0, help="Limit article count for smoke tests.")
    parser.add_argument("--sleep-sec", type=float, default=0.2, help="Sleep seconds between article fetches.")
    parser.add_argument("--timeout-sec", type=int, default=30, help="HTTP timeout seconds.")
    parser.add_argument("--dry-run", action="store_true", help="Only collect target links.")
    args = parser.parse_args()

    started_at = datetime.now(timezone.utc).isoformat()
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)

    targets = collect_targets(timeout_sec=args.timeout_sec)
    if args.limit > 0:
        targets = targets[: args.limit]

    if args.dry_run:
        write_links_csv(targets)
        write_manifest(targets, started_at, datetime.now(timezone.utc).isoformat())
        print(f"collected targets={len(targets)} -> {LINKS_CSV}")
        return

    for idx, target in enumerate(targets, start=1):
        try:
            html = fetch_html(target.url, timeout_sec=args.timeout_sec)
            markdown, law_basis_date = render_markdown(target, html)
            out_path = markdown_path_for(target)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(markdown, encoding="utf-8")
            target.markdown_path = str(out_path.relative_to(BASE_DIR)).replace("\\", "/")
            target.law_basis_date = law_basis_date
            target.fetch_status = "ok"
            print(f"[{idx}/{len(targets)}] ok {target.code} {target.title}")
        except Exception as exc:
            target.fetch_status = "error"
            target.error = str(exc)
            print(f"[{idx}/{len(targets)}] error {target.code} {target.url}: {exc}")
        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    finished_at = datetime.now(timezone.utc).isoformat()
    write_links_csv(targets)
    write_manifest(targets, started_at, finished_at)

    ok_count = sum(1 for row in targets if row.fetch_status == "ok")
    print(f"fetched markdown={ok_count}/{len(targets)}")
    print(f"links csv: {LINKS_CSV}")
    print(f"manifest: {MANIFEST_JSON}")
    print(f"markdown dir: {MARKDOWN_DIR}")


if __name__ == "__main__":
    main()
