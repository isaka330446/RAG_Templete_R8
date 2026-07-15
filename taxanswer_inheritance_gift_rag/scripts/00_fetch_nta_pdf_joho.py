# 国税庁の相続税・贈与税関係PDFを取得し、Markdown化の素材を準備します。
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from api.config import get_nta_value


BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_ID = "nta_sozoku_joho_zeikaishaku_pdf"
PDF_DIR = BASE_DIR / "data" / "pdf" / CORPUS_ID
MARKDOWN_DIR = BASE_DIR / "data" / "markdown" / CORPUS_ID
SOURCE_DIR = BASE_DIR / "data" / "sources"
LINKS_CSV = SOURCE_DIR / "nta_sozoku_joho_zeikaishaku_pdf_links.csv"
MANIFEST_JSON = SOURCE_DIR / "nta_sozoku_joho_zeikaishaku_pdf_manifest.json"

REQUEST_HEADERS = {
    "User-Agent": "rag-template-nta-pdf-fetcher/1.0 (+offline-rag-prep)",
}
PDF_SOURCE_CONFIG = get_nta_value("joho_zeikaishaku_pdf_sources")


@dataclass
class PdfSource:
    document_id: str
    url: str
    title: str
    document_type: str = "情報・質疑応答PDF"
    tax_type: str = "相続税・贈与税"
    corpus_id: str = CORPUS_ID
    pdf_path: str = ""
    markdown_path: str = ""
    page_count: int = 0
    extracted_chars: int = 0
    fetch_status: str = "pending"
    error: str = ""


SOURCES = [
    PdfSource(
        document_id="0025006-064",
        url=str(PDF_SOURCE_CONFIG[0]["url"]),
        title="国税庁 相続税・贈与税関係 情報（0025006-064）",
    ),
    PdfSource(
        document_id="0025005-103",
        url=str(PDF_SOURCE_CONFIG[1]["url"]),
        title="国税庁 相続税・贈与税関係 情報（0025005-103）",
    ),
    PdfSource(
        document_id="0024006-159",
        url=str(PDF_SOURCE_CONFIG[2]["url"]),
        title="国税庁 相続税・贈与税関係 情報（0024006-159）",
    ),
    PdfSource(
        document_id="0024005-164",
        url=str(PDF_SOURCE_CONFIG[3]["url"]),
        title="国税庁 相続税・贈与税関係 情報（0024005-164）",
    ),
]


def require_pypdf():
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pypdf is required. Install it with: python -m pip install pypdf"
        ) from exc
    return PdfReader


def download_pdf(url: str, out_path: Path, timeout_sec: int) -> None:
    req = Request(url, headers=REQUEST_HEADERS)
    with urlopen(req, timeout=timeout_sec) as res:
        content_type = res.headers.get("Content-Type", "")
        raw = res.read()
    if not raw.startswith(b"%PDF"):
        raise RuntimeError(f"downloaded content is not a PDF: content_type={content_type}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(raw)


def normalize_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    previous_blank = True
    for raw_line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not line:
            if not previous_blank:
                lines.append("")
            previous_blank = True
            continue
        lines.append(line)
        previous_blank = False
    return "\n".join(lines).strip()


def infer_title(default_title: str, metadata_title: str | None, first_page_text: str) -> str:
    if metadata_title:
        clean = normalize_text(metadata_title).splitlines()[0].strip()
        is_document_id = bool(re.fullmatch(r"[0-9]{6,}-[0-9]+", clean))
        if clean and not clean.lower().endswith(".pdf") and not is_document_id:
            return clean[:120]

    title_lines: list[str] = []
    for line in normalize_text(first_page_text).splitlines()[:20]:
        clean = line.strip(" -―—\t")
        if not clean:
            if title_lines:
                break
            continue
        if re.fullmatch(r"\d+", clean):
            continue
        if any(skip in clean for skip in ["資 産 課 税 課 情 報", "国 税 庁", "資産課税 課"]):
            break
        title_lines.append(clean)
        joined = "".join(title_lines)
        if "について（情報）" in joined or "質疑応答事例" in joined:
            return joined[:160]
    if title_lines:
        return "".join(title_lines)[:160]
    return default_title


def extract_pdf_text(pdf_path: Path, default_title: str) -> tuple[str, str, int, int]:
    PdfReader = require_pypdf()
    reader = PdfReader(str(pdf_path))
    metadata_title = None
    try:
        metadata_title = reader.metadata.title if reader.metadata else None
    except Exception:
        metadata_title = None

    page_texts: list[str] = []
    for idx, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            text = f"[PDF text extraction error on page {idx}: {exc}]"
        page_texts.append(normalize_text(text))

    first_page_text = page_texts[0] if page_texts else ""
    title = infer_title(default_title, metadata_title, first_page_text)
    extracted_chars = sum(len(text) for text in page_texts)

    body_parts = []
    for idx, text in enumerate(page_texts, start=1):
        body_parts.append(f"## Page {idx}\n\n{text or '[No extractable text on this page]'}")

    return title, "\n\n".join(body_parts).strip(), len(page_texts), extracted_chars


def yaml_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def safe_filename(value: str) -> str:
    value = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "_", value)
    value = re.sub(r"\s+", "_", value).strip("._ ")
    return value or "document"


def pdf_filename(source: PdfSource) -> str:
    parsed = urlparse(source.url)
    name = Path(parsed.path).name
    return safe_filename(name)


def markdown_filename(source: PdfSource, title: str) -> str:
    safe_id = re.sub(r"[^0-9A-Za-z_\-.]+", "_", source.document_id).strip("._")
    return f"joho_{safe_id or 'unknown'}.md"


def render_markdown(source: PdfSource, title: str, body: str) -> str:
    frontmatter = {
        "title": title,
        "corpus_id": source.corpus_id,
        "document_id": source.document_id,
        "document_type": source.document_type,
        "tax_type": source.tax_type,
        "source_url": source.url,
        "source_site": "国税庁",
        "pdf_path": source.pdf_path,
    }
    yaml_lines = ["---"]
    yaml_lines.extend(f"{key}: {yaml_quote(value)}" for key, value in frontmatter.items())
    yaml_lines.append("---")

    metadata_lines = [
        f"# {title}",
        "",
        f"- 文書種別: {source.document_type}",
        f"- 文書ID: {source.document_id}",
        f"- 税目: {source.tax_type}",
        f"- 出典URL: {source.url}",
        f"- PDF保存先: {source.pdf_path}",
    ]

    return "\n".join(yaml_lines + [""] + metadata_lines + ["", body]).strip() + "\n"


def write_links_csv(rows: list[PdfSource]) -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "document_id",
        "title",
        "url",
        "document_type",
        "tax_type",
        "corpus_id",
        "pdf_path",
        "markdown_path",
        "page_count",
        "extracted_chars",
        "fetch_status",
        "error",
    ]
    with LINKS_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: getattr(row, key) for key in fieldnames})


def write_manifest(rows: list[PdfSource], started_at: str, finished_at: str) -> None:
    statuses: dict[str, int] = {}
    for row in rows:
        statuses[row.fetch_status] = statuses.get(row.fetch_status, 0) + 1
    manifest = {
        "corpus_id": CORPUS_ID,
        "source_site": "国税庁",
        "document_type": "情報・質疑応答PDF",
        "started_at": started_at,
        "finished_at": finished_at,
        "target_count": len(rows),
        "markdown_count": sum(1 for row in rows if row.fetch_status == "ok"),
        "page_count": sum(row.page_count for row in rows),
        "extracted_chars": sum(row.extracted_chars for row in rows),
        "statuses": statuses,
        "outputs": {
            "links_csv": str(LINKS_CSV.relative_to(BASE_DIR)).replace("\\", "/"),
            "pdf_dir": str(PDF_DIR.relative_to(BASE_DIR)).replace("\\", "/"),
            "markdown_dir": str(MARKDOWN_DIR.relative_to(BASE_DIR)).replace("\\", "/"),
        },
        "targets": [asdict(row) for row in rows],
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch NTA inheritance/gift PDF documents and convert them to Markdown.")
    parser.add_argument("--timeout-sec", type=int, default=60, help="HTTP timeout seconds.")
    parser.add_argument("--dry-run", action="store_true", help="Only write source manifests.")
    parser.add_argument("--clean", action="store_true", help="Remove existing PDF/Markdown outputs before writing.")
    args = parser.parse_args()

    started_at = datetime.now(timezone.utc).isoformat()
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    rows = [PdfSource(**asdict(source)) for source in SOURCES]
    if args.dry_run:
        write_links_csv(rows)
        write_manifest(rows, started_at, datetime.now(timezone.utc).isoformat())
        print(f"collected pdf sources={len(rows)} -> {LINKS_CSV}")
        return

    if args.clean:
        for target_dir in [PDF_DIR, MARKDOWN_DIR]:
            if target_dir.exists():
                if not target_dir.resolve().is_relative_to((BASE_DIR / "data").resolve()):
                    raise RuntimeError(f"Refusing to clean outside data directory: {target_dir}")
                shutil.rmtree(target_dir)

    require_pypdf()
    for idx, source in enumerate(rows, start=1):
        try:
            pdf_path = PDF_DIR / pdf_filename(source)
            download_pdf(source.url, pdf_path, timeout_sec=args.timeout_sec)
            source.pdf_path = str(pdf_path.relative_to(BASE_DIR)).replace("\\", "/")
            title, body, page_count, extracted_chars = extract_pdf_text(pdf_path, source.title)
            source.title = title
            source.page_count = page_count
            source.extracted_chars = extracted_chars

            md_path = MARKDOWN_DIR / markdown_filename(source, title)
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text(render_markdown(source, title, body), encoding="utf-8")
            source.markdown_path = str(md_path.relative_to(BASE_DIR)).replace("\\", "/")
            source.fetch_status = "ok"
            print(f"[{idx}/{len(rows)}] ok {source.document_id} pages={page_count} chars={extracted_chars} {title}")
        except Exception as exc:
            source.fetch_status = "error"
            source.error = str(exc)
            print(f"[{idx}/{len(rows)}] error {source.document_id}: {exc}")

    finished_at = datetime.now(timezone.utc).isoformat()
    write_links_csv(rows)
    write_manifest(rows, started_at, finished_at)
    ok_count = sum(1 for row in rows if row.fetch_status == "ok")
    print(f"fetched pdf markdown={ok_count}/{len(rows)}")
    print(f"links csv: {LINKS_CSV}")
    print(f"manifest: {MANIFEST_JSON}")


if __name__ == "__main__":
    main()
