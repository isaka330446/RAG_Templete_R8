# LLMなしで税務向けの簡易SearchTagと検索用テキストを生成します。
from __future__ import annotations

import json
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
CHILD_IN = BASE_DIR / "chunks" / "child_chunks.jsonl"
CHILD_OUT = BASE_DIR / "chunks" / "child_chunks_with_tags.jsonl"

TAX_TERMS = [
    "相続税",
    "贈与税",
    "財産評価",
    "財産の評価",
    "譲渡所得",
    "所得税",
    "相続",
    "遺贈",
    "贈与",
    "被相続人",
    "相続人",
    "受贈者",
    "法定相続人",
    "基礎控除",
    "相続時精算課税",
    "暦年課税",
    "小規模宅地等の特例",
    "配偶者控除",
    "税額控除",
    "申告期限",
    "納税",
    "延納",
    "物納",
    "死亡保険金",
    "死亡退職金",
    "債務控除",
    "葬式費用",
    "土地評価",
    "家屋評価",
    "宅地評価",
    "路線価方式",
    "倍率方式",
    "借地権",
    "貸家建付地",
    "農地",
    "山林",
    "非上場株式",
    "株式評価",
    "取得費",
    "譲渡費用",
    "譲渡収入",
    "長期譲渡所得",
    "短期譲渡所得",
    "分離課税",
    "居住用財産",
    "空き家特例",
    "国外転出時課税",
    "非居住者",
    "取得時期",
    "関連コード",
    "根拠法令等",
    "通達",
    "基本通達",
    "法令解釈通達",
    "相続税法基本通達",
    "財産評価基本通達",
    "評価通達",
    "租税特別措置法",
    "措置法",
    "措置法通達",
    "小規模宅地等",
    "事業承継税制",
    "農地等",
    "山林",
    "非上場株式等",
    "情報PDF",
    "税解釈",
    "質疑応答",
    "質疑応答事例",
    "照会要旨",
    "回答要旨",
    "回答",
    "事例",
    "令和",
    "税制改正",
    "あらまし",
]

CATEGORY_TAGS = {
    "01_inheritance_tax": ["相続税", "相続", "遺産", "被相続人"],
    "02_gift_tax": ["贈与税", "贈与", "受贈者"],
    "03_property_valuation": ["財産評価", "財産の評価", "評価"],
    "04_transfer_income": ["譲渡所得", "所得税", "資産譲渡"],
    "05_related_income_tax": ["所得税", "相続・贈与関連所得税"],
    "nta_sozoku_kihon_tsutatsu": ["相続税法基本通達", "相続税", "基本通達", "法令解釈通達"],
    "nta_zaisan_hyoka_kihon_tsutatsu": ["財産評価基本通達", "財産評価", "評価通達", "法令解釈通達"],
    "nta_sozoku_sochiho_tsutatsu": ["租税特別措置法", "措置法通達", "相続税", "特例", "法令解釈通達"],
    "nta_sozoku_joho_zeikaishaku_pdf": ["相続税・贈与税関係情報PDF", "情報PDF", "税解釈", "相続税", "贈与税"],
    "nta_sozoku_shitsugi": ["質疑応答事例", "相続税", "贈与税", "照会要旨", "回答要旨"],
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_unique(tags: list[str], value: str) -> None:
    value = re.sub(r"\s+", " ", value).strip()
    if not value or re.fullmatch(r"[\W_]+", value):
        return
    if value and value not in tags:
        tags.append(value)


def normalize_code(value: str) -> str:
    table = str.maketrans("０１２３４５６７８９", "0123456789")
    value = value.translate(table)
    value = re.sub(r"[^0-9-]", "", value)
    if not re.search(r"\d", value):
        return ""
    return value


def infer_source_tags(row: dict) -> list[str]:
    tags: list[str] = []
    source_file = row.get("source_file", "")
    corpus_id = row.get("corpus_id", "")
    for key, values in CATEGORY_TAGS.items():
        if key in source_file or key == corpus_id:
            for value in values:
                append_unique(tags, value)
    for key in ["document_type", "document_id", "tsutatsu_name", "tax_type", "asset_tax_category", "category"]:
        if row.get(key):
            append_unique(tags, str(row[key]))
    return tags


def extract_code_tags(text: str) -> list[str]:
    tags: list[str] = []
    patterns = [
        r"TaxAnswer No\.\s*:\s*([0-9０-９-]+)",
        r"No\.([0-9０-９-]+)",
        r"コード\s*([0-9０-９-]+)",
    ]
    for pattern in patterns:
        for raw_code in re.findall(pattern, text):
            code = normalize_code(raw_code)
            if len(code.replace("-", "")) < 3:
                continue
            append_unique(tags, code)
            append_unique(tags, f"No.{code}")
            append_unique(tags, f"コード{code}")
    return tags


def extract_title_phrases(row: dict) -> list[str]:
    tags: list[str] = []
    title = str(row.get("title", ""))
    title = re.sub(r"^[0-9-]+_", "", title)
    title = re.sub(r"^No\.[0-9-]+\s*", "", title)
    title = title.replace("_", " ")
    append_unique(tags, title)
    for phrase in re.split(r"[、。・（）()\[\]「」『』\s]+", title):
        if 2 <= len(phrase) <= 30:
            append_unique(tags, phrase)
    return tags


def make_tags(row: dict) -> list[str]:
    haystack = "\n".join(
        [
            str(row.get("title", "")),
            str(row.get("heading_path", "")),
            str(row.get("source_file", "")),
            str(row.get("text", "")),
        ]
    )
    tags: list[str] = []
    for value in infer_source_tags(row):
        append_unique(tags, value)
    for value in extract_code_tags(haystack):
        append_unique(tags, value)
    for value in extract_title_phrases(row):
        append_unique(tags, value)
    for term in TAX_TERMS:
        if term in haystack:
            append_unique(tags, term)

    return tags[:40]


def main() -> None:
    rows = load_jsonl(CHILD_IN)
    for row in rows:
        tags = make_tags(row)
        row["search_tags"] = tags
        row["search_text"] = "\n".join(
            [
                row.get("title", ""),
                row.get("heading_path", ""),
                row.get("text", ""),
                " ".join(tags),
            ]
        ).strip()

    with CHILD_OUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote: {CHILD_OUT} rows={len(rows)}")


if __name__ == "__main__":
    main()
