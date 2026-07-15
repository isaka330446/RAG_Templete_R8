# 100問評価セットから、管理者レビュー用の承認済みQAキャッシュseed候補を生成します。
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
QA100_CSV = BASE_DIR / "eval" / "qa_100_questions.csv"
CHILD_CHUNKS = BASE_DIR / "chunks" / "child_chunks_with_tags.jsonl"
SEED_DIR = BASE_DIR / "data" / "approved_qa_seed"
SEED_JSONL = SEED_DIR / "approved_qa_seed_100.jsonl"
SEED_CSV = SEED_DIR / "approved_qa_seed_100_summary.csv"

DEFAULT_CORPUS_VERSION = "nta_taxanswer_inheritance_gift_v1"
DEFAULT_INDEX_VERSION = "bge-m3_chroma_v1"

GENERIC_SOURCE_CORPORA = {
    "財産評価基本通達": "nta_zaisan_hyoka_kihon_tsutatsu",
    "相続税法基本通達": "nta_sozoku_kihon_tsutatsu",
    "措置法通達": "nta_sozoku_sochiho_tsutatsu",
}

SOURCE_HINTS = [
    ("路線価", ["路線価", "倍率方式"]),
    ("倍率方式", ["倍率方式", "路線価"]),
    ("不整形地", ["不整形地"]),
    ("貸宅地", ["貸宅地", "自用地", "土地の上に存する権利が競合する場合の宅地"]),
    ("借地権", ["借地権", "定期借地権"]),
    ("家屋", ["家屋及び家屋の上に存する権利", "家屋", "固定資産税評価額"]),
    ("非上場株式", ["取引相場のない株式", "類似業種比準価額", "純資産価額"]),
    ("株式", ["取引相場のない株式", "株式", "類似業種比準価額"]),
    ("農地", ["農地及び農地の上に存する権利", "農地", "市街地農地"]),
    ("山林", ["山林及び山林の上に存する権利", "山林"]),
    ("定期金", ["定期金に関する権利", "契約に基づかない定期金", "定期金"]),
    ("死亡保険金", ["第3条", "生命保険金", "保険金"]),
    ("土地を無償", ["借地権", "使用貸借", "宅地及び宅地の上に存する権利"]),
    ("財産評価と小規模宅地", ["財産評価", "評価単位", "宅地"]),
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def query_terms(*values: str) -> set[str]:
    text = normalize_text(" ".join(values))
    terms = set(re.findall(r"[a-zA-Z0-9]+|[\u3040-\u30ff\u3400-\u9fff]{2,}", text))
    compact = re.sub(r"\s+", "", text)
    terms.update(compact[i : i + 2] for i in range(max(0, len(compact) - 1)))
    return {term for term in terms if term}


def row_text(row: dict[str, Any]) -> str:
    return normalize_text(
        "\n".join(
            [
                str(row.get("title") or ""),
                str(row.get("heading_path") or ""),
                str(row.get("search_text") or ""),
                str(row.get("text") or ""),
                " ".join(row.get("search_tags") or []),
            ]
        )
    )


def score_row(row: dict[str, Any], terms: set[str], hints: list[str] | None = None) -> float:
    text = row_text(row)
    score = 0.0
    title = normalize_text(str(row.get("title") or ""))
    heading = normalize_text(str(row.get("heading_path") or ""))
    for hint in hints or []:
        hint = normalize_text(hint)
        if not hint:
            continue
        if hint in title:
            score += 80.0
        elif hint in heading:
            score += 55.0
        elif hint in text:
            score += 25.0
    for term in terms:
        if not term:
            continue
        if term in title:
            score += 6.0
        elif term in heading:
            score += 3.0
        elif term in text:
            score += 1.0
    return score


def best_rows(
    rows: list[dict[str, Any]],
    terms: set[str],
    hints: list[str] | None = None,
    limit: int = 1,
) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: score_row(row, terms, hints), reverse=True)
    return [row for row in ranked[:limit] if score_row(row, terms, hints) > 0 or ranked]


def source_hints(question_row: dict[str, str], source_ref: str) -> list[str]:
    text = " ".join(
        [
            question_row.get("question", ""),
            question_row.get("expected_answer", ""),
            question_row.get("memo", ""),
            source_ref,
        ]
    )
    hints: list[str] = []
    for keyword, values in SOURCE_HINTS:
        if keyword in text:
            hints.extend(values)
    if "相続税法基本通達 3条" in source_ref:
        hints.extend(["第3条", "生命保険金", "保険金"])
    return list(dict.fromkeys(hints))


def candidates_for_source(source_ref: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_ref = source_ref.strip()
    if not source_ref:
        return []

    taxanswer_match = re.search(r"No\.?([0-9]{4})", source_ref, flags=re.IGNORECASE)
    if taxanswer_match:
        code = taxanswer_match.group(1)
        return [row for row in rows if str(row.get("taxanswer_no") or "") == code]

    pdf_match = re.search(r"([0-9]{7}-[0-9]{3})", source_ref)
    if pdf_match:
        pdf_id = pdf_match.group(1)
        return [row for row in rows if pdf_id in str(row.get("source_file") or "")]

    shitsugi_match = re.search(r"質疑応答\s+([0-9A-Za-z]+(?:/[0-9]+)?)", source_ref)
    if shitsugi_match:
        document_id = shitsugi_match.group(1)
        if "/" in document_id:
            return [
                row
                for row in rows
                if row.get("corpus_id") == "nta_sozoku_shitsugi"
                and str(row.get("document_id") or "") == document_id
            ]
        return [
            row
            for row in rows
            if row.get("corpus_id") == "nta_sozoku_shitsugi"
            and str(row.get("document_id") or "").startswith(f"{document_id}/")
        ]

    for label, corpus_id in GENERIC_SOURCE_CORPORA.items():
        if label in source_ref:
            return [row for row in rows if row.get("corpus_id") == corpus_id]

    if "空き家" in source_ref:
        return [
            row
            for row in rows
            if row.get("corpus_id") == "nta_taxanswer_asset_tax"
            and ("空き家" in row_text(row) or "3306" == str(row.get("taxanswer_no") or ""))
        ]

    return [row for row in rows if source_ref in row_text(row)]


def evidence_from_child(row: dict[str, Any], source_ref: str) -> dict[str, Any]:
    keys = [
        "corpus_id",
        "parent_id",
        "child_id",
        "title",
        "heading_path",
        "source_file",
        "source_url",
        "document_type",
        "document_id",
        "taxanswer_no",
        "tsutatsu_name",
        "law_basis_date",
        "search_tags",
    ]
    evidence = {key: row.get(key) for key in keys if row.get(key) not in (None, "")}
    evidence["child_text"] = row.get("text", "")
    evidence["score"] = 1.0
    evidence["source_ref"] = source_ref
    return evidence


def build_seed_item(
    question_row: dict[str, str],
    child_rows: list[dict[str, Any]],
    *,
    corpus_version: str,
    index_version: str,
) -> dict[str, Any]:
    terms = query_terms(
        question_row.get("question", ""),
        question_row.get("expected_answer", ""),
        question_row.get("category", ""),
        question_row.get("memo", ""),
    )
    evidence: list[dict[str, Any]] = []
    seen_child_ids: set[str] = set()
    source_refs = [item.strip() for item in question_row.get("expected_sources", "").split("|") if item.strip()]
    if not source_refs:
        source_refs = [question_row.get("expected_sources", "")]

    for source_ref in source_refs:
        candidates = candidates_for_source(source_ref, child_rows)
        hints = source_hints(question_row, source_ref)
        for child in best_rows(candidates, terms, hints=hints, limit=1):
            child_id = str(child.get("child_id") or "")
            if child_id and child_id not in seen_child_ids:
                evidence.append(evidence_from_child(child, source_ref))
                seen_child_ids.add(child_id)

    if not evidence:
        allowed = {item.strip() for item in question_row.get("corpus_ids", "").split("|") if item.strip()}
        candidates = [row for row in child_rows if not allowed or row.get("corpus_id") in allowed]
        for child in best_rows(candidates, terms, hints=source_hints(question_row, "fallback"), limit=1):
            evidence.append(evidence_from_child(child, "fallback"))

    return {
        "seed_id": question_row["question_id"].replace("QA", "CACHE"),
        "source_question_id": question_row["question_id"],
        "approval_status": "candidate_needs_admin_review",
        "question": question_row["question"],
        "answer": question_row["expected_answer"],
        "evidence": evidence,
        "expected_sources": question_row.get("expected_sources", ""),
        "corpus_ids": [item.strip() for item in question_row.get("corpus_ids", "").split("|") if item.strip()],
        "reference_scope": question_row.get("reference_scope", ""),
        "trap_type": question_row.get("trap_type", ""),
        "style": question_row.get("style", ""),
        "category": question_row.get("category", ""),
        "difficulty": question_row.get("difficulty", ""),
        "corpus_version": corpus_version,
        "index_version": index_version,
        "memo": f"seed from {question_row['question_id']}: {question_row.get('memo', '')}",
    }


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_summary_csv(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed_id",
        "source_question_id",
        "question",
        "answer",
        "expected_sources",
        "evidence_count",
        "evidence_child_ids",
        "evidence_titles",
        "evidence_source_files",
        "reference_scope",
        "trap_type",
        "style",
        "category",
        "difficulty",
        "approval_status",
        "corpus_version",
        "index_version",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            evidence = item.get("evidence") or []
            writer.writerow(
                {
                    "seed_id": item["seed_id"],
                    "source_question_id": item["source_question_id"],
                    "question": item["question"],
                    "answer": item["answer"],
                    "expected_sources": item["expected_sources"],
                    "evidence_count": len(evidence),
                    "evidence_child_ids": "|".join(str(e.get("child_id") or "") for e in evidence),
                    "evidence_titles": "|".join(str(e.get("title") or "") for e in evidence),
                    "evidence_source_files": "|".join(str(e.get("source_file") or "") for e in evidence),
                    "reference_scope": item["reference_scope"],
                    "trap_type": item["trap_type"],
                    "style": item["style"],
                    "category": item["category"],
                    "difficulty": item["difficulty"],
                    "approval_status": item["approval_status"],
                    "corpus_version": item["corpus_version"],
                    "index_version": item["index_version"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create approved-QA seed candidates from the 100-question QA set.")
    parser.add_argument("--input", default=str(QA100_CSV), help="Input qa_100_questions.csv")
    parser.add_argument("--chunks", default=str(CHILD_CHUNKS), help="child_chunks_with_tags.jsonl")
    parser.add_argument("--output-jsonl", default=str(SEED_JSONL), help="Output seed JSONL")
    parser.add_argument("--output-csv", default=str(SEED_CSV), help="Output summary CSV")
    parser.add_argument("--corpus-version", default=DEFAULT_CORPUS_VERSION, help="Corpus version stored in seed rows")
    parser.add_argument("--index-version", default=DEFAULT_INDEX_VERSION, help="Index version stored in seed rows")
    args = parser.parse_args()

    questions = load_csv(Path(args.input))
    child_rows = load_jsonl(Path(args.chunks))
    items = [
        build_seed_item(
            row,
            child_rows,
            corpus_version=args.corpus_version,
            index_version=args.index_version,
        )
        for row in questions
    ]

    write_jsonl(Path(args.output_jsonl), items)
    write_summary_csv(Path(args.output_csv), items)

    missing_evidence = [item["seed_id"] for item in items if not item.get("evidence")]
    print(f"wrote jsonl: {args.output_jsonl}")
    print(f"wrote csv: {args.output_csv}")
    print(f"items={len(items)} missing_evidence={len(missing_evidence)}")
    if missing_evidence:
        print("missing:", ", ".join(missing_evidence))


if __name__ == "__main__":
    main()
