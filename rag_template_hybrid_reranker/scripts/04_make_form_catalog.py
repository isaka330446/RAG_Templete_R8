# RAG回答に添付候補として出せる帳票・様式ファイルのカタログを作成します。
from pathlib import Path
import csv
import re

BASE_DIR = Path(__file__).resolve().parent.parent
FORMS_DIR = BASE_DIR / "data" / "forms"
OUT = FORMS_DIR / "form_catalog.csv"

SUPPORTED = {".xlsx", ".xlsm", ".xls", ".docx", ".doc"}


def normalize_form_name(path: Path) -> str:
    name = path.stem
    # 先頭番号や版数を軽く除去
    name = re.sub(r"^[0-9０-９]+[_\-　\s]*", "", name)
    name = re.sub(r"（.*?）|\(.*?\)", "", name).strip()
    return name


def main():
    rows = []
    for p in sorted(FORMS_DIR.rglob("*")):
        if p.name == OUT.name:
            continue
        if p.suffix.lower() not in SUPPORTED:
            continue

        rel = p.relative_to(BASE_DIR).as_posix()
        rows.append({
            "form_name": normalize_form_name(p),
            "file_name": p.name,
            "file_path": rel,
            "file_type": p.suffix.lower().lstrip("."),
            "description": "",
        })

    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["form_name", "file_name", "file_path", "file_type", "description"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote: {OUT} rows={len(rows)}")


if __name__ == "__main__":
    main()
