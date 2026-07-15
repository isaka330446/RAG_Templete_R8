# 評価質問CSVをRAG APIへ順番に投げ、回答結果をCSVに保存します。
from pathlib import Path
import argparse
import csv
import json
import time
import requests
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="評価用CSV")
    parser.add_argument("--output", default="eval/eval_results.csv")
    parser.add_argument("--api", default="http://127.0.0.1:8000/ask")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = BASE_DIR / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    rows = []

    for _, r in df.iterrows():
        q = str(r["question"])
        corpus_ids = []
        if "corpus_ids" in r and pd.notna(r["corpus_ids"]):
            corpus_ids = [x.strip() for x in str(r["corpus_ids"]).split("|") if x.strip()]

        payload = {
            "question": q,
            "corpus_ids": corpus_ids or None,
            "show_debug": True,
        }
        started = time.time()
        res = requests.post(args.api, json=payload, timeout=240)
        elapsed = time.time() - started
        res.raise_for_status()
        data = res.json()

        source_summary = []
        for s in data.get("sources", []):
            source_summary.append(f'{s.get("corpus_id")}::{s.get("source_file")}::{s.get("heading_path")}')

        rows.append({
            "question_id": r.get("question_id", ""),
            "question": q,
            "expected_answer": r.get("expected_answer", ""),
            "answer": data.get("answer", ""),
            "expected_source": r.get("expected_source", ""),
            "actual_sources": " || ".join(source_summary),
            "corpus_ids": "|".join(corpus_ids),
            "elapsed_sec": round(elapsed, 3),
            "manual_score": "",
            "manual_comment": "",
        })

    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
