# ローカルLLMで子チャンクごとのSearchTagを生成します。
from pathlib import Path
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
import requests
from dotenv import load_dotenv

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from api.config import load_settings

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
CHILD_IN = BASE_DIR / "chunks" / "child_chunks.jsonl"
CHILD_OUT = BASE_DIR / "chunks" / "child_chunks_with_tags.jsonl"

SETTINGS = load_settings().get("llm", {})
LLM_BASE_URL = (os.getenv("LLM_BASE_URL") or SETTINGS.get("base_url", "http://127.0.0.1:8001/v1")).rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY") or SETTINGS.get("api_key", "dummy")
LLM_MODEL = os.getenv("LLM_MODEL") or SETTINGS.get("model", "local-model")
LLM_TIMEOUT_SEC = int(SETTINGS.get("timeout_sec", 180))

MAX_WORKERS = int(os.getenv("SEARCH_TAG_WORKERS", "12"))


TAG_PROMPT = """あなたはRAG検索品質を改善するためのSearchTag作成担当です。

以下の子チャンクに対して、検索に役立つ短いタグを日本語中心で10〜20個作成してください。

条件:
- 文書に明示されている概念・手続・様式・条件・対象者を優先する。
- ユーザーが質問で使いそうな言い換えも含める。
- 本文にない制度を勝手に追加しない。
- 出力はJSON配列のみ。
- 例: ["申請手続", "承認", "サンプル申請書"]

# 子チャンク
{chunk}
"""


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def call_llm(chunk: str) -> List[str]:
    url = f"{LLM_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {LLM_API_KEY}"}
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "JSONだけを返してください。"},
            {"role": "user", "content": TAG_PROMPT.format(chunk=chunk[:5000])},
        ],
        "temperature": 0.1,
        "max_tokens": 512,
    }
    res = requests.post(url, json=payload, headers=headers, timeout=LLM_TIMEOUT_SEC)
    res.raise_for_status()
    content = res.json()["choices"][0]["message"]["content"].strip()

    # ```json 対策
    content = re.sub(r"^```json\s*", "", content)
    content = re.sub(r"^```\s*", "", content)
    content = re.sub(r"\s*```$", "", content)

    try:
        tags = json.loads(content)
        if isinstance(tags, list):
            return [str(x).strip() for x in tags if str(x).strip()]
    except Exception:
        pass

    # 壊れた場合のフォールバック
    return []


def process(row: dict) -> dict:
    tags = call_llm(row.get("text", ""))
    row["search_tags"] = tags
    row["search_text"] = "\n".join([
        row.get("title", ""),
        row.get("heading_path", ""),
        row.get("text", ""),
        " ".join(tags),
    ])
    return row


def main():
    rows = load_jsonl(CHILD_IN)
    done = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(process, row): row for row in rows}
        for fut in as_completed(futures):
            row = futures[fut]
            try:
                done.append(fut.result())
            except Exception as e:
                row["search_tags"] = []
                row["search_text"] = "\n".join([row.get("title", ""), row.get("heading_path", ""), row.get("text", "")])
                row["tag_error"] = str(e)
                done.append(row)

    done.sort(key=lambda r: r["child_id"])

    with CHILD_OUT.open("w", encoding="utf-8") as f:
        for row in done:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote: {CHILD_OUT} rows={len(done)}")


if __name__ == "__main__":
    main()
