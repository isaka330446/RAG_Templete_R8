# 子チャンクをEmbeddingし、Chromaの永続インデックスを再作成します。
from pathlib import Path
import json
import os
from typing import List
from tqdm import tqdm
import chromadb

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from api.config import load_settings, project_path
from api.llm_client import OpenAICompatibleEmbedding
from api.text_cleaning import clean_rag_text


BASE_DIR = Path(__file__).resolve().parent.parent
CHUNK_DIR = BASE_DIR / "chunks"
SETTINGS = load_settings()
RETRIEVAL_SETTINGS = SETTINGS.get("retrieval", {})
CHROMA_DIR = project_path(RETRIEVAL_SETTINGS.get("chroma_path", "indexes/chroma"))
COLLECTION_NAME = RETRIEVAL_SETTINGS.get("collection_name") or f'{RETRIEVAL_SETTINGS.get("collection_prefix", "rag_")}children'


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def batch_iter(rows, size=64):
    for i in range(0, len(rows), size):
        yield rows[i:i+size]


def child_chunk_source_path() -> Path:
    tagged_path = CHUNK_DIR / "child_chunks_with_tags.jsonl"
    base_path = CHUNK_DIR / "child_chunks.jsonl"
    if tagged_path.exists() and (
        not base_path.exists() or tagged_path.stat().st_mtime >= base_path.stat().st_mtime
    ):
        return tagged_path
    return base_path


def main():
    child_path = child_chunk_source_path()

    rows = load_jsonl(child_path)
    if not rows:
        print(f"no rows to index: {child_path}")
        return

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedder = OpenAICompatibleEmbedding()
    indexed_batches = []

    for batch in tqdm(list(batch_iter(rows, 32)), desc="indexing"):
        docs = []
        ids = []
        metadatas = []

        for r in batch:
            search_text = r.get("search_text") or "\n".join([
                r.get("title", ""),
                r.get("heading_path", ""),
                r.get("text", ""),
                " ".join(r.get("search_tags", [])),
            ])
            search_text = clean_rag_text(search_text)
            docs.append(search_text)
            ids.append(r["child_id"])
            metadatas.append({
                "child_id": r["child_id"],
                "parent_id": r["parent_id"],
                "corpus_id": r["corpus_id"],
                "title": r.get("title", ""),
                "source_file": r.get("source_file", ""),
                "heading_path": r.get("heading_path", ""),
                "valid_from": r.get("valid_from", ""),
                "valid_until": r.get("valid_until", ""),
            })

        embeddings = embedder.embed(docs)
        indexed_batches.append({
            "ids": ids,
            "documents": docs,
            "metadatas": metadatas,
            "embeddings": embeddings,
        })

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(COLLECTION_NAME)
    for batch in indexed_batches:
        collection.add(**batch)

    print(f"indexed rows={len(rows)} collection={COLLECTION_NAME} -> {CHROMA_DIR}")


if __name__ == "__main__":
    main()
