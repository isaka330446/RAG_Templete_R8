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


def chroma_metadata(row: dict) -> dict:
    keys = [
        "child_id",
        "parent_id",
        "corpus_id",
        "title",
        "source_file",
        "heading_path",
        "document_type",
        "source_type",
        "meeting_id",
        "meeting_name",
        "meeting_date",
        "agenda",
        "topic",
        "section_title",
        "slide_title",
        "content_type",
    ]
    metadata = {}
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            metadata[key] = value
        else:
            metadata[key] = str(value)
    if row.get("slide_no") is not None:
        metadata["slide_no"] = int(row.get("slide_no") or 0)
    return metadata


def main():
    child_path = CHUNK_DIR / "child_chunks_with_tags.jsonl"
    if not child_path.exists():
        child_path = CHUNK_DIR / "child_chunks.jsonl"

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
            docs.append(search_text)
            ids.append(r["child_id"])
            metadatas.append(chroma_metadata(r))

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
