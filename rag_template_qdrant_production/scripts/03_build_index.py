# 子チャンクをEmbeddingし、Qdrantの永続インデックスを再作成します。
from pathlib import Path
import argparse
import json
from typing import List

import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))


BASE_DIR = Path(__file__).resolve().parent.parent
CHUNK_DIR = BASE_DIR / "chunks"


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def batch_iter(rows, size=64):
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def search_text(row: dict) -> str:
    return row.get("search_text") or "\n".join(
        [
            row.get("title", ""),
            row.get("heading_path", ""),
            row.get("text", ""),
            " ".join(row.get("search_tags", [])),
        ]
    )


def payload(row: dict, release: dict) -> dict:
    return {
        "child_id": row["child_id"],
        "parent_id": row["parent_id"],
        "corpus_id": row["corpus_id"],
        "title": row.get("title", ""),
        "source_file": row.get("source_file", ""),
        "heading_path": row.get("heading_path", ""),
        "release_id": release["release_id"],
        "corpus_version": release["corpus_version"],
        "index_version": release["index_version"],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Build a non-destructive versioned vector index.")
    parser.add_argument("--corpus-version", default=None)
    parser.add_argument("--index-version", default=None)
    parser.add_argument("--collection-name", default=None)
    parser.add_argument(
        "--activate",
        action="store_true",
        help="Mark this release active after a successful build. Without this, it remains staging.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        from tqdm import tqdm
    except Exception:
        def tqdm(iterable, desc=None):
            return iterable

    from api.config import load_settings
    from api.llm_client import OpenAICompatibleEmbedding
    from api.release_manager import ReleaseManager
    from api.vector_store import create_vector_store

    child_path = CHUNK_DIR / "child_chunks_with_tags.jsonl"
    if not child_path.exists():
        child_path = CHUNK_DIR / "child_chunks.jsonl"

    rows = load_jsonl(child_path)
    if not rows:
        print(f"no rows to index: {child_path}")
        return

    settings = load_settings()
    release_manager = ReleaseManager(settings)
    release = release_manager.begin_build(
        corpus_version=args.corpus_version,
        index_version=args.index_version,
        collection_name=args.collection_name,
    )
    vector_store = create_vector_store(settings, collection_name=release["collection_name"])
    embedder = OpenAICompatibleEmbedding()
    indexed_batches = []
    embedding_size = None

    try:
        for batch in tqdm(list(batch_iter(rows, 32)), desc=f"indexing:{vector_store.provider}:{release['release_id']}"):
            docs = [search_text(row) for row in batch]
            embeddings = embedder.embed(docs)
            if embeddings and embedding_size is None:
                embedding_size = len(embeddings[0])
            indexed_batches.append(
                [
                    {
                        "child_id": row["child_id"],
                        "document": doc,
                        "payload": payload(row, release),
                        "embedding": embedding,
                    }
                    for row, doc, embedding in zip(batch, docs, embeddings)
                ]
            )

        if not embedding_size:
            release_manager.mark_failed(release["release_id"], "no embeddings generated")
            print("no embeddings generated")
            return

        vector_store.recreate(embedding_size)
        for batch in indexed_batches:
            vector_store.upsert(batch)

        completed = release_manager.complete_build(
            release["release_id"],
            row_count=len(rows),
            embedding_size=embedding_size,
            activate=args.activate,
        )

        print(
            f"indexed rows={len(rows)} provider={vector_store.provider} "
            f"collection={vector_store.collection_name} release_id={completed['release_id']} "
            f"status={completed['status']}"
        )
        if completed["status"] == "staging":
            print(f"activate later: python scripts/06_manage_releases.py --activate {completed['release_id']}")
    except Exception as exc:
        release_manager.mark_failed(release["release_id"], str(exc))
        raise


if __name__ == "__main__":
    main()
