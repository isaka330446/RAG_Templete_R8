# BM25とChromaのベクトル検索を組み合わせて根拠チャンクを返します。
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import chromadb

from api.config import load_settings, project_path
from api.llm_client import OpenAICompatibleEmbedding


BASE_DIR = Path(__file__).resolve().parent.parent
CHUNK_DIR = BASE_DIR / "chunks"
FORM_CATALOG_PATH = BASE_DIR / "data" / "forms" / "form_catalog.csv"


def load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_form_catalog(path: Path = FORM_CATALOG_PATH) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = []
        for row in csv.DictReader(f):
            form_name = (row.get("form_name") or "").strip()
            if not form_name:
                continue
            file_path = (row.get("file_path") or "").strip()
            rows.append({
                "form_name": form_name,
                "file_name": (row.get("file_name") or "").strip(),
                "file_path": file_path,
                "file_type": (row.get("file_type") or "").strip(),
                "description": (row.get("description") or "").strip(),
                "exists": bool(file_path and project_path(file_path).exists()),
            })
        return rows


class DenseRetriever:
    def __init__(self, top_k_final: int | None = None):
        settings = load_settings()
        retrieval_settings = settings.get("retrieval", {})

        self.top_k_final = top_k_final or int(retrieval_settings.get("top_k_final", 8))
        chroma_path = project_path(retrieval_settings.get("chroma_path", "indexes/chroma"))
        self.collection_name = retrieval_settings.get("collection_name", "rag_children")

        child_path = CHUNK_DIR / "child_chunks_with_tags.jsonl"
        if not child_path.exists():
            child_path = CHUNK_DIR / "child_chunks.jsonl"

        self.children = load_jsonl(child_path)
        self.parents = {p["parent_id"]: p for p in load_jsonl(CHUNK_DIR / "parent_chunks.jsonl")}
        self.child_by_id = {c["child_id"]: c for c in self.children}
        self.forms = load_form_catalog()

        self.embedding = OpenAICompatibleEmbedding()
        self.chroma = chromadb.PersistentClient(path=str(chroma_path))
        self.collection = self.chroma.get_or_create_collection(self.collection_name)

    def _match_forms(self, child: dict, parent: dict) -> List[dict]:
        if not self.forms:
            return []
        haystack = "\n".join([
            child.get("title", ""),
            child.get("heading_path", ""),
            child.get("text", ""),
            parent.get("text", ""),
            " ".join(child.get("search_tags", [])),
        ])
        matched = []
        seen = set()
        for form in self.forms:
            name = form.get("form_name", "")
            if name and name in haystack and name not in seen:
                matched.append(form)
                seen.add(name)
        return matched

    def search(self, query: str, corpus_ids: Optional[List[str]] = None, top_k: Optional[int] = None) -> List[dict]:
        top_k = top_k or self.top_k_final
        if corpus_ids is not None and len(corpus_ids) == 0:
            return []

        allowed = set(corpus_ids) if corpus_ids is not None else None
        try:
            q_emb = self.embedding.embed([query])[0]
            where = {"corpus_id": {"$in": list(allowed)}} if allowed is not None else None
            res = self.collection.query(
                query_embeddings=[q_emb],
                n_results=top_k,
                where=where,
                include=["metadatas", "documents", "distances"],
            )
        except Exception:
            return []

        ids = res.get("ids", [[]])[0]
        distances = res.get("distances", [[]])[0]
        rows = []
        for cid, dist in zip(ids, distances):
            child = self.child_by_id.get(cid)
            if not child:
                continue
            parent = self.parents.get(child.get("parent_id"), {})
            score = 1.0 / (1.0 + float(dist))
            rows.append({
                **child,
                "child_text": child.get("text", ""),
                "parent_text": parent.get("text", ""),
                "score": score,
                "forms": self._match_forms(child, parent),
            })

        rows.sort(key=lambda x: x["score"], reverse=True)
        return rows
