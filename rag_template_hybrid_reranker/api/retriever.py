# BM25、Chromaのベクトル検索、rerankerを組み合わせて根拠チャンクを返します。
import json
import re
import csv
from pathlib import Path
from typing import Dict, List, Optional, Any
import chromadb
import requests
from rank_bm25 import BM25Okapi

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


def simple_tokenize(text: str) -> List[str]:
    # 日本語は厳密分かち書きではなく、文字N-gram寄りの簡易実装。
    # 本番では Sudachi / MeCab / TinySegmenter 等に置き換え可能。
    text = text.lower()
    words = re.findall(r"[a-zA-Z0-9_]+|[\u3040-\u30ff\u3400-\u9fff]", text)
    bigrams = [text[i:i+2] for i in range(max(0, len(text)-1)) if not text[i:i+2].isspace()]
    return words + bigrams[:1000]


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
            item = {
                "form_name": form_name,
                "file_name": (row.get("file_name") or "").strip(),
                "file_path": file_path,
                "file_type": (row.get("file_type") or "").strip(),
                "description": (row.get("description") or "").strip(),
                "exists": bool(file_path and project_path(file_path).exists()),
            }
            rows.append(item)
        return rows


class HybridRetriever:
    def __init__(self, top_k_dense: int | None = None, top_k_bm25: int | None = None, top_k_final: int | None = None):
        settings = load_settings()
        retrieval_settings = settings.get("retrieval", {})
        reranker_settings = settings.get("reranker", {})

        self.top_k_dense = top_k_dense or int(retrieval_settings.get("top_k_dense", 20))
        self.top_k_bm25 = top_k_bm25 or int(retrieval_settings.get("top_k_bm25", 20))
        self.top_k_final = top_k_final or int(retrieval_settings.get("top_k_final", 8))
        self.bm25_weight = float(retrieval_settings.get("bm25_weight", 0.35))
        self.dense_weight = float(retrieval_settings.get("dense_weight", 0.65))
        total_weight = self.bm25_weight + self.dense_weight
        if total_weight <= 0:
            self.bm25_weight = 0.35
            self.dense_weight = 0.65
        else:
            self.bm25_weight = self.bm25_weight / total_weight
            self.dense_weight = self.dense_weight / total_weight

        chroma_path = project_path(retrieval_settings.get("chroma_path", "indexes/chroma"))
        self.collection_name = retrieval_settings.get("collection_name") or f'{retrieval_settings.get("collection_prefix", "rag_")}children'
        self.reranker_enabled = bool(reranker_settings.get("enabled", False))
        self.reranker_base_url = str(reranker_settings.get("base_url", "http://127.0.0.1:8003")).rstrip("/")
        self.reranker_top_k = int(reranker_settings.get("top_k", self.top_k_final))
        self.reranker_timeout_sec = int(reranker_settings.get("timeout_sec", 60))

        child_path = CHUNK_DIR / "child_chunks_with_tags.jsonl"
        if not child_path.exists():
            child_path = CHUNK_DIR / "child_chunks.jsonl"

        self.children = load_jsonl(child_path)
        self.parents = {p["parent_id"]: p for p in load_jsonl(CHUNK_DIR / "parent_chunks.jsonl")}
        self.forms = load_form_catalog()

        self.embedding = OpenAICompatibleEmbedding()
        self.chroma = chromadb.PersistentClient(path=str(chroma_path))
        self.collection = self.chroma.get_or_create_collection(self.collection_name)

        self._bm25_docs = []
        self._bm25_child_ids = []
        for c in self.children:
            text = self._search_text(c)
            self._bm25_docs.append(simple_tokenize(text))
            self._bm25_child_ids.append(c["child_id"])
        self.bm25 = BM25Okapi(self._bm25_docs) if self._bm25_docs else None
        self.child_by_id = {c["child_id"]: c for c in self.children}

    def _search_text(self, c: dict) -> str:
        tags = " ".join(c.get("search_tags", []))
        return "\n".join([
            c.get("title", ""),
            c.get("heading_path", ""),
            c.get("text", ""),
            tags,
        ])

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

    def _rerank(self, query: str, rows: List[dict], top_k: int) -> List[dict]:
        if not self.reranker_enabled or not rows:
            return rows

        candidate_count = min(len(rows), max(top_k, self.reranker_top_k))
        candidates = rows[:candidate_count]
        documents = [
            "\n".join([
                r.get("heading_path", ""),
                r.get("child_text", ""),
                (r.get("parent_text", "") or "")[:4000],
            ]).strip()
            for r in candidates
        ]
        payload = {"query": query, "documents": documents, "top_k": candidate_count}

        try:
            res = requests.post(f"{self.reranker_base_url}/rerank", json=payload, timeout=self.reranker_timeout_sec)
            res.raise_for_status()
            data = res.json()
        except Exception:
            return rows

        scored: Dict[int, float] = {}
        if isinstance(data, dict) and isinstance(data.get("scores"), list):
            scored = {i: float(score) for i, score in enumerate(data["scores"][:candidate_count])}
        else:
            results = data.get("results", data) if isinstance(data, dict) else data
            if isinstance(results, list):
                for rank, item in enumerate(results):
                    if isinstance(item, dict):
                        idx = item.get("index", item.get("document_index", rank))
                        score = item.get("relevance_score", item.get("score", item.get("rerank_score", 0.0)))
                        scored[int(idx)] = float(score)

        if not scored:
            return rows

        reranked = []
        for idx, row in enumerate(candidates):
            row = dict(row)
            row["hybrid_score"] = row.get("score", 0.0)
            if idx in scored:
                row["rerank_score"] = scored[idx]
                row["score"] = scored[idx]
            reranked.append(row)
        reranked.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        return reranked + rows[candidate_count:]

    def search(self, query: str, corpus_ids: Optional[List[str]] = None, top_k: Optional[int] = None) -> List[dict]:
        top_k = top_k or self.top_k_final
        if corpus_ids is not None and len(corpus_ids) == 0:
            return []
        allowed = set(corpus_ids) if corpus_ids is not None else None

        dense_scores: Dict[str, float] = {}
        try:
            q_emb = self.embedding.embed([query])[0]
            where = {"corpus_id": {"$in": list(allowed)}} if allowed is not None else None
            res = self.collection.query(
                query_embeddings=[q_emb],
                n_results=self.top_k_dense,
                where=where,
                include=["metadatas", "documents", "distances"],
            )
            ids = res.get("ids", [[]])[0]
            distances = res.get("distances", [[]])[0]
            for cid, dist in zip(ids, distances):
                dense_scores[cid] = 1.0 / (1.0 + float(dist))
        except Exception:
            # インデックス未作成やEmbedding未起動でも、BM25だけで最低限動かす
            dense_scores = {}

        bm25_scores: Dict[str, float] = {}
        if self.bm25:
            tokenized = simple_tokenize(query)
            scores = self.bm25.get_scores(tokenized)
            ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.top_k_bm25]
            max_score = max([scores[i] for i in ranked_idx], default=0.0)
            if max_score <= 0:
                ranked_idx = []
                max_score = 1.0
            for i in ranked_idx:
                cid = self._bm25_child_ids[i]
                child = self.child_by_id[cid]
                if allowed is not None and child.get("corpus_id") not in allowed:
                    continue
                bm25_scores[cid] = float(scores[i]) / float(max_score)

        merged_ids = set(dense_scores) | set(bm25_scores)
        rows = []
        for cid in merged_ids:
            child = self.child_by_id.get(cid)
            if not child:
                continue
            score = self.dense_weight * dense_scores.get(cid, 0.0) + self.bm25_weight * bm25_scores.get(cid, 0.0)
            parent = self.parents.get(child.get("parent_id"), {})
            row = {
                **child,
                "child_text": child.get("text", ""),
                "parent_text": parent.get("text", ""),
                "score": score,
                "forms": self._match_forms(child, parent),
            }
            rows.append(row)

        rows.sort(key=lambda x: x["score"], reverse=True)
        rows = self._rerank(query, rows, top_k)

        # 同一parent_idばかりに偏らないよう軽く間引く
        final = []
        seen_parent = set()
        for r in rows:
            if r.get("parent_id") in seen_parent and len(final) >= max(3, top_k // 2):
                continue
            final.append(r)
            seen_parent.add(r.get("parent_id"))
            if len(final) >= top_k:
                break

        return final
