# BM25、Chromaのベクトル検索、任意のrerankerを組み合わせて根拠チャンクを返します。
import json
import re
import csv
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Any
import chromadb
import requests
from rank_bm25 import BM25Okapi

from api.config import get_required_url, load_settings, project_path
from api.llm_client import OpenAICompatibleEmbedding
from api.text_cleaning import clean_rag_text


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


def child_chunk_source_path() -> Path:
    tagged_path = CHUNK_DIR / "child_chunks_with_tags.jsonl"
    base_path = CHUNK_DIR / "child_chunks.jsonl"
    if tagged_path.exists() and (
        not base_path.exists() or tagged_path.stat().st_mtime >= base_path.stat().st_mtime
    ):
        return tagged_path
    return base_path


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
        search_tag_settings = settings.get("search_tags", {})

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
        self.search_tags_enabled = bool(search_tag_settings.get("enabled_in_retrieval", True))
        self.reranker_base_url = get_required_url("reranker_base_url")
        self.reranker_top_k = int(reranker_settings.get("top_k", self.top_k_final))
        self.reranker_timeout_sec = int(reranker_settings.get("timeout_sec", 60))
        self.reranker_min_score = float(reranker_settings.get("min_score", 0.0))
        self.reranker_min_keep = max(1, int(reranker_settings.get("min_keep", 1)))
        self.reranker_max_keep = max(1, int(reranker_settings.get("max_keep", self.top_k_final)))
        self.reranker_dedupe_parent = bool(reranker_settings.get("dedupe_parent", False))

        child_path = child_chunk_source_path()

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
        tags = " ".join(c.get("search_tags", [])) if self.search_tags_enabled else ""
        return clean_rag_text("\n".join([
            c.get("title", ""),
            c.get("heading_path", ""),
            c.get("text", ""),
            tags,
        ]))

    def _match_forms(self, child: dict, parent: dict) -> List[dict]:
        if not self.forms:
            return []
        haystack = clean_rag_text("\n".join([
            child.get("title", ""),
            child.get("heading_path", ""),
            child.get("text", ""),
            parent.get("text", ""),
            " ".join(child.get("search_tags", [])) if self.search_tags_enabled else "",
        ]))
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

    def _allowed_corpus_ids(self, corpus_ids: Optional[List[str]]) -> Optional[set]:
        if corpus_ids is not None and len(corpus_ids) == 0:
            return set()
        return set(corpus_ids) if corpus_ids is not None else None

    def _validity_status(self, row: dict, today: date | None = None) -> str:
        today = today or date.today()
        valid_from = str(row.get("valid_from") or "").strip()
        valid_until = str(row.get("valid_until") or "").strip()
        start = None
        end = None
        try:
            if valid_from:
                start = date.fromisoformat(valid_from)
            if valid_until:
                end = date.fromisoformat(valid_until)
        except ValueError:
            return "invalid_period"
        if start and end and start > end:
            return "invalid_period"
        if start and today < start:
            return "not_started"
        if end and today > end:
            return "expired"
        if not start and not end:
            return "unbounded"
        return "current"

    def _is_current_document(self, row: dict) -> bool:
        return self._validity_status(row) in {"current", "unbounded"}

    def _row_from_child(self, child: dict, score: float, **extra: Any) -> dict:
        parent = self.parents.get(child.get("parent_id"), {})
        return {
            **child,
            "child_text": clean_rag_text(child.get("text", "")),
            "parent_text": clean_rag_text(parent.get("text", "")),
            "valid_status": self._validity_status(child),
            "score": score,
            "forms": self._match_forms(child, parent),
            **extra,
        }

    def _diversify_by_parent(self, rows: List[dict], top_k: int) -> List[dict]:
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

    def _dedupe_parent_strict(self, rows: List[dict]) -> List[dict]:
        final = []
        seen_parent = set()
        for row in rows:
            parent_id = row.get("parent_id")
            if parent_id and parent_id in seen_parent:
                continue
            final.append(row)
            if parent_id:
                seen_parent.add(parent_id)
        return final

    def _apply_reranker_filters(self, rows: List[dict], top_k: int) -> List[dict]:
        if not rows:
            return []

        target_count = min(top_k, self.reranker_max_keep) if self.reranker_max_keep > 0 else top_k
        target_count = max(1, target_count)
        min_keep = min(max(1, self.reranker_min_keep), target_count)

        scored_rows = [row for row in rows if "rerank_score" in row]
        if not scored_rows:
            return self._diversify_by_parent(rows, top_k)

        passed = [row for row in scored_rows if float(row.get("rerank_score") or 0.0) >= self.reranker_min_score]
        candidates = passed if len(passed) >= min_keep else scored_rows
        if self.reranker_dedupe_parent:
            candidates = self._dedupe_parent_strict(candidates)

        final = list(candidates[:target_count])

        def fill_to_min_keep(*, allow_parent_duplicate: bool) -> None:
            seen_children = {row.get("child_id") for row in final}
            seen_parents = {row.get("parent_id") for row in final if row.get("parent_id")}
            for row in scored_rows:
                if row.get("child_id") in seen_children:
                    continue
                parent_id = row.get("parent_id")
                if (
                    self.reranker_dedupe_parent
                    and not allow_parent_duplicate
                    and parent_id
                    and parent_id in seen_parents
                ):
                    continue
                final.append(row)
                seen_children.add(row.get("child_id"))
                if parent_id:
                    seen_parents.add(parent_id)
                if len(final) >= min_keep:
                    break

        if len(final) < min_keep:
            fill_to_min_keep(allow_parent_duplicate=False)
        if len(final) < min_keep:
            fill_to_min_keep(allow_parent_duplicate=True)

        for row in final:
            row["reranker_filter_min_score"] = self.reranker_min_score
            row["reranker_filter_min_keep"] = min_keep
            row["reranker_filter_max_keep"] = target_count
            row["reranker_filter_parent_dedupe"] = self.reranker_dedupe_parent
        return final[:target_count]

    def search_vector_only(self, query: str, corpus_ids: Optional[List[str]] = None, top_k: Optional[int] = None) -> List[dict]:
        """Chromaのdense vector検索だけで根拠候補を返す。BM25/リランカー/parent間引きは行わない。"""
        top_k = top_k or self.top_k_final
        allowed = self._allowed_corpus_ids(corpus_ids)
        if allowed == set():
            return []
        try:
            q_emb = self.embedding.embed([query])[0]
            where = {"corpus_id": {"$in": list(allowed)}} if allowed is not None else None
            res = self.collection.query(
                query_embeddings=[q_emb],
                n_results=max(top_k * 3, top_k),
                where=where,
                include=["metadatas", "documents", "distances"],
            )
        except Exception:
            return []

        rows = []
        ids = res.get("ids", [[]])[0]
        distances = res.get("distances", [[]])[0]
        for rank, (cid, dist) in enumerate(zip(ids, distances), start=1):
            child = self.child_by_id.get(cid)
            if not child:
                continue
            if not self._is_current_document(child):
                continue
            dense_score = 1.0 / (1.0 + float(dist))
            rows.append(
                self._row_from_child(
                    child,
                    dense_score,
                    dense_score=dense_score,
                    retrieval_mode="vector",
                    retrieval_rank=rank,
                )
            )
        return rows[:top_k]

    def _dense_scores(self, query: str, allowed: Optional[set]) -> Dict[str, float]:
        dense_scores: Dict[str, float] = {}
        try:
            q_emb = self.embedding.embed([query])[0]
            where = {"corpus_id": {"$in": list(allowed)}} if allowed is not None else None
            res = self.collection.query(
                query_embeddings=[q_emb],
                n_results=max(self.top_k_dense * 3, self.top_k_dense),
                where=where,
                include=["metadatas", "documents", "distances"],
            )
            ids = res.get("ids", [[]])[0]
            distances = res.get("distances", [[]])[0]
            for cid, dist in zip(ids, distances):
                child = self.child_by_id.get(cid)
                if not child or not self._is_current_document(child):
                    continue
                dense_scores[cid] = 1.0 / (1.0 + float(dist))
        except Exception:
            # インデックス未作成やEmbedding未起動でも、BM25だけで最低限動かす
            dense_scores = {}
        return dense_scores

    def _bm25_scores(self, query: str, allowed: Optional[set]) -> Dict[str, float]:
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
                if not self._is_current_document(child):
                    continue
                bm25_scores[cid] = float(scores[i]) / float(max_score)
        return bm25_scores

    def search_hybrid_no_reranker(
        self,
        query: str,
        corpus_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        diversify: bool = True,
    ) -> List[dict]:
        """dense vector + BM25のハイブリッド検索結果を返す。リランカーは使わない。"""
        top_k = top_k or self.top_k_final
        allowed = self._allowed_corpus_ids(corpus_ids)
        if allowed == set():
            return []

        dense_scores = self._dense_scores(query, allowed)
        bm25_scores = self._bm25_scores(query, allowed)

        merged_ids = set(dense_scores) | set(bm25_scores)
        rows = []
        for cid in merged_ids:
            child = self.child_by_id.get(cid)
            if not child:
                continue
            if not self._is_current_document(child):
                continue
            score = self.dense_weight * dense_scores.get(cid, 0.0) + self.bm25_weight * bm25_scores.get(cid, 0.0)
            rows.append(
                self._row_from_child(
                    child,
                    score,
                    dense_score=dense_scores.get(cid, 0.0),
                    bm25_score=bm25_scores.get(cid, 0.0),
                    retrieval_mode="hybrid",
                )
            )

        rows.sort(key=lambda x: x["score"], reverse=True)
        if diversify:
            rows = self._diversify_by_parent(rows, top_k)
        else:
            rows = rows[:top_k]
        for rank, row in enumerate(rows, start=1):
            row["retrieval_rank"] = rank
        return rows

    def search(self, query: str, corpus_ids: Optional[List[str]] = None, top_k: Optional[int] = None) -> List[dict]:
        top_k = top_k or self.top_k_final
        if corpus_ids is not None and len(corpus_ids) == 0:
            return []

        rows = self.search_hybrid_no_reranker(
            query,
            corpus_ids=corpus_ids,
            top_k=max(top_k, self.reranker_top_k, self.top_k_dense + self.top_k_bm25),
            diversify=False,
        )
        rows = self._rerank(query, rows, top_k)
        if self.reranker_enabled and any("rerank_score" in row for row in rows):
            rows = self._apply_reranker_filters(rows, top_k)
        else:
            rows = self._diversify_by_parent(rows, top_k)
        for rank, row in enumerate(rows, start=1):
            row["retrieval_rank"] = rank
            row["retrieval_mode"] = "hybrid_reranker" if self.reranker_enabled else "hybrid"

        return rows
