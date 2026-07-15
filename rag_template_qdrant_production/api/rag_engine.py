# 承認済みQAキャッシュ、検索、LLM回答、ログ保存をまとめるRAG本体です。
import uuid
from typing import List, Optional, Dict, Any
from api.answer_cache import AnswerCacheStore
from api.config import load_settings
from api.logging_store import SQLiteLogStore
from api.llm_client import OpenAICompatibleLLM
from api.prompt import SYSTEM_PROMPT, build_retrieval_query, build_user_prompt
from api.retriever import HybridRetriever


class RAGEngine:
    def __init__(self):
        settings = load_settings()
        self.retriever = HybridRetriever()
        self.llm = OpenAICompatibleLLM()
        self.log_store = SQLiteLogStore()
        self.answer_cache = AnswerCacheStore()
        self.max_parent_context_chars = int(settings.get("retrieval", {}).get("max_parent_context_chars", 18000))
        self.max_history_messages = int(settings.get("chat", {}).get("max_history_messages", 10))

    def reload_retriever(self) -> dict:
        self.retriever = HybridRetriever()
        return self.retriever.release

    def _normalize_history(self, history: Optional[List[dict]]) -> List[dict]:
        rows = []
        for item in history or []:
            role = item.get("role")
            content = str(item.get("content", "")).strip()
            if role in {"user", "assistant"} and content:
                rows.append({"role": role, "content": content})
        return rows[-self.max_history_messages :]

    def _normalize_cache_sources(self, evidence: List[dict]) -> List[dict]:
        sources = []
        for idx, item in enumerate(evidence or [], start=1):
            child_text = item.get("child_text") or item.get("text") or ""
            row = {
                "corpus_id": str(item.get("corpus_id") or ""),
                "parent_id": str(item.get("parent_id") or ""),
                "child_id": str(item.get("child_id") or f"approved_evidence_{idx}"),
                "title": item.get("title"),
                "heading_path": item.get("heading_path"),
                "child_text": child_text,
                "parent_text": item.get("parent_text") or "",
                "score": float(item.get("score", 1.0) or 0.0),
                "source_file": item.get("source_file"),
                "search_tags": item.get("search_tags") or [],
                "forms": item.get("forms") or [],
            }
            if row["corpus_id"] and row["parent_id"] and row["child_id"] and row["child_text"]:
                sources.append(row)
        return sources

    def _find_approved_cache_match(
        self,
        retrieval_query: str,
        corpus_ids: Optional[List[str]],
        show_debug: bool,
    ) -> tuple[Optional[dict], Optional[dict]]:
        try:
            query_embedding = self.retriever.embedding.embed([retrieval_query])[0]
            match = self.answer_cache.find_match(
                query_embedding,
                corpus_ids=corpus_ids,
                corpus_version=self.retriever.corpus_version,
                index_version=self.retriever.index_version,
            )
            return match, None
        except Exception as exc:
            if show_debug:
                return None, {"cache_lookup_error": str(exc)}
            return None, None

    def ask(
        self,
        question: str,
        corpus_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        show_debug: bool = False,
        session_id: Optional[str] = None,
        history: Optional[List[dict]] = None,
    ) -> Dict[str, Any]:
        session_id = session_id or str(uuid.uuid4())
        normalized_history = self._normalize_history(history)
        retrieval_query = build_retrieval_query(question, normalized_history)
        cache_match, cache_debug = self._find_approved_cache_match(retrieval_query, corpus_ids, show_debug)
        if cache_match:
            response_sources = self._normalize_cache_sources(cache_match.get("evidence", []))
            debug = None
            if show_debug:
                debug = {
                    "answer_source": "approved_qa_cache",
                    "qa_cache_id": cache_match.get("id"),
                    "cache_similarity": cache_match.get("similarity"),
                    "cache_threshold": self.answer_cache.high_similarity_threshold,
                    "corpus_version": cache_match.get("corpus_version"),
                    "index_version": cache_match.get("index_version"),
                    "active_release": self.retriever.release,
                    "retrieval_query": retrieval_query,
                }
            log_id = self.log_store.log_ask(
                session_id=session_id,
                question=question,
                answer=cache_match["answer"],
                corpus_ids=corpus_ids,
                top_k=top_k,
                history=normalized_history,
                sources=response_sources,
                debug=debug,
            )
            return {
                "answer": cache_match["answer"],
                "sources": response_sources,
                "debug": debug,
                "session_id": session_id,
                "log_id": log_id,
                "answer_source": "approved_qa_cache",
                "cache_hit": True,
                "qa_cache_id": cache_match.get("id"),
                "cache_similarity": cache_match.get("similarity"),
            }

        sources = self.retriever.search(retrieval_query, corpus_ids=corpus_ids, top_k=top_k)

        contexts = []
        remaining_chars = max(0, self.max_parent_context_chars)
        limit_context = self.max_parent_context_chars > 0
        for s in sources:
            parent_text = s.get("parent_text") or ""
            if limit_context:
                if remaining_chars <= 0:
                    parent_text = ""
                elif len(parent_text) > remaining_chars:
                    parent_text = parent_text[:remaining_chars] + "\n...[省略]"
                remaining_chars = max(0, remaining_chars - len(parent_text))
            contexts.append({
                "corpus_id": s.get("corpus_id"),
                "source_file": s.get("source_file"),
                "heading_path": s.get("heading_path"),
                "parent_id": s.get("parent_id"),
                "child_text": s.get("child_text") or s.get("text", ""),
                "parent_text": parent_text,
                "forms": s.get("forms", []),
            })

        if not contexts:
            result = {
                "answer": "該当する根拠文書が見つかりませんでした。検索対象文書を確認してください。",
                "sources": [],
                "debug": {"reason": "no_context", "active_release": self.retriever.release} if show_debug else None,
                "session_id": session_id,
                "log_id": None,
                "answer_source": "rag",
                "cache_hit": False,
                "qa_cache_id": None,
                "cache_similarity": None,
            }
            if result["debug"] and cache_debug:
                result["debug"].update(cache_debug)
            result["log_id"] = self.log_store.log_ask(
                session_id=session_id,
                question=question,
                answer=result["answer"],
                corpus_ids=corpus_ids,
                top_k=top_k,
                history=normalized_history,
                sources=[],
                debug=result["debug"],
            )
            return result

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, contexts, normalized_history)},
        ]
        answer = self.llm.chat(messages)

        response_sources = [
            {**s, "parent_text": c.get("parent_text", s.get("parent_text", ""))}
            for s, c in zip(sources, contexts)
        ]
        debug = {
            "context_count": len(contexts),
            "max_parent_context_chars": self.max_parent_context_chars,
            "retrieval_query": retrieval_query,
            "active_release": self.retriever.release,
        } if show_debug else None
        if debug and cache_debug:
            debug.update(cache_debug)
        log_id = self.log_store.log_ask(
            session_id=session_id,
            question=question,
            answer=answer,
            corpus_ids=corpus_ids,
            top_k=top_k,
            history=normalized_history,
            sources=response_sources,
            debug=debug,
        )

        return {
            "answer": answer,
            "sources": response_sources,
            "debug": debug,
            "session_id": session_id,
            "log_id": log_id,
            "answer_source": "rag",
            "cache_hit": False,
            "qa_cache_id": None,
            "cache_similarity": None,
        }
