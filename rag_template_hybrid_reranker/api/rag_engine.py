# ハイブリッド検索、reranker、LLM回答、ログ保存をまとめるRAG本体です。
import uuid
from typing import List, Optional, Dict, Any
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
        self.max_parent_context_chars = int(settings.get("retrieval", {}).get("max_parent_context_chars", 18000))
        self.max_history_messages = int(settings.get("chat", {}).get("max_history_messages", 10))

    def _normalize_history(self, history: Optional[List[dict]]) -> List[dict]:
        rows = []
        for item in history or []:
            role = item.get("role")
            content = str(item.get("content", "")).strip()
            if role in {"user", "assistant"} and content:
                rows.append({"role": role, "content": content})
        return rows[-self.max_history_messages :]

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
                "debug": {"reason": "no_context"} if show_debug else None,
                "session_id": session_id,
                "log_id": None,
            }
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
        } if show_debug else None
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
        }
