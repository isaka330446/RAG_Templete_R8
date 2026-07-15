# 承認済みQAキャッシュ、検索、LLM回答、ログ保存をまとめるRAG本体です。
import uuid
import time
from typing import Any, Dict, Iterator, List, Optional
from api.answer_cache import AnswerCacheStore, parse_json_object_from_llm
from api.config import load_settings
from api.logging_store import SQLiteLogStore
from api.llm_client import OpenAICompatibleLLM
from api.prompt import SYSTEM_PROMPT, build_retrieval_query, build_user_prompt
from api.retriever import HybridRetriever
from api.text_cleaning import clean_rag_text


class RAGEngine:
    def __init__(self):
        settings = load_settings()
        self.retriever = HybridRetriever()
        self.llm = OpenAICompatibleLLM()
        alias_llm_settings = settings.get("alias_llm", {})
        self.alias_llm = OpenAICompatibleLLM(
            api_key=alias_llm_settings.get("api_key"),
            model=alias_llm_settings.get("model"),
            temperature=float(alias_llm_settings.get("temperature", 0.0)),
            max_tokens=int(alias_llm_settings.get("max_tokens", 800)),
            url_key="alias_llm_base_url",
        )
        if alias_llm_settings.get("timeout_sec"):
            self.alias_llm.timeout_sec = int(alias_llm_settings.get("timeout_sec"))
        self.log_store = SQLiteLogStore()
        self.answer_cache = AnswerCacheStore()
        self.max_parent_context_chars = int(settings.get("retrieval", {}).get("max_parent_context_chars", 18000))
        self.max_history_messages = int(settings.get("chat", {}).get("max_history_messages", 10))

    def reload_retriever(self) -> dict:
        self.retriever = HybridRetriever()
        return {
            "corpus_version": getattr(self.retriever, "corpus_version", None),
            "index_version": getattr(self.retriever, "index_version", None),
            "collection_name": getattr(self.retriever, "collection_name", None),
        }

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
            child_text = clean_rag_text(item.get("child_text") or item.get("text") or "")
            row = {
                "corpus_id": str(item.get("corpus_id") or ""),
                "parent_id": str(item.get("parent_id") or ""),
                "child_id": str(item.get("child_id") or f"approved_evidence_{idx}"),
                "title": item.get("title"),
                "heading_path": item.get("heading_path"),
                "child_text": child_text,
                "parent_text": clean_rag_text(item.get("parent_text") or ""),
                "score": float(item.get("score", 1.0) or 0.0),
                "source_file": item.get("source_file"),
                "source_url": item.get("source_url"),
                "document_type": item.get("document_type"),
                "document_id": item.get("document_id"),
                "taxanswer_no": item.get("taxanswer_no"),
                "tsutatsu_name": item.get("tsutatsu_name"),
                "law_basis_date": item.get("law_basis_date"),
                "valid_from": item.get("valid_from"),
                "valid_until": item.get("valid_until"),
                "valid_status": item.get("valid_status"),
                "source_site": item.get("source_site"),
                "search_tags": item.get("search_tags") or [],
                "forms": item.get("forms") or [],
            }
            if row["corpus_id"] and row["parent_id"] and row["child_id"] and row["child_text"]:
                sources.append(row)
        return sources

    def _find_approved_cache_match(
        self,
        question: str,
        retrieval_query: str,
        corpus_ids: Optional[List[str]],
        show_debug: bool,
    ) -> tuple[Optional[dict], Optional[dict]]:
        try:
            cache_lookup_query = question
            query_embedding = self.retriever.embedding.embed([cache_lookup_query])[0]
            debug = self.answer_cache.match_debug(
                query_embedding,
                question=cache_lookup_query,
                corpus_ids=corpus_ids,
            )
            debug["retrieval_query"] = retrieval_query
            debug["cache_lookup_query"] = cache_lookup_query
            match = debug.get("best") if debug.get("decision") == "hit" else None
            can_judge, skip_reason = self._can_apply_cache_intent_judge(debug)
            debug["intent_judge_gray_min"] = self.answer_cache.intent_judge_gray_min
            debug["intent_judge_applied"] = False
            debug["intent_judge_skipped_reason"] = None if (can_judge and self.answer_cache.enable_llm_intent_judge) else (skip_reason or "disabled")
            if match is None and self.answer_cache.enable_llm_intent_judge and can_judge:
                debug["intent_judge_applied"] = True
                judged = self._judge_cache_intent(cache_lookup_query, debug.get("best") or {})
                debug["llm_intent_judge_result"] = judged.get("same_intent")
                debug["llm_intent_judge_reason"] = judged.get("reason")
                if judged.get("same_intent"):
                    debug["decision"] = "llm_judge_hit"
                    debug["miss_reason"] = None
                    match = debug.get("best")
                    if match:
                        match["match_method"] = "llm_intent_judge"
            if match:
                return match, debug
            return None, debug
        except Exception as exc:
            if show_debug:
                return None, {"cache_lookup_error": str(exc)}
            return None, None

    def debug_cache_match(
        self,
        question: str,
        *,
        corpus_ids: Optional[List[str]] = None,
        corpus_version: Optional[str] = None,
        index_version: Optional[str] = None,
        top_n: int = 10,
        threshold: Optional[float] = None,
        apply_llm_intent_judge: bool = False,
        include_disabled: bool = False,
    ) -> dict[str, Any]:
        query_embedding = self.retriever.embedding.embed([question])[0]
        debug = self.answer_cache.match_debug(
            query_embedding,
            question=question,
            corpus_ids=corpus_ids,
            corpus_version=corpus_version,
            index_version=index_version,
            top_n=top_n,
            threshold=threshold,
            include_disabled_qa=include_disabled,
            include_disabled_aliases=include_disabled,
        )
        debug["apply_llm_intent_judge"] = apply_llm_intent_judge
        can_judge, skip_reason = self._can_apply_cache_intent_judge(debug)
        debug["intent_judge_gray_min"] = self.answer_cache.intent_judge_gray_min
        debug["intent_judge_applied"] = False
        debug["intent_judge_skipped_reason"] = None if (can_judge and apply_llm_intent_judge and self.answer_cache.enable_llm_intent_judge) else (skip_reason or "disabled")
        if apply_llm_intent_judge and self.answer_cache.enable_llm_intent_judge and can_judge:
            debug["intent_judge_applied"] = True
            judged = self._judge_cache_intent(question, debug.get("best") or {})
            debug["llm_intent_judge_result"] = judged.get("same_intent")
            debug["llm_intent_judge_reason"] = judged.get("reason")
            if judged.get("same_intent"):
                debug["decision"] = "llm_judge_hit"
                debug["miss_reason"] = None
                if isinstance(debug.get("best"), dict):
                    debug["best"]["match_method"] = "llm_intent_judge"
            else:
                debug["decision"] = "llm_judge_reject"
                debug["miss_reason"] = "llm_judge_rejected"
        return debug

    def _can_apply_cache_intent_judge(self, debug: dict[str, Any]) -> tuple[bool, str]:
        if debug.get("decision") != "gray":
            return False, str(debug.get("miss_reason") or "not_gray")
        miss_reason = str(debug.get("miss_reason") or "")
        if miss_reason != "below_accept_threshold":
            return False, miss_reason or "unsupported_gray_reason"
        best = debug.get("best") or {}
        if not best:
            return False, "no_candidate"
        try:
            similarity = float(best.get("similarity") or 0.0)
        except (TypeError, ValueError):
            similarity = 0.0
        try:
            margin = float(best.get("margin") or 0.0)
        except (TypeError, ValueError):
            margin = 0.0
        if similarity < float(self.answer_cache.intent_judge_gray_min):
            return False, "below_intent_judge_gray_min"
        if margin < float(self.answer_cache.margin_threshold):
            return False, "margin_too_small"
        return True, ""

    def _judge_cache_intent(self, question: str, candidate: dict) -> dict[str, Any]:
        if not candidate:
            return {"same_intent": False, "reason": "no candidate"}
        prompt = f"""次のユーザー質問に、候補QAの承認済み回答をそのまま返して安全かだけを判定してください。
税目違い、制度違い、条件追加、回答範囲拡大、個別判断が必要な場合はfalseです。
JSONのみで返してください。
形式: {{"same_intent": true, "reason": "..."}}

ユーザー質問:
{question}

候補QAの正式質問:
{candidate.get("question")}

候補QAの回答:
{candidate.get("answer")}
"""
        try:
            text = self.alias_llm.chat([
                {"role": "system", "content": "あなたは税務RAGキャッシュの安全判定を行います。JSONだけを返してください。"},
                {"role": "user", "content": prompt},
            ])
            payload = parse_json_object_from_llm(text)
            return {
                "same_intent": bool(payload.get("same_intent")),
                "reason": str(payload.get("reason") or ""),
            }
        except Exception as exc:
            return {"same_intent": False, "reason": f"judge_failed: {exc}"}

    @staticmethod
    def _cache_log_fields(cache_debug: Optional[dict]) -> dict[str, Any]:
        debug = cache_debug if isinstance(cache_debug, dict) else {}
        best = debug.get("best") if isinstance(debug.get("best"), dict) else {}
        return {
            "cache_candidate_qa_id": best.get("qa_id") or best.get("id"),
            "cache_candidate_alias_id": best.get("alias_id") or best.get("matched_alias_id"),
            "cache_candidate_similarity": best.get("similarity"),
            "cache_miss_reason": debug.get("miss_reason"),
            "cache_match_method": best.get("match_method") or debug.get("cache_match_method"),
        }

    @staticmethod
    def _latency_ms(started: float) -> int:
        return int((time.perf_counter() - started) * 1000)

    def _retrieve_for_mode(
        self,
        mode: str,
        query: str,
        corpus_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[dict]:
        mode_key = str(mode or "").strip().lower().replace("-", "_")
        if mode_key == "vector":
            return self.retriever.search_vector_only(query, corpus_ids=corpus_ids, top_k=top_k)
        if mode_key == "hybrid":
            return self.retriever.search_hybrid_no_reranker(query, corpus_ids=corpus_ids, top_k=top_k)
        if mode_key in {"hybrid_reranker", "reranker", "hybrid_rerank", "hybrid_reranked"}:
            return self.retriever.search(query, corpus_ids=corpus_ids, top_k=top_k)
        raise ValueError(f"unsupported retrieval mode: {mode}")

    def _build_contexts_from_sources(self, sources: List[dict]) -> List[dict]:
        contexts = []
        remaining_chars = max(0, self.max_parent_context_chars)
        limit_context = self.max_parent_context_chars > 0
        for s in sources:
            parent_text = clean_rag_text(s.get("parent_text") or "")
            if limit_context:
                if remaining_chars <= 0:
                    parent_text = ""
                elif len(parent_text) > remaining_chars:
                    parent_text = parent_text[:remaining_chars] + "\n...[省略]"
                remaining_chars = max(0, remaining_chars - len(parent_text))
            contexts.append({
                "corpus_id": s.get("corpus_id"),
                "source_file": s.get("source_file"),
                "source_url": s.get("source_url"),
                "document_type": s.get("document_type"),
                "document_id": s.get("document_id"),
                "tsutatsu_name": s.get("tsutatsu_name"),
                "taxanswer_no": s.get("taxanswer_no"),
                "law_basis_date": s.get("law_basis_date"),
                "valid_from": s.get("valid_from"),
                "valid_until": s.get("valid_until"),
                "valid_status": s.get("valid_status"),
                "heading_path": clean_rag_text(s.get("heading_path")),
                "parent_id": s.get("parent_id"),
                "child_text": clean_rag_text(s.get("child_text") or s.get("text", "")),
                "parent_text": parent_text,
                "forms": s.get("forms", []),
            })
        return contexts

    @staticmethod
    def _sources_with_context_text(sources: List[dict], contexts: List[dict]) -> List[dict]:
        return [
            {
                **s,
                "child_text": clean_rag_text(c.get("child_text", s.get("child_text", s.get("text", "")))),
                "parent_text": clean_rag_text(c.get("parent_text", s.get("parent_text", ""))),
            }
            for s, c in zip(sources, contexts)
        ]

    def answer_with_retrieval_mode(
        self,
        question: str,
        retrieval_mode: str,
        corpus_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        show_debug: bool = False,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        session_id = session_id or str(uuid.uuid4())
        mode_key = "hybrid_reranker" if retrieval_mode in {"reranker", "hybrid_rerank", "hybrid_reranked"} else retrieval_mode
        retrieval_query = build_retrieval_query(question, [])
        try:
            sources = self._retrieve_for_mode(mode_key, retrieval_query, corpus_ids=corpus_ids, top_k=top_k)
            contexts = self._build_contexts_from_sources(sources)
            answer_source = f"rag_eval_{mode_key}"
            if not contexts:
                result = {
                    "answer": "該当する根拠文書が見つかりませんでした。検索対象文書を確認してください。",
                    "sources": [],
                    "debug": {
                        "reason": "no_context",
                        "retrieval_mode": mode_key,
                        "retrieval_query": retrieval_query,
                        "cache_lookup_query": question,
                    } if show_debug else None,
                    "session_id": session_id,
                    "log_id": None,
                    "answer_source": answer_source,
                    "cache_hit": False,
                    "qa_cache_id": None,
                    "cache_similarity": None,
                    "retrieval_mode": mode_key,
                }
                result["log_id"] = self.log_store.log_ask(
                    session_id=session_id,
                    question=question,
                    answer=result["answer"],
                    corpus_ids=corpus_ids,
                    top_k=top_k,
                    history=[],
                    sources=[],
                    debug=result["debug"],
                    answer_source=answer_source,
                    cache_hit=False,
                    latency_ms=self._latency_ms(started),
                )
                return result

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(question, contexts, [])},
            ]
            answer = self.llm.chat(messages)
            response_sources = self._sources_with_context_text(sources, contexts)
            debug = {
                "context_count": len(contexts),
                "max_parent_context_chars": self.max_parent_context_chars,
                "retrieval_mode": mode_key,
                "retrieval_query": retrieval_query,
                "cache_lookup_query": question,
                "cache_bypassed": True,
            } if show_debug else None
            log_id = self.log_store.log_ask(
                session_id=session_id,
                question=question,
                answer=answer,
                corpus_ids=corpus_ids,
                top_k=top_k,
                history=[],
                sources=response_sources,
                debug=debug,
                answer_source=answer_source,
                cache_hit=False,
                latency_ms=self._latency_ms(started),
            )
            return {
                "answer": answer,
                "sources": response_sources,
                "debug": debug,
                "session_id": session_id,
                "log_id": log_id,
                "answer_source": answer_source,
                "cache_hit": False,
                "qa_cache_id": None,
                "cache_similarity": None,
                "retrieval_mode": mode_key,
            }
        except Exception as exc:
            self.log_store.log_ask(
                session_id=session_id,
                question=question,
                answer="",
                corpus_ids=corpus_ids,
                top_k=top_k,
                history=[],
                sources=[],
                debug={
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "retrieval_mode": mode_key,
                    "retrieval_query": retrieval_query,
                },
                answer_source="error",
                cache_hit=False,
                latency_ms=self._latency_ms(started),
                error_type=type(exc).__name__,
            )
            raise

    def ask(
        self,
        question: str,
        corpus_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        show_debug: bool = False,
        session_id: Optional[str] = None,
        history: Optional[List[dict]] = None,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        session_id = session_id or str(uuid.uuid4())
        try:
            return self._ask_impl(
                question=question,
                corpus_ids=corpus_ids,
                top_k=top_k,
                show_debug=show_debug,
                session_id=session_id,
                history=history,
            )
        except Exception as exc:
            self.log_store.log_ask(
                session_id=session_id,
                question=question,
                answer="",
                corpus_ids=corpus_ids,
                top_k=top_k,
                history=self._normalize_history(history),
                sources=[],
                debug={"error": str(exc), "error_type": type(exc).__name__},
                answer_source="error",
                cache_hit=False,
                latency_ms=self._latency_ms(started),
                error_type=type(exc).__name__,
            )
            raise

    def _ask_impl(
        self,
        question: str,
        corpus_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        show_debug: bool = False,
        session_id: Optional[str] = None,
        history: Optional[List[dict]] = None,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        session_id = session_id or str(uuid.uuid4())
        normalized_history = self._normalize_history(history)
        retrieval_query = build_retrieval_query(question, normalized_history)
        cache_match, cache_debug = self._find_approved_cache_match(question, retrieval_query, corpus_ids, show_debug)
        if cache_match:
            response_sources = self._normalize_cache_sources(cache_match.get("evidence", []))
            debug = None
            if show_debug:
                debug = {
                    "answer_source": "approved_qa_cache",
                    "qa_cache_id": cache_match.get("id"),
                    "cache_similarity": cache_match.get("similarity"),
                    "cache_threshold": self.answer_cache.semantic_accept_threshold,
                    "cache_lookup_query": question,
                    "matched_alias_id": cache_match.get("matched_alias_id") or cache_match.get("alias_id"),
                    "matched_alias_text": cache_match.get("matched_alias_text") or cache_match.get("alias_text"),
                    "matched_alias_type": cache_match.get("matched_alias_type") or cache_match.get("alias_type"),
                    "cache_match_method": cache_match.get("match_method"),
                    "corpus_version": cache_match.get("corpus_version"),
                    "index_version": cache_match.get("index_version"),
                    "retrieval_query": retrieval_query,
                }
                if cache_debug:
                    debug.update(cache_debug)
            log_id = self.log_store.log_ask(
                session_id=session_id,
                question=question,
                answer=cache_match["answer"],
                corpus_ids=corpus_ids,
                top_k=top_k,
                history=normalized_history,
                sources=response_sources,
                debug=debug,
                answer_source="approved_qa_cache",
                cache_hit=True,
                qa_cache_id=cache_match.get("id"),
                cache_similarity=cache_match.get("similarity"),
                cache_candidate_qa_id=cache_match.get("qa_id") or cache_match.get("id"),
                cache_candidate_alias_id=cache_match.get("alias_id") or cache_match.get("matched_alias_id"),
                cache_candidate_similarity=cache_match.get("similarity"),
                cache_match_method=cache_match.get("match_method"),
                latency_ms=self._latency_ms(started),
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
            parent_text = clean_rag_text(s.get("parent_text") or "")
            if limit_context:
                if remaining_chars <= 0:
                    parent_text = ""
                elif len(parent_text) > remaining_chars:
                    parent_text = parent_text[:remaining_chars] + "\n...[省略]"
                remaining_chars = max(0, remaining_chars - len(parent_text))
            contexts.append({
                "corpus_id": s.get("corpus_id"),
                "source_file": s.get("source_file"),
                "source_url": s.get("source_url"),
                "document_type": s.get("document_type"),
                "document_id": s.get("document_id"),
                "tsutatsu_name": s.get("tsutatsu_name"),
                "taxanswer_no": s.get("taxanswer_no"),
                "law_basis_date": s.get("law_basis_date"),
                "valid_from": s.get("valid_from"),
                "valid_until": s.get("valid_until"),
                "valid_status": s.get("valid_status"),
                "heading_path": clean_rag_text(s.get("heading_path")),
                "parent_id": s.get("parent_id"),
                "child_text": clean_rag_text(s.get("child_text") or s.get("text", "")),
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
                "answer_source": "rag",
                "cache_hit": False,
                "qa_cache_id": None,
                "cache_similarity": None,
            }
            if result["debug"]:
                result["debug"]["retrieval_query"] = retrieval_query
                result["debug"]["cache_lookup_query"] = question
                if cache_debug:
                    result["debug"].update(cache_debug)
            cache_log_fields = self._cache_log_fields(cache_debug)
            result["log_id"] = self.log_store.log_ask(
                session_id=session_id,
                question=question,
                answer=result["answer"],
                corpus_ids=corpus_ids,
                top_k=top_k,
                history=normalized_history,
                sources=[],
                debug=result["debug"],
                answer_source="rag",
                cache_hit=False,
                latency_ms=self._latency_ms(started),
                **cache_log_fields,
            )
            return result

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, contexts, normalized_history)},
        ]
        answer = self.llm.chat(messages)

        response_sources = self._sources_with_context_text(sources, contexts)
        debug = {
            "context_count": len(contexts),
            "max_parent_context_chars": self.max_parent_context_chars,
            "retrieval_query": retrieval_query,
        } if show_debug else None
        if debug:
            debug["cache_lookup_query"] = question
            if cache_debug:
                debug.update(cache_debug)
        cache_log_fields = self._cache_log_fields(cache_debug)
        log_id = self.log_store.log_ask(
            session_id=session_id,
            question=question,
            answer=answer,
            corpus_ids=corpus_ids,
            top_k=top_k,
            history=normalized_history,
            sources=response_sources,
            debug=debug,
            answer_source="rag",
            cache_hit=False,
            latency_ms=self._latency_ms(started),
            **cache_log_fields,
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

    def ask_events(
        self,
        question: str,
        corpus_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        show_debug: bool = False,
        session_id: Optional[str] = None,
        history: Optional[List[dict]] = None,
    ) -> Iterator[Dict[str, Any]]:
        started = time.perf_counter()
        session_id = session_id or str(uuid.uuid4())
        try:
            for event in self._ask_events_impl(
                question=question,
                corpus_ids=corpus_ids,
                top_k=top_k,
                show_debug=show_debug,
                session_id=session_id,
                history=history,
            ):
                yield event
        except Exception as exc:
            self.log_store.log_ask(
                session_id=session_id,
                question=question,
                answer="",
                corpus_ids=corpus_ids,
                top_k=top_k,
                history=self._normalize_history(history),
                sources=[],
                debug={"error": str(exc), "error_type": type(exc).__name__},
                answer_source="error",
                cache_hit=False,
                latency_ms=self._latency_ms(started),
                error_type=type(exc).__name__,
            )
            raise

    def _ask_events_impl(
        self,
        question: str,
        corpus_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        show_debug: bool = False,
        session_id: Optional[str] = None,
        history: Optional[List[dict]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """RAGの主要工程をUIへ通知しながら、最後にask互換の結果を返す。"""
        started = time.perf_counter()
        session_id = session_id or str(uuid.uuid4())
        yield {
            "type": "progress",
            "stage": "accepted",
            "message": "質問を受け付けました",
            "session_id": session_id,
        }

        normalized_history = self._normalize_history(history)
        retrieval_query = build_retrieval_query(question, normalized_history)
        yield {
            "type": "progress",
            "stage": "cache_lookup",
            "message": "承認済みQAを確認しています",
            "session_id": session_id,
        }

        cache_match, cache_debug = self._find_approved_cache_match(question, retrieval_query, corpus_ids, show_debug)
        if cache_match:
            yield {
                "type": "progress",
                "stage": "cache_hit",
                "message": "承認済みQAから回答を準備しています",
                "session_id": session_id,
            }
            response_sources = self._normalize_cache_sources(cache_match.get("evidence", []))
            debug = None
            if show_debug:
                debug = {
                    "answer_source": "approved_qa_cache",
                    "qa_cache_id": cache_match.get("id"),
                    "cache_similarity": cache_match.get("similarity"),
                    "cache_threshold": self.answer_cache.semantic_accept_threshold,
                    "cache_lookup_query": question,
                    "matched_alias_id": cache_match.get("matched_alias_id") or cache_match.get("alias_id"),
                    "matched_alias_text": cache_match.get("matched_alias_text") or cache_match.get("alias_text"),
                    "matched_alias_type": cache_match.get("matched_alias_type") or cache_match.get("alias_type"),
                    "cache_match_method": cache_match.get("match_method"),
                    "corpus_version": cache_match.get("corpus_version"),
                    "index_version": cache_match.get("index_version"),
                    "retrieval_query": retrieval_query,
                }
                if cache_debug:
                    debug.update(cache_debug)
            yield {
                "type": "progress",
                "stage": "logging",
                "message": "回答ログを保存しています",
                "session_id": session_id,
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
                answer_source="approved_qa_cache",
                cache_hit=True,
                qa_cache_id=cache_match.get("id"),
                cache_similarity=cache_match.get("similarity"),
                cache_candidate_qa_id=cache_match.get("qa_id") or cache_match.get("id"),
                cache_candidate_alias_id=cache_match.get("alias_id") or cache_match.get("matched_alias_id"),
                cache_candidate_similarity=cache_match.get("similarity"),
                cache_match_method=cache_match.get("match_method"),
                latency_ms=self._latency_ms(started),
            )
            yield {
                "type": "done",
                "stage": "done",
                "message": "回答が完了しました",
                "session_id": session_id,
                "data": {
                    "answer": cache_match["answer"],
                    "sources": response_sources,
                    "debug": debug,
                    "session_id": session_id,
                    "log_id": log_id,
                    "answer_source": "approved_qa_cache",
                    "cache_hit": True,
                    "qa_cache_id": cache_match.get("id"),
                    "cache_similarity": cache_match.get("similarity"),
                },
            }
            return

        yield {
            "type": "progress",
            "stage": "retrieval",
            "message": "関連文書を検索しています",
            "session_id": session_id,
        }
        sources = self.retriever.search(retrieval_query, corpus_ids=corpus_ids, top_k=top_k)

        yield {
            "type": "progress",
            "stage": "context",
            "message": "根拠チャンクを確認しています",
            "session_id": session_id,
            "source_count": len(sources),
        }
        contexts = []
        remaining_chars = max(0, self.max_parent_context_chars)
        limit_context = self.max_parent_context_chars > 0
        for s in sources:
            parent_text = clean_rag_text(s.get("parent_text") or "")
            if limit_context:
                if remaining_chars <= 0:
                    parent_text = ""
                elif len(parent_text) > remaining_chars:
                    parent_text = parent_text[:remaining_chars] + "\n...[省略]"
                remaining_chars = max(0, remaining_chars - len(parent_text))
            contexts.append({
                "corpus_id": s.get("corpus_id"),
                "source_file": s.get("source_file"),
                "source_url": s.get("source_url"),
                "document_type": s.get("document_type"),
                "document_id": s.get("document_id"),
                "tsutatsu_name": s.get("tsutatsu_name"),
                "taxanswer_no": s.get("taxanswer_no"),
                "law_basis_date": s.get("law_basis_date"),
                "valid_from": s.get("valid_from"),
                "valid_until": s.get("valid_until"),
                "valid_status": s.get("valid_status"),
                "heading_path": clean_rag_text(s.get("heading_path")),
                "parent_id": s.get("parent_id"),
                "child_text": clean_rag_text(s.get("child_text") or s.get("text", "")),
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
                "answer_source": "rag",
                "cache_hit": False,
                "qa_cache_id": None,
                "cache_similarity": None,
            }
            if result["debug"]:
                result["debug"]["retrieval_query"] = retrieval_query
                result["debug"]["cache_lookup_query"] = question
                if cache_debug:
                    result["debug"].update(cache_debug)
            cache_log_fields = self._cache_log_fields(cache_debug)
            yield {
                "type": "progress",
                "stage": "logging",
                "message": "回答ログを保存しています",
                "session_id": session_id,
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
                answer_source="rag",
                cache_hit=False,
                latency_ms=self._latency_ms(started),
                **cache_log_fields,
            )
            yield {
                "type": "done",
                "stage": "done",
                "message": "回答が完了しました",
                "session_id": session_id,
                "data": result,
            }
            return

        yield {
            "type": "progress",
            "stage": "llm",
            "message": "LLMで回答を作成しています",
            "session_id": session_id,
        }
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, contexts, normalized_history)},
        ]
        answer = self.llm.chat(messages)

        response_sources = self._sources_with_context_text(sources, contexts)
        debug = {
            "context_count": len(contexts),
            "max_parent_context_chars": self.max_parent_context_chars,
            "retrieval_query": retrieval_query,
        } if show_debug else None
        if debug:
            debug["cache_lookup_query"] = question
            if cache_debug:
                debug.update(cache_debug)
        cache_log_fields = self._cache_log_fields(cache_debug)

        yield {
            "type": "progress",
            "stage": "logging",
            "message": "回答ログを保存しています",
            "session_id": session_id,
        }
        log_id = self.log_store.log_ask(
            session_id=session_id,
            question=question,
            answer=answer,
            corpus_ids=corpus_ids,
            top_k=top_k,
            history=normalized_history,
            sources=response_sources,
            debug=debug,
            answer_source="rag",
            cache_hit=False,
            latency_ms=self._latency_ms(started),
            **cache_log_fields,
        )

        yield {
            "type": "done",
            "stage": "done",
            "message": "回答が完了しました",
            "session_id": session_id,
            "data": {
                "answer": answer,
                "sources": response_sources,
                "debug": debug,
                "session_id": session_id,
                "log_id": log_id,
                "answer_source": "rag",
                "cache_hit": False,
                "qa_cache_id": None,
                "cache_similarity": None,
            },
        }
