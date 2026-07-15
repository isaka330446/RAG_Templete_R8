import re
import uuid
from typing import Any, Dict, List, Optional

from api.answer_cache import AnswerCacheStore
from api.config import load_settings
from api.logging_store import SQLiteLogStore
from api.llm_client import OpenAICompatibleLLM
from api.meeting_event_store import MeetingEventStore
from api.meeting_timeline import event_sources as build_event_sources
from api.meeting_timeline import format_timeline_answer
from api.prompt import SYSTEM_PROMPT, build_retrieval_query, build_user_prompt
from api.retriever import HybridRetriever


TIMELINE_KEYWORDS = (
    "時系列",
    "経緯",
    "これまでの流れ",
    "いつ決まった",
    "変遷",
    "前回から",
    "何が変わった",
    "宿題の状況",
    "未完了事項",
    "決定事項だけ",
    "方針変更だけ",
    "懸念事項の変遷",
)

EVENT_TYPE_KEYWORDS = [
    ("decision", ("決定事項", "決まった", "決定だけ", "いつ決まった")),
    ("action_item", ("宿題", "対応事項", "アクションアイテム")),
    ("pending", ("未完了", "保留", "ペンディング")),
    ("concern", ("懸念", "リスク")),
    ("change", ("方針変更", "変更")),
    ("rejection", ("却下", "見送り")),
    ("completion", ("完了", "対応済み")),
    ("discussion", ("議論", "論点")),
    ("proposal", ("提案", "案")),
    ("report", ("報告",)),
]


class RAGEngine:
    def __init__(self):
        settings = load_settings()
        self.retriever = HybridRetriever()
        self.llm = OpenAICompatibleLLM()
        self.log_store = SQLiteLogStore()
        self.answer_cache = AnswerCacheStore()
        self.event_store = MeetingEventStore()
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
        extra_keys = [
            "document_type",
            "source_type",
            "meeting_id",
            "meeting_name",
            "meeting_date",
            "agenda",
            "topic",
            "section_title",
            "slide_no",
            "slide_title",
            "content_type",
        ]
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
            for key in extra_keys:
                if item.get(key) is not None:
                    row[key] = item.get(key)
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
            match = self.answer_cache.find_match(query_embedding, corpus_ids=corpus_ids)
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
        answer_mode: str = "auto",
        event_type: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        status: Optional[str] = None,
        owner: Optional[str] = None,
        source_type: Optional[str] = None,
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

        normalized_mode = (answer_mode or "auto").lower()
        if normalized_mode in {"auto", "timeline"}:
            timeline_result = self._ask_timeline(
                question=question,
                retrieval_query=retrieval_query,
                corpus_ids=corpus_ids,
                top_k=top_k,
                show_debug=show_debug,
                session_id=session_id,
                history=normalized_history,
                explicit=normalized_mode == "timeline",
                event_type=event_type,
                date_from=date_from,
                date_to=date_to,
                status=status,
                owner=owner,
                source_type=source_type,
                cache_debug=cache_debug,
            )
            if timeline_result is not None:
                return timeline_result

        return self._ask_rag(
            question=question,
            retrieval_query=retrieval_query,
            corpus_ids=corpus_ids,
            top_k=top_k,
            show_debug=show_debug,
            session_id=session_id,
            history=normalized_history,
            cache_debug=cache_debug,
        )

    def _ask_rag(
        self,
        *,
        question: str,
        retrieval_query: str,
        corpus_ids: Optional[List[str]],
        top_k: Optional[int],
        show_debug: bool,
        session_id: str,
        history: List[dict],
        cache_debug: Optional[dict],
    ) -> Dict[str, Any]:
        sources = self.retriever.search(retrieval_query, corpus_ids=corpus_ids, top_k=top_k)
        contexts = self._build_contexts(sources)

        if not contexts:
            result = {
                "answer": "該当する根拠文書が見つかりませんでした。検索対象文書とインデックス作成状況を確認してください。",
                "sources": [],
                "debug": {"reason": "no_context"} if show_debug else None,
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
                history=history,
                sources=[],
                debug=result["debug"],
            )
            return result

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, contexts, history)},
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
        if debug and cache_debug:
            debug.update(cache_debug)
        log_id = self.log_store.log_ask(
            session_id=session_id,
            question=question,
            answer=answer,
            corpus_ids=corpus_ids,
            top_k=top_k,
            history=history,
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

    def _build_contexts(self, sources: list[dict]) -> list[dict]:
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
                "meeting_id": s.get("meeting_id"),
                "source_type": s.get("source_type"),
                "slide_no": s.get("slide_no"),
                "agenda": s.get("agenda"),
                "topic": s.get("topic"),
            })
        return contexts

    def _ask_timeline(
        self,
        *,
        question: str,
        retrieval_query: str,
        corpus_ids: Optional[List[str]],
        top_k: Optional[int],
        show_debug: bool,
        session_id: str,
        history: List[dict],
        explicit: bool,
        event_type: Optional[str],
        date_from: Optional[str],
        date_to: Optional[str],
        status: Optional[str],
        owner: Optional[str],
        source_type: Optional[str],
        cache_debug: Optional[dict],
    ) -> Optional[Dict[str, Any]]:
        if not explicit and not self._is_timeline_query(question):
            return None

        detected_event_type = self._detect_event_type(question, event_type)
        seed_sources = self._safe_search(retrieval_query, corpus_ids, top_k)
        meeting_ids = sorted({str(s.get("meeting_id")) for s in seed_sources if s.get("meeting_id")})
        topic = None if meeting_ids else self._extract_topic_candidate(question)

        events = self.event_store.query(
            meeting_ids=meeting_ids or None,
            topic=topic,
            event_type=detected_event_type,
            date_from=date_from,
            date_to=date_to,
            status=status,
            owner=owner,
            source_type=source_type,
            limit=120,
        )
        if not events and not meeting_ids:
            events = self.event_store.query(
                event_type=detected_event_type,
                date_from=date_from,
                date_to=date_to,
                status=status,
                owner=owner,
                source_type=source_type,
                limit=120,
            )

        if not events and not explicit:
            return None

        answer = format_timeline_answer(events)
        sources = build_event_sources(events)
        debug = {
            "answer_source": "meeting_timeline",
            "event_count": len(events),
            "event_type": detected_event_type,
            "meeting_ids": meeting_ids,
            "topic_candidate": topic,
            "date_from": date_from,
            "date_to": date_to,
            "source_type": source_type,
            "retrieval_query": retrieval_query,
        } if show_debug else None
        if debug and cache_debug:
            debug.update(cache_debug)

        log_id = self.log_store.log_ask(
            session_id=session_id,
            question=question,
            answer=answer,
            corpus_ids=corpus_ids,
            top_k=top_k,
            history=history,
            sources=sources,
            debug=debug,
        )
        return {
            "answer": answer,
            "sources": sources,
            "debug": debug,
            "session_id": session_id,
            "log_id": log_id,
            "answer_source": "meeting_timeline",
            "cache_hit": False,
            "qa_cache_id": None,
            "cache_similarity": None,
        }

    def _safe_search(self, query: str, corpus_ids: Optional[List[str]], top_k: Optional[int]) -> list[dict]:
        try:
            return self.retriever.search(query, corpus_ids=corpus_ids, top_k=top_k)
        except Exception:
            return []

    def _is_timeline_query(self, question: str) -> bool:
        return any(keyword in question for keyword in TIMELINE_KEYWORDS)

    def _detect_event_type(self, question: str, explicit_event_type: Optional[str]) -> Optional[str]:
        if explicit_event_type:
            return explicit_event_type
        for event_type, keywords in EVENT_TYPE_KEYWORDS:
            if any(keyword in question for keyword in keywords):
                return event_type
        return None

    def _extract_topic_candidate(self, question: str) -> Optional[str]:
        cleaned = question
        for keyword in TIMELINE_KEYWORDS:
            cleaned = cleaned.replace(keyword, " ")
        for _, keywords in EVENT_TYPE_KEYWORDS:
            for keyword in keywords:
                cleaned = cleaned.replace(keyword, " ")
        cleaned = re.sub(r"[?？。、「」『』についてを教えてください下さい]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned if len(cleaned) >= 2 else None

    def _format_timeline_answer(self, events: list[dict]) -> str:
        if not events:
            return (
                "## 1. 時系列サマリ\n"
                "該当するMeetingEventは見つかりませんでした。\n\n"
                "## 2. 詳細タイムライン表\n"
                "| 日付 | 会議名 | 種別 | トピック | 内容 | 担当 | 期限 | 根拠 |\n"
                "|---|---|---|---|---|---|---|---|\n\n"
                "## 3. 現在の状態\n確認できません。\n\n"
                "## 4. 未完了事項・宿題\n確認できません。\n\n"
                "## 5. 根拠一覧\nなし"
            )

        table_rows = []
        for event in events:
            table_rows.append(
                "| {date} | {meeting} | {etype} | {topic} | {summary} | {owner} | {due} | {source} |".format(
                    date=_escape_table(event.get("event_date") or event.get("meeting_date") or ""),
                    meeting=_escape_table(event.get("meeting_name") or ""),
                    etype=_escape_table(event.get("event_type") or ""),
                    topic=_escape_table(event.get("topic") or ""),
                    summary=_escape_table(event.get("event_summary") or ""),
                    owner=_escape_table(event.get("owner") or ""),
                    due=_escape_table(event.get("due_date") or ""),
                    source=_escape_table(_source_label(event)),
                )
            )

        current = self._current_state(events)
        open_items = [
            event for event in events
            if event.get("event_type") in {"action_item", "pending"} and not _is_done(event.get("status"))
        ]
        open_text = "\n".join(
            f"- {event.get('event_date') or event.get('meeting_date')}: {event.get('event_summary')} "
            f"(担当: {event.get('owner') or '未記載'} / 期限: {event.get('due_date') or '未記載'})"
            for event in open_items
        ) or "確認できる未完了事項・宿題はありません。"
        evidence_text = "\n".join(f"- {_source_label(event)}" for event in events) or "なし"

        return (
            f"## 1. 時系列サマリ\n"
            f"{len(events)}件のMeetingEventをevent_date、meeting_dateの昇順で整理しました。\n\n"
            f"## 2. 詳細タイムライン表\n"
            "| 日付 | 会議名 | 種別 | トピック | 内容 | 担当 | 期限 | 根拠 |\n"
            "|---|---|---|---|---|---|---|---|\n"
            + "\n".join(table_rows)
            + "\n\n"
            f"## 3. 現在の状態\n{current}\n\n"
            f"## 4. 未完了事項・宿題\n{open_text}\n\n"
            f"## 5. 根拠一覧\n{evidence_text}"
        )

    def _current_state(self, events: list[dict]) -> str:
        for event in reversed(events):
            if event.get("event_type") in {"decision", "change", "completion", "report"}:
                return (
                    f"{event.get('event_date') or event.get('meeting_date')}時点では、"
                    f"{event.get('event_summary')} と記録されています。"
                )
        last = events[-1]
        return f"最新イベントは {last.get('event_date') or last.get('meeting_date')} の {last.get('event_summary')} です。"

    def _event_sources(self, events: list[dict]) -> list[dict]:
        sources = []
        for idx, event in enumerate(events, start=1):
            ref = _first_source_ref(event)
            sources.append({
                "corpus_id": "meeting_events",
                "parent_id": str(event.get("event_id") or f"meeting_event_{idx}"),
                "child_id": f"{event.get('event_id') or idx}:event",
                "title": event.get("meeting_name"),
                "heading_path": f"{event.get('topic') or ''} > {event.get('event_type') or ''}",
                "child_text": event.get("event_summary") or "",
                "parent_text": "",
                "score": 1.0,
                "source_file": ref.get("source_file"),
                "search_tags": [],
                "forms": [],
                "document_type": ref.get("document_type"),
                "source_type": ref.get("source_type"),
                "meeting_id": event.get("meeting_id"),
                "meeting_name": event.get("meeting_name"),
                "meeting_date": event.get("meeting_date"),
                "agenda": ref.get("agenda"),
                "topic": event.get("topic"),
                "section_title": ref.get("section_title"),
                "slide_no": ref.get("slide_no"),
                "slide_title": ref.get("slide_title"),
            })
        return sources


def _escape_table(value: Any) -> str:
    text = str(value or "").replace("\n", " ")
    return text.replace("|", "\\|")


def _is_done(status: Any) -> bool:
    text = str(status or "").strip().lower()
    return text in {"done", "closed", "completed", "完了", "対応済み", "終了"}


def _first_source_ref(event: dict) -> dict:
    refs = event.get("source_refs") or []
    for ref in refs:
        if isinstance(ref, dict):
            return ref
    return {}


def _source_label(event: dict) -> str:
    ref = _first_source_ref(event)
    source_type = ref.get("source_type") or ""
    source_file = ref.get("source_file") or ""
    if source_type == "slide":
        return f"{source_file} / Slide {ref.get('slide_no')}: {ref.get('slide_title')}"
    if source_type == "minutes":
        return f"{source_file} / {ref.get('agenda') or ''} / {ref.get('section_title') or ''} / {ref.get('item_type') or ''}"
    return source_file or str(event.get("event_id") or "")
