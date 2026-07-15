# FastAPIのリクエスト/レスポンスで使うPydanticスキーマ定義です。
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$", description="会話ロール")
    content: str = Field(..., min_length=1, max_length=4000, description="メッセージ本文")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="ユーザー質問")
    corpus_ids: Optional[List[str]] = Field(default=None, description="検索対象corpus_id。未指定ならenabled全件")
    top_k: Optional[int] = Field(default=None, ge=1, le=50, description="最終的にLLMへ渡す根拠数")
    show_debug: bool = False
    session_id: Optional[str] = Field(default=None, description="会話セッションID")
    history: List[ChatMessage] = Field(default_factory=list, description="直前までの会話履歴")


class SourceChunk(BaseModel):
    corpus_id: str
    parent_id: str
    child_id: str
    title: Optional[str] = None
    heading_path: Optional[str] = None
    child_text: str
    parent_text: Optional[str] = None
    score: float
    source_file: Optional[str] = None
    source_url: Optional[str] = None
    document_type: Optional[str] = None
    document_id: Optional[str] = None
    document_series: Optional[str] = None
    version_date: Optional[str] = None
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    valid_status: Optional[str] = None
    source_site: Optional[str] = None
    search_tags: List[str] = Field(default_factory=list)
    forms: List[Dict[str, Any]] = Field(default_factory=list)


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    debug: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    log_id: Optional[int] = None
    answer_source: str = "rag"
    cache_hit: bool = False
    qa_cache_id: Optional[int] = None
    cache_similarity: Optional[float] = None


class HallucinationReportRequest(BaseModel):
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    log_id: Optional[int] = None
    comment: str = ""


class EvidenceSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    corpus_ids: Optional[List[str]] = None
    top_k: int = Field(default=8, ge=1, le=50)


class SearchTagUpdateRequest(BaseModel):
    search_tags: List[str] = Field(default_factory=list)
    reload_retriever: bool = True


class ApprovedQARequest(BaseModel):
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    corpus_version: Optional[str] = None
    index_version: Optional[str] = None
    approved_by: str = ""
    source_report_id: Optional[int] = None
    memo: str = ""
    generate_aliases: bool = True
    alias_texts: List[str] = Field(default_factory=list)


class ApprovedQAUpdateRequest(BaseModel):
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    corpus_version: Optional[str] = None
    index_version: Optional[str] = None
    approved_by: str = ""
    memo: str = ""
    status: str = Field(default="approved", pattern="^(approved|disabled)$")


class QAAliasAddRequest(BaseModel):
    aliases: List[str] = Field(default_factory=list)
    alias_type: str = Field(default="admin_alias", pattern="^(admin_alias|llm_paraphrase|normalized)$")
    status: str = Field(default="active", pattern="^(active|disabled)$")
    memo: str = ""
    force_active_conflict: bool = False


class QAAliasUpdateRequest(BaseModel):
    alias_text: Optional[str] = None
    status: Optional[str] = Field(default=None, pattern="^(active|disabled)$")
    memo: str = ""


class QAAliasGenerateRequest(BaseModel):
    max_aliases: int = Field(default=8, ge=1, le=20)
    replace_existing_generated: bool = False
    dry_run: bool = False
    status: str = Field(default="active", pattern="^(active|disabled)$")


class QAAliasBackfillRequest(BaseModel):
    ensure_original: bool = True
    generate_llm_aliases: bool = False
    only_without_llm_aliases: bool = True
    limit: int = Field(default=50, ge=1, le=1000)
    dry_run: bool = True
    corpus_version: Optional[str] = None
    index_version: Optional[str] = None
    max_aliases_per_qa: int = Field(default=8, ge=1, le=20)


class QAMatchDebugRequest(BaseModel):
    question: str = Field(..., min_length=1)
    corpus_ids: Optional[List[str]] = None
    corpus_version: Optional[str] = None
    index_version: Optional[str] = None
    top_n: int = Field(default=10, ge=1, le=50)
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    apply_llm_intent_judge: bool = False
    include_disabled: bool = False


class ReportStatusUpdate(BaseModel):
    status: str = Field(..., pattern="^(open|resolved|ignored)$")


class ReportAnalysisUpdate(BaseModel):
    status: Optional[str] = Field(default=None, pattern="^(open|resolved|ignored)$")
    issue_type: Optional[str] = Field(
        default=None,
        pattern="^(retrieval_miss|corpus_missing|generation_error|out_of_scope|user_misunderstanding|other)$",
    )
    resolution_type: Optional[str] = Field(
        default=None,
        pattern="^(qa_created|search_tag_updated|document_update_needed|prompt_or_generation_fix_needed|marked_out_of_scope|no_action|other)$",
    )
    admin_memo: str = ""
    linked_child_id: Optional[str] = None
    resolved_qa_id: Optional[int] = None


class QASimilarRequest(BaseModel):
    question: str = Field(..., min_length=1)
    corpus_ids: Optional[List[str]] = None
    corpus_version: Optional[str] = None
    index_version: Optional[str] = None
    top_n: int = Field(default=5, ge=1, le=20)
    threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    include_disabled_qa: bool = False
    include_disabled_aliases: bool = False


class QATestMatchRequest(BaseModel):
    question: str = Field(..., min_length=1)
    corpus_ids: Optional[List[str]] = None
    corpus_version: Optional[str] = None
    index_version: Optional[str] = None
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    include_disabled: bool = False


class LogTrendReportRequest(BaseModel):
    days: int = Field(default=7, ge=1, le=90)
    sample_limit: int = Field(default=1000, ge=1, le=5000)
    top_n: int = Field(default=20, ge=1, le=100)
    low_score_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
