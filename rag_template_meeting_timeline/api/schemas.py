# FastAPIのリクエスト/レスポンスで使うPydanticスキーマ定義です。
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$", description="会話ロール")
    content: str = Field(..., min_length=1, description="メッセージ本文")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="ユーザー質問")
    corpus_ids: Optional[List[str]] = Field(default=None, description="検索対象corpus_id。未指定ならenabled全件")
    top_k: Optional[int] = Field(default=None, ge=1, le=50, description="最終的にLLMへ渡す根拠数")
    show_debug: bool = False
    session_id: Optional[str] = Field(default=None, description="会話セッションID")
    history: List[ChatMessage] = Field(default_factory=list, description="直前までの会話履歴")
    answer_mode: str = Field(default="auto", pattern="^(auto|rag|timeline)$")
    event_type: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    status: Optional[str] = None
    owner: Optional[str] = None
    source_type: Optional[str] = Field(default=None, pattern="^(slide|minutes)$")


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
    search_tags: List[str] = Field(default_factory=list)
    forms: List[Dict[str, Any]] = Field(default_factory=list)
    document_type: Optional[str] = None
    source_type: Optional[str] = None
    meeting_id: Optional[str] = None
    meeting_name: Optional[str] = None
    meeting_date: Optional[str] = None
    agenda: Optional[str] = None
    topic: Optional[str] = None
    section_title: Optional[str] = None
    slide_no: Optional[int] = None
    slide_title: Optional[str] = None
    content_type: Optional[str] = None
    auxiliary_context: bool = False
    auxiliary_reason: Optional[str] = None


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


class ApprovedQAUpdateRequest(BaseModel):
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    corpus_version: Optional[str] = None
    index_version: Optional[str] = None
    approved_by: str = ""
    memo: str = ""
    status: str = Field(default="approved", pattern="^(approved|disabled)$")


class ReportStatusUpdate(BaseModel):
    status: str = Field(..., pattern="^(open|resolved|ignored)$")


class LogTrendReportRequest(BaseModel):
    days: int = Field(default=7, ge=1, le=90)
    sample_limit: int = Field(default=1000, ge=1, le=5000)
    top_n: int = Field(default=20, ge=1, le=100)
    low_score_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
