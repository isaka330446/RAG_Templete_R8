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


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    debug: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    log_id: Optional[int] = None
