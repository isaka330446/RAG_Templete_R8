# 設定ファイル、環境変数、プロジェクト相対パスを読み解く共通処理です。
import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, List


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"


DEFAULT_SETTINGS: Dict[str, Any] = {
    "api": {
        "cors_allow_origins": [
            "http://localhost:8501",
            "http://127.0.0.1:8501",
        ],
        "allow_credentials": False,
    },
    "llm": {
        "base_url": "http://127.0.0.1:8001/v1",
        "api_key": "dummy",
        "model": "local-model",
        "temperature": 0.1,
        "max_tokens": 2048,
        "timeout_sec": 180,
    },
    "embedding": {
        "base_url": "http://127.0.0.1:8002/v1",
        "api_key": "dummy",
        "model": "BAAI/bge-m3",
        "timeout_sec": 180,
    },
    "retrieval": {
        "chroma_path": "indexes/chroma",
        "collection_name": "rag_children",
        "top_k_dense": 20,
        "top_k_bm25": 20,
        "top_k_final": 8,
        "bm25_weight": 0.35,
        "dense_weight": 0.65,
        "max_parent_context_chars": 18000,
    },
    "vector_db": {
        "provider": "qdrant",
        "chroma_path": "indexes/chroma",
        "qdrant": {
            "url": "http://127.0.0.1:6333",
            "api_key": "",
            "collection_name": "rag_children",
            "local_path": "",
            "prefer_grpc": False,
            "timeout_sec": 60,
            "distance": "cosine",
            "payload_indexes": ["corpus_id", "parent_id", "source_file"],
        },
    },
    "reranker": {
        "enabled": False,
        "base_url": "http://127.0.0.1:8003",
        "top_k": 8,
        "timeout_sec": 60,
    },
    "chat": {
        "max_history_messages": 10,
    },
    "logging": {
        "enabled": True,
        "sqlite_path": "logs/rag_chat_logs.sqlite",
    },
    "answer_cache": {
        "enabled": True,
        "sqlite_path": "logs/answer_cache.sqlite",
        "high_similarity_threshold": 0.88,
        "corpus_version": "default",
        "index_version": "default",
    },
    "release": {
        "manifest_path": "indexes/release_manifest.json",
        "default_corpus_version": "default",
        "default_index_version": "default",
    },
}


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _env_bool(name: str) -> bool | None:
    value = os.getenv(name)
    if value is None:
        return None
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _split_origins(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def load_settings() -> Dict[str, Any]:
    settings = copy.deepcopy(DEFAULT_SETTINGS)

    example_path = CONFIG_DIR / "settings.example.json"
    if example_path.exists():
        _deep_update(settings, json.loads(example_path.read_text(encoding="utf-8")))

    local_path = CONFIG_DIR / "settings.json"
    if local_path.exists():
        _deep_update(settings, json.loads(local_path.read_text(encoding="utf-8")))

    env_overrides = {
        ("llm", "base_url"): os.getenv("LLM_BASE_URL"),
        ("llm", "api_key"): os.getenv("LLM_API_KEY"),
        ("llm", "model"): os.getenv("LLM_MODEL"),
        ("embedding", "base_url"): os.getenv("EMBEDDING_BASE_URL"),
        ("embedding", "api_key"): os.getenv("EMBEDDING_API_KEY"),
        ("embedding", "model"): os.getenv("EMBEDDING_MODEL"),
        ("reranker", "base_url"): os.getenv("RERANKER_BASE_URL"),
        ("logging", "sqlite_path"): os.getenv("RAG_LOG_SQLITE_PATH"),
        ("answer_cache", "sqlite_path"): os.getenv("ANSWER_CACHE_SQLITE_PATH"),
        ("answer_cache", "corpus_version"): os.getenv("ANSWER_CACHE_CORPUS_VERSION"),
        ("answer_cache", "index_version"): os.getenv("ANSWER_CACHE_INDEX_VERSION"),
        ("vector_db", "provider"): os.getenv("VECTOR_DB_PROVIDER"),
        ("vector_db", "chroma_path"): os.getenv("VECTOR_DB_CHROMA_PATH"),
        ("vector_db", "qdrant", "url"): os.getenv("QDRANT_URL"),
        ("vector_db", "qdrant", "api_key"): os.getenv("QDRANT_API_KEY"),
        ("vector_db", "qdrant", "collection_name"): os.getenv("QDRANT_COLLECTION_NAME"),
        ("vector_db", "qdrant", "local_path"): os.getenv("QDRANT_LOCAL_PATH"),
        ("release", "manifest_path"): os.getenv("RELEASE_MANIFEST_PATH"),
        ("release", "default_corpus_version"): os.getenv("RELEASE_DEFAULT_CORPUS_VERSION"),
        ("release", "default_index_version"): os.getenv("RELEASE_DEFAULT_INDEX_VERSION"),
    }
    for path, value in env_overrides.items():
        if value:
            cursor = settings
            for key in path[:-1]:
                cursor = cursor.setdefault(key, {})
            cursor[path[-1]] = value

    reranker_enabled = _env_bool("RERANKER_ENABLED")
    if reranker_enabled is not None:
        settings.setdefault("reranker", {})["enabled"] = reranker_enabled

    qdrant_prefer_grpc = _env_bool("QDRANT_PREFER_GRPC")
    if qdrant_prefer_grpc is not None:
        settings.setdefault("vector_db", {}).setdefault("qdrant", {})["prefer_grpc"] = qdrant_prefer_grpc

    cors_origins = os.getenv("CORS_ALLOW_ORIGINS")
    if cors_origins:
        settings.setdefault("api", {})["cors_allow_origins"] = _split_origins(cors_origins)

    log_enabled = _env_bool("RAG_LOG_ENABLED")
    if log_enabled is not None:
        settings.setdefault("logging", {})["enabled"] = log_enabled

    answer_cache_enabled = _env_bool("ANSWER_CACHE_ENABLED")
    if answer_cache_enabled is not None:
        settings.setdefault("answer_cache", {})["enabled"] = answer_cache_enabled

    answer_cache_threshold = os.getenv("ANSWER_CACHE_HIGH_SIMILARITY_THRESHOLD")
    if answer_cache_threshold:
        settings.setdefault("answer_cache", {})["high_similarity_threshold"] = float(answer_cache_threshold)

    return settings


def project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return BASE_DIR / path
