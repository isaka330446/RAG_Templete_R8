from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
DEFAULT_SETTINGS_PATH = CONFIG_DIR / "settings.json"
EXAMPLE_SETTINGS_PATH = CONFIG_DIR / "settings.example.json"


DEFAULT_SETTINGS: Dict[str, Any] = {
    "api": {
        "allow_credentials": False,
    },
    "admin": {
        "variant_key": "analyst_saas",
    },
    "llm": {
        "api_key": "dummy",
        "model": "local-model",
        "temperature": 0.1,
        "max_tokens": 2048,
        "timeout_sec": 180,
    },
    "embedding": {
        "api_key": "dummy",
        "model": "BAAI/bge-m3",
        "timeout_sec": 180,
    },
    "alias_llm": {
        "api_key": "dummy",
        "model": "small-local-model",
        "temperature": 0.0,
        "max_tokens": 800,
        "timeout_sec": 60,
    },
    "retrieval": {
        "chroma_path": "indexes/chroma",
        "collection_name": "rag_children",
        "collection_prefix": "rag_",
        "top_k_dense": 20,
        "top_k_bm25": 20,
        "top_k_final": 8,
        "bm25_weight": 0.35,
        "dense_weight": 0.65,
        "max_parent_context_chars": 18000,
    },
    "reranker": {
        "enabled": False,
        "top_k": 8,
        "timeout_sec": 60,
    },
    "chat": {
        "max_history_messages": 10,
    },
    "urls": {
        "allow_runtime_api_url_override": False,
    },
    "logging": {
        "enabled": True,
        "required": True,
        "sqlite_path": "logs/rag_chat_logs.sqlite",
        "busy_timeout_ms": 10000,
        "connect_timeout_sec": 30,
        "journal_mode": "DELETE",
        "startup_write_check": True,
        "max_source_preview_chars": 500,
        "store_full_source_text": False,
    },
    "answer_cache": {
        "enabled": True,
        "required": True,
        "sqlite_path": "logs/answer_cache.sqlite",
        "busy_timeout_ms": 10000,
        "connect_timeout_sec": 30,
        "journal_mode": "DELETE",
        "startup_write_check": True,
        "high_similarity_threshold": 0.88,
        "semantic_accept_threshold": 0.88,
        "semantic_gray_threshold": 0.82,
        "margin_threshold": 0.03,
        "corpus_version": "nta_taxanswer_inheritance_gift_v1",
        "index_version": "bge-m3_chroma_v1",
        "enable_alias_search": True,
        "enable_alias_generation": True,
        "max_aliases_per_qa": 8,
        "alias_default_status": "active",
        "generated_alias_default_status": "active",
        "enable_llm_intent_judge": False,
        "intent_judge_gray_min": 0.82,
        "alias_generation_temperature": 0.0,
        "alias_generation_max_tokens": 1200,
    },
    "search_tags": {
        "enabled_in_retrieval": True,
        "admin_edit_enabled": False,
        "show_improvement_candidates": False,
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


def _settings_path() -> Path:
    configured = os.getenv("SETTINGS_PATH")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else BASE_DIR / path
    return DEFAULT_SETTINGS_PATH


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def _local_settings() -> dict[str, Any]:
    path = _settings_path()
    if not path.exists():
        raise RuntimeError(f"{path} が存在しません。config/settings.example.json をコピーして config/settings.json を作成してください")
    return _read_json(path)


def load_settings() -> Dict[str, Any]:
    settings = copy.deepcopy(DEFAULT_SETTINGS)

    if EXAMPLE_SETTINGS_PATH.exists():
        _deep_update(settings, _read_json(EXAMPLE_SETTINGS_PATH))

    local_path = _settings_path()
    if local_path.exists():
        _deep_update(settings, _read_json(local_path))

    non_url_overrides = {
        ("llm", "api_key"): os.getenv("LLM_API_KEY"),
        ("llm", "model"): os.getenv("LLM_MODEL"),
        ("alias_llm", "api_key"): os.getenv("ALIAS_LLM_API_KEY"),
        ("alias_llm", "model"): os.getenv("ALIAS_LLM_MODEL"),
        ("embedding", "api_key"): os.getenv("EMBEDDING_API_KEY"),
        ("embedding", "model"): os.getenv("EMBEDDING_MODEL"),
        ("logging", "sqlite_path"): os.getenv("RAG_LOG_SQLITE_PATH"),
        ("answer_cache", "sqlite_path"): os.getenv("ANSWER_CACHE_SQLITE_PATH"),
        ("answer_cache", "corpus_version"): os.getenv("ANSWER_CACHE_CORPUS_VERSION"),
        ("answer_cache", "index_version"): os.getenv("ANSWER_CACHE_INDEX_VERSION"),
    }
    for (section, key), value in non_url_overrides.items():
        if value:
            settings.setdefault(section, {})[key] = value

    bool_env = {
        ("urls", "allow_runtime_api_url_override"): "RAG_ALLOW_RUNTIME_API_URL_OVERRIDE",
        ("reranker", "enabled"): "RERANKER_ENABLED",
        ("logging", "enabled"): "RAG_LOG_ENABLED",
        ("logging", "required"): "RAG_LOG_REQUIRED",
        ("logging", "startup_write_check"): "RAG_LOG_STARTUP_WRITE_CHECK",
        ("logging", "store_full_source_text"): "RAG_LOG_STORE_FULL_SOURCE_TEXT",
        ("answer_cache", "enabled"): "ANSWER_CACHE_ENABLED",
        ("answer_cache", "required"): "ANSWER_CACHE_REQUIRED",
        ("answer_cache", "startup_write_check"): "ANSWER_CACHE_STARTUP_WRITE_CHECK",
        ("answer_cache", "enable_llm_intent_judge"): "ANSWER_CACHE_ENABLE_LLM_INTENT_JUDGE",
        ("answer_cache", "enable_alias_generation"): "ANSWER_CACHE_ENABLE_ALIAS_GENERATION",
        ("search_tags", "enabled_in_retrieval"): "SEARCH_TAGS_ENABLED",
        ("search_tags", "admin_edit_enabled"): "SEARCH_TAGS_ADMIN_EDIT_ENABLED",
        ("search_tags", "show_improvement_candidates"): "SEARCH_TAGS_SHOW_IMPROVEMENT_CANDIDATES",
    }
    for (section, key), env_name in bool_env.items():
        value = _env_bool(env_name)
        if value is not None:
            settings.setdefault(section, {})[key] = value

    numeric_env = {
        ("logging", "busy_timeout_ms"): ("RAG_LOG_BUSY_TIMEOUT_MS", int),
        ("logging", "connect_timeout_sec"): ("RAG_LOG_CONNECT_TIMEOUT_SEC", int),
        ("logging", "max_source_preview_chars"): ("RAG_LOG_MAX_SOURCE_PREVIEW_CHARS", int),
        ("answer_cache", "semantic_accept_threshold"): ("ANSWER_CACHE_SEMANTIC_ACCEPT_THRESHOLD", float),
        ("answer_cache", "semantic_gray_threshold"): ("ANSWER_CACHE_SEMANTIC_GRAY_THRESHOLD", float),
        ("answer_cache", "margin_threshold"): ("ANSWER_CACHE_MARGIN_THRESHOLD", float),
        ("answer_cache", "intent_judge_gray_min"): ("ANSWER_CACHE_INTENT_JUDGE_GRAY_MIN", float),
        ("answer_cache", "max_aliases_per_qa"): ("ANSWER_CACHE_MAX_ALIASES_PER_QA", int),
        ("alias_llm", "temperature"): ("ALIAS_LLM_TEMPERATURE", float),
        ("alias_llm", "max_tokens"): ("ALIAS_LLM_MAX_TOKENS", int),
        ("alias_llm", "timeout_sec"): ("ALIAS_LLM_TIMEOUT_SEC", int),
    }
    for (section, key), (env_name, caster) in numeric_env.items():
        value = os.getenv(env_name)
        if value:
            settings.setdefault(section, {})[key] = caster(value)

    high_similarity_threshold = os.getenv("ANSWER_CACHE_HIGH_SIMILARITY_THRESHOLD")
    if high_similarity_threshold:
        value = float(high_similarity_threshold)
        settings.setdefault("answer_cache", {})["high_similarity_threshold"] = value
        settings.setdefault("answer_cache", {})["semantic_accept_threshold"] = value

    generated_alias_status = os.getenv("ANSWER_CACHE_GENERATED_ALIAS_DEFAULT_STATUS")
    if generated_alias_status:
        settings.setdefault("answer_cache", {})["generated_alias_default_status"] = generated_alias_status

    return settings


def _missing_config_message(path: str) -> str:
    return f"config/settings.json の {path} が未設定です"


def _configured_urls() -> dict[str, Any]:
    urls = _local_settings().get("urls")
    if not isinstance(urls, dict):
        raise RuntimeError(_missing_config_message("urls"))
    return urls


def get_required_url(name: str) -> str:
    value = _configured_urls().get(name)
    if not value:
        raise RuntimeError(_missing_config_message(f"urls.{name}"))
    return str(value).rstrip("/")


def get_required_url_value(name: str) -> str:
    value = _configured_urls().get(name)
    if value is None or value == "":
        raise RuntimeError(_missing_config_message(f"urls.{name}"))
    return str(value)


def get_url_number(name: str) -> int:
    value = _configured_urls().get(name)
    if value is None or value == "":
        raise RuntimeError(_missing_config_message(f"urls.{name}"))
    return int(value)


def get_cors_allow_origins() -> list[str]:
    value = _configured_urls().get("cors_allow_origins")
    if not isinstance(value, list) or not value:
        raise RuntimeError(_missing_config_message("urls.cors_allow_origins"))
    return [str(item) for item in value]


def get_nta_url(name: str) -> str:
    value = _configured_urls().get("nta", {}).get(name)
    if not value:
        raise RuntimeError(_missing_config_message(f"urls.nta.{name}"))
    return str(value).rstrip("/")


def get_nta_value(name: str) -> Any:
    value = _configured_urls().get("nta", {}).get(name)
    if value is None:
        raise RuntimeError(_missing_config_message(f"urls.nta.{name}"))
    return value


def rag_api_base_url() -> str:
    return get_required_url("rag_api_base_url")


def rag_ask_url() -> str:
    return rag_api_base_url().rstrip("/") + "/ask"


def rag_ask_stream_url() -> str:
    return rag_api_base_url().rstrip("/") + "/ask_stream"


def allow_runtime_api_url_override() -> bool:
    return bool(load_settings().get("urls", {}).get("allow_runtime_api_url_override", False))


def runtime_rag_api_base_url(candidate: str | None = None) -> str:
    if candidate and allow_runtime_api_url_override():
        return str(candidate).rstrip("/")
    return rag_api_base_url()


def project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return BASE_DIR / path
