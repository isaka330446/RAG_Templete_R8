from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import sys
import types
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvalDependency:
    import_name: str
    package_name: str
    required_for_ragas: bool = True


EVAL_DEPENDENCIES: tuple[EvalDependency, ...] = (
    EvalDependency("ragas", "ragas"),
    EvalDependency("datasets", "datasets"),
    EvalDependency("langchain_openai", "langchain-openai"),
    EvalDependency("langchain_community", "langchain-community"),
    EvalDependency("openai", "openai"),
    EvalDependency("deepeval", "deepeval", required_for_ragas=False),
)


def _find_spec(name: str) -> Any:
    try:
        return importlib.util.find_spec(name)
    except (ImportError, AttributeError, ValueError):
        return None


def _package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return ""


def _clean_kwargs(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value not in (None, "")}


def _instantiate_with_fallbacks(cls: Any, attempts: list[dict[str, Any]]) -> Any:
    errors: list[str] = []
    for kwargs in attempts:
        try:
            return cls(**_clean_kwargs(kwargs))
        except TypeError as exc:
            errors.append(str(exc))
    raise RuntimeError(
        f"Could not initialize {getattr(cls, '__name__', cls)} for RAGAS evaluation. "
        f"Tried {len(attempts)} constructor shapes. Last errors: {' | '.join(errors[-3:])}"
    )


def inspect_eval_dependencies() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dep in EVAL_DEPENDENCIES:
        available = _find_spec(dep.import_name) is not None
        rows.append(
            {
                "import_name": dep.import_name,
                "package_name": dep.package_name,
                "available": available,
                "version": _package_version(dep.package_name) if available else "",
                "required_for_ragas": dep.required_for_ragas,
            }
        )
    return rows


def missing_ragas_packages() -> list[str]:
    missing: list[str] = []
    for row in inspect_eval_dependencies():
        if row["required_for_ragas"] and not row["available"]:
            missing.append(str(row["package_name"]))
    return missing


def format_dependency_report(rows: list[dict[str, Any]]) -> str:
    lines = []
    for row in rows:
        status = "ok" if row["available"] else "missing"
        version = f" {row['version']}" if row.get("version") else ""
        required = "required" if row["required_for_ragas"] else "optional"
        lines.append(f"- {row['package_name']}: {status}{version} ({required})")
    return "\n".join(lines)


def install_hint(packages: list[str] | None = None) -> str:
    names = " ".join(packages or [])
    if names:
        direct = f"python -m pip install -U {names}"
    else:
        direct = "python -m pip install -U -r requirements_eval.txt"
    return (
        "Install or refresh the evaluation dependencies, then retry:\n"
        "  python -m pip install -U -r requirements_eval.txt\n"
        f"  {direct}"
    )


def _install_vertexai_import_shim() -> bool:
    module_name = "langchain_community.chat_models.vertexai"
    if _find_spec(module_name) is not None:
        return False
    if _find_spec("langchain_community") is None:
        return False

    try:
        importlib.import_module("langchain_community.chat_models")
    except Exception:
        # Some langchain-community versions may no longer expose chat_models as a package.
        # The shim is only for RAGAS import-time compatibility; this app does not use VertexAI.
        parent = types.ModuleType("langchain_community.chat_models")
        sys.modules.setdefault("langchain_community.chat_models", parent)

    shim = types.ModuleType(module_name)

    class ChatVertexAI:  # pragma: no cover - only used if someone explicitly selects VertexAI.
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "ChatVertexAI is not available in this evaluation environment. "
                "This project uses a local OpenAI-compatible evaluation endpoint; "
                "do not select VertexAI for RAGAS evaluation."
            )

    shim.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = shim
    return True


def ensure_ragas_import_compat() -> dict[str, Any]:
    rows = inspect_eval_dependencies()
    missing = [
        str(row["package_name"])
        for row in rows
        if row["required_for_ragas"] and not row["available"]
    ]
    if missing:
        raise RuntimeError(
            "RAGAS dependencies are incomplete.\n"
            f"Missing packages: {', '.join(missing)}\n"
            f"{install_hint(missing)}\n\n"
            f"Current dependency status:\n{format_dependency_report(rows)}"
        )
    shim_installed = _install_vertexai_import_shim()
    return {
        "dependency_status": rows,
        "vertexai_import_shim_installed": shim_installed,
    }


def build_openai_compatible_ragas_models(
    *,
    llm_base_url: str,
    llm_model: str,
    llm_api_key: str,
    embedding_base_url: str,
    embedding_model: str,
    embedding_api_key: str,
) -> dict[str, Any]:
    if not llm_model:
        raise RuntimeError(
            "Evaluation LLM model is not set. Set eval_llm.model or alias_llm.model "
            "in config/settings.json, or pass --model."
        )
    if not embedding_model:
        raise RuntimeError(
            "Evaluation embedding model is not set. Set embedding.model in "
            "config/settings.json, or pass --embedding-model."
        )

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    try:
        from ragas.llms import LangchainLLMWrapper
    except ImportError:
        LangchainLLMWrapper = None
    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except ImportError:
        LangchainEmbeddingsWrapper = None

    chat_model = _instantiate_with_fallbacks(
        ChatOpenAI,
        [
            {
                "model": llm_model,
                "base_url": llm_base_url,
                "api_key": llm_api_key,
                "temperature": 0.0,
            },
            {
                "model_name": llm_model,
                "openai_api_base": llm_base_url,
                "openai_api_key": llm_api_key,
                "temperature": 0.0,
            },
            {
                "model": llm_model,
                "openai_api_base": llm_base_url,
                "openai_api_key": llm_api_key,
                "temperature": 0.0,
            },
        ],
    )
    embedding_model_obj = _instantiate_with_fallbacks(
        OpenAIEmbeddings,
        [
            {
                "model": embedding_model,
                "base_url": embedding_base_url,
                "api_key": embedding_api_key,
            },
            {
                "model": embedding_model,
                "openai_api_base": embedding_base_url,
                "openai_api_key": embedding_api_key,
            },
            {
                "model": embedding_model,
                "openai_api_base": embedding_base_url,
                "api_key": embedding_api_key,
            },
        ],
    )
    return {
        "llm": LangchainLLMWrapper(chat_model) if LangchainLLMWrapper else chat_model,
        "embeddings": (
            LangchainEmbeddingsWrapper(embedding_model_obj)
            if LangchainEmbeddingsWrapper
            else embedding_model_obj
        ),
        "wrapped_llm": bool(LangchainLLMWrapper),
        "wrapped_embeddings": bool(LangchainEmbeddingsWrapper),
    }
