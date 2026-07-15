from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DOC_TARGETS = [BASE_DIR / "README.md", BASE_DIR / "docs"]
TOKENS = [
    "http" + "://",
    "https" + "://",
    "127" + ".0.0.1",
    "local" + "host",
    "0" + ".0.0.0",
    "uvicorn api.main:app",
    "app_fasthtml_admin_common.core:app",
    "embedding" + ".base_url",
    "app_fasthtml_modern" + "_popup",
    "RAG_EVIDENCE" + "_UI_MODE",
]


def iter_markdown_files() -> list[Path]:
    files: list[Path] = []
    for target in DOC_TARGETS:
        if target.is_file():
            files.append(target)
        elif target.exists():
            files.extend(path for path in target.rglob("*.md") if path.is_file())
    return sorted(files)


def main() -> int:
    violations: list[str] = []
    for path in iter_markdown_files():
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            for token in TOKENS:
                if token in line:
                    violations.append(f"{path.relative_to(BASE_DIR)}:{line_no}: {token}")
    if violations:
        print("Stale URL or startup instructions were found in docs:")
        print("\n".join(violations))
        return 1
    print("ok: no stale URL/startup instructions found in docs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
