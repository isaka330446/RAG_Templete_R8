from __future__ import annotations

import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
TARGETS = [
    "api",
    "app",
    "app_dev_eval",
    "app_fasthtml_modern",
    "app_fasthtml_admin_common",
    "scripts",
]
EXCLUDED_DIR_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    "data",
    "chunks",
    "docs",
}
EXCLUDED_FILES = {
    BASE_DIR / "README.md",
    BASE_DIR / "config" / "settings.json",
    BASE_DIR / "config" / "settings.example.json",
}
TOKENS = [
    "http" + "://",
    "https" + "://",
    "local" + "host",
    "127" + ".0.0.1",
    "0" + ".0.0.0",
]


def is_excluded(path: Path) -> bool:
    if path in EXCLUDED_FILES:
        return True
    parts = set(path.relative_to(BASE_DIR).parts)
    return bool(parts & EXCLUDED_DIR_NAMES)


def iter_files() -> list[Path]:
    files: list[Path] = []
    target_names = list(TARGETS)
    target_names.extend(
        path.name
        for path in BASE_DIR.glob("app_fasthtml_admin_*")
        if path.is_dir() and path.name not in target_names
    )
    for target in target_names:
        root = BASE_DIR / target
        if not root.exists():
            continue
        if root.is_file():
            files.append(root)
            continue
        files.extend(path for path in root.rglob("*") if path.is_file())
    return sorted(path for path in files if not is_excluded(path))


def main() -> int:
    violations: list[str] = []
    for path in iter_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            for token in TOKENS:
                if token in line:
                    rel = path.relative_to(BASE_DIR)
                    violations.append(f"{rel}:{line_no}: {token}")
    if violations:
        print("Hardcoded URL/host values were found outside config/settings.json:")
        print("\n".join(violations))
        return 1
    print("ok: no hardcoded URL/host values found in code targets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
