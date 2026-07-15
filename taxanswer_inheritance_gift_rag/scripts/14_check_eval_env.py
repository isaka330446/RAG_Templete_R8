from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from ragas_compat import ensure_ragas_import_compat, format_dependency_report, inspect_eval_dependencies


def main() -> None:
    rows = inspect_eval_dependencies()
    print("Evaluation dependency status:")
    print(format_dependency_report(rows))
    try:
        status = ensure_ragas_import_compat()
        from ragas import evaluate  # noqa: F401
    except Exception as exc:
        print("\nRAGAS import check: failed", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

    print("\nRAGAS import check: ok")
    print(json.dumps({
        "ragas_import_ok": True,
        "vertexai_import_shim_installed": bool(status.get("vertexai_import_shim_installed")),
        "dependency_status": status.get("dependency_status", rows),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
