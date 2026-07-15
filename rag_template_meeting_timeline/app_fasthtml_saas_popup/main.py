from __future__ import annotations

import os

from fasthtml.common import serve


os.environ.setdefault("RAG_EVIDENCE_UI_MODE", "popup")

from app_fasthtml_saas.main import app  # noqa: E402


if __name__ == "__main__":
    serve()

