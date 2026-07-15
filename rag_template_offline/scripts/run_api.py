from __future__ import annotations

import uvicorn

from api.config import get_required_url_value, get_url_number
from api.main import app


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=get_required_url_value("api_bind_host"),
        port=get_url_number("api_bind_port"),
    )
