from __future__ import annotations

import uvicorn

from api.config import get_required_url_value, get_url_number, load_settings
from app_fasthtml_admin_common.core import VARIANTS, create_admin_app


def _variant_key() -> str:
    settings = load_settings()
    key = str(settings.get("admin", {}).get("variant_key") or "analyst_saas")
    if key not in VARIANTS:
        allowed = ", ".join(sorted(VARIANTS))
        raise RuntimeError(f"config/settings.json の admin.variant_key が不正です: {key}. allowed={allowed}")
    return key


app = create_admin_app(_variant_key())


def main() -> None:
    uvicorn.run(
        app,
        host=get_required_url_value("admin_bind_host"),
        port=get_url_number("admin_bind_port"),
    )


if __name__ == "__main__":
    main()
