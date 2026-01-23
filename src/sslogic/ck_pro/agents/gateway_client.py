from __future__ import annotations

import os
from typing import Optional

CUSTOM_GATEWAY_BASE_URL = os.getenv("SSLOGIC_GATEWAY_BASE_URL", "")
CUSTOM_GATEWAY_API_KEY = os.getenv("SSLOGIC_GATEWAY_API_KEY", "")
CHAT_COMPLETIONS_ENDPOINT = (
    f"{CUSTOM_GATEWAY_BASE_URL}/chat/completions" if CUSTOM_GATEWAY_BASE_URL else ""
)
CUSTOM_GATEWAY_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {CUSTOM_GATEWAY_API_KEY}" if CUSTOM_GATEWAY_API_KEY else "",
}

_client: Optional[OpenAI] = None


def get_gateway_client() -> OpenAI:
    global _client
    if _client is None:
        if not CUSTOM_GATEWAY_BASE_URL:
            raise RuntimeError("SSLOGIC_GATEWAY_BASE_URL is not set")
        from openai import OpenAI

        _client = OpenAI(
            base_url=CUSTOM_GATEWAY_BASE_URL,
            api_key=CUSTOM_GATEWAY_API_KEY or "EMPTY",
        )
    return _client
