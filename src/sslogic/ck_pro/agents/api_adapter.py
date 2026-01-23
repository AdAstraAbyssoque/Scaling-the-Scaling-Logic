"""
API Adapter for SSLogic
Generic API interface adapter for model calls.
"""

import datetime
import json
import os
import time
import uuid
import requests
from typing import Any, Dict, List, Optional


class APIHelper:
    # Model fallback strategy
    MODEL_FALLBACK_CHAIN = {
        "claude-sonnet-4-5-20250929-az": [
            "claude-sonnet-4-5-20250929",
            "claude-haiku-4-5-20251001",
        ],
        "deepseek-v3-1-terminus": [
            "gpt-4o-latest",
            "o4-mini",
        ],
        "o4-mini": [
            "o3-mini",
            "gpt-5",
            "claude-sonnet-4-5-20250929-az",
        ],
        "o3-mini": [
            "gpt-5",
            "claude-sonnet-4-5-20250929-az",
        ],
    }

    # Model mapping - using standard names
    MODEL_MARKER_DICT = {
        "gpt-4o": "gpt-4o",
        "gpt-4o-latest": "chatgpt-4o-latest",
        "deepseek-chat": "deepseek-chat",
        "claude-3-5-sonnet": "claude-3-5-sonnet-20240620",
        "o1-preview": "o1-preview",
        "o1-mini": "o1-mini",
    }

    # API Configuration
    COMMON_API_URL = os.getenv(
        "SSLOGIC_API_URL", "https://api.openai.com/v1/chat/completions"
    )
    DEFAULT_API_KEY = os.getenv("SSLOGIC_API_KEY", "")

    @staticmethod
    def _get_headers() -> dict:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {APIHelper.DEFAULT_API_KEY}",
        }
        return headers

    @staticmethod
    def _convert_messages_to_internal_format(messages: List[Dict]) -> tuple:
        messages_list = list(messages)
        system_prompt = ""
        if messages_list and messages_list[0]["role"] == "system":
            system_prompt = messages_list[0]["content"]
        return system_prompt, messages_list

    @staticmethod
    def _call_common_api(
        messages: List[Dict],
        model_name: str,
        stat: Optional[Dict] = None,
        timeout: int = 300,
        **kwargs,
    ) -> str:
        real_model_name = APIHelper.MODEL_MARKER_DICT.get(model_name, model_name)
        params = {}
        if isinstance(real_model_name, list):
            real_model_name, params = real_model_name

        payload = {"model": real_model_name, "messages": messages, "stream": False}
        if params:
            payload.update(params)
        payload.update(kwargs)
        payload.pop("thinking", None)

        headers = APIHelper._get_headers()
        url = APIHelper.COMMON_API_URL

        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=timeout
            )
            response.raise_for_status()
            result = response.json()

            if stat is not None and "usage" in result:
                usage = result["usage"]
                stat["prompt_tokens"] = stat.get("prompt_tokens", 0) + usage.get(
                    "prompt_tokens", 0
                )
                stat["completion_tokens"] = stat.get(
                    "completion_tokens", 0
                ) + usage.get("completion_tokens", 0)
                stat["total_tokens"] = stat.get("total_tokens", 0) + usage.get(
                    "total_tokens", 0
                )

            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0].get("message", {}).get("content", "")
            else:
                raise ValueError(f"Invalid API response: {result}")
        except Exception as e:
            raise RuntimeError(f"API call failed: {str(e)}")

    @staticmethod
    def call_chat(
        messages: List[Dict],
        model_name: str,
        stat: Optional[Dict] = None,
        timeout: int = 300,
        **kwargs,
    ) -> str:
        fallback_models = APIHelper.MODEL_FALLBACK_CHAIN.get(model_name, [])
        try:
            return APIHelper._call_common_api(
                messages, model_name, stat, timeout, **kwargs
            )
        except Exception as e:
            print(f"Primary model {model_name} failed: {e}")
            for fallback_model in fallback_models:
                try:
                    print(f"Trying fallback model: {fallback_model}...")
                    return APIHelper._call_common_api(
                        messages, fallback_model, stat, timeout, **kwargs
                    )
                except Exception as fb_e:
                    print(f"Fallback model {fallback_model} failed: {fb_e}")
            raise e
