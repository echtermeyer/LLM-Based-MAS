import json
import os
from pathlib import Path
from typing import Callable, Dict

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

load_dotenv()

_creds_path = Path(__file__).parents[2] / "gen_ai_credential.json"
_creds = json.loads(_creds_path.read_text())

os.environ["AICORE_AUTH_URL"] = _creds["url"]
os.environ["AICORE_CLIENT_ID"] = _creds["clientid"]
os.environ["AICORE_CLIENT_SECRET"] = _creds["clientsecret"]
os.environ["AICORE_BASE_URL"] = _creds["serviceurls"]["AI_API_URL"]

from gen_ai_hub.proxy.langchain.init_models import init_llm
from gen_ai_hub.proxy.langchain.amazon import (
    init_chat_converse_model as _amazon_converse,
)
from gen_ai_hub.proxy.langchain.google_genai import init_chat_model as _google_genai

_MAX_TOKENS = 64_000
_REQUEST_TIMEOUT = 300


def _with_timeout(llm: BaseChatModel) -> BaseChatModel:
    if hasattr(llm, "request_timeout"):
        llm.request_timeout = _REQUEST_TIMEOUT
    return llm


_TEMPERATURES: Dict[str, float] = {
    "gpt-4o": 1.0,
    "claude-sonnet-4": 1.0,
    "claude-sonnet-4.5": 1.0,
    "gemini-pro": 1.0,
    "nova-pro": 1.0,
    "mistral-large": 0.7,
    "mistral-medium": 0.7,
    "mistral-small": 0.7,
}

_FACTORIES: Dict[str, Callable[[], BaseChatModel]] = {
    "gpt-4o": lambda: _with_timeout(init_llm(
        "gpt-4o", temperature=_TEMPERATURES["gpt-4o"], max_tokens=_MAX_TOKENS,
    )),
    "claude-sonnet-4": lambda: _with_timeout(init_llm(
        "anthropic--claude-4-sonnet",
        model_id="anthropic.claude-sonnet-4-20250514-v1:0",
        init_func=_amazon_converse,
        max_tokens=_MAX_TOKENS,
        temperature=_TEMPERATURES["claude-sonnet-4"],
    )),
    "claude-sonnet-4.5": lambda: _with_timeout(init_llm(
        "anthropic--claude-4.5-sonnet",
        model_id="anthropic.claude-sonnet-4-5-20251101-v1:0",
        init_func=_amazon_converse,
        max_tokens=_MAX_TOKENS,
        top_p=None,
        temperature=_TEMPERATURES["claude-sonnet-4.5"],
    )),
    "gemini-pro": lambda: _with_timeout(init_llm(
        "gemini-2.5-pro",
        init_func=_google_genai,
        max_tokens=_MAX_TOKENS,
        temperature=_TEMPERATURES["gemini-pro"],
    )),
    "nova-pro": lambda: _with_timeout(init_llm(
        "amazon--nova-pro",
        model_id="amazon.nova-pro-v1:0",
        init_func=_amazon_converse,
        max_tokens=_MAX_TOKENS,
        top_p=None,
        temperature=_TEMPERATURES["nova-pro"],
    )),
    "mistral-large": lambda: _with_timeout(init_llm(
        "mistralai--mistral-large-instruct",
        temperature=_TEMPERATURES["mistral-large"],
        max_tokens=_MAX_TOKENS,
        model_kwargs={"reasoning_effort": "none"},
    )),
    "mistral-medium": lambda: _with_timeout(init_llm(
        "mistralai--mistral-medium-instruct",
        temperature=_TEMPERATURES["mistral-medium"],
        max_tokens=_MAX_TOKENS,
        model_kwargs={"reasoning_effort": "none"},
    )),
    "mistral-small": lambda: _with_timeout(init_llm(
        "mistralai--mistral-small-instruct",
        temperature=_TEMPERATURES["mistral-small"],
        max_tokens=_MAX_TOKENS,
        model_kwargs={"reasoning_effort": "none"},
    )),
}


class Models:
    GPT_4O = _FACTORIES["gpt-4o"]()
    CLAUDE_SONNET_4 = _FACTORIES["claude-sonnet-4"]()
    CLAUDE_SONNET_45 = _FACTORIES["claude-sonnet-4.5"]()
    GEMINI_PRO = _FACTORIES["gemini-pro"]()
    NOVA_PRO = _FACTORIES["nova-pro"]()
    MISTRAL_LARGE = _FACTORIES["mistral-large"]()
    MISTRAL_MEDIUM = _FACTORIES["mistral-medium"]()
    MISTRAL_SMALL = _FACTORIES["mistral-small"]()

    NAMES = list(_FACTORIES.keys())
    TEMPERATURES = _TEMPERATURES

    ALL = [
        GPT_4O,
        CLAUDE_SONNET_4,
        CLAUDE_SONNET_45,
        GEMINI_PRO,
        NOVA_PRO,
        MISTRAL_LARGE,
        MISTRAL_MEDIUM,
        MISTRAL_SMALL,
    ]

    @classmethod
    def create(cls, name: str) -> BaseChatModel:
        if name not in _FACTORIES:
            raise ValueError(f"Unknown model '{name}'. Available: {cls.NAMES}")
        return _FACTORIES[name]()
