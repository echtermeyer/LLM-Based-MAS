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

_DEFAULT_TEMPERATURE = 1.0
_MAX_TOKENS = 16_384


_FACTORIES: Dict[str, Callable[[float], BaseChatModel]] = {
    "gpt-4o": lambda t: init_llm("gpt-4o", temperature=t, max_tokens=_MAX_TOKENS),
    "claude-sonnet-4": lambda t: init_llm(
        "anthropic--claude-4-sonnet",
        model_id="anthropic.claude-sonnet-4-20250514-v1:0",
        init_func=_amazon_converse,
        max_tokens=_MAX_TOKENS,
        temperature=t,
    ),
    "claude-sonnet-4.5": lambda t: init_llm(
        "anthropic--claude-4.5-sonnet",
        model_id="anthropic.claude-sonnet-4-5-20251101-v1:0",
        init_func=_amazon_converse,
        max_tokens=_MAX_TOKENS,
        top_p=None,
        temperature=t,
    ),
    "gemini-pro": lambda t: init_llm(
        "gemini-2.5-pro", init_func=_google_genai, max_tokens=_MAX_TOKENS, temperature=t
    ),
    "nova-pro": lambda t: init_llm(
        "amazon--nova-pro",
        model_id="amazon.nova-pro-v1:0",
        init_func=_amazon_converse,
        max_tokens=_MAX_TOKENS,
        top_p=None,
        temperature=t,
    ),
    "mistral-large": lambda t: init_llm(
        "mistralai--mistral-large-instruct", temperature=t, max_tokens=_MAX_TOKENS
    ),
}


class Models:
    GPT_4O = _FACTORIES["gpt-4o"](_DEFAULT_TEMPERATURE)
    CLAUDE_SONNET_4 = _FACTORIES["claude-sonnet-4"](_DEFAULT_TEMPERATURE)
    CLAUDE_SONNET_45 = _FACTORIES["claude-sonnet-4.5"](_DEFAULT_TEMPERATURE)
    GEMINI_PRO = _FACTORIES["gemini-pro"](_DEFAULT_TEMPERATURE)
    NOVA_PRO = _FACTORIES["nova-pro"](_DEFAULT_TEMPERATURE)
    MISTRAL_LARGE = _FACTORIES["mistral-large"](_DEFAULT_TEMPERATURE)

    NAMES = list(_FACTORIES.keys())

    ALL = [
        GPT_4O,
        CLAUDE_SONNET_4,
        CLAUDE_SONNET_45,
        GEMINI_PRO,
        NOVA_PRO,
        MISTRAL_LARGE,
    ]

    @classmethod
    def create(
        cls, name: str, temperature: float = _DEFAULT_TEMPERATURE
    ) -> BaseChatModel:
        if name not in _FACTORIES:
            raise ValueError(f"Unknown model '{name}'. Available: {cls.NAMES}")
        return _FACTORIES[name](temperature)
