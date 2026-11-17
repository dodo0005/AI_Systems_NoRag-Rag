"""One LLM client for the whole app, configured from the environment.

    LLM_URL      base URL of any OpenAI-compatible chat completions API
    LLM_API_KEY  its key (several keys, comma-separated, are tried in turn on 429)
    LLM_MODEL    the model name

Read from .env.local first, then .env, then the real environment.
Google Gemini is only a default: if LLM_URL is not set, the Gemini
OpenAI-compatible endpoint is used, and GOOGLE_API_KEY still works as a key.

Every call reports its token usage and cost to the current trace (if any).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from backend.tracing import current_trace

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# .env.local wins over .env; neither overrides variables already set.
load_dotenv(PROJECT_ROOT / ".env.local")
load_dotenv(PROJECT_ROOT / ".env")

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_DEFAULT_MODEL = "gemini-3.5-flash-lite"


class LLMNotConfigured(RuntimeError):
    """No key or no model: the API answers 503, not 502."""


class LLMRateLimited(RuntimeError):
    """Every key got a 429: the API answers 429."""


@dataclass
class LLMSettings:
    url: str
    model: str
    keys: list[str]
    price_input_per_m: float
    price_output_per_m: float


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def settings() -> LLMSettings:
    """Read the configuration at call time, so a changed .env needs no code change."""
    url = (os.getenv("LLM_URL") or GEMINI_URL).strip()
    model = (os.getenv("LLM_MODEL") or os.getenv("GEMINI_MODEL") or GEMINI_DEFAULT_MODEL).strip()

    raw_keys = os.getenv("LLM_API_KEY") or os.getenv("GOOGLE_API_KEYS") or os.getenv("GOOGLE_API_KEY") or ""
    keys = [k.strip() for k in raw_keys.split(",") if k.strip()]

    return LLMSettings(
        url=url,
        model=model,
        keys=keys,
        price_input_per_m=_float_env("LLM_PRICE_INPUT_PER_M", 0.0),
        price_output_per_m=_float_env("LLM_PRICE_OUTPUT_PER_M", 0.0),
    )


def describe() -> dict:
    """What the traces and the UI may show about the LLM (never the key)."""
    s = settings()
    return {"llm_url": s.url, "llm_model": s.model}


def secrets() -> list[str]:
    return settings().keys


def cost_usd(input_tokens: int, output_tokens: int, s: LLMSettings | None = None) -> float:
    s = s or settings()
    return round(
        input_tokens * s.price_input_per_m / 1_000_000
        + output_tokens * s.price_output_per_m / 1_000_000,
        8,
    )


@dataclass
class LLMResult:
    content: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    model: str  # the model that answered, as reported by the provider
    seconds: float


def _client(url: str, key: str) -> OpenAI:
    return OpenAI(api_key=key, base_url=url, max_retries=0, timeout=120)


def chat(
    prompt: str,
    *,
    system: str | None = None,
    json_mode: bool = True,
    temperature: float | None = 0,
    model: str | None = None,
    max_attempts: int = 4,
) -> LLMResult:
    """One chat completion. On 429: back off, retry, then try the next key.
    On 5xx (e.g. Gemini's 503 "model overloaded") or a lost connection:
    back off and retry the same key; these are usually gone a few seconds later."""
    s = settings()
    if not s.keys:
        raise LLMNotConfigured("LLM_API_KEY is not set. Check your .env or .env.local file.")
    if not s.model and not model:
        raise LLMNotConfigured("LLM_MODEL is not set.")

    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    kwargs = {"model": model or s.model, "messages": messages}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    last_error: Exception | None = None
    for key in s.keys:
        for attempt in range(max_attempts):
            try:
                started = time.perf_counter()
                response = _client(s.url, key).chat.completions.create(**kwargs)
                seconds = round(time.perf_counter() - started, 3)
            except RateLimitError as exc:
                last_error = exc
                time.sleep(min(30, 2 * 2**attempt))
                continue
            except (InternalServerError, APIConnectionError, APITimeoutError) as exc:
                last_error = exc
                if attempt < max_attempts - 1:
                    time.sleep(min(30, 2 * 2**attempt))
                continue

            usage = getattr(response, "usage", None)
            input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(usage, "completion_tokens", 0) or 0
            result = LLMResult(
                content=response.choices[0].message.content or "",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd(input_tokens, output_tokens, s),
                model=getattr(response, "model", None) or kwargs["model"],
                seconds=seconds,
            )

            tr = current_trace()
            if tr is not None:
                tr.add_usage(input_tokens, output_tokens, result.cost_usd)

            return result

    if isinstance(last_error, RateLimitError):
        raise LLMRateLimited("The LLM provider is rate-limiting every key. Try again later.") from last_error
    raise last_error
