"""
llm_client.py

Abstraction layer over LLM providers.
Exposes a single call_llm() function that routes to the active provider
configured via LLM_PROVIDER in settings. Adding a new provider requires
a new entry in _DISPATCH and a corresponding _call_* implementation.

Supported providers:
    anthropic — Anthropic SDK (claude-* models).
    openai    — OpenAI SDK (gpt-* models).
    deepseek  — OpenAI-compatible API at api.deepseek.com.
    gemini    — OpenAI-compatible API at generativelanguage.googleapis.com.

DeepSeek and Gemini share _call_openai_compatible because both implement
the OpenAI /v1/chat/completions spec, differing only in base URL and key.
"""

import logging
from collections.abc import Callable

from config.settings import settings

logger = logging.getLogger(__name__)

# Base URLs for providers that implement the OpenAI chat completions spec.
# Kept as module constants so they are visible without reading the functions.
_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"

# Maximum tokens the LLM may generate per call. 1024 is sufficient for the
# structured JSON responses the RAG agent expects — larger values waste quota.
_MAX_TOKENS = 1024


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def call_llm(prompt: str) -> str:
    """Send a prompt to the configured LLM and return the response text.

    Routes to the provider implementation selected by LLM_PROVIDER in
    settings. The prompt is sent as a single user message — system-level
    instructions are embedded in the prompt by the RAG agent.

    Args:
        prompt: Complete prompt string built by rag_agent.py.

    Returns:
        Raw response text from the LLM. The RAG agent expects this to be
        valid JSON as instructed by the prompt's output specification.

    Raises:
        RuntimeError: If the provider is unsupported or the API call fails.
    """
    provider = settings.llm_provider
    logger.debug(
        "LLM call: provider=%s model=%s prompt_len=%d",
        provider,
        settings.llm_model,
        len(prompt),
    )

    handler = _DISPATCH.get(provider)
    if handler is None:
        raise RuntimeError(
            f"Unsupported LLM provider: {provider!r}. "
            f"Choose one of: {', '.join(sorted(_DISPATCH))}"
        )

    try:
        response = handler(prompt)
        logger.debug("LLM response: %d chars", len(response))
        return response
    except Exception as e:
        logger.exception(
            "LLM call failed (provider=%s model=%s)", provider, settings.llm_model
        )
        raise RuntimeError(f"LLM call failed [{provider}]: {e}") from e


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------


def _call_anthropic(prompt: str) -> str:
    """Call the Anthropic Messages API.

    Imported lazily so the module loads even when the anthropic package
    is not installed or the API key is not set.

    Args:
        prompt: Complete prompt string.

    Returns:
        Response text from the model.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    message = client.messages.create(
        model=settings.llm_model,
        max_tokens=_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )

    logger.debug(
        "Anthropic usage: in=%d out=%d tokens",
        message.usage.input_tokens,
        message.usage.output_tokens,
    )
    return message.content[0].text


def _call_openai(prompt: str) -> str:
    """Call the OpenAI Chat Completions API.

    Args:
        prompt: Complete prompt string.

    Returns:
        Response text from the model.
    """
    import openai

    client = openai.OpenAI(api_key=settings.openai_api_key)
    completion = client.chat.completions.create(
        model=settings.llm_model,
        max_tokens=_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )

    logger.debug(
        "OpenAI usage: in=%d out=%d tokens",
        completion.usage.prompt_tokens,
        completion.usage.completion_tokens,
    )
    return completion.choices[0].message.content or ""


def _call_openai_compatible(prompt: str) -> str:
    """Call a provider that implements the OpenAI Chat Completions spec.

    DeepSeek and Gemini both expose /v1/chat/completions endpoints compatible
    with the OpenAI SDK. The correct base URL and API key are selected from
    _PROVIDER_CONFIG based on the active provider in settings.

    Args:
        prompt: Complete prompt string.

    Returns:
        Response text from the model.
    """
    import openai

    provider = settings.llm_provider
    base_url, api_key = _PROVIDER_CONFIG[provider]

    client = openai.OpenAI(base_url=base_url, api_key=api_key)
    completion = client.chat.completions.create(
        model=settings.llm_model,
        max_tokens=_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )

    logger.debug(
        "%s usage: in=%d out=%d tokens",
        provider,
        completion.usage.prompt_tokens,
        completion.usage.completion_tokens,
    )
    return completion.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Configuration tables
# ---------------------------------------------------------------------------
# Defined after the functions they reference so the file reads top-down:
# public API → implementations → configuration.

# Maps each OpenAI-compatible provider to its (base_url, api_key) tuple.
# Add a new entry here when adding a provider that uses _call_openai_compatible.
_PROVIDER_CONFIG: dict[str, tuple[str, str]] = {
    "deepseek": (_DEEPSEEK_BASE_URL, settings.deepseek_api_key),
    "gemini": (_GEMINI_BASE_URL, settings.gemini_api_key),
}

# Routes provider names to their call implementations.
# _call_openai_compatible handles any provider in _PROVIDER_CONFIG.
_DISPATCH: dict[str, Callable[[str], str]] = {
    "anthropic": _call_anthropic,
    "openai": _call_openai,
    "deepseek": _call_openai_compatible,
    "gemini": _call_openai_compatible,
}
