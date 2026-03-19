"""
llm_client.py

Abstraction layer for LLM providers.
Exposes a unified interface regardless of whether the active provider
is Anthropic or OpenAI. Provider and model are read from settings,
so switching requires only an environment variable change.
"""
