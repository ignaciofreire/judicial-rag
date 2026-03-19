"""
settings.py

Application settings via Pydantic Settings.
Reads all configuration from environment variables (or .env file).
Single source of truth for LLM provider, model, parallelism limits,
chunk sizes and session timeout.
"""
