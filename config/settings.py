"""
settings.py

Application settings via Pydantic Settings.
Reads all configuration from environment variables (or .env file).
Single source of truth for LLM provider, model, parallelism limits,
chunk sizes and session timeout.
"""

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# To add a new LLM provider: extend this set, add its API key field below,
# and add an entry in the required_keys dict inside validate_settings.
SUPPORTED_LLM_PROVIDERS = {"anthropic", "openai", "gemini", "deepseek"}

# To add a new embedding provider: extend this set, add its API key field
# below if needed, and handle it in embedder.py.
SUPPORTED_EMBEDDING_PROVIDERS = {"huggingface"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Silently ignore system environment variables not defined in this class
        extra="ignore",
    )

    # LLM
    # Only the active provider's key is required, the rest can be left empty in .env
    # repr=False prevents key values from appearing in logs or debug output
    anthropic_api_key: str = Field(default="", repr=False)
    openai_api_key: str = Field(default="", repr=False)
    gemini_api_key: str = Field(default="", repr=False)
    deepseek_api_key: str = Field(default="", repr=False)
    llm_provider: str = Field(default="deepseek")
    llm_model: str = Field(default="deepseek-reasoner", min_length=1)

    # Embeddings
    # API key is shared with the HF Spaces deployment key — same account.
    # repr=False prevents the key from appearing in logs or debug output.
    huggingface_api_key: str = Field(default="", repr=False)
    embedding_provider: str = Field(default="huggingface")
    # Best open-source multilingual retrieval model for Spanish judicial text
    embedding_model: str = Field(
        default="intfloat/multilingual-e5-large-instruct", min_length=1
    )

    # Pipeline
    # Keep max_parallel_pdfs low on memory-constrained environments
    max_parallel_pdfs: int = Field(default=4, ge=1, le=10)
    # chunk_overlap is the number of characters repeated between consecutive chunks
    chunk_size: int = Field(default=1000, ge=100, le=8000)
    chunk_overlap: int = Field(default=200, ge=0)

    # Session
    # After this many minutes of inactivity
    # the session temp directory will be deleted automatically
    session_timeout_minutes: int = Field(default=60, ge=5, le=1440)

    @model_validator(mode="after")
    def validate_settings(self) -> "Settings":
        # Validate LLM provider
        if self.llm_provider not in SUPPORTED_LLM_PROVIDERS:
            raise ValueError(
                f"LLM_PROVIDER '{self.llm_provider}' is not supported. "
                f"Choose one of: {', '.join(sorted(SUPPORTED_LLM_PROVIDERS))}"
            )

        # Maps each LLM provider to its field name and .env variable name.
        # Add a new entry here when extending SUPPORTED_LLM_PROVIDERS.
        required_llm_keys: dict[str, tuple[str, str]] = {
            "anthropic": ("anthropic_api_key", "ANTHROPIC_API_KEY"),
            "openai": ("openai_api_key", "OPENAI_API_KEY"),
            "gemini": ("gemini_api_key", "GEMINI_API_KEY"),
            "deepseek": ("deepseek_api_key", "DEEPSEEK_API_KEY"),
        }

        # Only the active LLM provider's key is validated, the others may be empty
        field_name, env_var = required_llm_keys[self.llm_provider]
        if not getattr(self, field_name):
            raise ValueError(
                f"{env_var} is required when LLM_PROVIDER={self.llm_provider}"
            )

        # Validate embedding provider
        if self.embedding_provider not in SUPPORTED_EMBEDDING_PROVIDERS:
            raise ValueError(
                f"EMBEDDING_PROVIDER '{self.embedding_provider}' is not supported. "
                f"Choose one of: {', '.join(sorted(SUPPORTED_EMBEDDING_PROVIDERS))}"
            )

        # Hugging Face key is required for embedding API calls
        if self.embedding_provider == "huggingface" and not self.huggingface_api_key:
            raise ValueError(
                "HUGGINGFACE_API_KEY is required when " "EMBEDDING_PROVIDER=huggingface"
            )

        # Overlap must be strictly smaller than chunk size,
        # otherwise chunks would be entirely contained within the previous one
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) must be smaller "
                f"than CHUNK_SIZE ({self.chunk_size})"
            )

        return self


settings = Settings()
