"""
CAG configuration.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    OLLAMA = "ollama"


class VectorDB(str, Enum):
    CHROMA = "chroma"
    PINECONE = "pinecone"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: LLMProvider = LLMProvider.OPENAI

    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    embedding_model: str = "text-embedding-3-large"
    embedding_dim: int = 3072
    embedding_tiktoken_enabled: bool = False
    embedding_check_ctx_length: bool = False
    cag_api_key: str = ""

    vector_db: VectorDB = VectorDB.CHROMA
    chroma_persist_dir: Path = Path("./data/chroma_db")
    chroma_collection: str = "cag_documents"
    pinecone_api_key: str = ""
    pinecone_index: str = "cag-documents-index"
    pinecone_env: str = "us-east-1"
    knowledge_db_path: Path = Path("./data/knowledge.db")
    enable_knowledge_compiler: bool = True

    relevance_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    retrieval_top_k: int = Field(default=20, ge=1, le=50)
    hallucination_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    moderate_relevance_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    max_reason_retries: int = Field(default=2, ge=0, le=5)
    fast_context_selection: bool = True
    fast_context_min_score: float = Field(default=4.0, ge=0.0)
    enable_conversation_router_llm: bool = False
    chunk_size: int = Field(default=1200, ge=300, le=4000)
    chunk_overlap: int = Field(default=180, ge=0, le=800)
    context_selection_limit: int = Field(default=8, ge=1, le=20)
    complex_context_selection_limit: int = Field(default=10, ge=1, le=20)
    adaptive_retrieval_retry: bool = True
    adaptive_retry_top_k_boost: int = Field(default=5, ge=1, le=20)
    hybrid_lexical_retrieval: bool = True
    hybrid_lexical_top_k: int = Field(default=8, ge=1, le=30)
    strict_fast_profile: bool = False

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_file: Path = Path("./logs/cag.log")

    @property
    def active_model_id(self) -> str:
        """Return the active model identifier for the current provider."""

        return {
            LLMProvider.OPENAI: self.openai_model,
            LLMProvider.ANTHROPIC: self.anthropic_model,
            LLMProvider.GROQ: self.groq_model,
            LLMProvider.OLLAMA: self.ollama_model,
        }[self.llm_provider]

    @field_validator("chroma_persist_dir", "knowledge_db_path", "log_file", mode="before")
    @classmethod
    def ensure_path(cls, value: str | Path) -> Path:
        return Path(value)

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        return value.upper() if isinstance(value, str) else value

    @model_validator(mode="after")
    def apply_strict_fast_profile(self):
        """Apply speed-focused overrides while keeping validation/escalation rigor."""

        if not self.strict_fast_profile:
            return self

        self.retrieval_top_k = min(self.retrieval_top_k, 10)
        self.context_selection_limit = min(self.context_selection_limit, 6)
        self.complex_context_selection_limit = min(self.complex_context_selection_limit, 8)
        self.adaptive_retrieval_retry = False
        self.hybrid_lexical_retrieval = False
        self.fast_context_selection = True
        self.fast_context_min_score = min(self.fast_context_min_score, 3.5)
        return self


settings = Settings()
