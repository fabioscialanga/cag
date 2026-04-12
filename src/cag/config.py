"""
CAG configuration.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
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

    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    cag_api_key: str = ""

    vector_db: VectorDB = VectorDB.CHROMA
    chroma_persist_dir: Path = Path("./data/chroma_db")
    chroma_collection: str = "cag_documents"
    pinecone_api_key: str = ""
    pinecone_index: str = "cag-documents-index"
    pinecone_env: str = "us-east-1"

    relevance_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    retrieval_top_k: int = Field(default=10, ge=1, le=50)
    hallucination_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    moderate_relevance_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    max_reason_retries: int = Field(default=2, ge=0, le=5)

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

    @field_validator("chroma_persist_dir", "log_file", mode="before")
    @classmethod
    def ensure_path(cls, value: str | Path) -> Path:
        return Path(value)


settings = Settings()
