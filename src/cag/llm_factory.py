"""
CAG LLM factory.
"""
from __future__ import annotations

from cag.config import LLMProvider, settings


def get_agno_model():
    """Return the Agno model configured for the active provider."""

    provider = settings.llm_provider

    if provider == LLMProvider.OPENAI:
        from agno.models.openai import OpenAIChat

        return OpenAIChat(
            id=settings.openai_model,
            api_key=settings.openai_api_key,
        )
    if provider == LLMProvider.ANTHROPIC:
        from agno.models.anthropic import Claude

        return Claude(
            id=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
        )
    if provider == LLMProvider.GROQ:
        from agno.models.groq import Groq

        return Groq(
            id=settings.groq_model,
            api_key=settings.groq_api_key,
        )
    if provider == LLMProvider.OLLAMA:
        from agno.models.ollama import Ollama

        return Ollama(
            id=settings.ollama_model,
            host=settings.ollama_base_url,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")
