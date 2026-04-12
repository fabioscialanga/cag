"""
Runtime configuration helpers for per-request CAG execution.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from cag.config import settings


class RuntimeConfig(BaseModel):
    """Per-request runtime overrides for graph execution."""

    relevance_threshold: float = Field(default_factory=lambda: settings.relevance_threshold, ge=0.0, le=1.0)
    confidence_threshold: float = Field(default_factory=lambda: settings.confidence_threshold, ge=0.0, le=1.0)
    hallucination_threshold: float = Field(default_factory=lambda: settings.hallucination_threshold, ge=0.0, le=1.0)


def resolve_runtime_config(runtime_config: RuntimeConfig | None = None) -> RuntimeConfig:
    """Return the explicit runtime config or derive one from current settings."""

    return runtime_config or RuntimeConfig()
