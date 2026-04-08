"""Global pytest configuration for CAG."""

import os


def pytest_configure(config):
    """Set test environment variables before importing project modules."""

    os.environ.setdefault("LLM_PROVIDER", "openai")
    os.environ.setdefault("OPENAI_API_KEY", "sk-test-fake-key")
    os.environ.setdefault("OPENAI_MODEL", "gpt-4o")
    os.environ.setdefault("VECTOR_DB", "chroma")
    os.environ.setdefault("CHROMA_PERSIST_DIR", "/tmp/cag_test_chroma")
    os.environ.setdefault("RELEVANCE_THRESHOLD", "0.7")
    os.environ.setdefault("HALLUCINATION_THRESHOLD", "0.3")
    os.environ.setdefault("CONFIDENCE_THRESHOLD", "0.6")
    os.environ.setdefault("MAX_REASON_RETRIES", "2")
