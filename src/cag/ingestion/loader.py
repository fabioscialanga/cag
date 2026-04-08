"""
Document loading utilities for the CAG corpus.
"""
from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_documents(data_dir: str | Path = "./data/raw") -> list[Document]:
    """Load supported documents and attach normalized metadata."""

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Directory not found: {data_path}")

    documents: list[Document] = []

    for file_path in sorted(data_path.rglob("*")):
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        logger.info("Loading %s", file_path.name)
        try:
            docs = _load_single(file_path)
            for document in docs:
                document.metadata.update(
                    {
                        "source": str(file_path),
                        "filename": file_path.name,
                        "domain_module": _extract_domain_module(file_path.stem),
                    }
                )
            documents.extend(docs)
            logger.info("Loaded %s page(s)/section(s)", len(docs))
        except Exception as exc:
            logger.warning("Failed to load %s: %s", file_path.name, exc)

    logger.info("Total documents loaded: %s", len(documents))
    return documents


def _load_single(file_path: Path) -> list[Document]:
    """Load one supported file."""

    extension = file_path.suffix.lower()
    if extension == ".pdf":
        return PyPDFLoader(str(file_path)).load()
    if extension in {".txt", ".md"}:
        return TextLoader(str(file_path), encoding="utf-8").load()
    raise ValueError(f"Unsupported extension: {extension}")


def _extract_domain_module(stem: str) -> str:
    """Derive a neutral domain/module label from a filename stem."""

    parts = stem.lower().replace("-", "_").split("_")
    skip = {
        "guide",
        "doc",
        "manual",
        "complete",
        "signed",
        "systems",
        "system",
        "data",
    }
    for part in parts:
        if part not in skip and len(part) > 2:
            return part
    return "general"
