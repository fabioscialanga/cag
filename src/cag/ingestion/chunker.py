"""
Semantic document chunking.
"""
from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from cag.config import settings

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 180
CONTEXT_HEADER_SEPARATOR = "\n\n---\n\n"


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks while preserving metadata."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n## ", "\n\n### ", "\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    all_chunks: list[Document] = []

    for document in documents:
        chunks = splitter.split_documents([document])
        for index, chunk in enumerate(chunks):
            chunk.metadata.update(
                {
                    "chunk_index": index,
                    "total_chunks": len(chunks),
                    "char_count": len(chunk.page_content),
                }
            )
            all_chunks.append(chunk)

        logger.debug("%s -> %s chunks", document.metadata.get("filename", "?"), len(chunks))

    logger.info(
        "Chunking completed: %s chunks generated from %s documents",
        len(all_chunks),
        len(documents),
    )
    return all_chunks


def build_contextual_header(chunk: Document) -> str:
    """Build retrieval-only context that helps embeddings locate a chunk."""

    metadata = chunk.metadata
    parts = []
    filename = str(metadata.get("filename") or metadata.get("source") or "").strip()
    if filename:
        parts.append(f"Document: {filename}")
    domain_module = str(metadata.get("domain_module") or "").strip()
    if domain_module and domain_module != "general":
        parts.append(f"Domain: {domain_module}")
    chunk_index = metadata.get("chunk_index")
    total_chunks = metadata.get("total_chunks")
    if isinstance(chunk_index, int):
        if isinstance(total_chunks, int) and total_chunks > 0:
            parts.append(f"Chunk: {chunk_index + 1} of {total_chunks}")
        else:
            parts.append(f"Chunk: {chunk_index + 1}")
    topics = metadata.get("document_topics") or []
    if topics:
        parts.append(f"Topics: {', '.join(str(topic) for topic in topics[:6])}")
    summary = str(metadata.get("document_summary") or "").strip()
    if summary:
        parts.append(f"Document summary: {summary[:420]}")
    return "\n".join(parts)


def add_contextual_headers(chunks: list[Document]) -> list[Document]:
    """Return copies whose page content is enriched for retrieval embeddings."""

    enriched: list[Document] = []
    for chunk in chunks:
        header = build_contextual_header(chunk)
        if not header:
            enriched.append(chunk)
            continue
        metadata = dict(chunk.metadata)
        metadata["original_content"] = chunk.page_content
        metadata["contextual_header"] = header
        enriched.append(
            Document(
                page_content=f"{header}{CONTEXT_HEADER_SEPARATOR}{chunk.page_content}",
                metadata=metadata,
            )
        )
    return enriched
