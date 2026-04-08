"""
Semantic document chunking.
"""
from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks while preserving metadata."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
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
