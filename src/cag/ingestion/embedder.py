"""
Embedding and vector store management.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from cag.config import VectorDB, settings
from cag.ingestion.chunker import add_contextual_headers
from cag.retrieval.lexical import (
    build_weighted_query_terms,
    discriminative_document_score,
    document_terms,
    expand_query_concepts,
)

logger = logging.getLogger(__name__)

def get_embeddings():
    """Return the embedding model used by the project."""

    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key or None,
        tiktoken_enabled=settings.embedding_tiktoken_enabled,
        check_embedding_ctx_length=settings.embedding_check_ctx_length,
    )


def get_vector_store(embeddings=None) -> VectorStore:
    """Return the configured vector store instance."""

    if embeddings is None:
        embeddings = get_embeddings()

    if settings.vector_db == VectorDB.CHROMA:
        return _get_chroma(embeddings)
    if settings.vector_db == VectorDB.PINECONE:
        return _get_pinecone(embeddings)

    raise ValueError(f"Unsupported vector database: {settings.vector_db}")


def _get_chroma(embeddings) -> VectorStore:
    from langchain_community.vectorstores import Chroma

    persist_dir = settings.chroma_persist_dir
    persist_dir.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )


def _get_pinecone(embeddings) -> VectorStore:
    import pinecone
    from langchain_community.vectorstores import Pinecone as PineconeVS

    pinecone.init(
        api_key=settings.pinecone_api_key,
        environment=settings.pinecone_env,
    )
    return PineconeVS.from_existing_index(
        index_name=settings.pinecone_index,
        embedding=embeddings,
    )


def upsert_chunks(chunks: list[Document], vector_store: VectorStore | None = None) -> int:
    """Insert chunks into the configured vector store."""

    if vector_store is None:
        vector_store = get_vector_store()

    if not chunks:
        logger.warning("No chunks to insert.")
        return 0

    enriched_chunks = add_contextual_headers(chunks)
    logger.info("Upserting %s chunks into %s", len(enriched_chunks), settings.vector_db.value)
    vector_store.add_documents(enriched_chunks)
    logger.info("Upsert completed")
    return len(chunks)


def similarity_search(query: str, k: int | None = None) -> list[Document]:
    """Run similarity search against the configured vector store."""

    top_k = k or settings.retrieval_top_k
    try:
        vector_store = get_vector_store()
        results = vector_store.similarity_search(query, k=top_k)
        logger.info("Similarity search '%s...' -> %s results", query[:60], len(results))
        return results
    except Exception as exc:
        logger.warning("Vector similarity search failed; using local lexical fallback: %s", exc)
        return lexical_file_search(query, k=top_k)


def _expand_lexical_query(query: str) -> str:
    return expand_query_concepts(query)


def lexical_file_search(query: str, k: int | None = None, data_dir: str | Path = "./data/raw") -> list[Document]:
    """Search local source files when the configured vector store is unavailable."""

    from cag.ingestion.chunker import chunk_documents
    from cag.ingestion.loader import load_documents

    top_k = k or settings.retrieval_top_k
    try:
        documents = load_documents(data_dir)
        chunks = chunk_documents(documents)
    except Exception as exc:
        logger.warning("Local lexical fallback could not load documents from %s: %s", data_dir, exc)
        return []

    if not chunks:
        return []

    expanded_query = _expand_lexical_query(query)
    weighted_query_terms = build_weighted_query_terms(query, [expanded_query])
    doc_frequencies: dict[str, int] = {}
    for chunk in chunks:
        for term in set(document_terms(chunk)):
            doc_frequencies[term] = doc_frequencies.get(term, 0) + 1

    corpus_size = max(1, len(chunks))
    scored_chunks = [
        (
            discriminative_document_score(
                chunk,
                weighted_query_terms,
                doc_frequencies,
                corpus_size,
            ),
            chunk,
        )
        for chunk in chunks
    ]
    scored_chunks.sort(key=lambda item: item[0], reverse=True)

    results = [chunk for score, chunk in scored_chunks if score > 0.0][:top_k]
    logger.info("Local lexical fallback '%s...' -> %s results", query[:60], len(results))
    return results


def main(argv=None):
    """CLI entrypoint for load -> chunk -> embed -> upsert."""

    import argparse

    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="CAG ingestion pipeline")
    parser.add_argument("--data-dir", default="./data/raw", help="Directory containing source documents")
    parser.add_argument("--reset", action="store_true", help="Reset the vector store before ingestion")
    args = parser.parse_args(argv)

    from cag.ingestion.chunker import chunk_documents
    from cag.ingestion.loader import load_documents

    logger.info("=== CAG Ingestion Pipeline ===")
    logger.info("Vector DB: %s", settings.vector_db.value)
    logger.info("LLM Provider: %s", settings.llm_provider.value)

    documents = load_documents(args.data_dir)
    if not documents:
        logger.error("No documents found. Check the data directory.")
        sys.exit(1)

    chunks = chunk_documents(documents)

    if args.reset and settings.knowledge_db_path.exists():
        logger.warning("Reset enabled: removing compiled knowledge DB at %s", settings.knowledge_db_path)
        settings.knowledge_db_path.unlink()

    from cag.knowledge.compiler import compile_chunks

    summary = compile_chunks(chunks, settings.knowledge_db_path)
    logger.info("Knowledge compiler completed: %s", summary)

    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)

    if args.reset:
        logger.warning("Reset enabled: clearing the vector store")
        if hasattr(vector_store, "delete_collection"):
            vector_store.delete_collection()
        vector_store = get_vector_store(embeddings)

    try:
        count = upsert_chunks(chunks, vector_store)
    except Exception as exc:
        count = 0
        logger.warning(
            "Vector upsert failed; compiled knowledge and lexical fallback remain available: %s",
            exc,
        )
    logger.info("Pipeline completed: %s chunks indexed.", count)


if __name__ == "__main__":
    main()
