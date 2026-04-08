"""
Embedding and vector store management.
"""
from __future__ import annotations

import logging
import sys

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from cag.config import VectorDB, settings

logger = logging.getLogger(__name__)


def get_embeddings():
    """Return the embedding model used by the project."""

    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key or None,
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

    logger.info("Upserting %s chunks into %s", len(chunks), settings.vector_db.value)
    vector_store.add_documents(chunks)
    logger.info("Upsert completed")
    return len(chunks)


def similarity_search(query: str, k: int | None = None) -> list[Document]:
    """Run similarity search against the configured vector store."""

    vector_store = get_vector_store()
    top_k = k or settings.retrieval_top_k
    results = vector_store.similarity_search(query, k=top_k)
    logger.info("Similarity search '%s...' -> %s results", query[:60], len(results))
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
    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)

    if args.reset:
        logger.warning("Reset enabled: clearing the vector store")
        if hasattr(vector_store, "delete_collection"):
            vector_store.delete_collection()
        vector_store = get_vector_store(embeddings)

    count = upsert_chunks(chunks, vector_store)
    logger.info("Pipeline completed: %s chunks indexed.", count)


if __name__ == "__main__":
    main()
