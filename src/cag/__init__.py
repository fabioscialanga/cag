"""CAG - Cognitive Augmented Generation for grounded document intelligence."""

from cag.config import settings
from cag.graph.graph import build_graph, get_graph, run_query
from cag.ingestion.chunker import chunk_documents
from cag.ingestion.embedder import get_embeddings, get_vector_store, similarity_search, upsert_chunks
from cag.ingestion.loader import load_documents

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "settings",
    "run_query",
    "build_graph",
    "get_graph",
    "load_documents",
    "chunk_documents",
    "get_embeddings",
    "get_vector_store",
    "upsert_chunks",
    "similarity_search",
]
