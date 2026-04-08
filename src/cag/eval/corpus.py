from __future__ import annotations

import gc
import json
import tempfile
import uuid
from contextlib import AbstractContextManager
from importlib.resources import files
from pathlib import Path

from langchain_core.documents import Document

from cag.eval.models import BenchmarkItem
from cag.ingestion.chunker import chunk_documents
from cag.ingestion.embedder import get_embeddings
from cag.ingestion.loader import SUPPORTED_EXTENSIONS, _extract_domain_module, _load_single


def get_default_dataset_path() -> Path:
    return Path(str(files("cag.eval").joinpath("benchmark_dataset.jsonl")))


def load_benchmark_dataset(path: str | Path | None = None) -> list[BenchmarkItem]:
    dataset_path = Path(path) if path else get_default_dataset_path()
    items: list[BenchmarkItem] = []

    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            items.append(BenchmarkItem(**json.loads(raw)))

    if not items:
        raise ValueError(f"Benchmark dataset is empty: {dataset_path}")

    return items


def collect_benchmark_sources(items: list[BenchmarkItem]) -> list[str]:
    sources = {source for item in items for source in item.gold_sources}
    return sorted(sources)


def load_selected_documents(
    data_dir: str | Path,
    allowed_filenames: set[str] | None = None,
) -> list[Document]:
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    selected = {name.lower() for name in allowed_filenames} if allowed_filenames else None
    documents: list[Document] = []

    for file_path in sorted(data_path.rglob("*")):
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if selected and file_path.name.lower() not in selected:
            continue

        docs = _load_single(file_path)
        for doc in docs:
            doc.metadata.update(
                {
                    "source": str(file_path),
                    "filename": file_path.name,
                    "domain_module": _extract_domain_module(file_path.stem),
                }
            )
        documents.extend(docs)

    if not documents:
        target = sorted(selected or [])
        raise ValueError(f"No supported documents found in {data_path} for {target}")

    return documents


class BenchmarkVectorIndex(AbstractContextManager["BenchmarkVectorIndex"]):
    def __init__(self, data_dir: str | Path, benchmark_items: list[BenchmarkItem]):
        self.data_dir = Path(data_dir)
        self.benchmark_items = benchmark_items
        self.allowed_sources = collect_benchmark_sources(benchmark_items)
        self._tempdir: tempfile.TemporaryDirectory[str] | None = None
        self.vector_store = None
        self.chunks: list[Document] = []

    def build(self) -> "BenchmarkVectorIndex":
        from langchain_community.vectorstores import Chroma

        self._tempdir = tempfile.TemporaryDirectory(prefix="cag_eval_")
        documents = load_selected_documents(self.data_dir, set(self.allowed_sources))
        self.chunks = chunk_documents(documents)

        embeddings = get_embeddings()
        self.vector_store = Chroma(
            collection_name=f"cag_eval_{uuid.uuid4().hex[:8]}",
            embedding_function=embeddings,
            persist_directory=self._tempdir.name,
        )
        self.vector_store.add_documents(self.chunks)
        return self

    def similarity_search(self, query: str, k: int) -> list[Document]:
        if self.vector_store is None:
            raise RuntimeError("Benchmark vector index has not been built.")
        return self.vector_store.similarity_search(query, k=k)

    def close(self) -> None:
        self.vector_store = None
        gc.collect()
        if self._tempdir is not None:
            try:
                self._tempdir.cleanup()
            except Exception:
                pass
            self._tempdir = None

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
        return None
