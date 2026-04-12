from __future__ import annotations

import asyncio
import gc
import json
import re
import tempfile
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from cag.config import settings
from cag.eval.corpus import cleanup_temp_path, collect_benchmark_sources, load_selected_documents
from cag.eval.models import BenchmarkItem, CitationRecord, SystemOutput
from cag.eval.systems import estimate_cost_units


def classify_query_type(question: str) -> str:
    query_lower = question.lower()
    if any(
        token in query_lower
        for token in ["error", "not working", "problem", "diagnostic", "fault", "404", "500"]
    ):
        return "DIAGNOSTIC"
    if any(
        token in query_lower
        for token in [
            "configure", "configuration", "setup", "setting",
            "settings", "parameter", "parameters", "enable", "disable",
            "prerequisite", "prerequisites", "requirement", "requirements",
        ]
    ):
        return "CONFIGURATION"
    if any(
        token in query_lower
        for token in ["how do", "how can", "procedure", "step", "steps", "workflow"]
    ):
        return "PROCEDURAL"
    return "GENERAL"


def _coerce_payload_to_dict(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, dict):
        return payload

    raw_data = getattr(payload, "raw_data", None)
    if isinstance(raw_data, dict):
        return raw_data.get("data", raw_data)

    response = getattr(payload, "response", None)
    references = getattr(payload, "references", None)
    if response is not None or references is not None:
        return {"response": response, "references": references or []}

    return None


def _parse_ndjson_payload(raw: str) -> dict[str, Any] | None:
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return None

    references: list[dict[str, Any]] = []
    response_parts: list[str] = []

    for line in lines:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None

        if isinstance(data, dict):
            if "references" in data and isinstance(data["references"], list):
                references = data["references"]
            if "response" in data and data["response"]:
                response_parts.append(str(data["response"]))

    if not references and not response_parts:
        return None

    return {"response": "".join(response_parts).strip(), "references": references}


def _extract_references_from_markdown(answer: str) -> tuple[str, list[CitationRecord]]:
    marker_match = re.search(r"\n#{2,3}\s+References\s*\n", answer, flags=re.IGNORECASE)
    if not marker_match:
        return answer.strip(), []

    body = answer[: marker_match.start()].rstrip()
    references_block = answer[marker_match.end() :].strip()
    citations: list[CitationRecord] = []

    for line in references_block.splitlines():
        match = re.search(r"[-*]\s*\[\d+\]\s+(.+)", line.strip())
        if not match:
            continue
        raw_source = match.group(1).strip()
        citations.append(
            CitationRecord(
                source=raw_source.replace("\\", "/").split("/")[-1],
                text="",
                domain_module="general",
            )
        )

    return body, citations


def parse_lightrag_response(payload: Any) -> tuple[str, list[CitationRecord]]:
    response_data = _coerce_payload_to_dict(payload)
    if response_data is None:
        raw = str(payload).strip()
        try:
            response_data = json.loads(raw)
        except json.JSONDecodeError:
            response_data = _parse_ndjson_payload(raw)
            if response_data is None:
                return _extract_references_from_markdown(raw)

    answer = str(response_data.get("response", "")).strip()
    references = response_data.get("references", []) or []
    citations: list[CitationRecord] = []

    for reference in references:
        if not isinstance(reference, dict):
            continue
        file_path = reference.get("file_path", "")
        content = reference.get("content", [])
        text = ""
        if isinstance(content, list) and content:
            text = "\n".join(str(part) for part in content[:2])
        elif isinstance(content, str):
            text = content

        citations.append(
            CitationRecord(
                source=str(file_path).replace("\\", "/").split("/")[-1],
                text=text,
                domain_module="general",
            )
        )

    if not citations:
        return _extract_references_from_markdown(answer)

    return answer, citations


def infer_should_escalate(answer: str) -> bool:
    normalized = answer.lower().strip()
    if not normalized:
        return True

    insufficiency_signals = [
        "insufficient",
        "not enough information",
        "not provided in the context",
        "not available in the provided context",
        "cannot determine",
        "i don't know",
        "i do not know",
        "documentation available does not cover",
    ]
    return any(signal in normalized for signal in insufficiency_signals)


class LightRAGRuntime(AbstractContextManager["LightRAGRuntime"]):
    def __init__(self, data_dir: str | Path, benchmark_items: list[BenchmarkItem], top_k: int):
        self.data_dir = Path(data_dir)
        self.benchmark_items = benchmark_items
        self.top_k = top_k
        self.allowed_sources = collect_benchmark_sources(benchmark_items)
        self._tempdir_path: Path | None = None
        self._rag = None

    def build(self) -> "LightRAGRuntime":
        from lightrag import LightRAG
        from lightrag.llm.openai import openai_complete_if_cache, openai_embed
        from lightrag.utils import wrap_embedding_func_with_attrs

        if str(settings.llm_provider.value) != "openai":
            raise RuntimeError(
                "The LightRAG baseline currently supports only the OpenAI-backed evaluation path. "
                "Set LLM_PROVIDER=openai before running --system lightrag_baseline."
            )
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required to run the LightRAG baseline.")

        self._tempdir_path = Path(tempfile.mkdtemp(prefix="cag_lightrag_"))

        async def llm_model_func(
            prompt,
            system_prompt=None,
            history_messages=None,
            keyword_extraction=False,
            **kwargs,
        ) -> str:
            return await openai_complete_if_cache(
                settings.active_model_id,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                api_key=settings.openai_api_key or None,
                **kwargs,
            )

        @wrap_embedding_func_with_attrs(
            embedding_dim=settings.embedding_dim,
            max_token_size=8192,
            model_name=settings.embedding_model,
        )
        async def embedding_func(texts: list[str]):
            return await openai_embed.func(
                texts,
                model=settings.embedding_model,
                api_key=settings.openai_api_key or None,
                embedding_dim=settings.embedding_dim,
            )

        self._rag = LightRAG(
            working_dir=str(self._tempdir_path),
            llm_model_func=llm_model_func,
            llm_model_name=settings.active_model_id,
            embedding_func=embedding_func,
            top_k=self.top_k,
            chunk_top_k=min(self.top_k, 20),
            enable_llm_cache=False,
            enable_llm_cache_for_entity_extract=False,
        )
        asyncio.run(self._rag.initialize_storages())

        docs = load_selected_documents(self.data_dir, set(self.allowed_sources))
        texts = [doc.page_content for doc in docs]
        file_paths = [doc.metadata.get("source", doc.metadata.get("filename", "unknown")) for doc in docs]
        self._rag.insert(texts, file_paths=file_paths)
        return self

    def query(self, question_id: str, question: str) -> SystemOutput:
        from lightrag import QueryParam
        import time

        if self._rag is None:
            raise RuntimeError("LightRAG runtime has not been built.")

        started = time.perf_counter()
        raw_response = self._rag.query(
            question,
            param=QueryParam(
                mode="mix",
                top_k=self.top_k,
                chunk_top_k=min(self.top_k, 20),
                include_references=True,
                enable_rerank=True,
            ),
        )
        latency_ms = (time.perf_counter() - started) * 1000
        answer, citations = parse_lightrag_response(raw_response)
        cost_estimate = estimate_cost_units(question, answer, multiplier=2.0)

        return SystemOutput(
            question_id=question_id,
            question=question,
            system="lightrag_baseline",
            answer=answer,
            citations=citations,
            query_type=classify_query_type(question),
            confidence=0.0,
            hallucination_risk=0.0,
            should_escalate=infer_should_escalate(answer),
            latency_ms=round(latency_ms, 2),
            cost_estimate=cost_estimate,
            node_trace=["LIGHTRAG"],
        )

    def close(self) -> None:
        if self._rag is not None:
            try:
                asyncio.run(self._rag.finalize_storages())
            except Exception:
                pass
            self._rag = None

        gc.collect()

        if self._tempdir_path is not None:
            cleanup_temp_path(self._tempdir_path)
            self._tempdir_path = None

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
        return None
