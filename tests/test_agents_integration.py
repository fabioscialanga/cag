from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path.cwd() / "src"))

from cag.agents.models import Citation, ReasoningOutput, RankedChunk, RetrievalOutput
from cag.graph.graph import run_query


def test_run_query_basic():
    """Run a lightweight end-to-end graph invocation with mocked agent outputs."""

    class FakeDoc:
        page_content = "Open Settings > Workflow and complete the required fields."
        metadata = {
            "filename": "team_handbook_setup.txt",
            "source": "team_handbook_setup.txt",
            "domain_module": "workflow",
            "chunk_index": 0,
        }

    retrieval_output = RetrievalOutput(
        chunks_ranked=[
            RankedChunk(
                content=FakeDoc.page_content,
                source="team_handbook_setup.txt",
                domain_module="workflow",
                relevance_score=0.91,
                relevance_reason="Directly answers the setup question.",
            )
        ],
        gaps=[],
        relevance_score=0.91,
        summary="Relevant onboarding workflow instructions found.",
    )
    reasoning_output = ReasoningOutput(
        answer="Open Settings > Workflow and complete the required fields.",
        query_type="CONFIGURATION",
        confidence=0.86,
        citations=[
            Citation(
                text="Open Settings > Workflow and complete the required fields.",
                source="team_handbook_setup.txt",
                domain_module="workflow",
            )
        ],
        hallucination_risk=0.05,
        hallucination_reason="Answer is fully grounded in the provided chunk.",
    )

    with patch("cag.graph.nodes.similarity_search", return_value=[FakeDoc()]), patch(
        "cag.graph.nodes.run_retrieval_agent",
        return_value=retrieval_output,
    ), patch(
        "cag.graph.nodes.run_reasoning_agent",
        return_value=reasoning_output,
    ):
        result = run_query("What fields are required to configure the workflow?")

    assert isinstance(result, dict)
    for key in ("answer", "node_trace", "should_escalate", "confidence", "query_type"):
        assert key in result
    assert isinstance(result.get("answer", ""), str)
    assert isinstance(result.get("node_trace", []), list)
    assert isinstance(result.get("should_escalate", False), bool)
    assert result["query_type"] == "CONFIGURATION"
