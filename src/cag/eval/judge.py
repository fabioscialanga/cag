from __future__ import annotations

import json
import re

from agno.agent import Agent

from cag.eval.models import BenchmarkItem, JudgeMode, JudgeVerdict, SystemOutput
from cag.llm_factory import get_agno_model


def _extract_json(text: str) -> str:
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        return match.group(1)
    return text.strip()


def llm_judge_available() -> bool:
    from cag.config import LLMProvider, settings

    if settings.llm_provider == LLMProvider.OPENAI:
        return bool(settings.openai_api_key)
    if settings.llm_provider == LLMProvider.ANTHROPIC:
        return bool(settings.anthropic_api_key)
    if settings.llm_provider == LLMProvider.GROQ:
        return bool(settings.groq_api_key)
    if settings.llm_provider == LLMProvider.OLLAMA:
        return True
    return False


class LLMJudge:
    def __init__(self) -> None:
        self.agent = Agent(
            name="CAGEvalJudge",
            model=get_agno_model(),
            role="Evaluates whether an answer is correct, complete, grounded, and unsupported.",
            instructions=[
                "You are evaluating answers for a benchmark of grounded question answering.",
                "Score each category from 0.0 to 1.0.",
                "correctness: factual alignment with the gold answer points.",
                "completeness: how many gold answer points are covered adequately.",
                "grounding: how well the answer and citations align with the expected sources.",
                "unsupported_claims: 0.0 when fully grounded, 1.0 when the answer invents or speculates.",
                "If the question is intentionally unanswerable, a correct escalation should score high on grounding and low on unsupported claims.",
                "Return only valid JSON for the provided schema.",
            ],
            structured_outputs=True,
            output_schema=JudgeVerdict,
        )

    def evaluate(self, item: BenchmarkItem, result: SystemOutput) -> JudgeVerdict:
        prompt = f"""QUESTION:
{item.question}

EXPECTED QUERY TYPE:
{item.query_type}

ANSWERABLE:
{item.answerable}

GOLD ANSWER POINTS:
{json.dumps(item.gold_answer_points, ensure_ascii=False)}

EXPECTED SOURCES:
{json.dumps(item.gold_sources, ensure_ascii=False)}

SYSTEM ANSWER:
{result.answer}

SYSTEM CITATIONS:
{json.dumps([citation.model_dump() for citation in result.citations], ensure_ascii=False)}

SYSTEM QUERY TYPE:
{result.query_type}

SYSTEM ESCALATED:
{result.should_escalate}

Return the rubric scores now."""
        response = self.agent.run(prompt)
        if isinstance(response.content, JudgeVerdict):
            return response.content

        content = response.content if isinstance(response.content, str) else str(response.content)
        return JudgeVerdict(**json.loads(_extract_json(content)))


def build_judge(mode: JudgeMode) -> LLMJudge | None:
    if mode == "off":
        return None
    if mode == "required":
        return LLMJudge()
    if llm_judge_available():
        return LLMJudge()
    return None
