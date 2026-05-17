"""Answer review and cleanup agent."""
from __future__ import annotations

import re

from cag.agents.models import ReasoningOutput


DETAIL_HEADINGS = (
    "Dettagli supportati",
    "Supported details",
)


def run_review_agent(
    *,
    query: str,
    output: ReasoningOutput,
    ranked_chunks: list[dict],
    gaps: list[str],
    response_language: str,
) -> ReasoningOutput:
    """Review and polish a generated answer without adding unsupported facts."""

    reviewed_answer = _remove_noisy_detail_section(output.answer)
    reviewed_answer = _normalize_answer_spacing(reviewed_answer)

    if reviewed_answer == output.answer:
        return output

    reason = output.hallucination_reason.strip()
    review_note = (
        "ReviewAgent ha rimosso dettagli duplicati o frammentati."
        if response_language == "it"
        else "ReviewAgent removed duplicated or fragmented details."
    )
    if reason and review_note not in reason:
        reason = f"{reason} {review_note}"
    elif not reason:
        reason = review_note

    return output.model_copy(
        update={
            "answer": reviewed_answer,
            "hallucination_reason": reason,
        }
    )


def _remove_noisy_detail_section(answer: str) -> str:
    sections = re.split(r"\n\n+", answer.strip())
    if len(sections) < 2:
        return answer

    lead = sections[0].strip()
    kept_sections = [lead]
    for section in sections[1:]:
        if not _is_detail_section(section):
            kept_sections.append(section.strip())
            continue

        bullets = _extract_bullets(section)
        useful_bullets = [
            bullet for bullet in bullets
            if not _is_fragmented_or_duplicate_detail(lead, bullet)
        ]
        if useful_bullets:
            heading = section.splitlines()[0].strip()
            kept_sections.append(f"{heading}\n" + "\n".join(f"- {bullet}" for bullet in useful_bullets))

    return "\n\n".join(part for part in kept_sections if part).strip()


def _is_detail_section(section: str) -> bool:
    first_line = section.strip().splitlines()[0].strip(" :")
    return any(first_line.lower() == heading.lower() for heading in DETAIL_HEADINGS)


def _extract_bullets(section: str) -> list[str]:
    bullets: list[str] = []
    for line in section.splitlines()[1:]:
        clean = line.strip()
        if clean.startswith("- "):
            clean = clean[2:].strip()
        if clean:
            bullets.append(clean)
    return bullets


def _is_fragmented_or_duplicate_detail(lead: str, bullet: str) -> bool:
    clean_bullet = bullet.strip()
    if len(clean_bullet.split()) < 6:
        return True
    if re.match(r"^[\d,.;:/\-\s]+[A-Za-zÀ-ÿ]", clean_bullet):
        return True

    lead_terms = set(_terms(lead))
    bullet_terms = set(_terms(clean_bullet))
    if not bullet_terms:
        return True
    overlap = len(lead_terms & bullet_terms) / max(1, len(bullet_terms))
    if overlap >= 0.62:
        return True

    address_markers = {"via", "roma", "napoli", "lazio", "campania"}
    if len(address_markers & bullet_terms) >= 2 and len(address_markers & lead_terms) >= 2:
        return True

    return False


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[A-Za-zÀ-ÿ0-9]+", text)
        if len(token) > 2
    ]


def _normalize_answer_spacing(answer: str) -> str:
    answer = re.sub(r"[ \t]+\n", "\n", answer.strip())
    answer = re.sub(r"\n{3,}", "\n\n", answer)
    return answer
