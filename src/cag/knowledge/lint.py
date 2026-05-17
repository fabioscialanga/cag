"""Lint checks for the DB-first compiled knowledge layer."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from cag.knowledge.store import connect, initialize, log_event, rows, store_contradiction

_NEGATION_RE = re.compile(r"\b(not|never|no|cannot|can't|must not|should not)\b", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]{4,}")
_STOPWORDS = {
    "about", "after", "before", "from", "have", "must", "only", "should", "that", "their", "there", "this",
    "when", "where", "with",
}


@dataclass
class KnowledgeLintReport:
    stale_claims: list[str] = field(default_factory=list)
    contradictions: list[dict[str, str]] = field(default_factory=list)
    orphan_topics: list[str] = field(default_factory=list)
    missing_evidence_claims: list[str] = field(default_factory=list)

    @property
    def issue_count(self) -> int:
        return (
            len(self.stale_claims)
            + len(self.contradictions)
            + len(self.orphan_topics)
            + len(self.missing_evidence_claims)
        )

    def model_dump(self) -> dict:
        return {
            "stale_claims": self.stale_claims,
            "contradictions": self.contradictions,
            "orphan_topics": self.orphan_topics,
            "missing_evidence_claims": self.missing_evidence_claims,
            "issue_count": self.issue_count,
        }


def _signature(text: str) -> tuple[bool, frozenset[str]]:
    normalized = text.lower()
    negated = bool(_NEGATION_RE.search(normalized))
    tokens = frozenset(
        token
        for token in _TOKEN_RE.findall(normalized)
        if token not in _STOPWORDS and not _NEGATION_RE.fullmatch(token)
    )
    return negated, tokens


def lint_knowledge(db_path: str | Path | None = None, record: bool = True) -> KnowledgeLintReport:
    """Run deterministic lint checks over the compiled knowledge DB."""

    connection = connect(db_path)
    try:
        initialize(connection)
        report = KnowledgeLintReport()

        report.stale_claims = [
            row["id"]
            for row in rows(connection, "SELECT id FROM claims WHERE status = 'stale'")
        ]
        report.missing_evidence_claims = [
            row["id"]
            for row in rows(
                connection,
                """
                SELECT claims.id
                FROM claims
                LEFT JOIN claim_evidence ON claim_evidence.claim_id = claims.id
                WHERE claim_evidence.claim_id IS NULL
                """,
            )
        ]
        report.orphan_topics = [
            row["id"]
            for row in rows(
                connection,
                """
                SELECT topics.id
                FROM topics
                LEFT JOIN topic_claims ON topic_claims.topic_id = topics.id
                WHERE topic_claims.topic_id IS NULL
                """,
            )
        ]

        active_claims = rows(connection, "SELECT id, claim_text FROM claims WHERE status = 'active'")
        for left_index, left in enumerate(active_claims):
            left_negated, left_terms = _signature(left["claim_text"])
            if len(left_terms) < 3:
                continue
            for right in active_claims[left_index + 1:]:
                right_negated, right_terms = _signature(right["claim_text"])
                if left_negated == right_negated:
                    continue
                overlap = left_terms & right_terms
                if len(overlap) >= 3:
                    reason = f"Opposite polarity with shared terms: {', '.join(sorted(overlap)[:6])}"
                    report.contradictions.append(
                        {
                            "left_claim_id": left["id"],
                            "right_claim_id": right["id"],
                            "reason": reason,
                        }
                    )
                    if record:
                        store_contradiction(
                            connection,
                            left_claim_id=left["id"],
                            right_claim_id=right["id"],
                            reason=reason,
                        )

        if record:
            log_event(connection, "lint", report.model_dump())
            connection.commit()
        return report
    finally:
        connection.close()

