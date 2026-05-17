from __future__ import annotations

import math
import re

STOPWORDS = {
    "a", "about", "and", "do", "for", "how", "i", "in", "is", "of", "on",
    "the", "to", "what", "which",
    "che", "chi", "ciao", "come", "con", "cosa", "dei", "del", "della", "di",
    "gli", "il", "la", "le", "lo", "mi", "per", "quale", "quali", "siete",
    "avete", "dove", "qualche", "sono", "trova", "un", "una", "vi", "voi",
}

CONCEPT_ALIASES: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("di cosa", "cosa fate", "occupate", "chi siete", "what do you do", "about"),
        "overview company mission purpose organization business services products solutions "
        "azienda profilo aziendale missione servizi prodotti soluzioni specializzata",
    ),
    (
        ("servizi", "prodotti", "soluzioni", "approfond", "dettagli", "services", "products", "solutions"),
        "servizi prodotti soluzioni consulenza teamsystem alyante infrastrutture networking cloud sicurezza "
        "tecnologie innovative information technology azienda clienti competenze supporto",
    ),
    (
        ("dove si trova", "dove si trova studio", "dove operate", "sede", "sedi", "indirizzo", "location", "address"),
        "sede sedi indirizzo via roma napoli cornelia benedetto brin filiale studio 81 data systems",
    ),
    (
        ("nuvola", "cloud", "dati nella nuvola", "applicazioni nella nuvola"),
        "cloud nuvola applicazioni dati accessibilita accessibilità servizi infrastruttura iperconvergenza networking",
    ),
    (
        ("gestionale", "gestionali", "software gestionale", "erp", "management software", "management solutions"),
        "gestionale gestionali software gestionale erp enterprise resource planning business software "
        "management solutions prodotti soluzioni teamsystem alyante acg enterprise hr all in one "
        "amministrazione finanza logistica produzione risorse umane",
    ),
    (
        (
            "sviluppate applicazioni",
            "sviluppate app",
            "sviluppate software",
            "sviluppo applicazioni",
            "sviluppo software",
            "applicazioni",
            "applications",
            "custom software",
        ),
        "sviluppo sviluppate sviluppiamo applicazioni app software sviluppo software personalizzato "
        "custom software web siti web crm data integration digital marketing soluzioni tecnologiche "
        "servizi information technology",
    ),
    (
        ("memory footprint", "memory requirement", "memoria", "ram"),
        "ram memory gb requirement prerequisite minimum hardware memoria requisito",
    ),
    (
        ("credential", "programmatic access", "api access", "token", "api key", "chiave"),
        "api key token authentication credential generate rotate access dashboard integrations",
    ),
    (
        ("mint", "fresh", "generate", "create", "crea", "genera"),
        "generate create new rotate fresh issue",
    ),
    (
        ("throttled", "throttle", "too many requests", "rate limit", "429"),
        "rate limit exceeded http 429 nx 1003 request frequency reduce retry after throttled",
    ),
    (
        ("network admission list", "admission list", "allowlist", "whitelist", "access list"),
        "ip allowlist cidr ipv4 ipv6 network access workspace admin rules ranges",
    ),
    (
        ("tune", "configure", "settings", "configura", "imposta"),
        "configure configuration settings options rules update",
    ),
    (
        ("fix", "solve", "resolve", "risolvere", "errore", "problema"),
        "resolution troubleshooting checks symptoms causes workaround error incident",
    ),
    (
        ("step by step", "how do i", "come faccio", "procedura", "passo"),
        "procedure steps ordered navigation menu path workflow",
    ),
)


def normalize_text(text: str) -> str:
    normalized = "".join(character.lower() if character.isalnum() or character.isspace() else " " for character in text)
    return re.sub(r"([a-zà-ÿ])\1{2,}", r"\1\1", normalized)


def extract_keywords(text: str) -> list[str]:
    keywords: list[str] = []
    for token in normalize_text(text).split():
        if len(token) <= 2 or token in STOPWORDS or token in keywords:
            continue
        keywords.append(token)
    return keywords


def expand_query_concepts(query: str) -> str:
    """Add deterministic concept aliases for lexical fallbacks and metadata search."""

    lowered = query.lower()
    additions = [
        expansion
        for markers, expansion in CONCEPT_ALIASES
        if any(marker in lowered for marker in markers)
    ]
    if not additions:
        return query
    return " ".join([query, *additions])


def document_terms(doc) -> list[str]:
    return extract_keywords(
        f"{doc.page_content} "
        f"{doc.metadata.get('filename', doc.metadata.get('source', 'N/A'))} "
        f"{doc.metadata.get('domain_module', 'general')}"
    )


def _corpus_validated_terms(
    terms: list[str],
    *,
    doc_frequencies: dict[str, int] | None = None,
    corpus_size: int | None = None,
    max_df_ratio: float = 0.65,
) -> list[str]:
    if not doc_frequencies or not corpus_size:
        return terms
    validated: list[str] = []
    max_doc_frequency = max(1, int(corpus_size * max_df_ratio))
    for term in terms:
        freq = int(doc_frequencies.get(term, 0))
        if freq <= 0:
            continue
        if freq <= max_doc_frequency:
            validated.append(term)
    return validated


def build_weighted_query_terms(
    query: str,
    lexical_queries: list[str] | None = None,
    *,
    corpus_doc_frequencies: dict[str, int] | None = None,
    corpus_size: int | None = None,
) -> dict[str, float]:
    weighted_terms: dict[str, float] = {}
    for term in extract_keywords(query):
        weighted_terms[term] = max(weighted_terms.get(term, 0.0), 2.0)

    expanded_query_terms = _corpus_validated_terms(
        extract_keywords(expand_query_concepts(query)),
        doc_frequencies=corpus_doc_frequencies,
        corpus_size=corpus_size,
    )
    for term in expanded_query_terms:
        weighted_terms[term] = max(weighted_terms.get(term, 0.0), 1.2)

    for lexical_query in lexical_queries or []:
        for term in extract_keywords(lexical_query):
            weighted_terms[term] = max(weighted_terms.get(term, 0.0), 1.0)
        expanded_lexical_terms = _corpus_validated_terms(
            extract_keywords(expand_query_concepts(lexical_query)),
            doc_frequencies=corpus_doc_frequencies,
            corpus_size=corpus_size,
        )
        for term in expanded_lexical_terms:
            weighted_terms[term] = max(weighted_terms.get(term, 0.0), 0.8)

    return weighted_terms


def discriminative_document_score(
    doc,
    weighted_query_terms: dict[str, float],
    doc_frequencies: dict[str, int],
    corpus_size: int,
) -> float:
    if not weighted_query_terms:
        return 0.0

    terms = document_terms(doc)
    if not terms:
        return 0.0

    term_counts: dict[str, int] = {}
    for term in terms:
        term_counts[term] = term_counts.get(term, 0) + 1

    doc_length = len(terms)
    score = 0.0
    for term, query_weight in weighted_query_terms.items():
        frequency = term_counts.get(term, 0)
        if frequency == 0:
            continue
        document_frequency = doc_frequencies.get(term, 0)
        idf = math.log((corpus_size + 1) / (document_frequency + 0.5)) + 1.0
        bm25_tf = (frequency * 2.2) / (frequency + 1.2 + (0.75 * doc_length / 120.0))
        score += query_weight * idf * bm25_tf

    return score


def document_priority_boost(doc) -> float:
    if doc.metadata.get("domain_module") == "document_profile":
        return 1.5
    if doc.metadata.get("compiled_knowledge"):
        return 0.5
    return 0.0


def dedupe_documents(
    query: str,
    docs: list,
    lexical_queries: list[str] | None = None,
    *,
    corpus_doc_frequencies: dict[str, int] | None = None,
    corpus_size: int | None = None,
) -> list:
    weighted_query_terms = build_weighted_query_terms(
        query,
        lexical_queries,
        corpus_doc_frequencies=corpus_doc_frequencies,
        corpus_size=corpus_size,
    )
    keyed: dict[tuple[str, int, str], object] = {}
    original_positions: dict[tuple[str, int, str], int] = {}
    for position, doc in enumerate(docs):
        key = (
            str(doc.metadata.get("filename", doc.metadata.get("source", "N/A"))),
            int(doc.metadata.get("chunk_index", 0)),
            doc.page_content[:160],
        )
        keyed.setdefault(key, doc)
        original_positions.setdefault(key, position)

    unique_docs = list(keyed.values())
    doc_frequencies: dict[str, int] = {}
    for doc in unique_docs:
        for term in set(document_terms(doc)):
            doc_frequencies[term] = doc_frequencies.get(term, 0) + 1

    corpus_size = max(1, len(unique_docs))
    unique_docs.sort(
        key=lambda doc: (
            discriminative_document_score(doc, weighted_query_terms, doc_frequencies, corpus_size)
            + document_priority_boost(doc),
            -original_positions[
                (
                    str(doc.metadata.get("filename", doc.metadata.get("source", "N/A"))),
                    int(doc.metadata.get("chunk_index", 0)),
                    doc.page_content[:160],
                )
            ],
        ),
        reverse=True,
    )
    return unique_docs
