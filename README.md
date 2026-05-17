# CAG

![Status](https://img.shields.io/badge/status-preview-yellow)
![Version](https://img.shields.io/badge/version-0.2-blue)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**EN:** A safer RAG pipeline that selects evidence, validates answers, and refuses when docs are insufficient.  
**IT:** Una pipeline RAG piu' sicura che seleziona evidenze, valida le risposte e rifiuta quando la documentazione non basta.

## EN - What Is CAG?

CAG is a graph-driven document QA runtime for builders who need more control than plain `retrieve -> generate`.

Core loop:

`ENTRY -> RETRIEVE -> SELECT_CONTEXT -> REASON -> VALIDATE -> EXIT`

Why it matters:
- Query-type aware retrieval strategy.
- Explicit context selection before generation.
- Validation with confidence, hallucination risk, and escalation.
- Better inspectability for benchmarking and debugging.

## EN - Core Idea

The central idea behind CAG is simple and strict:

`Grounding quality is mostly a retrieval-orchestration problem, not only a generation problem.`

Most RAG failures happen before the final answer is written:
- wrong retrieval strategy for the question type
- redundant or weak context passed to the model
- no explicit validation of evidence sufficiency

CAG treats these as first-class architecture concerns.  
Instead of just improving prompts, it improves the decision process that controls retrieval, context selection, and refusal/escalation.

## IT - Cos'e CAG?

CAG e' un runtime di document QA guidato da grafo, pensato per chi vuole piu' controllo rispetto al classico `retrieve -> generate`.

Loop principale:

`ENTRY -> RETRIEVE -> SELECT_CONTEXT -> REASON -> VALIDATE -> EXIT`

Perche' e' utile:
- Strategia di retrieval adattata al tipo di query.
- Selezione esplicita del contesto prima della generazione.
- Validazione con confidence, rischio allucinazione ed escalation.
- Maggiore ispezionabilita' per benchmark e debug.

## IT - Idea Centrale

L'idea centrale di CAG e' semplice e rigorosa:

`La qualita' del grounding e' soprattutto un problema di orchestrazione del retrieval, non solo di generazione.`

La maggior parte degli errori RAG nasce prima della risposta finale:
- strategia di retrieval sbagliata per il tipo di domanda
- contesto ridondante o debole passato al modello
- assenza di validazione esplicita della sufficienza delle evidenze

CAG tratta questi punti come scelte architetturali di primo livello.  
Invece di migliorare solo i prompt, migliora il processo decisionale che governa retrieval, selezione contesto e refusal/escalation.

## EN - CAG Logic (Full Flow)

The runtime is a stateful graph with explicit decision points:

`ENTRY -> CONTEXTUALIZE_QUERY -> RETRIEVE -> SELECT_CONTEXT -> REASON -> REVIEW -> POST_GROUNDING -> VALIDATE -> EXIT`

### 1. ENTRY
- classifies query type (`GENERAL`, `PROCEDURAL`, `DIAGNOSTIC`, `CONFIGURATION`)
- infers question scope (domain vs consultative vs personal)
- detects response language
- chooses retrieval strategy and initializes retrieval plan
- resets runtime fields for a clean turn

### 2. CONTEXTUALIZE_QUERY
- checks conversation history
- rewrites vague follow-ups into explicit retrieval-ready queries
- preserves original intent while injecting context anchors from prior assistant answers

### 3. RETRIEVE
- builds query variants (rewrites + concept expansion)
- queries document profiles first (Document Map)
- if needed, falls back to global semantic retrieval
- optionally merges hybrid lexical candidates
- deduplicates and reranks chunks with discriminative lexical scoring

### 4. SELECT_CONTEXT
- ranks chunks for answerability, not only similarity
- applies relevance/category/source diversity balancing
- keeps a compact evidence set under context budget limits
- produces ranked chunks + explicit gaps

### 5. REASON + REVIEW
- generates answer from selected evidence only
- emits confidence, citations, hallucination risk
- review pass tightens unsupported language and citation discipline

### 6. POST_GROUNDING
- sentence-level support checks against selected evidence
- detects unsupported claims
- adjusts confidence/risk when support is weak

### 7. VALIDATE (Control Node)
- decides one of three actions:
  - `EXIT` with final answer
  - `REASON` retry (narrower/safer generation)
  - `RETRIEVE` adaptive retry (expanded query/strategy/top-k)
- escalates when evidence is insufficient or risk remains high

### 8. EXIT
- returns final answer or escalation message
- attaches diagnostics (`node_trace`, risk/confidence, fallback flags, actions)

## EN - Decision Rules (Why It Refuses or Retries)

CAG does not trust generation by default. It validates:
- evidence sufficiency (`relevance_threshold`, moderate support checks)
- confidence (`confidence_threshold`)
- hallucination risk (`hallucination_threshold`)
- unsupported-claim patterns in the produced answer

If validation fails:
- first try adaptive retrieval retry (if enabled and conditions match)
- then try constrained reason retry
- finally escalate instead of guessing

## EN - Retrieval Design Details

- Query typing drives strategy (`semantic`, `multi_evidence`, `hierarchical`)
- Document Map is prioritized to reduce random top-k drift
- Lexical discriminative scoring reduces redundant generic chunks
- Hybrid lexical retrieval can be enabled for sparse semantic cases
- Procedural flows can include neighboring chunks for sequence continuity

## EN - Reasoning and Safety Contract

The contract is:
- answer only from evidence
- cite what was used
- state gaps explicitly
- escalate when support is inadequate

This makes CAG slower than naive RAG, but far more inspectable and safer for high-stakes document QA.

## EN - ASCII Diagrams

Full ASCII diagrams live here:
- [CAG ASCII Flow](docs/cag-flow-ascii.md)

Quick inline view:

```text
ENTRY -> CONTEXTUALIZE_QUERY -> RETRIEVE -> SELECT_CONTEXT -> REASON -> REVIEW -> POST_GROUNDING -> VALIDATE -> EXIT
                                              ^                                            |
                                              |                                            |
                                              +---------------- RETRIEVE RETRY <-----------+
                                                                   ^
                                                                   |
                                                           REASON RETRY
```

## EN - Animated GIF Walkthroughs

Add animated walkthroughs in `docs/assets/`:

- `docs/assets/cag-flow.gif`
- `docs/assets/cag-validate-routing.gif`
- `docs/assets/cag-retrieval-ladder.gif`

Embed examples:

![CAG Full Flow](docs/assets/cag-flow.gif)
![CAG Validate Routing](docs/assets/cag-validate-routing.gif)
![CAG Retrieval Ladder](docs/assets/cag-retrieval-ladder.gif)

## IT - Logica CAG (Flusso Completo)

Il runtime e' un grafo stateful con punti decisionali espliciti:

`ENTRY -> CONTEXTUALIZE_QUERY -> RETRIEVE -> SELECT_CONTEXT -> REASON -> REVIEW -> POST_GROUNDING -> VALIDATE -> EXIT`

### 1. ENTRY
- classifica il tipo di query (`GENERAL`, `PROCEDURAL`, `DIAGNOSTIC`, `CONFIGURATION`)
- identifica lo scope della domanda (domain/consultative/personal)
- rileva la lingua di risposta
- sceglie strategia retrieval e inizializza il piano
- resetta i campi runtime del turno

### 2. CONTEXTUALIZE_QUERY
- usa la history conversazionale
- riscrive follow-up vaghi in query piu' esplicite
- mantiene intento originale aggiungendo anchor dal contesto precedente

### 3. RETRIEVE
- costruisce varianti query (rewrite + concept expansion)
- interroga prima i document profile (Document Map)
- se necessario, fallback su retrieval semantico globale
- opzionalmente unisce candidati lexical hybrid
- deduplica e reranka con scoring lessicale discriminativo

### 4. SELECT_CONTEXT
- ordina i chunk per answerability, non solo similarita'
- bilancia rilevanza/diversita' categoria/diversita' fonte
- mantiene un set evidenze compatto entro i limiti di budget
- produce chunk ordinati + gap espliciti

### 5. REASON + REVIEW
- genera risposta solo dal contesto selezionato
- emette confidence, citazioni, rischio allucinazione
- passaggio review per stringere linguaggio non supportato

### 6. POST_GROUNDING
- controlli frase-per-frase contro le evidenze selezionate
- rileva claim non supportati
- aggiorna confidence/risk quando il supporto e' debole

### 7. VALIDATE (Nodo di Controllo)
- decide una delle tre azioni:
  - `EXIT` con risposta finale
  - retry `REASON` (generazione piu' stretta/sicura)
  - retry `RETRIEVE` adattivo (query/strategia/top-k espansi)
- scala quando evidenza e' insufficiente o rischio resta alto

### 8. EXIT
- restituisce risposta finale o messaggio di escalation
- allega diagnostica (`node_trace`, risk/confidence, fallback, azioni)

## IT - Regole Decisionali (Perche' Riprova o Escala)

CAG non si fida della generazione in automatico. Valida:
- sufficienza evidenze (`relevance_threshold` e supporti moderati)
- confidence (`confidence_threshold`)
- rischio allucinazione (`hallucination_threshold`)
- pattern di claim non supportati nella risposta

Se la validazione fallisce:
- prima retry retrieval adattivo (se abilitato)
- poi retry reason vincolato
- infine escalation invece di indovinare

## IT - Diagrammi ASCII

I diagrammi ASCII completi sono qui:
- [Flusso ASCII CAG](docs/cag-flow-ascii.md)

Vista inline rapida:

```text
ENTRY -> CONTEXTUALIZE_QUERY -> RETRIEVE -> SELECT_CONTEXT -> REASON -> REVIEW -> POST_GROUNDING -> VALIDATE -> EXIT
                                              ^                                            |
                                              |                                            |
                                              +---------------- RETRIEVE RETRY <-----------+
                                                                   ^
                                                                   |
                                                           REASON RETRY
```

## IT - GIF Animate (Walkthrough)

Aggiungi GIF animate in `docs/assets/`:

- `docs/assets/cag-flow.gif`
- `docs/assets/cag-validate-routing.gif`
- `docs/assets/cag-retrieval-ladder.gif`

## EN - Start Here

- [Quickstart](docs/quickstart.md)
- [API](docs/api.md)
- [Configuration](docs/configuration.md)
- [Evaluation](docs/evaluation.md)
- [Examples](docs/examples.md)
- [Integration](docs/integration.md)
- [Architecture](docs/cag-architecture.md)
- [Knowledge Compiler](docs/knowledge-compiler.md)
- [Community](docs/community.md)

Fast path:

```bash
pip install -e ".[dev,eval,api]"
cag demo --reset --json
```

## IT - Inizia Da Qui

- [Quickstart](docs/quickstart.md)
- [API](docs/api.md)
- [Configurazione](docs/configuration.md)
- [Valutazione](docs/evaluation.md)
- [Esempi](docs/examples.md)
- [Integrazione](docs/integration.md)
- [Architettura](docs/cag-architecture.md)
- [Knowledge Compiler](docs/knowledge-compiler.md)
- [Community](docs/community.md)

Percorso veloce:

```bash
pip install -e ".[dev,eval,api]"
cag demo --reset --json
```

## EN - 5 Minute Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -e ".[dev,eval,api]"
cag demo --reset --json
```

Expected output: JSON answer with confidence, citations, escalation state, and node trace.

## IT - Quickstart In 5 Minuti

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -e ".[dev,eval,api]"
cag demo --reset --json
```

Output atteso: risposta JSON con confidence, citazioni, stato escalation e node trace.

## EN - Community-First Goals

CAG is a public repository and aims to become a community reference project.

Current priorities:
- Reproducible benchmarks and transparent claims.
- Small, frequent releases with clear changelog entries.
- Strong onboarding and reliable local demo path.
- Honest documentation of limits and tradeoffs.

## IT - Obiettivi Community-First

CAG e' un repository pubblico e punta a diventare un punto di riferimento per la community.

Priorita' attuali:
- Benchmark riproducibili e claim trasparenti.
- Release piccole e frequenti con changelog chiaro.
- Onboarding solido e demo locale affidabile.
- Documentazione onesta di limiti e tradeoff.

## EN - Data Safety (Raw Documents)

- `data/raw` is for local ingestion only.
- Source documents used for experiments should not be committed to Git.
- Keep only sanitized demo corpora intended for public distribution.
- Never commit API keys, secrets, customer data, or confidential files.

Current repository policy:
- raw `.pdf`, `.txt`, `.md` under `data/raw/` are ignored by `.gitignore`
- benchmark corpus stays in `data/benchmark_corpus/`

## IT - Sicurezza Dati (Documenti Raw)

- `data/raw` e' solo per ingest locale.
- I documenti sorgente usati negli esperimenti non vanno committati su Git.
- Nel repository pubblico devono restare solo corpora demo sanitizzati.
- Non committare mai API key, segreti, dati cliente o file confidenziali.

Policy attuale del repository:
- i file raw `.pdf`, `.txt`, `.md` in `data/raw/` sono ignorati da `.gitignore`
- il corpus benchmark rimane in `data/benchmark_corpus/`

## EN - Known Limits

- Benchmark coverage is still limited compared to real production variability.
- Cost and latency overhead are higher than simple RAG.
- Some retrieval gains are dataset-sensitive and should be validated per domain.
- Multi-provider behavior may vary due to model differences.

## EN - STRICT_FAST Mode (Faster, Still Rigorous)

If you need lower latency while preserving validation/escalation logic:

```env
STRICT_FAST_PROFILE=true
```

This profile automatically applies speed-oriented runtime caps:
- lower retrieval/context budgets
- disables adaptive retrieval retry
- disables hybrid lexical merge
- keeps fast deterministic context selection on

What stays strict:
- `VALIDATE` node still controls final routing
- confidence and hallucination thresholds are still enforced
- escalation still happens when evidence is insufficient

## IT - Limiti Noti

- La copertura benchmark e' ancora limitata rispetto alla variabilita' reale in produzione.
- Costo e latenza sono piu' alti rispetto a RAG semplice.
- Alcuni miglioramenti di retrieval sono sensibili al dataset e vanno validati per dominio.
- Il comportamento multi-provider puo' variare per differenze tra modelli.

## IT - Modalita STRICT_FAST (Piu' Veloce, Ancora Rigorosa)

Se vuoi ridurre la latenza mantenendo validazione/escalation:

```env
STRICT_FAST_PROFILE=true
```

Questo profilo applica automaticamente override orientati alla velocita':
- budget retrieval/context piu' bassi
- disattiva adaptive retrieval retry
- disattiva merge lexical hybrid
- mantiene attiva la context selection deterministica veloce

Cosa resta rigoroso:
- il nodo `VALIDATE` continua a controllare il routing finale
- le soglie di confidence/hallucination restano attive
- l'escalation resta attiva quando l'evidenza non basta

## EN - 30-Day Roadmap

1. Publish updated triplet benchmark snapshot after hybrid retrieval changes.
2. Add retrieval diagnostics to public outputs with clearer score explanations.
3. Strengthen release automation (benchmark artifact checks + release checklist).
4. Expand docs/examples for contributor onboarding and reproducibility.

## IT - Roadmap 30 Giorni

1. Pubblicare snapshot benchmark triplet aggiornato dopo i cambi hybrid retrieval.
2. Aggiungere diagnostica retrieval negli output pubblici con spiegazioni score piu' chiare.
3. Rafforzare automazione release (controlli artifact benchmark + checklist release).
4. Espandere docs/esempi per onboarding contributor e riproducibilita'.

## EN - Pre-Push Public Checklist

1. Rotate any exposed API keys and verify `.env` is never committed.
2. Confirm `data/raw` contains no tracked private documents.
3. Run tests and benchmark smoke checks.
4. Update benchmark snapshot in docs with date and command.
5. Publish changelog/release notes with known limits.

## IT - Checklist Pre-Push Pubblico

1. Ruota eventuali API key esposte e verifica che `.env` non venga mai committato.
2. Conferma che in `data/raw` non ci siano documenti privati tracciati.
3. Esegui test e benchmark smoke.
4. Aggiorna snapshot benchmark nelle docs con data e comando.
5. Pubblica changelog/release notes con limiti noti.

## EN - RAG vs CAG (Short)

Standard RAG:

`question -> retrieve -> generate -> answer`

CAG:

`question -> classify -> retrieve -> select context -> reason -> validate -> answer/retry/escalate`

## IT - RAG vs CAG (Breve)

RAG standard:

`domanda -> retrieve -> generate -> risposta`

CAG:

`domanda -> classify -> retrieve -> select context -> reason -> validate -> risposta/retry/escalation`

## EN - Local Validation Flow

```bash
pip install -e ".[dev,eval,api]"
pytest
cag eval --system cag --limit 3 --judge-mode off
python -m uvicorn cag.api.upload:app --reload --port 8000
```

Optional frontend:

```bash
cd frontend
npm install
npm run dev
```

## IT - Flusso Di Validazione Locale

```bash
pip install -e ".[dev,eval,api]"
pytest
cag eval --system cag --limit 3 --judge-mode off
python -m uvicorn cag.api.upload:app --reload --port 8000
```

Frontend opzionale:

```bash
cd frontend
npm install
npm run dev
```

## EN - Repository Structure

```text
src/cag/
  agents/        # Retrieval + reasoning agents
  api/           # FastAPI upload/query endpoints
  eval/          # Benchmarks, scoring, comparisons
  graph/         # Nodes, state, routing
  ingestion/     # Loader, chunker, embeddings, vector store
  retrieval/     # Lexical scoring and context dedupe
  knowledge/     # DB-first compiled knowledge layer
frontend/        # React preview UI
docs/            # Project docs
tests/           # Unit and integration tests
```

## IT - Struttura Del Repository

```text
src/cag/
  agents/        # Agenti retrieval + reasoning
  api/           # Endpoint FastAPI upload/query
  eval/          # Benchmark, scoring, confronti
  graph/         # Nodi, stato, routing
  ingestion/     # Loader, chunker, embedding, vector store
  retrieval/     # Scoring lessicale e dedupe contesto
  knowledge/     # Layer DB-first di knowledge compilata
frontend/        # UI React di anteprima
docs/            # Documentazione progetto
tests/           # Test unitari e integrazione
```

## EN - Community

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- [SECURITY.md](SECURITY.md)
- [NOTICE](NOTICE)

## IT - Community

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- [SECURITY.md](SECURITY.md)
- [NOTICE](NOTICE)

## EN - Attribution Request

CAG is released under MIT.  
If you use CAG publicly, attribution is appreciated:

`Built with CAG by Fabio Scialanga`

## IT - Richiesta Di Attribuzione

CAG e' rilasciato con licenza MIT.  
Se usi CAG in pubblico, l'attribuzione e' apprezzata:

`Built with CAG by Fabio Scialanga`

## License

[MIT](LICENSE)

