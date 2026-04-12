# CAG Target Spec For External Optimizer

Questo file descrive CAG come bersaglio di ottimizzazione per un agente esterno.

## Identita' del progetto

CAG non e' un "RAG con piu' step".
E' un sistema che:

- classifica la query
- recupera evidenza candidata
- seleziona il contesto migliore
- ragiona sull'evidenza
- valida la risposta
- puo' rifiutare o fare escalation

Il punto piu' importante da preservare e' questo:

`retrieval orchestration first, generation second`

## Claim attuale piu' forte

Sul benchmark corrente CAG dimostra soprattutto:

- migliore `context_precision_score`
- migliore `task_success_rate`
- minore rischio di allucinazione rispetto ad alcune baseline

Non ottimizzare il progetto come se il claim fosse gia':

- "CAG e' universalmente migliore su ogni metrica finale"

Ottimizzalo invece per rafforzare:

- qualita' della selezione del contesto
- grounding
- leggibilita' delle failure modes
- robustezza operativa

## Aree dove cambiare e' consentito

Alta priorita':

- `src/cag/agents/retrieval_agent.py`
- `src/cag/agents/reasoning_agent.py`
- `src/cag/graph/nodes.py`
- `src/cag/api/upload.py`
- `frontend/src/App.jsx`
- `src/cag/eval/*`

Media priorita':

- `src/cag/graph/graph.py`
- `src/cag/graph/runtime.py`
- `tests/*`
- `README.md`

Bassa priorita':

- ingest pipeline strutturale
- dipendenze
- grandi refactor architetturali

## Aree da evitare salvo evidenza forte

- riscrittura del grafo
- sostituzione del benchmark harness
- introduzione di nuove dipendenze pesanti
- cambi che spostano il progetto da "benchmarkable" a "prompt soup"

## Loop minimo di validazione

Prima di promuovere una modifica:

1. test mirati verdi
2. nessuna regressione evidente in API o frontend preview
3. smoke eval ok
4. se la modifica e' sostanziale, confronto con baseline

## Comandi canonici

Test:

```powershell
pytest tests/test_cag_pipeline.py tests/test_agents_integration.py tests/test_api.py
```

Smoke eval:

```powershell
cag eval --system cag --limit 3 --judge-mode off
```

Triplet benchmark:

```powershell
.\run_eval_triplet.ps1 -JudgeMode off
```

## Failure mode da evitare

Non ottimizzare solo per far "sembrare" le risposte piu' intelligenti.

Una modifica e' cattiva se:

- alza la verbosita' ma non migliora grounding
- riduce le escalation solo perche' il modello inventa di piu'
- migliora una smoke demo ma peggiora il benchmark
- rende il sistema piu' opaco o meno spiegabile
