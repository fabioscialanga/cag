# External Optimizer Backlog

Questa lista ordina gli esperimenti da provare per primi.

## Tier 1: Alta leva, basso rischio

### 1. API key nel frontend preview

Problema:

- la UI locale puo' rompersi quando `CAG_API_KEY` e' attiva

Esperimento:

- permettere al frontend di leggere una chiave preview da config locale
- inviare `X-API-Key` su `/query` e `/upload`

Successo:

- nessun `401/403` nella UI quando la chiave e' configurata

### 2. Avvio preview piu' robusto

Problema:

- collisione frequente sulla porta `8000`
- confusione tra backend giusto e backend sbagliato

Esperimento:

- introdurre uno script di avvio preview dedicato
- usare una porta fallback chiara tipo `8010`

Successo:

- riduzione del tempo necessario per arrivare a "proof of life"

### 3. Migliorare il detector lingua

Problema:

- il rilevamento attuale e' volutamente leggero

Esperimento:

- migliorare la detection per query brevi o miste
- mantenere il comportamento semplice e trasparente

Successo:

- meno mismatch lingua query/risposta

## Tier 2: Alta leva, rischio medio

### 4. Retry adattivo vero

Problema:

- oggi il retry rifa' il reasoner, ma non cambia davvero strategia

Esperimento:

- quando il validate fallisce, modificare retrieval budget o contesto
- usare i `gaps` per guidare la seconda iterazione

Successo:

- migliore `task_success_rate`
- nessun aumento netto di `hallucination_rate`

### 5. Diversity-aware context selection

Problema:

- il selector e' gia' migliore di prima, ma puo' ancora collassare su chunk molto simili

Esperimento:

- aumentare la penalita' per near-duplicate nei top chunk
- garantire piu' copertura cluster/categoria

Successo:

- aumento di `context_precision_score`
- eventuale miglioramento di `grounded_answer_score`

### 6. Baseline hardening

Problema:

- alcune baseline hanno failure mode rumorose o parsing fragile

Esperimento:

- ridurre fragilita' di parsing
- migliorare affidabilita' del confronto

Successo:

- benchmark piu' pulito
- claim piu' difendibile

## Tier 3: Da fare dopo

### 7. Secondo corpus

Obiettivo:

- testare generalizzazione del vantaggio di CAG

### 8. Chunk-level labels

Obiettivo:

- rendere `context_precision_score` piu' fine e meno proxy-based

### 9. Report automatico piu' forte

Obiettivo:

- generare narrative benchmark pronte da leggere e confrontare
