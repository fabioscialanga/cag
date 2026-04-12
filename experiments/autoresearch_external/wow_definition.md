# What "Wow" Means For CAG

"Wow" non significa solo che la demo sembra piu' intelligente.

Per CAG, "Wow" vuol dire che una persona tecnica prova il sistema e pensa:

- capisce la domanda
- sceglie bene il contesto
- non inventa
- sa fermarsi quando non sa
- e' misurabilmente migliore, non solo piu' fluido

## Wow tecnico

Il sistema deve mostrare:

- risposte piu' grounded
- meno rumore nel contesto
- escalation piu' chiare
- metriche migliori in benchmark

## Wow di prodotto

Il sistema deve risultare:

- facile da avviare
- prevedibile in locale
- chiaro nelle failure modes
- credibile quando dice "non basta evidenza"

## Segnali concreti di Wow

Segnali forti:

- `context_precision_score` stabilmente alto
- `task_success_rate` in crescita
- `hallucination_rate` non in peggioramento
- UI preview che funziona senza attrito inutile
- README che porta a un "proof of life" rapido

## Anti-Wow

Non e' Wow se:

- la risposta suona meglio ma e' meno grounded
- il sistema risponde di piu' ma sbaglia di piu'
- la demo funziona solo in condizioni perfette
- il benchmark diventa meno affidabile

## Regola pratica

Una modifica e' "Wow-compatible" se migliora almeno una di queste dimensioni
senza degradare in modo chiaro le altre:

- grounding
- context selection
- reliability
- benchmark clarity
- local developer experience
