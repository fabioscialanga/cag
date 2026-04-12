# External AutoResearch Trial

Questo scaffold serve a provare un layer esterno in stile `autoresearch`
senza farlo entrare nel core di CAG.

Obiettivo:

- trattare CAG come target repository
- lasciare invariato il runtime CAG
- usare un agente esterno per proporre piccole modifiche
- validare ogni modifica con test ed eval prima di promuoverla

Questa cartella non viene importata dal codice applicativo. E' solo un
supporto sperimentale e si puo' eliminare senza effetti collaterali.

Contenuto:

- `program.md`: policy operativa per l'agente esterno
- `target_spec.md`: mappa del progetto CAG vista dall'optimizer
- `experiment_backlog.md`: lista di esperimenti ad alta leva
- `wow_definition.md`: definizione concreta di cosa significa "Wow"

## Uso consigliato

1. Crea una repo esterna dedicata alla ricerca automatica.
2. Copia o riusa il file `program.md` di questa cartella come policy iniziale.
3. Punta l'agente esterno a questa repo CAG come target di lavoro.
4. Permetti solo questo ciclo:
   - leggere il codice
   - proporre una modifica piccola
   - eseguire test mirati
   - eseguire smoke eval
   - confrontare i risultati con la baseline
   - tenere la modifica solo se migliora o non degrada il sistema

## Metriche da ottimizzare

Metriche primarie:

- `context_precision_score`
- `grounded_answer_score`
- `task_success_rate`
- `hallucination_rate`

Vincoli:

- nessun test rotto
- nessun peggioramento netto della robustezza API
- nessuna modifica distruttiva al benchmark harness

## Comandi utili per il loop esterno

Test:

```powershell
pytest tests/test_cag_pipeline.py tests/test_agents_integration.py tests/test_api.py
```

Smoke eval:

```powershell
cag eval --system cag --limit 3 --judge-mode off
```

Triplet compare:

```powershell
.\run_eval_triplet.ps1 -JudgeMode off
```

## Cosa NON deve fare l'agente esterno

- non deve riscrivere l'architettura del grafo senza evidenza benchmark
- non deve introdurre nuove dipendenze pesanti senza un chiaro beneficio
- non deve toccare file non rilevanti per la modifica proposta
- non deve promuovere cambi che migliorano una metrica ma peggiorano nettamente le altre

## Rollback

Se l'esperimento non ti piace, il rollback e' semplicissimo:

- elimina la cartella `experiments/autoresearch_external/`

Non ci sono hook runtime, import applicativi o dipendenze da rimuovere.
