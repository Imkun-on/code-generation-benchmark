"""
plot2code.py — Caricamento del benchmark Plot2Code (TencentARC/Plot2Code).

Plot2Code (Wu et al., 2024) valuta la generazione di codice che riproduce un
grafico scientifico. Ogni esempio è un plot della gallery di matplotlib con:
  url          link alla pagina della gallery di origine
  code         codice matplotlib di riferimento (gold)
  instruction  descrizione testuale dettagliata della figura (generata da GPT-4)
  image        il plot renderizzato di riferimento (PIL Image)

USO NEL NOSTRO PROGETTO (scelta di progetto)
--------------------------------------------
Input al modello = `instruction` (SOLO testo, NON multimodale): così anche i
modelli text-only del confronto per architettura (DeepSeek MoE, Ministral SLM)
possono partecipare. Il modello genera lo script matplotlib; lo valutiamo con:
  1. CodeBLEU      — codice generato vs `code` (metrics.py)
  2. correttezza funzionale — il codice GIRA e produce una figura (executor.py)
  3. similarità immagine — render del codice generato vs `image` di riferimento
                           (metrica da definire; per ora salviamo solo le immagini)

A differenza degli altri benchmark NON ci sono unit test: la "correttezza" è il
fatto che il codice esegua e produca un PNG non vuoto (il "code pass rate" del paper).

DATI SU DISCO
-------------
Come gli altri benchmark, al primo run il dataset è salvato in Benchmark/plot2code/
(save_to_disk, nomi corti per il limite di 260 char di Windows). Le immagini di
riferimento vengono estratte una volta in Benchmark/plot2code/ref_images/<id>.png
e nei record passiamo il PATH (le PIL Image non sono serializzabili nel checkpoint
JSON). Su Windows il download HF richiede HF_HUB_DISABLE_SYMLINKS=1 (vedi memoria
env-gotchas), che impostiamo qui.
"""

import os
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "Benchmark"
PLOT2CODE_DIR = BENCHMARK_DIR / "plot2code"
DATASET_DIR = PLOT2CODE_DIR / "dataset"           # arrow (save_to_disk)
REF_IMAGES_DIR = PLOT2CODE_DIR / "ref_images"     # <task_id>.png di riferimento


def _ensure_ref_image(image, task_id: int) -> str:
    """Salva (una volta) l'immagine di riferimento PIL su disco e ne ritorna il
    path assoluto. Le immagini servono per il confronto visivo e NON possono
    stare nel record (non serializzabili in JSON)."""
    REF_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    path = REF_IMAGES_DIR / f"{task_id}.png"
    if not path.exists():
        image.save(path)
    return str(path)


def load_plot2code(limit: int | None = None) -> list[dict]:
    """
    Carica i problemi di Plot2Code.

    Primo run: scarica TencentARC/Plot2Code da Hugging Face, lo salva in
    Benchmark/plot2code/dataset ed estrae le immagini di riferimento.
    Run successivi: rilegge da disco (nessun download).

    limit: se valorizzato, prende solo i primi N esempi (giri rapidi/economici).

    Ogni record:
      task_id      indice intero stabile (checkpoint/dedup/report)
      instruction  descrizione della figura (input al modello)
      code         codice matplotlib di riferimento (riferimento CodeBLEU)
      url          pagina di origine
      ref_image    path al PNG di riferimento (per il confronto visivo)
    """
    BENCHMARK_DIR.mkdir(exist_ok=True)
    # Windows: evita il fallimento symlink della cache HF (WinError 1314).
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    from datasets import load_from_disk

    if DATASET_DIR.exists():
        ds = load_from_disk(str(DATASET_DIR))
    else:
        from datasets import load_dataset
        loaded = load_dataset("TencentARC/Plot2Code")
        ds = loaded[list(loaded.keys())[0]]        # unico split: 'test'
        DATASET_DIR.parent.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(DATASET_DIR))

    n = len(ds) if limit is None else min(limit, len(ds))
    records: list[dict] = []
    for i in range(n):
        row = ds[i]
        records.append({
            "task_id": i,
            "instruction": row.get("instruction", "") or "",
            "code": row.get("code", "") or "",
            "url": row.get("url", "") or "",
            "ref_image": _ensure_ref_image(row["image"], i),
        })
    return records
