"""
humaneval.py — Caricamento del benchmark HumanEval da Hugging Face.

Ogni problema è un dict con le colonne descritte nel README:
  task_id, prompt, canonical_solution, test, entry_point

A queste aggiungiamo una colonna derivata "codice_completo" = prompt +
canonical_solution: in HumanEval la `prompt` contiene import + firma + docstring
e la `canonical_solution` contiene SOLO il corpo (indentato). Concatenandole si
ottiene la funzione di riferimento COMPLETA ed eseguibile, che usiamo come
riferimento per il CodeBLEU. Dalla funzione così ottenuta RIMUOVIAMO la
docstring (strip_docstrings): il modello genera la funzione completa ma di
norma senza docstring, e tenerla solo nel riferimento penalizzerebbe il
CodeBLEU a parità di logica.

I dataset, una volta scaricati, vengono salvati nella cartella locale del
progetto "Benchmark/" (con save_to_disk) e da lì riletti ai run successivi.

Nota Windows: non usiamo cache_dir=Benchmark perché HuggingFace crea un file
di lock il cui nome incorpora l'intero path assoluto; su percorsi profondi
supera il limite di 260 caratteri di Windows. save_to_disk usa nomi corti.
"""

from pathlib import Path

from datasets import load_dataset, load_from_disk

from .code_extractor import strip_docstrings

# Cartella dove finiscono i dataset salvati: <progetto>/Benchmark/
BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "Benchmark"
HUMANEVAL_DIR = BENCHMARK_DIR / "humaneval"


def load_humaneval(limit: int | None = None) -> list[dict]:
    """
    Carica HumanEval (164 problemi).

    Primo run: scarica da Hugging Face e salva in Benchmark/humaneval/.
    Run successivi: rilegge direttamente da quella cartella (nessun download).

    limit: se valorizzato, prende solo i primi N problemi
           (utile per test rapidi senza spendere molte API call).
    """
    BENCHMARK_DIR.mkdir(exist_ok=True)

    if HUMANEVAL_DIR.exists():
        ds = load_from_disk(str(HUMANEVAL_DIR))
    else:
        ds = load_dataset("openai/openai_humaneval", split="test")
        ds.save_to_disk(str(HUMANEVAL_DIR))

    problems = [dict(row) for row in ds]
    for p in problems:
        # Funzione di riferimento completa: import + firma + corpo. Concateniamo
        # prompt (import + firma + docstring) e canonical_solution (corpo), poi
        # RIMUOVIAMO la docstring: il codice generato dal modello di norma non la
        # contiene, e tenerla solo nel riferimento abbasserebbe ingiustamente il
        # CodeBLEU. È questo il riferimento per il CodeBLEU (vedi metrics.py).
        p["codice_completo"] = strip_docstrings(
            p.get("prompt", "") + p.get("canonical_solution", "")
        )
    if limit is not None:
        problems = problems[:limit]
    return problems
