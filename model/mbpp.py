"""
mbpp.py — Caricamento del benchmark MBPP (Mostly Basic Python Problems).

Usiamo la configurazione "full" e teniamo SOLO lo split di TEST ufficiale:
i 500 problemi con task_id 11–510 (Austin et al., 2021). Il perché della scelta
di "full" (e non "sanitized") è spiegato nel README.

Struttura di un problema MBPP (config "full"):
  task_id, text, code, test_list, test_setup_code, challenge_test_list

Nota: a differenza di HumanEval, in MBPP la colonna `code` è GIÀ la funzione
completa (def + corpo), non solo il corpo. Quindi non serve ricostruire un
"codice_completo": `code` è già il riferimento completo per il CodeBLEU.

Come per HumanEval, il dataset viene salvato in locale con save_to_disk in
Benchmark/mbpp/ e riletto da lì ai run successivi (nomi corti per il limite
di 260 caratteri di Windows; vedi humaneval.py per il dettaglio).
"""

from pathlib import Path

from datasets import load_dataset, load_from_disk, concatenate_datasets

# Cartella dove finiscono i dataset salvati: <progetto>/Benchmark/
BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "Benchmark"
MBPP_DIR = BENCHMARK_DIR / "mbpp"

# Intervallo ufficiale dello split di test di MBPP (i 500 problemi valutati nei paper).
# Layout completo per task_id: prompt 1–10, TEST 11–510, validation 511–600, train 601–974.
TEST_TASK_ID_MIN = 11
TEST_TASK_ID_MAX = 510


def load_mbpp(limit: int | None = None) -> list[dict]:
    """
    Carica i 500 problemi di test di MBPP (task_id 11–510).

    Primo run: scarica la config "full" da Hugging Face, seleziona il test
    ufficiale per task_id e salva in Benchmark/mbpp/.
    Run successivi: rilegge direttamente da quella cartella (nessun download).

    limit: se valorizzato, prende solo i primi N problemi (test rapidi/economici).
    """
    BENCHMARK_DIR.mkdir(exist_ok=True)

    if MBPP_DIR.exists():
        ds = load_from_disk(str(MBPP_DIR))
    else:
        raw = load_dataset("google-research-datasets/mbpp", "full")
        # Non ci fidiamo dei NOMI degli split (cambiano tra le versioni del
        # dataset): uniamo tutte le righe e selezioniamo il test ufficiale
        # tramite il task_id, che è l'identità STABILE del problema.
        all_rows = concatenate_datasets(list(raw.values()))
        ds = all_rows.filter(
            lambda r: TEST_TASK_ID_MIN <= r["task_id"] <= TEST_TASK_ID_MAX
        ).sort("task_id")
        ds.save_to_disk(str(MBPP_DIR))

    problems = [dict(row) for row in ds]
    if limit is not None:
        problems = problems[:limit]
    return problems
