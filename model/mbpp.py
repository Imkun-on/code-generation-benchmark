"""
mbpp.py — Loading the MBPP benchmark (Mostly Basic Python Problems).

We use the "full" configuration and keep ONLY the official TEST split: the 500
problems with task_id 11–510 (Austin et al., 2021). The rationale for choosing
"full" (and not "sanitized") is explained in the README.

Structure of an MBPP problem (config "full"):
  task_id, text, code, test_list, test_setup_code, challenge_test_list

Note: unlike HumanEval, in MBPP the `code` column is ALREADY the complete
function (def + body), not just the body. So there is no need to rebuild a
"codice_completo": `code` is already the complete reference for CodeBLEU.

As with HumanEval, the dataset is saved locally with save_to_disk in
Benchmark/mbpp/ and read back from there on subsequent runs (short names for
Windows' 260-character limit; see humaneval.py for details).
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
    Load the 500 MBPP test problems (task_id 11–510).

    First run: downloads the "full" config from Hugging Face, selects the
    official test set by task_id and saves it to Benchmark/mbpp/.
    Subsequent runs: reads directly from that folder (no download).

    limit: if set, takes only the first N problems (quick/cheap tests).
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
