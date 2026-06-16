"""
humaneval.py — Loading the HumanEval benchmark from Hugging Face.

Each problem is a dict with the columns described in the README:
  task_id, prompt, canonical_solution, test, entry_point

To these we add a derived column "codice_completo" = prompt +
canonical_solution: in HumanEval the `prompt` contains imports + signature +
docstring and the `canonical_solution` contains ONLY the body (indented).
Concatenating them yields the COMPLETE, executable reference function, which we
use as the reference for CodeBLEU. From the resulting function we REMOVE the
docstring (strip_docstrings): the model generates the complete function but
usually without a docstring, and keeping it only in the reference would
penalize CodeBLEU for equivalent logic.

Once downloaded, the datasets are saved to the project's local "Benchmark/"
folder (with save_to_disk) and read back from there on subsequent runs.

Windows note: we do not use cache_dir=Benchmark because Hugging Face creates a
lock file whose name embeds the entire absolute path; on deep paths it exceeds
Windows' 260-character limit. save_to_disk uses short names.
"""

from pathlib import Path

from datasets import load_dataset, load_from_disk

from .code_extractor import strip_docstrings

# Cartella dove finiscono i dataset salvati: <progetto>/Benchmark/
BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "Benchmark"
HUMANEVAL_DIR = BENCHMARK_DIR / "humaneval"


def load_humaneval(limit: int | None = None) -> list[dict]:
    """
    Load HumanEval (164 problems).

    First run: downloads from Hugging Face and saves to Benchmark/humaneval/.
    Subsequent runs: reads directly from that folder (no download).

    limit: if set, takes only the first N problems
           (useful for quick tests without spending many API calls).
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
