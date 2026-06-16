"""
ds1000.py — Loading the DS-1000 benchmark (xlangai/DS-1000).

DS-1000 (Lai et al., 2022) gathers 1000 data science problems taken from
StackOverflow over 7 Python libraries (Pandas, NumPy, Matplotlib, Scikit-learn,
SciPy, PyTorch, TensorFlow). Each problem asks to complete a code snippet;
evaluation is by execution.

Structure of a problem (test.jsonl):
  prompt          description + code skeleton with the point to complete
  reference_code  gold solution (fragment) — reference for CodeBLEU
  code_context    COMPLETE test harness: defines test_execution(solution)
                  (and possibly test_string), which raises AssertionError if
                  the solution is wrong. Contains the [insert] placeholder.
  metadata        dict with problem_id, library, perturbation_type, test_case_cnt…

Unlike HumanEval/MBPP, the task_id is not a top-level column: we derive it from
metadata.problem_id (integer 0–999) and add it for compatibility with the
pipeline (checkpoint, dedup, report).

CodeBLEU note: the reference is `reference_code` (the gold solution), NEVER the
`code_context` (which is the grader, not a code reference).

As with the other benchmarks, the dataset is saved locally in Benchmark/ds1000/
(with save_to_disk) on the first run and read back from there afterward.

⚠️ Execution: unlike HumanEval/MBPP (stdlib only), DS-1000 requires the
data-science libraries installed (numpy, pandas, scipy, scikit-learn, matplotlib
and — for the respective subsets — tensorflow, pytorch). Problems of the
uninstalled libraries will fail with ModuleNotFoundError. You can filter by
library with the `libraries` parameter.
"""

import importlib.util
from pathlib import Path

from datasets import Dataset, load_from_disk

# Cartella dove finiscono i dataset salvati: <progetto>/Benchmark/
BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "Benchmark"
DS1000_DIR = BENCHMARK_DIR / "ds1000"

# Nomi delle 7 librerie così come compaiono ESATTAMENTE in metadata["library"],
# con il numero di problemi (totale 1000). Attenzione: è "Sklearn", non
# "Scikit-learn". Il confronto in load_ds1000 è comunque case-insensitive.
#   Pandas 291, Numpy 220, Matplotlib 155, Sklearn 115, Scipy 106, Pytorch 68, Tensorflow 45
DS1000_LIBRARIES = (
    "Pandas", "Numpy", "Matplotlib", "Sklearn", "Scipy", "Pytorch", "Tensorflow",
)

# Nome della libreria DS-1000 -> nome del modulo Python da importare. Quasi tutti
# coincidono a meno del casing, tranne "Sklearn" (modulo `sklearn`) e "Pytorch"
# (modulo `torch`). Serve a rilevare quali librerie sono installate (vedi
# available_libraries) e quindi quali sottoinsiemi sono eseguibili.
DS1000_IMPORT_NAMES = {
    "Pandas": "pandas",
    "Numpy": "numpy",
    "Matplotlib": "matplotlib",
    "Sklearn": "sklearn",
    "Scipy": "scipy",
    "Pytorch": "torch",
    "Tensorflow": "tensorflow",
}

# Lookup case-insensitive nome-libreria -> casing canonico (quello di metadata).
_CANONICAL = {name.lower(): name for name in DS1000_LIBRARIES}


def _canonical(library: str) -> str:
    """Normalize a library name to its canonical casing (case-insensitive).
    Raises ValueError with the list of valid names if the name is unknown."""
    key = library.strip().lower()
    if key not in _CANONICAL:
        raise ValueError(
            f"Libreria DS-1000 sconosciuta: {library!r}. "
            f"Valide: {', '.join(DS1000_LIBRARIES)}"
        )
    return _CANONICAL[key]


def _is_importable(library: str) -> bool:
    """True if the library's module is importable, WITHOUT actually importing it
    (uses find_spec: no side effects and no cost of loading TF/Torch)."""
    module = DS1000_IMPORT_NAMES[library]
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        # find_spec può sollevare se un pacchetto genitore è rotto: lo trattiamo
        # come "non disponibile" anziché far crashare la pipeline.
        return False


def available_libraries(requested: list[str] | None = None
                        ) -> tuple[list[str], list[str]]:
    """Split the requested libraries into (available, missing) based on which
    modules are installed in the environment.

    requested: names chosen by the user (--libraries), case-insensitive; None =
               all 7. Unknown names raise ValueError.

    Returns (available, missing) with canonical-cased names, preserving order
    and deduplicating. E.g. on Python 3.14 without TF:
        available_libraries() -> ([Pandas..Pytorch], ["Tensorflow"]).
    """
    names = [_canonical(l) for l in requested] if requested else list(DS1000_LIBRARIES)
    seen: set[str] = set()
    ordered = [n for n in names if not (n in seen or seen.add(n))]
    available = [n for n in ordered if _is_importable(n)]
    missing = [n for n in ordered if n not in available]
    return available, missing


def library_counts() -> dict[str, int]:
    """Number of problems per library over the whole set (1000). Useful for the
    report to state how many problems are skipped when a library is missing."""
    counts: dict[str, int] = {}
    for p in load_ds1000():
        lib = (p.get("metadata") or {}).get("library", "")
        counts[lib] = counts.get(lib, 0) + 1
    return counts


def plan_run(limit: int | None = None,
             requested: list[str] | None = None) -> tuple[list[dict], dict]:
    """Resolve which libraries to run and load ONLY the runnable problems.

    A single read of the dataset: detects the installed libraries among the
    requested ones, loads the 1000 problems, keeps the available subsets and
    computes the summary for the report (no silent truncation).

    requested: chosen libraries (--libraries) or None = all.
    limit: cap on the problems AFTER the per-library filter (quick tests).

    Returns (problems, info) with info = {
        "available": [...],   libraries being run
        "missing":   [...],   requested but uninstalled libraries (skipped)
        "counts":    {lib: n} problems per library over the whole set
        "total":     1000,    total problems of the full set
        "selected":  N,       problems actually run (= len(problems))
    }."""
    available, missing = available_libraries(requested)
    all_problems = load_ds1000()
    counts: dict[str, int] = {}
    for p in all_problems:
        lib = (p.get("metadata") or {}).get("library", "")
        counts[lib] = counts.get(lib, 0) + 1

    wanted = set(available)
    problems = [p for p in all_problems
                if (p.get("metadata") or {}).get("library", "") in wanted]
    problems.sort(key=lambda p: p["task_id"])
    if limit is not None:
        problems = problems[:limit]

    info = {
        "available": available,
        "missing": missing,
        "counts": counts,
        "total": len(all_problems),
        "selected": len(problems),
    }
    return problems, info


def load_ds1000(limit: int | None = None,
                libraries: list[str] | None = None) -> list[dict]:
    """
    Load the 1000 DS-1000 problems.

    First run: downloads test.jsonl from Hugging Face and saves to Benchmark/ds1000/.
    Subsequent runs: reads directly from that folder (no download).

    limit: if set, takes only the first N problems (quick/cheap tests).
    libraries: if set, keeps only the problems of the listed libraries
               (e.g. ["Pandas", "Numpy"]) — useful to skip the subsets whose
               libraries are not installed (e.g. Tensorflow/Pytorch).
    """
    BENCHMARK_DIR.mkdir(exist_ok=True)

    if DS1000_DIR.exists():
        ds = load_from_disk(str(DS1000_DIR))
    else:
        # Usiamo pandas + hf:// (percorso ufficiale del dataset) e convertiamo in
        # datasets.Dataset, così riusiamo save_to_disk/load_from_disk come per gli
        # altri benchmark (nomi corti: limite 260 caratteri di Windows).
        import pandas as pd
        df = pd.read_json("hf://datasets/xlangai/DS-1000/test.jsonl", lines=True)
        ds = Dataset.from_pandas(df)
        ds.save_to_disk(str(DS1000_DIR))

    problems = [dict(row) for row in ds]
    for p in problems:
        # task_id stabile per la pipeline (checkpoint/dedup/report).
        p["task_id"] = p["metadata"]["problem_id"]

    if libraries is not None:
        wanted = {lib.lower() for lib in libraries}
        problems = [p for p in problems
                    if p["metadata"]["library"].lower() in wanted]

    problems.sort(key=lambda p: p["task_id"])
    if limit is not None:
        problems = problems[:limit]
    return problems
