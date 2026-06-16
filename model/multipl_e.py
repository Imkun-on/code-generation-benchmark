"""
multipl_e.py — Loading the MultiPL-E benchmark (nuprl/MultiPL-E).

MultiPL-E (Cassano et al., 2023) translates HumanEval and MBPP into ~24
programming languages. Here we use ONLY the **HumanEval** set (config
`humaneval-<lang>`), with ALL the available languages, gathered into ONE SINGLE
benchmark (the `language` column distinguishes the language) → a single result
file as for the other benchmarks.

DESIGN CHARACTERISTICS
---------------------------
  - It is an **execution-only** benchmark: each example carries the `prompt`
    (signature + documentation in the target language, left OPEN) and the `tests`
    in the target language, but **NOT a gold solution**. Consequences:
      * the metric is **pass@1** (generated code + tests run without errors);
      * **CodeBLEU is NOT computable** (there is no reference in the target
        language) → in the records `metrics.codebleu` stays None.
  - **pass@1 requires the language runtime** installed. Today the runnable ones
    are JS (Node), PHP, R, Java (JDK); C++ if g++ is present; the others
    (go/rust/c#/swift/scala/haskell/…) yield `RuntimeMissing` until the toolchain
    is installed (see executor.py).

COMPLETION MODE
----------------------
The `prompt` ends with the OPEN function signature (e.g. JS `function f(args){`);
the `tests` are built to be APPENDED after the body (in Java they even start with
`}` that closes the method). So the program to execute is `prompt +
generated_body + tests` (assembled in executor.py).

DATA ON DISK
-------------
Like the other benchmarks, on the first run each config is saved to
Benchmark/multipl_e/<config>/ (save_to_disk, short names for Windows' MAX_PATH
limit); subsequent runs read from disk. The HF download on Windows requires
HF_HUB_DISABLE_SYMLINKS=1 (set here, see the env-gotchas memo).
"""

import os
import re
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "Benchmark"
MULTIPLE_DIR = BENCHMARK_DIR / "multipl_e"

# Tutti i linguaggi del set HumanEval di MultiPL-E (suffisso del config
# `humaneval-<lang>`), in ordine stabile (= ordine dei record nel file unico).
LANGUAGES = [
    "cpp", "cs", "d", "go", "java", "jl", "js", "lua", "php", "pl", "r", "rb",
    "rkt", "rs", "scala", "sh", "swift", "ts", "clj", "dart", "elixir", "hs",
    "ml", "adb",
]

# Nome leggibile del linguaggio (per il prompt) dal suffisso del config.
LANG_LABELS = {
    "cpp": "C++", "cs": "C#", "d": "D", "go": "Go", "java": "Java",
    "jl": "Julia", "js": "JavaScript", "lua": "Lua", "php": "PHP", "pl": "Perl",
    "r": "R", "rb": "Ruby", "rkt": "Racket", "rs": "Rust", "scala": "Scala",
    "sh": "Bash", "swift": "Swift", "ts": "TypeScript", "clj": "Clojure",
    "dart": "Dart", "elixir": "Elixir", "hs": "Haskell", "ml": "OCaml",
    "adb": "Ada",
}

# Estrae il nome della funzione dal campo `name` (es.
# "HumanEval_0_has_close_elements" -> "has_close_elements"), usato come stimolo
# direzionale ("usa esattamente questo nome"). Best-effort: se non matcha, None.
_NAME_RE = re.compile(r"^(?:HumanEval|MBPP)_\d+_(.+)$")


def function_name(name: str) -> str | None:
    """Extract the function name from the `name` field (e.g.
    "HumanEval_0_has_close_elements" -> "has_close_elements"), used as a
    directional stimulus ("use exactly this name"). Best-effort: returns None if
    it does not match."""
    m = _NAME_RE.match(name or "")
    return m.group(1) if m else None


def _load_config(lang: str, limit: int | None):
    """Load (and cache on disk) a single `humaneval-<lang>` config. Returns
    (dataset, n) where n is the number of examples to use after applying `limit`."""
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from datasets import load_from_disk

    config = f"humaneval-{lang}"
    cfg_dir = MULTIPLE_DIR / config
    if cfg_dir.exists():
        ds = load_from_disk(str(cfg_dir))
    else:
        from datasets import load_dataset
        ds = load_dataset("nuprl/MultiPL-E", config, split="test")
        cfg_dir.parent.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(cfg_dir))

    n = len(ds) if limit is None else min(limit, len(ds))
    return ds, n


def load_multipl_e(limit: int | None = None,
                   languages: list[str] | None = None) -> list[dict]:
    """
    Load MultiPL-E (HumanEval set) as ONE SINGLE list of problems (all the
    languages together).

    First run: downloads `nuprl/MultiPL-E` (config `humaneval-<lang>`) from
    Hugging Face and saves it to Benchmark/multipl_e/<config>. Subsequent runs:
    reads from disk.

    limit: if set, takes the first N examples **per language** (so a cheap trial
           run still exercises every language).
    languages: subset of languages to load (default: all).

    Each record:
      task_id     = "<lang>/<name>" (unique across languages → no checkpoint
                    collisions, e.g. "js/HumanEval_0_has_close_elements")
      name        problem identifier (same across languages)
      language    target language (js/php/r/java/cpp/…)
      prompt      signature + doc in the target language, left OPEN (to complete)
      tests       test harness in the target language (to append after the body)
      stop_tokens tokens that signal the end of generation (informative)
    """
    BENCHMARK_DIR.mkdir(exist_ok=True)
    langs = languages or LANGUAGES

    records: list[dict] = []
    for lang in langs:
        if lang not in LANGUAGES:
            continue
        ds, n = _load_config(lang, limit)
        for i in range(n):
            row = ds[i]
            name = row["name"]
            records.append({
                "task_id": f"{lang}/{name}",
                "name": name,
                "language": lang,
                "prompt": row.get("prompt", "") or "",
                "tests": row.get("tests", "") or "",
                "stop_tokens": list(row.get("stop_tokens", []) or []),
            })
    return records


def plan_run(limit: int | None = None,
             languages: list[str] | None = None) -> tuple[list[dict], dict]:
    """Plan a MultiPL-E run, generating ONLY for the RUNNABLE languages.

    Like DS-1000 with installed libraries: detects which languages have a runtime
    (via executor.multipl_e_runnable) and loads/generates ONLY those, **skipping**
    the others — so no API is spent on problems that would yield `RuntimeMissing`
    anyway. No silent truncation: the skipped ones are reported in `info`.

    languages: requested subset (default: all 24). Among these, only the runnable
               ones are executed.
    Returns (problems, info) with info = {available, missing, requested}.
    """
    from .executor import multipl_e_runnable

    requested = [l for l in (languages or LANGUAGES) if l in LANGUAGES]
    available = [l for l in requested if multipl_e_runnable(l)]
    missing = [l for l in requested if l not in available]

    problems = load_multipl_e(limit=limit, languages=available) if available else []
    info = {"available": available, "missing": missing, "requested": requested}
    return problems, info
