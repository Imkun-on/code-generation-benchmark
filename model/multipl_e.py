"""
multipl_e.py — Caricamento del benchmark MultiPL-E (nuprl/MultiPL-E).

MultiPL-E (Cassano et al., 2023) traduce HumanEval e MBPP in ~24 linguaggi di
programmazione. Qui usiamo SOLO il set **HumanEval** (config `humaneval-<lang>`),
con TUTTI i linguaggi disponibili, riuniti in UN UNICO benchmark (la colonna
`language` distingue il linguaggio) → un solo file di risultato come per gli altri
benchmark.

CARATTERISTICHE DI PROGETTO
---------------------------
  - È un benchmark di **SOLA esecuzione**: ogni esempio porta il `prompt` (firma +
    documentazione nel linguaggio target, lasciata APERTA) e i `tests` nel
    linguaggio target, ma **NON una soluzione gold**. Conseguenze:
      * la metrica è il **pass@1** (codice generato + test eseguiti senza errori);
      * **CodeBLEU NON è calcolabile** (manca un riferimento nel linguaggio
        target) → nei record `metrics.codebleu` resta None.
  - Il **pass@1 richiede il runtime** del linguaggio installato. Oggi eseguono
    Python(n/a, non c'è in MultiPL-E), JS (Node), PHP, R, Java (JDK); C++ se c'è
    g++; gli altri (go/rust/c#/swift/scala/haskell/…) danno `RuntimeMissing`
    finché non si installa il toolchain (vedi executor.py).

MODALITÀ COMPLETAMENTO
----------------------
Il `prompt` finisce con la firma APERTA della funzione (es. JS
`function f(args){`); i `tests` sono costruiti per essere APPESI dopo il corpo
(in Java iniziano addirittura con `}` che chiude il metodo). Quindi il programma
da eseguire è `prompt + corpo_generato + tests` (assemblato in executor.py).

DATI SU DISCO
-------------
Come gli altri benchmark, al primo run ogni config è salvata in
Benchmark/multipl_e/<config>/ (save_to_disk, nomi corti per il limite MAX_PATH di
Windows); i run successivi rileggono da disco. Il download HF su Windows richiede
HF_HUB_DISABLE_SYMLINKS=1 (impostato qui, vedi memoria env-gotchas).
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
    m = _NAME_RE.match(name or "")
    return m.group(1) if m else None


def _load_config(lang: str, limit: int | None):
    """Carica (e mette in cache su disco) una singola config `humaneval-<lang>`."""
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
    Carica MultiPL-E (set HumanEval) come UN SOLO elenco di problemi (tutti i
    linguaggi insieme).

    Primo run: scarica `nuprl/MultiPL-E` (config `humaneval-<lang>`) da Hugging
    Face e la salva in Benchmark/multipl_e/<config>. Run successivi: rilegge da
    disco.

    limit: se valorizzato, prende i primi N esempi **per ciascun linguaggio** (così
           un giro di prova economico esercita comunque tutti i linguaggi).
    languages: sottoinsieme dei linguaggi da caricare (default: tutti).

    Ogni record:
      task_id     = "<lang>/<name>" (unico tra linguaggi → no collisioni nel
                    checkpoint, es. "js/HumanEval_0_has_close_elements")
      name        identificatore del problema (uguale tra i linguaggi)
      language    linguaggio target (js/php/r/java/cpp/…)
      prompt      firma + doc nel linguaggio target, lasciata APERTA (da completare)
      tests       harness di test nel linguaggio target (da appendere dopo il corpo)
      stop_tokens token che segnalano la fine della generazione (informativo)
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
    """Pianifica un run MultiPL-E generando SOLO per i linguaggi ESEGUIBILI.

    Come DS-1000 con le librerie installate: rileva quali linguaggi hanno il runtime
    (via executor.multipl_e_runnable) e carica/genererà SOLO quelli, **saltando**
    gli altri — così non si spende API per problemi che darebbero comunque
    `RuntimeMissing`. Nessun taglio silenzioso: i saltati sono riportati in `info`.

    languages: sottoinsieme richiesto (default: tutti i 24). Tra questi, vengono
               eseguiti solo quelli eseguibili.
    Ritorna (problems, info) con info = {available, missing, requested}.
    """
    from .executor import multipl_e_runnable

    requested = [l for l in (languages or LANGUAGES) if l in LANGUAGES]
    available = [l for l in requested if multipl_e_runnable(l)]
    missing = [l for l in requested if l not in available]

    problems = load_multipl_e(limit=limit, languages=available) if available else []
    info = {"available": available, "missing": missing, "requested": requested}
    return problems, info
