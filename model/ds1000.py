"""
ds1000.py — Caricamento del benchmark DS-1000 (xlangai/DS-1000).

DS-1000 (Lai et al., 2022) raccoglie 1000 problemi di data science presi da
StackOverflow su 7 librerie Python (Pandas, NumPy, Matplotlib, Scikit-learn,
SciPy, PyTorch, TensorFlow). Ogni problema chiede di completare uno snippet di
codice; la valutazione è per esecuzione.

Struttura di un problema (test.jsonl):
  prompt          descrizione + scheletro di codice con il punto da completare
  reference_code  soluzione gold (frammento) — riferimento per il CodeBLEU
  code_context    harness di test COMPLETO: definisce test_execution(solution)
                  (ed eventualmente test_string), che solleva AssertionError se
                  la soluzione è sbagliata. Contiene il segnaposto [insert].
  metadata        dict con problem_id, library, perturbation_type, test_case_cnt…

A differenza di HumanEval/MBPP, il task_id non è una colonna top-level: lo
ricaviamo da metadata.problem_id (intero 0–999) e lo aggiungiamo per compatibilità
con la pipeline (checkpoint, dedup, report).

Nota CodeBLEU: il riferimento è `reference_code` (la soluzione gold), MAI il
`code_context` (che è il grader, non un riferimento di codice).

Come per gli altri benchmark, il dataset viene salvato in locale in
Benchmark/ds1000/ (con save_to_disk) al primo run e riletto da lì in seguito.

⚠️ Esecuzione: a differenza di HumanEval/MBPP (solo stdlib), DS-1000 richiede le
librerie data-science installate (numpy, pandas, scipy, scikit-learn, matplotlib
e — per i rispettivi sottoinsiemi — tensorflow, pytorch). I problemi delle
librerie non installate falliranno con ModuleNotFoundError. Si può filtrare per
libreria con il parametro `libraries`.
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
    """Normalizza un nome di libreria al casing canonico (case-insensitive).
    Solleva ValueError con l'elenco valido se il nome è sconosciuto."""
    key = library.strip().lower()
    if key not in _CANONICAL:
        raise ValueError(
            f"Libreria DS-1000 sconosciuta: {library!r}. "
            f"Valide: {', '.join(DS1000_LIBRARIES)}"
        )
    return _CANONICAL[key]


def _is_importable(library: str) -> bool:
    """True se il modulo della libreria è importabile, SENZA importarlo davvero
    (usa find_spec: niente effetti collaterali né il costo di caricare TF/Torch)."""
    module = DS1000_IMPORT_NAMES[library]
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        # find_spec può sollevare se un pacchetto genitore è rotto: lo trattiamo
        # come "non disponibile" anziché far crashare la pipeline.
        return False


def available_libraries(requested: list[str] | None = None
                        ) -> tuple[list[str], list[str]]:
    """Divide le librerie richieste in (disponibili, mancanti) in base a quali
    moduli sono installati nell'ambiente.

    requested: nomi scelti dall'utente (--libraries), case-insensitive; None =
               tutte e 7. I nomi sconosciuti sollevano ValueError.

    Ritorna (available, missing) con i nomi al casing canonico, preservando
    l'ordine e deduplicando. Es. su Python 3.14 senza TF:
        available_libraries() -> ([Pandas..Pytorch], ["Tensorflow"]).
    """
    names = [_canonical(l) for l in requested] if requested else list(DS1000_LIBRARIES)
    seen: set[str] = set()
    ordered = [n for n in names if not (n in seen or seen.add(n))]
    available = [n for n in ordered if _is_importable(n)]
    missing = [n for n in ordered if n not in available]
    return available, missing


def library_counts() -> dict[str, int]:
    """Numero di problemi per libreria sull'intero set (1000). Utile al report
    per dire quanti problemi vengono saltati quando una libreria non c'è."""
    counts: dict[str, int] = {}
    for p in load_ds1000():
        lib = (p.get("metadata") or {}).get("library", "")
        counts[lib] = counts.get(lib, 0) + 1
    return counts


def plan_run(limit: int | None = None,
             requested: list[str] | None = None) -> tuple[list[dict], dict]:
    """Risolve quali librerie eseguire e carica SOLO i problemi eseguibili.

    Una sola lettura del dataset: rileva le librerie installate fra quelle
    richieste, carica i 1000 problemi, ne tiene i sottoinsiemi disponibili e
    calcola il riepilogo per il report (nessun taglio silenzioso).

    requested: librerie scelte (--libraries) o None = tutte.
    limit: tetto sui problemi DOPO il filtro per libreria (test rapidi).

    Ritorna (problems, info) con info = {
        "available": [...],   librerie eseguite
        "missing":   [...],   librerie richieste ma non installate (saltate)
        "counts":    {lib: n} problemi per libreria sull'intero set
        "total":     1000,    problemi totali del set completo
        "selected":  N,       problemi effettivamente eseguiti (= len(problems))
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
    Carica i 1000 problemi di DS-1000.

    Primo run: scarica test.jsonl da Hugging Face e salva in Benchmark/ds1000/.
    Run successivi: rilegge direttamente da quella cartella (nessun download).

    limit: se valorizzato, prende solo i primi N problemi (test rapidi/economici).
    libraries: se valorizzato, tiene solo i problemi delle librerie indicate
               (es. ["Pandas", "Numpy"]) — utile per saltare i sottoinsiemi le
               cui librerie non sono installate (es. Tensorflow/Pytorch).
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
