"""
pipeline.py — Benchmark execution (orchestration) with PAUSE and RESUME.

  run_model(spec, problems)   -> tests ONE model, saving every outcome to a checkpoint
  run_benchmark(models, ...)  -> tests the models, saves the json and prints the report

Used by cli.py and by `python -m model.claude`.

PAUSE / RESUME
---------------
Press Ctrl+C to stop cleanly: the problem in progress is completed and saved,
then the run stops. On the next relaunch it resumes from the problems NOT yet
done, without redoing (and re-paying for) the queries already spent.

Progress is saved problem-by-problem in `results/<model>.jsonl` (append-only
checkpoint): this is the source of truth for resuming, robust even to a forced
stop (double Ctrl+C). API errors (infrastructure) do NOT count as "done" and are
retried on resume. Use fresh=True to ignore the checkpoint and restart from
scratch.

Progress is shown with a rich bar: current problem, percentage, N/M completed,
elapsed time and estimated remaining time (ETA).
"""

import json
import os
import signal
import threading
from pathlib import Path

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from rich.box import ROUNDED, DOUBLE
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    MofNCompleteColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn,
)

from .config import (
    ModelSpec, PROVIDER_ENV_KEYS, CLAUDE_EFFORT, CLAUDE_THINKING, MAX_TOKENS,
)
from . import code_extractor, executor, metrics, report, export
from .prompting import build_prompt
from .providers import generate
from .humaneval import load_humaneval
from .mbpp import load_mbpp
from . import ds1000 as ds1000_loader
from .ds1000 import load_ds1000
from .plot2code import load_plot2code
from .multipl_e import load_multipl_e, plan_run as multipl_e_plan

console = Console()
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Benchmark supportati: nome -> (loader, etichetta, n. problemi del set completo).
# I file di output finiscono in results/<modello>/<benchmark>/ (una cartella per
# modello, così l'Excel per-modello non si sovrascrive tra modelli diversi).
# `timeout` = secondi per test, default per benchmark. HumanEval/MBPP usano solo
# la stdlib (esecuzione rapida). DS-1000 importa librerie data-science pesanti in
# un subprocess fresco a ogni problema: a freddo (cache OneDrive/AV) scipy/sklearn
# possono richiedere 20-30s solo per l'import, quindi serve un margine ampio per
# non scambiare un import lento per un fallimento.
# Plot2Code: input = descrizione testuale (`instruction`), output = script
# matplotlib. La "correttezza funzionale" è il rendering riuscito (il codice gira
# e produce una figura), non un assert. Timeout ampio: ogni esempio è un subprocess
# fresco che importa matplotlib/numpy (lenti a freddo) + disegna.
# MultiPL-E (multilinguaggio, nuprl/MultiPL-E, set HumanEval): UN solo benchmark
# con TUTTI i 24 linguaggi insieme (~161 problemi × 24 ≈ 3864), distinti dalla
# colonna `language`. Benchmark di SOLA esecuzione: niente soluzione gold → niente
# CodeBLEU, metrica = solo pass@1. Il pass@1 richiede il runtime del linguaggio:
# oggi girano JS/PHP/R/Java (+ C++ se installi g++); gli altri danno RuntimeMissing
# finché non installi il toolchain. Timeout ampio: Java/C++ compilano a ogni problema.
BENCHMARKS = {
    "humaneval": {"loader": load_humaneval, "label": "HumanEval", "full_size": 164, "timeout": 10.0},
    "mbpp": {"loader": load_mbpp, "label": "MBPP", "full_size": 500, "timeout": 10.0},
    "ds1000": {"loader": load_ds1000, "label": "DS-1000", "full_size": 1000, "timeout": 60.0},
    "plot2code": {"loader": load_plot2code, "label": "Plot2Code", "full_size": 368, "timeout": 60.0},
    "multipl-e": {"loader": load_multipl_e, "label": "MultiPL-E", "full_size": 3864, "timeout": 30.0},
}

# Evento di stop condiviso: il gestore di Ctrl+C lo "alza", il loop lo controlla.
_stop_event = threading.Event()


def _sigint_handler(signum, frame):
    """First Ctrl+C: requests a clean stop (raises the flag, the loop stops after
    the current problem). Second Ctrl+C: forced stop."""
    if _stop_event.is_set():
        raise KeyboardInterrupt
    _stop_event.set()
    console.print("\n[yellow]⏸  Interruzione richiesta: completo il problema corrente, "
                  "salvo e mi fermo.[/] [dim](Ctrl+C di nuovo = forza)[/]")


def has_key(provider: str) -> bool:
    """True if the provider's API key is set in the environment (used to skip
    models without credentials instead of failing the whole run)."""
    return bool(os.environ.get(PROVIDER_ENV_KEYS[provider], "").strip())


def _progress_columns():
    """Columns of the progress bar: spinner, description, bar, %, N/M, time, ETA."""
    return (
        SpinnerColumn("dots", style="bright_cyan"),
        TextColumn("{task.description}"),
        BarColumn(bar_width=42, style="grey30", complete_style="bright_cyan",
                  finished_style="bold green", pulse_style="cyan"),
        TaskProgressColumn(text_format="[bold]{task.percentage:>3.0f}%[/]"),
        TextColumn("[dim]·[/]"),
        MofNCompleteColumn(),
        TextColumn("[dim]│[/]"),
        TimeElapsedColumn(),
        TextColumn("[dim]· ETA[/]"),
        TimeRemainingColumn(),
    )


def _checkpoint_path(out_dir: Path, key: str) -> Path:
    """Path of a model's append-only checkpoint file (`<key>.jsonl`) inside its
    output directory."""
    return out_dir / f"{key}.jsonl"


def _reference_for(problem: dict, benchmark: str) -> str:
    """Reference for CodeBLEU (code-vs-code, NEVER the tests).

    - MBPP: the `code` field is already the complete reference function.
    - DS-1000: the `reference_code` field (the gold solution), NEVER code_context.
    - Plot2Code: the `code` field (the reference matplotlib script).
    - MultiPL-E: no gold → "" (CodeBLEU not computable).
    - HumanEval: `codice_completo` (prompt+solution, docstring removed)."""
    if benchmark == "mbpp":
        return problem.get("code", "")
    if benchmark == "ds1000":
        return problem.get("reference_code", "")
    if benchmark == "plot2code":
        return problem.get("code", "")
    if benchmark == "multipl-e":
        return ""                       # MultiPL-E non ha gold: niente CodeBLEU
    return problem.get("codice_completo") or problem.get("canonical_solution", "")


def _build_record(problem: dict, code: str, result: dict, usage,
                  spec: ModelSpec, benchmark: str) -> dict:
    """Per-problem record. The common fields (identity, outcome, metrics, tokens)
    are identical across benchmarks; the dataset's original fields differ per
    benchmark (MBPP/DS-1000/Plot2Code/MultiPL-E/HumanEval)."""
    reference = _reference_for(problem, benchmark)
    record = {
        "task_id": problem["task_id"],
        "model": spec.key,
        "model_id": spec.model_id,
        "architecture": spec.architecture,
        "benchmark": benchmark,
    }
    if benchmark == "mbpp":
        # Campi originali MBPP. NB: `code` qui è l'OUTPUT del modello; la soluzione
        # di riferimento del dataset la salviamo come `code_reference`.
        record.update({
            "text": problem.get("text", ""),
            "code_reference": problem.get("code", ""),
            "test_setup_code": problem.get("test_setup_code", ""),
            "test_list": problem.get("test_list", []),
        })
    elif benchmark == "ds1000":
        # Campi DS-1000. NON salviamo `code_context` (l'harness di test, fino a
        # ~120k caratteri: gonfierebbe l'output e non serve all'analisi). La
        # soluzione gold (riferimento CodeBLEU) è `code_reference`.
        meta = problem.get("metadata") or {}
        record.update({
            "library": meta.get("library", ""),
            "perturbation_type": meta.get("perturbation_type", ""),
            "prompt": problem.get("prompt", ""),
            "code_reference": problem.get("reference_code", ""),
        })
    elif benchmark == "plot2code":
        # Campi Plot2Code. `code_reference` = script matplotlib gold (riferimento
        # CodeBLEU). `ref_image` = path al PNG di riferimento; `render_path` = path
        # al PNG generato dal modello (vuoto se non ha prodotto figura), pronti per
        # il confronto visivo. `image_similarity` = confronto visivo composito
        # {text_match, ssim, color_sim, composite} (None se non c'è figura generata).
        from .image_similarity import image_similarity
        record.update({
            "instruction": problem.get("instruction", ""),
            "code_reference": problem.get("code", ""),
            "url": problem.get("url", ""),
            "ref_image": problem.get("ref_image", ""),
            "render_path": result.get("render_path", ""),
            "image_similarity": image_similarity(
                problem.get("ref_image", ""), result.get("render_path", "")),
        })
    elif benchmark == "multipl-e":
        # MultiPL-E (multilinguaggio). Metrica = solo pass@1 (no gold → no CodeBLEU).
        # Teniamo `name`, `language`, `prompt` (firma aperta) e `tests` per l'analisi.
        record.update({
            "name": problem.get("name", ""),
            "language": problem.get("language", ""),
            "prompt": problem.get("prompt", ""),
            "tests": problem.get("tests", ""),
        })
    else:
        record.update({
            "entry_point": problem.get("entry_point", ""),
            "prompt": problem.get("prompt", ""),
            "canonical_solution": problem.get("canonical_solution", ""),
            "codice_completo": reference,
            "test": problem.get("test", ""),
        })
    record.update({
        "code": code,                       # output del modello (tutti i benchmark)
        "passed": result["passed"],
        "category": result["category"],
        # MultiPL-E non ha gold → CodeBLEU non calcolabile (None). Altrove: normale.
        "metrics": {"codebleu": None} if benchmark == "multipl-e"
                   else metrics.all_metrics(reference, code),
        "usage": usage,
        "stderr": result.get("stderr", ""),
    })
    return record


def _load_checkpoint(path: Path) -> dict:
    """Read the checkpoint -> {task_id: record}, keeping the LAST record for each
    task_id (so a retry overwrites the old outcome)."""
    done: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done[r["task_id"]] = r
            except (json.JSONDecodeError, KeyError):
                continue  # riga corrotta (es. crash a metà scrittura): la saltiamo
    return done


def _is_done(record: dict | None) -> bool:
    """A problem is 'done' if it has a real model outcome (even a failed one), but
    NOT if it is an API error: that one must be retried on resume."""
    return record is not None and record.get("category") != "APIError"


def run_model(spec: ModelSpec, problems: list[dict], out_dir: Path,
              benchmark: str = "humaneval", timeout: float = 10.0,
              progress: Progress | None = None, task_id=None) -> list[dict]:
    """Run `spec` over `problems`, resuming from the checkpoint and saving each
    outcome as it goes. Stops cleanly if `_stop_event` is raised."""
    ckpt = _checkpoint_path(out_dir, spec.key)
    done = _load_checkpoint(ckpt)
    todo = [p for p in problems if not _is_done(done.get(p["task_id"]))]
    passed = sum(1 for r in done.values() if r.get("passed"))

    # La barra riflette il progresso reale: parte già dai problemi ripresi.
    if progress is not None and task_id is not None:
        progress.update(task_id, completed=len(problems) - len(todo))

    # 'a' = append: non riscriviamo mai righe vecchie, aggiungiamo solo le nuove.
    with open(ckpt, "a", encoding="utf-8") as f:
        for problem in todo:
            if _stop_event.is_set():
                break  # stop pulito: non iniziamo un nuovo problema

            if progress is not None and task_id is not None:
                progress.update(
                    task_id,
                    description=f"[bold cyan]{spec.key}[/] [dim]·[/] "
                                f"[white]{problem['task_id']}[/] [green]✓{passed}[/]",
                )

            try:
                # DSP: enunciato + stimolo direzionale per-problema (vedi prompting.py)
                raw, usage = generate(spec, build_prompt(problem))
                code = code_extractor.extract_code(raw)
                gen_error = None
            except Exception as e:
                raw, code, usage, gen_error = "", "", None, f"{type(e).__name__}: {e}"

            if gen_error:
                result = {"passed": False, "category": "APIError", "stderr": gen_error}
            else:
                # Plot2Code: salviamo il PNG generato (per il confronto visivo) in
                # results/<modello>/plot2code/rendered/<task_id>.png. `out_dir` è già
                # per-modello, quindi non serve più la sottocartella col nome del modello.
                render_to = None
                if benchmark == "plot2code":
                    render_to = str(out_dir / "rendered" /
                                    f"{problem['task_id']}.png")
                result = executor.run_one(problem, code, timeout=timeout,
                                          render_to=render_to)
                if render_to:
                    result["render_path"] = render_to if result["passed"] else ""

            if result["passed"]:
                passed += 1

            record = _build_record(problem, code, result, usage, spec, benchmark)
            done[problem["task_id"]] = record
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()  # subito su disco: se crasha ora, il problema è già salvato

            if progress is not None and task_id is not None:
                progress.advance(task_id)

    # Set completo nell'ordine dei problemi (deduplicato per task_id).
    return [done[p["task_id"]] for p in problems if p["task_id"] in done]


def run_benchmark(models: list[ModelSpec], benchmark: str = "humaneval",
                  limit: int | None = None, timeout: float | None = None,
                  fresh: bool = False, libraries: list[str] | None = None) -> dict:
    """Load the chosen benchmark, test the models that have an API key, save the
    results to results/<model>/<benchmark>/ and print the report.

    benchmark: "humaneval" (default), "mbpp" or "ds1000".
    fresh=True ignores the existing checkpoints and restarts from scratch (deletes
    the .jsonl).
    libraries: (DS-1000 only) restricts to the listed subsets; the uninstalled
               libraries are skipped with a warning (no silent truncation)."""
    if benchmark not in BENCHMARKS:
        raise ValueError(f"Benchmark sconosciuto: {benchmark!r}. "
                         f"Disponibili: {', '.join(BENCHMARKS)}")
    bench = BENCHMARKS[benchmark]
    if timeout is None:                         # default per benchmark (DS-1000 più alto)
        timeout = bench.get("timeout", 10.0)

    # Ogni modello scrive in una cartella PROPRIA: results/<modello>/<benchmark>/.
    # Così i file (json/csv/jsonl + un results.xlsx per-modello) non si sovrascrivono
    # mai tra modelli diversi (prima il results.xlsx era condiviso per benchmark →
    # il run di un modello cancellava i dati dell'altro nell'Excel).
    def out_dir_for(key: str) -> Path:
        """Output directory for one model: results/<model>/<benchmark>/, so files
        never overwrite each other across different models."""
        return RESULTS_DIR / key / benchmark

    runnable = [m for m in models if has_key(m.provider)]
    for m in (m for m in models if not has_key(m.provider)):
        console.print(f"[yellow]Salto {m.key}: manca {PROVIDER_ENV_KEYS[m.provider]}[/]")
    if not runnable:
        console.print("[red]Nessun modello eseguibile: imposta una API key nel file .env[/]")
        return {"per_model": {}, "errors_by_architecture": {}}

    if fresh:
        for m in runnable:
            _checkpoint_path(out_dir_for(m.key), m.key).unlink(missing_ok=True)

    # --libraries ha senso solo per DS-1000 (gli altri set non hanno librerie).
    if libraries and benchmark != "ds1000":
        console.print(f"[yellow]--libraries è ignorato per il benchmark "
                      f"{bench['label']} (vale solo per DS-1000).[/]")

    ds1000_info = None
    mpe_info = None
    with console.status(f"[bold cyan]Carico {bench['label']}…[/]", spinner="dots"):
        if benchmark == "ds1000":
            # Soluzione A: auto-rileva le librerie installate, esegue solo i loro
            # problemi e salta le mancanti (es. Tensorflow su Python 3.14) con
            # avviso esplicito — vedi il pannello qui sotto.
            problems, ds1000_info = ds1000_loader.plan_run(limit=limit, requested=libraries)
        elif benchmark == "multipl-e":
            # Auto-rileva i linguaggi ESEGUIBILI (runtime presente) e genera SOLO per
            # quelli: i linguaggi senza runtime darebbero `RuntimeMissing`, quindi NON
            # li generiamo affatto (niente spreco di API). I saltati sono nel pannello.
            problems, mpe_info = multipl_e_plan(limit=limit)
        else:
            problems = bench["loader"](limit=limit)

    # Riga DS-1000 dedicata: quante librerie eseguite e quali saltate (con il
    # numero di problemi esclusi), così è chiaro su quanti problemi è il pass@1.
    ds1000_md = ""
    if ds1000_info is not None:
        counts = ds1000_info["counts"]
        total = ds1000_info["total"]
        selected = ds1000_info["selected"]
        ds1000_md = (
            f"\n[bold]Librerie:[/] {', '.join(ds1000_info['available']) or '—'}  "
            f"[dim](pass@1 su {selected}/{total} problemi)[/]"
        )
        if ds1000_info["missing"]:
            skipped = ", ".join(
                f"{lib} ([bold]{counts.get(lib, 0)}[/])"
                for lib in ds1000_info["missing"]
            )
            n_skip = sum(counts.get(lib, 0) for lib in ds1000_info["missing"])
            ds1000_md += (
                f"\n[yellow]⚠ Saltate (modulo non installato):[/] {skipped}  "
                f"[dim]→ {n_skip} problemi esclusi, non eseguiti.[/]"
            )

    # Riga MultiPL-E: quali linguaggi ESEGUONO e quali sono saltati (runtime assente,
    # NON generati → nessun costo API), così è chiaro su quanti linguaggi è il pass@1.
    mpe_md = ""
    if mpe_info is not None:
        avail, missing = mpe_info["available"], mpe_info["missing"]
        mpe_md = (
            f"\n[bold]🌐 Linguaggi:[/] [green]{len(avail)}[/]/{len(mpe_info['requested'])} "
            f"eseguibili  [dim]({', '.join(avail) or '—'})[/]"
        )
        if missing:
            mpe_md += (
                f"\n[yellow]⚠ Saltati (runtime assente → NON generati, nessun costo "
                f"API):[/] {', '.join(missing)}  [dim]({len(missing)} linguaggi)[/]"
            )

    console.print(Panel(
        f"[bold]👥 Modelli:[/]  {', '.join(m.key for m in runnable)}\n"
        f"[bold]🧩 Problemi:[/] [green]{len(problems):,}[/]  [dim]({bench['label']})[/]"
        f"{ds1000_md}{mpe_md}\n"
        f"[bold]⚙️  Config:[/]   effort=[cyan]{CLAUDE_EFFORT}[/]  "
        f"thinking=[cyan]{CLAUDE_THINKING}[/]  max_tokens=[cyan]{MAX_TOKENS}[/]  "
        f"timeout=[cyan]{timeout:g}s[/]\n"
        f"[bold]📊 Metriche:[/] {'pass@1' if benchmark == 'multipl-e' else 'pass@1 + CodeBLEU'}\n"
        f"[dim]⏸  Ctrl+C per mettere in pausa: riprende dal punto di stop al prossimo avvio.[/]",
        title="[bold bright_cyan]⚙️  Configurazione del run[/]",
        subtitle=f"[dim]{bench['label']}[/]",
        border_style="bright_cyan", box=ROUNDED, expand=False, padding=(1, 2),
    ))

    if not problems:
        if ds1000_info is not None and ds1000_info["missing"]:
            console.print(
                "[red]Nessun problema da eseguire:[/] le librerie richieste "
                f"([bold]{', '.join(ds1000_info['missing'])}[/]) non sono installate. "
                "Installale o scegli altre librerie con [cyan]--libraries[/]."
            )
        else:
            console.print("[red]Nessun problema da eseguire.[/]")
        return {"per_model": {}, "errors_by_architecture": {}}

    _stop_event.clear()
    prev_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _sigint_handler)
    try:
        all_records: list[dict] = []
        for spec in runnable:
            out_dir = out_dir_for(spec.key)            # results/<modello>/<benchmark>/
            out_dir.mkdir(parents=True, exist_ok=True)
            with Progress(*_progress_columns(), console=console, expand=False) as progress:
                task = progress.add_task(f"[bold cyan]{spec.key}[/]", total=len(problems))
                records = run_model(spec, problems, out_dir, benchmark=benchmark,
                                    timeout=timeout, progress=progress, task_id=task)
            all_records.extend(records)

            out = out_dir / f"{spec.key}.json"
            out.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
            export.records_to_csv(records, out_dir / f"{spec.key}.csv")
            # Excel PER-MODELLO: scrive solo i record di QUESTO modello, nella sua
            # cartella → niente più collisioni con gli altri modelli.
            try:
                export.to_xlsx(records, out_dir / "results.xlsx")  # solo Dettaglio
            except ImportError:
                console.print("[yellow]openpyxl non installato: salto l'export XLSX "
                              "([dim]pip install openpyxl[/]).[/]")
            except PermissionError:
                # File .xlsx bloccato (tipicamente aperto in Excel): non è un motivo
                # per far fallire l'intero run. I dati sono già salvi in json/csv/jsonl.
                console.print("[yellow]Non riesco a scrivere results.xlsx: il file è "
                              "aperto in un altro programma (Excel?).[/] [dim]I risultati "
                              "sono comunque salvati in .json/.csv/.jsonl; chiudi il file "
                              "e rilancia per aggiornare anche l'Excel.[/]")
            except Exception as e:
                # Qualsiasi altro errore nell'export Excel NON deve far perdere il run:
                # i dati sono già in .json/.csv/.jsonl. Avvisa e prosegui.
                console.print(f"[yellow]Export XLSX saltato ([dim]{type(e).__name__}: "
                              f"{e}[/]).[/] [dim]I risultati sono comunque in "
                              f".json/.csv/.jsonl.[/]")
            passed = sum(1 for r in records if r["passed"])
            pct = passed / len(records) * 100 if records else 0.0
            console.print(f"  [green]✓[/] [bold]{spec.key}[/]: pass@1 = "
                          f"[bold green]{passed}/{len(records)}[/] ({pct:.1f}%)  "
                          f"[dim]· {len(records)}/{len(problems)} completati → {out.name}[/]\n")

            if _stop_event.is_set():
                break  # non passiamo ai modelli successivi

        # Riepilogo aggregato: calcolato per la VISTA a schermo, NON salvato su
        # file (niente summary.json/.csv/sheet, come richiesto).
        summary = report.aggregate(all_records)

        # --- UN UNICO BLOCCO finale: tabelle risultati + percorsi file ---
        files_md = (
            f"\n[bold]File salvati in[/] [bold]{RESULTS_DIR}\\<modello>\\{benchmark}\\[/]\n"
            f"[cyan]•[/] [bold]<modello>.json[/]   [dim]— record completo per problema "
            f"(tutti i campi: HumanEval, codice, esito, CodeBLEU, token, stderr)[/]\n"
            f"[cyan]•[/] [bold]<modello>.csv[/]    [dim]— dettaglio per l'analisi "
            f"(HumanEval + Codice Completo, codice, pass@1, CodeBLEU)[/]\n"
            f"[cyan]•[/] [bold]<modello>.jsonl[/]  [dim]— checkpoint per la ripresa (append-only)[/]\n"
            f"[cyan]•[/] [bold]results.xlsx[/]     [dim]— Excel per-modello (foglio Dettaglio, stesse colonne del CSV)[/]"
        )
        if _stop_event.is_set():
            files_md += (
                "\n\n[yellow]⏸  Run in pausa.[/] Per [bold]riprendere[/] dal punto di stop, "
                "rilancia lo stesso comando ([cyan]python cli.py[/]): i problemi già fatti "
                "verranno saltati."
            )
            title = "[bold yellow]⏸  Run in pausa — riepilogo parziale[/]"
            border = "yellow"
        else:
            title = "[bold green]✓ Fatto — risultati[/]"
            border = "green"

        console.print(Panel(
            Group(report.build_summary_group(summary), Text.from_markup(files_md)),
            title=title, border_style=border, box=DOUBLE, padding=(1, 2), expand=False,
        ))
        return summary
    finally:
        signal.signal(signal.SIGINT, prev_handler)  # ripristina il gestore precedente
