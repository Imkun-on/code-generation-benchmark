"""
cli.py — Benchmark entrypoint (Typer + Rich).

Typer handles the argument parsing (with the `--help` formatted by Rich and the
options grouped into panels); the rest of the output — interactive menu, tables,
progress and summary panels — stays on `rich.Console`.

Equivalent to `python -m model.claude`.

Examples:
    python cli.py                              # interactive menu: choose the benchmark
    python cli.py --list                       # list the models and which have a key
    python cli.py --benchmark mbpp --limit 5   # cheap trial run
    python cli.py -b multipl-e                 # MultiPL-E (24 languages, single file)
    python cli.py --models claude-opus-4.7 --models deepseek-v4-flash   # subset of models

Results in results/<model>/<benchmark>/: <model>.json (complete record), <model>.csv
and results.xlsx (detail, per-model), <model>.jsonl (checkpoint for resuming).
"""

import os
import sys
from typing import List, Optional

import typer
from dotenv import load_dotenv
from rich.console import Console, Group
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from rich.align import Align
from rich.box import ROUNDED, DOUBLE, HEAVY_HEAD

from model.providers import ALL_MODELS, models_by_keys, get_balance
from model.config import PROVIDER_ENV_KEYS
from model.pipeline import run_benchmark, BENCHMARKS

console = Console()

app = typer.Typer(
    add_completion=False,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
    help="🧪 [bold bright_cyan]Code Generation Benchmark[/] — misura il [bold]pass@1[/] "
         "(e il CodeBLEU) di Claude Opus 4.7, confrontandolo [bold]per architettura[/] con "
         "un MoE e un SLM su HumanEval · MBPP · DS-1000 · Plot2Code · MultiPL-E.",
)


def has_key(provider: str) -> bool:
    """True if the provider's API key is set in the environment (drives which
    models appear as runnable in the menus and --list)."""
    return bool(os.environ.get(PROVIDER_ENV_KEYS[provider], "").strip())


# Metadati per la presentazione (icona · descrizione · metrica) di ogni benchmark.
# `label` e `full_size` arrivano da BENCHMARKS; qui aggiungiamo solo l'estetica.
BENCH_META = {
    "humaneval": ("🐍", "Funzioni da firma + docstring",        "pass@1 + CodeBLEU"),
    "mbpp":      ("📝", "Problemi base in linguaggio naturale",  "pass@1 + CodeBLEU"),
    "ds1000":    ("📊", "Data science · 7 librerie",             "pass@1 + CodeBLEU"),
    "plot2code": ("📈", "Descrizione → grafico matplotlib",      "pass@1 + img-sim"),
    "multipl-e": ("🌐", "linguaggi eseguibili · file unico",     "pass@1"),
}

# Icona per architettura (usata da --list).
ARCH_ICON = {"LLM": "🧠", "MoE": "🧩", "SLM": "⚡", "VLM": "👁️"}


def _multipl_e_runnable() -> tuple[int, int]:
    """(number of MultiPL-E languages RUNNABLE now, total number). MultiPL-E only
    generates for the languages whose runtime is present, so in the menu we show
    that number (not 24): it is what will actually be tested. Best-effort: (None,
    total) if not detectable."""
    try:
        from model.executor import multipl_e_runnable
        from model.multipl_e import LANGUAGES
        return sum(1 for lang in LANGUAGES if multipl_e_runnable(lang)), len(LANGUAGES)
    except Exception:
        return None, 24


def print_banner() -> None:
    """Styled header shown at CLI startup."""
    title = Text("🧪  CODE GENERATION BENCHMARK", style="bold bright_cyan", justify="center")
    subtitle = Text("Claude Opus 4.7 · confronto per architettura  LLM · MoE · SLM",
                    style="italic grey70", justify="center")
    benches = Text("HumanEval   MBPP   DS-1000   Plot2Code   MultiPL-E",
                   style="cyan", justify="center")
    console.print(Panel(
        Group(title, Text(""), subtitle, Text(""), Align.center(benches)),
        box=DOUBLE, border_style="bright_cyan", padding=(1, 6), expand=False,
    ))


def _validate_benchmark(value: Optional[str]) -> Optional[str]:
    """Validate the benchmark name (None = it will be asked via the interactive
    menu). Raises typer.BadParameter for an unknown name."""
    if value is not None and value not in BENCHMARKS:
        raise typer.BadParameter(
            f"{value!r} non è un benchmark valido. Disponibili: {', '.join(BENCHMARKS)}.")
    return value


def choose_benchmark() -> str:
    """Interactive menu to choose the benchmark when it is not passed via CLI.

    Shows the available benchmarks (with their problem counts) and asks which to
    use, with HumanEval as the default. Used only in interactive mode (terminal);
    if the input is not a terminal it falls back to the default without blocking."""
    if not sys.stdin.isatty():
        return "humaneval"  # non interattivo (pipe/CI): default, niente blocchi

    table = Table(
        title="🧪  Quale benchmark vuoi testare?", title_style="bold bright_cyan",
        box=HEAVY_HEAD, border_style="bright_cyan", header_style="bold cyan",
        expand=False, padding=(0, 1),
    )
    table.add_column("#", justify="right", style="bright_cyan")
    table.add_column("Benchmark", style="bold")
    table.add_column("Problemi", justify="right", style="green")
    table.add_column("Metrica", style="grey70")
    table.add_column("Descrizione", style="grey70")

    names = list(BENCHMARKS)
    mpe_run, mpe_tot = _multipl_e_runnable()   # linguaggi eseguibili ORA (non 24)
    for i, name in enumerate(names, start=1):
        b = BENCHMARKS[name]
        icon, desc, metric = BENCH_META.get(name, ("•", "", ""))
        default = "  [dim](default)[/]" if name == "humaneval" else ""
        problemi = f"{b['full_size']:,}"
        # MultiPL-E: mostriamo i linguaggi ESEGUIBILI e i problemi corrispondenti
        # (solo quelli vengono generati), non il totale teorico dei 24 linguaggi.
        if name == "multipl-e" and mpe_run is not None:
            desc = f"{mpe_run}/{mpe_tot} linguaggi eseguibili · file unico"
            problemi = f"~{mpe_run * (b['full_size'] // mpe_tot):,}"
        table.add_row(str(i), f"{icon}  {b['label']}{default}",
                      problemi, metric, desc)

    console.print(table)
    console.print("[dim]↳ Rispondi col numero o col nome  ·  Invio = HumanEval[/]")

    choices = [str(i) for i in range(1, len(names) + 1)] + names
    sel = Prompt.ask("[bold bright_cyan]Scelta[/]", choices=choices,
                     default="humaneval", show_choices=False)
    return names[int(sel) - 1] if sel.isdigit() else sel


def cmd_list() -> None:
    """List the configured models and whether their API key is set."""
    table = Table(
        title="🤖  Modelli configurati", title_style="bold bright_cyan",
        box=HEAVY_HEAD, border_style="bright_cyan", header_style="bold cyan",
        expand=False, padding=(0, 1),
    )
    table.add_column("modello", style="bold")
    table.add_column("architettura")
    table.add_column("provider", style="grey70")
    table.add_column("API key", justify="center")
    table.add_column("nota", style="grey70")
    for m in ALL_MODELS:
        ok = "[bold green]✓[/]" if has_key(m.provider) else "[red]✗[/]"
        icon = ARCH_ICON.get(m.architecture, "")
        table.add_row(m.key, f"{icon}  {m.architecture}", m.provider, ok, m.note)
    console.print(table)
    console.print("[dim]↳ Vengono eseguiti solo i modelli con la [green]✓[/dim] "
                  "[dim](API key impostata in .env).[/]")


# Pagine di ricarica credito per provider (mostrate nell'avviso "saldo a zero").
TOPUP_URLS = {
    "deepseek": "https://platform.deepseek.com/top_up",
    "anthropic": "https://console.anthropic.com/settings/billing",
    "mistral": "https://console.mistral.ai/billing",
}


def _balance_cell(provider: str) -> str:
    """'Balance' cell for the models table: a colored amount (red if zero) if the
    provider exposes the balance via API (DeepSeek), otherwise 'n/a'."""
    bal = get_balance(provider)
    if bal is None:
        return "[grey50]n/d (solo console)[/]"
    zero = (not bal["available"]) or bal["amount"] <= 0
    color = "red" if zero else "green"
    flag = "  [red]⚠[/]" if zero else ""
    return f"[{color}]{bal['amount']:.2f} {bal['currency']}[/]{flag}"


def choose_models(benchmark: str) -> Optional[List[str]]:
    """Interactive menu: which model to test, with the API balance where the
    provider exposes it. Returns the chosen keys; None = all those with a key
    (also when it is not a terminal, so as not to block pipe/CI)."""
    runnable = [m for m in ALL_MODELS if has_key(m.provider)]
    if not sys.stdin.isatty() or not runnable:
        return None

    table = Table(
        title="🤖  Quale modello vuoi testare?", title_style="bold bright_cyan",
        box=HEAVY_HEAD, border_style="bright_cyan", header_style="bold cyan",
        expand=False, padding=(0, 1),
    )
    table.add_column("#", justify="right", style="bright_cyan")
    table.add_column("modello", style="bold")
    table.add_column("architettura")
    table.add_column("API key", justify="center")
    table.add_column("saldo", justify="right")

    # Il saldo richiede una chiamata di rete (solo DeepSeek): mostriamo uno spinner.
    rows = []
    with console.status("[bold cyan]Controllo i saldi API…[/]", spinner="dots"):
        for i, m in enumerate(ALL_MODELS, start=1):
            ok = has_key(m.provider)
            saldo = _balance_cell(m.provider) if ok else "[grey50]—[/]"
            rows.append((i, m, ok, saldo))

    for i, m, ok, saldo in rows:
        icon = ARCH_ICON.get(m.architecture, "")
        keytxt = "[bold green]✓[/]" if ok else "[red]✗[/]"
        idx = str(i) if ok else "[grey50]–[/]"
        table.add_row(idx, m.key, f"{icon}  {m.architecture}", keytxt, saldo)
    console.print(table)
    console.print("[dim]↳ Numero o nome del modello  ·  Invio = tutti quelli con la chiave[/]")

    choices = [str(i) for i, m, ok, _ in rows if ok] + [m.key for m in runnable] + ["tutti"]
    sel = Prompt.ask("[bold bright_cyan]Scelta[/]", choices=choices,
                     default="tutti", show_choices=False)
    if sel == "tutti":
        return [m.key for m in runnable]
    if sel.isdigit():
        return [ALL_MODELS[int(sel) - 1].key]
    return [sel]


def warn_zero_balance(specs) -> None:
    """Prominent warning if a selected model has zero credit (only where the
    balance is known, i.e. DeepSeek). Applies both from the menu and from
    --models."""
    for m in specs:
        if not has_key(m.provider):
            continue
        bal = get_balance(m.provider)
        if bal is None or (bal["available"] and bal["amount"] > 0):
            continue
        url = TOPUP_URLS.get(m.provider, "(vedi la console del provider)")
        console.print(Panel(
            f"[bold]{m.key}[/] ({m.provider}): saldo "
            f"[bold red]{bal['amount']:.2f} {bal['currency']}[/].\n"
            f"Le chiamate falliranno con [red]APIError[/] finché non ricarichi.\n"
            f"[bold]Ricarica qui:[/] {url}",
            title="⚠️  Credito a zero — ricarica necessaria",
            border_style="red", expand=False, padding=(1, 2),
        ))


def main(
    benchmark: Optional[str] = typer.Option(
        None, "--benchmark", "-b", metavar="NAME", callback=_validate_benchmark,
        rich_help_panel="Benchmark",
        help="Benchmark da eseguire: humaneval / mbpp / ds1000 / plot2code / multipl-e. "
             "Se omesso, lo chiede con un menù interattivo."),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-n", min=1, metavar="N", rich_help_panel="Benchmark",
        help="Numero di problemi (default: tutti). Per MultiPL-E è PER LINGUAGGIO."),
    models: Optional[List[str]] = typer.Option(
        None, "--models", "-m", metavar="KEY", rich_help_panel="Modelli",
        help="Chiavi modello da testare (ripeti per più modelli). Default: tutti quelli con API key."),
    libraries: Optional[List[str]] = typer.Option(
        None, "--libraries", metavar="LIB", rich_help_panel="Esecuzione",
        help="(solo DS-1000) limita alle librerie indicate (ripeti per più, es. "
             "--libraries Pandas --libraries Numpy). Default: tutte le installate."),
    timeout: Optional[float] = typer.Option(
        None, "--timeout", "-t", metavar="SEC", rich_help_panel="Esecuzione",
        help="Timeout per test in secondi (default per benchmark: 10 HumanEval/MBPP, "
             "60 DS-1000/Plot2Code, 30 MultiPL-E)."),
    fresh: bool = typer.Option(
        False, "--fresh", rich_help_panel="Esecuzione",
        help="Ignora il checkpoint e riparte da zero (cancella i .jsonl)."),
    list_models: bool = typer.Option(
        False, "--list", rich_help_panel="Info",
        help="Elenca i modelli configurati ed esci."),
) -> None:
    """Run a code generation benchmark and save the results to [bold]results/<model>/<benchmark>/[/].

    This is the Typer command body: it loads the .env, resolves the benchmark and
    the models (from flags or interactive menus), warns about zero balances and
    then delegates to run_benchmark."""
    load_dotenv()
    print_banner()

    if list_models:
        cmd_list()
        raise typer.Exit()

    # Nessun --benchmark da CLI: lo chiediamo in modo interattivo.
    chosen = benchmark or choose_benchmark()

    # Quale modello testare: --models da CLI, altrimenti menù interattivo (con saldo).
    selected = models if models else choose_models(chosen)
    specs = models_by_keys(selected)

    # Avviso ricarica se un modello scelto ha credito a zero (dove il saldo è noto).
    warn_zero_balance(specs)

    run_benchmark(specs, benchmark=chosen,
                  limit=limit, timeout=timeout, fresh=fresh, libraries=libraries)


app.command()(main)


if __name__ == "__main__":
    app()
