"""
report.py — Aggregazione dei risultati e stampa delle tabelle.

Produce le due viste chiave:
  1. pass@1 per modello
  2. distribuzione degli errori per ARCHITETTURA  (la tua analisi principale)
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

from rich.console import Console, Group
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.box import ROUNDED

from .config import ARCHITECTURES, PRICING

console = Console()


def aggregate(all_records: list[dict]) -> dict:
    """pass@1 per modello + distribuzione errori per architettura."""
    by_model = defaultdict(list)
    by_arch_errors = defaultdict(Counter)
    for r in all_records:
        by_model[r["model"]].append(r)
        if not r["passed"]:
            by_arch_errors[r["architecture"]][r["category"]] += 1

    per_model = {}
    for model, recs in by_model.items():
        passed = sum(1 for r in recs if r["passed"])
        # CodeBLEU medio (ignora i None: problemi dove la metrica non è calcolabile)
        cb_vals = [r["metrics"]["codebleu"] for r in recs
                   if r.get("metrics", {}).get("codebleu") is not None]
        # Token e costo stimato
        in_tok = sum((r.get("usage") or {}).get("input_tokens", 0) for r in recs)
        out_tok = sum((r.get("usage") or {}).get("output_tokens", 0) for r in recs)
        price_in, price_out = PRICING.get(recs[0].get("model_id"), (0.0, 0.0))
        cost = in_tok / 1_000_000 * price_in + out_tok / 1_000_000 * price_out
        # Similarità immagine (solo Plot2Code): media per componente delle metriche
        # disponibili (ignora i None). None se nessun record la riporta.
        sims = [r.get("image_similarity") for r in recs
                if isinstance(r.get("image_similarity"), dict)]

        def _img_mean(key, _sims=sims):
            vals = [s[key] for s in _sims if s.get(key) is not None]
            return sum(vals) / len(vals) if vals else None

        img_sim = ({k: _img_mean(k) for k in
                    ("text_match", "ssim", "color_sim", "composite")} if sims else None)
        per_model[model] = {
            "architecture": recs[0]["architecture"],
            "total": len(recs),
            "passed": passed,
            "pass@1": passed / len(recs) if recs else 0.0,
            "codebleu": sum(cb_vals) / len(cb_vals) if cb_vals else None,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cost_usd": round(cost, 4),
            "errors": dict(Counter(r["category"] for r in recs if not r["passed"])),
            "image_similarity": img_sim,
        }

    return {
        "per_model": per_model,
        "errors_by_architecture": {a: dict(c) for a, c in by_arch_errors.items()},
    }


def save_summary(summary: dict, results_dir: Path) -> Path:
    out = results_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def build_summary_group(summary: dict) -> Group:
    """Costruisce (senza stampare) le tre tabelle raggruppate: pass@1+CodeBLEU,
    distribuzione errori, token/costo. Restituisce un Group pronto da inserire
    in un Panel."""
    title_style = "bold bright_cyan"

    t = Table(title="pass@1 e CodeBLEU per modello", title_style=title_style,
              box=ROUNDED, border_style="grey50")
    t.add_column("modello"); t.add_column("arch")
    t.add_column("pass@1", justify="right"); t.add_column("passati", justify="right")
    t.add_column("CodeBLEU", justify="right")
    for model, s in sorted(summary["per_model"].items(),
                           key=lambda kv: kv[1]["pass@1"], reverse=True):
        cb = s.get("codebleu")
        cb_str = f"{cb:.3f}" if cb is not None else "—"
        t.add_row(model, s["architecture"], f"[bold green]{s['pass@1']*100:.1f}%[/]",
                  f"{s['passed']}/{s['total']}", cb_str)

    t2 = Table(title="Distribuzione errori per ARCHITETTURA", title_style=title_style,
               box=ROUNDED, border_style="grey50")
    t2.add_column("architettura"); t2.add_column("categoria errore")
    t2.add_column("n", justify="right")
    for arch, counter in summary["errors_by_architecture"].items():
        label = ARCHITECTURES.get(arch, arch).split(" (")[0]
        for cat, n in sorted(counter.items(), key=lambda kv: kv[1], reverse=True):
            t2.add_row(label, f"[yellow]{cat}[/]", str(n))
    if not summary["errors_by_architecture"]:
        t2.add_row("[dim]—[/]", "[dim]nessun errore[/]", "[dim]0[/]")

    t3 = Table(title="Token e costo stimato (USD)", title_style=title_style,
               box=ROUNDED, border_style="grey50")
    t3.add_column("modello")
    t3.add_column("token input", justify="right")
    t3.add_column("token output", justify="right")
    t3.add_column("costo $", justify="right")
    tot_in = tot_out = tot_cost = 0
    for model, s in summary["per_model"].items():
        tot_in += s.get("input_tokens", 0)
        tot_out += s.get("output_tokens", 0)
        tot_cost += s.get("cost_usd", 0.0)
        t3.add_row(model, f"{s.get('input_tokens', 0):,}",
                   f"{s.get('output_tokens', 0):,}", f"[bold]${s.get('cost_usd', 0.0):.4f}[/]")
    if len(summary["per_model"]) > 1:
        t3.add_section()
        t3.add_row("[bold]TOTALE[/]", f"[bold]{tot_in:,}[/]",
                   f"[bold]{tot_out:,}[/]", f"[bold]${tot_cost:.4f}[/]")

    parts = [t, Text(""), t2, Text(""), t3]

    # Tabella similarità immagine: solo per Plot2Code (compare se almeno un
    # modello riporta il confronto visivo composito).
    has_img = any(s.get("image_similarity") for s in summary["per_model"].values())
    if has_img:
        fmt = lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else "—"
        t4 = Table(title="Similarità immagine — Plot2Code (media, 1 = identici)",
                   title_style=title_style, box=ROUNDED, border_style="grey50")
        t4.add_column("modello")
        t4.add_column("testo (OCR)", justify="right")
        t4.add_column("struttura (SSIM)", justify="right")
        t4.add_column("colore", justify="right")
        t4.add_column("composito", justify="right")
        for model, s in sorted(summary["per_model"].items(),
                               key=lambda kv: (kv[1].get("image_similarity") or {})
                               .get("composite") or -1, reverse=True):
            img = s.get("image_similarity")
            if not img:
                continue
            t4.add_row(model, fmt(img.get("text_match")), fmt(img.get("ssim")),
                       fmt(img.get("color_sim")),
                       f"[bold green]{fmt(img.get('composite'))}[/]")
        parts += [Text(""), t4]

    return Group(*parts)


def print_summary(summary: dict) -> None:
    """Stampa le tre viste in un unico Panel (uso standalone)."""
    console.print(Panel(
        build_summary_group(summary),
        title="[bold bright_cyan]📊 Risultati benchmark[/]",
        border_style="bright_cyan", box=ROUNDED, padding=(1, 2), expand=False,
    ))
