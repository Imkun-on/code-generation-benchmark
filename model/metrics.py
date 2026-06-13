"""
metrics.py — Metrica strutturale CodeBLEU.

Teniamo SOLO CodeBLEU (qualità strutturale del codice rispetto alla soluzione
di riferimento), accanto a pass@1 (correttezza funzionale, calcolata in
report.py). ROUGE e BLEU sono stati rimossi: non valutano la struttura del
codice e non servono allo scopo "il modello risolve al primo tentativo?".

CodeBLEU combina: BLEU sugli n-grammi + BLEU pesato sulle keyword +
match degli AST + match del data-flow. Richiede `codebleu` e `tree-sitter`.
Se la libreria non è disponibile, restituiamo None invece di interrompere il run.
"""

import logging

from .code_extractor import strip_docstrings


class _DropDataflowWarning(logging.Filter):
    """codebleu logga un WARNING quando il data-flow degenera su funzioni
    banali: rumore innocuo su 164 problemi, lo filtriamo via."""
    def filter(self, record: logging.LogRecord) -> bool:
        return "data-flow match score degenerates" not in record.getMessage()


logging.getLogger().addFilter(_DropDataflowWarning())


def codebleu(reference: str, generated: str, lang: str = "python") -> float | None:
    # Rimuoviamo le docstring da entrambi i lati: il riferimento le contiene
    # sempre (vengono dal prompt HumanEval), il codice generato spesso no.
    # Confrontarli con/senza docstring in modo asimmetrico falserebbe il punteggio.
    reference = strip_docstrings(reference)
    generated = strip_docstrings(generated)
    try:
        from codebleu import calc_codebleu
        result = calc_codebleu([reference], [generated], lang=lang,
                               weights=(0.25, 0.25, 0.25, 0.25))
        return result["codebleu"]
    except Exception:
        return None


def all_metrics(reference: str, generated: str) -> dict:
    """Metriche per-problema. Attualmente solo CodeBLEU (sul Python: gli altri
    benchmark con gold sono in Python; MultiPL-E non calcola CodeBLEU)."""
    return {"codebleu": codebleu(reference, generated)}
