"""
metrics.py — CodeBLEU structural metric.

We keep ONLY CodeBLEU (structural quality of the code relative to the reference
solution), alongside pass@1 (functional correctness, computed in report.py).
ROUGE and BLEU were removed: they do not assess code structure and are not
needed for the goal "does the model solve on the first attempt?".

CodeBLEU combines: BLEU over n-grams + BLEU weighted on keywords + AST match +
data-flow match. It requires `codebleu` and `tree-sitter`. If the library is
unavailable, we return None instead of aborting the run.
"""

import logging

from .code_extractor import strip_docstrings


class _DropDataflowWarning(logging.Filter):
    """codebleu logs a WARNING when the data-flow degenerates on trivial
    functions: harmless noise over 164 problems, so we filter it out."""
    def filter(self, record: logging.LogRecord) -> bool:
        """Return False (drop the log record) for the known degenerate-data-flow
        warning, True (keep it) for everything else."""
        return "data-flow match score degenerates" not in record.getMessage()


logging.getLogger().addFilter(_DropDataflowWarning())


def codebleu(reference: str, generated: str, lang: str = "python") -> float | None:
    """Compute the CodeBLEU score between a reference and a generated solution.

    Strips docstrings from both sides first (so the comparison is symmetric and
    fair, see strip_docstrings), then calls `codebleu` with uniform weights.
    Returns the score in [0, 1], or None if the library is unavailable or the
    computation fails (so a missing dependency never aborts the run)."""
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
    """Per-problem metrics. Currently only CodeBLEU (on Python: the other
    benchmarks with a gold solution are in Python; MultiPL-E does not compute
    CodeBLEU). Returned as a dict so more metrics can be added without changing
    callers."""
    return {"codebleu": codebleu(reference, generated)}
