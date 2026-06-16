"""
export.py — Export of the per-problem DETAIL to CSV and XLSX.

Adds, alongside the JSON file, two formats that are more convenient to analyze.
The aggregated summary is NOT exported to file (it is shown on screen only, see
report.py).

The CSV is saved in UTF-8 with BOM (utf-8-sig) so Excel displays accents
correctly. The 'code' column contains the generated code (may span lines).
"""

import csv
from pathlib import Path


def _xlsx_safe(value):
    """Keep only the characters valid in XML 1.0 (the .xlsx is XML): drop the C0
    control chars (NULL & co.), the surrogates and the "non-characters" like
    U+FFFF. These can appear in the model's generated code or in stderr and would
    crash openpyxl/lxml (`ValueError: All strings must be XML compatible`).
    Applied to string cells before writing the Excel; other values pass through
    unchanged."""
    if not isinstance(value, str):
        return value
    return "".join(
        c for c in value
        if c in "\t\n\r" or 0x20 <= ord(c) <= 0xD7FF
        or 0xE000 <= ord(c) <= 0xFFFD or 0x10000 <= ord(c) <= 0x10FFFF
    )

# Colonne del dettaglio per-problema: identità + colonne originali del benchmark
# (entry_point, prompt, canonical_solution, test) + "Codice Completo" (riferimento
# CodeBLEU = prompt + canonical_solution) + output del modello e valutazione.
# Una sola colonna esito, "pass@1" (True/False); niente colonne token/costo/stderr.
DETAIL_HEADERS = [
    "task_id", "model_id", "architecture",
    "entry_point", "prompt", "canonical_solution", "test", "Codice Completo",
    "code", "pass@1", "codebleu",
]

# Colonne MBPP: il benchmark non ha firma/entry_point/prompt. Al loro posto la
# descrizione naturale (text), la soluzione di riferimento del dataset
# (code_reference = riferimento CodeBLEU) e i test (test_setup_code + test_list).
# Resta la stessa coppia di metriche finali (pass@1, codebleu) di HumanEval.
MBPP_DETAIL_HEADERS = [
    "task_id", "model_id", "architecture",
    "text", "code_reference", "test_setup_code", "test_list",
    "code", "pass@1", "codebleu",
]

# Colonne DS-1000: oltre a identità ed esito, la libreria target e il tipo di
# perturbazione (Origin/Semantic/Surface/Difficult-Rewrite) — utili per l'analisi
# per-libreria. Il `code_context` (l'harness) NON è esportato (troppo grande).
DS1000_DETAIL_HEADERS = [
    "task_id", "model_id", "architecture",
    "library", "perturbation_type", "prompt", "code_reference",
    "code", "pass@1", "codebleu",
]

# Colonne Plot2Code: input = descrizione (`instruction`), riferimento = script
# matplotlib (`code_reference`). Oltre a pass@1 (= rendering riuscito) e codebleu,
# esplodiamo il confronto visivo composito nelle sue componenti — img_text (OCR),
# img_ssim (struttura), img_color (palette) — più img_visual (composito pesato),
# e teniamo i path alle due immagini (riferimento e generata) per l'ispezione.
PLOT2CODE_DETAIL_HEADERS = [
    "task_id", "model_id", "architecture",
    "url", "instruction", "code_reference",
    "code", "pass@1", "codebleu",
    "img_text", "img_ssim", "img_color", "img_visual",
    "ref_image", "render_path",
]


# Colonne MultiPL-E: benchmark multilinguaggio di SOLA esecuzione (niente gold →
# niente CodeBLEU, quindi nessuna colonna codebleu). Oltre a identità ed esito, il
# `name` del problema, il `language`, l'inizio del programma (`prompt`, firma
# aperta), l'output del modello (`code`) e i `tests` nel linguaggio target.
MULTIPLE_DETAIL_HEADERS = [
    "task_id", "model_id", "architecture",
    "name", "language", "prompt",
    "code", "pass@1", "tests",
]


def _is_multipl_e_record(rec: dict) -> bool:
    """MultiPL-E records have benchmark='multipl-e' (or the language+tests fields,
    unique to this benchmark)."""
    return (rec.get("benchmark") == "multipl-e"
            or ("language" in rec and "tests" in rec))


def _is_mbpp_record(rec: dict) -> bool:
    """MBPP records have `test_list`; HumanEval ones have `test`/`prompt`."""
    return rec.get("benchmark") == "mbpp" or "test_list" in rec


def _is_ds1000_record(rec: dict) -> bool:
    """DS-1000 records have benchmark='ds1000' (or the `library` field)."""
    return rec.get("benchmark") == "ds1000" or "library" in rec


def _is_plot2code_record(rec: dict) -> bool:
    """Plot2Code records have benchmark='plot2code' (or the `instruction` field)."""
    return rec.get("benchmark") == "plot2code" or "instruction" in rec


def _detail_headers(records: list[dict]) -> list[str]:
    """Headers appropriate for the records' benchmark (inferred from the first
    record), since each benchmark exports a different set of columns."""
    if records and _is_multipl_e_record(records[0]):
        return MULTIPLE_DETAIL_HEADERS
    if records and _is_plot2code_record(records[0]):
        return PLOT2CODE_DETAIL_HEADERS
    if records and _is_ds1000_record(records[0]):
        return DS1000_DETAIL_HEADERS
    if records and _is_mbpp_record(records[0]):
        return MBPP_DETAIL_HEADERS
    return DETAIL_HEADERS


def _pass_cell(rec: dict):
    """Render the single pass@1 outcome cell: "passed" if solved, otherwise the
    error category (SyntaxError, AssertionError, TimeoutError, …)."""
    # pass@1: "passed" se risolto, altrimenti la categoria di errore
    # (SyntaxError, AssertionError, TimeoutError, …).
    return "passed" if rec.get("passed") else (rec.get("category") or "")


def _detail_row(rec: dict) -> list:
    """Build the detail row for one record, in the column order of its benchmark
    (MultiPL-E / Plot2Code / DS-1000 / MBPP / HumanEval). Returns the cell values
    aligned with the headers from _detail_headers."""
    cb = (rec.get("metrics") or {}).get("codebleu")
    cb_cell = round(cb, 4) if cb is not None else ""

    if _is_multipl_e_record(rec):
        return [
            rec.get("task_id", ""),
            rec.get("model_id", ""),
            rec.get("architecture", ""),
            rec.get("name", ""),
            rec.get("language", ""),
            rec.get("prompt", ""),
            rec.get("code", ""),
            _pass_cell(rec),
            rec.get("tests", ""),
        ]

    if _is_plot2code_record(rec):
        sim = rec.get("image_similarity") or {}        # None (vecchi record) -> {}
        cell = lambda v: round(v, 4) if isinstance(v, (int, float)) else ""
        return [
            rec.get("task_id", ""),
            rec.get("model_id", ""),
            rec.get("architecture", ""),
            rec.get("url", ""),
            rec.get("instruction", ""),
            rec.get("code_reference", ""),
            rec.get("code", ""),
            _pass_cell(rec),
            cb_cell,
            cell(sim.get("text_match")),
            cell(sim.get("ssim")),
            cell(sim.get("color_sim")),
            cell(sim.get("composite")),
            rec.get("ref_image", ""),
            rec.get("render_path", ""),
        ]

    if _is_ds1000_record(rec):
        return [
            rec.get("task_id", ""),
            rec.get("model_id", ""),
            rec.get("architecture", ""),
            rec.get("library", ""),
            rec.get("perturbation_type", ""),
            rec.get("prompt", ""),
            rec.get("code_reference", ""),
            rec.get("code", ""),
            _pass_cell(rec),
            cb_cell,
        ]

    if _is_mbpp_record(rec):
        tests = rec.get("test_list") or []
        return [
            rec.get("task_id", ""),
            rec.get("model_id", ""),
            rec.get("architecture", ""),
            rec.get("text", ""),
            rec.get("code_reference", ""),
            rec.get("test_setup_code", ""),
            "\n".join(tests) if isinstance(tests, list) else tests,
            rec.get("code", ""),
            _pass_cell(rec),
            cb_cell,
        ]

    return [
        rec.get("task_id", ""),
        rec.get("model_id", ""),
        rec.get("architecture", ""),
        rec.get("entry_point", ""),
        rec.get("prompt", ""),
        rec.get("canonical_solution", ""),
        rec.get("test", ""),
        rec.get("codice_completo", ""),
        rec.get("code", ""),
        _pass_cell(rec),
        cb_cell,
    ]


def records_to_csv(records: list[dict], path: Path) -> None:
    """Write the per-problem detail of `records` to a CSV at `path` (UTF-8 with
    BOM so Excel reads accents correctly), with the benchmark-appropriate
    headers."""
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(_detail_headers(records))
        for rec in records:
            w.writerow(_detail_row(rec))


def to_xlsx(records: list[dict], path: Path) -> None:
    """Write an Excel workbook with the single 'Dettaglio' sheet (per-problem).
    Requires openpyxl (raises ImportError if absent: the caller handles it).
    String cells are sanitized with _xlsx_safe so invalid XML chars don't crash
    the writer."""
    from openpyxl import Workbook
    from openpyxl.styles import Font

    wb = Workbook()
    ws = wb.active
    ws.title = "Dettaglio"
    ws.append(_detail_headers(records))
    for rec in records:
        # Ripuliamo i caratteri non validi in XML: l'.xlsx è XML e li rifiuta.
        ws.append([_xlsx_safe(v) for v in _detail_row(rec)])
    for cell in ws[1]:           # intestazioni in grassetto
        cell.font = Font(bold=True)

    wb.save(path)
