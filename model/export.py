"""
export.py — Esportazione del DETTAGLIO per-problema in CSV e XLSX.

Affianca al file JSON due formati più comodi da analizzare. Il riepilogo
aggregato NON viene esportato su file (si vede solo a schermo, vedi report.py).

Il CSV è salvato in UTF-8 con BOM (utf-8-sig) così Excel mostra correttamente
gli accenti. La colonna 'code' contiene il codice generato (può andare a capo).
"""

import csv
from pathlib import Path


def _xlsx_safe(value):
    """Tiene solo i caratteri validi in XML 1.0 (l'.xlsx è XML): scarta i control
    C0 (NULL & co.), i surrogati e i "non-caratteri" come U+FFFF. Questi possono
    comparire nel codice generato dal modello o nello stderr e farebbero crashare
    openpyxl/lxml (`ValueError: All strings must be XML compatible`). Applicato alle
    celle stringa prima di scrivere l'Excel; gli altri valori passano invariati."""
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
    """I record MultiPL-E hanno benchmark='multipl-e' (o i campi language+tests,
    unici a questo benchmark)."""
    return (rec.get("benchmark") == "multipl-e"
            or ("language" in rec and "tests" in rec))


def _is_mbpp_record(rec: dict) -> bool:
    """I record MBPP hanno `test_list`; quelli HumanEval hanno `test`/`prompt`."""
    return rec.get("benchmark") == "mbpp" or "test_list" in rec


def _is_ds1000_record(rec: dict) -> bool:
    """I record DS-1000 hanno benchmark='ds1000' (o il campo `library`)."""
    return rec.get("benchmark") == "ds1000" or "library" in rec


def _is_plot2code_record(rec: dict) -> bool:
    """I record Plot2Code hanno benchmark='plot2code' (o il campo `instruction`)."""
    return rec.get("benchmark") == "plot2code" or "instruction" in rec


def _detail_headers(records: list[dict]) -> list[str]:
    """Intestazioni adatte al benchmark dei record (dedotto dal primo record)."""
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
    # pass@1: "passed" se risolto, altrimenti la categoria di errore
    # (SyntaxError, AssertionError, TimeoutError, …).
    return "passed" if rec.get("passed") else (rec.get("category") or "")


def _detail_row(rec: dict) -> list:
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
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(_detail_headers(records))
        for rec in records:
            w.writerow(_detail_row(rec))


def to_xlsx(records: list[dict], path: Path) -> None:
    """Scrive un workbook Excel con il solo foglio 'Dettaglio' (per-problema).
    Richiede openpyxl (solleva ImportError se assente: il chiamante lo gestisce)."""
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
