"""
code_extractor.py — Extracts the Python code from the model's raw response.

The models almost always wrap the code in a ```python ... ``` block. Sometimes
they repeat the function signature, sometimes they add prose. This module
isolates only the executable code.
"""

import ast
import re
import warnings

# Blocco markdown ```<lang>\n...```. L'etichetta di linguaggio è opzionale e
# QUALSIASI (python, py, javascript, js, php, r, java, …): MultiPL-E genera codice
# non-Python, quindi il vecchio `(?:python|py)?` perdeva i fence ```javascript.
_FENCE_RE = re.compile(r"```[^\n`]*\n(.*?)```", re.DOTALL)

# Nodi che possono avere una docstring come PRIMO statement del corpo.
_DOCSTRING_OWNERS = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def strip_docstrings(code: str) -> str:
    """
    Remove docstrings (module / functions / classes) from `code`, leaving
    everything else intact (imports, signatures, body) and the original
    formatting.

    Its purpose is to make the CodeBLEU comparison FAIR: the HumanEval reference
    (`codice_completo`) always includes the docstring in the `prompt`, while the
    model's generated code often does not. If one side has the docstring and the
    other does not, the n-grams/AST/data-flow diverge and CodeBLEU collapses even
    though the logic is identical. By removing it from BOTH sides, the comparison
    measures only the code structure.

    Implementation: it uses the AST to locate the lines occupied by docstrings
    and removes them from the textual source (no `ast.unparse`, which would
    rewrite and alter the formatting). If the source is not parseable, it returns
    it unchanged (better to compare it than discard it).
    """
    if not code:
        return code
    try:
        # Molte soluzioni MBPP/HumanEval usano regex con escape non-raw (es.
        # "\w", "\*"): ast.parse emette un SyntaxWarning innocuo per ciascuna.
        # Lo silenziamo SOLO qui (l'analisi del codice altrui non è un nostro
        # bug), senza nascondere altri warning del programma.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(code)
    except SyntaxError:
        return code

    remove: set[int] = set()  # numeri di riga (1-based) da eliminare
    for node in ast.walk(tree):
        if not isinstance(node, _DOCSTRING_OWNERS):
            continue
        body = getattr(node, "body", [])
        # Docstring = primo statement è un'espressione costante stringa.
        # Richiediamo len(body) > 1 per non svuotare un corpo (che diventerebbe
        # sintatticamente invalido) quando la docstring è l'unico statement.
        if (len(body) > 1
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            doc = body[0]
            remove.update(range(doc.lineno, (doc.end_lineno or doc.lineno) + 1))

    if not remove:
        return code

    lines = code.splitlines()
    kept = [ln for i, ln in enumerate(lines, start=1) if i not in remove]
    return "\n".join(kept)

# Marcatori del formato risposta DS-1000 che il modello può ripetere nell'output:
# <code>/</code>, BEGIN/END SOLUTION, # SOLUTION START/END. Non sono Python valido
# (romperebbero l'esecuzione e abbasserebbero il CodeBLEU), quindi li rimuoviamo.
# Sono innocui per HumanEval/MBPP, dove non compaiono mai.
_DS1000_MARKER_RE = re.compile(
    r"^[ \t]*(?:</?code>|BEGIN\s+SOLUTION|END\s+SOLUTION|#\s*SOLUTION\s+(?:START|END))[ \t]*$",
    re.MULTILINE | re.IGNORECASE,
)


def strip_ds1000_markers(code: str) -> str:
    """Remove the DS-1000 format marker lines (`<code>`, `BEGIN/END SOLUTION`,
    `# SOLUTION START/END`) from the model's output. No-op for HumanEval/MBPP,
    where they never appear; needed on DS-1000 so the markers don't break
    execution or lower CodeBLEU."""
    return _DS1000_MARKER_RE.sub("", code)


# Una riga "di codice" tipica inizia con uno di questi token.
_CODE_START_RE = re.compile(r"^\s*(?:import\s|from\s|def\s|class\s|@|#)")


def _trim_blank_edges(code: str) -> str:
    """Remove leading/trailing blank lines and trailing whitespace on each line,
    but PRESERVE the indentation of the first code line.

    Replaces the old global `.strip()`, which erased the first line's
    indentation: harmless for HumanEval/MBPP (there the solution starts at column
    0) but disastrous for DS-1000 'Insertion', where the solution is the indented
    BODY of a function (`def f(df):` + body) and losing the first line's 4 spaces
    produces an IndentationError even though the logic is correct."""
    lines = code.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(line.rstrip() for line in lines)


def _strip_leading_prose(code: str) -> str:
    """
    Remove any leading prose lines (e.g. "Here is the solution:") up to the
    first line that looks like Python code.

    If no line looks like code, returns the text unchanged (better to try to run
    it than to discard it).
    """
    lines = code.splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        if _CODE_START_RE.match(line):
            return _trim_blank_edges("\n".join(lines[i:]))
        # Prima riga non vuota e NON di codice: prosa → la salto.
    return _trim_blank_edges(code)


def extract_code(raw: str) -> str:
    """
    Return the Python code contained in the response.

    The SYSTEM_PROMPT asks for code only without fences, but a model may
    disobey. Strategy, in order:
      1. if there is a ```...``` markdown block, take the first one;
      2. otherwise strip any leading prose and keep the rest.

    This is the function the pipeline calls on every raw model response before
    executing it / scoring it.
    """
    if not raw:
        return ""

    match = _FENCE_RE.search(raw)
    code = match.group(1) if match else _strip_leading_prose(raw)
    # Rimuove eventuali marcatori DS-1000 ripetuti dal modello (no-op altrove).
    # _trim_blank_edges (non .strip()) preserva l'indentazione della prima riga,
    # necessaria per il formato 'Insertion' di DS-1000.
    return _trim_blank_edges(strip_ds1000_markers(code))
