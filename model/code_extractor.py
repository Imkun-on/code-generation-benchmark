"""
code_extractor.py — Estrae il codice Python dalla risposta grezza del modello.

I modelli racchiudono quasi sempre il codice in un blocco ```python ... ```.
A volte ripetono la firma della funzione, a volte aggiungono testo.
Questa funzione isola solo il codice eseguibile.
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
    Rimuove le docstring (modulo / funzioni / classi) da `code`, lasciando
    intatto tutto il resto (import, firme, corpo) e la formattazione originale.

    Serve a rendere EQUO il confronto CodeBLEU: il riferimento HumanEval
    (`codice_completo`) include sempre la docstring nel `prompt`, mentre il
    codice generato dal modello spesso no. Se uno dei due ha la docstring e
    l'altro no, gli n-grammi/AST/data-flow divergono e il CodeBLEU crolla pur
    essendo la logica identica. Eliminandola da ENTRAMBI i lati il confronto
    misura solo la struttura del codice.

    Implementazione: individua con l'AST le righe occupate dalle docstring e le
    rimuove dal sorgente testuale (niente `ast.unparse`, che riscriverebbe e
    altererebbe la formattazione). Se il sorgente non è parsabile, lo
    restituisce invariato (meglio confrontarlo che scartarlo).
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
    """Rimuove le righe-marcatore del formato DS-1000 dall'output del modello."""
    return _DS1000_MARKER_RE.sub("", code)


# Una riga "di codice" tipica inizia con uno di questi token.
_CODE_START_RE = re.compile(r"^\s*(?:import\s|from\s|def\s|class\s|@|#)")


def _trim_blank_edges(code: str) -> str:
    """Rimuove righe vuote iniziali/finali e spazi in coda a ogni riga, ma
    PRESERVA l'indentazione della prima riga di codice.

    Sostituisce il vecchio `.strip()` globale, che cancellava l'indentazione
    della prima riga: innocuo per HumanEval/MBPP (lì la soluzione parte a colonna
    0) ma rovinoso per DS-1000 'Insertion', dove la soluzione è il CORPO indentato
    di una funzione (`def f(df):` + corpo) e perdere i 4 spazi della prima riga
    produce un IndentationError pur essendo la logica corretta."""
    lines = code.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(line.rstrip() for line in lines)


def _strip_leading_prose(code: str) -> str:
    """
    Rimuove eventuali righe di prosa iniziali (es. "Ecco la soluzione:")
    fino alla prima riga che sembra codice Python.

    Se nessuna riga sembra codice, restituisce il testo invariato (meglio
    provare a eseguirlo che scartarlo).
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
    Restituisce il codice Python contenuto nella risposta.

    Il SYSTEM_PROMPT chiede solo codice senza fence, ma un modello può
    disobbedire. Strategia, in ordine:
      1. se c'è un blocco markdown ```...```, prende il primo;
      2. altrimenti rimuove eventuale prosa iniziale e tiene il resto.
    """
    if not raw:
        return ""

    match = _FENCE_RE.search(raw)
    code = match.group(1) if match else _strip_leading_prose(raw)
    # Rimuove eventuali marcatori DS-1000 ripetuti dal modello (no-op altrove).
    # _trim_blank_edges (non .strip()) preserva l'indentazione della prima riga,
    # necessaria per il formato 'Insertion' di DS-1000.
    return _trim_blank_edges(strip_ds1000_markers(code))
