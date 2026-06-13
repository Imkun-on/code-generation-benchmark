"""
prompting.py — Directional Stimulus Prompting (DSP) per la generazione di codice.

DSP (Li et al., 2023) NON è un semplice role-prompt fisso: l'idea è aggiungere,
per OGNI input, dei piccoli "stimoli direzionali" (hint) che puntano il modello
verso la soluzione giusta di QUEL problema specifico.

Qui gli stimoli si estraggono LOCALMENTE dal problema HumanEval (firma, tipo di
ritorno, nome funzione, esempi doctest, parole-chiave di edge case): nessuna
chiamata API aggiuntiva, quindi nessuno spreco di risorse.

  build_prompt(problem) -> str   il messaggio user = enunciato + stimolo direzionale

Il vincolo "rispondi SOLO con codice" vive nel SYSTEM_PROMPT (config.py); qui ci
occupiamo solo di COSA dire al modello, non del formato di output.
"""

import re

# ------------------------------------------------------------------
# Il prompt-guida (directional stimulus, parte fissa).
#
# Versione migliorata del prompt iniziale: tolte le due parti che
# fuorviavano su HumanEval (la variabile 'results' e il ramo "correggi
# codice rotto"), aggiunti i passi direzionali che spingono il modello a
# ragionare su firma/esempi/casi-limite prima di scrivere la soluzione.
# Questa istruzione viene anteposta a OGNI problema da build_prompt().
# ------------------------------------------------------------------
DSP_INSTRUCTION = (
    "Sei un programmatore Python esperto e devi risolvere il problema di "
    "programmazione qui sotto.\n"
    "Ragiona internamente (senza scriverlo) seguendo questi passi:\n"
    "1. Leggi la firma e il docstring: individua input, output e tipo di ritorno.\n"
    "2. Studia gli esempi forniti e i casi limite segnalati negli indizi.\n"
    "3. Scrivi un'implementazione che soddisfi TUTTI gli esempi e i casi limite.\n"
    "Rispondi SOLO con il codice Python completo (firma della funzione + import "
    "necessari), senza spiegazioni, senza testo introduttivo e senza blocchi "
    "markdown o backtick."
)

# Istruzione DSP per MBPP. Differenza chiave rispetto a HumanEval: qui NON c'è
# una firma di funzione da rispettare, solo una descrizione in linguaggio
# naturale (`text`) + una lista di `assert`. Il nome/firma della funzione vanno
# DEDOTTI dai test, che in MBPP sono parte della specifica (Austin et al., 2021):
# mostrarli al modello è la prassi standard, non un aiuto improprio.
DSP_INSTRUCTION_MBPP = (
    "Sei un programmatore Python esperto e devi risolvere il problema di "
    "programmazione qui sotto.\n"
    "Ragiona internamente (senza scriverlo) seguendo questi passi:\n"
    "1. Leggi la descrizione del problema.\n"
    "2. Ricava nome della funzione, numero e ordine degli argomenti dagli "
    "assert di test: la tua funzione DEVE chiamarsi e comportarsi come lì "
    "richiesto.\n"
    "3. Scrivi un'implementazione che superi TUTTI i test elencati.\n"
    "Rispondi SOLO con il codice Python completo (def della funzione + import "
    "necessari), senza spiegazioni, senza testo introduttivo e senza blocchi "
    "markdown o backtick."
)

# Parole che, se presenti nel docstring, segnalano casi limite da non sbagliare.
# Surfacciarle come stimolo direzionale aiuta il modello a non dimenticarli.
_EDGE_CASE_KEYWORDS = [
    "empty", "negative", "zero", "none", "null", "duplicate", "duplicates",
    "sorted", "unsorted", "case", "uppercase", "lowercase", "whitespace",
    "space", "even", "odd", "prime", "round", "rounding", "float", "integer",
    "overflow", "boundary", "edge", "single", "unique", "order",
]

_DEF_RE = re.compile(r"^\s*def\s+\w+\s*\(.*?\)\s*(->\s*[^:]+)?:", re.MULTILINE | re.DOTALL)
_RETURN_RE = re.compile(r"->\s*([^:]+):")
_DOCTEST_RE = re.compile(r"^\s*>>>.*$", re.MULTILINE)
# Nome della funzione da testare: primo identificatore chiamato dentro un assert
# MBPP, es. "assert min_cost([...], 2, 2) == 8" -> "min_cost".
_MBPP_CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")


def _signature_line(prompt: str) -> str | None:
    """Estrae la riga 'def ...:' (firma completa, anche su più righe)."""
    m = _DEF_RE.search(prompt)
    if not m:
        return None
    # Comprime eventuali a-capo della firma in una riga sola e leggibile.
    return re.sub(r"\s+", " ", m.group(0)).strip()


def _return_type(prompt: str) -> str | None:
    m = _RETURN_RE.search(prompt)
    return m.group(1).strip() if m else None


def _doctest_examples(prompt: str, max_examples: int = 3) -> list[str]:
    """Righe doctest (>>> ...) presenti nel docstring: input/output attesi."""
    examples = [line.strip() for line in _DOCTEST_RE.findall(prompt)]
    return examples[:max_examples]


def _edge_cases(prompt: str) -> list[str]:
    """Parole-chiave di edge case citate nel docstring, deduplicate e in ordine."""
    low = prompt.lower()
    found: list[str] = []
    for kw in _EDGE_CASE_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", low) and kw not in found:
            found.append(kw)
    return found


def directional_stimulus(problem: dict) -> str:
    """
    Costruisce lo *stimolo direzionale* (gli hint) per un problema HumanEval.

    Restituisce un blocco di testo con i puntatori chiave verso la soluzione.
    """
    prompt = problem.get("prompt", "")
    entry = problem.get("entry_point") or ""

    hints: list[str] = []
    if entry:
        hints.append(f"- Implementa esattamente la funzione `{entry}`.")

    sig = _signature_line(prompt)
    if sig:
        hints.append(f"- Rispetta la firma: {sig}")

    ret = _return_type(prompt)
    if ret:
        hints.append(f"- Tipo di ritorno atteso: {ret}")

    examples = _doctest_examples(prompt)
    if examples:
        hints.append("- I tuoi risultati devono soddisfare questi esempi:")
        hints.extend(f"    {ex}" for ex in examples)

    edges = _edge_cases(prompt)
    if edges:
        hints.append(f"- Gestisci con attenzione: {', '.join(edges)}.")

    return "\n".join(hints)


# Wrapper comuni negli assert MBPP: NON sono la funzione da implementare, ma la
# avvolgono (es. "assert math.isclose(area(...), 3.14)" o "assert set(f(...)) == ...").
# Quando il primo identificatore chiamato è uno di questi, saltiamo al successivo.
_MBPP_WRAPPERS = {
    "assert", "abs", "round", "len", "set", "sorted", "list", "tuple", "dict",
    "frozenset", "str", "int", "float", "bool", "isclose", "math", "all", "any",
    "sum", "max", "min", "sorted", "tuple",
}


def _mbpp_function_name(problem: dict) -> str | None:
    """Deduce il nome della funzione da testare dal primo assert utile.

    Scorre gli identificatori chiamati (`nome(`) nell'assert e ritorna il primo
    che non sia un wrapper noto (set/sorted/math.isclose/…)."""
    for test in problem.get("test_list", []):
        for name in _MBPP_CALL_RE.findall(test):
            if name not in _MBPP_WRAPPERS:
                return name
    return None


def directional_stimulus_mbpp(problem: dict) -> str:
    """Stimolo direzionale per un problema MBPP.

    Gli hint puntano il modello su: nome della funzione (dedotto dai test) e i
    test che deve superare (in MBPP i test SONO la specifica). Aggiungiamo, se
    presenti nel testo, le parole-chiave di edge case."""
    hints: list[str] = []

    fn = _mbpp_function_name(problem)
    if fn:
        hints.append(f"- Implementa esattamente la funzione `{fn}` (nome desunto dai test).")

    tests = problem.get("test_list", [])
    if tests:
        hints.append("- La tua funzione deve superare ESATTAMENTE questi test:")
        hints.extend(f"    {t}" for t in tests)

    edges = _edge_cases(problem.get("text", ""))
    if edges:
        hints.append(f"- Gestisci con attenzione: {', '.join(edges)}.")

    return "\n".join(hints)


def _build_prompt_mbpp(problem: dict) -> str:
    """Messaggio user per MBPP = descrizione naturale + stimolo direzionale
    (nome funzione dedotto + test da superare)."""
    base = problem.get("text", "").rstrip()
    stimulus = directional_stimulus_mbpp(problem)

    parts = [DSP_INSTRUCTION_MBPP, "# Problema\n" + base]
    if stimulus:
        parts.append("# Stimolo direzionale (indizi per la soluzione)\n" + stimulus)
    return "\n\n".join(parts)


# Istruzione DSP per DS-1000. Il prompt DS-1000 è già auto-contenuto (problema +
# scheletro di codice + marcatori tipo "BEGIN SOLUTION"/"# SOLUTION START"). Al
# modello chiediamo SOLO lo snippet che completa la soluzione: niente marcatori,
# niente reimport di ciò che è già nello scheletro, niente test, niente prosa.
DSP_INSTRUCTION_DS1000 = (
    "Sei un programmatore Python esperto di data science e devi completare lo "
    "snippet di codice qui sotto.\n"
    "Ragiona internamente (senza scriverlo) seguendo questi passi:\n"
    "1. Leggi il problema e lo scaffold di codice già fornito (import, dati).\n"
    "2. Capisci quale risultato è richiesto e con quale libreria.\n"
    "3. Scrivi SOLO il codice della soluzione che va al posto del segnaposto "
    "(dopo `BEGIN SOLUTION`/`# SOLUTION START`).\n"
    "Rispondi SOLO con quel frammento di codice Python: non ripetere i marcatori "
    "(`<code>`, `BEGIN SOLUTION`, …), non reimportare ciò che è già nello scaffold, "
    "non aggiungere test o esempi, niente prosa né blocchi markdown. Segui le "
    "indicazioni di formato nello stimolo direzionale."
)


def _ds1000_solution_form(problem: dict) -> str:
    """Deduce DOVE va inserita la soluzione guardando la riga che precede
    `[insert]` nell'harness (`code_context`):

      - 'function-body' se il blank è dentro un blocco aperto da `def ...:`
        (formato 'Insertion' di DS-1000): la soluzione è il CORPO della funzione,
        va indentata e termina con `return`.
      - 'module' altrimenti (formato 'Completion'): la soluzione sta a livello di
        modulo e deve definire la variabile `result`.

    È uno stimolo DSP estratto LOCALMENTE dal problema (nessuna chiamata extra)."""
    cc = problem.get("code_context", "")
    i = cc.find("[insert]")
    if i == -1:
        return "module"
    before = cc[:i].splitlines()
    prev = before[-1].strip() if before else ""
    if prev.startswith("def ") and prev.endswith(":"):
        return "function-body"
    return "module"


def directional_stimulus_ds1000(problem: dict) -> str:
    """Stimolo direzionale per DS-1000: libreria target + formato della soluzione.

    Il formato (corpo di funzione vs assegnazione a `result` a livello di modulo)
    è dedotto dallo scaffold e allinea l'output a ciò che l'harness si aspetta —
    necessario per il pass@1:
      - 'Insertion': il blank è dentro un `def`; il modello scrive il corpo a
        colonna 0 (l'indentazione la mette l'executor, vedi executor.py), evitando
        il tipico rientro incoerente che causa IndentationError.
      - 'Completion': il blank è a livello di modulo; il modello assegna `result`
        con la sua forma naturale (snippet), usando i dati già forniti SENZA
        reimportarli/ridefinirli (così non sovrascrive l'input del test).

    Nota di progetto: NON forziamo il modello a riprodurre la forma "function-
    wrapper" della gold per alzare il CodeBLEU — sarebbe modellare l'output su una
    metrica anziché misurarlo, aggiunge rischio al pass@1 (il modello tende a
    ricreare anche il setup dati e a sovrascrivere il test) e il CodeBLEU resta
    comunque debole su DS-1000 (soluzioni equivalenti scritte diversamente). Il
    CodeBLEU è metrica SECONDARIA qui; la primaria è il pass@1."""
    hints: list[str] = []
    lib = (problem.get("metadata") or {}).get("library", "")
    if lib:
        hints.append(f"- Usa la libreria {lib}.")

    if _ds1000_solution_form(problem) == "function-body":
        hints.append(
            "- La soluzione è il CORPO della funzione già definita nello scaffold "
            "(`def ...:`): scrivi le istruzioni a partire da colonna 0 (NON "
            "indentare la prima riga) e termina con `return <risultato>`. "
            "All'indentazione per inserirle nella funzione pensa il sistema; non "
            "ripetere la riga `def`."
        )
    else:
        hints.append(
            "- Assegna il risultato finale alla variabile `result`, usando "
            "DIRETTAMENTE i dati e le variabili già definiti nello scaffold "
            "(es. `df`): NON reimportare librerie né ridefinire quei dati, "
            "altrimenti sovrascrivi l'input del test e la verifica fallisce."
        )
    return "\n".join(hints)


def _build_prompt_ds1000(problem: dict) -> str:
    """Messaggio user per DS-1000 = istruzione + il prompt originale (già completo)
    + stimolo direzionale (libreria target + formato della soluzione)."""
    base = problem.get("prompt", "").rstrip()
    stimulus = directional_stimulus_ds1000(problem)

    parts = [DSP_INSTRUCTION_DS1000, "# Problema\n" + base]
    if stimulus:
        parts.append("# Stimolo direzionale (indizi per la soluzione)\n" + stimulus)
    return "\n\n".join(parts)


# Istruzione DSP per Plot2Code. Il task: data la DESCRIZIONE testuale di una
# figura, generare lo script matplotlib completo che la riproduce. Niente input
# multimodale: il modello lavora sulla descrizione. L'output deve essere uno
# script eseguibile end-to-end (import inclusi) che PRODUCE la figura.
DSP_INSTRUCTION_PLOT2CODE = (
    "Sei un programmatore Python esperto di visualizzazione dati con matplotlib. "
    "Ti viene data la descrizione dettagliata di una figura: devi scrivere lo "
    "script Python COMPLETO che la riproduce il più fedelmente possibile.\n"
    "Ragiona internamente (senza scriverlo) seguendo questi passi:\n"
    "1. Individua il tipo di grafico, il numero e la disposizione dei subplot.\n"
    "2. Ricava dati, colori, etichette, titoli, legende e stili dalla descrizione.\n"
    "3. Scrivi uno script eseguibile che generi la figura.\n"
    "Rispondi SOLO con lo script Python completo (import inclusi, es. "
    "`import matplotlib.pyplot as plt`), senza spiegazioni, senza testo "
    "introduttivo e senza blocchi markdown o backtick. Usa matplotlib. NON è "
    "necessario salvare la figura su file: al salvataggio pensa il sistema."
)


def _build_prompt_plot2code(problem: dict) -> str:
    """Messaggio user per Plot2Code = istruzione + descrizione della figura."""
    desc = (problem.get("instruction") or "").rstrip()
    return "\n\n".join([DSP_INSTRUCTION_PLOT2CODE, "# Descrizione della figura\n" + desc])


# Istruzione DSP per MultiPL-E. Task in modalità COMPLETAMENTO: al modello diamo
# l'INIZIO di un programma nel linguaggio target (firma + doc, lasciata aperta) e
# chiediamo di scriverne SOLO la continuazione (il corpo della funzione, chiusura
# inclusa). Il riferimento alla firma resta quello mostrato: ricopiarla è ok (un
# de-dup in executor.py rimuove l'eventuale firma ripetuta), ma il modello NON
# deve riscrivere i commenti né aggiungere test o esempi.
DSP_INSTRUCTION_MULTIPLE = (
    "Sei un programmatore esperto di {lang}. Qui sotto trovi l'INIZIO di un "
    "programma {lang}: una funzione con la sua documentazione, lasciata aperta. "
    "Devi COMPLETARLA.\n"
    "Ragiona internamente (senza scriverlo) seguendo questi passi:\n"
    "1. Leggi la firma e gli esempi nei commenti: individua input, output e "
    "comportamento atteso.\n"
    "2. Implementa il CORPO della funzione in {lang}, gestendo i casi limite.\n"
    "Rispondi SOLO con il codice {lang} che COMPLETA lo snippet (il corpo della "
    "funzione, COMPRESA la sua chiusura). Mantieni ESATTAMENTE il nome e la firma "
    "mostrati. NON ripetere i commenti, NON aggiungere test, esempi o stampe, "
    "niente spiegazioni né blocchi markdown o backtick."
)


def directional_stimulus_multipl_e(problem: dict) -> str:
    """Stimolo direzionale per MultiPL-E: nome funzione (da rispettare) + esempi
    estratti dai commenti `>>> ...` del prompt."""
    hints: list[str] = []

    from .multipl_e import function_name
    fn = function_name(problem.get("name", ""))
    if fn:
        hints.append(f"- Implementa esattamente la funzione `{fn}` (non rinominarla).")

    # Gli esempi nei prompt MultiPL-E sono righe di commento contenenti `>>>`.
    examples = [ln.strip() for ln in problem.get("prompt", "").splitlines()
                if ">>>" in ln][:3]
    if examples:
        hints.append("- La funzione deve soddisfare questi esempi:")
        hints.extend(f"    {ex}" for ex in examples)

    return "\n".join(hints)


def _build_prompt_multipl_e(problem: dict) -> str:
    """Messaggio user per MultiPL-E = istruzione (modalità completamento) +
    l'inizio del programma da completare + stimolo direzionale."""
    from .multipl_e import LANG_LABELS
    lang = LANG_LABELS.get(problem.get("language", ""),
                           problem.get("language", "") or "il linguaggio indicato")
    base = (problem.get("prompt") or "").rstrip()
    stimulus = directional_stimulus_multipl_e(problem)

    parts = [DSP_INSTRUCTION_MULTIPLE.format(lang=lang),
             f"# Inizio del programma da completare ({lang})\n" + base]
    if stimulus:
        parts.append("# Stimolo direzionale (indizi per la soluzione)\n" + stimulus)
    return "\n\n".join(parts)


def build_prompt(problem: dict) -> str:
    """
    Messaggio user finale = enunciato del problema + stimolo direzionale.

    È questo il punto in cui DSP entra in gioco: oltre all'istruzione fissa nel
    system prompt, ogni problema riceve hint mirati che lo guidano alla soluzione.

    Riconosce automaticamente il benchmark: MultiPL-E (presenza di `stop_tokens`),
    Plot2Code (presenza di `instruction`), DS-1000 (presenza di `code_context`),
    MBPP (presenza di `test_list`, descrizione naturale + test), HumanEval (firma
    + docstring).
    """
    if "stop_tokens" in problem:
        return _build_prompt_multipl_e(problem)
    if "instruction" in problem:
        return _build_prompt_plot2code(problem)
    if "code_context" in problem:
        return _build_prompt_ds1000(problem)
    if "test_list" in problem:
        return _build_prompt_mbpp(problem)

    base = problem.get("prompt", "").rstrip()
    stimulus = directional_stimulus(problem)

    parts = [DSP_INSTRUCTION, "# Problema\n" + base]
    if stimulus:
        parts.append("# Stimolo direzionale (indizi per la soluzione)\n" + stimulus)
    return "\n\n".join(parts)
