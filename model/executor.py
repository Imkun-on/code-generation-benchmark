"""
executor.py — Esegue il codice generato contro gli unit test di HumanEval
e classifica l'esito (pass@1) e, in caso di fallimento, la CATEGORIA di errore.

La classificazione per categoria è il dato centrale per analizzare
"dove sbagliano" le diverse architetture.

⚠️  SICUREZZA: il codice generato dai modelli è codice non fidato.
Lo eseguiamo in un sottoprocesso separato con timeout. Lancia il benchmark
solo su una macchina/ambiente che puoi sacrificare (o in un container).
"""

import glob
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import os


def _kill_tree(pid: int) -> None:
    """Uccide il processo `pid` E TUTTI i suoi discendenti.

    Necessario perché `proc.kill()` su Windows (TerminateProcess) termina SOLO il
    processo diretto, non i suoi figli. Quando il codice generato spawna un
    sottoprocesso (es. bash che apre una pipe `| bc`, una command-substitution in
    loop, un job in `&`), quel nipote EREDITA le pipe di stdout/stderr e resta vivo:
    la `communicate()` che segue il kill aspetterebbe per sempre l'EOF su una pipe
    ancora aperta → l'intero benchmark si blocca all'infinito su quel problema."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                       capture_output=True)
    else:
        import signal as _signal
        try:
            os.killpg(os.getpgid(pid), _signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def _run_guarded(cmd, timeout=None, **kwargs):
    """Come `subprocess.run(..., timeout=...)`, ma al timeout uccide l'INTERO albero
    di processi invece del solo figlio diretto (vedi _kill_tree). Solleva
    `subprocess.TimeoutExpired` esattamente come `subprocess.run`, così i call-site
    esistenti che lo intercettano non cambiano. Comportamento identico a
    `subprocess.run` quando non c'è timeout."""
    capture = kwargs.pop("capture_output", False)
    if capture:
        kwargs.setdefault("stdout", subprocess.PIPE)
        kwargs.setdefault("stderr", subprocess.PIPE)
    if os.name != "nt":
        kwargs.setdefault("start_new_session", True)  # gruppo proprio → killpg
    with subprocess.Popen(cmd, **kwargs) as proc:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_tree(proc.pid)
            try:
                # Ora i discendenti sono morti: le pipe si chiudono, communicate ritorna.
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
            raise subprocess.TimeoutExpired(cmd, timeout, output=stdout, stderr=stderr)
        except BaseException:
            _kill_tree(proc.pid)
            raise
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


# Categorie di errore (allineate al README).
ERROR_CATEGORIES = [
    "Passed",
    "SyntaxError",
    "IndentationError",
    "NameError",
    "TypeError",
    "ValueError",
    "IndexError",
    "KeyError",
    "AttributeError",
    "AssertionError",
    "RecursionError",
    "ImportError",          # incl. ModuleNotFoundError: libreria non installata (DS-1000)
    "TimeoutError",
    "RuntimeError",   # catch-all per eccezioni a runtime non elencate
    "EmptyOutput",    # il modello non ha prodotto codice
    "NoFigure",       # Plot2Code: il codice gira ma non produce alcuna figura
    "RuntimeMissing", # MultiPL-E: runtime del linguaggio non installato/non configurato (es. JDK/Rscript/php/ruby/g++)
]

# Eccezioni che mappiamo 1:1 sul loro nome. ModuleNotFoundError è sottoclasse di
# ImportError: lo normalizziamo a "ImportError" (vedi _classify_stderr) così su
# DS-1000 una libreria mancante è subito riconoscibile.
_KNOWN = {
    "SyntaxError", "IndentationError", "NameError", "TypeError", "ValueError",
    "IndexError", "KeyError", "AttributeError", "AssertionError", "RecursionError",
    "ImportError", "ModuleNotFoundError",
}

# L'ultima riga di un traceback è l'eccezione: "NomeErrore" oppure "NomeErrore: messaggio".
# Accettiamo anche nomi con punto (es. modulo.Eccezione) e prendiamo l'ultimo segmento.
_EXC_LINE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_.]*)(?::.*)?$")


def _imports_header(prompt: str) -> str:
    """Estrae le righe di import iniziali dal prompt HumanEval.

    Servono perché il test (e a volte la soluzione) usano tipi come List/Dict
    importati in cima al prompt originale.

    ⚠️ Prendiamo SOLO gli import a livello di modulo (colonna 0, NON indentati).
    Se usassimo line.strip(), una riga di docstring che va a capo iniziando con
    "from"/"import" verrebbe scambiata per un import e iniettata nel programma.
    È esattamente ciò che accadeva su HumanEval/99, la cui docstring contiene
    "...equidistant\n    from two integers, round it away from zero.": quella riga
    finiva in cima al file → IndentationError → una soluzione CORRETTA falliva.
    """
    header = []
    for line in prompt.splitlines():
        if line.startswith(("import ", "from ")):  # niente strip: solo colonna 0
            header.append(line)
    return "\n".join(header)


def _build_program(problem: dict, code: str) -> str:
    """Assembla: import + funzione generata + test + chiamata di check."""
    header = _imports_header(problem["prompt"])
    entry = problem["entry_point"]
    return (
        f"{header}\n\n"
        f"{code}\n\n"
        f"{problem['test']}\n\n"
        f"check({entry})\n"
    )


def _build_program_mbpp(problem: dict, code: str) -> str:
    """Assembla il programma di verifica per un problema MBPP.

    MBPP è strutturato diversamente da HumanEval:
      - niente `prompt` con import né `entry_point`/funzione `check`;
      - `test_setup_code` (può essere vuoto) va anteposto: a volte definisce
        import o strutture di supporto richiesti dagli assert;
      - `test_list` è una LISTA di `assert` già auto-contenuti: li mettiamo in
        coda, uno per riga. Se uno fallisce solleva AssertionError, esattamente
        come per HumanEval → stessa classificazione in _classify_stderr.

    Nota: usiamo SOLO `code` (il codice del modello) + i test. Il riferimento
    per il CodeBLEU resta il campo `code` originale del dataset, calcolato
    altrove (metrics.py): qui non c'entra e i test NON vi finiscono dentro.
    """
    setup = problem.get("test_setup_code", "") or ""
    asserts = "\n".join(problem.get("test_list", []))
    return (
        f"{setup}\n\n"
        f"{code}\n\n"
        f"{asserts}\n"
    )


def _is_mbpp(problem: dict) -> bool:
    """MBPP ha `test_list` (lista di assert); HumanEval ha `test`/`entry_point`."""
    return "test_list" in problem


def _is_ds1000(problem: dict) -> bool:
    """DS-1000 ha `code_context` (l'harness di test con [insert])."""
    return "code_context" in problem


def _ds1000_insert_indent(code_context: str) -> str | None:
    """Indentazione richiesta dalla soluzione nel formato 'Insertion' di DS-1000.

    Se nell'harness `[insert]` sta DENTRO un blocco — la riga che lo precede
    termina con ':' (tipicamente `def f(df):`) — la soluzione deve essere il corpo
    indentato di quel blocco: ritorna l'indentazione da applicare (indent della
    riga di apertura + 4 spazi). Altrimenti `None`: inserimento a livello di
    modulo (formato 'Completion'), nessuna reindentazione."""
    i = code_context.find("[insert]")
    if i == -1:
        return None
    before = code_context[:i].splitlines()
    prev = before[-1] if before else ""
    if prev.strip().endswith(":"):
        return " " * ((len(prev) - len(prev.lstrip())) + 4)
    return None


def _build_program_ds1000(problem: dict, code: str) -> str:
    """Assembla il programma di verifica per un problema DS-1000.

    Il `code_context` è già un programma completo che DEFINISCE `test_execution(solution)`
    (ed eventualmente `test_string(solution)`) ma non lo chiama. La soluzione del
    modello va passata come STRINGA: test_execution la inserisce al posto di
    `[insert]` nel proprio template ed esegue, sollevando AssertionError se sbagliata.

    Quindi: code_context + chiamata a test_execution(<soluzione>). Passiamo la
    soluzione con repr() per ottenere un literal Python sicuro (gestisce a capo,
    apici, ecc.). Per i problemi Matplotlib forziamo il backend non interattivo
    'Agg', così non si tenta di aprire finestre in un processo headless.

    Formato 'Insertion': quando `[insert]` è il corpo di un `def`, NON ci affidiamo
    al modello per indentare (lo fa in modo incoerente: prima riga a colonna 0,
    resto indentato → IndentationError). Reindentiamo noi: `dedent` normalizza un
    eventuale rientro uniforme, `indent` applica quello richiesto preservando la
    struttura relativa (anche con cicli annidati). Il prompt chiede infatti il
    corpo a colonna 0."""
    cc = problem["code_context"]
    header = ""
    if problem.get("metadata", {}).get("library") == "Matplotlib":
        header = "import matplotlib\nmatplotlib.use('Agg')\n\n"

    indent = _ds1000_insert_indent(cc)
    if indent is not None and code.strip():
        code = textwrap.indent(textwrap.dedent(code), indent)

    parts = [header + cc, "", f"solution = {code!r}", "test_execution(solution)"]
    if "def test_string" in cc:
        parts.append("test_string(solution)")
    return "\n".join(parts)


def _is_plot2code(problem: dict) -> bool:
    """Plot2Code ha `instruction` (descrizione della figura) e `ref_image`."""
    return "instruction" in problem


# ----------------------------------------------------------------------------
# MultiPL-E (multilinguaggio)
# ----------------------------------------------------------------------------

# Registro dei linguaggi INTERPRETATI: estensione del file, eseguibile, eventuali
# argomenti da anteporre al file e percorsi-fallback se non è nel PATH. Hanno un
# handler DEDICATO a parte (compilati o con toolchain): java, cpp, rust, go, c#,
# dart (vedi _run_*). I restanti del set HumanEval di MultiPL-E (swift, scala, hs,
# ml, elixir, clj, rkt, d, adb) NON sono ancora gestiti → RuntimeMissing finché non
# si aggiunge una regola/handler. Per il pass@1 serve comunque il runtime nel PATH;
# se manca, l'executor segnala RuntimeMissing e prosegue.
#   Nota R: l'installer di R su Windows NON mette Rscript nel PATH → fallback glob.
#
# Cartella bin di MSYS2 (default winget): fornisce `bc` (assente in Git for Windows),
# usato dai problemi bash di HumanEval per i float. Aggiunta al PATH del subprocess
# `sh` da _interp_env (dopo la cartella di Git). Vedi quel commento per i dettagli.
_MSYS2_BIN = r"C:\msys64\usr\bin"

_MULTIPLE_INTERP = {
    "js":  {"ext": ".js",  "interp": "node"},
    # TypeScript: i test MultiPL-E usano `require('node:assert')` (API di Node, non
    # di Deno). Lo eseguiamo con Node: estensione .cts (CommonJS, così `require`
    # esiste) e type-stripping (Node ≥23.6 lo fa di default; il flag lo abilita
    # anche su 22/23 ed è accettato su 24).
    "ts":  {"ext": ".cts", "interp": "node", "pre_args": ["--experimental-strip-types"]},
    "php": {"ext": ".php", "interp": "php"},
    "rb":  {"ext": ".rb",  "interp": "ruby"},
    # Perl/Bash arrivano da Git for Windows (usr\bin), che spesso NON è sul PATH di
    # PowerShell → fallback al percorso di Git. Per `bash` il PATH può risolvere
    # PRIMA il bash di WSL (System32\bash.exe), che fallisce se non c'è una distro:
    # con prefer_fallbacks usiamo direttamente il bash di Git.
    # Perl: preferiamo Strawberry Perl (Perl completo per Windows, con cpan e i
    # moduli CPAN come Test::Deep richiesto dai test) al perl MINIMALE di Git, che
    # non ha Test::Deep né cpan. prefer_fallbacks lo mette davanti al PATH.
    "pl":  {"ext": ".pl",  "interp": "perl", "prefer_fallbacks": True,
            "fallbacks": [r"C:\Strawberry\perl\bin\perl.exe",
                          r"C:\Program Files\Git\usr\bin\perl.exe"]},
    "lua": {"ext": ".lua", "interp": "lua",
            "fallbacks": [r"C:\Users\*\AppData\Local\Programs\Lua\bin\lua.exe"]},
    # Julia: l'installer winget va in %LOCALAPPDATA%\Programs\Julia-<ver>\bin, spesso
    # NON nel PATH di PowerShell → fallback glob (qualsiasi versione installata).
    "jl":  {"ext": ".jl",  "interp": "julia",
            "fallbacks": [r"C:\Users\*\AppData\Local\Programs\Julia-*\bin\julia.exe"]},
    "sh":  {"ext": ".sh",  "interp": "bash", "prefer_fallbacks": True,
            "fallbacks": [r"C:\Program Files\Git\usr\bin\bash.exe"]},
    "r":   {"ext": ".r",   "interp": "Rscript",
            "fallbacks": [r"C:\Program Files\R\R-*\bin\Rscript.exe",
                          r"C:\Program Files\R\R-*\bin\x64\Rscript.exe"]},
}

# Cartella con i jar di supporto per Java (es. javatuples, importato da molti
# problemi humaneval-java). Tutti i .jar qui finiscono nel classpath.
_LIB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib")


def _resolve_exe(name: str, fallbacks=(), prefer_fallbacks: bool = False) -> str | None:
    """Percorso dell'eseguibile: dal PATH (shutil.which) e dai fallback (glob,
    versione più recente). Di norma il PATH ha precedenza; con prefer_fallbacks i
    fallback vengono provati PRIMA (serve per `bash`, dove il PATH può risolvere il
    bash di WSL invece di quello di Git). None se introvabile."""
    def _from_fallbacks():
        for pattern in fallbacks:
            hits = sorted(glob.glob(pattern))
            if hits:
                return hits[-1]
        return None

    if prefer_fallbacks:
        return _from_fallbacks() or shutil.which(name)
    return shutil.which(name) or _from_fallbacks()


def _is_multipl_e(problem: dict) -> bool:
    """I problemi MultiPL-E hanno `stop_tokens` (e `language`/`tests`)."""
    return "stop_tokens" in problem


def _strip_prompt_echo(prompt: str, code: str) -> str:
    """Modalità completamento: il programma è `prompt + corpo + tests`. Se il
    modello, invece del solo corpo, ha RIPETUTO la firma (e magari il preambolo:
    `<?php`, gli import, la riga `def/function ...`), va rimosso l'eco per non
    duplicare la dichiarazione.

    Euristica: cerco nell'output del modello la riga che corrisponde (a meno di
    spazi) all'ULTIMA riga non vuota del prompt — cioè la firma aperta — e tengo
    solo ciò che viene DOPO. Se non la trovo, l'output è già il solo corpo e lo
    restituisco invariato."""
    norm = lambda s: re.sub(r"\s+", "", s)
    plines = [ln for ln in prompt.splitlines() if ln.strip()]
    if not plines or not code.strip():
        return code
    sig = norm(plines[-1])
    if not sig:
        return code
    clines = code.splitlines()
    for i in range(len(clines) - 1, -1, -1):          # ultima occorrenza
        if sig in norm(clines[i]):
            return "\n".join(clines[i + 1:])
    return code


def _balance_braces_body(body: str) -> str:
    """Per i linguaggi C-like in cui i `tests` iniziano con `}` (Java/Rust/C#/Dart):
    i tests CHIUDONO la funzione/metodo/classe aperti dal `prompt`, quindi il corpo
    generato deve restare BILANCIATO (non deve richiuderli da sé). Claude però
    spesso aggiunge 1-2 `}` di troppo: le rimuoviamo finché `{` e `}` nel corpo non
    pareggiano, così `prompt`(apre) + corpo(bilanciato) + `tests`(chiude) risulta
    corretto. Conteggio naïve (ok per questi snippet brevi)."""
    closes = body.count("}")
    opens = body.count("{")
    while closes > opens:
        idx = body.rfind("}")
        if idx == -1:
            break
        body = body[:idx] + body[idx + 1:]
        closes -= 1
    return body.rstrip()


# Linguaggi i cui `tests` iniziano con `}` → il corpo va lasciato bilanciato (i
# test chiudono i blocchi/funzioni aperti dal prompt). Include bash (sh): il prompt
# apre `f() {` e i tests iniziano con `}` che chiude la funzione; nei corpi bash i
# `${...}` sono bilanciati, quindi il conteggio rimuove solo la `}` finale di
# troppo. Negli altri (js, go, php, …) il corpo chiude da sé i propri blocchi.
_BRACE_BALANCED_LANGS = {"java", "rs", "cs", "dart", "cpp", "sh"}


def _build_program_multipl_e(problem: dict, code: str) -> str:
    """Assembla il programma MultiPL-E: `prompt` (firma aperta) + corpo generato
    (ripulito dall'eventuale eco della firma; bilanciato per java/rust/c#/dart) +
    `tests` (che chiudono la funzione e aggiungono le asserzioni)."""
    body = _strip_prompt_echo(problem.get("prompt", ""), code)
    if problem.get("language") in _BRACE_BALANCED_LANGS:
        body = _balance_braces_body(body)
    prompt = (problem.get("prompt", "") or "").rstrip()
    tests = problem.get("tests", "") or ""
    return f"{prompt}\n{body}\n\n{tests}\n"


def _classify_multipl_e(stderr: str, compile_error: bool = False) -> str:
    """Categoria d'errore per i linguaggi non-Python (no traceback Python): deduce
    il TIPO dal messaggio d'errore, con firme comuni a js/ts/php/r/perl/lua/bash.
    Controlli dal più specifico al generico; fallback RuntimeError. `compile_error`
    (compilati: java/cpp/…) forza SyntaxError."""
    if compile_error:
        return "SyntaxError"
    low = (stderr or "").lower()
    # Uscita ≠ 0 SENZA alcun messaggio = TEST FALLITO (risposta sbagliata), non un
    # crash di linguaggio. Gli harness MultiPL-E segnalano il fallimento col solo
    # exit code: bash usa `set -e` + `[[ … = … ]]`, perl Test::More esce ≠ 0. Un
    # vero errore di runtime stamperebbe SEMPRE una diagnostica; l'assenza totale di
    # output è quindi un assert fallito → AssertionError (altrimenti finirebbe nel
    # catch-all RuntimeError, sovrastimando i "crash" rispetto agli errori di logica).
    if not low.strip():
        return "AssertionError"
    # test fallito (asserzione) — "logica sbagliata", non un crash di linguaggio
    if any(k in low for k in ("assertionerror", "assertion failed", "test failed",
                              "stopifnot", "panicked", "\nnot ok", "not ok ")):
        return "AssertionError"
    # errore di sintassi / parsing (codice non valido nel linguaggio)
    if any(k in low for k in ("syntaxerror", "syntax error", "parse error",
                              "parseerror", "unexpected token", "unexpected symbol",
                              "unexpected end", "near unexpected token", "malformed",
                              "expected ", "unexpected '", "unterminated")):
        return "SyntaxError"
    # modulo/libreria non trovata
    if any(k in low for k in ("can't locate", "cannot find module", "module not found",
                              "there is no package", "no such module")):
        return "ImportError"
    # nome/riferimento non definito (variabile o funzione)
    if any(k in low for k in ("is not defined", "referenceerror", "undefined variable",
                              "nil value", "command not found", "could not find function",
                              "object '", "use of undeclared", "undefined subroutine")):
        return "NameError"
    # tipo non compatibile
    if any(k in low for k in ("typeerror", "is not a function", "cannot read propert",
                              "non-numeric argument", "not callable")):
        return "TypeError"
    # indice fuori range
    if any(k in low for k in ("index out of", "out of range", "subscript out of bounds",
                              "indexerror")):
        return "IndexError"
    # valore non valido / divisione per zero
    if any(k in low for k in ("valueerror", "division by zero", "divide by zero",
                              "/ 0")):
        return "ValueError"
    return "RuntimeError"


def _run_java(problem: dict, code: str, timeout: float) -> dict:
    """Compila ed esegue un problema MultiPL-E Java. La classe si chiama `Problem`
    e i `tests` forniscono la chiusura + il main. Si esegue con `java -ea` (i test
    usano assert); i jar di lib/ (javatuples & co.) finiscono nel classpath. Se
    manca il JDK -> RuntimeMissing."""
    javac = _resolve_exe("javac")
    java = _resolve_exe("java")
    if javac is None or java is None:
        return {"passed": False, "category": "RuntimeMissing",
                "stderr": "JDK non installato (servono javac e java nel PATH)"}
    jars = glob.glob(os.path.join(_LIB_DIR, "*.jar"))
    program = _build_program_multipl_e(problem, code)
    try:
        with tempfile.TemporaryDirectory() as wd:
            cp = os.pathsep.join([wd] + jars)
            src = os.path.join(wd, "Problem.java")
            with open(src, "w", encoding="utf-8") as f:
                f.write(program)
            comp = _run_guarded([javac, "-cp", cp, src], capture_output=True,
                                  text=True, timeout=timeout, cwd=wd)
            if comp.returncode != 0:
                return {"passed": False, "category": "SyntaxError",
                        "stderr": comp.stderr[-2000:]}
            run = _run_guarded([java, "-ea", "-cp", cp, "Problem"],
                                 capture_output=True, text=True, timeout=timeout, cwd=wd)
    except subprocess.TimeoutExpired:
        return {"passed": False, "category": "TimeoutError", "stderr": "timeout"}
    if run.returncode == 0:
        return {"passed": True, "category": "Passed", "stderr": ""}
    out = (run.stderr or "") + "\n" + (run.stdout or "")
    return {"passed": False, "category": _classify_multipl_e(out), "stderr": out[-2000:]}


def _run_cpp(problem: dict, code: str, timeout: float) -> dict:
    """Compila (g++ -std=c++17) ed esegue un problema MultiPL-E C++. Un test
    fallito aborta con uscita ≠ 0. Se manca un compilatore C++ -> RuntimeMissing."""
    gpp = _resolve_exe("g++") or _resolve_exe("clang++") or _resolve_exe("c++")
    if gpp is None:
        return {"passed": False, "category": "RuntimeMissing",
                "stderr": "compilatore C++ non installato (g++/clang++ nel PATH)"}
    program = _build_program_multipl_e(problem, code)
    try:
        with tempfile.TemporaryDirectory() as wd:
            src = os.path.join(wd, "program.cpp")
            exe = os.path.join(wd, "prog.exe")
            with open(src, "w", encoding="utf-8") as f:
                f.write(program)
            comp = _run_guarded([gpp, "-std=c++17", src, "-o", exe],
                                  capture_output=True, text=True, timeout=timeout, cwd=wd)
            if comp.returncode != 0:
                return {"passed": False, "category": "SyntaxError",
                        "stderr": comp.stderr[-2000:]}
            run = _run_guarded([exe], capture_output=True, text=True,
                                 timeout=timeout, cwd=wd)
    except subprocess.TimeoutExpired:
        return {"passed": False, "category": "TimeoutError", "stderr": "timeout"}
    if run.returncode == 0:
        return {"passed": True, "category": "Passed", "stderr": ""}
    out = (run.stderr or "") + "\n" + (run.stdout or "")
    return {"passed": False, "category": _classify_multipl_e(out), "stderr": out[-2000:]}


def _run_rust(problem: dict, code: str, timeout: float) -> dict:
    """Compila (rustc) ed esegue un problema MultiPL-E Rust. Il `test` aggiunge
    `fn main(){ assert_eq!(...) }`: un caso fallito fa panic → uscita ≠ 0. Rust ha
    bisogno di un linker (la toolchain GNU lo include). rustc assente -> RuntimeMissing."""
    rustc = _resolve_exe("rustc")
    if rustc is None:
        return {"passed": False, "category": "RuntimeMissing",
                "stderr": "rustc non installato nel PATH"}
    program = _build_program_multipl_e(problem, code)
    try:
        with tempfile.TemporaryDirectory() as wd:
            src = os.path.join(wd, "prog.rs")
            exe = os.path.join(wd, "prog.exe")
            with open(src, "w", encoding="utf-8") as f:
                f.write(program)
            comp = _run_guarded([rustc, "-A", "warnings", src, "-o", exe],
                                  capture_output=True, text=True, timeout=timeout, cwd=wd)
            if comp.returncode != 0:
                return {"passed": False, "category": "SyntaxError",
                        "stderr": comp.stderr[-2000:]}
            run = _run_guarded([exe], capture_output=True, text=True,
                                 timeout=timeout, cwd=wd)
    except subprocess.TimeoutExpired:
        return {"passed": False, "category": "TimeoutError", "stderr": "timeout"}
    if run.returncode == 0:
        return {"passed": True, "category": "Passed", "stderr": ""}
    out = (run.stderr or "") + "\n" + (run.stdout or "")
    return {"passed": False, "category": _classify_multipl_e(out), "stderr": out[-2000:]}


def _run_dart(problem: dict, code: str, timeout: float) -> dict:
    """Esegue un problema MultiPL-E Dart con `dart run`. Il `test` definisce un
    `expect` (lancia un'eccezione se un caso non torna) e un `main()`: un fallimento
    → eccezione non gestita → uscita ≠ 0. `dart` assente -> RuntimeMissing."""
    dart = _resolve_exe("dart")
    if dart is None:
        return {"passed": False, "category": "RuntimeMissing",
                "stderr": "dart non installato nel PATH"}
    program = _build_program_multipl_e(problem, code)
    try:
        with tempfile.TemporaryDirectory() as wd:
            src = os.path.join(wd, "program.dart")
            with open(src, "w", encoding="utf-8") as f:
                f.write(program)
            proc = _run_guarded([dart, "run", src], capture_output=True,
                                  text=True, timeout=timeout, cwd=wd)
    except subprocess.TimeoutExpired:
        return {"passed": False, "category": "TimeoutError", "stderr": "timeout"}
    if proc.returncode == 0:
        return {"passed": True, "category": "Passed", "stderr": ""}
    out = (proc.stderr or "") + "\n" + (proc.stdout or "")
    return {"passed": False, "category": _classify_multipl_e(out), "stderr": out[-2000:]}


# Cache di compilazione persistente per Go (condivisa tra problemi): senza, ogni
# `go test` ricompila la stdlib da zero (lentissimo). Sta nella temp di sistema.
_GO_CACHE = os.path.join(tempfile.gettempdir(), "mpe_go_cache")


def _run_go(problem: dict, code: str, timeout: float) -> dict:
    """Esegue un problema MultiPL-E Go con `go test`. Il `test` è una funzione
    `TestXxx(t *testing.T)` nel package `<nome>_test` → serve `go test` (non `go
    run`). Scriviamo il programma come `prog_test.go`, uno stub non-test nello
    stesso package (go test pretende almeno un file non-test) e un go.mod minimale.
    pass = `go test` esce 0. `go` assente -> RuntimeMissing."""
    go = _resolve_exe("go")
    if go is None:
        return {"passed": False, "category": "RuntimeMissing",
                "stderr": "go non installato nel PATH"}
    program = _build_program_multipl_e(problem, code)
    m = re.search(r"^package\s+(\w+)", program, re.MULTILINE)
    pkg = m.group(1) if m else "main"
    # Il programma è in un file `_test.go` con package `<base>_test`: è l'EXTERNAL
    # test package del package base `<base>`. Lo stub non-test deve quindi dichiarare
    # il package BASE (senza il suffisso `_test`), altrimenti go test vede due
    # package diversi nella stessa cartella e fallisce il setup.
    base = pkg[:-5] if pkg.endswith("_test") else pkg
    env = os.environ.copy()
    env["GOCACHE"] = _GO_CACHE
    env["GOTOOLCHAIN"] = "local"      # niente download di toolchain
    env["GOFLAGS"] = "-mod=mod"
    try:
        with tempfile.TemporaryDirectory() as wd:
            with open(os.path.join(wd, "go.mod"), "w", encoding="utf-8") as f:
                f.write("module prob\n\ngo 1.21\n")
            with open(os.path.join(wd, "stub.go"), "w", encoding="utf-8") as f:
                f.write(f"package {base}\n")
            with open(os.path.join(wd, "prog_test.go"), "w", encoding="utf-8") as f:
                f.write(program)
            run = _run_guarded([go, "test"], capture_output=True, text=True,
                                 timeout=timeout, cwd=wd, env=env)
    except subprocess.TimeoutExpired:
        return {"passed": False, "category": "TimeoutError", "stderr": "timeout"}
    if run.returncode == 0:
        return {"passed": True, "category": "Passed", "stderr": ""}
    out = (run.stdout or "") + "\n" + (run.stderr or "")
    low = out.lower()
    # errore di build (sintassi/tipi) vs test fallito. NB: i messaggi dei test
    # contengono "expected"/".go:" → non vanno usati per dedurre un SyntaxError.
    if "build failed" in low or "setup failed" in low:
        cat = "SyntaxError"
    elif "--- fail" in low:
        cat = "AssertionError"
    else:
        cat = "RuntimeError"
    return {"passed": False, "category": cat, "stderr": out[-2000:]}


def _run_cs(problem: dict, code: str, timeout: float) -> dict:
    """Esegue un problema MultiPL-E C# col .NET SDK. Il `test` è un `Main` con
    `Debug.Assert(...)` (attive in build Debug). ⚠️ Debug.Assert non sempre termina
    con uscita ≠ 0: rileviamo il fallimento anche dal testo ("assert"/"fail")
    nell'output. Crea un progetto console minimale e fa `dotnet run`. ⚠️ il primo
    problema può essere lento (restore NuGet). `dotnet` assente -> RuntimeMissing."""
    dotnet = _resolve_exe("dotnet")
    if dotnet is None:
        return {"passed": False, "category": "RuntimeMissing",
                "stderr": "dotnet (.NET SDK) non installato nel PATH"}
    program = _build_program_multipl_e(problem, code)
    csproj = ('<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup>'
              '<OutputType>Exe</OutputType><TargetFramework>net8.0</TargetFramework>'
              '<Nullable>disable</Nullable><ImplicitUsings>disable</ImplicitUsings>'
              '<AssemblyName>prob</AssemblyName></PropertyGroup></Project>')
    env = os.environ.copy()
    env["DOTNET_CLI_TELEMETRY_OPTOUT"] = "1"
    env["DOTNET_NOLOGO"] = "1"
    try:
        with tempfile.TemporaryDirectory() as wd:
            with open(os.path.join(wd, "prob.csproj"), "w", encoding="utf-8") as f:
                f.write(csproj)
            with open(os.path.join(wd, "Program.cs"), "w", encoding="utf-8") as f:
                f.write(program)
            run = _run_guarded([dotnet, "run", "-c", "Debug", "--project", wd],
                                 capture_output=True, text=True, timeout=timeout, cwd=wd, env=env)
    except subprocess.TimeoutExpired:
        return {"passed": False, "category": "TimeoutError", "stderr": "timeout"}
    out = (run.stdout or "") + "\n" + (run.stderr or "")
    low = out.lower()
    if "error cs" in low or "build failed" in low or "msb" in low and "error" in low:
        return {"passed": False, "category": "SyntaxError", "stderr": out[-2000:]}
    # Debug.Assert può fallire senza uscita ≠ 0: controlliamo anche il testo.
    if run.returncode == 0 and not ("assert" in low and "fail" in low):
        return {"passed": True, "category": "Passed", "stderr": ""}
    return {"passed": False, "category": "AssertionError", "stderr": out[-2000:]}


def multipl_e_runnable(lang: str) -> bool:
    """Vero se il linguaggio MultiPL-E è ESEGUIBILE su questa macchina ORA: ha un
    handler (compilato dedicato o nel registro interpretato) E il runtime/compilatore
    è risolvibile (PATH o fallback). Serve a NON generare (e non pagare le API) per i
    linguaggi che darebbero comunque `RuntimeMissing` (vedi pipeline/plan_run)."""
    if lang == "java":
        return _resolve_exe("javac") is not None and _resolve_exe("java") is not None
    if lang == "cpp":
        return (_resolve_exe("g++") or _resolve_exe("clang++") or _resolve_exe("c++")) is not None
    if lang == "rs":
        return _resolve_exe("rustc") is not None
    if lang == "go":
        return _resolve_exe("go") is not None
    if lang == "dart":
        return _resolve_exe("dart") is not None
    if lang == "cs":
        return _resolve_exe("dotnet") is not None
    spec = _MULTIPLE_INTERP.get(lang)
    if spec is None:
        return False
    return _resolve_exe(spec["interp"], spec.get("fallbacks", ()),
                        prefer_fallbacks=spec.get("prefer_fallbacks", False)) is not None


def _lua_env() -> dict:
    """Ambiente per Lua con LUA_PATH/LUA_CPATH verso i moduli installati da luarocks
    (es. `luaunit`, richiesto dai test MultiPL-E). luarocks installa nel tree utente
    `~/.luarocks`, che il `lua` di default NON cerca: senza questi path
    `require('luaunit')` fallisce."""
    env = os.environ.copy()
    rocks = os.path.join(os.path.expanduser("~"), ".luarocks")
    vers = sorted(os.path.basename(d) for d in
                  glob.glob(os.path.join(rocks, "share", "lua", "*")))
    if vers:
        ver = vers[-1]
        lp = os.path.join(rocks, "share", "lua", ver).replace("\\", "/")
        cp = os.path.join(rocks, "lib", "lua", ver).replace("\\", "/")
        env["LUA_PATH"] = f"{lp}/?.lua;{lp}/?/init.lua;;"
        env["LUA_CPATH"] = f"{cp}/?.dll;;"
    return env


def _interp_env(lang: str, interp: str) -> dict | None:
    """Ambiente del subprocess per gli interpretati MultiPL-E (None = eredita os.environ).

    - lua: LUA_PATH/LUA_CPATH per luaunit (vedi _lua_env).
    - sh (Git-bash): i problemi HumanEval in bash usano le coreutils Unix
      (tr/sed/awk/grep/seq/fold…), che NON sono nel PATH di Windows ma vivono nella
      STESSA cartella del bash di Git (C:\\Program Files\\Git\\usr\\bin). Senza questo
      prepend al PATH danno tutte "command not found" → falsi negativi d'ambiente
      (NON colpa del modello). `bc` (calcolatrice per i float) NON è spedito con Git
      for Windows: lo prendiamo da MSYS2 (C:\\msys64\\usr\\bin, installato via
      `winget MSYS2.MSYS2` + `pacman -S bc`), aggiunto DOPO la cartella di Git così
      le coreutils restano quelle di Git già validate e da MSYS2 si pesca solo `bc`.
      bc.exe trova il suo msys-2.0.dll nella propria cartella (DLL search)."""
    if lang == "lua":
        return _lua_env()
    if lang == "sh":
        env = os.environ.copy()
        parts = [os.path.dirname(interp)]          # Git\usr\bin: coreutils validate
        if os.path.isdir(_MSYS2_BIN):              # MSYS2: fornisce `bc` (assente in Git)
            parts.append(_MSYS2_BIN)
        env["PATH"] = os.pathsep.join(parts + [env.get("PATH", "")])
        return env
    return None


def _run_multipl_e(problem: dict, code: str, timeout: float) -> dict:
    """Esegue un problema MultiPL-E nel suo linguaggio. pass = uscita 0.
    Compilati/con-toolchain: java/cpp/rust/go/c#/dart gestiti a parte. Interpretati:
    dal registro `_MULTIPLE_INTERP`. Runtime non installato o linguaggio non
    configurato -> RuntimeMissing (non è colpa del modello)."""
    lang = problem.get("language", "")
    if lang == "java":
        return _run_java(problem, code, timeout)
    if lang == "cpp":
        return _run_cpp(problem, code, timeout)
    if lang == "rs":
        return _run_rust(problem, code, timeout)
    if lang == "go":
        return _run_go(problem, code, timeout)
    if lang == "dart":
        return _run_dart(problem, code, timeout)
    if lang == "cs":
        return _run_cs(problem, code, timeout)

    spec = _MULTIPLE_INTERP.get(lang)
    if spec is None:
        return {"passed": False, "category": "RuntimeMissing",
                "stderr": f"linguaggio non configurato nell'executor: {lang!r}"}
    interp = _resolve_exe(spec["interp"], spec.get("fallbacks", ()),
                          prefer_fallbacks=spec.get("prefer_fallbacks", False))
    if interp is None:
        return {"passed": False, "category": "RuntimeMissing",
                "stderr": f"runtime '{spec['interp']}' non installato nel PATH"}

    program = _build_program_multipl_e(problem, code)
    env = _interp_env(lang, interp)   # lua: LUA_PATH per luaunit; sh: usr\bin di Git nel PATH
    try:
        with tempfile.TemporaryDirectory() as wd:
            fp = os.path.join(wd, "program" + spec["ext"])
            with open(fp, "w", encoding="utf-8") as f:
                f.write(program)
            cmd = [interp] + list(spec.get("pre_args", [])) + [fp]
            proc = _run_guarded(cmd, capture_output=True,
                                  text=True, timeout=timeout, cwd=wd, env=env)
    except subprocess.TimeoutExpired:
        return {"passed": False, "category": "TimeoutError", "stderr": "timeout"}
    if proc.returncode == 0:
        return {"passed": True, "category": "Passed", "stderr": ""}
    out = (proc.stderr or "") + "\n" + (proc.stdout or "")  # il messaggio può finire su stdout (es. PHP)
    return {"passed": False, "category": _classify_multipl_e(out), "stderr": out[-2000:]}


def _build_program_plot2code(code: str, out_png: str) -> str:
    """Programma per Plot2Code: esegue lo script generato e salva la figura.

    Lo script generato disegna ma di solito fa `plt.show()` (non salva). Noi:
      - forziamo il backend non interattivo via MPLBACKEND=Agg (impostato
        nell'ambiente del subprocess, vedi run_one): robusto anche se lo script
        importa matplotlib prima di qualunque nostra riga;
      - in coda salviamo su `out_png` la figura prodotta. Salviamo l'ULTIMA figura
        aperta (le gallery hanno una figura per esempio, eventualmente con più
        subplot) a dpi=100 — le immagini di riferimento sono 640×480 = figsize
        di default × 100 dpi. Se non c'è alcuna figura, non scriviamo nulla:
        run_one lo classifica come `NoFigure`."""
    saver = (
        "\nimport matplotlib.pyplot as _plt\n"
        "_nums = _plt.get_fignums()\n"
        "if _nums:\n"
        f"    _plt.figure(_nums[-1]).savefig(r{out_png!r}, dpi=100)\n"
    )
    return code + "\n" + saver


def _classify_stderr(stderr: str) -> str:
    """Deduce la categoria di errore dall'ultima riga del traceback."""
    lines = [ln.rstrip() for ln in stderr.splitlines() if ln.strip()]
    # Scorri dal fondo: la prima riga che è "NomeErrore[: ...]" è l'eccezione finale.
    for line in reversed(lines):
        match = _EXC_LINE_RE.match(line)
        if match:
            exc = match.group(1).split(".")[-1]  # scarta eventuale prefisso di modulo
            if exc == "ModuleNotFoundError":     # sottoclasse di ImportError
                return "ImportError"
            return exc if exc in _KNOWN else "RuntimeError"
    return "RuntimeError"


def run_one(problem: dict, code: str, timeout: float = 10.0,
            render_to: str | None = None) -> dict:
    """
    Esegue una soluzione e restituisce:
      { "passed": bool, "category": str, "stderr": str }

    Plot2Code: invece di assert, la "correttezza funzionale" è che il codice GIRA
    e produce una figura. `render_to` è il path dove salvare il PNG generato (per
    il confronto visivo successivo); pass = exit 0 + PNG non vuoto. Negli altri
    benchmark `render_to` è ignorato.
    """
    if not code.strip():
        return {"passed": False, "category": "EmptyOutput", "stderr": ""}

    # MultiPL-E: multilinguaggio, esecuzione e classificazione dedicate per lingua.
    if _is_multipl_e(problem):
        return _run_multipl_e(problem, code, timeout)

    is_p2c = _is_plot2code(problem)
    if is_p2c:
        out_png = render_to or os.path.join(tempfile.gettempdir(), "p2c_out.png")
        program = _build_program_plot2code(code, out_png)
    elif _is_ds1000(problem):
        program = _build_program_ds1000(problem, code)
    elif _is_mbpp(problem):
        program = _build_program_mbpp(problem, code)
    else:
        program = _build_program(problem, code)

    # Backend matplotlib non interattivo per Plot2Code: via env, così vale anche
    # se lo script importa matplotlib prima di qualunque nostra riga.
    env = os.environ.copy()
    if is_p2c:
        env["MPLBACKEND"] = "Agg"
        if render_to:
            os.makedirs(os.path.dirname(render_to), exist_ok=True)

    # Eseguiamo in una cartella temporanea ISOLATA usata come cwd: se la soluzione
    # scrive file (es. df.to_csv(...), torch.save(..., 'my_model.pt'), pickle.dump),
    # questi restano confinati nella tempdir e vengono rimossi a fine esecuzione,
    # invece di sporcare la working directory del progetto. È anche più sicuro: il
    # codice non fidato gira in una sandbox di filesystem usa-e-getta.
    try:
        with tempfile.TemporaryDirectory() as workdir:
            script_path = os.path.join(workdir, "program.py")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(program)
            proc = _run_guarded(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=workdir,
                env=env,
            )
    except subprocess.TimeoutExpired:
        return {"passed": False, "category": "TimeoutError", "stderr": "timeout"}

    if is_p2c:
        # Pass solo se l'esecuzione è andata a buon fine E ha prodotto un PNG.
        if proc.returncode != 0:
            return {"passed": False, "category": _classify_stderr(proc.stderr),
                    "stderr": proc.stderr[-2000:]}
        produced = bool(render_to) and os.path.exists(render_to) and os.path.getsize(render_to) > 0
        if produced:
            return {"passed": True, "category": "Passed", "stderr": ""}
        return {"passed": False, "category": "NoFigure",
                "stderr": (proc.stderr or "il codice è stato eseguito ma non ha "
                           "prodotto alcuna figura")[-2000:]}

    if proc.returncode == 0:
        return {"passed": True, "category": "Passed", "stderr": ""}

    return {
        "passed": False,
        "category": _classify_stderr(proc.stderr),
        "stderr": proc.stderr[-2000:],  # tieni solo la coda, utile per debug
    }
