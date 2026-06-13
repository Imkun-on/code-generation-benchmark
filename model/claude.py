"""
claude.py — Entrypoint standalone: `python -m model.claude`.

Equivalente a `python cli.py`: delega alla **stessa** CLI Typer definita in
`cli.py` (così i due entrypoint accettano esattamente gli stessi argomenti e
mostrano lo stesso `--help` formattato con Rich). La definizione dei modelli e la
chiamata alle API vivono in `model/providers.py`.

Esempio:
    python -m model.claude --benchmark ds1000 --limit 5
    python -m model.claude -b multipl-e -n 3
"""


def standalone_main() -> None:
    """Avvia la CLI Typer condivisa (vedi cli.py)."""
    # `cli` è il modulo a livello di progetto (la root è sul sys.path quando si
    # lancia `python -m model.claude`). Importarlo definisce l'app Typer senza
    # eseguirla; qui la eseguiamo.
    from cli import app
    app()


if __name__ == "__main__":
    standalone_main()
