"""
claude.py — Standalone entrypoint: `python -m model.claude`.

Equivalent to `python cli.py`: it delegates to the **same** Typer CLI defined in
`cli.py` (so the two entrypoints accept exactly the same arguments and show the
same Rich-formatted `--help`). The model definitions and the API calls live in
`model/providers.py`.

Example:
    python -m model.claude --benchmark ds1000 --limit 5
    python -m model.claude -b multipl-e -n 3
"""


def standalone_main() -> None:
    """Launch the shared Typer CLI (see cli.py).

    Provided so the project can also be run as a module (`python -m model.claude`)
    in addition to `python cli.py`; both paths end up running the same app."""
    # `cli` è il modulo a livello di progetto (la root è sul sys.path quando si
    # lancia `python -m model.claude`). Importarlo definisce l'app Typer senza
    # eseguirla; qui la eseguiamo.
    from cli import app
    app()


if __name__ == "__main__":
    standalone_main()
