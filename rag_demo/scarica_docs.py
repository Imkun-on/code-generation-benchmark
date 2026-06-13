"""
scarica_docs.py — Scarica pagine web (es. la documentazione Python), estrae
SOLO il testo principale (niente menu/HTML) e lo salva in rag_demo/documenti/
come file .txt, pronti per essere indicizzati da rag_locale.py.

Esegui con:  python rag_demo/scarica_docs.py
"""

import time
from pathlib import Path

import trafilatura

# --------------------------------------------------------------------------
# Le pagine da scaricare. Mettine quante vuoi: qui qualche pagina del
# tutorial ufficiale di Python. Cambiale con gli URL che ti interessano.
# --------------------------------------------------------------------------
URLS = [
    "https://docs.python.org/3/tutorial/introduction.html",
    "https://docs.python.org/3/tutorial/datastructures.html",
    "https://docs.python.org/3/tutorial/controlflow.html",
]

# I file scaricati finiscono qui (la cartella viene creata se non esiste).
CARTELLA_OUT = Path(__file__).parent / "documenti"
CARTELLA_OUT.mkdir(exist_ok=True)


def scarica():
    for i, url in enumerate(URLS):
        print(f"Scarico: {url}")

        # 1. scarica l'HTML grezzo della pagina
        html = trafilatura.fetch_url(url)
        if not html:
            print("  -> non sono riuscito a scaricarla, salto.")
            continue

        # 2. estrae SOLO il contenuto principale come testo pulito
        testo = trafilatura.extract(html)
        if not testo:
            print("  -> nessun testo estratto, salto.")
            continue

        # 3. salva su file (nome ricavato dall'ultimo pezzo dell'URL)
        nome = url.rstrip("/").split("/")[-1].replace(".html", "") or f"pagina_{i}"
        (CARTELLA_OUT / f"{nome}.txt").write_text(testo, encoding="utf-8")
        print(f"  -> salvato '{nome}.txt' ({len(testo)} caratteri)")

        # 4. pausa di cortesia: non martellare il server (1 secondo)
        time.sleep(1)

    print(f"\nFatto. File salvati in: {CARTELLA_OUT}")
    print("Ora aprili pure, oppure indicizzali con rag_locale.py.")


if __name__ == "__main__":
    scarica()
