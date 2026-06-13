"""
rag_locale.py — Il "Hello World" del RAG, 100% locale e gratuito.

Pipeline (la stessa di OGNI sistema RAG, dal giocattolo a quelli in produzione):

    1. CHUNK   -> spezzo i documenti in pezzetti (vedi spezza_in_chunk)
    2. EMBED   -> trasformo ogni pezzetto in un vettore di numeri (sentence-transformers)
    3. STORE   -> li tengo in memoria (qui una lista numpy; in produzione: Chroma/FAISS)
    4. QUERY   -> la domanda diventa un vettore -> cerco i pezzetti piu' "vicini"
    5. GENERATE-> incollo i pezzetti trovati nel prompt e li do all'LLM locale (Ollama)

Usa AUTOMATICAMENTE i file .txt in rag_demo/documenti/ se ce ne sono
(quelli scaricati con scarica_docs.py); altrimenti usa delle frasi-demo.

Esegui con:  python rag_demo/rag_locale.py
"""

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

CARTELLA_DOCS = Path(__file__).parent / "documenti"


# --------------------------------------------------------------------------
# PASSO 1 (CHUNK) - funzioni per preparare i documenti.
# --------------------------------------------------------------------------
def spezza_in_chunk(testo: str, parole_per_chunk: int = 200, sovrapposizione: int = 40) -> list[str]:
    """Spezza un testo lungo in pezzi piu' piccoli.

    Una pagina di documentazione e' troppo lunga per il prompt: la tagliamo in
    blocchi da ~200 parole. La 'sovrapposizione' fa ripetere le ultime 40 parole
    all'inizio del blocco successivo, cosi' una frase a cavallo tra due chunk non
    viene spezzata a meta' e persa. (E' la regolazione che piu' influenza la qualita'.)"""
    parole = testo.split()
    chunk, i = [], 0
    while i < len(parole):
        chunk.append(" ".join(parole[i:i + parole_per_chunk]))
        i += parole_per_chunk - sovrapposizione
    return chunk


def carica_da_cartella(percorso: Path) -> list[str]:
    """Legge tutti i .txt di una cartella e li spezza in chunk pronti da indicizzare."""
    chunk = []
    for file in percorso.glob("*.txt"):
        testo = file.read_text(encoding="utf-8")
        chunk.extend(spezza_in_chunk(testo))
    return chunk


# Frasi-demo: usate solo se la cartella documenti/ e' vuota, per imparare subito.
DOCS_DEMO = [
    "Python e' un linguaggio interpretato, tipizzato dinamicamente, creato da Guido van Rossum nel 1991.",
    "Le list comprehension in Python: [x*2 for x in range(5)] produce [0, 2, 4, 6, 8].",
    "Per installare un pacchetto si usa pip, ad esempio: pip install requests.",
    "Un dizionario in Python si crea con {chiave: valore} e si accede con d[chiave].",
    "La funzione open() apre un file; con 'with open(...) as f' il file viene chiuso da solo.",
    "NumPy serve per il calcolo numerico veloce su array; il cuore e' l'oggetto ndarray.",
    "Il gatto domestico (Felis catus) dorme in media dalle 12 alle 16 ore al giorno.",  # rumore
]

# Se hai scaricato documenti veri (scarica_docs.py), usali; altrimenti i demo.
if CARTELLA_DOCS.exists() and any(CARTELLA_DOCS.glob("*.txt")):
    DOCS = carica_da_cartella(CARTELLA_DOCS)
    print(f"Uso i documenti scaricati in {CARTELLA_DOCS} ({len(DOCS)} chunk).")
else:
    DOCS = DOCS_DEMO
    print("Uso le frasi-demo (nessun file in rag_demo/documenti/).")


# --------------------------------------------------------------------------
# PASSO 2-3 (EMBED + STORE) - una volta sola: carichiamo il modello di embedding
# e trasformiamo ogni chunk in un vettore. Il modello e' piccolo e gira su CPU.
# La PRIMA volta lo scarica da internet (~80 MB), poi resta in cache locale.
# --------------------------------------------------------------------------
print("Carico il modello di embedding (la prima volta lo scarica)...")
modello_embed = SentenceTransformer("all-MiniLM-L6-v2")

# normalize_embeddings=True -> i vettori hanno lunghezza 1, cosi' il prodotto
# scalare e' direttamente la "cosine similarity" (quanto due testi si somigliano).
vettori_docs = modello_embed.encode(DOCS, normalize_embeddings=True)
print(f"Indicizzati {len(DOCS)} chunk. Ogni vettore ha {vettori_docs.shape[1]} numeri.\n")


def cerca(domanda: str, k: int = 3):
    """PASSO 4 (QUERY): trova i k chunk piu' vicini alla domanda.

    Questo e' ESATTAMENTE cio' che fa un database vettoriale, ma fatto a mano
    cosi' lo vedi: trasformo la domanda in vettore e calcolo la somiglianza
    con ogni chunk, poi prendo i migliori."""
    v_domanda = modello_embed.encode([domanda], normalize_embeddings=True)[0]
    # prodotto scalare = cosine similarity (perche' i vettori sono normalizzati)
    somiglianze = vettori_docs @ v_domanda
    # indici dei k punteggi piu' alti, dal migliore al peggiore
    migliori = np.argsort(somiglianze)[::-1][:k]
    return [(DOCS[i], float(somiglianze[i])) for i in migliori]


def rispondi(domanda: str):
    """PASSO 5 (GENERATE): costruisco il prompt con i chunk trovati e lo mando
    all'LLM locale via Ollama. Se Ollama non c'e', mostro solo i chunk
    recuperati (la parte 'retrieval' funziona comunque)."""
    trovati = cerca(domanda)

    print(f"DOMANDA: {domanda}")
    print("\nChunk recuperati (contesto):")
    for testo, punteggio in trovati:
        anteprima = testo[:120].replace("\n", " ")
        print(f"  [{punteggio:.2f}] {anteprima}...")

    # Questo e' il "augmented": incolliamo il contesto trovato nel prompt.
    contesto = "\n".join(f"- {testo}" for testo, _ in trovati)
    prompt = (
        "Rispondi alla domanda usando SOLO le informazioni qui sotto. "
        "Se la risposta non c'e', dillo onestamente.\n\n"
        f"INFORMAZIONI:\n{contesto}\n\n"
        f"DOMANDA: {domanda}\nRISPOSTA:"
    )

    try:
        import ollama
        risposta = ollama.chat(
            model="llama3.2",  # cambia col modello che hai fatto 'ollama pull'
            messages=[{"role": "user", "content": prompt}],
        )
        print("\nRISPOSTA dell'LLM:")
        print(risposta["message"]["content"])
    except Exception as e:
        print(f"\n(LLM non disponibile: {e})")
        print("La ricerca pero' ha funzionato! Avvia Ollama e fai 'ollama pull llama3.2' "
              "per vedere anche la risposta generata.")
    print("\n" + "-" * 70 + "\n")


if __name__ == "__main__":
    # Cambia queste domande in base ai documenti che hai indicizzato.
    rispondi("Come installo un pacchetto in Python?")
    rispondi("A cosa servono le list comprehension?")
    rispondi("Come si crea una lista in Python?")
