"""
image_similarity.py — Confronto visivo DETERMINISTICO tra il plot generato e
quello di riferimento (Plot2Code).

Non esiste una singola metrica che catturi INSIEME colori, testo (titoli, label
assi, legenda) e disposizione dei dati: sono assi ortogonali e un numero unico li
mischia in modo non interpretabile. Usiamo quindi un punteggio COMPOSITO,
decomposto in tre componenti indipendenti — ognuna con la metrica adatta — più
una media pesata `composite`:

  text_match  Testo (titolo, assi, legenda, tick) via OCR Tesseract -> F1 sui
              token. Richiede il MOTORE Tesseract installato (non basta il
              wrapper `pytesseract`). Se manca, la componente è None: il run NON
              si interrompe (degradazione pulita, come codebleu in metrics.py).
  ssim        Struttura / posizione dei dati (punti, linee): SSIM in scala di
              grigi. Implementato con numpy+scipy (NIENTE scikit-image, che su
              Python 3.14/Windows non ha wheel affidabili — vedi env-gotchas).
  color_sim   Palette dei colori: istogramma RGB 3D dei soli pixel NON di sfondo
              (mascheriamo il bianco, che altrimenti domina e appiattisce tutto)
              -> intersezione di istogrammi.

Tutte le componenti sono in [0,1] (1 = identiche). `composite` è la media pesata
delle componenti DISPONIBILI: le None vengono ignorate e i pesi rinormalizzati,
così il composito resta calcolabile anche senza Tesseract.

Robustezza: qualsiasi errore (immagine mancante/corrotta, dipendenza assente)
degrada a None per quella componente, senza mai interrompere il run del benchmark.
numpy/PIL sono importati a livello di modulo: questo file viene importato SOLO nel
ramo Plot2Code della pipeline, quindi un'installazione per il solo HumanEval non è
costretta a installarli.
"""

from collections import Counter

import numpy as np
from PIL import Image

# Pesi del composito. Testo e struttura sono le componenti più discriminanti e
# affidabili; il colore si gonfia facilmente (molti plot condividono la palette
# di default di matplotlib), quindi pesa meno.
_WEIGHTS = {"text_match": 0.4, "ssim": 0.4, "color_sim": 0.2}

# Dimensione canonica di confronto = figsize di default (6.4x4.8) x dpi=100.
# Le immagini di riferimento sono 640x480; il render generato lo riportiamo qui.
_REF_SIZE = (640, 480)                       # (width, height) per PIL

# Coefficienti di luminanza ITU-R BT.601 per la conversione in scala di grigi.
_LUMA = np.array([0.299, 0.587, 0.114])


def _try(fn):
    """Esegue `fn` e ritorna None su qualunque eccezione (componente non
    calcolabile = None, mai un crash)."""
    try:
        return fn()
    except Exception:
        return None


def _load_rgb(path: str, size) -> np.ndarray:
    """Carica un PNG come array RGB float (H, W, 3), riportato a `size`."""
    img = Image.open(path).convert("RGB")
    if img.size != size:
        img = img.resize(size, Image.BILINEAR)
    return np.asarray(img, dtype=np.float64)


def _ssim(a_gray: np.ndarray, b_gray: np.ndarray) -> float:
    """SSIM globale (media della mappa) con finestra gaussiana 11x11 sigma=1.5,
    formulazione standard di Wang et al. 2004. Solo numpy+scipy."""
    from scipy.ndimage import gaussian_filter

    L = 255.0
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    blur = lambda x: gaussian_filter(x, sigma=1.5, truncate=3.5)

    mu_a, mu_b = blur(a_gray), blur(b_gray)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    var_a = blur(a_gray * a_gray) - mu_a2
    var_b = blur(b_gray * b_gray) - mu_b2
    cov_ab = blur(a_gray * b_gray) - mu_ab

    num = (2 * mu_ab + C1) * (2 * cov_ab + C2)
    den = (mu_a2 + mu_b2 + C1) * (var_a + var_b + C2)
    return float(np.clip((num / den).mean(), 0.0, 1.0))


def _color_sim(a_rgb: np.ndarray, b_rgb: np.ndarray,
               bins: int = 8, white_thresh: int = 240) -> float:
    """Intersezione di istogrammi RGB 3D calcolati sui soli pixel NON di sfondo.
    Il bianco dello sfondo, se incluso, domina e rende tutti i plot 'simili':
    lo mascheriamo (un pixel è 'inchiostro' se il suo canale minimo <= soglia).
    Se resta quasi solo sfondo, ripieghiamo sull'immagine intera."""
    def hist(rgb):
        flat = rgb.reshape(-1, 3)
        ink = flat[flat.min(axis=1) <= white_thresh]
        if ink.shape[0] < 50:                # quasi tutto sfondo: usa tutti i pixel
            ink = flat
        h, _ = np.histogramdd(ink, bins=bins, range=[(0, 255)] * 3)
        total = h.sum()
        return h / total if total else h

    return float(np.minimum(hist(a_rgb), hist(b_rgb)).sum())   # intersezione in [0,1]


def _ocr_tokens(path: str) -> list[str]:
    """Token alfanumerici (minuscoli) estratti dall'immagine via Tesseract.
    Solleva se il motore non è installato: il chiamante lo intercetta."""
    import pytesseract

    text = pytesseract.image_to_string(Image.open(path)).lower()
    cleaned = "".join(c if c.isalnum() else " " for c in text)
    return [t for t in cleaned.split() if t]


def _text_match(ref_path: str, gen_path: str):
    """F1 sui multiset di token OCR (titoli/assi/legenda/tick). None se il motore
    Tesseract non è disponibile."""
    try:
        ref = _ocr_tokens(ref_path)
        gen = _ocr_tokens(gen_path)
    except Exception:
        return None
    if not ref and not gen:
        return 1.0                           # entrambi senza testo: coincidono
    if not ref or not gen:
        return 0.0
    overlap = sum((Counter(ref) & Counter(gen)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(gen)
    recall = overlap / len(ref)
    return float(2 * precision * recall / (precision + recall))


def _composite(parts: dict):
    """Media pesata delle componenti disponibili (None ignorate, pesi
    rinormalizzati). None se nessuna componente è calcolabile."""
    avail = {k: v for k, v in parts.items() if v is not None}
    if not avail:
        return None
    wsum = sum(_WEIGHTS[k] for k in avail)
    return float(sum(v * _WEIGHTS[k] for k, v in avail.items()) / wsum)


def image_similarity(ref_path: str, gen_path: str):
    """Confronta il PNG di riferimento con quello generato dal modello.

    Ritorna {text_match, ssim, color_sim, composite} (valori in [0,1] o None per
    le componenti non calcolabili), oppure None se manca il render generato
    (nessuna figura da confrontare — es. il codice non ha prodotto un PNG)."""
    if not ref_path or not gen_path:
        return None
    try:
        a_rgb = _load_rgb(ref_path, _REF_SIZE)
        b_rgb = _load_rgb(gen_path, _REF_SIZE)
    except Exception:
        return None                          # immagine mancante/corrotta

    a_gray, b_gray = a_rgb @ _LUMA, b_rgb @ _LUMA
    parts = {
        "text_match": _text_match(ref_path, gen_path),
        "ssim": _try(lambda: _ssim(a_gray, b_gray)),
        "color_sim": _try(lambda: _color_sim(a_rgb, b_rgb)),
    }
    return {**parts, "composite": _composite(parts)}
