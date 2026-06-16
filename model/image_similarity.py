"""
image_similarity.py — DETERMINISTIC visual comparison between the generated plot
and the reference one (Plot2Code).

There is no single metric that captures colors, text (titles, axis labels,
legend) and data layout TOGETHER: they are orthogonal axes and a single number
mixes them in a non-interpretable way. So we use a COMPOSITE score, decomposed
into three independent components — each with the appropriate metric — plus a
weighted average `composite`:

  text_match  Text (title, axes, legend, ticks) via Tesseract OCR -> token F1.
              Requires the Tesseract ENGINE installed (the `pytesseract` wrapper
              is not enough). If missing, the component is None: the run does NOT
              abort (clean degradation, like codebleu in metrics.py).
  ssim        Structure / position of the data (points, lines): grayscale SSIM.
              Implemented with numpy+scipy (NO scikit-image, which on Python
              3.14/Windows has no reliable wheels — see env-gotchas).
  color_sim   Color palette: 3D RGB histogram of the NON-background pixels only
              (we mask the white, which would otherwise dominate and flatten
              everything) -> histogram intersection.

All components are in [0,1] (1 = identical). `composite` is the weighted average
of the AVAILABLE components: the None ones are ignored and the weights
renormalized, so the composite stays computable even without Tesseract.

Robustness: any error (missing/corrupt image, absent dependency) degrades to None
for that component, without ever aborting the benchmark run. numpy/PIL are
imported at module level: this file is imported ONLY in the Plot2Code branch of
the pipeline, so an install for HumanEval-only is not forced to install them.
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
    """Run `fn` and return None on any exception (a non-computable component =
    None, never a crash)."""
    try:
        return fn()
    except Exception:
        return None


def _load_rgb(path: str, size) -> np.ndarray:
    """Load a PNG as a float RGB array (H, W, 3), resized to `size`."""
    img = Image.open(path).convert("RGB")
    if img.size != size:
        img = img.resize(size, Image.BILINEAR)
    return np.asarray(img, dtype=np.float64)


def _ssim(a_gray: np.ndarray, b_gray: np.ndarray) -> float:
    """Global SSIM (mean of the map) with an 11x11 gaussian window sigma=1.5, the
    standard formulation of Wang et al. 2004. numpy+scipy only."""
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
    """Intersection of 3D RGB histograms computed on the NON-background pixels
    only. The white background, if included, dominates and makes all plots look
    'similar': we mask it (a pixel is 'ink' if its minimum channel <= threshold).
    If almost only background remains, we fall back to the whole image."""
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
    """Alphanumeric (lowercase) tokens extracted from the image via Tesseract.
    Raises if the engine is not installed: the caller catches it."""
    import pytesseract

    text = pytesseract.image_to_string(Image.open(path)).lower()
    cleaned = "".join(c if c.isalnum() else " " for c in text)
    return [t for t in cleaned.split() if t]


def _text_match(ref_path: str, gen_path: str):
    """F1 over the multisets of OCR tokens (titles/axes/legend/ticks). None if the
    Tesseract engine is unavailable."""
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
    """Weighted average of the available components (None ones ignored, weights
    renormalized). None if no component is computable."""
    avail = {k: v for k, v in parts.items() if v is not None}
    if not avail:
        return None
    wsum = sum(_WEIGHTS[k] for k in avail)
    return float(sum(v * _WEIGHTS[k] for k, v in avail.items()) / wsum)


def image_similarity(ref_path: str, gen_path: str):
    """Compare the reference PNG with the one generated by the model.

    Returns {text_match, ssim, color_sim, composite} (values in [0,1] or None for
    the non-computable components), or None if the generated render is missing (no
    figure to compare — e.g. the code did not produce a PNG)."""
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
