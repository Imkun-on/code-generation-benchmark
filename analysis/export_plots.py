# -*- coding: utf-8 -*-
"""Extracts the charts from the executed notebook `results_analysis.ipynb` and saves
them to `../plots/` as PNGs with descriptive names, so they can be reused in the README."""
import base64
import re
from pathlib import Path

import nbformat

NB = Path(__file__).parent / "results_analysis.ipynb"
PLOTS = Path(__file__).parent.parent / "plots"
PLOTS.mkdir(exist_ok=True)


def name_for(src: str):
    """File name based on the content of the cell that produced the chart (so each
    exported PNG gets a stable, descriptive name). Returns None for cells that do
    not produce a tracked chart."""
    if "p_pass" in src:
        return "01_pass1_per_benchmark_e_modello"
    if "scatter_for_model(" in src:
        m = re.search(r'scatter_for_model\("([^"]+)"\)', src)
        return f"02_scatter_codebleu_{m.group(1)}" if m else None
    if "AnnotationBbox" in src and "MODEL_COLORS" in src:
        return "03_multipl-e_risolti_per_linguaggio_e_modello"
    if "p_cross" in src:
        return "04_cross_modello_recuperi"
    if "p_p2c" in src:
        return "05_plot2code_imgvisual_vs_codebleu"
    return None


def main():
    """Read the executed notebook, find the cells that produced a tracked chart,
    decode their PNG output and write each to ../plots/<name>.png; print a summary
    of the saved files."""
    nb = nbformat.read(NB, as_version=4)
    saved = []
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = "".join(cell["source"])
        base = name_for(src)
        if base is None:
            continue
        for out in cell.get("outputs", []):
            png = out.get("data", {}).get("image/png")
            if not png:
                continue
            fp = PLOTS / f"{base}.png"
            fp.write_bytes(base64.b64decode(png))
            saved.append(fp.name)
    print(f"{len(saved)} grafici salvati in {PLOTS}:")
    for s in saved:
        print("  -", s)


if __name__ == "__main__":
    main()
