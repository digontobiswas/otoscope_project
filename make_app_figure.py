"""
Paper figure for the application: runs the SAME analysis function the Gradio
app uses (gradio_app.analyze) on one raw otoscope image and lays out what the
user sees: (a) input, (b) class probabilities, (c) explanation heatmap,
(d) SAM region of interest, (e) the generated structured report.

Use a RAW image from the original dataset folder (eardrum\<class>\...), not one
from the test folder: the app applies CLAHE itself, and test images already
have it.

Usage:
  python make_app_figure.py "C:\\...\\eardrum\\AOM\\some_image.png"
  python make_app_figure.py "<raw image>" --method Grad-CAM++

Saves outputs/figures/fig_app_report.png (300 dpi) and a copy of the PDF
report the app produced (outputs/figures/app_report_example.pdf).
"""
import os
import sys
import shutil
import argparse
import textwrap

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="raw otoscope image (not from the test folder)")
    ap.add_argument("--method", default="Grad-CAM",
                    choices=["Grad-CAM", "Grad-CAM++", "Eigen-CAM", "Score-CAM"])
    a = ap.parse_args()
    if not os.path.exists(a.image):
        sys.exit(f"Image not found: {a.image}")

    import gradio_app                     # loads the model once, as the app does
    pil = Image.open(a.image).convert("RGB")
    (xai, conf, seg, report_txt, notes,
     json_path, pdf_path) = gradio_app.analyze(pil, a.method, False)

    out = os.path.join(config.OUT_DIR, "figures")
    os.makedirs(out, exist_ok=True)
    if pdf_path and os.path.exists(pdf_path):
        shutil.copy(pdf_path, os.path.join(out, "app_report_example.pdf"))

    fig = plt.figure(figsize=(13, 6.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1.35])

    ax = fig.add_subplot(gs[0, 0]); ax.imshow(pil); ax.set_title("(a) Input (raw upload)", fontsize=10)
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 0])
    names = [k.split(" (")[0] for k in conf]
    names = [{"Miringoskleroz": "Myringoscl.", "OtitisEksterna": "Otitis ext."}.get(n, n) for n in names]
    vals = list(conf.values())
    order = np.argsort(vals)
    ax.barh([names[i] for i in order], [vals[i] for i in order], color="#4a6fa5")
    ax.set_xlim(0, 1); ax.set_xlabel("probability (uncalibrated)", fontsize=8)
    ax.tick_params(labelsize=8); ax.set_title("(b) Class probabilities", fontsize=10)

    ax = fig.add_subplot(gs[0, 1]); ax.imshow(xai); ax.axis("off")
    ax.set_title(f"(c) {a.method} heatmap", fontsize=10)

    ax = fig.add_subplot(gs[1, 1])
    if seg is not None:
        ax.imshow(seg)
    else:
        ax.text(0.5, 0.5, "SAM not available", ha="center", va="center")
    ax.axis("off"); ax.set_title("(d) SAM region of interest", fontsize=10)

    ax = fig.add_subplot(gs[:, 2:]); ax.axis("off")
    ax.set_title("(e) Generated report (template-filled, no language model)", fontsize=10)
    lines = []
    for ln in report_txt.splitlines():
        lines.extend(textwrap.wrap(ln, 70) or [""])
    ax.text(0.0, 1.0, "\n".join(lines), family="monospace", fontsize=8.6,
            va="top", ha="left", transform=ax.transAxes)

    fig.tight_layout(pad=0.6)
    path = os.path.join(out, "fig_app_report.png")
    fig.savefig(path, dpi=300)
    top = max(conf, key=conf.get)
    print(f"Saved {path}")
    print(f"Prediction: {top}  ({conf[top]*100:.1f}%)")
    print(notes)


if __name__ == "__main__":
    main()
