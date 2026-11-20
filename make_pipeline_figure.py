"""
Overview figure of the system (Fig. 1 of the paper). Pure matplotlib, no data
needed. Saves outputs/figures/fig_pipeline.png (300 dpi) and fig_pipeline.pdf
(vector; use the PDF in Overleaf if you prefer sharper text).

Usage:  python make_pipeline_figure.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

try:
    from src import config
    OUT = os.path.join(config.OUT_DIR, "figures")
except Exception:                       # runnable outside the project too
    OUT = os.path.join("outputs", "figures")

INK = "#1f2933"
MUTED = "#52606d"
FILL = {"io": "#e4e7eb", "prep": "#dbe7f3", "model": "#d9ecdf",
        "xai": "#f6e3c7", "seg": "#e9dcf2", "out": "#f3d9d9"}
EDGE = {"io": "#7b8794", "prep": "#4a6fa5", "model": "#3f7d58",
        "xai": "#b7791f", "seg": "#7a4fa0", "out": "#a94442"}


def box(ax, x, y, w, h, kind, title, lines=(), title_size=8.6, body_size=7.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                fc=FILL[kind], ec=EDGE[kind], lw=1.1))
    ax.text(x + w / 2, y + h - 0.022, title, ha="center", va="top",
            fontsize=title_size, weight="bold", color=INK)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.060 - i * 0.031, ln, ha="center", va="top",
                fontsize=body_size, color=MUTED)
    return (x, y, w, h)


def arrow(ax, p, q, text=None, rad=0.0, dy=0.012):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9,
                                 lw=1.0, color=INK, shrinkA=2, shrinkB=2,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + dy, text, ha="center",
                va="bottom", fontsize=6.4, color=MUTED, style="italic")


def right(b, f=0.5):  return (b[0] + b[2], b[1] + b[3] * f)
def left(b, f=0.5):   return (b[0], b[1] + b[3] * f)


def main():
    os.makedirs(OUT, exist_ok=True)
    fig = plt.figure(figsize=(14, 5.0))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1.4); ax.set_ylim(0, 0.5)
    ax.axis("off")

    # column 1-2: input and preprocessing
    inp = box(ax, 0.01, 0.19, 0.12, 0.13, "io", "Otoscope image",
              ["raw frame / upload"])
    prep = box(ax, 0.17, 0.12, 0.17, 0.27, "prep", "Preprocessing",
               ["resize to 500×500 (area)", "CLAHE on L of CIELAB",
                "clip 2.0, 8×8 tiles", "resize to 224×224", "(applied once)"])

    # column 3: network
    net = box(ax, 0.38, 0.06, 0.25, 0.39, "model", "CBAM-MobileNetV3-Small",
              ["1,028,577 parameters", "", "MobileNetV3-Small backbone",
               "(ImageNet init.)", "feature map F: 7×7×576",
               "CBAM: channel Mc, spatial Ms", "GAP → BN → dropout 0.4",
               "dense, 6-way softmax"])

    # column 4: four branches
    bw, bh, bx = 0.30, 0.105, 0.70
    pred = box(ax, bx, 0.385, bw, bh, "model", "Prediction",
               ["6 class probabilities (+ flip TTA)", "calibration: ECE, reliability"])
    xai = box(ax, bx, 0.265, bw, bh, "xai", "Post-hoc explanations (from F)",
              ["Grad-CAM, Grad-CAM++, Eigen-CAM, Score-CAM",
               "deletion / insertion vs. reference maps"])
    att = box(ax, bx, 0.145, bw, bh, "xai", "Attention analysis",
              ["CBAM spatial map Ms vs. each CAM",
               "Pearson (7×7), vs. center prior"])
    seg = box(ax, bx, 0.012, bw, bh + 0.01, "seg", "SAM ViT-B region of interest",
              ["central box prompt (60%) on 500×500",
               "CAM mass in ROI, masked re-classification"])

    # column 5: report + deployment
    rep = box(ax, 1.07, 0.20, 0.31, 0.25, "out", "Structured report",
              ["template-filled, deterministic", "no language model",
               "class, top-3, certainty band,", "heatmap, ROI, fixed caveats",
               "JSON · text · PDF"])
    dep = box(ax, 1.07, 0.03, 0.31, 0.13, "io", "Delivery",
              ["browser app (Gradio), CPU only", "target: Raspberry Pi"])

    # arrows
    arrow(ax, right(inp), left(prep))
    arrow(ax, right(prep, 0.62), left(net, 0.62), "224×224")
    arrow(ax, right(net, 0.92), left(pred))
    arrow(ax, right(net, 0.62), left(xai), "F")
    arrow(ax, right(net, 0.33), left(att), "Ms")
    arrow(ax, (xai[0] + 0.05, xai[1]), (att[0] + 0.05, att[1] + att[3]))
    # SAM gets the 500x500 image straight from preprocessing
    ax.add_patch(FancyArrowPatch((prep[0] + prep[2] / 2, prep[1]),
                                 (seg[0], seg[1] + seg[3] * 0.16),
                                 arrowstyle="-|>", mutation_scale=9, lw=1.0,
                                 color=INK, shrinkA=2, shrinkB=2,
                                 connectionstyle="angle,angleA=-90,angleB=180,rad=6"))
    ax.text(0.40, 0.004, "500×500 image", fontsize=6.4, color=MUTED, style="italic")
    arrow(ax, right(pred), left(rep, 0.85))
    arrow(ax, right(xai), left(rep, 0.55))
    arrow(ax, right(seg), left(rep, 0.12), rad=0.15)
    arrow(ax, (rep[0] + rep[2] / 2, rep[1]), (dep[0] + dep[2] / 2, dep[1] + dep[3]))

    png = os.path.join(OUT, "fig_pipeline.png")
    fig.savefig(png, dpi=300)
    fig.savefig(os.path.join(OUT, "fig_pipeline.pdf"))
    print("Saved", png)


if __name__ == "__main__":
    main()
