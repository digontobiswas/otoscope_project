"""
Paper figure for the SAM ROI step: examples of masks you marked acceptable (Y)
and rejected (N) in outputs/seg/qa_sheet.csv. Showing failures as well as
successes is what makes a weak-supervision figure credible.

Run after run_seg_batch.py and after filling in qa_sheet.csv:
  python make_seg_figure.py
  python make_seg_figure.py --good 4 --bad 2

Saves outputs/figures/fig_seg_examples.png at 300 dpi.
"""
import os
import csv
import argparse

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config

SEG = os.path.join(config.OUT_DIR, "seg")
QA_DIR = os.path.join(SEG, "qa")
QA_SHEET = os.path.join(SEG, "qa_sheet.csv")
SHORT = {"Miringoskleroz": "Myringosclerosis", "OtitisEksterna": "Otitis externa"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--good", type=int, default=4)
    ap.add_argument("--bad", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if not os.path.exists(QA_SHEET):
        raise SystemExit("Run run_seg_batch.py first (no qa_sheet.csv yet).")
    with open(QA_SHEET, newline="") as f:
        rows = list(csv.DictReader(f))
    key = next(k for k in rows[0] if k.startswith("acceptable"))
    good = [r for r in rows if r[key].strip().upper() == "Y"]
    bad = [r for r in rows if r[key].strip().upper() == "N"]
    if not good and not bad:
        raise SystemExit("Fill in the last column of qa_sheet.csv (Y/N) first.")

    rng = np.random.default_rng(a.seed)
    pick = lambda lst, k: [lst[i] for i in sorted(rng.choice(len(lst), min(k, len(lst)), replace=False))] if lst else []
    chosen = [(r, "acceptable") for r in pick(good, a.good)] + \
             [(r, "rejected") for r in pick(bad, a.bad)]

    fig, axes = plt.subplots(1, len(chosen), figsize=(2.0 * len(chosen), 2.35))
    axes = np.atleast_1d(axes)
    for ax, (r, verdict) in zip(axes, chosen):
        img = cv2.cvtColor(cv2.imread(os.path.join(QA_DIR, r["file"])), cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        src = " (box fallback)" if r["source"] == "box_fallback" else ""
        ax.set_title(f"{SHORT.get(r['class'], r['class'])}\n{verdict}{src}",
                     fontsize=7, color="black" if verdict == "acceptable" else "firebrick")
    fig.tight_layout(pad=0.3)
    out = os.path.join(config.OUT_DIR, "figures")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "fig_seg_examples.png")
    fig.savefig(path, dpi=300)
    print(f"Saved {path}  ({len(good)} Y / {len(bad)} N in the QA sheet)")


if __name__ == "__main__":
    main()
