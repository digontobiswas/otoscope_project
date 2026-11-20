"""
Paper figure: one correctly classified test image per class, with the input,
the CBAM spatial-attention map, the four CAMs and (if SAM is installed) the
SAM ROI. Saves outputs/figures/fig_cam_examples.png at 300 dpi.

Images are picked by a fixed seed among correctly classified ones, not by hand,
so the figure is not cherry-picked. Say so in the caption.

Usage:
  python make_cam_figure.py "D:\\otoscope_intelligent_app\\test"
  python make_cam_figure.py "D:\\otoscope_intelligent_app\\test" --no-sam
"""
import os
import argparse

import numpy as np
import cv2
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config
from src import model_loader
from src import test_data
from src.xai import (eigencam, scorecam, attention_consistency, logit_cams,
                     gradcam, visualize)

SHORT = {"Miringoskleroz": "Myringosclerosis", "OtitisEksterna": "Otitis externa"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("test_dir")
    ap.add_argument("--no-sam", action="store_true")
    ap.add_argument("--score", choices=["logit", "prob"], default="logit")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out_dir = os.path.join(config.OUT_DIR, "figures")
    os.makedirs(out_dir, exist_ok=True)
    paths, labels = test_data.list_images(a.test_dir)
    model = model_loader.load_model()
    sam = None
    if not a.no_sam:
        from src.segmentation import sam_pseudomask
        sam = sam_pseudomask

    rng = np.random.default_rng(a.seed)
    cols = ["Input", "CBAM attention", "Grad-CAM", "Grad-CAM++",
            "Eigen-CAM", "Score-CAM"] + (["SAM ROI"] if sam else [])
    fig, axes = plt.subplots(config.NUM_CLASSES, len(cols),
                             figsize=(1.6 * len(cols), 1.6 * config.NUM_CLASSES))
    for ci in range(config.NUM_CLASSES):
        cand = [int(i) for i in np.where(labels == ci)[0]]
        rng.shuffle(cand)
        chosen, batch = cand[0], model_loader.preprocess_path(paths[cand[0]])
        for i in cand:                       # first correctly classified one
            b = model_loader.preprocess_path(paths[i])
            if int(np.argmax(model(tf.convert_to_tensor(b)).numpy()[0])) == ci:
                chosen, batch = i, b
                break
        rgb = batch[0].astype("uint8")
        if a.score == "logit":
            gc = logit_cams.grad_cam_logit(model, batch, ci)
            gpp = logit_cams.grad_cam_pp_logit(model, batch, ci)
        else:
            gc = gradcam.grad_cam(model, batch, class_index=ci)[0]
            gpp = gradcam.grad_cam_pp(model, batch, class_index=ci)[0]
        panels = [rgb,
                  visualize.overlay_heatmap(rgb, attention_consistency.cbam_attention_map(model, batch)),
                  visualize.overlay_heatmap(rgb, gc),
                  visualize.overlay_heatmap(rgb, gpp),
                  visualize.overlay_heatmap(rgb, eigencam.eigen_cam(model, batch, class_index=ci)[0]),
                  visualize.overlay_heatmap(rgb, scorecam.score_cam(model, batch, class_index=ci,
                                                                    max_channels=64)[0])]
        if sam:
            full = cv2.cvtColor(cv2.imread(paths[chosen]), cv2.COLOR_BGR2RGB)
            mask, _ = sam.segment_roi(full)
            panels.append(cv2.resize(visualize.mask_overlay(full, mask),
                                     (config.IMG_SIZE, config.IMG_SIZE)))
        for j, p in enumerate(panels):
            ax = axes[ci, j]
            ax.imshow(p)
            ax.set_xticks([]); ax.set_yticks([])
            if ci == 0:
                ax.set_title(cols[j], fontsize=8)
            if j == 0:
                ax.set_ylabel(SHORT.get(config.CLASSES[ci], config.CLASSES[ci]), fontsize=8)
        print(f"{config.CLASSES[ci]}: {os.path.basename(paths[chosen])}")
    fig.tight_layout(pad=0.3)
    path = os.path.join(out_dir, "fig_cam_examples.png")
    fig.savefig(path, dpi=300)
    print("Saved", path)


if __name__ == "__main__":
    main()
