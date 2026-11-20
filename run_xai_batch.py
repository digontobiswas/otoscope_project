"""
XAI evaluation over the test set, with the reference maps a reviewer will ask
for. Nothing is trained; it runs on weights/cbam_mnv3.keras.

For every image it computes four CAMs (Grad-CAM, Grad-CAM++, Eigen-CAM,
Score-CAM) plus reference maps (random, center prior, and Grad-CAM /
Grad-CAM++ on a randomly initialised copy of the network), then:

  - deletion / insertion AUC of every map, scored on the TRAINED model
  - agreement of every map with the CBAM spatial-attention map
    (IoU at 0.5, Pearson at 224x224 and at the native 7x7 grid, Spearman,
    centroid distance)
  - raw dynamic range of the CBAM attention map (is it nearly flat?)
  - sanity check: rank correlation between each gradient CAM on the trained
    model and on the randomized model (high = method ignores the weights)

Rows are appended to outputs/xai/xai_per_image.csv as each image finishes, so
an interrupted run resumes where it stopped. Summaries are written at the end
(or with --summarize-only).

Usage (from the project folder, venv active):
  python run_xai_batch.py "D:\\otoscope_intelligent_app\\test"
  python run_xai_batch.py "D:\\otoscope_intelligent_app\\test" --n 12     # quick test
  python run_xai_batch.py "D:\\otoscope_intelligent_app\\test" --score prob
  python run_xai_batch.py --summarize-only

--score logit (default) differentiates the pre-softmax class score, which is
the standard definition. --score prob reproduces the older numbers, which
differentiated the softmax probability.
"""
import os
import sys
import csv
import time
import argparse

import numpy as np
import cv2
import tensorflow as tf
from scipy.stats import pearsonr, spearmanr

from src import config
from src import model_loader
from src import test_data
from src.xai import (gradcam, eigencam, scorecam, faithfulness,
                     attention_consistency, logit_cams, reference_maps)

OUT = os.path.join(config.OUT_DIR, "xai")
PER_IMAGE = os.path.join(OUT, "xai_per_image.csv")

CAMS = ["GradCAM", "GradCAM++", "EigenCAM", "ScoreCAM"]
REFS = ["Random", "CenterPrior", "RandModel-GradCAM", "RandModel-GradCAM++"]
MAPS = CAMS + REFS
MAP_FIELDS = ["del", "ins", "iou", "pearson224", "pearson7", "spearman",
              "centroid", "flat"]


def _corr(a, b, fn):
    a, b = a.ravel(), b.ravel()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(fn(a, b)[0])


def _gradient_cams(model, batch, ci, score, layer_name=None):
    if score == "logit":
        return (logit_cams.grad_cam_logit(model, batch, ci, layer_name),
                logit_cams.grad_cam_pp_logit(model, batch, ci, layer_name))
    return (gradcam.grad_cam(model, batch, class_index=ci, layer_name=layer_name)[0],
            gradcam.grad_cam_pp(model, batch, class_index=ci, layer_name=layer_name)[0])


def _cbam_raw_and_norm(sp_model, batch):
    """Raw 7x7 sigmoid attention, and the same map upsampled + min-max scaled
    exactly as attention_consistency.cbam_attention_map does."""
    raw = sp_model(tf.convert_to_tensor(batch, dtype=tf.float32))[0].numpy()[..., 0]
    up = cv2.resize(raw, (config.IMG_SIZE, config.IMG_SIZE))
    up = up - up.min()
    if up.max() > 0:
        up = up / up.max()
    return raw, up.astype("float32")


def header():
    h = ["idx", "path", "true", "pred", "correct", "conf",
         "cbam_min", "cbam_max", "cbam_std", "cbam_range"]
    for m in MAPS:
        h += [f"{m}|{f}" for f in MAP_FIELDS]
    h += ["sanity_GradCAM", "sanity_GradCAM++"]
    return h


def _check_run_config(score):
    """Stop if a resumed run would mix --score settings in one CSV."""
    p = os.path.join(OUT, "run_config.txt")
    if os.path.exists(p):
        old = open(p).read().strip()
        if old != f"score={score}":
            sys.exit(f"{OUT} holds results with {old}. Delete that folder or "
                     f"use the same --score to resume.")
    else:
        with open(p, "w") as f:
            f.write(f"score={score}\n")


def run(test_dir, n, score, sc_channels):
    os.makedirs(OUT, exist_ok=True)
    _check_run_config(score)
    paths, labels = test_data.list_images(test_dir)
    if not paths:
        sys.exit(f"No images found under {test_dir}")
    idx = test_data.stratified_subset(labels, n)
    print(f"{len(paths)} images found, evaluating {len(idx)} "
          f"(score={score}, Score-CAM channels={sc_channels})")

    # trained model FIRST, so its auto-generated layer names match config.py
    model = model_loader.load_model()
    rmodel, rlayer = reference_maps.randomized_model(model)
    sp_model = tf.keras.models.Model(
        model.inputs, model.get_layer(config.CBAM_SPATIAL_LAYER).output)
    center = reference_maps.center_prior()

    done = set()
    new_file = not os.path.exists(PER_IMAGE)
    if not new_file:
        with open(PER_IMAGE, newline="") as f:
            done = {int(r["idx"]) for r in csv.DictReader(f)}
        print(f"Resuming: {len(done)} images already done")
    f = open(PER_IMAGE, "a", newline="")
    w = csv.writer(f)
    if new_file:
        w.writerow(header())

    for k, i in enumerate(idx):
        if i in done:
            continue
        t0 = time.time()
        batch = model_loader.preprocess_path(paths[i])
        probs = model(tf.convert_to_tensor(batch, dtype=tf.float32)).numpy()[0]
        ci = int(np.argmax(probs))
        rng = np.random.default_rng(1000 + i)

        gc, gpp = _gradient_cams(model, batch, ci, score)
        rgc, rgpp = _gradient_cams(rmodel, batch, ci, score, layer_name=rlayer)
        maps = {
            "GradCAM": gc,
            "GradCAM++": gpp,
            "EigenCAM": eigencam.eigen_cam(model, batch, class_index=ci)[0],
            "ScoreCAM": scorecam.score_cam(model, batch, class_index=ci,
                                           max_channels=sc_channels)[0],
            "Random": reference_maps.random_map(rng),
            "CenterPrior": center,
            "RandModel-GradCAM": rgc,
            "RandModel-GradCAM++": rgpp,
        }

        raw, cbam = _cbam_raw_and_norm(sp_model, batch)
        cbam7 = reference_maps.to_grid(cbam)

        row = [i, paths[i], int(labels[i]), ci, int(ci == labels[i]),
               round(float(probs[ci]), 4), float(raw.min()), float(raw.max()),
               float(raw.std()), float(raw.max() - raw.min())]
        for name in MAPS:
            m = maps[name]
            fa = faithfulness.deletion_insertion(model, batch, m,
                                                 class_index=ci, steps=50)
            co = attention_consistency.consistency(cbam, m)
            p7 = _corr(cbam7, reference_maps.to_grid(m), pearsonr)
            row += [fa["deletion_auc"], fa["insertion_auc"], co["iou"],
                    co["pearson"], p7, co["spearman"], co["centroid_dist"],
                    int(m.max() == 0)]
        row += [_corr(gc, rgc, spearmanr), _corr(gpp, rgpp, spearmanr)]
        w.writerow(row)
        f.flush()
        print(f"  {k+1}/{len(idx)}  {os.path.basename(paths[i])}  "
              f"{time.time()-t0:.1f}s", flush=True)
    f.close()


# ------------------------------------------------------------------ summaries
def _load():
    with open(PER_IMAGE, newline="") as f:
        return list(csv.DictReader(f))


def _vals(rows, key):
    v = np.array([float(r[key]) for r in rows], dtype=float)
    return v[~np.isnan(v)]


def _ms(v):
    return (round(float(v.mean()), 4), round(float(v.std()), 4)) if len(v) else ("", "")


def summarize():
    if not os.path.exists(PER_IMAGE):
        sys.exit(f"No results yet: {PER_IMAGE} does not exist.")
    rows = _load()
    subsets = {"all": rows,
               "correct": [r for r in rows if r["correct"] == "1"],
               "incorrect": [r for r in rows if r["correct"] == "0"]}

    with open(os.path.join(OUT, "faithfulness_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["map", "subset", "n", "del_mean", "del_std",
                    "ins_mean", "ins_std", "n_flat_maps"])
        for m in MAPS:
            for s, rs in subsets.items():
                if not rs:
                    continue
                w.writerow([m, s, len(rs), *_ms(_vals(rs, f"{m}|del")),
                            *_ms(_vals(rs, f"{m}|ins")),
                            int(sum(int(r[f"{m}|flat"]) for r in rs))])

    with open(os.path.join(OUT, "consistency_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["map", "subset", "n", "iou_mean", "iou_std",
                    "pearson224_mean", "pearson224_std", "pearson7_mean",
                    "pearson7_std", "spearman_mean", "centroid_mean"])
        for m in MAPS:
            for s, rs in subsets.items():
                if not rs:
                    continue
                w.writerow([m, s, len(rs), *_ms(_vals(rs, f"{m}|iou")),
                            *_ms(_vals(rs, f"{m}|pearson224")),
                            *_ms(_vals(rs, f"{m}|pearson7")),
                            _ms(_vals(rs, f"{m}|spearman"))[0],
                            _ms(_vals(rs, f"{m}|centroid"))[0]])

    # does agreement with CBAM track faithfulness, image by image?
    with open(os.path.join(OUT, "consistency_vs_deletion.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cam", "n", "spearman_rho(pearson7, deletion_auc)", "p_value"])
        for m in CAMS:
            a = np.array([float(r[f"{m}|pearson7"]) for r in rows])
            b = np.array([float(r[f"{m}|del"]) for r in rows])
            ok = ~np.isnan(a) & ~np.isnan(b)
            if ok.sum() > 3:
                rho, p = spearmanr(a[ok], b[ok])
                w.writerow([m, int(ok.sum()), round(float(rho), 4), round(float(p), 4)])

    with open(os.path.join(OUT, "sanity_and_cbam.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "mean", "std"])
        for key in ["sanity_GradCAM", "sanity_GradCAM++"]:
            w.writerow([key + " (spearman trained vs random weights)", *_ms(_vals(rows, key))])
        for key in ["cbam_min", "cbam_max", "cbam_std", "cbam_range"]:
            w.writerow([key + " (raw sigmoid, 7x7)", *_ms(_vals(rows, key))])

    # paired per-image tests: each CAM against each reference map (same images)
    from scipy.stats import wilcoxon
    with open(os.path.join(OUT, "paired_tests.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cam", "reference", "metric", "median_diff(cam-ref)",
                    "pct_images_cam_better", "wilcoxon_p"])
        for ref in REFS:
            for cam in CAMS:
                for met in ["del", "ins"]:
                    a = np.array([float(r[f"{cam}|{met}"]) for r in rows])
                    b = np.array([float(r[f"{ref}|{met}"]) for r in rows])
                    d = a - b
                    better = (d < 0).mean() if met == "del" else (d > 0).mean()
                    p = wilcoxon(a, b).pvalue if np.any(d != 0) else 1.0
                    w.writerow([cam, ref, met, round(float(np.median(d)), 4),
                                round(100 * float(better), 1), f"{p:.2g}"])

    print(f"\nSummaries written to {OUT} (n={len(rows)} images)")
    print("\nFaithfulness, all images (deletion lower = better, insertion higher = better):")
    for m in MAPS:
        d, i = _ms(_vals(rows, f"{m}|del")), _ms(_vals(rows, f"{m}|ins"))
        print(f"  {m:<22} del {d[0]} +- {d[1]}   ins {i[0]} +- {i[1]}")
    print("\nAgreement with CBAM attention (Pearson at 7x7):")
    for m in MAPS:
        p = _ms(_vals(rows, f"{m}|pearson7"))
        print(f"  {m:<22} {p[0]} +- {p[1]}")
    r = _ms(_vals(rows, "cbam_range"))
    print(f"\nCBAM raw attention range (max-min): {r[0]} +- {r[1]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("test_dir", nargs="?")
    ap.add_argument("--n", type=int, default=None, help="stratified subset size")
    ap.add_argument("--score", choices=["logit", "prob"], default="logit")
    ap.add_argument("--scorecam-channels", type=int, default=64)
    ap.add_argument("--summarize-only", action="store_true")
    a = ap.parse_args()
    if not a.summarize_only:
        if not a.test_dir:
            ap.error("test_dir is required unless --summarize-only")
        run(a.test_dir, a.n, a.score, a.scorecam_channels)
    summarize()


if __name__ == "__main__":
    main()
