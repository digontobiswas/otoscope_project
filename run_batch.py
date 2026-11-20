"""
Batch evaluation over the whole test set, with horizontal-flip TTA.

Produces every number and table the paper's Results section needs, saved as
CSV/PNG under outputs/batch/:

  - metrics_overall.csv      accuracy, macro P/R/F1, sensitivity, specificity,
                             macro-AUC, MCC, Cohen's kappa (no-TTA and TTA)
  - metrics_per_class.csv    per-class precision/recall/F1/sensitivity/specificity
  - confusion_matrix.csv     + confusion_matrix.png
  - calibration.csv          ECE / MCE (+ temperature-scaled), reliability.png
  - bootstrap_ci.csv         95% CI on accuracy
  - xai_faithfulness.csv     deletion/insertion AUC per CAM (on a subset)
  - attention_consistency.csv IoU / Pearson per CAM (on a subset)

Nothing is trained. Everything runs on the existing cbam_mnv3.keras weights.

Usage:
  python run_batch.py                 # uses TEST_DIR below
  python run_batch.py /path/to/test   # override test dir
  python run_batch.py --xai-subset 40 # how many images for the (slow) XAI tables
"""
import os
import sys
import csv

import numpy as np
import tensorflow as tf
import cv2

from src import config
from src import model_loader
from src import evaluation_metrics as em

# ----- where the test images live (edit if different) -----
TEST_DIR = os.path.join(
    os.path.dirname(config.BASE_DIR),
    "base data of otoscope", "pipeline", "05_split", "test")
# On your machine this is usually:
#   .../pipeline/05_split/test/<class>/*
# Pass a path as the first CLI arg to override.

OUT = os.path.join(config.OUT_DIR, "batch")
os.makedirs(OUT, exist_ok=True)

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _list(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if f.lower().endswith(IMG_EXTS))


def load_test(test_dir):
    paths, labels = [], []
    for ci, c in enumerate(config.CLASSES):
        for p in _list(os.path.join(test_dir, c)):
            paths.append(p); labels.append(ci)
    return paths, np.array(labels)


def predict_all(model, paths, tta=False):
    """Return (probs, logits). Logits are needed for temperature scaling."""
    # a logits model = same graph minus the final softmax
    logit_layer = model.layers[-1]
    logit_model = tf.keras.models.Model(
        model.inputs, logit_layer.input)   # pre-softmax dense output feeds softmax

    probs, logits = [], []
    for p in paths:
        batch = model_loader.preprocess_path(p)      # (1,224,224,3)
        pr = model.predict(batch, verbose=0)
        lg = logit_model.predict(batch, verbose=0)
        if tta:
            flip = batch[:, :, ::-1, :]
            pr = (pr + model.predict(flip, verbose=0)) / 2.0
            lg = (lg + logit_model.predict(flip, verbose=0)) / 2.0
        probs.append(pr[0]); logits.append(lg[0])
    return np.array(probs), np.array(logits)


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def save_overall(m_no, m_tta):
    keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro",
            "sensitivity_macro", "specificity_macro", "auc_macro", "mcc",
            "cohen_kappa"]
    rows = [[k, round(m_no[k], 4), round(m_tta[k], 4)] for k in keys]
    _write_csv(os.path.join(OUT, "metrics_overall.csv"),
               ["metric", "no_TTA", "TTA"], rows)
    print("\nOverall metrics (no-TTA -> TTA):")
    for k, a, b in rows:
        print(f"  {k:<20} {a:.4f} -> {b:.4f}")


def save_per_class(probs, labels):
    from sklearn.metrics import precision_recall_fscore_support
    preds = probs.argmax(1)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=list(range(config.NUM_CLASSES)), zero_division=0)
    m = em.classification_metrics(probs, labels)
    rows = []
    for i, c in enumerate(config.CLASSES):
        rows.append([c, round(p[i], 4), round(r[i], 4), round(f1[i], 4),
                     round(m["per_class_sensitivity"][i], 4),
                     round(m["per_class_specificity"][i], 4)])
    _write_csv(os.path.join(OUT, "metrics_per_class.csv"),
               ["class", "precision", "recall", "f1", "sensitivity",
                "specificity"], rows)


def save_confusion(probs, labels):
    m = em.classification_metrics(probs, labels)
    cm = m["confusion_matrix"]
    _write_csv(os.path.join(OUT, "confusion_matrix.csv"),
               [""] + config.CLASSES,
               [[config.CLASSES[i]] + list(map(int, cm[i])) for i in range(len(cm))])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(config.CLASSES)))
        ax.set_yticks(range(len(config.CLASSES)))
        ax.set_xticklabels(config.CLASSES, rotation=45, ha="right")
        ax.set_yticklabels(config.CLASSES)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for i in range(len(cm)):
            for j in range(len(cm)):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im); fig.tight_layout()
        fig.savefig(os.path.join(OUT, "confusion_matrix.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        print("  (matplotlib not available, skipped PNG:", e, ")")


def save_calibration(probs, logits, labels):
    cal = em.expected_calibration_error(probs, labels)
    # temperature scaling fit on the test logits (report honestly as such, or
    # refit on a validation split if you have one cached)
    T = em.fit_temperature(logits, labels, steps=300)
    probs_ts = em.apply_temperature(logits, T)
    cal_ts = em.expected_calibration_error(probs_ts, labels)
    _write_csv(os.path.join(OUT, "calibration.csv"),
               ["setting", "ECE", "MCE", "temperature"],
               [["raw", round(cal["ece"], 4), round(cal["mce"], 4), 1.0],
                ["temperature_scaled", round(cal_ts["ece"], 4),
                 round(cal_ts["mce"], 4), round(T, 3)]])
    print(f"\nCalibration: ECE {cal['ece']:.4f} -> {cal_ts['ece']:.4f} "
          f"(T={T:.3f})")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        conf = probs.max(1); correct = (probs.argmax(1) == labels).astype(float)
        bins = np.linspace(0, 1, 11)
        xs, ys = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mm = (conf > lo) & (conf <= hi)
            if mm.sum():
                xs.append(conf[mm].mean()); ys.append(correct[mm].mean())
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "k--", label="perfect")
        ax.plot(xs, ys, "o-", label="model")
        ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
        ax.set_title("Reliability diagram"); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "reliability.png"), dpi=150)
        plt.close(fig)
    except Exception:
        pass


def save_bootstrap(probs, labels):
    ci = em.bootstrap_ci(probs, labels, n_boot=10000)
    _write_csv(os.path.join(OUT, "bootstrap_ci.csv"),
               ["metric", "mean", "ci_low", "ci_high"],
               [["accuracy", round(ci["mean"], 4), round(ci["ci_low"], 4),
                 round(ci["ci_high"], 4)]])
    print(f"Accuracy 95% CI: {ci['mean']*100:.2f}% "
          f"[{ci['ci_low']*100:.2f}, {ci['ci_high']*100:.2f}]")


def save_xai_tables(model, paths, labels, subset):
    """Faithfulness + attention-consistency, averaged over a subset (these are
    slow, so we sample `subset` images stratified across classes)."""
    from src.xai import registry
    rng = np.random.default_rng(config.__dict__.get("RANDOM_SEED", 42))
    idx = []
    per = max(1, subset // config.NUM_CLASSES)
    for ci in range(config.NUM_CLASSES):
        cls_idx = np.where(labels == ci)[0]
        if len(cls_idx):
            idx.extend(rng.choice(cls_idx, min(per, len(cls_idx)), replace=False))
    idx = idx[:subset]

    methods = ["GradCAM", "GradCAM++", "EigenCAM", "ScoreCAM"]
    faith = {m: {"del": [], "ins": []} for m in methods}
    cons = {m: {"iou": [], "pearson": []} for m in methods}

    for k, i in enumerate(idx):
        batch = model_loader.preprocess_path(paths[i])
        res = registry.explain_all(model, batch, methods=methods,
                                   run_faithfulness=True, run_consistency=True,
                                   score_cam_channels=64)
        for m in methods:
            faith[m]["del"].append(res["faithfulness"][m]["deletion_auc"])
            faith[m]["ins"].append(res["faithfulness"][m]["insertion_auc"])
            cons[m]["iou"].append(res["consistency"][m]["iou"])
            cons[m]["pearson"].append(res["consistency"][m]["pearson"])
        print(f"  XAI {k+1}/{len(idx)}", end="\r")
    print()

    _write_csv(os.path.join(OUT, "xai_faithfulness.csv"),
               ["method", "deletion_auc_mean", "deletion_auc_std",
                "insertion_auc_mean", "insertion_auc_std"],
               [[m, round(np.mean(faith[m]["del"]), 4), round(np.std(faith[m]["del"]), 4),
                 round(np.mean(faith[m]["ins"]), 4), round(np.std(faith[m]["ins"]), 4)]
                for m in methods])
    _write_csv(os.path.join(OUT, "attention_consistency.csv"),
               ["method", "iou_mean", "iou_std", "pearson_mean", "pearson_std"],
               [[m, round(np.mean(cons[m]["iou"]), 4), round(np.std(cons[m]["iou"]), 4),
                 round(np.mean(cons[m]["pearson"]), 4), round(np.std(cons[m]["pearson"]), 4)]
                for m in methods])
    print(f"\nXAI tables saved (n={len(idx)} images).")
    print("Faithfulness (deletion-AUC lower=better):")
    for m in methods:
        print(f"  {m:<12} del {np.mean(faith[m]['del']):.4f}  "
              f"ins {np.mean(faith[m]['ins']):.4f}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    test_dir = args[0] if args else TEST_DIR
    xai_subset = 40
    if "--xai-subset" in sys.argv:
        xai_subset = int(sys.argv[sys.argv.index("--xai-subset") + 1])

    print("Test dir:", test_dir)
    paths, labels = load_test(test_dir)
    if not paths:
        print("No images found. Pass the correct test dir as the first argument.")
        return
    print(f"{len(paths)} test images across {config.NUM_CLASSES} classes")

    model = model_loader.load_model()

    print("\nPredicting (no TTA)...")
    probs_no, _ = predict_all(model, paths, tta=False)
    print("Predicting (TTA)...")
    probs_tta, logits_tta = predict_all(model, paths, tta=True)

    m_no = em.classification_metrics(probs_no, labels)
    m_tta = em.classification_metrics(probs_tta, labels)

    save_overall(m_no, m_tta)
    save_per_class(probs_tta, labels)
    save_confusion(probs_tta, labels)
    save_calibration(probs_tta, logits_tta, labels)
    save_bootstrap(probs_tta, labels)
    save_xai_tables(model, paths, labels, xai_subset)

    # McNemar: does TTA change predictions significantly?
    mc = em.mcnemar_test(labels, probs_no.argmax(1), probs_tta.argmax(1))
    print(f"\nMcNemar (no-TTA vs TTA): n01={mc['n01']} n10={mc['n10']} "
          f"p={mc['p_value']:.4f}")

    print(f"\nAll tables saved to {OUT}")


if __name__ == "__main__":
    main()
