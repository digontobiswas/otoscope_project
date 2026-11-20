"""
Evaluation metrics for the paper (Section: Model Evaluation + Statistical
Validation). Operates on cached probability matrices so it is fast and
re-runnable, and needs no GPU.

Provides:
  - full classification suite: accuracy, per-class + macro precision/recall/F1,
    sensitivity, specificity, macro-AUC, MCC, Cohen's kappa
  - calibration: ECE, MCE, plus temperature-scaling fit
  - statistical validation: McNemar (paired model comparison), bootstrap CI,
    Wilcoxon signed-rank across folds
"""
import numpy as np
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score, matthews_corrcoef, cohen_kappa_score)
from scipy import stats

from src import config

NUM_CLASSES = config.NUM_CLASSES


# ---------------------------------------------------------------- classification
def classification_metrics(probs, labels):
    preds = probs.argmax(1)
    acc = float((preds == labels).mean())
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=list(range(NUM_CLASSES)), average="macro",
        zero_division=0)
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))

    sens, spec = [], []
    for i in range(NUM_CLASSES):
        tp = cm[i, i]; fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp; tn = cm.sum() - tp - fn - fp
        sens.append(tp / (tp + fn) if (tp + fn) else np.nan)
        spec.append(tn / (tn + fp) if (tn + fp) else np.nan)

    try:
        y1h = np.eye(NUM_CLASSES)[labels]
        auc = float(roc_auc_score(y1h, probs, average="macro", multi_class="ovr"))
    except Exception:
        auc = np.nan

    return {
        "accuracy": acc, "precision_macro": float(p), "recall_macro": float(r),
        "f1_macro": float(f1), "sensitivity_macro": float(np.nanmean(sens)),
        "specificity_macro": float(np.nanmean(spec)), "auc_macro": auc,
        "mcc": float(matthews_corrcoef(labels, preds)),
        "cohen_kappa": float(cohen_kappa_score(labels, preds)),
        "confusion_matrix": cm,
        "per_class_sensitivity": sens, "per_class_specificity": spec,
    }


# ---------------------------------------------------------------- calibration
def expected_calibration_error(probs, labels, n_bins=15):
    conf = probs.max(1)
    preds = probs.argmax(1)
    correct = (preds == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece, mce = 0.0, 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        gap = abs(correct[m].mean() - conf[m].mean())
        ece += (m.mean()) * gap
        mce = max(mce, gap)
    return {"ece": float(ece), "mce": float(mce)}


def fit_temperature(logits_val, labels_val, lr=0.01, steps=300):
    """Fit a single temperature T on validation logits (T>1 softens). Returns T.
    logits_val: pre-softmax scores (N, C)."""
    import tensorflow as tf
    T = tf.Variable(1.0, dtype=tf.float32)
    x = tf.constant(logits_val, dtype=tf.float32)
    y = tf.constant(labels_val, dtype=tf.int32)
    opt = tf.optimizers.Adam(lr)
    for _ in range(steps):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(y, x / T))
        g = tape.gradient(loss, [T])
        opt.apply_gradients(zip(g, [T]))
    return float(T.numpy())


def apply_temperature(logits, T):
    import tensorflow as tf
    return tf.nn.softmax(logits / T, axis=1).numpy()


# ---------------------------------------------------------------- statistics
def mcnemar_test(labels, preds_a, preds_b):
    """Paired comparison of two models on the same test items. Returns the
    statistic and p-value (exact binomial for small discordant counts)."""
    a_correct = preds_a == labels
    b_correct = preds_b == labels
    n01 = int(np.sum(~a_correct & b_correct))   # A wrong, B right
    n10 = int(np.sum(a_correct & ~b_correct))   # A right, B wrong
    n = n01 + n10
    if n == 0:
        return {"n01": n01, "n10": n10, "p_value": 1.0, "statistic": 0.0}
    # exact two-sided binomial test on the discordant pairs
    p = float(stats.binomtest(min(n01, n10), n, 0.5).pvalue)
    stat = (abs(n01 - n10) - 1) ** 2 / n        # continuity-corrected chi-square
    return {"n01": n01, "n10": n10, "statistic": float(stat), "p_value": p}


def bootstrap_ci(probs, labels, metric="accuracy", n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    N = len(labels)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, N, N)
        if metric == "accuracy":
            vals.append((probs[idx].argmax(1) == labels[idx]).mean())
        else:
            vals.append(classification_metrics(probs[idx], labels[idx])[metric])
    vals = np.array(vals)
    return {"mean": float(vals.mean()),
            "ci_low": float(np.percentile(vals, 2.5)),
            "ci_high": float(np.percentile(vals, 97.5))}


def wilcoxon_folds(scores_a, scores_b):
    """Paired Wilcoxon signed-rank across k folds (e.g. per-fold accuracy of two
    methods). Returns statistic and p-value."""
    stat, p = stats.wilcoxon(scores_a, scores_b)
    return {"statistic": float(stat), "p_value": float(p)}
