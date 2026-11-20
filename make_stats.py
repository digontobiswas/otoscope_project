"""
Statistical analysis and statistical figures for the paper. Reads the
per-image CSVs produced by run_xai_batch.py and run_seg_batch.py, plus the
confusion matrices recorded in docs/RESULTS_LOG.md (Entries 3 and 4).

Writes:
  outputs/stats/classification_stats.csv   accuracy + Wilson CI, balanced acc.,
                                           macro-F1 / kappa / MCC with bootstrap CIs
  outputs/stats/protocol_tests.csv         leaky vs leak-free and model-vs-model tests
  outputs/stats/xai_tests.csv              Friedman omnibus, Holm-corrected paired
                                           Wilcoxon, effect sizes, bootstrap CIs
  outputs/stats/roi_tests.csv              McNemar tests for background removal
  outputs/figures/fig_leakage.png          accuracy with 95% CIs; per-class recall A vs CV
  outputs/figures/fig_cv_confusion.png     row-normalised CV confusion matrices
  outputs/figures/fig_faithfulness_box.png distributions of deletion / insertion AUC

Nothing is selected or tuned here; it only describes results already measured.

Usage:  python make_stats.py
"""
import os
import csv

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from src import config
    OUT = config.OUT_DIR
except Exception:
    OUT = "outputs"
STATS = os.path.join(OUT, "stats")
FIGS = os.path.join(OUT, "figures")
os.makedirs(STATS, exist_ok=True); os.makedirs(FIGS, exist_ok=True)

CLASSES = ["AOM", "CSOM", "Earwax", "Myringoscl.", "Normal", "Otitis ext."]
# rows = true, cols = predicted (docs/RESULTS_LOG.md)
CM = {
    "CBAM, Protocol A": np.array([[31, 1, 0, 0, 6, 1], [0, 41, 0, 0, 0, 0], [0, 0, 40, 0, 0, 0],
                                  [0, 0, 0, 30, 0, 0], [3, 0, 1, 1, 69, 0], [0, 0, 0, 0, 0, 39]]),
    "CBAM, Protocol B (CV)": np.array([[35, 2, 2, 0, 28, 2], [5, 3, 0, 0, 4, 2], [6, 1, 5, 0, 8, 1],
                                       [0, 0, 0, 0, 3, 1], [17, 0, 1, 0, 135, 1], [4, 1, 2, 0, 9, 2]]),
    "Baseline re-impl., Protocol B (CV)": np.array([[40, 0, 1, 0, 28, 0], [7, 0, 0, 0, 6, 1], [7, 0, 6, 0, 6, 2],
                                                    [0, 0, 0, 0, 4, 0], [33, 0, 0, 0, 121, 0], [9, 0, 2, 0, 5, 2]]),
}
BASE_A = (256, 263)        # baseline re-implementation, Protocol A (correct, n)

# palette: validated categorical pair + neutral inks
C_OURS, C_BASE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, REF = "#0b0b0b", "#52514e", "#dddcd6", "#8a8984"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "figure.dpi": 100})


# ------------------------------------------------------------------ helpers
def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def cm_metrics(cm):
    n = cm.sum(); po = np.trace(cm) / n
    rec = np.diag(cm) / np.maximum(cm.sum(1), 1)
    pre = np.diag(cm) / np.maximum(cm.sum(0), 1)
    f1 = np.where(pre + rec > 0, 2 * pre * rec / np.maximum(pre + rec, 1e-12), 0.0)
    pe = (cm.sum(0) * cm.sum(1)).sum() / n ** 2
    kappa = (po - pe) / (1 - pe)
    t, p_ = cm.sum(1).astype(float), cm.sum(0).astype(float)
    c = np.trace(cm)
    den = np.sqrt(n ** 2 - (p_ ** 2).sum()) * np.sqrt(n ** 2 - (t ** 2).sum())
    mcc = (c * n - (t * p_).sum()) / den if den > 0 else 0.0
    return {"acc": po, "bal_acc": rec.mean(), "f1": f1.mean(), "kappa": kappa, "mcc": mcc}


def boot_cm(cm, n_boot=10000, seed=0):
    """Bootstrap the (true, predicted) pairs behind a confusion matrix."""
    rng = np.random.default_rng(seed)
    k = cm.shape[0]; flat = cm.ravel(); n = flat.sum()
    draws = rng.multinomial(n, flat / n, size=n_boot).reshape(n_boot, k, k)
    out = {m: [] for m in ["bal_acc", "f1", "kappa", "mcc"]}
    for d in draws:
        r = cm_metrics(d)
        for m in out:
            out[m].append(r[m])
    return {m: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for m, v in out.items()}


def cohens_h(p1, p2):
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))


def holm(pvals):
    p = np.asarray(pvals, float); order = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def rank_biserial(a, b):
    """Matched-pairs rank-biserial correlation for a - b (range -1..1)."""
    d = a - b; d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def boot_mean_diff(a, b, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed); d = a - b
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    m = d[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def write(name, header, rows):
    with open(os.path.join(STATS, name), "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f"-- {name}")
    for r in rows:
        print("   ", r)


# ------------------------------------------------------------------ 1. classification
rows = []
for name, cm in CM.items():
    n, k = int(cm.sum()), int(np.trace(cm))
    lo, hi = wilson(k, n); m = cm_metrics(cm); b = boot_cm(cm)
    rows.append([name, n, k, round(m["acc"], 4), round(lo, 4), round(hi, 4),
                 round(m["bal_acc"], 4), *[round(x, 4) for x in b["bal_acc"]],
                 round(m["f1"], 4), *[round(x, 4) for x in b["f1"]],
                 round(m["kappa"], 4), *[round(x, 4) for x in b["kappa"]],
                 round(m["mcc"], 4), *[round(x, 4) for x in b["mcc"]]])
write("classification_stats.csv",
      ["setting", "n", "correct", "accuracy", "acc_lo", "acc_hi", "balanced_acc", "ba_lo", "ba_hi",
       "macro_f1", "f1_lo", "f1_hi", "kappa", "kappa_lo", "kappa_hi", "mcc", "mcc_lo", "mcc_hi"], rows)

a, b_, c_ = CM["CBAM, Protocol A"], CM["CBAM, Protocol B (CV)"], CM["Baseline re-impl., Protocol B (CV)"]
ka, na, kb, nb, kc, nc = np.trace(a), a.sum(), np.trace(b_), b_.sum(), np.trace(c_), c_.sum()


def two_prop(k1, n1, k2, n2):
    p = (k1 + k2) / (n1 + n2)
    z = (k1 / n1 - k2 / n2) / np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return float(z), float(2 * stats.norm.sf(abs(z)))


tests = []
z, p = two_prop(ka, na, kb, nb)
tests.append(["CBAM: Protocol A vs Protocol B (CV)", f"{ka}/{na} vs {kb}/{nb}", "two-proportion z (independent samples)",
              round(z, 2), f"{p:.2g}", round(cohens_h(ka / na, kb / nb), 3)])
z, p = two_prop(BASE_A[0], BASE_A[1], kc, nc)
tests.append(["Baseline: Protocol A vs Protocol B (CV)", f"{BASE_A[0]}/{BASE_A[1]} vs {kc}/{nc}",
              "two-proportion z (independent samples)", round(z, 2), f"{p:.2g}",
              round(cohens_h(BASE_A[0] / BASE_A[1], kc / nc), 3)])
z, p = two_prop(kb, nb, kc, nc)
tests.append(["Protocol B (CV): CBAM vs baseline", f"{kb}/{nb} vs {kc}/{nc}",
              "two-proportion z (unpaired; same images, paired data not available)", round(z, 2), f"{p:.2g}",
              round(cohens_h(kb / nb, kc / nc), 3)])
z, p = two_prop(ka, na, BASE_A[0], BASE_A[1])
tests.append(["Protocol A: CBAM vs baseline", f"{ka}/{na} vs {BASE_A[0]}/{BASE_A[1]}",
              "two-proportion z (unpaired; same images, paired data not available)", round(z, 2), f"{p:.2g}",
              round(cohens_h(ka / na, BASE_A[0] / BASE_A[1]), 3)])
maj = b_.sum(1).max() / nb
for nm, k in [("CBAM", kb), ("Baseline", kc)]:
    pv = stats.binomtest(int(k), int(nb), maj, alternative="greater").pvalue
    tests.append([f"Protocol B (CV): {nm} vs majority class ({maj:.3f})", f"{k}/{nb}",
                  "exact binomial, one-sided", "", f"{pv:.2g}", round(cohens_h(k / nb, maj), 3)])
write("protocol_tests.csv", ["comparison", "counts", "test", "z", "p", "cohens_h"], tests)

# ------------------------------------------------------------------ 2. XAI
xai_csv = os.path.join(OUT, "xai", "xai_per_image.csv")
X = None
if os.path.exists(xai_csv):
    with open(xai_csv, newline="") as f:
        R = list(csv.DictReader(f))
    col = lambda k: np.array([float(r[k]) for r in R])
    CAMS = ["GradCAM", "GradCAM++", "EigenCAM", "ScoreCAM"]
    X = {m: {"del": col(f"{m}|del"), "ins": col(f"{m}|ins")}
         for m in CAMS + ["CenterPrior", "Random", "RandModel-GradCAM"]}
    out = []
    for met in ["del", "ins"]:
        chi, p = stats.friedmanchisquare(*[X[m][met] for m in CAMS])
        out.append([f"Friedman, 4 CAMs, {met}", "", "", round(float(chi), 2), f"{p:.2g}", "", "", "", ""])
    raw, keys = [], []
    for ref in ["CenterPrior", "Random", "RandModel-GradCAM"]:
        for cam in CAMS:
            for met in ["del", "ins"]:
                raw.append(stats.wilcoxon(X[cam][met], X[ref][met]).pvalue); keys.append((cam, ref, met))
    for fam in ["CenterPrior", "Random", "RandModel-GradCAM"]:
        ix = [i for i, k in enumerate(keys) if k[1] == fam]
        adj = holm([raw[i] for i in ix])
        for i, pa in zip(ix, adj):
            cam, ref, met = keys[i]
            d = X[cam][met] - X[ref][met]
            lo, hi = boot_mean_diff(X[cam][met], X[ref][met])
            better = (d < 0).mean() if met == "del" else (d > 0).mean()
            out.append([cam, ref, met, round(float(d.mean()), 4), f"{raw[i]:.2g}", f"{pa:.2g}",
                        round(rank_biserial(X[cam][met], X[ref][met]), 3),
                        f"[{lo:.4f}, {hi:.4f}]", round(100 * float(better), 1)])
    write("xai_tests.csv", ["cam_or_test", "reference", "metric", "mean_diff_or_chi2", "p_raw", "p_holm",
                            "rank_biserial", "boot95_mean_diff", "pct_images_cam_better"], out)

# ------------------------------------------------------------------ 3. ROI
seg_csv = os.path.join(OUT, "seg", "seg_per_image.csv")
if os.path.exists(seg_csv):
    with open(seg_csv, newline="") as f:
        S = list(csv.DictReader(f))
    full = np.array([r["correct"] == "1" for r in S])
    out = []
    for key, nm in [("correct_blur", "blurred background"), ("correct_black", "black background")]:
        alt = np.array([r[key] == "1" for r in S])
        n10, n01 = int((full & ~alt).sum()), int((~full & alt).sum())
        p = stats.binomtest(min(n10, n01), n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
        lo, hi = wilson(int(alt.sum()), len(alt))
        out.append([f"full vs {nm}", int(full.sum()), int(alt.sum()), len(alt), n10, n01, f"{p:.2g}",
                    round(lo, 4), round(hi, 4)])
    write("roi_tests.csv", ["comparison", "correct_full", "correct_masked", "n", "full_right_masked_wrong",
                            "full_wrong_masked_right", "mcnemar_exact_p", "masked_acc_lo", "masked_acc_hi"], out)

# ------------------------------------------------------------------ figures
# Fig: leakage. (a) accuracy with 95% Wilson CIs, (b) per-class recall, CBAM, A vs CV
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1, 1.2]})
groups = ["Protocol A\n(leaky)", "Protocol B, 4-fold CV\n(leak-free)"]
vals = {"CBAM-MobileNetV3 (1.03 M)": [(ka, na), (kb, nb)],
        "Baseline re-impl. (16.1 M)": [BASE_A, (kc, nc)]}
majority = [a.sum(1).max() / na, maj]
w = 0.34
for j, ((name, kn), colr) in enumerate(zip(vals.items(), [C_OURS, C_BASE])):
    for g, (k, n) in enumerate(kn):
        p = k / n; lo, hi = wilson(k, n); x = g + (j - 0.5) * (w + 0.03)
        ax1.bar(x, 100 * p, w, color=colr, label=name if g == 0 else None, zorder=2)
        ax1.errorbar(x, 100 * p, yerr=[[100 * (p - lo)], [100 * (hi - p)]], color=INK, lw=0.9, capsize=2.5, zorder=3)
        ax1.text(x, 100 * hi + 2, f"{100*p:.1f}", ha="center", va="bottom", fontsize=7.5, color=INK)
for g, m in enumerate(majority):
    ax1.plot([g - 0.42, g + 0.42], [100 * m, 100 * m], color=REF, lw=1.2, ls=(0, (4, 2)), zorder=4,
             label="Majority-class rate" if g == 0 else None)
ax1.set_xticks([0, 1]); ax1.set_xticklabels(groups); ax1.set_ylim(0, 112); ax1.set_ylabel("Accuracy (%)")
ax1.set_yticks([0, 25, 50, 75, 100]); ax1.yaxis.grid(True, color=GRID, lw=0.6, zorder=0); ax1.set_axisbelow(True)
ax1.legend(frameon=False, fontsize=6.8, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1)
ax1.set_title("(a) Accuracy, 95% confidence intervals", fontsize=8.5, color=INK, loc="left")

rA = np.diag(a) / a.sum(1); rB = np.diag(b_) / b_.sum(1)
y = np.arange(len(CLASSES))[::-1]
for yi, ra, rb, nB in zip(y, rA, rB, b_.sum(1)):
    ax2.plot([rb, ra], [yi, yi], color=GRID, lw=2.2, zorder=1, solid_capstyle="round")
    lo, hi = wilson(int(round(rb * nB)), int(nB))
    ax2.plot([lo, hi], [yi - 0.18, yi - 0.18], color=C_OURS, lw=0.9, zorder=2)
ax2.scatter(rA, y, s=34, color="white", edgecolor=C_OURS, linewidth=1.4, zorder=3, label="Protocol A")
ax2.scatter(rB, y, s=34, color=C_OURS, zorder=3, label="Protocol B, 4-fold CV (line: 95% CI)")
ax2.set_yticks(y); ax2.set_yticklabels([f"{c} (n={n})" for c, n in zip(CLASSES, b_.sum(1))])
ax2.set_xlim(-0.04, 1.06); ax2.set_xlabel("Recall"); ax2.set_ylim(-0.6, len(CLASSES) - 0.4)
ax2.xaxis.grid(True, color=GRID, lw=0.6, zorder=0); ax2.set_axisbelow(True)
ax2.legend(frameon=False, fontsize=6.8, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1)
ax2.set_title("(b) Per-class recall, CBAM-MobileNetV3", fontsize=8.5, color=INK, loc="left")
fig.tight_layout(w_pad=1.6)
fig.savefig(os.path.join(FIGS, "fig_leakage.png"), dpi=300, facecolor="white"); plt.close(fig)

# Fig: row-normalised CV confusion matrices (sequential single hue)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
for ax, (name, cm) in zip(axes, [("CBAM-MobileNetV3", b_), ("Baseline re-impl.", c_)]):
    norm = cm / cm.sum(1, keepdims=True)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(6):
        for j in range(6):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7.5,
                        color="white" if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xticklabels(CLASSES, rotation=40, ha="right"); ax.set_yticklabels(CLASSES if ax is axes[0] else [])
    ax.set_xlabel("Predicted"); ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_title(f"{name}: {np.trace(cm)}/{cm.sum()} correct", fontsize=8.5, color=INK)
axes[0].set_ylabel("True")
cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02); cb.set_label("Share of true class", fontsize=7.5)
cb.outline.set_visible(False)
fig.savefig(os.path.join(FIGS, "fig_cv_confusion.png"), dpi=300, facecolor="white", bbox_inches="tight"); plt.close(fig)

# Fig: distributions of deletion / insertion AUC
if X is not None:
    order = ["GradCAM", "GradCAM++", "ScoreCAM", "EigenCAM", "CenterPrior", "Random", "RandModel-GradCAM"]
    labels = ["Grad-\nCAM", "Grad-\nCAM++", "Score-\nCAM", "Eigen-\nCAM", "Center\nprior", "Random\nmap", "Grad-CAM,\nrand. wts"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, met, ttl in zip(axes, ["del", "ins"], ["(a) Deletion AUC (lower is better)",
                                                   "(b) Insertion AUC (higher is better)"]):
        data = [X[m][met] for m in order]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=True,
                        medianprops=dict(color=INK, lw=1.2), whiskerprops=dict(color=MUTED, lw=0.8),
                        capprops=dict(color=MUTED, lw=0.8), boxprops=dict(linewidth=0.8),
                        flierprops=dict(marker="o", markersize=1.6, markerfacecolor=MUTED, markeredgecolor="none", alpha=0.5))
        for k, box in enumerate(bp["boxes"]):
            box.set_facecolor(C_OURS if k < 4 else "#c9c8c2"); box.set_edgecolor(MUTED); box.set_alpha(0.85 if k < 4 else 1)
        ax.scatter(range(1, len(order) + 1), [d.mean() for d in data], marker="D", s=11, color="white",
                   edgecolor=INK, linewidth=0.8, zorder=4)
        ax.axhline(np.median(X["CenterPrior"][met]), color=REF, lw=0.9, ls=(0, (4, 2)), zorder=0)
        ax.axvline(4.5, color=GRID, lw=0.8)
        ax.set_xticks(range(1, len(order) + 1)); ax.set_xticklabels(labels, fontsize=6.6)
        ax.set_ylim(-0.02, 1.12); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.set_title(ttl, fontsize=8.5, color=INK, loc="left")
        ax.text(2.5, 1.11, "explanations", ha="center", va="top", fontsize=6.8, color=MUTED)
        ax.text(6.0, 1.11, "reference maps", ha="center", va="top", fontsize=6.8, color=MUTED)
    fig.tight_layout(w_pad=1.4)
    fig.savefig(os.path.join(FIGS, "fig_faithfulness_box.png"), dpi=300, facecolor="white"); plt.close(fig)

print("\nFigures saved to", FIGS)
