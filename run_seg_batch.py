"""
Weakly supervised ROI extraction with SAM over the test set, evaluated by what
the ROI says about the classifier (there are no ground-truth masks).

For every image:
  - SAM ViT-B with a central box prompt (config.SAM_BOX_FRACTION), run on the
    full-resolution file, keep the best mask with 3-85% area; otherwise the box
    itself is used and the image is counted as a fallback
  - share of Grad-CAM / Grad-CAM++ mass inside the ROI, next to three
    references: the ROI's area fraction (what a uniform map would score), the
    share of a centered Gaussian inside the ROI, and the CAM's share inside the
    plain prompt box
  - re-classification with everything outside the ROI set to black, and set
    to a blurred copy of the image (black alone is out of distribution)
  - IoU between the SAM mask and the prompt box: close to 1 means SAM simply
    filled the box instead of finding the membrane
  - overlays for visual QA and a qa_sheet.csv for you to fill in

Rows are appended to outputs/seg/seg_per_image.csv as they finish; an
interrupted run resumes. Summaries are written at the end.

Usage (venv active, SAM installed, weights/sam_vit_b.pth present):
  python run_seg_batch.py "D:\\otoscope_intelligent_app\\test"
  python run_seg_batch.py "D:\\otoscope_intelligent_app\\test" --n 12   # quick test
  python run_seg_batch.py --summarize-only      # after filling qa_sheet.csv
"""
import os
import sys
import csv
import time
import argparse

import numpy as np
import cv2
import tensorflow as tf

from src import config
from src import model_loader
from src import test_data
from src.xai import gradcam, logit_cams, reference_maps, visualize
from src.segmentation import roi_analysis

OUT = os.path.join(config.OUT_DIR, "seg")
QA_DIR = os.path.join(OUT, "qa")
PER_IMAGE = os.path.join(OUT, "seg_per_image.csv")
QA_SHEET = os.path.join(OUT, "qa_sheet.csv")

HEADER = ["idx", "path", "true", "pred", "correct", "conf",
          "source", "sam_score", "area_frac", "mask_box_iou",
          "gc_in_roi", "gpp_in_roi", "center_in_roi",
          "gc_in_box", "gpp_in_box",
          "pred_black", "correct_black", "conf_black",
          "pred_blur", "correct_blur", "conf_blur"]


def _cams(model, batch, ci, score):
    if score == "logit":
        return (logit_cams.grad_cam_logit(model, batch, ci),
                logit_cams.grad_cam_pp_logit(model, batch, ci))
    return (gradcam.grad_cam(model, batch, class_index=ci)[0],
            gradcam.grad_cam_pp(model, batch, class_index=ci)[0])


def _box_mask(size, frac):
    m = np.zeros((size, size), dtype="float32")
    bw = int(size * frac)
    o = (size - bw) // 2
    m[o:o + bw, o:o + bw] = 1.0
    return m


def _predict(model, img):
    return model(tf.convert_to_tensor(img[None], dtype=tf.float32)).numpy()[0]


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


def run(test_dir, n, score, qa_total):
    from src.segmentation import sam_pseudomask     # fails clearly if SAM missing
    os.makedirs(QA_DIR, exist_ok=True)
    _check_run_config(score)
    paths, labels = test_data.list_images(test_dir)
    if not paths:
        sys.exit(f"No images found under {test_dir}")
    idx = test_data.stratified_subset(labels, n)
    qa_idx = set(test_data.stratified_subset(labels, qa_total, seed=7))
    print(f"{len(paths)} images found, evaluating {len(idx)}; "
          f"{len(qa_idx & set(idx))} overlays for visual QA")

    model = model_loader.load_model()
    S = config.IMG_SIZE
    box = _box_mask(S, config.SAM_BOX_FRACTION)
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
        w.writerow(HEADER)
    qa_rows = []

    for k, i in enumerate(idx):
        if i in done:
            continue
        t0 = time.time()
        bgr = cv2.imread(paths[i], cv2.IMREAD_COLOR)
        rgb_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        batch = model_loader.preprocess_bgr(bgr)            # no CLAHE: already applied
        img = batch[0]
        probs = _predict(model, img)
        ci = int(np.argmax(probs))

        mask_full, meta = sam_pseudomask.segment_roi(rgb_full)
        mask = cv2.resize(mask_full.astype("float32"), (S, S),
                          interpolation=cv2.INTER_NEAREST)
        gc, gpp = _cams(model, batch, ci, score)

        m3 = (mask > 0.5).astype("float32")[..., None]
        blur = cv2.GaussianBlur(img, (0, 0), sigmaX=15)
        p_black = _predict(model, img * m3)
        p_blur = _predict(model, img * m3 + blur * (1.0 - m3))

        area = float((mask > 0.5).mean())
        mb = mask > 0.5
        mbox = box > 0.5
        box_iou = float((mb & mbox).sum() / max(1, (mb | mbox).sum()))
        row = [i, paths[i], int(labels[i]), ci, int(ci == labels[i]),
               round(float(probs[ci]), 4), meta["source"],
               "" if meta["score"] is None else round(meta["score"], 4),
               round(area, 4), round(box_iou, 4),
               roi_analysis.xai_in_roi_fraction(gc, mask),
               roi_analysis.xai_in_roi_fraction(gpp, mask),
               roi_analysis.xai_in_roi_fraction(center, mask),
               roi_analysis.xai_in_roi_fraction(gc, box),
               roi_analysis.xai_in_roi_fraction(gpp, box),
               int(p_black.argmax()), int(p_black.argmax() == labels[i]),
               round(float(p_black[ci]), 4),
               int(p_blur.argmax()), int(p_blur.argmax() == labels[i]),
               round(float(p_blur[ci]), 4)]
        w.writerow(row)
        f.flush()

        if i in qa_idx:
            ov = visualize.mask_overlay(rgb_full, mask_full)
            h, wd = rgb_full.shape[:2]
            bw, bh = int(wd * config.SAM_BOX_FRACTION), int(h * config.SAM_BOX_FRACTION)
            x0, y0 = (wd - bw) // 2, (h - bh) // 2
            cv2.rectangle(ov, (x0, y0), (x0 + bw, y0 + bh), (255, 255, 0), 1)
            stem = os.path.splitext(os.path.basename(paths[i]))[0]
            name = f"{config.CLASSES[labels[i]]}_{stem}.png"
            cv2.imwrite(os.path.join(QA_DIR, name), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
            qa_rows.append([name, config.CLASSES[labels[i]], meta["source"],
                            round(area, 3), ""])
        print(f"  {k+1}/{len(idx)}  {os.path.basename(paths[i])}  "
              f"{meta['source']}  {time.time()-t0:.1f}s", flush=True)
    f.close()

    # never overwrite a QA sheet you may already have filled in
    if qa_rows and not os.path.exists(QA_SHEET):
        with open(QA_SHEET, "w", newline="") as q:
            qw = csv.writer(q)
            qw.writerow(["file", "class", "source", "area_frac",
                         "acceptable (Y/N)"])
            qw.writerows(qa_rows)
        print(f"\nQA sheet: {QA_SHEET}  (open each image in {QA_DIR}, "
              f"write Y or N in the last column)")


def summarize():
    if not os.path.exists(PER_IMAGE):
        sys.exit(f"No results yet: {PER_IMAGE} does not exist.")
    with open(PER_IMAGE, newline="") as f:
        rows = list(csv.DictReader(f))

    def col(k):
        return np.array([float(r[k]) for r in rows])

    def ms(v):
        return round(float(v.mean()), 4), round(float(v.std()), 4)

    n = len(rows)
    fb = sum(r["source"] == "box_fallback" for r in rows)
    out = [["n_images", n, ""],
           ["sam_fallback_count", fb, ""],
           ["sam_fallback_rate", round(fb / n, 4), ""],
           ["roi_area_frac (= uniform-map baseline)", *ms(col("area_frac"))],
           ["mask_vs_box_iou", *ms(col("mask_box_iou"))],
           ["share_masks_nearly_the_box (IoU>0.9)", round(float((col("mask_box_iou") > 0.9).mean()), 4), ""],
           ["gradcam_mass_in_roi", *ms(col("gc_in_roi"))],
           ["gradcampp_mass_in_roi", *ms(col("gpp_in_roi"))],
           ["center_prior_mass_in_roi", *ms(col("center_in_roi"))],
           ["gradcam_mass_in_prompt_box", *ms(col("gc_in_box"))],
           ["gradcampp_mass_in_prompt_box", *ms(col("gpp_in_box"))],
           ["accuracy_full_image", round(float(col("correct").mean()), 4), ""],
           ["accuracy_roi_black_background", round(float(col("correct_black").mean()), 4), ""],
           ["accuracy_roi_blurred_background", round(float(col("correct_blur").mean()), 4), ""],
           ["conf_full (pred class)", *ms(col("conf"))],
           ["conf_roi_black (same class)", *ms(col("conf_black"))],
           ["conf_roi_blur (same class)", *ms(col("conf_blur"))]]
    if os.path.exists(QA_SHEET):
        with open(QA_SHEET, newline="") as q:
            qa = list(csv.DictReader(q))
        key = next((k for k in (qa[0] if qa else {}) if k.startswith("acceptable")), None)
        filled = [r for r in qa if key and r[key].strip().upper() in ("Y", "N")]
        if filled:
            yes = sum(r[key].strip().upper() == "Y" for r in filled)
            out.append(["visual_QA_acceptable", f"{yes}/{len(filled)}", ""])
        else:
            out.append(["visual_QA_acceptable", "qa_sheet.csv not filled yet", ""])
    with open(os.path.join(OUT, "seg_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "value_or_mean", "std"])
        w.writerows(out)
    print(f"\nSegmentation summary ({os.path.join(OUT, 'seg_summary.csv')}):")
    for q, v, s in out:
        print(f"  {q:<42} {v} {'+- ' + str(s) if s != '' else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("test_dir", nargs="?")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--score", choices=["logit", "prob"], default="logit")
    ap.add_argument("--qa", type=int, default=60, help="overlays to save for visual QA")
    ap.add_argument("--summarize-only", action="store_true")
    a = ap.parse_args()
    if not a.summarize_only:
        if not a.test_dir:
            ap.error("test_dir is required unless --summarize-only")
        run(a.test_dir, a.n, a.score, a.qa)
    summarize()


if __name__ == "__main__":
    main()
