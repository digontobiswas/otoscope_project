"""
One PDF per image for an UNLABELLED folder: 001.pdf, 002.pdf, ...

Each PDF is a single page for the clinician:
    1. the original image, untouched, with the predicted class (main content)
    2. Grad-CAM heatmap for the predicted class
    3. SAM region of interest
    4. an agree / disagree line

Nothing is trained and nothing is scored: there are no ground-truth labels, so
this reports predictions only. Accuracy on these images can only be computed
after the doctor has labelled them.

Run from the project root:

    python predict_folder.py image
    python predict_folder.py image --no-seg        # skip SAM (much faster)
    python predict_folder.py image --blinded       # also write image-only PDFs
    python predict_folder.py some_folder --input-is-clahe

Outputs, in outputs/doctor_review/<folder name>/ :

    001.pdf, 002.pdf, ...   one page per image, numbered in file-name order
    predictions.csv         id, file, predicted class, all six scores, ROI numbers
    run_info.txt            weights and preprocessing used, class counts
    blinded/001.pdf, ...    (--blinded) original image + blank diagnosis line

The original files are never modified. Raw input (the default) follows the
dataset pipeline and gradio_app._prepare_raw: resize to 500x500 (INTER_AREA)
-> CLAHE once -> 224 NEAREST. Pass --input-is-clahe for images that already
went through pipeline step 02, so CLAHE is not applied twice.
"""
import io
import os
import re
import csv
import sys
import argparse
import datetime

import numpy as np
from PIL import Image, ImageOps

from src import config

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
PREP_SIZE = 500               # pipeline step 01 size; CLAHE was applied at this size
EMBED_MAX_SIDE = 1400         # px; larger non-JPEG originals are downscaled in the PDF
XAI_METHOD = "Grad-CAM (class logit)"


# ----------------------------------------------------------------- file listing
def _natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_images(root):
    """All image files under root (recursive), as paths relative to root."""
    found = []
    for folder, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                found.append(os.path.relpath(os.path.join(folder, f), root))
    return sorted(found, key=_natural_key)


def read_bgr(path):
    """cv2.imread fails on some non-ASCII Windows paths, so decode from bytes."""
    import cv2
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


# ---------------------------------------------------------------- preprocessing
def prepare(bgr, raw_input):
    """-> (network batch 1x224x224x3, RGB image the overlays are drawn on).

    Raw input: 500x500 resize, then CLAHE exactly once (L channel of LAB,
    clipLimit 2.0, 8x8 tiles: the pipeline step 02 parameters). Input that is
    already CLAHE-processed is passed through untouched."""
    import cv2
    from src import model_loader
    if raw_input:
        bgr = cv2.resize(bgr, (PREP_SIZE, PREP_SIZE), interpolation=cv2.INTER_AREA)
        l, a, b = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        bgr = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
    batch = model_loader.preprocess_bgr(bgr, apply_clahe=False)
    return batch, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------- analysis
def analyse_image(model, path, raw_input, use_seg):
    """Classification first; the heatmap and the ROI are extras, so a failure
    in either is recorded on the page and never blocks the prediction."""
    import cv2
    from src.xai import visualize

    bgr = read_bgr(path)
    if bgr is None:
        return None
    batch, rgb_full = prepare(bgr, raw_input)

    probs = model.predict(batch, verbose=0)[0]
    order = np.argsort(probs)[::-1]
    ci = int(order[0])
    rec = {
        "pred": config.CLASSES[ci], "score": float(probs[ci]),
        "second": config.CLASSES[order[1]], "second_score": float(probs[order[1]]),
        "probs": [float(v) for v in probs],
        "cam_img": None, "cam_error": "", "seg_img": None, "seg_error": "",
        "roi_source": "", "roi_area_frac": None, "cam_in_roi": None,
    }

    cam = None
    try:
        from src.xai import logit_cams
        cam = logit_cams.grad_cam_logit(model, batch, ci)
        rec["cam_img"] = visualize.overlay_heatmap(rgb_full, cam)
    except Exception as e:
        rec["cam_error"] = str(e)[:150]

    if use_seg:
        try:
            from src.segmentation import sam_pseudomask, roi_analysis
            mask_full, meta = sam_pseudomask.segment_roi(rgb_full)
            rec["seg_img"] = visualize.mask_overlay(rgb_full, mask_full)
            rec["roi_source"] = meta["source"]
            rec["roi_area_frac"] = float(meta["area_frac"])
            if cam is not None:
                mask = cv2.resize(mask_full.astype("float32"),
                                  (config.IMG_SIZE, config.IMG_SIZE),
                                  interpolation=cv2.INTER_NEAREST)
                rec["cam_in_roi"] = roi_analysis.xai_in_roi_fraction(cam, mask)
        except Exception as e:
            rec["seg_error"] = str(e).strip().splitlines()[0][:150]
    else:
        rec["seg_error"] = "skipped (--no-seg)"
    return rec


# -------------------------------------------------------------------------- PDF
def clinical_name(cls):
    return config.CLASS_DESCRIPTIONS.get(cls, cls)


def _original_for_pdf(path):
    """An upright JPEG is passed by path, so reportlab embeds its original
    bytes (no re-compression). Anything else is decoded, rotated upright and
    capped in size."""
    from reportlab.lib.utils import ImageReader
    img = Image.open(path)
    upright = img.getexif().get(0x0112, 1) in (1, None)
    if img.format == "JPEG" and img.mode == "RGB" and upright:
        return path
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((EMBED_MAX_SIDE, EMBED_MAX_SIDE))
    return ImageReader(img)


def _array_for_pdf(rgb):
    from reportlab.lib.utils import ImageReader
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=92)
    buf.seek(0)
    return ImageReader(buf)


def _fit(c, text, font, size, max_width):
    """Shorten text in the middle until it fits max_width."""
    if c.stringWidth(text, font, size) <= max_width:
        return text
    keep = len(text)
    while keep > 8:
        keep -= 2
        short = text[:keep // 2] + "..." + text[-(keep // 2):]
        if c.stringWidth(short, font, size) <= max_width:
            return short
    return text[:8]


def _panel(c, x, top, size, image, title, lines, missing):
    """Square image panel with a title above and small grey lines below."""
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x, top + 5, title)
    if image is not None:
        c.drawImage(image, x, top - size, width=size, height=size,
                    preserveAspectRatio=True, anchor="c")
    else:
        c.setStrokeColorRGB(0.75, 0.75, 0.75)
        c.rect(x, top - size, size, size)
        c.setFillColorRGB(0.45, 0.45, 0.45)
        c.setFont("Helvetica", 8)
        c.drawCentredString(x + size / 2, top - size / 2 + 4, "not available")
        c.drawCentredString(x + size / 2, top - size / 2 - 8,
                            _fit(c, missing, "Helvetica", 8, size - 12))
    c.setFillColorRGB(0.35, 0.35, 0.35)
    c.setFont("Helvetica", 7.5)
    for k, line in enumerate(lines):
        c.drawString(x, top - size - 11 - 10 * k, line)


def write_page(rec, src_path, out_path, source_name):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    W, H = A4
    m = 40
    grey = (0.35, 0.35, 0.35)
    c = canvas.Canvas(out_path, pagesize=A4)
    c.setTitle(f"Otoscope image {rec['id']}")

    # header
    c.setFont("Helvetica-Bold", 14)
    c.drawString(m, H - 42, f"Otoscope image {rec['id']}")
    c.setFillColorRGB(*grey)
    c.setFont("Helvetica", 8)
    c.drawRightString(W - m, H - 42, _fit(c, f"{source_name} / {rec['file']}",
                                          "Helvetica", 8, W - 2 * m - 170))
    c.setStrokeColorRGB(0.8, 0.8, 0.8)
    c.line(m, H - 50, W - m, H - 50)

    # 1. predicted class: the main content of the page
    c.setFont("Helvetica", 9)
    c.drawString(m, H - 70, "Predicted class (AI model, to be verified):")
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 19)
    c.drawString(m, H - 93, f"{clinical_name(rec['pred'])} ({rec['pred']})")
    c.setFillColorRGB(*grey)
    c.setFont("Helvetica", 9)
    c.drawString(m, H - 109, f"model score {rec['score']:.2f}      second choice: "
                 f"{clinical_name(rec['second'])} {rec['second_score']:.2f}")

    # original image, untouched
    size = 300
    top = H - 138
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(m, top + 5, "Original image (unmodified)")
    c.drawImage(_original_for_pdf(src_path), m, top - size, width=size,
                height=size, preserveAspectRatio=True, anchor="c")

    # all six scores, beside the image
    x = m + size + 22
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x, top + 5, "Model score per class")
    bar_w = W - m - x
    y = top - 16
    for cls, p in sorted(zip(config.CLASSES, rec["probs"]), key=lambda t: -t[1]):
        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica", 8.5)
        c.drawString(x, y, clinical_name(cls))
        c.drawRightString(x + bar_w, y, f"{p:.2f}")
        c.setFillColorRGB(0.9, 0.9, 0.9)
        c.rect(x, y - 9, bar_w, 4, stroke=0, fill=1)
        c.setFillColorRGB(0.2, 0.35, 0.6)
        c.rect(x, y - 9, bar_w * float(p), 4, stroke=0, fill=1)
        y -= 30
    c.setFillColorRGB(*grey)
    c.setFont("Helvetica", 7.5)
    for k, line in enumerate(["Scores are uncalibrated and always sum",
                              "to 1 over these six classes, so the model",
                              "cannot answer 'other' or 'not gradable'."]):
        c.drawString(x, y - 6 - 10 * k, line)

    # 2 + 3. heatmap and region of interest
    psize = 236
    ptop = top - size - 34
    _panel(c, m, ptop, psize,
           _array_for_pdf(rec["cam_img"]) if rec["cam_img"] is not None else None,
           f"Explanation heatmap: {XAI_METHOD}",
           ["Red = regions that raise the score of the predicted class.",
            "Drawn on the contrast-enhanced model input. Not a lesion outline."],
           rec["cam_error"])
    roi_lines = ["Automatic proposal from SAM (central box prompt); unvalidated."]
    if rec["roi_area_frac"] is not None:
        txt = f"Source: {rec['roi_source']}. Covers {rec['roi_area_frac'] * 100:.0f}% of the frame"
        if rec["cam_in_roi"] is not None:
            txt += f", holds {rec['cam_in_roi'] * 100:.0f}% of the heatmap"
        roi_lines.append(txt + ".")
    _panel(c, W - m - psize, ptop, psize,
           _array_for_pdf(rec["seg_img"]) if rec["seg_img"] is not None else None,
           "Region of interest (SAM)", roi_lines, rec["seg_error"])

    # 4. clinician
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(m, 66, "Clinician:")
    c.setFont("Helvetica", 10)
    c.drawString(m + 58, 66, "[    ] agree       [    ] disagree, diagnosis: "
                             "______________________________")
    c.setFillColorRGB(*grey)
    c.setFont("Helvetica", 7)
    c.drawString(m, 38, f"{config.MODEL_VERSION}. Research decision-support output, "
                        "not a diagnosis and not a medical device.")
    c.drawString(m, 28, "Generated " + datetime.datetime.now().strftime("%Y-%m-%d")
                 + ". Every field is filled from model outputs by a fixed template.")
    c.showPage()
    c.save()


def write_blinded_page(rec, src_path, out_path):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    W, H = A4
    m = 40
    c = canvas.Canvas(out_path, pagesize=A4)
    c.setTitle(f"Otoscope image {rec['id']}")
    c.setFont("Helvetica-Bold", 14)
    c.drawString(m, H - 42, f"Otoscope image {rec['id']}")
    size = W - 2 * m
    c.drawImage(_original_for_pdf(src_path), m, H - 70 - size, width=size,
                height=size, preserveAspectRatio=True, anchor="c")
    c.setFont("Helvetica", 10)
    c.drawString(m, H - 70 - size - 40,
                 "Diagnosis: ____________________________________________")
    c.setFillColorRGB(0.35, 0.35, 0.35)
    c.setFont("Helvetica", 8)
    c.drawString(m, H - 70 - size - 62, "Classes under study: " + ", ".join(
        clinical_name(k) for k in config.CLASSES) + ".")
    c.drawString(m, H - 70 - size - 74, "Write 'other' or 'not gradable' if none fits.")
    c.showPage()
    c.save()


# ------------------------------------------------------------------ CSV / info
def write_csv(records, out_dir):
    def num(v):
        return "" if v is None else f"{v:.4f}"
    path = os.path.join(out_dir, "predictions.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "pdf", "file", "predicted_class", "predicted_class_full",
                    "model_score", "second_class", "second_score"]
                   + [f"score_{c}" for c in config.CLASSES]
                   + ["roi_source", "roi_area_frac", "heatmap_in_roi",
                      "doctor_label", "doctor_comment"])
        for r in records:
            w.writerow([r["id"], r["id"] + ".pdf", r["file"], r["pred"],
                        clinical_name(r["pred"]), num(r["score"]), r["second"],
                        num(r["second_score"])]
                       + [num(v) for v in r["probs"]]
                       + [r["roi_source"], num(r["roi_area_frac"]),
                          num(r["cam_in_roi"]), "", ""])
    return path


def write_run_info(out_dir, root, raw_input, use_seg, weights, records, unreadable):
    counts = {}
    for r in records:
        counts[r["pred"]] = counts.get(r["pred"], 0) + 1
    n_cam = sum(r["cam_img"] is not None for r in records)
    n_seg = sum(r["seg_img"] is not None for r in records)
    seg_errors = sorted({r["seg_error"] for r in records if r["seg_error"]})
    lines = [
        f"date            : {datetime.datetime.now():%Y-%m-%d %H:%M}",
        f"input folder    : {root}",
        f"images predicted: {len(records)}",
        f"unreadable files: {len(unreadable)}",
        f"model           : {config.MODEL_VERSION}",
        f"weights         : {weights}",
        "preprocessing   : " + (
            f"raw input -> resize {PREP_SIZE}x{PREP_SIZE} -> CLAHE once -> 224"
            if raw_input else "input already CLAHE-processed -> 224 (no CLAHE)"),
        "TTA             : no",
        f"heatmaps        : {n_cam}/{len(records)}  ({XAI_METHOD})",
        f"SAM ROI         : {n_seg}/{len(records)}" + ("" if use_seg else "  (skipped)"),
    ]
    lines += [f"  ROI problem   : {e}" for e in seg_errors if use_seg]
    lines += ["", "Predicted class counts (model outputs, NOT verified labels):"]
    lines += [f"  {c:<16} {counts.get(c, 0)}" for c in config.CLASSES]
    if unreadable:
        lines += ["", "Unreadable files (no PDF written):"]
        lines += [f"  {u}" for u in unreadable]
    with open(os.path.join(out_dir, "run_info.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return lines


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folder", help="folder of unlabelled images (searched recursively)")
    ap.add_argument("--input-is-clahe", action="store_true",
                    help="images are already CLAHE-processed; do not apply it again")
    ap.add_argument("--no-seg", action="store_true", help="skip the SAM region of interest")
    ap.add_argument("--blinded", action="store_true",
                    help="also write image-only PDFs for a blinded reading")
    ap.add_argument("--weights", default=None,
                    help="weights file (default: config.WEIGHTS_PATH)")
    ap.add_argument("--out", default=None, help="output folder")
    args = ap.parse_args()

    root = os.path.abspath(args.folder)
    if not os.path.isdir(root):
        sys.exit(f"Not a folder: {root}")
    rel_paths = list_images(root)
    if not rel_paths:
        sys.exit(f"No images found under {root}")
    print(f"{len(rel_paths)} images found in {root}")

    name = os.path.basename(root.rstrip("\\/")) or "images"
    out_dir = args.out or os.path.join(config.OUT_DIR, "doctor_review", name)
    os.makedirs(out_dir, exist_ok=True)
    if args.blinded:
        os.makedirs(os.path.join(out_dir, "blinded"), exist_ok=True)

    from src import model_loader             # imports TensorFlow, so keep it lazy
    weights = args.weights or config.WEIGHTS_PATH
    model = model_loader.load_model(weights)     # rebuild + load_weights
    raw_input = not args.input_is_clahe
    use_seg = not args.no_seg

    records, unreadable = [], []
    for rel in rel_paths:
        src = os.path.join(root, rel)
        rec = analyse_image(model, src, raw_input, use_seg)
        if rec is None:                      # no number is used up, so the
            unreadable.append(rel)           # PDFs stay 001, 002, ... with no gap
            print(f"  UNREADABLE, skipped: {rel}")
            continue
        img_id = f"{len(records) + 1:03d}"
        rec.update(id=img_id, file=rel)
        write_page(rec, src, os.path.join(out_dir, img_id + ".pdf"), name)
        if args.blinded:
            write_blinded_page(rec, src, os.path.join(out_dir, "blinded", img_id + ".pdf"))
        records.append(rec)
        print(f"  {img_id}.pdf  {rec['pred']:<15} {rec['score']:.3f}   "
              f"2nd {rec['second']:<15} {rec['second_score']:.3f}   {rel}")

    write_csv(records, out_dir)
    info = write_run_info(out_dir, root, raw_input, use_seg, weights, records,
                          unreadable)
    print("\n" + "\n".join(info))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
