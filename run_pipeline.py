"""
End-to-end single-image pipeline:
  classify -> XAI (all CAMs + faithfulness + attention-consistency)
           -> SAM ROI (optional; skipped cleanly if SAM not installed)
           -> ROI downstream analysis
           -> structured report -> JSON + PDF + overlays

Run:  python run_pipeline.py path/to/image.jpg
"""
import os
import sys
import json

import numpy as np
import cv2

from src import config
from src import model_loader
from src.xai import registry, visualize
from src.segmentation import roi_analysis
from src.report import generate_report, export_pdf


def run(image_path, out_dir=None, use_sam=True):
    out_dir = out_dir or config.OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    model = model_loader.load_model()
    batch = model_loader.preprocess_path(image_path)
    rgb = batch[0].astype("uint8")

    # 1. classify + full XAI analysis
    res = registry.explain_all(model, batch)
    probs, ci = res["probs"], res["class_index"]
    pred_class = config.CLASSES[ci]
    print(f"Prediction: {pred_class}  ({probs[ci]*100:.2f}%)")

    # 2. save overlays for each CAM
    overlay_paths = []
    for name, cam in res["cams"].items():
        ov = visualize.overlay_heatmap(rgb, cam)
        p = os.path.join(out_dir, f"overlay_{name}.png")
        cv2.imwrite(p, cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
        overlay_paths.append(p)

    # 3. quantitative XAI diagnostics
    print("\nXAI faithfulness (deletion-AUC lower=better, insertion-AUC higher=better):")
    for name, f in res["faithfulness"].items():
        print(f"  {name:<12} del {f['deletion_auc']:.4f}  ins {f['insertion_auc']:.4f}")
    print("Attention-explanation consistency (CBAM vs CAM):")
    for name, cconf in res["consistency"].items():
        print(f"  {name:<12} IoU {cconf['iou']:.3f}  Pearson {cconf['pearson']:.3f}")

    # 4. SAM ROI (optional)
    roi_meta = {"available": False}
    if use_sam:
        try:
            from src.segmentation import sam_pseudomask
            mask, meta = sam_pseudomask.segment_roi(rgb)
            seg_ov = visualize.mask_overlay(rgb, mask)
            seg_path = os.path.join(out_dir, "overlay_segmentation.png")
            cv2.imwrite(seg_path, cv2.cvtColor(seg_ov, cv2.COLOR_RGB2BGR))
            overlay_paths.insert(0, seg_path)

            gradcampp = res["cams"]["GradCAM++"]
            frac = roi_analysis.xai_in_roi_fraction(gradcampp, mask)
            rep = roi_analysis.roi_masked_reprediction(model, batch, mask, ci)
            roi_meta = {"available": True, "source": meta["source"],
                        "area_frac": round(meta["area_frac"], 3),
                        "xai_in_roi": round(frac, 3),
                        "conf_full": round(rep["conf_full"], 3),
                        "conf_roi": round(rep["conf_roi"], 3)}
            print(f"\nROI: source={meta['source']}  XAI-in-ROI={frac:.3f}  "
                  f"conf full={rep['conf_full']:.3f} roi={rep['conf_roi']:.3f}")
        except ImportError as e:
            print(f"\n[SAM skipped] {e}")

    # 5. structured report -> JSON + text + PDF
    report = generate_report.build_report(
        probs, calibrated=False, xai_method="GradCAM++", roi_meta=roi_meta)
    generate_report.save_json(report, os.path.join(out_dir, "report.json"))
    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        f.write(generate_report.report_to_text(report))
    try:
        export_pdf.save_pdf(report, os.path.join(out_dir, "report.pdf"),
                            overlay_paths=overlay_paths)
        print(f"\nSaved report.json / report.txt / report.pdf + overlays to {out_dir}")
    except ImportError:
        print(f"\nSaved report.json / report.txt + overlays to {out_dir} "
              f"(install reportlab for PDF)")

    return report


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python run_pipeline.py path/to/image.jpg")
        sys.exit(1)
    run(sys.argv[1], use_sam="--no-sam" not in sys.argv)
