"""
Otoscope Intelligent Diagnostic App: one Gradio UI for

  upload image -> classification + class probabilities
               -> XAI overlay (Grad-CAM / Grad-CAM++ / Eigen-CAM / Score-CAM)
               -> SAM ROI overlay (box prompt; box fallback if SAM finds nothing)
               -> optional deletion/insertion faithfulness for this image
               -> deterministic structured report (text / JSON / PDF)

Matches the paper's methods:
  - raw uploads go through the training chain once: resize to 500x500
    (INTER_AREA) -> CLAHE on L of LAB -> resize to 224 (nearest)
  - Grad-CAM / Grad-CAM++ use the pre-softmax class score (src/xai/logit_cams.py)
  - SAM runs on the 500x500 image, not the 224 network input
  - the CBAM attention map is NOT shown as an explanation (the paper finds it
    does not agree with post-hoc explanations)

Runs on CPU. Research / educational decision-support demo only. NOT a medical
device.

Local:       python gradio_app.py            (http://127.0.0.1:7860)
Public link: python gradio_app.py --share    (temporary *.gradio.live link)
On a Hugging Face Space it starts automatically.
"""
import os
import sys
import tempfile
import urllib.request

import numpy as np
import cv2
import gradio as gr

from src import config
from src import model_loader
from src.xai import eigencam, scorecam, faithfulness, visualize, logit_cams
from src.report import generate_report, export_pdf

SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
PREP_SIZE = 500          # size the pipeline resized to before CLAHE

# Shown in red, in brackets, on the same line as the title. Edit the wording here.
APP_TITLE = "Otoscope Intelligent Diagnostic System"
COLLABORATION = ("A collaborative project of the IIT Kharagpur, AIIMS Bhopal, "
                 "ICMR-RMNIHR and ICMR-NIREH")

MODEL = model_loader.load_model()


def _ensure_sam():
    """True if segment-anything is importable and the checkpoint exists.
    Downloads the ViT-B checkpoint (~375 MB) once if it is missing."""
    try:
        import segment_anything  # noqa: F401
    except ImportError:
        return False
    if not os.path.exists(config.SAM_CHECKPOINT):
        try:
            os.makedirs(os.path.dirname(config.SAM_CHECKPOINT), exist_ok=True)
            print("Downloading SAM ViT-B checkpoint (one time, ~375 MB)...")
            urllib.request.urlretrieve(SAM_URL, config.SAM_CHECKPOINT)
        except Exception as e:                          # no network, disk full...
            print("SAM checkpoint download failed:", e)
            return False
    return True


SAM_OK = _ensure_sam()

CAM_FUNCS = {
    "Grad-CAM": lambda b, ci: logit_cams.grad_cam_logit(MODEL, b, ci),
    "Grad-CAM++": lambda b, ci: logit_cams.grad_cam_pp_logit(MODEL, b, ci),
    "Eigen-CAM": lambda b, ci: eigencam.eigen_cam(MODEL, b, class_index=ci)[0],
    "Score-CAM": lambda b, ci: scorecam.score_cam(MODEL, b, class_index=ci,
                                                  max_channels=64)[0],
}


def _prepare_raw(pil):
    """Raw upload -> (network batch, CLAHE'd 500x500 RGB image).
    Same order as the dataset pipeline, so CLAHE is applied exactly once."""
    rgb = np.array(pil.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(bgr, (PREP_SIZE, PREP_SIZE), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    bgr = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
    batch = model_loader.preprocess_bgr(bgr, apply_clahe=False)
    return batch, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def analyze(image, cam_choice, run_faith):
    if image is None:
        return None, {}, None, "Please upload an image first.", "", None, None

    batch, rgb_full = _prepare_raw(image)
    rgb = batch[0].astype("uint8")

    # classification
    probs = MODEL.predict(batch, verbose=0)[0]
    ci = int(np.argmax(probs))
    conf_dict = {f"{c} ({config.CLASS_DESCRIPTIONS[c]})": float(p)
                 for c, p in zip(config.CLASSES, probs)}

    # explanation for the predicted class
    cam = CAM_FUNCS[cam_choice](batch, ci)
    xai_overlay = visualize.overlay_heatmap(rgb, cam)
    notes = [f"Explanation: {cam_choice} for the predicted class "
             f"({config.CLASSES[ci]})."]
    if run_faith:
        f = faithfulness.deletion_insertion(MODEL, batch, cam, class_index=ci,
                                            steps=50)
        notes.append(f"Faithfulness for this image: deletion AUC "
                     f"{f['deletion_auc']:.3f} (lower = better), insertion AUC "
                     f"{f['insertion_auc']:.3f} (higher = better). On the test "
                     f"set a centered Gaussian scores 0.234 / 0.678.")

    # SAM region of interest on the 500x500 image
    roi_meta = {"available": False}
    seg_overlay = None
    if SAM_OK:
        try:
            from src.segmentation import sam_pseudomask, roi_analysis
            mask_full, meta = sam_pseudomask.segment_roi(rgb_full)
            seg_overlay = visualize.mask_overlay(rgb_full, mask_full)
            mask = cv2.resize(mask_full.astype("float32"),
                              (config.IMG_SIZE, config.IMG_SIZE),
                              interpolation=cv2.INTER_NEAREST)
            frac = roi_analysis.xai_in_roi_fraction(cam, mask)
            roi_meta = {"available": True, "source": meta["source"],
                        "area_frac": round(meta["area_frac"], 3),
                        "xai_in_roi": round(frac, 3)}
            notes.append(f"ROI ({meta['source']}): covers "
                         f"{meta['area_frac']*100:.0f}% of the frame and holds "
                         f"{frac*100:.0f}% of the heatmap. The ROI is a weak, "
                         f"unvalidated proposal.")
        except Exception as e:
            roi_meta = {"available": False, "error": str(e)[:120]}
            notes.append("Segmentation failed for this image: " + str(e)[:120])
    else:
        notes.append("Segmentation unavailable (segment-anything or the SAM "
                     "checkpoint is missing).")

    # deterministic, template-filled report (no language model)
    report = generate_report.build_report(
        probs, calibrated=False, xai_method=cam_choice, roi_meta=roi_meta)
    report_txt = generate_report.report_to_text(report)

    tmp = tempfile.mkdtemp()
    json_path = os.path.join(tmp, "otoscope_report.json")
    generate_report.save_json(report, json_path)
    pdf_path = None
    try:
        ov_paths = []
        for nm, im in (("xai", xai_overlay), ("seg", seg_overlay)):
            if im is not None:
                p = os.path.join(tmp, f"{nm}.png")
                cv2.imwrite(p, cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
                ov_paths.append(p)
        pdf_path = os.path.join(tmp, "otoscope_report.pdf")
        export_pdf.save_pdf(report, pdf_path, overlay_paths=ov_paths)
    except Exception:
        pdf_path = None

    return (xai_overlay, conf_dict, seg_overlay, report_txt, "\n".join(notes),
            json_path, pdf_path)


with gr.Blocks(title=APP_TITLE) as demo:
    gr.HTML(f'<h1 style="font-size:1.9em; font-weight:700; line-height:1.3; margin:0.2em 0;">'
            f'{APP_TITLE} <span style="color:#e00000;">({COLLABORATION})</span></h1>')
    gr.Markdown(
        "Classification, explanation heatmaps, region of interest and an "
        "automatic structured report for one otoscope image. "
        "CBAM-MobileNetV3 (1.03M parameters), CPU only.")
    gr.Markdown(
        "> **Disclaimer:** research / educational decision-support demo only. "
        "**Not a medical device**; not for clinical diagnosis.\n>\n"
        "> **Privacy:** do not upload images that could identify a patient. "
        "Upload raw otoscope images (the app applies contrast enhancement itself).")

    with gr.Row():
        with gr.Column(scale=1):
            image_in = gr.Image(type="pil", sources=["upload", "webcam"],
                                label="Otoscope image")
            cam_choice = gr.Radio(list(CAM_FUNCS.keys()), value="Grad-CAM",
                                  label="Explanation method")
            run_faith = gr.Checkbox(value=False,
                                    label="Compute faithfulness (slower)")
            go = gr.Button("Analyze", variant="primary")
        with gr.Column(scale=1):
            conf_out = gr.Label(num_top_classes=6, label="Classification")

    with gr.Row():
        xai_out = gr.Image(label="Explanation heatmap", interactive=False)
        seg_out = gr.Image(label="Region of interest (SAM)", interactive=False)

    notes_out = gr.Textbox(label="Notes", lines=4, interactive=False)
    report_out = gr.Textbox(label="Structured report", lines=16,
                            interactive=False)
    with gr.Row():
        json_file = gr.File(label="Download JSON", interactive=False)
        pdf_file = gr.File(label="Download PDF", interactive=False)

    go.click(analyze, inputs=[image_in, cam_choice, run_faith],
             outputs=[xai_out, conf_out, seg_out, report_out, notes_out,
                      json_file, pdf_file])


if __name__ == "__main__":
    demo.launch(share="--share" in sys.argv)
