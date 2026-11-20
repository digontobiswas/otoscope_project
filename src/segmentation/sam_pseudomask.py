"""
Segmentation WITHOUT ground truth, via SAM pseudo-masks.

There are no dense annotations for this dataset, so we do not train a supervised
segmenter. Instead we use the Segment Anything Model (SAM) with a geometric box
prompt to isolate the tympanic-membrane / lesion region, then filter by area.
This is *weak supervision / ROI extraction*, not a segmentation method evaluated
against ground truth — framed honestly in the paper.

Key settings (from project experience):
  - a CENTRAL box prompt covering ~60% of the frame (box_fraction=0.6) avoids
    SAM locking onto the bright speculum ring instead of the eardrum
  - area filtering (3%-85% of the frame) rejects tiny specular highlights and
    full-frame masks

Published precedent for SAM pseudo-masks in otoscopy: Camalan et al., SPIE 2024
(Dice 0.9486 vs expert masks), which supports the approach where a small
expert-annotated subset is available for spot-checking.

Requires: segment-anything + a SAM checkpoint (e.g. sam_vit_b.pth). If SAM is
not installed, `segment_roi` raises a clear error so the rest of the pipeline
still runs.
"""
import numpy as np
import cv2

from src import config


def _central_box(h, w, frac):
    bw, bh = int(w * frac), int(h * frac)
    x0, y0 = (w - bw) // 2, (h - bh) // 2
    return np.array([x0, y0, x0 + bw, y0 + bh])


def _load_predictor():
    try:
        from segment_anything import sam_model_registry, SamPredictor
    except ImportError as e:
        raise ImportError(
            "segment-anything is not installed. Install with:\n"
            "  pip install git+https://github.com/facebookresearch/segment-anything.git\n"
            "and download a checkpoint (e.g. sam_vit_b.pth)."
        ) from e
    import torch
    sam = sam_model_registry[config.SAM_MODEL_TYPE](checkpoint=config.SAM_CHECKPOINT)
    sam.to("cuda" if torch.cuda.is_available() else "cpu")
    return SamPredictor(sam)


_PREDICTOR = None


def segment_roi(rgb_uint8):
    """Return (mask float32 HxW in {0,1}, meta dict). Falls back to the box
    region as a coarse mask if SAM produces nothing usable."""
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = _load_predictor()

    h, w = rgb_uint8.shape[:2]
    box = _central_box(h, w, config.SAM_BOX_FRACTION)

    _PREDICTOR.set_image(rgb_uint8)
    masks, scores, _ = _PREDICTOR.predict(box=box[None, :], multimask_output=True)

    frame_area = h * w
    best, best_score = None, -1.0
    for m, s in zip(masks, scores):
        area_frac = m.sum() / frame_area
        if config.SAM_MIN_AREA_FRAC <= area_frac <= config.SAM_MAX_AREA_FRAC and s > best_score:
            best, best_score = m, s

    if best is None:                              # fallback: box region
        best = np.zeros((h, w), dtype=bool)
        best[box[1]:box[3], box[0]:box[2]] = True
        meta = {"source": "box_fallback", "score": None,
                "area_frac": float(best.sum() / frame_area)}
    else:
        meta = {"source": "sam", "score": float(best_score),
                "area_frac": float(best.sum() / frame_area)}

    return best.astype("float32"), meta


def roi_bbox(mask):
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
