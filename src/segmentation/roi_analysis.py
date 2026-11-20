"""
Downstream ROI analysis — how to use the pseudo-mask WITHOUT ground truth.

Because we cannot score the mask against dense annotations, we instead measure
its *downstream usefulness*, which needs no ground truth and is more convincing
to reviewers:

  1. XAI-in-ROI fraction: what share of the model's saliency (Grad-CAM++) falls
     inside the segmented ROI. A well-behaved model should attend to the TM/lesion
     region the ROI captures, not the periphery/speculum. Higher = the model and
     the ROI agree.

  2. ROI-masked re-prediction: mask the image to the ROI and re-run the classifier.
     If confidence holds (or improves), the decision was driven by the ROI, not by
     background artefacts. A large drop signals reliance on non-clinical context.

These two numbers turn an unlabeled segmentation step into measurable evidence.
"""
import numpy as np
import tensorflow as tf

from src import config


def xai_in_roi_fraction(cam, mask):
    """Fraction of total saliency mass that lies inside the ROI mask."""
    m = (mask > 0.5).astype("float32")
    total = cam.sum() + 1e-8
    inside = (cam * m).sum()
    return float(inside / total)


def roi_masked_reprediction(model, img_batch, mask, class_index):
    """Confidence for `class_index` on the full image vs the ROI-masked image."""
    x_full = tf.convert_to_tensor(img_batch, dtype=tf.float32)
    conf_full = float(model(x_full).numpy()[0, class_index])

    m = (mask > 0.5).astype("float32")[..., None]     # (H, W, 1)
    masked = img_batch[0] * m                          # zero outside ROI
    x_roi = tf.convert_to_tensor(masked[None, ...], dtype=tf.float32)
    conf_roi = float(model(x_roi).numpy()[0, class_index])

    return {"conf_full": conf_full, "conf_roi": conf_roi,
            "delta": conf_roi - conf_full}
