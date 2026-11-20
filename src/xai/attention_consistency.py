"""
Attention-Explanation Consistency (the paper's novel metric).

The CBAM block already produces a *built-in* spatial-attention map (where the
network chose to look). Post-hoc methods (Grad-CAM++, etc.) produce a *derived*
explanation of the same prediction. If the model is internally coherent, the two
should agree on the relevant region.

This module quantifies that agreement, giving a single interpretable number that
prior otoscopy XAI papers do not report:

  - IoU of the thresholded maps
  - Pearson and Spearman correlation of the raw maps
  - centroid distance (how far apart the peaks are)

High consistency = the explanation reflects where the model actually attended,
not an artefact of the CAM method. This is a faithfulness signal that needs no
ground-truth mask, which is exactly the setting here.
"""
import numpy as np
import tensorflow as tf
import cv2
from scipy.stats import pearsonr, spearmanr

from src import config


def cbam_attention_map(model, img_batch):
    """Extract and upsample the CBAM spatial-attention map to image size."""
    sp_model = tf.keras.models.Model(
        model.inputs, model.get_layer(config.CBAM_SPATIAL_LAYER).output)
    sp = sp_model(tf.convert_to_tensor(img_batch, dtype=tf.float32))[0].numpy()
    sp = sp[..., 0]                                    # (7, 7)
    sp = cv2.resize(sp, (config.IMG_SIZE, config.IMG_SIZE))
    sp = sp - sp.min()
    if sp.max() > 0:
        sp = sp / sp.max()
    return sp.astype("float32")


def _centroid(m):
    ys, xs = np.mgrid[0:m.shape[0], 0:m.shape[1]]
    s = m.sum() + 1e-8
    return np.array([(m * ys).sum() / s, (m * xs).sum() / s])


def consistency(cbam_map, cam_map, thresh=0.5):
    """All-in-one agreement between the CBAM attention map and a post-hoc CAM."""
    a = cbam_map.ravel()
    b = cam_map.ravel()

    # correlation of raw maps
    pear = float(pearsonr(a, b)[0]) if a.std() > 0 and b.std() > 0 else 0.0
    spear = float(spearmanr(a, b)[0]) if a.std() > 0 and b.std() > 0 else 0.0

    # IoU of thresholded binary masks
    ba = cbam_map >= thresh
    bb = cam_map >= thresh
    inter = np.logical_and(ba, bb).sum()
    union = np.logical_or(ba, bb).sum()
    iou = float(inter / union) if union > 0 else 0.0

    # centroid distance, normalized by image diagonal
    diag = np.hypot(config.IMG_SIZE, config.IMG_SIZE)
    cdist = float(np.linalg.norm(_centroid(cbam_map) - _centroid(cam_map)) / diag)

    return {"pearson": pear, "spearman": spear, "iou": iou,
            "centroid_dist": cdist}
