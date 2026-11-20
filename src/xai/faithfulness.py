"""
Quantitative XAI faithfulness metrics: Deletion and Insertion AUC.

A heatmap that "looks reasonable" is not evidence. These metrics test whether the
highlighted pixels actually drive the prediction:

  Deletion:  progressively REMOVE the most-salient pixels; a faithful map makes
             confidence fall fast -> LOWER deletion-AUC is better.
  Insertion: progressively ADD the most-salient pixels onto a blurred baseline;
             a faithful map makes confidence rise fast -> HIGHER insertion-AUC
             is better.

Reference: Petsiuk et al., BMVC 2018 (RISE).

Same algorithm and same curve points as the original loop version; the
perturbed images are now built first and scored in batches instead of one
forward pass per step, which is many times faster on CPU. The model runs in
inference mode, so batching does not change the scores.
"""
import numpy as np
import tensorflow as tf
import cv2

from src import config


def _blurred_baseline(img):
    return cv2.GaussianBlur(img, (0, 0), sigmaX=15)


def _auc(curve):
    # trapezoidal AUC of a curve sampled at equal steps, normalized to [0, 1]
    trap = getattr(np, "trapezoid", getattr(np, "trapz", None))  # numpy 1.x/2.x
    return float(trap(curve, dx=1.0 / (len(curve) - 1)))


def _score(model, images, class_index, batch_size):
    out = []
    for i in range(0, len(images), batch_size):
        chunk = tf.convert_to_tensor(images[i:i + batch_size], dtype=tf.float32)
        out.append(model(chunk, training=False).numpy()[:, class_index])
    return np.concatenate(out)


def deletion_insertion(model, img_batch, cam, class_index=None, steps=50,
                       batch_size=26):
    """Returns dict with deletion_auc, insertion_auc, and both curves."""
    S = config.IMG_SIZE
    img = img_batch[0].astype("float32")              # (224, 224, 3), 0-255
    if class_index is None:
        x = tf.convert_to_tensor(img_batch, dtype=tf.float32)
        class_index = int(tf.argmax(model(x, training=False)[0]))

    order = np.argsort(cam.ravel())[::-1]              # most salient first
    n = len(order)
    step = max(1, n // steps)

    flat_real = img.reshape(-1, 3)
    del_flat = img.copy().reshape(-1, 3)               # start: full image
    ins_flat = _blurred_baseline(img).reshape(-1, 3)   # start: blurred image

    # snapshot the image before each step, exactly as the loop version scored it
    del_imgs, ins_imgs = [], []
    for i in range(0, n, step):
        del_imgs.append(del_flat.reshape(S, S, 3).copy())
        ins_imgs.append(ins_flat.reshape(S, S, 3).copy())
        px = order[i:i + step]
        del_flat[px] = 0.0                            # remove (black)
        ins_flat[px] = flat_real[px]                  # reveal real pixels
    del_imgs.append(del_flat.reshape(S, S, 3).copy())  # final point
    ins_imgs.append(ins_flat.reshape(S, S, 3).copy())

    del_curve = _score(model, np.stack(del_imgs), class_index, batch_size)
    ins_curve = _score(model, np.stack(ins_imgs), class_index, batch_size)

    return {
        "deletion_auc": _auc(del_curve),
        "insertion_auc": _auc(ins_curve),
        "deletion_curve": del_curve,
        "insertion_curve": ins_curve,
        "class_index": class_index,
    }
