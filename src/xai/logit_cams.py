"""
Grad-CAM and Grad-CAM++ computed from the pre-softmax class score (logit).

The original gradcam.py differentiates the softmax probability. Both papers
define the method on the class score before softmax, and the closed-form
Grad-CAM++ weights assume the score is exp(logit). This module does it the
standard way without touching the model: the final Dense layer's kernel and
bias are applied to its input inside the gradient tape, so the model is still
built with build_model() + load_weights().

References:
  Grad-CAM:    Selvaraju et al., ICCV 2017
  Grad-CAM++:  Chattopadhay et al., WACV 2018
"""
import numpy as np
import tensorflow as tf
import cv2

from src import config


_CACHE = {}


def _feature_and_logit(model, layer_name):
    """Model that returns (conv feature map, input to the final Dense layer),
    plus a function mapping that input to the 6 class logits. Cached per
    (model, layer) because building a sub-model on every call is slow."""
    dense = model.layers[-1]
    key = (id(model), layer_name)
    if key not in _CACHE:
        _CACHE[key] = tf.keras.models.Model(
            model.inputs, [model.get_layer(layer_name).output, dense.input])
    fm = _CACHE[key]

    def to_logits(pre):
        return tf.matmul(pre, dense.kernel) + dense.bias

    return fm, to_logits


def class_logits(model, img_batch):
    """Pre-softmax class scores (N, C). Useful for temperature scaling too."""
    dense = model.layers[-1]
    pre = tf.keras.models.Model(model.inputs, dense.input)(
        tf.convert_to_tensor(img_batch, dtype=tf.float32))
    return (tf.matmul(pre, dense.kernel) + dense.bias).numpy()


def _postprocess(cam, size=None):
    size = size or config.IMG_SIZE
    cam = cv2.resize(cam.astype("float32"), (size, size))
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam.astype("float32")


def grad_cam_logit(model, img_batch, class_index, layer_name=None):
    layer_name = layer_name or config.LAST_CONV_LAYER
    fm, to_logits = _feature_and_logit(model, layer_name)
    x = tf.convert_to_tensor(img_batch, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv, pre = fm(x)
        score = to_logits(pre)[:, class_index]
    grads = tape.gradient(score, conv)[0]                 # (h, w, c)
    weights = tf.reduce_mean(grads, axis=(0, 1))          # (c,)
    cam = tf.nn.relu(tf.reduce_sum(conv[0] * weights, axis=-1)).numpy()
    return _postprocess(cam)


def grad_cam_pp_logit(model, img_batch, class_index, layer_name=None):
    """Grad-CAM++ with the standard closed form: with S = exp(logit), the
    second and third derivatives reduce to powers of the first."""
    layer_name = layer_name or config.LAST_CONV_LAYER
    fm, to_logits = _feature_and_logit(model, layer_name)
    x = tf.convert_to_tensor(img_batch, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv, pre = fm(x)
        score = to_logits(pre)[:, class_index]
    g = tape.gradient(score, conv)[0]                     # (h, w, c)
    a = conv[0]
    g2, g3 = g ** 2, g ** 3
    denom = 2.0 * g2 + tf.reduce_sum(a, axis=(0, 1)) * g3
    denom = tf.where(denom != 0.0, denom, tf.ones_like(denom))
    alpha = g2 / denom
    weights = tf.reduce_sum(alpha * tf.nn.relu(g), axis=(0, 1))
    cam = tf.nn.relu(tf.reduce_sum(a * weights, axis=-1)).numpy()
    return _postprocess(cam)
