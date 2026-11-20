"""
Grad-CAM and Grad-CAM++ for the CBAM-MobileNetV3 classifier.

Both produce a class-discriminative heatmap from the last convolutional feature
map. Written from scratch (no external CAM library) so it is transparent and
matches the exact model graph.

References:
  Grad-CAM:    Selvaraju et al., ICCV 2017
  Grad-CAM++:  Chattopadhay et al., WACV 2018
"""
import numpy as np
import tensorflow as tf
import cv2

from src import config


def _grad_model(model, layer_name):
    layer = model.get_layer(layer_name)
    return tf.keras.models.Model(model.inputs, [layer.output, model.output])


def grad_cam(model, img_batch, class_index=None, layer_name=None):
    layer_name = layer_name or config.LAST_CONV_LAYER
    gm = _grad_model(model, layer_name)
    x = tf.convert_to_tensor(img_batch, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_out, preds = gm(x)
        if class_index is None:
            class_index = int(tf.argmax(preds[0]))
        score = preds[:, class_index]

    grads = tape.gradient(score, conv_out)                # (1, h, w, c)
    weights = tf.reduce_mean(grads, axis=(1, 2))          # (1, c) global-avg-pool
    conv_out = conv_out[0]                                 # (h, w, c)
    cam = tf.reduce_sum(conv_out * weights[0], axis=-1)    # (h, w)
    cam = tf.nn.relu(cam).numpy()
    return _postprocess(cam), class_index


def grad_cam_pp(model, img_batch, class_index=None, layer_name=None):
    layer_name = layer_name or config.LAST_CONV_LAYER
    gm = _grad_model(model, layer_name)
    x = tf.convert_to_tensor(img_batch, dtype=tf.float32)

    with tf.GradientTape() as t3:
        with tf.GradientTape() as t2:
            with tf.GradientTape() as t1:
                conv_out, preds = gm(x)
                if class_index is None:
                    class_index = int(tf.argmax(preds[0]))
                score = preds[:, class_index]
            g1 = t1.gradient(score, conv_out)
        g2 = t2.gradient(g1, conv_out)
    g3 = t3.gradient(g2, conv_out)

    conv = conv_out[0]                                     # (h, w, c)
    g1, g2, g3 = g1[0], g2[0], g3[0]
    global_sum = tf.reduce_sum(conv, axis=(0, 1))         # (c,)
    alpha_denom = 2.0 * g2 + g3 * global_sum
    alpha_denom = tf.where(alpha_denom != 0, alpha_denom, tf.ones_like(alpha_denom))
    alpha = g2 / alpha_denom
    weights = tf.reduce_sum(alpha * tf.nn.relu(g1), axis=(0, 1))   # (c,)
    cam = tf.reduce_sum(conv * weights, axis=-1)
    cam = tf.nn.relu(cam).numpy()
    return _postprocess(cam), class_index


def _postprocess(cam, size=None):
    size = size or config.IMG_SIZE
    cam = cv2.resize(cam, (size, size))
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam.astype("float32")
