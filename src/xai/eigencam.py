"""
EigenCAM for the CBAM-MobileNetV3 classifier.

EigenCAM is gradient-free: it takes the first principal component of the last
conv feature map as the saliency map. Fast and stable, which matters on edge
hardware where gradient-based methods are costly.

Reference: Muhammad & Yeasin, IJCNN 2020.
"""
import numpy as np
import tensorflow as tf
import cv2

from src import config


def eigen_cam(model, img_batch, class_index=None, layer_name=None):
    layer_name = layer_name or config.LAST_CONV_LAYER
    feat_model = tf.keras.models.Model(model.inputs,
                                       model.get_layer(layer_name).output)
    conv = feat_model(tf.convert_to_tensor(img_batch, dtype=tf.float32))[0].numpy()
    h, w, c = conv.shape
    flat = conv.reshape(h * w, c)                     # (hw, c)
    flat = flat - flat.mean(axis=0, keepdims=True)    # center
    # first principal component via SVD
    _, _, vt = np.linalg.svd(flat, full_matrices=False)
    proj = flat @ vt[0]                               # (hw,)
    cam = proj.reshape(h, w)
    cam = np.maximum(cam, 0)
    # class_index kept for a uniform API; EigenCAM is class-agnostic
    if class_index is None:
        preds = model(tf.convert_to_tensor(img_batch, dtype=tf.float32))
        class_index = int(tf.argmax(preds[0]))
    return _postprocess(cam), class_index


def _postprocess(cam, size=None):
    size = size or config.IMG_SIZE
    cam = cv2.resize(cam, (size, size))
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam.astype("float32")
