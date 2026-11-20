"""
Model loading and preprocessing for the CBAM-MobileNetV3 otoscope classifier.

The model uses Lambda layers in its CBAM spatial-attention block, which make the
full .keras file fragile to load across TensorFlow versions. So we rebuild the
exact architecture and load only the weights — robust and version-independent.
"""
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV3Small

from src import config


def _cbam_block(x, ratio=8, name="cbam"):
    ch = x.shape[-1]
    avg = layers.GlobalAveragePooling2D(keepdims=True)(x)
    mx = layers.GlobalMaxPooling2D(keepdims=True)(x)
    s1 = layers.Dense(max(ch // ratio, 1), activation="relu", name=f"{name}_d1")
    s2 = layers.Dense(ch, name=f"{name}_d2")
    ca = layers.Activation("sigmoid")(layers.Add()([s2(s1(avg)), s2(s1(mx))]))
    x = layers.Multiply()([x, ca])
    avg_s = layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True),
                          output_shape=lambda s: (s[0], s[1], s[2], 1))(x)
    max_s = layers.Lambda(lambda t: tf.reduce_max(t, axis=-1, keepdims=True),
                          output_shape=lambda s: (s[0], s[1], s[2], 1))(x)
    sa = layers.Conv2D(1, 7, padding="same", activation="sigmoid",
                       name=f"{name}_sp")(layers.Concatenate()([avg_s, max_s]))
    return layers.Multiply()([x, sa])


def build_model():
    base = MobileNetV3Small(weights=None, include_top=False,
                            input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
    x = _cbam_block(base.output)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(config.NUM_CLASSES, activation="softmax")(x)
    return models.Model(base.input, out)


def load_model(weights_path=None):
    model = build_model()
    model.load_weights(weights_path or config.WEIGHTS_PATH)
    return model


def preprocess_bgr(bgr, apply_clahe=False):
    """Resize to 224 -> RGB, raw 0-255 (MobileNetV3 rescales internally).

    apply_clahe=False is the DEFAULT and matches how the model was trained:
    the 05_split folders already contain CLAHE-processed images, and the Colab
    generators used preprocessing_function=None (resize only). Re-applying
    CLAHE here would double-process the image and cost ~3 accuracy points.

    Set apply_clahe=True only for RAW input that has never been CLAHE'd — e.g.
    a live otoscope frame uploaded through the Gradio app.

    Resizing goes through PIL NEAREST to match Keras flow_from_directory, which
    is what produced the reported Colab numbers.
    """
    if apply_clahe:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab = cv2.merge((clahe.apply(l), a, b))
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).resize((config.IMG_SIZE, config.IMG_SIZE),
                                      Image.NEAREST)
    return np.asarray(pil).astype("float32")[None, ...]


def preprocess_path(path, apply_clahe=False):
    """Load an image file and preprocess it. Default apply_clahe=False, since
    files on disk in 05_split are already CLAHE-processed."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"Could not read image: {path}")
    return preprocess_bgr(bgr, apply_clahe=apply_clahe)
