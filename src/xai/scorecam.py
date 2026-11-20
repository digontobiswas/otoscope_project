"""
ScoreCAM for the CBAM-MobileNetV3 classifier.

ScoreCAM is gradient-free: each conv feature map is upsampled to a mask, applied
to the input, and weighted by the resulting class confidence. More faithful than
Grad-CAM in some cases but much slower (one forward pass per channel), so on the
Raspberry Pi it is reported CPU-only / offline, not real-time.

Reference: Wang et al., CVPR-W 2020.
"""
import numpy as np
import tensorflow as tf
import cv2

from src import config


def score_cam(model, img_batch, class_index=None, layer_name=None,
              max_channels=None, batch=32):
    layer_name = layer_name or config.LAST_CONV_LAYER
    feat_model = tf.keras.models.Model(model.inputs,
                                       model.get_layer(layer_name).output)
    x = tf.convert_to_tensor(img_batch, dtype=tf.float32)

    conv = feat_model(x)[0].numpy()                   # (h, w, c)
    if class_index is None:
        class_index = int(tf.argmax(model(x)[0]))

    h, w, c = conv.shape
    size = config.IMG_SIZE
    base = img_batch[0]                               # (224, 224, 3), 0-255

    # optionally subsample channels for speed (report the setting in the paper)
    idx = np.arange(c)
    if max_channels and max_channels < c:
        # pick channels with the highest activation energy
        energy = conv.reshape(-1, c).sum(0)
        idx = np.argsort(energy)[-max_channels:]

    # build masked inputs
    masks = []
    for k in idx:
        m = conv[:, :, k]
        m = cv2.resize(m, (size, size))
        mn, mx = m.min(), m.max()
        m = (m - mn) / (mx - mn + 1e-8)
        masks.append(m)
    masks = np.stack(masks)                           # (K, 224, 224)

    scores = np.zeros(len(idx), dtype="float32")
    for i in range(0, len(idx), batch):
        chunk = masks[i:i + batch]
        inp = base[None] * chunk[..., None]           # (b, 224, 224, 3)
        preds = model(tf.convert_to_tensor(inp, dtype=tf.float32)).numpy()
        scores[i:i + batch] = preds[:, class_index]

    scores = tf.nn.softmax(scores).numpy()            # normalize weights
    cam = np.tensordot(scores, masks, axes=(0, 0))    # (224, 224)
    cam = np.maximum(cam, 0)
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam.astype("float32"), class_index
