"""Listing test images and drawing a stratified subset. Shared by the batch
scripts so they all see the same images in the same order."""
import os
import numpy as np

from src import config

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_images(test_dir):
    """Return (paths, labels) for test_dir/<class>/*, classes in config order."""
    paths, labels = [], []
    for ci, c in enumerate(config.CLASSES):
        folder = os.path.join(test_dir, c)
        if not os.path.isdir(folder):
            continue
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(folder, f))
                labels.append(ci)
    return paths, np.array(labels, dtype=int)


def stratified_subset(labels, n, seed=42):
    """Indices of about n images, equal per class, fixed seed. n=None -> all."""
    if n is None or n >= len(labels):
        return list(range(len(labels)))
    rng = np.random.default_rng(seed)
    per = max(1, n // config.NUM_CLASSES)
    idx = []
    for ci in range(config.NUM_CLASSES):
        cls = np.where(labels == ci)[0]
        if len(cls):
            idx.extend(sorted(rng.choice(cls, min(per, len(cls)), replace=False)))
    return sorted(int(i) for i in idx)
