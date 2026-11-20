"""
Reference ("null") saliency maps. A faithfulness or consistency number only
means something next to what a trivial map scores on the same test.

  random_map    - uniform noise on the same 7x7 grid as the CAMs, upsampled the
                  same way, so it has the same smoothness but no information
  center_prior  - isotropic Gaussian at the image center; in otoscopy the
                  membrane is usually central, so this is the "look at the
                  middle" map any CAM has to beat
  randomized    - the same CAM method run on the same architecture with random
                  weights (Adebayo et al., NeurIPS 2018 sanity check)
"""
import numpy as np
import cv2

from src import config
from src import model_loader


def _norm(m):
    m = m.astype("float32")
    m = m - m.min()
    return m / m.max() if m.max() > 0 else m


def random_map(rng, grid=7, size=None):
    size = size or config.IMG_SIZE
    return _norm(cv2.resize(rng.random((grid, grid)).astype("float32"),
                            (size, size)))


def center_prior(size=None, sigma_frac=0.25):
    size = size or config.IMG_SIZE
    ys, xs = np.mgrid[0:size, 0:size].astype("float32")
    c = (size - 1) / 2.0
    s = sigma_frac * size
    return _norm(np.exp(-((xs - c) ** 2 + (ys - c) ** 2) / (2 * s * s)))


def to_grid(m, grid=7):
    """Downsample a 224x224 map to the native 7x7 grid (area average)."""
    return cv2.resize(m.astype("float32"), (grid, grid),
                      interpolation=cv2.INTER_AREA)


def randomized_model(trained_model):
    """Same architecture, random weights. Call this AFTER the trained model is
    built: Keras auto-names layers per session, so the second build gets
    different names. Returns (model, name of its last-conv layer)."""
    names = [l.name for l in trained_model.layers]
    idx = names.index(config.LAST_CONV_LAYER)
    rmodel = model_loader.build_model()          # weights=None -> random init
    return rmodel, rmodel.layers[idx].name
