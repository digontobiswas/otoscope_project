"""Heatmap overlay helpers shared by the XAI demo and the Gradio app."""
import numpy as np
import cv2


def overlay_heatmap(rgb_uint8, cam, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """rgb_uint8: (H, W, 3) 0-255 uint8. cam: (H, W) in [0, 1]. Returns RGB uint8."""
    heat = cv2.applyColorMap((cam * 255).astype("uint8"), colormap)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    if heat.shape[:2] != rgb_uint8.shape[:2]:
        heat = cv2.resize(heat, (rgb_uint8.shape[1], rgb_uint8.shape[0]))
    out = (alpha * heat + (1 - alpha) * rgb_uint8).astype("uint8")
    return out


def mask_overlay(rgb_uint8, mask, color=(0, 255, 0), alpha=0.4):
    """Overlay a binary/float mask (H, W) on an RGB image."""
    m = (mask > 0.5).astype("uint8")
    colored = np.zeros_like(rgb_uint8)
    colored[m == 1] = color
    out = rgb_uint8.copy()
    idx = m == 1
    out[idx] = (alpha * colored[idx] + (1 - alpha) * rgb_uint8[idx]).astype("uint8")
    # draw contour
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    return out
