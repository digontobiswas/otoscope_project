"""
Registry that exposes every CAM method behind one call, plus an `explain_all`
that runs the full XAI + faithfulness + attention-consistency analysis for a
single image. Used by the demo, the paper experiments, and the Gradio app.
"""
import numpy as np
import tensorflow as tf

from src import config
from src.xai import gradcam, eigencam, scorecam, faithfulness, attention_consistency

CAM_METHODS = {
    "GradCAM":     lambda m, b, ci: gradcam.grad_cam(m, b, class_index=ci)[0],
    "GradCAM++":   lambda m, b, ci: gradcam.grad_cam_pp(m, b, class_index=ci)[0],
    "EigenCAM":    lambda m, b, ci: eigencam.eigen_cam(m, b, class_index=ci)[0],
    "ScoreCAM":    lambda m, b, ci: scorecam.score_cam(m, b, class_index=ci)[0],
}


def predict(model, img_batch):
    probs = model(tf.convert_to_tensor(img_batch, dtype=tf.float32)).numpy()[0]
    ci = int(np.argmax(probs))
    return probs, ci


def explain_all(model, img_batch, methods=None, run_faithfulness=True,
                run_consistency=True, score_cam_channels=64):
    """Run selected CAMs and their quantitative diagnostics for one image.

    Returns a dict:
      probs, class_index, cams{name:heatmap}, faithfulness{name:{...}},
      consistency{name:{...}}, cbam_map
    """
    methods = methods or list(CAM_METHODS.keys())
    probs, ci = predict(model, img_batch)

    cams = {}
    for name in methods:
        if name == "ScoreCAM":
            cams[name] = scorecam.score_cam(model, img_batch, class_index=ci,
                                            max_channels=score_cam_channels)[0]
        else:
            cams[name] = CAM_METHODS[name](model, img_batch, ci)

    result = {"probs": probs, "class_index": ci, "cams": cams}

    if run_faithfulness:
        result["faithfulness"] = {
            name: {k: v for k, v in
                   faithfulness.deletion_insertion(model, img_batch, cam,
                                                   class_index=ci, steps=50).items()
                   if k.endswith("_auc")}
            for name, cam in cams.items()
        }

    if run_consistency:
        cbam_map = attention_consistency.cbam_attention_map(model, img_batch)
        result["cbam_map"] = cbam_map
        result["consistency"] = {
            name: attention_consistency.consistency(cbam_map, cam)
            for name, cam in cams.items()
        }

    return result
