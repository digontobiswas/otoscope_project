"""
Deterministic structured report generation.

DESIGN CHOICE (safety): this is a TEMPLATE-FILLED, fully traceable report. It does
NOT use an LLM to write free-text "recommendations" or "diagnosis". Every field is
derived deterministically from model outputs, so the report is reproducible and
auditable — the appropriate framing for a decision-support tool, not an autonomous
diagnostic device. All clinical-interpretation strings are fixed, reviewed
templates keyed by predicted class, and the report is explicitly labelled as
decision support requiring clinician confirmation.

Outputs: a Python dict (-> JSON/CSV) and a formatted text/HTML block (-> PDF).
"""
import json
from datetime import datetime, timezone

from src import config

# fixed, reviewable interpretation templates per class (NOT model-generated prose)
INTERPRETATION = {
    "AOM": "Findings consistent with acute otitis media (e.g. bulging / "
           "erythematous tympanic membrane). Correlate with otalgia, fever and "
           "otoscopic mobility.",
    "CSOM": "Findings consistent with chronic suppurative otitis media (e.g. "
            "tympanic-membrane perforation / discharge). Assess hearing and "
            "chronicity.",
    "Earwax": "Findings consistent with cerumen impaction. Consider clearance "
              "before re-examination if the membrane is obscured.",
    "Miringoskleroz": "Findings consistent with myringosclerosis (tympanic "
                      "scarring / calcific plaques). Usually assess for hearing "
                      "impact.",
    "Normal": "No abnormal otoscopic features detected. Tympanic membrane "
              "appears within normal limits.",
    "OtitisEksterna": "Findings consistent with otitis externa (external canal "
                      "inflammation). Correlate with canal tenderness and "
                      "discharge.",
}

# calibrated-confidence bands -> qualitative certainty label (not severity)
def _certainty_band(conf):
    if conf >= 0.85:
        return "high"
    if conf >= 0.60:
        return "moderate"
    return "low"


def build_report(probs, region_descriptor=None, calibrated=True,
                 model_version=None, xai_method=None, roi_meta=None):
    """probs: length-6 array of class probabilities (calibrated if available)."""
    order = list(sorted(range(config.NUM_CLASSES), key=lambda i: -probs[i]))
    top_idx = order[0]
    top_class = config.CLASSES[top_idx]
    conf = float(probs[top_idx])

    top3 = [{"class": config.CLASSES[i],
             "description": config.CLASS_DESCRIPTIONS[config.CLASSES[i]],
             "confidence": round(float(probs[i]), 4)} for i in order[:3]]

    certainty = _certainty_band(conf)
    report = {
        "predicted_class": top_class,
        "predicted_description": config.CLASS_DESCRIPTIONS[top_class],
        "confidence": round(conf, 4),
        "confidence_is_calibrated": bool(calibrated),
        "certainty_band": certainty,
        "top_3_predictions": top3,
        "clinical_interpretation": INTERPRETATION[top_class],
        "highlighted_region": region_descriptor or
            "See attached explanation heatmap for the region driving this prediction.",
        "xai_method": xai_method or "GradCAM++",
        "roi_segmentation": roi_meta or {"available": False},
        "decision_support_note": (
            "Automated decision-support output. Not a diagnosis. Requires "
            "confirmation by a qualified clinician."),
        "uncertainty_flag": certainty == "low",
        "follow_up": (
            "Low model certainty — recommend clinician review and, if the view "
            "was obscured, re-imaging." if certainty == "low"
            else "Routine clinician confirmation advised."),
        "model_version": model_version or config.MODEL_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    return report


def report_to_text(report):
    lines = [
        "OTOSCOPIC AI DECISION-SUPPORT REPORT",
        "=" * 44,
        f"Timestamp        : {report['timestamp_utc']}",
        f"Model            : {report['model_version']}",
        "",
        f"Predicted class  : {report['predicted_class']} "
        f"({report['predicted_description']})",
        f"Confidence       : {report['confidence']*100:.2f}% "
        f"({'calibrated' if report['confidence_is_calibrated'] else 'uncalibrated'}, "
        f"{report['certainty_band']} certainty)",
        "",
        "Top-3 predictions:",
    ]
    for i, t in enumerate(report["top_3_predictions"], 1):
        lines.append(f"  {i}. {t['class']:<16} {t['confidence']*100:5.2f}%  "
                     f"({t['description']})")
    lines += [
        "",
        f"Clinical interpretation : {report['clinical_interpretation']}",
        f"Highlighted region      : {report['highlighted_region']}",
        f"Explanation method      : {report['xai_method']}",
        f"Follow-up               : {report['follow_up']}",
        "",
        report["decision_support_note"],
    ]
    return "\n".join(lines)


def save_json(report, path):
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path
