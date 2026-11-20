"""
PDF export of the structured report + overlays.

Uses reportlab (pure-Python, no system deps — works on Raspberry Pi). If
reportlab is missing, `save_pdf` raises a clear install hint.
"""
import os

from src import config


def save_pdf(report, out_path, overlay_paths=None):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader
    except ImportError as e:
        raise ImportError("Install reportlab:  pip install reportlab") from e

    c = canvas.Canvas(out_path, pagesize=A4)
    W, H = A4
    y = H - 2 * cm

    c.setFont("Helvetica-Bold", 15)
    c.drawString(2 * cm, y, "Otoscopic AI Decision-Support Report")
    y -= 0.9 * cm
    c.setFont("Helvetica", 9)
    c.drawString(2 * cm, y, report["decision_support_note"])
    y -= 0.8 * cm

    c.setFont("Helvetica", 10)
    rows = [
        ("Timestamp", report["timestamp_utc"]),
        ("Model", report["model_version"]),
        ("Predicted", f"{report['predicted_class']} "
                      f"({report['predicted_description']})"),
        ("Confidence", f"{report['confidence']*100:.2f}%  "
                       f"({report['certainty_band']} certainty)"),
        ("Explanation", report["xai_method"]),
        ("Follow-up", report["follow_up"]),
    ]
    for k, v in rows:
        c.setFont("Helvetica-Bold", 10)
        c.drawString(2 * cm, y, f"{k}:")
        c.setFont("Helvetica", 10)
        c.drawString(5.2 * cm, y, str(v)[:70])
        y -= 0.6 * cm

    y -= 0.2 * cm
    c.setFont("Helvetica-Bold", 10)
    c.drawString(2 * cm, y, "Top-3 predictions:")
    y -= 0.55 * cm
    c.setFont("Helvetica", 10)
    for i, t in enumerate(report["top_3_predictions"], 1):
        c.drawString(2.3 * cm, y,
                     f"{i}. {t['class']:<16} {t['confidence']*100:5.2f}%")
        y -= 0.5 * cm

    y -= 0.2 * cm
    c.setFont("Helvetica-Bold", 10)
    c.drawString(2 * cm, y, "Clinical interpretation:")
    y -= 0.5 * cm
    c.setFont("Helvetica", 9)
    for line in _wrap(report["clinical_interpretation"], 95):
        c.drawString(2.3 * cm, y, line); y -= 0.45 * cm

    # overlays (original / heatmap / segmentation), if provided
    if overlay_paths:
        x = 2 * cm
        img_y = 2 * cm
        for p in overlay_paths[:3]:
            if os.path.exists(p):
                c.drawImage(ImageReader(p), x, img_y, width=5.5 * cm,
                            height=5.5 * cm, preserveAspectRatio=True)
                x += 6 * cm

    c.showPage()
    c.save()
    return out_path


def _wrap(text, width):
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line); line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out
