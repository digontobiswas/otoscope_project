"""
Flat configuration for the otoscope XAI + segmentation + report pipeline.
No YAML — edit values here directly.
"""
import os

# src/ -> project root
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)

# --- weights ---
WEIGHTS_PATH = os.path.join(BASE_DIR, "weights", "cbam_mnv3.keras")

# --- task constants (must match training) ---
IMG_SIZE = 224
CLASSES = ["AOM", "CSOM", "Earwax", "Miringoskleroz", "Normal", "OtitisEksterna"]
NUM_CLASSES = len(CLASSES)
CLASS_DESCRIPTIONS = {
    "AOM": "Acute Otitis Media",
    "CSOM": "Chronic Suppurative Otitis Media",
    "Earwax": "Earwax / Cerumen impaction",
    "Miringoskleroz": "Myringosclerosis",
    "Normal": "Normal Tympanic Membrane",
    "OtitisEksterna": "Otitis Externa",
}

# --- XAI hook layers (found by inspecting the CBAM-MobileNetV3 graph) ---
# last conv feature map used by Grad-CAM / Grad-CAM++ / EigenCAM / ScoreCAM
LAST_CONV_LAYER = "activation_17"          # (7, 7, 576)
# CBAM spatial-attention map, used for attention-explanation consistency
CBAM_SPATIAL_LAYER = "cbam_sp"             # (7, 7, 1)

# --- SAM (segmentation without ground truth) ---
# central box prompt covering this fraction of the image (per project learnings)
SAM_BOX_FRACTION = 0.6
# keep masks whose area is within this fraction range of the image
SAM_MIN_AREA_FRAC = 0.03
SAM_MAX_AREA_FRAC = 0.85
SAM_CHECKPOINT = os.path.join(BASE_DIR, "weights", "sam_vit_b.pth")   # download separately
SAM_MODEL_TYPE = "vit_b"

# --- report ---
MODEL_VERSION = "CBAM-MobileNetV3 v1.0"

# --- output dir for demo artifacts ---
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
