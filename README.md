# Otoscope XAI: lightweight explainable classification of otoscopic images

Code and trained weights for a small attention-based classifier of otoscopic
ear images, with post-hoc explanation maps, a ground-truth-free region of
interest from SAM, and a fixed-template report.

> **Research code only.** This is not a medical device and its output is not a
> diagnosis. Do not use it for clinical decisions.

| | |
|---|---|
| Model | CBAM-MobileNetV3 (MobileNetV3-Small + Convolutional Block Attention Module) |
| Parameters | 1,028,577 |
| Input | RGB, 224 × 224 |
| Classes | AOM, CSOM, Earwax, Miringoskleroz, Normal, OtitisEksterna |
| Framework | TensorFlow 2.17 (Keras 3), CPU is enough |

Class names follow the folder names of the public dataset:

| Label | Condition |
|---|---|
| `AOM` | Acute otitis media |
| `CSOM` | Chronic suppurative otitis media |
| `Earwax` | Earwax / cerumen impaction |
| `Miringoskleroz` | Myringosclerosis |
| `Normal` | Normal tympanic membrane |
| `OtitisEksterna` | Otitis externa |

---

## What is in this repository

```
├── README.md
├── requirements.txt
├── weights/
│   └── cbam_mnv3.keras          trained classifier weights (12.8 MB)
├── src/
│   ├── config.py                flat configuration: paths, classes, layers, SAM settings
│   ├── model_loader.py          rebuilds the network, loads weights, preprocessing
│   ├── evaluation_metrics.py    metrics, calibration, McNemar / bootstrap / Wilcoxon
│   ├── test_data.py             lists test images, stratified subsets
│   ├── xai/                     Grad-CAM, Grad-CAM++, Eigen-CAM, Score-CAM,
│   │                            deletion/insertion faithfulness, reference maps,
│   │                            CBAM-attention vs CAM agreement, overlays
│   ├── segmentation/            SAM box-prompt ROI and ROI-based analyses
│   └── report/                  template-filled report (JSON / text / PDF)
├── gradio_app.py                web app: one image in, class + heatmap + ROI + report out
├── run_pipeline.py              the same analysis for one image from the command line
├── predict_folder.py            one PDF per image for an unlabelled folder (clinician review)
├── run_batch.py                 test-set metrics, confusion matrix, calibration, bootstrap CI
├── run_xai_batch.py             explanation faithfulness with reference maps, per image
├── run_seg_batch.py             SAM ROI over the test set, background-removal tests
├── make_stats.py                statistical tests and statistics figures
└── make_*_figure.py             paper figures
```

Not included: the dataset, any patient images, experiment outputs, the SAM
checkpoint (375 MB, over GitHub's file limit) and the training notebooks.

---

## Setup

Python 3.10 was used.

```bash
python -m venv venv
venv\Scripts\activate              # Windows
# source venv/bin/activate         # macOS / Linux
pip install -r requirements.txt
```

Check that the weights load:

```bash
python -c "from src import model_loader; m = model_loader.load_model(); print(f'{m.count_params():,}')"
```

It should print `1,028,577`.

### Optional: SAM for the region of interest

Only needed for the segmentation panel, `run_seg_batch.py` and the ROI parts of
the figures. Everything else runs without it.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Download the ViT-B checkpoint
(`https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth`) and
save it as `weights/sam_vit_b.pth`. `gradio_app.py` downloads it on first start
if it is missing.

---

## Two things to get right

**The model is rebuilt, then the weights are loaded.** The CBAM block uses
Lambda layers, so loading the full `.keras` file with `keras.models.load_model`
is fragile across TensorFlow versions. `src/model_loader.py` rebuilds the exact
architecture and calls `load_weights()`. Use `model_loader.load_model()`.

**CLAHE is applied exactly once.** Training images were resized to 500 × 500,
contrast-enhanced with CLAHE (L channel of LAB, clip limit 2.0, 8 × 8 tiles),
then resized to 224 × 224 with nearest-neighbour interpolation.

| Input | What to do |
|---|---|
| Raw otoscope image (camera, upload) | resize to 500, apply CLAHE once, then `preprocess_bgr(..., apply_clahe=False)`. `gradio_app.py`, `run_pipeline.py` and `predict_folder.py` do this. |
| Image already processed by the dataset pipeline | `preprocess_bgr(..., apply_clahe=False)` only. Applying CLAHE again costs about 3 accuracy points. |

---

## Usage

All commands run from the repository root with the venv active.

### Web app

```bash
python gradio_app.py               # then open http://127.0.0.1:7860
```

Upload a raw otoscope image. The app shows the six class scores, an explanation
heatmap (choice of four CAMs), the SAM region of interest if SAM is installed,
and a structured report you can download as JSON or PDF.

### One image

```bash
python run_pipeline.py path/to/image.jpg
python run_pipeline.py path/to/image.jpg --no-sam
```

### A folder of unlabelled images, for clinician review

```bash
python predict_folder.py path/to/folder
python predict_folder.py path/to/folder --no-seg     # skip SAM, much faster
python predict_folder.py path/to/folder --blinded    # also image-only sheets
python predict_folder.py path/to/folder --input-is-clahe
```

Writes `outputs/doctor_review/<folder>/001.pdf, 002.pdf, ...`, one page per
image: the original image untouched, the predicted class and all six scores,
the Grad-CAM heatmap, the SAM region of interest and an agree / disagree line.
`predictions.csv` maps each PDF number to its file and holds every score;
`run_info.txt` records the weights and preprocessing used. `--blinded` adds
`blinded/001.pdf, ...` with the image and a blank diagnosis line only, for a
reading that is not influenced by the model.

### Test-set evaluation

These expect a labelled folder laid out as `<test_dir>/<class>/<image>`, with
the six class names above, and images that are already CLAHE-processed.

```bash
python run_batch.py <test_dir>                 # metrics, confusion matrix, calibration, CI
python run_xai_batch.py <test_dir>             # faithfulness of every CAM vs reference maps
python run_xai_batch.py <test_dir> --n 12      # quick check on 12 images
python run_xai_batch.py --summarize-only       # rebuild summaries from the per-image CSV
python run_seg_batch.py <test_dir>             # SAM ROI over the test set (needs SAM)
python run_seg_batch.py --summarize-only
python make_stats.py                           # statistical tests and figures
```

`run_xai_batch.py` and `run_seg_batch.py` write one row per image as they go,
so an interrupted run resumes where it stopped. Results go to `outputs/`.

Explanation maps are computed from the pre-softmax class score by default
(`--score logit`). Each map is compared with three references: a random map, a
centred Gaussian, and the same CAM on a randomly initialised copy of the
network. A heatmap is only worth showing if it beats these.

### Figures

```bash
python make_pipeline_figure.py                 # system overview, no data needed
python make_cam_figure.py <test_dir>           # one correctly classified image per class
python make_app_figure.py <raw_image>          # what the app shows for one image
python make_seg_figure.py                      # after run_seg_batch.py and the QA sheet
```

---

## Data

The model was trained on a public otoscopic eardrum dataset (280 images, six
classes), available on
[Kaggle](https://www.kaggle.com/datasets/ucimachinelearning/otoscopic-image-dataset)
and [figshare](https://figshare.com/articles/dataset/eardrum_zip/13648166).
The data are not redistributed here.

## Known limitations

- **Data leakage in the standard protocol.** The usual protocol for this
  dataset, which these weights were trained under, augments the 280 images first and
  splits second, so augmented copies of one image can sit in both training and
  test sets. Accuracy measured on such a split is inflated. Under a split
  where the original images are separated before augmentation, accuracy is
  far lower and the small classes are mostly missed. Treat any number from
  the leaky split as an upper bound, not as expected performance.
- **Small classes.** Myringosclerosis has 4 original images and CSOM 14.
- **Domain shift.** The dataset comes from one device and one centre. Images
  from other otoscopes (different resolution, framing, colour) are outside
  what the model has seen.
- **Six forced classes.** The scores always sum to 1 over the six classes; the
  model cannot answer "other" or "not gradable".
- **Scores are uncalibrated.** Read them as a ranking, not as probabilities.
- **The ROI is a proposal.** SAM masks are not validated against expert
  annotations.
- **Explanations are post-hoc.** A heatmap shows what raised the class score,
  not where the lesion is.

## Report

The report is filled in by a fixed template from the model outputs. No
language model writes any part of it, and it is labelled as decision support,
not diagnosis.

## Citation

A paper describing this work is in preparation. Citation details will be added
here.
