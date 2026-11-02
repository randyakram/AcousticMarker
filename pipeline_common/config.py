"""
Shared configuration for the MI acoustic-marker-discovery pipeline.

Every stage notebook imports from here instead of redefining these values, so a change made
once (e.g. a different DATA_DIR, or a different confirmation-set fraction) is guaranteed to be
seen identically by every stage. Stage-specific parameters (e.g. which segmentation method,
which frequency bands to test) live in that stage's own notebook / config, not here.
"""

import pathlib

# ---- Project layout -------------------------------------------------------
# All paths are relative to wherever this pipeline's folder lives, so the project is portable
# (works the same whether it's on your laptop, a lab machine, or a cloud notebook).
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# ---- Data source ------------------------------------------------------------
# Folder containing your STEMI/ and Normal/ subfolders (edit this to point at your real dataset;
# defaults to the pipeline's own sample_data/ folder for a quick smoke test).
DATA_DIR = "/content/drive/MyDrive/Research 2026/Segmented"
CLASS_FOLDERS = {"normal_segment": 0, "myocardial_segment": 1}      # folder name -> label (1 = positive / MI class)
CLASSES = ["Normal", "Myocardial"]                      # index 0 = Normal, index 1 = MI

# ---- Reproducibility --------------------------------------------------------
SEED = 42

# ---- Stage 1: patient-level split -------------------------------------------
# Fraction of PATIENTS (not recordings) set aside as the locked confirmation cohort. These
# patients are touched exactly once, at Stage 5, and never used for training, tuning, or
# candidate-marker discovery.
CONFIRMATION_FRACTION = 0.20
MIN_PATIENTS_PER_CLASS_FOR_SPLIT = 3   # below this, a held-out split isn't meaningful

# ---- Audio / feature extraction (used from Stage 3 onward) -----------------
SAMPLE_RATE = 4000            # Hz, everything is resampled to this
FREQUENCY_HIGH = 800           # Hz, spectrogram content above this is discarded
WINDOW_LENGTH_S = 0.050         # seconds, STFT window
WINDOW_STEP_S = 0.020           # seconds, STFT hop
CLIP_SECONDS = 5               # seconds, every recording is padded/trimmed to this length

# ---- Stage 3: k-fold training + out-of-fold explanations -------------------
N_FOLDS = 5                    # outer folds -- each recording gets explained by a model that
                                # never trained on its patient (see Stage 3 notebook for why)
MIN_PATIENTS_PER_CLASS_FOR_KFOLD = N_FOLDS
INNER_VAL_FRACTION = 0.15      # within each fold's training pool, held out for early stopping
BATCH_SIZE = 16
EPOCHS = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
EARLY_STOPPING_PATIENCE = 10
GRADCAM_TARGET_CLASS = 1       # always explain "evidence for MI" (index 1), never "predicted
                                # class" -- so a CAM value means the same thing on every
                                # recording regardless of its true or predicted label, which is
                                # what makes MI-vs-Normal energy comparisons in Stage 4 valid

# ---- Stage 4: candidate marker discovery ------------------------------------
FREQUENCY_BAND_WIDTH_HZ = 100   # candidate frequency bands are fixed round numbers (0-100,
                                 # 100-200, ...), decided in advance rather than searched over,
                                 # so the search itself isn't a hidden extra multiple-comparison
MIN_PATIENTS_PER_GROUP_FOR_TEST = 3   # per class, per cell, to attempt a statistical test at all
FDR_ALPHA = 0.05                # Benjamini-Hochberg significance threshold, applied across
                                 # every (site, phase, frequency-band) cell tested
ENERGY_AGGREGATION = "mean"     # how a recording's heatmap values within one cell are reduced
                                 # to a single number: "mean" (robust) or "max" (peakier)
N_RANDOM_CONTROLS = 20          # random same-size regions per candidate, for the occlusion null
TOP_K_CANDIDATES_FOR_OCCLUSION = 10   # occlusion-test only the K most significant candidates
                                       # (it's the expensive step -- re-runs the model many times)
FOLD_STABILITY_THRESHOLD = 0.5  # a candidate needs direction agreement in MORE than this
                                 # fraction of folds to be considered stable, not a fluke split
OCCLUSION_SEED = 42
