# AcousticMarker

**Discovering Candidate Acoustic Markers of Myocardial Infarction in Heart Sound Recordings**

A multi-stage pipeline that uses a deep-learning classifier as an *instrument* for locating candidate acoustic biomarkers of myocardial infarction (MI) in phonocardiogram (PCG) recordings, then subjects those candidates to statistical, causal, out-of-sample, and signal-domain verification before treating them as credible.

> **Research code, not a diagnostic tool.** The findings here are candidate markers that establish association and relevance to a classifier. They are not a verified physiological mechanism and have not been clinically validated.

---

## Overview

Most Grad-CAM-based discovery work stops at the saliency map and asserts that the highlighted region is clinically plausible. This project goes further:

1. **Discover** where in (site × cardiac phase × frequency band) a CRNN finds MI-vs-normal evidence.
2. **Test causally** whether the classifier actually depends on those regions (occlusion).
3. **Confirm** on a patient-level cohort held out from every earlier stage.
4. **Re-derive** the markers from the raw waveform with no model or Grad-CAM involved.
5. **Check the explanation method itself** with sanity and faithfulness metrics.

### Data

- 140 patients (70 MI / 70 normal), 3,360 five-second recordings
- 4 auscultation sites per patient: apex, LLSB, LUSB, RUSB
- Source: Hasan Sadikin Hospital
- 80/20 patient-level split (seed = 42): 112 development / 28 confirmation patients. No patient appears on both sides at any stage.

## Pipeline

Each stage is a notebook, intended to be run in order.

| Stage | Notebook | What it does |
|---|---|---|
| 1 | `stage1_patient_split.ipynb` | Patient-level development/confirmation split, fixed at the start of the pipeline |
| 2 | `stage2_segmentation.ipynb` | Heuristic cardiac-cycle segmentation (S1, systole, S2, diastole; systole and diastole split into early/mid/late) via a Shannon-energy envelope with peak detection. Recordings are quality-flagged; failures are excluded rather than zero-filled |
| 3 | `stage3_train_and_explain.ipynb` | Patient-grouped 5-fold CRNN training (two conv blocks → bidirectional GRU → mean-pooled binary output). Out-of-fold Grad-CAM computed against the "evidence for MI" class uniformly |
| 4 | `stage4_candidate_discovery.ipynb` | Bins Grad-CAM into site × phase × 100 Hz-band cells (256 total), one value per patient per cell, Mann-Whitney U + Cliff's δ, Benjamini-Hochberg FDR, fold-stability filter, then occlusion testing of the top 10 candidates against 20 random same-sized controls |
| 5 | `stage5_confirmation.ipynb` | Re-tests occlusion-validated candidates, unchanged, on the locked 28-patient confirmation cohort |
| 6 | `stage6_signal_domain_verification.ipynb` | Model-free re-test: zero-phase Butterworth band-pass to the candidate band within the candidate phase, then four spectral descriptors (log-bandpower/RMS, Welch mean PSD, spectral entropy, normalized band power) |
| 7 | `stage7_faithfulness_metrics.ipynb` | Explanation-method checks: cascading randomization sanity check, Average Drop / Increase, Deletion / Insertion AUC |

Shared helpers live in [`pipeline_common/`](pipeline_common/).

## Key results

**Classifier** (patient-grouped, out-of-fold):

| Evaluation | Accuracy | Sensitivity | Specificity | AUC |
|---|---|---|---|---|
| Development, recording level (5-fold OOF) | 88.6% | 88.0% | 89.2% | 0.947 |
| Development, patient level (mean probability) | 99.1% | 98.2% | 100% | n/a |
| Confirmation cohort, recording level | 87.6% | 82.7% | 92.6% | 0.942 |

**Candidate funnel:** 256 cells tested → 18 significant after FDR and fold-stable → 10 occlusion-tested → **2 validated** → 2 replicated on the confirmation cohort.

**The two candidate markers** (both show *lower* MI energy than normal):

| Marker | Dev. Cliff's δ | Confirmation Cliff's δ | Confirmation p |
|---|---|---|---|
| Apex, S1, 100–200 Hz | −0.40 | −0.74 | 8.6 × 10⁻⁴ |
| LUSB, S2, 200–300 Hz | −0.48 | −0.82 | 2.6 × 10⁻⁴ |

**Signal-domain verification** (pooled, no model involved): both markers replicate in direction. The apex marker is significant on all four descriptors; the LUSB marker only on the two raw energy descriptors, suggesting the two may reflect different acoustic phenomena. Neither reached significance on the 28-patient confirmation cohort alone, which a power approximation suggests is under-powered rather than evidence against the markers.

**Explanation faithfulness:** the cascading randomization test passes (Spearman similarity to the original Grad-CAM collapses to ~0 once the head is randomized), and Deletion AUC clearly favors Grad-CAM order over random (0.264 vs 0.409). **Insertion AUC is essentially at chance** (0.419 vs 0.415). This is reported as a genuine mixed result.

See the paper for full tables, figures, and discussion.

## Repository structure

```
AcousticMarker/
├── pipeline_common/                      # shared utilities
├── stage1_patient_split.ipynb
├── stage2_segmentation.ipynb
├── stage3_train_and_explain.ipynb
├── stage4_candidate_discovery.ipynb
├── stage5_confirmation.ipynb
├── stage6_signal_domain_verification.ipynb
└── stage7_faithfulness_metrics.ipynb
```

## Getting started

```bash
git clone https://github.com/randyakram/AcousticMarker.git
cd AcousticMarker
```

1. Install the Python dependencies.
2. Place the recordings in the data directory.
3. Run the notebooks in order, `stage1` through `stage7`. Later stages depend on artifacts written by earlier ones.

**Reproducibility notes**

- The patient split uses `seed = 42` and is created once in Stage 1. Do not regenerate it mid-experiment.
- The confirmation cohort must stay untouched until Stage 5. It is segmented and explained for the first time at that stage.

## Data availability

The recordings originate from Hasan Sadikin Hospital. The full 140-patient dataset is **not** publicly distributed due to patient privacy.

## Limitations

- Modest cohort (140 patients) for a deep-learning study.
- Only 2 of 18 statistically robust candidates survived the full validation chain; occlusion testing was run on only 10 of 18, so some true markers may remain untested.
- Single collection source. Demographic, comorbidity, and MI subtype/severity information was not available, so generalization across etiologies, equipment, and noise conditions is unknown.
- Both classifier and marker evaluations are patient-independent but come from the same collection. True external validation needs an independently collected dataset.
- Insertion AUC did not support Grad-CAM faithfulness, so the explanation method is not faithful by every available metric.
