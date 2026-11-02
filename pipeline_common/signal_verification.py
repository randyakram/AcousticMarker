"""
Stage 6 — independent signal-domain verification of Stage 4/5's two occlusion-validated,
confirmed candidate markers, WITHOUT Grad-CAM anywhere in the loop.

Rationale (this is what a reviewer/advisor is checking for): every earlier stage's evidence
for "apex S1 100-200 Hz" and "LUSB S2 200-300 Hz" is model-internal -- it comes from what the
CRNN's Grad-CAM attends to. That is real evidence (it survived FDR correction, fold-stability,
occlusion causality, AND out-of-sample confirmation), but it can never, by itself, rule out the
possibility that the *model* learned an idiosyncratic shortcut rather than a genuine property
of the raw audio. This module answers a narrower, model-free question: restricted to exactly
the same (site, phase, frequency band) cell, does the RAW WAVEFORM itself -- band-pass filtered,
no CNN, no Grad-CAM -- actually differ between MI and Normal patients?

Pipeline, per candidate cell:
  1. Take each recording's full waveform (already resampled/padded by Stage 2/5's conventions).
  2. Zero-phase Butterworth band-pass the WHOLE recording to the candidate's frequency band
     (filtering the full 5s clip rather than the ~100ms phase burst directly avoids edge/
     transient artifacts a short-signal filter would introduce).
  3. Slice out only the samples that fall inside that recording's phase-label time intervals
     (there are usually several per 5s clip -- one per cardiac cycle) and concatenate them.
  4. Compute four physical descriptors on that phase-restricted signal:
       - log_bandpower_db   : 10*log10(mean(filtered_signal**2)), i.e. log RMS energy of the
                               band-passed signal -- the direct time-domain analogue of
                               "band energy", computed with zero reference to the model.
       - mean_psd_band       : Welch PSD (on the *unfiltered* phase segment) integrated over the
                               candidate band and divided by bandwidth -- mean spectral density
                               in that band, in the original signal's own units.
       - spectral_entropy    : normalised Shannon entropy of the Welch PSD over 0-FREQUENCY_HIGH
                               -- how concentrated vs. spread-out the phase segment's spectrum is
                               (not band-specific; a global shape descriptor).
       - normalized_band_power: band_power / total_power_over_0_to_FREQUENCY_HIGH -- the band's
                               *share* of this segment's total spectral energy, which controls
                               for recording-to-recording loudness differences.
  5. Average to one value per patient per descriptor (same "statistical unit = patient" rule as
     Stage 4), then Mann-Whitney U + Cliff's delta + a patient-level bootstrap CI on delta, with
     BH-FDR applied across every (candidate x descriptor) test run in the same cohort.

Every function here is stateless, exactly like discovery.py, and reuses discovery.py's
cliffs_delta / benjamini_hochberg rather than redefining them.
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import mannwhitneyu

from .discovery import cliffs_delta, benjamini_hochberg, phase_label_from_row

EPS = 1e-12

# np.trapz was removed in NumPy 2.0 (renamed np.trapezoid); support either version.
_trapz = getattr(np, "trapezoid", None) or np.trapz


# ---- Band-pass filtering ----------------------------------------------------

def design_bandpass_sos(freq_low, freq_high, sr, order=4):
    """Zero-phase-ready SOS Butterworth band-pass. Edges are clipped just inside (0, Nyquist)
    so a candidate band that happens to touch DC or the Nyquist edge doesn't raise."""
    nyq = sr / 2.0
    low = max(freq_low / nyq, 1e-6)
    high = min(freq_high / nyq, 1.0 - 1e-6)
    return butter(order, [low, high], btype="band", output="sos")


def bandpass_filter(signal, freq_low, freq_high, sr, order=4):
    """Zero-phase band-pass (sosfiltfilt) of a full recording. Applied to the WHOLE clip, not
    the short phase burst, so the filter has enough samples to settle -- see module docstring."""
    if len(signal) < 3 * max(order, 1) * 2:
        return signal.astype(np.float64) * np.nan
    sos = design_bandpass_sos(freq_low, freq_high, sr, order=order)
    return sosfiltfilt(sos, signal.astype(np.float64))


# ---- Phase-segment extraction ------------------------------------------------

def extract_phase_samples(signal, sr, phase_rows):
    """Concatenates every (start_time_s, end_time_s) interval for this recording's phase_label
    (there is normally one per cardiac cycle) into a single 1-D array of samples."""
    n = len(signal)
    segments = []
    for _, r in phase_rows.iterrows():
        i0 = max(int(round(r["start_time_s"] * sr)), 0)
        i1 = min(int(round(r["end_time_s"] * sr)), n)
        if i1 > i0:
            segments.append(signal[i0:i1])
    if not segments:
        return np.array([], dtype=np.float64)
    return np.concatenate(segments)


# ---- Descriptors --------------------------------------------------------------

def _welch_psd(x, sr):
    n = len(x)
    if n < 8:
        return None, None
    nperseg = int(min(256, n))
    noverlap = nperseg // 2
    freqs, psd = welch(x, fs=sr, nperseg=nperseg, noverlap=noverlap)
    return freqs, psd


def compute_segment_descriptors(raw_phase_segment, filtered_phase_segment,
                                freq_low, freq_high, sr, freq_high_total):
    """The four physical descriptors for one recording's one (candidate) phase segment.
    Returns a dict of NaNs if the segment is too short to analyse (excluded downstream, not
    zero-filled -- same philosophy as Stage 2's failed-segmentation handling)."""
    out = {"log_bandpower_db": np.nan, "mean_psd_band": np.nan,
           "spectral_entropy": np.nan, "normalized_band_power": np.nan,
           "n_samples": int(len(raw_phase_segment))}

    if len(filtered_phase_segment) >= 8 and not np.any(np.isnan(filtered_phase_segment)):
        power = float(np.mean(filtered_phase_segment ** 2))
        out["log_bandpower_db"] = 10.0 * np.log10(power + EPS)

    freqs, psd = _welch_psd(raw_phase_segment, sr)
    if freqs is not None:
        mask_total = freqs <= freq_high_total
        mask_band = (freqs >= freq_low) & (freqs < freq_high)
        if mask_total.sum() >= 2:
            total_power = float(_trapz(psd[mask_total], freqs[mask_total]))
            band_power = float(_trapz(psd[mask_band], freqs[mask_band])) if mask_band.sum() >= 1 else 0.0
            out["mean_psd_band"] = band_power / max(freq_high - freq_low, EPS)
            out["normalized_band_power"] = band_power / (total_power + EPS)

            p = psd[mask_total]
            p = p / (p.sum() + EPS)
            entropy = float(-(p * np.log(p + EPS)).sum())
            max_entropy = np.log(mask_total.sum())
            out["spectral_entropy"] = entropy / max_entropy if max_entropy > 0 else np.nan

    return out


DESCRIPTOR_COLUMNS = ["log_bandpower_db", "mean_psd_band", "spectral_entropy", "normalized_band_power"]


def compute_recording_descriptor_table(metadata_df, boundaries_df, candidates, sr, freq_high_total,
                                       clip_seconds, load_signal_fn, filter_order=4):
    """One row per (recording, candidate): the 4 descriptors for that recording's phase segment
    in that candidate's (site, phase_label, freq_low, freq_high) cell. `load_signal_fn(filepath)`
    should return the same resampled/padded-to-`clip_seconds` waveform Stage 2/5 segmented --
    passed in rather than imported, so this module has no hard dependency on audio I/O.

    Recordings whose site doesn't match the candidate, or that have no boundary rows for the
    candidate's phase_label (e.g. failed segmentation), are skipped -- not zero-filled.
    """
    boundaries_df = boundaries_df.copy()
    boundaries_df["phase_label"] = [
        phase_label_from_row(p, s) for p, s in zip(boundaries_df["phase"], boundaries_df["subphase"])
    ]
    boundaries_by_file = {fp: g for fp, g in boundaries_df.groupby("filepath")}

    rows = []
    # cache one band-pass filtered version of each recording per unique candidate band, so a
    # recording appearing under >1 candidate doesn't get refiltered from scratch each time
    filtered_cache = {}

    for cand in candidates:
        cand_metadata = metadata_df[metadata_df["site"] == cand["site"]]
        for _, row in cand_metadata.iterrows():
            file_boundaries = boundaries_by_file.get(row["filepath"])
            if file_boundaries is None:
                continue
            phase_rows = file_boundaries[file_boundaries["phase_label"] == cand["phase_label"]]
            if len(phase_rows) == 0:
                continue

            cache_key = (row["filepath"], cand["freq_low"], cand["freq_high"])
            if cache_key not in filtered_cache:
                signal = load_signal_fn(row["filepath"])
                filtered = bandpass_filter(signal, cand["freq_low"], cand["freq_high"], sr, order=filter_order)
                filtered_cache[cache_key] = (signal, filtered)
            signal, filtered = filtered_cache[cache_key]

            raw_seg = extract_phase_samples(signal, sr, phase_rows)
            filt_seg = extract_phase_samples(filtered, sr, phase_rows)

            desc = compute_segment_descriptors(raw_seg, filt_seg, cand["freq_low"], cand["freq_high"],
                                               sr, freq_high_total)
            rows.append({
                "candidate": cand["name"], "site": cand["site"], "phase_label": cand["phase_label"],
                "freq_low": cand["freq_low"], "freq_high": cand["freq_high"],
                "filepath": row["filepath"], "patient_id": row["patient_id"],
                "class_name": row["class_name"], "cohort": row.get("cohort", "unknown"),
                **desc,
            })
        filtered_cache.clear()  # candidates use different bands -- don't carry the cache across

    return pd.DataFrame(rows)


def aggregate_descriptors_to_patient_level(recording_descriptor_df):
    """Mean across a patient's recordings within each candidate -- same 'statistical unit =
    patient, not recording' rule Stage 4 uses for Grad-CAM energy, applied here to physical
    descriptors instead."""
    group_cols = ["patient_id", "class_name", "cohort", "candidate", "site", "phase_label",
                  "freq_low", "freq_high"]
    return (
        recording_descriptor_df
        .groupby(group_cols)[DESCRIPTOR_COLUMNS]
        .mean()
        .reset_index()
    )


# ---- Statistics: Mann-Whitney + Cliff's delta + bootstrap CI -----------------

def bootstrap_cliffs_delta_ci(x, y, n_boot=2000, ci=0.95, seed=42):
    """Percentile-method bootstrap CI on Cliff's delta, resampling patients (with replacement)
    independently within each group. Returns (ci_low, ci_high); (nan, nan) if either group is
    empty."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    nx, ny = len(x), len(y)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        xs = x[rng.integers(0, nx, nx)]
        ys = y[rng.integers(0, ny, ny)]
        boot[b] = cliffs_delta(xs, ys)
    alpha = (1 - ci) / 2
    return float(np.nanpercentile(boot, 100 * alpha)), float(np.nanpercentile(boot, 100 * (1 - alpha)))


def run_signal_domain_tests(patient_descriptor_df, descriptor_cols=DESCRIPTOR_COLUMNS,
                            min_patients_per_group=3, alpha=0.05, ci=0.95, n_boot=2000, seed=42,
                            positive_class="Myocardial", negative_class="Normal"):
    """For every (candidate, descriptor) combination present in the input: Mann-Whitney U,
    Cliff's delta + bootstrap CI, at the PATIENT level. BH-FDR correction is applied across all
    (candidate x descriptor) tests run here together (e.g. 2 candidates x 4 descriptors = 8
    tests), separately for whatever cohort(s) the caller passes in (call once per cohort)."""
    rows = []
    group_keys = patient_descriptor_df[["candidate", "site", "phase_label", "freq_low", "freq_high"]] \
        .drop_duplicates().to_dict("records")

    for key in group_keys:
        subset = patient_descriptor_df[
            (patient_descriptor_df["candidate"] == key["candidate"])
        ]
        for col in descriptor_cols:
            mi_vals = subset.loc[subset["class_name"] == positive_class, col].dropna().values
            normal_vals = subset.loc[subset["class_name"] == negative_class, col].dropna().values
            base = {**key, "descriptor": col, "n_mi": len(mi_vals), "n_normal": len(normal_vals)}

            if len(mi_vals) < min_patients_per_group or len(normal_vals) < min_patients_per_group:
                rows.append({**base, "median_mi": np.nan, "median_normal": np.nan,
                            "cliffs_delta": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                            "p_value": np.nan, "insufficient_data": True})
                continue

            _, p = mannwhitneyu(mi_vals, normal_vals, alternative="two-sided")
            delta = cliffs_delta(mi_vals, normal_vals)
            ci_lo, ci_hi = bootstrap_cliffs_delta_ci(mi_vals, normal_vals, n_boot=n_boot, ci=ci, seed=seed)

            rows.append({**base, "median_mi": float(np.median(mi_vals)),
                        "median_normal": float(np.median(normal_vals)),
                        "cliffs_delta": delta, "ci_low": ci_lo, "ci_high": ci_hi,
                        "p_value": float(p), "insufficient_data": False})

    results_df = pd.DataFrame(rows)
    testable_mask = ~results_df["insufficient_data"]
    results_df["fdr_q_value"] = np.nan
    results_df["significant"] = False
    if testable_mask.sum() > 0:
        results_df.loc[testable_mask, "fdr_q_value"] = benjamini_hochberg(
            results_df.loc[testable_mask, "p_value"].values
        )
        results_df.loc[testable_mask, "significant"] = results_df.loc[testable_mask, "fdr_q_value"] < alpha
    return results_df
