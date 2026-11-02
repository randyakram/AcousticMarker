"""
Turns out-of-fold Grad-CAM heatmaps (Stage 3) + phase boundaries (Stage 2) into candidate
acoustic markers: phase x frequency-band x site cells where MI and Normal patients' heatmap
energy differs statistically, survives multiple-comparison correction and fold-to-fold
stability, and causally matters to the classifier (occlusion testing).

Every function here is stateless and reusable -- Stage 5 will call the statistical-testing and
occlusion functions again, unchanged, on the confirmation cohort.
"""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from . import config


# ---- Frequency / time binning -----------------------------------------------

def generate_frequency_bands(freq_high, band_width):
    """Fixed, round-number frequency bands decided in advance (not searched over), e.g.
    0-100, 100-200, ... -- so the choice of bin edges isn't itself a hidden extra multiple
    comparison on top of the (phase, site) search."""
    edges = np.arange(0, freq_high + 1e-6, band_width)
    if edges[-1] < freq_high:
        edges = np.append(edges, freq_high)
    return [(float(edges[i]), float(edges[i + 1])) for i in range(len(edges) - 1)]


def phase_label_from_row(phase, subphase):
    """S1/S2 have no subphase; systole/diastole get early/mid/late appended, e.g.
    'systole_late' -- this is what makes "late systole"-style claims directly expressible."""
    if subphase is None or (isinstance(subphase, float) and np.isnan(subphase)):
        return phase
    return f"{phase}_{subphase}"


# ---- Heatmap -> phase x frequency-band energy -------------------------------

def compute_recording_energy_table(oof_df, cams_array, boundaries_df, freq_bands,
                                   freq_high, clip_seconds, aggregation="mean"):
    """One row per (recording, phase_label, frequency_band): the mean (or max) Grad-CAM value
    within that recording's time range(s) for that phase label, restricted to that frequency
    band. Recordings with no phase boundaries (failed segmentation) are skipped -- not
    zero-filled -- so they don't quietly bias any cell's statistics."""
    n_cam_freq, n_cam_time = cams_array.shape[1], cams_array.shape[2]
    freq_edges_cam = np.linspace(0, freq_high, n_cam_freq + 1)
    time_edges_cam = np.linspace(0, clip_seconds, n_cam_time + 1)

    boundaries_df = boundaries_df.copy()
    boundaries_df["phase_label"] = [
        phase_label_from_row(p, s) for p, s in zip(boundaries_df["phase"], boundaries_df["subphase"])
    ]
    boundaries_by_file = {fp: g for fp, g in boundaries_df.groupby("filepath")}

    oof_df = oof_df.reset_index(drop=True)
    rows = []
    for i, row in oof_df.iterrows():
        file_boundaries = boundaries_by_file.get(row["filepath"])
        if file_boundaries is None or len(file_boundaries) == 0:
            continue
        cam = cams_array[i]

        for phase_label, group in file_boundaries.groupby("phase_label"):
            time_mask = np.zeros(n_cam_time, dtype=bool)
            for _, b in group.iterrows():
                t0 = max(np.searchsorted(time_edges_cam, b["start_time_s"], side="right") - 1, 0)
                t1 = min(np.searchsorted(time_edges_cam, b["end_time_s"], side="left"), n_cam_time)
                if t1 > t0:
                    time_mask[t0:t1] = True
            if not time_mask.any():
                continue

            for freq_low, freq_hi in freq_bands:
                f0 = max(np.searchsorted(freq_edges_cam, freq_low, side="right") - 1, 0)
                f1 = min(np.searchsorted(freq_edges_cam, freq_hi, side="left"), n_cam_freq)
                if f1 <= f0:
                    continue
                sub_block = cam[f0:f1][:, time_mask]
                if sub_block.size == 0:
                    continue
                value = float(sub_block.mean()) if aggregation == "mean" else float(sub_block.max())
                rows.append({
                    "filepath": row["filepath"], "patient_id": row["patient_id"],
                    "class_name": row["class_name"], "site": row["site"],
                    "fold_index": row["fold_index"], "phase_label": phase_label,
                    "freq_low": freq_low, "freq_high": freq_hi, "energy": value,
                })
    return pd.DataFrame(rows)


def aggregate_to_patient_level(recording_energy_df):
    """The statistical unit is the PATIENT, not the recording -- a patient's segments/sites are
    correlated, not independent samples, so testing at recording level would understate
    variance and overstate significance. This averages every patient's recordings within each
    (site, phase_label, frequency-band) cell down to one value before any test is run."""
    return (
        recording_energy_df
        .groupby(["patient_id", "class_name", "site", "phase_label", "freq_low", "freq_high"])
        .agg(energy=("energy", "mean"), fold_index=("fold_index", "first"))
        .reset_index()
    )


# ---- Statistics --------------------------------------------------------------

def cliffs_delta(x, y):
    """Effect size paired with Mann-Whitney: P(x > y) - P(x < y), in [-1, 1]. 0 = no
    separation between the two groups' distributions; +-1 = complete separation."""
    x = np.asarray(x, dtype=float)[:, None]
    y = np.asarray(y, dtype=float)[None, :]
    if x.size == 0 or y.size == 0:
        return float("nan")
    diff = x - y
    return float(((diff > 0).sum() - (diff < 0).sum()) / (x.size * y.size))


def benjamini_hochberg(pvalues):
    """Standard BH step-up FDR correction. Returns q-values in the same order as the input."""
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    result = np.empty(n)
    result[order] = q
    return result


def run_statistical_tests(patient_energy_df, min_patients_per_group=3, alpha=0.05,
                          positive_class=None, negative_class=None):
    """Mann-Whitney U test (MI vs Normal) + Cliff's delta for every (site, phase_label,
    frequency-band) cell, then Benjamini-Hochberg correction across every cell actually tested.
    Cells without enough patients per group are recorded (so nothing silently disappears) but
    marked `insufficient_data` and excluded from correction.

    `positive_class`/`negative_class` default to config.CLASSES[1]/[0] -- pulled from config
    rather than hardcoded, since a hardcoded "MI" here would silently match zero rows (and
    every cell would look like it has 0 patients) for anyone using a different class label,
    e.g. "Myocardial" instead of "MI". The output columns are still named n_mi/median_mi/etc
    regardless of the actual label string -- those names just mean "the positive class"."""
    positive_class = positive_class or config.CLASSES[1]
    negative_class = negative_class or config.CLASSES[0]

    group_cols = ["site", "phase_label", "freq_low", "freq_high"]
    rows = []
    for keys, group in patient_energy_df.groupby(group_cols):
        site, phase_label, freq_low, freq_high = keys
        mi_vals = group.loc[group["class_name"] == positive_class, "energy"].dropna().values
        normal_vals = group.loc[group["class_name"] == negative_class, "energy"].dropna().values
        base = {"site": site, "phase_label": phase_label, "freq_low": freq_low, "freq_high": freq_high,
               "n_mi": len(mi_vals), "n_normal": len(normal_vals)}
        if len(mi_vals) < min_patients_per_group or len(normal_vals) < min_patients_per_group:
            rows.append({**base, "median_mi": np.nan, "median_normal": np.nan,
                        "cliffs_delta": np.nan, "p_value": np.nan, "insufficient_data": True})
            continue
        _, p = mannwhitneyu(mi_vals, normal_vals, alternative="two-sided")
        rows.append({**base, "median_mi": float(np.median(mi_vals)),
                    "median_normal": float(np.median(normal_vals)),
                    "cliffs_delta": cliffs_delta(mi_vals, normal_vals),
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


def check_fold_stability(patient_energy_df, candidate_row, positive_class=None, negative_class=None):
    """For one candidate cell, checks whether the MI-vs-Normal direction (which class has the
    higher median) agrees across folds, using only each fold's own held-out patients. With
    small per-fold patient counts this isn't expected to be independently significant per fold
    -- it's a simpler, honest question: is the direction consistent, or did the pooled result
    depend on which patients happened to land in which fold?"""
    positive_class = positive_class or config.CLASSES[1]
    negative_class = negative_class or config.CLASSES[0]

    subset = patient_energy_df[
        (patient_energy_df["site"] == candidate_row["site"]) &
        (patient_energy_df["phase_label"] == candidate_row["phase_label"]) &
        (patient_energy_df["freq_low"] == candidate_row["freq_low"]) &
        (patient_energy_df["freq_high"] == candidate_row["freq_high"])
    ]
    overall_direction = np.sign(candidate_row["median_mi"] - candidate_row["median_normal"])
    agree, total = 0, 0
    for _, fold_group in subset.groupby("fold_index"):
        mi_vals = fold_group.loc[fold_group["class_name"] == positive_class, "energy"].dropna()
        normal_vals = fold_group.loc[fold_group["class_name"] == negative_class, "energy"].dropna()
        if len(mi_vals) == 0 or len(normal_vals) == 0:
            continue
        total += 1
        if np.sign(mi_vals.median() - normal_vals.median()) == overall_direction:
            agree += 1
    return agree, total


# ---- Confirmation (Stage 5) --------------------------------------------------

def confirm_candidates(confirmation_patient_energy_df, candidates_df, min_patients_per_group=3,
                       alpha=0.05, positive_class=None, negative_class=None):
    """The Stage 5 test: re-runs the SAME statistical test as Stage 4, but only on the specific
    (site, phase_label, frequency-band) cells already shortlisted there -- no new search, no
    re-tuning -- against the confirmation cohort. A candidate counts as `confirmed` only if it's
    BOTH still FDR-significant (corrected across just this shortlist, not the original 256-cell
    scan) AND its direction (which class has higher energy) agrees with what was found in
    development. Agreeing on significance but flipping direction is not a replication."""
    positive_class = positive_class or config.CLASSES[1]
    negative_class = negative_class or config.CLASSES[0]

    rows = []
    for _, candidate in candidates_df.iterrows():
        subset = confirmation_patient_energy_df[
            (confirmation_patient_energy_df["site"] == candidate["site"]) &
            (confirmation_patient_energy_df["phase_label"] == candidate["phase_label"]) &
            (confirmation_patient_energy_df["freq_low"] == candidate["freq_low"]) &
            (confirmation_patient_energy_df["freq_high"] == candidate["freq_high"])
        ]
        mi_vals = subset.loc[subset["class_name"] == positive_class, "energy"].dropna().values
        normal_vals = subset.loc[subset["class_name"] == negative_class, "energy"].dropna().values

        base = {"site": candidate["site"], "phase_label": candidate["phase_label"],
               "freq_low": candidate["freq_low"], "freq_high": candidate["freq_high"],
               "dev_median_mi": candidate["median_mi"], "dev_median_normal": candidate["median_normal"],
               "dev_cliffs_delta": candidate["cliffs_delta"], "dev_p_value": candidate["p_value"],
               "confirm_n_mi": len(mi_vals), "confirm_n_normal": len(normal_vals)}

        if len(mi_vals) < min_patients_per_group or len(normal_vals) < min_patients_per_group:
            rows.append({**base, "confirm_median_mi": np.nan, "confirm_median_normal": np.nan,
                        "confirm_cliffs_delta": np.nan, "confirm_p_value": np.nan,
                        "direction_agrees": None, "insufficient_data": True})
            continue

        _, p = mannwhitneyu(mi_vals, normal_vals, alternative="two-sided")
        dev_direction = np.sign(candidate["median_mi"] - candidate["median_normal"])
        confirm_direction = np.sign(np.median(mi_vals) - np.median(normal_vals))

        rows.append({**base, "confirm_median_mi": float(np.median(mi_vals)),
                    "confirm_median_normal": float(np.median(normal_vals)),
                    "confirm_cliffs_delta": cliffs_delta(mi_vals, normal_vals),
                    "confirm_p_value": float(p),
                    "direction_agrees": bool(dev_direction == confirm_direction),
                    "insufficient_data": False})

    results_df = pd.DataFrame(rows)
    testable_mask = ~results_df["insufficient_data"]
    results_df["confirm_fdr_q_value"] = np.nan
    results_df["confirmed"] = False
    if testable_mask.sum() > 0:
        results_df.loc[testable_mask, "confirm_fdr_q_value"] = benjamini_hochberg(
            results_df.loc[testable_mask, "confirm_p_value"].values
        )
        results_df.loc[testable_mask, "confirmed"] = (
            (results_df.loc[testable_mask, "confirm_fdr_q_value"] < alpha) &
            (results_df.loc[testable_mask, "direction_agrees"] == True)  # noqa: E712 (tri-state column)
        )
    return results_df


# ---- Occlusion (causal) validation -------------------------------------------

def occlude_region(feat, freq_low, freq_high, time_intervals_s, freq_high_total, clip_seconds):
    """Replaces the given (frequency band x time interval(s)) region of a spectrogram with
    that recording's own mean value -- a gentler erase than hard zero, which would itself be an
    unusually strong, out-of-distribution signal the model might key off of."""
    feat = feat.copy()
    n_freq, n_time = feat.shape
    freq_edges = np.linspace(0, freq_high_total, n_freq + 1)
    time_edges = np.linspace(0, clip_seconds, n_time + 1)
    fill_value = feat.mean()

    f0 = max(np.searchsorted(freq_edges, freq_low, side="right") - 1, 0)
    f1 = min(np.searchsorted(freq_edges, freq_high, side="left"), n_freq)
    if f1 <= f0:
        return feat

    for start_s, end_s in time_intervals_s:
        t0 = max(np.searchsorted(time_edges, start_s, side="right") - 1, 0)
        t1 = min(np.searchsorted(time_edges, end_s, side="left"), n_time)
        if t1 > t0:
            feat[f0:f1, t0:t1] = fill_value
    return feat


def run_occlusion_test(candidate, oof_df, boundaries_df, fold_models, device, config,
                       n_random_controls=20, seed=42):
    """Causal check for one candidate cell: does masking exactly this (frequency band, this
    recording's own phase-label time interval(s)) region hurt the classifier's accuracy on the
    recordings that actually have this phase present, more than masking a same-size region
    picked at random? Uses each recording's own out-of-fold model (via fold_index), so this
    stays a fair out-of-fold evaluation, not the model grading its own training data."""
    from .features import calculate_features
    from .audio import load_and_resample, pad_or_trim
    import torch

    rng = np.random.default_rng(seed)
    boundaries_df = boundaries_df.copy()
    boundaries_df["phase_label"] = [
        phase_label_from_row(p, s) for p, s in zip(boundaries_df["phase"], boundaries_df["subphase"])
    ]
    boundaries_by_file = {fp: g for fp, g in boundaries_df.groupby("filepath")}

    relevant = oof_df[oof_df["site"] == candidate["site"]].reset_index(drop=True)
    rows_to_test = []
    for _, row in relevant.iterrows():
        fb = boundaries_by_file.get(row["filepath"])
        if fb is None:
            continue
        phase_rows = fb[fb["phase_label"] == candidate["phase_label"]]
        if len(phase_rows) == 0:
            continue
        rows_to_test.append((row, phase_rows))

    if len(rows_to_test) == 0:
        return {"n_recordings_tested": 0, "baseline_accuracy": float("nan"),
                "occluded_accuracy": float("nan"), "occlusion_effect": float("nan"),
                "random_control_effects": [], "percentile_vs_random": float("nan")}

    def eval_with_occlusion(freq_low, freq_high, use_actual_intervals):
        correct_before, correct_after, total = 0, 0, 0
        for row, phase_rows in rows_to_test:
            model = fold_models[int(row["fold_index"])]
            signal = pad_or_trim(load_and_resample(row["filepath"], config.SAMPLE_RATE),
                                 config.SAMPLE_RATE, config.CLIP_SECONDS)
            feat = calculate_features(signal)
            true_label = int(row["true_label"])

            with torch.no_grad():
                x = torch.from_numpy(feat).float().unsqueeze(0).to(device)
                probs_before = torch.softmax(model(x), dim=1)[0].cpu().numpy()

            if use_actual_intervals:
                intervals = list(zip(phase_rows["start_time_s"], phase_rows["end_time_s"]))
            else:
                total_dur = float((phase_rows["end_time_s"] - phase_rows["start_time_s"]).sum())
                start = rng.uniform(0, max(config.CLIP_SECONDS - total_dur, 0))
                intervals = [(start, start + total_dur)]

            occluded_feat = occlude_region(feat, freq_low, freq_high, intervals,
                                          config.FREQUENCY_HIGH, config.CLIP_SECONDS)
            with torch.no_grad():
                x_occ = torch.from_numpy(occluded_feat).float().unsqueeze(0).to(device)
                probs_after = torch.softmax(model(x_occ), dim=1)[0].cpu().numpy()

            correct_before += int(probs_before.argmax() == true_label)
            correct_after += int(probs_after.argmax() == true_label)
            total += 1
        return correct_before / total, correct_after / total, total

    baseline_acc, occluded_acc, n_tested = eval_with_occlusion(
        candidate["freq_low"], candidate["freq_high"], use_actual_intervals=True
    )
    occlusion_effect = baseline_acc - occluded_acc

    band_width = candidate["freq_high"] - candidate["freq_low"]
    random_effects = []
    for _ in range(n_random_controls):
        rand_freq_low = rng.uniform(0, max(config.FREQUENCY_HIGH - band_width, 0))
        b_acc, o_acc, _ = eval_with_occlusion(rand_freq_low, rand_freq_low + band_width,
                                              use_actual_intervals=False)
        random_effects.append(b_acc - o_acc)

    random_effects = np.array(random_effects)
    percentile = float((random_effects < occlusion_effect).mean()) if len(random_effects) else float("nan")

    return {"n_recordings_tested": n_tested, "baseline_accuracy": baseline_acc,
            "occluded_accuracy": occluded_acc, "occlusion_effect": occlusion_effect,
            "random_control_effects": random_effects.tolist(), "percentile_vs_random": percentile}
