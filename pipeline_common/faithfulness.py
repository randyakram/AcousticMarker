"""
Stage 7 -- standard, literature-comparable metrics for how faithful the Grad-CAM explanations
actually are to the CRNN's decisions. Stages 4-6 asked "is the region Grad-CAM points to a real
acoustic marker?" (statistical + causal + out-of-sample + signal-domain). This module asks a
different, complementary question: "is Grad-CAM itself, as an explanation method, trustworthy
here?" -- using metrics reported in the saliency-map literature generally, so results here are
comparable to numbers in *other* Grad-CAM/XAI papers, not just internally to this project.

Three metric families, each from a specific, citable source:

1. Deletion / Insertion AUC (Petsiuk, Das & Saenko, "RISE", BMVC 2018).
   Rank every time-frequency cell by Grad-CAM importance. Deletion: start from the full
   spectrogram and progressively replace the most-important cells with a neutral baseline,
   tracking P(MI) as a function of the fraction removed -- the AUC of that curve is the
   "deletion AUC" (LOWER is better: a faithful explanation should cause P(MI) to collapse
   quickly once the cells it called important are gone). Insertion is the mirror image: start
   from an all-baseline spectrogram and progressively reveal the most-important cells (HIGHER
   AUC is better). Both are computed with a matched random-order control curve, so the
   reported number can be the raw AUC or the (cam - random) gap, whichever a given paper
   reports.

2. Average Drop % / Average Increase % (Chattopadhay et al., "Grad-CAM++", WACV 2018).
   For each recording, mask everything except the top-K% most important cells and re-run the
   model. Average Drop% = mean(max(0, P_full - P_masked) / P_full) * 100 (lower is better --
   the explanation captured most of what the model needed). Average Increase% = the fraction
   of recordings where P_masked > P_full (confidence went UP despite seeing far less input --
   evidence the kept region was sufficient on its own).

3. Cascading-randomization sanity check (Adebayo et al., "Sanity Checks for Saliency Maps",
   NeurIPS 2018). Progressively re-randomize the trained model's weights from the output layer
   back toward the input, recomputing Grad-CAM after each step, and measure how much the
   heatmap changes (Spearman rank correlation against the original, fully-trained explanation).
   A meaningful explanation should degrade sharply as more of the model is randomized; an
   explanation that stays nearly identical even for a fully-randomized model is behaving like a
   fixed edge detector, independent of anything the model learned -- a well-documented failure
   mode for some saliency methods that this test is designed to catch.

All numeric logic here (masking, curves, AUCs, drop/increase formulas) is plain numpy and is
unit-testable without a GPU or even torch. The functions that need to run a live model
(`*_with_model` / the randomization sweep) take a lightweight callable `forward_fn(batch) ->
probs[:, n_classes]` rather than importing torch directly, so this module has no hard torch
dependency -- the notebook wires an actual torch model into that callable.
"""

import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz


# ---- Cell ranking + masking ---------------------------------------------------

def rank_cells_by_importance(cam):
    """Flat indices into `cam` (any 2-D array, e.g. [freq, time]), most-important first."""
    return np.argsort(cam.ravel())[::-1]


def resize_cam_nearest_free(cam, target_shape):
    """Pure-numpy (no torch/scipy) nearest-neighbour resize, used ONLY by this module's own unit
    tests (which must not depend on torch). The shipped notebook uses
    torch.nn.functional.interpolate(..., mode='bilinear') instead -- smoother and the standard
    choice in the saliency literature -- since it already depends on torch for the model itself.
    """
    src_f, src_t = cam.shape
    tgt_f, tgt_t = target_shape
    freq_idx = (np.arange(tgt_f) * src_f // tgt_f).clip(0, src_f - 1)
    time_idx = (np.arange(tgt_t) * src_t // tgt_t).clip(0, src_t - 1)
    return cam[np.ix_(freq_idx, time_idx)]


def build_step_masks(shape, order, n_steps):
    """`order`: flat indices (most-important-first) covering every cell exactly once.
    Returns a boolean array [n_steps+1, *shape] where mask[k] is True at every cell already
    "revealed" after step k (k=0 -> nothing revealed, k=n_steps -> everything revealed).
    Steps are as equal-sized as integer division allows."""
    n_cells = int(np.prod(shape))
    revealed = np.zeros((n_steps + 1, n_cells), dtype=bool)
    cut_points = np.linspace(0, n_cells, n_steps + 1).astype(int)
    for k, cut in enumerate(cut_points):
        revealed[k, order[:cut]] = True
    return revealed.reshape((n_steps + 1, *shape))


def apply_reveal_mask(feature, revealed_mask, baseline_value=0.0):
    """Where revealed_mask is True, keep `feature`'s real value; elsewhere, replace with
    `baseline_value`. `feature`: [freq, time]. `revealed_mask`: same shape, or a stack of them
    ([n_steps, freq, time]) -> returns a matching stack of masked features."""
    baseline = np.full_like(feature, baseline_value)
    return np.where(revealed_mask, feature, baseline)


# ---- Deletion / insertion curves + AUC ----------------------------------------

def auc_from_curve(fractions, values):
    """Trapezoidal AUC of `values` against `fractions` (both 1-D, `fractions` in [0, 1])."""
    return float(_trapz(values, fractions))


def deletion_insertion_curves(forward_fn, feature, cam, target_class, n_steps=20,
                              baseline_value=0.0, rng=None):
    """Runs both curves (Grad-CAM order and a matched random-order control) for ONE recording.
    `forward_fn(batch: np.ndarray[N, freq, time]) -> np.ndarray[N, n_classes]` (probabilities,
    already softmaxed) is the only thing here that touches the model.

    IMPORTANT: `cam` and `feature` must be the SAME shape. Grad-CAM is read from a conv layer's
    activations, which are almost always at a COARSER spatial resolution than the raw input
    (e.g. after a 2x2 max-pool, half the input's resolution in each dimension) -- resize `cam`
    up to `feature.shape` (e.g. bilinear interpolation) BEFORE calling this function. Passing a
    lower-resolution cam in silently produces meaningless results: numpy's fancy indexing does
    not raise on the resulting out-of-range-but-technically-valid indices, it just quietly
    restricts every mask to the first len(cam)/len(feature) fraction of the flattened feature
    array, which looks like it worked but is not the intended time-frequency alignment at all.
    This function raises instead of allowing that failure mode.

    Returns a dict with `fractions` (shared x-axis) and four probability curves:
    `deletion_cam`, `deletion_random`, `insertion_cam`, `insertion_random`, plus their AUCs.
    """
    if cam.shape != feature.shape:
        raise ValueError(
            f"cam.shape {cam.shape} != feature.shape {feature.shape} -- resize cam up to "
            f"feature's resolution (e.g. torch.nn.functional.interpolate(..., mode='bilinear')) "
            f"before calling deletion_insertion_curves. See this function's docstring."
        )
    rng = rng or np.random.default_rng(0)
    shape = feature.shape
    n_cells = int(np.prod(shape))

    order_cam = rank_cells_by_importance(cam)
    order_random = rng.permutation(n_cells)

    fractions = np.linspace(0, 1, n_steps + 1)
    out = {"fractions": fractions}

    for label, order in [("cam", order_cam), ("random", order_random)]:
        revealed = build_step_masks(shape, order, n_steps)   # [n_steps+1, freq, time]
        # revealed[k] = True at the top k/n_steps most-important cells (insertion's state at
        # step k). Deletion at step k means "the top k/n_steps most-important cells have been
        # REMOVED, everything else kept" -- i.e. exactly the logical complement of `revealed`,
        # not a reversal along the step axis (those are not the same thing: reversing would
        # remove the LEAST important cells first, which is backwards).
        deleted_so_far = ~revealed

        insertion_batch = apply_reveal_mask(feature, revealed, baseline_value)
        insertion_probs = forward_fn(insertion_batch)[:, target_class]

        deletion_batch = apply_reveal_mask(feature, deleted_so_far, baseline_value)
        deletion_probs = forward_fn(deletion_batch)[:, target_class]

        out[f"insertion_{label}"] = insertion_probs
        out[f"deletion_{label}"] = deletion_probs
        out[f"insertion_{label}_auc"] = auc_from_curve(fractions, insertion_probs)
        out[f"deletion_{label}_auc"] = auc_from_curve(fractions, deletion_probs)

    return out


# ---- Average Drop% / Average Increase% ----------------------------------------

def top_k_mask(cam, top_fraction, feature_shape=None):
    """Boolean mask, same shape as `cam`, True for the top `top_fraction` of cells by
    importance (e.g. top_fraction=0.25 keeps the most-important quarter).

    If `feature_shape` is given and differs from `cam.shape`, raises -- same silent-corruption
    risk as `deletion_insertion_curves` (see its docstring); resize cam to feature_shape first.
    """
    if feature_shape is not None and cam.shape != feature_shape:
        raise ValueError(
            f"cam.shape {cam.shape} != feature_shape {feature_shape} -- resize cam up to the "
            f"feature's resolution before building a mask meant to be applied to the feature."
        )
    n_cells = cam.size
    n_keep = max(int(round(n_cells * top_fraction)), 1)
    order = rank_cells_by_importance(cam)
    mask = np.zeros(n_cells, dtype=bool)
    mask[order[:n_keep]] = True
    return mask.reshape(cam.shape)


def average_drop_increase(p_full, p_masked):
    """Both args: 1-D arrays, one value per recording (P(target class) with the full input vs.
    with only the top-K% Grad-CAM region kept). Returns (average_drop_pct, average_increase_pct)
    -- the exact Grad-CAM++ (Chattopadhay et al. 2018) definitions. Recordings with p_full <= 0
    are excluded from the drop calculation (division undefined) but still count for "increase".
    """
    p_full = np.asarray(p_full, dtype=float)
    p_masked = np.asarray(p_masked, dtype=float)

    valid = p_full > 1e-8
    drop = np.maximum(0.0, p_full[valid] - p_masked[valid]) / p_full[valid]
    average_drop_pct = float(np.mean(drop) * 100) if valid.any() else float("nan")

    average_increase_pct = float(np.mean(p_masked > p_full) * 100)
    return average_drop_pct, average_increase_pct


# ---- Cascading-randomization sanity check -------------------------------------

def rank_correlation(a, b):
    """Spearman rank correlation between two same-shape arrays, flattened. Pure numpy (no
    scipy dependency) via rankdata-by-argsort-of-argsort, then Pearson correlation of ranks."""
    def _rank(x):
        order = x.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(x))
        return ranks

    a = a.ravel().astype(float)
    b = b.ravel().astype(float)
    ra, rb = _rank(a), _rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def summarize_randomization_sweep(rows):
    """`rows`: list of {"cascade_step": ..., "layer_randomized": ..., "similarity": ...} across
    recordings and steps. Returns a DataFrame-ready summary (mean +/- std similarity per step) --
    kept as plain dicts here so this module doesn't require pandas for its core logic; the
    notebook wraps the result in a DataFrame for display/saving."""
    from collections import defaultdict
    by_step = defaultdict(list)
    layer_names = {}
    for r in rows:
        by_step[r["cascade_step"]].append(r["similarity"])
        layer_names[r["cascade_step"]] = r["layer_randomized"]
    summary = []
    for step in sorted(by_step):
        vals = np.array(by_step[step], dtype=float)
        summary.append({
            "cascade_step": step, "layer_randomized": layer_names[step],
            "mean_similarity": float(np.nanmean(vals)), "std_similarity": float(np.nanstd(vals)),
            "n": int(np.sum(~np.isnan(vals))),
        })
    return summary
