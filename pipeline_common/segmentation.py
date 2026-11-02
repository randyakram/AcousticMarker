"""
Heuristic phase segmenter: finds S1/S2 heart sounds via a Shannon-energy envelope, labels them
by the "shorter alternating interval is systole" heuristic (systole is shorter and more fixed
in duration than diastole across most physiological heart rates), and builds S1/systole/S2/
diastole phase boundaries per cardiac cycle.

This is deliberately simple -- no training data, no learned duration model -- so it can run
immediately on any recording. The CUED reference repo's HSMM segmenter (src/segmenter.py) is
the planned alternative to compare against once this heuristic path is validated end-to-end;
both will save to sibling folders under artifacts/02_segmentation/ so Stage 4 can compare
candidate markers found under either.

Every function here is intentionally stateless (no training / fitting step) so it can be
applied identically to development-set and, later, confirmation-set recordings.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

MIN_HR_BPM = 40
MAX_HR_BPM = 200
MIN_PEAKS_FOR_CYCLE = 3
MAX_CYCLE_LENGTH_CV = 0.30   # coefficient of variation above this -> "irregular / low confidence"


def shannon_energy_envelope(signal, sr, window_s=0.02):
    """Normalized average Shannon energy envelope (Liang & Hartimo 1997 / Schmidt et al. 2010
    style): emphasises the high-amplitude heart-sound bursts (S1/S2) relative to the quieter
    systolic/diastolic background, more so than a plain amplitude envelope would."""
    eps = 1e-10
    x = signal / (np.max(np.abs(signal)) + eps)
    shannon_energy = -(x ** 2) * np.log(x ** 2 + eps)
    window = max(int(window_s * sr), 1)
    kernel = np.ones(window) / window
    envelope = np.convolve(shannon_energy, kernel, mode="same")
    envelope = (envelope - envelope.mean()) / (envelope.std() + eps)   # z-normalise
    return envelope


def detect_heart_sound_peaks(envelope, sr, min_distance_s=0.15, prominence=1.0):
    """Peaks in the envelope, one per candidate heart sound (S1 or S2). min_distance_s is a
    physiological floor -- true heart sounds are essentially never less than ~150ms apart even
    at high heart rates -- prominence=1.0 (in envelope z-score units) filters out smaller
    shoulder/murmur bumps that aren't the main S1/S2 burst."""
    min_distance = max(int(min_distance_s * sr), 1)
    peaks, properties = find_peaks(envelope, distance=min_distance, prominence=prominence)
    return peaks, properties


def label_s1_s2(peaks, sr):
    """Labels alternate peaks S1/S2 using the standard heuristic: systole (S1->S2) is shorter
    and physiologically more fixed than diastole (S2->S1) across most resting-to-moderate heart
    rates, so whichever alternating set of inter-peak intervals is shorter on average is
    systole, which tells us the phase of the very first peak. Returns None if there aren't
    enough peaks to determine the pattern."""
    if len(peaks) < MIN_PEAKS_FOR_CYCLE:
        return None
    intervals = np.diff(peaks) / sr
    even_mean = intervals[0::2].mean() if len(intervals[0::2]) > 0 else np.inf
    odd_mean = intervals[1::2].mean() if len(intervals[1::2]) > 0 else np.inf
    start_label = "S1" if even_mean <= odd_mean else "S2"
    cycle_pattern = ["S1", "S2"] if start_label == "S1" else ["S2", "S1"]
    return [cycle_pattern[i % 2] for i in range(len(peaks))]


def compute_quality_metrics(peaks, labels, sr):
    """Per-recording segmentation quality: how many cycles were found, the implied heart rate,
    and how consistent cycle lengths are. Downstream stages can use quality_flag to exclude or
    down-weight recordings whose heuristic segmentation looks unreliable, rather than silently
    trusting every recording equally."""
    if labels is None:
        return {"n_peaks_detected": int(len(peaks)), "n_cycles_detected": 0,
                "estimated_hr_bpm": float("nan"), "cycle_length_cv": float("nan"),
                "quality_flag": "failed_too_few_peaks"}

    s1_times = np.array([p for p, l in zip(peaks, labels) if l == "S1"]) / sr
    if len(s1_times) < 2:
        return {"n_peaks_detected": int(len(peaks)), "n_cycles_detected": 0,
                "estimated_hr_bpm": float("nan"), "cycle_length_cv": float("nan"),
                "quality_flag": "failed_too_few_cycles"}

    cycle_lengths = np.diff(s1_times)
    hr_bpm = 60.0 / cycle_lengths.mean()
    cv = float(cycle_lengths.std() / cycle_lengths.mean()) if cycle_lengths.mean() > 0 else float("nan")

    if not (MIN_HR_BPM <= hr_bpm <= MAX_HR_BPM):
        flag = "low_confidence_hr_out_of_range"
    elif cv > MAX_CYCLE_LENGTH_CV:
        flag = "low_confidence_irregular_cycles"
    else:
        flag = "ok"

    return {"n_peaks_detected": int(len(peaks)), "n_cycles_detected": int(len(cycle_lengths)),
            "estimated_hr_bpm": float(hr_bpm), "cycle_length_cv": cv, "quality_flag": flag}


def build_phase_boundaries(peaks, labels, sr, signal_len, sound_half_width_s=0.05, n_subphases=3):
    """Turns labeled peaks into phase boundary intervals: a short fixed-width window around
    each S1/S2 peak, and the gap between consecutive sounds assigned to systole (after S1) or
    diastole (after S2) -- further split into `n_subphases` equal sub-intervals (default 3:
    early/mid/late) so "late systole"-style claims are directly expressible, not just "systole"
    as a whole. The leading fragment before the first S1 is dropped (it's not a complete cycle).
    """
    if labels is None:
        return []

    peak_times = peaks / sr
    duration_s = signal_len / sr
    subphase_names = ["early", "mid", "late"] if n_subphases == 3 else [f"part{k+1}" for k in range(n_subphases)]

    rows = []
    for i in range(len(peaks)):
        cur_label = labels[i]
        cur_time = peak_times[i]
        start = max(cur_time - sound_half_width_s, 0.0)
        end = min(cur_time + sound_half_width_s, duration_s)
        rows.append({"phase": cur_label, "subphase": None, "start_time_s": start, "end_time_s": end})

        if i < len(peaks) - 1:
            gap_start = end
            gap_end = max(peak_times[i + 1] - sound_half_width_s, gap_start)
            if gap_end > gap_start:
                gap_phase = "systole" if cur_label == "S1" else "diastole"
                sub_bounds = np.linspace(gap_start, gap_end, n_subphases + 1)
                for k in range(n_subphases):
                    rows.append({"phase": gap_phase, "subphase": subphase_names[k],
                                "start_time_s": float(sub_bounds[k]), "end_time_s": float(sub_bounds[k + 1])})

    # Assign cycle_index (increments at each S1) and drop the leading fragment before the first S1
    cycle_index = -1
    for r in rows:
        if r["phase"] == "S1":
            cycle_index += 1
        r["cycle_index"] = cycle_index
    rows = [r for r in rows if r["cycle_index"] >= 0]
    return rows


def segment_recording_heuristic(signal, sr, sound_half_width_s=0.05, n_subphases=3,
                                min_distance_s=0.15, prominence=1.0):
    """Top-level entry point: raw waveform in, (phase boundary rows, quality metrics dict) out."""
    envelope = shannon_energy_envelope(signal, sr)
    peaks, _ = detect_heart_sound_peaks(envelope, sr, min_distance_s=min_distance_s, prominence=prominence)
    labels = label_s1_s2(peaks, sr)
    quality = compute_quality_metrics(peaks, labels, sr)

    if quality["quality_flag"].startswith("failed"):
        return [], quality

    boundaries = build_phase_boundaries(peaks, labels, sr, len(signal), sound_half_width_s, n_subphases)
    return boundaries, quality
