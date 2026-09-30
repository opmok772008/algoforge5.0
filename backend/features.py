"""
backend/features.py — RR-Feature extraction and Heart Rate Variability (HRV) metrics.
Ported from open-source ECG signal processing algorithms:
  - R-peak detection & RR intervals
  - Heart Rate (HR bpm)
  - SDNN (Standard Deviation of NN intervals in ms)
  - RMSSD (Root Mean Square of Successive Differences in ms)
  - NN50 (Count of successive differences > 50 ms)
  - CV (Coefficient of Variation of RR intervals)
  - Rhythm classification: Normal sinus rhythm, Atrial fibrillation suspected, Other arrhythmia, Bradycardia, Tachycardia
"""

from __future__ import annotations
from typing import Any, Dict, List
import numpy as np


def analyze_rr_features(signal: np.ndarray, fs: int = 250) -> Dict[str, Any]:
    """
    Extract R-peak, RR-interval, heart-rate, and HRV metrics from one ECG trace.

    Parameters
    ----------
    signal : np.ndarray
        Finite, one-dimensional ECG samples. Traces shorter than two seconds return
        an ``Insufficient beats`` result without attempting peak detection.
    fs : int
        Sampling rate in hertz; must be a positive integer.

    Returns
    -------
    Dict[str, Any]
        JSON-compatible metrics: peak indices, filtered samples, RR intervals,
        heart rate, SDNN, RMSSD, NN50, rhythm label, and explanatory advice.

    Raises
    ------
    ValueError
        If the trace is non-numeric, not one-dimensional, or contains NaN/inf.
    TypeError, ValueError
        If ``fs`` is not a positive integer.

    Complexity
    ----------
    Time and auxiliary space are O(N) for N ECG samples.
    """
    if isinstance(fs, bool) or not isinstance(fs, (int, np.integer)):
        raise TypeError("fs must be an integer sampling rate")
    if fs < 1:
        raise ValueError("sampling rate fs must be at least 1 Hz")
    try:
        signal = np.asarray(signal, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("signal must contain numeric samples") from exc
    if signal.ndim != 1:
        raise ValueError(f"signal must be one-dimensional; got shape {signal.shape}")
    if not np.all(np.isfinite(signal)):
        raise ValueError("signal must contain only finite samples")

    n = int(signal.shape[0])
    if n < fs * 2:
        return {
            "p": [],
            "y": signal.tolist() if isinstance(signal, np.ndarray) else signal,
            "rr": [],
            "hr": 0.0,
            "m": 0.0,
            "sd": 0.0,
            "rms": 0.0,
            "nn50": 0,
            "cv": 0.0,
            "rhythm": "Insufficient beats",
            "condition": "Indeterminate",
            "prog": "Need at least 2-3 seconds of ECG recording for analysis.",
            "items": [],
        }

    # Baseline removal using moving average of length fs
    w_base = max(5, int(fs))
    half_base = w_base // 2
    cumsum = np.cumsum(np.insert(signal, 0, 0))
    base = (cumsum[np.minimum(n, np.arange(n) + half_base + 1)] - cumsum[np.maximum(0, np.arange(n) - half_base)]) / (
        np.minimum(n, np.arange(n) + half_base + 1) - np.maximum(0, np.arange(n) - half_base)
    )
    y = signal - base

    # Small smoothing
    w_sm = 5
    half_sm = w_sm // 2
    cumsum_y = np.cumsum(np.insert(y, 0, 0))
    h = (cumsum_y[np.minimum(n, np.arange(n) + half_sm + 1)] - cumsum_y[np.maximum(0, np.arange(n) - half_sm)]) / (
        np.minimum(n, np.arange(n) + half_sm + 1) - np.maximum(0, np.arange(n) - half_sm)
    )

    mx = float(np.max(h))
    threshold = 0.45 * mx
    peaks: List[int] = []
    last = -1e9
    min_dist = 0.25 * fs  # 250 ms minimum distance

    for i in range(1, n - 1):
        if h[i] > h[i - 1] and h[i] > h[i + 1] and h[i] > threshold and (i - last) > min_dist:
            peaks.append(i)
            last = i

    rr: List[float] = []
    for i in range(1, len(peaks)):
        rr.append((peaks[i] - peaks[i - 1]) / float(fs))

    if len(rr) < 2:
        m = rr[0] if rr else 0.0
        sd = 0.0
        rms = 0.0
        nn50 = 0
        hr = 60.0 / m if m > 0 else 0.0
        cv = 0.0
    else:
        m = float(np.mean(rr))
        sd = float(np.std(rr, ddof=0))
        diffs = np.diff(rr)
        rms = float(np.sqrt(np.mean(diffs * diffs)))
        nn50 = int(np.sum(np.abs(diffs) > 0.050))  # > 50 ms difference
        hr = 60.0 / m if m > 0 else 0.0
        cv = (sd / m) if m > 0 else 0.0

    # Rhythm classification logic
    cond = "Bradycardia" if hr < 60.0 else ("Tachycardia" if hr > 100.0 else "Normal heart rate")
    if len(rr) < 3:
        rhy = "Insufficient beats"
    elif cv > 0.22:
        rhy = "Atrial fibrillation suspected"
    elif cv > 0.09:
        rhy = "Other arrhythmia suspected"
    else:
        rhy = "Normal sinus rhythm"

    # Clinical advice
    if rhy == "Insufficient beats":
        prog = "Could not detect enough beats: check the sampling rate, column and signal quality."
    elif rhy == "Normal sinus rhythm" and 60.0 <= hr <= 100.0:
        prog = "No problem detected"
    elif "Atrial" in rhy or "Other" in rhy:
        prog = "Consult doctor ASAP" if hr < 60.0 else "Doctor review needed"
    else:
        prog = "Heart rate outside normal range: doctor review suggested"

    items = [
        ["Rhythm", rhy],
        ["Condition", cond],
        ["Heart rate", f"{hr:.0f} bpm"],
        ["R peaks", str(len(peaks))],
        ["Avg RR", f"{m:.2f} s"],
        ["HRV (SDNN)", f"{sd * 1000:.0f} ms"],
        ["RMSSD", f"{rms * 1000:.0f} ms"],
        ["NN50", str(nn50)],
        ["RR variation", f"{cv * 100:.0f}%"],
    ]

    return {
        "p": peaks,
        "y": h.tolist(),
        "rr": rr,
        "hr": float(hr),
        "m": float(m),
        "sd": float(sd),
        "rms": float(rms),
        "nn50": int(nn50),
        "cv": float(cv),
        "rhythm": rhy,
        "condition": cond,
        "prog": prog,
        "advice": f"Advice: {prog} (research demo, not a medical device)",
        "items": items,
    }
