"""
backend/dsp.py — Digital Signal Processing module for Real-Time ICU Arrhythmia Monitoring.

Implements:
1. Moving-average baseline wander removal (Clean) preserving QRS amplitude >= 85%.
2. Adaptive-threshold detector with 200 ms refractory blanking period (Detect).
3. Statistical Signal Quality Index (SQI) computation.

Compliant with PEP-484 type annotations and structured complexity analysis.
"""

from __future__ import annotations
import numpy as np
from scipy import signal as sp_signal
from typing import Dict, List, Tuple, Any, Optional

FS_DEFAULT: int = 250  # Hz default sampling rate


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """
    Computes a centered moving average using cumulative sum with boundary clamping.

    Parameters
    ----------
    x : np.ndarray
        1D input signal array of dtype float32 or float64. Shape: (N,).
    window : int
        Window width in samples. Must be positive.

    Returns
    -------
    np.ndarray
        Smoothed output array with identical length and shape as input: (N,).

    Pre-conditions
    --------------
    - `x` must be a 1D real-valued numpy array.
    - `window` >= 1.

    Post-conditions
    ---------------
    - Output array shape equals `x.shape`.
    - Memory allocated is O(N) with zero infinite or NaN entries unless present in `x`.

    Complexity
    ----------
    - Time Complexity: O(N) where N is number of samples.
    - Space Complexity: O(N) auxiliary space for cumulative sum buffer.
    """
    if not isinstance(x, np.ndarray):
        x = np.asarray(x, dtype=np.float32)

    n: int = int(x.shape[0])
    if window <= 1 or n == 0:
        return x.copy()

    window = min(window, n)
    half: int = window // 2
    cumsum: np.ndarray = np.empty(n + 1, dtype=np.float64)
    cumsum[0] = 0.0
    np.cumsum(x, out=cumsum[1:])

    # Boundary-clamped moving average calculation
    idx_a: np.ndarray = np.maximum(0, np.arange(n) - half)
    idx_b: np.ndarray = np.minimum(n, np.arange(n) + half + 1)
    counts: np.ndarray = idx_b - idx_a
    out: np.ndarray = (cumsum[idx_b] - cumsum[idx_a]) / counts
    return out.astype(np.float32)


def clean_baseline(x: np.ndarray, bw: int) -> np.ndarray:
    """
    Subtracts moving-average baseline estimate from raw ECG input to eliminate
    respiration wander and perspiration sway while strictly preserving high-frequency
    QRS spike amplitude (>= 85% of true 1.0 mV).

    Parameters
    ----------
    x : np.ndarray
        Raw ECG voltage series in mV. Shape: (N,).
    bw : int
        Baseline removal filter window size in samples (typically 50-500).

    Returns
    -------
    np.ndarray
        Cleaned zero-centered ECG trace in mV. Shape: (N,).

    Pre-conditions
    --------------
    - `x` is a 1D float32 or float64 numpy array.
    - `bw` >= 10 samples.

    Post-conditions
    ---------------
    - Signal DC offset is centered around 0 mV.
    - Output shape is identical to input: (N,).

    Complexity
    ----------
    - Time Complexity: O(N) linear time filter.
    - Space Complexity: O(N) auxiliary memory.
    """
    if not isinstance(x, np.ndarray):
        x = np.asarray(x, dtype=np.float32)
    baseline: np.ndarray = moving_average(x, bw)
    cleaned: np.ndarray = x - baseline
    return cleaned


def detect_rpeaks(
    y: np.ndarray,
    sm: int,
    th: float,
    mw: int,
    fs: int = FS_DEFAULT
) -> Dict[str, Any]:
    """
    Adaptive-threshold QRS R-peak detector featuring an explicit 200 ms refractory
    blanking period to eliminate double-counts and T-wave triggering.

    Parameters
    ----------
    y : np.ndarray
        Cleaned ECG signal array in mV. Shape: (N,).
    sm : int
        Pre-filter smoothing window size in samples (e.g., 1-15).
    th : float
        Adaptive threshold factor applied to the 97th percentile energy (0.05 - 0.90).
    mw : int
        Moving window integrator duration in samples (e.g., 8-50).
    fs : int, optional
        Sampling frequency in Hertz (default: 250 Hz).

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - "pk": List[int] — Discrete sample indices of detected R-peaks.
        - "threshold": float — Computed adaptive energy threshold value.
        - "energy": np.ndarray — Integrated slope energy trace. Shape: (N,).

    Pre-conditions
    --------------
    - `y` must be a 1D real-valued array of length >= 3.
    - `sm` >= 1, `mw` >= 1, `th` > 0.0, `fs` >= 50.

    Post-conditions
    ---------------
    - Every consecutive detected peak pair (p_{k}, p_{k-1}) satisfies:
      (p_{k} - p_{k-1}) >= 0.20 * fs (>= 200 ms refractory guarantee).

    Complexity
    ----------
    - Time Complexity: O(N) linear pass across the discrete sample window.
    - Space Complexity: O(N) for difference and energy envelopes.
    """
    if not isinstance(y, np.ndarray):
        y = np.asarray(y, dtype=np.float32)

    n: int = int(y.shape[0])
    if n < 3:
        return {"pk": [], "threshold": 0.0, "energy": np.zeros(n, dtype=np.float32)}

    # Step 1: Smooth
    s: np.ndarray = moving_average(y, max(1, int(sm)))

    # Step 2: Differentiate & Square (emphasizing QRS steep slopes)
    diff: np.ndarray = np.zeros(n, dtype=np.float32)
    diff[2:] = s[2:] - s[:-2]
    energy: np.ndarray = diff * diff

    # Step 3: Moving Window Integration
    e_int: np.ndarray = moving_average(energy, max(1, int(mw)))

    # Step 4: Adaptive Threshold Calculation
    p97: float = float(np.percentile(e_int, 97))
    p50: float = float(np.percentile(e_int, 50))
    threshold: float = max(float(th) * p97, 2.5 * p50)

    # Step 5: Peak search with 200 ms refractory blanking
    refractory_samples: int = int(0.20 * fs)  # Exactly 200 ms blanking interval
    peaks: List[int] = []
    last_peak: int = -999999

    for i in range(1, n - 1):
        if e_int[i] > threshold and e_int[i] >= e_int[i - 1] and e_int[i] > e_int[i + 1]:
            # Refine peak to maximum in original smoothed signal within +- 15 samples
            a: int = max(0, i - 15)
            b: int = min(n, i + 16)
            best_idx: int = a + int(np.argmax(s[a:b]))

            if best_idx - last_peak > refractory_samples:
                peaks.append(best_idx)
                last_peak = best_idx

    return {"pk": peaks, "threshold": threshold, "energy": e_int}


def compute_sqi(y: np.ndarray) -> float:
    """
    Computes statistical Signal Quality Index (SQI) mapped to [0.0, 1.0].
    Measures ratio between 98th percentile peak envelope and second-derivative
    high-frequency noise dispersion.

    Parameters
    ----------
    y : np.ndarray
        Cleaned ECG signal slice. Shape: (N,).

    Returns
    -------
    float
        Continuous quality metric in [0.0, 1.0].
        (1.0 = Pristine diagnostic signal, 0.0 = Severe artifact / Disconnected).

    Complexity
    ----------
    - Time Complexity: O(N log N) dominated by percentile sort.
    - Space Complexity: O(N) for absolute derivative vector.
    """
    if not isinstance(y, np.ndarray):
        y = np.asarray(y, dtype=np.float32)

    n: int = int(y.shape[0])
    if n < 5:
        return 1.0

    d2: np.ndarray = np.abs(y[2:] - 2.0 * y[1:-1] + y[:-2])
    abs_y: np.ndarray = np.abs(y)

    sg: float = float(np.percentile(d2, 50)) / 1.652 + 1e-6
    r: float = float(np.percentile(abs_y, 98)) / sg
    sqi: float = (np.log10(max(r, 1e-6)) - 0.4) / 0.9
    return float(np.clip(sqi, 0.0, 1.0))
