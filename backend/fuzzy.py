"""
backend/fuzzy.py — Fuzzy Logic Reasoning Engine & Safety Gate for ICU Arrhythmia Alarms.

Combines:
  - Heart Rate (HR in bpm)
  - Rhythm Regularity (CV: Coefficient of Variation of RR intervals)
  - Signal Quality (Q / SQI in [0.0, 1.0])

Rule Matrix:
  - VT: High HR + Regular rhythm
  - VF: Very High / Chaotic HR + Good signal quality
  - Noise: Poor quality + Irregular variation -> Held for clinical review (no false alarm)
  - Alarm: Confidence >= 0.6 and Noise < 0.5
  - Hold: (Confidence < 0.6 and Noise >= 0.5) or (Poor quality and low confidence)

Compliant with PEP-484 type annotations and structured complexity analysis.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np


def trapmf(x: Optional[float], a: float, b: float, c: float, d: float) -> float:
    """
    Evaluates a trapezoidal fuzzy membership function at point `x`.

    Parameters
    ----------
    x : Optional[float]
        Evaluation coordinate (e.g. heart rate or CV). If None, returns 0.0.
    a : float
        Left zero-membership threshold (feet).
    b : float
        Left full-membership shoulder.
    c : float
        Right full-membership shoulder.
    d : float
        Right zero-membership threshold (feet).

    Returns
    -------
    float
        Fuzzy membership grade in closed interval [0.0, 1.0].

    Pre-conditions
    --------------
    - a <= b <= c <= d.

    Post-conditions
    ---------------
    - 0.0 <= result <= 1.0.

    Complexity
    ----------
    - Time Complexity: O(1) constant time.
    - Space Complexity: O(1) constant space.
    """
    if not (a <= b <= c <= d):
        raise ValueError("trapezoid bounds must satisfy a <= b <= c <= d")
    if x is None:
        return 0.0
    if not np.isfinite(x):
        raise ValueError("x must be finite or None")
    val_ab: float = (x - a) / (b - a) if b != a else 1.0
    val_dc: float = (d - x) / (d - c) if d != c else 1.0
    membership: float = min(val_ab, 1.0, val_dc)
    return float(np.clip(membership, 0.0, 1.0))


def evaluate_fuzzy(hr: Optional[float], cv: Optional[float], q: float) -> Dict[str, Any]:
    """
    Performs fuzzy inference over physiological vitals and signal quality metrics,
    routing noisy transient artifacts to 'Hold for Review' and confirmed ventricular
    events to emergency 'Lethal Alarm'.

    Parameters
    ----------
    hr : Optional[float]
        Patient heart rate in beats per minute (bpm). None if uncomputed.
    cv : Optional[float]
        Coefficient of variation of RR intervals (standard deviation / mean RR).
    q : float
        Signal Quality Index in [0.0, 1.0].

    Returns
    -------
    Dict[str, Any]
        Inference results dictionary containing:
        - "hr": Optional[float] — Evaluated heart rate.
        - "cv": Optional[float] — Evaluated regularity coefficient.
        - "q": float — Evaluated signal quality.
        - "memberships": Dict[str, float] — Grades for hrHigh, hrVery, regular, irregular, good, poor.
        - "vt": float — VT rule activation strength in [0, 1].
        - "vf": float — VF rule activation strength in [0, 1].
        - "noise": float — Noise suppression rule activation strength in [0, 1].
        - "confidence": float — Overall arrhythmia confidence max(vt, vf).
        - "alarm": bool — True if emergency audible/visual alarm must fire.
        - "hold": bool — True if signal must be held for review to avoid false alarm.
        - "status": str — Diagnostic status label ("LETHAL ALARM", "HOLD FOR REVIEW", "NORMAL MONITORING").

    Pre-conditions
    --------------
    - 0.0 <= q <= 1.0.

    Complexity
    ----------
    - Time Complexity: O(1) evaluation of fixed trapezoidal boundaries.
    - Space Complexity: O(1) static dictionary allocation.
    """
    if not np.isfinite(q) or not 0.0 <= q <= 1.0:
        raise ValueError("q must be a finite value in [0.0, 1.0]")
    if hr is not None and not np.isfinite(hr):
        raise ValueError("hr must be finite or None")
    if cv is not None and (not np.isfinite(cv) or cv < 0.0):
        raise ValueError("cv must be finite, non-negative, or None")

    m: Dict[str, float] = {
        "hrHigh": 0.0,
        "hrVery": 0.0,
        "regular": 0.0,
        "irregular": 0.0,
        "good": trapmf(q, 0.35, 0.6, 1.0, 1.01),
        "poor": trapmf(q, -1.0, -0.5, 0.25, 0.5),
    }

    if hr is not None:
        m["hrHigh"] = trapmf(hr, 110.0, 150.0, 400.0, 401.0)
        m["hrVery"] = trapmf(hr, 180.0, 230.0, 400.0, 401.0)

    if cv is not None:
        m["regular"] = trapmf(cv, -1.0, -0.5, 0.12, 0.3)
        m["irregular"] = trapmf(cv, 0.15, 0.35, 5.0, 6.0)

    # Consequent Rule Evaluation:
    vt: float = min(m["hrHigh"], m["regular"])
    vf: float = min(m["hrVery"], m["good"])
    noise: float = min(m["poor"], m["irregular"])

    confidence: float = max(vt, vf)

    # Decision Logic:
    alarm: bool = bool((confidence >= 0.6) and (noise < 0.5))
    hold: bool = bool((confidence < 0.6 and noise >= 0.5) or (hr is not None and m["poor"] > 0.5 and confidence < 0.6))

    status_label: str = "LETHAL ALARM" if alarm else ("HOLD FOR REVIEW" if hold else "NORMAL MONITORING")

    return {
        "hr": hr,
        "cv": cv,
        "q": q,
        "memberships": m,
        "vt": vt,
        "vf": vf,
        "noise": noise,
        "confidence": confidence,
        "alarm": alarm,
        "hold": hold,
        "status": status_label,
    }
