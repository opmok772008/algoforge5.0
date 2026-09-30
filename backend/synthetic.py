"""
backend/synthetic.py — Realistic synthetic ECG signal generator with ground truth.
Supports:
  - Demonstration scenarios: normal rhythm, Ventricular Tachycardia (VT), and
    Ventricular Fibrillation (VF).
  - Noise: Baseline wander, Motion artifact, Severe EMG burst.
  - Generates ground-truth R-peak locations for quantitative accuracy and jitter benchmarking.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List
import numpy as np

FS = 250


class SeededRNG:
    """Small deterministic pseudo-random generator used by repeatable demos."""

    def __init__(self, seed: int = 42) -> None:
        """Initialize the generator from an integer seed."""
        self.state = int(seed) & 0xFFFFFFFF

    def next(self) -> float:
        """Return the next deterministic value in the half-open interval [0, 1)."""
        # Mulberry32 PRNG for deterministic, reproducible pseudo-random numbers
        self.state = (self.state + 0x6D2B79F5) & 0xFFFFFFFF
        t = (self.state ^ (self.state >> 15)) * (1 | self.state)
        t = t & 0xFFFFFFFF
        t = (t + ((t ^ (t >> 7)) * (61 | t))) & 0xFFFFFFFF
        res = (t ^ (t >> 14)) & 0xFFFFFFFF
        return res / 4294967296.0

    def gauss(self) -> float:
        """Return one standard-normal sample using the Box-Muller transform."""
        # Box-Muller transform
        u1 = max(1e-12, self.next())
        u2 = self.next()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def gaussian_peak(x: float, m: float, s: float, a: float) -> float:
    """Evaluate a Gaussian pulse with center ``m``, width ``s``, and amplitude ``a``."""
    if not math.isfinite(s) or s <= 0.0:
        raise ValueError("s must be a finite positive width")
    return a * math.exp(-((x - m) * (x - m)) / (2.0 * s * s))


class ECGSimulator:
    """Generate deterministic synthetic ECG samples and known R-peak locations."""

    def __init__(self, seed: int = 42, fs: int = FS) -> None:
        """Create a simulator; ``fs`` is the positive sample rate in hertz."""
        if isinstance(fs, bool) or not isinstance(fs, int) or fs < 1:
            raise ValueError("fs must be a positive integer sampling rate")
        self.fs = fs
        self.rng = SeededRNG(seed)
        self.phase = 0.0
        self.rr = 0.85
        self.t = 0.0
        self.vp = 0.0
        self.index = 0
        self.ground_truth_rpeaks: List[int] = []

    def next_sample(
        self,
        scenario: str = "normal",
        wander: bool = False,
        motion: bool = False,
        emg: bool = False,
    ) -> float:
        """Advance the simulator by one sample for ``normal``, ``vt``, or ``vf``."""
        if scenario not in {"normal", "vt", "vf"}:
            raise ValueError("scenario must be one of: normal, vt, vf")
        t = self.t + 1.0 / self.fs
        self.t = t
        i = self.index
        self.index += 1
        two_pi = 6.2831853

        if scenario == "vf":
            # Ventricular Fibrillation: chaotic sinusoidal frequency modulated waveform
            f = 5.0 + 1.2 * math.sin(two_pi * 0.3 * t) + 0.6 * math.sin(two_pi * 0.83 * t)
            self.vp += two_pi * f / self.fs
            x = (0.7 + 0.3 * math.sin(two_pi * 0.9 * t)) * math.sin(self.vp) + 0.3 * math.sin(1.7 * self.vp + 1.0)
        else:
            vt = (scenario == "vt")
            prev_phase = self.phase
            peak_phase = 0.35 if vt else 0.385
            self.phase += 1.0 / (self.fs * self.rr)
            
            if self.phase >= 1.0:
                self.phase -= 1.0
                self.rr = (0.33 + self.rng.next() * 0.02) if vt else (0.85 + self.rng.next() * 0.06)
                
            p = self.phase
            if prev_phase < peak_phase and p >= peak_phase:
                self.ground_truth_rpeaks.append(i)
                
            if vt:
                # Wide, bizarre QRS complexes typical of VT
                x = gaussian_peak(p, 0.35, 0.05, 1.0) + gaussian_peak(p, 0.70, 0.07, -0.45)
            else:
                # Classic P-Q-R-S-T wave components
                p_wave = gaussian_peak(p, 0.16, 0.025, 0.12)
                q_wave = gaussian_peak(p, 0.36, 0.008, -0.15)
                r_wave = gaussian_peak(p, 0.385, 0.011, 1.0)
                s_wave = gaussian_peak(p, 0.41, 0.009, -0.25)
                t_wave = gaussian_peak(p, 0.62, 0.045, 0.30)
                x = p_wave + q_wave + r_wave + s_wave + t_wave

        # Add Noise and Artifacts
        if wander:
            # Baseline wander (respiration and perspiration 0.1-0.3 Hz)
            x += 0.8 * math.sin(two_pi * 0.25 * t) + 0.4 * math.sin(two_pi * 0.11 * t + 1.0)
            
        if motion:
            # Patient movement artifact
            env = math.sin(two_pi * 0.2 * t)
            x += 1.2 * (env * env) * math.sin(two_pi * 2.2 * t)
            
        # EMG burst / Gaussian noise
        noise_level = 0.9 if emg else 0.03
        x += (self.rng.next() - 0.5) * noise_level

        return x

    def generate_record(
        self,
        seconds: float,
        scenario: str = "normal",
        wander: bool = False,
        motion: bool = False,
        emg: bool = False,
    ) -> Dict[str, Any]:
        """Generate a record dictionary with signal, ground truth, rate, and scenario.

        ``seconds`` must be a finite positive duration. The returned signal has
        ``int(seconds * fs)`` float32 samples; peak indices use the same sample clock.
        """
        if not math.isfinite(seconds) or seconds <= 0.0:
            raise ValueError("seconds must be a finite positive duration")
        if scenario not in {"normal", "vt", "vf"}:
            raise ValueError("scenario must be one of: normal, vt, vf")
        num_samples = int(seconds * self.fs)
        signal = np.empty(num_samples, dtype=np.float32)
        for i in range(num_samples):
            signal[i] = self.next_sample(scenario=scenario, wander=wander, motion=motion, emg=emg)
        return {
            "signal": signal,
            "ground_truth": list(self.ground_truth_rpeaks),
            "fs": self.fs,
            "duration": seconds,
            "scenario": scenario,
        }
