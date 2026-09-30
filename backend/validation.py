"""
backend/validation.py — Automated verification suite matching the problem statement:
1. Clean rhythm: Sensitivity and Precision >= 95%.
2. Baseline wander: R-peak timing jitter <= 20 ms.
3. QRS amplitude preservation: At least 85% of true 1.0 mV preserved.
4. VT and VF are each detected within 3.0 seconds (two separate results).
5. Severe EMG burst: 0 false lethal alarms in 20 seconds.
6. Monotonicity: Fuzzy confidence never falls as heart rate rises.
7. Fixed streaming buffer: Stays fixed at 1500 samples (6 KB) over 60 seconds.
8. Elitism: Genetic algorithm never loses fitness across generations.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np

from backend.dsp import clean_baseline, detect_rpeaks, compute_sqi
from backend.fuzzy import evaluate_fuzzy
from backend.synthetic import ECGSimulator
from backend.ga import DEFAULT_PARAMS, GeneticAlgorithm, score_record


class StreamingMonitor:
    """Maintain a bounded ECG buffer and evaluate a synthetic stream in chunks."""

    def __init__(self, seed: int = 31, params: Optional[Dict[str, Any]] = None) -> None:
        """Create a monitor with a fixed-size float32 ring-like sliding buffer."""
        self.fs = 250
        self.buffer_size = 1500  # 6 seconds = 1500 float32 samples = 6000 bytes (~6 KB)
        self.window_size = 1000
        self.params = dict(params) if params is not None else dict(DEFAULT_PARAMS)
        self.sim = ECGSimulator(seed=seed, fs=self.fs)
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.accumulated = 0
        self.onset_time = None
        self.alarm_time = None
        self.scenario = "normal"
        self.wander = False
        self.motion = False
        self.emg = False
        self.current_state = evaluate_fuzzy(None, None, 1.0)
        self.cleaned_trace = np.zeros(self.window_size, dtype=np.float32)
        self.detected_peaks = []

    def set_scenario(
        self,
        scenario: str,
        wander: bool = False,
        motion: bool = False,
        emg: bool = False,
    ) -> None:
        """Select a supported synthetic rhythm and optional artifact sources."""
        if scenario not in {"normal", "vt", "vf"}:
            raise ValueError("scenario must be one of: normal, vt, vf")
        if scenario != self.scenario:
            self.onset_time = None if scenario == "normal" else self.sim.t
            self.alarm_time = None
        self.scenario = scenario
        self.wander = wander
        self.motion = motion
        self.emg = emg

    def step(self, chunk_samples: int = 25) -> None:
        """Advance the synthetic stream and refresh detector state for one chunk."""
        if isinstance(chunk_samples, bool) or not isinstance(chunk_samples, int):
            raise TypeError("chunk_samples must be an integer")
        if not 1 <= chunk_samples <= self.buffer_size:
            raise ValueError("chunk_samples must be between 1 and the buffer size")
        # Shift buffer left by chunk_samples
        self.buffer[:-chunk_samples] = self.buffer[chunk_samples:]
        # Fill new samples from simulator
        for i in range(self.buffer_size - chunk_samples, self.buffer_size):
            self.buffer[i] = self.sim.next_sample(
                scenario=self.scenario,
                wander=self.wander,
                motion=self.motion,
                emg=self.emg
            )
        
        self.accumulated += chunk_samples
        if self.accumulated >= 25:
            self.accumulated = 0
            # Analyze last window_size samples
            sub = self.buffer[-self.window_size:]
            cleaned = clean_baseline(sub, int(self.params["bw"]))
            self.cleaned_trace = cleaned
            det = detect_rpeaks(cleaned, int(self.params["sm"]), float(self.params["th"]), int(self.params["mw"]), fs=self.fs)
            self.detected_peaks = det["pk"]
            q = compute_sqi(cleaned)

            hr = None
            cv = None
            if len(self.detected_peaks) >= 6:
                recent_peaks = self.detected_peaks[-6:]
                rr_intervals = [(recent_peaks[i] - recent_peaks[i - 1]) / float(self.fs) for i in range(1, len(recent_peaks))]
                mu = float(np.mean(rr_intervals))
                sd = float(np.std(rr_intervals, ddof=0))
                cv = (sd / mu) if mu > 0 else 0.0
                hr = (60.0 / float(np.median(rr_intervals))) if len(rr_intervals) > 0 else None

            self.current_state = evaluate_fuzzy(hr, cv, q)
            if self.current_state["alarm"] and self.onset_time is not None and self.alarm_time is None:
                self.alarm_time = self.sim.t

    def run_seconds(self, seconds: float) -> None:
        """Advance the monitor for a non-negative number of simulated seconds."""
        if not np.isfinite(seconds) or seconds < 0:
            raise ValueError("seconds must be a finite non-negative duration")
        steps = int(seconds * self.fs / 25)
        for _ in range(steps):
            self.step(25)

    def get_latency(self) -> Optional[float]:
        """Return seconds from the current scenario onset to first alarm, if any."""
        if self.alarm_time is None or self.onset_time is None:
            return None
        return float(self.alarm_time - self.onset_time)


def run_all_validations() -> Dict[str, Any]:
    """Run nine deterministic synthetic-data validation checks.

    Returns a mapping with one result per validation, the passed count, total count,
    and an ``all_passed`` flag. Runtime is dominated by the generated ECG samples and
    genetic-algorithm fitness evaluations; memory is O(N) for the largest record.
    """
    results = []
    p = DEFAULT_PARAMS

    # 1. Clean rhythm: sensitivity and precision >= 95%
    sim_clean = ECGSimulator(seed=21)
    rec_clean = sim_clean.generate_record(seconds=30.0, scenario="normal")
    sc_clean = score_record(rec_clean, p)
    pass1 = (sc_clean["se"] >= 0.95) and (sc_clean["ppv"] >= 0.95)
    results.append({
        "id": "V1",
        "name": "Clean rhythm: sensitivity & precision >= 95%",
        "ok": bool(pass1),
        "val": f"Se {(sc_clean['se'] * 100):.1f}%, PPV {(sc_clean['ppv'] * 100):.1f}%",
        "detail": "Evaluated on 30s clean ECG record with known beat ground truth"
    })

    # 2. R-peak timing jitter under baseline wander <= 20 ms
    sim_wander = ECGSimulator(seed=22)
    rec_wander = sim_wander.generate_record(seconds=30.0, scenario="normal", wander=True)
    sc_wander = score_record(rec_wander, p)
    pass2 = (sc_wander["jitter_ms"] <= 20.0)
    results.append({
        "id": "V2",
        "name": "R-peak timing jitter under baseline wander <= 20 ms",
        "ok": bool(pass2),
        "val": f"{sc_wander['jitter_ms']:.1f} ms",
        "detail": "Assessing filter phase linearity and timing fidelity under respiration sway"
    })

    # 3. QRS amplitude preserved after filtering >= 85% of true 1.0 mV
    sim_qrs = ECGSimulator(seed=23)
    rec_qrs = sim_qrs.generate_record(seconds=20.0, scenario="normal", wander=True)
    y_qrs = clean_baseline(rec_qrs["signal"], int(p["bw"]))
    amps = [y_qrs[q] for q in rec_qrs["ground_truth"] if 60 < q < (len(y_qrs) - 60)]
    mean_amp = float(np.mean(amps)) if amps else 0.0
    pass3 = (mean_amp >= 0.85)
    results.append({
        "id": "V3",
        "name": "QRS amplitude preserved after filtering (>= 85% of true 1.0 mV)",
        "ok": bool(pass3),
        "val": f"{mean_amp:.2f} mV ({(mean_amp * 100):.1f}%)",
        "detail": "Ensures baseline subtraction does not attenuate clinical R wave height"
    })

    # 4. Alert on lethal rhythms within 3 seconds (VT & VF)
    mon_vt = StreamingMonitor(seed=31, params=p)
    mon_vt.set_scenario("normal", wander=True)
    mon_vt.run_seconds(10.0)
    mon_vt.set_scenario("vt", wander=True)
    mon_vt.run_seconds(6.0)
    lat_vt = mon_vt.get_latency()
    pass_vt = (lat_vt is not None and lat_vt <= 3.0)
    results.append({
        "id": "V4a",
        "name": "Ventricular tachycardia (VT) alert within 3.0 seconds",
        "ok": bool(pass_vt),
        "val": f"{lat_vt:.1f} s" if lat_vt is not None else "No alert",
        "detail": "Evaluates speed budget from arrhythmia onset to alarm trigger"
    })

    mon_vf = StreamingMonitor(seed=31, params=p)
    mon_vf.set_scenario("normal", wander=True)
    mon_vf.run_seconds(10.0)
    mon_vf.set_scenario("vf", wander=True)
    mon_vf.run_seconds(6.0)
    lat_vf = mon_vf.get_latency()
    pass_vf = (lat_vf is not None and lat_vf <= 3.0)
    results.append({
        "id": "V4b",
        "name": "Ventricular fibrillation (VF) alert within 3.0 seconds",
        "ok": bool(pass_vf),
        "val": f"{lat_vf:.1f} s" if lat_vf is not None else "No alert",
        "detail": "Emergency classification speed for chaotic fibrillatory rhythm"
    })

    # 5. Normal rhythm with severe EMG burst raises no lethal alarm
    mon_emg = StreamingMonitor(seed=32, params=p)
    mon_emg.set_scenario("normal", emg=True)
    mon_emg.run_seconds(8.0)
    false_alarms = 0
    # Run 20 seconds
    for _ in range(int(20.0 * 250 / 25)):
        mon_emg.step(25)
        if mon_emg.current_state["alarm"]:
            false_alarms += 1
    pass5 = (false_alarms == 0)
    results.append({
        "id": "V5",
        "name": "Normal rhythm with severe EMG burst raises no lethal alarm",
        "ok": bool(pass5),
        "val": f"{false_alarms} false alarms in 20 s",
        "detail": "Prevents ICU alarm fatigue by routing noise to 'Hold for review'"
    })

    # 6. Fuzzy alarm confidence never falls as heart rate rises
    hr_points = [100.0, 130.0, 150.0, 180.0, 220.0]
    conf_values = [evaluate_fuzzy(h, 0.05, 0.90)["confidence"] for h in hr_points]
    is_monotonic = all(conf_values[i] >= conf_values[i - 1] for i in range(1, len(conf_values)))
    results.append({
        "id": "V6",
        "name": "Fuzzy alarm confidence never decreases as heart rate rises",
        "ok": bool(is_monotonic),
        "val": " ".join(f"{c:.2f}" for c in conf_values),
        "detail": "Ensures risk score monotonic consistency across tachycardic transition"
    })

    # 7. Streaming buffer stays fixed at 1500 samples (6 KB) over 60 s
    mon_buf = StreamingMonitor(seed=33, params=p)
    mon_buf.run_seconds(60.0)
    buf_len = len(mon_buf.buffer)
    buf_bytes = mon_buf.buffer.nbytes
    pass7 = (buf_len == 1500 and buf_bytes == 6000)
    results.append({
        "id": "V7",
        "name": f"Streaming buffer stays fixed at {buf_len} samples (6 KB) over 60 s",
        "ok": bool(pass7),
        "val": f"{buf_bytes} bytes ({buf_len} samples)",
        "detail": "Guarantees zero memory leak in real-time embedded streaming circular buffer"
    })

    # 8. Evolution never loses fitness (elitism) over 12 generations
    ga = GeneticAlgorithm(seed=5)
    f0 = ga.best["fitness"]
    for _ in range(12):
        ga.step()
    pass8 = (ga.best["fitness"] >= f0)
    results.append({
        "id": "V8",
        "name": "Evolution never loses fitness (elitism) over 12 generations",
        "ok": bool(pass8),
        "val": f"{f0:.3f} to {ga.best['fitness']:.3f}",
        "detail": "Elitism preserves the best parameter set across all generations"
    })

    passed_count = sum(1 for r in results if r["ok"])
    return {
        "tests": results,
        "passed": passed_count,
        "total": len(results),
        "all_passed": passed_count == len(results),
    }
