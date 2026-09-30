"""Measure synthetic ECG pipeline latency and traced peak memory.

Run ``python benchmark.py`` or choose sizes with ``python benchmark.py --sizes 250 1250 5000``.
The script uses deterministic synthetic samples and does not load patient data.
Memory is the Python/NumPy allocation peak reported by ``tracemalloc``; it is not
the operating system's total resident-set size.
"""

from __future__ import annotations

import argparse
import statistics
import time
import tracemalloc
from typing import Any, Dict, List

import numpy as np

from backend.dsp import clean_baseline, compute_sqi, detect_rpeaks
from backend.fuzzy import evaluate_fuzzy
from backend.synthetic import ECGSimulator, FS


def analyze_record(samples: np.ndarray, fs: int = FS) -> Dict[str, Any]:
    """Run cleaning, peak detection, signal quality, and fuzzy inference.

    Args:
        samples: Finite one-dimensional ECG trace.
        fs: Sampling frequency in hertz.

    Returns:
        Detector peaks, SQI, and fuzzy classification result.

    Complexity:
        O(N) temporary memory and several linear passes plus NumPy percentile
        selection for N input samples.
    """
    cleaned = clean_baseline(samples, bw=min(200, max(1, len(samples))))
    detection = detect_rpeaks(cleaned, sm=5, th=0.30, mw=25, fs=fs)
    peaks: List[int] = detection["pk"]
    quality = compute_sqi(cleaned)

    rr_seconds = np.diff(peaks).astype(np.float64) / fs if len(peaks) > 1 else np.array([])
    heart_rate = float(60.0 / np.median(rr_seconds)) if rr_seconds.size else None
    mean_rr = float(np.mean(rr_seconds)) if rr_seconds.size else 0.0
    cv = float(np.std(rr_seconds) / mean_rr) if mean_rr > 0.0 else None
    classification = evaluate_fuzzy(heart_rate, cv, quality)

    return {
        "peaks": peaks,
        "sqi": quality,
        "classification": classification,
    }


def benchmark_case(sample_count: int, repeats: int, seed: int = 2026) -> Dict[str, float]:
    """Benchmark one synthetic input size and return latency/throughput/memory.

    The signal generation cost is excluded from timing so results represent the
    analysis pipeline rather than the Python ECG simulator.
    """
    if sample_count < 5:
        raise ValueError("sample_count must be at least 5")
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    simulator = ECGSimulator(seed=seed, fs=FS)
    record = simulator.generate_record(seconds=sample_count / FS)
    samples = record["signal"]

    analyze_record(samples, FS)  # one warm-up pass, excluded from measurements
    timings_ms: List[float] = []
    peak_bytes = 0
    tracemalloc.start()
    try:
        for _ in range(repeats):
            tracemalloc.reset_peak()
            started = time.perf_counter()
            analyze_record(samples, FS)
            timings_ms.append((time.perf_counter() - started) * 1000.0)
            _, case_peak = tracemalloc.get_traced_memory()
            peak_bytes = max(peak_bytes, case_peak)
    finally:
        tracemalloc.stop()

    median_ms = statistics.median(timings_ms)
    p95_ms = float(np.percentile(timings_ms, 95))
    return {
        "samples": float(sample_count),
        "median_ms": float(median_ms),
        "p95_ms": p95_ms,
        "samples_per_second": float(sample_count / (median_ms / 1000.0)),
        "traced_peak_kib": float(peak_bytes / 1024.0),
    }


def main() -> None:
    """Print benchmark rows for requested input sizes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[250, 1250, 5000, 12500],
        help="sample counts to benchmark (default: 1s, 5s, 20s, 50s at 250 Hz)",
    )
    parser.add_argument("--repeats", type=int, default=5, help="measured runs per size")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if any(size < 5 for size in args.sizes):
        parser.error("each --sizes value must be at least 5")

    print("Synthetic ECG analysis benchmark (generation excluded)")
    print(f"{'samples':>9} {'median ms':>12} {'p95 ms':>10} {'samples/s':>14} {'traced peak KiB':>17}")
    for sample_count in args.sizes:
        result = benchmark_case(sample_count, args.repeats)
        print(
            f"{int(result['samples']):>9} "
            f"{result['median_ms']:>12.3f} "
            f"{result['p95_ms']:>10.3f} "
            f"{result['samples_per_second']:>14.1f} "
            f"{result['traced_peak_kib']:>17.1f}"
        )


if __name__ == "__main__":
    main()
