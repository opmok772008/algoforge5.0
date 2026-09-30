"""
backend/ga.py — Genetic Algorithm (GA) for detector parameter optimization.
Tunes 4 settings:
  - bw (baseline window): [50, 500]
  - sm (smoothing window): [1, 15]
  - th (threshold): [0.05, 0.90]
  - mw (moving window integration): [8, 50]

Evaluated on 3 noisy training records:
  1. Record 1: Baseline wander
  2. Record 2: Severe EMG burst
  3. Record 3: Baseline wander + Motion artifact + Severe EMG burst

Fitness = mean( F1 - 0.005 * jitter_ms ) across the 3 noisy records.
Elitism preserves the best solution in every generation.
"""

from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
from backend.dsp import clean_baseline, detect_rpeaks
from backend.synthetic import ECGSimulator, SeededRNG

PARAM_BOUNDS = {
    "bw": (50, 500),
    "sm": (1, 15),
    "th": (0.05, 0.90),
    "mw": (8, 50),
}

DEFAULT_PARAMS = {"bw": 200, "sm": 5, "th": 0.30, "mw": 25}

_CACHED_TRAIN_SET: List[Dict[str, Any]] = []


def get_training_records() -> List[Dict[str, Any]]:
    """Return the three deterministic synthetic records used to score candidates.

    The records are generated once per process and then reused. Runtime and cached
    storage are O(KN), where K is the fixed record count and N samples per record.
    """
    global _CACHED_TRAIN_SET
    if not _CACHED_TRAIN_SET:
        # Create 3 deterministic noisy training records (10 seconds each at 250 Hz)
        sim1 = ECGSimulator(seed=11)
        r1 = sim1.generate_record(seconds=10.0, scenario="normal", wander=True)

        sim2 = ECGSimulator(seed=12)
        r2 = sim2.generate_record(seconds=10.0, scenario="normal", emg=True)

        sim3 = ECGSimulator(seed=13)
        r3 = sim3.generate_record(seconds=10.0, scenario="normal", wander=True, motion=True, emg=True)

        _CACHED_TRAIN_SET = [r1, r2, r3]
    return _CACHED_TRAIN_SET


def score_record(record: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, float]:
    """Measure detector sensitivity, precision, and timing error for one record.

    ``record`` must contain ``signal``, ``ground_truth``, and ``fs``. ``params`` must
    contain the detector values ``bw``, ``sm``, ``th``, and ``mw``. Results include
    the ratios ``se`` and ``ppv``, mean matched-peak error in milliseconds, and TP/FP/FN.
    The matching pass is O(P*T) for P detections and T true peaks.
    """
    signal = record["signal"]
    truth = record["ground_truth"]
    fs = record["fs"]
    n = len(signal)

    bw = int(params["bw"])
    sm = int(params["sm"])
    th = float(params["th"])
    mw = int(params["mw"])

    cleaned = clean_baseline(signal, bw)
    det = detect_rpeaks(cleaned, sm, th, mw, fs=fs)
    detected_peaks = det["pk"]

    # Exclude boundary beats (< 40 samples from ends)
    det_filtered = [p for p in detected_peaks if 40 < p < (n - 40)]
    truth_filtered = [t for t in truth if 40 < t < (n - 40)]

    tp = 0
    jitter_sum = 0.0
    used_truth = set()
    tolerance_samples = 11  # ~44 ms match window at 250 Hz

    for p in det_filtered:
        best_dist = tolerance_samples
        best_idx = -1
        for idx, t in enumerate(truth_filtered):
            if idx not in used_truth:
                dist = abs(t - p)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
        if best_idx >= 0:
            used_truth.add(best_idx)
            tp += 1
            jitter_sum += best_dist

    fp = len(det_filtered) - tp
    fn = len(truth_filtered) - tp

    se = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    jitter_ms = (jitter_sum / tp) * (1000.0 / fs) if tp > 0 else 0.0

    return {"se": se, "ppv": ppv, "jitter_ms": jitter_ms, "tp": tp, "fp": fp, "fn": fn}


def calculate_fitness(params: Dict[str, Any]) -> float:
    """Return mean F1 minus a fixed R-peak jitter penalty over training records.

    The objective is ``mean(F1) - 0.005 * mean(jitter_ms)``. The function assumes
    the detector parameter keys and ranges defined by ``PARAM_BOUNDS``.
    """
    records = get_training_records()
    total_f1 = 0.0
    total_jitter = 0.0

    for rec in records:
        metrics = score_record(rec, params)
        se = metrics["se"]
        ppv = metrics["ppv"]
        f1 = (2.0 * se * ppv) / (se + ppv) if (se + ppv) > 0 else 0.0
        total_f1 += f1
        total_jitter += metrics["jitter_ms"]

    mean_f1 = total_f1 / len(records)
    mean_jitter = total_jitter / len(records)
    # Fitness is F1 score minus an R-peak jitter penalty
    fitness = mean_f1 - 0.005 * mean_jitter
    return float(fitness)


class GeneticAlgorithm:
    """Optimize the four detector settings using deterministic tournament selection."""

    def __init__(self, seed: int = 5, pop_size: int = 14) -> None:
        """Create and score an initial population; ``pop_size`` must be at least two."""
        if isinstance(pop_size, bool) or not isinstance(pop_size, int) or pop_size < 2:
            raise ValueError("pop_size must be an integer of at least 2")
        self.rng = SeededRNG(seed)
        self.pop_size = pop_size
        self.generation = 0
        self.population: List[Dict[str, Any]] = []
        
        # Initialize population
        for _ in range(pop_size):
            p = self.random_individual()
            self.population.append(p)
        
        # Ensure default is evaluated in initial population
        self.population[0] = dict(DEFAULT_PARAMS)
        self.evaluate_population()

    def random_individual(self) -> Dict[str, Any]:
        """Sample one parameter dictionary inside the configured bounds in O(1)."""
        return {
            "bw": int(round(PARAM_BOUNDS["bw"][0] + self.rng.next() * (PARAM_BOUNDS["bw"][1] - PARAM_BOUNDS["bw"][0]))),
            "sm": int(round(PARAM_BOUNDS["sm"][0] + self.rng.next() * (PARAM_BOUNDS["sm"][1] - PARAM_BOUNDS["sm"][0]))),
            "th": float(PARAM_BOUNDS["th"][0] + self.rng.next() * (PARAM_BOUNDS["th"][1] - PARAM_BOUNDS["th"][0])),
            "mw": int(round(PARAM_BOUNDS["mw"][0] + self.rng.next() * (PARAM_BOUNDS["mw"][1] - PARAM_BOUNDS["mw"][0]))),
        }

    def evaluate_population(self) -> None:
        """Score and sort the current population, updating best and mean fitness."""
        for ind in self.population:
            if "fitness" not in ind:
                ind["fitness"] = calculate_fitness(ind)
        # Sort descending by fitness
        self.population.sort(key=lambda x: x["fitness"], reverse=True)
        self.best = dict(self.population[0])
        self.mean_fitness = sum(x["fitness"] for x in self.population) / len(self.population)

    def step(self) -> Dict[str, Any]:
        """Evolve one generation and return fitness and best-parameter summaries.

        Elitism copies the current best candidate unchanged. Runtime scales with the
        population size times the cost of scoring one candidate over all records.
        """
        # Elitism: retain exact best individual
        new_pop = [dict(self.population[0])]

        def tournament_pick() -> Dict[str, Any]:
            i1 = int(self.rng.next() * self.pop_size)
            i2 = int(self.rng.next() * self.pop_size)
            p1 = self.population[min(i1, self.pop_size - 1)]
            p2 = self.population[min(i2, self.pop_size - 1)]
            return p1 if p1["fitness"] > p2["fitness"] else p2

        while len(new_pop) < self.pop_size:
            p1 = tournament_pick()
            p2 = tournament_pick()
            child = {}
            for k in ["bw", "sm", "th", "mw"]:
                weight = self.rng.next()
                val = weight * p1[k] + (1.0 - weight) * p2[k]
                # Mutation with 30% probability
                if self.rng.next() < 0.30:
                    span = PARAM_BOUNDS[k][1] - PARAM_BOUNDS[k][0]
                    val += (self.rng.next() - 0.5) * 0.30 * span
                
                # Clamp within boundaries
                val = np.clip(val, PARAM_BOUNDS[k][0], PARAM_BOUNDS[k][1])
                child[k] = float(val) if k == "th" else int(round(val))
            
            new_pop.append(child)

        self.population = new_pop
        self.generation += 1
        self.evaluate_population()

        return {
            "generation": self.generation,
            "best_fitness": self.best["fitness"],
            "mean_fitness": self.mean_fitness,
            "best_params": {
                "bw": self.best["bw"],
                "sm": self.best["sm"],
                "th": round(self.best["th"], 2),
                "mw": self.best["mw"],
            },
        }
