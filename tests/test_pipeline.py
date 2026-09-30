"""Shape, boundary, and route-contract tests for the ECG demonstration app."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from benchmark import benchmark_case
from backend.app import AnalyzeRequest, OptimizeRequest, ParamsModel, app, health_check
from backend.dsp import clean_baseline, compute_sqi, detect_rpeaks, moving_average
from backend.features import analyze_rr_features
from backend.fuzzy import evaluate_fuzzy, trapmf
from backend.ga import GeneticAlgorithm
from backend.validation import run_all_validations


def test_moving_average_preserves_shape_and_clamps_edges() -> None:
    samples = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    result = moving_average(samples, window=3)

    assert result.shape == samples.shape
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, [1.5, 2.0, 2.5])


def test_signal_functions_reject_bad_shapes_and_non_finite_values() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        moving_average(np.zeros((2, 3), dtype=np.float32), window=2)
    with pytest.raises(ValueError, match="finite"):
        clean_baseline(np.array([0.0, np.nan, 1.0]), bw=2)


def test_detector_checks_sampling_rate_and_returns_array_shape(clean_record: dict[str, object]) -> None:
    signal = clean_record["signal"]
    assert isinstance(signal, np.ndarray)

    cleaned = clean_baseline(signal, bw=200)
    result = detect_rpeaks(cleaned, sm=5, th=0.30, mw=25, fs=250)

    assert result["energy"].shape == signal.shape
    assert isinstance(result["pk"], list)
    assert all(0 <= peak < len(signal) for peak in result["pk"])
    assert all(
        right - left >= 50
        for left, right in zip(result["pk"], result["pk"][1:])
    )
    with pytest.raises(ValueError, match="at least 50"):
        detect_rpeaks(cleaned, sm=5, th=0.30, mw=25, fs=25)


def test_signal_quality_is_bounded(wander_record: dict[str, object]) -> None:
    signal = wander_record["signal"]
    assert isinstance(signal, np.ndarray)

    quality = compute_sqi(clean_baseline(signal, bw=200))

    assert 0.0 <= quality <= 1.0


def test_fuzzy_membership_and_quality_boundaries() -> None:
    assert trapmf(0.5, 0.0, 0.5, 1.0, 1.5) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="q must"):
        evaluate_fuzzy(hr=120.0, cv=0.1, q=1.1)
    with pytest.raises(ValueError, match="trapezoid bounds"):
        trapmf(0.5, 1.0, 0.0, 1.0, 2.0)


def test_rr_features_report_short_trace_and_reject_wrong_shape(short_signal: np.ndarray) -> None:
    result = analyze_rr_features(short_signal, fs=250)
    assert result["rhythm"] == "Insufficient beats"
    assert len(result["y"]) == len(short_signal)

    with pytest.raises(ValueError, match="one-dimensional"):
        analyze_rr_features(np.zeros((2, 300), dtype=np.float32), fs=250)
    with pytest.raises(ValueError, match="sampling rate"):
        analyze_rr_features(short_signal, fs=0)


def test_analyze_request_rejects_non_numeric_signal_values() -> None:
    with pytest.raises(ValidationError):
        AnalyzeRequest(samples=[0.0, float("nan")], fs=250)


def test_parameter_models_enforce_detector_ranges() -> None:
    with pytest.raises(ValidationError):
        ParamsModel(bw=10, sm=5, th=0.30, mw=25)
    with pytest.raises(ValidationError):
        OptimizeRequest(iterations=0)
    assert ParamsModel().bw == 200


def test_monitor_keeps_fixed_buffer_and_rejects_invalid_chunk(monitor) -> None:
    monitor.run_seconds(0.4)

    assert monitor.buffer.shape == (1500,)
    assert monitor.buffer.nbytes == 6000
    assert monitor.cleaned_trace.shape == (1000,)
    with pytest.raises(ValueError, match="buffer size"):
        monitor.step(chunk_samples=1501)
    with pytest.raises(ValueError, match="scenario"):
        monitor.set_scenario("unknown")


def test_health_route_function_returns_status_payload() -> None:
    payload = asyncio.run(health_check())

    assert payload["status"] == "online"
    assert payload["streaming_buffer_bytes"] == 6000
    assert "active_params" in payload


def test_api_schema_exposes_streaming_and_health_routes() -> None:
    paths = app.openapi()["paths"]

    assert "/health" in paths
    assert "/api/stream" in paths
    assert "/api/analyze" in paths
    assert "/api/optimize" in paths


def test_builtin_synthetic_validation_suite_passes() -> None:
    result = run_all_validations()

    assert result["total"] == 9
    assert result["passed"] == 9
    assert result["all_passed"] is True


def test_benchmark_reports_latency_throughput_and_memory() -> None:
    result = benchmark_case(sample_count=250, repeats=1)

    assert result["median_ms"] > 0.0
    assert result["p95_ms"] > 0.0
    assert result["samples_per_second"] > 0.0
    assert result["traced_peak_kib"] > 0.0


def test_environment_template_documents_runtime_configuration() -> None:
    template = Path(__file__).resolve().parents[1] / ".env.example"
    contents = template.read_text(encoding="utf-8")

    assert "API_HOST=" in contents
    assert "API_PORT=" in contents
    assert "DEFAULT_FS=" in contents


def test_genetic_algorithm_rejects_too_small_population() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        GeneticAlgorithm(seed=1, pop_size=1)
