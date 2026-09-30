"""Reusable deterministic fixtures for the ECG pipeline test suite."""

from __future__ import annotations

import numpy as np
import pytest

from backend.synthetic import ECGSimulator
from backend.validation import StreamingMonitor


@pytest.fixture
def clean_record() -> dict[str, object]:
    """A deterministic five-second clean rhythm record with known R peaks."""
    return ECGSimulator(seed=101).generate_record(seconds=5.0, scenario="normal")


@pytest.fixture
def wander_record() -> dict[str, object]:
    """A deterministic record containing baseline wander for filter checks."""
    return ECGSimulator(seed=102).generate_record(
        seconds=5.0, scenario="normal", wander=True
    )


@pytest.fixture
def short_signal() -> np.ndarray:
    """A finite one-dimensional trace shorter than the feature minimum."""
    return np.linspace(-0.1, 0.1, 100, dtype=np.float32)


@pytest.fixture
def monitor() -> StreamingMonitor:
    """An isolated streaming monitor with deterministic state."""
    return StreamingMonitor(seed=103)
