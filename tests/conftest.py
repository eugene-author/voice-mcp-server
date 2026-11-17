"""Shared fixtures."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tests.harness import Harness, build_harness
from tests.mocks.clock import MockClock, MockMonotonicClock


@pytest.fixture
def mock_clock() -> MockClock:
    """Return a controllable wall clock."""
    return MockClock()


@pytest.fixture
def mock_monotonic() -> MockMonotonicClock:
    """Return a controllable monotonic clock."""
    return MockMonotonicClock()


@pytest.fixture
def harness(tmp_path: Path) -> Iterator[Harness]:
    """Return the default test application and shut its executor down afterwards."""
    app = build_harness(tmp_path)
    yield app
    app.executor.shutdown()


def write_tone(path: Path, *, seconds: float = 2.0, sample_rate: int = 16000, amplitude: float = 0.2) -> Path:
    """Write a mono sine-wave WAV file and return its path."""
    samples = amplitude * np.sin(2 * np.pi * 220 * np.arange(int(seconds * sample_rate)) / sample_rate)
    sf.write(str(path), samples.astype(np.float32), sample_rate)
    return path
