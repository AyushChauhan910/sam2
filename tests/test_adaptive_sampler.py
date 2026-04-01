"""
Unit tests for AdaptiveTemporalSampler â€” CPU-only, no external data.

Run with:
    pytest tests/test_adaptive_sampler.py -v
"""

import math

import numpy as np
import torch
from PIL import Image

from sam2.utils.adaptive_sampler import AdaptiveTemporalSampler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_synthetic_frames(tmp_path, n=40, burst_start=15, burst_end=25):
    """
    Creates n grayscale 32x32 PNG files.
    Frames in [burst_start, burst_end] contain a moving white block.
    Returns sorted list of file path strings.
    """
    paths = []
    rng = np.random.RandomState(0)
    for i in range(n):
        arr = np.zeros((32, 32), dtype=np.uint8)
        if burst_start <= i < burst_end:
            x = (i - burst_start) * 2
            x = min(x, 32 - 8)
            arr[10:20, x : x + 8] = 255
        arr = arr + rng.randint(0, 10, arr.shape, dtype=np.uint8)
        p = str(tmp_path / f"{i:04d}.png")
        Image.fromarray(arr).save(p)
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_smoke_runs_without_error(tmp_path):
    """Basic call to sample() must not raise."""
    paths = make_synthetic_frames(tmp_path, n=20)
    sampler = AdaptiveTemporalSampler(total_frames=8)
    result = sampler.sample(paths)
    assert isinstance(result, list), "sample() must return a list"


def test_output_is_sorted_unique_correct_length(tmp_path):
    """Output must be sorted, contain no duplicates, and have exact length."""
    paths = make_synthetic_frames(tmp_path, n=40)
    sampler = AdaptiveTemporalSampler(total_frames=12)
    result = sampler.sample(paths, seed=0)

    assert len(result) == 12, f"Expected 12 frames, got {len(result)}"
    assert result == sorted(result), f"Result must be sorted, got {result}"
    assert len(result) == len(
        set(result)
    ), f"Result must have no duplicates, got {result}"


def test_always_includes_frame_zero(tmp_path):
    """Frame 0 must always be present, even with lowest motion score."""
    paths = make_synthetic_frames(tmp_path, n=40)
    sampler = AdaptiveTemporalSampler(total_frames=8)
    for seed in range(20):
        result = sampler.sample(paths, seed=seed)
        assert 0 in result, f"Frame 0 missing with seed={seed}; result={result}"


def test_motion_dense_region_overrepresented(tmp_path):
    """Burst region must get more samples than expected under uniform."""
    paths = make_synthetic_frames(tmp_path, n=40, burst_start=15, burst_end=25)
    sampler = AdaptiveTemporalSampler(total_frames=16, motion_threshold=0.03)
    result = sampler.sample(paths, seed=0)

    # Frames in [14, 26] (one frame padding around [15,25]) should be
    # overrepresented vs uniform expectation.
    # Uniform expectation: 11/40 * 16 â‰ˆ 4.4
    burst_count = sum(1 for i in result if 14 <= i <= 26)
    assert burst_count >= 6, (
        f"Burst region [14,26] should have >= 6 samples "
        f"(uniform expects ~4.4), got {burst_count}; indices={result}"
    )


def test_determinism(tmp_path):
    """Same seed must produce identical results."""
    paths = make_synthetic_frames(tmp_path, n=40)
    sampler = AdaptiveTemporalSampler(total_frames=12)

    result_a = sampler.sample(paths, seed=42)
    result_b = sampler.sample(paths, seed=42)
    assert (
        result_a == result_b
    ), f"Same seed must give identical results: {result_a} vs {result_b}"


def test_single_frame_clip(tmp_path):
    """Single-frame clip must return [0] with no exception."""
    paths = make_synthetic_frames(tmp_path, n=1)
    sampler = AdaptiveTemporalSampler(total_frames=8)
    result = sampler.sample(paths)
    assert result == [0], f"Single-frame clip must return [0], got {result}"


def test_uniform_fallback_on_all_zero_scores(tmp_path, monkeypatch):
    """When all motion scores are zero, output must be approximately uniform."""
    paths = make_synthetic_frames(tmp_path, n=40)
    sampler = AdaptiveTemporalSampler(total_frames=10, fallback_uniform=True)

    # Monkeypatch to force all-zero scores (simulates static video)
    monkeypatch.setattr(
        sampler,
        "compute_motion_scores",
        lambda frame_paths: torch.zeros(len(frame_paths) - 1, dtype=torch.float32),
    )

    result = sampler.sample(paths, seed=0)
    assert len(result) == 10, f"Expected 10 frames, got {len(result)}"
    assert 0 in result, "Frame 0 must always be included"

    # With all-zero importance, the algorithm fills with uniform from
    # low-motion candidates. Max gap between consecutive indices should
    # be bounded.
    max_gap = max(result[i + 1] - result[i] for i in range(len(result) - 1))
    expected_max_gap = math.ceil(40 / 10) + 1  # 5
    assert max_gap <= expected_max_gap, (
        f"Max gap {max_gap} exceeds expected {expected_max_gap} "
        f"for uniform distribution; indices={result}"
    )


def test_shorter_clip_than_budget(tmp_path):
    """When clip has fewer frames than budget, return all frames."""
    paths = make_synthetic_frames(tmp_path, n=5)
    sampler = AdaptiveTemporalSampler(total_frames=10)
    result = sampler.sample(paths, seed=0)

    assert len(result) == 5, f"Expected 5 frames (clip length), got {len(result)}"
    assert result == [
        0,
        1,
        2,
        3,
        4,
    ], f"Must return all frames when budget > clip length, got {result}"
