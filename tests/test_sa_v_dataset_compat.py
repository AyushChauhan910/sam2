"""
Backward compatibility tests for adaptive sampler integration.

Verifies that RandomUniformSampler works unchanged when sampler_cfg
is omitted, set to uniform, or set to adaptive.

Run with:
    pytest tests/test_sa_v_dataset_compat.py -v
"""

from unittest.mock import MagicMock, patch

import pytest

from training.dataset.vos_sampler import RandomUniformSampler, VOSSampler


# ---------------------------------------------------------------------------
# RandomUniformSampler backward compat
# ---------------------------------------------------------------------------


def test_default_no_sampler_cfg():
    """No sampler_cfg â†’ _adaptive_sampler must be None."""
    sampler = RandomUniformSampler(num_frames=8, max_num_objects=3)
    assert sampler._adaptive_sampler is None, (
        "Default init (no sampler_cfg) must not create an adaptive sampler"
    )
    assert sampler.sampler_cfg == {}, (
        "Default sampler_cfg must be empty dict"
    )


def test_sampler_cfg_type_uniform():
    """sampler_cfg with type='uniform' â†’ _adaptive_sampler must be None."""
    sampler = RandomUniformSampler(
        num_frames=8,
        max_num_objects=3,
        sampler_cfg={"type": "uniform"},
    )
    assert sampler._adaptive_sampler is None, (
        "sampler_cfg type='uniform' must not create an adaptive sampler"
    )


def test_sampler_cfg_type_adaptive():
    """sampler_cfg with type='adaptive' â†’ _adaptive_sampler must be set."""
    from sam2.utils.adaptive_sampler import AdaptiveTemporalSampler

    sampler = RandomUniformSampler(
        num_frames=8,
        max_num_objects=3,
        sampler_cfg={
            "type": "adaptive",
            "total_frames": 8,
            "motion_threshold": 0.03,
            "budget_ratio": 0.7,
        },
    )
    assert sampler._adaptive_sampler is not None, (
        "sampler_cfg type='adaptive' must create an adaptive sampler"
    )
    assert isinstance(sampler._adaptive_sampler, AdaptiveTemporalSampler), (
        f"Expected AdaptiveTemporalSampler, got {type(sampler._adaptive_sampler)}"
    )


def test_sampler_cfg_adaptive_defaults():
    """Adaptive sampler inherits total_frames from num_frames if not set."""
    from sam2.utils.adaptive_sampler import AdaptiveTemporalSampler

    sampler = RandomUniformSampler(
        num_frames=16,
        max_num_objects=5,
        sampler_cfg={"type": "adaptive"},
    )
    assert isinstance(sampler._adaptive_sampler, AdaptiveTemporalSampler)
    assert sampler._adaptive_sampler.total_frames == 16, (
        f"Expected total_frames=16 (from num_frames), "
        f"got {sampler._adaptive_sampler.total_frames}"
    )


def test_sampler_cfg_adaptive_bad_import_falls_back():
    """If adaptive import fails, _adaptive_sampler stays None with a warning."""
    # Patch the import to raise
    with patch.dict("sys.modules", {"sam2.utils.adaptive_sampler": None}):
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sampler = RandomUniformSampler(
                num_frames=8,
                max_num_objects=3,
                sampler_cfg={"type": "adaptive"},
            )
            assert sampler._adaptive_sampler is None, (
                "Failed import must fall back to None"
            )
            assert len(w) >= 1, "Should have emitted a warning"
            assert "AdaptiveTemporalSampler init failed" in str(w[0].message)


# ---------------------------------------------------------------------------
# VOSDataset integration (mocked)
# ---------------------------------------------------------------------------


def test_vos_dataset_accepts_sampler_with_cfg():
    """VOSDataset stores the sampler without error."""
    from training.dataset.vos_dataset import VOSDataset

    mock_video_ds = MagicMock()
    mock_video_ds.__len__ = MagicMock(return_value=10)

    sampler = RandomUniformSampler(
        num_frames=8,
        max_num_objects=3,
        sampler_cfg={"type": "adaptive", "total_frames": 8},
    )

    ds = VOSDataset(
        transforms=None,
        training=True,
        video_dataset=mock_video_ds,
        sampler=sampler,
        multiplier=2,
    )

    assert ds.sampler is sampler, "VOSDataset must store the sampler reference"
    assert ds.sampler._adaptive_sampler is not None, (
        "Adaptive sampler must survive VOSDataset wrapping"
    )


def test_vos_dataset_default_sampler():
    """VOSDataset works with default sampler (no adaptive)."""
    from training.dataset.vos_dataset import VOSDataset

    mock_video_ds = MagicMock()
    mock_video_ds.__len__ = MagicMock(return_value=10)

    sampler = RandomUniformSampler(num_frames=8, max_num_objects=3)

    ds = VOSDataset(
        transforms=None,
        training=True,
        video_dataset=mock_video_ds,
        sampler=sampler,
        multiplier=2,
    )

    assert ds.sampler._adaptive_sampler is None, (
        "Default sampler must have no adaptive component"
    )
