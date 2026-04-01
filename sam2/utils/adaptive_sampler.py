from __future__ import annotations

import random
from typing import List, Optional

import torch
from PIL import Image


class AdaptiveTemporalSampler:
    """
    Motion-adaptive frame sampler for video segmentation training.

    Replaces uniform temporal stride with a budget-proportional
    allocation that favors frames around high-motion transitions.

    Algorithm:
      1. Subsample every SCORE_STRIDE-th frame for motion scoring
      2. Compute L1 pixel diff between consecutive subsampled frames
      3. Interpolate scores to full clip length
      4. Normalize to [0, 1]
      5. Allocate floor(budget_ratio * total_frames) frames to
         positions with score > motion_threshold (weighted random)
      6. Fill remainder with uniform samples from low-motion region
      7. Always include frame index 0
    """

    SCORE_STRIDE: int = 4  # subsample rate for motion scoring

    def __init__(
        self,
        total_frames: int,
        motion_threshold: float = 0.03,
        budget_ratio: float = 0.7,
        fallback_uniform: bool = True,
    ) -> None:
        self.total_frames = total_frames
        self.motion_threshold = motion_threshold
        self.budget_ratio = budget_ratio
        self.fallback_uniform = fallback_uniform

    # ------------------------------------------------------------------
    # Core scoring helpers
    # ------------------------------------------------------------------

    def _load_gray_tensor(self, path: str) -> torch.Tensor:
        """Load image as (1, H, W) float32 tensor in [0, 1]."""
        img = Image.open(path).convert("L")
        import numpy as np

        arr = torch.from_numpy(np.array(img, dtype=np.float32)) / 255.0
        return arr.unsqueeze(0)  # (1, H, W)

    def compute_motion_scores(
        self,
        frame_paths: List[str],
    ) -> torch.Tensor:
        """
        Returns shape (len(frame_paths) - 1,) float32 tensor.
        Loads only every SCORE_STRIDE-th frame then interpolates.
        Uses PIL + torch only. Handles clips shorter than SCORE_STRIDE.
        """
        n = len(frame_paths)
        if n < 2:
            return torch.zeros(max(0, n - 1), dtype=torch.float32)

        stride = min(self.SCORE_STRIDE, n - 1)

        # --- sparse pass: load every stride-th frame --------------------
        sparse_indices = list(range(0, n, stride))
        # ensure the last frame is included for proper endpoint
        if sparse_indices[-1] != n - 1:
            sparse_indices.append(n - 1)

        prev = self._load_gray_tensor(frame_paths[sparse_indices[0]])
        sparse_scores: list[float] = []
        for idx in sparse_indices[1:]:
            curr = self._load_gray_tensor(frame_paths[idx])
            sparse_scores.append((curr - prev).abs().mean().item())
            prev = curr

        sparse_tensor = torch.tensor(sparse_scores, dtype=torch.float32)
        # number of full-res score slots = n - 1
        return self._interpolate_scores(sparse_tensor, target_len=n - 1)

    def _interpolate_scores(
        self,
        sparse_scores: torch.Tensor,
        target_len: int,
    ) -> torch.Tensor:
        """Linear interpolation from sparse to full-resolution scores."""
        if sparse_scores.numel() == 0:
            return torch.zeros(target_len, dtype=torch.float32)

        if sparse_scores.numel() == 1:
            return sparse_scores.expand(target_len).clone()

        # Reshape for F.interpolate: (batch=1, channels=1, length)
        src = sparse_scores.unsqueeze(0).unsqueeze(0)
        dst = torch.nn.functional.interpolate(
            src, size=target_len, mode="linear", align_corners=True
        )
        return dst.squeeze(0).squeeze(0)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        frame_paths: List[str],
        seed: Optional[int] = None,
    ) -> List[int]:
        """
        Returns sorted list of selected frame indices.
        Length == min(self.total_frames, len(frame_paths)).
        Always includes index 0.
        Falls back to uniform if compute_motion_scores raises.
        """
        n_frames = len(frame_paths)
        budget = min(self.total_frames, n_frames)
        if budget <= 0:
            return []

        # deterministic RNG when seed is provided
        rng = random.Random(seed) if seed is not None else random.Random()

        # --- compute scores ---------------------------------------------
        try:
            scores = self.compute_motion_scores(frame_paths)
        except Exception:
            if self.fallback_uniform:
                return self._uniform_fallback(n_frames)
            raise

        # scores has length n_frames - 1; pad to n_frames for index-align
        # score[i] is the transition between frame i and i+1
        # We assign a per-frame importance by max of adjacent transitions:
        #   frame 0:      scores[0]
        #   frame i (mid): max(scores[i-1], scores[i])
        #   frame last:    scores[-1]
        importance = torch.zeros(n_frames, dtype=torch.float32)
        if scores.numel() > 0:
            importance[0] = scores[0]
            importance[-1] = scores[-1]
            for i in range(1, n_frames - 1):
                importance[i] = max(scores[i - 1], scores[i])

        # normalize to [0, 1]
        imp_min, imp_max = importance.min(), importance.max()
        if imp_max > imp_min:
            importance = (importance - imp_min) / (imp_max - imp_min)
        else:
            importance.zero_()

        # --- split budget -----------------------------------------------
        high_motion_budget = max(1, int(self.budget_ratio * budget))
        low_motion_budget = budget - high_motion_budget

        selected: set[int] = set()

        # always include frame 0
        selected.add(0)

        # --- high-motion: weighted random sample ------------------------
        high_mask = importance > self.motion_threshold
        high_indices = high_mask.nonzero(as_tuple=True)[0].tolist()
        # ensure 0 is not double-counted
        high_indices = [i for i in high_indices if i != 0]

        if high_indices:
            weights = [importance[i].item() for i in high_indices]
            k = min(high_motion_budget, len(high_indices))
            picked = _weighted_sample_without_replacement(rng, high_indices, weights, k)
            selected.update(picked)

        # --- low-motion: uniform fill from remaining --------------------
        low_motion_budget = budget - len(selected)
        if low_motion_budget > 0:
            candidates = sorted(set(range(n_frames)) - selected)
            if len(candidates) <= low_motion_budget:
                selected.update(candidates)
            else:
                step = len(candidates) / low_motion_budget
                for j in range(low_motion_budget):
                    selected.add(candidates[int(j * step)])

        return sorted(selected)

    def _uniform_fallback(self, n_frames: int) -> List[int]:
        """Uniform stride fallback, always includes frame 0."""
        budget = min(self.total_frames, n_frames)
        if budget <= 0:
            return []
        if budget == 1:
            return [0]
        step = (n_frames - 1) / (budget - 1)
        indices = sorted(
            {0} | {min(n_frames - 1, round(i * step)) for i in range(budget)}
        )
        # ensure exact length
        while len(indices) < budget:
            indices = sorted(set(indices) | {0})
        return indices[:budget]


# ---------------------------------------------------------------------------
# Utility: weighted sampling without replacement (no numpy needed)
# ---------------------------------------------------------------------------


def _weighted_sample_without_replacement(
    rng: random.Random,
    population: List[int],
    weights: List[float],
    k: int,
) -> List[int]:
    """Reservoir-style weighted sample without replacement."""
    pool = list(population)
    w = list(weights)
    result: list[int] = []
    for _ in range(min(k, len(pool))):
        total = sum(w)
        if total <= 0:
            break
        r = rng.random() * total
        cumul = 0.0
        for i in range(len(pool)):
            cumul += w[i]
            if cumul >= r:
                result.append(pool[i])
                pool.pop(i)
                w.pop(i)
                break
    return result


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import tempfile

    import numpy as np
    from PIL import Image as PILImage

    with tempfile.TemporaryDirectory() as tmp:
        # Create 40 synthetic 32x32 grayscale frames
        paths = []
        for i in range(40):
            arr = np.zeros((32, 32), dtype=np.uint8)
            if 15 <= i <= 25:  # motion burst: moving white block
                x = (i - 15) * 2
                arr[10:20, x : x + 8] = 255
            arr = arr + np.random.randint(0, 10, arr.shape, dtype=np.uint8)
            p = os.path.join(tmp, f"{i:04d}.png")
            PILImage.fromarray(arr).save(p)
            paths.append(p)

        sampler = AdaptiveTemporalSampler(total_frames=12, motion_threshold=0.03)
        indices = sampler.sample(paths, seed=0)

        assert 0 in indices, "Frame 0 must always be included"
        assert indices == sorted(set(indices)), "Must be sorted and unique"
        assert len(indices) == 12, f"Expected 12, got {len(indices)}"

        # Verify motion-dense region [15,25] gets more samples than [0,10]
        burst_count = sum(1 for i in indices if 15 <= i <= 25)
        static_count = sum(1 for i in indices if 0 <= i <= 10)
        assert burst_count > static_count, (
            f"Burst region should have more samples: "
            f"burst={burst_count} static={static_count}"
        )
        print(f"Selected indices: {indices}")
        print(f"Burst frames selected: {burst_count}/11, Static: {static_count}/11")
        print("PASS")
