#!/usr/bin/env python3
"""
Synthetic motion analysis â€” proof-of-concept for adaptive frame sampling.

Runs entirely on CPU. No GPU, no external datasets, no OpenCV required.

Usage:
    python tools/motion_analysis.py
"""

import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"


# ---------------------------------------------------------------------------
# 1. Synthetic clip generator
# ---------------------------------------------------------------------------


def generate_synthetic_clip(
    num_frames: int,
    motion_burst_start: int,
    motion_burst_end: int,
    noise_level: float = 0.02,
    seed: int = 0,
) -> List[Image.Image]:
    """
    Generate a list of grayscale PIL Images (64x64) simulating a video clip.

    Outside [motion_burst_start, motion_burst_end] the scene is static
    (pure Gaussian noise).  Inside the burst window a 10x10 white rectangle
    moves 3 px/frame to the right, plus noise.

    Deterministic for identical (num_frames, burst, noise_level, seed).
    """
    rng = np.random.RandomState(seed)
    H, W = 64, 64
    rect_size = 10
    speed = 3  # pixels per frame

    frames = []
    for t in range(num_frames):
        # Base: low-amplitude Gaussian noise
        canvas = rng.normal(0.0, noise_level, (H, W)).astype(np.float32)

        if motion_burst_start <= t < motion_burst_end:
            # Offset within burst determines rectangle x-position
            offset = t - motion_burst_start
            x = 20 + offset * speed  # start at x=20, move right
            y = 27  # vertically centred
            # Clamp so the rectangle stays on-canvas
            x = min(x, W - rect_size)
            y = min(y, H - rect_size)
            canvas[y : y + rect_size, x : x + rect_size] = 1.0

        # Clip to [0, 1] then scale to uint8
        canvas = np.clip(canvas, 0.0, 1.0)
        img = Image.fromarray((canvas * 255).astype(np.uint8), mode="L")
        frames.append(img)

    return frames


# ---------------------------------------------------------------------------
# 2. Motion scoring (torch-only, no OpenCV)
# ---------------------------------------------------------------------------


def compute_motion_scores(frames: List[Image.Image]) -> List[float]:
    """
    Compute mean absolute pixel difference between each consecutive pair.

    Returns a list of length (num_frames - 1).
    All computation uses torch tensors on CPU.
    """
    tensors = []
    for img in frames:
        arr = np.array(img, dtype=np.float32) / 255.0  # (H, W), float32
        tensors.append(torch.from_numpy(arr))

    scores = []
    for i in range(len(tensors) - 1):
        diff = (tensors[i + 1] - tensors[i]).abs().mean().item()
        scores.append(diff)

    return scores


# ---------------------------------------------------------------------------
# 3. Main â€” generate clips, validate, save summary
# ---------------------------------------------------------------------------


def main() -> bool:
    """
    Returns True if all assertions pass (PASS), False otherwise (FAIL).
    """
    # Five test clips with varying burst positions
    clips = [
        {"num_frames": 64, "burst": [10, 20], "seed": 0},
        {"num_frames": 64, "burst": [20, 35], "seed": 1},
        {"num_frames": 64, "burst": [0, 15], "seed": 2},  # burst at start
        {"num_frames": 64, "burst": [50, 64], "seed": 3},  # burst at end
        {"num_frames": 64, "burst": [30, 45], "seed": 4},
    ]

    summary = {}
    all_pass = True

    for idx, cfg in enumerate(clips):
        tag = f"clip_{idx}"
        burst_start, burst_end = cfg["burst"]
        num_frames = cfg["num_frames"]

        frames = generate_synthetic_clip(
            num_frames=num_frames,
            motion_burst_start=burst_start,
            motion_burst_end=burst_end,
            noise_level=0.02,
            seed=cfg["seed"],
        )

        scores = compute_motion_scores(frames)
        peak_idx = int(np.argmax(scores))

        # The score at index i corresponds to the diff between frame i and i+1.
        # A burst starting at frame S means:
        #   - score[S-1] (transition from static â†’ burst) should spike
        #   - scores inside [S, E-2] should be elevated (moving rect)
        #   - score[E-1] (transition from burst â†’ static) should spike
        #
        # We assert that the peak falls within or at the edge of the burst window:
        #   peak_idx in [burst_start - 1, burst_end - 1]
        #   (bounded by [0, num_frames - 2])
        lo = max(0, burst_start - 1)
        hi = min(num_frames - 2, burst_end - 1)
        peak_ok = lo <= peak_idx <= hi

        status = "OK" if peak_ok else "FAIL"
        print(
            f"[{tag}] burst=[{burst_start},{burst_end})  "
            f"peak_idx={peak_idx}  (valid [{lo},{hi}])  {status}"
        )
        print(f"       scores = [{', '.join(f'{s:.4f}' for s in scores)}]")

        if not peak_ok:
            all_pass = False

        summary[tag] = {
            "scores": [round(s, 6) for s in scores],
            "burst": [burst_start, burst_end],
            "peak_idx": peak_idx,
            "peak_in_burst": peak_ok,
        }

    # Save summary
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BENCHMARKS_DIR / "motion_analysis_synthetic.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {out_path}")

    # Final verdict
    if all_pass:
        print("\nRESULT: PASS")
    else:
        print("\nRESULT: FAIL")

    return all_pass


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
