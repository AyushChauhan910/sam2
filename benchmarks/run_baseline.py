#!/usr/bin/env python3
"""
Baseline benchmark for SAM 2.1 Hiera Large on DAVIS-2017 val set.

Usage:
    python benchmarks/run_baseline.py [--davis_root /path/to/DAVIS]

If DAVIS-2017 is not found, this script will attempt to download it.
Requires: CUDA GPU, torch 2.5.1+ with CUDA 12.1
"""

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Ensure sam2 is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sam2.build_sam import build_sam2_video_predictor
from sav_dataset.utils.sav_benchmark import benchmark as run_jf_benchmark

# â”€â”€ Constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CKPT = REPO_ROOT / "checkpoints" / "sam2.1_hiera_large.pt"
DAVIS_URL = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"
DAVIS_VAL_VIDEOS = [
    "bear", "bmx-bumps", "boat", "breakdance", "camel",
    "car-roundabout", "car-shadow", "cows", "crossroads",
    "dog", "drift-chicane", "drift-straight", "goat",
    "horsejump-high", "kite-surf", "libby", "motocross-bumps",
    "motorbike", "paragliding", "parkour", "scooter-black",
    "soapbox",
]
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
OUTPUT_DIR = BENCHMARKS_DIR / "davis_2017_pred_pngs"


def download_davis(davis_root: Path) -> None:
    """Download and extract DAVIS-2017 TrainVal 480p."""
    import urllib.request
    import ssl

    zip_path = davis_root.parent / "DAVIS-2017-trainval-480p.zip"
    if not zip_path.exists():
        print(f"Downloading DAVIS-2017 to {zip_path} ...")
        ssl._create_default_https_context = ssl._create_unverified_context
        urllib.request.urlretrieve(DAVIS_URL, str(zip_path))
        print("Download complete.")

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(davis_root.parent))
    print("Extraction complete.")


def ensure_davis(davis_root: Path) -> None:
    """Ensure DAVIS-2017 dataset exists, downloading if needed."""
    jpeg_dir = davis_root / "JPEGImages" / "480p"
    annot_dir = davis_root / "Annotations" / "480p"
    val_txt = davis_root / "ImageSets" / "2017" / "val.txt"

    if jpeg_dir.exists() and annot_dir.exists() and val_txt.exists():
        print(f"DAVIS-2017 found at {davis_root}")
        return

    print(f"DAVIS-2017 not found at {davis_root}, attempting download...")
    download_davis(davis_root)

    if not (jpeg_dir.exists() and annot_dir.exists()):
        raise RuntimeError(
            f"DAVIS-2017 still not found after download at {davis_root}. "
            "Please download manually from https://davischallenge.org/davis2017/code.html "
            "and extract to the expected path."
        )


def get_val_videos(davis_root: Path) -> list:
    """Read validation video list from DAVIS ImageSets."""
    val_txt = davis_root / "ImageSets" / "2017" / "val.txt"
    if val_txt.exists():
        with open(val_txt, "r") as f:
            videos = [line.strip() for line in f if line.strip()]
        return videos
    # Fallback to hardcoded list
    return DAVIS_VAL_VIDEOS


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def run_inference_and_measure(predictor, base_video_dir, input_mask_dir,
                               output_mask_dir, video_names):
    """Run VOS inference on all videos, measure FPS and peak VRAM."""

    # Load masks from DAVIS-style annotation directory
    def load_davis_mask(mask_path):
        mask = np.array(Image.open(mask_path)).astype(np.uint8)
        return mask

    def get_per_obj(mask):
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids > 0].tolist()
        return {oid: (mask == oid) for oid in obj_ids}

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    total_frames = 0
    total_time = 0.0

    for i, video_name in enumerate(video_names):
        print(f"  [{i+1}/{len(video_names)}] {video_name}", end="", flush=True)
        video_dir = os.path.join(base_video_dir, video_name)
        frame_names = [
            os.path.splitext(p)[0]
            for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
        ]
        frame_names.sort(key=lambda p: int(p))

        # Init state
        inference_state = predictor.init_state(
            video_path=video_dir, async_loading_frames=False
        )
        height = inference_state["video_height"]
        width = inference_state["video_width"]

        # Load first-frame mask and add objects
        first_mask_path = os.path.join(input_mask_dir, video_name,
                                        f"{frame_names[0]}.png")
        gt_mask = load_davis_mask(first_mask_path)
        per_obj = get_per_obj(gt_mask)

        for obj_id, obj_mask in per_obj.items():
            predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=obj_id,
                mask=obj_mask,
            )

        # Propagate and time it
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state
        ):
            per_obj_output = {
                oid: (out_mask_logits[j] > 0.0).cpu().numpy()
                for j, oid in enumerate(out_obj_ids)
            }
            video_segments[out_frame_idx] = per_obj_output

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        n_frames = len(frame_names)
        total_frames += n_frames
        total_time += elapsed
        fps = n_frames / elapsed if elapsed > 0 else float("inf")
        print(f"  {fps:.1f} FPS ({n_frames} frames)")

        # Save output masks
        from tools.vos_inference import save_masks_to_dir, DAVIS_PALETTE
        out_dir = os.path.join(output_mask_dir, video_name)
        os.makedirs(out_dir, exist_ok=True)
        for frame_idx, per_obj_mask in video_segments.items():
            save_masks_to_dir(
                output_mask_dir=output_mask_dir,
                video_name=video_name,
                frame_name=frame_names[frame_idx],
                per_obj_output_mask=per_obj_mask,
                height=height,
                width=width,
                per_obj_png_file=False,
                output_palette=DAVIS_PALETTE,
            )

        # Reset for next video
        predictor.reset_state(inference_state)

    avg_fps = total_frames / total_time if total_time > 0 else 0
    peak_vram_bytes = torch.cuda.max_memory_allocated()
    peak_vram_gb = peak_vram_bytes / (1024 ** 3)

    return avg_fps, peak_vram_gb


def evaluate_jf(davis_root: Path, output_mask_dir: Path, video_names: list = None) -> float:
    gt_root = str(davis_root / "Annotations" / "480p")
    pred_root = str(output_mask_dir)

    print("\nEvaluating J&F scores...")
    all_global_jf, all_global_j, all_global_f, _ = run_jf_benchmark(
        gt_roots=[gt_root],
        mask_roots=[pred_root],
        strict=False,
        num_processes=4,
        verbose=True,
        skip_first_and_last=True,
    )
    return all_global_jf[0]


def main():
    parser = argparse.ArgumentParser(description="SAM 2.1 Hiera Large baseline benchmark")
    parser.add_argument(
        "--davis_root",
        type=str,
        default=str(REPO_ROOT.parent / "davis2017" / "DAVIS"),
        help="Path to DAVIS-2017 root directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(SAM2_CKPT),
        help="Path to SAM 2.1 checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=SAM2_CFG,
        help="SAM 2 model config",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: benchmarks/baseline.json)",
    )
    parser.add_argument(
        "--adaptive_sampler",
        action="store_true",
        help="Use adaptive temporal sampling for initial frame selection",
    )
    parser.add_argument(
        "--motion_threshold",
        type=float,
        default=0.03,
        help="Motion threshold for adaptive sampler",
    )
    parser.add_argument(
        "--budget_ratio",
        type=float,
        default=0.7,
        help="Budget ratio for adaptive sampler",
    )
    parser.add_argument("--num_sequences", type=int, default=None,
        help="Limit number of sequences for dry runs")
    args = parser.parse_args()

    # Label the run
    run_label = "adaptive" if args.adaptive_sampler else "uniform"

    davis_root = Path(args.davis_root)

    # â”€â”€ Verify CUDA â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU is required for this benchmark.")
        print("The vos_inference pipeline uses @torch.autocast(device_type='cuda').")
        print("No NVIDIA GPU detected. Please run on a machine with a CUDA GPU.")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print()

    # â”€â”€ Ensure DAVIS dataset â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ensure_davis(davis_root)
    video_names = get_val_videos(davis_root)
    if args.num_sequences is not None:
        video_names = video_names[:args.num_sequences]
    print(f"DAVIS-2017 val: {len(video_names)} videos\n")

    # â”€â”€ Build predictor â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"Loading model: {args.config}")
    print(f"Checkpoint:    {args.checkpoint}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()

    predictor = build_sam2_video_predictor(
        config_file=args.config,
        ckpt_path=args.checkpoint,
        device="cuda",
    )

    # â”€â”€ Run inference â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\nRunning VOS inference on DAVIS-2017 val set...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    avg_fps, peak_vram_gb = run_inference_and_measure(
        predictor=predictor,
        base_video_dir=str(davis_root / "JPEGImages" / "480p"),
        input_mask_dir=str(davis_root / "Annotations" / "480p"),
        output_mask_dir=str(OUTPUT_DIR),
        video_names=video_names,
    )

    # â”€â”€ Evaluate J&F â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    jf_mean = evaluate_jf(davis_root, OUTPUT_DIR, video_names)

    # â”€â”€ Report â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    results = {
        "jf_mean": round(jf_mean, 4),
        "fps": round(avg_fps, 2),
        "peak_vram_gb": round(peak_vram_gb, 2),
        "model": "sam2.1_hiera_large",
        "sampling": run_label,
    }

    print("\n" + "=" * 60)
    print(f"  {run_label.upper()} SAMPLING RESULTS")
    print("=" * 60)
    print(f"  J&F-Mean:      {results['jf_mean']:.4f}")
    print(f"  Avg FPS:       {results['fps']:.2f}")
    print(f"  Peak VRAM:     {results['peak_vram_gb']:.2f} GB")
    print(f"  Model:         {results['model']}")
    print(f"  Sampling:      {results['sampling']}")
    print("=" * 60)

    # â”€â”€ Save â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = BENCHMARKS_DIR / "baseline.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
