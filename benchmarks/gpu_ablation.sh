#!/usr/bin/env bash
# ===========================================================================
# benchmarks/gpu_ablation.sh
#
# Final validation script for adaptive temporal sampling.
# Must run on a GPU machine (A100/H100/RTX 3090+) with DAVIS-2017 available.
#
# Usage:
#     cd facebookresearch/sam2
#     bash benchmarks/gpu_ablation.sh
#
# Prerequisites:
#   - CUDA GPU (torch.cuda.is_available() == True)
#   - DAVIS-2017 dataset (auto-downloaded if missing)
#   - SAM 2.1 checkpoint (auto-downloaded if missing)
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DAVIS_ROOT="${1:-$REPO_ROOT/../davis2017/DAVIS}"
CKPT="$REPO_ROOT/checkpoints/sam2.1_hiera_large.pt"
OUTPUT_BASELINE="$REPO_ROOT/benchmarks/baseline_gpu.json"
OUTPUT_ADAPTIVE="$REPO_ROOT/benchmarks/adaptive_gpu.json"

cd "$REPO_ROOT"

echo "======================================================================="
echo " SAM 2.1 Adaptive Sampler â€” GPU Ablation"
echo "======================================================================="
echo ""

# â”€â”€ Step 1: Verify CUDA â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
echo "[1/6] Checking CUDA availability..."
python -c "
import torch
assert torch.cuda.is_available(), 'No CUDA device found. GPU required.'
print(f'  GPU:   {torch.cuda.get_device_name(0)}')
print(f'  CUDA:  {torch.version.cuda}')
print(f'  Torch: {torch.__version__}')
"
echo "  OK"
echo ""

# â”€â”€ Step 2: Download DAVIS-2017 if not present â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
echo "[2/6] Checking DAVIS-2017 dataset..."
if [ -d "$DAVIS_ROOT/JPEGImages/480p" ] && [ -d "$DAVIS_ROOT/Annotations/480p" ]; then
    echo "  Found at $DAVIS_ROOT"
else
    echo "  Not found at $DAVIS_ROOT, downloading..."
    python benchmarks/download_davis.py --output_dir "$(dirname "$DAVIS_ROOT")"
fi
echo ""

# â”€â”€ Step 3: Download SAM 2.1 checkpoint if not present â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
echo "[3/6] Checking SAM 2.1 checkpoint..."
if [ -f "$CKPT" ]; then
    CKPT_SIZE=$(du -h "$CKPT" | cut -f1)
    echo "  Found: $CKPT ($CKPT_SIZE)"
else
    echo "  Not found, downloading..."
    cd checkpoints
    bash download_ckpts.sh
    cd "$REPO_ROOT"
fi
echo ""

# â”€â”€ Step 4: Run baseline (uniform sampling) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
echo "[4/6] Running BASELINE inference (uniform sampling)..."
python benchmarks/run_baseline.py \
    --davis_root "$DAVIS_ROOT" \
    --checkpoint "$CKPT" \
    --config configs/sam2.1/sam2.1_hiera_l.yaml \
    --output "$OUTPUT_BASELINE"
echo ""

# â”€â”€ Step 5: Run adaptive sampling inference â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
echo "[5/6] Running ADAPTIVE inference (motion-adaptive sampling)..."

# For adaptive inference, we need to patch the predictor's initial frame
# selection to use AdaptiveTemporalSampler. Since vos_inference.py uses
# first-frame-mask propagation (not training-time sampling), we run a
# modified inference that uses the adaptive sampler to select the best
# initial conditioning frames from a candidate window.
python benchmarks/run_baseline.py \
    --davis_root "$DAVIS_ROOT" \
    --checkpoint "$CKPT" \
    --config configs/sam2.1/sam2.1_hiera_l.yaml \
    --output "$OUTPUT_ADAPTIVE" \
    --adaptive_sampler \
    --motion_threshold 0.03 \
    --budget_ratio 0.7
echo ""

# â”€â”€ Step 6: Compare results â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
echo "[6/6] Comparing results..."
python benchmarks/compare_results.py \
    "$OUTPUT_BASELINE" \
    "$OUTPUT_ADAPTIVE"
EXIT_CODE=$?

echo ""
echo "======================================================================="
echo " Ablation complete."
echo " Baseline: $OUTPUT_BASELINE"
echo " Adaptive: $OUTPUT_ADAPTIVE"
echo "======================================================================="

exit $EXIT_CODE
