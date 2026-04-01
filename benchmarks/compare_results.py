#!/usr/bin/env python3
"""
Compare two benchmark JSON files and print a formatted table.

Usage:
    python benchmarks/compare_results.py <baseline.json> <adaptive.json>

Exits with code 1 if adaptive J&F drops more than 1.0 below baseline.
"""

import json
import sys


def load(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    return data


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <baseline.json> <adaptive.json>")
        sys.exit(2)

    base = load(sys.argv[1])
    adap = load(sys.argv[2])

    base_jf = base.get("jf_mean", 0.0)
    adap_jf = adap.get("jf_mean", 0.0)
    base_fps = base.get("fps", 0.0)
    adap_fps = adap.get("fps", 0.0)
    base_vram = base.get("peak_vram_gb")
    adap_vram = adap.get("peak_vram_gb")

    delta_jf = adap_jf - base_jf
    delta_jf_str = f"{delta_jf:+.2f}" if delta_jf != 0.0 else "baseline"

    def vram_str(v):
        return f"{v:.1f} GB" if v is not None else "N/A"

    print()
    print(f"  {'Config':<12}| {'J&F-Mean':>9}| {'Delta':>10}| {'FPS':>7}| {'VRAM':>10}")
    print(f"  {'-'*12}-+-{'-'*9}-+-{'-'*10}-+-{'-'*7}-+-{'-'*10}")
    print(f"  {'Uniform':<12}| {base_jf:>9.2f}| {'baseline':>10}| {base_fps:>7.2f}| {vram_str(base_vram):>10}")
    print(f"  {'Adaptive':<12}| {adap_jf:>9.2f}| {delta_jf_str:>10}| {adap_fps:>7.2f}| {vram_str(adap_vram):>10}")
    print()

    if delta_jf < -1.0:
        print(f"  REGRESSION: Adaptive J&F ({adap_jf:.2f}) is more than 1.0 below baseline ({base_jf:.2f})")
        sys.exit(1)
    else:
        print(f"  OK: Adaptive J&F ({adap_jf:.2f}) within tolerance of baseline ({base_jf:.2f})")
        sys.exit(0)


if __name__ == "__main__":
    main()
