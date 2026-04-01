#!/usr/bin/env python3
"""
Download DAVIS-2017 TrainVal 480p with retry support.

Usage:
    python benchmarks/download_davis.py [--output_dir /path/to/download]
"""

import argparse
import os
import sys
import time
import urllib.request
import ssl
import zipfile
from pathlib import Path

DAVIS_URL = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"
MAX_RETRIES = 10
RETRY_DELAY = 10  # seconds


def download_with_resume(url: str, output_path: Path) -> None:
    """Download a file with resume support and retries."""
    ssl_ctx = ssl._create_unverified_context()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Check existing size for resume
            resume_from = 0
            if output_path.exists():
                resume_from = output_path.stat().st_size
                print(f"  Resuming from byte {resume_from} ({resume_from / 1e6:.1f} MB)")

            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            if resume_from > 0:
                req.add_header("Range", f"bytes={resume_from}-")

            resp = urllib.request.urlopen(req, timeout=120, context=ssl_ctx)
            total = int(resp.headers.get("Content-Length", 0))
            if resume_from > 0 and resp.status == 206:
                total += resume_from
            elif resp.status == 206:
                pass  # partial content
            else:
                # Server doesn't support resume, start over
                resume_from = 0

            mode = "ab" if resume_from > 0 and resp.status == 206 else "wb"
            downloaded = resume_from

            with open(str(output_path), mode) as f:
                while True:
                    chunk = resp.read(1 << 16)  # 64KB
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded * 100 / total
                        print(f"\r  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.1f}%)",
                              end="", flush=True)

            print(f"\n  Download complete: {output_path.stat().st_size / 1e6:.1f} MB")
            return

        except Exception as e:
            print(f"\n  Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                print(f"  Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print("  Max retries reached. Please download manually:")
                print(f"    {url}")
                sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default=str(Path(__file__).parent.parent.parent / "davis2017"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "DAVIS-2017-trainval-480p.zip"

    # Check if already extracted
    davis_root = output_dir / "DAVIS"
    if (davis_root / "JPEGImages" / "480p").exists():
        print(f"DAVIS-2017 already exists at {davis_root}")
        return

    # Download
    if not zip_path.exists() or zip_path.stat().st_size < 50_000_000:
        print(f"Downloading DAVIS-2017 TrainVal 480p...")
        download_with_resume(DAVIS_URL, zip_path)
    else:
        print(f"Zip already exists: {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")

    # Extract
    print(f"Extracting {zip_path}...")
    try:
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(output_dir))
        print("Extraction complete.")
    except zipfile.BadZipFile:
        print("ERROR: Corrupt zip file. Delete it and re-run this script.")
        sys.exit(1)


if __name__ == "__main__":
    main()
