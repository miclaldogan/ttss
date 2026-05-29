"""TTSS: unzip and organise UCF-Crime videos after Dropbox download.

Usage::

    python -m ttss.scripts.unzip_dataset \\
        --zip  data/raw/UCF-Crime/ucf_crime_videos.zip \\
        --dest data/raw/UCF-Crime/videos
"""
from __future__ import annotations
import argparse
import zipfile
from pathlib import Path


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--zip",  default="data/raw/UCF-Crime/ucf_crime_videos.zip")
    p.add_argument("--dest", default="data/raw/UCF-Crime/videos")
    return p


def main():
    args = build_parser().parse_args()
    zip_path = Path(args.zip)
    dest = Path(args.dest)

    if not zip_path.exists():
        print(f"Zip not found: {zip_path}")
        print("Is the download still running?  Check: tail -f data/raw/UCF-Crime/download.log")
        return

    size_gb = zip_path.stat().st_size / 1e9
    print(f"Zip: {zip_path}  ({size_gb:.1f} GB)")
    print(f"Extracting to: {dest}")
    dest.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        print(f"Files in zip: {len(members)}")
        for i, member in enumerate(members):
            zf.extract(member, dest)
            if (i + 1) % 100 == 0 or i == len(members) - 1:
                print(f"  {i+1}/{len(members)}  {member}")

    # Count extracted videos
    exts = {".mp4", ".avi", ".mov", ".mkv"}
    videos = [p for p in dest.rglob("*") if p.suffix.lower() in exts]
    print(f"\nExtracted {len(videos)} video files to {dest}")
    print("Next step: python -m ttss.scripts.extract_features --device cuda")


if __name__ == "__main__":
    main()
