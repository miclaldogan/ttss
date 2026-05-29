"""TTSS: unzip UCF-Crime nested zip structure.

The Dropbox download is a zip-of-zips:
    ucf_crime_videos.zip
        Anomaly-Videos-Part-1.zip  (crime videos, parts 1-4)
        Anomaly-Videos-Part-2.zip
        Anomaly-Videos-Part-3.zip
        Anomaly-Videos-Part-4.zip
        Testing_Normal_Videos.zip
        Training-Normal-Videos-Part-1.zip
        Training-Normal-Videos-Part-2.zip
        Normal_Videos_for_Event_Recognition.zip

Each inner zip is extracted to data/raw/UCF-Crime/videos/ and deleted
immediately to conserve disk space (only ~89 GB free, 96 GB outer zip).

Usage::

    python -m ttss.scripts.unzip_dataset
    python -m ttss.scripts.unzip_dataset --zip data/raw/UCF-Crime/ucf_crime_videos.zip
"""
from __future__ import annotations
import argparse
import os
import time
import zipfile
from pathlib import Path


INNER_VIDEO_ZIPS = [
    "Anomaly-Videos-Part-1.zip",
    "Anomaly-Videos-Part-2.zip",
    "Anomaly-Videos-Part-3.zip",
    "Anomaly-Videos-Part-4.zip",
    "Testing_Normal_Videos.zip",
    "Training-Normal-Videos-Part-1.zip",
    "Training-Normal-Videos-Part-2.zip",
    "Normal_Videos_for_Event_Recognition.zip",
]

ANNOTATION_FILES = [
    "Temporal_Anomaly_Annotation_for_Testing_Videos.txt",
    "Anomaly_Train.txt",
    "UCF_Crimes-Train-Test-Split.zip",
    "ReadMe-Anomaly-Detection.txt",
]


def _free_gb(path: Path) -> float:
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize / 1e9


def build_parser():
    p = argparse.ArgumentParser(description="Unzip UCF-Crime nested structure")
    p.add_argument("--zip",  default="data/raw/UCF-Crime/ucf_crime_videos.zip")
    p.add_argument("--dest", default="data/raw/UCF-Crime/videos")
    p.add_argument("--ann-dest", default="data/raw/UCF-Crime/annotations")
    return p


def main():
    args = build_parser().parse_args()
    outer_zip = Path(args.zip)
    dest = Path(args.dest)
    ann_dest = Path(args.ann_dest)
    dest.mkdir(parents=True, exist_ok=True)
    ann_dest.mkdir(parents=True, exist_ok=True)

    if not outer_zip.exists():
        print(f"Zip not found: {outer_zip}")
        return

    size_gb = outer_zip.stat().st_size / 1e9
    print(f"Outer zip: {outer_zip} ({size_gb:.1f} GB)")
    print(f"Free disk:  {_free_gb(outer_zip):.1f} GB\n")

    outer = zipfile.ZipFile(outer_zip, "r")

    # 1. Extract annotation files first (tiny, always safe)
    print("Extracting annotation files...")
    for ann_file in ANNOTATION_FILES:
        if ann_file in outer.namelist():
            outer.extract(ann_file, ann_dest)
            print(f"  ✓ {ann_file}")

    # 2. Extract each video zip, unpack its videos, delete the zip
    total_videos = 0
    for inner_name in INNER_VIDEO_ZIPS:
        if inner_name not in outer.namelist():
            print(f"  Skipping {inner_name} (not in outer zip)")
            continue

        inner_info = outer.getinfo(inner_name)
        inner_size_gb = inner_info.file_size / 1e9
        free = _free_gb(outer_zip)
        print(f"\n[{inner_name}]  compressed={inner_size_gb:.1f} GB  free={free:.1f} GB")

        if free < inner_size_gb + 2:
            print(f"  WARNING: only {free:.1f} GB free, need ~{inner_size_gb+2:.1f} GB — skipping")
            continue

        # Extract inner zip from outer
        tmp_inner = dest.parent / inner_name
        print(f"  Extracting inner zip → {tmp_inner} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        outer.extract(inner_name, dest.parent)
        print(f"done ({time.perf_counter()-t0:.0f}s)")

        # Unpack videos from inner zip
        print(f"  Unpacking videos → {dest} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        n = 0
        with zipfile.ZipFile(tmp_inner, "r") as inner_z:
            members = [m for m in inner_z.namelist()
                       if m.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
            for member in members:
                inner_z.extract(member, dest)
                n += 1
            # Also extract any subdirectory structure
            if n == 0:
                inner_z.extractall(dest)
                n = len(inner_z.namelist())
        elapsed = time.perf_counter() - t0
        total_videos += n
        print(f"done — {n} files in {elapsed:.0f}s")

        # Delete inner zip to free space
        tmp_inner.unlink()
        print(f"  Deleted {inner_name}  (free now: {_free_gb(outer_zip):.1f} GB)")

    outer.close()

    # Count extracted videos
    exts = {".mp4", ".avi", ".mov", ".mkv"}
    all_videos = [p for p in dest.rglob("*") if p.suffix.lower() in exts]
    print(f"\nDone — {len(all_videos)} video files extracted to {dest}")
    print(f"Free disk: {_free_gb(dest):.1f} GB")
    print("\nNext: python -m ttss.scripts.extract_features --device cuda")


if __name__ == "__main__":
    main()
