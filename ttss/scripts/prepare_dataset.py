"""Temporal Threat Scoring System (TTSS): UCF-Crime preparation CLI.

This script downloads or ingests a local UCF-Crime archive, extracts the raw
videos, and materializes TTSS temporal annotations as JSON for training.
"""

from __future__ import annotations

import argparse
import csv
import json
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from ttss.data.temporal_labeler import TemporalThreatLabeler
from ttss.data.ucf_crime import load_annotation_records


VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv"}


def build_parser() -> argparse.ArgumentParser:
    """Build the dataset preparation CLI parser."""
    parser = argparse.ArgumentParser(description="Prepare TTSS dataset metadata.")
    parser.add_argument("--data-root", required=True, help="Dataset root directory")
    parser.add_argument(
        "--annotation-file",
        required=True,
        help="Source CSV or JSON annotations describing anomaly spans",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where prepared annotations will be written",
    )
    parser.add_argument(
        "--download-url",
        default=None,
        help="Optional direct archive URL for UCF-Crime or a mirror",
    )
    parser.add_argument(
        "--archive-path",
        default=None,
        help="Optional local dataset archive path instead of download",
    )
    parser.add_argument(
        "--extract-dir",
        default=None,
        help="Optional explicit extraction directory under data-root",
    )
    parser.add_argument(
        "--pre-window",
        type=int,
        default=90,
        help="Frames before the anomaly used for pre-crime labeling",
    )
    parser.add_argument(
        "--post-window",
        type=int,
        default=90,
        help="Frames after the anomaly used for post-crime labeling",
    )
    parser.add_argument(
        "--write-csv-index",
        action="store_true",
        help="Also write a compact CSV manifest beside the JSON annotations",
    )
    return parser


def download_archive(download_url: str, destination: Path) -> Path:
    """Download a dataset archive to the requested destination path."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(download_url, destination)
    return destination


def extract_archive(archive_path: Path, extract_dir: Path) -> Path:
    """Extract zip or tar archives into the requested directory."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(extract_dir)
        return extract_dir

    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as archive:
            archive.extractall(extract_dir)
        return extract_dir

    raise ValueError(f"Unsupported archive format: {archive_path}")


def discover_video_map(data_root: Path) -> dict[str, str]:
    """Index video files by stem for annotation-to-file resolution."""
    mapping: dict[str, str] = {}
    for path in data_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            mapping.setdefault(path.stem, str(path.relative_to(data_root)))
    return mapping


def write_csv_index(records: list[dict[str, object]], output_path: Path) -> None:
    """Write a compact CSV index for downstream dataset loading."""
    fieldnames = [
        "video_id",
        "label",
        "split",
        "video_path",
        "fps",
        "total_frames",
        "crime_start_frame",
        "crime_end_frame",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            segments = record["segments"]
            crime = segments["crime"]
            writer.writerow(
                {
                    "video_id": record["video_id"],
                    "label": record["label"],
                    "split": record.get("split", "train"),
                    "video_path": record["video_path"],
                    "fps": record["fps"],
                    "total_frames": record["total_frames"],
                    "crime_start_frame": crime["start_frame"],
                    "crime_end_frame": crime["end_frame"],
                }
            )


def prepare_annotations(
    annotation_file: Path,
    data_root: Path,
    labeler: TemporalThreatLabeler,
) -> list[dict[str, object]]:
    """Convert source metadata into prepared TTSS annotation JSON objects."""
    records = load_annotation_records(annotation_file)
    video_map = discover_video_map(data_root)
    prepared: list[dict[str, object]] = []
    for record in records:
        resolved_video_path = record.video_path
        if not Path(resolved_video_path).is_absolute():
            resolved_video_path = video_map.get(record.video_id, record.video_path)

        total_frames = record.total_frames or max(
            record.crime_end_frame + labeler.post_window + 1,
            1,
        )
        payload = labeler.build_annotation_payload(
            video_id=record.video_id,
            total_frames=total_frames,
            crime_start_frame=record.crime_start_frame,
            crime_end_frame=record.crime_end_frame,
            label=record.label,
            fps=record.fps,
        )
        payload["split"] = record.split
        payload["video_path"] = resolved_video_path
        payload["anomaly_spans"] = [
            {
                "start_frame": span.start_frame,
                "end_frame": span.end_frame,
            }
            for span in record.anomaly_spans
        ]
        prepared.append(payload)
    return prepared


def main() -> None:
    """Run the TTSS dataset preparation scaffold."""
    args = build_parser().parse_args()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_root = data_root
    archive_path: Path | None = None

    # TODO: Wire in the official UCF-Crime credentialed endpoint if your lab has access.
    if args.download_url:
        archive_name = Path(args.download_url).name or "ucf_crime_archive.zip"
        archive_path = download_archive(args.download_url, output_dir / archive_name)
    elif args.archive_path:
        archive_path = Path(args.archive_path).resolve()

    if archive_path is not None:
        extract_dir = Path(args.extract_dir).resolve() if args.extract_dir else data_root / "raw"
        raw_root = extract_archive(archive_path, extract_dir)
    elif not raw_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {raw_root}")

    labeler = TemporalThreatLabeler(
        pre_window=args.pre_window,
        post_window=args.post_window,
    )
    prepared_records = prepare_annotations(
        annotation_file=Path(args.annotation_file).resolve(),
        data_root=raw_root,
        labeler=labeler,
    )

    prepared_json = output_dir / "ttss_annotations.json"
    with prepared_json.open("w", encoding="utf-8") as handle:
        json.dump({"videos": prepared_records}, handle, indent=2)

    if args.write_csv_index:
        write_csv_index(prepared_records, output_dir / "ttss_annotations.csv")

    print(f"prepared_videos={len(prepared_records)}")
    print(f"annotation_json={prepared_json}")
    print(f"raw_root={raw_root}")


if __name__ == "__main__":
    main()
