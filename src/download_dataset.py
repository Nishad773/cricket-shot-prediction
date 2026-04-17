"""Download and trim labeled cricket shot clips from YouTube."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import tempfile
from pathlib import Path

from yt_dlp import YoutubeDL


DATA_DIR = Path("data")
LABELS = {"cover_drive", "pull_shot", "sweep"}


def ensure_dataset_folders(data_dir: str | Path = DATA_DIR) -> None:
    """Create the expected dataset folder structure."""

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for label in sorted(LABELS):
        (data_dir / label).mkdir(parents=True, exist_ok=True)


def download_video(url: str, download_dir: str | Path) -> Path:
    """Download a YouTube video and return the local file path."""

    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(download_dir / "%(id)s.%(ext)s")

    options = {
        "format": "mp4/bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "outtmpl": output_template,
        "quiet": False,
        "noplaylist": True,
    }

    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded_path = Path(ydl.prepare_filename(info))

    if downloaded_path.suffix.lower() != ".mp4":
        downloaded_path = downloaded_path.with_suffix(".mp4")

    if not downloaded_path.exists():
        raise FileNotFoundError(f"Downloaded file not found for URL: {url}")

    return downloaded_path


def trim_clip(source_path: str | Path, output_path: str | Path, start_time: float, end_time: float) -> Path:
    """Trim a clip using ffmpeg."""

    if end_time <= start_time:
        raise ValueError("end_time must be greater than start_time.")

    clip_duration = end_time - start_time
    if clip_duration < 3 or clip_duration > 8:
        raise ValueError("Clip duration must be between 3 and 8 seconds.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg is required for trimming clips. Install ffmpeg and add it to PATH.")

    command = [
        ffmpeg_path,
        "-y",
        "-ss",
        str(start_time),
        "-i",
        str(source_path),
        "-t",
        str(clip_duration),
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        str(output_path),
    ]
    subprocess.run(command, check=True, capture_output=True)
    return output_path


def download_and_trim_clip(
    url: str,
    label: str,
    start_time: float,
    end_time: float,
    clip_name: str,
    data_dir: str | Path = DATA_DIR,
) -> Path:
    """Download a YouTube video and save a trimmed labeled clip."""

    if label not in LABELS:
        raise ValueError(f"Unsupported label '{label}'. Expected one of: {sorted(LABELS)}")

    ensure_dataset_folders(data_dir)
    data_dir = Path(data_dir)

    with tempfile.TemporaryDirectory() as temp_dir:
        source_path = download_video(url, temp_dir)
        output_path = data_dir / label / f"{clip_name}.mp4"
        return trim_clip(source_path, output_path, start_time=start_time, end_time=end_time)


def download_from_manifest(manifest_path: str | Path, data_dir: str | Path = DATA_DIR) -> list[Path]:
    """Download and trim all clips listed in a CSV manifest."""

    manifest_path = Path(manifest_path)
    outputs: list[Path] = []

    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"url", "label", "start_time", "end_time", "clip_name"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Manifest is missing columns: {sorted(missing_columns)}")

        for row in reader:
            clip_path = download_and_trim_clip(
                url=row["url"],
                label=row["label"],
                start_time=float(row["start_time"]),
                end_time=float(row["end_time"]),
                clip_name=row["clip_name"],
                data_dir=data_dir,
            )
            outputs.append(clip_path)

    return outputs


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Download labeled cricket shot clips from YouTube.")
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to a CSV file with columns: url,label,start_time,end_time,clip_name",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DATA_DIR),
        help="Root dataset directory. Defaults to ./data",
    )
    return parser.parse_args()


def main() -> None:
    """Download all clips listed in the manifest."""

    args = parse_args()
    downloaded_clips = download_from_manifest(args.manifest, data_dir=args.data_dir)
    print(f"Saved {len(downloaded_clips)} clips.")
    for clip_path in downloaded_clips:
        print(clip_path)


if __name__ == "__main__":
    main()
