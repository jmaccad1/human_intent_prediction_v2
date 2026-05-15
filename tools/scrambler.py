import argparse
import csv
import os
import random
import string
from pathlib import Path


VIDEO_EXTENSIONS = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
    ".wmv",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Rename video files in a directory and write/read a CSV filename mapping."
        )
    )
    parser.add_argument(
        "--video-dir",
        required=True,
        help="Directory containing videos to rename.",
    )
    parser.add_argument(
        "--csv-file",
        required=True,
        help=(
            "CSV mapping path. In generation mode, this file is written. "
            "In --from-csv mode, this file is read."
        ),
    )
    parser.add_argument(
        "--from-csv",
        action="store_true",
        help="Read existing old_filename,new_filename pairs from --csv-file.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Include videos in subdirectories. CSV names are relative paths.",
    )
    parser.add_argument(
        "--prefix",
        default="video",
        help="Prefix for generated filenames when not using --from-csv.",
    )
    parser.add_argument(
        "--token-length",
        type=int,
        default=10,
        help="Random token length for generated filenames.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for repeatable generated names.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rename files. Without this, only prints a dry-run preview.",
    )
    return parser.parse_args()


def stays_inside(base_dir, path):
    return os.path.commonpath([str(base_dir), str(path.resolve())]) == str(base_dir)


def is_video(path):
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def list_videos(video_dir, recursive):
    pattern = "**/*" if recursive else "*"
    return sorted(path for path in video_dir.glob(pattern) if is_video(path))


def random_token(length):
    alphabet = string.ascii_lowercase + string.digits
    return "".join(random.choice(alphabet) for _ in range(length))


def read_mapping(csv_file):
    with csv_file.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    if rows and {"old_filename", "new_filename"}.issubset(rows[0].keys()):
        return [
            (row["old_filename"].strip(), row["new_filename"].strip())
            for row in rows
            if row.get("old_filename") and row.get("new_filename")
        ]

    with csv_file.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        return [
            (row[0].strip(), row[1].strip())
            for row in reader
            if len(row) >= 2 and row[0].strip() and row[1].strip()
        ]


def write_mapping(csv_file, mapping):
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    with csv_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["old_filename", "new_filename"])
        writer.writerows(mapping)


def generate_mapping(video_dir, videos, prefix, token_length):
    used_names = {path.name for path in videos}
    generated = set()
    mapping = []

    for video in videos:
        while True:
            new_name = f"{prefix}_{random_token(token_length)}{video.suffix.lower()}"
            if new_name not in used_names and new_name not in generated:
                break
        generated.add(new_name)
        old_rel = video.relative_to(video_dir)
        new_rel = old_rel.with_name(new_name)
        mapping.append((str(old_rel), str(new_rel)))

    return mapping


def validate_mapping(video_dir, mapping):
    new_paths = []
    errors = []
    source_paths = {
        str((video_dir / old_name).resolve()).lower() for old_name, _new_name in mapping
    }

    for old_name, new_name in mapping:
        old_path = video_dir / old_name
        new_path = video_dir / new_name
        new_paths.append(new_path)

        if not stays_inside(video_dir, old_path):
            errors.append(f"Source path escapes video directory: {old_name}")
        if not stays_inside(video_dir, new_path):
            errors.append(f"Target path escapes video directory: {new_name}")
        if not old_path.exists():
            errors.append(f"Missing source file: {old_name}")
        if old_path.resolve() == new_path.resolve():
            errors.append(f"Source and target are identical: {old_name}")
        if new_path.exists() and str(new_path.resolve()).lower() not in source_paths:
            errors.append(f"Target already exists: {new_name}")

    normalized_targets = [str(path.resolve()).lower() for path in new_paths]
    if len(normalized_targets) != len(set(normalized_targets)):
        errors.append("Duplicate target filenames found in mapping.")

    return errors


def rename_files(video_dir, mapping, apply_changes):
    planned = [
        (video_dir / old_name, video_dir / new_name) for old_name, new_name in mapping
    ]

    print(f"{'APPLY' if apply_changes else 'DRY RUN'}: {len(planned)} rename(s)")
    for old_path, new_path in planned:
        print(f"{old_path} -> {new_path}")

    if not apply_changes:
        return

    temp_pairs = []
    for index, (old_path, new_path) in enumerate(planned):
        while True:
            temp_path = old_path.with_name(
                f".scrambler_tmp_{index}_{random_token(6)}_{old_path.name}"
            )
            if not temp_path.exists():
                break
        old_path.rename(temp_path)
        temp_pairs.append((temp_path, new_path))

    for temp_path, new_path in temp_pairs:
        new_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.rename(new_path)


def main():
    args = parse_args()
    if args.token_length < 1:
        raise SystemExit("--token-length must be at least 1")

    if args.seed is not None:
        random.seed(args.seed)

    video_dir = Path(args.video_dir).resolve()
    csv_file = Path(args.csv_file).resolve()

    if not video_dir.is_dir():
        raise SystemExit(f"Video directory does not exist: {video_dir}")

    if args.from_csv:
        if not csv_file.is_file():
            raise SystemExit(f"CSV mapping file does not exist: {csv_file}")
        mapping = read_mapping(csv_file)
    else:
        videos = list_videos(video_dir, args.recursive)
        if not videos:
            raise SystemExit(f"No video files found in: {video_dir}")
        mapping = generate_mapping(video_dir, videos, args.prefix, args.token_length)
        write_mapping(csv_file, mapping)
        print(f"Wrote mapping CSV: {csv_file}")

    if not mapping:
        raise SystemExit("No rename mappings found.")

    errors = validate_mapping(video_dir, mapping)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    rename_files(video_dir, mapping, args.apply)


if __name__ == "__main__":
    main()
