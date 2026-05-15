import argparse
import csv
import json
from collections import Counter
from pathlib import Path


DEFAULT_CSV = Path("hiphop_gaze") / "annotations" / "hiphop_intents.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count intent labels from hiphop_intents.csv."
    )
    parser.add_argument(
        "--csv-file",
        default=str(DEFAULT_CSV),
        help="Path to hiphop_intents.csv.",
    )
    return parser.parse_args()


def count_intents(csv_file):
    counts = Counter()

    with csv_file.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if "intent" not in reader.fieldnames:
            raise ValueError(f"CSV is missing required 'intent' column: {csv_file}")

        for row in reader:
            intent = row.get("intent", "").strip()
            counts[intent or "None"] += 1

    return counts


def count_gazed_objects(csv_file):
    counts = Counter()

    with csv_file.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if "frames" not in reader.fieldnames:
            raise ValueError(f"CSV is missing required 'frames' column: {csv_file}")

        for row_num, row in enumerate(reader, start=2):
            frames_json = row.get("frames", "").strip()
            if not frames_json:
                continue

            try:
                frames = json.loads(frames_json)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in 'frames' column at row {row_num}: {exc}"
                ) from exc

            for frame_annotation in frames:
                if len(frame_annotation) < 2:
                    continue
                gazed_object = str(frame_annotation[1]).strip()
                counts[gazed_object or "None"] += 1

    return counts


def main():
    args = parse_args()
    csv_file = Path(args.csv_file)

    if not csv_file.is_file():
        raise SystemExit(f"CSV file does not exist: {csv_file}")

    counts = count_intents(csv_file)
    total = sum(counts.values())
    object_counts = count_gazed_objects(csv_file)
    total_gaze_frames = sum(object_counts.values())

    print(f"File: {csv_file}")
    print(f"Total videos: {total}")
    print("Intent counts:")
    for intent, count in sorted(counts.items()):
        print(f"{intent}: {count}")

    print("")
    print(f"Total gaze frame annotations: {total_gaze_frames}")
    print("Gazed object counts:")
    for gazed_object, count in sorted(object_counts.items()):
        print(f"{gazed_object}: {count}")


if __name__ == "__main__":
    main()
