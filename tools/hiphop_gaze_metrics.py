import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


DEFAULT_CSV = Path("hiphop_gaze") / "annotations" / "hiphop_intents.csv"
DEFAULT_OUTPUT_CSV = Path("output") / "hiphop_gaze_object_counts_by_participant.csv"
GAZED_OBJECT_ORDER = [
    "Pillow",
    "Rug",
    "Sandwich",
    "Racket",
    "Broom",
    "Utensils",
    "Umbrella",
    "Bowl",
    "Fruits",
    "Bag",
    "Bottle",
    "Cup",
    "Laptop",
    "Book",
    "Chair",
    "None",
]


def parse_participant(value):
    match = re.fullmatch(r"[Pp]?(\d+)", value.strip())
    if not match:
        raise argparse.ArgumentTypeError(
            f"Invalid participant {value!r}. Use forms like P1 or 1."
        )
    return int(match.group(1))


def parse_participant_range(value):
    match = re.fullmatch(r"\s*[Pp]?(\d+)\s*-\s*[Pp]?(\d+)\s*", value)
    if not match:
        raise argparse.ArgumentTypeError(
            f"Invalid range {value!r}. Use forms like P1-P40 or 1-40."
        )

    start = int(match.group(1))
    end = int(match.group(2))
    if start > end:
        raise argparse.ArgumentTypeError(
            f"Invalid range {value!r}. Start participant must be <= end participant."
        )
    return start, end


def normalize_gazed_item(value):
    item = str(value).strip()
    if not item or item.lower() in {"none", "nan", "null"}:
        return "None"
    return item


def sort_gazed_items(items):
    order = {item: index for index, item in enumerate(GAZED_OBJECT_ORDER)}
    return sorted(items, key=lambda item: (order.get(item, len(order)), item))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute gazed-item frame metrics from hiphop_gaze/annotations/"
            "hiphop_intents.csv for a participant range."
        )
    )
    parser.add_argument(
        "participant_range",
        nargs="?",
        type=parse_participant_range,
        help="Participant range, e.g. P1-P40 or P21-P40.",
    )
    parser.add_argument(
        "--start",
        type=parse_participant,
        help="Start participant, e.g. P1. Alternative to participant_range.",
    )
    parser.add_argument(
        "--end",
        type=parse_participant,
        help="End participant, e.g. P40. Alternative to participant_range.",
    )
    parser.add_argument(
        "--csv-file",
        default=str(DEFAULT_CSV),
        help=f"Path to hiphop_intents.csv. Default: {DEFAULT_CSV}",
    )
    parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_OUTPUT_CSV),
        help=f"Path to write the participant counts CSV. Default: {DEFAULT_OUTPUT_CSV}",
    )
    return parser.parse_args()


def resolve_range(args):
    if args.participant_range is not None:
        if args.start is not None or args.end is not None:
            raise SystemExit("Use either participant_range or --start/--end, not both.")
        return args.participant_range

    if args.start is None or args.end is None:
        raise SystemExit("Provide a range like P1-P40, or both --start and --end.")
    if args.start > args.end:
        raise SystemExit("--start must be <= --end.")
    return args.start, args.end


def load_video_counts(csv_file, start_participant, end_participant):
    video_counts = []
    all_items = set()
    participants_found = set()

    with csv_file.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"participant", "video", "frames"}
        missing = required_columns.difference(reader.fieldnames or [])
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"CSV is missing required column(s): {missing_text}")

        for row_num, row in enumerate(reader, start=2):
            participant_text = row.get("participant", "")
            participant_num = parse_participant(participant_text)
            if not start_participant <= participant_num <= end_participant:
                continue
            participants_found.add(participant_num)

            frames_json = row.get("frames", "").strip()
            if not frames_json:
                frames = []
            else:
                try:
                    frames = json.loads(frames_json)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in 'frames' column at row {row_num}: {exc}"
                    ) from exc

            counts = Counter()
            for frame_annotation in frames:
                if len(frame_annotation) < 2:
                    item = "None"
                else:
                    item = normalize_gazed_item(frame_annotation[1])
                counts[item] += 1
                all_items.add(item)

            video_counts.append(
                {
                    "participant": f"P{participant_num}",
                    "video": row.get("video", "").strip(),
                    "counts": counts,
                    "frames": sum(counts.values()),
                }
            )

    return (
        video_counts,
        sort_gazed_items(all_items),
        participants_found,
    )


def compute_participant_counts(video_counts, all_items):
    participant_counts = {}

    for video in video_counts:
        participant = video["participant"]
        if participant not in participant_counts:
            participant_counts[participant] = Counter()
        participant_counts[participant].update(video["counts"])

    rows = []
    for participant in sorted(
        participant_counts,
        key=lambda value: int(value[1:]) if value.startswith("P") else value,
    ):
        counts = participant_counts[participant]
        row = {
            "participant": participant,
            "total_frames": sum(counts.values()),
        }
        for item in all_items:
            row[item] = counts[item]
        rows.append(row)

    return rows


def write_participant_csv(path, rows, all_items):
    fieldnames = ["participant", "total_frames", *all_items]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_participant_table(rows, all_items):
    headers = ["participant", "total_frames", *all_items]
    widths = {header: len(header) for header in headers}
    formatted_rows = []

    for row in rows:
        formatted = {header: str(row.get(header, 0)) for header in headers}
        formatted_rows.append(formatted)
        for header, value in formatted.items():
            widths[header] = max(widths[header], len(value))

    header_line = "  ".join(header.ljust(widths[header]) for header in headers)
    separator = "  ".join("-" * widths[header] for header in headers)
    print(header_line)
    print(separator)
    for row in formatted_rows:
        print("  ".join(row[header].ljust(widths[header]) for header in headers))


def main():
    args = parse_args()
    start_participant, end_participant = resolve_range(args)
    csv_file = Path(args.csv_file)

    if not csv_file.is_file():
        raise SystemExit(f"CSV file does not exist: {csv_file}")

    video_counts, all_items, participants_found = load_video_counts(
        csv_file, start_participant, end_participant
    )
    if not video_counts:
        raise SystemExit(
            f"No videos found for P{start_participant}-P{end_participant} in {csv_file}"
        )

    rows = compute_participant_counts(video_counts, all_items)
    total_frames = sum(video["frames"] for video in video_counts)

    print(f"CSV file: {csv_file}")
    print(f"Participant range: P{start_participant}-P{end_participant}")
    print(f"Videos included: {len(video_counts)}")
    print(f"Total frame annotations: {total_frames}")
    missing_participants = [
        participant
        for participant in range(start_participant, end_participant + 1)
        if participant not in participants_found
    ]
    if missing_participants:
        missing_text = ", ".join(f"P{participant}" for participant in missing_participants)
        print(f"Missing participants in CSV: {missing_text}")
    print("Output mode: one row per participant, object-gazed totals as columns")
    print("")
    print_participant_table(rows, all_items)

    output_path = Path(args.output_csv)
    write_participant_csv(output_path, rows, all_items)
    print("")
    print(f"Wrote CSV: {output_path}")


if __name__ == "__main__":
    main()
