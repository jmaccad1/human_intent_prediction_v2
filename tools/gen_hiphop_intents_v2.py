import argparse
import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOT_DIR = ROOT / "hiphop_v2" / "annot"
DEFAULT_OUTPUT_CSV = ROOT / "hiphop_gaze" / "annotations" / "hiphop_intents_v2.csv"


def participant_from_filename(filename, output_offset):
    match = re.match(r"P(\d+)_dataset2_annot\.ndjson$", filename)
    if not match:
        raise ValueError(f"Unexpected annotation filename: {filename}")
    return int(match.group(1)) + output_offset


def video_from_external_id(external_id):
    match = re.match(r"V(\d+)\.mp4$", external_id)
    if not match:
        raise ValueError(f"Unexpected video external_id: {external_id}")
    return int(match.group(1))


def get_annotations(labelbox_row):
    projects = labelbox_row.get("projects", {})
    if not projects:
        return {}

    project = next(iter(projects.values()))
    labels = project.get("labels", [])
    if not labels:
        return {}

    return labels[0].get("annotations", {})


def get_intent_value(annotations):
    for classification in annotations.get("classifications", []):
        if classification.get("value") != "intention":
            continue

        radio_answer = classification.get("radio_answer")
        if isinstance(radio_answer, dict):
            return radio_answer.get("value") or radio_answer.get("name") or "None"

        answer = classification.get("answer")
        if isinstance(answer, dict):
            return answer.get("value") or answer.get("title") or "None"

    return "None"


def has_gaze_classification(obj):
    for classification in obj.get("classifications", []):
        if classification.get("value") == "is_gazed":
            return any(
                answer.get("value") == "yes"
                for answer in classification.get("checklist_answers", [])
            )

        answer = classification.get("answer")
        if isinstance(answer, dict):
            answer_value = str(answer.get("value", "")).lower()
            answer_title = str(answer.get("title", "")).lower()
            if answer_value in {"yes", "true", "is_gazed"} or answer_title in {
                "yes",
                "true",
                "isgazed",
            }:
                return True

        if classification:
            return True

    return False


def normalize_gazed_object(value):
    value = str(value or "").strip()
    if not value or value.lower() in {"none", "nan", "null"}:
        return "None"
    return value[:1].upper() + value[1:]


def gazed_object_from_frame(frame):
    for obj in frame.get("objects", {}).values():
        if has_gaze_classification(obj):
            return normalize_gazed_object(obj.get("name") or obj.get("value"))
    return "None"


def frame_count_from_row(labelbox_row, frames):
    frame_count = int(labelbox_row.get("media_attributes", {}).get("frame_count") or 0)
    if frame_count:
        return frame_count
    if frames:
        return max(int(frame_number) for frame_number in frames)
    return 0


def load_video_intent_row(labelbox_row, participant_name, annotation_filename):
    annotations = get_annotations(labelbox_row)
    frames = annotations.get("frames", {})
    frame_count = frame_count_from_row(labelbox_row, frames)
    video_id = video_from_external_id(labelbox_row["data_row"]["external_id"])

    frame_annotations = []
    for frame_number in range(1, frame_count + 1):
        frame = frames.get(str(frame_number))
        if frame is None:
            gazed_object = "None"
        else:
            gazed_object = gazed_object_from_frame(frame)

        frame_annotations.append(
            [f"frame_{frame_number - 1:04d}.jpg", gazed_object]
        )

    return [
        participant_name,
        f"V{video_id}",
        annotation_filename,
        json.dumps(frame_annotations),
        get_intent_value(annotations),
    ]


def process_file(annotation_file, output_offset):
    participant_id = participant_from_filename(annotation_file.name, output_offset)
    participant_name = f"P{participant_id}"
    rows = []

    with annotation_file.open("r", encoding="utf-8") as file:
        for line_num, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                labelbox_row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {annotation_file} at line {line_num}: {exc}"
                ) from exc

            video_id = video_from_external_id(labelbox_row["data_row"]["external_id"])
            rows.append(
                load_video_intent_row(
                    labelbox_row,
                    participant_name,
                    f"{participant_name}_V{video_id}.ndjson",
                )
            )

    return participant_name, rows


def write_csv(rows, output_csv):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["participant", "video", "annotation_file", "frames", "intent"])
        writer.writerows(rows)


def process_dataset(annotation_dir, output_csv, output_offset):
    annotation_files = sorted(annotation_dir.glob("P*_dataset2_annot.ndjson"))
    if not annotation_files:
        raise FileNotFoundError(f"No ndjson files found in {annotation_dir}")

    all_rows = []
    for annotation_file in annotation_files:
        participant_name, rows = process_file(annotation_file, output_offset)
        all_rows.extend(rows)
        print(f"Loaded {len(rows)} videos for {participant_name}")

    all_rows.sort(
        key=lambda row: (
            int(row[0][1:]),
            int(row[1][1:]),
        )
    )
    write_csv(all_rows, output_csv)
    print(f"Wrote {len(all_rows)} video annotations to {output_csv}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate hiphop_intents-style CSV from HIP-HOP v2 Labelbox "
            "ndjson exports."
        )
    )
    parser.add_argument(
        "--annot-dir",
        type=Path,
        default=DEFAULT_ANNOT_DIR,
        help="Directory containing P##_dataset2_annot.ndjson files.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output-offset",
        type=int,
        default=20,
        help="Participant number offset. Default maps P01-P20 to P21-P40.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    process_dataset(args.annot_dir, args.output_csv, args.output_offset)


if __name__ == "__main__":
    main()
