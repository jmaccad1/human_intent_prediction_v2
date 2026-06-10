import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "P21_V1.ndjson"
DEFAULT_OUTPUT_CSV = ROOT / "hiphop_gaze" / "annotations" / "hiphop_intents_v2_p21_v1.csv"
DEFAULT_PARTICIPANT = "P21"
DEFAULT_VIDEO = "V1"


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


def read_single_labelbox_row(input_file):
    with input_file.open("r", encoding="utf-8") as file:
        for line_num, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {input_file} at line {line_num}: {exc}"
                ) from exc

    raise ValueError(f"Input file is empty: {input_file}")


def build_csv_row(labelbox_row, participant, video):
    annotations = get_annotations(labelbox_row)
    frames = annotations.get("frames", {})
    frame_count = frame_count_from_row(labelbox_row, frames)

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
        participant,
        video,
        f"{participant}_{video}.ndjson",
        json.dumps(frame_annotations),
        get_intent_value(annotations),
    ]


def write_csv(row, output_csv):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["participant", "video", "annotation_file", "frames", "intent"])
        writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a hiphop_intents-style CSV for cleaned P21_V1.ndjson."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input cleaned Labelbox ndjson. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output CSV path. Default: {DEFAULT_OUTPUT_CSV}",
    )
    parser.add_argument(
        "--participant",
        default=DEFAULT_PARTICIPANT,
        help=f"Participant id to write. Default: {DEFAULT_PARTICIPANT}",
    )
    parser.add_argument(
        "--video",
        default=DEFAULT_VIDEO,
        help=f"Video id to write. Default: {DEFAULT_VIDEO}",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input file does not exist: {args.input}")

    labelbox_row = read_single_labelbox_row(args.input)
    row = build_csv_row(labelbox_row, args.participant, args.video)
    write_csv(row, args.output_csv)
    print(f"Wrote 1 video annotation to {args.output_csv}")


if __name__ == "__main__":
    main()
