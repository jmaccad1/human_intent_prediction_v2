import argparse
import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOT_DIR = ROOT / "hiphop_v2" / "annot"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "hiphop_gts"


def bbox_to_xyxy(bbox):
    x1 = int(bbox["left"])
    y1 = int(bbox["top"])
    x2 = x1 + int(bbox["width"])
    y2 = y1 + int(bbox["height"])
    return x1, y1, x2, y2


def bbox_center(bbox):
    x = int(bbox["left"]) + int(bbox["width"]) / 2
    y = int(bbox["top"]) + int(bbox["height"]) / 2
    return x, y


def has_gaze_classification(obj):
    for classification in obj.get("classifications", []):
        if classification.get("value") == "is_gazed":
            return any(
                answer.get("value") == "yes"
                for answer in classification.get("checklist_answers", [])
            )
        if classification:
            return True
    return False


def get_annotations(labelbox_row):
    projects = labelbox_row.get("projects", {})
    if not projects:
        return {}

    project = next(iter(projects.values()))
    labels = project.get("labels", [])
    if not labels:
        return {}

    return labels[0].get("annotations", {})


def load_video_annotations(labelbox_row):
    annotations = get_annotations(labelbox_row)
    frames = annotations.get("frames", {})
    rows = []

    for frame_number in sorted(frames, key=lambda value: int(value)):
        objects = frames[frame_number].get("objects", {}).values()
        frame_name = f"frame_{int(frame_number) - 1:04d}.jpg"
        head_bbox = None
        gaze_point = (-1, -1)

        for obj in objects:
            bbox = obj.get("bounding_box")
            if not bbox:
                continue

            if obj.get("value", "").lower() == "head":
                head_bbox = bbox_to_xyxy(bbox)
            elif has_gaze_classification(obj):
                gaze_point = bbox_center(bbox)

        if head_bbox is None:
            continue

        rows.append((frame_name, *head_bbox, *gaze_point))

    return rows


def write_annotations(rows, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open(mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rows)


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


def process_file(annotation_file, output_dir, output_offset, video_count):
    participant_id = participant_from_filename(annotation_file.name, output_offset)
    participant_name = f"P{participant_id}"
    written = 0

    for video_id in range(1, video_count + 1):
        output_file = (
            output_dir
            / participant_name
            / f"V{video_id}"
            / f"{participant_name}_V{video_id}.txt"
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.touch()

    with annotation_file.open() as file:
        for line in file:
            if not line.strip():
                continue

            labelbox_row = json.loads(line)
            video_id = video_from_external_id(labelbox_row["data_row"]["external_id"])
            rows = load_video_annotations(labelbox_row)
            output_file = (
                output_dir
                / participant_name
                / f"V{video_id}"
                / f"{participant_name}_V{video_id}.txt"
            )
            write_annotations(rows, output_file)
            written += 1

    return participant_name, written


def process_dataset(annotation_dir, output_dir, output_offset, video_count):
    annotation_files = sorted(annotation_dir.glob("P*_dataset2_annot.ndjson"))
    if not annotation_files:
        raise FileNotFoundError(f"No ndjson files found in {annotation_dir}")

    for annotation_file in annotation_files:
        participant_name, written = process_file(
            annotation_file,
            output_dir,
            output_offset,
            video_count,
        )
        print(f"Wrote {written} videos for {participant_name}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate HIP-HOP v2 gaze target files from Labelbox ndjson exports."
    )
    parser.add_argument(
        "--annot-dir",
        type=Path,
        default=DEFAULT_ANNOT_DIR,
        help="Directory containing P##_dataset2_annot.ndjson files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for generated P##/V##/*.txt files.",
    )
    parser.add_argument(
        "--output-offset",
        type=int,
        default=20,
        help="Participant number offset. Default maps P01-P20 to P21-P40.",
    )
    parser.add_argument(
        "--video-count",
        type=int,
        default=50,
        help="Number of V folders to create per participant.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    process_dataset(
        args.annot_dir,
        args.output_dir,
        args.output_offset,
        args.video_count,
    )


if __name__ == "__main__":
    main()
