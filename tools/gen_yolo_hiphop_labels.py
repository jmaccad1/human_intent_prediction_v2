import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, Iterator, List, Tuple


DEFAULT_CLASSES = [
    "Bag",
    "Book",
    "Bottle",
    "Bowl",
    "Broom",
    "Chair",
    "Cup",
    "Fruits",
    "Laptop",
    "Pillow",
    "Racket",
    "Rug",
    "Sandwich",
    "Umbrella",
    "Utensils",
    "Head",
]

TRAIN_PARTICIPANTS = {f"P{idx}" for idx in range(21, 37)}
TEST_PARTICIPANTS = {f"P{idx}" for idx in range(37, 41)}

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOT_DIR = ROOT / "hiphop_v2" / "annot"
DEFAULT_OUTPUT_DIR = ROOT / "yolo_hiphop_v2"
DEFAULT_CLEANED_P21_V1 = ROOT / "P21_V1.ndjson"


def normalise_class_name(value: str) -> str:
    value = str(value).replace("_", " ").strip().title().replace(" ", "")
    aliases = {
        "Fruit": "Fruits",
        "Utensil": "Utensils",
    }
    return aliases.get(value, value)


def participant_from_filename(filename: str, output_offset: int) -> str:
    match = re.match(r"P(\d+)_dataset2_annot\.ndjson$", filename)
    if not match:
        raise ValueError(f"Unexpected annotation filename: {filename}")
    return f"P{int(match.group(1)) + output_offset}"


def video_from_external_id(external_id: str) -> str:
    match = re.match(r"V(\d+)\.mp4$", external_id)
    if not match:
        raise ValueError(f"Unexpected video external_id: {external_id}")
    return f"V{int(match.group(1))}"


def participant_sort_key(participant: str) -> int:
    return int(participant[1:])


def video_sort_key(video: str) -> int:
    return int(video[1:])


def get_annotations(labelbox_row: dict) -> dict:
    projects = labelbox_row.get("projects", {})
    if not projects:
        return {}

    project = next(iter(projects.values()))
    labels = project.get("labels", [])
    if not labels:
        return {}

    return labels[0].get("annotations", {})


def load_labelbox_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def bbox_to_yolo(
    bbox: dict,
    image_width: int,
    image_height: int,
) -> Tuple[float, float, float, float]:
    left = float(bbox["left"])
    top = float(bbox["top"])
    width = float(bbox["width"])
    height = float(bbox["height"])

    x_center = (left + width / 2.0) / image_width
    y_center = (top + height / 2.0) / image_height
    norm_width = width / image_width
    norm_height = height / image_height

    return (
        min(max(x_center, 0.0), 1.0),
        min(max(y_center, 0.0), 1.0),
        min(max(norm_width, 0.0), 1.0),
        min(max(norm_height, 0.0), 1.0),
    )


def yolo_label_lines(
    frame: dict,
    class_to_id: Dict[str, int],
    image_width: int,
    image_height: int,
) -> List[str]:
    lines = []
    objects = frame.get("objects", {})
    if isinstance(objects, dict):
        objects_iter = objects.values()
    else:
        objects_iter = objects

    for obj in objects_iter:
        bbox = obj.get("bounding_box") or obj.get("bbox")
        if not bbox:
            continue

        class_name = normalise_class_name(obj.get("name") or obj.get("value") or "")
        if class_name not in class_to_id:
            continue

        x_center, y_center, width, height = bbox_to_yolo(
            bbox,
            image_width,
            image_height,
        )
        lines.append(
            f"{class_to_id[class_name]} "
            f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    return lines


def frame_count_from_row(labelbox_row: dict, frames: dict) -> int:
    frame_count = int(labelbox_row.get("media_attributes", {}).get("frame_count") or 0)
    if frame_count:
        return frame_count
    if frames:
        return max(int(frame_number) for frame_number in frames)
    return 0


def image_size_from_row(labelbox_row: dict) -> Tuple[int, int]:
    media = labelbox_row.get("media_attributes", {})
    width = int(media.get("width") or 0)
    height = int(media.get("height") or 0)
    if width <= 0 or height <= 0:
        raise ValueError("Labelbox row is missing media width/height.")
    return width, height


def split_for_participant(participant: str) -> str:
    if participant in TRAIN_PARTICIPANTS:
        return "train"
    if participant in TEST_PARTICIPANTS:
        return "test"
    raise ValueError(f"Participant {participant} is outside P21-P40.")


def iter_v2_rows(
    annotation_dir: Path,
    cleaned_p21_v1: Path,
    output_offset: int,
) -> Iterator[Tuple[str, str, dict, str]]:
    annotation_files = sorted(annotation_dir.glob("P*_dataset2_annot.ndjson"))
    if not annotation_files:
        raise FileNotFoundError(f"No ndjson files found in {annotation_dir}")

    for annotation_file in annotation_files:
        participant = participant_from_filename(annotation_file.name, output_offset)
        for labelbox_row in load_labelbox_rows(annotation_file):
            video = video_from_external_id(labelbox_row["data_row"]["external_id"])
            if participant == "P21" and video == "V1":
                continue
            yield participant, video, labelbox_row, str(annotation_file)

    if cleaned_p21_v1.is_file():
        rows = load_labelbox_rows(cleaned_p21_v1)
        if rows:
            yield "P21", "V1", rows[0], str(cleaned_p21_v1)
    else:
        print(f"Warning: cleaned P21/V1 annotation not found: {cleaned_p21_v1}")


def write_video_labels(
    labelbox_row: dict,
    output_dir: Path,
    split: str,
    participant: str,
    video: str,
    class_to_id: Dict[str, int],
) -> int:
    annotations = get_annotations(labelbox_row)
    frames = annotations.get("frames", {})
    frame_count = frame_count_from_row(labelbox_row, frames)
    image_width, image_height = image_size_from_row(labelbox_row)
    video_label_dir = output_dir / "labels" / split / participant / video
    video_label_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for frame_number in range(1, frame_count + 1):
        frame_name = f"frame_{frame_number - 1:04d}.txt"
        frame = frames.get(str(frame_number), {})
        lines = yolo_label_lines(frame, class_to_id, image_width, image_height)
        (video_label_dir / frame_name).write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )
        written += 1

    return written


def write_metadata(output_dir: Path, class_names: List[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "classes.txt").write_text(
        "\n".join(class_names) + "\n",
        encoding="utf-8",
    )
    yaml_lines = [
        f"path: {output_dir.as_posix()}",
        "train: labels/train",
        "val: labels/test",
        "test: labels/test",
        f"nc: {len(class_names)}",
        "names:",
    ]
    yaml_lines.extend(f"  {idx}: {name}" for idx, name in enumerate(class_names))
    (output_dir / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")


def convert(args: argparse.Namespace) -> None:
    annotation_dir = args.annot_dir.resolve()
    cleaned_p21_v1 = args.cleaned_p21_v1.resolve()
    output_dir = args.output_dir.resolve()

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)

    class_names = [normalise_class_name(name) for name in args.classes]
    class_to_id = {class_name: idx for idx, class_name in enumerate(class_names)}
    stats = {
        "train": {"videos": 0, "labels": 0},
        "test": {"videos": 0, "labels": 0},
    }

    rows = sorted(
        iter_v2_rows(annotation_dir, cleaned_p21_v1, args.output_offset),
        key=lambda item: (participant_sort_key(item[0]), video_sort_key(item[1])),
    )

    for participant, video, labelbox_row, source in rows:
        split = split_for_participant(participant)
        label_count = write_video_labels(
            labelbox_row,
            output_dir,
            split,
            participant,
            video,
            class_to_id,
        )
        stats[split]["videos"] += 1
        stats[split]["labels"] += label_count
        print(f"{split}: wrote {label_count} labels for {participant}/{video} from {source}")

    write_metadata(output_dir, class_names)

    print(f"Read annotations from: {annotation_dir}")
    print(f"Read cleaned P21/V1 from: {cleaned_p21_v1}")
    for split, split_stats in stats.items():
        print(
            f"{split}: {split_stats['videos']} videos, "
            f"{split_stats['labels']} label files"
        )
    print(f"Wrote metadata: {output_dir / 'classes.txt'}")
    print(f"Wrote metadata: {output_dir / 'data.yaml'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate YOLO head/object label files from HIP-HOP v2 Labelbox "
            "annotations with P21-P36 as train and P37-P40 as test."
        )
    )
    parser.add_argument(
        "--annot-dir",
        type=Path,
        default=DEFAULT_ANNOT_DIR,
        help="Directory containing P##_dataset2_annot.ndjson files.",
    )
    parser.add_argument(
        "--cleaned-p21-v1",
        type=Path,
        default=DEFAULT_CLEANED_P21_V1,
        help="Cleaned P21/V1 Labelbox ndjson to use instead of P01 V1.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for generated YOLO labels and metadata.",
    )
    parser.add_argument(
        "--output-offset",
        type=int,
        default=20,
        help="Participant number offset. Default maps P01-P20 to P21-P40.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES,
        help="YOLO class list in class-id order.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the output directory before writing labels.",
    )
    return parser.parse_args()


def main() -> None:
    convert(parse_args())


if __name__ == "__main__":
    main()
