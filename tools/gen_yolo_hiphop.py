import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple, Union

import cv2


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

TRAIN_PARTICIPANTS = {f"P{idx}" for idx in range(1, 17)}
TEST_PARTICIPANTS = {f"P{idx}" for idx in range(17, 21)}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _normalise_class_name(value: str) -> str:
    value = str(value).replace("_", " ").strip().title().replace(" ", "")
    aliases = {
        "Fruit": "Fruits",
        "Utensil": "Utensils",
    }
    return aliases.get(value, value)


def load_ndjson(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def resolve_repo_path(path: Union[str, Path]) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = _repo_root() / path
    return path.resolve()


def participant_number(name: str) -> int:
    if not name.startswith("P"):
        return 0
    try:
        return int(name[1:])
    except ValueError:
        return 0


def video_number(name: str) -> int:
    if not name.startswith("V"):
        return 0
    try:
        return int(name[1:])
    except ValueError:
        return 0


def iter_hiphop_videos(source_root: Path) -> Iterator[Tuple[str, str, Path]]:
    participants = sorted(
        (path for path in source_root.iterdir() if path.is_dir()),
        key=lambda path: participant_number(path.name),
    )
    for participant_dir in participants:
        videos = sorted(
            (path for path in participant_dir.iterdir() if path.is_dir()),
            key=lambda path: video_number(path.name),
        )
        for video_dir in videos:
            ndjson_path = video_dir / f"{participant_dir.name}_{video_dir.name}.ndjson"
            if ndjson_path.exists():
                yield participant_dir.name, video_dir.name, ndjson_path


def get_image_size(image_dir: Path) -> Tuple[int, int]:
    frame_paths = sorted(image_dir.glob("frame_*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"No frame_*.jpg files found in {image_dir}")

    image = cv2.imread(str(frame_paths[0]))
    if image is None:
        raise RuntimeError(f"Could not read image: {frame_paths[0]}")

    height, width = image.shape[:2]
    return width, height


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
    frame_annotation: dict,
    class_to_id: Dict[str, int],
    image_width: int,
    image_height: int,
) -> List[str]:
    lines = []
    for obj in frame_annotation.get("objects", []):
        if "bbox" not in obj:
            continue

        class_name = _normalise_class_name(obj.get("title") or obj.get("value") or "")
        if class_name not in class_to_id:
            continue

        x_center, y_center, width, height = bbox_to_yolo(
            obj["bbox"], image_width, image_height
        )
        lines.append(
            f"{class_to_id[class_name]} "
            f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )
    return lines


def write_video_split(
    annotations: Iterable[dict],
    source_image_dir: Path,
    output_image_dir: Path,
    labels_dir: Path,
    class_to_id: Dict[str, int],
    image_width: int,
    image_height: int,
    image_size: int,
    relative_video_dir: Path,
) -> Tuple[int, int]:
    output_image_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    image_count = 0
    label_count = 0

    for fallback_idx, frame_annotation in enumerate(annotations):
        frame_number = int(frame_annotation.get("frameNumber", fallback_idx + 1))
        frame_index = frame_number - 1
        source_path = source_image_dir / f"frame_{frame_index:04d}.jpg"
        if not source_path.exists():
            continue

        image = cv2.imread(str(source_path))
        if image is None:
            continue

        resized = cv2.resize(
            image,
            (image_size, image_size),
            interpolation=cv2.INTER_AREA,
        )
        target_image_path = output_image_dir / relative_video_dir / source_path.name
        target_label_path = labels_dir / relative_video_dir / source_path.with_suffix(
            ".txt"
        ).name

        target_image_path.parent.mkdir(parents=True, exist_ok=True)
        target_label_path.parent.mkdir(parents=True, exist_ok=True)

        lines = yolo_label_lines(frame_annotation, class_to_id, image_width, image_height)
        cv2.imwrite(str(target_image_path), resized)
        target_label_path.write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )

        image_count += 1
        label_count += 1

    return image_count, label_count


def write_metadata(
    out_dir: Path,
    class_names: List[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "classes.txt").write_text(
        "\n".join(class_names) + "\n", encoding="utf-8"
    )
    yaml_lines = [
        f"path: {out_dir.as_posix()}",
        "train: images/train",
        "val: images/test",
        "test: images/test",
        f"nc: {len(class_names)}",
        "names:",
    ]
    yaml_lines.extend(f"  {idx}: {name}" for idx, name in enumerate(class_names))
    (out_dir / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")


def convert(args: argparse.Namespace) -> None:
    source_root = resolve_repo_path(args.source_root)
    frames_root = resolve_repo_path(args.frames_root)
    out_dir = resolve_repo_path(args.output_dir)

    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)

    labels_root = out_dir / "labels"
    images_root = out_dir / "images"
    class_names = [_normalise_class_name(name) for name in args.classes]
    class_to_id = {class_name: idx for idx, class_name in enumerate(class_names)}

    split_stats = {
        "train": {"videos": 0, "images": 0, "labels": 0},
        "test": {"videos": 0, "images": 0, "labels": 0},
    }
    skipped = []

    for participant, video, ndjson_path in iter_hiphop_videos(source_root):
        if participant in TRAIN_PARTICIPANTS:
            split = "train"
        elif participant in TEST_PARTICIPANTS:
            split = "test"
        else:
            continue

        image_dir = frames_root / participant / video
        if not image_dir.exists():
            skipped.append(f"{participant}/{video}: missing frames at {image_dir}")
            continue

        annotations = load_ndjson(ndjson_path)
        try:
            image_width, image_height = get_image_size(image_dir)
        except (FileNotFoundError, RuntimeError) as exc:
            skipped.append(f"{participant}/{video}: {exc}")
            continue

        relative_video_dir = Path(participant) / video
        image_count, label_count = write_video_split(
            annotations,
            image_dir,
            images_root / split,
            labels_root / split,
            class_to_id,
            image_width,
            image_height,
            args.image_size,
            relative_video_dir,
        )
        split_stats[split]["videos"] += 1
        split_stats[split]["images"] += image_count
        split_stats[split]["labels"] += label_count

    write_metadata(out_dir, class_names)

    print(f"Read annotations from: {source_root}")
    print(f"Read source images from: {frames_root}")
    for split, stats in split_stats.items():
        print(
            f"{split}: {stats['videos']} videos, "
            f"{stats['images']} images, {stats['labels']} labels"
        )
    if skipped:
        print(f"Skipped {len(skipped)} videos:")
        for item in skipped:
            print(f"  {item}")
    print(f"Wrote metadata: {out_dir / 'classes.txt'}")
    print(f"Wrote metadata: {out_dir / 'data.yaml'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert hiphop_v1 LabelBox frame annotations to YOLO dataset format "
            "with P1-P16 as train and P17-P20 as test."
        )
    )
    parser.add_argument(
        "--source_root",
        default="hiphop_v1",
        help="Root directory containing P*/V* annotation folders.",
    )
    parser.add_argument(
        "--frames_root",
        default=os.path.join("hiphop_gaze", "images"),
        help="Root directory containing extracted frame folders matching P*/V*.",
    )
    parser.add_argument(
        "--output_dir",
        default="yolo_hiphop",
        help="Output directory for resized YOLO images, labels, and metadata.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=640,
        help="Square output image size for generated YOLO images.",
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
        help="Delete the output directory before writing the generated dataset.",
    )
    return parser.parse_args()


def main() -> None:
    convert(parse_args())


if __name__ == "__main__":
    main()
