import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

'''

python visuals\visualize_godet.py `
  --input hiphop_gaze\images\P17\V1 `
  --yolo_weights yolov8n\runs\yolo_hiphop_full_train10\weights\best.pt `
  --vitgaze_config configs\hiphop_dataset_full_6gb.py `
  --vitgaze_weights output\hiphop_dataset_full_6gb\model_final.pth `
  --output_dir visuals\godet_outputs `
  --max_frames 20

python visuals\visualize_godet.py `
  --hiphop_gaze_dir hiphop_gaze `
  --split test `
  --participants v1 `
  --max_videos 1 `
  --max_frames 20

'''


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.data_utils import get_head_box_channel  # noqa: E402
from gazed_object_det import (  # noqa: E402
    DEFAULT_GAZE_CONFIG,
    DEFAULT_GAZE_WEIGHTS,
    DEFAULT_YOLO_WEIGHTS,
    HEAD_CLASS_NAMES,
    HIPHOP_OBJECT_CLASS_NAMES,
    Detection,
    aggregate_gazed_object,
    argmax_pts,
    dark_inference,
    filter_by_confidence,
    filter_by_names,
    image_transform,
    input_size_from_config,
    load_hiphop_gaze_model,
    non_head_objects,
    repo_path,
    require_file,
    yolo_detections,
)


TRAIN_PARTICIPANTS = set(range(1, 17)) | set(range(21, 37))
TEST_PARTICIPANTS = set(range(17, 21)) | set(range(37, 41))
V1_PARTICIPANTS = set(range(1, 21))
V2_PARTICIPANTS = set(range(21, 41))
MERGED_PARTICIPANTS = set(range(1, 41))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def participant_num(participant: str) -> int:
    return int(participant.strip().upper().lstrip("P"))


def natural_key(path: Path) -> Tuple[int, str]:
    digits = "".join(char for char in path.stem if char.isdigit())
    return (int(digits) if digits else -1, path.name)


def selected_participants(split: str, participants: Optional[str]) -> set[int]:
    if split == "all":
        split_participants = MERGED_PARTICIPANTS
    elif split == "train":
        split_participants = TRAIN_PARTICIPANTS
    elif split == "test":
        split_participants = TEST_PARTICIPANTS
    else:
        raise ValueError("--split must be one of train, test, all.")

    if participants is None or participants == "merged":
        version_participants = MERGED_PARTICIPANTS
    elif participants == "v1":
        version_participants = V1_PARTICIPANTS
    elif participants == "v2":
        version_participants = V2_PARTICIPANTS
    else:
        raise ValueError("--participants must be one of v1, v2, merged.")

    return split_participants & version_participants


def iter_video_dirs(
    hiphop_gaze_dir: Path,
    split: str,
    participants: Optional[str],
) -> Iterable[Tuple[str, str, Path]]:
    images_dir = hiphop_gaze_dir / "images"
    require_file(images_dir, "HIPHOP gaze images directory")

    allowed = selected_participants(split, participants)
    for participant_dir in sorted(images_dir.iterdir(), key=lambda path: participant_num(path.name)):
        if not participant_dir.is_dir() or not participant_dir.name.upper().startswith("P"):
            continue
        if participant_num(participant_dir.name) not in allowed:
            continue

        for video_dir in sorted(participant_dir.iterdir(), key=natural_key):
            if video_dir.is_dir() and video_dir.name.upper().startswith("V"):
                yield participant_dir.name, video_dir.name, video_dir


def collect_input_images(input_path: Path, recursive: bool) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    iterator = input_path.rglob("*") if recursive else input_path.iterdir()
    images = sorted(
        (path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=natural_key,
    )
    if not images:
        raise ValueError(f"No images found under {input_path}")
    return images


def iter_frame_paths(video_dir: Path, max_frames: Optional[int], skip_frame: int) -> List[Path]:
    frames = [
        path
        for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    frames = sorted(frames, key=natural_key)[::skip_frame]
    if max_frames is not None:
        frames = frames[:max_frames]
    return frames


def chunks(items: Sequence[Path], chunk_size: int) -> Iterable[Sequence[Path]]:
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def choose_head(heads: Sequence[Detection]) -> Optional[Detection]:
    if not heads:
        return None
    return max(heads, key=lambda detection: detection.confidence)


def result_image_to_pil(result, fallback_path: Path) -> Image.Image:
    orig_img = getattr(result, "orig_img", None)
    if orig_img is not None:
        return Image.fromarray(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))

    image = cv2.imread(str(fallback_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {fallback_path}")
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def gaze_batch_predict(
    gaze_model: torch.nn.Module,
    images: Sequence[Image.Image],
    head_boxes: Sequence[Tuple[float, float, float, float]],
    transform,
    input_size: int,
    device: str,
    use_dark_inference: bool,
) -> List[Tuple[Tuple[float, float], Tuple[float, float], float]]:
    image_tensors = []
    head_channels = []
    sizes = []

    for image, head_box in zip(images, head_boxes):
        width, height = image.size
        sizes.append((width, height))
        image_tensors.append(transform(image))
        head_channels.append(
            get_head_box_channel(
                *head_box,
                width=width,
                height=height,
                resolution=input_size,
                coordconv=False,
            )
        )

    batch = {
        "images": torch.stack(image_tensors).to(device),
        "head_channels": torch.stack(head_channels).unsqueeze(1).to(device),
    }

    with torch.inference_mode():
        heatmaps, gaze_inouts = gaze_model(batch)

    predictions = []
    heatmaps_np = heatmaps.detach().cpu().numpy()
    gaze_inouts_np = gaze_inouts.detach().cpu().reshape(-1).numpy()
    for heatmap, gaze_inout, (width, height) in zip(heatmaps_np, gaze_inouts_np, sizes):
        heatmap = np.squeeze(heatmap)
        pred_x, pred_y = dark_inference(heatmap) if use_dark_inference else argmax_pts(heatmap)
        heatmap_h, heatmap_w = heatmap.shape[-2:]
        norm_x = float(np.clip(pred_x / heatmap_w, 0.0, 1.0))
        norm_y = float(np.clip(pred_y / heatmap_h, 0.0, 1.0))
        predictions.append(((norm_x * width, norm_y * height), (norm_x, norm_y), float(gaze_inout)))
    return predictions


def draw_detection_box(
    frame,
    detection: Detection,
    color: Tuple[int, int, int],
    thickness: int,
    label: Optional[str] = None,
) -> None:
    x1, y1, x2, y2 = map(int, detection.xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(
            frame,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2 if thickness > 1 else 1,
            cv2.LINE_AA,
        )


def draw_godet_visualization(
    image_path: Path,
    output_path: Path,
    *,
    head: Optional[Detection],
    objects: Sequence[Detection],
    gazed_object: Optional[Detection],
    gaze_point: Optional[Tuple[float, float]],
    gaze_inout: Optional[float],
    label: str,
    reason: str,
) -> None:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    for obj in objects:
        color = (90, 90, 90)
        thickness = 1
        if gazed_object is not None and obj.xyxy == gazed_object.xyxy:
            color = (0, 180, 0)
            thickness = 3
        draw_detection_box(frame, obj, color, thickness, f"{obj.class_name} {obj.confidence:.2f}")

    if head is not None:
        draw_detection_box(frame, head, (255, 0, 0), 2, f"Head {head.confidence:.2f}")

    if gaze_point is not None:
        gx, gy = map(int, gaze_point)
        cv2.circle(frame, (gx, gy), 6, (0, 255, 255), -1)
        cv2.drawMarker(
            frame,
            (gx, gy),
            (0, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=24,
            thickness=2,
        )
        if head is not None:
            hx1, hy1, hx2, hy2 = head.xyxy
            cv2.line(
                frame,
                (int((hx1 + hx2) / 2), int((hy1 + hy2) / 2)),
                (gx, gy),
                (0, 255, 255),
                2,
            )

    title = f"gazed object: {label}"
    detail = f"inout={gaze_inout:.3f}  {reason}" if gaze_inout is not None else reason
    cv2.putText(frame, title, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, detail, (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)


def output_path_for(image_path: Path, input_root: Path, output_dir: Path) -> Path:
    if input_root.is_dir():
        try:
            rel_path = image_path.relative_to(input_root)
        except ValueError:
            rel_path = Path(image_path.name)
        return output_dir / rel_path.with_name(f"{rel_path.stem}_godet.png")
    return output_dir / f"{image_path.stem}_godet.png"


def visualize_frame_batch(
    frame_paths: Sequence[Path],
    *,
    input_root: Path,
    batch_output_dir: Path,
    yolo: YOLO,
    gaze_model: torch.nn.Module,
    gaze_transform,
    gaze_input_size: int,
    yolo_conf: float,
    head_conf: float,
    gaze_inout_threshold: float,
    max_object_distance: Optional[float],
    max_object_distance_ratio: float,
    object_class_names: Sequence[str],
    device: str,
    use_dark_inference: bool,
) -> List[dict]:
    rows = []
    yolo_results = yolo([str(path) for path in frame_paths], conf=min(yolo_conf, head_conf), verbose=False)
    prepared = []

    for frame_path, yolo_result in zip(frame_paths, yolo_results):
        detections = yolo_detections(yolo_result)
        heads = filter_by_confidence(filter_by_names(detections, HEAD_CLASS_NAMES), head_conf)
        objects = filter_by_confidence(
            non_head_objects(detections, HEAD_CLASS_NAMES, object_class_names),
            yolo_conf,
        )
        head = choose_head(heads)

        if head is None:
            output_path = output_path_for(frame_path, input_root, batch_output_dir)
            draw_godet_visualization(
                frame_path,
                output_path,
                head=None,
                objects=objects,
                gazed_object=None,
                gaze_point=None,
                gaze_inout=None,
                label="None",
                reason="no_head_detection",
            )
            rows.append(
                {
                    "image": str(frame_path),
                    "output": str(output_path),
                    "label": "None",
                    "gaze_inout": "",
                    "gaze_x": "",
                    "gaze_y": "",
                    "reason": "no_head_detection",
                }
            )
            continue

        image = result_image_to_pil(yolo_result, frame_path)
        width, height = image.size
        max_distance_px = (
            max_object_distance
            if max_object_distance is not None
            else ((width * width + height * height) ** 0.5) * max_object_distance_ratio
        )
        prepared.append((frame_path, image, head, objects, max_distance_px))

    if not prepared:
        return rows

    gaze_predictions = gaze_batch_predict(
        gaze_model,
        [item[1] for item in prepared],
        [item[2].xyxy for item in prepared],
        gaze_transform,
        gaze_input_size,
        device,
        use_dark_inference,
    )

    for (frame_path, _, head, objects, max_distance_px), (gaze_point, _, gaze_inout) in zip(
        prepared,
        gaze_predictions,
    ):
        if gaze_inout >= gaze_inout_threshold:
            gazed_object, reason, object_distance = aggregate_gazed_object(
                objects,
                gaze_point,
                max_distance_px,
            )
        else:
            gazed_object = None
            reason = "gaze_out_of_frame"
            object_distance = None

        label = gazed_object.class_name if gazed_object is not None else "None"
        output_path = output_path_for(frame_path, input_root, batch_output_dir)
        draw_godet_visualization(
            frame_path,
            output_path,
            head=head,
            objects=objects,
            gazed_object=gazed_object,
            gaze_point=gaze_point,
            gaze_inout=gaze_inout,
            label=label,
            reason=reason,
        )
        rows.append(
            {
                "image": str(frame_path),
                "output": str(output_path),
                "label": label,
                "gaze_inout": f"{gaze_inout:.6f}",
                "gaze_x": f"{gaze_point[0]:.3f}",
                "gaze_y": f"{gaze_point[1]:.3f}",
                "reason": reason,
                "object_distance_px": "" if object_distance is None else f"{object_distance:.3f}",
                "max_object_distance_px": f"{max_distance_px:.3f}",
            }
        )

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize GODet-style gaze-object inference using YOLOv8n object/head "
            "detections plus ViTGaze gaze prediction."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        default=None,
        help=(
            "Optional single image or directory. If omitted, the script uses "
            "--hiphop_gaze_dir/--split/--participants like infer_gaze_object_hiphop_gaze_faster.py."
        ),
    )
    parser.add_argument("--hiphop_gaze_dir", "--hiphop-gaze-dir", default="hiphop_gaze")
    parser.add_argument("--split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--participants", choices=["v1", "v2", "merged"], default=None)
    parser.add_argument("-o", "--output_dir", "--output-dir", default="visuals/godet_outputs")
    parser.add_argument("--summary_csv", "--summary-csv", default=None)
    parser.add_argument("--yolo_weights", "--yolo-weights", default=DEFAULT_YOLO_WEIGHTS)
    parser.add_argument("--vitgaze_config", "--gaze-config", default=DEFAULT_GAZE_CONFIG)
    parser.add_argument("--vitgaze_weights", "--gaze-weights", default=DEFAULT_GAZE_WEIGHTS)
    parser.add_argument("--yolo_conf", "--yolo-conf", type=float, default=0.6)
    parser.add_argument("--head_conf", "--head-conf", type=float, default=0.6)
    parser.add_argument("--gaze_inout_threshold", "--gaze-inout-threshold", type=float, default=0.5)
    parser.add_argument("--max_object_distance", "--max-object-distance", type=float, default=None)
    parser.add_argument("--max_object_distance_ratio", "--max-object-distance-ratio", type=float, default=0.08)
    parser.add_argument("--object_classes", "--object-classes", nargs="+", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_dark_inference", "--use-dark-inference", action="store_true")
    parser.add_argument("--skip_frame", "--skip-frame", type=int, default=1)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=16)
    parser.add_argument("--max_videos", "--max-videos", type=int, default=None)
    parser.add_argument("--max_frames", "--max-frames", type=int, default=None)
    parser.add_argument("--recursive", action="store_true", help="Recursively search --input directories.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.skip_frame < 1:
        raise ValueError("--skip_frame must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")

    output_dir = repo_path(args.output_dir)
    yolo_weights = repo_path(args.yolo_weights)
    gaze_config = repo_path(args.vitgaze_config)
    gaze_weights = repo_path(args.vitgaze_weights)
    require_file(yolo_weights, "YOLO weights")
    require_file(gaze_config, "ViTGaze config")
    require_file(gaze_weights, "ViTGaze checkpoint")

    yolo = YOLO(str(yolo_weights))
    gaze_input_size = input_size_from_config(gaze_config)
    gaze_transform = image_transform(gaze_input_size)
    gaze_model = load_hiphop_gaze_model(gaze_config, gaze_weights, args.device)
    object_class_names = args.object_classes or HIPHOP_OBJECT_CLASS_NAMES

    if args.input:
        input_root = repo_path(args.input)
        require_file(input_root, "Input image or directory")
        frame_paths = collect_input_images(input_root, args.recursive)
        if args.max_frames is not None:
            frame_paths = frame_paths[: args.max_frames]
        batches = [
            (input_root, output_dir, batch_paths)
            for batch_paths in chunks(frame_paths, args.batch_size)
        ]
    else:
        hiphop_gaze_dir = repo_path(args.hiphop_gaze_dir)
        require_file(hiphop_gaze_dir, "HIPHOP gaze directory")
        video_dirs = list(iter_video_dirs(hiphop_gaze_dir, args.split, args.participants))
        if args.max_videos is not None:
            video_dirs = video_dirs[: args.max_videos]
        batches = []
        for participant, video, video_dir in video_dirs:
            frame_paths = iter_frame_paths(video_dir, args.max_frames, args.skip_frame)
            video_output_dir = output_dir / participant / video
            batches.extend(
                (video_dir, video_output_dir, batch_paths)
                for batch_paths in chunks(frame_paths, args.batch_size)
            )
            print(f"{participant}/{video}: {len(frame_paths)} frames -> {video_output_dir}", file=sys.stderr)

    rows = []
    for input_root, batch_output_dir, batch_paths in batches:
        rows.extend(
            visualize_frame_batch(
                batch_paths,
                input_root=input_root,
                batch_output_dir=batch_output_dir,
                yolo=yolo,
                gaze_model=gaze_model,
                gaze_transform=gaze_transform,
                gaze_input_size=gaze_input_size,
                yolo_conf=args.yolo_conf,
                head_conf=args.head_conf,
                gaze_inout_threshold=args.gaze_inout_threshold,
                max_object_distance=args.max_object_distance,
                max_object_distance_ratio=args.max_object_distance_ratio,
                object_class_names=object_class_names,
                device=args.device,
                use_dark_inference=args.use_dark_inference,
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = repo_path(args.summary_csv) if args.summary_csv else output_dir / "godet_predictions.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image",
        "output",
        "label",
        "gaze_inout",
        "gaze_x",
        "gaze_y",
        "reason",
        "object_distance_px",
        "max_object_distance_px",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({"visualizations": len(rows), "output_dir": str(output_dir), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
