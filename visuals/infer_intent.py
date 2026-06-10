import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[1]
VITGAZE_DIR = REPO_ROOT / "ViTGaze"
INTENT_CLASSIFIER_DIR = REPO_ROOT / "intent_classifier"
for import_dir in (REPO_ROOT, VITGAZE_DIR, INTENT_CLASSIFIER_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from gazed_object_det import (  # noqa: E402
    HEAD_CLASS_NAMES,
    HIPHOP_OBJECT_CLASS_NAMES,
    Detection,
    aggregate_gazed_object,
    filter_by_confidence,
    filter_by_names,
    image_transform,
    input_size_from_config,
    load_hiphop_gaze_model,
    non_head_objects,
    require_file,
    yolo_detections,
)
from models import get_model  # noqa: E402
from utils.dataset_keys import (  # noqa: E402
    GAZED_OBJECT_MAPPING,
    INTENTIONS_MAPPING_R,
)
from visualize_godet import gaze_batch_predict  # noqa: E402


DEFAULT_GAZE_CONFIG = "ViTGaze/configs/hiphop_dataset_full_6gb.py"
DEFAULT_GAZE_WEIGHTS = "output/hiphop_dataset_full_6gb/model_final.pth"
DEFAULT_YOLO_WEIGHTS = "yolov8n/models/best.pt"
DEFAULT_INTENT_CONFIG = "intent_classifier/config_small.yaml"
DEFAULT_INTENT_WEIGHTS = "intent_classifier/weights/3enc_best.ckpt"


def repo_path(path: str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved.resolve()


def choose_head(heads: Sequence[Detection]) -> Optional[Detection]:
    if not heads:
        return None
    return max(heads, key=lambda detection: detection.confidence)


def normalize_gazed_object(value: Optional[str]) -> str:
    if value is None:
        return "none"

    value = str(value).strip()
    if value.lower() in {"", "nan", "none", "null"}:
        return "none"

    normalized = value.replace("_", " ").title().replace(" ", "")
    aliases = {"None": "none", "Utensil": "Utensils"}
    return aliases.get(normalized, normalized)


def read_video_batch(
    capture: cv2.VideoCapture,
    batch_size: int,
    frame_stride: int,
    max_frames: Optional[int],
    state: Dict[str, int],
) -> Tuple[List[np.ndarray], List[int]]:
    frames = []
    frame_numbers = []

    while len(frames) < batch_size:
        ok, frame = capture.read()
        if not ok:
            break

        frame_number = state["decoded"]
        state["decoded"] += 1
        if frame_number % frame_stride != 0:
            continue

        frames.append(frame)
        frame_numbers.append(frame_number)
        state["selected"] += 1
        if max_frames is not None and state["selected"] >= max_frames:
            break

    return frames, frame_numbers


def infer_gazed_object_batch(
    frames_bgr: Sequence[np.ndarray],
    frame_numbers: Sequence[int],
    *,
    fps: float,
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
    rows = [
        {
            "frame": frame_number,
            "timestamp_seconds": frame_number / fps if fps > 0 else None,
            "gazed_object": "none",
            "reason": "unprocessed",
            "gaze_inout": None,
            "gaze_x": None,
            "gaze_y": None,
            "object_distance_px": None,
        }
        for frame_number in frame_numbers
    ]

    yolo_results = yolo(
        list(frames_bgr),
        conf=min(yolo_conf, head_conf),
        verbose=False,
    )
    prepared = []

    for index, (frame_bgr, yolo_result) in enumerate(
        zip(frames_bgr, yolo_results)
    ):
        detections = yolo_detections(yolo_result)
        heads = filter_by_confidence(
            filter_by_names(detections, HEAD_CLASS_NAMES),
            head_conf,
        )
        objects = filter_by_confidence(
            non_head_objects(detections, HEAD_CLASS_NAMES, object_class_names),
            yolo_conf,
        )
        head = choose_head(heads)
        if head is None:
            rows[index]["reason"] = "no_head_detection"
            continue

        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        width, height = image.size
        distance_limit = (
            max_object_distance
            if max_object_distance is not None
            else float(np.hypot(width, height)) * max_object_distance_ratio
        )
        prepared.append((index, image, head, objects, distance_limit))

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

    for (
        index,
        _,
        _,
        objects,
        distance_limit,
    ), (gaze_point, _, gaze_inout) in zip(prepared, gaze_predictions):
        if gaze_inout < gaze_inout_threshold:
            rows[index].update(
                {
                    "reason": "gaze_out_of_frame",
                    "gaze_inout": gaze_inout,
                    "gaze_x": gaze_point[0],
                    "gaze_y": gaze_point[1],
                }
            )
            continue

        gazed_object, reason, object_distance = aggregate_gazed_object(
            objects,
            gaze_point,
            distance_limit,
        )
        rows[index].update(
            {
                "gazed_object": normalize_gazed_object(
                    gazed_object.class_name if gazed_object else None
                ),
                "reason": reason,
                "gaze_inout": gaze_inout,
                "gaze_x": gaze_point[0],
                "gaze_y": gaze_point[1],
                "object_distance_px": object_distance,
            }
        )

    return rows


def load_intent_config(config_path: Path) -> dict:
    require_file(config_path, "Intent classifier config")
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if int(config.get("depth", -1)) != 3:
        raise ValueError(
            f"Expected a three-encoder intent config, but depth is "
            f"{config.get('depth')!r} in {config_path}"
        )
    return config


def load_intent_model(
    config: dict,
    checkpoint_path: Path,
    device: str,
) -> torch.nn.Module:
    require_file(checkpoint_path, "Intent classifier checkpoint")
    model = get_model(config)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def encode_sequence(sequence: Sequence[str], max_length: int) -> torch.Tensor:
    normalized = [normalize_gazed_object(label) for label in sequence]
    unknown = sorted(
        {label for label in normalized if label not in GAZED_OBJECT_MAPPING}
    )
    if unknown:
        raise ValueError(f"Unknown gazed-object labels: {unknown}")

    padded = normalized[:max_length]
    padded.extend(["none"] * (max_length - len(padded)))
    encoded = [int(GAZED_OBJECT_MAPPING[label]) for label in padded]
    return torch.tensor([encoded], dtype=torch.int64)


def infer_intent(
    model: torch.nn.Module,
    sequence: Sequence[str],
    max_length: int,
    device: str,
) -> dict:
    encoded = encode_sequence(sequence, max_length).to(device)
    with torch.inference_mode():
        probabilities = model(encoded).softmax(dim=-1).squeeze(0).cpu()

    predicted_index = int(probabilities.argmax().item())
    return {
        "intent": INTENTIONS_MAPPING_R[predicted_index],
        "confidence": float(probabilities[predicted_index].item()),
        "probabilities": {
            INTENTIONS_MAPPING_R[index]: float(probability.item())
            for index, probability in enumerate(probabilities)
        },
    }


def write_frame_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame",
        "timestamp_seconds",
        "gazed_object",
        "reason",
        "gaze_inout",
        "gaze_x",
        "gaze_y",
        "object_distance_px",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Infer one intent class from a video's frame-by-frame gazed-object "
            "sequence using GODet and the three-encoder intent classifier."
        )
    )
    parser.add_argument("-i", "--input", required=True, help="Input video path.")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output JSON path. Defaults to visuals/intent_outputs/<video>.json.",
    )
    parser.add_argument(
        "--frames-csv",
        default=None,
        help="Per-frame CSV path. Defaults beside the output JSON.",
    )
    parser.add_argument(
        "--intent-config",
        default=DEFAULT_INTENT_CONFIG,
        help="Three-encoder intent classifier YAML config.",
    )
    parser.add_argument(
        "--intent-weights",
        default=DEFAULT_INTENT_WEIGHTS,
        help="Intent classifier checkpoint.",
    )
    parser.add_argument("--yolo-weights", default=DEFAULT_YOLO_WEIGHTS)
    parser.add_argument("--vitgaze-config", default=DEFAULT_GAZE_CONFIG)
    parser.add_argument("--vitgaze-weights", default=DEFAULT_GAZE_WEIGHTS)
    parser.add_argument("--yolo-conf", type=float, default=0.6)
    parser.add_argument("--head-conf", type=float, default=0.6)
    parser.add_argument("--gaze-inout-threshold", type=float, default=0.5)
    parser.add_argument("--max-object-distance", type=float, default=None)
    parser.add_argument("--max-object-distance-ratio", type=float, default=0.08)
    parser.add_argument("--object-classes", nargs="+", default=None)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Process every Nth decoded frame. The default processes every frame.",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--use-dark-inference", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be >= 1")

    input_path = require_file(repo_path(args.input), "Input video")
    intent_config_path = repo_path(args.intent_config)
    intent_weights_path = repo_path(args.intent_weights)
    yolo_weights_path = require_file(
        repo_path(args.yolo_weights),
        "YOLO weights",
    )
    gaze_config_path = require_file(
        repo_path(args.vitgaze_config),
        "ViTGaze config",
    )
    gaze_weights_path = require_file(
        repo_path(args.vitgaze_weights),
        "ViTGaze checkpoint",
    )

    default_output = (
        REPO_ROOT / "visuals" / "intent_outputs" / f"{input_path.stem}.json"
    )
    output_path = repo_path(args.output) if args.output else default_output
    frames_csv_path = (
        repo_path(args.frames_csv)
        if args.frames_csv
        else output_path.with_name(f"{output_path.stem}_frames.csv")
    )

    intent_config = load_intent_config(intent_config_path)
    intent_model = load_intent_model(
        intent_config,
        intent_weights_path,
        args.device,
    )
    yolo = YOLO(str(yolo_weights_path))
    gaze_input_size = input_size_from_config(gaze_config_path)
    gaze_transform = image_transform(gaze_input_size)
    gaze_model = load_hiphop_gaze_model(
        gaze_config_path,
        gaze_weights_path,
        args.device,
    )

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    decoded_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    state = {"decoded": 0, "selected": 0}
    frame_rows = []
    try:
        while True:
            frames, frame_numbers = read_video_batch(
                capture,
                args.batch_size,
                args.frame_stride,
                args.max_frames,
                state,
            )
            if not frames:
                break

            frame_rows.extend(
                infer_gazed_object_batch(
                    frames,
                    frame_numbers,
                    fps=fps,
                    yolo=yolo,
                    gaze_model=gaze_model,
                    gaze_transform=gaze_transform,
                    gaze_input_size=gaze_input_size,
                    yolo_conf=args.yolo_conf,
                    head_conf=args.head_conf,
                    gaze_inout_threshold=args.gaze_inout_threshold,
                    max_object_distance=args.max_object_distance,
                    max_object_distance_ratio=args.max_object_distance_ratio,
                    object_class_names=(
                        args.object_classes or HIPHOP_OBJECT_CLASS_NAMES
                    ),
                    device=args.device,
                    use_dark_inference=args.use_dark_inference,
                )
            )
            print(
                f"Processed {len(frame_rows)} frames",
                file=sys.stderr,
            )
            if (
                args.max_frames is not None
                and state["selected"] >= args.max_frames
            ):
                break
    finally:
        capture.release()

    if not frame_rows:
        raise RuntimeError(f"No frames were decoded from {input_path}")

    sequence = [row["gazed_object"] for row in frame_rows]
    intent_result = infer_intent(
        intent_model,
        sequence,
        int(intent_config["input_size"]),
        args.device,
    )
    result = {
        "input_video": str(input_path),
        "fps": fps,
        "video_frame_count": decoded_frame_count,
        "processed_frames": len(frame_rows),
        "frame_stride": args.frame_stride,
        "intent_sequence_max_length": int(intent_config["input_size"]),
        "sequence_length_used": min(
            len(sequence),
            int(intent_config["input_size"]),
        ),
        "sequence_truncated": len(sequence) > int(intent_config["input_size"]),
        "intent_config": str(intent_config_path),
        "intent_weights": str(intent_weights_path),
        **intent_result,
        "gazed_objects": sequence,
        "frames_csv": str(frames_csv_path),
    }

    write_frame_csv(frames_csv_path, frame_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
