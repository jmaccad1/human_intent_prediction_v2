import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent
VITGAZE_ROOT = REPO_ROOT / "ViTGaze"
DETECTRON2_ROOT = VITGAZE_ROOT / "src" / "detectron2"

for import_root in (DETECTRON2_ROOT, VITGAZE_ROOT, REPO_ROOT):
    import_root_str = str(import_root)
    if import_root_str not in sys.path:
        sys.path.insert(0, import_root_str)

import cv2
import numpy as np
import torch
from detectron2.config import LazyConfig, instantiate
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

from data.data_utils import get_head_box_channel  # noqa: E402


DEFAULT_GAZE_CONFIG = "configs/hiphop_dataset_full_6gb.py"
DEFAULT_GAZE_WEIGHTS = "output/hiphop_dataset_full_6gb/model_final.pth"
DEFAULT_YOLO_WEIGHTS = "yolov8n/runs/yolo_hiphop_full_train10/weights/best.pt"
HEAD_CLASS_NAMES = ("Head",)


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    xyxy: Tuple[float, float, float, float]


HIPHOP_OBJECT_CLASS_NAMES = (
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
)


def detection_to_dict(detection: Optional[Detection]) -> Optional[dict]:
    if detection is None:
        return None
    return asdict(detection)


def repo_path(path: str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved.resolve()


def require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def normalize_name(name: str) -> str:
    return str(name).replace("_", "").replace(" ", "").strip().lower()


def choose_head(heads: Sequence[Detection]) -> Optional[Detection]:
    if not heads:
        return None
    return max(heads, key=lambda det: det.confidence)


def filter_by_names(
    detections: Sequence[Detection],
    names: Iterable[str],
) -> List[Detection]:
    wanted = {normalize_name(name) for name in names}
    return [det for det in detections if normalize_name(det.class_name) in wanted]


def filter_by_confidence(
    detections: Sequence[Detection],
    min_confidence: float,
) -> List[Detection]:
    return [det for det in detections if det.confidence >= min_confidence]


def non_head_objects(
    detections: Sequence[Detection],
    head_class_names: Iterable[str],
    object_class_names: Optional[Iterable[str]],
) -> List[Detection]:
    if object_class_names:
        return filter_by_names(detections, object_class_names)

    head_names = {normalize_name(name) for name in head_class_names}
    return [det for det in detections if normalize_name(det.class_name) not in head_names]


def yolo_detections(result) -> List[Detection]:
    detections = []
    if result.boxes is None:
        return detections

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)

    for box, conf, class_id in zip(boxes, confs, classes):
        detections.append(
            Detection(
                class_id=int(class_id),
                class_name=str(result.names[int(class_id)]),
                confidence=float(conf),
                xyxy=tuple(float(value) for value in box),
            )
        )
    return detections


def input_size_from_config(config_file: Path) -> int:
    cfg = LazyConfig.load(str(config_file))
    if hasattr(cfg.dataloader, "val") and hasattr(cfg.dataloader.val, "input_size"):
        return int(cfg.dataloader.val.input_size)
    if hasattr(cfg.dataloader, "train") and hasattr(cfg.dataloader.train, "input_size"):
        return int(cfg.dataloader.train.input_size)
    return 224


def load_hiphop_gaze_model(config_file: Path, weights: Path, device: str) -> torch.nn.Module:
    require_file(config_file, "ViTGaze config")
    require_file(weights, "ViTGaze checkpoint")

    cfg = LazyConfig.load(str(config_file))
    cfg.train.device = device
    cfg.model.device = device

    model = instantiate(cfg.model)
    checkpoint = torch.load(str(weights), map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def image_transform(input_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def argmax_pts(heatmap: np.ndarray) -> Tuple[float, float]:
    idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
    pred_y, pred_x = map(float, idx)
    return pred_x, pred_y


def dark_inference(heatmap: np.ndarray, gaussian_kernel: int = 39) -> Tuple[float, float]:
    pred_x, pred_y = argmax_pts(heatmap)
    pred_x, pred_y = int(pred_x), int(pred_y)
    height, width = heatmap.shape[-2:]

    orig_max = heatmap.max()
    border = (gaussian_kernel - 1) // 2
    dark = np.zeros((height + 2 * border, width + 2 * border))
    dark[border:-border, border:-border] = heatmap.copy()
    dark = cv2.GaussianBlur(dark, (gaussian_kernel, gaussian_kernel), 0)
    heatmap = dark[border:-border, border:-border].copy()
    heatmap *= orig_max / np.max(heatmap)

    heatmap = np.maximum(heatmap, 1e-10)
    heatmap = np.log(heatmap)
    if 1 < pred_x < width - 2 and 1 < pred_y < height - 2:
        dx = 0.5 * (heatmap[pred_y][pred_x + 1] - heatmap[pred_y][pred_x - 1])
        dy = 0.5 * (heatmap[pred_y + 1][pred_x] - heatmap[pred_y - 1][pred_x])
        dxx = 0.25 * (
            heatmap[pred_y][pred_x + 2]
            - 2 * heatmap[pred_y][pred_x]
            + heatmap[pred_y][pred_x - 2]
        )
        dxy = 0.25 * (
            heatmap[pred_y + 1][pred_x + 1]
            - heatmap[pred_y - 1][pred_x + 1]
            - heatmap[pred_y + 1][pred_x - 1]
            + heatmap[pred_y - 1][pred_x - 1]
        )
        dyy = 0.25 * (
            heatmap[pred_y + 2][pred_x]
            - 2 * heatmap[pred_y][pred_x]
            + heatmap[pred_y - 2][pred_x]
        )
        determinant = dxx * dyy - dxy**2
        if determinant != 0:
            derivative = np.array([dx, dy])
            hessian = np.array([[dxx, dxy], [dxy, dyy]])
            offset_x, offset_y = -np.linalg.inv(hessian).dot(derivative)
            pred_x += offset_x
            pred_y += offset_y
    return pred_x, pred_y


def run_gaze_model(
    model: torch.nn.Module,
    image: Image.Image,
    head_box: Tuple[float, float, float, float],
    input_size: int,
    device: str,
    use_dark_inference: bool,
) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    width, height = image.size
    transform = image_transform(input_size)
    head_channel = get_head_box_channel(
        *head_box,
        width=width,
        height=height,
        resolution=input_size,
        coordconv=False,
    )

    batch = {
        "images": transform(image).unsqueeze(0).to(device),
        "head_channels": head_channel.unsqueeze(0).unsqueeze(0).to(device),
    }

    with torch.no_grad():
        heatmap, gaze_inout = model(batch)

    heatmap_np = heatmap.squeeze().detach().cpu().numpy()
    if use_dark_inference:
        pred_x, pred_y = dark_inference(heatmap_np)
    else:
        pred_x, pred_y = argmax_pts(heatmap_np)

    heatmap_h, heatmap_w = heatmap_np.shape[-2:]
    norm_x = float(np.clip(pred_x / heatmap_w, 0.0, 1.0))
    norm_y = float(np.clip(pred_y / heatmap_h, 0.0, 1.0))
    gaze_point = (norm_x * width, norm_y * height)
    inout_score = float(gaze_inout.squeeze().detach().cpu().item())
    return gaze_point, (norm_x, norm_y), inout_score


def draw_prediction(
    frame,
    head: Optional[Detection],
    objects: Sequence[Detection],
    gazed_object: Optional[Detection],
    gaze_point: Optional[Tuple[float, float]],
    label: str,
) -> None:
    for obj in objects:
        x1, y1, x2, y2 = map(int, obj.xyxy)
        color = (80, 80, 80)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        cv2.putText(
            frame,
            f"{obj.class_name} {obj.confidence:.2f}",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    if head is not None:
        x1, y1, x2, y2 = map(int, head.xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if gaze_point is not None:
        gx, gy = map(int, gaze_point)
        cv2.circle(frame, (gx, gy), 5, (0, 0, 255), -1)
        if head is not None:
            hx1, hy1, hx2, hy2 = head.xyxy
            cv2.line(
                frame,
                (int((hx1 + hx2) / 2), int((hy1 + hy2) / 2)),
                (gx, gy),
                (0, 0, 255),
                2,
            )

    if gazed_object is not None:
        x1, y1, x2, y2 = map(int, gazed_object.xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 0), 2)

    cv2.putText(
        frame,
        label,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def read_image_bgr(image_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    return image


def point_box_distance(
    box: Tuple[float, float, float, float],
    point: Tuple[float, float],
) -> float:
    x1, y1, x2, y2 = box
    x, y = point
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return (dx * dx + dy * dy) ** 0.5


def point_box_center_distance(
    box: Tuple[float, float, float, float],
    point: Tuple[float, float],
) -> float:
    x1, y1, x2, y2 = box
    x, y = point
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    return ((center_x - x) ** 2 + (center_y - y) ** 2) ** 0.5


def aggregate_gazed_object(
    objects: Sequence[Detection],
    gaze_point: Tuple[float, float],
    max_distance_px: float,
) -> Tuple[Optional[Detection], str, Optional[float]]:
    if not objects:
        return None, "no_object_detections", None

    ranked = sorted(
        (
            (
                obj,
                point_box_distance(obj.xyxy, gaze_point),
                point_box_center_distance(obj.xyxy, gaze_point),
            )
            for obj in objects
        ),
        key=lambda item: item[2],
    )
    gazed_object, box_distance, center_distance = ranked[0]

    if box_distance == 0:
        return gazed_object, "gaze_point_inside_box_closest_center", center_distance
    if center_distance <= max_distance_px:
        return gazed_object, "closest_center_within_distance", center_distance
    return None, "gaze_point_too_far_from_objects", center_distance


def infer_image(
    image_path: Path,
    *,
    yolo: YOLO,
    gaze_model: torch.nn.Module,
    gaze_input_size: int,
    yolo_conf: float,
    head_conf: float,
    gaze_inout_threshold: float,
    max_object_distance: Optional[float],
    max_object_distance_ratio: float,
    object_class_names: Optional[Sequence[str]],
    device: str,
    use_dark_inference: bool,
) -> Tuple[dict, List[Detection]]:
    frame_bgr = read_image_bgr(image_path)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    width, height = image.size
    image_diagonal = (width * width + height * height) ** 0.5
    max_distance_px = (
        max_object_distance
        if max_object_distance is not None
        else image_diagonal * max_object_distance_ratio
    )

    yolo_result = yolo(frame_rgb, conf=min(yolo_conf, head_conf), verbose=False)[0]
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
        return (
            {
                "label": None,
                "reason": "no_head_detection",
                "looking_inside_frame": False,
                "gaze_inout": None,
                "gaze_inout_threshold": gaze_inout_threshold,
                "gaze_point": None,
                "gaze_point_norm": None,
                "object_distance_px": None,
                "max_object_distance_px": max_distance_px,
                "head": None,
                "gazed_object": None,
                "objects": [detection_to_dict(obj) for obj in objects],
            },
            detections,
        )

    gaze_point, gaze_point_norm, gaze_inout = run_gaze_model(
        gaze_model,
        image,
        head.xyxy,
        gaze_input_size,
        device,
        use_dark_inference,
    )
    looking_inside_frame = gaze_inout >= gaze_inout_threshold

    if looking_inside_frame:
        gazed_object, reason, object_distance = aggregate_gazed_object(
            objects,
            gaze_point,
            max_distance_px,
        )
    else:
        gazed_object = None
        reason = "gaze_out_of_frame"
        object_distance = None

    return (
        {
            "label": gazed_object.class_name if gazed_object else None,
            "reason": reason,
            "looking_inside_frame": looking_inside_frame,
            "gaze_inout": gaze_inout,
            "gaze_inout_threshold": gaze_inout_threshold,
            "gaze_point": {
                "x": gaze_point[0],
                "y": gaze_point[1],
            },
            "gaze_point_norm": {
                "x": gaze_point_norm[0],
                "y": gaze_point_norm[1],
            },
            "object_distance_px": object_distance,
            "max_object_distance_px": max_distance_px,
            "head": detection_to_dict(head),
            "gazed_object": detection_to_dict(gazed_object),
            "objects": [detection_to_dict(obj) for obj in objects],
        },
        detections,
    )


def save_visualization(
    image_path: Path,
    result: dict,
    output_path: Path,
) -> None:
    frame = read_image_bgr(image_path)
    head = Detection(**result["head"]) if result["head"] else None
    objects = [Detection(**obj) for obj in result["objects"]]
    gazed_object = Detection(**result["gazed_object"]) if result["gazed_object"] else None
    gaze_point = None
    if result["gaze_point"] is not None:
        gaze_point = (result["gaze_point"]["x"], result["gaze_point"]["y"])
    label = result["label"] if result["label"] is not None else "None"

    draw_prediction(
        frame,
        head,
        objects,
        gazed_object,
        gaze_point,
        label,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one-image gaze-object inference. YOLOv8n detects objects and "
            "head boxes; ViTGaze receives the image plus selected head box and "
            "predicts the gaze point and in/out score."
        )
    )
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument(
        "--yolo_weights",
        "--yolo-weights",
        default=DEFAULT_YOLO_WEIGHTS,
        help="YOLOv8n weights path.",
    )
    parser.add_argument(
        "--vitgaze_config",
        "--gaze_config",
        "--gaze-config",
        default=DEFAULT_GAZE_CONFIG,
        help="ViTGaze LazyConfig path.",
    )
    parser.add_argument(
        "--vitgaze_weights",
        "--gaze_weights",
        "--gaze-weights",
        default=DEFAULT_GAZE_WEIGHTS,
        help="ViTGaze checkpoint path.",
    )
    parser.add_argument("--yolo_conf", "--yolo-conf", type=float, default=0.6)
    parser.add_argument("--head_conf", "--head-conf", type=float, default=0.6)
    parser.add_argument(
        "--gaze_inout_threshold",
        "--gaze-inout-threshold",
        type=float,
        default=0.5,
        help="Scores at or above this threshold are treated as inside-frame gaze.",
    )
    parser.add_argument(
        "--object_classes",
        "--object-classes",
        nargs="+",
        default=None,
        help=(
            "Optional YOLO class names to consider as gaze-target objects. "
            "Defaults to the 15 HIPHOP object classes."
        ),
    )
    parser.add_argument(
        "--max_object_distance",
        "--max-object-distance",
        type=float,
        default=None,
        help=(
            "Maximum gaze-point distance in pixels from an object box. "
            "Overrides --max_object_distance_ratio when set."
        ),
    )
    parser.add_argument(
        "--max_object_distance_ratio",
        "--max-object-distance-ratio",
        type=float,
        default=0.08,
        help=(
            "Maximum gaze-point distance as a fraction of the image diagonal "
            "when --max_object_distance is not set."
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_dark_inference", "--use-dark-inference", action="store_true")
    parser.add_argument("--save_vis", "--save-vis", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_path = repo_path(args.image)
    yolo_weights = repo_path(args.yolo_weights)
    gaze_config = repo_path(args.vitgaze_config)
    gaze_weights = repo_path(args.vitgaze_weights)

    require_file(image_path, "Input image")
    require_file(yolo_weights, "YOLO weights")
    require_file(gaze_config, "ViTGaze config")
    require_file(gaze_weights, "ViTGaze checkpoint")

    yolo = YOLO(str(yolo_weights))
    gaze_input_size = input_size_from_config(gaze_config)
    gaze_model = load_hiphop_gaze_model(gaze_config, gaze_weights, args.device)
    object_class_names = args.object_classes or HIPHOP_OBJECT_CLASS_NAMES

    result, detections = infer_image(
        image_path,
        yolo=yolo,
        gaze_model=gaze_model,
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
    result["image"] = str(image_path)
    result["detections"] = [detection_to_dict(det) for det in detections]

    summary = {
        "gazed_item": result["label"] if result["label"] is not None else "None",
        "gaze_inout": result["gaze_inout"],
        "head": result["head"]["xyxy"] if result["head"] is not None else None,
        "gaze_pred": result["gaze_point"],
    }
    print(json.dumps(summary, indent=2))

    if args.save_vis:
        save_visualization(image_path, result, repo_path(args.save_vis))


if __name__ == "__main__":
    main()
