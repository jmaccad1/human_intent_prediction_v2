import argparse
from pathlib import Path

from ultralytics import YOLO


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_checkpoint() -> Path:
    root = repo_root()
    preferred = root / "yolov8n" / "models" / "best.pt"
    if preferred.exists():
        return preferred

    checkpoints = sorted(
        (root / "yolov8n" / "runs").glob("yolo_hiphop*_train*/weights/best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return checkpoints[0] if checkpoints else root / "yolov8n.pt"


def default_source() -> Path:
    return repo_root() / "yolo_hiphop" / "images" / "test"


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Run HIPHOP YOLOv8n inference and save visualized detections with "
            "class labels and confidence values."
        )
    )
    parser.add_argument(
        "--source",
        default=str(default_source()),
        help="Image, video, directory, glob, or webcam source accepted by Ultralytics.",
    )
    parser.add_argument(
        "--weights",
        default=str(default_checkpoint()),
        help="YOLOv8n checkpoint. Defaults to the newest yolo_hiphop training best.pt.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="Use 0 for GPU or cpu.")
    parser.add_argument(
        "--project",
        default=str(root / "yolov8n" / "runs"),
        help="Directory where visualization runs are saved.",
    )
    parser.add_argument("--name", default="yolo_hiphop_visualize")
    parser.add_argument("--line_width", "--line-width", type=int, default=None)
    parser.add_argument(
        "--save_txt",
        "--save-txt",
        action="store_true",
        help="Save predicted boxes as YOLO txt labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights}")

    model = YOLO(str(weights))
    model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        save=True,
        save_txt=args.save_txt,
        save_conf=True,
        show_labels=True,
        show_conf=True,
        show_boxes=True,
        line_width=args.line_width,
    )


if __name__ == "__main__":
    main()
