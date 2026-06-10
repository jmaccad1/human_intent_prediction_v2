import argparse
from pathlib import Path

from ultralytics import YOLO

from dataset_utils import build_skip_frame_data_yaml

RUN_NAME = "yolo_hiphop_appended_train"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_data_yaml() -> Path:
    return repo_root() / "yolo_hiphop" / "data.yaml"


def default_checkpoint() -> Path:
    root = repo_root()
    preferred = root / "yolov8n" / "runs" / RUN_NAME / "weights" / "best.pt"
    if preferred.exists():
        return preferred

    checkpoints = sorted(
        (root / "yolov8n" / "runs").glob("yolo_hiphop*_train*/weights/best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return checkpoints[0] if checkpoints else root / "yolov8n.pt"


def validate_dataset(data_yaml: Path) -> None:
    if not data_yaml.exists():
        raise FileNotFoundError(f"YOLO data.yaml not found: {data_yaml}")


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Validate a YOLOv8n checkpoint on the combined yolo_hiphop dataset."
    )
    parser.add_argument(
        "--data",
        default=str(default_data_yaml()),
        help="Path to the combined yolo_hiphop data.yaml.",
    )
    parser.add_argument(
        "--weights",
        default=str(default_checkpoint()),
        help="Checkpoint to evaluate. Defaults to the trained best.pt if present.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="Use 0 for GPU or cpu.")
    parser.add_argument(
        "--skip-frame",
        "--skip_frame",
        type=int,
        default=1,
        help="Use every Nth frame per video for validation/testing.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "val", "test"),
        help="Dataset split from data.yaml to evaluate.",
    )
    parser.add_argument(
        "--project",
        default=str(root / "yolov8n" / "runs"),
        help="Directory where validation runs are saved.",
    )
    parser.add_argument("--name", default="yolo_hiphop_test")
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.skip_frame < 1:
        raise ValueError("--skip-frame must be 1 or greater.")

    data_yaml = build_skip_frame_data_yaml(Path(args.data), args.skip_frame)
    validate_dataset(Path(args.data))

    model = YOLO(args.weights)
    model.val(
        data=str(data_yaml),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split=args.split,
        project=args.project,
        name=args.name,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
