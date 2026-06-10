import argparse
import os
from pathlib import Path

from ultralytics import YOLO

from dataset_utils import build_skip_frame_data_yaml

RUN_NAME = "yolo_hiphop_full_train"
DEFAULT_BATCH = 32
DEFAULT_IMGSZ = 704
DEFAULT_WORKERS = 6
#DEFAULT_WORKERS = min(8, max(2, (os.cpu_count() or 4) // 2))


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_data_yaml() -> Path:
    return repo_root() / "yolo_hiphop" / "data.yaml"


def default_last_checkpoint() -> Path:
    runs_root = repo_root() / "yolov8n" / "runs"
    checkpoints = sorted(
        runs_root.glob("yolo_hiphop*/weights/last.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not checkpoints:
        raise FileNotFoundError(f"No last.pt checkpoint found under {runs_root}")
    return checkpoints[0]


def validate_dataset(data_yaml: Path) -> None:
    if not data_yaml.exists():
        raise FileNotFoundError(f"YOLO data.yaml not found: {data_yaml}")

    dataset_root = data_yaml.parent
    required_dirs = (
        dataset_root / "images" / "train",
        dataset_root / "images" / "test",
        dataset_root / "labels" / "train",
        dataset_root / "labels" / "test",
    )
    missing = [path for path in required_dirs if not path.exists()]
    if missing:
        formatted = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(f"Missing YOLO dataset directories:\n{formatted}")


def cache_arg(value: str):
    if value == "none":
        return False
    return value


def augmentation_args(preset: str) -> dict:
    common = {
        "augment": True,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "translate": 0.1,
        "scale": 0.5,
        "perspective": 0.0,
        "fliplr": 0.5,
        "flipud": 0.0,
        "copy_paste": 0.0,
    }
    if preset == "fast":
        return {
            **common,
            "degrees": 0.0,
            "shear": 0.0,
            "mosaic": 0.0,
            "mixup": 0.0,
        }
    if preset == "strong":
        return {
            **common,
            "degrees": 5.0,
            "shear": 2.0,
            "mosaic": 1.0,
            "mixup": 0.1,
        }
    return {
        **common,
        "degrees": 2.0,
        "shear": 0.0,
        "mosaic": 0.5,
        "mixup": 0.0,
    }


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Train YOLOv8n on the combined yolo_hiphop dataset "
            "(P1-P16 and P21-P36 train, P17-P20 and P37-P40 test)."
        )
    )
    parser.add_argument(
        "--data",
        default=str(default_data_yaml()),
        help="Path to the combined yolo_hiphop data.yaml.",
    )
    parser.add_argument(
        "--weights",
        default=str(root / "yolov8n.pt"),
        help="Initial YOLOv8n weights.",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help=(
            "Resume an interrupted/completed run from last.pt. Use without a "
            "value to auto-pick the newest yolo_hiphop*/weights/last.pt, or "
            "pass a checkpoint path."
        ),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help=(
            "Training image size. Default targets higher GPU memory usage; "
            "lower this if training is too slow or CUDA runs out of memory."
        ),
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_BATCH,
        help=(
            "Training batch size. Default targets about 5.5GB GPU memory on "
            "the current setup; lower this if CUDA runs out of memory."
        ),
    )
    parser.add_argument("--device", default="0", help="Use 0 for GPU or cpu.")
    parser.add_argument(
        "--skip-frame",
        "--skip_frame",
        type=int,
        default=1,
        help="Use every Nth frame per video without modifying the dataset.",
    )
    parser.add_argument(
        "--project",
        default=str(root / "yolov8n" / "runs"),
        help="Directory where training runs are saved.",
    )
    parser.add_argument("--name", default=RUN_NAME)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Dataloader workers. Increase if GPU utilization is low.",
    )
    parser.add_argument(
        "--cache",
        choices=("none", "ram", "disk"),
        default="none",
        help=(
            "Cache images for faster epochs. Default is 'none' to avoid using "
            "extra storage space."
        ),
    )
    parser.add_argument(
        "--aug-preset",
        choices=("fast", "balanced", "strong"),
        default="fast",
        help=(
            "Augmentation profile. 'fast' keeps basic color/flip/scale "
            "augmentation but disables slower mosaic and mixup."
        ),
    )
    parser.add_argument(
        "--val",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run validation during training. Use --no-val to shorten each epoch.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use mixed precision training. Leave enabled for speed/stability; "
            "use --no-amp only if you deliberately want more VRAM pressure."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.skip_frame < 1:
        raise ValueError("--skip-frame must be 1 or greater.")

    data_yaml = build_skip_frame_data_yaml(Path(args.data), args.skip_frame)
    validate_dataset(Path(args.data))

    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = (
            default_last_checkpoint()
            if args.resume == "auto"
            else Path(args.resume).resolve()
        )
        if not resume_checkpoint.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")
        print(f"Resuming from: {resume_checkpoint}")

    model = YOLO(str(resume_checkpoint) if resume_checkpoint else args.weights)
    train_kwargs = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        cache=cache_arg(args.cache),
        val=args.val,
        amp=args.amp,
        **augmentation_args(args.aug_preset),
    )
    if resume_checkpoint:
        train_kwargs["resume"] = True

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
