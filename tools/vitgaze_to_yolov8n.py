import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable, Iterator, Tuple, Union


TRAIN_PARTICIPANTS = tuple(
    [f"P{i}" for i in range(1, 17)] + [f"P{i}" for i in range(21, 37)]
)
TEST_PARTICIPANTS = tuple(
    [f"P{i}" for i in range(17, 21)] + [f"P{i}" for i in range(37, 41)]
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_repo_path(value: Union[str, Path]) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root() / path
    return path.resolve()


def participant_number(path: Path) -> int:
    if not path.name.startswith("P"):
        return 0
    try:
        return int(path.name[1:])
    except ValueError:
        return 0


def video_number(path: Path) -> int:
    if not path.name.startswith("V"):
        return 0
    try:
        return int(path.name[1:])
    except ValueError:
        return 0


def iter_video_dirs(image_root: Path) -> Iterator[Tuple[str, str, Path]]:
    participant_dirs = sorted(
        (path for path in image_root.iterdir() if path.is_dir()),
        key=participant_number,
    )
    for participant_dir in participant_dirs:
        video_dirs = sorted(
            (path for path in participant_dir.iterdir() if path.is_dir()),
            key=video_number,
        )
        for video_dir in video_dirs:
            yield participant_dir.name, video_dir.name, video_dir


def image_files(video_dir: Path) -> Iterable[Path]:
    return sorted(
        path
        for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def remove_existing_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def link_file(source: Path, target: Path, mode: str) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return "exists"

    if mode == "copy":
        shutil.copy2(source, target)
        return "copy"

    if mode == "symlink":
        target.symlink_to(source)
        return "symlink"

    if mode == "hardlink":
        os.link(source, target)
        return "hardlink"

    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        target.symlink_to(source)
        return "symlink"


def link_video_dir(
    source_dir: Path,
    target_dir: Path,
    mode: str,
    prefer_dir_symlink: bool = True,
) -> Tuple[int, str]:
    if target_dir.exists() or target_dir.is_symlink():
        return 0, "exists"

    if prefer_dir_symlink and mode in {"auto", "symlink"}:
        try:
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            target_dir.symlink_to(source_dir, target_is_directory=True)
            return 0, "dir_symlink"
        except OSError:
            if mode == "symlink":
                raise

    target_dir.mkdir(parents=True, exist_ok=True)
    used_mode = "none"
    count = 0
    for source_file in image_files(source_dir):
        target_file = target_dir / source_file.name
        used_mode = link_file(source_file, target_file, mode)
        count += int(used_mode != "exists")
    return count, used_mode


def remove_broken_symlink(path: Path) -> bool:
    if path.is_symlink() and not path.exists():
        path.unlink()
        return True
    return False


def write_data_yaml(output_root: Path) -> None:
    data_yaml = output_root / "data.yaml"
    if data_yaml.exists():
        return

    lines = [
        f"path: {output_root.as_posix()}",
        "train: images/train",
        "val: images/test",
        "test: images/test",
        "# Keep labels in labels/train and labels/test with matching P*/V*/frame_*.txt paths.",
    ]
    data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")


def convert_to_yolo(args: argparse.Namespace) -> None:
    image_root = resolve_repo_path(args.image_root)
    output_root = resolve_repo_path(args.output_root)
    output_images = output_root / "images"

    if not image_root.exists():
        raise FileNotFoundError(f"ViTGaze image root not found: {image_root}")

    if args.clean_images and output_images.exists():
        remove_existing_path(output_images)

    participant_to_split = {
        **{participant: "train" for participant in TRAIN_PARTICIPANTS},
        **{participant: "test" for participant in TEST_PARTICIPANTS},
    }
    stats = {
        "train": {"videos": 0, "linked_items": 0},
        "test": {"videos": 0, "linked_items": 0},
    }
    skipped = []

    for participant, video, source_dir in iter_video_dirs(image_root):
        split = participant_to_split.get(participant)
        if split is None:
            skipped.append(f"{participant}/{video}: participant is outside full split")
            continue

        target_dir = output_images / split / participant / video
        linked_items, link_mode = link_video_dir(
            source_dir,
            target_dir,
            args.link_mode,
            prefer_dir_symlink=not args.file_links,
        )
        stats[split]["videos"] += 1
        stats[split]["linked_items"] += linked_items
        if link_mode == "exists" and not args.quiet:
            print(f"[SKIP] {target_dir} already exists")

    write_data_yaml(output_root)

    print(f"Source images: {image_root}")
    print(f"YOLO root:     {output_root}")
    print(f"Link mode:     {args.link_mode}")
    for split, split_stats in stats.items():
        print(
            f"{split}: {split_stats['videos']} videos, "
            f"{split_stats['linked_items']} newly linked/copied files"
        )
    if skipped and not args.quiet:
        print(f"Skipped {len(skipped)} videos:")
        for item in skipped:
            print(f"  {item}")
    print("ViTGaze annotations were not modified.")


def convert_to_vitgaze(args: argparse.Namespace) -> None:
    yolo_root = resolve_repo_path(args.output_root)
    yolo_images = yolo_root / "images"
    vitgaze_image_root = resolve_repo_path(args.image_root)

    if not yolo_images.exists():
        raise FileNotFoundError(f"YOLO images root not found: {yolo_images}")

    if args.clean_images and vitgaze_image_root.exists():
        remove_existing_path(vitgaze_image_root)

    stats = {
        "train": {"videos": 0, "linked_items": 0},
        "test": {"videos": 0, "linked_items": 0},
    }
    skipped = []
    replaced_broken = 0

    for split in ("train", "test"):
        split_root = yolo_images / split
        if not split_root.exists():
            skipped.append(f"{split}: missing {split_root}")
            continue

        for participant, video, source_dir in iter_video_dirs(split_root):
            target_dir = vitgaze_image_root / participant / video
            if remove_broken_symlink(target_dir):
                replaced_broken += 1

            linked_items, _ = link_video_dir(
                source_dir,
                target_dir,
                args.link_mode,
                prefer_dir_symlink=not args.file_links,
            )
            stats[split]["videos"] += 1
            stats[split]["linked_items"] += linked_items

    print(f"YOLO images:          {yolo_images}")
    print(f"ViTGaze image root:   {vitgaze_image_root}")
    print(f"Link mode:            {args.link_mode}")
    for split, split_stats in stats.items():
        print(
            f"{split}: {split_stats['videos']} videos, "
            f"{split_stats['linked_items']} newly linked/copied files"
        )
    if replaced_broken:
        print(f"Replaced {replaced_broken} broken video-directory symlinks.")
    if skipped and not args.quiet:
        print(f"Skipped {len(skipped)} items:")
        for item in skipped:
            print(f"  {item}")
    print("YOLO labels and ViTGaze annotations were not modified.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Expose hiphop_gaze/images as a YOLOv8-style images/train and "
            "images/test tree without touching ViTGaze annotations or YOLO labels."
        )
    )
    parser.add_argument(
        "--image_root",
        "--image-root",
        default=os.path.join("hiphop_gaze", "images"),
        help="Source ViTGaze HIPHOP image root laid out as P*/V*/frame_*.jpg.",
    )
    parser.add_argument(
        "--output_root",
        "--output-root",
        default="yolo_hiphop",
        help="YOLO dataset root. Existing labels/ are left untouched.",
    )
    parser.add_argument(
        "--link_mode",
        "--link-mode",
        choices=("auto", "symlink", "hardlink", "copy"),
        default="auto",
        help=(
            "How to expose images. auto tries directory symlinks first, then "
            "file hardlinks, then file symlinks."
        ),
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help=(
            "Convert YOLO-style images/train|test/P*/V* back to "
            "ViTGaze-style images/P*/V*. Uses output_root as the YOLO root "
            "and image_root as the destination."
        ),
    )
    parser.add_argument(
        "--file_links",
        "--file-links",
        action="store_true",
        help=(
            "Create links per image file instead of linking whole video "
            "directories. This is useful when a target loader cannot follow "
            "directory symlinks."
        ),
    )
    parser.add_argument(
        "--clean_images",
        "--clean-images",
        action="store_true",
        help=(
            "Forward mode: remove only output_root/images before rebuilding. "
            "Reverse mode: remove only image_root before rebuilding."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-video skip messages.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reverse:
        convert_to_vitgaze(args)
    else:
        convert_to_yolo(args)


if __name__ == "__main__":
    main()
