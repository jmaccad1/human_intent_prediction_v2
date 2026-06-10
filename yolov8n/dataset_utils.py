import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _unquote(value: str) -> str:
    return value.strip().strip("'\"")


def _read_top_level_yaml(data_yaml: Path) -> Dict[str, str]:
    values = {}
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip() != line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = _unquote(value)
    return values


def dataset_root(data_yaml: Path) -> Path:
    values = _read_top_level_yaml(data_yaml)
    root_value = values.get("path", "")
    root = Path(root_value) if root_value else data_yaml.parent
    if not root.is_absolute():
        root = data_yaml.parent / root
    return root.resolve()


def resolve_split_path(data_yaml: Path, split: str) -> Path:
    values = _read_top_level_yaml(data_yaml)
    split_value = values.get(split)
    if not split_value:
        raise ValueError(f"{data_yaml} does not define a '{split}' split.")

    split_path = Path(split_value)
    if not split_path.is_absolute():
        split_path = dataset_root(data_yaml) / split_path
    return split_path.resolve()


def _iter_image_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        for line in root.read_text(encoding="utf-8").splitlines():
            path = Path(line.strip())
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path.absolute()
        return

    pending = [root]
    while pending:
        current = pending.pop()
        for path in current.iterdir():
            if path.is_dir():
                pending.append(path)
            elif path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path.absolute()


def _video_sort_key(path: Path) -> Tuple[str, int, str]:
    match = re.search(r"(\d+)(?!.*\d)", path.stem)
    frame_number = int(match.group(1)) if match else -1
    return (path.parent.as_posix(), frame_number, path.name)


def filtered_image_list(split_root: Path, skip_frame: int) -> List[Path]:
    images = sorted(_iter_image_files(split_root), key=_video_sort_key)
    if skip_frame <= 1:
        return images

    selected = []
    current_parent = None
    frame_index = 0
    for image_path in images:
        if image_path.parent != current_parent:
            current_parent = image_path.parent
            frame_index = 0
        if frame_index % skip_frame == 0:
            selected.append(image_path)
        frame_index += 1
    return selected


def build_skip_frame_data_yaml(data_yaml: Path, skip_frame: int) -> Path:
    if skip_frame <= 1:
        return data_yaml

    data_yaml = data_yaml.resolve()
    output_dir = data_yaml.parent / "_skip_frame"
    output_dir.mkdir(parents=True, exist_ok=True)

    split_lists = {}
    for split in ("train", "val", "test"):
        split_root = resolve_split_path(data_yaml, split)
        images = filtered_image_list(split_root, skip_frame)
        if not images:
            raise FileNotFoundError(
                f"No images found for split '{split}' at {split_root}"
            )

        list_path = output_dir / f"skip_{skip_frame:03d}_{split}.txt"
        list_path.write_text(
            "\n".join(path.as_posix() for path in images) + "\n",
            encoding="utf-8",
        )
        split_lists[split] = list_path
        print(
            f"skip_frame={skip_frame}: {split} uses "
            f"{len(images)} images from {split_root}"
        )

    generated_yaml = output_dir / f"data_skip_{skip_frame:03d}.yaml"
    rewritten_lines = []
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        replaced = False
        for split, list_path in split_lists.items():
            if stripped.startswith(f"{split}:") and line.lstrip() == line:
                rewritten_lines.append(f"{split}: {list_path.as_posix()}")
                replaced = True
                break
        if not replaced:
            rewritten_lines.append(line)

    generated_yaml.write_text("\n".join(rewritten_lines) + "\n", encoding="utf-8")
    return generated_yaml
