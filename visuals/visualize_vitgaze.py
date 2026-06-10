import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from detectron2.config import LazyConfig, instantiate
from PIL import Image


'''
python visuals\visualize_vitgaze.py `
  --input hiphop_gaze\images\P17\V1 `
  --model-weights output\hiphop_dataset_vat_6gb\model_final.pth `
  --annotations hiphop_gaze\annotations\test\P17\V1\P17_V1.txt `
  --output-dir visuals\vitgaze_outputs

python visuals\visualize_vitgaze.py --input hiphop_gaze\images\P17\V1 --model-weights output\hiphop_dataset_full_6gb\model_final.pth --annotations hiphop_gaze\annotations\test\P17\V1\P17_V1.txt --output-dir visuals\vitgaze_outputs  

Example single image:
python visuals\visualize_vitgaze.py `
  -i test_img.jpg `
  -w output\hiphop_dataset_vat_6gb\model_final.pth `
  --head-bbox 980 163 1085 301

'''


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.data_utils import get_head_box_channel, get_transform  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PATH_KEYS = ("path", "image", "image_path", "file", "filename", "frame_name", "frame")
HEAD_BOX_KEYS = (
    ("head_x1", "head_y1", "head_x2", "head_y2"),
    ("bbox_x_min", "bbox_y_min", "bbox_x_max", "bbox_y_max"),
    ("x_min", "y_min", "x_max", "y_max"),
    ("xmin", "ymin", "xmax", "ymax"),
    ("head_bbox_x1", "head_bbox_y1", "head_bbox_x2", "head_bbox_y2"),
)
GAZE_KEYS = (
    ("gaze_x", "gaze_y"),
    ("target_x", "target_y"),
    ("gt_x", "gt_y"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize ViTGaze predictions for a single image or a directory of "
            "images. The model requires a head box; provide one with annotations "
            "or --head-bbox, otherwise the full image is used as a fallback."
        )
    )
    parser.add_argument("-i", "--input", required=True, help="Image file or image directory.")
    parser.add_argument("-w", "--model-weights", required=True, help="Trained ViTGaze .pth weights.")
    parser.add_argument(
        "-c",
        "--config-file",
        default=str(REPO_ROOT / "configs" / "hiphop_dataset_vat_6gb.py"),
        help="LazyConfig file used to build the model.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=str(REPO_ROOT / "visuals" / "vitgaze_outputs"),
        help="Directory where visualized images and predictions.csv are written.",
    )
    parser.add_argument(
        "-a",
        "--annotations",
        default=None,
        help=(
            "Optional CSV/TXT/JSON/JSONL annotations. Supported columns include "
            "frame_name/path, head_x1/head_y1/head_x2/head_y2, and gaze_x/gaze_y."
        ),
    )
    parser.add_argument(
        "--head-bbox",
        nargs=4,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Head box for single-image use, in pixels or normalized [0, 1] coordinates.",
    )
    parser.add_argument("--input-size", type=int, default=None, help="Override model input size.")
    parser.add_argument("--device", default="cuda", help="Inference device, e.g. cuda or cpu.")
    parser.add_argument("--alpha", type=float, default=0.45, help="Heatmap overlay opacity.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search image directories recursively.",
    )
    return parser.parse_args()


def collect_images(input_path: Path, recursive: bool) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    iterator = input_path.rglob("*") if recursive else input_path.iterdir()
    images = sorted(p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise ValueError(f"No images found under {input_path}")
    return images


def normalize_match_key(value: Any) -> str:
    key = str(value).replace("\\", "/").strip()
    return key.lower()


def annotation_keys(record: Dict[str, Any]) -> Iterable[str]:
    for key in PATH_KEYS:
        if key in record and record[key] not in (None, ""):
            value = normalize_match_key(record[key])
            yield value
            yield Path(value).name.lower()
            yield Path(value).stem.lower()


def read_annotations(path: Optional[str]) -> Dict[str, List[Dict[str, Any]]]:
    if path is None:
        return {}

    anno_path = Path(path)
    if not anno_path.exists():
        raise FileNotFoundError(f"Annotation file does not exist: {anno_path}")

    records: List[Dict[str, Any]]
    suffix = anno_path.suffix.lower()
    if suffix in {".json", ".jsonl", ".ndjson"}:
        records = read_json_annotations(anno_path)
    else:
        records = read_table_annotations(anno_path)

    by_key: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        for key in annotation_keys(record):
            by_key.setdefault(key, []).append(record)
    return by_key


def read_json_annotations(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        with path.open(encoding="utf-8-sig") as file:
            return [json.loads(line) for line in file if line.strip()]

    with path.open(encoding="utf-8-sig") as file:
        data = json.load(file)
    if isinstance(data, list):
        return [flatten_record(item) for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("annotations", "items", "samples", "data"):
            if isinstance(data.get(key), list):
                return [flatten_record(item) for item in data[key] if isinstance(item, dict)]
        return [flatten_record(data)]
    raise ValueError(f"Unsupported JSON annotation structure in {path}")


def read_table_annotations(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        sample = file.read(4096)
        file.seek(0)
        sniffer = csv.Sniffer()
        try:
            has_header = sniffer.has_header(sample)
            dialect = sniffer.sniff(sample, delimiters=",\t ")
        except csv.Error:
            has_header = True
            dialect = csv.excel

        rows = list(csv.reader(file, dialect=dialect))
        if not rows:
            return []

        header = [column.strip().lower() for column in rows[0]]
        has_known_header = any(key in header for key in PATH_KEYS)
        if has_header and has_known_header:
            return [dict(zip(header, row)) for row in rows[1:]]

        records = []
        for row in rows:
            if len(row) >= 7:
                records.append(
                    {
                        "frame_name": row[0],
                        "head_x1": row[1],
                        "head_y1": row[2],
                        "head_x2": row[3],
                        "head_y2": row[4],
                        "gaze_x": row[5],
                        "gaze_y": row[6],
                    }
                )
        return records


def flatten_record(record: Dict[str, Any]) -> Dict[str, Any]:
    flattened = dict(record)
    for key in ("head_bbox", "bbox", "box"):
        value = record.get(key)
        if isinstance(value, Sequence) and len(value) >= 4 and not isinstance(value, str):
            flattened.update({"head_x1": value[0], "head_y1": value[1], "head_x2": value[2], "head_y2": value[3]})
    for key in ("gaze", "target", "gaze_point"):
        value = record.get(key)
        if isinstance(value, Sequence) and len(value) >= 2 and not isinstance(value, str):
            flattened.update({"gaze_x": value[0], "gaze_y": value[1]})
    return flattened


def find_annotation_records(
    image_path: Path,
    input_root: Path,
    annotations: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    if not annotations:
        return []

    candidates = [
        normalize_match_key(image_path),
        image_path.name.lower(),
        image_path.stem.lower(),
    ]
    if input_root.is_dir():
        try:
            candidates.insert(0, normalize_match_key(image_path.relative_to(input_root)))
        except ValueError:
            pass

    for key in candidates:
        if key in annotations:
            return annotations[key]
    return []


def as_float(record: Dict[str, Any], key: str) -> Optional[float]:
    value = record.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_head_box(records: Sequence[Dict[str, Any]]) -> Optional[Tuple[float, float, float, float]]:
    for record in records:
        record = flatten_record(record)
        for keys in HEAD_BOX_KEYS:
            values = [as_float(record, key) for key in keys]
            if all(value is not None for value in values):
                return tuple(values)  # type: ignore[return-value]
    return None


def extract_gaze_points(records: Sequence[Dict[str, Any]]) -> List[Tuple[float, float]]:
    points = []
    for record in records:
        record = flatten_record(record)
        for keys in GAZE_KEYS:
            values = [as_float(record, key) for key in keys]
            if all(value is not None for value in values):
                points.append((float(values[0]), float(values[1])))
                break
    return points


def to_pixel_box(
    box: Tuple[float, float, float, float], width: int, height: int
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = min(max(0.0, x1), float(width - 1))
    x2 = min(max(0.0, x2), float(width - 1))
    y1 = min(max(0.0, y1), float(height - 1))
    y2 = min(max(0.0, y2), float(height - 1))
    return x1, y1, x2, y2


def to_normalized_point(point: Tuple[float, float], width: int, height: int) -> Tuple[float, float]:
    x, y = point
    if max(abs(x), abs(y)) > 1.0:
        x /= width
        y /= height
    return min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)


def load_model(config_file: str, weights_path: str, device: str, input_size: Optional[int]) -> Tuple[torch.nn.Module, int]:
    cfg = LazyConfig.load(config_file)
    cfg.train.device = device
    cfg.model.device = device
    if input_size is not None:
        cfg.dataloader.val.input_size = input_size

    model = instantiate(cfg.model)
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    resolved_input_size = int(input_size or cfg.dataloader.val.input_size)
    return model, resolved_input_size


def build_batch(
    image_path: Path,
    transform,
    input_size: int,
    head_box: Tuple[float, float, float, float],
) -> Tuple[Dict[str, torch.Tensor], Image.Image]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    x1, y1, x2, y2 = to_pixel_box(head_box, width, height)

    head_channel = get_head_box_channel(
        x1,
        y1,
        x2,
        y2,
        width,
        height,
        resolution=input_size,
        coordconv=False,
    ).unsqueeze(0)

    return {
        "images": transform(image).unsqueeze(0),
        "head_channels": head_channel.unsqueeze(0),
    }, image


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    heatmap = heatmap.astype(np.float32)
    heatmap -= float(heatmap.min())
    peak_to_peak = float(heatmap.max())
    if peak_to_peak > 0:
        heatmap /= peak_to_peak
    return heatmap


def draw_visualization(
    image: Image.Image,
    heatmap: np.ndarray,
    inout: float,
    head_box: Tuple[float, float, float, float],
    gt_points: Sequence[Tuple[float, float]],
    out_path: Path,
    alpha: float,
) -> Tuple[float, float]:
    width, height = image.size
    pred_y, pred_x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    pred_norm = (pred_x / heatmap.shape[1], pred_y / heatmap.shape[0])

    resized_heatmap = cv2.resize(normalize_heatmap(heatmap), (width, height), interpolation=cv2.INTER_LINEAR)
    color_heatmap = cv2.applyColorMap((resized_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(image_bgr, 1.0, color_heatmap, alpha, 0)

    x1, y1, x2, y2 = map(int, to_pixel_box(head_box, width, height))
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2)

    pred_px = (int(pred_norm[0] * width), int(pred_norm[1] * height))
    cv2.drawMarker(overlay, pred_px, (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=24, thickness=3)
    cv2.circle(overlay, pred_px, 6, (0, 255, 255), -1)

    for gt_point in gt_points:
        gt_norm = to_normalized_point(gt_point, width, height)
        gt_px = (int(gt_norm[0] * width), int(gt_norm[1] * height))
        cv2.circle(overlay, gt_px, 8, (0, 255, 0), 2)

    cv2.putText(
        overlay,
        f"pred=yellow  gt=green  inout={inout:.3f}",
        (12, max(28, height - 16)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)
    return pred_norm


def output_path_for(image_path: Path, input_root: Path, output_dir: Path) -> Path:
    if input_root.is_dir():
        try:
            rel = image_path.relative_to(input_root)
        except ValueError:
            rel = Path(image_path.name)
        return output_dir / rel.with_name(f"{rel.stem}_vitgaze.png")
    return output_dir / f"{image_path.stem}_vitgaze.png"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        device = "cpu"

    annotations = read_annotations(args.annotations)
    images = collect_images(input_path, args.recursive)
    model, input_size = load_model(args.config_file, args.model_weights, device, args.input_size)
    transform = get_transform(input_size, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    summary_rows = []
    warned_full_image = False
    with torch.no_grad():
        for image_path in images:
            records = find_annotation_records(image_path, input_path, annotations)
            image_for_size = Image.open(image_path)
            width, height = image_for_size.size
            image_for_size.close()

            head_box = extract_head_box(records)
            if head_box is None and args.head_bbox is not None:
                head_box = tuple(args.head_bbox)
            if head_box is None:
                head_box = (0.0, 0.0, float(width - 1), float(height - 1))
                if not warned_full_image:
                    print("Warning: no head box found; using the full image as the head region.")
                    warned_full_image = True

            gt_points = extract_gaze_points(records)
            batch, image = build_batch(image_path, transform, input_size, head_box)
            batch = {key: value.to(device) for key, value in batch.items()}
            pred_heatmap, pred_inout = model(batch)
            heatmap = pred_heatmap.squeeze(0).squeeze(0).detach().cpu().numpy()
            inout = float(pred_inout.squeeze().detach().cpu())

            out_path = output_path_for(image_path, input_path, output_dir)
            pred_x, pred_y = draw_visualization(image, heatmap, inout, head_box, gt_points, out_path, args.alpha)
            summary_rows.append(
                {
                    "image": str(image_path),
                    "output": str(out_path),
                    "pred_x": f"{pred_x:.6f}",
                    "pred_y": f"{pred_y:.6f}",
                    "inout": f"{inout:.6f}",
                    "gt_count": len(gt_points),
                }
            )
            print(f"Wrote {out_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "predictions.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image", "output", "pred_x", "pred_y", "inout", "gt_count"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
