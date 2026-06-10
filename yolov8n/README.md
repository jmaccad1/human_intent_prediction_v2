# YOLOv8n HIPHOP Object Detector

This folder contains entry points for training and validating `yolov8n.pt` on
the generated `yolo_hiphop` custom dataset.

The `yolo_hiphop/data.yaml` split is generated from `hiphop_v1` with `P1`-`P16`
as the training set and `P17`-`P20` as the test set.

## Generate Dataset

Run this from the repo root after frames have been extracted into
`hiphop_gaze/images`:

```
python tools\gen_yolo_hiphop.py --clean
```

This writes:

```text
yolo_hiphop/
  data.yaml
  images/train
  images/test
  labels/train
  labels/test
```

## Train

```
python yolov8n\train.py --epochs 100 --batch 8 --device 0
```

For CPU:

```
python yolov8n\train.py --epochs 100 --batch 8 --device cpu
```

The best checkpoint is saved by default at:

```text
yolov8n/runs/yolo_hiphop_train/weights/best.pt
```

## Test

```
python yolov8n\test.py --device 0
```

To validate a specific checkpoint:

```
python yolov8n\test.py --weights path_to_yolov8n_weights.pt --device 0
```

`test.py` evaluates the `test` split by default. To run against another split:

```
python yolov8n\test.py --split val --device 0
```
