# YOLOv8n HIPHOP Object Detector

This folder contains entry points for training and validating `yolov8n.pt` on
the generated `yolo_hiphop` custom dataset.

The `yolo_hiphop/data.yaml` split is generated from `merged dataset` with `P1`-`P16`, `P21`-`P36`
as the training set and `P17`-`P20`,`P37`-`P40`  as the test set.

## Generate Dataset

Run this from the repo root to restructure the ViTGaze schema image frames to yolov8n schema and copy the annotations from the datasets link titled `yolo_hiphop_merged`

```
python tools\vitgaze_to_yolov8n.py
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
