# Train and Test ViTGaze on Human Intent Prediction Dataset Full (V1 + V2)

This guide trains and evaluates ViTGaze with:

```text
configs/hiphop_dataset_full_6gb.py
```

Run all commands from this directory:

```powershell
cd \ViTGaze
```

## Configuration Summary

The `hiphop_dataset_full_6gb.py` configuration is intended for a CUDA GPU with
approximately 6 GB of VRAM. It uses:

- DINOv2 ViT-S/14 backbone
- input resolution `434`
- batch size `2` on one GPU
- sequence length `4`
- three training epochs
- `sota/videoattentiontarget.pth` as the initial checkpoint

The HIP-HOP participant split is:

- Training: `P1-P16` and `P21-P36`
- Testing: `P17-P20` and `P37-P40`

## 1. Set Up the Environment

Python 3.9 is recommended.

```powershell
conda create -n ViTGaze python=3.9.18
conda activate ViTGaze
```

Install the project dependencies from the parent directory:

```powershell
pip install -r ..\requirements.txt
```

Install the Detectron2 revision used by ViTGaze:

```powershell
pip install "git+https://github.com/facebookresearch/detectron2.git@017abbfa5f2c2a2afa045200c2af9ccf2fc6227f"
```

Use PyTorch, torchvision, xFormers, and a CUDA toolkit combination that is
compatible with the installed NVIDIA driver. Verify CUDA before training:

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Training requires CUDA because the config sets the model device to `cuda`.

## 2. Check the Required Files

The expected layout is:

```text
Main Folder
|-- hiphop_gaze\
|   |-- images\
|   |   `-- P1\V1\frame_0000.jpg
|   |-- annotations\
|   |   |-- train\
|   |   `-- test\
|   `-- head_masks\
|       `-- images\
`-- ViTGaze\
    |-- configs\
    |-- data\
    |-- pretrained\
    |   `-- dinov2_small.pth
    |-- sota\
    |   `-- videoattentiontarget.pth
    `-- tools\
```

The training dataset root is currently set in
`configs/common/dataloader.py`:

```python
DATA_ROOT = r"\human_intent_prediction_v2\ViTGaze"
```

Change `DATA_ROOT` if the dataset is moved. The following files must exist
before training:

```powershell
Test-Path ..\hiphop_gaze\images
Test-Path ..\hiphop_gaze\annotations\train
Test-Path ..\hiphop_gaze\annotations\test
Test-Path ..\hiphop_gaze\head_masks\images
Test-Path .\pretrained\dinov2_small.pth
Test-Path .\sota\videoattentiontarget.pth
```

Each command should print `True`.

## 3. Train

Train on one GPU:

```powershell
python tools\train.py --config-file configs\hiphop_dataset_full_6gb.py --num-gpus 1
```

Logs and checkpoints are written to:

```text
output/hiphop_dataset_full_6gb/
```

The completed checkpoint is normally:

```text
output/hiphop_dataset_full_6gb/model_final.pth
```

Intermediate checkpoints are retained at roughly one-epoch intervals.

### Resume an Interrupted Run

The trainer records the latest checkpoint in
`output/hiphop_dataset_full_6gb/last_checkpoint`. Resume it with:

```powershell
python tools\train.py --config-file configs\hiphop_dataset_full_6gb.py --num-gpus 1 --resume
```

Without `--resume`, training initializes from
`sota/videoattentiontarget.pth`.

### Optional Memory Adjustments

If CUDA runs out of memory, override the per-GPU batch size:

```powershell
python tools\train.py --config-file configs\hiphop_dataset_full_6gb.py --num-gpus 1 dataloader.train.batch_size=1
```

Reducing the batch size changes the effective optimization setup. For a
permanent change, update `ins_per_iter`, `train.max_iter`, and checkpoint
periods together in the config.

## 4. Test

Evaluate the final checkpoint on the HIP-HOP full test split:

```powershell
python tools\eval_on_hiphop_dataset_full.py `
  --config-file configs\hiphop_dataset_full_6gb.py `
  --model-weights output\hiphop_dataset_full_6gb\model_final.pth `
  --dataset-root ..\hiphop_gaze
```

The evaluator reports:

```text
|AUC   |dist    |AP     |
```

- `AUC`: gaze heatmap area under the ROC curve; higher is better.
- `dist`: normalized L2 distance between predicted and target gaze; lower is
  better.
- `AP`: average precision for in-frame/out-of-frame gaze; higher is better.

To test an intermediate checkpoint, replace `model_final.pth` with the desired
`model_*.pth` file.

Optional evaluation flags:

```powershell
# Evaluate every fourth annotated frame.
python tools\eval_on_hiphop_dataset_full.py `
  --model-weights output\hiphop_dataset_full_6gb\model_final.pth `
  --dataset-root ..\hiphop_gaze `
  --skip-frame 4

# Use DARK heatmap coordinate refinement.
python tools\eval_on_hiphop_dataset_full.py `
  --model-weights output\hiphop_dataset_full_6gb\model_final.pth `
  --dataset-root ..\hiphop_gaze `
  --use-dark-inference
```

Use `--skip-frame 1` for the full test set and comparable results.
