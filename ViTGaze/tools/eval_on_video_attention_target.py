import sys
from os import path as osp
import argparse
import os
import warnings
import torch
import numpy as np
from PIL import Image

PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
DETECTRON2_ROOT = osp.join(PROJECT_ROOT, "src", "detectron2")

for import_root in (DETECTRON2_ROOT, PROJECT_ROOT):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from detectron2.config import instantiate, LazyConfig

from utils import *


warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="expandable_segments not supported")



def apply_eval_memory_overrides(cfg, args):
    cfg.dataloader.val.batch_size = args.batch_size
    cfg.dataloader.val.seq_len = args.seq_len
    cfg.dataloader.val.input_size = args.input_size
    cfg.dataloader.val.num_workers = args.num_workers
    cfg.dataloader.val.persistent_workers = args.num_workers > 0
    cfg.dataloader.val.distributed = False
    cfg.dataloader.val.drop_last = False
    cfg.dataloader.val.quant_labelmap = False
    cfg.train.device = args.device


def do_test(cfg, model, use_dark_inference=False):
    val_loader = instantiate(cfg.dataloader.val)

    model.train(False)
    AUC = []
    dist = []
    inout_gt = []
    inout_pred = []
    with torch.no_grad():
        for data in val_loader:
            val_gaze_heatmap_pred, val_gaze_inout_pred = model(data)
            val_gaze_heatmap_pred = (
                val_gaze_heatmap_pred.squeeze(1).cpu().detach().numpy()
            )
            val_gaze_inout_pred = val_gaze_inout_pred.cpu().detach().numpy()

            # go through each data point and record AUC, dist, ap
            for b_i in range(len(val_gaze_heatmap_pred)):
                auc_batch = []
                dist_batch = []
                if data["gaze_inouts"][b_i]:
                    # remove padding and recover valid ground truth points
                    valid_gaze = data["gazes"][b_i]
                    # AUC: area under curve of ROC
                    multi_hot = data["heatmaps"][b_i]
                    multi_hot = (multi_hot > 0).float().numpy()
                    if use_dark_inference:
                        pred_x, pred_y = dark_inference(val_gaze_heatmap_pred[b_i])
                    else:
                        pred_x, pred_y = argmax_pts(val_gaze_heatmap_pred[b_i])
                    norm_p = [
                        pred_x / val_gaze_heatmap_pred[b_i].shape[-1],
                        pred_y / val_gaze_heatmap_pred[b_i].shape[-2],
                    ]
                    scaled_heatmap = np.array(
                        Image.fromarray(val_gaze_heatmap_pred[b_i]).resize(
                            (64, 64),
                            resample=Image.Resampling.BILINEAR,
                        )
                    )
                    auc_score = auc(scaled_heatmap, multi_hot)
                    auc_batch.append(auc_score)
                    dist_batch.append(L2_dist(valid_gaze.numpy(), norm_p))
                AUC.extend(auc_batch)
                dist.extend(dist_batch)
            inout_gt.extend(data["gaze_inouts"].cpu().numpy())
            inout_pred.extend(val_gaze_inout_pred)

    print("|AUC   |dist    |AP     |")
    print(
        "|{:.4f}|{:.4f}  |{:.4f}  |".format(
            torch.mean(torch.tensor(AUC)),
            torch.mean(torch.tensor(dist)),
            ap(inout_gt, inout_pred),
        )
    )


def main(args):
    cfg = LazyConfig.load(args.config_file)
    apply_eval_memory_overrides(cfg, args)
    model: torch.Module = instantiate(cfg.model)
    checkpoint = torch.load(args.model_weights, map_location="cpu")
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.to(cfg.train.device)
    do_test(cfg, model, use_dark_inference=args.use_dark_inference)


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", type=str, help="config file")
    parser.add_argument(
        "-w",
        "--model_weights",
        type=str,
        help="model weights",
    )
    parser.add_argument("--use_dark_inference", action="store_true")
    parser.add_argument(
        "--input_size",
        type=int,
        default=322,
        help="Validation input size. 322 is safer for 6GB GPUs.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=2,
        help="Validation sequence length. 2 is safer for 6GB GPUs.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(args)
