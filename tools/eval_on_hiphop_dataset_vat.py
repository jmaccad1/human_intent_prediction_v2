import argparse
import sys
import warnings
from os import path as osp

import numpy as np
import torch
from detectron2.config import LazyConfig, instantiate
from PIL import Image

sys.path.append(osp.dirname(osp.dirname(__file__)))
from utils import *  # noqa: E402,F403


warnings.simplefilter(action="ignore", category=FutureWarning)


def _repo_path(*parts):
    return osp.join(osp.dirname(osp.dirname(__file__)), *parts)


def _load_model_weights(model, model_weights):
    checkpoint = torch.load(model_weights, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)


def _force_hiphop_test_paths(cfg, dataset_root):
    cfg.dataloader.val.train_root = osp.join(dataset_root, "images")
    cfg.dataloader.val.val_root = osp.join(dataset_root, "images")
    cfg.dataloader.val.train_anno = osp.join(dataset_root, "annotations", "test")
    cfg.dataloader.val.val_anno = osp.join(dataset_root, "annotations", "test")
    cfg.dataloader.val.head_root = osp.join(dataset_root, "head_masks", "images")
    cfg.dataloader.val.is_train = False
    cfg.dataloader.val.batch_size = 1
    cfg.dataloader.val.distributed = False
    cfg.dataloader.val.drop_last = False


def do_test(cfg, model, use_dark_inference=False):
    val_loader = instantiate(cfg.dataloader.val)

    model.train(False)
    auc_values = []
    dist_values = []
    inout_gt = []
    inout_pred = []

    with torch.no_grad():
        for data in val_loader:
            val_gaze_heatmap_pred, val_gaze_inout_pred = model(data)
            val_gaze_heatmap_pred = (
                val_gaze_heatmap_pred.squeeze(1).cpu().detach().numpy()
            )
            val_gaze_inout_pred = val_gaze_inout_pred.cpu().detach().numpy()

            for b_i in range(len(val_gaze_heatmap_pred)):
                if data["gaze_inouts"][b_i]:
                    valid_gaze = data["gazes"][b_i]
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

                    auc_values.append(auc(scaled_heatmap, multi_hot))
                    dist_values.append(L2_dist(valid_gaze.numpy(), norm_p))

            inout_gt.extend(data["gaze_inouts"].cpu().numpy().reshape(-1))
            inout_pred.extend(val_gaze_inout_pred.reshape(-1))

    if not auc_values:
        raise RuntimeError("No in-frame gaze samples were found in hiphop_gaze test.")

    print("|AUC   |dist    |AP     |")
    print(
        "|{:.4f}|{:.4f}  |{:.4f}  |".format(
            torch.mean(torch.tensor(auc_values)),
            torch.mean(torch.tensor(dist_values)),
            ap(inout_gt, inout_pred),
        )
    )


def main(args):
    cfg = LazyConfig.load(args.config_file)
    dataset_root = osp.abspath(args.dataset_root)
    _force_hiphop_test_paths(cfg, dataset_root)

    model: torch.nn.Module = instantiate(cfg.model)
    _load_model_weights(model, args.model_weights)
    model.to(cfg.train.device)

    do_test(cfg, model, use_dark_inference=args.use_dark_inference)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a VAT-style model on hiphop_gaze test annotations."
    )
    parser.add_argument(
        "--config_file",
        "--config-file",
        default=_repo_path("configs", "hiphop_dataset_vat_6gb.py"),
        help="HIPHOP VAT config file.",
    )
    parser.add_argument(
        "--model_weights",
        "--model-weights",
        required=True,
        help="Path to a checkpoint containing a model state dict.",
    )
    parser.add_argument(
        "--dataset_root",
        "--dataset-root",
        default=_repo_path("hiphop_gaze"),
        help="Root of the hiphop_gaze dataset.",
    )
    parser.add_argument("--use_dark_inference", action="store_true")
    args = parser.parse_args()
    main(args)
