from os import path as osp
from typing import Literal

from omegaconf import OmegaConf
from detectron2.config import LazyCall as L
from detectron2.config import instantiate
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data import *


DATA_ROOT = r"${Root to Datasets}"
if DATA_ROOT == "${Root to Datasets}":
    raise Exception(
        f"""{osp.abspath(__file__)}: Rewrite `DATA_ROOT` with the root to the datasets.
The directory structure should be:
-DATA_ROOT
|-videoattentiontarget
|--images
|--annotations
|---train
|---test
|--head_masks
|---images
|-gazefollow
|--train
|--test2
|--train_annotations_release.txt
|--test_annotations_release.txt
|--head_masks
|---train
|---test2
"""
    )

# Basic Config for Video Attention Target dataset and preprocessing
data_info = OmegaConf.create()
data_info.video_attention_target = OmegaConf.create()
data_info.video_attention_target.train_root = osp.join(
    DATA_ROOT, "videoattentiontarget/images"
)
data_info.video_attention_target.train_anno = osp.join(
    DATA_ROOT, "videoattentiontarget/annotations/train"
)
data_info.video_attention_target.val_root = osp.join(
    DATA_ROOT, "videoattentiontarget/images"
)
data_info.video_attention_target.val_anno = osp.join(
    DATA_ROOT, "videoattentiontarget/annotations/test"
)
data_info.video_attention_target.head_root = osp.join(
    DATA_ROOT, "videoattentiontarget/head_masks/images"
)

data_info.video_attention_target_video = OmegaConf.create()
data_info.video_attention_target_video.train_root = osp.join(
    DATA_ROOT, "videoattentiontarget/images"
)
data_info.video_attention_target_video.train_anno = osp.join(
    DATA_ROOT, "videoattentiontarget/annotations/train"
)
data_info.video_attention_target_video.val_root = osp.join(
    DATA_ROOT, "videoattentiontarget/images"
)
data_info.video_attention_target_video.val_anno = osp.join(
    DATA_ROOT, "videoattentiontarget/annotations/test"
)
data_info.video_attention_target_video.head_root = osp.join(
    DATA_ROOT, "videoattentiontarget/head_masks/images"
)

data_info.hiphop_dataset_vat = OmegaConf.create()
data_info.hiphop_dataset_vat.train_root = osp.join(DATA_ROOT, "hiphop_gaze/images")
data_info.hiphop_dataset_vat.train_anno = osp.join(
    DATA_ROOT, "hiphop_gaze/annotations/train"
)
data_info.hiphop_dataset_vat.val_root = osp.join(DATA_ROOT, "hiphop_gaze/images")
data_info.hiphop_dataset_vat.val_anno = osp.join(
    DATA_ROOT, "hiphop_gaze/annotations/test"
)
data_info.hiphop_dataset_vat.head_root = osp.join(
    DATA_ROOT, "hiphop_gaze/head_masks/images"
)

data_info.hiphop_dataset_full = OmegaConf.create()
data_info.hiphop_dataset_full.train_root = osp.join(DATA_ROOT, "hiphop_gaze/images")
data_info.hiphop_dataset_full.train_anno = osp.join(
    DATA_ROOT, "hiphop_gaze/annotations/train"
)
data_info.hiphop_dataset_full.val_root = osp.join(DATA_ROOT, "hiphop_gaze/images")
data_info.hiphop_dataset_full.val_anno = osp.join(
    DATA_ROOT, "hiphop_gaze/annotations/test"
)
data_info.hiphop_dataset_full.head_root = osp.join(
    DATA_ROOT, "hiphop_gaze/head_masks/images"
)

data_info.hiphop_dataset_prev = OmegaConf.create()
data_info.hiphop_dataset_prev.train_root = osp.join(DATA_ROOT, "hiphop_gaze/images")
data_info.hiphop_dataset_prev.train_anno = osp.join(
    DATA_ROOT, "split/data/gaze_dataset.json"
)
data_info.hiphop_dataset_prev.val_root = osp.join(DATA_ROOT, "hiphop_gaze/images")
data_info.hiphop_dataset_prev.val_anno = osp.join(
    DATA_ROOT, "split/data/gaze_dataset.json"
)
data_info.hiphop_dataset_prev.head_root = osp.join(
    DATA_ROOT, "hiphop_gaze/head_masks/images"
)
data_info.hiphop_dataset_prev.dataset_root = DATA_ROOT
data_info.hiphop_dataset_prev.split_file = osp.join(DATA_ROOT, "split/data/gaze_dataset.json")
data_info.hiphop_dataset_prev.annotation_root = osp.join(
    DATA_ROOT, "hiphop_gaze/annotations"
)

data_info.gazefollow = OmegaConf.create()
data_info.gazefollow.train_root = osp.join(DATA_ROOT, "gazefollow")
data_info.gazefollow.train_anno = osp.join(
    DATA_ROOT, "gazefollow/train_annotations_release.txt"
)
data_info.gazefollow.val_root = osp.join(DATA_ROOT, "gazefollow")
data_info.gazefollow.val_anno = osp.join(
    DATA_ROOT, "gazefollow/test_annotations_release.txt"
)
data_info.gazefollow.head_root = osp.join(DATA_ROOT, "gazefollow/head_masks")

data_info.input_size = 224
data_info.output_size = 64
data_info.quant_labelmap = True
data_info.mean = (0.485, 0.456, 0.406)
data_info.std = (0.229, 0.224, 0.225)
data_info.bbox_jitter = 0.5
data_info.rand_crop = 0.5
data_info.rand_flip = 0.5
data_info.color_jitter = 0.5
data_info.rand_rotate = 0.0
data_info.rand_lsj = 0.0

data_info.mask_size = 24
data_info.mask_scene = False
data_info.mask_head = False
data_info.max_scene_patches_ratio = 0.5
data_info.max_head_patches_ratio = 0.3
data_info.mask_prob = 0.2

data_info.seq_len = 16
data_info.max_len = 32
data_info.skip_frame = 1


# Dataloader(gazefollow/video_atention_target, train/val)
def __build_dataloader(
    name: Literal[
        "gazefollow",
        "video_attention_target",
        "video_attention_target_video",
        "hiphop_dataset_vat",
        "hiphop_dataset_full",
        "hiphop_dataset_prev",
    ],
    is_train: bool,
    batch_size: int = 64,
    num_workers: int = 14,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    drop_last: bool = True,
    distributed: bool = False,
    **kwargs,
):
    assert name in [
        "gazefollow",
        "video_attention_target",
        "video_attention_target_video",
        "hiphop_dataset_vat",
        "hiphop_dataset_full",
        "hiphop_dataset_prev",
    ], f'{name} not in ("gazefollow", "video_attention_target", "video_attention_target_video", "hiphop_dataset_vat", "hiphop_dataset_full", "hiphop_dataset_prev")'

    for k, v in kwargs.items():
        if k in [
            "train_root",
            "train_anno",
            "val_root",
            "val_anno",
            "head_root",
            "dataset_root",
            "split_file",
            "annotation_root",
        ]:
            data_info[name][k] = v
        else:
            data_info[k] = v

    datasets = {
        "gazefollow": GazeFollow,
        "video_attention_target": VideoAttentionTarget,
        "video_attention_target_video": VideoAttentionTargetVideo,
        "hiphop_dataset_vat": HipHopDatasetVAT,
        "hiphop_dataset_full": HipHopDatasetFull,
        "hiphop_dataset_prev": HipHopDatasetPrev,
    }
    dataset = L(datasets[name])(
        image_root=data_info[name]["train_root" if is_train else "val_root"],
        anno_root=data_info[name]["train_anno" if is_train else "val_anno"],
        head_root=data_info[name]["head_root"],
        transform=get_transform(
            input_resolution=data_info.input_size,
            mean=data_info.mean,
            std=data_info.std,
        ),
        input_size=data_info.input_size,
        output_size=data_info.output_size,
        quant_labelmap=data_info.quant_labelmap,
        is_train=is_train,
        bbox_jitter=data_info.bbox_jitter,
        rand_crop=data_info.rand_crop,
        rand_flip=data_info.rand_flip,
        color_jitter=data_info.color_jitter,
        rand_rotate=data_info.rand_rotate,
        rand_lsj=data_info.rand_lsj,
        mask_generator=(
            MaskGenerator(
                input_size=data_info.mask_size,
                mask_scene=data_info.mask_scene,
                mask_head=data_info.mask_head,
                max_scene_patches_ratio=data_info.max_scene_patches_ratio,
                max_head_patches_ratio=data_info.max_head_patches_ratio,
                mask_prob=data_info.mask_prob,
            )
            if is_train
            else None
        ),
    )
    video_dataset_names = [
        "video_attention_target_video",
        "hiphop_dataset_vat",
        "hiphop_dataset_full",
        "hiphop_dataset_prev",
    ]
    if name in video_dataset_names:
        dataset.seq_len = data_info.seq_len
        dataset.max_len = data_info.max_len
        dataset.skip_frame = data_info.skip_frame
    if name == "hiphop_dataset_prev":
        dataset.dataset_root = data_info.hiphop_dataset_prev.dataset_root
        dataset.split_file = data_info.hiphop_dataset_prev.split_file
        dataset.annotation_root = data_info.hiphop_dataset_prev.annotation_root
    dataset = instantiate(dataset)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=video_collate if name in video_dataset_names else None,
        sampler=DistributedSampler(dataset, shuffle=is_train) if distributed else None,
        drop_last=drop_last,
    )


dataloader = OmegaConf.create()
dataloader.gazefollow = OmegaConf.create()
dataloader.gazefollow.train = L(__build_dataloader)(
    name="gazefollow",
    is_train=True,
)
dataloader.gazefollow.val = L(__build_dataloader)(
    name="gazefollow",
    is_train=False,
)
dataloader.video_attention_target = OmegaConf.create()
dataloader.video_attention_target.train = L(__build_dataloader)(
    name="video_attention_target",
    is_train=True,
)
dataloader.video_attention_target.val = L(__build_dataloader)(
    name="video_attention_target",
    is_train=False,
)
dataloader.video_attention_target_video = OmegaConf.create()
dataloader.video_attention_target_video.train = L(__build_dataloader)(
    name="video_attention_target_video",
    is_train=True,
)
dataloader.video_attention_target_video.val = L(__build_dataloader)(
    name="video_attention_target_video",
    is_train=False,
)
dataloader.hiphop_dataset_vat = OmegaConf.create()
dataloader.hiphop_dataset_vat.train = L(__build_dataloader)(
    name="hiphop_dataset_vat",
    is_train=True,
)
dataloader.hiphop_dataset_vat.val = L(__build_dataloader)(
    name="hiphop_dataset_vat",
    is_train=False,
)
dataloader.hiphop_dataset_full = OmegaConf.create()
dataloader.hiphop_dataset_full.train = L(__build_dataloader)(
    name="hiphop_dataset_full",
    is_train=True,
)
dataloader.hiphop_dataset_full.val = L(__build_dataloader)(
    name="hiphop_dataset_full",
    is_train=False,
)
dataloader.hiphop_dataset_prev = OmegaConf.create()
dataloader.hiphop_dataset_prev.train = L(__build_dataloader)(
    name="hiphop_dataset_prev",
    is_train=True,
)
dataloader.hiphop_dataset_prev.val = L(__build_dataloader)(
    name="hiphop_dataset_prev",
    is_train=False,
)
