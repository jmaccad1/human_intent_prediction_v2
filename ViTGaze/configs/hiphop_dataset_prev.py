import json
from os import environ
from os import path as osp

from torch.cuda import device_count

from .common.dataloader import dataloader
from .common.model import model
from .common.optimizer import optimizer
from .common.scheduler import lr_multiplier
from .common.train import train


def _count_train_clips(split_file, seq_len, max_len, skip_frame):
    with open(split_file, encoding="utf-8") as file:
        samples = json.load(file)["hiphop"]["gaze"]["train"]

    clip_count = 0
    for sample in samples:
        seq_length = len(sample["gaze_seq"])
        seq_length = (seq_length + skip_frame - 1) // skip_frame

        if seq_length <= max_len:
            if seq_length >= seq_len:
                clip_count += 1
            continue

        remainder = seq_length % max_len
        clip_count += len(range(0, seq_length - max_len + 1, max_len))
        if remainder >= seq_len:
            clip_count += 1

    return clip_count


repo_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
split_file = osp.join(repo_root, "split", "data", "gaze_dataset.json")
num_gpu = device_count()
ins_per_iter = 2
seq_len = 4
max_len = 16
skip_frame = int(environ.get("SKIP_FRAME", "4"))
len_dataset = _count_train_clips(split_file, seq_len, max_len, skip_frame)
num_epoch = 5

# dataloader
dataloader = dataloader.hiphop_dataset_prev
dataloader.train.batch_size = max(1, ins_per_iter // max(1, num_gpu))
dataloader.train.num_workers = dataloader.val.num_workers = 8
dataloader.train.persistent_workers = dataloader.val.persistent_workers = False
dataloader.train.distributed = num_gpu > 1
dataloader.train.drop_last = True
dataloader.train.rand_rotate = 0.5
dataloader.train.rand_lsj = 0.5
dataloader.train.input_size = dataloader.val.input_size = 434
dataloader.train.mask_scene = True
dataloader.train.mask_head = True
dataloader.train.mask_prob = 0.3
dataloader.train.mask_size = dataloader.train.input_size // 14
dataloader.train.seq_len = seq_len
dataloader.train.max_len = max_len
dataloader.train.skip_frame = skip_frame

dataloader.val.quant_labelmap = False
dataloader.val.seq_len = seq_len
dataloader.val.max_len = max_len
dataloader.val.skip_frame = skip_frame
dataloader.val.batch_size = 1
dataloader.val.distributed = False
dataloader.val.drop_last = False

# train
train.init_checkpoint = "sota/videoattentiontarget.pth"
train.output_dir = osp.join("./output", osp.basename(__file__).split(".")[0])
train.max_iter = max(1, len_dataset * num_epoch // ins_per_iter)
train.log_period = max(1, train.max_iter // 100)
train.checkpointer.max_to_keep = 20
train.checkpointer.period = max(1, len_dataset // ins_per_iter)
train.seed = 0

# optimizer
optimizer.lr = 3e-6
optimizer.params.lr_factor_func.backbone_multiplier = 0.1
lr_multiplier.scheduler.values = [1.0, 0.3, 0.1]
lr_multiplier.scheduler.milestones = [
    train.max_iter * 3 // 5,
    train.max_iter * 4 // 5,
]
lr_multiplier.scheduler.num_updates = train.max_iter
lr_multiplier.warmup_length = 0

# model
model.use_aux_loss = model.pam.use_aux_loss = model.criterion.use_aux_loss = True
model.pam.name = "PatchPAM"
model.pam.embed_dim = 8
model.pam.patch_size = 14
model.backbone.name = "dinov2_small"
model.backbone.return_softmax_attn = True
model.backbone.out_attn = [2, 5, 8, 11]
model.backbone.use_cls_token = True
model.backbone.use_mask_token = True
model.regressor.name = "UpSampleConv"
model.regressor.in_channel = 24
model.regressor.use_conv = False
model.regressor.dim = 24
model.regressor.deconv_cfgs = [
    dict(
        in_channels=24,
        out_channels=16,
        kernel_size=3,
        stride=1,
        padding=1,
    ),
    dict(
        in_channels=16,
        out_channels=8,
        kernel_size=3,
        stride=1,
        padding=1,
    ),
    dict(
        in_channels=8,
        out_channels=1,
        kernel_size=3,
        stride=1,
        padding=1,
    ),
]
model.regressor.feat_type = "attn"
model.classifier.name = "SimpleMlp"
model.classifier.in_channel = 384
model.criterion.aux_weight = 0
model.criterion.aux_head_thres = 0.05
model.criterion.use_focal_loss = True
model.device = "cuda"
