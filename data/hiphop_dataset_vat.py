from os import path as osp
from typing import Callable, Optional

from .masking import MaskGenerator
from .video_attention_target_video import VideoAttentionTargetVideo, video_collate


class HipHopDatasetVAT(VideoAttentionTargetVideo):
    """HIPHOP loader for data laid out like VideoAttentionTarget.

    Expected default structure:
        hiphop_gaze/
            images/P1/V1/frame_0000.jpg
            annotations/train/P1/V1/P1_V1.txt
            annotations/test/P17/V1/P17_V1.txt

    Annotation rows must use the VAT columns:
        frame_name, head_x1, head_y1, head_x2, head_y2, gaze_x, gaze_y
    """

    def __init__(
        self,
        image_root: Optional[str] = None,
        anno_root: Optional[str] = None,
        head_root: Optional[str] = None,
        transform: Optional[Callable] = None,
        input_size: int = 224,
        output_size: int = 64,
        quant_labelmap: bool = True,
        is_train: bool = True,
        seq_len: int = 8,
        max_len: int = 32,
        *,
        dataset_root: Optional[str] = None,
        split: Optional[str] = None,
        mask_generator: Optional[MaskGenerator] = None,
        bbox_jitter: float = 0.5,
        rand_crop: float = 0.5,
        rand_flip: float = 0.5,
        color_jitter: float = 0.5,
        rand_rotate: float = 0.0,
        rand_lsj: float = 0.0,
    ):
        if dataset_root is None:
            dataset_root = osp.abspath(
                osp.join(osp.dirname(__file__), osp.pardir, "hiphop_gaze")
            )
        if split is None:
            split = "train" if is_train else "test"

        image_root = image_root or osp.join(dataset_root, "images")
        anno_root = anno_root or osp.join(dataset_root, "annotations", split)
        head_root = head_root or osp.join(dataset_root, "head_masks", "images")

        super().__init__(
            image_root=image_root,
            anno_root=anno_root,
            head_root=head_root,
            transform=transform,
            input_size=input_size,
            output_size=output_size,
            quant_labelmap=quant_labelmap,
            is_train=is_train,
            seq_len=seq_len,
            max_len=max_len,
            mask_generator=mask_generator,
            bbox_jitter=bbox_jitter,
            rand_crop=rand_crop,
            rand_flip=rand_flip,
            color_jitter=color_jitter,
            rand_rotate=rand_rotate,
            rand_lsj=rand_lsj,
        )


# Backwards-compatible alias for older imports.
GOMDataset = HipHopDatasetVAT

