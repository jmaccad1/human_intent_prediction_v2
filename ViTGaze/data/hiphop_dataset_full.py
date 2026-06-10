from os import path as osp
from typing import Callable, Optional

from .hiphop_dataset_vat import HipHopDatasetVAT
from .masking import MaskGenerator


class HipHopDatasetFull(HipHopDatasetVAT):
    """HIPHOP loader for the combined P1-P40 gaze dataset.

    Split:
        train: P1-P16, P21-P36
        test:  P17-P20, P37-P40

    Expected default structure:
        hiphop_gaze/
            images/P1/V1/frame_0000.jpg
            annotations/train/P1/V1/P1_V1.txt
            annotations/test/P17/V1/P17_V1.txt

    Annotation rows use the VAT columns:
        frame_name, head_x1, head_y1, head_x2, head_y2, gaze_x, gaze_y
    """

    TRAIN_PARTICIPANTS = tuple(
        [f"P{i}" for i in range(1, 17)] + [f"P{i}" for i in range(21, 37)]
    )
    TEST_PARTICIPANTS = tuple(
        [f"P{i}" for i in range(17, 21)] + [f"P{i}" for i in range(37, 41)]
    )

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
        skip_frame: int = 1,
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
        if osp.basename(anno_root) == "annotations":
            anno_root = osp.join(anno_root, split)
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
            skip_frame=skip_frame,
            dataset_root=dataset_root,
            split=split,
            mask_generator=mask_generator,
            bbox_jitter=bbox_jitter,
            rand_crop=rand_crop,
            rand_flip=rand_flip,
            color_jitter=color_jitter,
            rand_rotate=rand_rotate,
            rand_lsj=rand_lsj,
        )

        allowed_participants = (
            self.TRAIN_PARTICIPANTS if is_train else self.TEST_PARTICIPANTS
        )
        allowed_participants = set(allowed_participants)

        def participant_name(df):
            if len(df.index) == 0:
                return ""
            return str(df["path"].iloc[0]).replace("\\", "/").split("/")[0]

        self.dfs = [
            df for df in self.dfs if participant_name(df) in allowed_participants
        ]
        self.length = len(self.dfs)
