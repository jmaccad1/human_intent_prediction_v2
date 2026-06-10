import json
from os import path as osp
from typing import Callable, Optional

import pandas as pd

from .masking import MaskGenerator
from .video_attention_target_video import VideoAttentionTargetVideo
from . import data_utils as utils


class HipHopDatasetPrev(VideoAttentionTargetVideo):
    """HIPHOP loader that follows split/data/gaze_dataset.json exactly.

    The JSON file stores the sampled train/test videos. Per-frame gaze
    coordinates are still read from the VAT-style annotation txt files.
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
        skip_frame: int = 1,
        *,
        dataset_root: Optional[str] = None,
        split: Optional[str] = None,
        split_file: Optional[str] = None,
        annotation_root: Optional[str] = None,
        mask_generator: Optional[MaskGenerator] = None,
        bbox_jitter: float = 0.5,
        rand_crop: float = 0.5,
        rand_flip: float = 0.5,
        color_jitter: float = 0.5,
        rand_rotate: float = 0.0,
        rand_lsj: float = 0.0,
    ):
        if dataset_root is None:
            dataset_root = osp.abspath(osp.join(osp.dirname(__file__), osp.pardir))
        if split is None:
            split = "train" if is_train else "test"

        image_root = image_root or osp.join(dataset_root, "hiphop_gaze", "images")
        head_root = head_root or osp.join(
            dataset_root, "hiphop_gaze", "head_masks", "images"
        )
        if split_file is None:
            split_file = (
                anno_root
                if anno_root and str(anno_root).lower().endswith(".json")
                else osp.join(dataset_root, "split", "data", "gaze_dataset.json")
            )
        annotation_root = annotation_root or osp.join(
            dataset_root, "hiphop_gaze", "annotations"
        )

        skip_frame = max(1, int(skip_frame))
        dfs = []
        for sample in self._load_split_samples(split_file, split):
            df = self._read_video_annotation(sample, annotation_root)
            if skip_frame > 1:
                df = df.iloc[::skip_frame].copy()
            cur_len = len(df.index)
            if is_train:
                if cur_len <= max_len:
                    if cur_len >= seq_len:
                        dfs.append(df)
                    continue
                remainder = cur_len % max_len
                df_splits = [
                    df[i : i + max_len]
                    for i in range(0, cur_len - max_len + 1, max_len)
                ]
                if remainder >= seq_len:
                    df_splits.append(df[-remainder:])
                dfs.extend(df_splits)
            else:
                if cur_len < seq_len:
                    continue
                dfs.extend(
                    [df[i : i + seq_len] for i in range(0, cur_len - seq_len, seq_len)]
                )

        for df in dfs:
            df.reset_index(inplace=True)
        self.dfs = dfs
        self.length = len(dfs)

        self.data_dir = image_root
        self.head_dir = head_root
        self.transform = transform
        self.draw_labelmap = (
            utils.draw_labelmap if quant_labelmap else utils.draw_labelmap_no_quant
        )
        self.is_train = is_train

        self.input_size = input_size
        self.output_size = output_size
        self.seq_len = seq_len
        self.skip_frame = skip_frame

        if self.is_train:
            self.bbox_jitter = bbox_jitter
            self.rand_crop = rand_crop
            self.rand_flip = rand_flip
            self.color_jitter = color_jitter
            self.rand_rotate = rand_rotate
            self.rand_lsj = rand_lsj
            self.mask_generator = mask_generator

    @staticmethod
    def _load_split_samples(split_file, split):
        with open(split_file, encoding="utf-8") as file:
            dataset = json.load(file)
        return dataset["hiphop"]["gaze"][split]

    @staticmethod
    def _video_parts(sample):
        parts = sample["video"].replace("\\", "/").rstrip("/").split("/")
        return parts[-2], parts[-1]

    @classmethod
    def _read_video_annotation(cls, sample, annotation_root):
        participant, video = cls._video_parts(sample)
        annotation_name = f"{participant}_{video}.txt"
        candidates = [
            osp.join(annotation_root, split, participant, video, annotation_name)
            for split in ("train", "test")
        ]
        annotation_path = next((path for path in candidates if osp.exists(path)), None)
        if annotation_path is None:
            raise FileNotFoundError(
                f"Missing annotation for {participant}/{video} under {annotation_root}"
            )

        df = pd.read_csv(
            annotation_path,
            header=None,
            index_col=False,
            names=["path", "x_min", "y_min", "x_max", "y_max", "gaze_x", "gaze_y"],
        )
        expected_len = len(sample.get("gaze_seq", []))
        if expected_len and len(df.index) != expected_len:
            raise ValueError(
                f"{annotation_path} has {len(df.index)} rows, "
                f"but split JSON has {expected_len} frames"
            )
        df["path"] = df["path"].apply(lambda path: osp.join(participant, video, path))
        df["gaze_label"] = sample.get("gaze_seq", [])
        df["intent"] = sample.get("intent", "")
        return df
