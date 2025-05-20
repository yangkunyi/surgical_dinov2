# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple
import os
import torch
from PIL import Image
from torchvision.datasets import VisionDataset
import torchvision.transforms as T
import numpy as np
import cv2

logger = logging.getLogger("dinov2")


class HemaStandardDataset(VisionDataset):
    def __init__(
        self,
        *,
        root: str = "",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        shuffle: bool = False,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.patches = []

        all_dataset_files = Path(root).glob("cholec.txt")

        for dataset_file in all_dataset_files:
            print("Loading ", dataset_file)
            with open(dataset_file, "r") as file:
                content = file.read()
            file_list = content.splitlines()
            self.patches.extend(file_list)
        self.true_len = len(self.patches)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            image, filepath = self.get_image_data(index)
            raw_image = image
        except Exception as e:
            adjusted_index = index % self.true_len
            filepath = self.patches[adjusted_index]
            print(f"can not read image for sample {index, e, filepath}")
            return self.__getitem__(index + 1)

        try:
            depth, depthpath = self.get_depth_data(index)
        except Exception as e:
            adjusted_index = index % self.true_len
            filepath = self.patches[adjusted_index]
            print(f"can not read depth for sample {index, e, filepath}")
            return self.__getitem__(index + 1)

        mask = self.get_seg_data(index)
        target = self.get_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return (
            image,
            target,
            filepath,
            T.ToTensor()(raw_image),
            T.ToTensor()(depth),
            torch.as_tensor(np.array(mask), dtype=torch.int8).unsqueeze(0),
        )

    def get_image_data(self, index: int, dimension=224) -> Image:
        # Load image from jpeg file
        adjusted_index = index % self.true_len
        filepath = self.patches[adjusted_index]
        patch = (
            Image.open(filepath)
            .convert(mode="RGB")
            .resize((dimension, dimension), Image.Resampling.LANCZOS)
        )
        return patch, filepath

    def get_depth_data(
        self,
        index: int,
        dimension=224,
        depth_base_path="/bd_byta6000i0/users/surgicaldinov2/kyyang/Depth-Anything-V2/out",
    ) -> Image:
        adjusted_index = index % self.true_len
        filepath = self.patches[adjusted_index]
        splited_path = filepath.split("/")
        depth_file_name = splited_path[-2] + "_" + splited_path[-1]
        depthpath = os.path.join(depth_base_path, depth_file_name)
        depth = (
            Image.open(depthpath)
            .convert(mode="L")
            .resize((dimension, dimension), Image.Resampling.LANCZOS)
        )
        return depth, depthpath

    def get_seg_data(
        self,
        index: int,
        dimension=224,
        seg_base_path="/bd_byta6000i0/users/surgicaldinov2/kyyang/segmentation_on_cholec80/masks",
    ) -> Image:
        adjusted_index = index % self.true_len
        filepath = self.patches[adjusted_index]
        splited_path = filepath.split("/")
        seg_file_name = splited_path[-2] + "_" + splited_path[-1]
        maskpath = os.path.join(seg_base_path, seg_file_name)
        mask = (
            Image.open(maskpath)
            .convert(mode="L")
            .resize(
                (dimension, dimension),
                Image.Resampling.NEAREST,
            )
        )
        return mask

    def get_target(self, index: int) -> torch.Tensor:
        # labels are not used for training
        return torch.zeros((1,))

    def __len__(self) -> int:
        return 120000000  # large number for infinite data sampling
