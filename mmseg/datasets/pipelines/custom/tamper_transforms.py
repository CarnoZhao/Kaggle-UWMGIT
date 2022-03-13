import copy
import os
import inspect
from collections import defaultdict

import cv2
import mmcv
import numpy as np
from copy import deepcopy
from mmcv.utils import deprecated_api_warning, is_tuple_of
from numpy import random

import traceback

from torch import matrix_power

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None

from ...builder import PIPELINES

@PIPELINES.register_module()
class RandomCopyMove(object):
    def __init__(self, prob=0., mix_prob=0., save_buffer=16, crop_ratio=(0.05, 0.15)):
        self.prob = prob
        self.mix_prob = mix_prob
        self.save_buffer = save_buffer
        self.saves = [None for _ in range(save_buffer)]
        self.crop_ratio = crop_ratio

    def get_crop_bbox(self, img, crop_size = None):
        """Randomly get a crop bounding box."""
        if crop_size is None:
            crop_size = (
                int(img.shape[0] * (np.random.rand() * (self.crop_ratio[1] - self.crop_ratio[0]) + self.crop_ratio[0])),
                int(img.shape[1] * (np.random.rand() * (self.crop_ratio[1] - self.crop_ratio[0]) + self.crop_ratio[0]))
            )
        margin_h = max(img.shape[0] - crop_size[0], 0)
        margin_w = max(img.shape[1] - crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

        return (crop_y1, crop_y2, crop_x1, crop_x2), crop_size

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into
                result dict.
        """
        self.save_results(results)
        if np.random.rand() < self.prob:
            if np.random.rand() < self.mix_prob:
                mix_idx = np.random.randint(0, len(self.saves))
                results = self.copy_move(results, self.saves[mix_idx])
            else:
                results = self.copy_move(results, results)
        return results

    def copy_move(self, results1, results2):
        results2 = results1 if results2 is None else results2
        img1 = results1['img']
        seg1 = results1['gt_semantic_seg']
        crop_bbox1, crop_size = self.get_crop_bbox(img1)
        crop_y11, crop_y21, crop_x11, crop_x21 = crop_bbox1
        img2 = results2['img']
        crop_bbox2, _ = self.get_crop_bbox(img2, crop_size)
        crop_y12, crop_y22, crop_x12, crop_x22 = crop_bbox2
        img1[crop_y11:crop_y21, crop_x11:crop_x21,...] = img2[crop_y12:crop_y22, crop_x12:crop_x22,...]
        seg1[crop_y11:crop_y21, crop_x11:crop_x21,...] = 1
        results1['img'] = img1
        results1['gt_semantic_seg'] = seg1
        return results1

    def save_results(self, results):
        mix_idx = np.random.randint(0, len(self.saves))
        if np.random.rand() < 0.5:
            self.saves[mix_idx] = deepcopy(results)

    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob}, mix_prob={self.mix_prob}, save_buffer={self.save_buffer})'
