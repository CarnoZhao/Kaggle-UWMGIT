# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch

from monai.metrics.utils import do_metric_reduction
from monai.metrics.utils import get_mask_edges, get_surface_distance

from monai.metrics import CumulativeIterationMetric


class HausdorffScore(CumulativeIterationMetric):
    """
    Modify MONAI's HausdorffDistanceMetric for Kaggle UW-Madison GI Tract Image Segmentation

    """

    def __init__(
        self,
        reduction = "mean",
    ) -> None:
        super().__init__()
        self.reduction = reduction

    def _compute_tensor(self, pred, gt):

        return compute_hausdorff_score(pred, gt)


    def aggregate(self):
        """
        Execute reduction logic for the output of `compute_hausdorff_distance`.

        """
        data = self.get_buffer()
        # do metric reduction
        f, _ = do_metric_reduction(data, self.reduction)
        return f

def compute_directed_hausdorff(pred, gt, max_dist):
    if np.all(pred == gt):
        return 0.0
    if np.sum(pred) == 0:
        return 1.0
    if np.sum(gt) == 0:
        return 1.0
    (edges_pred, edges_gt) = get_mask_edges(pred, gt)
    surface_distance = get_surface_distance(edges_pred, edges_gt, distance_metric="euclidean")
    if surface_distance.shape == (0,):
        return 0.0
    dist = surface_distance.max()

    if dist > max_dist:
        return 1.0
    return dist / max_dist

def compute_hausdorff_score(pred, gt):

    y = gt.float().to("cpu").numpy()
    y_pred = pred.float().to("cpu").numpy()

    # hausdorff distance score
    batch_size, n_class = y_pred.shape[:2]
    spatial_size = y_pred.shape[2:]
    max_dist = np.sqrt(np.sum([l**2 for l in spatial_size]))
    hd_score = np.empty((batch_size, n_class))
    for b, c in np.ndindex(batch_size, n_class):
        hd_score[b, c] = 1 - compute_directed_hausdorff(y_pred[b, c], y[b, c], max_dist)

    return torch.from_numpy(hd_score)
