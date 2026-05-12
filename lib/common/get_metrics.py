import fnmatch
import os
import pickle
from dataclasses import dataclass
from typing import Optional

import lib.data_utils.fs as fs
import numpy as np
from lib.common import metric_utils
from lib.data_utils import bundles

@dataclass
class Metrics:
    keypoint_errors: np.ndarray
    # pampjpe: np.ndarray
    keypoint_accelerations: np.ndarray
    gt_keypoint_accelerations: np.ndarray


def _compute_metrics(
        gt_keypoints: np.ndarray, tracked_keypoints: np.ndarray, valid_tracking: np.ndarray
) -> Metrics:
    def _compute_accelerations(pts: np.ndarray):
        acc = pts[:, 0:-2] + pts[:, 2:] - 2 * pts[:, 1:-1]
        return np.linalg.norm(acc, axis=-1).mean(axis=-1)

    diff_keypoints = gt_keypoints - tracked_keypoints
    keypoint_errors = np.linalg.norm(diff_keypoints, axis=-1).mean(axis=-1)
    # pampjpes = []
    # for i in range(2):
    #     tmp = []
    #     for j in range(gt_keypoints.shape[1]):
    #         pampjpe = procrustes_analysis_mean_per_joint_postition_error(gt_keypoints[i,j],tracked_keypoints[i,j])
    #         tmp.append([pampjpe])
    #     # print(tmp)
    #     pampjpes.append(tmp)

    # pampjpes = np.stack(pampjpes,axis=0)
    # print(pampjpes.shape)
    valid_accelerations = (
            valid_tracking[:, 0:-2] & valid_tracking[:, 1:-1] & valid_tracking[:, 2:]
    )
    keypoint_accelerations = _compute_accelerations(tracked_keypoints)
    gt_keypoint_accelerations = _compute_accelerations(gt_keypoints)

    return Metrics(
        keypoint_errors=keypoint_errors[valid_tracking],
        # pampjpe = pampjpes[valid_tracking],
        keypoint_accelerations=keypoint_accelerations[valid_accelerations],
        gt_keypoint_accelerations=gt_keypoint_accelerations[valid_accelerations],
    )

def get_metrics():
    valid_tracking_all = []
    metrics_all = []
    pass