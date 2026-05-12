# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple

import lib.common.camera as camera
import numpy as np
import torch
from lib.common.crop import gen_crop_parameters_from_points, gen_crop_parameters_from_box
from lib.common.hand import HandModel, NUM_JOINTS_PER_HAND, RIGHT_HAND_INDEX
from lib.common.hand_skinning import skin_landmarks

from .tracking_result import SingleHandPose

from lib.common.tools import point_vis
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def neutral_joint_angles(up: HandModel, lower_factor: float = 0.5) -> torch.Tensor:
    joint_limits = up.joint_limits
    assert joint_limits is not None
    return joint_limits[..., 0] * lower_factor + joint_limits[..., 1] * (
        1 - lower_factor
    )


def skin_landmarks_np(
    hand_model: HandModel,
    joint_angles: np.ndarray,
    wrist_transforms: np.ndarray,
) -> np.ndarray:
    # skin_landmarks:
    # input: hand_model   joint_angles(22)     wrist_transforms(4 4)
    landmarks = skin_landmarks(
        hand_model,
        torch.from_numpy(joint_angles).float(),
        torch.from_numpy(wrist_transforms).float(),
    ) #output: 21 3 变换完的landmark世界坐标 将rest的坐标变换到当前pose下的坐标
    return landmarks.numpy()


def landmarks_from_hand_pose(
    hand_model: HandModel, hand_pose: SingleHandPose, hand_idx: int
) -> np.ndarray:
    """
    # input: 手模 当前pose(joint旋转 + wrist变换矩阵)
    # output: landmark世界坐标 21*3
    Compute 3D landmarks in the world space given the hand model and hand pose.
    """
    xf = hand_pose.wrist_xform.copy()
    # print(xf)
    # This function expects the user hand model to be a left hand.
    if hand_idx == RIGHT_HAND_INDEX:
        xf[:, 0] *= -1
    landmarks = skin_landmarks_np(hand_model, hand_pose.joint_angles, xf) #根据当前的位姿（joint旋转）算landmark
    return landmarks


def rank_hand_visibility_in_cameras(
    cameras: List[camera.CameraModel],
    hand_model: HandModel,
    hand_pose: SingleHandPose,
    hand_idx: int,
    min_required_vis_landmarks: int,
) -> List[int]:
    #此处用了pose的ground truth，是否需要换成检测网络
    landmarks_world = landmarks_from_hand_pose(hand_model, hand_pose, hand_idx)
    n_landmarks_in_view = []
    ranked_cam_indices = []
    for cam_idx, camera in enumerate(cameras):
        landmarks_eye = camera.world_to_eye(landmarks_world)
        landmarks_win2 = camera.eye_to_window(landmarks_eye)

        n_visible = (
            (landmarks_win2[..., 0] >= 0)
            & (landmarks_win2[..., 0] <= camera.width - 1)
            & (landmarks_win2[..., 1] >= 0)
            & (landmarks_win2[..., 1] <= camera.height - 1)
            & (landmarks_eye[..., 2] > 0)
        ).sum()

        n_landmarks_in_view.append(n_visible)
        # Only push the cameras that can see enough hand points
        if n_visible >= min_required_vis_landmarks:
            ranked_cam_indices.append(cam_idx)

    #  Favor the view that sees more landmarks
    ranked_cam_indices.sort(
        reverse=True,
        key=lambda x: n_landmarks_in_view[x],
    )
    return ranked_cam_indices

# 算 当前pose、紧握、rest 三个状态下的landmark坐标，并拼接到一起
def _get_crop_points_from_hand_pose(
    hand_model: HandModel,
    gt_hand_pose: SingleHandPose,
    hand_idx: int,
    num_crop_points: int,
) -> np.ndarray:
    assert num_crop_points in [21, 42, 63]
    neutral_hand_pose = SingleHandPose(
        joint_angles=neutral_joint_angles(hand_model).numpy(),
        wrist_xform=gt_hand_pose.wrist_xform,
    )
    open_hand_pose = SingleHandPose(
        joint_angles=np.zeros(NUM_JOINTS_PER_HAND, dtype=np.float32),
        wrist_xform=gt_hand_pose.wrist_xform,
    )

    crop_points = []
    crop_points.append(landmarks_from_hand_pose(hand_model, gt_hand_pose, hand_idx))
    if num_crop_points > 21:
        crop_points.append(
            landmarks_from_hand_pose(hand_model, neutral_hand_pose, hand_idx)
        )
    if num_crop_points > 42:
        crop_points.append(
            landmarks_from_hand_pose(hand_model, open_hand_pose, hand_idx)
        )

    # 画landmark
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # point_vis(crop_points[0], fig, ax, point_type="landmark")
    # point_vis(crop_points[1], fig, ax, point_type="landmark")
    # point_vis(crop_points[2], fig, ax, point_type="landmark")

    return np.concatenate(crop_points, axis=0)


def gen_crop_cameras_from_pose(
    cameras: List[camera.CameraModel],
    camera_angles: List[float],
    hand_model: HandModel,
    hand_pose: SingleHandPose,
    hand_idx: int,
    num_crop_points: int,
    new_image_size: Tuple[int, int],
    max_view_num: Optional[int] = None,
    sort_camera_index: bool = False,
    focal_multiplier: float = 0.95,
    mirror_right_hand: bool = True,
    min_required_vis_landmarks: int = 19,
) -> Dict[int, camera.PinholePlaneCameraModel]:
    crop_cameras: Dict[int, camera.PinholePlaneCameraModel] = {}

    # 算 当前pose、紧握、rest 三个状态下的landmark坐标，并拼接到一起
    crop_points = _get_crop_points_from_hand_pose(
        hand_model,
        hand_pose,
        hand_idx,
        num_crop_points,
    )

    # 得到可以拍到手的相机序号（按包含更多landmark数目的顺序排序）（此处考虑用检测网络）
    cam_indices = rank_hand_visibility_in_cameras(
        cameras=cameras,
        hand_model=hand_model,
        hand_pose=hand_pose,
        hand_idx=hand_idx,
        min_required_vis_landmarks=min_required_vis_landmarks,
    )

    if sort_camera_index:
        cam_indices = sorted(cam_indices)

    for cam_idx in cam_indices:
        crop_cameras[cam_idx] = gen_crop_parameters_from_points(
            cameras[cam_idx], # 有手的相机
            crop_points, # 63 3 上面算出的landmark
            new_image_size, # 96 96
            mirror_img_x=(mirror_right_hand and hand_idx == 1), # 是否是翻转手
            camera_angle=camera_angles[cam_idx], # 相机轴
            focal_multiplier=focal_multiplier, # 0.95
        )
        if len(crop_cameras) == max_view_num:
            break

    return crop_cameras


def det_box_from_gt(cameras, hand_model, gt_tracking):
    img_landmark_list = {} #[hand_idx][camera_idx]  21*2
    center_length_list = {}  #[hand_idx][camera_idx] 1*3: x,y,l (中心坐标 + 边长)
    hand_viscam_list = {}

    for hand_idx in gt_tracking.keys():

        cam_indices = rank_hand_visibility_in_cameras(
            cameras=cameras,
            hand_model=hand_model,
            hand_pose=gt_tracking[hand_idx],
            hand_idx=hand_idx,
            min_required_vis_landmarks=19,
        )

        # 将各手pose的gt转为landmark
        hand_landmark = landmarks_from_hand_pose(hand_model, gt_tracking[hand_idx], hand_idx)
        # hand_camera_idx = crop_cameras[hand_idx].keys()
        img_landmark_onehand = {}
        center_length_onehand = {}
        for camera_idx in cam_indices:
            camera = cameras[camera_idx]
            cam_landmark = camera.world_to_eye(hand_landmark)  # 世界-相机
            img_landmark = camera.eye_to_window(cam_landmark)  # 相机-像素

            # 找到最小正方形的边界框
            min_x = np.min(img_landmark[:, 0])
            max_x = np.max(img_landmark[:, 0])
            min_y = np.min(img_landmark[:, 1])
            max_y = np.max(img_landmark[:, 1])
            # 计算正方形的中心点
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            # 计算正方形的边长
            side_length = max(max_x - min_x, max_y - min_y) + 20
            # # 构建最小正方形的四个角点
            # detection_square = [(center_x - side_length / 2, center_y - side_length / 2),
            #                     (center_x - side_length / 2, center_y + side_length / 2),
            #                     (center_x + side_length / 2, center_y + side_length / 2),
            #                     (center_x + side_length / 2, center_y - side_length / 2)]

            center_length_cat = np.array([center_x, center_y, side_length])
            img_landmark_onehand[camera_idx] = img_landmark
            center_length_onehand[camera_idx] = center_length_cat


        img_landmark_list[hand_idx] = img_landmark_onehand  # [hand_idx][camera_idx]
        center_length_list[hand_idx] = center_length_onehand # [hand_idx][camera_idx]
        hand_viscam_list[hand_idx] = cam_indices # [hand_idx]
    return img_landmark_list, center_length_list, hand_viscam_list

def gen_crop_cameras_from_box(
    center_length: Dict[int, np.ndarray], # 某手在多个相机中的box
    hand_viscam: List[int], # 某手能被哪几个相机看到
    cameras: List[camera.CameraModel],
    camera_angles: List[float],
    hand_model: HandModel,
    hand_idx: int,
    new_image_size: Tuple[int, int],
    max_view_num: Optional[int] = None,
    focal_multiplier: float = 0.95,
    mirror_right_hand: bool = True,
) -> Dict[int, camera.PinholePlaneCameraModel]:
    crop_cameras: Dict[int, camera.PinholePlaneCameraModel] = {}

    for cam_idx in hand_viscam:
        crop_cameras[cam_idx] = gen_crop_parameters_from_box(
            cameras[cam_idx], # 有手的相机
            center_length[cam_idx], # 检测box
            new_image_size,  # 96 96
            mirror_img_x=(mirror_right_hand and hand_idx == 1),  # 是否是翻转手
            camera_angle=camera_angles[cam_idx],  # 相机轴
            focal_multiplier=focal_multiplier,  # 0.95
        )
        if len(crop_cameras) == max_view_num:
            break

    return crop_cameras


