# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import lib.common.camera as camera
import numpy as np
from scipy.spatial.transform import Rotation
from . import affine

def gen_intrinsics_from_bounding_pts(
    pts_eye: np.ndarray, image_w: int, image_h: int, min_focal: float = 5
) -> Tuple[np.ndarray, np.ndarray]:
    pts_ndc = pts_eye[..., 0:2] / pts_eye[..., 2:] #归一化相机坐标 z为1
    img_size = np.array([image_w, image_h], dtype=pts_eye.dtype)
    # Given our convention, we need to shift one pixel before dividing by 2.
    cx_cy = (img_size - 1) / 2

    test = np.absolute(pts_ndc)
    testtest = np.absolute(pts_ndc).max()

    fx_fy = cx_cy / np.absolute(pts_ndc).max()

    # Some sanity checks
    if np.any(pts_eye[..., 2:] < 0.0001) or np.any(fx_fy < min_focal):
        raise ValueError("Unable to create crop camera", fx_fy)

    return fx_fy, cx_cy


def gen_crop_parameters_from_points(
    camera_orig: camera.CameraModel, # 有手的相机
    pts_world, # 63 3 上面算出的landmark
    new_image_size: Tuple[int, int], # 96 96
    mirror_img_x: bool, # 是否是翻转手
    camera_angle: float = 0, # 相机轴
    focal_multiplier: float = 0.95,  # 0.95
) -> camera.PinholePlaneCameraModel:
    """
    Given the original camera transform and a list of 3D points in the world space,
    compute the new perspective camera that makes sure after projection all the points
    can be projected inside the image.

    Auguments:
    * camera_orig: the original camera used for generating an image. The returned camera
        will have the same position but different rotation and intrinsics parameters.
    * pts_world: points in the world space that must be projected inside the image by
        the generated world to eye transform and intrinsics.
    * new_image_size: target image size
    * mirror_img_x: whether to flip the image. A typical use case is we usually mirror the
        right hand images so that a model need to handle left hand data only
    * camera_angle: how the camera is oriented physically so that we can rotate the object of
        interest to the 'upright' direction
    * focal_multiplier: when less than 1, we are zooming out a little. The effect on the image
        is some margin will be left at the boundary.
    """
    orig_world_to_eye_xf = np.linalg.inv(camera_orig.camera_to_world_xf) # 变换 世界->相机

    crop_center = (pts_world.min(axis=0) + pts_world.max(axis=0)) / 2.0
    new_world_to_eye = affine.make_look_at_matrix(
        orig_world_to_eye_xf, crop_center, camera_angle
    ) # 输入: 世界-相机T  中心landmark  相机角     输出: 虚拟相机外参:新的世界-相机T
    if mirror_img_x:
        mirrorx = np.eye(4, dtype=np.float32)
        mirrorx[0, 0] = -1
        new_world_to_eye = mirrorx @ new_world_to_eye

    fx_fy, cx_cy = gen_intrinsics_from_bounding_pts(
        affine.transform3(new_world_to_eye, pts_world),
        new_image_size[0],
        new_image_size[1],
    ) # 虚拟相机内参
    fx_fy = focal_multiplier * fx_fy

    return camera.PinholePlaneCameraModel(
        width=new_image_size[0],
        height=new_image_size[1],
        f=fx_fy,
        c=cx_cy,
        distort_coeffs=[],
        camera_to_world_xf=np.linalg.inv(new_world_to_eye),
    )

def gen_intrinsics_from_box(
    camera_orig,
    image_w: int,
    image_h: int,
    Rvc: np.ndarray,
    camera_angle: float,  # 相机轴
    center_pixel: np.ndarray,
    length: float,
)-> Tuple[np.ndarray, np.ndarray]:

    img_size = np.array([image_w, image_h], dtype=np.float32)

    cx_cy = (img_size - 1) / 2

    center_x = center_pixel[0]
    center_y = center_pixel[1] # center的xy像素坐标

    box_corner = np.array([[center_x - length / 2, center_y - length / 2],
                                [center_x - length / 2, center_y + length / 2],
                                [center_x + length / 2, center_y + length / 2],
                                [center_x + length / 2, center_y - length / 2]])
    pts_pixel = np.concatenate((box_corner, center_pixel[None,:]), axis=0)

    pts_eye = camera_orig.window_to_eye(pts_pixel)  # 像素-相机
    # pts_ndc = pts_eye[..., 0:2] / pts_eye[..., 2:] # 归一化相面
    # Rcv =
    z_local_rot = Rotation.from_euler("z", camera_angle, degrees=True).as_matrix()

    Rvc_new = Rvc @ z_local_rot # 虚拟相机-真实相机
    Rcv_new = np.linalg.inv(Rvc_new) # 真实相机-虚拟相机 （两相机间只有旋转）
    pts_eye_virt = affine.transform_vec3(Rcv_new, pts_eye)# 真实相机-虚拟相机
    pts_ndc = pts_eye_virt[..., 0:2] / pts_eye_virt[..., 2:] # 虚拟归一化相面

    test = np.absolute(pts_ndc)
    testtest = np.absolute(pts_ndc).max()
    fx_fy = cx_cy / np.absolute(pts_ndc).max()

    # Some sanity checks
    if np.any(pts_eye_virt[..., 2:] < 0.0001) or np.any(fx_fy < 5.0):
        raise ValueError("Unable to create crop camera", fx_fy)

    return fx_fy, cx_cy


def gen_extrinsics_as_umetrack(
    camera_orig: camera.CameraModel,
    center_pixel: np.ndarray,
    orig_world_to_eye: np.ndarray,
    camera_angle: float = 0,
) -> np.ndarray:

    center_cam = camera_orig.window_to_eye(center_pixel)  # 像素-相机 (已是标准化的相机坐标，除以的原始相机坐标的模)
    z_dir_local = center_cam

    delta_r_local = affine.from_two_vectors(
        np.array([0, 0, 1], dtype=center_pixel.dtype), z_dir_local
    ) # 相机系中z轴与center之间的旋转矩阵 (虚拟相机-相机)

    orig_eye_to_world = np.linalg.inv(orig_world_to_eye)  # 变换矩阵 相机->世界

    new_eye_to_world = orig_eye_to_world.copy() # 虚拟相机-世界
    new_eye_to_world[0:3, 0:3] = orig_eye_to_world[0:3, 0:3] @ delta_r_local # # 虚拟相机-世界 (Tcw @ Rvc) @ Pv = Pw

    # Locally rotate the z axis to align with the camera angle
    # 局部旋转 z 轴以与相机角度对齐
    z_local_rot = Rotation.from_euler("z", camera_angle, degrees=True).as_matrix()
    new_eye_to_world[0:3, 0:3] = new_eye_to_world[0:3, 0:3] @ z_local_rot

    return np.linalg.inv(new_eye_to_world), delta_r_local


def gen_extrinsics_as_PCLs(
    camera_orig: camera.CameraModel,
    center_pixel: np.ndarray,
    orig_world_to_eye: np.ndarray,
    camera_angle: float = 0,
) -> np.ndarray:

    center_cam = camera_orig.window_to_eye(center_pixel)  # 像素-相机
    center_cam_z_1 = center_cam / (center_cam[2]) # 归一化相面

    px = center_cam_z_1[0]
    py = center_cam_z_1[1]

    r11 = 1 / np.sqrt(1 + px*px)
    r12 = -px*py / np.sqrt((1 + px*px + py*py)*(1 + px*px))
    r13 = px / np.sqrt(1 + px*px + py*py)
    r21 = 0
    r22 = np.sqrt(1 + px*px) / np.sqrt(1 + px*px + py*py)
    r23 = py / np.sqrt(1 + px*px + py*py)
    r31 =  -px / np.sqrt(1 + px*px)
    r32 = -py / np.sqrt((1 + px*px + py*py)*(1 + px*px))
    r33 = 1 / np.sqrt(1 + px*px + py*py)

    # 虚拟相机-相机
    Rvc = np.array([[r11, r12, r13],
                    [r21, r22, r23],
                    [r31, r32, r33]])

    orig_eye_to_world = np.linalg.inv(orig_world_to_eye)  # 变换矩阵 相机->世界
    new_eye_to_world = orig_eye_to_world.copy()  # 虚拟相机-世界
    new_eye_to_world[0:3, 0:3] = orig_eye_to_world[0:3, 0:3] @ Rvc  # # 虚拟相机-世界 (Tcw @ Rvc) @ Pv = Pw

    # Locally rotate the z axis to align with the camera angle
    # 局部旋转 z 轴以与相机角度对齐
    z_local_rot = Rotation.from_euler("z", camera_angle, degrees=True).as_matrix()
    new_eye_to_world[0:3, 0:3] = new_eye_to_world[0:3, 0:3] @ z_local_rot

    return np.linalg.inv(new_eye_to_world), Rvc

def gen_crop_parameters_from_box(
    camera_orig: camera.CameraModel,  # 有手的相机
    center_length: np.ndarray,
    new_image_size: Tuple[int, int],  # 96 96
    mirror_img_x: bool,  # 是否是翻转手
    camera_angle: float = 0,  # 相机轴
    focal_multiplier: float = 0.95,  # 0.95
) -> camera.PinholePlaneCameraModel:

    orig_world_to_eye_xf = np.linalg.inv(camera_orig.camera_to_world_xf)  # 变换 世界->相机

    center_x = center_length[0]
    center_y = center_length[1] # box中心点
    length = center_length[2] # 正方box边长

    center_pixel = center_length[0:2]

    # 直接用center相素坐标算，与PCL论文方法相同(需要将像素坐标反投影到归一化相机平面)
    new_world_to_eye, Rvc = gen_extrinsics_as_PCLs(camera_orig, center_pixel, orig_world_to_eye_xf, camera_angle) # 世界-虚拟相机

    # 与原方法相同，将像素坐标转为世界坐标
    # new_world_to_eye, Rvc = gen_extrinsics_as_umetrack(camera_orig, center_pixel, orig_world_to_eye_xf, camera_angle)

    if mirror_img_x:
        mirrorx = np.eye(4, dtype=np.float32)
        mirrorx[0, 0] = -1
        new_world_to_eye = mirrorx @ new_world_to_eye


    fx_fy, cx_cy = gen_intrinsics_from_box(
        camera_orig,
        new_image_size[0],
        new_image_size[1],
        Rvc,
        camera_angle,
        center_pixel,
        length,
    )

    fx_fy = focal_multiplier * fx_fy

    return camera.PinholePlaneCameraModel(
        width=new_image_size[0],
        height=new_image_size[1],
        f=fx_fy,
        c=cx_cy,
        distort_coeffs=[],
        camera_to_world_xf=np.linalg.inv(new_world_to_eye),
    )