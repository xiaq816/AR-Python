# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import numpy as np
import torch

from pytorch3d.transforms import so3_exp_map

from .hand import DOF_PER_FINGER, NUM_DIGITS, NUM_JOINT_FRAMES, HandModel


def _finger_fk( # input: 1 4 4 4 某手指的joint的变换矩阵（局部 相对根节点）        1 4 4 根节点变换矩阵
    joint_local_xfs: torch.Tensor, parent_transform: torch.Tensor
) -> List[torch.Tensor]:
    """
    each finger consisits 4 DoF / Joints,
    and returns 3 transformation frames
    Input:
        joint_local_xfs: (B, 4, 4)
        parent_transform: (B, 4, 4)
    Return:
        transform_mats: (B, 3, 4, 4)
    """
    transform_mats = [parent_transform]
    for i in range(4):
        transform_mats.append(torch.matmul(transform_mats[-1], joint_local_xfs[:, i]))
    test = transform_mats[2:]
    return transform_mats[2:] # 每跟手指留后3个joint


def _joint_local_transform( # input只选前20个joint: 各joint转轴(1 20 3)   各joint的rest坐标(1 20 3)   各joint转角pose(1 20)
    rotation_axis: torch.Tensor, rest_pose: torch.Tensor, joint_angles: torch.Tensor
) -> torch.Tensor:
    rotation_axis_flat = rotation_axis.reshape(-1, 3) # 各joint转轴 20 3
    rest_pose_flat = rest_pose.reshape(-1, 3) # 各joint的rest坐标 20 3
    joint_angles_flat = joint_angles.reshape(-1) #  各joint转角pose 20

    angle_axis = rotation_axis_flat * joint_angles_flat.unsqueeze(-1) # 各joint的轴角 20 3
    local_transform = torch.eye(4, dtype=angle_axis.dtype, device=angle_axis.device) # 对角阵 4 4
    local_transform = local_transform.unsqueeze(dim=0).repeat(angle_axis.shape[0], 1, 1) # 20 4 4 每个joint都有个对角阵

    rot_mat = so3_exp_map(angle_axis) # 20 3 3 每个joint的旋转矩阵
    translation = rest_pose_flat - torch.matmul(
        rot_mat, rest_pose_flat.unsqueeze(dim=-1)
    ).squeeze(dim=-1) # 20 3 每个joint的平移向量
    local_transform[:, :3, :3] = rot_mat
    local_transform[:, 0:3, 3] = torch.squeeze(translation, dim=-1)  # 20 4 4 每个joint的变换矩阵

    return local_transform.reshape(*rotation_axis.shape[0:-1], 4, 4)


def _lbs(trans_mats: torch.Tensor, skinned_points: torch.Tensor) -> torch.Tensor:
    """
    Input:
        trans_mats: (B, 17, 4, 4)
        skinned_points: (B, V, 17, 4)
    Return:
        fk_points: (B, V, 4)
    """
    trans_mats = trans_mats.unsqueeze(dim=1) # 1 1 17 4 4 每个joint的变换矩阵
    skinned_points = skinned_points.unsqueeze(dim=-1)   # 1 21 17 4 1 每个landmark有个17*4的矩阵 rest的
    #test = torch.matmul(trans_mats, skinned_points)

    # sum把17个点求和 加权的过程在算verts处。matmul为T*X得到齐次坐标，将rest的坐标变换到当前pose下的坐标
    fk_points = torch.matmul(trans_mats, skinned_points).sum(dim=2).squeeze(dim=-1)
    return fk_points # 1 21 4 变换完的landmark世界坐标 将rest的坐标变换到当前pose下的坐标


def _get_skinning_weights(
    bone_indices: torch.Tensor, bone_weights: torch.Tensor, n_frames: int
) -> torch.Tensor:
    """
    Input:
        bone_indices: (B, V, K)
        bone_weights: (B, V, K)
        n_frames: (or n_bones) Number of frames/bones (17 for hands)
          Note: K is number of bones.
    Return:
        skin_mat: (B, V, n_frames)
    """

    bs = bone_indices.shape[0]
    n_lms = bone_indices.shape[1]
    # Offset all the bones linearly from 0 to (bs*n_lms*n_frames) so that we can directly
    # index into the flattened weight matrix and set the corresponding skinning weights
    flat_idx_offset = torch.arange(0, bs * n_lms, device=bone_indices.device) * n_frames
    bone_flat_idx = bone_indices.long() + flat_idx_offset.reshape(bs, n_lms, 1)
    skin_mat = torch.zeros(
        bs * n_lms * n_frames, device=bone_weights.device, dtype=bone_weights.dtype
    )
    non0_w_mask = bone_weights != 0
    non0_indices = bone_flat_idx[non0_w_mask]
    skin_mat[non0_indices] = bone_weights[non0_w_mask]
    skin_mat = skin_mat.reshape(bs, n_lms, n_frames)

    return skin_mat


def _hand_skinning_transform(
    rotation_axis: torch.Tensor,  # 1 22 3  各joint转轴
    rest_poses: torch.Tensor,  # 1 22 3  各joint的rest坐标
    joint_angles: torch.Tensor,  # 1 22   各joint转角 pose
    wrist_transforms: torch.Tensor,  # 1 4 4  根节点变换矩阵
) -> torch.Tensor:
    """
    Input:
        rotation_axis: (B, 20, 3)
        rest_poses: (B, 20, 3)
        joint_angles: (B, 20)
        wrist_transforms: (B, 4, 4)
    Return:
        skinning_matrics: (B, 17, 4, 4)
    """
    transform_mats = [wrist_transforms] * 2  # [root_transform, wrist_transfor] 2 * (1 4 4)
    d = DOF_PER_FINGER  # 4

    joint_local_xfs = _joint_local_transform( # input: 各joint转轴  各joint的rest坐标  各joint转角pose
        rotation_axis[:, 0:20], rest_poses[:, 0:20], joint_angles[:, 0:20]
    ) # 1 20 4 4 各joint的变换矩阵
    # 前向计算
    for finger_idx in range(NUM_DIGITS): # NUM_DIGITS=5
        transform_mats += _finger_fk( # 0-4  4-8  8-12  12-16  16-20
            joint_local_xfs[:, d * finger_idx : d * finger_idx + d], wrist_transforms
        )
    transform_mats = torch.cat([m.unsqueeze(1) for m in transform_mats], dim=1)
    return transform_mats # 1 17 4 4 各joint的变换矩阵，由相对根变换×根变换求得 前两个是根变换 后15个是每根手指的三个


def _get_skinned_vertices(vertices: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Input:
        vertices: (B, V, 3) or (B, V, 4)  1 21 3
        weights: (B, V, 17)   1 21 17
    Return:
        skinned_vertices: (B, V, 17, 4)
    """
    if vertices.shape[2] == 3:
        n_vertices = vertices.shape[1]
        homo = torch.ones(
            vertices.shape[0],
            n_vertices,
            1,
            dtype=vertices.dtype,
            device=vertices.device,
        )
        vertices = torch.cat([vertices, homo], dim=-1)  # 变为齐次坐标 1 21 4  rest的landmark坐标

    vertices = vertices.unsqueeze(dim=2) # 1 21 1  4
    weights = weights.unsqueeze(dim=-1)  # 1 21 17 1
    return vertices * weights  # 1 21 17 4


def _skin_points(
    joint_rest_positions: torch.Tensor, # 22 3 rest的joint坐标
    joint_rotation_axes: torch.Tensor, # 22 3 每个joint的转轴
    skin_mat: torch.Tensor, # 1 21 17
    joint_angles: torch.Tensor, # 22 每个joint的旋转
    points: torch.Tensor, # 21 3 rest的landmark坐标
    wrist_transforms: torch.Tensor, # 4 4 根节点变换矩阵
) -> torch.Tensor:
    leading_dims = joint_angles.shape[:-1]
    assert joint_rest_positions.shape[:-2] == leading_dims, (
        "Leading dimensions do not match, "
        + f"got {leading_dims} and {joint_rest_positions.shape[:-2]}"
    )

    # This allows querying the product of leading dimensions without making the
    # model specialized to a particular shape
    numel = torch.flatten(joint_angles, end_dim=-2).shape[0] if len(leading_dims) else 1    # 1

    # rest下joint的坐标
    batched_joint_rest_positions = joint_rest_positions.reshape(numel, -1, 3)   # 1 22 3

    # 1.根据当前各joint的pose算轴角
    # 2.用轴角算旋转矩阵，再根rest的坐标相减算平移向量，然后合成4*4的变换矩阵（此时每个joint有一个变换矩阵，应该是相对父节点的）
    # 3.每个手指分别算。用根变换乘上面算的joint相对变换，一个joint一个joint往上传
    # 4.最后得到各joint的绝对变换矩阵(1 17 4 4)，前两个是根变换 后15个是每根手指的三个
    skin_xfs = _hand_skinning_transform(  #output: 1 17 4 4 每个joint的变换矩阵
        rotation_axis=joint_rotation_axes.reshape(numel, -1, 3), # 1 22 3  各joint转轴
        rest_poses=batched_joint_rest_positions, # 1 22 3  各joint的rest坐标
        joint_angles=joint_angles.reshape(numel, -1), # 1 22   各joint转角 pose
        wrist_transforms=wrist_transforms.reshape(numel, 4, 4), # 1 4 4  根节点变换矩阵
    ) #前两个是根变换 后15个是每根手指的三个 （由 相对根变换*根变换 计算得到）

    # 下面的_get_skinned_vertices 和 _lbs 要计算landmark的坐标，
    # verts是加权的joint但是还没求和，skinned_vecs是将rest的坐标变换到当前pose下的坐标，包含了求和过程

    # _get_skinned_vertices input: rest的landmark坐标(1 21 3)   权重矩阵(1 21 17)
    # _get_skinned_vertices output: 1 21 17 4 rest的 每个landmark有个17*4的矩阵，代表从17个joint中的采样，多数只有一行有值，其余为0.最后一个landmark有3行有值
    verts = _get_skinned_vertices(points.reshape(numel, -1, 3), skin_mat)

    # _lbs input: 每个joint的变换矩阵(1 17 4 4)   每个landmark有个17*4的矩阵(1 21 17 4) rest下
    # _lbs output: 1 21 3 变换完的landmark世界坐标 将rest的坐标变换到当前pose下的坐标
    skinned_vecs = _lbs(skin_xfs, verts)[..., :3]
    skinned_vecs = skinned_vecs.reshape(
        list(leading_dims) + list(skinned_vecs.shape[-2:])
    )
    return skinned_vecs  # 21 3 变换完的landmark世界坐标 将rest的坐标变换到当前pose下的坐标


def skin_landmarks(
    hand_model: HandModel,
    joint_angles: torch.Tensor,
    wrist_transforms: torch.Tensor,
) -> torch.Tensor:
    leading_dims = joint_angles.shape[:-1]
    numel = torch.flatten(joint_angles, end_dim=-2).shape[0] if len(leading_dims) else 1
    max_weights = hand_model.landmark_rest_bone_indices.shape[-1]
    skin_mat = _get_skinning_weights(
        hand_model.landmark_rest_bone_indices.reshape(numel, -1, max_weights),
        hand_model.landmark_rest_bone_weights.reshape(numel, -1, max_weights),
        NUM_JOINT_FRAMES,
    ) # 1 21 17  一行代表一个landmark点，一行有17个数，只有一个位置为1，最后一个点有三个和为1，因为最后一个点是由三个joint点加权求和算出来的
    return _skin_points(
        hand_model.joint_rest_positions, # 22 3 rest的joint坐标
        hand_model.joint_rotation_axes, # 22 3 每个joint的转轴
        skin_mat, # 1 21 17
        joint_angles, # 22 每个joint的旋转
        hand_model.landmark_rest_positions, # 21 3 rest的landmark坐标
        wrist_transforms, # 4 4 根节点变换矩阵
    ) #output: 21 3 变换完的landmark世界坐标 将rest的坐标变换到当前pose下的坐标
