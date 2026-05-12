# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import cv2
from PIL import Image
import lib.common.camera as camera
import numpy as np
import torch
from lib.common.hand import HandModel, NUM_HANDS, scaled_hand_model
from lib.data_utils import bundles
from lib.models.regressor import RegressorOutput
from lib.models.umetrack_model import InputFrameData, InputFrameDesc, InputSkeletonData
import torchvision.transforms as transforms
from .perspective_crop import gen_crop_cameras_from_pose, det_box_from_gt, gen_crop_cameras_from_box
from .tracking_result import SingleHandPose, TrackingResult
from lib.common.tools import point_vis, point_vis_two_hands, draw_landmark_to_img, draw_landmark_to_img_two_hands, draw_detection_box, draw_test, draw_box_pre_frame
from lib.common.hand import NUM_HANDS, NUM_LANDMARKS_PER_HAND




logger = logging.getLogger(__name__)

MM_TO_M = 0.001
M_TO_MM = 1000.0
MIN_OBSERVED_LANDMARKS = 21
CONFIDENCE_THRESHOLD = 0.5
MAX_VIEW_NUM = 2


@dataclass
class ViewData:
    image: np.ndarray
    camera: camera.CameraModel
    camera_angle: float


@dataclass
class InputFrame:
    views: List[ViewData]


@dataclass
class HandTrackerOpts:
    num_crop_points: int = 63
    enable_memory: bool = True
    use_stored_pose_for_crop: bool = True
    hand_ratio_in_crop: float = 0.95
    min_required_vis_landmarks: int = 19


def _warp_image(
    src_camera: camera.CameraModel,
    dst_camera: camera.CameraModel,
    src_image: np.ndarray,
    interpolation: int = cv2.INTER_LINEAR,
    depth_check: bool = True,
) -> np.ndarray:
    W, H = dst_camera.width, dst_camera.height
    px, py = np.meshgrid(np.arange(W), np.arange(H))
    dst_win_pts = np.column_stack((px.flatten(), py.flatten()))

    dst_eye_pts = dst_camera.window_to_eye(dst_win_pts) # 反投影 像素-虚拟相机
    world_pts = dst_camera.eye_to_world(dst_eye_pts) # 虚拟相机-世界
    src_eye_pts = src_camera.world_to_eye(world_pts) # 世界-真实相机
    src_win_pts = src_camera.eye_to_window(src_eye_pts) # 真实相机-像素

    # Mask out points with negative z coordinates
    if depth_check:
        mask = src_eye_pts[:, 2] < 0
        src_win_pts[mask] = -1

    src_win_pts = src_win_pts.astype(np.float32)

    map_x = src_win_pts[:, 0].reshape((H, W))
    map_y = src_win_pts[:, 1].reshape((H, W))

    # current_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 精确到毫秒，去掉最后三位微秒
    # test2 = cv2.remap(src_image, map_x, map_y, interpolation)
    # cv2.imshow('test2', test2)
    # cv2.waitKey(100)
    # # cv2.imwrite("./test_result/crop/" + f"{current_time}.jpg", test2)

    return cv2.remap(src_image, map_x, map_y, interpolation)


class HandTracker:
    def __init__(self, model, det_model, opts: HandTrackerOpts) -> None:
        self._device: str = "cuda" if torch.cuda.device_count() else "cpu"
        logger.info(f"Using device: {self._device}")

        self._model = model
        self._model.to(self._device) # 姿态估计模型
        self._det_model = det_model
        self._det_model.to(self._device) # 检测模型

        self._input_size = np.array(self._model.getInputImageSizes())
        self._num_crop_points = opts.num_crop_points
        self._enable_memory = opts.enable_memory
        self._hand_ratio_in_crop: float = opts.hand_ratio_in_crop
        self._min_required_vis_landmarks: int = opts.min_required_vis_landmarks
        self._valid_tracking_history = np.zeros(2, dtype=bool)
        self._pose_confidence = np.ones(2, dtype=np.float32)

    def reset_history(self) -> None:
        self._valid_tracking_history[:] = False

    # 根据ground truth算虚拟相机
    def gen_crop_cameras(
        self,
        cameras: List[camera.CameraModel],
        camera_angles: List[float],
        hand_model: HandModel,
        gt_tracking: Dict[int, SingleHandPose],
        min_num_crops: int,
    ) -> Dict[int, Dict[int, camera.PinholePlaneCameraModel]]:
        crop_cameras: Dict[int, Dict[int, camera.PinholePlaneCameraModel]] = {}
        if not gt_tracking:
            return crop_cameras

        for hand_idx, gt_hand_pose in gt_tracking.items():
            if gt_hand_pose.hand_confidence < CONFIDENCE_THRESHOLD:
                continue
            crop_cameras[hand_idx] = gen_crop_cameras_from_pose(
                cameras,
                camera_angles,
                hand_model,
                gt_hand_pose,
                hand_idx,
                self._num_crop_points,
                self._input_size,
                max_view_num=MAX_VIEW_NUM,
                sort_camera_index=True,
                focal_multiplier=self._hand_ratio_in_crop,
                mirror_right_hand=True,
                min_required_vis_landmarks=self._min_required_vis_landmarks,
            )

        # Remove empty crop_cameras
        del_list = []
        for hand_idx, per_hand_crop_cameras in crop_cameras.items():
            if not per_hand_crop_cameras or len(per_hand_crop_cameras) < min_num_crops:
                del_list.append(hand_idx)
        for hand_idx in del_list:
            del crop_cameras[hand_idx]

        return crop_cameras

    def image_norm(self, image):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.24043], [0.24709])
        ])
        image = transform(image)
        return image

    def center_length_to_pixel(self, center_length, H, W):
        center_length[:, :, 0] = center_length[:, :, 0] * W  # cx
        center_length[:, :, 1] = center_length[:, :, 1] * H  # cy
        center_length[:, :, 2] = center_length[:, :, 2] * W  # length
        return center_length

    def det_hand_box(self, cameras, hand_model, images):
        center_length_list = {} #[hand_idx][camera_idx] 1*3: x,y,l (中心坐标 + 边长)
        hand_viscam_list = {}

        left_cam = []  # 能看见左手的相机id
        right_cam = [] # 能看见右手的相机id

        all_center_length = np.zeros([len(images), NUM_HANDS, 3])

        img_norm_list = []
        for img_idx, img_ori in enumerate(images):
            if img_ori.shape[0] != 480 or img_ori.shape[1] != 640:
                img_image = Image.fromarray(img_ori.astype('uint8'))
                img_image = transforms.Resize((480, 640))(img_image)
                img_ori = np.array(img_image)
            img_norm = self.image_norm(img_ori[:, :, None]).unsqueeze(0)
            img_norm_list.append(img_norm)

        img_norm_input = torch.cat(img_norm_list, dim=0).cuda()
        with torch.no_grad():
            center_length_all, cls_all = self._det_model(img_norm_input)

        H = img_norm_input.size()[2]
        W = img_norm_input.size()[3]

        for i in range(len(img_norm_list)):
            center_length = center_length_all[i, :, :].unsqueeze(0)
            cls = cls_all[i, :].unsqueeze(0)
            center_length = self.center_length_to_pixel(center_length, H, W).cpu()  # 反归一化
            cls = torch.where(cls.cpu() > 0.9, torch.tensor(1), torch.tensor(0))  # 大于0.7为1，小于0.7为0
            center_length_np = center_length.squeeze(0).detach().numpy()
            cls_np = cls.squeeze(0).detach().numpy()

            for hand_idx, cls_conf in enumerate(cls_np):
                if cls_conf == 1:
                    all_center_length[i, hand_idx, :] = center_length_np[hand_idx]
                    if hand_idx == 0:
                        left_cam.append(i)
                    elif hand_idx == 1:
                        right_cam.append(i)


        # for img_idx, img_ori in enumerate(images):
        #     if img_ori.shape[0] != 480 or img_ori.shape[1] != 640:
        #         img_image = Image.fromarray(img_ori.astype('uint8'))
        #         img_image = transforms.Resize((480, 640))(img_image)
        #         img_ori = np.array(img_image)
        #     img_norm = self.image_norm(img_ori[:, :, None]).unsqueeze(0).cuda()
        #     with torch.no_grad():
        #         center_length, cls = self._det_model(img_norm)
        #         H = img_ori.shape[0]
        #         W = img_ori.shape[1]
        #         center_length = self.center_length_to_pixel(center_length, H, W).cpu()  # 反归一化
        #         cls = torch.where(cls.cpu() > 0.7, torch.tensor(1), torch.tensor(0))  # 大于0.7为1，小于0.7为0
        #
        #         center_length_np = center_length.squeeze(0).detach().numpy()
        #         cls_np = cls.squeeze(0).detach().numpy()
        #
        #         for hand_idx, cls_conf in enumerate(cls_np):
        #             if cls_conf == 1:
        #                 all_center_length[img_idx, hand_idx, :] = center_length_np[hand_idx]
        #                 if hand_idx == 0:
        #                     left_cam.append(img_idx)
        #                 elif hand_idx == 1:
        #                     right_cam.append(img_idx)

        # # test
        # for hand_idx in range(NUM_HANDS):
        #     l_center_length = all_center_length[:, hand_idx, :]
        #     one_hand_center = []
        #     camera_0 = cameras[0]
        #     for camera_idx, camera in enumerate(cameras):
        #         center_pixel = l_center_length[camera_idx][0:2]
        #         center_cam = camera.window_to_eye(center_pixel)  # 像素-相机
        #         center_world = camera.eye_to_world(center_cam) # 像素-世界
        #         center_cam_0 = camera_0.world_to_eye(center_world) # 世界-相机0
        #         center_pixel_0 = camera_0.eye_to_window(center_cam_0) # 相机0-图像0
        #         one_hand_center.append(center_pixel_0)
        #     print('test')

        if len(left_cam) != 0:
            hand_viscam_list[0] = left_cam
        if len(right_cam) != 0:
            hand_viscam_list[1] = right_cam

        for hand_idx in hand_viscam_list.keys():
            cam_indices = hand_viscam_list[hand_idx]
            center_length_onehand = {}
            for camera_idx in cam_indices:
                center_length_tmp = all_center_length[camera_idx, hand_idx, :]
                center_length_onehand[camera_idx] = center_length_tmp

            center_length_list[hand_idx] = center_length_onehand

        return center_length_list, hand_viscam_list

    def gen_crop_cameras_detection(
        self,
        images: List[np.ndarray],
        cameras: List[camera.CameraModel],
        camera_angles: List[float],
        hand_model: HandModel,
        gt_tracking: Dict[int, SingleHandPose],
        min_num_crops: int,
    ) -> Dict[int, Dict[int, camera.PinholePlaneCameraModel]]:
        crop_cameras: Dict[int, Dict[int, camera.PinholePlaneCameraModel]] = {}

        # 后面换成检测网络
        # img_landmark_list, center_length_list, hand_viscam_list = det_box_from_gt(cameras, hand_model, gt_tracking)
        center_length_list, hand_viscam_list = self.det_hand_box(cameras, hand_model, images)

        # -------------------- 画图 landmark/检测框box ---------------------
        # draw_landmark_to_img_two_hands(images[2], img_landmark_list, 2)
        # draw_detection_box(images[2], center_length_list, 2)

        for hand_idx in hand_viscam_list.keys():
            crop_cameras[hand_idx] = gen_crop_cameras_from_box(
                center_length_list[hand_idx],
                hand_viscam_list[hand_idx],
                cameras,
                camera_angles,
                hand_model,
                hand_idx,
                self._input_size,
                max_view_num=MAX_VIEW_NUM,
                focal_multiplier=self._hand_ratio_in_crop,
                mirror_right_hand=True,
            )

            # Remove empty crop_cameras
        del_list = []
        for hand_idx, per_hand_crop_cameras in crop_cameras.items():
            if not per_hand_crop_cameras or len(per_hand_crop_cameras) < min_num_crops:
                del_list.append(hand_idx)
        for hand_idx in del_list:
            del crop_cameras[hand_idx]


        return crop_cameras


    # 根据前一帧计算的位姿算虚拟相机
    def gen_crop_cameras_memory(
        self,
        cameras: List[camera.CameraModel],
        camera_angles: List[float],
        hand_model: HandModel,
        gt_tracking: Dict[int, SingleHandPose],
        min_num_crops: int,
    ) -> Dict[int, Dict[int, camera.PinholePlaneCameraModel]]:
        crop_cameras: Dict[int, Dict[int, camera.PinholePlaneCameraModel]] = {}
        if not gt_tracking:
            return crop_cameras

        for hand_idx, gt_hand_pose in gt_tracking.items():
            if gt_hand_pose.hand_confidence < CONFIDENCE_THRESHOLD:
                continue
            crop_cameras[hand_idx] = gen_crop_cameras_from_pose(
                cameras,
                camera_angles,
                hand_model,
                gt_hand_pose,
                hand_idx,
                self._num_crop_points,
                self._input_size,
                max_view_num=MAX_VIEW_NUM,
                sort_camera_index=True,
                focal_multiplier=self._hand_ratio_in_crop,
                mirror_right_hand=True,
                min_required_vis_landmarks=self._min_required_vis_landmarks,
            )

        # Remove empty crop_cameras
        del_list = []
        for hand_idx, per_hand_crop_cameras in crop_cameras.items():
            if not per_hand_crop_cameras or len(per_hand_crop_cameras) < min_num_crops:
                del_list.append(hand_idx)
        for hand_idx in del_list:
            del crop_cameras[hand_idx]

        return crop_cameras

    def gen_crop_cameras_det_mem(
        self,
        images: List[np.ndarray],
        cameras: List[camera.CameraModel],
        camera_angles: List[float],
        hand_model: HandModel,
        previous_tracking: Dict[int, SingleHandPose],
        valid_tracking_history: np.ndarray,
        pose_confidence: np.ndarray,
        min_num_crops: int,
    ) -> Dict[int, Dict[int, camera.PinholePlaneCameraModel]]:
        crop_cameras: Dict[int, Dict[int, camera.PinholePlaneCameraModel]] = {}
        # use_method: Dict[int, str]
        use_method = {0:'None', 1:'None'}

        center_length_list = {} #[hand_idx][camera_idx] 1*3: x,y,l (中心坐标 + 边长)
        hand_viscam_list = {}
        det_fig = 0 # 是否检测过手的标志 0为没检测过 1为检测过

        # print(valid_tracking_history)
        # print(pose_confidence)
        pose_threshold = 0.3


        for hand_idx in range(2):
            # 若上一帧有这只手则用上一帧的结果 且 pose_confidence<0.5
            if valid_tracking_history[hand_idx] == True and hand_idx in previous_tracking.keys() and pose_confidence[hand_idx] < pose_threshold: # and pose_confidence[hand_idx] < pose_threshold
                hand_pose =  previous_tracking[hand_idx]
                if hand_pose.hand_confidence < CONFIDENCE_THRESHOLD:
                    continue
                crop_cameras[hand_idx] = gen_crop_cameras_from_pose(
                    cameras,
                    camera_angles,
                    hand_model,
                    hand_pose,
                    hand_idx,
                    self._num_crop_points,
                    self._input_size,
                    max_view_num=MAX_VIEW_NUM,
                    sort_camera_index=True,
                    focal_multiplier=self._hand_ratio_in_crop,
                    mirror_right_hand=True,
                    min_required_vis_landmarks=self._min_required_vis_landmarks,
                )
                use_method[hand_idx] = 'memory'
            # 若上一帧没有这只手则做检测
            else:
                if det_fig == 0:
                    center_length_list, hand_viscam_list = self.det_hand_box(cameras, hand_model, images)
                    det_fig = 1
                if hand_idx in hand_viscam_list.keys():
                    crop_cameras[hand_idx] = gen_crop_cameras_from_box(
                        center_length_list[hand_idx],
                        hand_viscam_list[hand_idx],
                        cameras,
                        camera_angles,
                        hand_model,
                        hand_idx,
                        self._input_size,
                        max_view_num=MAX_VIEW_NUM,
                        focal_multiplier=self._hand_ratio_in_crop,
                        mirror_right_hand=True,
                    )
                    use_method[hand_idx] = 'detection'

        # print('left:', use_method[0], '', pose_confidence[0] ,'   right:',use_method[1], '', pose_confidence[1] , '    det_fig:', det_fig)


        # Remove empty crop_cameras
        del_list = []
        for hand_idx, per_hand_crop_cameras in crop_cameras.items():
            if not per_hand_crop_cameras or len(per_hand_crop_cameras) < min_num_crops:
                del_list.append(hand_idx)
        for hand_idx in del_list:
            del crop_cameras[hand_idx]

        return crop_cameras





    def track_frame(
        self,
        sample: InputFrame,
        hand_model: HandModel,
        crop_cameras: Dict[int, Dict[int, camera.PinholePlaneCameraModel]],
    ) -> TrackingResult:
        if not crop_cameras:
            # Frame without hands
            self.reset_history()
            return TrackingResult()

        frame_data, frame_desc, skeleton_data = self._make_inputs(
            sample, hand_model, crop_cameras
        )
        with torch.no_grad():
            regressor_output = bundles.to_device(
                self._model.regress_pose_use_skeleton(
                    frame_data, frame_desc, skeleton_data
                ),
                torch.device("cuda"),
            )

        tracking_result = self._gen_tracking_result(
            regressor_output,
            frame_desc.hand_idx.cpu().numpy(),
            crop_cameras,
        )

        # # 获得位姿估计置信度
        # pose_confidence = torch.sum(regressor_output.landmark_uncertainty_sigmas, dim=1).cpu().numpy()
        # if self._valid_tracking_history.sum() == 2:
        #     self._pose_confidence = pose_confidence
        # elif  self._valid_tracking_history.sum() == 1:
        #     true_idx = [i for i, value in enumerate(self._valid_tracking_history) if value]
        #     self._pose_confidence[true_idx] = pose_confidence
        # else:
        #     self._pose_confidence = np.ones(2, dtype=np.float32)

        return tracking_result

    def track_frame_and_calibrate_scale(
        self,
        sample: InputFrame,
        crop_cameras: Dict[int, Dict[int, camera.PinholePlaneCameraModel]],
    ) -> TrackingResult:
        if not crop_cameras:
            # Frame without hands
            self.reset_history()
            return TrackingResult()
        frame_data, frame_desc, _ = self._make_inputs(sample, None, crop_cameras)

        with torch.no_grad():
            regressor_output = bundles.to_device(
                self._model.regress_pose_pred_skel_scale(frame_data, frame_desc),
                torch.device("cpu"),
            )

        tracking_result = self._gen_tracking_result(
            regressor_output,
            frame_desc.hand_idx.cpu().numpy(),
            crop_cameras,
        )
        return tracking_result

    def _make_inputs(
        self,
        sample: InputFrame,
        hand_model_mm: Optional[HandModel],
        crop_cameras: Dict[int, Dict[int, camera.PinholePlaneCameraModel]],
    ):
        image_idx = 0
        left_images = []
        intrinsics = []
        extrinsics_xf = []
        sample_range_n_hands = []
        hand_indices = []
        for hand_idx, crop_camera_info in crop_cameras.items():
            sample_range_start = image_idx
            for cam_idx, crop_camera in crop_camera_info.items():
                view_data = sample.views[cam_idx]
                crop_image = _warp_image(view_data.camera, crop_camera, view_data.image)
                left_images.append(crop_image.astype(np.float32) / 255.0)
                intrinsics.append(crop_camera.uv_to_window_matrix())

                crop_world_to_eye_xf = np.linalg.inv(crop_camera.camera_to_world_xf)
                crop_world_to_eye_xf[:3, 3] *= MM_TO_M
                extrinsics_xf.append(crop_world_to_eye_xf)

                image_idx += 1

            if image_idx > sample_range_start:
                hand_indices.append(hand_idx)
                sample_range_n_hands.append(np.array([sample_range_start, image_idx]))

        hand_indices = np.array(hand_indices)
        frame_data = InputFrameData(
            left_images=torch.from_numpy(np.stack(left_images)).float(),
            intrinsics=torch.from_numpy(np.stack(intrinsics)).float(),
            extrinsics_xf=torch.from_numpy(np.stack(extrinsics_xf)).float(),
        )
        frame_desc = InputFrameDesc(
            sample_range=torch.from_numpy(np.stack(sample_range_n_hands)).long(),
            memory_idx=torch.from_numpy(hand_indices).long(),
            # use memory if the hand is previously valid
            use_memory=torch.from_numpy(
                self._valid_tracking_history[hand_indices]
            ).bool(),
            hand_idx=torch.from_numpy(hand_indices).long(),
        )
        skeleton_data = None
        if hand_model_mm is not None:
            # m -> mm
            hand_model_m = scaled_hand_model(hand_model_mm, MM_TO_M)
            skeleton_data = InputSkeletonData(
                joint_rotation_axes=hand_model_m.joint_rotation_axes.float(),
                joint_rest_positions=hand_model_m.joint_rest_positions.float(),
            )
        # test11 = bundles.to_device((frame_data, frame_desc, skeleton_data), self._device)
        return bundles.to_device((frame_data, frame_desc, skeleton_data), self._device)

    def _gen_tracking_result(
        self,
        regressor_output: RegressorOutput,
        hand_indices: np.ndarray,
        crop_cameras: Dict[int, Dict[int, camera.PinholePlaneCameraModel]],
    ) -> TrackingResult:

        output_joint_angles = regressor_output.joint_angles.to("cpu").numpy()
        output_wrist_xforms = regressor_output.wrist_xfs.to("cpu").numpy()
        output_wrist_xforms[..., :3, 3] *= M_TO_MM
        output_scales = None
        if regressor_output.skel_scales is not None:
            output_scales = regressor_output.skel_scales.to("cpu").numpy()

        hand_poses = {}
        num_views = {}
        predicted_scales = {}
        pose_confidences = {}

        # 存pose_confidence
        pose_confidence = torch.sum(regressor_output.landmark_uncertainty_sigmas, dim=1).cpu().numpy()
        for output_idx, hand_idx in enumerate(hand_indices):
            pose_confidences[hand_idx] = pose_confidence[output_idx]

        threshold = 0.4
        for output_idx, hand_idx in enumerate(hand_indices):
            if pose_confidences[hand_idx] > threshold: # 若结果大于阈值则认为检测的有问题 不存储计算的pose
                continue

            raw_handpose = SingleHandPose(
                joint_angles=output_joint_angles[output_idx],
                wrist_xform=output_wrist_xforms[output_idx],
                hand_confidence=1.0,
            )
            hand_poses[hand_idx] = raw_handpose
            num_views[hand_idx] = len(crop_cameras[hand_idx])
            if output_scales is not None:
                predicted_scales[hand_idx] = output_scales[output_idx]

        for hand_idx in range(NUM_HANDS):
            hand_valid = False
            if hand_idx in hand_poses:
                self._valid_tracking_history[hand_idx] = True
                hand_valid = True
            if hand_valid:
                continue
            self._valid_tracking_history[hand_idx] = False

        for hand_idx in range(NUM_HANDS):
            pose_confid = False
            if hand_idx in hand_poses:
                self._pose_confidence[hand_idx] = pose_confidences[hand_idx]
                pose_confid = True
            if pose_confid:
                continue
            self._pose_confidence[hand_idx] = 1.0

        # print(pose_confidence)

        return TrackingResult(
            hand_poses=hand_poses,
            num_views=num_views,
            predicted_scales=predicted_scales,
        )
