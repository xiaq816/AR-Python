import logging
import os
import sys
import threading
import time
import socket
from typing import Dict, List, Optional, Tuple

import numpy as np

from lib.common.hand import NUM_HANDS, NUM_LANDMARKS_PER_HAND
from lib.common.tools import (
    draw_landmark_to_img_two_hands,
    point_vis,
    point_vis_two_hands,
    draw_landmark_to_img,
    draw_detection_box,
    draw_test,
    draw_box_pre_frame,
    save_draw_landmark_to_img_two_hands,
)
from lib.Gesture.GestureRecognition import GestureRecognition
from lib.Gesture.GestureRecognitionInterface import global_gesture_interface
from lib.models.model_loader import (
    load_pretrained_model,
    load_detection_model,
    load_gesture_model,
)
from lib.tracker.perspective_crop import landmarks_from_hand_pose
from lib.tracker.tracker import HandTracker, HandTrackerOpts, InputFrame, ViewData
from lib.tracker.video_pose_data import SyncedImagePoseStream, VideoStream

# Constants
DEFAULT_FRAME_GAP = 0
DEFAULT_WINDOW_SIZE = 24
GESTURE_LABELS = [
    '左扫', '右扫', '上扫', '下扫', '前推', '后拉', '挥手', '抓握', '释放',
    'Pitch（单击）拇指+食指', 'Pitch（单击）拇指+小指', 'Pitch（双击）拇指+食指',
    'Pitch（双击）拇指+小指', '拇指左扫', '拇指右扫', '食指顺时针转', '食指逆时针转',
    '双手握', '双手交叉', '右手前切', '左手前切', '拇指上扫', '拇指下扫',
    '食指向前点击', '手掌顺时针转', '手掌逆时针转'
]
CONFIDENCE_THRESHOLD_HIGH = 0.95
CONFIDENCE_THRESHOLD_LOW = 0.65
DEFAULT_LABEL = -1
SPECIAL_LABEL = 14

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _track_sequence(
        input_path: str,
        json_path: str,
        rec_frame_gap: int = DEFAULT_FRAME_GAP,
        window_size: int = DEFAULT_WINDOW_SIZE,
        label_name: List[str] = GESTURE_LABELS,
        model_path: str = "",
        det_model_path: str = "",
        ges_model_path: List[str] = None
) -> Optional[np.ndarray]:
    """
    Track hand sequence and recognize gestures.
    Optimized socket connection: persistent with reconnect.
    """
    if ges_model_path is None:
        ges_model_path = []

    logger.info(f"Processing {input_path}...")

    # Initialize socket (persistent connection)
    loopback_socket = None
    loopback_address = ('127.0.0.1', 8000)
    last_connection_attempt = 0
    CONNECTION_RETRY_INTERVAL = 10  # seconds

    def _ensure_connected():
        nonlocal loopback_socket, last_connection_attempt
        current_time = time.time()
        if loopback_socket is None and (current_time - last_connection_attempt) >= CONNECTION_RETRY_INTERVAL:
            try:
                loopback_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                loopback_socket.settimeout(1.0)  # 1-second timeout
                loopback_socket.connect(loopback_address)
                logger.info("Socket connected to 127.0.0.1:8000")
                last_connection_attempt = current_time
            except Exception as e:
                logger.warning(f"Socket connection failed: {e}")
                if loopback_socket:
                    loopback_socket.close()
                loopback_socket = None
                last_connection_attempt = current_time

    _ensure_connected()  # Initial connection

    # Load models (original code)
    model = load_pretrained_model(model_path)
    model.eval()

    det_model = load_detection_model(det_model_path)
    det_model.eval()

    SH_ges_rec_model, HH_ges_rec_model = load_gesture_model(ges_model_path)
    SH_ges_rec_model.eval()
    HH_ges_rec_model.eval()

    # Initialize streams (original code)
    image_pose_stream = SyncedImagePoseStream(input_path, json_path)
    stereo_receiver = image_pose_stream.stereo_receiver

    if stereo_receiver is None:
        logger.error("StereoReceiver not available!")
        return None

    if not hasattr(stereo_receiver, 'unity_socket') or stereo_receiver.unity_socket is None:
        if not stereo_receiver.connect_to_unity():
            logger.error("Failed to connect to Unity for gesture data!")

    iter_image = iter(image_pose_stream)
    gesture_rec_stream = GestureRecognition(
        SH_ges_rec_model,
        HH_ges_rec_model,
        frame_gap=rec_frame_gap,
        window_size=window_size,
        label_name=label_name
    )

    # Initialize tracking variables (original code)
    tracked_keypoints = np.zeros([NUM_HANDS, NUM_LANDMARKS_PER_HAND, 3])
    valid_tracking = np.zeros([NUM_HANDS], dtype=bool)
    tracker = HandTracker(model, det_model, HandTrackerOpts())
    last_hand_pose = 0
    frame_id = 0

    while True:
        start_time = time.time()
        sys.stdout.flush()

        try:
            input_frame = next(iter_image)
        except StopIteration:
            logger.info("End of video stream reached")
            break

        hand_model = image_pose_stream._hand_pose_labels.hand_model

        # Generate crop cameras (original code)
        crop_cameras = tracker.gen_crop_cameras_det_mem(
            [view.image for view in input_frame.views],
            [view.camera for view in input_frame.views],
            image_pose_stream._hand_pose_labels.camera_angles,
            hand_model,
            last_hand_pose,
            tracker._valid_tracking_history,
            tracker._pose_confidence,
            min_num_crops=1,
        )

        # Track frame (original code)
        res = tracker.track_frame(input_frame, hand_model, crop_cameras)

        # Update tracked keypoints (original code)
        for hand_idx in res.hand_poses.keys():
            tracked_keypoints[hand_idx] = landmarks_from_hand_pose(
                hand_model, res.hand_poses[hand_idx], hand_idx
            )
            valid_tracking[hand_idx] = True

        last_hand_pose = res.hand_poses

        # Process landmarks for visualization (original code)
        draw_cam_id = 1
        cur_frame_landmark = tracked_keypoints
        img_landmark_list = {}

        for hand_idx in res.hand_poses.keys():
            hand_landmark = cur_frame_landmark[hand_idx]
            hand_camera_idx = crop_cameras[hand_idx].keys()
            img_landmark_onehand = {}

            for camera_idx in hand_camera_idx:
                camera = input_frame.views[camera_idx].camera
                cam_landmark = camera.world_to_eye(hand_landmark)
                img_landmark = camera.eye_to_window(cam_landmark)
                img_landmark_onehand[camera_idx] = img_landmark

            img_landmark_list[hand_idx] = img_landmark_onehand

        end_time = time.time()

        # Draw landmarks (original code)
        draw_landmark_to_img_two_hands(
            input_frame.views[draw_cam_id].image,
            img_landmark_list,
            draw_cam_id
        )

        sys.stdout.flush()

        # Gesture recognition (original code)
        predict_label, prob, _ = gesture_rec_stream.Recognition(
            cur_frame_landmark,
            tracker._valid_tracking_history
        )

        # Process prediction results (original code)
        predict_label = _process_prediction(predict_label, prob)

        if predict_label != DEFAULT_LABEL:
            predict_name = label_name[predict_label]
            logger.info(f"label: {predict_label} pred: {prob} name: {predict_name}")

        # Send landmarks to Unity (original code)
        if hasattr(stereo_receiver, 'landmarks_to_unity'):
            stereo_receiver.landmarks_to_unity(
                cur_frame_landmark,
                predict_label,
                tracker._valid_tracking_history
            )

        # Optimized socket send with reconnect
        if predict_label != DEFAULT_LABEL:
            try:
                if loopback_socket is None:
                    _ensure_connected()  # Reconnect if needed

                if loopback_socket is not None:
                    server_timestamp = int(time.time() * 1000)
                    msg = f"2,{server_timestamp},{predict_label}\n"
                    loopback_socket.send(msg.encode('utf-8'))
            except Exception as e:
                logger.warning(f"Failed to send data: {e}")
                if loopback_socket:
                    loopback_socket.close()
                loopback_socket = None  # Trigger reconnect next time

        frame_id += 1

    # Cleanup
    if loopback_socket:
        loopback_socket.close()


def _process_prediction(predict_label: Optional[int], prob: float) -> int:
    """Original prediction processing logic."""
    if predict_label is None:
        return DEFAULT_LABEL

    if prob < CONFIDENCE_THRESHOLD_HIGH:
        return DEFAULT_LABEL

    return predict_label

if __name__ == '__main__':
    root = os.path.dirname(__file__)

    # -------- 模型权重路径 --------
    model_path = os.path.join(root, "pretrained_models", "pretrained_weights.torch")
    det_model_path = os.path.join(
        root, "pretrained_models",
        "UmeDet_gap10_separate_rot_random_add_0_2_catourlarge_gap0_6721.pt"
    )
    ges_model_path = [
        os.path.join(root, "pretrained_models", "SH22_stgcn_CS_24frames_rot_10_20_9939.pt"),
        os.path.join(root, "pretrained_models", "HH4_stgcn_tiny_CV_24frames_rot_10_20_fullgraph_1000.pt")
    ]

    input_path = "real_time"
    json_path = "ourdata/parameters.json"

    _track_sequence(
        input_path=input_path,
        json_path=json_path,
        rec_frame_gap=DEFAULT_FRAME_GAP,
        window_size=DEFAULT_WINDOW_SIZE,
        label_name=GESTURE_LABELS,
        model_path=model_path,
        det_model_path=det_model_path,
        ges_model_path=ges_model_path
    )


    logger.info('Processing completed!')