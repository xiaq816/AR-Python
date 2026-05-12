# video_pose_data.py
import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import av
import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

import lib.data_utils.fs as fs
from lib.common.camera import CameraModel, read_camera_from_json
from lib.common.hand import HandModel
from lib.tracker.StereoReceiver import StereoReceiver
from lib.tracker.tracker import InputFrame, ViewData
from .tracking_result import SingleHandPose

# Initialize logger
logger = logging.getLogger(__name__)

# Constants
REAL_TIME_MODE = "real_time"
DEFAULT_IMAGE_SIZE = (480, 640)
BASELINE_DELTAS = [-40, 40]  # Baseline offsets for stereo cameras
FRAME_READ_RETRY_DELAY = 0.001  # seconds
ERROR_RETRY_DELAY = 0.1  # seconds
IDENTITY_MATRIX = np.eye(4)


@dataclass
class HandPoseLabels:
    """Container for hand pose labels and related camera information."""
    cameras: List[CameraModel]
    camera_angles: List[float]
    hand_model: HandModel


class VideoStream:
    """Video stream handler for both real-time and file-based video sources."""

    def __init__(self, data_path: str):
        """
        Initialize video stream from either real-time source or video file.

        Args:
            data_path: Path to video file or "real_time" for live stream
        """
        self._data_path = data_path
        self.stereo_receiver = None
        self.processing_thread = None
        self.container = None
        self.stream = None

        self._initialize_stream()

    def _initialize_stream(self) -> None:
        """Initialize the appropriate video stream based on data_path."""
        if self._data_path == REAL_TIME_MODE:
            self._initialize_real_time_stream()
        else:
            self._initialize_file_stream()

    def _initialize_real_time_stream(self) -> None:
        """Initialize real-time stereo receiver stream."""
        try:
            self.stereo_receiver = StereoReceiver()
            if not self.stereo_receiver.videocapture():
                raise RuntimeError("Failed to initialize StereoReceiver")

            self.processing_thread = threading.Thread(
                target=self.stereo_receiver.process_frames,
                daemon=True,
                name="StereoReceiver-ProcessingThread"
            )
            self.processing_thread.start()
            logger.info("Real-time video stream initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize real-time stream: {str(e)}")
            raise

    def _initialize_file_stream(self) -> None:
        """Initialize video file stream."""
        try:
            self.container = av.open(self._data_path)
            self.stream = self.container.streams.video[0]
            logger.info(
                f"Opened video file ({int(self.stream.average_rate)} fps) from {self._data_path}"
            )
        except Exception as e:
            logger.error(f"Failed to open video file {self._data_path}: {str(e)}")
            raise

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Iterate through video frames."""
        if self._data_path == REAL_TIME_MODE:
            yield from self._iter_real_time_frames()
        else:
            yield from self._iter_file_frames()

    def _iter_real_time_frames(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate frames from real-time stream."""
        while True:
            try:
                success, stereo_img = self.stereo_receiver.read()
                if not success:
                    time.sleep(FRAME_READ_RETRY_DELAY)
                    continue

                raw_mono_image_np = np.array(stereo_img)
                yield raw_mono_image_np, IDENTITY_MATRIX

            except StopIteration:
                logger.info("Real-time frame iteration stopped")
                break
            except Exception as e:
                logger.error(f"Error reading real-time frame: {str(e)}")
                time.sleep(ERROR_RETRY_DELAY)

    def _iter_file_frames(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate frames from video file."""
        try:
            for frame in self.container.decode(self.stream):
                try:
                    raw_mono_image_np = np.array(frame.to_image())[..., 0]
                    yield raw_mono_image_np, IDENTITY_MATRIX
                except Exception as e:
                    logger.warning(f"Skipping corrupt frame: {str(e)}")
                    continue
        except Exception as e:
            logger.error(f"Error reading video file frames: {str(e)}")
            raise
        finally:
            logger.info(f"Finished reading video file: {self._data_path}")

    def release(self) -> None:
        """Release resources and clean up."""
        try:
            if self._data_path == REAL_TIME_MODE:
                self._release_real_time_stream()
            else:
                self._release_file_stream()
        except Exception as e:
            logger.error(f"Error releasing resources: {str(e)}")

    def _release_real_time_stream(self) -> None:
        """Release real-time stream resources."""
        if self.stereo_receiver:
            try:
                if hasattr(self, 'processing_thread'):
                    self.stereo_receiver.running = False
                    self.processing_thread.join(timeout=1.0)
                    if self.processing_thread.is_alive():
                        logger.warning("Processing thread did not terminate cleanly")
                self.stereo_receiver.release()
                logger.info("Real-time stream resources released")
            except Exception as e:
                logger.error(f"Error releasing real-time stream: {str(e)}")

    def _release_file_stream(self) -> None:
        """Release file stream resources."""
        if self.container:
            try:
                self.container.close()
                logger.info(f"Video file {self._data_path} closed")
            except Exception as e:
                logger.error(f"Error closing video file: {str(e)}")


def _load_json(file_path: str) -> Dict:
    """Load JSON data from file."""
    try:
        with fs.open(file_path, "rb") as bf:
            return json.load(bf)
    except Exception as e:
        logger.error(f"Failed to load JSON file {file_path}: {str(e)}")
        raise


def load_hand_model_from_dict(hand_model_dict: Dict) -> HandModel:
    """Create HandModel instance from dictionary."""
    try:
        hand_tensor_dict = {
            k: torch.Tensor(v) if isinstance(v, list) else v
            for k, v in hand_model_dict.items()
        }
        return HandModel(**hand_tensor_dict)
    except Exception as e:
        logger.error(f"Failed to create HandModel from dict: {str(e)}")
        raise


def _load_hand_pose_labels(json_path: str) -> HandPoseLabels:
    """Load hand pose labels from JSON file."""
    try:
        labels = _load_json(json_path)
        cameras = [read_camera_from_json(c) for c in labels["cameras"]]
        camera_angles = labels["camera_angles"]
        hand_model = load_hand_model_from_dict(labels["hand_model"])

        logger.info(f"Successfully loaded hand pose labels from {json_path}")
        return HandPoseLabels(
            cameras=cameras,
            camera_angles=camera_angles,
            hand_model=hand_model,
        )
    except Exception as e:
        logger.error(f"Failed to load hand pose labels from {json_path}: {str(e)}")
        raise


class SyncedImagePoseStream:
    """Synchronized stream of images and pose data."""

    def __init__(self, data_path: str, json_path: str):
        """
        Initialize synchronized stream.

        Args:
            data_path: Path to video source
            json_path: Path to JSON config file
        """
        self._data_path = data_path
        try:
            self._hand_pose_labels = _load_hand_pose_labels(json_path)
            self._video_stream = VideoStream(data_path)
            self._image_stream = iter(self._video_stream)
            logger.info(f"Initialized SyncedImagePoseStream with {data_path}")
        except Exception as e:
            logger.error(f"Failed to initialize SyncedImagePoseStream: {str(e)}")
            raise

    @property
    def stereo_receiver(self) -> Optional[StereoReceiver]:
        """Get the stereo receiver instance if available."""
        return getattr(self._video_stream, 'stereo_receiver', None)

    def __iter__(self) -> Iterator[InputFrame]:
        """Generate synchronized input frames with camera data."""
        try:
            for raw_mono, camera_to_world in self._image_stream:
                try:
                    multi_view_images = self._prepare_multi_view_images(raw_mono)
                    views = self._create_views(multi_view_images, camera_to_world)
                    yield InputFrame(views=views)
                except Exception as e:
                    logger.warning(f"Skipping frame due to error: {str(e)}")
                    continue
        except Exception as e:
            logger.error(f"Error in frame iteration: {str(e)}")
            raise
        finally:
            logger.info("SyncedImagePoseStream iteration completed")

    def _prepare_multi_view_images(self, raw_mono: np.ndarray) -> np.ndarray:
        """Prepare multi-view images from raw mono image."""
        try:
            num_cameras = len(self._hand_pose_labels.cameras)
            return raw_mono.reshape((num_cameras, -1, raw_mono.shape[1]))
        except Exception as e:
            logger.error(f"Error preparing multi-view images: {str(e)}")
            raise

    def _create_views(self, multi_view_images: np.ndarray, camera_to_world: np.ndarray) -> List[ViewData]:
        """Create ViewData objects for each camera view."""
        views = []
        try:
            for cam_idx, baseline_delta in enumerate(BASELINE_DELTAS):
                try:
                    cur_camera_to_world = self._compute_camera_pose(camera_to_world, baseline_delta)
                    cur_camera = self._hand_pose_labels.cameras[cam_idx].copy(
                        camera_to_world_xf=cur_camera_to_world
                    )
                    self._hand_pose_labels.cameras[cam_idx].camera_to_world_xf = cur_camera_to_world

                    image_cur = self._process_image(multi_view_images[cam_idx])
                    views.append(ViewData(
                        image=image_cur,
                        camera=cur_camera,
                        camera_angle=self._hand_pose_labels.camera_angles[cam_idx]
                    ))
                except Exception as e:
                    logger.warning(f"Skipping camera view {cam_idx} due to error: {str(e)}")
                    continue
            return views
        except Exception as e:
            logger.error(f"Error creating views: {str(e)}")
            raise

    def _process_image(self, image: np.ndarray) -> np.ndarray:
        """Process image to ensure correct dimensions."""
        try:
            if image.shape[:2] != DEFAULT_IMAGE_SIZE:
                img_pil = Image.fromarray(image.astype('uint8'))
                img_pil = transforms.Resize(DEFAULT_IMAGE_SIZE)(img_pil)
                return np.array(img_pil)
            return image
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise

    def _compute_camera_pose(self, left_camera_pose: np.ndarray, baseline: float) -> np.ndarray:
        """Compute right camera pose from left camera pose and baseline."""
        try:
            left_rotation = left_camera_pose[:3, :3]
            left_translation = left_camera_pose[:3, 3]

            right_translation = left_translation - baseline * left_rotation[:, 0]
            right_rotation = left_rotation

            right_camera_pose = IDENTITY_MATRIX.copy()
            right_camera_pose[:3, :3] = right_rotation
            right_camera_pose[:3, 3] = right_translation

            return right_camera_pose
        except Exception as e:
            logger.error(f"Error computing camera pose: {str(e)}")
            raise