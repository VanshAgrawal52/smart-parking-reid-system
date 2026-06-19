"""
core/perception.py

Handles video ingestion, deep-learning based vehicle detection, and localized object tracking.
This module functions as the initial perception layer for the Intelligent Parking Management System,
extracting tracking metadata from raw video frames.
"""

import cv2
import numpy as np
import time
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
from ultralytics import YOLO


@dataclass
class Detection:
    """
    Represents a single tracked vehicle in a specific frame.
    
    Attributes:
        track_id (int): Unique identifier assigned by the tracker (ByteTrack).
        bbox (List[float]): Bounding box coordinates in [x1, y1, x2, y2] format.
        confidence (float): Detection confidence score (0.0 to 1.0).
        class_id (int): COCO dataset class identifier for the vehicle.
        timestamp (datetime): UTC timestamp of when the detection occurred.
    """
    track_id: int
    bbox: List[float]
    confidence: float
    class_id: int
    timestamp: datetime
    frame_width: int = 0
    frame_height: int = 0


class CameraStream:
    """
    Robust video stream handler capable of managing RTSP feeds, IP cameras, or local files.
    Includes built-in fault tolerance and automatic reconnection logic.
    """

    def __init__(self, source: Union[str, int], reconnect_attempts: int = 5, reconnect_delay: float = 2.0):
        """
        Initializes the CameraStream.
        
        Args:
            source (Union[str, int]): RTSP URL, file path, or webcam index (e.g., 0).
            reconnect_attempts (int): Maximum number of sequential reconnection tries.
            reconnect_delay (float): Seconds to wait between reconnection attempts.
        """
        self.source = source
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.cap: Optional[cv2.VideoCapture] = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self._connect()

    def _connect(self) -> bool:
        """Internal method to establish or re-establish the cv2 VideoCapture connection."""
        if self.cap is not None:
            self.cap.release()
            
        self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            self.logger.error(f"Failed to open video source: {self.source}")
            return False
            
        self.logger.info(f"Successfully connected to video source: {self.source}")
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Safely reads the next frame from the stream. Attempts to reconnect if the stream drops.
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: A boolean indicating success, and the frame (or None).
        """
        if self.cap is None or not self.cap.isOpened():
            if not self._connect():
                return False, None

        ret, frame = self.cap.read()
        
        # If frame read fails, trigger reconnection protocol
        if not ret:
            self.logger.warning(f"Frame drop detected on {self.source}. Attempting to reconnect...")
            for attempt in range(self.reconnect_attempts):
                time.sleep(self.reconnect_delay)
                self.logger.info(f"Reconnection attempt {attempt + 1}/{self.reconnect_attempts}...")
                
                if self._connect():
                    ret, frame = self.cap.read()
                    if ret:
                        self.logger.info("Stream recovered successfully.")
                        return True, frame
                        
            self.logger.error("Max reconnect attempts reached. Stream offline.")
            return False, None
            
        return True, frame

    def release(self) -> None:
        """Releases the video capture resources gracefully."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.logger.info(f"Video stream {self.source} released.")


class PerceptionEngine:
    """
    Executes YOLOv10 object detection and ByteTrack object tracking.
    Filters out non-vehicle entities and yields standardized Detection objects.
    """

    # COCO Class IDs: 2=Car, 3=Motorcycle, 5=Bus, 7=Truck
    TARGET_CLASSES = [2, 3, 5, 7]

    def __init__(
        self, 
        model_path: str, 
        conf_thresh: float = 0.50, 
        iou_thresh: float = 0.45,
        device: str = "cpu",
        tracker_config: str = "bytetrack.yaml"
    ):
        """
        Initializes the deep learning perception model.
        
        Args:
            model_path (str): Path to the Ultralytics YOLO weights file (.pt).
            conf_thresh (float): Minimum confidence threshold for valid detections.
            iou_thresh (float): Non-Maximum Suppression (NMS) intersection over union threshold.
            device (str): Compute device ('cpu', 'cuda:0', etc.).
            tracker_config (str): Tracker definition file utilized by Ultralytics.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device
        self.tracker_config = tracker_config
        
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            self.logger.info(f"Successfully loaded YOLO model '{model_path}' on {self.device}")
        except Exception as e:
            self.logger.critical(f"Failed to load YOLO model: {e}")
            raise

    def filter_vehicle_classes(self, class_id: int) -> bool:
        """
        Validates if a detected class ID belongs to the approved vehicle target list.
        
        Args:
            class_id (int): The COCO class ID to check.
            
        Returns:
            bool: True if the object is a targeted vehicle, False otherwise.
        """
        return class_id in self.TARGET_CLASSES

    def detect_and_track(self, frame: np.ndarray) -> List[Detection]:
        """
        Processes a single frame through the YOLO network and ByteTrack algorithm.
        
        Args:
            frame (np.ndarray): The raw BGR OpenCV video frame.
            
        Returns:
            List[Detection]: A list of localized and tracked vehicle Detection objects.
        """
        try:
            # Ultralytics track() natively combines YOLO inference + ByteTrack assignment
            # passing `classes=self.TARGET_CLASSES` pre-filters NMS at the tensor level
            results = self.model.track(
                frame,
                persist=True,
                tracker=self.tracker_config,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                classes=self.TARGET_CLASSES,
                stream=False,
                verbose=False
            )
        except Exception as e:
            self.logger.error(f"Inference and Tracking engine failed: {e}")
            return []

        active_detections: List[Detection] = []
        now = datetime.utcnow()

        if not results or len(results) == 0:
            return active_detections

        boxes = results[0].boxes
        
        # If no objects are found, or objects are found but not yet assigned an ID by ByteTrack
        if boxes is None or boxes.id is None:
            return active_detections

        # Extract tensor data directly into Python native types
        for i in range(len(boxes.id)):
            track_id = int(boxes.id[i].item())
            bbox = boxes.xyxy[i].tolist()  # Returns [x1, y1, x2, y2]
            conf = float(boxes.conf[i].item())
            cls_id = int(boxes.cls[i].item())

            # Double-check class filtering as a fail-safe
            if self.filter_vehicle_classes(cls_id):
                det = Detection(
                    track_id=track_id,
                    bbox=bbox,
                    confidence=conf,
                    class_id=cls_id,
                    timestamp=now,
                    frame_width=frame.shape[1],
                    frame_height=frame.shape[0]
                )
                active_detections.append(det)

        return active_detections