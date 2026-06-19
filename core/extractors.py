"""
core/extractors.py

Handles the extraction of secondary features from detected vehicles.
This module is responsible for executing Optical Character Recognition (OCR) 
on license plates and extracting visual attributes (Color, Type) from vehicle crops.
"""

import cv2
import numpy as np
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from paddleocr import PaddleOCR


@dataclass
class VehicleAttributes:
    """
    Data structure representing extracted secondary features of a vehicle.
    
    Attributes:
        plate_text (Optional[str]): The recognized alphanumeric string from the license plate.
        plate_confidence (Optional[float]): The OCR confidence score (0.0 to 1.0).
        vehicle_color (Optional[str]): The dominant localized color of the vehicle.
        vehicle_type (Optional[str]): The mapped vehicle classification (e.g., 'car', 'truck').
        timestamp (datetime): The exact time the extraction took place.
    """
    plate_text: Optional[str]
    plate_confidence: Optional[float]
    vehicle_color: Optional[str]
    vehicle_type: Optional[str]
    timestamp: datetime


class OCRExtractor:
    """
    Manages the execution of PaddleOCR on cropped license plate images.
    Extracts alphanumeric characters and confidence scores.
    """

    def __init__(self, lang: str = 'en', use_gpu: bool = False, confidence_threshold: float = 0.60):
        """
        Initializes the PaddleOCR engine.
        
        Args:
            lang (str): Language code for OCR (default is 'en').
            use_gpu (bool): Whether to utilize GPU acceleration.
            confidence_threshold (float): Minimum confidence to consider a plate valid.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.confidence_threshold = confidence_threshold
        
        try:
            # show_log=False suppresses PaddleOCR's verbose standard output
            self.ocr_engine = PaddleOCR(use_angle_cls=False, lang=lang, use_gpu=use_gpu, show_log=False)
            self.logger.info(f"PaddleOCR initialized successfully (GPU: {use_gpu}, Lang: {lang})")
        except Exception as e:
            self.logger.critical(f"Failed to initialize PaddleOCR: {e}")
            raise

    def extract_plate_text(self, plate_crop: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
        """
        Performs OCR on a specific image crop of a license plate.
        Returns the text candidate with the highest confidence.
        
        Args:
            plate_crop (np.ndarray): The BGR image array of the localized plate.
            
        Returns:
            Tuple[Optional[str], Optional[float]]: The extracted text and its confidence score.
                Returns (None, None) if extraction fails or no text is found.
        """
        if plate_crop is None or plate_crop.size == 0:
            self.logger.debug("Received empty image array for OCR.")
            return None, None

        try:
            # Execute inference
            results = self.ocr_engine.ocr(plate_crop, cls=False)
            
            # PaddleOCR returns a nested list.
            # results[0] contains the inferences for the first image in the batch.
            if not results or not results[0]:
                return None, None

            best_text: str = ""
            best_confidence: float = 0.0

            # Iterate through all detected text boxes in the crop
            for line in results[0]:
                if line and len(line) == 2:
                    text, conf = line[1]  # line[1] is a tuple of (text, confidence)
                    if conf > best_confidence:
                        best_text = text
                        best_confidence = conf

            # Filter out low-confidence reads or empty strings
            if best_confidence >= self.confidence_threshold and best_text.strip():
                # Clean alphanumeric characters only (remove special chars)
                clean_text = ''.join(e for e in best_text if e.isalnum()).upper()
                return clean_text, best_confidence
            
            return None, None

        except Exception as e:
            self.logger.error(f"OCR Extraction failed: {e}")
            return None, None


class AttributeExtractor:
    """
    Determines categorical visual features of a vehicle, specifically 
    mapping object detection class IDs to strings, and estimating the vehicle's color.
    """

    # COCO Class ID mappings predefined by YOLO model
    CLASS_MAP: Dict[int, str] = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck"
    }

    # HSV Color boundaries for dominant color extraction
    # Format: "color_name": [[lower_bound_1, upper_bound_1], [lower_bound_2, upper_bound_2]]
    COLOR_RANGES = {
        "white":  [[np.array([0, 0, 200]), np.array([179, 30, 255])]],
        "black":  [[np.array([0, 0, 0]), np.array([179, 255, 50])]],
        "gray":   [[np.array([0, 0, 50]), np.array([179, 40, 200])]],
        "silver": [[np.array([0, 0, 150]), np.array([179, 40, 230])]], # Slightly brighter/tighter gray
        "blue":   [[np.array([90, 50, 50]), np.array([130, 255, 255])]],
        "green":  [[np.array([35, 50, 50]), np.array([85, 255, 255])]],
        "yellow": [[np.array([15, 50, 50]), np.array([35, 255, 255])]],
        # Red crosses the 0/180 degree boundary in OpenCV HSV, requiring two masks
        "red":    [[np.array([0, 50, 50]), np.array([10, 255, 255])], 
                   [np.array([160, 50, 50]), np.array([179, 255, 255])]]
    }

    def __init__(self):
        """Initializes the attribute extractor logic."""
        self.logger = logging.getLogger(self.__class__.__name__)

    def extract_vehicle_type(self, class_id: int) -> Optional[str]:
        """
        Maps a numeric class ID to a human-readable vehicle type.
        
        Args:
            class_id (int): The integer class ID from the detection model.
            
        Returns:
            Optional[str]: The vehicle type string, or None if unknown.
        """
        return self.CLASS_MAP.get(class_id, None)

    def extract_vehicle_color(self, vehicle_crop: np.ndarray) -> Optional[str]:
        """
        Determines the dominant color of the vehicle crop using HSV thresholding.
        Calculates a center-weighted region to avoid background noise.
        
        Args:
            vehicle_crop (np.ndarray): BGR image array of the vehicle.
            
        Returns:
            Optional[str]: The name of the dominant color, or None if extraction fails.
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None

        try:
            # Crop the center 60% of the image to ignore the road/background
            h, w = vehicle_crop.shape[:2]
            margin_h, margin_w = int(h * 0.2), int(w * 0.2)
            roi = vehicle_crop[margin_h:h-margin_h, margin_w:w-margin_w]
            
            if roi.size == 0:
                roi = vehicle_crop # Fallback to full image if too small

            # Convert to HSV color space
            hsv_image = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            max_pixels = 0
            dominant_color = None

            # Iterate through defined ranges and count matching pixels
            for color_name, boundaries in self.COLOR_RANGES.items():
                mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
                
                # Combine masks if multiple boundaries exist (e.g., Red)
                for (lower, upper) in boundaries:
                    temp_mask = cv2.inRange(hsv_image, lower, upper)
                    mask = cv2.bitwise_or(mask, temp_mask)

                pixel_count = cv2.countNonZero(mask)
                
                if pixel_count > max_pixels:
                    max_pixels = pixel_count
                    dominant_color = color_name

            # Require at least 5% of the ROI to match a color to return a confident result
            min_required_pixels = int((roi.shape[0] * roi.shape[1]) * 0.05)
            if max_pixels > min_required_pixels:
                return dominant_color
            
            return None

        except Exception as e:
            self.logger.error(f"Color Extraction failed: {e}")
            return None