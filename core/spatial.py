"""
core/spatial.py

Handles spatial reasoning for the Intelligent Multi-Camera Parking Management System.
Responsible for converting logical vehicle bounding boxes and parking slot coordinates 
into mathematical polygons, computing their overlap, and determining real-time occupancy.
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any

import numpy as np
from shapely.geometry import Polygon, box
from shapely.errors import TopologicalError

# Assuming Detection is imported from perception module for type hinting
try:
    from core.perception import Detection
except ImportError:
    # Fallback dummy dataclass for standalone execution/type checking
    @dataclass
    class Detection:
        track_id: int
        bbox: List[float]
        confidence: float
        class_id: int
        timestamp: Any


@dataclass
class ParkingSlot:
    """
    Represents a physical parking space defined by a polygon in the camera frame.
    
    Attributes:
        slot_id (str | int): Unique identifier for the parking slot.
        polygon_points (List[Tuple[float, float]]): List of (x, y) coordinates defining the slot.
        status (str): Current state, e.g., 'vacant', 'occupied', 'unauthorized'.
    """
    slot_id: str | int
    polygon_points: List[Tuple[float, float]]
    status: str = "vacant"


@dataclass
class OccupancyResult:
    """
    Represents the spatial analysis outcome for a single parking slot.
    
    Attributes:
        slot_id (str | int): The evaluated parking slot.
        occupied (bool): True if a vehicle sufficiently overlaps the slot.
        track_id (Optional[int]): The ID of the vehicle occupying the slot, if any.
    """
    slot_id: str | int
    occupied: bool
    track_id: Optional[int]


class SpatialEngine:
    """
    Executes geometric computations to map tracked vehicles to physical parking infrastructure.
    """

    def __init__(self, iou_threshold: float = 0.30):
        """
        Initializes the spatial reasoning engine.
        
        Args:
            iou_threshold (float): Minimum Intersection over Union (IoU) overlap 
                                   required to classify a slot as occupied.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.iou_threshold = iou_threshold
        self.logger.info(f"SpatialEngine initialized with IoU threshold: {self.iou_threshold:.2f}")

    def create_polygon(self, points: List[Tuple[float, float]]) -> Optional[Polygon]:
        """
        Converts a list of (x, y) coordinates into a Shapely Polygon.
        
        Args:
            points (List[Tuple[float, float]]): The coordinate vertices of the parking slot.
            
        Returns:
            Optional[Polygon]: Valid Shapely Polygon, or None if creation fails.
        """
        if not points or len(points) < 3:
            self.logger.error("A polygon requires at least 3 points.")
            return None
            
        try:
            poly = Polygon(points)
            if not poly.is_valid:
                # Attempt to fix self-intersecting polygons (e.g., bow-tie shapes)
                poly = poly.buffer(0)
            return poly
        except Exception as e:
            self.logger.error(f"Failed to create polygon from points {points}: {e}")
            return None

    def bbox_to_polygon(self, bbox: List[float]) -> Optional[Polygon]:
        """
        Converts a [x1, y1, x2, y2] bounding box into a Shapely Polygon.
        
        Args:
            bbox (List[float]): Bounding box coordinates.
            
        Returns:
            Optional[Polygon]: Rectangular Shapely Polygon, or None if invalid.
        """
        if not bbox or len(bbox) != 4:
            self.logger.error("Invalid bounding box format. Expected [x1, y1, x2, y2].")
            return None
            
        try:
            x1, y1, x2, y2 = bbox
            # Shapely's box creates a rectangular polygon (minx, miny, maxx, maxy)
            return box(x1, y1, x2, y2)
        except Exception as e:
            self.logger.error(f"Failed to convert bbox {bbox} to polygon: {e}")
            return None

    def calculate_iou(self, poly1: Polygon, poly2: Polygon) -> float:
        """
        Calculates the Intersection over Union (IoU) of two polygons.
        
        Args:
            poly1 (Polygon): The first polygon (e.g., the parking slot).
            poly2 (Polygon): The second polygon (e.g., the vehicle footprint).
            
        Returns:
            float: IoU score between 0.0 and 1.0.
        """
        if not poly1 or not poly2 or not poly1.is_valid or not poly2.is_valid:
            return 0.0

        try:
            intersection_area = poly1.intersection(poly2).area
            union_area = poly1.union(poly2).area
            
            if union_area <= 0:
                return 0.0
                
            return float(intersection_area / union_area)
        except TopologicalError as te:
            self.logger.warning(f"Topological error during IoU calculation: {te}")
            return 0.0
        except Exception as e:
            self.logger.error(f"Unexpected error calculating IoU: {e}")
            return 0.0

    def determine_occupancy(self, detections: List[Detection], slots: List[ParkingSlot]) -> List[OccupancyResult]:
        """
        Evaluates all parking slots against current vehicle detections to determine occupancy.
        
        Args:
            detections (List[Detection]): Active vehicles in the current frame.
            slots (List[ParkingSlot]): Predefined parking slots for the camera view.
            
        Returns:
            List[OccupancyResult]: The occupancy status and matched track ID for every slot.
        """
        results: List[OccupancyResult] = []

        # Convert vehicle bounding boxes to polygons once to save compute
        vehicle_polys = []
        for det in detections:
            v_poly = self.bbox_to_polygon(det.bbox)
            if v_poly:
                vehicle_polys.append((det.track_id, v_poly))

        for slot in slots:
            slot_poly = self.create_polygon(slot.polygon_points)
            if not slot_poly:
                self.logger.warning(f"Skipping occupancy check for invalid slot {slot.slot_id}")
                continue

            max_iou = 0.0
            best_track_id = None

            # Find the vehicle that overlaps this slot the most
            for track_id, v_poly in vehicle_polys:
                iou = self.calculate_iou(slot_poly, v_poly)
                if iou > max_iou:
                    max_iou = iou
                    best_track_id = track_id

            # Apply the 0.30 threshold rule
            is_occupied = max_iou >= self.iou_threshold
            
            results.append(OccupancyResult(
                slot_id=slot.slot_id,
                occupied=is_occupied,
                track_id=best_track_id if is_occupied else None
            ))

        return results

    def update_slot_status(self, slots: List[ParkingSlot], results: List[OccupancyResult]) -> None:
        """
        Updates the string status of ParkingSlot objects in-place based on the evaluation results.
        
        Args:
            slots (List[ParkingSlot]): The slots to update.
            results (List[OccupancyResult]): The computed occupancy outcomes.
        """
        result_map = {res.slot_id: res for res in results}
        
        for slot in slots:
            res = result_map.get(slot.slot_id)
            if res:
                # Update status strings based on the boolean result
                old_status = slot.status
                slot.status = "occupied" if res.occupied else "vacant"
                
                if old_status != slot.status:
                    self.logger.debug(f"Slot {slot.slot_id} status changed: {old_status} -> {slot.status}")