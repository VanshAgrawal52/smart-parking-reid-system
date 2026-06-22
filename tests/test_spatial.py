"""
tests/test_spatial.py

Unit tests for the spatial reasoning engine.
Tests polygon creation, bounding box conversion, mathematical Intersection over Union (IoU),
and occupancy logic based on vehicle overlaps.
"""

import pytest
from datetime import datetime
from shapely.geometry import Polygon

from core.spatial import SpatialEngine, ParkingSlot, OccupancyResult
from core.perception import Detection


@pytest.fixture
def spatial_engine():
    """
    Provides a configured SpatialEngine instance for testing.
    Uses default IoU threshold of 0.30.
    """
    return SpatialEngine(iou_threshold=0.30)


@pytest.fixture
def base_parking_slot():
    """Provides a standard 100x100 square parking slot."""
    return ParkingSlot(
        slot_id="A1",
        polygon_points=[(0, 0), (100, 0), (100, 100), (0, 100)]
    )


def test_valid_polygon_creation(spatial_engine):
    """Test that a valid list of coordinates successfully generates a Shapely Polygon."""
    points = [(0, 0), (100, 0), (100, 100), (0, 100)]
    poly = spatial_engine.create_polygon(points)
    
    assert poly is not None
    assert isinstance(poly, Polygon)
    assert poly.is_valid
    assert poly.area == 10000.0


def test_invalid_polygon_creation(spatial_engine):
    """Test that providing insufficient points (less than 3) fails gracefully and returns None."""
    points = [(0, 0), (100, 100)]  # Only a line, not a polygon
    poly = spatial_engine.create_polygon(points)
    
    assert poly is None


def test_bbox_to_polygon_conversion(spatial_engine):
    """Test that a standard [x1, y1, x2, y2] bounding box converts to a valid rectangular Polygon."""
    bbox = [0, 0, 100, 100]
    poly = spatial_engine.bbox_to_polygon(bbox)
    
    assert poly is not None
    assert isinstance(poly, Polygon)
    assert poly.is_valid
    assert poly.area == 10000.0


def test_iou_identical_polygons(spatial_engine):
    """Test that computing IoU for two perfectly overlapping polygons yields 1.0."""
    poly1 = spatial_engine.bbox_to_polygon([0, 0, 100, 100])
    poly2 = spatial_engine.bbox_to_polygon([0, 0, 100, 100])
    
    iou = spatial_engine.calculate_iou(poly1, poly2)
    
    assert iou == 1.0


def test_iou_no_overlap(spatial_engine):
    """Test that computing IoU for two completely separate polygons yields 0.0."""
    poly1 = spatial_engine.bbox_to_polygon([0, 0, 100, 100])
    poly2 = spatial_engine.bbox_to_polygon([200, 200, 300, 300])
    
    iou = spatial_engine.calculate_iou(poly1, poly2)
    
    assert iou == 0.0


def test_occupancy_detection_overlapping(spatial_engine, base_parking_slot):
    """
    Test that a detection with a bounding box heavily overlapping a slot 
    triggers a True occupancy status and maps the track_id.
    """
    slots = [base_parking_slot]
    
    # Vehicle perfectly inside the slot
    detections = [
        Detection(
            track_id=42,
            bbox=[10, 10, 90, 90],
            confidence=0.95,
            class_id=2,  # Car
            timestamp=datetime.utcnow()
        )
    ]
    
    results = spatial_engine.determine_occupancy(detections, slots)
    
    assert len(results) == 1
    assert results[0].slot_id == "A1"
    assert results[0].occupied is True
    assert results[0].track_id == 42


def test_occupancy_detection_empty(spatial_engine, base_parking_slot):
    """
    Test that providing no vehicle detections results in all parking slots 
    being marked as vacant.
    """
    slots = [base_parking_slot]
    detections = []  # No vehicles detected
    
    results = spatial_engine.determine_occupancy(detections, slots)
    
    assert len(results) == 1
    assert results[0].slot_id == "A1"
    assert results[0].occupied is False
    assert results[0].track_id is None