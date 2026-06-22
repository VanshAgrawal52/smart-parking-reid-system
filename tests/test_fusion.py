"""
tests/test_fusion.py

Unit tests for the decision-level multi-modal vehicle fusion engine.
Tests OCR similarity (Levenshtein distance), attribute matching, 
and overall identity assignment thresholds.
"""

import pytest
from datetime import datetime
from core.fusion import (
    VehicleFusionEngine,
    VehicleObservation,
    ExistingVehicleRecord,
    FusionResult
)

@pytest.fixture
def fusion_engine():
    """
    Provides a configured VehicleFusionEngine instance for testing.
    Uses default weights: Plate=0.70, Color=0.15, Type=0.15, Threshold=0.75.
    """
    return VehicleFusionEngine()

@pytest.fixture
def base_observation():
    """Provides a baseline observation for reuse in tests."""
    return VehicleObservation(
        plate_text="RJ14AB1234",
        plate_confidence=0.95,
        vehicle_color="white",
        vehicle_type="car",
        timestamp=datetime.utcnow()
    )

def test_exact_plate_match(fusion_engine, base_observation):
    """
    Test that an exact match on all attributes yields a 1.0 confidence score
    and successfully matches the vehicle.
    """
    records = [
        ExistingVehicleRecord(
            vehicle_id=1,
            plate_text="RJ14AB1234",
            vehicle_color="white",
            vehicle_type="car"
        )
    ]
    
    result: FusionResult = fusion_engine.match_vehicle(base_observation, records)
    
    assert result.matched is True
    assert result.vehicle_id == 1
    assert result.confidence_score == 1.0

def test_small_ocr_error(fusion_engine):
    """
    Test that a minor OCR mistake (e.g., reading 'B' as '8') still yields
    a high enough similarity score to cross the 0.75 match threshold when 
    color and type match perfectly.
    """
    # Observation has '8' instead of 'B'
    observation = VehicleObservation(
        plate_text="RJ14A81234",
        plate_confidence=0.85,
        vehicle_color="white",
        vehicle_type="car",
        timestamp=datetime.utcnow()
    )
    
    records = [
        ExistingVehicleRecord(
            vehicle_id=2,
            plate_text="RJ14AB1234",  # Actual plate
            vehicle_color="white",
            vehicle_type="car"
        )
    ]
    
    result = fusion_engine.match_vehicle(observation, records)
    
    # Levenshtein distance is 1. String length is 10. 
    # Plate similarity = 0.90. Score = (0.90 * 0.70) + 0.15 + 0.15 = 0.93
    assert result.matched is True
    assert result.vehicle_id == 2
    assert result.confidence_score >= 0.90

def test_completely_different_plates(fusion_engine, base_observation):
    """
    Test that completely different plates fail to match, even if the 
    color and vehicle type are identical.
    """
    records = [
        ExistingVehicleRecord(
            vehicle_id=3,
            plate_text="DL01XY9999",
            vehicle_color="white",  # Same color
            vehicle_type="car"      # Same type
        )
    ]
    
    result = fusion_engine.match_vehicle(base_observation, records)
    
    # Plate weight is 0.70. Maximum score with perfect attributes but 0 plate match is 0.30.
    # 0.30 < 0.75 threshold.
    assert result.matched is False
    assert result.vehicle_id is None
    assert result.confidence_score < 0.75

def test_color_similarity(fusion_engine):
    """Test the color similarity evaluation logic, including case insensitivity."""
    assert fusion_engine.color_similarity("white", "white") == 1.0
    assert fusion_engine.color_similarity("White", "white") == 1.0  # Case check
    assert fusion_engine.color_similarity(" white ", "white") == 1.0 # Whitespace check
    assert fusion_engine.color_similarity("white", "black") == 0.0
    assert fusion_engine.color_similarity(None, "white") == 0.0

def test_type_similarity(fusion_engine):
    """Test the vehicle type similarity evaluation logic."""
    assert fusion_engine.type_similarity("car", "car") == 1.0
    assert fusion_engine.type_similarity("CAR", "car") == 1.0
    assert fusion_engine.type_similarity("car", "truck") == 0.0
    assert fusion_engine.type_similarity("car", None) == 0.0

def test_match_vehicle_empty_database(fusion_engine, base_observation):
    """
    Test that comparing an observation against an empty database safely 
    returns a False match rather than throwing an exception.
    """
    empty_records = []
    
    result = fusion_engine.match_vehicle(base_observation, empty_records)
    
    assert result.matched is False
    assert result.vehicle_id is None
    assert result.confidence_score == 0.0

def test_plate_similarity_edge_cases(fusion_engine):
    """Test the Levenshtein distance plate matcher with edge cases."""
    # Exact match
    assert fusion_engine.plate_similarity("ABC123", "ABC123") == 1.0
    # Empty string vs string
    assert fusion_engine.plate_similarity("", "ABC123") == 0.0
    # None values
    assert fusion_engine.plate_similarity(None, "ABC123") == 0.0
    assert fusion_engine.plate_similarity(None, None) == 0.0
    # Completely different lengths
    # Dist(A, ABCD) = 3. MaxLen = 4. Sim = 1 - (3/4) = 0.25
    assert fusion_engine.plate_similarity("A", "ABCD") == 0.25

def test_low_ocr_confidence_penalty(fusion_engine):
    """
    Test that low OCR confidence reduces the overall match score.
    """

    observation = VehicleObservation(
        plate_text="RJ14AB1234",
        plate_confidence=0.20,
        vehicle_color="white",
        vehicle_type="car",
        timestamp=datetime.utcnow()
    )

    records = [
        ExistingVehicleRecord(
            vehicle_id=10,
            plate_text="RJ14AB1234",
            vehicle_color="white",
            vehicle_type="car"
        )
    ]

    score = fusion_engine.compute_match_score(
        observation,
        records[0]
    )

    assert score < 1.0