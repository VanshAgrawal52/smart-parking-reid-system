"""
core/fusion.py

Provides the decision-level multi-modal fusion engine.
This module is responsible for cross-referencing new vehicle observations against 
historical vehicle records to maintain a stable, global vehicle identity across 
multiple cameras and timeframes.
"""

import logging
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class VehicleObservation:
    """
    Represents a new, incoming vehicle detection and its extracted features.
    
    Attributes:
        plate_text (Optional[str]): The alphanumeric string from the license plate.
        plate_confidence (Optional[float]): The confidence score of the OCR read.
        vehicle_color (Optional[str]): The dominant localized color of the vehicle.
        vehicle_type (Optional[str]): The classification (e.g., 'car', 'truck').
        timestamp (datetime): When the observation occurred.
    """
    plate_text: Optional[str]
    plate_confidence: Optional[float]
    vehicle_color: Optional[str]
    vehicle_type: Optional[str]
    timestamp: datetime


@dataclass
class ExistingVehicleRecord:
    """
    Represents a known vehicle entity from the system's tracking history.
    Used as an abstraction to avoid database module dependencies.
    
    Attributes:
        vehicle_id (int): The unique global identifier.
        plate_text (Optional[str]): The previously recorded license plate string.
        vehicle_color (Optional[str]): The previously recorded color.
        vehicle_type (Optional[str]): The previously recorded type.
    """
    vehicle_id: int
    plate_text: Optional[str]
    vehicle_color: Optional[str]
    vehicle_type: Optional[str]


@dataclass
class FusionResult:
    """
    Represents the output of the fusion matching process.
    
    Attributes:
        matched (bool): True if a match exceeding the confidence threshold was found.
        vehicle_id (Optional[int]): The ID of the matched vehicle, or None if no match.
        confidence_score (float): The final weighted similarity score of the best match.
    """
    matched: bool
    vehicle_id: Optional[int]
    confidence_score: float


class VehicleFusionEngine:
    """
    Executes multi-modal feature fusion to assign persistent identities.
    Combines Levenshtein string similarity for OCR data and exact matching for attributes.
    """

    def __init__(self, 
                 match_threshold: float = 0.75, 
                 weight_plate: float = 0.70, 
                 weight_color: float = 0.15, 
                 weight_type: float = 0.15):
        """
        Initializes the Fusion Engine with specified weights and thresholds.
        
        Args:
            match_threshold (float): Minimum score required to confirm an identity match.
            weight_plate (float): Importance weight for license plate similarity.
            weight_color (float): Importance weight for vehicle color similarity.
            weight_type (float): Importance weight for vehicle type similarity.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.match_threshold = match_threshold
        
        # Normalize weights just in case they don't sum to 1.0
        total_weight = weight_plate + weight_color + weight_type
        self.weight_plate = weight_plate / total_weight
        self.weight_color = weight_color / total_weight
        self.weight_type = weight_type / total_weight
        
        self.logger.info(f"FusionEngine initialized. Threshold: {self.match_threshold}. "
                         f"Weights - Plate: {self.weight_plate:.2f}, "
                         f"Color: {self.weight_color:.2f}, Type: {self.weight_type:.2f}")

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Calculates the minimum number of single-character edits required to change one word into the other.
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
            
        return previous_row[-1]

    def plate_similarity(self, text1: Optional[str], text2: Optional[str]) -> float:
        """
        Calculates the similarity between two license plates using a normalized Levenshtein distance.
        
        Args:
            text1 (Optional[str]): First plate string.
            text2 (Optional[str]): Second plate string.
            
        Returns:
            float: Similarity score between 0.0 and 1.0.
        """
        if not text1 or not text2:
            return 0.0
            
        # Standardize for comparison
        t1, t2 = text1.strip().upper(), text2.strip().upper()
        
        if t1 == t2:
            return 1.0
            
        max_len = max(len(t1), len(t2))
        if max_len == 0:
            return 0.0
            
        distance = self._levenshtein_distance(t1, t2)
        
        # Normalize distance to a 0.0 - 1.0 similarity score
        similarity = 1.0 - (distance / max_len)
        return max(0.0, similarity)

    def color_similarity(self, color1: Optional[str], color2: Optional[str]) -> float:
        """
        Calculates similarity between vehicle colors.
        
        Args:
            color1 (Optional[str]): First color.
            color2 (Optional[str]): Second color.
            
        Returns:
            float: 1.0 for an exact match, 0.0 otherwise.
        """
        if not color1 or not color2:
            return 0.0
        return 1.0 if color1.strip().lower() == color2.strip().lower() else 0.0

    def type_similarity(self, type1: Optional[str], type2: Optional[str]) -> float:
        """
        Calculates similarity between vehicle classifications (e.g., 'car' vs 'truck').
        
        Args:
            type1 (Optional[str]): First vehicle type.
            type2 (Optional[str]): Second vehicle type.
            
        Returns:
            float: 1.0 for an exact match, 0.0 otherwise.
        """
        if not type1 or not type2:
            return 0.0
        return 1.0 if type1.strip().lower() == type2.strip().lower() else 0.0

    def compute_match_score(self, obs: VehicleObservation, record: ExistingVehicleRecord) -> float:
        """
        Computes the weighted multi-modal similarity score between an observation and a known record.
        
        Args:
            obs (VehicleObservation): The incoming vehicle detection.
            record (ExistingVehicleRecord): The historical vehicle entity.
            
        Returns:
            float: A unified confidence score between 0.0 and 1.0.
        """
        try:
            s_plate = self.plate_similarity(obs.plate_text, record.plate_text)
            plate_conf = obs.plate_confidence if obs.plate_confidence is not None else 0.0
            if plate_conf < 0.5:
                s_plate *= 0.5
                
            s_color = self.color_similarity(obs.vehicle_color, record.vehicle_color)
            s_type = self.type_similarity(obs.vehicle_type, record.vehicle_type)
            
            final_score = (self.weight_plate * s_plate) + \
                          (self.weight_color * s_color) + \
                          (self.weight_type * s_type)
                          
            return final_score
        except Exception as e:
            self.logger.error(f"Error computing match score for observation vs record {record.vehicle_id}: {e}")
            return 0.0

    def match_vehicle(self, observation: VehicleObservation, existing_records: List[ExistingVehicleRecord]) -> FusionResult:
        """
        Iterates over a pool of existing vehicle records to find the best identity match.
        
        Args:
            observation (VehicleObservation): The current vehicle feature payload.
            existing_records (List[ExistingVehicleRecord]): Pool of known vehicles to compare against.
            
        Returns:
            FusionResult: The structured outcome indicating match status and assigned ID.
        """
        if not existing_records:
            self.logger.debug("No existing records provided. Returning no match.")
            return FusionResult(matched=False, vehicle_id=None, confidence_score=0.0)

        best_score = 0.0
        best_match_id = None

        for record in existing_records:
            score = self.compute_match_score(observation, record)
            
            if score > best_score:
                best_score = score
                best_match_id = record.vehicle_id

        # Determine if the best score meets the required system threshold
        is_matched = best_score >= self.match_threshold
        
        if is_matched:
            self.logger.info(f"Successfully matched observation to Vehicle ID {best_match_id} (Score: {best_score:.2f})")
        else:
            self.logger.info(f"Failed to match observation. Highest score was {best_score:.2f} (Threshold: {self.match_threshold})")

        return FusionResult(
            matched=is_matched,
            vehicle_id=best_match_id if is_matched else None,
            confidence_score=best_score
        )