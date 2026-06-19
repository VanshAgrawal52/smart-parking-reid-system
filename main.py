"""
main.py

Central orchestration layer for the Intelligent Multi-Camera Parking Management System.
Initializes configuration, instantiates all core modules, and runs the continuous 
video processing and database logging loop.
"""

import sys
import yaml
import time
import signal
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

# Database
from database.operations import DatabaseManager

# Core Modules
from core.perception import CameraStream, PerceptionEngine, Detection
from core.extractors import OCRExtractor, AttributeExtractor, VehicleAttributes
from core.fusion import VehicleFusionEngine, VehicleObservation, ExistingVehicleRecord
from core.spatial import SpatialEngine
from core.spatial import ParkingSlot as SpatialParkingSlot


class SmartParkingSystem:
    """
    Main system orchestrator linking Perception, Extraction, Fusion, Spatial, and Database layers.
    """

    def __init__(self, config_path: str = "config/params.yaml"):
        """
        Initializes the system by loading the configuration and setting up logging.
        
        Args:
            config_path (str): Path to the YAML configuration file.
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = self._load_config()
        self._setup_logging()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.is_running = False
        self.frame_counter = 0
        
        # Component placeholders
        self.db: Optional[DatabaseManager] = None
        self.stream: Optional[CameraStream] = None
        self.perception: Optional[PerceptionEngine] = None
        self.ocr: Optional[OCRExtractor] = None
        self.attributes: Optional[AttributeExtractor] = None
        self.fusion: Optional[VehicleFusionEngine] = None
        self.spatial: Optional[SpatialEngine] = None
        
        # Register shutdown signals
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _load_config(self) -> Dict[str, Any]:
        """Loads the YAML configuration file safely."""
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"CRITICAL: Failed to load config from {self.config_path}: {e}")
            sys.exit(1)

    def _setup_logging(self) -> None:
        """Configures the global Python logging mechanism based on config/params.yaml."""
        log_cfg = self.config.get("logging", {})
        log_level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
        log_format = log_cfg.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.StreamHandler(sys.stdout)
                # Note: File rotation handlers can be added here using log_cfg['file_path']
            ]
        )

    def _signal_handler(self, sig: int, frame: Any) -> None:
        """Handles termination signals (Ctrl+C, kill) to trigger a graceful shutdown."""
        self.logger.info(f"Received shutdown signal ({sig}). Initiating graceful shutdown...")
        self.is_running = False

    def initialize_components(self) -> None:
        """
        Instantiates all functional modules using settings defined in the config.
        Terminates the program if any critical component fails to initialize.
        """
        self.logger.info("Initializing system components...")
        try:
            # 1. Database Initialization
            db_cfg = self.config["database"]
            self.db = DatabaseManager(db_url=db_cfg["connection_string"])
            
            # 2. Perception & Camera Initialization
            # NOTE: Currently processing a single camera as per requirements constraint
            cam_cfg = self.config["cameras"][0]
            self.stream = CameraStream(source=cam_cfg["rtsp_url"])
            
            perc_cfg = self.config["perception"]["vehicle_detection"]
            trk_cfg = self.config["perception"]["tracking"]
            self.perception = PerceptionEngine(
                model_path=perc_cfg["model_path"],
                conf_thresh=perc_cfg["confidence_threshold"],
                iou_thresh=perc_cfg["nms_iou_threshold"],
                device=perc_cfg.get("device", "cpu"),
                tracker_config="bytetrack.yaml" # Assuming default provided by ultralytics
            )
            
            # 3. Extraction Initialization
            ext_cfg = self.config["extractors"]
            self.ocr = OCRExtractor(
                lang=ext_cfg["ocr"]["language"],
                use_gpu=ext_cfg["ocr"]["use_gpu"],
                confidence_threshold=ext_cfg["ocr"]["confidence_threshold"]
            )
            self.attributes = AttributeExtractor()
            
            # 4. Fusion Initialization
            fusion_cfg = self.config["fusion_engine"]
            self.fusion = VehicleFusionEngine(
                match_threshold=fusion_cfg["match_threshold"],
                weight_plate=fusion_cfg["weights"]["plate_text"],
                weight_color=fusion_cfg["weights"]["color"],
                weight_type=fusion_cfg["weights"]["vehicle_type"]
            )
            
            # 5. Spatial Initialization
            spatial_cfg = self.config["spatial"]["occupancy"]
            self.spatial = SpatialEngine(iou_threshold=spatial_cfg["parked_iou_threshold"])
            
            self.logger.info("All components initialized successfully.")
            
        except Exception as e:
            self.logger.critical(f"System initialization failed: {e}")
            self.shutdown()
            sys.exit(1)

    def process_frame(self, frame) -> None:
        """
        Executes the main pipeline: Detect -> Extract -> Fuse -> Spatial Map -> Database Log.
        
        Args:
            frame: Raw BGR NumPy array from the video stream.
        """
        # 1. Perception Layer
        detections = self.perception.detect_and_track(frame)
        if not detections:
            return

        # 2. Fetch Active Database State for Fusion
        active_db_vehicles = self.db.get_active_vehicles()
        existing_records: List[ExistingVehicleRecord] = [
            ExistingVehicleRecord(
                vehicle_id=v.id,
                plate_text=v.plate_string,
                vehicle_color=v.attribute_color,
                vehicle_type=v.attribute_class
            ) for v in active_db_vehicles
        ]

        # 3. Process Each Detected Vehicle
        for det in detections:
            # TODO: 
            # Crop the vehicle from the frame using det.bbox
            # vehicle_crop = frame[int(det.bbox[1]):int(det.bbox[3]), int(det.bbox[0]):int(det.bbox[2])]
            # For brevity/placeholder, assuming crop extraction is successful
            # Extract actual vehicle crop from frame using det.bbox.
            # Placeholder currently disables OCR and color extraction.
            vehicle_crop = None 
            
            # Extract Attributes
            plate_text, plate_conf = self.ocr.extract_plate_text(vehicle_crop) if self.ocr else (None, 0.0)
            color = self.attributes.extract_vehicle_color(vehicle_crop) if self.attributes else None
            v_type = self.attributes.extract_vehicle_type(det.class_id) if self.attributes else None
            
            observation = VehicleObservation(
                plate_text=plate_text,
                plate_confidence=plate_conf,
                vehicle_color=color,
                vehicle_type=v_type,
                timestamp=datetime.utcnow()
            )

            # 4. Identity Fusion
            fusion_result = self.fusion.match_vehicle(observation, existing_records)
            
            if fusion_result.matched and fusion_result.vehicle_id is not None:
                # Vehicle known -> Update status
                self.db.update_vehicle_status(fusion_result.vehicle_id, is_active=True)
            else:
                # Vehicle unknown -> Register new
                # TODO: Extract OSNet Vector (ReID module not yet built, passing placeholder)
                vector_placeholder = "0.0,0.0,0.0"
                new_veh_id = self.db.create_vehicle(
                    plate_string=plate_text,
                    confidence_score=plate_conf,
                    vector_embedding=vector_placeholder,
                    attribute_color=color,
                    attribute_class=v_type
                )
                
                # Create initial parking log (entry)
                # Assuming camera_id=1 for now. 
                if new_veh_id:
                    self.db.create_parking_log(camera_id=1, vehicle_id=new_veh_id, slot_id=None)

        # 5. Spatial Engine (Occupancy Mapping)
        # TODO: Fetch slots from DB. database/operations.py currently doesn't have `get_all_slots`.
        # Placeholder mapping:
        spatial_slots: List[SpatialParkingSlot] = [] 
        
        occupancy_results = self.spatial.determine_occupancy(detections, spatial_slots)
        
        # 6. Database Update
        for res in occupancy_results:
            # Convert string ID back to DB integer ID (if applicable) and update
            # self.db.update_slot_status(slot_id=int(res.slot_id), status="occupied" if res.occupied else "vacant")
            pass

    def run(self) -> None:
        """
        The main infinite loop that reads frames and controls the execution cadence.
        Handles frame-skipping logic to optimize CPU/GPU utilization.
        """
        self.logger.info("Starting Main Processing Loop...")
        self.is_running = True
        
        cam_cfg = self.config["cameras"][0]
        skip_frames = cam_cfg.get("skip_frames", 0)

        while self.is_running:
            success, frame = self.stream.read_frame()
            if not success:
                self.logger.warning("Failed to read frame. Retrying...")
                time.sleep(1)
                continue
                
            self.frame_counter += 1
            
            # Optimization: Skip frames to maintain real-time performance
            if self.frame_counter % (skip_frames + 1) != 0:
                continue

            try:
                self.process_frame(frame)
            except Exception as e:
                self.logger.error(f"Error processing frame {self.frame_counter}: {e}", exc_info=True)
                
        self.shutdown()

    def shutdown(self) -> None:
        """Gracefully closes all streams, flushes buffers, and releases resources."""
        self.logger.info("Commencing system shutdown sequence...")
        self.is_running = False
        
        if self.stream:
            self.stream.release()
            self.logger.info("Camera streams released.")
            
        # SQLAlchemy engine disposes connections automatically, but custom close logic goes here
        self.logger.info("System shutdown complete.")


if __name__ == "__main__":
    system = SmartParkingSystem(config_path="config/params.yaml")
    system.initialize_components()
    system.run()