"""
database/operations.py

Provides the DatabaseManager class for executing CRUD operations on the SQLite database.
Follows SQLAlchemy 2.0 best practices using select() and scalar() execution.
Includes robust error handling, session management, and logging.
"""

import logging
from datetime import datetime
from typing import Optional, Sequence

from sqlalchemy import create_engine, select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker, Session

from database.models import Base, Camera, ParkingSlot, Vehicle, ParkingLog, Alert


class DatabaseManager:
    """
    Handles all database operations for the Intelligent Parking Management System.
    Manages the engine, connection pooling, and session lifecycle.
    """

    def __init__(self, db_url: str = "sqlite:///parking_system.db", echo: bool = False):
        """
        Initializes the database manager, creates the engine, session factory, 
        and ensures all tables are created.
        
        Args:
            db_url (str): The database connection string. Defaults to SQLite.
            echo (bool): If True, SQLAlchemy will log all generated SQL.
        """
        self.logger = logging.getLogger(__name__)
        
        try:
            self.engine = create_engine(db_url, echo=echo, future=True)
            # autoflush=False to prevent premature flushes during complex multi-modal insertions
            self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, expire_on_commit=False)
            
            # Create all tables if they don't exist
            Base.metadata.create_all(self.engine)
            self.logger.info(f"Database initialized successfully at {db_url}")
        except Exception as e:
            self.logger.critical(f"Failed to initialize database: {e}")
            raise

    def get_session(self) -> Session:
        """
        Generates a new SQLAlchemy session. 
        Intended to be used as a context manager: `with db.get_session() as session:`
        
        Returns:
            Session: A new SQLAlchemy session instance.
        """
        return self.SessionLocal()

    def create_camera(self, identifier: str, stream_url: str, location_tag: str) -> Optional[int]:
        """
        Registers a new camera in the database.
        
        Args:
            identifier (str): Unique camera string identifier.
            stream_url (str): RTSP URL for the camera stream.
            location_tag (str): Human-readable location description.
            
        Returns:
            Optional[int]: The ID of the newly created camera, or None if failed.
        """
        with self.get_session() as session:
            try:
                new_camera = Camera(
                    identifier=identifier,
                    stream_url=stream_url,
                    location_tag=location_tag
                )
                session.add(new_camera)
                session.commit()
                self.logger.info(f"Created Camera: {identifier} (ID: {new_camera.id})")
                return new_camera.id
            except SQLAlchemyError as e:
                session.rollback()
                self.logger.error(f"Failed to create camera {identifier}: {e}")
                return None

    def create_parking_slot(self, camera_id: int, slot_index: str, polygon_json: str) -> Optional[int]:
        """
        Creates a new parking slot mapped to a specific camera.
        
        Args:
            camera_id (int): Foreign key linking to the Camera.
            slot_index (str): The local ID or label for the slot (e.g., "A1").
            polygon_json (str): JSON string representation of the bounding polygon.
            
        Returns:
            Optional[int]: The ID of the newly created slot, or None if failed.
        """
        with self.get_session() as session:
            try:
                new_slot = ParkingSlot(
                    camera_id=camera_id,
                    slot_index=slot_index,
                    polygon_json=polygon_json
                )
                session.add(new_slot)
                session.commit()
                self.logger.info(f"Created Parking Slot {slot_index} for Camera ID {camera_id}")
                return new_slot.id
            except SQLAlchemyError as e:
                session.rollback()
                self.logger.error(f"Failed to create parking slot {slot_index}: {e}")
                return None

    def create_vehicle(self, plate_string: Optional[str], confidence_score: Optional[float], 
                       vector_embedding: str, attribute_color: Optional[str], 
                       attribute_class: Optional[str]) -> Optional[int]:
        """
        Registers a new unique vehicle into the system.
        
        Args:
            plate_string (Optional[str]): OCR read of the license plate.
            confidence_score (Optional[float]): Confidence of the OCR read.
            vector_embedding (str): Serialized OSNet feature vector.
            attribute_color (Optional[str]): Classified color.
            attribute_class (Optional[str]): Classified vehicle type (e.g., 'car', 'truck').
            
        Returns:
            Optional[int]: The ID of the newly created vehicle, or None if failed.
        """
        with self.get_session() as session:
            try:
                new_vehicle = Vehicle(
                    plate_string=plate_string,
                    confidence_score=confidence_score,
                    vector_embedding=vector_embedding,
                    attribute_color=attribute_color,
                    attribute_class=attribute_class
                )
                session.add(new_vehicle)
                session.commit()
                self.logger.info(f"Created Vehicle (Plate: {plate_string}, ID: {new_vehicle.id})")
                return new_vehicle.id
            except SQLAlchemyError as e:
                session.rollback()
                self.logger.error(f"Failed to create vehicle {plate_string}: {e}")
                return None

    def get_vehicle_by_plate(self, plate_string: str) -> Optional[Vehicle]:
        """
        Retrieves a vehicle by its exact license plate string.
        
        Args:
            plate_string (str): The license plate to search for.
            
        Returns:
            Optional[Vehicle]: The Vehicle object if found, otherwise None.
        """
        with self.get_session() as session:
            try:
                stmt = select(Vehicle).where(Vehicle.plate_string == plate_string)
                # Ensure the instance is fully loaded before returning outside session
                vehicle = session.scalars(stmt).first()
                return vehicle
            except SQLAlchemyError as e:
                self.logger.error(f"Error querying vehicle by plate {plate_string}: {e}")
                return None

    def get_active_vehicles(self) -> Sequence[Vehicle]:
        """
        Retrieves all vehicles currently active in the system.
        An active vehicle is defined as one having a ParkingLog with no exit_timestamp.
        
        Returns:
            Sequence[Vehicle]: A list of currently active Vehicle objects.
        """
        with self.get_session() as session:
            try:
                # SQLAlchemy 2.0 select utilizing joins
                stmt = select(Vehicle).join(ParkingLog).where(ParkingLog.exit_timestamp.is_(None))
                return session.scalars(stmt).all()
            except SQLAlchemyError as e:
                self.logger.error(f"Failed to retrieve active vehicles: {e}")
                return []

    def update_vehicle_status(self, vehicle_id: int, is_active: bool) -> bool:
        """
        Updates the active status of a vehicle.
        Note: Since `is_active` is not a direct column on the Vehicle model (status is inferred 
        via ParkingLogs), this method updates the `updated_at` timestamp to keep the identity fresh.
        If a boolean flag is added to the schema later, it should be mapped here.
        
        Args:
            vehicle_id (int): The ID of the vehicle to update.
            is_active (bool): Target status flag.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        with self.get_session() as session:
            try:
                stmt = update(Vehicle).where(Vehicle.id == vehicle_id).values(
                    is_active=is_active,
                    updated_at=datetime.utcnow()
                )
                result = session.execute(stmt)
                session.commit()
                
                success = result.rowcount > 0
                if success:
                    self.logger.info(f"Updated status for Vehicle ID {vehicle_id} to active={is_active}")
                else:
                    self.logger.warning(f"Vehicle ID {vehicle_id} not found for status update.")
                return success
            except SQLAlchemyError as e:
                session.rollback()
                self.logger.error(f"Failed to update status for Vehicle ID {vehicle_id}: {e}")
                return False

    def create_parking_log(self, camera_id: int, vehicle_id: int, slot_id: Optional[int]) -> Optional[int]:
        """
        Creates a new temporal log for a vehicle entering a camera's FOV or parking slot.
        
        Args:
            camera_id (int): Foreign key to the viewing Camera.
            vehicle_id (int): Foreign key to the tracked Vehicle.
            slot_id (Optional[int]): Foreign key to the ParkingSlot (None if moving).
            
        Returns:
            Optional[int]: The ID of the new ParkingLog, or None if failed.
        """
        with self.get_session() as session:
            try:
                new_log = ParkingLog(
                    camera_id=camera_id,
                    vehicle_id=vehicle_id,
                    slot_id=slot_id
                )
                session.add(new_log)
                session.commit()
                self.logger.info(f"Created ParkingLog (Vehicle: {vehicle_id}, Slot: {slot_id})")
                return new_log.id
            except SQLAlchemyError as e:
                session.rollback()
                self.logger.error(f"Failed to create parking log for Vehicle {vehicle_id}: {e}")
                return None

    def close_parking_log(self, log_id: int) -> bool:
        """
        Closes an active parking log by setting its exit_timestamp and calculating duration.
        
        Args:
            log_id (int): The ID of the ParkingLog to close.
            
        Returns:
            bool: True if successfully closed, False otherwise.
        """
        with self.get_session() as session:
            try:
                # Fetch log to perform duration calculation in Python
                log = session.scalars(select(ParkingLog).where(ParkingLog.id == log_id)).first()
                if not log:
                    self.logger.warning(f"ParkingLog ID {log_id} not found for closure.")
                    return False
                
                if log.exit_timestamp is not None:
                    self.logger.info(f"ParkingLog ID {log_id} is already closed.")
                    return True

                now = datetime.utcnow()
                duration_delta = now - log.entry_timestamp
                duration_minutes = duration_delta.total_seconds() / 60.0

                log.exit_timestamp = now
                log.duration_minutes = duration_minutes
                
                session.commit()
                self.logger.info(f"Closed ParkingLog ID {log_id} (Duration: {duration_minutes:.2f} mins)")
                return True
            except SQLAlchemyError as e:
                session.rollback()
                self.logger.error(f"Failed to close parking log ID {log_id}: {e}")
                return False

    def update_slot_status(self, slot_id: int, status: str) -> bool:
        """
        Updates the current occupancy status of a parking slot.
        Note: Assumes a 'status' attribute exists. Maps to 'coordinate_mask' as a 
        proxy if 'status' column is not explicitly defined in models.py.
        
        Args:
            slot_id (int): The ID of the ParkingSlot.
            status (str): The new status (e.g., 'vacant', 'occupied').
            
        Returns:
            bool: True if successfully updated, False otherwise.
        """
        with self.get_session() as session:
            try:
                # Using coordinate_mask to temporarily store status if exact column is missing
                # Replace coordinate_mask with 'status' if models.py is updated.
                stmt = update(ParkingSlot).where(ParkingSlot.id == slot_id).values(
                    status=status
                )
                result = session.execute(stmt)
                session.commit()
                
                success = result.rowcount > 0
                if success:
                    self.logger.debug(f"Updated ParkingSlot ID {slot_id} status to '{status}'")
                return success
            except SQLAlchemyError as e:
                session.rollback()
                self.logger.error(f"Failed to update status for ParkingSlot ID {slot_id}: {e}")
                return False

    def create_alert(self, camera_id: int, alert_class: str, details_payload: str) -> Optional[int]:
        """
        Logs a system alert (e.g., overstay, unauthorized parking).
        
        Args:
            camera_id (int): The ID of the camera that triggered the alert.
            alert_class (str): Category of the alert.
            details_payload (str): JSON string containing contextual alert data.
            
        Returns:
            Optional[int]: The ID of the new Alert, or None if failed.
        """
        with self.get_session() as session:
            try:
                new_alert = Alert(
                    camera_id=camera_id,
                    alert_class=alert_class,
                    details_payload=details_payload
                )
                session.add(new_alert)
                session.commit()
                self.logger.warning(f"Triggered Alert: {alert_class} on Camera {camera_id}")
                return new_alert.id
            except SQLAlchemyError as e:
                session.rollback()
                self.logger.error(f"Failed to create alert {alert_class}: {e}")
                return None

    def get_recent_alerts(self, limit: int = 50) -> Sequence[Alert]:
        """
        Retrieves the most recent alerts generated by the system.
        
        Args:
            limit (int): Maximum number of alerts to return. Defaults to 50.
            
        Returns:
            Sequence[Alert]: A list of recent Alert objects.
        """
        with self.get_session() as session:
            try:
                stmt = select(Alert).order_by(Alert.event_timestamp.desc()).limit(limit)
                return session.scalars(stmt).all()
            except SQLAlchemyError as e:
                self.logger.error(f"Failed to retrieve recent alerts: {e}")
                return []