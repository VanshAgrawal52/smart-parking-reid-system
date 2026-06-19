"""
database/models.py

SQLAlchemy 2.0 declarative models for the Intelligent Multi-Camera Parking Management System.
Defines the schema for Cameras, Parking Slots, Vehicles, Parking Logs, and Alerts.
Designed for SQLite compatibility with seamless future migration to PostgreSQL.
"""

from datetime import datetime
from typing import List, Optional

from sqlalchemy import String, Float, DateTime, ForeignKey, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy declarative models."""
    pass


class Camera(Base):
    """
    Represents a physical CCTV/IP camera in the parking network.
    Maintains 1-to-Many relationships with Slots, Logs, and Alerts.
    """
    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    identifier: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    stream_url: Mapped[str] = mapped_column(String(255), nullable=False)
    location_tag: Mapped[str] = mapped_column(String(100), nullable=False)

    # Relationships
    slots: Mapped[List["ParkingSlot"]] = relationship(back_populates="camera", cascade="all, delete-orphan")
    logs: Mapped[List["ParkingLog"]] = relationship(back_populates="camera", cascade="all, delete-orphan")
    alerts: Mapped[List["Alert"]] = relationship(back_populates="camera", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Camera(id={self.id}, identifier='{self.identifier}', location='{self.location_tag}')>"


class ParkingSlot(Base):
    """
    Represents a discovered or predefined parking space mapped to a specific camera.
    """
    __tablename__ = "parking_slots"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"), index=True)
    slot_index: Mapped[str] = mapped_column(String(50), nullable=False)
    
    status: Mapped[str] = mapped_column(
        String(20),
        default="vacant",
        index=True
    )

    # Serialized JSON string storing parking slot polygon coordinates
    polygon_json: Mapped[str] = mapped_column(Text, nullable=False)
    coordinate_mask: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    camera: Mapped["Camera"] = relationship(back_populates="slots")
    logs: Mapped[List["ParkingLog"]] = relationship(back_populates="slot")

    def __repr__(self) -> str:
        return f"<ParkingSlot(id={self.id}, camera_id={self.camera_id}, slot_index='{self.slot_index}')>"


class Vehicle(Base):
    """
    Represents a unique vehicle tracked across the system. 
    Stores multi-modal fusion data (Plate, OSNet Embedding, Attributes).
    """
    __tablename__ = "vehicles"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    
    # Nullable because a vehicle might be tracked via embedding before OCR reads the plate
    plate_string: Mapped[Optional[str]] = mapped_column(String(20), unique=True, index=True, nullable=True)
    confidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Serialized JSON string containing Re-ID embedding vector
    vector_embedding: Mapped[str] = mapped_column(Text, nullable=False)
    
    attribute_color: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    attribute_class: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    is_active: Mapped[bool] = mapped_column(default=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    logs: Mapped[List["ParkingLog"]] = relationship(back_populates="vehicle", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Vehicle(id={self.id}, plate='{self.plate_string}', class='{self.attribute_class}')>"


class ParkingLog(Base):
    """
    Represents a temporal record of a vehicle occupying a specific space or zone.
    Functions as the core state-tracking entity for analytics.
    """
    __tablename__ = "parking_logs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"), index=True)
    vehicle_id: Mapped[int] = mapped_column(ForeignKey("vehicles.id", ondelete="CASCADE"), index=True)
    
    # Nullable in case the vehicle is in the FOV but not parked in a designated slot
    slot_id: Mapped[Optional[int]] = mapped_column(ForeignKey("parking_slots.id", ondelete="SET NULL"), index=True, nullable=True)

    entry_timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    exit_timestamp: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, index=True)
    duration_minutes: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Relationships
    camera: Mapped["Camera"] = relationship(back_populates="logs")
    vehicle: Mapped["Vehicle"] = relationship(back_populates="logs")
    slot: Mapped["ParkingSlot"] = relationship(back_populates="logs")

    def __repr__(self) -> str:
        return f"<ParkingLog(id={self.id}, vehicle_id={self.vehicle_id}, slot_id={self.slot_id})>"


class Alert(Base):
    """
    Stores system-generated alerts (e.g., unauthorized parking, overstay).
    """
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"), index=True)
    
    alert_class: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    details_payload: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )  # Serialized JSON string containing alert context
    event_timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    camera: Mapped["Camera"] = relationship(back_populates="alerts")

    def __repr__(self) -> str:
        return f"<Alert(id={self.id}, class='{self.alert_class}', time={self.event_timestamp})>"