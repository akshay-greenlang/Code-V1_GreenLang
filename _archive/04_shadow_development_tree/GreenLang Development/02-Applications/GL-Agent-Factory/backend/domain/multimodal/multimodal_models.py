# -*- coding: utf-8 -*-
"""
Multi-Modal Data Models for GreenLang
=====================================

Pydantic models for representing various asset types in the multi-modal pipeline:
- ImageAsset: General images with metadata and embeddings
- DiagramAsset: P&ID and PFD diagrams with extracted entities
- ThermalImage: IR thermal images with temperature data
- TimeSeriesAsset: Sensor time-series data
- DocumentAsset: PDF, Word documents with extracted content
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, computed_field


class AssetType(str, Enum):
    """Enumeration of supported asset types."""
    IMAGE = "image"
    DIAGRAM = "diagram"
    THERMAL = "thermal"
    TIMESERIES = "timeseries"
    DOCUMENT = "document"
    VIDEO = "video"
    AUDIO = "audio"


class ProcessingStatus(str, Enum):
    """Processing status for assets."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class ImageFormat(str, Enum):
    """Supported image formats."""
    JPEG = "jpeg"
    PNG = "png"
    TIFF = "tiff"
    BMP = "bmp"
    WEBP = "webp"
    RAW = "raw"


class DiagramType(str, Enum):
    """Types of engineering diagrams."""
    PID = "pid"
    PFD = "pfd"
    ELECTRICAL = "electrical"
    ISOMETRIC = "isometric"
    LAYOUT = "layout"
    SCHEMATIC = "schematic"


class DocumentType(str, Enum):
    """Types of documents."""
    PDF = "pdf"
    WORD = "word"
    EXCEL = "excel"
    TEXT = "text"
    HTML = "html"


class EquipmentType(str, Enum):
    """Standard equipment types in P&ID diagrams."""
    PUMP = "pump"
    COMPRESSOR = "compressor"
    HEAT_EXCHANGER = "heat_exchanger"
    VESSEL = "vessel"
    TANK = "tank"
    REACTOR = "reactor"
    COLUMN = "column"
    VALVE = "valve"
    INSTRUMENT = "instrument"
    BOILER = "boiler"
    TURBINE = "turbine"
    MOTOR = "motor"
    FILTER = "filter"
    FURNACE = "furnace"
    HEATER = "heater"
    OTHER = "other"


class HotspotClassification(str, Enum):
    """Classification of thermal hotspots."""
    NORMAL = "normal"
    ATTENTION = "attention"
    WARNING = "warning"
    CRITICAL = "critical"
    INSULATION_DEFECT = "insulation_defect"
    STEAM_LEAK = "steam_leak"
    ELECTRICAL_FAULT = "electrical_fault"
    BEARING_ISSUE = "bearing_issue"


class SensorType(str, Enum):
    """Types of sensors."""
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    FLOW = "flow"
    LEVEL = "level"
    VIBRATION = "vibration"
    CURRENT = "current"
    VOLTAGE = "voltage"
    POWER = "power"
    OTHER = "other"


class DataQuality(str, Enum):
    """Data quality classification."""
    GOOD = "good"
    SUSPECT = "suspect"
    BAD = "bad"
    MISSING = "missing"
    INTERPOLATED = "interpolated"


# =============================================================================
# Base Models
# =============================================================================

class AssetMetadata(BaseModel):
    """Common metadata for all assets."""

    asset_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Human-readable asset name")
    description: Optional[str] = None
    asset_type: AssetType

    # Source information
    source_path: Optional[str] = None
    source_url: Optional[str] = None
    file_size_bytes: Optional[int] = Field(None, ge=0)
    checksum_sha256: Optional[str] = None

    # Temporal metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    captured_at: Optional[datetime] = None

    # Processing status
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    processing_error: Optional[str] = None
    processing_duration_ms: Optional[int] = None

    # Organizational metadata
    tags: List[str] = Field(default_factory=list)
    labels: Dict[str, str] = Field(default_factory=dict)

    # Provenance
    provenance: Dict[str, Any] = Field(default_factory=dict)

    # Version control
    version: str = "1.0.0"
    parent_asset_id: Optional[str] = None


class BoundingBox(BaseModel):
    """Bounding box coordinates for detected objects."""
    x: float = Field(..., ge=0)
    y: float = Field(..., ge=0)
    width: float = Field(..., gt=0)
    height: float = Field(..., gt=0)
    confidence: float = Field(default=1.0, ge=0, le=1)

    @computed_field
    @property
    def area(self) -> float:
        return self.width * self.height

    @computed_field
    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)


class EmbeddingVector(BaseModel):
    """Vector embedding for similarity search."""
    vector: List[float]
    model_name: str
    model_version: str = "1.0"
    dimensions: int = Field(..., gt=0)
    normalized: bool = True


# =============================================================================
# Image Asset
# =============================================================================

class ImageAsset(BaseModel):
    """General image asset with metadata and embeddings."""

    metadata: AssetMetadata

    # Image properties
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    channels: int = Field(default=3, ge=1, le=4)
    format: ImageFormat = ImageFormat.JPEG
    color_space: str = "RGB"

    # EXIF metadata
    exif_data: Dict[str, Any] = Field(default_factory=dict)

    # Location information
    gps_latitude: Optional[float] = Field(None, ge=-90, le=90)
    gps_longitude: Optional[float] = Field(None, ge=-180, le=180)

    # Embeddings
    embeddings: List[EmbeddingVector] = Field(default_factory=list)

    # Detected objects
    detected_objects: List[Dict[str, Any]] = Field(default_factory=list)

    # OCR
    extracted_text: Optional[str] = None
    text_regions: List[Dict[str, Any]] = Field(default_factory=list)

    # Quality metrics
    quality_score: Optional[float] = Field(None, ge=0, le=100)
    blur_score: Optional[float] = Field(None, ge=0, le=1)

    @computed_field
    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"

    @computed_field
    @property
    def megapixels(self) -> float:
        return (self.width * self.height) / 1_000_000


# =============================================================================
# Diagram Asset (P&ID)
# =============================================================================

class EquipmentEntity(BaseModel):
    """Extracted equipment entity from a diagram."""

    entity_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    equipment_type: EquipmentType
    tag: str
    name: Optional[str] = None
    bounding_box: BoundingBox
    attributes: Dict[str, Any] = Field(default_factory=dict)

    # Design parameters
    design_pressure: Optional[float] = None
    design_temperature: Optional[float] = None
    material: Optional[str] = None

    # Confidence
    detection_confidence: float = Field(default=1.0, ge=0, le=1)
    classification_confidence: float = Field(default=1.0, ge=0, le=1)


class PipeConnection(BaseModel):
    """Piping connection between equipment."""

    connection_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_tag: str
    source_port: Optional[str] = None
    target_tag: str
    target_port: Optional[str] = None

    # Pipe specifications
    line_number: Optional[str] = None
    pipe_size: Optional[str] = None
    material: Optional[str] = None
    fluid: Optional[str] = None

    # Geometry
    path_points: List[Tuple[float, float]] = Field(default_factory=list)
    detection_confidence: float = Field(default=1.0, ge=0, le=1)


class DiagramAsset(BaseModel):
    """P&ID and PFD diagram asset with extracted entities."""

    metadata: AssetMetadata
    diagram_type: DiagramType
    diagram_number: Optional[str] = None
    revision: Optional[str] = None

    # Image properties
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)

    # Title block
    title: Optional[str] = None
    project_name: Optional[str] = None
    project_number: Optional[str] = None

    # Extracted entities
    equipment: List[EquipmentEntity] = Field(default_factory=list)
    connections: List[PipeConnection] = Field(default_factory=list)

    # Notes and annotations
    notes: List[str] = Field(default_factory=list)

    # Embeddings
    embeddings: List[EmbeddingVector] = Field(default_factory=list)

    # Quality
    extraction_quality_score: float = Field(default=0.0, ge=0, le=100)

    @computed_field
    @property
    def equipment_count(self) -> int:
        return len(self.equipment)

    @computed_field
    @property
    def connection_count(self) -> int:
        return len(self.connections)

    def get_equipment_by_type(self, equipment_type: EquipmentType) -> List[EquipmentEntity]:
        return [e for e in self.equipment if e.equipment_type == equipment_type]


# =============================================================================
# Thermal Image
# =============================================================================

class ThermalHotspot(BaseModel):
    """Detected hotspot in thermal image."""

    hotspot_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    bounding_box: BoundingBox
    center_x: float = Field(..., ge=0)
    center_y: float = Field(..., ge=0)

    # Temperature data
    max_temperature_c: float
    min_temperature_c: float
    avg_temperature_c: float

    # Classification
    classification: HotspotClassification = HotspotClassification.NORMAL
    severity_score: float = Field(default=0.0, ge=0, le=1)

    # Reference
    reference_temperature_c: Optional[float] = None
    delta_temperature_c: Optional[float] = None

    # Associated equipment
    associated_equipment_tag: Optional[str] = None
    annotation: Optional[str] = None
    detection_confidence: float = Field(default=1.0, ge=0, le=1)


class InsulationDefect(BaseModel):
    """Detected insulation defect."""

    defect_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    bounding_box: BoundingBox
    defect_type: str
    affected_area_m2: Optional[float] = Field(None, gt=0)

    # Temperature analysis
    surface_temperature_c: float
    ambient_temperature_c: float
    expected_temperature_c: Optional[float] = None

    # Heat loss
    estimated_heat_loss_w: Optional[float] = Field(None, ge=0)
    annual_energy_loss_kwh: Optional[float] = Field(None, ge=0)

    # Severity
    severity: str = "low"
    repair_priority: int = Field(default=3, ge=1, le=5)
    recommended_action: Optional[str] = None


class ThermalImage(BaseModel):
    """Thermal/IR image asset with temperature analysis."""

    metadata: AssetMetadata
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)

    # Thermal camera info
    camera_model: Optional[str] = None
    emissivity: float = Field(default=0.95, gt=0, le=1)
    distance_m: Optional[float] = Field(None, gt=0)

    # Temperature data
    min_temperature_c: float
    max_temperature_c: float
    avg_temperature_c: float
    ambient_temperature_c: Optional[float] = None

    # Analysis results
    hotspots: List[ThermalHotspot] = Field(default_factory=list)
    insulation_defects: List[InsulationDefect] = Field(default_factory=list)

    # Embeddings
    embeddings: List[EmbeddingVector] = Field(default_factory=list)

    # Associated equipment
    equipment_tag: Optional[str] = None
    location_description: Optional[str] = None

    @computed_field
    @property
    def temperature_range_c(self) -> float:
        return self.max_temperature_c - self.min_temperature_c

    @computed_field
    @property
    def hotspot_count(self) -> int:
        return len(self.hotspots)

    @computed_field
    @property
    def critical_hotspot_count(self) -> int:
        return len([h for h in self.hotspots if h.classification == HotspotClassification.CRITICAL])


# =============================================================================
# Time Series Asset
# =============================================================================

class TimeSeriesPoint(BaseModel):
    """Single time-series data point."""
    timestamp: datetime
    value: float
    quality: DataQuality = DataQuality.GOOD
    annotation: Optional[str] = None


class TimeSeriesStatistics(BaseModel):
    """Statistical summary of time-series data."""
    count: int = Field(..., ge=0)
    min_value: float
    max_value: float
    mean_value: float
    std_dev: float = Field(..., ge=0)
    median_value: float
    start_time: datetime
    end_time: datetime
    good_data_pct: float = Field(default=100.0, ge=0, le=100)


class AnomalyDetection(BaseModel):
    """Detected anomaly in time-series."""
    anomaly_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    start_time: datetime
    end_time: datetime
    anomaly_type: str
    severity: str = "low"
    confidence: float = Field(default=1.0, ge=0, le=1)
    description: Optional[str] = None


class TimeSeriesAsset(BaseModel):
    """Time-series sensor data asset."""

    metadata: AssetMetadata

    # Sensor info
    sensor_tag: str
    sensor_type: SensorType
    sensor_name: Optional[str] = None

    # Measurement info
    unit: str
    range_min: Optional[float] = None
    range_max: Optional[float] = None
    sample_rate_hz: Optional[float] = Field(None, gt=0)

    # Associated equipment
    equipment_tag: Optional[str] = None
    location: Optional[str] = None

    # Data storage
    data_points: List[TimeSeriesPoint] = Field(default_factory=list)
    data_url: Optional[str] = None
    data_format: Optional[str] = None

    # Statistics
    statistics: Optional[TimeSeriesStatistics] = None

    # Anomalies
    anomalies: List[AnomalyDetection] = Field(default_factory=list)

    # Trends
    trend_direction: Optional[str] = None
    trend_slope: Optional[float] = None

    # Embeddings
    embeddings: List[EmbeddingVector] = Field(default_factory=list)

    # Alarms
    high_alarm: Optional[float] = None
    high_warning: Optional[float] = None
    low_warning: Optional[float] = None
    low_alarm: Optional[float] = None

    @computed_field
    @property
    def point_count(self) -> int:
        return len(self.data_points)

    @computed_field
    @property
    def anomaly_count(self) -> int:
        return len(self.anomalies)


# =============================================================================
# Document Asset
# =============================================================================

class ExtractedTable(BaseModel):
    """Extracted table from document."""
    table_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    page_number: int = Field(..., ge=1)
    headers: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)
    caption: Optional[str] = None
    extraction_confidence: float = Field(default=1.0, ge=0, le=1)


class DocumentSection(BaseModel):
    """Document section/chapter."""
    section_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    level: int = Field(..., ge=1, le=10)
    start_page: int = Field(..., ge=1)
    end_page: Optional[int] = Field(None, ge=1)
    content: Optional[str] = None


class TechnicalSpecification(BaseModel):
    """Extracted technical specification."""
    spec_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parameter: str
    value: str
    unit: Optional[str] = None
    source_page: Optional[int] = Field(None, ge=1)
    extraction_confidence: float = Field(default=1.0, ge=0, le=1)


class DocumentAsset(BaseModel):
    """Document asset (PDF, Word) with extracted content."""

    metadata: AssetMetadata
    document_type: DocumentType
    file_name: str

    # Document metadata
    title: Optional[str] = None
    author: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)

    # Page info
    page_count: int = Field(..., ge=1)
    language: Optional[str] = None

    # Extracted content
    full_text: Optional[str] = None
    pages_text: List[str] = Field(default_factory=list)

    # Structure
    sections: List[DocumentSection] = Field(default_factory=list)

    # Extracted elements
    tables: List[ExtractedTable] = Field(default_factory=list)
    specifications: List[TechnicalSpecification] = Field(default_factory=list)

    # Equipment references
    referenced_equipment: List[str] = Field(default_factory=list)

    # Document type flags
    is_oem_manual: bool = False
    equipment_model: Optional[str] = None
    equipment_manufacturer: Optional[str] = None

    # Search index
    search_index: Dict[str, List[int]] = Field(default_factory=dict)

    # Embeddings
    embeddings: List[EmbeddingVector] = Field(default_factory=list)
    page_embeddings: List[EmbeddingVector] = Field(default_factory=list)

    # OCR
    is_scanned: bool = False
    ocr_confidence: Optional[float] = Field(None, ge=0, le=1)

    @computed_field
    @property
    def table_count(self) -> int:
        return len(self.tables)

    @computed_field
    @property
    def word_count(self) -> int:
        if self.full_text:
            return len(self.full_text.split())
        return sum(len(page.split()) for page in self.pages_text)

    def search_text(self, query: str) -> List[int]:
        """Search for text and return page numbers."""
        query_lower = query.lower()
        return [
            i + 1 for i, page in enumerate(self.pages_text)
            if query_lower in page.lower()
        ]
