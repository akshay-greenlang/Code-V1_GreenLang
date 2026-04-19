"""
GL-EUDR-002: Database Integration with PostGIS

SQLAlchemy models and repository layer for persisting geolocation data
with full PostGIS spatial support.

Features:
- SQLAlchemy ORM models with GeoAlchemy2 spatial columns
- Repository pattern for database operations
- Support for Point and Polygon geometries
- Spatial indexing for efficient queries
- Multi-tenancy support (organization-scoped)

Requirements:
    pip install sqlalchemy geoalchemy2 psycopg2-binary asyncpg

PostGIS Setup:
    CREATE EXTENSION IF NOT EXISTS postgis;
"""

import logging
import uuid
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
from enum import Enum as PyEnum
from typing import Any, Dict, Generator, List, Optional, Tuple, Type, TypeVar, Union
from uuid import UUID

try:
    from sqlalchemy import (
        Boolean,
        Column,
        Date,
        DateTime,
        Enum,
        Float,
        ForeignKey,
        Index,
        Integer,
        JSON,
        Numeric,
        String,
        Text,
        UniqueConstraint,
        create_engine,
        event,
        func,
        text,
    )
    from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
    from sqlalchemy.orm import (
        Session,
        declarative_base,
        relationship,
        sessionmaker,
    )
    from sqlalchemy.pool import QueuePool

    # GeoAlchemy2 for PostGIS support
    try:
        from geoalchemy2 import Geometry, WKBElement
        from geoalchemy2.shape import from_shape, to_shape
        HAS_GEOALCHEMY = True
    except ImportError:
        HAS_GEOALCHEMY = False
        Geometry = String  # Fallback to WKT string
        WKBElement = None

    HAS_SQLALCHEMY = True

except ImportError:
    HAS_SQLALCHEMY = False
    HAS_GEOALCHEMY = False

from .agent import (
    BulkJobStatus,
    BulkUploadFormat,
    BulkUploadJob,
    CollectionMethod,
    CommodityType,
    GeometryType,
    Plot,
    PlotValidationHistory,
    PointCoordinates,
    PolygonCoordinates,
    ValidationError,
    ValidationSeverity,
    ValidationStatus,
)


logger = logging.getLogger(__name__)

# Type variable for generic repository
T = TypeVar('T')

# Base class for declarative models
if HAS_SQLALCHEMY:
    Base = declarative_base()
else:
    Base = object


# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

class DatabaseConfig:
    """Database configuration."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "gl_eudr",
        username: str = "postgres",
        password: str = "",
        schema: str = "geolocation",
        pool_size: int = 5,
        max_overflow: int = 10,
        echo: bool = False
    ):
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password
        self.schema = schema
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.echo = echo

    @property
    def connection_string(self) -> str:
        """Get PostgreSQL connection string."""
        return (
            f"postgresql://{self.username}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
        )

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Create config from environment variables."""
        import os
        return cls(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "gl_eudr"),
            username=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            schema=os.getenv("DB_SCHEMA", "geolocation"),
            pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "10")),
            echo=os.getenv("DB_ECHO", "false").lower() == "true"
        )


# =============================================================================
# SQLALCHEMY MODELS
# =============================================================================

if HAS_SQLALCHEMY:

    # Define table args based on spatial support
    _base_table_args = [
        Index("idx_plots_supplier", "supplier_id"),
        Index("idx_plots_country", "country_code"),
        Index("idx_plots_commodity", "commodity"),
        Index("idx_plots_status", "validation_status"),
        Index("idx_plots_org", "organization_id"),
        Index("idx_plots_created", "created_at"),
        UniqueConstraint("organization_id", "external_id", name="uq_org_external_id"),
    ]

    # Add GIST spatial index if GeoAlchemy2 is available (FR-018: Spatial Query Performance)
    if HAS_GEOALCHEMY:
        _base_table_args.append(
            Index("idx_plots_geometry_gist", "geometry", postgresql_using="gist")
        )

    class PlotModel(Base):
        """SQLAlchemy model for production plots."""
        __tablename__ = "plots"
        __table_args__ = tuple(_base_table_args) + ({"schema": "geolocation"},)

        # Primary key
        plot_id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

        # Foreign keys
        supplier_id = Column(PGUUID(as_uuid=True), nullable=False, index=True)
        organization_id = Column(PGUUID(as_uuid=True), nullable=False, index=True)

        # External reference
        external_id = Column(String(255), nullable=True)

        # Geometry - PostGIS spatial column
        geometry_type = Column(
            Enum(GeometryType, name="geometry_type_enum", schema="geolocation"),
            nullable=False
        )

        if HAS_GEOALCHEMY:
            # Full PostGIS support
            geometry = Column(
                Geometry(geometry_type='GEOMETRY', srid=4326),
                nullable=False
            )
            centroid = Column(
                Geometry(geometry_type='POINT', srid=4326),
                nullable=True
            )
        else:
            # Fallback to GeoJSON text
            geometry = Column(Text, nullable=False)
            centroid = Column(Text, nullable=True)

        # Bounding box
        bbox_min_lon = Column(Float, nullable=True)
        bbox_min_lat = Column(Float, nullable=True)
        bbox_max_lon = Column(Float, nullable=True)
        bbox_max_lat = Column(Float, nullable=True)

        # Measurements
        area_hectares = Column(Numeric(12, 4), nullable=True)
        perimeter_km = Column(Numeric(10, 4), nullable=True)

        # Location
        country_code = Column(String(2), nullable=False, index=True)
        admin_level_1 = Column(String(255), nullable=True)
        admin_level_2 = Column(String(255), nullable=True)
        admin_level_3 = Column(String(255), nullable=True)
        nearest_place = Column(String(255), nullable=True)

        # Commodity
        commodity = Column(
            Enum(CommodityType, name="commodity_type_enum", schema="geolocation"),
            nullable=False,
            index=True
        )
        crop_type = Column(String(100), nullable=True)

        # Validation
        validation_status = Column(
            Enum(ValidationStatus, name="validation_status_enum", schema="geolocation"),
            nullable=False,
            default=ValidationStatus.PENDING,
            index=True
        )
        validation_errors = Column(JSONB, default=list)
        validation_warnings = Column(JSONB, default=list)
        precision_lat = Column(Integer, nullable=True)
        precision_lon = Column(Integer, nullable=True)
        last_validated_at = Column(DateTime, nullable=True)

        # Collection metadata
        collection_method = Column(
            Enum(CollectionMethod, name="collection_method_enum", schema="geolocation"),
            nullable=False,
            default=CollectionMethod.API
        )
        collection_device = Column(String(255), nullable=True)
        collection_accuracy_m = Column(Float, nullable=True)
        collection_date = Column(Date, nullable=True)
        collected_by = Column(String(255), nullable=True)

        # Enrichment
        biome = Column(String(100), nullable=True)
        ecosystem = Column(String(100), nullable=True)
        elevation_m = Column(Integer, nullable=True)
        slope_degrees = Column(Float, nullable=True)
        in_protected_area = Column(Boolean, default=False)
        protected_area_name = Column(String(255), nullable=True)
        in_urban_area = Column(Boolean, default=False)

        # Version tracking
        version = Column(Integer, default=1)
        previous_version_id = Column(PGUUID(as_uuid=True), nullable=True)

        # Audit
        created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        created_by = Column(String(255), nullable=True)

        # Additional metadata (renamed to avoid SQLAlchemy Base.metadata conflict)
        plot_metadata = Column(JSONB, default=dict)

        # Relationships
        validation_history = relationship(
            "ValidationHistoryModel",
            back_populates="plot",
            cascade="all, delete-orphan"
        )

        def to_domain(self) -> Plot:
            """Convert to domain model."""
            # Convert geometry
            if HAS_GEOALCHEMY and isinstance(self.geometry, WKBElement):
                from shapely import wkb
                shape = to_shape(self.geometry)
                if shape.geom_type == "Point":
                    coordinates = PointCoordinates(
                        latitude=shape.y,
                        longitude=shape.x
                    )
                else:
                    # Polygon - extract exterior coordinates
                    coords = [[c[0], c[1]] for c in shape.exterior.coords]
                    coordinates = PolygonCoordinates(coordinates=coords)
            else:
                # Parse from GeoJSON text
                import json
                geojson = json.loads(self.geometry) if isinstance(self.geometry, str) else self.geometry
                if geojson["type"] == "Point":
                    coordinates = PointCoordinates(
                        latitude=geojson["coordinates"][1],
                        longitude=geojson["coordinates"][0]
                    )
                else:
                    coordinates = PolygonCoordinates(
                        coordinates=geojson["coordinates"][0]
                    )

            # Convert centroid
            centroid = None
            if self.centroid:
                if HAS_GEOALCHEMY and isinstance(self.centroid, WKBElement):
                    shape = to_shape(self.centroid)
                    centroid = PointCoordinates(latitude=shape.y, longitude=shape.x)
                else:
                    import json
                    geojson = json.loads(self.centroid) if isinstance(self.centroid, str) else self.centroid
                    centroid = PointCoordinates(
                        latitude=geojson["coordinates"][1],
                        longitude=geojson["coordinates"][0]
                    )

            # Convert validation errors
            errors = [
                ValidationError(
                    code=e["code"],
                    message=e["message"],
                    severity=ValidationSeverity(e["severity"]),
                    coordinate=tuple(e["coordinate"]) if e.get("coordinate") else None,
                    metadata=e.get("metadata", {})
                )
                for e in (self.validation_errors or [])
            ]

            warnings = [
                ValidationError(
                    code=w["code"],
                    message=w["message"],
                    severity=ValidationSeverity(w["severity"]),
                    coordinate=tuple(w["coordinate"]) if w.get("coordinate") else None,
                    metadata=w.get("metadata", {})
                )
                for w in (self.validation_warnings or [])
            ]

            # Bounding box
            bbox = None
            if all([self.bbox_min_lon, self.bbox_min_lat, self.bbox_max_lon, self.bbox_max_lat]):
                bbox = [self.bbox_min_lon, self.bbox_min_lat, self.bbox_max_lon, self.bbox_max_lat]

            return Plot(
                plot_id=self.plot_id,
                supplier_id=self.supplier_id,
                external_id=self.external_id,
                geometry_type=self.geometry_type,
                coordinates=coordinates,
                centroid=centroid,
                bounding_box=bbox,
                area_hectares=self.area_hectares,
                perimeter_km=self.perimeter_km,
                country_code=self.country_code,
                admin_level_1=self.admin_level_1,
                admin_level_2=self.admin_level_2,
                admin_level_3=self.admin_level_3,
                nearest_place=self.nearest_place,
                commodity=self.commodity,
                crop_type=self.crop_type,
                validation_status=self.validation_status,
                validation_errors=errors,
                validation_warnings=warnings,
                precision_lat=self.precision_lat,
                precision_lon=self.precision_lon,
                last_validated_at=self.last_validated_at,
                collection_method=self.collection_method,
                collection_device=self.collection_device,
                collection_accuracy_m=self.collection_accuracy_m,
                collection_date=self.collection_date,
                collected_by=self.collected_by,
                biome=self.biome,
                ecosystem=self.ecosystem,
                elevation_m=self.elevation_m,
                slope_degrees=self.slope_degrees,
                in_protected_area=self.in_protected_area,
                protected_area_name=self.protected_area_name,
                in_urban_area=self.in_urban_area,
                version=self.version,
                previous_version_id=self.previous_version_id,
                created_at=self.created_at,
                updated_at=self.updated_at,
                created_by=self.created_by,
                organization_id=self.organization_id,
                metadata=self.plot_metadata or {}
            )

        @classmethod
        def from_domain(cls, plot: Plot, organization_id: UUID) -> "PlotModel":
            """Create from domain model."""
            # Convert coordinates to geometry
            if isinstance(plot.coordinates, PointCoordinates):
                if HAS_GEOALCHEMY:
                    from shapely.geometry import Point
                    geom = from_shape(
                        Point(plot.coordinates.longitude, plot.coordinates.latitude),
                        srid=4326
                    )
                else:
                    import json
                    geom = json.dumps(plot.coordinates.to_geojson())
            else:
                if HAS_GEOALCHEMY:
                    from shapely.geometry import Polygon
                    coords = [(c[0], c[1]) for c in plot.coordinates.coordinates]
                    geom = from_shape(Polygon(coords), srid=4326)
                else:
                    import json
                    geom = json.dumps(plot.coordinates.to_geojson())

            # Convert centroid
            centroid_geom = None
            if plot.centroid:
                if HAS_GEOALCHEMY:
                    from shapely.geometry import Point
                    centroid_geom = from_shape(
                        Point(plot.centroid.longitude, plot.centroid.latitude),
                        srid=4326
                    )
                else:
                    import json
                    centroid_geom = json.dumps(plot.centroid.to_geojson())

            # Convert errors to JSON
            errors = [
                {
                    "code": e.code.value if hasattr(e.code, 'value') else e.code,
                    "message": e.message,
                    "severity": e.severity.value if hasattr(e.severity, 'value') else e.severity,
                    "coordinate": list(e.coordinate) if e.coordinate else None,
                    "metadata": e.metadata
                }
                for e in plot.validation_errors
            ]

            warnings = [
                {
                    "code": w.code.value if hasattr(w.code, 'value') else w.code,
                    "message": w.message,
                    "severity": w.severity.value if hasattr(w.severity, 'value') else w.severity,
                    "coordinate": list(w.coordinate) if w.coordinate else None,
                    "metadata": w.metadata
                }
                for w in plot.validation_warnings
            ]

            # Bounding box
            bbox = plot.bounding_box or [None, None, None, None]

            return cls(
                plot_id=plot.plot_id,
                supplier_id=plot.supplier_id,
                organization_id=organization_id,
                external_id=plot.external_id,
                geometry_type=plot.geometry_type,
                geometry=geom,
                centroid=centroid_geom,
                bbox_min_lon=bbox[0] if len(bbox) > 0 else None,
                bbox_min_lat=bbox[1] if len(bbox) > 1 else None,
                bbox_max_lon=bbox[2] if len(bbox) > 2 else None,
                bbox_max_lat=bbox[3] if len(bbox) > 3 else None,
                area_hectares=plot.area_hectares,
                perimeter_km=plot.perimeter_km,
                country_code=plot.country_code,
                admin_level_1=plot.admin_level_1,
                admin_level_2=plot.admin_level_2,
                admin_level_3=plot.admin_level_3,
                nearest_place=plot.nearest_place,
                commodity=plot.commodity,
                crop_type=plot.crop_type,
                validation_status=plot.validation_status,
                validation_errors=errors,
                validation_warnings=warnings,
                precision_lat=plot.precision_lat,
                precision_lon=plot.precision_lon,
                last_validated_at=plot.last_validated_at,
                collection_method=plot.collection_method,
                collection_device=plot.collection_device,
                collection_accuracy_m=plot.collection_accuracy_m,
                collection_date=plot.collection_date,
                collected_by=plot.collected_by,
                biome=plot.biome,
                ecosystem=plot.ecosystem,
                elevation_m=plot.elevation_m,
                slope_degrees=plot.slope_degrees,
                in_protected_area=plot.in_protected_area,
                protected_area_name=plot.protected_area_name,
                in_urban_area=plot.in_urban_area,
                version=plot.version,
                previous_version_id=plot.previous_version_id,
                created_at=plot.created_at,
                updated_at=plot.updated_at,
                created_by=plot.created_by,
                plot_metadata=plot.metadata
            )


    class ValidationHistoryModel(Base):
        """SQLAlchemy model for validation history."""
        __tablename__ = "validation_history"
        __table_args__ = (
            Index("idx_history_plot", "plot_id"),
            Index("idx_history_date", "validation_date"),
            {"schema": "geolocation"}
        )

        validation_id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
        plot_id = Column(
            PGUUID(as_uuid=True),
            ForeignKey("geolocation.plots.plot_id", ondelete="CASCADE"),
            nullable=False,
            index=True
        )
        validation_date = Column(DateTime, default=datetime.utcnow, nullable=False)
        status = Column(
            Enum(ValidationStatus, name="validation_status_enum", schema="geolocation"),
            nullable=False
        )
        errors = Column(JSONB, default=list)
        warnings = Column(JSONB, default=list)
        validated_by = Column(String(255), nullable=True)
        validation_method = Column(String(50), default="AUTO")

        # Relationship
        plot = relationship("PlotModel", back_populates="validation_history")

        def to_domain(self) -> PlotValidationHistory:
            """Convert to domain model."""
            errors = [
                ValidationError(
                    code=e["code"],
                    message=e["message"],
                    severity=ValidationSeverity(e["severity"]),
                    coordinate=tuple(e["coordinate"]) if e.get("coordinate") else None,
                    metadata=e.get("metadata", {})
                )
                for e in (self.errors or [])
            ]

            warnings = [
                ValidationError(
                    code=w["code"],
                    message=w["message"],
                    severity=ValidationSeverity(w["severity"]),
                    coordinate=tuple(w["coordinate"]) if w.get("coordinate") else None,
                    metadata=w.get("metadata", {})
                )
                for w in (self.warnings or [])
            ]

            return PlotValidationHistory(
                validation_id=self.validation_id,
                plot_id=self.plot_id,
                validation_date=self.validation_date,
                status=self.status,
                errors=errors,
                warnings=warnings,
                validated_by=self.validated_by,
                validation_method=self.validation_method
            )


    class BulkJobModel(Base):
        """SQLAlchemy model for bulk upload jobs."""
        __tablename__ = "bulk_jobs"
        __table_args__ = (
            Index("idx_bulk_supplier", "supplier_id"),
            Index("idx_bulk_status", "status"),
            Index("idx_bulk_org", "organization_id"),
            {"schema": "geolocation"}
        )

        job_id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
        supplier_id = Column(PGUUID(as_uuid=True), nullable=False, index=True)
        organization_id = Column(PGUUID(as_uuid=True), nullable=False, index=True)
        file_format = Column(
            Enum(BulkUploadFormat, name="bulk_format_enum", schema="geolocation"),
            nullable=False
        )
        file_name = Column(String(255), nullable=False)
        file_size_bytes = Column(Integer, nullable=False)
        status = Column(
            Enum(BulkJobStatus, name="bulk_status_enum", schema="geolocation"),
            nullable=False,
            default=BulkJobStatus.QUEUED
        )

        # Progress
        total_count = Column(Integer, default=0)
        processed_count = Column(Integer, default=0)
        valid_count = Column(Integer, default=0)
        invalid_count = Column(Integer, default=0)
        warning_count = Column(Integer, default=0)

        # Timing
        created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
        started_at = Column(DateTime, nullable=True)
        completed_at = Column(DateTime, nullable=True)

        # Results
        report_url = Column(String(500), nullable=True)
        error_message = Column(Text, nullable=True)
        created_by = Column(String(255), nullable=True)

        def to_domain(self) -> BulkUploadJob:
            """Convert to domain model."""
            return BulkUploadJob(
                job_id=self.job_id,
                supplier_id=self.supplier_id,
                file_format=self.file_format,
                file_name=self.file_name,
                file_size_bytes=self.file_size_bytes,
                status=self.status,
                total_count=self.total_count,
                processed_count=self.processed_count,
                valid_count=self.valid_count,
                invalid_count=self.invalid_count,
                warning_count=self.warning_count,
                created_at=self.created_at,
                started_at=self.started_at,
                completed_at=self.completed_at,
                report_url=self.report_url,
                error_message=self.error_message,
                created_by=self.created_by,
                organization_id=self.organization_id
            )


# =============================================================================
# DATABASE SESSION MANAGEMENT
# =============================================================================

class DatabaseSession:
    """Database session manager."""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._engine = None
        self._session_factory = None

    def initialize(self) -> None:
        """Initialize database connection."""
        if not HAS_SQLALCHEMY:
            raise RuntimeError("SQLAlchemy not installed")

        self._engine = create_engine(
            self.config.connection_string,
            poolclass=QueuePool,
            pool_size=self.config.pool_size,
            max_overflow=self.config.max_overflow,
            echo=self.config.echo
        )

        self._session_factory = sessionmaker(
            bind=self._engine,
            autocommit=False,
            autoflush=False
        )

        logger.info(f"Database initialized: {self.config.host}:{self.config.port}/{self.config.database}")

    def create_tables(self) -> None:
        """Create all tables."""
        if not self._engine:
            raise RuntimeError("Database not initialized")

        # Create schema if not exists
        with self._engine.connect() as conn:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {self.config.schema}"))
            conn.commit()

        Base.metadata.create_all(self._engine)
        logger.info("Database tables created")

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Get a database session."""
        if not self._session_factory:
            raise RuntimeError("Database not initialized")

        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def close(self) -> None:
        """Close database connection."""
        if self._engine:
            self._engine.dispose()
            logger.info("Database connection closed")


# =============================================================================
# REPOSITORY PATTERN
# =============================================================================

class PlotRepository:
    """Repository for plot database operations."""

    def __init__(self, session: Session, organization_id: UUID):
        self.session = session
        self.organization_id = organization_id

    def create(self, plot: Plot) -> Plot:
        """Create a new plot."""
        model = PlotModel.from_domain(plot, self.organization_id)
        self.session.add(model)
        self.session.flush()
        return model.to_domain()

    def get(self, plot_id: UUID) -> Optional[Plot]:
        """Get a plot by ID."""
        model = self.session.query(PlotModel).filter(
            PlotModel.plot_id == plot_id,
            PlotModel.organization_id == self.organization_id
        ).first()

        return model.to_domain() if model else None

    def update(self, plot: Plot) -> Plot:
        """Update an existing plot."""
        model = self.session.query(PlotModel).filter(
            PlotModel.plot_id == plot.plot_id,
            PlotModel.organization_id == self.organization_id
        ).first()

        if not model:
            raise ValueError(f"Plot {plot.plot_id} not found")

        # Update fields
        updated_model = PlotModel.from_domain(plot, self.organization_id)
        for key, value in updated_model.__dict__.items():
            if not key.startswith('_'):
                setattr(model, key, value)

        model.updated_at = datetime.utcnow()
        self.session.flush()
        return model.to_domain()

    def delete(self, plot_id: UUID) -> bool:
        """Delete a plot."""
        result = self.session.query(PlotModel).filter(
            PlotModel.plot_id == plot_id,
            PlotModel.organization_id == self.organization_id
        ).delete()

        return result > 0

    def list(
        self,
        supplier_id: Optional[UUID] = None,
        commodity: Optional[CommodityType] = None,
        validation_status: Optional[ValidationStatus] = None,
        country_code: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[Plot], int]:
        """List plots with filtering."""
        query = self.session.query(PlotModel).filter(
            PlotModel.organization_id == self.organization_id
        )

        if supplier_id:
            query = query.filter(PlotModel.supplier_id == supplier_id)
        if commodity:
            query = query.filter(PlotModel.commodity == commodity)
        if validation_status:
            query = query.filter(PlotModel.validation_status == validation_status)
        if country_code:
            query = query.filter(PlotModel.country_code == country_code)

        total = query.count()
        models = query.order_by(PlotModel.created_at.desc()).offset(offset).limit(limit).all()

        return [m.to_domain() for m in models], total

    def find_by_supplier(self, supplier_id: UUID) -> List[Plot]:
        """Find all plots for a supplier."""
        models = self.session.query(PlotModel).filter(
            PlotModel.supplier_id == supplier_id,
            PlotModel.organization_id == self.organization_id
        ).all()

        return [m.to_domain() for m in models]

    def find_within_bbox(
        self,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float
    ) -> List[Plot]:
        """Find plots within a bounding box."""
        if HAS_GEOALCHEMY:
            # Use PostGIS spatial query
            bbox_wkt = f"POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, {max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))"
            models = self.session.query(PlotModel).filter(
                PlotModel.organization_id == self.organization_id,
                func.ST_Intersects(
                    PlotModel.geometry,
                    func.ST_GeomFromText(bbox_wkt, 4326)
                )
            ).all()
        else:
            # Fallback to bbox columns
            models = self.session.query(PlotModel).filter(
                PlotModel.organization_id == self.organization_id,
                PlotModel.bbox_min_lon >= min_lon,
                PlotModel.bbox_max_lon <= max_lon,
                PlotModel.bbox_min_lat >= min_lat,
                PlotModel.bbox_max_lat <= max_lat
            ).all()

        return [m.to_domain() for m in models]

    def get_stats(self) -> Dict[str, int]:
        """Get validation statistics."""
        total = self.session.query(PlotModel).filter(
            PlotModel.organization_id == self.organization_id
        ).count()

        valid = self.session.query(PlotModel).filter(
            PlotModel.organization_id == self.organization_id,
            PlotModel.validation_status == ValidationStatus.VALID
        ).count()

        invalid = self.session.query(PlotModel).filter(
            PlotModel.organization_id == self.organization_id,
            PlotModel.validation_status == ValidationStatus.INVALID
        ).count()

        needs_review = self.session.query(PlotModel).filter(
            PlotModel.organization_id == self.organization_id,
            PlotModel.validation_status == ValidationStatus.NEEDS_REVIEW
        ).count()

        return {
            "total_plots": total,
            "valid_count": valid,
            "invalid_count": invalid,
            "needs_review_count": needs_review
        }


class ValidationHistoryRepository:
    """Repository for validation history operations."""

    def __init__(self, session: Session):
        self.session = session

    def create(self, history: PlotValidationHistory) -> PlotValidationHistory:
        """Create a validation history entry."""
        errors = [
            {
                "code": e.code.value if hasattr(e.code, 'value') else e.code,
                "message": e.message,
                "severity": e.severity.value if hasattr(e.severity, 'value') else e.severity,
                "coordinate": list(e.coordinate) if e.coordinate else None,
                "metadata": e.metadata
            }
            for e in history.errors
        ]

        warnings = [
            {
                "code": w.code.value if hasattr(w.code, 'value') else w.code,
                "message": w.message,
                "severity": w.severity.value if hasattr(w.severity, 'value') else w.severity,
                "coordinate": list(w.coordinate) if w.coordinate else None,
                "metadata": w.metadata
            }
            for w in history.warnings
        ]

        model = ValidationHistoryModel(
            validation_id=history.validation_id,
            plot_id=history.plot_id,
            validation_date=history.validation_date,
            status=history.status,
            errors=errors,
            warnings=warnings,
            validated_by=history.validated_by,
            validation_method=history.validation_method
        )

        self.session.add(model)
        self.session.flush()
        return model.to_domain()

    def get_by_plot(self, plot_id: UUID) -> List[PlotValidationHistory]:
        """Get validation history for a plot."""
        models = self.session.query(ValidationHistoryModel).filter(
            ValidationHistoryModel.plot_id == plot_id
        ).order_by(ValidationHistoryModel.validation_date.desc()).all()

        return [m.to_domain() for m in models]


class BulkJobRepository:
    """Repository for bulk job operations."""

    def __init__(self, session: Session, organization_id: UUID):
        self.session = session
        self.organization_id = organization_id

    def create(self, job: BulkUploadJob) -> BulkUploadJob:
        """Create a bulk job."""
        model = BulkJobModel(
            job_id=job.job_id,
            supplier_id=job.supplier_id,
            organization_id=self.organization_id,
            file_format=job.file_format,
            file_name=job.file_name,
            file_size_bytes=job.file_size_bytes,
            status=job.status,
            created_by=job.created_by
        )

        self.session.add(model)
        self.session.flush()
        return model.to_domain()

    def get(self, job_id: UUID) -> Optional[BulkUploadJob]:
        """Get a bulk job by ID."""
        model = self.session.query(BulkJobModel).filter(
            BulkJobModel.job_id == job_id,
            BulkJobModel.organization_id == self.organization_id
        ).first()

        return model.to_domain() if model else None

    def update_progress(
        self,
        job_id: UUID,
        processed_count: int,
        valid_count: int,
        invalid_count: int,
        warning_count: int
    ) -> None:
        """Update job progress."""
        self.session.query(BulkJobModel).filter(
            BulkJobModel.job_id == job_id
        ).update({
            "processed_count": processed_count,
            "valid_count": valid_count,
            "invalid_count": invalid_count,
            "warning_count": warning_count
        })

    def complete(
        self,
        job_id: UUID,
        status: BulkJobStatus,
        report_url: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> None:
        """Mark job as complete."""
        self.session.query(BulkJobModel).filter(
            BulkJobModel.job_id == job_id
        ).update({
            "status": status,
            "completed_at": datetime.utcnow(),
            "report_url": report_url,
            "error_message": error_message
        })
