"""
GL-EUDR-002: Async Bulk Upload Processing

Handles asynchronous processing of bulk plot uploads:
- File parsing (CSV, GeoJSON, KML, Shapefile)
- Async job queue with status polling
- Batch validation
- Report generation

Processing Flow:
1. File uploaded -> Job created (QUEUED)
2. Worker picks up job -> Status: RUNNING
3. Parse file -> Validate each record -> Update progress
4. Generate report -> Status: COMPLETE or FAILED

Interview Decision: Async with polling status endpoint
"""

import asyncio
import csv
import io
import json
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, Generator, List, Optional, Tuple, Union
from uuid import UUID

from .agent import (
    BulkJobStatus,
    BulkUploadFormat,
    BulkUploadJob,
    BulkUploadResult,
    CommodityType,
    CollectionMethod,
    GeolocationValidator,
    GeometryType,
    Plot,
    PlotSubmission,
    PointCoordinates,
    PolygonCoordinates,
    ValidationResult,
    ValidationStatus,
)

logger = logging.getLogger(__name__)


# =============================================================================
# PARSER INTERFACES
# =============================================================================

@dataclass
class ParsedRecord:
    """A single record parsed from bulk upload file."""
    row_number: int
    external_id: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    polygon_coords: Optional[List[List[float]]] = None
    country_code: Optional[str] = None
    commodity: Optional[str] = None
    declared_area: Optional[float] = None
    collection_method: Optional[str] = None
    collection_date: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    parse_errors: List[str] = field(default_factory=list)
    raw_data: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return len(self.parse_errors) == 0

    @property
    def geometry_type(self) -> GeometryType:
        if self.polygon_coords:
            return GeometryType.POLYGON
        return GeometryType.POINT


@dataclass
class ValidationBatch:
    """Batch of records for validation."""
    records: List[ParsedRecord]
    start_index: int
    batch_number: int


@dataclass
class ProcessingProgress:
    """Progress update during processing."""
    total_count: int
    processed_count: int
    valid_count: int
    invalid_count: int
    warning_count: int
    current_batch: int
    total_batches: int

    @property
    def progress_percent(self) -> float:
        if self.total_count == 0:
            return 0.0
        return (self.processed_count / self.total_count) * 100


@dataclass
class ProcessingResult:
    """Result of processing a single record."""
    row_number: int
    external_id: Optional[str]
    plot_id: Optional[UUID]
    status: ValidationStatus
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]


# =============================================================================
# FILE PARSERS
# =============================================================================

class BulkFileParser:
    """Base class for bulk file parsers."""

    def parse(self, file_path: str) -> Generator[ParsedRecord, None, None]:
        """Parse file and yield records."""
        raise NotImplementedError

    def parse_content(self, content: bytes, filename: str) -> Generator[ParsedRecord, None, None]:
        """Parse content bytes and yield records."""
        raise NotImplementedError


class CSVParser(BulkFileParser):
    """Parser for CSV files."""

    # Expected column names (with variations)
    COLUMN_MAPPINGS = {
        "latitude": ["latitude", "lat", "y", "coord_lat"],
        "longitude": ["longitude", "lon", "lng", "x", "coord_lon"],
        "external_id": ["external_id", "plot_id", "id", "reference"],
        "country_code": ["country_code", "country", "iso"],
        "commodity": ["commodity", "crop", "product"],
        "area": ["area", "area_hectares", "hectares", "size"],
        "collection_method": ["collection_method", "method", "source"],
        "collection_date": ["collection_date", "date", "collected_on"],
    }

    def parse(self, file_path: str) -> Generator[ParsedRecord, None, None]:
        """Parse CSV file."""
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            yield from self._parse_csv(f)

    def parse_content(self, content: bytes, filename: str) -> Generator[ParsedRecord, None, None]:
        """Parse CSV content."""
        text = content.decode('utf-8-sig')
        f = io.StringIO(text)
        yield from self._parse_csv(f)

    def _parse_csv(self, file_obj) -> Generator[ParsedRecord, None, None]:
        """Internal CSV parsing."""
        reader = csv.DictReader(file_obj)

        # Map columns to standard names
        column_map = self._build_column_map(reader.fieldnames or [])

        for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is 1)
            record = self._parse_row(row, row_num, column_map)
            yield record

    def _build_column_map(self, fieldnames: List[str]) -> Dict[str, str]:
        """Map CSV column names to standard names."""
        column_map = {}
        fieldnames_lower = {f.lower().strip(): f for f in fieldnames}

        for standard_name, variations in self.COLUMN_MAPPINGS.items():
            for variation in variations:
                if variation in fieldnames_lower:
                    column_map[standard_name] = fieldnames_lower[variation]
                    break

        return column_map

    def _parse_row(
        self,
        row: Dict[str, str],
        row_num: int,
        column_map: Dict[str, str]
    ) -> ParsedRecord:
        """Parse a single CSV row."""
        errors = []

        # Get latitude
        lat = None
        lat_col = column_map.get("latitude")
        if lat_col and row.get(lat_col):
            try:
                lat = float(row[lat_col])
            except ValueError:
                errors.append(f"Invalid latitude value: {row.get(lat_col)}")

        # Get longitude
        lon = None
        lon_col = column_map.get("longitude")
        if lon_col and row.get(lon_col):
            try:
                lon = float(row[lon_col])
            except ValueError:
                errors.append(f"Invalid longitude value: {row.get(lon_col)}")

        # Get other fields
        external_id = row.get(column_map.get("external_id", ""))
        country_code = row.get(column_map.get("country_code", ""))
        commodity = row.get(column_map.get("commodity", ""))

        # Get area
        area = None
        area_col = column_map.get("area")
        if area_col and row.get(area_col):
            try:
                area = float(row[area_col])
            except ValueError:
                pass

        # Validate required fields
        if lat is None:
            errors.append("Missing latitude")
        if lon is None:
            errors.append("Missing longitude")
        if not country_code:
            errors.append("Missing country code")
        if not commodity:
            errors.append("Missing commodity")

        return ParsedRecord(
            row_number=row_num,
            external_id=external_id or None,
            latitude=lat,
            longitude=lon,
            country_code=country_code.upper() if country_code else None,
            commodity=commodity.upper() if commodity else None,
            declared_area=area,
            collection_method=row.get(column_map.get("collection_method", "")),
            collection_date=row.get(column_map.get("collection_date", "")),
            parse_errors=errors,
            raw_data=dict(row)
        )


class GeoJSONParser(BulkFileParser):
    """Parser for GeoJSON files."""

    def parse(self, file_path: str) -> Generator[ParsedRecord, None, None]:
        """Parse GeoJSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            geojson = json.load(f)
        yield from self._parse_geojson(geojson)

    def parse_content(self, content: bytes, filename: str) -> Generator[ParsedRecord, None, None]:
        """Parse GeoJSON content."""
        geojson = json.loads(content.decode('utf-8'))
        yield from self._parse_geojson(geojson)

    def _parse_geojson(self, geojson: Dict) -> Generator[ParsedRecord, None, None]:
        """Parse GeoJSON structure."""
        features = []

        if geojson.get("type") == "FeatureCollection":
            features = geojson.get("features", [])
        elif geojson.get("type") == "Feature":
            features = [geojson]

        for idx, feature in enumerate(features):
            yield self._parse_feature(feature, idx + 1)

    def _parse_feature(self, feature: Dict, row_num: int) -> ParsedRecord:
        """Parse a single GeoJSON feature."""
        errors = []
        properties = feature.get("properties", {})
        geometry = feature.get("geometry", {})

        geom_type = geometry.get("type")
        coords = geometry.get("coordinates")

        lat, lon = None, None
        polygon_coords = None

        if geom_type == "Point":
            if coords and len(coords) >= 2:
                lon, lat = coords[0], coords[1]
            else:
                errors.append("Invalid Point coordinates")

        elif geom_type == "Polygon":
            if coords and len(coords) > 0 and len(coords[0]) >= 4:
                polygon_coords = coords[0]
                # Calculate centroid for reference
                n = len(polygon_coords) - 1
                lon = sum(c[0] for c in polygon_coords[:n]) / n
                lat = sum(c[1] for c in polygon_coords[:n]) / n
            else:
                errors.append("Invalid Polygon coordinates")

        else:
            errors.append(f"Unsupported geometry type: {geom_type}")

        # Get properties
        external_id = properties.get("external_id") or properties.get("id")
        country_code = properties.get("country_code") or properties.get("country")
        commodity = properties.get("commodity") or properties.get("crop")
        area = properties.get("area_hectares") or properties.get("area")

        # Validate required
        if not country_code:
            errors.append("Missing country_code property")
        if not commodity:
            errors.append("Missing commodity property")

        return ParsedRecord(
            row_number=row_num,
            external_id=str(external_id) if external_id else None,
            latitude=lat,
            longitude=lon,
            polygon_coords=polygon_coords,
            country_code=str(country_code).upper() if country_code else None,
            commodity=str(commodity).upper() if commodity else None,
            declared_area=float(area) if area else None,
            parse_errors=errors,
            raw_data=feature
        )


class KMLParser(BulkFileParser):
    """Parser for KML files (basic implementation)."""

    def parse(self, file_path: str) -> Generator[ParsedRecord, None, None]:
        """Parse KML file."""
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(file_path)
            yield from self._parse_kml_tree(tree)
        except ImportError:
            logger.error("XML parsing not available")
        except Exception as e:
            logger.error(f"KML parsing error: {e}")

    def parse_content(self, content: bytes, filename: str) -> Generator[ParsedRecord, None, None]:
        """Parse KML content."""
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(content)
            tree = ET.ElementTree(root)
            yield from self._parse_kml_tree(tree)
        except Exception as e:
            logger.error(f"KML parsing error: {e}")

    def _parse_kml_tree(self, tree) -> Generator[ParsedRecord, None, None]:
        """Parse KML XML tree."""
        import xml.etree.ElementTree as ET

        # KML namespace
        ns = {'kml': 'http://www.opengis.net/kml/2.2'}

        root = tree.getroot()
        placemarks = root.findall('.//kml:Placemark', ns)

        if not placemarks:
            # Try without namespace
            placemarks = root.findall('.//Placemark')

        for idx, placemark in enumerate(placemarks):
            yield self._parse_placemark(placemark, idx + 1, ns)

    def _parse_placemark(self, placemark, row_num: int, ns: Dict) -> ParsedRecord:
        """Parse a KML Placemark."""
        errors = []

        # Get name
        name_elem = placemark.find('kml:name', ns) or placemark.find('name')
        name = name_elem.text if name_elem is not None else None

        # Get coordinates
        lat, lon = None, None
        polygon_coords = None

        # Try Point
        point = placemark.find('.//kml:Point/kml:coordinates', ns)
        if point is None:
            point = placemark.find('.//Point/coordinates')

        if point is not None and point.text:
            coords = point.text.strip().split(',')
            if len(coords) >= 2:
                lon = float(coords[0])
                lat = float(coords[1])

        # Try Polygon
        polygon = placemark.find('.//kml:Polygon//kml:coordinates', ns)
        if polygon is None:
            polygon = placemark.find('.//Polygon//coordinates')

        if polygon is not None and polygon.text:
            coord_text = polygon.text.strip()
            coord_pairs = coord_text.split()
            polygon_coords = []
            for pair in coord_pairs:
                parts = pair.split(',')
                if len(parts) >= 2:
                    polygon_coords.append([float(parts[0]), float(parts[1])])

            if polygon_coords:
                n = len(polygon_coords)
                lon = sum(c[0] for c in polygon_coords) / n
                lat = sum(c[1] for c in polygon_coords) / n

        if lat is None or lon is None:
            errors.append("No valid coordinates found")

        # Get extended data
        country_code = None
        commodity = None

        for data in placemark.findall('.//kml:ExtendedData/kml:Data', ns):
            data_name = data.get('name', '').lower()
            value_elem = data.find('kml:value', ns)
            value = value_elem.text if value_elem is not None else None

            if data_name in ['country_code', 'country']:
                country_code = value
            elif data_name in ['commodity', 'crop']:
                commodity = value

        if not country_code:
            errors.append("Missing country_code")
        if not commodity:
            errors.append("Missing commodity")

        return ParsedRecord(
            row_number=row_num,
            external_id=name,
            latitude=lat,
            longitude=lon,
            polygon_coords=polygon_coords,
            country_code=country_code.upper() if country_code else None,
            commodity=commodity.upper() if commodity else None,
            parse_errors=errors,
            raw_data={"name": name}
        )


# =============================================================================
# BULK UPLOAD PROCESSOR
# =============================================================================

class BulkUploadProcessor:
    """
    Async bulk upload processor.

    Handles file parsing, validation, and progress reporting.
    """

    # Batch size for processing
    BATCH_SIZE = 100

    def __init__(
        self,
        validator: GeolocationValidator = None,
        on_progress: Optional[Callable[[ProcessingProgress], None]] = None
    ):
        """
        Initialize processor.

        Args:
            validator: GeolocationValidator instance
            on_progress: Callback for progress updates
        """
        self.validator = validator or GeolocationValidator()
        self.on_progress = on_progress
        self._parsers = {
            BulkUploadFormat.CSV: CSVParser(),
            BulkUploadFormat.GEOJSON: GeoJSONParser(),
            BulkUploadFormat.KML: KMLParser(),
        }

    def get_parser(self, format: BulkUploadFormat) -> Optional[BulkFileParser]:
        """Get parser for file format."""
        return self._parsers.get(format)

    async def process_file(
        self,
        file_path: str,
        file_format: BulkUploadFormat,
        supplier_id: UUID,
        default_country: Optional[str] = None,
        default_commodity: Optional[CommodityType] = None
    ) -> BulkUploadResult:
        """
        Process a bulk upload file asynchronously.

        Args:
            file_path: Path to uploaded file
            file_format: File format
            supplier_id: Supplier ID for all plots
            default_country: Default country code if not in file
            default_commodity: Default commodity if not in file

        Returns:
            BulkUploadResult with processing results
        """
        start_time = datetime.utcnow()
        job_id = uuid.uuid4()

        # Get parser
        parser = self.get_parser(file_format)
        if not parser:
            return BulkUploadResult(
                job_id=job_id,
                status=BulkJobStatus.FAILED,
                total_count=0,
                processed_count=0,
                valid_count=0,
                invalid_count=0,
                warning_count=0,
                processing_time_ms=0,
                errors=[{"error": f"Unsupported format: {file_format}"}]
            )

        # Parse and collect records
        try:
            records = list(parser.parse(file_path))
        except Exception as e:
            logger.exception(f"File parsing error: {e}")
            return BulkUploadResult(
                job_id=job_id,
                status=BulkJobStatus.FAILED,
                total_count=0,
                processed_count=0,
                valid_count=0,
                invalid_count=0,
                warning_count=0,
                processing_time_ms=0,
                errors=[{"error": f"File parsing failed: {str(e)}"}]
            )

        total_count = len(records)

        if total_count == 0:
            return BulkUploadResult(
                job_id=job_id,
                status=BulkJobStatus.COMPLETE,
                total_count=0,
                processed_count=0,
                valid_count=0,
                invalid_count=0,
                warning_count=0,
                processing_time_ms=0,
                errors=[{"error": "No records found in file"}]
            )

        # Process in batches
        results: List[ProcessingResult] = []
        plots: List[Plot] = []
        valid_count = 0
        invalid_count = 0
        warning_count = 0

        num_batches = (total_count + self.BATCH_SIZE - 1) // self.BATCH_SIZE

        for batch_num, batch_start in enumerate(range(0, total_count, self.BATCH_SIZE)):
            batch_end = min(batch_start + self.BATCH_SIZE, total_count)
            batch_records = records[batch_start:batch_end]

            # Process batch
            for record in batch_records:
                result = self._process_record(
                    record,
                    supplier_id,
                    default_country,
                    default_commodity
                )
                results.append(result)

                if result.status == ValidationStatus.VALID:
                    valid_count += 1
                elif result.status == ValidationStatus.INVALID:
                    invalid_count += 1
                elif result.status == ValidationStatus.NEEDS_REVIEW:
                    warning_count += 1

            # Report progress
            if self.on_progress:
                progress = ProcessingProgress(
                    total_count=total_count,
                    processed_count=batch_end,
                    valid_count=valid_count,
                    invalid_count=invalid_count,
                    warning_count=warning_count,
                    current_batch=batch_num + 1,
                    total_batches=num_batches
                )
                self.on_progress(progress)

            # Yield control to allow other tasks
            await asyncio.sleep(0)

        # Calculate processing time
        end_time = datetime.utcnow()
        processing_time_ms = int((end_time - start_time).total_seconds() * 1000)

        # Build error list for failed records
        error_list = [
            {
                "row": r.row_number,
                "external_id": r.external_id,
                "errors": r.errors
            }
            for r in results
            if r.status == ValidationStatus.INVALID
        ]

        return BulkUploadResult(
            job_id=job_id,
            status=BulkJobStatus.COMPLETE,
            total_count=total_count,
            processed_count=total_count,
            valid_count=valid_count,
            invalid_count=invalid_count,
            warning_count=warning_count,
            processing_time_ms=processing_time_ms,
            plots=plots,
            errors=error_list
        )

    def _process_record(
        self,
        record: ParsedRecord,
        supplier_id: UUID,
        default_country: Optional[str],
        default_commodity: Optional[CommodityType]
    ) -> ProcessingResult:
        """Process a single record."""
        # Check for parse errors
        if not record.is_valid:
            return ProcessingResult(
                row_number=record.row_number,
                external_id=record.external_id,
                plot_id=None,
                status=ValidationStatus.INVALID,
                errors=[{"code": "PARSE_ERROR", "message": e} for e in record.parse_errors],
                warnings=[]
            )

        # Build coordinates
        try:
            if record.polygon_coords:
                coordinates = PolygonCoordinates(coordinates=record.polygon_coords)
            else:
                coordinates = PointCoordinates(
                    latitude=record.latitude,
                    longitude=record.longitude
                )
        except Exception as e:
            return ProcessingResult(
                row_number=record.row_number,
                external_id=record.external_id,
                plot_id=None,
                status=ValidationStatus.INVALID,
                errors=[{"code": "COORDINATE_ERROR", "message": str(e)}],
                warnings=[]
            )

        # Get country and commodity
        country_code = record.country_code or default_country
        commodity_str = record.commodity or (default_commodity.value if default_commodity else None)

        if not country_code:
            return ProcessingResult(
                row_number=record.row_number,
                external_id=record.external_id,
                plot_id=None,
                status=ValidationStatus.INVALID,
                errors=[{"code": "MISSING_COUNTRY", "message": "Country code required"}],
                warnings=[]
            )

        if not commodity_str:
            return ProcessingResult(
                row_number=record.row_number,
                external_id=record.external_id,
                plot_id=None,
                status=ValidationStatus.INVALID,
                errors=[{"code": "MISSING_COMMODITY", "message": "Commodity required"}],
                warnings=[]
            )

        # Convert commodity
        try:
            commodity = CommodityType(commodity_str)
        except ValueError:
            return ProcessingResult(
                row_number=record.row_number,
                external_id=record.external_id,
                plot_id=None,
                status=ValidationStatus.INVALID,
                errors=[{"code": "INVALID_COMMODITY", "message": f"Unknown commodity: {commodity_str}"}],
                warnings=[]
            )

        # Validate
        validation_result = self.validator.validate(
            coordinates=coordinates,
            country_code=country_code,
            commodity=commodity,
            declared_area=record.declared_area
        )

        # Convert errors/warnings to dicts
        errors = [
            {
                "code": e.code.value if hasattr(e.code, 'value') else e.code,
                "message": e.message,
                "severity": e.severity.value if hasattr(e.severity, 'value') else e.severity
            }
            for e in validation_result.errors
        ]

        warnings = [
            {
                "code": w.code.value if hasattr(w.code, 'value') else w.code,
                "message": w.message,
                "severity": w.severity.value if hasattr(w.severity, 'value') else w.severity
            }
            for w in validation_result.warnings
        ]

        plot_id = uuid.uuid4() if validation_result.valid else None

        return ProcessingResult(
            row_number=record.row_number,
            external_id=record.external_id,
            plot_id=plot_id,
            status=validation_result.status,
            errors=errors,
            warnings=warnings
        )

    async def process_content(
        self,
        content: bytes,
        filename: str,
        file_format: BulkUploadFormat,
        supplier_id: UUID,
        default_country: Optional[str] = None,
        default_commodity: Optional[CommodityType] = None
    ) -> BulkUploadResult:
        """
        Process uploaded file content directly.

        Args:
            content: File content as bytes
            filename: Original filename
            file_format: File format
            supplier_id: Supplier ID
            default_country: Default country code
            default_commodity: Default commodity

        Returns:
            BulkUploadResult
        """
        # Write to temp file and process
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_format.value}") as f:
            f.write(content)
            temp_path = f.name

        try:
            result = await self.process_file(
                temp_path,
                file_format,
                supplier_id,
                default_country,
                default_commodity
            )
            return result
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except Exception:
                pass


# =============================================================================
# JOB QUEUE (In-Memory for Demo)
# =============================================================================

class BulkJobQueue:
    """
    In-memory job queue for bulk uploads.

    In production, this would be replaced with:
    - Redis Queue (RQ)
    - Celery
    - AWS SQS
    - PostgreSQL-based queue (pg_boss)
    """

    def __init__(self):
        self._jobs: Dict[UUID, BulkUploadJob] = {}
        self._results: Dict[UUID, BulkUploadResult] = {}
        self._lock = asyncio.Lock()

    async def enqueue(self, job: BulkUploadJob) -> UUID:
        """Add job to queue."""
        async with self._lock:
            self._jobs[job.job_id] = job
        return job.job_id

    async def get_job(self, job_id: UUID) -> Optional[BulkUploadJob]:
        """Get job by ID."""
        return self._jobs.get(job_id)

    async def update_status(
        self,
        job_id: UUID,
        status: BulkJobStatus,
        progress: Optional[ProcessingProgress] = None
    ) -> None:
        """Update job status."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = status
                if progress:
                    job.processed_count = progress.processed_count
                    job.valid_count = progress.valid_count
                    job.invalid_count = progress.invalid_count
                    job.warning_count = progress.warning_count

                if status == BulkJobStatus.RUNNING and not job.started_at:
                    job.started_at = datetime.utcnow()
                elif status in [BulkJobStatus.COMPLETE, BulkJobStatus.FAILED]:
                    job.completed_at = datetime.utcnow()

    async def complete_job(
        self,
        job_id: UUID,
        result: BulkUploadResult
    ) -> None:
        """Mark job as complete with result."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = result.status
                job.total_count = result.total_count
                job.processed_count = result.processed_count
                job.valid_count = result.valid_count
                job.invalid_count = result.invalid_count
                job.warning_count = result.warning_count
                job.completed_at = datetime.utcnow()

            self._results[job_id] = result

    async def get_result(self, job_id: UUID) -> Optional[BulkUploadResult]:
        """Get job result."""
        return self._results.get(job_id)


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_processing_report(
    job: BulkUploadJob,
    result: BulkUploadResult
) -> str:
    """
    Generate a text report of bulk processing results.

    Args:
        job: The bulk upload job
        result: Processing result

    Returns:
        Formatted report string
    """
    lines = [
        "=" * 60,
        "BULK UPLOAD PROCESSING REPORT",
        "=" * 60,
        "",
        f"Job ID: {job.job_id}",
        f"File: {job.file_name}",
        f"Format: {job.file_format}",
        f"Status: {result.status.value}",
        "",
        "-" * 40,
        "SUMMARY",
        "-" * 40,
        f"Total Records: {result.total_count}",
        f"Valid: {result.valid_count}",
        f"Invalid: {result.invalid_count}",
        f"Warnings: {result.warning_count}",
        f"Processing Time: {result.processing_time_ms}ms",
        "",
    ]

    # Add success rate
    if result.total_count > 0:
        success_rate = (result.valid_count / result.total_count) * 100
        lines.append(f"Success Rate: {success_rate:.1f}%")
        lines.append("")

    # Add errors if any
    if result.errors:
        lines.extend([
            "-" * 40,
            "ERRORS",
            "-" * 40,
        ])

        for error in result.errors[:50]:  # Limit to first 50
            row = error.get("row", "?")
            ext_id = error.get("external_id", "N/A")
            err_list = error.get("errors", [])
            lines.append(f"Row {row} (ID: {ext_id}):")
            for e in err_list:
                if isinstance(e, dict):
                    lines.append(f"  - {e.get('code', 'ERROR')}: {e.get('message', 'Unknown error')}")
                else:
                    lines.append(f"  - {e}")
            lines.append("")

        if len(result.errors) > 50:
            lines.append(f"... and {len(result.errors) - 50} more errors")

    lines.extend([
        "",
        "=" * 60,
        "END OF REPORT",
        "=" * 60,
    ])

    return "\n".join(lines)
