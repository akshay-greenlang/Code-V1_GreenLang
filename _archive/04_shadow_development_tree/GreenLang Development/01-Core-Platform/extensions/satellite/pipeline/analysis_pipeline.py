"""
Analysis Pipeline Orchestrator for Satellite-Based Deforestation Detection.

Provides a unified pipeline for EUDR compliance verification:
1. Satellite imagery acquisition
2. Vegetation index calculation
3. Forest classification
4. Change detection
5. Alert integration
6. Compliance report generation

Features:
- Caching of intermediate results
- Parallel processing for multiple polygons
- Confidence scoring
- Comprehensive error handling
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional
import hashlib
import json
import logging
import threading
import time

import numpy as np

from greenlang.satellite.clients.sentinel2_client import (
    Sentinel2Client,
    Sentinel2Image,
    BoundingBox,
    SearchResult,
)
from greenlang.satellite.clients.landsat_client import (
    LandsatClient,
    LandsatImage,
    HarmonizedSatelliteClient,
)
from greenlang.satellite.analysis.vegetation_indices import (
    VegetationIndexCalculator,
    IndexType,
    IndexResult,
)
from greenlang.satellite.analysis.change_detection import (
    BiTemporalChangeDetector,
    MultiTemporalAnalyzer,
    ChangeDetectionResult,
    generate_change_report,
)
from greenlang.satellite.models.forest_classifier import (
    ForestClassifier,
    ForestClassificationResult,
    ClassificationThresholds,
)
from greenlang.satellite.alerts.deforestation_alert import (
    DeforestationAlertSystem,
    AlertAggregation,
    GeoPolygon,
    create_polygon_from_bbox,
)

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline processing stages."""
    INITIALIZATION = "initialization"
    IMAGE_ACQUISITION = "image_acquisition"
    INDEX_CALCULATION = "index_calculation"
    CLASSIFICATION = "classification"
    CHANGE_DETECTION = "change_detection"
    ALERT_INTEGRATION = "alert_integration"
    REPORT_GENERATION = "report_generation"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineConfig:
    """Configuration for analysis pipeline."""
    # Imagery settings
    max_cloud_cover: float = 20.0
    preferred_satellite: str = "sentinel2"  # "sentinel2" or "landsat"
    fallback_enabled: bool = True

    # Analysis settings
    classification_thresholds: Optional[ClassificationThresholds] = None
    pixel_size_m: float = 10.0
    scale_factor: float = 10000.0

    # Change detection
    ndvi_change_threshold: float = -0.15
    min_change_area_ha: float = 0.1

    # Alert settings
    alert_sources: list[str] = field(default_factory=lambda: ["GLAD", "RADD"])
    min_alert_confidence: str = "nominal"

    # Caching
    cache_enabled: bool = True
    cache_dir: Optional[Path] = None
    cache_ttl_days: int = 7

    # Parallel processing
    max_workers: int = 4

    # EUDR compliance
    eudr_cutoff_date: datetime = field(default_factory=lambda: datetime(2020, 12, 31))


@dataclass
class PipelineProgress:
    """Tracks pipeline execution progress."""
    stage: PipelineStage = PipelineStage.INITIALIZATION
    progress_percent: float = 0.0
    message: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get execution duration in seconds."""
        if self.started_at is None:
            return None
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()


@dataclass
class AnalysisResult:
    """Complete analysis result for a polygon."""
    polygon_id: str
    bbox: BoundingBox
    analysis_date: datetime

    # Pre-change analysis
    pre_image_date: Optional[datetime] = None
    pre_classification: Optional[ForestClassificationResult] = None
    pre_indices: Optional[dict[IndexType, IndexResult]] = None

    # Post-change analysis
    post_image_date: Optional[datetime] = None
    post_classification: Optional[ForestClassificationResult] = None
    post_indices: Optional[dict[IndexType, IndexResult]] = None

    # Change detection
    change_result: Optional[ChangeDetectionResult] = None
    change_report: Optional[dict[str, Any]] = None

    # Alerts
    alert_aggregation: Optional[AlertAggregation] = None
    eudr_compliance: Optional[dict[str, Any]] = None

    # Summary
    confidence_score: float = 0.0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "polygon_id": self.polygon_id,
            "bbox": {
                "min_lon": self.bbox.min_lon,
                "min_lat": self.bbox.min_lat,
                "max_lon": self.bbox.max_lon,
                "max_lat": self.bbox.max_lat,
            },
            "analysis_date": self.analysis_date.isoformat(),
            "pre_image_date": self.pre_image_date.isoformat() if self.pre_image_date else None,
            "post_image_date": self.post_image_date.isoformat() if self.post_image_date else None,
            "change_report": self.change_report,
            "eudr_compliance": self.eudr_compliance,
            "confidence_score": self.confidence_score,
            "warnings": self.warnings,
            "errors": self.errors,
            "metadata": self.metadata,
        }


class PipelineError(Exception):
    """Base exception for pipeline errors."""
    pass


class ImageAcquisitionError(PipelineError):
    """Failed to acquire satellite imagery."""
    pass


class AnalysisError(PipelineError):
    """Analysis processing error."""
    pass


class DeforestationAnalysisPipeline:
    """
    Main orchestrator for satellite-based deforestation analysis.

    Manages the complete workflow from image acquisition to
    EUDR compliance reporting.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        sentinel2_client: Optional[Sentinel2Client] = None,
        landsat_client: Optional[LandsatClient] = None,
        use_mock: bool = False,
    ):
        """
        Initialize analysis pipeline.

        Args:
            config: Pipeline configuration
            sentinel2_client: Sentinel-2 client instance
            landsat_client: Landsat client instance
            use_mock: Use mock data for all external services
        """
        self.config = config or PipelineConfig()
        self.use_mock = use_mock

        # Initialize clients
        self.sentinel2_client = sentinel2_client or Sentinel2Client(use_mock=use_mock)
        self.landsat_client = landsat_client or LandsatClient(use_mock=use_mock)
        self.harmonized_client = HarmonizedSatelliteClient(
            self.sentinel2_client,
            self.landsat_client,
            prefer_sentinel2=(self.config.preferred_satellite == "sentinel2")
        )

        # Initialize analysis components
        self.index_calculator = VegetationIndexCalculator(scale_factor=self.config.scale_factor)
        self.classifier = ForestClassifier(
            thresholds=self.config.classification_thresholds,
            pixel_size_m=self.config.pixel_size_m,
        )
        self.change_detector = BiTemporalChangeDetector(
            pixel_size_m=self.config.pixel_size_m
        )
        self.multi_temporal = MultiTemporalAnalyzer(pixel_size_m=self.config.pixel_size_m)
        self.alert_system = DeforestationAlertSystem(use_mock=use_mock)

        # Progress tracking
        self._progress: dict[str, PipelineProgress] = {}
        self._progress_lock = threading.Lock()

        # Cache setup
        if self.config.cache_enabled and self.config.cache_dir:
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    def _update_progress(
        self,
        polygon_id: str,
        stage: PipelineStage,
        progress: float,
        message: str,
    ) -> None:
        """Update progress for a polygon analysis."""
        with self._progress_lock:
            if polygon_id not in self._progress:
                self._progress[polygon_id] = PipelineProgress(started_at=datetime.now())

            self._progress[polygon_id].stage = stage
            self._progress[polygon_id].progress_percent = progress
            self._progress[polygon_id].message = message

            logger.debug(f"[{polygon_id}] {stage.value}: {progress:.1f}% - {message}")

    def get_progress(self, polygon_id: str) -> Optional[PipelineProgress]:
        """Get current progress for a polygon analysis."""
        with self._progress_lock:
            return self._progress.get(polygon_id)

    def _get_cache_key(self, bbox: BoundingBox, date: datetime, analysis_type: str) -> str:
        """Generate cache key."""
        key_data = f"{bbox.min_lon}_{bbox.min_lat}_{bbox.max_lon}_{bbox.max_lat}_{date.date()}_{analysis_type}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _check_cache(self, cache_key: str) -> Optional[dict]:
        """Check if cached result exists."""
        if not self.config.cache_enabled or not self.config.cache_dir:
            return None

        cache_file = self.config.cache_dir / f"{cache_key}.json"
        if not cache_file.exists():
            return None

        # Check TTL
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.now() - mtime > timedelta(days=self.config.cache_ttl_days):
            cache_file.unlink()
            return None

        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def _save_cache(self, cache_key: str, data: dict) -> None:
        """Save result to cache."""
        if not self.config.cache_enabled or not self.config.cache_dir:
            return

        cache_file = self.config.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(data, f, default=str)
        except IOError as e:
            logger.warning(f"Failed to save cache: {e}")

    def _acquire_image(
        self,
        bbox: BoundingBox,
        target_date: datetime,
        date_tolerance_days: int = 30,
    ) -> tuple[Any, str, datetime]:
        """
        Acquire satellite image for analysis.

        Returns tuple of (image, source, actual_date).
        """
        start_date = target_date - timedelta(days=date_tolerance_days)
        end_date = target_date + timedelta(days=date_tolerance_days)

        # Try Sentinel-2 first if preferred
        if self.config.preferred_satellite == "sentinel2":
            try:
                results = self.sentinel2_client.search(
                    bbox, start_date, end_date,
                    max_cloud_cover=self.config.max_cloud_cover,
                    max_results=1
                )
                if results:
                    image = self.sentinel2_client.download_image(results[0])
                    return image, "sentinel2", results[0].acquisition_date
            except Exception as e:
                logger.warning(f"Sentinel-2 acquisition failed: {e}")

        # Try Landsat if fallback enabled
        if self.config.fallback_enabled:
            try:
                from greenlang.satellite.clients.landsat_client import BoundingBox as LBBox
                l_bbox = LBBox(bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat)
                results = self.landsat_client.search(
                    l_bbox, start_date, end_date,
                    max_cloud_cover=self.config.max_cloud_cover,
                    max_results=1
                )
                if results:
                    image = self.landsat_client.download_image(results[0])
                    return image, "landsat", results[0].acquisition_date
            except Exception as e:
                logger.warning(f"Landsat acquisition failed: {e}")

        raise ImageAcquisitionError(
            f"No suitable imagery found for {target_date.date()} within {date_tolerance_days} days"
        )

    def _extract_bands(self, image: Any, source: str) -> dict[str, np.ndarray]:
        """Extract band data from image, normalizing naming."""
        if source == "sentinel2":
            return {name: band.data for name, band in image.bands.items() if band.data is not None}
        elif source == "landsat":
            return image.to_sentinel2_bands()
        else:
            raise ValueError(f"Unknown source: {source}")

    def analyze_polygon(
        self,
        polygon_id: str,
        bbox: BoundingBox,
        pre_date: datetime,
        post_date: datetime,
        include_alerts: bool = True,
        progress_callback: Optional[Callable[[PipelineProgress], None]] = None,
    ) -> AnalysisResult:
        """
        Run complete analysis for a single polygon.

        Args:
            polygon_id: Unique identifier for the polygon
            bbox: Bounding box defining the area
            pre_date: Target date for pre-change imagery
            post_date: Target date for post-change imagery
            include_alerts: Whether to fetch external alerts
            progress_callback: Optional callback for progress updates

        Returns:
            AnalysisResult with complete analysis data
        """
        logger.info(f"Starting analysis for polygon {polygon_id}")
        self._update_progress(polygon_id, PipelineStage.INITIALIZATION, 0, "Initializing analysis")

        result = AnalysisResult(
            polygon_id=polygon_id,
            bbox=bbox,
            analysis_date=datetime.now(),
        )

        try:
            # Stage 1: Image Acquisition
            self._update_progress(polygon_id, PipelineStage.IMAGE_ACQUISITION, 10, "Acquiring pre-change imagery")

            pre_image, pre_source, pre_actual_date = self._acquire_image(bbox, pre_date)
            result.pre_image_date = pre_actual_date
            result.metadata["pre_source"] = pre_source

            self._update_progress(polygon_id, PipelineStage.IMAGE_ACQUISITION, 25, "Acquiring post-change imagery")

            post_image, post_source, post_actual_date = self._acquire_image(bbox, post_date)
            result.post_image_date = post_actual_date
            result.metadata["post_source"] = post_source

            # Extract bands
            pre_bands = self._extract_bands(pre_image, pre_source)
            post_bands = self._extract_bands(post_image, post_source)

            # Stage 2: Index Calculation
            self._update_progress(polygon_id, PipelineStage.INDEX_CALCULATION, 35, "Calculating vegetation indices")

            result.pre_indices = self.index_calculator.calculate_all(pre_bands)
            result.post_indices = self.index_calculator.calculate_all(post_bands)

            # Stage 3: Classification
            self._update_progress(polygon_id, PipelineStage.CLASSIFICATION, 50, "Classifying forest cover")

            result.pre_classification = self.classifier.classify(pre_bands, self.config.scale_factor)
            result.post_classification = self.classifier.classify(post_bands, self.config.scale_factor)

            # Stage 4: Change Detection
            self._update_progress(polygon_id, PipelineStage.CHANGE_DETECTION, 65, "Detecting changes")

            result.change_result = self.change_detector.detect_change(
                pre_bands, post_bands,
                pre_actual_date, post_actual_date,
                self.config.scale_factor
            )
            result.change_report = generate_change_report(result.change_result)

            # Stage 5: Alert Integration
            if include_alerts:
                self._update_progress(polygon_id, PipelineStage.ALERT_INTEGRATION, 80, "Fetching external alerts")

                geo_polygon = create_polygon_from_bbox(
                    bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat
                )

                result.alert_aggregation = self.alert_system.get_alerts_for_polygon(
                    geo_polygon,
                    pre_actual_date,
                    post_actual_date,
                )

                result.eudr_compliance = self.alert_system.assess_eudr_compliance(
                    result.alert_aggregation,
                    cutoff_date=self.config.eudr_cutoff_date,
                )

            # Stage 6: Report Generation
            self._update_progress(polygon_id, PipelineStage.REPORT_GENERATION, 90, "Generating report")

            # Calculate overall confidence score
            confidence_factors = []

            # Data quality confidence
            if result.change_result:
                valid_fraction = result.change_result.valid_pixels / result.change_result.total_pixels
                confidence_factors.append(valid_fraction)

            # Classification confidence
            if result.post_classification:
                mean_conf = float(np.nanmean(result.post_classification.confidence_map))
                confidence_factors.append(mean_conf)

            # Source consistency (higher if same sensor used)
            if pre_source == post_source:
                confidence_factors.append(0.9)
            else:
                confidence_factors.append(0.7)
                result.warnings.append(f"Mixed sensors used: {pre_source} and {post_source}")

            result.confidence_score = float(np.mean(confidence_factors)) if confidence_factors else 0.5

            # Check for potential issues
            if result.change_result and result.change_result.forest_loss_ha > 0:
                result.warnings.append(
                    f"Forest loss detected: {result.change_result.forest_loss_ha:.2f} ha"
                )

            if result.eudr_compliance and result.eudr_compliance.get("compliance_status") == "NON_COMPLIANT":
                result.warnings.append("EUDR compliance issues detected")

            # Complete
            self._update_progress(polygon_id, PipelineStage.COMPLETED, 100, "Analysis complete")

            with self._progress_lock:
                self._progress[polygon_id].completed_at = datetime.now()

            logger.info(
                f"Analysis complete for {polygon_id}: "
                f"confidence={result.confidence_score:.2f}, "
                f"forest_loss={result.change_result.forest_loss_ha if result.change_result else 0:.2f} ha"
            )

        except ImageAcquisitionError as e:
            result.errors.append(f"Image acquisition failed: {str(e)}")
            self._update_progress(polygon_id, PipelineStage.FAILED, 0, str(e))
            with self._progress_lock:
                self._progress[polygon_id].error = str(e)
            logger.error(f"Analysis failed for {polygon_id}: {e}")

        except Exception as e:
            result.errors.append(f"Analysis error: {str(e)}")
            self._update_progress(polygon_id, PipelineStage.FAILED, 0, str(e))
            with self._progress_lock:
                self._progress[polygon_id].error = str(e)
            logger.exception(f"Unexpected error for {polygon_id}")

        # Call progress callback if provided
        if progress_callback:
            progress_callback(self._progress[polygon_id])

        return result

    def analyze_multiple_polygons(
        self,
        polygons: list[tuple[str, BoundingBox]],
        pre_date: datetime,
        post_date: datetime,
        include_alerts: bool = True,
        progress_callback: Optional[Callable[[str, PipelineProgress], None]] = None,
    ) -> dict[str, AnalysisResult]:
        """
        Analyze multiple polygons in parallel.

        Args:
            polygons: List of (polygon_id, bbox) tuples
            pre_date: Target date for pre-change imagery
            post_date: Target date for post-change imagery
            include_alerts: Whether to fetch external alerts
            progress_callback: Optional callback (polygon_id, progress)

        Returns:
            Dict mapping polygon_id to AnalysisResult
        """
        logger.info(f"Starting parallel analysis for {len(polygons)} polygons")

        results = {}

        def analyze_with_callback(polygon_id: str, bbox: BoundingBox) -> tuple[str, AnalysisResult]:
            def callback(progress: PipelineProgress) -> None:
                if progress_callback:
                    progress_callback(polygon_id, progress)

            result = self.analyze_polygon(
                polygon_id, bbox, pre_date, post_date,
                include_alerts=include_alerts,
                progress_callback=callback
            )
            return polygon_id, result

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(analyze_with_callback, pid, bbox): pid
                for pid, bbox in polygons
            }

            for future in as_completed(futures):
                polygon_id = futures[future]
                try:
                    pid, result = future.result()
                    results[pid] = result
                except Exception as e:
                    logger.error(f"Failed to process {polygon_id}: {e}")
                    results[polygon_id] = AnalysisResult(
                        polygon_id=polygon_id,
                        bbox=dict(polygons)[polygon_id],
                        analysis_date=datetime.now(),
                        errors=[str(e)]
                    )

        return results

    def generate_compliance_report(
        self,
        results: dict[str, AnalysisResult],
    ) -> dict[str, Any]:
        """
        Generate consolidated EUDR compliance report.

        Args:
            results: Dict of analysis results by polygon_id

        Returns:
            Consolidated compliance report
        """
        report = {
            "report_date": datetime.now().isoformat(),
            "total_polygons": len(results),
            "successful_analyses": sum(1 for r in results.values() if not r.errors),
            "failed_analyses": sum(1 for r in results.values() if r.errors),
            "summary": {
                "compliant": 0,
                "non_compliant": 0,
                "review_required": 0,
                "unknown": 0,
            },
            "total_area_ha": 0.0,
            "total_forest_loss_ha": 0.0,
            "polygons": {},
        }

        for polygon_id, result in results.items():
            polygon_report = {
                "analysis_date": result.analysis_date.isoformat(),
                "confidence_score": result.confidence_score,
                "warnings": result.warnings,
                "errors": result.errors,
            }

            if result.change_result:
                polygon_report["forest_loss_ha"] = result.change_result.forest_loss_ha
                polygon_report["total_area_ha"] = result.change_result.total_area_ha
                report["total_area_ha"] += result.change_result.total_area_ha
                report["total_forest_loss_ha"] += result.change_result.forest_loss_ha

            if result.eudr_compliance:
                status = result.eudr_compliance.get("compliance_status", "unknown")
                polygon_report["compliance_status"] = status
                report["summary"][status.lower()] = report["summary"].get(status.lower(), 0) + 1
            else:
                report["summary"]["unknown"] += 1

            report["polygons"][polygon_id] = polygon_report

        # Overall compliance status
        if report["summary"]["non_compliant"] > 0:
            report["overall_status"] = "NON_COMPLIANT"
        elif report["summary"]["review_required"] > 0:
            report["overall_status"] = "REVIEW_REQUIRED"
        elif report["summary"]["unknown"] > 0:
            report["overall_status"] = "INCOMPLETE"
        else:
            report["overall_status"] = "COMPLIANT"

        return report


def create_pipeline(
    use_mock: bool = False,
    config: Optional[PipelineConfig] = None,
) -> DeforestationAnalysisPipeline:
    """
    Factory function to create a configured pipeline.

    Args:
        use_mock: Use mock data for testing
        config: Optional pipeline configuration

    Returns:
        Configured DeforestationAnalysisPipeline instance
    """
    return DeforestationAnalysisPipeline(config=config, use_mock=use_mock)


def quick_analysis(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    pre_date: datetime,
    post_date: datetime,
    use_mock: bool = True,
) -> AnalysisResult:
    """
    Quick single-polygon analysis for testing.

    Args:
        min_lon: Minimum longitude
        min_lat: Minimum latitude
        max_lon: Maximum longitude
        max_lat: Maximum latitude
        pre_date: Pre-change date
        post_date: Post-change date
        use_mock: Use mock data

    Returns:
        AnalysisResult
    """
    pipeline = create_pipeline(use_mock=use_mock)
    bbox = BoundingBox(min_lon, min_lat, max_lon, max_lat)
    polygon_id = f"quick_{int(time.time())}"

    return pipeline.analyze_polygon(polygon_id, bbox, pre_date, post_date)
