# -*- coding: utf-8 -*-
"""
Monitoring Pipeline Engine - AGENT-DATA-007: GL-DATA-GEO-003

Orchestrates the 7-stage deforestation monitoring pipeline from satellite
imagery acquisition through EUDR compliance report generation. Manages
monitoring jobs lifecycle and tracks pipeline stage execution.

Pipeline Stages:
    1. INITIALIZATION      - Job setup, validation, configuration
    2. IMAGE_ACQUISITION   - Satellite scene acquisition
    3. INDEX_CALCULATION   - Vegetation index computation
    4. CLASSIFICATION      - Land cover classification
    5. CHANGE_DETECTION    - Bi-temporal change detection
    6. ALERT_INTEGRATION   - External alert aggregation
    7. REPORT_GENERATION   - EUDR compliance report generation

Features:
    - Sequential 7-stage pipeline execution with per-stage timing
    - Monitoring job lifecycle management (start, stop, list)
    - Pipeline result tracking with stage-level metadata
    - Service-wide statistics aggregation
    - All 7 sub-engines coordinated through single pipeline
    - Provenance tracking for all pipeline operations

Zero-Hallucination Guarantees:
    - Pipeline stages execute in deterministic sequential order
    - Each stage delegates to the appropriate engine
    - No stochastic or parallel execution non-determinism
    - Timing is wall-clock measured, not estimated

Example:
    >>> from greenlang.deforestation_satellite.monitoring_pipeline import MonitoringPipelineEngine
    >>> engine = MonitoringPipelineEngine()
    >>> from greenlang.deforestation_satellite.models import StartMonitoringRequest
    >>> request = StartMonitoringRequest(
    ...     polygon_coordinates=[[-60.0, -3.0], [-59.0, -3.0],
    ...                          [-59.0, -2.0], [-60.0, -2.0], [-60.0, -3.0]],
    ...     country_iso3="BRA",
    ... )
    >>> job = engine.start_monitoring(request)
    >>> print(job.is_running, job.current_stage)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-007 Deforestation Satellite Connector Agent (GL-DATA-GEO-003)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from greenlang.deforestation_satellite.config import get_config
from greenlang.deforestation_satellite.models import (
    AcquireSatelliteRequest,
    AlertAggregation,
    CheckBaselinePolygonRequest,
    ComplianceReport,
    DeforestationStatistics,
    DetectChangeRequest,
    MonitoringJob,
    PipelineResult,
    PipelineStage,
    QueryAlertsRequest,
    StartMonitoringRequest,
    VegetationIndex,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default dates for pipeline execution when not specified
_DEFAULT_LOOKBACK_DAYS = 365


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _polygon_to_wkt(polygon_coords: List[List[float]]) -> str:
    """Convert polygon coordinate list to WKT string."""
    if not polygon_coords:
        return "POLYGON EMPTY"
    pairs = " ".join(f"{c[0]} {c[1]}" for c in polygon_coords)
    return f"POLYGON(({pairs}))"


# =============================================================================
# MonitoringPipelineEngine
# =============================================================================


class MonitoringPipelineEngine:
    """Engine for orchestrating the 7-stage deforestation monitoring pipeline.

    Coordinates all sub-engines (satellite data, forest change, alert
    aggregation, baseline assessment, classifier, compliance report)
    through a sequential pipeline with per-stage timing and result tracking.

    Class Constants:
        PIPELINE_STAGES: Ordered list of 7 PipelineStage values defining
            the execution sequence.

    Attributes:
        config: DeforestationSatelliteConfig instance.
        satellite_engine: SatelliteDataEngine for imagery.
        change_engine: ForestChangeEngine for change detection.
        alert_engine: AlertAggregationEngine for alert integration.
        baseline_engine: BaselineAssessmentEngine for EUDR baseline.
        classifier_engine: DeforestationClassifierEngine for land cover.
        report_engine: ComplianceReportEngine for compliance reports.
        provenance: Optional ProvenanceTracker for audit trails.

    Example:
        >>> engine = MonitoringPipelineEngine()
        >>> print(engine.job_count)
        0
    """

    # Ordered pipeline stages
    PIPELINE_STAGES: List[PipelineStage] = [
        PipelineStage.INITIALIZATION,
        PipelineStage.IMAGE_ACQUISITION,
        PipelineStage.INDEX_CALCULATION,
        PipelineStage.CLASSIFICATION,
        PipelineStage.CHANGE_DETECTION,
        PipelineStage.ALERT_INTEGRATION,
        PipelineStage.REPORT_GENERATION,
    ]

    def __init__(
        self,
        config: Any = None,
        satellite_engine: Any = None,
        change_engine: Any = None,
        alert_engine: Any = None,
        baseline_engine: Any = None,
        classifier_engine: Any = None,
        report_engine: Any = None,
        provenance: Any = None,
    ) -> None:
        """Initialize MonitoringPipelineEngine.

        All engine dependencies are optional; when None, they are
        lazily imported and instantiated from the SDK modules.

        Args:
            config: Optional DeforestationSatelliteConfig.
            satellite_engine: Optional SatelliteDataEngine.
            change_engine: Optional ForestChangeEngine.
            alert_engine: Optional AlertAggregationEngine.
            baseline_engine: Optional BaselineAssessmentEngine.
            classifier_engine: Optional DeforestationClassifierEngine.
            report_engine: Optional ComplianceReportEngine.
            provenance: Optional ProvenanceTracker.
        """
        self.config = config or get_config()
        self.provenance = provenance

        # Sub-engines: use provided or lazy-init
        self.satellite_engine = satellite_engine
        self.change_engine = change_engine
        self.alert_engine = alert_engine
        self.baseline_engine = baseline_engine
        self.classifier_engine = classifier_engine
        self.report_engine = report_engine

        self._init_engines()

        # Job storage
        self._jobs: Dict[str, MonitoringJob] = {}
        self._pipeline_results: Dict[str, List[PipelineResult]] = {}
        self._job_count: int = 0

        # Statistics
        self._stats = DeforestationStatistics()

        logger.info(
            "MonitoringPipelineEngine initialized: %d stages",
            len(self.PIPELINE_STAGES),
        )

    # ------------------------------------------------------------------
    # Engine initialization
    # ------------------------------------------------------------------

    def _init_engines(self) -> None:
        """Lazy-initialize sub-engines if not provided at construction."""
        if self.satellite_engine is None:
            try:
                from greenlang.deforestation_satellite.satellite_data import SatelliteDataEngine
                self.satellite_engine = SatelliteDataEngine(
                    config=self.config, provenance=self.provenance,
                )
            except ImportError:
                logger.warning("SatelliteDataEngine not available")

        if self.change_engine is None:
            try:
                from greenlang.deforestation_satellite.forest_change import ForestChangeEngine
                self.change_engine = ForestChangeEngine(
                    config=self.config, provenance=self.provenance,
                )
            except ImportError:
                logger.warning("ForestChangeEngine not available")

        if self.alert_engine is None:
            try:
                from greenlang.deforestation_satellite.alert_aggregation import AlertAggregationEngine
                self.alert_engine = AlertAggregationEngine(
                    config=self.config, provenance=self.provenance,
                )
            except ImportError:
                logger.warning("AlertAggregationEngine not available")

        if self.baseline_engine is None:
            try:
                from greenlang.deforestation_satellite.baseline_assessment import BaselineAssessmentEngine
                self.baseline_engine = BaselineAssessmentEngine(
                    config=self.config, provenance=self.provenance,
                )
            except ImportError:
                logger.warning("BaselineAssessmentEngine not available")

        if self.classifier_engine is None:
            try:
                from greenlang.deforestation_satellite.deforestation_classifier import DeforestationClassifierEngine
                self.classifier_engine = DeforestationClassifierEngine(
                    config=self.config, provenance=self.provenance,
                )
            except ImportError:
                logger.warning("DeforestationClassifierEngine not available")

        if self.report_engine is None:
            try:
                from greenlang.deforestation_satellite.compliance_report import ComplianceReportEngine
                self.report_engine = ComplianceReportEngine(
                    config=self.config, provenance=self.provenance,
                )
            except ImportError:
                logger.warning("ComplianceReportEngine not available")

    # ------------------------------------------------------------------
    # Monitoring job lifecycle
    # ------------------------------------------------------------------

    def start_monitoring(self, request: StartMonitoringRequest) -> MonitoringJob:
        """Create and start a deforestation monitoring job.

        Initializes a new monitoring job, executes the 7-stage pipeline,
        and stores the final compliance report.

        Args:
            request: Monitoring start request with polygon, country, and
                frequency.

        Returns:
            MonitoringJob with pipeline execution results.

        Raises:
            ValueError: If polygon_coordinates or country_iso3 is empty.
        """
        if not request.polygon_coordinates:
            raise ValueError("polygon_coordinates must not be empty")
        if not request.country_iso3:
            raise ValueError("country_iso3 is required")

        job_id = self._generate_job_id()
        polygon_wkt = _polygon_to_wkt(request.polygon_coordinates)

        job = MonitoringJob(
            job_id=job_id,
            polygon_wkt=polygon_wkt,
            country_iso3=request.country_iso3.upper(),
            frequency=request.frequency or "monthly",
            current_stage=PipelineStage.INITIALIZATION.value,
            is_running=True,
        )

        # Store job
        self._jobs[job_id] = job
        self._pipeline_results[job_id] = []
        self._job_count += 1
        self._stats.total_monitoring_jobs += 1
        self._stats.active_jobs += 1

        logger.info(
            "Monitoring job %s started: country=%s, frequency=%s, polygon=%s",
            job_id, request.country_iso3, request.frequency, polygon_wkt[:60],
        )

        # Execute pipeline
        try:
            report = self.run_pipeline(job, request)
            job.is_running = False
            job.completed_at = _utcnow().isoformat()
            if report:
                job.last_result = report.model_dump(mode="json")
            self._stats.active_jobs = max(0, self._stats.active_jobs - 1)

        except Exception as exc:
            job.is_running = False
            job.error_message = str(exc)
            self._stats.active_jobs = max(0, self._stats.active_jobs - 1)
            logger.error(
                "Monitoring job %s failed: %s", job_id, exc, exc_info=True,
            )

        # Record provenance
        if self.provenance is not None:
            data_hash = hashlib.sha256(
                json.dumps(job.model_dump(mode="json"), sort_keys=True, default=str).encode()
            ).hexdigest()
            self.provenance.record(
                entity_type="pipeline_execution",
                entity_id=job_id,
                action="complete",
                data_hash=data_hash,
            )

        return job

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        job: MonitoringJob,
        request: Optional[StartMonitoringRequest] = None,
    ) -> Optional[ComplianceReport]:
        """Execute the 7-stage monitoring pipeline.

        Runs each stage sequentially, recording duration and results
        per stage. Returns the final compliance report.

        Args:
            job: MonitoringJob to execute pipeline for.
            request: Optional StartMonitoringRequest for stage parameters.

        Returns:
            ComplianceReport from the final stage, or None on failure.
        """
        polygon_coords = []
        country_iso3 = job.country_iso3
        satellite = None

        if request:
            polygon_coords = request.polygon_coordinates
            satellite = request.satellite
        else:
            # Parse polygon from WKT (simplified)
            polygon_coords = []

        # Shared pipeline context
        context: Dict[str, Any] = {
            "polygon_coords": polygon_coords,
            "polygon_wkt": job.polygon_wkt,
            "country_iso3": country_iso3,
            "satellite": satellite or self.config.default_satellite,
            "scene": None,
            "indices": {},
            "classification": None,
            "change_result": None,
            "alert_aggregation": None,
            "baseline": None,
            "report": None,
        }

        now = _utcnow()
        end_date = now.strftime("%Y-%m-%d")
        start_date = (now - timedelta(days=_DEFAULT_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        pre_start = (now - timedelta(days=_DEFAULT_LOOKBACK_DAYS * 2)).strftime("%Y-%m-%d")
        pre_end = (now - timedelta(days=_DEFAULT_LOOKBACK_DAYS)).strftime("%Y-%m-%d")

        context["start_date"] = start_date
        context["end_date"] = end_date
        context["pre_start"] = pre_start
        context["pre_end"] = pre_end

        for stage in self.PIPELINE_STAGES:
            if not job.is_running:
                logger.warning("Job %s stopped; aborting pipeline", job.job_id)
                break

            result = self._run_stage(job, stage, context)
            self._pipeline_results[job.job_id].append(result)

            if result.status == "failed":
                logger.error(
                    "Pipeline stage %s failed for job %s",
                    stage.value, job.job_id,
                )
                break

        return context.get("report")

    def _run_stage(
        self,
        job: MonitoringJob,
        stage: PipelineStage,
        context: Dict[str, Any],
    ) -> PipelineResult:
        """Execute a single pipeline stage and record metrics.

        Args:
            job: Parent monitoring job.
            stage: Pipeline stage to execute.
            context: Shared pipeline context dictionary.

        Returns:
            PipelineResult with stage execution details.
        """
        start_time = time.time()
        job.current_stage = stage.value
        result_data: Dict[str, Any] = {}
        status = "completed"

        try:
            if stage == PipelineStage.INITIALIZATION:
                result_data = self._stage_initialization(context)

            elif stage == PipelineStage.IMAGE_ACQUISITION:
                result_data = self._stage_image_acquisition(context)

            elif stage == PipelineStage.INDEX_CALCULATION:
                result_data = self._stage_index_calculation(context)

            elif stage == PipelineStage.CLASSIFICATION:
                result_data = self._stage_classification(context)

            elif stage == PipelineStage.CHANGE_DETECTION:
                result_data = self._stage_change_detection(context)

            elif stage == PipelineStage.ALERT_INTEGRATION:
                result_data = self._stage_alert_integration(context)

            elif stage == PipelineStage.REPORT_GENERATION:
                result_data = self._stage_report_generation(context)

        except Exception as exc:
            status = "failed"
            result_data = {"error": str(exc)}
            logger.error(
                "Stage %s failed for job %s: %s",
                stage.value, job.job_id, exc,
            )

        duration = time.time() - start_time

        if status == "completed":
            job.stages_completed.append(stage.value)

        pipeline_id = self._generate_pipeline_id()

        result = PipelineResult(
            pipeline_id=pipeline_id,
            job_id=job.job_id,
            stage=stage.value,
            status=status,
            result_data=result_data,
            duration_seconds=round(duration, 4),
        )

        logger.info(
            "Pipeline stage %s for job %s: status=%s, duration=%.3fs",
            stage.value, job.job_id, status, duration,
        )

        return result

    # ------------------------------------------------------------------
    # Individual stage implementations
    # ------------------------------------------------------------------

    def _stage_initialization(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 1: Initialize pipeline context and validate inputs."""
        return {
            "polygon_wkt": context["polygon_wkt"],
            "country_iso3": context["country_iso3"],
            "satellite": context["satellite"],
            "date_range": {
                "start": context.get("start_date", ""),
                "end": context.get("end_date", ""),
            },
            "status": "initialized",
        }

    def _stage_image_acquisition(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 2: Acquire satellite imagery."""
        if self.satellite_engine is None:
            return {"status": "skipped", "reason": "satellite_engine not available"}

        polygon_coords = context.get("polygon_coords", [])
        if not polygon_coords:
            return {"status": "skipped", "reason": "no polygon coordinates"}

        request = AcquireSatelliteRequest(
            polygon_coordinates=polygon_coords,
            satellite=context.get("satellite"),
            start_date=context.get("start_date", ""),
            end_date=context.get("end_date", ""),
        )
        scene = self.satellite_engine.acquire(request)
        context["scene"] = scene
        self._stats.total_scenes += 1

        return {
            "scene_id": scene.scene_id,
            "satellite": scene.satellite,
            "acquisition_date": scene.acquisition_date,
            "cloud_cover": scene.cloud_cover_percent,
            "band_count": len(scene.bands),
        }

    def _stage_index_calculation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 3: Compute vegetation indices."""
        if self.satellite_engine is None or context.get("scene") is None:
            return {"status": "skipped", "reason": "no scene available"}

        scene = context["scene"]
        indices = self.satellite_engine.calculate_indices(
            scene,
            [VegetationIndex.NDVI, VegetationIndex.EVI,
             VegetationIndex.NDWI, VegetationIndex.NBR],
        )
        context["indices"] = indices

        summary = {}
        for idx_type, idx_result in indices.items():
            summary[idx_type.value] = {
                "mean": idx_result.mean_value,
                "min": idx_result.min_value,
                "max": idx_result.max_value,
            }

        return {
            "indices_computed": list(summary.keys()),
            "summary": summary,
        }

    def _stage_classification(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 4: Classify land cover."""
        if self.classifier_engine is None:
            return {"status": "skipped", "reason": "classifier_engine not available"}

        indices = context.get("indices", {})
        ndvi_result = indices.get(VegetationIndex.NDVI)
        evi_result = indices.get(VegetationIndex.EVI)
        ndwi_result = indices.get(VegetationIndex.NDWI)

        ndvi_mean = ndvi_result.mean_value if ndvi_result else 0.5
        evi_mean = evi_result.mean_value if evi_result else None
        ndwi_mean = ndwi_result.mean_value if ndwi_result else None

        classification = self.classifier_engine.classify(
            ndvi=ndvi_mean, evi=evi_mean, ndwi=ndwi_mean,
        )
        context["classification"] = classification

        return {
            "classification_id": classification.classification_id,
            "land_cover_class": classification.land_cover_class,
            "tree_cover_percent": classification.tree_cover_percent,
            "is_forest": classification.is_forest,
            "confidence": classification.confidence,
        }

    def _stage_change_detection(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 5: Detect forest cover change."""
        if self.change_engine is None or self.satellite_engine is None:
            return {"status": "skipped", "reason": "engines not available"}

        polygon_coords = context.get("polygon_coords", [])
        if not polygon_coords:
            return {"status": "skipped", "reason": "no polygon coordinates"}

        request = DetectChangeRequest(
            polygon_coordinates=polygon_coords,
            pre_start_date=context.get("pre_start", ""),
            pre_end_date=context.get("start_date", ""),
            post_start_date=context.get("start_date", ""),
            post_end_date=context.get("end_date", ""),
            satellite=context.get("satellite"),
        )
        change_result = self.change_engine.detect_change(
            request, self.satellite_engine,
        )
        context["change_result"] = change_result

        return {
            "change_id": change_result.change_id,
            "change_type": change_result.change_type,
            "delta_ndvi": change_result.delta_ndvi,
            "area_ha": change_result.area_ha,
            "confidence": change_result.confidence,
        }

    def _stage_alert_integration(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 6: Integrate external deforestation alerts."""
        if self.alert_engine is None:
            return {"status": "skipped", "reason": "alert_engine not available"}

        polygon_coords = context.get("polygon_coords", [])
        if not polygon_coords:
            return {"status": "skipped", "reason": "no polygon coordinates"}

        request = QueryAlertsRequest(
            polygon_coordinates=polygon_coords,
            start_date=context.get("start_date", ""),
            end_date=context.get("end_date", ""),
        )
        aggregation = self.alert_engine.query_alerts(request)
        context["alert_aggregation"] = aggregation
        self._stats.total_alerts += aggregation.total_alerts

        return {
            "aggregation_id": aggregation.aggregation_id,
            "total_alerts": aggregation.total_alerts,
            "alerts_by_source": aggregation.alerts_by_source,
            "has_critical": aggregation.has_critical,
            "total_affected_area_ha": aggregation.total_affected_area_ha,
        }

    def _stage_report_generation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 7: Generate EUDR compliance report."""
        # First, run baseline assessment
        baseline = None
        if self.baseline_engine is not None:
            polygon_coords = context.get("polygon_coords", [])
            country = context.get("country_iso3", "")
            if polygon_coords and country:
                baseline_request = CheckBaselinePolygonRequest(
                    polygon_coordinates=polygon_coords,
                    country_iso3=country,
                )
                baseline = self.baseline_engine.check_baseline_polygon(baseline_request)
                context["baseline"] = baseline
                self._stats.total_assessments += 1

        if baseline is None:
            return {"status": "skipped", "reason": "baseline_engine not available"}

        alert_aggregation = context.get("alert_aggregation")
        if alert_aggregation is None:
            # Create empty aggregation
            alert_aggregation = AlertAggregation(
                polygon_wkt=context.get("polygon_wkt", ""),
                date_range_start=context.get("start_date", ""),
                date_range_end=context.get("end_date", ""),
            )

        if self.report_engine is not None:
            report = self.report_engine.generate_report(
                baseline=baseline,
                alerts=alert_aggregation,
                polygon_wkt=context.get("polygon_wkt", ""),
            )
            context["report"] = report
            self._stats.total_reports += 1

            return {
                "report_id": report.report_id,
                "compliance_status": report.compliance_status,
                "risk_score": report.risk_score,
                "risk_level": report.risk_level,
                "total_alerts": report.total_alerts,
            }

        return {"status": "skipped", "reason": "report_engine not available"}

    # ------------------------------------------------------------------
    # Job management
    # ------------------------------------------------------------------

    def get_job(self, job_id: str) -> Optional[MonitoringJob]:
        """Retrieve a monitoring job by ID.

        Args:
            job_id: Unique monitoring job identifier.

        Returns:
            MonitoringJob or None if not found.
        """
        return self._jobs.get(job_id)

    def stop_job(self, job_id: str) -> bool:
        """Stop a running monitoring job.

        Args:
            job_id: Unique monitoring job identifier.

        Returns:
            True if the job was stopped, False if not found or not running.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return False
        if not job.is_running:
            return False

        job.is_running = False
        job.completed_at = _utcnow().isoformat()
        job.error_message = "Stopped by user"
        self._stats.active_jobs = max(0, self._stats.active_jobs - 1)

        logger.info("Monitoring job %s stopped", job_id)
        return True

    def list_jobs(self) -> List[MonitoringJob]:
        """List all monitoring jobs.

        Returns:
            List of MonitoringJob instances ordered by start time.
        """
        return sorted(
            self._jobs.values(),
            key=lambda j: j.started_at,
            reverse=True,
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> DeforestationStatistics:
        """Get aggregate service statistics.

        Computes compliance rate from stored reports and returns
        up-to-date statistics.

        Returns:
            DeforestationStatistics summary.
        """
        # Update compliance rate
        if self.report_engine and self.report_engine.report_count > 0:
            compliant_count = sum(
                1 for r in self.report_engine.list_reports()
                if r.compliance_status == ComplianceStatus.COMPLIANT.value
            )
            total = self.report_engine.report_count
            self._stats.compliance_rate_percent = round(
                (compliant_count / total) * 100.0, 2,
            ) if total > 0 else 0.0

        # Import only needed for type reference
        from greenlang.deforestation_satellite.models import ComplianceStatus

        return self._stats

    def get_pipeline_results(self, job_id: str) -> List[PipelineResult]:
        """Get pipeline stage results for a specific job.

        Args:
            job_id: Monitoring job identifier.

        Returns:
            List of PipelineResult instances for the job.
        """
        return self._pipeline_results.get(job_id, [])

    # ------------------------------------------------------------------
    # ID generation
    # ------------------------------------------------------------------

    def _generate_job_id(self) -> str:
        """Generate a unique monitoring job identifier.

        Returns:
            String in format "MON-{12 hex chars}".
        """
        return f"MON-{uuid.uuid4().hex[:12]}"

    def _generate_pipeline_id(self) -> str:
        """Generate a unique pipeline result identifier.

        Returns:
            String in format "PIP-{12 hex chars}".
        """
        return f"PIP-{uuid.uuid4().hex[:12]}"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def job_count(self) -> int:
        """Return the total number of monitoring jobs created.

        Returns:
            Integer count of monitoring jobs.
        """
        return self._job_count


__all__ = [
    "MonitoringPipelineEngine",
]
