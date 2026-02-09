# -*- coding: utf-8 -*-
"""
Deforestation Satellite Connector Service Setup - AGENT-DATA-007: GL-DATA-GEO-003

Provides ``configure_deforestation_satellite(app)`` which wires up the
Deforestation Satellite Connector SDK (satellite data engine, forest
change engine, alert aggregation engine, baseline assessment engine,
deforestation classifier engine, compliance report engine, monitoring
pipeline engine, provenance tracker) and mounts the REST API.

Also exposes ``get_deforestation_satellite(app)`` for programmatic access
and the ``DeforestationSatelliteService`` facade class.

Usage:
    >>> from fastapi import FastAPI
    >>> from greenlang.deforestation_satellite.setup import configure_deforestation_satellite
    >>> app = FastAPI()
    >>> import asyncio
    >>> service = asyncio.run(configure_deforestation_satellite(app))

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-007 Deforestation Satellite Connector Agent (GL-DATA-GEO-003)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.deforestation_satellite.config import get_config
from greenlang.deforestation_satellite.metrics import (
    PROMETHEUS_AVAILABLE,
    record_scene_acquired,
    record_change_detection,
    record_alert_processed,
    record_baseline_check,
    record_classification,
    record_compliance_report,
    record_pipeline_run,
    record_processing_error,
    update_active_jobs,
    update_forest_area,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FastAPI import
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI
    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None  # type: ignore[assignment, misc]
    FASTAPI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, list, str, or Pydantic model).

    Returns:
        SHA-256 hex digest string.
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    else:
        serializable = data
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ===================================================================
# Thread-safe singleton
# ===================================================================

_singleton_lock = threading.Lock()
_singleton_instance: Optional["DeforestationSatelliteService"] = None


# ===================================================================
# DeforestationSatelliteService facade
# ===================================================================


class DeforestationSatelliteService:
    """Unified facade over the Deforestation Satellite Connector SDK.

    Aggregates all 7 connector engines (satellite data, forest change,
    alert aggregation, baseline assessment, deforestation classifier,
    compliance report, monitoring pipeline) plus provenance tracker
    through a single entry point with convenience methods.

    Each method records provenance and updates self-monitoring metrics.

    Attributes:
        config: DeforestationSatelliteConfig instance.
        provenance: ProvenanceTracker instance for SHA-256 audit trails.
        satellite_engine: SatelliteDataEngine for imagery acquisition.
        change_engine: ForestChangeEngine for change detection.
        alert_engine: AlertAggregationEngine for alert integration.
        baseline_engine: BaselineAssessmentEngine for EUDR baseline.
        classifier_engine: DeforestationClassifierEngine for land cover.
        report_engine: ComplianceReportEngine for compliance reports.
        pipeline_engine: MonitoringPipelineEngine for full pipeline.

    Example:
        >>> service = DeforestationSatelliteService()
        >>> print(service.get_statistics().total_scenes)
        0
    """

    def __init__(
        self,
        config: Any = None,
    ) -> None:
        """Initialize the Deforestation Satellite Connector Service facade.

        Instantiates all 7 internal engines plus the provenance tracker.

        Args:
            config: Optional configuration. Uses global config if None.
        """
        self.config = config or get_config()

        # Initialize provenance tracker
        self._init_provenance()

        # Initialize engines
        self._init_engines()

        self._started = False
        logger.info("DeforestationSatelliteService facade created")

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_provenance(self) -> None:
        """Initialize the provenance tracker."""
        try:
            from greenlang.deforestation_satellite.provenance import ProvenanceTracker
            self.provenance = ProvenanceTracker()
        except ImportError:
            self.provenance = None
            logger.warning("ProvenanceTracker not available")

    def _init_engines(self) -> None:
        """Attempt to import and initialise SDK engines.

        Engines are optional; missing imports are logged as warnings.
        """
        # Satellite Data Engine
        try:
            from greenlang.deforestation_satellite.satellite_data import SatelliteDataEngine
            self.satellite_engine = SatelliteDataEngine(
                config=self.config, provenance=self.provenance,
            )
        except ImportError:
            self.satellite_engine = None
            logger.warning("SatelliteDataEngine not available")

        # Forest Change Engine
        try:
            from greenlang.deforestation_satellite.forest_change import ForestChangeEngine
            self.change_engine = ForestChangeEngine(
                config=self.config, provenance=self.provenance,
            )
        except ImportError:
            self.change_engine = None
            logger.warning("ForestChangeEngine not available")

        # Alert Aggregation Engine
        try:
            from greenlang.deforestation_satellite.alert_aggregation import AlertAggregationEngine
            self.alert_engine = AlertAggregationEngine(
                config=self.config, provenance=self.provenance,
            )
        except ImportError:
            self.alert_engine = None
            logger.warning("AlertAggregationEngine not available")

        # Baseline Assessment Engine
        try:
            from greenlang.deforestation_satellite.baseline_assessment import BaselineAssessmentEngine
            self.baseline_engine = BaselineAssessmentEngine(
                config=self.config, provenance=self.provenance,
            )
        except ImportError:
            self.baseline_engine = None
            logger.warning("BaselineAssessmentEngine not available")

        # Deforestation Classifier Engine
        try:
            from greenlang.deforestation_satellite.deforestation_classifier import DeforestationClassifierEngine
            self.classifier_engine = DeforestationClassifierEngine(
                config=self.config, provenance=self.provenance,
            )
        except ImportError:
            self.classifier_engine = None
            logger.warning("DeforestationClassifierEngine not available")

        # Compliance Report Engine
        try:
            from greenlang.deforestation_satellite.compliance_report import ComplianceReportEngine
            self.report_engine = ComplianceReportEngine(
                config=self.config, provenance=self.provenance,
            )
        except ImportError:
            self.report_engine = None
            logger.warning("ComplianceReportEngine not available")

        # Monitoring Pipeline Engine
        try:
            from greenlang.deforestation_satellite.monitoring_pipeline import MonitoringPipelineEngine
            self.pipeline_engine = MonitoringPipelineEngine(
                config=self.config,
                satellite_engine=self.satellite_engine,
                change_engine=self.change_engine,
                alert_engine=self.alert_engine,
                baseline_engine=self.baseline_engine,
                classifier_engine=self.classifier_engine,
                report_engine=self.report_engine,
                provenance=self.provenance,
            )
        except ImportError:
            self.pipeline_engine = None
            logger.warning("MonitoringPipelineEngine not available")

    # ------------------------------------------------------------------
    # Facade methods
    # ------------------------------------------------------------------

    def acquire_imagery(self, request: Any) -> Any:
        """Acquire satellite imagery for a polygon.

        Args:
            request: AcquireSatelliteRequest instance.

        Returns:
            SatelliteScene with populated bands.

        Raises:
            RuntimeError: If satellite engine is not available.
        """
        if self.satellite_engine is None:
            raise RuntimeError("SatelliteDataEngine not available")

        scene = self.satellite_engine.acquire(request)
        record_scene_acquired(scene.satellite, "success")
        return scene

    def detect_change(self, request: Any) -> Any:
        """Detect forest cover change between two dates.

        Args:
            request: DetectChangeRequest instance.

        Returns:
            ChangeDetectionResult with classification.

        Raises:
            RuntimeError: If change or satellite engine is not available.
        """
        if self.change_engine is None:
            raise RuntimeError("ForestChangeEngine not available")
        if self.satellite_engine is None:
            raise RuntimeError("SatelliteDataEngine not available")

        result = self.change_engine.detect_change(request, self.satellite_engine)
        record_change_detection(result.change_type, "success")
        return result

    def query_alerts(self, request: Any) -> Any:
        """Query deforestation alerts for a polygon.

        Args:
            request: QueryAlertsRequest instance.

        Returns:
            AlertAggregation with alert statistics.

        Raises:
            RuntimeError: If alert engine is not available.
        """
        if self.alert_engine is None:
            raise RuntimeError("AlertAggregationEngine not available")

        aggregation = self.alert_engine.query_alerts(request)

        # Record metrics for each source
        if aggregation.alerts_by_source:
            for source, count in aggregation.alerts_by_source.items():
                for _ in range(count):
                    record_alert_processed(source, "medium")

        return aggregation

    def check_baseline(self, request: Any) -> Any:
        """Check EUDR baseline compliance for a single coordinate.

        Args:
            request: CheckBaselineRequest instance.

        Returns:
            BaselineAssessment with compliance determination.

        Raises:
            RuntimeError: If baseline engine is not available.
        """
        if self.baseline_engine is None:
            raise RuntimeError("BaselineAssessmentEngine not available")

        assessment = self.baseline_engine.check_baseline(request)
        compliance = "compliant" if assessment.is_eudr_compliant else "non_compliant"
        record_baseline_check(assessment.country_iso3, compliance)
        return assessment

    def check_baseline_polygon(self, request: Any) -> Any:
        """Check EUDR baseline compliance for a polygon area.

        Args:
            request: CheckBaselinePolygonRequest instance.

        Returns:
            BaselineAssessment with polygon-aggregated results.

        Raises:
            RuntimeError: If baseline engine is not available.
        """
        if self.baseline_engine is None:
            raise RuntimeError("BaselineAssessmentEngine not available")

        assessment = self.baseline_engine.check_baseline_polygon(request)
        compliance = "compliant" if assessment.is_eudr_compliant else "non_compliant"
        record_baseline_check(assessment.country_iso3, compliance)
        return assessment

    def classify_land_cover(
        self,
        ndvi: float,
        evi: Optional[float] = None,
        ndwi: Optional[float] = None,
        savi: Optional[float] = None,
    ) -> Any:
        """Classify land cover from vegetation indices.

        Args:
            ndvi: NDVI value.
            evi: Optional EVI value.
            ndwi: Optional NDWI value.
            savi: Optional SAVI value.

        Returns:
            ForestClassification result.

        Raises:
            RuntimeError: If classifier engine is not available.
        """
        if self.classifier_engine is None:
            raise RuntimeError("DeforestationClassifierEngine not available")

        classification = self.classifier_engine.classify(
            ndvi=ndvi, evi=evi, ndwi=ndwi, savi=savi,
        )
        record_classification(classification.land_cover_class)
        return classification

    def generate_compliance_report(
        self,
        baseline: Any,
        alerts: Any,
        polygon_wkt: str,
    ) -> Any:
        """Generate an EUDR compliance report.

        Args:
            baseline: BaselineAssessment instance.
            alerts: AlertAggregation instance.
            polygon_wkt: WKT polygon string.

        Returns:
            ComplianceReport with full determination.

        Raises:
            RuntimeError: If report engine is not available.
        """
        if self.report_engine is None:
            raise RuntimeError("ComplianceReportEngine not available")

        report = self.report_engine.generate_report(baseline, alerts, polygon_wkt)
        record_compliance_report(report.compliance_status)
        return report

    def start_monitoring(self, request: Any) -> Any:
        """Start a deforestation monitoring job.

        Args:
            request: StartMonitoringRequest instance.

        Returns:
            MonitoringJob with pipeline results.

        Raises:
            RuntimeError: If pipeline engine is not available.
        """
        if self.pipeline_engine is None:
            raise RuntimeError("MonitoringPipelineEngine not available")

        job = self.pipeline_engine.start_monitoring(request)
        update_active_jobs(self.pipeline_engine._stats.active_jobs)
        return job

    def get_monitoring_status(self, job_id: str) -> Any:
        """Get the status of a monitoring job.

        Args:
            job_id: Monitoring job identifier.

        Returns:
            MonitoringJob or None if not found.

        Raises:
            RuntimeError: If pipeline engine is not available.
        """
        if self.pipeline_engine is None:
            raise RuntimeError("MonitoringPipelineEngine not available")

        return self.pipeline_engine.get_job(job_id)

    def stop_monitoring(self, job_id: str) -> bool:
        """Stop a running monitoring job.

        Args:
            job_id: Monitoring job identifier.

        Returns:
            True if the job was stopped.

        Raises:
            RuntimeError: If pipeline engine is not available.
        """
        if self.pipeline_engine is None:
            raise RuntimeError("MonitoringPipelineEngine not available")

        result = self.pipeline_engine.stop_job(job_id)
        update_active_jobs(self.pipeline_engine._stats.active_jobs)
        return result

    def get_statistics(self) -> Any:
        """Get aggregate service statistics.

        Returns:
            DeforestationStatistics summary.
        """
        if self.pipeline_engine is not None:
            return self.pipeline_engine.get_statistics()

        # Fallback: build stats from individual engines
        from greenlang.deforestation_satellite.models import DeforestationStatistics
        stats = DeforestationStatistics()
        if self.satellite_engine:
            stats.total_scenes = self.satellite_engine.scene_count
        if self.baseline_engine:
            stats.total_assessments = self.baseline_engine.assessment_count
        if self.report_engine:
            stats.total_reports = self.report_engine.report_count
        return stats

    # ------------------------------------------------------------------
    # Convenience getters
    # ------------------------------------------------------------------

    def get_provenance(self) -> Any:
        """Get the ProvenanceTracker instance.

        Returns:
            ProvenanceTracker used by this service.
        """
        return self.provenance

    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics summary.

        Returns:
            Dictionary with service metric summaries.
        """
        stats = self.get_statistics()
        return {
            "prometheus_available": PROMETHEUS_AVAILABLE,
            "started": self._started,
            "total_scenes": stats.total_scenes,
            "total_assessments": stats.total_assessments,
            "total_alerts": stats.total_alerts,
            "total_reports": stats.total_reports,
            "total_monitoring_jobs": stats.total_monitoring_jobs,
            "active_jobs": stats.active_jobs,
            "forest_area_monitored_ha": stats.forest_area_monitored_ha,
            "compliance_rate_percent": stats.compliance_rate_percent,
            "provenance_entries": self.provenance.entry_count if self.provenance else 0,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def startup(self) -> None:
        """Start the deforestation satellite connector service.

        Safe to call multiple times.
        """
        if self._started:
            logger.debug("DeforestationSatelliteService already started; skipping")
            return

        logger.info("DeforestationSatelliteService starting up...")
        self._started = True
        logger.info("DeforestationSatelliteService startup complete")

    def shutdown(self) -> None:
        """Shutdown the service and release resources."""
        if not self._started:
            return

        self._started = False
        logger.info("DeforestationSatelliteService shut down")


# ===================================================================
# Thread-safe singleton access
# ===================================================================


def _get_singleton() -> DeforestationSatelliteService:
    """Get or create the singleton DeforestationSatelliteService instance.

    Returns:
        The singleton DeforestationSatelliteService.
    """
    global _singleton_instance
    if _singleton_instance is None:
        with _singleton_lock:
            if _singleton_instance is None:
                _singleton_instance = DeforestationSatelliteService()
    return _singleton_instance


# ===================================================================
# FastAPI integration
# ===================================================================


async def configure_deforestation_satellite(
    app: Any,
    config: Any = None,
) -> DeforestationSatelliteService:
    """Configure the Deforestation Satellite Connector Service on a FastAPI application.

    Creates the DeforestationSatelliteService, stores it in app.state,
    mounts the REST API router, and starts the service.

    Args:
        app: FastAPI application instance.
        config: Optional DeforestationSatelliteConfig.

    Returns:
        DeforestationSatelliteService instance.
    """
    global _singleton_instance

    service = DeforestationSatelliteService(config=config)

    # Store as singleton
    with _singleton_lock:
        _singleton_instance = service

    # Attach to app state
    app.state.deforestation_satellite_service = service

    # Mount API router
    try:
        from greenlang.deforestation_satellite.api.router import router as deforestation_router
        if deforestation_router is not None:
            app.include_router(deforestation_router)
            logger.info("Deforestation satellite API router mounted")
    except ImportError:
        logger.warning("Deforestation satellite router not available; API not mounted")

    # Start service
    service.startup()

    logger.info("Deforestation satellite connector service configured on app")
    return service


def get_deforestation_satellite(app: Any) -> DeforestationSatelliteService:
    """Get the DeforestationSatelliteService instance from app state.

    Args:
        app: FastAPI application instance.

    Returns:
        DeforestationSatelliteService instance.

    Raises:
        RuntimeError: If service not configured.
    """
    service = getattr(app.state, "deforestation_satellite_service", None)
    if service is None:
        raise RuntimeError(
            "Deforestation satellite service not configured. "
            "Call configure_deforestation_satellite(app) first."
        )
    return service


def get_router(service: Optional[DeforestationSatelliteService] = None) -> Any:
    """Get the deforestation satellite API router.

    Args:
        service: Optional service instance (unused, kept for API compat).

    Returns:
        FastAPI APIRouter or None if FastAPI not available.
    """
    try:
        from greenlang.deforestation_satellite.api.router import router
        return router
    except ImportError:
        return None


__all__ = [
    "DeforestationSatelliteService",
    "configure_deforestation_satellite",
    "get_deforestation_satellite",
    "get_router",
]
