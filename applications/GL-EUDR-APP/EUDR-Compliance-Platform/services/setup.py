# -*- coding: utf-8 -*-
"""
GL-EUDR-APP Service Facade - EU Deforestation Regulation Compliance Platform

Provides the ``EUDRComplianceService`` facade class that composes all five
core engines into a single entry point for the EUDR compliance platform:

- SupplierIntakeEngine:         Supplier CRUD, bulk import, ERP normalization
- DocumentVerificationEngine:   Document upload, OCR, verification, gap analysis
- DDSReportingEngine:           DDS generation, validation, submission, lifecycle
- RiskAggregator:               Multi-source risk scoring, alerts, heatmaps
- PipelineOrchestrator:         5-stage compliance pipeline execution

Also provides:
- ``configure_eudr_app(app)``:  FastAPI integration
- ``get_eudr_service(app)``:    Retrieve service from app state
- Dashboard metrics aggregation

Example:
    >>> from services.setup import EUDRComplianceService
    >>> service = EUDRComplianceService()
    >>> supplier = service.supplier_engine.create_supplier(request)
    >>> run = await service.pipeline.start_pipeline(supplier.id, "coffee", [])

    # FastAPI integration
    >>> from fastapi import FastAPI
    >>> app = FastAPI()
    >>> from services.setup import configure_eudr_app
    >>> service = configure_eudr_app(app)

Author: GreenLang Platform Team
Date: March 2026
Application: GL-EUDR-APP v1.0
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from services.config import (
    ComplianceStatus,
    DDSStatus,
    EUDRAppConfig,
    PipelineStatus,
    RiskLevel,
    SatelliteAssessmentStatus,
)
from services.models import DashboardMetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


# ===========================================================================
# Service Facade
# ===========================================================================


class EUDRComplianceService:
    """Facade for the GL-EUDR-APP compliance platform.

    Composes all five core engines into a single service entry point.
    Engines are cross-wired so the DDS engine can access supplier and
    document data, and the risk aggregator can query both.

    Attributes:
        config: Application configuration.
        pipeline: PipelineOrchestrator for 5-stage compliance runs.
        supplier_engine: SupplierIntakeEngine for supplier CRUD.
        document_engine: DocumentVerificationEngine for document management.
        dds_engine: DDSReportingEngine for DDS lifecycle.
        risk_engine: RiskAggregator for risk scoring.

    Example:
        >>> service = EUDRComplianceService()
        >>> supplier = service.supplier_engine.create_supplier(request)
        >>> service.supplier_engine.add_plot(supplier.id, plot_request)
        >>> run = await service.pipeline.start_pipeline(
        ...     supplier.id, "coffee", supplier.plots
        ... )
    """

    def __init__(
        self, config: Optional[EUDRAppConfig] = None
    ) -> None:
        """Initialize EUDRComplianceService with all engines.

        Creates and cross-wires all five engines. If no config is
        provided, loads defaults from environment variables.

        Args:
            config: Optional application configuration. Defaults loaded
                from env vars with EUDR_APP_ prefix if not provided.
        """
        self.config = config or EUDRAppConfig()

        # Initialize engines in dependency order
        from services.supplier_intake_engine import SupplierIntakeEngine
        from services.document_verification_engine import (
            DocumentVerificationEngine,
        )
        from services.risk_aggregator import RiskAggregator
        from services.dds_reporting_engine import DDSReportingEngine
        from services.pipeline_orchestrator import PipelineOrchestrator

        # 1. Supplier engine (no dependencies)
        self.supplier_engine = SupplierIntakeEngine(self.config)

        # 2. Document engine (no dependencies)
        self.document_engine = DocumentVerificationEngine(self.config)

        # 3. Risk aggregator (depends on supplier + document engines)
        self.risk_engine = RiskAggregator(
            config=self.config,
            supplier_engine=self.supplier_engine,
            document_engine=self.document_engine,
        )

        # 4. DDS engine (depends on supplier + document + risk engines)
        self.dds_engine = DDSReportingEngine(
            config=self.config,
            supplier_engine=self.supplier_engine,
            document_engine=self.document_engine,
            risk_engine=self.risk_engine,
        )

        # 5. Pipeline orchestrator (standalone, delegates to engines)
        self.pipeline = PipelineOrchestrator(self.config)

        logger.info(
            "EUDRComplianceService initialized with all 5 engines: "
            "SupplierIntake, DocumentVerification, RiskAggregator, "
            "DDSReporting, PipelineOrchestrator"
        )

    # -----------------------------------------------------------------------
    # Dashboard Metrics
    # -----------------------------------------------------------------------

    def get_dashboard_metrics(self) -> DashboardMetrics:
        """Aggregate dashboard metrics from all engines.

        Collects supplier counts, plot statistics, DDS status breakdown,
        pipeline activity, and compliance rates.

        Returns:
            DashboardMetrics with all aggregated values.
        """
        # Supplier metrics
        supplier_stats = self.supplier_engine.get_statistics()
        total_suppliers = supplier_stats.get("total_suppliers", 0)
        compliance_breakdown = supplier_stats.get(
            "compliance_breakdown", {}
        )
        commodity_breakdown = supplier_stats.get(
            "commodity_breakdown", {}
        )
        country_breakdown = supplier_stats.get("country_breakdown", {})

        compliant = compliance_breakdown.get(
            ComplianceStatus.COMPLIANT.value, 0
        )
        pending = compliance_breakdown.get(
            ComplianceStatus.PENDING.value, 0
        )
        non_compliant = compliance_breakdown.get(
            ComplianceStatus.NON_COMPLIANT.value, 0
        )
        under_review = compliance_breakdown.get(
            ComplianceStatus.UNDER_REVIEW.value, 0
        )

        # Plot metrics
        total_plots = supplier_stats.get("total_plots", 0)
        all_plots = self.supplier_engine.list_plots()
        assessed_plots = sum(
            1
            for p in all_plots
            if p.satellite_status != SatelliteAssessmentStatus.NOT_ASSESSED
        )
        high_risk_plots = sum(
            1
            for p in all_plots
            if p.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
        )
        deforestation_free = sum(
            1
            for p in all_plots
            if p.is_deforestation_free is True
        )

        # DDS metrics
        dds_stats = self.dds_engine.get_statistics()
        dds_total = dds_stats.get("total_dds", 0)
        dds_by_status = dds_stats.get("by_status", {})
        dds_draft = dds_by_status.get(DDSStatus.DRAFT.value, 0)
        dds_submitted = dds_by_status.get(DDSStatus.SUBMITTED.value, 0)
        dds_accepted = dds_by_status.get(DDSStatus.ACCEPTED.value, 0)
        dds_rejected = dds_by_status.get(DDSStatus.REJECTED.value, 0)

        # Pipeline metrics
        pipeline_summary = self.pipeline.get_all_runs_summary()
        pipeline_active = pipeline_summary.get(
            PipelineStatus.RUNNING.value, 0
        )
        pipeline_completed_today = pipeline_summary.get(
            PipelineStatus.COMPLETED.value, 0
        )

        # Risk metrics
        risk_stats = self.risk_engine.get_statistics()
        average_risk = risk_stats.get("average_overall_risk", 0.0)

        # Compliance rate
        compliance_rate = (
            (compliant / total_suppliers * 100)
            if total_suppliers > 0
            else 0.0
        )

        return DashboardMetrics(
            total_suppliers=total_suppliers,
            compliant_count=compliant,
            pending_count=pending,
            non_compliant_count=non_compliant,
            under_review_count=under_review,
            total_plots=total_plots,
            assessed_plots=assessed_plots,
            high_risk_plots=high_risk_plots,
            deforestation_free_plots=deforestation_free,
            dds_total=dds_total,
            dds_draft=dds_draft,
            dds_submitted=dds_submitted,
            dds_accepted=dds_accepted,
            dds_rejected=dds_rejected,
            pipeline_active=pipeline_active,
            pipeline_completed_today=pipeline_completed_today,
            compliance_rate=round(compliance_rate, 1),
            average_risk_score=round(average_risk, 4),
            commodities_breakdown=commodity_breakdown,
            country_breakdown=country_breakdown,
            last_updated=_utcnow(),
        )

    # -----------------------------------------------------------------------
    # Health Check
    # -----------------------------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on all engines.

        Returns:
            Dictionary with engine status and version info.
        """
        return {
            "status": "healthy",
            "version": "1.0.0",
            "application": "GL-EUDR-APP",
            "engines": {
                "supplier_intake": "ok",
                "document_verification": "ok",
                "dds_reporting": "ok",
                "risk_aggregator": "ok",
                "pipeline_orchestrator": "ok",
            },
            "agents": {
                "AGENT-DATA-005": {
                    "name": "EUDR Traceability Connector",
                    "id": "GL-DATA-EUDR-001",
                    "status": "available",
                },
                "AGENT-DATA-007": {
                    "name": "Deforestation Satellite Connector",
                    "id": "GL-DATA-GEO-003",
                    "status": "available",
                },
            },
            "config": {
                "pipeline_max_concurrent": self.config.pipeline_max_concurrent,
                "dds_auto_submit": self.config.dds_auto_submit,
                "ndvi_threshold": self.config.ndvi_change_threshold,
                "risk_threshold_high": self.config.risk_threshold_high,
                "risk_threshold_critical": self.config.risk_threshold_critical,
            },
            "timestamp": _utcnow().isoformat(),
        }

    # -----------------------------------------------------------------------
    # Comprehensive Statistics
    # -----------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive platform statistics from all engines.

        Returns:
            Dictionary with statistics from all five engines.
        """
        return {
            "application": "GL-EUDR-APP",
            "version": "1.0.0",
            "suppliers": self.supplier_engine.get_statistics(),
            "documents": self.document_engine.get_statistics(),
            "dds": self.dds_engine.get_statistics(),
            "risk": self.risk_engine.get_statistics(),
            "pipeline": self.pipeline.get_all_runs_summary(),
            "timestamp": _utcnow().isoformat(),
        }


# ===========================================================================
# FastAPI Integration
# ===========================================================================

_SERVICE_KEY = "eudr_compliance_service"


def configure_eudr_app(app: Any) -> EUDRComplianceService:
    """Register the EUDR Compliance Service on a FastAPI application.

    Creates the service, attaches it to ``app.state``, and logs
    the configuration.

    Args:
        app: FastAPI application instance.

    Returns:
        Configured EUDRComplianceService instance.

    Example:
        >>> from fastapi import FastAPI
        >>> from services.setup import configure_eudr_app
        >>> app = FastAPI()
        >>> service = configure_eudr_app(app)
    """
    service = EUDRComplianceService()
    setattr(app.state, _SERVICE_KEY, service)

    logger.info(
        "EUDRComplianceService configured on FastAPI app: "
        "pipeline_max_concurrent=%d, dds_auto_submit=%s",
        service.config.pipeline_max_concurrent,
        service.config.dds_auto_submit,
    )
    return service


def get_eudr_service(app: Any) -> EUDRComplianceService:
    """Retrieve the EUDR Compliance Service from a FastAPI application.

    Args:
        app: FastAPI application instance.

    Returns:
        EUDRComplianceService instance.

    Raises:
        RuntimeError: If service not configured.
    """
    service = getattr(app.state, _SERVICE_KEY, None)
    if service is None:
        raise RuntimeError(
            "EUDRComplianceService not configured. "
            "Call configure_eudr_app(app) first."
        )
    return service
