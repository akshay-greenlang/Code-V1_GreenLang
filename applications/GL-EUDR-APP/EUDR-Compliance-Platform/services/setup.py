# -*- coding: utf-8 -*-
"""
GL-EUDR-APP Service Facade - EU Deforestation Regulation Compliance Platform

Provides the ``EUDRComplianceService`` facade class that composes all five
core engines plus the AGENT-EUDR-001 supply chain mapper into a single
entry point for the EUDR compliance platform:

- SupplierIntakeEngine:         Supplier CRUD, bulk import, ERP normalization
- DocumentVerificationEngine:   Document upload, OCR, verification, gap analysis
- DDSReportingEngine:           DDS generation, validation, submission, lifecycle
- RiskAggregator:               Multi-source risk scoring, alerts, heatmaps
- PipelineOrchestrator:         5-stage compliance pipeline execution
- SupplyChainAppService:        Supply chain mapping (AGENT-EUDR-001, 9 engines)

Also provides:
- ``configure_eudr_app(app)``:  FastAPI integration
- ``get_eudr_service(app)``:    Retrieve service from app state
- ``register_routes(app)``:     Register all API routers (core + SCM)
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
Application: GL-EUDR-APP v1.0 + AGENT-EUDR-001 Integration
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

    Composes all five core engines plus the AGENT-EUDR-001 supply chain
    mapper into a single service entry point. Engines are cross-wired
    so the DDS engine can access supplier, document, and supply chain
    data, and the risk aggregator can query both supplier and document
    engines.

    Attributes:
        config: Application configuration.
        pipeline: PipelineOrchestrator for 5-stage compliance runs.
        supplier_engine: SupplierIntakeEngine for supplier CRUD.
        document_engine: DocumentVerificationEngine for document management.
        dds_engine: DDSReportingEngine for DDS lifecycle.
        risk_engine: RiskAggregator for risk scoring.
        supply_chain_service: SupplyChainAppService for AGENT-EUDR-001.

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

        Creates and cross-wires all five core engines. If no config is
        provided, loads defaults from environment variables.

        The AGENT-EUDR-001 SupplyChainAppService is created but NOT
        initialized (async startup). Call ``initialize_supply_chain()``
        during the app startup event to connect DB, Redis, and engines.

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

        # 6. Supply Chain Mapper service (AGENT-EUDR-001)
        #    Created here but async initialization must be called separately.
        self.supply_chain_service: Optional[Any] = None
        if self.config.scm_enabled:
            try:
                from services.supply_chain import SupplyChainAppService

                self.supply_chain_service = SupplyChainAppService(
                    config=self.config
                )
                logger.info(
                    "SupplyChainAppService created (pending async init)"
                )
            except ImportError as exc:
                logger.warning(
                    "SupplyChainAppService not available: %s", exc
                )
                self.supply_chain_service = None

        engine_count = 5 + (1 if self.supply_chain_service else 0)
        logger.info(
            "EUDRComplianceService initialized with %d components: "
            "SupplierIntake, DocumentVerification, RiskAggregator, "
            "DDSReporting, PipelineOrchestrator%s",
            engine_count,
            ", SupplyChainMapper" if self.supply_chain_service else "",
        )

    # -----------------------------------------------------------------------
    # Supply Chain Lifecycle
    # -----------------------------------------------------------------------

    async def initialize_supply_chain(self) -> bool:
        """Initialize the AGENT-EUDR-001 SupplyChainAppService.

        Must be called during the async startup phase (e.g., FastAPI
        lifespan or on_event("startup")). Connects DB, Redis, and
        starts all 9 supply chain engines.

        After initialization, the supply chain service is wired into
        the DDS engine for automatic DDS supply chain section generation.

        Returns:
            True if initialization succeeded, False otherwise.
        """
        if self.supply_chain_service is None:
            logger.info(
                "Supply chain service not configured; skipping init"
            )
            return False

        try:
            await self.supply_chain_service.initialize()

            # Wire supply chain service into DDS engine
            self.dds_engine.set_supply_chain_service(
                self.supply_chain_service
            )

            logger.info(
                "Supply chain service initialized and wired to DDS engine"
            )
            return True

        except Exception as exc:
            logger.error(
                "Failed to initialize supply chain service: %s",
                exc,
                exc_info=True,
            )
            return False

    async def shutdown_supply_chain(self) -> None:
        """Gracefully shut down the supply chain service.

        Should be called during the async shutdown phase.
        """
        if self.supply_chain_service is not None:
            try:
                await self.supply_chain_service.shutdown()
                logger.info("Supply chain service shut down")
            except Exception as exc:
                logger.error(
                    "Error shutting down supply chain service: %s", exc
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
        # Supply chain mapper status
        scm_status = "disabled"
        if self.supply_chain_service is not None:
            if self.supply_chain_service.is_initialized:
                scm_status = "healthy"
            else:
                scm_status = "not_initialized"

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
                "supply_chain_mapper": scm_status,
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
                "AGENT-EUDR-001": {
                    "name": "Supply Chain Mapping Master",
                    "id": "GL-EUDR-SCM-001",
                    "status": scm_status,
                },
            },
            "config": {
                "pipeline_max_concurrent": self.config.pipeline_max_concurrent,
                "dds_auto_submit": self.config.dds_auto_submit,
                "ndvi_threshold": self.config.ndvi_change_threshold,
                "risk_threshold_high": self.config.risk_threshold_high,
                "risk_threshold_critical": self.config.risk_threshold_critical,
                "scm_enabled": self.config.scm_enabled,
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

    Creates the service, attaches it to ``app.state``, registers all
    API routers (core + AGENT-EUDR-001 supply chain mapper), and logs
    the configuration.

    NOTE: The supply chain mapper requires async initialization. Call
    ``await service.initialize_supply_chain()`` during the FastAPI
    startup event or lifespan context.

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

    # Register all routers (core platform + supply chain mapper)
    try:
        from services.api.routers import register_all_routers

        route_summary = register_all_routers(
            app,
            scm_prefix=service.config.scm_route_prefix,
            scm_enabled=service.config.scm_enabled,
        )
        logger.info(
            "Routes registered: %s", route_summary
        )
    except ImportError:
        logger.warning(
            "Router registration module not available; "
            "routes must be registered manually"
        )

    logger.info(
        "EUDRComplianceService configured on FastAPI app: "
        "pipeline_max_concurrent=%d, dds_auto_submit=%s, "
        "scm_enabled=%s",
        service.config.pipeline_max_concurrent,
        service.config.dds_auto_submit,
        service.config.scm_enabled,
    )
    return service


async def startup_eudr_app(app: Any) -> None:
    """Async startup hook for the GL-EUDR-APP.

    Initializes the AGENT-EUDR-001 SupplyChainMapperService, which
    requires async operations for DB pool and Redis connections.

    Should be called from FastAPI's startup event or lifespan context:

        @app.on_event("startup")
        async def on_startup():
            await startup_eudr_app(app)

    Args:
        app: FastAPI application instance with EUDRComplianceService
            already configured via ``configure_eudr_app(app)``.
    """
    service = get_eudr_service(app)
    scm_ok = await service.initialize_supply_chain()
    if scm_ok:
        logger.info(
            "GL-EUDR-APP async startup complete: "
            "supply chain mapper initialized"
        )
    else:
        logger.info(
            "GL-EUDR-APP async startup complete: "
            "supply chain mapper not available"
        )


async def shutdown_eudr_app(app: Any) -> None:
    """Async shutdown hook for the GL-EUDR-APP.

    Gracefully shuts down the AGENT-EUDR-001 SupplyChainMapperService.

    Should be called from FastAPI's shutdown event or lifespan context:

        @app.on_event("shutdown")
        async def on_shutdown():
            await shutdown_eudr_app(app)

    Args:
        app: FastAPI application instance.
    """
    try:
        service = get_eudr_service(app)
        await service.shutdown_supply_chain()
        logger.info("GL-EUDR-APP async shutdown complete")
    except RuntimeError:
        pass  # Service was never configured


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
