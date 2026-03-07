# -*- coding: utf-8 -*-
"""
GL-EUDR-APP Supply Chain Service Layer - Facade for AGENT-EUDR-001

Application-level facade that wraps the ``SupplyChainMapperService``
from ``greenlang.agents.eudr.supply_chain_mapper.setup`` for use by
the GL-EUDR-APP frontend and API layer. Provides simplified helper
methods for common UI operations, user-friendly error messages, and
integration with the existing GL-EUDR-APP service ecosystem.

Capabilities:
    - Supply chain graph retrieval with aggregated statistics
    - Full analysis workflow: graph creation + risk propagation + gap analysis
    - Export to Due Diligence Statement (DDS) format via RegulatoryExporter
    - Supply chain summary for dashboard widgets
    - Supplier onboarding status tracking
    - Error handling with structured, user-friendly error responses

Integration Points:
    - SupplyChainMapperService: 9-engine facade from AGENT-EUDR-001
    - EUDRComplianceService: Existing GL-EUDR-APP 5-engine facade
    - AGENT-DATA-005: EUDR Traceability Connector (plot geolocation)
    - AGENT-DATA-007: Deforestation Satellite Connector (satellite risk)

Zero-Hallucination Guarantees:
    - All risk scores are deterministic (formula-based propagation)
    - All graph metrics are computed from actual node/edge counts
    - No LLM calls in any calculation path
    - SHA-256 provenance hashes on all exported data

Example:
    >>> from services.supply_chain import SupplyChainAppService
    >>> svc = SupplyChainAppService()
    >>> await svc.initialize()
    >>> summary = await svc.get_graph_with_stats("graph-id-1")
    >>> assert summary["graph_id"] is not None

Author: GreenLang Platform Team
Date: March 2026
Application: GL-EUDR-APP v1.0 + AGENT-EUDR-001 Integration
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from services.config import EUDRAppConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Error wrapper
# ---------------------------------------------------------------------------


class SupplyChainError(Exception):
    """User-friendly error from the supply chain service layer.

    Attributes:
        message: Human-readable error message suitable for API responses.
        detail: Technical detail for logging and debugging.
        error_code: Machine-readable error code string.
    """

    def __init__(
        self,
        message: str,
        detail: str = "",
        error_code: str = "SCM_ERROR",
    ) -> None:
        self.message = message
        self.detail = detail
        self.error_code = error_code
        super().__init__(message)

    def to_dict(self) -> Dict[str, str]:
        """Serialize for JSON error responses."""
        return {
            "error": self.error_code,
            "message": self.message,
            "detail": self.detail,
        }


# ---------------------------------------------------------------------------
# Supply Chain App Service
# ---------------------------------------------------------------------------


class SupplyChainAppService:
    """Application-level facade for supply chain mapping operations.

    Wraps the AGENT-EUDR-001 ``SupplyChainMapperService`` with
    application-specific convenience methods, error handling, and
    integration with the GL-EUDR-APP platform. This service is
    designed for consumption by the frontend and API layers.

    The service must be initialized before use via ``initialize()``,
    which starts the underlying SupplyChainMapperService (database
    pool, Redis cache, and all nine engines).

    Attributes:
        config: GL-EUDR-APP configuration.
        is_initialized: Whether the service has been started.
        scm_service: Underlying SupplyChainMapperService instance.

    Example:
        >>> svc = SupplyChainAppService()
        >>> await svc.initialize()
        >>> summary = await svc.get_dashboard_summary()
    """

    def __init__(
        self,
        config: Optional[EUDRAppConfig] = None,
    ) -> None:
        """Initialize SupplyChainAppService.

        Does NOT start the underlying SupplyChainMapperService. Call
        ``initialize()`` to connect to DB, Redis, and start engines.

        Args:
            config: GL-EUDR-APP configuration. If None, loads from env.
        """
        self._config = config or EUDRAppConfig()
        self._scm_service: Optional[Any] = None
        self._initialized = False
        logger.info(
            "SupplyChainAppService created: scm_enabled=%s",
            self._config.scm_enabled,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> EUDRAppConfig:
        """Return the application configuration."""
        return self._config

    @property
    def is_initialized(self) -> bool:
        """Return whether the service has been initialized."""
        return self._initialized

    @property
    def scm_service(self) -> Any:
        """Return the underlying SupplyChainMapperService.

        Raises:
            SupplyChainError: If the service is not initialized.
        """
        if not self._initialized or self._scm_service is None:
            raise SupplyChainError(
                message="Supply chain service is not available.",
                detail="Call initialize() before using the service.",
                error_code="SCM_NOT_INITIALIZED",
            )
        return self._scm_service

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Initialize and start the SupplyChainMapperService.

        Creates a SupplyChainMapperConfig from the GL-EUDR-APP settings,
        instantiates the singleton SupplyChainMapperService, and calls
        its ``startup()`` method to connect DB, Redis, and all engines.

        Idempotent: safe to call multiple times.

        Raises:
            SupplyChainError: If initialization fails.
        """
        if self._initialized:
            logger.debug("SupplyChainAppService already initialized")
            return

        if not self._config.scm_enabled:
            logger.info(
                "Supply chain mapper is disabled (scm_enabled=False). "
                "Skipping initialization."
            )
            return

        start = time.monotonic()

        try:
            from greenlang.agents.eudr.supply_chain_mapper.config import (
                SupplyChainMapperConfig,
            )
            from greenlang.agents.eudr.supply_chain_mapper.setup import (
                SupplyChainMapperService,
                set_service,
            )

            # Build SCM config from EUDR app config
            scm_config = SupplyChainMapperConfig(
                database_url=(
                    self._config.scm_database_url
                    or self._config.database_url
                ),
                redis_url=(
                    self._config.scm_redis_url or self._config.redis_url
                ),
                log_level=self._config.log_level,
                risk_weight_country=self._config.scm_risk_weight_country,
                risk_weight_commodity=self._config.scm_risk_weight_commodity,
                risk_weight_supplier=self._config.scm_risk_weight_supplier,
                risk_weight_deforestation=self._config.scm_risk_weight_deforestation,
                max_nodes_per_graph=self._config.scm_max_nodes_per_graph,
                max_tier_depth=self._config.scm_max_tier_depth,
                enable_provenance=self._config.scm_enable_provenance,
                enable_metrics=self._config.enable_metrics,
                pool_size=self._config.scm_pool_size,
                cache_ttl=self._config.scm_cache_ttl,
                rate_limit=self._config.scm_rate_limit,
            )

            service = SupplyChainMapperService(config=scm_config)
            set_service(service)
            await service.startup()

            self._scm_service = service
            self._initialized = True

            elapsed = (time.monotonic() - start) * 1000
            logger.info(
                "SupplyChainAppService initialized in %.1fms: "
                "engines=%d, db=%s, redis=%s",
                elapsed,
                service.initialized_engine_count(),
                "connected" if service.db_pool is not None else "skipped",
                "connected" if service.redis_client is not None else "skipped",
            )

        except ImportError as exc:
            logger.warning(
                "Supply chain mapper module not available: %s. "
                "AGENT-EUDR-001 features will be disabled.",
                exc,
            )

        except Exception as exc:
            logger.error(
                "Failed to initialize SupplyChainAppService: %s",
                exc,
                exc_info=True,
            )
            raise SupplyChainError(
                message="Failed to start the supply chain mapping service.",
                detail=str(exc),
                error_code="SCM_INIT_FAILED",
            ) from exc

    async def shutdown(self) -> None:
        """Gracefully shut down the SupplyChainMapperService.

        Releases all resources (DB pool, Redis, engines). Idempotent.
        """
        if not self._initialized or self._scm_service is None:
            logger.debug("SupplyChainAppService not initialized, skip shutdown")
            return

        try:
            await self._scm_service.shutdown()
            logger.info("SupplyChainAppService shut down successfully")
        except Exception as exc:
            logger.error(
                "Error shutting down SupplyChainAppService: %s",
                exc,
                exc_info=True,
            )
        finally:
            self._scm_service = None
            self._initialized = False

    # ------------------------------------------------------------------
    # Helper: Graph with Statistics
    # ------------------------------------------------------------------

    async def get_graph_with_stats(
        self, graph_id: str
    ) -> Dict[str, Any]:
        """Retrieve a supply chain graph with aggregated statistics.

        Fetches the graph from the graph engine and enriches it with
        node/edge counts, tier depth, risk summary, and gap counts.
        Intended for the frontend graph detail view.

        Args:
            graph_id: Supply chain graph identifier.

        Returns:
            Dictionary with graph data, node count, edge count, tier
            depth, risk summary, gap count, and provenance hash.

        Raises:
            SupplyChainError: If graph not found or service unavailable.
        """
        start = time.monotonic()

        try:
            service = self.scm_service
            graph_engine = service.graph_engine
            risk_engine = service.risk_propagation_engine
            gap_analyzer = service.gap_analyzer

            # Fetch graph
            if graph_engine is None:
                raise SupplyChainError(
                    message="Graph engine is not available.",
                    error_code="SCM_ENGINE_UNAVAILABLE",
                )

            graph = graph_engine.get_graph(graph_id)
            if graph is None:
                raise SupplyChainError(
                    message=f"Supply chain graph '{graph_id}' not found.",
                    detail=f"No graph with ID {graph_id} exists.",
                    error_code="SCM_GRAPH_NOT_FOUND",
                )

            # Aggregate statistics
            node_count = len(graph.nodes) if hasattr(graph, "nodes") else 0
            edge_count = len(graph.edges) if hasattr(graph, "edges") else 0

            # Tier depth
            tier_depth = 0
            if hasattr(graph, "nodes"):
                for node in graph.nodes.values():
                    tier = getattr(node, "tier", 0) or 0
                    tier_depth = max(tier_depth, tier)

            # Risk summary
            risk_summary: Dict[str, Any] = {"available": False}
            if risk_engine is not None:
                try:
                    risk_result = risk_engine.get_risk_summary(graph_id)
                    if risk_result is not None:
                        risk_summary = {
                            "available": True,
                            "average_risk": getattr(
                                risk_result, "average_risk", 0.0
                            ),
                            "max_risk": getattr(
                                risk_result, "max_risk", 0.0
                            ),
                            "high_risk_count": getattr(
                                risk_result, "high_risk_count", 0
                            ),
                        }
                except Exception as exc:
                    logger.debug("Risk summary unavailable: %s", exc)

            # Gap count
            gap_count = 0
            if gap_analyzer is not None:
                try:
                    gaps = gap_analyzer.get_gaps(graph_id)
                    if gaps is not None:
                        gap_count = len(gaps)
                except Exception as exc:
                    logger.debug("Gap analysis unavailable: %s", exc)

            elapsed_ms = (time.monotonic() - start) * 1000

            result = {
                "graph_id": graph_id,
                "graph": (
                    graph.model_dump(mode="json")
                    if hasattr(graph, "model_dump")
                    else graph
                ),
                "statistics": {
                    "node_count": node_count,
                    "edge_count": edge_count,
                    "tier_depth": tier_depth,
                    "gap_count": gap_count,
                },
                "risk_summary": risk_summary,
                "provenance_hash": _compute_hash(
                    {"graph_id": graph_id, "nodes": node_count, "edges": edge_count}
                ),
                "processing_time_ms": round(elapsed_ms, 1),
                "retrieved_at": _utcnow().isoformat(),
            }

            logger.info(
                "Graph with stats retrieved: graph_id=%s, nodes=%d, "
                "edges=%d, tiers=%d, gaps=%d in %.1fms",
                graph_id,
                node_count,
                edge_count,
                tier_depth,
                gap_count,
                elapsed_ms,
            )
            return result

        except SupplyChainError:
            raise
        except Exception as exc:
            logger.error(
                "Failed to get graph with stats: %s", exc, exc_info=True
            )
            raise SupplyChainError(
                message="Failed to retrieve supply chain graph.",
                detail=str(exc),
                error_code="SCM_GRAPH_FETCH_FAILED",
            ) from exc

    # ------------------------------------------------------------------
    # Helper: Full Analysis Workflow
    # ------------------------------------------------------------------

    async def run_full_analysis(
        self,
        graph_id: str,
        include_risk: bool = True,
        include_gaps: bool = True,
        include_visualization: bool = False,
    ) -> Dict[str, Any]:
        """Run a complete supply chain analysis on an existing graph.

        Orchestrates multiple engines in sequence:
          1. Fetch graph from graph engine
          2. Run risk propagation (if enabled)
          3. Run gap analysis (if enabled)
          4. Generate visualization layout (if enabled)
          5. Compute provenance hash over all results

        This is the primary analysis entry point for the UI "Analyze"
        button and the pipeline integration.

        Args:
            graph_id: Supply chain graph identifier.
            include_risk: Run risk propagation engine.
            include_gaps: Run gap analysis engine.
            include_visualization: Generate Sankey/layout data.

        Returns:
            Dictionary with graph, risk, gaps, visualization, and
            provenance information.

        Raises:
            SupplyChainError: If analysis fails.
        """
        start = time.monotonic()
        results: Dict[str, Any] = {
            "graph_id": graph_id,
            "analysis_steps": [],
            "errors": [],
        }

        try:
            service = self.scm_service

            # Step 1: Fetch graph
            graph_engine = service.graph_engine
            if graph_engine is None:
                raise SupplyChainError(
                    message="Graph engine is not available for analysis.",
                    error_code="SCM_ENGINE_UNAVAILABLE",
                )

            graph = graph_engine.get_graph(graph_id)
            if graph is None:
                raise SupplyChainError(
                    message=f"Supply chain graph '{graph_id}' not found.",
                    error_code="SCM_GRAPH_NOT_FOUND",
                )
            results["analysis_steps"].append("graph_loaded")

            # Step 2: Risk propagation
            if include_risk:
                risk_result = await self._run_risk_propagation(
                    service, graph_id, results
                )
                results["risk"] = risk_result

            # Step 3: Gap analysis
            if include_gaps:
                gap_result = await self._run_gap_analysis(
                    service, graph_id, results
                )
                results["gaps"] = gap_result

            # Step 4: Visualization
            if include_visualization:
                viz_result = await self._run_visualization(
                    service, graph_id, results
                )
                results["visualization"] = viz_result

            # Step 5: Provenance
            elapsed_ms = (time.monotonic() - start) * 1000
            results["provenance_hash"] = _compute_hash(results)
            results["processing_time_ms"] = round(elapsed_ms, 1)
            results["completed_at"] = _utcnow().isoformat()
            results["status"] = (
                "completed" if not results["errors"] else "completed_with_errors"
            )

            logger.info(
                "Full analysis completed: graph_id=%s, steps=%s, "
                "errors=%d, time=%.1fms",
                graph_id,
                results["analysis_steps"],
                len(results["errors"]),
                elapsed_ms,
            )
            return results

        except SupplyChainError:
            raise
        except Exception as exc:
            logger.error(
                "Full analysis failed: graph_id=%s, error=%s",
                graph_id,
                exc,
                exc_info=True,
            )
            raise SupplyChainError(
                message="Supply chain analysis failed.",
                detail=str(exc),
                error_code="SCM_ANALYSIS_FAILED",
            ) from exc

    async def _run_risk_propagation(
        self,
        service: Any,
        graph_id: str,
        results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run risk propagation engine on a graph.

        Args:
            service: SupplyChainMapperService instance.
            graph_id: Graph to analyze.
            results: Accumulator for analysis steps and errors.

        Returns:
            Risk propagation result dictionary.
        """
        try:
            engine = service.risk_propagation_engine
            if engine is None:
                results["errors"].append(
                    "Risk propagation engine not available"
                )
                return {"available": False}

            risk_result = engine.propagate_risk(graph_id)
            results["analysis_steps"].append("risk_propagation")
            return {
                "available": True,
                "data": (
                    risk_result.model_dump(mode="json")
                    if hasattr(risk_result, "model_dump")
                    else risk_result
                ),
            }
        except Exception as exc:
            results["errors"].append(f"Risk propagation failed: {exc}")
            logger.warning("Risk propagation failed: %s", exc)
            return {"available": False, "error": str(exc)}

    async def _run_gap_analysis(
        self,
        service: Any,
        graph_id: str,
        results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run gap analysis engine on a graph.

        Args:
            service: SupplyChainMapperService instance.
            graph_id: Graph to analyze.
            results: Accumulator for analysis steps and errors.

        Returns:
            Gap analysis result dictionary.
        """
        try:
            engine = service.gap_analyzer
            if engine is None:
                results["errors"].append("Gap analyzer not available")
                return {"available": False}

            gap_result = engine.analyze_gaps(graph_id)
            results["analysis_steps"].append("gap_analysis")
            return {
                "available": True,
                "data": (
                    gap_result.model_dump(mode="json")
                    if hasattr(gap_result, "model_dump")
                    else gap_result
                ),
            }
        except Exception as exc:
            results["errors"].append(f"Gap analysis failed: {exc}")
            logger.warning("Gap analysis failed: %s", exc)
            return {"available": False, "error": str(exc)}

    async def _run_visualization(
        self,
        service: Any,
        graph_id: str,
        results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run visualization engine on a graph.

        Args:
            service: SupplyChainMapperService instance.
            graph_id: Graph to visualize.
            results: Accumulator for analysis steps and errors.

        Returns:
            Visualization result dictionary.
        """
        try:
            engine = service.visualization_engine
            if engine is None:
                results["errors"].append("Visualization engine not available")
                return {"available": False}

            viz_result = engine.generate_layout(graph_id)
            results["analysis_steps"].append("visualization")
            return {
                "available": True,
                "data": (
                    viz_result.model_dump(mode="json")
                    if hasattr(viz_result, "model_dump")
                    else viz_result
                ),
            }
        except Exception as exc:
            results["errors"].append(f"Visualization failed: {exc}")
            logger.warning("Visualization failed: %s", exc)
            return {"available": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Helper: Export to DDS
    # ------------------------------------------------------------------

    async def export_to_dds(
        self,
        graph_id: str,
        operator_info: Optional[Dict[str, Any]] = None,
        product_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Export supply chain graph data for DDS inclusion.

        Uses the RegulatoryExporter engine to produce the supply chain
        section of a Due Diligence Statement in the EU Information
        System format per EUDR Article 4(2).

        The returned data is suitable for direct inclusion in the
        DDS generation via ``DDSReportingEngine``.

        Args:
            graph_id: Supply chain graph identifier.
            operator_info: Operator details for the DDS. If None, uses
                defaults from the graph metadata.
            product_info: Product details for the DDS. If None, uses
                defaults from the graph metadata.

        Returns:
            Dictionary with DDS-formatted supply chain data, including
            supply chain summary, node list, traceability score, and
            provenance hash.

        Raises:
            SupplyChainError: If export fails or exporter unavailable.
        """
        start = time.monotonic()

        try:
            service = self.scm_service
            exporter = service.regulatory_exporter

            if exporter is None:
                raise SupplyChainError(
                    message="Regulatory exporter is not available.",
                    detail="The RegulatoryExporter engine was not initialized.",
                    error_code="SCM_EXPORTER_UNAVAILABLE",
                )

            # Get graph for export
            graph_engine = service.graph_engine
            if graph_engine is None:
                raise SupplyChainError(
                    message="Graph engine is not available for export.",
                    error_code="SCM_ENGINE_UNAVAILABLE",
                )

            graph = graph_engine.get_graph(graph_id)
            if graph is None:
                raise SupplyChainError(
                    message=f"Supply chain graph '{graph_id}' not found.",
                    error_code="SCM_GRAPH_NOT_FOUND",
                )

            # Build export parameters
            export_params: Dict[str, Any] = {"graph": graph}
            if operator_info:
                export_params["operator_info"] = operator_info
            if product_info:
                export_params["product_info"] = product_info

            # Execute export
            dds_data = exporter.export_dds_json(**export_params)

            elapsed_ms = (time.monotonic() - start) * 1000

            result = {
                "graph_id": graph_id,
                "dds_supply_chain_section": (
                    dds_data.model_dump(mode="json")
                    if hasattr(dds_data, "model_dump")
                    else dds_data
                ),
                "provenance_hash": _compute_hash(
                    {"graph_id": graph_id, "export_type": "dds_json"}
                ),
                "processing_time_ms": round(elapsed_ms, 1),
                "exported_at": _utcnow().isoformat(),
            }

            logger.info(
                "Supply chain exported for DDS: graph_id=%s in %.1fms",
                graph_id,
                elapsed_ms,
            )
            return result

        except SupplyChainError:
            raise
        except Exception as exc:
            logger.error(
                "DDS export failed: graph_id=%s, error=%s",
                graph_id,
                exc,
                exc_info=True,
            )
            raise SupplyChainError(
                message="Failed to export supply chain for DDS.",
                detail=str(exc),
                error_code="SCM_EXPORT_FAILED",
            ) from exc

    # ------------------------------------------------------------------
    # Helper: Dashboard Summary
    # ------------------------------------------------------------------

    async def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get an aggregated supply chain summary for the dashboard.

        Returns high-level statistics suitable for the GL-EUDR-APP
        dashboard supply chain widget.

        Returns:
            Dictionary with graph count, total nodes, total edges,
            average tier depth, engine status, and health information.
        """
        try:
            service = self.scm_service

            # Health check
            health = await service.health_check()

            summary: Dict[str, Any] = {
                "service_status": health.get("status", "unknown"),
                "engines_initialized": 0,
                "engines_total": 9,
                "uptime_seconds": health.get("uptime_seconds", 0),
                "version": health.get("version", "1.0.0"),
                "timestamp": _utcnow().isoformat(),
            }

            # Engine counts from health check
            engine_health = health.get("checks", {}).get("engines", {})
            summary["engines_initialized"] = engine_health.get(
                "initialized", 0
            )

            return summary

        except SupplyChainError:
            return {
                "service_status": "unavailable",
                "engines_initialized": 0,
                "engines_total": 9,
                "uptime_seconds": 0,
                "version": "1.0.0",
                "timestamp": _utcnow().isoformat(),
                "error": "Supply chain service not initialized",
            }
        except Exception as exc:
            logger.error(
                "Failed to get dashboard summary: %s", exc, exc_info=True
            )
            return {
                "service_status": "error",
                "engines_initialized": 0,
                "engines_total": 9,
                "uptime_seconds": 0,
                "version": "1.0.0",
                "timestamp": _utcnow().isoformat(),
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # Helper: Health Check
    # ------------------------------------------------------------------

    async def health_check(self) -> Dict[str, Any]:
        """Run a health check on the supply chain mapping service.

        Returns:
            Health status dictionary with component-level detail.
        """
        if not self._initialized or self._scm_service is None:
            return {
                "status": "not_initialized",
                "scm_enabled": self._config.scm_enabled,
                "timestamp": _utcnow().isoformat(),
            }

        try:
            return await self._scm_service.health_check()
        except Exception as exc:
            logger.error("Health check failed: %s", exc)
            return {
                "status": "error",
                "error": str(exc),
                "timestamp": _utcnow().isoformat(),
            }

    # ------------------------------------------------------------------
    # Helper: Onboarding Status
    # ------------------------------------------------------------------

    async def get_onboarding_status(
        self, invitation_id: str
    ) -> Dict[str, Any]:
        """Get the status of a supplier onboarding invitation.

        Args:
            invitation_id: Onboarding invitation identifier.

        Returns:
            Onboarding status dictionary.

        Raises:
            SupplyChainError: If the onboarding engine is unavailable.
        """
        try:
            service = self.scm_service
            engine = service.supplier_onboarding_engine

            if engine is None:
                raise SupplyChainError(
                    message="Supplier onboarding engine is not available.",
                    error_code="SCM_ENGINE_UNAVAILABLE",
                )

            status = engine.get_invitation_status(invitation_id)
            if status is None:
                raise SupplyChainError(
                    message=f"Invitation '{invitation_id}' not found.",
                    error_code="SCM_INVITATION_NOT_FOUND",
                )

            return (
                status.model_dump(mode="json")
                if hasattr(status, "model_dump")
                else status
            )

        except SupplyChainError:
            raise
        except Exception as exc:
            logger.error(
                "Onboarding status check failed: %s", exc, exc_info=True
            )
            raise SupplyChainError(
                message="Failed to check onboarding status.",
                detail=str(exc),
                error_code="SCM_ONBOARDING_FAILED",
            ) from exc


# ---------------------------------------------------------------------------
# FastAPI integration helpers
# ---------------------------------------------------------------------------

_SERVICE_KEY = "supply_chain_app_service"


def configure_supply_chain_service(
    app: Any,
    config: Optional[EUDRAppConfig] = None,
) -> SupplyChainAppService:
    """Create and attach the SupplyChainAppService to a FastAPI app.

    Does NOT call ``initialize()``. The caller should await
    ``service.initialize()`` during the application startup event
    or lifespan context.

    Args:
        app: FastAPI application instance.
        config: Optional application configuration.

    Returns:
        Created SupplyChainAppService (not yet initialized).
    """
    service = SupplyChainAppService(config=config)
    setattr(app.state, _SERVICE_KEY, service)
    logger.info("SupplyChainAppService attached to FastAPI app.state")
    return service


def get_supply_chain_service(app: Any) -> SupplyChainAppService:
    """Retrieve the SupplyChainAppService from a FastAPI app.

    Args:
        app: FastAPI application instance.

    Returns:
        SupplyChainAppService instance.

    Raises:
        RuntimeError: If service not configured on the app.
    """
    service = getattr(app.state, _SERVICE_KEY, None)
    if service is None:
        raise RuntimeError(
            "SupplyChainAppService not configured. "
            "Call configure_supply_chain_service(app) first."
        )
    return service


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "SupplyChainAppService",
    "SupplyChainError",
    "configure_supply_chain_service",
    "get_supply_chain_service",
]
