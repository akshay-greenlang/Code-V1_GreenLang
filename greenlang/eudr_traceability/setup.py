# -*- coding: utf-8 -*-
"""
EUDR Traceability Service Facade - AGENT-DATA-004: EUDR Traceability

Provides the main service class and FastAPI integration functions:
- EUDRTraceabilityService: Composes all 7 engines into a single facade
- configure_eudr_traceability(app): Register service on FastAPI app
- get_eudr_traceability(app): Retrieve service from app state
- get_router(): Return FastAPI router for mounting

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-004 EUDR Traceability Connector
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EUDRTraceabilityService:
    """Facade composing all EUDR Traceability engines.

    Provides a single entry point for all EUDR traceability operations,
    delegating to the appropriate engine for each operation type.

    Attributes:
        config: EUDRTraceabilityConfig instance.
        plot_registry: PlotRegistryEngine instance.
        chain_of_custody: ChainOfCustodyEngine instance.
        due_diligence: DueDiligenceEngine instance.
        risk_assessment: RiskAssessmentEngine instance.
        commodity_classifier: CommodityClassifier instance.
        compliance_verifier: ComplianceVerifier instance.
        eu_system: EUSystemConnector instance.
    """

    def __init__(self, config: Optional[Any] = None):
        """Initialize the EUDR Traceability Service with all engines.

        Args:
            config: EUDRTraceabilityConfig instance. If None, loads from env.
        """
        if config is None:
            from greenlang.eudr_traceability.config import get_config
            config = get_config()

        self.config = config

        # Initialize engines
        from greenlang.eudr_traceability.plot_registry import PlotRegistryEngine
        from greenlang.eudr_traceability.chain_of_custody import ChainOfCustodyEngine
        from greenlang.eudr_traceability.due_diligence import DueDiligenceEngine
        from greenlang.eudr_traceability.risk_assessment import RiskAssessmentEngine
        from greenlang.eudr_traceability.commodity_classifier import CommodityClassifier
        from greenlang.eudr_traceability.compliance_verifier import ComplianceVerifier
        from greenlang.eudr_traceability.eu_system_connector import EUSystemConnector

        self.plot_registry = PlotRegistryEngine(config=config)
        self.risk_assessment = RiskAssessmentEngine(config=config)
        self.chain_of_custody = ChainOfCustodyEngine(
            config=config,
            plot_registry=self.plot_registry,
        )
        self.due_diligence = DueDiligenceEngine(
            config=config,
            plot_registry=self.plot_registry,
            chain_of_custody=self.chain_of_custody,
            risk_engine=self.risk_assessment,
        )
        self.commodity_classifier = CommodityClassifier(config=config)
        self.compliance_verifier = ComplianceVerifier(
            config=config,
            plot_registry=self.plot_registry,
            chain_of_custody=self.chain_of_custody,
        )
        self.eu_system = EUSystemConnector(config=config)

        logger.info(
            "EUDRTraceabilityService initialized with all 7 engines"
        )

    # =========================================================================
    # Plot Registry Delegation
    # =========================================================================

    def register_plot(self, request: Any) -> Any:
        """Register a production plot. Delegates to PlotRegistryEngine.

        Args:
            request: RegisterPlotRequest instance.

        Returns:
            PlotRecord instance.
        """
        return self.plot_registry.register_plot(request)

    def get_plot(self, plot_id: str) -> Any:
        """Get plot by ID. Delegates to PlotRegistryEngine.

        Args:
            plot_id: Plot identifier.

        Returns:
            PlotRecord or None.
        """
        return self.plot_registry.get_plot(plot_id)

    def list_plots(self, **kwargs) -> List[Any]:
        """List plots with filters. Delegates to PlotRegistryEngine.

        Returns:
            List of PlotRecord instances.
        """
        return self.plot_registry.list_plots(**kwargs)

    # =========================================================================
    # Chain of Custody Delegation
    # =========================================================================

    def record_transfer(self, request: Any) -> Any:
        """Record custody transfer. Delegates to ChainOfCustodyEngine.

        Args:
            request: RecordTransferRequest instance.

        Returns:
            CustodyTransfer instance.
        """
        return self.chain_of_custody.record_transfer(request)

    def trace_to_origin(self, batch_id: str) -> List[Any]:
        """Trace batch to origin plots. Delegates to ChainOfCustodyEngine.

        Args:
            batch_id: Batch identifier.

        Returns:
            List of PlotRecord instances.
        """
        return self.chain_of_custody.trace_to_origin(batch_id)

    # =========================================================================
    # Due Diligence Delegation
    # =========================================================================

    def generate_dds(self, request: Any) -> Any:
        """Generate DDS. Delegates to DueDiligenceEngine.

        Args:
            request: GenerateDDSRequest instance.

        Returns:
            DueDiligenceStatement instance.
        """
        return self.due_diligence.generate_dds(request)

    def submit_dds(self, dds_id: str) -> Any:
        """Submit DDS to EU system. Orchestrates DueDiligence + EUSystem.

        Args:
            dds_id: DDS identifier.

        Returns:
            Updated DueDiligenceStatement.
        """
        dds = self.due_diligence.submit_dds(dds_id)
        if dds:
            eu_data = self.due_diligence.export_for_eu_system(dds_id)
            if eu_data:
                self.eu_system.prepare_submission(dds_id, eu_data)
        return dds

    # =========================================================================
    # Risk Assessment Delegation
    # =========================================================================

    def assess_risk(self, request: Any) -> Any:
        """Assess risk. Delegates to RiskAssessmentEngine.

        Args:
            request: AssessRiskRequest instance.

        Returns:
            RiskScore instance.
        """
        return self.risk_assessment.assess_risk(request)

    # =========================================================================
    # Commodity Classification Delegation
    # =========================================================================

    def classify_commodity(self, request: Any) -> Any:
        """Classify commodity. Delegates to CommodityClassifier.

        Args:
            request: ClassifyCommodityRequest instance.

        Returns:
            CommodityClassification instance.
        """
        return self.commodity_classifier.classify(request)

    # =========================================================================
    # Compliance Verification Delegation
    # =========================================================================

    def verify_compliance(
        self,
        target_type: str,
        target_id: str,
    ) -> List[Any]:
        """Verify compliance. Delegates to ComplianceVerifier.

        Args:
            target_type: Type of target (plot, dds, operator).
            target_id: Target identifier.

        Returns:
            List of ComplianceCheckResult instances.
        """
        return self.compliance_verifier.verify_compliance(
            target_type, target_id
        )

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive service statistics.

        Returns:
            Dictionary with statistics from all engines.
        """
        return {
            "agent_id": "GL-DATA-EUDR-001",
            "agent_name": "EUDR Traceability Connector Agent",
            "version": "1.0.0",
            "plots": self.plot_registry.get_statistics()
            if hasattr(self.plot_registry, 'get_statistics') else {},
            "chain_of_custody": self.chain_of_custody.get_statistics()
            if hasattr(self.chain_of_custody, 'get_statistics') else {},
            "due_diligence": self.due_diligence.get_dds_statistics()
            if hasattr(self.due_diligence, 'get_dds_statistics') else {},
            "risk_assessment": {},
            "eu_submissions": self.eu_system.get_statistics()
            if hasattr(self.eu_system, 'get_statistics') else {},
            "compliance": self.compliance_verifier.get_compliance_summary()
            if hasattr(self.compliance_verifier, 'get_compliance_summary')
            else {},
        }


# =============================================================================
# FastAPI Integration
# =============================================================================

_SERVICE_KEY = "eudr_traceability_service"


def configure_eudr_traceability(app: Any) -> EUDRTraceabilityService:
    """Register the EUDR Traceability Service on a FastAPI application.

    Creates the service, attaches it to app.state, and includes the
    API router.

    Args:
        app: FastAPI application instance.

    Returns:
        Configured EUDRTraceabilityService instance.
    """
    service = EUDRTraceabilityService()
    app.state.eudr_traceability_service = service

    # Include router
    from greenlang.eudr_traceability.api.router import router
    app.include_router(router)

    logger.info("EUDR Traceability Service configured on FastAPI app")
    return service


def get_eudr_traceability(app: Any) -> EUDRTraceabilityService:
    """Retrieve the EUDR Traceability Service from a FastAPI application.

    Args:
        app: FastAPI application instance.

    Returns:
        EUDRTraceabilityService instance.

    Raises:
        RuntimeError: If service not configured.
    """
    service = getattr(app.state, _SERVICE_KEY, None)
    if service is None:
        raise RuntimeError(
            "EUDR Traceability Service not configured. "
            "Call configure_eudr_traceability(app) first."
        )
    return service


def get_router():
    """Return the FastAPI router for the EUDR Traceability Service.

    Returns:
        FastAPI APIRouter instance.
    """
    from greenlang.eudr_traceability.api.router import router
    return router
