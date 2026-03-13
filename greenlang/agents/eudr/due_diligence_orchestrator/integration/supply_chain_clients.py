# -*- coding: utf-8 -*-
"""
Supply Chain Integration Clients - AGENT-EUDR-026

Typed wrapper clients for all 15 Phase 1 EUDR agents (EUDR-001 through
EUDR-015) responsible for supply chain traceability, geospatial
verification, satellite monitoring, chain of custody, and document
evidence collection.

Each client class encapsulates the agent-specific input preparation,
output validation, and error handling for its upstream agent, while
delegating the actual HTTP call to the shared AgentClient.

Clients:
    - SupplyChainMappingClient     (EUDR-001)
    - GeolocationVerificationClient (EUDR-002)
    - SatelliteMonitoringClient     (EUDR-003)
    - ForestCoverAnalysisClient     (EUDR-004)
    - LandUseChangeClient           (EUDR-005)
    - PlotBoundaryClient            (EUDR-006)
    - GPSValidationClient           (EUDR-007)
    - MultiTierSupplierClient       (EUDR-008)
    - ChainOfCustodyClient          (EUDR-009)
    - SegregationVerifierClient     (EUDR-010)
    - MassBalanceClient             (EUDR-011)
    - DocumentAuthenticationClient  (EUDR-012)
    - BlockchainIntegrationClient   (EUDR-013)
    - QRCodeGeneratorClient         (EUDR-014)
    - MobileDataCollectorClient     (EUDR-015)

Zero-Hallucination:
    These clients only transform and relay agent-produced data; no
    numeric computation or LLM generation occurs here.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-026 Due Diligence Orchestrator (GL-EUDR-DDO-026)
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from greenlang.agents.eudr.due_diligence_orchestrator.config import (
    DueDiligenceOrchestratorConfig,
    get_config,
)
from greenlang.agents.eudr.due_diligence_orchestrator.integration.agent_client import (
    AgentCallResult,
    AgentClient,
)
from greenlang.agents.eudr.due_diligence_orchestrator.models import (
    AGENT_NAMES,
    _new_uuid,
    _utcnow,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base Phase 1 Client
# ---------------------------------------------------------------------------


class _BasePhase1Client:
    """Base class for all Phase 1 supply chain agent clients.

    Provides shared infrastructure for input preparation, output
    extraction, and health checking.

    Attributes:
        _agent_id: EUDR agent identifier (e.g., "EUDR-001").
        _agent_name: Human-readable agent name.
        _client: Shared AgentClient instance.
        _config: Orchestrator configuration.
    """

    def __init__(
        self,
        agent_id: str,
        client: Optional[AgentClient] = None,
        config: Optional[DueDiligenceOrchestratorConfig] = None,
    ) -> None:
        """Initialize base Phase 1 client.

        Args:
            agent_id: EUDR agent identifier.
            client: Optional shared AgentClient instance.
            config: Optional configuration override.
        """
        self._agent_id = agent_id
        self._agent_name = AGENT_NAMES.get(agent_id, agent_id)
        self._config = config or get_config()
        self._client = client or AgentClient(self._config)

    def call(
        self,
        input_data: Dict[str, Any],
        timeout_s: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> AgentCallResult:
        """Call the agent with the given input data.

        Args:
            input_data: Agent-specific input payload.
            timeout_s: Optional timeout override in seconds.
            headers: Optional additional HTTP headers.

        Returns:
            AgentCallResult with response data or error details.
        """
        logger.info(
            f"Calling {self._agent_name} ({self._agent_id}) "
            f"with {len(input_data)} input fields"
        )
        return self._client.call_agent(
            self._agent_id,
            input_data,
            timeout_s=timeout_s,
            headers=headers,
        )

    def is_healthy(self) -> bool:
        """Check if the agent endpoint is healthy.

        Returns:
            True if the agent responds to health check.
        """
        return self._client.check_agent_health(self._agent_id)

    @property
    def agent_id(self) -> str:
        """Return the EUDR agent identifier."""
        return self._agent_id

    @property
    def agent_name(self) -> str:
        """Return the human-readable agent name."""
        return self._agent_name


# ---------------------------------------------------------------------------
# EUDR-001: Supply Chain Mapping Master
# ---------------------------------------------------------------------------


class SupplyChainMappingClient(_BasePhase1Client):
    """Client for EUDR-001 Supply Chain Mapping Master.

    Invokes the supply chain mapping agent to establish the complete
    supply chain topology including all operators, suppliers, and
    production origins.

    This is always the first agent in any workflow DAG.

    Example:
        >>> client = SupplyChainMappingClient()
        >>> result = client.map_supply_chain(
        ...     operator_id="OP-001",
        ...     commodity="cocoa",
        ...     countries=["GH", "CI"]
        ... )
    """

    def __init__(
        self,
        client: Optional[AgentClient] = None,
        config: Optional[DueDiligenceOrchestratorConfig] = None,
    ) -> None:
        """Initialize EUDR-001 client."""
        super().__init__("EUDR-001", client, config)

    def map_supply_chain(
        self,
        operator_id: str,
        commodity: str,
        countries: List[str],
        hs_codes: Optional[List[str]] = None,
        depth: int = 4,
    ) -> AgentCallResult:
        """Map the supply chain for an operator and commodity.

        Args:
            operator_id: Operator identifier.
            commodity: EUDR commodity type.
            countries: List of ISO 3166-1 alpha-2 country codes.
            hs_codes: Optional HS tariff codes.
            depth: Maximum supply chain depth to map.

        Returns:
            AgentCallResult with supply chain topology.
        """
        input_data: Dict[str, Any] = {
            "operator_id": operator_id,
            "commodity": commodity,
            "countries": countries,
            "depth": depth,
        }
        if hs_codes:
            input_data["hs_codes"] = hs_codes

        return self.call(input_data)

    def extract_operator_info(
        self, result: AgentCallResult
    ) -> Dict[str, Any]:
        """Extract operator identification fields from result.

        Args:
            result: Successful AgentCallResult from EUDR-001.

        Returns:
            Dictionary with operator_name, postal_address, email, eori.
        """
        if not result.success:
            return {}

        data = result.output_data
        return {
            "operator_name": data.get("operator_name", ""),
            "postal_address": data.get("postal_address", ""),
            "email_address": data.get("email_address", ""),
            "eori_number": data.get("eori_number", ""),
        }

    def extract_product_info(
        self, result: AgentCallResult
    ) -> Dict[str, Any]:
        """Extract product description fields from result.

        Args:
            result: Successful AgentCallResult from EUDR-001.

        Returns:
            Dictionary with product_description, trade_name, etc.
        """
        if not result.success:
            return {}

        data = result.output_data
        return {
            "product_description": data.get("product_description", ""),
            "trade_name": data.get("trade_name", ""),
            "hs_heading": data.get("hs_heading", ""),
            "commodity_type": data.get("commodity_type", ""),
        }


# ---------------------------------------------------------------------------
# EUDR-002: Geolocation Verification
# ---------------------------------------------------------------------------


class GeolocationVerificationClient(_BasePhase1Client):
    """Client for EUDR-002 Geolocation Verification Agent.

    Verifies GPS coordinates of production plots against reference
    data and validates coordinate accuracy per EUDR requirements.

    Example:
        >>> client = GeolocationVerificationClient()
        >>> result = client.verify_coordinates(
        ...     coordinates=[{"lat": -3.45, "lon": -62.21}],
        ...     country="BR"
        ... )
    """

    def __init__(
        self,
        client: Optional[AgentClient] = None,
        config: Optional[DueDiligenceOrchestratorConfig] = None,
    ) -> None:
        """Initialize EUDR-002 client."""
        super().__init__("EUDR-002", client, config)

    def verify_coordinates(
        self,
        coordinates: List[Dict[str, Any]],
        country: str,
        accuracy_threshold_m: float = 10.0,
    ) -> AgentCallResult:
        """Verify production plot coordinates.

        Args:
            coordinates: List of coordinate dicts with lat/lon.
            country: ISO 3166-1 alpha-2 country code.
            accuracy_threshold_m: Required accuracy in meters.

        Returns:
            AgentCallResult with verification results.
        """
        return self.call({
            "coordinates": coordinates,
            "country": country,
            "accuracy_threshold_m": accuracy_threshold_m,
        })

    def extract_verified_plots(
        self, result: AgentCallResult
    ) -> List[Dict[str, Any]]:
        """Extract verified plot coordinates from result.

        Args:
            result: Successful AgentCallResult from EUDR-002.

        Returns:
            List of verified plot coordinate dictionaries.
        """
        if not result.success:
            return []
        return result.output_data.get("verified_plots", [])


# ---------------------------------------------------------------------------
# EUDR-003: Satellite Monitoring
# ---------------------------------------------------------------------------


class SatelliteMonitoringClient(_BasePhase1Client):
    """Client for EUDR-003 Satellite Monitoring Agent.

    Initiates satellite imagery analysis for production areas to detect
    land use changes and deforestation indicators.

    Example:
        >>> client = SatelliteMonitoringClient()
        >>> result = client.analyze_area(
        ...     plot_boundaries=[...],
        ...     start_date="2020-12-31",
        ...     end_date="2025-06-01"
        ... )
    """

    def __init__(
        self,
        client: Optional[AgentClient] = None,
        config: Optional[DueDiligenceOrchestratorConfig] = None,
    ) -> None:
        """Initialize EUDR-003 client."""
        super().__init__("EUDR-003", client, config)

    def analyze_area(
        self,
        plot_boundaries: List[Dict[str, Any]],
        start_date: str,
        end_date: str,
        resolution_m: int = 10,
    ) -> AgentCallResult:
        """Analyze satellite imagery for production areas.

        Args:
            plot_boundaries: List of plot boundary GeoJSON objects.
            start_date: Analysis start date (ISO 8601).
            end_date: Analysis end date (ISO 8601).
            resolution_m: Spatial resolution in meters.

        Returns:
            AgentCallResult with satellite analysis data.
        """
        return self.call({
            "plot_boundaries": plot_boundaries,
            "start_date": start_date,
            "end_date": end_date,
            "resolution_m": resolution_m,
        })

    def extract_monitoring_data(
        self, result: AgentCallResult
    ) -> Dict[str, Any]:
        """Extract satellite monitoring summary from result.

        Args:
            result: Successful AgentCallResult from EUDR-003.

        Returns:
            Monitoring data summary dictionary.
        """
        if not result.success:
            return {}
        return {
            "coverage_percentage": result.output_data.get(
                "coverage_percentage", 0
            ),
            "imagery_count": result.output_data.get("imagery_count", 0),
            "analysis_period": result.output_data.get(
                "analysis_period", {}
            ),
        }


# ---------------------------------------------------------------------------
# EUDR-004: Forest Cover Analysis
# ---------------------------------------------------------------------------


class ForestCoverAnalysisClient(_BasePhase1Client):
    """Client for EUDR-004 Forest Cover Analysis Agent.

    Analyzes forest cover changes in production areas using satellite
    imagery and reference forest maps.

    Example:
        >>> client = ForestCoverAnalysisClient()
        >>> result = client.analyze_forest_cover(
        ...     plot_ids=["PLOT-001", "PLOT-002"],
        ...     cutoff_date="2020-12-31"
        ... )
    """

    def __init__(
        self,
        client: Optional[AgentClient] = None,
        config: Optional[DueDiligenceOrchestratorConfig] = None,
    ) -> None:
        """Initialize EUDR-004 client."""
        super().__init__("EUDR-004", client, config)

    def analyze_forest_cover(
        self,
        plot_ids: List[str],
        cutoff_date: str = "2020-12-31",
    ) -> AgentCallResult:
        """Analyze forest cover for production plots.

        Args:
            plot_ids: List of production plot identifiers.
            cutoff_date: EUDR cutoff date (default Dec 31, 2020).

        Returns:
            AgentCallResult with forest cover analysis.
        """
        return self.call({
            "plot_ids": plot_ids,
            "cutoff_date": cutoff_date,
        })


# ---------------------------------------------------------------------------
# EUDR-005: Land Use Change Detector
# ---------------------------------------------------------------------------


class LandUseChangeClient(_BasePhase1Client):
    """Client for EUDR-005 Land Use Change Detector Agent.

    Detects land use changes since the EUDR cutoff date (Dec 31, 2020)
    to determine deforestation-free status.

    Example:
        >>> client = LandUseChangeClient()
        >>> result = client.detect_changes(
        ...     plot_ids=["PLOT-001"],
        ...     cutoff_date="2020-12-31"
        ... )
    """

    def __init__(
        self,
        client: Optional[AgentClient] = None,
        config: Optional[DueDiligenceOrchestratorConfig] = None,
    ) -> None:
        """Initialize EUDR-005 client."""
        super().__init__("EUDR-005", client, config)

    def detect_changes(
        self,
        plot_ids: List[str],
        cutoff_date: str = "2020-12-31",
    ) -> AgentCallResult:
        """Detect land use changes since cutoff date.

        Args:
            plot_ids: List of production plot identifiers.
            cutoff_date: EUDR cutoff date.

        Returns:
            AgentCallResult with land use change analysis.
        """
        return self.call({
            "plot_ids": plot_ids,
            "cutoff_date": cutoff_date,
        })


# ---------------------------------------------------------------------------
# EUDR-006: Plot Boundary Manager
# ---------------------------------------------------------------------------


class PlotBoundaryClient(_BasePhase1Client):
    """Client for EUDR-006 Plot Boundary Manager Agent.

    Manages GeoJSON polygon boundaries for production plots exceeding
    4 hectares, as required by EUDR Article 9.

    Example:
        >>> client = PlotBoundaryClient()
        >>> result = client.generate_boundaries(
        ...     coordinates=[{"lat": -3.45, "lon": -62.21}],
        ...     area_threshold_ha=4.0
        ... )
    """

    def __init__(
        self,
        client: Optional[AgentClient] = None,
        config: Optional[DueDiligenceOrchestratorConfig] = None,
    ) -> None:
        """Initialize EUDR-006 client."""
        super().__init__("EUDR-006", client, config)

    def generate_boundaries(
        self,
        coordinates: List[Dict[str, Any]],
        area_threshold_ha: float = 4.0,
    ) -> AgentCallResult:
        """Generate polygon boundaries for large production plots.

        Args:
            coordinates: Plot center coordinates.
            area_threshold_ha: Minimum area for polygon requirement.

        Returns:
            AgentCallResult with GeoJSON boundaries.
        """
        return self.call({
            "coordinates": coordinates,
            "area_threshold_ha": area_threshold_ha,
        })


# ---------------------------------------------------------------------------
# EUDR-007: GPS Coordinate Validator
# ---------------------------------------------------------------------------


class GPSValidationClient(_BasePhase1Client):
    """Client for EUDR-007 GPS Coordinate Validator Agent.

    Validates GPS coordinate accuracy, consistency, and plausibility
    for EUDR compliance requirements.

    Example:
        >>> client = GPSValidationClient()
        >>> result = client.validate_gps(
        ...     coordinates=[{"lat": -3.45, "lon": -62.21, "accuracy": 5.2}]
        ... )
    """

    def __init__(
        self,
        client: Optional[AgentClient] = None,
        config: Optional[DueDiligenceOrchestratorConfig] = None,
    ) -> None:
        """Initialize EUDR-007 client."""
        super().__init__("EUDR-007", client, config)

    def validate_gps(
        self,
        coordinates: List[Dict[str, Any]],
        accuracy_required_m: float = 10.0,
    ) -> AgentCallResult:
        """Validate GPS coordinates for accuracy and plausibility.

        Args:
            coordinates: List of coordinate dicts with lat/lon/accuracy.
            accuracy_required_m: Required accuracy threshold in meters.

        Returns:
            AgentCallResult with validation results.
        """
        return self.call({
            "coordinates": coordinates,
            "accuracy_required_m": accuracy_required_m,
        })


# ---------------------------------------------------------------------------
# EUDR-008: Multi-Tier Supplier Tracker
# ---------------------------------------------------------------------------


class MultiTierSupplierClient(_BasePhase1Client):
    """Client for EUDR-008 Multi-Tier Supplier Tracker Agent.

    Maps and tracks suppliers across multiple tiers of the supply chain
    to ensure complete traceability to the point of production.

    Example:
        >>> client = MultiTierSupplierClient()
        >>> result = client.track_suppliers(
        ...     operator_id="OP-001",
        ...     commodity="cocoa",
        ...     max_depth=4
        ... )
    """

    def __init__(
        self,
        client: Optional[AgentClient] = None,
        config: Optional[DueDiligenceOrchestratorConfig] = None,
    ) -> None:
        """Initialize EUDR-008 client."""
        super().__init__("EUDR-008", client, config)

    def track_suppliers(
        self,
        operator_id: str,
        commodity: str,
        max_depth: int = 4,
        supplier_ids: Optional[List[str]] = None,
    ) -> AgentCallResult:
        """Map multi-tier supplier network.

        Args:
            operator_id: Root operator identifier.
            commodity: EUDR commodity type.
            max_depth: Maximum supplier tiers to map.
            supplier_ids: Optional known supplier IDs to seed.

        Returns:
            AgentCallResult with supplier tier mapping.
        """
        input_data: Dict[str, Any] = {
            "operator_id": operator_id,
            "commodity": commodity,
            "max_depth": max_depth,
        }
        if supplier_ids:
            input_data["supplier_ids"] = supplier_ids

        return self.call(input_data)


# ---------------------------------------------------------------------------
# EUDR-009: Chain of Custody
# ---------------------------------------------------------------------------


class ChainOfCustodyClient(_BasePhase1Client):
    """Client for EUDR-009 Chain of Custody Agent.

    Verifies and documents the chain of custody from production
    origin through all intermediary handling points to the operator.

    Example:
        >>> client = ChainOfCustodyClient()
        >>> result = client.verify_custody_chain(
        ...     shipment_id="SHIP-001",
        ...     supply_chain_nodes=[...]
        ... )
    """

    def __init__(
        self,
        client: Optional[AgentClient] = None,
        config: Optional[DueDiligenceOrchestratorConfig] = None,
    ) -> None:
        """Initialize EUDR-009 client."""
        super().__init__("EUDR-009", client, config)

    def verify_custody_chain(
        self,
        shipment_id: str,
        supply_chain_nodes: List[Dict[str, Any]],
    ) -> AgentCallResult:
        """Verify chain of custody for a shipment.

        Args:
            shipment_id: Shipment identifier.
            supply_chain_nodes: Ordered list of custody transfer points.

        Returns:
            AgentCallResult with custody verification status.
        """
        return self.call({
            "shipment_id": shipment_id,
            "supply_chain_nodes": supply_chain_nodes,
        })


# ---------------------------------------------------------------------------
# EUDR-010: Segregation Verifier
# ---------------------------------------------------------------------------


class SegregationVerifierClient(_BasePhase1Client):
    """Client for EUDR-010 Segregation Verifier Agent.

    Verifies that EUDR-compliant products are properly segregated
    from non-compliant products throughout the supply chain.

    Example:
        >>> client = SegregationVerifierClient()
        >>> result = client.verify_segregation(
        ...     facility_id="FAC-001",
        ...     product_batches=[...]
        ... )
    """

    def __init__(
        self,
        client: Optional[AgentClient] = None,
        config: Optional[DueDiligenceOrchestratorConfig] = None,
    ) -> None:
        """Initialize EUDR-010 client."""
        super().__init__("EUDR-010", client, config)

    def verify_segregation(
        self,
        facility_id: str,
        product_batches: List[Dict[str, Any]],
    ) -> AgentCallResult:
        """Verify product segregation at a facility.

        Args:
            facility_id: Facility identifier.
            product_batches: List of product batch records.

        Returns:
            AgentCallResult with segregation verification status.
        """
        return self.call({
            "facility_id": facility_id,
            "product_batches": product_batches,
        })


# ---------------------------------------------------------------------------
# EUDR-011: Mass Balance Calculator
# ---------------------------------------------------------------------------


class MassBalanceClient(_BasePhase1Client):
    """Client for EUDR-011 Mass Balance Calculator Agent.

    Computes and verifies mass balance accounting for commodity
    volumes across the supply chain.

    Example:
        >>> client = MassBalanceClient()
        >>> result = client.calculate_mass_balance(
        ...     inputs=[{"source": "FARM-01", "kg": 5000}],
        ...     outputs=[{"destination": "WAREHOUSE-01", "kg": 4900}]
        ... )
    """

    def __init__(
        self,
        client: Optional[AgentClient] = None,
        config: Optional[DueDiligenceOrchestratorConfig] = None,
    ) -> None:
        """Initialize EUDR-011 client."""
        super().__init__("EUDR-011", client, config)

    def calculate_mass_balance(
        self,
        inputs: List[Dict[str, Any]],
        outputs: List[Dict[str, Any]],
        tolerance_pct: float = 2.0,
    ) -> AgentCallResult:
        """Calculate mass balance for commodity flows.

        Args:
            inputs: List of input commodity records.
            outputs: List of output commodity records.
            tolerance_pct: Acceptable variance percentage.

        Returns:
            AgentCallResult with mass balance calculation.
        """
        return self.call({
            "inputs": inputs,
            "outputs": outputs,
            "tolerance_pct": tolerance_pct,
        })


# ---------------------------------------------------------------------------
# EUDR-012: Document Authentication
# ---------------------------------------------------------------------------


class DocumentAuthenticationClient(_BasePhase1Client):
    """Client for EUDR-012 Document Authentication Agent.

    Authenticates supporting documents (certificates, permits,
    invoices) used as evidence in the due diligence process.

    Example:
        >>> client = DocumentAuthenticationClient()
        >>> result = client.authenticate_documents(
        ...     documents=[{"type": "phyto_cert", "id": "DOC-001"}]
        ... )
    """

    def __init__(
        self,
        client: Optional[AgentClient] = None,
        config: Optional[DueDiligenceOrchestratorConfig] = None,
    ) -> None:
        """Initialize EUDR-012 client."""
        super().__init__("EUDR-012", client, config)

    def authenticate_documents(
        self,
        documents: List[Dict[str, Any]],
    ) -> AgentCallResult:
        """Authenticate supporting documents.

        Args:
            documents: List of document records with type and ID.

        Returns:
            AgentCallResult with authentication results.
        """
        return self.call({"documents": documents})


# ---------------------------------------------------------------------------
# EUDR-013: Blockchain Integration
# ---------------------------------------------------------------------------


class BlockchainIntegrationClient(_BasePhase1Client):
    """Client for EUDR-013 Blockchain Integration Agent.

    Records supply chain events on a distributed ledger for
    tamper-evident traceability.

    Example:
        >>> client = BlockchainIntegrationClient()
        >>> result = client.record_event(
        ...     event_type="custody_transfer",
        ...     event_data={...}
        ... )
    """

    def __init__(
        self,
        client: Optional[AgentClient] = None,
        config: Optional[DueDiligenceOrchestratorConfig] = None,
    ) -> None:
        """Initialize EUDR-013 client."""
        super().__init__("EUDR-013", client, config)

    def record_event(
        self,
        event_type: str,
        event_data: Dict[str, Any],
    ) -> AgentCallResult:
        """Record a supply chain event on the blockchain.

        Args:
            event_type: Type of supply chain event.
            event_data: Event payload data.

        Returns:
            AgentCallResult with blockchain transaction reference.
        """
        return self.call({
            "event_type": event_type,
            "event_data": event_data,
        })


# ---------------------------------------------------------------------------
# EUDR-014: QR Code Generator
# ---------------------------------------------------------------------------


class QRCodeGeneratorClient(_BasePhase1Client):
    """Client for EUDR-014 QR Code Generator Agent.

    Generates traceability QR codes for EUDR-regulated products
    linking to their due diligence statement.

    Example:
        >>> client = QRCodeGeneratorClient()
        >>> result = client.generate_qr_code(
        ...     product_id="PROD-001",
        ...     dds_reference="DDS-2025-0001"
        ... )
    """

    def __init__(
        self,
        client: Optional[AgentClient] = None,
        config: Optional[DueDiligenceOrchestratorConfig] = None,
    ) -> None:
        """Initialize EUDR-014 client."""
        super().__init__("EUDR-014", client, config)

    def generate_qr_code(
        self,
        product_id: str,
        dds_reference: str,
        format: str = "svg",
    ) -> AgentCallResult:
        """Generate a traceability QR code.

        Args:
            product_id: Product identifier.
            dds_reference: DDS reference number.
            format: Output format (svg, png, pdf).

        Returns:
            AgentCallResult with QR code data or URL.
        """
        return self.call({
            "product_id": product_id,
            "dds_reference": dds_reference,
            "format": format,
        })


# ---------------------------------------------------------------------------
# EUDR-015: Mobile Data Collector
# ---------------------------------------------------------------------------


class MobileDataCollectorClient(_BasePhase1Client):
    """Client for EUDR-015 Mobile Data Collector Agent.

    Collects field-level data from mobile devices including GPS
    coordinates, photos, and survey responses.

    Example:
        >>> client = MobileDataCollectorClient()
        >>> result = client.collect_field_data(
        ...     survey_id="SURVEY-001",
        ...     field_data=[...]
        ... )
    """

    def __init__(
        self,
        client: Optional[AgentClient] = None,
        config: Optional[DueDiligenceOrchestratorConfig] = None,
    ) -> None:
        """Initialize EUDR-015 client."""
        super().__init__("EUDR-015", client, config)

    def collect_field_data(
        self,
        survey_id: str,
        field_data: List[Dict[str, Any]],
    ) -> AgentCallResult:
        """Submit collected field data from mobile devices.

        Args:
            survey_id: Survey/form identifier.
            field_data: Collected field data records.

        Returns:
            AgentCallResult with processing status.
        """
        return self.call({
            "survey_id": survey_id,
            "field_data": field_data,
        })


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Map of agent_id to client class for dynamic instantiation.
PHASE1_CLIENT_REGISTRY: Dict[str, type] = {
    "EUDR-001": SupplyChainMappingClient,
    "EUDR-002": GeolocationVerificationClient,
    "EUDR-003": SatelliteMonitoringClient,
    "EUDR-004": ForestCoverAnalysisClient,
    "EUDR-005": LandUseChangeClient,
    "EUDR-006": PlotBoundaryClient,
    "EUDR-007": GPSValidationClient,
    "EUDR-008": MultiTierSupplierClient,
    "EUDR-009": ChainOfCustodyClient,
    "EUDR-010": SegregationVerifierClient,
    "EUDR-011": MassBalanceClient,
    "EUDR-012": DocumentAuthenticationClient,
    "EUDR-013": BlockchainIntegrationClient,
    "EUDR-014": QRCodeGeneratorClient,
    "EUDR-015": MobileDataCollectorClient,
}


def get_phase1_client(
    agent_id: str,
    shared_client: Optional[AgentClient] = None,
    config: Optional[DueDiligenceOrchestratorConfig] = None,
) -> _BasePhase1Client:
    """Factory function to get a Phase 1 client by agent ID.

    Args:
        agent_id: EUDR agent identifier (EUDR-001 through EUDR-015).
        shared_client: Optional shared AgentClient instance.
        config: Optional configuration override.

    Returns:
        Initialized Phase 1 client instance.

    Raises:
        ValueError: If agent_id is not a Phase 1 agent.
    """
    client_cls = PHASE1_CLIENT_REGISTRY.get(agent_id)
    if client_cls is None:
        raise ValueError(
            f"Agent {agent_id} is not a Phase 1 agent. "
            f"Valid: {sorted(PHASE1_CLIENT_REGISTRY.keys())}"
        )
    return client_cls(client=shared_client, config=config)


def get_all_phase1_clients(
    shared_client: Optional[AgentClient] = None,
    config: Optional[DueDiligenceOrchestratorConfig] = None,
) -> Dict[str, _BasePhase1Client]:
    """Create all 15 Phase 1 clients with a shared AgentClient.

    Args:
        shared_client: Optional shared AgentClient instance.
        config: Optional configuration override.

    Returns:
        Dictionary mapping agent_id to client instance.
    """
    _config = config or get_config()
    _client = shared_client or AgentClient(_config)
    return {
        agent_id: get_phase1_client(agent_id, _client, _config)
        for agent_id in sorted(PHASE1_CLIENT_REGISTRY.keys())
    }
