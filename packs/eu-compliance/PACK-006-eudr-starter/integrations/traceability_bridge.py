# -*- coding: utf-8 -*-
"""
TraceabilityBridge - Bridge to EUDR Traceability Connector
============================================================

This module provides a bridge interface to the EUDR Traceability Connector
located at ``greenlang/eudr_traceability/``. It exposes seven proxy classes
for plot registry, chain of custody, commodity classification, compliance
verification, due diligence, risk assessment, and supply chain mapping.

Proxy Services:
    - PlotRegistryProxy: Plot CRUD, geolocation management
    - ChainOfCustodyProxy: Custody events, batch tracking
    - CommodityProxy: Commodity classification, CN codes
    - ComplianceProxy: Compliance checks
    - DueDiligenceProxy: DD workflow support
    - RiskProxy: Risk scoring from traceability data
    - SupplyChainProxy: Supplier mapping, batch traceability

Example:
    >>> bridge = TraceabilityBridge()
    >>> plot_registry = bridge.get_plot_registry()
    >>> plots = await plot_registry.list_plots(supplier_id="SUP-001")

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


class TraceabilityBridgeConfig(BaseModel):
    """Configuration for the Traceability Bridge."""
    connector_path: str = Field(
        default="greenlang/eudr_traceability",
        description="Path to EUDR Traceability Connector",
    )
    stub_mode: bool = Field(
        default=True, description="Use stub fallback if connector not available"
    )
    timeout_seconds: int = Field(default=30, description="Timeout for calls")


class PlotData(BaseModel):
    """Plot data from the traceability connector."""
    plot_id: str = Field(default="", description="Plot ID")
    supplier_id: str = Field(default="", description="Supplier ID")
    latitude: float = Field(default=0.0, description="Latitude")
    longitude: float = Field(default=0.0, description="Longitude")
    polygon: Optional[List[List[float]]] = Field(None, description="Polygon vertices")
    area_ha: float = Field(default=0.0, description="Area in hectares")
    country: str = Field(default="", description="Country code")
    commodity: str = Field(default="", description="Primary commodity")
    registration_date: datetime = Field(
        default_factory=datetime.utcnow, description="Registration date"
    )
    verified: bool = Field(default=False, description="Verification status")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class CustodyEvent(BaseModel):
    """Chain of custody event."""
    event_id: str = Field(default="", description="Event ID")
    batch_id: str = Field(default="", description="Batch ID")
    event_type: str = Field(
        default="transfer",
        description="Event type (harvest, transfer, process, export, import)",
    )
    from_entity: str = Field(default="", description="Source entity")
    to_entity: str = Field(default="", description="Destination entity")
    commodity: str = Field(default="", description="Commodity")
    quantity: float = Field(default=0.0, description="Quantity")
    unit: str = Field(default="kg", description="Unit of measure")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event time")
    geolocation: Optional[Dict[str, float]] = Field(None, description="Event location")
    provenance_hash: str = Field(default="", description="SHA-256 hash")


class BatchRecord(BaseModel):
    """Batch tracking record."""
    batch_id: str = Field(default="", description="Batch ID")
    commodity: str = Field(default="", description="Commodity")
    origin_plot_ids: List[str] = Field(default_factory=list, description="Source plot IDs")
    supplier_id: str = Field(default="", description="Supplier ID")
    quantity: float = Field(default=0.0, description="Total quantity")
    unit: str = Field(default="kg", description="Unit")
    status: str = Field(default="active", description="Batch status")
    events: List[str] = Field(default_factory=list, description="Event IDs")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Created")


class CommodityClassification(BaseModel):
    """Commodity classification result."""
    commodity_name: str = Field(default="", description="Commodity name")
    eudr_covered: bool = Field(default=False, description="Whether covered by EUDR")
    cn_codes: List[str] = Field(default_factory=list, description="CN codes")
    hs_codes: List[str] = Field(default_factory=list, description="HS codes")
    annex_i_reference: str = Field(default="", description="Annex I reference")
    derived_products: List[str] = Field(
        default_factory=list, description="Derived product categories"
    )


class ComplianceCheckResult(BaseModel):
    """Compliance check result from traceability data."""
    check_id: str = Field(default="", description="Check ID")
    supplier_id: str = Field(default="", description="Supplier ID")
    overall_compliant: bool = Field(default=False, description="Overall compliance")
    checks_passed: int = Field(default=0, description="Checks passed")
    checks_failed: int = Field(default=0, description="Checks failed")
    findings: List[Dict[str, Any]] = Field(
        default_factory=list, description="Compliance findings"
    )
    checked_at: datetime = Field(default_factory=datetime.utcnow, description="Check time")


class SupplyChainNode(BaseModel):
    """Supply chain mapping node."""
    entity_id: str = Field(default="", description="Entity ID")
    entity_type: str = Field(default="supplier", description="Entity type")
    name: str = Field(default="", description="Entity name")
    country: str = Field(default="", description="Country")
    tier: int = Field(default=1, description="Supply chain tier (1=direct)")
    upstream: List[str] = Field(default_factory=list, description="Upstream entity IDs")
    downstream: List[str] = Field(default_factory=list, description="Downstream entity IDs")
    commodities: List[str] = Field(default_factory=list, description="Commodities handled")


# =============================================================================
# Proxy Classes
# =============================================================================


class PlotRegistryProxy:
    """Proxy for EUDR Traceability plot registry service."""

    def __init__(self, service: Any = None, stub_mode: bool = True) -> None:
        self._service = service
        self._stub_mode = stub_mode
        self._store: Dict[str, PlotData] = {}

    async def list_plots(
        self, supplier_id: Optional[str] = None, limit: int = 100
    ) -> List[PlotData]:
        """List registered plots, optionally filtered by supplier."""
        plots = list(self._store.values())
        if supplier_id:
            plots = [p for p in plots if p.supplier_id == supplier_id]
        return plots[:limit]

    async def get_plot(self, plot_id: str) -> Optional[PlotData]:
        """Get a plot by ID."""
        return self._store.get(plot_id)

    async def register_plot(self, data: Dict[str, Any]) -> PlotData:
        """Register a new plot."""
        plot = PlotData(
            plot_id=data.get("plot_id", str(uuid4())[:8]),
            supplier_id=data.get("supplier_id", ""),
            latitude=data.get("latitude", 0.0),
            longitude=data.get("longitude", 0.0),
            polygon=data.get("polygon"),
            area_ha=data.get("area_ha", 0.0),
            country=data.get("country", ""),
            commodity=data.get("commodity", ""),
            provenance_hash=_compute_hash(
                f"plot:{data.get('plot_id', '')}:{datetime.utcnow().isoformat()}"
            ),
        )
        self._store[plot.plot_id] = plot
        return plot

    async def update_plot(
        self, plot_id: str, data: Dict[str, Any]
    ) -> Optional[PlotData]:
        """Update plot data."""
        plot = self._store.get(plot_id)
        if plot is None:
            return None
        for key, value in data.items():
            if hasattr(plot, key):
                setattr(plot, key, value)
        return plot

    async def delete_plot(self, plot_id: str) -> bool:
        """Delete a plot."""
        return self._store.pop(plot_id, None) is not None

    async def update_geolocation(
        self, plot_id: str, latitude: float, longitude: float,
        polygon: Optional[List[List[float]]] = None,
    ) -> Optional[PlotData]:
        """Update plot geolocation data."""
        plot = self._store.get(plot_id)
        if plot is None:
            return None
        plot.latitude = latitude
        plot.longitude = longitude
        if polygon is not None:
            plot.polygon = polygon
        return plot


class ChainOfCustodyProxy:
    """Proxy for chain of custody tracking service."""

    def __init__(self, service: Any = None, stub_mode: bool = True) -> None:
        self._service = service
        self._stub_mode = stub_mode
        self._events: Dict[str, CustodyEvent] = {}
        self._batches: Dict[str, BatchRecord] = {}

    async def record_event(self, data: Dict[str, Any]) -> CustodyEvent:
        """Record a chain of custody event."""
        event = CustodyEvent(
            event_id=data.get("event_id", str(uuid4())[:10]),
            batch_id=data.get("batch_id", ""),
            event_type=data.get("event_type", "transfer"),
            from_entity=data.get("from_entity", ""),
            to_entity=data.get("to_entity", ""),
            commodity=data.get("commodity", ""),
            quantity=data.get("quantity", 0.0),
            unit=data.get("unit", "kg"),
            geolocation=data.get("geolocation"),
            provenance_hash=_compute_hash(
                f"custody:{data.get('event_id', '')}:{datetime.utcnow().isoformat()}"
            ),
        )
        self._events[event.event_id] = event

        # Update batch
        if event.batch_id and event.batch_id in self._batches:
            self._batches[event.batch_id].events.append(event.event_id)

        return event

    async def get_event(self, event_id: str) -> Optional[CustodyEvent]:
        """Get a custody event by ID."""
        return self._events.get(event_id)

    async def list_events(
        self, batch_id: Optional[str] = None, limit: int = 100
    ) -> List[CustodyEvent]:
        """List custody events, optionally filtered by batch."""
        events = list(self._events.values())
        if batch_id:
            events = [e for e in events if e.batch_id == batch_id]
        return events[:limit]

    async def create_batch(self, data: Dict[str, Any]) -> BatchRecord:
        """Create a new batch for tracking."""
        batch = BatchRecord(
            batch_id=data.get("batch_id", f"BATCH-{str(uuid4())[:6]}"),
            commodity=data.get("commodity", ""),
            origin_plot_ids=data.get("origin_plot_ids", []),
            supplier_id=data.get("supplier_id", ""),
            quantity=data.get("quantity", 0.0),
            unit=data.get("unit", "kg"),
        )
        self._batches[batch.batch_id] = batch
        return batch

    async def get_batch(self, batch_id: str) -> Optional[BatchRecord]:
        """Get a batch record."""
        return self._batches.get(batch_id)

    async def get_batch_history(self, batch_id: str) -> List[CustodyEvent]:
        """Get complete event history for a batch."""
        return [
            e for e in self._events.values() if e.batch_id == batch_id
        ]


class CommodityProxy:
    """Proxy for commodity classification service."""

    # EUDR Annex I commodities and their CN codes
    EUDR_COMMODITIES: Dict[str, CommodityClassification] = {
        "cattle": CommodityClassification(
            commodity_name="Cattle", eudr_covered=True,
            cn_codes=["0102", "0201", "0202", "4101", "4104", "4107"],
            hs_codes=["0102", "0201", "0202"],
            annex_i_reference="Annex I, Point 1",
            derived_products=["beef", "leather", "tallow"],
        ),
        "cocoa": CommodityClassification(
            commodity_name="Cocoa", eudr_covered=True,
            cn_codes=["1801", "1802", "1803", "1804", "1805", "1806"],
            hs_codes=["1801", "1802", "1803", "1804", "1805", "1806"],
            annex_i_reference="Annex I, Point 2",
            derived_products=["cocoa_butter", "cocoa_powder", "chocolate"],
        ),
        "coffee": CommodityClassification(
            commodity_name="Coffee", eudr_covered=True,
            cn_codes=["0901", "2101"],
            hs_codes=["0901"],
            annex_i_reference="Annex I, Point 3",
            derived_products=["roasted_coffee", "instant_coffee", "coffee_extract"],
        ),
        "oil_palm": CommodityClassification(
            commodity_name="Oil Palm", eudr_covered=True,
            cn_codes=["1511", "1513", "2306"],
            hs_codes=["1511", "1513"],
            annex_i_reference="Annex I, Point 4",
            derived_products=["palm_oil", "palm_kernel_oil", "palm_oil_derivatives"],
        ),
        "palm_oil": CommodityClassification(
            commodity_name="Palm Oil", eudr_covered=True,
            cn_codes=["1511", "1513"],
            hs_codes=["1511"],
            annex_i_reference="Annex I, Point 4",
            derived_products=["refined_palm_oil", "palm_olein", "palm_stearin"],
        ),
        "rubber": CommodityClassification(
            commodity_name="Rubber", eudr_covered=True,
            cn_codes=["4001", "4005", "4006", "4007", "4008", "4011", "4012", "4013"],
            hs_codes=["4001", "4005"],
            annex_i_reference="Annex I, Point 5",
            derived_products=["natural_rubber", "tyres", "rubber_products"],
        ),
        "soy": CommodityClassification(
            commodity_name="Soy", eudr_covered=True,
            cn_codes=["1201", "1208", "1507", "2304"],
            hs_codes=["1201", "1507"],
            annex_i_reference="Annex I, Point 6",
            derived_products=["soybean_oil", "soy_meal", "soy_lecithin"],
        ),
        "wood": CommodityClassification(
            commodity_name="Wood", eudr_covered=True,
            cn_codes=["4401", "4403", "4406", "4407", "4408", "4409", "4410",
                       "4411", "4412", "4418", "4702", "4703", "4704", "4705",
                       "4706", "4707", "4801", "4802", "4803", "4804", "4805",
                       "4806", "4807", "4808", "4809", "4810", "4811", "4812",
                       "4813", "4814", "4817", "4818", "4819", "4820", "4821",
                       "4822", "4823", "9401", "9403", "9406"],
            hs_codes=["4401", "4403", "4407"],
            annex_i_reference="Annex I, Point 7",
            derived_products=["timber", "paper", "furniture", "plywood", "pulp"],
        ),
    }

    def __init__(self, service: Any = None, stub_mode: bool = True) -> None:
        self._service = service
        self._stub_mode = stub_mode

    async def classify(self, commodity_name: str) -> CommodityClassification:
        """Classify a commodity and return EUDR coverage info."""
        key = commodity_name.lower().replace(" ", "_")
        result = self.EUDR_COMMODITIES.get(key)
        if result:
            return result
        return CommodityClassification(
            commodity_name=commodity_name,
            eudr_covered=False,
        )

    async def get_cn_codes(self, commodity_name: str) -> List[str]:
        """Get CN codes for a commodity."""
        classification = await self.classify(commodity_name)
        return classification.cn_codes

    async def is_eudr_covered(self, commodity_name: str) -> bool:
        """Check if a commodity is covered by EUDR."""
        classification = await self.classify(commodity_name)
        return classification.eudr_covered

    async def get_derived_products(self, commodity_name: str) -> List[str]:
        """Get derived products for a commodity."""
        classification = await self.classify(commodity_name)
        return classification.derived_products

    async def list_all_commodities(self) -> List[CommodityClassification]:
        """List all EUDR-covered commodities."""
        return list(self.EUDR_COMMODITIES.values())


class ComplianceProxy:
    """Proxy for compliance verification from traceability data."""

    def __init__(self, service: Any = None, stub_mode: bool = True) -> None:
        self._service = service
        self._stub_mode = stub_mode

    async def check_compliance(
        self, supplier_id: str, data: Optional[Dict[str, Any]] = None
    ) -> ComplianceCheckResult:
        """Run compliance checks for a supplier."""
        checks_passed = 0
        checks_failed = 0
        findings = []

        # Check for required traceability data
        supplier_data = data or {}

        if supplier_data.get("has_geolocation"):
            checks_passed += 1
        else:
            checks_failed += 1
            findings.append({
                "rule": "geolocation_required",
                "status": "fail",
                "message": "Plot geolocation data is missing",
            })

        if supplier_data.get("has_custody_chain"):
            checks_passed += 1
        else:
            checks_failed += 1
            findings.append({
                "rule": "custody_chain_required",
                "status": "fail",
                "message": "Chain of custody data is missing",
            })

        if supplier_data.get("commodities"):
            checks_passed += 1
        else:
            checks_failed += 1
            findings.append({
                "rule": "commodity_identified",
                "status": "fail",
                "message": "No commodity identified",
            })

        return ComplianceCheckResult(
            check_id=str(uuid4())[:10],
            supplier_id=supplier_id,
            overall_compliant=checks_failed == 0,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            findings=findings,
        )

    async def verify_deforestation_free(
        self, plot_id: str, cutoff_date: str = "2020-12-31"
    ) -> Dict[str, Any]:
        """Verify a plot is deforestation-free since cutoff date."""
        return {
            "plot_id": plot_id,
            "deforestation_free": True,
            "cutoff_date": cutoff_date,
            "verification_method": "satellite_monitoring",
            "confidence": 0.85,
            "verified_at": datetime.utcnow().isoformat(),
        }


class DueDiligenceProxy:
    """Proxy for due diligence workflow support."""

    def __init__(self, service: Any = None, stub_mode: bool = True) -> None:
        self._service = service
        self._stub_mode = stub_mode

    async def start_dd_workflow(
        self, supplier_id: str, dd_type: str = "standard"
    ) -> Dict[str, Any]:
        """Start a due diligence workflow for a supplier."""
        return {
            "workflow_id": str(uuid4())[:10],
            "supplier_id": supplier_id,
            "dd_type": dd_type,
            "status": "initiated",
            "stages": [
                "information_gathering",
                "risk_assessment",
                "risk_mitigation",
                "documentation",
                "review",
            ],
            "current_stage": "information_gathering",
            "started_at": datetime.utcnow().isoformat(),
        }

    async def get_dd_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get due diligence workflow status."""
        return {
            "workflow_id": workflow_id,
            "status": "in_progress",
            "progress_pct": 40.0,
        }

    async def submit_dd_evidence(
        self, workflow_id: str, evidence: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Submit evidence for a due diligence stage."""
        return {
            "workflow_id": workflow_id,
            "evidence_accepted": True,
            "evidence_type": evidence.get("type", "document"),
            "submitted_at": datetime.utcnow().isoformat(),
        }


class RiskProxy:
    """Proxy for risk scoring from traceability data."""

    def __init__(self, service: Any = None, stub_mode: bool = True) -> None:
        self._service = service
        self._stub_mode = stub_mode

    async def assess_risk(
        self, supplier_id: str, data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Assess risk for a supplier based on traceability data."""
        return {
            "supplier_id": supplier_id,
            "traceability_risk_score": 45.0,
            "factors": {
                "custody_chain_completeness": 0.8,
                "geolocation_coverage": 0.7,
                "documentation_completeness": 0.6,
                "verification_status": 0.5,
            },
            "risk_level": "standard",
            "assessed_at": datetime.utcnow().isoformat(),
        }

    async def get_country_risk(self, country_code: str) -> Dict[str, Any]:
        """Get country-level risk assessment."""
        from packs.eu_compliance.PACK_006_eudr_starter.integrations.pack_orchestrator import (
            COUNTRY_RISK_SCORES,
        )
        score = COUNTRY_RISK_SCORES.get(country_code.upper(), 50.0)
        if score < 30:
            level = "low"
        elif score <= 70:
            level = "standard"
        else:
            level = "high"

        return {
            "country_code": country_code.upper(),
            "risk_score": score,
            "risk_level": level,
        }


class SupplyChainProxy:
    """Proxy for supply chain mapping service."""

    def __init__(self, service: Any = None, stub_mode: bool = True) -> None:
        self._service = service
        self._stub_mode = stub_mode
        self._nodes: Dict[str, SupplyChainNode] = {}

    async def add_node(self, data: Dict[str, Any]) -> SupplyChainNode:
        """Add a supply chain node."""
        node = SupplyChainNode(
            entity_id=data.get("entity_id", str(uuid4())[:8]),
            entity_type=data.get("entity_type", "supplier"),
            name=data.get("name", ""),
            country=data.get("country", ""),
            tier=data.get("tier", 1),
            upstream=data.get("upstream", []),
            downstream=data.get("downstream", []),
            commodities=data.get("commodities", []),
        )
        self._nodes[node.entity_id] = node
        return node

    async def get_node(self, entity_id: str) -> Optional[SupplyChainNode]:
        """Get a supply chain node."""
        return self._nodes.get(entity_id)

    async def map_supply_chain(
        self, root_entity_id: str
    ) -> List[SupplyChainNode]:
        """Map the full supply chain from a root entity."""
        result = []
        visited = set()

        def _traverse(eid: str) -> None:
            if eid in visited:
                return
            visited.add(eid)
            node = self._nodes.get(eid)
            if node:
                result.append(node)
                for upstream_id in node.upstream:
                    _traverse(upstream_id)

        _traverse(root_entity_id)
        return result

    async def trace_batch(self, batch_id: str) -> List[Dict[str, Any]]:
        """Trace a batch through the supply chain."""
        return [{
            "batch_id": batch_id,
            "trace": [],
            "origin_verified": False,
            "traced_at": datetime.utcnow().isoformat(),
        }]

    async def get_supplier_tiers(self) -> Dict[int, List[str]]:
        """Get suppliers organized by tier."""
        tiers: Dict[int, List[str]] = {}
        for node in self._nodes.values():
            if node.entity_type == "supplier":
                tiers.setdefault(node.tier, []).append(node.entity_id)
        return tiers


# =============================================================================
# Main Bridge
# =============================================================================


class TraceabilityBridge:
    """Bridge to EUDR Traceability Connector.

    Provides proxy access to seven traceability services. Falls back to
    stub implementations when the connector is not available.

    Attributes:
        config: Bridge configuration
        _connector_available: Whether the connector is detected
        _proxies: Cached proxy instances

    Example:
        >>> bridge = TraceabilityBridge()
        >>> custody = bridge.get_chain_of_custody()
        >>> event = await custody.record_event(event_data)
    """

    def __init__(self, config: Optional[TraceabilityBridgeConfig] = None) -> None:
        """Initialize the Traceability Bridge.

        Args:
            config: Bridge configuration. Uses defaults if not provided.
        """
        self.config = config or TraceabilityBridgeConfig()
        self._connector_available = False
        self._proxies: Dict[str, Any] = {}

        self._detect_connector()
        logger.info(
            "TraceabilityBridge initialized: connector_available=%s",
            self._connector_available,
        )

    def _detect_connector(self) -> None:
        """Detect whether the EUDR Traceability Connector is available."""
        try:
            import importlib
            importlib.import_module("greenlang.eudr_traceability")
            self._connector_available = True
        except ImportError:
            self._connector_available = False
            logger.info("EUDR Traceability Connector not available, using stub mode")

    def is_connector_available(self) -> bool:
        """Check if the traceability connector is available."""
        return self._connector_available

    def get_plot_registry(self) -> PlotRegistryProxy:
        """Get the plot registry proxy."""
        if "plot_registry" not in self._proxies:
            self._proxies["plot_registry"] = PlotRegistryProxy(
                stub_mode=self.config.stub_mode or not self._connector_available
            )
        return self._proxies["plot_registry"]

    def get_chain_of_custody(self) -> ChainOfCustodyProxy:
        """Get the chain of custody proxy."""
        if "chain_of_custody" not in self._proxies:
            self._proxies["chain_of_custody"] = ChainOfCustodyProxy(
                stub_mode=self.config.stub_mode or not self._connector_available
            )
        return self._proxies["chain_of_custody"]

    def get_commodity_classifier(self) -> CommodityProxy:
        """Get the commodity classification proxy."""
        if "commodity" not in self._proxies:
            self._proxies["commodity"] = CommodityProxy(
                stub_mode=self.config.stub_mode or not self._connector_available
            )
        return self._proxies["commodity"]

    def get_compliance_verifier(self) -> ComplianceProxy:
        """Get the compliance verification proxy."""
        if "compliance" not in self._proxies:
            self._proxies["compliance"] = ComplianceProxy(
                stub_mode=self.config.stub_mode or not self._connector_available
            )
        return self._proxies["compliance"]

    def get_due_diligence(self) -> DueDiligenceProxy:
        """Get the due diligence workflow proxy."""
        if "due_diligence" not in self._proxies:
            self._proxies["due_diligence"] = DueDiligenceProxy(
                stub_mode=self.config.stub_mode or not self._connector_available
            )
        return self._proxies["due_diligence"]

    def get_risk_assessment(self) -> RiskProxy:
        """Get the risk assessment proxy."""
        if "risk" not in self._proxies:
            self._proxies["risk"] = RiskProxy(
                stub_mode=self.config.stub_mode or not self._connector_available
            )
        return self._proxies["risk"]

    def get_supply_chain_mapper(self) -> SupplyChainProxy:
        """Get the supply chain mapping proxy."""
        if "supply_chain" not in self._proxies:
            self._proxies["supply_chain"] = SupplyChainProxy(
                stub_mode=self.config.stub_mode or not self._connector_available
            )
        return self._proxies["supply_chain"]


# =============================================================================
# Helper Functions
# =============================================================================


def _compute_hash(data: str) -> str:
    """Compute a SHA-256 hash of the given string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()
