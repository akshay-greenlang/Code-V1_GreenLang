# -*- coding: utf-8 -*-
"""
Agent Loader - GreenLang Agent ID to Module Registry
=====================================================

Centralized registry mapping pack agent IDs (e.g., GL-MRV-X-001) to their
GreenLang module paths. Provides lazy loading with graceful skip behavior
for missing dependencies.

Usage:
    >>> from packs.eu_compliance.agent_loader import AgentLoader
    >>> loader = AgentLoader()
    >>> available = loader.load_all_available()
    >>> print(f"Loaded {len(available)} agents")
    >>> engine = loader.get("GL-MRV-X-001")

Author: GreenLang Team
Version: 1.0.0
"""

import importlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class AgentEntry:
    """Registry entry for a GreenLang agent."""
    agent_id: str
    module_path: str
    category: str  # "mrv", "data", "eudr", "found", "app"
    description: str = ""
    service_class: str = ""  # Preferred class name to instantiate
    loaded: bool = False
    instance: Any = None
    error: Optional[str] = None


# =============================================================================
# Agent Registry - maps agent IDs to greenlang module paths
# =============================================================================

AGENT_REGISTRY: Dict[str, AgentEntry] = {
    # -------------------------------------------------------------------------
    # MRV Agents (Scope 1)
    # -------------------------------------------------------------------------
    "GL-MRV-X-001": AgentEntry(
        agent_id="GL-MRV-X-001",
        module_path="greenlang.stationary_combustion",
        category="mrv",
        description="Stationary Combustion Agent",
        service_class="StationaryCombustionService",
    ),
    "GL-MRV-X-002": AgentEntry(
        agent_id="GL-MRV-X-002",
        module_path="greenlang.refrigerants_fgas",
        category="mrv",
        description="Refrigerants & F-Gas Agent",
    ),
    "GL-MRV-X-003": AgentEntry(
        agent_id="GL-MRV-X-003",
        module_path="greenlang.mobile_combustion",
        category="mrv",
        description="Mobile Combustion Agent",
    ),
    "GL-MRV-X-004": AgentEntry(
        agent_id="GL-MRV-X-004",
        module_path="greenlang.process_emissions",
        category="mrv",
        description="Process Emissions Agent",
    ),
    "GL-MRV-X-005": AgentEntry(
        agent_id="GL-MRV-X-005",
        module_path="greenlang.fugitive_emissions",
        category="mrv",
        description="Fugitive Emissions Agent",
    ),
    "GL-MRV-X-006": AgentEntry(
        agent_id="GL-MRV-X-006",
        module_path="greenlang.land_use_emissions",
        category="mrv",
        description="Land Use Emissions Agent",
    ),
    "GL-MRV-X-007": AgentEntry(
        agent_id="GL-MRV-X-007",
        module_path="greenlang.waste_treatment",
        category="mrv",
        description="Waste Treatment Emissions Agent",
    ),
    "GL-MRV-X-008": AgentEntry(
        agent_id="GL-MRV-X-008",
        module_path="greenlang.agricultural_emissions",
        category="mrv",
        description="Agricultural Emissions Agent",
    ),
    # -------------------------------------------------------------------------
    # MRV Agents (Scope 2)
    # -------------------------------------------------------------------------
    "GL-MRV-X-009": AgentEntry(
        agent_id="GL-MRV-X-009",
        module_path="greenlang.scope2_location",
        category="mrv",
        description="Scope 2 Location-Based Agent",
    ),
    "GL-MRV-X-010": AgentEntry(
        agent_id="GL-MRV-X-010",
        module_path="greenlang.scope2_market",
        category="mrv",
        description="Scope 2 Market-Based Agent",
    ),
    "GL-MRV-X-011": AgentEntry(
        agent_id="GL-MRV-X-011",
        module_path="greenlang.steam_heat",
        category="mrv",
        description="Steam/Heat Purchase Agent",
    ),
    "GL-MRV-X-012": AgentEntry(
        agent_id="GL-MRV-X-012",
        module_path="greenlang.cooling_purchase",
        category="mrv",
        description="Cooling Purchase Agent",
    ),
    "GL-MRV-X-013": AgentEntry(
        agent_id="GL-MRV-X-013",
        module_path="greenlang.dual_reporting",
        category="mrv",
        description="Dual Reporting Reconciliation",
    ),
    # -------------------------------------------------------------------------
    # MRV Agents (Scope 3, Categories 1-15 + cross-cutting)
    # -------------------------------------------------------------------------
    "GL-MRV-X-014": AgentEntry(
        agent_id="GL-MRV-X-014",
        module_path="greenlang.purchased_goods",
        category="mrv",
        description="Purchased Goods & Services (Cat 1)",
    ),
    "GL-MRV-X-015": AgentEntry(
        agent_id="GL-MRV-X-015",
        module_path="greenlang.capital_goods",
        category="mrv",
        description="Capital Goods (Cat 2)",
    ),
    "GL-MRV-X-016": AgentEntry(
        agent_id="GL-MRV-X-016",
        module_path="greenlang.fuel_energy_activities",
        category="mrv",
        description="Fuel & Energy Activities (Cat 3)",
    ),
    "GL-MRV-X-017": AgentEntry(
        agent_id="GL-MRV-X-017",
        module_path="greenlang.upstream_transportation",
        category="mrv",
        description="Upstream Transportation (Cat 4)",
    ),
    "GL-MRV-X-018": AgentEntry(
        agent_id="GL-MRV-X-018",
        module_path="greenlang.waste_generated",
        category="mrv",
        description="Waste Generated (Cat 5)",
    ),
    "GL-MRV-X-019": AgentEntry(
        agent_id="GL-MRV-X-019",
        module_path="greenlang.business_travel",
        category="mrv",
        description="Business Travel (Cat 6)",
    ),
    "GL-MRV-X-020": AgentEntry(
        agent_id="GL-MRV-X-020",
        module_path="greenlang.employee_commuting",
        category="mrv",
        description="Employee Commuting (Cat 7)",
    ),
    "GL-MRV-X-021": AgentEntry(
        agent_id="GL-MRV-X-021",
        module_path="greenlang.upstream_leased",
        category="mrv",
        description="Upstream Leased Assets (Cat 8)",
    ),
    "GL-MRV-X-022": AgentEntry(
        agent_id="GL-MRV-X-022",
        module_path="greenlang.downstream_transportation",
        category="mrv",
        description="Downstream Transportation (Cat 9)",
    ),
    "GL-MRV-X-023": AgentEntry(
        agent_id="GL-MRV-X-023",
        module_path="greenlang.processing_sold_products",
        category="mrv",
        description="Processing of Sold Products (Cat 10)",
    ),
    "GL-MRV-X-024": AgentEntry(
        agent_id="GL-MRV-X-024",
        module_path="greenlang.use_sold_products",
        category="mrv",
        description="Use of Sold Products (Cat 11)",
    ),
    "GL-MRV-X-025": AgentEntry(
        agent_id="GL-MRV-X-025",
        module_path="greenlang.end_of_life",
        category="mrv",
        description="End-of-Life Treatment (Cat 12)",
    ),
    "GL-MRV-X-026": AgentEntry(
        agent_id="GL-MRV-X-026",
        module_path="greenlang.downstream_leased",
        category="mrv",
        description="Downstream Leased Assets (Cat 13)",
    ),
    "GL-MRV-X-027": AgentEntry(
        agent_id="GL-MRV-X-027",
        module_path="greenlang.franchises",
        category="mrv",
        description="Franchises (Cat 14)",
    ),
    "GL-MRV-X-028": AgentEntry(
        agent_id="GL-MRV-X-028",
        module_path="greenlang.investments",
        category="mrv",
        description="Investments (Cat 15)",
    ),
    "GL-MRV-X-029": AgentEntry(
        agent_id="GL-MRV-X-029",
        module_path="greenlang.scope3_category_mapper",
        category="mrv",
        description="Scope 3 Category Mapper",
    ),
    "GL-MRV-X-030": AgentEntry(
        agent_id="GL-MRV-X-030",
        module_path="greenlang.audit_trail_lineage",
        category="mrv",
        description="Audit Trail & Lineage",
    ),
    # -------------------------------------------------------------------------
    # Data Agents (Intake)
    # -------------------------------------------------------------------------
    "GL-DATA-X-001": AgentEntry(
        agent_id="GL-DATA-X-001",
        module_path="greenlang.pdf_extractor",
        category="data",
        description="PDF & Invoice Extractor",
    ),
    "GL-DATA-X-002": AgentEntry(
        agent_id="GL-DATA-X-002",
        module_path="greenlang.excel_normalizer",
        category="data",
        description="Excel/CSV Normalizer",
    ),
    "GL-DATA-X-003": AgentEntry(
        agent_id="GL-DATA-X-003",
        module_path="greenlang.erp_connector",
        category="data",
        description="ERP/Finance Connector",
    ),
    "GL-DATA-X-004": AgentEntry(
        agent_id="GL-DATA-X-004",
        module_path="greenlang.api_gateway_agent",
        category="data",
        description="API Gateway Agent",
    ),
    "GL-DATA-X-005": AgentEntry(
        agent_id="GL-DATA-X-005",
        module_path="greenlang.eudr_traceability",
        category="data",
        description="EUDR Traceability Connector",
    ),
    "GL-DATA-X-006": AgentEntry(
        agent_id="GL-DATA-X-006",
        module_path="greenlang.gis_connector",
        category="data",
        description="GIS/Mapping Connector",
    ),
    "GL-DATA-X-007": AgentEntry(
        agent_id="GL-DATA-X-007",
        module_path="greenlang.satellite_connector",
        category="data",
        description="Deforestation Satellite Connector",
    ),
    # -------------------------------------------------------------------------
    # Data Agents (Quality)
    # -------------------------------------------------------------------------
    "GL-DATA-X-008": AgentEntry(
        agent_id="GL-DATA-X-008",
        module_path="greenlang.supplier_questionnaire",
        category="data",
        description="Supplier Questionnaire Processor",
    ),
    "GL-DATA-X-009": AgentEntry(
        agent_id="GL-DATA-X-009",
        module_path="greenlang.spend_categorizer",
        category="data",
        description="Spend Data Categorizer",
    ),
    "GL-DATA-X-010": AgentEntry(
        agent_id="GL-DATA-X-010",
        module_path="greenlang.data_quality_profiler",
        category="data",
        description="Data Quality Profiler",
    ),
    "GL-DATA-X-011": AgentEntry(
        agent_id="GL-DATA-X-011",
        module_path="greenlang.duplicate_detector",
        category="data",
        description="Duplicate Detection Agent",
    ),
    "GL-DATA-X-012": AgentEntry(
        agent_id="GL-DATA-X-012",
        module_path="greenlang.missing_value_imputer",
        category="data",
        description="Missing Value Imputer",
    ),
    "GL-DATA-X-013": AgentEntry(
        agent_id="GL-DATA-X-013",
        module_path="greenlang.outlier_detector",
        category="data",
        description="Outlier Detection Agent",
    ),
    "GL-DATA-X-019": AgentEntry(
        agent_id="GL-DATA-X-019",
        module_path="greenlang.validation_rule_engine",
        category="data",
        description="Validation Rule Engine",
    ),
    # -------------------------------------------------------------------------
    # EUDR Agents (Supply Chain Traceability - 001-015)
    # -------------------------------------------------------------------------
    "GL-EUDR-X-001": AgentEntry(
        agent_id="GL-EUDR-X-001",
        module_path="greenlang.agents.eudr.supply_chain_mapper",
        category="eudr",
        description="EUDR Supply Chain Mapper",
    ),
    "GL-EUDR-X-002": AgentEntry(
        agent_id="GL-EUDR-X-002",
        module_path="greenlang.agents.eudr.geolocation_verification",
        category="eudr",
        description="EUDR Geolocation Verification Agent",
    ),
    "GL-EUDR-X-003": AgentEntry(
        agent_id="GL-EUDR-X-003",
        module_path="greenlang.agents.eudr.satellite_monitoring",
        category="eudr",
        description="EUDR Satellite Monitoring Agent",
    ),
    "GL-EUDR-X-004": AgentEntry(
        agent_id="GL-EUDR-X-004",
        module_path="greenlang.agents.eudr.forest_cover_analysis",
        category="eudr",
        description="EUDR Forest Cover Analysis Agent",
    ),
    "GL-EUDR-X-005": AgentEntry(
        agent_id="GL-EUDR-X-005",
        module_path="greenlang.agents.eudr.land_use_change",
        category="eudr",
        description="EUDR Land Use Change Detection Agent",
    ),
    "GL-EUDR-X-006": AgentEntry(
        agent_id="GL-EUDR-X-006",
        module_path="greenlang.agents.eudr.plot_boundary",
        category="eudr",
        description="EUDR Plot Boundary Validation Agent",
    ),
    "GL-EUDR-X-007": AgentEntry(
        agent_id="GL-EUDR-X-007",
        module_path="greenlang.agents.eudr.gps_coordinate_validator",
        category="eudr",
        description="EUDR GPS Coordinate Validator",
    ),
    "GL-EUDR-X-008": AgentEntry(
        agent_id="GL-EUDR-X-008",
        module_path="greenlang.agents.eudr.multi_tier_supplier",
        category="eudr",
        description="EUDR Multi-Tier Supplier Tracker",
    ),
    "GL-EUDR-X-009": AgentEntry(
        agent_id="GL-EUDR-X-009",
        module_path="greenlang.agents.eudr.chain_of_custody",
        category="eudr",
        description="EUDR Chain of Custody Agent",
    ),
    "GL-EUDR-X-010": AgentEntry(
        agent_id="GL-EUDR-X-010",
        module_path="greenlang.agents.eudr.segregation_verifier",
        category="eudr",
        description="EUDR Segregation Verifier",
    ),
    "GL-EUDR-X-011": AgentEntry(
        agent_id="GL-EUDR-X-011",
        module_path="greenlang.agents.eudr.mass_balance_calculator",
        category="eudr",
        description="EUDR Mass Balance Calculator",
    ),
    "GL-EUDR-X-012": AgentEntry(
        agent_id="GL-EUDR-X-012",
        module_path="greenlang.agents.eudr.document_authentication",
        category="eudr",
        description="EUDR Document Authentication Agent",
    ),
    "GL-EUDR-X-013": AgentEntry(
        agent_id="GL-EUDR-X-013",
        module_path="greenlang.agents.eudr.blockchain_integration",
        category="eudr",
        description="EUDR Blockchain Integration Agent",
    ),
    "GL-EUDR-X-014": AgentEntry(
        agent_id="GL-EUDR-X-014",
        module_path="greenlang.agents.eudr.qr_code_generator",
        category="eudr",
        description="EUDR QR Code Generator",
    ),
    "GL-EUDR-X-015": AgentEntry(
        agent_id="GL-EUDR-X-015",
        module_path="greenlang.agents.eudr.mobile_data_collector",
        category="eudr",
        description="EUDR Mobile Data Collector",
    ),
    # -------------------------------------------------------------------------
    # EUDR Agents (Risk Assessment - 016-020)
    # -------------------------------------------------------------------------
    "GL-EUDR-X-016": AgentEntry(
        agent_id="GL-EUDR-X-016",
        module_path="greenlang.agents.eudr.country_risk_evaluator",
        category="eudr",
        description="EUDR Country Risk Evaluator",
    ),
    "GL-EUDR-X-017": AgentEntry(
        agent_id="GL-EUDR-X-017",
        module_path="greenlang.agents.eudr.supplier_risk_scorer",
        category="eudr",
        description="EUDR Supplier Risk Scorer",
    ),
    "GL-EUDR-X-018": AgentEntry(
        agent_id="GL-EUDR-X-018",
        module_path="greenlang.agents.eudr.commodity_risk_analyzer",
        category="eudr",
        description="EUDR Commodity Risk Analyzer",
    ),
    "GL-EUDR-X-019": AgentEntry(
        agent_id="GL-EUDR-X-019",
        module_path="greenlang.agents.eudr.corruption_index_monitor",
        category="eudr",
        description="EUDR Corruption Index Monitor",
    ),
    "GL-EUDR-X-020": AgentEntry(
        agent_id="GL-EUDR-X-020",
        module_path="greenlang.agents.eudr.deforestation_alert_system",
        category="eudr",
        description="EUDR Deforestation Alert System",
    ),
    # -------------------------------------------------------------------------
    # EUDR Agents (Due Diligence Core - 021-026)
    # -------------------------------------------------------------------------
    "GL-EUDR-X-021": AgentEntry(
        agent_id="GL-EUDR-X-021",
        module_path="greenlang.agents.eudr.indigenous_rights_checker",
        category="eudr",
        description="EUDR Indigenous Rights Checker",
    ),
    "GL-EUDR-X-022": AgentEntry(
        agent_id="GL-EUDR-X-022",
        module_path="greenlang.agents.eudr.protected_area_validator",
        category="eudr",
        description="EUDR Protected Area Validator",
    ),
    "GL-EUDR-X-023": AgentEntry(
        agent_id="GL-EUDR-X-023",
        module_path="greenlang.agents.eudr.legal_compliance_verifier",
        category="eudr",
        description="EUDR Legal Compliance Verifier",
    ),
    "GL-EUDR-X-024": AgentEntry(
        agent_id="GL-EUDR-X-024",
        module_path="greenlang.agents.eudr.third_party_audit_manager",
        category="eudr",
        description="EUDR Third Party Audit Manager",
    ),
    "GL-EUDR-X-025": AgentEntry(
        agent_id="GL-EUDR-X-025",
        module_path="greenlang.agents.eudr.risk_mitigation_advisor",
        category="eudr",
        description="EUDR Risk Mitigation Advisor",
    ),
    "GL-EUDR-X-026": AgentEntry(
        agent_id="GL-EUDR-X-026",
        module_path="greenlang.agents.eudr.due_diligence_orchestrator",
        category="eudr",
        description="EUDR Due Diligence Orchestrator",
    ),
    # -------------------------------------------------------------------------
    # EUDR Agents (Support Agents - 027-029)
    # -------------------------------------------------------------------------
    "GL-EUDR-X-027": AgentEntry(
        agent_id="GL-EUDR-X-027",
        module_path="greenlang.agents.eudr.information_gathering",
        category="eudr",
        description="EUDR Information Gathering Agent",
    ),
    "GL-EUDR-X-028": AgentEntry(
        agent_id="GL-EUDR-X-028",
        module_path="greenlang.agents.eudr.risk_assessment_engine",
        category="eudr",
        description="EUDR Risk Assessment Engine",
    ),
    "GL-EUDR-X-029": AgentEntry(
        agent_id="GL-EUDR-X-029",
        module_path="greenlang.agents.eudr.mitigation_measure_designer",
        category="eudr",
        description="EUDR Mitigation Measure Designer",
    ),
    # -------------------------------------------------------------------------
    # EUDR Agents (Due Diligence Workflow - 030-040)
    # -------------------------------------------------------------------------
    "GL-EUDR-X-030": AgentEntry(
        agent_id="GL-EUDR-X-030",
        module_path="greenlang.agents.eudr.documentation_generator",
        category="eudr",
        description="EUDR Documentation Generator",
    ),
    "GL-EUDR-X-031": AgentEntry(
        agent_id="GL-EUDR-X-031",
        module_path="greenlang.agents.eudr.stakeholder_engagement",
        category="eudr",
        description="EUDR Stakeholder Engagement Agent",
    ),
    "GL-EUDR-X-032": AgentEntry(
        agent_id="GL-EUDR-X-032",
        module_path="greenlang.agents.eudr.grievance_mechanism_manager",
        category="eudr",
        description="EUDR Grievance Mechanism Manager",
    ),
    "GL-EUDR-X-033": AgentEntry(
        agent_id="GL-EUDR-X-033",
        module_path="greenlang.agents.eudr.continuous_monitoring",
        category="eudr",
        description="EUDR Continuous Monitoring Agent",
    ),
    "GL-EUDR-X-034": AgentEntry(
        agent_id="GL-EUDR-X-034",
        module_path="greenlang.agents.eudr.annual_review_scheduler",
        category="eudr",
        description="EUDR Annual Review Scheduler",
    ),
    "GL-EUDR-X-035": AgentEntry(
        agent_id="GL-EUDR-X-035",
        module_path="greenlang.agents.eudr.improvement_plan_creator",
        category="eudr",
        description="EUDR Improvement Plan Creator",
    ),
    "GL-EUDR-X-036": AgentEntry(
        agent_id="GL-EUDR-X-036",
        module_path="greenlang.agents.eudr.eu_information_system_interface",
        category="eudr",
        description="EUDR EU Information System Interface",
    ),
    "GL-EUDR-X-037": AgentEntry(
        agent_id="GL-EUDR-X-037",
        module_path="greenlang.agents.eudr.due_diligence_statement_creator",
        category="eudr",
        description="EUDR Due Diligence Statement Creator",
    ),
    "GL-EUDR-X-038": AgentEntry(
        agent_id="GL-EUDR-X-038",
        module_path="greenlang.agents.eudr.reference_number_generator",
        category="eudr",
        description="EUDR Reference Number Generator",
    ),
    "GL-EUDR-X-039": AgentEntry(
        agent_id="GL-EUDR-X-039",
        module_path="greenlang.agents.eudr.customs_declaration_support",
        category="eudr",
        description="EUDR Customs Declaration Support",
    ),
    "GL-EUDR-X-040": AgentEntry(
        agent_id="GL-EUDR-X-040",
        module_path="greenlang.agents.eudr.authority_communication_manager",
        category="eudr",
        description="EUDR Authority Communication Manager",
    ),
    # -------------------------------------------------------------------------
    # App modules
    # -------------------------------------------------------------------------
    "GL-CBAM-APP": AgentEntry(
        agent_id="GL-CBAM-APP",
        module_path="greenlang.apps.cbam",
        category="app",
        description="GL-CBAM-APP v1.1",
    ),
    "GL-EUDR-APP": AgentEntry(
        agent_id="GL-EUDR-APP",
        module_path="greenlang.apps.eudr",
        category="app",
        description="GL-EUDR-APP v1.0",
    ),
}


class AgentLoader:
    """Loads GreenLang agent modules on demand with graceful skip behavior.

    Maintains a session-scoped cache of loaded agent instances. Agents that
    fail to import are recorded with their error message and skipped in
    subsequent load attempts.

    Attributes:
        _loaded: Dict of successfully loaded agent ID to module reference
        _failed: Dict of failed agent IDs to error messages
        _instances: Dict of agent ID to instantiated service/engine objects
    """

    def __init__(self) -> None:
        self._loaded: Dict[str, Any] = {}
        self._failed: Dict[str, str] = {}
        self._instances: Dict[str, Any] = {}

    def load(self, agent_id: str) -> Optional[Any]:
        """Load a single agent module by ID.

        Args:
            agent_id: The agent identifier (e.g., "GL-MRV-X-001").

        Returns:
            The loaded module, or None if import failed.
        """
        if agent_id in self._loaded:
            return self._loaded[agent_id]
        if agent_id in self._failed:
            return None

        entry = AGENT_REGISTRY.get(agent_id)
        if entry is None:
            self._failed[agent_id] = f"Unknown agent ID: {agent_id}"
            return None

        try:
            mod = importlib.import_module(entry.module_path)
            self._loaded[agent_id] = mod
            entry.loaded = True
            logger.info("Loaded agent %s from %s", agent_id, entry.module_path)
            return mod
        except ImportError as exc:
            error_msg = str(exc)
            self._failed[agent_id] = error_msg
            entry.error = error_msg
            logger.debug("Failed to load %s: %s", agent_id, error_msg)
            return None

    def load_category(self, category: str) -> Dict[str, Any]:
        """Load all agents in a category (mrv, data, eudr, app).

        Args:
            category: Agent category to load.

        Returns:
            Dict of agent_id to loaded module for successful loads.
        """
        results: Dict[str, Any] = {}
        for agent_id, entry in AGENT_REGISTRY.items():
            if entry.category == category:
                mod = self.load(agent_id)
                if mod is not None:
                    results[agent_id] = mod
        return results

    def load_all_available(self) -> Dict[str, Any]:
        """Attempt to load all registered agents, skipping failures.

        Returns:
            Dict of agent_id to loaded module for all successful loads.
        """
        results: Dict[str, Any] = {}
        for agent_id in AGENT_REGISTRY:
            mod = self.load(agent_id)
            if mod is not None:
                results[agent_id] = mod
        return results

    def get(self, agent_id: str) -> Optional[Any]:
        """Get a previously loaded module, or load on demand.

        Args:
            agent_id: The agent identifier.

        Returns:
            The loaded module, or None.
        """
        return self._loaded.get(agent_id) or self.load(agent_id)

    def get_instance(self, agent_id: str) -> Optional[Any]:
        """Get or create an agent service instance.

        Attempts to instantiate the agent's service class. Returns cached
        instances on subsequent calls.

        Args:
            agent_id: The agent identifier.

        Returns:
            An instantiated service object, or None.
        """
        if agent_id in self._instances:
            return self._instances[agent_id]

        mod = self.get(agent_id)
        if mod is None:
            return None

        entry = AGENT_REGISTRY.get(agent_id)
        if entry is None:
            return None

        instance = _resolve_instance(mod, entry.service_class)
        if instance is not None:
            self._instances[agent_id] = instance
        return instance

    @property
    def loaded_ids(self) -> Set[str]:
        """Set of successfully loaded agent IDs."""
        return set(self._loaded.keys())

    @property
    def failed_ids(self) -> Dict[str, str]:
        """Dict of failed agent IDs to error messages."""
        return dict(self._failed)

    @property
    def available_count(self) -> int:
        """Number of successfully loaded agents."""
        return len(self._loaded)

    def summary(self) -> Dict[str, Any]:
        """Return a summary of load results."""
        return {
            "total_registered": len(AGENT_REGISTRY),
            "loaded": len(self._loaded),
            "failed": len(self._failed),
            "loaded_ids": sorted(self._loaded.keys()),
            "failed_ids": sorted(self._failed.keys()),
        }


def _resolve_instance(module: Any, preferred_class: str = "") -> Optional[Any]:
    """Try to instantiate an agent service from a module.

    Args:
        module: The imported module.
        preferred_class: Preferred class name to instantiate.

    Returns:
        An instantiated object, or None.
    """
    # Try preferred class first
    if preferred_class:
        cls = getattr(module, preferred_class, None)
        if cls is not None and isinstance(cls, type):
            try:
                return cls()
            except Exception:
                pass

    # Try get_service() factory
    factory = getattr(module, "get_service", None)
    if factory is not None and callable(factory):
        try:
            return factory()
        except Exception:
            pass

    # Try common patterns
    for attr_name in dir(module):
        if attr_name.startswith("_"):
            continue
        if attr_name.endswith("Service"):
            cls = getattr(module, attr_name, None)
            if isinstance(cls, type):
                try:
                    return cls()
                except Exception:
                    continue

    return None
