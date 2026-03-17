# -*- coding: utf-8 -*-
"""
CrossFrameworkDataMapperEngine - PACK-009 EU Climate Compliance Bundle Engine 1

Maps data fields between CSRD, CBAM, EUDR, and EU Taxonomy regulatory
frameworks. Provides bidirectional field-level mappings organized by
thematic category (GHG emissions, supply chain, activity classification,
financial data, climate risk, water/pollution, biodiversity) with
confidence scoring and optional conversion functions.

Mapping Categories:
    1. GHG_EMISSIONS   - CSRD E1 <-> CBAM embedded <-> Taxonomy CCM (~25)
    2. SUPPLY_CHAIN    - EUDR traceability <-> CSRD S1/S2 due diligence (~15)
    3. ACTIVITY_CLASS  - Taxonomy NACE <-> CBAM CN codes <-> CSRD sector (~15)
    4. FINANCIAL_DATA  - Taxonomy KPIs <-> CSRD financial <-> CBAM costs (~15)
    5. CLIMATE_RISK    - Taxonomy CCA <-> CSRD E1 risk <-> CBAM carbon price (~10)
    6. WATER_POLLUTION - Taxonomy WTR/PPC <-> CSRD E2/E3 (~10)
    7. BIODIVERSITY    - Taxonomy BIO <-> EUDR deforestation <-> CSRD E4 (~10)

Zero-Hallucination:
    - All field mappings derived from published regulation text
    - Confidence scores are deterministic lookup values
    - No LLM involvement in mapping logic
    - SHA-256 provenance hash on all results

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-009 EU Climate Compliance Bundle
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MappingType(str, Enum):
    """Type of field mapping between regulations."""
    EXACT = "EXACT"
    APPROXIMATE = "APPROXIMATE"
    DERIVED = "DERIVED"


class Regulation(str, Enum):
    """Supported EU regulations."""
    CSRD = "CSRD"
    CBAM = "CBAM"
    EUDR = "EUDR"
    EU_TAXONOMY = "EU_TAXONOMY"


class MappingCategory(str, Enum):
    """Thematic mapping category."""
    GHG_EMISSIONS = "GHG_EMISSIONS"
    SUPPLY_CHAIN = "SUPPLY_CHAIN"
    ACTIVITY_CLASSIFICATION = "ACTIVITY_CLASSIFICATION"
    FINANCIAL_DATA = "FINANCIAL_DATA"
    CLIMATE_RISK = "CLIMATE_RISK"
    WATER_POLLUTION = "WATER_POLLUTION"
    BIODIVERSITY = "BIODIVERSITY"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class FieldMapping(BaseModel):
    """Single field-level mapping between two regulations."""
    mapping_id: str = Field(default_factory=_new_uuid, description="Unique mapping identifier")
    source_regulation: str = Field(..., description="Source regulation (CSRD, CBAM, EUDR, EU_TAXONOMY)")
    source_field: str = Field(..., description="Source field identifier")
    source_description: str = Field(default="", description="Human-readable source field description")
    target_regulation: str = Field(..., description="Target regulation")
    target_field: str = Field(..., description="Target field identifier")
    target_description: str = Field(default="", description="Human-readable target field description")
    mapping_type: str = Field(default="EXACT", description="EXACT, APPROXIMATE, or DERIVED")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Mapping confidence 0.0-1.0")
    category: str = Field(default="", description="Thematic mapping category")
    conversion_notes: str = Field(default="", description="Notes on any conversion needed")
    bidirectional: bool = Field(default=True, description="Whether mapping works both directions")

    @field_validator("mapping_type", mode="before")
    @classmethod
    def _validate_mapping_type(cls, v: Any) -> str:
        allowed = {"EXACT", "APPROXIMATE", "DERIVED"}
        val = str(v).upper()
        if val not in allowed:
            return "APPROXIMATE"
        return val


class MappingResult(BaseModel):
    """Result of a single field mapping operation."""
    result_id: str = Field(default_factory=_new_uuid, description="Result identifier")
    source_regulation: str = Field(default="", description="Source regulation")
    source_field: str = Field(default="", description="Source field")
    source_value: Any = Field(default=None, description="Source value")
    target_regulation: str = Field(default="", description="Target regulation")
    target_field: str = Field(default="", description="Target field")
    mapped_value: Any = Field(default=None, description="Mapped/converted value")
    mapping_type: str = Field(default="EXACT", description="Mapping type used")
    confidence: float = Field(default=0.0, description="Mapping confidence")
    conversion_applied: bool = Field(default=False, description="Whether conversion was applied")
    conversion_notes: str = Field(default="", description="Conversion details")
    timestamp: str = Field(default_factory=lambda: _utcnow().isoformat(), description="Mapping timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class BatchMappingResult(BaseModel):
    """Result of a batch mapping operation."""
    result_id: str = Field(default_factory=_new_uuid, description="Batch result identifier")
    total_fields: int = Field(default=0, description="Total fields processed")
    mapped_count: int = Field(default=0, description="Successfully mapped fields")
    unmapped_count: int = Field(default=0, description="Fields with no mapping found")
    mappings: List[MappingResult] = Field(default_factory=list, description="Individual mapping results")
    unmapped_fields: List[str] = Field(default_factory=list, description="Fields that could not be mapped")
    average_confidence: float = Field(default=0.0, description="Average mapping confidence")
    timestamp: str = Field(default_factory=lambda: _utcnow().isoformat(), description="Batch timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class OverlapStatistics(BaseModel):
    """Statistics on field overlap between regulations."""
    result_id: str = Field(default_factory=_new_uuid, description="Statistics identifier")
    regulation_pair: str = Field(default="", description="Regulation pair (e.g. CSRD-CBAM)")
    total_source_fields: int = Field(default=0, description="Fields in source regulation")
    total_target_fields: int = Field(default=0, description="Fields in target regulation")
    exact_matches: int = Field(default=0, description="EXACT mappings count")
    approximate_matches: int = Field(default=0, description="APPROXIMATE mappings count")
    derived_matches: int = Field(default=0, description="DERIVED mappings count")
    overlap_percentage: float = Field(default=0.0, description="Overlap percentage")
    categories_covered: List[str] = Field(default_factory=list, description="Categories with mappings")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class MappingPath(BaseModel):
    """Multi-hop mapping path between two fields via intermediate regulations."""
    path_id: str = Field(default_factory=_new_uuid, description="Path identifier")
    source_regulation: str = Field(default="", description="Source regulation")
    source_field: str = Field(default="", description="Source field")
    target_regulation: str = Field(default="", description="Target regulation")
    target_field: str = Field(default="", description="Target field")
    hops: List[Dict[str, str]] = Field(default_factory=list, description="Intermediate mapping hops")
    total_confidence: float = Field(default=0.0, description="Compound confidence across hops")
    path_length: int = Field(default=0, description="Number of hops")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class CrossFrameworkDataMapperConfig(BaseModel):
    """Configuration for the CrossFrameworkDataMapperEngine."""
    min_confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Minimum confidence threshold for mappings"
    )
    enable_derived_mappings: bool = Field(
        default=True, description="Include DERIVED-type mappings in results"
    )
    enable_bidirectional_search: bool = Field(
        default=True, description="Search mappings in both directions"
    )
    max_path_hops: int = Field(
        default=3, ge=1, le=5,
        description="Maximum hops for multi-hop mapping path search"
    )
    default_conversion_unit: str = Field(
        default="tCO2e", description="Default emission unit for conversion"
    )


# ---------------------------------------------------------------------------
# Model rebuilds for forward reference resolution
# ---------------------------------------------------------------------------

CrossFrameworkDataMapperConfig.model_rebuild()
FieldMapping.model_rebuild()
MappingResult.model_rebuild()
BatchMappingResult.model_rebuild()
OverlapStatistics.model_rebuild()
MappingPath.model_rebuild()


# ---------------------------------------------------------------------------
# Cross-Framework Mapping Database (100+ mappings)
# ---------------------------------------------------------------------------

CROSS_FRAMEWORK_MAPPINGS: Dict[str, List[Dict[str, Any]]] = {
    # ===================================================================
    # Category 1: GHG EMISSIONS (~25 mappings)
    # CSRD E1 <-> CBAM embedded emissions <-> Taxonomy CCM metrics
    # ===================================================================
    "GHG_EMISSIONS": [
        {"source_regulation": "CSRD", "source_field": "E1_6_scope1_ghg_emissions", "source_description": "ESRS E1-6 Gross Scope 1 GHG emissions",
         "target_regulation": "CBAM", "target_field": "direct_emissions_tco2e", "target_description": "CBAM direct (Scope 1) embedded emissions",
         "mapping_type": "EXACT", "confidence": 0.95, "conversion_notes": "Same metric, both in tCO2e", "bidirectional": True},
        {"source_regulation": "CSRD", "source_field": "E1_6_scope2_ghg_emissions", "source_description": "ESRS E1-6 Gross Scope 2 GHG emissions",
         "target_regulation": "CBAM", "target_field": "indirect_emissions_tco2e", "target_description": "CBAM indirect embedded emissions from electricity",
         "mapping_type": "EXACT", "confidence": 0.90, "conversion_notes": "CBAM uses electricity-only indirect emissions", "bidirectional": True},
        {"source_regulation": "CSRD", "source_field": "E1_6_scope3_cat1_emissions", "source_description": "ESRS E1-6 Scope 3 Category 1 purchased goods",
         "target_regulation": "CBAM", "target_field": "total_embedded_emissions", "target_description": "CBAM total embedded emissions in imported goods",
         "mapping_type": "APPROXIMATE", "confidence": 0.75, "conversion_notes": "CBAM covers subset of Scope 3 Cat 1 (only CBAM goods)", "bidirectional": True},
        {"source_regulation": "CSRD", "source_field": "E1_6_total_ghg_emissions", "source_description": "ESRS E1-6 Total GHG emissions (all scopes)",
         "target_regulation": "EU_TAXONOMY", "target_field": "ccm_ghg_emissions_total", "target_description": "Taxonomy CCM total GHG emissions metric",
         "mapping_type": "EXACT", "confidence": 0.95, "conversion_notes": "Same metric, same unit", "bidirectional": True},
        {"source_regulation": "CBAM", "source_field": "specific_embedded_emissions", "source_description": "CBAM specific embedded emissions per tonne product",
         "target_regulation": "EU_TAXONOMY", "target_field": "ccm_emission_intensity", "target_description": "Taxonomy CCM emission intensity per unit output",
         "mapping_type": "APPROXIMATE", "confidence": 0.80, "conversion_notes": "Both measure intensity but boundary definitions may differ", "bidirectional": True},
        {"source_regulation": "CSRD", "source_field": "E1_4_ghg_reduction_target_pct", "source_description": "ESRS E1-4 GHG reduction target percentage",
         "target_regulation": "EU_TAXONOMY", "target_field": "ccm_reduction_pathway", "target_description": "Taxonomy CCM emission reduction pathway alignment",
         "mapping_type": "DERIVED", "confidence": 0.70, "conversion_notes": "CSRD target must be checked against Taxonomy thresholds", "bidirectional": False},
        {"source_regulation": "CBAM", "source_field": "emission_factor_direct", "source_description": "CBAM direct emission factor for production process",
         "target_regulation": "CSRD", "target_field": "E1_5_energy_intensity_by_activity", "target_description": "ESRS E1-5 energy-related emission intensity",
         "mapping_type": "DERIVED", "confidence": 0.65, "conversion_notes": "Requires energy consumption data to derive intensity", "bidirectional": False},
        {"source_regulation": "CBAM", "source_field": "emission_factor_electricity", "source_description": "CBAM electricity emission factor (grid/contract)",
         "target_regulation": "CSRD", "target_field": "E1_5_electricity_emission_factor", "target_description": "ESRS E1-5 electricity consumption emission factor",
         "mapping_type": "EXACT", "confidence": 0.85, "conversion_notes": "Both use tCO2e/MWh", "bidirectional": True},
        {"source_regulation": "CSRD", "source_field": "E1_1_transition_plan_targets", "source_description": "ESRS E1-1 Climate transition plan targets",
         "target_regulation": "EU_TAXONOMY", "target_field": "ccm_transition_plan_alignment", "target_description": "Taxonomy CCM transition plan alignment check",
         "mapping_type": "DERIVED", "confidence": 0.70, "conversion_notes": "Transition plan must meet Taxonomy minimum criteria", "bidirectional": False},
        {"source_regulation": "CBAM", "source_field": "production_process_route", "source_description": "CBAM production process route (BOF/EAF/etc)",
         "target_regulation": "EU_TAXONOMY", "target_field": "ccm_production_technology", "target_description": "Taxonomy CCM production technology classification",
         "mapping_type": "APPROXIMATE", "confidence": 0.80, "conversion_notes": "CBAM routes map to Taxonomy technology categories", "bidirectional": True},
        {"source_regulation": "CSRD", "source_field": "E1_6_biogenic_emissions", "source_description": "ESRS E1-6 Biogenic CO2 emissions",
         "target_regulation": "CBAM", "target_field": "biogenic_emissions_tco2", "target_description": "CBAM biogenic emissions (reported separately)",
         "mapping_type": "EXACT", "confidence": 0.90, "conversion_notes": "Both report biogenic separately from fossil", "bidirectional": True},
        {"source_regulation": "CSRD", "source_field": "E1_9_carbon_price_exposure", "source_description": "ESRS E1-9 Financial exposure to carbon pricing",
         "target_regulation": "CBAM", "target_field": "cbam_certificate_cost_eur", "target_description": "CBAM certificate cost in EUR",
         "mapping_type": "DERIVED", "confidence": 0.75, "conversion_notes": "CBAM cost is one component of total carbon price exposure", "bidirectional": False},
        {"source_regulation": "EU_TAXONOMY", "source_field": "ccm_scope1_absolute_emissions", "source_description": "Taxonomy CCM absolute Scope 1 emissions",
         "target_regulation": "CBAM", "target_field": "direct_emissions_tco2e", "target_description": "CBAM direct embedded emissions",
         "mapping_type": "EXACT", "confidence": 0.90, "conversion_notes": "Same metric scope and unit", "bidirectional": True},
        {"source_regulation": "CSRD", "source_field": "E1_6_ghg_removals", "source_description": "ESRS E1-6 GHG removals and carbon credits",
         "target_regulation": "EU_TAXONOMY", "target_field": "ccm_ghg_removals", "target_description": "Taxonomy CCM GHG removal activities",
         "mapping_type": "APPROXIMATE", "confidence": 0.70, "conversion_notes": "Taxonomy has stricter eligibility for removal activities", "bidirectional": True},
        {"source_regulation": "CBAM", "source_field": "monitoring_methodology", "source_description": "CBAM monitoring methodology reference",
         "target_regulation": "CSRD", "target_field": "E1_6_methodology_reference", "target_description": "ESRS E1-6 GHG accounting methodology",
         "mapping_type": "APPROXIMATE", "confidence": 0.65, "conversion_notes": "CBAM uses Implementing Regulation methodology, CSRD uses GHG Protocol", "bidirectional": True},
        {"source_regulation": "CBAM", "source_field": "goods_category_emissions", "source_description": "CBAM emissions by goods category (iron, steel, cement, etc)",
         "target_regulation": "CSRD", "target_field": "E1_6_emissions_by_sector", "target_description": "ESRS E1-6 emissions disaggregated by sector",
         "mapping_type": "APPROXIMATE", "confidence": 0.75, "conversion_notes": "CBAM categories map to CSRD sector disaggregation", "bidirectional": True},
        {"source_regulation": "EU_TAXONOMY", "source_field": "ccm_energy_efficiency_metric", "source_description": "Taxonomy CCM energy efficiency per unit output",
         "target_regulation": "CSRD", "target_field": "E1_5_energy_consumption_per_revenue", "target_description": "ESRS E1-5 energy intensity per net revenue",
         "mapping_type": "DERIVED", "confidence": 0.60, "conversion_notes": "Requires revenue normalization to convert", "bidirectional": False},
        {"source_regulation": "CSRD", "source_field": "E1_6_location_based_scope2", "source_description": "ESRS E1-6 Location-based Scope 2 emissions",
         "target_regulation": "CBAM", "target_field": "indirect_emissions_grid_default", "target_description": "CBAM indirect emissions using grid default factor",
         "mapping_type": "EXACT", "confidence": 0.85, "conversion_notes": "Both use location-based grid emission factors", "bidirectional": True},
        {"source_regulation": "CSRD", "source_field": "E1_6_market_based_scope2", "source_description": "ESRS E1-6 Market-based Scope 2 emissions",
         "target_regulation": "CBAM", "target_field": "indirect_emissions_contract", "target_description": "CBAM indirect emissions using contractual factor",
         "mapping_type": "EXACT", "confidence": 0.85, "conversion_notes": "Both use contract-based emission factors", "bidirectional": True},
        {"source_regulation": "CBAM", "source_field": "precursor_emissions", "source_description": "CBAM precursor material embedded emissions",
         "target_regulation": "CSRD", "target_field": "E1_6_scope3_cat1_precursor", "target_description": "ESRS E1-6 Scope 3 Cat 1 precursor component",
         "mapping_type": "APPROXIMATE", "confidence": 0.70, "conversion_notes": "CBAM precursor chain maps to upstream Scope 3", "bidirectional": True},
        {"source_regulation": "EU_TAXONOMY", "source_field": "ccm_benchmark_threshold", "source_description": "Taxonomy CCM sector benchmark threshold tCO2e/t",
         "target_regulation": "CBAM", "target_field": "eu_ets_benchmark_value", "target_description": "CBAM reference EU ETS benchmark value",
         "mapping_type": "APPROXIMATE", "confidence": 0.75, "conversion_notes": "Taxonomy benchmarks are derived from but may differ from ETS benchmarks", "bidirectional": True},
        {"source_regulation": "CSRD", "source_field": "E1_3_ghg_reduction_actions", "source_description": "ESRS E1-3 Actions and resources for GHG reduction",
         "target_regulation": "EU_TAXONOMY", "target_field": "ccm_capex_plan_alignment", "target_description": "Taxonomy CCM CapEx plan alignment with CCM",
         "mapping_type": "DERIVED", "confidence": 0.60, "conversion_notes": "GHG actions must be evaluated for Taxonomy-alignment", "bidirectional": False},
        {"source_regulation": "CBAM", "source_field": "carbon_price_paid_origin", "source_description": "CBAM carbon price paid in country of origin",
         "target_regulation": "CSRD", "target_field": "E1_9_carbon_pricing_mechanisms", "target_description": "ESRS E1-9 Carbon pricing mechanism exposures",
         "mapping_type": "EXACT", "confidence": 0.85, "conversion_notes": "Direct feed into carbon pricing exposure disclosure", "bidirectional": True},
        {"source_regulation": "CSRD", "source_field": "E1_7_ghg_absorption_sinks", "source_description": "ESRS E1-7 GHG removals by sinks",
         "target_regulation": "EU_TAXONOMY", "target_field": "ccm_carbon_sink_activities", "target_description": "Taxonomy CCM carbon sequestration activities",
         "mapping_type": "APPROXIMATE", "confidence": 0.65, "conversion_notes": "CSRD broader scope; Taxonomy has specific eligibility criteria", "bidirectional": True},
        {"source_regulation": "EU_TAXONOMY", "source_field": "ccm_renewable_energy_pct", "source_description": "Taxonomy CCM renewable energy share percentage",
         "target_regulation": "CSRD", "target_field": "E1_5_renewable_energy_share", "target_description": "ESRS E1-5 Share of renewable energy in energy mix",
         "mapping_type": "EXACT", "confidence": 0.90, "conversion_notes": "Same metric, same definition", "bidirectional": True},
    ],

    # ===================================================================
    # Category 2: SUPPLY CHAIN (~15 mappings)
    # EUDR traceability <-> CSRD S1/S2 due diligence
    # ===================================================================
    "SUPPLY_CHAIN": [
        {"source_regulation": "EUDR", "source_field": "supplier_geolocation", "source_description": "EUDR supplier production plot GPS coordinates",
         "target_regulation": "CSRD", "target_field": "S2_supply_chain_mapping", "target_description": "ESRS S2 supply chain geographical mapping",
         "mapping_type": "APPROXIMATE", "confidence": 0.70, "conversion_notes": "EUDR has plot-level detail; CSRD needs country/region level", "bidirectional": True},
        {"source_regulation": "EUDR", "source_field": "due_diligence_statement", "source_description": "EUDR due diligence statement submission",
         "target_regulation": "CSRD", "target_field": "S1_due_diligence_process", "target_description": "ESRS S1 Human rights due diligence process",
         "mapping_type": "APPROXIMATE", "confidence": 0.60, "conversion_notes": "EUDR focuses on deforestation; CSRD on broader human rights", "bidirectional": True},
        {"source_regulation": "EUDR", "source_field": "risk_assessment_result", "source_description": "EUDR country/region risk assessment result",
         "target_regulation": "CSRD", "target_field": "S2_risk_identification", "target_description": "ESRS S2 Supply chain risk identification",
         "mapping_type": "APPROXIMATE", "confidence": 0.65, "conversion_notes": "Different risk dimensions but common methodology", "bidirectional": True},
        {"source_regulation": "EUDR", "source_field": "supplier_certification_status", "source_description": "EUDR supplier certification (organic, FSC, RSPO)",
         "target_regulation": "CSRD", "target_field": "S2_supplier_standards_compliance", "target_description": "ESRS S2 Supplier compliance with standards",
         "mapping_type": "APPROXIMATE", "confidence": 0.75, "conversion_notes": "Same certificates relevant to both", "bidirectional": True},
        {"source_regulation": "CSRD", "source_field": "S1_worker_consultation_process", "source_description": "ESRS S1 Worker consultation and engagement",
         "target_regulation": "EUDR", "target_field": "stakeholder_consultation", "target_description": "EUDR stakeholder consultation for risk mitigation",
         "mapping_type": "DERIVED", "confidence": 0.55, "conversion_notes": "Overlapping stakeholder engagement requirements", "bidirectional": False},
        {"source_regulation": "EUDR", "source_field": "traceability_chain_complete", "source_description": "EUDR complete traceability to production source",
         "target_regulation": "CSRD", "target_field": "G1_supply_chain_transparency", "target_description": "ESRS G1 Supply chain transparency disclosure",
         "mapping_type": "APPROXIMATE", "confidence": 0.70, "conversion_notes": "EUDR traceability supports CSRD transparency requirements", "bidirectional": True},
        {"source_regulation": "CSRD", "source_field": "S2_supplier_incident_reporting", "source_description": "ESRS S2 Supplier incident and grievance reporting",
         "target_regulation": "EUDR", "target_field": "non_compliance_reporting", "target_description": "EUDR non-compliance incident reporting",
         "mapping_type": "APPROXIMATE", "confidence": 0.60, "conversion_notes": "Different triggers but common reporting infrastructure", "bidirectional": True},
        {"source_regulation": "EUDR", "source_field": "commodity_origin_country", "source_description": "EUDR commodity country of origin",
         "target_regulation": "CBAM", "target_field": "country_of_origin", "target_description": "CBAM country of origin for imported goods",
         "mapping_type": "EXACT", "confidence": 0.90, "conversion_notes": "Same ISO country code", "bidirectional": True},
        {"source_regulation": "EUDR", "source_field": "supplier_due_diligence_evidence", "source_description": "EUDR due diligence supporting evidence",
         "target_regulation": "CSRD", "target_field": "S2_due_diligence_evidence", "target_description": "ESRS S2 Due diligence evidence and documentation",
         "mapping_type": "APPROXIMATE", "confidence": 0.65, "conversion_notes": "Evidence types differ but documentation framework is shared", "bidirectional": True},
        {"source_regulation": "CSRD", "source_field": "S2_tier1_supplier_list", "source_description": "ESRS S2 Tier 1 supplier identification",
         "target_regulation": "EUDR", "target_field": "operator_supplier_registry", "target_description": "EUDR operator/trader supplier registry",
         "mapping_type": "APPROXIMATE", "confidence": 0.75, "conversion_notes": "EUDR may require deeper tier visibility", "bidirectional": True},
        {"source_regulation": "EUDR", "source_field": "product_quantity_kg", "source_description": "EUDR product quantity in kilograms",
         "target_regulation": "CBAM", "target_field": "imported_quantity_tonnes", "target_description": "CBAM imported quantity in tonnes",
         "mapping_type": "DERIVED", "confidence": 0.95, "conversion_notes": "Unit conversion: kg to tonnes (divide by 1000)", "bidirectional": True},
        {"source_regulation": "CSRD", "source_field": "S2_remediation_actions", "source_description": "ESRS S2 Remediation of negative supply chain impacts",
         "target_regulation": "EUDR", "target_field": "risk_mitigation_measures", "target_description": "EUDR risk mitigation and corrective actions",
         "mapping_type": "APPROXIMATE", "confidence": 0.60, "conversion_notes": "Different impact types but similar remediation framework", "bidirectional": True},
        {"source_regulation": "EUDR", "source_field": "competent_authority_submission", "source_description": "EUDR submission to competent authority",
         "target_regulation": "CSRD", "target_field": "regulatory_filing_status", "target_description": "CSRD regulatory filing and submission status",
         "mapping_type": "APPROXIMATE", "confidence": 0.50, "conversion_notes": "Different authorities, different submission formats", "bidirectional": False},
        {"source_regulation": "CSRD", "source_field": "S2_supply_chain_audit_results", "source_description": "ESRS S2 Supply chain audit findings",
         "target_regulation": "EUDR", "target_field": "third_party_verification", "target_description": "EUDR third-party verification results",
         "mapping_type": "APPROXIMATE", "confidence": 0.65, "conversion_notes": "Audit scope differs but methodology overlaps", "bidirectional": True},
        {"source_regulation": "EUDR", "source_field": "smallholder_assessment", "source_description": "EUDR smallholder farmer impact assessment",
         "target_regulation": "CSRD", "target_field": "S2_vulnerable_group_impact", "target_description": "ESRS S2 Impact on vulnerable groups in supply chain",
         "mapping_type": "DERIVED", "confidence": 0.55, "conversion_notes": "Smallholder assessment contributes to broader vulnerability analysis", "bidirectional": False},
    ],

    # ===================================================================
    # Category 3: ACTIVITY CLASSIFICATION (~15 mappings)
    # Taxonomy NACE <-> CBAM CN codes <-> CSRD sector
    # ===================================================================
    "ACTIVITY_CLASSIFICATION": [
        {"source_regulation": "EU_TAXONOMY", "source_field": "nace_code_primary", "source_description": "Taxonomy primary NACE Rev.2 activity code",
         "target_regulation": "CSRD", "target_field": "sector_classification_nace", "target_description": "CSRD sector classification via NACE code",
         "mapping_type": "EXACT", "confidence": 0.95, "conversion_notes": "Both use NACE Rev.2 classification", "bidirectional": True},
        {"source_regulation": "CBAM", "source_field": "cn_code_8digit", "source_description": "CBAM Combined Nomenclature 8-digit product code",
         "target_regulation": "EU_TAXONOMY", "target_field": "nace_code_primary", "target_description": "Taxonomy NACE code for the economic activity",
         "mapping_type": "DERIVED", "confidence": 0.75, "conversion_notes": "CN codes map to NACE via Eurostat correspondence tables", "bidirectional": True},
        {"source_regulation": "CBAM", "source_field": "goods_category", "source_description": "CBAM goods category (iron_steel, cement, aluminium, etc)",
         "target_regulation": "CSRD", "target_field": "sector_high_impact", "target_description": "CSRD high-impact sector classification",
         "mapping_type": "APPROXIMATE", "confidence": 0.80, "conversion_notes": "CBAM goods categories map to CSRD high-impact sectors", "bidirectional": True},
        {"source_regulation": "EU_TAXONOMY", "source_field": "activity_3_9_iron_steel", "source_description": "Taxonomy Activity 3.9 Iron and steel manufacturing",
         "target_regulation": "CBAM", "target_field": "goods_category_iron_steel", "target_description": "CBAM iron and steel goods category",
         "mapping_type": "EXACT", "confidence": 0.90, "conversion_notes": "Direct sector correspondence", "bidirectional": True},
        {"source_regulation": "EU_TAXONOMY", "source_field": "activity_3_7_cement", "source_description": "Taxonomy Activity 3.7 Cement manufacturing",
         "target_regulation": "CBAM", "target_field": "goods_category_cement", "target_description": "CBAM cement goods category",
         "mapping_type": "EXACT", "confidence": 0.90, "conversion_notes": "Direct sector correspondence", "bidirectional": True},
        {"source_regulation": "EU_TAXONOMY", "source_field": "activity_3_8_aluminium", "source_description": "Taxonomy Activity 3.8 Aluminium manufacturing",
         "target_regulation": "CBAM", "target_field": "goods_category_aluminium", "target_description": "CBAM aluminium goods category",
         "mapping_type": "EXACT", "confidence": 0.90, "conversion_notes": "Direct sector correspondence", "bidirectional": True},
        {"source_regulation": "EU_TAXONOMY", "source_field": "activity_3_13_hydrogen", "source_description": "Taxonomy Activity 3.13 Hydrogen manufacturing",
         "target_regulation": "CBAM", "target_field": "goods_category_hydrogen", "target_description": "CBAM hydrogen goods category",
         "mapping_type": "EXACT", "confidence": 0.90, "conversion_notes": "Direct sector correspondence", "bidirectional": True},
        {"source_regulation": "EU_TAXONOMY", "source_field": "activity_3_12_fertilizers", "source_description": "Taxonomy Activity 3.12 Fertilizer manufacturing",
         "target_regulation": "CBAM", "target_field": "goods_category_fertilizers", "target_description": "CBAM fertilizers goods category",
         "mapping_type": "EXACT", "confidence": 0.90, "conversion_notes": "Direct sector correspondence", "bidirectional": True},
        {"source_regulation": "CBAM", "source_field": "cn_code_chapter72", "source_description": "CBAM CN Chapter 72 (iron and steel products)",
         "target_regulation": "EU_TAXONOMY", "target_field": "nace_c241_iron_steel", "target_description": "Taxonomy NACE C24.1 Iron and steel",
         "mapping_type": "DERIVED", "confidence": 0.85, "conversion_notes": "CN 72xx maps to NACE C24.1 via concordance", "bidirectional": True},
        {"source_regulation": "CBAM", "source_field": "cn_code_chapter73", "source_description": "CBAM CN Chapter 73 (articles of iron or steel)",
         "target_regulation": "EU_TAXONOMY", "target_field": "nace_c251_metal_products", "target_description": "Taxonomy NACE C25.1 Structural metal products",
         "mapping_type": "DERIVED", "confidence": 0.75, "conversion_notes": "CN 73xx may map to multiple NACE codes", "bidirectional": True},
        {"source_regulation": "CBAM", "source_field": "cn_code_chapter76", "source_description": "CBAM CN Chapter 76 (aluminium and articles thereof)",
         "target_regulation": "EU_TAXONOMY", "target_field": "nace_c242_aluminium", "target_description": "Taxonomy NACE C24.2 Aluminium production",
         "mapping_type": "DERIVED", "confidence": 0.85, "conversion_notes": "CN 76xx maps to NACE C24.2 via concordance", "bidirectional": True},
        {"source_regulation": "CSRD", "source_field": "sector_nace_revenue_breakdown", "source_description": "CSRD revenue breakdown by NACE sector",
         "target_regulation": "EU_TAXONOMY", "target_field": "turnover_kpi_by_activity", "target_description": "Taxonomy turnover KPI by eligible activity",
         "mapping_type": "APPROXIMATE", "confidence": 0.80, "conversion_notes": "Same NACE basis but Taxonomy only counts eligible revenue", "bidirectional": True},
        {"source_regulation": "EUDR", "source_field": "commodity_type_code", "source_description": "EUDR regulated commodity type (palm, soy, cocoa, etc)",
         "target_regulation": "CSRD", "target_field": "sector_classification_commodity", "target_description": "CSRD commodity sector classification",
         "mapping_type": "APPROXIMATE", "confidence": 0.70, "conversion_notes": "EUDR commodity codes map to agricultural NACE sectors", "bidirectional": True},
        {"source_regulation": "EU_TAXONOMY", "source_field": "activity_eligibility_flag", "source_description": "Taxonomy eligible activity flag (yes/no)",
         "target_regulation": "CSRD", "target_field": "taxonomy_eligible_revenue_flag", "target_description": "CSRD Taxonomy eligibility in Article 8 disclosure",
         "mapping_type": "EXACT", "confidence": 0.95, "conversion_notes": "Direct passthrough from Taxonomy assessment", "bidirectional": True},
        {"source_regulation": "EUDR", "source_field": "hs_code_product", "source_description": "EUDR Harmonized System product code",
         "target_regulation": "CBAM", "target_field": "cn_code_8digit", "target_description": "CBAM Combined Nomenclature code",
         "mapping_type": "DERIVED", "confidence": 0.80, "conversion_notes": "HS 6-digit extends to CN 8-digit via EU taric", "bidirectional": True},
    ],

    # ===================================================================
    # Category 4: FINANCIAL DATA (~15 mappings)
    # Taxonomy KPIs <-> CSRD financial metrics <-> CBAM costs
    # ===================================================================
    "FINANCIAL_DATA": [
        {"source_regulation": "EU_TAXONOMY", "source_field": "turnover_kpi_pct", "source_description": "Taxonomy Article 8 turnover KPI percentage",
         "target_regulation": "CSRD", "target_field": "taxonomy_turnover_disclosure", "target_description": "CSRD Taxonomy-aligned turnover percentage",
         "mapping_type": "EXACT", "confidence": 0.95, "conversion_notes": "Direct Article 8 KPI disclosure in CSRD", "bidirectional": True},
        {"source_regulation": "EU_TAXONOMY", "source_field": "capex_kpi_pct", "source_description": "Taxonomy Article 8 CapEx KPI percentage",
         "target_regulation": "CSRD", "target_field": "taxonomy_capex_disclosure", "target_description": "CSRD Taxonomy-aligned CapEx percentage",
         "mapping_type": "EXACT", "confidence": 0.95, "conversion_notes": "Direct Article 8 KPI disclosure in CSRD", "bidirectional": True},
        {"source_regulation": "EU_TAXONOMY", "source_field": "opex_kpi_pct", "source_description": "Taxonomy Article 8 OpEx KPI percentage",
         "target_regulation": "CSRD", "target_field": "taxonomy_opex_disclosure", "target_description": "CSRD Taxonomy-aligned OpEx percentage",
         "mapping_type": "EXACT", "confidence": 0.95, "conversion_notes": "Direct Article 8 KPI disclosure in CSRD", "bidirectional": True},
        {"source_regulation": "CBAM", "source_field": "cbam_certificate_cost_total", "source_description": "CBAM total certificate purchase cost EUR",
         "target_regulation": "CSRD", "target_field": "E1_9_carbon_pricing_cost", "target_description": "ESRS E1-9 Carbon pricing financial impact",
         "mapping_type": "EXACT", "confidence": 0.90, "conversion_notes": "CBAM cost directly feeds CSRD carbon pricing disclosure", "bidirectional": True},
        {"source_regulation": "CBAM", "source_field": "cbam_cost_per_tonne_product", "source_description": "CBAM cost per tonne of imported product",
         "target_regulation": "EU_TAXONOMY", "target_field": "ccm_carbon_cost_internalized", "target_description": "Taxonomy internalized carbon cost metric",
         "mapping_type": "DERIVED", "confidence": 0.70, "conversion_notes": "CBAM cost represents externalized carbon cost now internalized", "bidirectional": False},
        {"source_regulation": "CSRD", "source_field": "E1_9_stranded_asset_exposure", "source_description": "ESRS E1-9 Stranded asset exposure from transition",
         "target_regulation": "EU_TAXONOMY", "target_field": "ccm_transition_risk_exposure", "target_description": "Taxonomy transition risk financial exposure",
         "mapping_type": "APPROXIMATE", "confidence": 0.70, "conversion_notes": "CSRD broader scope; Taxonomy focuses on activity-level risk", "bidirectional": True},
        {"source_regulation": "EU_TAXONOMY", "source_field": "gar_green_asset_ratio", "source_description": "Taxonomy Green Asset Ratio (financial institutions)",
         "target_regulation": "CSRD", "target_field": "taxonomy_gar_disclosure", "target_description": "CSRD Green Asset Ratio in financial sector disclosures",
         "mapping_type": "EXACT", "confidence": 0.95, "conversion_notes": "GAR directly reported in CSRD for financial sector", "bidirectional": True},
        {"source_regulation": "CBAM", "source_field": "financial_guarantee_amount", "source_description": "CBAM financial guarantee amount posted",
         "target_regulation": "CSRD", "target_field": "contingent_liabilities_cbam", "target_description": "CSRD contingent liabilities from CBAM obligations",
         "mapping_type": "DERIVED", "confidence": 0.65, "conversion_notes": "Guarantee relates to contingent liability disclosure", "bidirectional": False},
        {"source_regulation": "CSRD", "source_field": "revenue_by_geography", "source_description": "CSRD revenue disaggregated by geography",
         "target_regulation": "CBAM", "target_field": "import_value_by_country", "target_description": "CBAM import value by country of origin",
         "mapping_type": "APPROXIMATE", "confidence": 0.55, "conversion_notes": "Different financial perspectives but geographic overlap", "bidirectional": True},
        {"source_regulation": "EU_TAXONOMY", "source_field": "capex_plan_taxonomy_aligned", "source_description": "Taxonomy CapEx plan for alignment improvement",
         "target_regulation": "CSRD", "target_field": "E1_3_investment_in_decarbonization", "target_description": "ESRS E1-3 Investment in decarbonization actions",
         "mapping_type": "APPROXIMATE", "confidence": 0.80, "conversion_notes": "Taxonomy CapEx plan supports CSRD decarbonization investment disclosure", "bidirectional": True},
        {"source_regulation": "CBAM", "source_field": "carbon_price_paid_in_origin", "source_description": "CBAM carbon price paid in country of origin",
         "target_regulation": "EU_TAXONOMY", "target_field": "ccm_carbon_pricing_coverage", "target_description": "Taxonomy carbon pricing coverage assessment",
         "mapping_type": "DERIVED", "confidence": 0.65, "conversion_notes": "Carbon price coverage relevant to transition risk assessment", "bidirectional": False},
        {"source_regulation": "CSRD", "source_field": "E1_9_opportunity_revenue", "source_description": "ESRS E1-9 Revenue from climate opportunities",
         "target_regulation": "EU_TAXONOMY", "target_field": "turnover_kpi_ccm_aligned", "target_description": "Taxonomy CCM-aligned turnover KPI",
         "mapping_type": "APPROXIMATE", "confidence": 0.70, "conversion_notes": "Climate opportunity revenue may overlap with Taxonomy-aligned turnover", "bidirectional": True},
        {"source_regulation": "EUDR", "source_field": "due_diligence_cost_total", "source_description": "EUDR total due diligence compliance cost",
         "target_regulation": "CSRD", "target_field": "compliance_cost_environmental", "target_description": "CSRD environmental compliance cost disclosure",
         "mapping_type": "APPROXIMATE", "confidence": 0.60, "conversion_notes": "EUDR compliance cost is subset of total environmental compliance spend", "bidirectional": False},
        {"source_regulation": "EU_TAXONOMY", "source_field": "nuclear_gas_complementary_kpi", "source_description": "Taxonomy nuclear/gas complementary disclosure KPIs",
         "target_regulation": "CSRD", "target_field": "taxonomy_complementary_disclosure", "target_description": "CSRD nuclear/gas complementary disclosures",
         "mapping_type": "EXACT", "confidence": 0.90, "conversion_notes": "Direct passthrough of Delegated Act requirements", "bidirectional": True},
        {"source_regulation": "CBAM", "source_field": "deminimis_threshold_check", "source_description": "CBAM de minimis threshold value check (EUR 150 / 150kg)",
         "target_regulation": "CSRD", "target_field": "materiality_financial_threshold", "target_description": "CSRD financial materiality threshold assessment",
         "mapping_type": "DERIVED", "confidence": 0.50, "conversion_notes": "Different threshold concepts but both assess materiality/significance", "bidirectional": False},
    ],

    # ===================================================================
    # Category 5: CLIMATE RISK (~10 mappings)
    # Taxonomy CCA <-> CSRD E1 risk <-> CBAM carbon price
    # ===================================================================
    "CLIMATE_RISK": [
        {"source_regulation": "EU_TAXONOMY", "source_field": "cca_physical_risk_assessment", "source_description": "Taxonomy CCA physical climate risk assessment",
         "target_regulation": "CSRD", "target_field": "E1_9_physical_risk_exposure", "target_description": "ESRS E1-9 Physical climate risk exposure",
         "mapping_type": "APPROXIMATE", "confidence": 0.80, "conversion_notes": "Taxonomy CCA has specific asset-level criteria; CSRD is entity-level", "bidirectional": True},
        {"source_regulation": "EU_TAXONOMY", "source_field": "cca_adaptation_plan", "source_description": "Taxonomy CCA climate adaptation plan",
         "target_regulation": "CSRD", "target_field": "E1_2_adaptation_actions", "target_description": "ESRS E1-2 Climate change adaptation policies and actions",
         "mapping_type": "APPROXIMATE", "confidence": 0.75, "conversion_notes": "Taxonomy has prescriptive criteria; CSRD is disclosure-oriented", "bidirectional": True},
        {"source_regulation": "CBAM", "source_field": "cbam_price_per_tco2e_current", "source_description": "CBAM current price per tCO2e for certificates",
         "target_regulation": "CSRD", "target_field": "E1_9_transition_risk_carbon_price", "target_description": "ESRS E1-9 Transition risk from carbon price increases",
         "mapping_type": "DERIVED", "confidence": 0.75, "conversion_notes": "CBAM price feeds into carbon price risk scenario analysis", "bidirectional": False},
        {"source_regulation": "CSRD", "source_field": "E1_9_scenario_analysis_results", "source_description": "ESRS E1-9 Climate scenario analysis results",
         "target_regulation": "EU_TAXONOMY", "target_field": "cca_scenario_analysis", "target_description": "Taxonomy CCA climate scenario analysis requirements",
         "mapping_type": "APPROXIMATE", "confidence": 0.70, "conversion_notes": "Both use IPCC-based scenarios but with different granularity", "bidirectional": True},
        {"source_regulation": "EU_TAXONOMY", "source_field": "cca_vulnerability_assessment", "source_description": "Taxonomy CCA climate vulnerability assessment output",
         "target_regulation": "CSRD", "target_field": "E1_1_climate_vulnerability_mapping", "target_description": "ESRS E1-1 Climate-related vulnerability identification",
         "mapping_type": "APPROXIMATE", "confidence": 0.75, "conversion_notes": "Taxonomy is asset-level; CSRD covers entire value chain", "bidirectional": True},
        {"source_regulation": "CBAM", "source_field": "carbon_leakage_sector_flag", "source_description": "CBAM carbon leakage sector classification",
         "target_regulation": "CSRD", "target_field": "E1_9_carbon_leakage_risk", "target_description": "ESRS E1-9 Carbon leakage risk assessment",
         "mapping_type": "EXACT", "confidence": 0.85, "conversion_notes": "Same EU carbon leakage list used by both", "bidirectional": True},
        {"source_regulation": "CSRD", "source_field": "E1_9_financial_impact_physical", "source_description": "ESRS E1-9 Financial impact from physical risks",
         "target_regulation": "EU_TAXONOMY", "target_field": "cca_financial_impact_assessment", "target_description": "Taxonomy CCA financial impact of climate hazards",
         "mapping_type": "APPROXIMATE", "confidence": 0.70, "conversion_notes": "Different scope but overlapping methodology", "bidirectional": True},
        {"source_regulation": "EU_TAXONOMY", "source_field": "cca_adaptation_solution_revenue", "source_description": "Taxonomy CCA revenue from adaptation solutions",
         "target_regulation": "CSRD", "target_field": "E1_9_adaptation_opportunity_revenue", "target_description": "ESRS E1-9 Revenue opportunity from adaptation",
         "mapping_type": "APPROXIMATE", "confidence": 0.75, "conversion_notes": "Taxonomy only counts eligible revenue; CSRD is broader", "bidirectional": True},
        {"source_regulation": "CBAM", "source_field": "ets_free_allocation_phase_out", "source_description": "CBAM ETS free allocation phase-out schedule impact",
         "target_regulation": "CSRD", "target_field": "E1_9_regulatory_transition_risk", "target_description": "ESRS E1-9 Regulatory transition risk from policy changes",
         "mapping_type": "DERIVED", "confidence": 0.70, "conversion_notes": "ETS phase-out creates transition risk for CSRD reporting", "bidirectional": False},
        {"source_regulation": "EU_TAXONOMY", "source_field": "cca_hazard_classification", "source_description": "Taxonomy CCA climate hazard type classification",
         "target_regulation": "CSRD", "target_field": "E1_9_hazard_type_mapping", "target_description": "ESRS E1-9 Physical climate hazard classification",
         "mapping_type": "EXACT", "confidence": 0.85, "conversion_notes": "Both use IPCC hazard classification taxonomy", "bidirectional": True},
    ],

    # ===================================================================
    # Category 6: WATER AND POLLUTION (~10 mappings)
    # Taxonomy WTR/PPC <-> CSRD E2/E3
    # ===================================================================
    "WATER_POLLUTION": [
        {"source_regulation": "EU_TAXONOMY", "source_field": "wtr_water_consumption_m3", "source_description": "Taxonomy WTR total water consumption in cubic metres",
         "target_regulation": "CSRD", "target_field": "E3_4_water_consumption_total", "target_description": "ESRS E3-4 Total water consumption",
         "mapping_type": "EXACT", "confidence": 0.90, "conversion_notes": "Same metric, same unit (m3)", "bidirectional": True},
        {"source_regulation": "EU_TAXONOMY", "source_field": "wtr_water_stress_area_flag", "source_description": "Taxonomy WTR water-stressed area identification flag",
         "target_regulation": "CSRD", "target_field": "E3_4_water_stress_exposure", "target_description": "ESRS E3-4 Operations in water-stressed areas",
         "mapping_type": "EXACT", "confidence": 0.85, "conversion_notes": "Both use WRI Aqueduct water stress classification", "bidirectional": True},
        {"source_regulation": "EU_TAXONOMY", "source_field": "ppc_pollutant_emissions_air", "source_description": "Taxonomy PPC air pollutant emissions (SOx, NOx, PM)",
         "target_regulation": "CSRD", "target_field": "E2_4_pollutant_emissions_air", "target_description": "ESRS E2-4 Air pollution emissions",
         "mapping_type": "EXACT", "confidence": 0.90, "conversion_notes": "Same pollutant categories and units", "bidirectional": True},
        {"source_regulation": "EU_TAXONOMY", "source_field": "ppc_pollutant_emissions_water", "source_description": "Taxonomy PPC water pollutant discharges",
         "target_regulation": "CSRD", "target_field": "E2_4_pollutant_emissions_water", "target_description": "ESRS E2-4 Water pollution discharges",
         "mapping_type": "EXACT", "confidence": 0.90, "conversion_notes": "Same pollutant categories and measurement basis", "bidirectional": True},
        {"source_regulation": "EU_TAXONOMY", "source_field": "wtr_water_recycling_pct", "source_description": "Taxonomy WTR water recycling/reuse percentage",
         "target_regulation": "CSRD", "target_field": "E3_4_water_recycled_pct", "target_description": "ESRS E3-4 Water recycled and reused percentage",
         "mapping_type": "EXACT", "confidence": 0.90, "conversion_notes": "Same metric definition", "bidirectional": True},
        {"source_regulation": "CSRD", "source_field": "E2_1_pollution_prevention_policy", "source_description": "ESRS E2-1 Pollution prevention policies",
         "target_regulation": "EU_TAXONOMY", "target_field": "ppc_pollution_prevention_plan", "target_description": "Taxonomy PPC pollution prevention and control plan",
         "mapping_type": "APPROXIMATE", "confidence": 0.70, "conversion_notes": "CSRD disclosure-oriented; Taxonomy has specific technical criteria", "bidirectional": True},
        {"source_regulation": "EU_TAXONOMY", "source_field": "ppc_substances_of_concern", "source_description": "Taxonomy PPC substances of concern management",
         "target_regulation": "CSRD", "target_field": "E2_5_substances_of_concern", "target_description": "ESRS E2-5 Substances of concern production and use",
         "mapping_type": "EXACT", "confidence": 0.85, "conversion_notes": "Both reference REACH/CLP regulation substance lists", "bidirectional": True},
        {"source_regulation": "CSRD", "source_field": "E3_3_water_withdrawal_source", "source_description": "ESRS E3-3 Water withdrawal by source type",
         "target_regulation": "EU_TAXONOMY", "target_field": "wtr_water_withdrawal_source", "target_description": "Taxonomy WTR water withdrawal source breakdown",
         "mapping_type": "EXACT", "confidence": 0.85, "conversion_notes": "Same source categories (surface, ground, third-party, seawater)", "bidirectional": True},
        {"source_regulation": "EU_TAXONOMY", "source_field": "ppc_waste_water_treatment_level", "source_description": "Taxonomy PPC wastewater treatment level achieved",
         "target_regulation": "CSRD", "target_field": "E3_4_wastewater_treatment_quality", "target_description": "ESRS E3-4 Wastewater treatment quality metrics",
         "mapping_type": "APPROXIMATE", "confidence": 0.75, "conversion_notes": "Taxonomy specifies BAT; CSRD is disclosure of treatment level", "bidirectional": True},
        {"source_regulation": "CSRD", "source_field": "E2_6_pollution_incident_count", "source_description": "ESRS E2-6 Pollution incidents and fines",
         "target_regulation": "EU_TAXONOMY", "target_field": "ppc_dnsh_pollution_compliance", "target_description": "Taxonomy PPC DNSH pollution compliance check",
         "mapping_type": "DERIVED", "confidence": 0.65, "conversion_notes": "Pollution incidents may trigger DNSH non-compliance assessment", "bidirectional": False},
    ],

    # ===================================================================
    # Category 7: BIODIVERSITY (~10 mappings)
    # Taxonomy BIO <-> EUDR deforestation <-> CSRD E4
    # ===================================================================
    "BIODIVERSITY": [
        {"source_regulation": "EU_TAXONOMY", "source_field": "bio_biodiversity_impact_assessment", "source_description": "Taxonomy BIO biodiversity impact assessment (EIA/SEA)",
         "target_regulation": "CSRD", "target_field": "E4_4_biodiversity_impact_assessment", "target_description": "ESRS E4-4 Biodiversity impact assessment results",
         "mapping_type": "APPROXIMATE", "confidence": 0.80, "conversion_notes": "Taxonomy has site-level criteria; CSRD is value-chain scope", "bidirectional": True},
        {"source_regulation": "EUDR", "source_field": "deforestation_free_status", "source_description": "EUDR deforestation-free status declaration",
         "target_regulation": "CSRD", "target_field": "E4_5_deforestation_commitment", "target_description": "ESRS E4-5 Deforestation and land degradation commitments",
         "mapping_type": "APPROXIMATE", "confidence": 0.80, "conversion_notes": "EUDR compliance supports CSRD deforestation disclosure", "bidirectional": True},
        {"source_regulation": "EUDR", "source_field": "deforestation_free_status", "source_description": "EUDR deforestation-free status declaration",
         "target_regulation": "EU_TAXONOMY", "target_field": "bio_dnsh_deforestation_check", "target_description": "Taxonomy BIO DNSH deforestation screening",
         "mapping_type": "EXACT", "confidence": 0.85, "conversion_notes": "EUDR compliance directly satisfies Taxonomy DNSH BIO deforestation criterion", "bidirectional": True},
        {"source_regulation": "EU_TAXONOMY", "source_field": "bio_protected_area_proximity", "source_description": "Taxonomy BIO proximity to protected/Natura 2000 areas",
         "target_regulation": "CSRD", "target_field": "E4_4_operations_near_protected_areas", "target_description": "ESRS E4-4 Operations in/near biodiversity-sensitive areas",
         "mapping_type": "EXACT", "confidence": 0.85, "conversion_notes": "Both reference Natura 2000 and IUCN protected area databases", "bidirectional": True},
        {"source_regulation": "EUDR", "source_field": "forest_degradation_assessment", "source_description": "EUDR forest degradation risk assessment result",
         "target_regulation": "CSRD", "target_field": "E4_6_land_degradation_impact", "target_description": "ESRS E4-6 Land degradation and soil sealing impacts",
         "mapping_type": "APPROXIMATE", "confidence": 0.70, "conversion_notes": "EUDR forest-specific; CSRD covers all land degradation types", "bidirectional": True},
        {"source_regulation": "EU_TAXONOMY", "source_field": "bio_ecosystem_restoration_plan", "source_description": "Taxonomy BIO ecosystem restoration and offsetting plan",
         "target_regulation": "CSRD", "target_field": "E4_3_biodiversity_restoration_targets", "target_description": "ESRS E4-3 Biodiversity restoration and offset targets",
         "mapping_type": "APPROXIMATE", "confidence": 0.75, "conversion_notes": "Taxonomy prescribes specific restoration criteria", "bidirectional": True},
        {"source_regulation": "EUDR", "source_field": "indigenous_peoples_rights_check", "source_description": "EUDR indigenous peoples and local community rights compliance",
         "target_regulation": "CSRD", "target_field": "S3_indigenous_community_impact", "target_description": "ESRS S3 Affected communities - indigenous peoples impacts",
         "mapping_type": "APPROXIMATE", "confidence": 0.70, "conversion_notes": "EUDR is commodity-specific; CSRD covers all operations", "bidirectional": True},
        {"source_regulation": "EU_TAXONOMY", "source_field": "bio_sustainable_forestry_criteria", "source_description": "Taxonomy BIO sustainable forestry management criteria",
         "target_regulation": "EUDR", "target_field": "sustainable_production_evidence", "target_description": "EUDR evidence of sustainable production practices",
         "mapping_type": "APPROXIMATE", "confidence": 0.75, "conversion_notes": "Taxonomy forestry criteria align with EUDR sustainability evidence", "bidirectional": True},
        {"source_regulation": "CSRD", "source_field": "E4_1_biodiversity_policy", "source_description": "ESRS E4-1 Biodiversity and ecosystems policy",
         "target_regulation": "EU_TAXONOMY", "target_field": "bio_biodiversity_management_plan", "target_description": "Taxonomy BIO biodiversity management plan requirement",
         "mapping_type": "APPROXIMATE", "confidence": 0.65, "conversion_notes": "CSRD disclosure vs Taxonomy prescriptive plan criteria", "bidirectional": True},
        {"source_regulation": "EUDR", "source_field": "satellite_monitoring_data", "source_description": "EUDR satellite/remote sensing monitoring data",
         "target_regulation": "CSRD", "target_field": "E4_4_monitoring_methodology", "target_description": "ESRS E4-4 Biodiversity monitoring methodology and data",
         "mapping_type": "DERIVED", "confidence": 0.60, "conversion_notes": "EUDR satellite data can support CSRD biodiversity monitoring disclosure", "bidirectional": False},
    ],
}


# ---------------------------------------------------------------------------
# Conversion Functions Registry
# ---------------------------------------------------------------------------

CONVERSION_FUNCTIONS: Dict[str, Callable] = {
    "kg_to_tonnes": lambda v: v / 1000.0 if isinstance(v, (int, float)) else v,
    "tonnes_to_kg": lambda v: v * 1000.0 if isinstance(v, (int, float)) else v,
    "pct_to_fraction": lambda v: v / 100.0 if isinstance(v, (int, float)) else v,
    "fraction_to_pct": lambda v: v * 100.0 if isinstance(v, (int, float)) else v,
    "mwh_to_gj": lambda v: v * 3.6 if isinstance(v, (int, float)) else v,
    "gj_to_mwh": lambda v: v / 3.6 if isinstance(v, (int, float)) else v,
    "identity": lambda v: v,
}


# ---------------------------------------------------------------------------
# CrossFrameworkDataMapperEngine
# ---------------------------------------------------------------------------


class CrossFrameworkDataMapperEngine:
    """
    Cross-framework data field mapping engine for EU climate regulations.

    Maps data fields between CSRD, CBAM, EUDR, and EU Taxonomy with
    confidence scoring, bidirectional search, and multi-hop path finding.
    Contains 100+ predefined field mappings organized by 7 thematic
    categories.

    Attributes:
        config: Engine configuration.
        _mappings_by_key: Indexed mappings for fast lookup.
        _all_mappings: Flat list of all FieldMapping objects.

    Example:
        >>> engine = CrossFrameworkDataMapperEngine()
        >>> result = engine.map_field("CSRD", "E1_6_scope1_ghg_emissions", "CBAM", 42000.0)
        >>> assert result.confidence >= 0.9
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize CrossFrameworkDataMapperEngine.

        Args:
            config: Optional configuration dictionary or CrossFrameworkDataMapperConfig.
        """
        if config and isinstance(config, dict):
            self.config = CrossFrameworkDataMapperConfig(**config)
        elif config and isinstance(config, CrossFrameworkDataMapperConfig):
            self.config = config
        else:
            self.config = CrossFrameworkDataMapperConfig()

        self._all_mappings: List[FieldMapping] = []
        self._mappings_by_key: Dict[str, List[FieldMapping]] = {}
        self._build_mapping_index()
        logger.info(
            "CrossFrameworkDataMapperEngine initialized (v%s): %d mappings loaded",
            _MODULE_VERSION, len(self._all_mappings),
        )

    def _build_mapping_index(self) -> None:
        """Build indexed lookup structures from the raw mapping database."""
        for category, raw_mappings in CROSS_FRAMEWORK_MAPPINGS.items():
            for raw in raw_mappings:
                mapping = FieldMapping(
                    source_regulation=raw["source_regulation"],
                    source_field=raw["source_field"],
                    source_description=raw.get("source_description", ""),
                    target_regulation=raw["target_regulation"],
                    target_field=raw["target_field"],
                    target_description=raw.get("target_description", ""),
                    mapping_type=raw.get("mapping_type", "APPROXIMATE"),
                    confidence=raw.get("confidence", 0.5),
                    category=category,
                    conversion_notes=raw.get("conversion_notes", ""),
                    bidirectional=raw.get("bidirectional", True),
                )
                self._all_mappings.append(mapping)

                fwd_key = f"{mapping.source_regulation}::{mapping.source_field}::{mapping.target_regulation}"
                self._mappings_by_key.setdefault(fwd_key, []).append(mapping)

                if mapping.bidirectional:
                    rev_key = f"{mapping.target_regulation}::{mapping.target_field}::{mapping.source_regulation}"
                    rev_mapping = FieldMapping(
                        source_regulation=mapping.target_regulation,
                        source_field=mapping.target_field,
                        source_description=mapping.target_description,
                        target_regulation=mapping.source_regulation,
                        target_field=mapping.source_field,
                        target_description=mapping.source_description,
                        mapping_type=mapping.mapping_type,
                        confidence=mapping.confidence,
                        category=category,
                        conversion_notes=f"Reverse: {mapping.conversion_notes}",
                        bidirectional=True,
                    )
                    self._mappings_by_key.setdefault(rev_key, []).append(rev_mapping)

    # -------------------------------------------------------------------
    # map_field
    # -------------------------------------------------------------------

    def map_field(
        self,
        source_regulation: str,
        source_field: str,
        target_regulation: str,
        source_value: Any = None,
        conversion_function: Optional[str] = None,
    ) -> MappingResult:
        """Map a single field from source regulation to target regulation.

        Args:
            source_regulation: Source regulation name.
            source_field: Source field identifier.
            target_regulation: Target regulation name.
            source_value: Optional value to map/convert.
            conversion_function: Optional conversion function name from registry.

        Returns:
            MappingResult with mapped field and confidence.
        """
        start_time = datetime.now(timezone.utc)
        key = f"{source_regulation}::{source_field}::{target_regulation}"
        candidates = self._mappings_by_key.get(key, [])

        candidates = [
            c for c in candidates
            if c.confidence >= self.config.min_confidence
        ]

        if not self.config.enable_derived_mappings:
            candidates = [c for c in candidates if c.mapping_type != "DERIVED"]

        if not candidates:
            logger.warning(
                "No mapping found: %s.%s -> %s",
                source_regulation, source_field, target_regulation,
            )
            result = MappingResult(
                source_regulation=source_regulation,
                source_field=source_field,
                source_value=source_value,
                target_regulation=target_regulation,
                target_field="",
                mapped_value=None,
                mapping_type="NONE",
                confidence=0.0,
                conversion_applied=False,
                conversion_notes="No mapping found",
            )
            result.provenance_hash = _compute_hash(result)
            return result

        best = max(candidates, key=lambda m: m.confidence)

        mapped_value = source_value
        conversion_applied = False
        notes = best.conversion_notes

        if conversion_function and conversion_function in CONVERSION_FUNCTIONS:
            mapped_value = CONVERSION_FUNCTIONS[conversion_function](source_value)
            conversion_applied = True
            notes = f"Applied {conversion_function}: {notes}"

        result = MappingResult(
            source_regulation=source_regulation,
            source_field=source_field,
            source_value=source_value,
            target_regulation=target_regulation,
            target_field=best.target_field,
            mapped_value=mapped_value,
            mapping_type=best.mapping_type,
            confidence=best.confidence,
            conversion_applied=conversion_applied,
            conversion_notes=notes,
        )
        result.provenance_hash = _compute_hash(result)

        elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        logger.info(
            "Mapped %s.%s -> %s.%s (confidence=%.2f, %.1fms)",
            source_regulation, source_field,
            target_regulation, best.target_field,
            best.confidence, elapsed_ms,
        )
        return result

    # -------------------------------------------------------------------
    # map_batch
    # -------------------------------------------------------------------

    def map_batch(
        self,
        source_regulation: str,
        fields: Dict[str, Any],
        target_regulation: str,
    ) -> BatchMappingResult:
        """Map multiple fields from source to target regulation in batch.

        Args:
            source_regulation: Source regulation name.
            fields: Dict of field_name -> value to map.
            target_regulation: Target regulation name.

        Returns:
            BatchMappingResult with all mapping results.
        """
        start_time = datetime.now(timezone.utc)
        results: List[MappingResult] = []
        unmapped: List[str] = []

        for field_name, value in fields.items():
            mapping = self.map_field(source_regulation, field_name, target_regulation, value)
            if mapping.confidence > 0.0:
                results.append(mapping)
            else:
                unmapped.append(field_name)

        total = len(fields)
        mapped_count = len(results)
        avg_confidence = (
            sum(r.confidence for r in results) / mapped_count
            if mapped_count > 0 else 0.0
        )

        batch_result = BatchMappingResult(
            total_fields=total,
            mapped_count=mapped_count,
            unmapped_count=len(unmapped),
            mappings=results,
            unmapped_fields=unmapped,
            average_confidence=round(avg_confidence, 4),
        )
        batch_result.provenance_hash = _compute_hash(batch_result)

        elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        logger.info(
            "Batch mapping %s -> %s: %d/%d mapped (avg confidence=%.2f, %.1fms)",
            source_regulation, target_regulation, mapped_count, total,
            avg_confidence, elapsed_ms,
        )
        return batch_result

    # -------------------------------------------------------------------
    # get_mappings_for_regulation
    # -------------------------------------------------------------------

    def get_mappings_for_regulation(
        self,
        regulation: str,
        direction: str = "source",
        category: Optional[str] = None,
    ) -> List[FieldMapping]:
        """Get all mappings where a regulation appears as source or target.

        Args:
            regulation: Regulation name (CSRD, CBAM, EUDR, EU_TAXONOMY).
            direction: 'source' for outgoing, 'target' for incoming, 'both' for all.
            category: Optional category filter.

        Returns:
            List of FieldMapping objects.
        """
        results: List[FieldMapping] = []
        for mapping in self._all_mappings:
            match = False
            if direction in ("source", "both") and mapping.source_regulation == regulation:
                match = True
            if direction in ("target", "both") and mapping.target_regulation == regulation:
                match = True
            if match and category and mapping.category != category:
                match = False
            if match:
                results.append(mapping)

        logger.info(
            "Found %d mappings for %s (direction=%s, category=%s)",
            len(results), regulation, direction, category,
        )
        return results

    # -------------------------------------------------------------------
    # get_overlap_statistics
    # -------------------------------------------------------------------

    def get_overlap_statistics(
        self,
        regulation_a: str,
        regulation_b: str,
    ) -> OverlapStatistics:
        """Calculate overlap statistics between two regulations.

        Args:
            regulation_a: First regulation.
            regulation_b: Second regulation.

        Returns:
            OverlapStatistics with match counts and overlap percentage.
        """
        pair_mappings = [
            m for m in self._all_mappings
            if (m.source_regulation == regulation_a and m.target_regulation == regulation_b)
            or (m.source_regulation == regulation_b and m.target_regulation == regulation_a)
        ]

        source_fields = set()
        target_fields = set()
        exact = 0
        approximate = 0
        derived = 0
        categories_covered: set = set()

        for m in pair_mappings:
            source_fields.add(m.source_field)
            target_fields.add(m.target_field)
            categories_covered.add(m.category)
            if m.mapping_type == "EXACT":
                exact += 1
            elif m.mapping_type == "APPROXIMATE":
                approximate += 1
            elif m.mapping_type == "DERIVED":
                derived += 1

        total_fields = max(len(source_fields) + len(target_fields), 1)
        matched_fields = len(source_fields.intersection(target_fields))
        overlap_pct = round(len(pair_mappings) / total_fields * 100, 2) if total_fields > 0 else 0.0

        result = OverlapStatistics(
            regulation_pair=f"{regulation_a}-{regulation_b}",
            total_source_fields=len(source_fields),
            total_target_fields=len(target_fields),
            exact_matches=exact,
            approximate_matches=approximate,
            derived_matches=derived,
            overlap_percentage=overlap_pct,
            categories_covered=sorted(categories_covered),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Overlap %s-%s: exact=%d, approx=%d, derived=%d, overlap=%.1f%%",
            regulation_a, regulation_b, exact, approximate, derived, overlap_pct,
        )
        return result

    # -------------------------------------------------------------------
    # find_mapping_path
    # -------------------------------------------------------------------

    def find_mapping_path(
        self,
        source_regulation: str,
        source_field: str,
        target_regulation: str,
        target_field: Optional[str] = None,
    ) -> MappingPath:
        """Find a mapping path between fields, possibly via intermediate regulations.

        Uses breadth-first search to find the shortest path with highest
        compound confidence through intermediate regulation mappings.

        Args:
            source_regulation: Source regulation.
            source_field: Source field identifier.
            target_regulation: Target regulation.
            target_field: Optional specific target field (if None, finds any path).

        Returns:
            MappingPath with hop details and compound confidence.
        """
        if source_regulation == target_regulation:
            return MappingPath(
                source_regulation=source_regulation,
                source_field=source_field,
                target_regulation=target_regulation,
                target_field=source_field,
                hops=[],
                total_confidence=1.0,
                path_length=0,
                provenance_hash=_compute_hash({"direct": True}),
            )

        all_regulations = {"CSRD", "CBAM", "EUDR", "EU_TAXONOMY"}
        max_hops = self.config.max_path_hops

        queue: List[Tuple[str, str, List[Dict[str, str]], float]] = [
            (source_regulation, source_field, [], 1.0)
        ]
        visited: set = {(source_regulation, source_field)}
        best_path: Optional[Tuple[List[Dict[str, str]], float, str]] = None

        while queue:
            current_reg, current_field, hops, compound_conf = queue.pop(0)

            if len(hops) > max_hops:
                continue

            for next_reg in all_regulations:
                if next_reg == current_reg:
                    continue

                key = f"{current_reg}::{current_field}::{next_reg}"
                candidates = self._mappings_by_key.get(key, [])
                candidates = [c for c in candidates if c.confidence >= self.config.min_confidence]

                for candidate in candidates:
                    new_conf = compound_conf * candidate.confidence
                    hop_info = {
                        "from_regulation": current_reg,
                        "from_field": current_field,
                        "to_regulation": next_reg,
                        "to_field": candidate.target_field,
                        "mapping_type": candidate.mapping_type,
                        "hop_confidence": str(candidate.confidence),
                    }
                    new_hops = hops + [hop_info]

                    if next_reg == target_regulation:
                        if target_field is None or candidate.target_field == target_field:
                            if best_path is None or new_conf > best_path[1]:
                                best_path = (new_hops, new_conf, candidate.target_field)
                    else:
                        visit_key = (next_reg, candidate.target_field)
                        if visit_key not in visited and len(new_hops) < max_hops:
                            visited.add(visit_key)
                            queue.append((next_reg, candidate.target_field, new_hops, new_conf))

        if best_path is not None:
            path_hops, path_conf, resolved_target = best_path
            result = MappingPath(
                source_regulation=source_regulation,
                source_field=source_field,
                target_regulation=target_regulation,
                target_field=resolved_target,
                hops=path_hops,
                total_confidence=round(path_conf, 6),
                path_length=len(path_hops),
            )
            result.provenance_hash = _compute_hash(result)
            logger.info(
                "Found mapping path %s.%s -> %s.%s: %d hops, confidence=%.4f",
                source_regulation, source_field, target_regulation, resolved_target,
                len(path_hops), path_conf,
            )
            return result

        result = MappingPath(
            source_regulation=source_regulation,
            source_field=source_field,
            target_regulation=target_regulation,
            target_field=target_field or "",
            hops=[],
            total_confidence=0.0,
            path_length=0,
        )
        result.provenance_hash = _compute_hash(result)
        logger.warning(
            "No mapping path found: %s.%s -> %s",
            source_regulation, source_field, target_regulation,
        )
        return result

    # -------------------------------------------------------------------
    # get_all_categories
    # -------------------------------------------------------------------

    def get_all_categories(self) -> Dict[str, int]:
        """Get all mapping categories with counts.

        Returns:
            Dict of category name to number of mappings.
        """
        category_counts: Dict[str, int] = {}
        for mapping in self._all_mappings:
            category_counts[mapping.category] = category_counts.get(mapping.category, 0) + 1
        return category_counts

    # -------------------------------------------------------------------
    # get_total_mapping_count
    # -------------------------------------------------------------------

    def get_total_mapping_count(self) -> int:
        """Get total number of field mappings loaded.

        Returns:
            Total mapping count.
        """
        return len(self._all_mappings)

    # -------------------------------------------------------------------
    # get_regulation_field_list
    # -------------------------------------------------------------------

    def get_regulation_field_list(self, regulation: str) -> List[str]:
        """Get all unique fields referenced for a given regulation.

        Args:
            regulation: Regulation name.

        Returns:
            Sorted list of unique field names.
        """
        fields: set = set()
        for mapping in self._all_mappings:
            if mapping.source_regulation == regulation:
                fields.add(mapping.source_field)
            if mapping.target_regulation == regulation:
                fields.add(mapping.target_field)
        return sorted(fields)

    # -------------------------------------------------------------------
    # validate_mapping_coverage
    # -------------------------------------------------------------------

    def validate_mapping_coverage(
        self,
        regulation: str,
        required_fields: List[str],
    ) -> Dict[str, Any]:
        """Validate that required fields have at least one mapping.

        Args:
            regulation: Regulation to check.
            required_fields: List of required field names.

        Returns:
            Dict with coverage results and gaps.
        """
        available = set(self.get_regulation_field_list(regulation))
        covered = [f for f in required_fields if f in available]
        gaps = [f for f in required_fields if f not in available]
        coverage_pct = round(len(covered) / max(len(required_fields), 1) * 100, 2)

        result = {
            "regulation": regulation,
            "total_required": len(required_fields),
            "covered": len(covered),
            "gaps": len(gaps),
            "coverage_pct": coverage_pct,
            "covered_fields": covered,
            "gap_fields": gaps,
            "provenance_hash": _compute_hash({
                "regulation": regulation,
                "required": required_fields,
                "covered": covered,
                "gaps": gaps,
            }),
        }
        logger.info(
            "Mapping coverage for %s: %d/%d (%.1f%%)",
            regulation, len(covered), len(required_fields), coverage_pct,
        )
        return result
