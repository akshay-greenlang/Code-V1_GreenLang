# -*- coding: utf-8 -*-
"""
DataDeduplicationEngine - PACK-009 EU Climate Compliance Bundle Engine 2

Identifies and merges duplicate data collection requirements across CSRD,
CBAM, EUDR, and EU Taxonomy regulations. Reduces reporting burden by
detecting overlapping data fields, grouping duplicates, and generating
golden records with configurable merge strategies.

Core Capabilities:
    1. Requirement scanning across 4 EU regulations (~200 data fields)
    2. Duplicate detection via exact name, normalized name, and semantic matching
    3. Configurable merge strategies (FIRST_WINS, HIGHEST_CONFIDENCE, MOST_RECENT, MANUAL)
    4. Savings estimation (hours, cost reduction percentage)
    5. Golden record generation with full provenance tracking
    6. Conflict detection and resolution reporting

Zero-Hallucination:
    - All requirement definitions from published regulation text
    - Similarity scoring uses deterministic string algorithms
    - No LLM involvement in deduplication logic
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
import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

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


def _normalize_field_name(name: str) -> str:
    """Normalize a field name for comparison.

    Strips regulation prefixes, converts to lowercase, replaces
    separators with underscores, and removes common suffixes.
    """
    normalized = name.lower().strip()
    prefixes = ["e1_", "e2_", "e3_", "e4_", "s1_", "s2_", "s3_", "g1_",
                 "esrs_", "csrd_", "cbam_", "eudr_", "taxonomy_", "ccm_",
                 "cca_", "wtr_", "ppc_", "bio_"]
    for prefix in prefixes:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break
    normalized = re.sub(r"[-.\s]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    normalized = normalized.strip("_")
    return normalized


def _compute_similarity(name_a: str, name_b: str) -> float:
    """Compute Jaccard similarity between two normalized field names.

    Uses word-level n-grams for a fast deterministic similarity score.

    Args:
        name_a: First normalized field name.
        name_b: Second normalized field name.

    Returns:
        Float between 0.0 and 1.0.
    """
    tokens_a = set(name_a.split("_"))
    tokens_b = set(name_b.split("_"))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a.intersection(tokens_b)
    union = tokens_a.union(tokens_b)
    return len(intersection) / len(union) if union else 0.0


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MergeStrategy(str, Enum):
    """Strategy for merging duplicate requirements."""
    FIRST_WINS = "FIRST_WINS"
    HIGHEST_CONFIDENCE = "HIGHEST_CONFIDENCE"
    MOST_RECENT = "MOST_RECENT"
    MANUAL = "MANUAL"


class MatchType(str, Enum):
    """How a duplicate match was identified."""
    EXACT_NAME = "EXACT_NAME"
    NORMALIZED_NAME = "NORMALIZED_NAME"
    SEMANTIC_SIMILARITY = "SEMANTIC_SIMILARITY"
    TYPE_AND_DESCRIPTION = "TYPE_AND_DESCRIPTION"


class ConflictSeverity(str, Enum):
    """Severity of a merge conflict."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class DataRequirement(BaseModel):
    """A single data collection requirement from a regulation."""
    requirement_id: str = Field(default_factory=_new_uuid, description="Unique requirement ID")
    regulation: str = Field(..., description="Source regulation (CSRD, CBAM, EUDR, EU_TAXONOMY)")
    field_name: str = Field(..., description="Canonical field name")
    data_type: str = Field(default="string", description="Data type (string, numeric, date, boolean, enum)")
    description: str = Field(default="", description="Human-readable description of the data requirement")
    required: bool = Field(default=True, description="Whether the field is mandatory")
    frequency: str = Field(default="annual", description="Collection frequency (annual, quarterly, continuous)")
    unit: str = Field(default="", description="Unit of measurement if applicable")
    category: str = Field(default="", description="Thematic category (emissions, financial, supply_chain, etc)")
    collection_effort_hours: float = Field(default=1.0, ge=0.0, description="Estimated effort to collect in hours")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Data quality confidence")
    last_updated: str = Field(
        default_factory=lambda: _utcnow().isoformat(),
        description="Last update timestamp"
    )


class MergeConflict(BaseModel):
    """A conflict detected during duplicate merging."""
    conflict_id: str = Field(default_factory=_new_uuid, description="Conflict identifier")
    field_attribute: str = Field(default="", description="Attribute where conflict exists")
    values_by_regulation: Dict[str, str] = Field(default_factory=dict, description="Conflicting values by regulation")
    severity: str = Field(default="LOW", description="Conflict severity")
    resolution_suggestion: str = Field(default="", description="Suggested resolution")


class DeduplicationGroup(BaseModel):
    """A group of duplicate data requirements with merge decision."""
    group_id: str = Field(default_factory=_new_uuid, description="Group identifier")
    canonical_field: str = Field(default="", description="Canonical golden field name")
    requirements: List[DataRequirement] = Field(default_factory=list, description="Grouped requirements")
    match_type: str = Field(default="EXACT_NAME", description="How the match was identified")
    similarity_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Match similarity score")
    merge_decision: str = Field(default="PENDING", description="Merge decision (MERGED, KEPT_SEPARATE, PENDING)")
    conflicts: List[MergeConflict] = Field(default_factory=list, description="Merge conflicts")
    savings_estimate_hours: float = Field(default=0.0, description="Estimated hours saved by deduplication")
    regulations_involved: List[str] = Field(default_factory=list, description="Regulations in this group")


class GoldenRecord(BaseModel):
    """Merged golden record after deduplication."""
    record_id: str = Field(default_factory=_new_uuid, description="Record identifier")
    canonical_field: str = Field(default="", description="Canonical field name")
    data_type: str = Field(default="string", description="Resolved data type")
    description: str = Field(default="", description="Merged description")
    required: bool = Field(default=True, description="Whether mandatory in any regulation")
    frequency: str = Field(default="annual", description="Highest frequency needed")
    unit: str = Field(default="", description="Resolved unit")
    source_regulations: List[str] = Field(default_factory=list, description="All source regulations")
    source_fields: Dict[str, str] = Field(default_factory=dict, description="Original field names by regulation")
    merge_strategy_used: str = Field(default="", description="Strategy used for merging")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class DeduplicationResult(BaseModel):
    """Complete result of a deduplication scan."""
    result_id: str = Field(default_factory=_new_uuid, description="Result identifier")
    total_requirements_scanned: int = Field(default=0, description="Total requirements scanned")
    unique_requirements: int = Field(default=0, description="Unique requirements (no duplicates)")
    duplicate_groups: int = Field(default=0, description="Number of duplicate groups")
    total_duplicates: int = Field(default=0, description="Total duplicate requirements found")
    groups: List[DeduplicationGroup] = Field(default_factory=list, description="Deduplication groups")
    golden_records: List[GoldenRecord] = Field(default_factory=list, description="Merged golden records")
    total_savings_hours: float = Field(default=0.0, description="Total estimated hours saved")
    savings_percentage: float = Field(default=0.0, description="Effort reduction percentage")
    conflicts: List[MergeConflict] = Field(default_factory=list, description="All unresolved conflicts")
    regulations_analyzed: List[str] = Field(default_factory=list, description="Regulations analyzed")
    timestamp: str = Field(default_factory=lambda: _utcnow().isoformat(), description="Analysis timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class DedupReport(BaseModel):
    """Human-readable deduplication report."""
    report_id: str = Field(default_factory=_new_uuid, description="Report identifier")
    summary: str = Field(default="", description="Executive summary")
    regulation_breakdown: Dict[str, Dict[str, int]] = Field(
        default_factory=dict, description="Breakdown by regulation"
    )
    top_savings_opportunities: List[Dict[str, Any]] = Field(
        default_factory=list, description="Top savings opportunities"
    )
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class DataDeduplicationConfig(BaseModel):
    """Configuration for the DataDeduplicationEngine."""
    merge_strategy: str = Field(
        default="HIGHEST_CONFIDENCE",
        description="Merge strategy: FIRST_WINS, HIGHEST_CONFIDENCE, MOST_RECENT, MANUAL"
    )
    conflict_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Similarity threshold above which fields are considered duplicates"
    )
    enable_fuzzy_matching: bool = Field(
        default=True, description="Enable fuzzy/semantic similarity matching"
    )
    fuzzy_threshold: float = Field(
        default=0.6, ge=0.0, le=1.0,
        description="Minimum similarity score for fuzzy matches"
    )
    hourly_rate_eur: float = Field(
        default=75.0, ge=0.0,
        description="Hourly rate for cost savings estimation in EUR"
    )

    @field_validator("merge_strategy", mode="before")
    @classmethod
    def _validate_strategy(cls, v: Any) -> str:
        allowed = {"FIRST_WINS", "HIGHEST_CONFIDENCE", "MOST_RECENT", "MANUAL"}
        val = str(v).upper()
        if val not in allowed:
            return "HIGHEST_CONFIDENCE"
        return val


# ---------------------------------------------------------------------------
# Model rebuilds
# ---------------------------------------------------------------------------

DataDeduplicationConfig.model_rebuild()
DataRequirement.model_rebuild()
DeduplicationGroup.model_rebuild()
GoldenRecord.model_rebuild()
DeduplicationResult.model_rebuild()
DedupReport.model_rebuild()
MergeConflict.model_rebuild()


# ---------------------------------------------------------------------------
# Regulation Data Requirements Database (~200 fields across 4 regulations)
# ---------------------------------------------------------------------------

REGULATION_DATA_REQUIREMENTS: Dict[str, List[Dict[str, Any]]] = {
    "CSRD": [
        {"field_name": "scope1_ghg_emissions", "data_type": "numeric", "description": "Gross Scope 1 GHG emissions in tCO2e", "required": True, "frequency": "annual", "unit": "tCO2e", "category": "emissions", "collection_effort_hours": 8.0},
        {"field_name": "scope2_ghg_emissions_location", "data_type": "numeric", "description": "Scope 2 GHG emissions location-based", "required": True, "frequency": "annual", "unit": "tCO2e", "category": "emissions", "collection_effort_hours": 6.0},
        {"field_name": "scope2_ghg_emissions_market", "data_type": "numeric", "description": "Scope 2 GHG emissions market-based", "required": True, "frequency": "annual", "unit": "tCO2e", "category": "emissions", "collection_effort_hours": 6.0},
        {"field_name": "scope3_category1_emissions", "data_type": "numeric", "description": "Scope 3 Cat 1 purchased goods & services emissions", "required": True, "frequency": "annual", "unit": "tCO2e", "category": "emissions", "collection_effort_hours": 20.0},
        {"field_name": "total_energy_consumption", "data_type": "numeric", "description": "Total energy consumption from non-renewable sources", "required": True, "frequency": "annual", "unit": "MWh", "category": "energy", "collection_effort_hours": 4.0},
        {"field_name": "renewable_energy_share", "data_type": "numeric", "description": "Share of renewable energy in total mix", "required": True, "frequency": "annual", "unit": "percent", "category": "energy", "collection_effort_hours": 3.0},
        {"field_name": "ghg_reduction_targets", "data_type": "string", "description": "GHG emission reduction targets", "required": True, "frequency": "annual", "unit": "", "category": "emissions", "collection_effort_hours": 5.0},
        {"field_name": "transition_plan_actions", "data_type": "string", "description": "Climate transition plan actions and investments", "required": True, "frequency": "annual", "unit": "", "category": "strategy", "collection_effort_hours": 10.0},
        {"field_name": "carbon_pricing_exposure", "data_type": "numeric", "description": "Financial exposure to carbon pricing mechanisms", "required": True, "frequency": "annual", "unit": "EUR", "category": "financial", "collection_effort_hours": 6.0},
        {"field_name": "taxonomy_turnover_kpi", "data_type": "numeric", "description": "EU Taxonomy aligned turnover percentage", "required": True, "frequency": "annual", "unit": "percent", "category": "financial", "collection_effort_hours": 15.0},
        {"field_name": "taxonomy_capex_kpi", "data_type": "numeric", "description": "EU Taxonomy aligned CapEx percentage", "required": True, "frequency": "annual", "unit": "percent", "category": "financial", "collection_effort_hours": 15.0},
        {"field_name": "taxonomy_opex_kpi", "data_type": "numeric", "description": "EU Taxonomy aligned OpEx percentage", "required": True, "frequency": "annual", "unit": "percent", "category": "financial", "collection_effort_hours": 15.0},
        {"field_name": "water_consumption_total", "data_type": "numeric", "description": "Total water consumption in cubic metres", "required": True, "frequency": "annual", "unit": "m3", "category": "water", "collection_effort_hours": 4.0},
        {"field_name": "water_stressed_area_operations", "data_type": "boolean", "description": "Operations in water-stressed areas flag", "required": True, "frequency": "annual", "unit": "", "category": "water", "collection_effort_hours": 3.0},
        {"field_name": "air_pollutant_emissions", "data_type": "numeric", "description": "Air pollutant emissions (SOx, NOx, PM)", "required": True, "frequency": "annual", "unit": "tonnes", "category": "pollution", "collection_effort_hours": 5.0},
        {"field_name": "water_pollutant_discharges", "data_type": "numeric", "description": "Water pollutant discharge amounts", "required": True, "frequency": "annual", "unit": "tonnes", "category": "pollution", "collection_effort_hours": 5.0},
        {"field_name": "substances_of_concern_volume", "data_type": "numeric", "description": "Volume of substances of concern used", "required": True, "frequency": "annual", "unit": "tonnes", "category": "pollution", "collection_effort_hours": 6.0},
        {"field_name": "biodiversity_impact_assessment", "data_type": "string", "description": "Biodiversity impact assessment results", "required": True, "frequency": "annual", "unit": "", "category": "biodiversity", "collection_effort_hours": 8.0},
        {"field_name": "deforestation_policy", "data_type": "string", "description": "Deforestation and land degradation policy", "required": True, "frequency": "annual", "unit": "", "category": "biodiversity", "collection_effort_hours": 4.0},
        {"field_name": "protected_area_proximity", "data_type": "boolean", "description": "Operations near protected/Natura 2000 areas", "required": True, "frequency": "annual", "unit": "", "category": "biodiversity", "collection_effort_hours": 3.0},
        {"field_name": "supply_chain_due_diligence", "data_type": "string", "description": "Supply chain due diligence process description", "required": True, "frequency": "annual", "unit": "", "category": "supply_chain", "collection_effort_hours": 10.0},
        {"field_name": "supplier_tier1_list", "data_type": "string", "description": "Tier 1 supplier identification and mapping", "required": True, "frequency": "annual", "unit": "", "category": "supply_chain", "collection_effort_hours": 8.0},
        {"field_name": "worker_rights_policy", "data_type": "string", "description": "Worker rights and consultation policy", "required": True, "frequency": "annual", "unit": "", "category": "social", "collection_effort_hours": 4.0},
        {"field_name": "remediation_actions", "data_type": "string", "description": "Remediation of negative impacts", "required": True, "frequency": "annual", "unit": "", "category": "supply_chain", "collection_effort_hours": 5.0},
        {"field_name": "revenue_by_geography", "data_type": "numeric", "description": "Revenue disaggregated by geography", "required": True, "frequency": "annual", "unit": "EUR", "category": "financial", "collection_effort_hours": 4.0},
        {"field_name": "nace_sector_classification", "data_type": "string", "description": "NACE Rev.2 sector classification codes", "required": True, "frequency": "annual", "unit": "", "category": "classification", "collection_effort_hours": 2.0},
        {"field_name": "scenario_analysis_results", "data_type": "string", "description": "Climate scenario analysis results (1.5C/2C/4C)", "required": True, "frequency": "annual", "unit": "", "category": "risk", "collection_effort_hours": 12.0},
        {"field_name": "physical_climate_risk_exposure", "data_type": "string", "description": "Physical climate risk exposure assessment", "required": True, "frequency": "annual", "unit": "", "category": "risk", "collection_effort_hours": 10.0},
        {"field_name": "transition_risk_exposure", "data_type": "string", "description": "Transition risk exposure from policy changes", "required": True, "frequency": "annual", "unit": "", "category": "risk", "collection_effort_hours": 8.0},
        {"field_name": "stranded_asset_exposure", "data_type": "numeric", "description": "Stranded asset financial exposure", "required": True, "frequency": "annual", "unit": "EUR", "category": "financial", "collection_effort_hours": 6.0},
        {"field_name": "pollution_incident_count", "data_type": "numeric", "description": "Number of pollution incidents in reporting period", "required": True, "frequency": "annual", "unit": "", "category": "pollution", "collection_effort_hours": 2.0},
        {"field_name": "emission_intensity_per_revenue", "data_type": "numeric", "description": "GHG emission intensity per net revenue", "required": True, "frequency": "annual", "unit": "tCO2e/EUR", "category": "emissions", "collection_effort_hours": 3.0},
        {"field_name": "climate_adaptation_actions", "data_type": "string", "description": "Climate change adaptation policies and actions", "required": True, "frequency": "annual", "unit": "", "category": "strategy", "collection_effort_hours": 6.0},
        {"field_name": "decarbonization_investment", "data_type": "numeric", "description": "Investment in decarbonization actions", "required": True, "frequency": "annual", "unit": "EUR", "category": "financial", "collection_effort_hours": 5.0},
        {"field_name": "supply_chain_audit_results", "data_type": "string", "description": "Supply chain audit findings and results", "required": False, "frequency": "annual", "unit": "", "category": "supply_chain", "collection_effort_hours": 8.0},
        {"field_name": "biodiversity_restoration_targets", "data_type": "string", "description": "Biodiversity restoration targets and progress", "required": True, "frequency": "annual", "unit": "", "category": "biodiversity", "collection_effort_hours": 5.0},
        {"field_name": "water_withdrawal_by_source", "data_type": "numeric", "description": "Water withdrawal disaggregated by source", "required": True, "frequency": "annual", "unit": "m3", "category": "water", "collection_effort_hours": 4.0},
        {"field_name": "wastewater_treatment_quality", "data_type": "string", "description": "Wastewater treatment quality metrics", "required": True, "frequency": "annual", "unit": "", "category": "water", "collection_effort_hours": 3.0},
        {"field_name": "indigenous_community_impact", "data_type": "string", "description": "Impact on indigenous and local communities", "required": False, "frequency": "annual", "unit": "", "category": "social", "collection_effort_hours": 6.0},
        {"field_name": "compliance_cost_environmental", "data_type": "numeric", "description": "Total environmental compliance cost", "required": False, "frequency": "annual", "unit": "EUR", "category": "financial", "collection_effort_hours": 4.0},
        {"field_name": "biodiversity_monitoring_data", "data_type": "string", "description": "Biodiversity monitoring methodology and data", "required": False, "frequency": "annual", "unit": "", "category": "biodiversity", "collection_effort_hours": 6.0},
        {"field_name": "green_asset_ratio", "data_type": "numeric", "description": "Green Asset Ratio for financial institutions", "required": False, "frequency": "annual", "unit": "percent", "category": "financial", "collection_effort_hours": 12.0},
        {"field_name": "biogenic_co2_emissions", "data_type": "numeric", "description": "Biogenic CO2 emissions reported separately", "required": True, "frequency": "annual", "unit": "tCO2", "category": "emissions", "collection_effort_hours": 3.0},
        {"field_name": "ghg_removals_credits", "data_type": "numeric", "description": "GHG removals and carbon credits", "required": True, "frequency": "annual", "unit": "tCO2e", "category": "emissions", "collection_effort_hours": 4.0},
        {"field_name": "carbon_leakage_risk", "data_type": "string", "description": "Carbon leakage risk assessment", "required": False, "frequency": "annual", "unit": "", "category": "risk", "collection_effort_hours": 5.0},
        {"field_name": "vulnerability_mapping", "data_type": "string", "description": "Climate vulnerability identification mapping", "required": True, "frequency": "annual", "unit": "", "category": "risk", "collection_effort_hours": 8.0},
        {"field_name": "hazard_type_classification", "data_type": "string", "description": "Physical climate hazard classification", "required": True, "frequency": "annual", "unit": "", "category": "risk", "collection_effort_hours": 4.0},
        {"field_name": "adaptation_opportunity_revenue", "data_type": "numeric", "description": "Revenue from climate adaptation opportunities", "required": False, "frequency": "annual", "unit": "EUR", "category": "financial", "collection_effort_hours": 5.0},
        {"field_name": "water_recycled_percentage", "data_type": "numeric", "description": "Water recycled and reused percentage", "required": True, "frequency": "annual", "unit": "percent", "category": "water", "collection_effort_hours": 3.0},
        {"field_name": "pollution_prevention_policy", "data_type": "string", "description": "Pollution prevention policies", "required": True, "frequency": "annual", "unit": "", "category": "pollution", "collection_effort_hours": 3.0},
    ],
    "CBAM": [
        {"field_name": "direct_emissions_tco2e", "data_type": "numeric", "description": "Direct (Scope 1) embedded emissions per installation", "required": True, "frequency": "quarterly", "unit": "tCO2e", "category": "emissions", "collection_effort_hours": 10.0},
        {"field_name": "indirect_emissions_tco2e", "data_type": "numeric", "description": "Indirect embedded emissions from electricity", "required": True, "frequency": "quarterly", "unit": "tCO2e", "category": "emissions", "collection_effort_hours": 8.0},
        {"field_name": "total_embedded_emissions", "data_type": "numeric", "description": "Total embedded emissions in imported goods", "required": True, "frequency": "quarterly", "unit": "tCO2e", "category": "emissions", "collection_effort_hours": 4.0},
        {"field_name": "specific_embedded_emissions", "data_type": "numeric", "description": "Specific embedded emissions per tonne of product", "required": True, "frequency": "quarterly", "unit": "tCO2e/t", "category": "emissions", "collection_effort_hours": 6.0},
        {"field_name": "imported_quantity_tonnes", "data_type": "numeric", "description": "Quantity of goods imported in tonnes", "required": True, "frequency": "quarterly", "unit": "tonnes", "category": "trade", "collection_effort_hours": 2.0},
        {"field_name": "cn_code_8digit", "data_type": "string", "description": "Combined Nomenclature 8-digit product code", "required": True, "frequency": "quarterly", "unit": "", "category": "classification", "collection_effort_hours": 1.0},
        {"field_name": "goods_category", "data_type": "enum", "description": "CBAM goods category (iron_steel, cement, aluminium, etc)", "required": True, "frequency": "quarterly", "unit": "", "category": "classification", "collection_effort_hours": 1.0},
        {"field_name": "country_of_origin", "data_type": "string", "description": "Country of origin for imported goods (ISO code)", "required": True, "frequency": "quarterly", "unit": "", "category": "trade", "collection_effort_hours": 1.0},
        {"field_name": "installation_name", "data_type": "string", "description": "Name of production installation", "required": True, "frequency": "quarterly", "unit": "", "category": "supply_chain", "collection_effort_hours": 2.0},
        {"field_name": "installation_country", "data_type": "string", "description": "Country where installation is located", "required": True, "frequency": "quarterly", "unit": "", "category": "supply_chain", "collection_effort_hours": 1.0},
        {"field_name": "carbon_price_paid_origin", "data_type": "numeric", "description": "Carbon price effectively paid in country of origin", "required": True, "frequency": "quarterly", "unit": "EUR/tCO2e", "category": "financial", "collection_effort_hours": 4.0},
        {"field_name": "cbam_certificate_cost", "data_type": "numeric", "description": "Total CBAM certificate purchase cost", "required": True, "frequency": "annual", "unit": "EUR", "category": "financial", "collection_effort_hours": 3.0},
        {"field_name": "certificate_quantity", "data_type": "numeric", "description": "Number of CBAM certificates purchased", "required": True, "frequency": "annual", "unit": "", "category": "financial", "collection_effort_hours": 2.0},
        {"field_name": "production_process_route", "data_type": "string", "description": "Production process route (BOF, EAF, etc)", "required": True, "frequency": "quarterly", "unit": "", "category": "classification", "collection_effort_hours": 3.0},
        {"field_name": "emission_factor_electricity", "data_type": "numeric", "description": "Electricity emission factor for installation", "required": True, "frequency": "quarterly", "unit": "tCO2e/MWh", "category": "emissions", "collection_effort_hours": 3.0},
        {"field_name": "precursor_emissions", "data_type": "numeric", "description": "Embedded emissions from precursor materials", "required": True, "frequency": "quarterly", "unit": "tCO2e", "category": "emissions", "collection_effort_hours": 8.0},
        {"field_name": "monitoring_methodology", "data_type": "string", "description": "Emission monitoring methodology reference", "required": True, "frequency": "quarterly", "unit": "", "category": "emissions", "collection_effort_hours": 3.0},
        {"field_name": "eori_number", "data_type": "string", "description": "EORI number of the importer/declarant", "required": True, "frequency": "quarterly", "unit": "", "category": "trade", "collection_effort_hours": 1.0},
        {"field_name": "financial_guarantee_amount", "data_type": "numeric", "description": "Financial guarantee amount posted", "required": True, "frequency": "annual", "unit": "EUR", "category": "financial", "collection_effort_hours": 2.0},
        {"field_name": "deminimis_check_result", "data_type": "boolean", "description": "De minimis threshold check result", "required": True, "frequency": "quarterly", "unit": "", "category": "compliance", "collection_effort_hours": 1.0},
        {"field_name": "biogenic_emissions_tco2", "data_type": "numeric", "description": "Biogenic emissions reported separately", "required": True, "frequency": "quarterly", "unit": "tCO2", "category": "emissions", "collection_effort_hours": 3.0},
        {"field_name": "ets_free_allocation_factor", "data_type": "numeric", "description": "EU ETS free allocation phase-out factor", "required": True, "frequency": "annual", "unit": "", "category": "financial", "collection_effort_hours": 2.0},
        {"field_name": "eu_ets_benchmark_value", "data_type": "numeric", "description": "EU ETS product benchmark value", "required": True, "frequency": "annual", "unit": "tCO2e/t", "category": "emissions", "collection_effort_hours": 2.0},
        {"field_name": "carbon_leakage_sector", "data_type": "boolean", "description": "Carbon leakage sector classification", "required": True, "frequency": "annual", "unit": "", "category": "risk", "collection_effort_hours": 1.0},
        {"field_name": "import_value_by_country", "data_type": "numeric", "description": "Import value by country of origin", "required": True, "frequency": "quarterly", "unit": "EUR", "category": "financial", "collection_effort_hours": 2.0},
    ],
    "EUDR": [
        {"field_name": "commodity_type", "data_type": "enum", "description": "Regulated commodity type (palm, soy, cocoa, coffee, rubber, wood, cattle)", "required": True, "frequency": "continuous", "unit": "", "category": "classification", "collection_effort_hours": 1.0},
        {"field_name": "product_quantity_kg", "data_type": "numeric", "description": "Product quantity in kilograms", "required": True, "frequency": "continuous", "unit": "kg", "category": "trade", "collection_effort_hours": 1.0},
        {"field_name": "geolocation_coordinates", "data_type": "string", "description": "GPS coordinates of production plots", "required": True, "frequency": "continuous", "unit": "", "category": "supply_chain", "collection_effort_hours": 10.0},
        {"field_name": "deforestation_free_status", "data_type": "boolean", "description": "Deforestation-free compliance status", "required": True, "frequency": "continuous", "unit": "", "category": "biodiversity", "collection_effort_hours": 8.0},
        {"field_name": "deforestation_cutoff_date", "data_type": "date", "description": "Deforestation cutoff date (31 December 2020)", "required": True, "frequency": "continuous", "unit": "", "category": "biodiversity", "collection_effort_hours": 2.0},
        {"field_name": "country_of_production", "data_type": "string", "description": "Country where commodity was produced", "required": True, "frequency": "continuous", "unit": "", "category": "supply_chain", "collection_effort_hours": 1.0},
        {"field_name": "supplier_name", "data_type": "string", "description": "Name of the supplier/operator", "required": True, "frequency": "continuous", "unit": "", "category": "supply_chain", "collection_effort_hours": 1.0},
        {"field_name": "due_diligence_statement_id", "data_type": "string", "description": "DDS reference number submitted to authority", "required": True, "frequency": "continuous", "unit": "", "category": "compliance", "collection_effort_hours": 4.0},
        {"field_name": "risk_assessment_result", "data_type": "string", "description": "Country/region risk assessment (negligible/standard/high)", "required": True, "frequency": "continuous", "unit": "", "category": "risk", "collection_effort_hours": 6.0},
        {"field_name": "risk_mitigation_measures", "data_type": "string", "description": "Risk mitigation and corrective actions", "required": True, "frequency": "continuous", "unit": "", "category": "risk", "collection_effort_hours": 5.0},
        {"field_name": "certification_status", "data_type": "string", "description": "Sustainability certification status (FSC, RSPO, etc)", "required": False, "frequency": "continuous", "unit": "", "category": "supply_chain", "collection_effort_hours": 3.0},
        {"field_name": "traceability_chain_complete", "data_type": "boolean", "description": "Complete traceability chain to production source", "required": True, "frequency": "continuous", "unit": "", "category": "supply_chain", "collection_effort_hours": 12.0},
        {"field_name": "satellite_monitoring_data", "data_type": "string", "description": "Satellite/remote sensing monitoring data reference", "required": False, "frequency": "continuous", "unit": "", "category": "biodiversity", "collection_effort_hours": 6.0},
        {"field_name": "forest_degradation_assessment", "data_type": "string", "description": "Forest degradation risk assessment", "required": True, "frequency": "continuous", "unit": "", "category": "biodiversity", "collection_effort_hours": 5.0},
        {"field_name": "indigenous_peoples_rights", "data_type": "string", "description": "Indigenous peoples and local community rights compliance", "required": True, "frequency": "continuous", "unit": "", "category": "social", "collection_effort_hours": 6.0},
        {"field_name": "legal_compliance_origin", "data_type": "boolean", "description": "Compliance with laws of country of production", "required": True, "frequency": "continuous", "unit": "", "category": "compliance", "collection_effort_hours": 4.0},
        {"field_name": "hs_code_product", "data_type": "string", "description": "Harmonized System product code", "required": True, "frequency": "continuous", "unit": "", "category": "classification", "collection_effort_hours": 1.0},
        {"field_name": "third_party_verification", "data_type": "string", "description": "Third-party verification results", "required": False, "frequency": "annual", "unit": "", "category": "compliance", "collection_effort_hours": 8.0},
        {"field_name": "smallholder_assessment", "data_type": "string", "description": "Smallholder farmer impact assessment", "required": False, "frequency": "annual", "unit": "", "category": "social", "collection_effort_hours": 5.0},
        {"field_name": "non_compliance_reporting", "data_type": "string", "description": "Non-compliance incident reporting", "required": True, "frequency": "continuous", "unit": "", "category": "compliance", "collection_effort_hours": 3.0},
        {"field_name": "sustainable_production_evidence", "data_type": "string", "description": "Evidence of sustainable production practices", "required": False, "frequency": "annual", "unit": "", "category": "supply_chain", "collection_effort_hours": 6.0},
        {"field_name": "stakeholder_consultation_log", "data_type": "string", "description": "Stakeholder consultation for risk mitigation", "required": False, "frequency": "annual", "unit": "", "category": "social", "collection_effort_hours": 4.0},
        {"field_name": "due_diligence_cost_total", "data_type": "numeric", "description": "Total due diligence compliance cost", "required": False, "frequency": "annual", "unit": "EUR", "category": "financial", "collection_effort_hours": 3.0},
        {"field_name": "operator_supplier_registry", "data_type": "string", "description": "Operator/trader supplier registry", "required": True, "frequency": "continuous", "unit": "", "category": "supply_chain", "collection_effort_hours": 5.0},
        {"field_name": "annual_review_result", "data_type": "string", "description": "Annual DDS system review result", "required": True, "frequency": "annual", "unit": "", "category": "compliance", "collection_effort_hours": 6.0},
    ],
    "EU_TAXONOMY": [
        {"field_name": "turnover_kpi_percentage", "data_type": "numeric", "description": "Article 8 Taxonomy-aligned turnover KPI", "required": True, "frequency": "annual", "unit": "percent", "category": "financial", "collection_effort_hours": 15.0},
        {"field_name": "capex_kpi_percentage", "data_type": "numeric", "description": "Article 8 Taxonomy-aligned CapEx KPI", "required": True, "frequency": "annual", "unit": "percent", "category": "financial", "collection_effort_hours": 15.0},
        {"field_name": "opex_kpi_percentage", "data_type": "numeric", "description": "Article 8 Taxonomy-aligned OpEx KPI", "required": True, "frequency": "annual", "unit": "percent", "category": "financial", "collection_effort_hours": 15.0},
        {"field_name": "nace_code_primary", "data_type": "string", "description": "Primary NACE Rev.2 activity code", "required": True, "frequency": "annual", "unit": "", "category": "classification", "collection_effort_hours": 2.0},
        {"field_name": "ccm_ghg_emissions_total", "data_type": "numeric", "description": "CCM total GHG emissions metric", "required": True, "frequency": "annual", "unit": "tCO2e", "category": "emissions", "collection_effort_hours": 8.0},
        {"field_name": "ccm_emission_intensity", "data_type": "numeric", "description": "CCM emission intensity per unit output", "required": True, "frequency": "annual", "unit": "tCO2e/unit", "category": "emissions", "collection_effort_hours": 6.0},
        {"field_name": "ccm_reduction_pathway", "data_type": "string", "description": "CCM emission reduction pathway alignment", "required": True, "frequency": "annual", "unit": "", "category": "strategy", "collection_effort_hours": 8.0},
        {"field_name": "ccm_benchmark_threshold", "data_type": "numeric", "description": "CCM sector benchmark threshold", "required": True, "frequency": "annual", "unit": "tCO2e/t", "category": "emissions", "collection_effort_hours": 3.0},
        {"field_name": "ccm_renewable_energy_pct", "data_type": "numeric", "description": "CCM renewable energy share percentage", "required": True, "frequency": "annual", "unit": "percent", "category": "energy", "collection_effort_hours": 3.0},
        {"field_name": "ccm_energy_efficiency_metric", "data_type": "numeric", "description": "CCM energy efficiency per unit output", "required": True, "frequency": "annual", "unit": "MWh/unit", "category": "energy", "collection_effort_hours": 4.0},
        {"field_name": "ccm_production_technology", "data_type": "string", "description": "CCM production technology classification", "required": True, "frequency": "annual", "unit": "", "category": "classification", "collection_effort_hours": 2.0},
        {"field_name": "ccm_transition_plan_alignment", "data_type": "string", "description": "CCM transition plan alignment assessment", "required": True, "frequency": "annual", "unit": "", "category": "strategy", "collection_effort_hours": 10.0},
        {"field_name": "cca_physical_risk_assessment", "data_type": "string", "description": "CCA physical climate risk assessment", "required": True, "frequency": "annual", "unit": "", "category": "risk", "collection_effort_hours": 10.0},
        {"field_name": "cca_adaptation_plan", "data_type": "string", "description": "CCA climate adaptation plan", "required": True, "frequency": "annual", "unit": "", "category": "strategy", "collection_effort_hours": 8.0},
        {"field_name": "cca_scenario_analysis", "data_type": "string", "description": "CCA climate scenario analysis requirements", "required": True, "frequency": "annual", "unit": "", "category": "risk", "collection_effort_hours": 10.0},
        {"field_name": "cca_vulnerability_assessment", "data_type": "string", "description": "CCA climate vulnerability assessment output", "required": True, "frequency": "annual", "unit": "", "category": "risk", "collection_effort_hours": 8.0},
        {"field_name": "cca_hazard_classification", "data_type": "string", "description": "CCA climate hazard type classification", "required": True, "frequency": "annual", "unit": "", "category": "risk", "collection_effort_hours": 4.0},
        {"field_name": "wtr_water_consumption_m3", "data_type": "numeric", "description": "WTR total water consumption in m3", "required": True, "frequency": "annual", "unit": "m3", "category": "water", "collection_effort_hours": 4.0},
        {"field_name": "wtr_water_stress_area_flag", "data_type": "boolean", "description": "WTR water-stressed area identification", "required": True, "frequency": "annual", "unit": "", "category": "water", "collection_effort_hours": 3.0},
        {"field_name": "wtr_water_recycling_pct", "data_type": "numeric", "description": "WTR water recycling/reuse percentage", "required": True, "frequency": "annual", "unit": "percent", "category": "water", "collection_effort_hours": 3.0},
        {"field_name": "wtr_water_withdrawal_source", "data_type": "string", "description": "WTR water withdrawal source breakdown", "required": True, "frequency": "annual", "unit": "", "category": "water", "collection_effort_hours": 4.0},
        {"field_name": "ppc_pollutant_emissions_air", "data_type": "numeric", "description": "PPC air pollutant emissions", "required": True, "frequency": "annual", "unit": "tonnes", "category": "pollution", "collection_effort_hours": 5.0},
        {"field_name": "ppc_pollutant_emissions_water", "data_type": "numeric", "description": "PPC water pollutant discharges", "required": True, "frequency": "annual", "unit": "tonnes", "category": "pollution", "collection_effort_hours": 5.0},
        {"field_name": "ppc_substances_of_concern", "data_type": "string", "description": "PPC substances of concern management", "required": True, "frequency": "annual", "unit": "", "category": "pollution", "collection_effort_hours": 6.0},
        {"field_name": "ppc_pollution_prevention_plan", "data_type": "string", "description": "PPC pollution prevention and control plan", "required": True, "frequency": "annual", "unit": "", "category": "pollution", "collection_effort_hours": 5.0},
        {"field_name": "bio_biodiversity_impact_assessment", "data_type": "string", "description": "BIO biodiversity impact assessment (EIA/SEA)", "required": True, "frequency": "annual", "unit": "", "category": "biodiversity", "collection_effort_hours": 8.0},
        {"field_name": "bio_protected_area_proximity", "data_type": "boolean", "description": "BIO proximity to protected/Natura 2000 areas", "required": True, "frequency": "annual", "unit": "", "category": "biodiversity", "collection_effort_hours": 3.0},
        {"field_name": "bio_ecosystem_restoration_plan", "data_type": "string", "description": "BIO ecosystem restoration and offsetting plan", "required": True, "frequency": "annual", "unit": "", "category": "biodiversity", "collection_effort_hours": 6.0},
        {"field_name": "bio_sustainable_forestry_criteria", "data_type": "string", "description": "BIO sustainable forestry management criteria", "required": True, "frequency": "annual", "unit": "", "category": "biodiversity", "collection_effort_hours": 5.0},
        {"field_name": "bio_dnsh_deforestation_check", "data_type": "boolean", "description": "BIO DNSH deforestation screening result", "required": True, "frequency": "annual", "unit": "", "category": "biodiversity", "collection_effort_hours": 4.0},
        {"field_name": "activity_eligibility_flag", "data_type": "boolean", "description": "Taxonomy eligible activity flag", "required": True, "frequency": "annual", "unit": "", "category": "classification", "collection_effort_hours": 3.0},
        {"field_name": "dnsh_assessment_complete", "data_type": "boolean", "description": "DNSH assessment completion for all 6 objectives", "required": True, "frequency": "annual", "unit": "", "category": "compliance", "collection_effort_hours": 12.0},
        {"field_name": "minimum_safeguards_check", "data_type": "boolean", "description": "Minimum social safeguards compliance check", "required": True, "frequency": "annual", "unit": "", "category": "social", "collection_effort_hours": 4.0},
        {"field_name": "ccm_scope1_absolute_emissions", "data_type": "numeric", "description": "CCM absolute Scope 1 emissions", "required": True, "frequency": "annual", "unit": "tCO2e", "category": "emissions", "collection_effort_hours": 6.0},
        {"field_name": "ccm_ghg_removals", "data_type": "numeric", "description": "CCM GHG removal activities", "required": False, "frequency": "annual", "unit": "tCO2e", "category": "emissions", "collection_effort_hours": 4.0},
        {"field_name": "ccm_carbon_cost_internalized", "data_type": "numeric", "description": "CCM internalized carbon cost metric", "required": False, "frequency": "annual", "unit": "EUR/tCO2e", "category": "financial", "collection_effort_hours": 3.0},
        {"field_name": "ccm_capex_plan_alignment", "data_type": "string", "description": "CCM CapEx plan alignment with CCM criteria", "required": True, "frequency": "annual", "unit": "", "category": "financial", "collection_effort_hours": 10.0},
        {"field_name": "cca_adaptation_solution_revenue", "data_type": "numeric", "description": "CCA revenue from adaptation solutions", "required": False, "frequency": "annual", "unit": "EUR", "category": "financial", "collection_effort_hours": 5.0},
        {"field_name": "cca_financial_impact_assessment", "data_type": "string", "description": "CCA financial impact of climate hazards", "required": True, "frequency": "annual", "unit": "", "category": "financial", "collection_effort_hours": 8.0},
        {"field_name": "ccm_transition_risk_exposure", "data_type": "string", "description": "CCM transition risk financial exposure", "required": True, "frequency": "annual", "unit": "", "category": "risk", "collection_effort_hours": 6.0},
        {"field_name": "gar_green_asset_ratio", "data_type": "numeric", "description": "Green Asset Ratio for financial institutions", "required": False, "frequency": "annual", "unit": "percent", "category": "financial", "collection_effort_hours": 12.0},
        {"field_name": "nuclear_gas_complementary_kpi", "data_type": "string", "description": "Nuclear/gas complementary disclosure KPIs", "required": False, "frequency": "annual", "unit": "", "category": "financial", "collection_effort_hours": 6.0},
        {"field_name": "bio_biodiversity_management_plan", "data_type": "string", "description": "BIO biodiversity management plan", "required": True, "frequency": "annual", "unit": "", "category": "biodiversity", "collection_effort_hours": 6.0},
        {"field_name": "ccm_carbon_sink_activities", "data_type": "string", "description": "CCM carbon sequestration activities", "required": False, "frequency": "annual", "unit": "", "category": "emissions", "collection_effort_hours": 4.0},
        {"field_name": "ppc_waste_water_treatment_level", "data_type": "string", "description": "PPC wastewater treatment level achieved", "required": True, "frequency": "annual", "unit": "", "category": "water", "collection_effort_hours": 3.0},
        {"field_name": "ppc_dnsh_pollution_compliance", "data_type": "boolean", "description": "PPC DNSH pollution compliance check", "required": True, "frequency": "annual", "unit": "", "category": "pollution", "collection_effort_hours": 4.0},
        {"field_name": "ccm_carbon_pricing_coverage", "data_type": "string", "description": "CCM carbon pricing coverage assessment", "required": False, "frequency": "annual", "unit": "", "category": "financial", "collection_effort_hours": 3.0},
    ],
}


# ---------------------------------------------------------------------------
# DataDeduplicationEngine
# ---------------------------------------------------------------------------


class DataDeduplicationEngine:
    """
    Data deduplication engine for EU climate regulation requirements.

    Scans data collection requirements across CSRD, CBAM, EUDR, and
    EU Taxonomy to identify duplicates, generate golden records, and
    estimate compliance effort savings.

    Attributes:
        config: Engine configuration.
        _requirements: All loaded requirements indexed by regulation.

    Example:
        >>> engine = DataDeduplicationEngine()
        >>> result = engine.scan_requirements()
        >>> assert result.duplicate_groups > 0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize DataDeduplicationEngine.

        Args:
            config: Optional configuration dictionary or DataDeduplicationConfig.
        """
        if config and isinstance(config, dict):
            self.config = DataDeduplicationConfig(**config)
        elif config and isinstance(config, DataDeduplicationConfig):
            self.config = config
        else:
            self.config = DataDeduplicationConfig()

        self._requirements: Dict[str, List[DataRequirement]] = {}
        self._load_requirements()
        logger.info(
            "DataDeduplicationEngine initialized (v%s): %d requirements loaded",
            _MODULE_VERSION, sum(len(v) for v in self._requirements.values()),
        )

    def _load_requirements(self) -> None:
        """Load all regulation data requirements into typed models."""
        for regulation, raw_list in REGULATION_DATA_REQUIREMENTS.items():
            reqs: List[DataRequirement] = []
            for raw in raw_list:
                req = DataRequirement(
                    regulation=regulation,
                    field_name=raw["field_name"],
                    data_type=raw.get("data_type", "string"),
                    description=raw.get("description", ""),
                    required=raw.get("required", True),
                    frequency=raw.get("frequency", "annual"),
                    unit=raw.get("unit", ""),
                    category=raw.get("category", ""),
                    collection_effort_hours=raw.get("collection_effort_hours", 1.0),
                )
                reqs.append(req)
            self._requirements[regulation] = reqs

    # -------------------------------------------------------------------
    # scan_requirements
    # -------------------------------------------------------------------

    def scan_requirements(
        self,
        regulations: Optional[List[str]] = None,
    ) -> DeduplicationResult:
        """Scan all requirements and identify duplicates.

        Args:
            regulations: Optional list of regulations to scan (default: all).

        Returns:
            DeduplicationResult with groups, golden records, and savings.
        """
        start_time = datetime.now(timezone.utc)
        target_regs = regulations or list(self._requirements.keys())

        all_reqs: List[DataRequirement] = []
        for reg in target_regs:
            all_reqs.extend(self._requirements.get(reg, []))

        groups = self._find_duplicate_groups(all_reqs)
        golden_records = self._generate_golden_records(groups)

        total_effort = sum(r.collection_effort_hours for r in all_reqs)
        savings = sum(g.savings_estimate_hours for g in groups)
        savings_pct = round(savings / max(total_effort, 1.0) * 100, 2)

        all_conflicts: List[MergeConflict] = []
        for group in groups:
            all_conflicts.extend(group.conflicts)

        unique_count = len(all_reqs) - sum(len(g.requirements) - 1 for g in groups)

        result = DeduplicationResult(
            total_requirements_scanned=len(all_reqs),
            unique_requirements=unique_count,
            duplicate_groups=len(groups),
            total_duplicates=sum(len(g.requirements) for g in groups),
            groups=groups,
            golden_records=golden_records,
            total_savings_hours=round(savings, 2),
            savings_percentage=savings_pct,
            conflicts=all_conflicts,
            regulations_analyzed=target_regs,
        )
        result.provenance_hash = _compute_hash(result)

        elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        logger.info(
            "Dedup scan: %d reqs, %d groups, %.1f hours savings (%.1f%%), %.1fms",
            len(all_reqs), len(groups), savings, savings_pct, elapsed_ms,
        )
        return result

    # -------------------------------------------------------------------
    # find_duplicates
    # -------------------------------------------------------------------

    def find_duplicates(
        self,
        requirements: List[DataRequirement],
    ) -> List[DeduplicationGroup]:
        """Find duplicate groups in a list of requirements.

        Args:
            requirements: List of DataRequirement objects.

        Returns:
            List of DeduplicationGroup objects.
        """
        return self._find_duplicate_groups(requirements)

    # -------------------------------------------------------------------
    # Internal: _find_duplicate_groups
    # -------------------------------------------------------------------

    def _find_duplicate_groups(
        self,
        requirements: List[DataRequirement],
    ) -> List[DeduplicationGroup]:
        """Core duplicate detection logic.

        Three-pass approach:
        1. Exact field name match
        2. Normalized field name match
        3. Fuzzy similarity match (if enabled)
        """
        groups: List[DeduplicationGroup] = []
        assigned: Set[str] = set()

        # Pass 1: Exact field name matches (cross-regulation)
        by_exact_name: Dict[str, List[DataRequirement]] = {}
        for req in requirements:
            by_exact_name.setdefault(req.field_name, []).append(req)

        for field_name, reqs in by_exact_name.items():
            regulations_seen = set(r.regulation for r in reqs)
            if len(regulations_seen) > 1:
                group = self._build_group(field_name, reqs, MatchType.EXACT_NAME, 1.0)
                groups.append(group)
                for r in reqs:
                    assigned.add(r.requirement_id)

        # Pass 2: Normalized name matches
        remaining = [r for r in requirements if r.requirement_id not in assigned]
        by_normalized: Dict[str, List[DataRequirement]] = {}
        for req in remaining:
            norm_name = _normalize_field_name(req.field_name)
            by_normalized.setdefault(norm_name, []).append(req)

        for norm_name, reqs in by_normalized.items():
            regulations_seen = set(r.regulation for r in reqs)
            if len(regulations_seen) > 1:
                group = self._build_group(norm_name, reqs, MatchType.NORMALIZED_NAME, 0.9)
                groups.append(group)
                for r in reqs:
                    assigned.add(r.requirement_id)

        # Pass 3: Fuzzy similarity matches
        if self.config.enable_fuzzy_matching:
            remaining = [r for r in requirements if r.requirement_id not in assigned]
            fuzzy_groups = self._fuzzy_match(remaining)
            for group in fuzzy_groups:
                groups.append(group)
                for r in group.requirements:
                    assigned.add(r.requirement_id)

        return groups

    def _fuzzy_match(
        self,
        requirements: List[DataRequirement],
    ) -> List[DeduplicationGroup]:
        """Perform fuzzy matching on remaining unmatched requirements."""
        groups: List[DeduplicationGroup] = []
        matched_ids: Set[str] = set()
        threshold = self.config.fuzzy_threshold

        for i, req_a in enumerate(requirements):
            if req_a.requirement_id in matched_ids:
                continue
            cluster = [req_a]
            best_similarity = 0.0
            norm_a = _normalize_field_name(req_a.field_name)

            for j in range(i + 1, len(requirements)):
                req_b = requirements[j]
                if req_b.requirement_id in matched_ids:
                    continue
                if req_a.regulation == req_b.regulation:
                    continue

                norm_b = _normalize_field_name(req_b.field_name)
                sim = _compute_similarity(norm_a, norm_b)

                if sim >= threshold and req_a.category == req_b.category:
                    cluster.append(req_b)
                    best_similarity = max(best_similarity, sim)

            if len(set(r.regulation for r in cluster)) > 1:
                canonical = _normalize_field_name(req_a.field_name)
                group = self._build_group(
                    canonical, cluster,
                    MatchType.SEMANTIC_SIMILARITY,
                    round(best_similarity, 4),
                )
                groups.append(group)
                for r in cluster:
                    matched_ids.add(r.requirement_id)

        return groups

    def _build_group(
        self,
        canonical: str,
        reqs: List[DataRequirement],
        match_type: MatchType,
        similarity: float,
    ) -> DeduplicationGroup:
        """Build a deduplication group from matched requirements."""
        regulations = sorted(set(r.regulation for r in reqs))
        conflicts = self._detect_conflicts(reqs)

        base_effort = max(r.collection_effort_hours for r in reqs)
        total_effort = sum(r.collection_effort_hours for r in reqs)
        savings = total_effort - base_effort

        return DeduplicationGroup(
            canonical_field=canonical,
            requirements=reqs,
            match_type=match_type.value,
            similarity_score=similarity,
            merge_decision="MERGED" if not conflicts else "PENDING",
            conflicts=conflicts,
            savings_estimate_hours=round(savings, 2),
            regulations_involved=regulations,
        )

    def _detect_conflicts(
        self,
        reqs: List[DataRequirement],
    ) -> List[MergeConflict]:
        """Detect merge conflicts between duplicate requirements."""
        conflicts: List[MergeConflict] = []

        data_types = {r.regulation: r.data_type for r in reqs}
        if len(set(data_types.values())) > 1:
            conflicts.append(MergeConflict(
                field_attribute="data_type",
                values_by_regulation=data_types,
                severity="HIGH",
                resolution_suggestion="Use the most specific data type",
            ))

        units = {r.regulation: r.unit for r in reqs if r.unit}
        if len(set(units.values())) > 1:
            conflicts.append(MergeConflict(
                field_attribute="unit",
                values_by_regulation=units,
                severity="MEDIUM",
                resolution_suggestion="Standardize on SI units with conversion factors",
            ))

        frequencies = {r.regulation: r.frequency for r in reqs}
        if len(set(frequencies.values())) > 1:
            conflicts.append(MergeConflict(
                field_attribute="frequency",
                values_by_regulation=frequencies,
                severity="LOW",
                resolution_suggestion="Use the highest frequency to satisfy all regulations",
            ))

        required_flags = {r.regulation: str(r.required) for r in reqs}
        if len(set(required_flags.values())) > 1:
            conflicts.append(MergeConflict(
                field_attribute="required",
                values_by_regulation=required_flags,
                severity="LOW",
                resolution_suggestion="Treat as required if mandatory in any regulation",
            ))

        return conflicts

    # -------------------------------------------------------------------
    # merge_duplicates / get_golden_record
    # -------------------------------------------------------------------

    def merge_duplicates(
        self,
        group: DeduplicationGroup,
        strategy: Optional[str] = None,
    ) -> GoldenRecord:
        """Merge a deduplication group into a golden record.

        Args:
            group: Deduplication group to merge.
            strategy: Override merge strategy (default: use config).

        Returns:
            GoldenRecord with merged data.
        """
        effective_strategy = strategy or self.config.merge_strategy
        return self._create_golden_record(group, effective_strategy)

    def get_golden_record(
        self,
        group: DeduplicationGroup,
    ) -> GoldenRecord:
        """Alias for merge_duplicates using configured strategy.

        Args:
            group: Deduplication group.

        Returns:
            GoldenRecord.
        """
        return self.merge_duplicates(group)

    def _generate_golden_records(
        self,
        groups: List[DeduplicationGroup],
    ) -> List[GoldenRecord]:
        """Generate golden records for all groups."""
        return [
            self._create_golden_record(g, self.config.merge_strategy)
            for g in groups
        ]

    def _create_golden_record(
        self,
        group: DeduplicationGroup,
        strategy: str,
    ) -> GoldenRecord:
        """Create a single golden record from a deduplication group."""
        reqs = group.requirements
        if not reqs:
            return GoldenRecord(canonical_field=group.canonical_field)

        if strategy == "HIGHEST_CONFIDENCE":
            primary = max(reqs, key=lambda r: r.confidence)
        elif strategy == "MOST_RECENT":
            primary = max(reqs, key=lambda r: r.last_updated)
        elif strategy == "FIRST_WINS":
            primary = reqs[0]
        else:
            primary = reqs[0]

        frequency_priority = {"continuous": 3, "quarterly": 2, "annual": 1}
        highest_freq = max(reqs, key=lambda r: frequency_priority.get(r.frequency, 0))

        source_fields = {r.regulation: r.field_name for r in reqs}

        record = GoldenRecord(
            canonical_field=group.canonical_field,
            data_type=primary.data_type,
            description=primary.description,
            required=any(r.required for r in reqs),
            frequency=highest_freq.frequency,
            unit=primary.unit,
            source_regulations=sorted(set(r.regulation for r in reqs)),
            source_fields=source_fields,
            merge_strategy_used=strategy,
        )
        record.provenance_hash = _compute_hash(record)
        return record

    # -------------------------------------------------------------------
    # calculate_savings
    # -------------------------------------------------------------------

    def calculate_savings(
        self,
        result: DeduplicationResult,
    ) -> Dict[str, Any]:
        """Calculate detailed savings metrics from deduplication.

        Args:
            result: DeduplicationResult from a scan.

        Returns:
            Dict with hours saved, cost saved, and breakdown.
        """
        hours_saved = result.total_savings_hours
        cost_saved = round(hours_saved * self.config.hourly_rate_eur, 2)

        by_category: Dict[str, float] = {}
        for group in result.groups:
            for req in group.requirements:
                by_category[req.category] = by_category.get(req.category, 0) + group.savings_estimate_hours / len(group.requirements)

        savings_result = {
            "total_hours_saved": hours_saved,
            "cost_saved_eur": cost_saved,
            "savings_percentage": result.savings_percentage,
            "hourly_rate_eur": self.config.hourly_rate_eur,
            "savings_by_category": {k: round(v, 2) for k, v in sorted(by_category.items(), key=lambda x: -x[1])},
            "duplicate_groups": result.duplicate_groups,
            "provenance_hash": _compute_hash({
                "hours": hours_saved,
                "cost": cost_saved,
                "pct": result.savings_percentage,
            }),
        }

        logger.info(
            "Savings calculation: %.1f hours, EUR %.2f (%.1f%%)",
            hours_saved, cost_saved, result.savings_percentage,
        )
        return savings_result

    # -------------------------------------------------------------------
    # generate_dedup_report
    # -------------------------------------------------------------------

    def generate_dedup_report(
        self,
        result: DeduplicationResult,
    ) -> DedupReport:
        """Generate a human-readable deduplication report.

        Args:
            result: DeduplicationResult from a scan.

        Returns:
            DedupReport with summary, breakdown, and recommendations.
        """
        breakdown: Dict[str, Dict[str, int]] = {}
        for reg in result.regulations_analyzed:
            reg_reqs = self._requirements.get(reg, [])
            reg_in_groups = sum(
                1 for g in result.groups
                for r in g.requirements
                if r.regulation == reg
            )
            breakdown[reg] = {
                "total_fields": len(reg_reqs),
                "in_duplicate_groups": reg_in_groups,
                "unique_only": len(reg_reqs) - reg_in_groups,
            }

        top_savings = sorted(
            result.groups, key=lambda g: g.savings_estimate_hours, reverse=True
        )[:10]
        top_savings_list = [
            {
                "canonical_field": g.canonical_field,
                "regulations": g.regulations_involved,
                "savings_hours": g.savings_estimate_hours,
                "match_type": g.match_type,
                "similarity": g.similarity_score,
            }
            for g in top_savings
        ]

        recommendations = [
            f"Consolidate {result.duplicate_groups} duplicate field groups into golden records",
            f"Estimated savings: {result.total_savings_hours:.1f} hours per reporting cycle",
        ]
        if result.conflicts:
            recommendations.append(
                f"Resolve {len(result.conflicts)} merge conflicts before finalizing golden records"
            )
        recommendations.append(
            "Implement a centralized data collection platform to avoid re-collection"
        )
        recommendations.append(
            "Align collection frequencies to the highest requirement across regulations"
        )

        summary = (
            f"Scanned {result.total_requirements_scanned} data requirements across "
            f"{len(result.regulations_analyzed)} regulations. Identified "
            f"{result.duplicate_groups} duplicate groups with "
            f"{result.total_savings_hours:.1f} hours potential savings "
            f"({result.savings_percentage:.1f}% reduction)."
        )

        report = DedupReport(
            summary=summary,
            regulation_breakdown=breakdown,
            top_savings_opportunities=top_savings_list,
            recommendations=recommendations,
        )
        report.provenance_hash = _compute_hash(report)

        logger.info("Generated dedup report: %s", summary)
        return report
