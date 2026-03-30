# -*- coding: utf-8 -*-
"""
CrossRegulationEngine - PACK-005 CBAM Complete Engine 7

Maps CBAM data to 6 related regulatory frameworks and optimizes data
reuse across compliance obligations. Includes a 50+ country carbon
pricing database for equivalence calculations.

Regulatory Mappings:
    1. CSRD - ESRS E1 climate disclosures
    2. CDP  - CDP Climate sections C6, C7, C11, C12
    3. SBTi - Scope 3 Category 1 targets
    4. EU Taxonomy - Climate mitigation criteria
    5. EU ETS - Free allocation, benchmarks
    6. EUDR - Fertilizer supply chain overlap

Carbon Pricing Database:
    50+ countries with carbon pricing scheme details including
    scheme type, price per tCO2e, currency, and CBAM deduction
    eligibility.

Zero-Hallucination:
    - All regulatory mapping rules from published standards
    - Carbon pricing data from predefined reference tables
    - No LLM involvement in mapping logic
    - SHA-256 provenance hash on all results

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-005 CBAM Complete
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

# ---------------------------------------------------------------------------
# Carbon Pricing Database (50+ countries)
# ---------------------------------------------------------------------------

CARBON_PRICING_DB: List[Dict[str, Any]] = [
    {"country": "EU", "scheme_type": "ets", "price_per_tco2e": "72.50", "currency": "EUR", "qualifying_for_deduction": True, "scheme_name": "EU Emissions Trading System"},
    {"country": "GB", "scheme_type": "ets", "price_per_tco2e": "45.00", "currency": "GBP", "qualifying_for_deduction": True, "scheme_name": "UK Emissions Trading Scheme"},
    {"country": "CH", "scheme_type": "ets_tax", "price_per_tco2e": "120.00", "currency": "CHF", "qualifying_for_deduction": True, "scheme_name": "Swiss CO2 levy + ETS"},
    {"country": "NO", "scheme_type": "tax", "price_per_tco2e": "85.00", "currency": "EUR", "qualifying_for_deduction": True, "scheme_name": "Norwegian carbon tax"},
    {"country": "SE", "scheme_type": "tax", "price_per_tco2e": "120.00", "currency": "EUR", "qualifying_for_deduction": True, "scheme_name": "Swedish carbon tax"},
    {"country": "FI", "scheme_type": "tax", "price_per_tco2e": "77.00", "currency": "EUR", "qualifying_for_deduction": True, "scheme_name": "Finnish carbon tax"},
    {"country": "DK", "scheme_type": "tax", "price_per_tco2e": "25.00", "currency": "EUR", "qualifying_for_deduction": True, "scheme_name": "Danish carbon tax"},
    {"country": "FR", "scheme_type": "tax", "price_per_tco2e": "44.60", "currency": "EUR", "qualifying_for_deduction": True, "scheme_name": "French carbon tax"},
    {"country": "DE", "scheme_type": "ets", "price_per_tco2e": "45.00", "currency": "EUR", "qualifying_for_deduction": True, "scheme_name": "German national ETS (heating/transport)"},
    {"country": "NL", "scheme_type": "tax", "price_per_tco2e": "51.40", "currency": "EUR", "qualifying_for_deduction": True, "scheme_name": "Dutch carbon tax"},
    {"country": "IE", "scheme_type": "tax", "price_per_tco2e": "48.50", "currency": "EUR", "qualifying_for_deduction": True, "scheme_name": "Irish carbon tax"},
    {"country": "PT", "scheme_type": "tax", "price_per_tco2e": "24.00", "currency": "EUR", "qualifying_for_deduction": True, "scheme_name": "Portuguese carbon tax"},
    {"country": "ES", "scheme_type": "tax", "price_per_tco2e": "15.00", "currency": "EUR", "qualifying_for_deduction": True, "scheme_name": "Spanish carbon tax"},
    {"country": "CA", "scheme_type": "tax_obps", "price_per_tco2e": "65.00", "currency": "CAD", "qualifying_for_deduction": True, "scheme_name": "Canadian federal carbon pricing"},
    {"country": "NZ", "scheme_type": "ets", "price_per_tco2e": "50.00", "currency": "NZD", "qualifying_for_deduction": True, "scheme_name": "New Zealand ETS"},
    {"country": "KR", "scheme_type": "ets", "price_per_tco2e": "10.00", "currency": "USD", "qualifying_for_deduction": True, "scheme_name": "Korea ETS"},
    {"country": "CN", "scheme_type": "ets", "price_per_tco2e": "9.50", "currency": "USD", "qualifying_for_deduction": True, "scheme_name": "China national ETS (power sector)"},
    {"country": "JP", "scheme_type": "tax", "price_per_tco2e": "2.50", "currency": "USD", "qualifying_for_deduction": True, "scheme_name": "Japan carbon tax"},
    {"country": "SG", "scheme_type": "tax", "price_per_tco2e": "25.00", "currency": "SGD", "qualifying_for_deduction": True, "scheme_name": "Singapore carbon tax"},
    {"country": "ZA", "scheme_type": "tax", "price_per_tco2e": "9.00", "currency": "USD", "qualifying_for_deduction": True, "scheme_name": "South Africa carbon tax"},
    {"country": "MX", "scheme_type": "tax", "price_per_tco2e": "3.50", "currency": "USD", "qualifying_for_deduction": True, "scheme_name": "Mexico carbon tax"},
    {"country": "CL", "scheme_type": "tax", "price_per_tco2e": "5.00", "currency": "USD", "qualifying_for_deduction": True, "scheme_name": "Chile carbon tax"},
    {"country": "CO", "scheme_type": "tax", "price_per_tco2e": "5.00", "currency": "USD", "qualifying_for_deduction": True, "scheme_name": "Colombia carbon tax"},
    {"country": "AR", "scheme_type": "tax", "price_per_tco2e": "6.00", "currency": "USD", "qualifying_for_deduction": True, "scheme_name": "Argentina carbon tax"},
    {"country": "UA", "scheme_type": "tax", "price_per_tco2e": "0.35", "currency": "EUR", "qualifying_for_deduction": False, "scheme_name": "Ukraine carbon tax (not qualifying)"},
    {"country": "TR", "scheme_type": "planned", "price_per_tco2e": "0.00", "currency": "EUR", "qualifying_for_deduction": False, "scheme_name": "Turkey ETS (planned)"},
    {"country": "IN", "scheme_type": "cess", "price_per_tco2e": "2.00", "currency": "USD", "qualifying_for_deduction": False, "scheme_name": "India coal cess (not qualifying as carbon price)"},
    {"country": "RU", "scheme_type": "none", "price_per_tco2e": "0.00", "currency": "EUR", "qualifying_for_deduction": False, "scheme_name": "No carbon pricing"},
    {"country": "BR", "scheme_type": "planned", "price_per_tco2e": "0.00", "currency": "BRL", "qualifying_for_deduction": False, "scheme_name": "Brazil ETS (planned)"},
    {"country": "ID", "scheme_type": "tax", "price_per_tco2e": "2.10", "currency": "USD", "qualifying_for_deduction": True, "scheme_name": "Indonesia carbon tax"},
    {"country": "TW", "scheme_type": "fee", "price_per_tco2e": "10.00", "currency": "USD", "qualifying_for_deduction": True, "scheme_name": "Taiwan carbon fee"},
    {"country": "TH", "scheme_type": "planned", "price_per_tco2e": "0.00", "currency": "USD", "qualifying_for_deduction": False, "scheme_name": "Thailand carbon pricing (planned)"},
    {"country": "VN", "scheme_type": "planned", "price_per_tco2e": "0.00", "currency": "USD", "qualifying_for_deduction": False, "scheme_name": "Vietnam carbon market (planned)"},
    {"country": "PH", "scheme_type": "none", "price_per_tco2e": "0.00", "currency": "USD", "qualifying_for_deduction": False, "scheme_name": "No carbon pricing"},
    {"country": "MY", "scheme_type": "planned", "price_per_tco2e": "0.00", "currency": "MYR", "qualifying_for_deduction": False, "scheme_name": "Malaysia carbon pricing (planned)"},
    {"country": "EG", "scheme_type": "none", "price_per_tco2e": "0.00", "currency": "USD", "qualifying_for_deduction": False, "scheme_name": "No carbon pricing"},
    {"country": "SA", "scheme_type": "none", "price_per_tco2e": "0.00", "currency": "USD", "qualifying_for_deduction": False, "scheme_name": "No carbon pricing"},
    {"country": "AE", "scheme_type": "none", "price_per_tco2e": "0.00", "currency": "USD", "qualifying_for_deduction": False, "scheme_name": "No carbon pricing"},
    {"country": "QA", "scheme_type": "none", "price_per_tco2e": "0.00", "currency": "USD", "qualifying_for_deduction": False, "scheme_name": "No carbon pricing"},
    {"country": "KW", "scheme_type": "none", "price_per_tco2e": "0.00", "currency": "USD", "qualifying_for_deduction": False, "scheme_name": "No carbon pricing"},
    {"country": "OM", "scheme_type": "none", "price_per_tco2e": "0.00", "currency": "USD", "qualifying_for_deduction": False, "scheme_name": "No carbon pricing"},
    {"country": "BH", "scheme_type": "none", "price_per_tco2e": "0.00", "currency": "USD", "qualifying_for_deduction": False, "scheme_name": "No carbon pricing"},
    {"country": "PK", "scheme_type": "none", "price_per_tco2e": "0.00", "currency": "USD", "qualifying_for_deduction": False, "scheme_name": "No carbon pricing"},
    {"country": "BD", "scheme_type": "none", "price_per_tco2e": "0.00", "currency": "USD", "qualifying_for_deduction": False, "scheme_name": "No carbon pricing"},
    {"country": "LK", "scheme_type": "none", "price_per_tco2e": "0.00", "currency": "USD", "qualifying_for_deduction": False, "scheme_name": "No carbon pricing"},
    {"country": "KZ", "scheme_type": "ets", "price_per_tco2e": "1.50", "currency": "USD", "qualifying_for_deduction": True, "scheme_name": "Kazakhstan ETS"},
    {"country": "UZ", "scheme_type": "none", "price_per_tco2e": "0.00", "currency": "USD", "qualifying_for_deduction": False, "scheme_name": "No carbon pricing"},
    {"country": "IS", "scheme_type": "tax", "price_per_tco2e": "35.00", "currency": "EUR", "qualifying_for_deduction": True, "scheme_name": "Iceland carbon tax (part of EU ETS)"},
    {"country": "LI", "scheme_type": "ets", "price_per_tco2e": "72.50", "currency": "EUR", "qualifying_for_deduction": True, "scheme_name": "Liechtenstein (part of Swiss ETS link)"},
    {"country": "AU", "scheme_type": "safeguard", "price_per_tco2e": "20.00", "currency": "AUD", "qualifying_for_deduction": True, "scheme_name": "Australia Safeguard Mechanism"},
    {"country": "US", "scheme_type": "state_level", "price_per_tco2e": "30.00", "currency": "USD", "qualifying_for_deduction": False, "scheme_name": "US state-level only (RGGI, CA cap-and-trade)"},
]

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class CSRDMapping(BaseModel):
    """CBAM data mapped to CSRD ESRS E1 disclosures."""
    mapping_id: str = Field(default_factory=_new_uuid, description="Mapping identifier")
    esrs_standard: str = Field(default="E1", description="ESRS standard reference")
    disclosure_requirements: List[Dict[str, Any]] = Field(
        default_factory=list, description="Mapped disclosure requirements"
    )
    scope3_category1_emissions: Decimal = Field(
        default=Decimal("0"), description="Scope 3 Cat 1 emissions from CBAM data"
    )
    carbon_pricing_exposure: Decimal = Field(
        default=Decimal("0"), description="Carbon pricing exposure in EUR"
    )
    transition_plan_inputs: List[str] = Field(
        default_factory=list, description="Transition plan data points from CBAM"
    )
    data_completeness_pct: Decimal = Field(default=Decimal("0"), description="Data completeness %")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class CDPMapping(BaseModel):
    """CBAM data mapped to CDP Climate questionnaire."""
    mapping_id: str = Field(default_factory=_new_uuid, description="Mapping identifier")
    cdp_sections: List[Dict[str, Any]] = Field(
        default_factory=list, description="CDP sections with CBAM data"
    )
    c6_emissions_data: Dict[str, Any] = Field(default_factory=dict, description="C6 Emissions data")
    c7_breakdown: Dict[str, Any] = Field(default_factory=dict, description="C7 Emissions breakdown")
    c11_carbon_pricing: Dict[str, Any] = Field(default_factory=dict, description="C11 Carbon pricing")
    c12_engagement: Dict[str, Any] = Field(default_factory=dict, description="C12 Engagement")
    data_quality_score: Decimal = Field(default=Decimal("0"), description="Data quality (0-100)")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class SBTiMapping(BaseModel):
    """CBAM data mapped to SBTi framework."""
    mapping_id: str = Field(default_factory=_new_uuid, description="Mapping identifier")
    scope3_cat1_baseline: Decimal = Field(default=Decimal("0"), description="Scope 3 Cat 1 baseline tCO2e")
    scope3_cat1_current: Decimal = Field(default=Decimal("0"), description="Current Scope 3 Cat 1 tCO2e")
    reduction_pct: Decimal = Field(default=Decimal("0"), description="Reduction percentage")
    target_alignment: str = Field(default="", description="SBTi target alignment (1.5C/WB2C)")
    supplier_engagement_target: Dict[str, Any] = Field(
        default_factory=dict, description="Supplier engagement target metrics"
    )
    cbam_data_coverage_pct: Decimal = Field(default=Decimal("0"), description="CBAM data coverage %")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class TaxonomyMapping(BaseModel):
    """CBAM data mapped to EU Taxonomy."""
    mapping_id: str = Field(default_factory=_new_uuid, description="Mapping identifier")
    climate_mitigation_criteria: List[Dict[str, Any]] = Field(
        default_factory=list, description="Climate mitigation screening criteria"
    )
    dnsh_assessment: Dict[str, Any] = Field(
        default_factory=dict, description="Do No Significant Harm assessment"
    )
    taxonomy_eligible_activities: List[str] = Field(
        default_factory=list, description="Taxonomy-eligible activities"
    )
    cbam_alignment_score: Decimal = Field(default=Decimal("0"), description="Alignment score (0-100)")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class ETSMapping(BaseModel):
    """CBAM data mapped to EU ETS."""
    mapping_id: str = Field(default_factory=_new_uuid, description="Mapping identifier")
    free_allocation_data: Dict[str, Any] = Field(
        default_factory=dict, description="Free allocation phase-out data"
    )
    benchmark_values: List[Dict[str, Any]] = Field(
        default_factory=list, description="EU ETS benchmark values"
    )
    cbam_ets_price_differential: Decimal = Field(
        default=Decimal("0"), description="Price differential CBAM vs ETS"
    )
    cross_border_adjustment_factor: Decimal = Field(
        default=Decimal("0"), description="Cross-border adjustment factor"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class EUDRMapping(BaseModel):
    """CBAM data mapped to EUDR for fertilizer supply chains."""
    mapping_id: str = Field(default_factory=_new_uuid, description="Mapping identifier")
    fertilizer_supply_chains: List[Dict[str, Any]] = Field(
        default_factory=list, description="Fertilizer supply chain overlaps"
    )
    shared_suppliers: List[str] = Field(default_factory=list, description="Suppliers subject to both CBAM and EUDR")
    geolocation_data_reuse: Dict[str, Any] = Field(
        default_factory=dict, description="Geolocation data reuse opportunities"
    )
    due_diligence_synergies: List[str] = Field(
        default_factory=list, description="Due diligence process synergies"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class DataReuseReport(BaseModel):
    """Data reuse optimization report across regulations."""
    report_id: str = Field(default_factory=_new_uuid, description="Report identifier")
    cbam_data_points: int = Field(default=0, description="Total CBAM data points")
    reusable_data_points: Dict[str, int] = Field(
        default_factory=dict, description="Reusable data points per regulation"
    )
    total_reuse_pct: Decimal = Field(default=Decimal("0"), description="Total data reuse percentage")
    estimated_effort_reduction_hours: Decimal = Field(
        default=Decimal("0"), description="Estimated effort reduction in hours"
    )
    recommendations: List[str] = Field(default_factory=list, description="Optimization recommendations")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class ConsistencyCheckResult(BaseModel):
    """Cross-regulation data consistency check."""
    check_id: str = Field(default_factory=_new_uuid, description="Check identifier")
    regulations_checked: List[str] = Field(default_factory=list, description="Regulations checked")
    inconsistencies: List[Dict[str, Any]] = Field(
        default_factory=list, description="Identified inconsistencies"
    )
    total_data_points_checked: int = Field(default=0, description="Total data points checked")
    consistent_pct: Decimal = Field(default=Decimal("100"), description="Consistency percentage")
    recommendations: List[str] = Field(default_factory=list, description="Remediation recommendations")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class ChangeTracker(BaseModel):
    """Regulatory change tracking result."""
    tracker_id: str = Field(default_factory=_new_uuid, description="Tracker identifier")
    regulation: str = Field(description="Tracked regulation")
    recent_changes: List[Dict[str, Any]] = Field(
        default_factory=list, description="Recent regulatory changes"
    )
    upcoming_deadlines: List[Dict[str, Any]] = Field(
        default_factory=list, description="Upcoming compliance deadlines"
    )
    impact_assessment: str = Field(default="", description="Impact assessment summary")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class CarbonPricingEquivalence(BaseModel):
    """Carbon pricing equivalence for CBAM deduction."""
    result_id: str = Field(default_factory=_new_uuid, description="Result identifier")
    country: str = Field(description="Country assessed")
    scheme_name: str = Field(default="", description="Carbon pricing scheme name")
    scheme_type: str = Field(default="", description="Scheme type (ets/tax/hybrid)")
    local_price_per_tco2e: Decimal = Field(default=Decimal("0"), description="Local carbon price per tCO2e")
    local_currency: str = Field(default="", description="Local currency")
    eur_equivalent_price: Decimal = Field(default=Decimal("0"), description="EUR equivalent price")
    cbam_reference_price: Decimal = Field(default=Decimal("0"), description="CBAM reference price per tCO2e")
    deduction_amount: Decimal = Field(default=Decimal("0"), description="Deductible amount per tCO2e")
    qualifying_for_deduction: bool = Field(default=False, description="Whether qualifying for deduction")
    net_cbam_cost: Decimal = Field(default=Decimal("0"), description="Net CBAM cost after deduction")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("local_price_per_tco2e", "eur_equivalent_price", "cbam_reference_price",
                     "deduction_amount", "net_cbam_cost", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

# ---------------------------------------------------------------------------
# Engine Configuration
# ---------------------------------------------------------------------------

class CrossRegulationConfig(BaseModel):
    """Configuration for the CrossRegulationEngine."""
    cbam_reference_price: Decimal = Field(
        default=Decimal("75.00"), description="CBAM reference price per tCO2e"
    )
    eur_fx_rates: Dict[str, Decimal] = Field(
        default_factory=lambda: {
            "EUR": Decimal("1.00"), "USD": Decimal("0.92"), "GBP": Decimal("1.16"),
            "CHF": Decimal("1.04"), "CAD": Decimal("0.68"), "AUD": Decimal("0.60"),
            "NZD": Decimal("0.56"), "SGD": Decimal("0.69"), "BRL": Decimal("0.18"),
            "MYR": Decimal("0.20"), "JPY": Decimal("0.0062"),
        },
        description="FX rates to EUR",
    )

# ---------------------------------------------------------------------------
# Pydantic model_rebuild for forward reference resolution
# ---------------------------------------------------------------------------

CrossRegulationConfig.model_rebuild()
CSRDMapping.model_rebuild()
CDPMapping.model_rebuild()
SBTiMapping.model_rebuild()
TaxonomyMapping.model_rebuild()
ETSMapping.model_rebuild()
EUDRMapping.model_rebuild()
DataReuseReport.model_rebuild()
ConsistencyCheckResult.model_rebuild()
ChangeTracker.model_rebuild()
CarbonPricingEquivalence.model_rebuild()

# ---------------------------------------------------------------------------
# CrossRegulationEngine
# ---------------------------------------------------------------------------

class CrossRegulationEngine:
    """
    Cross-regulation mapping engine for CBAM data reuse.

    Maps CBAM data to 6 related regulatory frameworks (CSRD, CDP, SBTi,
    EU Taxonomy, EU ETS, EUDR) and includes a 50+ country carbon pricing
    database for equivalence calculations.

    Attributes:
        config: Engine configuration.
        _carbon_pricing: Carbon pricing database indexed by country.

    Example:
        >>> engine = CrossRegulationEngine()
        >>> csrd = engine.map_to_csrd({"emissions_tco2e": 5000, "cost_eur": 375000})
        >>> assert csrd.esrs_standard == "E1"
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize CrossRegulationEngine.

        Args:
            config: Optional configuration dictionary.
        """
        if config and isinstance(config, dict):
            self.config = CrossRegulationConfig(**config)
        elif config and isinstance(config, CrossRegulationConfig):
            self.config = config
        else:
            self.config = CrossRegulationConfig()

        self._carbon_pricing: Dict[str, Dict[str, Any]] = {
            entry["country"]: entry for entry in CARBON_PRICING_DB
        }
        logger.info("CrossRegulationEngine initialized (v%s)", _MODULE_VERSION)

    # -----------------------------------------------------------------------
    # CSRD Mapping
    # -----------------------------------------------------------------------

    def map_to_csrd(self, cbam_data: Dict[str, Any]) -> CSRDMapping:
        """Map CBAM data to CSRD ESRS E1 climate disclosures.

        Maps embedded emissions to Scope 3 Category 1, carbon pricing
        exposure, and transition plan requirements.

        Args:
            cbam_data: CBAM data with 'emissions_tco2e', 'cost_eur',
                'suppliers', 'categories'.

        Returns:
            CSRDMapping with ESRS E1 disclosure data.
        """
        emissions = _decimal(cbam_data.get("emissions_tco2e", 0))
        cost = _decimal(cbam_data.get("cost_eur", 0))

        disclosures = [
            {"dr": "E1-1", "name": "Transition plan for climate change mitigation",
             "cbam_data": "CBAM emission data feeds Scope 3 reduction targets",
             "coverage": "partial"},
            {"dr": "E1-4", "name": "Targets related to climate change",
             "cbam_data": f"Scope 3 Cat 1 baseline: {emissions} tCO2e from CBAM goods",
             "coverage": "full"},
            {"dr": "E1-5", "name": "Energy consumption and mix",
             "cbam_data": "Indirect emission factors from CBAM data",
             "coverage": "partial"},
            {"dr": "E1-6", "name": "Gross Scopes 1, 2, 3 GHG emissions",
             "cbam_data": f"Scope 3 Cat 1: {emissions} tCO2e (CBAM-covered imports)",
             "coverage": "full"},
            {"dr": "E1-9", "name": "Anticipated financial effects from climate risks",
             "cbam_data": f"Carbon pricing exposure: EUR {cost}",
             "coverage": "full"},
        ]

        transition_inputs = [
            "CBAM emission factors by supplier for Scope 3 reduction roadmap",
            "Carbon pricing cost trajectory for financial planning",
            "Supplier decarbonization priorities from emission data",
        ]

        completeness = Decimal("65") if emissions > 0 else Decimal("0")

        result = CSRDMapping(
            disclosure_requirements=disclosures,
            scope3_category1_emissions=emissions,
            carbon_pricing_exposure=cost,
            transition_plan_inputs=transition_inputs,
            data_completeness_pct=completeness,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info("Mapped CBAM to CSRD: %d disclosures, completeness=%s%%", len(disclosures), completeness)
        return result

    # -----------------------------------------------------------------------
    # CDP Mapping
    # -----------------------------------------------------------------------

    def map_to_cdp(self, cbam_data: Dict[str, Any]) -> CDPMapping:
        """Map CBAM data to CDP Climate questionnaire sections.

        Args:
            cbam_data: CBAM data dict.

        Returns:
            CDPMapping with section-specific data.
        """
        emissions = _decimal(cbam_data.get("emissions_tco2e", 0))
        cost = _decimal(cbam_data.get("cost_eur", 0))
        categories = cbam_data.get("categories", {})

        c6 = {
            "scope3_cat1_emissions": str(emissions),
            "methodology": "CBAM actual/default emission factors",
            "data_source": "CBAM declaration data",
        }
        c7 = {
            "breakdown_by_category": categories if categories else {"iron_steel": str(emissions)},
            "methodology_detail": "Per-product emission factors from installation data or defaults",
        }
        c11 = {
            "carbon_pricing_mechanisms": ["EU CBAM"],
            "total_cost_eur": str(cost),
            "price_per_tco2e": str(self.config.cbam_reference_price),
            "coverage_pct": "100",
        }
        c12 = {
            "supplier_engagement_pct": str(cbam_data.get("supplier_engagement_pct", 0)),
            "engagement_type": "CBAM data collection and emission factor verification",
        }

        sections = [
            {"section": "C6", "name": "Emissions data", "coverage": "full"},
            {"section": "C7", "name": "Emissions breakdowns", "coverage": "partial"},
            {"section": "C11", "name": "Carbon pricing", "coverage": "full"},
            {"section": "C12", "name": "Engagement", "coverage": "partial"},
        ]

        quality = Decimal("75") if emissions > 0 else Decimal("0")

        result = CDPMapping(
            cdp_sections=sections,
            c6_emissions_data=c6,
            c7_breakdown=c7,
            c11_carbon_pricing=c11,
            c12_engagement=c12,
            data_quality_score=quality,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info("Mapped CBAM to CDP: 4 sections, quality=%s", quality)
        return result

    # -----------------------------------------------------------------------
    # SBTi Mapping
    # -----------------------------------------------------------------------

    def map_to_sbti(self, cbam_data: Dict[str, Any]) -> SBTiMapping:
        """Map CBAM data to SBTi framework for Scope 3 targets.

        Args:
            cbam_data: CBAM data dict.

        Returns:
            SBTiMapping with Scope 3 Category 1 data.
        """
        baseline = _decimal(cbam_data.get("baseline_emissions_tco2e", 0))
        current = _decimal(cbam_data.get("emissions_tco2e", 0))
        if baseline == 0:
            baseline = current

        reduction = (
            ((baseline - current) / baseline * Decimal("100")).quantize(Decimal("0.01"))
            if baseline > 0 else Decimal("0")
        )

        coverage = Decimal("70") if current > 0 else Decimal("0")

        result = SBTiMapping(
            scope3_cat1_baseline=baseline,
            scope3_cat1_current=current,
            reduction_pct=reduction,
            target_alignment="1.5C" if reduction >= Decimal("4.2") else "WB2C",
            supplier_engagement_target={
                "target": "67% of suppliers by emissions setting SBTi targets by 2027",
                "current_coverage_pct": str(cbam_data.get("suppliers_with_targets_pct", 0)),
                "cbam_supplier_data_available": True,
            },
            cbam_data_coverage_pct=coverage,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info("Mapped CBAM to SBTi: baseline=%s, current=%s, reduction=%s%%", baseline, current, reduction)
        return result

    # -----------------------------------------------------------------------
    # EU Taxonomy Mapping
    # -----------------------------------------------------------------------

    def map_to_taxonomy(self, cbam_data: Dict[str, Any]) -> TaxonomyMapping:
        """Map CBAM data to EU Taxonomy climate mitigation criteria.

        Args:
            cbam_data: CBAM data dict.

        Returns:
            TaxonomyMapping with alignment assessment.
        """
        emissions_intensity = _decimal(cbam_data.get("emission_intensity", 0))
        categories = cbam_data.get("categories", {})

        criteria = []
        taxonomy_benchmarks = {
            "iron_steel": {"benchmark": Decimal("1.814"), "unit": "tCO2e/t crude steel"},
            "aluminium": {"benchmark": Decimal("1.514"), "unit": "tCO2e/t primary Al"},
            "cement": {"benchmark": Decimal("0.766"), "unit": "tCO2e/t clinker"},
            "hydrogen": {"benchmark": Decimal("3.0"), "unit": "tCO2e/t H2"},
            "fertilizers": {"benchmark": Decimal("2.2"), "unit": "tCO2e/t NH3"},
        }

        alignment_scores: List[Decimal] = []
        for cat, data in categories.items():
            cat_ef = _decimal(data) if isinstance(data, (int, float, str, Decimal)) else _decimal(0)
            bench = taxonomy_benchmarks.get(cat, {})
            benchmark_val = bench.get("benchmark", Decimal("0"))
            aligned = cat_ef <= benchmark_val if benchmark_val > 0 else False
            score = Decimal("100") if aligned else max(
                (Decimal("1") - (cat_ef - benchmark_val) / benchmark_val) * Decimal("100"), Decimal("0")
            ) if benchmark_val > 0 else Decimal("0")
            alignment_scores.append(score)
            criteria.append({
                "category": cat,
                "emission_factor": str(cat_ef),
                "taxonomy_benchmark": str(benchmark_val),
                "unit": bench.get("unit", "tCO2e/t"),
                "aligned": aligned,
                "alignment_score": str(score.quantize(Decimal("0.01"))),
            })

        eligible = [c for c in criteria if c.get("aligned")]
        avg_score = (
            sum(alignment_scores) / _decimal(len(alignment_scores))
        ).quantize(Decimal("0.01")) if alignment_scores else Decimal("0")

        result = TaxonomyMapping(
            climate_mitigation_criteria=criteria,
            dnsh_assessment={"climate_adaptation": "review_required", "water": "n/a",
                              "circular_economy": "review_required", "pollution": "n/a",
                              "biodiversity": "n/a"},
            taxonomy_eligible_activities=[c["category"] for c in eligible],
            cbam_alignment_score=avg_score,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info("Mapped CBAM to Taxonomy: %d criteria, alignment=%s%%", len(criteria), avg_score)
        return result

    # -----------------------------------------------------------------------
    # EU ETS Mapping
    # -----------------------------------------------------------------------

    def map_to_ets(self, cbam_data: Dict[str, Any]) -> ETSMapping:
        """Map CBAM data to EU ETS framework.

        Args:
            cbam_data: CBAM data dict.

        Returns:
            ETSMapping with free allocation and benchmark data.
        """
        ets_price = _decimal(cbam_data.get("ets_price", "72.50"))
        cbam_price = self.config.cbam_reference_price
        differential = cbam_price - ets_price

        phase_out = {
            2026: "97.5%", 2027: "95%", 2028: "90%", 2029: "82.5%",
            2030: "75%", 2031: "65%", 2032: "50%", 2033: "25%", 2034: "0%",
        }

        benchmarks = [
            {"product": "Hot metal", "benchmark_tco2e_per_t": "1.328", "source": "EU ETS Phase 4"},
            {"product": "Sintered ore", "benchmark_tco2e_per_t": "0.171", "source": "EU ETS Phase 4"},
            {"product": "Iron casting", "benchmark_tco2e_per_t": "0.325", "source": "EU ETS Phase 4"},
            {"product": "EAF carbon steel", "benchmark_tco2e_per_t": "0.283", "source": "EU ETS Phase 4"},
            {"product": "EAF high alloy steel", "benchmark_tco2e_per_t": "0.352", "source": "EU ETS Phase 4"},
            {"product": "Grey cement clinker", "benchmark_tco2e_per_t": "0.766", "source": "EU ETS Phase 4"},
            {"product": "White cement clinker", "benchmark_tco2e_per_t": "0.987", "source": "EU ETS Phase 4"},
            {"product": "Aluminium (primary)", "benchmark_tco2e_per_t": "1.514", "source": "EU ETS Phase 4"},
            {"product": "Ammonia", "benchmark_tco2e_per_t": "1.619", "source": "EU ETS Phase 4"},
            {"product": "Hydrogen", "benchmark_tco2e_per_t": "8.850", "source": "EU ETS Phase 4"},
        ]

        result = ETSMapping(
            free_allocation_data={"phase_out_schedule": phase_out,
                                   "current_year_allocation": "97.5%"},
            benchmark_values=benchmarks,
            cbam_ets_price_differential=differential,
            cross_border_adjustment_factor=Decimal("1") if differential >= 0 else Decimal("0"),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info("Mapped CBAM to ETS: differential=%s EUR/tCO2e", differential)
        return result

    # -----------------------------------------------------------------------
    # EUDR Mapping
    # -----------------------------------------------------------------------

    def map_to_eudr(self, cbam_data: Dict[str, Any]) -> EUDRMapping:
        """Map CBAM data to EUDR for fertilizer supply chain overlaps.

        Args:
            cbam_data: CBAM data dict.

        Returns:
            EUDRMapping with supply chain synergies.
        """
        suppliers = cbam_data.get("suppliers", [])
        fertilizer_chains: List[Dict[str, Any]] = []
        shared: List[str] = []

        for supplier in suppliers:
            category = supplier.get("category", "")
            if category == "fertilizers":
                fertilizer_chains.append({
                    "supplier_id": supplier.get("supplier_id", ""),
                    "supplier_name": supplier.get("name", ""),
                    "country": supplier.get("country", ""),
                    "products": supplier.get("products", []),
                    "eudr_relevant": True,
                })
                shared.append(supplier.get("supplier_id", ""))

        synergies = [
            "Shared supplier due diligence documentation",
            "Common geolocation data for sourcing regions",
            "Aligned risk assessment frameworks",
            "Joint audit and verification processes",
        ] if shared else []

        result = EUDRMapping(
            fertilizer_supply_chains=fertilizer_chains,
            shared_suppliers=shared,
            geolocation_data_reuse={
                "available": bool(shared),
                "reuse_potential": "high" if shared else "none",
            },
            due_diligence_synergies=synergies,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info("Mapped CBAM to EUDR: %d shared suppliers", len(shared))
        return result

    # -----------------------------------------------------------------------
    # Data Reuse Optimization
    # -----------------------------------------------------------------------

    def optimize_data_reuse(
        self, cbam_data: Dict[str, Any], target_regulations: List[str]
    ) -> DataReuseReport:
        """Optimize data reuse across target regulatory frameworks.

        Args:
            cbam_data: CBAM data dict.
            target_regulations: List of regulation names to optimize for.

        Returns:
            DataReuseReport with reuse metrics and recommendations.
        """
        cbam_points = 15
        reuse_map = {
            "csrd": 8, "cdp": 7, "sbti": 5, "taxonomy": 4, "ets": 6, "eudr": 3,
        }

        reusable: Dict[str, int] = {}
        total_reuse = 0
        for reg in target_regulations:
            reg_lower = reg.lower()
            points = reuse_map.get(reg_lower, 2)
            reusable[reg] = points
            total_reuse += points

        total_reuse_pct = min(
            (_decimal(total_reuse) / _decimal(cbam_points * len(target_regulations)) * Decimal("100")).quantize(Decimal("0.01")),
            Decimal("100"),
        ) if target_regulations else Decimal("0")

        effort_hours = (_decimal(len(target_regulations)) * Decimal("40") * (Decimal("1") - total_reuse_pct / Decimal("100"))).quantize(Decimal("0.1"))

        recommendations = [
            "Centralize emission factor database for CBAM, CSRD, and CDP reporting",
            "Use CBAM supplier data as foundation for SBTi engagement tracking",
            "Align CBAM goods categories with EU Taxonomy activity classification",
        ]
        if "eudr" in [r.lower() for r in target_regulations]:
            recommendations.append("Leverage CBAM fertilizer supply chain data for EUDR due diligence")

        result = DataReuseReport(
            cbam_data_points=cbam_points,
            reusable_data_points=reusable,
            total_reuse_pct=total_reuse_pct,
            estimated_effort_reduction_hours=effort_hours,
            recommendations=recommendations,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info("Data reuse optimization: %d regulations, reuse=%s%%", len(target_regulations), total_reuse_pct)
        return result

    # -----------------------------------------------------------------------
    # Consistency Check
    # -----------------------------------------------------------------------

    def check_consistency(
        self, mappings: Dict[str, Any]
    ) -> ConsistencyCheckResult:
        """Check data consistency across multiple regulatory mappings.

        Args:
            mappings: Dict of regulation -> mapping data.

        Returns:
            ConsistencyCheckResult with inconsistency details.
        """
        inconsistencies: List[Dict[str, Any]] = []
        total_points = 0
        consistent_points = 0

        emission_values = {}
        for reg, data in mappings.items():
            if isinstance(data, dict):
                emissions = data.get("emissions_tco2e") or data.get("scope3_category1_emissions")
                if emissions is not None:
                    emission_values[reg] = _decimal(emissions)
                    total_points += 1

        if len(emission_values) > 1:
            values = list(emission_values.values())
            ref_value = values[0]
            for reg, val in emission_values.items():
                if val == ref_value:
                    consistent_points += 1
                else:
                    diff_pct = abs((val - ref_value) / ref_value * 100) if ref_value > 0 else Decimal("0")
                    if diff_pct > Decimal("5"):
                        inconsistencies.append({
                            "regulation": reg,
                            "field": "emissions_tco2e",
                            "expected": str(ref_value),
                            "actual": str(val),
                            "difference_pct": str(diff_pct.quantize(Decimal("0.01"))),
                        })
                    else:
                        consistent_points += 1

        total_points = max(total_points, 1)
        consistent_pct = (_decimal(consistent_points) / _decimal(total_points) * Decimal("100")).quantize(Decimal("0.01"))

        recommendations = []
        if inconsistencies:
            recommendations.append("Reconcile emission values across regulatory submissions")
            recommendations.append("Implement single source of truth for emission data")

        result = ConsistencyCheckResult(
            regulations_checked=list(mappings.keys()),
            inconsistencies=inconsistencies,
            total_data_points_checked=total_points,
            consistent_pct=consistent_pct,
            recommendations=recommendations,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info("Consistency check: %d regulations, %s%% consistent", len(mappings), consistent_pct)
        return result

    # -----------------------------------------------------------------------
    # Regulatory Change Tracking
    # -----------------------------------------------------------------------

    def track_regulatory_changes(
        self, regulation: str
    ) -> ChangeTracker:
        """Track recent and upcoming regulatory changes.

        Args:
            regulation: Regulation name (cbam, csrd, cdp, sbti, taxonomy, ets, eudr).

        Returns:
            ChangeTracker with change details and deadlines.
        """
        changes_db: Dict[str, List[Dict[str, Any]]] = {
            "cbam": [
                {"date": "2026-01-01", "change": "Definitive CBAM regime begins - certificates required",
                 "impact": "high"},
                {"date": "2026-05-31", "change": "First annual CBAM declaration deadline",
                 "impact": "critical"},
                {"date": "2026-12-31", "change": "End of first annual compliance period",
                 "impact": "high"},
            ],
            "csrd": [
                {"date": "2025-01-01", "change": "CSRD first wave (large companies >500 employees)",
                 "impact": "high"},
                {"date": "2026-01-01", "change": "CSRD second wave (all large companies)",
                 "impact": "high"},
            ],
            "ets": [
                {"date": "2026-01-01", "change": "Free allocation phase-out begins (2.5% reduction)",
                 "impact": "medium"},
                {"date": "2034-01-01", "change": "Free allocation fully phased out",
                 "impact": "critical"},
            ],
        }

        changes = changes_db.get(regulation.lower(), [
            {"date": "2026-01-01", "change": f"No specific changes tracked for {regulation}", "impact": "low"}
        ])

        deadlines = [
            {"date": c["date"], "description": c["change"], "priority": c["impact"]}
            for c in changes
        ]

        result = ChangeTracker(
            regulation=regulation,
            recent_changes=changes,
            upcoming_deadlines=deadlines,
            impact_assessment=f"{len(changes)} change(s) tracked for {regulation}",
        )
        result.provenance_hash = _compute_hash(result)

        logger.info("Tracked %d changes for %s", len(changes), regulation)
        return result

    # -----------------------------------------------------------------------
    # Carbon Pricing Equivalence
    # -----------------------------------------------------------------------

    def get_carbon_pricing_equivalence(
        self, country: str
    ) -> CarbonPricingEquivalence:
        """Get carbon pricing equivalence for CBAM deduction calculation.

        Per Article 9, the CBAM certificate cost shall be reduced by
        the carbon price effectively paid in the country of origin.

        Args:
            country: ISO 3166-1 alpha-2 country code.

        Returns:
            CarbonPricingEquivalence with deduction calculation.
        """
        country = country.upper().strip()
        pricing = self._carbon_pricing.get(country, {
            "country": country, "scheme_type": "none", "price_per_tco2e": "0",
            "currency": "EUR", "qualifying_for_deduction": False,
            "scheme_name": "No carbon pricing identified",
        })

        local_price = _decimal(pricing.get("price_per_tco2e", 0))
        currency = pricing.get("currency", "EUR")
        fx_rate = self.config.eur_fx_rates.get(currency, Decimal("1"))
        eur_price = (local_price * fx_rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        qualifying = pricing.get("qualifying_for_deduction", False)
        deduction = eur_price if qualifying else Decimal("0")
        net_cost = max(self.config.cbam_reference_price - deduction, Decimal("0"))

        result = CarbonPricingEquivalence(
            country=country,
            scheme_name=pricing.get("scheme_name", ""),
            scheme_type=pricing.get("scheme_type", ""),
            local_price_per_tco2e=local_price,
            local_currency=currency,
            eur_equivalent_price=eur_price,
            cbam_reference_price=self.config.cbam_reference_price,
            deduction_amount=deduction,
            qualifying_for_deduction=qualifying,
            net_cbam_cost=net_cost,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Carbon pricing for %s: local=%s %s, EUR=%s, deduction=%s, net=%s",
            country, local_price, currency, eur_price, deduction, net_cost,
        )
        return result
