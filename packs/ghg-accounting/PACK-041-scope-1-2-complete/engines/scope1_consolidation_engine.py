# -*- coding: utf-8 -*-
"""
Scope1ConsolidationEngine - PACK-041 Scope 1-2 Complete Engine 4
=================================================================

Scope 1 emission consolidation engine that aggregates facility-level
emission results across all eight Scope 1 source categories (stationary
combustion, mobile combustion, process emissions, fugitive emissions,
refrigerant leakage, land use, waste treatment, agricultural) into a
single, consolidated Scope 1 total for the organisation.

Performs multi-dimensional aggregation (by category, by gas, by facility,
by entity), detects and resolves double-counting between overlapping
categories, and applies organisational boundary percentages for equity-
share consolidation.

Calculation Methodology:
    Per-Facility Scope 1:
        S1_facility = sum(category_emissions_i) for i in applicable_categories

    Per-Entity Scope 1:
        S1_entity = sum(S1_facility_j * boundary_pct_j / 100)

    Organisation Scope 1:
        S1_org = sum(S1_entity_k)

    Per-Gas Aggregation:
        CO2e_gas = mass_gas * GWP_gas
        S1_org_by_gas = {gas: sum(CO2e_gas across all sources)}

    Double-Counting Detection:
        For each pair (cat_A, cat_B) in DOUBLE_COUNTING_RULES:
            If same emission source appears in both -> flag overlap
            overlap_amount = min(cat_A_amount, cat_B_amount) * overlap_fraction

    Boundary Percentage Application:
        For equity share approach:
            S1_entity_included = S1_entity_total * (equity_pct / 100)

Regulatory References:
    - GHG Protocol Corporate Standard (Revised), Chapter 4 (Setting Boundaries)
    - GHG Protocol Corporate Standard, Chapter 6 (Scope 1 Emissions)
    - ISO 14064-1:2018, Clause 5.2.3 (Direct GHG Emissions)
    - IPCC 2006/2019 Guidelines, Vol. 1 Ch. 1 (General Guidance)
    - EU ETS Monitoring & Reporting Regulation (MRR)

Zero-Hallucination:
    - All aggregation uses deterministic Decimal arithmetic
    - Double-counting rules from published GHG Protocol guidance
    - GWP values from IPCC AR4/AR5/AR6 (deterministic lookup)
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-041 Scope 1-2 Complete
Engine:  4 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

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
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
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

def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100)."""
    return _safe_divide(part * Decimal("100"), whole)

def _round_val(value: Decimal, places: int = 6) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Scope1Category(str, Enum):
    """Scope 1 emission source categories per GHG Protocol Chapter 4.

    STATIONARY:    Stationary combustion (boilers, furnaces, turbines).
    MOBILE:        Mobile combustion (company-owned vehicles).
    PROCESS:       Process emissions (chemical/physical transformations).
    FUGITIVE:      Fugitive emissions (leaks from equipment).
    REFRIGERANT:   Refrigerant leakage (HFC/PFC from HVAC).
    LAND_USE:      Land use and land use change.
    WASTE:         On-site waste treatment and disposal.
    AGRICULTURAL:  Agricultural emissions (enteric fermentation, soils).
    """
    STATIONARY = "stationary_combustion"
    MOBILE = "mobile_combustion"
    PROCESS = "process_emissions"
    FUGITIVE = "fugitive_emissions"
    REFRIGERANT = "refrigerant_leakage"
    LAND_USE = "land_use"
    WASTE = "waste_treatment"
    AGRICULTURAL = "agricultural"

class GasType(str, Enum):
    """Greenhouse gas types reported under the GHG Protocol.

    CO2:    Carbon dioxide (fossil).
    CO2_BIO: Carbon dioxide (biogenic, reported separately).
    CH4:    Methane.
    N2O:    Nitrous oxide.
    HFC:    Hydrofluorocarbons (basket).
    PFC:    Perfluorocarbons (basket).
    SF6:    Sulphur hexafluoride.
    NF3:    Nitrogen trifluoride.
    """
    CO2 = "co2"
    CO2_BIO = "co2_biogenic"
    CH4 = "ch4"
    N2O = "n2o"
    HFC = "hfc"
    PFC = "pfc"
    SF6 = "sf6"
    NF3 = "nf3"

class DoubleCountingType(str, Enum):
    """Type of double-counting overlap.

    WASTE_STATIONARY:   Waste incineration counted as both waste treatment
                        and stationary combustion.
    CHP_ALLOCATION:     Combined heat and power: emissions may be split
                        between stationary and process if not careful.
    BIOGAS_FUGITIVE:    Biogas capture counted under both waste treatment
                        (avoided) and fugitive (leakage).
    REFRIGERANT_PROCESS: Refrigerant used in process cooling may overlap
                        with refrigerant leakage category.
    LAND_USE_AGRICULTURAL: Overlap between land use change and agricultural
                        soil emissions.
    """
    WASTE_STATIONARY = "waste_stationary"
    CHP_ALLOCATION = "chp_allocation"
    BIOGAS_FUGITIVE = "biogas_fugitive"
    REFRIGERANT_PROCESS = "refrigerant_process"
    LAND_USE_AGRICULTURAL = "land_use_agricultural"

class ResolutionStrategy(str, Enum):
    """Strategy for resolving double-counting.

    DEDUCT_FROM_SECONDARY: Deduct the overlap from the secondary category.
    SPLIT_EQUALLY:         Split the overlap equally between categories.
    ALLOCATE_BY_SOURCE:    Allocate based on the originating source.
    MANUAL_REVIEW:         Flag for manual expert review.
    """
    DEDUCT_FROM_SECONDARY = "deduct_from_secondary"
    SPLIT_EQUALLY = "split_equally"
    ALLOCATE_BY_SOURCE = "allocate_by_source"
    MANUAL_REVIEW = "manual_review"

class ConsolidationStatus(str, Enum):
    """Status of the consolidation process.

    COMPLETE:          All categories consolidated successfully.
    PARTIAL:           Some categories missing or incomplete.
    DOUBLE_COUNTING:   Double-counting detected and resolved.
    ERROR:             Consolidation failed due to errors.
    """
    COMPLETE = "complete"
    PARTIAL = "partial"
    DOUBLE_COUNTING = "double_counting"
    ERROR = "error"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Double-counting rules: (primary_category, secondary_category, overlap_fraction).
# These define known overlap risks between Scope 1 categories.
DOUBLE_COUNTING_RULES: List[Dict[str, Any]] = [
    {
        "type": DoubleCountingType.WASTE_STATIONARY.value,
        "primary": Scope1Category.STATIONARY.value,
        "secondary": Scope1Category.WASTE.value,
        "description": (
            "Waste incineration with energy recovery may be counted under "
            "both stationary combustion and waste treatment."
        ),
        "default_strategy": ResolutionStrategy.DEDUCT_FROM_SECONDARY.value,
        "overlap_keywords": ["incineration", "waste_to_energy", "wte"],
    },
    {
        "type": DoubleCountingType.CHP_ALLOCATION.value,
        "primary": Scope1Category.STATIONARY.value,
        "secondary": Scope1Category.PROCESS.value,
        "description": (
            "CHP plant emissions may be allocated to both stationary "
            "combustion and process categories."
        ),
        "default_strategy": ResolutionStrategy.ALLOCATE_BY_SOURCE.value,
        "overlap_keywords": ["chp", "cogeneration", "combined_heat_power"],
    },
    {
        "type": DoubleCountingType.BIOGAS_FUGITIVE.value,
        "primary": Scope1Category.WASTE.value,
        "secondary": Scope1Category.FUGITIVE.value,
        "description": (
            "Biogas from waste treatment: combusted biogas counted under "
            "waste, but fugitive leakage from capture system may overlap."
        ),
        "default_strategy": ResolutionStrategy.DEDUCT_FROM_SECONDARY.value,
        "overlap_keywords": ["biogas", "landfill_gas", "anaerobic_digestion"],
    },
    {
        "type": DoubleCountingType.REFRIGERANT_PROCESS.value,
        "primary": Scope1Category.REFRIGERANT.value,
        "secondary": Scope1Category.PROCESS.value,
        "description": (
            "Refrigerant used in industrial process cooling may be "
            "reported under both refrigerant leakage and process emissions."
        ),
        "default_strategy": ResolutionStrategy.DEDUCT_FROM_SECONDARY.value,
        "overlap_keywords": ["process_cooling", "industrial_refrigeration"],
    },
    {
        "type": DoubleCountingType.LAND_USE_AGRICULTURAL.value,
        "primary": Scope1Category.AGRICULTURAL.value,
        "secondary": Scope1Category.LAND_USE.value,
        "description": (
            "Agricultural soil emissions may overlap with land use change "
            "emissions for cropland management."
        ),
        "default_strategy": ResolutionStrategy.MANUAL_REVIEW.value,
        "overlap_keywords": ["soil_carbon", "cropland", "tillage"],
    },
]

# Default GWP values (AR5, 100-year) for Scope 1 consolidation.
DEFAULT_GWP: Dict[str, Decimal] = {
    GasType.CO2.value: Decimal("1"),
    GasType.CO2_BIO.value: Decimal("0"),  # Biogenic CO2 reported separately
    GasType.CH4.value: Decimal("28"),
    GasType.N2O.value: Decimal("265"),
    GasType.HFC.value: Decimal("1300"),    # Representative: R-134a AR5
    GasType.PFC.value: Decimal("7390"),    # Representative: C2F6 AR5
    GasType.SF6.value: Decimal("23500"),
    GasType.NF3.value: Decimal("16100"),
}

# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------

class GasEmission(BaseModel):
    """Emission of a single greenhouse gas from a source.

    Attributes:
        gas: Greenhouse gas type.
        mass_kg: Mass of gas emitted (kg).
        gwp: GWP value used for conversion.
        co2e_kg: Emissions in kgCO2e (mass_kg * gwp).
    """
    gas: GasType = Field(default=GasType.CO2, description="Gas type")
    mass_kg: Decimal = Field(default=Decimal("0"), ge=0, description="Mass (kg)")
    gwp: Decimal = Field(default=Decimal("1"), ge=0, description="GWP value")
    co2e_kg: Decimal = Field(default=Decimal("0"), ge=0, description="kgCO2e")

class CategoryEmissions(BaseModel):
    """Emissions from a single Scope 1 category at a facility.

    Attributes:
        category: Scope 1 category.
        facility_id: Facility ID.
        entity_id: Entity ID.
        total_co2e_kg: Total emissions in kgCO2e.
        total_co2e_tonnes: Total emissions in tCO2e.
        gas_breakdown: Per-gas emissions.
        source_description: Description of the emission source.
        data_quality: Data quality flag (high, medium, low, estimated).
        emission_factor_ids: IDs of emission factors used.
        calculation_method: Calculation method used.
        tags: Tags for double-counting detection (e.g. 'incineration', 'chp').
    """
    category: Scope1Category = Field(..., description="Scope 1 category")
    facility_id: str = Field(default="", description="Facility ID")
    entity_id: str = Field(default="", description="Entity ID")
    total_co2e_kg: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total kgCO2e"
    )
    total_co2e_tonnes: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total tCO2e"
    )
    gas_breakdown: List[GasEmission] = Field(
        default_factory=list, description="Per-gas breakdown"
    )
    source_description: str = Field(default="", description="Source description")
    data_quality: str = Field(default="medium", description="Data quality")
    emission_factor_ids: List[str] = Field(
        default_factory=list, description="EF IDs used"
    )
    calculation_method: str = Field(default="", description="Calculation method")
    tags: List[str] = Field(
        default_factory=list, description="Tags for overlap detection"
    )

class FacilityScope1(BaseModel):
    """All Scope 1 emissions for a single facility.

    Attributes:
        facility_id: Facility ID.
        facility_name: Facility name.
        entity_id: Parent entity ID.
        country: Country code.
        categories: Per-category emission results.
        total_co2e_tonnes: Total Scope 1 in tCO2e.
        boundary_inclusion_pct: Inclusion percentage from boundary engine.
    """
    facility_id: str = Field(default="", description="Facility ID")
    facility_name: str = Field(default="", max_length=300, description="Facility name")
    entity_id: str = Field(default="", description="Entity ID")
    country: str = Field(default="", max_length=2, description="Country")
    categories: List[CategoryEmissions] = Field(
        default_factory=list, description="Per-category results"
    )
    total_co2e_tonnes: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total Scope 1 tCO2e"
    )
    boundary_inclusion_pct: Decimal = Field(
        default=Decimal("100"), ge=0, le=100,
        description="Boundary inclusion % (0-100)",
    )

class EntityScope1(BaseModel):
    """Scope 1 emissions for a single legal entity.

    Attributes:
        entity_id: Entity ID.
        entity_name: Entity name.
        facilities: Facility-level Scope 1 results.
        total_co2e_tonnes: Total entity Scope 1 in tCO2e.
        included_co2e_tonnes: Included Scope 1 after boundary % applied.
    """
    entity_id: str = Field(default="", description="Entity ID")
    entity_name: str = Field(default="", max_length=500, description="Entity name")
    facilities: List[FacilityScope1] = Field(
        default_factory=list, description="Facility results"
    )
    total_co2e_tonnes: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total entity Scope 1"
    )
    included_co2e_tonnes: Decimal = Field(
        default=Decimal("0"), ge=0, description="Included after boundary"
    )

# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------

class DoubleCountingFlag(BaseModel):
    """A detected double-counting overlap.

    Attributes:
        flag_id: Unique flag ID.
        overlap_type: Type of double-counting.
        primary_category: Primary category (emissions kept here).
        secondary_category: Secondary category (emissions may be deducted).
        facility_id: Facility where overlap detected.
        overlap_amount_tco2e: Estimated overlap amount.
        matched_tags: Tags that triggered the detection.
        description: Human-readable description.
        suggested_strategy: Suggested resolution strategy.
    """
    flag_id: str = Field(default_factory=_new_uuid, description="Flag ID")
    overlap_type: DoubleCountingType = Field(..., description="Overlap type")
    primary_category: Scope1Category = Field(..., description="Primary category")
    secondary_category: Scope1Category = Field(..., description="Secondary category")
    facility_id: str = Field(default="", description="Facility ID")
    overlap_amount_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Overlap amount (tCO2e)"
    )
    matched_tags: List[str] = Field(
        default_factory=list, description="Matched tags"
    )
    description: str = Field(default="", description="Description")
    suggested_strategy: ResolutionStrategy = Field(
        default=ResolutionStrategy.MANUAL_REVIEW, description="Suggested strategy"
    )

class Resolution(BaseModel):
    """Resolution of a double-counting flag.

    Attributes:
        resolution_id: Unique resolution ID.
        flag_id: ID of the flag being resolved.
        strategy: Resolution strategy applied.
        deducted_from_category: Category from which emissions were deducted.
        deduction_amount_tco2e: Amount deducted.
        rationale: Explanation of the resolution.
    """
    resolution_id: str = Field(default_factory=_new_uuid, description="Resolution ID")
    flag_id: str = Field(default="", description="Flag ID")
    strategy: ResolutionStrategy = Field(..., description="Strategy")
    deducted_from_category: Optional[Scope1Category] = Field(
        default=None, description="Category deducted from"
    )
    deduction_amount_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Deduction (tCO2e)"
    )
    rationale: str = Field(default="", description="Rationale")

class Scope1Total(BaseModel):
    """Consolidated Scope 1 total for the organisation.

    Attributes:
        total_co2e_tonnes: Total Scope 1 in tCO2e.
        by_category: Breakdown by category.
        by_gas: Breakdown by gas.
        by_facility: Breakdown by facility.
        by_entity: Breakdown by entity.
        biogenic_co2_tonnes: Biogenic CO2 reported separately.
        double_counting_deductions_tco2e: Total deductions for double-counting.
    """
    total_co2e_tonnes: Decimal = Field(
        default=Decimal("0"), description="Total Scope 1 tCO2e"
    )
    by_category: Dict[str, Decimal] = Field(
        default_factory=dict, description="By category"
    )
    by_gas: Dict[str, Decimal] = Field(
        default_factory=dict, description="By gas"
    )
    by_facility: Dict[str, Decimal] = Field(
        default_factory=dict, description="By facility"
    )
    by_entity: Dict[str, Decimal] = Field(
        default_factory=dict, description="By entity"
    )
    biogenic_co2_tonnes: Decimal = Field(
        default=Decimal("0"), description="Biogenic CO2 tCO2e"
    )
    double_counting_deductions_tco2e: Decimal = Field(
        default=Decimal("0"), description="DC deductions tCO2e"
    )

class Scope1ConsolidationResult(BaseModel):
    """Complete Scope 1 consolidation result.

    Attributes:
        result_id: Unique result ID.
        scope1_total: Consolidated Scope 1 total.
        entity_results: Per-entity results.
        double_counting_flags: Detected overlaps.
        resolutions: Applied resolutions.
        status: Consolidation status.
        total_facilities: Number of facilities.
        total_entities: Number of entities.
        gwp_values_used: GWP values used.
        warnings: Warnings.
        calculated_at: Timestamp.
        processing_time_ms: Processing time.
        provenance_hash: SHA-256 hash.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    scope1_total: Scope1Total = Field(
        default_factory=Scope1Total, description="Scope 1 total"
    )
    entity_results: List[EntityScope1] = Field(
        default_factory=list, description="Entity results"
    )
    double_counting_flags: List[DoubleCountingFlag] = Field(
        default_factory=list, description="DC flags"
    )
    resolutions: List[Resolution] = Field(
        default_factory=list, description="Resolutions"
    )
    status: ConsolidationStatus = Field(
        default=ConsolidationStatus.COMPLETE, description="Status"
    )
    total_facilities: int = Field(default=0, description="Total facilities")
    total_entities: int = Field(default=0, description="Total entities")
    gwp_values_used: Dict[str, Decimal] = Field(
        default_factory=dict, description="GWP values used"
    )
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    calculated_at: datetime = Field(
        default_factory=utcnow, description="Timestamp"
    )
    processing_time_ms: Decimal = Field(
        default=Decimal("0"), description="Processing time (ms)"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")

# ---------------------------------------------------------------------------
# Model Rebuild (resolve forward references from __future__ annotations)
# ---------------------------------------------------------------------------

GasEmission.model_rebuild()
CategoryEmissions.model_rebuild()
FacilityScope1.model_rebuild()
EntityScope1.model_rebuild()
DoubleCountingFlag.model_rebuild()
Resolution.model_rebuild()
Scope1Total.model_rebuild()
Scope1ConsolidationResult.model_rebuild()

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class Scope1ConsolidationEngine:
    """Scope 1 emission consolidation engine.

    Aggregates facility-level Scope 1 results across all eight categories
    into a consolidated organisational total with multi-dimensional
    breakdowns and double-counting detection.

    Attributes:
        _gwp_values: GWP values to use for consolidation.
        _flags: Detected double-counting flags.
        _resolutions: Applied resolutions.
        _warnings: Warnings.

    Example:
        >>> engine = Scope1ConsolidationEngine()
        >>> facility_results = [FacilityScope1(...)]
        >>> boundary_def = BoundaryDefinition(...)
        >>> result = engine.consolidate(facility_results, boundary_def)
        >>> print(result.scope1_total.total_co2e_tonnes)
    """

    def __init__(
        self,
        gwp_values: Optional[Dict[str, Decimal]] = None,
    ) -> None:
        """Initialise Scope1ConsolidationEngine.

        Args:
            gwp_values: Custom GWP values. Defaults to AR5 100-year.
        """
        self._gwp_values = gwp_values or dict(DEFAULT_GWP)
        self._flags: List[DoubleCountingFlag] = []
        self._resolutions: List[Resolution] = []
        self._warnings: List[str] = []
        logger.info(
            "Scope1ConsolidationEngine v%s initialised", _MODULE_VERSION
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def consolidate(
        self,
        facility_results: List[FacilityScope1],
        boundary_pcts: Optional[Dict[str, Decimal]] = None,
    ) -> Scope1ConsolidationResult:
        """Consolidate all Scope 1 emissions.

        Main entry point. Applies boundary percentages, detects double-
        counting, resolves overlaps, and produces the final consolidated
        Scope 1 total.

        Args:
            facility_results: Per-facility Scope 1 emission results.
            boundary_pcts: Optional mapping of entity_id to inclusion %.
                If None, uses each facility's boundary_inclusion_pct.

        Returns:
            Scope1ConsolidationResult.

        Raises:
            ValueError: If no facility results provided.
        """
        t0 = time.perf_counter()
        self._flags = []
        self._resolutions = []
        self._warnings = []

        if not facility_results:
            raise ValueError("At least one facility result is required")

        logger.info(
            "Consolidating Scope 1 for %d facilities", len(facility_results)
        )

        # Step 1: Apply boundary percentages.
        adjusted = self.apply_boundary_percentages(facility_results, boundary_pcts)

        # Step 2: Detect double counting.
        flags = self.detect_double_counting(adjusted)

        # Step 3: Resolve double counting (auto-resolve where possible).
        resolutions = self.resolve_double_counting(flags)

        # Step 4: Calculate deductions.
        total_deductions = sum(
            (r.deduction_amount_tco2e for r in resolutions), Decimal("0")
        )

        # Step 5: Aggregate.
        by_category = self.aggregate_by_category(adjusted)
        by_gas = self.aggregate_by_gas(adjusted)
        by_facility = self.aggregate_by_facility(adjusted)
        by_entity_dict = self._aggregate_by_entity_dict(adjusted)

        # Step 6: Compute total.
        raw_total = sum(by_category.values(), Decimal("0"))
        net_total = raw_total - total_deductions

        # Step 7: Compute biogenic CO2.
        biogenic = self._compute_biogenic(adjusted)

        # Step 8: Build entity results.
        entity_results = self._build_entity_results(adjusted, boundary_pcts)

        # Step 9: Determine status.
        status = ConsolidationStatus.COMPLETE
        if flags:
            status = ConsolidationStatus.DOUBLE_COUNTING

        scope1_total = Scope1Total(
            total_co2e_tonnes=_round_val(net_total, 4),
            by_category={k: _round_val(v, 4) for k, v in by_category.items()},
            by_gas={k: _round_val(v, 4) for k, v in by_gas.items()},
            by_facility={k: _round_val(v, 4) for k, v in by_facility.items()},
            by_entity={k: _round_val(v, 4) for k, v in by_entity_dict.items()},
            biogenic_co2_tonnes=_round_val(biogenic, 4),
            double_counting_deductions_tco2e=_round_val(total_deductions, 4),
        )

        elapsed = Decimal(str(round((time.perf_counter() - t0) * 1000, 3)))

        result = Scope1ConsolidationResult(
            scope1_total=scope1_total,
            entity_results=entity_results,
            double_counting_flags=self._flags,
            resolutions=self._resolutions,
            status=status,
            total_facilities=len(facility_results),
            total_entities=len(set(f.entity_id for f in facility_results)),
            gwp_values_used=dict(self._gwp_values),
            warnings=self._warnings,
            processing_time_ms=elapsed,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Scope 1 consolidation complete: %.2f tCO2e "
            "(%d flags, %d resolutions, deductions=%.2f tCO2e)",
            float(net_total), len(flags), len(resolutions),
            float(total_deductions),
        )
        return result

    def aggregate_by_category(
        self, results: List[FacilityScope1]
    ) -> Dict[str, Decimal]:
        """Aggregate emissions by Scope 1 category.

        Args:
            results: Facility Scope 1 results.

        Returns:
            Dict mapping category name to tCO2e.
        """
        agg: Dict[str, Decimal] = {}
        for cat in Scope1Category:
            agg[cat.value] = Decimal("0")

        for fac in results:
            fraction = _safe_divide(
                _decimal(fac.boundary_inclusion_pct), Decimal("100")
            )
            for cat_em in fac.categories:
                key = cat_em.category.value
                agg[key] = agg.get(key, Decimal("0")) + (
                    cat_em.total_co2e_tonnes * fraction
                )

        return agg

    def aggregate_by_gas(
        self, results: List[FacilityScope1]
    ) -> Dict[str, Decimal]:
        """Aggregate emissions by greenhouse gas type.

        Args:
            results: Facility Scope 1 results.

        Returns:
            Dict mapping gas name to tCO2e.
        """
        agg: Dict[str, Decimal] = {}
        for gas in GasType:
            agg[gas.value] = Decimal("0")

        for fac in results:
            fraction = _safe_divide(
                _decimal(fac.boundary_inclusion_pct), Decimal("100")
            )
            for cat_em in fac.categories:
                for ge in cat_em.gas_breakdown:
                    key = ge.gas.value
                    co2e_tonnes = _safe_divide(ge.co2e_kg, Decimal("1000"))
                    agg[key] = agg.get(key, Decimal("0")) + (
                        co2e_tonnes * fraction
                    )

        return agg

    def aggregate_by_facility(
        self, results: List[FacilityScope1]
    ) -> Dict[str, Decimal]:
        """Aggregate emissions by facility.

        Args:
            results: Facility Scope 1 results.

        Returns:
            Dict mapping facility_id to tCO2e.
        """
        agg: Dict[str, Decimal] = {}
        for fac in results:
            fraction = _safe_divide(
                _decimal(fac.boundary_inclusion_pct), Decimal("100")
            )
            included = fac.total_co2e_tonnes * fraction
            agg[fac.facility_id] = agg.get(
                fac.facility_id, Decimal("0")
            ) + included

        return agg

    def detect_double_counting(
        self, results: List[FacilityScope1]
    ) -> List[DoubleCountingFlag]:
        """Detect potential double-counting between Scope 1 categories.

        Checks each facility's category emissions against the known
        double-counting rules using tag matching.

        Args:
            results: Facility Scope 1 results.

        Returns:
            List of DoubleCountingFlag objects.
        """
        logger.info("Detecting double-counting across %d facilities", len(results))
        flags: List[DoubleCountingFlag] = []

        for fac in results:
            # Build category lookup for this facility.
            cat_lookup: Dict[str, CategoryEmissions] = {}
            for cat_em in fac.categories:
                cat_lookup[cat_em.category.value] = cat_em

            for rule in DOUBLE_COUNTING_RULES:
                primary_key = rule["primary"]
                secondary_key = rule["secondary"]

                primary = cat_lookup.get(primary_key)
                secondary = cat_lookup.get(secondary_key)

                if primary is None or secondary is None:
                    continue

                if (primary.total_co2e_tonnes == Decimal("0") or
                        secondary.total_co2e_tonnes == Decimal("0")):
                    continue

                # Check for tag overlap.
                overlap_keywords = set(rule.get("overlap_keywords", []))
                primary_tags = set(t.lower() for t in primary.tags)
                secondary_tags = set(t.lower() for t in secondary.tags)
                all_tags = primary_tags | secondary_tags

                matched = overlap_keywords & all_tags
                if not matched:
                    continue

                # Estimate overlap amount.
                overlap_amount = min(
                    primary.total_co2e_tonnes,
                    secondary.total_co2e_tonnes,
                )

                strategy_str = rule.get(
                    "default_strategy",
                    ResolutionStrategy.MANUAL_REVIEW.value,
                )
                strategy = ResolutionStrategy(strategy_str)

                flag = DoubleCountingFlag(
                    overlap_type=DoubleCountingType(rule["type"]),
                    primary_category=Scope1Category(primary_key),
                    secondary_category=Scope1Category(secondary_key),
                    facility_id=fac.facility_id,
                    overlap_amount_tco2e=_round_val(overlap_amount, 4),
                    matched_tags=sorted(matched),
                    description=rule["description"],
                    suggested_strategy=strategy,
                )
                flags.append(flag)

        self._flags = flags
        logger.info("Double-counting detection: %d flags raised", len(flags))
        return flags

    def resolve_double_counting(
        self,
        flags: List[DoubleCountingFlag],
        resolution_rules: Optional[Dict[str, ResolutionStrategy]] = None,
    ) -> List[Resolution]:
        """Resolve double-counting flags.

        Applies the suggested or overridden resolution strategy to each
        detected overlap.

        Args:
            flags: List of double-counting flags.
            resolution_rules: Optional overrides for resolution strategies.
                Maps overlap_type to ResolutionStrategy.

        Returns:
            List of Resolution objects.
        """
        logger.info("Resolving %d double-counting flags", len(flags))
        resolutions: List[Resolution] = []

        for flag in flags:
            # Determine strategy.
            strategy = flag.suggested_strategy
            if resolution_rules and flag.overlap_type.value in resolution_rules:
                strategy = resolution_rules[flag.overlap_type.value]

            deduction = Decimal("0")
            deducted_from: Optional[Scope1Category] = None
            rationale = ""

            if strategy == ResolutionStrategy.DEDUCT_FROM_SECONDARY:
                deduction = flag.overlap_amount_tco2e
                deducted_from = flag.secondary_category
                rationale = (
                    f"Deducted {deduction} tCO2e from "
                    f"{flag.secondary_category.value} to avoid double-counting "
                    f"with {flag.primary_category.value}. "
                    f"Tags matched: {', '.join(flag.matched_tags)}."
                )

            elif strategy == ResolutionStrategy.SPLIT_EQUALLY:
                deduction = _safe_divide(
                    flag.overlap_amount_tco2e, Decimal("2")
                )
                deducted_from = flag.secondary_category
                rationale = (
                    f"Split overlap of {flag.overlap_amount_tco2e} tCO2e "
                    f"equally. Deducted {deduction} tCO2e from "
                    f"{flag.secondary_category.value}."
                )

            elif strategy == ResolutionStrategy.ALLOCATE_BY_SOURCE:
                # Allocate to primary; deduct from secondary.
                deduction = flag.overlap_amount_tco2e
                deducted_from = flag.secondary_category
                rationale = (
                    f"Allocated overlap to primary category "
                    f"({flag.primary_category.value}). Deducted "
                    f"{deduction} tCO2e from "
                    f"{flag.secondary_category.value}."
                )

            elif strategy == ResolutionStrategy.MANUAL_REVIEW:
                deduction = Decimal("0")
                rationale = (
                    f"Overlap between {flag.primary_category.value} and "
                    f"{flag.secondary_category.value} flagged for manual review. "
                    f"No automatic deduction applied."
                )
                self._warnings.append(
                    f"Manual review required: {flag.overlap_type.value} at "
                    f"facility {flag.facility_id} ({flag.overlap_amount_tco2e} "
                    f"tCO2e potential overlap)."
                )

            resolution = Resolution(
                flag_id=flag.flag_id,
                strategy=strategy,
                deducted_from_category=deducted_from,
                deduction_amount_tco2e=_round_val(deduction, 4),
                rationale=rationale,
            )
            resolutions.append(resolution)

        self._resolutions = resolutions
        logger.info(
            "Resolved %d flags, total deductions=%.2f tCO2e",
            len(resolutions),
            float(sum(r.deduction_amount_tco2e for r in resolutions)),
        )
        return resolutions

    def apply_boundary_percentages(
        self,
        results: List[FacilityScope1],
        boundary_pcts: Optional[Dict[str, Decimal]] = None,
    ) -> List[FacilityScope1]:
        """Apply organisational boundary percentages to facility results.

        If boundary_pcts is provided, overrides each facility's
        boundary_inclusion_pct based on entity_id mapping.

        Args:
            results: Facility Scope 1 results.
            boundary_pcts: Optional entity_id -> inclusion % mapping.

        Returns:
            Updated list of FacilityScope1 (same objects, pct updated).
        """
        if boundary_pcts is None:
            return results

        logger.info(
            "Applying boundary percentages for %d entities",
            len(boundary_pcts),
        )

        for fac in results:
            if fac.entity_id in boundary_pcts:
                pct = _decimal(boundary_pcts[fac.entity_id])
                fac.boundary_inclusion_pct = pct
                logger.debug(
                    "Facility %s (entity %s): boundary_pct=%.2f%%",
                    fac.facility_id, fac.entity_id, float(pct),
                )

        return results

    # ------------------------------------------------------------------
    # Private Methods
    # ------------------------------------------------------------------

    def _aggregate_by_entity_dict(
        self, results: List[FacilityScope1]
    ) -> Dict[str, Decimal]:
        """Aggregate emissions by entity.

        Args:
            results: Facility Scope 1 results.

        Returns:
            Dict mapping entity_id to tCO2e.
        """
        agg: Dict[str, Decimal] = {}
        for fac in results:
            fraction = _safe_divide(
                _decimal(fac.boundary_inclusion_pct), Decimal("100")
            )
            included = fac.total_co2e_tonnes * fraction
            agg[fac.entity_id] = agg.get(
                fac.entity_id, Decimal("0")
            ) + included
        return agg

    def _build_entity_results(
        self,
        facilities: List[FacilityScope1],
        boundary_pcts: Optional[Dict[str, Decimal]],
    ) -> List[EntityScope1]:
        """Build per-entity consolidated results.

        Args:
            facilities: Facility results.
            boundary_pcts: Optional boundary percentages.

        Returns:
            List of EntityScope1.
        """
        # Group facilities by entity.
        entity_facs: Dict[str, List[FacilityScope1]] = {}
        for fac in facilities:
            entity_facs.setdefault(fac.entity_id, []).append(fac)

        results: List[EntityScope1] = []
        for entity_id, facs in entity_facs.items():
            total = sum(
                (f.total_co2e_tonnes for f in facs), Decimal("0")
            )
            included = Decimal("0")
            for f in facs:
                fraction = _safe_divide(
                    _decimal(f.boundary_inclusion_pct), Decimal("100")
                )
                included += f.total_co2e_tonnes * fraction

            # Get entity name from first facility (if available).
            entity_name = ""
            if boundary_pcts and entity_id in boundary_pcts:
                entity_name = entity_id  # placeholder

            results.append(EntityScope1(
                entity_id=entity_id,
                entity_name=entity_name,
                facilities=facs,
                total_co2e_tonnes=_round_val(total, 4),
                included_co2e_tonnes=_round_val(included, 4),
            ))

        return results

    def _compute_biogenic(
        self, results: List[FacilityScope1]
    ) -> Decimal:
        """Compute total biogenic CO2 emissions.

        Per GHG Protocol, biogenic CO2 is reported outside the scopes
        but must be tracked separately.

        Args:
            results: Facility Scope 1 results.

        Returns:
            Total biogenic CO2 in tCO2e.
        """
        total = Decimal("0")
        for fac in results:
            fraction = _safe_divide(
                _decimal(fac.boundary_inclusion_pct), Decimal("100")
            )
            for cat_em in fac.categories:
                for ge in cat_em.gas_breakdown:
                    if ge.gas == GasType.CO2_BIO:
                        total += _safe_divide(
                            ge.co2e_kg, Decimal("1000")
                        ) * fraction
        return total
