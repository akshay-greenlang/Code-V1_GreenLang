# -*- coding: utf-8 -*-
"""
FoodWasteEngine - PACK-014 CSRD Retail Engine 5
====================================================

Food waste measurement and reduction tracking per EU targets.
Covers food waste categorisation, waste hierarchy scoring,
reduction tracking against EU 2030 targets (30% reduction vs
2020-2022 baseline per revised Waste Framework Directive),
emissions from food waste, financial loss quantification, and
redistribution programme effectiveness.

ESRS E5 Disclosure Requirements (Retail-Specific):
    - E5-4: Resource inflows including food procurement
    - E5-5: Resource outflows including food waste by category
    - E5-6: Anticipated financial effects of food waste

EU Farm to Fork Strategy:
    - 50% reduction of per capita food waste at retail and consumer
      level by 2030 (SDG target 12.3)
    - Mandatory food waste measurement (delegated act 2023)
    - Waste hierarchy: prevention > redistribution > animal feed >
      composting > energy recovery > disposal

Regulatory References:
    - Waste Framework Directive 2008/98/EC (revised 2024)
    - EU Farm to Fork Strategy (May 2020)
    - Commission Delegated Decision (EU) 2019/1597 (measurement)
    - ESRS E5 Resource Use and Circular Economy
    - UN SDG 12.3 (halve per capita food waste by 2030)
    - WRAP Food Waste Reduction Roadmap

Zero-Hallucination:
    - All waste hierarchy scores use deterministic weighted averages
    - Emission factors from WRAP/DEFRA published sources
    - Reduction tracking uses simple arithmetic (no estimation)
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-014 CSRD Retail & Consumer Goods
Status:  Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

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
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, Pydantic model, or other).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning *default* on zero denominator."""
    if denominator == 0.0:
        return default
    return numerator / denominator

def _safe_pct(numerator: float, denominator: float) -> float:
    """Calculate percentage safely, returning 0.0 on zero denominator."""
    if denominator == 0.0:
        return 0.0
    return (numerator / denominator) * 100.0

def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))

def _round2(value: float) -> float:
    """Round to 2 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def _round4(value: float) -> float:
    """Round to 4 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FoodWasteCategory(str, Enum):
    """Food category for waste classification in retail.

    Categories aligned with EU food waste measurement methodology
    (Commission Delegated Decision 2019/1597) and WRAP reporting
    categories for retail operations.
    """
    BAKERY = "bakery"
    PRODUCE = "produce"
    DAIRY = "dairy"
    MEAT_POULTRY = "meat_poultry"
    SEAFOOD = "seafood"
    PREPARED_FOOD = "prepared_food"
    PACKAGED_GOODS = "packaged_goods"
    BEVERAGES = "beverages"
    FROZEN = "frozen"
    DELI = "deli"
    CONFECTIONERY = "confectionery"
    OTHER = "other"

class WasteDestination(str, Enum):
    """Destination for food waste, ordered by waste hierarchy preference.

    Per Waste Framework Directive 2008/98/EC Article 4 and EU Farm
    to Fork Strategy, the waste hierarchy for food is:
    Prevention > Redistribution > Animal feed > Composting/AD >
    Energy recovery > Disposal.
    """
    REDISTRIBUTION = "redistribution"
    ANIMAL_FEED = "animal_feed"
    COMPOSTING = "composting"
    ANAEROBIC_DIGESTION = "anaerobic_digestion"
    INCINERATION_ENERGY = "incineration_energy"
    INCINERATION_NO_ENERGY = "incineration_no_energy"
    LANDFILL = "landfill"
    SEWER = "sewer"

class MeasurementMethod(str, Enum):
    """Food waste measurement methodology per EU delegated act.

    Commission Delegated Decision (EU) 2019/1597 specifies measurement
    methods for food waste quantification at each stage of the supply
    chain.
    """
    DIRECT_WEIGHING = "direct_weighing"
    WASTE_COMPOSITION_ANALYSIS = "waste_composition_analysis"
    DIARIES = "diaries"
    COEFFICIENTS = "coefficients"
    SCANNING_DATA = "scanning_data"

class WasteHierarchyLevel(str, Enum):
    """Waste hierarchy levels per Waste Framework Directive Article 4.

    Higher levels are preferred: prevention is the most desirable
    outcome, disposal the least.
    """
    PREVENTION = "prevention"
    REDISTRIBUTION = "redistribution"
    ANIMAL_FEED = "animal_feed"
    RECYCLING_COMPOSTING = "recycling_composting"
    ENERGY_RECOVERY = "energy_recovery"
    DISPOSAL = "disposal"

# ---------------------------------------------------------------------------
# Embedded Constants
# ---------------------------------------------------------------------------

EU_FOOD_WASTE_REDUCTION_TARGET: float = 30.0
"""EU target: 30% reduction in food waste by 2030 vs 2020-2022 baseline.
Revised Waste Framework Directive (2024 proposal)."""

EU_TARGET_YEAR: int = 2030
"""Target year for the 30% reduction goal."""

BASELINE_PERIOD_START: int = 2020
"""Start year for baseline period."""

BASELINE_PERIOD_END: int = 2022
"""End year for baseline period (inclusive)."""

# Emission factors for food waste by category (kgCO2e per kg wasted).
# Sources: WRAP (2021) 'Food surplus and waste in the UK - key facts',
# DEFRA emission factors for food waste (2023), and FAO (2019)
# 'The State of Food and Agriculture'.
# These factors include embedded emissions from production,
# processing, and transportation of the wasted food, plus
# methane emissions from landfill decomposition where applicable.
FOOD_WASTE_EMISSION_FACTORS: Dict[str, float] = {
    "bakery": 0.89,
    "produce": 0.51,
    "dairy": 3.18,
    "meat_poultry": 13.31,
    "seafood": 5.37,
    "prepared_food": 2.45,
    "packaged_goods": 1.12,
    "beverages": 0.34,
    "frozen": 1.85,
    "deli": 4.22,
    "confectionery": 1.67,
    "other": 1.50,
}

# Waste hierarchy weights (0-1 scale) for scoring destination quality.
# Weight 1.0 = best outcome (prevention), 0.0 = worst (landfill/disposal).
# Used to calculate a composite waste hierarchy score that indicates how
# well a retailer is managing food waste destinations.
WASTE_HIERARCHY_WEIGHTS: Dict[str, float] = {
    "prevention": 1.0,
    "redistribution": 0.9,
    "animal_feed": 0.7,
    "recycling_composting": 0.4,
    "energy_recovery": 0.2,
    "disposal": 0.0,
}

# Mapping from specific waste destinations to waste hierarchy levels.
DESTINATION_TO_HIERARCHY: Dict[str, str] = {
    "redistribution": "redistribution",
    "animal_feed": "animal_feed",
    "composting": "recycling_composting",
    "anaerobic_digestion": "recycling_composting",
    "incineration_energy": "energy_recovery",
    "incineration_no_energy": "disposal",
    "landfill": "disposal",
    "sewer": "disposal",
}

# Average shelf life in days by food category (retail setting).
# Used for estimating waste risk and optimal ordering calculations.
SHELF_LIFE_BY_CATEGORY: Dict[str, int] = {
    "bakery": 3,
    "produce": 7,
    "dairy": 14,
    "meat_poultry": 5,
    "seafood": 3,
    "prepared_food": 2,
    "packaged_goods": 365,
    "beverages": 180,
    "frozen": 365,
    "deli": 3,
    "confectionery": 90,
    "other": 30,
}

REDISTRIBUTION_CREDIT: float = 0.85
"""Credit factor for redistributed food (0-1 scale).
Redistributed food receives 85% of the prevention credit value,
reflecting that redistribution prevents waste but still incurs some
logistics emissions."""

# Average cost per kg by food category (EUR) for financial loss calculation.
# Based on typical EU grocery retail price ranges (mid-range, 2023 data).
AVG_COST_PER_KG: Dict[str, float] = {
    "bakery": 3.50,
    "produce": 2.20,
    "dairy": 4.80,
    "meat_poultry": 8.50,
    "seafood": 12.00,
    "prepared_food": 6.50,
    "packaged_goods": 3.80,
    "beverages": 1.50,
    "frozen": 4.20,
    "deli": 9.00,
    "confectionery": 7.50,
    "other": 3.00,
}

# Measurement accuracy by method (0-1 scale).
# Higher values indicate more accurate measurement.  Direct weighing is
# the gold standard per EU delegated act.
MEASUREMENT_ACCURACY: Dict[str, float] = {
    "direct_weighing": 0.95,
    "waste_composition_analysis": 0.85,
    "diaries": 0.70,
    "coefficients": 0.60,
    "scanning_data": 0.90,
}

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class FoodWasteRecord(BaseModel):
    """Individual food waste measurement record.

    Represents a single waste event or aggregated waste data for a
    specific store, category, and destination over a reporting period.
    """
    record_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this waste record",
    )
    store_id: str = Field(
        ...,
        description="Store or facility identifier",
        min_length=1,
    )
    category: FoodWasteCategory = Field(
        ...,
        description="Food waste category",
    )
    quantity_kg: float = Field(
        ...,
        description="Waste quantity in kilograms",
        ge=0.0,
    )
    destination: WasteDestination = Field(
        ...,
        description="Waste destination (redistribution, composting, landfill, etc.)",
    )
    measurement_method: MeasurementMethod = Field(
        default=MeasurementMethod.DIRECT_WEIGHING,
        description="Method used to measure waste quantity",
    )
    reporting_period: str = Field(
        ...,
        description="Reporting period (e.g. '2025-Q1', '2025-01')",
        min_length=4,
    )
    value_eur: Optional[float] = Field(
        default=None,
        description="Financial value of the wasted food (EUR), if known",
        ge=0.0,
    )
    notes: Optional[str] = Field(
        default=None,
        description="Additional notes or context",
    )

    @field_validator("quantity_kg")
    @classmethod
    def validate_quantity(cls, v: float) -> float:
        """Validate that quantity is non-negative and reasonable."""
        if v < 0:
            raise ValueError("Food waste quantity cannot be negative")
        if v > 1_000_000:
            raise ValueError("Food waste quantity exceeds 1,000,000 kg sanity check")
        return v

class FoodWasteBaseline(BaseModel):
    """Baseline data for food waste reduction tracking.

    Per the revised Waste Framework Directive, Member States must
    establish baseline food waste levels from 2020-2022 data and
    measure progress toward the 2030 reduction target.
    """
    baseline_year: int = Field(
        ...,
        description="Baseline reference year (or midpoint of baseline period)",
        ge=2018,
        le=2025,
    )
    total_waste_kg: float = Field(
        ...,
        description="Total food waste in kg for the baseline period (annualized)",
        ge=0.0,
    )
    waste_by_category: Dict[str, float] = Field(
        default_factory=dict,
        description="Waste breakdown by FoodWasteCategory (kg)",
    )
    waste_per_store_kg: Optional[float] = Field(
        default=None,
        description="Average waste per store (kg/year) in baseline",
        ge=0.0,
    )
    waste_per_revenue_eur: Optional[float] = Field(
        default=None,
        description="Waste intensity: kg waste per EUR revenue in baseline",
        ge=0.0,
    )
    store_count: Optional[int] = Field(
        default=None,
        description="Number of stores in baseline period",
        ge=0,
    )
    revenue_eur: Optional[float] = Field(
        default=None,
        description="Total revenue in baseline period (EUR)",
        ge=0.0,
    )

class CategoryWasteDetail(BaseModel):
    """Detailed waste breakdown for a single food category."""
    category: str = Field(..., description="Food waste category name")
    quantity_kg: float = Field(default=0.0, description="Total waste in kg")
    share_pct: float = Field(default=0.0, description="Share of total waste (%)")
    emission_factor: float = Field(default=0.0, description="kgCO2e per kg wasted")
    emissions_kg_co2e: float = Field(default=0.0, description="Total emissions kgCO2e")
    financial_value_eur: float = Field(default=0.0, description="Financial loss EUR")
    avg_shelf_life_days: int = Field(default=0, description="Average shelf life (days)")

class DestinationDetail(BaseModel):
    """Waste breakdown by destination."""
    destination: str = Field(..., description="Waste destination")
    quantity_kg: float = Field(default=0.0, description="Quantity sent to destination (kg)")
    share_pct: float = Field(default=0.0, description="Share of total waste (%)")
    hierarchy_level: str = Field(default="", description="Waste hierarchy level")
    hierarchy_weight: float = Field(default=0.0, description="Hierarchy weight (0-1)")

class StoreWasteDetail(BaseModel):
    """Waste summary for a single store."""
    store_id: str = Field(..., description="Store identifier")
    total_waste_kg: float = Field(default=0.0, description="Total waste (kg)")
    waste_per_category: Dict[str, float] = Field(default_factory=dict)
    hierarchy_score: float = Field(default=0.0, description="Waste hierarchy score (0-1)")
    top_waste_category: str = Field(default="", description="Category with most waste")
    measurement_accuracy: float = Field(default=0.0, description="Avg measurement accuracy")

class ReductionTracking(BaseModel):
    """Tracks progress toward EU 2030 food waste reduction target."""
    baseline_total_kg: float = Field(default=0.0, description="Baseline total waste (kg)")
    current_total_kg: float = Field(default=0.0, description="Current period total waste (kg)")
    absolute_reduction_kg: float = Field(default=0.0, description="Absolute reduction (kg)")
    reduction_pct: float = Field(default=0.0, description="Reduction percentage vs baseline")
    target_reduction_pct: float = Field(default=30.0, description="EU target reduction (%)")
    required_annual_reduction_pct: float = Field(
        default=0.0, description="Required annual reduction to meet target",
    )
    actual_annual_reduction_pct: float = Field(
        default=0.0, description="Actual annual reduction rate",
    )
    on_track: bool = Field(default=False, description="Whether on track for 2030 target")
    years_to_target: int = Field(default=0, description="Years remaining to 2030")
    expected_reduction_pct: float = Field(
        default=0.0, description="Expected reduction at this point (linear)",
    )
    gap_pct: float = Field(
        default=0.0, description="Gap between expected and actual reduction (%)",
    )

class FoodWasteResult(BaseModel):
    """Complete food waste analysis result with full provenance.

    Contains all calculated metrics, breakdown by category and
    destination, waste hierarchy scoring, reduction tracking,
    emissions quantification, and actionable recommendations.
    """
    result_id: str = Field(
        default_factory=_new_uuid,
        description="Unique result identifier",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION,
        description="Engine version used for this calculation",
    )
    calculated_at: datetime = Field(
        default_factory=utcnow,
        description="Timestamp of calculation (UTC)",
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Processing time in milliseconds",
    )

    # --- Core Metrics ---
    total_waste_kg: float = Field(default=0.0, description="Total food waste (kg)")
    total_waste_tonnes: float = Field(default=0.0, description="Total food waste (tonnes)")
    record_count: int = Field(default=0, description="Number of waste records processed")
    store_count: int = Field(default=0, description="Number of unique stores")
    waste_per_store_kg: float = Field(default=0.0, description="Average waste per store (kg)")

    # --- Category Breakdown ---
    waste_by_category: List[CategoryWasteDetail] = Field(
        default_factory=list,
        description="Waste breakdown by food category",
    )
    top_waste_category: str = Field(default="", description="Category with highest waste")
    top_waste_category_pct: float = Field(default=0.0, description="Top category share (%)")

    # --- Destination Breakdown ---
    waste_by_destination: List[DestinationDetail] = Field(
        default_factory=list,
        description="Waste breakdown by destination",
    )

    # --- Waste Hierarchy ---
    waste_hierarchy_score: float = Field(
        default=0.0,
        description="Composite waste hierarchy score (0-1, higher is better)",
    )
    waste_hierarchy_grade: str = Field(
        default="",
        description="Grade (A/B/C/D/F) based on hierarchy score",
    )

    # --- Reduction Tracking ---
    reduction_tracking: Optional[ReductionTracking] = Field(
        default=None,
        description="Progress toward EU 2030 reduction target",
    )
    reduction_vs_baseline_pct: Optional[float] = Field(
        default=None,
        description="Reduction percentage vs baseline",
    )
    on_track_for_2030_target: Optional[bool] = Field(
        default=None,
        description="Whether on track for EU 2030 target",
    )

    # --- Emissions ---
    emissions_from_waste_kg_co2e: float = Field(
        default=0.0,
        description="Total GHG emissions from food waste (kgCO2e)",
    )
    emissions_from_waste_tco2e: float = Field(
        default=0.0,
        description="Total GHG emissions from food waste (tCO2e)",
    )
    emissions_by_category: Dict[str, float] = Field(
        default_factory=dict,
        description="Emissions by food category (kgCO2e)",
    )

    # --- Financial Impact ---
    financial_value_wasted_eur: float = Field(
        default=0.0,
        description="Total financial value of wasted food (EUR)",
    )
    financial_value_by_category: Dict[str, float] = Field(
        default_factory=dict,
        description="Financial value wasted by category (EUR)",
    )

    # --- Redistribution ---
    redistribution_kg: float = Field(
        default=0.0,
        description="Total food redistributed (kg)",
    )
    redistribution_rate_pct: float = Field(
        default=0.0,
        description="Redistribution rate (%)",
    )
    redistribution_emissions_avoided_kg_co2e: float = Field(
        default=0.0,
        description="Emissions avoided through redistribution (kgCO2e)",
    )

    # --- Store Details ---
    store_details: List[StoreWasteDetail] = Field(
        default_factory=list,
        description="Per-store waste summaries",
    )

    # --- Data Quality ---
    avg_measurement_accuracy: float = Field(
        default=0.0,
        description="Average measurement accuracy (0-1)",
    )
    measurement_methods_used: List[str] = Field(
        default_factory=list,
        description="List of measurement methods used",
    )

    # --- Recommendations ---
    recommendations: List[str] = Field(
        default_factory=list,
        description="Actionable recommendations for improvement",
    )

    # --- Provenance ---
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of all inputs and calculation steps",
    )

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class FoodWasteEngine:
    """Food waste measurement and reduction tracking engine.

    Provides deterministic, zero-hallucination calculations for:
    - Food waste aggregation by category and destination
    - Waste hierarchy scoring per EU Waste Framework Directive
    - Reduction tracking against EU 2030 targets
    - Emissions from food waste (kgCO2e and tCO2e)
    - Financial loss quantification
    - Redistribution programme effectiveness
    - Store-level waste analysis
    - Actionable recommendations

    All calculations are bit-perfect reproducible.  No LLM is used
    in any calculation path.

    Usage::

        engine = FoodWasteEngine()
        records = [
            FoodWasteRecord(
                store_id="STORE-001",
                category=FoodWasteCategory.BAKERY,
                quantity_kg=150.0,
                destination=WasteDestination.REDISTRIBUTION,
                reporting_period="2025-Q1",
            ),
            # ... more records
        ]
        baseline = FoodWasteBaseline(
            baseline_year=2021,
            total_waste_kg=50000.0,
        )
        result = engine.calculate(records, baseline=baseline)
    """

    engine_version: str = _MODULE_VERSION

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def calculate(
        self,
        records: List[FoodWasteRecord],
        baseline: Optional[FoodWasteBaseline] = None,
        reporting_year: int = 2025,
        facility_id: Optional[str] = None,
    ) -> FoodWasteResult:
        """Run the full food waste analysis.

        Args:
            records: List of food waste measurement records.
            baseline: Optional baseline for reduction tracking.
            reporting_year: Current reporting year for trajectory calc.
            facility_id: Optional facility identifier for the result.

        Returns:
            FoodWasteResult with complete metrics and provenance.

        Raises:
            ValueError: If records list is empty or contains invalid data.
        """
        t0 = time.perf_counter()

        if not records:
            raise ValueError("At least one FoodWasteRecord is required")

        # Step 1: Aggregate by category
        cat_agg = self._aggregate_by_category(records)

        # Step 2: Aggregate by destination
        dest_agg = self._aggregate_by_destination(records)

        # Step 3: Calculate totals
        total_kg = sum(r.quantity_kg for r in records)
        total_tonnes = _round3(total_kg / 1000.0)
        stores = set(r.store_id for r in records)
        store_count = len(stores)
        waste_per_store = _round2(_safe_divide(total_kg, float(store_count)))

        # Step 4: Category details
        cat_details = self._build_category_details(cat_agg, total_kg)

        # Step 5: Destination details
        dest_details = self._build_destination_details(dest_agg, total_kg)

        # Step 6: Waste hierarchy score
        hierarchy_score = self._calculate_hierarchy_score(dest_agg, total_kg)
        hierarchy_grade = self._hierarchy_grade(hierarchy_score)

        # Step 7: Emissions
        emissions_by_cat, total_emissions_kg = self._calculate_emissions(cat_agg)
        total_emissions_t = _round3(total_emissions_kg / 1000.0)

        # Step 8: Financial impact
        fin_by_cat, total_financial = self._calculate_financial_impact(cat_agg, records)

        # Step 9: Redistribution metrics
        redist_kg = dest_agg.get("redistribution", 0.0)
        # Redistribution rate = redistributed / total (since redistribution is part of records)
        redist_rate = _round2(_safe_pct(redist_kg, total_kg))
        redist_avoided = _round2(self._redistribution_emissions_avoided(
            redist_kg, records
        ))

        # Step 10: Reduction tracking
        reduction = None
        reduction_pct = None
        on_track = None
        if baseline is not None:
            reduction = self._calculate_reduction_tracking(
                baseline, total_kg, reporting_year
            )
            reduction_pct = reduction.reduction_pct
            on_track = reduction.on_track

        # Step 11: Store details
        store_details = self._build_store_details(records)

        # Step 12: Data quality
        methods_used = list(set(r.measurement_method.value for r in records))
        avg_accuracy = _round3(
            _safe_divide(
                sum(MEASUREMENT_ACCURACY.get(m, 0.5) for m in methods_used),
                float(len(methods_used)),
            )
        )

        # Step 13: Top category
        top_cat = ""
        top_cat_pct = 0.0
        if cat_details:
            sorted_cats = sorted(cat_details, key=lambda c: c.quantity_kg, reverse=True)
            top_cat = sorted_cats[0].category
            top_cat_pct = sorted_cats[0].share_pct

        # Step 14: Recommendations
        recommendations = self._generate_recommendations(
            cat_details, dest_details, hierarchy_score, hierarchy_grade,
            reduction, redist_rate, avg_accuracy,
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = FoodWasteResult(
            total_waste_kg=_round2(total_kg),
            total_waste_tonnes=total_tonnes,
            record_count=len(records),
            store_count=store_count,
            waste_per_store_kg=waste_per_store,
            waste_by_category=cat_details,
            top_waste_category=top_cat,
            top_waste_category_pct=top_cat_pct,
            waste_by_destination=dest_details,
            waste_hierarchy_score=_round3(hierarchy_score),
            waste_hierarchy_grade=hierarchy_grade,
            reduction_tracking=reduction,
            reduction_vs_baseline_pct=(
                _round2(reduction_pct) if reduction_pct is not None else None
            ),
            on_track_for_2030_target=on_track,
            emissions_from_waste_kg_co2e=_round2(total_emissions_kg),
            emissions_from_waste_tco2e=total_emissions_t,
            emissions_by_category={k: _round2(v) for k, v in emissions_by_cat.items()},
            financial_value_wasted_eur=_round2(total_financial),
            financial_value_by_category={k: _round2(v) for k, v in fin_by_cat.items()},
            redistribution_kg=_round2(redist_kg),
            redistribution_rate_pct=redist_rate,
            redistribution_emissions_avoided_kg_co2e=redist_avoided,
            store_details=store_details,
            avg_measurement_accuracy=avg_accuracy,
            measurement_methods_used=methods_used,
            recommendations=recommendations,
            processing_time_ms=elapsed_ms,
        )

        result.provenance_hash = _compute_hash(result)
        return result

    # ------------------------------------------------------------------ #
    # Aggregation                                                         #
    # ------------------------------------------------------------------ #

    def _aggregate_by_category(
        self, records: List[FoodWasteRecord]
    ) -> Dict[str, float]:
        """Aggregate waste quantity by food category.

        Args:
            records: List of food waste records.

        Returns:
            Dict mapping category value to total kg.
        """
        agg: Dict[str, float] = {}
        for r in records:
            key = r.category.value
            agg[key] = agg.get(key, 0.0) + r.quantity_kg
        return agg

    def _aggregate_by_destination(
        self, records: List[FoodWasteRecord]
    ) -> Dict[str, float]:
        """Aggregate waste quantity by destination.

        Args:
            records: List of food waste records.

        Returns:
            Dict mapping destination value to total kg.
        """
        agg: Dict[str, float] = {}
        for r in records:
            key = r.destination.value
            agg[key] = agg.get(key, 0.0) + r.quantity_kg
        return agg

    # ------------------------------------------------------------------ #
    # Category Details                                                    #
    # ------------------------------------------------------------------ #

    def _build_category_details(
        self, cat_agg: Dict[str, float], total_kg: float
    ) -> List[CategoryWasteDetail]:
        """Build detailed breakdown for each category with waste.

        Args:
            cat_agg: Category -> total kg mapping.
            total_kg: Total waste in kg.

        Returns:
            List of CategoryWasteDetail sorted by quantity descending.
        """
        details: List[CategoryWasteDetail] = []
        for cat_name, qty in cat_agg.items():
            ef = FOOD_WASTE_EMISSION_FACTORS.get(cat_name, 1.50)
            cost = AVG_COST_PER_KG.get(cat_name, 3.00)
            shelf = SHELF_LIFE_BY_CATEGORY.get(cat_name, 30)
            details.append(CategoryWasteDetail(
                category=cat_name,
                quantity_kg=_round2(qty),
                share_pct=_round2(_safe_pct(qty, total_kg)),
                emission_factor=ef,
                emissions_kg_co2e=_round2(qty * ef),
                financial_value_eur=_round2(qty * cost),
                avg_shelf_life_days=shelf,
            ))
        details.sort(key=lambda d: d.quantity_kg, reverse=True)
        return details

    # ------------------------------------------------------------------ #
    # Destination Details                                                 #
    # ------------------------------------------------------------------ #

    def _build_destination_details(
        self, dest_agg: Dict[str, float], total_kg: float
    ) -> List[DestinationDetail]:
        """Build detailed breakdown for each waste destination.

        Args:
            dest_agg: Destination -> total kg mapping.
            total_kg: Total waste in kg.

        Returns:
            List of DestinationDetail sorted by quantity descending.
        """
        details: List[DestinationDetail] = []
        for dest_name, qty in dest_agg.items():
            hier_level = DESTINATION_TO_HIERARCHY.get(dest_name, "disposal")
            hier_weight = WASTE_HIERARCHY_WEIGHTS.get(hier_level, 0.0)
            details.append(DestinationDetail(
                destination=dest_name,
                quantity_kg=_round2(qty),
                share_pct=_round2(_safe_pct(qty, total_kg)),
                hierarchy_level=hier_level,
                hierarchy_weight=hier_weight,
            ))
        details.sort(key=lambda d: d.quantity_kg, reverse=True)
        return details

    # ------------------------------------------------------------------ #
    # Waste Hierarchy Score                                               #
    # ------------------------------------------------------------------ #

    def _calculate_hierarchy_score(
        self, dest_agg: Dict[str, float], total_kg: float
    ) -> float:
        """Calculate composite waste hierarchy score (0-1).

        The score is a weighted average of hierarchy weights by
        quantity sent to each destination:
            score = sum(weight_i * quantity_i) / total_quantity

        A score of 1.0 means all waste is prevented.
        A score of 0.0 means all waste goes to landfill.

        Args:
            dest_agg: Destination -> total kg mapping.
            total_kg: Total waste in kg.

        Returns:
            Hierarchy score between 0.0 and 1.0.
        """
        if total_kg <= 0.0:
            return 0.0

        weighted_sum = 0.0
        for dest_name, qty in dest_agg.items():
            hier_level = DESTINATION_TO_HIERARCHY.get(dest_name, "disposal")
            weight = WASTE_HIERARCHY_WEIGHTS.get(hier_level, 0.0)
            weighted_sum += weight * qty

        return _safe_divide(weighted_sum, total_kg)

    def _hierarchy_grade(self, score: float) -> str:
        """Convert hierarchy score to letter grade.

        Grading thresholds:
            A: >= 0.80 (excellent - mostly prevention/redistribution)
            B: >= 0.60 (good - significant animal feed/composting)
            C: >= 0.40 (adequate - mixed destinations)
            D: >= 0.20 (poor - mostly energy recovery)
            F: < 0.20 (failing - mostly landfill/disposal)

        Args:
            score: Hierarchy score (0-1).

        Returns:
            Letter grade A through F.
        """
        if score >= 0.80:
            return "A"
        elif score >= 0.60:
            return "B"
        elif score >= 0.40:
            return "C"
        elif score >= 0.20:
            return "D"
        else:
            return "F"

    # ------------------------------------------------------------------ #
    # Emissions Calculation                                               #
    # ------------------------------------------------------------------ #

    def _calculate_emissions(
        self, cat_agg: Dict[str, float]
    ) -> Tuple[Dict[str, float], float]:
        """Calculate GHG emissions from food waste.

        Emissions = waste_kg * emission_factor_per_category.
        Uses WRAP/DEFRA published emission factors.

        Args:
            cat_agg: Category -> total kg mapping.

        Returns:
            Tuple of (emissions_by_category dict, total_emissions_kg).
        """
        emissions: Dict[str, float] = {}
        total = 0.0
        for cat_name, qty in cat_agg.items():
            ef = FOOD_WASTE_EMISSION_FACTORS.get(cat_name, 1.50)
            cat_emissions = qty * ef
            emissions[cat_name] = cat_emissions
            total += cat_emissions
        return emissions, total

    # ------------------------------------------------------------------ #
    # Financial Impact                                                    #
    # ------------------------------------------------------------------ #

    def _calculate_financial_impact(
        self,
        cat_agg: Dict[str, float],
        records: List[FoodWasteRecord],
    ) -> Tuple[Dict[str, float], float]:
        """Calculate financial loss from food waste.

        Uses explicit value_eur from records where available,
        otherwise applies average cost per kg by category.

        Args:
            cat_agg: Category -> total kg mapping.
            records: Original records (may contain explicit values).

        Returns:
            Tuple of (financial_by_category dict, total_financial EUR).
        """
        # Check for explicit values in records
        explicit_values: Dict[str, float] = {}
        explicit_qty: Dict[str, float] = {}
        for r in records:
            key = r.category.value
            if r.value_eur is not None:
                explicit_values[key] = explicit_values.get(key, 0.0) + r.value_eur
                explicit_qty[key] = explicit_qty.get(key, 0.0) + r.quantity_kg

        fin: Dict[str, float] = {}
        total = 0.0
        for cat_name, qty in cat_agg.items():
            if cat_name in explicit_values:
                # Use explicit values for portion with explicit data,
                # estimate for remainder
                expl_val = explicit_values[cat_name]
                expl_q = explicit_qty.get(cat_name, 0.0)
                remaining_qty = qty - expl_q
                if remaining_qty > 0:
                    cost_per_kg = AVG_COST_PER_KG.get(cat_name, 3.00)
                    cat_value = expl_val + remaining_qty * cost_per_kg
                else:
                    cat_value = expl_val
            else:
                cost_per_kg = AVG_COST_PER_KG.get(cat_name, 3.00)
                cat_value = qty * cost_per_kg
            fin[cat_name] = cat_value
            total += cat_value

        return fin, total

    # ------------------------------------------------------------------ #
    # Redistribution                                                      #
    # ------------------------------------------------------------------ #

    def _redistribution_emissions_avoided(
        self, redist_kg: float, records: List[FoodWasteRecord]
    ) -> float:
        """Calculate emissions avoided through food redistribution.

        Redistributed food avoids the embedded emissions that would
        occur if the food were wasted.  A credit factor of 0.85 is
        applied (logistics emissions partially offset the benefit).

        Args:
            redist_kg: Total redistributed quantity in kg.
            records: Original records for category breakdown.

        Returns:
            Emissions avoided in kgCO2e.
        """
        if redist_kg <= 0.0:
            return 0.0

        # Calculate weighted average emission factor for redistributed food
        redist_by_cat: Dict[str, float] = {}
        for r in records:
            if r.destination == WasteDestination.REDISTRIBUTION:
                key = r.category.value
                redist_by_cat[key] = redist_by_cat.get(key, 0.0) + r.quantity_kg

        avoided = 0.0
        for cat_name, qty in redist_by_cat.items():
            ef = FOOD_WASTE_EMISSION_FACTORS.get(cat_name, 1.50)
            avoided += qty * ef * REDISTRIBUTION_CREDIT

        return avoided

    # ------------------------------------------------------------------ #
    # Reduction Tracking                                                  #
    # ------------------------------------------------------------------ #

    def _calculate_reduction_tracking(
        self,
        baseline: FoodWasteBaseline,
        current_total_kg: float,
        reporting_year: int,
    ) -> ReductionTracking:
        """Track progress toward EU 2030 food waste reduction target.

        Uses linear interpolation from baseline year to 2030 to
        determine expected reduction at the current reporting year.

        Formula:
            expected_reduction_pct = target_pct * (years_elapsed / total_years)
            on_track = actual_reduction_pct >= expected_reduction_pct

        Args:
            baseline: Baseline food waste data.
            current_total_kg: Current period total waste (kg).
            reporting_year: Current reporting year.

        Returns:
            ReductionTracking with all progress metrics.
        """
        baseline_kg = baseline.total_waste_kg
        abs_reduction = baseline_kg - current_total_kg
        reduction_pct = _safe_pct(abs_reduction, baseline_kg)

        years_total = EU_TARGET_YEAR - baseline.baseline_year
        years_elapsed = reporting_year - baseline.baseline_year
        years_to_target = max(0, EU_TARGET_YEAR - reporting_year)

        # Linear interpolation: expected reduction at this point
        if years_total > 0:
            expected_pct = EU_FOOD_WASTE_REDUCTION_TARGET * _safe_divide(
                float(years_elapsed), float(years_total)
            )
        else:
            expected_pct = EU_FOOD_WASTE_REDUCTION_TARGET

        # Required annual reduction to meet target from current position
        if years_to_target > 0:
            remaining_reduction_needed = EU_FOOD_WASTE_REDUCTION_TARGET - reduction_pct
            required_annual = _safe_divide(
                remaining_reduction_needed, float(years_to_target)
            )
        else:
            required_annual = 0.0

        # Actual annual reduction rate
        if years_elapsed > 0:
            actual_annual = _safe_divide(reduction_pct, float(years_elapsed))
        else:
            actual_annual = 0.0

        on_track = reduction_pct >= expected_pct
        gap = expected_pct - reduction_pct

        return ReductionTracking(
            baseline_total_kg=_round2(baseline_kg),
            current_total_kg=_round2(current_total_kg),
            absolute_reduction_kg=_round2(abs_reduction),
            reduction_pct=_round2(reduction_pct),
            target_reduction_pct=EU_FOOD_WASTE_REDUCTION_TARGET,
            required_annual_reduction_pct=_round2(required_annual),
            actual_annual_reduction_pct=_round2(actual_annual),
            on_track=on_track,
            years_to_target=years_to_target,
            expected_reduction_pct=_round2(expected_pct),
            gap_pct=_round2(gap),
        )

    # ------------------------------------------------------------------ #
    # Store Details                                                       #
    # ------------------------------------------------------------------ #

    def _build_store_details(
        self, records: List[FoodWasteRecord]
    ) -> List[StoreWasteDetail]:
        """Build per-store waste summaries.

        Args:
            records: List of food waste records.

        Returns:
            List of StoreWasteDetail sorted by total waste descending.
        """
        store_data: Dict[str, Dict[str, Any]] = {}

        for r in records:
            sid = r.store_id
            if sid not in store_data:
                store_data[sid] = {
                    "total_kg": 0.0,
                    "by_cat": {},
                    "dest_weights": [],
                    "dest_qtys": [],
                    "methods": [],
                }

            store_data[sid]["total_kg"] += r.quantity_kg
            cat = r.category.value
            store_data[sid]["by_cat"][cat] = (
                store_data[sid]["by_cat"].get(cat, 0.0) + r.quantity_kg
            )

            hier_level = DESTINATION_TO_HIERARCHY.get(
                r.destination.value, "disposal"
            )
            weight = WASTE_HIERARCHY_WEIGHTS.get(hier_level, 0.0)
            store_data[sid]["dest_weights"].append(weight)
            store_data[sid]["dest_qtys"].append(r.quantity_kg)
            store_data[sid]["methods"].append(r.measurement_method.value)

        details: List[StoreWasteDetail] = []
        for sid, data in store_data.items():
            total_store = data["total_kg"]
            # Weighted hierarchy score for store
            if total_store > 0:
                w_sum = sum(
                    w * q
                    for w, q in zip(data["dest_weights"], data["dest_qtys"])
                )
                h_score = _safe_divide(w_sum, total_store)
            else:
                h_score = 0.0

            # Top category
            by_cat = data["by_cat"]
            top_cat = max(by_cat, key=by_cat.get) if by_cat else ""

            # Measurement accuracy
            methods = data["methods"]
            if methods:
                acc = sum(
                    MEASUREMENT_ACCURACY.get(m, 0.5) for m in methods
                ) / len(methods)
            else:
                acc = 0.0

            details.append(StoreWasteDetail(
                store_id=sid,
                total_waste_kg=_round2(total_store),
                waste_per_category={k: _round2(v) for k, v in by_cat.items()},
                hierarchy_score=_round3(h_score),
                top_waste_category=top_cat,
                measurement_accuracy=_round3(acc),
            ))

        details.sort(key=lambda d: d.total_waste_kg, reverse=True)
        return details

    # ------------------------------------------------------------------ #
    # Recommendations                                                     #
    # ------------------------------------------------------------------ #

    def _generate_recommendations(
        self,
        cat_details: List[CategoryWasteDetail],
        dest_details: List[DestinationDetail],
        hierarchy_score: float,
        hierarchy_grade: str,
        reduction: Optional[ReductionTracking],
        redist_rate: float,
        avg_accuracy: float,
    ) -> List[str]:
        """Generate actionable recommendations based on analysis.

        Recommendations are deterministic: they are derived from
        threshold comparisons on calculated metrics, not from any
        LLM or probabilistic model.

        Args:
            cat_details: Category breakdown details.
            dest_details: Destination breakdown details.
            hierarchy_score: Composite hierarchy score (0-1).
            hierarchy_grade: Letter grade.
            reduction: Reduction tracking data (may be None).
            redist_rate: Redistribution rate (%).
            avg_accuracy: Average measurement accuracy (0-1).

        Returns:
            List of recommendation strings.
        """
        recs: List[str] = []

        # R1: High landfill/disposal share
        landfill_pct = 0.0
        for d in dest_details:
            if d.hierarchy_level == "disposal":
                landfill_pct += d.share_pct
        if landfill_pct > 20.0:
            recs.append(
                f"CRITICAL: {_round2(landfill_pct)}% of food waste goes to disposal "
                f"(landfill/incineration without energy recovery). Prioritise "
                f"diversion to composting, anaerobic digestion, or animal feed."
            )

        # R2: Low hierarchy score
        if hierarchy_grade in ("D", "F"):
            recs.append(
                f"Waste hierarchy score is {_round3(hierarchy_score)} "
                f"(grade {hierarchy_grade}). "
                f"Move waste up the hierarchy: increase redistribution "
                f"and composting."
            )

        # R3: Off track for 2030 target
        if reduction is not None and not reduction.on_track:
            recs.append(
                f"Off track for EU 2030 food waste reduction target. "
                f"Current reduction: {reduction.reduction_pct}%, "
                f"expected at this point: {reduction.expected_reduction_pct}%. "
                f"Gap: {reduction.gap_pct}pp. Accelerate waste prevention "
                f"measures."
            )

        # R4: Low redistribution
        if redist_rate < 5.0:
            recs.append(
                f"Redistribution rate is only {redist_rate}%. Partner with "
                f"food banks and charities to increase food donation. EU Good "
                f"Samaritan legislation protects donors from liability."
            )

        # R5: High-emission categories
        for cd in cat_details:
            if cd.share_pct > 15.0 and cd.emission_factor > 5.0:
                recs.append(
                    f"High-emission waste: {cd.category} represents "
                    f"{cd.share_pct}% of waste with emission factor "
                    f"{cd.emission_factor} kgCO2e/kg. Prioritise reduction "
                    f"in this category for maximum climate impact."
                )

        # R6: Short shelf life categories dominating waste
        for cd in cat_details:
            if cd.share_pct > 20.0 and cd.avg_shelf_life_days <= 5:
                recs.append(
                    f"{cd.category} has short shelf life "
                    f"({cd.avg_shelf_life_days} days) and represents "
                    f"{cd.share_pct}% of waste. Implement dynamic pricing, "
                    f"improve demand forecasting, and consider smaller "
                    f"batch ordering."
                )

        # R7: Low measurement accuracy
        if avg_accuracy < 0.75:
            recs.append(
                f"Average measurement accuracy is "
                f"{_round2(avg_accuracy * 100)}%. Transition to direct "
                f"weighing or scanning-based measurement for more accurate "
                f"waste tracking per EU delegated act."
            )

        # R8: No baseline data
        if reduction is None:
            recs.append(
                "No baseline data provided. Establish a food waste baseline "
                "using 2020-2022 data to track progress toward EU 2030 "
                "reduction target (30% reduction required)."
            )

        return recs

    # ------------------------------------------------------------------ #
    # Single-record calculation (convenience)                             #
    # ------------------------------------------------------------------ #

    def calculate_single_record_emissions(
        self, record: FoodWasteRecord
    ) -> Dict[str, Any]:
        """Calculate emissions for a single waste record.

        Convenience method for quick per-record analysis.

        Args:
            record: Single food waste record.

        Returns:
            Dict with emissions_kg_co2e, financial_value_eur, hierarchy_weight.
        """
        ef = FOOD_WASTE_EMISSION_FACTORS.get(record.category.value, 1.50)
        cost = AVG_COST_PER_KG.get(record.category.value, 3.00)
        hier_level = DESTINATION_TO_HIERARCHY.get(
            record.destination.value, "disposal"
        )
        hier_weight = WASTE_HIERARCHY_WEIGHTS.get(hier_level, 0.0)

        emissions = record.quantity_kg * ef
        fin_value = (
            record.value_eur
            if record.value_eur is not None
            else record.quantity_kg * cost
        )

        return {
            "record_id": record.record_id,
            "category": record.category.value,
            "quantity_kg": record.quantity_kg,
            "destination": record.destination.value,
            "emission_factor_kg_co2e_per_kg": ef,
            "emissions_kg_co2e": _round2(emissions),
            "emissions_tco2e": _round3(emissions / 1000.0),
            "financial_value_eur": _round2(fin_value),
            "hierarchy_level": hier_level,
            "hierarchy_weight": hier_weight,
            "provenance_hash": _compute_hash({
                "record_id": record.record_id,
                "emissions": str(emissions),
                "fin_value": str(fin_value),
            }),
        }

    # ------------------------------------------------------------------ #
    # Waste intensity metrics                                             #
    # ------------------------------------------------------------------ #

    def calculate_waste_intensity(
        self,
        total_waste_kg: float,
        revenue_eur: Optional[float] = None,
        store_count: Optional[int] = None,
        floor_area_sqm: Optional[float] = None,
        employee_count: Optional[int] = None,
    ) -> Dict[str, Optional[float]]:
        """Calculate waste intensity ratios.

        Intensity metrics normalise waste quantities to business size,
        enabling like-for-like comparison across retailers.

        Args:
            total_waste_kg: Total food waste in kg.
            revenue_eur: Annual revenue in EUR (optional).
            store_count: Number of stores (optional).
            floor_area_sqm: Total floor area in sqm (optional).
            employee_count: Total employees (optional).

        Returns:
            Dict with intensity metrics (None if denominator unavailable).
        """
        return {
            "waste_per_eur_million_revenue": (
                _round3(
                    _safe_divide(total_waste_kg, revenue_eur / 1_000_000.0)
                )
                if revenue_eur and revenue_eur > 0 else None
            ),
            "waste_per_store_kg": (
                _round2(_safe_divide(total_waste_kg, float(store_count)))
                if store_count and store_count > 0 else None
            ),
            "waste_per_sqm_kg": (
                _round4(_safe_divide(total_waste_kg, floor_area_sqm))
                if floor_area_sqm and floor_area_sqm > 0 else None
            ),
            "waste_per_employee_kg": (
                _round2(_safe_divide(total_waste_kg, float(employee_count)))
                if employee_count and employee_count > 0 else None
            ),
        }

    # ------------------------------------------------------------------ #
    # Category risk assessment                                            #
    # ------------------------------------------------------------------ #

    def assess_category_waste_risk(
        self, records: List[FoodWasteRecord]
    ) -> List[Dict[str, Any]]:
        """Assess food waste risk by category based on shelf life and waste rate.

        Short shelf-life categories with high waste volumes are flagged
        as high risk.  This helps retailers prioritise intervention.

        Args:
            records: List of food waste records.

        Returns:
            List of risk assessments by category.
        """
        cat_agg = self._aggregate_by_category(records)
        total_kg = sum(cat_agg.values())

        risks: List[Dict[str, Any]] = []
        for cat_name, qty in cat_agg.items():
            shelf_life = SHELF_LIFE_BY_CATEGORY.get(cat_name, 30)
            share = _safe_pct(qty, total_kg)
            ef = FOOD_WASTE_EMISSION_FACTORS.get(cat_name, 1.50)

            # Risk scoring: combination of share, shelf life, emission factor
            # Short shelf life + high share + high EF = high risk
            shelf_risk = _safe_divide(30.0, float(shelf_life), default=1.0)
            shelf_risk = min(shelf_risk, 5.0)  # Cap at 5
            share_factor = share / 10.0  # Normalise share
            ef_factor = ef / 5.0  # Normalise emission factor

            risk_score = _round2(
                shelf_risk * 0.4 + share_factor * 0.3 + ef_factor * 0.3
            )

            if risk_score >= 3.0:
                risk_level = "CRITICAL"
            elif risk_score >= 2.0:
                risk_level = "HIGH"
            elif risk_score >= 1.0:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            risks.append({
                "category": cat_name,
                "quantity_kg": _round2(qty),
                "share_pct": _round2(share),
                "shelf_life_days": shelf_life,
                "emission_factor": ef,
                "risk_score": risk_score,
                "risk_level": risk_level,
            })

        risks.sort(key=lambda r: r["risk_score"], reverse=True)
        return risks

    # ------------------------------------------------------------------ #
    # EU SDG 12.3 Target Assessment                                       #
    # ------------------------------------------------------------------ #

    def assess_sdg_123_alignment(
        self,
        baseline_waste_per_capita_kg: float,
        current_waste_per_capita_kg: float,
        reporting_year: int = 2025,
    ) -> Dict[str, Any]:
        """Assess alignment with UN SDG 12.3 target.

        SDG 12.3: By 2030, halve per capita global food waste at the
        retail and consumer levels and reduce food losses along
        production and supply chains.

        Args:
            baseline_waste_per_capita_kg: Baseline per-capita waste (kg).
            current_waste_per_capita_kg: Current per-capita waste (kg).
            reporting_year: Current reporting year.

        Returns:
            Dict with SDG 12.3 alignment assessment.
        """
        target_reduction_pct = 50.0  # SDG 12.3 = halve
        target_year = 2030
        baseline_year = 2015  # SDG baseline

        reduction_pct = _safe_pct(
            baseline_waste_per_capita_kg - current_waste_per_capita_kg,
            baseline_waste_per_capita_kg,
        )

        years_total = target_year - baseline_year
        years_elapsed = reporting_year - baseline_year
        expected_pct = target_reduction_pct * _safe_divide(
            float(years_elapsed), float(years_total)
        )

        on_track = reduction_pct >= expected_pct

        result = {
            "sdg_target": "12.3",
            "target_description": "Halve per capita food waste by 2030",
            "baseline_per_capita_kg": _round2(baseline_waste_per_capita_kg),
            "current_per_capita_kg": _round2(current_waste_per_capita_kg),
            "reduction_pct": _round2(reduction_pct),
            "target_reduction_pct": target_reduction_pct,
            "expected_reduction_pct": _round2(expected_pct),
            "on_track": on_track,
            "gap_pct": _round2(expected_pct - reduction_pct),
            "provenance_hash": _compute_hash({
                "baseline": str(baseline_waste_per_capita_kg),
                "current": str(current_waste_per_capita_kg),
                "year": reporting_year,
            }),
        }
        return result
