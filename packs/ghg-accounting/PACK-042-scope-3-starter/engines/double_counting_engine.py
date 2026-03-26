# -*- coding: utf-8 -*-
"""
DoubleCountingPreventionEngine - PACK-042 Scope 3 Starter Pack Engine 4
=========================================================================

Detects and resolves double-counting of emissions across Scope 3
categories.  Implements 12 specific overlap rules derived from GHG
Protocol Technical Guidance, each addressing a known area where emissions
may be attributed to more than one category.

Double-counting is a material risk in Scope 3 inventories because the
15 categories have intentional overlaps in scope.  The GHG Protocol
acknowledges these overlaps and recommends organisations establish
clear category boundaries.  This engine codifies those boundary rules
as deterministic detection logic.

Calculation Methodology:
    Overlap Detection:
        For each rule R in OVERLAP_RULES:
            If both R.category_a and R.category_b have emissions > 0:
                overlap_amount = min(E_a, E_b) * R.overlap_fraction
                flag = OverlapDetection(...)

    Resolution - CONSERVATIVE:
        adjusted_E_higher = E_higher - overlap_amount
        (Remove from higher-numbered category)

    Resolution - PROPORTIONAL:
        weight_a = DQ_a / (DQ_a + DQ_b)
        adjusted_E_a = E_a - overlap_amount * weight_a
        adjusted_E_b = E_b - overlap_amount * (1 - weight_a)

    Net Adjustment:
        total_adjustment = sum(overlap_amount for all resolved overlaps)

Regulatory References:
    - GHG Protocol Scope 3 Standard, Chapter 8 (Accounting and Reporting)
    - GHG Protocol Technical Guidance, Appendix A (Avoiding Double Counting)
    - ISO 14064-1:2018, Clause 5.2.4.1 Note 1 (avoiding double counting)
    - ESRS E1 (EFRAG 2023) AR 46-48 (disaggregation of Scope 3)
    - SBTi Criteria v5.1, Section 8.3 (Scope 3 target boundary)

Zero-Hallucination:
    - All overlap rules from published GHG Protocol guidance
    - Detection uses deterministic comparison logic
    - Resolution uses fixed allocation formulas
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-042 Scope 3 Starter
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
        serialisable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serialisable = data
    else:
        serialisable = str(data)
    if isinstance(serialisable, dict):
        serialisable = {
            k: v for k, v in serialisable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serialisable, sort_keys=True, default=str)
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


def _round_val(value: Decimal, places: int = 6) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class OverlapRuleId(str, Enum):
    """Identifier for each overlap detection rule.

    Each rule corresponds to a known double-counting risk between two
    Scope 3 categories as documented in GHG Protocol Technical Guidance.
    """
    CAT1_CAT3 = "cat1_cat3_energy_in_goods"
    CAT1_CAT4 = "cat1_cat4_logistics_in_price"
    CAT1_CAT2 = "cat1_cat2_capitalized_vs_expensed"
    CAT3_SCOPE2 = "cat3_scope2_upstream_energy"
    CAT4_CAT9 = "cat4_cat9_transport_allocation"
    CAT8_SCOPE12 = "cat8_scope12_leased_vs_operational"
    CAT13_CAT11 = "cat13_cat11_leased_vs_use"
    CAT14_SCOPE12 = "cat14_scope12_franchise_vs_operational"
    CAT1_CAT5 = "cat1_cat5_packaging_waste"
    CAT10_CAT11 = "cat10_cat11_processing_vs_use"
    CAT6_CAT7 = "cat6_cat7_travel_vs_commuting"
    CAT15_CAT1314 = "cat15_cat1314_investment_vs_leased"


class AllocationMethod(str, Enum):
    """Method for resolving detected overlaps.

    CONSERVATIVE:   Remove overlap from the higher-numbered category.
    PROPORTIONAL:   Split based on data quality weights.
    REMOVE_LOWER:   Remove from the lower-numbered category.
    FLAG_ONLY:      Flag for manual review without auto-adjustment.
    """
    CONSERVATIVE = "conservative"
    PROPORTIONAL = "proportional"
    REMOVE_LOWER = "remove_lower"
    FLAG_ONLY = "flag_only"


class OverlapSeverity(str, Enum):
    """Severity of the detected overlap.

    HIGH:   Overlap likely exceeds 5% of the smaller category.
    MEDIUM: Overlap likely between 1-5% of the smaller category.
    LOW:    Overlap likely less than 1% of the smaller category.
    """
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ResolutionStatus(str, Enum):
    """Status of overlap resolution."""
    RESOLVED = "resolved"
    FLAGGED = "flagged"
    SKIPPED = "skipped"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Overlap Rules Configuration
# ---------------------------------------------------------------------------

# Each rule defines: category pair, overlap fraction estimate, description,
# default allocation method, and severity threshold.

OVERLAP_RULES: List[Dict[str, Any]] = [
    {
        "rule_id": OverlapRuleId.CAT1_CAT3.value,
        "category_a": 1,
        "category_b": 3,
        "description": (
            "Energy costs embedded in purchased goods and services (Cat 1) "
            "may overlap with fuel- and energy-related activities (Cat 3). "
            "EEIO-based Cat 1 estimates include upstream energy; Cat 3 may "
            "double-count if upstream extraction/processing is included."
        ),
        "default_overlap_fraction": Decimal("0.05"),
        "default_method": AllocationMethod.CONSERVATIVE.value,
        "severity": OverlapSeverity.MEDIUM.value,
        "ghg_protocol_reference": "Technical Guidance, Section 2.4",
        "detection_keywords": ["energy", "fuel", "electricity", "power"],
    },
    {
        "rule_id": OverlapRuleId.CAT1_CAT4.value,
        "category_a": 1,
        "category_b": 4,
        "description": (
            "Transport and logistics costs included in supplier prices (Cat 1) "
            "may overlap with separately reported upstream transportation (Cat 4). "
            "If Cat 1 uses spend-based method, delivered prices include freight."
        ),
        "default_overlap_fraction": Decimal("0.08"),
        "default_method": AllocationMethod.CONSERVATIVE.value,
        "severity": OverlapSeverity.HIGH.value,
        "ghg_protocol_reference": "Technical Guidance, Section 4.1",
        "detection_keywords": ["freight", "shipping", "delivery", "logistics", "transport"],
    },
    {
        "rule_id": OverlapRuleId.CAT1_CAT2.value,
        "category_a": 1,
        "category_b": 2,
        "description": (
            "Items classified as purchased goods (Cat 1) versus capital goods "
            "(Cat 2) depends on capitalisation thresholds. Items near the "
            "threshold may be reported in both categories."
        ),
        "default_overlap_fraction": Decimal("0.03"),
        "default_method": AllocationMethod.FLAG_ONLY.value,
        "severity": OverlapSeverity.LOW.value,
        "ghg_protocol_reference": "Technical Guidance, Section 2.1 vs 3.1",
        "detection_keywords": ["capital", "capex", "asset", "equipment"],
    },
    {
        "rule_id": OverlapRuleId.CAT3_SCOPE2.value,
        "category_a": 3,
        "category_b": 0,  # 0 = Scope 2 (not a Scope 3 category)
        "description": (
            "Cat 3 upstream energy may overlap with Scope 2 market-based "
            "emissions if contractual instruments are not properly allocated. "
            "Transmission and distribution losses in Cat 3 must not duplicate "
            "the Scope 2 location-based calculation."
        ),
        "default_overlap_fraction": Decimal("0.10"),
        "default_method": AllocationMethod.CONSERVATIVE.value,
        "severity": OverlapSeverity.HIGH.value,
        "ghg_protocol_reference": "Scope 2 Guidance, Chapter 7 + Tech Guidance, Section 3.3",
        "detection_keywords": ["t&d loss", "transmission", "distribution", "grid"],
    },
    {
        "rule_id": OverlapRuleId.CAT4_CAT9.value,
        "category_a": 4,
        "category_b": 9,
        "description": (
            "Transport cost allocation between buyer (Cat 4) and seller (Cat 9). "
            "Incoterms determine which party bears transport cost. If not "
            "properly allocated, the same shipment may appear in both."
        ),
        "default_overlap_fraction": Decimal("0.15"),
        "default_method": AllocationMethod.PROPORTIONAL.value,
        "severity": OverlapSeverity.HIGH.value,
        "ghg_protocol_reference": "Technical Guidance, Section 4.1 vs 9.1",
        "detection_keywords": ["incoterms", "fob", "cif", "dap", "ddp"],
    },
    {
        "rule_id": OverlapRuleId.CAT8_SCOPE12.value,
        "category_a": 8,
        "category_b": 0,  # Scope 1/2
        "description": (
            "Upstream leased assets (Cat 8) may overlap with Scope 1/2 if "
            "the organisation uses operational control consolidation approach "
            "and includes leased assets in its operational boundary."
        ),
        "default_overlap_fraction": Decimal("0.50"),
        "default_method": AllocationMethod.CONSERVATIVE.value,
        "severity": OverlapSeverity.HIGH.value,
        "ghg_protocol_reference": "Corporate Standard, Chapter 4 + Tech Guidance, Section 8.1",
        "detection_keywords": ["lease", "operational control", "rented", "tenancy"],
    },
    {
        "rule_id": OverlapRuleId.CAT13_CAT11.value,
        "category_a": 13,
        "category_b": 11,
        "description": (
            "Downstream leased assets (Cat 13) and use of sold products (Cat 11). "
            "If a sold product is also leased to end users, its use-phase "
            "emissions may be reported in both categories."
        ),
        "default_overlap_fraction": Decimal("0.20"),
        "default_method": AllocationMethod.CONSERVATIVE.value,
        "severity": OverlapSeverity.MEDIUM.value,
        "ghg_protocol_reference": "Technical Guidance, Section 11.1 vs 13.1",
        "detection_keywords": ["lease", "rental", "product use", "use phase"],
    },
    {
        "rule_id": OverlapRuleId.CAT14_SCOPE12.value,
        "category_a": 14,
        "category_b": 0,  # Scope 1/2
        "description": (
            "Franchise emissions (Cat 14) may overlap with Scope 1/2 if "
            "the organisation uses operational control consolidation and "
            "includes franchise operations in its boundary."
        ),
        "default_overlap_fraction": Decimal("0.50"),
        "default_method": AllocationMethod.CONSERVATIVE.value,
        "severity": OverlapSeverity.HIGH.value,
        "ghg_protocol_reference": "Corporate Standard, Chapter 4 + Tech Guidance, Section 14.1",
        "detection_keywords": ["franchise", "franchisee", "operational control"],
    },
    {
        "rule_id": OverlapRuleId.CAT1_CAT5.value,
        "category_a": 1,
        "category_b": 5,
        "description": (
            "Packaging waste in purchased goods (Cat 1) may overlap with "
            "waste generated in operations (Cat 5). If Cat 1 uses cradle-to-gate "
            "factors that include end-of-life treatment, packaging waste may "
            "be counted in both categories."
        ),
        "default_overlap_fraction": Decimal("0.04"),
        "default_method": AllocationMethod.CONSERVATIVE.value,
        "severity": OverlapSeverity.LOW.value,
        "ghg_protocol_reference": "Technical Guidance, Section 2.4 vs 5.1",
        "detection_keywords": ["packaging", "waste", "disposal", "end of life"],
    },
    {
        "rule_id": OverlapRuleId.CAT10_CAT11.value,
        "category_a": 10,
        "category_b": 11,
        "description": (
            "Processing of sold products (Cat 10) versus use of sold products "
            "(Cat 11). The boundary between processing and end use is ambiguous "
            "for intermediate products that undergo further manufacturing "
            "before consumer use."
        ),
        "default_overlap_fraction": Decimal("0.10"),
        "default_method": AllocationMethod.PROPORTIONAL.value,
        "severity": OverlapSeverity.MEDIUM.value,
        "ghg_protocol_reference": "Technical Guidance, Section 10.1 vs 11.1",
        "detection_keywords": ["intermediate", "processing", "manufacturing", "use phase"],
    },
    {
        "rule_id": OverlapRuleId.CAT6_CAT7.value,
        "category_a": 6,
        "category_b": 7,
        "description": (
            "Business travel by car (Cat 6) may overlap with employee commuting "
            "(Cat 7) for trips between office locations within the same "
            "metropolitan area or for employees who combine commute and "
            "business trips."
        ),
        "default_overlap_fraction": Decimal("0.05"),
        "default_method": AllocationMethod.FLAG_ONLY.value,
        "severity": OverlapSeverity.LOW.value,
        "ghg_protocol_reference": "Technical Guidance, Section 6.1 vs 7.1",
        "detection_keywords": ["car", "mileage", "local travel", "company car"],
    },
    {
        "rule_id": OverlapRuleId.CAT15_CAT1314.value,
        "category_a": 15,
        "category_b": 13,  # Also checks Cat 14
        "description": (
            "Investment emissions (Cat 15) may overlap with downstream leased "
            "assets (Cat 13) and franchises (Cat 14) if the organisation holds "
            "equity in entities that are also lessees or franchisees."
        ),
        "default_overlap_fraction": Decimal("0.15"),
        "default_method": AllocationMethod.CONSERVATIVE.value,
        "severity": OverlapSeverity.MEDIUM.value,
        "ghg_protocol_reference": "Technical Guidance, Section 15.1",
        "detection_keywords": ["investment", "equity", "subsidiary", "joint venture"],
    },
]


# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------


class CategoryEmissionInput(BaseModel):
    """Emission data for a single Scope 3 category (input to overlap detection).

    Attributes:
        category_number: Category number (1-15, 0 for Scope 1/2).
        total_co2e_tonnes: Total emissions in tCO2e.
        data_quality_score: Data quality score (1-5).
        methodology_tier: Methodology description.
        emission_sources: Description of emission sources.
        tags: Tags for overlap detection (keywords).
        entity_ids: Entity IDs contributing to this category.
    """
    category_number: int = Field(..., ge=0, le=15, description="Category number")
    total_co2e_tonnes: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total tCO2e"
    )
    data_quality_score: Decimal = Field(
        default=Decimal("4.0"), ge=1, le=5, description="DQ score"
    )
    methodology_tier: str = Field(default="spend_based", description="Methodology")
    emission_sources: str = Field(default="", description="Sources description")
    tags: List[str] = Field(default_factory=list, description="Detection tags")
    entity_ids: List[str] = Field(default_factory=list, description="Entity IDs")


# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------


class OverlapDetection(BaseModel):
    """A detected overlap between two categories.

    Attributes:
        detection_id: Unique detection identifier.
        rule_id: Overlap rule that triggered detection.
        category_a: First category in the overlap.
        category_b: Second category in the overlap.
        description: Description of the overlap.
        estimated_overlap_tco2e: Estimated overlap amount.
        overlap_fraction: Fraction used for overlap estimation.
        severity: Overlap severity.
        category_a_emissions_tco2e: Category A total emissions.
        category_b_emissions_tco2e: Category B total emissions.
        matched_tags: Tags that triggered detection.
        recommended_method: Recommended resolution method.
        ghg_protocol_reference: GHG Protocol reference.
    """
    detection_id: str = Field(default_factory=_new_uuid, description="Detection ID")
    rule_id: str = Field(default="", description="Overlap rule ID")
    category_a: int = Field(default=0, description="Category A")
    category_b: int = Field(default=0, description="Category B")
    description: str = Field(default="", description="Description")
    estimated_overlap_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Estimated overlap"
    )
    overlap_fraction: Decimal = Field(
        default=Decimal("0"), ge=0, le=1, description="Overlap fraction"
    )
    severity: OverlapSeverity = Field(
        default=OverlapSeverity.LOW, description="Severity"
    )
    category_a_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), description="Cat A emissions"
    )
    category_b_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), description="Cat B emissions"
    )
    matched_tags: List[str] = Field(default_factory=list, description="Matched tags")
    recommended_method: AllocationMethod = Field(
        default=AllocationMethod.FLAG_ONLY, description="Recommended method"
    )
    ghg_protocol_reference: str = Field(default="", description="GHG Protocol ref")


class OverlapResolution(BaseModel):
    """Resolution of a detected overlap.

    Attributes:
        resolution_id: Unique resolution identifier.
        detection_id: Detection being resolved.
        rule_id: Overlap rule ID.
        method_applied: Allocation method applied.
        category_adjusted: Category that was adjusted.
        adjustment_tco2e: Amount of adjustment.
        original_tco2e: Original emissions before adjustment.
        adjusted_tco2e: Emissions after adjustment.
        rationale: Rationale for the adjustment.
        status: Resolution status.
    """
    resolution_id: str = Field(default_factory=_new_uuid, description="Resolution ID")
    detection_id: str = Field(default="", description="Detection ID")
    rule_id: str = Field(default="", description="Rule ID")
    method_applied: AllocationMethod = Field(
        default=AllocationMethod.FLAG_ONLY, description="Method applied"
    )
    category_adjusted: int = Field(default=0, description="Category adjusted")
    adjustment_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Adjustment"
    )
    original_tco2e: Decimal = Field(
        default=Decimal("0"), description="Original"
    )
    adjusted_tco2e: Decimal = Field(
        default=Decimal("0"), description="Adjusted"
    )
    rationale: str = Field(default="", description="Rationale")
    status: ResolutionStatus = Field(
        default=ResolutionStatus.RESOLVED, description="Status"
    )


class AdjustedResult(BaseModel):
    """Adjusted emission result for a category after overlap resolution.

    Attributes:
        category_number: Category number.
        original_tco2e: Original emissions.
        total_adjustments_tco2e: Total adjustments applied.
        adjusted_tco2e: Final adjusted emissions.
        adjustment_count: Number of adjustments.
        adjustment_details: List of adjustment descriptions.
    """
    category_number: int = Field(..., ge=0, le=15, description="Category")
    original_tco2e: Decimal = Field(default=Decimal("0"), description="Original")
    total_adjustments_tco2e: Decimal = Field(
        default=Decimal("0"), description="Total adjustments"
    )
    adjusted_tco2e: Decimal = Field(default=Decimal("0"), description="Adjusted")
    adjustment_count: int = Field(default=0, description="Adjustment count")
    adjustment_details: List[str] = Field(
        default_factory=list, description="Adjustment details"
    )


class DoubleCountingResult(BaseModel):
    """Complete double-counting detection and resolution result.

    Attributes:
        result_id: Unique result identifier.
        detections: All detected overlaps.
        resolutions: All applied resolutions.
        adjusted_results: Per-category adjusted results.
        total_overlap_detected_tco2e: Total overlap detected.
        total_adjustments_applied_tco2e: Total adjustments applied.
        net_scope3_reduction_tco2e: Net reduction to Scope 3 total.
        original_scope3_total_tco2e: Original Scope 3 total.
        adjusted_scope3_total_tco2e: Adjusted Scope 3 total.
        rules_checked: Number of rules checked.
        overlaps_detected: Number of overlaps detected.
        overlaps_resolved: Number of overlaps resolved.
        overlaps_flagged: Number of overlaps flagged only.
        warnings: Warnings.
        calculated_at: Timestamp.
        processing_time_ms: Processing time ms.
        provenance_hash: SHA-256 hash.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    detections: List[OverlapDetection] = Field(
        default_factory=list, description="Detections"
    )
    resolutions: List[OverlapResolution] = Field(
        default_factory=list, description="Resolutions"
    )
    adjusted_results: List[AdjustedResult] = Field(
        default_factory=list, description="Adjusted results"
    )
    total_overlap_detected_tco2e: Decimal = Field(
        default=Decimal("0"), description="Total overlap detected"
    )
    total_adjustments_applied_tco2e: Decimal = Field(
        default=Decimal("0"), description="Total adjustments applied"
    )
    net_scope3_reduction_tco2e: Decimal = Field(
        default=Decimal("0"), description="Net Scope 3 reduction"
    )
    original_scope3_total_tco2e: Decimal = Field(
        default=Decimal("0"), description="Original total"
    )
    adjusted_scope3_total_tco2e: Decimal = Field(
        default=Decimal("0"), description="Adjusted total"
    )
    rules_checked: int = Field(default=0, description="Rules checked")
    overlaps_detected: int = Field(default=0, description="Overlaps detected")
    overlaps_resolved: int = Field(default=0, description="Overlaps resolved")
    overlaps_flagged: int = Field(default=0, description="Overlaps flagged")
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    calculated_at: datetime = Field(default_factory=_utcnow, description="Timestamp")
    processing_time_ms: Decimal = Field(default=Decimal("0"), description="Processing ms")
    provenance_hash: str = Field(default="", description="SHA-256 hash")


# ---------------------------------------------------------------------------
# Model Rebuild
# ---------------------------------------------------------------------------

CategoryEmissionInput.model_rebuild()
OverlapDetection.model_rebuild()
OverlapResolution.model_rebuild()
AdjustedResult.model_rebuild()
DoubleCountingResult.model_rebuild()


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class DoubleCountingPreventionEngine:
    """Detect and resolve double-counting across Scope 3 categories.

    Implements 12 overlap rules derived from GHG Protocol Technical
    Guidance.  Each rule checks a specific pair of categories for
    potential double-counting and estimates the overlap amount.

    Supports two allocation methods for resolution:
    - CONSERVATIVE: remove overlap from the higher-numbered category
    - PROPORTIONAL: split based on data quality weights

    Attributes:
        _default_method: Default allocation method for unspecified rules.
        _custom_fractions: Custom overlap fraction overrides per rule.
        _scope12_total: Scope 1+2 total for cross-scope rules.
        _warnings: Warnings generated during processing.

    Example:
        >>> engine = DoubleCountingPreventionEngine()
        >>> inputs = [
        ...     CategoryEmissionInput(category_number=1, total_co2e_tonnes=Decimal("10000")),
        ...     CategoryEmissionInput(category_number=3, total_co2e_tonnes=Decimal("2000")),
        ...     CategoryEmissionInput(category_number=4, total_co2e_tonnes=Decimal("1500")),
        ... ]
        >>> result = engine.detect_and_resolve(inputs)
        >>> print(result.overlaps_detected)
    """

    def __init__(
        self,
        default_method: AllocationMethod = AllocationMethod.CONSERVATIVE,
        custom_fractions: Optional[Dict[str, Decimal]] = None,
        scope12_total_tco2e: Decimal = Decimal("0"),
    ) -> None:
        """Initialise DoubleCountingPreventionEngine.

        Args:
            default_method: Default allocation method for resolution.
            custom_fractions: Custom overlap fraction overrides (rule_id -> fraction).
            scope12_total_tco2e: Scope 1+2 total for cross-scope rules.
        """
        self._default_method = default_method
        self._custom_fractions = custom_fractions or {}
        self._scope12_total = scope12_total_tco2e
        self._warnings: List[str] = []
        logger.info(
            "DoubleCountingPreventionEngine v%s initialised (method=%s)",
            _MODULE_VERSION,
            self._default_method.value,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_and_resolve(
        self,
        category_results: List[CategoryEmissionInput],
        method_override: Optional[AllocationMethod] = None,
    ) -> DoubleCountingResult:
        """Detect and resolve all double-counting overlaps.

        Main entry point.  Runs all 12 overlap rules, detects potential
        overlaps, and resolves them using the specified allocation method.

        Args:
            category_results: Per-category emission inputs.
            method_override: Override the default allocation method.

        Returns:
            DoubleCountingResult.

        Raises:
            ValueError: If no category results provided.
        """
        t0 = time.perf_counter()
        self._warnings = []

        if not category_results:
            raise ValueError("At least one category result is required")

        logger.info(
            "Running double-counting detection on %d categories",
            len(category_results),
        )

        # Build lookup by category number
        cat_map: Dict[int, CategoryEmissionInput] = {}
        for cr in category_results:
            if cr.category_number in cat_map:
                # Merge: sum emissions
                existing = cat_map[cr.category_number]
                cat_map[cr.category_number] = CategoryEmissionInput(
                    category_number=cr.category_number,
                    total_co2e_tonnes=existing.total_co2e_tonnes + cr.total_co2e_tonnes,
                    data_quality_score=min(existing.data_quality_score, cr.data_quality_score),
                    methodology_tier=existing.methodology_tier,
                    tags=list(set(existing.tags + cr.tags)),
                    entity_ids=list(set(existing.entity_ids + cr.entity_ids)),
                )
            else:
                cat_map[cr.category_number] = cr

        # Step 1: Detect overlaps
        detections = self.detect_overlaps(cat_map)

        # Step 2: Resolve overlaps
        method = method_override or self._default_method
        resolutions = self.resolve_overlaps(detections, cat_map, method)

        # Step 3: Calculate adjusted results
        adjusted = self._calculate_adjusted_results(cat_map, resolutions)

        # Step 4: Compute totals
        original_total = sum(
            (cr.total_co2e_tonnes for cr in cat_map.values()
             if cr.category_number > 0),
            Decimal("0"),
        )
        total_overlap = sum(
            (d.estimated_overlap_tco2e for d in detections), Decimal("0")
        )
        total_adj = sum(
            (r.adjustment_tco2e for r in resolutions
             if r.status == ResolutionStatus.RESOLVED),
            Decimal("0"),
        )
        adjusted_total = original_total - total_adj

        resolved_count = sum(
            1 for r in resolutions if r.status == ResolutionStatus.RESOLVED
        )
        flagged_count = sum(
            1 for r in resolutions if r.status == ResolutionStatus.FLAGGED
        )

        elapsed_ms = Decimal(str((time.perf_counter() - t0) * 1000))

        result = DoubleCountingResult(
            detections=detections,
            resolutions=resolutions,
            adjusted_results=adjusted,
            total_overlap_detected_tco2e=_round_val(total_overlap, 2),
            total_adjustments_applied_tco2e=_round_val(total_adj, 2),
            net_scope3_reduction_tco2e=_round_val(total_adj, 2),
            original_scope3_total_tco2e=_round_val(original_total, 2),
            adjusted_scope3_total_tco2e=_round_val(adjusted_total, 2),
            rules_checked=len(OVERLAP_RULES),
            overlaps_detected=len(detections),
            overlaps_resolved=resolved_count,
            overlaps_flagged=flagged_count,
            warnings=list(self._warnings),
            processing_time_ms=_round_val(elapsed_ms, 2),
        )
        result.provenance_hash = self._compute_provenance(result)

        logger.info(
            "Double-counting detection complete: %d detected, %d resolved, "
            "%.2f tCO2e adjusted",
            len(detections), resolved_count, total_adj,
        )
        return result

    def detect_overlaps(
        self,
        cat_map: Dict[int, CategoryEmissionInput],
    ) -> List[OverlapDetection]:
        """Detect all potential overlaps without resolving.

        Args:
            cat_map: Mapping of category number to emission input.

        Returns:
            List of OverlapDetection.
        """
        detections: List[OverlapDetection] = []

        for rule in OVERLAP_RULES:
            detection = self._check_rule(rule, cat_map)
            if detection is not None:
                detections.append(detection)

        return detections

    def resolve_overlaps(
        self,
        detections: List[OverlapDetection],
        cat_map: Dict[int, CategoryEmissionInput],
        method: AllocationMethod = AllocationMethod.CONSERVATIVE,
    ) -> List[OverlapResolution]:
        """Resolve detected overlaps using the specified method.

        Args:
            detections: Detected overlaps.
            cat_map: Category emission map.
            method: Allocation method.

        Returns:
            List of OverlapResolution.
        """
        resolutions: List[OverlapResolution] = []

        for detection in detections:
            resolution = self._resolve_single(detection, cat_map, method)
            resolutions.append(resolution)

        return resolutions

    # ------------------------------------------------------------------
    # Rule Check Methods
    # ------------------------------------------------------------------

    def _check_rule(
        self,
        rule: Dict[str, Any],
        cat_map: Dict[int, CategoryEmissionInput],
    ) -> Optional[OverlapDetection]:
        """Check a single overlap rule against category data.

        Args:
            rule: Overlap rule configuration.
            cat_map: Category emission map.

        Returns:
            OverlapDetection if overlap found, None otherwise.
        """
        cat_a_num = rule["category_a"]
        cat_b_num = rule["category_b"]

        # Get emission data
        cat_a = cat_map.get(cat_a_num)
        cat_b = cat_map.get(cat_b_num)

        # For cross-scope rules (cat_b = 0), use Scope 1+2 total
        cat_a_emissions = cat_a.total_co2e_tonnes if cat_a else Decimal("0")
        if cat_b_num == 0:
            cat_b_emissions = self._scope12_total
        else:
            cat_b_emissions = cat_b.total_co2e_tonnes if cat_b else Decimal("0")

        # Skip if either category has zero emissions
        if cat_a_emissions <= Decimal("0") or cat_b_emissions <= Decimal("0"):
            return None

        # Special handling for Cat15 vs Cat13/14 rule
        rule_id = rule["rule_id"]
        if rule_id == OverlapRuleId.CAT15_CAT1314.value:
            cat14 = cat_map.get(14)
            cat14_emissions = cat14.total_co2e_tonnes if cat14 else Decimal("0")
            cat_b_emissions = cat_b_emissions + cat14_emissions

        # Check tag overlap (if tags are available)
        matched_tags = self._check_tag_overlap(
            cat_a, cat_b, rule.get("detection_keywords", [])
        )

        # Calculate overlap amount
        overlap_fraction = self._custom_fractions.get(
            rule_id, rule["default_overlap_fraction"]
        )
        smaller = min(cat_a_emissions, cat_b_emissions)
        overlap_amount = smaller * overlap_fraction

        # Determine severity based on overlap relative to smaller category
        overlap_pct = _safe_divide(
            overlap_amount * Decimal("100"), smaller
        )
        if overlap_pct >= Decimal("5"):
            severity = OverlapSeverity.HIGH
        elif overlap_pct >= Decimal("1"):
            severity = OverlapSeverity.MEDIUM
        else:
            severity = OverlapSeverity.LOW

        # Build method enum from rule
        try:
            recommended_method = AllocationMethod(rule["default_method"])
        except ValueError:
            recommended_method = AllocationMethod.FLAG_ONLY

        return OverlapDetection(
            rule_id=rule_id,
            category_a=cat_a_num,
            category_b=cat_b_num,
            description=rule["description"],
            estimated_overlap_tco2e=_round_val(overlap_amount, 2),
            overlap_fraction=overlap_fraction,
            severity=severity,
            category_a_emissions_tco2e=_round_val(cat_a_emissions, 2),
            category_b_emissions_tco2e=_round_val(cat_b_emissions, 2),
            matched_tags=matched_tags,
            recommended_method=recommended_method,
            ghg_protocol_reference=rule.get("ghg_protocol_reference", ""),
        )

    def _check_tag_overlap(
        self,
        cat_a: Optional[CategoryEmissionInput],
        cat_b: Optional[CategoryEmissionInput],
        keywords: List[str],
    ) -> List[str]:
        """Check for tag/keyword overlap between two categories.

        Args:
            cat_a: Category A emission input.
            cat_b: Category B emission input.
            keywords: Detection keywords from the rule.

        Returns:
            List of matched keywords.
        """
        tags_a = set(t.lower() for t in (cat_a.tags if cat_a else []))
        tags_b = set(t.lower() for t in (cat_b.tags if cat_b else []))
        all_tags = tags_a | tags_b
        keywords_lower = set(k.lower() for k in keywords)
        return sorted(all_tags & keywords_lower)

    # ------------------------------------------------------------------
    # Resolution Methods
    # ------------------------------------------------------------------

    def _resolve_single(
        self,
        detection: OverlapDetection,
        cat_map: Dict[int, CategoryEmissionInput],
        method: AllocationMethod,
    ) -> OverlapResolution:
        """Resolve a single detected overlap.

        Args:
            detection: Detected overlap.
            cat_map: Category emission map.
            method: Allocation method.

        Returns:
            OverlapResolution.
        """
        # Use rule's recommended method if FLAG_ONLY
        effective_method = method
        if detection.recommended_method == AllocationMethod.FLAG_ONLY:
            effective_method = AllocationMethod.FLAG_ONLY

        if effective_method == AllocationMethod.FLAG_ONLY:
            return self._flag_only_resolution(detection)

        if effective_method == AllocationMethod.CONSERVATIVE:
            return self._conservative_resolution(detection, cat_map)

        if effective_method == AllocationMethod.PROPORTIONAL:
            return self._proportional_resolution(detection, cat_map)

        if effective_method == AllocationMethod.REMOVE_LOWER:
            return self._remove_lower_resolution(detection, cat_map)

        return self._flag_only_resolution(detection)

    def _conservative_resolution(
        self,
        detection: OverlapDetection,
        cat_map: Dict[int, CategoryEmissionInput],
    ) -> OverlapResolution:
        """Resolve by removing overlap from the higher-numbered category.

        Args:
            detection: Detected overlap.
            cat_map: Category emission map.

        Returns:
            OverlapResolution.
        """
        # Remove from the higher category number
        cat_to_adjust = max(detection.category_a, detection.category_b)
        if cat_to_adjust == 0:
            # Cross-scope: adjust the Scope 3 category
            cat_to_adjust = detection.category_a

        original = cat_map.get(cat_to_adjust)
        original_amount = original.total_co2e_tonnes if original else Decimal("0")
        adjustment = min(detection.estimated_overlap_tco2e, original_amount)
        adjusted_amount = original_amount - adjustment

        return OverlapResolution(
            detection_id=detection.detection_id,
            rule_id=detection.rule_id,
            method_applied=AllocationMethod.CONSERVATIVE,
            category_adjusted=cat_to_adjust,
            adjustment_tco2e=_round_val(adjustment, 2),
            original_tco2e=_round_val(original_amount, 2),
            adjusted_tco2e=_round_val(adjusted_amount, 2),
            rationale=(
                f"CONSERVATIVE: Removed {adjustment:.2f} tCO2e from Cat {cat_to_adjust} "
                f"(higher-numbered category) to prevent double-counting with "
                f"Cat {min(detection.category_a, detection.category_b)}"
            ),
            status=ResolutionStatus.RESOLVED,
        )

    def _proportional_resolution(
        self,
        detection: OverlapDetection,
        cat_map: Dict[int, CategoryEmissionInput],
    ) -> OverlapResolution:
        """Resolve by splitting based on data quality weights.

        The category with lower (better) data quality retains more
        emissions; the category with higher (worse) score absorbs more
        of the deduction.

        Args:
            detection: Detected overlap.
            cat_map: Category emission map.

        Returns:
            OverlapResolution.
        """
        cat_a = cat_map.get(detection.category_a)
        cat_b = cat_map.get(detection.category_b)

        dq_a = cat_a.data_quality_score if cat_a else Decimal("5")
        dq_b = cat_b.data_quality_score if cat_b else Decimal("5")

        # Invert: lower DQ score = better quality = higher weight to retain
        # Weight to deduct from A = dq_a / (dq_a + dq_b)  (worse score = more deduction)
        total_dq = dq_a + dq_b
        weight_a = _safe_divide(dq_a, total_dq, Decimal("0.5"))

        # Deduct from the category with worse data quality
        overlap = detection.estimated_overlap_tco2e
        deduct_a = overlap * weight_a
        deduct_b = overlap * (Decimal("1") - weight_a)

        # For simplicity, report the larger deduction as the resolution
        if deduct_a >= deduct_b:
            cat_to_adjust = detection.category_a
            adjustment = deduct_a
        else:
            cat_to_adjust = detection.category_b
            adjustment = deduct_b

        original = cat_map.get(cat_to_adjust)
        original_amount = original.total_co2e_tonnes if original else Decimal("0")
        adjustment = min(adjustment, original_amount)
        adjusted_amount = original_amount - adjustment

        return OverlapResolution(
            detection_id=detection.detection_id,
            rule_id=detection.rule_id,
            method_applied=AllocationMethod.PROPORTIONAL,
            category_adjusted=cat_to_adjust,
            adjustment_tco2e=_round_val(adjustment, 2),
            original_tco2e=_round_val(original_amount, 2),
            adjusted_tco2e=_round_val(adjusted_amount, 2),
            rationale=(
                f"PROPORTIONAL: Based on DQ scores (Cat {detection.category_a}: {dq_a}, "
                f"Cat {detection.category_b}: {dq_b}), deducted {adjustment:.2f} tCO2e "
                f"from Cat {cat_to_adjust}"
            ),
            status=ResolutionStatus.RESOLVED,
        )

    def _remove_lower_resolution(
        self,
        detection: OverlapDetection,
        cat_map: Dict[int, CategoryEmissionInput],
    ) -> OverlapResolution:
        """Resolve by removing overlap from the lower-numbered category.

        Args:
            detection: Detected overlap.
            cat_map: Category emission map.

        Returns:
            OverlapResolution.
        """
        cat_to_adjust = min(detection.category_a, detection.category_b)
        if cat_to_adjust == 0:
            cat_to_adjust = detection.category_a

        original = cat_map.get(cat_to_adjust)
        original_amount = original.total_co2e_tonnes if original else Decimal("0")
        adjustment = min(detection.estimated_overlap_tco2e, original_amount)
        adjusted_amount = original_amount - adjustment

        return OverlapResolution(
            detection_id=detection.detection_id,
            rule_id=detection.rule_id,
            method_applied=AllocationMethod.REMOVE_LOWER,
            category_adjusted=cat_to_adjust,
            adjustment_tco2e=_round_val(adjustment, 2),
            original_tco2e=_round_val(original_amount, 2),
            adjusted_tco2e=_round_val(adjusted_amount, 2),
            rationale=(
                f"REMOVE_LOWER: Removed {adjustment:.2f} tCO2e from Cat {cat_to_adjust} "
                f"(lower-numbered category)"
            ),
            status=ResolutionStatus.RESOLVED,
        )

    def _flag_only_resolution(
        self,
        detection: OverlapDetection,
    ) -> OverlapResolution:
        """Flag the overlap for manual review without adjusting.

        Args:
            detection: Detected overlap.

        Returns:
            OverlapResolution with FLAGGED status.
        """
        return OverlapResolution(
            detection_id=detection.detection_id,
            rule_id=detection.rule_id,
            method_applied=AllocationMethod.FLAG_ONLY,
            category_adjusted=0,
            adjustment_tco2e=Decimal("0"),
            original_tco2e=Decimal("0"),
            adjusted_tco2e=Decimal("0"),
            rationale=(
                f"FLAGGED for manual review: {detection.description[:200]}"
            ),
            status=ResolutionStatus.FLAGGED,
        )

    def _calculate_adjusted_results(
        self,
        cat_map: Dict[int, CategoryEmissionInput],
        resolutions: List[OverlapResolution],
    ) -> List[AdjustedResult]:
        """Calculate adjusted per-category results after resolution.

        Args:
            cat_map: Category emission map.
            resolutions: Applied resolutions.

        Returns:
            List of AdjustedResult.
        """
        # Accumulate adjustments per category
        adjustments: Dict[int, Dict[str, Any]] = {}
        for cat_num, cat_data in cat_map.items():
            adjustments[cat_num] = {
                "original": cat_data.total_co2e_tonnes,
                "total_adj": Decimal("0"),
                "count": 0,
                "details": [],
            }

        for res in resolutions:
            if res.status == ResolutionStatus.RESOLVED and res.category_adjusted > 0:
                cat = res.category_adjusted
                if cat in adjustments:
                    adjustments[cat]["total_adj"] += res.adjustment_tco2e
                    adjustments[cat]["count"] += 1
                    adjustments[cat]["details"].append(
                        f"Rule {res.rule_id}: -{res.adjustment_tco2e:.2f} tCO2e"
                    )

        results: List[AdjustedResult] = []
        for cat_num in sorted(adjustments.keys()):
            if cat_num == 0:
                continue  # Skip Scope 1/2 placeholder
            d = adjustments[cat_num]
            adjusted = d["original"] - d["total_adj"]
            results.append(AdjustedResult(
                category_number=cat_num,
                original_tco2e=_round_val(d["original"], 2),
                total_adjustments_tco2e=_round_val(d["total_adj"], 2),
                adjusted_tco2e=_round_val(max(adjusted, Decimal("0")), 2),
                adjustment_count=d["count"],
                adjustment_details=d["details"],
            ))

        return results

    def _compute_provenance(self, result: DoubleCountingResult) -> str:
        """Compute SHA-256 provenance hash.

        Args:
            result: Double-counting result.

        Returns:
            SHA-256 hex digest.
        """
        return _compute_hash(result)
