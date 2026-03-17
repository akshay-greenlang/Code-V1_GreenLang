# -*- coding: utf-8 -*-
"""
TaxonomyAlignmentRatioEngine - PACK-010 SFDR Article 8 Engine 2
=================================================================

Calculates EU Taxonomy alignment percentages for Article 8 financial products
as required by the SFDR Regulatory Technical Standards (RTS).

Article 8 products that promote environmental or social characteristics must
disclose the proportion of their investments that are aligned with the EU
Taxonomy. This engine computes fund-level alignment ratios, provides
breakdowns by environmental objective, prevents double-counting across
objectives, and generates data for the mandatory RTS pie chart visualization.

Key Regulatory References:
    - Regulation (EU) 2019/2088 (SFDR) Article 6, 8, 9
    - Delegated Regulation (EU) 2022/1288 (SFDR RTS) Annexes II-V
    - Regulation (EU) 2020/852 (Taxonomy Regulation) Articles 3, 9

Zero-Hallucination:
    - All ratios use deterministic Decimal arithmetic
    - Double-counting prevention via primary objective assignment
    - SHA-256 provenance hashing on every result
    - No LLM involvement in any numeric calculation path

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-010 SFDR Article 8
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

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


def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def _round_val(value: Decimal, places: int = 6) -> float:
    """Round a Decimal to specified places and return float."""
    rounded = value.quantize(Decimal(10) ** -places, rounding=ROUND_HALF_UP)
    return float(rounded)


def _safe_divide(numerator: Decimal, denominator: Decimal) -> Decimal:
    """Divide safely, returning zero when denominator is zero."""
    if denominator == Decimal("0"):
        return Decimal("0")
    return numerator / denominator


def _pct(numerator: Decimal, denominator: Decimal) -> Decimal:
    """Calculate percentage safely."""
    return _safe_divide(numerator, denominator) * Decimal("100")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EnvironmentalObjective(str, Enum):
    """EU Taxonomy six environmental objectives (Article 9)."""

    CCM = "CCM"   # Climate Change Mitigation
    CCA = "CCA"   # Climate Change Adaptation
    WTR = "WTR"   # Sustainable Use & Protection of Water & Marine Resources
    CE = "CE"     # Transition to a Circular Economy
    PPC = "PPC"   # Pollution Prevention and Control
    BIO = "BIO"   # Protection & Restoration of Biodiversity & Ecosystems


OBJECTIVE_NAMES: Dict[str, str] = {
    "CCM": "Climate Change Mitigation",
    "CCA": "Climate Change Adaptation",
    "WTR": "Sustainable Use of Water and Marine Resources",
    "CE": "Transition to a Circular Economy",
    "PPC": "Pollution Prevention and Control",
    "BIO": "Protection and Restoration of Biodiversity and Ecosystems",
}


class AlignmentCategory(str, Enum):
    """Taxonomy alignment classification for a holding."""

    ALIGNED = "ALIGNED"                 # Meets SC + DNSH + MSS
    ELIGIBLE_NOT_ALIGNED = "ELIGIBLE_NOT_ALIGNED"  # Eligible but fails alignment test
    NOT_ELIGIBLE = "NOT_ELIGIBLE"       # Not covered by Taxonomy
    NOT_ASSESSED = "NOT_ASSESSED"       # Insufficient data to determine
    SOVEREIGN = "SOVEREIGN"             # Sovereign bonds (excluded from Taxonomy)
    CASH_DERIVATIVES = "CASH_DERIVATIVES"  # Cash and derivatives (excluded)


class GASExposureType(str, Enum):
    """Types of natural gas/nuclear energy exposure under CDA."""

    FOSSIL_GAS = "fossil_gas"
    NUCLEAR = "nuclear"
    NONE = "none"


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class TaxonomyAlignmentConfig(BaseModel):
    """Configuration for the Taxonomy Alignment Ratio Engine.

    Attributes:
        total_nav_eur: Total Net Asset Value of the fund in EUR.
        reporting_date: Date for which alignment is calculated.
        pre_contractual_commitment_pct: Minimum alignment committed in prospectus.
        include_sovereign_in_denominator: Whether to include sovereign bonds in NAV.
        include_derivatives_in_denominator: Whether to include derivatives in NAV.
        gas_nuclear_disclosure_enabled: Whether to enable CDA gas/nuclear split.
    """

    total_nav_eur: float = Field(
        ..., gt=0,
        description="Total Net Asset Value of the fund (EUR)",
    )
    reporting_date: datetime = Field(
        ..., description="Reporting reference date",
    )
    pre_contractual_commitment_pct: Optional[float] = Field(
        None, ge=0.0, le=100.0,
        description="Pre-contractual minimum Taxonomy alignment commitment (%)",
    )
    include_sovereign_in_denominator: bool = Field(
        default=False,
        description="Include sovereign bonds in NAV denominator",
    )
    include_derivatives_in_denominator: bool = Field(
        default=False,
        description="Include cash/derivatives in NAV denominator",
    )
    gas_nuclear_disclosure_enabled: bool = Field(
        default=True,
        description="Enable Complementary Delegated Act gas/nuclear disclosure",
    )


class HoldingAlignmentData(BaseModel):
    """Taxonomy alignment data for a single portfolio holding.

    Captures the alignment assessment result at the holding level,
    including eligible/aligned revenue, CapEx, and OpEx KPIs.

    Attributes:
        holding_id: Unique holding identifier.
        holding_name: Name of the holding.
        holding_type: CORPORATE, SOVEREIGN, or CASH_DERIVATIVES.
        value_eur: Current market value in EUR.
        alignment_category: Alignment classification.
        aligned_revenue_pct: Percentage of revenue that is Taxonomy-aligned.
        aligned_capex_pct: Percentage of CapEx that is Taxonomy-aligned.
        aligned_opex_pct: Percentage of OpEx that is Taxonomy-aligned.
        eligible_revenue_pct: Percentage of revenue that is Taxonomy-eligible.
        eligible_capex_pct: Percentage of CapEx that is Taxonomy-eligible.
        eligible_opex_pct: Percentage of OpEx that is Taxonomy-eligible.
        primary_objective: Primary environmental objective for alignment.
        contributing_objectives: All objectives the activity contributes to.
        dnsh_passed: Whether DNSH assessment passed.
        minimum_safeguards_passed: Whether minimum safeguards are met.
        gas_nuclear_exposure: Type of gas/nuclear exposure (CDA).
    """

    holding_id: str = Field(
        ..., min_length=1, max_length=200,
        description="Unique holding identifier (ISIN, LEI, etc.)",
    )
    holding_name: str = Field(
        ..., min_length=1, max_length=500,
        description="Holding name",
    )
    holding_type: str = Field(
        ..., description="CORPORATE, SOVEREIGN, or CASH_DERIVATIVES",
    )
    value_eur: float = Field(
        ..., gt=0,
        description="Current market value (EUR)",
    )
    alignment_category: AlignmentCategory = Field(
        default=AlignmentCategory.NOT_ASSESSED,
        description="Taxonomy alignment classification",
    )
    aligned_revenue_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Taxonomy-aligned revenue share (%)",
    )
    aligned_capex_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Taxonomy-aligned CapEx share (%)",
    )
    aligned_opex_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Taxonomy-aligned OpEx share (%)",
    )
    eligible_revenue_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Taxonomy-eligible revenue share (%)",
    )
    eligible_capex_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Taxonomy-eligible CapEx share (%)",
    )
    eligible_opex_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Taxonomy-eligible OpEx share (%)",
    )
    primary_objective: Optional[EnvironmentalObjective] = Field(
        None, description="Primary environmental objective",
    )
    contributing_objectives: List[EnvironmentalObjective] = Field(
        default_factory=list,
        description="All objectives the activity contributes to",
    )
    dnsh_passed: Optional[bool] = Field(
        None, description="Whether DNSH passed for all non-SC objectives",
    )
    minimum_safeguards_passed: Optional[bool] = Field(
        None, description="Whether minimum safeguards (UNGC, OECD, ILO) are met",
    )
    gas_nuclear_exposure: GASExposureType = Field(
        default=GASExposureType.NONE,
        description="Gas/nuclear exposure type under CDA",
    )

    @field_validator("holding_type")
    @classmethod
    def validate_holding_type(cls, v: str) -> str:
        """Validate holding type."""
        allowed = {"CORPORATE", "SOVEREIGN", "CASH_DERIVATIVES"}
        upper = v.strip().upper()
        if upper not in allowed:
            raise ValueError(f"holding_type must be one of {allowed}, got '{v}'")
        return upper


class ObjectiveBreakdown(BaseModel):
    """Alignment breakdown for a single environmental objective."""

    objective: EnvironmentalObjective = Field(
        ..., description="Environmental objective",
    )
    objective_name: str = Field(
        ..., description="Full objective name",
    )
    aligned_value_eur: float = Field(
        ..., ge=0, description="Aligned value in EUR",
    )
    aligned_pct_of_nav: float = Field(
        ..., ge=0.0, le=100.0,
        description="Aligned percentage of total NAV",
    )
    aligned_revenue_pct: float = Field(
        ..., ge=0.0, le=100.0,
        description="Weighted average aligned revenue (%)",
    )
    aligned_capex_pct: float = Field(
        ..., ge=0.0, le=100.0,
        description="Weighted average aligned CapEx (%)",
    )
    aligned_opex_pct: float = Field(
        ..., ge=0.0, le=100.0,
        description="Weighted average aligned OpEx (%)",
    )
    holding_count: int = Field(
        ..., ge=0, description="Number of aligned holdings",
    )


class PieChartSlice(BaseModel):
    """A single slice for the mandatory SFDR RTS pie chart."""

    label: str = Field(..., description="Slice label")
    value_pct: float = Field(..., ge=0.0, le=100.0, description="Slice percentage")
    value_eur: float = Field(..., ge=0, description="Slice value in EUR")
    color_hint: str = Field(default="", description="Suggested color code")


class CommitmentAdherence(BaseModel):
    """Tracks whether actual alignment meets pre-contractual commitment."""

    pre_contractual_commitment_pct: Optional[float] = Field(
        None, description="Pre-contractual minimum alignment (%)",
    )
    actual_alignment_pct: float = Field(
        ..., ge=0.0, le=100.0,
        description="Actual computed alignment (%)",
    )
    meets_commitment: Optional[bool] = Field(
        None, description="Whether actual >= commitment",
    )
    gap_pct: Optional[float] = Field(
        None, description="Gap between commitment and actual (if negative = shortfall)",
    )
    status: str = Field(
        default="not_applicable",
        description="COMPLIANT, SHORTFALL, or not_applicable",
    )


class AlignmentResult(BaseModel):
    """Complete Taxonomy alignment calculation result for a fund.

    Contains the fund-level alignment ratios across all three KPIs
    (revenue, CapEx, OpEx), objective-level breakdowns, gas/nuclear
    disclosure, pie chart data, and commitment adherence.
    """

    result_id: str = Field(
        default_factory=_new_uuid, description="Unique result identifier",
    )
    fund_name: Optional[str] = Field(
        None, description="Fund or product name",
    )
    reporting_date: datetime = Field(
        ..., description="Reporting reference date",
    )
    total_nav_eur: float = Field(
        ..., gt=0, description="Total NAV",
    )
    effective_denominator_eur: float = Field(
        ..., ge=0,
        description="Effective denominator after sovereign/derivative exclusions",
    )
    total_holdings: int = Field(
        ..., ge=0, description="Total number of holdings",
    )

    # --- Fund-level alignment ratios ---
    aligned_revenue_pct: float = Field(
        ..., ge=0.0, le=100.0,
        description="Fund-level Taxonomy-aligned revenue ratio (%)",
    )
    aligned_capex_pct: float = Field(
        ..., ge=0.0, le=100.0,
        description="Fund-level Taxonomy-aligned CapEx ratio (%)",
    )
    aligned_opex_pct: float = Field(
        ..., ge=0.0, le=100.0,
        description="Fund-level Taxonomy-aligned OpEx ratio (%)",
    )

    # --- Eligible ratios ---
    eligible_revenue_pct: float = Field(
        ..., ge=0.0, le=100.0,
        description="Fund-level Taxonomy-eligible revenue ratio (%)",
    )
    eligible_capex_pct: float = Field(
        ..., ge=0.0, le=100.0,
        description="Fund-level Taxonomy-eligible CapEx ratio (%)",
    )
    eligible_opex_pct: float = Field(
        ..., ge=0.0, le=100.0,
        description="Fund-level Taxonomy-eligible OpEx ratio (%)",
    )

    # --- Category proportions ---
    aligned_value_eur: float = Field(
        ..., ge=0, description="Total aligned value (EUR)",
    )
    eligible_not_aligned_value_eur: float = Field(
        ..., ge=0, description="Eligible but not aligned (EUR)",
    )
    not_eligible_value_eur: float = Field(
        ..., ge=0, description="Not Taxonomy-eligible (EUR)",
    )
    sovereign_value_eur: float = Field(
        ..., ge=0, description="Sovereign bond value (EUR)",
    )
    cash_derivatives_value_eur: float = Field(
        ..., ge=0, description="Cash/derivatives value (EUR)",
    )
    not_assessed_value_eur: float = Field(
        ..., ge=0, description="Not assessed value (EUR)",
    )

    # --- Breakdowns ---
    objective_breakdown: Dict[str, ObjectiveBreakdown] = Field(
        default_factory=dict,
        description="Alignment breakdown by environmental objective",
    )

    # --- Gas / Nuclear ---
    gas_aligned_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Taxonomy-aligned investments in fossil gas (%)",
    )
    nuclear_aligned_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Taxonomy-aligned investments in nuclear (%)",
    )

    # --- Commitment ---
    commitment_adherence: Optional[CommitmentAdherence] = Field(
        None, description="Pre-contractual commitment adherence",
    )

    # --- Pie chart ---
    pie_chart_data: List[PieChartSlice] = Field(
        default_factory=list,
        description="Data for mandatory RTS pie chart visualization",
    )

    # --- Provenance ---
    calculation_timestamp: datetime = Field(
        default_factory=_utcnow, description="When calculated",
    )
    processing_time_ms: float = Field(
        default=0.0, description="Processing time (ms)",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION, description="Engine version",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash",
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class TaxonomyAlignmentRatioEngine:
    """EU Taxonomy Alignment Ratio Calculator for SFDR Article 8 Products.

    Calculates fund-level Taxonomy alignment ratios across revenue, CapEx,
    and OpEx KPIs. Provides objective-level breakdowns with double-counting
    prevention, gas/nuclear CDA disclosure, pie chart data generation, and
    pre-contractual commitment adherence checking.

    Zero-Hallucination Guarantees:
        - All ratios use deterministic Decimal arithmetic
        - Double-counting prevented by primary objective assignment
        - SHA-256 provenance hashing on every result
        - No LLM involvement in any numeric path

    Attributes:
        config: Engine configuration.
        _calculation_count: Running count of calculations performed.

    Example:
        >>> config = TaxonomyAlignmentConfig(
        ...     total_nav_eur=100_000_000.0,
        ...     reporting_date=datetime(2025, 12, 31, tzinfo=timezone.utc),
        ... )
        >>> engine = TaxonomyAlignmentRatioEngine(config)
        >>> result = engine.calculate_alignment_ratio(holdings)
        >>> print(f"Alignment: {result.aligned_revenue_pct}%")
    """

    def __init__(self, config: TaxonomyAlignmentConfig) -> None:
        """Initialize the Taxonomy Alignment Ratio Engine.

        Args:
            config: Configuration including NAV, reporting date, and options.
        """
        self.config = config
        self._calculation_count: int = 0
        self._nav_decimal = _decimal(config.total_nav_eur)
        logger.info(
            "TaxonomyAlignmentRatioEngine initialized (v%s, NAV=%.2f EUR)",
            _MODULE_VERSION, config.total_nav_eur,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_alignment_ratio(
        self,
        holdings: List[HoldingAlignmentData],
        fund_name: Optional[str] = None,
    ) -> AlignmentResult:
        """Calculate complete Taxonomy alignment ratios for the fund.

        Computes fund-level alignment across revenue, CapEx, and OpEx KPIs.
        Applies double-counting prevention, generates objective breakdowns,
        and creates mandatory pie chart data.

        Args:
            holdings: List of holding alignment data.
            fund_name: Optional fund name for reporting.

        Returns:
            AlignmentResult with all ratios, breakdowns, and visualizations.

        Raises:
            ValueError: If holdings list is empty.
        """
        start = _utcnow()
        self._calculation_count += 1

        if not holdings:
            raise ValueError("Holdings list cannot be empty")

        logger.info(
            "Calculating Taxonomy alignment for %d holdings (NAV=%.2f EUR)",
            len(holdings), self.config.total_nav_eur,
        )

        # Step 1: Compute effective denominator
        denominator = self._compute_effective_denominator(holdings)

        # Step 2: Classify holdings and aggregate by category
        category_values = self._aggregate_by_category(holdings)

        # Step 3: Compute fund-level alignment KPIs
        aligned_kpis = self._compute_fund_level_kpis(holdings, denominator, aligned_only=True)
        eligible_kpis = self._compute_fund_level_kpis(holdings, denominator, aligned_only=False)

        # Step 4: Compute objective breakdown (with double-counting prevention)
        obj_breakdown = self._compute_objective_breakdown(holdings, denominator)

        # Step 5: Gas/nuclear CDA disclosure
        gas_pct, nuclear_pct = self._compute_gas_nuclear(holdings, denominator)

        # Step 6: Commitment adherence
        commitment = self._check_commitment_adherence(aligned_kpis["revenue"])

        # Step 7: Pie chart data
        pie_data = self._generate_pie_chart_data(category_values, denominator)

        elapsed_ms = (_utcnow() - start).total_seconds() * 1000

        result = AlignmentResult(
            fund_name=fund_name,
            reporting_date=self.config.reporting_date,
            total_nav_eur=self.config.total_nav_eur,
            effective_denominator_eur=_round_val(denominator, 2),
            total_holdings=len(holdings),
            aligned_revenue_pct=aligned_kpis["revenue"],
            aligned_capex_pct=aligned_kpis["capex"],
            aligned_opex_pct=aligned_kpis["opex"],
            eligible_revenue_pct=eligible_kpis["revenue"],
            eligible_capex_pct=eligible_kpis["capex"],
            eligible_opex_pct=eligible_kpis["opex"],
            aligned_value_eur=_round_val(category_values["aligned"], 2),
            eligible_not_aligned_value_eur=_round_val(
                category_values["eligible_not_aligned"], 2
            ),
            not_eligible_value_eur=_round_val(category_values["not_eligible"], 2),
            sovereign_value_eur=_round_val(category_values["sovereign"], 2),
            cash_derivatives_value_eur=_round_val(
                category_values["cash_derivatives"], 2
            ),
            not_assessed_value_eur=_round_val(category_values["not_assessed"], 2),
            objective_breakdown=obj_breakdown,
            gas_aligned_pct=gas_pct,
            nuclear_aligned_pct=nuclear_pct,
            commitment_adherence=commitment,
            pie_chart_data=pie_data,
            processing_time_ms=round(elapsed_ms, 2),
        )

        result.provenance_hash = _compute_hash({
            "config": self.config.model_dump(mode="json"),
            "holdings_count": len(holdings),
            "aligned_revenue_pct": result.aligned_revenue_pct,
            "aligned_capex_pct": result.aligned_capex_pct,
            "aligned_opex_pct": result.aligned_opex_pct,
        })

        logger.info(
            "Taxonomy alignment complete: revenue=%.2f%%, capex=%.2f%%, "
            "opex=%.2f%%, time=%.1fms",
            result.aligned_revenue_pct, result.aligned_capex_pct,
            result.aligned_opex_pct, elapsed_ms,
        )

        return result

    def breakdown_by_objective(
        self,
        holdings: List[HoldingAlignmentData],
    ) -> Dict[str, ObjectiveBreakdown]:
        """Compute alignment breakdown by environmental objective.

        Prevents double-counting by assigning each holding to its primary
        objective only. Holdings contributing to multiple objectives are
        counted once under their primary_objective.

        Args:
            holdings: List of holding alignment data.

        Returns:
            Dictionary keyed by objective value with ObjectiveBreakdown.
        """
        denominator = self._compute_effective_denominator(holdings)
        return self._compute_objective_breakdown(holdings, denominator)

    def check_commitment_adherence(
        self,
        holdings: List[HoldingAlignmentData],
    ) -> CommitmentAdherence:
        """Check whether actual alignment meets pre-contractual commitment.

        Compares the fund-level aligned revenue ratio against the minimum
        Taxonomy alignment committed in the fund's pre-contractual documents.

        Args:
            holdings: List of holding alignment data.

        Returns:
            CommitmentAdherence with status and gap analysis.
        """
        denominator = self._compute_effective_denominator(holdings)
        kpis = self._compute_fund_level_kpis(holdings, denominator, aligned_only=True)
        return self._check_commitment_adherence(kpis["revenue"])

    def generate_pie_chart_data(
        self,
        holdings: List[HoldingAlignmentData],
    ) -> List[PieChartSlice]:
        """Generate data for the mandatory SFDR RTS pie chart.

        The pie chart must show:
            1. Taxonomy-aligned investments (green)
            2. Eligible but not aligned (light green)
            3. Non-eligible (grey)
            4. Sovereign bonds (excluded)
            5. Cash / derivatives
            6. Not assessed

        Args:
            holdings: List of holding alignment data.

        Returns:
            List of PieChartSlice for visualization.
        """
        denominator = self._compute_effective_denominator(holdings)
        category_values = self._aggregate_by_category(holdings)
        return self._generate_pie_chart_data(category_values, denominator)

    # ------------------------------------------------------------------
    # Private: Effective Denominator
    # ------------------------------------------------------------------

    def _compute_effective_denominator(
        self,
        holdings: List[HoldingAlignmentData],
    ) -> Decimal:
        """Compute the effective NAV denominator for ratio calculations.

        Excludes sovereign bonds and/or cash/derivatives from the denominator
        based on configuration. Per SFDR RTS, exposures to sovereign issuers
        and certain cash positions may be excluded from the denominator to
        avoid understating the alignment ratio of the investable universe.

        Args:
            holdings: All portfolio holdings.

        Returns:
            Effective denominator as Decimal.
        """
        total = self._nav_decimal

        if not self.config.include_sovereign_in_denominator:
            sovereign_val = sum(
                _decimal(h.value_eur)
                for h in holdings
                if h.holding_type == "SOVEREIGN"
            )
            total -= sovereign_val

        if not self.config.include_derivatives_in_denominator:
            cash_val = sum(
                _decimal(h.value_eur)
                for h in holdings
                if h.holding_type == "CASH_DERIVATIVES"
            )
            total -= cash_val

        # Denominator cannot be negative
        if total < Decimal("0"):
            logger.warning(
                "Effective denominator went negative; clamping to NAV"
            )
            total = self._nav_decimal

        return total

    # ------------------------------------------------------------------
    # Private: Category Aggregation
    # ------------------------------------------------------------------

    def _aggregate_by_category(
        self,
        holdings: List[HoldingAlignmentData],
    ) -> Dict[str, Decimal]:
        """Aggregate holding values by alignment category.

        Args:
            holdings: All portfolio holdings.

        Returns:
            Dict with keys: aligned, eligible_not_aligned, not_eligible,
            sovereign, cash_derivatives, not_assessed.
        """
        categories: Dict[str, Decimal] = {
            "aligned": Decimal("0"),
            "eligible_not_aligned": Decimal("0"),
            "not_eligible": Decimal("0"),
            "sovereign": Decimal("0"),
            "cash_derivatives": Decimal("0"),
            "not_assessed": Decimal("0"),
        }

        category_map = {
            AlignmentCategory.ALIGNED: "aligned",
            AlignmentCategory.ELIGIBLE_NOT_ALIGNED: "eligible_not_aligned",
            AlignmentCategory.NOT_ELIGIBLE: "not_eligible",
            AlignmentCategory.SOVEREIGN: "sovereign",
            AlignmentCategory.CASH_DERIVATIVES: "cash_derivatives",
            AlignmentCategory.NOT_ASSESSED: "not_assessed",
        }

        for h in holdings:
            key = category_map.get(h.alignment_category, "not_assessed")
            categories[key] += _decimal(h.value_eur)

        return categories

    # ------------------------------------------------------------------
    # Private: Fund-Level KPI Computation
    # ------------------------------------------------------------------

    def _compute_fund_level_kpis(
        self,
        holdings: List[HoldingAlignmentData],
        denominator: Decimal,
        aligned_only: bool = True,
    ) -> Dict[str, float]:
        """Compute fund-level alignment KPIs (revenue, CapEx, OpEx).

        For each KPI, the fund-level ratio is:
            fund_ratio = SUM(holding_value * holding_kpi_pct) / denominator

        When aligned_only=True, only ALIGNED holdings contribute.
        When aligned_only=False, both ALIGNED and ELIGIBLE_NOT_ALIGNED contribute.

        Args:
            holdings: All portfolio holdings.
            denominator: Effective NAV denominator.
            aligned_only: If True, only aligned holdings; if False, all eligible.

        Returns:
            Dict with keys: revenue, capex, opex (float percentages).
        """
        revenue_sum = Decimal("0")
        capex_sum = Decimal("0")
        opex_sum = Decimal("0")

        target_categories = {AlignmentCategory.ALIGNED}
        if not aligned_only:
            target_categories.add(AlignmentCategory.ELIGIBLE_NOT_ALIGNED)

        for h in holdings:
            if h.alignment_category not in target_categories:
                continue

            holding_val = _decimal(h.value_eur)

            if aligned_only:
                rev_pct = _decimal(h.aligned_revenue_pct)
                cap_pct = _decimal(h.aligned_capex_pct)
                opx_pct = _decimal(h.aligned_opex_pct)
            else:
                rev_pct = _decimal(h.eligible_revenue_pct)
                cap_pct = _decimal(h.eligible_capex_pct)
                opx_pct = _decimal(h.eligible_opex_pct)

            # Weighted contribution: (holding_value * holding_pct / 100) / denominator * 100
            revenue_sum += holding_val * rev_pct / Decimal("100")
            capex_sum += holding_val * cap_pct / Decimal("100")
            opex_sum += holding_val * opx_pct / Decimal("100")

        return {
            "revenue": _round_val(_pct(revenue_sum, denominator), 4),
            "capex": _round_val(_pct(capex_sum, denominator), 4),
            "opex": _round_val(_pct(opex_sum, denominator), 4),
        }

    # ------------------------------------------------------------------
    # Private: Objective Breakdown
    # ------------------------------------------------------------------

    def _compute_objective_breakdown(
        self,
        holdings: List[HoldingAlignmentData],
        denominator: Decimal,
    ) -> Dict[str, ObjectiveBreakdown]:
        """Compute alignment breakdown by environmental objective.

        Prevents double-counting by using each holding's primary_objective.
        If a holding has no primary_objective, it uses the first in
        contributing_objectives.

        Args:
            holdings: All portfolio holdings.
            denominator: Effective NAV denominator.

        Returns:
            Dict keyed by objective value with ObjectiveBreakdown.
        """
        obj_data: Dict[str, Dict[str, Any]] = {}
        for obj in EnvironmentalObjective:
            obj_data[obj.value] = {
                "value": Decimal("0"),
                "revenue_weighted": Decimal("0"),
                "capex_weighted": Decimal("0"),
                "opex_weighted": Decimal("0"),
                "count": 0,
                "total_value": Decimal("0"),
            }

        aligned_holdings = [
            h for h in holdings
            if h.alignment_category == AlignmentCategory.ALIGNED
        ]

        for h in aligned_holdings:
            primary = self._resolve_primary_objective(h)
            if primary is None:
                continue

            hval = _decimal(h.value_eur)
            obj_data[primary.value]["value"] += hval
            obj_data[primary.value]["revenue_weighted"] += (
                hval * _decimal(h.aligned_revenue_pct) / Decimal("100")
            )
            obj_data[primary.value]["capex_weighted"] += (
                hval * _decimal(h.aligned_capex_pct) / Decimal("100")
            )
            obj_data[primary.value]["opex_weighted"] += (
                hval * _decimal(h.aligned_opex_pct) / Decimal("100")
            )
            obj_data[primary.value]["count"] += 1
            obj_data[primary.value]["total_value"] += hval

        result: Dict[str, ObjectiveBreakdown] = {}
        for obj in EnvironmentalObjective:
            data = obj_data[obj.value]
            total_obj_value = data["total_value"]

            result[obj.value] = ObjectiveBreakdown(
                objective=obj,
                objective_name=OBJECTIVE_NAMES.get(obj.value, obj.value),
                aligned_value_eur=_round_val(data["value"], 2),
                aligned_pct_of_nav=_round_val(
                    _pct(data["value"], denominator), 4
                ),
                aligned_revenue_pct=_round_val(
                    _pct(data["revenue_weighted"], total_obj_value)
                    if total_obj_value > 0 else Decimal("0"), 4
                ),
                aligned_capex_pct=_round_val(
                    _pct(data["capex_weighted"], total_obj_value)
                    if total_obj_value > 0 else Decimal("0"), 4
                ),
                aligned_opex_pct=_round_val(
                    _pct(data["opex_weighted"], total_obj_value)
                    if total_obj_value > 0 else Decimal("0"), 4
                ),
                holding_count=data["count"],
            )

        return result

    def _resolve_primary_objective(
        self,
        holding: HoldingAlignmentData,
    ) -> Optional[EnvironmentalObjective]:
        """Resolve the primary environmental objective for a holding.

        Falls back to the first contributing_objective if primary is not set.

        Args:
            holding: The holding to resolve.

        Returns:
            Primary EnvironmentalObjective or None.
        """
        if holding.primary_objective is not None:
            return holding.primary_objective

        if holding.contributing_objectives:
            return holding.contributing_objectives[0]

        return None

    # ------------------------------------------------------------------
    # Private: Gas/Nuclear CDA Disclosure
    # ------------------------------------------------------------------

    def _compute_gas_nuclear(
        self,
        holdings: List[HoldingAlignmentData],
        denominator: Decimal,
    ) -> Tuple[float, float]:
        """Compute gas and nuclear alignment percentages under CDA.

        Under the Complementary Delegated Act (CDA), Article 8 products must
        separately disclose the proportion of Taxonomy-aligned investments
        related to fossil gas and nuclear energy activities.

        Args:
            holdings: All portfolio holdings.
            denominator: Effective NAV denominator.

        Returns:
            Tuple of (gas_aligned_pct, nuclear_aligned_pct).
        """
        if not self.config.gas_nuclear_disclosure_enabled:
            return 0.0, 0.0

        gas_value = Decimal("0")
        nuclear_value = Decimal("0")

        for h in holdings:
            if h.alignment_category != AlignmentCategory.ALIGNED:
                continue

            hval = _decimal(h.value_eur) * _decimal(h.aligned_revenue_pct) / Decimal("100")

            if h.gas_nuclear_exposure == GASExposureType.FOSSIL_GAS:
                gas_value += hval
            elif h.gas_nuclear_exposure == GASExposureType.NUCLEAR:
                nuclear_value += hval

        gas_pct = _round_val(_pct(gas_value, denominator), 4)
        nuclear_pct = _round_val(_pct(nuclear_value, denominator), 4)

        return gas_pct, nuclear_pct

    # ------------------------------------------------------------------
    # Private: Commitment Adherence
    # ------------------------------------------------------------------

    def _check_commitment_adherence(
        self,
        actual_alignment_pct: float,
    ) -> CommitmentAdherence:
        """Check whether actual alignment meets pre-contractual commitment.

        Args:
            actual_alignment_pct: Computed fund-level alignment (revenue-based).

        Returns:
            CommitmentAdherence with status.
        """
        commitment = self.config.pre_contractual_commitment_pct

        if commitment is None:
            return CommitmentAdherence(
                actual_alignment_pct=actual_alignment_pct,
                status="not_applicable",
            )

        gap = actual_alignment_pct - commitment
        meets = actual_alignment_pct >= commitment

        return CommitmentAdherence(
            pre_contractual_commitment_pct=commitment,
            actual_alignment_pct=actual_alignment_pct,
            meets_commitment=meets,
            gap_pct=round(gap, 4),
            status="COMPLIANT" if meets else "SHORTFALL",
        )

    # ------------------------------------------------------------------
    # Private: Pie Chart Data Generation
    # ------------------------------------------------------------------

    def _generate_pie_chart_data(
        self,
        category_values: Dict[str, Decimal],
        denominator: Decimal,
    ) -> List[PieChartSlice]:
        """Generate pie chart slices for the mandatory SFDR RTS visualization.

        The RTS requires a pie chart showing the proportions of:
            1. Taxonomy-aligned investments
            2. Eligible but not aligned
            3. Not Taxonomy-eligible
            4. Sovereign bonds (if applicable)
            5. Cash and derivatives (if applicable)
            6. Not assessed

        Args:
            category_values: Aggregated values by alignment category.
            denominator: Effective denominator (total NAV for pie chart).

        Returns:
            List of PieChartSlice objects.
        """
        # Use total NAV for pie chart (not effective denominator)
        total = self._nav_decimal
        if total == Decimal("0"):
            return []

        slices_config = [
            ("Taxonomy-aligned", "aligned", "#1B5E20"),
            ("Eligible, not aligned", "eligible_not_aligned", "#66BB6A"),
            ("Not Taxonomy-eligible", "not_eligible", "#9E9E9E"),
            ("Sovereign bonds", "sovereign", "#42A5F5"),
            ("Cash and derivatives", "cash_derivatives", "#BDBDBD"),
            ("Not assessed", "not_assessed", "#FFA726"),
        ]

        slices: List[PieChartSlice] = []
        for label, key, color in slices_config:
            value = category_values.get(key, Decimal("0"))
            pct = _round_val(_pct(value, total), 2)
            if pct > 0 or key in ("aligned", "not_eligible"):
                slices.append(PieChartSlice(
                    label=label,
                    value_pct=pct,
                    value_eur=_round_val(value, 2),
                    color_hint=color,
                ))

        return slices

    # ------------------------------------------------------------------
    # Read-only Properties
    # ------------------------------------------------------------------

    @property
    def calculation_count(self) -> int:
        """Number of alignment calculations performed since initialization."""
        return self._calculation_count

    @property
    def supported_objectives(self) -> List[str]:
        """List of all supported environmental objectives."""
        return [o.value for o in EnvironmentalObjective]

    @property
    def alignment_categories(self) -> List[str]:
        """List of all alignment category values."""
        return [c.value for c in AlignmentCategory]
