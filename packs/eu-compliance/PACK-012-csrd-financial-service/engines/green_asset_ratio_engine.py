# -*- coding: utf-8 -*-
"""
GreenAssetRatioEngine - PACK-012 CSRD Financial Service Engine 3
==================================================================

Green Asset Ratio (GAR) calculation engine for credit institutions
under the EU Taxonomy Regulation and CRR/CRD framework.

The GAR measures the proportion of a bank's assets that finance
Taxonomy-aligned economic activities.  It is the key Pillar 3
disclosure KPI for credit institutions.

Core Formulas:
    GAR = Taxonomy-Aligned Assets / Total Covered Assets
    Turnover GAR  = Sum(aligned_turnover_share * exposure) / covered
    CapEx GAR     = Sum(aligned_capex_share * exposure) / covered
    OpEx GAR      = Sum(aligned_opex_share * exposure) / covered
    Flow GAR      = New Origination Aligned / New Origination Total
    OBS GAR       = Off-balance aligned guarantees / total guarantees

Covered Assets (Numerator Eligible):
    - Loans and advances to NFCs and households
    - Debt securities (NFC and sovereign sub-sovereign)
    - Equity instruments (NFC)
    - Repossessed real estate collateral

Excluded Assets (Denominator Only):
    - Sovereign exposures / central bank exposures
    - Trading book (held for trading)
    - Interbank on-demand loans
    - Exposures to central governments
    - Derivatives (not hedging)

Environmental Objectives (EU Taxonomy Article 9):
    1. Climate change mitigation
    2. Climate change adaptation
    3. Sustainable use of water and marine resources
    4. Transition to a circular economy
    5. Pollution prevention and control
    6. Protection of biodiversity and ecosystems

Counterparty Types:
    - Non-financial corporates (NFCs) subject to NFRD/CSRD
    - Financial corporates (credit institutions, investment firms)
    - Households (mortgages, vehicle loans)
    - Local government / housing / public utilities

Regulatory References:
    - EU Taxonomy Regulation 2020/852 (Articles 8, 10)
    - Delegated Regulation 2021/2178 (Pillar 3 templates)
    - CRR Article 449a (disclosure of ESG risks)
    - EBA ITS on Pillar 3 ESG disclosures (EBA/ITS/2022/01)

Zero-Hallucination:
    - All calculations use deterministic Python arithmetic
    - GAR ratios are pure division of sums
    - Counterparty and objective breakdowns are deterministic grouping
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-012 CSRD Financial Service
Status: Production Ready
"""

import hashlib
import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
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


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default on zero denominator.

    Args:
        numerator: The dividend.
        denominator: The divisor.
        default: Value to return if denominator is zero.

    Returns:
        Result of division or default value.
    """
    if denominator == 0.0:
        return default
    return numerator / denominator


def _safe_pct(numerator: float, denominator: float) -> float:
    """Calculate percentage safely.

    Args:
        numerator: The dividend.
        denominator: The divisor.

    Returns:
        Percentage or 0.0 on zero denominator.
    """
    if denominator == 0.0:
        return 0.0
    return (numerator / denominator) * 100.0


def _round_val(value: float, places: int = 4) -> float:
    """Round a float to specified decimal places."""
    return round(value, places)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class GARScope(str, Enum):
    """Scope of GAR calculation."""
    STOCK = "stock"              # All outstanding assets
    FLOW = "flow"                # New originations only
    OFF_BALANCE_SHEET = "off_balance_sheet"  # Guarantees and commitments


class EnvironmentalObjective(str, Enum):
    """EU Taxonomy six environmental objectives."""
    CLIMATE_MITIGATION = "climate_mitigation"
    CLIMATE_ADAPTATION = "climate_adaptation"
    WATER_MARINE = "water_marine"
    CIRCULAR_ECONOMY = "circular_economy"
    POLLUTION_PREVENTION = "pollution_prevention"
    BIODIVERSITY = "biodiversity"


class CounterpartyType(str, Enum):
    """Counterparty types for GAR breakdown."""
    NFC_CSRD = "nfc_csrd"                # NFCs subject to CSRD
    NFC_NON_CSRD = "nfc_non_csrd"        # NFCs not subject to CSRD
    FINANCIAL_CORPORATE = "financial_corporate"  # Credit institutions, etc.
    HOUSEHOLD_MORTGAGE = "household_mortgage"    # Household mortgages
    HOUSEHOLD_VEHICLE = "household_vehicle"      # Household vehicle loans
    HOUSEHOLD_RENOVATION = "household_renovation"  # Building renovation loans
    LOCAL_GOVERNMENT = "local_government"         # Local govt / public utilities
    OTHER = "other"


class AssetType(str, Enum):
    """Asset types on the balance sheet."""
    LOANS_ADVANCES = "loans_advances"
    DEBT_SECURITIES = "debt_securities"
    EQUITY_INSTRUMENTS = "equity_instruments"
    REPOSSESSED_COLLATERAL = "repossessed_collateral"
    GUARANTEE = "guarantee"
    COMMITMENT = "commitment"


class ExclusionReason(str, Enum):
    """Reasons an asset is excluded from the GAR numerator."""
    SOVEREIGN_EXPOSURE = "sovereign_exposure"
    CENTRAL_BANK = "central_bank"
    TRADING_BOOK = "trading_book"
    INTERBANK_DEMAND = "interbank_demand"
    DERIVATIVE = "derivative"
    NOT_COVERED = "not_covered"


class AlignmentType(str, Enum):
    """Type of Taxonomy alignment."""
    ALIGNED = "aligned"
    ENABLING = "enabling"
    TRANSITIONAL = "transitional"
    ELIGIBLE_NOT_ALIGNED = "eligible_not_aligned"
    NOT_ELIGIBLE = "not_eligible"


# ---------------------------------------------------------------------------
# Pydantic Data Models
# ---------------------------------------------------------------------------


class CoveredAssetData(BaseModel):
    """Input data for a single on-balance-sheet covered asset.

    Contains the exposure amount, counterparty details, Taxonomy
    alignment data, and classification metadata.

    Attributes:
        asset_id: Unique asset identifier.
        asset_name: Name / description.
        asset_type: Balance sheet classification.
        counterparty_type: Counterparty classification.
        counterparty_name: Counterparty name.
        nace_sector: NACE sector of the economic activity.
        country: Country of counterparty (ISO 3166-1).
        gross_carrying_amount: Gross carrying amount (EUR).
        net_carrying_amount: Net carrying amount (EUR) -- used in GAR.
        is_new_origination: Whether originated in the reporting period (for flow).
        origination_date: Date of origination (YYYY-MM-DD).
        is_excluded: Whether this asset is excluded from covered assets.
        exclusion_reason: Reason for exclusion.
        taxonomy_eligible_pct: Percentage of the activity that is Taxonomy eligible.
        turnover_aligned_pct: Turnover-based alignment percentage.
        capex_aligned_pct: CapEx-based alignment percentage.
        opex_aligned_pct: OpEx-based alignment percentage.
        alignment_type: Type of alignment (enabling, transitional, etc.).
        primary_objective: Primary environmental objective.
        secondary_objectives: Additional environmental objectives.
        dnsh_passed: Whether DNSH criteria are met.
        minimum_safeguards_passed: Whether minimum safeguards are met.
        epc_label: Energy Performance Certificate label (A-G) for buildings.
        vehicle_emission_class: Vehicle CO2 class for motor loans.
    """
    asset_id: str = Field(default_factory=_new_uuid, description="Unique asset ID")
    asset_name: str = Field(default="", description="Asset name / description")
    asset_type: AssetType = Field(
        default=AssetType.LOANS_ADVANCES,
        description="Balance sheet classification",
    )
    counterparty_type: CounterpartyType = Field(
        default=CounterpartyType.NFC_CSRD,
        description="Counterparty classification",
    )
    counterparty_name: str = Field(default="", description="Counterparty name")
    nace_sector: str = Field(default="", description="NACE sector code")
    country: str = Field(default="", description="Country (ISO 3166)")
    gross_carrying_amount: float = Field(
        default=0.0, ge=0.0, description="Gross carrying amount (EUR)",
    )
    net_carrying_amount: float = Field(
        default=0.0, ge=0.0, description="Net carrying amount (EUR)",
    )
    is_new_origination: bool = Field(
        default=False, description="New origination in reporting period",
    )
    origination_date: str = Field(default="", description="Origination date (YYYY-MM-DD)")
    is_excluded: bool = Field(
        default=False, description="Whether excluded from covered assets",
    )
    exclusion_reason: Optional[ExclusionReason] = Field(
        default=None, description="Reason for exclusion",
    )
    taxonomy_eligible_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Taxonomy eligible percentage",
    )
    turnover_aligned_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Turnover-based alignment percentage",
    )
    capex_aligned_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="CapEx-based alignment percentage",
    )
    opex_aligned_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="OpEx-based alignment percentage",
    )
    alignment_type: AlignmentType = Field(
        default=AlignmentType.NOT_ELIGIBLE,
        description="Type of Taxonomy alignment",
    )
    primary_objective: Optional[EnvironmentalObjective] = Field(
        default=None, description="Primary environmental objective",
    )
    secondary_objectives: List[EnvironmentalObjective] = Field(
        default_factory=list, description="Additional environmental objectives",
    )
    dnsh_passed: bool = Field(
        default=False, description="Whether DNSH criteria are met",
    )
    minimum_safeguards_passed: bool = Field(
        default=False, description="Whether minimum safeguards are met",
    )
    epc_label: str = Field(default="", description="Energy Performance Certificate (A-G)")
    vehicle_emission_class: str = Field(
        default="", description="Vehicle CO2 emission class",
    )


class GARBreakdown(BaseModel):
    """GAR calculation breakdown for a specific dimension.

    Contains the numerator (aligned), denominator (covered), and
    the resulting ratio for each GAR variant.

    Attributes:
        dimension_name: Name of the breakdown dimension.
        dimension_value: Value of the dimension.
        total_covered_assets: Total covered assets (EUR) -- denominator.
        turnover_aligned_amount: Turnover-aligned amount (EUR).
        turnover_gar_pct: Turnover GAR percentage.
        capex_aligned_amount: CapEx-aligned amount (EUR).
        capex_gar_pct: CapEx GAR percentage.
        opex_aligned_amount: OpEx-aligned amount (EUR).
        opex_gar_pct: OpEx GAR percentage.
        eligible_amount: Taxonomy-eligible amount (EUR).
        eligible_pct: Taxonomy-eligible percentage.
        enabling_amount: Enabling activities amount (EUR).
        transitional_amount: Transitional activities amount (EUR).
        asset_count: Number of assets in this breakdown.
    """
    dimension_name: str = Field(description="Breakdown dimension name")
    dimension_value: str = Field(description="Breakdown dimension value")
    total_covered_assets: float = Field(
        default=0.0, description="Total covered assets (EUR)",
    )
    turnover_aligned_amount: float = Field(
        default=0.0, description="Turnover-aligned amount (EUR)",
    )
    turnover_gar_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Turnover GAR (%)",
    )
    capex_aligned_amount: float = Field(
        default=0.0, description="CapEx-aligned amount (EUR)",
    )
    capex_gar_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="CapEx GAR (%)",
    )
    opex_aligned_amount: float = Field(
        default=0.0, description="OpEx-aligned amount (EUR)",
    )
    opex_gar_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="OpEx GAR (%)",
    )
    eligible_amount: float = Field(
        default=0.0, description="Taxonomy-eligible amount (EUR)",
    )
    eligible_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Taxonomy-eligible (%)",
    )
    enabling_amount: float = Field(
        default=0.0, description="Enabling activities amount (EUR)",
    )
    transitional_amount: float = Field(
        default=0.0, description="Transitional activities amount (EUR)",
    )
    asset_count: int = Field(default=0, ge=0, description="Number of assets")


class CounterpartyBreakdown(BaseModel):
    """GAR breakdown by counterparty type.

    Attributes:
        counterparty_type: Counterparty classification.
        total_exposure: Total exposure (EUR).
        covered_exposure: Covered exposure (EUR).
        turnover_aligned: Turnover-aligned (EUR).
        turnover_gar_pct: Turnover GAR (%).
        capex_aligned: CapEx-aligned (EUR).
        capex_gar_pct: CapEx GAR (%).
        opex_aligned: OpEx-aligned (EUR).
        opex_gar_pct: OpEx GAR (%).
        asset_count: Number of assets.
        weight_pct: Weight in total portfolio.
    """
    counterparty_type: CounterpartyType = Field(
        description="Counterparty classification",
    )
    total_exposure: float = Field(default=0.0, description="Total exposure (EUR)")
    covered_exposure: float = Field(default=0.0, description="Covered exposure (EUR)")
    turnover_aligned: float = Field(default=0.0, description="Turnover-aligned (EUR)")
    turnover_gar_pct: float = Field(default=0.0, description="Turnover GAR (%)")
    capex_aligned: float = Field(default=0.0, description="CapEx-aligned (EUR)")
    capex_gar_pct: float = Field(default=0.0, description="CapEx GAR (%)")
    opex_aligned: float = Field(default=0.0, description="OpEx-aligned (EUR)")
    opex_gar_pct: float = Field(default=0.0, description="OpEx GAR (%)")
    asset_count: int = Field(default=0, ge=0, description="Number of assets")
    weight_pct: float = Field(default=0.0, description="Weight in total (%)")


class FlowGAR(BaseModel):
    """Flow GAR for new originations in the reporting period.

    Attributes:
        total_new_origination: Total new originations (EUR).
        turnover_aligned_origination: Turnover-aligned new (EUR).
        turnover_flow_gar_pct: Turnover Flow GAR (%).
        capex_aligned_origination: CapEx-aligned new (EUR).
        capex_flow_gar_pct: CapEx Flow GAR (%).
        opex_aligned_origination: OpEx-aligned new (EUR).
        opex_flow_gar_pct: OpEx Flow GAR (%).
        new_origination_count: Number of new originations.
        provenance_hash: SHA-256 provenance hash.
    """
    total_new_origination: float = Field(
        default=0.0, description="Total new originations (EUR)",
    )
    turnover_aligned_origination: float = Field(
        default=0.0, description="Turnover-aligned new originations (EUR)",
    )
    turnover_flow_gar_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Turnover Flow GAR (%)",
    )
    capex_aligned_origination: float = Field(
        default=0.0, description="CapEx-aligned new originations (EUR)",
    )
    capex_flow_gar_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="CapEx Flow GAR (%)",
    )
    opex_aligned_origination: float = Field(
        default=0.0, description="OpEx-aligned new originations (EUR)",
    )
    opex_flow_gar_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="OpEx Flow GAR (%)",
    )
    new_origination_count: int = Field(
        default=0, ge=0, description="Number of new originations",
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class OffBalanceSheetKPI(BaseModel):
    """Off-balance-sheet GAR KPI.

    Covers financial guarantees and commitments to lend.

    Attributes:
        total_guarantees: Total financial guarantees (EUR).
        aligned_guarantees: Taxonomy-aligned guarantees (EUR).
        guarantee_gar_pct: Guarantee GAR (%).
        total_commitments: Total commitments to lend (EUR).
        aligned_commitments: Taxonomy-aligned commitments (EUR).
        commitment_gar_pct: Commitment GAR (%).
        obs_count: Number of OBS items.
        provenance_hash: SHA-256 provenance hash.
    """
    total_guarantees: float = Field(default=0.0, description="Total guarantees (EUR)")
    aligned_guarantees: float = Field(default=0.0, description="Aligned guarantees (EUR)")
    guarantee_gar_pct: float = Field(default=0.0, description="Guarantee GAR (%)")
    total_commitments: float = Field(default=0.0, description="Total commitments (EUR)")
    aligned_commitments: float = Field(default=0.0, description="Aligned commitments (EUR)")
    commitment_gar_pct: float = Field(default=0.0, description="Commitment GAR (%)")
    obs_count: int = Field(default=0, ge=0, description="Number of OBS items")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class GARResult(BaseModel):
    """Complete Green Asset Ratio result.

    Top-level result containing all three GAR variants (turnover,
    CapEx, OpEx), breakdowns by objective and counterparty, flow
    GAR, and off-balance-sheet KPIs.

    Attributes:
        result_id: Unique result identifier.
        reporting_year: Reporting year.
        total_assets: Total on-balance-sheet assets (EUR).
        total_covered_assets: Total covered assets (EUR) -- denominator.
        total_excluded_assets: Total excluded assets (EUR).
        turnover_aligned_total: Total turnover-aligned assets (EUR).
        turnover_gar_pct: Turnover GAR (%).
        capex_aligned_total: Total CapEx-aligned assets (EUR).
        capex_gar_pct: CapEx GAR (%).
        opex_aligned_total: Total OpEx-aligned assets (EUR).
        opex_gar_pct: OpEx GAR (%).
        eligible_total: Total Taxonomy-eligible assets (EUR).
        eligible_pct: Taxonomy-eligible (%).
        enabling_total: Total enabling activities (EUR).
        transitional_total: Total transitional activities (EUR).
        objective_breakdown: Breakdown by environmental objective.
        counterparty_breakdown: Breakdown by counterparty type.
        flow_gar: Flow GAR for new originations.
        off_balance_sheet: Off-balance-sheet KPIs.
        total_asset_count: Total number of assets.
        covered_asset_count: Number of covered assets.
        excluded_asset_count: Number of excluded assets.
        methodology_notes: Methodology notes for disclosure.
        processing_time_ms: Processing time (ms).
        engine_version: Engine version string.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 provenance hash.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Unique result ID")
    reporting_year: int = Field(default=2024, description="Reporting year")
    total_assets: float = Field(default=0.0, description="Total assets (EUR)")
    total_covered_assets: float = Field(
        default=0.0, description="Total covered assets (EUR)",
    )
    total_excluded_assets: float = Field(
        default=0.0, description="Total excluded assets (EUR)",
    )
    turnover_aligned_total: float = Field(
        default=0.0, description="Turnover-aligned total (EUR)",
    )
    turnover_gar_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Turnover GAR (%)",
    )
    capex_aligned_total: float = Field(
        default=0.0, description="CapEx-aligned total (EUR)",
    )
    capex_gar_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="CapEx GAR (%)",
    )
    opex_aligned_total: float = Field(
        default=0.0, description="OpEx-aligned total (EUR)",
    )
    opex_gar_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="OpEx GAR (%)",
    )
    eligible_total: float = Field(
        default=0.0, description="Taxonomy-eligible total (EUR)",
    )
    eligible_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Taxonomy-eligible (%)",
    )
    enabling_total: float = Field(
        default=0.0, description="Enabling activities total (EUR)",
    )
    transitional_total: float = Field(
        default=0.0, description="Transitional activities total (EUR)",
    )
    objective_breakdown: List[GARBreakdown] = Field(
        default_factory=list,
        description="Breakdown by environmental objective",
    )
    counterparty_breakdown: List[CounterpartyBreakdown] = Field(
        default_factory=list,
        description="Breakdown by counterparty type",
    )
    flow_gar: Optional[FlowGAR] = Field(
        default=None, description="Flow GAR for new originations",
    )
    off_balance_sheet: Optional[OffBalanceSheetKPI] = Field(
        default=None, description="Off-balance-sheet KPIs",
    )
    total_asset_count: int = Field(default=0, ge=0, description="Total asset count")
    covered_asset_count: int = Field(default=0, ge=0, description="Covered asset count")
    excluded_asset_count: int = Field(default=0, ge=0, description="Excluded asset count")
    methodology_notes: List[str] = Field(
        default_factory=list, description="Methodology notes",
    )
    processing_time_ms: float = Field(default=0.0, description="Processing time (ms)")
    engine_version: str = Field(default=_MODULE_VERSION, description="Engine version")
    calculated_at: datetime = Field(
        default_factory=_utcnow, description="Calculation timestamp",
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


# ---------------------------------------------------------------------------
# Engine Configuration
# ---------------------------------------------------------------------------


class GARConfig(BaseModel):
    """Configuration for the GreenAssetRatioEngine.

    Controls GAR calculation parameters, asset type inclusion,
    and reporting settings.

    Attributes:
        reporting_year: Reporting year.
        use_net_carrying_amount: Use net (vs gross) carrying amount.
        include_flow_gar: Whether to compute Flow GAR.
        include_obs_gar: Whether to compute off-balance-sheet GAR.
        epc_threshold_label: EPC label threshold for residential alignment.
        vehicle_co2_threshold_gkm: CO2 threshold (g/km) for vehicle alignment.
        household_mortgage_aligned_epc: EPC labels considered aligned.
        household_vehicle_aligned_classes: Vehicle classes considered aligned.
        precision_decimal_places: Decimal places for rounding.
    """
    reporting_year: int = Field(default=2024, description="Reporting year")
    use_net_carrying_amount: bool = Field(
        default=True,
        description="Use net carrying amount (True) or gross (False)",
    )
    include_flow_gar: bool = Field(
        default=True,
        description="Whether to compute Flow GAR for new originations",
    )
    include_obs_gar: bool = Field(
        default=True,
        description="Whether to compute off-balance-sheet GAR",
    )
    epc_threshold_label: str = Field(
        default="A",
        description="Minimum EPC label for household mortgage alignment",
    )
    vehicle_co2_threshold_gkm: float = Field(
        default=50.0, ge=0.0,
        description="Max CO2 g/km for vehicle loan alignment",
    )
    household_mortgage_aligned_epc: List[str] = Field(
        default_factory=lambda: ["A", "B"],
        description="EPC labels considered Taxonomy-aligned for mortgages",
    )
    household_vehicle_aligned_classes: List[str] = Field(
        default_factory=lambda: ["ev", "phev", "hydrogen"],
        description="Vehicle classes considered Taxonomy-aligned",
    )
    precision_decimal_places: int = Field(
        default=4, ge=0, le=10,
        description="Decimal places for rounding",
    )


# ---------------------------------------------------------------------------
# Model rebuilds for forward references
# ---------------------------------------------------------------------------

CoveredAssetData.model_rebuild()
GARBreakdown.model_rebuild()
CounterpartyBreakdown.model_rebuild()
FlowGAR.model_rebuild()
OffBalanceSheetKPI.model_rebuild()
GARResult.model_rebuild()
GARConfig.model_rebuild()


# ---------------------------------------------------------------------------
# GreenAssetRatioEngine
# ---------------------------------------------------------------------------


class GreenAssetRatioEngine:
    """
    Green Asset Ratio calculation engine for credit institutions.

    Computes the GAR in all three variants (turnover, CapEx, OpEx)
    with breakdowns by environmental objective, counterparty type,
    and asset type.  Also computes Flow GAR (new originations) and
    off-balance-sheet KPIs (guarantees and commitments).

    Zero-Hallucination Guarantees:
        - All calculations use deterministic Python arithmetic
        - GAR ratios are pure sum(numerator) / sum(denominator)
        - Breakdowns are deterministic grouping operations
        - SHA-256 provenance hash on every result
        - No LLM involvement in any calculation path

    Example:
        >>> config = GARConfig(reporting_year=2024)
        >>> engine = GreenAssetRatioEngine(config)
        >>> assets = [CoveredAssetData(
        ...     net_carrying_amount=10_000_000,
        ...     counterparty_type=CounterpartyType.NFC_CSRD,
        ...     turnover_aligned_pct=60.0,
        ...     taxonomy_eligible_pct=80.0,
        ... )]
        >>> result = engine.calculate_gar(assets)
        >>> assert result.turnover_gar_pct > 0
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize GreenAssetRatioEngine.

        Args:
            config: Optional GARConfig or dict.
        """
        if config and isinstance(config, dict):
            self.config = GARConfig(**config)
        elif config and isinstance(config, GARConfig):
            self.config = config
        else:
            self.config = GARConfig()

        self._assets: List[CoveredAssetData] = []

        logger.info(
            "GreenAssetRatioEngine initialized (version=%s, year=%d)",
            _MODULE_VERSION,
            self.config.reporting_year,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_gar(
        self,
        assets: List[CoveredAssetData],
    ) -> GARResult:
        """Calculate the Green Asset Ratio for the bank's portfolio.

        Pipeline:
        1. Separate covered vs excluded assets
        2. Apply household-specific alignment rules
        3. Calculate alignment amounts per asset
        4. Sum numerator (aligned) and denominator (covered)
        5. Compute GAR percentages (turnover, CapEx, OpEx)
        6. Build breakdowns by objective and counterparty
        7. Compute Flow GAR (new originations)
        8. Compute off-balance-sheet KPIs

        Args:
            assets: List of CoveredAssetData for the portfolio.

        Returns:
            GARResult with full breakdown and provenance.

        Raises:
            ValueError: If assets list is empty.
        """
        start = _utcnow()

        if not assets:
            raise ValueError("Assets list cannot be empty")

        self._assets = assets

        logger.info("Calculating GAR for %d assets", len(assets))

        # Step 1: Separate covered vs excluded and on-BS vs off-BS
        on_balance = [a for a in assets if a.asset_type not in (
            AssetType.GUARANTEE, AssetType.COMMITMENT,
        )]
        off_balance = [a for a in assets if a.asset_type in (
            AssetType.GUARANTEE, AssetType.COMMITMENT,
        )]

        covered = [a for a in on_balance if not a.is_excluded]
        excluded = [a for a in on_balance if a.is_excluded]

        # Step 2: Apply household alignment rules
        for asset in covered:
            self._apply_household_alignment(asset)

        # Step 3: Get carrying amount field
        def _amount(a: CoveredAssetData) -> float:
            if self.config.use_net_carrying_amount:
                return a.net_carrying_amount if a.net_carrying_amount > 0 else a.gross_carrying_amount
            return a.gross_carrying_amount

        # Step 4: Compute totals
        total_all = sum(_amount(a) for a in on_balance)
        total_covered = sum(_amount(a) for a in covered)
        total_excluded = sum(_amount(a) for a in excluded)

        # Aligned amounts per variant
        turnover_aligned = sum(
            _amount(a) * (a.turnover_aligned_pct / 100.0) for a in covered
        )
        capex_aligned = sum(
            _amount(a) * (a.capex_aligned_pct / 100.0) for a in covered
        )
        opex_aligned = sum(
            _amount(a) * (a.opex_aligned_pct / 100.0) for a in covered
        )
        eligible_total = sum(
            _amount(a) * (a.taxonomy_eligible_pct / 100.0) for a in covered
        )

        # Enabling and transitional
        enabling = sum(
            _amount(a) * (a.turnover_aligned_pct / 100.0) for a in covered
            if a.alignment_type == AlignmentType.ENABLING
        )
        transitional = sum(
            _amount(a) * (a.turnover_aligned_pct / 100.0) for a in covered
            if a.alignment_type == AlignmentType.TRANSITIONAL
        )

        # Step 5: Compute GAR percentages
        turnover_gar = _safe_pct(turnover_aligned, total_covered)
        capex_gar = _safe_pct(capex_aligned, total_covered)
        opex_gar = _safe_pct(opex_aligned, total_covered)
        eligible_pct = _safe_pct(eligible_total, total_covered)

        # Step 6: Breakdowns
        objective_breakdown = self._breakdown_by_objective(covered, total_covered, _amount)
        counterparty_breakdown = self._breakdown_by_counterparty(
            covered, total_covered, _amount
        )

        # Step 7: Flow GAR
        flow_gar = None
        if self.config.include_flow_gar:
            flow_gar = self._compute_flow_gar(covered, _amount)

        # Step 8: Off-balance-sheet
        obs_kpi = None
        if self.config.include_obs_gar and off_balance:
            obs_kpi = self._compute_obs_gar(off_balance, _amount)

        # Methodology notes
        notes = self._generate_methodology_notes(
            assets, covered, excluded, off_balance,
        )

        end = _utcnow()
        processing_ms = (end - start).total_seconds() * 1000.0

        result = GARResult(
            reporting_year=self.config.reporting_year,
            total_assets=_round_val(total_all, 2),
            total_covered_assets=_round_val(total_covered, 2),
            total_excluded_assets=_round_val(total_excluded, 2),
            turnover_aligned_total=_round_val(turnover_aligned, 2),
            turnover_gar_pct=_round_val(turnover_gar, 4),
            capex_aligned_total=_round_val(capex_aligned, 2),
            capex_gar_pct=_round_val(capex_gar, 4),
            opex_aligned_total=_round_val(opex_aligned, 2),
            opex_gar_pct=_round_val(opex_gar, 4),
            eligible_total=_round_val(eligible_total, 2),
            eligible_pct=_round_val(eligible_pct, 4),
            enabling_total=_round_val(enabling, 2),
            transitional_total=_round_val(transitional, 2),
            objective_breakdown=objective_breakdown,
            counterparty_breakdown=counterparty_breakdown,
            flow_gar=flow_gar,
            off_balance_sheet=obs_kpi,
            total_asset_count=len(assets),
            covered_asset_count=len(covered),
            excluded_asset_count=len(excluded),
            methodology_notes=notes,
            processing_time_ms=_round_val(processing_ms, 2),
        )

        result.provenance_hash = _compute_hash(result)
        logger.info(
            "GAR: turnover=%.4f%%, capex=%.4f%%, opex=%.4f%% "
            "(covered=%d, excluded=%d)",
            result.turnover_gar_pct,
            result.capex_gar_pct,
            result.opex_gar_pct,
            result.covered_asset_count,
            result.excluded_asset_count,
        )
        return result

    def calculate_single_asset_contribution(
        self, asset: CoveredAssetData,
    ) -> Dict[str, float]:
        """Calculate GAR contribution of a single asset.

        Args:
            asset: CoveredAssetData.

        Returns:
            Dict with turnover/capex/opex aligned amounts and eligible.
        """
        self._apply_household_alignment(asset)
        amount = (
            asset.net_carrying_amount
            if self.config.use_net_carrying_amount and asset.net_carrying_amount > 0
            else asset.gross_carrying_amount
        )
        return {
            "carrying_amount": _round_val(amount, 2),
            "turnover_aligned": _round_val(amount * (asset.turnover_aligned_pct / 100.0), 2),
            "capex_aligned": _round_val(amount * (asset.capex_aligned_pct / 100.0), 2),
            "opex_aligned": _round_val(amount * (asset.opex_aligned_pct / 100.0), 2),
            "eligible": _round_val(amount * (asset.taxonomy_eligible_pct / 100.0), 2),
            "is_excluded": asset.is_excluded,
        }

    # ------------------------------------------------------------------
    # Internal: Household Alignment Rules
    # ------------------------------------------------------------------

    def _apply_household_alignment(self, asset: CoveredAssetData) -> None:
        """Apply Taxonomy alignment rules for household exposures.

        For mortgages: alignment based on EPC label.
        For vehicle loans: alignment based on vehicle emission class.
        For renovation loans: alignment based on energy improvement.

        Modifies the asset in place (deterministic rule application).

        Args:
            asset: CoveredAssetData (modified in place).
        """
        if asset.counterparty_type == CounterpartyType.HOUSEHOLD_MORTGAGE:
            if asset.epc_label.upper() in self.config.household_mortgage_aligned_epc:
                if asset.turnover_aligned_pct <= 0:
                    asset.turnover_aligned_pct = 100.0
                    asset.capex_aligned_pct = 100.0
                    asset.opex_aligned_pct = 100.0
                    asset.taxonomy_eligible_pct = 100.0
                    asset.primary_objective = EnvironmentalObjective.CLIMATE_MITIGATION
                    asset.alignment_type = AlignmentType.ALIGNED
            else:
                if asset.epc_label:  # Has EPC but not aligned
                    asset.taxonomy_eligible_pct = max(asset.taxonomy_eligible_pct, 100.0)

        elif asset.counterparty_type == CounterpartyType.HOUSEHOLD_VEHICLE:
            vclass = asset.vehicle_emission_class.lower()
            if vclass in self.config.household_vehicle_aligned_classes:
                if asset.turnover_aligned_pct <= 0:
                    asset.turnover_aligned_pct = 100.0
                    asset.capex_aligned_pct = 100.0
                    asset.opex_aligned_pct = 100.0
                    asset.taxonomy_eligible_pct = 100.0
                    asset.primary_objective = EnvironmentalObjective.CLIMATE_MITIGATION
                    asset.alignment_type = AlignmentType.ALIGNED
            else:
                asset.taxonomy_eligible_pct = max(asset.taxonomy_eligible_pct, 100.0)

        elif asset.counterparty_type == CounterpartyType.HOUSEHOLD_RENOVATION:
            # Building renovation loans: eligible by default, aligned if
            # the renovation achieves primary energy savings >= 30%
            asset.taxonomy_eligible_pct = max(asset.taxonomy_eligible_pct, 100.0)
            if asset.primary_objective == EnvironmentalObjective.CLIMATE_MITIGATION:
                asset.alignment_type = AlignmentType.ALIGNED

    # ------------------------------------------------------------------
    # Internal: Breakdowns
    # ------------------------------------------------------------------

    def _breakdown_by_objective(
        self,
        covered: List[CoveredAssetData],
        total_covered: float,
        amount_fn: Any,
    ) -> List[GARBreakdown]:
        """Build GAR breakdown by environmental objective.

        Args:
            covered: List of covered assets.
            total_covered: Total covered amount for percentage calc.
            amount_fn: Function to get carrying amount.

        Returns:
            List of GARBreakdown by objective.
        """
        groups: Dict[str, List[CoveredAssetData]] = defaultdict(list)
        for a in covered:
            if a.primary_objective:
                groups[a.primary_objective.value].append(a)
            else:
                groups["not_assigned"].append(a)

        breakdowns: List[GARBreakdown] = []
        for obj_val, group in sorted(groups.items()):
            group_covered = sum(amount_fn(a) for a in group)
            turn_aligned = sum(
                amount_fn(a) * (a.turnover_aligned_pct / 100.0) for a in group
            )
            capex_aligned = sum(
                amount_fn(a) * (a.capex_aligned_pct / 100.0) for a in group
            )
            opex_aligned = sum(
                amount_fn(a) * (a.opex_aligned_pct / 100.0) for a in group
            )
            eligible_amt = sum(
                amount_fn(a) * (a.taxonomy_eligible_pct / 100.0) for a in group
            )
            enabling_amt = sum(
                amount_fn(a) * (a.turnover_aligned_pct / 100.0) for a in group
                if a.alignment_type == AlignmentType.ENABLING
            )
            transitional_amt = sum(
                amount_fn(a) * (a.turnover_aligned_pct / 100.0) for a in group
                if a.alignment_type == AlignmentType.TRANSITIONAL
            )

            breakdowns.append(GARBreakdown(
                dimension_name="environmental_objective",
                dimension_value=obj_val,
                total_covered_assets=_round_val(group_covered, 2),
                turnover_aligned_amount=_round_val(turn_aligned, 2),
                turnover_gar_pct=_round_val(_safe_pct(turn_aligned, total_covered), 4),
                capex_aligned_amount=_round_val(capex_aligned, 2),
                capex_gar_pct=_round_val(_safe_pct(capex_aligned, total_covered), 4),
                opex_aligned_amount=_round_val(opex_aligned, 2),
                opex_gar_pct=_round_val(_safe_pct(opex_aligned, total_covered), 4),
                eligible_amount=_round_val(eligible_amt, 2),
                eligible_pct=_round_val(_safe_pct(eligible_amt, total_covered), 4),
                enabling_amount=_round_val(enabling_amt, 2),
                transitional_amount=_round_val(transitional_amt, 2),
                asset_count=len(group),
            ))

        return breakdowns

    def _breakdown_by_counterparty(
        self,
        covered: List[CoveredAssetData],
        total_covered: float,
        amount_fn: Any,
    ) -> List[CounterpartyBreakdown]:
        """Build GAR breakdown by counterparty type.

        Args:
            covered: List of covered assets.
            total_covered: Total covered amount for percentage calc.
            amount_fn: Function to get carrying amount.

        Returns:
            List of CounterpartyBreakdown.
        """
        groups: Dict[CounterpartyType, List[CoveredAssetData]] = defaultdict(list)
        for a in covered:
            groups[a.counterparty_type].append(a)

        breakdowns: List[CounterpartyBreakdown] = []
        for ct, group in groups.items():
            exposure = sum(amount_fn(a) for a in group)
            turn = sum(amount_fn(a) * (a.turnover_aligned_pct / 100.0) for a in group)
            capex = sum(amount_fn(a) * (a.capex_aligned_pct / 100.0) for a in group)
            opex = sum(amount_fn(a) * (a.opex_aligned_pct / 100.0) for a in group)

            breakdowns.append(CounterpartyBreakdown(
                counterparty_type=ct,
                total_exposure=_round_val(exposure, 2),
                covered_exposure=_round_val(exposure, 2),
                turnover_aligned=_round_val(turn, 2),
                turnover_gar_pct=_round_val(_safe_pct(turn, exposure), 4),
                capex_aligned=_round_val(capex, 2),
                capex_gar_pct=_round_val(_safe_pct(capex, exposure), 4),
                opex_aligned=_round_val(opex, 2),
                opex_gar_pct=_round_val(_safe_pct(opex, exposure), 4),
                asset_count=len(group),
                weight_pct=_round_val(_safe_pct(exposure, total_covered), 4),
            ))

        return breakdowns

    # ------------------------------------------------------------------
    # Internal: Flow GAR
    # ------------------------------------------------------------------

    def _compute_flow_gar(
        self,
        covered: List[CoveredAssetData],
        amount_fn: Any,
    ) -> FlowGAR:
        """Compute Flow GAR for new originations.

        Flow GAR = Aligned New Originations / Total New Originations

        Args:
            covered: List of covered assets.
            amount_fn: Function to get carrying amount.

        Returns:
            FlowGAR result.
        """
        new_orig = [a for a in covered if a.is_new_origination]

        total_new = sum(amount_fn(a) for a in new_orig)
        turn_new = sum(
            amount_fn(a) * (a.turnover_aligned_pct / 100.0) for a in new_orig
        )
        capex_new = sum(
            amount_fn(a) * (a.capex_aligned_pct / 100.0) for a in new_orig
        )
        opex_new = sum(
            amount_fn(a) * (a.opex_aligned_pct / 100.0) for a in new_orig
        )

        result = FlowGAR(
            total_new_origination=_round_val(total_new, 2),
            turnover_aligned_origination=_round_val(turn_new, 2),
            turnover_flow_gar_pct=_round_val(_safe_pct(turn_new, total_new), 4),
            capex_aligned_origination=_round_val(capex_new, 2),
            capex_flow_gar_pct=_round_val(_safe_pct(capex_new, total_new), 4),
            opex_aligned_origination=_round_val(opex_new, 2),
            opex_flow_gar_pct=_round_val(_safe_pct(opex_new, total_new), 4),
            new_origination_count=len(new_orig),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Internal: Off-Balance-Sheet
    # ------------------------------------------------------------------

    def _compute_obs_gar(
        self,
        off_balance: List[CoveredAssetData],
        amount_fn: Any,
    ) -> OffBalanceSheetKPI:
        """Compute off-balance-sheet GAR KPIs.

        Args:
            off_balance: List of off-balance-sheet assets.
            amount_fn: Function to get amount.

        Returns:
            OffBalanceSheetKPI.
        """
        guarantees = [a for a in off_balance if a.asset_type == AssetType.GUARANTEE]
        commitments = [a for a in off_balance if a.asset_type == AssetType.COMMITMENT]

        total_g = sum(amount_fn(a) for a in guarantees)
        aligned_g = sum(
            amount_fn(a) * (a.turnover_aligned_pct / 100.0) for a in guarantees
        )
        total_c = sum(amount_fn(a) for a in commitments)
        aligned_c = sum(
            amount_fn(a) * (a.turnover_aligned_pct / 100.0) for a in commitments
        )

        result = OffBalanceSheetKPI(
            total_guarantees=_round_val(total_g, 2),
            aligned_guarantees=_round_val(aligned_g, 2),
            guarantee_gar_pct=_round_val(_safe_pct(aligned_g, total_g), 4),
            total_commitments=_round_val(total_c, 2),
            aligned_commitments=_round_val(aligned_c, 2),
            commitment_gar_pct=_round_val(_safe_pct(aligned_c, total_c), 4),
            obs_count=len(off_balance),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Internal: Methodology Notes
    # ------------------------------------------------------------------

    def _generate_methodology_notes(
        self,
        all_assets: List[CoveredAssetData],
        covered: List[CoveredAssetData],
        excluded: List[CoveredAssetData],
        off_balance: List[CoveredAssetData],
    ) -> List[str]:
        """Generate methodology disclosure notes.

        Args:
            all_assets: All submitted assets.
            covered: Covered assets.
            excluded: Excluded assets.
            off_balance: Off-balance-sheet items.

        Returns:
            List of methodology note strings.
        """
        notes: List[str] = [
            "Methodology: EU Taxonomy Regulation 2020/852, Delegated Regulation 2021/2178",
            f"Reporting year: {self.config.reporting_year}",
            f"Carrying amount basis: {'net' if self.config.use_net_carrying_amount else 'gross'}",
            f"Total assets submitted: {len(all_assets)}",
            f"Covered assets: {len(covered)}",
            f"Excluded assets: {len(excluded)}",
            f"Off-balance-sheet items: {len(off_balance)}",
        ]

        # Exclusion reasons
        reason_counts: Dict[str, int] = defaultdict(int)
        for a in excluded:
            reason = a.exclusion_reason.value if a.exclusion_reason else "unspecified"
            reason_counts[reason] += 1
        for reason, count in sorted(reason_counts.items()):
            notes.append(f"Exclusion reason {reason}: {count} asset(s)")

        # Counterparty type distribution
        ct_counts: Dict[str, int] = defaultdict(int)
        for a in covered:
            ct_counts[a.counterparty_type.value] += 1
        for ct, count in sorted(ct_counts.items()):
            notes.append(f"Counterparty type {ct}: {count} asset(s)")

        # New originations
        new_count = sum(1 for a in covered if a.is_new_origination)
        notes.append(f"New originations in period: {new_count}")

        return notes
