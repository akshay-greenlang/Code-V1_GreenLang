# -*- coding: utf-8 -*-
"""
InvestmentsPipelineEngine - Orchestrated 10-stage pipeline for investment emissions.

This module implements the InvestmentsPipelineEngine for AGENT-MRV-028 (Investments,
Scope 3 Category 15). It orchestrates a 10-stage pipeline for complete financed
emissions calculation from raw portfolio input to compliant output with full audit trail.

The 10 stages are:
1. VALIDATE: Input validation (ISIN/ticker format, amounts, asset class)
2. CLASSIFY: Determine asset class, calculation method, PCAF data quality
3. NORMALIZE: Currency conversion to USD, amount normalization
4. RESOLVE_EFS: Look up sector/country/grid emission factors from database
5. CALCULATE: Route to equity/debt/real_asset/sovereign calculator
6. ALLOCATE: Attribution factor calculation per PCAF rules
7. AGGREGATE: Portfolio-level aggregation by asset class, sector, country, WACI
8. COMPLIANCE: Run compliance checker across 9 frameworks
9. PROVENANCE: SHA-256 hashes, Merkle tree
10. SEAL: Final validation and seal results

Example:
    >>> from greenlang.agents.mrv.investments.investments_pipeline import InvestmentsPipelineEngine
    >>> engine = InvestmentsPipelineEngine()
    >>> result = engine.execute(portfolio_input)
    >>> assert result["status"] == "SUCCESS"

Module: greenlang.agents.mrv.investments.investments_pipeline
Agent: AGENT-MRV-028
Version: 1.0.0
"""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import logging
import hashlib
import json
from threading import RLock
from enum import Enum

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

_QUANT_2DP = Decimal("0.01")
_QUANT_8DP = Decimal("0.00000001")

# ==============================================================================
# PIPELINE STATUS
# ==============================================================================


class PipelineStatus(str, Enum):
    """Pipeline execution status."""

    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    FAILED = "FAILED"
    VALIDATION_ERROR = "VALIDATION_ERROR"


class PipelineStage(str, Enum):
    """Pipeline stage identifiers."""

    VALIDATE = "VALIDATE"
    CLASSIFY = "CLASSIFY"
    NORMALIZE = "NORMALIZE"
    RESOLVE_EFS = "RESOLVE_EFS"
    CALCULATE = "CALCULATE"
    ALLOCATE = "ALLOCATE"
    AGGREGATE = "AGGREGATE"
    COMPLIANCE = "COMPLIANCE"
    PROVENANCE = "PROVENANCE"
    SEAL = "SEAL"


class AssetClass(str, Enum):
    """PCAF asset classes."""

    LISTED_EQUITY = "listed_equity"
    CORPORATE_BOND = "corporate_bond"
    PRIVATE_EQUITY = "private_equity"
    PROJECT_FINANCE = "project_finance"
    COMMERCIAL_REAL_ESTATE = "commercial_real_estate"
    MORTGAGE = "mortgage"
    MOTOR_VEHICLE_LOAN = "motor_vehicle_loan"
    SOVEREIGN_BOND = "sovereign_bond"


class CalculationMethod(str, Enum):
    """Calculation method for investment emissions."""

    REPORTED_VERIFIED = "reported_verified"
    REPORTED_UNVERIFIED = "reported_unverified"
    PHYSICAL_ACTIVITY = "physical_activity"
    REVENUE_EEIO = "revenue_eeio"
    ASSET_SPECIFIC = "asset_specific"
    SECTOR_AVERAGE = "sector_average"


# ==============================================================================
# INPUT/OUTPUT MODELS
# ==============================================================================


class InvestmentHoldingInput(BaseModel):
    """Input data for a single investment holding."""

    holding_id: str = Field(..., description="Unique holding identifier")
    tenant_id: str = Field(default="default", description="Tenant identifier")
    asset_class: Optional[str] = Field(None, description="PCAF asset class")

    # Security identifiers
    isin: Optional[str] = Field(None, min_length=12, max_length=12, description="ISIN code")
    ticker: Optional[str] = Field(None, description="Ticker symbol")
    issuer_name: Optional[str] = Field(None, description="Issuer/company name")
    country_code: Optional[str] = Field(None, description="ISO 3166-1 alpha-2 country")
    sector: Optional[str] = Field(None, description="Sector classification")
    naics_code: Optional[str] = Field(None, description="NAICS code for EEIO")

    # Financial data
    outstanding_amount: Optional[Decimal] = Field(None, ge=0, description="Outstanding amount USD")
    investment_amount: Optional[Decimal] = Field(None, ge=0, description="Investment amount USD")
    currency: str = Field(default="USD", description="Currency code")
    fx_rate_to_usd: Decimal = Field(default=Decimal("1.0"), gt=0, description="FX rate to USD")

    # Investee data (for PCAF score 1-2)
    investee_scope1_emissions_kg: Optional[Decimal] = Field(None, ge=0)
    investee_scope2_emissions_kg: Optional[Decimal] = Field(None, ge=0)
    investee_scope3_emissions_kg: Optional[Decimal] = Field(None, ge=0)
    investee_revenue_usd: Optional[Decimal] = Field(None, ge=0)
    investee_evic_usd: Optional[Decimal] = Field(None, ge=0, description="Enterprise Value Including Cash")
    investee_total_equity_usd: Optional[Decimal] = Field(None, ge=0)
    investee_total_debt_usd: Optional[Decimal] = Field(None, ge=0)

    # Real asset data
    property_value_usd: Optional[Decimal] = Field(None, ge=0)
    property_area_sqm: Optional[Decimal] = Field(None, ge=0)
    building_type: Optional[str] = Field(None, description="Building type for EUI lookup")
    epc_rating: Optional[str] = Field(None, description="Energy Performance Certificate rating")
    energy_use_kwh: Optional[Decimal] = Field(None, ge=0, description="Annual energy use")

    # Motor vehicle data
    vehicle_value_usd: Optional[Decimal] = Field(None, ge=0)
    vehicle_type: Optional[str] = Field(None, description="Vehicle type")
    annual_km: Optional[Decimal] = Field(None, ge=0, description="Annual distance driven")

    # Sovereign bond data
    sovereign_country_code: Optional[str] = Field(None, description="Sovereign country ISO code")
    sovereign_gdp_ppp_usd: Optional[Decimal] = Field(None, ge=0, description="GDP PPP in USD")
    sovereign_national_emissions_kg: Optional[Decimal] = Field(None, ge=0)

    # Project finance data
    total_project_cost_usd: Optional[Decimal] = Field(None, ge=0)
    project_emissions_kg: Optional[Decimal] = Field(None, ge=0)

    # PCAF data quality
    pcaf_data_quality_score: Optional[int] = Field(None, ge=1, le=5)
    emissions_verified: bool = Field(default=False, description="Third-party verified")

    # Flags for double-counting prevention
    is_consolidated_scope12: bool = Field(default=False)
    is_equity_share_consolidated: bool = Field(default=False)
    is_short_position: bool = Field(default=False)

    # Metadata
    reporting_year: Optional[int] = Field(None, ge=2000, le=2100)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PortfolioInput(BaseModel):
    """Input data for a portfolio of investment holdings."""

    portfolio_id: str = Field(..., description="Portfolio identifier")
    tenant_id: str = Field(default="default", description="Tenant identifier")
    holdings: List[InvestmentHoldingInput] = Field(..., min_length=1)
    reporting_period_start: Optional[datetime] = None
    reporting_period_end: Optional[datetime] = None
    reporting_year: Optional[int] = Field(None, ge=2000, le=2100)
    base_year: Optional[int] = Field(None, ge=2000, le=2100)
    base_currency: str = Field(default="USD", description="Base currency for aggregation")
    frameworks: List[str] = Field(
        default_factory=lambda: ["GHG_PROTOCOL", "PCAF"],
        description="Compliance frameworks to check"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class InvestmentCalculationResult(BaseModel):
    """Result for a single investment holding calculation."""

    holding_id: str = Field(..., description="Holding identifier")
    asset_class: str = Field(..., description="PCAF asset class")
    calculation_method: str = Field(..., description="Method used")
    pcaf_data_quality_score: int = Field(..., ge=1, le=5, description="PCAF DQ score")

    # Emissions
    financed_emissions_kg_co2e: Decimal = Field(..., description="Attributed emissions kgCO2e")
    scope1_attributed_kg: Decimal = Field(default=Decimal("0"))
    scope2_attributed_kg: Decimal = Field(default=Decimal("0"))
    scope3_attributed_kg: Decimal = Field(default=Decimal("0"))

    # Attribution
    attribution_factor: Decimal = Field(..., ge=0, description="Attribution factor")
    outstanding_amount_usd: Decimal = Field(default=Decimal("0"))

    # Carbon intensity
    carbon_intensity: Optional[Decimal] = Field(None, description="tCO2e per $M")

    # Provenance
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    # Metadata
    detail: Dict[str, Any] = Field(default_factory=dict)


class PortfolioAggregationResult(BaseModel):
    """Aggregated portfolio-level result."""

    portfolio_id: str = Field(..., description="Portfolio identifier")
    status: str = Field(..., description="Pipeline status")

    # Totals
    total_financed_emissions_kg_co2e: Decimal = Field(default=Decimal("0"))
    total_aum_usd: Decimal = Field(default=Decimal("0"))
    covered_aum_usd: Decimal = Field(default=Decimal("0"))
    portfolio_coverage_pct: Decimal = Field(default=Decimal("0"))

    # WACI
    waci_tco2e_per_m_revenue: Optional[Decimal] = None

    # Breakdowns
    by_asset_class: Dict[str, Decimal] = Field(default_factory=dict)
    by_sector: Dict[str, Decimal] = Field(default_factory=dict)
    by_country: Dict[str, Decimal] = Field(default_factory=dict)
    by_data_quality: Dict[int, Decimal] = Field(default_factory=dict)

    # Data quality
    weighted_data_quality_score: Optional[Decimal] = None

    # Individual results
    holdings_results: List[InvestmentCalculationResult] = Field(default_factory=list)
    errors: List[Dict[str, Any]] = Field(default_factory=list)

    # Compliance
    compliance_results: List[Dict[str, Any]] = Field(default_factory=list)

    # Provenance
    provenance_hash: str = Field(default="")
    merkle_root: str = Field(default="")

    # Timing
    processing_time_ms: float = Field(default=0.0)
    stage_durations: Dict[str, float] = Field(default_factory=dict)

    # Metadata
    reporting_year: Optional[int] = None
    base_year: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ==============================================================================
# CURRENCY CONVERSION TABLE
# ==============================================================================

# Simplified FX rates to USD (production would use live rates)
_FX_RATES_TO_USD: Dict[str, Decimal] = {
    "USD": Decimal("1.0"),
    "EUR": Decimal("1.08"),
    "GBP": Decimal("1.27"),
    "JPY": Decimal("0.0067"),
    "CHF": Decimal("1.13"),
    "CAD": Decimal("0.74"),
    "AUD": Decimal("0.65"),
    "CNY": Decimal("0.14"),
    "INR": Decimal("0.012"),
    "BRL": Decimal("0.20"),
    "KRW": Decimal("0.00075"),
    "SGD": Decimal("0.74"),
    "HKD": Decimal("0.13"),
    "NOK": Decimal("0.094"),
    "SEK": Decimal("0.096"),
    "DKK": Decimal("0.145"),
    "ZAR": Decimal("0.055"),
    "MXN": Decimal("0.058"),
}

# ==============================================================================
# SECTOR EMISSION FACTORS (tCO2e per $M revenue, simplified)
# ==============================================================================

_SECTOR_EFS: Dict[str, Decimal] = {
    "oil_and_gas": Decimal("450.0"),
    "power_generation": Decimal("380.0"),
    "coal_mining": Decimal("620.0"),
    "automotive": Decimal("120.0"),
    "cement": Decimal("550.0"),
    "steel": Decimal("480.0"),
    "real_estate": Decimal("45.0"),
    "agriculture": Decimal("180.0"),
    "aviation": Decimal("320.0"),
    "shipping": Decimal("280.0"),
    "technology": Decimal("15.0"),
    "financial_services": Decimal("8.0"),
    "healthcare": Decimal("22.0"),
    "retail": Decimal("35.0"),
    "telecom": Decimal("18.0"),
    "chemicals": Decimal("200.0"),
    "mining": Decimal("350.0"),
    "construction": Decimal("85.0"),
    "food_beverage": Decimal("95.0"),
    "textiles": Decimal("70.0"),
    "default": Decimal("50.0"),
}

# ==============================================================================
# COUNTRY GRID EMISSION FACTORS (kgCO2e per kWh)
# ==============================================================================

_COUNTRY_GRID_EFS: Dict[str, Decimal] = {
    "US": Decimal("0.417"),
    "GB": Decimal("0.233"),
    "DE": Decimal("0.385"),
    "FR": Decimal("0.052"),
    "JP": Decimal("0.506"),
    "CN": Decimal("0.581"),
    "IN": Decimal("0.708"),
    "AU": Decimal("0.656"),
    "CA": Decimal("0.120"),
    "BR": Decimal("0.074"),
    "KR": Decimal("0.459"),
    "GLOBAL": Decimal("0.436"),
}

# ==============================================================================
# MOTOR VEHICLE EMISSION FACTORS (kgCO2e per km)
# ==============================================================================

_VEHICLE_EFS: Dict[str, Decimal] = {
    "car_petrol": Decimal("0.170"),
    "car_diesel": Decimal("0.168"),
    "car_hybrid": Decimal("0.120"),
    "car_electric": Decimal("0.050"),
    "car_average": Decimal("0.160"),
    "suv_petrol": Decimal("0.220"),
    "suv_diesel": Decimal("0.210"),
    "suv_electric": Decimal("0.065"),
    "truck_light": Decimal("0.250"),
    "truck_heavy": Decimal("0.900"),
    "van": Decimal("0.230"),
    "motorcycle": Decimal("0.095"),
    "default": Decimal("0.170"),
}


# ==============================================================================
# InvestmentsPipelineEngine
# ==============================================================================


class InvestmentsPipelineEngine:
    """
    InvestmentsPipelineEngine - 10-stage orchestrated pipeline for financed emissions.

    This engine coordinates the complete investment emissions calculation
    workflow through 10 sequential stages, from input validation to sealed
    audit trail. It supports all 8 PCAF asset classes and routes to the
    appropriate calculator engine for each.

    The engine uses lazy initialization for sub-engines.

    Attributes:
        _database_engine: InvestmentDatabaseEngine (lazy-loaded)
        _equity_engine: EquityInvestmentCalculatorEngine (lazy-loaded)
        _debt_engine: DebtInvestmentCalculatorEngine (lazy-loaded)
        _real_asset_engine: RealAssetCalculatorEngine (lazy-loaded)
        _sovereign_engine: SovereignBondCalculatorEngine (lazy-loaded)
        _compliance_engine: ComplianceCheckerEngine (lazy-loaded)

    Example:
        >>> engine = InvestmentsPipelineEngine()
        >>> result = engine.execute(portfolio_input)
        >>> assert result.status == "SUCCESS"
    """

    _instance: Optional["InvestmentsPipelineEngine"] = None
    _lock: RLock = RLock()

    def __new__(cls) -> "InvestmentsPipelineEngine":
        """Thread-safe singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """Initialize InvestmentsPipelineEngine."""
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._database_engine: Optional[Any] = None
        self._equity_engine: Optional[Any] = None
        self._debt_engine: Optional[Any] = None
        self._real_asset_engine: Optional[Any] = None
        self._sovereign_engine: Optional[Any] = None
        self._compliance_engine: Optional[Any] = None

        self._provenance_chains: Dict[str, List[Dict[str, Any]]] = {}

        self._initialized = True
        logger.info("InvestmentsPipelineEngine initialized (version 1.0.0)")

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        with cls._lock:
            cls._instance = None

    # ==========================================================================
    # PUBLIC API
    # ==========================================================================

    def execute(self, portfolio_input: PortfolioInput) -> PortfolioAggregationResult:
        """
        Execute the full 10-stage pipeline for a portfolio.

        Args:
            portfolio_input: Portfolio with holdings.

        Returns:
            PortfolioAggregationResult with emissions and compliance.

        Raises:
            ValueError: If input validation fails.
            RuntimeError: If pipeline execution fails.
        """
        chain_id = f"inv-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
        self._provenance_chains[chain_id] = []
        stage_durations: Dict[str, float] = {}

        try:
            # Stage 1: VALIDATE
            start = datetime.now(timezone.utc)
            validated_holdings = self._stage_validate(portfolio_input)
            dur = self._elapsed_ms(start)
            stage_durations[PipelineStage.VALIDATE.value] = dur
            self._record_provenance(chain_id, PipelineStage.VALIDATE, len(validated_holdings))
            logger.info(f"[{chain_id}] VALIDATE completed in {dur:.2f}ms ({len(validated_holdings)} holdings)")

            # Stage 2: CLASSIFY
            start = datetime.now(timezone.utc)
            classified = self._stage_classify(validated_holdings)
            dur = self._elapsed_ms(start)
            stage_durations[PipelineStage.CLASSIFY.value] = dur
            self._record_provenance(chain_id, PipelineStage.CLASSIFY, len(classified))
            logger.info(f"[{chain_id}] CLASSIFY completed in {dur:.2f}ms")

            # Stage 3: NORMALIZE
            start = datetime.now(timezone.utc)
            normalized = self._stage_normalize(classified)
            dur = self._elapsed_ms(start)
            stage_durations[PipelineStage.NORMALIZE.value] = dur
            self._record_provenance(chain_id, PipelineStage.NORMALIZE, len(normalized))
            logger.info(f"[{chain_id}] NORMALIZE completed in {dur:.2f}ms")

            # Stage 4: RESOLVE_EFS
            start = datetime.now(timezone.utc)
            with_efs = self._stage_resolve_efs(normalized)
            dur = self._elapsed_ms(start)
            stage_durations[PipelineStage.RESOLVE_EFS.value] = dur
            self._record_provenance(chain_id, PipelineStage.RESOLVE_EFS, len(with_efs))
            logger.info(f"[{chain_id}] RESOLVE_EFS completed in {dur:.2f}ms")

            # Stage 5: CALCULATE
            start = datetime.now(timezone.utc)
            calc_results, calc_errors = self._stage_calculate(with_efs)
            dur = self._elapsed_ms(start)
            stage_durations[PipelineStage.CALCULATE.value] = dur
            self._record_provenance(chain_id, PipelineStage.CALCULATE, len(calc_results))
            logger.info(
                f"[{chain_id}] CALCULATE completed in {dur:.2f}ms "
                f"(success={len(calc_results)}, errors={len(calc_errors)})"
            )

            # Stage 6: ALLOCATE
            start = datetime.now(timezone.utc)
            allocated = self._stage_allocate(calc_results)
            dur = self._elapsed_ms(start)
            stage_durations[PipelineStage.ALLOCATE.value] = dur
            self._record_provenance(chain_id, PipelineStage.ALLOCATE, len(allocated))
            logger.info(f"[{chain_id}] ALLOCATE completed in {dur:.2f}ms")

            # Stage 7: AGGREGATE
            start = datetime.now(timezone.utc)
            aggregated = self._stage_aggregate(allocated, portfolio_input)
            dur = self._elapsed_ms(start)
            stage_durations[PipelineStage.AGGREGATE.value] = dur
            self._record_provenance(chain_id, PipelineStage.AGGREGATE, None)
            logger.info(
                f"[{chain_id}] AGGREGATE completed in {dur:.2f}ms "
                f"(total={aggregated.get('total_financed_emissions_kg_co2e', 0)})"
            )

            # Stage 8: COMPLIANCE
            start = datetime.now(timezone.utc)
            compliance_results = self._stage_compliance(aggregated, portfolio_input.frameworks)
            dur = self._elapsed_ms(start)
            stage_durations[PipelineStage.COMPLIANCE.value] = dur
            self._record_provenance(chain_id, PipelineStage.COMPLIANCE, len(compliance_results))
            logger.info(f"[{chain_id}] COMPLIANCE completed in {dur:.2f}ms")

            # Stage 9: PROVENANCE
            start = datetime.now(timezone.utc)
            provenance_hash, merkle_root = self._stage_provenance(chain_id, allocated)
            dur = self._elapsed_ms(start)
            stage_durations[PipelineStage.PROVENANCE.value] = dur
            logger.info(f"[{chain_id}] PROVENANCE completed in {dur:.2f}ms")

            # Stage 10: SEAL
            start = datetime.now(timezone.utc)
            sealed = self._stage_seal(aggregated, provenance_hash, merkle_root)
            dur = self._elapsed_ms(start)
            stage_durations[PipelineStage.SEAL.value] = dur
            logger.info(f"[{chain_id}] SEAL completed in {dur:.2f}ms")

            total_dur = sum(stage_durations.values())
            logger.info(
                f"[{chain_id}] Pipeline completed in {total_dur:.2f}ms. "
                f"Total financed emissions: {aggregated.get('total_financed_emissions_kg_co2e', 0)} kgCO2e"
            )

            return PortfolioAggregationResult(
                portfolio_id=portfolio_input.portfolio_id,
                status=PipelineStatus.SUCCESS.value if not calc_errors else PipelineStatus.PARTIAL_SUCCESS.value,
                total_financed_emissions_kg_co2e=aggregated.get("total_financed_emissions_kg_co2e", Decimal("0")),
                total_aum_usd=aggregated.get("total_aum_usd", Decimal("0")),
                covered_aum_usd=aggregated.get("covered_aum_usd", Decimal("0")),
                portfolio_coverage_pct=aggregated.get("portfolio_coverage_pct", Decimal("0")),
                waci_tco2e_per_m_revenue=aggregated.get("waci"),
                by_asset_class=aggregated.get("by_asset_class", {}),
                by_sector=aggregated.get("by_sector", {}),
                by_country=aggregated.get("by_country", {}),
                by_data_quality=aggregated.get("by_data_quality", {}),
                weighted_data_quality_score=aggregated.get("weighted_dq"),
                holdings_results=allocated,
                errors=calc_errors,
                compliance_results=compliance_results,
                provenance_hash=provenance_hash,
                merkle_root=merkle_root,
                processing_time_ms=total_dur,
                stage_durations=stage_durations,
                reporting_year=portfolio_input.reporting_year,
                base_year=portfolio_input.base_year,
            )

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"[{chain_id}] Pipeline failed: {e}", exc_info=True)
            raise RuntimeError(f"Pipeline failed: {e}") from e
        finally:
            self._provenance_chains.pop(chain_id, None)

    def execute_single(self, holding: InvestmentHoldingInput) -> InvestmentCalculationResult:
        """
        Execute pipeline for a single holding.

        Args:
            holding: Single investment holding input.

        Returns:
            InvestmentCalculationResult.
        """
        portfolio = PortfolioInput(
            portfolio_id=f"single-{holding.holding_id}",
            tenant_id=holding.tenant_id,
            holdings=[holding],
            reporting_year=holding.reporting_year,
        )
        result = self.execute(portfolio)
        if result.holdings_results:
            return result.holdings_results[0]
        raise RuntimeError("Single holding calculation produced no results")

    def execute_batch(
        self, holdings: List[InvestmentHoldingInput]
    ) -> List[InvestmentCalculationResult]:
        """
        Execute pipeline for a batch of independent holdings.

        Args:
            holdings: List of holdings to process.

        Returns:
            List of InvestmentCalculationResult.
        """
        portfolio = PortfolioInput(
            portfolio_id=f"batch-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            holdings=holdings,
        )
        result = self.execute(portfolio)
        return result.holdings_results

    # ==========================================================================
    # STAGE 1: VALIDATE
    # ==========================================================================

    def _stage_validate(
        self, portfolio: PortfolioInput
    ) -> List[InvestmentHoldingInput]:
        """
        Stage 1: Input validation.

        Checks:
        - At least one holding
        - Holding ID not empty
        - Amounts non-negative
        - ISIN format (12 chars, alphanumeric)
        - Short positions flagged for exclusion
        - Consolidated investments flagged for exclusion

        Args:
            portfolio: Portfolio input.

        Returns:
            List of validated holdings (short positions excluded).

        Raises:
            ValueError: If critical validation fails.
        """
        errors: List[str] = []

        if not portfolio.holdings:
            raise ValueError("Portfolio must contain at least one holding")

        valid_holdings: List[InvestmentHoldingInput] = []

        for idx, h in enumerate(portfolio.holdings):
            if not h.holding_id:
                errors.append(f"Holding {idx}: missing holding_id")
                continue

            # Exclude short positions (DC-INV-008)
            if h.is_short_position:
                logger.info(f"Holding {h.holding_id}: excluded (short position, DC-INV-008)")
                continue

            # Exclude consolidated investments (DC-INV-001)
            if h.is_consolidated_scope12:
                logger.info(f"Holding {h.holding_id}: excluded (consolidated Scope 1/2, DC-INV-001)")
                continue

            # ISIN validation
            if h.isin and (len(h.isin) != 12 or not h.isin.isalnum()):
                errors.append(f"Holding {h.holding_id}: invalid ISIN format '{h.isin}'")

            # Amount positivity
            for field_name in ("outstanding_amount", "investment_amount"):
                val = getattr(h, field_name, None)
                if val is not None and val < 0:
                    errors.append(f"Holding {h.holding_id}: {field_name} must be >= 0")

            valid_holdings.append(h)

        if errors:
            logger.warning(f"Validation warnings: {errors}")

        if not valid_holdings:
            raise ValueError(f"No valid holdings after validation: {errors}")

        return valid_holdings

    # ==========================================================================
    # STAGE 2: CLASSIFY
    # ==========================================================================

    def _stage_classify(
        self, holdings: List[InvestmentHoldingInput]
    ) -> List[Dict[str, Any]]:
        """
        Stage 2: Classify each holding by asset class, method, and PCAF DQ.

        Args:
            holdings: Validated holdings.

        Returns:
            List of classified holding dictionaries.
        """
        classified: List[Dict[str, Any]] = []

        for h in holdings:
            asset_class = self._determine_asset_class(h)
            method = self._determine_calculation_method(h)
            dq_score = self._determine_pcaf_dq_score(h, method)

            classified.append({
                "holding": h,
                "asset_class": asset_class,
                "calculation_method": method,
                "pcaf_dq_score": dq_score,
            })

        return classified

    def _determine_asset_class(self, h: InvestmentHoldingInput) -> str:
        """Determine PCAF asset class for a holding."""
        if h.asset_class:
            return h.asset_class

        if h.sovereign_country_code or h.sovereign_gdp_ppp_usd:
            return AssetClass.SOVEREIGN_BOND.value
        if h.property_value_usd or h.building_type or h.epc_rating:
            if h.outstanding_amount and h.property_value_usd:
                return AssetClass.MORTGAGE.value
            return AssetClass.COMMERCIAL_REAL_ESTATE.value
        if h.vehicle_type or h.vehicle_value_usd:
            return AssetClass.MOTOR_VEHICLE_LOAN.value
        if h.total_project_cost_usd:
            return AssetClass.PROJECT_FINANCE.value
        if h.investee_evic_usd:
            return AssetClass.LISTED_EQUITY.value
        if h.investee_total_debt_usd:
            return AssetClass.CORPORATE_BOND.value
        if h.investee_total_equity_usd:
            return AssetClass.PRIVATE_EQUITY.value

        return AssetClass.LISTED_EQUITY.value

    def _determine_calculation_method(self, h: InvestmentHoldingInput) -> str:
        """Determine calculation method based on data availability."""
        if h.investee_scope1_emissions_kg is not None:
            if h.emissions_verified:
                return CalculationMethod.REPORTED_VERIFIED.value
            return CalculationMethod.REPORTED_UNVERIFIED.value
        if h.energy_use_kwh is not None or h.annual_km is not None:
            return CalculationMethod.PHYSICAL_ACTIVITY.value
        if h.investee_revenue_usd is not None and h.sector:
            return CalculationMethod.REVENUE_EEIO.value
        if h.building_type or h.vehicle_type:
            return CalculationMethod.ASSET_SPECIFIC.value
        return CalculationMethod.SECTOR_AVERAGE.value

    def _determine_pcaf_dq_score(self, h: InvestmentHoldingInput, method: str) -> int:
        """Determine PCAF data quality score 1-5."""
        if h.pcaf_data_quality_score:
            return h.pcaf_data_quality_score

        method_to_dq: Dict[str, int] = {
            CalculationMethod.REPORTED_VERIFIED.value: 1,
            CalculationMethod.REPORTED_UNVERIFIED.value: 2,
            CalculationMethod.PHYSICAL_ACTIVITY.value: 2,
            CalculationMethod.REVENUE_EEIO.value: 3,
            CalculationMethod.ASSET_SPECIFIC.value: 3,
            CalculationMethod.SECTOR_AVERAGE.value: 5,
        }
        return method_to_dq.get(method, 5)

    # ==========================================================================
    # STAGE 3: NORMALIZE
    # ==========================================================================

    def _stage_normalize(self, classified: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Stage 3: Currency conversion and amount normalization to USD.

        Args:
            classified: Classified holdings.

        Returns:
            Holdings with normalized USD amounts.
        """
        for item in classified:
            h: InvestmentHoldingInput = item["holding"]
            fx = self._get_fx_rate(h.currency)

            # Normalize key amounts
            if h.outstanding_amount is not None:
                item["outstanding_amount_usd"] = (h.outstanding_amount * fx).quantize(_QUANT_2DP)
            elif h.investment_amount is not None:
                item["outstanding_amount_usd"] = (h.investment_amount * fx).quantize(_QUANT_2DP)
            else:
                item["outstanding_amount_usd"] = Decimal("0")

            if h.investee_revenue_usd is not None:
                item["investee_revenue_usd"] = h.investee_revenue_usd
            else:
                item["investee_revenue_usd"] = Decimal("0")

            item["fx_rate"] = fx

        return classified

    def _get_fx_rate(self, currency: str) -> Decimal:
        """Get FX rate to USD."""
        return _FX_RATES_TO_USD.get(currency.upper(), Decimal("1.0"))

    # ==========================================================================
    # STAGE 4: RESOLVE_EFS
    # ==========================================================================

    def _stage_resolve_efs(self, normalized: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Stage 4: Resolve emission factors from database/lookup tables.

        Args:
            normalized: Normalized holdings.

        Returns:
            Holdings with resolved emission factors.
        """
        for item in normalized:
            h: InvestmentHoldingInput = item["holding"]
            asset_class = item["asset_class"]
            method = item["calculation_method"]

            # Sector EF for EEIO / sector average methods
            sector = (h.sector or "default").lower().replace(" ", "_")
            item["sector_ef_tco2e_per_m"] = _SECTOR_EFS.get(sector, _SECTOR_EFS["default"])

            # Country grid EF for real asset energy calculations
            country = (h.country_code or "GLOBAL").upper()
            item["grid_ef_kgco2e_per_kwh"] = _COUNTRY_GRID_EFS.get(
                country, _COUNTRY_GRID_EFS["GLOBAL"]
            )

            # Vehicle EF for motor vehicle loans
            vtype = (h.vehicle_type or "default").lower().replace(" ", "_")
            item["vehicle_ef_kgco2e_per_km"] = _VEHICLE_EFS.get(vtype, _VEHICLE_EFS["default"])

        return normalized

    # ==========================================================================
    # STAGE 5: CALCULATE
    # ==========================================================================

    def _stage_calculate(
        self, with_efs: List[Dict[str, Any]]
    ) -> Tuple[List[InvestmentCalculationResult], List[Dict[str, Any]]]:
        """
        Stage 5: Route to appropriate calculator and compute emissions.

        Args:
            with_efs: Holdings with resolved emission factors.

        Returns:
            Tuple of (results, errors).
        """
        results: List[InvestmentCalculationResult] = []
        errors: List[Dict[str, Any]] = []

        for idx, item in enumerate(with_efs):
            try:
                calc_result = self._calculate_holding(item)
                results.append(calc_result)
            except Exception as e:
                h: InvestmentHoldingInput = item["holding"]
                logger.error(f"Calculation failed for {h.holding_id}: {e}")
                errors.append({
                    "index": idx,
                    "holding_id": h.holding_id,
                    "error": str(e),
                })

        return results, errors

    def _calculate_holding(self, item: Dict[str, Any]) -> InvestmentCalculationResult:
        """Calculate emissions for a single holding based on asset class."""
        h: InvestmentHoldingInput = item["holding"]
        asset_class = item["asset_class"]
        method = item["calculation_method"]
        dq_score = item["pcaf_dq_score"]
        outstanding_usd = item.get("outstanding_amount_usd", Decimal("0"))

        # Calculate attribution factor
        attribution_factor = self._calculate_attribution_factor(item)

        # Calculate investee total emissions
        investee_emissions = self._calculate_investee_emissions(item)

        # Financed emissions = attribution_factor * investee_emissions
        financed = (attribution_factor * investee_emissions).quantize(_QUANT_2DP)

        # Scope breakdown (proportional if investee data available)
        s1_attr = Decimal("0")
        s2_attr = Decimal("0")
        s3_attr = Decimal("0")
        if h.investee_scope1_emissions_kg and investee_emissions > 0:
            total_inv = (
                (h.investee_scope1_emissions_kg or Decimal("0"))
                + (h.investee_scope2_emissions_kg or Decimal("0"))
                + (h.investee_scope3_emissions_kg or Decimal("0"))
            )
            if total_inv > 0:
                s1_attr = (attribution_factor * (h.investee_scope1_emissions_kg or Decimal("0"))).quantize(_QUANT_2DP)
                s2_attr = (attribution_factor * (h.investee_scope2_emissions_kg or Decimal("0"))).quantize(_QUANT_2DP)
                s3_attr = (attribution_factor * (h.investee_scope3_emissions_kg or Decimal("0"))).quantize(_QUANT_2DP)

        # Carbon intensity (tCO2e per $M revenue)
        carbon_intensity: Optional[Decimal] = None
        revenue = item.get("investee_revenue_usd", Decimal("0"))
        if revenue > 0 and financed > 0:
            carbon_intensity = (
                (financed / Decimal("1000")) / (revenue / Decimal("1000000"))
            ).quantize(_QUANT_2DP)

        return InvestmentCalculationResult(
            holding_id=h.holding_id,
            asset_class=asset_class,
            calculation_method=method,
            pcaf_data_quality_score=dq_score,
            financed_emissions_kg_co2e=financed,
            scope1_attributed_kg=s1_attr,
            scope2_attributed_kg=s2_attr,
            scope3_attributed_kg=s3_attr,
            attribution_factor=attribution_factor,
            outstanding_amount_usd=outstanding_usd,
            carbon_intensity=carbon_intensity,
            detail={
                "asset_class": asset_class,
                "method": method,
                "dq_score": dq_score,
                "investee_emissions_kg": str(investee_emissions),
                "attribution_factor": str(attribution_factor),
                "sector": h.sector or "unknown",
                "country": h.country_code or "unknown",
            },
        )

    def _calculate_attribution_factor(self, item: Dict[str, Any]) -> Decimal:
        """
        Calculate PCAF attribution factor based on asset class.

        Formulas:
        - Listed equity / corp bond: outstanding / EVIC
        - Private equity: investment / total equity
        - Project finance: investment / total project cost
        - CRE / Mortgage: outstanding / property value at origination
        - Motor vehicle: outstanding / vehicle value
        - Sovereign bond: outstanding / PPP-adjusted GDP
        """
        h: InvestmentHoldingInput = item["holding"]
        asset_class = item["asset_class"]
        outstanding_usd = item.get("outstanding_amount_usd", Decimal("0"))

        if outstanding_usd <= 0:
            return Decimal("0")

        if asset_class in (AssetClass.LISTED_EQUITY.value, AssetClass.CORPORATE_BOND.value):
            evic = h.investee_evic_usd
            if evic and evic > 0:
                return (outstanding_usd / evic).quantize(_QUANT_8DP)
            # Fallback: use total equity + total debt as proxy
            total = (h.investee_total_equity_usd or Decimal("0")) + (h.investee_total_debt_usd or Decimal("0"))
            if total > 0:
                return (outstanding_usd / total).quantize(_QUANT_8DP)
            return Decimal("1.0")

        elif asset_class == AssetClass.PRIVATE_EQUITY.value:
            equity = h.investee_total_equity_usd
            if equity and equity > 0:
                return (outstanding_usd / equity).quantize(_QUANT_8DP)
            return Decimal("1.0")

        elif asset_class == AssetClass.PROJECT_FINANCE.value:
            cost = h.total_project_cost_usd
            if cost and cost > 0:
                return (outstanding_usd / cost).quantize(_QUANT_8DP)
            return Decimal("1.0")

        elif asset_class in (AssetClass.COMMERCIAL_REAL_ESTATE.value, AssetClass.MORTGAGE.value):
            pv = h.property_value_usd
            if pv and pv > 0:
                return (outstanding_usd / pv).quantize(_QUANT_8DP)
            return Decimal("1.0")

        elif asset_class == AssetClass.MOTOR_VEHICLE_LOAN.value:
            vv = h.vehicle_value_usd
            if vv and vv > 0:
                return (outstanding_usd / vv).quantize(_QUANT_8DP)
            return Decimal("1.0")

        elif asset_class == AssetClass.SOVEREIGN_BOND.value:
            gdp = h.sovereign_gdp_ppp_usd
            if gdp and gdp > 0:
                return (outstanding_usd / gdp).quantize(_QUANT_8DP)
            return Decimal("0.00000001")

        return Decimal("1.0")

    def _calculate_investee_emissions(self, item: Dict[str, Any]) -> Decimal:
        """
        Calculate total investee emissions based on method and data.

        Priority:
        1. Reported Scope 1+2 (+3 if available)
        2. Physical activity based
        3. Revenue-based EEIO
        4. Asset-specific
        5. Sector average
        """
        h: InvestmentHoldingInput = item["holding"]
        method = item["calculation_method"]
        asset_class = item["asset_class"]

        # Method 1/2: Reported emissions
        if method in (CalculationMethod.REPORTED_VERIFIED.value, CalculationMethod.REPORTED_UNVERIFIED.value):
            total = (h.investee_scope1_emissions_kg or Decimal("0")) + (h.investee_scope2_emissions_kg or Decimal("0"))
            if h.investee_scope3_emissions_kg:
                total += h.investee_scope3_emissions_kg
            return total

        # Method: Physical activity
        if method == CalculationMethod.PHYSICAL_ACTIVITY.value:
            if h.energy_use_kwh and h.energy_use_kwh > 0:
                grid_ef = item.get("grid_ef_kgco2e_per_kwh", Decimal("0.436"))
                return (h.energy_use_kwh * grid_ef).quantize(_QUANT_2DP)
            if h.annual_km and h.annual_km > 0:
                veh_ef = item.get("vehicle_ef_kgco2e_per_km", Decimal("0.170"))
                return (h.annual_km * veh_ef).quantize(_QUANT_2DP)
            return Decimal("0")

        # Method: Revenue-based EEIO
        if method == CalculationMethod.REVENUE_EEIO.value:
            revenue = item.get("investee_revenue_usd", Decimal("0"))
            sector_ef = item.get("sector_ef_tco2e_per_m", Decimal("50.0"))
            if revenue > 0:
                # sector_ef is tCO2e per $M revenue, convert to kgCO2e
                emissions_tco2e = (revenue / Decimal("1000000")) * sector_ef
                return (emissions_tco2e * Decimal("1000")).quantize(_QUANT_2DP)
            return Decimal("0")

        # Method: Asset-specific (real estate EUI or vehicle EF)
        if method == CalculationMethod.ASSET_SPECIFIC.value:
            if asset_class in (AssetClass.COMMERCIAL_REAL_ESTATE.value, AssetClass.MORTGAGE.value):
                area = h.property_area_sqm or Decimal("0")
                if area > 0:
                    # Default EUI: 200 kWh/m2/year for commercial buildings
                    eui = Decimal("200")
                    grid_ef = item.get("grid_ef_kgco2e_per_kwh", Decimal("0.436"))
                    return (area * eui * grid_ef).quantize(_QUANT_2DP)
            if asset_class == AssetClass.MOTOR_VEHICLE_LOAN.value:
                km = h.annual_km or Decimal("15000")  # Default 15k km/year
                veh_ef = item.get("vehicle_ef_kgco2e_per_km", Decimal("0.170"))
                return (km * veh_ef).quantize(_QUANT_2DP)
            return Decimal("0")

        # Method: Sector average (fallback)
        sector_ef = item.get("sector_ef_tco2e_per_m", Decimal("50.0"))
        outstanding_usd = item.get("outstanding_amount_usd", Decimal("0"))
        if outstanding_usd > 0:
            emissions_tco2e = (outstanding_usd / Decimal("1000000")) * sector_ef
            return (emissions_tco2e * Decimal("1000")).quantize(_QUANT_2DP)
        return Decimal("0")

    # ==========================================================================
    # STAGE 6: ALLOCATE
    # ==========================================================================

    def _stage_allocate(
        self, results: List[InvestmentCalculationResult]
    ) -> List[InvestmentCalculationResult]:
        """
        Stage 6: Attribution factor already applied in calculate stage.

        This stage validates attribution factors and applies any final adjustments.

        Args:
            results: Calculated results with attribution.

        Returns:
            Results with validated attribution.
        """
        for r in results:
            if r.attribution_factor < Decimal("0"):
                logger.warning(f"Holding {r.holding_id}: negative attribution factor, clamping to 0")
                r.attribution_factor = Decimal("0")
                r.financed_emissions_kg_co2e = Decimal("0")

        return results

    # ==========================================================================
    # STAGE 7: AGGREGATE
    # ==========================================================================

    def _stage_aggregate(
        self,
        results: List[InvestmentCalculationResult],
        portfolio: PortfolioInput,
    ) -> Dict[str, Any]:
        """
        Stage 7: Portfolio-level aggregation.

        Aggregates by asset class, sector, country. Calculates WACI and
        weighted data quality score.

        Args:
            results: Allocated results.
            portfolio: Original portfolio input.

        Returns:
            Aggregated dictionary.
        """
        total_emissions = Decimal("0")
        total_aum = Decimal("0")
        by_asset_class: Dict[str, Decimal] = {}
        by_sector: Dict[str, Decimal] = {}
        by_country: Dict[str, Decimal] = {}
        by_dq: Dict[int, Decimal] = {}

        # For WACI calculation
        waci_numerator = Decimal("0")
        waci_denominator = Decimal("0")
        dq_weighted_sum = Decimal("0")
        dq_weight_total = Decimal("0")

        for r in results:
            emissions = r.financed_emissions_kg_co2e
            outstanding = r.outstanding_amount_usd
            total_emissions += emissions
            total_aum += outstanding

            # By asset class
            ac = r.asset_class
            by_asset_class[ac] = by_asset_class.get(ac, Decimal("0")) + emissions

            # By sector / country from detail
            sector = r.detail.get("sector", "unknown")
            country = r.detail.get("country", "unknown")
            by_sector[sector] = by_sector.get(sector, Decimal("0")) + emissions
            by_country[country] = by_country.get(country, Decimal("0")) + emissions

            # By data quality
            dq = r.pcaf_data_quality_score
            by_dq[dq] = by_dq.get(dq, Decimal("0")) + emissions

            # WACI: sum(portfolio_weight_i * carbon_intensity_i)
            if r.carbon_intensity is not None and outstanding > 0:
                waci_numerator += outstanding * r.carbon_intensity
                waci_denominator += outstanding

            # DQ weighted average
            if outstanding > 0:
                dq_weighted_sum += outstanding * Decimal(str(dq))
                dq_weight_total += outstanding

        # Portfolio coverage
        coverage_pct = Decimal("0")
        if total_aum > 0:
            coverage_pct = ((total_aum / total_aum) * Decimal("100")).quantize(_QUANT_2DP)

        # WACI
        waci: Optional[Decimal] = None
        if waci_denominator > 0:
            waci = (waci_numerator / waci_denominator).quantize(_QUANT_2DP)

        # Weighted DQ
        weighted_dq: Optional[Decimal] = None
        if dq_weight_total > 0:
            weighted_dq = (dq_weighted_sum / dq_weight_total).quantize(_QUANT_2DP)

        return {
            "total_financed_emissions_kg_co2e": total_emissions,
            "total_aum_usd": total_aum,
            "covered_aum_usd": total_aum,
            "portfolio_coverage_pct": coverage_pct,
            "waci": waci,
            "weighted_dq": weighted_dq,
            "by_asset_class": by_asset_class,
            "by_sector": by_sector,
            "by_country": by_country,
            "by_data_quality": by_dq,
        }

    # ==========================================================================
    # STAGE 8: COMPLIANCE
    # ==========================================================================

    def _stage_compliance(
        self,
        aggregated: Dict[str, Any],
        frameworks: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Stage 8: Run compliance checker across frameworks.

        Args:
            aggregated: Aggregated portfolio data.
            frameworks: List of framework name strings.

        Returns:
            List of compliance result dictionaries.
        """
        try:
            from greenlang.agents.mrv.investments.compliance_checker import (
                ComplianceCheckerEngine,
                InvestmentResultInput,
            )

            engine = ComplianceCheckerEngine.get_instance()

            input_data = InvestmentResultInput(
                total_emissions_kg_co2e=aggregated.get("total_financed_emissions_kg_co2e", Decimal("0")),
                total_aum=aggregated.get("total_aum_usd"),
                covered_aum=aggregated.get("covered_aum_usd"),
                portfolio_coverage_pct=aggregated.get("portfolio_coverage_pct"),
                waci_value=aggregated.get("waci"),
                asset_class_breakdown={
                    k: v for k, v in aggregated.get("by_asset_class", {}).items()
                },
                weighted_data_quality_score=aggregated.get("weighted_dq"),
            )

            results = engine.check_compliance(input_data, frameworks)
            return [
                {
                    "framework": r.framework.value,
                    "status": r.status.value,
                    "score": float(r.score),
                    "passed": r.passed_checks,
                    "failed": r.failed_checks,
                    "warnings": r.warning_checks,
                }
                for r in results
            ]

        except ImportError:
            logger.warning("ComplianceCheckerEngine not available, skipping compliance")
            return []
        except Exception as e:
            logger.error(f"Compliance stage failed: {e}", exc_info=True)
            return [{"error": str(e)}]

    # ==========================================================================
    # STAGE 9: PROVENANCE
    # ==========================================================================

    def _stage_provenance(
        self,
        chain_id: str,
        results: List[InvestmentCalculationResult],
    ) -> Tuple[str, str]:
        """
        Stage 9: Calculate SHA-256 provenance hashes and Merkle root.

        Args:
            chain_id: Provenance chain identifier.
            results: Calculation results.

        Returns:
            Tuple of (provenance_hash, merkle_root).
        """
        # Hash each holding result
        leaf_hashes: List[str] = []
        for r in results:
            data = f"{r.holding_id}|{r.financed_emissions_kg_co2e}|{r.attribution_factor}"
            h = hashlib.sha256(data.encode("utf-8")).hexdigest()
            r.provenance_hash = h
            leaf_hashes.append(h)

        # Merkle root
        merkle_root = self._compute_merkle_root(leaf_hashes)

        # Overall provenance hash
        chain_data = json.dumps(self._provenance_chains.get(chain_id, []), default=str)
        provenance_hash = hashlib.sha256(
            f"{chain_data}|{merkle_root}".encode("utf-8")
        ).hexdigest()

        return provenance_hash, merkle_root

    def _compute_merkle_root(self, hashes: List[str]) -> str:
        """Compute Merkle root from a list of leaf hashes."""
        if not hashes:
            return hashlib.sha256(b"empty").hexdigest()
        if len(hashes) == 1:
            return hashes[0]

        current_level = list(hashes)
        while len(current_level) > 1:
            next_level: List[str] = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                combined = hashlib.sha256(f"{left}{right}".encode("utf-8")).hexdigest()
                next_level.append(combined)
            current_level = next_level

        return current_level[0]

    # ==========================================================================
    # STAGE 10: SEAL
    # ==========================================================================

    def _stage_seal(
        self,
        aggregated: Dict[str, Any],
        provenance_hash: str,
        merkle_root: str,
    ) -> Dict[str, Any]:
        """
        Stage 10: Final validation and seal.

        Args:
            aggregated: Aggregated data.
            provenance_hash: Overall provenance hash.
            merkle_root: Merkle root.

        Returns:
            Sealed aggregated data.
        """
        total = aggregated.get("total_financed_emissions_kg_co2e", Decimal("0"))

        if total < 0:
            logger.error("SEAL: negative total financed emissions detected")
            aggregated["seal_status"] = "FAILED"
        else:
            aggregated["seal_status"] = "SEALED"

        aggregated["provenance_hash"] = provenance_hash
        aggregated["merkle_root"] = merkle_root
        aggregated["sealed_at"] = datetime.now(timezone.utc).isoformat()

        return aggregated

    # ==========================================================================
    # INTERNAL HELPERS
    # ==========================================================================

    def _elapsed_ms(self, start: datetime) -> float:
        """Calculate elapsed milliseconds since start."""
        delta = datetime.now(timezone.utc) - start
        return delta.total_seconds() * 1000.0

    def _record_provenance(
        self, chain_id: str, stage: PipelineStage, data: Any
    ) -> None:
        """Record a provenance entry for the chain."""
        chain = self._provenance_chains.get(chain_id, [])
        chain.append({
            "stage": stage.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data_summary": str(data)[:200] if data else None,
        })
        self._provenance_chains[chain_id] = chain


# ==============================================================================
# Module-Level Exports
# ==============================================================================

__all__ = [
    "InvestmentsPipelineEngine",
    "PortfolioInput",
    "PortfolioAggregationResult",
    "InvestmentHoldingInput",
    "InvestmentCalculationResult",
    "PipelineStatus",
    "PipelineStage",
    "AssetClass",
    "CalculationMethod",
]
