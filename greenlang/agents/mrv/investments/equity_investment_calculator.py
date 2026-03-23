# -*- coding: utf-8 -*-
"""
EquityInvestmentCalculatorEngine - Calculator for equity investment emissions.

This module implements the EquityInvestmentCalculatorEngine for AGENT-MRV-028
(Investments, GHG Protocol Scope 3 Category 15). It provides thread-safe
singleton calculations for listed equity, private equity, and unlisted equity
financed emissions using PCAF (Partnership for Carbon Accounting Financials)
methodology.

Calculation Methods (by data quality hierarchy):
    1. Reported emissions (PCAF Score 1-2): Verified/reported Scope 1+2
    2. Physical activity (PCAF Score 3): Physical activity data + EFs
    3. Revenue EEIO (PCAF Score 4): Revenue x sector EEIO factor
    4. Sector average (PCAF Score 5): Sector average x attribution

Listed Equity / Corporate Bond Attribution:
    EVIC = market_cap + total_debt
    attribution_factor = outstanding_amount / EVIC
    financed_emissions = attribution_factor x (company_scope1 + company_scope2)

Private Equity / Unlisted Equity Attribution:
    attribution_factor = outstanding_amount / (total_equity + total_debt)
    financed_emissions = attribution_factor x company_emissions

Carbon Intensity Metrics:
    WACI = SUM[(position_value / total_portfolio) x (company_emissions / company_revenue)]
    Financed intensity = financed_emissions / total_invested ($M)
    Revenue intensity = financed_emissions / attributed_revenue ($M)

Features:
    - PCAF Global GHG Standard 2022 compliant
    - 5 calculation methods based on data quality score
    - EVIC-based attribution for listed equity
    - Equity+debt attribution for private equity
    - WACI, financed intensity, revenue intensity metrics
    - Double-counting prevention (DC-INV-001)
    - Batch calculation support
    - All Decimal with ROUND_HALF_UP
    - Thread-safe singleton with threading.RLock()
    - Provenance tracking via SHA-256 hashes
    - Prometheus metrics via gl_inv_ prefix

Example:
    >>> from greenlang.agents.mrv.investments.equity_investment_calculator import (
    ...     EquityInvestmentCalculatorEngine,
    ... )
    >>> engine = EquityInvestmentCalculatorEngine()
    >>> result = engine.calculate_listed_equity(
    ...     EquityInvestmentInput(
    ...         investment_id="INV-001",
    ...         asset_class="listed_equity",
    ...         outstanding_amount=Decimal("5000000"),
    ...         market_cap=Decimal("500000000"),
    ...         total_debt=Decimal("200000000"),
    ...         company_scope1=Decimal("50000"),
    ...         company_scope2=Decimal("30000"),
    ...         company_revenue=Decimal("1000000000"),
    ...         sector="energy",
    ...     )
    ... )
    >>> result.financed_emissions > 0
    True

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-015
"""

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

AGENT_ID: str = "GL-MRV-S3-015"
AGENT_COMPONENT: str = "AGENT-MRV-028"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_inv_"

# Quantization constants
_QUANT_8DP = Decimal("0.00000001")
_QUANT_2DP = Decimal("0.01")
_ZERO = Decimal("0")
_ONE = Decimal("1")
_MILLION = Decimal("1000000")


# =============================================================================
# ENUMERATIONS
# =============================================================================


class EquityType(str, Enum):
    """Equity investment sub-types."""

    LISTED_EQUITY = "listed_equity"
    PRIVATE_EQUITY = "private_equity"
    UNLISTED_EQUITY = "unlisted_equity"
    CORPORATE_BOND = "corporate_bond"


class CalculationMethod(str, Enum):
    """Calculation method determined by PCAF data quality score."""

    REPORTED = "reported"  # Score 1-2: verified/reported emissions
    PHYSICAL = "physical"  # Score 3: physical activity data + EFs
    EEIO = "eeio"  # Score 4: revenue x EEIO factor
    SECTOR_AVERAGE = "sector_average"  # Score 5: sector average


class PCAFDataQuality(str, Enum):
    """PCAF data quality scores."""

    SCORE_1 = "1"  # Verified emissions
    SCORE_2 = "2"  # Reported unverified
    SCORE_3 = "3"  # Physical activity
    SCORE_4 = "4"  # Revenue EEIO
    SCORE_5 = "5"  # Sector average


# =============================================================================
# INPUT / OUTPUT DATA MODELS
# =============================================================================


@dataclass
class EquityInvestmentInput:
    """
    Input data model for equity investment emissions calculation.

    All monetary values are in USD. All emission values are in tCO2e.

    Attributes:
        investment_id: Unique identifier for this investment position.
        asset_class: PCAF asset class ("listed_equity", "private_equity",
            "unlisted_equity", "corporate_bond").
        outstanding_amount: Outstanding investment amount in USD.
        market_cap: Investee market capitalization in USD (listed only).
        total_debt: Investee total debt in USD.
        total_equity: Investee total equity in USD (private/unlisted only).
        company_scope1: Investee Scope 1 emissions in tCO2e (Score 1-2).
        company_scope2: Investee Scope 2 emissions in tCO2e (Score 1-2).
        company_scope3: Optional investee Scope 3 emissions in tCO2e.
        company_revenue: Investee annual revenue in USD.
        sector: GICS sector key for emission factor lookup.
        pcaf_score: PCAF data quality score (1-5). Auto-determined if None.
        energy_consumption_kwh: Physical energy use in kWh (Score 3).
        grid_country: Country code for grid EF (Score 3).
        grid_region: eGRID subregion code (Score 3, optional).
        is_verified: Whether emissions data is third-party verified.
        is_consolidated: Whether investee is a consolidated subsidiary.
        equity_share_pct: Equity ownership percentage (for DC check).
        reporting_year: Reporting year for temporal alignment.
        currency: Original currency if not USD.
        investee_name: Name of the investee company (optional).
        include_scope3: Whether to include Scope 3 in financed emissions.
    """

    investment_id: str = ""
    asset_class: str = "listed_equity"
    outstanding_amount: Decimal = _ZERO
    market_cap: Optional[Decimal] = None
    total_debt: Optional[Decimal] = None
    total_equity: Optional[Decimal] = None
    company_scope1: Optional[Decimal] = None
    company_scope2: Optional[Decimal] = None
    company_scope3: Optional[Decimal] = None
    company_revenue: Optional[Decimal] = None
    sector: str = "other"
    pcaf_score: Optional[int] = None
    energy_consumption_kwh: Optional[Decimal] = None
    grid_country: Optional[str] = None
    grid_region: Optional[str] = None
    is_verified: bool = False
    is_consolidated: bool = False
    equity_share_pct: Optional[Decimal] = None
    reporting_year: int = 2024
    currency: str = "USD"
    investee_name: Optional[str] = None
    include_scope3: bool = False


@dataclass
class CarbonIntensityResult:
    """
    Carbon intensity metrics for an equity investment.

    Attributes:
        waci_contribution: Contribution to portfolio WACI (tCO2e/$M).
        financed_intensity: Financed emissions per $M invested (tCO2e/$M).
        revenue_intensity: Financed emissions per $M attributed revenue.
        portfolio_weight: Position weight in portfolio (0-1).
        company_carbon_intensity: Company-level emissions/revenue ratio.
    """

    waci_contribution: Decimal = _ZERO
    financed_intensity: Decimal = _ZERO
    revenue_intensity: Decimal = _ZERO
    portfolio_weight: Decimal = _ZERO
    company_carbon_intensity: Decimal = _ZERO


@dataclass
class InvestmentCalculationResult:
    """
    Output data model for equity investment emissions calculation.

    All emission values are in tCO2e. All monetary values are in USD.

    Attributes:
        investment_id: Input investment identifier.
        asset_class: PCAF asset class used.
        calculation_method: Method used (reported/physical/eeio/sector_average).
        evic: Enterprise Value Including Cash (listed only).
        attribution_factor: PCAF attribution factor (0-1).
        company_emissions: Total company emissions attributed to (tCO2e).
        financed_emissions: Attributed financed emissions (tCO2e).
        financed_scope1: Attributed Scope 1 financed emissions (tCO2e).
        financed_scope2: Attributed Scope 2 financed emissions (tCO2e).
        financed_scope3: Attributed Scope 3 financed emissions (tCO2e).
        pcaf_score: PCAF data quality score (1-5).
        uncertainty_pct: Uncertainty percentage based on PCAF score.
        lower_bound: Lower uncertainty bound (tCO2e).
        upper_bound: Upper uncertainty bound (tCO2e).
        carbon_intensity: Carbon intensity metrics.
        dc_check_passed: Double-counting check result.
        dc_flags: List of DC warning/error messages.
        provenance_hash: SHA-256 provenance hash.
        processing_time_ms: Calculation duration in milliseconds.
        timestamp: ISO 8601 UTC timestamp.
        errors: List of validation or calculation errors.
        warnings: List of warnings generated during calculation.
        metadata: Additional metadata for audit trail.
    """

    investment_id: str = ""
    asset_class: str = ""
    calculation_method: str = ""
    evic: Decimal = _ZERO
    attribution_factor: Decimal = _ZERO
    company_emissions: Decimal = _ZERO
    financed_emissions: Decimal = _ZERO
    financed_scope1: Decimal = _ZERO
    financed_scope2: Decimal = _ZERO
    financed_scope3: Decimal = _ZERO
    pcaf_score: int = 5
    uncertainty_pct: Decimal = _ZERO
    lower_bound: Decimal = _ZERO
    upper_bound: Decimal = _ZERO
    carbon_intensity: CarbonIntensityResult = field(
        default_factory=CarbonIntensityResult
    )
    dc_check_passed: bool = True
    dc_flags: List[str] = field(default_factory=list)
    provenance_hash: str = ""
    processing_time_ms: float = 0.0
    timestamp: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "investment_id": self.investment_id,
            "asset_class": self.asset_class,
            "calculation_method": self.calculation_method,
            "evic": str(self.evic),
            "attribution_factor": str(self.attribution_factor),
            "company_emissions": str(self.company_emissions),
            "financed_emissions": str(self.financed_emissions),
            "financed_scope1": str(self.financed_scope1),
            "financed_scope2": str(self.financed_scope2),
            "financed_scope3": str(self.financed_scope3),
            "pcaf_score": self.pcaf_score,
            "uncertainty_pct": str(self.uncertainty_pct),
            "lower_bound": str(self.lower_bound),
            "upper_bound": str(self.upper_bound),
            "carbon_intensity": {
                "waci_contribution": str(self.carbon_intensity.waci_contribution),
                "financed_intensity": str(self.carbon_intensity.financed_intensity),
                "revenue_intensity": str(self.carbon_intensity.revenue_intensity),
                "portfolio_weight": str(self.carbon_intensity.portfolio_weight),
                "company_carbon_intensity": str(
                    self.carbon_intensity.company_carbon_intensity
                ),
            },
            "dc_check_passed": self.dc_check_passed,
            "dc_flags": self.dc_flags,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


# =============================================================================
# HASH UTILITIES
# =============================================================================


def _serialize_for_hash(obj: Any) -> str:
    """
    Serialize object to deterministic JSON for hashing.

    Args:
        obj: Object to serialize.

    Returns:
        Deterministic JSON string.
    """

    def default_handler(o: Any) -> Any:
        if isinstance(o, Decimal):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, Enum):
            return o.value
        if hasattr(o, "to_dict"):
            return o.to_dict()
        if hasattr(o, "__dict__"):
            return o.__dict__
        return str(o)

    return json.dumps(obj, sort_keys=True, default=default_handler)


def _compute_hash(data: Any) -> str:
    """
    Compute SHA-256 hash of data.

    Args:
        data: Data to hash.

    Returns:
        Lowercase hex SHA-256 hash.
    """
    serialized = _serialize_for_hash(data)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# =============================================================================
# METRICS / PROVENANCE LAZY LOADERS
# =============================================================================


def _get_metrics_collector() -> Any:
    """Lazy-import metrics collector to avoid circular imports."""
    try:
        from greenlang.agents.mrv.investments.metrics import get_metrics_collector
        return get_metrics_collector()
    except (ImportError, Exception) as exc:
        logger.debug("Metrics collector unavailable: %s", exc)
        return None


def _get_provenance_manager() -> Any:
    """Lazy-import provenance manager to avoid circular imports."""
    try:
        from greenlang.agents.mrv.investments.provenance import get_provenance_manager
        return get_provenance_manager()
    except (ImportError, Exception) as exc:
        logger.debug("Provenance manager unavailable: %s", exc)
        return None


def _get_database_engine() -> Any:
    """Lazy-import database engine to avoid circular imports."""
    try:
        from greenlang.agents.mrv.investments.investment_database import get_database_engine
        return get_database_engine()
    except (ImportError, Exception) as exc:
        logger.debug("Database engine unavailable: %s", exc)
        return None


# =============================================================================
# ENGINE CLASS
# =============================================================================


class EquityInvestmentCalculatorEngine:
    """
    Thread-safe singleton calculator for equity investment financed emissions.

    Implements the PCAF Global GHG Accounting Standard methodology for
    listed equity, private equity, unlisted equity, and corporate bond
    investments. All arithmetic uses Python Decimal with ROUND_HALF_UP
    quantization for regulatory precision.

    This engine is ZERO-HALLUCINATION: all calculations are deterministic
    Python arithmetic. No LLM calls are used for any numeric computation.

    Thread Safety:
        Uses __new__ singleton pattern with threading.Lock. The
        _calculation_count attribute is protected by a dedicated lock.

    Attributes:
        _database_engine: Lazy-loaded InvestmentDatabaseEngine
        _calculation_count: Total number of calculations performed

    Example:
        >>> engine = EquityInvestmentCalculatorEngine()
        >>> result = engine.calculate_listed_equity(input_data)
        >>> assert result.provenance_hash  # SHA-256 present
    """

    _instance: Optional["EquityInvestmentCalculatorEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "EquityInvestmentCalculatorEngine":
        """Thread-safe singleton instantiation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the equity calculator engine (only once due to singleton)."""
        if hasattr(self, "_initialized"):
            return

        self._initialized: bool = True
        self._database_engine: Any = None
        self._calculation_count: int = 0
        self._count_lock: threading.RLock = threading.RLock()

        logger.info(
            "EquityInvestmentCalculatorEngine initialized: agent=%s, version=%s",
            AGENT_ID,
            VERSION,
        )

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _get_db(self) -> Any:
        """
        Lazy-load the database engine singleton.

        Returns:
            InvestmentDatabaseEngine instance.
        """
        if self._database_engine is None:
            self._database_engine = _get_database_engine()
        return self._database_engine

    def _increment_count(self) -> int:
        """
        Increment and return the calculation counter thread-safely.

        Returns:
            Updated calculation count.
        """
        with self._count_lock:
            self._calculation_count += 1
            return self._calculation_count

    def _quantize(self, value: Decimal, precision: Decimal = _QUANT_8DP) -> Decimal:
        """
        Quantize a Decimal value with ROUND_HALF_UP.

        Args:
            value: Decimal value to quantize.
            precision: Quantization precision.

        Returns:
            Quantized Decimal value.
        """
        try:
            return value.quantize(precision, rounding=ROUND_HALF_UP)
        except (InvalidOperation, OverflowError):
            logger.warning("Failed to quantize value %s, returning zero", value)
            return _ZERO

    def _safe_decimal(self, value: Optional[Decimal]) -> Decimal:
        """
        Safely convert Optional[Decimal] to Decimal, defaulting to zero.

        Args:
            value: Optional Decimal value.

        Returns:
            Decimal value or zero.
        """
        if value is None:
            return _ZERO
        try:
            return Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError):
            return _ZERO

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def _validate_equity_input(
        self, input_data: EquityInvestmentInput
    ) -> List[str]:
        """
        Validate input data for equity investment calculation.

        Checks:
        - outstanding_amount > 0
        - asset_class is valid
        - Sufficient data for at least one calculation method
        - No negative emissions values

        Args:
            input_data: EquityInvestmentInput to validate.

        Returns:
            List of error messages (empty if valid).
        """
        errors: List[str] = []

        # Outstanding amount must be positive
        if input_data.outstanding_amount <= _ZERO:
            errors.append(
                f"outstanding_amount must be > 0, got {input_data.outstanding_amount}"
            )

        # Valid asset classes
        valid_classes = {
            "listed_equity", "private_equity", "unlisted_equity", "corporate_bond"
        }
        if input_data.asset_class not in valid_classes:
            errors.append(
                f"Invalid asset_class '{input_data.asset_class}'. "
                f"Must be one of: {sorted(valid_classes)}"
            )

        # For listed equity / corporate bond, need market_cap or total_debt
        if input_data.asset_class in ("listed_equity", "corporate_bond"):
            if (
                input_data.market_cap is None or input_data.market_cap <= _ZERO
            ) and (
                input_data.total_debt is None or input_data.total_debt <= _ZERO
            ):
                errors.append(
                    "Listed equity/corporate bond requires market_cap or "
                    "total_debt > 0 for EVIC calculation"
                )

        # For private equity / unlisted, need equity or debt
        if input_data.asset_class in ("private_equity", "unlisted_equity"):
            total_eq = self._safe_decimal(input_data.total_equity)
            total_dt = self._safe_decimal(input_data.total_debt)
            if (total_eq + total_dt) <= _ZERO:
                errors.append(
                    "Private/unlisted equity requires total_equity + "
                    "total_debt > 0 for attribution"
                )

        # No negative emissions
        for field_name in ("company_scope1", "company_scope2", "company_scope3"):
            val = getattr(input_data, field_name, None)
            if val is not None and val < _ZERO:
                errors.append(f"{field_name} cannot be negative, got {val}")

        # Validate PCAF score range
        if input_data.pcaf_score is not None:
            if input_data.pcaf_score < 1 or input_data.pcaf_score > 5:
                errors.append(
                    f"pcaf_score must be 1-5, got {input_data.pcaf_score}"
                )

        # Validate sector
        if input_data.sector:
            db = self._get_db()
            if db is not None:
                try:
                    db.get_sector_ef(input_data.sector)
                except ValueError:
                    errors.append(
                        f"Unknown sector '{input_data.sector}'"
                    )

        return errors

    # =========================================================================
    # PCAF DATA QUALITY DETERMINATION
    # =========================================================================

    def _determine_pcaf_quality(
        self, input_data: EquityInvestmentInput
    ) -> Tuple[int, CalculationMethod]:
        """
        Determine PCAF data quality score and calculation method.

        Follows the PCAF hierarchy:
        - Score 1: Verified Scope 1+2 emissions available
        - Score 2: Unverified reported Scope 1+2 available
        - Score 3: Physical activity data (energy use) available
        - Score 4: Revenue data available (for EEIO)
        - Score 5: Only sector information available

        If pcaf_score is explicitly set on input, use that instead.

        Args:
            input_data: Input data to assess.

        Returns:
            Tuple of (pcaf_score, calculation_method).
        """
        # If explicitly set, use it
        if input_data.pcaf_score is not None:
            score = input_data.pcaf_score
            if score <= 2:
                return score, CalculationMethod.REPORTED
            elif score == 3:
                return score, CalculationMethod.PHYSICAL
            elif score == 4:
                return score, CalculationMethod.EEIO
            else:
                return score, CalculationMethod.SECTOR_AVERAGE

        # Auto-determine based on available data
        scope1 = self._safe_decimal(input_data.company_scope1)
        scope2 = self._safe_decimal(input_data.company_scope2)
        has_emissions = (scope1 + scope2) > _ZERO

        if has_emissions and input_data.is_verified:
            return 1, CalculationMethod.REPORTED

        if has_emissions and not input_data.is_verified:
            return 2, CalculationMethod.REPORTED

        energy = self._safe_decimal(input_data.energy_consumption_kwh)
        if energy > _ZERO and input_data.grid_country:
            return 3, CalculationMethod.PHYSICAL

        revenue = self._safe_decimal(input_data.company_revenue)
        if revenue > _ZERO and input_data.sector:
            return 4, CalculationMethod.EEIO

        return 5, CalculationMethod.SECTOR_AVERAGE

    # =========================================================================
    # EVIC AND ATTRIBUTION FACTOR
    # =========================================================================

    def _calculate_evic(
        self,
        market_cap: Optional[Decimal],
        total_debt: Optional[Decimal],
    ) -> Decimal:
        """
        Calculate Enterprise Value Including Cash (EVIC).

        EVIC = market_cap + total_debt
        Per PCAF, EVIC is used as the denominator for listed equity
        and corporate bond attribution.

        Args:
            market_cap: Market capitalization in USD.
            total_debt: Total debt in USD.

        Returns:
            EVIC as Decimal.
        """
        mc = self._safe_decimal(market_cap)
        td = self._safe_decimal(total_debt)
        evic = mc + td

        logger.debug(
            "EVIC calculation: market_cap=%s + total_debt=%s = EVIC=%s",
            mc, td, evic,
        )

        return self._quantize(evic, _QUANT_2DP)

    def _calculate_attribution_factor(
        self,
        outstanding: Decimal,
        denominator: Decimal,
    ) -> Decimal:
        """
        Calculate PCAF attribution factor.

        AF = outstanding_amount / denominator
        Capped at 1.0 (cannot attribute more than 100% of emissions).

        Args:
            outstanding: Outstanding investment amount.
            denominator: EVIC or (total_equity + total_debt).

        Returns:
            Attribution factor as Decimal (0-1).
        """
        if denominator <= _ZERO:
            logger.warning(
                "Attribution factor denominator <= 0 (%s), returning 0",
                denominator,
            )
            return _ZERO

        af = outstanding / denominator

        # Cap at 1.0
        if af > _ONE:
            logger.warning(
                "Attribution factor %s > 1.0, capping at 1.0 "
                "(outstanding=%s, denominator=%s)",
                af, outstanding, denominator,
            )
            af = _ONE

        result = self._quantize(af)

        logger.debug(
            "Attribution factor: outstanding=%s / denominator=%s = AF=%s",
            outstanding, denominator, result,
        )

        return result

    # =========================================================================
    # CALCULATION METHODS BY DATA QUALITY
    # =========================================================================

    def _calculate_by_reported(
        self,
        input_data: EquityInvestmentInput,
        af: Decimal,
    ) -> Tuple[Decimal, Decimal, Decimal, Decimal]:
        """
        Calculate financed emissions using reported emissions (Score 1-2).

        financed_scope1 = AF x company_scope1
        financed_scope2 = AF x company_scope2
        financed_scope3 = AF x company_scope3 (if included)

        Args:
            input_data: Input with company emissions data.
            af: Attribution factor.

        Returns:
            Tuple of (total, scope1, scope2, scope3) financed emissions.
        """
        scope1 = self._safe_decimal(input_data.company_scope1)
        scope2 = self._safe_decimal(input_data.company_scope2)
        scope3 = _ZERO
        if input_data.include_scope3:
            scope3 = self._safe_decimal(input_data.company_scope3)

        fin_s1 = self._quantize(af * scope1)
        fin_s2 = self._quantize(af * scope2)
        fin_s3 = self._quantize(af * scope3)
        total = self._quantize(fin_s1 + fin_s2 + fin_s3)

        logger.debug(
            "Reported emissions calc: AF=%s x (S1=%s + S2=%s + S3=%s) = "
            "fin_S1=%s + fin_S2=%s + fin_S3=%s = total=%s tCO2e",
            af, scope1, scope2, scope3, fin_s1, fin_s2, fin_s3, total,
        )

        return total, fin_s1, fin_s2, fin_s3

    def _calculate_by_physical(
        self,
        input_data: EquityInvestmentInput,
        af: Decimal,
    ) -> Tuple[Decimal, Decimal, Decimal, Decimal]:
        """
        Calculate financed emissions using physical activity data (Score 3).

        company_emissions = energy_consumption_kwh x grid_ef / 1000 (to tCO2e)
        financed_emissions = AF x company_emissions

        Args:
            input_data: Input with energy consumption data.
            af: Attribution factor.

        Returns:
            Tuple of (total, scope1, scope2, scope3) financed emissions.
        """
        energy_kwh = self._safe_decimal(input_data.energy_consumption_kwh)
        grid_ef = _ZERO

        db = self._get_db()
        if db is not None and input_data.grid_country:
            try:
                grid_ef = db.get_grid_ef(
                    input_data.grid_country,
                    input_data.grid_region,
                )
            except (ValueError, Exception) as exc:
                logger.warning("Grid EF lookup failed: %s", exc)

        # kgCO2e = kWh x kgCO2e/kWh, then /1000 for tCO2e
        company_emissions_t = self._quantize(
            energy_kwh * grid_ef / Decimal("1000")
        )

        # Physical activity data is assumed to represent Scope 2
        fin_s2 = self._quantize(af * company_emissions_t)

        logger.debug(
            "Physical activity calc: energy=%s kWh x grid_ef=%s = "
            "company=%s tCO2e, AF=%s -> financed=%s tCO2e",
            energy_kwh, grid_ef, company_emissions_t, af, fin_s2,
        )

        return fin_s2, _ZERO, fin_s2, _ZERO

    def _calculate_by_eeio(
        self,
        input_data: EquityInvestmentInput,
        af: Decimal,
    ) -> Tuple[Decimal, Decimal, Decimal, Decimal]:
        """
        Calculate financed emissions using revenue EEIO method (Score 4).

        company_emissions = revenue_usd x eeio_factor / 1000 (to tCO2e)
        financed_emissions = AF x company_emissions

        Args:
            input_data: Input with revenue data.
            af: Attribution factor.

        Returns:
            Tuple of (total, scope1, scope2, scope3) financed emissions.
        """
        revenue = self._safe_decimal(input_data.company_revenue)

        # Convert currency if needed
        if input_data.currency != "USD":
            db = self._get_db()
            if db is not None:
                try:
                    rate = db.get_currency_rate(input_data.currency)
                    revenue = self._quantize(revenue * rate, _QUANT_2DP)
                except (ValueError, Exception) as exc:
                    logger.warning("Currency conversion failed: %s", exc)

        # Get EEIO factor (kgCO2e per USD)
        eeio_factor = _ZERO
        db = self._get_db()
        if db is not None:
            try:
                eeio_factor = db.get_eeio_factor(input_data.sector)
            except (ValueError, Exception) as exc:
                logger.warning("EEIO factor lookup failed: %s", exc)

        # company_emissions (tCO2e) = revenue ($) x eeio (kgCO2e/$) / 1000
        company_emissions_t = self._quantize(
            revenue * eeio_factor / Decimal("1000")
        )

        # EEIO gives total emissions; split using sector defaults
        fin_total = self._quantize(af * company_emissions_t)

        # Use sector split if available
        scope1_pct = Decimal("0.50")
        scope2_pct = Decimal("0.50")
        db = self._get_db()
        if db is not None:
            try:
                sector_detail = db.get_sector_ef_detail(input_data.sector)
                scope1_pct = sector_detail.get("typical_scope1_pct", Decimal("0.50"))
                scope2_pct = sector_detail.get("typical_scope2_pct", Decimal("0.50"))
            except (ValueError, Exception):
                pass

        fin_s1 = self._quantize(fin_total * scope1_pct)
        fin_s2 = self._quantize(fin_total * scope2_pct)

        logger.debug(
            "EEIO calc: revenue=%s x eeio=%s = company=%s tCO2e, "
            "AF=%s -> financed=%s (S1=%s, S2=%s) tCO2e",
            revenue, eeio_factor, company_emissions_t,
            af, fin_total, fin_s1, fin_s2,
        )

        return fin_total, fin_s1, fin_s2, _ZERO

    def _calculate_by_sector_avg(
        self,
        input_data: EquityInvestmentInput,
        af: Decimal,
    ) -> Tuple[Decimal, Decimal, Decimal, Decimal]:
        """
        Calculate financed emissions using sector average method (Score 5).

        Uses sector average tCO2e per $M revenue applied to outstanding
        amount to estimate company-level emissions, then applies AF.

        company_emissions = outstanding_amount_$M x sector_ef (tCO2e/$M)
        financed_emissions = AF x company_emissions

        Args:
            input_data: Input with sector information.
            af: Attribution factor.

        Returns:
            Tuple of (total, scope1, scope2, scope3) financed emissions.
        """
        sector_ef = Decimal("150")  # Default "other"
        db = self._get_db()
        if db is not None:
            try:
                sector_ef = db.get_sector_ef(input_data.sector)
            except (ValueError, Exception) as exc:
                logger.warning("Sector EF lookup failed: %s", exc)

        # Convert outstanding to $M
        outstanding_m = self._quantize(
            input_data.outstanding_amount / _MILLION, _QUANT_8DP
        )

        # Estimate company-level emissions from outstanding amount
        # This is a rough proxy when revenue is unavailable
        company_emissions_t = self._quantize(outstanding_m * sector_ef)
        fin_total = self._quantize(af * company_emissions_t)

        # Use sector split
        scope1_pct = Decimal("0.50")
        scope2_pct = Decimal("0.50")
        if db is not None:
            try:
                detail = db.get_sector_ef_detail(input_data.sector)
                scope1_pct = detail.get("typical_scope1_pct", Decimal("0.50"))
                scope2_pct = detail.get("typical_scope2_pct", Decimal("0.50"))
            except (ValueError, Exception):
                pass

        fin_s1 = self._quantize(fin_total * scope1_pct)
        fin_s2 = self._quantize(fin_total * scope2_pct)

        logger.debug(
            "Sector avg calc: outstanding_$M=%s x sector_ef=%s = "
            "company=%s tCO2e, AF=%s -> financed=%s tCO2e",
            outstanding_m, sector_ef, company_emissions_t, af, fin_total,
        )

        return fin_total, fin_s1, fin_s2, _ZERO

    # =========================================================================
    # CARBON INTENSITY METRICS
    # =========================================================================

    def _calculate_carbon_intensity(
        self,
        financed_emissions: Decimal,
        investment_value: Decimal,
        company_revenue: Optional[Decimal],
        company_emissions: Decimal,
        af: Decimal,
    ) -> CarbonIntensityResult:
        """
        Calculate carbon intensity metrics.

        - WACI contribution: (company_emissions / company_revenue)
          weighted by portfolio weight (computed externally)
        - Financed intensity: financed_emissions / invested_amount_$M
        - Revenue intensity: financed_emissions / attributed_revenue_$M
        - Company carbon intensity: company_emissions / company_revenue_$M

        Args:
            financed_emissions: Attributed financed emissions (tCO2e).
            investment_value: Position value in USD.
            company_revenue: Company annual revenue in USD.
            company_emissions: Total company emissions (tCO2e).
            af: Attribution factor.

        Returns:
            CarbonIntensityResult with all intensity metrics.
        """
        result = CarbonIntensityResult()

        # Financed intensity (tCO2e per $M invested)
        if investment_value > _ZERO:
            invested_m = investment_value / _MILLION
            if invested_m > _ZERO:
                result.financed_intensity = self._quantize(
                    financed_emissions / invested_m
                )

        # Revenue intensity and WACI contribution
        revenue = self._safe_decimal(company_revenue)
        if revenue > _ZERO:
            revenue_m = revenue / _MILLION

            # Company carbon intensity (tCO2e per $M revenue)
            if revenue_m > _ZERO:
                result.company_carbon_intensity = self._quantize(
                    company_emissions / revenue_m
                )

            # Revenue intensity (financed emissions per $M attributed revenue)
            attributed_revenue = self._quantize(af * revenue, _QUANT_2DP)
            attributed_revenue_m = attributed_revenue / _MILLION
            if attributed_revenue_m > _ZERO:
                result.revenue_intensity = self._quantize(
                    financed_emissions / attributed_revenue_m
                )

            # WACI contribution (will be weighted by portfolio weight externally)
            result.waci_contribution = result.company_carbon_intensity

        logger.debug(
            "Carbon intensity: financed=%s tCO2e/$M, revenue=%s tCO2e/$M, "
            "company=%s tCO2e/$M revenue",
            result.financed_intensity,
            result.revenue_intensity,
            result.company_carbon_intensity,
        )

        return result

    # =========================================================================
    # DOUBLE-COUNTING CHECK
    # =========================================================================

    def _check_consolidation_boundary(
        self, input_data: EquityInvestmentInput
    ) -> Tuple[bool, List[str]]:
        """
        Check for double-counting with Scope 1/2 (DC-INV-001).

        If the investee is a consolidated subsidiary (equity_share >= 50%
        or is_consolidated flag), its emissions should already be in the
        reporting entity's Scope 1/2 and must NOT be counted in Cat 15.

        Args:
            input_data: Investment input data.

        Returns:
            Tuple of (passed, list_of_flags).
            passed=True means no DC issue; passed=False means exclude.
        """
        flags: List[str] = []

        # Check explicit consolidation flag
        if input_data.is_consolidated:
            flags.append(
                "DC-INV-001: Investee is flagged as consolidated subsidiary. "
                "Its emissions are in Scope 1/2 -- EXCLUDE from Cat 15."
            )
            return False, flags

        # Check equity ownership percentage
        eq_pct = self._safe_decimal(input_data.equity_share_pct)
        if eq_pct >= Decimal("50"):
            flags.append(
                f"DC-INV-001: Equity ownership {eq_pct}% >= 50%. "
                f"Investee likely consolidated -- EXCLUDE from Cat 15."
            )
            return False, flags

        # Check for equity > 20% (associate -- flag warning)
        if eq_pct >= Decimal("20"):
            flags.append(
                f"DC-INV-002: Equity ownership {eq_pct}% >= 20%. "
                f"Investee may be an associate -- verify not double-counted."
            )
            # Warning only, do not exclude
            return True, flags

        return True, flags

    # =========================================================================
    # UNCERTAINTY BOUNDS
    # =========================================================================

    def _calculate_uncertainty(
        self,
        financed_emissions: Decimal,
        pcaf_score: int,
    ) -> Tuple[Decimal, Decimal, Decimal]:
        """
        Calculate uncertainty bounds based on PCAF score.

        Args:
            financed_emissions: Central estimate of financed emissions.
            pcaf_score: PCAF data quality score (1-5).

        Returns:
            Tuple of (uncertainty_pct, lower_bound, upper_bound).
        """
        db = self._get_db()
        uncertainty_pct = Decimal("60")  # Default to worst case

        if db is not None:
            try:
                unc_data = db.get_uncertainty_range(pcaf_score)
                uncertainty_pct = unc_data["uncertainty_pct"]
                lower_mult = unc_data["lower_bound_multiplier"]
                upper_mult = unc_data["upper_bound_multiplier"]
                lower = self._quantize(financed_emissions * lower_mult)
                upper = self._quantize(financed_emissions * upper_mult)
                return uncertainty_pct, lower, upper
            except (ValueError, Exception) as exc:
                logger.warning("Uncertainty lookup failed: %s", exc)

        # Fallback manual calculation
        pct_decimal = uncertainty_pct / Decimal("100")
        lower = self._quantize(financed_emissions * (_ONE - pct_decimal))
        upper = self._quantize(financed_emissions * (_ONE + pct_decimal))

        return uncertainty_pct, lower, upper

    # =========================================================================
    # PROVENANCE
    # =========================================================================

    def _compute_provenance_hash(
        self,
        input_data: EquityInvestmentInput,
        result: InvestmentCalculationResult,
    ) -> str:
        """
        Compute SHA-256 provenance hash for the calculation.

        Hashes the input data and key output fields to create an
        immutable audit trail record.

        Args:
            input_data: Calculation input.
            result: Calculation output.

        Returns:
            SHA-256 hex hash string.
        """
        provenance_data = {
            "agent_id": AGENT_ID,
            "version": VERSION,
            "investment_id": input_data.investment_id,
            "asset_class": input_data.asset_class,
            "outstanding_amount": str(input_data.outstanding_amount),
            "calculation_method": result.calculation_method,
            "attribution_factor": str(result.attribution_factor),
            "financed_emissions": str(result.financed_emissions),
            "pcaf_score": result.pcaf_score,
            "timestamp": result.timestamp,
        }

        return _compute_hash(provenance_data)

    # =========================================================================
    # CORE CALCULATION DISPATCH
    # =========================================================================

    def _execute_calculation(
        self,
        input_data: EquityInvestmentInput,
        af: Decimal,
        evic: Decimal,
        pcaf_score: int,
        calc_method: CalculationMethod,
    ) -> InvestmentCalculationResult:
        """
        Execute the calculation based on method and build result.

        Args:
            input_data: Validated input data.
            af: Attribution factor.
            evic: EVIC value (0 for private equity).
            pcaf_score: PCAF quality score.
            calc_method: Calculation method to use.

        Returns:
            InvestmentCalculationResult with all fields populated.
        """
        result = InvestmentCalculationResult(
            investment_id=input_data.investment_id,
            asset_class=input_data.asset_class,
            calculation_method=calc_method.value,
            evic=evic,
            attribution_factor=af,
            pcaf_score=pcaf_score,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Dispatch to appropriate calculation method
        if calc_method == CalculationMethod.REPORTED:
            total, s1, s2, s3 = self._calculate_by_reported(input_data, af)
        elif calc_method == CalculationMethod.PHYSICAL:
            total, s1, s2, s3 = self._calculate_by_physical(input_data, af)
        elif calc_method == CalculationMethod.EEIO:
            total, s1, s2, s3 = self._calculate_by_eeio(input_data, af)
        else:
            total, s1, s2, s3 = self._calculate_by_sector_avg(input_data, af)

        result.financed_emissions = total
        result.financed_scope1 = s1
        result.financed_scope2 = s2
        result.financed_scope3 = s3

        # Company emissions (pre-attribution)
        scope1 = self._safe_decimal(input_data.company_scope1)
        scope2 = self._safe_decimal(input_data.company_scope2)
        result.company_emissions = scope1 + scope2

        # Uncertainty bounds
        unc_pct, lower, upper = self._calculate_uncertainty(total, pcaf_score)
        result.uncertainty_pct = unc_pct
        result.lower_bound = lower
        result.upper_bound = upper

        # Carbon intensity
        result.carbon_intensity = self._calculate_carbon_intensity(
            financed_emissions=total,
            investment_value=input_data.outstanding_amount,
            company_revenue=input_data.company_revenue,
            company_emissions=result.company_emissions,
            af=af,
        )

        # Metadata
        result.metadata = {
            "agent_id": AGENT_ID,
            "version": VERSION,
            "sector": input_data.sector,
            "reporting_year": input_data.reporting_year,
            "currency": input_data.currency,
            "investee_name": input_data.investee_name,
        }

        return result

    # =========================================================================
    # PUBLIC: CALCULATE (GENERIC)
    # =========================================================================

    def calculate(
        self, input_data: EquityInvestmentInput
    ) -> InvestmentCalculationResult:
        """
        Calculate financed emissions for an equity investment.

        Dispatches to the appropriate method based on asset_class:
        - listed_equity, corporate_bond -> calculate_listed_equity
        - private_equity, unlisted_equity -> calculate_private_equity

        Args:
            input_data: EquityInvestmentInput with investment details.

        Returns:
            InvestmentCalculationResult with financed emissions.

        Example:
            >>> engine = EquityInvestmentCalculatorEngine()
            >>> result = engine.calculate(input_data)
            >>> result.financed_emissions > 0
            True
        """
        ac = input_data.asset_class.lower().strip()

        if ac in ("listed_equity", "corporate_bond"):
            return self.calculate_listed_equity(input_data)
        elif ac in ("private_equity", "unlisted_equity"):
            return self.calculate_private_equity(input_data)
        else:
            # Unknown asset class -- try listed equity as default
            logger.warning(
                "Unknown equity asset class '%s', attempting listed equity calc",
                ac,
            )
            return self.calculate_listed_equity(input_data)

    # =========================================================================
    # PUBLIC: LISTED EQUITY
    # =========================================================================

    def calculate_listed_equity(
        self, input_data: EquityInvestmentInput
    ) -> InvestmentCalculationResult:
        """
        Calculate financed emissions for listed equity / corporate bond.

        EVIC = market_cap + total_debt
        AF = outstanding_amount / EVIC
        financed_emissions = AF x company_emissions

        Args:
            input_data: EquityInvestmentInput with listed equity data.

        Returns:
            InvestmentCalculationResult.

        Example:
            >>> engine = EquityInvestmentCalculatorEngine()
            >>> result = engine.calculate_listed_equity(input_data)
        """
        start_time = time.monotonic()
        self._increment_count()

        # Validate
        errors = self._validate_equity_input(input_data)

        # DC check
        dc_passed, dc_flags = self._check_consolidation_boundary(input_data)

        if not dc_passed:
            result = InvestmentCalculationResult(
                investment_id=input_data.investment_id,
                asset_class=input_data.asset_class,
                dc_check_passed=False,
                dc_flags=dc_flags,
                errors=errors + [
                    "Excluded by double-counting rule DC-INV-001"
                ],
                timestamp=datetime.now(timezone.utc).isoformat(),
                processing_time_ms=(time.monotonic() - start_time) * 1000,
            )
            result.provenance_hash = self._compute_provenance_hash(
                input_data, result
            )
            return result

        if errors:
            result = InvestmentCalculationResult(
                investment_id=input_data.investment_id,
                asset_class=input_data.asset_class,
                errors=errors,
                dc_flags=dc_flags,
                timestamp=datetime.now(timezone.utc).isoformat(),
                processing_time_ms=(time.monotonic() - start_time) * 1000,
            )
            result.provenance_hash = self._compute_provenance_hash(
                input_data, result
            )
            return result

        # EVIC
        evic = self._calculate_evic(input_data.market_cap, input_data.total_debt)

        # Attribution factor
        af = self._calculate_attribution_factor(
            input_data.outstanding_amount, evic
        )

        # PCAF quality
        pcaf_score, calc_method = self._determine_pcaf_quality(input_data)

        # Execute calculation
        result = self._execute_calculation(
            input_data, af, evic, pcaf_score, calc_method
        )
        result.dc_check_passed = dc_passed
        result.dc_flags = dc_flags
        if dc_flags:
            result.warnings.extend(dc_flags)

        # Processing time
        result.processing_time_ms = (time.monotonic() - start_time) * 1000

        # Provenance
        result.provenance_hash = self._compute_provenance_hash(
            input_data, result
        )

        logger.info(
            "Listed equity calculation: id=%s, EVIC=%s, AF=%s, "
            "financed=%s tCO2e, PCAF=%d, method=%s, time=%.1fms",
            input_data.investment_id,
            evic,
            af,
            result.financed_emissions,
            pcaf_score,
            calc_method.value,
            result.processing_time_ms,
        )

        # Record metrics
        self._record_metrics(result)

        return result

    # =========================================================================
    # PUBLIC: PRIVATE EQUITY
    # =========================================================================

    def calculate_private_equity(
        self, input_data: EquityInvestmentInput
    ) -> InvestmentCalculationResult:
        """
        Calculate financed emissions for private/unlisted equity.

        AF = outstanding_amount / (total_equity + total_debt)
        financed_emissions = AF x company_emissions

        Args:
            input_data: EquityInvestmentInput with private equity data.

        Returns:
            InvestmentCalculationResult.

        Example:
            >>> engine = EquityInvestmentCalculatorEngine()
            >>> result = engine.calculate_private_equity(input_data)
        """
        start_time = time.monotonic()
        self._increment_count()

        # Validate
        errors = self._validate_equity_input(input_data)

        # DC check
        dc_passed, dc_flags = self._check_consolidation_boundary(input_data)

        if not dc_passed:
            result = InvestmentCalculationResult(
                investment_id=input_data.investment_id,
                asset_class=input_data.asset_class,
                dc_check_passed=False,
                dc_flags=dc_flags,
                errors=errors + [
                    "Excluded by double-counting rule DC-INV-001"
                ],
                timestamp=datetime.now(timezone.utc).isoformat(),
                processing_time_ms=(time.monotonic() - start_time) * 1000,
            )
            result.provenance_hash = self._compute_provenance_hash(
                input_data, result
            )
            return result

        if errors:
            result = InvestmentCalculationResult(
                investment_id=input_data.investment_id,
                asset_class=input_data.asset_class,
                errors=errors,
                dc_flags=dc_flags,
                timestamp=datetime.now(timezone.utc).isoformat(),
                processing_time_ms=(time.monotonic() - start_time) * 1000,
            )
            result.provenance_hash = self._compute_provenance_hash(
                input_data, result
            )
            return result

        # Denominator: total_equity + total_debt
        total_equity = self._safe_decimal(input_data.total_equity)
        total_debt = self._safe_decimal(input_data.total_debt)
        denominator = self._quantize(total_equity + total_debt, _QUANT_2DP)

        # Attribution factor
        af = self._calculate_attribution_factor(
            input_data.outstanding_amount, denominator
        )

        # PCAF quality
        pcaf_score, calc_method = self._determine_pcaf_quality(input_data)

        # Execute calculation (EVIC = 0 for private equity)
        result = self._execute_calculation(
            input_data, af, _ZERO, pcaf_score, calc_method
        )
        result.dc_check_passed = dc_passed
        result.dc_flags = dc_flags
        if dc_flags:
            result.warnings.extend(dc_flags)

        # Processing time
        result.processing_time_ms = (time.monotonic() - start_time) * 1000

        # Provenance
        result.provenance_hash = self._compute_provenance_hash(
            input_data, result
        )

        logger.info(
            "Private equity calculation: id=%s, denominator=%s, AF=%s, "
            "financed=%s tCO2e, PCAF=%d, method=%s, time=%.1fms",
            input_data.investment_id,
            denominator,
            af,
            result.financed_emissions,
            pcaf_score,
            calc_method.value,
            result.processing_time_ms,
        )

        # Record metrics
        self._record_metrics(result)

        return result

    # =========================================================================
    # PUBLIC: UNLISTED EQUITY (ALIAS)
    # =========================================================================

    def calculate_unlisted_equity(
        self, input_data: EquityInvestmentInput
    ) -> InvestmentCalculationResult:
        """
        Calculate financed emissions for unlisted equity.

        Uses the same methodology as private equity per PCAF.

        Args:
            input_data: EquityInvestmentInput.

        Returns:
            InvestmentCalculationResult.
        """
        return self.calculate_private_equity(input_data)

    # =========================================================================
    # PUBLIC: BATCH CALCULATION
    # =========================================================================

    def calculate_batch(
        self,
        inputs: List[EquityInvestmentInput],
        batch_size: int = 100,
    ) -> List[InvestmentCalculationResult]:
        """
        Calculate financed emissions for a batch of equity investments.

        Processes inputs in chunks for memory efficiency. Records batch
        metrics on completion.

        Args:
            inputs: List of EquityInvestmentInput records.
            batch_size: Chunk size for processing (default 100).

        Returns:
            List of InvestmentCalculationResult, one per input.

        Example:
            >>> engine = EquityInvestmentCalculatorEngine()
            >>> results = engine.calculate_batch(input_list)
            >>> len(results) == len(input_list)
            True
        """
        start_time = time.monotonic()
        results: List[InvestmentCalculationResult] = []

        total = len(inputs)
        logger.info(
            "Starting batch calculation: %d investments, batch_size=%d",
            total,
            batch_size,
        )

        for i in range(0, total, batch_size):
            chunk = inputs[i: i + batch_size]
            for item in chunk:
                try:
                    result = self.calculate(item)
                    results.append(result)
                except Exception as exc:
                    logger.error(
                        "Batch item %s failed: %s",
                        item.investment_id,
                        exc,
                        exc_info=True,
                    )
                    error_result = InvestmentCalculationResult(
                        investment_id=item.investment_id,
                        asset_class=item.asset_class,
                        errors=[f"Calculation failed: {str(exc)}"],
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                    results.append(error_result)

            logger.debug(
                "Batch progress: %d / %d completed",
                min(i + batch_size, total),
                total,
            )

        elapsed_ms = (time.monotonic() - start_time) * 1000

        # Record batch metrics
        metrics = _get_metrics_collector()
        if metrics is not None:
            try:
                metrics.record_batch(
                    status="completed",
                    size=total,
                )
            except Exception:
                pass

        successful = sum(1 for r in results if not r.errors)
        logger.info(
            "Batch calculation complete: %d/%d successful, %.1fms total, "
            "%.1fms/item avg",
            successful,
            total,
            elapsed_ms,
            elapsed_ms / total if total > 0 else 0,
        )

        return results

    # =========================================================================
    # METRICS RECORDING
    # =========================================================================

    def _record_metrics(self, result: InvestmentCalculationResult) -> None:
        """
        Record calculation metrics to Prometheus.

        Args:
            result: Completed calculation result.
        """
        try:
            metrics = _get_metrics_collector()
            if metrics is None:
                return

            metrics.record_calculation(
                method=result.calculation_method,
                asset_class=result.asset_class,
                status="success" if not result.errors else "error",
                duration=result.processing_time_ms / 1000,
                co2e=float(result.financed_emissions),
            )

        except Exception as exc:
            logger.warning("Failed to record metrics: %s", exc)

    # =========================================================================
    # SUMMARY AND STATS
    # =========================================================================

    def get_calculation_count(self) -> int:
        """
        Get the total number of calculations performed.

        Returns:
            Integer count of calculations.
        """
        with self._count_lock:
            return self._calculation_count

    def get_engine_summary(self) -> Dict[str, Any]:
        """
        Get engine status summary.

        Returns:
            Dict with engine stats.
        """
        return {
            "agent_id": AGENT_ID,
            "version": VERSION,
            "engine": "EquityInvestmentCalculatorEngine",
            "calculation_count": self.get_calculation_count(),
            "supported_asset_classes": [
                "listed_equity",
                "corporate_bond",
                "private_equity",
                "unlisted_equity",
            ],
            "calculation_methods": [m.value for m in CalculationMethod],
            "pcaf_scores": [1, 2, 3, 4, 5],
        }

    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton instance (for testing only).

        Warning: Intended for test fixtures only.
        """
        with cls._lock:
            if cls._instance is not None:
                if hasattr(cls._instance, "_initialized"):
                    del cls._instance._initialized
                cls._instance = None


# =============================================================================
# MODULE-LEVEL ACCESSOR
# =============================================================================

_calculator_instance: Optional[EquityInvestmentCalculatorEngine] = None
_calculator_lock: threading.Lock = threading.Lock()


def get_equity_calculator() -> EquityInvestmentCalculatorEngine:
    """
    Get the singleton EquityInvestmentCalculatorEngine instance.

    Thread-safe accessor for the global calculator instance.

    Returns:
        EquityInvestmentCalculatorEngine singleton instance.

    Example:
        >>> calc = get_equity_calculator()
        >>> result = calc.calculate(input_data)
    """
    global _calculator_instance
    with _calculator_lock:
        if _calculator_instance is None:
            _calculator_instance = EquityInvestmentCalculatorEngine()
        return _calculator_instance


def reset_equity_calculator() -> None:
    """
    Reset the module-level calculator instance (for testing only).

    Warning: Intended for test fixtures only.
    """
    global _calculator_instance
    with _calculator_lock:
        _calculator_instance = None
    EquityInvestmentCalculatorEngine.reset()


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "AGENT_ID",
    "AGENT_COMPONENT",
    "VERSION",
    "TABLE_PREFIX",
    # Enumerations
    "EquityType",
    "CalculationMethod",
    "PCAFDataQuality",
    # Data models
    "EquityInvestmentInput",
    "CarbonIntensityResult",
    "InvestmentCalculationResult",
    # Engine class
    "EquityInvestmentCalculatorEngine",
    # Module-level accessors
    "get_equity_calculator",
    "reset_equity_calculator",
]
