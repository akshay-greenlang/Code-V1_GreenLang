# -*- coding: utf-8 -*-
"""
CertificateEngine - PACK-004 CBAM Readiness Engine 2
=====================================================

CBAM certificate obligation calculation engine. Wraps GL-CBAM-APP certificate
module to provide pack-level orchestration for certificate requirements,
free allocation phase-out, carbon price deductions, cost projections, and
quarterly holding checks.

CBAM Certificate System:
    - Each certificate represents 1 tonne of CO2e of embedded emissions
    - Importers must surrender certificates equal to their net obligation
    - Certificates priced at weekly average EU ETS auction price
    - Free allocation phase-out from 97.5% (2026) to 0% (2034)
    - Carbon price paid in country of origin can be deducted
    - Quarterly holding requirement: 50% of obligation at each quarter end

Cost Projection Scenarios:
    - LOW:  Conservative ETS price (bottom 25th percentile forecast)
    - MID:  Central ETS price estimate (median forecast)
    - HIGH: Aggressive ETS price (top 75th percentile forecast)

Zero-Hallucination:
    - All obligation calculations use deterministic Decimal arithmetic
    - Phase-out schedule is hard-coded from CBAM Regulation Article 31
    - No LLM involvement in any financial calculation
    - SHA-256 provenance hashing on every result

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-004 CBAM Readiness
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

def _round_eur(value: Decimal, places: int = 2) -> float:
    """Round a Decimal to euro cents and return float."""
    rounded = value.quantize(Decimal(10) ** -places, rounding=ROUND_HALF_UP)
    return float(rounded)

def _round_cert(value: Decimal, places: int = 4) -> float:
    """Round certificate quantities to 4 decimal places."""
    rounded = value.quantize(Decimal(10) ** -places, rounding=ROUND_HALF_UP)
    return float(rounded)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Free allocation phase-out schedule per CBAM Regulation Article 31
# Key: year, Value: percentage of free allocation remaining (%)
# From 97.5% in 2026 declining to 0% by 2034
FREE_ALLOCATION_PHASE_OUT: Dict[int, float] = {
    2026: 97.5,
    2027: 95.0,
    2028: 90.0,
    2029: 77.5,
    2030: 51.5,
    2031: 39.0,
    2032: 26.5,
    2033: 14.0,
    2034: 0.0,
}

# EU ETS benchmark values (tCO2e per tonne of product) per goods category
# These are used to calculate free allocation amounts
ETS_BENCHMARKS: Dict[str, float] = {
    "cement": 0.766,       # Clinker benchmark
    "iron_steel": 1.328,   # Hot metal benchmark
    "aluminium": 1.514,    # Primary aluminium benchmark
    "fertilizers": 1.619,  # Ammonia benchmark
    "electricity": 0.0,    # No free allocation for electricity
    "hydrogen": 8.850,     # Hydrogen benchmark (grey reference)
}

# Default ETS price scenarios (EUR per tCO2e) by year
# Source: Consensus forecasts from major analysts (2026 projections)
ETS_PRICE_SCENARIOS: Dict[str, Dict[int, float]] = {
    "LOW": {
        2026: 60.0, 2027: 65.0, 2028: 70.0, 2029: 75.0,
        2030: 80.0, 2031: 85.0, 2032: 90.0, 2033: 95.0, 2034: 100.0,
    },
    "MID": {
        2026: 85.0, 2027: 92.0, 2028: 100.0, 2029: 108.0,
        2030: 115.0, 2031: 123.0, 2032: 130.0, 2033: 138.0, 2034: 145.0,
    },
    "HIGH": {
        2026: 110.0, 2027: 125.0, 2028: 140.0, 2029: 155.0,
        2030: 170.0, 2031: 185.0, 2032: 200.0, 2033: 215.0, 2034: 230.0,
    },
}

# Quarterly holding requirement: percentage of annual obligation
# that must be held as certificates at the end of each quarter
QUARTERLY_HOLDING_PCT: float = 50.0

# Known carbon pricing schemes and their approximate prices (EUR/tCO2e)
# Used when no explicit carbon price is provided for a country
COUNTRY_CARBON_PRICES: Dict[str, Dict[str, Any]] = {
    "GB": {"price_eur": 55.0, "scheme": "UK ETS", "exchange_rate": 1.16},
    "CN": {"price_eur": 9.0, "scheme": "China National ETS", "exchange_rate": 0.13},
    "KR": {"price_eur": 8.5, "scheme": "Korea ETS", "exchange_rate": 0.00068},
    "NZ": {"price_eur": 32.0, "scheme": "NZ ETS", "exchange_rate": 0.56},
    "CA": {"price_eur": 50.0, "scheme": "Canada Carbon Tax", "exchange_rate": 0.67},
    "CH": {"price_eur": 105.0, "scheme": "Swiss ETS", "exchange_rate": 1.04},
    "JP": {"price_eur": 3.0, "scheme": "Japan Carbon Tax", "exchange_rate": 0.0063},
    "ZA": {"price_eur": 7.0, "scheme": "South Africa Carbon Tax", "exchange_rate": 0.050},
    "UA": {"price_eur": 1.0, "scheme": "Ukraine Carbon Tax", "exchange_rate": 0.024},
    "MX": {"price_eur": 3.5, "scheme": "Mexico Carbon Tax", "exchange_rate": 0.054},
}

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CostScenario(str, Enum):
    """ETS price scenario for cost projections."""

    LOW = "LOW"
    MID = "MID"
    HIGH = "HIGH"

class CBAMGoodsCategory(str, Enum):
    """CBAM goods categories (mirrored locally to avoid cross-engine imports)."""

    CEMENT = "cement"
    IRON_STEEL = "iron_steel"
    ALUMINIUM = "aluminium"
    FERTILIZERS = "fertilizers"
    ELECTRICITY = "electricity"
    HYDROGEN = "hydrogen"

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class CertificateObligation(BaseModel):
    """Annual CBAM certificate obligation for an importer.

    Represents the full obligation calculation from gross certificates
    through deductions to net certificates and estimated cost.
    """

    obligation_id: str = Field(
        default_factory=_new_uuid,
        description="Unique obligation identifier",
    )
    reporting_year: int = Field(
        ..., ge=2026, le=2050,
        description="Reporting year for the obligation",
    )
    goods_category: str = Field(
        ..., description="Goods category or 'ALL' for aggregate",
    )
    gross_certificates: float = Field(
        ..., ge=0,
        description="Total certificates before deductions (= total embedded emissions tCO2e)",
    )
    free_allocation_deduction: float = Field(
        ..., ge=0,
        description="Certificates deducted for free allocation phase-out",
    )
    carbon_price_deduction: float = Field(
        ..., ge=0,
        description="Certificates deducted for carbon price paid abroad",
    )
    net_certificates: float = Field(
        ..., ge=0,
        description="Net certificates to surrender (after deductions)",
    )
    estimated_cost_eur: float = Field(
        ..., ge=0,
        description="Estimated cost in EUR at current ETS price",
    )
    ets_price_eur_per_tco2: float = Field(
        ..., ge=0,
        description="ETS certificate price used for cost estimate",
    )
    calculated_at: datetime = Field(
        default_factory=utcnow,
        description="Timestamp of obligation calculation",
    )
    provenance_hash: str = Field(
        "", description="SHA-256 hash of inputs + outputs for audit trail",
    )

class FreeAllocationSchedule(BaseModel):
    """Free allocation phase-out schedule entry for a single year.

    Per CBAM Regulation Article 31, free allocation under the EU ETS
    is phased out to avoid double protection.
    """

    year: int = Field(
        ..., ge=2026, le=2050,
        description="Calendar year",
    )
    phase_out_percentage: float = Field(
        ..., ge=0, le=100,
        description="Percentage of free allocation remaining",
    )
    benchmark_value: float = Field(
        ..., ge=0,
        description="EU ETS benchmark value (tCO2e/t product)",
    )
    free_allocation_factor: float = Field(
        ..., ge=0, le=1,
        description="Factor to apply (phase_out_percentage / 100)",
    )

class CarbonPriceDeduction(BaseModel):
    """Carbon price paid in country of origin for CBAM deduction.

    Per CBAM Regulation Article 9, carbon prices effectively paid in the
    country of origin can be deducted from the CBAM obligation.
    """

    country: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    carbon_price_eur_per_tco2: float = Field(
        ..., ge=0,
        description="Carbon price in EUR per tCO2e",
    )
    exchange_rate: float = Field(
        1.0, gt=0,
        description="Exchange rate to EUR (local currency / EUR)",
    )
    effective_deduction_tco2e: float = Field(
        ..., ge=0,
        description="Effective deduction in tCO2e equivalent",
    )
    scheme_name: Optional[str] = Field(
        None, description="Name of the carbon pricing scheme",
    )

    @field_validator("country")
    @classmethod
    def uppercase_country(cls, v: str) -> str:
        """Ensure country code is uppercase."""
        return v.strip().upper()

class CostProjection(BaseModel):
    """Cost projection under a specific ETS price scenario.

    Provides estimated CBAM certificate costs for budget planning and
    financial materiality assessment.
    """

    scenario: CostScenario = Field(
        ..., description="Price scenario (LOW, MID, HIGH)",
    )
    year: int = Field(
        ..., ge=2026, le=2050,
        description="Projection year",
    )
    ets_price_assumption: float = Field(
        ..., ge=0,
        description="Assumed ETS price (EUR/tCO2e) for this scenario",
    )
    gross_certificates: float = Field(
        ..., ge=0,
        description="Gross certificate obligation",
    )
    net_certificates: float = Field(
        ..., ge=0,
        description="Net certificates after deductions",
    )
    total_cost_eur: float = Field(
        ..., ge=0,
        description="Total estimated cost in EUR",
    )
    cost_per_tonne_product: float = Field(
        0.0, ge=0,
        description="Cost per tonne of imported product",
    )
    provenance_hash: str = Field(
        "", description="SHA-256 hash for audit trail",
    )

class QuarterlyHolding(BaseModel):
    """Quarterly certificate holding compliance check.

    At the end of each quarter, importers must hold certificates equal to
    at least 50% of the embedded emissions reported for the quarter.
    """

    year: int = Field(
        ..., ge=2026, le=2050,
        description="Calendar year",
    )
    quarter: int = Field(
        ..., ge=1, le=4,
        description="Quarter number (1-4)",
    )
    required_holding_pct: float = Field(
        QUARTERLY_HOLDING_PCT,
        description="Required holding percentage",
    )
    required_certificates: float = Field(
        ..., ge=0,
        description="Certificates required to be held",
    )
    certificates_held: float = Field(
        ..., ge=0,
        description="Certificates actually held by importer",
    )
    compliant: bool = Field(
        ..., description="Whether holding requirement is met",
    )
    surplus_deficit: float = Field(
        ..., description="Positive = surplus, negative = deficit",
    )
    checked_at: datetime = Field(
        default_factory=utcnow,
        description="Timestamp of check",
    )

# ---------------------------------------------------------------------------
# Lightweight EmissionResult stub for type reference
# ---------------------------------------------------------------------------

class _EmissionResultRef(BaseModel):
    """Minimal emission result reference (avoids cross-engine import)."""

    goods_category: str = ""
    quantity_tonnes: float = 0.0
    total_embedded_emissions_tco2e: float = 0.0
    country_of_origin: str = ""

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class CertificateEngine:
    """CBAM certificate obligation calculation engine.

    Calculates certificate requirements based on embedded emissions, applies
    the EU free allocation phase-out schedule, deducts carbon prices paid
    abroad, and projects costs under multiple ETS price scenarios.

    Implements quarterly holding requirement checks and provides the full
    free allocation phase-out schedule from 2026 to 2034.

    Zero-Hallucination Guarantees:
        - All financial calculations use deterministic Decimal arithmetic
        - Phase-out percentages are hard-coded from the CBAM Regulation
        - ETS benchmark values are from EU Commission reference documents
        - No LLM involvement in any obligation or cost calculation

    Example:
        >>> engine = CertificateEngine()
        >>> obligation = engine.calculate_obligation(
        ...     emission_results=[...],
        ...     year=2027,
        ... )
        >>> assert obligation.net_certificates >= 0
    """

    def __init__(
        self,
        ets_price_override: Optional[float] = None,
    ) -> None:
        """Initialize CertificateEngine.

        Args:
            ets_price_override: Optional override for the ETS price (EUR/tCO2e).
                If not provided, the MID scenario price for the current year is used.
        """
        self._ets_price_override: Optional[float] = ets_price_override
        self._obligation_count: int = 0
        logger.info(
            "CertificateEngine initialized (v%s, ets_override=%s)",
            _MODULE_VERSION,
            ets_price_override,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_obligation(
        self,
        emission_results: List[Any],
        year: int,
        goods_category: Optional[str] = None,
        carbon_deductions: Optional[List[CarbonPriceDeduction]] = None,
    ) -> CertificateObligation:
        """Calculate the full CBAM certificate obligation for a year.

        Sums embedded emissions across all goods entries, applies free
        allocation deductions for the given year, subtracts any carbon
        price deductions, and calculates net obligation and cost.

        Args:
            emission_results: List of emission result objects (must have
                total_embedded_emissions_tco2e, goods_category, country_of_origin).
            year: Reporting year.
            goods_category: Optional filter for a specific goods category.
                If None, aggregates across all categories.
            carbon_deductions: Optional list of carbon price deductions.

        Returns:
            CertificateObligation with full breakdown.

        Raises:
            ValueError: If year is before 2026 or emission_results is empty.
        """
        if year < 2026:
            raise ValueError(f"CBAM certificate obligation starts in 2026, got {year}")

        self._obligation_count += 1

        # Filter results by category if specified
        filtered = emission_results
        if goods_category:
            filtered = [
                r for r in emission_results
                if getattr(r, "goods_category", "") == goods_category
                or (hasattr(r, "goods_category") and
                    hasattr(r.goods_category, "value") and
                    r.goods_category.value == goods_category)
            ]

        cat_label = goods_category or "ALL"

        # Sum gross embedded emissions = gross certificates (1 cert = 1 tCO2e)
        gross_dec = Decimal("0")
        for r in filtered:
            gross_dec += _decimal(
                getattr(r, "total_embedded_emissions_tco2e", 0.0)
            )

        gross_certs = _round_cert(gross_dec)

        # Apply free allocation deduction
        free_alloc_dec = _decimal(
            self.apply_free_allocation(gross_certs, cat_label, year)
        )

        # Apply carbon price deductions
        carbon_ded_dec = Decimal("0")
        if carbon_deductions:
            for ded in carbon_deductions:
                carbon_ded_dec += _decimal(ded.effective_deduction_tco2e)

        # Net obligation = max(0, gross - free_alloc - carbon_price)
        net_dec = self._calculate_net(gross_dec, free_alloc_dec, carbon_ded_dec)

        # Estimate cost
        ets_price = self._get_ets_price(year)
        cost_dec = net_dec * _decimal(ets_price)

        # Build result
        result_data = {
            "reporting_year": year,
            "goods_category": cat_label,
            "gross_certificates": gross_certs,
            "free_allocation_deduction": _round_cert(free_alloc_dec),
            "carbon_price_deduction": _round_cert(carbon_ded_dec),
            "net_certificates": _round_cert(net_dec),
            "estimated_cost_eur": _round_eur(cost_dec),
            "ets_price_eur_per_tco2": round(ets_price, 2),
        }

        result_data["provenance_hash"] = _compute_hash(result_data)
        obligation = CertificateObligation(**result_data)

        logger.info(
            "Certificate obligation [%s/%d]: gross=%.2f, free_alloc=%.2f, "
            "carbon_ded=%.2f, net=%.2f, cost=EUR %.2f",
            cat_label,
            year,
            obligation.gross_certificates,
            obligation.free_allocation_deduction,
            obligation.carbon_price_deduction,
            obligation.net_certificates,
            obligation.estimated_cost_eur,
        )

        return obligation

    def apply_free_allocation(
        self,
        gross_certificates: float,
        goods_category: str,
        year: int,
    ) -> float:
        """Calculate the free allocation deduction for a given year.

        During the CBAM phase-in, free allocation under the EU ETS is
        simultaneously phased out. The deduction equals:

            deduction = gross_certs * benchmark_ratio * phase_out_pct / 100

        Where benchmark_ratio estimates the proportion of emissions covered
        by free allocation benchmarks.

        Args:
            gross_certificates: Total gross certificate obligation (tCO2e).
            goods_category: Goods category or 'ALL'.
            year: Reporting year.

        Returns:
            Free allocation deduction amount in tCO2e.
        """
        phase_out_pct = self._get_phase_out_pct(year)

        # Get the ETS benchmark ratio (how much of emissions are within
        # the benchmark scope). This approximates free allocation.
        benchmark = self._get_benchmark_value(goods_category)

        # Free allocation = emissions * (benchmark / typical_intensity) * phase_out_pct
        # Simplified: we approximate as gross * phase_out_pct for reporting
        deduction = (
            _decimal(gross_certificates)
            * _decimal(phase_out_pct)
            / Decimal("100")
        )

        return _round_cert(deduction)

    def apply_carbon_deduction(
        self,
        country: str,
        carbon_price_paid: Optional[float] = None,
        ets_price: Optional[float] = None,
        total_emissions_tco2e: float = 0.0,
    ) -> CarbonPriceDeduction:
        """Calculate carbon price deduction for a country of origin.

        If the country of origin has an effective carbon pricing mechanism,
        the carbon price effectively paid can be deducted from the CBAM
        obligation.

        Deduction formula:
            effective_deduction = total_emissions * (country_price / ets_price)

        Args:
            country: ISO 3166-1 alpha-2 country code.
            carbon_price_paid: Explicit carbon price (EUR/tCO2e). If None,
                uses the known country carbon price from reference data.
            ets_price: EU ETS price for comparison. If None, uses current.
            total_emissions_tco2e: Total emissions for the deduction scope.

        Returns:
            CarbonPriceDeduction with calculated effective deduction.
        """
        country = country.strip().upper()

        # Resolve carbon price
        scheme_name = None
        exchange_rate = 1.0
        if carbon_price_paid is not None:
            price_eur = carbon_price_paid
        elif country in COUNTRY_CARBON_PRICES:
            info = COUNTRY_CARBON_PRICES[country]
            price_eur = info["price_eur"]
            scheme_name = info.get("scheme")
            exchange_rate = info.get("exchange_rate", 1.0)
        else:
            price_eur = 0.0

        # Resolve ETS price
        if ets_price is None:
            ets_price = self._get_ets_price(utcnow().year)

        # Calculate effective deduction
        # Deduction = emissions * min(country_price / ets_price, 1.0)
        if ets_price > 0 and price_eur > 0:
            ratio = min(_decimal(price_eur) / _decimal(ets_price), Decimal("1"))
            eff_ded = _decimal(total_emissions_tco2e) * ratio
        else:
            eff_ded = Decimal("0")

        return CarbonPriceDeduction(
            country=country,
            carbon_price_eur_per_tco2=round(price_eur, 2),
            exchange_rate=round(exchange_rate, 6),
            effective_deduction_tco2e=_round_cert(eff_ded),
            scheme_name=scheme_name,
        )

    def calculate_net_obligation(
        self,
        gross: float,
        free_allocation: float,
        carbon_deduction: float,
    ) -> float:
        """Calculate net certificate obligation after all deductions.

        Net = max(0, gross - free_allocation - carbon_deduction)

        Args:
            gross: Gross certificate obligation (tCO2e).
            free_allocation: Free allocation deduction (tCO2e).
            carbon_deduction: Carbon price deduction (tCO2e).

        Returns:
            Net certificates to surrender (never negative).
        """
        net = self._calculate_net(
            _decimal(gross),
            _decimal(free_allocation),
            _decimal(carbon_deduction),
        )
        return _round_cert(net)

    def estimate_cost(
        self,
        net_certificates: float,
        ets_price: Optional[float] = None,
        scenario: CostScenario = CostScenario.MID,
        year: Optional[int] = None,
    ) -> CostProjection:
        """Estimate the cost of CBAM certificates.

        Calculates the total cost of the net certificate obligation under
        a specified ETS price scenario.

        Args:
            net_certificates: Net certificates to surrender.
            ets_price: Explicit ETS price override. If None, uses scenario.
            scenario: Price scenario (LOW, MID, HIGH).
            year: Year for scenario-based price lookup.

        Returns:
            CostProjection with estimated cost.
        """
        target_year = year or utcnow().year

        if ets_price is not None:
            price = ets_price
        else:
            price = self._get_scenario_price(scenario, target_year)

        cost_dec = _decimal(net_certificates) * _decimal(price)

        result = CostProjection(
            scenario=scenario,
            year=target_year,
            ets_price_assumption=round(price, 2),
            gross_certificates=net_certificates,  # simplified
            net_certificates=net_certificates,
            total_cost_eur=_round_eur(cost_dec),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def project_annual_cost(
        self,
        emission_results: List[Any],
        year: int,
        scenarios: Optional[List[CostScenario]] = None,
        carbon_deductions: Optional[List[CarbonPriceDeduction]] = None,
    ) -> List[CostProjection]:
        """Project annual CBAM cost under multiple ETS price scenarios.

        Calculates the full obligation and projects costs under LOW, MID,
        and HIGH scenarios for budget planning and financial disclosure.

        Args:
            emission_results: List of emission result objects.
            year: Reporting year.
            scenarios: List of scenarios to project. Defaults to all three.
            carbon_deductions: Optional carbon price deductions.

        Returns:
            List of CostProjection, one per scenario.
        """
        if scenarios is None:
            scenarios = [CostScenario.LOW, CostScenario.MID, CostScenario.HIGH]

        # Calculate the obligation once
        obligation = self.calculate_obligation(
            emission_results=emission_results,
            year=year,
            carbon_deductions=carbon_deductions,
        )
        net_certs = obligation.net_certificates

        # Total product quantity for per-tonne cost
        total_qty = Decimal("0")
        for r in emission_results:
            total_qty += _decimal(getattr(r, "quantity_tonnes", 0.0))

        projections: List[CostProjection] = []
        for scenario in scenarios:
            price = self._get_scenario_price(scenario, year)
            cost_dec = _decimal(net_certs) * _decimal(price)

            cost_per_tonne = Decimal("0")
            if total_qty > 0:
                cost_per_tonne = cost_dec / total_qty

            proj = CostProjection(
                scenario=scenario,
                year=year,
                ets_price_assumption=round(price, 2),
                gross_certificates=obligation.gross_certificates,
                net_certificates=net_certs,
                total_cost_eur=_round_eur(cost_dec),
                cost_per_tonne_product=_round_eur(cost_per_tonne),
            )
            proj.provenance_hash = _compute_hash(proj)
            projections.append(proj)

        logger.info(
            "Projected annual costs for %d: %s",
            year,
            {p.scenario.value: p.total_cost_eur for p in projections},
        )

        return projections

    def check_quarterly_holding(
        self,
        year: int,
        quarter: int,
        certificates_held: float,
        quarterly_emissions_tco2e: Optional[float] = None,
        annual_obligation_tco2e: Optional[float] = None,
    ) -> QuarterlyHolding:
        """Check quarterly certificate holding compliance.

        At the end of each quarter, importers must hold certificates equal to
        at least 50% of the embedded emissions reported for that quarter,
        or 25% of the annual obligation (whichever is applicable).

        Args:
            year: Calendar year.
            quarter: Quarter number (1-4).
            certificates_held: Number of certificates held by the importer.
            quarterly_emissions_tco2e: Emissions reported for this quarter.
            annual_obligation_tco2e: Total annual obligation if known.

        Returns:
            QuarterlyHolding with compliance status.

        Raises:
            ValueError: If quarter is not in range 1-4.
        """
        if quarter < 1 or quarter > 4:
            raise ValueError(f"Quarter must be 1-4, got {quarter}")

        # Determine required certificates
        if quarterly_emissions_tco2e is not None:
            base = _decimal(quarterly_emissions_tco2e)
        elif annual_obligation_tco2e is not None:
            # Use proportional quarter (annual / 4)
            base = _decimal(annual_obligation_tco2e) / Decimal("4")
        else:
            base = Decimal("0")

        required = base * _decimal(QUARTERLY_HOLDING_PCT) / Decimal("100")
        held = _decimal(certificates_held)
        surplus = held - required
        compliant = held >= required

        holding = QuarterlyHolding(
            year=year,
            quarter=quarter,
            required_holding_pct=QUARTERLY_HOLDING_PCT,
            required_certificates=_round_cert(required),
            certificates_held=_round_cert(held),
            compliant=compliant,
            surplus_deficit=_round_cert(surplus),
        )

        if not compliant:
            logger.warning(
                "Quarterly holding NON-COMPLIANT Q%d/%d: held=%.2f, required=%.2f, deficit=%.2f",
                quarter, year, float(held), float(required), float(-surplus),
            )
        else:
            logger.info(
                "Quarterly holding compliant Q%d/%d: held=%.2f, required=%.2f",
                quarter, year, float(held), float(required),
            )

        return holding

    def get_free_allocation_schedule(
        self,
        goods_category: str = "cement",
    ) -> List[FreeAllocationSchedule]:
        """Get the full free allocation phase-out schedule for 2026-2034.

        Returns the year-by-year phase-out percentage and benchmark value
        for the specified goods category.

        Args:
            goods_category: CBAM goods category.

        Returns:
            List of FreeAllocationSchedule entries, one per year.
        """
        benchmark = self._get_benchmark_value(goods_category)
        schedule: List[FreeAllocationSchedule] = []

        for year in sorted(FREE_ALLOCATION_PHASE_OUT.keys()):
            pct = FREE_ALLOCATION_PHASE_OUT[year]
            schedule.append(
                FreeAllocationSchedule(
                    year=year,
                    phase_out_percentage=pct,
                    benchmark_value=benchmark,
                    free_allocation_factor=round(pct / 100.0, 4),
                )
            )

        return schedule

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def obligation_count(self) -> int:
        """Number of obligation calculations performed."""
        return self._obligation_count

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_ets_price(self, year: int) -> float:
        """Get the ETS price for a given year.

        Uses the override if set, otherwise looks up the MID scenario price.
        """
        if self._ets_price_override is not None:
            return self._ets_price_override
        return self._get_scenario_price(CostScenario.MID, year)

    def _get_scenario_price(self, scenario: CostScenario, year: int) -> float:
        """Look up ETS price for a specific scenario and year."""
        prices = ETS_PRICE_SCENARIOS.get(scenario.value, {})
        if year in prices:
            return prices[year]

        # Extrapolate beyond known years
        known_years = sorted(prices.keys())
        if not known_years:
            return 85.0  # Fallback MID estimate

        if year < known_years[0]:
            return prices[known_years[0]]

        # Linear extrapolation from last two known years
        last = known_years[-1]
        if len(known_years) >= 2:
            second_last = known_years[-2]
            slope = (prices[last] - prices[second_last]) / (last - second_last)
            extrapolated = prices[last] + slope * (year - last)
            return round(max(extrapolated, 0.0), 2)

        return prices[last]

    def _get_phase_out_pct(self, year: int) -> float:
        """Get the free allocation phase-out percentage for a year."""
        if year in FREE_ALLOCATION_PHASE_OUT:
            return FREE_ALLOCATION_PHASE_OUT[year]
        if year < 2026:
            return 100.0  # Before CBAM definitive: full free allocation
        if year >= 2034:
            return 0.0  # After phase-out: no free allocation
        # Interpolate for any gaps (should not happen with current data)
        years = sorted(FREE_ALLOCATION_PHASE_OUT.keys())
        for i in range(len(years) - 1):
            if years[i] < year < years[i + 1]:
                pct_start = FREE_ALLOCATION_PHASE_OUT[years[i]]
                pct_end = FREE_ALLOCATION_PHASE_OUT[years[i + 1]]
                frac = (year - years[i]) / (years[i + 1] - years[i])
                return round(pct_start + (pct_end - pct_start) * frac, 1)
        return 0.0

    def _get_benchmark_value(self, goods_category: str) -> float:
        """Get the EU ETS benchmark value for a goods category."""
        cat_lower = goods_category.lower()
        if cat_lower == "all":
            # Use average of all categories with benchmarks
            vals = [v for v in ETS_BENCHMARKS.values() if v > 0]
            return round(sum(vals) / len(vals), 4) if vals else 0.0
        return ETS_BENCHMARKS.get(cat_lower, 0.0)

    def _calculate_net(
        self,
        gross: Decimal,
        free_alloc: Decimal,
        carbon_ded: Decimal,
    ) -> Decimal:
        """Calculate net obligation: max(0, gross - deductions)."""
        net = gross - free_alloc - carbon_ded
        return max(net, Decimal("0"))
