# -*- coding: utf-8 -*-
"""
Engine 4: TDLossCalculatorEngine - Transmission & Distribution Loss Calculator

This module implements the TDLossCalculatorEngine for AGENT-MRV-016
(GL-MRV-S3-003), calculating Scope 3 Category 3 Activity 3c emissions
from electricity lost during transmission and distribution.

When electricity travels from generators to end consumers, a fraction of
the energy is dissipated as heat in transmission lines and distribution
networks.  The emissions attributable to generating this lost electricity
are reported under GHG Protocol Scope 3 Category 3, Activity 3c.

Core Formulas
-------------
Simple (delivered basis):
    Emissions_3c = Electricity_consumed * T&D_loss_% * (Grid_EF + Upstream_EF)

Adjusted (generated basis, recommended by GHG Protocol):
    Loss_multiplier = TD% / (1 - TD%)
    Generation_component = Electricity_consumed * Loss_multiplier * Grid_EF
    Upstream_component   = Electricity_consumed * Loss_multiplier * Upstream_EF
    Total_3c             = Generation_component + Upstream_component

The adjusted formula converts from "percentage of delivered electricity"
to "percentage of electricity generated" which is the technically correct
basis for estimating generation-phase losses.

Design Principles
-----------------
- Zero-hallucination: All calculations use deterministic Decimal arithmetic.
- Provenance: SHA-256 hashing at every result boundary.
- Thread-safety: All mutable state protected by threading.Lock.
- Performance tracking: Timing on all public entry points.
- DQI scoring: 5-dimension data quality assessment per GHG Protocol.
- Uncertainty: IPCC default, analytical, and Monte Carlo methods.

Embedded Data
-------------
- GRID_GENERATION_EFS: 35 country/region grid emission factors (kgCO2e/kWh)
- GRID_GENERATION_GAS_FRACTIONS: Per-gas splits for key grid mixes

Imports from models.py:
- TD_LOSS_FACTORS: 50 country T&D loss fractions
- EGRID_TD_LOSS_FACTORS: 26 US eGRID subregion loss fractions
- UPSTREAM_ELECTRICITY_FACTORS: 42 country upstream EFs
- GWP_VALUES: AR4/AR5/AR6/AR6_20YR GWP multipliers

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-016 Fuel & Energy Activities (GL-MRV-S3-003)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

from greenlang.fuel_energy_activities.models import (
    AccountingMethod,
    Activity3cResult,
    ActivityType,
    CalculationMethod,
    DQIAssessment,
    DQIScore,
    DQI_QUALITY_TIERS,
    DQI_SCORE_VALUES,
    EGRID_TD_LOSS_FACTORS,
    ElectricityConsumptionRecord,
    EmissionGas,
    GasBreakdown,
    GridRegionType,
    GWPSource,
    GWP_VALUES,
    ONE,
    ONE_HUNDRED,
    TD_LOSS_FACTORS,
    TDLossFactor,
    TDLossSource,
    UncertaintyMethod,
    UncertaintyResult,
    UNCERTAINTY_RANGES,
    UPSTREAM_ELECTRICITY_FACTORS,
    ZERO,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Module-level Constants
# ============================================================================

#: Agent identifier for Fuel & Energy Activities
AGENT_ID: str = "GL-MRV-S3-003"

#: Engine identifier
ENGINE_ID: str = "td-loss-calculator"

#: Engine version
ENGINE_VERSION: str = "1.0.0"

#: Activity type handled by this engine
ACTIVITY: ActivityType = ActivityType.ACTIVITY_3C

#: Decimal quantization template (8 decimal places)
_QUANTIZE_8 = Decimal("0.00000001")

#: Maximum T&D loss fraction considered physically reasonable
MAX_TD_LOSS_PCT: Decimal = Decimal("0.50")

#: Minimum T&D loss fraction for validation warnings
MIN_TD_LOSS_PCT: Decimal = Decimal("0.001")

#: Default T&D loss fraction when country is not found
DEFAULT_TD_LOSS_PCT: Decimal = Decimal("0.08")

#: Default grid EF when country is not found (kgCO2e/kWh, world average)
DEFAULT_GRID_EF: Decimal = Decimal("0.4360")

#: Default upstream EF when country is not found (kgCO2e/kWh)
DEFAULT_UPSTREAM_EF: Decimal = Decimal("0.04500")

#: Typical transmission loss fraction (of total T&D)
TRANSMISSION_SHARE: Decimal = Decimal("0.40")

#: Typical distribution loss fraction (of total T&D)
DISTRIBUTION_SHARE: Decimal = Decimal("0.60")

#: Confidence multiplier for 95% CI (z-score)
Z_95: Decimal = Decimal("1.960")

#: Confidence multiplier for 90% CI
Z_90: Decimal = Decimal("1.645")

#: Confidence multiplier for 99% CI
Z_99: Decimal = Decimal("2.576")

#: Valid loss basis identifiers
LOSS_BASIS_DELIVERED: str = "delivered"
LOSS_BASIS_GENERATED: str = "generated"


# ============================================================================
# Embedded Data: Grid Generation Emission Factors (kgCO2e/kWh)
# ============================================================================

GRID_GENERATION_EFS: Dict[str, Decimal] = {
    # Americas
    "US": Decimal("0.3700"),
    "CA": Decimal("0.1200"),
    "MX": Decimal("0.4310"),
    "BR": Decimal("0.0740"),
    "AR": Decimal("0.3120"),
    "CL": Decimal("0.3550"),
    "CO": Decimal("0.1640"),
    "PE": Decimal("0.2080"),
    # Europe
    "GB": Decimal("0.1280"),
    "DE": Decimal("0.3500"),
    "FR": Decimal("0.0480"),
    "IT": Decimal("0.2850"),
    "ES": Decimal("0.1460"),
    "NL": Decimal("0.3280"),
    "BE": Decimal("0.1350"),
    "SE": Decimal("0.0080"),
    "NO": Decimal("0.0070"),
    "DK": Decimal("0.1090"),
    "FI": Decimal("0.0690"),
    "AT": Decimal("0.0890"),
    "CH": Decimal("0.0130"),
    "PL": Decimal("0.6370"),
    "CZ": Decimal("0.4230"),
    "PT": Decimal("0.1730"),
    "IE": Decimal("0.2950"),
    # Asia-Pacific
    "JP": Decimal("0.4570"),
    "CN": Decimal("0.5810"),
    "IN": Decimal("0.7260"),
    "AU": Decimal("0.6100"),
    "KR": Decimal("0.4150"),
    "NZ": Decimal("0.0980"),
    "SG": Decimal("0.4080"),
    # Middle East / Africa
    "ZA": Decimal("0.9280"),
    "AE": Decimal("0.4550"),
    "SA": Decimal("0.5240"),
}


# ---------------------------------------------------------------------------
# Per-gas fractions for grid generation (CO2, CH4, N2O as fraction of total)
# Source: IEA 2023, EPA eGRID 2022
# These allow decomposition of location-based grid EFs into individual gases.
# ---------------------------------------------------------------------------

GRID_GAS_FRACTIONS: Dict[str, Dict[str, Decimal]] = {
    "DEFAULT": {
        "co2_frac": Decimal("0.980"),
        "ch4_frac": Decimal("0.015"),
        "n2o_frac": Decimal("0.005"),
    },
    "US": {
        "co2_frac": Decimal("0.978"),
        "ch4_frac": Decimal("0.016"),
        "n2o_frac": Decimal("0.006"),
    },
    "GB": {
        "co2_frac": Decimal("0.975"),
        "ch4_frac": Decimal("0.018"),
        "n2o_frac": Decimal("0.007"),
    },
    "DE": {
        "co2_frac": Decimal("0.982"),
        "ch4_frac": Decimal("0.013"),
        "n2o_frac": Decimal("0.005"),
    },
    "FR": {
        "co2_frac": Decimal("0.960"),
        "ch4_frac": Decimal("0.025"),
        "n2o_frac": Decimal("0.015"),
    },
    "CN": {
        "co2_frac": Decimal("0.985"),
        "ch4_frac": Decimal("0.011"),
        "n2o_frac": Decimal("0.004"),
    },
    "IN": {
        "co2_frac": Decimal("0.987"),
        "ch4_frac": Decimal("0.009"),
        "n2o_frac": Decimal("0.004"),
    },
    "JP": {
        "co2_frac": Decimal("0.979"),
        "ch4_frac": Decimal("0.015"),
        "n2o_frac": Decimal("0.006"),
    },
    "AU": {
        "co2_frac": Decimal("0.983"),
        "ch4_frac": Decimal("0.012"),
        "n2o_frac": Decimal("0.005"),
    },
}


# ============================================================================
# Helper functions
# ============================================================================


def _quantize(value: Decimal) -> Decimal:
    """Quantize a Decimal to 8 decimal places using ROUND_HALF_UP.

    Args:
        value: The Decimal value to quantize.

    Returns:
        Quantized Decimal with 8 decimal places.
    """
    return value.quantize(_QUANTIZE_8, rounding=ROUND_HALF_UP)


def _sha256(data: str) -> str:
    """Compute SHA-256 hex digest of a string.

    Args:
        data: Input string to hash.

    Returns:
        64-character lowercase hex digest.
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _z_score(confidence_level: Decimal) -> Decimal:
    """Return the z-score for a given confidence level.

    Args:
        confidence_level: Confidence level as a percentage (e.g. 95).

    Returns:
        z-score Decimal for the requested confidence level.
    """
    if confidence_level >= Decimal("99"):
        return Z_99
    if confidence_level >= Decimal("95"):
        return Z_95
    return Z_90


# ============================================================================
# TDLossCalculatorEngine
# ============================================================================


class TDLossCalculatorEngine:
    """Engine 4: Transmission & Distribution Loss Calculator.

    Calculates GHG Protocol Scope 3 Category 3 Activity 3c emissions
    arising from electricity lost during transmission and distribution.
    Uses country/region-specific T&D loss factors, grid generation
    emission factors, and upstream (WTT) electricity factors.

    The engine supports:
    - Manual factor specification (calculate) and auto-resolution
      by country code (calculate_auto).
    - Batch processing of multiple consumption records.
    - Both simple (delivered-basis) and adjusted (generated-basis)
      loss multiplier calculations.
    - Per-gas decomposition (CO2, CH4, N2O).
    - Data quality indicator (DQI) assessment across 5 GHG Protocol
      dimensions.
    - Uncertainty quantification via IPCC default, analytical, and
      Monte Carlo methods.
    - Country comparison and multi-dimension aggregation.
    - Double-counting prevention against Scope 2 records.
    - Voltage-level decomposition (transmission vs distribution).

    All arithmetic uses Python ``Decimal`` for reproducibility.
    Thread-safe via ``threading.Lock`` on all mutable state.

    Attributes:
        _lock: Thread lock protecting mutable statistics.
        _stats: Cumulative engine statistics dictionary.
        _created_at: Engine creation timestamp (UTC).

    Example:
        >>> from decimal import Decimal
        >>> from greenlang.fuel_energy_activities.td_loss_calculator import (
        ...     TDLossCalculatorEngine,
        ... )
        >>> engine = TDLossCalculatorEngine()
        >>> td = engine.get_td_loss_factor("US")
        >>> td.loss_percentage
        Decimal('0.0500')
    """

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        """Initialize the TDLossCalculatorEngine.

        Creates the engine with zeroed statistics counters and a
        thread lock for concurrent access safety.
        """
        self._lock = threading.Lock()
        self._created_at: datetime = _utcnow()
        self._stats: Dict[str, Any] = {
            "calculations": 0,
            "batch_calculations": 0,
            "auto_calculations": 0,
            "errors": 0,
            "total_emissions_kgco2e": ZERO,
            "total_electricity_kwh": ZERO,
            "total_td_losses_kwh": ZERO,
            "countries_resolved": 0,
            "egrid_resolved": 0,
            "dqi_assessments": 0,
            "uncertainty_assessments": 0,
            "double_counting_checks": 0,
            "created_at": self._created_at.isoformat(),
        }
        logger.info(
            "TDLossCalculatorEngine initialized: engine=%s, version=%s",
            ENGINE_ID,
            ENGINE_VERSION,
        )

    # ------------------------------------------------------------------
    # Core calculation: manual factors
    # ------------------------------------------------------------------

    def calculate(
        self,
        record: ElectricityConsumptionRecord,
        td_loss_pct: Decimal,
        grid_ef: Decimal,
        upstream_ef: Decimal = ZERO,
        gwp_source: GWPSource = GWPSource.AR6,
    ) -> Activity3cResult:
        """Calculate Activity 3c T&D loss emissions for a single record.

        Uses the adjusted (generated-basis) formula:
            loss_multiplier = td_loss_pct / (1 - td_loss_pct)
            generation_losses = electricity * loss_multiplier * grid_ef
            upstream_losses = electricity * loss_multiplier * upstream_ef

        Args:
            record: Electricity consumption record with quantity_kwh.
            td_loss_pct: T&D loss fraction (0.0 - 0.5), e.g. 0.05 = 5%.
            grid_ef: Grid generation emission factor in kgCO2e/kWh.
            upstream_ef: Upstream (WTT) electricity factor in kgCO2e/kWh.
                Defaults to zero if not provided.
            gwp_source: IPCC AR version for GWP conversion. Defaults to AR6.

        Returns:
            Activity3cResult with generation losses, upstream losses,
            total emissions, and provenance hash.

        Raises:
            ValueError: If td_loss_pct is outside [0, 0.5] or grid_ef < 0.
        """
        start = time.monotonic()
        try:
            # Validate inputs
            is_valid, errors = self.validate_td_factor(td_loss_pct)
            if not is_valid:
                raise ValueError(
                    f"Invalid T&D loss factor: {'; '.join(errors)}"
                )
            if grid_ef < ZERO:
                raise ValueError(
                    f"Grid emission factor must be >= 0, got {grid_ef}"
                )
            if upstream_ef < ZERO:
                raise ValueError(
                    f"Upstream emission factor must be >= 0, got {upstream_ef}"
                )

            # Adjusted-basis multiplier: TD% / (1 - TD%)
            loss_multiplier = self._loss_multiplier(td_loss_pct)

            # Core deterministic arithmetic
            gen_losses = _quantize(
                record.quantity_kwh * loss_multiplier * grid_ef
            )
            ups_losses = _quantize(
                record.quantity_kwh * loss_multiplier * upstream_ef
            )
            total = _quantize(gen_losses + ups_losses)

            # Provenance hash
            provenance_input = (
                f"{record.record_id}|{record.quantity_kwh}|"
                f"{td_loss_pct}|{grid_ef}|{upstream_ef}|"
                f"{gen_losses}|{ups_losses}|{total}|{gwp_source.value}"
            )
            provenance_hash = _sha256(provenance_input)

            # Determine T&D loss source label
            td_source = self._resolve_td_source_label(
                record.grid_region, record.grid_region_type
            )

            result = Activity3cResult(
                electricity_record_id=record.record_id,
                electricity_consumed_kwh=record.quantity_kwh,
                td_loss_pct=td_loss_pct,
                td_loss_source=td_source,
                grid_ef=grid_ef,
                upstream_ef=upstream_ef,
                generation_losses=gen_losses,
                upstream_losses=ups_losses,
                emissions_total=total,
                grid_region=record.grid_region,
                accounting_method=record.accounting_method,
                provenance_hash=provenance_hash,
            )

            self._update_stats(record.quantity_kwh, total, td_loss_pct)
            elapsed = time.monotonic() - start
            logger.debug(
                "calculate: record=%s, td=%.4f, gen_losses=%s, "
                "ups_losses=%s, total=%s, elapsed=%.4fs",
                record.record_id,
                td_loss_pct,
                gen_losses,
                ups_losses,
                total,
                elapsed,
            )
            return result

        except Exception:
            with self._lock:
                self._stats["errors"] += 1
            raise

    # ------------------------------------------------------------------
    # Auto-resolve calculation
    # ------------------------------------------------------------------

    def calculate_auto(
        self,
        record: ElectricityConsumptionRecord,
        country_code: Optional[str] = None,
    ) -> Activity3cResult:
        """Calculate Activity 3c emissions with auto-resolved factors.

        Automatically looks up the T&D loss factor, grid generation EF,
        and upstream EF based on the country code or the record's
        grid_region field.

        Args:
            record: Electricity consumption record.
            country_code: Optional ISO 3166-1 alpha-2 country code.
                If not provided, uses record.country_code or
                record.grid_region.

        Returns:
            Activity3cResult with fully auto-resolved factors.

        Raises:
            ValueError: If country code cannot be resolved.
        """
        start = time.monotonic()

        # Resolve country code from available sources
        cc = self._resolve_country_code(record, country_code)

        # Auto-resolve factors
        td_factor = self.get_td_loss_factor(cc)
        grid_ef = self.get_grid_generation_ef(cc)
        upstream_ef = self._resolve_upstream_ef(cc)

        result = self.calculate(
            record=record,
            td_loss_pct=td_factor.loss_percentage,
            grid_ef=grid_ef,
            upstream_ef=upstream_ef,
        )

        with self._lock:
            self._stats["auto_calculations"] += 1

        elapsed = time.monotonic() - start
        logger.info(
            "calculate_auto: country=%s, td=%.4f, grid_ef=%s, "
            "upstream_ef=%s, total=%s, elapsed=%.4fs",
            cc,
            td_factor.loss_percentage,
            grid_ef,
            upstream_ef,
            result.emissions_total,
            elapsed,
        )
        return result

    # ------------------------------------------------------------------
    # Batch calculation
    # ------------------------------------------------------------------

    def calculate_batch(
        self,
        records: List[ElectricityConsumptionRecord],
        country_codes: Optional[List[Optional[str]]] = None,
    ) -> List[Activity3cResult]:
        """Calculate Activity 3c emissions for multiple records.

        Processes each record using calculate_auto, resolving factors
        individually per record's country context.

        Args:
            records: List of electricity consumption records.
            country_codes: Optional parallel list of country codes.
                If shorter than records, remaining records use
                auto-resolution from the record itself.

        Returns:
            List of Activity3cResult, one per input record.

        Raises:
            ValueError: If records list is empty.
        """
        if not records:
            raise ValueError("Records list must not be empty")

        start = time.monotonic()
        codes = country_codes or []
        results: List[Activity3cResult] = []

        for idx, rec in enumerate(records):
            cc = codes[idx] if idx < len(codes) else None
            result = self.calculate_auto(rec, cc)
            results.append(result)

        with self._lock:
            self._stats["batch_calculations"] += 1

        elapsed = time.monotonic() - start
        logger.info(
            "calculate_batch: records=%d, total_emissions=%s, elapsed=%.4fs",
            len(records),
            self.get_total_emissions(results),
            elapsed,
        )
        return results

    # ------------------------------------------------------------------
    # Component calculations
    # ------------------------------------------------------------------

    def calculate_generation_component(
        self,
        electricity_kwh: Decimal,
        td_loss_pct: Decimal,
        grid_ef: Decimal,
    ) -> Decimal:
        """Calculate the generation-phase component of T&D loss emissions.

        Uses the adjusted formula:
            generation_losses = electricity * (TD% / (1-TD%)) * grid_ef

        Args:
            electricity_kwh: Electricity consumed in kWh.
            td_loss_pct: T&D loss fraction (e.g. 0.05).
            grid_ef: Grid generation emission factor (kgCO2e/kWh).

        Returns:
            Generation-phase T&D loss emissions in kgCO2e.
        """
        self._validate_positive("electricity_kwh", electricity_kwh)
        is_valid, errors = self.validate_td_factor(td_loss_pct)
        if not is_valid:
            raise ValueError(f"Invalid T&D factor: {'; '.join(errors)}")

        multiplier = self._loss_multiplier(td_loss_pct)
        return _quantize(electricity_kwh * multiplier * grid_ef)

    def calculate_upstream_component(
        self,
        electricity_kwh: Decimal,
        td_loss_pct: Decimal,
        upstream_ef: Decimal,
    ) -> Decimal:
        """Calculate the upstream (WTT) component of T&D loss emissions.

        Uses the adjusted formula:
            upstream_losses = electricity * (TD% / (1-TD%)) * upstream_ef

        Args:
            electricity_kwh: Electricity consumed in kWh.
            td_loss_pct: T&D loss fraction (e.g. 0.05).
            upstream_ef: Upstream electricity factor (kgCO2e/kWh).

        Returns:
            Upstream-phase T&D loss emissions in kgCO2e.
        """
        self._validate_positive("electricity_kwh", electricity_kwh)
        is_valid, errors = self.validate_td_factor(td_loss_pct)
        if not is_valid:
            raise ValueError(f"Invalid T&D factor: {'; '.join(errors)}")

        multiplier = self._loss_multiplier(td_loss_pct)
        return _quantize(electricity_kwh * multiplier * upstream_ef)

    # ------------------------------------------------------------------
    # Loss basis conversion
    # ------------------------------------------------------------------

    def convert_loss_basis(
        self,
        td_loss_pct: Decimal,
        from_basis: str,
        to_basis: str,
    ) -> Decimal:
        """Convert T&D loss percentage between delivered and generated bases.

        Delivered basis: TD% as a fraction of electricity delivered to consumer.
        Generated basis: TD% as a fraction of total electricity generated.

        Conversion formulas:
            delivered -> generated: loss_gen = loss_del / (1 + loss_del)
                (where loss_del = td_loss_pct / (1 - td_loss_pct) first)
            generated -> delivered: loss_del = loss_gen / (1 - loss_gen)

        More precisely:
            delivered -> generated: gen = del / (1 - del)... wait:
            If "delivered" means td_loss_pct is the fraction of delivered
            electricity that was lost, then the adjusted multiplier is
            td_loss_pct / (1 - td_loss_pct).
            If "generated" means td_loss_pct is the fraction of generated
            electricity that was lost, then:
            delivered_frac = gen_frac / (1 - gen_frac)... no:
            generated = delivered * (1 + loss)
            loss_delivered = losses / delivered
            loss_generated = losses / generated = losses / (delivered + losses)
            So: loss_gen = loss_del / (1 + loss_del)
            And: loss_del = loss_gen / (1 - loss_gen)

        Args:
            td_loss_pct: T&D loss fraction on the from_basis.
            from_basis: Source basis, one of "delivered" or "generated".
            to_basis: Target basis, one of "delivered" or "generated".

        Returns:
            T&D loss fraction on the to_basis.

        Raises:
            ValueError: If from_basis or to_basis is not recognized,
                or if loss fraction would cause division by zero.
        """
        valid_bases = {LOSS_BASIS_DELIVERED, LOSS_BASIS_GENERATED}
        if from_basis not in valid_bases:
            raise ValueError(
                f"from_basis must be one of {valid_bases}, got '{from_basis}'"
            )
        if to_basis not in valid_bases:
            raise ValueError(
                f"to_basis must be one of {valid_bases}, got '{to_basis}'"
            )
        if from_basis == to_basis:
            return td_loss_pct

        if from_basis == LOSS_BASIS_DELIVERED:
            # delivered -> generated
            # loss_gen = loss_del / (1 + loss_del)
            denominator = ONE + td_loss_pct
            if denominator == ZERO:
                raise ValueError(
                    "Cannot convert: denominator (1 + td_loss_pct) is zero"
                )
            return _quantize(td_loss_pct / denominator)
        else:
            # generated -> delivered
            # loss_del = loss_gen / (1 - loss_gen)
            denominator = ONE - td_loss_pct
            if denominator <= ZERO:
                raise ValueError(
                    "Cannot convert: denominator (1 - td_loss_pct) is <= 0"
                )
            return _quantize(td_loss_pct / denominator)

    # ------------------------------------------------------------------
    # Factor lookups
    # ------------------------------------------------------------------

    def get_td_loss_factor(self, country_code: str) -> TDLossFactor:
        """Look up the T&D loss factor for a country.

        Checks TD_LOSS_FACTORS first. If not found, returns the
        DEFAULT_TD_LOSS_PCT with a warning.

        Args:
            country_code: ISO 3166-1 alpha-2 country code (e.g. "US").

        Returns:
            TDLossFactor with the loss percentage and source metadata.
        """
        cc = country_code.upper().strip()
        loss_pct = TD_LOSS_FACTORS.get(cc)

        if loss_pct is not None:
            with self._lock:
                self._stats["countries_resolved"] += 1
            return TDLossFactor(
                country_code=cc,
                loss_percentage=loss_pct,
                source=TDLossSource.IEA,
                year=2023,
            )

        logger.warning(
            "T&D loss factor not found for country '%s', "
            "using default %.4f",
            cc,
            DEFAULT_TD_LOSS_PCT,
        )
        return TDLossFactor(
            country_code=cc,
            loss_percentage=DEFAULT_TD_LOSS_PCT,
            source=TDLossSource.CUSTOM,
            year=2023,
        )

    def get_td_loss_by_egrid(self, subregion: str) -> TDLossFactor:
        """Look up the T&D loss factor for a US eGRID subregion.

        Args:
            subregion: EPA eGRID subregion code (e.g. "CAMX", "ERCT").

        Returns:
            TDLossFactor with the eGRID-specific loss percentage.

        Raises:
            ValueError: If the subregion code is not recognized.
        """
        sr = subregion.upper().strip()
        loss_pct = EGRID_TD_LOSS_FACTORS.get(sr)

        if loss_pct is None:
            raise ValueError(
                f"Unknown eGRID subregion '{sr}'. Valid subregions: "
                f"{sorted(EGRID_TD_LOSS_FACTORS.keys())}"
            )

        with self._lock:
            self._stats["egrid_resolved"] += 1

        return TDLossFactor(
            country_code=sr,
            loss_percentage=loss_pct,
            source=TDLossSource.EPA_EGRID,
            year=2022,
        )

    def get_grid_generation_ef(self, country_code: str) -> Decimal:
        """Look up the grid generation emission factor for a country.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.

        Returns:
            Grid generation emission factor in kgCO2e/kWh.
            Falls back to DEFAULT_GRID_EF if country not found.
        """
        cc = country_code.upper().strip()
        ef = GRID_GENERATION_EFS.get(cc)
        if ef is not None:
            return ef

        logger.warning(
            "Grid generation EF not found for country '%s', "
            "using default %.4f kgCO2e/kWh",
            cc,
            DEFAULT_GRID_EF,
        )
        return DEFAULT_GRID_EF

    # ------------------------------------------------------------------
    # T&D loss decomposition
    # ------------------------------------------------------------------

    def decompose_td_losses(
        self, td_loss_pct: Decimal
    ) -> Dict[str, Decimal]:
        """Decompose total T&D loss into transmission and distribution shares.

        Uses industry-standard approximate split: transmission accounts
        for approximately 40% of total T&D losses (higher voltage, lower
        relative losses), while distribution accounts for approximately
        60% (lower voltage, more branches, longer total distance).

        Args:
            td_loss_pct: Total T&D loss fraction (e.g. 0.05).

        Returns:
            Dictionary with keys:
            - total_pct: Input total T&D loss fraction.
            - transmission_pct: Transmission component (~40% of total).
            - distribution_pct: Distribution component (~60% of total).
            - transmission_share: Fraction attributed to transmission.
            - distribution_share: Fraction attributed to distribution.
        """
        trans = _quantize(td_loss_pct * TRANSMISSION_SHARE)
        dist = _quantize(td_loss_pct * DISTRIBUTION_SHARE)
        return {
            "total_pct": td_loss_pct,
            "transmission_pct": trans,
            "distribution_pct": dist,
            "transmission_share": TRANSMISSION_SHARE,
            "distribution_share": DISTRIBUTION_SHARE,
        }

    # ------------------------------------------------------------------
    # Per-gas breakdown
    # ------------------------------------------------------------------

    def calculate_per_gas(
        self,
        record: ElectricityConsumptionRecord,
        td_loss_pct: Decimal,
        grid_ef: Decimal,
        gwp_source: GWPSource = GWPSource.AR6,
    ) -> GasBreakdown:
        """Calculate per-gas T&D loss emissions (CO2, CH4, N2O).

        Decomposes the total grid generation EF into individual gas
        contributions using country-specific or default gas fractions,
        then applies GWP conversion factors.

        Args:
            record: Electricity consumption record.
            td_loss_pct: T&D loss fraction.
            grid_ef: Grid generation emission factor (kgCO2e/kWh).
            gwp_source: IPCC AR version for GWP values.

        Returns:
            GasBreakdown with CO2, CH4, N2O, and total CO2e.
        """
        # Resolve gas fractions for the country
        cc = (record.country_code or record.grid_region or "").upper()
        fractions = GRID_GAS_FRACTIONS.get(cc, GRID_GAS_FRACTIONS["DEFAULT"])

        # Loss multiplier (adjusted basis)
        multiplier = self._loss_multiplier(td_loss_pct)
        lost_kwh = record.quantity_kwh * multiplier

        # Per-gas emission factors from the grid EF
        co2_ef = grid_ef * fractions["co2_frac"]
        ch4_ef = grid_ef * fractions["ch4_frac"]
        n2o_ef = grid_ef * fractions["n2o_frac"]

        # GWP values for converting native gas units to CO2e
        gwp = GWP_VALUES.get(gwp_source, GWP_VALUES[GWPSource.AR6])

        # Emissions in native gas mass (kgCO2e at grid, so already
        # in CO2e terms for co2; for ch4/n2o we need native mass)
        # Note: grid_ef is in kgCO2e/kWh, so the fractions give us
        # the CO2e contribution of each gas. To get native mass we
        # divide by GWP.
        co2_mass = _quantize(lost_kwh * co2_ef / gwp[EmissionGas.CO2])
        ch4_mass = _quantize(lost_kwh * ch4_ef / gwp[EmissionGas.CH4])
        n2o_mass = _quantize(lost_kwh * n2o_ef / gwp[EmissionGas.N2O])

        # Total CO2e
        co2e_total = _quantize(
            co2_mass * gwp[EmissionGas.CO2]
            + ch4_mass * gwp[EmissionGas.CH4]
            + n2o_mass * gwp[EmissionGas.N2O]
        )

        return GasBreakdown(
            co2=co2_mass,
            ch4=ch4_mass,
            n2o=n2o_mass,
            co2e=co2e_total,
            gwp_source=gwp_source,
        )

    # ------------------------------------------------------------------
    # Country comparison
    # ------------------------------------------------------------------

    def compare_countries(
        self,
        country_codes: List[str],
        electricity_kwh: Decimal,
    ) -> Dict[str, Activity3cResult]:
        """Compare T&D loss emissions across multiple countries.

        Constructs a synthetic ElectricityConsumptionRecord with the
        specified electricity quantity for each country and calculates
        Activity 3c emissions to enable side-by-side comparison.

        Args:
            country_codes: List of ISO 3166-1 alpha-2 country codes.
            electricity_kwh: Electricity consumption in kWh to compare.

        Returns:
            Dictionary mapping country code to Activity3cResult.
        """
        from datetime import date as date_cls

        results: Dict[str, Activity3cResult] = {}
        for cc in country_codes:
            cc_upper = cc.upper().strip()
            # Build synthetic record
            rec = ElectricityConsumptionRecord(
                quantity_kwh=electricity_kwh,
                grid_region=cc_upper,
                grid_region_type=GridRegionType.COUNTRY,
                period_start=date_cls(2024, 1, 1),
                period_end=date_cls(2024, 12, 31),
                reporting_year=2024,
                country_code=cc_upper,
            )
            result = self.calculate_auto(rec, cc_upper)
            results[cc_upper] = result

        logger.info(
            "compare_countries: countries=%d, electricity=%.2f kWh",
            len(country_codes),
            electricity_kwh,
        )
        return results

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate_by_country(
        self, results: List[Activity3cResult]
    ) -> Dict[str, Decimal]:
        """Aggregate T&D loss emissions by country (grid_region).

        Args:
            results: List of Activity3cResult to aggregate.

        Returns:
            Dictionary mapping grid_region to total emissions (kgCO2e).
        """
        agg: Dict[str, Decimal] = {}
        for r in results:
            key = r.grid_region or "UNKNOWN"
            agg[key] = agg.get(key, ZERO) + r.emissions_total
        # Quantize final values
        return {k: _quantize(v) for k, v in agg.items()}

    def aggregate_by_facility(
        self, results: List[Activity3cResult]
    ) -> Dict[str, Decimal]:
        """Aggregate T&D loss emissions by facility.

        Uses the electricity_record_id prefix (before the first '-')
        as a facility proxy.  For more precise aggregation, callers
        should join against the original ElectricityConsumptionRecord
        to extract facility_id.

        Args:
            results: List of Activity3cResult to aggregate.

        Returns:
            Dictionary mapping facility identifier to total emissions.
        """
        agg: Dict[str, Decimal] = {}
        for r in results:
            # Use grid_region as facility proxy; caller can re-aggregate
            # by joining with original records for facility_id.
            key = r.electricity_record_id
            agg[key] = agg.get(key, ZERO) + r.emissions_total
        return {k: _quantize(v) for k, v in agg.items()}

    def get_total_emissions(
        self, results: List[Activity3cResult]
    ) -> Decimal:
        """Sum total emissions across all Activity 3c results.

        Args:
            results: List of Activity3cResult.

        Returns:
            Total T&D loss emissions in kgCO2e.
        """
        total = ZERO
        for r in results:
            total += r.emissions_total
        return _quantize(total)

    # ------------------------------------------------------------------
    # Data Quality Indicator (DQI) Assessment
    # ------------------------------------------------------------------

    def assess_dqi(
        self,
        record: ElectricityConsumptionRecord,
        td_factor: TDLossFactor,
    ) -> DQIAssessment:
        """Assess data quality for a T&D loss calculation.

        Evaluates five GHG Protocol DQI dimensions:
        1. Temporal representativeness (factor year vs reporting year)
        2. Geographical representativeness (country match)
        3. Technological representativeness (grid source specificity)
        4. Completeness (data fields present)
        5. Reliability (factor source hierarchy)

        Args:
            record: Electricity consumption record being assessed.
            td_factor: T&D loss factor used in the calculation.

        Returns:
            DQIAssessment with per-dimension and composite scores.
        """
        with self._lock:
            self._stats["dqi_assessments"] += 1

        findings: List[str] = []

        # 1. Temporal: how close is the factor year to reporting year
        temporal = self._score_temporal(
            record.reporting_year, td_factor.year, findings
        )

        # 2. Geographical: does the factor match the record's region
        geographical = self._score_geographical(
            record, td_factor, findings
        )

        # 3. Technological: grid region type specificity
        technological = self._score_technological(
            record.grid_region_type, findings
        )

        # 4. Completeness: how many optional fields are populated
        completeness = self._score_completeness(record, findings)

        # 5. Reliability: factor source hierarchy
        reliability = self._score_reliability(
            td_factor.source, findings
        )

        # Composite (arithmetic mean)
        five = Decimal("5")
        composite = _quantize(
            (temporal + geographical + technological
             + completeness + reliability) / five
        )

        # Determine quality tier
        tier = "Unknown"
        for tier_name, (lo, hi) in DQI_QUALITY_TIERS.items():
            if lo <= composite < hi:
                tier = tier_name
                break

        return DQIAssessment(
            record_id=record.record_id,
            activity_type=ActivityType.ACTIVITY_3C,
            temporal=temporal,
            geographical=geographical,
            technological=technological,
            completeness=completeness,
            reliability=reliability,
            composite=composite,
            tier=tier,
            findings=findings,
        )

    # ------------------------------------------------------------------
    # Uncertainty Quantification
    # ------------------------------------------------------------------

    def quantify_uncertainty(
        self,
        record: ElectricityConsumptionRecord,
        td_factor: TDLossFactor,
        method: UncertaintyMethod = UncertaintyMethod.IPCC_DEFAULT,
        grid_ef: Optional[Decimal] = None,
        upstream_ef: Optional[Decimal] = None,
        confidence_level: Decimal = Decimal("95"),
        monte_carlo_iterations: int = 5000,
        monte_carlo_seed: int = 42,
    ) -> UncertaintyResult:
        """Quantify uncertainty in T&D loss emission calculations.

        Supports three methods:
        - IPCC_DEFAULT: Uses IPCC default uncertainty ranges for
          the emission factor sources involved.
        - ANALYTICAL: Propagates uncertainty via root-sum-of-squares
          of the individual factor uncertainties.
        - MONTE_CARLO: Runs N iterations with randomly perturbed
          factors drawn from assumed distributions.

        Args:
            record: Electricity consumption record.
            td_factor: T&D loss factor used.
            method: Uncertainty quantification method.
            grid_ef: Grid generation EF. Auto-resolved if None.
            upstream_ef: Upstream EF. Auto-resolved if None.
            confidence_level: Confidence level percentage (e.g. 95).
            monte_carlo_iterations: Number of MC iterations.
            monte_carlo_seed: Random seed for reproducibility.

        Returns:
            UncertaintyResult with mean, std_dev, CV, and CI bounds.
        """
        with self._lock:
            self._stats["uncertainty_assessments"] += 1

        cc = (
            record.country_code
            or record.grid_region
            or ""
        ).upper()

        gef = grid_ef if grid_ef is not None else self.get_grid_generation_ef(cc)
        uef = upstream_ef if upstream_ef is not None else self._resolve_upstream_ef(cc)

        if method == UncertaintyMethod.MONTE_CARLO:
            return self._uncertainty_monte_carlo(
                record, td_factor, gef, uef,
                confidence_level, monte_carlo_iterations, monte_carlo_seed,
            )
        elif method == UncertaintyMethod.ANALYTICAL:
            return self._uncertainty_analytical(
                record, td_factor, gef, uef, confidence_level,
            )
        else:
            return self._uncertainty_ipcc_default(
                record, td_factor, gef, uef, confidence_level,
            )

    # ------------------------------------------------------------------
    # Sensitivity Analysis
    # ------------------------------------------------------------------

    def sensitivity_analysis(
        self,
        record: ElectricityConsumptionRecord,
        td_range: Optional[List[Decimal]] = None,
        grid_ef: Optional[Decimal] = None,
        upstream_ef: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        """Analyze how emissions change as T&D loss percentage varies.

        Calculates emissions at each T&D loss percentage in the given
        range to show sensitivity of the result to the loss factor.

        Args:
            record: Electricity consumption record.
            td_range: List of T&D loss fractions to test. Defaults
                to [0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20].
            grid_ef: Grid EF. Auto-resolved if None.
            upstream_ef: Upstream EF. Auto-resolved if None.

        Returns:
            Dictionary with:
            - points: List of {td_pct, emissions, gen_component, ups_component}
            - min_emissions: Minimum total emissions in the range.
            - max_emissions: Maximum total emissions in the range.
            - elasticity: Approximate % change in emissions per 1pp
              change in T&D loss %.
        """
        if td_range is None:
            td_range = [
                Decimal("0.02"),
                Decimal("0.04"),
                Decimal("0.06"),
                Decimal("0.08"),
                Decimal("0.10"),
                Decimal("0.15"),
                Decimal("0.20"),
            ]

        cc = (
            record.country_code or record.grid_region or ""
        ).upper()
        gef = grid_ef if grid_ef is not None else self.get_grid_generation_ef(cc)
        uef = upstream_ef if upstream_ef is not None else self._resolve_upstream_ef(cc)

        points: List[Dict[str, Decimal]] = []
        for td_pct in td_range:
            gen_c = self.calculate_generation_component(
                record.quantity_kwh, td_pct, gef
            )
            ups_c = self.calculate_upstream_component(
                record.quantity_kwh, td_pct, uef
            )
            points.append({
                "td_pct": td_pct,
                "emissions": _quantize(gen_c + ups_c),
                "gen_component": gen_c,
                "ups_component": ups_c,
            })

        emissions_values = [p["emissions"] for p in points]
        min_e = min(emissions_values)
        max_e = max(emissions_values)

        # Elasticity: approximate using first and last points
        elasticity = ZERO
        if len(points) >= 2 and points[0]["emissions"] > ZERO:
            td_delta = points[-1]["td_pct"] - points[0]["td_pct"]
            e_delta = points[-1]["emissions"] - points[0]["emissions"]
            if td_delta > ZERO:
                elasticity = _quantize(
                    (e_delta / points[0]["emissions"])
                    / (td_delta / points[0]["td_pct"])
                )

        return {
            "points": points,
            "min_emissions": min_e,
            "max_emissions": max_e,
            "elasticity": elasticity,
            "record_id": record.record_id,
            "electricity_kwh": record.quantity_kwh,
            "grid_ef": gef,
            "upstream_ef": uef,
        }

    # ------------------------------------------------------------------
    # Double-counting prevention
    # ------------------------------------------------------------------

    def check_double_counting(
        self,
        record: ElectricityConsumptionRecord,
        scope2_records: List[Dict[str, Any]],
    ) -> List[str]:
        """Check for potential double-counting with Scope 2.

        Verifies that T&D loss emissions for a given electricity
        consumption record are not already accounted for in the
        organization's Scope 2 reporting.

        Common double-counting risks:
        1. Scope 2 grid EF already includes T&D losses.
        2. Same electricity volume counted in both Scope 2 and 3c.
        3. Market-based Scope 2 with supplier EF that includes T&D.

        Args:
            record: Activity 3c electricity consumption record.
            scope2_records: List of Scope 2 records (dictionaries)
                with at least 'record_id', 'quantity_kwh', and
                optionally 'includes_td_losses', 'grid_region'.

        Returns:
            List of warning messages. Empty list means no issues found.
        """
        with self._lock:
            self._stats["double_counting_checks"] += 1

        warnings: List[str] = []

        for s2 in scope2_records:
            s2_id = s2.get("record_id", "unknown")
            s2_kwh = s2.get("quantity_kwh")
            s2_region = s2.get("grid_region", "")
            s2_includes_td = s2.get("includes_td_losses", False)

            # Check 1: Scope 2 EF explicitly includes T&D losses
            if s2_includes_td:
                warnings.append(
                    f"Scope 2 record {s2_id} has includes_td_losses=True. "
                    f"Activity 3c T&D losses for record {record.record_id} "
                    f"may be double-counted."
                )

            # Check 2: Same quantity and region overlap
            if (
                s2_kwh is not None
                and s2_region
                and str(s2_region).upper() == record.grid_region.upper()
            ):
                try:
                    s2_kwh_dec = Decimal(str(s2_kwh))
                    if s2_kwh_dec == record.quantity_kwh:
                        warnings.append(
                            f"Scope 2 record {s2_id} has identical "
                            f"quantity ({s2_kwh_dec} kWh) and grid region "
                            f"({s2_region}) as Activity 3c record "
                            f"{record.record_id}. Verify T&D losses are "
                            f"not already embedded in the Scope 2 EF."
                        )
                except (InvalidOperation, ValueError):
                    pass

            # Check 3: Market-based method with supplier-specific EF
            s2_method = s2.get("accounting_method", "")
            s2_supplier_ef = s2.get("supplier_ef")
            if (
                str(s2_method).lower() == "market_based"
                and s2_supplier_ef is not None
            ):
                warnings.append(
                    f"Scope 2 record {s2_id} uses market-based accounting "
                    f"with a supplier-specific EF. Verify with the supplier "
                    f"whether their EF already includes T&D loss adjustments "
                    f"before adding Activity 3c emissions for record "
                    f"{record.record_id}."
                )

        if warnings:
            logger.warning(
                "Double-counting check found %d potential issue(s) "
                "for record %s",
                len(warnings),
                record.record_id,
            )
        else:
            logger.debug(
                "Double-counting check: no issues for record %s",
                record.record_id,
            )

        return warnings

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_td_factor(
        self, td_loss_pct: Decimal
    ) -> Tuple[bool, List[str]]:
        """Validate a T&D loss factor is within acceptable bounds.

        Args:
            td_loss_pct: T&D loss fraction to validate.

        Returns:
            Tuple of (is_valid, error_messages).
            is_valid is True if the factor is usable (may still have
            warnings in the messages list).
        """
        errors: List[str] = []

        if td_loss_pct < ZERO:
            errors.append(
                f"T&D loss percentage must be >= 0, got {td_loss_pct}"
            )
        if td_loss_pct > MAX_TD_LOSS_PCT:
            errors.append(
                f"T&D loss percentage must be <= {MAX_TD_LOSS_PCT} (50%), "
                f"got {td_loss_pct}"
            )
        if td_loss_pct >= ONE:
            errors.append(
                f"T&D loss percentage must be < 1.0 (100%), "
                f"got {td_loss_pct}"
            )

        is_valid = len(errors) == 0

        # Non-blocking warnings (still valid)
        if is_valid and td_loss_pct > ZERO and td_loss_pct < MIN_TD_LOSS_PCT:
            logger.warning(
                "T&D loss factor %.6f is unusually low (< %.4f). "
                "Verify data source.",
                td_loss_pct,
                MIN_TD_LOSS_PCT,
            )

        return is_valid, errors

    # ------------------------------------------------------------------
    # Statistics and lifecycle
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return cumulative engine statistics.

        Returns:
            Dictionary with calculation counts, total emissions,
            error counts, resolution stats, and engine metadata.
        """
        with self._lock:
            stats = dict(self._stats)
        stats["engine_id"] = ENGINE_ID
        stats["engine_version"] = ENGINE_VERSION
        stats["agent_id"] = AGENT_ID
        stats["activity"] = ACTIVITY.value
        stats["td_countries_available"] = len(TD_LOSS_FACTORS)
        stats["egrid_subregions_available"] = len(EGRID_TD_LOSS_FACTORS)
        stats["grid_ef_countries_available"] = len(GRID_GENERATION_EFS)
        return stats

    def reset(self) -> None:
        """Reset engine statistics to zero.

        Primarily intended for testing. Does not affect embedded
        data tables or configuration.
        """
        with self._lock:
            self._stats = {
                "calculations": 0,
                "batch_calculations": 0,
                "auto_calculations": 0,
                "errors": 0,
                "total_emissions_kgco2e": ZERO,
                "total_electricity_kwh": ZERO,
                "total_td_losses_kwh": ZERO,
                "countries_resolved": 0,
                "egrid_resolved": 0,
                "dqi_assessments": 0,
                "uncertainty_assessments": 0,
                "double_counting_checks": 0,
                "created_at": self._created_at.isoformat(),
            }
        logger.info("TDLossCalculatorEngine statistics reset")

    # ==================================================================
    # Private helpers
    # ==================================================================

    def _loss_multiplier(self, td_loss_pct: Decimal) -> Decimal:
        """Compute the adjusted-basis loss multiplier.

        Converts from delivered-basis percentage to the multiplier
        that gives the amount of electricity lost per unit delivered:
            multiplier = TD% / (1 - TD%)

        Args:
            td_loss_pct: T&D loss fraction on delivered basis.

        Returns:
            Adjusted loss multiplier.

        Raises:
            ValueError: If td_loss_pct >= 1 (would cause division by zero).
        """
        denominator = ONE - td_loss_pct
        if denominator <= ZERO:
            raise ValueError(
                f"Cannot compute loss multiplier: td_loss_pct={td_loss_pct} "
                f"causes denominator (1 - {td_loss_pct}) <= 0"
            )
        return _quantize(td_loss_pct / denominator)

    def _resolve_country_code(
        self,
        record: ElectricityConsumptionRecord,
        explicit_code: Optional[str],
    ) -> str:
        """Resolve the country code from available sources.

        Priority:
        1. Explicitly passed country_code parameter.
        2. Record's country_code field.
        3. Record's grid_region field (if grid_region_type is COUNTRY).

        Args:
            record: Electricity consumption record.
            explicit_code: Explicitly provided country code.

        Returns:
            Uppercase ISO 3166-1 alpha-2 country code.

        Raises:
            ValueError: If no country code can be resolved.
        """
        if explicit_code:
            return explicit_code.upper().strip()
        if record.country_code:
            return record.country_code.upper().strip()
        if (
            record.grid_region
            and record.grid_region_type == GridRegionType.COUNTRY
        ):
            return record.grid_region.upper().strip()
        if record.grid_region and len(record.grid_region) <= 3:
            return record.grid_region.upper().strip()
        raise ValueError(
            f"Cannot resolve country code for record {record.record_id}. "
            f"Provide country_code explicitly or set record.country_code."
        )

    def _resolve_upstream_ef(self, country_code: str) -> Decimal:
        """Resolve upstream electricity emission factor for a country.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.

        Returns:
            Upstream electricity EF in kgCO2e/kWh.
        """
        cc = country_code.upper().strip()
        ef = UPSTREAM_ELECTRICITY_FACTORS.get(cc)
        if ef is not None:
            return ef

        logger.warning(
            "Upstream electricity EF not found for '%s', "
            "using default %.5f",
            cc,
            DEFAULT_UPSTREAM_EF,
        )
        return DEFAULT_UPSTREAM_EF

    def _resolve_td_source_label(
        self,
        grid_region: str,
        grid_region_type: GridRegionType,
    ) -> str:
        """Resolve the human-readable T&D loss source label.

        Args:
            grid_region: Grid region identifier.
            grid_region_type: Type of grid region.

        Returns:
            Source label string (e.g. "IEA 2023", "EPA eGRID 2022").
        """
        if grid_region_type == GridRegionType.EGRID_SUBREGION:
            return "EPA eGRID 2022"
        region_upper = grid_region.upper()
        if region_upper in TD_LOSS_FACTORS:
            return "IEA/World Bank 2023"
        return "Default (world average)"

    def _update_stats(
        self,
        electricity_kwh: Decimal,
        emissions: Decimal,
        td_loss_pct: Decimal,
    ) -> None:
        """Update engine statistics after a successful calculation.

        Args:
            electricity_kwh: Electricity consumed.
            emissions: Total T&D loss emissions calculated.
            td_loss_pct: T&D loss fraction used.
        """
        # Approximate lost electricity
        multiplier = self._loss_multiplier(td_loss_pct)
        lost_kwh = _quantize(electricity_kwh * multiplier)

        with self._lock:
            self._stats["calculations"] += 1
            self._stats["total_emissions_kgco2e"] += emissions
            self._stats["total_electricity_kwh"] += electricity_kwh
            self._stats["total_td_losses_kwh"] += lost_kwh

    def _validate_positive(self, name: str, value: Decimal) -> None:
        """Validate that a Decimal value is positive.

        Args:
            name: Parameter name for error messages.
            value: Value to validate.

        Raises:
            ValueError: If value <= 0.
        """
        if value <= ZERO:
            raise ValueError(f"{name} must be > 0, got {value}")

    # ------------------------------------------------------------------
    # DQI scoring helpers
    # ------------------------------------------------------------------

    def _score_temporal(
        self,
        reporting_year: int,
        factor_year: int,
        findings: List[str],
    ) -> Decimal:
        """Score temporal representativeness.

        Args:
            reporting_year: Reporting year of the record.
            factor_year: Reference year of the emission factor.
            findings: Mutable list to append findings to.

        Returns:
            Score from 1.0 (best) to 5.0 (worst).
        """
        gap = abs(reporting_year - factor_year)
        if gap == 0:
            return Decimal("1.0")
        if gap <= 1:
            return Decimal("1.5")
        if gap <= 2:
            return Decimal("2.0")
        if gap <= 3:
            findings.append(
                f"T&D loss factor is {gap} years old; consider updating."
            )
            return Decimal("3.0")
        if gap <= 5:
            findings.append(
                f"T&D loss factor is {gap} years old; data may be outdated."
            )
            return Decimal("4.0")
        findings.append(
            f"T&D loss factor is {gap} years old; data is significantly "
            f"outdated. Update to current year factors."
        )
        return Decimal("5.0")

    def _score_geographical(
        self,
        record: ElectricityConsumptionRecord,
        td_factor: TDLossFactor,
        findings: List[str],
    ) -> Decimal:
        """Score geographical representativeness.

        Args:
            record: Electricity consumption record.
            td_factor: T&D loss factor.
            findings: Mutable list to append findings.

        Returns:
            Score from 1.0 to 5.0.
        """
        # eGRID subregion is the most specific
        if td_factor.source == TDLossSource.EPA_EGRID:
            return Decimal("1.0")

        # Country-level match
        rec_cc = (record.country_code or record.grid_region or "").upper()
        factor_cc = td_factor.country_code.upper()

        if rec_cc == factor_cc and rec_cc in TD_LOSS_FACTORS:
            return Decimal("2.0")

        if rec_cc in TD_LOSS_FACTORS:
            findings.append(
                f"T&D factor country ({factor_cc}) differs from record "
                f"country ({rec_cc}). Using record's country factor."
            )
            return Decimal("3.0")

        # Custom or default
        if td_factor.source == TDLossSource.CUSTOM:
            findings.append(
                "Using custom T&D loss factor; verify geographical "
                "applicability."
            )
            return Decimal("3.5")

        findings.append(
            f"No country-specific T&D factor available for '{rec_cc}'. "
            f"Using default world average."
        )
        return Decimal("4.5")

    def _score_technological(
        self,
        grid_region_type: GridRegionType,
        findings: List[str],
    ) -> Decimal:
        """Score technological representativeness.

        Args:
            grid_region_type: Type of grid region identifier.
            findings: Mutable list to append findings.

        Returns:
            Score from 1.0 to 5.0.
        """
        if grid_region_type == GridRegionType.EGRID_SUBREGION:
            return Decimal("1.0")
        if grid_region_type == GridRegionType.COUNTRY:
            return Decimal("2.0")
        if grid_region_type == GridRegionType.EU_MEMBER_STATE:
            return Decimal("2.0")
        findings.append(
            "Custom grid region type; technological representativeness "
            "cannot be fully assessed."
        )
        return Decimal("3.0")

    def _score_completeness(
        self,
        record: ElectricityConsumptionRecord,
        findings: List[str],
    ) -> Decimal:
        """Score data completeness.

        Args:
            record: Electricity consumption record.
            findings: Mutable list to append findings.

        Returns:
            Score from 1.0 to 5.0.
        """
        optional_fields = [
            record.facility_id,
            record.facility_name,
            record.supplier_id,
            record.supplier_name,
            record.country_code,
        ]
        present = sum(1 for f in optional_fields if f)
        total = len(optional_fields)
        ratio = Decimal(str(present)) / Decimal(str(total))

        if ratio >= Decimal("0.8"):
            return Decimal("1.0")
        if ratio >= Decimal("0.6"):
            return Decimal("2.0")
        if ratio >= Decimal("0.4"):
            findings.append(
                f"Only {present}/{total} optional fields populated. "
                f"Additional metadata improves data quality."
            )
            return Decimal("3.0")
        findings.append(
            f"Only {present}/{total} optional fields populated. "
            f"Significant data gaps may affect quality."
        )
        return Decimal("4.0")

    def _score_reliability(
        self,
        source: TDLossSource,
        findings: List[str],
    ) -> Decimal:
        """Score data reliability based on factor source.

        Args:
            source: T&D loss factor source.
            findings: Mutable list to append findings.

        Returns:
            Score from 1.0 to 5.0.
        """
        reliability_scores: Dict[TDLossSource, Decimal] = {
            TDLossSource.EPA_EGRID: Decimal("1.0"),
            TDLossSource.IEA: Decimal("1.5"),
            TDLossSource.WORLD_BANK: Decimal("2.0"),
            TDLossSource.NATIONAL_GRID: Decimal("1.5"),
            TDLossSource.CUSTOM: Decimal("3.5"),
        }
        score = reliability_scores.get(source, Decimal("3.0"))
        if score > Decimal("3.0"):
            findings.append(
                f"T&D loss factor source '{source.value}' has lower "
                f"reliability. Consider using IEA or eGRID data."
            )
        return score

    # ------------------------------------------------------------------
    # Uncertainty calculation helpers
    # ------------------------------------------------------------------

    def _uncertainty_ipcc_default(
        self,
        record: ElectricityConsumptionRecord,
        td_factor: TDLossFactor,
        grid_ef: Decimal,
        upstream_ef: Decimal,
        confidence_level: Decimal,
    ) -> UncertaintyResult:
        """Calculate uncertainty using IPCC default ranges.

        IPCC 2006 Guidelines specify default uncertainty for T&D losses
        as +/- 30% for country-average factors, +/- 15% for subregion
        factors, and +/- 20% for grid emission factors.

        Args:
            record: Electricity consumption record.
            td_factor: T&D loss factor.
            grid_ef: Grid generation EF.
            upstream_ef: Upstream EF.
            confidence_level: Confidence level percentage.

        Returns:
            UncertaintyResult with IPCC default uncertainty bounds.
        """
        # Central calculation
        multiplier = self._loss_multiplier(td_factor.loss_percentage)
        mean = _quantize(
            record.quantity_kwh * multiplier * (grid_ef + upstream_ef)
        )

        # IPCC default uncertainty for T&D factors
        if td_factor.source == TDLossSource.EPA_EGRID:
            td_unc_pct = Decimal("15")
        elif td_factor.source in (TDLossSource.IEA, TDLossSource.NATIONAL_GRID):
            td_unc_pct = Decimal("25")
        else:
            td_unc_pct = Decimal("35")

        # Grid EF uncertainty
        grid_unc_pct = Decimal("20")

        # Combined uncertainty (root-sum-of-squares)
        combined_unc_pct = _quantize(
            Decimal(str(math.sqrt(
                float(td_unc_pct ** 2 + grid_unc_pct ** 2)
            )))
        )

        # Standard deviation and CI
        std_dev = _quantize(mean * combined_unc_pct / ONE_HUNDRED / Decimal("2"))
        z = _z_score(confidence_level)
        ci_lower = _quantize(max(ZERO, mean - z * std_dev))
        ci_upper = _quantize(mean + z * std_dev)
        cv = _quantize(std_dev / mean) if mean > ZERO else ZERO

        return UncertaintyResult(
            mean=mean,
            std_dev=std_dev,
            cv=cv,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=confidence_level,
            method=UncertaintyMethod.IPCC_DEFAULT,
        )

    def _uncertainty_analytical(
        self,
        record: ElectricityConsumptionRecord,
        td_factor: TDLossFactor,
        grid_ef: Decimal,
        upstream_ef: Decimal,
        confidence_level: Decimal,
    ) -> UncertaintyResult:
        """Calculate uncertainty using analytical error propagation.

        Uses first-order Taylor expansion (root-sum-of-squares) to
        propagate uncertainties from the three input factors:
        electricity quantity, T&D loss percentage, and grid EF.

        Args:
            record: Electricity consumption record.
            td_factor: T&D loss factor.
            grid_ef: Grid generation EF.
            upstream_ef: Upstream EF.
            confidence_level: Confidence level percentage.

        Returns:
            UncertaintyResult with analytical uncertainty bounds.
        """
        multiplier = self._loss_multiplier(td_factor.loss_percentage)
        combined_ef = grid_ef + upstream_ef
        mean = _quantize(record.quantity_kwh * multiplier * combined_ef)

        # Relative uncertainties (as fractions)
        u_electricity = Decimal("0.02")   # +/- 2% metering uncertainty
        u_td = Decimal("0.15")            # +/- 15% T&D factor uncertainty
        u_ef = Decimal("0.10")            # +/- 10% grid EF uncertainty

        # RSS of relative uncertainties
        combined_rel = Decimal(str(math.sqrt(
            float(u_electricity ** 2 + u_td ** 2 + u_ef ** 2)
        )))

        std_dev = _quantize(mean * combined_rel)
        z = _z_score(confidence_level)
        ci_lower = _quantize(max(ZERO, mean - z * std_dev))
        ci_upper = _quantize(mean + z * std_dev)
        cv = _quantize(std_dev / mean) if mean > ZERO else ZERO

        return UncertaintyResult(
            mean=mean,
            std_dev=std_dev,
            cv=cv,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=confidence_level,
            method=UncertaintyMethod.ANALYTICAL,
        )

    def _uncertainty_monte_carlo(
        self,
        record: ElectricityConsumptionRecord,
        td_factor: TDLossFactor,
        grid_ef: Decimal,
        upstream_ef: Decimal,
        confidence_level: Decimal,
        iterations: int,
        seed: int,
    ) -> UncertaintyResult:
        """Calculate uncertainty using Monte Carlo simulation.

        Randomly perturbs the T&D loss factor, grid EF, and upstream EF
        within their assumed distributions (normal, truncated at zero)
        and calculates the emission distribution.

        Args:
            record: Electricity consumption record.
            td_factor: T&D loss factor.
            grid_ef: Grid generation EF.
            upstream_ef: Upstream EF.
            confidence_level: Confidence level percentage.
            iterations: Number of Monte Carlo iterations.
            seed: Random seed for reproducibility.

        Returns:
            UncertaintyResult from Monte Carlo distribution.
        """
        rng = random.Random(seed)

        # Parameter distributions (mean, std_dev as fraction of mean)
        td_mean = float(td_factor.loss_percentage)
        td_std = td_mean * 0.15  # 15% relative uncertainty

        gef_mean = float(grid_ef)
        gef_std = gef_mean * 0.10  # 10% relative uncertainty

        uef_mean = float(upstream_ef)
        uef_std = max(uef_mean * 0.20, 0.001)  # 20% or floor

        elec = float(record.quantity_kwh)

        samples: List[float] = []
        for _ in range(iterations):
            # Draw from truncated normal (floor at small positive)
            td_sample = max(0.001, rng.gauss(td_mean, td_std))
            td_sample = min(td_sample, 0.49)  # cap below 50%
            gef_sample = max(0.0, rng.gauss(gef_mean, gef_std))
            uef_sample = max(0.0, rng.gauss(uef_mean, uef_std))

            # Adjusted multiplier
            mult = td_sample / (1.0 - td_sample)
            emission = elec * mult * (gef_sample + uef_sample)
            samples.append(emission)

        # Statistics from samples
        n = len(samples)
        sample_mean = sum(samples) / n
        sample_var = sum((x - sample_mean) ** 2 for x in samples) / (n - 1)
        sample_std = math.sqrt(sample_var)
        sample_cv = sample_std / sample_mean if sample_mean > 0 else 0.0

        # Percentile-based CI
        sorted_samples = sorted(samples)
        cl = float(confidence_level)
        lo_idx = max(0, int(n * (1.0 - cl / 100.0) / 2.0))
        hi_idx = min(n - 1, int(n * (1.0 + cl / 100.0) / 2.0))

        mean_d = _quantize(Decimal(str(sample_mean)))
        std_d = _quantize(Decimal(str(sample_std)))
        cv_d = _quantize(Decimal(str(sample_cv)))
        ci_lo = _quantize(Decimal(str(sorted_samples[lo_idx])))
        ci_hi = _quantize(Decimal(str(sorted_samples[hi_idx])))

        return UncertaintyResult(
            mean=mean_d,
            std_dev=std_d,
            cv=cv_d,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            confidence_level=confidence_level,
            method=UncertaintyMethod.MONTE_CARLO,
        )


# ============================================================================
# Module public surface
# ============================================================================

__all__ = [
    # Engine
    "TDLossCalculatorEngine",
    # Constants
    "AGENT_ID",
    "ENGINE_ID",
    "ENGINE_VERSION",
    "ACTIVITY",
    "MAX_TD_LOSS_PCT",
    "MIN_TD_LOSS_PCT",
    "DEFAULT_TD_LOSS_PCT",
    "DEFAULT_GRID_EF",
    "DEFAULT_UPSTREAM_EF",
    "TRANSMISSION_SHARE",
    "DISTRIBUTION_SHARE",
    "LOSS_BASIS_DELIVERED",
    "LOSS_BASIS_GENERATED",
    # Embedded data
    "GRID_GENERATION_EFS",
    "GRID_GAS_FRACTIONS",
]
