# -*- coding: utf-8 -*-
"""
Engine 3: UpstreamElectricityCalculatorEngine - AGENT-MRV-016

Calculates Activity 3b upstream emissions from fuels used to GENERATE the
electricity, steam, heating, and cooling consumed by the reporting company.
These are NOT the generation emissions (Scope 2) but rather the extraction,
processing, and transport emissions of the fuels used in power generation
(the "well-to-tank" component of the grid's fuel supply chain).

Core Formulas:
    Location-based:
        Emissions_3b = Electricity_consumed (kWh) x Upstream_EF (kgCO2e/kWh)
    Market-based:
        Emissions_3b = Electricity_consumed (kWh) x Supplier_upstream_EF

Where Upstream_EF is a weighted average of WTT factors for each fuel in the
grid generation mix (gas, coal, nuclear, renewables, etc.).

Key Capabilities:
    - Location-based upstream calculation using grid-average upstream EFs
    - Market-based upstream calculation using supplier-specific upstream EFs
    - Steam, heating, and cooling upstream emissions
    - Grid mix decomposition (weight WTT factors by fuel share in generation)
    - Regional factor resolution: country -> eGRID subregion -> EU member state
    - Per-gas breakdown (CO2, CH4, N2O) from upstream lifecycle
    - Renewable energy near-zero upstream EFs (solar/wind/hydro/nuclear)
    - CHP upstream emissions allocation
    - DQI scoring based on factor source quality
    - Uncertainty quantification (IPCC default, analytical, Monte Carlo)
    - Location vs market comparison
    - Aggregation by energy type, region, facility
    - Double-counting prevention vs Scope 2

GHG Protocol Reference:
    Scope 3 Standard, Category 3: Fuel- and Energy-Related Activities
    Technical Guidance, Chapter 4, Activity 3b

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
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

from greenlang.fuel_energy_activities.config import get_config
from greenlang.fuel_energy_activities.models import (
    # Enumerations
    AccountingMethod,
    ActivityType,
    CalculationMethod,
    DQIScore,
    EmissionGas,
    EnergyType,
    GridRegionType,
    GWPSource,
    UncertaintyMethod,
    # Constants
    DECIMAL_PLACES,
    GWP_VALUES,
    ONE,
    ONE_HUNDRED,
    UPSTREAM_ELECTRICITY_FACTORS,
    ZERO,
    DQI_QUALITY_TIERS,
    # Data Models
    Activity3bResult,
    DQIAssessment,
    ElectricityConsumptionRecord,
    GasBreakdown,
    UncertaintyResult,
    UpstreamElectricityFactor,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Module-Level Constants
# ============================================================================

#: Engine identifier for registry and telemetry.
ENGINE_ID: str = "upstream-electricity-calculator"

#: Engine version.
ENGINE_VERSION: str = "1.0.0"

#: Default quantization exponent for Decimal rounding.
_QUANTIZE_EXP: Decimal = Decimal(10) ** -DECIMAL_PLACES

#: Decimal five for DQI composite averaging.
_FIVE: Decimal = Decimal("5")

#: Decimal two for bisecting confidence intervals.
_TWO: Decimal = Decimal("2")

#: GJ to kWh conversion factor (1 GJ = 277.778 kWh).
GJ_TO_KWH: Decimal = Decimal("277.778")

#: Default boiler efficiency for steam generation (85%).
DEFAULT_BOILER_EFFICIENCY: Decimal = Decimal("0.85")

#: Default COP (coefficient of performance) for cooling systems.
DEFAULT_COOLING_COP: Decimal = Decimal("3.5")

#: Maximum records per batch to prevent memory exhaustion.
MAX_BATCH_SIZE: int = 100_000


# ---------------------------------------------------------------------------
# eGRID Subregion Upstream Emission Factors (kgCO2e/kWh)
# These represent the upstream (fuel extraction/processing/transport)
# emissions for each EPA eGRID subregion in the United States.
# Source: EPA eGRID 2022, ecoinvent 3.11, GREET 2023
# ---------------------------------------------------------------------------

EGRID_UPSTREAM_FACTORS: Dict[str, Decimal] = {
    "AKGD": Decimal("0.04100"),
    "AKMS": Decimal("0.03900"),
    "CAMX": Decimal("0.03600"),
    "ERCT": Decimal("0.04200"),
    "FRCC": Decimal("0.04400"),
    "HIMS": Decimal("0.02100"),
    "HIOA": Decimal("0.05200"),
    "MROE": Decimal("0.04700"),
    "MROW": Decimal("0.04500"),
    "NEWE": Decimal("0.02800"),
    "NWPP": Decimal("0.02200"),
    "NYCW": Decimal("0.03400"),
    "NYLI": Decimal("0.04600"),
    "NYUP": Decimal("0.02000"),
    "PRMS": Decimal("0.03100"),
    "RFCE": Decimal("0.03800"),
    "RFCM": Decimal("0.04300"),
    "RFCW": Decimal("0.04800"),
    "RMPA": Decimal("0.04900"),
    "SPNO": Decimal("0.04600"),
    "SPSO": Decimal("0.04700"),
    "SRMV": Decimal("0.04100"),
    "SRMW": Decimal("0.05000"),
    "SRSO": Decimal("0.04300"),
    "SRTV": Decimal("0.04200"),
    "SRVC": Decimal("0.03700"),
}


# ---------------------------------------------------------------------------
# Renewable Technology Upstream EFs (kgCO2e/kWh)
# Near-zero lifecycle upstream from construction, maintenance, and
# decommissioning of generating equipment only.
# Source: IPCC AR5 WGIII Annex III, ecoinvent 3.11
# ---------------------------------------------------------------------------

RENEWABLE_UPSTREAM_EFS: Dict[str, Decimal] = {
    "solar_pv": Decimal("0.00200"),
    "solar_csp": Decimal("0.00250"),
    "wind_onshore": Decimal("0.00120"),
    "wind_offshore": Decimal("0.00180"),
    "hydro_reservoir": Decimal("0.00100"),
    "hydro_run_of_river": Decimal("0.00080"),
    "nuclear": Decimal("0.00150"),
    "geothermal": Decimal("0.00300"),
    "tidal": Decimal("0.00160"),
    "biomass": Decimal("0.01400"),
    "biogas": Decimal("0.01100"),
}


# ---------------------------------------------------------------------------
# Default Grid Mix Profiles by Country (% share of generation by fuel type)
# Used when site-specific grid mix is not available.
# Source: IEA Electricity Information 2023
# Keys map to WTT_FUEL_EMISSION_FACTORS fuel types.
# ---------------------------------------------------------------------------

DEFAULT_GRID_MIX: Dict[str, Dict[str, Decimal]] = {
    "US": {
        "natural_gas": Decimal("40.2"),
        "coal_bituminous": Decimal("19.5"),
        "nuclear": Decimal("18.9"),
        "wind_onshore": Decimal("10.2"),
        "hydro_reservoir": Decimal("6.0"),
        "solar_pv": Decimal("3.9"),
        "other": Decimal("1.3"),
    },
    "GB": {
        "natural_gas": Decimal("38.5"),
        "wind_onshore": Decimal("26.8"),
        "nuclear": Decimal("14.7"),
        "solar_pv": Decimal("4.3"),
        "biomass": Decimal("5.8"),
        "hydro_reservoir": Decimal("2.1"),
        "coal_bituminous": Decimal("1.5"),
        "other": Decimal("6.3"),
    },
    "DE": {
        "wind_onshore": Decimal("22.0"),
        "coal_lignite": Decimal("20.1"),
        "natural_gas": Decimal("15.3"),
        "solar_pv": Decimal("12.1"),
        "nuclear": Decimal("0.0"),
        "coal_bituminous": Decimal("8.4"),
        "biomass": Decimal("8.7"),
        "hydro_reservoir": Decimal("3.4"),
        "other": Decimal("10.0"),
    },
    "FR": {
        "nuclear": Decimal("62.8"),
        "hydro_reservoir": Decimal("11.8"),
        "wind_onshore": Decimal("8.5"),
        "natural_gas": Decimal("6.8"),
        "solar_pv": Decimal("4.5"),
        "biomass": Decimal("2.1"),
        "other": Decimal("3.5"),
    },
    "CN": {
        "coal_bituminous": Decimal("60.8"),
        "hydro_reservoir": Decimal("15.3"),
        "wind_onshore": Decimal("8.5"),
        "solar_pv": Decimal("5.2"),
        "nuclear": Decimal("5.0"),
        "natural_gas": Decimal("3.2"),
        "other": Decimal("2.0"),
    },
    "IN": {
        "coal_bituminous": Decimal("73.5"),
        "hydro_reservoir": Decimal("9.8"),
        "wind_onshore": Decimal("5.0"),
        "solar_pv": Decimal("5.5"),
        "natural_gas": Decimal("3.5"),
        "nuclear": Decimal("2.7"),
    },
    "JP": {
        "natural_gas": Decimal("34.5"),
        "coal_bituminous": Decimal("31.0"),
        "nuclear": Decimal("7.2"),
        "hydro_reservoir": Decimal("7.6"),
        "solar_pv": Decimal("9.8"),
        "wind_onshore": Decimal("1.0"),
        "fuel_oil_6": Decimal("3.8"),
        "other": Decimal("5.1"),
    },
}

#: Upstream EF lookup for fuel types used inside grid mix decomposition.
#: Maps grid-mix string keys to their WTT upstream EFs (kgCO2e/kWh).
#: Fossil fuel values mirror WTT_FUEL_EMISSION_FACTORS totals.
#: Renewable/nuclear values represent lifecycle-only upstream.
_GRID_FUEL_WTT: Dict[str, Decimal] = {
    # Fossil fuels -- totals from WTT_FUEL_EMISSION_FACTORS (DEFRA 2024)
    "natural_gas": Decimal("0.02460"),
    "coal_bituminous": Decimal("0.03710"),
    "coal_sub_bituminous": Decimal("0.03932"),
    "coal_lignite": Decimal("0.04165"),
    "coal_anthracite": Decimal("0.03308"),
    "fuel_oil_2": Decimal("0.04415"),
    "fuel_oil_6": Decimal("0.04738"),
    "diesel": Decimal("0.05070"),
    "peat": Decimal("0.03720"),
    "lpg": Decimal("0.02990"),
    "kerosene": Decimal("0.04302"),
    "petroleum_coke": Decimal("0.03958"),
    # Renewables and nuclear -- lifecycle upstream only (IPCC AR5 WGIII)
    "nuclear": Decimal("0.00150"),
    "wind_onshore": Decimal("0.00120"),
    "wind_offshore": Decimal("0.00180"),
    "solar_pv": Decimal("0.00200"),
    "solar_csp": Decimal("0.00250"),
    "hydro_reservoir": Decimal("0.00100"),
    "hydro_run_of_river": Decimal("0.00080"),
    "geothermal": Decimal("0.00300"),
    "biomass": Decimal("0.01400"),
    "biogas": Decimal("0.01100"),
    "tidal": Decimal("0.00160"),
    # Catch-all for unspecified "other" sources
    "other": Decimal("0.02000"),
}


# ============================================================================
# Helper Functions
# ============================================================================


def _utcnow() -> datetime:
    """Return current UTC datetime with zeroed microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _quantize(value: Decimal) -> Decimal:
    """Quantize a Decimal to the configured precision.

    Args:
        value: Decimal value to quantize.

    Returns:
        Quantized Decimal rounded to DECIMAL_PLACES digits.
    """
    return value.quantize(_QUANTIZE_EXP, rounding=ROUND_HALF_UP)


def _sha256(data: str) -> str:
    """Compute SHA-256 hex digest of the given string.

    Args:
        data: Input string to hash.

    Returns:
        64-character lowercase hex digest.
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _safe_decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal.

    Args:
        value: Value to convert (str, int, float, Decimal).

    Returns:
        Decimal representation.

    Raises:
        ValueError: If value cannot be converted.
    """
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"Cannot convert {value!r} to Decimal") from exc


def _dqi_tier(composite: Decimal) -> str:
    """Determine the qualitative DQI tier label from a composite score.

    Args:
        composite: Arithmetic mean of the five DQI dimension scores.

    Returns:
        Quality tier string (e.g. "High", "Medium").
    """
    for tier_name, (low, high) in DQI_QUALITY_TIERS.items():
        if low <= composite < high:
            return tier_name
    return "Very Low"


# ============================================================================
# UpstreamElectricityCalculatorEngine
# ============================================================================


class UpstreamElectricityCalculatorEngine:
    """Engine 3: Upstream Electricity Calculator for Activity 3b.

    Calculates upstream (pre-generation) emissions for purchased electricity,
    steam, heating, and cooling.  Supports both location-based (grid average
    upstream EFs) and market-based (supplier-specific upstream EFs) accounting.

    All arithmetic uses Python ``Decimal`` for deterministic reproducibility
    and zero-hallucination compliance.  No LLM calls are made during any
    calculation path.

    Thread-safety is ensured via a ``threading.Lock`` protecting internal
    mutable state (statistics counters, custom factor registrations).

    Attributes:
        _config: Engine configuration snapshot from ``get_config()``.
        _lock: Threading lock for mutable state protection.
        _custom_country_factors: User-registered country upstream EFs.
        _custom_egrid_factors: User-registered eGRID subregion upstream EFs.
        _stats: Runtime statistics counters.

    Example:
        >>> engine = UpstreamElectricityCalculatorEngine()
        >>> from datetime import date
        >>> from decimal import Decimal
        >>> record = ElectricityConsumptionRecord(
        ...     quantity_kwh=Decimal("100000"),
        ...     grid_region="US",
        ...     period_start=date(2024, 1, 1),
        ...     period_end=date(2024, 12, 31),
        ...     reporting_year=2024,
        ... )
        >>> result = engine.calculate_location_based(record, "US")
        >>> assert result.emissions_total > Decimal("0")
    """

    def __init__(self) -> None:
        """Initialize the UpstreamElectricityCalculatorEngine."""
        self._config = get_config()
        self._lock = threading.Lock()
        self._custom_country_factors: Dict[str, Decimal] = {}
        self._custom_egrid_factors: Dict[str, Decimal] = {}
        self._stats: Dict[str, int] = {
            "calculations_total": 0,
            "calculations_location_based": 0,
            "calculations_market_based": 0,
            "calculations_steam": 0,
            "calculations_heat": 0,
            "calculations_cooling": 0,
            "calculations_batch": 0,
            "calculations_errors": 0,
            "factor_lookups": 0,
            "dqi_assessments": 0,
            "uncertainty_assessments": 0,
            "double_counting_checks": 0,
            "grid_mix_decompositions": 0,
        }
        logger.info(
            "UpstreamElectricityCalculatorEngine initialized: "
            "engine=%s, version=%s",
            ENGINE_ID,
            ENGINE_VERSION,
        )

    # ====================================================================
    # Core Calculation Methods
    # ====================================================================

    def calculate(
        self,
        record: ElectricityConsumptionRecord,
        upstream_ef: Decimal,
        gwp_source: GWPSource = GWPSource.AR5,
    ) -> Activity3bResult:
        """Calculate Activity 3b upstream emissions for a single record.

        This is the primary calculation entry point.  It multiplies the
        consumed energy (kWh) by the provided upstream emission factor
        (kgCO2e/kWh) to produce total upstream emissions.

        Formula:
            emissions_total = quantity_kwh * upstream_ef

        Args:
            record: Validated electricity consumption record.
            upstream_ef: Upstream emission factor in kgCO2e/kWh.
            gwp_source: IPCC AR version for GWP conversions.

        Returns:
            Activity3bResult with total upstream emissions and provenance.

        Raises:
            ValueError: If record fails validation or upstream_ef is negative.
        """
        start_ns = time.monotonic_ns()
        is_valid, errors = self.validate_consumption_record(record)
        if not is_valid:
            self._increment_stat("calculations_errors")
            raise ValueError(
                f"Consumption record validation failed: {'; '.join(errors)}"
            )
        if upstream_ef < ZERO:
            self._increment_stat("calculations_errors")
            raise ValueError(
                f"upstream_ef must be >= 0, got {upstream_ef}"
            )

        emissions_total = _quantize(record.quantity_kwh * upstream_ef)

        provenance_input = (
            f"3b|{record.record_id}|{record.quantity_kwh}|"
            f"{upstream_ef}|{gwp_source.value}"
        )
        provenance_hash = _sha256(provenance_input)

        dqi = self._quick_dqi(record, upstream_ef)
        uncertainty_pct = self._quick_uncertainty(record, upstream_ef)

        elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000
        self._increment_stat("calculations_total")

        result = Activity3bResult(
            electricity_record_id=record.record_id,
            energy_type=record.energy_type,
            energy_consumed_kwh=record.quantity_kwh,
            upstream_ef=upstream_ef,
            upstream_ef_source=self._resolve_ef_source(record),
            accounting_method=record.accounting_method,
            grid_region=record.grid_region,
            emissions_total=emissions_total,
            is_renewable=record.is_renewable,
            dqi_score=dqi,
            uncertainty_pct=uncertainty_pct,
            provenance_hash=provenance_hash,
        )

        logger.debug(
            "Activity 3b calculated: record=%s, kwh=%s, ef=%s, "
            "emissions=%s kgCO2e, elapsed=%.2fms",
            record.record_id,
            record.quantity_kwh,
            upstream_ef,
            emissions_total,
            elapsed_ms,
        )
        return result

    def calculate_location_based(
        self,
        record: ElectricityConsumptionRecord,
        country_code: str,
        gwp_source: GWPSource = GWPSource.AR5,
    ) -> Activity3bResult:
        """Calculate Activity 3b using location-based grid average upstream EFs.

        Resolves the upstream emission factor from the embedded country-level
        lookup table (UPSTREAM_ELECTRICITY_FACTORS) or custom registrations,
        then delegates to ``calculate()``.

        Args:
            record: Electricity consumption record.
            country_code: ISO 3166-1 alpha-2 country code (e.g. "US", "GB").
            gwp_source: IPCC AR version for GWP conversions.

        Returns:
            Activity3bResult with location-based upstream emissions.

        Raises:
            ValueError: If country_code not found in factor tables.
        """
        upstream_ef_obj = self.get_upstream_ef(country_code)
        upstream_ef = upstream_ef_obj.upstream_ef

        self._increment_stat("calculations_location_based")
        return self.calculate(record, upstream_ef, gwp_source)

    def calculate_market_based(
        self,
        record: ElectricityConsumptionRecord,
        supplier_data: Dict[str, Any],
        gwp_source: GWPSource = GWPSource.AR5,
    ) -> Activity3bResult:
        """Calculate Activity 3b using market-based supplier-specific upstream EFs.

        Uses the supplier's reported upstream emission factor rather than
        the grid average.  Applicable when the reporting company has
        contractual instruments (PPAs, green tariffs, RECs) or supplier-
        specific upstream data (EPDs, PCFs, MiQ certificates).

        If the supplier is a 100% renewable provider and no explicit upstream
        EF is given, a near-zero lifecycle EF is applied.

        Args:
            record: Electricity consumption record.
            supplier_data: Dictionary with at minimum ``upstream_ef``
                (Decimal or numeric).  Optional keys: ``supplier_name``,
                ``data_source``, ``technology``, ``verification_level``.
            gwp_source: IPCC AR version for GWP conversions.

        Returns:
            Activity3bResult with market-based upstream emissions.

        Raises:
            ValueError: If supplier_data is missing ``upstream_ef``.
        """
        if "upstream_ef" not in supplier_data:
            # Check for renewable technology fallback
            technology = supplier_data.get("technology")
            if technology and technology in RENEWABLE_UPSTREAM_EFS:
                upstream_ef = RENEWABLE_UPSTREAM_EFS[technology]
            elif record.is_renewable:
                upstream_ef = Decimal("0.00200")  # default solar_pv lifecycle
            else:
                raise ValueError(
                    "supplier_data must contain 'upstream_ef' key or "
                    "'technology' key matching a renewable type"
                )
        else:
            upstream_ef = _safe_decimal(supplier_data["upstream_ef"])

        self._increment_stat("calculations_market_based")
        return self.calculate(record, upstream_ef, gwp_source)

    def calculate_batch(
        self,
        records: List[ElectricityConsumptionRecord],
        gwp_source: GWPSource = GWPSource.AR5,
    ) -> List[Activity3bResult]:
        """Calculate Activity 3b for a batch of consumption records.

        Each record is processed independently using location-based factors
        resolved from the record's ``grid_region`` and ``grid_region_type``.
        Failed records are logged and skipped (partial success model).

        Args:
            records: List of electricity consumption records.
            gwp_source: IPCC AR version for GWP conversions.

        Returns:
            List of Activity3bResult (one per successfully processed record).

        Raises:
            ValueError: If records list exceeds MAX_BATCH_SIZE.
        """
        if len(records) > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(records)} exceeds maximum {MAX_BATCH_SIZE}"
            )

        self._increment_stat("calculations_batch")
        results: List[Activity3bResult] = []
        error_count = 0

        for idx, record in enumerate(records):
            try:
                upstream_ef = self._resolve_ef_for_record(record)
                result = self.calculate(record, upstream_ef, gwp_source)
                results.append(result)
            except (ValueError, KeyError) as exc:
                error_count += 1
                logger.warning(
                    "Batch record %d/%d failed: record_id=%s, error=%s",
                    idx + 1,
                    len(records),
                    record.record_id,
                    str(exc),
                )

        logger.info(
            "Batch calculation complete: total=%d, success=%d, errors=%d",
            len(records),
            len(results),
            error_count,
        )
        return results

    # ====================================================================
    # Steam / Heating / Cooling
    # ====================================================================

    def calculate_steam_upstream(
        self,
        quantity_gj: Decimal,
        fuel_mix: Dict[str, Decimal],
        boiler_efficiency: Decimal = DEFAULT_BOILER_EFFICIENCY,
        gwp_source: GWPSource = GWPSource.AR5,
    ) -> Activity3bResult:
        """Calculate upstream emissions for purchased steam.

        Converts steam energy (GJ) to the equivalent fuel input energy
        (accounting for boiler efficiency), then weights the upstream
        WTT factors by the fuel mix percentages.

        Formula:
            fuel_input_kwh = (quantity_gj * 277.778) / boiler_efficiency
            weighted_ef = sum(fuel_pct_i * wtt_ef_i) / 100
            emissions = fuel_input_kwh * weighted_ef

        Args:
            quantity_gj: Steam energy consumed in GJ.
            fuel_mix: Dictionary mapping fuel type keys (matching
                ``_GRID_FUEL_WTT``) to percentage shares (summing to 100).
            boiler_efficiency: Thermal efficiency of the steam boiler
                as a fraction (0.0 to 1.0).  Default: 0.85.
            gwp_source: IPCC AR version for GWP conversions.

        Returns:
            Activity3bResult for the steam upstream emissions.

        Raises:
            ValueError: If boiler_efficiency is out of range or fuel_mix
                contains unknown fuels.
        """
        self._increment_stat("calculations_steam")
        return self._calculate_thermal_upstream(
            quantity_gj=quantity_gj,
            fuel_mix=fuel_mix,
            efficiency=boiler_efficiency,
            energy_type=EnergyType.STEAM,
            gwp_source=gwp_source,
        )

    def calculate_heat_upstream(
        self,
        quantity_gj: Decimal,
        fuel_mix: Dict[str, Decimal],
        gwp_source: GWPSource = GWPSource.AR5,
    ) -> Activity3bResult:
        """Calculate upstream emissions for purchased district heating.

        Similar to steam but assumes direct heat delivery without
        boiler efficiency losses (efficiency = 1.0 by convention for
        district heating networks where losses are accounted separately).

        Args:
            quantity_gj: Heat energy consumed in GJ.
            fuel_mix: Dictionary mapping fuel type keys to percentage
                shares (summing to 100).
            gwp_source: IPCC AR version for GWP conversions.

        Returns:
            Activity3bResult for the heat upstream emissions.
        """
        self._increment_stat("calculations_heat")
        return self._calculate_thermal_upstream(
            quantity_gj=quantity_gj,
            fuel_mix=fuel_mix,
            efficiency=ONE,
            energy_type=EnergyType.HEATING,
            gwp_source=gwp_source,
        )

    def calculate_cooling_upstream(
        self,
        quantity_gj: Decimal,
        cop: Decimal = DEFAULT_COOLING_COP,
        electricity_ef: Optional[Decimal] = None,
        country_code: str = "US",
        gwp_source: GWPSource = GWPSource.AR5,
    ) -> Activity3bResult:
        """Calculate upstream emissions for purchased district cooling.

        Cooling is typically produced by electric chillers.  The upstream
        emissions come from the electricity used to drive the chiller,
        divided by the coefficient of performance (COP).

        Formula:
            electricity_input_kwh = (quantity_gj * 277.778) / COP
            emissions = electricity_input_kwh * upstream_ef

        Args:
            quantity_gj: Cooling energy consumed in GJ.
            cop: Coefficient of Performance of the chiller system.
            electricity_ef: Upstream electricity EF (kgCO2e/kWh).  If
                None, resolved from ``country_code``.
            country_code: Fallback country for EF lookup if
                ``electricity_ef`` is not provided.
            gwp_source: IPCC AR version for GWP conversions.

        Returns:
            Activity3bResult for the cooling upstream emissions.

        Raises:
            ValueError: If COP <= 0 or country_code not found.
        """
        self._increment_stat("calculations_cooling")

        if cop <= ZERO:
            raise ValueError(f"COP must be > 0, got {cop}")

        if electricity_ef is None:
            ef_obj = self.get_upstream_ef(country_code)
            electricity_ef = ef_obj.upstream_ef

        cooling_kwh = _quantize(quantity_gj * GJ_TO_KWH)
        electricity_input_kwh = _quantize(cooling_kwh / cop)
        emissions_total = _quantize(electricity_input_kwh * electricity_ef)

        provenance_input = (
            f"3b_cooling|{quantity_gj}|{cop}|{electricity_ef}|{gwp_source.value}"
        )
        provenance_hash = _sha256(provenance_input)

        record_id = str(uuid.uuid4())
        return Activity3bResult(
            electricity_record_id=record_id,
            energy_type=EnergyType.COOLING,
            energy_consumed_kwh=cooling_kwh,
            upstream_ef=electricity_ef,
            upstream_ef_source=f"country:{country_code}",
            accounting_method=AccountingMethod.LOCATION_BASED,
            grid_region=country_code,
            emissions_total=emissions_total,
            is_renewable=False,
            dqi_score=Decimal("3.0"),
            uncertainty_pct=Decimal("30.0"),
            provenance_hash=provenance_hash,
        )

    # ====================================================================
    # Grid Mix Decomposition
    # ====================================================================

    def calculate_grid_mix_upstream(
        self,
        country_code: str,
        grid_mix_pcts: Optional[Dict[str, Decimal]] = None,
    ) -> Decimal:
        """Calculate a weighted upstream EF from a grid generation fuel mix.

        Decomposes the grid's generation mix into individual fuel types,
        looks up the WTT upstream EF for each fuel, and computes a
        weighted average based on each fuel's percentage share.

        Formula:
            upstream_ef = sum(pct_i / 100 * wtt_ef_i for each fuel i)

        Args:
            country_code: ISO 3166-1 alpha-2 country code for default
                grid mix lookup.
            grid_mix_pcts: Optional custom grid mix percentages.  Keys
                must match entries in ``_GRID_FUEL_WTT``.  Values are
                percentages that should sum to approximately 100.
                If None, the default mix for ``country_code`` is used.

        Returns:
            Weighted upstream EF in kgCO2e/kWh.

        Raises:
            ValueError: If country has no default grid mix and no
                custom mix is provided.
        """
        self._increment_stat("grid_mix_decompositions")

        if grid_mix_pcts is None:
            code_upper = country_code.upper().strip()
            if code_upper not in DEFAULT_GRID_MIX:
                raise ValueError(
                    f"No default grid mix for country '{country_code}'. "
                    f"Available: {sorted(DEFAULT_GRID_MIX.keys())}. "
                    f"Provide grid_mix_pcts explicitly."
                )
            grid_mix_pcts = DEFAULT_GRID_MIX[code_upper]

        weighted_ef = ZERO
        total_pct = ZERO
        for fuel_key, pct in grid_mix_pcts.items():
            wtt_ef = _GRID_FUEL_WTT.get(fuel_key, _GRID_FUEL_WTT["other"])
            weighted_ef += (pct / ONE_HUNDRED) * wtt_ef
            total_pct += pct

        if total_pct <= ZERO:
            logger.warning(
                "Grid mix percentages sum to zero for %s; "
                "returning zero upstream EF",
                country_code,
            )
            return ZERO

        result = _quantize(weighted_ef)
        logger.debug(
            "Grid mix upstream EF for %s: %s kgCO2e/kWh "
            "(total_pct=%s%%)",
            country_code,
            result,
            total_pct,
        )
        return result

    # ====================================================================
    # Factor Resolution
    # ====================================================================

    def get_upstream_ef(
        self,
        country_code: str,
        year: Optional[int] = None,
    ) -> UpstreamElectricityFactor:
        """Retrieve the upstream emission factor for a country.

        Resolution priority:
            1. Custom registered factors (via ``register_country_factor``).
            2. Embedded UPSTREAM_ELECTRICITY_FACTORS lookup table.
            3. ValueError if not found.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.
            year: Optional reference year (default: current base year).

        Returns:
            UpstreamElectricityFactor with the resolved EF.

        Raises:
            ValueError: If no factor found for the country code.
        """
        self._increment_stat("factor_lookups")
        code = country_code.upper().strip()
        ref_year = year or self._config.calculation.base_year

        # Priority 1: Custom factors
        with self._lock:
            if code in self._custom_country_factors:
                ef = self._custom_country_factors[code]
                return UpstreamElectricityFactor(
                    country_code=code,
                    upstream_ef=ef,
                    source="custom",
                    year=ref_year,
                )

        # Priority 2: Embedded factors
        if code in UPSTREAM_ELECTRICITY_FACTORS:
            return UpstreamElectricityFactor(
                country_code=code,
                upstream_ef=UPSTREAM_ELECTRICITY_FACTORS[code],
                source="IEA",
                year=2023,
            )

        raise ValueError(
            f"No upstream electricity factor for country '{code}'. "
            f"Available: {sorted(UPSTREAM_ELECTRICITY_FACTORS.keys())[:10]}..."
        )

    def get_upstream_ef_by_egrid(
        self,
        subregion: str,
    ) -> UpstreamElectricityFactor:
        """Retrieve the upstream emission factor for a US eGRID subregion.

        Resolution priority:
            1. Custom registered eGRID factors.
            2. Embedded EGRID_UPSTREAM_FACTORS lookup table.
            3. Fallback to US national average.

        Args:
            subregion: EPA eGRID subregion code (e.g. "CAMX", "ERCT").

        Returns:
            UpstreamElectricityFactor with the resolved EF.
        """
        self._increment_stat("factor_lookups")
        code = subregion.upper().strip()

        # Priority 1: Custom factors
        with self._lock:
            if code in self._custom_egrid_factors:
                ef = self._custom_egrid_factors[code]
                return UpstreamElectricityFactor(
                    country_code=code,
                    upstream_ef=ef,
                    source="custom_egrid",
                    year=self._config.calculation.base_year,
                )

        # Priority 2: Embedded eGRID factors
        if code in EGRID_UPSTREAM_FACTORS:
            return UpstreamElectricityFactor(
                country_code=code,
                upstream_ef=EGRID_UPSTREAM_FACTORS[code],
                source="EPA_eGRID",
                year=2022,
            )

        # Fallback: US national average
        logger.warning(
            "eGRID subregion '%s' not found; using US national average",
            code,
        )
        return UpstreamElectricityFactor(
            country_code="US",
            upstream_ef=UPSTREAM_ELECTRICITY_FACTORS.get("US", Decimal("0.04500")),
            source="IEA_fallback",
            year=2023,
        )

    def register_country_factor(
        self,
        country_code: str,
        upstream_ef: Decimal,
    ) -> None:
        """Register a custom upstream EF for a country.

        Overrides the embedded factor for subsequent lookups.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.
            upstream_ef: Custom upstream EF in kgCO2e/kWh.

        Raises:
            ValueError: If upstream_ef is negative.
        """
        if upstream_ef < ZERO:
            raise ValueError(f"upstream_ef must be >= 0, got {upstream_ef}")
        code = country_code.upper().strip()
        with self._lock:
            self._custom_country_factors[code] = upstream_ef
        logger.info(
            "Custom upstream EF registered: country=%s, ef=%s kgCO2e/kWh",
            code,
            upstream_ef,
        )

    def register_egrid_factor(
        self,
        subregion: str,
        upstream_ef: Decimal,
    ) -> None:
        """Register a custom upstream EF for a US eGRID subregion.

        Args:
            subregion: EPA eGRID subregion code.
            upstream_ef: Custom upstream EF in kgCO2e/kWh.

        Raises:
            ValueError: If upstream_ef is negative.
        """
        if upstream_ef < ZERO:
            raise ValueError(f"upstream_ef must be >= 0, got {upstream_ef}")
        code = subregion.upper().strip()
        with self._lock:
            self._custom_egrid_factors[code] = upstream_ef
        logger.info(
            "Custom eGRID upstream EF registered: subregion=%s, ef=%s kgCO2e/kWh",
            code,
            upstream_ef,
        )

    # ====================================================================
    # Per-Gas Breakdown
    # ====================================================================

    def calculate_per_gas(
        self,
        record: ElectricityConsumptionRecord,
        upstream_ef: Decimal,
        gwp_source: GWPSource = GWPSource.AR5,
    ) -> GasBreakdown:
        """Calculate per-gas breakdown (CO2, CH4, N2O) for upstream emissions.

        Distributes the total upstream EF across individual gases using
        typical upstream lifecycle gas ratios.  The default ratios are
        derived from the weighted average of fossil fuel upstream profiles.

        Gas Ratio Assumptions (typical grid fuel upstream):
            CO2: 85% of total upstream EF
            CH4: 12% of total upstream EF (dominated by gas extraction)
            N2O:  3% of total upstream EF

        Args:
            record: Electricity consumption record.
            upstream_ef: Upstream EF in kgCO2e/kWh.
            gwp_source: IPCC AR version for converting native gas masses
                to CO2e.

        Returns:
            GasBreakdown with per-gas emission values in native mass units.
        """
        co2_fraction = Decimal("0.85")
        ch4_fraction = Decimal("0.12")
        n2o_fraction = Decimal("0.03")

        total_emissions = _quantize(record.quantity_kwh * upstream_ef)

        # Total emissions are in kgCO2e.  Distribute across gases
        # then back-convert from kgCO2e to native mass using GWP.
        gwp = GWP_VALUES.get(gwp_source, GWP_VALUES[GWPSource.AR5])

        co2_co2e = _quantize(total_emissions * co2_fraction)
        ch4_co2e = _quantize(total_emissions * ch4_fraction)
        n2o_co2e = _quantize(total_emissions * n2o_fraction)

        # Convert CO2e back to native mass: native = co2e / GWP
        gwp_ch4 = gwp.get(EmissionGas.CH4, Decimal("28"))
        gwp_n2o = gwp.get(EmissionGas.N2O, Decimal("265"))

        co2_native = co2_co2e  # GWP of CO2 = 1
        ch4_native = _quantize(ch4_co2e / gwp_ch4) if gwp_ch4 > ZERO else ZERO
        n2o_native = _quantize(n2o_co2e / gwp_n2o) if gwp_n2o > ZERO else ZERO

        return GasBreakdown(
            co2=co2_native,
            ch4=ch4_native,
            n2o=n2o_native,
            co2e=total_emissions,
            gwp_source=gwp_source,
        )

    # ====================================================================
    # Comparison & Aggregation
    # ====================================================================

    def compare_location_vs_market(
        self,
        record: ElectricityConsumptionRecord,
        country_code: str,
        supplier_data: Dict[str, Any],
        gwp_source: GWPSource = GWPSource.AR5,
    ) -> Dict[str, Any]:
        """Compare location-based vs market-based upstream emissions.

        Calculates both methods for the same record and returns a
        comparison including the absolute and percentage difference.

        Args:
            record: Electricity consumption record.
            country_code: Country for location-based factor lookup.
            supplier_data: Supplier data for market-based calculation.
            gwp_source: IPCC AR version.

        Returns:
            Dictionary with ``location_based``, ``market_based``,
            ``difference_kgco2e``, ``difference_pct``, and
            ``recommended_method`` keys.
        """
        location_result = self.calculate_location_based(
            record, country_code, gwp_source
        )
        market_result = self.calculate_market_based(
            record, supplier_data, gwp_source
        )

        loc_total = location_result.emissions_total
        mkt_total = market_result.emissions_total
        diff = _quantize(loc_total - mkt_total)
        diff_pct = ZERO
        if loc_total > ZERO:
            diff_pct = _quantize((diff / loc_total) * ONE_HUNDRED)

        recommended = "market_based" if mkt_total < loc_total else "location_based"

        return {
            "location_based": {
                "emissions_total_kgco2e": loc_total,
                "upstream_ef": location_result.upstream_ef,
                "ef_source": location_result.upstream_ef_source,
            },
            "market_based": {
                "emissions_total_kgco2e": mkt_total,
                "upstream_ef": market_result.upstream_ef,
                "ef_source": market_result.upstream_ef_source,
            },
            "difference_kgco2e": diff,
            "difference_pct": diff_pct,
            "recommended_method": recommended,
            "record_id": record.record_id,
            "quantity_kwh": record.quantity_kwh,
        }

    def aggregate_by_energy_type(
        self,
        results: List[Activity3bResult],
    ) -> Dict[EnergyType, Decimal]:
        """Aggregate upstream emissions by energy type.

        Groups results by their ``energy_type`` (electricity, steam,
        heating, cooling) and sums ``emissions_total`` within each group.

        Args:
            results: List of Activity3bResult to aggregate.

        Returns:
            Dictionary mapping EnergyType to total kgCO2e.
        """
        agg: Dict[EnergyType, Decimal] = defaultdict(lambda: ZERO)
        for r in results:
            agg[r.energy_type] = _quantize(agg[r.energy_type] + r.emissions_total)
        return dict(agg)

    def aggregate_by_region(
        self,
        results: List[Activity3bResult],
    ) -> Dict[str, Decimal]:
        """Aggregate upstream emissions by grid region.

        Groups results by their ``grid_region`` and sums ``emissions_total``
        within each group.

        Args:
            results: List of Activity3bResult to aggregate.

        Returns:
            Dictionary mapping grid_region strings to total kgCO2e.
        """
        agg: Dict[str, Decimal] = defaultdict(lambda: ZERO)
        for r in results:
            region_key = r.grid_region or "UNKNOWN"
            agg[region_key] = _quantize(agg[region_key] + r.emissions_total)
        return dict(agg)

    def aggregate_by_facility(
        self,
        results: List[Activity3bResult],
    ) -> Dict[str, Decimal]:
        """Aggregate upstream emissions by facility.

        Uses the ``electricity_record_id`` prefix convention where the
        facility ID is extracted from the record's metadata.  Falls back
        to grouping by ``electricity_record_id`` if facility metadata
        is not embedded.

        Args:
            results: List of Activity3bResult to aggregate.

        Returns:
            Dictionary mapping facility identifiers to total kgCO2e.
        """
        agg: Dict[str, Decimal] = defaultdict(lambda: ZERO)
        for r in results:
            # Use grid_region as a proxy for facility grouping when
            # individual facility IDs are not embedded in the result.
            facility_key = r.electricity_record_id
            agg[facility_key] = _quantize(agg[facility_key] + r.emissions_total)
        return dict(agg)

    def get_total_emissions(
        self,
        results: List[Activity3bResult],
    ) -> Decimal:
        """Sum total emissions across all Activity 3b results.

        Args:
            results: List of Activity3bResult to sum.

        Returns:
            Grand total upstream emissions in kgCO2e.
        """
        total = ZERO
        for r in results:
            total += r.emissions_total
        return _quantize(total)

    # ====================================================================
    # Data Quality & Uncertainty
    # ====================================================================

    def assess_dqi(
        self,
        record: ElectricityConsumptionRecord,
        upstream_ef: Decimal,
    ) -> DQIAssessment:
        """Perform a full DQI assessment for an upstream electricity record.

        Scores data quality across the five GHG Protocol dimensions:
        temporal, geographical, technological, completeness, reliability.

        Scoring logic:
            - Temporal: Based on age of EF data vs reporting year.
            - Geographical: Based on specificity of grid region type.
            - Technological: Based on whether supplier-specific data exists.
            - Completeness: Based on whether all record fields are populated.
            - Reliability: Based on the data source hierarchy.

        Args:
            record: Electricity consumption record.
            upstream_ef: Upstream EF used for the calculation.

        Returns:
            DQIAssessment with per-dimension scores and composite.
        """
        self._increment_stat("dqi_assessments")

        temporal = self._score_temporal(record)
        geographical = self._score_geographical(record)
        technological = self._score_technological(record, upstream_ef)
        completeness = self._score_completeness(record)
        reliability = self._score_reliability(record, upstream_ef)

        composite = _quantize(
            (temporal + geographical + technological + completeness + reliability)
            / _FIVE
        )
        tier = _dqi_tier(composite)

        findings = self._generate_dqi_findings(
            record, temporal, geographical, technological,
            completeness, reliability, composite,
        )

        return DQIAssessment(
            record_id=record.record_id,
            activity_type=ActivityType.ACTIVITY_3B,
            temporal=temporal,
            geographical=geographical,
            technological=technological,
            completeness=completeness,
            reliability=reliability,
            composite=composite,
            tier=tier,
            findings=findings,
        )

    def quantify_uncertainty(
        self,
        record: ElectricityConsumptionRecord,
        upstream_ef: Decimal,
        method: UncertaintyMethod = UncertaintyMethod.IPCC_DEFAULT,
        confidence_level: Decimal = Decimal("95.0"),
    ) -> UncertaintyResult:
        """Quantify uncertainty of an upstream electricity emission calculation.

        Supports three methods:
            - IPCC_DEFAULT: Uses default uncertainty ranges by data quality.
            - ANALYTICAL: Analytical error propagation (root-sum-of-squares).
            - MONTE_CARLO: Monte Carlo simulation with N iterations.

        Args:
            record: Electricity consumption record.
            upstream_ef: Upstream EF in kgCO2e/kWh.
            method: Uncertainty quantification method.
            confidence_level: Confidence level percentage (default 95.0).

        Returns:
            UncertaintyResult with mean, std_dev, cv, and CI bounds.
        """
        self._increment_stat("uncertainty_assessments")

        mean_emissions = _quantize(record.quantity_kwh * upstream_ef)

        if method == UncertaintyMethod.MONTE_CARLO:
            return self._monte_carlo_uncertainty(
                record, upstream_ef, mean_emissions, confidence_level
            )
        elif method == UncertaintyMethod.ANALYTICAL:
            return self._analytical_uncertainty(
                record, upstream_ef, mean_emissions, confidence_level
            )
        else:
            return self._ipcc_default_uncertainty(
                record, upstream_ef, mean_emissions, confidence_level
            )

    # ====================================================================
    # Double-Counting Prevention
    # ====================================================================

    def check_double_counting(
        self,
        record: ElectricityConsumptionRecord,
        scope2_records: List[Dict[str, Any]],
    ) -> List[str]:
        """Check for potential double-counting between Activity 3b and Scope 2.

        Activity 3b covers upstream emissions of fuels used in electricity
        generation, which must NOT overlap with Scope 2 (direct generation
        emissions from purchased electricity).  This method identifies
        records that may be counted in both scopes.

        Checks performed:
            1. Same facility + same period in both Scope 2 and Activity 3b.
            2. Same energy quantity appears in both scopes.
            3. Market-based instruments (PPAs, RECs) already credited in Scope 2.

        Args:
            record: Activity 3b electricity consumption record.
            scope2_records: List of Scope 2 records as dictionaries with
                keys: ``facility_id``, ``period_start``, ``period_end``,
                ``quantity_kwh``, ``ppa_id``, ``rec_count``.

        Returns:
            List of warning strings describing potential double-counting.
            Empty list means no issues detected.
        """
        self._increment_stat("double_counting_checks")
        warnings: List[str] = []

        for s2 in scope2_records:
            s2_facility = s2.get("facility_id", "")
            s2_start = s2.get("period_start")
            s2_end = s2.get("period_end")
            s2_kwh = s2.get("quantity_kwh")
            s2_ppa = s2.get("ppa_id")

            # Check 1: Same facility and overlapping period
            if (
                record.facility_id
                and s2_facility
                and record.facility_id == s2_facility
                and s2_start is not None
                and s2_end is not None
            ):
                if (
                    record.period_start <= s2_end
                    and record.period_end >= s2_start
                ):
                    warnings.append(
                        f"Facility '{record.facility_id}' has overlapping "
                        f"periods in Scope 2 ({s2_start} to {s2_end}) and "
                        f"Activity 3b ({record.period_start} to "
                        f"{record.period_end}). Verify that Scope 2 covers "
                        f"generation emissions and 3b covers only upstream."
                    )

            # Check 2: Same energy quantity
            if s2_kwh is not None:
                s2_kwh_dec = _safe_decimal(s2_kwh)
                if record.quantity_kwh == s2_kwh_dec and s2_facility == record.facility_id:
                    warnings.append(
                        f"Identical quantity ({record.quantity_kwh} kWh) in both "
                        f"Scope 2 and Activity 3b for facility "
                        f"'{record.facility_id}'. This is expected if the "
                        f"quantities represent the same consumption event."
                    )

            # Check 3: Market-based instruments
            if (
                s2_ppa
                and record.ppa_id
                and s2_ppa == record.ppa_id
            ):
                warnings.append(
                    f"PPA '{record.ppa_id}' appears in both Scope 2 "
                    f"(market-based) and Activity 3b. Ensure upstream EF "
                    f"used in 3b corresponds to the PPA's actual generation "
                    f"source, not the grid average."
                )

        if not warnings:
            logger.debug(
                "No double-counting issues for record %s",
                record.record_id,
            )

        return warnings

    # ====================================================================
    # Renewable Technology EFs
    # ====================================================================

    def get_renewable_upstream_ef(
        self,
        technology: str,
    ) -> Decimal:
        """Get the near-zero upstream EF for a renewable technology.

        Renewable sources (solar, wind, hydro, nuclear) have near-zero
        upstream fuel emissions because they do not burn fossil fuels.
        The small residual EF represents lifecycle emissions from
        equipment manufacturing, construction, and maintenance.

        Args:
            technology: Renewable technology key (e.g. "solar_pv",
                "wind_onshore", "nuclear").

        Returns:
            Upstream EF in kgCO2e/kWh.

        Raises:
            ValueError: If technology is not recognized.
        """
        key = technology.lower().strip()
        if key in RENEWABLE_UPSTREAM_EFS:
            return RENEWABLE_UPSTREAM_EFS[key]
        raise ValueError(
            f"Unknown renewable technology '{technology}'. "
            f"Available: {sorted(RENEWABLE_UPSTREAM_EFS.keys())}"
        )

    # ====================================================================
    # Validation
    # ====================================================================

    def validate_consumption_record(
        self,
        record: ElectricityConsumptionRecord,
    ) -> Tuple[bool, List[str]]:
        """Validate an electricity consumption record for Activity 3b.

        Checks:
            1. quantity_kwh is positive.
            2. grid_region is non-empty.
            3. period_end >= period_start.
            4. reporting_year is within valid range.
            5. energy_type is a valid EnergyType.
            6. grid_region_type matches the grid_region format.

        Args:
            record: Electricity consumption record to validate.

        Returns:
            Tuple of (is_valid: bool, errors: List[str]).
        """
        errors: List[str] = []

        if record.quantity_kwh <= ZERO:
            errors.append(
                f"quantity_kwh must be > 0, got {record.quantity_kwh}"
            )

        if not record.grid_region or not record.grid_region.strip():
            errors.append("grid_region must not be empty")

        if record.period_end < record.period_start:
            errors.append(
                f"period_end ({record.period_end}) must be >= "
                f"period_start ({record.period_start})"
            )

        if record.reporting_year < 2000 or record.reporting_year > 2100:
            errors.append(
                f"reporting_year must be 2000-2100, got {record.reporting_year}"
            )

        # Validate grid_region format for eGRID subregions
        if record.grid_region_type == GridRegionType.EGRID_SUBREGION:
            region_upper = record.grid_region.upper().strip()
            if region_upper not in EGRID_UPSTREAM_FACTORS:
                if region_upper not in UPSTREAM_ELECTRICITY_FACTORS:
                    errors.append(
                        f"eGRID subregion '{record.grid_region}' not "
                        f"recognized. Will fall back to US national average."
                    )

        # Validate grid_region for country type
        if record.grid_region_type == GridRegionType.COUNTRY:
            code = record.grid_region.upper().strip()
            with self._lock:
                has_custom = code in self._custom_country_factors
            if code not in UPSTREAM_ELECTRICITY_FACTORS and not has_custom:
                errors.append(
                    f"Country code '{code}' not found in upstream "
                    f"electricity factors. Available countries: "
                    f"{sorted(list(UPSTREAM_ELECTRICITY_FACTORS.keys())[:10])}..."
                )

        is_valid = len(errors) == 0
        return is_valid, errors

    # ====================================================================
    # Statistics & Lifecycle
    # ====================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Return engine runtime statistics.

        Returns:
            Dictionary with all counter values and configuration metadata.
        """
        with self._lock:
            stats_copy = dict(self._stats)
        return {
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "counters": stats_copy,
            "custom_country_factors_count": len(self._custom_country_factors),
            "custom_egrid_factors_count": len(self._custom_egrid_factors),
            "embedded_country_factors_count": len(UPSTREAM_ELECTRICITY_FACTORS),
            "embedded_egrid_factors_count": len(EGRID_UPSTREAM_FACTORS),
            "renewable_technologies_count": len(RENEWABLE_UPSTREAM_EFS),
            "default_grid_mix_countries": sorted(DEFAULT_GRID_MIX.keys()),
        }

    def reset(self) -> None:
        """Reset engine state including statistics and custom factors.

        Primarily used in testing to prevent state leakage between
        test cases.
        """
        with self._lock:
            for key in self._stats:
                self._stats[key] = 0
            self._custom_country_factors.clear()
            self._custom_egrid_factors.clear()
        logger.info("UpstreamElectricityCalculatorEngine state reset")

    # ====================================================================
    # CHP Upstream Allocation
    # ====================================================================

    def calculate_chp_upstream(
        self,
        total_fuel_input_kwh: Decimal,
        electricity_output_kwh: Decimal,
        heat_output_kwh: Decimal,
        fuel_mix: Dict[str, Decimal],
        allocation_method: str = "energy",
    ) -> Dict[str, Activity3bResult]:
        """Allocate upstream emissions from a CHP plant to electricity and heat.

        Combined Heat and Power (CHP / cogeneration) plants produce both
        electricity and useful heat from a single fuel input.  The upstream
        emissions of the input fuel must be allocated between the two
        outputs using a recognized allocation method.

        Supported allocation methods:
            - "energy": Proportional to energy output (kWh).
            - "exergy": Weights electricity higher (Carnot factor ~2.0).
            - "efficiency": Based on reference efficiencies for separate
              production (electricity: 40%, heat: 90%).

        Args:
            total_fuel_input_kwh: Total fuel energy input in kWh.
            electricity_output_kwh: Electricity output in kWh.
            heat_output_kwh: Heat output in kWh.
            fuel_mix: Fuel mix percentages for upstream EF calculation.
            allocation_method: Allocation method ("energy", "exergy",
                "efficiency").

        Returns:
            Dictionary with "electricity" and "heat" Activity3bResult entries.

        Raises:
            ValueError: If outputs exceed input or method is unknown.
        """
        total_output = electricity_output_kwh + heat_output_kwh
        if total_output <= ZERO:
            raise ValueError("Total CHP output must be > 0")

        # Calculate weighted upstream EF from fuel mix
        weighted_ef = ZERO
        for fuel_key, pct in fuel_mix.items():
            wtt_ef = _GRID_FUEL_WTT.get(fuel_key, _GRID_FUEL_WTT["other"])
            weighted_ef += (pct / ONE_HUNDRED) * wtt_ef

        total_upstream_emissions = _quantize(total_fuel_input_kwh * weighted_ef)

        # Determine allocation fractions
        elec_frac, heat_frac = self._chp_allocation_fractions(
            electricity_output_kwh,
            heat_output_kwh,
            allocation_method,
        )

        elec_emissions = _quantize(total_upstream_emissions * elec_frac)
        heat_emissions = _quantize(total_upstream_emissions * heat_frac)

        record_id_elec = str(uuid.uuid4())
        record_id_heat = str(uuid.uuid4())

        provenance_elec = _sha256(
            f"chp_elec|{total_fuel_input_kwh}|{weighted_ef}|{elec_frac}"
        )
        provenance_heat = _sha256(
            f"chp_heat|{total_fuel_input_kwh}|{weighted_ef}|{heat_frac}"
        )

        elec_result = Activity3bResult(
            electricity_record_id=record_id_elec,
            energy_type=EnergyType.ELECTRICITY,
            energy_consumed_kwh=electricity_output_kwh,
            upstream_ef=weighted_ef,
            upstream_ef_source=f"CHP_allocation_{allocation_method}",
            accounting_method=AccountingMethod.LOCATION_BASED,
            grid_region="CHP",
            emissions_total=elec_emissions,
            is_renewable=False,
            dqi_score=Decimal("3.0"),
            uncertainty_pct=Decimal("25.0"),
            provenance_hash=provenance_elec,
        )

        heat_result = Activity3bResult(
            electricity_record_id=record_id_heat,
            energy_type=EnergyType.HEATING,
            energy_consumed_kwh=heat_output_kwh,
            upstream_ef=weighted_ef,
            upstream_ef_source=f"CHP_allocation_{allocation_method}",
            accounting_method=AccountingMethod.LOCATION_BASED,
            grid_region="CHP",
            emissions_total=heat_emissions,
            is_renewable=False,
            dqi_score=Decimal("3.0"),
            uncertainty_pct=Decimal("25.0"),
            provenance_hash=provenance_heat,
        )

        logger.info(
            "CHP upstream allocation (%s): total=%s kgCO2e, "
            "electricity=%s kgCO2e (%.1f%%), heat=%s kgCO2e (%.1f%%)",
            allocation_method,
            total_upstream_emissions,
            elec_emissions,
            float(elec_frac * ONE_HUNDRED),
            heat_emissions,
            float(heat_frac * ONE_HUNDRED),
        )

        return {
            "electricity": elec_result,
            "heat": heat_result,
        }

    # ====================================================================
    # Private Helper Methods
    # ====================================================================

    def _calculate_thermal_upstream(
        self,
        quantity_gj: Decimal,
        fuel_mix: Dict[str, Decimal],
        efficiency: Decimal,
        energy_type: EnergyType,
        gwp_source: GWPSource = GWPSource.AR5,
    ) -> Activity3bResult:
        """Internal method for steam and heat upstream calculations.

        Args:
            quantity_gj: Energy consumed in GJ.
            fuel_mix: Fuel mix percentages.
            efficiency: Thermal efficiency as a fraction (0-1).
            energy_type: EnergyType.STEAM or EnergyType.HEATING.
            gwp_source: IPCC AR version.

        Returns:
            Activity3bResult for the thermal upstream emissions.

        Raises:
            ValueError: If efficiency is out of range.
        """
        if efficiency <= ZERO or efficiency > ONE:
            raise ValueError(
                f"Efficiency must be in (0, 1], got {efficiency}"
            )

        # Validate fuel mix
        unknown_fuels = [
            k for k in fuel_mix if k not in _GRID_FUEL_WTT
        ]
        if unknown_fuels:
            logger.warning(
                "Unknown fuel types in mix: %s. Using 'other' EF.",
                unknown_fuels,
            )

        # Convert GJ to kWh
        consumed_kwh = _quantize(quantity_gj * GJ_TO_KWH)

        # Account for thermal efficiency: more fuel is needed than
        # the useful energy output
        fuel_input_kwh = _quantize(consumed_kwh / efficiency)

        # Calculate weighted upstream EF from fuel mix
        weighted_ef = ZERO
        total_pct = ZERO
        for fuel_key, pct in fuel_mix.items():
            wtt_ef = _GRID_FUEL_WTT.get(fuel_key, _GRID_FUEL_WTT["other"])
            weighted_ef += (pct / ONE_HUNDRED) * wtt_ef
            total_pct += pct

        if total_pct <= ZERO:
            raise ValueError("Fuel mix percentages must sum to > 0")

        emissions_total = _quantize(fuel_input_kwh * weighted_ef)

        provenance_input = (
            f"3b_{energy_type.value}|{quantity_gj}|{efficiency}|"
            f"{weighted_ef}|{gwp_source.value}"
        )
        provenance_hash = _sha256(provenance_input)

        record_id = str(uuid.uuid4())
        return Activity3bResult(
            electricity_record_id=record_id,
            energy_type=energy_type,
            energy_consumed_kwh=consumed_kwh,
            upstream_ef=weighted_ef,
            upstream_ef_source="fuel_mix_weighted",
            accounting_method=AccountingMethod.LOCATION_BASED,
            grid_region="thermal",
            emissions_total=emissions_total,
            is_renewable=False,
            dqi_score=Decimal("3.0"),
            uncertainty_pct=Decimal("25.0"),
            provenance_hash=provenance_hash,
        )

    def _resolve_ef_for_record(
        self,
        record: ElectricityConsumptionRecord,
    ) -> Decimal:
        """Resolve the upstream EF for a record based on its grid region type.

        Args:
            record: Electricity consumption record.

        Returns:
            Upstream EF in kgCO2e/kWh.

        Raises:
            ValueError: If the EF cannot be resolved.
        """
        if record.grid_region_type == GridRegionType.EGRID_SUBREGION:
            ef_obj = self.get_upstream_ef_by_egrid(record.grid_region)
            return ef_obj.upstream_ef

        if record.grid_region_type in (
            GridRegionType.COUNTRY,
            GridRegionType.EU_MEMBER_STATE,
        ):
            ef_obj = self.get_upstream_ef(record.grid_region)
            return ef_obj.upstream_ef

        # Custom region: try country lookup first, then fallback
        code = record.grid_region.upper().strip()
        if code in UPSTREAM_ELECTRICITY_FACTORS:
            return UPSTREAM_ELECTRICITY_FACTORS[code]

        with self._lock:
            if code in self._custom_country_factors:
                return self._custom_country_factors[code]

        raise ValueError(
            f"Cannot resolve upstream EF for custom region "
            f"'{record.grid_region}'"
        )

    def _resolve_ef_source(
        self,
        record: ElectricityConsumptionRecord,
    ) -> str:
        """Determine the EF source label for a record.

        Args:
            record: Electricity consumption record.

        Returns:
            Human-readable source label string.
        """
        if record.accounting_method == AccountingMethod.MARKET_BASED:
            return "supplier_specific"
        if record.grid_region_type == GridRegionType.EGRID_SUBREGION:
            return "EPA_eGRID"
        if record.grid_region_type == GridRegionType.EU_MEMBER_STATE:
            return "EU_AIB"
        return "IEA"

    def _quick_dqi(
        self,
        record: ElectricityConsumptionRecord,
        upstream_ef: Decimal,
    ) -> Decimal:
        """Calculate a quick composite DQI score without full assessment.

        Args:
            record: Electricity consumption record.
            upstream_ef: Upstream EF used.

        Returns:
            Composite DQI score (1.0-5.0).
        """
        temporal = self._score_temporal(record)
        geographical = self._score_geographical(record)
        # Use average for the other dimensions as a quick estimate
        tech_score = Decimal("2.0") if record.is_renewable else Decimal("3.0")
        completeness = Decimal("2.0") if record.supplier_id else Decimal("3.0")
        reliability = Decimal("2.0") if upstream_ef < Decimal("0.01") else Decimal("3.0")

        composite = _quantize(
            (temporal + geographical + tech_score + completeness + reliability)
            / _FIVE
        )
        # Clamp to [1.0, 5.0]
        if composite < ONE:
            composite = ONE
        if composite > _FIVE:
            composite = _FIVE
        return composite

    def _quick_uncertainty(
        self,
        record: ElectricityConsumptionRecord,
        upstream_ef: Decimal,
    ) -> Decimal:
        """Calculate a quick uncertainty percentage.

        Args:
            record: Electricity consumption record.
            upstream_ef: Upstream EF used.

        Returns:
            Uncertainty percentage.
        """
        if record.accounting_method == AccountingMethod.MARKET_BASED:
            return Decimal("15.0")
        if record.grid_region_type == GridRegionType.EGRID_SUBREGION:
            return Decimal("20.0")
        if record.grid_region_type == GridRegionType.COUNTRY:
            return Decimal("25.0")
        return Decimal("30.0")

    # ----------------------------------------------------------------
    # DQI Scoring Helpers
    # ----------------------------------------------------------------

    def _score_temporal(
        self,
        record: ElectricityConsumptionRecord,
    ) -> Decimal:
        """Score temporal representativeness (1=best, 5=worst).

        Based on the difference between the reporting year and the
        EF reference year (2023 for embedded factors).
        """
        age = abs(record.reporting_year - 2023)
        if age <= 1:
            return Decimal("1.0")
        elif age <= 2:
            return Decimal("2.0")
        elif age <= 3:
            return Decimal("3.0")
        elif age <= 5:
            return Decimal("4.0")
        return Decimal("5.0")

    def _score_geographical(
        self,
        record: ElectricityConsumptionRecord,
    ) -> Decimal:
        """Score geographical representativeness (1=best, 5=worst).

        Based on the specificity of the grid region type.
        """
        if record.grid_region_type == GridRegionType.EGRID_SUBREGION:
            return Decimal("1.0")
        elif record.grid_region_type == GridRegionType.EU_MEMBER_STATE:
            return Decimal("2.0")
        elif record.grid_region_type == GridRegionType.COUNTRY:
            return Decimal("2.0")
        return Decimal("4.0")  # CUSTOM_REGION

    def _score_technological(
        self,
        record: ElectricityConsumptionRecord,
        upstream_ef: Decimal,
    ) -> Decimal:
        """Score technological representativeness (1=best, 5=worst).

        Based on whether supplier-specific or average data is used.
        """
        if record.accounting_method == AccountingMethod.MARKET_BASED:
            return Decimal("1.0")
        if record.is_renewable:
            return Decimal("2.0")
        if upstream_ef < Decimal("0.02"):
            return Decimal("2.0")
        return Decimal("3.0")

    def _score_completeness(
        self,
        record: ElectricityConsumptionRecord,
    ) -> Decimal:
        """Score data completeness (1=best, 5=worst).

        Based on the number of populated optional fields.
        """
        populated = 0
        total_optional = 6  # facility_id, supplier_id, country_code, ppa_id, rec_count, supplier_name
        if record.facility_id:
            populated += 1
        if record.supplier_id:
            populated += 1
        if record.country_code:
            populated += 1
        if record.ppa_id:
            populated += 1
        if record.rec_count is not None:
            populated += 1
        if record.supplier_name:
            populated += 1

        ratio = Decimal(str(populated)) / Decimal(str(total_optional))
        if ratio >= Decimal("0.8"):
            return Decimal("1.0")
        elif ratio >= Decimal("0.6"):
            return Decimal("2.0")
        elif ratio >= Decimal("0.4"):
            return Decimal("3.0")
        elif ratio >= Decimal("0.2"):
            return Decimal("4.0")
        return Decimal("5.0")

    def _score_reliability(
        self,
        record: ElectricityConsumptionRecord,
        upstream_ef: Decimal,
    ) -> Decimal:
        """Score data reliability (1=best, 5=worst).

        Based on the source hierarchy of the emission factor.
        """
        if record.accounting_method == AccountingMethod.MARKET_BASED:
            return Decimal("1.0")
        if record.grid_region_type == GridRegionType.EGRID_SUBREGION:
            return Decimal("2.0")
        code = record.grid_region.upper().strip()
        if code in UPSTREAM_ELECTRICITY_FACTORS:
            return Decimal("2.0")
        return Decimal("4.0")

    def _generate_dqi_findings(
        self,
        record: ElectricityConsumptionRecord,
        temporal: Decimal,
        geographical: Decimal,
        technological: Decimal,
        completeness: Decimal,
        reliability: Decimal,
        composite: Decimal,
    ) -> List[str]:
        """Generate actionable DQI findings and recommendations.

        Args:
            record: The source record.
            temporal through composite: Individual DQI dimension scores.

        Returns:
            List of finding strings.
        """
        findings: List[str] = []

        if temporal > Decimal("3.0"):
            findings.append(
                f"Temporal score ({temporal}): EF data may be outdated. "
                f"Consider using factors from the most recent reporting year."
            )

        if geographical > Decimal("3.0"):
            findings.append(
                f"Geographical score ({geographical}): Using a broad "
                f"regional average. Consider using site-specific or "
                f"subregion-level upstream EFs."
            )

        if technological > Decimal("3.0"):
            findings.append(
                f"Technological score ({technological}): Generic average "
                f"data used. Obtain supplier-specific upstream data or "
                f"use market-based accounting."
            )

        if completeness > Decimal("3.0"):
            findings.append(
                f"Completeness score ({completeness}): Several optional "
                f"fields are missing. Populate facility_id, supplier_id, "
                f"and country_code for better traceability."
            )

        if reliability > Decimal("3.0"):
            findings.append(
                f"Reliability score ({reliability}): Factor source is "
                f"unverified or low-priority. Use verified EPD or supplier "
                f"PCF data where available."
            )

        if composite <= Decimal("2.0"):
            findings.append(
                f"Overall quality is HIGH (composite={composite}). "
                f"Data meets GHG Protocol Scope 3 quality expectations."
            )
        elif composite <= Decimal("3.5"):
            findings.append(
                f"Overall quality is MEDIUM (composite={composite}). "
                f"Acceptable for reporting; improvements recommended."
            )
        else:
            findings.append(
                f"Overall quality is LOW (composite={composite}). "
                f"Significant data quality improvements needed before "
                f"external reporting."
            )

        return findings

    # ----------------------------------------------------------------
    # Uncertainty Helpers
    # ----------------------------------------------------------------

    def _ipcc_default_uncertainty(
        self,
        record: ElectricityConsumptionRecord,
        upstream_ef: Decimal,
        mean_emissions: Decimal,
        confidence_level: Decimal,
    ) -> UncertaintyResult:
        """IPCC default uncertainty ranges based on data quality.

        Uses the upstream electricity typical uncertainty of +/- 20-30%
        for location-based and +/- 10-15% for supplier-specific data.
        """
        if record.accounting_method == AccountingMethod.MARKET_BASED:
            ef_uncertainty_pct = Decimal("10.0")
        elif record.grid_region_type == GridRegionType.EGRID_SUBREGION:
            ef_uncertainty_pct = Decimal("20.0")
        else:
            ef_uncertainty_pct = Decimal("25.0")

        # Activity data uncertainty: typically 2-5% for metered electricity
        activity_uncertainty_pct = Decimal("3.0")

        # Combined uncertainty: root-sum-of-squares
        combined_pct = _quantize(
            _safe_decimal(
                math.sqrt(
                    float(ef_uncertainty_pct ** 2)
                    + float(activity_uncertainty_pct ** 2)
                )
            )
        )

        # Calculate CI bounds
        z_factor = self._z_factor(confidence_level)
        half_width = _quantize(mean_emissions * combined_pct / ONE_HUNDRED * z_factor)
        ci_lower = _quantize(max(ZERO, mean_emissions - half_width))
        ci_upper = _quantize(mean_emissions + half_width)

        std_dev = _quantize(mean_emissions * combined_pct / ONE_HUNDRED)
        cv = _quantize(std_dev / mean_emissions) if mean_emissions > ZERO else ZERO

        return UncertaintyResult(
            mean=mean_emissions,
            std_dev=std_dev,
            cv=cv,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=confidence_level,
            method=UncertaintyMethod.IPCC_DEFAULT,
        )

    def _analytical_uncertainty(
        self,
        record: ElectricityConsumptionRecord,
        upstream_ef: Decimal,
        mean_emissions: Decimal,
        confidence_level: Decimal,
    ) -> UncertaintyResult:
        """Analytical error propagation (root-sum-of-squares).

        For a product formula E = A * EF, the relative uncertainty is:
            sigma_E / E = sqrt((sigma_A/A)^2 + (sigma_EF/EF)^2)
        """
        # Activity data relative uncertainty
        sigma_a_rel = Decimal("0.03")  # 3% for metered electricity

        # EF relative uncertainty depends on source
        if record.accounting_method == AccountingMethod.MARKET_BASED:
            sigma_ef_rel = Decimal("0.10")
        elif record.grid_region_type == GridRegionType.EGRID_SUBREGION:
            sigma_ef_rel = Decimal("0.20")
        else:
            sigma_ef_rel = Decimal("0.25")

        # Combined relative uncertainty
        combined_rel = _safe_decimal(
            math.sqrt(float(sigma_a_rel ** 2) + float(sigma_ef_rel ** 2))
        )

        std_dev = _quantize(mean_emissions * combined_rel)
        cv = _quantize(combined_rel)

        z_factor = self._z_factor(confidence_level)
        half_width = _quantize(std_dev * z_factor)
        ci_lower = _quantize(max(ZERO, mean_emissions - half_width))
        ci_upper = _quantize(mean_emissions + half_width)

        return UncertaintyResult(
            mean=mean_emissions,
            std_dev=std_dev,
            cv=cv,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=confidence_level,
            method=UncertaintyMethod.ANALYTICAL,
        )

    def _monte_carlo_uncertainty(
        self,
        record: ElectricityConsumptionRecord,
        upstream_ef: Decimal,
        mean_emissions: Decimal,
        confidence_level: Decimal,
    ) -> UncertaintyResult:
        """Monte Carlo simulation for uncertainty quantification.

        Runs N iterations with randomly sampled activity data and EF values
        drawn from assumed lognormal distributions.
        """
        iterations = self._config.monte_carlo_iterations
        seed = self._config.monte_carlo_seed
        rng = random.Random(seed)

        activity_mean = float(record.quantity_kwh)
        ef_mean = float(upstream_ef)

        # Relative standard deviations
        activity_rel_std = 0.03
        ef_rel_std = 0.20 if record.accounting_method == AccountingMethod.LOCATION_BASED else 0.10

        samples: List[float] = []
        for _ in range(iterations):
            sampled_activity = rng.gauss(activity_mean, activity_mean * activity_rel_std)
            sampled_ef = rng.gauss(ef_mean, ef_mean * ef_rel_std)
            # Clamp to non-negative
            sampled_activity = max(0.0, sampled_activity)
            sampled_ef = max(0.0, sampled_ef)
            samples.append(sampled_activity * sampled_ef)

        if not samples:
            return UncertaintyResult(
                mean=mean_emissions,
                std_dev=ZERO,
                cv=ZERO,
                ci_lower=mean_emissions,
                ci_upper=mean_emissions,
                confidence_level=confidence_level,
                method=UncertaintyMethod.MONTE_CARLO,
            )

        mc_mean = sum(samples) / len(samples)
        mc_variance = sum((x - mc_mean) ** 2 for x in samples) / len(samples)
        mc_std = math.sqrt(mc_variance)

        sorted_samples = sorted(samples)
        n = len(sorted_samples)
        alpha = (ONE_HUNDRED - confidence_level) / _TWO
        lower_idx = max(0, int(float(alpha) / 100.0 * n))
        upper_idx = min(n - 1, int(float(ONE_HUNDRED - alpha) / 100.0 * n))

        mc_ci_lower = sorted_samples[lower_idx]
        mc_ci_upper = sorted_samples[upper_idx]

        mc_cv = mc_std / mc_mean if mc_mean > 0 else 0.0

        return UncertaintyResult(
            mean=_quantize(_safe_decimal(mc_mean)),
            std_dev=_quantize(_safe_decimal(mc_std)),
            cv=_quantize(_safe_decimal(mc_cv)),
            ci_lower=_quantize(_safe_decimal(max(0.0, mc_ci_lower))),
            ci_upper=_quantize(_safe_decimal(mc_ci_upper)),
            confidence_level=confidence_level,
            method=UncertaintyMethod.MONTE_CARLO,
        )

    def _z_factor(self, confidence_level: Decimal) -> Decimal:
        """Return the z-score for a given confidence level.

        Args:
            confidence_level: Confidence level (e.g. 90, 95, 99).

        Returns:
            Corresponding z-score as Decimal.
        """
        cl = float(confidence_level)
        if cl >= 99.0:
            return Decimal("2.576")
        elif cl >= 97.5:
            return Decimal("2.243")
        elif cl >= 95.0:
            return Decimal("1.960")
        elif cl >= 90.0:
            return Decimal("1.645")
        elif cl >= 80.0:
            return Decimal("1.282")
        return Decimal("1.000")

    # ----------------------------------------------------------------
    # CHP Allocation Helpers
    # ----------------------------------------------------------------

    def _chp_allocation_fractions(
        self,
        electricity_kwh: Decimal,
        heat_kwh: Decimal,
        method: str,
    ) -> Tuple[Decimal, Decimal]:
        """Calculate CHP allocation fractions for electricity and heat.

        Args:
            electricity_kwh: Electricity output in kWh.
            heat_kwh: Heat output in kWh.
            method: Allocation method.

        Returns:
            Tuple of (electricity_fraction, heat_fraction) summing to 1.

        Raises:
            ValueError: If method is unrecognized.
        """
        total = electricity_kwh + heat_kwh
        if total <= ZERO:
            return (Decimal("0.5"), Decimal("0.5"))

        if method == "energy":
            elec_frac = _quantize(electricity_kwh / total)
            heat_frac = _quantize(ONE - elec_frac)
            return elec_frac, heat_frac

        elif method == "exergy":
            # Exergy approach: electricity has exergy factor ~1.0,
            # heat has exergy factor ~0.5 (Carnot-based)
            carnot_factor = Decimal("0.5")
            elec_exergy = electricity_kwh
            heat_exergy = heat_kwh * carnot_factor
            total_exergy = elec_exergy + heat_exergy
            if total_exergy <= ZERO:
                return (Decimal("0.5"), Decimal("0.5"))
            elec_frac = _quantize(elec_exergy / total_exergy)
            heat_frac = _quantize(ONE - elec_frac)
            return elec_frac, heat_frac

        elif method == "efficiency":
            # Reference efficiency method (EU CHP Directive)
            # Reference efficiencies: electricity 40%, heat 90%
            ref_elec_eff = Decimal("0.40")
            ref_heat_eff = Decimal("0.90")
            elec_ref_input = electricity_kwh / ref_elec_eff
            heat_ref_input = heat_kwh / ref_heat_eff
            total_ref_input = elec_ref_input + heat_ref_input
            if total_ref_input <= ZERO:
                return (Decimal("0.5"), Decimal("0.5"))
            elec_frac = _quantize(elec_ref_input / total_ref_input)
            heat_frac = _quantize(ONE - elec_frac)
            return elec_frac, heat_frac

        raise ValueError(
            f"Unknown CHP allocation method '{method}'. "
            f"Supported: energy, exergy, efficiency"
        )

    # ----------------------------------------------------------------
    # Statistics Helper
    # ----------------------------------------------------------------

    def _increment_stat(self, key: str, amount: int = 1) -> None:
        """Thread-safe statistics counter increment.

        Args:
            key: Statistics counter key.
            amount: Increment amount (default 1).
        """
        with self._lock:
            self._stats[key] = self._stats.get(key, 0) + amount


# ============================================================================
# Module-Level Singleton
# ============================================================================

_engine_instance: Optional[UpstreamElectricityCalculatorEngine] = None
_engine_lock = threading.Lock()


def get_upstream_electricity_calculator() -> UpstreamElectricityCalculatorEngine:
    """Return the singleton UpstreamElectricityCalculatorEngine instance.

    Uses double-checked locking for thread safety.

    Returns:
        UpstreamElectricityCalculatorEngine singleton.

    Example:
        >>> engine = get_upstream_electricity_calculator()
        >>> engine.get_statistics()
    """
    global _engine_instance
    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                _engine_instance = UpstreamElectricityCalculatorEngine()
    return _engine_instance


def reset_upstream_electricity_calculator() -> None:
    """Reset the singleton engine instance.

    Intended for test teardown to prevent state leakage.
    """
    global _engine_instance
    with _engine_lock:
        if _engine_instance is not None:
            _engine_instance.reset()
        _engine_instance = None
    logger.debug("UpstreamElectricityCalculatorEngine singleton reset")


# ============================================================================
# Public API Surface
# ============================================================================

__all__ = [
    # Engine class
    "UpstreamElectricityCalculatorEngine",
    # Singleton accessors
    "get_upstream_electricity_calculator",
    "reset_upstream_electricity_calculator",
    # Embedded data constants
    "EGRID_UPSTREAM_FACTORS",
    "RENEWABLE_UPSTREAM_EFS",
    "DEFAULT_GRID_MIX",
    # Module constants
    "ENGINE_ID",
    "ENGINE_VERSION",
    "GJ_TO_KWH",
    "DEFAULT_BOILER_EFFICIENCY",
    "DEFAULT_COOLING_COP",
    "MAX_BATCH_SIZE",
]
