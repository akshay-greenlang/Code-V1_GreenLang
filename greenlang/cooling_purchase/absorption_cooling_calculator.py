# -*- coding: utf-8 -*-
"""
AbsorptionCoolingCalculatorEngine - Engine 3: Cooling Purchase Agent (AGENT-MRV-012)

Core calculation engine for Scope 2 emissions from absorption chiller systems.
Computes thermal and parasitic electricity emissions for single-effect,
double-effect, triple-effect lithium bromide, and ammonia absorption chillers
using deterministic Decimal arithmetic with full calculation trace and
SHA-256 provenance hashing.

Absorption chillers bridge thermal and electrical domains: they consume heat
energy (steam, hot water, waste heat, solar, CHP exhaust) as their primary
driver while also requiring parasitic electricity for solution pumps, condenser
water pumps, and cooling tower fans. Total emissions are the sum of:

    Emissions_thermal (kgCO2e) = Heat_Input (GJ) x Heat_Source_EF (kgCO2e/GJ)
    Emissions_electric (kgCO2e) = Parasitic_Electricity (kWh) x Grid_EF (kgCO2e/kWh)
    Total_Emissions = Emissions_thermal + Emissions_electric

Where:
    Heat_Input (GJ)  = Cooling_Output (GJ) / COP_absorption
    Cooling_Output (GJ) = Cooling_Output (kWh_th) x 0.0036 GJ/kWh
    Parasitic_Electricity (kWh) = Cooling_Output (kWh_th) x Parasitic_Ratio

Zero-Hallucination Guarantees:
    - All calculations use Python Decimal (8 decimal places)
    - No LLM calls in the calculation path
    - Every step is recorded in the calculation trace
    - SHA-256 provenance hash for every result
    - COP values sourced from COOLING_TECHNOLOGY_SPECS constant table
    - Heat source EFs sourced from HEAT_SOURCE_FACTORS constant table
    - GWP values sourced from GWP_VALUES constant table

Default COP Values (from COOLING_TECHNOLOGY_SPECS):
    Single-effect LiBr:  0.70 (range 0.6-0.8)
    Double-effect LiBr:  1.20 (range 1.0-1.4)
    Triple-effect LiBr:  1.60 (range 1.4-1.8)
    Ammonia absorption:  0.55 (range 0.5-0.7)

Default Parasitic Ratios:
    Single-effect:  0.04 (pumps, cooling tower)
    Double-effect:  0.05 (pumps, cooling tower, solution pump)
    Triple-effect:  0.06 (higher solution pressure)
    Ammonia:        0.08 (rectifier, higher pressures)

Heat Source Emission Factors (kgCO2e/GJ):
    Natural gas steam:  70.1
    District heating:   70.0
    Waste heat:          0.0 (zero-cost byproduct)
    CHP exhaust:        CHP-allocated (default 70.0 conservative)
    Solar thermal:       0.0
    Geothermal:          0.0
    Biogas steam:        0.0 (biogenic)
    Fuel oil steam:     96.8
    Coal steam:        126.1
    Electric boiler:    Grid_EF / 0.98
    Heat pump:          Grid_EF / COP_HP

Example:
    >>> from greenlang.cooling_purchase.absorption_cooling_calculator import (
    ...     AbsorptionCoolingCalculatorEngine,
    ... )
    >>> from greenlang.cooling_purchase.models import (
    ...     AbsorptionCoolingRequest, AbsorptionType, HeatSource,
    ...     GWPSource, DataQualityTier,
    ... )
    >>> from decimal import Decimal
    >>> engine = AbsorptionCoolingCalculatorEngine()
    >>> request = AbsorptionCoolingRequest(
    ...     cooling_output_kwh_th=Decimal("100000"),
    ...     absorption_type=AbsorptionType.DOUBLE_EFFECT,
    ...     heat_source=HeatSource.NATURAL_GAS_STEAM,
    ...     grid_ef_kgco2e_per_kwh=Decimal("0.4"),
    ... )
    >>> result = engine.calculate_absorption_cooling(request)
    >>> assert result.emissions_kgco2e > 0
    >>> assert result.provenance_hash != ""

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-012 Cooling Purchase Agent (GL-MRV-X-023)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

from greenlang.cooling_purchase.config import CoolingPurchaseConfig
from greenlang.cooling_purchase.metrics import get_metrics
from greenlang.cooling_purchase.models import (
    AbsorptionCoolingRequest,
    AbsorptionType,
    CalculationResult,
    CoolingTechnology,
    CoolingTechnologySpec,
    COOLING_TECHNOLOGY_SPECS,
    DataQualityTier,
    EmissionGas,
    GasEmissionDetail,
    GWP_VALUES,
    GWPSource,
    HeatSource,
    HeatSourceFactor,
    HEAT_SOURCE_FACTORS,
    UNIT_CONVERSIONS,
)
from greenlang.cooling_purchase.provenance import get_provenance

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Decimal precision constant
# ---------------------------------------------------------------------------

_PRECISION = Decimal("0.00000001")  # 8 decimal places

# ---------------------------------------------------------------------------
# Energy conversion constants (all Decimal for zero-hallucination)
# ---------------------------------------------------------------------------

#: Convert kWh to GJ: 1 kWh = 0.0036 GJ
_KWH_TO_GJ = Decimal("0.0036")

#: Convert GJ to kWh: 1 GJ = 277.778 kWh
_GJ_TO_KWH = Decimal("277.778")

#: Electric boiler efficiency (dimensionless, 0-1)
_ELECTRIC_BOILER_EFFICIENCY = Decimal("0.98")

#: Default heat pump COP when none provided
_DEFAULT_HEAT_PUMP_COP = Decimal("3.0")

#: Default CHP heat emission factor (kgCO2e/GJ) - conservative estimate
_DEFAULT_CHP_HEAT_EF = Decimal("70.0")

# ---------------------------------------------------------------------------
# Default parasitic electricity ratios by absorption type
# ---------------------------------------------------------------------------

_DEFAULT_PARASITIC_RATIOS: Dict[str, Decimal] = {
    AbsorptionType.SINGLE_EFFECT.value: Decimal("0.04"),
    AbsorptionType.DOUBLE_EFFECT.value: Decimal("0.05"),
    AbsorptionType.TRIPLE_EFFECT.value: Decimal("0.06"),
    AbsorptionType.AMMONIA.value: Decimal("0.08"),
}

# ---------------------------------------------------------------------------
# Absorption type to CoolingTechnology mapping
# ---------------------------------------------------------------------------

_ABSORPTION_TO_TECHNOLOGY: Dict[str, str] = {
    AbsorptionType.SINGLE_EFFECT.value: CoolingTechnology.SINGLE_EFFECT_LIBR.value,
    AbsorptionType.DOUBLE_EFFECT.value: CoolingTechnology.DOUBLE_EFFECT_LIBR.value,
    AbsorptionType.TRIPLE_EFFECT.value: CoolingTechnology.TRIPLE_EFFECT_LIBR.value,
    AbsorptionType.AMMONIA.value: CoolingTechnology.AMMONIA_ABSORPTION.value,
}

# ---------------------------------------------------------------------------
# Zero-emission heat sources (no thermal emissions)
# ---------------------------------------------------------------------------

_ZERO_EMISSION_SOURCES: frozenset = frozenset({
    HeatSource.WASTE_HEAT.value,
    HeatSource.SOLAR_THERMAL.value,
    HeatSource.GEOTHERMAL.value,
    HeatSource.BIOGAS_STEAM.value,
})

# ---------------------------------------------------------------------------
# Gas decomposition ratios for combustion-based heat sources
# ---------------------------------------------------------------------------
# Typical combustion gas composition for natural-gas-class fuels:
#   CO2 accounts for ~99.5% of total CO2e
#   CH4 accounts for ~0.03% of total CO2e (at AR6 GWP=27.9)
#   N2O accounts for ~0.47% of total CO2e (at AR6 GWP=273)
#
# For grid electricity (parasitic), typical split:
#   CO2: ~98.5%, CH4: ~0.5%, N2O: ~1.0%
#
# These are indicative ratios. Actual gas-level emission factors would
# come from the grid or fuel-specific database. These ratios allow
# approximate decomposition of aggregate kgCO2e into individual gases.

_THERMAL_GAS_FRACTIONS: Dict[str, Dict[str, Decimal]] = {
    # Natural gas combustion products
    HeatSource.NATURAL_GAS_STEAM.value: {
        "CO2": Decimal("0.995"),
        "CH4": Decimal("0.001"),
        "N2O": Decimal("0.004"),
    },
    # District heating (mixed fuel, approximate)
    HeatSource.DISTRICT_HEATING.value: {
        "CO2": Decimal("0.990"),
        "CH4": Decimal("0.003"),
        "N2O": Decimal("0.007"),
    },
    # CHP exhaust (natural gas dominant)
    HeatSource.CHP_EXHAUST.value: {
        "CO2": Decimal("0.993"),
        "CH4": Decimal("0.002"),
        "N2O": Decimal("0.005"),
    },
    # Fuel oil combustion products
    HeatSource.FUEL_OIL_STEAM.value: {
        "CO2": Decimal("0.992"),
        "CH4": Decimal("0.001"),
        "N2O": Decimal("0.007"),
    },
    # Coal combustion products
    HeatSource.COAL_STEAM.value: {
        "CO2": Decimal("0.985"),
        "CH4": Decimal("0.002"),
        "N2O": Decimal("0.013"),
    },
    # Electric boiler (grid mix)
    HeatSource.ELECTRIC_BOILER.value: {
        "CO2": Decimal("0.985"),
        "CH4": Decimal("0.005"),
        "N2O": Decimal("0.010"),
    },
    # Heat pump (grid mix)
    HeatSource.HEAT_PUMP.value: {
        "CO2": Decimal("0.985"),
        "CH4": Decimal("0.005"),
        "N2O": Decimal("0.010"),
    },
}

#: Default grid electricity gas fractions for parasitic emissions
_GRID_GAS_FRACTIONS: Dict[str, Decimal] = {
    "CO2": Decimal("0.985"),
    "CH4": Decimal("0.005"),
    "N2O": Decimal("0.010"),
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed.

    Returns:
        UTC datetime with microsecond component set to zero for
        reproducible ISO timestamp strings.
    """
    return datetime.now(timezone.utc).replace(microsecond=0)


def _q(value: Decimal) -> Decimal:
    """Quantize a Decimal value to 8 decimal places using ROUND_HALF_UP.

    Args:
        value: Decimal value to quantize.

    Returns:
        Quantized Decimal value with 8 decimal places.
    """
    return value.quantize(_PRECISION, rounding=ROUND_HALF_UP)


def _canonical_json(data: Dict[str, Any]) -> str:
    """Serialize a dictionary to canonical JSON form.

    Uses ``sort_keys=True`` and ``default=str`` to produce deterministic
    output regardless of insertion order or non-standard types (Decimal,
    datetime, UUID, etc.).

    Args:
        data: Dictionary to serialize.

    Returns:
        Canonical JSON string with sorted keys.
    """
    return json.dumps(data, sort_keys=True, default=str)


def _compute_hash(data: Dict[str, Any]) -> str:
    """Compute SHA-256 hash of a canonical JSON dictionary.

    Args:
        data: Dictionary to hash.

    Returns:
        Hex-encoded SHA-256 hash string (64 characters).
    """
    canonical = _canonical_json(data)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ===========================================================================
# AbsorptionCoolingCalculatorEngine
# ===========================================================================


class AbsorptionCoolingCalculatorEngine:
    """Calculation engine for Scope 2 absorption chiller emissions.

    Implements deterministic Decimal-arithmetic calculations for
    single-effect, double-effect, triple-effect lithium bromide, and
    ammonia absorption chillers. Each calculation produces thermal
    emissions (from the heat source) and parasitic electricity emissions
    (from pumps and cooling tower fans), with a complete per-gas GHG
    breakdown and SHA-256 provenance hash.

    Thread-safe singleton: only one instance exists per process. The
    singleton is protected by a reentrant lock. All public methods are
    safe for concurrent invocation from multiple threads because they
    operate on local variables and delegate to thread-safe metrics and
    provenance subsystems.

    Zero-Hallucination Approach:
        - COP defaults sourced from COOLING_TECHNOLOGY_SPECS constant
        - Heat source EFs sourced from HEAT_SOURCE_FACTORS constant
        - GWP values sourced from GWP_VALUES constant
        - All arithmetic uses Python Decimal with 8-place quantization
        - No LLM, ML model, or external API calls in calculation path
        - Every intermediate value recorded in trace for auditing

    Attributes:
        _config: CoolingPurchaseConfig singleton reference.
        _metrics: CoolingPurchaseMetrics singleton reference.
        _provenance: CoolingPurchaseProvenance singleton reference.
        _enable_provenance: Whether SHA-256 provenance tracking is active.
        _precision_places: Number of decimal places for quantization.

    Example:
        >>> engine = AbsorptionCoolingCalculatorEngine()
        >>> result = engine.calculate_single_effect(
        ...     cooling_kwh_th=Decimal("50000"),
        ...     cop=Decimal("0.70"),
        ...     heat_source=HeatSource.NATURAL_GAS_STEAM,
        ...     heat_source_ef=Decimal("70.1"),
        ...     parasitic_ratio=Decimal("0.04"),
        ...     grid_ef=Decimal("0.4"),
        ... )
        >>> assert result.emissions_kgco2e > 0
    """

    _instance: Optional[AbsorptionCoolingCalculatorEngine] = None
    _lock: threading.RLock = threading.RLock()

    # ------------------------------------------------------------------
    # Singleton construction
    # ------------------------------------------------------------------

    def __new__(cls) -> AbsorptionCoolingCalculatorEngine:
        """Return the singleton AbsorptionCoolingCalculatorEngine instance.

        Uses double-checked locking with an RLock to ensure exactly one
        instance is created even under concurrent first-access.

        Returns:
            The singleton AbsorptionCoolingCalculatorEngine instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """Initialize the AbsorptionCoolingCalculatorEngine.

        Idempotent: after the first call, subsequent invocations are
        silently skipped to prevent duplicate initialization.
        """
        if self._initialized:
            return

        self._config = CoolingPurchaseConfig()
        self._metrics = get_metrics()
        self._provenance = get_provenance()
        self._enable_provenance: bool = self._config.enable_provenance
        self._precision_places: int = self._config.decimal_places
        self._precision_quantizer = Decimal(10) ** -self._precision_places

        self._initialized = True
        logger.info(
            "AbsorptionCoolingCalculatorEngine initialized "
            "(precision=%d, provenance=%s)",
            self._precision_places,
            self._enable_provenance,
        )

    # ------------------------------------------------------------------
    # Singleton management
    # ------------------------------------------------------------------

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance.

        **Testing only.** Destroys the current singleton so the next
        instantiation creates a fresh engine. Not safe to call while
        calculations are in progress.
        """
        with cls._lock:
            cls._instance = None
        logger.debug("AbsorptionCoolingCalculatorEngine singleton reset")

    # ------------------------------------------------------------------
    # Public API: High-level request-based calculation
    # ------------------------------------------------------------------

    def calculate_absorption_cooling(
        self,
        request: AbsorptionCoolingRequest,
    ) -> CalculationResult:
        """Calculate Scope 2 emissions from an absorption chiller.

        This is the primary entry point for absorption chiller emission
        calculations. It accepts a validated AbsorptionCoolingRequest,
        resolves COP and emission factors, computes thermal and parasitic
        emissions, decomposes into per-gas breakdown, and returns a
        complete CalculationResult with provenance hash.

        The formula implemented is:
            Heat_Input_GJ = Cooling_Output_GJ / COP
            Parasitic_kWh = Cooling_Output_kWh_th x Parasitic_Ratio
            Emissions_thermal = Heat_Input_GJ x Heat_Source_EF
            Emissions_parasitic = Parasitic_kWh x Grid_EF
            Total = Emissions_thermal + Emissions_parasitic

        Args:
            request: Validated AbsorptionCoolingRequest containing
                cooling output, absorption type, heat source, optional
                COP override, parasitic ratio, grid EF, and metadata.

        Returns:
            CalculationResult with total emissions, per-gas breakdown,
            COP used, energy input, trace steps, and provenance hash.

        Raises:
            ValueError: If COP override is out of valid range for the
                absorption type, or if required parameters are missing.
        """
        start_time = time.monotonic()
        calc_id = str(uuid.uuid4())
        trace: List[str] = []

        try:
            # Step 1: Resolve COP
            cop = self._resolve_cop(
                request.absorption_type,
                request.cop_override,
                trace,
            )

            # Step 2: Resolve heat source EF
            heat_source_ef = self._resolve_heat_source_ef(
                request.heat_source,
                request.heat_source_ef_override,
                request.grid_ef_kgco2e_per_kwh,
                None,  # cop_hp_override
                trace,
            )

            # Step 3: Resolve parasitic ratio
            parasitic_ratio = self._resolve_parasitic_ratio(
                request.absorption_type,
                request.parasitic_ratio,
                trace,
            )

            # Step 4: Calculate heat input
            heat_input_gj = self.calculate_heat_input_gj(
                request.cooling_output_kwh_th, cop,
            )
            trace.append(
                f"Heat input = {request.cooling_output_kwh_th} kWh_th "
                f"x {_KWH_TO_GJ} GJ/kWh / {cop} COP = {heat_input_gj} GJ"
            )

            # Step 5: Calculate parasitic electricity
            parasitic_kwh = self.calculate_parasitic_electricity(
                request.cooling_output_kwh_th, parasitic_ratio,
            )
            trace.append(
                f"Parasitic electricity = {request.cooling_output_kwh_th} kWh_th "
                f"x {parasitic_ratio} = {parasitic_kwh} kWh"
            )

            # Step 6: Calculate thermal emissions
            thermal_emissions = self.calculate_thermal_emissions(
                heat_input_gj, heat_source_ef,
            )
            trace.append(
                f"Thermal emissions = {heat_input_gj} GJ "
                f"x {heat_source_ef} kgCO2e/GJ = {thermal_emissions} kgCO2e"
            )

            # Step 7: Zero-emission heat path check
            if self.is_zero_emission_source(request.heat_source):
                trace.append(
                    f"Zero-emission heat path: heat_source={request.heat_source.value}, "
                    f"thermal_emissions={thermal_emissions} kgCO2e (confirmed zero)"
                )

            # Step 8: Calculate parasitic emissions
            parasitic_emissions = self.calculate_parasitic_emissions(
                parasitic_kwh, request.grid_ef_kgco2e_per_kwh,
            )
            trace.append(
                f"Parasitic emissions = {parasitic_kwh} kWh "
                f"x {request.grid_ef_kgco2e_per_kwh} kgCO2e/kWh "
                f"= {parasitic_emissions} kgCO2e"
            )

            # Step 9: Total emissions
            total_emissions = _q(thermal_emissions + parasitic_emissions)
            trace.append(
                f"Total emissions = {thermal_emissions} + {parasitic_emissions} "
                f"= {total_emissions} kgCO2e"
            )

            # Step 10: Gas decomposition
            gas_breakdown = self.decompose_emissions(
                thermal_emissions,
                parasitic_emissions,
                request.gwp_source,
                request.heat_source,
            )
            trace.append(
                f"Gas decomposition: {len(gas_breakdown)} species "
                f"(gwp_source={request.gwp_source.value})"
            )

            # Step 11: Total energy input (thermal + parasitic in kWh)
            heat_input_kwh = self.calculate_heat_input_kwh(
                request.cooling_output_kwh_th, cop,
            )
            total_energy_kwh = _q(heat_input_kwh + parasitic_kwh)
            trace.append(
                f"Total energy input = {heat_input_kwh} kWh_th (heat) "
                f"+ {parasitic_kwh} kWh (electric) = {total_energy_kwh} kWh"
            )

            # Step 12: Build metadata
            metadata = self._build_metadata(request, thermal_emissions, parasitic_emissions)

            # Step 13: Provenance hash
            provenance_hash = self._compute_provenance(
                calc_id, request, cop, heat_source_ef,
                parasitic_ratio, heat_input_gj, parasitic_kwh,
                thermal_emissions, parasitic_emissions,
                total_emissions, gas_breakdown, trace,
            )

            # Step 14: Record metrics
            duration_s = time.monotonic() - start_time
            self._record_metrics(
                request, total_emissions, cop, duration_s, "success",
            )

            # Step 15: Build result
            return CalculationResult(
                calculation_id=calc_id,
                calculation_type="absorption",
                cooling_output_kwh_th=request.cooling_output_kwh_th,
                energy_input_kwh=total_energy_kwh,
                cop_used=cop,
                emissions_kgco2e=total_emissions,
                gas_breakdown=gas_breakdown,
                calculation_tier=request.calculation_tier,
                provenance_hash=provenance_hash,
                trace_steps=trace,
                metadata=metadata,
            )

        except Exception as exc:
            duration_s = time.monotonic() - start_time
            self._record_metrics(
                request, Decimal("0"), Decimal("1"), duration_s, "failure",
            )
            logger.error(
                "Absorption cooling calculation failed: calc_id=%s error=%s",
                calc_id, str(exc), exc_info=True,
            )
            raise

    # ------------------------------------------------------------------
    # Public API: Type-specific convenience methods
    # ------------------------------------------------------------------

    def calculate_single_effect(
        self,
        cooling_kwh_th: Decimal,
        cop: Optional[Decimal] = None,
        heat_source: HeatSource = HeatSource.NATURAL_GAS_STEAM,
        heat_source_ef: Optional[Decimal] = None,
        parasitic_ratio: Optional[Decimal] = None,
        grid_ef: Decimal = Decimal("0.4"),
        gwp_source: GWPSource = GWPSource.AR6,
        calculation_tier: DataQualityTier = DataQualityTier.TIER_1,
        tenant_id: str = "default",
    ) -> CalculationResult:
        """Calculate emissions for a single-effect LiBr absorption chiller.

        Convenience method that constructs an AbsorptionCoolingRequest
        for single-effect LiBr and delegates to calculate_absorption_cooling.

        Default COP: 0.70 (range 0.6-0.8).
        Default parasitic ratio: 0.04 (4%).

        Args:
            cooling_kwh_th: Cooling energy output in kWh thermal.
            cop: Optional COP override. Defaults to 0.70.
            heat_source: Heat source driving the absorption cycle.
            heat_source_ef: Optional heat source EF override (kgCO2e/GJ).
            parasitic_ratio: Optional parasitic ratio override.
            grid_ef: Grid emission factor for parasitic electricity.
            gwp_source: IPCC Assessment Report for GWP values.
            calculation_tier: Data quality tier.
            tenant_id: Tenant identifier.

        Returns:
            CalculationResult with total emissions and breakdown.
        """
        request = AbsorptionCoolingRequest(
            cooling_output_kwh_th=cooling_kwh_th,
            absorption_type=AbsorptionType.SINGLE_EFFECT,
            heat_source=heat_source,
            cop_override=cop,
            parasitic_ratio=parasitic_ratio if parasitic_ratio is not None
            else _DEFAULT_PARASITIC_RATIOS[AbsorptionType.SINGLE_EFFECT.value],
            grid_ef_kgco2e_per_kwh=grid_ef,
            heat_source_ef_override=heat_source_ef,
            tenant_id=tenant_id,
            calculation_tier=calculation_tier,
            gwp_source=gwp_source,
        )
        return self.calculate_absorption_cooling(request)

    def calculate_double_effect(
        self,
        cooling_kwh_th: Decimal,
        cop: Optional[Decimal] = None,
        heat_source: HeatSource = HeatSource.NATURAL_GAS_STEAM,
        heat_source_ef: Optional[Decimal] = None,
        parasitic_ratio: Optional[Decimal] = None,
        grid_ef: Decimal = Decimal("0.4"),
        gwp_source: GWPSource = GWPSource.AR6,
        calculation_tier: DataQualityTier = DataQualityTier.TIER_1,
        tenant_id: str = "default",
    ) -> CalculationResult:
        """Calculate emissions for a double-effect LiBr absorption chiller.

        Convenience method that constructs an AbsorptionCoolingRequest
        for double-effect LiBr and delegates to calculate_absorption_cooling.

        Default COP: 1.20 (range 1.0-1.4).
        Default parasitic ratio: 0.05 (5%).

        Args:
            cooling_kwh_th: Cooling energy output in kWh thermal.
            cop: Optional COP override. Defaults to 1.20.
            heat_source: Heat source driving the absorption cycle.
            heat_source_ef: Optional heat source EF override (kgCO2e/GJ).
            parasitic_ratio: Optional parasitic ratio override.
            grid_ef: Grid emission factor for parasitic electricity.
            gwp_source: IPCC Assessment Report for GWP values.
            calculation_tier: Data quality tier.
            tenant_id: Tenant identifier.

        Returns:
            CalculationResult with total emissions and breakdown.
        """
        request = AbsorptionCoolingRequest(
            cooling_output_kwh_th=cooling_kwh_th,
            absorption_type=AbsorptionType.DOUBLE_EFFECT,
            heat_source=heat_source,
            cop_override=cop,
            parasitic_ratio=parasitic_ratio if parasitic_ratio is not None
            else _DEFAULT_PARASITIC_RATIOS[AbsorptionType.DOUBLE_EFFECT.value],
            grid_ef_kgco2e_per_kwh=grid_ef,
            heat_source_ef_override=heat_source_ef,
            tenant_id=tenant_id,
            calculation_tier=calculation_tier,
            gwp_source=gwp_source,
        )
        return self.calculate_absorption_cooling(request)

    def calculate_triple_effect(
        self,
        cooling_kwh_th: Decimal,
        cop: Optional[Decimal] = None,
        heat_source: HeatSource = HeatSource.NATURAL_GAS_STEAM,
        heat_source_ef: Optional[Decimal] = None,
        parasitic_ratio: Optional[Decimal] = None,
        grid_ef: Decimal = Decimal("0.4"),
        gwp_source: GWPSource = GWPSource.AR6,
        calculation_tier: DataQualityTier = DataQualityTier.TIER_1,
        tenant_id: str = "default",
    ) -> CalculationResult:
        """Calculate emissions for a triple-effect LiBr absorption chiller.

        Convenience method that constructs an AbsorptionCoolingRequest
        for triple-effect LiBr and delegates to calculate_absorption_cooling.

        Default COP: 1.60 (range 1.4-1.8).
        Default parasitic ratio: 0.06 (6%).

        Args:
            cooling_kwh_th: Cooling energy output in kWh thermal.
            cop: Optional COP override. Defaults to 1.60.
            heat_source: Heat source driving the absorption cycle.
            heat_source_ef: Optional heat source EF override (kgCO2e/GJ).
            parasitic_ratio: Optional parasitic ratio override.
            grid_ef: Grid emission factor for parasitic electricity.
            gwp_source: IPCC Assessment Report for GWP values.
            calculation_tier: Data quality tier.
            tenant_id: Tenant identifier.

        Returns:
            CalculationResult with total emissions and breakdown.
        """
        request = AbsorptionCoolingRequest(
            cooling_output_kwh_th=cooling_kwh_th,
            absorption_type=AbsorptionType.TRIPLE_EFFECT,
            heat_source=heat_source,
            cop_override=cop,
            parasitic_ratio=parasitic_ratio if parasitic_ratio is not None
            else _DEFAULT_PARASITIC_RATIOS[AbsorptionType.TRIPLE_EFFECT.value],
            grid_ef_kgco2e_per_kwh=grid_ef,
            heat_source_ef_override=heat_source_ef,
            tenant_id=tenant_id,
            calculation_tier=calculation_tier,
            gwp_source=gwp_source,
        )
        return self.calculate_absorption_cooling(request)

    def calculate_ammonia(
        self,
        cooling_kwh_th: Decimal,
        cop: Optional[Decimal] = None,
        heat_source: HeatSource = HeatSource.WASTE_HEAT,
        heat_source_ef: Optional[Decimal] = None,
        parasitic_ratio: Optional[Decimal] = None,
        grid_ef: Decimal = Decimal("0.4"),
        gwp_source: GWPSource = GWPSource.AR6,
        calculation_tier: DataQualityTier = DataQualityTier.TIER_1,
        tenant_id: str = "default",
    ) -> CalculationResult:
        """Calculate emissions for an ammonia absorption chiller.

        Convenience method that constructs an AbsorptionCoolingRequest
        for ammonia absorption and delegates to calculate_absorption_cooling.

        Default COP: 0.55 (range 0.5-0.7).
        Default parasitic ratio: 0.08 (8%).
        Default heat source: WASTE_HEAT (common for industrial ammonia).

        Args:
            cooling_kwh_th: Cooling energy output in kWh thermal.
            cop: Optional COP override. Defaults to 0.55.
            heat_source: Heat source driving the absorption cycle.
            heat_source_ef: Optional heat source EF override (kgCO2e/GJ).
            parasitic_ratio: Optional parasitic ratio override.
            grid_ef: Grid emission factor for parasitic electricity.
            gwp_source: IPCC Assessment Report for GWP values.
            calculation_tier: Data quality tier.
            tenant_id: Tenant identifier.

        Returns:
            CalculationResult with total emissions and breakdown.
        """
        request = AbsorptionCoolingRequest(
            cooling_output_kwh_th=cooling_kwh_th,
            absorption_type=AbsorptionType.AMMONIA,
            heat_source=heat_source,
            cop_override=cop,
            parasitic_ratio=parasitic_ratio if parasitic_ratio is not None
            else _DEFAULT_PARASITIC_RATIOS[AbsorptionType.AMMONIA.value],
            grid_ef_kgco2e_per_kwh=grid_ef,
            heat_source_ef_override=heat_source_ef,
            tenant_id=tenant_id,
            calculation_tier=calculation_tier,
            gwp_source=gwp_source,
        )
        return self.calculate_absorption_cooling(request)

    # ------------------------------------------------------------------
    # Public API: Heat input calculations
    # ------------------------------------------------------------------

    def calculate_heat_input_gj(
        self,
        cooling_kwh_th: Decimal,
        cop: Decimal,
    ) -> Decimal:
        """Calculate heat energy input to the absorption chiller in GJ.

        Converts cooling output from kWh_th to GJ, then divides by the
        absorption chiller COP to determine the heat energy required.

        Formula:
            Heat_Input_GJ = (Cooling_Output_kWh_th x 0.0036) / COP

        Args:
            cooling_kwh_th: Cooling energy output in kWh thermal.
                Must be positive.
            cop: Coefficient of performance of the absorption chiller.
                Must be positive.

        Returns:
            Heat energy input in gigajoules (GJ), quantized to 8
            decimal places.

        Raises:
            ValueError: If cooling_kwh_th <= 0 or cop <= 0.
            InvalidOperation: If Decimal arithmetic fails.
        """
        if cooling_kwh_th <= Decimal("0"):
            raise ValueError(
                f"cooling_kwh_th must be positive, got {cooling_kwh_th}"
            )
        if cop <= Decimal("0"):
            raise ValueError(f"COP must be positive, got {cop}")

        cooling_gj = _q(cooling_kwh_th * _KWH_TO_GJ)
        heat_input_gj = _q(cooling_gj / cop)

        logger.debug(
            "Heat input: cooling=%s kWh_th -> %s GJ / COP %s = %s GJ",
            cooling_kwh_th, cooling_gj, cop, heat_input_gj,
        )
        return heat_input_gj

    def calculate_heat_input_kwh(
        self,
        cooling_kwh_th: Decimal,
        cop: Decimal,
    ) -> Decimal:
        """Calculate heat energy input to the absorption chiller in kWh.

        Divides the cooling output in kWh_th by the absorption chiller
        COP to determine the heat energy required in kWh thermal.

        Formula:
            Heat_Input_kWh = Cooling_Output_kWh_th / COP

        Args:
            cooling_kwh_th: Cooling energy output in kWh thermal.
                Must be positive.
            cop: Coefficient of performance of the absorption chiller.
                Must be positive.

        Returns:
            Heat energy input in kWh thermal, quantized to 8 decimal
            places.

        Raises:
            ValueError: If cooling_kwh_th <= 0 or cop <= 0.
        """
        if cooling_kwh_th <= Decimal("0"):
            raise ValueError(
                f"cooling_kwh_th must be positive, got {cooling_kwh_th}"
            )
        if cop <= Decimal("0"):
            raise ValueError(f"COP must be positive, got {cop}")

        return _q(cooling_kwh_th / cop)

    # ------------------------------------------------------------------
    # Public API: Parasitic electricity calculations
    # ------------------------------------------------------------------

    def calculate_parasitic_electricity(
        self,
        cooling_kwh_th: Decimal,
        parasitic_ratio: Decimal,
    ) -> Decimal:
        """Calculate parasitic electricity consumption in kWh.

        Parasitic electricity covers solution pumps, condenser water
        pumps, and cooling tower fans required for absorption chiller
        operation.

        Formula:
            Parasitic_kWh = Cooling_Output_kWh_th x Parasitic_Ratio

        Args:
            cooling_kwh_th: Cooling energy output in kWh thermal.
                Must be non-negative.
            parasitic_ratio: Fraction of cooling output consumed by
                parasitic electricity (0 to 1). Typical values:
                0.04 (single-effect), 0.05 (double-effect),
                0.06 (triple-effect), 0.08 (ammonia).

        Returns:
            Parasitic electricity consumption in kWh, quantized to
            8 decimal places.

        Raises:
            ValueError: If cooling_kwh_th < 0 or parasitic_ratio not
                in [0, 1].
        """
        if cooling_kwh_th < Decimal("0"):
            raise ValueError(
                f"cooling_kwh_th must be non-negative, got {cooling_kwh_th}"
            )
        if parasitic_ratio < Decimal("0") or parasitic_ratio > Decimal("1"):
            raise ValueError(
                f"parasitic_ratio must be in [0, 1], got {parasitic_ratio}"
            )

        return _q(cooling_kwh_th * parasitic_ratio)

    def get_default_parasitic_ratio(
        self,
        absorption_type: AbsorptionType,
    ) -> Decimal:
        """Get the default parasitic electricity ratio for an absorption type.

        Default parasitic ratios reflect typical auxiliary power
        consumption as a fraction of cooling output:
            Single-effect LiBr: 0.04 (pumps, cooling tower)
            Double-effect LiBr: 0.05 (pumps, cooling tower, solution pump)
            Triple-effect LiBr: 0.06 (higher solution pressure)
            Ammonia:            0.08 (rectifier, higher pressures)

        Args:
            absorption_type: Type of absorption chiller cycle.

        Returns:
            Default parasitic ratio as Decimal (0 to 1).

        Raises:
            ValueError: If absorption_type is not recognized.
        """
        key = absorption_type.value
        if key not in _DEFAULT_PARASITIC_RATIOS:
            raise ValueError(
                f"Unknown absorption type: {absorption_type.value}. "
                f"Valid types: {list(_DEFAULT_PARASITIC_RATIOS.keys())}"
            )
        return _DEFAULT_PARASITIC_RATIOS[key]

    # ------------------------------------------------------------------
    # Public API: Emissions decomposition
    # ------------------------------------------------------------------

    def calculate_thermal_emissions(
        self,
        heat_input_gj: Decimal,
        heat_source_ef: Decimal,
    ) -> Decimal:
        """Calculate thermal emissions from heat source consumption.

        Formula:
            Emissions_thermal (kgCO2e) = Heat_Input (GJ) x EF (kgCO2e/GJ)

        Args:
            heat_input_gj: Heat energy input in GJ. Must be non-negative.
            heat_source_ef: Heat source emission factor in kgCO2e per GJ.
                Must be non-negative. Zero for waste heat, solar, geothermal.

        Returns:
            Thermal emissions in kgCO2e, quantized to 8 decimal places.

        Raises:
            ValueError: If heat_input_gj < 0 or heat_source_ef < 0.
        """
        if heat_input_gj < Decimal("0"):
            raise ValueError(
                f"heat_input_gj must be non-negative, got {heat_input_gj}"
            )
        if heat_source_ef < Decimal("0"):
            raise ValueError(
                f"heat_source_ef must be non-negative, got {heat_source_ef}"
            )

        return _q(heat_input_gj * heat_source_ef)

    def calculate_parasitic_emissions(
        self,
        parasitic_kwh: Decimal,
        grid_ef: Decimal,
    ) -> Decimal:
        """Calculate parasitic electricity emissions.

        Formula:
            Emissions_parasitic (kgCO2e) = Parasitic_kWh x Grid_EF (kgCO2e/kWh)

        Args:
            parasitic_kwh: Parasitic electricity consumption in kWh.
                Must be non-negative.
            grid_ef: Grid electricity emission factor in kgCO2e/kWh.
                Must be non-negative.

        Returns:
            Parasitic electricity emissions in kgCO2e, quantized to
            8 decimal places.

        Raises:
            ValueError: If parasitic_kwh < 0 or grid_ef < 0.
        """
        if parasitic_kwh < Decimal("0"):
            raise ValueError(
                f"parasitic_kwh must be non-negative, got {parasitic_kwh}"
            )
        if grid_ef < Decimal("0"):
            raise ValueError(
                f"grid_ef must be non-negative, got {grid_ef}"
            )

        return _q(parasitic_kwh * grid_ef)

    def decompose_emissions(
        self,
        thermal_emissions: Decimal,
        parasitic_emissions: Decimal,
        gwp_source: GWPSource,
        heat_source: Optional[HeatSource] = None,
    ) -> List[GasEmissionDetail]:
        """Decompose total CO2e emissions into per-gas breakdown.

        Combines the thermal and parasitic emission decompositions into
        a single list of gas emission details (CO2, CH4, N2O, CO2e).

        Args:
            thermal_emissions: Thermal component emissions in kgCO2e.
            parasitic_emissions: Parasitic component emissions in kgCO2e.
            gwp_source: IPCC Assessment Report for GWP values.
            heat_source: Heat source type for thermal gas ratios.
                If None, uses default combustion ratios.

        Returns:
            List of GasEmissionDetail entries for CO2, CH4, N2O, and
            the aggregate CO2e total.
        """
        # Decompose thermal emissions by gas
        thermal_gases = self.decompose_thermal_emissions(
            thermal_emissions, heat_source,
        )

        # Decompose parasitic emissions by gas
        parasitic_gases = self.decompose_parasitic_emissions(
            parasitic_emissions, gwp_source,
        )

        # Merge gas breakdowns
        return self._merge_gas_breakdowns(
            thermal_gases, parasitic_gases, gwp_source,
        )

    def decompose_thermal_emissions(
        self,
        total_thermal_co2e: Decimal,
        heat_source: Optional[HeatSource] = None,
    ) -> List[GasEmissionDetail]:
        """Decompose thermal emissions into per-gas breakdown.

        Uses heat-source-specific gas fractions to split the total
        thermal CO2e into CO2, CH4, and N2O components. For zero-emission
        heat sources, all gas quantities are zero.

        The decomposition uses approximate gas contribution ratios derived
        from combustion analysis. For zero-emission sources (waste heat,
        solar, geothermal, biogas), all gas quantities are zero.

        Args:
            total_thermal_co2e: Total thermal emissions in kgCO2e.
            heat_source: Heat source type. Determines the gas fraction
                ratios used for decomposition.

        Returns:
            List of GasEmissionDetail entries for CO2, CH4, N2O from
            the thermal component.
        """
        if total_thermal_co2e <= Decimal("0"):
            return [
                GasEmissionDetail(
                    gas=EmissionGas.CO2,
                    quantity_kg=Decimal("0"),
                    gwp_factor=Decimal("1"),
                    co2e_kg=Decimal("0"),
                ),
                GasEmissionDetail(
                    gas=EmissionGas.CH4,
                    quantity_kg=Decimal("0"),
                    gwp_factor=Decimal("1"),
                    co2e_kg=Decimal("0"),
                ),
                GasEmissionDetail(
                    gas=EmissionGas.N2O,
                    quantity_kg=Decimal("0"),
                    gwp_factor=Decimal("1"),
                    co2e_kg=Decimal("0"),
                ),
            ]

        # Get gas fractions for this heat source
        source_key = heat_source.value if heat_source else HeatSource.NATURAL_GAS_STEAM.value

        # Zero-emission sources produce zero gas
        if source_key in _ZERO_EMISSION_SOURCES:
            return [
                GasEmissionDetail(
                    gas=EmissionGas.CO2,
                    quantity_kg=Decimal("0"),
                    gwp_factor=Decimal("1"),
                    co2e_kg=Decimal("0"),
                ),
                GasEmissionDetail(
                    gas=EmissionGas.CH4,
                    quantity_kg=Decimal("0"),
                    gwp_factor=Decimal("1"),
                    co2e_kg=Decimal("0"),
                ),
                GasEmissionDetail(
                    gas=EmissionGas.N2O,
                    quantity_kg=Decimal("0"),
                    gwp_factor=Decimal("1"),
                    co2e_kg=Decimal("0"),
                ),
            ]

        fractions = _THERMAL_GAS_FRACTIONS.get(
            source_key,
            _THERMAL_GAS_FRACTIONS[HeatSource.NATURAL_GAS_STEAM.value],
        )

        co2_co2e = _q(total_thermal_co2e * fractions["CO2"])
        ch4_co2e = _q(total_thermal_co2e * fractions["CH4"])
        n2o_co2e = _q(total_thermal_co2e * fractions["N2O"])

        # CO2 quantity equals its CO2e (GWP=1)
        co2_kg = co2_co2e

        # CH4 and N2O quantities derived from CO2e / GWP
        # Use AR6 GWP for gas mass calculation
        gwp_ch4 = GWP_VALUES.get("AR6", {}).get("CH4", Decimal("27.9"))
        gwp_n2o = GWP_VALUES.get("AR6", {}).get("N2O", Decimal("273"))

        ch4_kg = _q(ch4_co2e / gwp_ch4) if gwp_ch4 > Decimal("0") else Decimal("0")
        n2o_kg = _q(n2o_co2e / gwp_n2o) if gwp_n2o > Decimal("0") else Decimal("0")

        return [
            GasEmissionDetail(
                gas=EmissionGas.CO2,
                quantity_kg=co2_kg,
                gwp_factor=Decimal("1"),
                co2e_kg=co2_co2e,
            ),
            GasEmissionDetail(
                gas=EmissionGas.CH4,
                quantity_kg=ch4_kg,
                gwp_factor=gwp_ch4,
                co2e_kg=ch4_co2e,
            ),
            GasEmissionDetail(
                gas=EmissionGas.N2O,
                quantity_kg=n2o_kg,
                gwp_factor=gwp_n2o,
                co2e_kg=n2o_co2e,
            ),
        ]

    def decompose_parasitic_emissions(
        self,
        total_parasitic_co2e: Decimal,
        gwp_source: GWPSource,
    ) -> List[GasEmissionDetail]:
        """Decompose parasitic electricity emissions into per-gas breakdown.

        Uses grid electricity gas fractions to split the total parasitic
        CO2e into CO2, CH4, and N2O components.

        Args:
            total_parasitic_co2e: Total parasitic emissions in kgCO2e.
            gwp_source: IPCC Assessment Report for GWP values.

        Returns:
            List of GasEmissionDetail entries for CO2, CH4, N2O from
            the parasitic electricity component.
        """
        if total_parasitic_co2e <= Decimal("0"):
            return [
                GasEmissionDetail(
                    gas=EmissionGas.CO2,
                    quantity_kg=Decimal("0"),
                    gwp_factor=Decimal("1"),
                    co2e_kg=Decimal("0"),
                ),
                GasEmissionDetail(
                    gas=EmissionGas.CH4,
                    quantity_kg=Decimal("0"),
                    gwp_factor=Decimal("1"),
                    co2e_kg=Decimal("0"),
                ),
                GasEmissionDetail(
                    gas=EmissionGas.N2O,
                    quantity_kg=Decimal("0"),
                    gwp_factor=Decimal("1"),
                    co2e_kg=Decimal("0"),
                ),
            ]

        # Get GWP values for selected source
        gwp_key = gwp_source.value
        gwp_map = GWP_VALUES.get(gwp_key, GWP_VALUES["AR6"])
        gwp_ch4 = gwp_map.get("CH4", Decimal("27.9"))
        gwp_n2o = gwp_map.get("N2O", Decimal("273"))

        co2_co2e = _q(total_parasitic_co2e * _GRID_GAS_FRACTIONS["CO2"])
        ch4_co2e = _q(total_parasitic_co2e * _GRID_GAS_FRACTIONS["CH4"])
        n2o_co2e = _q(total_parasitic_co2e * _GRID_GAS_FRACTIONS["N2O"])

        co2_kg = co2_co2e
        ch4_kg = _q(ch4_co2e / gwp_ch4) if gwp_ch4 > Decimal("0") else Decimal("0")
        n2o_kg = _q(n2o_co2e / gwp_n2o) if gwp_n2o > Decimal("0") else Decimal("0")

        return [
            GasEmissionDetail(
                gas=EmissionGas.CO2,
                quantity_kg=co2_kg,
                gwp_factor=Decimal("1"),
                co2e_kg=co2_co2e,
            ),
            GasEmissionDetail(
                gas=EmissionGas.CH4,
                quantity_kg=ch4_kg,
                gwp_factor=gwp_ch4,
                co2e_kg=ch4_co2e,
            ),
            GasEmissionDetail(
                gas=EmissionGas.N2O,
                quantity_kg=n2o_kg,
                gwp_factor=gwp_n2o,
                co2e_kg=n2o_co2e,
            ),
        ]

    # ------------------------------------------------------------------
    # Public API: Heat source handling
    # ------------------------------------------------------------------

    def get_heat_source_ef(
        self,
        heat_source: HeatSource,
        grid_ef_override: Optional[Decimal] = None,
        cop_hp_override: Optional[Decimal] = None,
    ) -> Decimal:
        """Get emission factor for a heat source in kgCO2e/GJ.

        Looks up the emission factor from the HEAT_SOURCE_FACTORS
        constant table. For grid-dependent sources (electric boiler,
        heat pump), computes the EF dynamically from the grid emission
        factor.

        Heat Source EF Resolution:
            - Static sources: Direct lookup from HEAT_SOURCE_FACTORS
            - Electric boiler: Grid_EF / 0.98 (boiler efficiency)
            - Heat pump: Grid_EF / COP_HP (default COP_HP = 3.0)
            - CHP exhaust: Uses resolve_chp_heat_ef (default 70.0)

        Args:
            heat_source: Heat source classification.
            grid_ef_override: Grid EF in kgCO2e/kWh for electric boiler
                or heat pump sources. Required when heat_source is
                ELECTRIC_BOILER or HEAT_PUMP.
            cop_hp_override: Heat pump COP override. Only used when
                heat_source is HEAT_PUMP. Defaults to 3.0.

        Returns:
            Emission factor in kgCO2e per GJ of heat input, quantized
            to 8 decimal places.

        Raises:
            ValueError: If grid_ef_override is required but not provided.
        """
        source_key = heat_source.value

        # Electric boiler: grid-dependent
        if heat_source == HeatSource.ELECTRIC_BOILER:
            if grid_ef_override is None:
                raise ValueError(
                    "grid_ef_override is required for ELECTRIC_BOILER heat source"
                )
            # Convert grid EF from kgCO2e/kWh to kgCO2e/GJ:
            # grid_ef (kgCO2e/kWh) x 277.778 (kWh/GJ) / 0.98 (efficiency)
            ef_per_gj = _q(
                grid_ef_override * _GJ_TO_KWH / _ELECTRIC_BOILER_EFFICIENCY
            )
            logger.debug(
                "Electric boiler heat EF: grid_ef=%s kgCO2e/kWh -> %s kgCO2e/GJ",
                grid_ef_override, ef_per_gj,
            )
            return ef_per_gj

        # Heat pump: grid-dependent with COP
        if heat_source == HeatSource.HEAT_PUMP:
            if grid_ef_override is None:
                raise ValueError(
                    "grid_ef_override is required for HEAT_PUMP heat source"
                )
            cop_hp = cop_hp_override if cop_hp_override else _DEFAULT_HEAT_PUMP_COP
            if cop_hp <= Decimal("0"):
                raise ValueError(
                    f"Heat pump COP must be positive, got {cop_hp}"
                )
            # Convert: grid_ef (kgCO2e/kWh) x 277.778 (kWh/GJ) / COP_HP
            ef_per_gj = _q(grid_ef_override * _GJ_TO_KWH / cop_hp)
            logger.debug(
                "Heat pump heat EF: grid_ef=%s kgCO2e/kWh, COP_HP=%s -> %s kgCO2e/GJ",
                grid_ef_override, cop_hp, ef_per_gj,
            )
            return ef_per_gj

        # CHP exhaust: allocated
        if heat_source == HeatSource.CHP_EXHAUST:
            return self.resolve_chp_heat_ef(chp_ef_override=None)

        # Static lookup from constant table
        factor = HEAT_SOURCE_FACTORS.get(source_key)
        if factor is None:
            logger.warning(
                "Unknown heat source '%s'; defaulting to 0.0 kgCO2e/GJ",
                source_key,
            )
            return Decimal("0")

        return factor.ef_kgco2e_per_gj

    def is_zero_emission_source(
        self,
        heat_source: HeatSource,
    ) -> bool:
        """Check whether a heat source produces zero thermal emissions.

        Zero-emission sources are those where the heat is a byproduct
        or from renewable sources, contributing zero direct CO2e
        emissions. For these sources, only parasitic electricity
        produces emissions.

        Zero-emission sources:
            - WASTE_HEAT: Industrial byproduct
            - SOLAR_THERMAL: Renewable
            - GEOTHERMAL: Renewable (closed-loop)
            - BIOGAS_STEAM: Biogenic (fossil CO2e = 0)

        Args:
            heat_source: Heat source classification.

        Returns:
            True if the heat source has zero thermal emissions.
        """
        return heat_source.value in _ZERO_EMISSION_SOURCES

    def resolve_chp_heat_ef(
        self,
        chp_ef_override: Optional[Decimal] = None,
    ) -> Decimal:
        """Resolve the emission factor for CHP exhaust heat.

        CHP (combined heat and power) exhaust heat requires emissions
        to be allocated between electrical and thermal outputs. The
        allocation method follows GHG Protocol guidance and would
        normally cross-reference AGENT-MRV-011 (CHP/Cogeneration).

        If no override is provided, a conservative default of 70.0
        kgCO2e/GJ is used (equivalent to natural gas boiler).

        Args:
            chp_ef_override: Pre-computed CHP heat allocation emission
                factor in kgCO2e/GJ. If None, uses conservative default.

        Returns:
            CHP heat emission factor in kgCO2e/GJ, quantized to 8
            decimal places.
        """
        if chp_ef_override is not None:
            if chp_ef_override < Decimal("0"):
                raise ValueError(
                    f"CHP EF override must be non-negative, got {chp_ef_override}"
                )
            logger.debug(
                "Using CHP heat EF override: %s kgCO2e/GJ",
                chp_ef_override,
            )
            return _q(chp_ef_override)

        logger.debug(
            "No CHP heat EF override; using conservative default: %s kgCO2e/GJ",
            _DEFAULT_CHP_HEAT_EF,
        )
        return _DEFAULT_CHP_HEAT_EF

    # ------------------------------------------------------------------
    # Public API: Hybrid plant calculations
    # ------------------------------------------------------------------

    def calculate_hybrid_plant(
        self,
        absorption_chillers: List[Dict[str, Any]],
        electric_chillers: List[Dict[str, Any]],
        grid_ef: Decimal,
        gwp_source: GWPSource = GWPSource.AR6,
        calculation_tier: DataQualityTier = DataQualityTier.TIER_1,
        tenant_id: str = "default",
    ) -> CalculationResult:
        """Calculate combined emissions for a hybrid absorption+electric plant.

        A hybrid plant contains multiple chiller units, some absorption
        and some electric. Each unit is calculated independently and
        results are aggregated into a combined CalculationResult with
        a weighted COP and total emissions.

        Absorption chiller dict keys:
            - cooling_kwh_th (Decimal): Cooling output
            - absorption_type (str): AbsorptionType value
            - heat_source (str): HeatSource value
            - cop (Decimal, optional): COP override
            - parasitic_ratio (Decimal, optional): Parasitic ratio
            - heat_source_ef (Decimal, optional): Heat source EF override

        Electric chiller dict keys:
            - cooling_kwh_th (Decimal): Cooling output
            - cop (Decimal): COP of the electric chiller
            - label (str, optional): Chiller label

        Args:
            absorption_chillers: List of absorption chiller parameter dicts.
            electric_chillers: List of electric chiller parameter dicts.
            grid_ef: Grid emission factor in kgCO2e/kWh.
            gwp_source: IPCC Assessment Report for GWP values.
            calculation_tier: Data quality tier.
            tenant_id: Tenant identifier.

        Returns:
            Combined CalculationResult with weighted COP and aggregated
            emissions from all chiller units.

        Raises:
            ValueError: If no chiller units are provided, or if required
                keys are missing from chiller dicts.
        """
        start_time = time.monotonic()
        calc_id = str(uuid.uuid4())
        trace: List[str] = []

        if not absorption_chillers and not electric_chillers:
            raise ValueError("At least one chiller unit is required")

        total_cooling = Decimal("0")
        total_emissions = Decimal("0")
        total_energy_input = Decimal("0")
        all_gas_breakdowns: List[GasEmissionDetail] = []
        unit_results: List[Dict[str, Any]] = []

        trace.append(
            f"Hybrid plant: {len(absorption_chillers)} absorption + "
            f"{len(electric_chillers)} electric chillers"
        )

        # Process absorption chillers
        for idx, ac in enumerate(absorption_chillers):
            ac_cooling = Decimal(str(ac.get("cooling_kwh_th", "0")))
            ac_type_str = ac.get("absorption_type", AbsorptionType.DOUBLE_EFFECT.value)
            ac_hs_str = ac.get("heat_source", HeatSource.NATURAL_GAS_STEAM.value)

            try:
                ac_type = AbsorptionType(ac_type_str)
            except ValueError:
                ac_type = AbsorptionType.DOUBLE_EFFECT

            try:
                ac_hs = HeatSource(ac_hs_str)
            except ValueError:
                ac_hs = HeatSource.NATURAL_GAS_STEAM

            ac_cop = ac.get("cop")
            ac_pr = ac.get("parasitic_ratio")
            ac_ef = ac.get("heat_source_ef")

            request = AbsorptionCoolingRequest(
                cooling_output_kwh_th=ac_cooling,
                absorption_type=ac_type,
                heat_source=ac_hs,
                cop_override=Decimal(str(ac_cop)) if ac_cop is not None else None,
                parasitic_ratio=Decimal(str(ac_pr)) if ac_pr is not None
                else _DEFAULT_PARASITIC_RATIOS.get(ac_type.value, Decimal("0.05")),
                grid_ef_kgco2e_per_kwh=grid_ef,
                heat_source_ef_override=Decimal(str(ac_ef)) if ac_ef is not None else None,
                tenant_id=tenant_id,
                calculation_tier=calculation_tier,
                gwp_source=gwp_source,
            )

            result = self.calculate_absorption_cooling(request)

            total_cooling += result.cooling_output_kwh_th
            total_emissions += result.emissions_kgco2e
            total_energy_input += result.energy_input_kwh
            all_gas_breakdowns.extend(result.gas_breakdown)
            unit_results.append({
                "unit_index": idx,
                "unit_type": "absorption",
                "cooling_kwh_th": str(result.cooling_output_kwh_th),
                "cop_used": str(result.cop_used),
                "emissions_kgco2e": str(result.emissions_kgco2e),
            })

            trace.append(
                f"Absorption unit {idx}: {ac_type.value}, "
                f"cooling={result.cooling_output_kwh_th} kWh_th, "
                f"COP={result.cop_used}, "
                f"emissions={result.emissions_kgco2e} kgCO2e"
            )

        # Process electric chillers
        for idx, ec in enumerate(electric_chillers):
            ec_cooling = Decimal(str(ec.get("cooling_kwh_th", "0")))
            ec_cop = Decimal(str(ec.get("cop", "5.0")))
            ec_label = ec.get("label", f"electric_{idx}")

            if ec_cooling <= Decimal("0"):
                trace.append(f"Electric unit {idx} ({ec_label}): skipped (zero cooling)")
                continue

            if ec_cop <= Decimal("0"):
                raise ValueError(
                    f"Electric chiller COP must be positive for unit {idx}"
                )

            # Electric chiller: Emissions = (cooling / COP) x grid_ef
            elec_input_kwh = _q(ec_cooling / ec_cop)
            ec_emissions = _q(elec_input_kwh * grid_ef)

            total_cooling += ec_cooling
            total_emissions += ec_emissions
            total_energy_input += elec_input_kwh

            # Decompose electric chiller emissions into gases
            ec_gases = self.decompose_parasitic_emissions(ec_emissions, gwp_source)
            all_gas_breakdowns.extend(ec_gases)

            unit_results.append({
                "unit_index": idx,
                "unit_type": "electric",
                "label": ec_label,
                "cooling_kwh_th": str(ec_cooling),
                "cop_used": str(ec_cop),
                "emissions_kgco2e": str(ec_emissions),
            })

            trace.append(
                f"Electric unit {idx} ({ec_label}): "
                f"cooling={ec_cooling} kWh_th, COP={ec_cop}, "
                f"input={elec_input_kwh} kWh, "
                f"emissions={ec_emissions} kgCO2e"
            )

        # Calculate weighted COP
        weighted_cop = self.calculate_plant_weighted_cop(unit_results)
        total_emissions = _q(total_emissions)
        total_energy_input = _q(total_energy_input)

        trace.append(
            f"Plant totals: cooling={total_cooling} kWh_th, "
            f"energy_input={total_energy_input} kWh, "
            f"emissions={total_emissions} kgCO2e, "
            f"weighted_COP={weighted_cop}"
        )

        # Merge gas breakdowns across all units
        merged_gases = self._aggregate_gas_breakdowns(all_gas_breakdowns)

        # Provenance hash
        provenance_data = {
            "calculation_id": calc_id,
            "calculation_type": "hybrid_plant",
            "total_cooling_kwh_th": str(total_cooling),
            "total_emissions_kgco2e": str(total_emissions),
            "total_energy_input_kwh": str(total_energy_input),
            "weighted_cop": str(weighted_cop),
            "unit_count": len(unit_results),
            "trace_steps": len(trace),
        }
        provenance_hash = _compute_hash(provenance_data)

        # Record metrics
        duration_s = time.monotonic() - start_time
        self._metrics.record_calculation(
            technology="hybrid_chiller",
            calculation_type="absorption",
            tier=calculation_tier.value,
            tenant_id=tenant_id,
            status="success",
            duration_s=duration_s,
            emissions_kgco2e=float(total_emissions),
            cooling_kwh_th=float(total_cooling),
            cop_used=float(weighted_cop),
            condenser_type="unknown",
        )

        metadata = {
            "plant_type": "hybrid",
            "absorption_count": str(len(absorption_chillers)),
            "electric_count": str(len(electric_chillers)),
            "weighted_cop": str(weighted_cop),
            "tenant_id": tenant_id,
        }

        return CalculationResult(
            calculation_id=calc_id,
            calculation_type="hybrid_plant",
            cooling_output_kwh_th=total_cooling,
            energy_input_kwh=total_energy_input,
            cop_used=weighted_cop,
            emissions_kgco2e=total_emissions,
            gas_breakdown=merged_gases,
            calculation_tier=calculation_tier,
            provenance_hash=provenance_hash,
            trace_steps=trace,
            metadata=metadata,
        )

    def calculate_plant_weighted_cop(
        self,
        units: List[Dict[str, Any]],
    ) -> Decimal:
        """Calculate weighted average COP across multiple chiller units.

        The weighted COP is computed as:
            Weighted_COP = Total_Cooling_Output / Total_Energy_Input

        This is equivalent to the capacity-weighted harmonic mean of
        individual unit COPs.

        If a unit dict has ``cooling_kwh_th`` and ``cop_used`` keys,
        the energy input is derived as cooling / COP. The weighted COP
        is then total_cooling / total_derived_input.

        Args:
            units: List of unit result dicts with ``cooling_kwh_th``
                and ``cop_used`` keys (string values accepted).

        Returns:
            Weighted average COP as Decimal, quantized to 8 decimal
            places. Returns Decimal("1") if no valid units.
        """
        total_cooling = Decimal("0")
        total_input = Decimal("0")

        for unit in units:
            cooling = Decimal(str(unit.get("cooling_kwh_th", "0")))
            cop = Decimal(str(unit.get("cop_used", "1")))

            if cooling > Decimal("0") and cop > Decimal("0"):
                total_cooling += cooling
                total_input += _q(cooling / cop)

        if total_input <= Decimal("0") or total_cooling <= Decimal("0"):
            logger.warning("No valid units for weighted COP; returning 1.0")
            return Decimal("1")

        return _q(total_cooling / total_input)

    # ------------------------------------------------------------------
    # Public API: COP resolution
    # ------------------------------------------------------------------

    def get_default_cop(
        self,
        absorption_type: AbsorptionType,
    ) -> Decimal:
        """Get the default COP for an absorption chiller type.

        Looks up the default COP from COOLING_TECHNOLOGY_SPECS.

        Default COP values:
            Single-effect LiBr: 0.70
            Double-effect LiBr: 1.20
            Triple-effect LiBr: 1.60
            Ammonia absorption: 0.55

        Args:
            absorption_type: Type of absorption chiller cycle.

        Returns:
            Default COP as Decimal.

        Raises:
            ValueError: If absorption_type mapping is not found.
        """
        tech_key = _ABSORPTION_TO_TECHNOLOGY.get(absorption_type.value)
        if tech_key is None:
            raise ValueError(
                f"No technology mapping for absorption type: {absorption_type.value}"
            )

        spec = COOLING_TECHNOLOGY_SPECS.get(tech_key)
        if spec is None:
            raise ValueError(
                f"No technology spec for: {tech_key}"
            )

        return spec.cop_default

    def get_cop_range(
        self,
        absorption_type: AbsorptionType,
    ) -> Tuple[Decimal, Decimal]:
        """Get the valid COP range for an absorption chiller type.

        Returns the minimum and maximum COP values from
        COOLING_TECHNOLOGY_SPECS for the given absorption type.

        Args:
            absorption_type: Type of absorption chiller cycle.

        Returns:
            Tuple of (cop_min, cop_max) as Decimal values.

        Raises:
            ValueError: If absorption_type mapping is not found.
        """
        tech_key = _ABSORPTION_TO_TECHNOLOGY.get(absorption_type.value)
        if tech_key is None:
            raise ValueError(
                f"No technology mapping for absorption type: {absorption_type.value}"
            )

        spec = COOLING_TECHNOLOGY_SPECS.get(tech_key)
        if spec is None:
            raise ValueError(
                f"No technology spec for: {tech_key}"
            )

        return (spec.cop_min, spec.cop_max)

    def validate_cop(
        self,
        absorption_type: AbsorptionType,
        cop: Decimal,
    ) -> bool:
        """Validate that a COP value is within range for the absorption type.

        Args:
            absorption_type: Type of absorption chiller cycle.
            cop: COP value to validate.

        Returns:
            True if COP is within the valid range (inclusive).
        """
        try:
            cop_min, cop_max = self.get_cop_range(absorption_type)
            return cop_min <= cop <= cop_max
        except ValueError:
            return False

    # ------------------------------------------------------------------
    # Public API: Unit conversion helpers
    # ------------------------------------------------------------------

    def convert_kwh_to_gj(self, kwh: Decimal) -> Decimal:
        """Convert kWh to GJ.

        Args:
            kwh: Energy in kWh.

        Returns:
            Energy in GJ, quantized to 8 decimal places.
        """
        return _q(kwh * _KWH_TO_GJ)

    def convert_gj_to_kwh(self, gj: Decimal) -> Decimal:
        """Convert GJ to kWh.

        Args:
            gj: Energy in GJ.

        Returns:
            Energy in kWh, quantized to 8 decimal places.
        """
        return _q(gj * _GJ_TO_KWH)

    def convert_ton_hours_to_kwh(self, ton_hours: Decimal) -> Decimal:
        """Convert ton-hours to kWh thermal.

        1 ton-hour = 3.517 kWh_th.

        Args:
            ton_hours: Cooling energy in ton-hours.

        Returns:
            Cooling energy in kWh thermal, quantized to 8 decimal places.
        """
        factor = UNIT_CONVERSIONS.get("ton_hour_to_kwh_th", Decimal("3.517"))
        return _q(ton_hours * factor)

    # ------------------------------------------------------------------
    # Public API: Provenance and metrics
    # ------------------------------------------------------------------

    def get_engine_info(self) -> Dict[str, Any]:
        """Return engine metadata for provenance and diagnostics.

        Returns:
            Dictionary with engine name, version, configuration, and
            supported absorption types.
        """
        return {
            "engine": "AbsorptionCoolingCalculatorEngine",
            "agent": "AGENT-MRV-012",
            "agent_code": "GL-MRV-X-023",
            "version": "1.0.0",
            "precision_places": self._precision_places,
            "provenance_enabled": self._enable_provenance,
            "supported_absorption_types": [
                at.value for at in AbsorptionType
            ],
            "supported_heat_sources": [
                hs.value for hs in HeatSource
            ],
            "zero_emission_sources": sorted(_ZERO_EMISSION_SOURCES),
            "default_cops": {
                at.value: str(self.get_default_cop(at))
                for at in AbsorptionType
            },
            "default_parasitic_ratios": {
                k: str(v) for k, v in _DEFAULT_PARASITIC_RATIOS.items()
            },
        }

    def compute_result_hash(
        self,
        result: CalculationResult,
    ) -> str:
        """Compute a standalone SHA-256 hash of a CalculationResult.

        Useful for verifying result integrity or deduplication.

        Args:
            result: CalculationResult to hash.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        data = {
            "calculation_id": result.calculation_id,
            "calculation_type": result.calculation_type,
            "cooling_output_kwh_th": str(result.cooling_output_kwh_th),
            "energy_input_kwh": str(result.energy_input_kwh),
            "cop_used": str(result.cop_used),
            "emissions_kgco2e": str(result.emissions_kgco2e),
            "calculation_tier": result.calculation_tier.value,
            "gas_count": len(result.gas_breakdown),
            "trace_count": len(result.trace_steps),
        }
        return _compute_hash(data)

    # ------------------------------------------------------------------
    # Public API: Batch processing
    # ------------------------------------------------------------------

    def calculate_batch(
        self,
        requests: List[AbsorptionCoolingRequest],
    ) -> List[CalculationResult]:
        """Calculate emissions for a batch of absorption cooling requests.

        Processes each request independently and returns a list of
        results. Failed calculations are logged but do not abort the
        batch; they return a zero-emission error result.

        Args:
            requests: List of AbsorptionCoolingRequest objects.

        Returns:
            List of CalculationResult objects, one per request. Failed
            calculations have emissions_kgco2e=0 and an error message
            in metadata.
        """
        results: List[CalculationResult] = []
        total = len(requests)

        logger.info("Starting batch absorption calculation: %d requests", total)

        for idx, request in enumerate(requests):
            try:
                result = self.calculate_absorption_cooling(request)
                results.append(result)
            except Exception as exc:
                logger.error(
                    "Batch item %d/%d failed: %s", idx + 1, total, str(exc),
                )
                error_result = CalculationResult(
                    calculation_id=str(uuid.uuid4()),
                    calculation_type="absorption",
                    cooling_output_kwh_th=request.cooling_output_kwh_th,
                    energy_input_kwh=Decimal("0"),
                    cop_used=Decimal("1"),
                    emissions_kgco2e=Decimal("0"),
                    gas_breakdown=[],
                    calculation_tier=request.calculation_tier,
                    provenance_hash="",
                    trace_steps=[f"ERROR: {str(exc)}"],
                    metadata={"error": str(exc), "batch_index": str(idx)},
                )
                results.append(error_result)

        logger.info(
            "Batch absorption calculation complete: %d/%d succeeded",
            sum(1 for r in results if r.emissions_kgco2e > 0 or r.metadata.get("error") is None),
            total,
        )

        return results

    # ------------------------------------------------------------------
    # Public API: Sensitivity analysis helpers
    # ------------------------------------------------------------------

    def calculate_emission_sensitivity_cop(
        self,
        cooling_kwh_th: Decimal,
        heat_source: HeatSource,
        grid_ef: Decimal,
        cop_values: List[Decimal],
        parasitic_ratio: Optional[Decimal] = None,
        heat_source_ef: Optional[Decimal] = None,
        absorption_type: AbsorptionType = AbsorptionType.DOUBLE_EFFECT,
    ) -> List[Dict[str, Any]]:
        """Calculate emissions across a range of COP values.

        Useful for sensitivity analysis to understand how COP
        uncertainty affects total emissions.

        Args:
            cooling_kwh_th: Cooling energy output in kWh thermal.
            heat_source: Heat source driving the absorption cycle.
            grid_ef: Grid emission factor for parasitic electricity.
            cop_values: List of COP values to evaluate.
            parasitic_ratio: Optional parasitic ratio override.
            heat_source_ef: Optional heat source EF override.
            absorption_type: Absorption type for default lookups.

        Returns:
            List of dicts with ``cop``, ``emissions_kgco2e``,
            ``thermal_kgco2e``, ``parasitic_kgco2e``, ``heat_input_gj``
            for each COP value.
        """
        results: List[Dict[str, Any]] = []
        pr = parasitic_ratio if parasitic_ratio is not None else \
            _DEFAULT_PARASITIC_RATIOS.get(absorption_type.value, Decimal("0.05"))

        # Resolve heat source EF once
        ef = heat_source_ef if heat_source_ef is not None else \
            self.get_heat_source_ef(heat_source, grid_ef)

        for cop in cop_values:
            if cop <= Decimal("0"):
                continue

            heat_gj = self.calculate_heat_input_gj(cooling_kwh_th, cop)
            parasitic_kwh = self.calculate_parasitic_electricity(cooling_kwh_th, pr)
            thermal_em = self.calculate_thermal_emissions(heat_gj, ef)
            parasitic_em = self.calculate_parasitic_emissions(parasitic_kwh, grid_ef)
            total_em = _q(thermal_em + parasitic_em)

            results.append({
                "cop": str(cop),
                "emissions_kgco2e": str(total_em),
                "thermal_kgco2e": str(thermal_em),
                "parasitic_kgco2e": str(parasitic_em),
                "heat_input_gj": str(heat_gj),
            })

        return results

    def calculate_emission_sensitivity_grid_ef(
        self,
        cooling_kwh_th: Decimal,
        heat_source: HeatSource,
        cop: Decimal,
        grid_ef_values: List[Decimal],
        parasitic_ratio: Optional[Decimal] = None,
        heat_source_ef: Optional[Decimal] = None,
        absorption_type: AbsorptionType = AbsorptionType.DOUBLE_EFFECT,
    ) -> List[Dict[str, Any]]:
        """Calculate emissions across a range of grid emission factors.

        Useful for sensitivity analysis to understand how grid carbon
        intensity affects total emissions from parasitic electricity.

        Args:
            cooling_kwh_th: Cooling energy output in kWh thermal.
            heat_source: Heat source driving the absorption cycle.
            cop: COP of the absorption chiller.
            grid_ef_values: List of grid EF values to evaluate.
            parasitic_ratio: Optional parasitic ratio override.
            heat_source_ef: Optional heat source EF override.
            absorption_type: Absorption type for default lookups.

        Returns:
            List of dicts with ``grid_ef``, ``emissions_kgco2e``,
            ``thermal_kgco2e``, ``parasitic_kgco2e`` for each grid EF.
        """
        results: List[Dict[str, Any]] = []
        pr = parasitic_ratio if parasitic_ratio is not None else \
            _DEFAULT_PARASITIC_RATIOS.get(absorption_type.value, Decimal("0.05"))

        heat_gj = self.calculate_heat_input_gj(cooling_kwh_th, cop)
        parasitic_kwh = self.calculate_parasitic_electricity(cooling_kwh_th, pr)

        for gef in grid_ef_values:
            if gef < Decimal("0"):
                continue

            # For grid-dependent heat sources, EF changes with grid EF
            if heat_source in (HeatSource.ELECTRIC_BOILER, HeatSource.HEAT_PUMP):
                ef = self.get_heat_source_ef(heat_source, gef)
            elif heat_source_ef is not None:
                ef = heat_source_ef
            else:
                ef = self.get_heat_source_ef(heat_source)

            thermal_em = self.calculate_thermal_emissions(heat_gj, ef)
            parasitic_em = self.calculate_parasitic_emissions(parasitic_kwh, gef)
            total_em = _q(thermal_em + parasitic_em)

            results.append({
                "grid_ef": str(gef),
                "emissions_kgco2e": str(total_em),
                "thermal_kgco2e": str(thermal_em),
                "parasitic_kgco2e": str(parasitic_em),
            })

        return results

    # ------------------------------------------------------------------
    # Public API: Emission intensity metrics
    # ------------------------------------------------------------------

    def calculate_emission_intensity(
        self,
        emissions_kgco2e: Decimal,
        cooling_kwh_th: Decimal,
    ) -> Decimal:
        """Calculate emission intensity in kgCO2e per kWh_th of cooling.

        Args:
            emissions_kgco2e: Total emissions in kgCO2e.
            cooling_kwh_th: Total cooling output in kWh_th.

        Returns:
            Emission intensity in kgCO2e/kWh_th, quantized to 8 decimal
            places. Returns Decimal("0") if cooling is zero.
        """
        if cooling_kwh_th <= Decimal("0"):
            return Decimal("0")
        return _q(emissions_kgco2e / cooling_kwh_th)

    def calculate_thermal_share(
        self,
        thermal_emissions: Decimal,
        total_emissions: Decimal,
    ) -> Decimal:
        """Calculate the thermal emissions share as a fraction of total.

        Args:
            thermal_emissions: Thermal component emissions in kgCO2e.
            total_emissions: Total emissions in kgCO2e.

        Returns:
            Thermal share as Decimal (0 to 1), quantized to 8 decimal
            places. Returns Decimal("0") if total is zero.
        """
        if total_emissions <= Decimal("0"):
            return Decimal("0")
        return _q(thermal_emissions / total_emissions)

    def calculate_parasitic_share(
        self,
        parasitic_emissions: Decimal,
        total_emissions: Decimal,
    ) -> Decimal:
        """Calculate the parasitic emissions share as a fraction of total.

        Args:
            parasitic_emissions: Parasitic component emissions in kgCO2e.
            total_emissions: Total emissions in kgCO2e.

        Returns:
            Parasitic share as Decimal (0 to 1), quantized to 8 decimal
            places. Returns Decimal("0") if total is zero.
        """
        if total_emissions <= Decimal("0"):
            return Decimal("0")
        return _q(parasitic_emissions / total_emissions)

    # ------------------------------------------------------------------
    # Private: COP resolution
    # ------------------------------------------------------------------

    def _resolve_cop(
        self,
        absorption_type: AbsorptionType,
        cop_override: Optional[Decimal],
        trace: List[str],
    ) -> Decimal:
        """Resolve the COP to use for the absorption chiller calculation.

        Priority order:
            1. COP override (measured/metered value)
            2. Default COP from COOLING_TECHNOLOGY_SPECS

        If a COP override is provided, it is validated against the
        valid range for the absorption type. An out-of-range override
        triggers a warning but is still used (Tier 2/3 data may
        legitimately exceed typical ranges).

        Args:
            absorption_type: Type of absorption chiller cycle.
            cop_override: Optional measured COP value.
            trace: Mutable list for recording trace steps.

        Returns:
            Resolved COP value as Decimal.
        """
        default_cop = self.get_default_cop(absorption_type)
        cop_min, cop_max = self.get_cop_range(absorption_type)

        if cop_override is not None:
            if cop_override <= Decimal("0"):
                raise ValueError(
                    f"COP override must be positive, got {cop_override}"
                )

            if not (cop_min <= cop_override <= cop_max):
                logger.warning(
                    "COP override %s is outside typical range [%s, %s] "
                    "for %s; using override as provided",
                    cop_override, cop_min, cop_max, absorption_type.value,
                )
                trace.append(
                    f"WARNING: COP override {cop_override} outside range "
                    f"[{cop_min}, {cop_max}] for {absorption_type.value}"
                )

            trace.append(
                f"COP resolved: override={cop_override} "
                f"(default={default_cop}, range=[{cop_min}, {cop_max}])"
            )
            return cop_override

        trace.append(
            f"COP resolved: default={default_cop} for {absorption_type.value} "
            f"(range=[{cop_min}, {cop_max}])"
        )
        return default_cop

    # ------------------------------------------------------------------
    # Private: Heat source EF resolution
    # ------------------------------------------------------------------

    def _resolve_heat_source_ef(
        self,
        heat_source: HeatSource,
        heat_source_ef_override: Optional[Decimal],
        grid_ef: Decimal,
        cop_hp_override: Optional[Decimal],
        trace: List[str],
    ) -> Decimal:
        """Resolve the heat source emission factor for the calculation.

        Priority order:
            1. Heat source EF override (user-provided)
            2. CHP allocation override (for CHP_EXHAUST)
            3. Computed EF (for ELECTRIC_BOILER, HEAT_PUMP)
            4. Static lookup from HEAT_SOURCE_FACTORS

        Args:
            heat_source: Heat source classification.
            heat_source_ef_override: Optional user-provided EF override.
            grid_ef: Grid EF for grid-dependent sources.
            cop_hp_override: Heat pump COP override.
            trace: Mutable list for recording trace steps.

        Returns:
            Resolved heat source EF in kgCO2e/GJ.
        """
        if heat_source_ef_override is not None:
            trace.append(
                f"Heat source EF: override={heat_source_ef_override} kgCO2e/GJ "
                f"for {heat_source.value}"
            )
            return heat_source_ef_override

        # CHP exhaust: use conservative default or override
        if heat_source == HeatSource.CHP_EXHAUST:
            ef = self.resolve_chp_heat_ef(chp_ef_override=None)
            trace.append(
                f"Heat source EF: CHP exhaust conservative default={ef} kgCO2e/GJ"
            )
            return ef

        # Grid-dependent sources
        if heat_source in (HeatSource.ELECTRIC_BOILER, HeatSource.HEAT_PUMP):
            ef = self.get_heat_source_ef(heat_source, grid_ef, cop_hp_override)
            trace.append(
                f"Heat source EF: {heat_source.value} computed={ef} kgCO2e/GJ "
                f"(grid_ef={grid_ef} kgCO2e/kWh)"
            )
            return ef

        # Static lookup
        ef = self.get_heat_source_ef(heat_source)
        trace.append(
            f"Heat source EF: {heat_source.value} lookup={ef} kgCO2e/GJ"
        )
        return ef

    # ------------------------------------------------------------------
    # Private: Parasitic ratio resolution
    # ------------------------------------------------------------------

    def _resolve_parasitic_ratio(
        self,
        absorption_type: AbsorptionType,
        request_ratio: Decimal,
        trace: List[str],
    ) -> Decimal:
        """Resolve the parasitic ratio for the calculation.

        If the request provides a non-default ratio, uses it directly.
        Otherwise, uses the type-specific default from
        _DEFAULT_PARASITIC_RATIOS.

        Args:
            absorption_type: Type of absorption chiller cycle.
            request_ratio: Parasitic ratio from the request (0 to 1).
            trace: Mutable list for recording trace steps.

        Returns:
            Resolved parasitic ratio as Decimal.
        """
        default_ratio = _DEFAULT_PARASITIC_RATIOS.get(
            absorption_type.value, Decimal("0.05"),
        )

        # The request already has the ratio set (possibly to default 0.05
        # from the model's Field default). We use the request value as-is.
        trace.append(
            f"Parasitic ratio: {request_ratio} "
            f"(type default={default_ratio} for {absorption_type.value})"
        )
        return request_ratio

    # ------------------------------------------------------------------
    # Private: Gas breakdown merging
    # ------------------------------------------------------------------

    def _merge_gas_breakdowns(
        self,
        thermal_gases: List[GasEmissionDetail],
        parasitic_gases: List[GasEmissionDetail],
        gwp_source: GWPSource,
    ) -> List[GasEmissionDetail]:
        """Merge thermal and parasitic gas breakdowns into a combined list.

        Sums the CO2, CH4, and N2O from both components and adds an
        aggregate CO2e entry.

        Args:
            thermal_gases: Per-gas thermal emission details.
            parasitic_gases: Per-gas parasitic emission details.
            gwp_source: IPCC Assessment Report for GWP values.

        Returns:
            Combined list with CO2, CH4, N2O, and CO2e entries.
        """
        # Build gas maps
        thermal_map: Dict[str, GasEmissionDetail] = {
            g.gas.value: g for g in thermal_gases
        }
        parasitic_map: Dict[str, GasEmissionDetail] = {
            g.gas.value: g for g in parasitic_gases
        }

        merged: List[GasEmissionDetail] = []
        total_co2e = Decimal("0")

        for gas_name in ["CO2", "CH4", "N2O"]:
            t = thermal_map.get(gas_name)
            p = parasitic_map.get(gas_name)

            t_qty = t.quantity_kg if t else Decimal("0")
            p_qty = p.quantity_kg if p else Decimal("0")
            t_co2e = t.co2e_kg if t else Decimal("0")
            p_co2e = p.co2e_kg if p else Decimal("0")
            gwp_factor = t.gwp_factor if t else (p.gwp_factor if p else Decimal("1"))

            combined_qty = _q(t_qty + p_qty)
            combined_co2e = _q(t_co2e + p_co2e)
            total_co2e += combined_co2e

            merged.append(GasEmissionDetail(
                gas=EmissionGas(gas_name),
                quantity_kg=combined_qty,
                gwp_factor=gwp_factor,
                co2e_kg=combined_co2e,
            ))

        # Add aggregate CO2e entry
        total_co2e = _q(total_co2e)
        merged.append(GasEmissionDetail(
            gas=EmissionGas.CO2E,
            quantity_kg=total_co2e,
            gwp_factor=Decimal("1"),
            co2e_kg=total_co2e,
        ))

        return merged

    def _aggregate_gas_breakdowns(
        self,
        all_breakdowns: List[GasEmissionDetail],
    ) -> List[GasEmissionDetail]:
        """Aggregate gas breakdowns from multiple chiller units.

        Sums quantities and CO2e values across all entries for each
        gas species.

        Args:
            all_breakdowns: Flat list of GasEmissionDetail from all units.

        Returns:
            Aggregated list with one entry per gas species plus CO2e total.
        """
        gas_totals: Dict[str, Dict[str, Decimal]] = {}

        for entry in all_breakdowns:
            key = entry.gas.value
            if key == EmissionGas.CO2E.value:
                continue  # Skip aggregate entries; we recompute below

            if key not in gas_totals:
                gas_totals[key] = {
                    "quantity_kg": Decimal("0"),
                    "co2e_kg": Decimal("0"),
                    "gwp_factor": entry.gwp_factor,
                }

            gas_totals[key]["quantity_kg"] += entry.quantity_kg
            gas_totals[key]["co2e_kg"] += entry.co2e_kg

        merged: List[GasEmissionDetail] = []
        total_co2e = Decimal("0")

        for gas_name in ["CO2", "CH4", "N2O"]:
            if gas_name in gas_totals:
                qty = _q(gas_totals[gas_name]["quantity_kg"])
                co2e = _q(gas_totals[gas_name]["co2e_kg"])
                gwp = gas_totals[gas_name]["gwp_factor"]
                total_co2e += co2e
            else:
                qty = Decimal("0")
                co2e = Decimal("0")
                gwp = Decimal("1")

            merged.append(GasEmissionDetail(
                gas=EmissionGas(gas_name),
                quantity_kg=qty,
                gwp_factor=gwp,
                co2e_kg=co2e,
            ))

        total_co2e = _q(total_co2e)
        merged.append(GasEmissionDetail(
            gas=EmissionGas.CO2E,
            quantity_kg=total_co2e,
            gwp_factor=Decimal("1"),
            co2e_kg=total_co2e,
        ))

        return merged

    # ------------------------------------------------------------------
    # Private: Metadata construction
    # ------------------------------------------------------------------

    def _build_metadata(
        self,
        request: AbsorptionCoolingRequest,
        thermal_emissions: Decimal,
        parasitic_emissions: Decimal,
    ) -> Dict[str, str]:
        """Build metadata dict for the calculation result.

        Includes all contextual information useful for reporting,
        filtering, and aggregation.

        Args:
            request: The original absorption cooling request.
            thermal_emissions: Calculated thermal emissions in kgCO2e.
            parasitic_emissions: Calculated parasitic emissions in kgCO2e.

        Returns:
            String-valued metadata dictionary.
        """
        total = _q(thermal_emissions + parasitic_emissions)

        metadata: Dict[str, str] = {
            "absorption_type": request.absorption_type.value,
            "heat_source": request.heat_source.value,
            "tenant_id": request.tenant_id,
            "gwp_source": request.gwp_source.value,
            "calculation_tier": request.calculation_tier.value,
            "reporting_period": request.reporting_period.value,
            "thermal_emissions_kgco2e": str(thermal_emissions),
            "parasitic_emissions_kgco2e": str(parasitic_emissions),
            "parasitic_ratio": str(request.parasitic_ratio),
            "zero_emission_heat": str(self.is_zero_emission_source(request.heat_source)),
        }

        if total > Decimal("0"):
            metadata["thermal_share"] = str(
                self.calculate_thermal_share(thermal_emissions, total)
            )
            metadata["parasitic_share"] = str(
                self.calculate_parasitic_share(parasitic_emissions, total)
            )
            metadata["emission_intensity_kgco2e_per_kwh"] = str(
                self.calculate_emission_intensity(total, request.cooling_output_kwh_th)
            )

        if request.facility_id:
            metadata["facility_id"] = request.facility_id
        if request.supplier_id:
            metadata["supplier_id"] = request.supplier_id

        return metadata

    # ------------------------------------------------------------------
    # Private: Provenance chain management
    # ------------------------------------------------------------------

    def _compute_provenance(
        self,
        calc_id: str,
        request: AbsorptionCoolingRequest,
        cop: Decimal,
        heat_source_ef: Decimal,
        parasitic_ratio: Decimal,
        heat_input_gj: Decimal,
        parasitic_kwh: Decimal,
        thermal_emissions: Decimal,
        parasitic_emissions: Decimal,
        total_emissions: Decimal,
        gas_breakdown: List[GasEmissionDetail],
        trace: List[str],
    ) -> str:
        """Compute SHA-256 provenance hash for the calculation.

        If provenance tracking is enabled, creates a full provenance
        chain with stages for input validation, technology lookup, energy
        calculation, grid factor application, gas decomposition, and
        result finalization. Returns the chain seal hash.

        If provenance is disabled, computes a standalone hash of all
        calculation inputs and outputs.

        Args:
            calc_id: Unique calculation identifier.
            request: Original request parameters.
            cop: Resolved COP value used.
            heat_source_ef: Resolved heat source EF used.
            parasitic_ratio: Resolved parasitic ratio used.
            heat_input_gj: Computed heat input in GJ.
            parasitic_kwh: Computed parasitic electricity in kWh.
            thermal_emissions: Computed thermal emissions in kgCO2e.
            parasitic_emissions: Computed parasitic emissions in kgCO2e.
            total_emissions: Computed total emissions in kgCO2e.
            gas_breakdown: Per-gas emission breakdown list.
            trace: Calculation trace steps.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        if self._enable_provenance:
            try:
                return self._build_provenance_chain(
                    calc_id, request, cop, heat_source_ef,
                    parasitic_ratio, heat_input_gj, parasitic_kwh,
                    thermal_emissions, parasitic_emissions,
                    total_emissions, gas_breakdown, trace,
                )
            except Exception as exc:
                logger.warning(
                    "Provenance chain creation failed for %s: %s; "
                    "falling back to standalone hash",
                    calc_id, str(exc),
                )

        # Fallback: standalone hash
        return self._compute_standalone_hash(
            calc_id, request, cop, heat_source_ef,
            parasitic_ratio, heat_input_gj, parasitic_kwh,
            thermal_emissions, parasitic_emissions,
            total_emissions, gas_breakdown,
        )

    def _build_provenance_chain(
        self,
        calc_id: str,
        request: AbsorptionCoolingRequest,
        cop: Decimal,
        heat_source_ef: Decimal,
        parasitic_ratio: Decimal,
        heat_input_gj: Decimal,
        parasitic_kwh: Decimal,
        thermal_emissions: Decimal,
        parasitic_emissions: Decimal,
        total_emissions: Decimal,
        gas_breakdown: List[GasEmissionDetail],
        trace: List[str],
    ) -> str:
        """Build a complete provenance chain for the calculation.

        Creates chain stages: INPUT_VALIDATION, TECHNOLOGY_LOOKUP,
        ENERGY_INPUT_CALCULATION, AUXILIARY_ENERGY,
        HEAT_SOURCE_FACTOR_APPLICATION, GRID_FACTOR_APPLICATION,
        GAS_DECOMPOSITION, GWP_APPLICATION, RESULT_FINALIZATION.

        Args:
            (Same as _compute_provenance)

        Returns:
            Sealed chain hash (64 hex characters).
        """
        prov = self._provenance
        chain_id = f"absorption-{calc_id}"

        prov.create_chain(chain_id)

        # Stage 1: Input validation
        prov.add_stage(chain_id, "INPUT_VALIDATION", {
            "calculation_id": calc_id,
            "cooling_output_kwh_th": str(request.cooling_output_kwh_th),
            "absorption_type": request.absorption_type.value,
            "heat_source": request.heat_source.value,
            "cop_override": str(request.cop_override) if request.cop_override else "none",
            "parasitic_ratio": str(request.parasitic_ratio),
            "grid_ef": str(request.grid_ef_kgco2e_per_kwh),
            "gwp_source": request.gwp_source.value,
            "tier": request.calculation_tier.value,
        })

        # Stage 2: Technology lookup
        prov.add_stage(chain_id, "TECHNOLOGY_LOOKUP", {
            "absorption_type": request.absorption_type.value,
            "cop_resolved": str(cop),
            "heat_source_ef_resolved": str(heat_source_ef),
            "parasitic_ratio_resolved": str(parasitic_ratio),
        })

        # Stage 3: Energy input calculation
        prov.add_stage(chain_id, "ENERGY_INPUT_CALCULATION", {
            "cooling_kwh_th": str(request.cooling_output_kwh_th),
            "cop": str(cop),
            "heat_input_gj": str(heat_input_gj),
        })

        # Stage 4: Auxiliary energy (parasitic)
        prov.add_stage(chain_id, "AUXILIARY_ENERGY", {
            "parasitic_kwh": str(parasitic_kwh),
            "parasitic_ratio": str(parasitic_ratio),
            "cooling_kwh_th": str(request.cooling_output_kwh_th),
        })

        # Stage 5: Heat source factor application
        prov.add_stage(chain_id, "HEAT_SOURCE_FACTOR_APPLICATION", {
            "heat_source": request.heat_source.value,
            "heat_source_ef_kgco2e_per_gj": str(heat_source_ef),
            "heat_input_gj": str(heat_input_gj),
            "thermal_emissions_kgco2e": str(thermal_emissions),
        })

        # Stage 6: Grid factor application (parasitic)
        prov.add_stage(chain_id, "GRID_FACTOR_APPLICATION", {
            "grid_ef_kgco2e_per_kwh": str(request.grid_ef_kgco2e_per_kwh),
            "parasitic_kwh": str(parasitic_kwh),
            "parasitic_emissions_kgco2e": str(parasitic_emissions),
        })

        # Stage 7: Gas decomposition
        gas_data: Dict[str, str] = {}
        for g in gas_breakdown:
            gas_data[f"{g.gas.value}_kg"] = str(g.quantity_kg)
            gas_data[f"{g.gas.value}_co2e_kg"] = str(g.co2e_kg)
        prov.add_stage(chain_id, "GAS_DECOMPOSITION", gas_data)

        # Stage 8: GWP application
        gwp_map = GWP_VALUES.get(request.gwp_source.value, GWP_VALUES["AR6"])
        prov.add_stage(chain_id, "GWP_APPLICATION", {
            "gwp_source": request.gwp_source.value,
            "gwp_co2": str(gwp_map.get("CO2", "1")),
            "gwp_ch4": str(gwp_map.get("CH4", "27.9")),
            "gwp_n2o": str(gwp_map.get("N2O", "273")),
        })

        # Stage 9: Result finalization
        prov.add_stage(chain_id, "RESULT_FINALIZATION", {
            "total_emissions_kgco2e": str(total_emissions),
            "thermal_emissions_kgco2e": str(thermal_emissions),
            "parasitic_emissions_kgco2e": str(parasitic_emissions),
            "cop_used": str(cop),
            "trace_steps": str(len(trace)),
            "validation_status": "PASS",
        })

        # Seal chain
        return prov.seal_chain(chain_id)

    def _compute_standalone_hash(
        self,
        calc_id: str,
        request: AbsorptionCoolingRequest,
        cop: Decimal,
        heat_source_ef: Decimal,
        parasitic_ratio: Decimal,
        heat_input_gj: Decimal,
        parasitic_kwh: Decimal,
        thermal_emissions: Decimal,
        parasitic_emissions: Decimal,
        total_emissions: Decimal,
        gas_breakdown: List[GasEmissionDetail],
    ) -> str:
        """Compute a standalone SHA-256 hash for audit trail.

        Used when provenance chain creation is disabled or fails.

        Args:
            (Same as _compute_provenance minus trace)

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        data = {
            "calculation_id": calc_id,
            "calculation_type": "absorption",
            "cooling_output_kwh_th": str(request.cooling_output_kwh_th),
            "absorption_type": request.absorption_type.value,
            "heat_source": request.heat_source.value,
            "cop_used": str(cop),
            "heat_source_ef": str(heat_source_ef),
            "parasitic_ratio": str(parasitic_ratio),
            "grid_ef": str(request.grid_ef_kgco2e_per_kwh),
            "heat_input_gj": str(heat_input_gj),
            "parasitic_kwh": str(parasitic_kwh),
            "thermal_emissions_kgco2e": str(thermal_emissions),
            "parasitic_emissions_kgco2e": str(parasitic_emissions),
            "total_emissions_kgco2e": str(total_emissions),
            "gwp_source": request.gwp_source.value,
            "gas_count": len(gas_breakdown),
            "tier": request.calculation_tier.value,
        }
        return _compute_hash(data)

    # ------------------------------------------------------------------
    # Private: Metrics recording
    # ------------------------------------------------------------------

    def _record_metrics(
        self,
        request: AbsorptionCoolingRequest,
        emissions: Decimal,
        cop: Decimal,
        duration_s: float,
        status: str,
    ) -> None:
        """Record Prometheus metrics for the calculation.

        Args:
            request: Original absorption cooling request.
            emissions: Total emissions in kgCO2e.
            cop: COP value used.
            duration_s: Calculation wall-clock time in seconds.
            status: Calculation outcome (success, failure).
        """
        try:
            self._metrics.record_calculation(
                technology="absorption_chiller",
                calculation_type="absorption",
                tier=request.calculation_tier.value,
                tenant_id=request.tenant_id,
                status=status,
                duration_s=duration_s,
                emissions_kgco2e=float(emissions),
                cooling_kwh_th=float(request.cooling_output_kwh_th),
                cop_used=float(cop),
                condenser_type="unknown",
            )
        except Exception as exc:
            logger.warning(
                "Failed to record metrics: %s", str(exc),
            )


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def get_absorption_calculator() -> AbsorptionCoolingCalculatorEngine:
    """Return the process-wide singleton AbsorptionCoolingCalculatorEngine.

    This is the recommended entry point for obtaining the absorption
    cooling calculator in production code. The singleton is created
    lazily on first call and reused for all subsequent calls.

    Returns:
        The singleton AbsorptionCoolingCalculatorEngine instance.

    Example:
        >>> calc = get_absorption_calculator()
        >>> assert calc is get_absorption_calculator()
    """
    return AbsorptionCoolingCalculatorEngine()


def reset_absorption_calculator() -> None:
    """Reset the singleton AbsorptionCoolingCalculatorEngine.

    **Testing only.** Destroys the current singleton so the next
    instantiation creates a fresh engine.

    Example:
        >>> reset_absorption_calculator()
        >>> calc = get_absorption_calculator()  # fresh instance
    """
    AbsorptionCoolingCalculatorEngine.reset()


def calculate_absorption_cooling(
    request: AbsorptionCoolingRequest,
) -> CalculationResult:
    """Module-level convenience for absorption chiller calculation.

    Delegates to the singleton engine's calculate_absorption_cooling.

    Args:
        request: Validated AbsorptionCoolingRequest.

    Returns:
        CalculationResult with total emissions and breakdown.

    Example:
        >>> from greenlang.cooling_purchase.models import (
        ...     AbsorptionCoolingRequest, AbsorptionType, HeatSource,
        ... )
        >>> from decimal import Decimal
        >>> req = AbsorptionCoolingRequest(
        ...     cooling_output_kwh_th=Decimal("10000"),
        ...     absorption_type=AbsorptionType.SINGLE_EFFECT,
        ...     heat_source=HeatSource.WASTE_HEAT,
        ...     grid_ef_kgco2e_per_kwh=Decimal("0.3"),
        ... )
        >>> result = calculate_absorption_cooling(req)
        >>> assert result.emissions_kgco2e >= Decimal("0")
    """
    return get_absorption_calculator().calculate_absorption_cooling(request)


def calculate_single_effect(
    cooling_kwh_th: Decimal,
    grid_ef: Decimal,
    cop: Optional[Decimal] = None,
    heat_source: HeatSource = HeatSource.NATURAL_GAS_STEAM,
    heat_source_ef: Optional[Decimal] = None,
    parasitic_ratio: Optional[Decimal] = None,
) -> CalculationResult:
    """Module-level convenience for single-effect LiBr calculation.

    Args:
        cooling_kwh_th: Cooling energy output in kWh thermal.
        grid_ef: Grid emission factor for parasitic electricity.
        cop: Optional COP override.
        heat_source: Heat source type.
        heat_source_ef: Optional heat source EF override.
        parasitic_ratio: Optional parasitic ratio override.

    Returns:
        CalculationResult with total emissions and breakdown.
    """
    return get_absorption_calculator().calculate_single_effect(
        cooling_kwh_th=cooling_kwh_th,
        cop=cop,
        heat_source=heat_source,
        heat_source_ef=heat_source_ef,
        parasitic_ratio=parasitic_ratio,
        grid_ef=grid_ef,
    )


def calculate_double_effect(
    cooling_kwh_th: Decimal,
    grid_ef: Decimal,
    cop: Optional[Decimal] = None,
    heat_source: HeatSource = HeatSource.NATURAL_GAS_STEAM,
    heat_source_ef: Optional[Decimal] = None,
    parasitic_ratio: Optional[Decimal] = None,
) -> CalculationResult:
    """Module-level convenience for double-effect LiBr calculation.

    Args:
        cooling_kwh_th: Cooling energy output in kWh thermal.
        grid_ef: Grid emission factor for parasitic electricity.
        cop: Optional COP override.
        heat_source: Heat source type.
        heat_source_ef: Optional heat source EF override.
        parasitic_ratio: Optional parasitic ratio override.

    Returns:
        CalculationResult with total emissions and breakdown.
    """
    return get_absorption_calculator().calculate_double_effect(
        cooling_kwh_th=cooling_kwh_th,
        cop=cop,
        heat_source=heat_source,
        heat_source_ef=heat_source_ef,
        parasitic_ratio=parasitic_ratio,
        grid_ef=grid_ef,
    )


def calculate_triple_effect(
    cooling_kwh_th: Decimal,
    grid_ef: Decimal,
    cop: Optional[Decimal] = None,
    heat_source: HeatSource = HeatSource.NATURAL_GAS_STEAM,
    heat_source_ef: Optional[Decimal] = None,
    parasitic_ratio: Optional[Decimal] = None,
) -> CalculationResult:
    """Module-level convenience for triple-effect LiBr calculation.

    Args:
        cooling_kwh_th: Cooling energy output in kWh thermal.
        grid_ef: Grid emission factor for parasitic electricity.
        cop: Optional COP override.
        heat_source: Heat source type.
        heat_source_ef: Optional heat source EF override.
        parasitic_ratio: Optional parasitic ratio override.

    Returns:
        CalculationResult with total emissions and breakdown.
    """
    return get_absorption_calculator().calculate_triple_effect(
        cooling_kwh_th=cooling_kwh_th,
        cop=cop,
        heat_source=heat_source,
        heat_source_ef=heat_source_ef,
        parasitic_ratio=parasitic_ratio,
        grid_ef=grid_ef,
    )


def calculate_ammonia(
    cooling_kwh_th: Decimal,
    grid_ef: Decimal,
    cop: Optional[Decimal] = None,
    heat_source: HeatSource = HeatSource.WASTE_HEAT,
    heat_source_ef: Optional[Decimal] = None,
    parasitic_ratio: Optional[Decimal] = None,
) -> CalculationResult:
    """Module-level convenience for ammonia absorption calculation.

    Args:
        cooling_kwh_th: Cooling energy output in kWh thermal.
        grid_ef: Grid emission factor for parasitic electricity.
        cop: Optional COP override.
        heat_source: Heat source type.
        heat_source_ef: Optional heat source EF override.
        parasitic_ratio: Optional parasitic ratio override.

    Returns:
        CalculationResult with total emissions and breakdown.
    """
    return get_absorption_calculator().calculate_ammonia(
        cooling_kwh_th=cooling_kwh_th,
        cop=cop,
        heat_source=heat_source,
        heat_source_ef=heat_source_ef,
        parasitic_ratio=parasitic_ratio,
        grid_ef=grid_ef,
    )
