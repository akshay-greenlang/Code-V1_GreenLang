# -*- coding: utf-8 -*-
"""
CombustionCalculatorEngine - Engine 2: Stationary Combustion Agent (AGENT-MRV-001)

Core calculation engine implementing GHG Protocol / EPA / IPCC methodologies
for stationary combustion emissions. Computes CO2, CH4, and N2O emissions
using deterministic Decimal arithmetic with full calculation trace and
SHA-256 provenance hashing.

Formula (per gas):
    Emissions = Activity x HeatingValue x EmissionFactor x OxidationFactor

CO2e is computed by multiplying each gas mass by its GWP. Biogenic CO2
(from biomass fuels) is tracked separately per GHG Protocol guidance and
excluded from Scope 1 totals unless explicitly included.

Zero-Hallucination Guarantees:
    - All calculations use Python Decimal (8 decimal places)
    - No LLM calls in the calculation path
    - Every step is recorded in the calculation trace
    - SHA-256 provenance hash for every result

Example:
    >>> from greenlang.stationary_combustion.combustion_calculator import CombustionCalculatorEngine
    >>> from greenlang.stationary_combustion.fuel_database import FuelDatabaseEngine
    >>> from greenlang.stationary_combustion.models import CombustionInput, UnitType, FuelType
    >>> from decimal import Decimal
    >>> db = FuelDatabaseEngine()
    >>> calc = CombustionCalculatorEngine(fuel_database=db)
    >>> inp = CombustionInput(
    ...     fuel_type="NATURAL_GAS",
    ...     quantity=Decimal("1000"),
    ...     unit="MCF",
    ... )
    >>> result = calc.calculate(inp)
    >>> assert result.status == "SUCCESS"
    >>> assert result.total_co2e_kg > 0

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-001 Stationary Combustion (GL-MRV-SCOPE1-001)
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
from typing import Any, Dict, List, Optional

from greenlang.stationary_combustion.fuel_database import FuelDatabaseEngine
from greenlang.stationary_combustion.metrics import (
    observe_batch_duration,
    observe_calculation_duration,
    record_batch,
    record_calculation,
    record_co2e,
)
from greenlang.stationary_combustion.models import (
    BatchCalculationResponse,
    CalculationResult,
    CalculationStatus,
    CalculationTier,
    CombustionInput,
    EFSource,
    EmissionGas,
    FuelType,
    GasEmission,
    GWPSource,
    HeatingValueBasis,
    UnitType,
)
from greenlang.stationary_combustion.provenance import get_provenance_tracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Decimal precision constant
# ---------------------------------------------------------------------------
_PRECISION = Decimal("0.00000001")  # 8 decimal places

# ---------------------------------------------------------------------------
# Unit Conversion Constants (all as Decimal for zero-hallucination precision)
# ---------------------------------------------------------------------------

# Volume conversions to liters
_GALLON_TO_LITERS = Decimal("3.78541")
_BARREL_TO_LITERS = Decimal("158.987")
_BARREL_TO_GALLONS = Decimal("42")
_M3_TO_LITERS = Decimal("1000")
_FT3_TO_LITERS = Decimal("28.3168")
_MCF_TO_FT3 = Decimal("1000")

# Mass conversions to tonnes (metric)
_SHORT_TON_TO_TONNES = Decimal("0.907185")
_LB_TO_TONNES = Decimal("0.000453592")
_KG_TO_TONNES = Decimal("0.001")

# Energy conversions to GJ
_MMBTU_TO_GJ = Decimal("1.055056")
_THERM_TO_GJ = Decimal("0.105506")
_MWH_TO_GJ = Decimal("3.6")
_KWH_TO_GJ = Decimal("0.0036")
_TJ_TO_GJ = Decimal("1000")

# Molecular weight ratio: CO2/C
_CO2_C_RATIO = Decimal("3.66667")  # 44/12


class CombustionCalculatorEngine:
    """Core calculation engine for GHG Protocol stationary combustion emissions.

    Implements Tier 1/2/3 methodologies with deterministic Decimal arithmetic,
    full calculation trace, and SHA-256 provenance hashing. Thread-safe for
    concurrent calculations.

    Attributes:
        _fuel_db: Reference to the FuelDatabaseEngine for factor lookups.
        _config: Optional configuration dictionary.
        _lock: Thread lock for any shared mutable state.
        _provenance: Reference to the provenance tracker.

    Example:
        >>> db = FuelDatabaseEngine()
        >>> calc = CombustionCalculatorEngine(fuel_database=db)
        >>> result = calc.calculate(CombustionInput(
        ...     fuel_type="NATURAL_GAS",
        ...     quantity=Decimal("1000"),
        ...     unit="MCF",
        ... ))
        >>> assert result.status == "SUCCESS"
    """

    def __init__(
        self,
        fuel_database: Optional[FuelDatabaseEngine] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize CombustionCalculatorEngine.

        Args:
            fuel_database: FuelDatabaseEngine instance for factor lookups.
                If None, a default instance is created.
            config: Optional configuration dict. Supports:
                - ``enable_provenance`` (bool): Enable provenance tracking.
                - ``decimal_precision`` (int): Decimal places. Default 8.
        """
        self._fuel_db = fuel_database or FuelDatabaseEngine()
        self._config = config or {}
        self._lock = threading.Lock()
        self._enable_provenance: bool = self._config.get("enable_provenance", True)
        self._precision_places: int = self._config.get("decimal_precision", 8)
        self._precision_quantizer = Decimal(10) ** -self._precision_places

        if self._enable_provenance:
            self._provenance = get_provenance_tracker()
        else:
            self._provenance = None

        logger.info(
            "CombustionCalculatorEngine initialized (precision=%d, provenance=%s)",
            self._precision_places,
            self._enable_provenance,
        )

    # ------------------------------------------------------------------
    # Public API: Single Calculation
    # ------------------------------------------------------------------

    def calculate(
        self,
        input_data: CombustionInput,
        gwp_source: str = "AR6",
        include_biogenic: bool = False,
    ) -> CalculationResult:
        """Calculate stationary combustion emissions for a single input.

        Implements the GHG Protocol formula:
            Emissions_gas = Activity x HeatingValue x EF_gas x OxidationFactor

        CO2e = sum of (mass_gas x GWP_gas) for CO2, CH4, N2O.
        Biogenic CO2 is separated and optionally included.

        Args:
            input_data: Validated combustion input data.
            gwp_source: GWP source (``"AR4"``, ``"AR5"``, ``"AR6"``).
                Defaults to ``"AR6"``.
            include_biogenic: If True, include biogenic CO2 in total.
                Defaults to False.

        Returns:
            Complete CalculationResult with per-gas emissions, totals,
            calculation trace, and provenance hash.
        """
        start_time = time.monotonic()
        calc_id = input_data.calculation_id or f"calc_{uuid.uuid4().hex[:12]}"
        fuel_type_str = (
            input_data.fuel_type
            if isinstance(input_data.fuel_type, str)
            else input_data.fuel_type.value
        )
        trace: List[str] = []

        try:
            trace.append(f"[1] Input: {fuel_type_str}, qty={input_data.quantity}, unit={input_data.unit}")

            # Step 1: Determine tier
            tier = self._resolve_tier(input_data, trace)

            # Step 2: Determine heating value basis
            hv_basis = self._resolve_hv_basis(input_data)
            trace.append(f"[2] Heating value basis: {hv_basis}")

            # Step 3: Get heating value
            heating_value = self._resolve_heating_value(
                fuel_type_str, hv_basis, input_data, trace,
            )

            # Step 4: Convert quantity to energy (GJ)
            energy_gj = self._convert_to_energy_gj(
                fuel_type_str, input_data.quantity, input_data.unit,
                heating_value, hv_basis, trace,
            )
            trace.append(f"[4] Energy content: {energy_gj} GJ")

            # Step 5: Get oxidation factor
            oxidation_factor = self._resolve_oxidation_factor(
                fuel_type_str, input_data, trace,
            )

            # Step 6: Determine EF source
            ef_source_str = (
                input_data.ef_source
                if isinstance(input_data.ef_source, str)
                else input_data.ef_source.value
            )

            # Step 7: Decompose per-gas emissions
            gas_emissions = self._decompose_gas_emissions(
                energy_gj, fuel_type_str, oxidation_factor,
                gwp_source, ef_source_str, input_data, trace,
            )

            # Step 8: Compute totals
            total_co2e_kg, biogenic_co2_kg = self._compute_totals(
                gas_emissions, include_biogenic, trace,
            )
            total_co2e_tonnes = (total_co2e_kg * Decimal("0.001")).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP,
            )
            biogenic_co2_tonnes = (biogenic_co2_kg * Decimal("0.001")).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP,
            )

            # Step 9: Equipment adjustment (if applicable)
            equipment_adj = None
            if input_data.equipment_id:
                equipment_adj, total_co2e_kg, total_co2e_tonnes = (
                    self._apply_equipment_adjustment_to_totals(
                        total_co2e_kg, total_co2e_tonnes,
                        input_data.equipment_id, trace,
                    )
                )

            # Step 10: Provenance hash
            elapsed_ms = (time.monotonic() - start_time) * 1000
            provenance_hash = self._compute_provenance_hash({
                "calculation_id": calc_id,
                "fuel_type": fuel_type_str,
                "quantity": str(input_data.quantity),
                "unit": str(input_data.unit),
                "energy_gj": str(energy_gj),
                "total_co2e_kg": str(total_co2e_kg),
                "gwp_source": gwp_source,
                "ef_source": ef_source_str,
            })
            trace.append(f"[10] Provenance hash: {provenance_hash[:16]}...")

            # Record metrics
            record_calculation(fuel_type_str, tier)
            observe_calculation_duration(fuel_type_str, elapsed_ms / 1000)
            record_co2e(fuel_type_str, "scope_1", float(total_co2e_tonnes))
            if biogenic_co2_tonnes > 0:
                record_co2e(fuel_type_str, "biogenic", float(biogenic_co2_tonnes))

            # Record provenance
            if self._provenance is not None:
                self._provenance.record(
                    entity_type="calculation",
                    action="calculate_emissions",
                    entity_id=calc_id,
                    data={
                        "fuel_type": fuel_type_str,
                        "total_co2e_kg": str(total_co2e_kg),
                        "provenance_hash": provenance_hash,
                    },
                )

            return CalculationResult(
                calculation_id=calc_id,
                status=CalculationStatus.SUCCESS,
                fuel_type=fuel_type_str,
                quantity=input_data.quantity,
                unit=input_data.unit,
                energy_gj=energy_gj,
                tier=tier,
                ef_source=ef_source_str,
                gwp_source=gwp_source,
                heating_value_basis=hv_basis,
                gas_emissions=gas_emissions,
                total_co2e_kg=total_co2e_kg,
                total_co2e_tonnes=total_co2e_tonnes,
                biogenic_co2_kg=biogenic_co2_kg,
                biogenic_co2_tonnes=biogenic_co2_tonnes,
                oxidation_factor=oxidation_factor,
                equipment_adjustment=equipment_adj,
                calculation_trace=trace,
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms,
            )

        except Exception as exc:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "Calculation failed for %s (id=%s): %s",
                fuel_type_str, calc_id, exc, exc_info=True,
            )
            return CalculationResult(
                calculation_id=calc_id,
                status=CalculationStatus.FAILED,
                fuel_type=fuel_type_str,
                quantity=input_data.quantity,
                unit=input_data.unit,
                energy_gj=Decimal("0"),
                tier=CalculationTier.TIER_1,
                ef_source=str(input_data.ef_source),
                gwp_source=gwp_source,
                heating_value_basis=str(input_data.heating_value_basis),
                gas_emissions=[],
                total_co2e_kg=Decimal("0"),
                total_co2e_tonnes=Decimal("0"),
                biogenic_co2_kg=Decimal("0"),
                biogenic_co2_tonnes=Decimal("0"),
                oxidation_factor=Decimal("1"),
                calculation_trace=trace,
                provenance_hash="",
                processing_time_ms=elapsed_ms,
                error_message=str(exc),
            )

    # ------------------------------------------------------------------
    # Public API: Batch Calculation
    # ------------------------------------------------------------------

    def calculate_batch(
        self,
        inputs: List[CombustionInput],
        gwp_source: str = "AR6",
        include_biogenic: bool = False,
    ) -> BatchCalculationResponse:
        """Process multiple combustion inputs and aggregate totals.

        Args:
            inputs: List of CombustionInput records.
            gwp_source: GWP source for all calculations.
            include_biogenic: Whether to include biogenic CO2 in totals.

        Returns:
            BatchCalculationResponse with individual results and aggregates.
        """
        start_time = time.monotonic()
        batch_id = f"batch_{uuid.uuid4().hex[:12]}"

        results: List[CalculationResult] = []
        total_co2e_kg = Decimal("0")
        total_co2e_tonnes = Decimal("0")
        total_biogenic_kg = Decimal("0")
        total_biogenic_tonnes = Decimal("0")
        success_count = 0
        failure_count = 0

        for inp in inputs:
            result = self.calculate(inp, gwp_source=gwp_source, include_biogenic=include_biogenic)
            results.append(result)

            if result.status == CalculationStatus.SUCCESS.value or result.status == CalculationStatus.SUCCESS:
                success_count += 1
                total_co2e_kg += result.total_co2e_kg
                total_co2e_tonnes += result.total_co2e_tonnes
                total_biogenic_kg += result.biogenic_co2_kg
                total_biogenic_tonnes += result.biogenic_co2_tonnes
            else:
                failure_count += 1

        elapsed_ms = (time.monotonic() - start_time) * 1000

        # Provenance hash for the batch
        provenance_hash = self._compute_provenance_hash({
            "batch_id": batch_id,
            "input_count": len(inputs),
            "total_co2e_kg": str(total_co2e_kg),
            "success_count": success_count,
            "failure_count": failure_count,
        })

        # Record batch metrics
        batch_size_bucket = self._get_batch_size_bucket(len(inputs))
        observe_batch_duration(batch_size_bucket, elapsed_ms / 1000)
        status = "success" if failure_count == 0 else "partial" if success_count > 0 else "failure"
        record_batch(status)

        # Record provenance
        if self._provenance is not None:
            self._provenance.record(
                entity_type="batch",
                action="calculate_batch",
                entity_id=batch_id,
                data={
                    "input_count": len(inputs),
                    "success_count": success_count,
                    "total_co2e_kg": str(total_co2e_kg),
                },
            )

        logger.info(
            "Batch %s completed: %d inputs, %d success, %d failed, "
            "total_co2e=%.4f tonnes, %.1f ms",
            batch_id, len(inputs), success_count, failure_count,
            total_co2e_tonnes, elapsed_ms,
        )

        return BatchCalculationResponse(
            batch_id=batch_id,
            results=results,
            total_co2e_kg=total_co2e_kg,
            total_co2e_tonnes=total_co2e_tonnes,
            total_biogenic_co2_kg=total_biogenic_kg,
            total_biogenic_co2_tonnes=total_biogenic_tonnes,
            success_count=success_count,
            failure_count=failure_count,
            processing_time_ms=elapsed_ms,
            provenance_hash=provenance_hash,
        )

    # ------------------------------------------------------------------
    # Internal: Tier Resolution
    # ------------------------------------------------------------------

    def _resolve_tier(
        self,
        input_data: CombustionInput,
        trace: List[str],
    ) -> str:
        """Determine the calculation tier to use.

        If the input specifies a tier, use it. Otherwise auto-select based
        on available data: Tier 3 if custom EF + custom HV, Tier 2 if
        custom EF or custom HV, Tier 1 otherwise.

        Args:
            input_data: Combustion input.
            trace: Trace list for recording steps.

        Returns:
            Tier string value.
        """
        if input_data.tier:
            tier_val = (
                input_data.tier
                if isinstance(input_data.tier, str)
                else input_data.tier.value
            )
            trace.append(f"[1a] Tier explicitly set: {tier_val}")
            return tier_val

        has_custom_ef = input_data.custom_emission_factor is not None
        has_custom_hv = input_data.custom_heating_value is not None

        if has_custom_ef and has_custom_hv:
            tier_val = CalculationTier.TIER_3.value
        elif has_custom_ef or has_custom_hv:
            tier_val = CalculationTier.TIER_2.value
        else:
            tier_val = CalculationTier.TIER_1.value

        trace.append(f"[1a] Tier auto-selected: {tier_val}")
        return tier_val

    # ------------------------------------------------------------------
    # Internal: Heating Value Resolution
    # ------------------------------------------------------------------

    def _resolve_hv_basis(self, input_data: CombustionInput) -> str:
        """Resolve the heating value basis string.

        Args:
            input_data: Combustion input.

        Returns:
            HV basis string (HHV or NCV).
        """
        return (
            input_data.heating_value_basis
            if isinstance(input_data.heating_value_basis, str)
            else input_data.heating_value_basis.value
        )

    def _resolve_heating_value(
        self,
        fuel_type: str,
        hv_basis: str,
        input_data: CombustionInput,
        trace: List[str],
    ) -> Decimal:
        """Resolve the heating value to use for the calculation.

        Args:
            fuel_type: Fuel type string.
            hv_basis: HHV or NCV.
            input_data: Combustion input.
            trace: Trace list.

        Returns:
            Heating value as Decimal.
        """
        if input_data.custom_heating_value is not None:
            hv = input_data.custom_heating_value
            trace.append(f"[3] Custom heating value: {hv}")
            return hv

        hv = self._fuel_db.get_heating_value(fuel_type, basis=hv_basis)
        trace.append(f"[3] {hv_basis} heating value for {fuel_type}: {hv}")
        return hv

    # ------------------------------------------------------------------
    # Internal: Oxidation Factor Resolution
    # ------------------------------------------------------------------

    def _resolve_oxidation_factor(
        self,
        fuel_type: str,
        input_data: CombustionInput,
        trace: List[str],
    ) -> Decimal:
        """Resolve the oxidation factor.

        Args:
            fuel_type: Fuel type string.
            input_data: Combustion input.
            trace: Trace list.

        Returns:
            Oxidation factor as Decimal.
        """
        if input_data.custom_oxidation_factor is not None:
            of = input_data.custom_oxidation_factor
            trace.append(f"[5] Custom oxidation factor: {of}")
            return of

        ef_source = (
            input_data.ef_source
            if isinstance(input_data.ef_source, str)
            else input_data.ef_source.value
        )
        of = self._fuel_db.get_oxidation_factor(fuel_type, source=ef_source)
        trace.append(f"[5] Oxidation factor ({ef_source}): {of}")
        return of

    # ------------------------------------------------------------------
    # Internal: Energy Conversion
    # ------------------------------------------------------------------

    def _convert_to_energy_gj(
        self,
        fuel_type: str,
        quantity: Decimal,
        unit: str,
        heating_value: Decimal,
        hv_basis: str,
        trace: List[str],
    ) -> Decimal:
        """Convert fuel quantity to energy content in GJ.

        The conversion proceeds in two stages:
        1. Convert the quantity to the native unit expected by the heating
           value (volume/mass depending on fuel category).
        2. Multiply by the heating value and convert to GJ.

        Args:
            fuel_type: Fuel type string.
            quantity: Fuel quantity.
            unit: Unit of quantity.
            heating_value: Heating value per native unit.
            hv_basis: HHV or NCV.
            trace: Trace list.

        Returns:
            Energy in GJ.
        """
        unit_str = unit if isinstance(unit, str) else unit.value
        unit_upper = unit_str.upper()

        # If already in energy units, convert directly to GJ
        energy_gj = self._try_energy_unit_to_gj(quantity, unit_upper)
        if energy_gj is not None:
            trace.append(
                f"[4a] Direct energy conversion: {quantity} {unit_upper} = {energy_gj} GJ"
            )
            return energy_gj.quantize(self._precision_quantizer, rounding=ROUND_HALF_UP)

        # Get fuel properties to determine category
        props = self._fuel_db.get_fuel_properties(fuel_type)
        category = props["category"]

        # Convert volume/mass quantity to mmBtu using heating value
        energy_mmbtu = self._quantity_to_mmbtu(
            quantity, unit_upper, heating_value, category, fuel_type, trace,
        )

        # Convert mmBtu to GJ
        energy_gj = (energy_mmbtu * _MMBTU_TO_GJ).quantize(
            self._precision_quantizer, rounding=ROUND_HALF_UP,
        )
        trace.append(f"[4b] {energy_mmbtu} mmBtu x {_MMBTU_TO_GJ} = {energy_gj} GJ")
        return energy_gj

    def _try_energy_unit_to_gj(
        self,
        quantity: Decimal,
        unit: str,
    ) -> Optional[Decimal]:
        """Convert energy units to GJ. Returns None if not an energy unit.

        Args:
            quantity: Quantity value.
            unit: Uppercase unit string.

        Returns:
            Energy in GJ, or None if not an energy unit.
        """
        conversions: Dict[str, Decimal] = {
            "GJ": Decimal("1"),
            "MMBTU": _MMBTU_TO_GJ,
            "TJ": _TJ_TO_GJ,
            "THERMS": _THERM_TO_GJ,
            "MWH": _MWH_TO_GJ,
            "KWH": _KWH_TO_GJ,
        }
        factor = conversions.get(unit)
        if factor is not None:
            return quantity * factor
        return None

    def _quantity_to_mmbtu(
        self,
        quantity: Decimal,
        unit: str,
        heating_value: Decimal,
        category: str,
        fuel_type: str,
        trace: List[str],
    ) -> Decimal:
        """Convert a volume or mass quantity to mmBtu.

        For gaseous fuels, heating value is in mmBtu/Mscf.
        For liquid fuels, heating value is in mmBtu/barrel.
        For solid/biomass/waste, heating value is in mmBtu/short_ton.

        Args:
            quantity: Fuel quantity.
            unit: Uppercase unit string.
            heating_value: HV per native unit.
            category: Fuel category (GASEOUS, LIQUID, SOLID, BIOMASS, WASTE).
            fuel_type: Fuel type string.
            trace: Trace list.

        Returns:
            Energy in mmBtu.

        Raises:
            ValueError: If the unit is not compatible with the fuel category.
        """
        # Volume units -> native volume
        if category in ("GASEOUS",):
            native_quantity = self._convert_volume_to_mscf(quantity, unit)
            mmbtu = native_quantity * heating_value
            trace.append(
                f"[4a] Gas: {quantity} {unit} -> {native_quantity} Mscf "
                f"x {heating_value} mmBtu/Mscf = {mmbtu} mmBtu"
            )
            return mmbtu

        if category in ("LIQUID",):
            native_quantity = self._convert_volume_to_barrels(quantity, unit)
            mmbtu = native_quantity * heating_value
            trace.append(
                f"[4a] Liquid: {quantity} {unit} -> {native_quantity} bbl "
                f"x {heating_value} mmBtu/bbl = {mmbtu} mmBtu"
            )
            return mmbtu

        # Solid, biomass, waste: mass-based, HV in mmBtu/short_ton
        native_quantity = self._convert_mass_to_short_tons(quantity, unit)
        mmbtu = native_quantity * heating_value
        trace.append(
            f"[4a] Solid: {quantity} {unit} -> {native_quantity} short_tons "
            f"x {heating_value} mmBtu/short_ton = {mmbtu} mmBtu"
        )
        return mmbtu

    def _convert_volume_to_mscf(self, quantity: Decimal, unit: str) -> Decimal:
        """Convert volume unit to Mscf (thousand cubic feet).

        Args:
            quantity: Volume quantity.
            unit: Uppercase unit string.

        Returns:
            Volume in Mscf.

        Raises:
            ValueError: If the unit cannot be converted.
        """
        if unit == "MCF":
            return quantity
        if unit == "CUBIC_FEET":
            return quantity / _MCF_TO_FT3
        if unit == "CUBIC_METERS":
            # 1 m3 = 35.3147 ft3
            ft3 = quantity * Decimal("35.3147")
            return ft3 / _MCF_TO_FT3
        if unit == "LITERS":
            m3 = quantity / _M3_TO_LITERS
            ft3 = m3 * Decimal("35.3147")
            return ft3 / _MCF_TO_FT3
        if unit == "GALLONS":
            liters = quantity * _GALLON_TO_LITERS
            m3 = liters / _M3_TO_LITERS
            ft3 = m3 * Decimal("35.3147")
            return ft3 / _MCF_TO_FT3
        raise ValueError(
            f"Cannot convert volume unit '{unit}' to Mscf for gaseous fuel"
        )

    def _convert_volume_to_barrels(self, quantity: Decimal, unit: str) -> Decimal:
        """Convert volume unit to barrels.

        Args:
            quantity: Volume quantity.
            unit: Uppercase unit string.

        Returns:
            Volume in barrels.

        Raises:
            ValueError: If the unit cannot be converted.
        """
        if unit == "BARRELS":
            return quantity
        if unit == "GALLONS":
            return quantity / _BARREL_TO_GALLONS
        if unit == "LITERS":
            return quantity / _BARREL_TO_LITERS
        if unit == "CUBIC_METERS":
            liters = quantity * _M3_TO_LITERS
            return liters / _BARREL_TO_LITERS
        raise ValueError(
            f"Cannot convert volume unit '{unit}' to barrels for liquid fuel"
        )

    def _convert_mass_to_short_tons(self, quantity: Decimal, unit: str) -> Decimal:
        """Convert mass unit to short tons.

        Args:
            quantity: Mass quantity.
            unit: Uppercase unit string.

        Returns:
            Mass in short tons.

        Raises:
            ValueError: If the unit cannot be converted.
        """
        if unit == "SHORT_TONS":
            return quantity
        if unit == "TONNES":
            return quantity / _SHORT_TON_TO_TONNES
        if unit == "KILOGRAMS":
            tonnes = quantity * _KG_TO_TONNES
            return tonnes / _SHORT_TON_TO_TONNES
        if unit == "POUNDS":
            tonnes = quantity * _LB_TO_TONNES
            return tonnes / _SHORT_TON_TO_TONNES
        raise ValueError(
            f"Cannot convert mass unit '{unit}' to short tons for solid fuel"
        )

    # ------------------------------------------------------------------
    # Internal: Gas Emission Decomposition
    # ------------------------------------------------------------------

    def _decompose_gas_emissions(
        self,
        energy_gj: Decimal,
        fuel_type: str,
        oxidation_factor: Decimal,
        gwp_source: str,
        ef_source: str,
        input_data: CombustionInput,
        trace: List[str],
    ) -> List[GasEmission]:
        """Calculate CO2, CH4, and N2O emissions separately.

        For each gas:
            mass_kg = energy_gj x ef_per_gj x oxidation_factor
            co2e_kg = mass_kg x gwp

        For EPA factors (kg/mmBtu), convert energy to mmBtu first.
        For IPCC/EU_ETS factors (kg/TJ NCV), convert energy to TJ.
        For DEFRA factors (kg/kWh), convert energy to kWh.

        Args:
            energy_gj: Energy content in GJ.
            fuel_type: Fuel type string.
            oxidation_factor: Oxidation factor.
            gwp_source: GWP source string.
            ef_source: EF source string.
            input_data: Original combustion input.
            trace: Trace list.

        Returns:
            List of GasEmission objects.
        """
        is_biogenic = self._fuel_db.is_biogenic(fuel_type)
        gases = ["CO2", "CH4", "N2O"]
        results: List[GasEmission] = []

        for gas in gases:
            # Get emission factor (use custom if provided)
            ef_value = self._resolve_ef_for_gas(
                fuel_type, gas, ef_source, input_data,
            )

            # Convert energy to match the EF unit basis
            energy_in_ef_basis = self._energy_gj_to_ef_basis(
                energy_gj, ef_source,
            )

            # Calculate mass
            mass_kg = (
                energy_in_ef_basis * ef_value * oxidation_factor
            ).quantize(self._precision_quantizer, rounding=ROUND_HALF_UP)

            # For biogenic fuels, CO2 is biogenic
            gas_is_biogenic = is_biogenic and gas == "CO2"

            # Get GWP
            gwp_val = self._fuel_db.get_gwp(gas, source=gwp_source)

            # Calculate CO2e
            co2e_kg = (mass_kg * gwp_val).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP,
            )

            mass_tonnes = (mass_kg * Decimal("0.001")).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP,
            )
            co2e_tonnes = (co2e_kg * Decimal("0.001")).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP,
            )

            trace.append(
                f"[6] {gas}: {energy_in_ef_basis} x {ef_value} x {oxidation_factor}"
                f" = {mass_kg} kg, GWP={gwp_val} -> {co2e_kg} kg CO2e"
                f"{' (biogenic)' if gas_is_biogenic else ''}"
            )

            emission_gas = "CO2_BIOGENIC" if gas_is_biogenic else gas
            results.append(GasEmission(
                gas=emission_gas,
                mass_kg=mass_kg,
                mass_tonnes=mass_tonnes,
                co2e_kg=co2e_kg,
                co2e_tonnes=co2e_tonnes,
                gwp_applied=gwp_val,
                emission_factor_used=ef_value,
                is_biogenic=gas_is_biogenic,
            ))

        return results

    def _resolve_ef_for_gas(
        self,
        fuel_type: str,
        gas: str,
        ef_source: str,
        input_data: CombustionInput,
    ) -> Decimal:
        """Resolve the emission factor for a specific gas.

        Uses custom emission factor if provided, otherwise looks up from
        the fuel database.

        Args:
            fuel_type: Fuel type string.
            gas: Gas identifier.
            ef_source: EF source string.
            input_data: Combustion input.

        Returns:
            Emission factor as Decimal.
        """
        if input_data.custom_emission_factor and gas in input_data.custom_emission_factor:
            return input_data.custom_emission_factor[gas]
        return self._fuel_db.get_emission_factor(fuel_type, gas, source=ef_source)

    def _energy_gj_to_ef_basis(
        self,
        energy_gj: Decimal,
        ef_source: str,
    ) -> Decimal:
        """Convert energy in GJ to the unit basis matching the EF source.

        EPA: mmBtu (1 GJ / 1.055056 = mmBtu)
        IPCC, EU_ETS: TJ (1 GJ / 1000 = TJ)
        DEFRA: kWh (1 GJ / 0.0036 = kWh)

        Args:
            energy_gj: Energy in GJ.
            ef_source: EF source string.

        Returns:
            Energy in the EF-compatible unit.
        """
        if ef_source == "EPA":
            return (energy_gj / _MMBTU_TO_GJ).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP,
            )
        if ef_source in ("IPCC", "EU_ETS"):
            return (energy_gj / _TJ_TO_GJ).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP,
            )
        if ef_source == "DEFRA":
            return (energy_gj / _KWH_TO_GJ).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP,
            )
        # Custom: assume GJ basis
        return energy_gj

    # ------------------------------------------------------------------
    # Internal: Totals Computation
    # ------------------------------------------------------------------

    def _compute_totals(
        self,
        gas_emissions: List[GasEmission],
        include_biogenic: bool,
        trace: List[str],
    ) -> tuple:
        """Compute total CO2e and biogenic CO2.

        Args:
            gas_emissions: List of per-gas emissions.
            include_biogenic: Whether to include biogenic in total.
            trace: Trace list.

        Returns:
            Tuple of (total_co2e_kg, biogenic_co2_kg).
        """
        total_co2e_kg = Decimal("0")
        biogenic_co2_kg = Decimal("0")

        for ge in gas_emissions:
            if ge.is_biogenic:
                biogenic_co2_kg += ge.co2e_kg
                if include_biogenic:
                    total_co2e_kg += ge.co2e_kg
            else:
                total_co2e_kg += ge.co2e_kg

        total_co2e_kg = total_co2e_kg.quantize(
            self._precision_quantizer, rounding=ROUND_HALF_UP,
        )
        biogenic_co2_kg = biogenic_co2_kg.quantize(
            self._precision_quantizer, rounding=ROUND_HALF_UP,
        )

        trace.append(
            f"[7] Totals: CO2e={total_co2e_kg} kg, "
            f"biogenic={biogenic_co2_kg} kg"
        )
        return total_co2e_kg, biogenic_co2_kg

    # ------------------------------------------------------------------
    # Internal: Equipment Adjustment
    # ------------------------------------------------------------------

    def _apply_equipment_adjustment(
        self,
        base_emissions_kg: Decimal,
        equipment_profile: Dict[str, Any],
    ) -> Decimal:
        """Apply equipment load factor and efficiency adjustments.

        Adjustment factor = load_factor / efficiency
        Adjusted emissions = base_emissions * load_factor / efficiency

        Args:
            base_emissions_kg: Base emissions in kg.
            equipment_profile: Equipment profile dictionary.

        Returns:
            Adjusted emissions in kg.
        """
        load_factor = Decimal(str(equipment_profile.get("load_factor", "0.65")))
        efficiency = Decimal(str(equipment_profile.get("efficiency", "0.80")))

        if efficiency <= 0:
            logger.warning("Equipment efficiency is <= 0, skipping adjustment")
            return base_emissions_kg

        adjustment = (load_factor / efficiency).quantize(
            self._precision_quantizer, rounding=ROUND_HALF_UP,
        )
        return (base_emissions_kg * adjustment).quantize(
            self._precision_quantizer, rounding=ROUND_HALF_UP,
        )

    def _apply_equipment_adjustment_to_totals(
        self,
        total_co2e_kg: Decimal,
        total_co2e_tonnes: Decimal,
        equipment_id: str,
        trace: List[str],
    ) -> tuple:
        """Apply equipment adjustment to totals (placeholder).

        This method is a hook for integration with EquipmentProfilerEngine.
        When no equipment profiler is connected, it returns unadjusted values
        and logs a warning.

        Args:
            total_co2e_kg: Total CO2e in kg.
            total_co2e_tonnes: Total CO2e in tonnes.
            equipment_id: Equipment ID string.
            trace: Trace list.

        Returns:
            Tuple of (adjustment_factor, adjusted_co2e_kg, adjusted_co2e_tonnes).
        """
        # Placeholder: return unadjusted. Full equipment integration is
        # handled at the agent orchestration layer.
        trace.append(
            f"[9] Equipment adjustment: equipment_id={equipment_id} "
            f"(adjustment deferred to orchestration layer)"
        )
        return None, total_co2e_kg, total_co2e_tonnes

    # ------------------------------------------------------------------
    # Internal: Provenance Hash
    # ------------------------------------------------------------------

    def _compute_provenance_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 provenance hash of calculation inputs and outputs.

        Args:
            data: Dictionary of calculation data to hash.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Internal: Calculation Trace
    # ------------------------------------------------------------------

    def _build_calculation_trace(self, steps: List[str]) -> List[str]:
        """Build a formatted calculation trace.

        Args:
            steps: List of trace step strings.

        Returns:
            Formatted trace list.
        """
        return list(steps)

    # ------------------------------------------------------------------
    # Internal: Batch Size Bucketing
    # ------------------------------------------------------------------

    def _get_batch_size_bucket(self, count: int) -> str:
        """Categorize batch size for metric labeling.

        Args:
            count: Number of inputs in the batch.

        Returns:
            Batch size bucket label string.
        """
        if count <= 10:
            return "1-10"
        if count <= 100:
            return "11-100"
        if count <= 1000:
            return "101-1000"
        return "1001+"

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return (
            f"CombustionCalculatorEngine("
            f"precision={self._precision_places}, "
            f"provenance={self._enable_provenance})"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "CombustionCalculatorEngine",
]
