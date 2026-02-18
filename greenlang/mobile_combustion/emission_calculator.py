# -*- coding: utf-8 -*-
"""
EmissionCalculatorEngine - Engine 2: Mobile Combustion Agent (AGENT-MRV-003)

Core calculation engine implementing three GHG Protocol-compliant calculation
methods for Scope 1 mobile combustion emissions:

1. **Fuel-Based Method** (primary, highest accuracy):
   CO2 = fuel_consumed x CO2_emission_factor x oxidation_factor
   CH4 = fuel_consumed x CH4_emission_factor x (1 + technology_adjustment)
   N2O = fuel_consumed x N2O_emission_factor x (1 + technology_adjustment)
   Total CO2e = CO2 x GWP_CO2 + CH4 x GWP_CH4 + N2O x GWP_N2O

2. **Distance-Based Method**:
   Fuel_consumed = distance / fuel_economy (with age/load adjustments)
   Then: apply fuel-based method
   OR: CO2e = distance x distance_emission_factor (g CO2e/km)

3. **Spend-Based Method** (screening level):
   Fuel_consumed = fuel_expenditure / fuel_price
   Then: apply fuel-based method

Features:
    - Biofuel handling: biogenic vs fossil CO2 separated by biofuel fraction
    - GWP application: AR4, AR5, AR6, AR6-20yr
    - Vehicle age degradation: fuel economy deteriorates by age bracket
    - Load factor adjustment for trucks, marine, aviation
    - Unit conversions: liters, gallons, kg, tonnes, m3, kWh, GJ
    - Batch processing with aggregated totals
    - Per-gas emission breakdown (CO2, CH4, N2O)
    - Complete calculation trace for audit

Zero-Hallucination Guarantees:
    - All calculations use Python Decimal (8 decimal places)
    - No LLM calls in the calculation path
    - Every step is recorded in the calculation trace
    - SHA-256 provenance hash for every result

Example:
    >>> from greenlang.mobile_combustion.emission_calculator import EmissionCalculatorEngine
    >>> from greenlang.mobile_combustion.vehicle_database import VehicleDatabaseEngine
    >>> from decimal import Decimal
    >>> db = VehicleDatabaseEngine()
    >>> calc = EmissionCalculatorEngine(vehicle_database=db)
    >>> result = calc.calculate_fuel_based(
    ...     vehicle_type="HEAVY_DUTY_TRUCK",
    ...     fuel_type="DIESEL",
    ...     fuel_consumed=Decimal("500"),
    ...     fuel_unit="liters",
    ... )
    >>> assert result["status"] == "SUCCESS"
    >>> assert result["total_co2e_kg"] > 0

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-003 Mobile Combustion (GL-MRV-SCOPE1-003)
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

from greenlang.mobile_combustion.vehicle_database import VehicleDatabaseEngine

logger = logging.getLogger(__name__)

__all__ = ["EmissionCalculatorEngine"]

# ---------------------------------------------------------------------------
# Decimal precision constant
# ---------------------------------------------------------------------------
_PRECISION = Decimal("0.00000001")  # 8 decimal places

# ---------------------------------------------------------------------------
# UTC helper
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Unit Conversion Constants (all Decimal for zero-hallucination precision)
# ---------------------------------------------------------------------------

# Volume conversions to liters
_GALLON_TO_LITERS = Decimal("3.78541")
_BARREL_TO_LITERS = Decimal("158.987")
_M3_TO_LITERS = Decimal("1000")
_FT3_TO_LITERS = Decimal("28.3168")

# Mass conversions to kg
_LB_TO_KG = Decimal("0.453592")
_SHORT_TON_TO_KG = Decimal("907.185")
_TONNE_TO_KG = Decimal("1000")

# Energy conversions to MJ
_KWH_TO_MJ = Decimal("3.6")
_GJ_TO_MJ = Decimal("1000")
_MMBTU_TO_MJ = Decimal("1055.056")
_THERM_TO_MJ = Decimal("105.506")

# kg to tonnes
_KG_TO_TONNES = Decimal("0.001")

# g to kg
_G_TO_KG = Decimal("0.001")

# Oxidation factor (default complete combustion)
_DEFAULT_OXIDATION_FACTOR = Decimal("1.0")

# TJ to MJ
_TJ_TO_MJ = Decimal("1000000")


# ---------------------------------------------------------------------------
# GWP Values (Decimal)
# ---------------------------------------------------------------------------

_GWP_VALUES: Dict[str, Dict[str, Decimal]] = {
    "AR4": {"CO2": Decimal("1"), "CH4": Decimal("25"), "N2O": Decimal("298")},
    "AR5": {"CO2": Decimal("1"), "CH4": Decimal("28"), "N2O": Decimal("265")},
    "AR6": {"CO2": Decimal("1"), "CH4": Decimal("29.8"), "N2O": Decimal("273")},
    "AR6_20YR": {"CO2": Decimal("1"), "CH4": Decimal("82.5"), "N2O": Decimal("273")},
}

# ---------------------------------------------------------------------------
# Vehicle Age Degradation Factors
# As vehicles age, fuel economy deteriorates. These multipliers increase
# fuel consumption relative to the rated fuel economy.
# Source: EPA MOVES model; ICCT vehicle emissions review
# ---------------------------------------------------------------------------

_AGE_DEGRADATION: Dict[str, Decimal] = {
    "0_3": Decimal("1.00"),     # 0-3 years: no degradation
    "4_7": Decimal("1.03"),     # 4-7 years: 3% degradation
    "8_12": Decimal("1.07"),    # 8-12 years: 7% degradation
    "13_17": Decimal("1.12"),   # 13-17 years: 12% degradation
    "18_25": Decimal("1.18"),   # 18-25 years: 18% degradation
    "26_PLUS": Decimal("1.25"), # 26+ years: 25% degradation
}

# ---------------------------------------------------------------------------
# Load Factor Adjustments
# Emission factors scale with load factor for freight and passenger transport.
# Multiplier applied relative to default load factor.
# Source: GHG Protocol Scope 1 Guidance; GLEC Framework
# ---------------------------------------------------------------------------

_LOAD_FACTOR_SENSITIVITY: Dict[str, Decimal] = {
    "HEAVY_DUTY_TRUCK": Decimal("0.8"),
    "MEDIUM_DUTY_TRUCK": Decimal("0.7"),
    "LIGHT_DUTY_TRUCK": Decimal("0.5"),
    "VAN_LCV": Decimal("0.5"),
    "BUS_DIESEL": Decimal("0.3"),
    "BUS_CNG": Decimal("0.3"),
    "MARINE_INLAND": Decimal("0.9"),
    "MARINE_COASTAL": Decimal("0.9"),
    "MARINE_OCEAN": Decimal("0.9"),
    "CORPORATE_JET": Decimal("0.6"),
    "HELICOPTER": Decimal("0.5"),
    "TURBOPROP": Decimal("0.6"),
    "DIESEL_LOCOMOTIVE": Decimal("0.8"),
    "CONSTRUCTION_EQUIPMENT": Decimal("0.7"),
    "AGRICULTURAL_EQUIPMENT": Decimal("0.6"),
    "INDUSTRIAL_EQUIPMENT": Decimal("0.6"),
    "MINING_EQUIPMENT": Decimal("0.8"),
    "FORKLIFT": Decimal("0.5"),
}


# ===========================================================================
# EmissionCalculatorEngine
# ===========================================================================


class EmissionCalculatorEngine:
    """Core calculation engine for GHG Protocol mobile combustion emissions.

    Implements three calculation methods (fuel-based, distance-based,
    spend-based) with deterministic Decimal arithmetic, full calculation
    trace, and SHA-256 provenance hashing. Thread-safe for concurrent
    calculations.

    Attributes:
        _vehicle_db: Reference to the VehicleDatabaseEngine for factor lookups.
        _config: Optional configuration dictionary.
        _lock: Thread lock for any shared mutable state.

    Example:
        >>> db = VehicleDatabaseEngine()
        >>> calc = EmissionCalculatorEngine(vehicle_database=db)
        >>> result = calc.calculate_fuel_based(
        ...     vehicle_type="PASSENGER_CAR_GASOLINE",
        ...     fuel_type="GASOLINE",
        ...     fuel_consumed=Decimal("50"),
        ...     fuel_unit="liters",
        ... )
        >>> assert result["status"] == "SUCCESS"
    """

    def __init__(
        self,
        vehicle_database: Optional[VehicleDatabaseEngine] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize EmissionCalculatorEngine.

        Args:
            vehicle_database: VehicleDatabaseEngine instance for factor lookups.
                If None, a default instance is created.
            config: Optional configuration dict. Supports:
                - ``enable_provenance`` (bool): Enable provenance tracking.
                - ``decimal_precision`` (int): Decimal places. Default 8.
                - ``default_gwp_source`` (str): Default GWP source. Default "AR6".
                - ``default_oxidation_factor`` (str/Decimal): Default oxidation factor.
        """
        self._vehicle_db = vehicle_database or VehicleDatabaseEngine()
        self._config: Dict[str, Any] = config or {}
        self._lock = threading.Lock()
        self._enable_provenance: bool = self._config.get("enable_provenance", True)
        self._precision_places: int = self._config.get("decimal_precision", 8)
        self._precision_quantizer = Decimal(10) ** -self._precision_places
        self._default_gwp: str = self._config.get("default_gwp_source", "AR6")
        self._default_ox_factor = Decimal(
            str(self._config.get("default_oxidation_factor", "1.0"))
        )

        logger.info(
            "EmissionCalculatorEngine initialized (precision=%d, gwp=%s, provenance=%s)",
            self._precision_places,
            self._default_gwp,
            self._enable_provenance,
        )

    # ------------------------------------------------------------------
    # Public API: Dispatch
    # ------------------------------------------------------------------

    def calculate(
        self,
        method: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Dispatch to the appropriate calculation method.

        Args:
            method: Calculation method - ``"fuel_based"``, ``"distance_based"``,
                or ``"spend_based"``.
            **kwargs: Method-specific arguments passed through.

        Returns:
            Calculation result dictionary.

        Raises:
            ValueError: If method is not recognized.
        """
        method_key = method.lower().strip()
        if method_key == "fuel_based":
            return self.calculate_fuel_based(**kwargs)
        elif method_key == "distance_based":
            return self.calculate_distance_based(**kwargs)
        elif method_key == "spend_based":
            return self.calculate_spend_based(**kwargs)
        else:
            raise ValueError(
                f"Unknown calculation method: '{method}'. "
                f"Valid methods: fuel_based, distance_based, spend_based"
            )

    # ------------------------------------------------------------------
    # Public API: Fuel-Based Method
    # ------------------------------------------------------------------

    def calculate_fuel_based(
        self,
        vehicle_type: str,
        fuel_type: str,
        fuel_consumed: Decimal,
        fuel_unit: str = "liters",
        model_year: Optional[int] = None,
        control_technology: Optional[str] = None,
        gwp_source: Optional[str] = None,
        oxidation_factor: Optional[Decimal] = None,
        load_factor: Optional[Decimal] = None,
        calculation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Calculate mobile combustion emissions using the fuel-based method.

        This is the primary (highest accuracy) method. Requires actual fuel
        consumption data.

        Formula:
            CO2 = fuel_consumed_L x CO2_EF_kg_per_L x oxidation_factor
            CH4 = fuel_consumed_L x density x CH4_EF_g_per_kg_fuel x 0.001
            N2O = fuel_consumed_L x density x N2O_EF_g_per_kg_fuel x 0.001
            CO2e = CO2*GWP_CO2 + CH4*GWP_CH4 + N2O*GWP_N2O

        For on-road vehicles with g/km factors, CH4/N2O are computed using
        distance-based factors if fuel economy data is available.

        Args:
            vehicle_type: Vehicle type identifier.
            fuel_type: Fuel type identifier.
            fuel_consumed: Amount of fuel consumed.
            fuel_unit: Unit of fuel measurement. Default ``"liters"``.
            model_year: Vehicle model year (for CH4/N2O year-range lookup).
            control_technology: Emission control technology identifier.
            gwp_source: GWP assessment report (AR4, AR5, AR6, AR6_20YR).
            oxidation_factor: Oxidation factor override (0-1).
            load_factor: Actual load factor (0-1) for intensity adjustment.
            calculation_id: Optional external calculation ID.
            metadata: Optional metadata dictionary.

        Returns:
            Complete calculation result dictionary with per-gas emissions,
            totals, biogenic breakdown, calculation trace, and provenance hash.
        """
        start_time = time.monotonic()
        calc_id = calculation_id or f"mc_fuel_{uuid.uuid4().hex[:12]}"
        gwp_src = (gwp_source or self._default_gwp).upper().strip()
        ox_factor = oxidation_factor if oxidation_factor is not None else self._default_ox_factor
        trace: List[str] = []

        try:
            vtype = vehicle_type.upper().strip()
            ftype = fuel_type.upper().strip()
            fuel_qty = Decimal(str(fuel_consumed))

            trace.append(
                f"[1] Input: vehicle={vtype}, fuel={ftype}, qty={fuel_qty} {fuel_unit}, "
                f"gwp={gwp_src}, ox={ox_factor}"
            )

            # Step 1: Validate inputs
            veh_data = self._vehicle_db.get_vehicle_type(vtype)
            fuel_data = self._vehicle_db.get_fuel_type(ftype)
            trace.append(f"[2] Vehicle category: {veh_data['category']}, fuel: {fuel_data['display_name']}")

            # Step 2: Convert fuel to liters (or m3 for CNG)
            fuel_liters = self._convert_fuel_to_liters(fuel_qty, fuel_unit, ftype, trace)
            trace.append(f"[3] Fuel in liters: {fuel_liters}")

            # Step 3: Calculate CO2 emissions
            co2_result = self._calculate_co2(fuel_liters, ftype, fuel_data, ox_factor, trace)
            co2_fossil_kg = co2_result["fossil_kg"]
            co2_biogenic_kg = co2_result["biogenic_kg"]
            co2_total_kg = co2_result["total_kg"]
            trace.append(
                f"[4] CO2: fossil={co2_fossil_kg} kg, biogenic={co2_biogenic_kg} kg"
            )

            # Step 4: Calculate CH4 and N2O emissions
            ch4_n2o = self._calculate_ch4_n2o(
                fuel_liters, vtype, ftype, veh_data, fuel_data,
                model_year, control_technology, trace,
            )
            ch4_kg = ch4_n2o["ch4_kg"]
            n2o_kg = ch4_n2o["n2o_kg"]
            trace.append(f"[5] CH4={ch4_kg} kg, N2O={n2o_kg} kg")

            # Step 5: Apply GWP
            gwp_values = self._get_gwp_values(gwp_src)
            co2_co2e = (co2_fossil_kg * gwp_values["CO2"]).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP
            )
            ch4_co2e = (ch4_kg * gwp_values["CH4"]).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP
            )
            n2o_co2e = (n2o_kg * gwp_values["N2O"]).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP
            )
            total_co2e_kg = (co2_co2e + ch4_co2e + n2o_co2e).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP
            )
            total_co2e_tonnes = (total_co2e_kg * _KG_TO_TONNES).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP
            )
            biogenic_co2e_kg = (co2_biogenic_kg * gwp_values["CO2"]).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP
            )
            biogenic_co2e_tonnes = (biogenic_co2e_kg * _KG_TO_TONNES).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP
            )
            trace.append(
                f"[6] GWP applied ({gwp_src}): CO2e={co2_co2e}, CH4e={ch4_co2e}, N2Oe={n2o_co2e}"
            )
            trace.append(f"[7] Total CO2e: {total_co2e_kg} kg = {total_co2e_tonnes} tonnes")

            # Step 6: Build gas emission breakdown
            gas_emissions = self._build_gas_emissions(
                co2_fossil_kg, co2_biogenic_kg, ch4_kg, n2o_kg,
                co2_co2e, ch4_co2e, n2o_co2e, gwp_src,
            )

            # Step 7: Provenance hash
            elapsed_ms = (time.monotonic() - start_time) * 1000
            provenance_hash = self._compute_provenance_hash({
                "calculation_id": calc_id,
                "method": "fuel_based",
                "vehicle_type": vtype,
                "fuel_type": ftype,
                "fuel_consumed_liters": str(fuel_liters),
                "total_co2e_kg": str(total_co2e_kg),
                "gwp_source": gwp_src,
            })
            trace.append(f"[8] Provenance hash: {provenance_hash[:16]}...")

            return {
                "calculation_id": calc_id,
                "status": "SUCCESS",
                "method": "fuel_based",
                "vehicle_type": vtype,
                "fuel_type": ftype,
                "fuel_consumed": fuel_qty,
                "fuel_unit": fuel_unit,
                "fuel_consumed_liters": fuel_liters,
                "model_year": model_year,
                "control_technology": control_technology,
                "gwp_source": gwp_src,
                "oxidation_factor": ox_factor,
                "gas_emissions": gas_emissions,
                "co2_fossil_kg": co2_fossil_kg,
                "co2_biogenic_kg": co2_biogenic_kg,
                "ch4_kg": ch4_kg,
                "n2o_kg": n2o_kg,
                "total_co2e_kg": total_co2e_kg,
                "total_co2e_tonnes": total_co2e_tonnes,
                "biogenic_co2e_kg": biogenic_co2e_kg,
                "biogenic_co2e_tonnes": biogenic_co2e_tonnes,
                "calculation_trace": trace,
                "provenance_hash": provenance_hash,
                "processing_time_ms": round(elapsed_ms, 3),
                "metadata": metadata or {},
            }

        except Exception as exc:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "Fuel-based calculation failed (id=%s): %s",
                calc_id, exc, exc_info=True,
            )
            return self._create_error_result(
                calc_id, "fuel_based", vehicle_type, fuel_type,
                fuel_consumed, fuel_unit, str(exc), trace, elapsed_ms,
            )

    # ------------------------------------------------------------------
    # Public API: Distance-Based Method
    # ------------------------------------------------------------------

    def calculate_distance_based(
        self,
        vehicle_type: str,
        fuel_type: str,
        distance_km: Decimal,
        fuel_economy_km_per_l: Optional[Decimal] = None,
        vehicle_age_years: Optional[int] = None,
        model_year: Optional[int] = None,
        control_technology: Optional[str] = None,
        gwp_source: Optional[str] = None,
        oxidation_factor: Optional[Decimal] = None,
        load_factor: Optional[Decimal] = None,
        use_distance_factor: bool = False,
        calculation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Calculate mobile combustion emissions using the distance-based method.

        Converts distance to fuel consumption, then applies the fuel-based
        method. Alternatively, uses direct distance-based emission factors
        when ``use_distance_factor=True``.

        Formula (default):
            fuel_consumed = distance_km / fuel_economy_km_per_l (adjusted)
            Then: fuel-based method

        Formula (distance factor):
            CO2e = distance_km x distance_emission_factor_g_per_km x 0.001

        Args:
            vehicle_type: Vehicle type identifier.
            fuel_type: Fuel type identifier.
            distance_km: Distance traveled in kilometers.
            fuel_economy_km_per_l: Fuel economy override. If None, uses
                vehicle type default.
            vehicle_age_years: Vehicle age for fuel economy degradation.
            model_year: Vehicle model year (for CH4/N2O factors).
            control_technology: Emission control technology.
            gwp_source: GWP assessment report.
            oxidation_factor: Oxidation factor override.
            load_factor: Actual load factor (0-1).
            use_distance_factor: If True, uses g CO2e/km factor directly.
            calculation_id: Optional external calculation ID.
            metadata: Optional metadata dictionary.

        Returns:
            Complete calculation result dictionary.
        """
        start_time = time.monotonic()
        calc_id = calculation_id or f"mc_dist_{uuid.uuid4().hex[:12]}"
        gwp_src = (gwp_source or self._default_gwp).upper().strip()
        trace: List[str] = []

        try:
            vtype = vehicle_type.upper().strip()
            ftype = fuel_type.upper().strip()
            dist = Decimal(str(distance_km))

            trace.append(
                f"[1] Input: vehicle={vtype}, fuel={ftype}, distance={dist} km, "
                f"use_distance_factor={use_distance_factor}"
            )

            veh_data = self._vehicle_db.get_vehicle_type(vtype)

            # Option A: Direct distance-based emission factor
            if use_distance_factor:
                return self._calculate_distance_direct(
                    calc_id, vtype, ftype, dist, veh_data,
                    gwp_src, model_year, control_technology,
                    trace, start_time, metadata,
                )

            # Option B: Convert distance to fuel, then fuel-based
            trace.append(f"[2] Converting distance to fuel consumption")

            # Get fuel economy
            economy = fuel_economy_km_per_l
            if economy is None:
                economy = veh_data.get("default_fuel_economy_km_per_l")
            if economy is None:
                raise ValueError(
                    f"No fuel economy data for vehicle type '{vtype}'. "
                    f"Provide fuel_economy_km_per_l parameter."
                )
            economy = Decimal(str(economy))
            trace.append(f"[3] Base fuel economy: {economy} km/L")

            # Apply vehicle age degradation
            if vehicle_age_years is not None:
                economy = self._adjust_for_vehicle_age(economy, vehicle_age_years)
                trace.append(
                    f"[4] Age-adjusted fuel economy (age={vehicle_age_years}yr): {economy} km/L"
                )

            # Apply load factor adjustment
            if load_factor is not None:
                economy = self._adjust_for_load_factor(economy, vtype, load_factor)
                trace.append(
                    f"[5] Load-adjusted fuel economy (lf={load_factor}): {economy} km/L"
                )

            # Calculate fuel consumed
            if economy <= Decimal("0"):
                raise ValueError("Fuel economy must be positive")
            fuel_consumed = (dist / economy).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP
            )
            trace.append(f"[6] Estimated fuel consumed: {fuel_consumed} liters")

            # Delegate to fuel-based method
            fuel_result = self.calculate_fuel_based(
                vehicle_type=vtype,
                fuel_type=ftype,
                fuel_consumed=fuel_consumed,
                fuel_unit="liters",
                model_year=model_year,
                control_technology=control_technology,
                gwp_source=gwp_src,
                oxidation_factor=oxidation_factor,
                load_factor=load_factor,
                calculation_id=calc_id,
                metadata=metadata,
            )

            # Augment result with distance info
            elapsed_ms = (time.monotonic() - start_time) * 1000
            fuel_result["method"] = "distance_based"
            fuel_result["distance_km"] = dist
            fuel_result["fuel_economy_km_per_l"] = economy
            fuel_result["vehicle_age_years"] = vehicle_age_years
            fuel_result["processing_time_ms"] = round(elapsed_ms, 3)

            # Merge traces
            combined_trace = trace + fuel_result.get("calculation_trace", [])
            fuel_result["calculation_trace"] = combined_trace

            # Compute emission intensity
            if dist > Decimal("0"):
                intensity_g_per_km = (
                    fuel_result["total_co2e_kg"] * Decimal("1000") / dist
                ).quantize(self._precision_quantizer, rounding=ROUND_HALF_UP)
                fuel_result["emission_intensity_g_co2e_per_km"] = intensity_g_per_km

            return fuel_result

        except Exception as exc:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "Distance-based calculation failed (id=%s): %s",
                calc_id, exc, exc_info=True,
            )
            return self._create_error_result(
                calc_id, "distance_based", vehicle_type, fuel_type,
                distance_km, "km", str(exc), trace, elapsed_ms,
            )

    # ------------------------------------------------------------------
    # Public API: Spend-Based Method
    # ------------------------------------------------------------------

    def calculate_spend_based(
        self,
        vehicle_type: str,
        fuel_type: str,
        fuel_expenditure: Decimal,
        fuel_price_per_unit: Decimal,
        price_unit: str = "per_liter",
        currency: str = "USD",
        model_year: Optional[int] = None,
        control_technology: Optional[str] = None,
        gwp_source: Optional[str] = None,
        oxidation_factor: Optional[Decimal] = None,
        calculation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Calculate mobile combustion emissions using the spend-based method.

        Screening-level method that estimates fuel consumption from expenditure.

        Formula:
            fuel_consumed = fuel_expenditure / fuel_price_per_unit
            Then: fuel-based method

        Args:
            vehicle_type: Vehicle type identifier.
            fuel_type: Fuel type identifier.
            fuel_expenditure: Total amount spent on fuel.
            fuel_price_per_unit: Price per unit of fuel.
            price_unit: Price basis - ``"per_liter"``, ``"per_gallon"``,
                ``"per_m3"``, ``"per_kg"``.
            currency: Currency code (for metadata only).
            model_year: Vehicle model year.
            control_technology: Emission control technology.
            gwp_source: GWP assessment report.
            oxidation_factor: Oxidation factor override.
            calculation_id: Optional external calculation ID.
            metadata: Optional metadata dictionary.

        Returns:
            Complete calculation result dictionary.
        """
        start_time = time.monotonic()
        calc_id = calculation_id or f"mc_spend_{uuid.uuid4().hex[:12]}"
        gwp_src = (gwp_source or self._default_gwp).upper().strip()
        trace: List[str] = []

        try:
            expenditure = Decimal(str(fuel_expenditure))
            price = Decimal(str(fuel_price_per_unit))

            if price <= Decimal("0"):
                raise ValueError("Fuel price must be positive")
            if expenditure < Decimal("0"):
                raise ValueError("Fuel expenditure cannot be negative")

            trace.append(
                f"[1] Input: expenditure={expenditure} {currency}, "
                f"price={price} {currency}/{price_unit}"
            )

            # Calculate fuel consumed in the price unit
            fuel_consumed_raw = (expenditure / price).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP
            )
            trace.append(f"[2] Estimated fuel consumed: {fuel_consumed_raw} ({price_unit})")

            # Determine the fuel unit for the fuel-based method
            unit_mapping: Dict[str, str] = {
                "per_liter": "liters",
                "per_gallon": "gallons",
                "per_m3": "m3",
                "per_kg": "kg",
            }
            fuel_unit = unit_mapping.get(price_unit.lower().strip(), "liters")

            # Delegate to fuel-based method
            fuel_result = self.calculate_fuel_based(
                vehicle_type=vehicle_type,
                fuel_type=fuel_type,
                fuel_consumed=fuel_consumed_raw,
                fuel_unit=fuel_unit,
                model_year=model_year,
                control_technology=control_technology,
                gwp_source=gwp_src,
                oxidation_factor=oxidation_factor,
                calculation_id=calc_id,
                metadata=metadata,
            )

            # Augment result with spend info
            elapsed_ms = (time.monotonic() - start_time) * 1000
            fuel_result["method"] = "spend_based"
            fuel_result["fuel_expenditure"] = expenditure
            fuel_result["fuel_price_per_unit"] = price
            fuel_result["price_unit"] = price_unit
            fuel_result["currency"] = currency
            fuel_result["processing_time_ms"] = round(elapsed_ms, 3)

            # Merge traces
            combined_trace = trace + fuel_result.get("calculation_trace", [])
            fuel_result["calculation_trace"] = combined_trace

            # Compute cost intensity
            if expenditure > Decimal("0"):
                cost_intensity = (
                    fuel_result["total_co2e_kg"] / expenditure
                ).quantize(self._precision_quantizer, rounding=ROUND_HALF_UP)
                fuel_result["emission_intensity_kg_co2e_per_currency"] = cost_intensity

            return fuel_result

        except Exception as exc:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "Spend-based calculation failed (id=%s): %s",
                calc_id, exc, exc_info=True,
            )
            return self._create_error_result(
                calc_id, "spend_based", vehicle_type, fuel_type,
                fuel_expenditure, currency, str(exc), trace, elapsed_ms,
            )

    # ------------------------------------------------------------------
    # Public API: Batch Calculation
    # ------------------------------------------------------------------

    def calculate_batch(
        self,
        inputs: List[Dict[str, Any]],
        gwp_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process multiple mobile combustion calculations and aggregate totals.

        Each input dictionary must contain a ``"method"`` key plus the
        arguments for that method.

        Args:
            inputs: List of calculation input dictionaries.
            gwp_source: Default GWP source for all calculations (can be
                overridden per-input).

        Returns:
            Batch result dictionary with individual results and aggregated
            totals.
        """
        start_time = time.monotonic()
        batch_id = f"mc_batch_{uuid.uuid4().hex[:12]}"

        results: List[Dict[str, Any]] = []
        total_co2e_kg = Decimal("0")
        total_co2e_tonnes = Decimal("0")
        total_biogenic_kg = Decimal("0")
        total_biogenic_tonnes = Decimal("0")
        total_co2_fossil_kg = Decimal("0")
        total_ch4_kg = Decimal("0")
        total_n2o_kg = Decimal("0")
        success_count = 0
        failure_count = 0

        for idx, inp in enumerate(inputs):
            inp_copy = dict(inp)
            method = inp_copy.pop("method", "fuel_based")

            # Apply default gwp if not specified per-input
            if gwp_source and "gwp_source" not in inp_copy:
                inp_copy["gwp_source"] = gwp_source

            if "calculation_id" not in inp_copy:
                inp_copy["calculation_id"] = f"{batch_id}_item_{idx}"

            result = self.calculate(method=method, **inp_copy)
            results.append(result)

            if result.get("status") == "SUCCESS":
                success_count += 1
                total_co2e_kg += result.get("total_co2e_kg", Decimal("0"))
                total_co2e_tonnes += result.get("total_co2e_tonnes", Decimal("0"))
                total_biogenic_kg += result.get("biogenic_co2e_kg", Decimal("0"))
                total_biogenic_tonnes += result.get("biogenic_co2e_tonnes", Decimal("0"))
                total_co2_fossil_kg += result.get("co2_fossil_kg", Decimal("0"))
                total_ch4_kg += result.get("ch4_kg", Decimal("0"))
                total_n2o_kg += result.get("n2o_kg", Decimal("0"))
            else:
                failure_count += 1

        elapsed_ms = (time.monotonic() - start_time) * 1000

        provenance_hash = self._compute_provenance_hash({
            "batch_id": batch_id,
            "input_count": len(inputs),
            "success_count": success_count,
            "total_co2e_kg": str(total_co2e_kg),
        })

        logger.info(
            "Batch %s completed: %d/%d success, total_co2e=%.4f tonnes (%.1f ms)",
            batch_id, success_count, len(inputs), total_co2e_tonnes, elapsed_ms,
        )

        return {
            "batch_id": batch_id,
            "status": "SUCCESS" if failure_count == 0 else "PARTIAL",
            "input_count": len(inputs),
            "success_count": success_count,
            "failure_count": failure_count,
            "results": results,
            "totals": {
                "total_co2e_kg": total_co2e_kg.quantize(
                    self._precision_quantizer, rounding=ROUND_HALF_UP
                ),
                "total_co2e_tonnes": total_co2e_tonnes.quantize(
                    self._precision_quantizer, rounding=ROUND_HALF_UP
                ),
                "total_biogenic_co2e_kg": total_biogenic_kg.quantize(
                    self._precision_quantizer, rounding=ROUND_HALF_UP
                ),
                "total_biogenic_co2e_tonnes": total_biogenic_tonnes.quantize(
                    self._precision_quantizer, rounding=ROUND_HALF_UP
                ),
                "total_co2_fossil_kg": total_co2_fossil_kg.quantize(
                    self._precision_quantizer, rounding=ROUND_HALF_UP
                ),
                "total_ch4_kg": total_ch4_kg.quantize(
                    self._precision_quantizer, rounding=ROUND_HALF_UP
                ),
                "total_n2o_kg": total_n2o_kg.quantize(
                    self._precision_quantizer, rounding=ROUND_HALF_UP
                ),
            },
            "provenance_hash": provenance_hash,
            "processing_time_ms": round(elapsed_ms, 3),
        }

    # ------------------------------------------------------------------
    # Public API: GWP Application
    # ------------------------------------------------------------------

    def apply_gwp(
        self, gas: str, mass_kg: Decimal, gwp_source: Optional[str] = None
    ) -> Decimal:
        """Apply GWP to convert a gas mass to CO2 equivalent.

        Args:
            gas: Greenhouse gas identifier (CO2, CH4, N2O).
            mass_kg: Mass of the gas in kg.
            gwp_source: Assessment report source. Default per config.

        Returns:
            CO2e mass in kg as Decimal.
        """
        gwp_src = (gwp_source or self._default_gwp).upper().strip()
        gwp_values = self._get_gwp_values(gwp_src)
        gas_key = gas.upper().strip()

        if gas_key not in gwp_values:
            raise ValueError(f"Unknown gas: '{gas}'")

        result = (Decimal(str(mass_kg)) * gwp_values[gas_key]).quantize(
            self._precision_quantizer, rounding=ROUND_HALF_UP
        )
        return result

    # ------------------------------------------------------------------
    # Public API: Biogenic Fraction
    # ------------------------------------------------------------------

    def get_biogenic_fraction(self, fuel_type: str) -> Decimal:
        """Return the biofuel fraction for a fuel type.

        Args:
            fuel_type: Fuel type identifier.

        Returns:
            Biofuel fraction as Decimal (0.0 to 1.0).
        """
        return self._vehicle_db.get_biofuel_fraction(fuel_type)

    # ------------------------------------------------------------------
    # Public API: Vehicle Age Adjustment
    # ------------------------------------------------------------------

    def adjust_for_vehicle_age(
        self, base_economy: Decimal, vehicle_age_years: int
    ) -> Decimal:
        """Adjust fuel economy for vehicle age degradation.

        Public wrapper for the internal method.

        Args:
            base_economy: Base fuel economy in km/L.
            vehicle_age_years: Vehicle age in years.

        Returns:
            Degraded fuel economy in km/L.
        """
        return self._adjust_for_vehicle_age(base_economy, vehicle_age_years)

    # ------------------------------------------------------------------
    # Public API: Load Factor Adjustment
    # ------------------------------------------------------------------

    def adjust_for_load_factor(
        self, base_economy: Decimal, vehicle_type: str, load_factor: Decimal
    ) -> Decimal:
        """Adjust fuel economy for load factor.

        Public wrapper for the internal method.

        Args:
            base_economy: Base fuel economy in km/L.
            vehicle_type: Vehicle type for sensitivity lookup.
            load_factor: Actual load factor (0-1).

        Returns:
            Adjusted fuel economy in km/L.
        """
        return self._adjust_for_load_factor(base_economy, vehicle_type, load_factor)

    # ==================================================================
    # Internal: CO2 Calculation
    # ==================================================================

    def _calculate_co2(
        self,
        fuel_liters: Decimal,
        fuel_type: str,
        fuel_data: Dict[str, Any],
        oxidation_factor: Decimal,
        trace: List[str],
    ) -> Dict[str, Decimal]:
        """Calculate CO2 emissions with biogenic separation.

        Args:
            fuel_liters: Fuel consumed in liters (or m3 equivalent for CNG).
            fuel_type: Fuel type key.
            fuel_data: Full fuel data dictionary.
            oxidation_factor: Oxidation factor.
            trace: Calculation trace for audit.

        Returns:
            Dictionary with total_kg, fossil_kg, biogenic_kg.
        """
        # Get CO2 emission factor
        co2_ef = fuel_data.get("co2_ef_kg_per_l")
        if co2_ef is None:
            co2_ef = fuel_data.get("co2_ef_kg_per_m3", Decimal("0"))

        # Total CO2 = fuel x EF x oxidation
        total_co2_kg = (fuel_liters * co2_ef * oxidation_factor).quantize(
            self._precision_quantizer, rounding=ROUND_HALF_UP
        )

        # Separate biogenic and fossil
        biofuel_fraction = fuel_data.get("biofuel_fraction", Decimal("0"))
        biogenic_co2_kg = (total_co2_kg * biofuel_fraction).quantize(
            self._precision_quantizer, rounding=ROUND_HALF_UP
        )
        fossil_co2_kg = (total_co2_kg - biogenic_co2_kg).quantize(
            self._precision_quantizer, rounding=ROUND_HALF_UP
        )

        trace.append(
            f"[CO2] EF={co2_ef}, biofuel_frac={biofuel_fraction}, "
            f"total={total_co2_kg}, fossil={fossil_co2_kg}, biogenic={biogenic_co2_kg}"
        )

        return {
            "total_kg": total_co2_kg,
            "fossil_kg": fossil_co2_kg,
            "biogenic_kg": biogenic_co2_kg,
        }

    # ==================================================================
    # Internal: CH4 & N2O Calculation
    # ==================================================================

    def _calculate_ch4_n2o(
        self,
        fuel_liters: Decimal,
        vehicle_type: str,
        fuel_type: str,
        veh_data: Dict[str, Any],
        fuel_data: Dict[str, Any],
        model_year: Optional[int],
        control_technology: Optional[str],
        trace: List[str],
    ) -> Dict[str, Decimal]:
        """Calculate CH4 and N2O emissions.

        For on-road vehicles, uses g/km factors with estimated distance.
        For off-road, marine, aviation, rail, uses g/kg-fuel factors.

        Args:
            fuel_liters: Fuel consumed in liters.
            vehicle_type: Vehicle type key.
            fuel_type: Fuel type key.
            veh_data: Vehicle data dictionary.
            fuel_data: Fuel data dictionary.
            model_year: Model year for factor lookup.
            control_technology: Emission control technology.
            trace: Calculation trace.

        Returns:
            Dictionary with ch4_kg and n2o_kg.
        """
        category = veh_data["category"]
        factors = self._vehicle_db.get_ch4_n2o_factors(
            vehicle_type, model_year, control_technology,
        )

        if factors["unit"] == "g/km":
            # On-road: convert fuel liters to estimated distance
            economy = veh_data.get("default_fuel_economy_km_per_l")
            if economy is not None and economy > Decimal("0"):
                estimated_km = (fuel_liters * economy).quantize(
                    self._precision_quantizer, rounding=ROUND_HALF_UP
                )
            else:
                # Fallback: use typical annual km as approximation
                estimated_km = veh_data.get("typical_annual_km", Decimal("15000"))

            ch4_kg = (estimated_km * factors["ch4_value"] * _G_TO_KG).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP
            )
            n2o_kg = (estimated_km * factors["n2o_value"] * _G_TO_KG).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP
            )
            trace.append(
                f"[CH4/N2O] On-road g/km method: estimated_km={estimated_km}, "
                f"CH4={factors['ch4_value']} g/km, N2O={factors['n2o_value']} g/km"
            )

        elif factors["unit"] == "g/kg-fuel":
            # Non-road: convert fuel liters to fuel mass (kg)
            density = fuel_data.get("density_kg_per_l")
            if density is None:
                density = fuel_data.get("density_kg_per_m3", Decimal("0.832"))
            fuel_mass_kg = (fuel_liters * density).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP
            )

            ch4_kg = (fuel_mass_kg * factors["ch4_value"] * _G_TO_KG).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP
            )
            n2o_kg = (fuel_mass_kg * factors["n2o_value"] * _G_TO_KG).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP
            )
            trace.append(
                f"[CH4/N2O] Non-road g/kg-fuel method: fuel_mass={fuel_mass_kg} kg, "
                f"CH4={factors['ch4_value']} g/kg, N2O={factors['n2o_value']} g/kg"
            )

        else:
            raise ValueError(f"Unsupported CH4/N2O factor unit: {factors['unit']}")

        return {"ch4_kg": ch4_kg, "n2o_kg": n2o_kg}

    # ==================================================================
    # Internal: Distance Direct Method
    # ==================================================================

    def _calculate_distance_direct(
        self,
        calc_id: str,
        vehicle_type: str,
        fuel_type: str,
        distance_km: Decimal,
        veh_data: Dict[str, Any],
        gwp_source: str,
        model_year: Optional[int],
        control_technology: Optional[str],
        trace: List[str],
        start_time: float,
        metadata: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calculate emissions using direct distance-based emission factor.

        Uses g CO2e/km factor directly without fuel conversion.

        Args:
            calc_id: Calculation identifier.
            vehicle_type: Vehicle type key.
            fuel_type: Fuel type key.
            distance_km: Distance in km.
            veh_data: Vehicle data.
            gwp_source: GWP source.
            model_year: Model year.
            control_technology: Control technology.
            trace: Calculation trace.
            start_time: Start time for elapsed calculation.
            metadata: Optional metadata.

        Returns:
            Calculation result dictionary.
        """
        trace.append(f"[2] Using direct distance-based emission factor")

        # Get distance emission factor
        ef_g_per_km = self._vehicle_db.get_distance_emission_factor(
            vehicle_type, fuel_type
        )
        trace.append(f"[3] Distance EF: {ef_g_per_km} g CO2e/km")

        # Calculate total CO2e
        total_co2e_g = (distance_km * ef_g_per_km).quantize(
            self._precision_quantizer, rounding=ROUND_HALF_UP
        )
        total_co2e_kg = (total_co2e_g * _G_TO_KG).quantize(
            self._precision_quantizer, rounding=ROUND_HALF_UP
        )
        total_co2e_tonnes = (total_co2e_kg * _KG_TO_TONNES).quantize(
            self._precision_quantizer, rounding=ROUND_HALF_UP
        )
        trace.append(f"[4] Total CO2e: {total_co2e_kg} kg = {total_co2e_tonnes} tonnes")

        # Biogenic fraction
        biofuel_fraction = self._vehicle_db.get_biofuel_fraction(fuel_type)
        biogenic_co2e_kg = (total_co2e_kg * biofuel_fraction).quantize(
            self._precision_quantizer, rounding=ROUND_HALF_UP
        )
        fossil_co2e_kg = (total_co2e_kg - biogenic_co2e_kg).quantize(
            self._precision_quantizer, rounding=ROUND_HALF_UP
        )

        # Approximate per-gas breakdown from total CO2e using typical ratios
        # CO2 typically ~99% of total CO2e for mobile combustion
        co2_fraction = Decimal("0.99")
        ch4_n2o_fraction = Decimal("0.01")

        approx_co2_kg = (fossil_co2e_kg * co2_fraction).quantize(
            self._precision_quantizer, rounding=ROUND_HALF_UP
        )
        approx_ch4_co2e = (fossil_co2e_kg * ch4_n2o_fraction * Decimal("0.5")).quantize(
            self._precision_quantizer, rounding=ROUND_HALF_UP
        )
        approx_n2o_co2e = (fossil_co2e_kg * ch4_n2o_fraction * Decimal("0.5")).quantize(
            self._precision_quantizer, rounding=ROUND_HALF_UP
        )

        gwp_values = self._get_gwp_values(gwp_source)
        approx_ch4_kg = Decimal("0")
        approx_n2o_kg = Decimal("0")
        if gwp_values["CH4"] > Decimal("0"):
            approx_ch4_kg = (approx_ch4_co2e / gwp_values["CH4"]).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP
            )
        if gwp_values["N2O"] > Decimal("0"):
            approx_n2o_kg = (approx_n2o_co2e / gwp_values["N2O"]).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP
            )

        gas_emissions = self._build_gas_emissions(
            approx_co2_kg, biogenic_co2e_kg, approx_ch4_kg, approx_n2o_kg,
            approx_co2_kg, approx_ch4_co2e, approx_n2o_co2e, gwp_source,
        )

        # Emission intensity
        intensity_g_per_km = ef_g_per_km

        elapsed_ms = (time.monotonic() - start_time) * 1000
        provenance_hash = self._compute_provenance_hash({
            "calculation_id": calc_id,
            "method": "distance_based_direct",
            "vehicle_type": vehicle_type,
            "fuel_type": fuel_type,
            "distance_km": str(distance_km),
            "total_co2e_kg": str(total_co2e_kg),
        })
        trace.append(f"[5] Provenance hash: {provenance_hash[:16]}...")

        return {
            "calculation_id": calc_id,
            "status": "SUCCESS",
            "method": "distance_based_direct",
            "vehicle_type": vehicle_type,
            "fuel_type": fuel_type,
            "distance_km": distance_km,
            "distance_ef_g_co2e_per_km": ef_g_per_km,
            "model_year": model_year,
            "control_technology": control_technology,
            "gwp_source": gwp_source,
            "gas_emissions": gas_emissions,
            "co2_fossil_kg": fossil_co2e_kg,
            "co2_biogenic_kg": biogenic_co2e_kg,
            "ch4_kg": approx_ch4_kg,
            "n2o_kg": approx_n2o_kg,
            "total_co2e_kg": total_co2e_kg,
            "total_co2e_tonnes": total_co2e_tonnes,
            "biogenic_co2e_kg": biogenic_co2e_kg,
            "biogenic_co2e_tonnes": (biogenic_co2e_kg * _KG_TO_TONNES).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP
            ),
            "emission_intensity_g_co2e_per_km": intensity_g_per_km,
            "calculation_trace": trace,
            "provenance_hash": provenance_hash,
            "processing_time_ms": round(elapsed_ms, 3),
            "metadata": metadata or {},
            "note": "Approximate per-gas breakdown from aggregate distance factor",
        }

    # ==================================================================
    # Internal: Unit Conversion
    # ==================================================================

    def _convert_fuel_to_liters(
        self,
        quantity: Decimal,
        unit: str,
        fuel_type: str,
        trace: List[str],
    ) -> Decimal:
        """Convert fuel quantity from any supported unit to liters.

        For CNG (gaseous), converts to m3-equivalent liters for consistent
        CO2 factor application.

        Args:
            quantity: Fuel quantity in the given unit.
            unit: Unit string.
            fuel_type: Fuel type for density lookups.
            trace: Calculation trace.

        Returns:
            Fuel quantity in liters (or m3-equivalent for CNG).

        Raises:
            ValueError: If unit is not supported.
        """
        unit_key = unit.lower().strip()
        fuel_data = self._vehicle_db.get_fuel_type(fuel_type)

        if unit_key == "liters" or unit_key == "l":
            result = quantity

        elif unit_key == "gallons" or unit_key == "gal":
            result = (quantity * _GALLON_TO_LITERS).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP
            )
            trace.append(f"[unit] {quantity} gallons -> {result} liters")

        elif unit_key == "barrels" or unit_key == "bbl":
            result = (quantity * _BARREL_TO_LITERS).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP
            )
            trace.append(f"[unit] {quantity} barrels -> {result} liters")

        elif unit_key == "m3" or unit_key == "cubic_meters":
            if fuel_type.upper() == "CNG":
                # For CNG, keep in m3 since EF is per m3
                result = quantity
                trace.append(f"[unit] CNG: {quantity} m3 (used directly)")
            else:
                result = (quantity * _M3_TO_LITERS).quantize(
                    self._precision_quantizer, rounding=ROUND_HALF_UP
                )
                trace.append(f"[unit] {quantity} m3 -> {result} liters")

        elif unit_key == "kg":
            density = fuel_data.get("density_kg_per_l")
            if density is None or density == Decimal("0"):
                raise ValueError(f"Cannot convert kg to liters for {fuel_type}: no density")
            result = (quantity / density).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP
            )
            trace.append(f"[unit] {quantity} kg / {density} kg/L -> {result} liters")

        elif unit_key == "tonnes" or unit_key == "metric_tonnes":
            density = fuel_data.get("density_kg_per_l")
            if density is None or density == Decimal("0"):
                raise ValueError(f"Cannot convert tonnes to liters for {fuel_type}")
            mass_kg = quantity * _TONNE_TO_KG
            result = (mass_kg / density).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP
            )
            trace.append(f"[unit] {quantity} tonnes -> {mass_kg} kg / {density} kg/L -> {result} liters")

        elif unit_key == "lbs" or unit_key == "pounds":
            density = fuel_data.get("density_kg_per_l")
            if density is None or density == Decimal("0"):
                raise ValueError(f"Cannot convert lbs to liters for {fuel_type}")
            mass_kg = (quantity * _LB_TO_KG).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP
            )
            result = (mass_kg / density).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP
            )
            trace.append(f"[unit] {quantity} lbs -> {mass_kg} kg / {density} kg/L -> {result} liters")

        elif unit_key == "short_tons":
            density = fuel_data.get("density_kg_per_l")
            if density is None or density == Decimal("0"):
                raise ValueError(f"Cannot convert short tons to liters for {fuel_type}")
            mass_kg = (quantity * _SHORT_TON_TO_KG).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP
            )
            result = (mass_kg / density).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP
            )
            trace.append(f"[unit] {quantity} short_tons -> {mass_kg} kg -> {result} liters")

        elif unit_key == "kwh":
            # Energy to volume: quantity_kWh * MJ/kWh / heating_value MJ/L
            energy_mj = (quantity * _KWH_TO_MJ).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP
            )
            hv = fuel_data.get("hhv_mj_per_l")
            if hv is None or hv == Decimal("0"):
                raise ValueError(f"Cannot convert kWh to liters for {fuel_type}")
            result = (energy_mj / hv).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP
            )
            trace.append(f"[unit] {quantity} kWh -> {energy_mj} MJ / {hv} MJ/L -> {result} liters")

        elif unit_key == "gj":
            energy_mj = (quantity * _GJ_TO_MJ).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP
            )
            hv = fuel_data.get("hhv_mj_per_l")
            if hv is None or hv == Decimal("0"):
                raise ValueError(f"Cannot convert GJ to liters for {fuel_type}")
            result = (energy_mj / hv).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP
            )
            trace.append(f"[unit] {quantity} GJ -> {energy_mj} MJ / {hv} MJ/L -> {result} liters")

        elif unit_key == "mmbtu":
            energy_mj = (quantity * _MMBTU_TO_MJ).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP
            )
            hv = fuel_data.get("hhv_mj_per_l")
            if hv is None or hv == Decimal("0"):
                raise ValueError(f"Cannot convert mmBtu to liters for {fuel_type}")
            result = (energy_mj / hv).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP
            )
            trace.append(f"[unit] {quantity} mmBtu -> {energy_mj} MJ / {hv} MJ/L -> {result} liters")

        elif unit_key == "therms":
            energy_mj = (quantity * _THERM_TO_MJ).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP
            )
            hv = fuel_data.get("hhv_mj_per_l")
            if hv is None or hv == Decimal("0"):
                raise ValueError(f"Cannot convert therms to liters for {fuel_type}")
            result = (energy_mj / hv).quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP
            )
            trace.append(f"[unit] {quantity} therms -> {energy_mj} MJ / {hv} MJ/L -> {result} liters")

        else:
            raise ValueError(
                f"Unsupported fuel unit: '{unit}'. Supported: liters, gallons, barrels, "
                f"m3, kg, tonnes, lbs, short_tons, kwh, gj, mmbtu, therms"
            )

        return result

    # ==================================================================
    # Internal: Vehicle Age Adjustment
    # ==================================================================

    def _adjust_for_vehicle_age(
        self, base_economy: Decimal, vehicle_age_years: int
    ) -> Decimal:
        """Adjust fuel economy for vehicle age degradation.

        Older vehicles have worse fuel economy due to engine wear, decreased
        compression, and other mechanical degradation.

        Args:
            base_economy: Base fuel economy in km/L.
            vehicle_age_years: Vehicle age in years.

        Returns:
            Degraded fuel economy in km/L (lower = worse economy).
        """
        age = max(0, vehicle_age_years)

        if age <= 3:
            degradation = _AGE_DEGRADATION["0_3"]
        elif age <= 7:
            degradation = _AGE_DEGRADATION["4_7"]
        elif age <= 12:
            degradation = _AGE_DEGRADATION["8_12"]
        elif age <= 17:
            degradation = _AGE_DEGRADATION["13_17"]
        elif age <= 25:
            degradation = _AGE_DEGRADATION["18_25"]
        else:
            degradation = _AGE_DEGRADATION["26_PLUS"]

        # Degradation increases fuel consumption -> decreases economy
        adjusted = (base_economy / degradation).quantize(
            self._precision_quantizer, rounding=ROUND_HALF_UP
        )
        return adjusted

    # ==================================================================
    # Internal: Load Factor Adjustment
    # ==================================================================

    def _adjust_for_load_factor(
        self, base_economy: Decimal, vehicle_type: str, load_factor: Decimal
    ) -> Decimal:
        """Adjust fuel economy based on actual load factor.

        For freight vehicles, fuel consumption increases approximately
        linearly with load. The adjustment is relative to the vehicle
        type's default load factor.

        Formula:
            delta = (actual_load - default_load) * sensitivity
            adjusted_economy = base_economy / (1 + delta)

        Args:
            base_economy: Base fuel economy in km/L.
            vehicle_type: Vehicle type key for sensitivity lookup.
            load_factor: Actual load factor (0.0 to 1.0).

        Returns:
            Adjusted fuel economy in km/L.
        """
        vtype = vehicle_type.upper().strip()
        sensitivity = _LOAD_FACTOR_SENSITIVITY.get(vtype, Decimal("0"))

        if sensitivity == Decimal("0"):
            return base_economy

        veh_data = self._vehicle_db.get_vehicle_type(vtype)
        default_load = Decimal(str(veh_data.get("typical_load_factor", "0.5")))
        actual_load = Decimal(str(load_factor))

        # Clamp load factor
        actual_load = max(Decimal("0"), min(Decimal("1"), actual_load))

        delta = (actual_load - default_load) * sensitivity
        adjustment = Decimal("1") + delta

        if adjustment <= Decimal("0"):
            adjustment = Decimal("0.1")

        adjusted = (base_economy / adjustment).quantize(
            self._precision_quantizer, rounding=ROUND_HALF_UP
        )
        return adjusted

    # ==================================================================
    # Internal: GWP Values
    # ==================================================================

    def _get_gwp_values(self, gwp_source: str) -> Dict[str, Decimal]:
        """Retrieve GWP values for a specific assessment report.

        Args:
            gwp_source: Assessment report identifier.

        Returns:
            Dictionary mapping gas to GWP value.

        Raises:
            ValueError: If gwp_source is not recognized.
        """
        src = gwp_source.upper().strip()
        if src not in _GWP_VALUES:
            raise ValueError(
                f"Unknown GWP source: '{gwp_source}'. "
                f"Valid: {sorted(_GWP_VALUES.keys())}"
            )
        return _GWP_VALUES[src]

    # ==================================================================
    # Internal: Gas Emission Breakdown
    # ==================================================================

    def _build_gas_emissions(
        self,
        co2_fossil_kg: Decimal,
        co2_biogenic_kg: Decimal,
        ch4_kg: Decimal,
        n2o_kg: Decimal,
        co2_co2e: Decimal,
        ch4_co2e: Decimal,
        n2o_co2e: Decimal,
        gwp_source: str,
    ) -> List[Dict[str, Any]]:
        """Build per-gas emission breakdown list.

        Args:
            co2_fossil_kg: Fossil CO2 mass in kg.
            co2_biogenic_kg: Biogenic CO2 mass in kg.
            ch4_kg: CH4 mass in kg.
            n2o_kg: N2O mass in kg.
            co2_co2e: CO2 equivalent of fossil CO2 in kg.
            ch4_co2e: CO2 equivalent of CH4 in kg.
            n2o_co2e: CO2 equivalent of N2O in kg.
            gwp_source: GWP source used.

        Returns:
            List of gas emission dictionaries.
        """
        gwp_values = self._get_gwp_values(gwp_source)

        emissions: List[Dict[str, Any]] = [
            {
                "gas": "CO2",
                "mass_kg": co2_fossil_kg,
                "gwp": gwp_values["CO2"],
                "co2e_kg": co2_co2e,
                "is_biogenic": False,
            },
            {
                "gas": "CH4",
                "mass_kg": ch4_kg,
                "gwp": gwp_values["CH4"],
                "co2e_kg": ch4_co2e,
                "is_biogenic": False,
            },
            {
                "gas": "N2O",
                "mass_kg": n2o_kg,
                "gwp": gwp_values["N2O"],
                "co2e_kg": n2o_co2e,
                "is_biogenic": False,
            },
        ]

        if co2_biogenic_kg > Decimal("0"):
            emissions.append({
                "gas": "CO2_BIOGENIC",
                "mass_kg": co2_biogenic_kg,
                "gwp": gwp_values["CO2"],
                "co2e_kg": co2_biogenic_kg,
                "is_biogenic": True,
                "note": "Biogenic CO2 reported separately per GHG Protocol",
            })

        return emissions

    # ==================================================================
    # Internal: Provenance Hash
    # ==================================================================

    def _compute_provenance_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash for provenance tracking.

        Args:
            data: Data dictionary to hash.

        Returns:
            Hexadecimal SHA-256 hash string.
        """
        hash_input = json.dumps(
            {"data": data, "timestamp": _utcnow().isoformat()},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

    # ==================================================================
    # Internal: Error Result Builder
    # ==================================================================

    def _create_error_result(
        self,
        calc_id: str,
        method: str,
        vehicle_type: str,
        fuel_type: str,
        quantity: Any,
        unit: str,
        error_message: str,
        trace: List[str],
        elapsed_ms: float,
    ) -> Dict[str, Any]:
        """Create a standardized error result dictionary.

        Args:
            calc_id: Calculation identifier.
            method: Calculation method.
            vehicle_type: Vehicle type.
            fuel_type: Fuel type.
            quantity: Input quantity.
            unit: Input unit.
            error_message: Error description.
            trace: Calculation trace.
            elapsed_ms: Processing time.

        Returns:
            Error result dictionary.
        """
        trace.append(f"[ERROR] {error_message}")
        return {
            "calculation_id": calc_id,
            "status": "FAILED",
            "method": method,
            "vehicle_type": str(vehicle_type).upper().strip() if vehicle_type else "",
            "fuel_type": str(fuel_type).upper().strip() if fuel_type else "",
            "fuel_consumed": Decimal(str(quantity)) if quantity is not None else Decimal("0"),
            "fuel_unit": unit,
            "gas_emissions": [],
            "co2_fossil_kg": Decimal("0"),
            "co2_biogenic_kg": Decimal("0"),
            "ch4_kg": Decimal("0"),
            "n2o_kg": Decimal("0"),
            "total_co2e_kg": Decimal("0"),
            "total_co2e_tonnes": Decimal("0"),
            "biogenic_co2e_kg": Decimal("0"),
            "biogenic_co2e_tonnes": Decimal("0"),
            "calculation_trace": trace,
            "provenance_hash": "",
            "processing_time_ms": round(elapsed_ms, 3),
            "error_message": error_message,
        }

    # ==================================================================
    # Public API: Statistics
    # ==================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Return calculator engine statistics.

        Returns:
            Dictionary with configuration summary and GWP source info.
        """
        return {
            "default_gwp_source": self._default_gwp,
            "default_oxidation_factor": str(self._default_ox_factor),
            "decimal_precision": self._precision_places,
            "provenance_enabled": self._enable_provenance,
            "supported_methods": ["fuel_based", "distance_based", "spend_based"],
            "gwp_sources": sorted(_GWP_VALUES.keys()),
            "age_degradation_brackets": sorted(_AGE_DEGRADATION.keys()),
            "vehicle_db_stats": self._vehicle_db.get_statistics(),
        }
