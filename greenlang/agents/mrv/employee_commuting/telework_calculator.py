# -*- coding: utf-8 -*-
"""
Telework Calculator Engine - Engine 4: Employee Commuting Agent (AGENT-MRV-020)

Calculates GHG emissions from telework / work-from-home activities including
home office electricity, heating, cooling, and equipment lifecycle emissions.
Uses IEA 2024 grid factors, DEFRA 2024 heating fuel factors, and GHG Protocol
Technical Guidance for Scope 3 Category 7 telework methodology.

Zero-Hallucination Guarantees:
    - All calculations use Python Decimal (8 decimal places, ROUND_HALF_UP)
    - No LLM calls anywhere in the calculation path
    - Every intermediate value is deterministic and reproducible
    - SHA-256 provenance hash on every result
    - Grid factors from IEA 2024, heating factors from DEFRA 2024

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-020 Employee Commuting (GL-MRV-S3-007)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ==============================================================================
# AGENT METADATA
# ==============================================================================

ENGINE_ID: str = "telework_calculator_engine"
ENGINE_VERSION: str = "1.0.0"
AGENT_ID: str = "GL-MRV-S3-007"
AGENT_COMPONENT: str = "AGENT-MRV-020"

# ==============================================================================
# DECIMAL PRECISION CONSTANTS
# ==============================================================================

_QUANT_8DP: Decimal = Decimal("0.00000001")
_ZERO: Decimal = Decimal("0")
_ONE: Decimal = Decimal("1")
_FIVE: Decimal = Decimal("5")
_TWENTY_FOUR: Decimal = Decimal("24")
ROUNDING = ROUND_HALF_UP

# Batch processing limits
_MAX_BATCH_SIZE: int = 10000


# ==============================================================================
# HELPER: Quantize a Decimal to 8 decimal places
# ==============================================================================


def _q(value: Decimal) -> Decimal:
    """
    Quantize a Decimal value to 8 decimal places with ROUND_HALF_UP.

    Args:
        value: The Decimal value to quantize.

    Returns:
        Quantized Decimal with exactly 8 decimal places.
    """
    return value.quantize(_QUANT_8DP, rounding=ROUNDING)


# ==============================================================================
# EMISSION FACTOR TABLES
# ==============================================================================

# Country grid emission factors (kg CO2e per kWh) - IEA 2024
COUNTRY_GRID_FACTORS: Dict[str, Decimal] = {
    "US": Decimal("0.37938"),
    "GB": Decimal("0.20705"),
    "DE": Decimal("0.33809"),
    "FR": Decimal("0.05101"),
    "JP": Decimal("0.43479"),
    "CN": Decimal("0.55010"),
    "IN": Decimal("0.70767"),
    "AU": Decimal("0.62109"),
    "CA": Decimal("0.12022"),
    "BR": Decimal("0.07350"),
    "KR": Decimal("0.41590"),
    "IT": Decimal("0.25610"),
    "ES": Decimal("0.14820"),
    "NL": Decimal("0.28940"),
    "SE": Decimal("0.00830"),
    "NO": Decimal("0.00760"),
    "PL": Decimal("0.63520"),
    "ZA": Decimal("0.91820"),
    "MX": Decimal("0.39450"),
    "SG": Decimal("0.40810"),
}

# US eGRID subregional factors (kg CO2e per kWh)
US_EGRID_FACTORS: Dict[str, Decimal] = {
    "RMPA": Decimal("0.57235"),
    "SRSO": Decimal("0.38791"),
    "RFCW": Decimal("0.45123"),
    "NEWE": Decimal("0.21456"),
    "CAMX": Decimal("0.22134"),
    "NWPP": Decimal("0.28976"),
    "SRMW": Decimal("0.62341"),
    "ERCT": Decimal("0.35678"),
}

# Heating fuel emission factors (kg CO2e per kWh) - DEFRA 2024
HEATING_FUEL_FACTORS: Dict[str, Decimal] = {
    "natural_gas": Decimal("0.18316"),
    "heating_oil": Decimal("0.24680"),
    "propane": Decimal("0.21447"),
    "electric_heat_pump": Decimal("0.00000"),  # Uses grid factor / COP
    "district_heating": Decimal("0.16950"),
    "biomass": Decimal("0.01514"),
}

# Home office energy defaults by climate zone (kWh per day)
CLIMATE_ZONE_DEFAULTS: Dict[str, Dict[str, Decimal]] = {
    "tropical": {
        "electricity": Decimal("3.20"),
        "heating": Decimal("0.00"),
        "cooling": Decimal("4.50"),
    },
    "arid": {
        "electricity": Decimal("3.00"),
        "heating": Decimal("1.50"),
        "cooling": Decimal("3.80"),
    },
    "temperate": {
        "electricity": Decimal("2.50"),
        "heating": Decimal("3.20"),
        "cooling": Decimal("1.80"),
    },
    "continental": {
        "electricity": Decimal("2.50"),
        "heating": Decimal("4.50"),
        "cooling": Decimal("1.50"),
    },
    "polar": {
        "electricity": Decimal("2.80"),
        "heating": Decimal("6.00"),
        "cooling": Decimal("0.20"),
    },
}

# Heat pump COP by climate zone
HEAT_PUMP_COP: Dict[str, Decimal] = {
    "tropical": Decimal("4.50"),
    "arid": Decimal("3.80"),
    "temperate": Decimal("3.50"),
    "continental": Decimal("3.00"),
    "polar": Decimal("2.50"),
}

# Equipment lifecycle emissions (kg CO2e per day, amortized over useful life)
EQUIPMENT_EMISSIONS: Dict[str, Decimal] = {
    "laptop": Decimal("0.04110"),       # ~300 kg over 5 yr / 1460 work days
    "monitor": Decimal("0.05480"),      # ~400 kg over 5 yr
    "desk_phone": Decimal("0.00548"),   # ~40 kg over 5 yr
    "router": Decimal("0.01370"),       # ~100 kg over 5 yr
    "webcam": Decimal("0.00274"),       # ~20 kg over 5 yr
    "headset": Decimal("0.00137"),      # ~10 kg over 5 yr
}

# Telework category -> WFH days per week
TELEWORK_WFH_DAYS: Dict[str, int] = {
    "full_remote": 5,
    "hybrid_4day": 4,
    "hybrid_3day": 3,
    "hybrid_2day": 2,
    "hybrid_1day": 1,
    "office_based": 0,
}

# Seasonal allocation factors (fraction of year heating/cooling applies)
SEASONAL_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "tropical": {
        "heating_fraction": Decimal("0.00"),
        "cooling_fraction": Decimal("0.83"),
    },
    "arid": {
        "heating_fraction": Decimal("0.25"),
        "cooling_fraction": Decimal("0.58"),
    },
    "temperate": {
        "heating_fraction": Decimal("0.42"),
        "cooling_fraction": Decimal("0.25"),
    },
    "continental": {
        "heating_fraction": Decimal("0.50"),
        "cooling_fraction": Decimal("0.25"),
    },
    "polar": {
        "heating_fraction": Decimal("0.67"),
        "cooling_fraction": Decimal("0.08"),
    },
}

# Default commute emission factors for net impact estimation (kg CO2e per km)
# Source: DEFRA 2024 average car, bus, rail factors
_COMMUTE_MODE_EFS: Dict[str, Decimal] = {
    "car_petrol": Decimal("0.17049"),
    "car_diesel": Decimal("0.16844"),
    "car_hybrid": Decimal("0.11807"),
    "car_ev": Decimal("0.04631"),
    "bus": Decimal("0.08920"),
    "rail": Decimal("0.03549"),
    "metro": Decimal("0.02846"),
    "motorcycle": Decimal("0.11337"),
    "cycling": Decimal("0.00000"),
    "walking": Decimal("0.00000"),
    "car_average": Decimal("0.16844"),
}

# Default office energy consumption per employee per day (kWh)
_DEFAULT_OFFICE_KWH_PER_EMPLOYEE_DAY: Decimal = Decimal("8.50")


# ==============================================================================
# SINGLETON INSTANCE MANAGEMENT
# ==============================================================================

_instance: Optional["TeleworkCalculatorEngine"] = None
_instance_lock: threading.Lock = threading.Lock()


# ==============================================================================
# TeleworkCalculatorEngine
# ==============================================================================


class TeleworkCalculatorEngine:
    """
    Engine 4: Telework / work-from-home emissions calculator.

    Calculates GHG emissions from remote work activities including home office
    electricity consumption (base load), heating energy, cooling energy, and
    optional equipment lifecycle emissions. Uses IEA 2024 country-level and
    US eGRID subregional grid emission factors, DEFRA 2024 heating fuel
    emission factors, and climate-zone-specific energy defaults.

    The engine follows GreenLang's zero-hallucination principle: all
    calculations are deterministic Python Decimal arithmetic with published
    emission factors. No LLM calls are used for any numeric computation.

    Calculation Pipeline:
        1. Validate all input parameters
        2. Determine annual WFH days from telework category and working days
        3. Calculate electricity emissions (base load, adjusted by work fraction)
        4. Calculate heating emissions (seasonal, fuel-specific or heat pump)
        5. Calculate cooling emissions (seasonal, electric)
        6. Calculate equipment lifecycle emissions (optional)
        7. Sum all components for total CO2e
        8. Generate SHA-256 provenance hash
        9. Return detailed result dictionary

    Thread Safety:
        This engine uses the singleton pattern with double-checked locking.
        All mutable state (calculation count) is protected by a reentrant
        lock. The singleton instance is created lazily on first access.

    Attributes:
        _lock: Reentrant lock for thread safety.
        _calculation_count: Running count of calculations performed.
        _initialized: Flag to prevent re-initialization on singleton access.

    Example:
        >>> engine = TeleworkCalculatorEngine.get_instance()
        >>> result = engine.calculate_telework_emissions(
        ...     telework_category="full_remote",
        ...     country_code="US",
        ...     climate_zone="temperate",
        ... )
        >>> assert result["total_co2e_kg"] > Decimal("0")
        >>> assert len(result["provenance_hash"]) == 64
    """

    # ------------------------------------------------------------------
    # Singleton Access
    # ------------------------------------------------------------------

    @staticmethod
    def get_instance() -> "TeleworkCalculatorEngine":
        """
        Get or create the singleton TeleworkCalculatorEngine instance.

        Thread-safe lazy initialization using double-checked locking.

        Returns:
            Singleton TeleworkCalculatorEngine instance.

        Example:
            >>> engine = TeleworkCalculatorEngine.get_instance()
            >>> isinstance(engine, TeleworkCalculatorEngine)
            True
        """
        global _instance
        if _instance is None:
            with _instance_lock:
                if _instance is None:
                    _instance = TeleworkCalculatorEngine()
        return _instance

    @staticmethod
    def reset_instance() -> None:
        """
        Reset the singleton instance (for testing only).

        This method is intended exclusively for unit tests that need
        a fresh engine instance. It should never be called in production.
        """
        global _instance
        with _instance_lock:
            _instance = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        """
        Initialise the TeleworkCalculatorEngine.

        Sets up internal state including the calculation counter and
        thread lock. This constructor should not be called directly;
        use get_instance() instead for singleton access.
        """
        self._lock: threading.RLock = threading.RLock()
        self._calculation_count: int = 0
        self._initialized: bool = True

        logger.info(
            "TeleworkCalculatorEngine initialised: engine=%s, version=%s, "
            "agent=%s, countries=%d, egrid_subregions=%d, climate_zones=%d, "
            "heating_fuels=%d, equipment_types=%d",
            ENGINE_ID,
            ENGINE_VERSION,
            AGENT_ID,
            len(COUNTRY_GRID_FACTORS),
            len(US_EGRID_FACTORS),
            len(CLIMATE_ZONE_DEFAULTS),
            len(HEATING_FUEL_FACTORS),
            len(EQUIPMENT_EMISSIONS),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def calculation_count(self) -> int:
        """Return the total number of calculations performed by this engine."""
        return self._calculation_count

    @property
    def engine_id(self) -> str:
        """Return the engine identifier."""
        return ENGINE_ID

    @property
    def engine_version(self) -> str:
        """Return the engine version."""
        return ENGINE_VERSION

    # ==================================================================
    # 1. calculate_telework_emissions - Core telework calculation
    # ==================================================================

    def calculate_telework_emissions(
        self,
        telework_category: str,
        country_code: str = "US",
        egrid_subregion: Optional[str] = None,
        climate_zone: str = "temperate",
        heating_fuel: str = "natural_gas",
        working_days: int = 240,
        include_equipment: bool = False,
        equipment_list: Optional[List[str]] = None,
        custom_electricity_kwh: Optional[Decimal] = None,
        custom_heating_kwh: Optional[Decimal] = None,
        custom_cooling_kwh: Optional[Decimal] = None,
        work_fraction: Decimal = Decimal("0.10"),
        working_hours: Decimal = Decimal("8.0"),
    ) -> Dict[str, Any]:
        """
        Calculate GHG emissions from telework / work-from-home activities.

        This is the primary calculation method. It computes electricity,
        heating, cooling, and optional equipment lifecycle emissions for
        a teleworker based on their work pattern, location, climate zone,
        and heating fuel type.

        Formula:
            wfh_days_per_week = TELEWORK_WFH_DAYS[category]
            annual_wfh_days = (wfh_days_per_week / 5) * working_days

            Electricity:
                daily_elec = (custom or default) * work_fraction * (hours / 24)
                annual_elec_kwh = daily_elec * annual_wfh_days
                elec_co2e = annual_elec_kwh * grid_factor

            Heating (seasonal):
                daily_heat = default * work_fraction
                annual_heat_kwh = daily_heat * annual_wfh_days * heating_fraction
                If heat pump: heat_co2e = (annual_heat_kwh / COP) * grid_factor
                Else: heat_co2e = annual_heat_kwh * fuel_factor

            Cooling (seasonal):
                daily_cool = default * work_fraction
                annual_cool_kwh = daily_cool * annual_wfh_days * cooling_fraction
                cool_co2e = annual_cool_kwh * grid_factor

            Equipment (optional):
                equip_co2e = SUM(EQUIPMENT_EMISSIONS[item] * annual_wfh_days)

            total_co2e = elec_co2e + heat_co2e + cool_co2e + equip_co2e

        Args:
            telework_category: Telework pattern key. Must be one of
                TELEWORK_WFH_DAYS keys: 'full_remote', 'hybrid_4day',
                'hybrid_3day', 'hybrid_2day', 'hybrid_1day', 'office_based'.
            country_code: ISO 3166-1 alpha-2 country code for grid factor.
                Defaults to 'US'.
            egrid_subregion: Optional US eGRID subregion code. If provided
                and country_code is 'US', overrides the national grid factor.
            climate_zone: Climate zone for energy defaults and seasonal
                factors. One of: 'tropical', 'arid', 'temperate',
                'continental', 'polar'. Defaults to 'temperate'.
            heating_fuel: Heating fuel type for heating emissions. One of
                HEATING_FUEL_FACTORS keys. Defaults to 'natural_gas'.
            working_days: Total annual working days. Defaults to 240.
            include_equipment: Whether to include equipment lifecycle
                emissions. Defaults to False.
            equipment_list: List of equipment type keys (e.g. ['laptop',
                'monitor']). Required if include_equipment is True. Defaults
                to ['laptop', 'monitor'] if include_equipment is True and
                no list is provided.
            custom_electricity_kwh: Override daily electricity consumption
                in kWh. If None, uses CLIMATE_ZONE_DEFAULTS.
            custom_heating_kwh: Override daily heating energy in kWh.
                If None, uses CLIMATE_ZONE_DEFAULTS.
            custom_cooling_kwh: Override daily cooling energy in kWh.
                If None, uses CLIMATE_ZONE_DEFAULTS.
            work_fraction: Fraction of home energy attributable to work
                (e.g. 0.10 = 10% of home). Defaults to Decimal('0.10').
            working_hours: Daily working hours. Defaults to Decimal('8.0').

        Returns:
            Dictionary containing:
                - total_co2e_kg: Total telework emissions (kg CO2e/year)
                - electricity_co2e_kg: Electricity component (kg CO2e/year)
                - heating_co2e_kg: Heating component (kg CO2e/year)
                - cooling_co2e_kg: Cooling component (kg CO2e/year)
                - equipment_co2e_kg: Equipment component (kg CO2e/year)
                - annual_wfh_days: Annual work-from-home days
                - annual_electricity_kwh: Annual electricity kWh
                - annual_heating_kwh: Annual heating kWh
                - annual_cooling_kwh: Annual cooling kWh
                - grid_factor_used: Grid emission factor applied
                - heating_fuel: Heating fuel type used
                - climate_zone: Climate zone used
                - country_code: Country code used
                - telework_category: Telework category used
                - method: Calculation method identifier
                - engine_id: Engine identifier
                - engine_version: Engine version
                - provenance_hash: SHA-256 hash for audit trail

        Raises:
            ValueError: If any input parameter fails validation.

        Example:
            >>> engine = TeleworkCalculatorEngine.get_instance()
            >>> result = engine.calculate_telework_emissions(
            ...     telework_category="full_remote",
            ...     country_code="US",
            ...     climate_zone="temperate",
            ... )
            >>> assert result["total_co2e_kg"] > Decimal("0")
        """
        start_time = time.monotonic()

        with self._lock:
            # Step 1: Validate all inputs
            errors = self._validate_inputs({
                "telework_category": telework_category,
                "country_code": country_code,
                "egrid_subregion": egrid_subregion,
                "climate_zone": climate_zone,
                "heating_fuel": heating_fuel,
                "working_days": working_days,
                "include_equipment": include_equipment,
                "equipment_list": equipment_list,
                "custom_electricity_kwh": custom_electricity_kwh,
                "custom_heating_kwh": custom_heating_kwh,
                "custom_cooling_kwh": custom_cooling_kwh,
                "work_fraction": work_fraction,
                "working_hours": working_hours,
            })
            if errors:
                raise ValueError(
                    f"Telework input validation failed: {'; '.join(errors)}"
                )

            # Step 2: Determine annual WFH days
            wfh_days_per_week = TELEWORK_WFH_DAYS[telework_category]
            annual_wfh_days = self._calculate_annual_wfh_days(
                wfh_days_per_week, working_days
            )

            # Short-circuit for office_based (0 WFH days)
            if annual_wfh_days == _ZERO:
                result = self._build_zero_result(
                    telework_category, country_code, climate_zone,
                    heating_fuel,
                )
                self._calculation_count += 1
                return result

            # Step 3: Resolve grid factor
            grid_factor = self._get_grid_factor(country_code, egrid_subregion)

            # Step 4: Calculate electricity emissions
            electricity_result = self._calculate_electricity_emissions(
                annual_wfh_days=annual_wfh_days,
                climate_zone=climate_zone,
                grid_factor=grid_factor,
                work_fraction=work_fraction,
                working_hours=working_hours,
                custom_electricity_kwh=custom_electricity_kwh,
            )

            # Step 5: Calculate heating emissions
            heating_result = self._calculate_heating_emissions(
                annual_wfh_days=annual_wfh_days,
                climate_zone=climate_zone,
                heating_fuel=heating_fuel,
                grid_factor=grid_factor,
                work_fraction=work_fraction,
                custom_heating_kwh=custom_heating_kwh,
            )

            # Step 6: Calculate cooling emissions
            cooling_result = self._calculate_cooling_emissions(
                annual_wfh_days=annual_wfh_days,
                climate_zone=climate_zone,
                grid_factor=grid_factor,
                work_fraction=work_fraction,
                custom_cooling_kwh=custom_cooling_kwh,
            )

            # Step 7: Calculate equipment emissions (optional)
            if include_equipment:
                resolved_equipment = equipment_list if equipment_list else [
                    "laptop", "monitor",
                ]
                equipment_co2e = self._calculate_equipment_emissions(
                    resolved_equipment, annual_wfh_days,
                )
            else:
                equipment_co2e = _ZERO

            # Step 8: Sum all components
            total_co2e = _q(
                electricity_result["co2e"]
                + heating_result["co2e"]
                + cooling_result["co2e"]
                + equipment_co2e
            )

            # Step 9: Build result dictionary
            result: Dict[str, Any] = {
                "total_co2e_kg": total_co2e,
                "electricity_co2e_kg": electricity_result["co2e"],
                "heating_co2e_kg": heating_result["co2e"],
                "cooling_co2e_kg": cooling_result["co2e"],
                "equipment_co2e_kg": _q(equipment_co2e),
                "annual_wfh_days": annual_wfh_days,
                "annual_electricity_kwh": electricity_result["annual_kwh"],
                "annual_heating_kwh": heating_result["annual_kwh"],
                "annual_cooling_kwh": cooling_result["annual_kwh"],
                "grid_factor_used": grid_factor,
                "heating_fuel": heating_fuel,
                "climate_zone": climate_zone,
                "country_code": country_code.upper(),
                "telework_category": telework_category,
                "method": "telework_energy_based",
                "engine_id": ENGINE_ID,
                "engine_version": ENGINE_VERSION,
            }

            # Step 10: Provenance hash
            provenance_hash = self._calculate_provenance_hash(
                {
                    "telework_category": telework_category,
                    "country_code": country_code,
                    "egrid_subregion": egrid_subregion,
                    "climate_zone": climate_zone,
                    "heating_fuel": heating_fuel,
                    "working_days": working_days,
                    "work_fraction": str(work_fraction),
                    "working_hours": str(working_hours),
                },
                {
                    "total_co2e_kg": str(total_co2e),
                    "electricity_co2e_kg": str(electricity_result["co2e"]),
                    "heating_co2e_kg": str(heating_result["co2e"]),
                    "cooling_co2e_kg": str(cooling_result["co2e"]),
                    "equipment_co2e_kg": str(equipment_co2e),
                    "annual_wfh_days": str(annual_wfh_days),
                },
            )
            result["provenance_hash"] = provenance_hash

            # Step 11: Update counter and log
            self._calculation_count += 1
            duration = time.monotonic() - start_time

            logger.debug(
                "Telework calculation complete: category=%s, country=%s, "
                "zone=%s, fuel=%s, wfh_days=%s, total_co2e=%s kg, "
                "duration=%.4fs",
                telework_category,
                country_code,
                climate_zone,
                heating_fuel,
                annual_wfh_days,
                total_co2e,
                duration,
            )

            return result

    # ==================================================================
    # 2. calculate_hybrid_worker - Hybrid worker with commute offset
    # ==================================================================

    def calculate_hybrid_worker(
        self,
        office_days_per_week: int,
        country_code: str = "US",
        egrid_subregion: Optional[str] = None,
        climate_zone: str = "temperate",
        heating_fuel: str = "natural_gas",
        working_days: int = 240,
        commute_avoided_co2e: Optional[Decimal] = None,
        include_equipment: bool = False,
        equipment_list: Optional[List[str]] = None,
        work_fraction: Decimal = Decimal("0.10"),
        working_hours: Decimal = Decimal("8.0"),
    ) -> Dict[str, Any]:
        """
        Calculate telework emissions for a hybrid worker with optional net calculation.

        Computes gross telework emissions and, if commute_avoided_co2e is
        provided, calculates the net impact (telework emissions minus avoided
        commute emissions).

        Args:
            office_days_per_week: Number of days in office per week (0-5).
            country_code: ISO 3166-1 alpha-2 country code. Defaults to 'US'.
            egrid_subregion: Optional US eGRID subregion code.
            climate_zone: Climate zone for energy defaults. Defaults to
                'temperate'.
            heating_fuel: Heating fuel type. Defaults to 'natural_gas'.
            working_days: Total annual working days. Defaults to 240.
            commute_avoided_co2e: Annual avoided commute emissions in kg CO2e
                for the WFH days. If provided, net impact is calculated.
            include_equipment: Whether to include equipment emissions.
            equipment_list: List of equipment type keys.
            work_fraction: Home energy work fraction. Defaults to 0.10.
            working_hours: Daily working hours. Defaults to 8.0.

        Returns:
            Dictionary containing:
                - gross_telework_co2e_kg: Gross telework emissions
                - avoided_commute_co2e_kg: Avoided commute emissions (or 0)
                - net_co2e_kg: Net impact (gross - avoided)
                - is_carbon_saving: True if net < 0 (WFH saves carbon)
                - office_days_per_week: Office days per week
                - wfh_days_per_week: WFH days per week
                - telework_details: Full telework calculation result
                - provenance_hash: SHA-256 hash

        Raises:
            ValueError: If office_days_per_week is not in range [0, 5].

        Example:
            >>> result = engine.calculate_hybrid_worker(
            ...     office_days_per_week=3,
            ...     country_code="GB",
            ...     commute_avoided_co2e=Decimal("450.0"),
            ... )
            >>> assert "net_co2e_kg" in result
        """
        start_time = time.monotonic()

        # Validate office days
        if not isinstance(office_days_per_week, int):
            raise ValueError(
                f"office_days_per_week must be an integer, "
                f"got {type(office_days_per_week).__name__}"
            )
        if office_days_per_week < 0 or office_days_per_week > 5:
            raise ValueError(
                f"office_days_per_week must be between 0 and 5, "
                f"got {office_days_per_week}"
            )

        # Map office days to telework category
        wfh_days = 5 - office_days_per_week
        category = self._wfh_days_to_category(wfh_days)

        # Calculate gross telework emissions
        telework_result = self.calculate_telework_emissions(
            telework_category=category,
            country_code=country_code,
            egrid_subregion=egrid_subregion,
            climate_zone=climate_zone,
            heating_fuel=heating_fuel,
            working_days=working_days,
            include_equipment=include_equipment,
            equipment_list=equipment_list,
            work_fraction=work_fraction,
            working_hours=working_hours,
        )

        gross_co2e = telework_result["total_co2e_kg"]

        # Calculate net impact
        avoided = _ZERO
        if commute_avoided_co2e is not None:
            avoided = _q(Decimal(str(commute_avoided_co2e)))

        net_co2e = _q(gross_co2e - avoided)
        is_carbon_saving = net_co2e < _ZERO

        # Provenance hash for the hybrid calculation
        provenance_hash = self._calculate_provenance_hash(
            {
                "type": "hybrid_worker",
                "office_days_per_week": office_days_per_week,
                "wfh_days_per_week": wfh_days,
                "country_code": country_code,
                "commute_avoided_co2e": str(avoided),
            },
            {
                "gross_co2e": str(gross_co2e),
                "net_co2e": str(net_co2e),
                "is_carbon_saving": str(is_carbon_saving),
            },
        )

        duration = time.monotonic() - start_time

        logger.debug(
            "Hybrid worker calculation: office=%d/wk, wfh=%d/wk, "
            "gross=%s kg, avoided=%s kg, net=%s kg, saving=%s, "
            "duration=%.4fs",
            office_days_per_week,
            wfh_days,
            gross_co2e,
            avoided,
            net_co2e,
            is_carbon_saving,
            duration,
        )

        return {
            "gross_telework_co2e_kg": gross_co2e,
            "avoided_commute_co2e_kg": avoided,
            "net_co2e_kg": net_co2e,
            "is_carbon_saving": is_carbon_saving,
            "office_days_per_week": office_days_per_week,
            "wfh_days_per_week": wfh_days,
            "telework_details": telework_result,
            "method": "hybrid_worker_net",
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "provenance_hash": provenance_hash,
        }

    # ==================================================================
    # 3. calculate_full_remote - Convenience for full remote workers
    # ==================================================================

    def calculate_full_remote(
        self,
        country_code: str = "US",
        egrid_subregion: Optional[str] = None,
        climate_zone: str = "temperate",
        heating_fuel: str = "natural_gas",
        working_days: int = 240,
        include_equipment: bool = False,
        equipment_list: Optional[List[str]] = None,
        work_fraction: Decimal = Decimal("0.10"),
        working_hours: Decimal = Decimal("8.0"),
    ) -> Dict[str, Any]:
        """
        Calculate telework emissions for a fully remote worker (5 days/week).

        Convenience wrapper around calculate_telework_emissions with
        telework_category='full_remote'.

        Args:
            country_code: ISO 3166-1 alpha-2 country code. Defaults to 'US'.
            egrid_subregion: Optional US eGRID subregion code.
            climate_zone: Climate zone. Defaults to 'temperate'.
            heating_fuel: Heating fuel type. Defaults to 'natural_gas'.
            working_days: Total annual working days. Defaults to 240.
            include_equipment: Whether to include equipment emissions.
            equipment_list: List of equipment type keys.
            work_fraction: Home energy work fraction. Defaults to 0.10.
            working_hours: Daily working hours. Defaults to 8.0.

        Returns:
            Full telework calculation result dictionary (same structure as
            calculate_telework_emissions).

        Example:
            >>> result = engine.calculate_full_remote(
            ...     country_code="DE",
            ...     climate_zone="continental",
            ...     include_equipment=True,
            ... )
            >>> assert result["telework_category"] == "full_remote"
        """
        return self.calculate_telework_emissions(
            telework_category="full_remote",
            country_code=country_code,
            egrid_subregion=egrid_subregion,
            climate_zone=climate_zone,
            heating_fuel=heating_fuel,
            working_days=working_days,
            include_equipment=include_equipment,
            equipment_list=equipment_list,
            work_fraction=work_fraction,
            working_hours=working_hours,
        )

    # ==================================================================
    # 4. calculate_partial_telework - Custom WFH schedule
    # ==================================================================

    def calculate_partial_telework(
        self,
        wfh_days_per_week: int,
        country_code: str = "US",
        egrid_subregion: Optional[str] = None,
        climate_zone: str = "temperate",
        heating_fuel: str = "natural_gas",
        working_days: int = 240,
        include_equipment: bool = False,
        equipment_list: Optional[List[str]] = None,
        work_fraction: Decimal = Decimal("0.10"),
        working_hours: Decimal = Decimal("8.0"),
    ) -> Dict[str, Any]:
        """
        Calculate telework emissions for a custom WFH schedule.

        For schedules that do not map directly to standard telework
        categories (e.g., 2.5 days per week), this method maps the
        provided WFH days to the nearest matching category.

        Args:
            wfh_days_per_week: Number of WFH days per week (0-5).
            country_code: ISO 3166-1 alpha-2 country code. Defaults to 'US'.
            egrid_subregion: Optional US eGRID subregion code.
            climate_zone: Climate zone. Defaults to 'temperate'.
            heating_fuel: Heating fuel type. Defaults to 'natural_gas'.
            working_days: Total annual working days. Defaults to 240.
            include_equipment: Whether to include equipment emissions.
            equipment_list: List of equipment type keys.
            work_fraction: Home energy work fraction. Defaults to 0.10.
            working_hours: Daily working hours. Defaults to 8.0.

        Returns:
            Full telework calculation result dictionary.

        Raises:
            ValueError: If wfh_days_per_week is not in range [0, 5].

        Example:
            >>> result = engine.calculate_partial_telework(
            ...     wfh_days_per_week=2,
            ...     country_code="FR",
            ... )
            >>> assert result["telework_category"] == "hybrid_2day"
        """
        if not isinstance(wfh_days_per_week, int):
            raise ValueError(
                f"wfh_days_per_week must be an integer, "
                f"got {type(wfh_days_per_week).__name__}"
            )
        if wfh_days_per_week < 0 or wfh_days_per_week > 5:
            raise ValueError(
                f"wfh_days_per_week must be between 0 and 5, "
                f"got {wfh_days_per_week}"
            )

        category = self._wfh_days_to_category(wfh_days_per_week)

        return self.calculate_telework_emissions(
            telework_category=category,
            country_code=country_code,
            egrid_subregion=egrid_subregion,
            climate_zone=climate_zone,
            heating_fuel=heating_fuel,
            working_days=working_days,
            include_equipment=include_equipment,
            equipment_list=equipment_list,
            work_fraction=work_fraction,
            working_hours=working_hours,
        )

    # ==================================================================
    # 5. calculate_office_energy_savings - Office energy reduction
    # ==================================================================

    def calculate_office_energy_savings(
        self,
        telework_category: str,
        office_kwh_per_employee_per_day: Optional[Decimal] = None,
        country_code: str = "US",
        egrid_subregion: Optional[str] = None,
        working_days: int = 240,
    ) -> Dict[str, Any]:
        """
        Estimate office energy savings from telework adoption.

        Calculates the avoided office energy consumption (and resulting
        CO2e reduction) when employees work from home instead of the
        office. This provides the "credit side" of the telework equation.

        Formula:
            wfh_days = (TELEWORK_WFH_DAYS[category] / 5) * working_days
            avoided_kwh = wfh_days * office_kwh_per_employee_per_day
            avoided_co2e = avoided_kwh * grid_factor

        Args:
            telework_category: Telework pattern key from TELEWORK_WFH_DAYS.
            office_kwh_per_employee_per_day: Daily office energy per employee
                in kWh. Defaults to 8.50 kWh if not provided.
            country_code: ISO 3166-1 alpha-2 country code. Defaults to 'US'.
            egrid_subregion: Optional US eGRID subregion code.
            working_days: Total annual working days. Defaults to 240.

        Returns:
            Dictionary containing:
                - avoided_office_kwh: Annual avoided office energy (kWh)
                - avoided_office_co2e_kg: Annual avoided office CO2e (kg)
                - office_kwh_per_employee_per_day: Office energy rate used
                - annual_wfh_days: Annual WFH days
                - grid_factor_used: Grid emission factor
                - telework_category: Category used
                - provenance_hash: SHA-256 hash

        Raises:
            ValueError: If telework_category is invalid.

        Example:
            >>> result = engine.calculate_office_energy_savings(
            ...     telework_category="hybrid_3day",
            ...     country_code="US",
            ... )
            >>> assert result["avoided_office_co2e_kg"] > Decimal("0")
        """
        start_time = time.monotonic()

        # Validate category
        if telework_category not in TELEWORK_WFH_DAYS:
            raise ValueError(
                f"Unknown telework_category '{telework_category}'. "
                f"Available: {list(TELEWORK_WFH_DAYS.keys())}"
            )

        # Resolve office energy rate
        office_rate = (
            Decimal(str(office_kwh_per_employee_per_day))
            if office_kwh_per_employee_per_day is not None
            else _DEFAULT_OFFICE_KWH_PER_EMPLOYEE_DAY
        )

        # Calculate annual WFH days
        wfh_days_per_week = TELEWORK_WFH_DAYS[telework_category]
        annual_wfh_days = self._calculate_annual_wfh_days(
            wfh_days_per_week, working_days
        )

        # Calculate avoided office energy
        avoided_kwh = _q(annual_wfh_days * office_rate)

        # Resolve grid factor
        grid_factor = self._get_grid_factor(country_code, egrid_subregion)

        # Calculate avoided CO2e
        avoided_co2e = _q(avoided_kwh * grid_factor)

        # Provenance hash
        provenance_hash = self._calculate_provenance_hash(
            {
                "type": "office_energy_savings",
                "telework_category": telework_category,
                "office_kwh_rate": str(office_rate),
                "country_code": country_code,
                "working_days": working_days,
            },
            {
                "avoided_kwh": str(avoided_kwh),
                "avoided_co2e": str(avoided_co2e),
                "annual_wfh_days": str(annual_wfh_days),
            },
        )

        duration = time.monotonic() - start_time

        logger.debug(
            "Office energy savings: category=%s, wfh_days=%s, "
            "avoided_kwh=%s, avoided_co2e=%s kg, duration=%.4fs",
            telework_category,
            annual_wfh_days,
            avoided_kwh,
            avoided_co2e,
            duration,
        )

        return {
            "avoided_office_kwh": avoided_kwh,
            "avoided_office_co2e_kg": avoided_co2e,
            "office_kwh_per_employee_per_day": office_rate,
            "annual_wfh_days": annual_wfh_days,
            "grid_factor_used": grid_factor,
            "telework_category": telework_category,
            "country_code": country_code.upper(),
            "method": "office_energy_savings",
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "provenance_hash": provenance_hash,
        }

    # ==================================================================
    # 6. calculate_batch - Batch processing
    # ==================================================================

    def calculate_batch(
        self,
        items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Process multiple telework calculations in a batch.

        Each item in the list is a dictionary of keyword arguments to
        calculate_telework_emissions. Failed items are logged but do not
        halt the batch.

        Args:
            items: List of dictionaries, each containing keyword arguments
                for calculate_telework_emissions.

        Returns:
            Dictionary containing:
                - results: List of successful result dictionaries
                - errors: List of error dictionaries with index and message
                - total_items: Total items in batch
                - successful: Count of successful calculations
                - failed: Count of failed calculations
                - total_co2e_kg: Aggregate total across all successful items
                - provenance_hash: SHA-256 hash of the batch

        Raises:
            ValueError: If items list exceeds _MAX_BATCH_SIZE.

        Example:
            >>> batch_result = engine.calculate_batch([
            ...     {"telework_category": "full_remote", "country_code": "US"},
            ...     {"telework_category": "hybrid_3day", "country_code": "GB"},
            ... ])
            >>> assert batch_result["successful"] == 2
        """
        if len(items) > _MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(items)} exceeds maximum {_MAX_BATCH_SIZE}"
            )

        start_time = time.monotonic()
        results: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        total_co2e = _ZERO

        for idx, item in enumerate(items):
            try:
                result = self.calculate_telework_emissions(**item)
                results.append(result)
                total_co2e = _q(total_co2e + result["total_co2e_kg"])
            except Exception as exc:
                logger.warning(
                    "Batch telework item %d failed: %s", idx, str(exc)
                )
                errors.append({
                    "index": idx,
                    "error": str(exc),
                    "input": {
                        k: str(v) if isinstance(v, Decimal) else v
                        for k, v in item.items()
                    },
                })

        # Batch provenance hash
        provenance_hash = self._calculate_provenance_hash(
            {
                "type": "batch",
                "total_items": len(items),
                "successful": len(results),
                "failed": len(errors),
            },
            {
                "total_co2e_kg": str(total_co2e),
            },
        )

        duration = time.monotonic() - start_time

        logger.info(
            "Batch telework calculation complete: total=%d, success=%d, "
            "errors=%d, total_co2e=%s kg, duration=%.4fs",
            len(items),
            len(results),
            len(errors),
            total_co2e,
            duration,
        )

        return {
            "results": results,
            "errors": errors,
            "total_items": len(items),
            "successful": len(results),
            "failed": len(errors),
            "total_co2e_kg": total_co2e,
            "method": "batch_telework",
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "provenance_hash": provenance_hash,
        }

    # ==================================================================
    # 7. compare_telework_scenarios - Scenario comparison
    # ==================================================================

    def compare_telework_scenarios(
        self,
        country_code: str = "US",
        egrid_subregion: Optional[str] = None,
        climate_zone: str = "temperate",
        heating_fuel: str = "natural_gas",
        working_days: int = 240,
        include_equipment: bool = False,
        equipment_list: Optional[List[str]] = None,
        work_fraction: Decimal = Decimal("0.10"),
        working_hours: Decimal = Decimal("8.0"),
    ) -> Dict[str, Any]:
        """
        Compare all telework categories for a given configuration.

        Calculates emissions for each standard telework category
        (full_remote through office_based) using the same parameters,
        then ranks them by total CO2e.

        Args:
            country_code: ISO 3166-1 alpha-2 country code. Defaults to 'US'.
            egrid_subregion: Optional US eGRID subregion code.
            climate_zone: Climate zone. Defaults to 'temperate'.
            heating_fuel: Heating fuel type. Defaults to 'natural_gas'.
            working_days: Total annual working days. Defaults to 240.
            include_equipment: Whether to include equipment emissions.
            equipment_list: List of equipment type keys.
            work_fraction: Home energy work fraction. Defaults to 0.10.
            working_hours: Daily working hours. Defaults to 8.0.

        Returns:
            Dictionary containing:
                - scenarios: List of dicts, one per category, sorted by
                    total_co2e_kg ascending (lowest emissions first)
                - highest_co2e: Category with highest emissions
                - lowest_co2e: Category with lowest non-zero emissions
                - country_code: Country code used
                - climate_zone: Climate zone used
                - provenance_hash: SHA-256 hash

        Example:
            >>> comparison = engine.compare_telework_scenarios(
            ...     country_code="US",
            ...     climate_zone="temperate",
            ... )
            >>> assert len(comparison["scenarios"]) == 6
            >>> assert comparison["scenarios"][0]["total_co2e_kg"] <= \\
            ...     comparison["scenarios"][-1]["total_co2e_kg"]
        """
        start_time = time.monotonic()
        scenarios: List[Dict[str, Any]] = []

        for category in TELEWORK_WFH_DAYS:
            try:
                result = self.calculate_telework_emissions(
                    telework_category=category,
                    country_code=country_code,
                    egrid_subregion=egrid_subregion,
                    climate_zone=climate_zone,
                    heating_fuel=heating_fuel,
                    working_days=working_days,
                    include_equipment=include_equipment,
                    equipment_list=equipment_list,
                    work_fraction=work_fraction,
                    working_hours=working_hours,
                )
                scenarios.append({
                    "telework_category": category,
                    "wfh_days_per_week": TELEWORK_WFH_DAYS[category],
                    "total_co2e_kg": result["total_co2e_kg"],
                    "electricity_co2e_kg": result["electricity_co2e_kg"],
                    "heating_co2e_kg": result["heating_co2e_kg"],
                    "cooling_co2e_kg": result["cooling_co2e_kg"],
                    "equipment_co2e_kg": result["equipment_co2e_kg"],
                    "annual_wfh_days": result["annual_wfh_days"],
                })
            except Exception as exc:
                logger.warning(
                    "Scenario comparison: category '%s' failed: %s",
                    category,
                    str(exc),
                )

        # Sort by total_co2e ascending
        scenarios.sort(key=lambda s: s["total_co2e_kg"])

        # Identify highest and lowest (excluding zero-emission office_based)
        non_zero_scenarios = [
            s for s in scenarios if s["total_co2e_kg"] > _ZERO
        ]
        highest = scenarios[-1]["telework_category"] if scenarios else None
        lowest = (
            non_zero_scenarios[0]["telework_category"]
            if non_zero_scenarios
            else None
        )

        # Provenance hash
        provenance_hash = self._calculate_provenance_hash(
            {
                "type": "scenario_comparison",
                "country_code": country_code,
                "climate_zone": climate_zone,
                "heating_fuel": heating_fuel,
                "num_scenarios": len(scenarios),
            },
            {
                "highest": highest,
                "lowest": lowest,
            },
        )

        duration = time.monotonic() - start_time

        logger.info(
            "Scenario comparison complete: %d scenarios, country=%s, "
            "zone=%s, highest=%s, lowest=%s, duration=%.4fs",
            len(scenarios),
            country_code,
            climate_zone,
            highest,
            lowest,
            duration,
        )

        return {
            "scenarios": scenarios,
            "highest_co2e": highest,
            "lowest_co2e": lowest,
            "country_code": country_code.upper(),
            "climate_zone": climate_zone,
            "heating_fuel": heating_fuel,
            "method": "scenario_comparison",
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "provenance_hash": provenance_hash,
        }

    # ==================================================================
    # 8. estimate_net_impact - Net impact vs avoided commute
    # ==================================================================

    def estimate_net_impact(
        self,
        telework_category: str,
        commute_mode: str = "car_average",
        commute_distance_km: Decimal = Decimal("15.0"),
        country_code: str = "US",
        egrid_subregion: Optional[str] = None,
        climate_zone: str = "temperate",
        heating_fuel: str = "natural_gas",
        working_days: int = 240,
        include_equipment: bool = False,
        equipment_list: Optional[List[str]] = None,
        work_fraction: Decimal = Decimal("0.10"),
        working_hours: Decimal = Decimal("8.0"),
    ) -> Dict[str, Any]:
        """
        Estimate net impact of telework vs avoided commute emissions.

        Calculates both telework (home office) emissions and the commute
        emissions that would be avoided, showing whether WFH results in
        a net carbon saving or net carbon increase.

        Avoided commute formula:
            round_trip_km = commute_distance_km * 2
            commute_ef = _COMMUTE_MODE_EFS[commute_mode]
            avoided_per_day = round_trip_km * commute_ef
            annual_avoided = avoided_per_day * annual_wfh_days

        Net impact:
            net = telework_co2e - avoided_commute_co2e
            If net < 0: WFH saves carbon
            If net > 0: WFH increases carbon

        Args:
            telework_category: Telework pattern key from TELEWORK_WFH_DAYS.
            commute_mode: Commute mode for avoided emissions. Must be a key
                in _COMMUTE_MODE_EFS. Defaults to 'car_average'.
            commute_distance_km: One-way commute distance in km.
                Defaults to Decimal('15.0').
            country_code: ISO 3166-1 alpha-2 country code. Defaults to 'US'.
            egrid_subregion: Optional US eGRID subregion code.
            climate_zone: Climate zone. Defaults to 'temperate'.
            heating_fuel: Heating fuel type. Defaults to 'natural_gas'.
            working_days: Annual working days. Defaults to 240.
            include_equipment: Whether to include equipment emissions.
            equipment_list: List of equipment type keys.
            work_fraction: Home energy work fraction. Defaults to 0.10.
            working_hours: Daily working hours. Defaults to 8.0.

        Returns:
            Dictionary containing:
                - telework_co2e_kg: Gross telework emissions (kg CO2e/year)
                - avoided_commute_co2e_kg: Avoided commute emissions (kg CO2e/year)
                - net_co2e_kg: Net impact (telework - avoided)
                - is_carbon_saving: True if net < 0
                - carbon_saving_kg: Absolute saving (positive if saving)
                - carbon_saving_pct: Percent reduction vs commute
                - commute_mode: Mode used for avoided calculation
                - commute_distance_km: One-way distance
                - commute_ef_per_km: Emission factor per km
                - telework_details: Full telework result
                - provenance_hash: SHA-256 hash

        Raises:
            ValueError: If commute_mode is not in _COMMUTE_MODE_EFS or
                commute_distance_km <= 0.

        Example:
            >>> result = engine.estimate_net_impact(
            ...     telework_category="full_remote",
            ...     commute_mode="car_petrol",
            ...     commute_distance_km=Decimal("25.0"),
            ...     country_code="US",
            ... )
            >>> assert result["is_carbon_saving"] is True
        """
        start_time = time.monotonic()

        # Validate commute inputs
        if commute_mode not in _COMMUTE_MODE_EFS:
            raise ValueError(
                f"Unknown commute_mode '{commute_mode}'. "
                f"Available: {list(_COMMUTE_MODE_EFS.keys())}"
            )
        distance_dec = Decimal(str(commute_distance_km))
        if distance_dec <= _ZERO:
            raise ValueError(
                f"commute_distance_km must be > 0, got {commute_distance_km}"
            )

        # Calculate telework emissions
        telework_result = self.calculate_telework_emissions(
            telework_category=telework_category,
            country_code=country_code,
            egrid_subregion=egrid_subregion,
            climate_zone=climate_zone,
            heating_fuel=heating_fuel,
            working_days=working_days,
            include_equipment=include_equipment,
            equipment_list=equipment_list,
            work_fraction=work_fraction,
            working_hours=working_hours,
        )

        telework_co2e = telework_result["total_co2e_kg"]
        annual_wfh_days = telework_result["annual_wfh_days"]

        # Calculate avoided commute emissions
        commute_ef = _COMMUTE_MODE_EFS[commute_mode]
        round_trip_km = _q(distance_dec * Decimal("2"))
        avoided_per_day = _q(round_trip_km * commute_ef)
        avoided_commute_co2e = _q(avoided_per_day * annual_wfh_days)

        # Calculate net impact
        net_co2e = _q(telework_co2e - avoided_commute_co2e)
        is_carbon_saving = net_co2e < _ZERO
        carbon_saving_kg = _q(avoided_commute_co2e - telework_co2e)

        # Calculate percentage saving relative to commute
        if avoided_commute_co2e > _ZERO:
            carbon_saving_pct = _q(
                (carbon_saving_kg / avoided_commute_co2e)
                * Decimal("100")
            )
        else:
            carbon_saving_pct = _ZERO

        # Provenance hash
        provenance_hash = self._calculate_provenance_hash(
            {
                "type": "net_impact",
                "telework_category": telework_category,
                "commute_mode": commute_mode,
                "commute_distance_km": str(distance_dec),
                "country_code": country_code,
                "climate_zone": climate_zone,
            },
            {
                "telework_co2e": str(telework_co2e),
                "avoided_commute_co2e": str(avoided_commute_co2e),
                "net_co2e": str(net_co2e),
                "is_carbon_saving": str(is_carbon_saving),
            },
        )

        duration = time.monotonic() - start_time

        logger.debug(
            "Net impact estimate: category=%s, mode=%s, distance=%s km, "
            "telework=%s kg, avoided=%s kg, net=%s kg, saving=%s, "
            "duration=%.4fs",
            telework_category,
            commute_mode,
            distance_dec,
            telework_co2e,
            avoided_commute_co2e,
            net_co2e,
            is_carbon_saving,
            duration,
        )

        return {
            "telework_co2e_kg": telework_co2e,
            "avoided_commute_co2e_kg": avoided_commute_co2e,
            "net_co2e_kg": net_co2e,
            "is_carbon_saving": is_carbon_saving,
            "carbon_saving_kg": carbon_saving_kg,
            "carbon_saving_pct": carbon_saving_pct,
            "commute_mode": commute_mode,
            "commute_distance_km": distance_dec,
            "commute_ef_per_km": commute_ef,
            "annual_wfh_days": annual_wfh_days,
            "telework_details": telework_result,
            "method": "net_impact_estimate",
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "provenance_hash": provenance_hash,
        }

    # ==================================================================
    # Internal: Electricity Emissions Calculation
    # ==================================================================

    def _calculate_electricity_emissions(
        self,
        annual_wfh_days: Decimal,
        climate_zone: str,
        grid_factor: Decimal,
        work_fraction: Decimal,
        working_hours: Decimal,
        custom_electricity_kwh: Optional[Decimal],
    ) -> Dict[str, Decimal]:
        """
        Calculate annual electricity emissions from home office base load.

        The base load covers lighting, computer, peripherals, and other
        typical office equipment electricity consumption. The daily kWh
        is adjusted by the work fraction (proportion of home used for
        work) and the working hours fraction of a 24-hour day.

        Formula:
            daily_elec = (custom or default) * work_fraction * (hours / 24)
            annual_kwh = daily_elec * annual_wfh_days
            co2e = annual_kwh * grid_factor

        Args:
            annual_wfh_days: Annual WFH days as Decimal.
            climate_zone: Climate zone key.
            grid_factor: Grid emission factor (kg CO2e/kWh).
            work_fraction: Proportion of home energy attributable to work.
            working_hours: Daily working hours.
            custom_electricity_kwh: Optional override for daily kWh.

        Returns:
            Dict with 'co2e' (kg CO2e) and 'annual_kwh'.
        """
        # Determine daily electricity consumption
        if custom_electricity_kwh is not None:
            daily_electricity = Decimal(str(custom_electricity_kwh))
        else:
            daily_electricity = CLIMATE_ZONE_DEFAULTS[climate_zone]["electricity"]

        # Apply work fraction and working hours adjustment
        hours_fraction = _q(Decimal(str(working_hours)) / _TWENTY_FOUR)
        adjusted_daily = _q(daily_electricity * work_fraction * hours_fraction)

        # Calculate annual kWh
        annual_kwh = _q(adjusted_daily * annual_wfh_days)

        # Calculate CO2e
        co2e = _q(annual_kwh * grid_factor)

        logger.debug(
            "Electricity: daily_base=%s kWh, adjusted=%s kWh, "
            "annual=%s kWh, co2e=%s kg",
            daily_electricity,
            adjusted_daily,
            annual_kwh,
            co2e,
        )

        return {"co2e": co2e, "annual_kwh": annual_kwh}

    # ==================================================================
    # Internal: Heating Emissions Calculation
    # ==================================================================

    def _calculate_heating_emissions(
        self,
        annual_wfh_days: Decimal,
        climate_zone: str,
        heating_fuel: str,
        grid_factor: Decimal,
        work_fraction: Decimal,
        custom_heating_kwh: Optional[Decimal],
    ) -> Dict[str, Decimal]:
        """
        Calculate annual heating emissions from home office.

        Heating emissions are seasonal: only a fraction of the year
        requires heating, determined by the climate zone's heating
        fraction. For electric heat pumps, the heating energy is
        divided by the COP and then multiplied by the grid factor.
        For fuel-based heating, the heating energy is multiplied
        by the fuel emission factor directly.

        Formula:
            daily_heat = (custom or default) * work_fraction
            annual_kwh = daily_heat * annual_wfh_days * heating_fraction
            If heat pump: co2e = (annual_kwh / COP) * grid_factor
            Else: co2e = annual_kwh * fuel_factor

        Args:
            annual_wfh_days: Annual WFH days as Decimal.
            climate_zone: Climate zone key.
            heating_fuel: Heating fuel type key.
            grid_factor: Grid emission factor (kg CO2e/kWh).
            work_fraction: Proportion of home energy attributable to work.
            custom_heating_kwh: Optional override for daily heating kWh.

        Returns:
            Dict with 'co2e' (kg CO2e) and 'annual_kwh'.
        """
        # Determine daily heating consumption
        if custom_heating_kwh is not None:
            daily_heating = Decimal(str(custom_heating_kwh))
        else:
            daily_heating = CLIMATE_ZONE_DEFAULTS[climate_zone]["heating"]

        # Apply work fraction
        adjusted_daily = _q(daily_heating * work_fraction)

        # Get seasonal heating fraction
        heating_fraction = SEASONAL_FACTORS[climate_zone]["heating_fraction"]

        # Calculate annual heating kWh (seasonal)
        annual_kwh = _q(adjusted_daily * annual_wfh_days * heating_fraction)

        # Calculate CO2e based on fuel type
        if heating_fuel == "electric_heat_pump":
            cop = HEAT_PUMP_COP[climate_zone]
            heating_electricity_kwh = _q(annual_kwh / cop)
            co2e = _q(heating_electricity_kwh * grid_factor)
            logger.debug(
                "Heating (heat pump): daily=%s kWh, annual=%s kWh, "
                "COP=%s, elec_kwh=%s, co2e=%s kg",
                daily_heating,
                annual_kwh,
                cop,
                heating_electricity_kwh,
                co2e,
            )
        else:
            fuel_factor = HEATING_FUEL_FACTORS[heating_fuel]
            co2e = _q(annual_kwh * fuel_factor)
            logger.debug(
                "Heating (%s): daily=%s kWh, annual=%s kWh, "
                "fuel_ef=%s, co2e=%s kg",
                heating_fuel,
                daily_heating,
                annual_kwh,
                fuel_factor,
                co2e,
            )

        return {"co2e": co2e, "annual_kwh": annual_kwh}

    # ==================================================================
    # Internal: Cooling Emissions Calculation
    # ==================================================================

    def _calculate_cooling_emissions(
        self,
        annual_wfh_days: Decimal,
        climate_zone: str,
        grid_factor: Decimal,
        work_fraction: Decimal,
        custom_cooling_kwh: Optional[Decimal],
    ) -> Dict[str, Decimal]:
        """
        Calculate annual cooling emissions from home office.

        Cooling is assumed to be electric (air conditioning), so
        emissions are calculated using the grid emission factor.
        Cooling is seasonal, with the fraction of the year requiring
        cooling determined by the climate zone.

        Formula:
            daily_cool = (custom or default) * work_fraction
            annual_kwh = daily_cool * annual_wfh_days * cooling_fraction
            co2e = annual_kwh * grid_factor

        Args:
            annual_wfh_days: Annual WFH days as Decimal.
            climate_zone: Climate zone key.
            grid_factor: Grid emission factor (kg CO2e/kWh).
            work_fraction: Proportion of home energy attributable to work.
            custom_cooling_kwh: Optional override for daily cooling kWh.

        Returns:
            Dict with 'co2e' (kg CO2e) and 'annual_kwh'.
        """
        # Determine daily cooling consumption
        if custom_cooling_kwh is not None:
            daily_cooling = Decimal(str(custom_cooling_kwh))
        else:
            daily_cooling = CLIMATE_ZONE_DEFAULTS[climate_zone]["cooling"]

        # Apply work fraction
        adjusted_daily = _q(daily_cooling * work_fraction)

        # Get seasonal cooling fraction
        cooling_fraction = SEASONAL_FACTORS[climate_zone]["cooling_fraction"]

        # Calculate annual cooling kWh (seasonal)
        annual_kwh = _q(adjusted_daily * annual_wfh_days * cooling_fraction)

        # Calculate CO2e (electric cooling uses grid factor)
        co2e = _q(annual_kwh * grid_factor)

        logger.debug(
            "Cooling: daily=%s kWh, adjusted=%s kWh, "
            "fraction=%s, annual=%s kWh, co2e=%s kg",
            daily_cooling,
            adjusted_daily,
            cooling_fraction,
            annual_kwh,
            co2e,
        )

        return {"co2e": co2e, "annual_kwh": annual_kwh}

    # ==================================================================
    # Internal: Equipment Emissions Calculation
    # ==================================================================

    def _calculate_equipment_emissions(
        self,
        equipment_list: List[str],
        annual_wfh_days: Decimal,
    ) -> Decimal:
        """
        Calculate lifecycle equipment emissions amortized over WFH days.

        Equipment lifecycle emissions represent the embodied carbon in
        home office equipment (manufacturing, transport, disposal),
        amortized over the equipment's useful life and allocated to
        work-from-home days.

        Formula:
            equipment_co2e = SUM(EQUIPMENT_EMISSIONS[item] * annual_wfh_days)
                for each item in equipment_list

        Args:
            equipment_list: List of equipment type keys (e.g. 'laptop',
                'monitor'). Invalid keys are logged and skipped.
            annual_wfh_days: Annual WFH days as Decimal.

        Returns:
            Total equipment emissions in kg CO2e.
        """
        total = _ZERO

        for item in equipment_list:
            item_lower = item.lower().strip()
            ef = EQUIPMENT_EMISSIONS.get(item_lower)
            if ef is None:
                logger.warning(
                    "Unknown equipment type '%s' - skipping. "
                    "Available: %s",
                    item,
                    list(EQUIPMENT_EMISSIONS.keys()),
                )
                continue

            item_co2e = _q(ef * annual_wfh_days)
            total = _q(total + item_co2e)

            logger.debug(
                "Equipment '%s': ef=%s kg/day, wfh_days=%s, co2e=%s kg",
                item_lower,
                ef,
                annual_wfh_days,
                item_co2e,
            )

        return total

    # ==================================================================
    # Internal: Grid Factor Resolution
    # ==================================================================

    def _get_grid_factor(
        self,
        country_code: str,
        egrid_subregion: Optional[str] = None,
    ) -> Decimal:
        """
        Resolve the electricity grid emission factor.

        If country_code is 'US' and an eGRID subregion is provided,
        uses the subregional factor for higher accuracy. Otherwise
        uses the national grid factor. Falls back to US factor if
        the country code is not found.

        Args:
            country_code: ISO 3166-1 alpha-2 country code (case-insensitive).
            egrid_subregion: Optional US eGRID subregion code.

        Returns:
            Grid emission factor in kg CO2e per kWh.

        Raises:
            ValueError: If eGRID subregion is provided but not found.
        """
        code = country_code.upper()

        # Check for US eGRID subregion
        if egrid_subregion is not None and code == "US":
            subregion = egrid_subregion.upper()
            if subregion in US_EGRID_FACTORS:
                logger.debug(
                    "Using eGRID subregion '%s' factor: %s kg CO2e/kWh",
                    subregion,
                    US_EGRID_FACTORS[subregion],
                )
                return US_EGRID_FACTORS[subregion]
            else:
                raise ValueError(
                    f"Unknown eGRID subregion '{egrid_subregion}'. "
                    f"Available: {list(US_EGRID_FACTORS.keys())}"
                )

        # Use country grid factor
        factor = COUNTRY_GRID_FACTORS.get(code)
        if factor is not None:
            logger.debug(
                "Using country '%s' grid factor: %s kg CO2e/kWh",
                code,
                factor,
            )
            return factor

        # Fallback to US as default
        logger.info(
            "Country '%s' not found in COUNTRY_GRID_FACTORS, "
            "falling back to US factor (%s kg CO2e/kWh)",
            code,
            COUNTRY_GRID_FACTORS["US"],
        )
        return COUNTRY_GRID_FACTORS["US"]

    # ==================================================================
    # Internal: Heating Factor Resolution
    # ==================================================================

    def _get_heating_factor(self, fuel_type: str) -> Decimal:
        """
        Resolve the heating fuel emission factor.

        Args:
            fuel_type: Heating fuel type key (must be in HEATING_FUEL_FACTORS).

        Returns:
            Heating fuel emission factor in kg CO2e per kWh.

        Raises:
            ValueError: If fuel_type is not found.
        """
        factor = HEATING_FUEL_FACTORS.get(fuel_type)
        if factor is None:
            raise ValueError(
                f"Unknown heating fuel type '{fuel_type}'. "
                f"Available: {list(HEATING_FUEL_FACTORS.keys())}"
            )
        return factor

    # ==================================================================
    # Internal: Climate Zone Defaults
    # ==================================================================

    def _get_climate_defaults(self, climate_zone: str) -> Dict[str, Decimal]:
        """
        Get energy consumption defaults for a climate zone.

        Args:
            climate_zone: Climate zone key.

        Returns:
            Dict with 'electricity', 'heating', 'cooling' in kWh/day.

        Raises:
            ValueError: If climate_zone is not found.
        """
        defaults = CLIMATE_ZONE_DEFAULTS.get(climate_zone)
        if defaults is None:
            raise ValueError(
                f"Unknown climate zone '{climate_zone}'. "
                f"Available: {list(CLIMATE_ZONE_DEFAULTS.keys())}"
            )
        return defaults

    # ==================================================================
    # Internal: Annual WFH Days Calculation
    # ==================================================================

    def _calculate_annual_wfh_days(
        self,
        wfh_days_per_week: int,
        working_days: int,
    ) -> Decimal:
        """
        Calculate annual WFH days from weekly pattern and total working days.

        Formula:
            annual_wfh_days = (wfh_days_per_week / 5) * working_days

        Args:
            wfh_days_per_week: WFH days per week (0-5).
            working_days: Total annual working days.

        Returns:
            Annual WFH days as Decimal (quantized to 8 dp).
        """
        wfh_fraction = _q(Decimal(str(wfh_days_per_week)) / _FIVE)
        annual = _q(wfh_fraction * Decimal(str(working_days)))
        return annual

    # ==================================================================
    # Internal: WFH Days to Category Mapping
    # ==================================================================

    def _wfh_days_to_category(self, wfh_days: int) -> str:
        """
        Map a WFH days-per-week count to a standard telework category.

        Args:
            wfh_days: Number of WFH days per week (0-5).

        Returns:
            Telework category string.
        """
        mapping: Dict[int, str] = {
            5: "full_remote",
            4: "hybrid_4day",
            3: "hybrid_3day",
            2: "hybrid_2day",
            1: "hybrid_1day",
            0: "office_based",
        }
        return mapping.get(wfh_days, "office_based")

    # ==================================================================
    # Internal: Input Validation
    # ==================================================================

    def _validate_inputs(self, params: Dict[str, Any]) -> List[str]:
        """
        Validate all input parameters for calculate_telework_emissions.

        Performs comprehensive validation of every parameter, collecting
        all errors into a list rather than failing on the first error.

        Args:
            params: Dictionary of all input parameters.

        Returns:
            List of error message strings. Empty if all inputs are valid.
        """
        errors: List[str] = []

        # Validate telework_category
        category = params.get("telework_category")
        if category not in TELEWORK_WFH_DAYS:
            errors.append(
                f"Unknown telework_category '{category}'. "
                f"Available: {list(TELEWORK_WFH_DAYS.keys())}"
            )

        # Validate country_code
        country = params.get("country_code", "")
        if not isinstance(country, str) or len(country) < 2:
            errors.append(
                f"country_code must be a 2-letter ISO code, got '{country}'"
            )

        # Validate egrid_subregion (if provided)
        egrid = params.get("egrid_subregion")
        if egrid is not None:
            code_upper = country.upper() if isinstance(country, str) else ""
            if code_upper == "US" and egrid.upper() not in US_EGRID_FACTORS:
                errors.append(
                    f"Unknown eGRID subregion '{egrid}'. "
                    f"Available: {list(US_EGRID_FACTORS.keys())}"
                )
            elif code_upper != "US":
                errors.append(
                    f"eGRID subregion is only valid for US, "
                    f"got country_code='{country}'"
                )

        # Validate climate_zone
        zone = params.get("climate_zone")
        if zone not in CLIMATE_ZONE_DEFAULTS:
            errors.append(
                f"Unknown climate_zone '{zone}'. "
                f"Available: {list(CLIMATE_ZONE_DEFAULTS.keys())}"
            )

        # Validate heating_fuel
        fuel = params.get("heating_fuel")
        if fuel not in HEATING_FUEL_FACTORS:
            errors.append(
                f"Unknown heating_fuel '{fuel}'. "
                f"Available: {list(HEATING_FUEL_FACTORS.keys())}"
            )

        # Validate working_days
        days = params.get("working_days")
        if not isinstance(days, int) or days <= 0:
            errors.append(
                f"working_days must be a positive integer, got {days}"
            )
        elif days > 366:
            errors.append(
                f"working_days cannot exceed 366, got {days}"
            )

        # Validate work_fraction
        fraction = params.get("work_fraction")
        if fraction is not None:
            try:
                frac_dec = Decimal(str(fraction))
                if frac_dec <= _ZERO or frac_dec > _ONE:
                    errors.append(
                        f"work_fraction must be > 0 and <= 1, got {fraction}"
                    )
            except (InvalidOperation, ValueError):
                errors.append(
                    f"work_fraction must be a valid number, got {fraction}"
                )

        # Validate working_hours
        hours = params.get("working_hours")
        if hours is not None:
            try:
                hrs_dec = Decimal(str(hours))
                if hrs_dec <= _ZERO or hrs_dec > _TWENTY_FOUR:
                    errors.append(
                        f"working_hours must be > 0 and <= 24, got {hours}"
                    )
            except (InvalidOperation, ValueError):
                errors.append(
                    f"working_hours must be a valid number, got {hours}"
                )

        # Validate custom energy overrides
        for key in ("custom_electricity_kwh", "custom_heating_kwh",
                     "custom_cooling_kwh"):
            val = params.get(key)
            if val is not None:
                try:
                    val_dec = Decimal(str(val))
                    if val_dec < _ZERO:
                        errors.append(
                            f"{key} must be >= 0, got {val}"
                        )
                except (InvalidOperation, ValueError):
                    errors.append(
                        f"{key} must be a valid number, got {val}"
                    )

        # Validate equipment list
        include_equipment = params.get("include_equipment", False)
        equipment_list = params.get("equipment_list")
        if include_equipment and equipment_list is not None:
            if not isinstance(equipment_list, list):
                errors.append(
                    f"equipment_list must be a list, "
                    f"got {type(equipment_list).__name__}"
                )
            else:
                for item in equipment_list:
                    if not isinstance(item, str):
                        errors.append(
                            f"equipment_list items must be strings, "
                            f"got {type(item).__name__}"
                        )

        return errors

    # ==================================================================
    # Internal: Build Zero Result
    # ==================================================================

    def _build_zero_result(
        self,
        telework_category: str,
        country_code: str,
        climate_zone: str,
        heating_fuel: str,
    ) -> Dict[str, Any]:
        """
        Build a zero-emission result for office_based workers.

        When a worker has 0 WFH days, all telework emissions are zero.
        This method constructs the standard result dictionary with all
        emissions set to zero.

        Args:
            telework_category: Telework category (should be 'office_based').
            country_code: Country code for the result.
            climate_zone: Climate zone for the result.
            heating_fuel: Heating fuel for the result.

        Returns:
            Standard result dictionary with all emissions at zero.
        """
        provenance_hash = self._calculate_provenance_hash(
            {
                "telework_category": telework_category,
                "country_code": country_code,
                "annual_wfh_days": "0",
            },
            {
                "total_co2e_kg": "0",
            },
        )

        return {
            "total_co2e_kg": _ZERO,
            "electricity_co2e_kg": _ZERO,
            "heating_co2e_kg": _ZERO,
            "cooling_co2e_kg": _ZERO,
            "equipment_co2e_kg": _ZERO,
            "annual_wfh_days": _ZERO,
            "annual_electricity_kwh": _ZERO,
            "annual_heating_kwh": _ZERO,
            "annual_cooling_kwh": _ZERO,
            "grid_factor_used": _ZERO,
            "heating_fuel": heating_fuel,
            "climate_zone": climate_zone,
            "country_code": country_code.upper(),
            "telework_category": telework_category,
            "method": "telework_energy_based",
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "provenance_hash": provenance_hash,
        }

    # ==================================================================
    # Internal: Provenance Hash
    # ==================================================================

    def _calculate_provenance_hash(
        self,
        input_data: Dict[str, Any],
        result_data: Dict[str, Any],
    ) -> str:
        """
        Calculate SHA-256 provenance hash from input and result data.

        Creates a deterministic hash by serialising both input and result
        dictionaries to sorted JSON, concatenating them with a separator,
        and computing the SHA-256 hex digest. This provides a tamper-proof
        audit trail for every calculation.

        Args:
            input_data: Dictionary of input parameters (stringified).
            result_data: Dictionary of result values (stringified).

        Returns:
            Hexadecimal SHA-256 hash string (64 characters).
        """
        input_str = json.dumps(input_data, sort_keys=True, default=str)
        result_str = json.dumps(result_data, sort_keys=True, default=str)
        combined = f"{ENGINE_ID}|{ENGINE_VERSION}|{input_str}|{result_str}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    # ==================================================================
    # Reference Data Accessors
    # ==================================================================

    @staticmethod
    def get_supported_countries() -> List[str]:
        """
        Return list of supported country codes with grid emission factors.

        Returns:
            Sorted list of ISO 3166-1 alpha-2 country codes.

        Example:
            >>> countries = TeleworkCalculatorEngine.get_supported_countries()
            >>> "US" in countries
            True
            >>> "GB" in countries
            True
        """
        return sorted(COUNTRY_GRID_FACTORS.keys())

    @staticmethod
    def get_supported_egrid_subregions() -> List[str]:
        """
        Return list of supported US eGRID subregion codes.

        Returns:
            Sorted list of eGRID subregion codes.

        Example:
            >>> subregions = TeleworkCalculatorEngine.get_supported_egrid_subregions()
            >>> "CAMX" in subregions
            True
        """
        return sorted(US_EGRID_FACTORS.keys())

    @staticmethod
    def get_supported_climate_zones() -> List[str]:
        """
        Return list of supported climate zones.

        Returns:
            Sorted list of climate zone keys.

        Example:
            >>> zones = TeleworkCalculatorEngine.get_supported_climate_zones()
            >>> "temperate" in zones
            True
        """
        return sorted(CLIMATE_ZONE_DEFAULTS.keys())

    @staticmethod
    def get_supported_heating_fuels() -> List[str]:
        """
        Return list of supported heating fuel types.

        Returns:
            Sorted list of heating fuel type keys.

        Example:
            >>> fuels = TeleworkCalculatorEngine.get_supported_heating_fuels()
            >>> "natural_gas" in fuels
            True
        """
        return sorted(HEATING_FUEL_FACTORS.keys())

    @staticmethod
    def get_supported_equipment_types() -> List[str]:
        """
        Return list of supported equipment types for lifecycle emissions.

        Returns:
            Sorted list of equipment type keys.

        Example:
            >>> equipment = TeleworkCalculatorEngine.get_supported_equipment_types()
            >>> "laptop" in equipment
            True
        """
        return sorted(EQUIPMENT_EMISSIONS.keys())

    @staticmethod
    def get_supported_telework_categories() -> List[str]:
        """
        Return list of supported telework category keys.

        Returns:
            List of telework category keys with their WFH days per week.

        Example:
            >>> categories = TeleworkCalculatorEngine.get_supported_telework_categories()
            >>> "full_remote" in categories
            True
        """
        return list(TELEWORK_WFH_DAYS.keys())

    @staticmethod
    def get_supported_commute_modes() -> List[str]:
        """
        Return list of supported commute modes for net impact estimation.

        Returns:
            Sorted list of commute mode keys.

        Example:
            >>> modes = TeleworkCalculatorEngine.get_supported_commute_modes()
            >>> "car_petrol" in modes
            True
        """
        return sorted(_COMMUTE_MODE_EFS.keys())

    @staticmethod
    def get_country_grid_factors() -> Dict[str, Decimal]:
        """
        Return all country grid emission factors.

        Returns:
            Dict mapping country code to grid factor (kg CO2e/kWh).

        Example:
            >>> factors = TeleworkCalculatorEngine.get_country_grid_factors()
            >>> factors["US"]
            Decimal('0.37938')
        """
        return dict(COUNTRY_GRID_FACTORS)

    @staticmethod
    def get_egrid_factors() -> Dict[str, Decimal]:
        """
        Return all US eGRID subregional factors.

        Returns:
            Dict mapping eGRID subregion code to factor (kg CO2e/kWh).

        Example:
            >>> factors = TeleworkCalculatorEngine.get_egrid_factors()
            >>> factors["CAMX"]
            Decimal('0.22134')
        """
        return dict(US_EGRID_FACTORS)

    @staticmethod
    def get_heating_fuel_factors() -> Dict[str, Decimal]:
        """
        Return all heating fuel emission factors.

        Returns:
            Dict mapping fuel type to factor (kg CO2e/kWh).

        Example:
            >>> factors = TeleworkCalculatorEngine.get_heating_fuel_factors()
            >>> factors["natural_gas"]
            Decimal('0.18316')
        """
        return dict(HEATING_FUEL_FACTORS)

    @staticmethod
    def get_equipment_emission_factors() -> Dict[str, Decimal]:
        """
        Return all equipment lifecycle emission factors.

        Returns:
            Dict mapping equipment type to daily factor (kg CO2e/day).

        Example:
            >>> factors = TeleworkCalculatorEngine.get_equipment_emission_factors()
            >>> factors["laptop"]
            Decimal('0.04110')
        """
        return dict(EQUIPMENT_EMISSIONS)

    @staticmethod
    def get_climate_zone_defaults() -> Dict[str, Dict[str, Decimal]]:
        """
        Return all climate zone energy defaults.

        Returns:
            Dict mapping climate zone to energy defaults (kWh/day).

        Example:
            >>> defaults = TeleworkCalculatorEngine.get_climate_zone_defaults()
            >>> defaults["temperate"]["electricity"]
            Decimal('2.50')
        """
        return {
            zone: dict(values)
            for zone, values in CLIMATE_ZONE_DEFAULTS.items()
        }

    @staticmethod
    def get_seasonal_factors() -> Dict[str, Dict[str, Decimal]]:
        """
        Return all seasonal allocation factors.

        Returns:
            Dict mapping climate zone to seasonal fractions.

        Example:
            >>> factors = TeleworkCalculatorEngine.get_seasonal_factors()
            >>> factors["temperate"]["heating_fraction"]
            Decimal('0.42')
        """
        return {
            zone: dict(values)
            for zone, values in SEASONAL_FACTORS.items()
        }

    @staticmethod
    def get_heat_pump_cop_values() -> Dict[str, Decimal]:
        """
        Return all heat pump COP values by climate zone.

        Returns:
            Dict mapping climate zone to COP value.

        Example:
            >>> cops = TeleworkCalculatorEngine.get_heat_pump_cop_values()
            >>> cops["temperate"]
            Decimal('3.50')
        """
        return dict(HEAT_PUMP_COP)


# ==============================================================================
# MODULE-LEVEL SINGLETON ACCESSORS
# ==============================================================================


def get_telework_calculator() -> TeleworkCalculatorEngine:
    """
    Get the singleton TeleworkCalculatorEngine instance.

    This is the recommended way to obtain the engine instance.
    Thread-safe, lazy initialization.

    Returns:
        Singleton TeleworkCalculatorEngine instance.

    Example:
        >>> engine = get_telework_calculator()
        >>> isinstance(engine, TeleworkCalculatorEngine)
        True
    """
    return TeleworkCalculatorEngine.get_instance()


def reset_telework_calculator() -> None:
    """
    Reset the singleton TeleworkCalculatorEngine instance.

    For testing purposes only. Clears the singleton so the next
    call to get_telework_calculator() creates a fresh instance.
    """
    TeleworkCalculatorEngine.reset_instance()


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    # Engine class
    "TeleworkCalculatorEngine",
    # Module-level accessors
    "get_telework_calculator",
    "reset_telework_calculator",
    # Constants - Emission factor tables
    "COUNTRY_GRID_FACTORS",
    "US_EGRID_FACTORS",
    "HEATING_FUEL_FACTORS",
    "CLIMATE_ZONE_DEFAULTS",
    "HEAT_PUMP_COP",
    "EQUIPMENT_EMISSIONS",
    "TELEWORK_WFH_DAYS",
    "SEASONAL_FACTORS",
    # Constants - Engine metadata
    "ENGINE_ID",
    "ENGINE_VERSION",
]
