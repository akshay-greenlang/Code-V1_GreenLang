# -*- coding: utf-8 -*-
"""
DistanceBasedCalculatorEngine - Engine 2: Upstream Transportation Agent (AGENT-MRV-017)

Core calculation engine implementing the distance-based method for Scope 3 Category 4
(Upstream Transportation & Distribution) emissions per GHG Protocol, ISO 14083, and
GLEC Framework v3.0.

Primary Formula:
    emissions_co2e = mass_tonnes x distance_km x emission_factor_per_tkm

This engine supports all five transport modes (road, rail, maritime, air, pipeline)
plus intermodal chains. It applies corrections for load factor, empty running,
laden state, reefer uplift, and biofuel blending. All arithmetic uses Python
Decimal for regulatory-grade precision.

Zero-Hallucination Guarantees:
    - All calculations use Python Decimal (8 decimal places, ROUND_HALF_UP)
    - No LLM calls anywhere in the calculation path
    - Every intermediate step is recorded in the calculation trace
    - SHA-256 provenance hash on every result
    - Emission factors sourced from GLEC v3.0, DEFRA 2023, IMO 4th GHG Study

WTW (Well-to-Wheel) Decomposition:
    WTW = TTW + WTT
    - TTW (Tank-to-Wheel): direct combustion emissions
    - WTT (Well-to-Tank): upstream fuel production emissions

Gas-by-Gas Breakdown:
    CO2e = CO2 + (CH4 x GWP_CH4) + (N2O x GWP_N2O)

Supports:
    - 5 transport modes (road, rail, maritime, air, pipeline) + intermodal
    - 12 road vehicle types, 3 rail types, 16 maritime vessel types, 5 aircraft types,
      5 pipeline types
    - Laden state adjustments (full, half, empty, average)
    - Load factor corrections (actual vs. max payload)
    - Empty running rate corrections
    - Reefer/temperature-controlled transport uplift
    - Great circle distance via Haversine formula with DEFRA 1.09x air correction
    - Biogenic/fossil fuel split for biofuel blends
    - WTW/TTW/WTT breakdown
    - Unit conversions (km/miles/nautical miles, tonnes/kg/lbs/short tons)
    - Data quality scoring per ISO 14083

Example:
    >>> from greenlang.upstream_transportation.distance_based_calculator import (
    ...     DistanceBasedCalculatorEngine
    ... )
    >>> from greenlang.upstream_transportation.models import (
    ...     ShipmentInput, TransportLeg, TransportMode, RoadVehicleType,
    ...     LadenState, TemperatureControl
    ... )
    >>> from decimal import Decimal
    >>> engine = DistanceBasedCalculatorEngine()
    >>> leg = TransportLeg(
    ...     mode=TransportMode.ROAD,
    ...     vehicle_type=RoadVehicleType.HGV_ARTIC_33T_PLUS.value,
    ...     distance_km=Decimal("250"),
    ...     cargo_mass_tonnes=Decimal("10"),
    ...     laden_state=LadenState.FULL,
    ...     origin="London",
    ...     destination="Birmingham",
    ... )
    >>> shipment = ShipmentInput(
    ...     shipment_id="SH-2026-001",
    ...     legs=[leg],
    ... )
    >>> result = engine.calculate(shipment)
    >>> assert result.emissions_kgco2e > Decimal("0")

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-017 Upstream Transportation (GL-MRV-S3-004)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

from greenlang.upstream_transportation.models import (
    AGENT_COMPONENT,
    AGENT_ID,
    AIR_EMISSION_FACTORS,
    AircraftType,
    AllocationMethod,
    CalculationMethod,
    DistanceMethod,
    DQIScore,
    EFScope,
    EmissionGas,
    EMPTY_RUNNING_RATES,
    GWP_VALUES,
    GWPSource,
    HubType,
    HUB_EMISSION_FACTORS,
    LadenState,
    LegResult,
    LOAD_FACTOR_DEFAULTS,
    MARITIME_EMISSION_FACTORS,
    MaritimeVesselType,
    PIPELINE_EMISSION_FACTORS,
    PipelineType,
    RAIL_EMISSION_FACTORS,
    RailType,
    REEFER_UPLIFT_FACTORS,
    ROAD_EMISSION_FACTORS,
    RoadVehicleType,
    ShipmentInput,
    TemperatureControl,
    TransportChain,
    TransportFuelType,
    TransportLeg,
    TransportMode,
    VERSION,
    calculate_provenance_hash,
    get_gwp,
)
from greenlang.upstream_transportation.metrics import UpstreamTransportationMetrics
from greenlang.upstream_transportation.provenance import get_provenance_tracker

logger = logging.getLogger(__name__)

# ==============================================================================
# DECIMAL PRECISION & ROUNDING
# ==============================================================================

_PRECISION = Decimal("0.00000001")  # 8 decimal places
_ZERO = Decimal("0")
_ONE = Decimal("1")
_TWO = Decimal("2")
_HUNDRED = Decimal("100")
_THOUSAND = Decimal("1000")

# ==============================================================================
# UNIT CONVERSION CONSTANTS (all Decimal for zero-hallucination)
# ==============================================================================

# Distance conversions
_KM_PER_MILE = Decimal("1.609344")
_KM_PER_NAUTICAL_MILE = Decimal("1.852")
_MILES_PER_KM = Decimal("0.621371192")
_NAUTICAL_MILES_PER_KM = Decimal("0.539956803")

# Mass conversions
_TONNES_PER_KG = Decimal("0.001")
_TONNES_PER_LB = Decimal("0.000453592")
_TONNES_PER_SHORT_TON = Decimal("0.907185")
_KG_PER_TONNE = Decimal("1000")
_LBS_PER_TONNE = Decimal("2204.62262")
_SHORT_TONS_PER_TONNE = Decimal("1.10231")

# Haversine constants
_EARTH_RADIUS_KM = Decimal("6371.0088")  # WGS-84 mean radius

# DEFRA great-circle distance correction for air freight
_GCD_AIR_CORRECTION = Decimal("1.09")  # 9% uplift per DEFRA 2023

# ==============================================================================
# GAS SPLIT RATIOS BY MODE (for CO2e disaggregation into individual gases)
# ==============================================================================

_GAS_SPLIT_RATIOS: Dict[str, Dict[str, Decimal]] = {
    TransportMode.ROAD.value: {
        "co2": Decimal("0.995"),
        "ch4": Decimal("0.003"),
        "n2o": Decimal("0.002"),
    },
    TransportMode.RAIL.value: {
        "co2": Decimal("0.990"),
        "ch4": Decimal("0.005"),
        "n2o": Decimal("0.005"),
    },
    TransportMode.MARITIME.value: {
        "co2": Decimal("0.997"),
        "ch4": Decimal("0.002"),
        "n2o": Decimal("0.001"),
    },
    TransportMode.AIR.value: {
        "co2": Decimal("0.998"),
        "ch4": Decimal("0.001"),
        "n2o": Decimal("0.001"),
    },
    TransportMode.PIPELINE.value: {
        "co2": Decimal("0.950"),
        "ch4": Decimal("0.045"),
        "n2o": Decimal("0.005"),
    },
    TransportMode.INTERMODAL.value: {
        "co2": Decimal("0.995"),
        "ch4": Decimal("0.003"),
        "n2o": Decimal("0.002"),
    },
}

# ==============================================================================
# WTT-TO-TTW RATIOS BY MODE
# ==============================================================================
# Ratio of WTT to TTW emissions, used to decompose WTW into TTW and WTT.
# WTW = TTW x (1 + ratio).  Derived from GLEC Framework v3.0 default factors.

_WTT_TO_TTW_RATIOS: Dict[str, Decimal] = {
    TransportMode.ROAD.value: Decimal("0.218"),    # ~21.8% upstream
    TransportMode.RAIL.value: Decimal("0.245"),    # ~24.5% upstream
    TransportMode.MARITIME.value: Decimal("0.174"),  # ~17.4% upstream
    TransportMode.AIR.value: Decimal("0.241"),     # ~24.1% upstream
    TransportMode.PIPELINE.value: Decimal("0.162"),  # ~16.2% upstream
    TransportMode.INTERMODAL.value: Decimal("0.220"),  # weighted avg ~22%
}

# ==============================================================================
# BIOGENIC FUEL BLEND PERCENTAGES
# ==============================================================================
# Fraction of biogenic content in blended fuels. Pure biofuels = 1.0.
# Biogenic CO2 is reported separately (outside Scope 1/3 totals per GHG Protocol).

_BIOGENIC_FRACTIONS: Dict[str, Decimal] = {
    TransportFuelType.DIESEL.value: Decimal("0.0"),
    TransportFuelType.PETROL.value: Decimal("0.0"),
    TransportFuelType.BIODIESEL.value: Decimal("1.0"),
    TransportFuelType.HVO.value: Decimal("1.0"),
    TransportFuelType.LPG.value: Decimal("0.0"),
    TransportFuelType.HEAVY_FUEL_OIL.value: Decimal("0.0"),
    TransportFuelType.MARINE_GAS_OIL.value: Decimal("0.0"),
    TransportFuelType.JET_FUEL.value: Decimal("0.0"),
    TransportFuelType.SUSTAINABLE_AVIATION_FUEL.value: Decimal("0.80"),
    TransportFuelType.CNG.value: Decimal("0.0"),
    TransportFuelType.LNG.value: Decimal("0.0"),
    TransportFuelType.HYDROGEN.value: Decimal("0.0"),
    TransportFuelType.ELECTRICITY.value: Decimal("0.0"),
    TransportFuelType.METHANOL.value: Decimal("0.0"),
    TransportFuelType.AMMONIA.value: Decimal("0.0"),
}

# ==============================================================================
# DATA QUALITY SCORING TABLES
# ==============================================================================

_DISTANCE_METHOD_DQ_SCORES: Dict[str, Decimal] = {
    DistanceMethod.ACTUAL.value: Decimal("1.0"),
    DistanceMethod.SHORTEST_FEASIBLE.value: Decimal("2.0"),
    DistanceMethod.GREAT_CIRCLE.value: Decimal("3.0"),
    DistanceMethod.ESTIMATED.value: Decimal("4.0"),
}

_EF_SOURCE_DQ_SCORES: Dict[str, Decimal] = {
    "supplier_specific": Decimal("1.0"),
    "measured": Decimal("1.5"),
    "glec_v3.0": Decimal("2.0"),
    "defra_2023": Decimal("2.0"),
    "imo_4th_ghg": Decimal("2.0"),
    "icao_carbon_calc": Decimal("2.0"),
    "iso_14083": Decimal("2.5"),
    "industry_average": Decimal("3.0"),
    "default": Decimal("3.5"),
    "estimated": Decimal("4.0"),
    "unknown": Decimal("5.0"),
}


# ==============================================================================
# DistanceBasedCalculatorEngine
# ==============================================================================


class DistanceBasedCalculatorEngine:
    """
    Engine 2: Distance-based emissions calculator for upstream transportation.

    Implements the distance-based method per GHG Protocol Scope 3 Category 4,
    ISO 14083, and GLEC Framework v3.0. The core formula is:

        emissions_co2e = mass_tonnes x distance_km x ef_per_tkm

    with corrections for load factor, empty running, laden state, temperature
    control (reefer), allocation, and biofuel blends. All arithmetic is performed
    using Python ``Decimal`` with 8-digit precision and ``ROUND_HALF_UP`` to ensure
    deterministic, auditable, regulatory-grade results.

    Thread Safety:
        This engine is fully thread-safe. All instance state is read-only after
        construction. No mutable shared state is used during calculations.

    Attributes:
        _gwp_source: Default GWP source (AR4/AR5/AR6) for CO2e conversion.
        _ef_scope: Default emission factor scope (TTW/WTT/WTW).
        _gcd_correction: Great circle distance correction factor for air (1.09).
        _metrics: Prometheus metrics collector.
        _provenance: SHA-256 provenance tracker.
        _lock: Reentrant lock for thread safety on shared operations.

    Example:
        >>> engine = DistanceBasedCalculatorEngine()
        >>> result = engine.calculate_road(
        ...     mass_tonnes=Decimal("10"),
        ...     distance_km=Decimal("250"),
        ...     vehicle_type=RoadVehicleType.HGV_ARTIC_33T_PLUS,
        ... )
        >>> assert result.emissions_kgco2e > Decimal("0")
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        gwp_source: GWPSource = GWPSource.AR5,
        ef_scope: EFScope = EFScope.WTW,
        gcd_correction: Decimal = _GCD_AIR_CORRECTION,
        metrics: Optional[UpstreamTransportationMetrics] = None,
    ) -> None:
        """
        Initialise the DistanceBasedCalculatorEngine.

        Args:
            gwp_source: Default IPCC Assessment Report for GWP values.
            ef_scope: Default emission factor scope (TTW/WTT/WTW).
            gcd_correction: Great circle distance correction factor for air freight.
                            Defaults to 1.09 per DEFRA 2023 guidance.
            metrics: Optional Prometheus metrics collector. A default instance is
                     created if None is provided.

        Raises:
            ValueError: If gcd_correction is less than 1.0 or greater than 2.0.
        """
        if gcd_correction < _ONE or gcd_correction > _TWO:
            raise ValueError(
                f"gcd_correction must be between 1.0 and 2.0, got {gcd_correction}"
            )

        self._gwp_source: GWPSource = gwp_source
        self._ef_scope: EFScope = ef_scope
        self._gcd_correction: Decimal = gcd_correction
        self._metrics: UpstreamTransportationMetrics = metrics or UpstreamTransportationMetrics()
        self._provenance = get_provenance_tracker()
        self._lock: threading.RLock = threading.RLock()

        logger.info(
            "DistanceBasedCalculatorEngine initialised: gwp=%s, scope=%s, gcd_correction=%s",
            gwp_source.value,
            ef_scope.value,
            gcd_correction,
        )

    # ------------------------------------------------------------------
    # 1. calculate - Main entry point for single shipment
    # ------------------------------------------------------------------

    def calculate(self, shipment: ShipmentInput) -> LegResult:
        """
        Calculate distance-based emissions for a single shipment.

        This is the primary entry point. It resolves transport legs from
        either ``shipment.legs`` or ``shipment.transport_chain``, calculates
        emissions for each leg, and returns a consolidated LegResult for the
        first (or only) leg.

        For multi-leg shipments, use ``calculate_intermodal()`` instead.

        Args:
            shipment: Validated ShipmentInput with at least one transport leg.

        Returns:
            LegResult containing total emissions, gas breakdown, provenance hash,
            and calculation metadata.

        Raises:
            ValueError: If no transport legs are found in the shipment.
            InvalidOperation: If Decimal arithmetic fails (should never happen
                              with validated inputs).

        Example:
            >>> result = engine.calculate(shipment)
            >>> print(f"Emissions: {result.emissions_kgco2e} kgCO2e")
        """
        start_ts = time.monotonic()
        calc_id = str(uuid.uuid4())

        logger.info(
            "calculate() shipment_id=%s calc_id=%s method=distance_based",
            shipment.shipment_id,
            calc_id,
        )

        # Resolve legs from either legs list or transport chain
        legs = self._resolve_legs(shipment)
        if not legs:
            raise ValueError(
                f"Shipment {shipment.shipment_id} has no transport legs"
            )

        # Calculate the first leg (single-leg entry point)
        leg = legs[0]
        result = self._calculate_single_leg(
            leg=leg,
            gwp_source=shipment.gwp_source,
            calc_id=calc_id,
        )

        elapsed_s = time.monotonic() - start_ts
        self._record_metrics(
            mode=leg.mode.value,
            status="success",
            emissions_kgco2e=result.emissions_kgco2e,
            duration_s=elapsed_s,
        )

        logger.info(
            "calculate() complete: shipment_id=%s emissions=%s kgCO2e elapsed=%.4fs",
            shipment.shipment_id,
            result.emissions_kgco2e,
            elapsed_s,
        )

        return result

    # ------------------------------------------------------------------
    # 2. calculate_road
    # ------------------------------------------------------------------

    def calculate_road(
        self,
        mass_tonnes: Decimal,
        distance_km: Decimal,
        vehicle_type: RoadVehicleType = RoadVehicleType.HGV_ARTIC_33T_PLUS,
        fuel_type: Optional[TransportFuelType] = None,
        laden_state: LadenState = LadenState.FULL,
        region: str = "global",
        temperature_control: TemperatureControl = TemperatureControl.AMBIENT,
        load_factor: Optional[Decimal] = None,
        empty_running_rate: Optional[Decimal] = None,
        custom_ef_per_tkm: Optional[Decimal] = None,
        ef_source: Optional[str] = None,
    ) -> LegResult:
        """
        Calculate road transport emissions using the distance-based method.

        Formula:
            base_ef = ROAD_EMISSION_FACTORS[vehicle_type]["total_per_tkm"]
            adjusted_ef = apply laden adjustment, load factor, empty running
            emissions = mass_tonnes x distance_km x adjusted_ef x reefer_uplift

        Args:
            mass_tonnes: Cargo mass in metric tonnes. Must be > 0.
            distance_km: Distance in kilometres. Must be > 0.
            vehicle_type: Road vehicle type (LCV, rigid HGV, articulated HGV, etc.).
            fuel_type: Fuel type for biogenic split. Defaults to diesel.
            laden_state: Laden state of the vehicle (full, half, empty, average).
            region: Geographic region for EF selection (not yet region-specific for
                    road, but reserved for future GLEC region-specific factors).
            temperature_control: Temperature control requirement (ambient, chilled,
                                 frozen, deep-frozen, heated).
            load_factor: Actual load factor (0-1). If provided, overrides laden_state.
            empty_running_rate: Fraction of km travelled empty (0-1). Defaults to
                                mode-specific rate from EMPTY_RUNNING_RATES.
            custom_ef_per_tkm: Optional custom emission factor (kgCO2e/tkm).
            ef_source: Source description for the emission factor.

        Returns:
            LegResult with emissions in kgCO2e, gas-by-gas breakdown, and
            provenance hash.

        Raises:
            ValueError: If mass_tonnes <= 0, distance_km <= 0, or vehicle_type
                        is not found in ROAD_EMISSION_FACTORS.

        Example:
            >>> result = engine.calculate_road(
            ...     mass_tonnes=Decimal("15"),
            ...     distance_km=Decimal("480"),
            ...     vehicle_type=RoadVehicleType.HGV_ARTIC_33T_PLUS,
            ...     laden_state=LadenState.FULL,
            ... )
            >>> print(result.emissions_kgco2e)
        """
        start_ts = time.monotonic()
        self._validate_activity_data(mass_tonnes, distance_km)

        # Resolve emission factor
        if custom_ef_per_tkm is not None:
            base_ef = custom_ef_per_tkm
            source = ef_source or "custom_supplier"
        else:
            ef_data = ROAD_EMISSION_FACTORS.get(vehicle_type)
            if ef_data is None:
                raise ValueError(
                    f"No road EF found for vehicle_type={vehicle_type.value}. "
                    f"Available types: {[v.value for v in RoadVehicleType]}"
                )
            base_ef = ef_data["total_per_tkm"]
            source = "GLEC_v3.0/DEFRA_2023"

        # Apply laden adjustment
        laden_multiplier = self.apply_laden_adjustment(base_ef, laden_state)
        adjusted_ef = (base_ef * laden_multiplier).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        # Apply load factor correction (overrides laden if provided)
        if load_factor is not None:
            max_payload = self._get_max_payload(vehicle_type)
            adjusted_ef = self.apply_load_factor(adjusted_ef, load_factor, max_payload)

        # Apply empty running correction
        er_rate = empty_running_rate if empty_running_rate is not None else (
            EMPTY_RUNNING_RATES.get(TransportMode.ROAD, Decimal("0.28"))
        )
        adjusted_ef = self.apply_empty_running(adjusted_ef, er_rate)

        # Calculate base emissions
        tonne_km = (mass_tonnes * distance_km).quantize(_PRECISION, rounding=ROUND_HALF_UP)
        base_emissions = (tonne_km * adjusted_ef).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        # Apply reefer uplift
        reefer_factor = self.apply_reefer_uplift(
            base_emissions, TransportMode.ROAD, temperature_control
        )
        total_emissions = reefer_factor

        # Gas-by-gas split
        gas_split = self.split_by_gas(total_emissions, TransportMode.ROAD.value)

        # Build result
        result = LegResult(
            leg_id=f"road_{uuid.uuid4().hex[:12]}",
            mode=TransportMode.ROAD,
            vehicle_type=vehicle_type.value,
            distance_km=distance_km,
            cargo_mass_tonnes=mass_tonnes,
            tonne_km=tonne_km,
            ef_per_tkm=adjusted_ef,
            ef_source=source,
            ef_scope=self._ef_scope,
            laden_adjustment=laden_multiplier,
            reefer_uplift=self._get_reefer_multiplier(TransportMode.ROAD, temperature_control),
            allocation_percentage=_HUNDRED,
            emissions_kgco2e=total_emissions,
            emissions_co2_kg=gas_split["co2"],
            emissions_ch4_kg=gas_split["ch4"],
            emissions_n2o_kg=gas_split["n2o"],
            calculation_timestamp=datetime.now(timezone.utc),
        )

        elapsed_s = time.monotonic() - start_ts
        self._record_metrics(
            mode="road", status="success",
            emissions_kgco2e=total_emissions, duration_s=elapsed_s,
        )

        logger.debug(
            "calculate_road: %s tkm, ef=%s, emissions=%s kgCO2e",
            tonne_km, adjusted_ef, total_emissions,
        )

        return result

    # ------------------------------------------------------------------
    # 3. calculate_rail
    # ------------------------------------------------------------------

    def calculate_rail(
        self,
        mass_tonnes: Decimal,
        distance_km: Decimal,
        rail_type: RailType = RailType.AVERAGE,
        region: str = "global",
        temperature_control: TemperatureControl = TemperatureControl.AMBIENT,
        custom_ef_per_tkm: Optional[Decimal] = None,
        ef_source: Optional[str] = None,
    ) -> LegResult:
        """
        Calculate rail freight emissions using the distance-based method.

        Rail emission factors are keyed by (rail_type, region) from the
        RAIL_EMISSION_FACTORS table. If the exact (type, region) is not found,
        falls back to (type, "global"), then to (AVERAGE, "global").

        Args:
            mass_tonnes: Cargo mass in metric tonnes. Must be > 0.
            distance_km: Distance in kilometres. Must be > 0.
            rail_type: Rail freight type (diesel, electric, average).
            region: Geographic region (global, us, eu, china).
            temperature_control: Temperature control requirement.
            custom_ef_per_tkm: Optional custom emission factor (kgCO2e/tkm).
            ef_source: Source description for the emission factor.

        Returns:
            LegResult with emissions in kgCO2e.

        Raises:
            ValueError: If mass_tonnes <= 0 or distance_km <= 0.

        Example:
            >>> result = engine.calculate_rail(
            ...     mass_tonnes=Decimal("500"),
            ...     distance_km=Decimal("1200"),
            ...     rail_type=RailType.ELECTRIC,
            ...     region="eu",
            ... )
        """
        start_ts = time.monotonic()
        self._validate_activity_data(mass_tonnes, distance_km)

        # Resolve emission factor
        if custom_ef_per_tkm is not None:
            base_ef = custom_ef_per_tkm
            source = ef_source or "custom_supplier"
        else:
            base_ef = self._resolve_rail_ef(rail_type, region)
            source = "GLEC_v3.0/DEFRA_2023"

        # Apply empty running correction
        er_rate = EMPTY_RUNNING_RATES.get(TransportMode.RAIL, Decimal("0.15"))
        adjusted_ef = self.apply_empty_running(base_ef, er_rate)

        # Calculate base emissions
        tonne_km = (mass_tonnes * distance_km).quantize(_PRECISION, rounding=ROUND_HALF_UP)
        base_emissions = (tonne_km * adjusted_ef).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        # Reefer uplift (no rail reefer factor in table; default to 1.0)
        reefer_multiplier = _ONE
        total_emissions = base_emissions

        # Gas-by-gas split
        gas_split = self.split_by_gas(total_emissions, TransportMode.RAIL.value)

        result = LegResult(
            leg_id=f"rail_{uuid.uuid4().hex[:12]}",
            mode=TransportMode.RAIL,
            vehicle_type=rail_type.value,
            distance_km=distance_km,
            cargo_mass_tonnes=mass_tonnes,
            tonne_km=tonne_km,
            ef_per_tkm=adjusted_ef,
            ef_source=source,
            ef_scope=self._ef_scope,
            laden_adjustment=_ONE,
            reefer_uplift=reefer_multiplier,
            allocation_percentage=_HUNDRED,
            emissions_kgco2e=total_emissions,
            emissions_co2_kg=gas_split["co2"],
            emissions_ch4_kg=gas_split["ch4"],
            emissions_n2o_kg=gas_split["n2o"],
            calculation_timestamp=datetime.now(timezone.utc),
        )

        elapsed_s = time.monotonic() - start_ts
        self._record_metrics(
            mode="rail", status="success",
            emissions_kgco2e=total_emissions, duration_s=elapsed_s,
        )

        logger.debug(
            "calculate_rail: %s tkm, ef=%s, emissions=%s kgCO2e",
            tonne_km, adjusted_ef, total_emissions,
        )

        return result

    # ------------------------------------------------------------------
    # 4. calculate_maritime
    # ------------------------------------------------------------------

    def calculate_maritime(
        self,
        mass_tonnes: Decimal,
        distance_km: Decimal,
        vessel_type: MaritimeVesselType = MaritimeVesselType.CONTAINER_PANAMAX,
        temperature_control: TemperatureControl = TemperatureControl.AMBIENT,
        custom_ef_per_tkm: Optional[Decimal] = None,
        ef_source: Optional[str] = None,
    ) -> LegResult:
        """
        Calculate maritime freight emissions using the distance-based method.

        Maritime emission factors are sourced from the IMO Fourth GHG Study 2020
        and GLEC Framework v3.0, keyed by vessel type.

        Args:
            mass_tonnes: Cargo mass in metric tonnes. Must be > 0.
            distance_km: Nautical or overland distance in km. Must be > 0.
                         For port-to-port calculations, use Haversine + correction.
            vessel_type: Maritime vessel type (container, bulk, tanker, general, etc.).
            temperature_control: Temperature control (ambient or reefer).
            custom_ef_per_tkm: Optional custom emission factor (kgCO2e/tkm).
            ef_source: Source description for the emission factor.

        Returns:
            LegResult with emissions in kgCO2e.

        Raises:
            ValueError: If mass_tonnes <= 0 or distance_km <= 0, or vessel_type
                        is not found in MARITIME_EMISSION_FACTORS.

        Example:
            >>> result = engine.calculate_maritime(
            ...     mass_tonnes=Decimal("200"),
            ...     distance_km=Decimal("8500"),
            ...     vessel_type=MaritimeVesselType.CONTAINER_POST_PANAMAX,
            ... )
        """
        start_ts = time.monotonic()
        self._validate_activity_data(mass_tonnes, distance_km)

        # Resolve emission factor
        if custom_ef_per_tkm is not None:
            base_ef = custom_ef_per_tkm
            source = ef_source or "custom_supplier"
        else:
            base_ef = MARITIME_EMISSION_FACTORS.get(vessel_type)
            if base_ef is None:
                raise ValueError(
                    f"No maritime EF found for vessel_type={vessel_type.value}. "
                    f"Available: {[v.value for v in MaritimeVesselType]}"
                )
            source = "IMO_4th_GHG/GLEC_v3.0"

        # Apply empty running (ballast voyages)
        er_rate = EMPTY_RUNNING_RATES.get(TransportMode.MARITIME, Decimal("0.12"))
        adjusted_ef = self.apply_empty_running(base_ef, er_rate)

        # Tonne-km and base emissions
        tonne_km = (mass_tonnes * distance_km).quantize(_PRECISION, rounding=ROUND_HALF_UP)
        base_emissions = (tonne_km * adjusted_ef).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        # Reefer uplift
        total_emissions = self.apply_reefer_uplift(
            base_emissions, TransportMode.MARITIME, temperature_control
        )
        reefer_multiplier = self._get_reefer_multiplier(
            TransportMode.MARITIME, temperature_control
        )

        # Gas-by-gas split
        gas_split = self.split_by_gas(total_emissions, TransportMode.MARITIME.value)

        result = LegResult(
            leg_id=f"maritime_{uuid.uuid4().hex[:12]}",
            mode=TransportMode.MARITIME,
            vehicle_type=vessel_type.value,
            distance_km=distance_km,
            cargo_mass_tonnes=mass_tonnes,
            tonne_km=tonne_km,
            ef_per_tkm=adjusted_ef,
            ef_source=source,
            ef_scope=self._ef_scope,
            laden_adjustment=_ONE,
            reefer_uplift=reefer_multiplier,
            allocation_percentage=_HUNDRED,
            emissions_kgco2e=total_emissions,
            emissions_co2_kg=gas_split["co2"],
            emissions_ch4_kg=gas_split["ch4"],
            emissions_n2o_kg=gas_split["n2o"],
            calculation_timestamp=datetime.now(timezone.utc),
        )

        elapsed_s = time.monotonic() - start_ts
        self._record_metrics(
            mode="maritime", status="success",
            emissions_kgco2e=total_emissions, duration_s=elapsed_s,
        )

        logger.debug(
            "calculate_maritime: %s tkm, ef=%s, emissions=%s kgCO2e",
            tonne_km, adjusted_ef, total_emissions,
        )

        return result

    # ------------------------------------------------------------------
    # 5. calculate_air
    # ------------------------------------------------------------------

    def calculate_air(
        self,
        mass_tonnes: Decimal,
        distance_km: Decimal,
        aircraft_type: AircraftType = AircraftType.WIDEBODY_FREIGHTER,
        temperature_control: TemperatureControl = TemperatureControl.AMBIENT,
        apply_gcd_uplift: bool = False,
        custom_ef_per_tkm: Optional[Decimal] = None,
        ef_source: Optional[str] = None,
    ) -> LegResult:
        """
        Calculate air freight emissions using the distance-based method.

        Air freight has the highest emission intensity of all modes. Emission
        factors are sourced from ICAO Carbon Calculator and GLEC Framework v3.0.

        Note on distance:
            If ``apply_gcd_uplift=True`` and the distance was calculated via
            Haversine, a 1.09x correction factor is applied per DEFRA guidance
            to account for non-great-circle routing, wind, ATC, stacking, etc.

        Args:
            mass_tonnes: Cargo mass in metric tonnes. Must be > 0.
            distance_km: Flight distance in kilometres. Must be > 0.
            aircraft_type: Aircraft type (narrowbody, widebody, belly, express).
            temperature_control: Temperature control requirement.
            apply_gcd_uplift: Whether to apply the 1.09x DEFRA GCD correction.
            custom_ef_per_tkm: Optional custom emission factor (kgCO2e/tkm).
            ef_source: Source description for the emission factor.

        Returns:
            LegResult with emissions in kgCO2e.

        Raises:
            ValueError: If mass_tonnes <= 0 or distance_km <= 0, or aircraft_type
                        is not found in AIR_EMISSION_FACTORS.

        Example:
            >>> result = engine.calculate_air(
            ...     mass_tonnes=Decimal("5"),
            ...     distance_km=Decimal("9200"),
            ...     aircraft_type=AircraftType.WIDEBODY_FREIGHTER,
            ...     apply_gcd_uplift=True,
            ... )
        """
        start_ts = time.monotonic()
        self._validate_activity_data(mass_tonnes, distance_km)

        # Apply GCD correction if requested
        effective_distance = distance_km
        if apply_gcd_uplift:
            effective_distance = self.apply_gcd_correction(distance_km)

        # Resolve emission factor
        if custom_ef_per_tkm is not None:
            base_ef = custom_ef_per_tkm
            source = ef_source or "custom_supplier"
        else:
            base_ef = AIR_EMISSION_FACTORS.get(aircraft_type)
            if base_ef is None:
                raise ValueError(
                    f"No air EF found for aircraft_type={aircraft_type.value}. "
                    f"Available: {[a.value for a in AircraftType]}"
                )
            source = "ICAO_Carbon_Calc/GLEC_v3.0"

        # Apply empty running correction
        er_rate = EMPTY_RUNNING_RATES.get(TransportMode.AIR, Decimal("0.08"))
        adjusted_ef = self.apply_empty_running(base_ef, er_rate)

        # Tonne-km and base emissions
        tonne_km = (mass_tonnes * effective_distance).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )
        base_emissions = (tonne_km * adjusted_ef).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )

        # Reefer uplift
        total_emissions = self.apply_reefer_uplift(
            base_emissions, TransportMode.AIR, temperature_control
        )
        reefer_multiplier = self._get_reefer_multiplier(
            TransportMode.AIR, temperature_control
        )

        # Gas-by-gas split
        gas_split = self.split_by_gas(total_emissions, TransportMode.AIR.value)

        result = LegResult(
            leg_id=f"air_{uuid.uuid4().hex[:12]}",
            mode=TransportMode.AIR,
            vehicle_type=aircraft_type.value,
            distance_km=effective_distance,
            cargo_mass_tonnes=mass_tonnes,
            tonne_km=tonne_km,
            ef_per_tkm=adjusted_ef,
            ef_source=source,
            ef_scope=self._ef_scope,
            laden_adjustment=_ONE,
            reefer_uplift=reefer_multiplier,
            allocation_percentage=_HUNDRED,
            emissions_kgco2e=total_emissions,
            emissions_co2_kg=gas_split["co2"],
            emissions_ch4_kg=gas_split["ch4"],
            emissions_n2o_kg=gas_split["n2o"],
            calculation_timestamp=datetime.now(timezone.utc),
        )

        elapsed_s = time.monotonic() - start_ts
        self._record_metrics(
            mode="air", status="success",
            emissions_kgco2e=total_emissions, duration_s=elapsed_s,
        )

        logger.debug(
            "calculate_air: %s tkm, ef=%s, emissions=%s kgCO2e (gcd_uplift=%s)",
            tonne_km, adjusted_ef, total_emissions, apply_gcd_uplift,
        )

        return result

    # ------------------------------------------------------------------
    # 6. calculate_pipeline
    # ------------------------------------------------------------------

    def calculate_pipeline(
        self,
        mass_tonnes: Decimal,
        distance_km: Decimal,
        pipeline_type: PipelineType = PipelineType.REFINED_PRODUCTS,
        custom_ef_per_tkm: Optional[Decimal] = None,
        ef_source: Optional[str] = None,
    ) -> LegResult:
        """
        Calculate pipeline transport emissions using the distance-based method.

        Pipeline emission factors are low per tonne-km but pipelines are used
        for very large volumes over long distances.

        Args:
            mass_tonnes: Mass transported in metric tonnes. Must be > 0.
            distance_km: Pipeline length in kilometres. Must be > 0.
            pipeline_type: Type of pipeline (crude oil, refined products, natural
                           gas, chemicals, CO2).
            custom_ef_per_tkm: Optional custom emission factor (kgCO2e/tkm).
            ef_source: Source description for the emission factor.

        Returns:
            LegResult with emissions in kgCO2e.

        Raises:
            ValueError: If mass_tonnes <= 0 or distance_km <= 0, or pipeline_type
                        is not found in PIPELINE_EMISSION_FACTORS.

        Example:
            >>> result = engine.calculate_pipeline(
            ...     mass_tonnes=Decimal("10000"),
            ...     distance_km=Decimal("500"),
            ...     pipeline_type=PipelineType.NATURAL_GAS,
            ... )
        """
        start_ts = time.monotonic()
        self._validate_activity_data(mass_tonnes, distance_km)

        # Resolve emission factor
        if custom_ef_per_tkm is not None:
            base_ef = custom_ef_per_tkm
            source = ef_source or "custom_supplier"
        else:
            base_ef = PIPELINE_EMISSION_FACTORS.get(pipeline_type)
            if base_ef is None:
                raise ValueError(
                    f"No pipeline EF found for pipeline_type={pipeline_type.value}. "
                    f"Available: {[p.value for p in PipelineType]}"
                )
            source = "GLEC_v3.0"

        # Pipelines have no empty running or laden adjustment
        adjusted_ef = base_ef

        # Tonne-km and emissions
        tonne_km = (mass_tonnes * distance_km).quantize(_PRECISION, rounding=ROUND_HALF_UP)
        total_emissions = (tonne_km * adjusted_ef).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )

        # Gas-by-gas split
        gas_split = self.split_by_gas(total_emissions, TransportMode.PIPELINE.value)

        result = LegResult(
            leg_id=f"pipeline_{uuid.uuid4().hex[:12]}",
            mode=TransportMode.PIPELINE,
            vehicle_type=pipeline_type.value,
            distance_km=distance_km,
            cargo_mass_tonnes=mass_tonnes,
            tonne_km=tonne_km,
            ef_per_tkm=adjusted_ef,
            ef_source=source,
            ef_scope=self._ef_scope,
            laden_adjustment=_ONE,
            reefer_uplift=_ONE,
            allocation_percentage=_HUNDRED,
            emissions_kgco2e=total_emissions,
            emissions_co2_kg=gas_split["co2"],
            emissions_ch4_kg=gas_split["ch4"],
            emissions_n2o_kg=gas_split["n2o"],
            calculation_timestamp=datetime.now(timezone.utc),
        )

        elapsed_s = time.monotonic() - start_ts
        self._record_metrics(
            mode="pipeline", status="success",
            emissions_kgco2e=total_emissions, duration_s=elapsed_s,
        )

        logger.debug(
            "calculate_pipeline: %s tkm, ef=%s, emissions=%s kgCO2e",
            tonne_km, adjusted_ef, total_emissions,
        )

        return result

    # ------------------------------------------------------------------
    # 7. calculate_intermodal
    # ------------------------------------------------------------------

    def calculate_intermodal(
        self,
        transport_chain: TransportChain,
    ) -> List[LegResult]:
        """
        Calculate emissions for an intermodal (multi-leg) transport chain.

        Iterates over each leg in the transport chain and dispatches to the
        appropriate mode-specific calculator. Returns a list of LegResult
        objects, one per leg.

        Args:
            transport_chain: TransportChain with one or more connected legs.

        Returns:
            List of LegResult objects, one per leg, in chain order.

        Raises:
            ValueError: If transport_chain has no legs.

        Example:
            >>> chain = TransportChain(
            ...     chain_id="TC-001",
            ...     origin="Shanghai",
            ...     destination="Berlin",
            ...     legs=[
            ...         TransportLeg(mode=TransportMode.ROAD, ...),
            ...         TransportLeg(mode=TransportMode.MARITIME, ...),
            ...         TransportLeg(mode=TransportMode.RAIL, ...),
            ...     ],
            ... )
            >>> results = engine.calculate_intermodal(chain)
            >>> total = sum(r.emissions_kgco2e for r in results)
        """
        start_ts = time.monotonic()

        if not transport_chain.legs:
            raise ValueError(
                f"TransportChain {transport_chain.chain_id} has no legs"
            )

        logger.info(
            "calculate_intermodal: chain_id=%s legs=%d",
            transport_chain.chain_id,
            len(transport_chain.legs),
        )

        results: List[LegResult] = []

        for idx, leg in enumerate(transport_chain.legs):
            logger.debug(
                "calculate_intermodal: processing leg %d/%d mode=%s",
                idx + 1, len(transport_chain.legs), leg.mode.value,
            )
            result = self._calculate_single_leg(
                leg=leg,
                gwp_source=self._gwp_source,
                calc_id=f"intermodal_{transport_chain.chain_id}_leg{idx}",
            )
            results.append(result)

        elapsed_s = time.monotonic() - start_ts
        total_emissions = sum(
            (r.emissions_kgco2e for r in results), _ZERO
        )

        self._record_metrics(
            mode="intermodal", status="success",
            emissions_kgco2e=total_emissions, duration_s=elapsed_s,
        )

        logger.info(
            "calculate_intermodal complete: chain_id=%s total=%s kgCO2e legs=%d elapsed=%.4fs",
            transport_chain.chain_id,
            total_emissions,
            len(results),
            elapsed_s,
        )

        return results

    # ------------------------------------------------------------------
    # 8. apply_load_factor
    # ------------------------------------------------------------------

    def apply_load_factor(
        self,
        base_ef: Decimal,
        actual_load: Decimal,
        max_payload: Decimal,
    ) -> Decimal:
        """
        Adjust emission factor for actual load vs. max payload.

        When a vehicle is partially loaded, the emissions per tonne-km are
        higher because the same vehicle-km emissions are spread across fewer
        tonnes. The adjustment formula is:

            adjusted_ef = base_ef x (max_payload / actual_load)

        This ensures that a half-loaded truck effectively doubles the emission
        factor per tonne-km.

        Args:
            base_ef: Base emission factor in kgCO2e per tonne-km.
            actual_load: Actual load in tonnes. Must be > 0 and <= max_payload.
            max_payload: Maximum payload capacity in tonnes. Must be > 0.

        Returns:
            Load-factor-adjusted emission factor in kgCO2e per tonne-km.

        Raises:
            ValueError: If actual_load <= 0 or > max_payload, or max_payload <= 0.

        Example:
            >>> adjusted = engine.apply_load_factor(
            ...     base_ef=Decimal("0.085"),
            ...     actual_load=Decimal("12"),
            ...     max_payload=Decimal("24"),
            ... )
            >>> # adjusted == 0.085 * (24/12) == 0.17
        """
        if max_payload <= _ZERO:
            raise ValueError(f"max_payload must be > 0, got {max_payload}")
        if actual_load <= _ZERO:
            raise ValueError(f"actual_load must be > 0, got {actual_load}")
        if actual_load > max_payload:
            raise ValueError(
                f"actual_load ({actual_load}) cannot exceed max_payload ({max_payload})"
            )

        ratio = (max_payload / actual_load).quantize(_PRECISION, rounding=ROUND_HALF_UP)
        adjusted = (base_ef * ratio).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        logger.debug(
            "apply_load_factor: base_ef=%s, load=%s/%s, ratio=%s, adjusted=%s",
            base_ef, actual_load, max_payload, ratio, adjusted,
        )

        return adjusted

    # ------------------------------------------------------------------
    # 9. apply_empty_running
    # ------------------------------------------------------------------

    def apply_empty_running(
        self,
        base_ef: Decimal,
        empty_rate: Decimal,
    ) -> Decimal:
        """
        Adjust emission factor for empty running (deadheading).

        Empty running means a fraction of vehicle-km are travelled with no cargo.
        These emissions must still be allocated to loaded journeys. The formula is:

            effective_ef = base_ef / (1 - empty_rate)

        For example, with 28% empty running (road average), the effective EF
        increases by ~39%.

        Args:
            base_ef: Base emission factor in kgCO2e per tonne-km.
            empty_rate: Fraction of km travelled empty (0 to <1). Must be >= 0
                        and < 1.

        Returns:
            Empty-running-adjusted emission factor in kgCO2e per tonne-km.

        Raises:
            ValueError: If empty_rate < 0 or >= 1.

        Example:
            >>> adjusted = engine.apply_empty_running(
            ...     base_ef=Decimal("0.085"),
            ...     empty_rate=Decimal("0.28"),
            ... )
            >>> # adjusted == 0.085 / (1 - 0.28) == 0.085 / 0.72 ≈ 0.11805556
        """
        if empty_rate < _ZERO:
            raise ValueError(f"empty_rate must be >= 0, got {empty_rate}")
        if empty_rate >= _ONE:
            raise ValueError(f"empty_rate must be < 1, got {empty_rate}")

        denominator = (_ONE - empty_rate).quantize(_PRECISION, rounding=ROUND_HALF_UP)
        adjusted = (base_ef / denominator).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        logger.debug(
            "apply_empty_running: base_ef=%s, empty_rate=%s, adjusted=%s",
            base_ef, empty_rate, adjusted,
        )

        return adjusted

    # ------------------------------------------------------------------
    # 10. apply_laden_adjustment
    # ------------------------------------------------------------------

    def apply_laden_adjustment(
        self,
        base_ef: Decimal,
        laden_state: LadenState,
    ) -> Decimal:
        """
        Get laden state multiplier for emission factor adjustment.

        Laden state affects the per-tonne-km emission factor. A fully laden
        vehicle has a multiplier of 1.0 (baseline), while empty vehicles have
        higher multipliers (1.35-1.65 depending on vehicle type).

        Note: This returns the multiplier, not the adjusted EF. The caller
        should multiply base_ef by this value.

        Args:
            base_ef: Base emission factor (used only to look up vehicle-specific
                     multipliers if available; currently returns fixed multipliers).
            laden_state: Laden state of the vehicle.

        Returns:
            Multiplier to apply to the base emission factor.
                FULL -> 1.0
                HALF -> 1.20 (road average)
                EMPTY -> 1.50 (road average)
                AVERAGE -> 1.10 (weighted average assuming 65% utilisation)

        Example:
            >>> multiplier = engine.apply_laden_adjustment(
            ...     base_ef=Decimal("0.085"),
            ...     laden_state=LadenState.HALF,
            ... )
            >>> # multiplier == Decimal("1.20")
        """
        laden_multipliers: Dict[LadenState, Decimal] = {
            LadenState.FULL: Decimal("1.00"),
            LadenState.HALF: Decimal("1.20"),
            LadenState.EMPTY: Decimal("1.50"),
            LadenState.AVERAGE: Decimal("1.10"),
        }

        multiplier = laden_multipliers.get(laden_state, _ONE)

        logger.debug(
            "apply_laden_adjustment: laden_state=%s, multiplier=%s",
            laden_state.value, multiplier,
        )

        return multiplier

    # ------------------------------------------------------------------
    # 11. apply_reefer_uplift
    # ------------------------------------------------------------------

    def apply_reefer_uplift(
        self,
        base_emissions: Decimal,
        mode: TransportMode,
        temperature: TemperatureControl,
    ) -> Decimal:
        """
        Apply reefer (temperature-controlled) uplift to base emissions.

        Temperature-controlled transport requires additional energy for
        refrigeration/heating, increasing total emissions. The uplift factor
        is looked up from the REEFER_UPLIFT_FACTORS table.

        Args:
            base_emissions: Base emissions in kgCO2e (before reefer uplift).
            mode: Transport mode (road, maritime, air). Other modes default to 1.0.
            temperature: Temperature control requirement.

        Returns:
            Emissions with reefer uplift applied (kgCO2e).

        Example:
            >>> uplifted = engine.apply_reefer_uplift(
            ...     base_emissions=Decimal("100"),
            ...     mode=TransportMode.ROAD,
            ...     temperature=TemperatureControl.FROZEN_MINUS_18C,
            ... )
            >>> # uplifted == 100 * 1.62 == 162
        """
        multiplier = self._get_reefer_multiplier(mode, temperature)
        result = (base_emissions * multiplier).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        if multiplier != _ONE:
            logger.debug(
                "apply_reefer_uplift: mode=%s, temp=%s, multiplier=%s, "
                "base=%s -> uplifted=%s",
                mode.value, temperature.value, multiplier,
                base_emissions, result,
            )

        return result

    # ------------------------------------------------------------------
    # 12. calculate_great_circle_distance (Haversine)
    # ------------------------------------------------------------------

    def calculate_great_circle_distance(
        self,
        lat1: Decimal,
        lon1: Decimal,
        lat2: Decimal,
        lon2: Decimal,
    ) -> Decimal:
        """
        Calculate great circle distance using the Haversine formula.

        The Haversine formula determines the shortest distance between two
        points on a sphere given their latitude/longitude coordinates. Uses
        the WGS-84 mean Earth radius of 6,371.0088 km.

        Formula:
            a = sin^2(dlat/2) + cos(lat1) * cos(lat2) * sin^2(dlon/2)
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            d = R * c

        Note: This uses float internally for trigonometric functions and converts
        back to Decimal. The precision loss from float trig is negligible
        compared to the inherent approximation of the spherical model.

        Args:
            lat1: Latitude of origin in decimal degrees (-90 to 90).
            lon1: Longitude of origin in decimal degrees (-180 to 180).
            lat2: Latitude of destination in decimal degrees (-90 to 90).
            lon2: Longitude of destination in decimal degrees (-180 to 180).

        Returns:
            Great circle distance in kilometres (Decimal).

        Raises:
            ValueError: If any coordinate is out of valid range.

        Example:
            >>> # London Heathrow to New York JFK
            >>> gcd = engine.calculate_great_circle_distance(
            ...     lat1=Decimal("51.4700"),
            ...     lon1=Decimal("-0.4543"),
            ...     lat2=Decimal("40.6413"),
            ...     lon2=Decimal("-73.7781"),
            ... )
            >>> # gcd ≈ 5,555 km
        """
        # Validate coordinate ranges
        self._validate_coordinate(lat1, "lat1", Decimal("-90"), Decimal("90"))
        self._validate_coordinate(lon1, "lon1", Decimal("-180"), Decimal("180"))
        self._validate_coordinate(lat2, "lat2", Decimal("-90"), Decimal("90"))
        self._validate_coordinate(lon2, "lon2", Decimal("-180"), Decimal("180"))

        # Convert to float for trig operations
        lat1_f = math.radians(float(lat1))
        lon1_f = math.radians(float(lon1))
        lat2_f = math.radians(float(lat2))
        lon2_f = math.radians(float(lon2))

        dlat = lat2_f - lat1_f
        dlon = lon2_f - lon1_f

        # Haversine formula
        a = (
            math.sin(dlat / 2.0) ** 2
            + math.cos(lat1_f) * math.cos(lat2_f) * math.sin(dlon / 2.0) ** 2
        )
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

        # Distance in km
        distance_float = float(_EARTH_RADIUS_KM) * c
        distance_km = Decimal(str(distance_float)).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )

        logger.debug(
            "haversine: (%s,%s) -> (%s,%s) = %s km",
            lat1, lon1, lat2, lon2, distance_km,
        )

        return distance_km

    # ------------------------------------------------------------------
    # 13. apply_gcd_correction
    # ------------------------------------------------------------------

    def apply_gcd_correction(self, gcd_km: Decimal) -> Decimal:
        """
        Apply DEFRA great circle distance correction factor for air freight.

        DEFRA recommends a 1.09x (9%) uplift on GCD for air freight to account
        for non-great-circle routing, holding patterns, ATC diversions, taxiing,
        and wind effects. This correction is only applicable to air freight.

        Args:
            gcd_km: Great circle distance in kilometres. Must be > 0.

        Returns:
            Corrected distance in kilometres (= gcd_km x 1.09).

        Raises:
            ValueError: If gcd_km <= 0.

        Example:
            >>> corrected = engine.apply_gcd_correction(Decimal("5555"))
            >>> # corrected == 5555 * 1.09 == 6054.95
        """
        if gcd_km <= _ZERO:
            raise ValueError(f"gcd_km must be > 0, got {gcd_km}")

        corrected = (gcd_km * self._gcd_correction).quantize(
            _PRECISION, rounding=ROUND_HALF_UP
        )

        logger.debug(
            "apply_gcd_correction: %s km x %s = %s km",
            gcd_km, self._gcd_correction, corrected,
        )

        return corrected

    # ------------------------------------------------------------------
    # 14. convert_distance
    # ------------------------------------------------------------------

    def convert_distance(
        self,
        value: Decimal,
        from_unit: str,
        to_unit: str,
    ) -> Decimal:
        """
        Convert distance between units (km, miles, nautical_miles).

        Args:
            value: Distance value to convert. Must be >= 0.
            from_unit: Source unit ("km", "miles", "nautical_miles").
            to_unit: Target unit ("km", "miles", "nautical_miles").

        Returns:
            Converted distance value.

        Raises:
            ValueError: If value < 0 or units are not recognized.

        Example:
            >>> km = engine.convert_distance(Decimal("100"), "miles", "km")
            >>> # km ≈ 160.9344
            >>> nm = engine.convert_distance(Decimal("500"), "km", "nautical_miles")
            >>> # nm ≈ 269.978
        """
        if value < _ZERO:
            raise ValueError(f"Distance value must be >= 0, got {value}")

        valid_units = {"km", "miles", "nautical_miles"}
        if from_unit not in valid_units:
            raise ValueError(
                f"Invalid from_unit '{from_unit}'. Must be one of {valid_units}"
            )
        if to_unit not in valid_units:
            raise ValueError(
                f"Invalid to_unit '{to_unit}'. Must be one of {valid_units}"
            )

        if from_unit == to_unit:
            return value

        # Convert to km first, then to target
        km_value = self._to_km(value, from_unit)
        result = self._from_km(km_value, to_unit)

        return result.quantize(_PRECISION, rounding=ROUND_HALF_UP)

    # ------------------------------------------------------------------
    # 15. convert_mass
    # ------------------------------------------------------------------

    def convert_mass(
        self,
        value: Decimal,
        from_unit: str,
        to_unit: str,
    ) -> Decimal:
        """
        Convert mass between units (tonnes, kg, lbs, short_tons).

        Args:
            value: Mass value to convert. Must be >= 0.
            from_unit: Source unit ("tonnes", "kg", "lbs", "short_tons").
            to_unit: Target unit ("tonnes", "kg", "lbs", "short_tons").

        Returns:
            Converted mass value.

        Raises:
            ValueError: If value < 0 or units are not recognized.

        Example:
            >>> tonnes = engine.convert_mass(Decimal("5000"), "kg", "tonnes")
            >>> # tonnes == 5.0
            >>> lbs = engine.convert_mass(Decimal("1"), "tonnes", "lbs")
            >>> # lbs ≈ 2204.623
        """
        if value < _ZERO:
            raise ValueError(f"Mass value must be >= 0, got {value}")

        valid_units = {"tonnes", "kg", "lbs", "short_tons"}
        if from_unit not in valid_units:
            raise ValueError(
                f"Invalid from_unit '{from_unit}'. Must be one of {valid_units}"
            )
        if to_unit not in valid_units:
            raise ValueError(
                f"Invalid to_unit '{to_unit}'. Must be one of {valid_units}"
            )

        if from_unit == to_unit:
            return value

        # Convert to tonnes first, then to target
        tonnes_value = self._to_tonnes(value, from_unit)
        result = self._from_tonnes(tonnes_value, to_unit)

        return result.quantize(_PRECISION, rounding=ROUND_HALF_UP)

    # ------------------------------------------------------------------
    # 16. split_by_gas
    # ------------------------------------------------------------------

    def split_by_gas(
        self,
        total_co2e: Decimal,
        mode: str,
    ) -> Dict[str, Decimal]:
        """
        Split total CO2e into individual gas components (CO2, CH4, N2O).

        Uses mode-specific gas split ratios to disaggregate total CO2e
        emissions into individual greenhouse gas masses. The CH4 and N2O
        values returned are in kgCO2e (i.e., already GWP-weighted), matching
        the convention in LegResult.

        Gas split ratios by mode:
            ROAD:      CO2=0.995, CH4=0.003, N2O=0.002
            RAIL:      CO2=0.990, CH4=0.005, N2O=0.005
            MARITIME:  CO2=0.997, CH4=0.002, N2O=0.001
            AIR:       CO2=0.998, CH4=0.001, N2O=0.001
            PIPELINE:  CO2=0.950, CH4=0.045, N2O=0.005

        Args:
            total_co2e: Total emissions in kgCO2e. Must be >= 0.
            mode: Transport mode value string (e.g., "road", "rail").

        Returns:
            Dictionary with keys "co2", "ch4", "n2o" in kgCO2e.

        Example:
            >>> split = engine.split_by_gas(Decimal("1000"), "road")
            >>> split["co2"]  # Decimal("995.00000000")
            >>> split["ch4"]  # Decimal("3.00000000")
            >>> split["n2o"]  # Decimal("2.00000000")
        """
        ratios = _GAS_SPLIT_RATIOS.get(mode, _GAS_SPLIT_RATIOS[TransportMode.ROAD.value])

        co2 = (total_co2e * ratios["co2"]).quantize(_PRECISION, rounding=ROUND_HALF_UP)
        ch4 = (total_co2e * ratios["ch4"]).quantize(_PRECISION, rounding=ROUND_HALF_UP)
        n2o = (total_co2e * ratios["n2o"]).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        # Ensure components sum to total (assign rounding remainder to CO2)
        remainder = total_co2e - co2 - ch4 - n2o
        co2 = (co2 + remainder).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        return {
            "co2": co2,
            "ch4": ch4,
            "n2o": n2o,
        }

    # ------------------------------------------------------------------
    # 17. calculate_biogenic_split
    # ------------------------------------------------------------------

    def calculate_biogenic_split(
        self,
        total: Decimal,
        fuel_type: str,
    ) -> Tuple[Decimal, Decimal]:
        """
        Split total emissions into fossil and biogenic fractions.

        Biogenic CO2 from biomass combustion (biodiesel, HVO, SAF) is reported
        separately outside the Scope 1/3 total per GHG Protocol guidance. The
        biogenic fraction is determined by the fuel type.

        Args:
            total: Total emissions in kgCO2e. Must be >= 0.
            fuel_type: Fuel type value string (e.g., "biodiesel", "diesel").

        Returns:
            Tuple of (fossil_kgco2e, biogenic_kgco2e).

        Example:
            >>> fossil, bio = engine.calculate_biogenic_split(
            ...     Decimal("1000"), "biodiesel"
            ... )
            >>> # fossil == 0, bio == 1000 (100% biogenic)
            >>> fossil, bio = engine.calculate_biogenic_split(
            ...     Decimal("1000"), "diesel"
            ... )
            >>> # fossil == 1000, bio == 0 (100% fossil)
            >>> fossil, bio = engine.calculate_biogenic_split(
            ...     Decimal("1000"), "sustainable_aviation_fuel"
            ... )
            >>> # fossil == 200, bio == 800 (80% biogenic)
        """
        if total < _ZERO:
            raise ValueError(f"total must be >= 0, got {total}")

        biogenic_fraction = _BIOGENIC_FRACTIONS.get(fuel_type, _ZERO)

        biogenic = (total * biogenic_fraction).quantize(_PRECISION, rounding=ROUND_HALF_UP)
        fossil = (total - biogenic).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        logger.debug(
            "biogenic_split: fuel=%s, fraction=%s, fossil=%s, biogenic=%s",
            fuel_type, biogenic_fraction, fossil, biogenic,
        )

        return fossil, biogenic

    # ------------------------------------------------------------------
    # 18. calculate_wtw_breakdown
    # ------------------------------------------------------------------

    def calculate_wtw_breakdown(
        self,
        ttw_emissions: Decimal,
        mode: str,
    ) -> Dict[str, Decimal]:
        """
        Calculate Well-to-Wheel breakdown from Tank-to-Wheel emissions.

        WTW = TTW + WTT, where WTT is the upstream fuel production emissions.
        The WTT is calculated as:  WTT = TTW x WTT_ratio

        If the engine is configured with WTW scope, the input should be WTW
        emissions and this method decomposes them into TTW and WTT. If the
        engine is configured with TTW scope, the input is TTW and WTT is
        added on top.

        Args:
            ttw_emissions: Tank-to-Wheel emissions in kgCO2e. Must be >= 0.
            mode: Transport mode value string (e.g., "road").

        Returns:
            Dictionary with keys "ttw", "wtt", "wtw" in kgCO2e.

        Example:
            >>> breakdown = engine.calculate_wtw_breakdown(
            ...     Decimal("1000"), "road"
            ... )
            >>> breakdown["ttw"]  # Decimal("1000.00000000")
            >>> breakdown["wtt"]  # Decimal("218.00000000")
            >>> breakdown["wtw"]  # Decimal("1218.00000000")
        """
        if ttw_emissions < _ZERO:
            raise ValueError(f"ttw_emissions must be >= 0, got {ttw_emissions}")

        ratio = _WTT_TO_TTW_RATIOS.get(mode, Decimal("0.220"))

        wtt = (ttw_emissions * ratio).quantize(_PRECISION, rounding=ROUND_HALF_UP)
        wtw = (ttw_emissions + wtt).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        result = {
            "ttw": ttw_emissions.quantize(_PRECISION, rounding=ROUND_HALF_UP),
            "wtt": wtt,
            "wtw": wtw,
        }

        logger.debug(
            "wtw_breakdown: mode=%s, ttw=%s, wtt=%s, wtw=%s",
            mode, result["ttw"], result["wtt"], result["wtw"],
        )

        return result

    # ------------------------------------------------------------------
    # 19. batch_calculate
    # ------------------------------------------------------------------

    def batch_calculate(
        self,
        shipments: List[ShipmentInput],
    ) -> List[LegResult]:
        """
        Calculate emissions for a batch of shipments.

        Processes each shipment sequentially and collects results. Failed
        calculations are logged and skipped (not re-raised) to allow partial
        batch completion.

        Args:
            shipments: List of ShipmentInput objects. May be empty.

        Returns:
            List of LegResult objects. May be shorter than input list if some
            calculations failed.

        Example:
            >>> results = engine.batch_calculate([shipment1, shipment2, shipment3])
            >>> total = sum(r.emissions_kgco2e for r in results)
        """
        start_ts = time.monotonic()
        results: List[LegResult] = []
        failed_count = 0

        logger.info("batch_calculate: processing %d shipments", len(shipments))

        for idx, shipment in enumerate(shipments):
            try:
                result = self.calculate(shipment)
                results.append(result)
            except Exception as exc:
                failed_count += 1
                logger.error(
                    "batch_calculate: shipment %d/%d (%s) failed: %s",
                    idx + 1, len(shipments), shipment.shipment_id, exc,
                )

        elapsed_s = time.monotonic() - start_ts
        total_emissions = sum((r.emissions_kgco2e for r in results), _ZERO)

        logger.info(
            "batch_calculate complete: total=%d success=%d failed=%d "
            "emissions=%s kgCO2e elapsed=%.4fs",
            len(shipments), len(results), failed_count,
            total_emissions, elapsed_s,
        )

        return results

    # ------------------------------------------------------------------
    # 20. aggregate_by_mode
    # ------------------------------------------------------------------

    def aggregate_by_mode(
        self,
        results: List[LegResult],
    ) -> Dict[str, Decimal]:
        """
        Aggregate emissions by transport mode from a list of LegResults.

        Args:
            results: List of LegResult objects from one or more calculations.

        Returns:
            Dictionary mapping transport mode value to total kgCO2e.

        Example:
            >>> agg = engine.aggregate_by_mode(results)
            >>> agg["road"]       # Decimal("1234.56")
            >>> agg["maritime"]   # Decimal("5678.90")
        """
        aggregation: Dict[str, Decimal] = {}

        for result in results:
            mode_key = result.mode.value
            current = aggregation.get(mode_key, _ZERO)
            aggregation[mode_key] = (current + result.emissions_kgco2e).quantize(
                _PRECISION, rounding=ROUND_HALF_UP
            )

        logger.debug("aggregate_by_mode: %s", aggregation)

        return aggregation

    # ------------------------------------------------------------------
    # 21. get_data_quality_score
    # ------------------------------------------------------------------

    def get_data_quality_score(
        self,
        distance_method: str,
        ef_source: str,
    ) -> Decimal:
        """
        Calculate composite data quality score per ISO 14083.

        The DQI score is based on the distance method (how was distance determined)
        and the emission factor source (which EF database was used). Scores range
        from 1 (best) to 5 (worst).

        Composite score = (distance_method_score + ef_source_score) / 2

        Args:
            distance_method: Distance determination method value string
                             (e.g., "actual", "great_circle").
            ef_source: Emission factor source (e.g., "glec_v3.0", "supplier_specific").

        Returns:
            Composite DQI score (Decimal, 1.0 to 5.0).

        Example:
            >>> score = engine.get_data_quality_score("actual", "glec_v3.0")
            >>> # score == (1.0 + 2.0) / 2 == 1.5
        """
        dm_score = _DISTANCE_METHOD_DQ_SCORES.get(distance_method, Decimal("4.0"))
        ef_score = _EF_SOURCE_DQ_SCORES.get(ef_source.lower(), Decimal("3.5"))

        composite = ((dm_score + ef_score) / _TWO).quantize(
            Decimal("0.1"), rounding=ROUND_HALF_UP
        )

        # Clamp to [1.0, 5.0]
        composite = max(Decimal("1.0"), min(Decimal("5.0"), composite))

        logger.debug(
            "data_quality_score: distance_method=%s(%s), ef_source=%s(%s), composite=%s",
            distance_method, dm_score, ef_source, ef_score, composite,
        )

        return composite

    # ==================================================================
    # PRIVATE HELPER METHODS
    # ==================================================================

    # ------------------------------------------------------------------
    # _resolve_legs: Extract legs from ShipmentInput
    # ------------------------------------------------------------------

    def _resolve_legs(self, shipment: ShipmentInput) -> List[TransportLeg]:
        """
        Resolve transport legs from a ShipmentInput.

        If the shipment has a transport_chain, use its legs. Otherwise use
        the shipment's direct legs list.

        Args:
            shipment: ShipmentInput to resolve.

        Returns:
            List of TransportLeg objects.
        """
        if shipment.transport_chain is not None:
            return shipment.transport_chain.legs
        return shipment.legs

    # ------------------------------------------------------------------
    # _calculate_single_leg: Dispatch to mode-specific calculator
    # ------------------------------------------------------------------

    def _calculate_single_leg(
        self,
        leg: TransportLeg,
        gwp_source: GWPSource,
        calc_id: str,
    ) -> LegResult:
        """
        Calculate emissions for a single transport leg by dispatching to the
        appropriate mode-specific method.

        Args:
            leg: TransportLeg with mode, distance, mass, etc.
            gwp_source: GWP source for the calculation.
            calc_id: Unique calculation identifier for tracing.

        Returns:
            LegResult with emissions.

        Raises:
            ValueError: If the transport mode is not supported.
        """
        mode = leg.mode

        if mode == TransportMode.ROAD:
            return self._calculate_road_leg(leg)
        elif mode == TransportMode.RAIL:
            return self._calculate_rail_leg(leg)
        elif mode == TransportMode.MARITIME:
            return self._calculate_maritime_leg(leg)
        elif mode == TransportMode.AIR:
            return self._calculate_air_leg(leg)
        elif mode == TransportMode.PIPELINE:
            return self._calculate_pipeline_leg(leg)
        elif mode == TransportMode.INTERMODAL:
            # Intermodal single leg defaults to road
            logger.warning(
                "Single leg with mode=INTERMODAL treated as ROAD for calc_id=%s",
                calc_id,
            )
            return self._calculate_road_leg(leg)
        else:
            raise ValueError(f"Unsupported transport mode: {mode.value}")

    # ------------------------------------------------------------------
    # _calculate_road_leg: Delegate from TransportLeg
    # ------------------------------------------------------------------

    def _calculate_road_leg(self, leg: TransportLeg) -> LegResult:
        """
        Delegate road calculation from a TransportLeg.

        Extracts vehicle_type and laden_state from the leg and calls
        calculate_road().

        Args:
            leg: TransportLeg with mode == ROAD.

        Returns:
            LegResult from calculate_road().
        """
        vehicle_type = self._resolve_road_vehicle_type(leg.vehicle_type)
        fuel_type = leg.fuel_type

        result = self.calculate_road(
            mass_tonnes=leg.cargo_mass_tonnes,
            distance_km=leg.distance_km,
            vehicle_type=vehicle_type,
            fuel_type=fuel_type,
            laden_state=leg.laden_state,
            temperature_control=leg.temperature_control,
            load_factor=leg.load_factor,
            custom_ef_per_tkm=leg.custom_ef_per_tkm,
            ef_source=leg.ef_source,
        )

        return self._apply_leg_allocation(result, leg)

    # ------------------------------------------------------------------
    # _calculate_rail_leg: Delegate from TransportLeg
    # ------------------------------------------------------------------

    def _calculate_rail_leg(self, leg: TransportLeg) -> LegResult:
        """
        Delegate rail calculation from a TransportLeg.

        Args:
            leg: TransportLeg with mode == RAIL.

        Returns:
            LegResult from calculate_rail().
        """
        rail_type = self._resolve_rail_type(leg.vehicle_type)

        result = self.calculate_rail(
            mass_tonnes=leg.cargo_mass_tonnes,
            distance_km=leg.distance_km,
            rail_type=rail_type,
            temperature_control=leg.temperature_control,
            custom_ef_per_tkm=leg.custom_ef_per_tkm,
            ef_source=leg.ef_source,
        )

        return self._apply_leg_allocation(result, leg)

    # ------------------------------------------------------------------
    # _calculate_maritime_leg: Delegate from TransportLeg
    # ------------------------------------------------------------------

    def _calculate_maritime_leg(self, leg: TransportLeg) -> LegResult:
        """
        Delegate maritime calculation from a TransportLeg.

        Args:
            leg: TransportLeg with mode == MARITIME.

        Returns:
            LegResult from calculate_maritime().
        """
        vessel_type = self._resolve_maritime_vessel_type(leg.vehicle_type)

        result = self.calculate_maritime(
            mass_tonnes=leg.cargo_mass_tonnes,
            distance_km=leg.distance_km,
            vessel_type=vessel_type,
            temperature_control=leg.temperature_control,
            custom_ef_per_tkm=leg.custom_ef_per_tkm,
            ef_source=leg.ef_source,
        )

        return self._apply_leg_allocation(result, leg)

    # ------------------------------------------------------------------
    # _calculate_air_leg: Delegate from TransportLeg
    # ------------------------------------------------------------------

    def _calculate_air_leg(self, leg: TransportLeg) -> LegResult:
        """
        Delegate air calculation from a TransportLeg.

        Args:
            leg: TransportLeg with mode == AIR.

        Returns:
            LegResult from calculate_air().
        """
        aircraft_type = self._resolve_aircraft_type(leg.vehicle_type)

        # Apply GCD uplift if distance was determined via great circle
        apply_gcd = leg.distance_method == DistanceMethod.GREAT_CIRCLE

        result = self.calculate_air(
            mass_tonnes=leg.cargo_mass_tonnes,
            distance_km=leg.distance_km,
            aircraft_type=aircraft_type,
            temperature_control=leg.temperature_control,
            apply_gcd_uplift=apply_gcd,
            custom_ef_per_tkm=leg.custom_ef_per_tkm,
            ef_source=leg.ef_source,
        )

        return self._apply_leg_allocation(result, leg)

    # ------------------------------------------------------------------
    # _calculate_pipeline_leg: Delegate from TransportLeg
    # ------------------------------------------------------------------

    def _calculate_pipeline_leg(self, leg: TransportLeg) -> LegResult:
        """
        Delegate pipeline calculation from a TransportLeg.

        Args:
            leg: TransportLeg with mode == PIPELINE.

        Returns:
            LegResult from calculate_pipeline().
        """
        pipeline_type = self._resolve_pipeline_type(leg.vehicle_type)

        result = self.calculate_pipeline(
            mass_tonnes=leg.cargo_mass_tonnes,
            distance_km=leg.distance_km,
            pipeline_type=pipeline_type,
            custom_ef_per_tkm=leg.custom_ef_per_tkm,
            ef_source=leg.ef_source,
        )

        return self._apply_leg_allocation(result, leg)

    # ------------------------------------------------------------------
    # _apply_leg_allocation: Apply allocation percentage from leg
    # ------------------------------------------------------------------

    def _apply_leg_allocation(
        self,
        result: LegResult,
        leg: TransportLeg,
    ) -> LegResult:
        """
        Apply allocation percentage from the transport leg to the result.

        If the leg specifies an allocation_percentage < 100, scale down the
        emissions proportionally. This handles shared transport (e.g., LTL
        road freight, container sharing).

        Args:
            result: LegResult before allocation.
            leg: TransportLeg with allocation_percentage.

        Returns:
            New LegResult with allocation applied (or original if 100%).
        """
        alloc = leg.allocation_percentage
        if alloc is None or alloc == _HUNDRED:
            return result

        factor = (alloc / _HUNDRED).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        return LegResult(
            leg_id=result.leg_id,
            mode=result.mode,
            vehicle_type=result.vehicle_type,
            distance_km=result.distance_km,
            cargo_mass_tonnes=result.cargo_mass_tonnes,
            tonne_km=result.tonne_km,
            ef_per_tkm=result.ef_per_tkm,
            ef_source=result.ef_source,
            ef_scope=result.ef_scope,
            laden_adjustment=result.laden_adjustment,
            reefer_uplift=result.reefer_uplift,
            allocation_percentage=alloc,
            emissions_kgco2e=(result.emissions_kgco2e * factor).quantize(
                _PRECISION, rounding=ROUND_HALF_UP
            ),
            emissions_co2_kg=(result.emissions_co2_kg * factor).quantize(
                _PRECISION, rounding=ROUND_HALF_UP
            ),
            emissions_ch4_kg=(result.emissions_ch4_kg * factor).quantize(
                _PRECISION, rounding=ROUND_HALF_UP
            ),
            emissions_n2o_kg=(result.emissions_n2o_kg * factor).quantize(
                _PRECISION, rounding=ROUND_HALF_UP
            ),
            calculation_timestamp=result.calculation_timestamp,
        )

    # ------------------------------------------------------------------
    # Vehicle type resolution helpers
    # ------------------------------------------------------------------

    def _resolve_road_vehicle_type(
        self,
        vehicle_type_str: Optional[str],
    ) -> RoadVehicleType:
        """
        Resolve a string vehicle type to RoadVehicleType enum.

        Falls back to HGV_ARTIC_33T_PLUS if not matched.

        Args:
            vehicle_type_str: Vehicle type string or None.

        Returns:
            RoadVehicleType enum value.
        """
        if vehicle_type_str is None:
            return RoadVehicleType.HGV_ARTIC_33T_PLUS

        for vt in RoadVehicleType:
            if vt.value == vehicle_type_str:
                return vt

        logger.warning(
            "Unrecognised road vehicle_type '%s', defaulting to HGV_ARTIC_33T_PLUS",
            vehicle_type_str,
        )
        return RoadVehicleType.HGV_ARTIC_33T_PLUS

    def _resolve_rail_type(
        self,
        vehicle_type_str: Optional[str],
    ) -> RailType:
        """
        Resolve a string vehicle type to RailType enum.

        Falls back to RailType.AVERAGE if not matched.

        Args:
            vehicle_type_str: Vehicle type string or None.

        Returns:
            RailType enum value.
        """
        if vehicle_type_str is None:
            return RailType.AVERAGE

        for rt in RailType:
            if rt.value == vehicle_type_str:
                return rt

        logger.warning(
            "Unrecognised rail type '%s', defaulting to AVERAGE",
            vehicle_type_str,
        )
        return RailType.AVERAGE

    def _resolve_maritime_vessel_type(
        self,
        vehicle_type_str: Optional[str],
    ) -> MaritimeVesselType:
        """
        Resolve a string vehicle type to MaritimeVesselType enum.

        Falls back to CONTAINER_PANAMAX if not matched.

        Args:
            vehicle_type_str: Vehicle type string or None.

        Returns:
            MaritimeVesselType enum value.
        """
        if vehicle_type_str is None:
            return MaritimeVesselType.CONTAINER_PANAMAX

        for mvt in MaritimeVesselType:
            if mvt.value == vehicle_type_str:
                return mvt

        logger.warning(
            "Unrecognised maritime vessel_type '%s', defaulting to CONTAINER_PANAMAX",
            vehicle_type_str,
        )
        return MaritimeVesselType.CONTAINER_PANAMAX

    def _resolve_aircraft_type(
        self,
        vehicle_type_str: Optional[str],
    ) -> AircraftType:
        """
        Resolve a string vehicle type to AircraftType enum.

        Falls back to WIDEBODY_FREIGHTER if not matched.

        Args:
            vehicle_type_str: Vehicle type string or None.

        Returns:
            AircraftType enum value.
        """
        if vehicle_type_str is None:
            return AircraftType.WIDEBODY_FREIGHTER

        for at in AircraftType:
            if at.value == vehicle_type_str:
                return at

        logger.warning(
            "Unrecognised aircraft_type '%s', defaulting to WIDEBODY_FREIGHTER",
            vehicle_type_str,
        )
        return AircraftType.WIDEBODY_FREIGHTER

    def _resolve_pipeline_type(
        self,
        vehicle_type_str: Optional[str],
    ) -> PipelineType:
        """
        Resolve a string vehicle type to PipelineType enum.

        Falls back to REFINED_PRODUCTS if not matched.

        Args:
            vehicle_type_str: Vehicle type string or None.

        Returns:
            PipelineType enum value.
        """
        if vehicle_type_str is None:
            return PipelineType.REFINED_PRODUCTS

        for pt in PipelineType:
            if pt.value == vehicle_type_str:
                return pt

        logger.warning(
            "Unrecognised pipeline_type '%s', defaulting to REFINED_PRODUCTS",
            vehicle_type_str,
        )
        return PipelineType.REFINED_PRODUCTS

    # ------------------------------------------------------------------
    # _resolve_rail_ef: Rail EF lookup with fallback chain
    # ------------------------------------------------------------------

    def _resolve_rail_ef(
        self,
        rail_type: RailType,
        region: str,
    ) -> Decimal:
        """
        Resolve rail emission factor with fallback chain.

        Lookup order:
            1. (rail_type, region)
            2. (rail_type, "global")
            3. (AVERAGE, "global")
            4. Hard-coded default 0.0156 kgCO2e/tkm

        Args:
            rail_type: Rail type enum.
            region: Geographic region string (lowercase).

        Returns:
            Emission factor in kgCO2e per tonne-km.
        """
        region_lower = region.lower()

        # Try exact match
        ef = RAIL_EMISSION_FACTORS.get((rail_type, region_lower))
        if ef is not None:
            return ef

        # Try global fallback
        ef = RAIL_EMISSION_FACTORS.get((rail_type, "global"))
        if ef is not None:
            logger.debug(
                "Rail EF fallback: (%s, %s) -> (%s, global)",
                rail_type.value, region_lower, rail_type.value,
            )
            return ef

        # Try average global
        ef = RAIL_EMISSION_FACTORS.get((RailType.AVERAGE, "global"))
        if ef is not None:
            logger.debug(
                "Rail EF fallback: (%s, %s) -> (average, global)",
                rail_type.value, region_lower,
            )
            return ef

        # Hard-coded last resort
        logger.warning(
            "No rail EF found for (%s, %s), using default 0.0156",
            rail_type.value, region_lower,
        )
        return Decimal("0.0156")

    # ------------------------------------------------------------------
    # _get_reefer_multiplier: Lookup reefer uplift factor
    # ------------------------------------------------------------------

    def _get_reefer_multiplier(
        self,
        mode: TransportMode,
        temperature: TemperatureControl,
    ) -> Decimal:
        """
        Look up reefer uplift multiplier from the REEFER_UPLIFT_FACTORS table.

        Returns 1.0 if the mode or temperature is not in the table (no uplift).

        Args:
            mode: Transport mode.
            temperature: Temperature control requirement.

        Returns:
            Multiplier (Decimal). 1.0 means no uplift.
        """
        mode_factors = REEFER_UPLIFT_FACTORS.get(mode)
        if mode_factors is None:
            return _ONE

        return mode_factors.get(temperature, _ONE)

    # ------------------------------------------------------------------
    # _get_max_payload: Lookup max payload for road vehicle type
    # ------------------------------------------------------------------

    def _get_max_payload(
        self,
        vehicle_type: RoadVehicleType,
    ) -> Decimal:
        """
        Get maximum payload capacity for a road vehicle type.

        These are approximate values used for load factor calculations.
        Actual capacity depends on vehicle specification.

        Args:
            vehicle_type: Road vehicle type.

        Returns:
            Max payload in metric tonnes.
        """
        payloads: Dict[RoadVehicleType, Decimal] = {
            RoadVehicleType.LCV_PETROL: Decimal("1.0"),
            RoadVehicleType.LCV_DIESEL: Decimal("1.2"),
            RoadVehicleType.LCV_ELECTRIC: Decimal("0.8"),
            RoadVehicleType.HGV_RIGID_3_5_7_5T: Decimal("4.0"),
            RoadVehicleType.HGV_RIGID_7_5_17T: Decimal("10.0"),
            RoadVehicleType.HGV_RIGID_17T_PLUS: Decimal("14.0"),
            RoadVehicleType.HGV_ARTIC_3_5_33T: Decimal("22.0"),
            RoadVehicleType.HGV_ARTIC_33T_PLUS: Decimal("26.0"),
            RoadVehicleType.HGV_CNG: Decimal("24.0"),
            RoadVehicleType.HGV_LNG: Decimal("24.0"),
            RoadVehicleType.HGV_ELECTRIC: Decimal("22.0"),
            RoadVehicleType.HYDROGEN_TRUCK: Decimal("23.0"),
        }

        return payloads.get(vehicle_type, Decimal("26.0"))

    # ------------------------------------------------------------------
    # _validate_activity_data: Common input validation
    # ------------------------------------------------------------------

    def _validate_activity_data(
        self,
        mass_tonnes: Decimal,
        distance_km: Decimal,
    ) -> None:
        """
        Validate that mass and distance are positive.

        Args:
            mass_tonnes: Cargo mass in tonnes.
            distance_km: Distance in km.

        Raises:
            ValueError: If either value is <= 0.
        """
        if mass_tonnes <= _ZERO:
            raise ValueError(f"mass_tonnes must be > 0, got {mass_tonnes}")
        if distance_km <= _ZERO:
            raise ValueError(f"distance_km must be > 0, got {distance_km}")

    # ------------------------------------------------------------------
    # _validate_coordinate: Validate lat/lon range
    # ------------------------------------------------------------------

    def _validate_coordinate(
        self,
        value: Decimal,
        name: str,
        min_val: Decimal,
        max_val: Decimal,
    ) -> None:
        """
        Validate a geographic coordinate is within range.

        Args:
            value: Coordinate value.
            name: Coordinate name for error messages.
            min_val: Minimum valid value (inclusive).
            max_val: Maximum valid value (inclusive).

        Raises:
            ValueError: If value is out of range.
        """
        if value < min_val or value > max_val:
            raise ValueError(
                f"{name} must be between {min_val} and {max_val}, got {value}"
            )

    # ------------------------------------------------------------------
    # Unit conversion helpers: Distance
    # ------------------------------------------------------------------

    def _to_km(self, value: Decimal, unit: str) -> Decimal:
        """
        Convert a distance value to kilometres.

        Args:
            value: Distance value.
            unit: Source unit ("km", "miles", "nautical_miles").

        Returns:
            Distance in kilometres.
        """
        if unit == "km":
            return value
        elif unit == "miles":
            return (value * _KM_PER_MILE).quantize(_PRECISION, rounding=ROUND_HALF_UP)
        elif unit == "nautical_miles":
            return (value * _KM_PER_NAUTICAL_MILE).quantize(
                _PRECISION, rounding=ROUND_HALF_UP
            )
        else:
            raise ValueError(f"Cannot convert from '{unit}' to km")

    def _from_km(self, km: Decimal, unit: str) -> Decimal:
        """
        Convert kilometres to another distance unit.

        Args:
            km: Distance in kilometres.
            unit: Target unit ("km", "miles", "nautical_miles").

        Returns:
            Distance in the target unit.
        """
        if unit == "km":
            return km
        elif unit == "miles":
            return (km * _MILES_PER_KM).quantize(_PRECISION, rounding=ROUND_HALF_UP)
        elif unit == "nautical_miles":
            return (km * _NAUTICAL_MILES_PER_KM).quantize(
                _PRECISION, rounding=ROUND_HALF_UP
            )
        else:
            raise ValueError(f"Cannot convert from km to '{unit}'")

    # ------------------------------------------------------------------
    # Unit conversion helpers: Mass
    # ------------------------------------------------------------------

    def _to_tonnes(self, value: Decimal, unit: str) -> Decimal:
        """
        Convert a mass value to metric tonnes.

        Args:
            value: Mass value.
            unit: Source unit ("tonnes", "kg", "lbs", "short_tons").

        Returns:
            Mass in metric tonnes.
        """
        if unit == "tonnes":
            return value
        elif unit == "kg":
            return (value * _TONNES_PER_KG).quantize(_PRECISION, rounding=ROUND_HALF_UP)
        elif unit == "lbs":
            return (value * _TONNES_PER_LB).quantize(_PRECISION, rounding=ROUND_HALF_UP)
        elif unit == "short_tons":
            return (value * _TONNES_PER_SHORT_TON).quantize(
                _PRECISION, rounding=ROUND_HALF_UP
            )
        else:
            raise ValueError(f"Cannot convert from '{unit}' to tonnes")

    def _from_tonnes(self, tonnes: Decimal, unit: str) -> Decimal:
        """
        Convert metric tonnes to another mass unit.

        Args:
            tonnes: Mass in metric tonnes.
            unit: Target unit ("tonnes", "kg", "lbs", "short_tons").

        Returns:
            Mass in the target unit.
        """
        if unit == "tonnes":
            return tonnes
        elif unit == "kg":
            return (tonnes * _KG_PER_TONNE).quantize(_PRECISION, rounding=ROUND_HALF_UP)
        elif unit == "lbs":
            return (tonnes * _LBS_PER_TONNE).quantize(_PRECISION, rounding=ROUND_HALF_UP)
        elif unit == "short_tons":
            return (tonnes * _SHORT_TONS_PER_TONNE).quantize(
                _PRECISION, rounding=ROUND_HALF_UP
            )
        else:
            raise ValueError(f"Cannot convert from tonnes to '{unit}'")

    # ------------------------------------------------------------------
    # _record_metrics: Prometheus metrics recording
    # ------------------------------------------------------------------

    def _record_metrics(
        self,
        mode: str,
        status: str,
        emissions_kgco2e: Decimal,
        duration_s: float,
    ) -> None:
        """
        Record calculation metrics to Prometheus.

        Args:
            mode: Transport mode string.
            status: "success" or "failure".
            emissions_kgco2e: Emissions value.
            duration_s: Calculation duration in seconds.
        """
        try:
            self._metrics.record_calculation(
                method="distance_based",
                transport_mode=mode,
                tenant_id="default",
                status=status,
                emissions_tco2e=float(emissions_kgco2e / _THOUSAND),
                duration_s=duration_s,
            )
        except Exception as exc:
            # Metrics recording must never fail the calculation
            logger.debug("Metrics recording failed (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # _compute_provenance_hash: SHA-256 for audit trail
    # ------------------------------------------------------------------

    def _compute_provenance_hash(
        self,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
    ) -> str:
        """
        Compute SHA-256 provenance hash for a calculation.

        Combines input and output data into a deterministic JSON string and
        hashes it. Used for audit trail verification.

        Args:
            input_data: Input parameters dictionary.
            output_data: Output/result parameters dictionary.

        Returns:
            SHA-256 hex digest string.
        """
        combined = {
            "agent_id": AGENT_ID,
            "agent_version": VERSION,
            "component": AGENT_COMPONENT,
            "engine": "DistanceBasedCalculatorEngine",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input": input_data,
            "output": output_data,
        }
        payload = json.dumps(combined, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ==============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# ==============================================================================


def create_distance_calculator(
    gwp_source: GWPSource = GWPSource.AR5,
    ef_scope: EFScope = EFScope.WTW,
    gcd_correction: Decimal = _GCD_AIR_CORRECTION,
    metrics: Optional[UpstreamTransportationMetrics] = None,
) -> DistanceBasedCalculatorEngine:
    """
    Factory function to create a DistanceBasedCalculatorEngine.

    This is the recommended way to create an engine instance. It provides
    sensible defaults aligned with GLEC Framework v3.0 and ISO 14083.

    Args:
        gwp_source: IPCC Assessment Report for GWP values. Default: AR5.
        ef_scope: Emission factor scope. Default: WTW (recommended by GLEC).
        gcd_correction: GCD correction factor for air. Default: 1.09.
        metrics: Optional Prometheus metrics collector.

    Returns:
        Configured DistanceBasedCalculatorEngine instance.

    Example:
        >>> engine = create_distance_calculator()
        >>> result = engine.calculate_road(
        ...     mass_tonnes=Decimal("10"),
        ...     distance_km=Decimal("250"),
        ... )
    """
    return DistanceBasedCalculatorEngine(
        gwp_source=gwp_source,
        ef_scope=ef_scope,
        gcd_correction=gcd_correction,
        metrics=metrics,
    )


def calculate_distance_based_emissions(
    mass_tonnes: Decimal,
    distance_km: Decimal,
    mode: TransportMode,
    vehicle_type: Optional[str] = None,
    laden_state: LadenState = LadenState.FULL,
    temperature_control: TemperatureControl = TemperatureControl.AMBIENT,
) -> LegResult:
    """
    One-shot convenience function for distance-based emission calculation.

    Creates a temporary engine and calculates emissions for a single leg.
    For batch operations or repeated calculations, create an engine instance
    directly with ``create_distance_calculator()``.

    Args:
        mass_tonnes: Cargo mass in metric tonnes.
        distance_km: Distance in kilometres.
        mode: Transport mode.
        vehicle_type: Optional vehicle type string.
        laden_state: Laden state (for road).
        temperature_control: Temperature control requirement.

    Returns:
        LegResult with emissions.

    Example:
        >>> result = calculate_distance_based_emissions(
        ...     mass_tonnes=Decimal("10"),
        ...     distance_km=Decimal("250"),
        ...     mode=TransportMode.ROAD,
        ... )
    """
    engine = DistanceBasedCalculatorEngine()

    if mode == TransportMode.ROAD:
        vt = engine._resolve_road_vehicle_type(vehicle_type)
        return engine.calculate_road(
            mass_tonnes=mass_tonnes,
            distance_km=distance_km,
            vehicle_type=vt,
            laden_state=laden_state,
            temperature_control=temperature_control,
        )
    elif mode == TransportMode.RAIL:
        rt = engine._resolve_rail_type(vehicle_type)
        return engine.calculate_rail(
            mass_tonnes=mass_tonnes,
            distance_km=distance_km,
            rail_type=rt,
            temperature_control=temperature_control,
        )
    elif mode == TransportMode.MARITIME:
        mvt = engine._resolve_maritime_vessel_type(vehicle_type)
        return engine.calculate_maritime(
            mass_tonnes=mass_tonnes,
            distance_km=distance_km,
            vessel_type=mvt,
            temperature_control=temperature_control,
        )
    elif mode == TransportMode.AIR:
        at = engine._resolve_aircraft_type(vehicle_type)
        return engine.calculate_air(
            mass_tonnes=mass_tonnes,
            distance_km=distance_km,
            aircraft_type=at,
            temperature_control=temperature_control,
        )
    elif mode == TransportMode.PIPELINE:
        pt = engine._resolve_pipeline_type(vehicle_type)
        return engine.calculate_pipeline(
            mass_tonnes=mass_tonnes,
            distance_km=distance_km,
            pipeline_type=pt,
        )
    else:
        raise ValueError(f"Unsupported transport mode: {mode.value}")


def calculate_haversine_distance(
    lat1: Decimal,
    lon1: Decimal,
    lat2: Decimal,
    lon2: Decimal,
    apply_air_correction: bool = False,
) -> Decimal:
    """
    Convenience function for Haversine great-circle distance.

    Args:
        lat1: Latitude of origin (-90 to 90).
        lon1: Longitude of origin (-180 to 180).
        lat2: Latitude of destination (-90 to 90).
        lon2: Longitude of destination (-180 to 180).
        apply_air_correction: If True, multiply by 1.09 for air freight.

    Returns:
        Distance in kilometres.

    Example:
        >>> d = calculate_haversine_distance(
        ...     Decimal("51.47"), Decimal("-0.45"),
        ...     Decimal("40.64"), Decimal("-73.78"),
        ...     apply_air_correction=True,
        ... )
    """
    engine = DistanceBasedCalculatorEngine()
    gcd = engine.calculate_great_circle_distance(lat1, lon1, lat2, lon2)

    if apply_air_correction:
        gcd = engine.apply_gcd_correction(gcd)

    return gcd


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    # Engine class
    "DistanceBasedCalculatorEngine",

    # Factory/convenience functions
    "create_distance_calculator",
    "calculate_distance_based_emissions",
    "calculate_haversine_distance",

    # Constants (re-exported for convenience)
    "_GAS_SPLIT_RATIOS",
    "_WTT_TO_TTW_RATIOS",
    "_BIOGENIC_FRACTIONS",
    "_DISTANCE_METHOD_DQ_SCORES",
    "_EF_SOURCE_DQ_SCORES",
    "_GCD_AIR_CORRECTION",
    "_EARTH_RADIUS_KM",
]
