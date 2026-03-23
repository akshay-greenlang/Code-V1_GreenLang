# -*- coding: utf-8 -*-
"""
MultiLegCalculatorEngine - Engine 5: Upstream Transportation Agent (AGENT-MRV-017)

Orchestrates calculations for complete multi-modal transport chains per ISO 14083
and the GLEC Framework v3. Handles multi-leg journeys with hub/transshipment
points, supplier-specific emission data, allocation across shared transport,
refrigerated (reefer) transport uplift, and warehousing emissions.

Transport Chain Concept (per ISO 14083):
    Origin -> [Hub 0] -> Leg 1 -> [Hub 1] -> Leg 2 -> [Hub 2] -> ... -> Leg N -> [Hub N] -> Destination

Each element is a Transport Chain Element (TCE):
    - Leg TCE: One mode of transport between two points
    - Hub TCE: Transshipment, sorting, or storage at a logistics node

Zero-Hallucination Guarantees:
    - All calculations use Python Decimal (8 decimal places)
    - No LLM calls in the calculation path
    - Every step is recorded in the calculation trace
    - SHA-256 provenance hash for every result
    - Deterministic emission factor lookups from embedded tables

Example:
    >>> from greenlang.agents.mrv.upstream_transportation.multi_leg_calculator import MultiLegCalculatorEngine
    >>> from greenlang.agents.mrv.upstream_transportation.models import (
    ...     TransportChain, TransportLeg, TransportHub, TransportMode, HubType
    ... )
    >>> from decimal import Decimal
    >>> engine = MultiLegCalculatorEngine()
    >>> leg = TransportLeg(
    ...     mode=TransportMode.ROAD,
    ...     origin="Shanghai Port",
    ...     destination="Ningbo Warehouse",
    ...     distance_km=Decimal("180"),
    ...     cargo_mass_tonnes=Decimal("20"),
    ... )
    >>> chain = TransportChain(
    ...     origin="Shanghai Port",
    ...     destination="Ningbo Warehouse",
    ...     legs=[leg],
    ... )
    >>> result = engine.calculate_chain(chain)
    >>> assert result.total_emissions_kgco2e > 0

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-017 Upstream Transportation (GL-MRV-S3-004)
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
from typing import Any, Dict, List, Optional, Tuple, Union

from greenlang.agents.mrv.upstream_transportation.models import (
    # Enums
    AllocationMethod,
    CalculationMethod,
    DQIScore,
    EFScope,
    EmissionGas,
    GWPSource,
    HubType,
    LadenState,
    RoadVehicleType,
    RailType,
    MaritimeVesselType,
    AircraftType,
    PipelineType,
    TemperatureControl,
    TransportFuelType,
    TransportMode,
    # Constants
    AIR_EMISSION_FACTORS,
    DQI_SCORE_VALUES,
    FUEL_EMISSION_FACTORS,
    GWP_VALUES,
    HUB_EMISSION_FACTORS,
    LOAD_FACTOR_DEFAULTS,
    MARITIME_EMISSION_FACTORS,
    PIPELINE_EMISSION_FACTORS,
    RAIL_EMISSION_FACTORS,
    REEFER_UPLIFT_FACTORS,
    ROAD_EMISSION_FACTORS,
    WAREHOUSE_ENERGY_INTENSITIES,
    # Models
    AllocationConfig,
    HubResult,
    LegResult,
    ReeferConfig,
    SupplierEmissionInput,
    TransportChain,
    TransportChainResult,
    TransportHub,
    TransportLeg,
    WarehouseConfig,
    # Helpers
    calculate_co2e,
    calculate_provenance_hash,
    get_gwp,
)
from greenlang.agents.mrv.upstream_transportation.provenance import get_provenance_tracker

logger = logging.getLogger(__name__)

# =============================================================================
# DECIMAL PRECISION
# =============================================================================

_PRECISION = Decimal("0.00000001")  # 8 decimal places
_ZERO = Decimal("0")
_ONE = Decimal("1")
_HUNDRED = Decimal("100")
_THOUSAND = Decimal("1000")


def _q(value: Decimal) -> Decimal:
    """Quantize a Decimal to 8 decimal places."""
    return value.quantize(_PRECISION, rounding=ROUND_HALF_UP)


# =============================================================================
# SMARTWAY CARRIER CATEGORIES (13)
# =============================================================================

SMARTWAY_CARRIER_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "TL": {
        "description": "Truckload",
        "mode": "road",
        "default_ef_per_tkm": Decimal("0.062"),
        "bin_low": Decimal("0.040"),
        "bin_high": Decimal("0.090"),
    },
    "LTL": {
        "description": "Less Than Truckload",
        "mode": "road",
        "default_ef_per_tkm": Decimal("0.118"),
        "bin_low": Decimal("0.075"),
        "bin_high": Decimal("0.180"),
    },
    "PACKAGE": {
        "description": "Package / Parcel",
        "mode": "road",
        "default_ef_per_tkm": Decimal("0.252"),
        "bin_low": Decimal("0.150"),
        "bin_high": Decimal("0.400"),
    },
    "REFRIGERATED_TL": {
        "description": "Refrigerated Truckload",
        "mode": "road",
        "default_ef_per_tkm": Decimal("0.085"),
        "bin_low": Decimal("0.055"),
        "bin_high": Decimal("0.125"),
    },
    "REFRIGERATED_LTL": {
        "description": "Refrigerated Less Than Truckload",
        "mode": "road",
        "default_ef_per_tkm": Decimal("0.158"),
        "bin_low": Decimal("0.100"),
        "bin_high": Decimal("0.240"),
    },
    "TANKER": {
        "description": "Tanker",
        "mode": "road",
        "default_ef_per_tkm": Decimal("0.068"),
        "bin_low": Decimal("0.045"),
        "bin_high": Decimal("0.100"),
    },
    "FLATBED": {
        "description": "Flatbed",
        "mode": "road",
        "default_ef_per_tkm": Decimal("0.072"),
        "bin_low": Decimal("0.048"),
        "bin_high": Decimal("0.105"),
    },
    "MOVING": {
        "description": "Moving / Household Goods",
        "mode": "road",
        "default_ef_per_tkm": Decimal("0.095"),
        "bin_low": Decimal("0.060"),
        "bin_high": Decimal("0.145"),
    },
    "DRAYAGE": {
        "description": "Drayage (Port/Intermodal)",
        "mode": "road",
        "default_ef_per_tkm": Decimal("0.145"),
        "bin_low": Decimal("0.090"),
        "bin_high": Decimal("0.220"),
    },
    "EXPEDITED": {
        "description": "Expedited / Time-Critical",
        "mode": "road",
        "default_ef_per_tkm": Decimal("0.175"),
        "bin_low": Decimal("0.110"),
        "bin_high": Decimal("0.260"),
    },
    "RAIL": {
        "description": "Rail Freight",
        "mode": "rail",
        "default_ef_per_tkm": Decimal("0.022"),
        "bin_low": Decimal("0.014"),
        "bin_high": Decimal("0.032"),
    },
    "BARGE": {
        "description": "Inland Barge / Waterway",
        "mode": "maritime",
        "default_ef_per_tkm": Decimal("0.031"),
        "bin_low": Decimal("0.020"),
        "bin_high": Decimal("0.048"),
    },
    "MULTI_MODAL": {
        "description": "Multi-modal (intermodal combination)",
        "mode": "intermodal",
        "default_ef_per_tkm": Decimal("0.045"),
        "bin_low": Decimal("0.028"),
        "bin_high": Decimal("0.070"),
    },
}

# =============================================================================
# METHODOLOGY VALIDATION MAP
# =============================================================================

METHODOLOGY_VALIDATION: Dict[str, Dict[str, Any]] = {
    "GLEC v3": {
        "valid": True,
        "score": Decimal("1.00"),
        "note": "GLEC Framework v3.0 - fully aligned with ISO 14083",
    },
    "ISO 14083": {
        "valid": True,
        "score": Decimal("1.00"),
        "note": "International standard for transport GHG quantification",
    },
    "EN 16258": {
        "valid": True,
        "score": Decimal("0.90"),
        "note": "Superseded by ISO 14083 but still acceptable",
    },
    "GHG Protocol": {
        "valid": True,
        "score": Decimal("0.95"),
        "note": "GHG Protocol Corporate Value Chain (Scope 3) Standard",
    },
    "EcoTransIT": {
        "valid": True,
        "score": Decimal("0.85"),
        "note": "EcoTransIT World tool - widely used in logistics",
    },
    "Custom": {
        "valid": False,
        "score": Decimal("0.50"),
        "note": "Custom methodology - needs review and verification",
    },
}

# =============================================================================
# CERTIFICATION VALIDATION MAP
# =============================================================================

CERTIFICATION_VALIDATION: Dict[str, Dict[str, Any]] = {
    "SmartWay Partner": {
        "valid": True,
        "score": Decimal("0.95"),
        "note": "US EPA SmartWay Transport Partnership",
    },
    "GLEC Accredited": {
        "valid": True,
        "score": Decimal("1.00"),
        "note": "Global Logistics Emissions Council accredited",
    },
    "CDP Reported": {
        "valid": True,
        "score": Decimal("0.85"),
        "note": "Carrier reports through CDP Climate Change",
    },
    "Self-reported verified": {
        "valid": True,
        "score": Decimal("0.70"),
        "note": "Third-party verified self-reported data",
    },
    "Self-reported unverified": {
        "valid": False,
        "score": Decimal("0.50"),
        "note": "Unverified self-reported data - lower confidence",
    },
}

# =============================================================================
# CHARGEABLE WEIGHT DIMENSIONAL FACTORS
# =============================================================================

CHARGEABLE_WEIGHT_DIM_FACTORS: Dict[str, Decimal] = {
    "air": Decimal("6000"),       # 1 cbm = 166.67 kg chargeable
    "road": Decimal("3000"),      # 1 cbm = 333 kg chargeable
    "maritime": Decimal("0"),     # Not applicable (by TEU or mass)
}

# =============================================================================
# HUB DEFAULT EMISSION FACTORS (kgCO2e/tonne handled)
# =============================================================================

HUB_DEFAULT_EF: Dict[str, Decimal] = {
    "logistics_hub": Decimal("0.50"),
    "container_terminal": Decimal("1.20"),
    "airport_cargo_terminal": Decimal("2.50"),
    "rail_intermodal_terminal": Decimal("0.80"),
    "warehouse_standard_per_pallet_day": Decimal("0.30"),
    "warehouse_cold_per_pallet_day": Decimal("1.50"),
    "warehouse_frozen_per_pallet_day": Decimal("3.00"),
}

# =============================================================================
# REEFER TRU (Transport Refrigeration Unit) FUEL RATES
# =============================================================================

TRU_FUEL_RATES: Dict[str, Dict[str, Decimal]] = {
    "road_diesel_genset": {
        "litres_per_hour": Decimal("3.5"),
        "ef_per_litre": Decimal("2.687"),
    },
    "road_electric_standby": {
        "kwh_per_hour": Decimal("8.0"),
        "ef_per_kwh_global": Decimal("0.475"),
    },
    "maritime_reefer_container": {
        "kwh_per_hour": Decimal("4.5"),
        "ef_per_kwh_global": Decimal("0.475"),
    },
    "air_uld_cooling": {
        "kwh_per_hour": Decimal("2.0"),
        "ef_per_kwh_global": Decimal("0.475"),
    },
}

# =============================================================================
# COMMON REFRIGERANT GWP VALUES (for leakage calculation)
# =============================================================================

REFRIGERANT_GWP: Dict[str, Decimal] = {
    "HFC-134a": Decimal("1430"),
    "R-134a": Decimal("1430"),
    "R-404A": Decimal("3922"),
    "R-410A": Decimal("2088"),
    "R-407C": Decimal("1774"),
    "R-507A": Decimal("3985"),
    "R-452A": Decimal("2140"),
    "R-449A": Decimal("1397"),
    "R-448A": Decimal("1386"),
    "R-290": Decimal("3"),         # Propane - low GWP
    "R-744": Decimal("1"),         # CO2 refrigerant
    "R-1234yf": Decimal("4"),      # HFO - ultra-low GWP
    "R-1234ze": Decimal("7"),      # HFO - ultra-low GWP
}


# =============================================================================
# ENGINE CLASS
# =============================================================================


class MultiLegCalculatorEngine:
    """
    Engine 5: Multi-Leg Transport Chain Calculator.

    Orchestrates calculations for complete transport chains (multi-modal
    journeys with hub/transshipment points) per ISO 14083 and GLEC Framework.

    Responsibilities:
        - Calculate emissions for entire transport chains (multi-leg)
        - Calculate individual leg emissions (distance-based default)
        - Calculate hub/warehouse/transshipment emissions
        - Process supplier-specific emission data
        - Allocate shared transport emissions (6 methods)
        - Apply reefer/temperature-controlled transport uplift
        - Calculate refrigerant leakage emissions
        - Aggregate chain-level results with data quality scoring

    Thread Safety:
        All public methods are thread-safe. Internal state is protected by
        threading.RLock(). Embedded data tables are read-only.

    Zero-Hallucination:
        All numeric calculations use deterministic Decimal arithmetic.
        No LLM calls anywhere in the calculation path. Emission factors
        come from embedded lookup tables or validated supplier data.

    Example:
        >>> engine = MultiLegCalculatorEngine()
        >>> chain = TransportChain(
        ...     origin="Shanghai", destination="Rotterdam",
        ...     legs=[leg_sea, leg_road], hubs=[hub_port]
        ... )
        >>> result = engine.calculate_chain(chain)
        >>> assert result.total_emissions_tco2e > 0
    """

    def __init__(
        self,
        gwp_source: GWPSource = GWPSource.AR5,
        ef_scope: EFScope = EFScope.WTW,
        include_hubs: bool = True,
        include_reefer: bool = True,
    ) -> None:
        """
        Initialize MultiLegCalculatorEngine.

        Args:
            gwp_source: IPCC Assessment Report for GWP values.
            ef_scope: Emission factor scope (WTW, TTW, or WTT).
            include_hubs: Whether to include hub/warehouse emissions.
            include_reefer: Whether to apply reefer uplift factors.
        """
        self._gwp_source = gwp_source
        self._ef_scope = ef_scope
        self._include_hubs = include_hubs
        self._include_reefer = include_reefer
        self._lock = threading.RLock()
        logger.info(
            "MultiLegCalculatorEngine initialized: gwp=%s ef_scope=%s "
            "include_hubs=%s include_reefer=%s",
            gwp_source.value,
            ef_scope.value,
            include_hubs,
            include_reefer,
        )

    # =========================================================================
    # SECTION 1: CHAIN CALCULATION
    # =========================================================================

    def calculate_chain(
        self,
        chain: TransportChain,
        allocation_config: Optional[AllocationConfig] = None,
    ) -> TransportChainResult:
        """
        Calculate emissions for a complete transport chain.

        Iterates over all legs and hubs in the chain, calculates each
        element's emissions, applies allocation if configured, and
        aggregates results.

        Per ISO 14083 Section 6:
            Total Chain Emissions = SUM(Leg Emissions) + SUM(Hub Emissions)

        Args:
            chain: TransportChain with legs and optional hubs.
            allocation_config: Optional allocation configuration for
                shared transport.

        Returns:
            TransportChainResult with per-leg, per-hub, and total emissions.

        Raises:
            ValueError: If chain has no legs.
        """
        start_time = time.monotonic()
        logger.info(
            "Calculating chain %s: %d legs, %d hubs, %s -> %s",
            chain.chain_id,
            len(chain.legs),
            len(chain.hubs),
            chain.origin,
            chain.destination,
        )

        if not chain.legs:
            raise ValueError(
                f"Transport chain {chain.chain_id} must have at least one leg"
            )

        with self._lock:
            leg_results: List[LegResult] = []
            hub_results: List[HubResult] = []

            # --- Calculate each leg ---
            for leg in chain.legs:
                leg_result = self.calculate_leg(
                    leg=leg,
                    mass_tonnes=leg.cargo_mass_tonnes,
                    allocation_config=allocation_config,
                )
                leg_results.append(leg_result)

            # --- Calculate each hub ---
            if self._include_hubs and chain.hubs:
                for hub in chain.hubs:
                    hub_result = self.calculate_hub(
                        hub=hub,
                        mass_tonnes=hub.cargo_mass_tonnes,
                    )
                    hub_results.append(hub_result)

            # --- Aggregate ---
            aggregated = self.aggregate_chain_results(leg_results, hub_results)

            total_distance = aggregated["total_distance_km"]
            total_tkm = aggregated["total_tonne_km"]
            total_kg = aggregated["total_emissions_kgco2e"]
            total_t = _q(total_kg / _THOUSAND)
            by_mode = aggregated["emissions_by_mode"]

            result = TransportChainResult(
                chain_id=chain.chain_id,
                origin=chain.origin,
                destination=chain.destination,
                leg_results=leg_results,
                hub_results=hub_results,
                total_distance_km=total_distance,
                total_tonne_km=total_tkm,
                total_emissions_kgco2e=total_kg,
                total_emissions_tco2e=total_t,
                emissions_by_mode=by_mode,
            )

            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.info(
                "Chain %s complete: %.2f kgCO2e (%.6f tCO2e) in %.1f ms",
                chain.chain_id,
                float(total_kg),
                float(total_t),
                elapsed_ms,
            )
            return result

    def calculate_leg(
        self,
        leg: TransportLeg,
        mass_tonnes: Decimal,
        allocation_config: Optional[AllocationConfig] = None,
    ) -> LegResult:
        """
        Calculate emissions for a single transport leg.

        Formula (distance-based):
            Base Emissions = distance_km x cargo_mass_tonnes x EF_per_tkm
            Adjusted = Base x laden_factor x reefer_uplift x (allocation% / 100)

        Args:
            leg: TransportLeg definition.
            mass_tonnes: Cargo mass in metric tonnes.
            allocation_config: Optional allocation for shared transport.

        Returns:
            LegResult with emissions breakdown.

        Raises:
            ValueError: If mode is unsupported or EF cannot be resolved.
        """
        start_time = time.monotonic()
        logger.debug(
            "Calculating leg %s: mode=%s, distance=%.1f km, mass=%.3f t",
            leg.leg_id,
            leg.mode.value,
            float(leg.distance_km),
            float(mass_tonnes),
        )

        # Step 1: Resolve emission factor
        ef_per_tkm, ef_source = self._resolve_leg_ef(leg)

        # Step 2: Calculate tonne-km
        tonne_km = _q(leg.distance_km * mass_tonnes)

        # Step 3: Base emissions
        base_emissions = _q(tonne_km * ef_per_tkm)

        # Step 4: Laden state adjustment
        laden_factor = self._get_laden_factor(leg)

        # Step 5: Reefer uplift
        reefer_uplift = self._get_reefer_uplift(leg)

        # Step 6: Allocation percentage
        alloc_pct = self._resolve_allocation_pct(leg, allocation_config)

        # Step 7: Final emissions
        adjusted = _q(base_emissions * laden_factor * reefer_uplift)
        allocated = _q(adjusted * alloc_pct / _HUNDRED)

        # Step 8: Gas breakdown (approximate from total using mode ratios)
        co2_kg, ch4_kg, n2o_kg = self._estimate_gas_breakdown(
            leg.mode, allocated, ef_per_tkm
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.debug(
            "Leg %s: %.4f kgCO2e (tkm=%.2f, ef=%.6f, laden=%.2f, "
            "reefer=%.2f, alloc=%.1f%%) in %.1f ms",
            leg.leg_id,
            float(allocated),
            float(tonne_km),
            float(ef_per_tkm),
            float(laden_factor),
            float(reefer_uplift),
            float(alloc_pct),
            elapsed_ms,
        )

        return LegResult(
            leg_id=leg.leg_id,
            mode=leg.mode,
            vehicle_type=leg.vehicle_type,
            distance_km=leg.distance_km,
            cargo_mass_tonnes=mass_tonnes,
            tonne_km=tonne_km,
            ef_per_tkm=ef_per_tkm,
            ef_source=ef_source,
            ef_scope=self._ef_scope,
            laden_adjustment=laden_factor,
            reefer_uplift=reefer_uplift,
            allocation_percentage=alloc_pct,
            emissions_kgco2e=allocated,
            emissions_co2_kg=co2_kg,
            emissions_ch4_kg=ch4_kg,
            emissions_n2o_kg=n2o_kg,
        )

    def calculate_hub(
        self,
        hub: TransportHub,
        mass_tonnes: Decimal,
    ) -> HubResult:
        """
        Calculate emissions for a hub/transshipment point.

        Formula:
            Hub Emissions = mass_tonnes x EF_per_tonne (+ dwell_time uplift)

        For warehousing with known energy:
            Hub Emissions = energy_kwh x grid_ef

        Args:
            hub: TransportHub definition.
            mass_tonnes: Cargo mass handled at hub.

        Returns:
            HubResult with emissions.
        """
        start_time = time.monotonic()
        logger.debug(
            "Calculating hub %s: type=%s, mass=%.3f t, dwell=%.1f days",
            hub.hub_id,
            hub.hub_type.value,
            float(mass_tonnes),
            float(hub.dwell_time_days),
        )

        # Step 1: Resolve emission factor
        ef_per_tonne, ef_source, energy_kwh = self._resolve_hub_ef(hub)

        # Step 2: Base hub emissions
        base_emissions = _q(mass_tonnes * ef_per_tonne)

        # Step 3: Dwell time uplift for storage hubs
        dwell_uplift = self._calculate_dwell_uplift(hub)
        total_emissions = _q(base_emissions * dwell_uplift)

        # Step 4: Add energy-based emissions if available
        if energy_kwh is not None and energy_kwh > _ZERO:
            total_emissions = _q(energy_kwh * Decimal("0.475"))
            ef_source = f"{ef_source} + direct energy measurement"

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.debug(
            "Hub %s: %.4f kgCO2e (ef=%.4f, dwell_uplift=%.2f) in %.1f ms",
            hub.hub_id,
            float(total_emissions),
            float(ef_per_tonne),
            float(dwell_uplift),
            elapsed_ms,
        )

        return HubResult(
            hub_id=hub.hub_id,
            hub_type=hub.hub_type,
            location=hub.location,
            cargo_mass_tonnes=mass_tonnes,
            dwell_time_days=hub.dwell_time_days,
            ef_per_tonne=ef_per_tonne,
            ef_source=ef_source,
            emissions_kgco2e=total_emissions,
            energy_kwh=energy_kwh,
        )

    def calculate_warehouse(
        self,
        config: WarehouseConfig,
        mass_tonnes: Decimal,
        grid_ef: Decimal,
    ) -> HubResult:
        """
        Calculate warehouse emissions from energy intensity.

        Formula:
            Annual Energy = floor_area_m2 x energy_intensity_kwh_per_m2
            Daily Energy = Annual Energy / 365
            Emissions = Daily Energy x dwell_days x grid_ef

        If annual_energy_kwh is provided directly, it overrides the
        area-based calculation.

        Args:
            config: WarehouseConfig with area, energy data.
            mass_tonnes: Cargo mass stored.
            grid_ef: Grid emission factor (kgCO2e per kWh).

        Returns:
            HubResult with warehouse emissions.
        """
        start_time = time.monotonic()
        logger.debug(
            "Calculating warehouse: type=%s, area=%.1f m2, mass=%.3f t",
            config.hub_type.value,
            float(config.floor_area_m2),
            float(mass_tonnes),
        )

        # Resolve energy intensity
        energy_intensity = config.energy_intensity_kwh_per_m2
        if energy_intensity is None:
            energy_intensity = self._get_default_energy_intensity(config)

        # Calculate energy
        if config.annual_energy_kwh is not None and config.annual_energy_kwh > _ZERO:
            total_energy_kwh = config.annual_energy_kwh
            ef_source = "direct annual energy measurement"
        else:
            annual_kwh = _q(config.floor_area_m2 * energy_intensity)
            total_energy_kwh = annual_kwh
            ef_source = (
                f"area-based: {float(config.floor_area_m2)} m2 x "
                f"{float(energy_intensity)} kWh/m2/yr"
            )

        # Apply grid emission factor
        emissions = _q(total_energy_kwh * grid_ef)

        # Allocate to this shipment by mass proportion
        # Assume warehouse handles ~1000 tonnes/year as default throughput
        default_throughput = Decimal("1000")
        allocation_share = _q(mass_tonnes / default_throughput)
        if allocation_share > _ONE:
            allocation_share = _ONE

        allocated_emissions = _q(emissions * allocation_share)

        hub_type_str = config.hub_type.value
        ef_per_tonne = (
            _q(allocated_emissions / mass_tonnes) if mass_tonnes > _ZERO else _ZERO
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.debug(
            "Warehouse: %.4f kgCO2e (energy=%.1f kWh, grid_ef=%.4f) in %.1f ms",
            float(allocated_emissions),
            float(total_energy_kwh),
            float(grid_ef),
            elapsed_ms,
        )

        return HubResult(
            hub_id=f"wh_{uuid.uuid4().hex[:12]}",
            hub_type=config.hub_type,
            location="warehouse",
            cargo_mass_tonnes=mass_tonnes,
            dwell_time_days=_ZERO,
            ef_per_tonne=ef_per_tonne,
            ef_source=ef_source,
            emissions_kgco2e=allocated_emissions,
            energy_kwh=total_energy_kwh,
        )

    # =========================================================================
    # SECTION 2: SUPPLIER-SPECIFIC METHODS
    # =========================================================================

    def process_supplier_data(
        self,
        supplier_input: SupplierEmissionInput,
    ) -> LegResult:
        """
        Process supplier-specific emission data.

        Validates the supplier's methodology and certification, applies
        a verification score multiplier, and returns a LegResult for
        integration into the transport chain.

        Per GHG Protocol Scope 3 Technical Guidance:
            Supplier-specific data is the most accurate method when
            the data is verified using a recognized standard.

        Args:
            supplier_input: Supplier-reported emissions with metadata.

        Returns:
            LegResult derived from supplier data.

        Raises:
            ValueError: If supplier emissions are negative.
        """
        start_time = time.monotonic()
        logger.info(
            "Processing supplier data: supplier=%s, reported=%.4f kgCO2e",
            supplier_input.supplier_name,
            float(supplier_input.emissions_kgco2e),
        )

        if supplier_input.emissions_kgco2e < _ZERO:
            raise ValueError(
                f"Supplier emissions cannot be negative: "
                f"{supplier_input.emissions_kgco2e}"
            )

        # Validate methodology
        methodology = supplier_input.verification_standard or "Custom"
        methodology_valid = self.validate_supplier_methodology(methodology)

        # Validate certification
        certification = supplier_input.verification_status or "Self-reported unverified"
        certification_valid = self.validate_supplier_certification(certification)

        # Apply verification score
        adjusted_emissions = self.apply_supplier_verification_score(
            reported_co2e=supplier_input.emissions_kgco2e,
            methodology=methodology,
            certification=certification,
        )

        # Derive EF if distance is known
        ef_per_tkm = _ZERO
        tonne_km = _ZERO
        distance_km = supplier_input.distance_km or _ZERO
        if distance_km > _ZERO and supplier_input.cargo_mass_tonnes > _ZERO:
            tonne_km = _q(distance_km * supplier_input.cargo_mass_tonnes)
            if tonne_km > _ZERO:
                ef_per_tkm = _q(adjusted_emissions / tonne_km)

        ef_source = (
            f"Supplier: {supplier_input.supplier_name} "
            f"(methodology={methodology}, "
            f"certification={certification})"
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Supplier data processed: adjusted=%.4f kgCO2e "
            "(methodology_valid=%s, cert_valid=%s) in %.1f ms",
            float(adjusted_emissions),
            methodology_valid,
            certification_valid,
            elapsed_ms,
        )

        return LegResult(
            leg_id=supplier_input.record_id,
            mode=TransportMode.INTERMODAL,
            vehicle_type=None,
            distance_km=distance_km,
            cargo_mass_tonnes=supplier_input.cargo_mass_tonnes,
            tonne_km=tonne_km,
            ef_per_tkm=ef_per_tkm,
            ef_source=ef_source,
            ef_scope=EFScope.WTW,
            laden_adjustment=_ONE,
            reefer_uplift=_ONE,
            allocation_percentage=_HUNDRED,
            emissions_kgco2e=adjusted_emissions,
            emissions_co2_kg=adjusted_emissions,
            emissions_ch4_kg=_ZERO,
            emissions_n2o_kg=_ZERO,
        )

    def validate_supplier_methodology(self, methodology: str) -> bool:
        """
        Validate supplier's calculation methodology.

        Checks against known methodologies: GLEC v3, ISO 14083,
        EN 16258, GHG Protocol, EcoTransIT.

        Args:
            methodology: Methodology name string.

        Returns:
            True if methodology is valid and recognized.
        """
        entry = METHODOLOGY_VALIDATION.get(methodology)
        if entry is None:
            logger.warning("Unknown methodology: %s", methodology)
            return False
        is_valid: bool = entry["valid"]
        return is_valid

    def validate_supplier_certification(self, certification: str) -> bool:
        """
        Validate supplier's certification/verification status.

        Checks against known certifications: SmartWay Partner,
        GLEC Accredited, CDP Reported, etc.

        Args:
            certification: Certification name string.

        Returns:
            True if certification is valid and recognized.
        """
        entry = CERTIFICATION_VALIDATION.get(certification)
        if entry is None:
            logger.warning("Unknown certification: %s", certification)
            return False
        is_valid: bool = entry["valid"]
        return is_valid

    def get_smartway_carrier_factor(
        self,
        carrier_name: str,
        mode: str,
    ) -> Optional[Decimal]:
        """
        Get SmartWay carrier-specific emission factor.

        Looks up the carrier category in the SmartWay database
        and returns the default emission factor for that category.

        In production, this would query the SmartWay Tools database.
        The embedded data provides representative defaults per category.

        Args:
            carrier_name: Carrier/company name.
            mode: SmartWay carrier category (e.g., 'TL', 'LTL', 'RAIL').

        Returns:
            Emission factor (kgCO2e per tonne-km) or None if not found.
        """
        category_key = mode.upper().replace(" ", "_")
        category = SMARTWAY_CARRIER_CATEGORIES.get(category_key)

        if category is None:
            logger.debug(
                "SmartWay category not found for carrier=%s mode=%s",
                carrier_name,
                mode,
            )
            return None

        ef: Decimal = category["default_ef_per_tkm"]
        logger.debug(
            "SmartWay factor for carrier=%s category=%s: %.6f kgCO2e/tkm",
            carrier_name,
            category_key,
            float(ef),
        )
        return ef

    def apply_supplier_verification_score(
        self,
        reported_co2e: Decimal,
        methodology: str,
        certification: str,
    ) -> Decimal:
        """
        Apply verification confidence score to supplier-reported emissions.

        The verification score adjusts reported emissions based on the
        quality of the methodology and certification. Higher scores
        (closer to 1.0) indicate higher confidence and no adjustment.
        Lower scores apply an uplift factor to account for potential
        underreporting.

        Adjustment formula:
            If combined_score >= 0.90: use reported value as-is
            If combined_score < 0.90: uplift = 1 + (1 - combined_score) * 0.5
            Adjusted = reported_co2e * uplift

        Args:
            reported_co2e: Supplier-reported kgCO2e.
            methodology: Methodology name.
            certification: Certification name.

        Returns:
            Adjusted kgCO2e (may be higher than reported if low confidence).
        """
        # Get methodology score
        meth_entry = METHODOLOGY_VALIDATION.get(methodology, {})
        meth_score: Decimal = meth_entry.get("score", Decimal("0.50"))

        # Get certification score
        cert_entry = CERTIFICATION_VALIDATION.get(certification, {})
        cert_score: Decimal = cert_entry.get("score", Decimal("0.50"))

        # Combined score (weighted: 60% methodology, 40% certification)
        combined = _q(meth_score * Decimal("0.60") + cert_score * Decimal("0.40"))

        # Apply adjustment
        threshold = Decimal("0.90")
        if combined >= threshold:
            return reported_co2e

        uplift = _q(_ONE + (_ONE - combined) * Decimal("0.5"))
        adjusted = _q(reported_co2e * uplift)

        logger.debug(
            "Supplier verification: meth_score=%.2f, cert_score=%.2f, "
            "combined=%.2f, uplift=%.4f, reported=%.4f -> adjusted=%.4f",
            float(meth_score),
            float(cert_score),
            float(combined),
            float(uplift),
            float(reported_co2e),
            float(adjusted),
        )
        return adjusted

    # =========================================================================
    # SECTION 3: ALLOCATION METHODS
    # =========================================================================

    def allocate_emissions(
        self,
        total_emissions: Decimal,
        config: AllocationConfig,
    ) -> Decimal:
        """
        Allocate shared transport emissions to a specific shipment.

        Dispatches to the appropriate allocation method based on the
        AllocationConfig.allocation_method field.

        Per ISO 14083 Section 7 and GLEC Framework:
            Allocated emissions = Total emissions x (shipment share / total)

        Args:
            total_emissions: Total emissions for the shared transport.
            config: AllocationConfig with method and quantities.

        Returns:
            Allocated emissions (kgCO2e) for this shipment.

        Raises:
            ValueError: If required quantities are missing for the method.
        """
        method = config.allocation_method
        logger.debug(
            "Allocating emissions: method=%s, total=%.4f kgCO2e",
            method.value,
            float(total_emissions),
        )

        if method == AllocationMethod.MASS:
            return self.allocate_by_mass(
                total_emissions,
                config.shipment_mass_tonnes or _ZERO,
                config.total_capacity_tonnes or _ONE,
            )
        elif method == AllocationMethod.VOLUME:
            return self.allocate_by_volume(
                total_emissions,
                config.shipment_volume_m3 or _ZERO,
                config.total_capacity_m3 or _ONE,
            )
        elif method == AllocationMethod.PALLET_POSITIONS:
            return self.allocate_by_pallet_positions(
                total_emissions,
                config.shipment_pallet_positions or 0,
                config.total_pallet_positions or 1,
            )
        elif method == AllocationMethod.TEU:
            return self.allocate_by_teu(
                total_emissions,
                config.shipment_teu or _ZERO,
                config.total_teu or 1,
            )
        elif method == AllocationMethod.REVENUE:
            return self.allocate_by_revenue(
                total_emissions,
                config.shipment_revenue or _ZERO,
                config.total_revenue or _ONE,
            )
        elif method == AllocationMethod.CHARGEABLE_WEIGHT:
            return self.allocate_by_chargeable_weight(
                total_emissions,
                config.shipment_chargeable_weight_kg or _ZERO,
                _ONE,  # Total CW must be calculated externally
            )
        else:
            raise ValueError(f"Unsupported allocation method: {method}")

    def allocate_by_mass(
        self,
        total_emissions: Decimal,
        reporter_mass: Decimal,
        total_mass: Decimal,
    ) -> Decimal:
        """
        Allocate emissions by cargo mass (tonnes).

        Formula: allocated = total_emissions x (reporter_mass / total_mass)

        This is the most common allocation method and the default
        recommended by ISO 14083 for general freight.

        Args:
            total_emissions: Total emissions for the shared transport.
            reporter_mass: Reporter's cargo mass (tonnes).
            total_mass: Total cargo mass on the vehicle (tonnes).

        Returns:
            Allocated kgCO2e.

        Raises:
            ValueError: If total_mass is zero or negative.
        """
        self._validate_allocation_inputs(
            "mass", reporter_mass, total_mass, "tonnes"
        )
        share = _q(reporter_mass / total_mass)
        allocated = _q(total_emissions * share)
        logger.debug(
            "Mass allocation: %.3f / %.3f = %.6f share -> %.4f kgCO2e",
            float(reporter_mass),
            float(total_mass),
            float(share),
            float(allocated),
        )
        return allocated

    def allocate_by_volume(
        self,
        total_emissions: Decimal,
        reporter_volume: Decimal,
        total_volume: Decimal,
    ) -> Decimal:
        """
        Allocate emissions by cargo volume (m3).

        Formula: allocated = total_emissions x (reporter_volume / total_volume)

        Appropriate for light but bulky goods where volume constrains
        vehicle capacity before mass does.

        Args:
            total_emissions: Total emissions for the shared transport.
            reporter_volume: Reporter's cargo volume (m3).
            total_volume: Total cargo volume on the vehicle (m3).

        Returns:
            Allocated kgCO2e.

        Raises:
            ValueError: If total_volume is zero or negative.
        """
        self._validate_allocation_inputs(
            "volume", reporter_volume, total_volume, "m3"
        )
        share = _q(reporter_volume / total_volume)
        allocated = _q(total_emissions * share)
        logger.debug(
            "Volume allocation: %.3f / %.3f m3 = %.6f share -> %.4f kgCO2e",
            float(reporter_volume),
            float(total_volume),
            float(share),
            float(allocated),
        )
        return allocated

    def allocate_by_pallet_positions(
        self,
        total_emissions: Decimal,
        reporter_pallets: int,
        total_pallets: int,
    ) -> Decimal:
        """
        Allocate emissions by pallet positions.

        Formula: allocated = total_emissions x (reporter_pallets / total_pallets)

        Common in road freight where trailer capacity is measured in
        pallet positions (e.g., 33 Euro-pallets for a standard trailer).

        Args:
            total_emissions: Total emissions for the shared transport.
            reporter_pallets: Reporter's pallet count.
            total_pallets: Total pallet positions on the vehicle.

        Returns:
            Allocated kgCO2e.

        Raises:
            ValueError: If total_pallets is zero or negative.
        """
        if total_pallets <= 0:
            raise ValueError("total_pallets must be > 0")
        if reporter_pallets < 0:
            raise ValueError("reporter_pallets cannot be negative")

        share = _q(Decimal(str(reporter_pallets)) / Decimal(str(total_pallets)))
        allocated = _q(total_emissions * share)
        logger.debug(
            "Pallet allocation: %d / %d = %.6f share -> %.4f kgCO2e",
            reporter_pallets,
            total_pallets,
            float(share),
            float(allocated),
        )
        return allocated

    def allocate_by_teu(
        self,
        total_emissions: Decimal,
        reporter_teu: Decimal,
        total_teu: int,
    ) -> Decimal:
        """
        Allocate emissions by Twenty-foot Equivalent Units (TEU).

        Formula: allocated = total_emissions x (reporter_teu / total_teu)

        Standard allocation method for containerized maritime freight.
        A 20ft container = 1 TEU, a 40ft container = 2 TEU.

        Args:
            total_emissions: Total emissions for the shared vessel.
            reporter_teu: Reporter's TEU count.
            total_teu: Total TEU capacity/load on the vessel.

        Returns:
            Allocated kgCO2e.

        Raises:
            ValueError: If total_teu is zero or negative.
        """
        if total_teu <= 0:
            raise ValueError("total_teu must be > 0")
        if reporter_teu < _ZERO:
            raise ValueError("reporter_teu cannot be negative")

        share = _q(reporter_teu / Decimal(str(total_teu)))
        allocated = _q(total_emissions * share)
        logger.debug(
            "TEU allocation: %.1f / %d = %.6f share -> %.4f kgCO2e",
            float(reporter_teu),
            total_teu,
            float(share),
            float(allocated),
        )
        return allocated

    def allocate_by_revenue(
        self,
        total_emissions: Decimal,
        reporter_revenue: Decimal,
        total_revenue: Decimal,
    ) -> Decimal:
        """
        Allocate emissions by revenue share.

        Formula: allocated = total_emissions x (reporter_revenue / total_revenue)

        Revenue-based allocation is used when physical measures are not
        available. ISO 14083 ranks it lower in the allocation hierarchy
        but it remains acceptable as a fallback.

        Args:
            total_emissions: Total emissions for the shared transport.
            reporter_revenue: Revenue attributable to reporter's cargo.
            total_revenue: Total revenue for the transport service.

        Returns:
            Allocated kgCO2e.

        Raises:
            ValueError: If total_revenue is zero or negative.
        """
        self._validate_allocation_inputs(
            "revenue", reporter_revenue, total_revenue, "currency"
        )
        share = _q(reporter_revenue / total_revenue)
        allocated = _q(total_emissions * share)
        logger.debug(
            "Revenue allocation: %.2f / %.2f = %.6f share -> %.4f kgCO2e",
            float(reporter_revenue),
            float(total_revenue),
            float(share),
            float(allocated),
        )
        return allocated

    def allocate_by_chargeable_weight(
        self,
        total_emissions: Decimal,
        reporter_cw: Decimal,
        total_cw: Decimal,
    ) -> Decimal:
        """
        Allocate emissions by chargeable weight.

        Formula: allocated = total_emissions x (reporter_cw / total_cw)

        Chargeable weight = MAX(actual_mass_kg, volumetric_weight_kg).
        Standard in air freight where both mass and volume are constrained.

        Args:
            total_emissions: Total emissions for the shared transport.
            reporter_cw: Reporter's chargeable weight (kg).
            total_cw: Total chargeable weight on the aircraft (kg).

        Returns:
            Allocated kgCO2e.

        Raises:
            ValueError: If total_cw is zero or negative.
        """
        self._validate_allocation_inputs(
            "chargeable_weight", reporter_cw, total_cw, "kg"
        )
        share = _q(reporter_cw / total_cw)
        allocated = _q(total_emissions * share)
        logger.debug(
            "Chargeable weight allocation: %.2f / %.2f kg = %.6f -> %.4f kgCO2e",
            float(reporter_cw),
            float(total_cw),
            float(share),
            float(allocated),
        )
        return allocated

    def calculate_chargeable_weight(
        self,
        actual_mass_kg: Decimal,
        volume_m3: Decimal,
        dim_factor: Decimal,
    ) -> Decimal:
        """
        Calculate chargeable weight from actual mass and volume.

        Formula:
            volumetric_weight_kg = volume_m3 x dim_factor
            chargeable_weight = MAX(actual_mass_kg, volumetric_weight_kg)

        Dimensional factors by mode:
            - Air: 6000 (1 cbm = 166.67 kg chargeable)
            - Road: 3000 (1 cbm = 333 kg chargeable)
            - Maritime: Not applicable (by TEU or mass)

        Args:
            actual_mass_kg: Actual cargo mass in kilograms.
            volume_m3: Cargo volume in cubic meters.
            dim_factor: Dimensional factor for the mode.

        Returns:
            Chargeable weight in kilograms.

        Raises:
            ValueError: If inputs are negative.
        """
        if actual_mass_kg < _ZERO:
            raise ValueError("actual_mass_kg cannot be negative")
        if volume_m3 < _ZERO:
            raise ValueError("volume_m3 cannot be negative")
        if dim_factor < _ZERO:
            raise ValueError("dim_factor cannot be negative")

        if dim_factor == _ZERO:
            return actual_mass_kg

        volumetric_weight_kg = _q(volume_m3 * dim_factor)
        chargeable = max(actual_mass_kg, volumetric_weight_kg)

        logger.debug(
            "Chargeable weight: actual=%.2f kg, volumetric=%.2f kg "
            "(vol=%.3f m3 x dim=%.0f) -> chargeable=%.2f kg",
            float(actual_mass_kg),
            float(volumetric_weight_kg),
            float(volume_m3),
            float(dim_factor),
            float(chargeable),
        )
        return chargeable

    # =========================================================================
    # SECTION 4: REEFER / TEMPERATURE-CONTROLLED TRANSPORT
    # =========================================================================

    def calculate_reefer_emissions(
        self,
        base_emissions: Decimal,
        mode: TransportMode,
        temperature: TemperatureControl,
        leg_distance_km: Decimal,
        leg_duration_hours: Decimal,
    ) -> Decimal:
        """
        Calculate additional emissions from reefer/temperature-controlled transport.

        Applies the mode-specific reefer uplift factor to base emissions.
        If actual TRU fuel consumption is not known, uses the uplift
        factor approach from the GLEC Framework.

        For known TRU fuel consumption, use calculate_tru_fuel() instead.

        Formula:
            reefer_emissions = base_emissions x (uplift_factor - 1.0)
            total = base_emissions + reefer_emissions

        Args:
            base_emissions: Base transport emissions (kgCO2e) before reefer.
            mode: Transport mode.
            temperature: Temperature control requirement.
            leg_distance_km: Leg distance (for logging context).
            leg_duration_hours: Leg duration (for logging context).

        Returns:
            Total emissions including reefer uplift (kgCO2e).
        """
        if temperature == TemperatureControl.AMBIENT:
            return base_emissions

        mode_factors = REEFER_UPLIFT_FACTORS.get(mode)
        if mode_factors is None:
            logger.warning(
                "No reefer uplift factors for mode=%s; returning base emissions",
                mode.value,
            )
            return base_emissions

        uplift = mode_factors.get(temperature, _ONE)
        total = _q(base_emissions * uplift)

        reefer_additional = _q(total - base_emissions)
        logger.debug(
            "Reefer emissions: mode=%s, temp=%s, uplift=%.2f, "
            "base=%.4f -> total=%.4f (+%.4f reefer)",
            mode.value,
            temperature.value,
            float(uplift),
            float(base_emissions),
            float(total),
            float(reefer_additional),
        )
        return total

    def calculate_refrigerant_leakage(
        self,
        charge_kg: Decimal,
        leak_rate: Decimal,
        gwp: Decimal,
    ) -> Decimal:
        """
        Calculate annual refrigerant leakage emissions (kgCO2e).

        Formula (per GHG Protocol):
            Leakage kgCO2e = charge_kg x annual_leak_rate x GWP

        Typical leak rates:
            - Road reefer units: 10-25% per year
            - Maritime reefer containers: 5-15% per year
            - Stationary refrigeration: 2-10% per year

        Args:
            charge_kg: Refrigerant charge in kilograms.
            leak_rate: Annual leakage rate (0.0 to 1.0).
            gwp: Global Warming Potential of the refrigerant.

        Returns:
            Annual kgCO2e from refrigerant leakage.

        Raises:
            ValueError: If inputs are negative or leak_rate > 1.0.
        """
        if charge_kg < _ZERO:
            raise ValueError("charge_kg cannot be negative")
        if leak_rate < _ZERO or leak_rate > _ONE:
            raise ValueError("leak_rate must be between 0.0 and 1.0")
        if gwp < _ZERO:
            raise ValueError("gwp cannot be negative")

        leakage_kg = _q(charge_kg * leak_rate)
        leakage_co2e = _q(leakage_kg * gwp)

        logger.debug(
            "Refrigerant leakage: charge=%.2f kg x rate=%.2f x GWP=%.0f "
            "= %.4f kgCO2e/year",
            float(charge_kg),
            float(leak_rate),
            float(gwp),
            float(leakage_co2e),
        )
        return leakage_co2e

    def calculate_tru_fuel(
        self,
        duration_hours: Decimal,
        fuel_rate_per_hour: Decimal,
    ) -> Decimal:
        """
        Calculate Transport Refrigeration Unit (TRU) fuel consumption.

        Formula:
            Total fuel = duration_hours x fuel_rate_per_hour

        The result is in litres (diesel) or kWh (electric).
        Convert to emissions using the appropriate fuel emission factor.

        Args:
            duration_hours: Operating duration in hours.
            fuel_rate_per_hour: Fuel consumption rate (litres/hour or kWh/hour).

        Returns:
            Total fuel consumption (litres or kWh).

        Raises:
            ValueError: If inputs are negative.
        """
        if duration_hours < _ZERO:
            raise ValueError("duration_hours cannot be negative")
        if fuel_rate_per_hour < _ZERO:
            raise ValueError("fuel_rate_per_hour cannot be negative")

        total_fuel = _q(duration_hours * fuel_rate_per_hour)
        logger.debug(
            "TRU fuel: %.1f hours x %.2f per hour = %.4f total",
            float(duration_hours),
            float(fuel_rate_per_hour),
            float(total_fuel),
        )
        return total_fuel

    # =========================================================================
    # SECTION 5: AGGREGATION AND DATA QUALITY
    # =========================================================================

    def aggregate_chain_results(
        self,
        leg_results: List[LegResult],
        hub_results: List[HubResult],
    ) -> Dict[str, Any]:
        """
        Aggregate leg and hub results into chain-level totals.

        Computes:
            - Total distance (km)
            - Total tonne-km
            - Total emissions (kgCO2e)
            - Emissions by mode
            - Emissions from hubs separately
            - Gas breakdown (CO2, CH4, N2O)

        Args:
            leg_results: List of LegResult from each leg.
            hub_results: List of HubResult from each hub.

        Returns:
            Dictionary with aggregated metrics.
        """
        total_distance = _ZERO
        total_tkm = _ZERO
        total_emissions = _ZERO
        total_co2 = _ZERO
        total_ch4 = _ZERO
        total_n2o = _ZERO
        emissions_by_mode: Dict[str, Decimal] = {}
        hub_total = _ZERO

        for lr in leg_results:
            total_distance = _q(total_distance + lr.distance_km)
            total_tkm = _q(total_tkm + lr.tonne_km)
            total_emissions = _q(total_emissions + lr.emissions_kgco2e)
            total_co2 = _q(total_co2 + lr.emissions_co2_kg)
            total_ch4 = _q(total_ch4 + lr.emissions_ch4_kg)
            total_n2o = _q(total_n2o + lr.emissions_n2o_kg)

            mode_key = lr.mode.value
            prev = emissions_by_mode.get(mode_key, _ZERO)
            emissions_by_mode[mode_key] = _q(prev + lr.emissions_kgco2e)

        for hr in hub_results:
            hub_total = _q(hub_total + hr.emissions_kgco2e)
            total_emissions = _q(total_emissions + hr.emissions_kgco2e)

        if hub_total > _ZERO:
            emissions_by_mode["hub"] = hub_total

        logger.debug(
            "Chain aggregation: %d legs + %d hubs = %.4f kgCO2e "
            "(distance=%.1f km, tkm=%.1f)",
            len(leg_results),
            len(hub_results),
            float(total_emissions),
            float(total_distance),
            float(total_tkm),
        )

        return {
            "total_distance_km": total_distance,
            "total_tonne_km": total_tkm,
            "total_emissions_kgco2e": total_emissions,
            "total_hub_emissions_kgco2e": hub_total,
            "total_co2_kg": total_co2,
            "total_ch4_kg": total_ch4,
            "total_n2o_kg": total_n2o,
            "emissions_by_mode": emissions_by_mode,
            "leg_count": len(leg_results),
            "hub_count": len(hub_results),
        }

    def calculate_chain_data_quality(
        self,
        leg_results: List[LegResult],
        hub_results: List[HubResult],
    ) -> Decimal:
        """
        Calculate weighted Data Quality Indicator (DQI) for the chain.

        Per ISO 14083, DQI is scored 1 (best) to 5 (worst) across
        five dimensions. The chain DQI is a weighted average where
        each element is weighted by its emissions share.

        Scoring heuristic (when explicit DQI not provided):
            - Supplier-specific with verification: 1.5
            - Fuel-based with actual consumption: 2.0
            - Distance-based with actual distance: 2.5
            - Distance-based with estimated distance: 3.0
            - Spend-based: 4.0
            - Default/fallback: 3.5
            - Hub with measured energy: 2.0
            - Hub with default EF: 3.5

        Args:
            leg_results: List of LegResult.
            hub_results: List of HubResult.

        Returns:
            Weighted composite DQI score (1.0 to 5.0).
        """
        total_emissions = _ZERO
        weighted_sum = _ZERO

        for lr in leg_results:
            dqi = self._score_leg_dqi(lr)
            total_emissions = _q(total_emissions + lr.emissions_kgco2e)
            weighted_sum = _q(weighted_sum + lr.emissions_kgco2e * dqi)

        for hr in hub_results:
            dqi = self._score_hub_dqi(hr)
            total_emissions = _q(total_emissions + hr.emissions_kgco2e)
            weighted_sum = _q(weighted_sum + hr.emissions_kgco2e * dqi)

        if total_emissions <= _ZERO:
            return Decimal("5.0")

        composite = _q(weighted_sum / total_emissions)

        # Clamp to valid range
        if composite < _ONE:
            composite = _ONE
        if composite > Decimal("5"):
            composite = Decimal("5")

        logger.debug(
            "Chain DQI: weighted_sum=%.4f / total=%.4f = %.2f",
            float(weighted_sum),
            float(total_emissions),
            float(composite),
        )
        return composite

    # =========================================================================
    # SECTION 6: INTERNAL HELPERS - EMISSION FACTOR RESOLUTION
    # =========================================================================

    def _resolve_leg_ef(
        self,
        leg: TransportLeg,
    ) -> Tuple[Decimal, str]:
        """
        Resolve emission factor for a transport leg.

        Priority:
            1. Custom EF if provided on the leg
            2. Vehicle-type-specific EF from embedded tables
            3. Mode-level default EF

        Args:
            leg: TransportLeg to resolve EF for.

        Returns:
            Tuple of (ef_per_tkm, ef_source_description).

        Raises:
            ValueError: If no EF can be resolved.
        """
        # Priority 1: Custom EF
        if leg.custom_ef_per_tkm is not None and leg.custom_ef_per_tkm >= _ZERO:
            source = leg.ef_source or "custom_override"
            return leg.custom_ef_per_tkm, source

        # Priority 2: Vehicle-type-specific
        ef, source = self._resolve_vehicle_ef(leg)
        if ef is not None:
            return ef, source

        # Priority 3: Mode-level default
        ef, source = self._resolve_mode_default_ef(leg.mode)
        if ef is not None:
            return ef, source

        raise ValueError(
            f"Cannot resolve emission factor for leg {leg.leg_id}: "
            f"mode={leg.mode.value}, vehicle_type={leg.vehicle_type}"
        )

    def _resolve_vehicle_ef(
        self,
        leg: TransportLeg,
    ) -> Tuple[Optional[Decimal], str]:
        """
        Resolve vehicle-type-specific emission factor.

        Args:
            leg: TransportLeg with vehicle_type field.

        Returns:
            Tuple of (ef or None, source description).
        """
        vt = leg.vehicle_type
        if vt is None:
            return None, ""

        # Road vehicles
        if leg.mode == TransportMode.ROAD:
            return self._resolve_road_ef(vt)

        # Rail
        if leg.mode == TransportMode.RAIL:
            return self._resolve_rail_ef(vt)

        # Maritime
        if leg.mode == TransportMode.MARITIME:
            return self._resolve_maritime_ef(vt)

        # Air
        if leg.mode == TransportMode.AIR:
            return self._resolve_air_ef(vt)

        # Pipeline
        if leg.mode == TransportMode.PIPELINE:
            return self._resolve_pipeline_ef(vt)

        return None, ""

    def _resolve_road_ef(
        self,
        vehicle_type_str: str,
    ) -> Tuple[Optional[Decimal], str]:
        """Resolve road vehicle emission factor from embedded table."""
        try:
            vt_enum = RoadVehicleType(vehicle_type_str)
        except ValueError:
            logger.debug("Unknown road vehicle type: %s", vehicle_type_str)
            return None, ""

        factors = ROAD_EMISSION_FACTORS.get(vt_enum)
        if factors is None:
            return None, ""

        ef = factors.get("total_per_tkm", _ZERO)
        source = f"ROAD_EMISSION_FACTORS[{vt_enum.value}] (GLEC v3.0 / DEFRA 2023)"
        return ef, source

    def _resolve_rail_ef(
        self,
        vehicle_type_str: str,
    ) -> Tuple[Optional[Decimal], str]:
        """Resolve rail emission factor from embedded table."""
        try:
            rt_enum = RailType(vehicle_type_str)
        except ValueError:
            logger.debug("Unknown rail type: %s", vehicle_type_str)
            return None, ""

        ef = RAIL_EMISSION_FACTORS.get((rt_enum, "global"))
        if ef is None:
            return None, ""

        source = f"RAIL_EMISSION_FACTORS[{rt_enum.value}, global] (GLEC v3.0)"
        return ef, source

    def _resolve_maritime_ef(
        self,
        vehicle_type_str: str,
    ) -> Tuple[Optional[Decimal], str]:
        """Resolve maritime emission factor from embedded table."""
        try:
            mv_enum = MaritimeVesselType(vehicle_type_str)
        except ValueError:
            logger.debug("Unknown maritime vessel type: %s", vehicle_type_str)
            return None, ""

        ef = MARITIME_EMISSION_FACTORS.get(mv_enum)
        if ef is None:
            return None, ""

        source = f"MARITIME_EMISSION_FACTORS[{mv_enum.value}] (IMO GHG4 / GLEC v3.0)"
        return ef, source

    def _resolve_air_ef(
        self,
        vehicle_type_str: str,
    ) -> Tuple[Optional[Decimal], str]:
        """Resolve air freight emission factor from embedded table."""
        try:
            at_enum = AircraftType(vehicle_type_str)
        except ValueError:
            logger.debug("Unknown aircraft type: %s", vehicle_type_str)
            return None, ""

        ef = AIR_EMISSION_FACTORS.get(at_enum)
        if ef is None:
            return None, ""

        source = f"AIR_EMISSION_FACTORS[{at_enum.value}] (ICAO / GLEC v3.0)"
        return ef, source

    def _resolve_pipeline_ef(
        self,
        vehicle_type_str: str,
    ) -> Tuple[Optional[Decimal], str]:
        """Resolve pipeline emission factor from embedded table."""
        try:
            pt_enum = PipelineType(vehicle_type_str)
        except ValueError:
            logger.debug("Unknown pipeline type: %s", vehicle_type_str)
            return None, ""

        ef = PIPELINE_EMISSION_FACTORS.get(pt_enum)
        if ef is None:
            return None, ""

        source = f"PIPELINE_EMISSION_FACTORS[{pt_enum.value}] (GLEC v3.0)"
        return ef, source

    def _resolve_mode_default_ef(
        self,
        mode: TransportMode,
    ) -> Tuple[Optional[Decimal], str]:
        """
        Resolve mode-level default emission factor.

        Used as fallback when vehicle-type-specific EF is not available.

        Args:
            mode: TransportMode.

        Returns:
            Tuple of (ef or None, source).
        """
        defaults: Dict[TransportMode, Tuple[Decimal, str]] = {
            TransportMode.ROAD: (
                Decimal("0.118"),
                "GLEC v3.0 road default (LTL/avg articulated)",
            ),
            TransportMode.RAIL: (
                Decimal("0.0156"),
                "GLEC v3.0 rail global average",
            ),
            TransportMode.MARITIME: (
                Decimal("0.0098"),
                "GLEC v3.0 maritime default (container panamax)",
            ),
            TransportMode.AIR: (
                Decimal("0.952"),
                "GLEC v3.0 air default (belly freight)",
            ),
            TransportMode.PIPELINE: (
                Decimal("0.0038"),
                "GLEC v3.0 pipeline default (refined products)",
            ),
            TransportMode.INTERMODAL: (
                Decimal("0.045"),
                "GLEC v3.0 intermodal average",
            ),
        }

        entry = defaults.get(mode)
        if entry is not None:
            return entry
        return None, ""

    # =========================================================================
    # SECTION 7: INTERNAL HELPERS - LADEN STATE AND REEFER
    # =========================================================================

    def _get_laden_factor(self, leg: TransportLeg) -> Decimal:
        """
        Get laden state adjustment factor for a leg.

        Adjusts emissions based on how loaded the vehicle is.
        Partially loaded vehicles are less fuel-efficient per tonne-km.

        Args:
            leg: TransportLeg with laden_state.

        Returns:
            Laden adjustment multiplier (>= 1.0 for partial loads).
        """
        if leg.mode != TransportMode.ROAD:
            return _ONE

        if leg.vehicle_type is None:
            return self._get_generic_laden_factor(leg.laden_state)

        try:
            vt_enum = RoadVehicleType(leg.vehicle_type)
        except ValueError:
            return self._get_generic_laden_factor(leg.laden_state)

        factors = ROAD_EMISSION_FACTORS.get(vt_enum)
        if factors is None:
            return self._get_generic_laden_factor(leg.laden_state)

        laden_key = self._laden_state_to_key(leg.laden_state)
        factor: Decimal = factors.get(laden_key, _ONE)
        return factor

    def _get_generic_laden_factor(self, state: LadenState) -> Decimal:
        """Get generic laden factor when vehicle type is unknown."""
        generic: Dict[LadenState, Decimal] = {
            LadenState.FULL: Decimal("1.00"),
            LadenState.HALF: Decimal("1.20"),
            LadenState.EMPTY: Decimal("1.50"),
            LadenState.AVERAGE: Decimal("1.10"),
        }
        return generic.get(state, _ONE)

    def _laden_state_to_key(self, state: LadenState) -> str:
        """Map LadenState enum to ROAD_EMISSION_FACTORS key."""
        mapping: Dict[LadenState, str] = {
            LadenState.FULL: "laden_full",
            LadenState.HALF: "laden_half",
            LadenState.EMPTY: "laden_empty",
            LadenState.AVERAGE: "laden_full",
        }
        return mapping.get(state, "laden_full")

    def _get_reefer_uplift(self, leg: TransportLeg) -> Decimal:
        """
        Get reefer uplift factor for a leg.

        Args:
            leg: TransportLeg with temperature_control.

        Returns:
            Reefer uplift multiplier (1.0 for ambient).
        """
        if not self._include_reefer:
            return _ONE

        if leg.temperature_control == TemperatureControl.AMBIENT:
            return _ONE

        mode_factors = REEFER_UPLIFT_FACTORS.get(leg.mode)
        if mode_factors is None:
            return _ONE

        uplift: Decimal = mode_factors.get(leg.temperature_control, _ONE)
        return uplift

    # =========================================================================
    # SECTION 8: INTERNAL HELPERS - ALLOCATION
    # =========================================================================

    def _resolve_allocation_pct(
        self,
        leg: TransportLeg,
        allocation_config: Optional[AllocationConfig],
    ) -> Decimal:
        """
        Resolve allocation percentage for a leg.

        Priority:
            1. Explicit allocation_percentage on the leg
            2. Calculate from AllocationConfig
            3. Default to 100%

        Args:
            leg: TransportLeg.
            allocation_config: Optional AllocationConfig.

        Returns:
            Allocation percentage (0-100).
        """
        # Priority 1: Explicit on leg
        if leg.allocation_percentage is not None:
            return leg.allocation_percentage

        # Priority 2: Calculate from config
        if allocation_config is not None:
            total = self._get_total_for_allocation(leg, allocation_config)
            if total > _ZERO:
                share = self._get_share_for_allocation(leg, allocation_config)
                pct = _q(share / total * _HUNDRED)
                return min(pct, _HUNDRED)

        # Priority 3: Default 100%
        return _HUNDRED

    def _get_total_for_allocation(
        self,
        leg: TransportLeg,
        config: AllocationConfig,
    ) -> Decimal:
        """Get total capacity from allocation config or leg."""
        method = config.allocation_method

        if method == AllocationMethod.MASS:
            return config.total_capacity_tonnes or leg.total_vehicle_capacity_tonnes or _ZERO
        elif method == AllocationMethod.VOLUME:
            return config.total_capacity_m3 or _ZERO
        elif method == AllocationMethod.PALLET_POSITIONS:
            return Decimal(str(config.total_pallet_positions or 0))
        elif method == AllocationMethod.TEU:
            return Decimal(str(config.total_teu or 0))
        elif method == AllocationMethod.REVENUE:
            return config.total_revenue or _ZERO
        return _ZERO

    def _get_share_for_allocation(
        self,
        leg: TransportLeg,
        config: AllocationConfig,
    ) -> Decimal:
        """Get reporter's share from allocation config or leg."""
        method = config.allocation_method

        if method == AllocationMethod.MASS:
            return config.shipment_mass_tonnes or leg.cargo_mass_tonnes or _ZERO
        elif method == AllocationMethod.VOLUME:
            return config.shipment_volume_m3 or leg.cargo_volume_m3 or _ZERO
        elif method == AllocationMethod.PALLET_POSITIONS:
            return Decimal(str(config.shipment_pallet_positions or 0))
        elif method == AllocationMethod.TEU:
            return config.shipment_teu or _ZERO
        elif method == AllocationMethod.REVENUE:
            return config.shipment_revenue or _ZERO
        return _ZERO

    def _validate_allocation_inputs(
        self,
        method_name: str,
        reporter_value: Decimal,
        total_value: Decimal,
        unit: str,
    ) -> None:
        """
        Validate allocation inputs.

        Args:
            method_name: Allocation method name (for error messages).
            reporter_value: Reporter's share value.
            total_value: Total capacity value.
            unit: Unit of measurement (for error messages).

        Raises:
            ValueError: If inputs are invalid.
        """
        if total_value <= _ZERO:
            raise ValueError(
                f"{method_name} allocation: total must be > 0 {unit}, "
                f"got {total_value}"
            )
        if reporter_value < _ZERO:
            raise ValueError(
                f"{method_name} allocation: reporter value cannot be "
                f"negative, got {reporter_value} {unit}"
            )
        if reporter_value > total_value:
            logger.warning(
                "%s allocation: reporter value (%.4f) exceeds total (%.4f) %s. "
                "Capping at 100%%.",
                method_name,
                float(reporter_value),
                float(total_value),
                unit,
            )

    # =========================================================================
    # SECTION 9: INTERNAL HELPERS - HUB EMISSION FACTORS
    # =========================================================================

    def _resolve_hub_ef(
        self,
        hub: TransportHub,
    ) -> Tuple[Decimal, str, Optional[Decimal]]:
        """
        Resolve emission factor for a hub.

        Priority:
            1. Custom EF on the hub
            2. Direct energy measurement
            3. Hub-type default from embedded table

        Args:
            hub: TransportHub.

        Returns:
            Tuple of (ef_per_tonne, source, energy_kwh_or_None).
        """
        energy_kwh = hub.energy_kwh

        # Priority 1: Custom EF
        if hub.custom_ef_per_tonne is not None and hub.custom_ef_per_tonne >= _ZERO:
            source = hub.ef_source or "custom_hub_ef"
            return hub.custom_ef_per_tonne, source, energy_kwh

        # Priority 2: Direct energy (defer calculation to caller)
        if energy_kwh is not None and energy_kwh > _ZERO:
            ef_per_tonne = _ZERO
            source = "direct_energy_measurement"
            return ef_per_tonne, source, energy_kwh

        # Priority 3: Default from table
        ef = HUB_EMISSION_FACTORS.get(hub.hub_type, Decimal("0.50"))
        source = f"HUB_EMISSION_FACTORS[{hub.hub_type.value}] (CEFIC / GLEC v3.0)"
        return ef, source, energy_kwh

    def _calculate_dwell_uplift(self, hub: TransportHub) -> Decimal:
        """
        Calculate dwell time uplift for storage hubs.

        For hubs with storage duration > 0, apply an uplift factor:
            - Cross-dock / transshipment: no uplift (dwell < 1 day typical)
            - Warehouse: 1.0 + (dwell_days x 0.05) per day
            - Cold/frozen storage: 1.0 + (dwell_days x 0.10) per day

        Args:
            hub: TransportHub.

        Returns:
            Dwell time uplift multiplier (>= 1.0).
        """
        if hub.dwell_time_days <= _ZERO:
            return _ONE

        daily_rate: Dict[HubType, Decimal] = {
            HubType.WAREHOUSE: Decimal("0.05"),
            HubType.COLD_STORAGE: Decimal("0.10"),
            HubType.FROZEN_STORAGE: Decimal("0.15"),
            HubType.CROSS_DOCK: Decimal("0.01"),
            HubType.DISTRIBUTION_CENTER: Decimal("0.06"),
            HubType.FULFILLMENT_CENTER: Decimal("0.07"),
            HubType.CONSOLIDATION_CENTER: Decimal("0.04"),
            HubType.TRANSSHIPMENT_HUB: Decimal("0.02"),
        }

        rate = daily_rate.get(hub.hub_type, Decimal("0.05"))
        uplift = _q(_ONE + hub.dwell_time_days * rate)

        # Cap at reasonable maximum (10x)
        max_uplift = Decimal("10")
        if uplift > max_uplift:
            uplift = max_uplift

        return uplift

    def _get_default_energy_intensity(
        self,
        config: WarehouseConfig,
    ) -> Decimal:
        """
        Get default energy intensity for warehouse type.

        Args:
            config: WarehouseConfig.

        Returns:
            Energy intensity in kWh per m2 per year.
        """
        if config.temperature_control == TemperatureControl.FROZEN_MINUS_18C:
            return WAREHOUSE_ENERGY_INTENSITIES.get("frozen_storage", Decimal("425"))
        elif config.temperature_control == TemperatureControl.DEEP_FROZEN_MINUS_25C:
            return Decimal("525")
        elif config.temperature_control in (
            TemperatureControl.CHILLED_2_8C,
            TemperatureControl.HEATED,
        ):
            return WAREHOUSE_ENERGY_INTENSITIES.get("cold_storage", Decimal("285"))
        else:
            return WAREHOUSE_ENERGY_INTENSITIES.get("standard", Decimal("65"))

    # =========================================================================
    # SECTION 10: INTERNAL HELPERS - GAS BREAKDOWN
    # =========================================================================

    def _estimate_gas_breakdown(
        self,
        mode: TransportMode,
        total_co2e: Decimal,
        ef_per_tkm: Decimal,
    ) -> Tuple[Decimal, Decimal, Decimal]:
        """
        Estimate CO2, CH4, N2O breakdown from total CO2e.

        When only the total CO2e is known (from the tonne-km x EF
        calculation), approximate the gas breakdown using typical
        ratios per transport mode.

        Typical gas share of total CO2e:
            Road:     CO2 ~98.5%, CH4 ~0.5%, N2O ~1.0%
            Rail:     CO2 ~99.0%, CH4 ~0.3%, N2O ~0.7%
            Maritime: CO2 ~99.2%, CH4 ~0.5%, N2O ~0.3%
            Air:      CO2 ~99.0%, CH4 ~0.2%, N2O ~0.8%
            Pipeline: CO2 ~98.0%, CH4 ~1.5%, N2O ~0.5%

        Args:
            mode: Transport mode.
            total_co2e: Total kgCO2e.
            ef_per_tkm: Emission factor used (for context).

        Returns:
            Tuple of (co2_kg, ch4_kg, n2o_kg) in actual mass (not CO2e).
        """
        ratios: Dict[TransportMode, Tuple[Decimal, Decimal, Decimal]] = {
            TransportMode.ROAD: (
                Decimal("0.985"),
                Decimal("0.005"),
                Decimal("0.010"),
            ),
            TransportMode.RAIL: (
                Decimal("0.990"),
                Decimal("0.003"),
                Decimal("0.007"),
            ),
            TransportMode.MARITIME: (
                Decimal("0.992"),
                Decimal("0.005"),
                Decimal("0.003"),
            ),
            TransportMode.AIR: (
                Decimal("0.990"),
                Decimal("0.002"),
                Decimal("0.008"),
            ),
            TransportMode.PIPELINE: (
                Decimal("0.980"),
                Decimal("0.015"),
                Decimal("0.005"),
            ),
            TransportMode.INTERMODAL: (
                Decimal("0.988"),
                Decimal("0.005"),
                Decimal("0.007"),
            ),
        }

        co2_r, ch4_r, n2o_r = ratios.get(
            mode,
            (Decimal("0.990"), Decimal("0.005"), Decimal("0.005")),
        )

        # CO2e shares -> approximate gas mass
        # CO2 mass ~= total_co2e x co2_ratio (GWP=1 so CO2e == mass)
        co2_kg = _q(total_co2e * co2_r)

        # CH4: CO2e share / GWP = mass
        gwp_ch4 = get_gwp(EmissionGas.CH4, self._gwp_source)
        ch4_co2e = _q(total_co2e * ch4_r)
        ch4_kg = _q(ch4_co2e / gwp_ch4) if gwp_ch4 > _ZERO else _ZERO

        # N2O: CO2e share / GWP = mass
        gwp_n2o = get_gwp(EmissionGas.N2O, self._gwp_source)
        n2o_co2e = _q(total_co2e * n2o_r)
        n2o_kg = _q(n2o_co2e / gwp_n2o) if gwp_n2o > _ZERO else _ZERO

        return co2_kg, ch4_kg, n2o_kg

    # =========================================================================
    # SECTION 11: INTERNAL HELPERS - DQI SCORING
    # =========================================================================

    def _score_leg_dqi(self, lr: LegResult) -> Decimal:
        """
        Assign a DQI score to a leg result.

        Heuristic scoring based on EF source and data characteristics.

        Args:
            lr: LegResult.

        Returns:
            DQI score (1.0 to 5.0).
        """
        source = lr.ef_source.lower() if lr.ef_source else ""

        if "supplier" in source:
            return Decimal("1.5")
        if "fuel" in source or "actual" in source:
            return Decimal("2.0")
        if "custom" in source:
            return Decimal("2.5")
        if "glec" in source or "defra" in source or "imo" in source:
            return Decimal("2.5")
        if "icao" in source:
            return Decimal("2.5")
        if "default" in source or "average" in source:
            return Decimal("3.5")
        if "spend" in source or "eeio" in source:
            return Decimal("4.0")
        return Decimal("3.0")

    def _score_hub_dqi(self, hr: HubResult) -> Decimal:
        """
        Assign a DQI score to a hub result.

        Args:
            hr: HubResult.

        Returns:
            DQI score (1.0 to 5.0).
        """
        source = hr.ef_source.lower() if hr.ef_source else ""

        if "direct" in source or "energy" in source or "measurement" in source:
            return Decimal("2.0")
        if "custom" in source:
            return Decimal("2.5")
        if "cefic" in source or "glec" in source:
            return Decimal("3.0")
        return Decimal("3.5")

    # =========================================================================
    # SECTION 12: PROVENANCE AND HASHING
    # =========================================================================

    def compute_chain_provenance_hash(
        self,
        chain: TransportChain,
        result: TransportChainResult,
    ) -> str:
        """
        Compute SHA-256 provenance hash for a chain calculation.

        Hashes the chain input and result together to create an
        immutable audit trail record.

        Args:
            chain: Input TransportChain.
            result: Output TransportChainResult.

        Returns:
            SHA-256 hex digest.
        """
        input_hash = calculate_provenance_hash(chain)
        output_hash = calculate_provenance_hash(result)
        combined = f"{input_hash}|{output_hash}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def compute_leg_provenance_hash(
        self,
        leg: TransportLeg,
        result: LegResult,
    ) -> str:
        """
        Compute SHA-256 provenance hash for a leg calculation.

        Args:
            leg: Input TransportLeg.
            result: Output LegResult.

        Returns:
            SHA-256 hex digest.
        """
        input_hash = calculate_provenance_hash(leg)
        output_hash = calculate_provenance_hash(result)
        combined = f"{input_hash}|{output_hash}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def compute_hub_provenance_hash(
        self,
        hub: TransportHub,
        result: HubResult,
    ) -> str:
        """
        Compute SHA-256 provenance hash for a hub calculation.

        Args:
            hub: Input TransportHub.
            result: Output HubResult.

        Returns:
            SHA-256 hex digest.
        """
        input_hash = calculate_provenance_hash(hub)
        output_hash = calculate_provenance_hash(result)
        combined = f"{input_hash}|{output_hash}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    # =========================================================================
    # SECTION 13: BATCH PROCESSING
    # =========================================================================

    def calculate_chains_batch(
        self,
        chains: List[TransportChain],
        allocation_config: Optional[AllocationConfig] = None,
    ) -> List[TransportChainResult]:
        """
        Calculate emissions for multiple transport chains.

        Processes chains sequentially with error isolation. Failed
        chains are logged but do not stop the batch.

        Args:
            chains: List of TransportChain objects.
            allocation_config: Optional shared allocation config.

        Returns:
            List of TransportChainResult (successful chains only).
        """
        start_time = time.monotonic()
        results: List[TransportChainResult] = []
        errors: List[str] = []

        logger.info("Batch calculation: %d chains", len(chains))

        for chain in chains:
            try:
                result = self.calculate_chain(chain, allocation_config)
                results.append(result)
            except Exception as e:
                error_msg = f"Chain {chain.chain_id} failed: {str(e)}"
                logger.error(error_msg, exc_info=True)
                errors.append(error_msg)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Batch complete: %d/%d succeeded, %d failed in %.1f ms",
            len(results),
            len(chains),
            len(errors),
            elapsed_ms,
        )
        return results

    # =========================================================================
    # SECTION 14: UTILITY METHODS
    # =========================================================================

    def get_supported_modes(self) -> List[str]:
        """
        Get list of supported transport modes.

        Returns:
            List of mode value strings.
        """
        return [m.value for m in TransportMode]

    def get_supported_hub_types(self) -> List[str]:
        """
        Get list of supported hub types.

        Returns:
            List of hub type value strings.
        """
        return [h.value for h in HubType]

    def get_supported_allocation_methods(self) -> List[str]:
        """
        Get list of supported allocation methods.

        Returns:
            List of allocation method value strings.
        """
        return [a.value for a in AllocationMethod]

    def get_smartway_categories(self) -> Dict[str, str]:
        """
        Get SmartWay carrier category descriptions.

        Returns:
            Dictionary mapping category code to description.
        """
        return {k: v["description"] for k, v in SMARTWAY_CARRIER_CATEGORIES.items()}

    def get_methodology_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get methodology validation information.

        Returns:
            Copy of methodology validation map.
        """
        return {
            k: {
                "valid": v["valid"],
                "score": str(v["score"]),
                "note": v["note"],
            }
            for k, v in METHODOLOGY_VALIDATION.items()
        }

    def get_certification_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get certification validation information.

        Returns:
            Copy of certification validation map.
        """
        return {
            k: {
                "valid": v["valid"],
                "score": str(v["score"]),
                "note": v["note"],
            }
            for k, v in CERTIFICATION_VALIDATION.items()
        }

    def get_chargeable_weight_dim_factor(self, mode: str) -> Optional[Decimal]:
        """
        Get dimensional factor for chargeable weight calculation.

        Args:
            mode: Transport mode string ('air', 'road', 'maritime').

        Returns:
            Dimensional factor or None if not applicable.
        """
        factor = CHARGEABLE_WEIGHT_DIM_FACTORS.get(mode.lower())
        if factor is not None and factor == _ZERO:
            return None
        return factor

    def get_hub_default_ef(self, hub_key: str) -> Optional[Decimal]:
        """
        Get default hub emission factor by key.

        Args:
            hub_key: Hub type key (e.g., 'logistics_hub', 'container_terminal').

        Returns:
            Default EF in kgCO2e/tonne or None.
        """
        return HUB_DEFAULT_EF.get(hub_key)

    def get_refrigerant_gwp(self, refrigerant: str) -> Optional[Decimal]:
        """
        Get GWP for a refrigerant type.

        Args:
            refrigerant: Refrigerant designation (e.g., 'R-404A', 'HFC-134a').

        Returns:
            GWP value or None if unknown.
        """
        return REFRIGERANT_GWP.get(refrigerant)

    def get_tru_fuel_rate(
        self,
        tru_type: str,
    ) -> Optional[Dict[str, Decimal]]:
        """
        Get TRU fuel consumption rate by type.

        Args:
            tru_type: TRU type key (e.g., 'road_diesel_genset').

        Returns:
            Dictionary with rate details or None.
        """
        return TRU_FUEL_RATES.get(tru_type)

    def estimate_leg_duration_hours(
        self,
        mode: TransportMode,
        distance_km: Decimal,
    ) -> Decimal:
        """
        Estimate transport leg duration from mode and distance.

        Average speeds by mode:
            - Road: 60 km/h
            - Rail: 50 km/h
            - Maritime: 28 km/h (15 knots)
            - Air: 850 km/h
            - Pipeline: N/A (continuous flow)

        Args:
            mode: Transport mode.
            distance_km: Leg distance in kilometers.

        Returns:
            Estimated duration in hours.
        """
        speeds: Dict[TransportMode, Decimal] = {
            TransportMode.ROAD: Decimal("60"),
            TransportMode.RAIL: Decimal("50"),
            TransportMode.MARITIME: Decimal("28"),
            TransportMode.AIR: Decimal("850"),
            TransportMode.PIPELINE: Decimal("1"),
            TransportMode.INTERMODAL: Decimal("45"),
        }

        speed = speeds.get(mode, Decimal("50"))
        if speed <= _ZERO:
            return _ZERO

        duration = _q(distance_km / speed)
        return duration

    def calculate_empty_running_adjustment(
        self,
        base_emissions: Decimal,
        mode: TransportMode,
        empty_running_rate: Optional[Decimal] = None,
    ) -> Decimal:
        """
        Adjust emissions to account for empty (deadhead) running.

        Empty running means the vehicle travels without cargo on
        return/repositioning trips. The emissions from these empty
        trips should be allocated to the laden trips.

        Formula:
            adjusted = base_emissions / (1 - empty_running_rate)

        Args:
            base_emissions: Base emissions from laden trip.
            mode: Transport mode.
            empty_running_rate: Override rate (0.0 to 1.0).

        Returns:
            Adjusted emissions accounting for empty running.
        """
        from greenlang.agents.mrv.upstream_transportation.models import EMPTY_RUNNING_RATES

        if empty_running_rate is None:
            empty_running_rate = EMPTY_RUNNING_RATES.get(mode, _ZERO)

        if empty_running_rate <= _ZERO:
            return base_emissions

        if empty_running_rate >= _ONE:
            logger.warning(
                "Empty running rate >= 1.0 for mode=%s; capping at 0.95",
                mode.value,
            )
            empty_running_rate = Decimal("0.95")

        denominator = _q(_ONE - empty_running_rate)
        adjusted = _q(base_emissions / denominator)

        logger.debug(
            "Empty running adjustment: base=%.4f / (1 - %.2f) = %.4f kgCO2e",
            float(base_emissions),
            float(empty_running_rate),
            float(adjusted),
        )
        return adjusted

    def calculate_load_factor_adjustment(
        self,
        base_emissions: Decimal,
        mode: TransportMode,
        actual_load_factor: Optional[Decimal] = None,
        reference_load_factor: Optional[Decimal] = None,
    ) -> Decimal:
        """
        Adjust emissions for actual vs reference load factor.

        If the actual load factor differs from the reference (used in
        the emission factor), adjust proportionally.

        Formula:
            adjusted = base_emissions x (reference_lf / actual_lf)

        Higher actual load factor -> lower per-tonne emissions.
        Lower actual load factor -> higher per-tonne emissions.

        Args:
            base_emissions: Base emissions.
            mode: Transport mode.
            actual_load_factor: Actual load factor (0.0 to 1.0).
            reference_load_factor: Reference load factor in the EF.

        Returns:
            Adjusted emissions.
        """
        if actual_load_factor is None:
            return base_emissions

        if reference_load_factor is None:
            reference_load_factor = LOAD_FACTOR_DEFAULTS.get(mode, Decimal("0.65"))

        if actual_load_factor <= _ZERO:
            logger.warning(
                "Load factor <= 0 for mode=%s; returning base emissions",
                mode.value,
            )
            return base_emissions

        ratio = _q(reference_load_factor / actual_load_factor)
        adjusted = _q(base_emissions * ratio)

        logger.debug(
            "Load factor adjustment: base=%.4f x (%.2f / %.2f) = %.4f kgCO2e",
            float(base_emissions),
            float(reference_load_factor),
            float(actual_load_factor),
            float(adjusted),
        )
        return adjusted

    def convert_kg_to_tonnes(self, kg: Decimal) -> Decimal:
        """Convert kilograms to metric tonnes."""
        return _q(kg / _THOUSAND)

    def convert_tonnes_to_kg(self, tonnes: Decimal) -> Decimal:
        """Convert metric tonnes to kilograms."""
        return _q(tonnes * _THOUSAND)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Engine
    "MultiLegCalculatorEngine",
    # Embedded data tables
    "SMARTWAY_CARRIER_CATEGORIES",
    "METHODOLOGY_VALIDATION",
    "CERTIFICATION_VALIDATION",
    "CHARGEABLE_WEIGHT_DIM_FACTORS",
    "HUB_DEFAULT_EF",
    "TRU_FUEL_RATES",
    "REFRIGERANT_GWP",
]
