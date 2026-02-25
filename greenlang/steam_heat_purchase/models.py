# -*- coding: utf-8 -*-
"""
Scope 2 Steam/Heat Purchase Agent Data Models - AGENT-MRV-011

Pydantic v2 data models for the Scope 2 Steam/Heat Purchase Agent SDK
covering GHG Protocol Scope 2 purchased steam, district heating, and
district cooling emission calculations including:
- 14 fuel types with combustion emission factors (CO2, CH4, N2O per GJ)
- 13 district heating regional factors with distribution loss percentages
- 9 cooling system technologies with COP ranges and energy sources
- 5 CHP fuel types with electrical, thermal, and overall efficiencies
- 3 CHP allocation methods (efficiency, energy, exergy)
- 4 calculation methods (direct EF, fuel-based, COP-based, CHP-allocated)
- 5 emission gas species (CO2, CH4, N2O, CO2e, biogenic CO2)
- GWP values from IPCC AR4, AR5, AR6, and AR6 20-year horizon
- Unit conversions between GJ, MWh, kWh, MMBtu, therm, and MJ
- Steam pressure classification (low, medium, high, very high)
- Steam quality classification (saturated, superheated, wet)
- District network types (municipal, industrial, campus, mixed)
- Facility metadata with supplier and network mapping
- Steam supplier profiles with fuel mix and boiler efficiency
- Condensate return percentage tracking for steam systems
- Batch calculation requests across multiple facilities
- Monte Carlo and analytical uncertainty quantification
- Multi-framework regulatory compliance checking
- Aggregation by facility, fuel, energy type, supplier, or period
- SHA-256 provenance chain for complete audit trails

Enumerations (18):
    - EnergyType, FuelType, CoolingTechnology, CHPAllocMethod,
      CalculationMethod, EmissionGas, GWPSource, ComplianceStatus,
      DataQualityTier, EnergyUnit, TemperatureUnit, SteamPressure,
      SteamQuality, NetworkType, FacilityType, ReportingPeriod,
      AggregationType, BatchStatus

Constants (all Decimal for zero-hallucination deterministic arithmetic):
    - GWP_VALUES: IPCC AR4/AR5/AR6/AR6_20YR GWP (CO2, CH4, N2O)
    - FUEL_EMISSION_FACTORS: 14 fuel types (CO2, CH4, N2O per GJ)
    - DISTRICT_HEATING_FACTORS: 13 regional heating factors (kgCO2e/GJ)
    - COOLING_SYSTEM_FACTORS: 9 cooling technologies (COP ranges)
    - CHP_DEFAULT_EFFICIENCIES: 5 CHP fuel types (elec/thermal/overall)
    - UNIT_CONVERSIONS: Energy unit conversion factors

Data Models (20):
    - FuelEmissionFactor, DistrictHeatingFactor, CoolingSystemFactor,
      CHPParameters, FacilityInfo, SteamSupplier,
      SteamCalculationRequest, HeatingCalculationRequest,
      CoolingCalculationRequest, CHPAllocationRequest,
      GasEmissionDetail, CalculationResult, CHPAllocationResult,
      BatchCalculationRequest, BatchCalculationResult,
      UncertaintyRequest, UncertaintyResult, ComplianceCheckResult,
      AggregationRequest, AggregationResult

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-011 Steam/Heat Purchase Agent (GL-MRV-X-022)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Service version string.
VERSION: str = "1.0.0"

#: Maximum number of calculations in a single batch request.
MAX_CALCULATIONS_PER_BATCH: int = 10_000

#: Maximum number of gas emission entries per calculation result.
MAX_GASES_PER_RESULT: int = 10

#: Maximum number of trace steps in a single calculation.
MAX_TRACE_STEPS: int = 200

#: Maximum number of facility records per tenant.
MAX_FACILITIES_PER_TENANT: int = 50_000

#: Default Monte Carlo simulation iterations for uncertainty analysis.
DEFAULT_MONTE_CARLO_ITERATIONS: int = 10_000

#: Default confidence level for uncertainty intervals.
DEFAULT_CONFIDENCE_LEVEL: Decimal = Decimal("0.95")

#: Prefix for all database table names in this module.
TABLE_PREFIX: str = "gl_shp_"


# =============================================================================
# Enumerations (18)
# =============================================================================


class EnergyType(str, Enum):
    """Classification of purchased thermal energy types for Scope 2.

    GHG Protocol Scope 2 Guidance requires organisations to report
    indirect emissions from purchased or acquired thermal energy.
    This enumeration covers the five categories of purchased thermal
    energy relevant to the Steam/Heat Purchase Agent.

    STEAM: Purchased steam from an external boiler plant, industrial
        supplier, or district steam network. Emission factors depend
        on the fuel mix of the steam generator and boiler efficiency.
        Common in industrial and institutional facilities. Steam
        consumption is typically metered in GJ, MMBtu, or klb.
    DISTRICT_HEATING: Purchased heat from a district heating network
        that distributes hot water or steam from a centralised plant
        to multiple buildings. Emission factors are region-specific
        and account for fuel mix, network losses, and seasonal
        variation. Common in Northern European and East Asian cities.
    DISTRICT_COOLING: Purchased chilled water or cooling capacity
        from a district cooling network. Emissions depend on the
        cooling technology (electric chiller, absorption chiller,
        free cooling) and the energy source (grid electricity or
        waste heat). COP-based calculation converts cooling output
        to energy input for emission factor application.
    CHP_STEAM: Steam produced by a combined heat and power (CHP)
        or cogeneration plant. Emissions must be allocated between
        the electrical and thermal outputs using one of three
        methods: efficiency-based, energy-based, or exergy-based
        allocation per GHG Protocol guidance.
    CHP_HEATING: Hot water heating produced by a CHP plant.
        Allocation of fuel emissions follows the same methodology
        as CHP_STEAM but may use different temperature parameters
        for exergy-based allocation.
    """

    STEAM = "steam"
    DISTRICT_HEATING = "district_heating"
    DISTRICT_COOLING = "district_cooling"
    CHP_STEAM = "chp_steam"
    CHP_HEATING = "chp_heating"


class FuelType(str, Enum):
    """Classification of fuel types used in steam and heat generation.

    Identifies the primary fuel burned in the steam boiler, district
    heating plant, or CHP unit. Determines the emission factors
    (CO2, CH4, N2O per GJ) applied to the fuel consumption for
    emission calculations. Biogenic fuels (biomass) have their CO2
    reported separately per GHG Protocol guidance.

    NATURAL_GAS: Pipeline-quality natural gas. Lowest CO2 intensity
        among fossil fuels. Most common fuel for commercial and
        industrial steam boilers. EF = 56.100 kgCO2/GJ.
    FUEL_OIL_2: Light distillate fuel oil (No. 2 / diesel oil).
        Used in smaller boilers and as backup fuel. EF = 74.100
        kgCO2/GJ.
    FUEL_OIL_6: Heavy residual fuel oil (No. 6 / bunker C). Used
        in large industrial boilers and older district heating
        plants. EF = 77.400 kgCO2/GJ.
    COAL_BITUMINOUS: Bituminous coal (hard coal). High energy
        density. Common in industrial and utility boilers. EF =
        94.600 kgCO2/GJ.
    COAL_SUBBITUMINOUS: Subbituminous coal. Lower energy density
        and higher moisture content than bituminous. EF = 96.100
        kgCO2/GJ.
    COAL_LIGNITE: Lignite (brown coal). Lowest energy density and
        highest CO2 intensity among coal types. Common in Eastern
        European district heating. EF = 101.000 kgCO2/GJ.
    LPG: Liquefied petroleum gas (propane/butane mix). Used in
        smaller boilers where natural gas is unavailable. EF =
        63.100 kgCO2/GJ.
    BIOMASS_WOOD: Wood chips, wood pellets, or other woody biomass.
        Biogenic CO2 reported separately; fossil CO2e = 0 for
        sustainably sourced biomass. EF = 112.000 kgCO2/GJ
        (biogenic). Default efficiency 70%.
    BIOMASS_BIOGAS: Biogas from anaerobic digestion (landfill gas,
        sewage gas, agricultural biogas). Biogenic CO2 reported
        separately. EF = 54.600 kgCO2/GJ (biogenic).
    MUNICIPAL_WASTE: Municipal solid waste incineration with heat
        recovery. Mixed biogenic and fossil carbon content. EF =
        91.700 kgCO2/GJ (fossil fraction).
    WASTE_HEAT: Recovered waste heat from industrial processes with
        no additional fuel combustion. Zero direct emissions.
        Distribution losses still apply.
    GEOTHERMAL: Geothermal heat extracted from underground
        reservoirs. Zero direct combustion emissions. May have
        minor fugitive emissions from dissolved gases.
    SOLAR_THERMAL: Solar thermal collectors (flat plate,
        evacuated tube, or concentrated). Zero direct emissions.
        Limited to supplementary heating in most climates.
    ELECTRIC: Electrically generated steam or heat (electric
        boiler, heat pump). Zero direct combustion emissions at
        the point of use; grid-dependent upstream emissions are
        calculated separately under Scope 2 electricity.
    """

    NATURAL_GAS = "natural_gas"
    FUEL_OIL_2 = "fuel_oil_2"
    FUEL_OIL_6 = "fuel_oil_6"
    COAL_BITUMINOUS = "coal_bituminous"
    COAL_SUBBITUMINOUS = "coal_subbituminous"
    COAL_LIGNITE = "coal_lignite"
    LPG = "lpg"
    BIOMASS_WOOD = "biomass_wood"
    BIOMASS_BIOGAS = "biomass_biogas"
    MUNICIPAL_WASTE = "municipal_waste"
    WASTE_HEAT = "waste_heat"
    GEOTHERMAL = "geothermal"
    SOLAR_THERMAL = "solar_thermal"
    ELECTRIC = "electric"


class CoolingTechnology(str, Enum):
    """Classification of cooling technologies for district cooling systems.

    Identifies the type of chiller or cooling equipment used in a
    district cooling network. The technology determines the
    coefficient of performance (COP) range, which is used to
    convert cooling output to energy input for emission calculations.

    CENTRIFUGAL_CHILLER: Large centrifugal compressor chiller.
        Highest COP among electric chillers (5.0-7.0). Standard
        choice for large district cooling plants. Energy source:
        electricity.
    SCREW_CHILLER: Rotary screw compressor chiller. Moderate COP
        (4.0-5.5). Common in medium-sized district cooling and
        industrial applications. Energy source: electricity.
    RECIPROCATING_CHILLER: Reciprocating compressor chiller. Lower
        COP (3.5-5.0) but suitable for variable loads. Common in
        smaller installations. Energy source: electricity.
    ABSORPTION_SINGLE: Single-effect absorption chiller driven by
        low-grade heat (hot water or low-pressure steam). Low COP
        (0.6-0.8) but uses waste heat or solar thermal as input.
        Energy source: heat.
    ABSORPTION_DOUBLE: Double-effect absorption chiller driven by
        higher-grade heat (high-pressure steam or direct-fired).
        Moderate COP (1.0-1.4). More efficient use of heat input.
        Energy source: heat.
    ABSORPTION_TRIPLE: Triple-effect absorption chiller. Highest
        COP among absorption types (1.5-1.8). Requires high-
        temperature heat input. Energy source: heat.
    FREE_COOLING: Free cooling using ambient air, seawater, lake
        water, or deep aquifer water when ambient temperature is
        below the supply temperature setpoint. Very high effective
        COP (15.0-30.0) but limited by climate and season. Energy
        source: electricity (pumps and fans only).
    ICE_STORAGE: Ice thermal energy storage with off-peak charging
        and peak-hour discharge. Lower effective COP (3.0-4.5)
        due to the lower evaporating temperature required for ice
        production. Energy source: electricity.
    THERMAL_STORAGE: Chilled water thermal energy storage (stratified
        tank or underground). Moderate COP (4.0-6.0). Enables load
        shifting without the COP penalty of ice storage. Energy
        source: electricity.
    """

    CENTRIFUGAL_CHILLER = "centrifugal_chiller"
    SCREW_CHILLER = "screw_chiller"
    RECIPROCATING_CHILLER = "reciprocating_chiller"
    ABSORPTION_SINGLE = "absorption_single"
    ABSORPTION_DOUBLE = "absorption_double"
    ABSORPTION_TRIPLE = "absorption_triple"
    FREE_COOLING = "free_cooling"
    ICE_STORAGE = "ice_storage"
    THERMAL_STORAGE = "thermal_storage"


class CHPAllocMethod(str, Enum):
    """Allocation method for CHP/cogeneration emission apportionment.

    Combined heat and power (CHP) plants produce both electricity
    and useful heat from a single fuel input. The total fuel
    emissions must be allocated between the electrical and thermal
    outputs for Scope 2 reporting. GHG Protocol provides three
    acceptable allocation methods.

    EFFICIENCY: Efficiency-based allocation. Allocates emissions
        proportionally to the energy content of each output divided
        by its respective conversion efficiency. This method
        reflects the thermodynamic quality difference between
        electricity and heat. Most commonly used method.
        Formula: heat_share = (Q_heat / eta_thermal) /
        ((Q_heat / eta_thermal) + (Q_elec / eta_elec))
    ENERGY: Energy-based allocation. Allocates emissions
        proportionally to the energy content of each output
        (MWh or GJ) without adjusting for conversion efficiency.
        Simpler but does not reflect the higher thermodynamic
        value of electricity versus heat. Sometimes called the
        "energy content method".
        Formula: heat_share = Q_heat / (Q_heat + Q_elec)
    EXERGY: Exergy-based allocation. Allocates emissions
        proportionally to the exergy (available work) content of
        each output. Accounts for the thermodynamic quality by
        applying a Carnot factor to the heat output based on
        its temperature. Most thermodynamically rigorous method.
        Formula: heat_share = (Q_heat * carnot) /
        ((Q_heat * carnot) + Q_elec) where
        carnot = 1 - (T_ambient / T_steam) in Kelvin.
    """

    EFFICIENCY = "efficiency"
    ENERGY = "energy"
    EXERGY = "exergy"


class CalculationMethod(str, Enum):
    """Methodology for calculating Scope 2 thermal energy emissions.

    Determines how emissions are computed from purchased steam,
    district heating, or district cooling consumption data.

    DIRECT_EF: Direct emission factor method. Applies a single
        composite emission factor (kgCO2e per GJ) directly to the
        consumed thermal energy quantity. Simplest method. Used
        when the supplier provides a verified composite EF or when
        only a regional default factor is available.
    FUEL_BASED: Fuel-based calculation method. Calculates emissions
        from the fuel type, fuel quantity, and boiler efficiency
        used to generate the steam or heat. Provides per-gas
        breakdown (CO2, CH4, N2O). Requires knowledge of the
        supplier's fuel mix and equipment efficiency.
        Formula: emissions = consumption_gj / boiler_efficiency *
        fuel_ef_per_gj
    COP_BASED: Coefficient of performance method for district
        cooling. Converts cooling output to energy input using the
        COP of the cooling technology, then applies the appropriate
        emission factor to the energy input (grid electricity EF
        for electric chillers, heat source EF for absorption
        chillers).
        Formula: energy_input = cooling_output / COP
        emissions = energy_input * energy_source_ef
    CHP_ALLOCATED: CHP allocation method. Calculates total fuel
        emissions from the CHP plant and allocates a share to the
        thermal output using the selected allocation method
        (efficiency, energy, or exergy). Used when the thermal
        energy is sourced from a CHP/cogeneration plant.
    """

    DIRECT_EF = "direct_ef"
    FUEL_BASED = "fuel_based"
    COP_BASED = "cop_based"
    CHP_ALLOCATED = "chp_allocated"


class EmissionGas(str, Enum):
    """Greenhouse gases tracked in Scope 2 steam/heat calculations.

    CO2: Carbon dioxide. Primary emission from fossil fuel combustion
        in boilers, district heating plants, and CHP units. Largest
        contributor to total CO2e for thermal energy.
    CH4: Methane. Emitted from incomplete combustion in boilers and
        furnaces. Small fraction of total emissions but included
        for complete GHG Protocol compliance. Higher for biomass
        and coal combustion.
    N2O: Nitrous oxide. Emitted from combustion processes,
        particularly at high temperatures and with coal or biomass
        fuels. Included for complete GHG Protocol compliance.
    CO2E: Carbon dioxide equivalent. Aggregated total of all
        greenhouse gases weighted by their Global Warming Potential
        (GWP) values. Used for reporting totals.
    BIOGENIC_CO2: Biogenic carbon dioxide from biomass combustion.
        Reported separately from fossil CO2 per GHG Protocol
        guidance. Not included in Scope 2 totals for sustainably
        sourced biomass.
    """

    CO2 = "CO2"
    CH4 = "CH4"
    N2O = "N2O"
    CO2E = "CO2e"
    BIOGENIC_CO2 = "biogenic_CO2"


class GWPSource(str, Enum):
    """IPCC Assessment Report edition used for Global Warming Potential values.

    Determines which set of 100-year (or 20-year) GWP multipliers to
    apply when converting individual gas emissions (CH4, N2O) to CO2
    equivalent totals.

    AR4: Fourth Assessment Report (2007). 100-year GWP.
        CO2=1, CH4=25, N2O=298.
    AR5: Fifth Assessment Report (2014). 100-year GWP.
        CO2=1, CH4=28, N2O=265.
    AR6: Sixth Assessment Report (2021). 100-year GWP.
        CO2=1, CH4=27.9, N2O=273.
    AR6_20YR: Sixth Assessment Report (2021). 20-year GWP.
        CO2=1, CH4=81.2, N2O=273. Highlights near-term climate
        impact of short-lived pollutants like methane.
    """

    AR4 = "AR4"
    AR5 = "AR5"
    AR6 = "AR6"
    AR6_20YR = "AR6_20YR"


class ComplianceStatus(str, Enum):
    """Result of a regulatory compliance check for a calculation.

    COMPLIANT: All requirements of the regulatory framework are
        fully satisfied for the given calculation.
    NON_COMPLIANT: One or more mandatory requirements are not met.
    PARTIAL: Some requirements are met but others are missing,
        incomplete, or require additional evidence.
    NOT_APPLICABLE: The regulatory framework does not apply to this
        type of thermal energy calculation or facility type.
    """

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL = "partial"
    NOT_APPLICABLE = "not_applicable"


class DataQualityTier(str, Enum):
    """Data quality classification for emission factor inputs.

    Tier classification follows the IPCC approach to data quality
    and determines the uncertainty range applied to calculation
    results.

    TIER_1: Default values from international databases (IPCC, IEA).
        Highest uncertainty range (+/- 25-50%). Used when no
        supplier-specific data is available. Applies regional or
        national default emission factors and boiler efficiencies.
    TIER_2: Supplier-specific data such as disclosed fuel mix,
        measured boiler efficiency, or supplier-provided emission
        factors. Moderate uncertainty (+/- 10-25%). Requires
        documentation from the thermal energy supplier.
    TIER_3: Facility-specific measured data including continuous
        emissions monitoring (CEMS), fuel analysis certificates,
        and independently verified boiler performance tests.
        Lowest uncertainty (+/- 5-10%). Highest data quality.
    """

    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"


class EnergyUnit(str, Enum):
    """Units of measurement for thermal energy consumption quantities.

    All thermal energy consumption data must be expressed in a known
    unit for conversion to gigajoules (GJ) before applying emission
    factors. GJ is the standard basis for steam, heat, and cooling
    emission factors in this module.

    GJ: Gigajoules. SI unit. Standard basis for steam and heat
        emission factors. 1 GJ = 277.778 kWh = 0.277778 MWh.
    MWH: Megawatt-hours. 1 MWh = 3.6 GJ. Common for large
        district heating and cooling systems.
    KWH: Kilowatt-hours. 1 kWh = 0.0036 GJ. Common for smaller
        commercial heating and cooling metering.
    MMBTU: Million British Thermal Units. 1 MMBtu = 1.055056 GJ.
        Common in US steam and natural gas metering.
    THERM: Therms. 1 therm = 100,000 BTU = 0.105506 GJ. Common
        in US residential and commercial gas billing.
    MJ: Megajoules. 1 MJ = 0.001 GJ. Used in some European and
        Asian metering systems for smaller quantities.
    """

    GJ = "gj"
    MWH = "mwh"
    KWH = "kwh"
    MMBTU = "mmbtu"
    THERM = "therm"
    MJ = "mj"


class TemperatureUnit(str, Enum):
    """Units of temperature measurement for steam and heating systems.

    Used in exergy-based CHP allocation calculations and steam
    enthalpy lookups.

    CELSIUS: Degrees Celsius. Standard metric temperature unit.
        T(K) = T(C) + 273.15.
    FAHRENHEIT: Degrees Fahrenheit. Common in US engineering.
        T(C) = (T(F) - 32) * 5/9.
    KELVIN: Kelvin. Absolute temperature scale. Required for
        Carnot factor calculation in exergy-based CHP allocation.
    """

    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"
    KELVIN = "kelvin"


class SteamPressure(str, Enum):
    """Classification of steam pressure levels in steam systems.

    Steam pressure affects the enthalpy content and therefore the
    emission intensity per unit of useful energy delivered. Higher
    pressure steam carries more energy per kilogram but may require
    more fuel input to generate.

    LOW: Low-pressure steam, typically below 1.0 MPa (150 psig).
        Common for space heating, domestic hot water, and low-
        temperature industrial processes. Lower boiler losses.
    MEDIUM: Medium-pressure steam, typically 1.0-2.5 MPa (150-365
        psig). Common for process heating, sterilisation, and
        medium-temperature industrial applications.
    HIGH: High-pressure steam, typically 2.5-6.0 MPa (365-870
        psig). Common for power generation turbines and high-
        temperature industrial processes.
    VERY_HIGH: Very-high-pressure steam, typically above 6.0 MPa
        (870 psig). Used in utility-scale power plants and
        specialised industrial processes. Requires specialised
        boiler equipment.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class SteamQuality(str, Enum):
    """Classification of steam thermodynamic quality.

    Steam quality (also called dryness fraction for wet steam)
    affects the usable energy content and therefore the emission
    allocation per unit of delivered energy.

    SATURATED: Saturated steam at the boiling point for its
        pressure. Contains no liquid water droplets. Standard
        reference condition for emission factor tables. Dryness
        fraction = 1.0.
    SUPERHEATED: Superheated steam heated beyond the saturation
        temperature for its pressure. Higher enthalpy content
        than saturated steam. Common in power generation and
        high-temperature processes.
    WET: Wet steam containing a mixture of steam and liquid water
        droplets. Dryness fraction < 1.0. Lower usable energy
        content per kilogram than saturated steam. Common in
        distribution systems with inadequate insulation.
    """

    SATURATED = "saturated"
    SUPERHEATED = "superheated"
    WET = "wet"


class NetworkType(str, Enum):
    """Classification of district energy network types.

    Determines default distribution loss factors and applicable
    regional emission factors for district heating and cooling
    calculations.

    MUNICIPAL: Municipality-operated district energy network serving
        a mix of residential, commercial, and institutional
        buildings across a city or district. Typically the largest
        networks with the most diverse customer base.
    INDUSTRIAL: Industrial district energy network serving a cluster
        of industrial facilities, often co-located in an industrial
        park or chemical complex. May use waste heat from industrial
        processes as a primary heat source.
    CAMPUS: Campus district energy network serving a university,
        hospital complex, military base, or corporate campus.
        Typically a single-owner network with centralised plant
        operations.
    MIXED: Mixed-use district energy network combining elements of
        municipal, industrial, and campus networks. Serves a
        diverse set of customers across multiple sectors.
    """

    MUNICIPAL = "municipal"
    INDUSTRIAL = "industrial"
    CAMPUS = "campus"
    MIXED = "mixed"


class FacilityType(str, Enum):
    """Classification of reporting facilities by primary function.

    Determines default energy intensity benchmarks and applicable
    reporting templates. Used for aggregation and comparative
    analysis across facility portfolios for thermal energy
    consumption.

    INDUSTRIAL: Industrial manufacturing or production facility.
        Typically the highest steam consumption intensity.
    COMMERCIAL: Commercial office building, retail space, or
        mixed-use commercial property.
    INSTITUTIONAL: Institutional facility such as a government
        building, museum, or public service building.
    RESIDENTIAL: Residential building or housing complex connected
        to a district heating or cooling network.
    DATA_CENTER: IT data centre or colocation facility. Significant
        cooling demand, often served by district cooling.
    CAMPUS: University, hospital, or corporate campus with
        centralised thermal energy distribution.
    """

    INDUSTRIAL = "industrial"
    COMMERCIAL = "commercial"
    INSTITUTIONAL = "institutional"
    RESIDENTIAL = "residential"
    DATA_CENTER = "data_center"
    CAMPUS = "campus"


class ReportingPeriod(str, Enum):
    """Time period for emission aggregation and reporting outputs.

    MONTHLY: Calendar month. Used for operational dashboards
        and trend analysis of thermal energy consumption.
    QUARTERLY: Calendar quarter (Q1-Q4). Used for interim
        management reporting and progress tracking.
    ANNUAL: Full calendar or fiscal year. Standard for CDP, CSRD,
        and GHG Protocol corporate inventories.
    """

    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


class AggregationType(str, Enum):
    """Dimension for aggregating calculation results.

    Determines how individual calculation results are grouped
    when producing summary reports and portfolio-level totals.

    BY_FACILITY: Aggregate by facility. Produces per-facility
        emission totals across all thermal energy types.
    BY_FUEL: Aggregate by fuel type. Produces per-fuel emission
        totals across all facilities. Useful for fuel-switching
        analysis.
    BY_ENERGY_TYPE: Aggregate by energy type (steam, heating,
        cooling). Produces per-type emission totals.
    BY_SUPPLIER: Aggregate by thermal energy supplier. Produces
        per-supplier emission totals for procurement analysis.
    BY_PERIOD: Aggregate by time period (month, quarter, year).
        Produces temporal trends for performance tracking.
    """

    BY_FACILITY = "by_facility"
    BY_FUEL = "by_fuel"
    BY_ENERGY_TYPE = "by_energy_type"
    BY_SUPPLIER = "by_supplier"
    BY_PERIOD = "by_period"


class BatchStatus(str, Enum):
    """Processing status of a batch calculation request.

    Tracks the lifecycle of a batch processing job from submission
    to completion.

    PENDING: Batch has been submitted but processing has not yet
        started. Waiting in the job queue.
    RUNNING: Batch processing is in progress. Individual
        calculations are being executed.
    COMPLETED: All calculations in the batch completed successfully.
    FAILED: The batch processing failed due to a system error.
        No results are available.
    PARTIAL: Some calculations completed successfully but others
        failed. Partial results are available with error details
        for the failed calculations.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


# =============================================================================
# Constant Tables (all Decimal for deterministic arithmetic)
# =============================================================================


# ---------------------------------------------------------------------------
# GWP values by IPCC Assessment Report
# ---------------------------------------------------------------------------

#: Global Warming Potential values for greenhouse gases by IPCC AR.
#: Units: dimensionless multiplier (kg CO2e per kg gas).
#: Decimal precision for zero-hallucination regulatory calculations.
#: Sources:
#:   AR4: IPCC Fourth Assessment Report (2007), Table 2.14.
#:   AR5: IPCC Fifth Assessment Report (2014), Table 8.A.1.
#:   AR6: IPCC Sixth Assessment Report (2021), Table 7.15.
#:   AR6_20YR: AR6 GWP-20yr timeframe, Table 7.15.
GWP_VALUES: Dict[str, Dict[str, Decimal]] = {
    "AR4": {
        "CO2": Decimal("1"),
        "CH4": Decimal("25"),
        "N2O": Decimal("298"),
    },
    "AR5": {
        "CO2": Decimal("1"),
        "CH4": Decimal("28"),
        "N2O": Decimal("265"),
    },
    "AR6": {
        "CO2": Decimal("1"),
        "CH4": Decimal("27.9"),
        "N2O": Decimal("273"),
    },
    "AR6_20YR": {
        "CO2": Decimal("1"),
        "CH4": Decimal("81.2"),
        "N2O": Decimal("273"),
    },
}


# ---------------------------------------------------------------------------
# Fuel Emission Factors
# ---------------------------------------------------------------------------

#: Emission factors by fuel type for steam and heat generation.
#: Units: kgCO2, kgCH4, kgN2O per GJ of fuel input (HHV basis).
#: Additional fields: default_efficiency (fraction 0-1), is_biogenic (bool).
#:
#: Sources:
#:   IPCC 2006 Guidelines for National Greenhouse Gas Inventories,
#:   Volume 2: Energy, Chapter 2 Table 2.2 (CO2) and Table 2.4 (CH4, N2O).
#:   US EPA AP-42 Compilation of Air Emission Factors.
#:   UK BEIS Greenhouse Gas Reporting Conversion Factors.
#:
#: Default efficiencies represent typical boiler/furnace thermal
#: efficiency (fuel energy to useful heat) for each fuel type.
#: Actual efficiencies should be obtained from equipment nameplate
#: or performance test data when available (Tier 2/3).
#:
#: Biogenic fuels (biomass_wood, biomass_biogas) have is_biogenic=True.
#: Their CO2 emissions are reported in the biogenic_co2_kg field and
#: excluded from Scope 2 fossil CO2e totals per GHG Protocol.
FUEL_EMISSION_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "natural_gas": {
        "co2_ef": Decimal("56.100"),
        "ch4_ef": Decimal("0.001"),
        "n2o_ef": Decimal("0.0001"),
        "default_efficiency": Decimal("0.85"),
        "is_biogenic": Decimal("0"),
    },
    "fuel_oil_2": {
        "co2_ef": Decimal("74.100"),
        "ch4_ef": Decimal("0.003"),
        "n2o_ef": Decimal("0.0006"),
        "default_efficiency": Decimal("0.82"),
        "is_biogenic": Decimal("0"),
    },
    "fuel_oil_6": {
        "co2_ef": Decimal("77.400"),
        "ch4_ef": Decimal("0.003"),
        "n2o_ef": Decimal("0.0006"),
        "default_efficiency": Decimal("0.80"),
        "is_biogenic": Decimal("0"),
    },
    "coal_bituminous": {
        "co2_ef": Decimal("94.600"),
        "ch4_ef": Decimal("0.001"),
        "n2o_ef": Decimal("0.0015"),
        "default_efficiency": Decimal("0.78"),
        "is_biogenic": Decimal("0"),
    },
    "coal_subbituminous": {
        "co2_ef": Decimal("96.100"),
        "ch4_ef": Decimal("0.001"),
        "n2o_ef": Decimal("0.0015"),
        "default_efficiency": Decimal("0.75"),
        "is_biogenic": Decimal("0"),
    },
    "coal_lignite": {
        "co2_ef": Decimal("101.000"),
        "ch4_ef": Decimal("0.001"),
        "n2o_ef": Decimal("0.0015"),
        "default_efficiency": Decimal("0.72"),
        "is_biogenic": Decimal("0"),
    },
    "lpg": {
        "co2_ef": Decimal("63.100"),
        "ch4_ef": Decimal("0.001"),
        "n2o_ef": Decimal("0.0001"),
        "default_efficiency": Decimal("0.85"),
        "is_biogenic": Decimal("0"),
    },
    "biomass_wood": {
        "co2_ef": Decimal("112.000"),
        "ch4_ef": Decimal("0.030"),
        "n2o_ef": Decimal("0.004"),
        "default_efficiency": Decimal("0.70"),
        "is_biogenic": Decimal("1"),
    },
    "biomass_biogas": {
        "co2_ef": Decimal("54.600"),
        "ch4_ef": Decimal("0.001"),
        "n2o_ef": Decimal("0.0001"),
        "default_efficiency": Decimal("0.80"),
        "is_biogenic": Decimal("1"),
    },
    "municipal_waste": {
        "co2_ef": Decimal("91.700"),
        "ch4_ef": Decimal("0.030"),
        "n2o_ef": Decimal("0.004"),
        "default_efficiency": Decimal("0.65"),
        "is_biogenic": Decimal("0"),
    },
    "waste_heat": {
        "co2_ef": Decimal("0.000"),
        "ch4_ef": Decimal("0.000"),
        "n2o_ef": Decimal("0.000"),
        "default_efficiency": Decimal("1.00"),
        "is_biogenic": Decimal("0"),
    },
    "geothermal": {
        "co2_ef": Decimal("0.000"),
        "ch4_ef": Decimal("0.000"),
        "n2o_ef": Decimal("0.000"),
        "default_efficiency": Decimal("1.00"),
        "is_biogenic": Decimal("0"),
    },
    "solar_thermal": {
        "co2_ef": Decimal("0.000"),
        "ch4_ef": Decimal("0.000"),
        "n2o_ef": Decimal("0.000"),
        "default_efficiency": Decimal("1.00"),
        "is_biogenic": Decimal("0"),
    },
    "electric": {
        "co2_ef": Decimal("0.000"),
        "ch4_ef": Decimal("0.000"),
        "n2o_ef": Decimal("0.000"),
        "default_efficiency": Decimal("0.98"),
        "is_biogenic": Decimal("0"),
    },
}


# ---------------------------------------------------------------------------
# District Heating Regional Factors
# ---------------------------------------------------------------------------

#: Regional emission factors for district heating networks.
#: Units: kgCO2e per GJ of delivered heat (at the meter).
#: distribution_loss_pct: fraction of heat lost in the distribution
#:     network (0-1). Applied to adjust consumption for network losses.
#:
#: Sources:
#:   Danish Energy Agency (DEA) district heating statistics.
#:   Swedish Energy Agency annual district heating report.
#:   Finnish Energy district heating statistics.
#:   German Federal Environment Agency (UBA) district heating EFs.
#:   Polish Energy Regulatory Office (URE) heating tariff data.
#:   Netherlands Enterprise Agency (RVO) heating sector data.
#:   French Environment and Energy Management Agency (ADEME).
#:   UK Department for Energy Security and Net Zero (DESNZ).
#:   US Department of Energy (DOE) district energy benchmarks.
#:   China National Bureau of Statistics heating sector data.
#:   Japan District Heating and Cooling Association statistics.
#:   South Korea Korea District Heating Corporation data.
#:   Global default: IEA World Energy Outlook weighted average.
#:
#: These factors represent the average emission intensity of district
#: heating networks in each region, accounting for the typical fuel
#: mix, plant efficiency, and network configuration. Actual supplier-
#: specific factors should be used when available (Tier 2/3).
DISTRICT_HEATING_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "denmark": {
        "ef_kgco2e_per_gj": Decimal("36.0"),
        "distribution_loss_pct": Decimal("0.10"),
    },
    "sweden": {
        "ef_kgco2e_per_gj": Decimal("18.0"),
        "distribution_loss_pct": Decimal("0.08"),
    },
    "finland": {
        "ef_kgco2e_per_gj": Decimal("55.0"),
        "distribution_loss_pct": Decimal("0.09"),
    },
    "germany": {
        "ef_kgco2e_per_gj": Decimal("72.0"),
        "distribution_loss_pct": Decimal("0.12"),
    },
    "poland": {
        "ef_kgco2e_per_gj": Decimal("105.0"),
        "distribution_loss_pct": Decimal("0.15"),
    },
    "netherlands": {
        "ef_kgco2e_per_gj": Decimal("58.0"),
        "distribution_loss_pct": Decimal("0.10"),
    },
    "france": {
        "ef_kgco2e_per_gj": Decimal("42.0"),
        "distribution_loss_pct": Decimal("0.10"),
    },
    "uk": {
        "ef_kgco2e_per_gj": Decimal("65.0"),
        "distribution_loss_pct": Decimal("0.12"),
    },
    "us": {
        "ef_kgco2e_per_gj": Decimal("75.0"),
        "distribution_loss_pct": Decimal("0.12"),
    },
    "china": {
        "ef_kgco2e_per_gj": Decimal("110.0"),
        "distribution_loss_pct": Decimal("0.15"),
    },
    "japan": {
        "ef_kgco2e_per_gj": Decimal("68.0"),
        "distribution_loss_pct": Decimal("0.10"),
    },
    "south_korea": {
        "ef_kgco2e_per_gj": Decimal("72.0"),
        "distribution_loss_pct": Decimal("0.10"),
    },
    "global_default": {
        "ef_kgco2e_per_gj": Decimal("70.0"),
        "distribution_loss_pct": Decimal("0.12"),
    },
}


# ---------------------------------------------------------------------------
# Cooling System Factors
# ---------------------------------------------------------------------------

#: Cooling system technology parameters for district cooling calculations.
#: cop_min, cop_max: Range of coefficient of performance for the technology.
#: cop_default: Default COP value used when no measured COP is available.
#: energy_source: Primary energy input type ("electricity" or "heat").
#:
#: Sources:
#:   ASHRAE Handbook - HVAC Systems and Equipment (2024).
#:   US DOE Federal Energy Management Program (FEMP) chiller benchmarks.
#:   European District Energy Association (Euroheat) cooling statistics.
#:   International District Energy Association (IDEA) design guides.
#:
#: COP = cooling_output / energy_input. Higher COP means more
#: efficient cooling per unit of energy consumed. For electric
#: chillers, energy input is electricity. For absorption chillers,
#: energy input is heat (steam or hot water).
COOLING_SYSTEM_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "centrifugal_chiller": {
        "cop_min": Decimal("5.0"),
        "cop_max": Decimal("7.0"),
        "cop_default": Decimal("6.0"),
        "energy_source": Decimal("0"),  # electricity (encoded)
    },
    "screw_chiller": {
        "cop_min": Decimal("4.0"),
        "cop_max": Decimal("5.5"),
        "cop_default": Decimal("4.5"),
        "energy_source": Decimal("0"),  # electricity
    },
    "reciprocating_chiller": {
        "cop_min": Decimal("3.5"),
        "cop_max": Decimal("5.0"),
        "cop_default": Decimal("4.0"),
        "energy_source": Decimal("0"),  # electricity
    },
    "absorption_single": {
        "cop_min": Decimal("0.6"),
        "cop_max": Decimal("0.8"),
        "cop_default": Decimal("0.7"),
        "energy_source": Decimal("1"),  # heat (encoded)
    },
    "absorption_double": {
        "cop_min": Decimal("1.0"),
        "cop_max": Decimal("1.4"),
        "cop_default": Decimal("1.2"),
        "energy_source": Decimal("1"),  # heat
    },
    "absorption_triple": {
        "cop_min": Decimal("1.5"),
        "cop_max": Decimal("1.8"),
        "cop_default": Decimal("1.6"),
        "energy_source": Decimal("1"),  # heat
    },
    "free_cooling": {
        "cop_min": Decimal("15.0"),
        "cop_max": Decimal("30.0"),
        "cop_default": Decimal("20.0"),
        "energy_source": Decimal("0"),  # electricity
    },
    "ice_storage": {
        "cop_min": Decimal("3.0"),
        "cop_max": Decimal("4.5"),
        "cop_default": Decimal("3.5"),
        "energy_source": Decimal("0"),  # electricity
    },
    "thermal_storage": {
        "cop_min": Decimal("4.0"),
        "cop_max": Decimal("6.0"),
        "cop_default": Decimal("5.0"),
        "energy_source": Decimal("0"),  # electricity
    },
}

#: String-based energy source lookup for cooling technologies.
#: Companion to COOLING_SYSTEM_FACTORS for human-readable access.
COOLING_ENERGY_SOURCE: Dict[str, str] = {
    "centrifugal_chiller": "electricity",
    "screw_chiller": "electricity",
    "reciprocating_chiller": "electricity",
    "absorption_single": "heat",
    "absorption_double": "heat",
    "absorption_triple": "heat",
    "free_cooling": "electricity",
    "ice_storage": "electricity",
    "thermal_storage": "electricity",
}


# ---------------------------------------------------------------------------
# CHP Default Efficiencies
# ---------------------------------------------------------------------------

#: Default CHP/cogeneration efficiencies by fuel type.
#: electrical_efficiency: Fraction of fuel energy converted to electricity.
#: thermal_efficiency: Fraction of fuel energy converted to useful heat.
#: overall_efficiency: Combined electrical + thermal efficiency.
#:
#: Sources:
#:   US EPA Combined Heat and Power Partnership programme data.
#:   European CHP Club (COGEN Europe) technology benchmarks.
#:   IEA CHP and District Heating Country Assessments.
#:
#: Actual CHP efficiencies vary significantly by plant size, age,
#: load factor, and technology (gas turbine, steam turbine, reciprocating
#: engine, fuel cell). These defaults represent typical mid-size
#: industrial CHP plants. Site-specific data should be used when
#: available (Tier 2/3).
CHP_DEFAULT_EFFICIENCIES: Dict[str, Dict[str, Decimal]] = {
    "natural_gas": {
        "electrical_efficiency": Decimal("0.35"),
        "thermal_efficiency": Decimal("0.45"),
        "overall_efficiency": Decimal("0.80"),
    },
    "coal": {
        "electrical_efficiency": Decimal("0.30"),
        "thermal_efficiency": Decimal("0.40"),
        "overall_efficiency": Decimal("0.70"),
    },
    "biomass": {
        "electrical_efficiency": Decimal("0.25"),
        "thermal_efficiency": Decimal("0.50"),
        "overall_efficiency": Decimal("0.75"),
    },
    "fuel_oil": {
        "electrical_efficiency": Decimal("0.32"),
        "thermal_efficiency": Decimal("0.43"),
        "overall_efficiency": Decimal("0.75"),
    },
    "municipal_waste": {
        "electrical_efficiency": Decimal("0.20"),
        "thermal_efficiency": Decimal("0.45"),
        "overall_efficiency": Decimal("0.65"),
    },
}


# ---------------------------------------------------------------------------
# Energy Unit Conversion Factors
# ---------------------------------------------------------------------------

#: Conversion factors between energy units.
#: All values are exact Decimal representations for zero-hallucination
#: deterministic arithmetic. No floating-point rounding errors.
#:
#: GJ_TO_MWH: 1 GJ = 0.277778 MWh (1/3.6, rounded to 6 dp).
#: MWH_TO_GJ: 1 MWh = 3.6 GJ (exact).
#: GJ_TO_KWH: 1 GJ = 277.778 kWh.
#: GJ_TO_MMBTU: 1 GJ = 0.947817 MMBtu (reciprocal of 1.055056).
#: MMBTU_TO_GJ: 1 MMBtu = 1.055056 GJ (EPA conversion).
#: THERM_TO_GJ: 1 therm = 0.105506 GJ (MMBtu/10).
#: GJ_TO_THERM: 1 GJ = 9.47817 therms (reciprocal).
#: MJ_TO_GJ: 1 MJ = 0.001 GJ (exact).
#: GJ_TO_MJ: 1 GJ = 1000 MJ (exact).
UNIT_CONVERSIONS: Dict[str, Decimal] = {
    "gj_to_mwh": Decimal("0.277778"),
    "mwh_to_gj": Decimal("3.6"),
    "gj_to_kwh": Decimal("277.778"),
    "gj_to_mmbtu": Decimal("0.947817"),
    "mmbtu_to_gj": Decimal("1.055056"),
    "therm_to_gj": Decimal("0.105506"),
    "gj_to_therm": Decimal("9.47817"),
    "mj_to_gj": Decimal("0.001"),
    "gj_to_mj": Decimal("1000.0"),
}


# =============================================================================
# Data Models (20)
# =============================================================================


# ---------------------------------------------------------------------------
# 1. FuelEmissionFactor
# ---------------------------------------------------------------------------


class FuelEmissionFactor(BaseModel):
    """Emission factor record for a specific fuel type.

    Encapsulates the CO2, CH4, and N2O emission factors per GJ of fuel
    input, along with the default boiler/furnace efficiency and a flag
    indicating whether the fuel is biogenic. Used as an intermediate
    data structure when looking up emission factors from the
    FUEL_EMISSION_FACTORS constant table.

    Attributes:
        fuel_type: Classification of the fuel.
        co2_ef_per_gj: CO2 emission factor in kg CO2 per GJ of fuel
            input (higher heating value basis). For biogenic fuels,
            this is the biogenic CO2 emission factor.
        ch4_ef_per_gj: CH4 emission factor in kg CH4 per GJ of fuel
            input (higher heating value basis).
        n2o_ef_per_gj: N2O emission factor in kg N2O per GJ of fuel
            input (higher heating value basis).
        default_efficiency: Default thermal efficiency of the boiler
            or furnace for this fuel type (fraction 0-1). Represents
            the fraction of fuel energy converted to useful heat.
        is_biogenic: Whether the fuel is biogenic (biomass). If True,
            CO2 emissions are reported separately and excluded from
            fossil CO2e totals per GHG Protocol guidance.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    fuel_type: FuelType = Field(
        ...,
        description="Classification of the fuel",
    )
    co2_ef_per_gj: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="CO2 emission factor in kgCO2 per GJ fuel input",
    )
    ch4_ef_per_gj: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="CH4 emission factor in kgCH4 per GJ fuel input",
    )
    n2o_ef_per_gj: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="N2O emission factor in kgN2O per GJ fuel input",
    )
    default_efficiency: Decimal = Field(
        ...,
        gt=Decimal("0"),
        le=Decimal("1"),
        description="Default boiler thermal efficiency (0-1)",
    )
    is_biogenic: bool = Field(
        ...,
        description="Whether fuel is biogenic (biomass)",
    )


# ---------------------------------------------------------------------------
# 2. DistrictHeatingFactor
# ---------------------------------------------------------------------------


class DistrictHeatingFactor(BaseModel):
    """Emission factor record for a district heating network region.

    Encapsulates the composite emission factor (kgCO2e per GJ) and
    distribution loss percentage for a specific region and network
    type. Used for district heating emission calculations when no
    supplier-specific data is available.

    Attributes:
        region: Geographic region identifier (e.g. 'denmark',
            'germany', 'global_default'). Must match a key in the
            DISTRICT_HEATING_FACTORS constant table or be a custom
            user-provided region.
        network_type: Type of district heating network. Influences
            the default distribution loss factor.
        ef_kgco2e_per_gj: Composite emission factor in kgCO2e per GJ
            of heat delivered at the building meter. Includes
            generation and distribution emissions.
        distribution_loss_pct: Fraction of heat lost in the
            distribution network (0-1). Used to gross up metered
            consumption to account for network losses when the
            reporting boundary includes distribution.
        source: Description of the data source for the emission
            factor (e.g. authority name, publication year).
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    region: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Geographic region identifier",
    )
    network_type: NetworkType = Field(
        default=NetworkType.MUNICIPAL,
        description="Type of district heating network",
    )
    ef_kgco2e_per_gj: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Emission factor in kgCO2e per GJ delivered heat",
    )
    distribution_loss_pct: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Distribution heat loss fraction (0-1)",
    )
    source: str = Field(
        default="",
        max_length=500,
        description="Data source for the emission factor",
    )

    @field_validator("region")
    @classmethod
    def _lowercase_region(cls, v: str) -> str:
        """Normalise region identifier to lowercase."""
        return v.strip().lower()


# ---------------------------------------------------------------------------
# 3. CoolingSystemFactor
# ---------------------------------------------------------------------------


class CoolingSystemFactor(BaseModel):
    """Performance parameters for a cooling system technology.

    Encapsulates the COP range and default COP value for a specific
    cooling technology, along with the primary energy source. Used
    for COP-based district cooling emission calculations.

    Attributes:
        technology: Classification of the cooling technology.
        cop_min: Minimum COP for this technology under typical
            operating conditions.
        cop_max: Maximum COP for this technology under optimal
            operating conditions.
        cop_default: Default COP value used when no measured or
            site-specific COP is available.
        energy_source: Primary energy input type. "electricity" for
            electric chillers; "heat" for absorption chillers.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    technology: CoolingTechnology = Field(
        ...,
        description="Classification of the cooling technology",
    )
    cop_min: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Minimum COP under typical conditions",
    )
    cop_max: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Maximum COP under optimal conditions",
    )
    cop_default: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Default COP when no measured value available",
    )
    energy_source: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Primary energy input type (electricity or heat)",
    )

    @field_validator("cop_max")
    @classmethod
    def _cop_max_ge_min(cls, v: Decimal, info: Any) -> Decimal:
        """Validate that cop_max is greater than or equal to cop_min."""
        cop_min = info.data.get("cop_min")
        if cop_min is not None and v < cop_min:
            raise ValueError(
                f"cop_max ({v}) must be >= cop_min ({cop_min})"
            )
        return v

    @field_validator("cop_default")
    @classmethod
    def _cop_default_in_range(cls, v: Decimal, info: Any) -> Decimal:
        """Validate that cop_default is within [cop_min, cop_max]."""
        cop_min = info.data.get("cop_min")
        cop_max = info.data.get("cop_max")
        if cop_min is not None and v < cop_min:
            raise ValueError(
                f"cop_default ({v}) must be >= cop_min ({cop_min})"
            )
        if cop_max is not None and v > cop_max:
            raise ValueError(
                f"cop_default ({v}) must be <= cop_max ({cop_max})"
            )
        return v


# ---------------------------------------------------------------------------
# 4. CHPParameters
# ---------------------------------------------------------------------------


class CHPParameters(BaseModel):
    """Configuration parameters for a CHP/cogeneration plant.

    Encapsulates the key performance characteristics of a combined
    heat and power plant needed for emission allocation calculations.

    Attributes:
        chp_id: Unique identifier for the CHP plant.
        electrical_efficiency: Fraction of fuel energy converted to
            electrical output (0-1). Typically 0.20-0.40 depending
            on technology and fuel type.
        thermal_efficiency: Fraction of fuel energy converted to
            useful thermal output (0-1). Typically 0.40-0.55.
        fuel_type: Primary fuel type used by the CHP plant.
        power_output_mw: Rated electrical power output capacity in
            megawatts. Used for plant identification and validation.
        heat_output_mw: Rated thermal heat output capacity in
            megawatts. Used for plant identification and validation.
        overall_efficiency: Combined electrical + thermal efficiency
            (0-1). Must be >= sum of individual efficiencies would
            imply, accounting for auxiliary losses.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    chp_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Unique identifier for the CHP plant",
    )
    electrical_efficiency: Decimal = Field(
        ...,
        gt=Decimal("0"),
        lt=Decimal("1"),
        description="Electrical conversion efficiency (0-1)",
    )
    thermal_efficiency: Decimal = Field(
        ...,
        gt=Decimal("0"),
        lt=Decimal("1"),
        description="Thermal conversion efficiency (0-1)",
    )
    fuel_type: FuelType = Field(
        ...,
        description="Primary fuel type used by the CHP plant",
    )
    power_output_mw: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Rated electrical power output in MW",
    )
    heat_output_mw: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Rated thermal heat output in MW",
    )
    overall_efficiency: Decimal = Field(
        ...,
        gt=Decimal("0"),
        le=Decimal("1"),
        description="Combined electrical + thermal efficiency (0-1)",
    )

    @field_validator("overall_efficiency")
    @classmethod
    def _overall_ge_components(cls, v: Decimal, info: Any) -> Decimal:
        """Validate overall efficiency >= electrical + thermal."""
        elec = info.data.get("electrical_efficiency")
        therm = info.data.get("thermal_efficiency")
        if elec is not None and therm is not None:
            component_sum = elec + therm
            if v < component_sum:
                raise ValueError(
                    f"overall_efficiency ({v}) must be >= "
                    f"electrical ({elec}) + thermal ({therm}) = "
                    f"{component_sum}"
                )
        return v


# ---------------------------------------------------------------------------
# 5. FacilityInfo
# ---------------------------------------------------------------------------


class FacilityInfo(BaseModel):
    """Metadata record for a reporting facility consuming thermal energy.

    Represents a single physical facility (building, campus, or site)
    for which Scope 2 steam, heating, and cooling emissions are
    calculated. Each facility may be connected to one or more steam
    suppliers, a district heating network, and/or a district cooling
    system.

    Attributes:
        facility_id: Unique system identifier for the facility (UUID).
        name: Human-readable facility name or label.
        facility_type: Classification of facility by primary function.
        country: ISO 3166-1 alpha-2 country code for the facility
            location. Used for regional emission factor lookup.
        region: Geographic region or sub-region identifier. Used for
            district heating factor lookup (e.g. 'germany', 'sweden').
        latitude: Geographic latitude in decimal degrees. Optional,
            used for mapping and spatial analysis.
        longitude: Geographic longitude in decimal degrees. Optional,
            used for mapping and spatial analysis.
        steam_suppliers: List of steam supplier IDs connected to this
            facility. References SteamSupplier records.
        heating_network: Identifier of the district heating network
            serving this facility. None if not connected.
        cooling_system: Identifier of the district cooling system
            serving this facility. None if not connected.
        tenant_id: Owning tenant identifier for multi-tenancy.
        created_at: UTC timestamp of the facility record creation.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    facility_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique system identifier for the facility (UUID)",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Human-readable facility name",
    )
    facility_type: FacilityType = Field(
        ...,
        description="Classification of facility by primary function",
    )
    country: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    region: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Geographic region for heating factor lookup",
    )
    latitude: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("-90"),
        le=Decimal("90"),
        description="Geographic latitude in decimal degrees",
    )
    longitude: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("-180"),
        le=Decimal("180"),
        description="Geographic longitude in decimal degrees",
    )
    steam_suppliers: List[str] = Field(
        default_factory=list,
        description="List of steam supplier IDs connected to facility",
    )
    heating_network: Optional[str] = Field(
        default=None,
        max_length=200,
        description="District heating network identifier",
    )
    cooling_system: Optional[str] = Field(
        default=None,
        max_length=200,
        description="District cooling system identifier",
    )
    tenant_id: str = Field(
        default="default",
        min_length=1,
        max_length=200,
        description="Owning tenant identifier for multi-tenancy",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of facility record creation",
    )

    @field_validator("country")
    @classmethod
    def _uppercase_country(cls, v: str) -> str:
        """Normalise country code to uppercase."""
        return v.upper()

    @field_validator("region")
    @classmethod
    def _lowercase_region(cls, v: str) -> str:
        """Normalise region identifier to lowercase."""
        return v.strip().lower()

    @field_validator("steam_suppliers")
    @classmethod
    def _validate_suppliers_size(cls, v: List[str]) -> List[str]:
        """Validate that supplier count does not exceed reasonable limit."""
        if len(v) > 100:
            raise ValueError(
                f"Steam supplier count {len(v)} exceeds maximum 100"
            )
        return v


# ---------------------------------------------------------------------------
# 6. SteamSupplier
# ---------------------------------------------------------------------------


class SteamSupplier(BaseModel):
    """Profile of a steam or heat supplier for emission calculations.

    Represents an external entity that generates and delivers steam
    or hot water to the reporting facility. The supplier profile
    includes fuel mix, boiler efficiency, and optionally a verified
    composite emission factor. Used for fuel-based and direct EF
    calculations.

    Attributes:
        supplier_id: Unique identifier for the steam supplier.
        name: Human-readable supplier name.
        fuel_mix: Dictionary of fuel type proportions in the supplier's
            generation mix. Keys are FuelType values, values are
            fractions (0-1) that should sum to 1.0.
        boiler_efficiency: Overall boiler thermal efficiency (0-1).
            If None, the default efficiency for the primary fuel type
            is used.
        supplier_ef_kgco2e_per_gj: Supplier-disclosed composite
            emission factor in kgCO2e per GJ. If provided, used for
            DIRECT_EF calculation method. Takes precedence over
            fuel-based calculation when the DIRECT_EF method is
            selected.
        country: ISO 3166-1 alpha-2 country code where the supplier
            operates.
        region: Geographic region for regional factor lookup.
        verified: Whether the supplier's emission data has been
            independently verified or audited by a third party.
        data_quality_tier: Data quality tier of the supplier's data.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    supplier_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the steam supplier",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Human-readable supplier name",
    )
    fuel_mix: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Fuel type proportions in supplier's generation mix",
    )
    boiler_efficiency: Optional[Decimal] = Field(
        default=None,
        gt=Decimal("0"),
        le=Decimal("1"),
        description="Overall boiler thermal efficiency (0-1)",
    )
    supplier_ef_kgco2e_per_gj: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description="Supplier composite EF in kgCO2e per GJ",
    )
    country: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    region: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Geographic region for factor lookup",
    )
    verified: bool = Field(
        default=False,
        description="Whether independently verified by a third party",
    )
    data_quality_tier: DataQualityTier = Field(
        default=DataQualityTier.TIER_1,
        description="Data quality tier of supplier's data",
    )

    @field_validator("country")
    @classmethod
    def _uppercase_country(cls, v: str) -> str:
        """Normalise country code to uppercase."""
        return v.upper()

    @field_validator("fuel_mix")
    @classmethod
    def _validate_fuel_mix(cls, v: Dict[str, Decimal]) -> Dict[str, Decimal]:
        """Validate fuel mix proportions are non-negative."""
        for fuel, fraction in v.items():
            if fraction < Decimal("0"):
                raise ValueError(
                    f"Fuel mix fraction for '{fuel}' must be >= 0, "
                    f"got {fraction}"
                )
        return v


# ---------------------------------------------------------------------------
# 7. SteamCalculationRequest
# ---------------------------------------------------------------------------


class SteamCalculationRequest(BaseModel):
    """Request for a purchased steam emission calculation.

    Contains all parameters needed to calculate Scope 2 emissions
    from purchased steam consumption at a facility. Supports both
    direct emission factor and fuel-based calculation methods.

    Attributes:
        facility_id: Reference to the consuming facility.
        consumption_gj: Steam consumption quantity in gigajoules
            (at the building meter).
        energy_type: Type of thermal energy. Defaults to STEAM.
        supplier_id: Optional reference to the steam supplier for
            supplier-specific emission factor or fuel mix lookup.
        fuel_type: Optional explicit fuel type override. If provided,
            overrides the supplier's fuel mix for the calculation.
        boiler_efficiency: Optional explicit boiler efficiency
            override (0-1). If provided, overrides the supplier's
            boiler efficiency.
        steam_pressure: Optional steam pressure classification.
            Used for enthalpy adjustment in advanced calculations.
        steam_quality: Optional steam quality classification. Used
            for dryness fraction adjustment in advanced calculations.
        condensate_return_pct: Percentage of condensate returned to
            the boiler (0-100 as a Decimal fraction 0-1 of metered
            steam). Reduces the effective steam consumption by the
            energy recovered in the returned condensate.
        gwp_source: IPCC Assessment Report for GWP values.
        data_quality_tier: Data quality tier for the calculation.
        reporting_period: Time period for the calculation.
        calculation_date: Date/time the calculation is performed.
        tenant_id: Owning tenant identifier for multi-tenancy.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    facility_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Reference to the consuming facility",
    )
    consumption_gj: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Steam consumption in GJ at the building meter",
    )
    energy_type: EnergyType = Field(
        default=EnergyType.STEAM,
        description="Type of thermal energy",
    )
    supplier_id: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Reference to the steam supplier",
    )
    fuel_type: Optional[FuelType] = Field(
        default=None,
        description="Explicit fuel type override for the calculation",
    )
    boiler_efficiency: Optional[Decimal] = Field(
        default=None,
        gt=Decimal("0"),
        le=Decimal("1"),
        description="Explicit boiler efficiency override (0-1)",
    )
    steam_pressure: Optional[SteamPressure] = Field(
        default=None,
        description="Steam pressure classification for enthalpy adjustment",
    )
    steam_quality: Optional[SteamQuality] = Field(
        default=None,
        description="Steam quality for dryness fraction adjustment",
    )
    condensate_return_pct: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Condensate return percentage (0-100)",
    )
    gwp_source: GWPSource = Field(
        default=GWPSource.AR6,
        description="IPCC Assessment Report for GWP values",
    )
    data_quality_tier: DataQualityTier = Field(
        default=DataQualityTier.TIER_1,
        description="Data quality tier for the calculation",
    )
    reporting_period: ReportingPeriod = Field(
        default=ReportingPeriod.ANNUAL,
        description="Time period for the calculation",
    )
    calculation_date: datetime = Field(
        default_factory=_utcnow,
        description="Date/time the calculation is performed",
    )
    tenant_id: str = Field(
        default="default",
        min_length=1,
        max_length=200,
        description="Owning tenant identifier for multi-tenancy",
    )


# ---------------------------------------------------------------------------
# 8. HeatingCalculationRequest
# ---------------------------------------------------------------------------


class HeatingCalculationRequest(BaseModel):
    """Request for a district heating emission calculation.

    Contains all parameters needed to calculate Scope 2 emissions
    from purchased district heating consumption at a facility.

    Attributes:
        facility_id: Reference to the consuming facility.
        consumption_gj: District heating consumption quantity in
            gigajoules (at the building meter).
        region: Geographic region identifier for regional factor
            lookup (e.g. 'denmark', 'germany', 'global_default').
        network_type: Type of district heating network. Influences
            default distribution loss factor.
        supplier_ef_kgco2e_per_gj: Optional supplier-specific
            emission factor in kgCO2e per GJ. If provided, overrides
            the regional default factor.
        distribution_loss_pct: Optional explicit distribution loss
            percentage override (0-1). If provided, overrides the
            regional default.
        gwp_source: IPCC Assessment Report for GWP values.
        data_quality_tier: Data quality tier for the calculation.
        tenant_id: Owning tenant identifier for multi-tenancy.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    facility_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Reference to the consuming facility",
    )
    consumption_gj: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="District heating consumption in GJ",
    )
    region: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Geographic region for factor lookup",
    )
    network_type: NetworkType = Field(
        default=NetworkType.MUNICIPAL,
        description="Type of district heating network",
    )
    supplier_ef_kgco2e_per_gj: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description="Supplier-specific EF override in kgCO2e per GJ",
    )
    distribution_loss_pct: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Distribution loss percentage override (0-1)",
    )
    gwp_source: GWPSource = Field(
        default=GWPSource.AR6,
        description="IPCC Assessment Report for GWP values",
    )
    data_quality_tier: DataQualityTier = Field(
        default=DataQualityTier.TIER_1,
        description="Data quality tier for the calculation",
    )
    tenant_id: str = Field(
        default="default",
        min_length=1,
        max_length=200,
        description="Owning tenant identifier for multi-tenancy",
    )

    @field_validator("region")
    @classmethod
    def _lowercase_region(cls, v: str) -> str:
        """Normalise region identifier to lowercase."""
        return v.strip().lower()


# ---------------------------------------------------------------------------
# 9. CoolingCalculationRequest
# ---------------------------------------------------------------------------


class CoolingCalculationRequest(BaseModel):
    """Request for a district cooling emission calculation.

    Contains all parameters needed to calculate Scope 2 emissions
    from purchased district cooling consumption at a facility using
    the COP-based method.

    Attributes:
        facility_id: Reference to the consuming facility.
        cooling_output_gj: Cooling output delivered to the facility
            in gigajoules.
        technology: Cooling technology type. Determines the default
            COP and energy source.
        cop: Optional measured or site-specific COP override. If
            provided, overrides the default COP for the technology.
        grid_ef_kgco2e_per_kwh: Optional grid electricity emission
            factor in kgCO2e per kWh. Required for electric chillers
            when not using a default regional factor.
        heat_source_ef_kgco2e_per_gj: Optional heat source emission
            factor in kgCO2e per GJ. Required for absorption chillers
            when the heat is from a specific source.
        gwp_source: IPCC Assessment Report for GWP values.
        data_quality_tier: Data quality tier for the calculation.
        tenant_id: Owning tenant identifier for multi-tenancy.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    facility_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Reference to the consuming facility",
    )
    cooling_output_gj: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Cooling output delivered in GJ",
    )
    technology: CoolingTechnology = Field(
        ...,
        description="Cooling technology type",
    )
    cop: Optional[Decimal] = Field(
        default=None,
        gt=Decimal("0"),
        description="Measured COP override for the cooling system",
    )
    grid_ef_kgco2e_per_kwh: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description="Grid electricity EF in kgCO2e per kWh",
    )
    heat_source_ef_kgco2e_per_gj: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description="Heat source EF in kgCO2e per GJ",
    )
    gwp_source: GWPSource = Field(
        default=GWPSource.AR6,
        description="IPCC Assessment Report for GWP values",
    )
    data_quality_tier: DataQualityTier = Field(
        default=DataQualityTier.TIER_1,
        description="Data quality tier for the calculation",
    )
    tenant_id: str = Field(
        default="default",
        min_length=1,
        max_length=200,
        description="Owning tenant identifier for multi-tenancy",
    )


# ---------------------------------------------------------------------------
# 10. CHPAllocationRequest
# ---------------------------------------------------------------------------


class CHPAllocationRequest(BaseModel):
    """Request for CHP emission allocation calculation.

    Contains all parameters needed to allocate total fuel emissions
    from a CHP/cogeneration plant between its electrical and thermal
    outputs using one of three allocation methods.

    Attributes:
        facility_id: Reference to the consuming facility receiving
            the thermal output.
        total_fuel_gj: Total fuel input to the CHP plant in GJ for
            the reporting period.
        fuel_type: Primary fuel type used by the CHP plant.
        heat_output_gj: Useful thermal (heat) output in GJ for the
            reporting period.
        power_output_gj: Electrical power output in GJ for the
            reporting period.
        cooling_output_gj: Cooling output from absorption chiller
            driven by CHP heat, in GJ. Defaults to zero if no
            cooling is produced.
        method: Allocation method for apportioning emissions.
        electrical_efficiency: Optional explicit electrical efficiency
            override (0-1). If None, defaults from CHP_DEFAULT_EFFICIENCIES
            are used based on fuel_type.
        thermal_efficiency: Optional explicit thermal efficiency
            override (0-1). If None, defaults from CHP_DEFAULT_EFFICIENCIES
            are used based on fuel_type.
        steam_temperature_c: Steam or hot water supply temperature
            in degrees Celsius. Required for exergy-based allocation
            (Carnot factor calculation). Optional for other methods.
        ambient_temperature_c: Ambient reference temperature in
            degrees Celsius. Used in exergy-based allocation for the
            Carnot factor. Defaults to 25 degrees C (298.15 K).
        gwp_source: IPCC Assessment Report for GWP values.
        tenant_id: Owning tenant identifier for multi-tenancy.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    facility_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Reference to the consuming facility",
    )
    total_fuel_gj: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Total fuel input to the CHP plant in GJ",
    )
    fuel_type: FuelType = Field(
        ...,
        description="Primary fuel type used by the CHP plant",
    )
    heat_output_gj: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Useful thermal output in GJ",
    )
    power_output_gj: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Electrical power output in GJ",
    )
    cooling_output_gj: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Cooling output in GJ (0 if no cooling)",
    )
    method: CHPAllocMethod = Field(
        default=CHPAllocMethod.EFFICIENCY,
        description="Allocation method for emission apportionment",
    )
    electrical_efficiency: Optional[Decimal] = Field(
        default=None,
        gt=Decimal("0"),
        lt=Decimal("1"),
        description="Electrical efficiency override (0-1)",
    )
    thermal_efficiency: Optional[Decimal] = Field(
        default=None,
        gt=Decimal("0"),
        lt=Decimal("1"),
        description="Thermal efficiency override (0-1)",
    )
    steam_temperature_c: Optional[Decimal] = Field(
        default=None,
        gt=Decimal("-273.15"),
        description="Steam supply temperature in degrees Celsius",
    )
    ambient_temperature_c: Decimal = Field(
        default=Decimal("25"),
        gt=Decimal("-273.15"),
        description="Ambient reference temperature in degrees Celsius",
    )
    gwp_source: GWPSource = Field(
        default=GWPSource.AR6,
        description="IPCC Assessment Report for GWP values",
    )
    tenant_id: str = Field(
        default="default",
        min_length=1,
        max_length=200,
        description="Owning tenant identifier for multi-tenancy",
    )

    @field_validator("steam_temperature_c")
    @classmethod
    def _steam_temp_above_ambient(
        cls, v: Optional[Decimal], info: Any,
    ) -> Optional[Decimal]:
        """Warn if steam temperature is not above ambient (for exergy)."""
        if v is None:
            return v
        ambient = info.data.get("ambient_temperature_c")
        if ambient is not None and v <= ambient:
            raise ValueError(
                f"steam_temperature_c ({v}) must be above "
                f"ambient_temperature_c ({ambient}) for meaningful "
                f"thermal energy delivery"
            )
        return v


# ---------------------------------------------------------------------------
# 11. GasEmissionDetail
# ---------------------------------------------------------------------------


class GasEmissionDetail(BaseModel):
    """Breakdown of emissions for a single greenhouse gas species.

    Provides the individual gas emission quantity, the GWP multiplier
    used, the GWP source, and the resulting CO2-equivalent value.
    Used as an element in the gas_details list of CalculationResult.

    Attributes:
        gas: Greenhouse gas species.
        emission_kg: Direct emission quantity in kilograms of the
            gas species.
        gwp_value: GWP multiplier applied for CO2e conversion.
            For CO2, this is always 1. For CH4 and N2O, depends
            on the selected GWP source (AR4/AR5/AR6/AR6_20YR).
        gwp_source: IPCC Assessment Report edition used for the
            GWP value.
        co2e_kg: CO2-equivalent emission in kilograms. Calculated
            as emission_kg * gwp_value.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
    )

    gas: EmissionGas = Field(
        ...,
        description="Greenhouse gas species",
    )
    emission_kg: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Direct emission quantity in kg",
    )
    gwp_value: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="GWP multiplier for CO2e conversion",
    )
    gwp_source: GWPSource = Field(
        ...,
        description="IPCC AR edition for the GWP value",
    )
    co2e_kg: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="CO2-equivalent emission in kg",
    )


# ---------------------------------------------------------------------------
# 12. CalculationResult
# ---------------------------------------------------------------------------


class CalculationResult(BaseModel):
    """Result of a Scope 2 thermal energy emission calculation.

    Contains the complete calculation output including total CO2e,
    fossil CO2e, biogenic CO2, per-gas breakdown, effective emission
    factor, and SHA-256 provenance hash for audit trail.

    Attributes:
        calc_id: Unique identifier for this calculation result (UUID).
        status: Processing status string (SUCCESS or FAILED).
        energy_type: Type of thermal energy calculated.
        calculation_method: Methodology used for the calculation.
        total_co2e_kg: Total CO2-equivalent emissions in kilograms
            (fossil + biogenic CH4/N2O contributions).
        fossil_co2e_kg: Fossil-only CO2-equivalent emissions in kg.
            Excludes biogenic CO2 but includes CH4 and N2O from
            biogenic fuel combustion weighted by GWP.
        biogenic_co2_kg: Biogenic CO2 emissions in kilograms.
            Reported separately per GHG Protocol guidance.
        gas_details: List of per-gas emission breakdowns.
        consumption_gj: Input thermal energy consumption in GJ.
        effective_ef_kgco2e_per_gj: Effective emission factor
            computed as total_co2e_kg / consumption_gj.
        data_quality_tier: Data quality tier of the calculation.
        trace: Ordered list of calculation trace steps for
            transparency and debugging. Each entry describes a
            calculation step or data source decision.
        provenance_hash: SHA-256 hash of all calculation inputs and
            outputs for complete audit trail.
        calculated_at: UTC timestamp of the calculation.
        tenant_id: Owning tenant identifier for multi-tenancy.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    calc_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique calculation result identifier (UUID)",
    )
    status: str = Field(
        default="SUCCESS",
        max_length=50,
        description="Processing status (SUCCESS or FAILED)",
    )
    energy_type: EnergyType = Field(
        ...,
        description="Type of thermal energy calculated",
    )
    calculation_method: CalculationMethod = Field(
        ...,
        description="Methodology used for the calculation",
    )
    total_co2e_kg: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Total CO2-equivalent emissions in kg",
    )
    fossil_co2e_kg: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Fossil-only CO2-equivalent emissions in kg",
    )
    biogenic_co2_kg: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Biogenic CO2 emissions in kg",
    )
    gas_details: List[GasEmissionDetail] = Field(
        default_factory=list,
        description="Per-gas emission breakdowns",
    )
    consumption_gj: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Input thermal energy consumption in GJ",
    )
    effective_ef_kgco2e_per_gj: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Effective emission factor (kgCO2e per GJ)",
    )
    data_quality_tier: DataQualityTier = Field(
        ...,
        description="Data quality tier of the calculation",
    )
    trace: List[str] = Field(
        default_factory=list,
        description="Ordered calculation trace steps",
    )
    provenance_hash: str = Field(
        default="",
        max_length=64,
        description="SHA-256 provenance hash for audit trail",
    )
    calculated_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of the calculation",
    )
    tenant_id: str = Field(
        default="default",
        min_length=1,
        max_length=200,
        description="Owning tenant identifier for multi-tenancy",
    )

    @field_validator("gas_details")
    @classmethod
    def _validate_gas_details_size(
        cls, v: List[GasEmissionDetail],
    ) -> List[GasEmissionDetail]:
        """Validate that gas detail count does not exceed maximum."""
        if len(v) > MAX_GASES_PER_RESULT:
            raise ValueError(
                f"Gas detail count {len(v)} exceeds maximum "
                f"{MAX_GASES_PER_RESULT}"
            )
        return v

    @field_validator("trace")
    @classmethod
    def _validate_trace_size(cls, v: List[str]) -> List[str]:
        """Validate that trace step count does not exceed maximum."""
        if len(v) > MAX_TRACE_STEPS:
            raise ValueError(
                f"Trace step count {len(v)} exceeds maximum "
                f"{MAX_TRACE_STEPS}"
            )
        return v


# ---------------------------------------------------------------------------
# 13. CHPAllocationResult
# ---------------------------------------------------------------------------


class CHPAllocationResult(BaseModel):
    """Result of a CHP emission allocation calculation.

    Contains the allocation shares, allocated emissions for heat,
    power, and cooling, total fuel emissions, primary energy savings,
    and provenance tracking.

    Attributes:
        allocation_id: Unique identifier for this allocation (UUID).
        method: Allocation method used (efficiency, energy, exergy).
        heat_share: Fraction of total emissions allocated to heat
            output (0-1).
        power_share: Fraction of total emissions allocated to power
            output (0-1).
        cooling_share: Fraction of total emissions allocated to
            cooling output (0-1). Zero if no cooling produced.
        heat_emissions_kgco2e: Emissions allocated to thermal output
            in kgCO2e.
        power_emissions_kgco2e: Emissions allocated to electrical
            output in kgCO2e.
        cooling_emissions_kgco2e: Emissions allocated to cooling
            output in kgCO2e. Zero if no cooling produced.
        total_fuel_emissions_kgco2e: Total fuel combustion emissions
            in kgCO2e before allocation.
        primary_energy_savings_pct: Primary energy savings percentage
            of the CHP plant compared to separate generation. A
            positive value indicates the CHP is more efficient than
            separate production of heat and electricity.
        trace: Ordered list of allocation trace steps.
        provenance_hash: SHA-256 hash for audit trail.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    allocation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique allocation result identifier (UUID)",
    )
    method: CHPAllocMethod = Field(
        ...,
        description="Allocation method used",
    )
    heat_share: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Fraction allocated to heat output (0-1)",
    )
    power_share: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Fraction allocated to power output (0-1)",
    )
    cooling_share: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Fraction allocated to cooling output (0-1)",
    )
    heat_emissions_kgco2e: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Emissions allocated to thermal output in kgCO2e",
    )
    power_emissions_kgco2e: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Emissions allocated to electrical output in kgCO2e",
    )
    cooling_emissions_kgco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Emissions allocated to cooling output in kgCO2e",
    )
    total_fuel_emissions_kgco2e: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Total fuel emissions before allocation in kgCO2e",
    )
    primary_energy_savings_pct: Decimal = Field(
        default=Decimal("0"),
        description="Primary energy savings vs separate generation (%)",
    )
    trace: List[str] = Field(
        default_factory=list,
        description="Ordered allocation trace steps",
    )
    provenance_hash: str = Field(
        default="",
        max_length=64,
        description="SHA-256 provenance hash for audit trail",
    )

    @field_validator("trace")
    @classmethod
    def _validate_trace_size(cls, v: List[str]) -> List[str]:
        """Validate that trace step count does not exceed maximum."""
        if len(v) > MAX_TRACE_STEPS:
            raise ValueError(
                f"Trace step count {len(v)} exceeds maximum "
                f"{MAX_TRACE_STEPS}"
            )
        return v


# ---------------------------------------------------------------------------
# 14. BatchCalculationRequest
# ---------------------------------------------------------------------------


class BatchCalculationRequest(BaseModel):
    """Batch request for multiple thermal energy emission calculations.

    Aggregates multiple calculation request instances (steam, heating,
    cooling, or CHP allocation) for parallel processing across a
    portfolio of facilities.

    Attributes:
        requests: List of individual calculation requests. Each
            element may be a SteamCalculationRequest,
            HeatingCalculationRequest, CoolingCalculationRequest,
            or CHPAllocationRequest (stored as Any for flexibility).
        tenant_id: Owning tenant identifier for multi-tenancy.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    requests: List[Any] = Field(
        ...,
        min_length=1,
        description="List of individual calculation requests",
    )
    tenant_id: str = Field(
        default="default",
        min_length=1,
        max_length=200,
        description="Owning tenant identifier for multi-tenancy",
    )

    @field_validator("requests")
    @classmethod
    def _validate_batch_size(cls, v: List[Any]) -> List[Any]:
        """Validate that batch size does not exceed maximum."""
        if len(v) > MAX_CALCULATIONS_PER_BATCH:
            raise ValueError(
                f"Batch size {len(v)} exceeds maximum "
                f"{MAX_CALCULATIONS_PER_BATCH}"
            )
        return v


# ---------------------------------------------------------------------------
# 15. BatchCalculationResult
# ---------------------------------------------------------------------------


class BatchCalculationResult(BaseModel):
    """Result of a batch thermal energy emission calculation.

    Aggregates results from all individual calculations in a batch
    with portfolio-level totals and status tracking.

    Attributes:
        batch_id: Unique identifier for this batch result (UUID).
        results: List of individual CalculationResult instances.
        total_co2e_kg: Portfolio-level total CO2e in kilograms
            across all successful calculations.
        total_fossil_co2e_kg: Portfolio-level total fossil CO2e
            in kilograms.
        total_biogenic_co2_kg: Portfolio-level total biogenic CO2
            in kilograms.
        success_count: Number of calculations that completed
            successfully.
        failure_count: Number of calculations that failed.
        status: Overall batch processing status.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    batch_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique batch result identifier (UUID)",
    )
    results: List[CalculationResult] = Field(
        default_factory=list,
        description="List of individual calculation results",
    )
    total_co2e_kg: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Portfolio total CO2e in kg",
    )
    total_fossil_co2e_kg: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Portfolio total fossil CO2e in kg",
    )
    total_biogenic_co2_kg: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Portfolio total biogenic CO2 in kg",
    )
    success_count: int = Field(
        ...,
        ge=0,
        description="Number of successful calculations",
    )
    failure_count: int = Field(
        ...,
        ge=0,
        description="Number of failed calculations",
    )
    status: BatchStatus = Field(
        ...,
        description="Overall batch processing status",
    )


# ---------------------------------------------------------------------------
# 16. UncertaintyRequest
# ---------------------------------------------------------------------------


class UncertaintyRequest(BaseModel):
    """Request for uncertainty quantification on a calculation result.

    Specifies the calculation result to analyse and the uncertainty
    method parameters (Monte Carlo simulation or analytical
    propagation), along with uncertainty percentages for each
    input parameter category.

    Attributes:
        calc_result: The CalculationResult to analyse for uncertainty.
        method: Uncertainty quantification method ('monte_carlo' or
            'analytical').
        iterations: Number of Monte Carlo iterations (applicable only
            when method is 'monte_carlo'). Higher values give more
            precise uncertainty estimates but take longer to compute.
        confidence_level: Confidence level for the uncertainty
            interval (e.g. 0.95 for 95% CI).
        activity_data_uncertainty_pct: Uncertainty in the activity
            data (consumption quantity) as a percentage. Typical
            range: 2-10% depending on metering quality.
        emission_factor_uncertainty_pct: Uncertainty in the emission
            factor as a percentage. Typical range: 5-25% depending
            on data quality tier and fuel type.
        efficiency_uncertainty_pct: Uncertainty in the boiler or CHP
            efficiency as a percentage. Typical range: 2-10%.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    calc_result: CalculationResult = Field(
        ...,
        description="Calculation result to analyse for uncertainty",
    )
    method: str = Field(
        default="monte_carlo",
        max_length=50,
        description="Uncertainty method (monte_carlo or analytical)",
    )
    iterations: int = Field(
        default=DEFAULT_MONTE_CARLO_ITERATIONS,
        ge=100,
        le=1_000_000,
        description="Number of Monte Carlo iterations",
    )
    confidence_level: Decimal = Field(
        default=DEFAULT_CONFIDENCE_LEVEL,
        ge=Decimal("0.50"),
        le=Decimal("0.9999"),
        description="Confidence level for the uncertainty interval",
    )
    activity_data_uncertainty_pct: Decimal = Field(
        default=Decimal("5.0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Activity data uncertainty percentage",
    )
    emission_factor_uncertainty_pct: Decimal = Field(
        default=Decimal("10.0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Emission factor uncertainty percentage",
    )
    efficiency_uncertainty_pct: Decimal = Field(
        default=Decimal("5.0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Efficiency uncertainty percentage",
    )

    @field_validator("method")
    @classmethod
    def _validate_method(cls, v: str) -> str:
        """Validate that method is one of the allowed values."""
        allowed = {"monte_carlo", "analytical"}
        if v not in allowed:
            raise ValueError(
                f"method must be one of {allowed}, got '{v}'"
            )
        return v


# ---------------------------------------------------------------------------
# 17. UncertaintyResult
# ---------------------------------------------------------------------------


class UncertaintyResult(BaseModel):
    """Result of uncertainty quantification for a thermal energy calculation.

    Provides the mean, standard deviation, confidence interval, and
    relative uncertainty for the CO2e emission estimate.

    Attributes:
        uncertainty_id: Unique identifier for this uncertainty result.
        mean_co2e_kg: Mean CO2e estimate in kilograms from the
            uncertainty analysis.
        std_dev_kg: Standard deviation of the CO2e estimate in kg.
        ci_lower_kg: Lower bound of the confidence interval in kg.
        ci_upper_kg: Upper bound of the confidence interval in kg.
        confidence_level: Confidence level of the interval (e.g. 0.95).
        method: Uncertainty method used ('monte_carlo' or 'analytical').
        relative_uncertainty_pct: Relative uncertainty as a percentage
            of the mean value. Calculated as (ci_upper - ci_lower) /
            (2 * mean) * 100.
        provenance_hash: SHA-256 hash for audit trail.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
    )

    uncertainty_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique uncertainty result identifier",
    )
    mean_co2e_kg: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Mean CO2e estimate in kg",
    )
    std_dev_kg: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Standard deviation in kg",
    )
    ci_lower_kg: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Lower bound of confidence interval in kg",
    )
    ci_upper_kg: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Upper bound of confidence interval in kg",
    )
    confidence_level: Decimal = Field(
        ...,
        ge=Decimal("0.50"),
        le=Decimal("0.9999"),
        description="Confidence level of the interval",
    )
    method: str = Field(
        ...,
        max_length=50,
        description="Uncertainty method used",
    )
    relative_uncertainty_pct: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Relative uncertainty as percentage of mean",
    )
    provenance_hash: str = Field(
        default="",
        max_length=64,
        description="SHA-256 provenance hash for audit trail",
    )


# ---------------------------------------------------------------------------
# 18. ComplianceCheckResult
# ---------------------------------------------------------------------------


class ComplianceCheckResult(BaseModel):
    """Result of a regulatory compliance check for a calculation.

    Evaluates a completed thermal energy calculation against a
    specific regulatory framework (GHG Protocol, CSRD, ISO 14064,
    CDP, etc.) and reports findings and score.

    Attributes:
        framework: Regulatory framework identifier (e.g.
            'GHG_PROTOCOL', 'CSRD', 'ISO_14064', 'CDP').
        status: Overall compliance status.
        total_requirements: Total number of requirements checked
            for the framework.
        met_requirements: Number of requirements that are fully
            satisfied.
        findings: List of specific compliance findings. Each string
            describes a requirement and its assessment result.
        score_pct: Compliance score as a percentage (0-100).
            Calculated as met_requirements / total_requirements * 100.
        checked_at: UTC timestamp of the compliance check.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    framework: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Regulatory framework identifier",
    )
    status: ComplianceStatus = Field(
        ...,
        description="Overall compliance status",
    )
    total_requirements: int = Field(
        ...,
        ge=0,
        description="Total number of requirements checked",
    )
    met_requirements: int = Field(
        ...,
        ge=0,
        description="Number of requirements fully satisfied",
    )
    findings: List[str] = Field(
        default_factory=list,
        description="Specific compliance findings",
    )
    score_pct: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Compliance score percentage (0-100)",
    )
    checked_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of the compliance check",
    )

    @field_validator("met_requirements")
    @classmethod
    def _met_le_total(cls, v: int, info: Any) -> int:
        """Validate that met requirements do not exceed total."""
        total = info.data.get("total_requirements")
        if total is not None and v > total:
            raise ValueError(
                f"met_requirements ({v}) cannot exceed "
                f"total_requirements ({total})"
            )
        return v

    @field_validator("findings")
    @classmethod
    def _validate_findings_size(cls, v: List[str]) -> List[str]:
        """Validate that findings count is reasonable."""
        if len(v) > 500:
            raise ValueError(
                f"Findings count {len(v)} exceeds maximum 500"
            )
        return v


# ---------------------------------------------------------------------------
# 19. AggregationRequest
# ---------------------------------------------------------------------------


class AggregationRequest(BaseModel):
    """Request for aggregating multiple calculation results.

    Specifies which calculation results to aggregate and the
    aggregation dimension.

    Attributes:
        calc_ids: List of CalculationResult calc_id values to include
            in the aggregation.
        aggregation_type: Dimension for grouping results.
        tenant_id: Owning tenant identifier for multi-tenancy.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    calc_ids: List[str] = Field(
        ...,
        min_length=1,
        description="List of calculation result IDs to aggregate",
    )
    aggregation_type: AggregationType = Field(
        ...,
        description="Dimension for grouping results",
    )
    tenant_id: str = Field(
        default="default",
        min_length=1,
        max_length=200,
        description="Owning tenant identifier for multi-tenancy",
    )

    @field_validator("calc_ids")
    @classmethod
    def _validate_calc_ids_size(cls, v: List[str]) -> List[str]:
        """Validate that calc_ids count does not exceed batch maximum."""
        if len(v) > MAX_CALCULATIONS_PER_BATCH:
            raise ValueError(
                f"Calc ID count {len(v)} exceeds maximum "
                f"{MAX_CALCULATIONS_PER_BATCH}"
            )
        return v


# ---------------------------------------------------------------------------
# 20. AggregationResult
# ---------------------------------------------------------------------------


class AggregationResult(BaseModel):
    """Aggregated emission result across multiple calculations.

    Provides portfolio-level or group-level totals for Scope 2
    thermal energy emissions, grouped by a specified dimension
    (facility, fuel, energy type, supplier, or time period).

    Attributes:
        aggregation_id: Unique identifier for this aggregation (UUID).
        aggregation_type: Dimension used for grouping.
        total_co2e_kg: Aggregated total CO2e in kilograms.
        total_fossil_co2e_kg: Aggregated total fossil CO2e in kg.
        total_biogenic_co2_kg: Aggregated total biogenic CO2 in kg.
        breakdown: Dictionary of group key to subtotal CO2e in kg.
            Keys depend on the aggregation_type (e.g. facility_id,
            fuel_type, energy_type, supplier_id, or period string).
        count: Number of individual calculation results included
            in the aggregation.
        provenance_hash: SHA-256 hash of the complete aggregation
            for audit trail.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    aggregation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique aggregation identifier (UUID)",
    )
    aggregation_type: AggregationType = Field(
        ...,
        description="Dimension used for grouping",
    )
    total_co2e_kg: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Aggregated total CO2e in kg",
    )
    total_fossil_co2e_kg: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Aggregated total fossil CO2e in kg",
    )
    total_biogenic_co2_kg: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Aggregated total biogenic CO2 in kg",
    )
    breakdown: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Group key to subtotal CO2e mapping",
    )
    count: int = Field(
        ...,
        ge=0,
        description="Number of calculation results aggregated",
    )
    provenance_hash: str = Field(
        default="",
        max_length=64,
        description="SHA-256 provenance hash for audit trail",
    )
