# -*- coding: utf-8 -*-
"""
Scope 2 Cooling Purchase Agent Data Models - AGENT-MRV-012

Pydantic v2 data models for the Scope 2 Cooling Purchase Agent SDK
covering GHG Protocol Scope 2 purchased cooling emission calculations
including:
- 18 cooling technologies with COP ranges and IPLV profiles
- 12 district cooling regional emission factors
- 11 heat source emission factors for absorption chillers
- 11 refrigerant GWP values (AR5 and AR6)
- 6 efficiency metric conversions (COP/EER/kW_per_ton/IPLV/NPLV/SEER)
- 7 cooling unit conversions (ton_hour/kWh_th/GJ/BTU/MMBTU/MJ/TR)
- AHRI 550/590 part-load weighting (IPLV/NPLV calculation)
- 4 emission gas species (CO2, CH4, N2O, CO2e)
- GWP values from IPCC AR4, AR5, AR6, and AR6 20-year horizon
- Electric chiller calculations (full-load and IPLV part-load)
- Absorption chiller calculations with parasitic electricity
- Free cooling from 4 natural sources (seawater/lake/river/air)
- Thermal energy storage with temporal emission shifting
- District cooling network losses and pump energy
- Refrigerant leakage tracking (cross-reference with MRV-002)
- Batch calculation requests across multiple facilities
- Monte Carlo and analytical uncertainty quantification
- Multi-framework regulatory compliance checking
- Aggregation by facility, technology, region, supplier, or period
- SHA-256 provenance chain for complete audit trails

Enumerations (18):
    - CoolingTechnology, CompressorType, CondenserType, AbsorptionType,
      FreeCoolingSource, TESType, HeatSource, EfficiencyMetric,
      CoolingUnit, EmissionGas, GWPSource, ComplianceStatus,
      DataQualityTier, FacilityType, ReportingPeriod,
      AggregationType, BatchStatus, Refrigerant

Constants (all Decimal for zero-hallucination deterministic arithmetic):
    - COOLING_TECHNOLOGY_SPECS: 18 cooling technologies (COP ranges, IPLV)
    - DISTRICT_COOLING_FACTORS: 12 regional cooling emission factors
    - HEAT_SOURCE_FACTORS: 11 heat source emission factors
    - REFRIGERANT_GWP: 11 refrigerant GWP values (AR5 + AR6)
    - GWP_VALUES: IPCC AR4/AR5/AR6/AR6_20YR GWP (CO2, CH4, N2O)
    - UNIT_CONVERSIONS: Cooling energy unit conversion factors
    - AHRI_PART_LOAD_WEIGHTS: Part-load weighting per AHRI 550/590
    - EFFICIENCY_CONVERSIONS: COP/EER/kW_per_ton/SEER conversions

Data Models (23):
    - CoolingTechnologySpec, DistrictCoolingFactor, HeatSourceFactor,
      RefrigerantData, PartLoadPoint, FacilityInfo, CoolingSupplier,
      ElectricChillerRequest, AbsorptionCoolingRequest,
      FreeCoolingRequest, TESRequest, DistrictCoolingRequest,
      GasEmissionDetail, CalculationResult, TESCalculationResult,
      RefrigerantLeakageResult, BatchCalculationRequest,
      BatchCalculationResult, UncertaintyRequest, UncertaintyResult,
      ComplianceCheckResult, AggregationRequest, AggregationResult

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-012 Cooling Purchase Agent (GL-MRV-X-023)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

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
TABLE_PREFIX: str = "gl_cp_"

# =============================================================================
# Enumerations (18)
# =============================================================================

class CoolingTechnology(str, Enum):
    """Classification of cooling generation technologies for Scope 2.

    GHG Protocol Scope 2 Guidance requires organisations to report
    indirect emissions from purchased cooling services. This
    enumeration covers the eighteen categories of cooling generation
    technology relevant to the Cooling Purchase Agent. Each technology
    has a characteristic coefficient of performance (COP) range and
    primary energy source that determine the emission intensity.

    WATER_COOLED_CENTRIFUGAL: Water-cooled centrifugal compressor
        chiller. Highest COP among electric chillers (5.0-7.5).
        Standard choice for large district cooling and campus plants.
        Uses cooling towers or seawater for heat rejection.
        IPLV typically 9.0. Energy source: electricity.
    AIR_COOLED_CENTRIFUGAL: Air-cooled centrifugal compressor chiller.
        Moderate COP (2.8-3.8) due to higher condensing temperatures.
        Used where cooling tower water is unavailable or restricted.
        IPLV typically 4.5. Energy source: electricity.
    WATER_COOLED_SCREW: Water-cooled rotary screw compressor chiller.
        Good COP (4.0-5.5). Common in medium-sized district cooling
        and industrial applications. Reliable at part loads.
        IPLV typically 6.5. Energy source: electricity.
    AIR_COOLED_SCREW: Air-cooled rotary screw compressor chiller.
        Moderate COP (2.5-3.5). Popular for commercial buildings
        without access to cooling water. IPLV typically 4.0.
        Energy source: electricity.
    WATER_COOLED_RECIPROCATING: Water-cooled reciprocating compressor
        chiller. Moderate COP (3.5-5.0). Good for variable loads and
        smaller installations. IPLV typically 5.5.
        Energy source: electricity.
    AIR_COOLED_SCROLL: Air-cooled scroll compressor chiller. Lower
        COP (2.5-3.5) but compact design suitable for smaller
        commercial installations. IPLV typically 3.8.
        Energy source: electricity.
    SINGLE_EFFECT_LIBR: Single-effect lithium bromide absorption
        chiller driven by low-grade heat (hot water or low-pressure
        steam). Low COP (0.6-0.8) but uses waste heat or solar
        thermal. IPLV typically 0.72. Energy source: steam/hot water.
    DOUBLE_EFFECT_LIBR: Double-effect lithium bromide absorption
        chiller driven by higher-grade heat (high-pressure steam or
        direct-fired). Moderate COP (1.0-1.4). More efficient use
        of heat input. IPLV typically 1.30.
        Energy source: steam/direct fire.
    TRIPLE_EFFECT_LIBR: Triple-effect lithium bromide absorption
        chiller. Highest COP among absorption types (1.4-1.8).
        Requires high-temperature heat input. IPLV typically 1.70.
        Energy source: direct fire.
    AMMONIA_ABSORPTION: Ammonia absorption chiller. Uses ammonia-water
        working pair instead of LiBr. Suitable for industrial
        applications and lower evaporating temperatures. COP
        (0.5-0.7). IPLV typically 0.58. Energy source: steam/waste
        heat.
    SEAWATER_FREE: Seawater free cooling using deep ocean or coastal
        water as a heat sink. Very high effective COP (15.0-30.0)
        with only pump energy required. Limited by coastal location.
        Energy source: electricity (pumps).
    LAKE_FREE: Lake water free cooling using deep stratified lakes.
        High effective COP (12.0-25.0). Seasonal limitation in
        warmer climates. Used in Scandinavian and alpine cities.
        Energy source: electricity (pumps).
    RIVER_FREE: River water free cooling using flowing river or canal
        water. Moderate-high COP (10.0-20.0). Subject to seasonal
        temperature and environmental flow restrictions.
        Energy source: electricity (pumps).
    AMBIENT_AIR_FREE: Ambient air free cooling using dry coolers or
        cooling towers when outdoor temperature drops below supply
        setpoint. COP (8.0-15.0). Limited to cooler climates and
        seasons. Energy source: electricity (fans).
    ICE_TES: Ice thermal energy storage. Charges by producing ice
        during off-peak hours using low-cost electricity and discharges
        during peak demand. Lower COP during charging (2.8-4.0) due
        to lower evaporating temperatures. Enables temporal emission
        shifting. Energy source: electricity.
    CHILLED_WATER_TES: Chilled water thermal energy storage using
        stratified water tanks. Higher COP during charging (4.5-6.5)
        than ice TES. Good round-trip efficiency (90-95%).
        Enables load shifting. Energy source: electricity.
    PCM_TES: Phase-change material thermal energy storage. Uses
        materials that melt and freeze at target temperatures.
        Moderate COP (3.5-5.5). Compact storage. Round-trip
        efficiency 80-90%. Energy source: electricity.
    DISTRICT_COOLING: District cooling network with mixed technology
        plant. Overall system COP varies by region and plant mix.
        Default COP 4.0. Includes distribution network losses and
        pump energy. Energy source: mixed (electricity + heat).
    """

    WATER_COOLED_CENTRIFUGAL = "water_cooled_centrifugal"
    AIR_COOLED_CENTRIFUGAL = "air_cooled_centrifugal"
    WATER_COOLED_SCREW = "water_cooled_screw"
    AIR_COOLED_SCREW = "air_cooled_screw"
    WATER_COOLED_RECIPROCATING = "water_cooled_reciprocating"
    AIR_COOLED_SCROLL = "air_cooled_scroll"
    SINGLE_EFFECT_LIBR = "single_effect_libr"
    DOUBLE_EFFECT_LIBR = "double_effect_libr"
    TRIPLE_EFFECT_LIBR = "triple_effect_libr"
    AMMONIA_ABSORPTION = "ammonia_absorption"
    SEAWATER_FREE = "seawater_free"
    LAKE_FREE = "lake_free"
    RIVER_FREE = "river_free"
    AMBIENT_AIR_FREE = "ambient_air_free"
    ICE_TES = "ice_tes"
    CHILLED_WATER_TES = "chilled_water_tes"
    PCM_TES = "pcm_tes"
    DISTRICT_COOLING = "district_cooling"

class CompressorType(str, Enum):
    """Classification of electric chiller compressor mechanisms.

    Identifies the mechanical compressor type used in vapour-compression
    cycle chillers. Each compressor type has characteristic efficiency
    profiles, capacity ranges, and part-load performance curves that
    affect the overall COP and emission intensity of the cooling system.

    CENTRIFUGAL: Centrifugal (turbo) compressor. Uses rotating impeller
        to accelerate refrigerant vapour and convert kinetic energy to
        pressure. Best suited for large-capacity applications (300-10,000
        tons). Highest full-load efficiency among compressor types.
        Efficiency drops significantly below 30% load due to surge
        limit. Variable-speed drives improve part-load performance.
    SCREW: Rotary screw (helical rotor) compressor. Uses two
        intermeshing helical rotors to compress refrigerant. Medium
        capacity range (50-1,500 tons). Good part-load performance
        with slide-valve capacity control. Compact design with fewer
        moving parts than reciprocating. Common in water-cooled and
        air-cooled configurations.
    RECIPROCATING: Reciprocating (piston) compressor. Uses pistons
        driven by a crankshaft to compress refrigerant. Smaller
        capacity range (5-200 tons). Good efficiency at full load
        with cylinder unloading for part-load capacity reduction.
        Higher maintenance than rotary types but robust for harsh
        operating conditions.
    SCROLL: Scroll (orbital) compressor. Uses two interleaving scroll
        elements to compress refrigerant. Smallest capacity range
        (1-60 tons). Quiet operation with minimal vibration. Limited
        capacity modulation without variable-speed drives. Common in
        air-cooled packaged units for commercial buildings.
    """

    CENTRIFUGAL = "centrifugal"
    SCREW = "screw"
    RECIPROCATING = "reciprocating"
    SCROLL = "scroll"

class CondenserType(str, Enum):
    """Classification of chiller condenser heat rejection methods.

    The condenser type determines the condensing temperature and
    therefore significantly affects the chiller COP. Water-cooled
    condensers achieve lower condensing temperatures (typically
    29-35 degrees C) than air-cooled condensers (typically 40-55
    degrees C), resulting in higher COP values.

    WATER_COOLED: Water-cooled condenser. Rejects heat to a cooling
        tower, seawater, river, or lake water loop. Lower condensing
        temperature yields higher COP (typically 20-40% better than
        air-cooled). Requires cooling water infrastructure. Dominant
        choice for large district cooling plants and data centres.
    AIR_COOLED: Air-cooled condenser. Rejects heat directly to ambient
        air using finned-tube heat exchangers and fans. Higher
        condensing temperature reduces COP, especially in hot climates.
        Simpler infrastructure with no cooling tower water treatment.
        Common for smaller commercial installations and where water
        conservation is required.
    """

    WATER_COOLED = "water_cooled"
    AIR_COOLED = "air_cooled"

class AbsorptionType(str, Enum):
    """Classification of absorption chiller cycle configurations.

    Absorption chillers use heat energy rather than mechanical
    compression to drive the refrigeration cycle. The number of
    effects (stages) determines the COP and the required heat
    input temperature. Higher-effect machines achieve better COP
    but require higher-temperature heat sources.

    SINGLE_EFFECT: Single-effect absorption cycle using lithium
        bromide-water (LiBr-H2O) working pair. Driven by low-grade
        heat (hot water at 80-100 degrees C or low-pressure steam
        at 0.5-1.0 bar). COP range 0.6-0.8. Suitable for waste
        heat and solar thermal applications where heat temperature
        is limited. Most common absorption chiller configuration
        globally.
    DOUBLE_EFFECT: Double-effect absorption cycle using LiBr-H2O.
        Driven by medium-grade heat (high-pressure steam at 5-8 bar
        or direct-fired gas burner). COP range 1.0-1.4. Higher
        efficiency than single-effect but requires higher temperature
        heat input. Common in district cooling plants with dedicated
        boilers or CHP steam supply.
    TRIPLE_EFFECT: Triple-effect absorption cycle using LiBr-H2O.
        Driven by high-grade heat (direct-fired at very high
        temperatures). COP range 1.4-1.8. Highest absorption chiller
        efficiency but most complex and expensive. Limited commercial
        deployments. Emerging technology for high-efficiency district
        cooling.
    AMMONIA: Ammonia-water (NH3-H2O) absorption cycle. Uses ammonia
        as the refrigerant and water as the absorbent (reverse of
        LiBr systems). COP range 0.5-0.7. Can achieve sub-zero
        evaporating temperatures unlike LiBr. Used in industrial
        refrigeration and process cooling. Compatible with waste
        heat and solar thermal sources.
    """

    SINGLE_EFFECT = "single_effect"
    DOUBLE_EFFECT = "double_effect"
    TRIPLE_EFFECT = "triple_effect"
    AMMONIA = "ammonia"

class FreeCoolingSource(str, Enum):
    """Classification of natural heat sinks for free cooling systems.

    Free cooling exploits naturally available cold sources (water
    bodies or cool ambient air) to provide cooling with minimal
    energy input. Only pump or fan electricity is consumed, yielding
    very high effective COP values compared to mechanical chillers.

    SEAWATER: Seawater free cooling. Uses deep ocean or coastal water
        as a heat sink. Seawater at depth is typically 4-10 degrees C
        year-round. Highest and most stable COP (15.0-30.0) among
        free cooling sources. Limited to coastal locations. Requires
        corrosion-resistant heat exchangers. Examples: Stockholm,
        Helsinki, Barcelona, Singapore (deep water).
    LAKE: Deep lake water free cooling. Uses cold water from
        stratified lakes (hypolimnion typically 4-6 degrees C).
        High COP (12.0-25.0) with seasonal stability in deep
        lakes. Examples: Toronto (Lake Ontario), Geneva (Lake
        Geneva), Zurich. Environmental impact assessment required.
    RIVER: River water free cooling. Uses flowing river or canal
        water as a heat sink. COP (10.0-20.0) varies significantly
        with season and climate. Subject to environmental flow
        restrictions and temperature discharge limits. Higher
        sediment and biological fouling risk. Examples: Paris
        (Seine), Amsterdam (canals).
    AMBIENT_AIR: Ambient air free cooling using dry coolers or
        economiser modes on cooling towers. Effective COP (8.0-15.0)
        when outdoor wet-bulb temperature is sufficiently below the
        chilled water supply temperature. Strongest in Nordic, alpine,
        and continental climates during winter months. No water
        consumption. Zero cooling tower chemical treatment.
    """

    SEAWATER = "seawater"
    LAKE = "lake"
    RIVER = "river"
    AMBIENT_AIR = "ambient_air"

class TESType(str, Enum):
    """Classification of thermal energy storage technologies.

    Thermal energy storage (TES) enables temporal shifting of cooling
    production from peak to off-peak hours, taking advantage of
    lower electricity rates and potentially lower grid carbon
    intensity during off-peak periods. This creates both cost
    savings and emission reduction opportunities.

    ICE: Ice thermal energy storage. Produces ice during off-peak
        hours by operating chillers at evaporating temperatures below
        0 degrees C (typically -5 to -7 degrees C). Ice is melted
        during peak demand to provide cooling. Storage density:
        approximately 334 kJ/L. COP penalty of 20-35% compared to
        chilled water production due to lower evaporating temperature.
        Round-trip efficiency typically 80-90% including thermal
        losses. Common in commercial buildings and district cooling
        for peak load management.
    CHILLED_WATER: Chilled water thermal energy storage. Uses
        stratified water tanks (typically 4-7 degrees C supply,
        12-15 degrees C return) to store sensible cooling. Higher
        COP during charging than ice (no sub-zero temperature
        penalty). Storage density: approximately 21-42 kJ/L
        depending on temperature differential. Round-trip efficiency
        90-95%. Requires larger storage volume than ice. Common in
        large campus and district cooling systems.
    PCM: Phase-change material thermal energy storage. Uses
        encapsulated materials that melt and freeze at a target
        temperature (typically 5-15 degrees C depending on material).
        Storage density between ice and chilled water. COP during
        charging depends on PCM melting point versus evaporating
        temperature. Round-trip efficiency 80-90%. Compact storage
        footprint. Emerging technology with growing adoption in
        commercial and district cooling.
    """

    ICE = "ice"
    CHILLED_WATER = "chilled_water"
    PCM = "pcm"

class HeatSource(str, Enum):
    """Classification of heat sources for absorption chiller systems.

    Identifies the source of thermal energy driving the absorption
    chiller cycle. The emission factor depends on the heat source
    type: fossil fuel combustion produces direct emissions, waste
    heat and renewables have zero direct emissions, and CHP-sourced
    heat requires allocated emissions from the cogeneration plant.

    NATURAL_GAS_STEAM: Steam from a natural gas-fired boiler. CO2
        intensity depends on boiler efficiency. Default EF: 70.1
        kgCO2e/GJ (56.1 kgCO2/GJ fuel / 0.80 boiler efficiency).
        Most common heat source for absorption chillers in commercial
        and institutional applications.
    DISTRICT_HEATING: Heat from a district heating network. Emission
        factor is region-specific, depending on the network fuel mix
        and plant efficiency. Default global EF: 70.0 kgCO2e/GJ.
        Common in cities with existing district heating infrastructure
        that extends to summer cooling via absorption.
    WASTE_HEAT: Recovered waste heat from industrial processes (e.g.
        exhaust gas, process steam, condenser heat). Zero direct
        emissions as the heat is a byproduct that would otherwise be
        rejected to the environment. Ideal driver for single-effect
        absorption chillers. EF: 0.0 kgCO2e/GJ.
    CHP_EXHAUST: Exhaust heat from a combined heat and power (CHP)
        or cogeneration plant. Emissions must be allocated between
        electrical and thermal outputs per GHG Protocol guidance.
        EF depends on CHP allocation method and fuel type. Cross-
        reference with AGENT-MRV-011 for allocation calculation.
    SOLAR_THERMAL: Heat from solar thermal collectors (flat plate,
        evacuated tube, or parabolic trough). Zero operational
        emissions. EF: 0.0 kgCO2e/GJ. Intermittent availability
        requires storage or backup heat source. Best suited for
        single-effect absorption in high-solar-resource regions.
    GEOTHERMAL: Heat from geothermal reservoirs (deep wells or
        ground-source heat exchangers). Zero direct combustion
        emissions. Minor fugitive emissions from dissolved gases
        in some geothermal fields. EF: 0.0 kgCO2e/GJ for
        closed-loop systems.
    BIOGAS_STEAM: Steam from biogas-fired boiler (anaerobic digestion
        gas, landfill gas, or sewage gas). Biogenic CO2 reported
        separately per GHG Protocol. Fossil CO2e: 0.0 (biogenic
        source). Minor CH4 and N2O from combustion still counted.
    FUEL_OIL_STEAM: Steam from fuel oil-fired boiler (No. 2 or
        No. 6 fuel oil). Higher emission intensity than natural gas.
        Default EF: 96.8 kgCO2e/GJ (77.4 / 0.80 boiler efficiency).
        Used as backup fuel in some absorption chiller installations.
    COAL_STEAM: Steam from coal-fired boiler (bituminous or
        subbituminous). Highest emission intensity among fossil
        fuels. Default EF: 126.1 kgCO2e/GJ (94.6 / 0.75 boiler
        efficiency). Common in some Asian district cooling plants.
    ELECTRIC_BOILER: Steam or hot water from an electric boiler.
        Zero direct combustion emissions at point of use. Grid-
        dependent upstream emissions. EF = Grid_EF / 0.98 (boiler
        efficiency). Used in regions with low-carbon grids.
    HEAT_PUMP: Heat from a heat pump system used to drive absorption
        cooling. Grid-dependent upstream emissions. EF = Grid_EF /
        COP_HP. Emerging configuration for high-efficiency hybrid
        cooling systems.
    """

    NATURAL_GAS_STEAM = "natural_gas_steam"
    DISTRICT_HEATING = "district_heating"
    WASTE_HEAT = "waste_heat"
    CHP_EXHAUST = "chp_exhaust"
    SOLAR_THERMAL = "solar_thermal"
    GEOTHERMAL = "geothermal"
    BIOGAS_STEAM = "biogas_steam"
    FUEL_OIL_STEAM = "fuel_oil_steam"
    COAL_STEAM = "coal_steam"
    ELECTRIC_BOILER = "electric_boiler"
    HEAT_PUMP = "heat_pump"

class EfficiencyMetric(str, Enum):
    """Efficiency metrics used for cooling equipment performance rating.

    Different standards and regions use different metrics to express
    the energy efficiency of cooling equipment. This enumeration
    covers the six most common metrics. Conversion functions allow
    standardisation to COP for emission calculations.

    COP: Coefficient of Performance. Dimensionless ratio of cooling
        output (kW_th) to energy input (kW). The fundamental
        thermodynamic efficiency metric. COP = Q_cool / W_input.
        SI-based metric used internationally. Typical electric chiller
        COP ranges from 2.5 (small air-cooled scroll) to 7.5 (large
        water-cooled centrifugal).
    EER: Energy Efficiency Ratio. Ratio of cooling output in BTU/h
        to electrical input in watts. US customary metric. EER =
        COP x 3.412 (BTU/Wh conversion). Used in AHRI standards
        and US equipment ratings.
    KW_PER_TON: Kilowatts of electrical input per refrigeration ton
        (12,000 BTU/h = 3.517 kW) of cooling output. Inverse
        efficiency metric: lower is better. kW/ton = 3.517 / COP.
        Common in US commercial HVAC and ASHRAE standards.
    IPLV: Integrated Part-Load Value. Weighted average efficiency
        across four operating conditions per AHRI 550/590. Weights:
        1% at 100%, 42% at 75%, 45% at 50%, 12% at 25% load.
        Reflects typical building load profile. Most meaningful
        single metric for annual chiller performance.
    NPLV: Non-standard Part-Load Value. Same calculation as IPLV
        but with user-defined condenser water and load conditions
        that differ from AHRI standard conditions. Used when actual
        site conditions differ significantly from AHRI test
        conditions (e.g. variable-flow condenser water, non-standard
        entering condenser water temperature).
    SEER: Seasonal Energy Efficiency Ratio. Weighted average EER
        across a range of outdoor temperatures representing a typical
        cooling season. Used for air-cooled equipment ratings in
        residential and light commercial applications. SEER / 3.412
        approximates annual average COP.
    """

    COP = "cop"
    EER = "eer"
    KW_PER_TON = "kw_per_ton"
    IPLV = "iplv"
    NPLV = "nplv"
    SEER = "seer"

class CoolingUnit(str, Enum):
    """Units of measurement for cooling energy quantities.

    All cooling consumption data must be expressed in a known unit
    for conversion to kilowatt-hours thermal (kWh_th) before
    applying emission factor calculations. kWh_th is the standard
    basis for cooling emission calculations in this module.

    TON_HOUR: Ton-hour of refrigeration. 1 ton-hour = 3.517 kWh_th.
        Common in US commercial HVAC for metering cooling consumption.
        One refrigeration ton = 12,000 BTU/h = 3.517 kW cooling.
    KWH_TH: Kilowatt-hour thermal. SI unit for cooling energy.
        Standard basis for emission factor calculations in this
        module. 1 kWh_th = 3,600 kJ of cooling energy.
    GJ: Gigajoules. SI unit for large quantities of thermal energy.
        1 GJ = 277.778 kWh_th. Common in district cooling metering
        for large networks and European reporting.
    BTU: British Thermal Unit. 1 BTU = 0.000293071 kWh_th. Used
        in US and UK for small-scale cooling energy metering.
    MMBTU: Million British Thermal Units. 1 MMBTU = 293.071 kWh_th.
        Used in US district cooling for large-scale metering.
    MJ: Megajoules. 1 MJ = 0.2778 kWh_th. Used in some Asian and
        European metering systems for medium quantities.
    TR: Refrigeration ton (rate). 1 TR = 3.517 kW of cooling capacity.
        Used for expressing cooling capacity and, when multiplied by
        operating hours, cooling energy consumption.
    """

    TON_HOUR = "ton_hour"
    KWH_TH = "kwh_th"
    GJ = "gj"
    BTU = "btu"
    MMBTU = "mmbtu"
    MJ = "mj"
    TR = "tr"

class EmissionGas(str, Enum):
    """Greenhouse gases tracked in Scope 2 cooling calculations.

    Identifies the individual greenhouse gas species for per-gas
    emission breakdowns in cooling purchase calculations. Grid
    electricity consumption for electric chillers and parasitic
    loads produces upstream CO2, CH4, and N2O emissions based on
    the grid fuel mix.

    CO2: Carbon dioxide. Primary greenhouse gas from fossil fuel
        combustion in power plants supplying grid electricity to
        cooling systems. Largest contributor to total CO2e for
        electric chiller operations.
    CH4: Methane. Emitted from incomplete combustion and fugitive
        emissions in the upstream electricity supply chain. Small
        fraction of total emissions but included for complete GHG
        Protocol compliance. Higher contribution from grids with
        significant natural gas generation.
    N2O: Nitrous oxide. Emitted from combustion processes in power
        plants, particularly coal-fired stations. Included for
        complete GHG Protocol compliance. Small but non-negligible
        contribution to total CO2e due to high GWP.
    CO2E: Carbon dioxide equivalent. Aggregated total of all
        greenhouse gases weighted by their Global Warming Potential
        (GWP) values. Used for reporting totals. Calculated as
        CO2 + (CH4 x GWP_CH4) + (N2O x GWP_N2O).
    """

    CO2 = "CO2"
    CH4 = "CH4"
    N2O = "N2O"
    CO2E = "CO2e"

class GWPSource(str, Enum):
    """IPCC Assessment Report edition used for Global Warming Potential values.

    Determines which set of 100-year (or 20-year) GWP multipliers to
    apply when converting individual gas emissions (CH4, N2O) to CO2
    equivalent totals. The choice of GWP source affects the total
    CO2e result and must be consistent within a reporting period.

    AR4: Fourth Assessment Report (2007). 100-year GWP.
        CO2=1, CH4=25, N2O=298.
        Still required by some regulatory frameworks and corporate
        reporting programmes.
    AR5: Fifth Assessment Report (2014). 100-year GWP.
        CO2=1, CH4=28, N2O=265.
        Current default for CDP and many corporate GHG inventories.
    AR6: Sixth Assessment Report (2021). 100-year GWP.
        CO2=1, CH4=27.9, N2O=273.
        Latest IPCC values. Recommended for new reporting periods.
    AR6_20YR: Sixth Assessment Report (2021). 20-year GWP.
        CO2=1, CH4=81.2, N2O=273.
        Highlights near-term climate impact of short-lived climate
        pollutants, particularly methane. Used for supplementary
        disclosure and climate urgency analysis.
    """

    AR4 = "AR4"
    AR5 = "AR5"
    AR6 = "AR6"
    AR6_20YR = "AR6_20YR"

class ComplianceStatus(str, Enum):
    """Result of a regulatory compliance check for a calculation.

    Evaluates whether a completed cooling emission calculation
    satisfies all requirements of a specific regulatory framework.

    COMPLIANT: All requirements of the regulatory framework are
        fully satisfied for the given calculation. Data quality,
        methodology, and documentation meet the framework standard.
    NON_COMPLIANT: One or more mandatory requirements are not met.
        The calculation cannot be used for reporting under this
        framework without remediation.
    PARTIAL: Some requirements are met but others are missing,
        incomplete, or require additional evidence. May be acceptable
        for interim reporting with documented gaps.
    NOT_APPLICABLE: The regulatory framework does not apply to this
        type of cooling calculation, facility type, or jurisdiction.
    """

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL = "partial"
    NOT_APPLICABLE = "not_applicable"

class DataQualityTier(str, Enum):
    """Data quality classification for cooling emission factor inputs.

    Tier classification follows the IPCC approach to data quality
    and determines the uncertainty range applied to calculation
    results. Higher tiers indicate more precise, facility-specific
    data with lower uncertainty.

    TIER_1: Default values from international databases (IPCC, IEA,
        ASHRAE). Highest uncertainty range (+/- 30-50%). Used when
        no supplier or equipment-specific data is available. Applies
        technology category default COP values and regional default
        grid emission factors.
    TIER_2: Equipment-specific data such as manufacturer nameplate
        COP, AHRI-certified IPLV, or supplier-provided emission
        factors. Moderate uncertainty (+/- 15-25%). Requires
        documentation from the cooling equipment manufacturer or
        district cooling service supplier.
    TIER_3: Facility-specific measured data including metered chiller
        performance (actual COP), continuous power monitoring, and
        supplier-verified emission factors. Lowest uncertainty
        (+/- 5-15%). Highest data quality suitable for assured
        reporting.
    """

    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"

class FacilityType(str, Enum):
    """Classification of reporting facilities by primary function.

    Determines default cooling intensity benchmarks, applicable
    reporting templates, and comparison groups for portfolio analysis.
    Cooling demand profiles vary significantly by facility type.

    COMMERCIAL: Commercial office building, retail space, or mixed-
        use commercial property. Moderate cooling demand primarily
        driven by internal heat gains from occupants, lighting, and
        IT equipment. Typical cooling intensity: 40-80 kWh_th/m2/yr.
    DATA_CENTER: IT data centre or colocation facility. Very high
        cooling demand driven by IT equipment heat loads. Cooling
        is typically 30-50% of total facility energy. Typical
        cooling intensity: 200-500 kWh_th/m2/yr. Often served by
        district cooling or dedicated chiller plants.
    HOSPITAL: Healthcare facility including hospitals, medical
        centres, and research laboratories. High cooling demand for
        clinical areas, operating theatres, and laboratory spaces.
        24/7 operation with strict temperature and humidity control.
        Typical cooling intensity: 100-200 kWh_th/m2/yr.
    CAMPUS: University, corporate, or institutional campus with
        centralised cooling distribution. Multiple buildings served
        by a central chiller plant via chilled water loop. Load
        diversity across buildings improves plant efficiency.
    INDUSTRIAL: Industrial manufacturing or production facility.
        Process cooling demand varies widely by industry sector.
        May include both comfort cooling and process cooling loads.
        Often connected to industrial district cooling networks.
    DISTRICT: District cooling network serving multiple customers
        across a geographic area. Includes the central plant,
        distribution piping, and energy transfer stations. Managed
        by a district cooling utility or operator.
    """

    COMMERCIAL = "commercial"
    DATA_CENTER = "data_center"
    HOSPITAL = "hospital"
    CAMPUS = "campus"
    INDUSTRIAL = "industrial"
    DISTRICT = "district"

class AggregationType(str, Enum):
    """Dimension for aggregating cooling calculation results.

    Determines how individual cooling calculation results are grouped
    when producing summary reports and portfolio-level totals.

    BY_FACILITY: Aggregate by facility. Produces per-facility
        emission totals across all cooling technologies and periods.
    BY_TECHNOLOGY: Aggregate by cooling technology. Produces per-
        technology emission totals across all facilities. Useful for
        technology comparison and optimisation analysis.
    BY_REGION: Aggregate by geographic region. Produces per-region
        emission totals for district cooling networks and portfolio
        geographic analysis.
    BY_SUPPLIER: Aggregate by cooling supplier. Produces per-supplier
        emission totals for procurement analysis and supplier
        engagement.
    BY_PERIOD: Aggregate by time period (month, quarter, year).
        Produces temporal trends for performance tracking and
        seasonal pattern analysis.
    """

    BY_FACILITY = "by_facility"
    BY_TECHNOLOGY = "by_technology"
    BY_REGION = "by_region"
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

class Refrigerant(str, Enum):
    """Common refrigerant types used in chiller systems.

    Identifies the refrigerant working fluid in vapour-compression
    chillers. The GWP of the refrigerant determines the climate
    impact of any leakage, which is tracked as informational Scope 1
    emissions and cross-referenced with AGENT-MRV-002 (Refrigerants
    and F-Gas Agent). The EU F-Gas Regulation and Kigali Amendment
    to the Montreal Protocol are driving a transition from high-GWP
    HFC refrigerants to low-GWP alternatives.

    R_134A: HFC-134a (1,1,1,2-Tetrafluoroethane). Standard refrigerant
        for centrifugal chillers. GWP(AR5)=1430, GWP(AR6)=1530.
        Subject to Kigali phase-down from 2029. Being replaced by
        R-1234ze(E) and R-513A in new equipment.
    R_410A: HFC-410A (R-32/R-125 azeotropic blend). Dominant
        refrigerant for scroll and screw chillers. GWP(AR5)=2088,
        GWP(AR6)=2088. Subject to Kigali phase-down from 2024.
        Being replaced by R-32 and R-454B.
    R_407C: HFC-407C (R-32/R-125/R-134a zeotropic blend). Used in
        reciprocating and scroll chillers as R-22 replacement.
        GWP(AR5)=1774, GWP(AR6)=1774. Subject to Kigali phase-down.
    R_32: HFC-32 (Difluoromethane). Lower-GWP alternative to R-410A.
        GWP(AR5)=675, GWP(AR6)=771. Transitional refrigerant.
        Mildly flammable (A2L classification). Increasingly adopted
        in new split systems and smaller chillers.
    R_1234ZE_E: HFO-1234ze(E). Ultra-low GWP next-generation
        refrigerant for centrifugal chillers. GWP(AR5)=7,
        GWP(AR6)=7. Non-flammable in some configurations. Direct
        replacement path for R-134a in new centrifugal designs.
    R_1234YF: HFO-1234yf. Ultra-low GWP refrigerant. GWP(AR5)=4,
        GWP(AR6)<1. Primarily used in automotive air conditioning
        and small commercial systems. Mildly flammable (A2L).
    R_513A: HFO blend (R-1234yf/R-134a). Drop-in replacement for
        R-134a in centrifugal chillers. GWP(AR5)=631, GWP(AR6)=631.
        Transitional solution for existing R-134a equipment.
        Non-flammable (A1 classification).
    R_514A: HFO blend (R-1336mzz(Z)/R-1130(E)). Ultra-low GWP
        alternative for centrifugal chillers. GWP(AR5)=2,
        GWP(AR6)=2. Non-flammable. Suitable for low-pressure
        centrifugal designs.
    R_290: HC-290 (Propane). Natural refrigerant with very low GWP.
        GWP(AR5)=3, GWP(AR6)=0.02. Highly flammable (A3). Used
        in small commercial and self-contained systems. Charge
        limits apply per safety standards.
    R_717: R-717 (Ammonia/NH3). Natural refrigerant with zero GWP.
        GWP(AR5)=0, GWP(AR6)=0. Used in large industrial
        refrigeration and ammonia absorption chillers. Toxic (B2L)
        but widely used in industrial settings with proper safety
        measures.
    R_718: R-718 (Water). Used as refrigerant in lithium bromide
        absorption chillers (LiBr-H2O systems). GWP=0. No
        environmental impact from leakage. Not used in vapour-
        compression systems.
    """

    R_134A = "r_134a"
    R_410A = "r_410a"
    R_407C = "r_407c"
    R_32 = "r_32"
    R_1234ZE_E = "r_1234ze_e"
    R_1234YF = "r_1234yf"
    R_513A = "r_513a"
    R_514A = "r_514a"
    R_290 = "r_290"
    R_717 = "r_717"
    R_718 = "r_718"

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
# AHRI 550/590 Part-Load Weights
# ---------------------------------------------------------------------------

#: AHRI Standard 550/590 part-load weighting factors for IPLV calculation.
#: Reflects typical building cooling load distribution over an annual
#: operating cycle. Chillers operate at part load the majority of the
#: time: 42% at 75% load, 45% at 50% load, and only 1% at full load.
#:
#: Sources:
#:   AHRI Standard 550/590 (2023) — Performance Rating of Water-
#:     Chilling and Heat Pump Water-Heating Packages Using the Vapor
#:     Compression Cycle.
#:   ASHRAE Standard 90.1-2022 — Energy Standard for Buildings.
#:
#: Formula: IPLV = 0.01 * COP_100% + 0.42 * COP_75%
#:                + 0.45 * COP_50% + 0.12 * COP_25%
AHRI_PART_LOAD_WEIGHTS: Dict[str, Decimal] = {
    "100%": Decimal("0.01"),
    "75%": Decimal("0.42"),
    "50%": Decimal("0.45"),
    "25%": Decimal("0.12"),
}

# ---------------------------------------------------------------------------
# Efficiency Metric Conversions
# ---------------------------------------------------------------------------

#: Conversion factors between cooling efficiency metrics.
#: All values are exact Decimal representations for zero-hallucination
#: deterministic arithmetic. No floating-point rounding errors.
#:
#: COP_TO_EER: COP * 3.412 = EER (BTU/Wh conversion factor).
#: EER_TO_COP: EER / 3.412 = COP.
#: COP_TO_KW_PER_TON: 3.517 / COP = kW/ton (1 RT = 3.517 kW).
#: SEER_TO_COP: SEER / 3.412 approximates annual average COP.
#:
#: Sources:
#:   ASHRAE Handbook — Fundamentals (2021), Chapter 38.
#:   AHRI Standard 550/590 (2023).
EFFICIENCY_CONVERSIONS: Dict[str, Decimal] = {
    "cop_to_eer": Decimal("3.412"),
    "eer_to_cop": Decimal("0.293083"),
    "cop_to_kw_per_ton": Decimal("3.517"),
    "seer_to_cop": Decimal("0.293083"),
}

# ---------------------------------------------------------------------------
# Cooling Unit Conversion Factors
# ---------------------------------------------------------------------------

#: Conversion factors between cooling energy units.
#: All values are exact Decimal representations for zero-hallucination
#: deterministic arithmetic. No floating-point rounding errors.
#:
#: TON_HOUR_TO_KWH_TH: 1 ton-hour = 3.517 kWh_th (1 RT = 3.517 kW).
#: GJ_TO_KWH_TH: 1 GJ = 277.778 kWh_th.
#: MMBTU_TO_KWH_TH: 1 MMBTU = 293.071 kWh_th.
#: MJ_TO_KWH_TH: 1 MJ = 0.2778 kWh_th.
#: BTU_TO_KWH_TH: 1 BTU = 0.000293071 kWh_th.
#: TR_TO_KW: 1 refrigeration ton = 3.517 kW cooling capacity.
#:
#: Sources:
#:   ASHRAE Handbook — Fundamentals (2021), Chapter 38.
#:   ISO 80000-5:2019 — Quantities and units.
UNIT_CONVERSIONS: Dict[str, Decimal] = {
    "ton_hour_to_kwh_th": Decimal("3.517"),
    "gj_to_kwh_th": Decimal("277.778"),
    "mmbtu_to_kwh_th": Decimal("293.071"),
    "mj_to_kwh_th": Decimal("0.2778"),
    "btu_to_kwh_th": Decimal("0.000293071"),
    "tr_to_kw": Decimal("3.517"),
}

# =============================================================================
# Data Models (23)
# =============================================================================

# ---------------------------------------------------------------------------
# 1. CoolingTechnologySpec
# ---------------------------------------------------------------------------

class CoolingTechnologySpec(GreenLangBase):
    """Performance specification for a cooling generation technology.

    Encapsulates the COP range, default COP, IPLV (if applicable),
    energy source, and optional compressor and condenser type for a
    specific cooling technology. Used by the CoolingDatabaseEngine
    for technology lookup and default parameter resolution.

    Attributes:
        cop_min: Minimum COP under typical operating conditions for
            this technology. Represents worst-case performance at
            design conditions (high outdoor temperature, full load).
        cop_max: Maximum COP under optimal operating conditions.
            Represents best-case performance at favourable conditions
            (low outdoor temperature, moderate load).
        cop_default: Default COP value used when no measured or
            site-specific COP is available. Represents typical
            annual average performance.
        iplv: Integrated Part-Load Value per AHRI 550/590. Weighted
            average COP across four part-load conditions. None for
            technologies where IPLV is not applicable (free cooling,
            TES, district). Typically 30-50% higher than full-load
            COP for electric chillers.
        energy_source: Primary energy input type. "electricity" for
            electric chillers, free cooling, and TES. "heat" for
            absorption chillers. "mixed" for district cooling.
        compressor_type: Type of mechanical compressor (for electric
            chillers only). None for absorption, free cooling, TES,
            and district cooling technologies.
        condenser_type: Type of condenser heat rejection method (for
            electric chillers only). None for absorption, free cooling,
            TES, and district cooling technologies.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    cop_min: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Minimum COP under typical operating conditions",
    )
    cop_max: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Maximum COP under optimal operating conditions",
    )
    cop_default: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Default COP when no measured value available",
    )
    iplv: Optional[Decimal] = Field(
        default=None,
        gt=Decimal("0"),
        description="IPLV per AHRI 550/590 (None if not applicable)",
    )
    energy_source: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Primary energy input type (electricity, heat, mixed)",
    )
    compressor_type: Optional[CompressorType] = Field(
        default=None,
        description="Compressor type (electric chillers only)",
    )
    condenser_type: Optional[CondenserType] = Field(
        default=None,
        description="Condenser type (electric chillers only)",
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
# Cooling Technology Specifications Constant Table
# ---------------------------------------------------------------------------

#: Performance specifications for all 18 cooling technologies.
#: COP ranges and IPLV values from AHRI, ASHRAE, and manufacturer data.
#:
#: Sources:
#:   ASHRAE Handbook — HVAC Systems and Equipment (2024).
#:   AHRI Standard 550/590 (2023) — Water-chilling packages.
#:   AHRI Standard 560 (2023) — Absorption water-chilling packages.
#:   US DOE Federal Energy Management Program (FEMP) chiller benchmarks.
#:   European District Energy Association (Euroheat) cooling statistics.
#:   International District Energy Association (IDEA) design guides.
#:   Deep lake/seawater cooling case studies (Stockholm, Toronto, Geneva).
COOLING_TECHNOLOGY_SPECS: Dict[str, CoolingTechnologySpec] = {
    # --- Electric Chillers ---
    CoolingTechnology.WATER_COOLED_CENTRIFUGAL.value: CoolingTechnologySpec(
        cop_min=Decimal("5.0"),
        cop_max=Decimal("7.5"),
        cop_default=Decimal("6.1"),
        iplv=Decimal("9.0"),
        energy_source="electricity",
        compressor_type=CompressorType.CENTRIFUGAL,
        condenser_type=CondenserType.WATER_COOLED,
    ),
    CoolingTechnology.AIR_COOLED_CENTRIFUGAL.value: CoolingTechnologySpec(
        cop_min=Decimal("2.8"),
        cop_max=Decimal("3.8"),
        cop_default=Decimal("3.2"),
        iplv=Decimal("4.5"),
        energy_source="electricity",
        compressor_type=CompressorType.CENTRIFUGAL,
        condenser_type=CondenserType.AIR_COOLED,
    ),
    CoolingTechnology.WATER_COOLED_SCREW.value: CoolingTechnologySpec(
        cop_min=Decimal("4.0"),
        cop_max=Decimal("5.5"),
        cop_default=Decimal("4.7"),
        iplv=Decimal("6.5"),
        energy_source="electricity",
        compressor_type=CompressorType.SCREW,
        condenser_type=CondenserType.WATER_COOLED,
    ),
    CoolingTechnology.AIR_COOLED_SCREW.value: CoolingTechnologySpec(
        cop_min=Decimal("2.5"),
        cop_max=Decimal("3.5"),
        cop_default=Decimal("3.0"),
        iplv=Decimal("4.0"),
        energy_source="electricity",
        compressor_type=CompressorType.SCREW,
        condenser_type=CondenserType.AIR_COOLED,
    ),
    CoolingTechnology.WATER_COOLED_RECIPROCATING.value: CoolingTechnologySpec(
        cop_min=Decimal("3.5"),
        cop_max=Decimal("5.0"),
        cop_default=Decimal("4.2"),
        iplv=Decimal("5.5"),
        energy_source="electricity",
        compressor_type=CompressorType.RECIPROCATING,
        condenser_type=CondenserType.WATER_COOLED,
    ),
    CoolingTechnology.AIR_COOLED_SCROLL.value: CoolingTechnologySpec(
        cop_min=Decimal("2.5"),
        cop_max=Decimal("3.5"),
        cop_default=Decimal("2.8"),
        iplv=Decimal("3.8"),
        energy_source="electricity",
        compressor_type=CompressorType.SCROLL,
        condenser_type=CondenserType.AIR_COOLED,
    ),
    # --- Absorption Chillers ---
    CoolingTechnology.SINGLE_EFFECT_LIBR.value: CoolingTechnologySpec(
        cop_min=Decimal("0.6"),
        cop_max=Decimal("0.8"),
        cop_default=Decimal("0.70"),
        iplv=Decimal("0.72"),
        energy_source="heat",
        compressor_type=None,
        condenser_type=None,
    ),
    CoolingTechnology.DOUBLE_EFFECT_LIBR.value: CoolingTechnologySpec(
        cop_min=Decimal("1.0"),
        cop_max=Decimal("1.4"),
        cop_default=Decimal("1.20"),
        iplv=Decimal("1.30"),
        energy_source="heat",
        compressor_type=None,
        condenser_type=None,
    ),
    CoolingTechnology.TRIPLE_EFFECT_LIBR.value: CoolingTechnologySpec(
        cop_min=Decimal("1.4"),
        cop_max=Decimal("1.8"),
        cop_default=Decimal("1.60"),
        iplv=Decimal("1.70"),
        energy_source="heat",
        compressor_type=None,
        condenser_type=None,
    ),
    CoolingTechnology.AMMONIA_ABSORPTION.value: CoolingTechnologySpec(
        cop_min=Decimal("0.5"),
        cop_max=Decimal("0.7"),
        cop_default=Decimal("0.55"),
        iplv=Decimal("0.58"),
        energy_source="heat",
        compressor_type=None,
        condenser_type=None,
    ),
    # --- Free Cooling ---
    CoolingTechnology.SEAWATER_FREE.value: CoolingTechnologySpec(
        cop_min=Decimal("15.0"),
        cop_max=Decimal("30.0"),
        cop_default=Decimal("20.0"),
        iplv=None,
        energy_source="electricity",
        compressor_type=None,
        condenser_type=None,
    ),
    CoolingTechnology.LAKE_FREE.value: CoolingTechnologySpec(
        cop_min=Decimal("12.0"),
        cop_max=Decimal("25.0"),
        cop_default=Decimal("18.0"),
        iplv=None,
        energy_source="electricity",
        compressor_type=None,
        condenser_type=None,
    ),
    CoolingTechnology.RIVER_FREE.value: CoolingTechnologySpec(
        cop_min=Decimal("10.0"),
        cop_max=Decimal("20.0"),
        cop_default=Decimal("15.0"),
        iplv=None,
        energy_source="electricity",
        compressor_type=None,
        condenser_type=None,
    ),
    CoolingTechnology.AMBIENT_AIR_FREE.value: CoolingTechnologySpec(
        cop_min=Decimal("8.0"),
        cop_max=Decimal("15.0"),
        cop_default=Decimal("10.0"),
        iplv=None,
        energy_source="electricity",
        compressor_type=None,
        condenser_type=None,
    ),
    # --- Thermal Energy Storage ---
    CoolingTechnology.ICE_TES.value: CoolingTechnologySpec(
        cop_min=Decimal("2.8"),
        cop_max=Decimal("4.0"),
        cop_default=Decimal("3.2"),
        iplv=None,
        energy_source="electricity",
        compressor_type=None,
        condenser_type=None,
    ),
    CoolingTechnology.CHILLED_WATER_TES.value: CoolingTechnologySpec(
        cop_min=Decimal("4.5"),
        cop_max=Decimal("6.5"),
        cop_default=Decimal("5.5"),
        iplv=None,
        energy_source="electricity",
        compressor_type=None,
        condenser_type=None,
    ),
    CoolingTechnology.PCM_TES.value: CoolingTechnologySpec(
        cop_min=Decimal("3.5"),
        cop_max=Decimal("5.5"),
        cop_default=Decimal("4.5"),
        iplv=None,
        energy_source="electricity",
        compressor_type=None,
        condenser_type=None,
    ),
    # --- District Cooling ---
    CoolingTechnology.DISTRICT_COOLING.value: CoolingTechnologySpec(
        cop_min=Decimal("3.0"),
        cop_max=Decimal("7.0"),
        cop_default=Decimal("4.0"),
        iplv=None,
        energy_source="mixed",
        compressor_type=None,
        condenser_type=None,
    ),
}

# ---------------------------------------------------------------------------
# 2. DistrictCoolingFactor
# ---------------------------------------------------------------------------

class DistrictCoolingFactor(GreenLangBase):
    """Emission factor record for a district cooling network region.

    Encapsulates the composite emission factor (kgCO2e per GJ of
    cooling delivered), typical technology mix, and notes for a
    specific district cooling region. Used for district cooling
    emission calculations when no supplier-specific data is available.

    Attributes:
        region: Geographic region identifier (e.g. 'dubai_uae',
            'singapore', 'global_default'). Must match a key in the
            DISTRICT_COOLING_FACTORS constant table or be a custom
            user-provided region.
        ef_kgco2e_per_gj: Composite emission factor in kgCO2e per GJ
            of cooling delivered at the building meter. Includes
            generation, distribution, and pump energy emissions.
        technology_mix: Description of the typical cooling technology
            mix for this region (e.g. 'Electric centrifugal',
            'Electric + absorption').
        notes: Additional context about the emission factor source,
            applicability, or limitations.
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
    ef_kgco2e_per_gj: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Emission factor in kgCO2e per GJ cooling delivered",
    )
    technology_mix: str = Field(
        default="",
        max_length=500,
        description="Typical cooling technology mix for this region",
    )
    notes: str = Field(
        default="",
        max_length=500,
        description="Additional context about the emission factor",
    )

    @field_validator("region")
    @classmethod
    def _lowercase_region(cls, v: str) -> str:
        """Normalise region identifier to lowercase."""
        return v.strip().lower()

# ---------------------------------------------------------------------------
# District Cooling Regional Factors Constant Table
# ---------------------------------------------------------------------------

#: Regional emission factors for district cooling networks.
#: Units: kgCO2e per GJ of cooling delivered (at the building meter).
#:
#: Sources:
#:   IEA District Cooling Report (2024).
#:   Dubai Electricity and Water Authority (DEWA) sustainability report.
#:   Singapore Building and Construction Authority (BCA) benchmarks.
#:   Hong Kong Electrical and Mechanical Services Department (EMSD).
#:   US DOE district energy benchmarks.
#:   European District Energy Association (Euroheat & Power).
#:   Japan District Heating and Cooling Association (JDHC).
#:   South Korea Korea District Heating Corporation (KDHC).
#:   India Bureau of Energy Efficiency (BEE).
#:   Australian Government Clean Energy Regulator.
#:   China National Bureau of Statistics heating/cooling sector.
#:   Global default: IEA World Energy Outlook weighted average.
#:
#: These factors represent the average emission intensity of district
#: cooling networks in each region, accounting for the typical
#: technology mix, plant efficiency, distribution losses, and grid
#: carbon intensity. Actual supplier-specific factors should be used
#: when available (Tier 2/3).
DISTRICT_COOLING_FACTORS: Dict[str, DistrictCoolingFactor] = {
    "dubai_uae": DistrictCoolingFactor(
        region="dubai_uae",
        ef_kgco2e_per_gj=Decimal("45.0"),
        technology_mix="Electric + absorption",
        notes="High grid carbon intensity (natural gas), large-scale DC",
    ),
    "singapore": DistrictCoolingFactor(
        region="singapore",
        ef_kgco2e_per_gj=Decimal("35.0"),
        technology_mix="Electric centrifugal",
        notes="Efficient district cooling, Marina Bay and Changi",
    ),
    "hong_kong": DistrictCoolingFactor(
        region="hong_kong",
        ef_kgco2e_per_gj=Decimal("38.0"),
        technology_mix="Electric centrifugal",
        notes="Seawater-cooled condensers, Kai Tak DC system",
    ),
    "us_sun_belt": DistrictCoolingFactor(
        region="us_sun_belt",
        ef_kgco2e_per_gj=Decimal("42.0"),
        technology_mix="Electric mixed",
        notes="High cooling demand, mixed grid carbon intensity",
    ),
    "eu_nordic": DistrictCoolingFactor(
        region="eu_nordic",
        ef_kgco2e_per_gj=Decimal("12.0"),
        technology_mix="Free cooling + electric",
        notes="Seawater/lake free cooling, low-carbon grid, Stockholm/Helsinki",
    ),
    "eu_central": DistrictCoolingFactor(
        region="eu_central",
        ef_kgco2e_per_gj=Decimal("25.0"),
        technology_mix="Electric + absorption",
        notes="Mixed technology, moderate grid carbon, Paris/Vienna",
    ),
    "japan": DistrictCoolingFactor(
        region="japan",
        ef_kgco2e_per_gj=Decimal("32.0"),
        technology_mix="Electric + absorption",
        notes="High efficiency, LNG-heavy grid, Tokyo/Osaka DC",
    ),
    "south_korea": DistrictCoolingFactor(
        region="south_korea",
        ef_kgco2e_per_gj=Decimal("35.0"),
        technology_mix="Electric centrifugal",
        notes="LNG-heavy grid, Seoul/Incheon DC systems",
    ),
    "india": DistrictCoolingFactor(
        region="india",
        ef_kgco2e_per_gj=Decimal("55.0"),
        technology_mix="Electric mixed",
        notes="High grid carbon intensity (coal-dominated), GIFT City DC",
    ),
    "australia": DistrictCoolingFactor(
        region="australia",
        ef_kgco2e_per_gj=Decimal("48.0"),
        technology_mix="Electric centrifugal",
        notes="Coal-heavy grid in some states, Sydney/Melbourne DC",
    ),
    "china": DistrictCoolingFactor(
        region="china",
        ef_kgco2e_per_gj=Decimal("52.0"),
        technology_mix="Electric + absorption",
        notes="Coal-dominated grid, rapid DC expansion in southern cities",
    ),
    "global_default": DistrictCoolingFactor(
        region="global_default",
        ef_kgco2e_per_gj=Decimal("40.0"),
        technology_mix="Mixed",
        notes="IEA global weighted average estimate for district cooling",
    ),
}

# ---------------------------------------------------------------------------
# 3. HeatSourceFactor
# ---------------------------------------------------------------------------

class HeatSourceFactor(GreenLangBase):
    """Emission factor record for an absorption chiller heat source.

    Encapsulates the emission factor (kgCO2e per GJ of heat input)
    for a specific heat source type driving an absorption chiller.
    Used to calculate the thermal component of absorption chiller
    emissions.

    Attributes:
        heat_source: Classification of the heat source.
        ef_kgco2e_per_gj: Emission factor in kgCO2e per GJ of heat
            input to the absorption chiller. Zero for waste heat,
            solar, geothermal, and biogas. Positive for fossil fuel
            combustion. Grid-dependent for electric boiler and heat
            pump sources.
        notes: Additional context about the emission factor source,
            calculation basis, or limitations.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    heat_source: HeatSource = Field(
        ...,
        description="Classification of the heat source",
    )
    ef_kgco2e_per_gj: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Emission factor in kgCO2e per GJ heat input",
    )
    notes: str = Field(
        default="",
        max_length=500,
        description="Additional context about the emission factor",
    )

# ---------------------------------------------------------------------------
# Heat Source Emission Factors Constant Table
# ---------------------------------------------------------------------------

#: Emission factors for heat sources driving absorption chillers.
#: Units: kgCO2e per GJ of heat input.
#:
#: Derived from fuel emission factors and typical boiler efficiencies:
#:   Natural gas: 56.1 kgCO2/GJ / 0.80 = 70.1 kgCO2e/GJ
#:   Fuel oil: 77.4 kgCO2/GJ / 0.80 = 96.8 kgCO2e/GJ
#:   Coal: 94.6 kgCO2/GJ / 0.75 = 126.1 kgCO2e/GJ
#:
#: Sources:
#:   IPCC 2006 Guidelines Vol 2, Ch 2 Table 2.2 (CO2 EFs).
#:   US EPA AP-42 Compilation of Air Emission Factors.
#:   UK DESNZ Greenhouse Gas Reporting Conversion Factors (2024).
#:   GHG Protocol Scope 2 Guidance (2015) Appendix A.
HEAT_SOURCE_FACTORS: Dict[str, HeatSourceFactor] = {
    HeatSource.NATURAL_GAS_STEAM.value: HeatSourceFactor(
        heat_source=HeatSource.NATURAL_GAS_STEAM,
        ef_kgco2e_per_gj=Decimal("70.1"),
        notes="56.1 kgCO2/GJ fuel / 0.80 boiler efficiency",
    ),
    HeatSource.DISTRICT_HEATING.value: HeatSourceFactor(
        heat_source=HeatSource.DISTRICT_HEATING,
        ef_kgco2e_per_gj=Decimal("70.0"),
        notes="Global default district heating EF",
    ),
    HeatSource.WASTE_HEAT.value: HeatSourceFactor(
        heat_source=HeatSource.WASTE_HEAT,
        ef_kgco2e_per_gj=Decimal("0.0"),
        notes="Zero-cost byproduct, no additional fuel combustion",
    ),
    HeatSource.CHP_EXHAUST.value: HeatSourceFactor(
        heat_source=HeatSource.CHP_EXHAUST,
        ef_kgco2e_per_gj=Decimal("0.0"),
        notes="CHP-allocated: cross-reference MRV-011 for actual EF",
    ),
    HeatSource.SOLAR_THERMAL.value: HeatSourceFactor(
        heat_source=HeatSource.SOLAR_THERMAL,
        ef_kgco2e_per_gj=Decimal("0.0"),
        notes="Zero operational emissions from solar thermal collectors",
    ),
    HeatSource.GEOTHERMAL.value: HeatSourceFactor(
        heat_source=HeatSource.GEOTHERMAL,
        ef_kgco2e_per_gj=Decimal("0.0"),
        notes="Zero direct combustion emissions for closed-loop systems",
    ),
    HeatSource.BIOGAS_STEAM.value: HeatSourceFactor(
        heat_source=HeatSource.BIOGAS_STEAM,
        ef_kgco2e_per_gj=Decimal("0.0"),
        notes="Biogenic source; fossil CO2e=0, CH4/N2O counted separately",
    ),
    HeatSource.FUEL_OIL_STEAM.value: HeatSourceFactor(
        heat_source=HeatSource.FUEL_OIL_STEAM,
        ef_kgco2e_per_gj=Decimal("96.8"),
        notes="77.4 kgCO2/GJ fuel / 0.80 boiler efficiency",
    ),
    HeatSource.COAL_STEAM.value: HeatSourceFactor(
        heat_source=HeatSource.COAL_STEAM,
        ef_kgco2e_per_gj=Decimal("126.1"),
        notes="94.6 kgCO2/GJ fuel / 0.75 boiler efficiency",
    ),
    HeatSource.ELECTRIC_BOILER.value: HeatSourceFactor(
        heat_source=HeatSource.ELECTRIC_BOILER,
        ef_kgco2e_per_gj=Decimal("0.0"),
        notes="Grid-dependent: EF = Grid_EF / 0.98 boiler efficiency",
    ),
    HeatSource.HEAT_PUMP.value: HeatSourceFactor(
        heat_source=HeatSource.HEAT_PUMP,
        ef_kgco2e_per_gj=Decimal("0.0"),
        notes="Grid-dependent: EF = Grid_EF / COP_HP",
    ),
}

# ---------------------------------------------------------------------------
# 4. RefrigerantData
# ---------------------------------------------------------------------------

class RefrigerantData(GreenLangBase):
    """GWP and phase-down data for a chiller refrigerant.

    Encapsulates the Global Warming Potential values from IPCC AR5
    and AR6, common usage context, and regulatory phase-down status
    for a specific refrigerant. Used for informational refrigerant
    leakage emission tracking (Scope 1 cross-reference with
    AGENT-MRV-002).

    Attributes:
        refrigerant: Refrigerant type identifier.
        gwp_ar5: 100-year GWP from IPCC Fifth Assessment Report.
        gwp_ar6: 100-year GWP from IPCC Sixth Assessment Report.
        common_use: Description of typical chiller applications.
        phase_down: Regulatory phase-down status and timeline under
            Kigali Amendment and EU F-Gas Regulation.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    refrigerant: Refrigerant = Field(
        ...,
        description="Refrigerant type identifier",
    )
    gwp_ar5: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="100-year GWP from IPCC AR5",
    )
    gwp_ar6: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="100-year GWP from IPCC AR6",
    )
    common_use: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Typical chiller applications",
    )
    phase_down: str = Field(
        default="",
        max_length=500,
        description="Regulatory phase-down status and timeline",
    )

# ---------------------------------------------------------------------------
# Refrigerant GWP Constant Table
# ---------------------------------------------------------------------------

#: GWP values and phase-down status for 11 common chiller refrigerants.
#:
#: Sources:
#:   IPCC Fifth Assessment Report (2014), Table 8.A.1.
#:   IPCC Sixth Assessment Report (2021), Table 7.15.
#:   EU F-Gas Regulation (2024/573).
#:   Kigali Amendment to the Montreal Protocol.
#:   ASHRAE Standard 34 — Designation and Safety Classification.
REFRIGERANT_GWP: Dict[str, RefrigerantData] = {
    Refrigerant.R_134A.value: RefrigerantData(
        refrigerant=Refrigerant.R_134A,
        gwp_ar5=Decimal("1430"),
        gwp_ar6=Decimal("1530"),
        common_use="Centrifugal chillers",
        phase_down="Kigali 2029",
    ),
    Refrigerant.R_410A.value: RefrigerantData(
        refrigerant=Refrigerant.R_410A,
        gwp_ar5=Decimal("2088"),
        gwp_ar6=Decimal("2088"),
        common_use="Scroll/screw chillers",
        phase_down="Kigali 2024+",
    ),
    Refrigerant.R_407C.value: RefrigerantData(
        refrigerant=Refrigerant.R_407C,
        gwp_ar5=Decimal("1774"),
        gwp_ar6=Decimal("1774"),
        common_use="Reciprocating/scroll chillers",
        phase_down="Kigali 2024+",
    ),
    Refrigerant.R_32.value: RefrigerantData(
        refrigerant=Refrigerant.R_32,
        gwp_ar5=Decimal("675"),
        gwp_ar6=Decimal("771"),
        common_use="New split systems",
        phase_down="Transitional",
    ),
    Refrigerant.R_1234ZE_E.value: RefrigerantData(
        refrigerant=Refrigerant.R_1234ZE_E,
        gwp_ar5=Decimal("7"),
        gwp_ar6=Decimal("7"),
        common_use="Low-GWP centrifugal chillers",
        phase_down="Next-gen",
    ),
    Refrigerant.R_1234YF.value: RefrigerantData(
        refrigerant=Refrigerant.R_1234YF,
        gwp_ar5=Decimal("4"),
        gwp_ar6=Decimal("1"),
        common_use="Automotive, small commercial",
        phase_down="Next-gen",
    ),
    Refrigerant.R_513A.value: RefrigerantData(
        refrigerant=Refrigerant.R_513A,
        gwp_ar5=Decimal("631"),
        gwp_ar6=Decimal("631"),
        common_use="Drop-in for R-134a centrifugal",
        phase_down="Transitional",
    ),
    Refrigerant.R_514A.value: RefrigerantData(
        refrigerant=Refrigerant.R_514A,
        gwp_ar5=Decimal("2"),
        gwp_ar6=Decimal("2"),
        common_use="Low-GWP centrifugal chillers",
        phase_down="Next-gen",
    ),
    Refrigerant.R_290.value: RefrigerantData(
        refrigerant=Refrigerant.R_290,
        gwp_ar5=Decimal("3"),
        gwp_ar6=Decimal("0.02"),
        common_use="Small commercial systems",
        phase_down="Natural refrigerant",
    ),
    Refrigerant.R_717.value: RefrigerantData(
        refrigerant=Refrigerant.R_717,
        gwp_ar5=Decimal("0"),
        gwp_ar6=Decimal("0"),
        common_use="Industrial, ammonia absorption",
        phase_down="Natural refrigerant",
    ),
    Refrigerant.R_718.value: RefrigerantData(
        refrigerant=Refrigerant.R_718,
        gwp_ar5=Decimal("0"),
        gwp_ar6=Decimal("0"),
        common_use="Absorption chillers (LiBr-H2O)",
        phase_down="N/A",
    ),
}

# ---------------------------------------------------------------------------
# 5. PartLoadPoint
# ---------------------------------------------------------------------------

class PartLoadPoint(GreenLangBase):
    """Single operating point in a part-load performance curve.

    Represents the COP multiplier and AHRI weighting at a specific
    percentage of full-load capacity. Used to construct custom IPLV
    calculations when per-point COP data is available.

    Attributes:
        load_pct: Load percentage as a decimal fraction (e.g. 0.75
            for 75% load). Must be between 0 and 1 inclusive.
        cop_multiplier: COP multiplier relative to full-load COP at
            this load point (e.g. 1.15 means COP is 15% higher than
            at full load). Typically > 1.0 at part load for
            centrifugal chillers with VSD.
        weighting: AHRI 550/590 weighting factor for this load point.
            The four standard AHRI weights are 0.01 (100%), 0.42
            (75%), 0.45 (50%), and 0.12 (25%). Must sum to 1.0
            across all points in a curve.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
    )

    load_pct: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Load percentage as decimal fraction (0-1)",
    )
    cop_multiplier: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="COP multiplier relative to full-load COP",
    )
    weighting: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="AHRI 550/590 weighting factor for this load point",
    )

# ---------------------------------------------------------------------------
# 6. FacilityInfo
# ---------------------------------------------------------------------------

class FacilityInfo(GreenLangBase):
    """Metadata record for a facility consuming purchased cooling.

    Represents a single physical facility (building, campus, data
    centre, or site) for which Scope 2 cooling purchase emissions
    are calculated. Each facility may be connected to one or more
    cooling suppliers, a district cooling network, and/or use
    on-site purchased cooling services.

    Attributes:
        facility_id: Unique system identifier for the facility (UUID).
            Auto-generated if not provided.
        name: Human-readable facility name or label.
        facility_type: Classification of facility by primary function.
            Determines default cooling intensity benchmarks.
        tenant_id: Owning tenant identifier for multi-tenancy isolation.
        cooling_demand_kwh_th: Annual cooling demand in kWh thermal.
            Optional reference value for benchmarking and validation.
        location: Free-text location description (city, country).
        latitude: Geographic latitude in decimal degrees. Optional,
            used for mapping and regional factor lookup.
        longitude: Geographic longitude in decimal degrees. Optional,
            used for mapping and spatial analysis.
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
    tenant_id: str = Field(
        default="default",
        min_length=1,
        max_length=200,
        description="Owning tenant identifier for multi-tenancy",
    )
    cooling_demand_kwh_th: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description="Annual cooling demand in kWh thermal",
    )
    location: str = Field(
        default="",
        max_length=500,
        description="Free-text location description",
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
    created_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp of facility record creation",
    )

# ---------------------------------------------------------------------------
# 7. CoolingSupplier
# ---------------------------------------------------------------------------

class CoolingSupplier(GreenLangBase):
    """Profile of a cooling service supplier or district cooling provider.

    Represents an external entity that generates and delivers cooling
    energy to the reporting facility. The supplier profile includes
    the cooling technology, rated COP/IPLV, refrigerant type, and
    leakage data for informational Scope 1 tracking.

    Attributes:
        supplier_id: Unique identifier for the cooling supplier (UUID).
            Auto-generated if not provided.
        name: Human-readable supplier name.
        technology: Primary cooling technology used by the supplier.
        cop_rated: Manufacturer-rated COP at design conditions.
            If None, the default COP for the technology is used.
        iplv_rated: AHRI-certified IPLV value. If None, the default
            IPLV for the technology is used (if applicable).
        refrigerant: Primary refrigerant type used in the chiller
            equipment. None for absorption chillers using LiBr-H2O.
        charge_kg: Refrigerant charge in kilograms. Used for leakage
            emission calculation. None if not tracked.
        annual_leak_rate: Annual refrigerant leakage rate as a decimal
            fraction (e.g. 0.02 for 2% per year). Typical range:
            0.5-8% depending on equipment age and maintenance.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    supplier_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the cooling supplier (UUID)",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Human-readable supplier name",
    )
    technology: CoolingTechnology = Field(
        ...,
        description="Primary cooling technology used by supplier",
    )
    cop_rated: Optional[Decimal] = Field(
        default=None,
        gt=Decimal("0"),
        description="Manufacturer-rated COP at design conditions",
    )
    iplv_rated: Optional[Decimal] = Field(
        default=None,
        gt=Decimal("0"),
        description="AHRI-certified IPLV value",
    )
    refrigerant: Optional[Refrigerant] = Field(
        default=None,
        description="Primary refrigerant type in chiller equipment",
    )
    charge_kg: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description="Refrigerant charge in kilograms",
    )
    annual_leak_rate: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Annual refrigerant leakage rate (0-1 fraction)",
    )

# ---------------------------------------------------------------------------
# 8. ElectricChillerRequest
# ---------------------------------------------------------------------------

class ElectricChillerRequest(GreenLangBase):
    """Request for an electric chiller emission calculation.

    Contains all parameters needed to calculate Scope 2 emissions
    from purchased cooling produced by an electric vapour-compression
    chiller. Supports both full-load COP and IPLV part-load weighted
    calculations per AHRI 550/590.

    Attributes:
        cooling_output_kwh_th: Cooling energy output delivered to the
            facility in kilowatt-hours thermal.
        technology: Cooling technology classification. Determines
            default COP, IPLV, compressor type, and condenser type.
        cop_override: Optional measured or site-specific COP override.
            If provided, overrides the default COP for the technology.
        use_iplv: Whether to use IPLV part-load weighted calculation
            instead of full-load COP. Defaults to True for more
            representative annual performance.
        cop_100: Optional COP at 100% load for custom IPLV calculation.
        cop_75: Optional COP at 75% load for custom IPLV calculation.
        cop_50: Optional COP at 50% load for custom IPLV calculation.
        cop_25: Optional COP at 25% load for custom IPLV calculation.
        grid_ef_kgco2e_per_kwh: Grid electricity emission factor in
            kgCO2e per kWh. Required for emission calculation.
        auxiliary_pct: Fraction of cooling output consumed by auxiliary
            equipment (cooling tower fans, condenser water pumps,
            chilled water pumps). Defaults to 5% (0.05).
        facility_id: Optional reference to the consuming facility.
        supplier_id: Optional reference to the cooling supplier.
        tenant_id: Owning tenant identifier for multi-tenancy.
        calculation_tier: Data quality tier for the calculation.
        gwp_source: IPCC Assessment Report for GWP values.
        reporting_period: Time period for the calculation.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    cooling_output_kwh_th: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Cooling output delivered in kWh thermal",
    )
    technology: CoolingTechnology = Field(
        ...,
        description="Cooling technology classification",
    )
    cop_override: Optional[Decimal] = Field(
        default=None,
        gt=Decimal("0"),
        description="Measured COP override for the chiller",
    )
    use_iplv: bool = Field(
        default=True,
        description="Use IPLV part-load weighted calculation",
    )
    cop_100: Optional[Decimal] = Field(
        default=None,
        gt=Decimal("0"),
        description="COP at 100% load for custom IPLV",
    )
    cop_75: Optional[Decimal] = Field(
        default=None,
        gt=Decimal("0"),
        description="COP at 75% load for custom IPLV",
    )
    cop_50: Optional[Decimal] = Field(
        default=None,
        gt=Decimal("0"),
        description="COP at 50% load for custom IPLV",
    )
    cop_25: Optional[Decimal] = Field(
        default=None,
        gt=Decimal("0"),
        description="COP at 25% load for custom IPLV",
    )
    grid_ef_kgco2e_per_kwh: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Grid electricity emission factor (kgCO2e/kWh)",
    )
    auxiliary_pct: Decimal = Field(
        default=Decimal("0.05"),
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Auxiliary energy as fraction of cooling output (0-1)",
    )
    facility_id: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Reference to the consuming facility",
    )
    supplier_id: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Reference to the cooling supplier",
    )
    tenant_id: str = Field(
        default="default",
        min_length=1,
        max_length=200,
        description="Owning tenant identifier for multi-tenancy",
    )
    calculation_tier: DataQualityTier = Field(
        default=DataQualityTier.TIER_1,
        description="Data quality tier for the calculation",
    )
    gwp_source: GWPSource = Field(
        default=GWPSource.AR6,
        description="IPCC Assessment Report for GWP values",
    )
    reporting_period: ReportingPeriod = Field(
        default=ReportingPeriod.ANNUAL,
        description="Time period for the calculation",
    )

# ---------------------------------------------------------------------------
# 9. AbsorptionCoolingRequest
# ---------------------------------------------------------------------------

class AbsorptionCoolingRequest(GreenLangBase):
    """Request for an absorption chiller emission calculation.

    Contains all parameters needed to calculate Scope 2 emissions
    from purchased cooling produced by an absorption chiller.
    Absorption chillers use heat energy (steam, hot water, waste
    heat, or direct fire) to drive the refrigeration cycle, with
    parasitic electricity for pumps and cooling tower fans.

    Attributes:
        cooling_output_kwh_th: Cooling energy output delivered to the
            facility in kilowatt-hours thermal.
        absorption_type: Type of absorption cycle (single/double/
            triple-effect or ammonia).
        heat_source: Heat source driving the absorption cycle.
        cop_override: Optional measured COP override. If provided,
            overrides the default COP for the absorption type.
        parasitic_ratio: Fraction of cooling output consumed by
            parasitic electricity (solution pumps, condenser water
            pumps, cooling tower fans). Defaults to 5% (0.05).
        grid_ef_kgco2e_per_kwh: Grid electricity emission factor for
            parasitic electricity in kgCO2e per kWh.
        heat_source_ef_override: Optional override for the heat source
            emission factor in kgCO2e per GJ. If provided, overrides
            the default from HEAT_SOURCE_FACTORS.
        facility_id: Optional reference to the consuming facility.
        supplier_id: Optional reference to the cooling supplier.
        tenant_id: Owning tenant identifier for multi-tenancy.
        calculation_tier: Data quality tier for the calculation.
        gwp_source: IPCC Assessment Report for GWP values.
        reporting_period: Time period for the calculation.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    cooling_output_kwh_th: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Cooling output delivered in kWh thermal",
    )
    absorption_type: AbsorptionType = Field(
        ...,
        description="Type of absorption cycle",
    )
    heat_source: HeatSource = Field(
        ...,
        description="Heat source driving the absorption cycle",
    )
    cop_override: Optional[Decimal] = Field(
        default=None,
        gt=Decimal("0"),
        description="Measured COP override for absorption chiller",
    )
    parasitic_ratio: Decimal = Field(
        default=Decimal("0.05"),
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Parasitic electricity as fraction of cooling (0-1)",
    )
    grid_ef_kgco2e_per_kwh: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Grid EF for parasitic electricity (kgCO2e/kWh)",
    )
    heat_source_ef_override: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description="Heat source EF override in kgCO2e per GJ",
    )
    facility_id: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Reference to the consuming facility",
    )
    supplier_id: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Reference to the cooling supplier",
    )
    tenant_id: str = Field(
        default="default",
        min_length=1,
        max_length=200,
        description="Owning tenant identifier for multi-tenancy",
    )
    calculation_tier: DataQualityTier = Field(
        default=DataQualityTier.TIER_1,
        description="Data quality tier for the calculation",
    )
    gwp_source: GWPSource = Field(
        default=GWPSource.AR6,
        description="IPCC Assessment Report for GWP values",
    )
    reporting_period: ReportingPeriod = Field(
        default=ReportingPeriod.ANNUAL,
        description="Time period for the calculation",
    )

# ---------------------------------------------------------------------------
# 10. FreeCoolingRequest
# ---------------------------------------------------------------------------

class FreeCoolingRequest(GreenLangBase):
    """Request for a free cooling emission calculation.

    Contains all parameters needed to calculate Scope 2 emissions
    from purchased cooling produced by a free cooling system using
    natural heat sinks (seawater, lake, river, or ambient air).
    Only pump or fan electricity is consumed.

    Attributes:
        cooling_output_kwh_th: Cooling energy output delivered to the
            facility in kilowatt-hours thermal.
        source: Natural heat sink source type.
        cop_override: Optional measured effective COP override. If
            provided, overrides the default COP for the source type.
        grid_ef_kgco2e_per_kwh: Grid electricity emission factor for
            pump/fan electricity in kgCO2e per kWh.
        facility_id: Optional reference to the consuming facility.
        tenant_id: Owning tenant identifier for multi-tenancy.
        calculation_tier: Data quality tier for the calculation.
        gwp_source: IPCC Assessment Report for GWP values.
        reporting_period: Time period for the calculation.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    cooling_output_kwh_th: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Cooling output delivered in kWh thermal",
    )
    source: FreeCoolingSource = Field(
        ...,
        description="Natural heat sink source type",
    )
    cop_override: Optional[Decimal] = Field(
        default=None,
        gt=Decimal("0"),
        description="Measured effective COP override",
    )
    grid_ef_kgco2e_per_kwh: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Grid EF for pump/fan electricity (kgCO2e/kWh)",
    )
    facility_id: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Reference to the consuming facility",
    )
    tenant_id: str = Field(
        default="default",
        min_length=1,
        max_length=200,
        description="Owning tenant identifier for multi-tenancy",
    )
    calculation_tier: DataQualityTier = Field(
        default=DataQualityTier.TIER_1,
        description="Data quality tier for the calculation",
    )
    gwp_source: GWPSource = Field(
        default=GWPSource.AR6,
        description="IPCC Assessment Report for GWP values",
    )
    reporting_period: ReportingPeriod = Field(
        default=ReportingPeriod.ANNUAL,
        description="Time period for the calculation",
    )

# ---------------------------------------------------------------------------
# 11. TESRequest
# ---------------------------------------------------------------------------

class TESRequest(GreenLangBase):
    """Request for a thermal energy storage emission calculation.

    Contains all parameters needed to calculate Scope 2 emissions
    from cooling produced via a thermal energy storage system (ice,
    chilled water, or PCM). TES enables temporal shifting of cooling
    production from peak to off-peak hours, potentially reducing
    emissions when off-peak grid carbon intensity is lower.

    Attributes:
        tes_capacity_kwh_th: TES storage capacity in kWh thermal.
            Represents the total cooling energy stored per charge
            cycle.
        tes_type: Type of thermal energy storage technology.
        cop_charge: Optional COP of the chiller during TES charging.
            If None, the default COP for the TES technology type is
            used. Charging COP is typically lower than standard
            chiller COP due to lower evaporating temperatures
            (especially for ice storage).
        round_trip_efficiency: Round-trip thermal efficiency of the
            TES system as a decimal fraction (0-1). Accounts for
            thermal losses during storage. Defaults vary by TES
            type: ice 0.85, chilled water 0.93, PCM 0.87.
        grid_ef_charge_kgco2e_per_kwh: Grid electricity emission
            factor during TES charging period (typically off-peak)
            in kgCO2e per kWh.
        grid_ef_peak_kgco2e_per_kwh: Optional grid electricity
            emission factor during peak hours. Used to calculate
            emission savings from temporal shifting. If None, no
            savings calculation is performed.
        cop_peak: Optional COP of the chiller that would run during
            peak hours without TES. Used for emission savings
            comparison. If None, the technology default COP is used.
        facility_id: Optional reference to the consuming facility.
        tenant_id: Owning tenant identifier for multi-tenancy.
        calculation_tier: Data quality tier for the calculation.
        gwp_source: IPCC Assessment Report for GWP values.
        reporting_period: Time period for the calculation.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    tes_capacity_kwh_th: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="TES storage capacity in kWh thermal",
    )
    tes_type: TESType = Field(
        ...,
        description="Type of thermal energy storage technology",
    )
    cop_charge: Optional[Decimal] = Field(
        default=None,
        gt=Decimal("0"),
        description="COP of chiller during TES charging",
    )
    round_trip_efficiency: Decimal = Field(
        default=Decimal("0.90"),
        gt=Decimal("0"),
        le=Decimal("1"),
        description="TES round-trip thermal efficiency (0-1)",
    )
    grid_ef_charge_kgco2e_per_kwh: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Grid EF during TES charging period (kgCO2e/kWh)",
    )
    grid_ef_peak_kgco2e_per_kwh: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description="Grid EF during peak hours for savings calc",
    )
    cop_peak: Optional[Decimal] = Field(
        default=None,
        gt=Decimal("0"),
        description="COP of peak-hour chiller for savings comparison",
    )
    facility_id: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Reference to the consuming facility",
    )
    tenant_id: str = Field(
        default="default",
        min_length=1,
        max_length=200,
        description="Owning tenant identifier for multi-tenancy",
    )
    calculation_tier: DataQualityTier = Field(
        default=DataQualityTier.TIER_1,
        description="Data quality tier for the calculation",
    )
    gwp_source: GWPSource = Field(
        default=GWPSource.AR6,
        description="IPCC Assessment Report for GWP values",
    )
    reporting_period: ReportingPeriod = Field(
        default=ReportingPeriod.ANNUAL,
        description="Time period for the calculation",
    )

# ---------------------------------------------------------------------------
# 12. DistrictCoolingRequest
# ---------------------------------------------------------------------------

class DistrictCoolingRequest(GreenLangBase):
    """Request for a district cooling network emission calculation.

    Contains all parameters needed to calculate Scope 2 emissions
    from purchased cooling from a district cooling network. Accounts

from greenlang.schemas import GreenLangBase, utcnow
from greenlang.schemas.enums import ReportingPeriod
    for generation efficiency, distribution losses, and pump energy.

    Attributes:
        cooling_output_kwh_th: Cooling energy output delivered to the
            facility in kilowatt-hours thermal (at the building meter).
        region: Geographic region identifier for regional emission
            factor lookup (e.g. 'dubai_uae', 'singapore',
            'global_default').
        distribution_loss_pct: Fraction of cooling lost in the
            distribution network (0-1). Defaults to 8% (0.08).
            Used to gross up metered consumption.
        pump_energy_kwh: Optional metered pump and distribution
            electricity consumption in kWh. If provided, used for
            precise pump emission calculation. If None, pump energy
            is estimated from distribution loss percentage.
        grid_ef_kgco2e_per_kwh: Optional grid electricity emission
            factor in kgCO2e per kWh. Used for pump energy emissions.
            If None, the regional default is used.
        facility_id: Optional reference to the consuming facility.
        supplier_id: Optional reference to the district cooling
            supplier.
        tenant_id: Owning tenant identifier for multi-tenancy.
        calculation_tier: Data quality tier for the calculation.
        gwp_source: IPCC Assessment Report for GWP values.
        reporting_period: Time period for the calculation.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    cooling_output_kwh_th: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Cooling output delivered in kWh thermal",
    )
    region: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Geographic region for factor lookup",
    )
    distribution_loss_pct: Decimal = Field(
        default=Decimal("0.08"),
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Distribution cooling loss fraction (0-1)",
    )
    pump_energy_kwh: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description="Metered pump/distribution electricity in kWh",
    )
    grid_ef_kgco2e_per_kwh: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description="Grid EF for pump energy emissions (kgCO2e/kWh)",
    )
    facility_id: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Reference to the consuming facility",
    )
    supplier_id: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Reference to the district cooling supplier",
    )
    tenant_id: str = Field(
        default="default",
        min_length=1,
        max_length=200,
        description="Owning tenant identifier for multi-tenancy",
    )
    calculation_tier: DataQualityTier = Field(
        default=DataQualityTier.TIER_1,
        description="Data quality tier for the calculation",
    )
    gwp_source: GWPSource = Field(
        default=GWPSource.AR6,
        description="IPCC Assessment Report for GWP values",
    )
    reporting_period: ReportingPeriod = Field(
        default=ReportingPeriod.ANNUAL,
        description="Time period for the calculation",
    )

    @field_validator("region")
    @classmethod
    def _lowercase_region(cls, v: str) -> str:
        """Normalise region identifier to lowercase."""
        return v.strip().lower()

# ---------------------------------------------------------------------------
# 13. GasEmissionDetail
# ---------------------------------------------------------------------------

class GasEmissionDetail(GreenLangBase):
    """Breakdown of emissions for a single greenhouse gas species.

    Provides the individual gas emission quantity, the GWP multiplier
    used, and the resulting CO2-equivalent value. Used as an element
    in the gas_breakdown list of CalculationResult.

    Attributes:
        gas: Greenhouse gas species.
        quantity_kg: Direct emission quantity in kilograms of the
            gas species.
        gwp_factor: GWP multiplier applied for CO2e conversion.
            For CO2, this is always 1. For CH4 and N2O, depends
            on the selected GWP source (AR4/AR5/AR6/AR6_20YR).
        co2e_kg: CO2-equivalent emission in kilograms. Calculated
            as quantity_kg * gwp_factor.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
    )

    gas: EmissionGas = Field(
        ...,
        description="Greenhouse gas species",
    )
    quantity_kg: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Direct emission quantity in kg",
    )
    gwp_factor: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="GWP multiplier for CO2e conversion",
    )
    co2e_kg: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="CO2-equivalent emission in kg",
    )

# ---------------------------------------------------------------------------
# 14. CalculationResult
# ---------------------------------------------------------------------------

class CalculationResult(GreenLangBase):
    """Result of a Scope 2 cooling purchase emission calculation.

    Contains the complete calculation output including cooling output,
    energy input, COP used, total CO2e emissions, per-gas breakdown,
    and SHA-256 provenance hash for audit trail. This is the primary
    output model for electric chiller, absorption chiller, free
    cooling, and district cooling calculations.

    Attributes:
        calculation_id: Unique identifier for this calculation (UUID).
        calculation_type: Type of cooling calculation performed
            (e.g. 'electric_chiller', 'absorption', 'free_cooling',
            'district_cooling').
        cooling_output_kwh_th: Cooling energy output in kWh thermal.
        energy_input_kwh: Total energy input in kWh (electrical for
            electric/free cooling, combined for absorption/district).
        cop_used: COP value used in the calculation. May be default,
            IPLV, or measured COP depending on request parameters.
        emissions_kgco2e: Total CO2-equivalent emissions in kg.
        gas_breakdown: List of per-gas emission breakdowns.
        calculation_tier: Data quality tier of the calculation.
        provenance_hash: SHA-256 hash of all calculation inputs and
            outputs for complete audit trail.
        trace_steps: Ordered list of calculation trace steps for
            transparency and debugging.
        timestamp: UTC timestamp of the calculation.
        metadata: Additional key-value metadata about the calculation
            (e.g. facility_id, supplier_id, technology, region).
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    calculation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique calculation identifier (UUID)",
    )
    calculation_type: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Type of cooling calculation performed",
    )
    cooling_output_kwh_th: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Cooling energy output in kWh thermal",
    )
    energy_input_kwh: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Total energy input in kWh",
    )
    cop_used: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="COP value used in the calculation",
    )
    emissions_kgco2e: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Total CO2-equivalent emissions in kg",
    )
    gas_breakdown: List[GasEmissionDetail] = Field(
        default_factory=list,
        description="Per-gas emission breakdowns",
    )
    calculation_tier: DataQualityTier = Field(
        ...,
        description="Data quality tier of the calculation",
    )
    provenance_hash: str = Field(
        default="",
        max_length=64,
        description="SHA-256 provenance hash for audit trail",
    )
    trace_steps: List[str] = Field(
        default_factory=list,
        description="Ordered calculation trace steps",
    )
    timestamp: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp of the calculation",
    )
    metadata: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional key-value metadata",
    )

    @field_validator("gas_breakdown")
    @classmethod
    def _validate_gas_breakdown_size(
        cls, v: List[GasEmissionDetail],
    ) -> List[GasEmissionDetail]:
        """Validate that gas breakdown count does not exceed maximum."""
        if len(v) > MAX_GASES_PER_RESULT:
            raise ValueError(
                f"Gas breakdown count {len(v)} exceeds maximum "
                f"{MAX_GASES_PER_RESULT}"
            )
        return v

    @field_validator("trace_steps")
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
# 15. TESCalculationResult
# ---------------------------------------------------------------------------

class TESCalculationResult(GreenLangBase):
    """Result of a thermal energy storage emission calculation.

    Extends the standard CalculationResult concepts with TES-specific
    fields for charge energy, emission savings from temporal shifting,
    and peak emissions avoided.

    Attributes:
        calculation_id: Unique identifier for this calculation (UUID).
        calculation_type: Always 'tes' for TES calculations.
        cooling_output_kwh_th: Cooling energy delivered from TES
            discharge in kWh thermal.
        charge_energy_kwh: Electrical energy consumed during TES
            charging in kWh. Calculated as:
            (tes_capacity / COP_charge) / round_trip_efficiency.
        cop_used: COP value used during TES charging.
        emissions_kgco2e: Total emissions from TES charging in kgCO2e.
        emission_savings_kgco2e: Net emission savings from temporal
            shifting (peak emissions avoided minus charge emissions).
            Positive value indicates emission reduction.
        peak_emissions_avoided_kgco2e: Emissions that would have
            occurred if cooling was produced by a chiller during
            peak hours instead of from TES discharge.
        gas_breakdown: List of per-gas emission breakdowns.
        calculation_tier: Data quality tier.
        provenance_hash: SHA-256 hash for audit trail.
        trace_steps: Ordered calculation trace steps.
        timestamp: UTC timestamp of the calculation.
        metadata: Additional key-value metadata.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    calculation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique calculation identifier (UUID)",
    )
    calculation_type: str = Field(
        default="tes",
        max_length=100,
        description="Calculation type (always 'tes')",
    )
    cooling_output_kwh_th: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Cooling energy delivered from TES in kWh thermal",
    )
    charge_energy_kwh: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Electrical energy consumed during charging in kWh",
    )
    cop_used: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="COP value used during TES charging",
    )
    emissions_kgco2e: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Total emissions from TES charging in kgCO2e",
    )
    emission_savings_kgco2e: Decimal = Field(
        default=Decimal("0"),
        description="Net emission savings from temporal shifting",
    )
    peak_emissions_avoided_kgco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Peak hour emissions avoided by TES discharge",
    )
    gas_breakdown: List[GasEmissionDetail] = Field(
        default_factory=list,
        description="Per-gas emission breakdowns",
    )
    calculation_tier: DataQualityTier = Field(
        ...,
        description="Data quality tier of the calculation",
    )
    provenance_hash: str = Field(
        default="",
        max_length=64,
        description="SHA-256 provenance hash for audit trail",
    )
    trace_steps: List[str] = Field(
        default_factory=list,
        description="Ordered calculation trace steps",
    )
    timestamp: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp of the calculation",
    )
    metadata: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional key-value metadata",
    )

    @field_validator("gas_breakdown")
    @classmethod
    def _validate_gas_breakdown_size(
        cls, v: List[GasEmissionDetail],
    ) -> List[GasEmissionDetail]:
        """Validate that gas breakdown count does not exceed maximum."""
        if len(v) > MAX_GASES_PER_RESULT:
            raise ValueError(
                f"Gas breakdown count {len(v)} exceeds maximum "
                f"{MAX_GASES_PER_RESULT}"
            )
        return v

    @field_validator("trace_steps")
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
# 16. RefrigerantLeakageResult
# ---------------------------------------------------------------------------

class RefrigerantLeakageResult(GreenLangBase):
    """Informational result of refrigerant leakage emission estimation.

    Calculates the CO2-equivalent emissions from estimated annual
    refrigerant leakage. These emissions are Scope 1 (direct
    emissions from the facility) and are tracked here for
    informational purposes only. The formal Scope 1 refrigerant
    accounting is handled by AGENT-MRV-002 (Refrigerants and F-Gas
    Agent).

    Attributes:
        refrigerant: Refrigerant type.
        charge_kg: Total refrigerant charge in kilograms.
        annual_leak_rate: Annual leakage rate as decimal fraction.
        leakage_kg: Estimated annual leakage in kilograms.
            Calculated as charge_kg * annual_leak_rate.
        gwp: GWP value used for CO2e conversion.
        emissions_kgco2e: CO2-equivalent emissions from leakage.
            Calculated as leakage_kg * gwp.
        note: Informational note indicating Scope 1 classification.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    refrigerant: Refrigerant = Field(
        ...,
        description="Refrigerant type",
    )
    charge_kg: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Total refrigerant charge in kg",
    )
    annual_leak_rate: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Annual leakage rate as decimal fraction (0-1)",
    )
    leakage_kg: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Estimated annual leakage in kg",
    )
    gwp: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="GWP value used for CO2e conversion",
    )
    emissions_kgco2e: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="CO2-equivalent emissions from leakage in kg",
    )
    note: str = Field(
        default="Scope 1 - informational only",
        max_length=500,
        description="Informational note on Scope 1 classification",
    )

# ---------------------------------------------------------------------------
# 17. BatchCalculationRequest
# ---------------------------------------------------------------------------

class BatchCalculationRequest(GreenLangBase):
    """Batch request for multiple cooling emission calculations.

    Aggregates multiple calculation request instances (electric
    chiller, absorption, free cooling, TES, or district cooling)
    for parallel processing across a portfolio of facilities.

    Attributes:
        calculations: List of individual calculation requests. Each
            element may be an ElectricChillerRequest,
            AbsorptionCoolingRequest, FreeCoolingRequest, TESRequest,
            or DistrictCoolingRequest (stored as Any for flexibility).
        tenant_id: Owning tenant identifier for multi-tenancy.
        batch_id: Unique identifier for this batch (UUID).
            Auto-generated if not provided.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    calculations: List[Any] = Field(
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
    batch_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique batch identifier (UUID)",
    )

    @field_validator("calculations")
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
# 18. BatchCalculationResult
# ---------------------------------------------------------------------------

class BatchCalculationResult(GreenLangBase):
    """Result of a batch cooling emission calculation.

    Aggregates results from all individual calculations in a batch
    with portfolio-level totals and status tracking.

    Attributes:
        batch_id: Unique identifier for this batch result (UUID).
        status: Overall batch processing status.
        total_calculations: Total number of calculations in the batch.
        completed: Number of calculations that completed successfully.
        failed: Number of calculations that failed.
        results: List of individual CalculationResult instances for
            successful calculations.
        total_emissions_kgco2e: Portfolio-level total CO2e in kg
            across all successful calculations.
        processing_time_ms: Total batch processing time in
            milliseconds.
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
    status: BatchStatus = Field(
        ...,
        description="Overall batch processing status",
    )
    total_calculations: int = Field(
        ...,
        ge=0,
        description="Total number of calculations in batch",
    )
    completed: int = Field(
        ...,
        ge=0,
        description="Number of successful calculations",
    )
    failed: int = Field(
        ...,
        ge=0,
        description="Number of failed calculations",
    )
    results: List[CalculationResult] = Field(
        default_factory=list,
        description="Individual calculation results",
    )
    total_emissions_kgco2e: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Portfolio total CO2e in kg",
    )
    processing_time_ms: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Total batch processing time in milliseconds",
    )

    @field_validator("completed")
    @classmethod
    def _completed_le_total(cls, v: int, info: Any) -> int:
        """Validate that completed does not exceed total."""
        total = info.data.get("total_calculations")
        if total is not None and v > total:
            raise ValueError(
                f"completed ({v}) cannot exceed "
                f"total_calculations ({total})"
            )
        return v

    @field_validator("failed")
    @classmethod
    def _failed_le_total(cls, v: int, info: Any) -> int:
        """Validate that failed does not exceed total."""
        total = info.data.get("total_calculations")
        if total is not None and v > total:
            raise ValueError(
                f"failed ({v}) cannot exceed "
                f"total_calculations ({total})"
            )
        return v

# ---------------------------------------------------------------------------
# 19. UncertaintyRequest
# ---------------------------------------------------------------------------

class UncertaintyRequest(GreenLangBase):
    """Request for uncertainty quantification on a cooling calculation.

    Specifies the calculation result to analyse and the uncertainty
    method parameters (Monte Carlo simulation or analytical
    propagation), along with uncertainty percentages for each
    input parameter category.

    Attributes:
        calculation_result: The CalculationResult to analyse for
            uncertainty.
        iterations: Number of Monte Carlo iterations. Higher values
            give more precise uncertainty estimates but take longer.
        confidence_level: Confidence level for the uncertainty
            interval (e.g. 0.95 for 95% CI).
        seed: Optional random seed for reproducible Monte Carlo runs.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    calculation_result: CalculationResult = Field(
        ...,
        description="Calculation result to analyse for uncertainty",
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
        description="Confidence level for uncertainty interval",
    )
    seed: Optional[int] = Field(
        default=None,
        ge=0,
        description="Random seed for reproducible Monte Carlo runs",
    )

# ---------------------------------------------------------------------------
# 20. UncertaintyResult
# ---------------------------------------------------------------------------

class UncertaintyResult(GreenLangBase):
    """Result of uncertainty quantification for a cooling calculation.

    Provides the mean, standard deviation, confidence interval,
    percentile distribution, coefficient of variation, and method
    details for the CO2e emission estimate.

    Attributes:
        calculation_id: Reference to the original calculation.
        mean_emissions: Mean CO2e estimate in kg from the uncertainty
            analysis.
        std_dev: Standard deviation of the CO2e estimate in kg.
        ci_lower: Lower bound of the confidence interval in kg.
        ci_upper: Upper bound of the confidence interval in kg.
        confidence_level: Confidence level of the interval.
        p5: 5th percentile of the emission distribution in kg.
        p25: 25th percentile in kg.
        p50: 50th percentile (median) in kg.
        p75: 75th percentile in kg.
        p95: 95th percentile in kg.
        cv: Coefficient of variation (std_dev / mean_emissions).
            Dimensionless measure of relative uncertainty.
        iterations: Number of Monte Carlo iterations performed.
        method: Uncertainty method used ('monte_carlo' or
            'analytical').
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
    )

    calculation_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Reference to the original calculation",
    )
    mean_emissions: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Mean CO2e estimate in kg",
    )
    std_dev: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Standard deviation in kg",
    )
    ci_lower: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Lower bound of confidence interval in kg",
    )
    ci_upper: Decimal = Field(
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
    p5: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="5th percentile of emission distribution in kg",
    )
    p25: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="25th percentile in kg",
    )
    p50: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="50th percentile (median) in kg",
    )
    p75: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="75th percentile in kg",
    )
    p95: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="95th percentile in kg",
    )
    cv: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Coefficient of variation (std_dev / mean)",
    )
    iterations: int = Field(
        ...,
        ge=0,
        description="Number of Monte Carlo iterations performed",
    )
    method: str = Field(
        ...,
        max_length=50,
        description="Uncertainty method used",
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
# 21. ComplianceCheckResult
# ---------------------------------------------------------------------------

class ComplianceCheckResult(GreenLangBase):
    """Result of a regulatory compliance check for a cooling calculation.

    Evaluates a completed cooling emission calculation against a
    specific regulatory framework (GHG Protocol, ISO 14064, CSRD,
    CDP, SBTi, ASHRAE 90.1, or EU F-Gas Regulation) and reports
    findings and score.

    Attributes:
        framework: Regulatory framework identifier (e.g.
            'GHG_PROTOCOL', 'ISO_14064', 'CSRD', 'CDP', 'SBTI',
            'ASHRAE_90_1', 'EU_FGAS').
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
        default_factory=utcnow,
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
# 22. AggregationRequest
# ---------------------------------------------------------------------------

class AggregationRequest(GreenLangBase):
    """Request for aggregating multiple cooling calculation results.

    Specifies which calculation results to aggregate and the
    aggregation dimension. Supports grouping by facility, technology,
    region, supplier, or time period.

    Attributes:
        calc_ids: List of CalculationResult calculation_id values
            to include in the aggregation.
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
# 23. AggregationResult
# ---------------------------------------------------------------------------

class AggregationResult(GreenLangBase):
    """Aggregated emission result across multiple cooling calculations.

    Provides portfolio-level or group-level totals for Scope 2
    cooling purchase emissions, grouped by a specified dimension
    (facility, technology, region, supplier, or time period).

    Attributes:
        aggregation_id: Unique identifier for this aggregation (UUID).
        aggregation_type: Dimension used for grouping.
        total_co2e_kg: Aggregated total CO2e in kilograms.
        breakdown: Dictionary of group key to subtotal CO2e in kg.
            Keys depend on the aggregation_type (e.g. facility_id,
            technology, region, supplier_id, or period string).
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
