# -*- coding: utf-8 -*-
# climatenza_app/schemas/feasibility.py

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class Site(BaseModel):
    """
    Defines the geographical and physical properties of the project site.
    """
    name: str = Field(..., description="Name of the site, e.g., 'Dairy Plant – Vijayawada'")
    country: str = Field(..., description="Two-letter ISO country code, e.g., 'IN'")
    lat: float = Field(..., description="Latitude in decimal degrees")
    lon: float = Field(..., description="Longitude in decimal degrees")
    tz: str = Field(..., description="Timezone, e.g., 'Asia/Kolkata'")
    elevation_m: Optional[float] = Field(None, description="Elevation in meters above sea level")
    land_area_m2: Optional[float] = Field(None, description="Available land area in square meters")
    roof_area_m2: Optional[float] = Field(None, description="Available roof area in square meters")
    ambient_dust_level: Optional[Literal["low", "med", "high"]] = Field("med", description="Ambient dust level for soiling calculations")


class ProcessDemand(BaseModel):
    """
    Defines the thermal energy demand of the industrial process.
    """
    medium: Literal["hot_water", "steam"] = Field(..., description="The heat transfer medium")
    temp_in_C: float = Field(..., description="Inlet temperature of the medium in Celsius")
    temp_out_C: float = Field(..., description="Required outlet temperature of the medium in Celsius")
    pressure_bar: Optional[float] = Field(None, description="For steam, the required pressure in bar")
    flow_profile: str = Field(..., description="Path to a CSV file with a timestamp and flow rate (kg/s or L/s)")
    schedule: Dict[str, Any] = Field(..., description="Defines the operational schedule, including workdays and shutdowns")


class Boiler(BaseModel):
    """
    Defines the existing boiler system to establish a baseline.
    """
    type: Literal["NG", "HSD", "coal", "biomass"] = Field(..., description="Type of fuel used by the boiler")
    rated_steam_tph: Optional[float] = Field(None, description="Rated steam generation in tonnes per hour")
    efficiency_pct: float = Field(..., description="Overall thermal efficiency of the boiler in percent")
    setpoints: Dict[str, float] = Field({}, description="Operational setpoints for the boiler")


class SolarConfig(BaseModel):
    """
    Configuration for the proposed solar thermal system.
    """
    tech: Literal["ASC", "T160"] = Field(..., description="The parabolic trough collector technology to be used")
    max_land_m2: Optional[float] = Field(None, description="Maximum land area to be used for the solar field")
    max_roof_m2: Optional[float] = Field(None, description="Maximum roof area to be used for the solar field")
    orientation: Literal["N-S", "E-W"] = Field("N-S", description="Orientation of the collector rows")
    row_spacing_factor: float = Field(2.2, description="Row spacing as a multiple of aperture width to minimize shading")
    tracking: Literal["1-axis", "none"] = Field("1-axis", description="Type of solar tracking")


class FinanceInputs(BaseModel):
    """
    Financial parameters for the economic analysis.
    """
    currency: str = Field("USD", description="Currency for all financial calculations")
    discount_rate_pct: float = Field(..., description="Discount rate for LCOH calculation")
    capex_breakdown: Dict[str, float] = Field(..., description="CAPEX breakdown per square meter of aperture")
    opex_pct_of_capex: float = Field(..., description="Annual OPEX as a percentage of total CAPEX")
    tariff_fuel_per_kWh: float = Field(..., description="Cost of the baseline fuel per kWh")
    tariff_elec_per_kWh: float = Field(..., description="Cost of electricity for parasitics per kWh")
    escalation_fuel_pct: float = Field(..., description="Annual escalation rate for fuel tariffs")
    escalation_elec_pct: float = Field(..., description="Annual escalation rate for electricity tariffs")


class Assumptions(BaseModel):
    """
    Key technical and operational assumptions for the simulation.
    """
    cleaning_cycle_days: int = Field(14, description="Frequency of collector cleaning")
    soiling_loss_pct: float = Field(3.0, description="Average optical loss due to soiling")
    availability_pct: float = Field(96.0, description="System uptime percentage")
    parasitic_kWh_per_m2yr: float = Field(12.0, description="Annual parasitic electricity consumption per square meter")
    insulation_kW_per_K_per_m: float = Field(0.002, description="Thermal loss factor for piping")


class FeasibilityInput(BaseModel):
    """
    The root model for a complete feasibility study input file.
    """
    site: Site
    process_demand: ProcessDemand
    boiler: Boiler
    solar_config: SolarConfig
    finance: FinanceInputs
    assumptions: Assumptions