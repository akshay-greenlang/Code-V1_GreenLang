"""
GL-001 ThermalCommand: Tag Dictionary

This module implements the minimum data dictionary for the ThermalCommand
ProcessHeatOrchestrator system. It provides:

1. Standardized tag naming conventions
2. Unit governance and conversion
3. Tag metadata (description, engineering units, limits)
4. Tag validation rules
5. Tag resolution and lookup

Minimum Required Tags (per specification):
- steam.headerA.pressure (float, bar(g))
- steam.headerA.temperature (float, C)
- steam.headerA.flow_total (float, t/h)
- boiler.B1.fuel_flow (float, kg/h or Nm3/h)
- boiler.B1.max_rate (float, t/h)
- unit.U1.heat_demand (float, MWth)
- sis.permissive.dispatch_enabled (bool)
- alarm.high_pressure.headerA (bool)
- price.electricity.rt (float, $/MWh)
- weather.temp_forecast (float, C)
- cmms.asset.B1.health_score (float, 0-1)

Author: GreenLang Data Integration Team
Version: 1.0.0
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Enumerations
# =============================================================================

class TagDataType(str, Enum):
    """Supported tag data types."""
    FLOAT = "float"
    INT = "int"
    BOOL = "bool"
    STRING = "string"
    ENUM = "enum"
    TIMESTAMP = "timestamp"


class TagCategory(str, Enum):
    """Tag categories for organization."""
    STEAM = "steam"
    BOILER = "boiler"
    UNIT = "unit"
    SIS = "sis"
    ALARM = "alarm"
    PRICE = "price"
    WEATHER = "weather"
    CMMS = "cmms"
    FURNACE = "furnace"
    TURBINE = "turbine"
    PUMP = "pump"
    HX = "heat_exchanger"
    EMISSIONS = "emissions"
    FUEL = "fuel"
    ELECTRICAL = "electrical"
    AMBIENT = "ambient"


class UnitCategory(str, Enum):
    """Engineering unit categories."""
    PRESSURE = "pressure"
    TEMPERATURE = "temperature"
    FLOW_MASS = "flow_mass"
    FLOW_VOLUME = "flow_volume"
    POWER = "power"
    ENERGY = "energy"
    PRICE = "price"
    PERCENTAGE = "percentage"
    DIMENSIONLESS = "dimensionless"
    TIME = "time"
    LENGTH = "length"
    VELOCITY = "velocity"
    CONCENTRATION = "concentration"


class QualityCode(str, Enum):
    """OPC-style quality codes for tag values."""
    GOOD = "good"
    UNCERTAIN = "uncertain"
    BAD = "bad"
    GOOD_LOCAL_OVERRIDE = "good_local_override"
    UNCERTAIN_LAST_KNOWN = "uncertain_last_known"
    UNCERTAIN_SENSOR_CAL = "uncertain_sensor_cal"
    BAD_SENSOR_FAILURE = "bad_sensor_failure"
    BAD_COMM_FAILURE = "bad_comm_failure"
    BAD_OUT_OF_SERVICE = "bad_out_of_service"


# =============================================================================
# Unit Conversion System
# =============================================================================

class UnitConversion:
    """
    Engineering unit conversion system.

    Supports conversion between common industrial units.
    All conversions maintain precision to 6 decimal places.
    """

    # Conversion factors to base SI units
    PRESSURE_TO_PA: Dict[str, float] = {
        "Pa": 1.0,
        "kPa": 1000.0,
        "MPa": 1000000.0,
        "bar": 100000.0,
        "bar(g)": 100000.0,  # Gauge, add atmospheric for absolute
        "bar(a)": 100000.0,
        "psi": 6894.76,
        "psi(g)": 6894.76,
        "psi(a)": 6894.76,
        "atm": 101325.0,
        "mmHg": 133.322,
        "inHg": 3386.39,
        "mbar": 100.0,
    }

    TEMPERATURE_OFFSETS: Dict[str, Tuple[float, float]] = {
        # (multiply factor, offset to Kelvin)
        "K": (1.0, 0.0),
        "C": (1.0, 273.15),
        "F": (5/9, 255.372),  # (F + 459.67) * 5/9 = K
        "R": (5/9, 0.0),
    }

    MASS_FLOW_TO_KGS: Dict[str, float] = {
        "kg/s": 1.0,
        "kg/h": 1/3600,
        "kg/min": 1/60,
        "t/h": 1000/3600,
        "t/d": 1000/86400,
        "lb/h": 0.453592/3600,
        "lb/s": 0.453592,
    }

    VOLUME_FLOW_TO_M3S: Dict[str, float] = {
        "m3/s": 1.0,
        "m3/h": 1/3600,
        "Nm3/h": 1/3600,  # Normal conditions
        "L/s": 0.001,
        "L/min": 0.001/60,
        "gal/min": 0.0000630902,
        "ft3/min": 0.000471947,
    }

    POWER_TO_W: Dict[str, float] = {
        "W": 1.0,
        "kW": 1000.0,
        "MW": 1000000.0,
        "MWth": 1000000.0,  # Thermal MW
        "MWe": 1000000.0,   # Electrical MW
        "hp": 745.7,
        "BTU/h": 0.293071,
        "MMBtu/h": 293071.0,
        "GJ/h": 277778.0,
    }

    ENERGY_TO_J: Dict[str, float] = {
        "J": 1.0,
        "kJ": 1000.0,
        "MJ": 1000000.0,
        "GJ": 1000000000.0,
        "kWh": 3600000.0,
        "MWh": 3600000000.0,
        "BTU": 1055.06,
        "MMBtu": 1055060000.0,
        "therm": 105506000.0,
    }

    @classmethod
    def convert_pressure(
        cls,
        value: float,
        from_unit: str,
        to_unit: str,
        is_gauge: bool = True
    ) -> float:
        """
        Convert pressure between units.

        Args:
            value: Pressure value to convert
            from_unit: Source unit
            to_unit: Target unit
            is_gauge: If True, handle gauge/absolute conversion

        Returns:
            Converted pressure value
        """
        if from_unit not in cls.PRESSURE_TO_PA:
            raise ValueError(f"Unknown pressure unit: {from_unit}")
        if to_unit not in cls.PRESSURE_TO_PA:
            raise ValueError(f"Unknown pressure unit: {to_unit}")

        # Convert to Pa
        pa_value = value * cls.PRESSURE_TO_PA[from_unit]

        # Handle gauge to absolute if needed
        if is_gauge and "(g)" in from_unit and "(a)" in to_unit:
            pa_value += 101325.0  # Add atmospheric pressure
        elif is_gauge and "(a)" in from_unit and "(g)" in to_unit:
            pa_value -= 101325.0  # Subtract atmospheric pressure

        # Convert from Pa to target
        return round(pa_value / cls.PRESSURE_TO_PA[to_unit], 6)

    @classmethod
    def convert_temperature(cls, value: float, from_unit: str, to_unit: str) -> float:
        """
        Convert temperature between units.

        Args:
            value: Temperature value to convert
            from_unit: Source unit (K, C, F, R)
            to_unit: Target unit (K, C, F, R)

        Returns:
            Converted temperature value
        """
        if from_unit not in cls.TEMPERATURE_OFFSETS:
            raise ValueError(f"Unknown temperature unit: {from_unit}")
        if to_unit not in cls.TEMPERATURE_OFFSETS:
            raise ValueError(f"Unknown temperature unit: {to_unit}")

        # Convert to Kelvin
        from_factor, from_offset = cls.TEMPERATURE_OFFSETS[from_unit]
        if from_unit == "C":
            kelvin = value + from_offset
        elif from_unit == "F":
            kelvin = (value + 459.67) * from_factor
        elif from_unit == "R":
            kelvin = value * from_factor
        else:  # K
            kelvin = value

        # Convert from Kelvin to target
        to_factor, to_offset = cls.TEMPERATURE_OFFSETS[to_unit]
        if to_unit == "C":
            return round(kelvin - to_offset, 6)
        elif to_unit == "F":
            return round(kelvin * 9/5 - 459.67, 6)
        elif to_unit == "R":
            return round(kelvin * 9/5, 6)
        else:  # K
            return round(kelvin, 6)

    @classmethod
    def convert_mass_flow(cls, value: float, from_unit: str, to_unit: str) -> float:
        """Convert mass flow rate between units."""
        if from_unit not in cls.MASS_FLOW_TO_KGS:
            raise ValueError(f"Unknown mass flow unit: {from_unit}")
        if to_unit not in cls.MASS_FLOW_TO_KGS:
            raise ValueError(f"Unknown mass flow unit: {to_unit}")

        kgs_value = value * cls.MASS_FLOW_TO_KGS[from_unit]
        return round(kgs_value / cls.MASS_FLOW_TO_KGS[to_unit], 6)

    @classmethod
    def convert_power(cls, value: float, from_unit: str, to_unit: str) -> float:
        """Convert power between units."""
        if from_unit not in cls.POWER_TO_W:
            raise ValueError(f"Unknown power unit: {from_unit}")
        if to_unit not in cls.POWER_TO_W:
            raise ValueError(f"Unknown power unit: {to_unit}")

        w_value = value * cls.POWER_TO_W[from_unit]
        return round(w_value / cls.POWER_TO_W[to_unit], 6)

    @classmethod
    def convert_energy(cls, value: float, from_unit: str, to_unit: str) -> float:
        """Convert energy between units."""
        if from_unit not in cls.ENERGY_TO_J:
            raise ValueError(f"Unknown energy unit: {from_unit}")
        if to_unit not in cls.ENERGY_TO_J:
            raise ValueError(f"Unknown energy unit: {to_unit}")

        j_value = value * cls.ENERGY_TO_J[from_unit]
        return round(j_value / cls.ENERGY_TO_J[to_unit], 6)


# =============================================================================
# Tag Definition Model
# =============================================================================

class TagDefinition(BaseModel):
    """
    Complete tag definition with metadata.

    Provides all information needed to:
    - Validate tag values
    - Convert units
    - Map to/from external systems
    - Display in UI
    """

    # Identity
    tag_name: str = Field(
        ...,
        pattern=r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$",
        description="Fully qualified tag name (e.g., 'steam.headerA.pressure')"
    )
    display_name: str = Field(
        ...,
        description="Human-readable display name"
    )
    description: str = Field(
        ...,
        description="Detailed tag description"
    )

    # Classification
    category: TagCategory = Field(
        ...,
        description="Tag category for organization"
    )
    subcategory: Optional[str] = Field(
        default=None,
        description="Optional subcategory"
    )

    # Data type
    data_type: TagDataType = Field(
        ...,
        description="Tag data type"
    )
    enum_values: Optional[List[str]] = Field(
        default=None,
        description="Valid values for enum type"
    )

    # Engineering units
    unit: str = Field(
        ...,
        description="Engineering unit (e.g., 'bar(g)', 'C', 't/h')"
    )
    unit_category: UnitCategory = Field(
        ...,
        description="Unit category for conversion"
    )

    # Limits
    low_limit: Optional[float] = Field(
        default=None,
        description="Low engineering limit"
    )
    high_limit: Optional[float] = Field(
        default=None,
        description="High engineering limit"
    )
    low_alarm: Optional[float] = Field(
        default=None,
        description="Low alarm setpoint"
    )
    low_warning: Optional[float] = Field(
        default=None,
        description="Low warning setpoint"
    )
    high_warning: Optional[float] = Field(
        default=None,
        description="High warning setpoint"
    )
    high_alarm: Optional[float] = Field(
        default=None,
        description="High alarm setpoint"
    )

    # Default value
    default_value: Optional[Any] = Field(
        default=None,
        description="Default value if not available"
    )

    # Data quality
    scan_rate_ms: int = Field(
        default=1000,
        ge=100,
        le=86400000,
        description="Expected scan rate in milliseconds"
    )
    is_critical: bool = Field(
        default=False,
        description="True if safety-critical tag"
    )

    # External system mapping
    scada_tag: Optional[str] = Field(
        default=None,
        description="Corresponding SCADA tag name"
    )
    opc_node_id: Optional[str] = Field(
        default=None,
        description="OPC UA node ID"
    )
    historian_tag: Optional[str] = Field(
        default=None,
        description="Historian tag name"
    )

    # Metadata
    version: str = Field(
        default="1.0.0",
        description="Tag definition version"
    )
    created_date: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation date"
    )
    modified_date: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last modification date"
    )

    def validate_value(self, value: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate a value against this tag definition.

        Args:
            value: Value to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Type validation
        if self.data_type == TagDataType.FLOAT:
            if not isinstance(value, (int, float)):
                return False, f"Expected float, got {type(value).__name__}"
            value = float(value)
        elif self.data_type == TagDataType.INT:
            if not isinstance(value, int):
                return False, f"Expected int, got {type(value).__name__}"
        elif self.data_type == TagDataType.BOOL:
            if not isinstance(value, bool):
                return False, f"Expected bool, got {type(value).__name__}"
        elif self.data_type == TagDataType.STRING:
            if not isinstance(value, str):
                return False, f"Expected string, got {type(value).__name__}"
        elif self.data_type == TagDataType.ENUM:
            if self.enum_values and str(value) not in self.enum_values:
                return False, f"Value '{value}' not in enum: {self.enum_values}"

        # Range validation for numeric types
        if self.data_type in (TagDataType.FLOAT, TagDataType.INT):
            if self.low_limit is not None and value < self.low_limit:
                return False, f"Value {value} below low limit {self.low_limit}"
            if self.high_limit is not None and value > self.high_limit:
                return False, f"Value {value} above high limit {self.high_limit}"

        return True, None

    def get_alarm_status(self, value: float) -> str:
        """
        Get alarm status for a numeric value.

        Returns: 'normal', 'low_alarm', 'low_warning', 'high_warning', 'high_alarm'
        """
        if self.data_type not in (TagDataType.FLOAT, TagDataType.INT):
            return "normal"

        if self.low_alarm is not None and value < self.low_alarm:
            return "low_alarm"
        if self.low_warning is not None and value < self.low_warning:
            return "low_warning"
        if self.high_alarm is not None and value > self.high_alarm:
            return "high_alarm"
        if self.high_warning is not None and value > self.high_warning:
            return "high_warning"

        return "normal"


# =============================================================================
# Tag Value Model
# =============================================================================

class TagValue(BaseModel):
    """
    Runtime tag value with quality and timestamp.
    """

    tag_name: str = Field(
        ...,
        description="Tag name reference"
    )
    value: Any = Field(
        ...,
        description="Tag value"
    )
    timestamp: datetime = Field(
        ...,
        description="Value timestamp (UTC)"
    )
    quality: QualityCode = Field(
        default=QualityCode.GOOD,
        description="Value quality code"
    )
    unit: Optional[str] = Field(
        default=None,
        description="Engineering unit (if different from definition)"
    )
    source: Optional[str] = Field(
        default=None,
        description="Data source identifier"
    )


# =============================================================================
# Tag Dictionary Implementation
# =============================================================================

class TagDictionary:
    """
    Central tag dictionary for ThermalCommand.

    Provides:
    - Tag registration and lookup
    - Value validation
    - Unit conversion
    - External system mapping
    - Tag pattern matching
    """

    def __init__(self):
        """Initialize empty tag dictionary."""
        self._tags: Dict[str, TagDefinition] = {}
        self._by_category: Dict[TagCategory, List[str]] = {}
        self._scada_mapping: Dict[str, str] = {}
        self._opc_mapping: Dict[str, str] = {}

        # Register minimum required tags
        self._register_minimum_tags()

    def _register_minimum_tags(self):
        """Register the minimum required tags per specification."""
        minimum_tags = [
            # Steam header tags
            TagDefinition(
                tag_name="steam.headerA.pressure",
                display_name="Header A Pressure",
                description="Steam header A pressure in bar gauge",
                category=TagCategory.STEAM,
                subcategory="header",
                data_type=TagDataType.FLOAT,
                unit="bar(g)",
                unit_category=UnitCategory.PRESSURE,
                low_limit=0.0,
                high_limit=50.0,
                low_alarm=5.0,
                low_warning=8.0,
                high_warning=42.0,
                high_alarm=45.0,
                scan_rate_ms=1000,
                is_critical=True,
                scada_tag="STEAM_HDR_A_PIC_001.PV",
            ),
            TagDefinition(
                tag_name="steam.headerA.temperature",
                display_name="Header A Temperature",
                description="Steam header A temperature in Celsius",
                category=TagCategory.STEAM,
                subcategory="header",
                data_type=TagDataType.FLOAT,
                unit="C",
                unit_category=UnitCategory.TEMPERATURE,
                low_limit=100.0,
                high_limit=550.0,
                low_alarm=200.0,
                low_warning=250.0,
                high_warning=480.0,
                high_alarm=500.0,
                scan_rate_ms=1000,
                is_critical=True,
                scada_tag="STEAM_HDR_A_TIC_001.PV",
            ),
            TagDefinition(
                tag_name="steam.headerA.flow_total",
                display_name="Header A Total Flow",
                description="Steam header A total flow rate in tonnes per hour",
                category=TagCategory.STEAM,
                subcategory="header",
                data_type=TagDataType.FLOAT,
                unit="t/h",
                unit_category=UnitCategory.FLOW_MASS,
                low_limit=0.0,
                high_limit=500.0,
                low_alarm=10.0,
                low_warning=20.0,
                high_warning=400.0,
                high_alarm=450.0,
                scan_rate_ms=1000,
                is_critical=False,
                scada_tag="STEAM_HDR_A_FIC_001.PV",
            ),

            # Boiler tags
            TagDefinition(
                tag_name="boiler.B1.fuel_flow",
                display_name="Boiler B1 Fuel Flow",
                description="Boiler B1 fuel flow rate in kg/h or Nm3/h depending on fuel type",
                category=TagCategory.BOILER,
                subcategory="fuel",
                data_type=TagDataType.FLOAT,
                unit="kg/h",
                unit_category=UnitCategory.FLOW_MASS,
                low_limit=0.0,
                high_limit=10000.0,
                low_warning=100.0,
                high_warning=9000.0,
                high_alarm=9500.0,
                scan_rate_ms=1000,
                is_critical=False,
                scada_tag="BLR_B1_FIC_FUEL_001.PV",
            ),
            TagDefinition(
                tag_name="boiler.B1.max_rate",
                display_name="Boiler B1 Maximum Rate",
                description="Boiler B1 maximum steam production rate in t/h",
                category=TagCategory.BOILER,
                subcategory="capacity",
                data_type=TagDataType.FLOAT,
                unit="t/h",
                unit_category=UnitCategory.FLOW_MASS,
                low_limit=0.0,
                high_limit=200.0,
                default_value=100.0,
                scan_rate_ms=60000,  # Updated less frequently
                is_critical=False,
                scada_tag="BLR_B1_MAX_CAPACITY",
            ),
            TagDefinition(
                tag_name="boiler.B1.steam_output",
                display_name="Boiler B1 Steam Output",
                description="Boiler B1 current steam output in t/h",
                category=TagCategory.BOILER,
                subcategory="output",
                data_type=TagDataType.FLOAT,
                unit="t/h",
                unit_category=UnitCategory.FLOW_MASS,
                low_limit=0.0,
                high_limit=200.0,
                scan_rate_ms=1000,
                is_critical=False,
                scada_tag="BLR_B1_FIC_STEAM_001.PV",
            ),
            TagDefinition(
                tag_name="boiler.B1.efficiency",
                display_name="Boiler B1 Efficiency",
                description="Boiler B1 calculated thermal efficiency percentage",
                category=TagCategory.BOILER,
                subcategory="performance",
                data_type=TagDataType.FLOAT,
                unit="%",
                unit_category=UnitCategory.PERCENTAGE,
                low_limit=0.0,
                high_limit=100.0,
                low_warning=75.0,
                low_alarm=70.0,
                scan_rate_ms=60000,
                is_critical=False,
            ),

            # Unit heat demand
            TagDefinition(
                tag_name="unit.U1.heat_demand",
                display_name="Unit U1 Heat Demand",
                description="Process unit U1 thermal heat demand in MWth",
                category=TagCategory.UNIT,
                subcategory="demand",
                data_type=TagDataType.FLOAT,
                unit="MWth",
                unit_category=UnitCategory.POWER,
                low_limit=0.0,
                high_limit=100.0,
                scan_rate_ms=5000,
                is_critical=False,
                scada_tag="UNIT_U1_HEAT_DEMAND",
            ),

            # SIS permissive
            TagDefinition(
                tag_name="sis.permissive.dispatch_enabled",
                display_name="Dispatch Enabled Permissive",
                description="SIS master permissive for automated dispatch operations",
                category=TagCategory.SIS,
                subcategory="permissive",
                data_type=TagDataType.BOOL,
                unit="",
                unit_category=UnitCategory.DIMENSIONLESS,
                default_value=False,
                scan_rate_ms=500,
                is_critical=True,
                scada_tag="SIS_DISPATCH_PERM",
            ),

            # Alarm tags
            TagDefinition(
                tag_name="alarm.high_pressure.headerA",
                display_name="Header A High Pressure Alarm",
                description="Steam header A high pressure alarm status",
                category=TagCategory.ALARM,
                subcategory="pressure",
                data_type=TagDataType.BOOL,
                unit="",
                unit_category=UnitCategory.DIMENSIONLESS,
                default_value=False,
                scan_rate_ms=500,
                is_critical=True,
                scada_tag="ALM_STEAM_HDR_A_PAH",
            ),

            # Price tags
            TagDefinition(
                tag_name="price.electricity.rt",
                display_name="Real-Time Electricity Price",
                description="Real-time electricity price from ISO/RTO market",
                category=TagCategory.PRICE,
                subcategory="electricity",
                data_type=TagDataType.FLOAT,
                unit="$/MWh",
                unit_category=UnitCategory.PRICE,
                low_limit=-500.0,  # Can be negative
                high_limit=10000.0,
                scan_rate_ms=300000,  # 5-minute updates
                is_critical=False,
            ),
            TagDefinition(
                tag_name="price.electricity.da",
                display_name="Day-Ahead Electricity Price",
                description="Day-ahead electricity price from ISO/RTO market",
                category=TagCategory.PRICE,
                subcategory="electricity",
                data_type=TagDataType.FLOAT,
                unit="$/MWh",
                unit_category=UnitCategory.PRICE,
                low_limit=-500.0,
                high_limit=10000.0,
                scan_rate_ms=3600000,  # Hourly updates
                is_critical=False,
            ),
            TagDefinition(
                tag_name="price.natural_gas.spot",
                display_name="Natural Gas Spot Price",
                description="Natural gas spot price",
                category=TagCategory.PRICE,
                subcategory="fuel",
                data_type=TagDataType.FLOAT,
                unit="$/MMBtu",
                unit_category=UnitCategory.PRICE,
                low_limit=0.0,
                high_limit=100.0,
                scan_rate_ms=3600000,
                is_critical=False,
            ),

            # Weather tags
            TagDefinition(
                tag_name="weather.temp_forecast",
                display_name="Temperature Forecast",
                description="Forecasted ambient temperature in Celsius",
                category=TagCategory.WEATHER,
                subcategory="forecast",
                data_type=TagDataType.FLOAT,
                unit="C",
                unit_category=UnitCategory.TEMPERATURE,
                low_limit=-50.0,
                high_limit=60.0,
                scan_rate_ms=3600000,  # Hourly
                is_critical=False,
            ),
            TagDefinition(
                tag_name="weather.temp_actual",
                display_name="Actual Temperature",
                description="Current ambient temperature in Celsius",
                category=TagCategory.WEATHER,
                subcategory="actual",
                data_type=TagDataType.FLOAT,
                unit="C",
                unit_category=UnitCategory.TEMPERATURE,
                low_limit=-50.0,
                high_limit=60.0,
                scan_rate_ms=60000,
                is_critical=False,
                scada_tag="AMBIENT_TIC_001.PV",
            ),
            TagDefinition(
                tag_name="weather.humidity",
                display_name="Ambient Humidity",
                description="Current ambient relative humidity percentage",
                category=TagCategory.WEATHER,
                subcategory="actual",
                data_type=TagDataType.FLOAT,
                unit="%",
                unit_category=UnitCategory.PERCENTAGE,
                low_limit=0.0,
                high_limit=100.0,
                scan_rate_ms=60000,
                is_critical=False,
            ),

            # CMMS health score
            TagDefinition(
                tag_name="cmms.asset.B1.health_score",
                display_name="Boiler B1 Health Score",
                description="CMMS-calculated equipment health score (0-1)",
                category=TagCategory.CMMS,
                subcategory="health",
                data_type=TagDataType.FLOAT,
                unit="",
                unit_category=UnitCategory.DIMENSIONLESS,
                low_limit=0.0,
                high_limit=1.0,
                low_alarm=0.3,
                low_warning=0.5,
                scan_rate_ms=3600000,  # Hourly from CMMS
                is_critical=False,
            ),

            # Additional common tags
            TagDefinition(
                tag_name="steam.headerB.pressure",
                display_name="Header B Pressure",
                description="Steam header B pressure in bar gauge",
                category=TagCategory.STEAM,
                subcategory="header",
                data_type=TagDataType.FLOAT,
                unit="bar(g)",
                unit_category=UnitCategory.PRESSURE,
                low_limit=0.0,
                high_limit=20.0,
                low_alarm=2.0,
                low_warning=3.0,
                high_warning=16.0,
                high_alarm=18.0,
                scan_rate_ms=1000,
                is_critical=True,
                scada_tag="STEAM_HDR_B_PIC_001.PV",
            ),
            TagDefinition(
                tag_name="steam.headerB.temperature",
                display_name="Header B Temperature",
                description="Steam header B temperature in Celsius",
                category=TagCategory.STEAM,
                subcategory="header",
                data_type=TagDataType.FLOAT,
                unit="C",
                unit_category=UnitCategory.TEMPERATURE,
                low_limit=100.0,
                high_limit=300.0,
                scan_rate_ms=1000,
                is_critical=True,
                scada_tag="STEAM_HDR_B_TIC_001.PV",
            ),
            TagDefinition(
                tag_name="emissions.nox.stack1",
                display_name="Stack 1 NOx",
                description="NOx concentration at stack 1 in ppm",
                category=TagCategory.EMISSIONS,
                subcategory="nox",
                data_type=TagDataType.FLOAT,
                unit="ppm",
                unit_category=UnitCategory.CONCENTRATION,
                low_limit=0.0,
                high_limit=500.0,
                high_warning=80.0,
                high_alarm=100.0,
                scan_rate_ms=1000,
                is_critical=True,
                scada_tag="CEMS_STACK1_NOX.PV",
            ),
            TagDefinition(
                tag_name="emissions.co2.stack1",
                display_name="Stack 1 CO2",
                description="CO2 concentration at stack 1 in percent",
                category=TagCategory.EMISSIONS,
                subcategory="co2",
                data_type=TagDataType.FLOAT,
                unit="%",
                unit_category=UnitCategory.PERCENTAGE,
                low_limit=0.0,
                high_limit=20.0,
                scan_rate_ms=1000,
                is_critical=False,
                scada_tag="CEMS_STACK1_CO2.PV",
            ),
        ]

        for tag in minimum_tags:
            self.register_tag(tag)

    def register_tag(self, tag: TagDefinition) -> None:
        """
        Register a tag in the dictionary.

        Args:
            tag: Tag definition to register
        """
        self._tags[tag.tag_name] = tag

        # Update category index
        if tag.category not in self._by_category:
            self._by_category[tag.category] = []
        if tag.tag_name not in self._by_category[tag.category]:
            self._by_category[tag.category].append(tag.tag_name)

        # Update external mappings
        if tag.scada_tag:
            self._scada_mapping[tag.scada_tag] = tag.tag_name
        if tag.opc_node_id:
            self._opc_mapping[tag.opc_node_id] = tag.tag_name

    def get_tag(self, tag_name: str) -> Optional[TagDefinition]:
        """
        Get tag definition by name.

        Args:
            tag_name: Fully qualified tag name

        Returns:
            Tag definition or None if not found
        """
        return self._tags.get(tag_name)

    def get_tags_by_category(self, category: TagCategory) -> List[TagDefinition]:
        """
        Get all tags in a category.

        Args:
            category: Tag category

        Returns:
            List of tag definitions
        """
        tag_names = self._by_category.get(category, [])
        return [self._tags[name] for name in tag_names]

    def get_tags_by_pattern(self, pattern: str) -> List[TagDefinition]:
        """
        Get tags matching a pattern.

        Args:
            pattern: Regex pattern to match tag names

        Returns:
            List of matching tag definitions
        """
        regex = re.compile(pattern)
        return [tag for name, tag in self._tags.items() if regex.match(name)]

    def resolve_scada_tag(self, scada_tag: str) -> Optional[str]:
        """
        Resolve SCADA tag to canonical tag name.

        Args:
            scada_tag: SCADA system tag name

        Returns:
            Canonical tag name or None
        """
        return self._scada_mapping.get(scada_tag)

    def resolve_opc_node(self, opc_node_id: str) -> Optional[str]:
        """
        Resolve OPC UA node ID to canonical tag name.

        Args:
            opc_node_id: OPC UA node identifier

        Returns:
            Canonical tag name or None
        """
        return self._opc_mapping.get(opc_node_id)

    def validate_value(
        self,
        tag_name: str,
        value: Any
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate a value against tag definition.

        Args:
            tag_name: Tag name
            value: Value to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        tag = self.get_tag(tag_name)
        if not tag:
            return False, f"Unknown tag: {tag_name}"
        return tag.validate_value(value)

    def convert_value(
        self,
        tag_name: str,
        value: float,
        from_unit: str,
        to_unit: Optional[str] = None
    ) -> float:
        """
        Convert a tag value between units.

        Args:
            tag_name: Tag name
            value: Value to convert
            from_unit: Source unit
            to_unit: Target unit (default: tag's defined unit)

        Returns:
            Converted value
        """
        tag = self.get_tag(tag_name)
        if not tag:
            raise ValueError(f"Unknown tag: {tag_name}")

        target_unit = to_unit or tag.unit

        if tag.unit_category == UnitCategory.PRESSURE:
            return UnitConversion.convert_pressure(value, from_unit, target_unit)
        elif tag.unit_category == UnitCategory.TEMPERATURE:
            return UnitConversion.convert_temperature(value, from_unit, target_unit)
        elif tag.unit_category == UnitCategory.FLOW_MASS:
            return UnitConversion.convert_mass_flow(value, from_unit, target_unit)
        elif tag.unit_category == UnitCategory.POWER:
            return UnitConversion.convert_power(value, from_unit, target_unit)
        elif tag.unit_category == UnitCategory.ENERGY:
            return UnitConversion.convert_energy(value, from_unit, target_unit)
        else:
            # No conversion available
            if from_unit != target_unit:
                raise ValueError(
                    f"Cannot convert {from_unit} to {target_unit} "
                    f"for unit category {tag.unit_category}"
                )
            return value

    def get_critical_tags(self) -> List[TagDefinition]:
        """Get all safety-critical tags."""
        return [tag for tag in self._tags.values() if tag.is_critical]

    def get_all_tags(self) -> Dict[str, TagDefinition]:
        """Get all registered tags."""
        return self._tags.copy()

    def export_to_dict(self) -> Dict[str, Any]:
        """Export tag dictionary to serializable format."""
        return {
            "version": "1.0.0",
            "generated": datetime.utcnow().isoformat(),
            "tag_count": len(self._tags),
            "tags": {
                name: tag.model_dump() for name, tag in self._tags.items()
            }
        }

    def __len__(self) -> int:
        return len(self._tags)

    def __contains__(self, tag_name: str) -> bool:
        return tag_name in self._tags

    def __iter__(self):
        return iter(self._tags.values())


# =============================================================================
# Singleton Instance
# =============================================================================

# Global tag dictionary instance
_tag_dictionary: Optional[TagDictionary] = None


def get_tag_dictionary() -> TagDictionary:
    """
    Get the global tag dictionary instance.

    Returns:
        TagDictionary singleton instance
    """
    global _tag_dictionary
    if _tag_dictionary is None:
        _tag_dictionary = TagDictionary()
    return _tag_dictionary


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "TagDataType",
    "TagCategory",
    "UnitCategory",
    "QualityCode",
    # Models
    "TagDefinition",
    "TagValue",
    # Classes
    "TagDictionary",
    "UnitConversion",
    # Functions
    "get_tag_dictionary",
]
