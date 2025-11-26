"""
Fuel Flow Metering Connector for GL-010 EMISSIONWATCH.

Provides integration with fuel flow meters and composition analyzers for
accurate heat input calculations and fuel tracking. Supports natural gas,
oil, and coal measurement systems per EPA 40 CFR Part 75 requirements.

Author: GL-DataIntegrationEngineer
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from decimal import Decimal
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
import asyncio
import logging
import time
import uuid

from pydantic import BaseModel, Field, ConfigDict, field_validator

from .base_connector import (
    BaseConnector,
    BaseConnectorConfig,
    ConnectionState,
    ConnectorType,
    HealthCheckResult,
    HealthStatus,
    ConnectorError,
    ConnectionError,
    ConfigurationError,
    ValidationError,
    DataQualityError,
    with_retry,
)

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class FuelType(str, Enum):
    """Types of fuels."""

    NATURAL_GAS = "natural_gas"
    DIESEL = "diesel"
    FUEL_OIL_NO2 = "fuel_oil_no2"
    FUEL_OIL_NO4 = "fuel_oil_no4"
    FUEL_OIL_NO6 = "fuel_oil_no6"
    RESIDUAL_OIL = "residual_oil"
    COAL_BITUMINOUS = "coal_bituminous"
    COAL_SUBBITUMINOUS = "coal_subbituminous"
    COAL_LIGNITE = "coal_lignite"
    COAL_ANTHRACITE = "coal_anthracite"
    PETROLEUM_COKE = "petroleum_coke"
    LPG = "lpg"
    PROPANE = "propane"
    BUTANE = "butane"
    BIOGAS = "biogas"
    BIOMASS = "biomass"
    WOOD = "wood"
    WASTE = "waste"
    HYDROGEN = "hydrogen"


class MeterType(str, Enum):
    """Types of flow meters."""

    # Gas meters
    ULTRASONIC = "ultrasonic"
    TURBINE = "turbine"
    ORIFICE = "orifice"
    VORTEX = "vortex"
    CORIOLIS = "coriolis"
    THERMAL_MASS = "thermal_mass"
    ROTARY = "rotary"
    DIAPHRAGM = "diaphragm"

    # Liquid meters
    POSITIVE_DISPLACEMENT = "positive_displacement"
    OVAL_GEAR = "oval_gear"
    NUTATING_DISC = "nutating_disc"
    MAGNETIC = "magnetic"

    # Solid fuel
    BELT_SCALE = "belt_scale"
    GRAVIMETRIC_FEEDER = "gravimetric_feeder"
    LOSS_IN_WEIGHT = "loss_in_weight"
    NUCLEAR_GAUGE = "nuclear_gauge"


class MeterVendor(str, Enum):
    """Flow meter vendors."""

    EMERSON = "emerson"
    DANIEL = "daniel"
    HONEYWELL = "honeywell"
    SIEMENS = "siemens"
    ABB = "abb"
    YOKOGAWA = "yokogawa"
    ENDRESS_HAUSER = "endress_hauser"
    KROHNE = "krohne"
    CAMERON = "cameron"
    THERMO_FISHER = "thermo_fisher"
    ITRON = "itron"
    ELSTER = "elster"
    GENERIC = "generic"


class FlowUnit(str, Enum):
    """Flow measurement units."""

    # Volumetric gas
    SCF = "scf"  # Standard cubic feet
    MCF = "mcf"  # Thousand standard cubic feet
    MMSCF = "mmscf"  # Million standard cubic feet
    SCM = "scm"  # Standard cubic meters
    NM3 = "nm3"  # Normal cubic meters

    # Volumetric liquid
    GALLON = "gallon"
    BARREL = "barrel"
    LITER = "liter"
    M3 = "m3"  # Cubic meters

    # Mass
    LB = "lb"  # Pounds
    TON = "ton"  # Short ton
    TONNE = "tonne"  # Metric ton
    KG = "kg"  # Kilograms


class EnergyUnit(str, Enum):
    """Energy units."""

    BTU = "btu"
    MMBTU = "mmbtu"  # Million BTU
    THERM = "therm"
    GJ = "gj"  # Gigajoule
    MJ = "mj"  # Megajoule
    MWH = "mwh"  # Megawatt-hour
    KWH = "kwh"  # Kilowatt-hour


class CompositionMethod(str, Enum):
    """Gas composition analysis methods."""

    GAS_CHROMATOGRAPHY = "gas_chromatography"
    INFRARED = "infrared"
    DEFAULT_VALUES = "default_values"
    SAMPLING = "sampling"


# =============================================================================
# Pydantic Models
# =============================================================================


class FlowReading(BaseModel):
    """Single flow meter reading."""

    model_config = ConfigDict(frozen=True)

    reading_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Reading identifier"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Reading timestamp"
    )
    meter_id: str = Field(..., description="Meter identifier")
    fuel_type: FuelType = Field(..., description="Fuel type")

    # Flow values
    flow_rate: float = Field(..., ge=0, description="Instantaneous flow rate")
    flow_rate_unit: str = Field(..., description="Flow rate unit per time")
    totalizer: float = Field(default=0, ge=0, description="Totalizer value")
    totalizer_unit: FlowUnit = Field(..., description="Totalizer unit")

    # Corrected values (for gas)
    corrected_flow_rate: Optional[float] = Field(
        default=None,
        ge=0,
        description="Temperature/pressure corrected flow"
    )
    correction_factor: Optional[float] = Field(default=None)

    # Operating conditions
    temperature_f: Optional[float] = Field(default=None, description="Temperature")
    pressure_psig: Optional[float] = Field(default=None, description="Pressure")

    # Quality
    is_valid: bool = Field(default=True, description="Reading validity")
    quality_code: str = Field(default="OK", description="Quality code")


class GasComposition(BaseModel):
    """Natural gas composition from GC analysis."""

    model_config = ConfigDict(frozen=True)

    analysis_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Analysis identifier"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Analysis timestamp"
    )
    method: CompositionMethod = Field(..., description="Analysis method")

    # Components (mole percent)
    methane: float = Field(default=0, ge=0, le=100, description="CH4 %")
    ethane: float = Field(default=0, ge=0, le=100, description="C2H6 %")
    propane: float = Field(default=0, ge=0, le=100, description="C3H8 %")
    n_butane: float = Field(default=0, ge=0, le=100, description="n-C4H10 %")
    i_butane: float = Field(default=0, ge=0, le=100, description="i-C4H10 %")
    n_pentane: float = Field(default=0, ge=0, le=100, description="n-C5H12 %")
    i_pentane: float = Field(default=0, ge=0, le=100, description="i-C5H12 %")
    hexanes_plus: float = Field(default=0, ge=0, le=100, description="C6+ %")
    nitrogen: float = Field(default=0, ge=0, le=100, description="N2 %")
    carbon_dioxide: float = Field(default=0, ge=0, le=100, description="CO2 %")
    oxygen: float = Field(default=0, ge=0, le=100, description="O2 %")
    hydrogen_sulfide: float = Field(default=0, ge=0, le=100, description="H2S ppm")
    water: float = Field(default=0, ge=0, le=100, description="H2O %")

    # Calculated properties
    gross_heating_value_btu_scf: Optional[float] = Field(
        default=None,
        ge=0,
        description="Gross heating value (BTU/scf)"
    )
    specific_gravity: Optional[float] = Field(
        default=None,
        ge=0,
        description="Specific gravity"
    )
    wobbe_index: Optional[float] = Field(default=None, ge=0)
    compressibility_factor: Optional[float] = Field(default=None, ge=0.8, le=1.2)

    @property
    def total_percent(self) -> float:
        """Calculate total composition percentage."""
        return (
            self.methane + self.ethane + self.propane +
            self.n_butane + self.i_butane + self.n_pentane +
            self.i_pentane + self.hexanes_plus + self.nitrogen +
            self.carbon_dioxide + self.oxygen + self.water
        )


class FuelAnalysis(BaseModel):
    """Fuel analysis for liquid/solid fuels."""

    model_config = ConfigDict(frozen=True)

    analysis_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Analysis identifier"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Analysis timestamp"
    )
    fuel_type: FuelType = Field(..., description="Fuel type")
    sample_date: date = Field(..., description="Sample collection date")
    lab_name: Optional[str] = Field(default=None, description="Laboratory name")

    # Heating value
    gross_heating_value: float = Field(..., ge=0, description="GHV (BTU/lb)")
    net_heating_value: Optional[float] = Field(default=None, ge=0, description="NHV")
    heating_value_basis: str = Field(
        default="as_received",
        description="Basis (as_received/dry)"
    )

    # Proximate analysis (weight percent)
    moisture: Optional[float] = Field(default=None, ge=0, le=100)
    ash: Optional[float] = Field(default=None, ge=0, le=100)
    volatile_matter: Optional[float] = Field(default=None, ge=0, le=100)
    fixed_carbon: Optional[float] = Field(default=None, ge=0, le=100)

    # Ultimate analysis (weight percent)
    carbon: Optional[float] = Field(default=None, ge=0, le=100)
    hydrogen: Optional[float] = Field(default=None, ge=0, le=100)
    nitrogen: Optional[float] = Field(default=None, ge=0, le=100)
    sulfur: Optional[float] = Field(default=None, ge=0, le=100)
    oxygen_by_diff: Optional[float] = Field(default=None, ge=0, le=100)

    # Density (for liquids)
    api_gravity: Optional[float] = Field(default=None, ge=-10, le=80)
    specific_gravity: Optional[float] = Field(default=None, ge=0.5, le=2.0)


class HeatInput(BaseModel):
    """Heat input calculation result."""

    model_config = ConfigDict(frozen=True)

    calculation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Calculation identifier"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Calculation timestamp"
    )
    period_start: datetime = Field(..., description="Period start")
    period_end: datetime = Field(..., description="Period end")

    fuel_type: FuelType = Field(..., description="Fuel type")
    meter_id: str = Field(..., description="Meter identifier")

    # Fuel consumption
    fuel_quantity: Decimal = Field(..., ge=0, description="Fuel consumed")
    fuel_unit: FlowUnit = Field(..., description="Fuel unit")

    # Heating value used
    heating_value: float = Field(..., ge=0, description="Heating value")
    heating_value_unit: str = Field(..., description="HV unit")

    # Heat input result
    heat_input_mmbtu: Decimal = Field(..., ge=0, description="Heat input (MMBtu)")
    heat_input_rate_mmbtu_hr: Optional[Decimal] = Field(
        default=None,
        ge=0,
        description="Heat input rate"
    )

    # Data quality
    calculation_method: str = Field(..., description="Calculation method")
    data_quality_flag: Optional[str] = Field(default=None)


class FuelBlend(BaseModel):
    """Fuel blend composition for multi-fuel systems."""

    model_config = ConfigDict(frozen=True)

    blend_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Blend identifier"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Blend components
    components: Dict[FuelType, float] = Field(
        ...,
        description="Fuel type to percentage"
    )

    # Calculated properties
    weighted_heating_value: Optional[float] = Field(default=None, ge=0)
    weighted_carbon_factor: Optional[float] = Field(default=None, ge=0)
    weighted_sulfur_content: Optional[float] = Field(default=None, ge=0)

    @field_validator("components")
    @classmethod
    def validate_blend(cls, v: Dict[FuelType, float]) -> Dict[FuelType, float]:
        """Validate blend percentages sum to 100."""
        total = sum(v.values())
        if abs(total - 100.0) > 0.1:
            raise ValueError(f"Blend percentages must sum to 100, got {total}")
        return v


class MeterConfiguration(BaseModel):
    """Flow meter configuration."""

    model_config = ConfigDict(frozen=True)

    meter_id: str = Field(..., description="Meter identifier")
    meter_name: str = Field(..., description="Meter name")
    meter_type: MeterType = Field(..., description="Meter type")
    vendor: MeterVendor = Field(..., description="Meter vendor")
    model: str = Field(..., description="Meter model")
    serial_number: Optional[str] = Field(default=None)

    fuel_type: FuelType = Field(..., description="Primary fuel type")
    flow_unit: FlowUnit = Field(..., description="Measurement unit")

    # Ranges
    min_flow: float = Field(default=0, ge=0, description="Minimum flow")
    max_flow: float = Field(..., ge=0, description="Maximum flow")

    # Correction factors
    meter_factor: float = Field(default=1.0, ge=0.9, le=1.1, description="K-factor")
    base_temperature_f: float = Field(default=60.0, description="Base temperature")
    base_pressure_psia: float = Field(default=14.73, description="Base pressure")

    # Communication
    protocol: str = Field(default="modbus_tcp", description="Protocol")
    address: Optional[str] = Field(default=None, description="IP/serial address")
    port: int = Field(default=502, ge=1, le=65535)


class FuelFlowConnectorConfig(BaseConnectorConfig):
    """Configuration for fuel flow connector."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    connector_type: ConnectorType = Field(
        default=ConnectorType.FUEL_FLOW,
        description="Connector type"
    )

    # Meter configurations
    meters: List[MeterConfiguration] = Field(
        default_factory=list,
        description="Meter configurations"
    )

    # Gas chromatograph settings
    gc_enabled: bool = Field(default=False, description="GC integration enabled")
    gc_address: Optional[str] = Field(default=None)
    gc_port: int = Field(default=502)
    gc_sample_interval_minutes: int = Field(default=15, ge=1, le=1440)

    # Default heating values (for missing data)
    default_heating_values: Dict[str, float] = Field(
        default_factory=lambda: {
            "natural_gas": 1020.0,  # BTU/scf
            "diesel": 138000.0,  # BTU/gallon
            "fuel_oil_no2": 140000.0,
            "fuel_oil_no6": 150000.0,
            "coal_bituminous": 12000.0,  # BTU/lb
        },
        description="Default heating values by fuel type"
    )

    # Polling settings
    polling_interval_seconds: float = Field(
        default=5.0,
        ge=1.0,
        le=300.0,
        description="Polling interval"
    )

    # Totalization settings
    totalization_period_minutes: int = Field(
        default=60,
        ge=1,
        le=1440,
        description="Totalization period"
    )


# =============================================================================
# Heat Input Calculator
# =============================================================================


class HeatInputCalculator:
    """
    Calculates heat input from fuel flow and composition data.

    Implements EPA 40 CFR Part 75 Appendix D and F methodologies.
    """

    # Default heating values by fuel type (BTU per unit)
    DEFAULT_HEATING_VALUES = {
        FuelType.NATURAL_GAS: 1020.0,  # BTU/scf
        FuelType.DIESEL: 138000.0,  # BTU/gallon
        FuelType.FUEL_OIL_NO2: 140000.0,  # BTU/gallon
        FuelType.FUEL_OIL_NO4: 144000.0,  # BTU/gallon
        FuelType.FUEL_OIL_NO6: 150000.0,  # BTU/gallon
        FuelType.RESIDUAL_OIL: 150000.0,
        FuelType.COAL_BITUMINOUS: 12000.0,  # BTU/lb
        FuelType.COAL_SUBBITUMINOUS: 9500.0,
        FuelType.COAL_LIGNITE: 6500.0,
        FuelType.COAL_ANTHRACITE: 13000.0,
        FuelType.PETROLEUM_COKE: 14500.0,
        FuelType.LPG: 91500.0,  # BTU/gallon
        FuelType.PROPANE: 91500.0,
    }

    # Gas component heating values (BTU/scf at 60F, 14.73 psia)
    GAS_COMPONENT_HV = {
        "methane": 1010.0,
        "ethane": 1769.6,
        "propane": 2516.1,
        "n_butane": 3262.2,
        "i_butane": 3251.9,
        "n_pentane": 4008.9,
        "i_pentane": 4000.8,
        "hexanes_plus": 4755.0,  # Average C6+
    }

    def __init__(self) -> None:
        """Initialize calculator."""
        self._logger = logging.getLogger("fuel_flow.calculator")

    def calculate_gas_heating_value(
        self,
        composition: GasComposition,
    ) -> float:
        """
        Calculate gross heating value from gas composition.

        Uses AGA 8 / GPA 2145 methodology.

        Args:
            composition: Gas composition analysis

        Returns:
            Gross heating value in BTU/scf
        """
        # Sum weighted component heating values
        ghv = 0.0

        ghv += composition.methane / 100.0 * self.GAS_COMPONENT_HV["methane"]
        ghv += composition.ethane / 100.0 * self.GAS_COMPONENT_HV["ethane"]
        ghv += composition.propane / 100.0 * self.GAS_COMPONENT_HV["propane"]
        ghv += composition.n_butane / 100.0 * self.GAS_COMPONENT_HV["n_butane"]
        ghv += composition.i_butane / 100.0 * self.GAS_COMPONENT_HV["i_butane"]
        ghv += composition.n_pentane / 100.0 * self.GAS_COMPONENT_HV["n_pentane"]
        ghv += composition.i_pentane / 100.0 * self.GAS_COMPONENT_HV["i_pentane"]
        ghv += composition.hexanes_plus / 100.0 * self.GAS_COMPONENT_HV["hexanes_plus"]

        # Inerts (N2, CO2, O2) have zero heating value

        self._logger.debug(f"Calculated GHV: {ghv:.1f} BTU/scf")
        return ghv

    def calculate_heat_input(
        self,
        fuel_quantity: Decimal,
        fuel_unit: FlowUnit,
        fuel_type: FuelType,
        heating_value: Optional[float] = None,
        period_hours: float = 1.0,
    ) -> HeatInput:
        """
        Calculate heat input from fuel consumption.

        Args:
            fuel_quantity: Fuel consumed
            fuel_unit: Unit of fuel
            fuel_type: Type of fuel
            heating_value: Optional custom heating value
            period_hours: Period duration in hours

        Returns:
            Heat input calculation
        """
        # Get heating value
        if heating_value is None:
            heating_value = self.DEFAULT_HEATING_VALUES.get(fuel_type, 1000.0)

        # Convert to common basis
        # For gas: BTU = volume (scf) * HV (BTU/scf)
        # For liquid: BTU = volume (gallon) * HV (BTU/gallon)
        # For solid: BTU = mass (lb) * HV (BTU/lb)

        fuel_qty = float(fuel_quantity)

        # Unit conversions
        if fuel_unit in [FlowUnit.SCF, FlowUnit.NM3]:
            heat_btu = fuel_qty * heating_value
        elif fuel_unit == FlowUnit.MCF:
            heat_btu = fuel_qty * 1000 * heating_value
        elif fuel_unit == FlowUnit.MMSCF:
            heat_btu = fuel_qty * 1000000 * heating_value
        elif fuel_unit in [FlowUnit.GALLON, FlowUnit.BARREL]:
            multiplier = 1.0 if fuel_unit == FlowUnit.GALLON else 42.0
            heat_btu = fuel_qty * multiplier * heating_value
        elif fuel_unit in [FlowUnit.LB, FlowUnit.TON, FlowUnit.TONNE]:
            if fuel_unit == FlowUnit.TON:
                fuel_qty *= 2000  # Short ton to pounds
            elif fuel_unit == FlowUnit.TONNE:
                fuel_qty *= 2204.62  # Metric ton to pounds
            heat_btu = fuel_qty * heating_value
        else:
            heat_btu = fuel_qty * heating_value

        # Convert to MMBtu
        heat_mmbtu = Decimal(str(heat_btu / 1_000_000))

        # Calculate rate
        heat_rate = None
        if period_hours > 0:
            heat_rate = heat_mmbtu / Decimal(str(period_hours))

        return HeatInput(
            period_start=datetime.utcnow() - timedelta(hours=period_hours),
            period_end=datetime.utcnow(),
            fuel_type=fuel_type,
            meter_id="calculated",
            fuel_quantity=fuel_quantity,
            fuel_unit=fuel_unit,
            heating_value=heating_value,
            heating_value_unit="BTU/unit",
            heat_input_mmbtu=heat_mmbtu.quantize(Decimal("0.001")),
            heat_input_rate_mmbtu_hr=heat_rate.quantize(Decimal("0.001")) if heat_rate else None,
            calculation_method="standard",
        )

    def calculate_blend_heat_input(
        self,
        fuel_quantities: Dict[FuelType, Decimal],
        fuel_unit: FlowUnit,
        heating_values: Optional[Dict[FuelType, float]] = None,
    ) -> Tuple[Decimal, Dict[FuelType, Decimal]]:
        """
        Calculate heat input for fuel blend.

        Args:
            fuel_quantities: Quantity by fuel type
            fuel_unit: Unit of fuel quantities
            heating_values: Optional custom heating values

        Returns:
            Tuple of (total_mmbtu, heat_by_fuel_type)
        """
        heat_by_fuel: Dict[FuelType, Decimal] = {}
        total_heat = Decimal("0")

        for fuel_type, quantity in fuel_quantities.items():
            hv = (
                heating_values.get(fuel_type)
                if heating_values
                else self.DEFAULT_HEATING_VALUES.get(fuel_type, 1000.0)
            )

            result = self.calculate_heat_input(
                fuel_quantity=quantity,
                fuel_unit=fuel_unit,
                fuel_type=fuel_type,
                heating_value=hv,
            )

            heat_by_fuel[fuel_type] = result.heat_input_mmbtu
            total_heat += result.heat_input_mmbtu

        return total_heat, heat_by_fuel


# =============================================================================
# Fuel Flow Protocol Handler
# =============================================================================


class FuelFlowProtocolHandler:
    """
    Protocol handler for fuel flow meter communication.
    """

    def __init__(self, meter_config: MeterConfiguration) -> None:
        """Initialize handler."""
        self._config = meter_config
        self._connected = False
        self._logger = logging.getLogger(f"fuel_flow.handler.{meter_config.meter_id}")

    async def connect(self) -> None:
        """Establish connection to meter."""
        self._logger.info(f"Connecting to meter {self._config.meter_name}")
        # In production, establish actual connection
        self._connected = True

    async def disconnect(self) -> None:
        """Close connection."""
        self._connected = False
        self._logger.info("Disconnected from meter")

    async def read_flow_rate(self) -> float:
        """Read instantaneous flow rate."""
        if not self._connected:
            raise ConnectionError("Not connected to meter")
        # Simulated read
        return 0.0

    async def read_totalizer(self) -> float:
        """Read totalizer value."""
        if not self._connected:
            raise ConnectionError("Not connected to meter")
        return 0.0

    async def read_temperature(self) -> Optional[float]:
        """Read temperature if available."""
        return None

    async def read_pressure(self) -> Optional[float]:
        """Read pressure if available."""
        return None

    async def read_all(self) -> Dict[str, Any]:
        """Read all meter values."""
        return {
            "flow_rate": await self.read_flow_rate(),
            "totalizer": await self.read_totalizer(),
            "temperature": await self.read_temperature(),
            "pressure": await self.read_pressure(),
        }


# =============================================================================
# Gas Chromatograph Handler
# =============================================================================


class GasChromatographHandler:
    """
    Handler for gas chromatograph communication.

    Reads natural gas composition data for heating value calculation.
    """

    def __init__(self, address: str, port: int) -> None:
        """Initialize GC handler."""
        self._address = address
        self._port = port
        self._connected = False
        self._logger = logging.getLogger("fuel_flow.gc")

    async def connect(self) -> None:
        """Connect to GC."""
        self._logger.info(f"Connecting to GC at {self._address}:{self._port}")
        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect from GC."""
        self._connected = False

    async def read_composition(self) -> GasComposition:
        """
        Read current gas composition.

        Returns:
            Gas composition analysis
        """
        if not self._connected:
            raise ConnectionError("Not connected to GC")

        # Simulated typical natural gas composition
        return GasComposition(
            method=CompositionMethod.GAS_CHROMATOGRAPHY,
            methane=94.5,
            ethane=2.5,
            propane=0.8,
            n_butane=0.2,
            i_butane=0.1,
            n_pentane=0.05,
            i_pentane=0.05,
            hexanes_plus=0.1,
            nitrogen=1.2,
            carbon_dioxide=0.5,
            oxygen=0.0,
            water=0.0,
        )


# =============================================================================
# Fuel Flow Connector
# =============================================================================


class FuelFlowConnector(BaseConnector):
    """
    Fuel Flow Metering Connector.

    Provides comprehensive integration for fuel metering:
    - Natural gas flow meters (ultrasonic, turbine, orifice)
    - Oil flow meters (Coriolis, positive displacement)
    - Coal feeders and belt scales
    - Gas chromatograph for composition analysis
    - Heat input calculations per EPA Part 75

    Features:
    - Multi-meter support
    - Fuel composition tracking
    - Heat input calculations
    - Fuel blend tracking
    - Data quality validation
    """

    def __init__(self, config: FuelFlowConnectorConfig) -> None:
        """
        Initialize fuel flow connector.

        Args:
            config: Connector configuration
        """
        super().__init__(config)
        self._fuel_config = config

        # Initialize meter handlers
        self._meter_handlers: Dict[str, FuelFlowProtocolHandler] = {}
        for meter_config in config.meters:
            self._meter_handlers[meter_config.meter_id] = FuelFlowProtocolHandler(
                meter_config
            )

        # Initialize GC handler if enabled
        self._gc_handler: Optional[GasChromatographHandler] = None
        if config.gc_enabled and config.gc_address:
            self._gc_handler = GasChromatographHandler(
                config.gc_address,
                config.gc_port,
            )

        # Initialize calculator
        self._heat_calculator = HeatInputCalculator()

        # Current data
        self._current_readings: Dict[str, FlowReading] = {}
        self._current_composition: Optional[GasComposition] = None
        self._totalizer_snapshot: Dict[str, Tuple[datetime, float]] = {}

        # Polling
        self._polling_task: Optional[asyncio.Task] = None
        self._gc_polling_task: Optional[asyncio.Task] = None
        self._polling_active = False

        # Callbacks
        self._data_callbacks: List[Callable[[Dict[str, FlowReading]], None]] = []

        self._logger = logging.getLogger(f"fuel_flow.connector")

    # -------------------------------------------------------------------------
    # Abstract Method Implementations
    # -------------------------------------------------------------------------

    async def connect(self) -> None:
        """
        Establish connections to all meters.

        Raises:
            ConnectionError: If connection fails
        """
        self._state = ConnectionState.CONNECTING
        self._logger.info("Connecting to fuel flow meters")

        try:
            # Connect to meters
            for meter_id, handler in self._meter_handlers.items():
                await handler.connect()

            # Connect to GC if enabled
            if self._gc_handler:
                await self._gc_handler.connect()

            self._state = ConnectionState.CONNECTED
            self._logger.info(f"Connected to {len(self._meter_handlers)} meters")

            await self._audit_logger.log_operation(
                operation="connect",
                status="success",
                response_summary=f"Connected to {len(self._meter_handlers)} meters",
            )

        except Exception as e:
            self._state = ConnectionState.ERROR
            raise ConnectionError(f"Failed to connect to meters: {e}")

    async def disconnect(self) -> None:
        """Disconnect from all meters."""
        self._logger.info("Disconnecting from fuel flow meters")

        await self.stop_polling()

        for handler in self._meter_handlers.values():
            await handler.disconnect()

        if self._gc_handler:
            await self._gc_handler.disconnect()

        self._state = ConnectionState.DISCONNECTED

        await self._audit_logger.log_operation(
            operation="disconnect",
            status="success",
        )

    async def health_check(self) -> HealthCheckResult:
        """
        Perform health check on meter connections.

        Returns:
            Health check result
        """
        start_time = time.time()

        try:
            # Check each meter
            meter_status = {}
            for meter_id, handler in self._meter_handlers.items():
                try:
                    await handler.read_flow_rate()
                    meter_status[meter_id] = "OK"
                except Exception as e:
                    meter_status[meter_id] = f"ERROR: {e}"

            latency_ms = (time.time() - start_time) * 1000

            # Determine overall status
            failed = sum(1 for s in meter_status.values() if s != "OK")
            if failed == 0:
                status = HealthStatus.HEALTHY
                message = "All meters healthy"
            elif failed < len(meter_status):
                status = HealthStatus.DEGRADED
                message = f"{failed} meters have issues"
            else:
                status = HealthStatus.UNHEALTHY
                message = "All meters unavailable"

            return HealthCheckResult(
                status=status,
                latency_ms=latency_ms,
                message=message,
                details={"meter_status": meter_status},
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency_ms,
                message=f"Health check failed: {e}",
            )

    async def validate_configuration(self) -> bool:
        """
        Validate fuel flow configuration.

        Returns:
            True if configuration is valid
        """
        issues: List[str] = []

        if not self._fuel_config.meters:
            issues.append("At least one meter must be configured")

        for meter in self._fuel_config.meters:
            if meter.max_flow <= meter.min_flow:
                issues.append(f"Meter {meter.meter_id}: max_flow must be > min_flow")

        if self._fuel_config.gc_enabled and not self._fuel_config.gc_address:
            issues.append("GC address required when GC is enabled")

        if issues:
            raise ConfigurationError(
                f"Invalid fuel flow configuration: {issues}",
                connector_id=self._config.connector_id,
            )

        return True

    # -------------------------------------------------------------------------
    # Fuel Flow-Specific Methods
    # -------------------------------------------------------------------------

    async def read_all_meters(self) -> Dict[str, FlowReading]:
        """
        Read all configured meters.

        Returns:
            Dictionary of meter ID to reading
        """
        start_time = time.time()
        readings: Dict[str, FlowReading] = {}

        for meter_id, handler in self._meter_handlers.items():
            try:
                meter_config = next(
                    m for m in self._fuel_config.meters
                    if m.meter_id == meter_id
                )

                data = await handler.read_all()

                reading = FlowReading(
                    meter_id=meter_id,
                    fuel_type=meter_config.fuel_type,
                    flow_rate=data["flow_rate"],
                    flow_rate_unit=f"{meter_config.flow_unit.value}/hr",
                    totalizer=data["totalizer"],
                    totalizer_unit=meter_config.flow_unit,
                    temperature_f=data.get("temperature"),
                    pressure_psig=data.get("pressure"),
                )

                readings[meter_id] = reading
                self._current_readings[meter_id] = reading

            except Exception as e:
                self._logger.error(f"Failed to read meter {meter_id}: {e}")

        duration_ms = (time.time() - start_time) * 1000
        await self._metrics.record_request(
            success=True,
            latency_ms=duration_ms,
        )

        return readings

    async def read_meter(self, meter_id: str) -> FlowReading:
        """
        Read a specific meter.

        Args:
            meter_id: Meter identifier

        Returns:
            Flow reading
        """
        handler = self._meter_handlers.get(meter_id)
        if not handler:
            raise ValidationError(f"Unknown meter: {meter_id}")

        meter_config = next(
            m for m in self._fuel_config.meters
            if m.meter_id == meter_id
        )

        data = await handler.read_all()

        return FlowReading(
            meter_id=meter_id,
            fuel_type=meter_config.fuel_type,
            flow_rate=data["flow_rate"],
            flow_rate_unit=f"{meter_config.flow_unit.value}/hr",
            totalizer=data["totalizer"],
            totalizer_unit=meter_config.flow_unit,
            temperature_f=data.get("temperature"),
            pressure_psig=data.get("pressure"),
        )

    async def get_gas_composition(self) -> GasComposition:
        """
        Get current gas composition from GC.

        Returns:
            Gas composition
        """
        if not self._gc_handler:
            raise ConfigurationError("GC not configured")

        composition = await self._gc_handler.read_composition()

        # Calculate heating value
        ghv = self._heat_calculator.calculate_gas_heating_value(composition)

        # Create new composition with calculated value
        self._current_composition = GasComposition(
            analysis_id=composition.analysis_id,
            timestamp=composition.timestamp,
            method=composition.method,
            methane=composition.methane,
            ethane=composition.ethane,
            propane=composition.propane,
            n_butane=composition.n_butane,
            i_butane=composition.i_butane,
            n_pentane=composition.n_pentane,
            i_pentane=composition.i_pentane,
            hexanes_plus=composition.hexanes_plus,
            nitrogen=composition.nitrogen,
            carbon_dioxide=composition.carbon_dioxide,
            oxygen=composition.oxygen,
            water=composition.water,
            gross_heating_value_btu_scf=ghv,
        )

        await self._audit_logger.log_operation(
            operation="get_gas_composition",
            status="success",
            response_summary=f"GHV: {ghv:.1f} BTU/scf",
        )

        return self._current_composition

    async def calculate_heat_input(
        self,
        meter_id: str,
        period_minutes: Optional[int] = None,
    ) -> HeatInput:
        """
        Calculate heat input for a meter.

        Args:
            meter_id: Meter identifier
            period_minutes: Calculation period (default: totalization period)

        Returns:
            Heat input calculation
        """
        period = period_minutes or self._fuel_config.totalization_period_minutes

        # Get current and previous totalizer
        reading = await self.read_meter(meter_id)
        current_totalizer = reading.totalizer

        meter_config = next(
            m for m in self._fuel_config.meters
            if m.meter_id == meter_id
        )

        # Get snapshot or use current as baseline
        prev_time, prev_totalizer = self._totalizer_snapshot.get(
            meter_id,
            (datetime.utcnow() - timedelta(minutes=period), current_totalizer),
        )

        # Calculate consumption
        fuel_consumed = Decimal(str(current_totalizer - prev_totalizer))

        # Get heating value
        heating_value = None
        if meter_config.fuel_type == FuelType.NATURAL_GAS and self._current_composition:
            heating_value = self._current_composition.gross_heating_value_btu_scf

        # Calculate heat input
        period_hours = period / 60.0
        heat_input = self._heat_calculator.calculate_heat_input(
            fuel_quantity=fuel_consumed,
            fuel_unit=meter_config.flow_unit,
            fuel_type=meter_config.fuel_type,
            heating_value=heating_value,
            period_hours=period_hours,
        )

        # Update snapshot
        self._totalizer_snapshot[meter_id] = (datetime.utcnow(), current_totalizer)

        await self._audit_logger.log_operation(
            operation="calculate_heat_input",
            status="success",
            request_data={"meter_id": meter_id, "period_minutes": period},
            response_summary=f"Heat input: {heat_input.heat_input_mmbtu} MMBtu",
        )

        return heat_input

    async def calculate_total_heat_input(
        self,
        period_minutes: Optional[int] = None,
    ) -> Tuple[HeatInput, Dict[str, HeatInput]]:
        """
        Calculate total heat input from all meters.

        Args:
            period_minutes: Calculation period

        Returns:
            Tuple of (total_heat_input, heat_by_meter)
        """
        heat_by_meter: Dict[str, HeatInput] = {}
        total_mmbtu = Decimal("0")

        for meter_id in self._meter_handlers:
            try:
                heat_input = await self.calculate_heat_input(
                    meter_id,
                    period_minutes,
                )
                heat_by_meter[meter_id] = heat_input
                total_mmbtu += heat_input.heat_input_mmbtu
            except Exception as e:
                self._logger.error(f"Failed to calculate heat for {meter_id}: {e}")

        # Create total heat input
        period = period_minutes or self._fuel_config.totalization_period_minutes
        total_heat = HeatInput(
            period_start=datetime.utcnow() - timedelta(minutes=period),
            period_end=datetime.utcnow(),
            fuel_type=FuelType.NATURAL_GAS,  # Primary fuel
            meter_id="total",
            fuel_quantity=Decimal("0"),
            fuel_unit=FlowUnit.SCF,
            heating_value=0,
            heating_value_unit="combined",
            heat_input_mmbtu=total_mmbtu.quantize(Decimal("0.001")),
            heat_input_rate_mmbtu_hr=(
                total_mmbtu / Decimal(str(period / 60))
            ).quantize(Decimal("0.001")),
            calculation_method="summed",
        )

        return total_heat, heat_by_meter

    async def record_fuel_analysis(
        self,
        analysis: FuelAnalysis,
    ) -> None:
        """
        Record fuel analysis results.

        Args:
            analysis: Fuel analysis data
        """
        await self._audit_logger.log_operation(
            operation="record_fuel_analysis",
            status="success",
            request_data={
                "fuel_type": analysis.fuel_type.value,
                "ghv": analysis.gross_heating_value,
                "sulfur": analysis.sulfur,
            },
        )

    async def start_polling(
        self,
        callback: Optional[Callable[[Dict[str, FlowReading]], None]] = None,
    ) -> None:
        """
        Start continuous data polling.

        Args:
            callback: Optional callback for new data
        """
        if self._polling_active:
            return

        if callback:
            self._data_callbacks.append(callback)

        self._polling_active = True
        self._polling_task = asyncio.create_task(self._polling_loop())

        if self._gc_handler:
            self._gc_polling_task = asyncio.create_task(self._gc_polling_loop())

        self._logger.info("Started polling")

    async def stop_polling(self) -> None:
        """Stop data polling."""
        self._polling_active = False

        if self._polling_task:
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass

        if self._gc_polling_task:
            self._gc_polling_task.cancel()
            try:
                await self._gc_polling_task
            except asyncio.CancelledError:
                pass

        self._logger.info("Stopped polling")

    async def _polling_loop(self) -> None:
        """Background meter polling loop."""
        while self._polling_active:
            try:
                readings = await self.read_all_meters()

                for callback in self._data_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(readings)
                        else:
                            callback(readings)
                    except Exception as e:
                        self._logger.error(f"Callback error: {e}")

                await asyncio.sleep(self._fuel_config.polling_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Polling error: {e}")
                await asyncio.sleep(self._fuel_config.polling_interval_seconds)

    async def _gc_polling_loop(self) -> None:
        """Background GC polling loop."""
        interval = self._fuel_config.gc_sample_interval_minutes * 60

        while self._polling_active:
            try:
                await self.get_gas_composition()
                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"GC polling error: {e}")
                await asyncio.sleep(interval)


# =============================================================================
# Factory Function
# =============================================================================


def create_fuel_flow_connector(
    meters: List[MeterConfiguration],
    gc_enabled: bool = False,
    gc_address: Optional[str] = None,
    **kwargs: Any,
) -> FuelFlowConnector:
    """
    Factory function to create fuel flow connector.

    Args:
        meters: Meter configurations
        gc_enabled: Enable GC integration
        gc_address: GC address
        **kwargs: Additional configuration

    Returns:
        Configured fuel flow connector
    """
    config = FuelFlowConnectorConfig(
        connector_name="FuelFlow",
        meters=meters,
        gc_enabled=gc_enabled,
        gc_address=gc_address,
        **kwargs,
    )

    return FuelFlowConnector(config)
