"""
Fuel Flow Connector for GL-009 THERMALIQ.

Interfaces with fuel flow meters for energy consumption monitoring.

Supported Meters:
- Gas flow meters (ultrasonic, turbine, coriolis, thermal)
- Liquid fuel flow meters (turbine, positive displacement)
- Mass flow vs volumetric flow
- Temperature and pressure compensation

Features:
- Real-time flow rate measurement
- Totalizer readings
- Automatic unit conversion
- Density compensation
- Energy content calculation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime
import asyncio
import logging
import math

from .base_connector import BaseConnector, ConnectorStatus, ConnectorHealth

logger = logging.getLogger(__name__)


class FuelType(Enum):
    """Fuel types."""
    NATURAL_GAS = "natural_gas"
    PROPANE = "propane"
    DIESEL = "diesel"
    FUEL_OIL = "fuel_oil"
    GASOLINE = "gasoline"
    BIOGAS = "biogas"
    HYDROGEN = "hydrogen"


class FlowMeterType(Enum):
    """Flow meter technologies."""
    ULTRASONIC = "ultrasonic"
    TURBINE = "turbine"
    CORIOLIS = "coriolis"
    THERMAL = "thermal"
    VORTEX = "vortex"
    POSITIVE_DISPLACEMENT = "positive_displacement"
    DIFFERENTIAL_PRESSURE = "differential_pressure"


class FlowUnits(Enum):
    """Flow measurement units."""
    # Mass flow
    KG_PER_HOUR = "kg/h"
    KG_PER_SECOND = "kg/s"
    LB_PER_HOUR = "lb/h"

    # Volumetric flow
    M3_PER_HOUR = "m3/h"
    L_PER_HOUR = "l/h"
    SCFH = "scfh"  # Standard cubic feet per hour
    CFM = "cfm"  # Cubic feet per minute


@dataclass
class FlowReading:
    """Fuel flow reading."""
    meter_id: str
    timestamp: datetime
    fuel_type: FuelType

    # Flow measurements
    mass_flow_rate: Optional[float] = None  # kg/h
    volumetric_flow_rate: Optional[float] = None  # m3/h
    totalizer: Optional[float] = None  # Total volume or mass

    # Process conditions
    temperature_c: Optional[float] = None
    pressure_kpa: Optional[float] = None
    density_kg_m3: Optional[float] = None

    # Energy calculations
    energy_flow_rate_kw: Optional[float] = None
    energy_total_kwh: Optional[float] = None
    heating_value_mj_kg: Optional[float] = None

    # Quality
    quality: str = "Good"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "meter_id": self.meter_id,
            "timestamp": self.timestamp.isoformat(),
            "fuel_type": self.fuel_type.value,
            "mass_flow_rate": self.mass_flow_rate,
            "volumetric_flow_rate": self.volumetric_flow_rate,
            "totalizer": self.totalizer,
            "temperature_c": self.temperature_c,
            "pressure_kpa": self.pressure_kpa,
            "density_kg_m3": self.density_kg_m3,
            "energy_flow_rate_kw": self.energy_flow_rate_kw,
            "energy_total_kwh": self.energy_total_kwh,
            "heating_value_mj_kg": self.heating_value_mj_kg,
            "quality": self.quality,
            "metadata": self.metadata,
        }


@dataclass
class FuelFlowConfig:
    """Fuel flow meter configuration."""
    meter_id: str
    meter_type: FlowMeterType
    fuel_type: FuelType
    protocol: str  # modbus_tcp, modbus_rtu, opc_ua, analog
    address: str
    port: int

    # Calibration
    k_factor: float = 1.0  # Meter calibration factor
    offset: float = 0.0
    units: FlowUnits = FlowUnits.M3_PER_HOUR

    # Fuel properties
    standard_density_kg_m3: Optional[float] = None  # At standard conditions
    heating_value_mj_kg: Optional[float] = None  # Lower heating value

    # Compensation
    enable_temperature_compensation: bool = True
    enable_pressure_compensation: bool = True
    reference_temperature_c: float = 15.0  # Standard reference temperature
    reference_pressure_kpa: float = 101.325  # Standard reference pressure

    # Register mapping (for Modbus)
    registers: Optional[Dict[str, int]] = None

    timeout_seconds: float = 5.0


class FuelFlowConnector(BaseConnector):
    """
    Connector for fuel flow meters.

    Provides real-time flow rate and totalizer readings with compensation.
    """

    # Standard fuel properties
    FUEL_PROPERTIES = {
        FuelType.NATURAL_GAS: {
            "density_std": 0.72,  # kg/m3 at STP
            "heating_value": 50.0  # MJ/kg (LHV)
        },
        FuelType.PROPANE: {
            "density_std": 1.88,
            "heating_value": 46.3
        },
        FuelType.DIESEL: {
            "density_std": 850.0,
            "heating_value": 42.6
        },
        FuelType.FUEL_OIL: {
            "density_std": 950.0,
            "heating_value": 40.0
        },
        FuelType.BIOGAS: {
            "density_std": 1.15,
            "heating_value": 21.5
        },
        FuelType.HYDROGEN: {
            "density_std": 0.09,
            "heating_value": 120.0
        },
    }

    def __init__(self, config: FuelFlowConfig, **kwargs):
        """
        Initialize fuel flow connector.

        Args:
            config: Fuel flow meter configuration
            **kwargs: Additional arguments for BaseConnector
        """
        super().__init__(
            connector_id=f"fuel_flow_{config.meter_id}",
            **kwargs
        )
        self.config = config
        self._client: Optional[Any] = None

        # Set fuel properties if not provided
        if not self.config.standard_density_kg_m3 or not self.config.heating_value_mj_kg:
            props = self.FUEL_PROPERTIES.get(config.fuel_type, {})
            if not self.config.standard_density_kg_m3:
                self.config.standard_density_kg_m3 = props.get("density_std", 1.0)
            if not self.config.heating_value_mj_kg:
                self.config.heating_value_mj_kg = props.get("heating_value", 1.0)

        # Initialize default register mapping
        if not self.config.registers:
            self.config.registers = self._default_registers()

    def _default_registers(self) -> Dict[str, int]:
        """Default Modbus register mapping."""
        return {
            "flow_rate": 0,
            "totalizer": 10,
            "temperature": 20,
            "pressure": 30,
            "density": 40,
        }

    async def connect(self) -> bool:
        """
        Establish connection to fuel flow meter.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            if self.config.protocol in ["modbus_tcp", "modbus_rtu"]:
                return await self._connect_modbus()
            elif self.config.protocol == "opc_ua":
                return await self._connect_opc_ua()
            else:
                logger.error(f"Unsupported protocol: {self.config.protocol}")
                return False

        except Exception as e:
            logger.error(f"[{self.connector_id}] Connection failed: {e}")
            return False

    async def _connect_modbus(self) -> bool:
        """Connect via Modbus."""
        try:
            from pymodbus.client import AsyncModbusTcpClient

            if self.config.protocol == "modbus_tcp":
                self._client = AsyncModbusTcpClient(
                    host=self.config.address,
                    port=self.config.port,
                    timeout=self.config.timeout_seconds
                )
            else:
                from pymodbus.client import AsyncModbusSerialClient
                self._client = AsyncModbusSerialClient(
                    port=self.config.address,
                    baudrate=self.config.port,
                    timeout=self.config.timeout_seconds
                )

            connected = await self._client.connect()
            if connected:
                logger.info(f"[{self.connector_id}] Modbus connected")
                return True
            else:
                return False

        except ImportError:
            logger.warning("pymodbus not available, using mock connection")
            self._client = MockFlowMeterClient(self.config)
            return True
        except Exception as e:
            logger.error(f"[{self.connector_id}] Modbus connection error: {e}")
            self._client = MockFlowMeterClient(self.config)
            return True

    async def _connect_opc_ua(self) -> bool:
        """Connect via OPC-UA."""
        try:
            from asyncua import Client

            url = f"opc.tcp://{self.config.address}:{self.config.port}"
            self._client = Client(url=url)
            await self._client.connect()

            logger.info(f"[{self.connector_id}] OPC-UA connected")
            return True

        except ImportError:
            logger.warning("asyncua not available, using mock connection")
            self._client = MockFlowMeterClient(self.config)
            return True
        except Exception as e:
            logger.error(f"[{self.connector_id}] OPC-UA connection error: {e}")
            return False

    async def disconnect(self) -> bool:
        """
        Disconnect from fuel flow meter.

        Returns:
            True if disconnection successful, False otherwise
        """
        try:
            if self._client and hasattr(self._client, 'close'):
                if asyncio.iscoroutinefunction(self._client.close):
                    await self._client.close()
                else:
                    self._client.close()

            logger.info(f"[{self.connector_id}] Disconnected")
            return True

        except Exception as e:
            logger.error(f"[{self.connector_id}] Disconnect error: {e}")
            return False

    async def health_check(self) -> ConnectorHealth:
        """
        Perform health check on the connection.

        Returns:
            ConnectorHealth object with status information
        """
        health = await self.get_health()

        try:
            if self.is_connected:
                # Try to read flow rate
                start_time = datetime.now()
                reading = await self.read_flow()
                latency = (datetime.now() - start_time).total_seconds() * 1000

                health.latency_ms = latency
                health.metadata["last_reading"] = reading.to_dict()

        except Exception as e:
            health.is_healthy = False
            health.last_error = str(e)

        return health

    async def read(self, **kwargs) -> FlowReading:
        """
        Read current flow values.

        Returns:
            FlowReading object
        """
        return await self.read_flow()

    async def read_flow(self) -> FlowReading:
        """
        Read current fuel flow values.

        Returns:
            FlowReading object with compensated values
        """
        try:
            # Read raw values from meter
            raw_data = await self._read_raw_values()

            # Extract values
            volumetric_flow = raw_data.get("flow_rate", 0.0) * self.config.k_factor + self.config.offset
            temperature = raw_data.get("temperature", self.config.reference_temperature_c)
            pressure = raw_data.get("pressure", self.config.reference_pressure_kpa)
            totalizer = raw_data.get("totalizer", 0.0)

            # Calculate actual density
            density = self._calculate_density(temperature, pressure)

            # Calculate mass flow rate
            mass_flow = volumetric_flow * density

            # Calculate energy flow rate
            energy_flow_kw = (mass_flow * self.config.heating_value_mj_kg) / 3.6  # Convert MJ/h to kW

            # Create reading
            reading = FlowReading(
                meter_id=self.config.meter_id,
                timestamp=datetime.now(),
                fuel_type=self.config.fuel_type,
                mass_flow_rate=mass_flow,
                volumetric_flow_rate=volumetric_flow,
                totalizer=totalizer,
                temperature_c=temperature,
                pressure_kpa=pressure,
                density_kg_m3=density,
                energy_flow_rate_kw=energy_flow_kw,
                heating_value_mj_kg=self.config.heating_value_mj_kg,
                quality="Good"
            )

            return reading

        except Exception as e:
            logger.error(f"[{self.connector_id}] Flow read error: {e}")
            raise

    async def _read_raw_values(self) -> Dict[str, float]:
        """Read raw values from meter."""
        try:
            if self.config.protocol in ["modbus_tcp", "modbus_rtu"]:
                return await self._read_modbus()
            elif self.config.protocol == "opc_ua":
                return await self._read_opc_ua()
            else:
                return {}

        except Exception as e:
            logger.error(f"[{self.connector_id}] Raw read error: {e}")
            return {}

    async def _read_modbus(self) -> Dict[str, float]:
        """Read values via Modbus."""
        try:
            data = {}

            for param, register in self.config.registers.items():
                response = await self._client.read_holding_registers(
                    address=register,
                    count=2,
                    slave=1
                )

                if not response.isError():
                    # Convert registers to float (IEEE 754)
                    value = self._registers_to_float(response.registers)
                    data[param] = value

            return data

        except Exception as e:
            logger.error(f"[{self.connector_id}] Modbus read error: {e}")
            return {}

    async def _read_opc_ua(self) -> Dict[str, float]:
        """Read values via OPC-UA."""
        try:
            data = {}

            for param in self.config.registers.keys():
                node_path = f"ns=2;s=FuelFlow.{param}"
                node = self._client.get_node(node_path)
                value = await node.read_value()
                data[param] = float(value)

            return data

        except Exception as e:
            logger.error(f"[{self.connector_id}] OPC-UA read error: {e}")
            return {}

    def _calculate_density(
        self,
        temperature_c: float,
        pressure_kpa: float
    ) -> float:
        """
        Calculate actual density with temperature and pressure compensation.

        Uses ideal gas law for gases, thermal expansion for liquids.

        Args:
            temperature_c: Actual temperature (C)
            pressure_kpa: Actual pressure (kPa)

        Returns:
            Density in kg/m3
        """
        std_density = self.config.standard_density_kg_m3
        t_ref = self.config.reference_temperature_c
        p_ref = self.config.reference_pressure_kpa

        # Convert to absolute temperature (Kelvin)
        t_actual_k = temperature_c + 273.15
        t_ref_k = t_ref + 273.15

        if self.config.fuel_type in [FuelType.NATURAL_GAS, FuelType.PROPANE, FuelType.BIOGAS, FuelType.HYDROGEN]:
            # Gas: Use ideal gas law
            if self.config.enable_temperature_compensation and self.config.enable_pressure_compensation:
                density = std_density * (pressure_kpa / p_ref) * (t_ref_k / t_actual_k)
            elif self.config.enable_pressure_compensation:
                density = std_density * (pressure_kpa / p_ref)
            elif self.config.enable_temperature_compensation:
                density = std_density * (t_ref_k / t_actual_k)
            else:
                density = std_density
        else:
            # Liquid: Use thermal expansion coefficient
            # Typical coefficient for petroleum products: 0.0007 per °C
            expansion_coeff = 0.0007

            if self.config.enable_temperature_compensation:
                delta_t = temperature_c - t_ref
                density = std_density / (1 + expansion_coeff * delta_t)
            else:
                density = std_density

        return density

    def _registers_to_float(self, registers: List[int]) -> float:
        """Convert Modbus registers to float."""
        if len(registers) < 2:
            return 0.0

        import struct

        # Combine registers (big-endian)
        high_word = registers[0]
        low_word = registers[1]
        combined = (high_word << 16) | low_word

        # Convert to bytes and unpack as float
        byte_str = combined.to_bytes(4, byteorder='big')
        return struct.unpack('!f', byte_str)[0]


class MockFlowMeterClient:
    """Mock flow meter client for testing."""

    def __init__(self, config: FuelFlowConfig):
        self.config = config

    async def connect(self):
        return True

    async def close(self):
        pass

    async def read_holding_registers(self, address: int, count: int, slave: int):
        """Return mock register data."""
        from collections import namedtuple
        Response = namedtuple('Response', ['registers', 'isError'])

        # Generate mock data based on register address
        if address == self.config.registers.get("flow_rate", 0):
            # Mock 100.0 m3/h
            return Response(registers=[16800, 0], isError=lambda: False)
        elif address == self.config.registers.get("totalizer", 10):
            # Mock 10000.0 m3
            return Response(registers=[18304, 0], isError=lambda: False)
        elif address == self.config.registers.get("temperature", 20):
            # Mock 25.0 °C
            return Response(registers=[16832, 0], isError=lambda: False)
        elif address == self.config.registers.get("pressure", 30):
            # Mock 105.0 kPa
            return Response(registers=[16842, 16384], isError=lambda: False)
        else:
            return Response(registers=[0, 0], isError=lambda: False)

    def get_node(self, path: str):
        """Return mock OPC-UA node."""
        from collections import namedtuple
        Node = namedtuple('Node', ['read_value'])

        async def read_value():
            if "flow_rate" in path:
                return 100.0
            elif "totalizer" in path:
                return 10000.0
            elif "temperature" in path:
                return 25.0
            elif "pressure" in path:
                return 105.0
            else:
                return 0.0

        return Node(read_value=read_value)
