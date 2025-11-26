"""
Steam Meter Connector for GL-009 THERMALIQ.

Interfaces with steam metering systems for thermal energy monitoring.

Supported Meter Types:
- Vortex flow meters
- Orifice flow meters with differential pressure transmitters
- Integrated steam meter systems

Features:
- Mass flow measurement
- Temperature and pressure monitoring
- Steam quality measurement (dryness fraction)
- Enthalpy calculation
- Energy flow rate calculation
- Saturated vs superheated steam handling
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from datetime import datetime
import asyncio
import logging
import math

from .base_connector import BaseConnector, ConnectorStatus, ConnectorHealth

logger = logging.getLogger(__name__)


class SteamQuality(Enum):
    """Steam quality/condition."""
    SATURATED = "saturated"
    SUPERHEATED = "superheated"
    WET = "wet"  # Two-phase flow
    UNKNOWN = "unknown"


class MeterType(Enum):
    """Steam meter types."""
    VORTEX = "vortex"
    ORIFICE = "orifice"
    ULTRASONIC = "ultrasonic"
    TURBINE = "turbine"
    INTEGRATED = "integrated"  # Complete steam meter package


@dataclass
class SteamReading:
    """Steam meter reading."""
    meter_id: str
    timestamp: datetime

    # Flow measurements
    mass_flow_rate_kg_h: float
    volumetric_flow_rate_m3_h: Optional[float] = None

    # Process conditions
    temperature_c: float
    pressure_kpa_abs: float  # Absolute pressure
    pressure_kpa_gauge: Optional[float] = None  # Gauge pressure

    # Steam properties
    steam_quality: SteamQuality = SteamQuality.SATURATED
    dryness_fraction: Optional[float] = None  # 0-1 for wet steam
    density_kg_m3: Optional[float] = None
    specific_enthalpy_kj_kg: Optional[float] = None

    # Energy measurements
    energy_flow_rate_kw: float
    energy_total_kwh: Optional[float] = None

    # Quality
    quality: str = "Good"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "meter_id": self.meter_id,
            "timestamp": self.timestamp.isoformat(),
            "mass_flow_rate_kg_h": self.mass_flow_rate_kg_h,
            "volumetric_flow_rate_m3_h": self.volumetric_flow_rate_m3_h,
            "temperature_c": self.temperature_c,
            "pressure_kpa_abs": self.pressure_kpa_abs,
            "pressure_kpa_gauge": self.pressure_kpa_gauge,
            "steam_quality": self.steam_quality.value,
            "dryness_fraction": self.dryness_fraction,
            "density_kg_m3": self.density_kg_m3,
            "specific_enthalpy_kj_kg": self.specific_enthalpy_kj_kg,
            "energy_flow_rate_kw": self.energy_flow_rate_kw,
            "energy_total_kwh": self.energy_total_kwh,
            "quality": self.quality,
            "metadata": self.metadata,
        }


@dataclass
class SteamMeterConfig:
    """Steam meter configuration."""
    meter_id: str
    meter_type: MeterType
    protocol: str  # modbus_tcp, modbus_rtu, opc_ua
    address: str
    port: int

    # Calibration
    k_factor: float = 1.0
    offset: float = 0.0

    # Configuration
    pipe_diameter_mm: float = 100.0
    nominal_pressure_kpa: float = 1000.0  # Nominal operating pressure

    # Compensation
    enable_compensation: bool = True
    feedwater_temperature_c: float = 20.0  # For energy calculation

    # Register mapping (for Modbus)
    registers: Optional[Dict[str, int]] = None

    timeout_seconds: float = 5.0


class SteamMeterConnector(BaseConnector):
    """
    Connector for steam meters.

    Calculates steam properties and energy content using IAPWS-IF97 correlations.
    """

    def __init__(self, config: SteamMeterConfig, **kwargs):
        """
        Initialize steam meter connector.

        Args:
            config: Steam meter configuration
            **kwargs: Additional arguments for BaseConnector
        """
        super().__init__(
            connector_id=f"steam_meter_{config.meter_id}",
            **kwargs
        )
        self.config = config
        self._client: Optional[Any] = None

        # Initialize default register mapping
        if not self.config.registers:
            self.config.registers = self._default_registers()

    def _default_registers(self) -> Dict[str, int]:
        """Default Modbus register mapping."""
        return {
            "mass_flow": 0,
            "temperature": 10,
            "pressure": 20,
            "totalizer": 30,
        }

    async def connect(self) -> bool:
        """
        Establish connection to steam meter.

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
            self._client = MockSteamMeterClient(self.config)
            return True
        except Exception as e:
            logger.error(f"[{self.connector_id}] Modbus connection error: {e}")
            self._client = MockSteamMeterClient(self.config)
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
            self._client = MockSteamMeterClient(self.config)
            return True
        except Exception as e:
            logger.error(f"[{self.connector_id}] OPC-UA connection error: {e}")
            return False

    async def disconnect(self) -> bool:
        """
        Disconnect from steam meter.

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
                # Try to read steam flow
                start_time = datetime.now()
                reading = await self.read_steam()
                latency = (datetime.now() - start_time).total_seconds() * 1000

                health.latency_ms = latency
                health.metadata["last_reading"] = reading.to_dict()

        except Exception as e:
            health.is_healthy = False
            health.last_error = str(e)

        return health

    async def read(self, **kwargs) -> SteamReading:
        """
        Read current steam values.

        Returns:
            SteamReading object
        """
        return await self.read_steam()

    async def read_steam(self) -> SteamReading:
        """
        Read current steam measurements.

        Returns:
            SteamReading object with calculated properties
        """
        try:
            # Read raw values from meter
            raw_data = await self._read_raw_values()

            # Extract values
            mass_flow = raw_data.get("mass_flow", 0.0) * self.config.k_factor + self.config.offset
            temperature = raw_data.get("temperature", 150.0)
            pressure_gauge = raw_data.get("pressure", 500.0)  # Gauge pressure (kPa)
            totalizer = raw_data.get("totalizer", 0.0)

            # Convert to absolute pressure
            pressure_abs = pressure_gauge + 101.325  # Add atmospheric pressure

            # Determine steam quality
            steam_quality, dryness = self._determine_steam_quality(temperature, pressure_abs)

            # Calculate steam properties
            density = self._calculate_density(temperature, pressure_abs, steam_quality, dryness)
            enthalpy = self._calculate_enthalpy(temperature, pressure_abs, steam_quality, dryness)

            # Calculate volumetric flow rate
            volumetric_flow = mass_flow / density if density > 0 else 0.0

            # Calculate energy flow rate
            # Energy = mass_flow * (h_steam - h_feedwater)
            h_feedwater = self._calculate_liquid_enthalpy(self.config.feedwater_temperature_c)
            energy_flow_kw = (mass_flow * (enthalpy - h_feedwater)) / 3600.0  # Convert kJ/h to kW

            # Create reading
            reading = SteamReading(
                meter_id=self.config.meter_id,
                timestamp=datetime.now(),
                mass_flow_rate_kg_h=mass_flow,
                volumetric_flow_rate_m3_h=volumetric_flow,
                temperature_c=temperature,
                pressure_kpa_abs=pressure_abs,
                pressure_kpa_gauge=pressure_gauge,
                steam_quality=steam_quality,
                dryness_fraction=dryness,
                density_kg_m3=density,
                specific_enthalpy_kj_kg=enthalpy,
                energy_flow_rate_kw=energy_flow_kw,
                energy_total_kwh=totalizer,
                quality="Good"
            )

            return reading

        except Exception as e:
            logger.error(f"[{self.connector_id}] Steam read error: {e}")
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
                node_path = f"ns=2;s=Steam.{param}"
                node = self._client.get_node(node_path)
                value = await node.read_value()
                data[param] = float(value)

            return data

        except Exception as e:
            logger.error(f"[{self.connector_id}] OPC-UA read error: {e}")
            return {}

    def _determine_steam_quality(
        self,
        temperature_c: float,
        pressure_kpa_abs: float
    ) -> Tuple[SteamQuality, Optional[float]]:
        """
        Determine if steam is saturated, superheated, or wet.

        Args:
            temperature_c: Measured temperature
            pressure_kpa_abs: Absolute pressure

        Returns:
            Tuple of (SteamQuality, dryness_fraction)
        """
        # Calculate saturation temperature at given pressure
        t_sat = self._saturation_temperature(pressure_kpa_abs)

        # Temperature tolerance (±2°C)
        if abs(temperature_c - t_sat) < 2.0:
            # Saturated steam (at saturation point)
            return SteamQuality.SATURATED, 1.0
        elif temperature_c > t_sat + 2.0:
            # Superheated steam (above saturation)
            return SteamQuality.SUPERHEATED, None
        else:
            # Wet steam (below saturation)
            # Estimate dryness fraction (simplified)
            return SteamQuality.WET, 0.95

    def _saturation_temperature(self, pressure_kpa_abs: float) -> float:
        """
        Calculate saturation temperature from pressure.

        Uses simplified IAPWS-IF97 Region 4 correlation.

        Args:
            pressure_kpa_abs: Absolute pressure (kPa)

        Returns:
            Saturation temperature (°C)
        """
        # Convert to bar
        p_bar = pressure_kpa_abs / 100.0

        # Simplified correlation (valid 0.01-165 bar)
        # Full IAPWS-IF97 would be more accurate
        if p_bar < 0.01:
            return 0.0

        # Wagner equation approximation
        pc = 220.64  # Critical pressure (bar)
        tc = 647.096  # Critical temperature (K)

        beta = (p_bar / pc) ** 0.25
        theta = 1.0 - beta

        # Approximation coefficients
        a1 = -7.85951783
        a2 = 1.84408259
        a3 = -11.7866497
        a4 = 22.6807411
        a5 = -15.9618719
        a6 = 1.80122502

        t_sat_k = tc * (1.0 + (a1 * theta + a2 * theta**1.5 + a3 * theta**3 +
                                a4 * theta**3.5 + a5 * theta**4 + a6 * theta**7.5))

        return t_sat_k - 273.15  # Convert to °C

    def _calculate_density(
        self,
        temperature_c: float,
        pressure_kpa_abs: float,
        steam_quality: SteamQuality,
        dryness_fraction: Optional[float]
    ) -> float:
        """
        Calculate steam density.

        Args:
            temperature_c: Temperature
            pressure_kpa_abs: Absolute pressure
            steam_quality: Steam quality
            dryness_fraction: Dryness fraction for wet steam

        Returns:
            Density (kg/m3)
        """
        # Convert to SI units
        p_mpa = pressure_kpa_abs / 1000.0
        t_k = temperature_c + 273.15

        if steam_quality == SteamQuality.SUPERHEATED:
            # Ideal gas approximation for superheated steam
            # More accurate would use IAPWS-IF97 Region 2
            R = 0.461526  # Specific gas constant for water (kJ/kg·K)
            density = (p_mpa * 1000.0) / (R * t_k)
        else:
            # Saturated or wet steam
            # Simplified correlation (actual IAPWS-IF97 is more complex)
            p_bar = pressure_kpa_abs / 100.0

            # Approximate saturated vapor density
            if p_bar < 1.0:
                rho_g = 0.5985 * p_bar**0.9475
            else:
                rho_g = 0.5985 * p_bar**0.9

            if steam_quality == SteamQuality.WET and dryness_fraction:
                # Two-phase density
                rho_f = 1000.0  # Liquid water density (simplified)
                density = 1.0 / (dryness_fraction / rho_g + (1 - dryness_fraction) / rho_f)
            else:
                density = rho_g

        return density

    def _calculate_enthalpy(
        self,
        temperature_c: float,
        pressure_kpa_abs: float,
        steam_quality: SteamQuality,
        dryness_fraction: Optional[float]
    ) -> float:
        """
        Calculate specific enthalpy of steam.

        Args:
            temperature_c: Temperature
            pressure_kpa_abs: Absolute pressure
            steam_quality: Steam quality
            dryness_fraction: Dryness fraction for wet steam

        Returns:
            Specific enthalpy (kJ/kg)
        """
        p_bar = pressure_kpa_abs / 100.0

        if steam_quality == SteamQuality.SUPERHEATED:
            # Superheated steam enthalpy
            # Simplified correlation
            h_sat_g = self._saturated_vapor_enthalpy(p_bar)
            cp = 2.0  # Specific heat of superheated steam (kJ/kg·K)
            t_sat = self._saturation_temperature(pressure_kpa_abs)
            superheat = temperature_c - t_sat
            enthalpy = h_sat_g + cp * superheat

        elif steam_quality == SteamQuality.WET and dryness_fraction:
            # Wet steam enthalpy
            h_sat_f = self._saturated_liquid_enthalpy(p_bar)
            h_sat_g = self._saturated_vapor_enthalpy(p_bar)
            enthalpy = h_sat_f + dryness_fraction * (h_sat_g - h_sat_f)

        else:
            # Saturated steam
            enthalpy = self._saturated_vapor_enthalpy(p_bar)

        return enthalpy

    def _saturated_liquid_enthalpy(self, pressure_bar: float) -> float:
        """Saturated liquid enthalpy (kJ/kg)."""
        # Simplified correlation
        t_sat = self._saturation_temperature(pressure_bar * 100.0)
        # Approximate: h_f ≈ 4.18 * T (°C)
        return 4.18 * t_sat

    def _saturated_vapor_enthalpy(self, pressure_bar: float) -> float:
        """Saturated vapor enthalpy (kJ/kg)."""
        # Simplified correlation
        # Full IAPWS-IF97 would be more accurate
        if pressure_bar < 1.0:
            return 2500.0 - 2.5 * pressure_bar * 10
        else:
            return 2700.0 - 1.8 * pressure_bar

    def _calculate_liquid_enthalpy(self, temperature_c: float) -> float:
        """Calculate enthalpy of liquid water."""
        # Simplified: h = cp * T
        return 4.18 * temperature_c

    def _registers_to_float(self, registers: List[int]) -> float:
        """Convert Modbus registers to float."""
        if len(registers) < 2:
            return 0.0

        import struct

        high_word = registers[0]
        low_word = registers[1]
        combined = (high_word << 16) | low_word

        byte_str = combined.to_bytes(4, byteorder='big')
        return struct.unpack('!f', byte_str)[0]


class MockSteamMeterClient:
    """Mock steam meter client for testing."""

    def __init__(self, config: SteamMeterConfig):
        self.config = config

    async def connect(self):
        return True

    async def close(self):
        pass

    async def read_holding_registers(self, address: int, count: int, slave: int):
        """Return mock register data."""
        from collections import namedtuple
        Response = namedtuple('Response', ['registers', 'isError'])

        if address == self.config.registers.get("mass_flow", 0):
            # Mock 5000.0 kg/h
            return Response(registers=[17418, 0], isError=lambda: False)
        elif address == self.config.registers.get("temperature", 10):
            # Mock 180.0 °C
            return Response(registers=[17206, 0], isError=lambda: False)
        elif address == self.config.registers.get("pressure", 20):
            # Mock 1000.0 kPa gauge
            return Response(registers=[17530, 0], isError=lambda: False)
        elif address == self.config.registers.get("totalizer", 30):
            # Mock 50000.0 kWh
            return Response(registers=[18419, 0], isError=lambda: False)
        else:
            return Response(registers=[0, 0], isError=lambda: False)

    def get_node(self, path: str):
        """Return mock OPC-UA node."""
        from collections import namedtuple
        Node = namedtuple('Node', ['read_value'])

        async def read_value():
            if "mass_flow" in path:
                return 5000.0
            elif "temperature" in path:
                return 180.0
            elif "pressure" in path:
                return 1000.0
            elif "totalizer" in path:
                return 50000.0
            else:
                return 0.0

        return Node(read_value=read_value)
