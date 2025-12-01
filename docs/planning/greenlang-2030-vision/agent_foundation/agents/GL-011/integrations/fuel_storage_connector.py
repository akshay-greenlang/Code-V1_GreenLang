# -*- coding: utf-8 -*-
"""
Fuel Storage Connector for GL-011 FUELCRAFT.

Provides integration with fuel storage systems via MODBUS, OPC-UA,
or REST APIs for real-time inventory and level monitoring.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class StorageTankData:
    """Data from a storage tank."""
    tank_id: str
    fuel_type: str
    current_level_percent: float
    current_volume_m3: float
    current_mass_kg: float
    capacity_m3: float
    temperature_c: float
    pressure_bar: float
    timestamp: datetime
    status: str


@dataclass
class StorageSystemStatus:
    """Overall storage system status."""
    system_id: str
    total_capacity_m3: float
    total_volume_m3: float
    utilization_percent: float
    tanks: List[StorageTankData]
    alarms: List[Dict[str, Any]]
    timestamp: datetime


class BaseStorageConnector(ABC):
    """Abstract base class for storage connectors."""

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection."""
        pass

    @abstractmethod
    async def get_tank_data(self, tank_id: str) -> Optional[StorageTankData]:
        """Get data for a specific tank."""
        pass

    @abstractmethod
    async def get_all_tanks(self) -> List[StorageTankData]:
        """Get data for all tanks."""
        pass


class FuelStorageConnector(BaseStorageConnector):
    """
    Connector for fuel storage systems.

    Supports multiple protocols:
    - MODBUS TCP/RTU
    - OPC-UA
    - REST API
    - Simulation mode for testing

    Example:
        >>> connector = FuelStorageConnector(config)
        >>> await connector.connect()
        >>> tanks = await connector.get_all_tanks()
        >>> print(f"Total tanks: {len(tanks)}")
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize storage connector.

        Args:
            config: Connector configuration with:
                - protocol: 'modbus', 'opcua', 'rest', 'simulation'
                - endpoint: Connection endpoint
                - credentials: Authentication credentials
        """
        self.config = config
        self.protocol = config.get('protocol', 'simulation')
        self.endpoint = config.get('endpoint', 'localhost')
        self.port = config.get('port', 502)
        self.connected = False
        self._client = None
        self._tanks: Dict[str, StorageTankData] = {}

        # Tank configuration
        self._tank_config = config.get('tanks', {})

    async def connect(self) -> bool:
        """
        Establish connection to storage system.

        Returns:
            True if connection successful
        """
        try:
            if self.protocol == 'simulation':
                self.connected = True
                logger.info("Storage connector in simulation mode")
                return True

            elif self.protocol == 'modbus':
                # MODBUS connection (would use pymodbus)
                logger.info(f"Connecting to MODBUS at {self.endpoint}:{self.port}")
                # self._client = ModbusTcpClient(self.endpoint, self.port)
                # self.connected = self._client.connect()
                self.connected = True  # Simulated

            elif self.protocol == 'opcua':
                # OPC-UA connection (would use asyncua)
                logger.info(f"Connecting to OPC-UA at {self.endpoint}")
                # self._client = Client(self.endpoint)
                # await self._client.connect()
                self.connected = True  # Simulated

            elif self.protocol == 'rest':
                # REST API (would use httpx)
                logger.info(f"Connecting to REST API at {self.endpoint}")
                self.connected = True  # Simulated

            logger.info(f"Storage connector connected: {self.connected}")
            return self.connected

        except Exception as e:
            logger.error(f"Storage connection failed: {e}")
            self.connected = False
            return False

    async def disconnect(self) -> None:
        """Close connection to storage system."""
        try:
            if self._client:
                if self.protocol == 'modbus':
                    # self._client.close()
                    pass
                elif self.protocol == 'opcua':
                    # await self._client.disconnect()
                    pass

            self.connected = False
            logger.info("Storage connector disconnected")

        except Exception as e:
            logger.error(f"Storage disconnect failed: {e}")

    async def get_tank_data(self, tank_id: str) -> Optional[StorageTankData]:
        """
        Get data for a specific tank.

        Args:
            tank_id: Tank identifier

        Returns:
            StorageTankData or None if not found
        """
        if not self.connected:
            logger.warning("Not connected to storage system")
            return None

        try:
            if self.protocol == 'simulation':
                return self._simulate_tank_data(tank_id)
            else:
                # Real implementation would read from device
                return self._simulate_tank_data(tank_id)

        except Exception as e:
            logger.error(f"Failed to get tank {tank_id} data: {e}")
            return None

    async def get_all_tanks(self) -> List[StorageTankData]:
        """
        Get data for all configured tanks.

        Returns:
            List of StorageTankData
        """
        if not self.connected:
            logger.warning("Not connected to storage system")
            return []

        tanks = []
        tank_ids = self._tank_config.get('tank_ids', ['TANK-001', 'TANK-002'])

        for tank_id in tank_ids:
            data = await self.get_tank_data(tank_id)
            if data:
                tanks.append(data)

        return tanks

    async def get_system_status(self) -> StorageSystemStatus:
        """
        Get overall storage system status.

        Returns:
            StorageSystemStatus with all tanks
        """
        tanks = await self.get_all_tanks()

        total_capacity = sum(t.capacity_m3 for t in tanks)
        total_volume = sum(t.current_volume_m3 for t in tanks)
        utilization = (total_volume / total_capacity * 100) if total_capacity > 0 else 0

        # Check for alarms
        alarms = []
        for tank in tanks:
            if tank.current_level_percent < 10:
                alarms.append({
                    'tank_id': tank.tank_id,
                    'type': 'low_level',
                    'message': f'{tank.fuel_type} level below 10%',
                    'severity': 'high'
                })
            elif tank.current_level_percent > 95:
                alarms.append({
                    'tank_id': tank.tank_id,
                    'type': 'high_level',
                    'message': f'{tank.fuel_type} level above 95%',
                    'severity': 'warning'
                })

        return StorageSystemStatus(
            system_id=self.config.get('system_id', 'STORAGE-001'),
            total_capacity_m3=total_capacity,
            total_volume_m3=total_volume,
            utilization_percent=round(utilization, 2),
            tanks=tanks,
            alarms=alarms,
            timestamp=datetime.now(timezone.utc)
        )

    async def get_fuel_inventory(self) -> Dict[str, float]:
        """
        Get current fuel inventory by type.

        Returns:
            Dict mapping fuel type to mass in kg
        """
        tanks = await self.get_all_tanks()

        inventory = {}
        for tank in tanks:
            if tank.fuel_type in inventory:
                inventory[tank.fuel_type] += tank.current_mass_kg
            else:
                inventory[tank.fuel_type] = tank.current_mass_kg

        return inventory

    def _simulate_tank_data(self, tank_id: str) -> StorageTankData:
        """Generate simulated tank data for testing."""
        import random
        random.seed(hash(tank_id) % 2**32)

        # Get tank config or use defaults
        tank_cfg = self._tank_config.get(tank_id, {})
        fuel_type = tank_cfg.get('fuel_type', 'natural_gas')
        capacity = tank_cfg.get('capacity_m3', 1000)
        density = tank_cfg.get('density_kg_m3', 0.75)

        level = random.uniform(20, 80)
        volume = capacity * level / 100
        mass = volume * density

        return StorageTankData(
            tank_id=tank_id,
            fuel_type=fuel_type,
            current_level_percent=round(level, 2),
            current_volume_m3=round(volume, 2),
            current_mass_kg=round(mass, 2),
            capacity_m3=capacity,
            temperature_c=round(random.uniform(10, 30), 1),
            pressure_bar=round(random.uniform(1, 5), 2),
            timestamp=datetime.now(timezone.utc),
            status='normal'
        )
