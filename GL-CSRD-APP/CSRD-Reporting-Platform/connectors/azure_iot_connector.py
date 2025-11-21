# -*- coding: utf-8 -*-
"""
Azure IoT Hub Connector

Connects to Azure IoT Hub to collect real-time data from IoT sensors and meters.
Supports energy meters, water meters, air quality sensors, etc.

Author: GreenLang AI Team
Date: 2025-10-18
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
from greenlang.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class AzureIoTConnector:
    """
    Connect to Azure IoT Hub for sensor data collection

    In production, this would use the Azure IoT SDK.
    For now, we provide the interface and mock data.
    """

    def __init__(self, connection_string: str):
        """
        Initialize Azure IoT connector

        Args:
            connection_string: Azure IoT Hub connection string
        """
        self.connection_string = connection_string
        self.device_registry = {}
        logger.info("Azure IoT Connector initialized")

    def register_device(self, device_id: str, device_type: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Register an IoT device

        Args:
            device_id: Unique device identifier
            device_type: Type of device (e.g., 'energy_meter', 'water_meter')
            metadata: Optional device metadata
        """
        self.device_registry[device_id] = {
            'device_id': device_id,
            'device_type': device_type,
            'metadata': metadata or {},
            'registered_at': DeterministicClock.now().isoformat()
        }
        logger.info(f"Registered device: {device_id} (type: {device_type})")

    async def fetch_sensor_data(
        self,
        device_id: str,
        metric_type: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Fetch sensor data from a specific device

        Args:
            device_id: Device ID
            metric_type: Type of metric to fetch
            start_time: Start of time range
            end_time: End of time range

        Returns:
            Sensor data with measurements
        """
        logger.info(f"Fetching data from device {device_id} for metric {metric_type}")

        # In production, this would query Azure IoT Hub
        # For now, return mock data

        # Default time range to last hour
        if not end_time:
            end_time = DeterministicClock.now()
        if not start_time:
            start_time = end_time - timedelta(hours=1)

        # Mock data based on device type
        device_info = self.device_registry.get(device_id, {'device_type': 'unknown'})

        mock_data = {
            'energy_meter': {
                'value': 125.5,
                'unit': 'kWh',
                'power_factor': 0.92
            },
            'water_meter': {
                'value': 2.3,
                'unit': 'm3',
                'pressure': 3.5
            },
            'air_quality': {
                'value': 42,
                'unit': 'AQI',
                'pm25': 15.2,
                'pm10': 22.1
            }
        }

        device_type = device_info.get('device_type', 'unknown')
        measurement = mock_data.get(device_type, {'value': 0, 'unit': 'unknown'})

        return {
            'device_id': device_id,
            'device_type': device_type,
            'metric_type': metric_type,
            'measurement': measurement,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'timestamp': DeterministicClock.now().isoformat()
        }

    async def fetch_all_devices(
        self,
        metric_type: str,
        device_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch data from all registered devices

        Args:
            metric_type: Type of metric to fetch
            device_type: Optional filter by device type

        Returns:
            List of sensor data from all devices
        """
        logger.info(f"Fetching data from all devices for metric {metric_type}")

        results = []

        for device_id, device_info in self.device_registry.items():
            # Filter by device type if specified
            if device_type and device_info['device_type'] != device_type:
                continue

            data = await self.fetch_sensor_data(device_id, metric_type)
            results.append(data)

        logger.info(f"Fetched data from {len(results)} devices")
        return results

    async def aggregate_readings(
        self,
        device_ids: List[str],
        metric_type: str,
        aggregation: str = 'sum'
    ) -> Dict[str, Any]:
        """
        Aggregate readings from multiple devices

        Args:
            device_ids: List of device IDs
            metric_type: Metric to aggregate
            aggregation: Aggregation method ('sum', 'avg', 'min', 'max')

        Returns:
            Aggregated result
        """
        logger.info(f"Aggregating {aggregation} for {metric_type} across {len(device_ids)} devices")

        # Fetch data from all devices
        all_data = []
        for device_id in device_ids:
            data = await self.fetch_sensor_data(device_id, metric_type)
            all_data.append(data['measurement']['value'])

        # Perform aggregation
        if aggregation == 'sum':
            result_value = sum(all_data)
        elif aggregation == 'avg':
            result_value = sum(all_data) / len(all_data) if all_data else 0
        elif aggregation == 'min':
            result_value = min(all_data) if all_data else 0
        elif aggregation == 'max':
            result_value = max(all_data) if all_data else 0
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        return {
            'metric_type': metric_type,
            'aggregation': aggregation,
            'value': result_value,
            'device_count': len(device_ids),
            'timestamp': DeterministicClock.now().isoformat()
        }

    def get_device_status(self, device_id: str) -> Dict[str, Any]:
        """
        Get status of a specific device

        Args:
            device_id: Device ID

        Returns:
            Device status information
        """
        if device_id not in self.device_registry:
            return {
                'device_id': device_id,
                'status': 'not_registered'
            }

        # In production, would check actual device connection status
        return {
            'device_id': device_id,
            'status': 'online',
            'last_activity': DeterministicClock.now().isoformat(),
            'connection_state': 'connected'
        }


# Example usage
async def main():
    """Example Azure IoT connector usage"""
    # Initialize connector
    connection_string = "HostName=example.azure-devices.net;SharedAccessKeyName=xxx;SharedAccessKey=xxx"
    connector = AzureIoTConnector(connection_string)

    # Register devices
    connector.register_device('device001', 'energy_meter', {'location': 'Building A'})
    connector.register_device('device002', 'energy_meter', {'location': 'Building B'})
    connector.register_device('device003', 'water_meter', {'location': 'Main Line'})
    connector.register_device('device004', 'air_quality', {'location': 'Outdoor'})

    # Fetch data from single device
    data = await connector.fetch_sensor_data('device001', 'energy_consumption')
    print(f"\nDevice 001 data: {data['measurement']}")

    # Fetch from all energy meters
    all_energy = await connector.fetch_all_devices('energy_consumption', device_type='energy_meter')
    print(f"\nTotal energy meters: {len(all_energy)}")

    # Aggregate energy consumption
    aggregated = await connector.aggregate_readings(
        ['device001', 'device002'],
        'energy_consumption',
        aggregation='sum'
    )
    print(f"\nTotal energy consumption: {aggregated['value']} kWh")

    # Check device status
    status = connector.get_device_status('device001')
    print(f"\nDevice 001 status: {status['status']}")


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    asyncio.run(main())
