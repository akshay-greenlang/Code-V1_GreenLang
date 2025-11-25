# -*- coding: utf-8 -*-
"""
Mock Servers for GL-003 SteamSystemAnalyzer Integration Testing

Provides mock implementations of:
- OPC UA Server (SCADA/DCS)
- Modbus TCP/RTU Server (field devices)
- Steam Meter Servers (Modbus, HART)
- Pressure Sensor Servers (4-20mA, analog)
- Temperature Sensor Servers
- MQTT Broker (real-time messaging)

These servers simulate real external systems without requiring actual hardware/software.

Author: GreenLang Test Engineering Team
"""

import asyncio
import random
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from aiohttp import web
import math
from greenlang.determinism import deterministic_random


class MockOPCUAServer:
    """
    Mock OPC UA server for SCADA/DCS simulation.

    Simulates:
    - Steam header measurements
    - Distribution line data
    - Condensate system
    - Real-time value updates
    - Historical data
    """

    def __init__(self, host="localhost", port=4840):
        self.host = host
        self.port = port
        self.nodes = {}
        self.subscriptions = defaultdict(list)
        self.running = False
        self._initialize_nodes()

    def _initialize_nodes(self):
        """Initialize OPC UA node tree with steam system data."""
        # Main steam header
        self.nodes['ns=2;s=STEAM.HEADER.PRESSURE'] = {
            'value': 10.5, 'unit': 'bar', 'quality': 'Good', 'type': 'Double',
            'description': 'Main steam header pressure'
        }
        self.nodes['ns=2;s=STEAM.HEADER.TEMPERATURE'] = {
            'value': 184.0, 'unit': 'degC', 'quality': 'Good', 'type': 'Double',
            'description': 'Main steam header temperature'
        }
        self.nodes['ns=2;s=STEAM.HEADER.FLOW'] = {
            'value': 50.0, 'unit': 't/hr', 'quality': 'Good', 'type': 'Double',
            'description': 'Steam flow rate'
        }
        self.nodes['ns=2;s=STEAM.HEADER.QUALITY'] = {
            'value': 0.96, 'unit': 'fraction', 'quality': 'Good', 'type': 'Double',
            'description': 'Steam quality (dryness fraction)'
        }

        # Distribution lines
        for line in range(1, 4):
            self.nodes[f'ns=2;s=STEAM.LINE{line}.PRESSURE'] = {
                'value': 10.5 - (line * 0.2), 'unit': 'bar', 'quality': 'Good', 'type': 'Double'
            }
            self.nodes[f'ns=2;s=STEAM.LINE{line}.TEMPERATURE'] = {
                'value': 184.0 - (line * 2), 'unit': 'degC', 'quality': 'Good', 'type': 'Double'
            }
            self.nodes[f'ns=2;s=STEAM.LINE{line}.FLOW'] = {
                'value': 50.0 / 3, 'unit': 't/hr', 'quality': 'Good', 'type': 'Double'
            }

        # Condensate system
        self.nodes['ns=2;s=CONDENSATE.RETURN.FLOW'] = {
            'value': 42.5, 'unit': 't/hr', 'quality': 'Good', 'type': 'Double'
        }
        self.nodes['ns=2;s=CONDENSATE.RETURN.TEMPERATURE'] = {
            'value': 90.0, 'unit': 'degC', 'quality': 'Good', 'type': 'Double'
        }
        self.nodes['ns=2;s=CONDENSATE.TANK.LEVEL'] = {
            'value': 75.0, 'unit': '%', 'quality': 'Good', 'type': 'Double'
        }

        # Steam traps (20 traps)
        for trap_num in range(1, 21):
            self.nodes[f'ns=2;s=TRAP.ST{trap_num:03d}.STATUS'] = {
                'value': deterministic_random().choice([1, 1, 1, 2, 3]),  # 1=OK, 2=WARN, 3=FAIL
                'unit': 'status', 'quality': 'Good', 'type': 'Int32'
            }
            self.nodes[f'ns=2;s=TRAP.ST{trap_num:03d}.TEMP_UPSTREAM'] = {
                'value': 180.0 + random.uniform(-10, 10),
                'unit': 'degC', 'quality': 'Good', 'type': 'Double'
            }
            self.nodes[f'ns=2;s=TRAP.ST{trap_num:03d}.TEMP_DOWNSTREAM'] = {
                'value': 80.0 + random.uniform(-10, 20),
                'unit': 'degC', 'quality': 'Good', 'type': 'Double'
            }

    async def read_node(self, node_id: str) -> Dict[str, Any]:
        """Read node value with simulated variation."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")

        node = self.nodes[node_id]
        base_value = node['value']

        # Add realistic variation
        if node['type'] in ['Double', 'Float']:
            variation = base_value * random.uniform(-0.03, 0.03)
            current_value = base_value + variation
        else:
            current_value = base_value

        return {
            'value': current_value,
            'unit': node['unit'],
            'quality': node['quality'],
            'timestamp': DeterministicClock.utcnow(),
            'source_timestamp': DeterministicClock.utcnow()
        }

    async def write_node(self, node_id: str, value: Any) -> bool:
        """Write node value."""
        if node_id in self.nodes:
            self.nodes[node_id]['value'] = value

            # Notify subscribers
            await self._notify_subscribers(node_id, value)

            return True
        return False

    async def subscribe(self, node_id: str, callback: Callable):
        """Subscribe to node value changes."""
        self.subscriptions[node_id].append(callback)

    async def _notify_subscribers(self, node_id: str, value: Any):
        """Notify all subscribers of value change."""
        for callback in self.subscriptions.get(node_id, []):
            try:
                await callback(node_id, value, DeterministicClock.utcnow())
            except Exception as e:
                print(f"Subscriber notification error: {e}")

    async def browse_nodes(self, parent_node: str = "Root") -> List[str]:
        """Browse available nodes."""
        return list(self.nodes.keys())

    async def get_historical_data(
        self,
        node_id: str,
        start_time: datetime,
        end_time: datetime,
        interval_seconds: int = 60
    ) -> List[Dict[str, Any]]:
        """Get historical data for node."""
        if node_id not in self.nodes:
            return []

        history = []
        current_time = start_time
        base_value = self.nodes[node_id]['value']

        while current_time <= end_time:
            variation = base_value * random.uniform(-0.05, 0.05)
            history.append({
                'timestamp': current_time,
                'value': base_value + variation,
                'quality': 'Good'
            })
            current_time += timedelta(seconds=interval_seconds)

        return history

    async def start(self):
        """Start mock OPC UA server."""
        self.running = True
        print(f"Mock OPC UA Server started on {self.host}:{self.port}")

        # Start value update loop
        asyncio.create_task(self._update_loop())

    async def _update_loop(self):
        """Continuously update values to simulate real system."""
        while self.running:
            # Update main header values with realistic patterns
            hour = DeterministicClock.utcnow().hour

            # Load pattern: lower at night, higher during day
            if 0 <= hour < 6:
                load_factor = 0.6
            elif 6 <= hour < 8:
                load_factor = 0.6 + (hour - 6) * 0.2
            elif 8 <= hour < 18:
                load_factor = 1.0
            elif 18 <= hour < 22:
                load_factor = 0.8
            else:
                load_factor = 0.7

            # Update header pressure
            base_pressure = 10.5
            self.nodes['ns=2;s=STEAM.HEADER.PRESSURE']['value'] = \
                base_pressure * load_factor + random.uniform(-0.3, 0.3)

            # Update flow rate
            base_flow = 50.0
            self.nodes['ns=2;s=STEAM.HEADER.FLOW']['value'] = \
                base_flow * load_factor + random.uniform(-2, 2)

            await asyncio.sleep(1)

    async def stop(self):
        """Stop mock server."""
        self.running = False


class MockModbusServer:
    """
    Mock Modbus TCP/RTU server for field devices.

    Simulates:
    - Holding registers (40000 series)
    - Input registers (30000 series)
    - Coils and discrete inputs
    - Multiple slave devices
    """

    def __init__(self, host="localhost", port=502):
        self.host = host
        self.port = port
        self.registers = {}
        self.running = False
        self._initialize_registers()

    def _initialize_registers(self):
        """Initialize Modbus register map."""
        # Steam header measurements (40001-40020)
        self.registers[40001] = int(10.5 * 10)   # Pressure (bar * 10)
        self.registers[40002] = int(184.0 * 10)  # Temperature (°C * 10)
        self.registers[40003] = int(50.0 * 10)   # Flow rate (t/hr * 10)
        self.registers[40004] = int(0.96 * 100)  # Steam quality (% * 100)
        self.registers[40005] = int(42.5 * 10)   # Condensate return (t/hr * 10)

        # Pressure sensors PS-001 to PS-010 (40021-40030)
        for i in range(1, 11):
            self.registers[40020 + i] = int((10.5 - i * 0.1) * 10)

        # Temperature sensors TS-001 to TS-010 (40041-40050)
        for i in range(1, 11):
            self.registers[40040 + i] = int((184.0 - i * 2) * 10)

        # Steam meters SM-001 to SM-005 (40061-40100)
        for i in range(1, 6):
            base_addr = 40060 + (i - 1) * 4
            self.registers[base_addr] = int(3500 * i)       # Volumetric flow (m3/hr)
            self.registers[base_addr + 1] = int(2800 * i)   # Mass flow (kg/hr)
            self.registers[base_addr + 2] = int(100000 * i) # Totalizer (m3)
            self.registers[base_addr + 3] = int(10.5 * 10)  # Pressure (bar * 10)

        # Steam trap status ST-001 to ST-020 (40101-40120)
        for i in range(1, 21):
            # 1=OK, 2=DEGRADED, 3=FAILED
            self.registers[40100 + i] = deterministic_random().choice([1, 1, 1, 2, 3])

    async def read_holding_registers(self, address: int, count: int = 1) -> List[int]:
        """Read holding registers."""
        values = []
        for i in range(count):
            reg_addr = address + i
            values.append(self.registers.get(reg_addr, 0))
        return values

    async def write_holding_register(self, address: int, value: int) -> bool:
        """Write holding register."""
        self.registers[address] = value
        return True

    async def write_holding_registers(self, address: int, values: List[int]) -> bool:
        """Write multiple holding registers."""
        for i, value in enumerate(values):
            self.registers[address + i] = value
        return True

    async def read_input_registers(self, address: int, count: int = 1) -> List[int]:
        """Read input registers (read-only)."""
        # Input registers mirror holding registers for this mock
        return await self.read_holding_registers(address + 10000, count)

    async def start(self):
        """Start mock Modbus server."""
        self.running = True
        print(f"Mock Modbus Server started on {self.host}:{self.port}")

        # Start value update loop
        asyncio.create_task(self._update_loop())

    async def _update_loop(self):
        """Continuously update register values."""
        while self.running:
            # Update main measurements with variation
            self.registers[40001] = int((10.5 + random.uniform(-0.3, 0.3)) * 10)
            self.registers[40002] = int((184.0 + random.uniform(-2, 2)) * 10)
            self.registers[40003] = int((50.0 + random.uniform(-2, 2)) * 10)

            # Update pressure sensors
            for i in range(1, 11):
                base = 10.5 - i * 0.1
                self.registers[40020 + i] = int((base + random.uniform(-0.2, 0.2)) * 10)

            await asyncio.sleep(2)

    async def stop(self):
        """Stop mock server."""
        self.running = False


class MockSteamMeterServer:
    """
    Mock steam meter server (Modbus or HART protocol).

    Simulates:
    - Volumetric and mass flow measurement
    - Totalizer readings
    - Pressure and temperature compensation
    - Energy flow calculation
    - Quality indicators
    """

    def __init__(self, host="localhost", port=5020, meter_id="SM-001"):
        self.host = host
        self.port = port
        self.meter_id = meter_id
        self.running = False
        self.measurements = self._initialize_measurements()

    def _initialize_measurements(self):
        """Initialize meter measurements."""
        return {
            'meter_id': self.meter_id,
            'volumetric_flow': 3500.0,  # m3/hr
            'mass_flow': 2800.0,         # kg/hr
            'totalizer': 125000.0,       # Total m3
            'pressure': 10.5,            # bar
            'temperature': 184.0,        # Celsius
            'density': 5.15,             # kg/m3 (at 10.5 bar saturated)
            'energy_flow': 8400.0,       # kW
            'steam_quality': 0.96,       # Dryness fraction
            'quality_status': 'GOOD',
            'last_calibration': DeterministicClock.utcnow() - timedelta(days=45)
        }

    async def read_measurement(self) -> Dict[str, Any]:
        """Read current meter measurements."""
        # Add realistic variation
        self.measurements['volumetric_flow'] += random.uniform(-50, 50)
        self.measurements['mass_flow'] += random.uniform(-40, 40)
        self.measurements['totalizer'] += self.measurements['volumetric_flow'] / 3600  # Increment totalizer
        self.measurements['pressure'] += random.uniform(-0.1, 0.1)
        self.measurements['temperature'] += random.uniform(-1, 1)

        # Calculate energy flow (approximation)
        # E = m * h, where h ≈ 2800 kJ/kg for saturated steam at 10 bar
        enthalpy = 2800  # kJ/kg
        self.measurements['energy_flow'] = \
            (self.measurements['mass_flow'] * enthalpy) / 3600  # kW

        return {
            **self.measurements,
            'timestamp': DeterministicClock.utcnow()
        }

    async def read_historical(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Read historical meter data."""
        history = []
        current_time = start_time

        while current_time <= end_time:
            data = await self.read_measurement()
            data['timestamp'] = current_time
            history.append(data)
            current_time += timedelta(minutes=5)

        return history

    async def reset_totalizer(self) -> bool:
        """Reset totalizer to zero."""
        self.measurements['totalizer'] = 0.0
        return True

    async def start(self):
        """Start mock steam meter server."""
        self.running = True
        print(f"Mock Steam Meter {self.meter_id} started on {self.host}:{self.port}")

    async def stop(self):
        """Stop mock server."""
        self.running = False


class MockPressureSensorServer:
    """
    Mock pressure sensor server (4-20mA analog or digital).

    Simulates:
    - Absolute, gauge, and differential pressure
    - High-frequency sampling (up to 100 Hz)
    - Sensor diagnostics
    - Calibration status
    """

    def __init__(self, host="localhost", port=5030, sensor_id="PS-001"):
        self.host = host
        self.port = port
        self.sensor_id = sensor_id
        self.running = False
        self.config = self._initialize_config()

    def _initialize_config(self):
        """Initialize sensor configuration."""
        return {
            'sensor_id': self.sensor_id,
            'pressure_type': deterministic_random().choice(['absolute', 'gauge', 'differential']),
            'range_min': 0.0,
            'range_max': 20.0,
            'unit': 'bar',
            'sampling_rate': 10,  # Hz
            'accuracy': 0.1,      # % of span
            'current_pressure': 10.5,
            'quality': 'GOOD',
            'last_calibration': DeterministicClock.utcnow() - timedelta(days=180)
        }

    async def read_pressure(self) -> Dict[str, Any]:
        """Read current pressure value."""
        # Add realistic noise
        noise = random.gauss(0, 0.05)
        self.config['current_pressure'] += noise

        # Keep within reasonable bounds
        self.config['current_pressure'] = max(
            self.config['range_min'],
            min(self.config['current_pressure'], self.config['range_max'])
        )

        return {
            'sensor_id': self.config['sensor_id'],
            'timestamp': DeterministicClock.utcnow(),
            'pressure': self.config['current_pressure'],
            'pressure_type': self.config['pressure_type'],
            'unit': self.config['unit'],
            'quality': self.config['quality']
        }

    async def read_high_frequency(self, duration_seconds: int = 1) -> List[Dict[str, Any]]:
        """Read high-frequency pressure data."""
        samples = []
        sample_interval = 1.0 / self.config['sampling_rate']
        num_samples = int(duration_seconds * self.config['sampling_rate'])

        start_time = DeterministicClock.utcnow()

        for i in range(num_samples):
            timestamp = start_time + timedelta(seconds=i * sample_interval)
            pressure = self.config['current_pressure'] + random.gauss(0, 0.02)

            samples.append({
                'sensor_id': self.config['sensor_id'],
                'timestamp': timestamp,
                'pressure': pressure,
                'unit': self.config['unit']
            })

        return samples

    async def get_diagnostics(self) -> Dict[str, Any]:
        """Get sensor diagnostics."""
        days_since_cal = (DeterministicClock.utcnow() - self.config['last_calibration']).days

        return {
            'sensor_id': self.config['sensor_id'],
            'status': 'HEALTHY',
            'quality': self.config['quality'],
            'days_since_calibration': days_since_cal,
            'calibration_due': days_since_cal > 365,
            'drift_estimate': random.uniform(0, 0.5),  # % of span
            'temperature': 25.0 + random.uniform(-5, 5)  # Ambient temp
        }

    async def start(self):
        """Start mock pressure sensor server."""
        self.running = True
        print(f"Mock Pressure Sensor {self.sensor_id} started on {self.host}:{self.port}")

    async def stop(self):
        """Stop mock server."""
        self.running = False


class MockTemperatureSensorServer:
    """
    Mock temperature sensor server (RTD, thermocouple).

    Simulates:
    - PT100/PT1000 RTD sensors
    - K/J-type thermocouples
    - Cold junction compensation
    - Sensor linearization
    """

    def __init__(self, host="localhost", port=5040, sensor_id="TS-001"):
        self.host = host
        self.port = port
        self.sensor_id = sensor_id
        self.running = False
        self.config = self._initialize_config()

    def _initialize_config(self):
        """Initialize sensor configuration."""
        return {
            'sensor_id': self.sensor_id,
            'sensor_type': deterministic_random().choice(['PT100', 'K-Type', 'J-Type']),
            'range_min': -50.0,
            'range_max': 500.0,
            'unit': 'degC',
            'current_temperature': 184.0,
            'quality': 'GOOD',
            'last_calibration': DeterministicClock.utcnow() - timedelta(days=200)
        }

    async def read_temperature(self) -> Dict[str, Any]:
        """Read current temperature value."""
        # Add realistic noise based on sensor type
        if self.config['sensor_type'] == 'PT100':
            noise = random.gauss(0, 0.1)  # Very accurate
        else:
            noise = random.gauss(0, 0.5)  # Less accurate thermocouples

        self.config['current_temperature'] += noise

        return {
            'sensor_id': self.config['sensor_id'],
            'timestamp': DeterministicClock.utcnow(),
            'temperature': self.config['current_temperature'],
            'sensor_type': self.config['sensor_type'],
            'unit': self.config['unit'],
            'quality': self.config['quality']
        }

    async def start(self):
        """Start mock temperature sensor server."""
        self.running = True
        print(f"Mock Temperature Sensor {self.sensor_id} started on {self.host}:{self.port}")

    async def stop(self):
        """Stop mock server."""
        self.running = False


class MockMQTTBroker:
    """
    Mock MQTT broker for real-time messaging.

    Simulates:
    - Topic subscription/publishing
    - QoS levels
    - Retained messages
    - Will messages
    """

    def __init__(self, host="localhost", port=1883):
        self.host = host
        self.port = port
        self.topics = defaultdict(list)
        self.subscribers = defaultdict(list)
        self.retained_messages = {}
        self.running = False

    async def publish(
        self,
        topic: str,
        message: str,
        qos: int = 0,
        retain: bool = False
    ):
        """Publish message to topic."""
        message_data = {
            'timestamp': DeterministicClock.utcnow(),
            'topic': topic,
            'payload': message,
            'qos': qos
        }

        self.topics[topic].append(message_data)

        if retain:
            self.retained_messages[topic] = message_data

        # Notify subscribers
        for callback in self.subscribers.get(topic, []):
            try:
                await callback(topic, message)
            except Exception as e:
                print(f"Subscriber notification error: {e}")

    async def subscribe(self, topic: str, callback: Callable):
        """Subscribe to topic."""
        self.subscribers[topic].append(callback)

        # Send retained message if exists
        if topic in self.retained_messages:
            msg = self.retained_messages[topic]
            await callback(topic, msg['payload'])

    async def unsubscribe(self, topic: str, callback: Callable):
        """Unsubscribe from topic."""
        if topic in self.subscribers and callback in self.subscribers[topic]:
            self.subscribers[topic].remove(callback)

    async def get_messages(self, topic: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent messages for topic."""
        return self.topics[topic][-limit:]

    async def start(self):
        """Start mock MQTT broker."""
        self.running = True
        print(f"Mock MQTT Broker started on {self.host}:{self.port}")

        # Start publishing simulated data
        asyncio.create_task(self._publish_loop())

    async def _publish_loop(self):
        """Publish simulated steam system data."""
        while self.running:
            # Publish steam header data
            steam_data = {
                'timestamp': DeterministicClock.utcnow().isoformat(),
                'pressure': 10.5 + random.uniform(-0.3, 0.3),
                'temperature': 184.0 + random.uniform(-2, 2),
                'flow_rate': 50.0 + random.uniform(-2, 2),
                'quality': 0.96 + random.uniform(-0.01, 0.01)
            }
            await self.publish('steam/header/data', json.dumps(steam_data))

            # Publish leak detection alerts (randomly)
            if deterministic_random().random() < 0.05:  # 5% chance
                leak_alert = {
                    'timestamp': DeterministicClock.utcnow().isoformat(),
                    'location': f"Valve-{deterministic_random().randint(1, 50)}",
                    'severity': deterministic_random().choice(['low', 'medium', 'high']),
                    'estimated_loss_kg_hr': random.uniform(5, 50)
                }
                await self.publish('steam/alerts/leak', json.dumps(leak_alert))

            await asyncio.sleep(5)

    async def stop(self):
        """Stop mock broker."""
        self.running = False


# Main function to start all mock servers
async def start_all_mock_servers():
    """Start all mock servers for testing."""
    servers = {
        'opcua': MockOPCUAServer(),
        'modbus': MockModbusServer(),
        'steam_meter_1': MockSteamMeterServer(port=5020, meter_id="SM-001"),
        'steam_meter_2': MockSteamMeterServer(port=5021, meter_id="SM-002"),
        'pressure_sensor_1': MockPressureSensorServer(port=5030, sensor_id="PS-001"),
        'pressure_sensor_2': MockPressureSensorServer(port=5031, sensor_id="PS-002"),
        'temperature_sensor_1': MockTemperatureSensorServer(port=5040, sensor_id="TS-001"),
        'mqtt': MockMQTTBroker()
    }

    for name, server in servers.items():
        await server.start()

    return servers


async def stop_all_mock_servers(servers: Dict[str, Any]):
    """Stop all mock servers."""
    for name, server in servers.items():
        await server.stop()


if __name__ == "__main__":
    async def main():
        servers = await start_all_mock_servers()
        print("\nAll mock servers running. Press Ctrl+C to stop...")

        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping servers...")
            await stop_all_mock_servers(servers)

    asyncio.run(main())
