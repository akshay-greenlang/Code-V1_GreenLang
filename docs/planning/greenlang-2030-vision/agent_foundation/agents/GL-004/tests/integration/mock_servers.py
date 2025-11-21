# -*- coding: utf-8 -*-
"""
Mock servers for GL-004 integration testing.

Provides realistic simulations of:
- Modbus server (burner controller, O2 analyzer, temperature sensors)
- MQTT broker (emissions monitor)
- HTTP API (flame scanner)
"""

import asyncio
import threading
import time
import random
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np

from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext
from pymodbus.datastore import ModbusSequentialDataBlock
from pymodbus.server.async_io import StartAsyncTcpServer
from pymodbus.device import ModbusDeviceIdentification

import paho.mqtt.client as mqtt
from flask import Flask, jsonify, request
from werkzeug.serving import make_server
from greenlang.determinism import DeterministicClock
from greenlang.determinism import deterministic_random


class BurnerState(Enum):
    """Burner operational states."""
    OFF = 'off'
    STARTUP = 'startup'
    RUNNING = 'running'
    OPTIMIZATION = 'optimization'
    SHUTDOWN = 'shutdown'
    EMERGENCY = 'emergency'


@dataclass
class BurnerParameters:
    """Burner operating parameters."""
    fuel_flow: float  # kg/h
    air_flow: float  # m³/h
    o2_level: float  # %
    temperature: float  # °C
    co_emissions: float  # ppm
    nox_emissions: float  # ppm
    efficiency: float  # %
    flame_intensity: float  # %
    flame_stability: float  # %
    state: BurnerState = BurnerState.RUNNING


class MockModbusServer:
    """
    Mock Modbus server simulating burner controller.

    Provides realistic equipment behavior including:
    - Gradual setpoint transitions
    - Safety interlocks
    - Equipment limits
    - Response delays
    """

    def __init__(self, host: str = 'localhost', port: int = 5502):
        """Initialize mock Modbus server."""
        self.host = host
        self.port = port
        self.context = None
        self.server_task = None
        self.running = False

        # Current burner state
        self.burner = BurnerParameters(
            fuel_flow=10.0,
            air_flow=120.0,
            o2_level=3.0,
            temperature=850.0,
            co_emissions=50.0,
            nox_emissions=120.0,
            efficiency=87.5,
            flame_intensity=85.0,
            flame_stability=92.0
        )

        # Setpoints
        self.setpoint_fuel_flow = 10.0
        self.setpoint_air_flow = 120.0

        # Equipment limits
        self.limits = {
            'fuel_flow': {'min': 5.0, 'max': 20.0, 'ramp_rate': 0.5},  # kg/h per second
            'air_flow': {'min': 50.0, 'max': 200.0, 'ramp_rate': 5.0},  # m³/h per second
            'o2_level': {'min': 1.0, 'max': 6.0},
            'temperature': {'min': 600.0, 'max': 1200.0},
            'co_emissions': {'alarm': 200.0, 'shutdown': 500.0},
            'nox_emissions': {'alarm': 180.0, 'shutdown': 300.0}
        }

        # Safety interlocks
        self.safety_interlock_enabled = True
        self.emergency_stop = False

        # Simulation thread
        self.sim_thread = None

    async def start(self):
        """Start the Modbus server."""
        # Initialize data store
        store = ModbusSlaveContext(
            di=ModbusSequentialDataBlock(0, [0] * 100),
            co=ModbusSequentialDataBlock(0, [0] * 100),
            hr=ModbusSequentialDataBlock(0, [0] * 200),
            ir=ModbusSequentialDataBlock(0, [0] * 100)
        )

        self.context = ModbusServerContext(slaves=store, single=True)

        # Update initial values
        self._update_registers()

        # Start server
        self.server_task = asyncio.create_task(
            StartAsyncTcpServer(
                context=self.context,
                address=(self.host, self.port)
            )
        )

        self.running = True

        # Start simulation thread
        self.sim_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.sim_thread.start()

        await asyncio.sleep(0.5)  # Wait for server to start

    async def stop(self):
        """Stop the Modbus server."""
        self.running = False

        if self.server_task:
            self.server_task.cancel()
            try:
                await self.server_task
            except asyncio.CancelledError:
                pass

    def _update_registers(self):
        """Update Modbus registers with current values."""
        if not self.context:
            return

        store = self.context[0]

        # Update holding registers
        values = [
            int(self.burner.fuel_flow * 100),  # 0: Fuel flow (x100)
            int(self.burner.air_flow * 10),    # 1: Air flow (x10)
            int(self.burner.o2_level * 100),   # 2: O2 level (x100)
            int(self.burner.temperature * 10), # 3: Temperature (x10)
            int(self.burner.co_emissions),     # 4: CO emissions
            int(self.burner.nox_emissions),    # 5: NOx emissions
            int(self.burner.efficiency * 100), # 6: Efficiency (x100)
            int(self.setpoint_fuel_flow * 100), # 7: Setpoint fuel flow
            int(self.setpoint_air_flow * 10),  # 8: Setpoint air flow
            int(self.safety_interlock_enabled), # 9: Safety interlock
            int(self.emergency_stop),          # 10: Emergency stop
            int(self.burner.flame_intensity * 100), # 11: Flame intensity
            int(self.burner.flame_stability * 100), # 12: Flame stability
            self.burner.state.value.__hash__() % 65536, # 13: State hash
        ]

        store.setValues(3, 0, values)

    def _simulation_loop(self):
        """Main simulation loop for burner behavior."""
        while self.running:
            try:
                # Update burner parameters
                self._update_burner_state()

                # Apply setpoint changes gradually
                self._apply_setpoints()

                # Simulate disturbances
                self._apply_disturbances()

                # Check safety limits
                self._check_safety_limits()

                # Update Modbus registers
                self._update_registers()

                time.sleep(0.1)  # 100ms update cycle

            except Exception as e:
                print(f"Simulation error: {e}")

    def _update_burner_state(self):
        """Update burner state based on current parameters."""
        # Calculate O2 level based on fuel/air ratio
        stoichiometric_ratio = 12.0  # Simplified
        actual_ratio = self.burner.air_flow / self.burner.fuel_flow
        excess_air = (actual_ratio / stoichiometric_ratio - 1) * 100

        # O2 level correlates with excess air
        self.burner.o2_level = 0.21 * excess_air / (100 + excess_air) * 100

        # Calculate efficiency based on O2 level
        optimal_o2 = 3.0
        efficiency_loss = abs(self.burner.o2_level - optimal_o2) * 3
        self.burner.efficiency = max(70, min(95, 90 - efficiency_loss))

        # Calculate emissions based on O2 level and temperature
        if self.burner.o2_level < 2.0:
            # Low O2 - high CO
            self.burner.co_emissions = 200 * (2.0 - self.burner.o2_level)
            self.burner.nox_emissions = 80
        elif self.burner.o2_level > 4.0:
            # High O2 - high NOx
            self.burner.co_emissions = 20
            self.burner.nox_emissions = 100 + 20 * (self.burner.o2_level - 4.0)
        else:
            # Optimal range
            self.burner.co_emissions = 30 + random.gauss(0, 5)
            self.burner.nox_emissions = 100 + random.gauss(0, 10)

        # Temperature depends on fuel flow
        self.burner.temperature = 600 + self.burner.fuel_flow * 30 + random.gauss(0, 10)

        # Flame characteristics
        if self.burner.fuel_flow < 7.0:
            # Unstable at low flow
            self.burner.flame_intensity = 60 + random.gauss(0, 5)
            self.burner.flame_stability = 70 + random.gauss(0, 8)
        else:
            self.burner.flame_intensity = 85 + random.gauss(0, 2)
            self.burner.flame_stability = 92 + random.gauss(0, 2)

    def _apply_setpoints(self):
        """Gradually apply setpoint changes."""
        if self.emergency_stop:
            # Emergency stop - immediate shutdown
            self.burner.fuel_flow = 0
            self.burner.air_flow = 0
            return

        if not self.safety_interlock_enabled:
            # Safety interlock disabled - no changes allowed
            return

        # Apply fuel flow setpoint with ramping
        fuel_diff = self.setpoint_fuel_flow - self.burner.fuel_flow
        max_change = self.limits['fuel_flow']['ramp_rate'] * 0.1  # Per cycle

        if abs(fuel_diff) > max_change:
            fuel_diff = max_change if fuel_diff > 0 else -max_change

        self.burner.fuel_flow = max(
            self.limits['fuel_flow']['min'],
            min(self.limits['fuel_flow']['max'],
                self.burner.fuel_flow + fuel_diff)
        )

        # Apply air flow setpoint with ramping
        air_diff = self.setpoint_air_flow - self.burner.air_flow
        max_change = self.limits['air_flow']['ramp_rate'] * 0.1  # Per cycle

        if abs(air_diff) > max_change:
            air_diff = max_change if air_diff > 0 else -max_change

        self.burner.air_flow = max(
            self.limits['air_flow']['min'],
            min(self.limits['air_flow']['max'],
                self.burner.air_flow + air_diff)
        )

    def _apply_disturbances(self):
        """Apply realistic disturbances to parameters."""
        # Random walk disturbances
        self.burner.fuel_flow += random.gauss(0, 0.02)
        self.burner.air_flow += random.gauss(0, 0.5)
        self.burner.temperature += random.gauss(0, 2)

    def _check_safety_limits(self):
        """Check safety limits and trigger alarms/shutdowns."""
        # CO emissions check
        if self.burner.co_emissions > self.limits['co_emissions']['shutdown']:
            self.emergency_stop = True
            self.burner.state = BurnerState.EMERGENCY
        elif self.burner.co_emissions > self.limits['co_emissions']['alarm']:
            # Alarm state
            pass

        # NOx emissions check
        if self.burner.nox_emissions > self.limits['nox_emissions']['shutdown']:
            self.emergency_stop = True
            self.burner.state = BurnerState.EMERGENCY

        # Temperature limits
        if self.burner.temperature > self.limits['temperature']['max']:
            self.safety_interlock_enabled = False

    def write_setpoint(self, register: int, value: int) -> bool:
        """Write setpoint to Modbus register."""
        if not self.safety_interlock_enabled:
            return False

        if register == 7:  # Fuel flow setpoint
            fuel_flow = value / 100.0
            if self.limits['fuel_flow']['min'] <= fuel_flow <= self.limits['fuel_flow']['max']:
                self.setpoint_fuel_flow = fuel_flow
                return True

        elif register == 8:  # Air flow setpoint
            air_flow = value / 10.0
            if self.limits['air_flow']['min'] <= air_flow <= self.limits['air_flow']['max']:
                self.setpoint_air_flow = air_flow
                return True

        return False


class MockMQTTBroker:
    """
    Mock MQTT broker for emissions monitoring.

    Simulates real-time emissions data with realistic patterns.
    """

    def __init__(self, host: str = 'localhost', port: int = 1883):
        """Initialize mock MQTT broker."""
        self.host = host
        self.port = port
        self.running = False
        self.messages = {}
        self.subscribers = {}
        self.sim_thread = None

        # Emissions baseline
        self.baseline = {
            'co': 50.0,
            'nox': 120.0,
            'sox': 15.0,
            'o2': 3.5,
            'particulates': 25.0
        }

        # Emissions state
        self.current_values = self.baseline.copy()

    def start(self):
        """Start the MQTT broker."""
        self.running = True
        self.sim_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.sim_thread.start()

    def stop(self):
        """Stop the MQTT broker."""
        self.running = False

    def publish(self, topic: str, payload: Dict[str, Any]):
        """Publish message to topic."""
        self.messages[topic] = json.dumps(payload)

        # Notify subscribers
        if topic in self.subscribers:
            for callback in self.subscribers[topic]:
                try:
                    callback(topic, payload)
                except Exception as e:
                    print(f"Subscriber callback error: {e}")

    def subscribe(self, topic: str, callback: Callable):
        """Subscribe to topic."""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)

    def unsubscribe(self, topic: str, callback: Callable):
        """Unsubscribe from topic."""
        if topic in self.subscribers and callback in self.subscribers[topic]:
            self.subscribers[topic].remove(callback)

    def get_latest(self, topic: str) -> Optional[Dict[str, Any]]:
        """Get latest message from topic."""
        if topic in self.messages:
            return json.loads(self.messages[topic])
        return None

    def _simulation_loop(self):
        """Simulate emissions data."""
        cycle = 0

        while self.running:
            try:
                # Apply time-based patterns
                hour_of_day = (cycle // 36) % 24  # Assuming 100ms cycles

                # Diurnal variation
                diurnal_factor = 1.0 + 0.2 * np.sin((hour_of_day - 6) * np.pi / 12)

                # Generate emissions with patterns
                for pollutant, baseline in self.baseline.items():
                    # Base value with diurnal variation
                    value = baseline * diurnal_factor

                    # Add noise
                    if pollutant == 'co':
                        value += random.gauss(0, 5)
                        # Occasional spikes
                        if deterministic_random().random() < 0.01:
                            value *= 2.5
                    elif pollutant == 'nox':
                        value += random.gauss(0, 10)
                    elif pollutant == 'o2':
                        value += random.gauss(0, 0.2)
                    else:
                        value += random.gauss(0, 2)

                    self.current_values[pollutant] = max(0, value)

                    # Publish to topic
                    topic = f'emissions/{pollutant}'
                    payload = {
                        'value': self.current_values[pollutant],
                        'unit': self._get_unit(pollutant),
                        'timestamp': DeterministicClock.now().isoformat(),
                        'quality': 'GOOD' if deterministic_random().random() > 0.05 else 'DEGRADED'
                    }

                    self.publish(topic, payload)

                # Also publish aggregate message
                self.publish('emissions/all', {
                    'timestamp': DeterministicClock.now().isoformat(),
                    'values': self.current_values.copy(),
                    'alarm_state': self._check_alarms()
                })

                cycle += 1
                time.sleep(0.1)  # 100ms update rate

            except Exception as e:
                print(f"MQTT simulation error: {e}")

    def _get_unit(self, pollutant: str) -> str:
        """Get unit for pollutant."""
        units = {
            'co': 'ppm',
            'nox': 'ppm',
            'sox': 'ppm',
            'o2': '%',
            'particulates': 'mg/m³'
        }
        return units.get(pollutant, '')

    def _check_alarms(self) -> str:
        """Check for alarm conditions."""
        if self.current_values['co'] > 200:
            return 'HIGH_CO'
        elif self.current_values['nox'] > 180:
            return 'HIGH_NOX'
        elif self.current_values['o2'] < 1.0 or self.current_values['o2'] > 5.0:
            return 'O2_OUT_OF_RANGE'
        return 'NORMAL'

    def inject_fault(self, pollutant: str, value: float):
        """Inject fault for testing."""
        if pollutant in self.current_values:
            self.current_values[pollutant] = value


class MockFlameScanner:
    """
    Mock HTTP API server for flame scanner.

    Provides flame intensity, stability, and spectral data.
    """

    def __init__(self, host: str = 'localhost', port: int = 5003):
        """Initialize mock flame scanner API."""
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.server = None
        self.server_thread = None

        # Flame state
        self.flame_state = {
            'intensity': 85.0,
            'stability': 92.0,
            'color_temp': 1850,
            'uv_intensity': 75.0,
            'ir_intensity': 88.0,
            'flicker_freq': 12.5
        }

        # History buffer
        self.history = []
        self.max_history = 3600  # 1 hour at 1Hz

        self._setup_routes()

    def _setup_routes(self):
        """Setup Flask routes."""

        @self.app.route('/api/flame/status', methods=['GET'])
        def get_status():
            """Get current flame status."""
            return jsonify({
                'status': self._get_flame_status(),
                'intensity': self.flame_state['intensity'],
                'stability': self.flame_state['stability'],
                'color_temp': self.flame_state['color_temp'],
                'uv_intensity': self.flame_state['uv_intensity'],
                'ir_intensity': self.flame_state['ir_intensity'],
                'flicker_freq': self.flame_state['flicker_freq'],
                'timestamp': DeterministicClock.now().isoformat()
            })

        @self.app.route('/api/flame/history', methods=['GET'])
        def get_history():
            """Get flame history."""
            limit = request.args.get('limit', 60, type=int)
            return jsonify(self.history[-limit:])

        @self.app.route('/api/flame/spectral', methods=['GET'])
        def get_spectral():
            """Get spectral analysis."""
            # Generate mock spectral data
            wavelengths = list(range(300, 900, 10))  # 300-900nm
            intensities = [
                self._spectral_intensity(w) for w in wavelengths
            ]

            return jsonify({
                'wavelengths': wavelengths,
                'intensities': intensities,
                'peak_wavelength': 589,  # Sodium D-line
                'timestamp': DeterministicClock.now().isoformat()
            })

        @self.app.route('/api/flame/control', methods=['POST'])
        def control():
            """Control flame scanner settings."""
            data = request.json

            if 'sampling_rate' in data:
                # Adjust sampling rate
                pass

            if 'gain' in data:
                # Adjust sensor gain
                pass

            return jsonify({'status': 'OK'})

    def start(self):
        """Start the HTTP server."""
        self.server = make_server(self.host, self.port, self.app)
        self.server_thread = threading.Thread(
            target=self.server.serve_forever,
            daemon=True
        )
        self.server_thread.start()

        # Start simulation
        sim_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        sim_thread.start()

    def stop(self):
        """Stop the HTTP server."""
        if self.server:
            self.server.shutdown()

    def _simulation_loop(self):
        """Simulate flame behavior."""
        while self.server:
            try:
                # Update flame state
                self._update_flame_state()

                # Record history
                self.history.append({
                    'timestamp': DeterministicClock.now().isoformat(),
                    'intensity': self.flame_state['intensity'],
                    'stability': self.flame_state['stability'],
                    'color_temp': self.flame_state['color_temp']
                })

                # Limit history size
                if len(self.history) > self.max_history:
                    self.history = self.history[-self.max_history:]

                time.sleep(1)  # 1Hz update

            except Exception as e:
                print(f"Flame scanner simulation error: {e}")
                break

    def _update_flame_state(self):
        """Update flame state with realistic behavior."""
        # Base variations
        self.flame_state['intensity'] = 85 + random.gauss(0, 2)
        self.flame_state['stability'] = 92 + random.gauss(0, 1.5)

        # Color temperature varies with combustion quality
        self.flame_state['color_temp'] = 1850 + random.gauss(0, 50)

        # UV/IR intensities
        self.flame_state['uv_intensity'] = 75 + random.gauss(0, 3)
        self.flame_state['ir_intensity'] = 88 + random.gauss(0, 2)

        # Flicker frequency
        self.flame_state['flicker_freq'] = 12.5 + random.gauss(0, 0.5)

    def _get_flame_status(self) -> str:
        """Determine flame status."""
        if self.flame_state['intensity'] < 50:
            return 'WEAK'
        elif self.flame_state['intensity'] > 95:
            return 'EXCESSIVE'
        elif self.flame_state['stability'] < 80:
            return 'UNSTABLE'
        else:
            return 'STABLE'

    def _spectral_intensity(self, wavelength: float) -> float:
        """Generate spectral intensity for wavelength."""
        # Simplified blackbody with emission lines
        base = 100 * np.exp(-(wavelength - 600)**2 / 10000)

        # Add emission lines
        if abs(wavelength - 589) < 5:  # Sodium
            base += 50
        elif abs(wavelength - 766) < 5:  # Potassium
            base += 30

        return base + random.gauss(0, 2)


# Composite Mock Server Manager
class MockServerManager:
    """Manage all mock servers for integration testing."""

    def __init__(self):
        """Initialize server manager."""
        self.modbus_server = None
        self.mqtt_broker = None
        self.flame_scanner = None

    async def start_all(self):
        """Start all mock servers."""
        # Start Modbus server
        self.modbus_server = MockModbusServer()
        await self.modbus_server.start()

        # Start MQTT broker
        self.mqtt_broker = MockMQTTBroker()
        self.mqtt_broker.start()

        # Start flame scanner API
        self.flame_scanner = MockFlameScanner()
        self.flame_scanner.start()

        # Wait for all servers to be ready
        await asyncio.sleep(1)

    async def stop_all(self):
        """Stop all mock servers."""
        if self.modbus_server:
            await self.modbus_server.stop()

        if self.mqtt_broker:
            self.mqtt_broker.stop()

        if self.flame_scanner:
            self.flame_scanner.stop()

    def get_modbus(self) -> MockModbusServer:
        """Get Modbus server instance."""
        return self.modbus_server

    def get_mqtt(self) -> MockMQTTBroker:
        """Get MQTT broker instance."""
        return self.mqtt_broker

    def get_flame_scanner(self) -> MockFlameScanner:
        """Get flame scanner instance."""
        return self.flame_scanner