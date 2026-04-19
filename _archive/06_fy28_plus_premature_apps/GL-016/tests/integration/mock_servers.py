# -*- coding: utf-8 -*-
"""
Mock servers for integration testing GL-016 WATERGUARD.

Provides mock implementations of:
- SCADA server
- Water analyzers
- Chemical dosing systems
- ERP system
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from unittest.mock import AsyncMock


logger = logging.getLogger(__name__)


class MockSCADAServer:
    """Mock SCADA server for testing."""

    def __init__(self, host: str = 'localhost', port: int = 4840):
        self.host = host
        self.port = port
        self.is_running = False
        self.connected_clients = []
        self.tag_values = {}
        self.alarms = []
        self.historical_data = {}

    async def start(self):
        """Start the mock SCADA server."""
        self.is_running = True
        self._initialize_default_tags()
        logger.info(f"Mock SCADA server started on {self.host}:{self.port}")

    async def stop(self):
        """Stop the mock SCADA server."""
        self.is_running = False
        self.connected_clients.clear()
        logger.info("Mock SCADA server stopped")

    def _initialize_default_tags(self):
        """Initialize default tag values."""
        self.tag_values = {
            'BOILER_PRESSURE': {'value': 40.5, 'quality': 'GOOD', 'unit': 'bar', 'timestamp': datetime.utcnow()},
            'FEEDWATER_TEMP': {'value': 105.2, 'quality': 'GOOD', 'unit': 'C', 'timestamp': datetime.utcnow()},
            'STEAM_TEMP': {'value': 250.8, 'quality': 'GOOD', 'unit': 'C', 'timestamp': datetime.utcnow()},
            'FEEDWATER_FLOW': {'value': 45.5, 'quality': 'GOOD', 'unit': 'm3/hr', 'timestamp': datetime.utcnow()},
            'BLOWDOWN_FLOW': {'value': 2.3, 'quality': 'GOOD', 'unit': 'm3/hr', 'timestamp': datetime.utcnow()},
            'BOILER_LEVEL': {'value': 75.0, 'quality': 'GOOD', 'unit': '%', 'timestamp': datetime.utcnow()},
            'CONDUCTIVITY': {'value': 1200.0, 'quality': 'GOOD', 'unit': 'uS/cm', 'timestamp': datetime.utcnow()},
            'PH_SENSOR': {'value': 8.5, 'quality': 'GOOD', 'unit': 'pH', 'timestamp': datetime.utcnow()},
        }

    async def connect_client(self, client_id: str) -> bool:
        """Connect a client to the SCADA server."""
        if not self.is_running:
            return False
        self.connected_clients.append(client_id)
        logger.info(f"Client {client_id} connected to SCADA server")
        return True

    async def disconnect_client(self, client_id: str):
        """Disconnect a client from the SCADA server."""
        if client_id in self.connected_clients:
            self.connected_clients.remove(client_id)
            logger.info(f"Client {client_id} disconnected from SCADA server")

    async def read_tag(self, tag_name: str) -> Optional[Dict[str, Any]]:
        """Read a tag value."""
        if tag_name in self.tag_values:
            return self.tag_values[tag_name]
        return None

    async def write_tag(self, tag_name: str, value: float) -> bool:
        """Write a tag value."""
        if tag_name in self.tag_values:
            self.tag_values[tag_name]['value'] = value
            self.tag_values[tag_name]['timestamp'] = datetime.utcnow()
            logger.info(f"Tag {tag_name} set to {value}")
            return True
        return False

    async def read_multiple_tags(self, tag_names: List[str]) -> Dict[str, Any]:
        """Read multiple tags."""
        result = {}
        for tag_name in tag_names:
            if tag_name in self.tag_values:
                result[tag_name] = self.tag_values[tag_name]
        return result

    async def subscribe_to_tag(self, tag_name: str, callback) -> bool:
        """Subscribe to tag changes."""
        # Mock implementation - always succeeds
        logger.info(f"Subscribed to tag {tag_name}")
        return True

    async def get_historical_data(self, tag_name: str, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Get historical data for a tag."""
        # Mock implementation - generate sample data
        data = []
        current = start_time
        while current <= end_time:
            data.append({
                'timestamp': current,
                'value': self.tag_values.get(tag_name, {}).get('value', 0.0),
                'quality': 'GOOD'
            })
            current += timedelta(minutes=5)
        return data

    async def raise_alarm(self, alarm_type: str, message: str, severity: str):
        """Raise an alarm."""
        alarm = {
            'timestamp': datetime.utcnow(),
            'type': alarm_type,
            'message': message,
            'severity': severity
        }
        self.alarms.append(alarm)
        logger.warning(f"Alarm raised: {alarm}")

    def get_alarms(self) -> List[Dict]:
        """Get all active alarms."""
        return self.alarms


class MockWaterAnalyzer:
    """Mock water analyzer device."""

    def __init__(self, analyzer_id: str):
        self.analyzer_id = analyzer_id
        self.is_running = False
        self.is_calibrated = True
        self.last_calibration = datetime.utcnow() - timedelta(hours=12)

    async def start(self):
        """Start the analyzer."""
        self.is_running = True
        logger.info(f"Water analyzer {self.analyzer_id} started")

    async def stop(self):
        """Stop the analyzer."""
        self.is_running = False
        logger.info(f"Water analyzer {self.analyzer_id} stopped")

    async def get_analysis(self) -> Dict[str, Any]:
        """Get water chemistry analysis."""
        if not self.is_running:
            raise RuntimeError(f"Analyzer {self.analyzer_id} is not running")

        return {
            'analyzer_id': self.analyzer_id,
            'timestamp': datetime.utcnow().isoformat(),
            'measurements': {
                'ph': 8.5,
                'alkalinity_ppm': 250.0,
                'hardness_ppm': 180.0,
                'calcium_ppm': 50.0,
                'magnesium_ppm': 30.0,
                'chloride_ppm': 150.0,
                'sulfate_ppm': 100.0,
                'silica_ppm': 25.0,
                'tds_ppm': 800.0,
                'conductivity_us_cm': 1200.0,
                'temperature_c': 85.0,
                'dissolved_oxygen_ppm': 0.02,
                'iron_ppm': 0.05,
                'copper_ppm': 0.01,
                'phosphate_ppm': 15.0,
                'sulfite_ppm': 20.0,
                'hydrazine_ppm': 0.05,
            },
            'status': 'OK',
            'calibration_status': 'VALID' if self.is_calibrated else 'EXPIRED',
            'last_calibration': self.last_calibration.isoformat()
        }

    async def calibrate(self, standard_solutions: Dict[str, float]) -> bool:
        """Calibrate the analyzer."""
        self.is_calibrated = True
        self.last_calibration = datetime.utcnow()
        logger.info(f"Analyzer {self.analyzer_id} calibrated")
        return True

    async def get_status(self) -> Dict[str, Any]:
        """Get analyzer status."""
        return {
            'analyzer_id': self.analyzer_id,
            'status': 'OK' if self.is_running else 'OFFLINE',
            'is_calibrated': self.is_calibrated,
            'last_calibration': self.last_calibration.isoformat(),
            'maintenance_due': (datetime.utcnow() + timedelta(days=30)).isoformat()
        }


class MockChemicalDosingSystem:
    """Mock chemical dosing system."""

    def __init__(self, system_id: str):
        self.system_id = system_id
        self.is_running = False
        self.pumps = {
            'phosphate': {'status': 'IDLE', 'flow_rate_l_hr': 0.0},
            'sulfite': {'status': 'IDLE', 'flow_rate_l_hr': 0.0},
            'caustic': {'status': 'IDLE', 'flow_rate_l_hr': 0.0},
            'inhibitor': {'status': 'IDLE', 'flow_rate_l_hr': 0.0}
        }
        self.inventory = {
            'phosphate': {'volume_liters': 500.0, 'concentration_percent': 30.0},
            'sulfite': {'volume_liters': 300.0, 'concentration_percent': 25.0},
            'caustic': {'volume_liters': 400.0, 'concentration_percent': 50.0},
            'inhibitor': {'volume_liters': 200.0, 'concentration_percent': 40.0}
        }

    async def start(self):
        """Start the dosing system."""
        self.is_running = True
        logger.info(f"Chemical dosing system {self.system_id} started")

    async def stop(self):
        """Stop the dosing system."""
        self.is_running = False
        for pump in self.pumps.values():
            pump['status'] = 'STOPPED'
        logger.info(f"Chemical dosing system {self.system_id} stopped")

    async def dose_chemical(self, chemical: str, flow_rate_l_hr: float) -> bool:
        """Start dosing a chemical."""
        if not self.is_running:
            return False

        if chemical not in self.pumps:
            return False

        self.pumps[chemical]['status'] = 'RUNNING'
        self.pumps[chemical]['flow_rate_l_hr'] = flow_rate_l_hr
        logger.info(f"Dosing {chemical} at {flow_rate_l_hr} L/hr")
        return True

    async def stop_dosing(self, chemical: str) -> bool:
        """Stop dosing a chemical."""
        if chemical in self.pumps:
            self.pumps[chemical]['status'] = 'IDLE'
            self.pumps[chemical]['flow_rate_l_hr'] = 0.0
            logger.info(f"Stopped dosing {chemical}")
            return True
        return False

    async def get_pump_status(self, chemical: str) -> Optional[Dict[str, Any]]:
        """Get pump status."""
        return self.pumps.get(chemical)

    async def get_chemical_inventory(self) -> Dict[str, Dict[str, float]]:
        """Get chemical inventory."""
        return self.inventory

    async def update_inventory(self, chemical: str, volume_liters: float):
        """Update chemical inventory after refill."""
        if chemical in self.inventory:
            self.inventory[chemical]['volume_liters'] = volume_liters
            logger.info(f"Updated {chemical} inventory to {volume_liters} L")


class MockERPSystem:
    """Mock ERP system."""

    def __init__(self, host: str = 'localhost', port: int = 8000):
        self.host = host
        self.port = port
        self.is_running = False
        self.chemical_costs = {
            'phosphate': 5.50,
            'sulfite': 4.25,
            'caustic': 3.75,
            'hydrazine': 12.00,
            'inhibitor': 8.50
        }
        self.work_orders = []

    async def start(self):
        """Start the ERP system."""
        self.is_running = True
        logger.info(f"Mock ERP system started on {self.host}:{self.port}")

    async def stop(self):
        """Stop the ERP system."""
        self.is_running = False
        logger.info("Mock ERP system stopped")

    async def get_chemical_cost(self, chemical: str) -> Optional[float]:
        """Get chemical cost per liter."""
        return self.chemical_costs.get(chemical)

    async def get_water_cost(self) -> float:
        """Get water cost per m3."""
        return 2.50

    async def get_energy_cost(self) -> float:
        """Get energy cost per kWh."""
        return 0.12

    async def create_work_order(self, work_order: Dict[str, Any]) -> str:
        """Create a maintenance work order."""
        wo_id = f"WO-{len(self.work_orders) + 1:04d}"
        work_order['work_order_id'] = wo_id
        work_order['created_at'] = datetime.utcnow().isoformat()
        work_order['status'] = 'PENDING'
        self.work_orders.append(work_order)
        logger.info(f"Created work order {wo_id}")
        return wo_id

    async def get_maintenance_schedule(self, equipment_id: str) -> List[Dict]:
        """Get maintenance schedule for equipment."""
        return [
            {
                'equipment_id': equipment_id,
                'maintenance_type': 'preventive',
                'scheduled_date': (datetime.utcnow() + timedelta(days=30)).isoformat(),
                'description': 'Routine inspection'
            }
        ]

    def get_work_orders(self) -> List[Dict]:
        """Get all work orders."""
        return self.work_orders
