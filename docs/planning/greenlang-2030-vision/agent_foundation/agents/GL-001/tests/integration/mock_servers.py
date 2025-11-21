# -*- coding: utf-8 -*-
"""
Mock Servers for GL-001 Integration Testing

Provides comprehensive mock implementations of:
- SCADA systems (OPC UA, Modbus TCP/RTU) for multiple plants
- ERP systems (SAP RFC, Oracle REST API)
- Sub-agents (GL-002, GL-003, GL-004, GL-005)
- Multi-plant coordinator
- MQTT message broker
"""

import asyncio
import random
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from aiohttp import web
import hashlib
from greenlang.determinism import deterministic_random


# ==============================================================================
# SCADA MOCK SERVERS
# ==============================================================================

class MockOPCUAServer:
    """Mock OPC UA server for SCADA simulation with multi-plant support."""

    def __init__(self, plant_id: str = "PLANT-001", host: str = "localhost", port: int = 4840):
        self.plant_id = plant_id
        self.host = host
        self.port = port
        self.tags = {}
        self.subscriptions = []
        self.running = False
        self.connection_count = 0
        self._initialize_tags()

    def _initialize_tags(self):
        """Initialize comprehensive mock tag values for process heat plant."""
        self.tags = {
            # Boiler metrics
            f'{self.plant_id}.BOILER.STEAM.PRESSURE': 100.0,  # bar
            f'{self.plant_id}.BOILER.STEAM.TEMPERATURE': 490.0,  # °C
            f'{self.plant_id}.BOILER.STEAM.FLOW': 50.0,  # kg/s
            f'{self.plant_id}.BOILER.EFFICIENCY': 89.5,  # %
            f'{self.plant_id}.BOILER.O2.CONTENT': 3.5,  # %
            f'{self.plant_id}.BOILER.DRUM.LEVEL': 0.0,  # mm
            f'{self.plant_id}.BOILER.FUEL.VALVE.POSITION': 50.0,  # %

            # Heat exchanger metrics
            f'{self.plant_id}.HX.PRIMARY.INLET.TEMP': 500.0,  # °C
            f'{self.plant_id}.HX.PRIMARY.OUTLET.TEMP': 150.0,  # °C
            f'{self.plant_id}.HX.SECONDARY.INLET.TEMP': 25.0,  # °C
            f'{self.plant_id}.HX.SECONDARY.OUTLET.TEMP': 85.0,  # °C
            f'{self.plant_id}.HX.HEAT.TRANSFER.RATE': 75.0,  # MW

            # Steam distribution
            f'{self.plant_id}.STEAM.HEADER.PRESSURE': 98.0,  # bar
            f'{self.plant_id}.STEAM.HEADER.TEMPERATURE': 485.0,  # °C
            f'{self.plant_id}.STEAM.CONSUMER.1.FLOW': 15.0,  # kg/s
            f'{self.plant_id}.STEAM.CONSUMER.2.FLOW': 20.0,  # kg/s
            f'{self.plant_id}.STEAM.CONSUMER.3.FLOW': 15.0,  # kg/s

            # Fuel system
            f'{self.plant_id}.FUEL.GAS.FLOW': 1500.0,  # m³/hr
            f'{self.plant_id}.FUEL.GAS.PRESSURE': 5.0,  # bar
            f'{self.plant_id}.FUEL.GAS.TEMPERATURE': 25.0,  # °C
            f'{self.plant_id}.FUEL.OIL.FLOW': 0.0,  # kg/hr
            f'{self.plant_id}.FUEL.BIOMASS.FLOW': 0.0,  # kg/hr

            # Heat recovery
            f'{self.plant_id}.HEAT.RECOVERY.ECONOMIZER.OUTLET.TEMP': 180.0,  # °C
            f'{self.plant_id}.HEAT.RECOVERY.AIR.PREHEATER.TEMP': 250.0,  # °C
            f'{self.plant_id}.HEAT.RECOVERY.TOTAL.MW': 10.0,  # MW

            # Emissions (continuous monitoring)
            f'{self.plant_id}.CEMS.CO2.CONCENTRATION': 12.5,  # %
            f'{self.plant_id}.CEMS.NOX.CONCENTRATION': 95.0,  # mg/Nm³
            f'{self.plant_id}.CEMS.SO2.CONCENTRATION': 45.0,  # mg/Nm³
            f'{self.plant_id}.CEMS.O2.CONCENTRATION': 4.2,  # %
            f'{self.plant_id}.CEMS.PM.CONCENTRATION': 8.5,  # mg/Nm³
            f'{self.plant_id}.CEMS.CO.CONCENTRATION': 15.0,  # ppm
            f'{self.plant_id}.CEMS.STACK.FLOW': 60000.0,  # Nm³/hr
            f'{self.plant_id}.CEMS.STACK.TEMP': 140.0,  # °C

            # Plant status
            f'{self.plant_id}.PLANT.STATUS': 1,  # 1=Running, 0=Stopped
            f'{self.plant_id}.PLANT.LOAD.PERCENT': 100.0,  # %
            f'{self.plant_id}.PLANT.ALARM.COUNT': 0,
            f'{self.plant_id}.PLANT.WARNING.COUNT': 2,
        }

    async def read_tag(self, tag_name: str) -> Dict[str, Any]:
        """Read tag value with simulated variation and quality."""
        base_value = self.tags.get(tag_name, 0.0)

        # Add realistic random variation (±2%)
        variation = base_value * random.uniform(-0.02, 0.02)
        value = base_value + variation

        return {
            'value': value,
            'quality': 'GOOD',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'tag_name': tag_name
        }

    async def read_tags_batch(self, tag_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """Read multiple tags in batch."""
        results = {}
        for tag_name in tag_names:
            results[tag_name] = await self.read_tag(tag_name)
        return results

    async def write_tag(self, tag_name: str, value: float) -> Dict[str, Any]:
        """Write tag value (for control operations)."""
        if tag_name in self.tags:
            self.tags[tag_name] = value
            return {
                'success': True,
                'tag_name': tag_name,
                'value': value,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        return {
            'success': False,
            'error': f'Tag {tag_name} not found'
        }

    async def subscribe_tags(self, tag_names: List[str], callback) -> str:
        """Subscribe to tag changes."""
        subscription_id = f"SUB-{len(self.subscriptions):04d}"
        subscription = {
            'id': subscription_id,
            'tags': tag_names,
            'callback': callback,
            'created': datetime.now(timezone.utc).isoformat()
        }
        self.subscriptions.append(subscription)
        return subscription_id

    async def start(self):
        """Start mock OPC UA server."""
        self.running = True
        print(f"Mock OPC UA Server started for {self.plant_id} on {self.host}:{self.port}")

    async def stop(self):
        """Stop mock server."""
        self.running = False
        print(f"Mock OPC UA Server stopped for {self.plant_id}")

    def simulate_fault(self, fault_type: str):
        """Simulate various fault conditions for testing."""
        if fault_type == 'low_efficiency':
            self.tags[f'{self.plant_id}.BOILER.EFFICIENCY'] = 75.0
            self.tags[f'{self.plant_id}.BOILER.O2.CONTENT'] = 6.5
        elif fault_type == 'high_emissions':
            self.tags[f'{self.plant_id}.CEMS.NOX.CONCENTRATION'] = 180.0
            self.tags[f'{self.plant_id}.CEMS.SO2.CONCENTRATION'] = 120.0
        elif fault_type == 'pressure_drop':
            self.tags[f'{self.plant_id}.BOILER.STEAM.PRESSURE'] = 85.0
            self.tags[f'{self.plant_id}.STEAM.HEADER.PRESSURE'] = 82.0
        elif fault_type == 'communication_error':
            # Simulate data quality degradation
            self.running = False


class MockModbusServer:
    """Mock Modbus TCP server for fuel management and emissions monitoring."""

    def __init__(self, plant_id: str = "PLANT-001", host: str = "localhost", port: int = 502):
        self.plant_id = plant_id
        self.host = host
        self.port = port
        self.registers = {}
        self.coils = {}
        self.running = False
        self._initialize_registers()

    def _initialize_registers(self):
        """Initialize Modbus register map."""
        # Holding registers (addresses 0-999)
        self.registers = {
            # Fuel flow meters (0-99)
            0: 1500.0,   # Natural gas flow m³/hr
            1: 0.0,      # Fuel oil flow kg/hr
            2: 0.0,      # Biomass flow kg/hr
            3: 5.0,      # Gas pressure bar
            4: 25.0,     # Gas temperature °C

            # Fuel tank levels (100-199)
            100: 75000.0,   # Gas storage m³
            101: 85000.0,   # Oil tank liters
            102: 350000.0,  # Biomass silo kg

            # Emissions CEMS (200-299)
            200: 125,    # CO2 % * 10
            201: 95,     # NOx mg/Nm³
            202: 45,     # SO2 mg/Nm³
            203: 42,     # O2 % * 10
            204: 85,     # PM mg/Nm³ * 10
            205: 15,     # CO ppm
            206: 600,    # Stack flow (Nm³/hr / 100)
            207: 140,    # Stack temp °C

            # Heat meters (300-399)
            300: 750,    # Heat output MW * 10
            301: 895,    # Heat input MW * 10
            302: 842,    # Thermal efficiency % * 10

            # Fuel quality (400-499)
            400: 485,    # HHV natural gas MJ/m³ * 10
            401: 520,    # Wobbe index MJ/m³ * 10
            402: 920,    # Methane content % * 10
        }

        # Coils (digital I/O, addresses 0-999)
        self.coils = {
            0: True,     # System running
            1: False,    # Alarm active
            2: True,     # Auto mode
            3: False,    # Manual mode
        }

    async def read_holding_register(self, address: int) -> float:
        """Read single holding register."""
        value = self.registers.get(address, 0.0)
        # Add small variation
        variation = value * random.uniform(-0.01, 0.01)
        return value + variation

    async def read_holding_registers(self, start_address: int, count: int) -> List[float]:
        """Read multiple holding registers."""
        values = []
        for i in range(count):
            address = start_address + i
            values.append(await self.read_holding_register(address))
        return values

    async def write_holding_register(self, address: int, value: float) -> bool:
        """Write single holding register."""
        self.registers[address] = value
        return True

    async def read_coil(self, address: int) -> bool:
        """Read coil (digital input)."""
        return self.coils.get(address, False)

    async def write_coil(self, address: int, value: bool) -> bool:
        """Write coil (digital output)."""
        self.coils[address] = value
        return True

    async def start(self):
        """Start mock Modbus server."""
        self.running = True
        print(f"Mock Modbus Server started for {self.plant_id} on {self.host}:{self.port}")

    async def stop(self):
        """Stop mock server."""
        self.running = False


# ==============================================================================
# ERP MOCK SERVERS
# ==============================================================================

class MockSAPServer:
    """Mock SAP RFC server for ERP integration."""

    def __init__(self, host: str = "localhost", port: int = 3300):
        self.host = host
        self.port = port
        self.app = None
        self.runner = None
        self.connection_count = 0
        self.rfc_calls = []

    async def handle_rfc(self, request):
        """Handle RFC function call."""
        try:
            data = await request.json()
            function_name = data.get('function')
            parameters = data.get('parameters', {})

            self.rfc_calls.append({
                'function': function_name,
                'parameters': parameters,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })

            # Mock various SAP RFC functions
            if function_name == "Z_GET_MATERIAL_DATA":
                response = self._handle_get_material_data(parameters)
            elif function_name == "Z_POST_PRODUCTION_DATA":
                response = self._handle_post_production_data(parameters)
            elif function_name == "Z_GET_FUEL_PRICES":
                response = self._handle_get_fuel_prices(parameters)
            elif function_name == "Z_GET_PRODUCTION_SCHEDULE":
                response = self._handle_get_production_schedule(parameters)
            elif function_name == "Z_POST_EMISSIONS_DATA":
                response = self._handle_post_emissions_data(parameters)
            elif function_name == "Z_GET_BUDGET_ALLOCATION":
                response = self._handle_get_budget_allocation(parameters)
            else:
                response = {
                    'SUCCESS': '',
                    'MESSAGE': f'Unknown function: {function_name}'
                }

            return web.json_response(response)

        except Exception as e:
            return web.json_response({
                'SUCCESS': '',
                'MESSAGE': f'Error: {str(e)}'
            }, status=500)

    def _handle_get_material_data(self, params):
        """Mock Z_GET_MATERIAL_DATA."""
        material_number = params.get('MATERIAL_NUMBER', '')
        return {
            'SUCCESS': 'X',
            'MATERIAL_NUMBER': material_number,
            'DESCRIPTION': 'Natural Gas - Pipeline Quality',
            'UNIT': 'M3',
            'PRICE': '0.35',
            'CURRENCY': 'USD',
            'HEATING_VALUE': '48.5',
            'SUPPLIER': 'National Gas Corp',
            'CONTRACT': 'CONT-2025-NG-001'
        }

    def _handle_post_production_data(self, params):
        """Mock Z_POST_PRODUCTION_DATA."""
        return {
            'SUCCESS': 'X',
            'MESSAGE': 'Production data posted successfully',
            'DOCUMENT_NUMBER': f"PROD{deterministic_random().randint(100000, 999999)}",
            'FISCAL_YEAR': '2025',
            'POSTING_DATE': datetime.now(timezone.utc).strftime('%Y%m%d')
        }

    def _handle_get_fuel_prices(self, params):
        """Mock Z_GET_FUEL_PRICES."""
        return {
            'SUCCESS': 'X',
            'PRICES': [
                {
                    'MATERIAL': 'NATURAL_GAS',
                    'PRICE': '0.35',
                    'UNIT': 'USD/M3',
                    'VALID_FROM': '20250101',
                    'VALID_TO': '20251231'
                },
                {
                    'MATERIAL': 'FUEL_OIL',
                    'PRICE': '0.75',
                    'UNIT': 'USD/KG',
                    'VALID_FROM': '20250101',
                    'VALID_TO': '20251231'
                }
            ]
        }

    def _handle_get_production_schedule(self, params):
        """Mock Z_GET_PRODUCTION_SCHEDULE."""
        plant_id = params.get('PLANT_ID', 'PLANT-001')
        return {
            'SUCCESS': 'X',
            'PLANT_ID': plant_id,
            'SCHEDULE': [
                {
                    'DATE': (datetime.now(timezone.utc)).strftime('%Y%m%d'),
                    'PLANNED_OUTPUT_MW': '95.0',
                    'DURATION_HOURS': '24',
                    'PRIORITY': 'HIGH'
                }
            ]
        }

    def _handle_post_emissions_data(self, params):
        """Mock Z_POST_EMISSIONS_DATA."""
        return {
            'SUCCESS': 'X',
            'MESSAGE': 'Emissions data posted successfully',
            'DOCUMENT_NUMBER': f"EMIS{deterministic_random().randint(100000, 999999)}"
        }

    def _handle_get_budget_allocation(self, params):
        """Mock Z_GET_BUDGET_ALLOCATION."""
        plant_id = params.get('PLANT_ID', 'PLANT-001')
        return {
            'SUCCESS': 'X',
            'PLANT_ID': plant_id,
            'FUEL_BUDGET': '50000.00',
            'MAINTENANCE_BUDGET': '10000.00',
            'EMISSIONS_CREDITS': '5000.00',
            'CURRENCY': 'USD',
            'FISCAL_YEAR': '2025'
        }

    async def start(self):
        """Start mock SAP server."""
        self.app = web.Application()
        self.app.router.add_post('/rfc', self.handle_rfc)
        self.app.router.add_post('/sap/rfc', self.handle_rfc)

        self.runner = web.AppRunner(self.app)
        await self.runner.setup()

        site = web.TCPSite(self.runner, self.host, self.port)
        await site.start()

        print(f"Mock SAP Server started on {self.host}:{self.port}")

    async def stop(self):
        """Stop mock server."""
        if self.runner:
            await self.runner.cleanup()


class MockOracleAPIServer:
    """Mock Oracle REST API server for ERP integration."""

    def __init__(self, host: str = "localhost", port: int = 8080):
        self.host = host
        self.port = port
        self.app = None
        self.runner = None
        self.api_calls = []

    async def handle_materials(self, request):
        """Handle /api/materials endpoint."""
        self.api_calls.append({
            'endpoint': '/api/materials',
            'method': request.method,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })

        materials = {
            'items': [
                {
                    'id': 'MAT-NG-001',
                    'name': 'Natural Gas - Pipeline',
                    'category': 'FUEL',
                    'unit': 'm3',
                    'price': 0.35,
                    'currency': 'USD',
                    'heating_value_mj_m3': 48.5
                },
                {
                    'id': 'MAT-FO-001',
                    'name': 'Fuel Oil #2',
                    'category': 'FUEL',
                    'unit': 'kg',
                    'price': 0.75,
                    'currency': 'USD',
                    'heating_value_mj_kg': 42.0
                },
                {
                    'id': 'MAT-BM-001',
                    'name': 'Wood Pellets',
                    'category': 'FUEL',
                    'unit': 'kg',
                    'price': 0.15,
                    'currency': 'USD',
                    'heating_value_mj_kg': 18.5
                }
            ],
            'total': 3
        }
        return web.json_response(materials)

    async def handle_production(self, request):
        """Handle /api/production endpoint."""
        if request.method == 'POST':
            data = await request.json()
            self.api_calls.append({
                'endpoint': '/api/production',
                'method': 'POST',
                'data': data,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })

            response = {
                'production_id': f"PROD-{deterministic_random().randint(100000, 999999)}",
                'status': 'confirmed',
                'plant_id': data.get('plant_id'),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        else:
            # GET request
            plant_id = request.query.get('plant_id', 'PLANT-001')
            response = {
                'productions': [
                    {
                        'id': f"PROD-{i}",
                        'plant_id': plant_id,
                        'output_mw': random.uniform(80, 100),
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }
                    for i in range(5)
                ]
            }

        return web.json_response(response)

    async def handle_schedule(self, request):
        """Handle /api/schedule endpoint."""
        plant_id = request.query.get('plant_id', 'PLANT-001')

        schedule = {
            'plant_id': plant_id,
            'schedules': [
                {
                    'date': (datetime.now(timezone.utc)).strftime('%Y-%m-%d'),
                    'planned_output_mw': 95.0,
                    'duration_hours': 24,
                    'priority': 'HIGH',
                    'fuel_type': 'natural_gas'
                }
            ]
        }
        return web.json_response(schedule)

    async def handle_budget(self, request):
        """Handle /api/budget endpoint."""
        plant_id = request.query.get('plant_id', 'PLANT-001')

        budget = {
            'plant_id': plant_id,
            'fiscal_year': 2025,
            'allocations': {
                'fuel': 50000.0,
                'maintenance': 10000.0,
                'emissions_credits': 5000.0,
                'operations': 15000.0
            },
            'currency': 'USD'
        }
        return web.json_response(budget)

    async def start(self):
        """Start mock Oracle API server."""
        self.app = web.Application()
        self.app.router.add_get('/api/materials', self.handle_materials)
        self.app.router.add_get('/api/production', self.handle_production)
        self.app.router.add_post('/api/production', self.handle_production)
        self.app.router.add_get('/api/schedule', self.handle_schedule)
        self.app.router.add_get('/api/budget', self.handle_budget)

        self.runner = web.AppRunner(self.app)
        await self.runner.setup()

        site = web.TCPSite(self.runner, self.host, self.port)
        await site.start()

        print(f"Mock Oracle API Server started on {self.host}:{self.port}")

    async def stop(self):
        """Stop mock server."""
        if self.runner:
            await self.runner.cleanup()


# ==============================================================================
# MQTT BROKER MOCK
# ==============================================================================

class MockMQTTBroker:
    """Mock MQTT broker for agent messaging."""

    def __init__(self, host: str = "localhost", port: int = 1883):
        self.host = host
        self.port = port
        self.running = False
        self.topics = {}
        self.subscribers = {}

    async def publish(self, topic: str, payload: str):
        """Publish message to topic."""
        if topic not in self.topics:
            self.topics[topic] = []

        message = {
            'topic': topic,
            'payload': payload,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        self.topics[topic].append(message)

        # Notify subscribers
        if topic in self.subscribers:
            for callback in self.subscribers[topic]:
                await callback(message)

    async def subscribe(self, topic: str, callback):
        """Subscribe to topic."""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)

    async def start(self):
        """Start mock broker."""
        self.running = True
        print(f"Mock MQTT Broker started on {self.host}:{self.port}")

    async def stop(self):
        """Stop mock broker."""
        self.running = False


# ==============================================================================
# SUB-AGENT MOCKS
# ==============================================================================

class MockSubAgent:
    """Mock sub-agent for coordination testing."""

    def __init__(self, agent_id: str, agent_type: str, port: int):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.port = port
        self.app = None
        self.runner = None
        self.task_queue = []
        self.results = []

    async def handle_execute(self, request):
        """Handle agent execution request."""
        try:
            data = await request.json()
            task_id = f"TASK-{len(self.task_queue):04d}"

            task = {
                'task_id': task_id,
                'agent_id': self.agent_id,
                'input_data': data,
                'received': datetime.now(timezone.utc).isoformat()
            }
            self.task_queue.append(task)

            # Simulate processing
            await asyncio.sleep(random.uniform(0.1, 0.5))

            # Generate mock result based on agent type
            result = self._generate_result(data)

            self.results.append({
                'task_id': task_id,
                'result': result,
                'completed': datetime.now(timezone.utc).isoformat()
            })

            return web.json_response({
                'task_id': task_id,
                'status': 'SUCCESS',
                'result': result
            })

        except Exception as e:
            return web.json_response({
                'status': 'ERROR',
                'error': str(e)
            }, status=500)

    def _generate_result(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock result based on agent type."""
        if self.agent_type == 'boiler_efficiency':
            return {
                'thermal_efficiency': random.uniform(88, 92),
                'combustion_efficiency': random.uniform(96, 98),
                'heat_rate_mj_kwh': random.uniform(10, 12),
                'optimization_recommendations': [
                    'Adjust O2 setpoint to 3.2%',
                    'Increase feedwater temperature'
                ]
            }
        elif self.agent_type == 'steam_distribution':
            return {
                'distribution_efficiency': random.uniform(95, 98),
                'pressure_drops': {
                    'header1': 0.5,
                    'header2': 0.3
                },
                'flow_balance': 'OPTIMAL'
            }
        elif self.agent_type == 'heat_recovery':
            return {
                'recovery_efficiency': random.uniform(70, 85),
                'recovered_heat_mw': random.uniform(8, 12),
                'economizer_effectiveness': random.uniform(80, 90)
            }
        elif self.agent_type == 'emissions_monitoring':
            return {
                'nox_mg_nm3': random.uniform(80, 120),
                'so2_mg_nm3': random.uniform(40, 60),
                'pm_mg_nm3': random.uniform(5, 15),
                'compliance_status': 'PASS'
            }
        else:
            return {'status': 'completed'}

    async def handle_status(self, request):
        """Handle status request."""
        return web.json_response({
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'status': 'RUNNING',
            'tasks_processed': len(self.results),
            'tasks_queued': len(self.task_queue)
        })

    async def start(self):
        """Start mock sub-agent server."""
        self.app = web.Application()
        self.app.router.add_post('/execute', self.handle_execute)
        self.app.router.add_get('/status', self.handle_status)

        self.runner = web.AppRunner(self.app)
        await self.runner.setup()

        site = web.TCPSite(self.runner, 'localhost', self.port)
        await site.start()

        print(f"Mock {self.agent_id} ({self.agent_type}) started on port {self.port}")

    async def stop(self):
        """Stop mock sub-agent."""
        if self.runner:
            await self.runner.cleanup()


# ==============================================================================
# MULTI-PLANT COORDINATOR MOCK
# ==============================================================================

class MockMultiPlantCoordinator:
    """Mock multi-plant coordinator for testing cross-plant orchestration."""

    def __init__(self, plant_count: int = 3):
        self.plant_count = plant_count
        self.plants = {}
        self.opcua_servers = []
        self.modbus_servers = []
        self._initialize_plants()

    def _initialize_plants(self):
        """Initialize mock plants."""
        for i in range(self.plant_count):
            plant_id = f"PLANT-{i+1:03d}"
            self.plants[plant_id] = {
                'id': plant_id,
                'name': f"Industrial Plant {i+1}",
                'capacity_mw': random.uniform(50, 500),
                'current_load_mw': random.uniform(40, 450),
                'efficiency': random.uniform(85, 92),
                'status': 'OPERATIONAL'
            }

    async def start(self):
        """Start all plant mock servers."""
        base_opcua_port = 4840
        base_modbus_port = 502

        for i, plant_id in enumerate(self.plants.keys()):
            # Start OPC UA server for plant
            opcua = MockOPCUAServer(
                plant_id=plant_id,
                host='localhost',
                port=base_opcua_port + i
            )
            await opcua.start()
            self.opcua_servers.append(opcua)

            # Start Modbus server for plant
            modbus = MockModbusServer(
                plant_id=plant_id,
                host='localhost',
                port=base_modbus_port + i
            )
            await modbus.start()
            self.modbus_servers.append(modbus)

        print(f"Mock Multi-Plant Coordinator started with {self.plant_count} plants")

    async def stop(self):
        """Stop all plant mock servers."""
        for server in self.opcua_servers:
            await server.stop()
        for server in self.modbus_servers:
            await server.stop()

    def get_plant_status(self, plant_id: str) -> Dict[str, Any]:
        """Get status of specific plant."""
        return self.plants.get(plant_id, {})

    def get_all_plants_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all plants."""
        return self.plants.copy()


# ==============================================================================
# SERVER LIFECYCLE MANAGEMENT
# ==============================================================================

async def start_all_mock_servers():
    """Start all mock servers for integration testing."""
    servers = {
        'opcua': MockOPCUAServer(),
        'modbus': MockModbusServer(),
        'sap': MockSAPServer(),
        'oracle': MockOracleAPIServer(),
        'mqtt': MockMQTTBroker(),
        'gl002': MockSubAgent('GL-002', 'boiler_efficiency', 5002),
        'gl003': MockSubAgent('GL-003', 'steam_distribution', 5003),
        'gl004': MockSubAgent('GL-004', 'heat_recovery', 5004),
        'gl005': MockSubAgent('GL-005', 'emissions_monitoring', 5005),
    }

    # Start all servers
    for name, server in servers.items():
        await server.start()

    print("All mock servers started successfully")
    return servers


async def stop_all_mock_servers(servers: Dict[str, Any]):
    """Stop all mock servers."""
    for name, server in servers.items():
        await server.stop()

    print("All mock servers stopped")
