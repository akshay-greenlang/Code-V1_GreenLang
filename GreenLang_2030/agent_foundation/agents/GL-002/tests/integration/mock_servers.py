"""
Mock Servers for Integration Testing

Provides mock SCADA, ERP, Fuel Management, and CEMS servers for integration testing.
These servers simulate real external systems without requiring actual hardware/software.
"""

import asyncio
import random
from datetime import datetime
from typing import Dict, Any, List
from aiohttp import web
import json


class MockOPCUAServer:
    """Mock OPC UA server for SCADA simulation."""

    def __init__(self, host="localhost", port=4840):
        self.host = host
        self.port = port
        self.tags = {}
        self.running = False
        self._initialize_tags()

    def _initialize_tags(self):
        """Initialize mock tag values."""
        self.tags = {
            'BOILER.STEAM.PRESSURE': 100.0,
            'BOILER.STEAM.TEMPERATURE': 490.0,
            'BOILER.STEAM.FLOW': 200.0,
            'BOILER.EFFICIENCY': 89.5,
            'BOILER.O2.CONTENT': 3.5,
            'BOILER.DRUM.LEVEL': 0.0,
            'BOILER.FUEL.VALVE.POSITION': 50.0,
            'BOILER.STATUS': 1
        }

    async def read_tag(self, tag_name: str) -> float:
        """Read tag value with simulated variation."""
        base_value = self.tags.get(tag_name, 0.0)
        # Add small random variation
        variation = base_value * random.uniform(-0.02, 0.02)
        return base_value + variation

    async def write_tag(self, tag_name: str, value: float) -> bool:
        """Write tag value."""
        if tag_name in self.tags:
            self.tags[tag_name] = value
            return True
        return False

    async def start(self):
        """Start mock server."""
        self.running = True
        print(f"Mock OPC UA Server started on {self.host}:{self.port}")

    async def stop(self):
        """Stop mock server."""
        self.running = False


class MockModbusServer:
    """Mock Modbus server for fuel/emissions systems."""

    def __init__(self, host="localhost", port=502):
        self.host = host
        self.port = port
        self.registers = {}
        self.running = False
        self._initialize_registers()

    def _initialize_registers(self):
        """Initialize mock register values."""
        self.registers = {
            # Fuel flow meters
            100: 1500.0,  # Gas flow m3/hr
            101: 800.0,   # Oil flow kg/hr
            102: 5000.0,  # Biomass flow kg/hr

            # Emissions
            200: 12.5,    # CO2 %
            201: 95.0,    # NOx ppm
            202: 45.0,    # SO2 ppm
            203: 4.2,     # O2 %
            204: 8.5,     # PM mg/m3

            # Tank levels
            300: 7500.0,  # Gas tank m3
            301: 65000.0, # Oil tank liters
            302: 250000.0 # Biomass silo kg
        }

    async def read_register(self, address: int) -> float:
        """Read register value."""
        return self.registers.get(address, 0.0)

    async def write_register(self, address: int, value: float) -> bool:
        """Write register value."""
        self.registers[address] = value
        return True

    async def start(self):
        """Start mock server."""
        self.running = True
        print(f"Mock Modbus Server started on {self.host}:{self.port}")

    async def stop(self):
        """Stop mock server."""
        self.running = False


class MockSAPServer:
    """Mock SAP RFC server."""

    def __init__(self, host="localhost", port=3300):
        self.host = host
        self.port = port
        self.app = None
        self.runner = None

    async def handle_rfc(self, request):
        """Handle RFC function call."""
        data = await request.json()
        function_name = data.get('function')
        parameters = data.get('parameters', {})

        # Mock responses
        if function_name == "Z_GET_MATERIAL_DATA":
            response = {
                'MATERIAL_NUMBER': parameters.get('MATERIAL_NUMBER'),
                'DESCRIPTION': 'Natural Gas',
                'UNIT': 'M3',
                'PRICE': '0.35'
            }
        elif function_name == "Z_POST_PRODUCTION_DATA":
            response = {
                'SUCCESS': 'X',
                'MESSAGE': 'Posted successfully',
                'DOCUMENT_NUMBER': f"DOC{random.randint(10000, 99999)}"
            }
        else:
            response = {'SUCCESS': 'X'}

        return web.json_response(response)

    async def start(self):
        """Start mock SAP server."""
        self.app = web.Application()
        self.app.router.add_post('/rfc', self.handle_rfc)

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
    """Mock Oracle REST API server."""

    def __init__(self, host="localhost", port=8080):
        self.host = host
        self.port = port
        self.app = None
        self.runner = None

    async def handle_materials(self, request):
        """Handle materials endpoint."""
        materials = {
            'items': [
                {'id': 'MAT001', 'name': 'Natural Gas', 'unit': 'm3', 'price': 0.35},
                {'id': 'MAT002', 'name': 'Fuel Oil #2', 'unit': 'kg', 'price': 0.75}
            ]
        }
        return web.json_response(materials)

    async def handle_orders(self, request):
        """Handle orders endpoint."""
        if request.method == 'POST':
            order = {
                'order_id': f"ORD{random.randint(10000, 99999)}",
                'status': 'confirmed',
                'timestamp': datetime.utcnow().isoformat()
            }
        else:
            order = {'orders': []}

        return web.json_response(order)

    async def start(self):
        """Start mock Oracle API server."""
        self.app = web.Application()
        self.app.router.add_get('/api/v1/materials', self.handle_materials)
        self.app.router.add_post('/api/v1/orders', self.handle_orders)
        self.app.router.add_get('/api/v1/orders', self.handle_orders)

        self.runner = web.AppRunner(self.app)
        await self.runner.setup()

        site = web.TCPSite(self.runner, self.host, self.port)
        await site.start()

        print(f"Mock Oracle API Server started on {self.host}:{self.port}")

    async def stop(self):
        """Stop mock server."""
        if self.runner:
            await self.runner.cleanup()


class MockMQTTBroker:
    """Mock MQTT broker for emissions data."""

    def __init__(self, host="localhost", port=1883):
        self.host = host
        self.port = port
        self.topics = {}
        self.subscribers = {}
        self.running = False

    async def publish(self, topic: str, message: str):
        """Publish message to topic."""
        if topic not in self.topics:
            self.topics[topic] = []

        self.topics[topic].append({
            'timestamp': datetime.utcnow(),
            'message': message
        })

        # Notify subscribers
        for callback in self.subscribers.get(topic, []):
            await callback(topic, message)

    async def subscribe(self, topic: str, callback):
        """Subscribe to topic."""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)

    async def start(self):
        """Start mock broker."""
        self.running = True
        print(f"Mock MQTT Broker started on {self.host}:{self.port}")

        # Start publishing simulated data
        asyncio.create_task(self._publish_loop())

    async def _publish_loop(self):
        """Publish simulated emissions data."""
        while self.running:
            emissions_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'co2': random.uniform(11, 14),
                'nox': random.uniform(80, 120),
                'so2': random.uniform(40, 60),
                'o2': random.uniform(3, 5)
            }

            await self.publish('boiler/emissions/data', json.dumps(emissions_data))
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
        'sap': MockSAPServer(),
        'oracle': MockOracleAPIServer(),
        'mqtt': MockMQTTBroker()
    }

    for name, server in servers.items():
        await server.start()

    return servers


async def stop_all_mock_servers(servers):
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
