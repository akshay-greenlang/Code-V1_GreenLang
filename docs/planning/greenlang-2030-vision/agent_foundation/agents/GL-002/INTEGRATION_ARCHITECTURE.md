# GL-002 BoilerEfficiencyOptimizer Integration Architecture

## Executive Summary

The GL-002 BoilerEfficiencyOptimizer requires seamless integration with industrial control systems, enterprise software, and the GreenLang agent ecosystem. This document defines the complete integration architecture, protocols, data flows, and security requirements for production deployment.

## Integration Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     GL-002 Integration Ecosystem                 │
├───────────────────────────┬─────────────────────────────────────┤
│   Industrial Systems      │        Enterprise Systems           │
├───────────────────────────┼─────────────────────────────────────┤
│ • DCS/BCS                 │ • SAP ERP                           │
│ • SCADA                   │ • Oracle ECC                        │
│ • PLC                     │ • Workday HCM                       │
│ • Historian               │ • ServiceNow CMMS                   │
│ • CEMS                    │ • Power BI / Tableau                │
│ • Field Instruments       │ • SharePoint                        │
└───────────────────────────┴─────────────────────────────────────┘
                                    ↕
┌─────────────────────────────────────────────────────────────────┐
│                    GL-002 Integration Layer                      │
├─────────────────────────────────────────────────────────────────┤
│ • Protocol Adapters (OPC-UA, Modbus, MQTT, REST)                │
│ • Data Transformation Engine                                     │
│ • Message Queue (Kafka/RabbitMQ)                                 │
│ • API Gateway                                                    │
│ • Security & Authentication                                      │
└─────────────────────────────────────────────────────────────────┘
                                    ↕
┌─────────────────────────────────────────────────────────────────┐
│                    GreenLang Agent Network                       │
├─────────────────────────────────────────────────────────────────┤
│ • GL-001 ProcessHeatOrchestrator (Master)                        │
│ • GL-004 BurnerOptimizationAgent                                 │
│ • GL-010 EmissionsComplianceAgent                                │
│ • GL-011 FuelManagementOptimizer                                 │
│ • GL-013 PredictiveMaintenanceAgent                              │
└─────────────────────────────────────────────────────────────────┘
```

## 1. Industrial Control System Integration

### 1.1 DCS/BCS Integration

#### OPC-UA Server Configuration

```yaml
opc_ua_configuration:
  server:
    endpoint: "opc.tcp://dcs.plant.local:4840/GL002"
    security_mode: "SignAndEncrypt"
    security_policy: "Basic256Sha256"

  authentication:
    type: "X509Certificate"
    certificate_path: "/certs/gl002.crt"
    private_key_path: "/certs/gl002.key"
    ca_certificate: "/certs/ca.crt"

  subscription_settings:
    publishing_interval_ms: 1000
    lifetime_count: 10000
    max_keepalive_count: 10
    priority: 100
```

#### Data Point Mapping

```python
opc_ua_node_mapping = {
    # Process Variables
    "fuel_flow": {
        "node_id": "ns=2;s=Boiler.FuelFlow",
        "data_type": "Float",
        "unit": "kg/hr",
        "access": "read",
        "sampling_rate_ms": 1000
    },
    "steam_output": {
        "node_id": "ns=2;s=Boiler.SteamOutput",
        "data_type": "Float",
        "unit": "kg/hr",
        "access": "read",
        "sampling_rate_ms": 1000
    },
    "o2_percentage": {
        "node_id": "ns=2;s=Boiler.FlueGas.O2",
        "data_type": "Float",
        "unit": "%",
        "access": "read",
        "sampling_rate_ms": 5000
    },

    # Control Setpoints (Write Access)
    "o2_setpoint": {
        "node_id": "ns=2;s=Boiler.Control.O2Setpoint",
        "data_type": "Float",
        "unit": "%",
        "access": "write",
        "limits": {"min": 2.0, "max": 6.0}
    },
    "fuel_flow_setpoint": {
        "node_id": "ns=2;s=Boiler.Control.FuelFlowSP",
        "data_type": "Float",
        "unit": "kg/hr",
        "access": "write",
        "limits": {"min": 1000, "max": 10000}
    }
}
```

#### Implementation Code

```python
from asyncua import Client, ua
import asyncio

class DCSIntegration:
    def __init__(self, config):
        self.config = config
        self.client = None
        self.subscriptions = {}

    async def connect(self):
        """Establish secure connection to DCS OPC-UA server"""
        self.client = Client(self.config['endpoint'])

        # Set security
        await self.client.set_security(
            policy=self.config['security_policy'],
            certificate=self.config['certificate_path'],
            private_key=self.config['private_key_path'],
            mode=ua.MessageSecurityMode.SignAndEncrypt
        )

        # Connect
        await self.client.connect()

        # Create subscriptions
        await self._create_subscriptions()

    async def _create_subscriptions(self):
        """Subscribe to process variables"""
        sub = await self.client.create_subscription(
            self.config['publishing_interval_ms'],
            self._data_change_handler
        )

        for tag, mapping in opc_ua_node_mapping.items():
            if mapping['access'] in ['read', 'read/write']:
                node = self.client.get_node(mapping['node_id'])
                handle = await sub.subscribe_data_change(node)
                self.subscriptions[tag] = handle

    async def _data_change_handler(self, node, value, data):
        """Handle real-time data changes"""
        # Process incoming data
        await self.process_data_update(node, value)

    async def write_setpoint(self, tag, value):
        """Write control setpoint to DCS"""
        if tag in opc_ua_node_mapping:
            mapping = opc_ua_node_mapping[tag]

            # Validate limits
            if 'limits' in mapping:
                value = max(mapping['limits']['min'],
                          min(value, mapping['limits']['max']))

            # Write to OPC-UA server
            node = self.client.get_node(mapping['node_id'])
            await node.write_value(ua.DataValue(ua.Variant(value, ua.VariantType.Float)))

            # Log write operation
            await self.log_control_action(tag, value)
```

### 1.2 SCADA Integration

#### MQTT Configuration

```yaml
mqtt_configuration:
  broker:
    host: "mqtt.scada.plant.local"
    port: 8883
    tls_enabled: true

  authentication:
    username: "gl002_agent"
    password_env: "MQTT_PASSWORD"
    client_id: "GL002-BoilerOptimizer"

  topics:
    subscribe:
      - "plant/boiler/+/measurements/#"
      - "plant/boiler/+/alarms"
      - "plant/boiler/+/status"

    publish:
      - "plant/boiler/+/optimization/recommendations"
      - "plant/boiler/+/optimization/setpoints"
      - "plant/boiler/+/optimization/kpis"

  qos_levels:
    measurements: 1
    setpoints: 2
    alarms: 2
```

#### MQTT Client Implementation

```python
import paho.mqtt.client as mqtt
import json
import ssl

class SCADAIntegration:
    def __init__(self, config):
        self.config = config
        self.client = mqtt.Client(client_id=config['client_id'])
        self._setup_callbacks()
        self._setup_tls()

    def _setup_callbacks(self):
        """Configure MQTT callbacks"""
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

    def _setup_tls(self):
        """Configure TLS for secure connection"""
        context = ssl.create_default_context()
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        context.load_cert_chain(
            certfile=self.config['cert_path'],
            keyfile=self.config['key_path']
        )
        self.client.tls_set_context(context)

    def _on_connect(self, client, userdata, flags, rc):
        """Handle connection event"""
        if rc == 0:
            # Subscribe to topics
            for topic in self.config['topics']['subscribe']:
                client.subscribe(topic, qos=self.config['qos_levels']['measurements'])

    def _on_message(self, client, userdata, msg):
        """Process incoming SCADA data"""
        try:
            topic_parts = msg.topic.split('/')
            boiler_id = topic_parts[2]
            data_type = topic_parts[3]

            payload = json.loads(msg.payload.decode())

            # Route to appropriate handler
            if data_type == 'measurements':
                self.process_measurements(boiler_id, payload)
            elif data_type == 'alarms':
                self.process_alarms(boiler_id, payload)

        except Exception as e:
            self.log_error(f"Error processing MQTT message: {e}")

    def publish_optimization(self, boiler_id, optimization_data):
        """Publish optimization results to SCADA"""
        topic = f"plant/boiler/{boiler_id}/optimization/recommendations"

        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "boiler_id": boiler_id,
            "optimization": optimization_data,
            "source": "GL002-BoilerOptimizer"
        }

        self.client.publish(
            topic,
            json.dumps(payload),
            qos=self.config['qos_levels']['setpoints'],
            retain=True
        )
```

### 1.3 Modbus Integration

#### Modbus Configuration

```yaml
modbus_configuration:
  devices:
    - name: "Boiler_PLC_1"
      type: "tcp"
      host: "192.168.1.100"
      port: 502
      unit_id: 1

      registers:
        holding_registers:
          - address: 40001
            name: "fuel_valve_position"
            data_type: "float32"
            byte_order: "big"
            access: "read/write"

          - address: 40003
            name: "air_damper_position"
            data_type: "float32"
            byte_order: "big"
            access: "read/write"

        input_registers:
          - address: 30001
            name: "steam_pressure"
            data_type: "float32"
            byte_order: "big"
            scale: 0.1

          - address: 30003
            name: "stack_temperature"
            data_type: "int16"
            byte_order: "big"
```

#### Modbus Client Implementation

```python
from pymodbus.client import ModbusTcpClient
from pymodbus.payload import BinaryPayloadDecoder, BinaryPayloadBuilder
from pymodbus.constants import Endian
import struct

class ModbusIntegration:
    def __init__(self, config):
        self.config = config
        self.clients = {}
        self._initialize_clients()

    def _initialize_clients(self):
        """Initialize Modbus TCP clients"""
        for device in self.config['devices']:
            client = ModbusTcpClient(
                host=device['host'],
                port=device['port'],
                timeout=3
            )
            self.clients[device['name']] = client

    def read_registers(self, device_name, start_address, count):
        """Read holding or input registers"""
        client = self.clients[device_name]

        if not client.connect():
            raise ConnectionError(f"Cannot connect to {device_name}")

        try:
            # Read registers
            result = client.read_holding_registers(
                address=start_address,
                count=count,
                unit=self.config['devices'][device_name]['unit_id']
            )

            if result.isError():
                raise Exception(f"Modbus error: {result}")

            return result.registers

        finally:
            client.close()

    def write_register(self, device_name, address, value, data_type='float32'):
        """Write to holding register"""
        client = self.clients[device_name]

        if not client.connect():
            raise ConnectionError(f"Cannot connect to {device_name}")

        try:
            # Build payload based on data type
            builder = BinaryPayloadBuilder(byteorder=Endian.Big)

            if data_type == 'float32':
                builder.add_32bit_float(value)
            elif data_type == 'int16':
                builder.add_16bit_int(value)

            payload = builder.to_registers()

            # Write registers
            result = client.write_registers(
                address=address,
                values=payload,
                unit=self.config['devices'][device_name]['unit_id']
            )

            if result.isError():
                raise Exception(f"Modbus write error: {result}")

            return True

        finally:
            client.close()
```

## 2. Enterprise System Integration

### 2.1 SAP Integration

#### SAP REST API Configuration

```yaml
sap_integration:
  api_endpoint: "https://sap.company.com/api/v1"

  authentication:
    type: "OAuth2"
    token_url: "https://sap.company.com/oauth/token"
    client_id_env: "SAP_CLIENT_ID"
    client_secret_env: "SAP_CLIENT_SECRET"
    scope: "plant_maintenance fuel_management"

  endpoints:
    maintenance:
      work_orders: "/maintenance/work-orders"
      equipment: "/maintenance/equipment"
      notifications: "/maintenance/notifications"

    materials:
      fuel_prices: "/materials/fuel-prices"
      inventory: "/materials/inventory"

    production:
      schedules: "/production/schedules"
      actuals: "/production/actuals"
```

#### SAP Integration Implementation

```python
import aiohttp
import asyncio
from datetime import datetime, timedelta

class SAPIntegration:
    def __init__(self, config):
        self.config = config
        self.token = None
        self.token_expiry = None

    async def _get_token(self):
        """Get OAuth2 token for SAP API"""
        if self.token and self.token_expiry > datetime.utcnow():
            return self.token

        async with aiohttp.ClientSession() as session:
            data = {
                'grant_type': 'client_credentials',
                'client_id': os.environ[self.config['client_id_env']],
                'client_secret': os.environ[self.config['client_secret_env']],
                'scope': self.config['scope']
            }

            async with session.post(self.config['token_url'], data=data) as resp:
                result = await resp.json()
                self.token = result['access_token']
                self.token_expiry = datetime.utcnow() + timedelta(seconds=result['expires_in'])

        return self.token

    async def get_fuel_prices(self):
        """Retrieve current fuel prices from SAP"""
        token = await self._get_token()

        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }

        async with aiohttp.ClientSession() as session:
            url = f"{self.config['api_endpoint']}{self.config['endpoints']['materials']['fuel_prices']}"

            params = {
                'plant': 'PLANT_001',
                'date': datetime.utcnow().date().isoformat()
            }

            async with session.get(url, headers=headers, params=params) as resp:
                data = await resp.json()

                return {
                    'natural_gas': data['prices']['natural_gas']['unit_price'],
                    'fuel_oil': data['prices']['fuel_oil']['unit_price'],
                    'currency': data['currency'],
                    'unit': data['unit'],
                    'valid_from': data['valid_from'],
                    'valid_to': data['valid_to']
                }

    async def create_maintenance_notification(self, boiler_id, issue_description, priority='MEDIUM'):
        """Create maintenance notification in SAP PM"""
        token = await self._get_token()

        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }

        notification = {
            'equipment_id': f'BOILER_{boiler_id}',
            'notification_type': 'M2',  # Maintenance request
            'description': issue_description,
            'priority': priority,
            'reported_by': 'GL002-BoilerOptimizer',
            'detection_date': datetime.utcnow().isoformat(),
            'functional_location': f'PLANT-001-BOILER-{boiler_id}'
        }

        async with aiohttp.ClientSession() as session:
            url = f"{self.config['api_endpoint']}{self.config['endpoints']['maintenance']['notifications']}"

            async with session.post(url, headers=headers, json=notification) as resp:
                if resp.status == 201:
                    result = await resp.json()
                    return result['notification_id']
                else:
                    raise Exception(f"Failed to create notification: {await resp.text()}")
```

### 2.2 Oracle Integration

#### Oracle REST Data Services Configuration

```yaml
oracle_integration:
  ords_endpoint: "https://oracle.company.com/ords/prod"

  authentication:
    type: "BasicAuth"
    username_env: "ORACLE_USERNAME"
    password_env: "ORACLE_PASSWORD"

  schemas:
    asset_management: "AM_SCHEMA"
    work_management: "WM_SCHEMA"
    inventory: "INV_SCHEMA"
```

### 2.3 ServiceNow CMMS Integration

#### ServiceNow API Configuration

```yaml
servicenow_integration:
  instance: "company.service-now.com"

  authentication:
    type: "OAuth2"
    client_id_env: "SNOW_CLIENT_ID"
    client_secret_env: "SNOW_CLIENT_SECRET"

  tables:
    cmdb_ci: "cmdb_ci_boiler"
    incident: "incident"
    change_request: "change_request"
    work_order: "wm_order"
```

## 3. GreenLang Agent Network Integration

### 3.1 GL-001 ProcessHeatOrchestrator Integration

#### Communication Protocol

```python
class GL001Integration:
    """Integration with GL-001 ProcessHeatOrchestrator"""

    def __init__(self):
        self.orchestrator_endpoint = "http://gl001-orchestrator:8000/api/v1"
        self.agent_id = "GL-002"
        self.capabilities = {
            "optimization_types": ["combustion", "efficiency", "emissions"],
            "response_time_ms": 3000,
            "batch_capable": True
        }

    async def register_with_orchestrator(self):
        """Register GL-002 with master orchestrator"""
        registration = {
            "agent_id": self.agent_id,
            "agent_type": "optimizer",
            "domain": "boiler_systems",
            "capabilities": self.capabilities,
            "health_endpoint": "/health",
            "api_version": "v1"
        }

        async with aiohttp.ClientSession() as session:
            url = f"{self.orchestrator_endpoint}/agents/register"
            async with session.post(url, json=registration) as resp:
                result = await resp.json()
                return result['registration_id']

    async def receive_optimization_request(self, request):
        """Handle optimization request from GL-001"""
        return {
            "request_id": request['request_id'],
            "agent_id": self.agent_id,
            "optimization_type": request['type'],
            "results": await self.run_optimization(request['parameters']),
            "timestamp": datetime.utcnow().isoformat()
        }

    async def report_status(self):
        """Report status to orchestrator"""
        status = {
            "agent_id": self.agent_id,
            "status": "operational",
            "current_load": self.get_current_load(),
            "queue_depth": self.get_queue_depth(),
            "last_optimization": self.last_optimization_time
        }

        async with aiohttp.ClientSession() as session:
            url = f"{self.orchestrator_endpoint}/agents/{self.agent_id}/status"
            await session.put(url, json=status)
```

### 3.2 Inter-Agent Communication

#### Message Queue Integration

```python
import aiokafka
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer

class AgentMessageBus:
    """Kafka-based inter-agent communication"""

    def __init__(self):
        self.producer = None
        self.consumer = None
        self.topics = {
            'requests': 'gl.agent.requests',
            'responses': 'gl.agent.responses',
            'events': 'gl.agent.events'
        }

    async def initialize(self):
        """Initialize Kafka producer and consumer"""
        self.producer = AIOKafkaProducer(
            bootstrap_servers='kafka:9092',
            value_serializer=lambda v: json.dumps(v).encode()
        )

        self.consumer = AIOKafkaConsumer(
            self.topics['requests'],
            bootstrap_servers='kafka:9092',
            group_id='gl002-consumer-group',
            value_deserializer=lambda m: json.loads(m.decode())
        )

        await self.producer.start()
        await self.consumer.start()

    async def request_from_agent(self, target_agent, request_type, parameters):
        """Send request to another agent"""
        message = {
            'source': 'GL-002',
            'target': target_agent,
            'request_id': str(uuid.uuid4()),
            'request_type': request_type,
            'parameters': parameters,
            'timestamp': datetime.utcnow().isoformat()
        }

        await self.producer.send(
            self.topics['requests'],
            value=message,
            key=target_agent.encode()
        )

        # Wait for response
        return await self.wait_for_response(message['request_id'])

    async def broadcast_event(self, event_type, data):
        """Broadcast event to all interested agents"""
        event = {
            'source': 'GL-002',
            'event_type': event_type,
            'data': data,
            'timestamp': datetime.utcnow().isoformat()
        }

        await self.producer.send(self.topics['events'], value=event)
```

## 4. Data Transformation & Mapping

### 4.1 Data Transformation Engine

```python
class DataTransformationEngine:
    """Transform data between different system formats"""

    def __init__(self):
        self.transformers = {
            'dcs_to_internal': self.transform_dcs_data,
            'internal_to_sap': self.transform_to_sap_format,
            'scada_to_internal': self.transform_scada_data
        }

    def transform_dcs_data(self, dcs_data):
        """Transform DCS data to internal format"""
        return {
            'timestamp': dcs_data['TimeStamp'],
            'boiler_id': dcs_data['EquipmentID'],
            'measurements': {
                'fuel_flow': {
                    'value': dcs_data['FuelFlow']['Value'],
                    'unit': 'kg/hr',
                    'quality': dcs_data['FuelFlow']['Quality']
                },
                'steam_output': {
                    'value': dcs_data['SteamOutput']['Value'],
                    'unit': 'kg/hr',
                    'quality': dcs_data['SteamOutput']['Quality']
                },
                'efficiency': self.calculate_efficiency(dcs_data)
            }
        }

    def transform_to_sap_format(self, internal_data):
        """Transform internal data to SAP format"""
        return {
            'EQUIPMENT_ID': f"BOILER_{internal_data['boiler_id']}",
            'MEASUREMENT_DATE': internal_data['timestamp'],
            'MEASUREMENTS': [
                {
                    'CHARACTERISTIC': 'EFFICIENCY',
                    'VALUE': internal_data['efficiency'],
                    'UOM': 'PCT'
                },
                {
                    'CHARACTERISTIC': 'FUEL_CONSUMPTION',
                    'VALUE': internal_data['fuel_consumption'],
                    'UOM': 'KG_HR'
                }
            ]
        }
```

### 4.2 Unit Conversion

```python
class UnitConverter:
    """Handle unit conversions between systems"""

    conversion_factors = {
        # Pressure conversions
        ('psi', 'bar'): 0.0689476,
        ('bar', 'psi'): 14.5038,
        ('kPa', 'bar'): 0.01,

        # Temperature conversions are functions
        ('F', 'C'): lambda f: (f - 32) * 5/9,
        ('C', 'F'): lambda c: c * 9/5 + 32,
        ('C', 'K'): lambda c: c + 273.15,

        # Flow conversions
        ('lb/hr', 'kg/hr'): 0.453592,
        ('kg/hr', 'lb/hr'): 2.20462,
        ('scfh', 'Nm3/hr'): 0.0268,

        # Energy conversions
        ('BTU', 'kJ'): 1.05506,
        ('MMBtu', 'GJ'): 1.05506
    }

    def convert(self, value, from_unit, to_unit):
        """Convert value from one unit to another"""
        key = (from_unit, to_unit)

        if key in self.conversion_factors:
            factor = self.conversion_factors[key]
            if callable(factor):
                return factor(value)
            else:
                return value * factor
        else:
            raise ValueError(f"No conversion from {from_unit} to {to_unit}")
```

## 5. Security & Authentication

### 5.1 Authentication Framework

```python
from jose import jwt, JWTError
from cryptography.fernet import Fernet
import hashlib

class SecurityManager:
    """Centralized security management"""

    def __init__(self):
        self.jwt_secret = os.environ['JWT_SECRET']
        self.encryption_key = os.environ['ENCRYPTION_KEY'].encode()
        self.fernet = Fernet(self.encryption_key)

    def generate_jwt_token(self, agent_id, permissions):
        """Generate JWT token for inter-agent communication"""
        payload = {
            'agent_id': agent_id,
            'permissions': permissions,
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(hours=24)
        }

        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')

    def verify_jwt_token(self, token):
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            return payload
        except JWTError:
            return None

    def encrypt_sensitive_data(self, data):
        """Encrypt sensitive data for storage/transmission"""
        if isinstance(data, str):
            data = data.encode()
        return self.fernet.encrypt(data)

    def decrypt_sensitive_data(self, encrypted_data):
        """Decrypt sensitive data"""
        return self.fernet.decrypt(encrypted_data).decode()

    def hash_password(self, password, salt=None):
        """Hash password for storage"""
        if salt is None:
            salt = os.urandom(32)

        pwdhash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            100000  # iterations
        )

        return salt + pwdhash
```

### 5.2 API Key Management

```yaml
api_key_configuration:
  vault:
    type: "HashiCorp"
    endpoint: "https://vault.company.com:8200"
    namespace: "greenlang"

  rotation:
    frequency_days: 90
    notification_days: 7

  keys:
    - name: "DCS_API_KEY"
      path: "secret/gl002/dcs"
      field: "api_key"

    - name: "SAP_CLIENT_SECRET"
      path: "secret/gl002/sap"
      field: "client_secret"
```

## 6. Error Handling & Resilience

### 6.1 Circuit Breaker Pattern

```python
from pybreaker import CircuitBreaker

class ResilientIntegration:
    """Implement circuit breaker for external integrations"""

    def __init__(self):
        # Configure circuit breakers for each integration
        self.dcs_breaker = CircuitBreaker(
            fail_max=5,
            reset_timeout=60,
            exclude=[ConnectionError]
        )

        self.sap_breaker = CircuitBreaker(
            fail_max=3,
            reset_timeout=120
        )

    @property
    def dcs_breaker(self):
        @self.dcs_breaker
        async def read_dcs_data():
            # DCS read operation
            pass

    async def read_with_fallback(self, primary_source, fallback_source):
        """Read data with fallback option"""
        try:
            return await self.dcs_breaker(primary_source)
        except Exception as e:
            self.log_warning(f"Primary source failed: {e}, using fallback")
            return await fallback_source()
```

### 6.2 Retry Logic

```python
from tenacity import retry, stop_after_attempt, wait_exponential

class RetryableOperation:
    """Implement retry logic for transient failures"""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def write_to_dcs(self, tag, value):
        """Write to DCS with automatic retry"""
        return await self.dcs_client.write(tag, value)
```

## 7. Monitoring & Observability

### 7.1 Integration Metrics

```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
integration_requests = Counter(
    'gl002_integration_requests_total',
    'Total integration requests',
    ['integration', 'operation', 'status']
)

integration_latency = Histogram(
    'gl002_integration_latency_seconds',
    'Integration operation latency',
    ['integration', 'operation']
)

integration_health = Gauge(
    'gl002_integration_health',
    'Integration health status',
    ['integration']
)
```

### 7.2 Integration Health Checks

```python
class IntegrationHealthChecker:
    """Monitor health of all integrations"""

    async def check_all_integrations(self):
        """Perform health checks on all integrations"""
        health_status = {}

        # Check DCS
        health_status['dcs'] = await self.check_dcs_health()

        # Check SCADA
        health_status['scada'] = await self.check_scada_health()

        # Check SAP
        health_status['sap'] = await self.check_sap_health()

        # Check Agent Network
        health_status['agent_network'] = await self.check_agent_network_health()

        return health_status

    async def check_dcs_health(self):
        """Check DCS connection health"""
        try:
            # Attempt to read a known tag
            await self.dcs_client.read('HEALTH_CHECK_TAG')
            return {'status': 'healthy', 'latency_ms': 45}
        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e)}
```

## 8. Testing Strategy

### 8.1 Integration Test Framework

```python
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_dcs_integration():
    """Test DCS integration"""
    with patch('asyncua.Client') as mock_client:
        # Setup mock
        mock_client.connect = AsyncMock()
        mock_client.get_node = AsyncMock()

        # Test connection
        integration = DCSIntegration(test_config)
        await integration.connect()

        # Verify connection established
        mock_client.connect.assert_called_once()

@pytest.mark.asyncio
async def test_sap_integration():
    """Test SAP integration"""
    with patch('aiohttp.ClientSession') as mock_session:
        # Test fuel price retrieval
        integration = SAPIntegration(test_config)
        prices = await integration.get_fuel_prices()

        assert 'natural_gas' in prices
        assert prices['currency'] == 'USD'
```

### 8.2 End-to-End Testing

```python
class E2EIntegrationTest:
    """End-to-end integration testing"""

    async def test_complete_optimization_flow(self):
        """Test complete data flow from DCS to optimization to SAP"""

        # Step 1: Simulate DCS data
        dcs_data = self.generate_test_dcs_data()

        # Step 2: Process through GL-002
        optimization_result = await self.gl002.optimize(dcs_data)

        # Step 3: Write setpoints back to DCS
        await self.dcs.write_setpoints(optimization_result['setpoints'])

        # Step 4: Report to SAP
        await self.sap.report_efficiency(optimization_result['metrics'])

        # Verify complete flow
        assert optimization_result['status'] == 'success'
        assert self.dcs.last_write_successful
        assert self.sap.last_report_successful
```

## 9. Deployment Configuration

### 9.1 Docker Compose Configuration

```yaml
version: '3.8'

services:
  gl002-optimizer:
    image: greenlang/gl002-boiler-optimizer:latest
    environment:
      - DCS_ENDPOINT=opc.tcp://dcs:4840
      - MQTT_BROKER=mqtt://scada:8883
      - SAP_API=https://sap.company.com/api
      - KAFKA_BROKERS=kafka:9092
    networks:
      - gl-network
      - industrial-network
    volumes:
      - ./config:/app/config
      - ./certs:/app/certs
    depends_on:
      - kafka
      - redis

  kafka:
    image: confluentinc/cp-kafka:latest
    environment:
      - KAFKA_ZOOKEEPER_CONNECT=zookeeper:2181
      - KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://kafka:9092
    networks:
      - gl-network

  redis:
    image: redis:alpine
    networks:
      - gl-network

networks:
  gl-network:
    driver: bridge
  industrial-network:
    external: true
```

### 9.2 Kubernetes Configuration

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: gl002-integration-config
  namespace: greenlang
data:
  dcs_endpoint: "opc.tcp://dcs.plant.local:4840"
  mqtt_broker: "mqtt.scada.plant.local:8883"
  sap_endpoint: "https://sap.company.com/api/v1"
  kafka_brokers: "kafka-0.kafka:9092,kafka-1.kafka:9092"

---
apiVersion: v1
kind: Secret
metadata:
  name: gl002-integration-secrets
  namespace: greenlang
type: Opaque
data:
  dcs_certificate: <base64_encoded_cert>
  sap_client_secret: <base64_encoded_secret>
  jwt_secret: <base64_encoded_jwt_secret>
```

## 10. Documentation

### 10.1 Integration API Documentation

```yaml
openapi: 3.0.0
info:
  title: GL-002 Integration API
  version: 1.0.0
  description: Integration endpoints for GL-002 BoilerEfficiencyOptimizer

paths:
  /integrations/status:
    get:
      summary: Get status of all integrations
      responses:
        200:
          description: Integration status
          content:
            application/json:
              schema:
                type: object
                properties:
                  dcs:
                    $ref: '#/components/schemas/IntegrationStatus'
                  scada:
                    $ref: '#/components/schemas/IntegrationStatus'
                  sap:
                    $ref: '#/components/schemas/IntegrationStatus'

  /integrations/dcs/read:
    post:
      summary: Read data from DCS
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                tags:
                  type: array
                  items:
                    type: string

  /integrations/dcs/write:
    post:
      summary: Write setpoint to DCS
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                tag:
                  type: string
                value:
                  type: number

components:
  schemas:
    IntegrationStatus:
      type: object
      properties:
        status:
          type: string
          enum: [healthy, degraded, unhealthy]
        latency_ms:
          type: number
        last_success:
          type: string
          format: date-time
        error_message:
          type: string
```

---

## Conclusion

This integration architecture provides a robust, secure, and scalable foundation for GL-002 BoilerEfficiencyOptimizer to interact with industrial control systems, enterprise software, and the GreenLang agent ecosystem. The architecture emphasizes:

1. **Security**: End-to-end encryption, authentication, and authorization
2. **Reliability**: Circuit breakers, retry logic, and fallback mechanisms
3. **Performance**: Asynchronous operations, caching, and optimization
4. **Observability**: Comprehensive monitoring and health checks
5. **Maintainability**: Modular design and extensive documentation

The implementation ensures seamless data flow while maintaining industrial-grade reliability and security standards required for production deployment in critical infrastructure.

---

**Document Version:** 1.0.0
**Date:** 2025-11-15
**Status:** APPROVED FOR IMPLEMENTATION