# GL-002: Boiler Efficiency Optimizer - Integration Guide

## Overview

The GL-002 integration modules provide seamless connectivity with industrial control systems, data historians, cloud services, and enterprise applications. This guide covers setup, configuration, and troubleshooting for all supported integrations.

## Supported Integrations

```
integrations/
├── industrial/
│   ├── opc_ua.py          # OPC UA client
│   ├── modbus.py          # Modbus TCP/RTU
│   ├── mqtt.py            # MQTT broker
│   └── bacnet.py          # BACnet protocol
├── historians/
│   ├── pi_system.py       # OSIsoft PI
│   ├── wonderware.py      # Wonderware Historian
│   └── ignition.py        # Ignition SCADA
├── cloud/
│   ├── aws.py             # AWS IoT Core
│   ├── azure.py           # Azure IoT Hub
│   └── gcp.py             # Google Cloud IoT
├── enterprise/
│   ├── sap.py             # SAP ERP
│   ├── oracle.py          # Oracle EBS
│   └── workday.py         # Workday
└── databases/
    ├── postgresql.py       # PostgreSQL
    ├── timescale.py       # TimescaleDB
    └── influxdb.py        # InfluxDB
```

## Industrial Protocol Integrations

### OPC UA Integration

**Purpose:** Connect to OPC UA servers for real-time data acquisition.

**Setup:**

```python
from gl002.integrations.industrial import OPCUAConnector

# Configure connection
config = {
    'server_url': 'opc.tcp://192.168.1.100:4840',
    'username': 'admin',
    'password': 'secure_password',
    'security_policy': 'Basic256Sha256',
    'security_mode': 'SignAndEncrypt',
    'certificate_path': '/path/to/cert.pem',
    'private_key_path': '/path/to/key.pem'
}

# Initialize connector
opc_connector = OPCUAConnector(config)

# Connect to server
await opc_connector.connect()

# Browse nodes
nodes = await opc_connector.browse_nodes('ns=2;s=Boiler1')

# Subscribe to real-time data
subscription = await opc_connector.subscribe([
    'ns=2;s=Boiler1.SteamFlow',
    'ns=2;s=Boiler1.Pressure',
    'ns=2;s=Boiler1.Temperature'
], callback=process_data)

# Read values
values = await opc_connector.read_values([
    'ns=2;s=Boiler1.Efficiency',
    'ns=2;s=Boiler1.FuelFlow'
])

# Write setpoint
await opc_connector.write_value('ns=2;s=Boiler1.SetPoint', 485.0)
```

**Configuration Options:**

```yaml
opc_ua:
  server:
    url: opc.tcp://localhost:4840
    timeout: 30000  # ms
    retry_count: 3
    retry_delay: 5000  # ms

  security:
    mode: SignAndEncrypt  # None, Sign, SignAndEncrypt
    policy: Basic256Sha256
    certificate: /certs/client.pem
    private_key: /certs/client.key
    trust_store: /certs/trusted/

  subscription:
    publishing_interval: 1000  # ms
    lifetime_count: 10000
    keepalive_count: 10
    max_notifications: 0
    priority: 0

  data_mapping:
    steam_flow:
      node_id: ns=2;s=FT101.PV
      scaling: 1.0
      offset: 0.0
      unit: lb/hr
```

### Modbus Integration

**Purpose:** Connect to Modbus TCP/RTU devices for data collection.

**Setup:**

```python
from gl002.integrations.industrial import ModbusConnector

# TCP Configuration
tcp_config = {
    'type': 'tcp',
    'host': '192.168.1.50',
    'port': 502,
    'slave_id': 1,
    'timeout': 3.0
}

# RTU Configuration
rtu_config = {
    'type': 'rtu',
    'port': '/dev/ttyUSB0',
    'baudrate': 9600,
    'parity': 'N',
    'stopbits': 1,
    'bytesize': 8,
    'slave_id': 1
}

# Initialize connector
modbus = ModbusConnector(tcp_config)

# Connect
modbus.connect()

# Read registers
# Holding registers (function code 03)
values = modbus.read_holding_registers(
    address=1000,
    count=10,
    data_type='float32'
)

# Input registers (function code 04)
inputs = modbus.read_input_registers(
    address=2000,
    count=5,
    data_type='int16'
)

# Write registers
# Single register (function code 06)
modbus.write_register(address=3000, value=1234)

# Multiple registers (function code 16)
modbus.write_registers(address=3001, values=[100, 200, 300])

# Register mapping
register_map = {
    'steam_flow': {'address': 1000, 'type': 'float32', 'scale': 1.0},
    'pressure': {'address': 1002, 'type': 'float32', 'scale': 1.0},
    'temperature': {'address': 1004, 'type': 'float32', 'scale': 0.1}
}

data = modbus.read_mapped_values(register_map)
```

**Modbus Register Configuration:**

```yaml
modbus:
  connection:
    type: tcp
    host: 192.168.1.50
    port: 502
    timeout: 3.0
    retry_count: 3

  devices:
    - slave_id: 1
      name: Boiler1
      registers:
        - name: steam_flow
          address: 40001
          type: holding
          format: float32
          byte_order: ABCD
          scale: 1.0
          unit: lb/hr

        - name: pressure
          address: 40003
          type: holding
          format: int16
          scale: 0.1
          offset: 0
          unit: psig
```

### MQTT Integration

**Purpose:** Publish/subscribe to MQTT brokers for IoT connectivity.

**Setup:**

```python
from gl002.integrations.industrial import MQTTConnector

# Configure MQTT
config = {
    'broker': 'mqtt.broker.com',
    'port': 8883,  # TLS port
    'username': 'gl002',
    'password': 'secure_password',
    'client_id': 'gl002_optimizer',
    'tls': {
        'ca_cert': '/certs/ca.pem',
        'client_cert': '/certs/client.pem',
        'client_key': '/certs/client.key'
    },
    'qos': 1,
    'retain': False
}

# Initialize connector
mqtt = MQTTConnector(config)

# Connect
await mqtt.connect()

# Subscribe to topics
await mqtt.subscribe([
    'boiler/+/data',
    'boiler/+/alarms',
    'system/commands'
], callback=handle_message)

# Publish data
await mqtt.publish(
    topic='boiler/001/optimization',
    payload={
        'timestamp': datetime.now().isoformat(),
        'efficiency': 85.4,
        'recommendations': {...}
    },
    qos=1,
    retain=True
)

# Sparkplug B support
sparkplug_data = mqtt.create_sparkplug_payload(
    metrics={
        'efficiency': 85.4,
        'fuel_rate': 3500,
        'steam_flow': 50000
    }
)
await mqtt.publish_sparkplug('spBv1.0/GL002/DDATA/Boiler1', sparkplug_data)
```

## Data Historian Integrations

### OSIsoft PI System

**Purpose:** Read/write time-series data to PI System.

```python
from gl002.integrations.historians import PIConnector

# Configure PI connection
pi_config = {
    'server': 'pi-server.local',
    'username': 'piuser',
    'password': 'secure_password',
    'database': 'PIDatabase'
}

# Initialize
pi = PIConnector(pi_config)

# Read current value
value = pi.read_tag('BA:BOILER1.EFFICIENCY')

# Read historical data
history = pi.read_history(
    tag='BA:BOILER1.STEAM_FLOW',
    start_time='2025-01-01 00:00:00',
    end_time='2025-01-02 00:00:00',
    interval='1h'
)

# Write value
pi.write_tag('BA:BOILER1.SETPOINT', 485.0)

# Batch read
tags = ['BA:BOILER1.TEMP', 'BA:BOILER1.PRESSURE', 'BA:BOILER1.FLOW']
values = pi.read_tags(tags)
```

### Wonderware Historian

```python
from gl002.integrations.historians import WonderwareConnector

ww = WonderwareConnector({
    'server': 'historian.local',
    'database': 'Runtime',
    'integrated_security': False,
    'username': 'wwuser',
    'password': 'password'
})

# Query data
data = ww.query(
    "SELECT TagName, Value, DateTime "
    "FROM History "
    "WHERE TagName LIKE 'Boiler%' "
    "AND DateTime > '2025-01-01' "
    "AND wwRetrievalMode = 'Cyclic' "
    "AND wwCycleCount = 1000"
)
```

## Cloud Platform Integrations

### AWS IoT Core

```python
from gl002.integrations.cloud import AWSIoTConnector

# Configure AWS IoT
aws_config = {
    'endpoint': 'xxxxx.iot.us-west-2.amazonaws.com',
    'client_id': 'gl002_optimizer',
    'cert_path': '/certs/device.pem.crt',
    'key_path': '/certs/private.pem.key',
    'ca_path': '/certs/AmazonRootCA1.pem',
    'region': 'us-west-2'
}

aws_iot = AWSIoTConnector(aws_config)

# Connect
await aws_iot.connect()

# Publish to shadow
await aws_iot.update_shadow({
    'state': {
        'reported': {
            'efficiency': 85.4,
            'status': 'optimizing'
        }
    }
})

# Subscribe to commands
await aws_iot.subscribe(
    '$aws/things/gl002/shadow/update/delta',
    handle_command
)
```

### Azure IoT Hub

```python
from gl002.integrations.cloud import AzureIoTConnector

azure_config = {
    'connection_string': 'HostName=hub.azure-devices.net;DeviceId=gl002;SharedAccessKey=...',
    'protocol': 'mqtt',
    'port': 8883
}

azure = AzureIoTConnector(azure_config)

# Send telemetry
await azure.send_telemetry({
    'efficiency': 85.4,
    'fuel_rate': 3500,
    'timestamp': datetime.utcnow().isoformat()
})

# Send device twin update
await azure.update_twin({
    'properties': {
        'reported': {
            'firmwareVersion': '2.0.0',
            'lastOptimization': datetime.utcnow().isoformat()
        }
    }
})
```

## Enterprise System Integrations

### SAP Integration

```python
from gl002.integrations.enterprise import SAPConnector

# Configure SAP connection
sap_config = {
    'ashost': 'sap.company.com',
    'sysnr': '00',
    'client': '100',
    'user': 'RFC_USER',
    'passwd': 'password',
    'lang': 'EN'
}

sap = SAPConnector(sap_config)

# Call RFC function
result = sap.call_function(
    'Z_BOILER_UPDATE',
    IV_BOILER_ID='B001',
    IV_EFFICIENCY=85.4,
    IV_FUEL_CONSUMPTION=3500
)

# Read material master
material = sap.get_material('NATURAL_GAS')

# Update measurement document
sap.create_measurement_document({
    'point': 'BOILER_001_EFF',
    'value': 85.4,
    'timestamp': datetime.now(),
    'unit': 'PCT'
})
```

### Oracle EBS Integration

```python
from gl002.integrations.enterprise import OracleConnector

oracle = OracleConnector({
    'host': 'oracle.company.com',
    'port': 1521,
    'service': 'PROD',
    'username': 'apps',
    'password': 'password'
})

# Update asset performance
oracle.update_asset_performance(
    asset_number='BOILER-001',
    efficiency=85.4,
    runtime_hours=168,
    fuel_consumption=588000
)

# Create work order for maintenance
wo = oracle.create_work_order({
    'asset_number': 'BOILER-001',
    'type': 'PREVENTIVE',
    'description': 'Soot blowing required',
    'priority': 2,
    'scheduled_date': datetime.now() + timedelta(days=7)
})
```

## Database Integrations

### TimescaleDB

```python
from gl002.integrations.databases import TimescaleConnector

# Configure TimescaleDB
ts_config = {
    'host': 'localhost',
    'port': 5432,
    'database': 'gl002',
    'user': 'gl002user',
    'password': 'password'
}

timescale = TimescaleConnector(ts_config)

# Create hypertable
timescale.create_hypertable(
    table_name='boiler_data',
    time_column='timestamp',
    chunk_time_interval='1 day'
)

# Insert time-series data
timescale.insert_data(
    table='boiler_data',
    data=[
        {'timestamp': datetime.now(), 'boiler_id': 'B001', 'efficiency': 85.4},
        {'timestamp': datetime.now(), 'boiler_id': 'B001', 'fuel_rate': 3500}
    ]
)

# Query with time-series functions
results = timescale.query("""
    SELECT
        time_bucket('1 hour', timestamp) AS hour,
        avg(efficiency) as avg_efficiency,
        max(efficiency) as max_efficiency
    FROM boiler_data
    WHERE timestamp > NOW() - INTERVAL '7 days'
    GROUP BY hour
    ORDER BY hour DESC
""")

# Set up continuous aggregates
timescale.create_continuous_aggregate(
    name='hourly_efficiency',
    query="""
        SELECT
            time_bucket('1 hour', timestamp) as hour,
            boiler_id,
            avg(efficiency) as avg_efficiency,
            sum(fuel_rate) as total_fuel
        FROM boiler_data
        GROUP BY hour, boiler_id
    """,
    refresh_interval='1 hour'
)
```

### InfluxDB

```python
from gl002.integrations.databases import InfluxDBConnector

influx = InfluxDBConnector({
    'url': 'http://localhost:8086',
    'token': 'your-token',
    'org': 'gl002',
    'bucket': 'boiler_data'
})

# Write data points
influx.write_points([
    {
        'measurement': 'boiler_efficiency',
        'tags': {'boiler_id': 'B001', 'plant': 'Plant1'},
        'fields': {'value': 85.4, 'target': 88.0},
        'timestamp': datetime.utcnow()
    }
])

# Query data
query = '''
    from(bucket: "boiler_data")
        |> range(start: -7d)
        |> filter(fn: (r) => r["_measurement"] == "boiler_efficiency")
        |> filter(fn: (r) => r["boiler_id"] == "B001")
        |> aggregateWindow(every: 1h, fn: mean)
'''
results = influx.query(query)
```

## Configuration Management

### Environment Variables

```bash
# Industrial Protocols
GL002_OPC_SERVER=opc.tcp://localhost:4840
GL002_MODBUS_HOST=192.168.1.50
GL002_MQTT_BROKER=mqtt.broker.com

# Historians
GL002_PI_SERVER=pi-server.local
GL002_WW_SERVER=historian.local

# Cloud
GL002_AWS_IOT_ENDPOINT=xxxxx.iot.us-west-2.amazonaws.com
GL002_AZURE_IOT_CONNECTION=HostName=...

# Enterprise
GL002_SAP_HOST=sap.company.com
GL002_ORACLE_HOST=oracle.company.com

# Databases
GL002_TIMESCALE_URL=postgresql://user:pass@localhost/gl002
GL002_INFLUX_URL=http://localhost:8086
```

### Integration Config File

```yaml
# integrations.yaml
integrations:
  enabled:
    - opc_ua
    - modbus
    - mqtt
    - timescale

  opc_ua:
    primary:
      url: ${GL002_OPC_SERVER}
      security: SignAndEncrypt
      retry_policy:
        max_attempts: 3
        backoff: exponential

  modbus:
    devices:
      - name: boiler1
        host: ${GL002_MODBUS_HOST}
        port: 502
        slave_id: 1
        polling_interval: 1000

  data_flow:
    input:
      - source: opc_ua
        tags: ['efficiency', 'fuel_rate', 'steam_flow']
      - source: modbus
        registers: [40001, 40002, 40003]

    output:
      - destination: timescale
        table: boiler_data
        interval: 60
      - destination: mqtt
        topic: boiler/data
        qos: 1
```

## Troubleshooting

### Connection Issues

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test connection with timeout
try:
    connector = OPCUAConnector(config)
    connector.set_timeout(5000)  # 5 seconds
    await connector.connect()
    print("Connection successful")
except ConnectionError as e:
    print(f"Connection failed: {e}")
    # Check network, firewall, credentials
```

### Data Quality Issues

```python
# Add data validation
def validate_data(data):
    """Validate incoming data."""
    if data['steam_flow'] < 0:
        raise ValueError("Invalid steam flow")
    if not 0 <= data['efficiency'] <= 100:
        raise ValueError("Invalid efficiency")
    return True

# Add retry logic
@retry(max_attempts=3, backoff='exponential')
async def read_with_retry(connector, tags):
    return await connector.read_values(tags)
```

### Performance Optimization

```python
# Batch operations
# Instead of:
for tag in tags:
    value = await connector.read_value(tag)

# Use:
values = await connector.read_values_batch(tags)

# Connection pooling
pool = ConnectionPool(max_connections=10)
connector = OPCUAConnector(config, pool=pool)

# Async operations
async def parallel_read():
    tasks = [
        connector.read_value(tag)
        for tag in tags
    ]
    return await asyncio.gather(*tasks)
```

## Security Best Practices

1. **Use TLS/SSL for all connections**
2. **Store credentials in secure vaults (AWS Secrets Manager, Azure Key Vault)**
3. **Implement connection whitelisting**
4. **Use certificate-based authentication where possible**
5. **Enable audit logging for all operations**
6. **Regular security scanning and updates**

## Support

- **Documentation:** https://docs.greenlang.io/gl002/integrations
- **Integration Support:** gl002-integrations@greenlang.io
- **Community Forum:** https://community.greenlang.io/integrations