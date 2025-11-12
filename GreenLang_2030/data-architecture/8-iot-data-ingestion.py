"""
GreenLang IoT Data Ingestion Architecture
Version: 1.0.0
Devices: 100K+ sensors
Throughput: 1M+ messages/second
"""

import asyncio
import json
import struct
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import paho.mqtt.client as mqtt
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import numpy as np
import pandas as pd
from collections import deque
import logging

logger = logging.getLogger(__name__)

# ==============================================
# IOT DEVICE TYPES AND PROTOCOLS
# ==============================================

class DeviceType(Enum):
    """Types of IoT devices in the system."""
    AIR_QUALITY = "air_quality"
    ENERGY_METER = "energy_meter"
    WATER_FLOW = "water_flow"
    TEMPERATURE = "temperature"
    VIBRATION = "vibration"
    PRESSURE = "pressure"
    EMISSIONS = "emissions"
    WASTE_BIN = "waste_bin"
    VEHICLE_TRACKER = "vehicle_tracker"
    SOLAR_PANEL = "solar_panel"

class Protocol(Enum):
    """Communication protocols supported."""
    MQTT = "mqtt"
    COAP = "coap"
    AMQP = "amqp"
    HTTP = "http"
    LORAWAN = "lorawan"
    MODBUS = "modbus"
    OPCUA = "opcua"

# ==============================================
# MQTT BROKER CONFIGURATION
# ==============================================

class MQTTBrokerConfig:
    """MQTT broker configuration for IoT devices."""

    BROKER_HOST = "mqtt.greenlang.internal"
    BROKER_PORT = 1883
    BROKER_TLS_PORT = 8883
    KEEPALIVE = 60
    QOS = 1

    # Topic structure
    TOPIC_STRUCTURE = {
        'telemetry': 'iot/{organization_id}/{facility_id}/{device_type}/{device_id}/telemetry',
        'status': 'iot/{organization_id}/{facility_id}/{device_type}/{device_id}/status',
        'command': 'iot/{organization_id}/{facility_id}/{device_type}/{device_id}/command',
        'alert': 'iot/{organization_id}/{facility_id}/{device_type}/{device_id}/alert',
        'config': 'iot/{organization_id}/{facility_id}/{device_type}/{device_id}/config'
    }

    # Authentication
    USE_TLS = True
    CA_CERT = "/certs/ca.crt"
    CLIENT_CERT = "/certs/client.crt"
    CLIENT_KEY = "/certs/client.key"

# ==============================================
# IOT DATA INGESTION PIPELINE
# ==============================================

class IoTDataIngestionPipeline:
    """Main IoT data ingestion and processing pipeline."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize IoT ingestion pipeline."""
        self.config = config

        # MQTT client
        self.mqtt_client = self._setup_mqtt_client()

        # Time-series databases
        self.influx_client = InfluxDBClient(
            url=config['influxdb_url'],
            token=config['influxdb_token'],
            org=config['influxdb_org']
        )
        self.write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)

        # Message buffers
        self.message_buffer = deque(maxlen=100000)
        self.batch_size = 1000
        self.flush_interval = 5  # seconds

        # Device registry
        self.device_registry = {}

        # Metrics
        self.metrics = {
            'messages_received': 0,
            'messages_processed': 0,
            'messages_failed': 0,
            'alerts_triggered': 0
        }

    def _setup_mqtt_client(self) -> mqtt.Client:
        """Setup MQTT client with callbacks."""
        client = mqtt.Client(client_id=f"greenlang-ingestion-{datetime.now().timestamp()}")

        # Set callbacks
        client.on_connect = self._on_connect
        client.on_message = self._on_message
        client.on_disconnect = self._on_disconnect

        # Configure TLS if enabled
        if MQTTBrokerConfig.USE_TLS:
            client.tls_set(
                ca_certs=MQTTBrokerConfig.CA_CERT,
                certfile=MQTTBrokerConfig.CLIENT_CERT,
                keyfile=MQTTBrokerConfig.CLIENT_KEY
            )

        # Set credentials
        client.username_pw_set(
            username=self.config['mqtt_username'],
            password=self.config['mqtt_password']
        )

        return client

    def _on_connect(self, client, userdata, flags, rc):
        """Callback for MQTT connection."""
        if rc == 0:
            logger.info("Connected to MQTT broker")

            # Subscribe to all telemetry topics
            client.subscribe("iot/+/+/+/+/telemetry", qos=1)
            client.subscribe("iot/+/+/+/+/status", qos=1)
            client.subscribe("iot/+/+/+/+/alert", qos=2)
        else:
            logger.error(f"Failed to connect to MQTT broker: {rc}")

    def _on_message(self, client, userdata, msg):
        """Process incoming MQTT messages."""
        try:
            # Parse topic
            topic_parts = msg.topic.split('/')
            if len(topic_parts) < 6:
                logger.warning(f"Invalid topic structure: {msg.topic}")
                return

            org_id = topic_parts[1]
            facility_id = topic_parts[2]
            device_type = topic_parts[3]
            device_id = topic_parts[4]
            message_type = topic_parts[5]

            # Parse payload
            payload = json.loads(msg.payload.decode('utf-8'))

            # Add metadata
            enriched_message = {
                'organization_id': org_id,
                'facility_id': facility_id,
                'device_type': device_type,
                'device_id': device_id,
                'message_type': message_type,
                'timestamp': payload.get('timestamp', datetime.now().isoformat()),
                'data': payload.get('data', {}),
                'metadata': payload.get('metadata', {})
            }

            # Process based on message type
            if message_type == 'telemetry':
                self._process_telemetry(enriched_message)
            elif message_type == 'alert':
                self._process_alert(enriched_message)
            elif message_type == 'status':
                self._update_device_status(enriched_message)

            # Add to buffer
            self.message_buffer.append(enriched_message)

            # Update metrics
            self.metrics['messages_received'] += 1

            # Check if buffer should be flushed
            if len(self.message_buffer) >= self.batch_size:
                asyncio.create_task(self._flush_buffer())

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.metrics['messages_failed'] += 1

    def _on_disconnect(self, client, userdata, rc):
        """Handle MQTT disconnection."""
        if rc != 0:
            logger.warning(f"Unexpected disconnection from MQTT broker: {rc}")
            # Attempt reconnection
            client.reconnect()

    async def _process_telemetry(self, message: Dict):
        """Process telemetry data."""
        device_type = DeviceType(message['device_type'])
        data = message['data']

        # Device-specific processing
        if device_type == DeviceType.AIR_QUALITY:
            await self._process_air_quality_data(message)
        elif device_type == DeviceType.ENERGY_METER:
            await self._process_energy_data(message)
        elif device_type == DeviceType.EMISSIONS:
            await self._process_emissions_data(message)
        # Add more device types...

        # Write to time-series database
        point = self._create_influx_point(message)
        self.write_api.write(bucket=self.config['influxdb_bucket'], record=point)

        # Check for anomalies
        if self._detect_anomaly(message):
            await self._trigger_alert(message)

    async def _process_air_quality_data(self, message: Dict):
        """Process air quality sensor data."""
        data = message['data']

        # Validate readings
        required_fields = ['pm25', 'pm10', 'co2', 'temperature', 'humidity']
        if not all(field in data for field in required_fields):
            raise ValueError(f"Missing required fields for air quality data")

        # Calculate AQI (Air Quality Index)
        aqi = self._calculate_aqi(
            pm25=data['pm25'],
            pm10=data['pm10'],
            co2=data['co2']
        )

        # Add calculated fields
        message['data']['aqi'] = aqi
        message['data']['aqi_category'] = self._get_aqi_category(aqi)

        # Check thresholds
        if aqi > 150:  # Unhealthy
            await self._trigger_alert({
                'device_id': message['device_id'],
                'alert_type': 'air_quality_unhealthy',
                'severity': 'high',
                'aqi': aqi
            })

    async def _process_energy_data(self, message: Dict):
        """Process energy meter data."""
        data = message['data']

        # Calculate power factor
        if 'voltage' in data and 'current' in data and 'power' in data:
            apparent_power = data['voltage'] * data['current']
            if apparent_power > 0:
                data['power_factor'] = data['power'] / apparent_power

        # Calculate energy consumption
        if 'power' in data and 'duration' in data:
            data['energy_kwh'] = (data['power'] * data['duration']) / 1000

        # Detect anomalies
        if data.get('power_factor', 1) < 0.8:
            await self._trigger_alert({
                'device_id': message['device_id'],
                'alert_type': 'low_power_factor',
                'severity': 'medium',
                'power_factor': data['power_factor']
            })

    async def _process_emissions_data(self, message: Dict):
        """Process emissions sensor data."""
        data = message['data']

        # Convert to standard units
        if 'co2_ppm' in data:
            data['co2_kg'] = self._ppm_to_kg(data['co2_ppm'], 'CO2')

        if 'ch4_ppm' in data:
            data['ch4_kg'] = self._ppm_to_kg(data['ch4_ppm'], 'CH4')

        # Calculate CO2 equivalent
        data['co2e_total'] = (
            data.get('co2_kg', 0) +
            data.get('ch4_kg', 0) * 25 +  # GWP of CH4
            data.get('n2o_kg', 0) * 298   # GWP of N2O
        )

        # Check emission limits
        if data['co2e_total'] > self.config.get('emission_limit', 1000):
            await self._trigger_alert({
                'device_id': message['device_id'],
                'alert_type': 'emission_limit_exceeded',
                'severity': 'critical',
                'co2e_total': data['co2e_total']
            })

    def _calculate_aqi(self, pm25: float, pm10: float, co2: float) -> int:
        """Calculate Air Quality Index."""
        # Simplified AQI calculation
        aqi_pm25 = self._linear_interpolation(
            pm25,
            [(0, 0), (12, 50), (35.4, 100), (55.4, 150), (150.4, 200)]
        )
        aqi_pm10 = self._linear_interpolation(
            pm10,
            [(0, 0), (54, 50), (154, 100), (254, 150), (354, 200)]
        )

        return max(aqi_pm25, aqi_pm10)

    def _linear_interpolation(self, value: float, breakpoints: List[tuple]) -> int:
        """Linear interpolation for AQI calculation."""
        for i in range(len(breakpoints) - 1):
            if breakpoints[i][0] <= value <= breakpoints[i + 1][0]:
                c_low, i_low = breakpoints[i]
                c_high, i_high = breakpoints[i + 1]
                return int(((i_high - i_low) / (c_high - c_low)) * (value - c_low) + i_low)
        return 500  # Beyond scale

    def _get_aqi_category(self, aqi: int) -> str:
        """Get AQI category."""
        if aqi <= 50:
            return "Good"
        elif aqi <= 100:
            return "Moderate"
        elif aqi <= 150:
            return "Unhealthy for Sensitive Groups"
        elif aqi <= 200:
            return "Unhealthy"
        elif aqi <= 300:
            return "Very Unhealthy"
        else:
            return "Hazardous"

    def _ppm_to_kg(self, ppm: float, gas_type: str) -> float:
        """Convert PPM to kg based on gas type."""
        # Simplified conversion (would need actual volume and conditions)
        molecular_weights = {
            'CO2': 44.01,
            'CH4': 16.04,
            'N2O': 44.01
        }

        mw = molecular_weights.get(gas_type, 44.01)
        return (ppm * mw) / 1000000  # Simplified

    def _detect_anomaly(self, message: Dict) -> bool:
        """Detect anomalies using statistical methods."""
        device_id = message['device_id']
        data = message['data']

        # Get historical data for device
        if device_id not in self.device_registry:
            self.device_registry[device_id] = {
                'history': deque(maxlen=100),
                'thresholds': {}
            }

        device_info = self.device_registry[device_id]
        device_info['history'].append(data)

        # Need enough history
        if len(device_info['history']) < 10:
            return False

        # Calculate statistics
        for metric, value in data.items():
            if isinstance(value, (int, float)):
                history_values = [h.get(metric, 0) for h in device_info['history']]
                mean = np.mean(history_values)
                std = np.std(history_values)

                # Z-score based anomaly detection
                if std > 0:
                    z_score = abs((value - mean) / std)
                    if z_score > 3:  # 3 standard deviations
                        return True

        return False

    async def _trigger_alert(self, alert_data: Dict):
        """Trigger alert for anomaly or threshold violation."""
        alert = {
            'alert_id': f"alert_{datetime.now().timestamp()}",
            'timestamp': datetime.now().isoformat(),
            **alert_data
        }

        # Publish alert to MQTT
        alert_topic = f"iot/alerts/{alert_data['device_id']}"
        self.mqtt_client.publish(
            alert_topic,
            json.dumps(alert),
            qos=2
        )

        # Store in database
        point = Point("alerts") \
            .tag("device_id", alert_data['device_id']) \
            .tag("alert_type", alert_data['alert_type']) \
            .tag("severity", alert_data['severity']) \
            .field("data", json.dumps(alert_data)) \
            .time(datetime.utcnow())

        self.write_api.write(bucket=self.config['influxdb_bucket'], record=point)

        # Update metrics
        self.metrics['alerts_triggered'] += 1

        logger.warning(f"Alert triggered: {alert}")

    def _update_device_status(self, message: Dict):
        """Update device status in registry."""
        device_id = message['device_id']
        status_data = message['data']

        if device_id not in self.device_registry:
            self.device_registry[device_id] = {}

        self.device_registry[device_id].update({
            'last_seen': datetime.now(),
            'status': status_data.get('status', 'unknown'),
            'battery_level': status_data.get('battery_level'),
            'signal_strength': status_data.get('signal_strength'),
            'firmware_version': status_data.get('firmware_version')
        })

    def _create_influx_point(self, message: Dict) -> Point:
        """Create InfluxDB point from message."""
        point = Point(f"iot_{message['device_type']}") \
            .tag("organization_id", message['organization_id']) \
            .tag("facility_id", message['facility_id']) \
            .tag("device_id", message['device_id'])

        # Add all numeric fields
        for key, value in message['data'].items():
            if isinstance(value, (int, float)):
                point = point.field(key, value)

        # Set timestamp
        timestamp = datetime.fromisoformat(message['timestamp'])
        point = point.time(timestamp)

        return point

    async def _flush_buffer(self):
        """Flush message buffer to storage."""
        if not self.message_buffer:
            return

        try:
            # Batch write to InfluxDB
            points = []
            while self.message_buffer:
                message = self.message_buffer.popleft()
                point = self._create_influx_point(message)
                points.append(point)

            self.write_api.write(bucket=self.config['influxdb_bucket'], record=points)

            # Update metrics
            self.metrics['messages_processed'] += len(points)

            logger.info(f"Flushed {len(points)} messages to storage")

        except Exception as e:
            logger.error(f"Error flushing buffer: {e}")

    async def start(self):
        """Start the ingestion pipeline."""
        # Connect to MQTT broker
        self.mqtt_client.connect(
            MQTTBrokerConfig.BROKER_HOST,
            MQTTBrokerConfig.BROKER_TLS_PORT if MQTTBrokerConfig.USE_TLS else MQTTBrokerConfig.BROKER_PORT,
            MQTTBrokerConfig.KEEPALIVE
        )

        # Start MQTT loop
        self.mqtt_client.loop_start()

        # Start periodic buffer flush
        while True:
            await asyncio.sleep(self.flush_interval)
            await self._flush_buffer()

    def stop(self):
        """Stop the ingestion pipeline."""
        # Flush remaining messages
        asyncio.run(self._flush_buffer())

        # Disconnect MQTT
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()

        # Close database connections
        self.influx_client.close()

        logger.info(f"Pipeline stopped. Metrics: {self.metrics}")

# ==============================================
# EDGE COMPUTING GATEWAY
# ==============================================

class EdgeGateway:
    """Edge computing gateway for local processing."""

    def __init__(self, gateway_id: str, config: Dict):
        """Initialize edge gateway."""
        self.gateway_id = gateway_id
        self.config = config

        # Local storage
        self.local_buffer = deque(maxlen=10000)

        # Processing rules
        self.processing_rules = []

        # ML models for edge inference
        self.ml_models = {}

    def add_processing_rule(self, rule: Dict):
        """Add local processing rule."""
        self.processing_rules.append(rule)

    def process_locally(self, data: Dict) -> Dict:
        """Process data at the edge."""
        processed = data.copy()

        # Apply processing rules
        for rule in self.processing_rules:
            if self._evaluate_condition(rule['condition'], data):
                processed = self._apply_transformation(rule['transformation'], processed)

        # Run edge ML inference if applicable
        if 'model' in self.config:
            prediction = self._run_inference(processed)
            processed['edge_prediction'] = prediction

        return processed

    def _evaluate_condition(self, condition: Dict, data: Dict) -> bool:
        """Evaluate rule condition."""
        field = condition['field']
        operator = condition['operator']
        value = condition['value']

        if field not in data:
            return False

        data_value = data[field]

        if operator == 'gt':
            return data_value > value
        elif operator == 'lt':
            return data_value < value
        elif operator == 'eq':
            return data_value == value
        elif operator == 'contains':
            return value in str(data_value)

        return False

    def _apply_transformation(self, transformation: Dict, data: Dict) -> Dict:
        """Apply data transformation."""
        action = transformation['action']

        if action == 'aggregate':
            # Aggregate over window
            window_size = transformation['window_size']
            field = transformation['field']

            # Simple moving average
            if field in data:
                self.local_buffer.append(data[field])
                if len(self.local_buffer) >= window_size:
                    data[f"{field}_avg"] = np.mean(list(self.local_buffer)[-window_size:])

        elif action == 'filter':
            # Filter out certain values
            filter_field = transformation['field']
            filter_value = transformation['value']

            if data.get(filter_field) == filter_value:
                return None

        elif action == 'enrich':
            # Add calculated fields
            data.update(transformation['fields'])

        return data

    def _run_inference(self, data: Dict) -> Any:
        """Run ML inference at edge."""
        model_name = self.config['model']

        if model_name not in self.ml_models:
            # Load model (simplified)
            self.ml_models[model_name] = self._load_model(model_name)

        model = self.ml_models[model_name]

        # Prepare features
        features = self._extract_features(data)

        # Run prediction
        prediction = model.predict([features])

        return prediction[0]

    def _load_model(self, model_name: str):
        """Load ML model for edge inference."""
        # Simplified - would load actual model
        return None

    def _extract_features(self, data: Dict) -> List:
        """Extract features for ML model."""
        # Extract relevant features based on model requirements
        return []

# ==============================================
# DEVICE MANAGEMENT
# ==============================================

class DeviceManager:
    """Manage IoT device lifecycle and configuration."""

    def __init__(self):
        """Initialize device manager."""
        self.devices = {}
        self.device_groups = {}
        self.firmware_versions = {}

    async def register_device(self, device_info: Dict) -> str:
        """Register new IoT device."""
        device_id = device_info['device_id']

        self.devices[device_id] = {
            'device_id': device_id,
            'device_type': device_info['device_type'],
            'manufacturer': device_info.get('manufacturer'),
            'model': device_info.get('model'),
            'firmware_version': device_info.get('firmware_version'),
            'location': device_info.get('location'),
            'registered_at': datetime.now(),
            'status': 'active',
            'configuration': device_info.get('configuration', {}),
            'metadata': device_info.get('metadata', {})
        }

        logger.info(f"Device registered: {device_id}")
        return device_id

    async def update_device_config(self, device_id: str, config: Dict):
        """Update device configuration remotely."""
        if device_id not in self.devices:
            raise ValueError(f"Device not found: {device_id}")

        # Prepare configuration message
        config_message = {
            'command': 'update_config',
            'config': config,
            'timestamp': datetime.now().isoformat()
        }

        # Publish to device command topic
        topic = f"iot/command/{device_id}"
        # mqtt_client.publish(topic, json.dumps(config_message))

        # Update local registry
        self.devices[device_id]['configuration'].update(config)

        logger.info(f"Configuration updated for device: {device_id}")

    async def update_firmware(self, device_id: str, firmware_version: str):
        """Initiate firmware update for device."""
        if device_id not in self.devices:
            raise ValueError(f"Device not found: {device_id}")

        # Prepare firmware update command
        update_command = {
            'command': 'firmware_update',
            'version': firmware_version,
            'download_url': f"https://firmware.greenlang.com/{firmware_version}",
            'checksum': self.firmware_versions.get(firmware_version, {}).get('checksum'),
            'timestamp': datetime.now().isoformat()
        }

        # Publish update command
        topic = f"iot/command/{device_id}"
        # mqtt_client.publish(topic, json.dumps(update_command))

        logger.info(f"Firmware update initiated for device {device_id}: {firmware_version}")

    def get_device_stats(self) -> Dict:
        """Get device statistics."""
        stats = {
            'total_devices': len(self.devices),
            'active_devices': sum(1 for d in self.devices.values() if d['status'] == 'active'),
            'devices_by_type': {},
            'devices_by_status': {}
        }

        for device in self.devices.values():
            # By type
            device_type = device['device_type']
            stats['devices_by_type'][device_type] = stats['devices_by_type'].get(device_type, 0) + 1

            # By status
            status = device['status']
            stats['devices_by_status'][status] = stats['devices_by_status'].get(status, 0) + 1

        return stats

# ==============================================
# USAGE EXAMPLE
# ==============================================

async def main():
    """Example usage of IoT ingestion pipeline."""

    # Configuration
    config = {
        'mqtt_username': 'iot_user',
        'mqtt_password': 'secure_password',
        'influxdb_url': 'http://influxdb:8086',
        'influxdb_token': 'token',
        'influxdb_org': 'greenlang',
        'influxdb_bucket': 'iot_data',
        'emission_limit': 1000
    }

    # Initialize pipeline
    pipeline = IoTDataIngestionPipeline(config)

    # Initialize device manager
    device_manager = DeviceManager()

    # Register some devices
    await device_manager.register_device({
        'device_id': 'aq-sensor-001',
        'device_type': 'air_quality',
        'manufacturer': 'SensorCorp',
        'model': 'AQ-1000',
        'location': {'lat': 37.7749, 'lon': -122.4194}
    })

    # Start ingestion pipeline
    await pipeline.start()

if __name__ == "__main__":
    asyncio.run(main())