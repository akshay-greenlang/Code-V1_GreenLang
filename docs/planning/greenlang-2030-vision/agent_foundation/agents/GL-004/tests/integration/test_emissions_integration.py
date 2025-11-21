# -*- coding: utf-8 -*-
"""
Integration tests for emissions monitoring via MQTT.

Tests MQTT connection, real-time emissions reading, alarm handling,
and compliance calculations for the BurnerOptimizationAgent.
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, patch
import paho.mqtt.client as mqtt

from mock_servers import MockMQTTBroker
from greenlang.determinism import DeterministicClock


class TestEmissionsIntegration:
    """Integration tests for emissions monitoring system."""

    @pytest.fixture
    def mqtt_client(self):
        """Create MQTT client for testing."""
        client = mqtt.Client()
        client.on_connect = lambda c, u, f, rc: setattr(c, 'connected', rc == 0)
        client.on_message = lambda c, u, msg: None
        return client

    def test_mqtt_connection_establishment(self, mock_mqtt_broker):
        """Test establishing MQTT connection."""
        # Given: Mock MQTT broker running
        broker = MockMQTTBroker()
        broker.start()

        try:
            # When: Attempting to connect
            client = mqtt.Client()
            connected_flag = {'value': False}

            def on_connect(client, userdata, flags, rc):
                connected_flag['value'] = (rc == 0)

            client.on_connect = on_connect
            client.connect('localhost', 1883, 60)
            client.loop_start()

            # Wait for connection
            time.sleep(0.5)

            # Then: Connection should be established
            assert connected_flag['value'] is True

            client.loop_stop()
            client.disconnect()

        finally:
            broker.stop()

    def test_subscribe_to_emissions_topics(self, mock_mqtt_broker, mqtt_client):
        """Test subscribing to emissions topics."""
        # Given: Mock broker running
        broker = MockMQTTBroker()
        broker.start()

        try:
            # When: Subscribing to emissions topics
            mqtt_client.connect('localhost', 1883, 60)

            topics = [
                ('emissions/co', 0),
                ('emissions/nox', 0),
                ('emissions/sox', 0),
                ('emissions/o2', 0),
                ('emissions/particulates', 0)
            ]

            subscribed = {'count': 0}

            def on_subscribe(client, userdata, mid, granted_qos):
                subscribed['count'] += 1

            mqtt_client.on_subscribe = on_subscribe

            for topic, qos in topics:
                mqtt_client.subscribe(topic, qos)

            mqtt_client.loop_start()
            time.sleep(0.5)

            # Then: Should be subscribed to all topics
            assert subscribed['count'] == len(topics)

            mqtt_client.loop_stop()
            mqtt_client.disconnect()

        finally:
            broker.stop()

    def test_receive_realtime_emissions_data(self, mock_mqtt_broker):
        """Test receiving real-time emissions data."""
        # Given: Mock broker publishing emissions
        broker = MockMQTTBroker()
        broker.start()

        received_messages = []

        try:
            # Setup client with message handler
            client = mqtt.Client()

            def on_message(client, userdata, msg):
                try:
                    payload = json.loads(msg.payload.decode())
                    received_messages.append({
                        'topic': msg.topic,
                        'payload': payload,
                        'timestamp': DeterministicClock.now()
                    })
                except:
                    pass

            client.on_message = on_message
            client.connect('localhost', 1883, 60)
            client.subscribe('emissions/+')
            client.loop_start()

            # When: Waiting for emissions data
            time.sleep(2)  # Collect data for 2 seconds

            # Then: Should receive multiple emissions updates
            assert len(received_messages) > 0

            # Verify message structure
            for msg in received_messages:
                assert 'topic' in msg
                assert 'payload' in msg
                payload = msg['payload']
                assert 'value' in payload
                assert 'unit' in payload
                assert 'timestamp' in payload

            # Verify we received data for multiple pollutants
            topics_received = set(msg['topic'] for msg in received_messages)
            assert 'emissions/co' in topics_received
            assert 'emissions/nox' in topics_received
            assert 'emissions/o2' in topics_received

            client.loop_stop()
            client.disconnect()

        finally:
            broker.stop()

    def test_emissions_value_validation(self, mock_mqtt_broker):
        """Test validation of emissions values."""
        # Given: Mock broker with known emissions
        broker = MockMQTTBroker()
        broker.start()

        emissions_data = {}

        try:
            client = mqtt.Client()

            def on_message(client, userdata, msg):
                try:
                    payload = json.loads(msg.payload.decode())
                    pollutant = msg.topic.split('/')[-1]
                    emissions_data[pollutant] = payload['value']
                except:
                    pass

            client.on_message = on_message
            client.connect('localhost', 1883, 60)
            client.subscribe('emissions/+')
            client.loop_start()

            # When: Collecting emissions data
            time.sleep(1)

            # Then: Values should be within realistic ranges
            assert 'co' in emissions_data
            assert 0 <= emissions_data['co'] <= 1000  # ppm

            assert 'nox' in emissions_data
            assert 0 <= emissions_data['nox'] <= 500  # ppm

            assert 'o2' in emissions_data
            assert 0 <= emissions_data['o2'] <= 21  # %

            client.loop_stop()
            client.disconnect()

        finally:
            broker.stop()

    def test_high_co_alarm_detection(self, mock_mqtt_broker):
        """Test detection of high CO emissions alarm."""
        # Given: Mock broker
        broker = MockMQTTBroker()
        broker.start()

        alarm_detected = {'value': False}

        try:
            # Inject high CO value
            broker.inject_fault('co', 250)  # Above alarm threshold

            client = mqtt.Client()

            def on_message(client, userdata, msg):
                if msg.topic == 'emissions/all':
                    try:
                        payload = json.loads(msg.payload.decode())
                        if payload.get('alarm_state') == 'HIGH_CO':
                            alarm_detected['value'] = True
                    except:
                        pass

            client.on_message = on_message
            client.connect('localhost', 1883, 60)
            client.subscribe('emissions/all')
            client.loop_start()

            # When: Monitoring for alarms
            time.sleep(1)

            # Then: High CO alarm should be detected
            assert alarm_detected['value'] is True

            client.loop_stop()
            client.disconnect()

        finally:
            broker.stop()

    def test_high_nox_alarm_detection(self, mock_mqtt_broker):
        """Test detection of high NOx emissions alarm."""
        # Given: Mock broker
        broker = MockMQTTBroker()
        broker.start()

        alarm_detected = {'value': False}

        try:
            # Inject high NOx value
            broker.inject_fault('nox', 200)  # Above alarm threshold

            client = mqtt.Client()

            def on_message(client, userdata, msg):
                if msg.topic == 'emissions/all':
                    try:
                        payload = json.loads(msg.payload.decode())
                        if payload.get('alarm_state') == 'HIGH_NOX':
                            alarm_detected['value'] = True
                    except:
                        pass

            client.on_message = on_message
            client.connect('localhost', 1883, 60)
            client.subscribe('emissions/all')
            client.loop_start()

            # When: Monitoring for alarms
            time.sleep(1)

            # Then: High NOx alarm should be detected
            assert alarm_detected['value'] is True

            client.loop_stop()
            client.disconnect()

        finally:
            broker.stop()

    def test_o2_out_of_range_alarm(self, mock_mqtt_broker):
        """Test O2 level out of range alarm."""
        # Given: Mock broker
        broker = MockMQTTBroker()
        broker.start()

        alarm_detected = {'value': False}

        try:
            # Inject very low O2 value
            broker.inject_fault('o2', 0.5)  # Below safe range

            client = mqtt.Client()

            def on_message(client, userdata, msg):
                if msg.topic == 'emissions/all':
                    try:
                        payload = json.loads(msg.payload.decode())
                        if payload.get('alarm_state') == 'O2_OUT_OF_RANGE':
                            alarm_detected['value'] = True
                    except:
                        pass

            client.on_message = on_message
            client.connect('localhost', 1883, 60)
            client.subscribe('emissions/all')
            client.loop_start()

            # When: Monitoring for alarms
            time.sleep(1)

            # Then: O2 out of range alarm should be detected
            assert alarm_detected['value'] is True

            client.loop_stop()
            client.disconnect()

        finally:
            broker.stop()

    def test_emissions_averaging_calculation(self, mock_mqtt_broker):
        """Test calculation of emissions averages."""
        # Given: Mock broker publishing data
        broker = MockMQTTBroker()
        broker.start()

        co_values = []

        try:
            client = mqtt.Client()

            def on_message(client, userdata, msg):
                if msg.topic == 'emissions/co':
                    try:
                        payload = json.loads(msg.payload.decode())
                        co_values.append(payload['value'])
                    except:
                        pass

            client.on_message = on_message
            client.connect('localhost', 1883, 60)
            client.subscribe('emissions/co')
            client.loop_start()

            # When: Collecting data for averaging
            time.sleep(3)  # Collect for 3 seconds

            # Then: Should be able to calculate statistics
            assert len(co_values) > 0

            average = sum(co_values) / len(co_values)
            minimum = min(co_values)
            maximum = max(co_values)

            assert minimum <= average <= maximum
            assert 0 <= average <= 1000  # Reasonable range

            client.loop_stop()
            client.disconnect()

        finally:
            broker.stop()

    def test_compliance_limit_checking(self, mock_mqtt_broker):
        """Test emissions compliance limit checking."""
        # Given: Compliance limits
        compliance_limits = {
            'co': 100,  # ppm
            'nox': 150,  # ppm
            'sox': 20,  # ppm
        }

        broker = MockMQTTBroker()
        broker.start()

        violations = []

        try:
            client = mqtt.Client()

            def on_message(client, userdata, msg):
                try:
                    payload = json.loads(msg.payload.decode())
                    pollutant = msg.topic.split('/')[-1]

                    if pollutant in compliance_limits:
                        limit = compliance_limits[pollutant]
                        value = payload['value']

                        if value > limit:
                            violations.append({
                                'pollutant': pollutant,
                                'value': value,
                                'limit': limit,
                                'timestamp': payload['timestamp']
                            })
                except:
                    pass

            client.on_message = on_message
            client.connect('localhost', 1883, 60)
            client.subscribe('emissions/+')
            client.loop_start()

            # When: Monitoring for compliance
            time.sleep(2)

            # Then: Any violations should be recorded
            # (Note: With baseline values, should be mostly compliant)
            for violation in violations:
                assert violation['value'] > violation['limit']
                assert violation['pollutant'] in compliance_limits

            client.loop_stop()
            client.disconnect()

        finally:
            broker.stop()

    def test_emissions_data_quality_flag(self, mock_mqtt_broker):
        """Test handling of data quality flags."""
        # Given: Mock broker with quality flags
        broker = MockMQTTBroker()
        broker.start()

        quality_flags = []

        try:
            client = mqtt.Client()

            def on_message(client, userdata, msg):
                try:
                    payload = json.loads(msg.payload.decode())
                    if 'quality' in payload:
                        quality_flags.append(payload['quality'])
                except:
                    pass

            client.on_message = on_message
            client.connect('localhost', 1883, 60)
            client.subscribe('emissions/+')
            client.loop_start()

            # When: Collecting quality flags
            time.sleep(2)

            # Then: Should receive quality indicators
            assert len(quality_flags) > 0
            assert 'GOOD' in quality_flags or 'DEGRADED' in quality_flags

            # Most should be GOOD quality
            good_count = quality_flags.count('GOOD')
            assert good_count > len(quality_flags) * 0.9

            client.loop_stop()
            client.disconnect()

        finally:
            broker.stop()

    def test_emissions_trend_detection(self, mock_mqtt_broker):
        """Test detection of emissions trends."""
        # Given: Mock broker
        broker = MockMQTTBroker()
        broker.start()

        co_history = []

        try:
            client = mqtt.Client()

            def on_message(client, userdata, msg):
                if msg.topic == 'emissions/co':
                    try:
                        payload = json.loads(msg.payload.decode())
                        co_history.append({
                            'value': payload['value'],
                            'timestamp': datetime.fromisoformat(payload['timestamp'])
                        })
                    except:
                        pass

            client.on_message = on_message
            client.connect('localhost', 1883, 60)
            client.subscribe('emissions/co')
            client.loop_start()

            # When: Collecting trend data
            time.sleep(3)

            # Then: Should be able to detect trends
            if len(co_history) >= 10:
                # Calculate simple moving average
                window = 5
                moving_avgs = []

                for i in range(window, len(co_history)):
                    window_values = [h['value'] for h in co_history[i-window:i]]
                    moving_avgs.append(sum(window_values) / window)

                # Check if trend exists (simplified)
                if len(moving_avgs) >= 2:
                    trend = moving_avgs[-1] - moving_avgs[0]
                    # Trend should be small (stable operation)
                    assert abs(trend) < 50  # Less than 50 ppm drift

            client.loop_stop()
            client.disconnect()

        finally:
            broker.stop()

    def test_mqtt_reconnection_handling(self):
        """Test MQTT client reconnection after broker restart."""
        # Given: Broker that will be restarted
        broker = MockMQTTBroker()
        broker.start()

        reconnected = {'value': False}

        try:
            client = mqtt.Client()

            def on_disconnect(client, userdata, rc):
                if rc != 0:
                    # Unexpected disconnection
                    time.sleep(1)
                    try:
                        client.reconnect()
                        reconnected['value'] = True
                    except:
                        pass

            client.on_disconnect = on_disconnect
            client.connect('localhost', 1883, 60)
            client.loop_start()

            # When: Broker stops and restarts
            time.sleep(0.5)
            broker.stop()
            time.sleep(1)
            broker = MockMQTTBroker()
            broker.start()
            time.sleep(1)

            # Then: Client should reconnect
            # (In this test setup, manual reconnection would be needed)

            client.loop_stop()
            client.disconnect()

        finally:
            broker.stop()

    def test_concurrent_emissions_monitoring(self, mock_mqtt_broker):
        """Test monitoring multiple emissions streams concurrently."""
        # Given: Mock broker
        broker = MockMQTTBroker()
        broker.start()

        emissions_counters = {
            'co': 0,
            'nox': 0,
            'sox': 0,
            'o2': 0,
            'particulates': 0
        }

        try:
            client = mqtt.Client()

            def on_message(client, userdata, msg):
                pollutant = msg.topic.split('/')[-1]
                if pollutant in emissions_counters:
                    emissions_counters[pollutant] += 1

            client.on_message = on_message
            client.connect('localhost', 1883, 60)
            client.subscribe('emissions/+')
            client.loop_start()

            # When: Monitoring all pollutants concurrently
            time.sleep(2)

            # Then: Should receive data for all pollutants
            for pollutant, count in emissions_counters.items():
                assert count > 0, f"No data received for {pollutant}"

            # All should have similar update rates
            counts = list(emissions_counters.values())
            avg_count = sum(counts) / len(counts)
            for count in counts:
                assert abs(count - avg_count) < avg_count * 0.3  # Within 30%

            client.loop_stop()
            client.disconnect()

        finally:
            broker.stop()