# -*- coding: utf-8 -*-
"""
Complete Integration Example for GL-003 SteamSystemAnalyzer

Demonstrates full integration of all connectors in a real-world steam system monitoring scenario.

This example shows:
- Multi-sensor integration (steam meters, pressure, temperature)
- SCADA connectivity
- Condensate monitoring
- Data transformation and quality validation
- Agent coordination
- Real-time monitoring and alerting
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

from greenlang.determinism import DeterministicClock
from integrations import (
    # Steam Meter
    SteamMeterConnector,
    SteamMeterConfig,
    MeterProtocol,

    # Pressure Sensors
    PressureSensorConnector,
    PressureSensorConfig,
    PressureType,
    PressureSensorType,

    # Temperature Sensors
    TemperatureSensorConnector,
    TemperatureSensorConfig,
    TemperatureSensorType,

    # SCADA
    SCADAConnector,
    SCADAConnectionConfig,
    SCADAProtocol,

    # Condensate
    CondensateMeterConnector,
    CondensateMeterConfig,

    # Agent Coordination
    AgentCoordinator,
    AgentRole,
    MessageType,
    MessagePriority,
    AgentMessage,

    # Data Transformation
    DataTransformationPipeline
)

logger = logging.getLogger(__name__)


class SteamSystemMonitor:
    """
    Complete steam system monitoring integration.

    Integrates all sensors, meters, and systems for comprehensive monitoring.
    """

    def __init__(self):
        """Initialize steam system monitor."""
        self.steam_meter: Optional[SteamMeterConnector] = None
        self.pressure_sensors: Optional[PressureSensorConnector] = None
        self.temperature_sensors: Optional[TemperatureSensorConnector] = None
        self.scada: Optional[SCADAConnector] = None
        self.condensate_meter: Optional[CondensateMeterConnector] = None
        self.coordinator: Optional[AgentCoordinator] = None
        self.data_pipeline = DataTransformationPipeline()

        self.monitoring = False
        self._monitoring_task = None

    async def initialize(self):
        """Initialize all integrations."""
        logger.info("Initializing steam system monitor...")

        # Initialize steam meter
        steam_config = SteamMeterConfig(
            host="192.168.1.100",
            port=502,
            protocol=MeterProtocol.MODBUS_TCP,
            meter_id="main_steam_header",
            sampling_rate_hz=1.0,
            units="t/hr",
            min_valid_flow=0.0,
            max_valid_flow=200.0,
            enable_totalizer=True
        )
        self.steam_meter = SteamMeterConnector(steam_config)

        # Initialize pressure sensors
        pressure_configs = [
            PressureSensorConfig(
                host="192.168.1.101",
                port=502,
                sensor_id="header_pressure",
                sensor_type=PressureSensorType.STRAIN_GAUGE,
                pressure_type=PressureType.GAUGE,
                min_pressure=0.0,
                max_pressure=20.0,
                sampling_rate_hz=2.0
            ),
            PressureSensorConfig(
                host="192.168.1.101",
                port=502,
                sensor_id="distribution_pressure",
                sensor_type=PressureSensorType.STRAIN_GAUGE,
                pressure_type=PressureType.GAUGE,
                min_pressure=0.0,
                max_pressure=15.0,
                sampling_rate_hz=2.0
            ),
            PressureSensorConfig(
                host="192.168.1.101",
                port=502,
                sensor_id="condensate_pressure",
                sensor_type=PressureSensorType.STRAIN_GAUGE,
                pressure_type=PressureType.GAUGE,
                min_pressure=0.0,
                max_pressure=5.0,
                sampling_rate_hz=1.0
            )
        ]
        self.pressure_sensors = PressureSensorConnector(pressure_configs)

        # Initialize temperature sensors
        temp_configs = [
            TemperatureSensorConfig(
                host="192.168.1.102",
                port=502,
                sensor_id="steam_temperature",
                sensor_type=TemperatureSensorType.RTD_PT100,
                min_temp=100.0,
                max_temp=400.0,
                smoothing_window=5
            ),
            TemperatureSensorConfig(
                host="192.168.1.102",
                port=502,
                sensor_id="condensate_temperature",
                sensor_type=TemperatureSensorType.THERMOCOUPLE_K,
                min_temp=50.0,
                max_temp=200.0,
                smoothing_window=5
            )
        ]
        self.temperature_sensors = TemperatureSensorConnector(temp_configs)

        # Initialize SCADA
        scada_config = SCADAConnectionConfig(
            protocol=SCADAProtocol.OPC_UA,
            host="192.168.1.200",
            port=4840,
            enable_subscriptions=True,
            subscription_interval_ms=1000
        )
        self.scada = SCADAConnector(scada_config)

        # Initialize condensate meter
        condensate_config = CondensateMeterConfig(
            host="192.168.1.103",
            port=502,
            meter_id="main_condensate_return",
            min_flow=0.0,
            max_flow=50.0,
            sampling_rate_hz=0.5,
            enable_quality_analysis=True
        )
        self.condensate_meter = CondensateMeterConnector(condensate_config)

        # Initialize agent coordinator
        self.coordinator = AgentCoordinator(
            agent_id="GL-003",
            role=AgentRole.HEAT_RECOVERY
        )

        logger.info("Initialization complete")

    async def connect_all(self) -> bool:
        """Connect to all systems."""
        logger.info("Connecting to all systems...")

        tasks = [
            self.steam_meter.connect(),
            self.pressure_sensors.connect(),
            self.temperature_sensors.connect(),
            self.scada.connect(),
            self.condensate_meter.connect(),
            self.coordinator.start()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = sum(1 for r in results if r is True)
        total = len(tasks)

        logger.info(f"Connected {success_count}/{total} systems")

        return success_count == total

    async def start_monitoring(self):
        """Start continuous monitoring."""
        self.monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Monitoring started")

    async def stop_monitoring(self):
        """Stop monitoring."""
        self.monitoring = False

        if self._monitoring_task:
            self._monitoring_task.cancel()

        logger.info("Monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Collect data from all systems
                data = await self.collect_system_data()

                # Analyze and process
                analysis = await self.analyze_system(data)

                # Check for alerts
                alerts = self.check_alerts(data, analysis)

                if alerts:
                    await self.handle_alerts(alerts)

                # Report to coordinator
                await self.report_status(data, analysis)

                # Log summary
                self.log_summary(data, analysis)

                await asyncio.sleep(10)  # 10-second monitoring interval

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(10)

    async def collect_system_data(self) -> Dict[str, Any]:
        """Collect data from all connected systems."""
        data = {
            'timestamp': DeterministicClock.utcnow(),
            'steam_meter': {},
            'pressure': {},
            'temperature': {},
            'condensate': {},
            'scada': {}
        }

        # Steam meter
        if self.steam_meter and self.steam_meter.current_reading:
            reading = self.steam_meter.current_reading
            data['steam_meter'] = {
                'flow_rate': reading.flow_rate,
                'totalizer': reading.totalizer,
                'velocity': reading.velocity,
                'quality_score': reading.quality_score,
                'unit': reading.unit
            }

        # Pressure sensors
        if self.pressure_sensors:
            data['pressure'] = self.pressure_sensors.get_all_pressures()

        # Temperature sensors
        if self.temperature_sensors:
            data['temperature'] = self.temperature_sensors.get_all_temperatures()

        # Condensate
        if self.condensate_meter and self.condensate_meter.current_reading:
            reading = self.condensate_meter.current_reading
            data['condensate'] = {
                'flow_rate': reading.flow_rate,
                'temperature': reading.temperature,
                'return_percentage': reading.return_percentage,
                'flash_steam_loss': reading.flash_steam_loss
            }

        # SCADA
        if self.scada:
            data['scada'] = self.scada.get_current_values()

        return data

    async def analyze_system(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze system performance."""
        analysis = {
            'efficiency': 0.0,
            'steam_balance': {},
            'pressure_drop': 0.0,
            'condensate_recovery': 0.0,
            'recommendations': []
        }

        # Calculate steam balance
        steam_flow = data['steam_meter'].get('flow_rate', 0.0)
        condensate_flow = data['condensate'].get('flow_rate', 0.0)

        if steam_flow > 0:
            recovery_rate = (condensate_flow / steam_flow) * 100
            analysis['condensate_recovery'] = recovery_rate

            if recovery_rate < 80:
                analysis['recommendations'].append(
                    f"Low condensate recovery: {recovery_rate:.1f}% (target: >80%)"
                )

        # Calculate pressure drop
        header_pressure = data['pressure'].get('header_pressure', 0.0)
        dist_pressure = data['pressure'].get('distribution_pressure', 0.0)

        if header_pressure > 0:
            pressure_drop = header_pressure - dist_pressure
            analysis['pressure_drop'] = pressure_drop

            if pressure_drop > 2.0:
                analysis['recommendations'].append(
                    f"High pressure drop: {pressure_drop:.2f} bar (check for restrictions)"
                )

        # Calculate overall efficiency
        temp = data['temperature'].get('steam_temperature', 0.0)
        pressure = data['pressure'].get('header_pressure', 0.0)

        # Simplified efficiency calculation
        if temp > 0 and pressure > 0:
            efficiency = min(100, (temp / 250) * (pressure / 12) * 100)
            analysis['efficiency'] = efficiency

        return analysis

    def check_alerts(self, data: Dict[str, Any], analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for alert conditions."""
        alerts = []

        # High pressure alert
        header_pressure = data['pressure'].get('header_pressure', 0.0)
        if header_pressure > 18.0:
            alerts.append({
                'severity': 'high',
                'type': 'pressure_high',
                'message': f"Header pressure high: {header_pressure:.2f} bar (limit: 18.0 bar)",
                'value': header_pressure,
                'timestamp': data['timestamp']
            })

        # Low steam flow alert
        steam_flow = data['steam_meter'].get('flow_rate', 0.0)
        if 0 < steam_flow < 50.0:
            alerts.append({
                'severity': 'medium',
                'type': 'flow_low',
                'message': f"Steam flow low: {steam_flow:.1f} t/hr",
                'value': steam_flow,
                'timestamp': data['timestamp']
            })

        # Poor data quality alert
        quality = data['steam_meter'].get('quality_score', 100.0)
        if quality < 70:
            alerts.append({
                'severity': 'low',
                'type': 'quality_poor',
                'message': f"Data quality poor: {quality:.0f}% (check sensors)",
                'value': quality,
                'timestamp': data['timestamp']
            })

        return alerts

    async def handle_alerts(self, alerts: List[Dict[str, Any]]):
        """Handle alert conditions."""
        for alert in alerts:
            logger.warning(f"ALERT [{alert['severity'].upper()}]: {alert['message']}")

            # Send to coordinator for parent notification
            if self.coordinator and alert['severity'] in ['high', 'critical']:
                message = AgentMessage(
                    message_id=str(DeterministicClock.utcnow().timestamp()),
                    sender_id="GL-003",
                    recipient_id="GL-001",
                    message_type=MessageType.NOTIFICATION,
                    priority=MessagePriority.HIGH if alert['severity'] == 'high' else MessagePriority.CRITICAL,
                    timestamp=DeterministicClock.utcnow(),
                    payload={
                        'alert': alert,
                        'agent': 'GL-003',
                        'system': 'steam_system'
                    },
                    requires_response=False
                )

                await self.coordinator.send_message(message)

    async def report_status(self, data: Dict[str, Any], analysis: Dict[str, Any]):
        """Report status to parent orchestrator."""
        if not self.coordinator:
            return

        # Send periodic status update
        status_message = AgentMessage(
            message_id=str(DeterministicClock.utcnow().timestamp()),
            sender_id="GL-003",
            recipient_id="GL-001",
            message_type=MessageType.STATUS,
            priority=MessagePriority.LOW,
            timestamp=DeterministicClock.utcnow(),
            payload={
                'steam_flow': data['steam_meter'].get('flow_rate', 0.0),
                'header_pressure': data['pressure'].get('header_pressure', 0.0),
                'condensate_recovery': analysis.get('condensate_recovery', 0.0),
                'efficiency': analysis.get('efficiency', 0.0),
                'health': 'good'
            },
            requires_response=False
        )

        await self.coordinator.send_message(status_message)

    def log_summary(self, data: Dict[str, Any], analysis: Dict[str, Any]):
        """Log monitoring summary."""
        logger.info("=== Steam System Status ===")
        logger.info(f"Steam Flow: {data['steam_meter'].get('flow_rate', 0.0):.1f} t/hr")
        logger.info(f"Header Pressure: {data['pressure'].get('header_pressure', 0.0):.2f} bar")
        logger.info(f"Steam Temperature: {data['temperature'].get('steam_temperature', 0.0):.1f}°C")
        logger.info(f"Condensate Flow: {data['condensate'].get('flow_rate', 0.0):.1f} t/hr")
        logger.info(f"Recovery Rate: {analysis.get('condensate_recovery', 0.0):.1f}%")
        logger.info(f"Pressure Drop: {analysis.get('pressure_drop', 0.0):.2f} bar")
        logger.info(f"Efficiency: {analysis.get('efficiency', 0.0):.1f}%")

        if analysis.get('recommendations'):
            logger.info("Recommendations:")
            for rec in analysis['recommendations']:
                logger.info(f"  - {rec}")

    async def disconnect_all(self):
        """Disconnect from all systems."""
        logger.info("Disconnecting from all systems...")

        tasks = []

        if self.steam_meter:
            tasks.append(self.steam_meter.disconnect())
        if self.pressure_sensors:
            tasks.append(self.pressure_sensors.disconnect())
        if self.temperature_sensors:
            tasks.append(self.temperature_sensors.disconnect())
        if self.scada:
            tasks.append(self.scada.disconnect())
        if self.condensate_meter:
            tasks.append(self.condensate_meter.disconnect())
        if self.coordinator:
            tasks.append(self.coordinator.stop())

        await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("Disconnected from all systems")


async def main():
    """Run complete integration example."""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create monitor
    monitor = SteamSystemMonitor()

    try:
        # Initialize
        await monitor.initialize()

        # Connect
        if await monitor.connect_all():
            logger.info("All systems connected successfully")

            # Start monitoring
            await monitor.start_monitoring()

            # Run for 60 seconds
            logger.info("Monitoring for 60 seconds...")
            await asyncio.sleep(60)

            # Stop monitoring
            await monitor.stop_monitoring()

        else:
            logger.error("Failed to connect to all systems")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)

    finally:
        # Cleanup
        await monitor.disconnect_all()
        logger.info("Example complete")


if __name__ == "__main__":
    asyncio.run(main())
