# -*- coding: utf-8 -*-
"""
Example Usage of GL-005 Integration Connectors

This file demonstrates how to use all 6 integration connectors
for industrial combustion control systems.

Author: GL-DataIntegrationEngineer
Date: 2025-11-18
Version: 1.0.0
"""

import asyncio
import logging
from datetime import datetime

# Import all connectors
from dcs_connector import DCSConnector, DCSConfig, ProcessVariable
from plc_connector import PLCConnector, PLCConfig, PLCCoil, PLCRegister, CoilType, RegisterType, DataType, PLCProtocol
from combustion_analyzer_connector import CombustionAnalyzerConnector, AnalyzerConfig, GasType, AnalyzerProtocol
from flame_scanner_connector import FlameScannerConnector, FlameScannerConfig, ScannerType
from temperature_sensor_array_connector import TemperatureSensorArrayConnector, SensorArrayConfig, TemperatureSensor, SensorType, TemperatureZone
from scada_integration import SCADAIntegration, SCADAConfig, SCADATag, SCADAAlarm, DataPriority, AlarmSeverity
from greenlang.determinism import DeterministicClock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def example_dcs_integration():
    """Example: DCS Connector Usage"""
    logger.info("=== DCS Connector Example ===")

    # Configure DCS connection
    config = DCSConfig(
        opcua_endpoint="opc.tcp://dcs.plant.com:4840",
        modbus_host="10.0.1.100",
        modbus_port=502
    )

    async with DCSConnector(config) as dcs:
        # Register process variables
        dcs.register_process_variable(ProcessVariable(
            tag_name="FurnaceTemp",
            node_id="ns=2;s=FurnaceTemp",
            description="Furnace Temperature",
            data_type="float",
            engineering_units="°C",
            alarm_high_high=950.0,
            alarm_high=900.0,
            alarm_low=700.0,
            writable=False
        ))

        dcs.register_process_variable(ProcessVariable(
            tag_name="FuelFlowSetpoint",
            node_id="ns=2;s=FuelFlowSetpoint",
            description="Fuel Flow Setpoint",
            data_type="float",
            engineering_units="kg/h",
            min_value=0.0,
            max_value=500.0,
            writable=True
        ))

        # Read process variables
        values = await dcs.read_process_variables([
            "FurnaceTemp",
            "FuelFlowSetpoint"
        ])

        for tag_name, data in values.items():
            logger.info(f"{tag_name}: {data['value']} {data['units']} (quality: {data['quality']})")

        # Write setpoint (if needed)
        write_result = await dcs.write_setpoints({
            "FuelFlowSetpoint": 150.5
        })

        for tag_name, success in write_result.items():
            logger.info(f"Write {tag_name}: {'SUCCESS' if success else 'FAILED'}")

        # Subscribe to alarms
        async def alarm_handler(alarm):
            logger.warning(f"ALARM: {alarm.message} (priority={alarm.priority})")

        await dcs.subscribe_to_alarms(alarm_handler)

        # Get historical data
        historical = await dcs.get_historical_data(
            "FurnaceTemp",
            start_time=DeterministicClock.now() - timedelta(hours=1),
            end_time=DeterministicClock.now()
        )

        logger.info(f"Retrieved {len(historical)} historical points")


async def example_plc_integration():
    """Example: PLC Connector Usage"""
    logger.info("=== PLC Connector Example ===")

    # Configure PLC connection
    config = PLCConfig(
        protocol=PLCProtocol.MODBUS_TCP,
        tcp_host="10.0.1.50",
        tcp_port=502,
        heartbeat_coil_address=100
    )

    async with PLCConnector(config) as plc:
        # Register digital coils
        plc.register_coil(PLCCoil(
            name="BurnerOn",
            coil_type=CoilType.COIL,
            address=0,
            description="Burner Enable Output"
        ))

        plc.register_coil(PLCCoil(
            name="FlameSensorActive",
            coil_type=CoilType.DISCRETE_INPUT,
            address=10,
            description="Flame Sensor Input"
        ))

        # Register analog registers
        plc.register_register(PLCRegister(
            name="FuelFlow",
            register_type=RegisterType.HOLDING,
            address=100,
            data_type=DataType.FLOAT32,
            description="Fuel Flow Rate",
            engineering_units="kg/h",
            min_value=0.0,
            max_value=500.0
        ))

        plc.register_register(PLCRegister(
            name="AirFlowSetpoint",
            register_type=RegisterType.HOLDING,
            address=200,
            data_type=DataType.FLOAT32,
            description="Air Flow Setpoint",
            engineering_units="Nm3/h"
        ))

        # Read digital inputs
        coil_values = await plc.read_coils(["BurnerOn", "FlameSensorActive"])
        for name, value in coil_values.items():
            logger.info(f"Digital {name}: {value}")

        # Read analog values
        register_values = await plc.read_registers(["FuelFlow", "AirFlowSetpoint"])
        for name, value in register_values.items():
            logger.info(f"Analog {name}: {value}")

        # Write control outputs
        write_result = await plc.write_coils({
            "BurnerOn": True
        })

        # Write setpoints
        setpoint_result = await plc.write_registers({
            "AirFlowSetpoint": 1200.0
        })

        # Monitor heartbeat
        heartbeat_ok = await plc.monitor_heartbeat()
        logger.info(f"PLC Heartbeat: {'OK' if heartbeat_ok else 'FAILED'}")

        # Get performance stats
        stats = plc.get_performance_stats()
        logger.info(f"PLC Performance: {stats}")


async def example_analyzer_integration():
    """Example: Combustion Analyzer Usage"""
    logger.info("=== Combustion Analyzer Example ===")

    # Configure analyzer
    config = AnalyzerConfig(
        analyzer_id="O2_ANALYZER_01",
        manufacturer="ABB",
        model="AO2020",
        primary_protocol=AnalyzerProtocol.MQTT,
        mqtt_broker="mqtt.plant.com",
        mqtt_port=1883,
        gases_measured=[GasType.O2, GasType.CO, GasType.NOx],
        measurement_units={
            GasType.O2: "%",
            GasType.CO: "ppm",
            GasType.NOx: "ppm"
        },
        auto_calibration_enabled=True
    )

    async with CombustionAnalyzerConnector(config) as analyzer:
        # Read O2 level
        o2 = await analyzer.read_o2_level()
        logger.info(f"O2 Concentration: {o2}%")

        # Read CO level
        co = await analyzer.read_co_level()
        logger.info(f"CO Concentration: {co} ppm")

        # Read NOx level
        nox = await analyzer.read_nox_level()
        logger.info(f"NOx Concentration: {nox} ppm")

        # Read all gases
        all_gases = await analyzer.read_all_gases()
        for gas_type, concentration in all_gases.items():
            logger.info(f"{gas_type.value}: {concentration}")

        # Subscribe to real-time measurements
        async def measurement_callback(measurement):
            logger.info(
                f"Measurement - {measurement.gas_type.value}: "
                f"{measurement.concentration} {measurement.units} "
                f"(quality: {measurement.quality.value})"
            )

        await analyzer.subscribe_to_measurements(measurement_callback)

        # Validate readings
        validation = await analyzer.validate_readings()
        logger.info(f"Data Quality Score: {validation['quality_score']}/100")

        # Run calibration (if needed)
        if validation['quality_score'] < 80:
            logger.info("Running analyzer calibration...")
            success = await analyzer.calibrate_analyzer()
            logger.info(f"Calibration: {'SUCCESS' if success else 'FAILED'}")


async def example_flame_scanner_integration():
    """Example: Flame Scanner Usage"""
    logger.info("=== Flame Scanner Example ===")

    # Configure flame scanner
    config = FlameScannerConfig(
        scanner_id="SCANNER_BURNER_01",
        scanner_type=ScannerType.UV_DETECTOR,
        burner_id="BURNER_01",
        modbus_host="10.0.1.60",
        modbus_port=502,
        scan_rate_hz=100,  # Fast scanning
        flame_failure_delay_ms=200
    )

    async with FlameScannerConnector(config) as scanner:
        # Detect flame presence
        flame_present = await scanner.detect_flame_presence()
        logger.info(f"Flame Status: {'PRESENT' if flame_present else 'ABSENT'}")

        # Measure flame intensity
        intensity = await scanner.measure_flame_intensity()
        logger.info(f"Flame Intensity: {intensity}%")

        # Analyze flame stability
        stability = await scanner.analyze_flame_stability()
        logger.info(
            f"Flame Stability:\n"
            f"  - Stability Index: {stability.stability_index}/100\n"
            f"  - Flicker Frequency: {stability.flicker_frequency_hz:.2f} Hz\n"
            f"  - Mean Intensity: {stability.intensity_mean:.1f}%\n"
            f"  - Std Dev: {stability.intensity_std_dev:.2f}%"
        )

        # Subscribe to flame events
        async def flame_event_handler(event):
            logger.info(
                f"Flame Event - Present: {event.flame_present}, "
                f"Intensity: {event.intensity}%, "
                f"Response: {event.response_time_ms:.1f}ms"
            )

        await scanner.subscribe_to_flame_events(flame_event_handler)

        # Subscribe to flame failures
        async def flame_failure_handler(burner_id):
            logger.critical(f"FLAME FAILURE DETECTED on {burner_id}!")

        await scanner.subscribe_to_flame_failures(flame_failure_handler)

        # Get performance stats
        stats = scanner.get_performance_stats()
        logger.info(f"Scanner Performance: {stats}")


async def example_temperature_array_integration():
    """Example: Temperature Sensor Array Usage"""
    logger.info("=== Temperature Sensor Array Example ===")

    # Configure sensor array
    config = SensorArrayConfig(
        array_id="TEMP_ARRAY_MAIN",
        serial_port="/dev/ttyUSB0",  # COM1 on Windows
        baudrate=9600,
        scan_rate_hz=1.0
    )

    async with TemperatureSensorArrayConnector(config) as array:
        # Register furnace temperature sensors
        array.register_sensor(TemperatureSensor(
            sensor_id="FURNACE_TEMP_01",
            sensor_type=SensorType.THERMOCOUPLE_K,
            zone=TemperatureZone.FURNACE,
            description="Furnace Zone 1",
            register_address=0,
            unit_id=1,
            max_temp_c=1200.0,
            alarm_high=900.0
        ))

        array.register_sensor(TemperatureSensor(
            sensor_id="FURNACE_TEMP_02",
            sensor_type=SensorType.THERMOCOUPLE_K,
            zone=TemperatureZone.FURNACE,
            description="Furnace Zone 2",
            register_address=2,
            unit_id=1,
            max_temp_c=1200.0
        ))

        # Register flue gas temperature sensor
        array.register_sensor(TemperatureSensor(
            sensor_id="FLUE_GAS_TEMP",
            sensor_type=SensorType.THERMOCOUPLE_K,
            zone=TemperatureZone.FLUE_GAS,
            description="Flue Gas Outlet",
            register_address=10,
            unit_id=1,
            max_temp_c=600.0
        ))

        # Read furnace temperature (average of all furnace sensors)
        furnace_temp = await array.read_furnace_temperature()
        logger.info(f"Furnace Temperature: {furnace_temp}°C")

        # Read flue gas temperature
        flue_temp = await array.read_flue_gas_temperature()
        logger.info(f"Flue Gas Temperature: {flue_temp}°C")

        # Read all zones
        all_temps = await array.read_all_zones()
        for zone, temp in all_temps.items():
            logger.info(f"{zone.value}: {temp}°C")

        # Validate sensor health
        health = await array.validate_sensor_health()
        logger.info(
            f"Sensor Health:\n"
            f"  - Overall: {health['overall_health']}\n"
            f"  - Healthy: {health['healthy_sensors']}\n"
            f"  - Degraded: {health['degraded_sensors']}\n"
            f"  - Failed: {health['failed_sensors']}"
        )

        # Get zone statistics
        stats = array.get_zone_statistics(TemperatureZone.FURNACE)
        if stats:
            logger.info(
                f"Furnace Zone Statistics:\n"
                f"  - Min: {stats['min_temp_c']:.1f}°C\n"
                f"  - Max: {stats['max_temp_c']:.1f}°C\n"
                f"  - Mean: {stats['mean_temp_c']:.1f}°C\n"
                f"  - Std Dev: {stats['std_dev_c']:.2f}°C"
            )


async def example_scada_integration():
    """Example: SCADA Integration Usage"""
    logger.info("=== SCADA Integration Example ===")

    # Configure SCADA integration
    config = SCADAConfig(
        system_id="GL005_COMBUSTION_CONTROL",
        opcua_enabled=True,
        opcua_endpoint="opc.tcp://0.0.0.0:4840",
        mqtt_enabled=True,
        mqtt_broker="mqtt.plant.com",
        mqtt_port=1883
    )

    async with SCADAIntegration(config) as scada:
        # Register SCADA tags
        scada.register_tag(SCADATag(
            tag_name="FurnaceTemp",
            description="Furnace Temperature",
            data_type="float",
            units="°C",
            priority=DataPriority.HIGH,
            deadband=0.5  # Only publish if changes >0.5%
        ))

        scada.register_tag(SCADATag(
            tag_name="SteamPressure",
            description="Steam Pressure",
            data_type="float",
            units="bar",
            priority=DataPriority.HIGH
        ))

        scada.register_tag(SCADATag(
            tag_name="O2Content",
            description="Oxygen Content",
            data_type="float",
            units="%",
            priority=DataPriority.CRITICAL
        ))

        scada.register_tag(SCADATag(
            tag_name="BurnerStatus",
            description="Burner On/Off",
            data_type="bool",
            units="",
            priority=DataPriority.CRITICAL
        ))

        # Publish real-time data
        await scada.publish_real_time_data({
            "FurnaceTemp": 850.5,
            "SteamPressure": 120.0,
            "O2Content": 3.5,
            "BurnerStatus": True
        })

        # Publish alarms
        await scada.publish_alarms([
            SCADAAlarm(
                alarm_id="TEMP_HIGH_001",
                source_tag="FurnaceTemp",
                severity=AlarmSeverity.HIGH,
                message="Furnace temperature high - approaching limit",
                timestamp=DeterministicClock.now(),
                value=900.0,
                limit=850.0
            )
        ])

        # Receive operator commands
        async def command_handler(command):
            logger.info(
                f"Operator Command Received:\n"
                f"  - Type: {command.command_type.value}\n"
                f"  - Target: {command.target_tag}\n"
                f"  - Value: {command.value}\n"
                f"  - Operator: {command.operator}"
            )

            # Execute command (example)
            if command.command_type == CommandType.SETPOINT_CHANGE:
                logger.info(f"Applying setpoint change: {command.target_tag} = {command.value}")
                return "SUCCESS"

            return "ACKNOWLEDGED"

        await scada.receive_operator_commands(command_handler)

        # Publish historical trends
        trends = await scada.publish_trends(
            "FurnaceTemp",
            start_time=DeterministicClock.now() - timedelta(hours=1),
            end_time=DeterministicClock.now()
        )

        logger.info(f"Published {len(trends)} trend data points")

        # Get performance stats
        stats = scada.get_performance_stats()
        logger.info(f"SCADA Performance: {stats}")


async def main():
    """Run all integration examples"""
    logger.info("Starting GL-005 Integration Connectors Examples")

    try:
        # Run examples sequentially
        await example_dcs_integration()
        await example_plc_integration()
        await example_analyzer_integration()
        await example_flame_scanner_integration()
        await example_temperature_array_integration()
        await example_scada_integration()

        logger.info("All integration examples completed successfully!")

    except Exception as e:
        logger.error(f"Error running integration examples: {e}", exc_info=True)


if __name__ == "__main__":
    # Run async main
    asyncio.run(main())
