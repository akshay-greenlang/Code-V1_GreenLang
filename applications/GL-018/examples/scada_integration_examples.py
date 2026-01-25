"""
SCADA Integration Examples for GL-018 FLUEFLOW

Demonstrates connecting to flue gas analyzers via OPC-UA and Modbus,
reading process values, writing control setpoints, and managing alarms.

Examples include:
1. ABB AO2000 series (OPC-UA)
2. SICK MARSIC series (OPC-UA)
3. Horiba PG series (Modbus TCP)
4. Real-time monitoring with callbacks
5. Combustion optimization control
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from integrations.scada_integration import (
    create_scada_client,
    SCADAConfig,
    FlueGasTag,
    TagDataPoint,
    ConnectionProtocol,
    ParameterType,
    MeasurementLocation,
    TagType,
    AnalyzerType,
    create_standard_flue_gas_tags,
    create_abb_ao2000_tags,
    create_sick_marsic_tags,
    create_horiba_pg_tags,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Example 1: ABB AO2000 OPC-UA Connection
# =============================================================================


async def example_abb_ao2000_opcua():
    """
    Example: Connect to ABB AO2000 analyzer via OPC-UA.

    ABB AO2000 is a multi-channel gas analyzer supporting O2, CO, CO2, NOx.
    """
    logger.info("=" * 60)
    logger.info("Example 1: ABB AO2000 OPC-UA Connection")
    logger.info("=" * 60)

    # Create client configuration
    config = SCADAConfig(
        protocol=ConnectionProtocol.OPC_UA,
        analyzer_type=AnalyzerType.ABB_AO2000,
        host="192.168.1.100",
        port=4840,
        endpoint_url="opc.tcp://192.168.1.100:4840/ABB/AO2000",
        namespace_index=2,
        username="operator",
        password="secure_password",
        connection_timeout=10.0,
        enable_historical_access=True,
    )

    # Create client
    client = create_scada_client(
        analyzer_type=AnalyzerType.ABB_AO2000,
        protocol=ConnectionProtocol.OPC_UA,
        host="192.168.1.100",
        port=4840,
        username="operator",
        password="secure_password",
    )

    # Register ABB-specific tags
    abb_tags = create_abb_ao2000_tags()
    client.register_tags(abb_tags)

    logger.info(f"Registered {len(abb_tags)} ABB AO2000 tags")

    try:
        # Connect to analyzer
        logger.info("Connecting to ABB AO2000 analyzer...")
        connected = await client.connect()

        if connected:
            logger.info("Connected successfully!")

            # Read O2 concentration
            o2_data = await client.read_tag("AO2000.Channel1.O2")
            logger.info(f"O2: {o2_data.value:.2f} {o2_data.engineering_unit}")

            # Read CO concentration
            co_data = await client.read_tag("AO2000.Channel2.CO")
            logger.info(f"CO: {co_data.value:.1f} {co_data.engineering_unit}")

            # Check health
            health = await client.health_check()
            logger.info(f"Health status: {health}")

            # Get statistics
            stats = client.get_statistics()
            logger.info(f"Statistics: {stats}")

        else:
            logger.error("Failed to connect to analyzer")

    except Exception as e:
        logger.error(f"Error: {e}")

    finally:
        await client.disconnect()
        logger.info("Disconnected from analyzer")


# =============================================================================
# Example 2: SICK MARSIC OPC-UA Connection
# =============================================================================


async def example_sick_marsic_opcua():
    """
    Example: Connect to SICK MARSIC analyzer via OPC-UA.

    SICK MARSIC is a multi-component gas analyzer for emissions monitoring.
    """
    logger.info("=" * 60)
    logger.info("Example 2: SICK MARSIC OPC-UA Connection")
    logger.info("=" * 60)

    client = create_scada_client(
        analyzer_type=AnalyzerType.SICK_MARSIC,
        protocol=ConnectionProtocol.OPC_UA,
        host="192.168.1.101",
        port=4840,
    )

    # Register SICK-specific tags
    sick_tags = create_sick_marsic_tags()
    client.register_tags(sick_tags)

    try:
        await client.connect()

        # Read multiple tags
        tag_names = ["MARSIC.O2.Value", "MARSIC.NOx.Value"]
        results = await client.read_tags(tag_names)

        for tag_name, data_point in results.items():
            logger.info(
                f"{tag_name}: {data_point.value:.2f} {data_point.engineering_unit}"
            )

    except Exception as e:
        logger.error(f"Error: {e}")

    finally:
        await client.disconnect()


# =============================================================================
# Example 3: Horiba PG Modbus TCP Connection
# =============================================================================


async def example_horiba_pg_modbus():
    """
    Example: Connect to Horiba PG analyzer via Modbus TCP.

    Horiba PG series uses Modbus TCP with holding registers for gas readings.
    """
    logger.info("=" * 60)
    logger.info("Example 3: Horiba PG Modbus TCP Connection")
    logger.info("=" * 60)

    client = create_scada_client(
        analyzer_type=AnalyzerType.HORIBA_PG,
        protocol=ConnectionProtocol.MODBUS_TCP,
        host="192.168.1.102",
        port=502,
        modbus_unit_id=1,
        modbus_timeout=3.0,
    )

    # Register Horiba-specific tags (Modbus registers)
    horiba_tags = create_horiba_pg_tags()
    client.register_tags(horiba_tags)

    logger.info(f"Registered {len(horiba_tags)} Horiba PG tags")

    try:
        await client.connect()

        # Read Modbus registers
        # 30001 = O2, 30002 = CO2, 30003 = CO, 30004 = NOx
        o2_data = await client.read_tag("30001")
        co2_data = await client.read_tag("30002")
        co_data = await client.read_tag("30003")
        nox_data = await client.read_tag("30004")

        logger.info(f"O2: {o2_data.value:.2f} %")
        logger.info(f"CO2: {co2_data.value:.2f} %")
        logger.info(f"CO: {co_data.value:.1f} ppm")
        logger.info(f"NOx: {nox_data.value:.1f} ppm")

    except Exception as e:
        logger.error(f"Error: {e}")

    finally:
        await client.disconnect()


# =============================================================================
# Example 4: Real-time Monitoring with Callbacks
# =============================================================================


async def example_realtime_monitoring():
    """
    Example: Real-time monitoring with tag subscriptions and callbacks.

    Monitor O2 and CO continuously and trigger actions on value changes.
    """
    logger.info("=" * 60)
    logger.info("Example 4: Real-time Monitoring with Callbacks")
    logger.info("=" * 60)

    client = create_scada_client(
        analyzer_type=AnalyzerType.GENERIC_OPC_UA,
        protocol=ConnectionProtocol.OPC_UA,
        host="192.168.1.100",
        port=4840,
    )

    # Register standard tags
    tags = create_standard_flue_gas_tags()
    client.register_tags(tags)

    # Define callback for O2 changes
    def on_o2_change(data_point: TagDataPoint):
        logger.info(
            f"O2 changed: {data_point.value:.2f} % at {data_point.timestamp}"
        )

        # Check if O2 is optimal (3-4% for natural gas)
        if data_point.value < 2.0:
            logger.warning("O2 too low! Risk of incomplete combustion")
        elif data_point.value > 6.0:
            logger.warning("O2 too high! Excess air reduces efficiency")

    # Define callback for CO changes
    def on_co_change(data_point: TagDataPoint):
        logger.info(
            f"CO changed: {data_point.value:.1f} ppm at {data_point.timestamp}"
        )

        # CO alarm threshold
        if data_point.value > 400.0:
            logger.error("HIGH CO ALARM! Incomplete combustion detected!")

    try:
        await client.connect()

        # Subscribe to tags
        await client.subscribe_tag("FG_O2_STACK", on_o2_change)
        await client.subscribe_tag("FG_CO_STACK", on_co_change)

        logger.info("Monitoring O2 and CO in real-time...")
        logger.info("Press Ctrl+C to stop")

        # Monitor for 60 seconds
        await asyncio.sleep(60)

    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")

    except Exception as e:
        logger.error(f"Error: {e}")

    finally:
        await client.disconnect()


# =============================================================================
# Example 5: Combustion Optimization Control
# =============================================================================


async def example_combustion_optimization():
    """
    Example: Combustion optimization by controlling air damper and fuel valve.

    Adjust air/fuel ratio to maintain optimal O2 (3-4%) while minimizing CO.
    """
    logger.info("=" * 60)
    logger.info("Example 5: Combustion Optimization Control")
    logger.info("=" * 60)

    client = create_scada_client(
        analyzer_type=AnalyzerType.GENERIC_OPC_UA,
        protocol=ConnectionProtocol.OPC_UA,
        host="192.168.1.100",
        port=4840,
    )

    # Register tags
    tags = create_standard_flue_gas_tags()
    client.register_tags(tags)

    # Target O2 setpoint (optimal for natural gas)
    target_o2 = 3.5  # %
    tolerance = 0.5  # %

    try:
        await client.connect()

        # Read current values
        o2_data = await client.read_tag("FG_O2_STACK")
        co_data = await client.read_tag("FG_CO_STACK")
        damper_data = await client.read_tag("AIR_DAMPER_POS")

        logger.info(f"Current O2: {o2_data.value:.2f} %")
        logger.info(f"Current CO: {co_data.value:.1f} ppm")
        logger.info(f"Current air damper: {damper_data.value:.1f} %")

        # Calculate control action
        o2_error = o2_data.value - target_o2

        if abs(o2_error) > tolerance:
            # Adjust air damper
            if o2_error > 0:
                # Too much O2 - reduce air
                new_damper_pos = damper_data.value - 2.0
                action = "Reducing air"
            else:
                # Too little O2 - increase air
                new_damper_pos = damper_data.value + 2.0
                action = "Increasing air"

            # Clamp to valid range
            new_damper_pos = max(20.0, min(90.0, new_damper_pos))

            logger.info(
                f"{action}: Adjusting damper from {damper_data.value:.1f}% "
                f"to {new_damper_pos:.1f}%"
            )

            # Write new setpoint
            success = await client.write_tag("AIR_DAMPER_POS", new_damper_pos)

            if success:
                logger.info("Damper position updated successfully")

                # Update O2 setpoint
                await client.write_tag("O2_SETPOINT", target_o2)
                logger.info(f"O2 setpoint updated to {target_o2}%")
            else:
                logger.error("Failed to update damper position")

        else:
            logger.info("O2 within optimal range - no adjustment needed")

        # Check CO levels
        if co_data.value > 100.0:
            logger.warning(
                f"Elevated CO detected ({co_data.value:.1f} ppm) - "
                "may need to increase air slightly"
            )

    except Exception as e:
        logger.error(f"Error: {e}")

    finally:
        await client.disconnect()


# =============================================================================
# Example 6: Historical Data Analysis
# =============================================================================


async def example_historical_data():
    """
    Example: Retrieve and analyze historical flue gas data.

    Useful for trend analysis, efficiency calculations, and reporting.
    """
    logger.info("=" * 60)
    logger.info("Example 6: Historical Data Analysis")
    logger.info("=" * 60)

    client = create_scada_client(
        analyzer_type=AnalyzerType.GENERIC_OPC_UA,
        protocol=ConnectionProtocol.OPC_UA,
        host="192.168.1.100",
        port=4840,
        enable_historical_access=True,
    )

    tags = create_standard_flue_gas_tags()
    client.register_tags(tags)

    try:
        await client.connect()

        # Simulate data collection
        logger.info("Collecting data for 10 seconds...")
        for _ in range(10):
            await client.read_tag("FG_O2_STACK", use_cache=False)
            await asyncio.sleep(1)

        # Retrieve historical data
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(seconds=15)

        historical = await client.get_historical_data(
            "FG_O2_STACK",
            start_time,
            end_time
        )

        logger.info(f"Retrieved {len(historical)} historical data points")

        # Calculate statistics
        if historical:
            values = [dp.value for dp in historical]
            avg_o2 = sum(values) / len(values)
            min_o2 = min(values)
            max_o2 = max(values)

            logger.info(f"Average O2: {avg_o2:.2f} %")
            logger.info(f"Min O2: {min_o2:.2f} %")
            logger.info(f"Max O2: {max_o2:.2f} %")
            logger.info(f"Range: {max_o2 - min_o2:.2f} %")

    except Exception as e:
        logger.error(f"Error: {e}")

    finally:
        await client.disconnect()


# =============================================================================
# Example 7: Alarm Management
# =============================================================================


async def example_alarm_management():
    """
    Example: Monitor and manage alarms.

    Track high/low alarms and acknowledge them.
    """
    logger.info("=" * 60)
    logger.info("Example 7: Alarm Management")
    logger.info("=" * 60)

    client = create_scada_client(
        analyzer_type=AnalyzerType.GENERIC_OPC_UA,
        protocol=ConnectionProtocol.OPC_UA,
        host="192.168.1.100",
        port=4840,
    )

    tags = create_standard_flue_gas_tags()
    client.register_tags(tags)

    try:
        await client.connect()

        # Read tags to potentially trigger alarms
        await client.read_tag("FG_O2_STACK")
        await client.read_tag("FG_CO_STACK")
        await client.read_tag("FG_NOX_STACK")

        # Check for active alarms
        active_alarms = client.get_active_alarms()

        if active_alarms:
            logger.info(f"Found {len(active_alarms)} active alarms:")

            for alarm in active_alarms:
                logger.warning(
                    f"  [{alarm.severity.value.upper()}] {alarm.message}"
                )
                logger.info(f"    Tag: {alarm.tag_name}")
                logger.info(f"    Current: {alarm.current_value}")
                logger.info(f"    Setpoint: {alarm.setpoint}")

                # Acknowledge alarm
                acknowledged = await client.acknowledge_alarm(
                    alarm.alarm_id,
                    acknowledged_by="operator@plant.com",
                    notes="Investigating alarm cause"
                )

                if acknowledged:
                    logger.info(f"    Alarm acknowledged")

        else:
            logger.info("No active alarms")

        # Get alarm history
        alarm_history = client.get_alarm_history(limit=10)
        logger.info(f"\nAlarm history ({len(alarm_history)} entries):")

        for alarm in alarm_history[-5:]:  # Show last 5
            logger.info(
                f"  {alarm.activated_at.strftime('%H:%M:%S')} - "
                f"{alarm.tag_name}: {alarm.message}"
            )

    except Exception as e:
        logger.error(f"Error: {e}")

    finally:
        await client.disconnect()


# =============================================================================
# Example 8: Multi-Analyzer Setup
# =============================================================================


async def example_multi_analyzer():
    """
    Example: Connect to multiple analyzers simultaneously.

    Common in large plants with multiple boilers or process units.
    """
    logger.info("=" * 60)
    logger.info("Example 8: Multi-Analyzer Setup")
    logger.info("=" * 60)

    # Create clients for multiple analyzers
    clients = {
        "boiler_1": create_scada_client(
            analyzer_type=AnalyzerType.ABB_AO2000,
            protocol=ConnectionProtocol.OPC_UA,
            host="192.168.1.100",
            port=4840,
        ),
        "boiler_2": create_scada_client(
            analyzer_type=AnalyzerType.SICK_MARSIC,
            protocol=ConnectionProtocol.OPC_UA,
            host="192.168.1.101",
            port=4840,
        ),
        "auxiliary": create_scada_client(
            analyzer_type=AnalyzerType.HORIBA_PG,
            protocol=ConnectionProtocol.MODBUS_TCP,
            host="192.168.1.102",
            port=502,
        ),
    }

    # Register tags for each
    for name, client in clients.items():
        tags = create_standard_flue_gas_tags()
        client.register_tags(tags)

    try:
        # Connect all clients
        logger.info("Connecting to all analyzers...")
        connect_tasks = [client.connect() for client in clients.values()]
        results = await asyncio.gather(*connect_tasks, return_exceptions=True)

        for name, result in zip(clients.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"{name}: Connection failed - {result}")
            elif result:
                logger.info(f"{name}: Connected")
            else:
                logger.warning(f"{name}: Connection failed")

        # Read O2 from all analyzers
        logger.info("\nReading O2 from all analyzers:")
        for name, client in clients.items():
            if client.is_connected():
                try:
                    o2_data = await client.read_tag("FG_O2_STACK")
                    logger.info(f"{name}: O2 = {o2_data.value:.2f} %")
                except Exception as e:
                    logger.error(f"{name}: Read failed - {e}")

    except Exception as e:
        logger.error(f"Error: {e}")

    finally:
        # Disconnect all
        logger.info("\nDisconnecting from all analyzers...")
        for client in clients.values():
            await client.disconnect()


# =============================================================================
# Main Runner
# =============================================================================


async def main():
    """Run all examples."""
    logger.info("SCADA Integration Examples for GL-018 FLUEFLOW")
    logger.info("=" * 60)

    examples = [
        ("ABB AO2000 OPC-UA", example_abb_ao2000_opcua),
        ("SICK MARSIC OPC-UA", example_sick_marsic_opcua),
        ("Horiba PG Modbus TCP", example_horiba_pg_modbus),
        ("Real-time Monitoring", example_realtime_monitoring),
        ("Combustion Optimization", example_combustion_optimization),
        ("Historical Data Analysis", example_historical_data),
        ("Alarm Management", example_alarm_management),
        ("Multi-Analyzer Setup", example_multi_analyzer),
    ]

    logger.info("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        logger.info(f"  {i}. {name}")

    logger.info("\nNOTE: These examples require actual SCADA connections.")
    logger.info("Update IP addresses and credentials before running.")
    logger.info("\nRunning Example 5 (Combustion Optimization)...")
    logger.info("=" * 60)

    # Run example 5 as default
    await example_combustion_optimization()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nExamples interrupted by user")
    except Exception as e:
        logger.error(f"Error running examples: {e}")
