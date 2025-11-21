# -*- coding: utf-8 -*-
"""
Steam Meter Integration Tests for GL-003 SteamSystemAnalyzer

Tests comprehensive steam meter integration including:
- Modbus steam meter connectivity
- HART protocol support
- Flow rate measurement (volumetric & mass)
- Totalizer readings and reset
- Pressure and temperature compensation
- Energy flow calculation
- Steam quality measurement
- Meter diagnostics and health
- Calibration validation
- Historical data retrieval

Test Scenarios: 35+
Coverage: Modbus meters, HART meters, flow measurement, diagnostics

Author: GreenLang Test Engineering Team
"""

import pytest
import asyncio
import random
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List
from decimal import Decimal

import sys
import os
from greenlang.determinism import DeterministicClock
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from integrations.steam_meter_connector import (
    SteamMeterConnector,
    SteamMeterConfig,
    SteamMeterProtocol,
    FlowMeasurement,
    MeterQuality,
    CalibrationStatus
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def steam_meter_config():
    """Create test steam meter configuration."""
    return SteamMeterConfig(
        meter_id="SM-001",
        protocol=SteamMeterProtocol.MODBUS_TCP,
        host="localhost",
        port=5020,
        unit_id=1,
        connection_timeout=10,
        read_timeout=5,
        measurement_units={
            'flow': 'm3/hr',
            'mass_flow': 'kg/hr',
            'pressure': 'bar',
            'temperature': 'degC',
            'energy': 'kW'
        },
        enable_compensation=True,
        compensation_method='saturated_steam'
    )


@pytest.fixture
async def steam_meter_connector(steam_meter_config):
    """Create steam meter connector instance."""
    connector = SteamMeterConnector(steam_meter_config)
    yield connector
    if connector.is_connected:
        await connector.disconnect()


# ============================================================================
# TEST CLASS: CONNECTION MANAGEMENT
# ============================================================================

@pytest.mark.integration
@pytest.mark.steam_meter
class TestSteamMeterConnection:
    """Test steam meter connection establishment and management."""

    @pytest.mark.asyncio
    async def test_modbus_meter_connection(self, steam_meter_connector):
        """Test Modbus steam meter connection."""
        result = await steam_meter_connector.connect()

        assert result is True
        assert steam_meter_connector.is_connected is True
        assert steam_meter_connector.meter_id == "SM-001"

    @pytest.mark.asyncio
    async def test_hart_meter_connection(self, steam_meter_config):
        """Test HART protocol steam meter connection."""
        steam_meter_config.protocol = SteamMeterProtocol.HART
        steam_meter_config.port = 5021
        connector = SteamMeterConnector(steam_meter_config)

        result = await connector.connect()

        assert result is True
        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_connection_health_check(self, steam_meter_connector):
        """Test meter connection health check."""
        await steam_meter_connector.connect()

        health = await steam_meter_connector.health_check()

        assert health is True

    @pytest.mark.asyncio
    async def test_connection_retry_logic(self, steam_meter_config):
        """Test automatic connection retry."""
        steam_meter_config.max_reconnect_attempts = 3
        connector = SteamMeterConnector(steam_meter_config)

        # Force initial failure
        attempt_count = [0]

        original_connect = connector._connect_impl

        async def mock_connect(*args, **kwargs):
            attempt_count[0] += 1
            if attempt_count[0] < 2:
                raise Exception("Connection failed")
            return await original_connect(*args, **kwargs)

        connector._connect_impl = mock_connect

        result = await connector.connect()

        assert result is True
        assert attempt_count[0] >= 2

        await connector.disconnect()


# ============================================================================
# TEST CLASS: FLOW MEASUREMENT
# ============================================================================

@pytest.mark.integration
@pytest.mark.steam_meter
class TestFlowMeasurement:
    """Test steam flow measurement operations."""

    @pytest.mark.asyncio
    async def test_read_volumetric_flow(self, steam_meter_connector):
        """Test reading volumetric flow rate."""
        await steam_meter_connector.connect()

        flow_data = await steam_meter_connector.read_volumetric_flow()

        assert flow_data is not None
        assert 'value' in flow_data
        assert 'unit' in flow_data
        assert 'timestamp' in flow_data
        assert flow_data['unit'] == 'm3/hr'
        assert flow_data['value'] > 0

    @pytest.mark.asyncio
    async def test_read_mass_flow(self, steam_meter_connector):
        """Test reading mass flow rate."""
        await steam_meter_connector.connect()

        mass_flow_data = await steam_meter_connector.read_mass_flow()

        assert mass_flow_data is not None
        assert 'value' in mass_flow_data
        assert 'unit' in mass_flow_data
        assert mass_flow_data['unit'] == 'kg/hr'
        assert mass_flow_data['value'] > 0

    @pytest.mark.asyncio
    async def test_read_complete_measurement(self, steam_meter_connector):
        """Test reading complete flow measurement."""
        await steam_meter_connector.connect()

        measurement = await steam_meter_connector.read_measurement()

        assert isinstance(measurement, FlowMeasurement)
        assert measurement.volumetric_flow > 0
        assert measurement.mass_flow > 0
        assert measurement.pressure > 0
        assert measurement.temperature > 0
        assert measurement.quality in [MeterQuality.GOOD, 'GOOD']

    @pytest.mark.asyncio
    async def test_flow_measurement_accuracy(self, steam_meter_connector):
        """Test flow measurement accuracy and consistency."""
        await steam_meter_connector.connect()

        # Take multiple readings
        readings = []
        for _ in range(10):
            measurement = await steam_meter_connector.read_measurement()
            readings.append(measurement.volumetric_flow)
            await asyncio.sleep(0.5)

        # Calculate standard deviation
        avg = sum(readings) / len(readings)
        variance = sum((x - avg) ** 2 for x in readings) / len(readings)
        std_dev = variance ** 0.5

        # Standard deviation should be < 5% of average
        assert std_dev / avg < 0.05

    @pytest.mark.asyncio
    async def test_flow_rate_range_validation(self, steam_meter_connector):
        """Test flow rate is within expected range."""
        await steam_meter_connector.connect()

        measurement = await steam_meter_connector.read_measurement()

        # Flow rate should be within reasonable range (0-10000 m3/hr)
        assert 0 < measurement.volumetric_flow < 10000
        assert 0 < measurement.mass_flow < 10000

    @pytest.mark.asyncio
    async def test_zero_flow_detection(self, steam_meter_connector):
        """Test detection of zero flow condition."""
        await steam_meter_connector.connect()

        # Simulate zero flow
        await steam_meter_connector._simulate_zero_flow()

        measurement = await steam_meter_connector.read_measurement()

        # Flow should be very close to zero
        assert measurement.volumetric_flow < 1.0

    @pytest.mark.asyncio
    async def test_reverse_flow_detection(self, steam_meter_connector):
        """Test detection of reverse flow."""
        await steam_meter_connector.connect()

        # Check if meter supports reverse flow detection
        capabilities = await steam_meter_connector.get_capabilities()

        if capabilities.get('reverse_flow_detection'):
            # Simulate reverse flow
            await steam_meter_connector._simulate_reverse_flow()

            measurement = await steam_meter_connector.read_measurement()

            # Flow should be negative or flagged
            assert measurement.volumetric_flow < 0 or \
                   measurement.flow_direction == 'REVERSE'


# ============================================================================
# TEST CLASS: TOTALIZER OPERATIONS
# ============================================================================

@pytest.mark.integration
@pytest.mark.steam_meter
class TestTotalizerOperations:
    """Test steam meter totalizer operations."""

    @pytest.mark.asyncio
    async def test_read_totalizer(self, steam_meter_connector):
        """Test reading totalizer value."""
        await steam_meter_connector.connect()

        totalizer = await steam_meter_connector.read_totalizer()

        assert totalizer is not None
        assert 'value' in totalizer
        assert 'unit' in totalizer
        assert totalizer['value'] >= 0

    @pytest.mark.asyncio
    async def test_totalizer_increment(self, steam_meter_connector):
        """Test totalizer increments over time."""
        await steam_meter_connector.connect()

        # Read initial totalizer
        initial = await steam_meter_connector.read_totalizer()
        initial_value = initial['value']

        # Wait for flow
        await asyncio.sleep(5)

        # Read again
        final = await steam_meter_connector.read_totalizer()
        final_value = final['value']

        # Should have incremented
        assert final_value >= initial_value

    @pytest.mark.asyncio
    async def test_totalizer_reset(self, steam_meter_connector):
        """Test totalizer reset operation."""
        await steam_meter_connector.connect()

        # Reset totalizer
        result = await steam_meter_connector.reset_totalizer()

        assert result is True

        # Read totalizer
        totalizer = await steam_meter_connector.read_totalizer()

        # Should be zero or very small
        assert totalizer['value'] < 10

    @pytest.mark.asyncio
    async def test_multiple_totalizers(self, steam_meter_connector):
        """Test meters with multiple totalizers."""
        await steam_meter_connector.connect()

        totalizers = await steam_meter_connector.read_all_totalizers()

        assert isinstance(totalizers, dict)
        # May have: total, forward, reverse totalizers
        assert len(totalizers) >= 1


# ============================================================================
# TEST CLASS: COMPENSATION
# ============================================================================

@pytest.mark.integration
@pytest.mark.steam_meter
class TestPressureTemperatureCompensation:
    """Test pressure and temperature compensation."""

    @pytest.mark.asyncio
    async def test_pressure_compensation(self, steam_meter_connector):
        """Test pressure compensation for flow measurement."""
        await steam_meter_connector.connect()

        measurement = await steam_meter_connector.read_measurement()

        # Should have compensation applied
        assert hasattr(measurement, 'compensated')
        if measurement.compensated:
            assert measurement.compensation_factor is not None

    @pytest.mark.asyncio
    async def test_temperature_compensation(self, steam_meter_connector):
        """Test temperature compensation."""
        await steam_meter_connector.connect()

        measurement = await steam_meter_connector.read_measurement()

        # Temperature should be used for density calculation
        assert measurement.temperature > 0
        assert measurement.density is not None

    @pytest.mark.asyncio
    async def test_density_calculation(self, steam_meter_connector):
        """Test steam density calculation."""
        await steam_meter_connector.connect()

        measurement = await steam_meter_connector.read_measurement()

        # Density should be realistic for steam (4-6 kg/m3 at 10 bar)
        assert 2 < measurement.density < 10

    @pytest.mark.asyncio
    async def test_saturated_steam_compensation(self, steam_meter_connector):
        """Test saturated steam compensation method."""
        await steam_meter_connector.connect()

        measurement = await steam_meter_connector.read_measurement()

        # For saturated steam, temperature should match saturation temp
        # at given pressure (approximately)
        # At 10 bar, saturation temp ≈ 184°C
        if 9 < measurement.pressure < 11:
            assert 175 < measurement.temperature < 195


# ============================================================================
# TEST CLASS: ENERGY MEASUREMENT
# ============================================================================

@pytest.mark.integration
@pytest.mark.steam_meter
class TestEnergyMeasurement:
    """Test energy flow measurement and calculation."""

    @pytest.mark.asyncio
    async def test_energy_flow_calculation(self, steam_meter_connector):
        """Test energy flow calculation."""
        await steam_meter_connector.connect()

        measurement = await steam_meter_connector.read_measurement()

        # Energy flow should be calculated
        assert hasattr(measurement, 'energy_flow')
        assert measurement.energy_flow > 0

    @pytest.mark.asyncio
    async def test_energy_flow_units(self, steam_meter_connector):
        """Test energy flow unit conversion."""
        await steam_meter_connector.connect()

        # Read in kW
        energy_kw = await steam_meter_connector.read_energy_flow(unit='kW')

        # Read in MW
        energy_mw = await steam_meter_connector.read_energy_flow(unit='MW')

        # Should be consistent (kW / 1000 = MW)
        assert abs(energy_kw['value'] / 1000 - energy_mw['value']) < 0.1

    @pytest.mark.asyncio
    async def test_energy_totalizer(self, steam_meter_connector):
        """Test cumulative energy measurement."""
        await steam_meter_connector.connect()

        energy_total = await steam_meter_connector.read_energy_totalizer()

        assert energy_total is not None
        assert energy_total['value'] >= 0
        assert energy_total['unit'] in ['kWh', 'MWh', 'GJ']

    @pytest.mark.asyncio
    async def test_energy_calculation_accuracy(self, steam_meter_connector):
        """Test energy calculation accuracy."""
        await steam_meter_connector.connect()

        measurement = await steam_meter_connector.read_measurement()

        # Manual calculation: E = m * h
        # where m = mass flow (kg/s), h = enthalpy (kJ/kg)
        mass_flow_kg_s = measurement.mass_flow / 3600
        enthalpy_kj_kg = 2800  # Approximate for 10 bar saturated steam

        expected_energy_kw = mass_flow_kg_s * enthalpy_kj_kg

        # Measured energy should be close to calculated
        assert abs(measurement.energy_flow - expected_energy_kw) / expected_energy_kw < 0.1


# ============================================================================
# TEST CLASS: STEAM QUALITY
# ============================================================================

@pytest.mark.integration
@pytest.mark.steam_meter
class TestSteamQualityMeasurement:
    """Test steam quality (dryness fraction) measurement."""

    @pytest.mark.asyncio
    async def test_steam_quality_reading(self, steam_meter_connector):
        """Test reading steam quality."""
        await steam_meter_connector.connect()

        measurement = await steam_meter_connector.read_measurement()

        # Steam quality should be between 0 and 1
        if hasattr(measurement, 'steam_quality'):
            assert 0 <= measurement.steam_quality <= 1

    @pytest.mark.asyncio
    async def test_wet_steam_detection(self, steam_meter_connector):
        """Test detection of wet steam (quality < 1)."""
        await steam_meter_connector.connect()

        # Simulate wet steam
        await steam_meter_connector._simulate_wet_steam(quality=0.90)

        measurement = await steam_meter_connector.read_measurement()

        if hasattr(measurement, 'steam_quality'):
            assert measurement.steam_quality < 0.95

    @pytest.mark.asyncio
    async def test_superheated_steam_detection(self, steam_meter_connector):
        """Test detection of superheated steam."""
        await steam_meter_connector.connect()

        # Simulate superheated steam (temp > saturation temp)
        await steam_meter_connector._simulate_superheat(degrees=20)

        measurement = await steam_meter_connector.read_measurement()

        # Temperature should exceed saturation temperature
        if hasattr(measurement, 'superheat'):
            assert measurement.superheat > 0


# ============================================================================
# TEST CLASS: DIAGNOSTICS
# ============================================================================

@pytest.mark.integration
@pytest.mark.steam_meter
class TestMeterDiagnostics:
    """Test steam meter diagnostics and health monitoring."""

    @pytest.mark.asyncio
    async def test_meter_diagnostics(self, steam_meter_connector):
        """Test reading meter diagnostics."""
        await steam_meter_connector.connect()

        diagnostics = await steam_meter_connector.get_diagnostics()

        assert diagnostics is not None
        assert 'status' in diagnostics
        assert 'quality' in diagnostics

    @pytest.mark.asyncio
    async def test_meter_health_status(self, steam_meter_connector):
        """Test meter health status indicators."""
        await steam_meter_connector.connect()

        health = await steam_meter_connector.get_health_status()

        assert 'overall_status' in health
        assert health['overall_status'] in ['HEALTHY', 'WARNING', 'FAULT']

    @pytest.mark.asyncio
    async def test_sensor_health(self, steam_meter_connector):
        """Test individual sensor health."""
        await steam_meter_connector.connect()

        diagnostics = await steam_meter_connector.get_diagnostics()

        if 'sensors' in diagnostics:
            for sensor, status in diagnostics['sensors'].items():
                assert status in ['OK', 'DEGRADED', 'FAILED']

    @pytest.mark.asyncio
    async def test_error_detection(self, steam_meter_connector):
        """Test error and fault detection."""
        await steam_meter_connector.connect()

        errors = await steam_meter_connector.get_active_errors()

        assert isinstance(errors, list)
        # May be empty if no errors

    @pytest.mark.asyncio
    async def test_signal_strength(self, steam_meter_connector):
        """Test signal strength monitoring."""
        await steam_meter_connector.connect()

        diagnostics = await steam_meter_connector.get_diagnostics()

        if 'signal_strength' in diagnostics:
            # Signal strength should be 0-100%
            assert 0 <= diagnostics['signal_strength'] <= 100


# ============================================================================
# TEST CLASS: CALIBRATION
# ============================================================================

@pytest.mark.integration
@pytest.mark.steam_meter
class TestMeterCalibration:
    """Test meter calibration status and operations."""

    @pytest.mark.asyncio
    async def test_calibration_status(self, steam_meter_connector):
        """Test reading calibration status."""
        await steam_meter_connector.connect()

        calibration = await steam_meter_connector.get_calibration_status()

        assert calibration is not None
        assert 'last_calibration_date' in calibration
        assert 'days_since_calibration' in calibration

    @pytest.mark.asyncio
    async def test_calibration_due_check(self, steam_meter_connector):
        """Test calibration due indicator."""
        await steam_meter_connector.connect()

        calibration = await steam_meter_connector.get_calibration_status()

        assert 'calibration_due' in calibration
        assert isinstance(calibration['calibration_due'], bool)

        # If more than 365 days, should be due
        if calibration['days_since_calibration'] > 365:
            assert calibration['calibration_due'] is True

    @pytest.mark.asyncio
    async def test_calibration_drift(self, steam_meter_connector):
        """Test calibration drift monitoring."""
        await steam_meter_connector.connect()

        calibration = await steam_meter_connector.get_calibration_status()

        if 'drift_estimate' in calibration:
            # Drift should be small percentage
            assert 0 <= calibration['drift_estimate'] < 5

    @pytest.mark.asyncio
    async def test_zero_point_calibration(self, steam_meter_connector):
        """Test zero point calibration."""
        await steam_meter_connector.connect()

        # Perform zero calibration
        result = await steam_meter_connector.calibrate_zero_point()

        # Should succeed or not be implemented
        assert result is True or result is None


# ============================================================================
# TEST CLASS: HISTORICAL DATA
# ============================================================================

@pytest.mark.integration
@pytest.mark.steam_meter
class TestHistoricalData:
    """Test historical data retrieval from meters."""

    @pytest.mark.asyncio
    async def test_historical_data_retrieval(self, steam_meter_connector):
        """Test retrieving historical meter data."""
        await steam_meter_connector.connect()

        end_time = DeterministicClock.utcnow()
        start_time = end_time - timedelta(hours=1)

        history = await steam_meter_connector.get_historical_data(
            start_time,
            end_time
        )

        assert isinstance(history, list)
        # May be empty if no historical data available

    @pytest.mark.asyncio
    async def test_historical_data_aggregation(self, steam_meter_connector):
        """Test historical data aggregation."""
        await steam_meter_connector.connect()

        end_time = DeterministicClock.utcnow()
        start_time = end_time - timedelta(hours=24)

        # Get hourly aggregated data
        aggregated = await steam_meter_connector.get_historical_data(
            start_time,
            end_time,
            aggregation='hourly'
        )

        if len(aggregated) > 0:
            # Should have ~24 data points (one per hour)
            assert len(aggregated) <= 25

    @pytest.mark.asyncio
    async def test_data_export(self, steam_meter_connector):
        """Test exporting meter data."""
        await steam_meter_connector.connect()

        # Export last 24 hours
        export_data = await steam_meter_connector.export_data(
            hours=24,
            format='csv'
        )

        assert export_data is not None
        # Should be CSV string or bytes
        assert isinstance(export_data, (str, bytes))


# ============================================================================
# TEST CLASS: MULTI-METER OPERATIONS
# ============================================================================

@pytest.mark.integration
@pytest.mark.steam_meter
class TestMultiMeterOperations:
    """Test operations with multiple steam meters."""

    @pytest.mark.asyncio
    async def test_connect_multiple_meters(self, steam_meter_config):
        """Test connecting to multiple meters."""
        meters = []

        for i in range(1, 4):
            config = steam_meter_config.copy()
            config.meter_id = f"SM-{i:03d}"
            config.port = 5020 + i - 1

            connector = SteamMeterConnector(config)
            await connector.connect()

            meters.append(connector)

        # All should be connected
        assert all(m.is_connected for m in meters)

        # Cleanup
        for meter in meters:
            await meter.disconnect()

    @pytest.mark.asyncio
    async def test_read_all_meters_parallel(self, steam_meter_config):
        """Test reading all meters in parallel."""
        meters = []

        for i in range(1, 3):
            config = steam_meter_config.copy()
            config.meter_id = f"SM-{i:03d}"
            config.port = 5020 + i - 1

            connector = SteamMeterConnector(config)
            await connector.connect()
            meters.append(connector)

        # Read all meters in parallel
        tasks = [m.read_measurement() for m in meters]
        results = await asyncio.gather(*tasks)

        assert len(results) == len(meters)
        assert all(r is not None for r in results)

        # Cleanup
        for meter in meters:
            await meter.disconnect()

    @pytest.mark.asyncio
    async def test_aggregate_meter_flows(self, steam_meter_config):
        """Test aggregating flow from multiple meters."""
        meters = []

        for i in range(1, 3):
            config = steam_meter_config.copy()
            config.meter_id = f"SM-{i:03d}"
            config.port = 5020 + i - 1

            connector = SteamMeterConnector(config)
            await connector.connect()
            meters.append(connector)

        # Read all meters
        measurements = []
        for meter in meters:
            measurement = await meter.read_measurement()
            measurements.append(measurement)

        # Calculate total flow
        total_flow = sum(m.mass_flow for m in measurements)

        assert total_flow > 0

        # Cleanup
        for meter in meters:
            await meter.disconnect()


# ============================================================================
# TEST CLASS: ERROR HANDLING
# ============================================================================

@pytest.mark.integration
@pytest.mark.steam_meter
class TestMeterErrorHandling:
    """Test error handling for steam meters."""

    @pytest.mark.asyncio
    async def test_read_error_handling(self, steam_meter_connector):
        """Test handling read errors gracefully."""
        await steam_meter_connector.connect()

        # Simulate communication error
        original_read = steam_meter_connector._read_modbus

        async def failing_read(*args, **kwargs):
            raise Exception("Communication timeout")

        steam_meter_connector._read_modbus = failing_read

        # Should handle error gracefully
        measurement = await steam_meter_connector.read_measurement()

        # Should return None or have error flag
        assert measurement is None or hasattr(measurement, 'error')

    @pytest.mark.asyncio
    async def test_out_of_range_handling(self, steam_meter_connector):
        """Test handling out-of-range values."""
        await steam_meter_connector.connect()

        # Simulate extreme value
        await steam_meter_connector._simulate_extreme_value(flow=999999)

        measurement = await steam_meter_connector.read_measurement()

        # Should flag as bad quality or limit value
        if measurement:
            assert measurement.quality == MeterQuality.BAD or \
                   measurement.volumetric_flow < 999999

    @pytest.mark.asyncio
    async def test_connection_loss_recovery(self, steam_meter_connector):
        """Test recovery from connection loss."""
        await steam_meter_connector.connect()

        # Simulate connection loss
        steam_meter_connector.is_connected = False

        # Attempt read (should trigger reconnection)
        measurement = await steam_meter_connector.read_measurement()

        # Should reconnect or handle gracefully
        assert measurement is None or steam_meter_connector.is_connected


# ============================================================================
# TEST CLASS: PERFORMANCE
# ============================================================================

@pytest.mark.integration
@pytest.mark.steam_meter
@pytest.mark.slow
class TestMeterPerformance:
    """Test steam meter performance characteristics."""

    @pytest.mark.asyncio
    async def test_read_throughput(self, steam_meter_connector):
        """Test meter read throughput."""
        await steam_meter_connector.connect()

        num_reads = 100
        start_time = DeterministicClock.utcnow()

        for _ in range(num_reads):
            await steam_meter_connector.read_measurement()

        end_time = DeterministicClock.utcnow()

        duration = (end_time - start_time).total_seconds()
        throughput = num_reads / duration

        # Should achieve reasonable throughput (>1 read/sec)
        assert throughput > 1

    @pytest.mark.asyncio
    async def test_continuous_reading_stability(self, steam_meter_connector):
        """Test continuous reading stability."""
        await steam_meter_connector.connect()

        # Read continuously for 30 seconds
        readings = []
        errors = 0

        for _ in range(60):
            try:
                measurement = await steam_meter_connector.read_measurement()
                if measurement:
                    readings.append(measurement.volumetric_flow)
            except Exception:
                errors += 1

            await asyncio.sleep(0.5)

        # Should have mostly successful reads
        assert len(readings) > 50
        assert errors < 5


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--asyncio-mode=auto"])
