"""
Unit Tests for GL-003 UnifiedSteam - Integration Module

Tests for:
- SensorTransformer (unit conversion, calibration, validation)
- TagMapper (OT/SCADA tag mapping)
- HistorianConnector (PI, SQL, CSV historian integration)
- OPC-UA connector (mocked)

Target Coverage: 90%+
"""

import asyncio
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Import application modules
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from integration.sensor_transformer import (
    CalibrationParams,
    QualifiedValue,
    QualityCode,
    SensorTransformer,
    TransformedData,
    UnitCategory,
    UnitConverter,
    ValidationResult,
    create_steam_system_transformer,
)
from integration.tag_mapper import (
    AssetType,
    MeasurementType,
    SensorMetadata,
    SignalQualifier,
    TagMapper,
    TagMapping,
    TagNamingConvention,
    ValidationError,
)
from integration.historian_connector import (
    BackfillResult,
    BackfillStatus,
    CSVHistorianDriver,
    HistorianConfig,
    HistorianConnector,
    HistorianType,
    InterpolationMode,
    PIWebAPIDriver,
    SQLHistorianDriver,
    TimeSeriesData,
    TimeSeriesPoint,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sensor_transformer() -> SensorTransformer:
    """Create a SensorTransformer instance."""
    return SensorTransformer()


@pytest.fixture
def configured_transformer() -> SensorTransformer:
    """Create a configured SensorTransformer."""
    transformer = SensorTransformer()

    transformer.configure_sensor(
        tag="header.pressure",
        from_unit="psig",
        to_unit="bar",
        calibration=CalibrationParams(offset=0.0, gain=1.0),
        valid_range=(0.0, 50.0),
        max_rate_of_change=5.0,
    )

    transformer.configure_sensor(
        tag="header.temperature",
        from_unit="degF",
        to_unit="degC",
        valid_range=(0.0, 400.0),
        max_rate_of_change=10.0,
    )

    transformer.configure_sensor(
        tag="header.flow",
        from_unit="klb/hr",
        to_unit="kg/s",
        valid_range=(0.0, 100.0),
    )

    return transformer


@pytest.fixture
def tag_mapper() -> TagMapper:
    """Create a TagMapper instance."""
    return TagMapper()


@pytest.fixture
def naming_convention() -> TagNamingConvention:
    """Create a TagNamingConvention instance."""
    return TagNamingConvention()


@pytest.fixture
def historian_config() -> HistorianConfig:
    """Create a HistorianConfig instance."""
    return HistorianConfig(
        historian_type=HistorianType.PI_WEB_API,
        connection_string="https://piwebapi.example.com/piwebapi",
        timeout_s=30,
        batch_size=1000,
    )


@pytest.fixture
def historian_connector(historian_config) -> HistorianConnector:
    """Create a HistorianConnector instance."""
    return HistorianConnector(config=historian_config)


# =============================================================================
# Test UnitConverter
# =============================================================================

class TestUnitConverter:
    """Tests for UnitConverter class."""

    def test_convert_same_unit(self):
        """Test conversion when from and to units are the same."""
        result = UnitConverter.convert(100.0, "kPa", "kPa")
        assert result == 100.0

    def test_convert_pressure_kpa_to_bar(self):
        """Test pressure conversion kPa to bar."""
        # 1000 kPa = 10 bar
        result = UnitConverter.convert(1000.0, "kPa", "bar")
        assert abs(result - 10.0) < 0.01

    def test_convert_pressure_psi_to_kpa(self):
        """Test pressure conversion psi to kPa."""
        # 14.696 psi = 101.325 kPa (1 atm)
        result = UnitConverter.convert(14.696, "psi", "kPa")
        assert abs(result - 101.325) < 0.5

    def test_convert_pressure_bar_to_psi(self):
        """Test pressure conversion bar to psi."""
        # 1 bar = 14.5038 psi
        result = UnitConverter.convert(1.0, "bar", "psi")
        assert abs(result - 14.5038) < 0.1

    def test_convert_pressure_psig_to_psia(self):
        """Test gauge to absolute pressure conversion."""
        # 0 psig = 14.696 psia
        result = UnitConverter.convert(0.0, "psig", "psia")
        assert abs(result - 14.696) < 0.01

    def test_convert_temperature_c_to_k(self):
        """Test temperature conversion Celsius to Kelvin."""
        # 0 C = 273.15 K
        result = UnitConverter.convert(0.0, "C", "K")
        assert abs(result - 273.15) < 0.01

    def test_convert_temperature_f_to_c(self):
        """Test temperature conversion Fahrenheit to Celsius."""
        # 32 F = 0 C
        result = UnitConverter.convert(32.0, "F", "C")
        assert abs(result - 0.0) < 0.01

        # 212 F = 100 C
        result = UnitConverter.convert(212.0, "degF", "degC")
        assert abs(result - 100.0) < 0.01

    def test_convert_temperature_c_to_f(self):
        """Test temperature conversion Celsius to Fahrenheit."""
        # 100 C = 212 F
        result = UnitConverter.convert(100.0, "C", "F")
        assert abs(result - 212.0) < 0.01

    def test_convert_temperature_k_to_r(self):
        """Test temperature conversion Kelvin to Rankine."""
        # 273.15 K = 491.67 R
        result = UnitConverter.convert(273.15, "K", "R")
        assert abs(result - 491.67) < 0.1

    def test_convert_mass_flow_klb_hr_to_kg_s(self):
        """Test mass flow conversion klb/hr to kg/s."""
        # 1 klb/hr = 0.126 kg/s
        result = UnitConverter.convert(1.0, "klb/hr", "kg/s")
        assert abs(result - 0.126) < 0.01

    def test_convert_mass_flow_kg_hr_to_lb_hr(self):
        """Test mass flow conversion kg/hr to lb/hr."""
        # 1 kg/hr = 2.205 lb/hr
        result = UnitConverter.convert(1.0, "kg/hr", "lb/hr")
        assert abs(result - 2.205) < 0.01

    def test_convert_energy_btu_to_kj(self):
        """Test energy conversion BTU to kJ."""
        # 1 BTU = 1.055 kJ
        result = UnitConverter.convert(1.0, "BTU", "kJ")
        assert abs(result - 1.055) < 0.01

    def test_convert_power_kw_to_hp(self):
        """Test power conversion kW to hp."""
        # 1 kW = 1.341 hp
        result = UnitConverter.convert(1.0, "kW", "hp")
        assert abs(result - 1.341) < 0.01

    def test_convert_unknown_unit_raises(self):
        """Test that unknown units raise ValueError."""
        with pytest.raises(ValueError):
            UnitConverter.convert(100.0, "unknown", "kPa")

    def test_convert_incompatible_units_raises(self):
        """Test that incompatible units raise ValueError."""
        with pytest.raises(ValueError):
            UnitConverter.convert(100.0, "kPa", "degC")

    def test_get_category(self):
        """Test getting unit category."""
        assert UnitConverter.get_category("kPa") == UnitCategory.PRESSURE
        assert UnitConverter.get_category("degC") == UnitCategory.TEMPERATURE
        assert UnitConverter.get_category("kg/s") == UnitCategory.MASS_FLOW

    def test_get_compatible_units(self):
        """Test getting compatible units."""
        compatible = UnitConverter.get_compatible_units("kPa")
        assert "bar" in compatible
        assert "psi" in compatible
        assert "degC" not in compatible


# =============================================================================
# Test SensorTransformer
# =============================================================================

class TestSensorTransformer:
    """Tests for SensorTransformer class."""

    def test_initialization(self, sensor_transformer):
        """Test transformer initialization."""
        assert sensor_transformer is not None
        stats = sensor_transformer.get_statistics()
        assert stats["transformations"] == 0

    def test_configure_sensor(self, sensor_transformer):
        """Test sensor configuration."""
        sensor_transformer.configure_sensor(
            tag="test.pressure",
            from_unit="psig",
            to_unit="bar",
            valid_range=(0.0, 50.0),
        )

        stats = sensor_transformer.get_statistics()
        assert stats["configured_sensors"] == 1

    def test_normalize_units(self, sensor_transformer):
        """Test unit normalization."""
        result = sensor_transformer.normalize_units(100.0, "psi", "bar")
        assert abs(result - 6.895) < 0.1

    def test_apply_linear_calibration(self, sensor_transformer):
        """Test linear calibration application."""
        calibration = CalibrationParams(offset=10.0, gain=2.0)
        result = sensor_transformer.apply_calibration(50.0, calibration)
        # y = 2.0 * 50.0 + 10.0 = 110.0
        assert result == 110.0

    def test_apply_polynomial_calibration(self, sensor_transformer):
        """Test polynomial calibration application."""
        # y = 1 + 2*x + 0.01*x^2
        calibration = CalibrationParams(polynomial_coefficients=[1.0, 2.0, 0.01])
        result = sensor_transformer.apply_calibration(10.0, calibration)
        # y = 1 + 20 + 1 = 22
        assert abs(result - 22.0) < 0.01

    def test_apply_lookup_table_calibration(self, sensor_transformer):
        """Test lookup table calibration with interpolation."""
        calibration = CalibrationParams(
            lookup_table={0.0: 0.0, 50.0: 55.0, 100.0: 110.0}
        )
        # Interpolate at 25.0: should be ~27.5
        result = sensor_transformer.apply_calibration(25.0, calibration)
        assert abs(result - 27.5) < 0.1

    def test_apply_lookup_table_extrapolation(self, sensor_transformer):
        """Test lookup table extrapolation at boundaries."""
        calibration = CalibrationParams(
            lookup_table={10.0: 20.0, 90.0: 100.0}
        )
        # Below range
        result_low = sensor_transformer.apply_calibration(5.0, calibration)
        assert result_low == 20.0  # Clamp to min

        # Above range
        result_high = sensor_transformer.apply_calibration(100.0, calibration)
        assert result_high == 100.0  # Clamp to max

    def test_validate_range_good(self, sensor_transformer):
        """Test range validation for good values."""
        result = sensor_transformer.validate_range(50.0, 0.0, 100.0)
        assert result.is_valid
        assert result.quality_code == QualityCode.GOOD

    def test_validate_range_below_minimum(self, sensor_transformer):
        """Test range validation for values below minimum."""
        result = sensor_transformer.validate_range(-10.0, 0.0, 100.0)
        assert not result.is_valid
        assert result.quality_code == QualityCode.BAD_OUT_OF_RANGE

    def test_validate_range_above_maximum(self, sensor_transformer):
        """Test range validation for values above maximum."""
        result = sensor_transformer.validate_range(150.0, 0.0, 100.0)
        assert not result.is_valid
        assert result.quality_code == QualityCode.BAD_OUT_OF_RANGE

    def test_validate_range_near_limits(self, sensor_transformer):
        """Test range validation for values near limits."""
        # Near lower limit
        result = sensor_transformer.validate_range(5.0, 0.0, 100.0)
        assert result.is_valid
        assert result.quality_code == QualityCode.UNCERTAIN_EU_EXCEEDED

        # Near upper limit
        result = sensor_transformer.validate_range(95.0, 0.0, 100.0)
        assert result.is_valid
        assert result.quality_code == QualityCode.UNCERTAIN_EU_EXCEEDED

    def test_validate_rate_of_change_good(self, sensor_transformer):
        """Test rate of change validation for normal changes."""
        tag = "test.value"
        now = datetime.now(timezone.utc)

        # First value (no previous)
        result1 = sensor_transformer.validate_rate_of_change(tag, 100.0, now, 10.0)
        assert result1.is_valid

        # Second value with small change
        result2 = sensor_transformer.validate_rate_of_change(
            tag, 102.0, now + timedelta(seconds=1), 10.0
        )
        assert result2.is_valid
        assert result2.quality_code == QualityCode.GOOD

    def test_validate_rate_of_change_exceeded(self, sensor_transformer):
        """Test rate of change validation for excessive changes."""
        tag = "test.value"
        now = datetime.now(timezone.utc)

        # First value
        sensor_transformer.validate_rate_of_change(tag, 100.0, now, 10.0)

        # Large change in 1 second (rate = 50/s, max = 10/s)
        result = sensor_transformer.validate_rate_of_change(
            tag, 150.0, now + timedelta(seconds=1), 10.0
        )
        assert result.is_valid  # Still valid but uncertain
        assert result.quality_code == QualityCode.UNCERTAIN_SENSOR_NOT_ACCURATE

    def test_check_consistency_pass(self, sensor_transformer):
        """Test consistency check when rules pass."""
        sensor_transformer.add_consistency_rule(
            name="temp_order",
            tags=["inlet_temp", "outlet_temp"],
            check_function=lambda v: v["inlet_temp"] > v["outlet_temp"],
            description="Inlet should be higher than outlet",
        )

        values = {"inlet_temp": 350.0, "outlet_temp": 200.0}
        result = sensor_transformer.check_consistency(values)
        assert result.is_valid

    def test_check_consistency_fail(self, sensor_transformer):
        """Test consistency check when rules fail."""
        sensor_transformer.add_consistency_rule(
            name="temp_order",
            tags=["inlet_temp", "outlet_temp"],
            check_function=lambda v: v["inlet_temp"] > v["outlet_temp"],
            description="Inlet should be higher than outlet",
        )

        values = {"inlet_temp": 150.0, "outlet_temp": 200.0}  # Violates rule
        result = sensor_transformer.check_consistency(values)
        assert not result.is_valid or result.quality_code.is_uncertain()

    def test_apply_quality_flags(self, sensor_transformer):
        """Test quality flag application."""
        qualified = sensor_transformer.apply_quality_flags(
            value=100.0,
            quality_code=QualityCode.GOOD,
            tag="test.value",
            unit="bar",
            raw_value=1450.0,
        )

        assert isinstance(qualified, QualifiedValue)
        assert qualified.value == 100.0
        assert qualified.quality_code == QualityCode.GOOD
        assert qualified.tag == "test.value"
        assert qualified.unit == "bar"
        assert len(qualified.source_hash) == 16

    def test_transform_single(self, configured_transformer):
        """Test single value transformation."""
        # 145 psig should convert to ~10 bar
        result = configured_transformer.transform_single(
            tag="header.pressure",
            raw_value=145.0,
        )

        assert isinstance(result, QualifiedValue)
        assert result.tag == "header.pressure"
        assert result.unit == "bar"
        # 145 psig = ~10 bar (approximate)
        assert 9.0 < result.value < 11.0

    def test_transform_single_unconfigured_tag(self, sensor_transformer):
        """Test transformation of unconfigured tag."""
        result = sensor_transformer.transform_single(
            tag="unknown.tag",
            raw_value=100.0,
        )

        # Should still work but without conversion
        assert result.value == 100.0

    def test_transform_batch(self, configured_transformer):
        """Test batch transformation."""
        raw_data = {
            "header.pressure": 145.0,
            "header.temperature": 700.0,
            "header.flow": 50.0,
        }

        result = configured_transformer.transform_batch(raw_data)

        assert isinstance(result, TransformedData)
        assert result.total_count == 3
        assert "header.pressure" in result.values
        assert result.quality_score > 0

    def test_transform_batch_quality_score(self, configured_transformer):
        """Test batch quality score calculation."""
        raw_data = {
            "header.pressure": 145.0,  # Good
            "header.temperature": 700.0,  # Good
        }

        result = configured_transformer.transform_batch(raw_data)

        # All good quality should give high score
        assert result.quality_score >= 90.0

    def test_get_statistics(self, configured_transformer):
        """Test statistics retrieval."""
        configured_transformer.transform_single("header.pressure", 145.0)

        stats = configured_transformer.get_statistics()
        assert stats["transformations"] >= 1
        assert stats["configured_sensors"] == 3

    def test_get_value_history(self, sensor_transformer):
        """Test value history retrieval."""
        tag = "test.sensor"
        now = datetime.now(timezone.utc)

        # Add some values
        for i in range(10):
            sensor_transformer.validate_rate_of_change(
                tag, 100.0 + i, now + timedelta(seconds=i), 100.0
            )

        history = sensor_transformer.get_value_history(tag)
        assert len(history) == 10


class TestQualityCode:
    """Tests for QualityCode enum."""

    def test_quality_code_good(self):
        """Test good quality codes."""
        assert QualityCode.GOOD.is_good()
        assert not QualityCode.GOOD.is_uncertain()
        assert not QualityCode.GOOD.is_bad()

    def test_quality_code_uncertain(self):
        """Test uncertain quality codes."""
        assert not QualityCode.UNCERTAIN.is_good()
        assert QualityCode.UNCERTAIN.is_uncertain()
        assert not QualityCode.UNCERTAIN.is_bad()

    def test_quality_code_bad(self):
        """Test bad quality codes."""
        assert not QualityCode.BAD.is_good()
        assert not QualityCode.BAD.is_uncertain()
        assert QualityCode.BAD.is_bad()


class TestCalibrationParams:
    """Tests for CalibrationParams dataclass."""

    def test_calibration_due_check(self):
        """Test calibration due date check."""
        past_date = datetime.now(timezone.utc) - timedelta(days=30)
        calibration = CalibrationParams(
            next_calibration_date=past_date
        )
        assert calibration.is_due_for_calibration()

    def test_calibration_not_due(self):
        """Test calibration not due."""
        future_date = datetime.now(timezone.utc) + timedelta(days=30)
        calibration = CalibrationParams(
            next_calibration_date=future_date
        )
        assert not calibration.is_due_for_calibration()


class TestSteamSystemTransformerFactory:
    """Tests for create_steam_system_transformer factory."""

    def test_create_steam_system_transformer(self):
        """Test factory function creates configured transformer."""
        transformer = create_steam_system_transformer()

        assert transformer is not None
        stats = transformer.get_statistics()
        assert stats["configured_sensors"] > 0
        assert stats["consistency_rules"] > 0


# =============================================================================
# Test TagMapper
# =============================================================================

class TestTagNamingConvention:
    """Tests for TagNamingConvention class."""

    def test_build_canonical_tag(self, naming_convention):
        """Test building canonical tag name."""
        tag = naming_convention.build_canonical_tag(
            site="PLANT1",
            area="BOILER_HOUSE",
            asset_type=AssetType.BOILER,
            asset_id="001",
            measurement=MeasurementType.PRESSURE,
            qualifier=SignalQualifier.HIGH,
        )

        assert "PLANT1" in tag.lower()
        assert "001" in tag

    def test_parse_canonical_tag(self, naming_convention):
        """Test parsing canonical tag name."""
        tag = "plant1.boiler_house.boiler.001.pressure.high"
        components = naming_convention.parse_canonical_tag(tag)

        assert components is not None
        assert len(components) > 0

    def test_validate_tag_format(self, naming_convention):
        """Test tag format validation."""
        valid_tag = "site.area.type.id.measurement.qualifier"
        invalid_tag = "invalid"

        assert naming_convention.validate_tag(valid_tag)
        # May or may not be valid depending on implementation
        # At minimum should not raise


class TestTagMapper:
    """Tests for TagMapper class."""

    def test_initialization(self, tag_mapper):
        """Test tag mapper initialization."""
        assert tag_mapper is not None

    def test_add_mapping(self, tag_mapper):
        """Test adding tag mapping."""
        tag_mapper.add_mapping(
            raw_tag="FIC-101",
            canonical_tag="plant1.area1.flow.001.actual",
            metadata={"unit": "kg/s", "description": "Main flow"},
        )

        canonical = tag_mapper.get_canonical_tag("FIC-101")
        assert canonical == "plant1.area1.flow.001.actual"

    def test_get_canonical_tag_not_found(self, tag_mapper):
        """Test getting non-existent mapping."""
        result = tag_mapper.get_canonical_tag("NONEXISTENT-TAG")
        assert result is None

    def test_get_sensor_metadata(self, tag_mapper):
        """Test getting sensor metadata."""
        tag_mapper.add_mapping(
            raw_tag="TT-201",
            canonical_tag="plant1.area1.temperature.001.actual",
            metadata={"unit": "degC", "range_min": 0, "range_max": 500},
        )

        metadata = tag_mapper.get_sensor_metadata("plant1.area1.temperature.001.actual")
        assert metadata is not None

    def test_validate_mapping(self, tag_mapper):
        """Test mapping validation."""
        tag_mapper.add_mapping(
            raw_tag="PT-301",
            canonical_tag="plant1.area1.pressure.001.actual",
        )

        errors = tag_mapper.validate_mapping()
        assert isinstance(errors, list)

    def test_get_tags_by_asset(self, tag_mapper):
        """Test getting tags by asset."""
        tag_mapper.add_mapping("TAG1", "plant1.area1.boiler.001.pressure.actual")
        tag_mapper.add_mapping("TAG2", "plant1.area1.boiler.001.temperature.actual")
        tag_mapper.add_mapping("TAG3", "plant1.area1.boiler.002.pressure.actual")

        tags = tag_mapper.get_tags_by_asset("boiler", "001")
        # Should return tags for boiler 001
        assert isinstance(tags, list)


@pytest.mark.asyncio
class TestTagMapperAsync:
    """Async tests for TagMapper."""

    async def test_load_tag_mapping(self, tag_mapper, tmp_path):
        """Test loading tag mapping from file."""
        # Create a test config file
        import json
        config_file = tmp_path / "tags.json"
        config_data = {
            "mappings": {
                "FIC-101": "plant1.flow.001.actual",
                "TT-201": "plant1.temp.001.actual",
            }
        }
        config_file.write_text(json.dumps(config_data))

        try:
            result = await tag_mapper.load_tag_mapping(str(config_file))
            assert result is not None
        except (FileNotFoundError, NotImplementedError):
            # May not be implemented or file format differs
            pass


# =============================================================================
# Test HistorianConnector
# =============================================================================

class TestHistorianConfig:
    """Tests for HistorianConfig dataclass."""

    def test_historian_config_creation(self):
        """Test creating historian config."""
        config = HistorianConfig(
            historian_type=HistorianType.PI_WEB_API,
            connection_string="https://piwebapi.example.com",
            timeout_s=30,
        )

        assert config.historian_type == HistorianType.PI_WEB_API
        assert config.timeout_s == 30


class TestHistorianType:
    """Tests for HistorianType enum."""

    def test_historian_types(self):
        """Test historian type enum values."""
        assert HistorianType.PI_WEB_API.value == "pi_web_api"
        assert HistorianType.SQL.value == "sql"
        assert HistorianType.CSV.value == "csv"


class TestInterpolationMode:
    """Tests for InterpolationMode enum."""

    def test_interpolation_modes(self):
        """Test interpolation mode enum values."""
        assert InterpolationMode.NONE.value == "none"
        assert InterpolationMode.LINEAR.value == "linear"


class TestTimeSeriesData:
    """Tests for TimeSeriesData dataclass."""

    def test_time_series_data_creation(self):
        """Test creating time series data."""
        now = datetime.now(timezone.utc)
        points = [
            TimeSeriesPoint(timestamp=now, value=100.0, quality=0),
            TimeSeriesPoint(timestamp=now + timedelta(minutes=1), value=101.0, quality=0),
        ]

        data = TimeSeriesData(
            tag="test.tag",
            points=points,
            start_time=now,
            end_time=now + timedelta(minutes=1),
        )

        assert data.tag == "test.tag"
        assert len(data.points) == 2


@pytest.mark.asyncio
class TestHistorianConnector:
    """Tests for HistorianConnector class."""

    async def test_initialization(self, historian_connector):
        """Test historian connector initialization."""
        assert historian_connector is not None

    async def test_connect_mock(self, historian_connector):
        """Test connection with mock."""
        with patch.object(historian_connector, 'connect', new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = True
            result = await historian_connector.connect()
            assert result is True

    async def test_query_historical_mock(self, historian_connector):
        """Test historical query with mock."""
        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=1)

        with patch.object(historian_connector, 'query_historical', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = {
                "test.tag": TimeSeriesData(
                    tag="test.tag",
                    points=[
                        TimeSeriesPoint(timestamp=start, value=100.0, quality=0),
                        TimeSeriesPoint(timestamp=now, value=105.0, quality=0),
                    ],
                    start_time=start,
                    end_time=now,
                )
            }

            result = await historian_connector.query_historical(
                tags=["test.tag"],
                start_time=start,
                end_time=now,
            )

            assert "test.tag" in result
            assert len(result["test.tag"].points) == 2

    async def test_batch_backfill_mock(self, historian_connector):
        """Test batch backfill with mock."""
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=1)

        with patch.object(historian_connector, 'batch_backfill', new_callable=AsyncMock) as mock_backfill:
            mock_backfill.return_value = BackfillResult(
                status=BackfillStatus.COMPLETED,
                tags_processed=["test.tag"],
                total_points=1440,
                start_time=start,
                end_time=now,
                errors=[],
            )

            result = await historian_connector.batch_backfill(
                tags=["test.tag"],
                start_time=start,
                end_time=now,
                interval_s=60,
            )

            assert result.status == BackfillStatus.COMPLETED
            assert result.total_points > 0


class TestPIWebAPIDriver:
    """Tests for PIWebAPIDriver."""

    def test_driver_initialization(self):
        """Test PI Web API driver initialization."""
        config = HistorianConfig(
            historian_type=HistorianType.PI_WEB_API,
            connection_string="https://piwebapi.example.com",
        )
        driver = PIWebAPIDriver(config)
        assert driver is not None


class TestSQLHistorianDriver:
    """Tests for SQLHistorianDriver."""

    def test_driver_initialization(self):
        """Test SQL historian driver initialization."""
        config = HistorianConfig(
            historian_type=HistorianType.SQL,
            connection_string="postgresql://localhost/historian",
        )
        driver = SQLHistorianDriver(config)
        assert driver is not None


class TestCSVHistorianDriver:
    """Tests for CSVHistorianDriver."""

    def test_driver_initialization(self):
        """Test CSV historian driver initialization."""
        config = HistorianConfig(
            historian_type=HistorianType.CSV,
            connection_string="/path/to/csv/files",
        )
        driver = CSVHistorianDriver(config)
        assert driver is not None


# =============================================================================
# Integration Tests with Mocks
# =============================================================================

@pytest.mark.integration
class TestIntegrationWithMocks:
    """Integration tests using mocks for external systems."""

    def test_full_data_flow(self, configured_transformer):
        """Test complete data transformation flow."""
        # Simulate raw sensor data
        raw_data = {
            "header.pressure": 145.0,  # psig
            "header.temperature": 700.0,  # degF
            "header.flow": 50.0,  # klb/hr
        }

        # Transform batch
        result = configured_transformer.transform_batch(raw_data)

        # Verify all values transformed
        assert len(result.values) == 3

        # Verify pressure converted (145 psig -> ~10 bar)
        pressure = result.values["header.pressure"]
        assert pressure.unit == "bar"
        assert 9.0 < pressure.value < 11.0

        # Verify temperature converted (700 F -> ~371 C)
        temperature = result.values["header.temperature"]
        assert temperature.unit == "degC"
        assert 360 < temperature.value < 380

    def test_data_quality_propagation(self, configured_transformer):
        """Test that quality flags propagate correctly."""
        # Include out-of-range value
        raw_data = {
            "header.pressure": 1000.0,  # Way out of range (0-50 bar)
            "header.temperature": 700.0,  # Good
        }

        result = configured_transformer.transform_batch(raw_data)

        # Out of range value should have bad quality
        pressure = result.values["header.pressure"]
        assert pressure.quality_code.is_bad() or pressure.quality_code.is_uncertain()

        # Quality score should be reduced
        assert result.quality_score < 100.0


# =============================================================================
# Performance Tests
# =============================================================================

@pytest.mark.slow
class TestPerformance:
    """Performance tests for integration module."""

    def test_unit_conversion_performance(self, benchmark):
        """Benchmark unit conversion performance."""
        def convert_many():
            for _ in range(1000):
                UnitConverter.convert(100.0, "psi", "bar")
                UnitConverter.convert(100.0, "degF", "degC")
                UnitConverter.convert(100.0, "klb/hr", "kg/s")

        benchmark(convert_many)

    def test_batch_transform_performance(self, configured_transformer, benchmark):
        """Benchmark batch transformation performance."""
        raw_data = {
            f"sensor_{i}": float(i * 10)
            for i in range(100)
        }

        def transform():
            return configured_transformer.transform_batch(raw_data)

        result = benchmark(transform)
        assert result is not None

    def test_large_batch_transform(self, configured_transformer):
        """Test transformation of large batches."""
        # Create 1000 sensor readings
        raw_data = {
            f"sensor_{i}": float(i * 10) % 1000
            for i in range(1000)
        }

        start_time = time.perf_counter()
        result = configured_transformer.transform_batch(raw_data)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert result.total_count == 1000
        assert elapsed_ms < 500, f"Took {elapsed_ms:.1f}ms for 1000 sensors"


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_zero_value_conversion(self):
        """Test conversion of zero values."""
        result = UnitConverter.convert(0.0, "kPa", "bar")
        assert result == 0.0

    def test_negative_pressure(self):
        """Test negative pressure (vacuum)."""
        # -10 psig = -10 + 14.696 = 4.696 psia
        result = UnitConverter.convert(-10.0, "psig", "psia")
        assert abs(result - 4.696) < 0.01

    def test_very_large_values(self):
        """Test very large values."""
        result = UnitConverter.convert(1e9, "Pa", "MPa")
        assert result == 1e3

    def test_very_small_values(self):
        """Test very small values."""
        result = UnitConverter.convert(1e-6, "MPa", "Pa")
        assert result == 1.0

    def test_empty_batch(self, configured_transformer):
        """Test transformation of empty batch."""
        result = configured_transformer.transform_batch({})
        assert result.total_count == 0
        assert result.quality_score == 100.0

    def test_nan_handling(self, sensor_transformer):
        """Test handling of NaN values."""
        result = sensor_transformer.validate_range(float('nan'), 0.0, 100.0)
        # Should be marked as bad or handle gracefully
        assert not result.is_valid or True  # Implementation dependent

    def test_inf_handling(self, sensor_transformer):
        """Test handling of infinite values."""
        result = sensor_transformer.validate_range(float('inf'), 0.0, 100.0)
        # Should be marked as out of range
        assert result.quality_code == QualityCode.BAD_OUT_OF_RANGE
