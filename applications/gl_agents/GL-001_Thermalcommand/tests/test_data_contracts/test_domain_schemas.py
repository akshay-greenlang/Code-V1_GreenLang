"""
GL-001 ThermalCommand: Domain Schema Tests

Comprehensive tests for all canonical domain schemas.

Test Coverage:
- Schema instantiation and validation
- Field type validation
- Range constraints
- Cross-field validation
- Serialization/deserialization
- Provenance tracking
"""

import json
from datetime import datetime, timezone, timedelta
from typing import Any, Dict

import pytest

# Import schemas to test
import sys
sys.path.insert(0, str(__file__).replace('\\', '/').rsplit('/tests/', 1)[0])

from data_contracts.domain_schemas import (
    # Enums
    DataQualityLevel,
    AlarmSeverity,
    EquipmentStatus,
    TripStatus,
    ForecastConfidence,
    PriceMarket,
    FuelType,
    # Base models
    ProvenanceInfo,
    DataQualityMetrics,
    BaseDataContract,
    # Domain schemas
    ProcessSensorData,
    EnergyConsumptionData,
    SafetySystemStatus,
    ProductionSchedule,
    WeatherForecast,
    EnergyPrices,
    EquipmentHealth,
    AlarmState,
    # Sub-schemas
    SteamHeaderData,
    ValvePosition,
    FuelConsumption,
    BoilerPerformance,
    SISPermissive,
    TripPoint,
    BypassRecord,
    BatchPlan,
    UnitTarget,
    Campaign,
    HourlyForecast,
    ElectricityPrice,
    FuelPrice,
    VibrationData,
    LubeOilAnalysis,
    FoulingIndicator,
    RemainingUsefulLife,
    AlarmRecord,
    AlarmStatistics,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def base_timestamp() -> datetime:
    """Provide a base timestamp for tests."""
    return datetime.now(timezone.utc)


@pytest.fixture
def sample_provenance(base_timestamp) -> ProvenanceInfo:
    """Create sample provenance info."""
    return ProvenanceInfo(
        source_system="SCADA",
        source_tag="STEAM_HDR_A_PIC_001.PV",
        timestamp_collected=base_timestamp,
    )


@pytest.fixture
def sample_steam_header() -> SteamHeaderData:
    """Create sample steam header data."""
    return SteamHeaderData(
        header_id="headerA",
        pressure_barg=42.5,
        temperature_c=420.0,
        flow_total_tph=150.0,
        flow_setpoint_tph=155.0,
        superheat_c=50.0,
    )


@pytest.fixture
def sample_boiler_performance() -> BoilerPerformance:
    """Create sample boiler performance data."""
    return BoilerPerformance(
        boiler_id="B1",
        fuel_flow_kgh=5000.0,
        steam_output_tph=75.0,
        max_rate_tph=100.0,
        efficiency_pct=88.5,
        turndown_ratio=4.0,
        load_pct=75.0,
        flue_gas_temp_c=180.0,
        excess_air_pct=15.0,
    )


# =============================================================================
# ProvenanceInfo Tests
# =============================================================================

class TestProvenanceInfo:
    """Tests for ProvenanceInfo model."""

    def test_create_basic(self, base_timestamp):
        """Test basic provenance creation."""
        provenance = ProvenanceInfo(
            source_system="SCADA",
            timestamp_collected=base_timestamp,
        )
        assert provenance.source_system == "SCADA"
        assert provenance.record_id is not None
        assert len(provenance.record_id) == 36  # UUID format

    def test_compute_hash(self, sample_provenance):
        """Test hash computation for integrity."""
        data = {"pressure": 42.5, "temperature": 420.0}
        hash1 = sample_provenance.compute_hash(data)
        hash2 = sample_provenance.compute_hash(data)
        assert hash1 == hash2  # Deterministic

        data_modified = {"pressure": 42.6, "temperature": 420.0}
        hash3 = sample_provenance.compute_hash(data_modified)
        assert hash1 != hash3  # Different data = different hash

    def test_transformation_chain(self, base_timestamp):
        """Test transformation chain tracking."""
        provenance = ProvenanceInfo(
            source_system="SCADA",
            timestamp_collected=base_timestamp,
            transformation_chain=["raw_ingest", "unit_conversion", "validation"],
        )
        assert len(provenance.transformation_chain) == 3
        assert "unit_conversion" in provenance.transformation_chain


# =============================================================================
# DataQualityMetrics Tests
# =============================================================================

class TestDataQualityMetrics:
    """Tests for DataQualityMetrics model."""

    def test_create_with_scores(self):
        """Test quality metrics creation with scores."""
        metrics = DataQualityMetrics(
            quality_level=DataQualityLevel.GOOD,
            completeness_score=0.98,
            validity_score=0.95,
            timeliness_score=0.99,
            consistency_score=0.97,
            overall_score=96.5,
        )
        assert metrics.quality_level == DataQualityLevel.GOOD
        assert metrics.overall_score == 96.5

    def test_score_boundaries(self):
        """Test score validation at boundaries."""
        # Valid at boundaries
        metrics = DataQualityMetrics(
            completeness_score=0.0,
            validity_score=1.0,
            overall_score=0.0,
        )
        assert metrics.completeness_score == 0.0
        assert metrics.validity_score == 1.0

    def test_invalid_score_rejected(self):
        """Test that invalid scores are rejected."""
        with pytest.raises(ValueError):
            DataQualityMetrics(completeness_score=1.5)  # > 1.0

        with pytest.raises(ValueError):
            DataQualityMetrics(overall_score=-1.0)  # < 0


# =============================================================================
# SteamHeaderData Tests
# =============================================================================

class TestSteamHeaderData:
    """Tests for SteamHeaderData model."""

    def test_create_valid(self, sample_steam_header):
        """Test valid steam header creation."""
        assert sample_steam_header.header_id == "headerA"
        assert sample_steam_header.pressure_barg == 42.5
        assert sample_steam_header.temperature_c == 420.0

    def test_header_id_pattern(self):
        """Test header ID pattern validation."""
        # Valid patterns
        for header_id in ["headerA", "headerB", "headerC"]:
            header = SteamHeaderData(
                header_id=header_id,
                pressure_barg=10.0,
                temperature_c=200.0,
                flow_total_tph=50.0,
            )
            assert header.header_id == header_id

    def test_pressure_limits(self):
        """Test pressure limit validation."""
        # Valid pressure
        header = SteamHeaderData(
            header_id="headerA",
            pressure_barg=150.0,
            temperature_c=500.0,
            flow_total_tph=100.0,
        )
        assert header.pressure_barg == 150.0

        # Invalid - too high
        with pytest.raises(ValueError):
            SteamHeaderData(
                header_id="headerA",
                pressure_barg=250.0,  # > 200 limit
                temperature_c=300.0,
                flow_total_tph=100.0,
            )

    def test_temperature_limits(self):
        """Test temperature limit validation."""
        # Valid at boundary
        header = SteamHeaderData(
            header_id="headerA",
            pressure_barg=100.0,
            temperature_c=550.0,  # Near max
            flow_total_tph=100.0,
        )
        assert header.temperature_c == 550.0

        # Invalid - too high
        with pytest.raises(ValueError):
            SteamHeaderData(
                header_id="headerA",
                pressure_barg=100.0,
                temperature_c=650.0,  # > 600 limit
                flow_total_tph=100.0,
            )

    def test_steam_conditions_validation(self):
        """Test physical steam conditions validation."""
        # Temperature too low for pressure (subcooled)
        with pytest.raises(ValueError, match="too low"):
            SteamHeaderData(
                header_id="headerA",
                pressure_barg=40.0,
                temperature_c=100.0,  # Should be much higher for 40 bar
                flow_total_tph=100.0,
            )


# =============================================================================
# ValvePosition Tests
# =============================================================================

class TestValvePosition:
    """Tests for ValvePosition model."""

    def test_create_valid(self):
        """Test valid valve position creation."""
        valve = ValvePosition(
            valve_id="FCV-001",
            position_pct=75.5,
            setpoint_pct=75.0,
            mode="auto",
        )
        assert valve.position_pct == 75.5
        assert valve.mode == "auto"

    def test_position_boundaries(self):
        """Test position boundaries."""
        # Valid at 0%
        valve = ValvePosition(valve_id="V1", position_pct=0.0)
        assert valve.position_pct == 0.0

        # Valid at 100%
        valve = ValvePosition(valve_id="V1", position_pct=100.0)
        assert valve.position_pct == 100.0

        # Invalid > 100%
        with pytest.raises(ValueError):
            ValvePosition(valve_id="V1", position_pct=105.0)

    def test_control_modes(self):
        """Test control mode validation."""
        for mode in ["auto", "manual", "cascade", "remote"]:
            valve = ValvePosition(valve_id="V1", position_pct=50.0, mode=mode)
            assert valve.mode == mode


# =============================================================================
# ProcessSensorData Tests
# =============================================================================

class TestProcessSensorData:
    """Tests for ProcessSensorData model."""

    def test_create_minimal(self, base_timestamp):
        """Test minimal process sensor data creation."""
        data = ProcessSensorData(
            facility_id="PLANT-001",
            area_id="BOILER-HOUSE",
            timestamp=base_timestamp,
        )
        assert data.facility_id == "PLANT-001"
        assert data.schema_version == "1.0.0"

    def test_create_with_steam_headers(self, base_timestamp, sample_steam_header):
        """Test with steam header data."""
        data = ProcessSensorData(
            facility_id="PLANT-001",
            area_id="BOILER-HOUSE",
            timestamp=base_timestamp,
            steam_headers={"headerA": sample_steam_header},
        )
        assert "headerA" in data.steam_headers
        assert data.steam_headers["headerA"].pressure_barg == 42.5

    def test_create_with_all_fields(self, base_timestamp, sample_steam_header, sample_provenance):
        """Test with all optional fields."""
        data = ProcessSensorData(
            facility_id="PLANT-001",
            area_id="BOILER-HOUSE",
            timestamp=base_timestamp,
            steam_headers={"headerA": sample_steam_header},
            valve_positions={
                "FCV-001": ValvePosition(valve_id="FCV-001", position_pct=75.0)
            },
            temperatures={"T001": 420.0, "T002": 380.0},
            flows={"F001": 150.0},
            pressures={"P001": 42.5},
            levels={"L001": 65.0},
            raw_tags={"custom_tag": 123.45},
            scan_rate_ms=500,
            provenance=sample_provenance,
        )
        assert len(data.temperatures) == 2
        assert data.scan_rate_ms == 500

    def test_scan_rate_limits(self, base_timestamp):
        """Test scan rate limit validation."""
        # Valid at minimum
        data = ProcessSensorData(
            facility_id="P1",
            area_id="A1",
            timestamp=base_timestamp,
            scan_rate_ms=100,
        )
        assert data.scan_rate_ms == 100

        # Invalid - too low
        with pytest.raises(ValueError):
            ProcessSensorData(
                facility_id="P1",
                area_id="A1",
                timestamp=base_timestamp,
                scan_rate_ms=50,  # < 100 minimum
            )

    def test_serialization(self, base_timestamp, sample_steam_header):
        """Test JSON serialization/deserialization."""
        data = ProcessSensorData(
            facility_id="PLANT-001",
            area_id="BOILER-HOUSE",
            timestamp=base_timestamp,
            steam_headers={"headerA": sample_steam_header},
        )

        # Serialize
        json_str = data.model_dump_json()
        assert isinstance(json_str, str)

        # Deserialize
        parsed = json.loads(json_str)
        assert parsed["facility_id"] == "PLANT-001"
        assert parsed["steam_headers"]["headerA"]["pressure_barg"] == 42.5


# =============================================================================
# EnergyConsumptionData Tests
# =============================================================================

class TestEnergyConsumptionData:
    """Tests for EnergyConsumptionData model."""

    def test_create_minimal(self, base_timestamp):
        """Test minimal energy consumption data."""
        data = EnergyConsumptionData(
            facility_id="PLANT-001",
            timestamp=base_timestamp,
            period_start=base_timestamp - timedelta(hours=1),
            period_end=base_timestamp,
        )
        assert data.facility_id == "PLANT-001"
        assert data.electricity_net_mwh == 0.0

    def test_net_electricity_calculation(self, base_timestamp):
        """Test net electricity auto-calculation."""
        data = EnergyConsumptionData(
            facility_id="PLANT-001",
            timestamp=base_timestamp,
            period_start=base_timestamp - timedelta(hours=1),
            period_end=base_timestamp,
            electricity_import_mwh=100.0,
            electricity_export_mwh=30.0,
        )
        assert data.electricity_net_mwh == 70.0

    def test_with_fuel_consumption(self, base_timestamp):
        """Test with fuel consumption data."""
        fuel = FuelConsumption(
            fuel_type=FuelType.NATURAL_GAS,
            flow_rate=5000.0,
            flow_unit="Nm3/h",
            heating_value_mj_kg=50.0,
            energy_rate_mw=25.0,
        )

        data = EnergyConsumptionData(
            facility_id="PLANT-001",
            timestamp=base_timestamp,
            period_start=base_timestamp - timedelta(hours=1),
            period_end=base_timestamp,
            fuel_consumption=[fuel],
        )
        assert len(data.fuel_consumption) == 1
        assert data.fuel_consumption[0].fuel_type == FuelType.NATURAL_GAS

    def test_with_boiler_performance(self, base_timestamp, sample_boiler_performance):
        """Test with boiler performance data."""
        data = EnergyConsumptionData(
            facility_id="PLANT-001",
            timestamp=base_timestamp,
            period_start=base_timestamp - timedelta(hours=1),
            period_end=base_timestamp,
            boiler_performance={"B1": sample_boiler_performance},
        )
        assert "B1" in data.boiler_performance
        assert data.boiler_performance["B1"].efficiency_pct == 88.5


# =============================================================================
# SafetySystemStatus Tests
# =============================================================================

class TestSafetySystemStatus:
    """Tests for SafetySystemStatus model."""

    def test_create_minimal(self, base_timestamp):
        """Test minimal safety status creation."""
        status = SafetySystemStatus(
            facility_id="PLANT-001",
            area_id="BOILER-HOUSE",
            timestamp=base_timestamp,
            dispatch_enabled=True,
        )
        assert status.dispatch_enabled is True
        assert status.sil_status == "operational"

    def test_with_permissives(self, base_timestamp):
        """Test with SIS permissives."""
        permissive = SISPermissive(
            permissive_id="PERM-001",
            description="Master dispatch enabled",
            is_enabled=True,
            required_for=["auto_dispatch", "load_balancing"],
            last_change=base_timestamp - timedelta(hours=2),
        )

        status = SafetySystemStatus(
            facility_id="PLANT-001",
            area_id="BOILER-HOUSE",
            timestamp=base_timestamp,
            permissives={"PERM-001": permissive},
            dispatch_enabled=True,
        )
        assert "PERM-001" in status.permissives
        assert status.permissives["PERM-001"].is_enabled is True

    def test_with_trip_points(self, base_timestamp):
        """Test with trip points."""
        trip = TripPoint(
            trip_id="TRIP-001",
            description="High pressure trip",
            status=TripStatus.NORMAL,
            setpoint=45.0,
            setpoint_unit="bar(g)",
            current_value=42.0,
            trip_count_24h=0,
        )

        status = SafetySystemStatus(
            facility_id="PLANT-001",
            area_id="BOILER-HOUSE",
            timestamp=base_timestamp,
            trip_points={"TRIP-001": trip},
            dispatch_enabled=True,
        )
        assert status.trip_points["TRIP-001"].status == TripStatus.NORMAL

    def test_with_active_bypasses(self, base_timestamp):
        """Test with active bypasses."""
        bypass = BypassRecord(
            bypass_id="BYP-001",
            element_id="TRIP-001",
            element_type="trip",
            reason="Maintenance on pressure transmitter PT-001",
            authorized_by="JohnDoe",
            start_time=base_timestamp - timedelta(hours=1),
            max_duration_hours=8,
            compensatory_measures=["Manual monitoring every 15 min"],
        )

        status = SafetySystemStatus(
            facility_id="PLANT-001",
            area_id="BOILER-HOUSE",
            timestamp=base_timestamp,
            active_bypasses=[bypass],
            dispatch_enabled=False,
        )
        assert status.bypass_count == 1
        assert len(status.active_bypasses[0].compensatory_measures) == 1


# =============================================================================
# ProductionSchedule Tests
# =============================================================================

class TestProductionSchedule:
    """Tests for ProductionSchedule model."""

    def test_create_minimal(self, base_timestamp):
        """Test minimal production schedule."""
        schedule = ProductionSchedule(
            facility_id="PLANT-001",
            timestamp=base_timestamp,
            schedule_horizon_start=base_timestamp,
            schedule_horizon_end=base_timestamp + timedelta(days=7),
        )
        assert schedule.schedule_version == "1"

    def test_with_batch_plans(self, base_timestamp):
        """Test with batch plans."""
        batch = BatchPlan(
            batch_id="BATCH-001",
            product_code="PROD-A",
            product_name="Product A",
            quantity=100.0,
            quantity_unit="tonnes",
            scheduled_start=base_timestamp + timedelta(hours=2),
            scheduled_end=base_timestamp + timedelta(hours=10),
            heat_demand_mwth=15.0,
            priority=3,
            unit_id="U1",
        )

        schedule = ProductionSchedule(
            facility_id="PLANT-001",
            timestamp=base_timestamp,
            schedule_horizon_start=base_timestamp,
            schedule_horizon_end=base_timestamp + timedelta(days=7),
            batch_plans=[batch],
        )
        assert len(schedule.batch_plans) == 1
        assert schedule.batch_plans[0].heat_demand_mwth == 15.0

    def test_batch_schedule_validation(self, base_timestamp):
        """Test batch schedule validation (end after start)."""
        with pytest.raises(ValueError, match="after"):
            BatchPlan(
                batch_id="BATCH-001",
                product_code="PROD-A",
                product_name="Product A",
                quantity=100.0,
                quantity_unit="tonnes",
                scheduled_start=base_timestamp + timedelta(hours=10),
                scheduled_end=base_timestamp + timedelta(hours=2),  # Before start!
                heat_demand_mwth=15.0,
                unit_id="U1",
            )


# =============================================================================
# WeatherForecast Tests
# =============================================================================

class TestWeatherForecast:
    """Tests for WeatherForecast model."""

    def test_create_minimal(self, base_timestamp):
        """Test minimal weather forecast."""
        forecast = WeatherForecast(
            facility_id="PLANT-001",
            timestamp=base_timestamp,
            location_lat=51.5074,
            location_lon=-0.1278,
            forecast_issued=base_timestamp,
            forecast_provider="NWS",
            current_temperature_c=15.0,
            current_humidity_pct=65.0,
        )
        assert forecast.current_temperature_c == 15.0

    def test_with_hourly_forecasts(self, base_timestamp):
        """Test with hourly forecasts."""
        hourly = [
            HourlyForecast(
                forecast_time=base_timestamp + timedelta(hours=i),
                temperature_c=15.0 + i,
                humidity_pct=65.0 - i,
                wind_speed_ms=5.0,
                wind_direction_deg=180.0,
            )
            for i in range(24)
        ]

        forecast = WeatherForecast(
            facility_id="PLANT-001",
            timestamp=base_timestamp,
            location_lat=51.5074,
            location_lon=-0.1278,
            forecast_issued=base_timestamp,
            forecast_provider="NWS",
            current_temperature_c=15.0,
            current_humidity_pct=65.0,
            hourly_forecasts=hourly,
        )
        assert len(forecast.hourly_forecasts) == 24

    def test_location_boundaries(self, base_timestamp):
        """Test location coordinate boundaries."""
        # Valid at extremes
        forecast = WeatherForecast(
            facility_id="P1",
            timestamp=base_timestamp,
            location_lat=90.0,  # North pole
            location_lon=180.0,
            forecast_issued=base_timestamp,
            forecast_provider="NWS",
            current_temperature_c=-50.0,
            current_humidity_pct=50.0,
        )
        assert forecast.location_lat == 90.0

        # Invalid latitude
        with pytest.raises(ValueError):
            WeatherForecast(
                facility_id="P1",
                timestamp=base_timestamp,
                location_lat=91.0,  # > 90 is invalid
                location_lon=0.0,
                forecast_issued=base_timestamp,
                forecast_provider="NWS",
                current_temperature_c=15.0,
                current_humidity_pct=65.0,
            )


# =============================================================================
# EnergyPrices Tests
# =============================================================================

class TestEnergyPrices:
    """Tests for EnergyPrices model."""

    def test_create_minimal(self, base_timestamp):
        """Test minimal energy prices."""
        prices = EnergyPrices(
            facility_id="PLANT-001",
            timestamp=base_timestamp,
            pricing_region="PJM",
            current_rt_price_usd_mwh=45.50,
        )
        assert prices.current_rt_price_usd_mwh == 45.50

    def test_negative_prices(self, base_timestamp):
        """Test negative prices (valid for oversupply conditions)."""
        prices = EnergyPrices(
            facility_id="PLANT-001",
            timestamp=base_timestamp,
            pricing_region="ERCOT",
            current_rt_price_usd_mwh=-25.00,  # Negative is valid
        )
        assert prices.current_rt_price_usd_mwh == -25.00

    def test_with_day_ahead_prices(self, base_timestamp):
        """Test with day-ahead prices."""
        da_prices = [
            ElectricityPrice(
                price_time=base_timestamp + timedelta(hours=i),
                market=PriceMarket.DAY_AHEAD,
                price_usd_mwh=40.0 + i * 2,
                is_forecast=True,
            )
            for i in range(24)
        ]

        prices = EnergyPrices(
            facility_id="PLANT-001",
            timestamp=base_timestamp,
            pricing_region="PJM",
            current_rt_price_usd_mwh=45.50,
            day_ahead_prices=da_prices,
        )
        assert len(prices.day_ahead_prices) == 24

    def test_with_fuel_prices(self, base_timestamp):
        """Test with fuel prices."""
        fuel = FuelPrice(
            fuel_type=FuelType.NATURAL_GAS,
            price=3.50,
            price_unit="$/MMBtu",
            effective_date=base_timestamp,
        )

        prices = EnergyPrices(
            facility_id="PLANT-001",
            timestamp=base_timestamp,
            pricing_region="PJM",
            current_rt_price_usd_mwh=45.50,
            fuel_prices=[fuel],
        )
        assert prices.fuel_prices[0].price == 3.50


# =============================================================================
# EquipmentHealth Tests
# =============================================================================

class TestEquipmentHealth:
    """Tests for EquipmentHealth model."""

    def test_create_minimal(self, base_timestamp):
        """Test minimal equipment health."""
        health = EquipmentHealth(
            facility_id="PLANT-001",
            timestamp=base_timestamp,
            equipment_id="PUMP-001",
            equipment_type="pump",
            health_score=0.85,
        )
        assert health.health_score == 0.85
        assert health.equipment_status == EquipmentStatus.RUNNING

    def test_with_vibration_data(self, base_timestamp):
        """Test with vibration data."""
        vibration = VibrationData(
            measurement_point="DE-H",
            velocity_mm_s=2.5,
            displacement_um=25.0,
            temperature_c=65.0,
            alarm_level=AlarmSeverity.DIAGNOSTIC,
            trend="stable",
        )

        health = EquipmentHealth(
            facility_id="PLANT-001",
            timestamp=base_timestamp,
            equipment_id="PUMP-001",
            equipment_type="pump",
            health_score=0.85,
            vibration_data={"DE-H": vibration},
        )
        assert "DE-H" in health.vibration_data
        assert health.vibration_data["DE-H"].velocity_mm_s == 2.5

    def test_with_rul_prediction(self, base_timestamp):
        """Test with RUL prediction."""
        rul = RemainingUsefulLife(
            equipment_id="PUMP-001",
            component="bearing",
            rul_hours=2500.0,
            rul_confidence_pct=85.0,
            failure_mode="wear",
            recommended_action="Schedule bearing replacement",
            action_deadline=base_timestamp + timedelta(days=30),
        )

        health = EquipmentHealth(
            facility_id="PLANT-001",
            timestamp=base_timestamp,
            equipment_id="PUMP-001",
            equipment_type="pump",
            health_score=0.75,
            rul_predictions=[rul],
        )
        assert health.rul_predictions[0].rul_hours == 2500.0

    def test_health_score_limits(self, base_timestamp):
        """Test health score boundaries."""
        # Valid at 0
        health = EquipmentHealth(
            facility_id="P1",
            timestamp=base_timestamp,
            equipment_id="E1",
            equipment_type="pump",
            health_score=0.0,
        )
        assert health.health_score == 0.0

        # Invalid > 1.0
        with pytest.raises(ValueError):
            EquipmentHealth(
                facility_id="P1",
                timestamp=base_timestamp,
                equipment_id="E1",
                equipment_type="pump",
                health_score=1.5,
            )


# =============================================================================
# AlarmState Tests
# =============================================================================

class TestAlarmState:
    """Tests for AlarmState model."""

    def test_create_minimal(self, base_timestamp):
        """Test minimal alarm state."""
        state = AlarmState(
            facility_id="PLANT-001",
            timestamp=base_timestamp,
        )
        assert state.active_count == 0
        assert state.is_alarm_flood is False

    def test_with_active_alarms(self, base_timestamp):
        """Test with active alarms."""
        alarm = AlarmRecord(
            alarm_id="ALM-001",
            tag="STEAM_HDR_A_PIC_001",
            description="High pressure header A",
            severity=AlarmSeverity.HIGH,
            state="active",
            alarm_time=base_timestamp - timedelta(minutes=5),
            value_at_alarm=46.5,
            setpoint=45.0,
            area="BOILER-HOUSE",
        )

        state = AlarmState(
            facility_id="PLANT-001",
            timestamp=base_timestamp,
            active_alarms=[alarm],
        )
        assert state.active_count == 1
        assert state.high_count == 1
        assert state.unacknowledged_count == 1

    def test_alarm_counts_updated(self, base_timestamp):
        """Test that alarm counts are auto-updated."""
        alarms = [
            AlarmRecord(
                alarm_id=f"ALM-{i}",
                tag=f"TAG-{i}",
                description=f"Alarm {i}",
                severity=severity,
                state="active",
                alarm_time=base_timestamp,
                area="AREA-1",
            )
            for i, severity in enumerate([
                AlarmSeverity.CRITICAL,
                AlarmSeverity.HIGH,
                AlarmSeverity.HIGH,
                AlarmSeverity.MEDIUM,
                AlarmSeverity.LOW,
            ])
        ]

        state = AlarmState(
            facility_id="PLANT-001",
            timestamp=base_timestamp,
            active_alarms=alarms,
        )
        assert state.critical_count == 1
        assert state.high_count == 2
        assert state.medium_count == 1
        assert state.low_count == 1
        assert state.active_count == 5


# =============================================================================
# JSON Schema Generation Tests
# =============================================================================

class TestJSONSchemaGeneration:
    """Tests for JSON Schema generation."""

    def test_process_sensor_data_schema(self):
        """Test ProcessSensorData JSON schema generation."""
        schema = ProcessSensorData.model_json_schema()
        assert "properties" in schema
        assert "facility_id" in schema["properties"]
        assert "timestamp" in schema["properties"]

    def test_energy_consumption_schema(self):
        """Test EnergyConsumptionData JSON schema generation."""
        schema = EnergyConsumptionData.model_json_schema()
        assert "properties" in schema
        assert "fuel_consumption" in schema["properties"]

    def test_all_domain_schemas_generate(self):
        """Test all domain schemas can generate JSON schemas."""
        schemas = [
            ProcessSensorData,
            EnergyConsumptionData,
            SafetySystemStatus,
            ProductionSchedule,
            WeatherForecast,
            EnergyPrices,
            EquipmentHealth,
            AlarmState,
        ]

        for schema_class in schemas:
            json_schema = schema_class.model_json_schema()
            assert "properties" in json_schema
            assert "title" in json_schema


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
