"""
Tests for GL-001 ThermalCommand API Schemas

Unit tests for Pydantic models and validation logic.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from api.api_schemas import (
    AssetCapacity,
    AssetCost,
    AssetEfficiency,
    AssetEmissions,
    AssetHealth,
    AssetState,
    AssetStatus,
    AssetType,
    Constraint,
    ConstraintPriority,
    ConstraintType,
    DemandUpdate,
    DispatchPlan,
    ForecastType,
    KPI,
    OptimizationObjective,
    SetpointRecommendation,
    AllocationRequest,
    AllocationResponse,
    AlarmEvent,
    AlarmSeverity,
    AlarmStatus,
    MaintenanceTrigger,
    MaintenanceType,
    MaintenanceUrgency,
    ExplainabilitySummary,
    FeatureImportance,
    DecisionExplanation,
    PaginationParams,
    TimeRangeFilter,
)


# =============================================================================
# Asset Model Tests
# =============================================================================

class TestAssetCapacity:
    """Tests for AssetCapacity model."""

    def test_valid_capacity(self):
        """Test creating valid asset capacity."""
        capacity = AssetCapacity(
            thermal_capacity_mw=100.0,
            min_output_mw=20.0,
            max_output_mw=100.0,
            ramp_up_rate_mw_min=2.0,
            ramp_down_rate_mw_min=3.0,
            min_uptime_hours=4.0,
            min_downtime_hours=2.0,
            startup_time_minutes=30.0,
        )
        assert capacity.thermal_capacity_mw == 100.0
        assert capacity.min_output_mw == 20.0

    def test_max_must_exceed_min(self):
        """Test that max_output_mw must be >= min_output_mw."""
        with pytest.raises(ValueError, match="max_output_mw must be >= min_output_mw"):
            AssetCapacity(
                thermal_capacity_mw=100.0,
                min_output_mw=50.0,
                max_output_mw=30.0,  # Invalid: less than min
                ramp_up_rate_mw_min=2.0,
                ramp_down_rate_mw_min=3.0,
            )

    def test_negative_values_rejected(self):
        """Test that negative values are rejected."""
        with pytest.raises(ValueError):
            AssetCapacity(
                thermal_capacity_mw=-10.0,  # Invalid
                min_output_mw=0,
                max_output_mw=50.0,
                ramp_up_rate_mw_min=2.0,
                ramp_down_rate_mw_min=3.0,
            )


class TestAssetEfficiency:
    """Tests for AssetEfficiency model."""

    def test_valid_efficiency(self):
        """Test creating valid asset efficiency."""
        efficiency = AssetEfficiency(
            thermal_efficiency=0.88,
            electrical_efficiency=0.42,
        )
        assert efficiency.thermal_efficiency == 0.88
        assert efficiency.electrical_efficiency == 0.42

    def test_efficiency_bounds(self):
        """Test efficiency must be between 0 and 1."""
        with pytest.raises(ValueError):
            AssetEfficiency(thermal_efficiency=1.5)  # Invalid: > 1

        with pytest.raises(ValueError):
            AssetEfficiency(thermal_efficiency=-0.1)  # Invalid: < 0


class TestAssetState:
    """Tests for AssetState model."""

    @pytest.fixture
    def valid_asset_state(self):
        """Create a valid asset state for testing."""
        return AssetState(
            asset_name="CHP Unit 1",
            asset_type=AssetType.CHP,
            status=AssetStatus.ONLINE,
            current_output_mw=45.5,
            current_setpoint_mw=50.0,
            supply_temperature_c=95.0,
            return_temperature_c=55.0,
            flow_rate_m3h=250.0,
            capacity=AssetCapacity(
                thermal_capacity_mw=100.0,
                min_output_mw=20.0,
                max_output_mw=100.0,
                ramp_up_rate_mw_min=2.0,
                ramp_down_rate_mw_min=3.0,
            ),
            efficiency=AssetEfficiency(
                thermal_efficiency=0.88,
                electrical_efficiency=0.42,
            ),
            emissions=AssetEmissions(
                co2_kg_per_mwh=180.0,
                nox_kg_per_mwh=0.5,
                so2_kg_per_mwh=0.1,
                particulate_kg_per_mwh=0.02,
            ),
            cost=AssetCost(
                fuel_cost_per_mwh=35.0,
                variable_om_per_mwh=2.5,
                fixed_om_per_day=500.0,
                startup_cost=1500.0,
                shutdown_cost=500.0,
            ),
            health=AssetHealth(
                health_score=92.5,
                operating_hours_since_maintenance=720.0,
                fault_indicators=[],
            ),
        )

    def test_valid_asset_state(self, valid_asset_state):
        """Test creating valid asset state."""
        assert valid_asset_state.asset_name == "CHP Unit 1"
        assert valid_asset_state.asset_type == AssetType.CHP
        assert valid_asset_state.status == AssetStatus.ONLINE
        assert valid_asset_state.asset_id is not None

    def test_asset_state_json_serialization(self, valid_asset_state):
        """Test JSON serialization of asset state."""
        json_data = valid_asset_state.json()
        assert "CHP Unit 1" in json_data
        assert "chp" in json_data

    def test_asset_state_with_storage(self):
        """Test asset state with storage fields."""
        asset = AssetState(
            asset_name="Heat Storage",
            asset_type=AssetType.HEAT_STORAGE,
            status=AssetStatus.ONLINE,
            current_output_mw=0.0,
            current_setpoint_mw=0.0,
            supply_temperature_c=90.0,
            return_temperature_c=50.0,
            flow_rate_m3h=0.0,
            storage_level_mwh=500.0,
            storage_capacity_mwh=1000.0,
            capacity=AssetCapacity(
                thermal_capacity_mw=50.0,
                min_output_mw=0.0,
                max_output_mw=50.0,
                ramp_up_rate_mw_min=10.0,
                ramp_down_rate_mw_min=10.0,
            ),
            efficiency=AssetEfficiency(thermal_efficiency=0.95),
            emissions=AssetEmissions(co2_kg_per_mwh=0.0),
            cost=AssetCost(fuel_cost_per_mwh=0.0),
            health=AssetHealth(health_score=98.0, fault_indicators=[]),
        )
        assert asset.storage_level_mwh == 500.0
        assert asset.storage_capacity_mwh == 1000.0


# =============================================================================
# Constraint Model Tests
# =============================================================================

class TestConstraint:
    """Tests for Constraint model."""

    def test_valid_constraint(self):
        """Test creating valid constraint."""
        constraint = Constraint(
            name="Max Supply Temperature",
            constraint_type=ConstraintType.TEMPERATURE_MAX,
            priority=ConstraintPriority.CRITICAL,
            max_value=120.0,
            tolerance=2.0,
            effective_from=datetime.utcnow(),
            is_active=True,
            is_violated=False,
            violation_count=0,
        )
        assert constraint.name == "Max Supply Temperature"
        assert constraint.priority == ConstraintPriority.CRITICAL

    def test_constraint_with_time_window(self):
        """Test constraint with time-of-day window."""
        constraint = Constraint(
            name="Peak Hour Limit",
            constraint_type=ConstraintType.CAPACITY_MAX,
            priority=ConstraintPriority.HIGH,
            max_value=80.0,
            effective_from=datetime.utcnow(),
            time_of_day_start="17:00",
            time_of_day_end="21:00",
        )
        assert constraint.time_of_day_start == "17:00"
        assert constraint.time_of_day_end == "21:00"


# =============================================================================
# KPI Model Tests
# =============================================================================

class TestKPI:
    """Tests for KPI model."""

    def test_valid_kpi(self):
        """Test creating valid KPI."""
        kpi = KPI(
            name="System Efficiency",
            category="efficiency",
            current_value=92.5,
            target_value=95.0,
            unit="%",
            measurement_timestamp=datetime.utcnow(),
            aggregation_period="hourly",
        )
        assert kpi.current_value == 92.5
        assert kpi.category == "efficiency"

    def test_kpi_with_trend(self):
        """Test KPI with trend analysis."""
        kpi = KPI(
            name="Total Cost",
            category="cost",
            current_value=15000.0,
            target_value=14000.0,
            unit="EUR",
            measurement_timestamp=datetime.utcnow(),
            previous_value=15500.0,
            trend_direction="down",
            percent_change=-3.2,
            target_achievement_percent=93.0,
            is_on_target=False,
        )
        assert kpi.trend_direction == "down"
        assert kpi.percent_change == -3.2


# =============================================================================
# Request/Response Model Tests
# =============================================================================

class TestDemandUpdate:
    """Tests for DemandUpdate model."""

    def test_valid_demand_update(self):
        """Test creating valid demand update."""
        now = datetime.utcnow()
        timestamps = [now + timedelta(minutes=15 * i) for i in range(4)]
        demand = DemandUpdate(
            demand_mw=[50.0, 55.0, 60.0, 58.0],
            demand_timestamps=timestamps,
            source_system="SCADA",
        )
        assert len(demand.demand_mw) == 4
        assert demand.source_system == "SCADA"

    def test_demand_update_length_mismatch(self):
        """Test that demand and timestamps must have same length."""
        now = datetime.utcnow()
        with pytest.raises(ValueError, match="same length"):
            DemandUpdate(
                demand_mw=[50.0, 55.0, 60.0],
                demand_timestamps=[now],  # Only 1 timestamp for 3 values
                source_system="SCADA",
            )


class TestAllocationRequest:
    """Tests for AllocationRequest model."""

    def test_valid_allocation_request(self):
        """Test creating valid allocation request."""
        request = AllocationRequest(
            target_output_mw=100.0,
            time_window_minutes=15,
            objective=OptimizationObjective.BALANCE_COST_EMISSIONS,
            cost_weight=0.6,
            emissions_weight=0.4,
        )
        assert request.target_output_mw == 100.0
        assert request.cost_weight == 0.6

    def test_weights_must_sum_to_one(self):
        """Test that cost and emissions weights must sum to 1."""
        with pytest.raises(ValueError, match="sum to 1"):
            AllocationRequest(
                target_output_mw=100.0,
                cost_weight=0.6,
                emissions_weight=0.6,  # Invalid: 0.6 + 0.6 = 1.2
            )


# =============================================================================
# Alarm Model Tests
# =============================================================================

class TestAlarmEvent:
    """Tests for AlarmEvent model."""

    def test_valid_alarm_event(self):
        """Test creating valid alarm event."""
        alarm = AlarmEvent(
            alarm_code="TEMP_HIGH_001",
            name="High Supply Temperature",
            description="Supply temperature exceeded threshold",
            severity=AlarmSeverity.HIGH,
            status=AlarmStatus.ACTIVE,
            subsystem="thermal",
            triggered_at=datetime.utcnow(),
            measured_value=125.0,
            threshold_value=120.0,
            unit="C",
            recommended_actions=["Reduce output", "Check valve"],
        )
        assert alarm.alarm_code == "TEMP_HIGH_001"
        assert alarm.severity == AlarmSeverity.HIGH

    def test_alarm_acknowledgement(self):
        """Test alarm with acknowledgement."""
        now = datetime.utcnow()
        alarm = AlarmEvent(
            alarm_code="PRES_LOW_001",
            name="Low Pressure Warning",
            description="System pressure below threshold",
            severity=AlarmSeverity.MEDIUM,
            status=AlarmStatus.ACKNOWLEDGED,
            subsystem="hydraulic",
            triggered_at=now - timedelta(minutes=10),
            acknowledged_at=now,
            acknowledged_by="operator1",
        )
        assert alarm.status == AlarmStatus.ACKNOWLEDGED
        assert alarm.acknowledged_by == "operator1"


# =============================================================================
# Maintenance Trigger Tests
# =============================================================================

class TestMaintenanceTrigger:
    """Tests for MaintenanceTrigger model."""

    def test_valid_maintenance_trigger(self):
        """Test creating valid maintenance trigger."""
        trigger = MaintenanceTrigger(
            asset_id=uuid4(),
            asset_name="CHP Unit 1",
            maintenance_type=MaintenanceType.PREDICTIVE,
            urgency=MaintenanceUrgency.SCHEDULED,
            trigger_reason="Operating hours threshold exceeded",
            trigger_metric="operating_hours",
            current_value=5000.0,
            threshold_value=4500.0,
            recommended_action="Inspect heat exchanger",
            estimated_duration_hours=8.0,
            recommended_start_date=datetime.utcnow() + timedelta(days=7),
            production_impact_mw=50.0,
            downtime_hours=8.0,
        )
        assert trigger.maintenance_type == MaintenanceType.PREDICTIVE
        assert trigger.urgency == MaintenanceUrgency.SCHEDULED


# =============================================================================
# Explainability Model Tests
# =============================================================================

class TestExplainabilitySummary:
    """Tests for ExplainabilitySummary model."""

    def test_valid_explainability_summary(self):
        """Test creating valid explainability summary."""
        summary = ExplainabilitySummary(
            plan_id=uuid4(),
            executive_summary="Dispatch optimized for cost-emissions balance",
            key_drivers=["Low gas prices", "High demand forecast"],
            global_feature_importance=[
                FeatureImportance(
                    feature_name="gas_price",
                    importance_score=0.35,
                    direction="negative",
                    description="Lower gas prices favor CHP",
                ),
            ],
            lime_explanations=[
                DecisionExplanation(
                    decision_type="setpoint",
                    decision_description="CHP Unit 1 setpoint increased",
                    primary_factors=[
                        FeatureImportance(
                            feature_name="efficiency",
                            importance_score=0.45,
                            direction="positive",
                        ),
                    ],
                    explanation_confidence=0.92,
                ),
            ],
            plain_english_summary="The optimizer increased CHP output because gas prices are low.",
        )
        assert len(summary.key_drivers) == 2
        assert summary.global_feature_importance[0].importance_score == 0.35


# =============================================================================
# Pagination and Filter Tests
# =============================================================================

class TestPaginationParams:
    """Tests for PaginationParams model."""

    def test_valid_pagination(self):
        """Test creating valid pagination params."""
        params = PaginationParams(
            page=1,
            page_size=20,
            sort_by="created_at",
            sort_order="desc",
        )
        assert params.page == 1
        assert params.page_size == 20

    def test_pagination_bounds(self):
        """Test pagination bounds validation."""
        with pytest.raises(ValueError):
            PaginationParams(page=0)  # Invalid: page must be >= 1

        with pytest.raises(ValueError):
            PaginationParams(page_size=200)  # Invalid: max 100


class TestTimeRangeFilter:
    """Tests for TimeRangeFilter model."""

    def test_valid_time_range(self):
        """Test creating valid time range filter."""
        now = datetime.utcnow()
        time_range = TimeRangeFilter(
            start_time=now - timedelta(hours=24),
            end_time=now,
        )
        assert time_range.start_time < time_range.end_time

    def test_end_must_be_after_start(self):
        """Test that end_time must be after start_time."""
        now = datetime.utcnow()
        with pytest.raises(ValueError, match="end_time must be after start_time"):
            TimeRangeFilter(
                start_time=now,
                end_time=now - timedelta(hours=1),  # Invalid: before start
            )


# =============================================================================
# Dispatch Plan Tests
# =============================================================================

class TestDispatchPlan:
    """Tests for DispatchPlan model."""

    def test_valid_dispatch_plan(self):
        """Test creating valid dispatch plan."""
        now = datetime.utcnow()
        plan = DispatchPlan(
            plan_version=1,
            plan_name="Day-Ahead Dispatch",
            objective=OptimizationObjective.BALANCE_COST_EMISSIONS,
            planning_horizon_hours=24,
            resolution_minutes=15,
            effective_from=now,
            effective_until=now + timedelta(hours=24),
            is_active=True,
            schedule=[],
            setpoint_recommendations=[
                SetpointRecommendation(
                    asset_id=uuid4(),
                    asset_name="CHP Unit 1",
                    current_setpoint_mw=50.0,
                    recommended_setpoint_mw=55.0,
                    confidence=0.92,
                    reason="Higher efficiency at 55 MW",
                ),
            ],
            total_thermal_output_mwh=1200.0,
            total_cost=15000.0,
            total_emissions_kg=5400.0,
            average_efficiency=0.89,
            constraints_satisfied=12,
            constraints_violated=0,
            optimization_score=95.5,
            solver_status="optimal",
            computation_time_seconds=2.3,
        )
        assert plan.plan_name == "Day-Ahead Dispatch"
        assert len(plan.setpoint_recommendations) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
