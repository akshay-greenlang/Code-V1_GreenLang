"""
Tests for GL-001 ThermalCommand gRPC Services

Unit tests for gRPC service implementation.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import Mock, AsyncMock, MagicMock

from api.grpc_services import (
    ThermalCommandServicer,
    GRPCDataStore,
    ProtoAssetType,
    ProtoAssetStatus,
    ProtoOptimizationObjective,
    ProtoForecastType,
    ProtoConstraintType,
    ProtoConstraintPriority,
    ProtoAlarmSeverity,
    asset_type_to_proto,
    proto_to_asset_type,
    asset_status_to_proto,
    forecast_type_to_proto,
    proto_to_forecast_type,
    proto_to_optimization_objective,
    datetime_to_timestamp,
)
from api.api_schemas import (
    AssetType,
    AssetStatus,
    ForecastType,
    OptimizationObjective,
)


# =============================================================================
# Type Conversion Tests
# =============================================================================

class TestTypeConversions:
    """Tests for type conversion utilities."""

    def test_asset_type_to_proto(self):
        """Test converting AssetType to proto enum."""
        assert asset_type_to_proto(AssetType.CHP) == ProtoAssetType.ASSET_TYPE_CHP
        assert asset_type_to_proto(AssetType.BOILER) == ProtoAssetType.ASSET_TYPE_BOILER
        assert asset_type_to_proto(AssetType.HEAT_PUMP) == ProtoAssetType.ASSET_TYPE_HEAT_PUMP
        assert asset_type_to_proto(AssetType.HEAT_STORAGE) == ProtoAssetType.ASSET_TYPE_HEAT_STORAGE

    def test_proto_to_asset_type(self):
        """Test converting proto enum to AssetType."""
        assert proto_to_asset_type(ProtoAssetType.ASSET_TYPE_CHP) == AssetType.CHP
        assert proto_to_asset_type(ProtoAssetType.ASSET_TYPE_BOILER) == AssetType.BOILER
        assert proto_to_asset_type(ProtoAssetType.ASSET_TYPE_HEAT_PUMP) == AssetType.HEAT_PUMP

    def test_asset_status_to_proto(self):
        """Test converting AssetStatus to proto enum."""
        assert asset_status_to_proto(AssetStatus.ONLINE) == ProtoAssetStatus.ASSET_STATUS_ONLINE
        assert asset_status_to_proto(AssetStatus.OFFLINE) == ProtoAssetStatus.ASSET_STATUS_OFFLINE
        assert asset_status_to_proto(AssetStatus.MAINTENANCE) == ProtoAssetStatus.ASSET_STATUS_MAINTENANCE

    def test_forecast_type_to_proto(self):
        """Test converting ForecastType to proto enum."""
        assert forecast_type_to_proto(ForecastType.DEMAND) == ProtoForecastType.FORECAST_TYPE_DEMAND
        assert forecast_type_to_proto(ForecastType.TEMPERATURE) == ProtoForecastType.FORECAST_TYPE_TEMPERATURE
        assert forecast_type_to_proto(ForecastType.ELECTRICITY_PRICE) == ProtoForecastType.FORECAST_TYPE_ELECTRICITY_PRICE

    def test_proto_to_forecast_type(self):
        """Test converting proto enum to ForecastType."""
        assert proto_to_forecast_type(ProtoForecastType.FORECAST_TYPE_DEMAND) == ForecastType.DEMAND
        assert proto_to_forecast_type(ProtoForecastType.FORECAST_TYPE_TEMPERATURE) == ForecastType.TEMPERATURE

    def test_proto_to_optimization_objective(self):
        """Test converting proto enum to OptimizationObjective."""
        assert proto_to_optimization_objective(
            ProtoOptimizationObjective.OPTIMIZATION_OBJECTIVE_MINIMIZE_COST
        ) == OptimizationObjective.MINIMIZE_COST
        assert proto_to_optimization_objective(
            ProtoOptimizationObjective.OPTIMIZATION_OBJECTIVE_BALANCE_COST_EMISSIONS
        ) == OptimizationObjective.BALANCE_COST_EMISSIONS

    def test_datetime_to_timestamp(self):
        """Test converting datetime to protobuf Timestamp."""
        dt = datetime(2025, 1, 15, 10, 30, 0)
        ts = datetime_to_timestamp(dt)
        assert ts is not None


# =============================================================================
# Data Store Tests
# =============================================================================

class TestGRPCDataStore:
    """Tests for gRPC data store."""

    @pytest.fixture
    def data_store(self):
        """Create fresh data store for testing."""
        return GRPCDataStore()

    @pytest.mark.asyncio
    async def test_get_current_plan(self, data_store):
        """Test getting current dispatch plan."""
        plan = await data_store.get_current_plan()
        assert plan is not None
        assert plan.plan_name == "Day-Ahead Dispatch Plan"
        assert plan.is_active is True

    @pytest.mark.asyncio
    async def test_get_asset_states(self, data_store):
        """Test getting asset states."""
        assets = await data_store.get_asset_states()
        assert len(assets) >= 1
        asset = assets[0]
        assert asset.asset_name == "CHP Unit 1"
        assert asset.asset_type == AssetType.CHP

    @pytest.mark.asyncio
    async def test_get_asset_states_with_filter(self, data_store):
        """Test getting asset states with filter."""
        assets = await data_store.get_asset_states(
            asset_ids=[str(asset.asset_id) for asset in (await data_store.get_asset_states())]
        )
        assert len(assets) >= 1

    @pytest.mark.asyncio
    async def test_get_constraints(self, data_store):
        """Test getting constraints."""
        constraints = await data_store.get_constraints()
        # May be empty if no constraints initialized
        assert isinstance(constraints, list)

    @pytest.mark.asyncio
    async def test_get_kpis(self, data_store):
        """Test getting KPIs."""
        kpis = await data_store.get_kpis()
        assert len(kpis) >= 1
        kpi = kpis[0]
        assert kpi.name == "System Efficiency"

    @pytest.mark.asyncio
    async def test_get_kpis_by_category(self, data_store):
        """Test getting KPIs filtered by category."""
        kpis = await data_store.get_kpis(category="efficiency")
        for kpi in kpis:
            assert kpi.category == "efficiency"

    @pytest.mark.asyncio
    async def test_get_latest_forecast(self, data_store):
        """Test getting latest forecast."""
        forecast = await data_store.get_latest_forecast(ForecastType.DEMAND)
        assert forecast is not None
        assert forecast.forecast_type == ForecastType.DEMAND
        assert forecast.unit == "MW"
        assert len(forecast.values) > 0

    @pytest.mark.asyncio
    async def test_request_allocation(self, data_store):
        """Test requesting allocation."""
        response = await data_store.request_allocation(
            target_output_mw=80.0,
            time_window_minutes=15,
            objective=OptimizationObjective.BALANCE_COST_EMISSIONS,
            cost_weight=0.5,
            emissions_weight=0.5,
            excluded_assets=[],
            must_run_assets=[],
            is_emergency=False,
        )
        assert response.success is True
        assert response.allocated_output_mw > 0
        assert len(response.asset_allocations) > 0

    @pytest.mark.asyncio
    async def test_submit_demand_update(self, data_store):
        """Test submitting demand update."""
        now = datetime.utcnow()
        timestamps = [now + timedelta(minutes=15 * i) for i in range(4)]

        request_id, records_received, records_validated, quality_score = \
            await data_store.submit_demand_update(
                forecast_type=ForecastType.DEMAND,
                forecast_horizon_hours=24,
                resolution_minutes=15,
                demand_mw=[50.0, 55.0, 60.0, 58.0],
                demand_timestamps=timestamps,
                source_system="SCADA",
                model_version="1.0",
            )

        assert request_id is not None
        assert records_received == 4
        assert records_validated == 4
        assert quality_score > 0

    @pytest.mark.asyncio
    async def test_acknowledge_alarm(self, data_store):
        """Test acknowledging alarm."""
        success, message, acknowledged_at = await data_store.acknowledge_alarm(
            alarm_id=str(uuid4()),
            acknowledged_by="operator1",
            acknowledgement_note="Investigating",
        )
        # May return False if alarm not found
        assert message is not None
        assert acknowledged_at is not None


# =============================================================================
# Service Tests
# =============================================================================

class TestThermalCommandServicer:
    """Tests for ThermalCommandServicer."""

    @pytest.fixture
    def servicer(self):
        """Create servicer with fresh data store."""
        data_store = GRPCDataStore()
        return ThermalCommandServicer(data_store)

    @pytest.fixture
    def mock_context(self):
        """Create mock gRPC context."""
        context = AsyncMock()
        context.cancelled = Mock(return_value=False)
        context.set_code = Mock()
        context.set_details = Mock()
        return context

    @pytest.mark.asyncio
    async def test_request_allocation(self, servicer, mock_context):
        """Test RequestAllocation RPC."""
        request = Mock()
        request.target_output_mw = 80.0
        request.time_window_minutes = 15
        request.objective = ProtoOptimizationObjective.OPTIMIZATION_OBJECTIVE_BALANCE_COST_EMISSIONS
        request.cost_weight = 0.5
        request.emissions_weight = 0.5
        request.excluded_assets = []
        request.must_run_assets = []
        request.is_emergency = False

        response = await servicer.RequestAllocation(request, mock_context)

        assert response["success"] is True
        assert response["allocated_output_mw"] > 0
        assert len(response["asset_allocations"]) > 0

    @pytest.mark.asyncio
    async def test_submit_demand_update(self, servicer, mock_context):
        """Test SubmitDemandUpdate RPC."""
        now = datetime.utcnow()
        from google.protobuf import timestamp_pb2

        request = Mock()
        request.forecast_type = ProtoForecastType.FORECAST_TYPE_DEMAND
        request.forecast_horizon_hours = 24
        request.resolution_minutes = 15
        request.demand_mw = [50.0, 55.0, 60.0, 58.0]

        # Create mock timestamps
        timestamps = []
        for i in range(4):
            ts = timestamp_pb2.Timestamp()
            ts.FromDatetime(now + timedelta(minutes=15 * i))
            timestamps.append(ts)
        request.demand_timestamps = timestamps
        request.source_system = "SCADA"
        request.HasField = Mock(return_value=False)

        response = await servicer.SubmitDemandUpdate(request, mock_context)

        assert response["success"] is True
        assert response["records_received"] == 4

    @pytest.mark.asyncio
    async def test_get_latest_forecast(self, servicer, mock_context):
        """Test GetLatestForecast RPC."""
        request = Mock()
        request.forecast_type = ProtoForecastType.FORECAST_TYPE_DEMAND

        response = await servicer.GetLatestForecast(request, mock_context)

        assert response["success"] is True
        assert response["forecast"] is not None
        assert len(response["forecast"]["values"]) > 0

    @pytest.mark.asyncio
    async def test_get_current_plan(self, servicer, mock_context):
        """Test GetCurrentPlan RPC."""
        request = Mock()

        response = await servicer.GetCurrentPlan(request, mock_context)

        assert response["success"] is True
        assert response["plan"] is not None
        assert response["plan"]["plan_name"] == "Day-Ahead Dispatch Plan"

    @pytest.mark.asyncio
    async def test_get_asset_states(self, servicer, mock_context):
        """Test GetAssetStates RPC."""
        request = Mock()
        request.asset_ids = []
        request.asset_types = []
        request.statuses = []
        request.page = 1
        request.page_size = 20

        response = await servicer.GetAssetStates(request, mock_context)

        assert response["success"] is True
        assert len(response["assets"]) >= 1
        assert response["assets"][0]["asset_name"] == "CHP Unit 1"

    @pytest.mark.asyncio
    async def test_get_constraints(self, servicer, mock_context):
        """Test GetConstraints RPC."""
        request = Mock()
        request.HasField = Mock(return_value=False)
        request.constraint_types = []
        request.priorities = []

        response = await servicer.GetConstraints(request, mock_context)

        assert response["success"] is True

    @pytest.mark.asyncio
    async def test_get_kpis(self, servicer, mock_context):
        """Test GetKPIs RPC."""
        request = Mock()
        request.HasField = Mock(return_value=False)

        response = await servicer.GetKPIs(request, mock_context)

        assert response["success"] is True
        assert len(response["kpis"]) >= 1

    @pytest.mark.asyncio
    async def test_acknowledge_alarm(self, servicer, mock_context):
        """Test AcknowledgeAlarm RPC."""
        request = Mock()
        request.alarm_id = str(uuid4())
        request.acknowledged_by = "operator1"
        request.HasField = Mock(return_value=False)

        response = await servicer.AcknowledgeAlarm(request, mock_context)

        assert response["alarm_id"] == request.alarm_id


# =============================================================================
# Proto Enum Tests
# =============================================================================

class TestProtoEnums:
    """Tests for proto enum values."""

    def test_asset_type_values(self):
        """Test asset type enum values."""
        assert ProtoAssetType.ASSET_TYPE_UNSPECIFIED == 0
        assert ProtoAssetType.ASSET_TYPE_CHP == 1
        assert ProtoAssetType.ASSET_TYPE_BOILER == 2
        assert ProtoAssetType.ASSET_TYPE_HEAT_PUMP == 3

    def test_asset_status_values(self):
        """Test asset status enum values."""
        assert ProtoAssetStatus.ASSET_STATUS_UNSPECIFIED == 0
        assert ProtoAssetStatus.ASSET_STATUS_ONLINE == 1
        assert ProtoAssetStatus.ASSET_STATUS_OFFLINE == 2
        assert ProtoAssetStatus.ASSET_STATUS_MAINTENANCE == 4

    def test_optimization_objective_values(self):
        """Test optimization objective enum values."""
        assert ProtoOptimizationObjective.OPTIMIZATION_OBJECTIVE_UNSPECIFIED == 0
        assert ProtoOptimizationObjective.OPTIMIZATION_OBJECTIVE_MINIMIZE_COST == 1
        assert ProtoOptimizationObjective.OPTIMIZATION_OBJECTIVE_MINIMIZE_EMISSIONS == 2
        assert ProtoOptimizationObjective.OPTIMIZATION_OBJECTIVE_BALANCE_COST_EMISSIONS == 4

    def test_forecast_type_values(self):
        """Test forecast type enum values."""
        assert ProtoForecastType.FORECAST_TYPE_UNSPECIFIED == 0
        assert ProtoForecastType.FORECAST_TYPE_DEMAND == 1
        assert ProtoForecastType.FORECAST_TYPE_TEMPERATURE == 2
        assert ProtoForecastType.FORECAST_TYPE_ELECTRICITY_PRICE == 3

    def test_constraint_type_values(self):
        """Test constraint type enum values."""
        assert ProtoConstraintType.CONSTRAINT_TYPE_UNSPECIFIED == 0
        assert ProtoConstraintType.CONSTRAINT_TYPE_CAPACITY_MIN == 1
        assert ProtoConstraintType.CONSTRAINT_TYPE_CAPACITY_MAX == 2
        assert ProtoConstraintType.CONSTRAINT_TYPE_TEMPERATURE_MAX == 5

    def test_constraint_priority_values(self):
        """Test constraint priority enum values."""
        assert ProtoConstraintPriority.CONSTRAINT_PRIORITY_UNSPECIFIED == 0
        assert ProtoConstraintPriority.CONSTRAINT_PRIORITY_CRITICAL == 1
        assert ProtoConstraintPriority.CONSTRAINT_PRIORITY_HIGH == 2

    def test_alarm_severity_values(self):
        """Test alarm severity enum values."""
        assert ProtoAlarmSeverity.ALARM_SEVERITY_UNSPECIFIED == 0
        assert ProtoAlarmSeverity.ALARM_SEVERITY_CRITICAL == 1
        assert ProtoAlarmSeverity.ALARM_SEVERITY_HIGH == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
