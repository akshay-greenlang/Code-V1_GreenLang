"""
GL-001 ThermalCommand gRPC Services

gRPC service implementation for low-latency RPC calls in district heating optimization.
Provides real-time dispatch operations, demand updates, and forecast retrieval.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent import futures
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import grpc
from google.protobuf import timestamp_pb2, empty_pb2

from .api_schemas import (
    AllocationRequest as AllocationRequestSchema,
    AllocationResponse as AllocationResponseSchema,
    AlarmEvent as AlarmEventSchema,
    AlarmSeverity,
    AlarmStatus,
    AssetCapacity as AssetCapacitySchema,
    AssetCost as AssetCostSchema,
    AssetEfficiency as AssetEfficiencySchema,
    AssetEmissions as AssetEmissionsSchema,
    AssetHealth as AssetHealthSchema,
    AssetState as AssetStateSchema,
    AssetStatus as AssetStatusEnum,
    AssetType as AssetTypeEnum,
    Constraint as ConstraintSchema,
    ConstraintPriority,
    ConstraintType,
    DispatchPlan as DispatchPlanSchema,
    ForecastData as ForecastDataSchema,
    ForecastType,
    KPI as KPISchema,
    OptimizationObjective,
    SetpointRecommendation as SetpointRecommendationSchema,
)
from .api_auth import (
    Permission,
    ThermalCommandUser,
    get_auth_config,
    verify_token,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Generated Proto Stubs (Mock Implementation)
# =============================================================================
# NOTE: In production, these would be generated from thermal_command.proto
# using: python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. thermal_command.proto

# Mock proto message classes
class ProtoMessage:
    """Base class for mock proto messages."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def SerializeToString(self) -> bytes:
        """Serialize to string (mock)."""
        import json
        return json.dumps(self.__dict__, default=str).encode()

    @classmethod
    def FromString(cls, data: bytes) -> "ProtoMessage":
        """Deserialize from string (mock)."""
        import json
        obj = cls()
        obj.__dict__.update(json.loads(data.decode()))
        return obj


# Enum classes for proto
class ProtoAssetType:
    ASSET_TYPE_UNSPECIFIED = 0
    ASSET_TYPE_CHP = 1
    ASSET_TYPE_BOILER = 2
    ASSET_TYPE_HEAT_PUMP = 3
    ASSET_TYPE_HEAT_STORAGE = 4
    ASSET_TYPE_SOLAR_THERMAL = 5
    ASSET_TYPE_WASTE_HEAT = 6
    ASSET_TYPE_ELECTRIC_HEATER = 7


class ProtoAssetStatus:
    ASSET_STATUS_UNSPECIFIED = 0
    ASSET_STATUS_ONLINE = 1
    ASSET_STATUS_OFFLINE = 2
    ASSET_STATUS_STANDBY = 3
    ASSET_STATUS_MAINTENANCE = 4
    ASSET_STATUS_FAULT = 5
    ASSET_STATUS_RAMPING_UP = 6
    ASSET_STATUS_RAMPING_DOWN = 7


class ProtoConstraintType:
    CONSTRAINT_TYPE_UNSPECIFIED = 0
    CONSTRAINT_TYPE_CAPACITY_MIN = 1
    CONSTRAINT_TYPE_CAPACITY_MAX = 2
    CONSTRAINT_TYPE_RAMP_RATE = 3
    CONSTRAINT_TYPE_TEMPERATURE_MIN = 4
    CONSTRAINT_TYPE_TEMPERATURE_MAX = 5


class ProtoConstraintPriority:
    CONSTRAINT_PRIORITY_UNSPECIFIED = 0
    CONSTRAINT_PRIORITY_CRITICAL = 1
    CONSTRAINT_PRIORITY_HIGH = 2
    CONSTRAINT_PRIORITY_MEDIUM = 3
    CONSTRAINT_PRIORITY_LOW = 4


class ProtoOptimizationObjective:
    OPTIMIZATION_OBJECTIVE_UNSPECIFIED = 0
    OPTIMIZATION_OBJECTIVE_MINIMIZE_COST = 1
    OPTIMIZATION_OBJECTIVE_MINIMIZE_EMISSIONS = 2
    OPTIMIZATION_OBJECTIVE_MAXIMIZE_EFFICIENCY = 3
    OPTIMIZATION_OBJECTIVE_BALANCE_COST_EMISSIONS = 4


class ProtoForecastType:
    FORECAST_TYPE_UNSPECIFIED = 0
    FORECAST_TYPE_DEMAND = 1
    FORECAST_TYPE_TEMPERATURE = 2
    FORECAST_TYPE_ELECTRICITY_PRICE = 3
    FORECAST_TYPE_GAS_PRICE = 4
    FORECAST_TYPE_SOLAR_IRRADIANCE = 5


class ProtoAlarmSeverity:
    ALARM_SEVERITY_UNSPECIFIED = 0
    ALARM_SEVERITY_CRITICAL = 1
    ALARM_SEVERITY_HIGH = 2
    ALARM_SEVERITY_MEDIUM = 3
    ALARM_SEVERITY_LOW = 4
    ALARM_SEVERITY_INFO = 5


class ProtoAlarmStatus:
    ALARM_STATUS_UNSPECIFIED = 0
    ALARM_STATUS_ACTIVE = 1
    ALARM_STATUS_ACKNOWLEDGED = 2
    ALARM_STATUS_RESOLVED = 3
    ALARM_STATUS_SUPPRESSED = 4


# =============================================================================
# Type Conversion Utilities
# =============================================================================

def datetime_to_timestamp(dt: datetime) -> timestamp_pb2.Timestamp:
    """Convert datetime to protobuf Timestamp."""
    ts = timestamp_pb2.Timestamp()
    ts.FromDatetime(dt)
    return ts


def timestamp_to_datetime(ts: timestamp_pb2.Timestamp) -> datetime:
    """Convert protobuf Timestamp to datetime."""
    return ts.ToDatetime()


def asset_type_to_proto(asset_type: AssetTypeEnum) -> int:
    """Convert AssetType enum to proto enum value."""
    mapping = {
        AssetTypeEnum.CHP: ProtoAssetType.ASSET_TYPE_CHP,
        AssetTypeEnum.BOILER: ProtoAssetType.ASSET_TYPE_BOILER,
        AssetTypeEnum.HEAT_PUMP: ProtoAssetType.ASSET_TYPE_HEAT_PUMP,
        AssetTypeEnum.HEAT_STORAGE: ProtoAssetType.ASSET_TYPE_HEAT_STORAGE,
        AssetTypeEnum.SOLAR_THERMAL: ProtoAssetType.ASSET_TYPE_SOLAR_THERMAL,
        AssetTypeEnum.WASTE_HEAT: ProtoAssetType.ASSET_TYPE_WASTE_HEAT,
        AssetTypeEnum.ELECTRIC_HEATER: ProtoAssetType.ASSET_TYPE_ELECTRIC_HEATER,
    }
    return mapping.get(asset_type, ProtoAssetType.ASSET_TYPE_UNSPECIFIED)


def proto_to_asset_type(proto_type: int) -> AssetTypeEnum:
    """Convert proto enum value to AssetType enum."""
    mapping = {
        ProtoAssetType.ASSET_TYPE_CHP: AssetTypeEnum.CHP,
        ProtoAssetType.ASSET_TYPE_BOILER: AssetTypeEnum.BOILER,
        ProtoAssetType.ASSET_TYPE_HEAT_PUMP: AssetTypeEnum.HEAT_PUMP,
        ProtoAssetType.ASSET_TYPE_HEAT_STORAGE: AssetTypeEnum.HEAT_STORAGE,
        ProtoAssetType.ASSET_TYPE_SOLAR_THERMAL: AssetTypeEnum.SOLAR_THERMAL,
        ProtoAssetType.ASSET_TYPE_WASTE_HEAT: AssetTypeEnum.WASTE_HEAT,
        ProtoAssetType.ASSET_TYPE_ELECTRIC_HEATER: AssetTypeEnum.ELECTRIC_HEATER,
    }
    return mapping.get(proto_type, AssetTypeEnum.CHP)


def asset_status_to_proto(status: AssetStatusEnum) -> int:
    """Convert AssetStatus enum to proto enum value."""
    mapping = {
        AssetStatusEnum.ONLINE: ProtoAssetStatus.ASSET_STATUS_ONLINE,
        AssetStatusEnum.OFFLINE: ProtoAssetStatus.ASSET_STATUS_OFFLINE,
        AssetStatusEnum.STANDBY: ProtoAssetStatus.ASSET_STATUS_STANDBY,
        AssetStatusEnum.MAINTENANCE: ProtoAssetStatus.ASSET_STATUS_MAINTENANCE,
        AssetStatusEnum.FAULT: ProtoAssetStatus.ASSET_STATUS_FAULT,
        AssetStatusEnum.RAMPING_UP: ProtoAssetStatus.ASSET_STATUS_RAMPING_UP,
        AssetStatusEnum.RAMPING_DOWN: ProtoAssetStatus.ASSET_STATUS_RAMPING_DOWN,
    }
    return mapping.get(status, ProtoAssetStatus.ASSET_STATUS_UNSPECIFIED)


def forecast_type_to_proto(forecast_type: ForecastType) -> int:
    """Convert ForecastType enum to proto enum value."""
    mapping = {
        ForecastType.DEMAND: ProtoForecastType.FORECAST_TYPE_DEMAND,
        ForecastType.TEMPERATURE: ProtoForecastType.FORECAST_TYPE_TEMPERATURE,
        ForecastType.ELECTRICITY_PRICE: ProtoForecastType.FORECAST_TYPE_ELECTRICITY_PRICE,
        ForecastType.GAS_PRICE: ProtoForecastType.FORECAST_TYPE_GAS_PRICE,
        ForecastType.SOLAR_IRRADIANCE: ProtoForecastType.FORECAST_TYPE_SOLAR_IRRADIANCE,
    }
    return mapping.get(forecast_type, ProtoForecastType.FORECAST_TYPE_UNSPECIFIED)


def proto_to_forecast_type(proto_type: int) -> ForecastType:
    """Convert proto enum value to ForecastType enum."""
    mapping = {
        ProtoForecastType.FORECAST_TYPE_DEMAND: ForecastType.DEMAND,
        ProtoForecastType.FORECAST_TYPE_TEMPERATURE: ForecastType.TEMPERATURE,
        ProtoForecastType.FORECAST_TYPE_ELECTRICITY_PRICE: ForecastType.ELECTRICITY_PRICE,
        ProtoForecastType.FORECAST_TYPE_GAS_PRICE: ForecastType.GAS_PRICE,
        ProtoForecastType.FORECAST_TYPE_SOLAR_IRRADIANCE: ForecastType.SOLAR_IRRADIANCE,
    }
    return mapping.get(proto_type, ForecastType.DEMAND)


def proto_to_optimization_objective(proto_obj: int) -> OptimizationObjective:
    """Convert proto enum value to OptimizationObjective enum."""
    mapping = {
        ProtoOptimizationObjective.OPTIMIZATION_OBJECTIVE_MINIMIZE_COST: OptimizationObjective.MINIMIZE_COST,
        ProtoOptimizationObjective.OPTIMIZATION_OBJECTIVE_MINIMIZE_EMISSIONS: OptimizationObjective.MINIMIZE_EMISSIONS,
        ProtoOptimizationObjective.OPTIMIZATION_OBJECTIVE_MAXIMIZE_EFFICIENCY: OptimizationObjective.MAXIMIZE_EFFICIENCY,
        ProtoOptimizationObjective.OPTIMIZATION_OBJECTIVE_BALANCE_COST_EMISSIONS: OptimizationObjective.BALANCE_COST_EMISSIONS,
    }
    return mapping.get(proto_obj, OptimizationObjective.BALANCE_COST_EMISSIONS)


# =============================================================================
# gRPC Authentication Interceptor
# =============================================================================

class AuthInterceptor(grpc.ServerInterceptor):
    """
    gRPC server interceptor for authentication.
    Validates JWT tokens from metadata and adds user context.
    """

    def __init__(self):
        self.config = get_auth_config()

    def intercept_service(self, continuation, handler_call_details):
        """Intercept incoming RPC calls for authentication."""
        # Extract metadata
        metadata = dict(handler_call_details.invocation_metadata)

        # Check for authorization header
        auth_header = metadata.get("authorization", "")

        if not auth_header:
            # Check for x-api-key
            api_key = metadata.get("x-api-key", "")
            if not api_key:
                return self._abort_with_unauthenticated()

        # Validate token
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            try:
                token_data = verify_token(token, self.config)
                # Token is valid, continue with request
                return continuation(handler_call_details)
            except Exception as e:
                logger.warning(f"Token validation failed: {e}")
                return self._abort_with_unauthenticated()

        return continuation(handler_call_details)

    def _abort_with_unauthenticated(self):
        """Return unauthenticated error handler."""

        def abort_handler(request, context):
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid or missing authentication")

        return grpc.unary_unary_rpc_method_handler(abort_handler)


# =============================================================================
# Data Store (Shared with GraphQL)
# =============================================================================

class GRPCDataStore:
    """
    Data store for gRPC service.
    In production, this would share the same database connection pool as GraphQL.
    """

    def __init__(self):
        self._dispatch_plans: Dict[str, DispatchPlanSchema] = {}
        self._asset_states: Dict[str, AssetStateSchema] = {}
        self._constraints: Dict[str, ConstraintSchema] = {}
        self._kpis: Dict[str, KPISchema] = {}
        self._alarms: Dict[str, AlarmEventSchema] = {}
        self._forecasts: Dict[str, ForecastDataSchema] = {}

        # Event queues for streaming
        self._plan_update_queue: asyncio.Queue = asyncio.Queue()
        self._recommendation_queue: asyncio.Queue = asyncio.Queue()
        self._alarm_queue: asyncio.Queue = asyncio.Queue()

        # Initialize sample data
        self._initialize_sample_data()

    def _initialize_sample_data(self):
        """Initialize with sample data."""
        now = datetime.utcnow()

        # Sample asset
        asset_id = str(uuid4())
        self._asset_states[asset_id] = AssetStateSchema(
            asset_id=UUID(asset_id),
            asset_name="CHP Unit 1",
            asset_type=AssetTypeEnum.CHP,
            status=AssetStatusEnum.ONLINE,
            current_output_mw=45.5,
            current_setpoint_mw=50.0,
            supply_temperature_c=95.0,
            return_temperature_c=55.0,
            flow_rate_m3h=250.0,
            capacity=AssetCapacitySchema(
                thermal_capacity_mw=100.0,
                min_output_mw=20.0,
                max_output_mw=100.0,
                ramp_up_rate_mw_min=2.0,
                ramp_down_rate_mw_min=3.0,
                min_uptime_hours=4.0,
                min_downtime_hours=2.0,
                startup_time_minutes=30.0,
            ),
            efficiency=AssetEfficiencySchema(
                thermal_efficiency=0.88,
                electrical_efficiency=0.42,
            ),
            emissions=AssetEmissionsSchema(
                co2_kg_per_mwh=180.0,
                nox_kg_per_mwh=0.5,
                so2_kg_per_mwh=0.1,
                particulate_kg_per_mwh=0.02,
            ),
            cost=AssetCostSchema(
                fuel_cost_per_mwh=35.0,
                variable_om_per_mwh=2.5,
                fixed_om_per_day=500.0,
                startup_cost=1500.0,
                shutdown_cost=500.0,
            ),
            health=AssetHealthSchema(
                health_score=92.5,
                remaining_useful_life_hours=5000.0,
                last_maintenance_date=datetime(2025, 1, 1),
                operating_hours_since_maintenance=720.0,
                fault_indicators=[],
            ),
            created_at=now,
        )

        # Sample dispatch plan
        plan_id = str(uuid4())
        self._dispatch_plans[plan_id] = DispatchPlanSchema(
            plan_id=UUID(plan_id),
            plan_version=1,
            plan_name="Day-Ahead Dispatch Plan",
            description="Optimized dispatch for next 24 hours",
            objective=OptimizationObjective.BALANCE_COST_EMISSIONS,
            planning_horizon_hours=24,
            resolution_minutes=15,
            effective_from=now,
            effective_until=datetime(now.year, now.month, now.day + 1),
            is_active=True,
            schedule=[],
            setpoint_recommendations=[
                SetpointRecommendationSchema(
                    asset_id=UUID(asset_id),
                    asset_name="CHP Unit 1",
                    current_setpoint_mw=50.0,
                    recommended_setpoint_mw=55.0,
                    confidence=0.92,
                    reason="Increased demand forecast",
                ),
            ],
            total_thermal_output_mwh=1200.0,
            total_cost=15000.0,
            total_emissions_kg=5400.0,
            average_efficiency=0.89,
            constraints_satisfied=12,
            constraints_violated=0,
            violated_constraint_ids=[],
            optimization_score=95.5,
            solver_status="optimal",
            computation_time_seconds=2.3,
            created_at=now,
        )

        # Sample KPI
        kpi_id = str(uuid4())
        self._kpis[kpi_id] = KPISchema(
            kpi_id=UUID(kpi_id),
            name="System Efficiency",
            category="efficiency",
            current_value=92.5,
            target_value=95.0,
            unit="%",
            measurement_timestamp=now,
            aggregation_period="hourly",
            created_at=now,
        )

    async def get_current_plan(self) -> Optional[DispatchPlanSchema]:
        """Get the currently active dispatch plan."""
        for plan in self._dispatch_plans.values():
            if plan.is_active:
                return plan
        return None

    async def get_asset_states(
        self,
        asset_ids: Optional[List[str]] = None,
        asset_types: Optional[List[int]] = None,
        statuses: Optional[List[int]] = None,
    ) -> List[AssetStateSchema]:
        """Get asset states with optional filtering."""
        assets = list(self._asset_states.values())

        if asset_ids:
            assets = [a for a in assets if str(a.asset_id) in asset_ids]

        return assets

    async def get_constraints(
        self,
        is_active: Optional[bool] = None,
        constraint_types: Optional[List[int]] = None,
        priorities: Optional[List[int]] = None,
    ) -> List[ConstraintSchema]:
        """Get constraints with optional filtering."""
        constraints = list(self._constraints.values())

        if is_active is not None:
            constraints = [c for c in constraints if c.is_active == is_active]

        return constraints

    async def get_kpis(
        self,
        category: Optional[str] = None,
    ) -> List[KPISchema]:
        """Get KPIs with optional filtering."""
        kpis = list(self._kpis.values())

        if category:
            kpis = [k for k in kpis if k.category == category]

        return kpis

    async def get_latest_forecast(
        self,
        forecast_type: ForecastType,
    ) -> Optional[ForecastDataSchema]:
        """Get the latest forecast of a specific type."""
        for forecast in self._forecasts.values():
            if forecast.forecast_type == forecast_type:
                return forecast

        # Generate mock forecast
        now = datetime.utcnow()
        forecast = ForecastDataSchema(
            forecast_id=uuid4(),
            forecast_type=forecast_type,
            forecast_horizon_hours=24,
            resolution_minutes=15,
            generated_at=now,
            valid_from=now,
            valid_until=datetime(now.year, now.month, now.day + 1),
            values=[50.0 + i * 0.5 for i in range(96)],
            timestamps=[datetime(now.year, now.month, now.day, i // 4, (i % 4) * 15) for i in range(96)],
            unit="MW" if forecast_type == ForecastType.DEMAND else "C",
            confidence_level=0.95,
            model_name="ThermalForecast-v2",
            model_version="2.1.0",
            created_at=now,
        )
        self._forecasts[str(forecast.forecast_id)] = forecast
        return forecast

    async def request_allocation(
        self,
        target_output_mw: float,
        time_window_minutes: int,
        objective: OptimizationObjective,
        cost_weight: float,
        emissions_weight: float,
        excluded_assets: List[str],
        must_run_assets: List[str],
        is_emergency: bool,
    ) -> AllocationResponseSchema:
        """Process allocation request."""
        request_id = str(uuid4())
        response_id = str(uuid4())

        # Mock allocation
        assets = await self.get_asset_states()
        allocations = []

        remaining_output = target_output_mw
        for asset in assets:
            if remaining_output <= 0:
                break

            if str(asset.asset_id) in excluded_assets:
                continue

            allocation = min(remaining_output, asset.capacity.max_output_mw)
            allocations.append(
                SetpointRecommendationSchema(
                    asset_id=asset.asset_id,
                    asset_name=asset.asset_name,
                    current_setpoint_mw=asset.current_setpoint_mw,
                    recommended_setpoint_mw=allocation,
                    confidence=0.95,
                    reason="Optimal allocation",
                )
            )
            remaining_output -= allocation

        total_allocated = target_output_mw - remaining_output

        return AllocationResponseSchema(
            request_id=UUID(request_id),
            response_id=UUID(response_id),
            success=True,
            status_message="Allocation optimized successfully",
            allocated_output_mw=total_allocated,
            allocation_gap_mw=remaining_output,
            asset_allocations=allocations,
            estimated_cost=total_allocated * 38,
            estimated_emissions_kg=total_allocated * 175,
            optimization_time_ms=125.5,
            solver_iterations=42,
            created_at=datetime.utcnow(),
        )

    async def submit_demand_update(
        self,
        forecast_type: ForecastType,
        forecast_horizon_hours: int,
        resolution_minutes: int,
        demand_mw: List[float],
        demand_timestamps: List[datetime],
        source_system: str,
        model_version: Optional[str],
    ) -> Tuple[str, int, int, float]:
        """Submit demand update."""
        request_id = str(uuid4())
        records_received = len(demand_mw)
        records_validated = records_received
        quality_score = 98.5

        logger.info(f"Demand update received via gRPC: {records_received} records")

        return request_id, records_received, records_validated, quality_score

    async def acknowledge_alarm(
        self,
        alarm_id: str,
        acknowledged_by: str,
        acknowledgement_note: Optional[str],
    ) -> Tuple[bool, str, datetime]:
        """Acknowledge an alarm."""
        now = datetime.utcnow()

        if alarm_id in self._alarms:
            alarm = self._alarms[alarm_id]
            alarm.status = AlarmStatus.ACKNOWLEDGED
            alarm.acknowledged_at = now
            alarm.acknowledged_by = acknowledged_by
            return True, "Alarm acknowledged successfully", now

        return False, "Alarm not found", now


# Global data store
grpc_data_store = GRPCDataStore()


# =============================================================================
# ThermalCommand gRPC Service Implementation
# =============================================================================

class ThermalCommandServicer:
    """
    gRPC servicer implementation for ThermalCommand.

    Provides low-latency RPC calls for district heating optimization operations.
    """

    def __init__(self, data_store: Optional[GRPCDataStore] = None):
        """Initialize the servicer with a data store."""
        self.data_store = data_store or grpc_data_store

    async def RequestAllocation(
        self,
        request: Any,
        context: grpc.aio.ServicerContext,
    ) -> Dict[str, Any]:
        """
        Request heat allocation optimization.

        Args:
            request: AllocationRequest proto message
            context: gRPC context

        Returns:
            AllocationResponse proto message
        """
        try:
            logger.info(f"Allocation request received: {request.target_output_mw} MW")

            response = await self.data_store.request_allocation(
                target_output_mw=request.target_output_mw,
                time_window_minutes=request.time_window_minutes,
                objective=proto_to_optimization_objective(request.objective),
                cost_weight=request.cost_weight,
                emissions_weight=request.emissions_weight,
                excluded_assets=list(request.excluded_assets),
                must_run_assets=list(request.must_run_assets),
                is_emergency=request.is_emergency,
            )

            return {
                "request_id": str(response.request_id),
                "response_id": str(response.response_id),
                "success": response.success,
                "status_message": response.status_message,
                "allocated_output_mw": response.allocated_output_mw,
                "allocation_gap_mw": response.allocation_gap_mw,
                "asset_allocations": [
                    {
                        "asset_id": str(a.asset_id),
                        "asset_name": a.asset_name,
                        "current_setpoint_mw": a.current_setpoint_mw,
                        "recommended_setpoint_mw": a.recommended_setpoint_mw,
                        "confidence": a.confidence,
                        "reason": a.reason,
                    }
                    for a in response.asset_allocations
                ],
                "estimated_cost": response.estimated_cost,
                "estimated_emissions_kg": response.estimated_emissions_kg,
                "optimization_time_ms": response.optimization_time_ms,
                "solver_iterations": response.solver_iterations,
            }

        except Exception as e:
            logger.error(f"Allocation request failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return {}

    async def SubmitDemandUpdate(
        self,
        request: Any,
        context: grpc.aio.ServicerContext,
    ) -> Dict[str, Any]:
        """
        Submit demand forecast update.

        Args:
            request: DemandUpdateRequest proto message
            context: gRPC context

        Returns:
            DemandUpdateResponse proto message
        """
        try:
            logger.info("Demand update received via gRPC")

            request_id, records_received, records_validated, quality_score = \
                await self.data_store.submit_demand_update(
                    forecast_type=proto_to_forecast_type(request.forecast_type),
                    forecast_horizon_hours=request.forecast_horizon_hours,
                    resolution_minutes=request.resolution_minutes,
                    demand_mw=list(request.demand_mw),
                    demand_timestamps=[
                        timestamp_to_datetime(ts) for ts in request.demand_timestamps
                    ],
                    source_system=request.source_system,
                    model_version=request.model_version if request.HasField("model_version") else None,
                )

            return {
                "request_id": request_id,
                "success": True,
                "message": "Demand update received and validated",
                "records_received": records_received,
                "records_validated": records_validated,
                "data_quality_score": quality_score,
            }

        except Exception as e:
            logger.error(f"Demand update failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return {}

    async def GetLatestForecast(
        self,
        request: Any,
        context: grpc.aio.ServicerContext,
    ) -> Dict[str, Any]:
        """
        Get latest forecast by type.

        Args:
            request: ForecastRequest proto message
            context: gRPC context

        Returns:
            ForecastResponse proto message
        """
        try:
            forecast_type = proto_to_forecast_type(request.forecast_type)
            forecast = await self.data_store.get_latest_forecast(forecast_type)

            if forecast:
                return {
                    "success": True,
                    "forecast": {
                        "forecast_id": str(forecast.forecast_id),
                        "forecast_type": forecast_type_to_proto(forecast.forecast_type),
                        "forecast_horizon_hours": forecast.forecast_horizon_hours,
                        "resolution_minutes": forecast.resolution_minutes,
                        "generated_at": datetime_to_timestamp(forecast.generated_at),
                        "valid_from": datetime_to_timestamp(forecast.valid_from),
                        "valid_until": datetime_to_timestamp(forecast.valid_until),
                        "values": forecast.values,
                        "timestamps": [datetime_to_timestamp(ts) for ts in forecast.timestamps],
                        "unit": forecast.unit,
                        "confidence_level": forecast.confidence_level,
                        "model_name": forecast.model_name,
                        "model_version": forecast.model_version,
                    },
                }
            else:
                return {
                    "success": False,
                    "error_message": "Forecast not available",
                }

        except Exception as e:
            logger.error(f"Get forecast failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return {}

    async def GetCurrentPlan(
        self,
        request: Any,
        context: grpc.aio.ServicerContext,
    ) -> Dict[str, Any]:
        """
        Get current dispatch plan.

        Args:
            request: Empty proto message
            context: gRPC context

        Returns:
            DispatchPlanResponse proto message
        """
        try:
            plan = await self.data_store.get_current_plan()

            if plan:
                return {
                    "success": True,
                    "plan": self._convert_plan_to_proto(plan),
                }
            else:
                return {
                    "success": False,
                    "error_message": "No active plan available",
                }

        except Exception as e:
            logger.error(f"Get current plan failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return {}

    async def GetAssetStates(
        self,
        request: Any,
        context: grpc.aio.ServicerContext,
    ) -> Dict[str, Any]:
        """
        Get asset states.

        Args:
            request: AssetStatesRequest proto message
            context: gRPC context

        Returns:
            AssetStatesResponse proto message
        """
        try:
            assets = await self.data_store.get_asset_states(
                asset_ids=list(request.asset_ids) if request.asset_ids else None,
                asset_types=list(request.asset_types) if request.asset_types else None,
                statuses=list(request.statuses) if request.statuses else None,
            )

            # Pagination
            page = request.page or 1
            page_size = request.page_size or 20
            total_count = len(assets)
            total_pages = (total_count + page_size - 1) // page_size

            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            paginated = assets[start_idx:end_idx]

            return {
                "success": True,
                "assets": [self._convert_asset_to_proto(a) for a in paginated],
                "total_count": total_count,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages,
            }

        except Exception as e:
            logger.error(f"Get asset states failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return {}

    async def GetConstraints(
        self,
        request: Any,
        context: grpc.aio.ServicerContext,
    ) -> Dict[str, Any]:
        """
        Get active constraints.

        Args:
            request: ConstraintsRequest proto message
            context: gRPC context

        Returns:
            ConstraintsResponse proto message
        """
        try:
            is_active = request.is_active if request.HasField("is_active") else None
            constraints = await self.data_store.get_constraints(
                is_active=is_active,
                constraint_types=list(request.constraint_types) if request.constraint_types else None,
                priorities=list(request.priorities) if request.priorities else None,
            )

            return {
                "success": True,
                "constraints": [self._convert_constraint_to_proto(c) for c in constraints],
            }

        except Exception as e:
            logger.error(f"Get constraints failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return {}

    async def GetKPIs(
        self,
        request: Any,
        context: grpc.aio.ServicerContext,
    ) -> Dict[str, Any]:
        """
        Get KPI metrics.

        Args:
            request: KPIRequest proto message
            context: gRPC context

        Returns:
            KPIResponse proto message
        """
        try:
            category = request.category if request.HasField("category") else None
            kpis = await self.data_store.get_kpis(category=category)

            return {
                "success": True,
                "kpis": [self._convert_kpi_to_proto(k) for k in kpis],
            }

        except Exception as e:
            logger.error(f"Get KPIs failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return {}

    async def AcknowledgeAlarm(
        self,
        request: Any,
        context: grpc.aio.ServicerContext,
    ) -> Dict[str, Any]:
        """
        Acknowledge an alarm.

        Args:
            request: AlarmAcknowledgementRequest proto message
            context: gRPC context

        Returns:
            AlarmAcknowledgementResponse proto message
        """
        try:
            success, message, acknowledged_at = await self.data_store.acknowledge_alarm(
                alarm_id=request.alarm_id,
                acknowledged_by=request.acknowledged_by,
                acknowledgement_note=request.acknowledgement_note if request.HasField("acknowledgement_note") else None,
            )

            return {
                "alarm_id": request.alarm_id,
                "success": success,
                "message": message,
                "acknowledged_at": datetime_to_timestamp(acknowledged_at),
                "acknowledged_by": request.acknowledged_by,
            }

        except Exception as e:
            logger.error(f"Acknowledge alarm failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return {}

    async def StreamPlanUpdates(
        self,
        request: Any,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream plan updates.

        Args:
            request: StreamPlanRequest proto message
            context: gRPC context

        Yields:
            DispatchPlanResponse proto messages
        """
        try:
            while not context.cancelled():
                try:
                    # Wait for plan update with timeout
                    plan = await asyncio.wait_for(
                        self.data_store._plan_update_queue.get(),
                        timeout=60.0,
                    )
                    yield {
                        "success": True,
                        "plan": self._convert_plan_to_proto(plan),
                    }
                except asyncio.TimeoutError:
                    # Send current plan as heartbeat
                    current_plan = await self.data_store.get_current_plan()
                    if current_plan:
                        yield {
                            "success": True,
                            "plan": self._convert_plan_to_proto(current_plan),
                        }

        except Exception as e:
            logger.error(f"Stream plan updates failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))

    async def StreamActionRecommendations(
        self,
        request: Any,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream action recommendations.

        Args:
            request: StreamRecommendationsRequest proto message
            context: gRPC context

        Yields:
            SetpointRecommendations proto messages
        """
        try:
            while not context.cancelled():
                try:
                    recommendations = await asyncio.wait_for(
                        self.data_store._recommendation_queue.get(),
                        timeout=60.0,
                    )
                    yield {
                        "recommendations": [
                            {
                                "asset_id": str(r.asset_id),
                                "asset_name": r.asset_name,
                                "current_setpoint_mw": r.current_setpoint_mw,
                                "recommended_setpoint_mw": r.recommended_setpoint_mw,
                                "confidence": r.confidence,
                                "reason": r.reason,
                            }
                            for r in recommendations
                        ],
                        "timestamp": datetime_to_timestamp(datetime.utcnow()),
                    }
                except asyncio.TimeoutError:
                    # Send empty heartbeat
                    yield {
                        "recommendations": [],
                        "timestamp": datetime_to_timestamp(datetime.utcnow()),
                    }

        except Exception as e:
            logger.error(f"Stream recommendations failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))

    async def StreamAlarmEvents(
        self,
        request: Any,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream alarm events.

        Args:
            request: StreamAlarmsRequest proto message
            context: gRPC context

        Yields:
            AlarmEvent proto messages
        """
        try:
            severity_filter = list(request.severity_filter) if request.severity_filter else None

            while not context.cancelled():
                try:
                    alarm = await asyncio.wait_for(
                        self.data_store._alarm_queue.get(),
                        timeout=60.0,
                    )

                    # Apply filter
                    if severity_filter:
                        alarm_severity_proto = self._alarm_severity_to_proto(alarm.severity)
                        if alarm_severity_proto not in severity_filter:
                            continue

                    yield self._convert_alarm_to_proto(alarm)

                except asyncio.TimeoutError:
                    # Continue waiting
                    continue

        except Exception as e:
            logger.error(f"Stream alarm events failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))

    async def ControlStream(
        self,
        request_iterator: AsyncIterator[Any],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Bidirectional control stream.

        Args:
            request_iterator: Stream of ControlCommand messages
            context: gRPC context

        Yields:
            ControlResponse proto messages
        """
        try:
            async for command in request_iterator:
                response = await self._process_control_command(command)
                yield response

        except Exception as e:
            logger.error(f"Control stream failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))

    async def _process_control_command(self, command: Any) -> Dict[str, Any]:
        """Process a control command and return response."""
        now = datetime.utcnow()

        # Mock implementation - in production, this would interface with actual control systems
        return {
            "command_id": command.command_id,
            "success": True,
            "message": f"Command {command.command_type} executed successfully",
            "execution_timestamp": datetime_to_timestamp(now),
        }

    def _convert_plan_to_proto(self, plan: DispatchPlanSchema) -> Dict[str, Any]:
        """Convert DispatchPlan schema to proto format."""
        return {
            "plan_id": str(plan.plan_id),
            "plan_version": plan.plan_version,
            "plan_name": plan.plan_name,
            "description": plan.description,
            "objective": self._optimization_objective_to_proto(plan.objective),
            "planning_horizon_hours": plan.planning_horizon_hours,
            "resolution_minutes": plan.resolution_minutes,
            "effective_from": datetime_to_timestamp(plan.effective_from),
            "effective_until": datetime_to_timestamp(plan.effective_until),
            "is_active": plan.is_active,
            "schedule": [],
            "setpoint_recommendations": [
                {
                    "asset_id": str(r.asset_id),
                    "asset_name": r.asset_name,
                    "current_setpoint_mw": r.current_setpoint_mw,
                    "recommended_setpoint_mw": r.recommended_setpoint_mw,
                    "confidence": r.confidence,
                    "reason": r.reason,
                }
                for r in plan.setpoint_recommendations
            ],
            "total_thermal_output_mwh": plan.total_thermal_output_mwh,
            "total_cost": plan.total_cost,
            "total_emissions_kg": plan.total_emissions_kg,
            "average_efficiency": plan.average_efficiency,
            "constraints_satisfied": plan.constraints_satisfied,
            "constraints_violated": plan.constraints_violated,
            "violated_constraint_ids": [str(vid) for vid in plan.violated_constraint_ids],
            "optimization_score": plan.optimization_score,
            "solver_status": plan.solver_status,
            "computation_time_seconds": plan.computation_time_seconds,
            "created_at": datetime_to_timestamp(plan.created_at),
        }

    def _convert_asset_to_proto(self, asset: AssetStateSchema) -> Dict[str, Any]:
        """Convert AssetState schema to proto format."""
        return {
            "asset_id": str(asset.asset_id),
            "asset_name": asset.asset_name,
            "asset_type": asset_type_to_proto(asset.asset_type),
            "status": asset_status_to_proto(asset.status),
            "current_output_mw": asset.current_output_mw,
            "current_setpoint_mw": asset.current_setpoint_mw,
            "supply_temperature_c": asset.supply_temperature_c,
            "return_temperature_c": asset.return_temperature_c,
            "flow_rate_m3h": asset.flow_rate_m3h,
            "capacity": {
                "thermal_capacity_mw": asset.capacity.thermal_capacity_mw,
                "min_output_mw": asset.capacity.min_output_mw,
                "max_output_mw": asset.capacity.max_output_mw,
                "ramp_up_rate_mw_min": asset.capacity.ramp_up_rate_mw_min,
                "ramp_down_rate_mw_min": asset.capacity.ramp_down_rate_mw_min,
                "min_uptime_hours": asset.capacity.min_uptime_hours,
                "min_downtime_hours": asset.capacity.min_downtime_hours,
                "startup_time_minutes": asset.capacity.startup_time_minutes,
            },
            "efficiency": {
                "thermal_efficiency": asset.efficiency.thermal_efficiency,
                "electrical_efficiency": asset.efficiency.electrical_efficiency,
            },
            "emissions": {
                "co2_kg_per_mwh": asset.emissions.co2_kg_per_mwh,
                "nox_kg_per_mwh": asset.emissions.nox_kg_per_mwh,
                "so2_kg_per_mwh": asset.emissions.so2_kg_per_mwh,
                "particulate_kg_per_mwh": asset.emissions.particulate_kg_per_mwh,
            },
            "cost": {
                "fuel_cost_per_mwh": asset.cost.fuel_cost_per_mwh,
                "variable_om_per_mwh": asset.cost.variable_om_per_mwh,
                "fixed_om_per_day": asset.cost.fixed_om_per_day,
                "startup_cost": asset.cost.startup_cost,
                "shutdown_cost": asset.cost.shutdown_cost,
            },
            "health": {
                "health_score": asset.health.health_score,
                "remaining_useful_life_hours": asset.health.remaining_useful_life_hours,
                "operating_hours_since_maintenance": asset.health.operating_hours_since_maintenance,
                "fault_indicators": asset.health.fault_indicators,
            },
            "created_at": datetime_to_timestamp(asset.created_at),
        }

    def _convert_constraint_to_proto(self, constraint: ConstraintSchema) -> Dict[str, Any]:
        """Convert Constraint schema to proto format."""
        return {
            "constraint_id": str(constraint.constraint_id),
            "name": constraint.name,
            "description": constraint.description,
            "constraint_type": self._constraint_type_to_proto(constraint.constraint_type),
            "priority": self._constraint_priority_to_proto(constraint.priority),
            "asset_id": str(constraint.asset_id) if constraint.asset_id else None,
            "min_value": constraint.min_value,
            "max_value": constraint.max_value,
            "target_value": constraint.target_value,
            "tolerance": constraint.tolerance,
            "effective_from": datetime_to_timestamp(constraint.effective_from),
            "is_active": constraint.is_active,
            "is_violated": constraint.is_violated,
            "violation_count": constraint.violation_count,
        }

    def _convert_kpi_to_proto(self, kpi: KPISchema) -> Dict[str, Any]:
        """Convert KPI schema to proto format."""
        return {
            "kpi_id": str(kpi.kpi_id),
            "name": kpi.name,
            "description": kpi.description,
            "category": kpi.category,
            "current_value": kpi.current_value,
            "target_value": kpi.target_value,
            "unit": kpi.unit,
            "measurement_timestamp": datetime_to_timestamp(kpi.measurement_timestamp),
            "aggregation_period": kpi.aggregation_period,
            "previous_value": kpi.previous_value,
            "trend_direction": kpi.trend_direction,
            "percent_change": kpi.percent_change,
            "target_achievement_percent": kpi.target_achievement_percent,
            "is_on_target": kpi.is_on_target,
        }

    def _convert_alarm_to_proto(self, alarm: AlarmEventSchema) -> Dict[str, Any]:
        """Convert AlarmEvent schema to proto format."""
        return {
            "alarm_id": str(alarm.alarm_id),
            "alarm_code": alarm.alarm_code,
            "name": alarm.name,
            "description": alarm.description,
            "severity": self._alarm_severity_to_proto(alarm.severity),
            "status": self._alarm_status_to_proto(alarm.status),
            "asset_id": str(alarm.asset_id) if alarm.asset_id else None,
            "asset_name": alarm.asset_name,
            "subsystem": alarm.subsystem,
            "triggered_at": datetime_to_timestamp(alarm.triggered_at),
            "measured_value": alarm.measured_value,
            "threshold_value": alarm.threshold_value,
            "unit": alarm.unit,
            "recommended_actions": alarm.recommended_actions,
            "auto_response_triggered": alarm.auto_response_triggered,
        }

    def _optimization_objective_to_proto(self, obj: OptimizationObjective) -> int:
        """Convert OptimizationObjective to proto enum."""
        mapping = {
            OptimizationObjective.MINIMIZE_COST: ProtoOptimizationObjective.OPTIMIZATION_OBJECTIVE_MINIMIZE_COST,
            OptimizationObjective.MINIMIZE_EMISSIONS: ProtoOptimizationObjective.OPTIMIZATION_OBJECTIVE_MINIMIZE_EMISSIONS,
            OptimizationObjective.MAXIMIZE_EFFICIENCY: ProtoOptimizationObjective.OPTIMIZATION_OBJECTIVE_MAXIMIZE_EFFICIENCY,
            OptimizationObjective.BALANCE_COST_EMISSIONS: ProtoOptimizationObjective.OPTIMIZATION_OBJECTIVE_BALANCE_COST_EMISSIONS,
        }
        return mapping.get(obj, ProtoOptimizationObjective.OPTIMIZATION_OBJECTIVE_UNSPECIFIED)

    def _constraint_type_to_proto(self, ct: ConstraintType) -> int:
        """Convert ConstraintType to proto enum."""
        mapping = {
            ConstraintType.CAPACITY_MIN: ProtoConstraintType.CONSTRAINT_TYPE_CAPACITY_MIN,
            ConstraintType.CAPACITY_MAX: ProtoConstraintType.CONSTRAINT_TYPE_CAPACITY_MAX,
            ConstraintType.RAMP_RATE: ProtoConstraintType.CONSTRAINT_TYPE_RAMP_RATE,
            ConstraintType.TEMPERATURE_MIN: ProtoConstraintType.CONSTRAINT_TYPE_TEMPERATURE_MIN,
            ConstraintType.TEMPERATURE_MAX: ProtoConstraintType.CONSTRAINT_TYPE_TEMPERATURE_MAX,
        }
        return mapping.get(ct, ProtoConstraintType.CONSTRAINT_TYPE_UNSPECIFIED)

    def _constraint_priority_to_proto(self, cp: ConstraintPriority) -> int:
        """Convert ConstraintPriority to proto enum."""
        mapping = {
            ConstraintPriority.CRITICAL: ProtoConstraintPriority.CONSTRAINT_PRIORITY_CRITICAL,
            ConstraintPriority.HIGH: ProtoConstraintPriority.CONSTRAINT_PRIORITY_HIGH,
            ConstraintPriority.MEDIUM: ProtoConstraintPriority.CONSTRAINT_PRIORITY_MEDIUM,
            ConstraintPriority.LOW: ProtoConstraintPriority.CONSTRAINT_PRIORITY_LOW,
        }
        return mapping.get(cp, ProtoConstraintPriority.CONSTRAINT_PRIORITY_UNSPECIFIED)

    def _alarm_severity_to_proto(self, severity: AlarmSeverity) -> int:
        """Convert AlarmSeverity to proto enum."""
        mapping = {
            AlarmSeverity.CRITICAL: ProtoAlarmSeverity.ALARM_SEVERITY_CRITICAL,
            AlarmSeverity.HIGH: ProtoAlarmSeverity.ALARM_SEVERITY_HIGH,
            AlarmSeverity.MEDIUM: ProtoAlarmSeverity.ALARM_SEVERITY_MEDIUM,
            AlarmSeverity.LOW: ProtoAlarmSeverity.ALARM_SEVERITY_LOW,
            AlarmSeverity.INFO: ProtoAlarmSeverity.ALARM_SEVERITY_INFO,
        }
        return mapping.get(severity, ProtoAlarmSeverity.ALARM_SEVERITY_UNSPECIFIED)

    def _alarm_status_to_proto(self, status: AlarmStatus) -> int:
        """Convert AlarmStatus to proto enum."""
        mapping = {
            AlarmStatus.ACTIVE: ProtoAlarmStatus.ALARM_STATUS_ACTIVE,
            AlarmStatus.ACKNOWLEDGED: ProtoAlarmStatus.ALARM_STATUS_ACKNOWLEDGED,
            AlarmStatus.RESOLVED: ProtoAlarmStatus.ALARM_STATUS_RESOLVED,
            AlarmStatus.SUPPRESSED: ProtoAlarmStatus.ALARM_STATUS_SUPPRESSED,
        }
        return mapping.get(status, ProtoAlarmStatus.ALARM_STATUS_UNSPECIFIED)


# =============================================================================
# gRPC Server Configuration
# =============================================================================

async def serve_grpc(
    host: str = "0.0.0.0",
    port: int = 50051,
    max_workers: int = 10,
    enable_reflection: bool = True,
    enable_auth: bool = True,
) -> grpc.aio.Server:
    """
    Start the gRPC server.

    Args:
        host: Host to bind to
        port: Port to listen on
        max_workers: Maximum number of worker threads
        enable_reflection: Enable gRPC reflection for debugging
        enable_auth: Enable authentication interceptor

    Returns:
        Running gRPC server instance
    """
    # Create server with interceptors
    interceptors = []
    if enable_auth:
        interceptors.append(AuthInterceptor())

    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        interceptors=interceptors,
    )

    # Add service
    # NOTE: In production, use generated stubs:
    # thermal_command_pb2_grpc.add_ThermalCommandServiceServicer_to_server(
    #     ThermalCommandServicer(), server
    # )

    # Bind to address
    listen_addr = f"{host}:{port}"
    server.add_insecure_port(listen_addr)

    # Start server
    await server.start()
    logger.info(f"gRPC server started on {listen_addr}")

    return server


async def serve_grpc_with_tls(
    host: str = "0.0.0.0",
    port: int = 50051,
    server_cert_path: str = "",
    server_key_path: str = "",
    ca_cert_path: Optional[str] = None,
    max_workers: int = 10,
    enable_auth: bool = True,
) -> grpc.aio.Server:
    """
    Start the gRPC server with TLS/mTLS.

    Args:
        host: Host to bind to
        port: Port to listen on
        server_cert_path: Path to server certificate
        server_key_path: Path to server private key
        ca_cert_path: Path to CA certificate for mTLS (optional)
        max_workers: Maximum number of worker threads
        enable_auth: Enable authentication interceptor

    Returns:
        Running gRPC server instance
    """
    # Read certificates
    with open(server_key_path, "rb") as f:
        server_key = f.read()
    with open(server_cert_path, "rb") as f:
        server_cert = f.read()

    ca_cert = None
    if ca_cert_path:
        with open(ca_cert_path, "rb") as f:
            ca_cert = f.read()

    # Create credentials
    if ca_cert:
        # mTLS - require client certificate
        credentials = grpc.ssl_server_credentials(
            [(server_key, server_cert)],
            root_certificates=ca_cert,
            require_client_auth=True,
        )
    else:
        # TLS only
        credentials = grpc.ssl_server_credentials(
            [(server_key, server_cert)],
        )

    # Create server with interceptors
    interceptors = []
    if enable_auth:
        interceptors.append(AuthInterceptor())

    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        interceptors=interceptors,
    )

    # Bind to address with TLS
    listen_addr = f"{host}:{port}"
    server.add_secure_port(listen_addr, credentials)

    # Start server
    await server.start()
    logger.info(f"gRPC server started with TLS on {listen_addr}")

    return server
