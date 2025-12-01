"""
Integration Test Utilities Module for GL-014 EXCHANGER-PRO.

Provides mock connectors, sample data generators, and response simulators
for testing heat exchanger integration components without requiring
actual external system connections.

Author: GL-DataIntegrationEngineer
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
import asyncio
import json
import logging
import math
import random
import uuid
from collections import defaultdict

from pydantic import BaseModel, Field

from .base_connector import (
    BaseConnector,
    BaseConnectorConfig,
    CircuitState,
    ConnectionState,
    ConnectorType,
    HealthCheckResult,
    HealthStatus,
    DataQualityResult,
    DataQualityLevel,
    ConnectorError,
)
from .process_historian_connector import (
    HistorianProvider,
    DataRetrievalMode,
    TagQuality,
    TagDataType,
    TagDefinition,
    TagValue,
    TimeSeriesData,
    BulkTimeSeriesRequest,
    BulkTimeSeriesResponse,
    HeatExchangerTagSet,
    HeatExchangerSnapshot,
    ProcessHistorianConnectorConfig,
)
from .cmms_connector import (
    CMSProvider,
    WorkOrderStatus,
    WorkOrderPriority,
    WorkOrderType,
    EquipmentStatus,
    CleaningMethod,
    HeatExchangerEquipment,
    CleaningWorkOrder,
    MaintenanceHistory,
    CleaningWorkOrderCreateRequest,
    CMSSConnectorConfig,
)
from .dcs_scada_connector import (
    DCSProvider,
    TagQuality as DCSTagQuality,
    AlarmPriority,
    AlarmState,
    ControlMode,
    RealtimeTagValue,
    DCSAlarm,
    HeatExchangerControlTags,
    HeatExchangerRealtimeData,
    DCSConnectorConfig,
)
from .agent_coordinator import (
    AgentID,
    MessageType,
    MessagePriority,
    AgentMessage,
    AgentResponse,
    AgentStatus,
    HeatExchangerPerformanceData,
    HeatRecoveryOpportunity,
    MaintenancePrediction,
    ThermalEfficiencyData,
    AgentCoordinatorConfig,
)
from .data_transformers import (
    UnitConverter,
    DataQualityScorer,
    OutlierDetector,
    DataTransformer,
)

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Sample Data Generators
# =============================================================================


class HeatExchangerDataGenerator:
    """Generates realistic sample data for heat exchangers."""

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize generator with optional seed for reproducibility."""
        if seed:
            random.seed(seed)

        # Define typical heat exchanger parameters
        self._typical_exchangers = {
            "shell_tube": {
                "hot_inlet_temp_range": (80, 200),
                "hot_outlet_temp_range": (50, 120),
                "cold_inlet_temp_range": (15, 40),
                "cold_outlet_temp_range": (40, 80),
                "flow_range": (1, 50),  # kg/s
                "pressure_range": (100, 2000),  # kPa
                "duty_range": (100, 5000),  # kW
                "ua_range": (1000, 50000),  # W/K
            },
            "plate": {
                "hot_inlet_temp_range": (60, 150),
                "hot_outlet_temp_range": (40, 100),
                "cold_inlet_temp_range": (10, 35),
                "cold_outlet_temp_range": (35, 70),
                "flow_range": (0.5, 20),
                "pressure_range": (50, 1000),
                "duty_range": (50, 2000),
                "ua_range": (500, 20000),
            },
            "air_cooled": {
                "hot_inlet_temp_range": (100, 300),
                "hot_outlet_temp_range": (50, 150),
                "cold_inlet_temp_range": (15, 40),  # Air ambient
                "cold_outlet_temp_range": (40, 100),
                "flow_range": (10, 200),  # Air flow
                "pressure_range": (100, 500),
                "duty_range": (500, 10000),
                "ua_range": (5000, 100000),
            },
        }

    def generate_equipment(
        self,
        equipment_id: Optional[str] = None,
        exchanger_type: str = "shell_tube"
    ) -> HeatExchangerEquipment:
        """Generate sample heat exchanger equipment master data."""
        params = self._typical_exchangers.get(exchanger_type, self._typical_exchangers["shell_tube"])

        equipment_id = equipment_id or f"HX-{random.randint(1000, 9999)}"

        return HeatExchangerEquipment(
            equipment_id=equipment_id,
            equipment_name=f"{exchanger_type.replace('_', ' ').title()} Heat Exchanger {equipment_id}",
            description=f"Sample {exchanger_type} heat exchanger for testing",
            equipment_type="heat_exchanger",
            equipment_subtype=exchanger_type,
            criticality=random.choice(list(EquipmentStatus)),
            status=EquipmentStatus.OPERATIONAL,
            location_id=f"LOC-{random.randint(100, 999)}",
            plant_id="PLANT-001",
            area=random.choice(["Process Area 1", "Utilities", "Tank Farm"]),
            manufacturer=random.choice(["Alfa Laval", "TEMA", "APV", "Koch Heat Transfer"]),
            model=f"Model-{random.randint(100, 999)}",
            design_duty_kw=random.uniform(*params["duty_range"]),
            design_area_m2=random.uniform(10, 500),
            design_ua_w_k=random.uniform(*params["ua_range"]),
            shell_material=random.choice(["Carbon Steel", "Stainless 316", "Titanium"]),
            tube_material=random.choice(["Carbon Steel", "Copper", "Stainless 304"]),
            number_of_tubes=random.randint(50, 500),
            tube_length_m=random.uniform(2, 8),
            shell_passes=random.choice([1, 2]),
            tube_passes=random.choice([1, 2, 4]),
            design_fouling_factor=random.uniform(0.0001, 0.0005),
            cleaning_interval_days=random.randint(90, 365),
            last_cleaning_date=datetime.utcnow() - timedelta(days=random.randint(30, 180)),
            installation_date=datetime.utcnow() - timedelta(days=random.randint(365, 3650)),
        )

    def generate_time_series(
        self,
        tag_name: str,
        start_time: datetime,
        end_time: datetime,
        interval_seconds: int = 60,
        base_value: float = 100.0,
        noise_percent: float = 2.0,
        trend_per_hour: float = 0.0,
        add_anomalies: bool = False
    ) -> TimeSeriesData:
        """Generate sample time series data."""
        values = []
        current_time = start_time

        hours_elapsed = 0
        while current_time <= end_time:
            # Base value with trend
            value = base_value + trend_per_hour * hours_elapsed

            # Add noise
            noise = value * noise_percent / 100 * random.gauss(0, 1)
            value += noise

            # Occasional anomalies
            if add_anomalies and random.random() < 0.01:
                value *= random.uniform(0.8, 1.2)

            quality = TagQuality.GOOD
            if random.random() < 0.02:  # 2% bad quality
                quality = random.choice([TagQuality.BAD, TagQuality.UNCERTAIN])

            values.append(TagValue(
                tag_name=tag_name,
                timestamp=current_time,
                value=value,
                quality=quality
            ))

            current_time += timedelta(seconds=interval_seconds)
            hours_elapsed = (current_time - start_time).total_seconds() / 3600

        # Calculate statistics
        numeric_values = [v.value for v in values if v.quality == TagQuality.GOOD]

        return TimeSeriesData(
            tag_name=tag_name,
            start_time=start_time,
            end_time=end_time,
            retrieval_mode=DataRetrievalMode.INTERPOLATED,
            interval_seconds=interval_seconds,
            values=values,
            point_count=len(values),
            min_value=min(numeric_values) if numeric_values else None,
            max_value=max(numeric_values) if numeric_values else None,
            avg_value=sum(numeric_values) / len(numeric_values) if numeric_values else None,
            good_quality_percent=(len(numeric_values) / len(values) * 100) if values else 100
        )

    def generate_heat_exchanger_snapshot(
        self,
        equipment_id: str,
        exchanger_type: str = "shell_tube",
        fouling_level: float = 0.5  # 0 = clean, 1 = heavily fouled
    ) -> HeatExchangerSnapshot:
        """Generate sample real-time snapshot for heat exchanger."""
        params = self._typical_exchangers.get(exchanger_type, self._typical_exchangers["shell_tube"])

        # Calculate values based on fouling
        ua_degradation = 1.0 - (fouling_level * 0.4)  # UA drops with fouling

        hot_inlet = random.uniform(*params["hot_inlet_temp_range"])
        cold_inlet = random.uniform(*params["cold_inlet_temp_range"])

        # Simplified heat exchanger model
        effectiveness = random.uniform(0.6, 0.9) * ua_degradation
        duty = random.uniform(*params["duty_range"]) * ua_degradation

        hot_outlet = hot_inlet - (effectiveness * (hot_inlet - cold_inlet))
        cold_outlet = cold_inlet + (effectiveness * (hot_inlet - cold_inlet) * 0.9)

        return HeatExchangerSnapshot(
            equipment_id=equipment_id,
            timestamp=datetime.utcnow(),
            hot_inlet_temp=hot_inlet,
            hot_outlet_temp=hot_outlet,
            cold_inlet_temp=cold_inlet,
            cold_outlet_temp=cold_outlet,
            hot_flow=random.uniform(*params["flow_range"]),
            cold_flow=random.uniform(*params["flow_range"]) * 0.9,
            hot_inlet_pressure=random.uniform(*params["pressure_range"]),
            hot_outlet_pressure=random.uniform(*params["pressure_range"]) * 0.95,
            cold_inlet_pressure=random.uniform(*params["pressure_range"]) * 0.8,
            cold_outlet_pressure=random.uniform(*params["pressure_range"]) * 0.75,
            duty=duty,
            ua_coefficient=random.uniform(*params["ua_range"]) * ua_degradation,
            lmtd=((hot_inlet - cold_outlet) - (hot_outlet - cold_inlet)) /
                  math.log((hot_inlet - cold_outlet) / max(hot_outlet - cold_inlet, 0.1)),
            effectiveness=effectiveness,
            fouling_factor=0.0001 + (fouling_level * 0.0004),
            data_quality_score=random.uniform(0.9, 1.0),
        )

    def generate_work_order(
        self,
        equipment_id: str,
        status: WorkOrderStatus = WorkOrderStatus.PENDING
    ) -> CleaningWorkOrder:
        """Generate sample cleaning work order."""
        return CleaningWorkOrder(
            work_order_id=f"WO-{random.randint(10000, 99999)}",
            work_order_number=f"WO{datetime.utcnow().strftime('%Y%m%d')}{random.randint(100, 999)}",
            title=f"Cleaning - {equipment_id}",
            description=f"Scheduled cleaning for heat exchanger {equipment_id}",
            work_order_type=WorkOrderType.CLEANING,
            priority=random.choice(list(WorkOrderPriority)),
            status=status,
            equipment_id=equipment_id,
            cleaning_method=random.choice(list(CleaningMethod)),
            cleaning_reason="scheduled_maintenance",
            current_fouling_factor=random.uniform(0.0002, 0.0005),
            current_ua_percent=random.uniform(60, 85),
            scheduled_start=datetime.utcnow() + timedelta(days=random.randint(1, 30)),
            estimated_hours=random.uniform(4, 24),
            estimated_downtime_hours=random.uniform(8, 48),
            estimated_cost=random.uniform(5000, 50000),
        )

    def generate_maintenance_history(
        self,
        equipment_id: str,
        num_records: int = 10
    ) -> List[MaintenanceHistory]:
        """Generate sample maintenance history."""
        history = []
        current_date = datetime.utcnow() - timedelta(days=num_records * 90)

        for _ in range(num_records):
            pre_fouling = random.uniform(0.0003, 0.0006)
            post_fouling = random.uniform(0.00005, 0.0002)

            history.append(MaintenanceHistory(
                history_id=str(uuid.uuid4()),
                equipment_id=equipment_id,
                maintenance_type="cleaning",
                maintenance_date=current_date,
                description="Chemical cleaning of heat exchanger",
                cleaning_method=random.choice(list(CleaningMethod)),
                pre_cleaning_fouling=pre_fouling,
                post_cleaning_fouling=post_fouling,
                fouling_removed_percent=(1 - post_fouling / pre_fouling) * 100,
                pre_cleaning_ua_percent=random.uniform(60, 80),
                post_cleaning_ua_percent=random.uniform(90, 98),
                downtime_hours=random.uniform(8, 24),
                labor_hours=random.uniform(16, 48),
                total_cost=random.uniform(10000, 50000),
            ))

            current_date += timedelta(days=random.randint(60, 120))

        return history

    def generate_alarm(
        self,
        equipment_id: str,
        tag_name: Optional[str] = None
    ) -> DCSAlarm:
        """Generate sample DCS alarm."""
        alarm_types = [
            ("High Temperature", AlarmPriority.HIGH, "hot_outlet_temp"),
            ("Low Flow", AlarmPriority.MEDIUM, "hot_flow"),
            ("High Pressure Drop", AlarmPriority.HIGH, "dp"),
            ("UA Degradation", AlarmPriority.MEDIUM, "ua"),
            ("Fouling Alert", AlarmPriority.LOW, "fouling"),
        ]

        alarm_type, priority, default_tag = random.choice(alarm_types)
        tag = tag_name or f"{equipment_id}.{default_tag}"

        return DCSAlarm(
            alarm_id=str(uuid.uuid4()),
            tag_name=tag,
            priority=priority,
            state=AlarmState.ACTIVE,
            message=f"{alarm_type} alarm for {equipment_id}",
            description=f"Heat exchanger {equipment_id} has triggered a {alarm_type.lower()} condition",
            alarm_value=random.uniform(80, 120),
            timestamp=datetime.utcnow() - timedelta(minutes=random.randint(1, 60)),
            equipment_id=equipment_id,
        )


# =============================================================================
# Mock Connectors
# =============================================================================


class MockProcessHistorianConnector:
    """Mock process historian connector for testing."""

    def __init__(
        self,
        provider: HistorianProvider = HistorianProvider.OSISOFT_PI,
        data_generator: Optional[HeatExchangerDataGenerator] = None
    ) -> None:
        """Initialize mock connector."""
        self._provider = provider
        self._generator = data_generator or HeatExchangerDataGenerator()
        self._state = ConnectionState.DISCONNECTED
        self._tags: Dict[str, TagDefinition] = {}
        self._heat_exchanger_tags: Dict[str, HeatExchangerTagSet] = {}

        # Response delays for realistic simulation
        self._min_latency_ms = 10
        self._max_latency_ms = 100

    @property
    def state(self) -> ConnectionState:
        return self._state

    async def connect(self) -> None:
        """Simulate connection."""
        await asyncio.sleep(random.uniform(0.1, 0.5))
        self._state = ConnectionState.CONNECTED
        logger.info(f"Mock {self._provider.value} historian connected")

    async def disconnect(self) -> None:
        """Simulate disconnection."""
        await asyncio.sleep(0.05)
        self._state = ConnectionState.DISCONNECTED
        logger.info("Mock historian disconnected")

    async def health_check(self) -> HealthCheckResult:
        """Return mock health check result."""
        return HealthCheckResult(
            status=HealthStatus.HEALTHY if self._state == ConnectionState.CONNECTED else HealthStatus.UNHEALTHY,
            latency_ms=random.uniform(self._min_latency_ms, self._max_latency_ms),
            message="Mock historian healthy"
        )

    def register_tags(self, tags: List[TagDefinition]) -> None:
        """Register tags for simulation."""
        for tag in tags:
            self._tags[tag.tag_name] = tag

    def register_heat_exchanger(self, tag_set: HeatExchangerTagSet) -> None:
        """Register heat exchanger tag set."""
        self._heat_exchanger_tags[tag_set.equipment_id] = tag_set

    async def get_time_series(
        self,
        tag_name: str,
        start_time: datetime,
        end_time: datetime,
        mode: DataRetrievalMode = DataRetrievalMode.INTERPOLATED,
        interval_seconds: int = 60
    ) -> TimeSeriesData:
        """Return simulated time series data."""
        await asyncio.sleep(random.uniform(
            self._min_latency_ms / 1000,
            self._max_latency_ms / 1000
        ))

        # Get tag definition for realistic values
        tag_def = self._tags.get(tag_name)

        base_value = 100.0
        if tag_def:
            if "temp" in tag_name.lower():
                base_value = random.uniform(50, 150)
            elif "flow" in tag_name.lower():
                base_value = random.uniform(5, 50)
            elif "pressure" in tag_name.lower():
                base_value = random.uniform(100, 500)

        return self._generator.generate_time_series(
            tag_name=tag_name,
            start_time=start_time,
            end_time=end_time,
            interval_seconds=interval_seconds,
            base_value=base_value
        )

    async def get_bulk_time_series(
        self,
        request: BulkTimeSeriesRequest
    ) -> BulkTimeSeriesResponse:
        """Return simulated bulk time series data."""
        results = {}
        successful_tags = []
        failed_tags = []

        for tag_name in request.tag_names:
            try:
                data = await self.get_time_series(
                    tag_name=tag_name,
                    start_time=request.start_time,
                    end_time=request.end_time,
                    mode=request.retrieval_mode,
                    interval_seconds=request.interval_seconds or 60
                )
                results[tag_name] = data
                successful_tags.append(tag_name)
            except Exception as e:
                failed_tags.append(tag_name)

        return BulkTimeSeriesResponse(
            request=request,
            results=results,
            successful_tags=successful_tags,
            failed_tags=failed_tags,
            total_tags=len(request.tag_names),
            total_points=sum(r.point_count for r in results.values()),
            overall_quality_score=0.95
        )

    async def get_heat_exchanger_snapshot(
        self,
        equipment_id: str
    ) -> HeatExchangerSnapshot:
        """Return simulated heat exchanger snapshot."""
        await asyncio.sleep(random.uniform(
            self._min_latency_ms / 1000,
            self._max_latency_ms / 1000
        ))
        return self._generator.generate_heat_exchanger_snapshot(equipment_id)


class MockCMSSConnector:
    """Mock CMMS connector for testing."""

    def __init__(
        self,
        provider: CMSProvider = CMSProvider.SAP_PM,
        data_generator: Optional[HeatExchangerDataGenerator] = None
    ) -> None:
        """Initialize mock connector."""
        self._provider = provider
        self._generator = data_generator or HeatExchangerDataGenerator()
        self._state = ConnectionState.DISCONNECTED

        # In-memory storage
        self._equipment: Dict[str, HeatExchangerEquipment] = {}
        self._work_orders: Dict[str, CleaningWorkOrder] = {}
        self._maintenance_history: Dict[str, List[MaintenanceHistory]] = {}

    @property
    def state(self) -> ConnectionState:
        return self._state

    async def connect(self) -> None:
        """Simulate connection."""
        await asyncio.sleep(random.uniform(0.2, 0.5))
        self._state = ConnectionState.CONNECTED
        logger.info(f"Mock {self._provider.value} CMMS connected")

    async def disconnect(self) -> None:
        """Simulate disconnection."""
        self._state = ConnectionState.DISCONNECTED

    async def health_check(self) -> HealthCheckResult:
        """Return mock health check."""
        return HealthCheckResult(
            status=HealthStatus.HEALTHY if self._state == ConnectionState.CONNECTED else HealthStatus.UNHEALTHY,
            message="Mock CMMS healthy"
        )

    def add_equipment(self, equipment: HeatExchangerEquipment) -> None:
        """Add equipment to mock database."""
        self._equipment[equipment.equipment_id] = equipment

    async def get_equipment(self, equipment_id: str) -> Optional[HeatExchangerEquipment]:
        """Get equipment from mock database."""
        await asyncio.sleep(0.05)

        if equipment_id in self._equipment:
            return self._equipment[equipment_id]

        # Generate if not exists
        equipment = self._generator.generate_equipment(equipment_id)
        self._equipment[equipment_id] = equipment
        return equipment

    async def list_equipment(self) -> List[HeatExchangerEquipment]:
        """List all equipment."""
        await asyncio.sleep(0.1)

        if not self._equipment:
            # Generate some sample equipment
            for i in range(5):
                equip = self._generator.generate_equipment()
                self._equipment[equip.equipment_id] = equip

        return list(self._equipment.values())

    async def create_cleaning_work_order(
        self,
        request: CleaningWorkOrderCreateRequest
    ) -> CleaningWorkOrder:
        """Create mock work order."""
        await asyncio.sleep(0.1)

        work_order = CleaningWorkOrder(
            work_order_id=f"WO-{random.randint(10000, 99999)}",
            title=request.title,
            description=request.description,
            equipment_id=request.equipment_id,
            priority=request.priority,
            status=WorkOrderStatus.PENDING,
            cleaning_method=request.cleaning_method,
            cleaning_reason=request.cleaning_reason,
            scheduled_start=request.scheduled_start,
            estimated_hours=request.estimated_hours,
        )

        self._work_orders[work_order.work_order_id] = work_order
        return work_order

    async def get_work_order(self, work_order_id: str) -> Optional[CleaningWorkOrder]:
        """Get work order."""
        await asyncio.sleep(0.05)
        return self._work_orders.get(work_order_id)

    async def list_work_orders(
        self,
        equipment_id: Optional[str] = None
    ) -> List[CleaningWorkOrder]:
        """List work orders."""
        await asyncio.sleep(0.1)

        if not self._work_orders:
            # Generate sample work orders
            for equip_id in list(self._equipment.keys())[:3]:
                wo = self._generator.generate_work_order(equip_id)
                self._work_orders[wo.work_order_id] = wo

        orders = list(self._work_orders.values())
        if equipment_id:
            orders = [o for o in orders if o.equipment_id == equipment_id]
        return orders

    async def get_maintenance_history(
        self,
        equipment_id: str,
        limit: int = 10
    ) -> List[MaintenanceHistory]:
        """Get maintenance history."""
        await asyncio.sleep(0.1)

        if equipment_id not in self._maintenance_history:
            self._maintenance_history[equipment_id] = self._generator.generate_maintenance_history(
                equipment_id,
                num_records=limit
            )

        return self._maintenance_history[equipment_id][:limit]


class MockDCSConnector:
    """Mock DCS/SCADA connector for testing."""

    def __init__(
        self,
        provider: DCSProvider = DCSProvider.EMERSON_DELTAV,
        data_generator: Optional[HeatExchangerDataGenerator] = None
    ) -> None:
        """Initialize mock connector."""
        self._provider = provider
        self._generator = data_generator or HeatExchangerDataGenerator()
        self._state = ConnectionState.DISCONNECTED

        self._subscriptions: Dict[str, Dict[str, Any]] = {}
        self._alarms: List[DCSAlarm] = []
        self._tag_values: Dict[str, RealtimeTagValue] = {}

    @property
    def state(self) -> ConnectionState:
        return self._state

    async def connect(self) -> None:
        """Simulate connection."""
        await asyncio.sleep(0.1)
        self._state = ConnectionState.CONNECTED
        logger.info(f"Mock {self._provider.value} DCS connected")

    async def disconnect(self) -> None:
        """Simulate disconnection."""
        self._state = ConnectionState.DISCONNECTED

    async def health_check(self) -> HealthCheckResult:
        """Return mock health check."""
        return HealthCheckResult(
            status=HealthStatus.HEALTHY if self._state == ConnectionState.CONNECTED else HealthStatus.UNHEALTHY,
            message="Mock DCS healthy"
        )

    async def read_tag(self, tag_name: str) -> RealtimeTagValue:
        """Read single tag value."""
        await asyncio.sleep(0.01)

        # Generate realistic value based on tag name
        if "temp" in tag_name.lower():
            value = random.uniform(50, 200)
        elif "flow" in tag_name.lower():
            value = random.uniform(1, 100)
        elif "pressure" in tag_name.lower():
            value = random.uniform(100, 1000)
        else:
            value = random.uniform(0, 100)

        return RealtimeTagValue(
            tag_name=tag_name,
            value=value,
            timestamp=datetime.utcnow(),
            quality=DCSTagQuality.GOOD if random.random() > 0.05 else DCSTagQuality.UNCERTAIN,
        )

    async def read_tags(self, tag_names: List[str]) -> Dict[str, RealtimeTagValue]:
        """Read multiple tag values."""
        results = {}
        for tag in tag_names:
            results[tag] = await self.read_tag(tag)
        return results

    async def subscribe(self, tag_name: str) -> str:
        """Subscribe to tag updates."""
        sub_id = str(uuid.uuid4())
        self._subscriptions[sub_id] = {"tag": tag_name, "active": True}
        return sub_id

    async def get_active_alarms(
        self,
        equipment_id: Optional[str] = None
    ) -> List[DCSAlarm]:
        """Get active alarms."""
        await asyncio.sleep(0.05)

        # Generate some alarms if none exist
        if not self._alarms:
            for _ in range(random.randint(0, 5)):
                self._alarms.append(
                    self._generator.generate_alarm(
                        equipment_id or f"HX-{random.randint(1000, 9999)}"
                    )
                )

        if equipment_id:
            return [a for a in self._alarms if a.equipment_id == equipment_id]
        return self._alarms


class MockAgentCoordinator:
    """Mock agent coordinator for testing."""

    def __init__(
        self,
        data_generator: Optional[HeatExchangerDataGenerator] = None
    ) -> None:
        """Initialize mock coordinator."""
        self._generator = data_generator or HeatExchangerDataGenerator()
        self._state = ConnectionState.DISCONNECTED
        self._connected_agents: Dict[AgentID, AgentStatus] = {}
        self._message_log: List[AgentMessage] = []

    @property
    def state(self) -> ConnectionState:
        return self._state

    async def connect(self) -> None:
        """Simulate connection."""
        await asyncio.sleep(0.2)
        self._state = ConnectionState.CONNECTED

        # Simulate connected agents
        for agent_id in [AgentID.GL_001_THERMOSYNC, AgentID.GL_006_HEATRECLAIM, AgentID.GL_013_PREDICTMAINT]:
            self._connected_agents[agent_id] = AgentStatus(
                agent_id=agent_id,
                agent_name=agent_id.value,
                agent_version="1.0.0",
                status="online",
                health_status=HealthStatus.HEALTHY,
            )

    async def disconnect(self) -> None:
        """Simulate disconnection."""
        self._state = ConnectionState.DISCONNECTED
        self._connected_agents.clear()

    async def send_message(
        self,
        message: AgentMessage,
        wait_for_response: bool = True
    ) -> Optional[AgentResponse]:
        """Simulate sending message and receiving response."""
        await asyncio.sleep(random.uniform(0.05, 0.2))

        self._message_log.append(message)

        if not wait_for_response:
            return None

        # Generate mock response based on action
        if message.action == "get_thermal_efficiency":
            return AgentResponse(
                correlation_id=message.message_id,
                source_agent=message.target_agent,
                target_agent=message.source_agent,
                success=True,
                result={
                    "system_id": message.payload.get("system_id"),
                    "overall_thermal_efficiency": random.uniform(0.7, 0.95),
                    "target_efficiency": 0.85,
                }
            )

        elif message.action == "get_heat_recovery_opportunities":
            return AgentResponse(
                correlation_id=message.message_id,
                source_agent=message.target_agent,
                target_agent=message.source_agent,
                success=True,
                result={
                    "opportunities": [
                        {
                            "opportunity_id": str(uuid.uuid4()),
                            "heat_exchanger_id": message.payload.get("heat_exchanger_id", "HX-001"),
                            "source_stream_id": "STREAM-HOT-1",
                            "sink_stream_id": "STREAM-COLD-1",
                            "source_temperature": 150,
                            "sink_temperature": 30,
                            "source_flow_rate": 10,
                            "sink_flow_rate": 8,
                            "source_heat_content_kw": 500,
                            "sink_heat_demand_kw": 400,
                            "recoverable_heat_kw": 350,
                            "recovery_efficiency": 0.7,
                            "annual_energy_savings_mwh": 2800,
                            "annual_cost_savings": 140000,
                            "co2_reduction_tonnes": 500,
                        }
                    ]
                }
            )

        elif message.action == "get_predictions":
            return AgentResponse(
                correlation_id=message.message_id,
                source_agent=message.target_agent,
                target_agent=message.source_agent,
                success=True,
                result={
                    "predictions": [
                        {
                            "prediction_id": str(uuid.uuid4()),
                            "equipment_id": message.payload.get("equipment_id"),
                            "prediction_type": "fouling",
                            "predicted_date": (datetime.utcnow() + timedelta(days=45)).isoformat(),
                            "confidence_score": random.uniform(0.7, 0.95),
                            "probability": random.uniform(0.6, 0.9),
                            "severity": "medium",
                            "recommended_action": "Schedule cleaning before predicted date",
                        }
                    ]
                }
            )

        # Default response
        return AgentResponse(
            correlation_id=message.message_id,
            source_agent=message.target_agent,
            target_agent=message.source_agent,
            success=True,
            result={"status": "acknowledged"}
        )

    def get_connected_agents(self) -> List[AgentID]:
        """Get connected agent list."""
        return list(self._connected_agents.keys())

    def get_message_log(self) -> List[AgentMessage]:
        """Get message log for testing assertions."""
        return self._message_log


# =============================================================================
# Response Simulators
# =============================================================================


class ResponseSimulator:
    """
    Simulates various response scenarios for integration testing.

    Allows testing error handling, timeouts, and edge cases.
    """

    def __init__(self) -> None:
        """Initialize response simulator."""
        self._scenarios: Dict[str, Dict[str, Any]] = {}
        self._call_counts: Dict[str, int] = defaultdict(int)

    def configure_scenario(
        self,
        name: str,
        response_type: str,  # "success", "error", "timeout", "partial"
        delay_seconds: float = 0.0,
        error_rate: float = 0.0,
        error_message: Optional[str] = None,
        response_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Configure a response scenario."""
        self._scenarios[name] = {
            "response_type": response_type,
            "delay": delay_seconds,
            "error_rate": error_rate,
            "error_message": error_message,
            "response_data": response_data,
        }

    async def simulate(
        self,
        scenario_name: str,
        default_response: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Simulate a response based on configured scenario.

        Args:
            scenario_name: Name of scenario to simulate
            default_response: Default response if scenario not found

        Returns:
            Simulated response

        Raises:
            ConnectorError: For error scenarios
            TimeoutError: For timeout scenarios
        """
        self._call_counts[scenario_name] += 1

        scenario = self._scenarios.get(scenario_name)
        if not scenario:
            return default_response or {"status": "success"}

        # Apply delay
        if scenario["delay"] > 0:
            await asyncio.sleep(scenario["delay"])

        # Check error rate
        if random.random() < scenario["error_rate"]:
            raise ConnectorError(
                scenario.get("error_message", "Simulated error")
            )

        response_type = scenario["response_type"]

        if response_type == "error":
            raise ConnectorError(
                scenario.get("error_message", "Simulated error")
            )

        elif response_type == "timeout":
            await asyncio.sleep(300)  # Will be cancelled

        elif response_type == "partial":
            # Return partial data
            data = scenario.get("response_data", {})
            return {"partial": True, "data": data}

        # Success
        return scenario.get("response_data", default_response or {"status": "success"})

    def get_call_count(self, scenario_name: str) -> int:
        """Get number of times a scenario was called."""
        return self._call_counts[scenario_name]

    def reset_counts(self) -> None:
        """Reset all call counts."""
        self._call_counts.clear()


# =============================================================================
# Test Fixtures and Utilities
# =============================================================================


class IntegrationTestFixture:
    """
    Test fixture providing pre-configured mock connectors and data.

    Usage:
        async with IntegrationTestFixture() as fixture:
            result = await fixture.historian.get_time_series(...)
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize test fixture."""
        self._generator = HeatExchangerDataGenerator(seed=seed)

        self.historian = MockProcessHistorianConnector(
            data_generator=self._generator
        )
        self.cmms = MockCMSSConnector(
            data_generator=self._generator
        )
        self.dcs = MockDCSConnector(
            data_generator=self._generator
        )
        self.coordinator = MockAgentCoordinator(
            data_generator=self._generator
        )
        self.simulator = ResponseSimulator()

    async def __aenter__(self) -> "IntegrationTestFixture":
        """Set up test fixture."""
        await self.historian.connect()
        await self.cmms.connect()
        await self.dcs.connect()
        await self.coordinator.connect()

        # Add sample data
        await self._setup_sample_data()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Tear down test fixture."""
        await self.historian.disconnect()
        await self.cmms.disconnect()
        await self.dcs.disconnect()
        await self.coordinator.disconnect()

    async def _setup_sample_data(self) -> None:
        """Set up sample data in mock connectors."""
        # Add sample equipment
        for i in range(3):
            equipment = self._generator.generate_equipment()
            self.cmms.add_equipment(equipment)

            # Register tag set
            tag_set = HeatExchangerTagSet(
                equipment_id=equipment.equipment_id,
                hot_inlet_temp_tag=f"{equipment.equipment_id}.TI.HOT_IN",
                hot_outlet_temp_tag=f"{equipment.equipment_id}.TI.HOT_OUT",
                cold_inlet_temp_tag=f"{equipment.equipment_id}.TI.COLD_IN",
                cold_outlet_temp_tag=f"{equipment.equipment_id}.TI.COLD_OUT",
                hot_flow_tag=f"{equipment.equipment_id}.FI.HOT",
                cold_flow_tag=f"{equipment.equipment_id}.FI.COLD",
            )
            self.historian.register_heat_exchanger(tag_set)

    @property
    def generator(self) -> HeatExchangerDataGenerator:
        """Get data generator."""
        return self._generator


def assert_health_check_passed(result: HealthCheckResult) -> None:
    """Assert that a health check passed."""
    assert result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED], \
        f"Health check failed: {result.message}"


def assert_data_quality_passed(
    data: Dict[str, Any],
    min_score: float = 0.6
) -> None:
    """Assert that data quality meets threshold."""
    score = data.get("_quality_score", 1.0)
    assert score >= min_score, \
        f"Data quality {score} below threshold {min_score}"


async def run_integration_test(
    test_func: Callable,
    timeout_seconds: float = 30.0
) -> Any:
    """
    Run an integration test with timeout.

    Args:
        test_func: Async test function
        timeout_seconds: Test timeout

    Returns:
        Test result
    """
    try:
        return await asyncio.wait_for(
            test_func(),
            timeout=timeout_seconds
        )
    except asyncio.TimeoutError:
        raise AssertionError(f"Test timed out after {timeout_seconds}s")
