"""
Integration Test Utilities Module for GL-015 INSULSCAN.

Provides mock connectors, sample data generators, and response simulators
for testing insulation inspection integration components without requiring
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

import numpy as np
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
from .thermal_camera_connector import (
    ThermalCameraProvider,
    CameraInfo,
    CalibrationData,
    RadiometricParameters,
    TemperatureMatrix,
    ThermalImage,
    ThermalFrame,
    ThermalCameraConnectorConfig,
    StreamingConfig,
)
from .cmms_connector import (
    CMSSProvider,
    WorkOrderStatus,
    WorkOrderPriority,
    WorkOrderType,
    InsulationCondition,
    InsulatedEquipment,
    InsulationRepairWorkOrder,
    MaterialRequisition,
    InspectionSchedule,
    RepairWorkOrderCreateRequest,
    CMSSConnectorConfig,
)
from .asset_management_connector import (
    LocationNode,
    LocationHierarchy,
    InsulatedEquipmentAsset,
    InsulationSpecification,
    InsulationMaterialStock,
    TagMapping,
    AssetManagementConnectorConfig,
)
from .weather_connector import (
    WeatherProvider,
    CurrentWeather,
    HourlyForecast,
    DailyForecast,
    InspectionWindow,
    InspectionPlanningReport,
    WeatherConnectorConfig,
)
from .agent_coordinator import (
    AgentID,
    MessageType,
    MessagePriority,
    AgentMessage,
    AgentResponse,
    AgentStatus,
    ThermalEfficiencyContext,
    HeatRecoveryOpportunity,
    HeatExchangerInsulationContext,
    InsulationDefectData,
    AgentCoordinatorConfig,
)
from .data_transformers import (
    UnitConverter,
    TemperatureMatrixNormalizer,
    SchemaMapper,
    ThermalDataQualityScorer,
    ThermalImageProcessor,
    InsulationDataTransformer,
)

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Sample Data Generators
# =============================================================================


class InsulationDataGenerator:
    """Generates realistic sample data for insulation inspection testing."""

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize generator with optional seed for reproducibility."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Define typical insulation parameters by type
        self._insulation_types = {
            "mineral_wool": {
                "thermal_conductivity_range": (0.035, 0.045),  # W/m-K
                "density_range": (40, 200),  # kg/m3
                "max_temperature": 650,  # C
                "typical_thickness_range": (25, 150),  # mm
            },
            "calcium_silicate": {
                "thermal_conductivity_range": (0.055, 0.075),
                "density_range": (200, 350),
                "max_temperature": 1000,
                "typical_thickness_range": (25, 100),
            },
            "cellular_glass": {
                "thermal_conductivity_range": (0.038, 0.055),
                "density_range": (100, 165),
                "max_temperature": 430,
                "typical_thickness_range": (40, 120),
            },
            "perlite": {
                "thermal_conductivity_range": (0.045, 0.065),
                "density_range": (128, 240),
                "max_temperature": 650,
                "typical_thickness_range": (25, 75),
            },
            "aerogel": {
                "thermal_conductivity_range": (0.012, 0.020),
                "density_range": (100, 200),
                "max_temperature": 650,
                "typical_thickness_range": (6, 25),
            },
        }

        # Define defect types and their typical thermal signatures
        self._defect_types = {
            "missing_insulation": {
                "temp_delta_range": (20, 80),  # Above ambient
                "size_range": (0.01, 1.0),  # m2
            },
            "wet_insulation": {
                "temp_delta_range": (5, 25),
                "size_range": (0.05, 2.0),
            },
            "compressed_insulation": {
                "temp_delta_range": (3, 15),
                "size_range": (0.02, 0.5),
            },
            "damaged_cladding": {
                "temp_delta_range": (8, 30),
                "size_range": (0.01, 0.3),
            },
            "thermal_bridge": {
                "temp_delta_range": (10, 40),
                "size_range": (0.001, 0.05),
            },
            "vapor_barrier_failure": {
                "temp_delta_range": (2, 12),
                "size_range": (0.1, 3.0),
            },
        }

        # Equipment types for insulation inspection
        self._equipment_types = [
            "pipe",
            "vessel",
            "tank",
            "heat_exchanger",
            "boiler",
            "column",
            "duct",
            "valve",
        ]

    def generate_temperature_matrix(
        self,
        width: int = 640,
        height: int = 480,
        ambient_temp: float = 25.0,
        hot_spots: int = 0,
        hot_spot_intensity: float = 30.0,
        add_noise: bool = True,
        noise_level: float = 0.5,
    ) -> TemperatureMatrix:
        """
        Generate sample temperature matrix with optional hot spots.

        Args:
            width: Matrix width in pixels
            height: Matrix height in pixels
            ambient_temp: Background ambient temperature (C)
            hot_spots: Number of hot spots to add
            hot_spot_intensity: Temperature increase at hot spots (C)
            add_noise: Add thermal noise
            noise_level: Standard deviation of noise (C)

        Returns:
            Generated TemperatureMatrix
        """
        # Create base temperature field
        data = np.full((height, width), ambient_temp, dtype=np.float32)

        # Add hot spots (simulating insulation defects)
        for _ in range(hot_spots):
            cx = random.randint(50, width - 50)
            cy = random.randint(50, height - 50)
            radius = random.randint(20, 80)

            y_indices, x_indices = np.ogrid[:height, :width]
            distance = np.sqrt((x_indices - cx) ** 2 + (y_indices - cy) ** 2)

            # Gaussian-like hot spot
            intensity = hot_spot_intensity * random.uniform(0.5, 1.5)
            hot_spot_mask = np.exp(-distance ** 2 / (2 * radius ** 2))
            data += intensity * hot_spot_mask

        # Add thermal noise
        if add_noise:
            noise = np.random.normal(0, noise_level, (height, width))
            data += noise.astype(np.float32)

        return TemperatureMatrix(
            data=data,
            width=width,
            height=height,
            min_temp=float(np.min(data)),
            max_temp=float(np.max(data)),
            mean_temp=float(np.mean(data)),
            std_temp=float(np.std(data)),
            unit="celsius",
            timestamp=datetime.utcnow(),
        )

    def generate_thermal_image(
        self,
        equipment_id: Optional[str] = None,
        defect_count: int = 0,
    ) -> ThermalImage:
        """Generate sample thermal image with optional defects."""
        equipment_id = equipment_id or f"EQ-{random.randint(1000, 9999)}"

        # Generate temperature matrix
        temp_matrix = self.generate_temperature_matrix(
            hot_spots=defect_count,
            hot_spot_intensity=random.uniform(15, 50),
        )

        return ThermalImage(
            image_id=str(uuid.uuid4()),
            equipment_id=equipment_id,
            timestamp=datetime.utcnow(),
            temperature_matrix=temp_matrix,
            radiometric_parameters=RadiometricParameters(
                emissivity=0.95,
                reflected_temperature=25.0,
                atmospheric_temperature=25.0,
                distance=3.0,
                relative_humidity=50.0,
            ),
            calibration=CalibrationData(
                calibration_date=datetime.utcnow() - timedelta(days=30),
                calibration_due_date=datetime.utcnow() + timedelta(days=335),
                calibration_temperature_accuracy=0.02,
                calibration_certificate="CAL-2024-001",
            ),
            metadata={
                "camera_model": "FLIR T650sc",
                "lens": "25mm",
                "location": f"Plant Area {random.randint(1, 5)}",
            },
        )

    def generate_camera_info(
        self,
        provider: ThermalCameraProvider = ThermalCameraProvider.FLIR_SYSTEMS,
    ) -> CameraInfo:
        """Generate sample camera information."""
        camera_models = {
            ThermalCameraProvider.FLIR_SYSTEMS: [
                ("FLIR T650sc", 640, 480, -40, 2000),
                ("FLIR T1030sc", 1024, 768, -40, 2000),
                ("FLIR A700", 640, 480, -20, 2000),
            ],
            ThermalCameraProvider.FLUKE_TI_SERIES: [
                ("Fluke Ti480 PRO", 640, 480, -20, 1000),
                ("Fluke TiX580", 640, 480, -20, 1000),
            ],
            ThermalCameraProvider.TESTO: [
                ("testo 890", 640, 480, -30, 1200),
                ("testo 885", 320, 240, -30, 1200),
            ],
            ThermalCameraProvider.OPTRIS_PI: [
                ("Optris PI 640", 640, 480, -20, 900),
                ("Optris PI 1M", 764, 480, 450, 1800),
            ],
            ThermalCameraProvider.INFRATEC_IMAGEIR: [
                ("ImageIR 8300", 640, 512, -40, 1500),
                ("ImageIR 9400", 1280, 1024, -40, 2500),
            ],
        }

        models = camera_models.get(provider, camera_models[ThermalCameraProvider.FLIR_SYSTEMS])
        model, width, height, min_t, max_t = random.choice(models)

        return CameraInfo(
            camera_id=str(uuid.uuid4()),
            provider=provider,
            model=model,
            serial_number=f"SN-{random.randint(100000, 999999)}",
            resolution_width=width,
            resolution_height=height,
            temperature_range_min=min_t,
            temperature_range_max=max_t,
            thermal_sensitivity=0.03,
            frame_rate=30.0,
            firmware_version=f"{random.randint(1, 5)}.{random.randint(0, 9)}.{random.randint(0, 99)}",
            last_calibration=datetime.utcnow() - timedelta(days=random.randint(10, 100)),
            is_radiometric=True,
            supports_streaming=True,
        )

    def generate_insulated_equipment(
        self,
        equipment_id: Optional[str] = None,
        equipment_type: Optional[str] = None,
        insulation_type: Optional[str] = None,
    ) -> InsulatedEquipmentAsset:
        """Generate sample insulated equipment asset."""
        equipment_id = equipment_id or f"EQ-{random.randint(1000, 9999)}"
        equipment_type = equipment_type or random.choice(self._equipment_types)
        insulation_type = insulation_type or random.choice(list(self._insulation_types.keys()))

        insul_params = self._insulation_types[insulation_type]

        return InsulatedEquipmentAsset(
            equipment_id=equipment_id,
            equipment_name=f"{equipment_type.replace('_', ' ').title()} {equipment_id}",
            equipment_type=equipment_type,
            location_id=f"LOC-{random.randint(100, 999)}",
            plant_id="PLANT-001",
            area=random.choice(["Process Area 1", "Utilities", "Tank Farm", "Heat Recovery"]),
            operating_temperature=random.uniform(50, 400),
            design_temperature=random.uniform(400, 600),
            insulation_type=insulation_type,
            insulation_thickness_mm=random.uniform(*insul_params["typical_thickness_range"]),
            thermal_conductivity=random.uniform(*insul_params["thermal_conductivity_range"]),
            cladding_material=random.choice(["Aluminum", "Stainless Steel", "Galvanized Steel"]),
            installation_date=datetime.utcnow() - timedelta(days=random.randint(365, 3650)),
            last_inspection_date=datetime.utcnow() - timedelta(days=random.randint(30, 365)),
            condition=random.choice(list(InsulationCondition)),
            surface_area_m2=random.uniform(5, 500),
            heat_loss_kw=random.uniform(0.5, 50),
        )

    def generate_work_order(
        self,
        equipment_id: str,
        status: WorkOrderStatus = WorkOrderStatus.PENDING,
        defect_type: Optional[str] = None,
    ) -> InsulationRepairWorkOrder:
        """Generate sample insulation repair work order."""
        defect_type = defect_type or random.choice(list(self._defect_types.keys()))
        defect_params = self._defect_types[defect_type]

        return InsulationRepairWorkOrder(
            work_order_id=f"WO-{random.randint(10000, 99999)}",
            work_order_number=f"WO{datetime.utcnow().strftime('%Y%m%d')}{random.randint(100, 999)}",
            title=f"Insulation Repair - {equipment_id}",
            description=f"Repair {defect_type.replace('_', ' ')} on equipment {equipment_id}",
            work_order_type=WorkOrderType.INSULATION_REPAIR,
            priority=random.choice(list(WorkOrderPriority)),
            status=status,
            equipment_id=equipment_id,
            defect_type=defect_type,
            defect_area_m2=random.uniform(*defect_params["size_range"]),
            heat_loss_kw=random.uniform(0.5, 10),
            annual_energy_loss_mwh=random.uniform(4, 90),
            annual_cost_impact=random.uniform(500, 20000),
            insulation_material_required=random.choice(list(self._insulation_types.keys())),
            scheduled_start=datetime.utcnow() + timedelta(days=random.randint(1, 30)),
            estimated_hours=random.uniform(2, 16),
            estimated_cost=random.uniform(500, 10000),
        )

    def generate_inspection_schedule(
        self,
        equipment_id: str,
    ) -> InspectionSchedule:
        """Generate sample inspection schedule."""
        return InspectionSchedule(
            schedule_id=f"SCH-{random.randint(10000, 99999)}",
            equipment_id=equipment_id,
            inspection_type="thermal_survey",
            frequency_days=random.choice([90, 180, 365]),
            last_inspection=datetime.utcnow() - timedelta(days=random.randint(30, 180)),
            next_inspection=datetime.utcnow() + timedelta(days=random.randint(30, 180)),
            assigned_technician=f"TECH-{random.randint(100, 999)}",
            estimated_duration_hours=random.uniform(1, 8),
            notes="Standard thermal insulation inspection",
        )

    def generate_weather_data(
        self,
        latitude: float = 40.7128,
        longitude: float = -74.0060,
    ) -> CurrentWeather:
        """Generate sample weather data."""
        return CurrentWeather(
            timestamp=datetime.utcnow(),
            latitude=latitude,
            longitude=longitude,
            temperature=random.uniform(-10, 35),
            feels_like=random.uniform(-15, 40),
            humidity=random.randint(20, 95),
            pressure=random.uniform(995, 1025),
            wind_speed=random.uniform(0, 25),
            wind_direction=random.randint(0, 360),
            wind_gust=random.uniform(0, 40),
            clouds=random.randint(0, 100),
            visibility=random.randint(1000, 10000),
            precipitation_1h=random.uniform(0, 10) if random.random() > 0.7 else 0,
            weather_main=random.choice(["Clear", "Clouds", "Rain", "Overcast"]),
            weather_description=random.choice([
                "clear sky",
                "few clouds",
                "scattered clouds",
                "overcast",
                "light rain",
            ]),
            sunrise=datetime.utcnow().replace(hour=6, minute=30),
            sunset=datetime.utcnow().replace(hour=18, minute=30),
        )

    def generate_inspection_window(
        self,
        start_time: Optional[datetime] = None,
    ) -> InspectionWindow:
        """Generate sample inspection window."""
        start_time = start_time or datetime.utcnow() + timedelta(hours=random.randint(1, 48))

        return InspectionWindow(
            start_time=start_time,
            end_time=start_time + timedelta(hours=random.randint(2, 6)),
            suitability_score=random.uniform(0.6, 1.0),
            temperature=random.uniform(5, 30),
            humidity=random.randint(30, 70),
            wind_speed=random.uniform(0, 10),
            precipitation_probability=random.uniform(0, 0.2),
            cloud_cover=random.randint(0, 50),
            recommendations=[
                "Good thermal contrast conditions",
                "Low wind suitable for stable readings",
            ],
        )

    def generate_defect_data(
        self,
        equipment_id: str,
        defect_count: int = 1,
    ) -> List[InsulationDefectData]:
        """Generate sample insulation defect data."""
        defects = []

        for i in range(defect_count):
            defect_type = random.choice(list(self._defect_types.keys()))
            defect_params = self._defect_types[defect_type]

            defects.append(InsulationDefectData(
                defect_id=str(uuid.uuid4()),
                equipment_id=equipment_id,
                defect_type=defect_type,
                location_description=f"Section {i + 1}, {random.choice(['North', 'South', 'East', 'West'])} side",
                surface_temperature=random.uniform(30, 100),
                ambient_temperature=random.uniform(15, 30),
                temperature_delta=random.uniform(*defect_params["temp_delta_range"]),
                area_m2=random.uniform(*defect_params["size_range"]),
                heat_loss_kw=random.uniform(0.1, 5),
                severity=random.choice(["low", "medium", "high", "critical"]),
                confidence_score=random.uniform(0.7, 0.99),
                thermal_image_id=str(uuid.uuid4()),
                detected_at=datetime.utcnow() - timedelta(days=random.randint(0, 30)),
            ))

        return defects


# =============================================================================
# Mock Connectors
# =============================================================================


class MockThermalCameraConnector:
    """Mock thermal camera connector for testing."""

    def __init__(
        self,
        provider: ThermalCameraProvider = ThermalCameraProvider.FLIR_SYSTEMS,
        data_generator: Optional[InsulationDataGenerator] = None,
    ) -> None:
        """Initialize mock connector."""
        self._provider = provider
        self._generator = data_generator or InsulationDataGenerator()
        self._state = ConnectionState.DISCONNECTED
        self._camera_info: Optional[CameraInfo] = None
        self._streaming_active = False

        # Response delays for realistic simulation
        self._min_latency_ms = 10
        self._max_latency_ms = 100

    @property
    def state(self) -> ConnectionState:
        return self._state

    @property
    def camera_info(self) -> Optional[CameraInfo]:
        return self._camera_info

    async def connect(self) -> None:
        """Simulate connection."""
        await asyncio.sleep(random.uniform(0.1, 0.5))
        self._state = ConnectionState.CONNECTED
        self._camera_info = self._generator.generate_camera_info(self._provider)
        logger.info(f"Mock {self._provider.value} camera connected")

    async def disconnect(self) -> None:
        """Simulate disconnection."""
        await asyncio.sleep(0.05)
        self._streaming_active = False
        self._state = ConnectionState.DISCONNECTED
        logger.info("Mock thermal camera disconnected")

    async def health_check(self) -> HealthCheckResult:
        """Return mock health check result."""
        return HealthCheckResult(
            status=HealthStatus.HEALTHY if self._state == ConnectionState.CONNECTED else HealthStatus.UNHEALTHY,
            latency_ms=random.uniform(self._min_latency_ms, self._max_latency_ms),
            message="Mock thermal camera healthy",
        )

    async def capture_image(
        self,
        equipment_id: Optional[str] = None,
    ) -> ThermalImage:
        """Capture simulated thermal image."""
        await asyncio.sleep(random.uniform(
            self._min_latency_ms / 1000,
            self._max_latency_ms / 1000,
        ))

        defect_count = random.choice([0, 0, 1, 1, 2, 3])  # Weighted toward fewer defects
        return self._generator.generate_thermal_image(
            equipment_id=equipment_id,
            defect_count=defect_count,
        )

    async def start_streaming(
        self,
        config: StreamingConfig,
    ) -> str:
        """Start simulated streaming."""
        await asyncio.sleep(0.1)
        self._streaming_active = True
        stream_id = str(uuid.uuid4())
        logger.info(f"Mock streaming started: {stream_id}")
        return stream_id

    async def stop_streaming(self, stream_id: str) -> None:
        """Stop simulated streaming."""
        await asyncio.sleep(0.05)
        self._streaming_active = False
        logger.info(f"Mock streaming stopped: {stream_id}")

    async def get_frame(self) -> Optional[ThermalFrame]:
        """Get simulated frame from stream."""
        if not self._streaming_active:
            return None

        await asyncio.sleep(1.0 / 30.0)  # 30 FPS

        temp_matrix = self._generator.generate_temperature_matrix()
        return ThermalFrame(
            frame_number=random.randint(1, 1000000),
            timestamp=datetime.utcnow(),
            temperature_matrix=temp_matrix,
        )

    async def get_calibration(self) -> CalibrationData:
        """Get simulated calibration data."""
        await asyncio.sleep(0.05)
        return CalibrationData(
            calibration_date=datetime.utcnow() - timedelta(days=random.randint(10, 100)),
            calibration_due_date=datetime.utcnow() + timedelta(days=random.randint(200, 350)),
            calibration_temperature_accuracy=0.02,
            calibration_certificate=f"CAL-{datetime.utcnow().year}-{random.randint(100, 999)}",
        )


class MockCMSSConnector:
    """Mock CMMS connector for testing insulation work orders."""

    def __init__(
        self,
        provider: CMSSProvider = CMSSProvider.SAP_PM,
        data_generator: Optional[InsulationDataGenerator] = None,
    ) -> None:
        """Initialize mock connector."""
        self._provider = provider
        self._generator = data_generator or InsulationDataGenerator()
        self._state = ConnectionState.DISCONNECTED

        # In-memory storage
        self._equipment: Dict[str, InsulatedEquipmentAsset] = {}
        self._work_orders: Dict[str, InsulationRepairWorkOrder] = {}
        self._schedules: Dict[str, InspectionSchedule] = {}
        self._materials: Dict[str, MaterialRequisition] = {}

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
            message="Mock CMMS healthy",
        )

    def add_equipment(self, equipment: InsulatedEquipmentAsset) -> None:
        """Add equipment to mock database."""
        self._equipment[equipment.equipment_id] = equipment

    async def get_equipment(self, equipment_id: str) -> Optional[InsulatedEquipmentAsset]:
        """Get equipment from mock database."""
        await asyncio.sleep(0.05)

        if equipment_id in self._equipment:
            return self._equipment[equipment_id]

        # Generate if not exists
        equipment = self._generator.generate_insulated_equipment(equipment_id)
        self._equipment[equipment_id] = equipment
        return equipment

    async def list_equipment(
        self,
        condition: Optional[InsulationCondition] = None,
    ) -> List[InsulatedEquipmentAsset]:
        """List all equipment."""
        await asyncio.sleep(0.1)

        if not self._equipment:
            # Generate some sample equipment
            for _ in range(5):
                equip = self._generator.generate_insulated_equipment()
                self._equipment[equip.equipment_id] = equip

        equipment_list = list(self._equipment.values())
        if condition:
            equipment_list = [e for e in equipment_list if e.condition == condition]

        return equipment_list

    async def create_repair_work_order(
        self,
        request: RepairWorkOrderCreateRequest,
    ) -> InsulationRepairWorkOrder:
        """Create mock repair work order."""
        await asyncio.sleep(0.1)

        work_order = InsulationRepairWorkOrder(
            work_order_id=f"WO-{random.randint(10000, 99999)}",
            work_order_number=f"WO{datetime.utcnow().strftime('%Y%m%d')}{random.randint(100, 999)}",
            title=request.title,
            description=request.description,
            equipment_id=request.equipment_id,
            priority=request.priority,
            status=WorkOrderStatus.PENDING,
            work_order_type=WorkOrderType.INSULATION_REPAIR,
            defect_type=request.defect_type,
            defect_area_m2=request.defect_area_m2,
            heat_loss_kw=request.heat_loss_kw,
            insulation_material_required=request.insulation_material_required,
            scheduled_start=request.scheduled_start,
            estimated_hours=request.estimated_hours,
        )

        self._work_orders[work_order.work_order_id] = work_order
        return work_order

    async def get_work_order(self, work_order_id: str) -> Optional[InsulationRepairWorkOrder]:
        """Get work order."""
        await asyncio.sleep(0.05)
        return self._work_orders.get(work_order_id)

    async def list_work_orders(
        self,
        equipment_id: Optional[str] = None,
        status: Optional[WorkOrderStatus] = None,
    ) -> List[InsulationRepairWorkOrder]:
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
        if status:
            orders = [o for o in orders if o.status == status]

        return orders

    async def get_inspection_schedule(
        self,
        equipment_id: str,
    ) -> Optional[InspectionSchedule]:
        """Get inspection schedule."""
        await asyncio.sleep(0.05)

        if equipment_id not in self._schedules:
            self._schedules[equipment_id] = self._generator.generate_inspection_schedule(
                equipment_id
            )

        return self._schedules[equipment_id]

    async def update_inspection_schedule(
        self,
        schedule: InspectionSchedule,
    ) -> InspectionSchedule:
        """Update inspection schedule."""
        await asyncio.sleep(0.05)
        self._schedules[schedule.equipment_id] = schedule
        return schedule


class MockAssetManagementConnector:
    """Mock asset management connector for testing."""

    def __init__(
        self,
        data_generator: Optional[InsulationDataGenerator] = None,
    ) -> None:
        """Initialize mock connector."""
        self._generator = data_generator or InsulationDataGenerator()
        self._state = ConnectionState.DISCONNECTED

        # In-memory storage
        self._assets: Dict[str, InsulatedEquipmentAsset] = {}
        self._locations: Dict[str, LocationNode] = {}
        self._materials: Dict[str, InsulationMaterialStock] = {}
        self._tag_mappings: Dict[str, TagMapping] = {}

    @property
    def state(self) -> ConnectionState:
        return self._state

    async def connect(self) -> None:
        """Simulate connection."""
        await asyncio.sleep(0.2)
        self._state = ConnectionState.CONNECTED
        logger.info("Mock asset management connected")

    async def disconnect(self) -> None:
        """Simulate disconnection."""
        self._state = ConnectionState.DISCONNECTED

    async def health_check(self) -> HealthCheckResult:
        """Return mock health check."""
        return HealthCheckResult(
            status=HealthStatus.HEALTHY if self._state == ConnectionState.CONNECTED else HealthStatus.UNHEALTHY,
            message="Mock asset management healthy",
        )

    async def get_asset(self, asset_id: str) -> Optional[InsulatedEquipmentAsset]:
        """Get asset by ID."""
        await asyncio.sleep(0.05)

        if asset_id in self._assets:
            return self._assets[asset_id]

        # Generate if not exists
        asset = self._generator.generate_insulated_equipment(asset_id)
        self._assets[asset_id] = asset
        return asset

    async def list_assets(
        self,
        location_id: Optional[str] = None,
        equipment_type: Optional[str] = None,
    ) -> List[InsulatedEquipmentAsset]:
        """List assets."""
        await asyncio.sleep(0.1)

        if not self._assets:
            # Generate sample assets
            for _ in range(10):
                asset = self._generator.generate_insulated_equipment()
                self._assets[asset.equipment_id] = asset

        assets = list(self._assets.values())
        if location_id:
            assets = [a for a in assets if a.location_id == location_id]
        if equipment_type:
            assets = [a for a in assets if a.equipment_type == equipment_type]

        return assets

    async def get_location_hierarchy(
        self,
        root_id: Optional[str] = None,
    ) -> LocationHierarchy:
        """Get location hierarchy."""
        await asyncio.sleep(0.1)

        # Generate sample hierarchy
        root = LocationNode(
            location_id=root_id or "PLANT-001",
            name="Main Plant",
            level=0,
            children=[
                LocationNode(
                    location_id="AREA-001",
                    name="Process Area 1",
                    level=1,
                    parent_id=root_id or "PLANT-001",
                    children=[],
                ),
                LocationNode(
                    location_id="AREA-002",
                    name="Utilities",
                    level=1,
                    parent_id=root_id or "PLANT-001",
                    children=[],
                ),
                LocationNode(
                    location_id="AREA-003",
                    name="Tank Farm",
                    level=1,
                    parent_id=root_id or "PLANT-001",
                    children=[],
                ),
            ],
        )

        return LocationHierarchy(
            root=root,
            total_nodes=4,
            max_depth=2,
        )

    async def get_insulation_material_inventory(
        self,
        material_type: Optional[str] = None,
    ) -> List[InsulationMaterialStock]:
        """Get insulation material inventory."""
        await asyncio.sleep(0.1)

        if not self._materials:
            # Generate sample inventory
            for insul_type in self._generator._insulation_types.keys():
                stock = InsulationMaterialStock(
                    material_id=f"MAT-{random.randint(1000, 9999)}",
                    material_type=insul_type,
                    description=f"{insul_type.replace('_', ' ').title()} insulation",
                    quantity_m3=random.uniform(10, 500),
                    unit_cost=random.uniform(50, 500),
                    location="Warehouse A",
                    reorder_level_m3=random.uniform(5, 50),
                    supplier=random.choice(["Owens Corning", "Rockwool", "Johns Manville"]),
                )
                self._materials[stock.material_id] = stock

        materials = list(self._materials.values())
        if material_type:
            materials = [m for m in materials if m.material_type == material_type]

        return materials


class MockWeatherConnector:
    """Mock weather connector for testing."""

    def __init__(
        self,
        provider: WeatherProvider = WeatherProvider.OPENWEATHERMAP,
        data_generator: Optional[InsulationDataGenerator] = None,
    ) -> None:
        """Initialize mock connector."""
        self._provider = provider
        self._generator = data_generator or InsulationDataGenerator()
        self._state = ConnectionState.DISCONNECTED

    @property
    def state(self) -> ConnectionState:
        return self._state

    async def connect(self) -> None:
        """Simulate connection."""
        await asyncio.sleep(0.1)
        self._state = ConnectionState.CONNECTED
        logger.info(f"Mock {self._provider.value} weather connector connected")

    async def disconnect(self) -> None:
        """Simulate disconnection."""
        self._state = ConnectionState.DISCONNECTED

    async def health_check(self) -> HealthCheckResult:
        """Return mock health check."""
        return HealthCheckResult(
            status=HealthStatus.HEALTHY if self._state == ConnectionState.CONNECTED else HealthStatus.UNHEALTHY,
            message="Mock weather connector healthy",
        )

    async def get_current_weather(
        self,
        latitude: float,
        longitude: float,
    ) -> CurrentWeather:
        """Get current weather."""
        await asyncio.sleep(0.05)
        return self._generator.generate_weather_data(latitude, longitude)

    async def get_hourly_forecast(
        self,
        latitude: float,
        longitude: float,
        hours: int = 24,
    ) -> List[HourlyForecast]:
        """Get hourly forecast."""
        await asyncio.sleep(0.1)

        forecasts = []
        base_time = datetime.utcnow()

        for i in range(hours):
            weather = self._generator.generate_weather_data(latitude, longitude)
            forecasts.append(HourlyForecast(
                timestamp=base_time + timedelta(hours=i),
                temperature=weather.temperature + random.uniform(-3, 3),
                feels_like=weather.feels_like + random.uniform(-3, 3),
                humidity=weather.humidity + random.randint(-10, 10),
                wind_speed=weather.wind_speed + random.uniform(-2, 2),
                precipitation_probability=random.uniform(0, 1),
                weather_main=weather.weather_main,
            ))

        return forecasts

    async def get_inspection_windows(
        self,
        latitude: float,
        longitude: float,
        start_date: datetime,
        end_date: datetime,
        min_suitability: float = 0.7,
    ) -> List[InspectionWindow]:
        """Get suitable inspection windows."""
        await asyncio.sleep(0.2)

        windows = []
        current = start_date

        while current < end_date:
            window = self._generator.generate_inspection_window(current)
            if window.suitability_score >= min_suitability:
                windows.append(window)

            current += timedelta(hours=random.randint(4, 12))

        return windows[:10]  # Return top 10 windows

    async def get_inspection_planning_report(
        self,
        latitude: float,
        longitude: float,
        planning_days: int = 7,
    ) -> InspectionPlanningReport:
        """Get inspection planning report."""
        await asyncio.sleep(0.2)

        windows = await self.get_inspection_windows(
            latitude,
            longitude,
            datetime.utcnow(),
            datetime.utcnow() + timedelta(days=planning_days),
        )

        return InspectionPlanningReport(
            location_latitude=latitude,
            location_longitude=longitude,
            planning_period_start=datetime.utcnow(),
            planning_period_end=datetime.utcnow() + timedelta(days=planning_days),
            suitable_windows=windows,
            total_suitable_hours=sum(
                (w.end_time - w.start_time).total_seconds() / 3600
                for w in windows
            ),
            best_window=windows[0] if windows else None,
            overall_suitability=random.uniform(0.6, 0.9),
            recommendations=[
                "Schedule inspections during morning hours for best thermal contrast",
                "Avoid inspection during rain forecasts",
            ],
        )


class MockAgentCoordinator:
    """Mock agent coordinator for testing."""

    def __init__(
        self,
        data_generator: Optional[InsulationDataGenerator] = None,
    ) -> None:
        """Initialize mock coordinator."""
        self._generator = data_generator or InsulationDataGenerator()
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
        for agent_id in [AgentID.GL_001_THERMOSYNC, AgentID.GL_006_HEATRECLAIM, AgentID.GL_014_EXCHANGER_PRO]:
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
        wait_for_response: bool = True,
    ) -> Optional[AgentResponse]:
        """Simulate sending message and receiving response."""
        await asyncio.sleep(random.uniform(0.05, 0.2))

        self._message_log.append(message)

        if not wait_for_response:
            return None

        # Generate mock response based on target agent and action
        if message.target_agent == AgentID.GL_001_THERMOSYNC:
            return self._generate_thermosync_response(message)
        elif message.target_agent == AgentID.GL_006_HEATRECLAIM:
            return self._generate_heatreclaim_response(message)
        elif message.target_agent == AgentID.GL_014_EXCHANGER_PRO:
            return self._generate_exchangerpro_response(message)

        # Default response
        return AgentResponse(
            correlation_id=message.message_id,
            source_agent=message.target_agent,
            target_agent=message.source_agent,
            success=True,
            result={"status": "acknowledged"},
        )

    def _generate_thermosync_response(self, message: AgentMessage) -> AgentResponse:
        """Generate GL-001 THERMOSYNC response."""
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
                    "efficiency_trend": random.choice(["improving", "stable", "degrading"]),
                },
            )
        elif message.action == "report_insulation_defect":
            return AgentResponse(
                correlation_id=message.message_id,
                source_agent=message.target_agent,
                target_agent=message.source_agent,
                success=True,
                result={
                    "acknowledged": True,
                    "impact_assessment": {
                        "efficiency_impact": random.uniform(0.01, 0.05),
                        "priority": random.choice(["low", "medium", "high"]),
                    },
                },
            )

        return AgentResponse(
            correlation_id=message.message_id,
            source_agent=message.target_agent,
            target_agent=message.source_agent,
            success=True,
            result={"status": "acknowledged"},
        )

    def _generate_heatreclaim_response(self, message: AgentMessage) -> AgentResponse:
        """Generate GL-006 HEATRECLAIM response."""
        if message.action == "get_heat_recovery_opportunities":
            return AgentResponse(
                correlation_id=message.message_id,
                source_agent=message.target_agent,
                target_agent=message.source_agent,
                success=True,
                result={
                    "opportunities": [
                        {
                            "opportunity_id": str(uuid.uuid4()),
                            "source_equipment_id": message.payload.get("equipment_id", "EQ-001"),
                            "recoverable_heat_kw": random.uniform(10, 100),
                            "annual_savings": random.uniform(10000, 100000),
                            "insulation_improvement_potential": random.uniform(0.1, 0.3),
                        }
                    ],
                },
            )

        return AgentResponse(
            correlation_id=message.message_id,
            source_agent=message.target_agent,
            target_agent=message.source_agent,
            success=True,
            result={"status": "acknowledged"},
        )

    def _generate_exchangerpro_response(self, message: AgentMessage) -> AgentResponse:
        """Generate GL-014 EXCHANGER-PRO response."""
        if message.action == "get_exchanger_insulation_status":
            return AgentResponse(
                correlation_id=message.message_id,
                source_agent=message.target_agent,
                target_agent=message.source_agent,
                success=True,
                result={
                    "equipment_id": message.payload.get("equipment_id"),
                    "insulation_condition": random.choice(["good", "fair", "poor"]),
                    "shell_surface_temp": random.uniform(30, 80),
                    "recommended_insulation_thickness": random.uniform(50, 150),
                    "heat_loss_estimate_kw": random.uniform(1, 20),
                },
            )

        return AgentResponse(
            correlation_id=message.message_id,
            source_agent=message.target_agent,
            target_agent=message.source_agent,
            success=True,
            result={"status": "acknowledged"},
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
        response_data: Optional[Dict[str, Any]] = None,
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
        default_response: Optional[Dict[str, Any]] = None,
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
            image = await fixture.thermal_camera.capture_image()
            equipment = await fixture.cmms.get_equipment("EQ-001")
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize test fixture."""
        self._generator = InsulationDataGenerator(seed=seed)

        self.thermal_camera = MockThermalCameraConnector(
            data_generator=self._generator,
        )
        self.cmms = MockCMSSConnector(
            data_generator=self._generator,
        )
        self.asset_management = MockAssetManagementConnector(
            data_generator=self._generator,
        )
        self.weather = MockWeatherConnector(
            data_generator=self._generator,
        )
        self.coordinator = MockAgentCoordinator(
            data_generator=self._generator,
        )
        self.simulator = ResponseSimulator()

    async def __aenter__(self) -> "IntegrationTestFixture":
        """Set up test fixture."""
        await self.thermal_camera.connect()
        await self.cmms.connect()
        await self.asset_management.connect()
        await self.weather.connect()
        await self.coordinator.connect()

        # Add sample data
        await self._setup_sample_data()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Tear down test fixture."""
        await self.thermal_camera.disconnect()
        await self.cmms.disconnect()
        await self.asset_management.disconnect()
        await self.weather.disconnect()
        await self.coordinator.disconnect()

    async def _setup_sample_data(self) -> None:
        """Set up sample data in mock connectors."""
        # Add sample equipment to CMMS
        for _ in range(5):
            equipment = self._generator.generate_insulated_equipment()
            self.cmms.add_equipment(equipment)

    @property
    def generator(self) -> InsulationDataGenerator:
        """Get data generator."""
        return self._generator


def assert_health_check_passed(result: HealthCheckResult) -> None:
    """Assert that a health check passed."""
    assert result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED], \
        f"Health check failed: {result.message}"


def assert_data_quality_passed(
    data: Dict[str, Any],
    min_score: float = 0.6,
) -> None:
    """Assert that data quality meets threshold."""
    score = data.get("_quality_score", 1.0)
    assert score >= min_score, \
        f"Data quality {score} below threshold {min_score}"


def assert_temperature_in_range(
    temperature: float,
    min_temp: float = -50.0,
    max_temp: float = 500.0,
) -> None:
    """Assert that temperature is within valid range."""
    assert min_temp <= temperature <= max_temp, \
        f"Temperature {temperature} outside valid range [{min_temp}, {max_temp}]"


def assert_defect_valid(defect: InsulationDefectData) -> None:
    """Assert that defect data is valid."""
    assert defect.defect_id, "Defect ID is required"
    assert defect.equipment_id, "Equipment ID is required"
    assert defect.defect_type, "Defect type is required"
    assert defect.temperature_delta > 0, "Temperature delta must be positive"
    assert defect.area_m2 > 0, "Defect area must be positive"
    assert 0 <= defect.confidence_score <= 1, "Confidence score must be between 0 and 1"


async def run_integration_test(
    test_func: Callable,
    timeout_seconds: float = 30.0,
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
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        raise AssertionError(f"Test timed out after {timeout_seconds}s")


async def run_connector_health_check_suite(
    fixture: IntegrationTestFixture,
) -> Dict[str, HealthCheckResult]:
    """
    Run health checks on all connectors in fixture.

    Args:
        fixture: Test fixture with mock connectors

    Returns:
        Dict of connector name to health check result
    """
    results = {}

    results["thermal_camera"] = await fixture.thermal_camera.health_check()
    results["cmms"] = await fixture.cmms.health_check()
    results["asset_management"] = await fixture.asset_management.health_check()
    results["weather"] = await fixture.weather.health_check()

    return results


# =============================================================================
# Sample Test Cases
# =============================================================================


async def sample_thermal_image_capture_test() -> None:
    """Sample test for thermal image capture."""
    async with IntegrationTestFixture(seed=42) as fixture:
        # Capture thermal image
        image = await fixture.thermal_camera.capture_image("EQ-001")

        # Validate image
        assert image.image_id, "Image ID is required"
        assert image.equipment_id == "EQ-001"
        assert image.temperature_matrix is not None
        assert image.temperature_matrix.width == 640
        assert image.temperature_matrix.height == 480

        # Check temperature range is valid
        assert_temperature_in_range(image.temperature_matrix.min_temp)
        assert_temperature_in_range(image.temperature_matrix.max_temp)


async def sample_work_order_creation_test() -> None:
    """Sample test for work order creation."""
    async with IntegrationTestFixture(seed=42) as fixture:
        # Create repair work order request
        request = RepairWorkOrderCreateRequest(
            title="Repair missing insulation on pipe EQ-001",
            description="Thermal survey detected missing insulation section",
            equipment_id="EQ-001",
            priority=WorkOrderPriority.HIGH,
            defect_type="missing_insulation",
            defect_area_m2=0.5,
            heat_loss_kw=5.0,
            insulation_material_required="mineral_wool",
            scheduled_start=datetime.utcnow() + timedelta(days=3),
            estimated_hours=4.0,
        )

        # Create work order
        work_order = await fixture.cmms.create_repair_work_order(request)

        # Validate work order
        assert work_order.work_order_id, "Work order ID is required"
        assert work_order.status == WorkOrderStatus.PENDING
        assert work_order.equipment_id == "EQ-001"


async def sample_agent_coordination_test() -> None:
    """Sample test for agent coordination."""
    async with IntegrationTestFixture(seed=42) as fixture:
        # Send message to GL-001 THERMOSYNC
        message = AgentMessage(
            message_id=str(uuid.uuid4()),
            source_agent=AgentID.GL_015_INSULSCAN,
            target_agent=AgentID.GL_001_THERMOSYNC,
            message_type=MessageType.REQUEST,
            action="report_insulation_defect",
            payload={
                "equipment_id": "EQ-001",
                "defect_type": "missing_insulation",
                "heat_loss_kw": 5.0,
            },
        )

        response = await fixture.coordinator.send_message(message)

        # Validate response
        assert response is not None
        assert response.success
        assert response.result.get("acknowledged")
