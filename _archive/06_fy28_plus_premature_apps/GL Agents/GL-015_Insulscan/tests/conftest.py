# -*- coding: utf-8 -*-
"""
GL-015 INSULSCAN - Comprehensive Test Fixtures

Pytest fixtures for testing the Insulation Scanning & Thermal Assessment Agent.
Provides mock data, test helpers, and shared configurations for:
- Heat loss calculations (ASTM C680 methodology)
- Thermal resistance calculations
- Condition scoring algorithms
- ROI and payback period calculations
- Provenance tracking and audit trails

Author: GL-TestEngineer
Version: 1.0.0
Target Coverage: 85%+
"""

import pytest
import numpy as np
import hashlib
import uuid
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import Mock, AsyncMock, MagicMock
from enum import Enum


# =============================================================================
# TEST CONFIGURATION
# =============================================================================

TEST_CONFIG = {
    "coverage_target": 0.85,
    "performance_threshold_ms": 100,
    "async_timeout_seconds": 30,
    "random_seed": 42,
    "float_tolerance": 1e-6,
    "thermal_tolerance_K": 0.1,
    "heat_loss_tolerance_percent": 2.0,
    "condition_score_tolerance": 1.0,
}


# =============================================================================
# ENUMERATIONS
# =============================================================================

class InsulationType(Enum):
    """Types of industrial insulation materials."""
    MINERAL_WOOL = "mineral_wool"
    CALCIUM_SILICATE = "calcium_silicate"
    CELLULAR_GLASS = "cellular_glass"
    PERLITE = "perlite"
    AEROGEL = "aerogel"
    FIBERGLASS = "fiberglass"
    POLYURETHANE_FOAM = "polyurethane_foam"
    PHENOLIC_FOAM = "phenolic_foam"
    CERAMIC_FIBER = "ceramic_fiber"


class PipeGeometry(Enum):
    """Pipe geometry types for heat loss calculations."""
    CYLINDRICAL = "cylindrical"
    FLAT = "flat"
    SPHERICAL = "spherical"


class ConditionGrade(Enum):
    """Condition grades for insulation assessment."""
    EXCELLENT = "excellent"  # Score 90-100
    GOOD = "good"            # Score 70-89
    FAIR = "fair"            # Score 50-69
    POOR = "poor"            # Score 30-49
    CRITICAL = "critical"    # Score 0-29


class DamageType(Enum):
    """Types of insulation damage."""
    MOISTURE_INGRESS = "moisture_ingress"
    MECHANICAL_DAMAGE = "mechanical_damage"
    THERMAL_DEGRADATION = "thermal_degradation"
    CORROSION_UNDER_INSULATION = "cui"
    MISSING_SECTIONS = "missing_sections"
    JACKET_FAILURE = "jacket_failure"
    COMPRESSION = "compression"
    NONE = "none"


class DataQuality(Enum):
    """Data quality levels for thermal measurements."""
    GOOD = "good"
    DEGRADED = "degraded"
    BAD = "bad"
    UNCERTAIN = "uncertain"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class InsulationMaterial:
    """Insulation material properties."""
    material_type: InsulationType
    thermal_conductivity_W_mK: float  # W/(m*K) at reference temp
    reference_temperature_C: float
    temperature_coefficient: float  # k = k0 * (1 + coef * (T - T_ref))
    density_kg_m3: float
    max_service_temp_C: float
    min_service_temp_C: float
    moisture_resistance: float  # 0-1 scale
    cost_per_m3_usd: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InsulationAsset:
    """Insulation asset definition for a pipe or equipment section."""
    asset_id: str
    asset_name: str
    location: str
    # Geometry
    pipe_outer_diameter_m: float
    insulation_thickness_m: float
    length_m: float
    geometry: PipeGeometry
    # Material
    material: InsulationMaterial
    # Operating conditions
    process_temperature_C: float
    ambient_temperature_C: float
    wind_speed_m_s: float
    # Installation info
    installation_date: datetime
    last_inspection_date: Optional[datetime]
    # Condition
    current_condition: ConditionGrade
    damage_types: List[DamageType] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThermalMeasurement:
    """Thermal measurement from IR camera or temperature sensor."""
    asset_id: str
    timestamp: datetime
    # Surface temperatures
    surface_temp_C: float
    ambient_temp_C: float
    process_temp_C: float
    # Measurement quality
    emissivity: float
    reflected_temp_C: float
    humidity_percent: float
    data_quality: DataQuality
    # Calculated values
    heat_flux_W_m2: Optional[float] = None
    thermal_anomaly_detected: bool = False
    anomaly_severity: float = 0.0  # 0-1 scale
    # Provenance
    measurement_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    provenance_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate provenance hash."""
        if not self.provenance_hash:
            content = (
                f"{self.asset_id}|{self.timestamp.isoformat()}|"
                f"{self.surface_temp_C:.6f}|{self.ambient_temp_C:.6f}"
            )
            self.provenance_hash = hashlib.sha256(content.encode()).hexdigest()


@dataclass
class HeatLossResult:
    """Result of heat loss calculation."""
    asset_id: str
    timestamp: datetime
    # Heat loss values
    heat_loss_W_m: float  # Watts per meter length
    heat_loss_total_W: float  # Total watts
    heat_loss_bare_W_m: float  # Heat loss if uninsulated
    insulation_efficiency_percent: float
    # Thermal resistance
    R_insulation_m2K_W: float
    R_surface_m2K_W: float
    R_total_m2K_W: float
    # Temperatures
    surface_temp_calculated_C: float
    # Economic impact
    energy_loss_kWh_year: float
    cost_loss_usd_year: float
    # Provenance
    calculation_method: str
    provenance_hash: str = ""
    computation_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate provenance hash."""
        if not self.provenance_hash:
            content = (
                f"{self.asset_id}|{self.timestamp.isoformat()}|"
                f"Q:{self.heat_loss_W_m:.6f}|R:{self.R_total_m2K_W:.6f}"
            )
            self.provenance_hash = hashlib.sha256(content.encode()).hexdigest()


@dataclass
class ConditionAssessment:
    """Condition assessment result for an insulation asset."""
    asset_id: str
    timestamp: datetime
    # Condition scores (0-100)
    overall_score: float
    thermal_performance_score: float
    physical_condition_score: float
    moisture_score: float
    age_factor_score: float
    # Grade
    condition_grade: ConditionGrade
    # Findings
    damage_detected: List[DamageType]
    recommendations: List[str]
    priority: str  # "low", "medium", "high", "critical"
    # Provenance
    provenance_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ROICalculation:
    """ROI calculation for insulation replacement/repair."""
    asset_id: str
    timestamp: datetime
    # Costs
    repair_cost_usd: float
    replacement_cost_usd: float
    current_energy_loss_usd_year: float
    projected_energy_loss_usd_year: float
    # Savings
    annual_savings_usd: float
    # Payback periods
    repair_payback_years: float
    replacement_payback_years: float
    # NPV analysis
    npv_repair_usd: float
    npv_replacement_usd: float
    discount_rate: float
    analysis_period_years: int
    # Recommendation
    recommended_action: str
    confidence_score: float
    provenance_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# INSULATION MATERIAL FIXTURES
# =============================================================================

@pytest.fixture
def mineral_wool_material() -> InsulationMaterial:
    """Create mineral wool insulation material."""
    return InsulationMaterial(
        material_type=InsulationType.MINERAL_WOOL,
        thermal_conductivity_W_mK=0.040,
        reference_temperature_C=24.0,
        temperature_coefficient=0.0002,  # k increases with temperature
        density_kg_m3=100.0,
        max_service_temp_C=650.0,
        min_service_temp_C=-40.0,
        moisture_resistance=0.6,
        cost_per_m3_usd=150.0,
        metadata={"standard": "ASTM C547", "grade": "industrial"},
    )


@pytest.fixture
def calcium_silicate_material() -> InsulationMaterial:
    """Create calcium silicate insulation material."""
    return InsulationMaterial(
        material_type=InsulationType.CALCIUM_SILICATE,
        thermal_conductivity_W_mK=0.055,
        reference_temperature_C=100.0,
        temperature_coefficient=0.00015,
        density_kg_m3=240.0,
        max_service_temp_C=650.0,
        min_service_temp_C=-18.0,
        moisture_resistance=0.4,
        cost_per_m3_usd=280.0,
        metadata={"standard": "ASTM C533"},
    )


@pytest.fixture
def cellular_glass_material() -> InsulationMaterial:
    """Create cellular glass insulation material."""
    return InsulationMaterial(
        material_type=InsulationType.CELLULAR_GLASS,
        thermal_conductivity_W_mK=0.048,
        reference_temperature_C=24.0,
        temperature_coefficient=0.00018,
        density_kg_m3=130.0,
        max_service_temp_C=430.0,
        min_service_temp_C=-268.0,  # Cryogenic applications
        moisture_resistance=0.99,  # Closed cell - excellent
        cost_per_m3_usd=450.0,
        metadata={"standard": "ASTM C552"},
    )


@pytest.fixture
def aerogel_material() -> InsulationMaterial:
    """Create aerogel insulation material (high performance)."""
    return InsulationMaterial(
        material_type=InsulationType.AEROGEL,
        thermal_conductivity_W_mK=0.015,  # Very low k
        reference_temperature_C=24.0,
        temperature_coefficient=0.00025,
        density_kg_m3=120.0,
        max_service_temp_C=650.0,
        min_service_temp_C=-200.0,
        moisture_resistance=0.95,
        cost_per_m3_usd=2500.0,  # Premium material
        metadata={"standard": "ASTM C1728"},
    )


@pytest.fixture
def sample_materials(
    mineral_wool_material,
    calcium_silicate_material,
    cellular_glass_material,
    aerogel_material,
) -> Dict[InsulationType, InsulationMaterial]:
    """Create dictionary of sample materials."""
    return {
        InsulationType.MINERAL_WOOL: mineral_wool_material,
        InsulationType.CALCIUM_SILICATE: calcium_silicate_material,
        InsulationType.CELLULAR_GLASS: cellular_glass_material,
        InsulationType.AEROGEL: aerogel_material,
    }


# =============================================================================
# INSULATION ASSET FIXTURES
# =============================================================================

@pytest.fixture
def sample_pipe_asset(mineral_wool_material) -> InsulationAsset:
    """Create a sample insulated pipe asset."""
    return InsulationAsset(
        asset_id="PIPE-001",
        asset_name="Steam Header A",
        location="Unit 100 - Rack 5",
        pipe_outer_diameter_m=0.2191,  # 8-inch NPS Schedule 40
        insulation_thickness_m=0.076,  # 3 inches
        length_m=50.0,
        geometry=PipeGeometry.CYLINDRICAL,
        material=mineral_wool_material,
        process_temperature_C=175.0,
        ambient_temperature_C=25.0,
        wind_speed_m_s=2.0,
        installation_date=datetime(2018, 6, 15, tzinfo=timezone.utc),
        last_inspection_date=datetime(2024, 6, 15, tzinfo=timezone.utc),
        current_condition=ConditionGrade.GOOD,
        damage_types=[],
        metadata={"service": "steam", "pressure_bar": 10.0},
    )


@pytest.fixture
def damaged_pipe_asset(calcium_silicate_material) -> InsulationAsset:
    """Create a damaged insulated pipe asset."""
    return InsulationAsset(
        asset_id="PIPE-002",
        asset_name="Hot Oil Return",
        location="Unit 200 - Rack 3",
        pipe_outer_diameter_m=0.3239,  # 12-inch NPS
        insulation_thickness_m=0.102,  # 4 inches
        length_m=30.0,
        geometry=PipeGeometry.CYLINDRICAL,
        material=calcium_silicate_material,
        process_temperature_C=280.0,
        ambient_temperature_C=30.0,
        wind_speed_m_s=3.5,
        installation_date=datetime(2010, 3, 20, tzinfo=timezone.utc),
        last_inspection_date=datetime(2023, 9, 10, tzinfo=timezone.utc),
        current_condition=ConditionGrade.POOR,
        damage_types=[DamageType.MOISTURE_INGRESS, DamageType.JACKET_FAILURE],
        metadata={"service": "hot_oil", "needs_attention": True},
    )


@pytest.fixture
def high_temp_asset(calcium_silicate_material) -> InsulationAsset:
    """Create a high-temperature insulated asset."""
    return InsulationAsset(
        asset_id="EQUIP-001",
        asset_name="Furnace Outlet Header",
        location="Furnace Area - Zone A",
        pipe_outer_diameter_m=0.508,  # 20-inch
        insulation_thickness_m=0.152,  # 6 inches
        length_m=15.0,
        geometry=PipeGeometry.CYLINDRICAL,
        material=calcium_silicate_material,
        process_temperature_C=450.0,
        ambient_temperature_C=35.0,
        wind_speed_m_s=1.0,
        installation_date=datetime(2015, 8, 10, tzinfo=timezone.utc),
        last_inspection_date=datetime(2024, 2, 15, tzinfo=timezone.utc),
        current_condition=ConditionGrade.FAIR,
        damage_types=[DamageType.THERMAL_DEGRADATION],
        metadata={"service": "process_gas", "critical": True},
    )


@pytest.fixture
def cryogenic_asset(cellular_glass_material) -> InsulationAsset:
    """Create a cryogenic insulated asset."""
    return InsulationAsset(
        asset_id="CRYO-001",
        asset_name="LNG Transfer Line",
        location="Tank Farm - Area 5",
        pipe_outer_diameter_m=0.406,  # 16-inch
        insulation_thickness_m=0.203,  # 8 inches
        length_m=100.0,
        geometry=PipeGeometry.CYLINDRICAL,
        material=cellular_glass_material,
        process_temperature_C=-160.0,
        ambient_temperature_C=25.0,
        wind_speed_m_s=4.0,
        installation_date=datetime(2020, 11, 5, tzinfo=timezone.utc),
        last_inspection_date=datetime(2024, 5, 20, tzinfo=timezone.utc),
        current_condition=ConditionGrade.EXCELLENT,
        damage_types=[],
        metadata={"service": "LNG", "vapor_barrier": "intact"},
    )


# =============================================================================
# THERMAL MEASUREMENT FIXTURES
# =============================================================================

@pytest.fixture
def sample_thermal_measurement() -> ThermalMeasurement:
    """Create a sample thermal measurement."""
    return ThermalMeasurement(
        asset_id="PIPE-001",
        timestamp=datetime.now(timezone.utc),
        surface_temp_C=45.0,
        ambient_temp_C=25.0,
        process_temp_C=175.0,
        emissivity=0.90,
        reflected_temp_C=25.0,
        humidity_percent=60.0,
        data_quality=DataQuality.GOOD,
        heat_flux_W_m2=120.0,
        thermal_anomaly_detected=False,
        anomaly_severity=0.0,
    )


@pytest.fixture
def anomaly_thermal_measurement() -> ThermalMeasurement:
    """Create a thermal measurement with detected anomaly."""
    return ThermalMeasurement(
        asset_id="PIPE-002",
        timestamp=datetime.now(timezone.utc),
        surface_temp_C=95.0,  # Much higher than expected
        ambient_temp_C=30.0,
        process_temp_C=280.0,
        emissivity=0.85,
        reflected_temp_C=30.0,
        humidity_percent=70.0,
        data_quality=DataQuality.GOOD,
        heat_flux_W_m2=450.0,  # High heat flux
        thermal_anomaly_detected=True,
        anomaly_severity=0.75,
        metadata={"anomaly_type": "hot_spot", "location_m": 12.5},
    )


@pytest.fixture
def measurement_time_series() -> List[ThermalMeasurement]:
    """Create a time series of thermal measurements."""
    base_time = datetime.now(timezone.utc)
    measurements = []

    for i in range(24):
        # Simulate daily temperature variation
        hour_offset = i
        temp_variation = 5.0 * math.sin(2 * math.pi * hour_offset / 24)

        measurement = ThermalMeasurement(
            asset_id="PIPE-001",
            timestamp=base_time - timedelta(hours=23 - i),
            surface_temp_C=45.0 + temp_variation,
            ambient_temp_C=25.0 + temp_variation * 0.5,
            process_temp_C=175.0,
            emissivity=0.90,
            reflected_temp_C=25.0,
            humidity_percent=60.0,
            data_quality=DataQuality.GOOD,
        )
        measurements.append(measurement)

    return measurements


# =============================================================================
# HEAT LOSS CALCULATION REFERENCE DATA
# =============================================================================

@pytest.fixture
def astm_c680_reference_cases() -> List[Dict[str, Any]]:
    """
    Reference cases from ASTM C680 for heat loss validation.

    These are known-good values for validating heat loss calculations.
    """
    return [
        {
            "case_name": "Steam Pipe - 4 inch, 2 inch insulation",
            "pipe_od_m": 0.1143,  # 4-inch NPS
            "insulation_thickness_m": 0.051,  # 2 inches
            "process_temp_C": 175.0,
            "ambient_temp_C": 25.0,
            "k_insulation_W_mK": 0.040,
            "wind_speed_m_s": 0.0,  # Still air
            "expected_heat_loss_W_m": 58.5,  # Reference value
            "tolerance_percent": 5.0,
        },
        {
            "case_name": "Steam Pipe - 8 inch, 3 inch insulation",
            "pipe_od_m": 0.2191,  # 8-inch NPS
            "insulation_thickness_m": 0.076,  # 3 inches
            "process_temp_C": 175.0,
            "ambient_temp_C": 25.0,
            "k_insulation_W_mK": 0.040,
            "wind_speed_m_s": 0.0,
            "expected_heat_loss_W_m": 72.8,
            "tolerance_percent": 5.0,
        },
        {
            "case_name": "High Temp Process - 12 inch, 4 inch insulation",
            "pipe_od_m": 0.3239,  # 12-inch NPS
            "insulation_thickness_m": 0.102,  # 4 inches
            "process_temp_C": 400.0,
            "ambient_temp_C": 30.0,
            "k_insulation_W_mK": 0.055,  # Higher k at high temp
            "wind_speed_m_s": 5.0,
            "expected_heat_loss_W_m": 185.0,
            "tolerance_percent": 8.0,  # Higher tolerance for complex case
        },
        {
            "case_name": "Cryogenic - 6 inch, 6 inch insulation",
            "pipe_od_m": 0.1683,  # 6-inch NPS
            "insulation_thickness_m": 0.152,  # 6 inches
            "process_temp_C": -160.0,  # LNG temperature
            "ambient_temp_C": 25.0,
            "k_insulation_W_mK": 0.045,
            "wind_speed_m_s": 2.0,
            "expected_heat_loss_W_m": -95.0,  # Negative = heat gain
            "tolerance_percent": 10.0,  # Cryogenic is complex
        },
    ]


@pytest.fixture
def thermal_resistance_reference() -> Dict[str, Any]:
    """Reference values for thermal resistance calculations."""
    return {
        # Surface coefficients (W/m2-K)
        "h_still_air_horizontal": 9.3,
        "h_still_air_vertical": 8.3,
        "h_wind_5ms": 25.0,
        "h_wind_10ms": 35.0,
        # Insulation R-values (m2-K/W per inch thickness)
        "R_mineral_wool_per_inch": 0.77,
        "R_calcium_silicate_per_inch": 0.46,
        "R_cellular_glass_per_inch": 0.53,
        "R_aerogel_per_inch": 1.7,
    }


# =============================================================================
# CONDITION SCORING REFERENCE DATA
# =============================================================================

@pytest.fixture
def condition_scoring_thresholds() -> Dict[str, Any]:
    """Thresholds for condition scoring algorithm."""
    return {
        "thermal_performance": {
            "excellent": {"max_efficiency_loss_percent": 5},
            "good": {"max_efficiency_loss_percent": 15},
            "fair": {"max_efficiency_loss_percent": 30},
            "poor": {"max_efficiency_loss_percent": 50},
            "critical": {"max_efficiency_loss_percent": 100},
        },
        "age_factor": {
            "expected_life_years": {
                InsulationType.MINERAL_WOOL: 25,
                InsulationType.CALCIUM_SILICATE: 30,
                InsulationType.CELLULAR_GLASS: 40,
                InsulationType.AEROGEL: 20,
            },
        },
        "damage_severity": {
            DamageType.MOISTURE_INGRESS: 0.3,
            DamageType.MECHANICAL_DAMAGE: 0.2,
            DamageType.THERMAL_DEGRADATION: 0.25,
            DamageType.CORROSION_UNDER_INSULATION: 0.4,
            DamageType.MISSING_SECTIONS: 0.5,
            DamageType.JACKET_FAILURE: 0.15,
            DamageType.COMPRESSION: 0.1,
            DamageType.NONE: 0.0,
        },
    }


# =============================================================================
# ROI CALCULATION REFERENCE DATA
# =============================================================================

@pytest.fixture
def economic_parameters() -> Dict[str, Any]:
    """Economic parameters for ROI calculations."""
    return {
        "energy_cost_usd_per_kWh": 0.10,
        "discount_rate": 0.08,
        "analysis_period_years": 10,
        "labor_cost_usd_per_hour": 75.0,
        "installation_hours_per_m": 0.5,
        "removal_hours_per_m": 0.3,
        "scaffolding_cost_per_m": 25.0,
        "operating_hours_per_year": 8760,
    }


@pytest.fixture
def roi_reference_cases() -> List[Dict[str, Any]]:
    """Reference cases for ROI calculation validation."""
    return [
        {
            "case_name": "Simple Repair",
            "current_energy_loss_kW": 5.0,
            "projected_energy_loss_kW": 1.5,
            "repair_cost_usd": 2000.0,
            "energy_cost_usd_kWh": 0.10,
            "operating_hours": 8760,
            "expected_annual_savings_usd": 3066.0,  # (5-1.5)*0.10*8760
            "expected_payback_years": 0.65,  # 2000/3066
        },
        {
            "case_name": "Full Replacement",
            "current_energy_loss_kW": 10.0,
            "projected_energy_loss_kW": 1.0,
            "replacement_cost_usd": 15000.0,
            "energy_cost_usd_kWh": 0.10,
            "operating_hours": 8760,
            "expected_annual_savings_usd": 7884.0,  # (10-1)*0.10*8760
            "expected_payback_years": 1.90,  # 15000/7884
        },
    ]


# =============================================================================
# MOCK SERVICE FIXTURES
# =============================================================================

@pytest.fixture
def mock_thermal_camera():
    """Create a mock thermal camera for integration testing."""
    camera = Mock()

    camera.connect = AsyncMock(return_value=True)
    camera.disconnect = AsyncMock(return_value=True)
    camera.is_connected = Mock(return_value=True)

    camera.health_check = AsyncMock(return_value={
        "status": "healthy",
        "model": "FLIR T865",
        "firmware": "2.0.1",
        "calibration_date": "2024-01-15",
    })

    async def capture_image(asset_id: str) -> Dict[str, Any]:
        """Simulate thermal image capture."""
        return {
            "asset_id": asset_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "min_temp_C": 30.0,
            "max_temp_C": 65.0,
            "avg_temp_C": 45.0,
            "resolution": "640x480",
            "image_path": f"/thermal_images/{asset_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.radiometric",
        }

    camera.capture_image = AsyncMock(side_effect=capture_image)

    return camera


@pytest.fixture
def mock_asset_registry():
    """Create a mock asset registry for testing."""
    registry = Mock()

    # Store for assets
    assets = {}

    def register_asset(asset: InsulationAsset) -> str:
        assets[asset.asset_id] = asset
        return asset.asset_id

    def get_asset(asset_id: str) -> Optional[InsulationAsset]:
        return assets.get(asset_id)

    def list_assets() -> List[InsulationAsset]:
        return list(assets.values())

    registry.register_asset = Mock(side_effect=register_asset)
    registry.get_asset = Mock(side_effect=get_asset)
    registry.list_assets = Mock(side_effect=list_assets)

    return registry


@pytest.fixture
def mock_cmms_connector():
    """Create a mock CMMS connector for work order creation."""
    connector = Mock()

    connector.connect = AsyncMock(return_value=True)
    connector.disconnect = AsyncMock(return_value=True)

    connector.health_check = AsyncMock(return_value={
        "status": "healthy",
        "cmms_type": "SAP_PM",
        "latency_ms": 35,
    })

    work_orders = {}

    async def create_work_order(request: Dict[str, Any]) -> Dict[str, Any]:
        wo_id = f"WO-{datetime.now().strftime('%Y%m%d')}-{len(work_orders) + 1:04d}"
        work_orders[wo_id] = {
            "work_order_id": wo_id,
            "asset_id": request.get("asset_id"),
            "description": request.get("description"),
            "priority": request.get("priority", "medium"),
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        return work_orders[wo_id]

    connector.create_work_order = AsyncMock(side_effect=create_work_order)
    connector.get_work_order = AsyncMock(return_value=None)

    return connector


# =============================================================================
# PERFORMANCE TESTING FIXTURES
# =============================================================================

@pytest.fixture
def performance_timer():
    """Create a performance timer context manager."""
    class Timer:
        def __init__(self):
            self.start_time: Optional[datetime] = None
            self.end_time: Optional[datetime] = None
            self.elapsed_ms: Optional[float] = None

        def __enter__(self):
            self.start_time = datetime.now()
            return self

        def __exit__(self, *args):
            self.end_time = datetime.now()
            self.elapsed_ms = (self.end_time - self.start_time).total_seconds() * 1000

        def assert_under(self, max_ms: float):
            assert self.elapsed_ms is not None, "Timer not used in context"
            assert self.elapsed_ms < max_ms, f"Elapsed {self.elapsed_ms:.2f}ms exceeds {max_ms}ms limit"

    return Timer


@pytest.fixture
def provenance_validator():
    """Validator for provenance hashes."""
    def validate(provenance_hash: str, expected_length: int = 64) -> bool:
        """Validate provenance hash format (SHA-256)."""
        if not provenance_hash:
            return False
        if len(provenance_hash) != expected_length:
            return False
        try:
            int(provenance_hash, 16)
            return True
        except ValueError:
            return False

    return validate


# =============================================================================
# CHAOS TESTING FIXTURES
# =============================================================================

@pytest.fixture
def chaos_injector():
    """Create a chaos injector for resilience testing."""
    class ChaosInjector:
        def __init__(
            self,
            failure_rate: float = 0.3,
            slow_rate: float = 0.2,
            slow_delay_ms: float = 500.0,
            seed: int = 42,
        ):
            self.failure_rate = failure_rate
            self.slow_rate = slow_rate
            self.slow_delay_ms = slow_delay_ms
            self.call_count = 0
            self.failure_count = 0
            self.slow_count = 0
            self.seed = seed
            np.random.seed(seed)

        async def chaotic_call(self) -> Dict[str, Any]:
            """Simulate a chaotic service call."""
            import asyncio
            self.call_count += 1

            # Random failure
            if np.random.random() < self.failure_rate:
                self.failure_count += 1
                raise ConnectionError(f"Chaos failure #{self.failure_count}")

            # Random slow response
            if np.random.random() < self.slow_rate:
                self.slow_count += 1
                await asyncio.sleep(self.slow_delay_ms / 1000.0)

            return {"status": "success", "call": self.call_count}

        def reset(self):
            """Reset counters."""
            self.call_count = 0
            self.failure_count = 0
            self.slow_count = 0
            np.random.seed(self.seed)

    return ChaosInjector


# =============================================================================
# DATABASE FIXTURES
# =============================================================================

@pytest.fixture
def mock_database():
    """Create a mock database for testing."""
    class MockDatabase:
        def __init__(self):
            self.assets: Dict[str, InsulationAsset] = {}
            self.measurements: List[ThermalMeasurement] = []
            self.assessments: List[ConditionAssessment] = []
            self.connected = False

        async def connect(self) -> bool:
            self.connected = True
            return True

        async def disconnect(self) -> bool:
            self.connected = False
            return True

        async def save_asset(self, asset: InsulationAsset) -> str:
            self.assets[asset.asset_id] = asset
            return asset.asset_id

        async def get_asset(self, asset_id: str) -> Optional[InsulationAsset]:
            return self.assets.get(asset_id)

        async def save_measurement(self, measurement: ThermalMeasurement) -> str:
            self.measurements.append(measurement)
            return measurement.measurement_id

        async def get_measurements(
            self,
            asset_id: str,
            start_time: Optional[datetime] = None,
            end_time: Optional[datetime] = None,
        ) -> List[ThermalMeasurement]:
            results = [m for m in self.measurements if m.asset_id == asset_id]
            if start_time:
                results = [m for m in results if m.timestamp >= start_time]
            if end_time:
                results = [m for m in results if m.timestamp <= end_time]
            return results

        async def save_assessment(self, assessment: ConditionAssessment) -> str:
            self.assessments.append(assessment)
            return assessment.provenance_hash

        async def get_latest_assessment(self, asset_id: str) -> Optional[ConditionAssessment]:
            asset_assessments = [a for a in self.assessments if a.asset_id == asset_id]
            if not asset_assessments:
                return None
            return max(asset_assessments, key=lambda x: x.timestamp)

    return MockDatabase()


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "golden: marks tests as golden master tests")
    config.addinivalue_line("markers", "chaos: marks tests as chaos engineering tests")
    config.addinivalue_line("markers", "property: marks tests as property-based tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "compliance: marks tests as compliance tests")
    config.addinivalue_line("markers", "slow: marks tests as slow running")


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility."""
    np.random.seed(TEST_CONFIG["random_seed"])


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    "TEST_CONFIG",
    # Enums
    "InsulationType",
    "PipeGeometry",
    "ConditionGrade",
    "DamageType",
    "DataQuality",
    # Data classes
    "InsulationMaterial",
    "InsulationAsset",
    "ThermalMeasurement",
    "HeatLossResult",
    "ConditionAssessment",
    "ROICalculation",
    # Fixtures are auto-discovered by pytest
]
