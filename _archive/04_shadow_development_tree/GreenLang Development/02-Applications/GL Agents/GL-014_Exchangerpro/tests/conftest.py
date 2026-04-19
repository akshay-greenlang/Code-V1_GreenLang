# -*- coding: utf-8 -*-
"""
GL-014 EXCHANGERPRO - Comprehensive Test Fixtures

Pytest fixtures for testing the Heat Exchanger Optimizer agent.
Provides mock data, test helpers, and shared configurations for:
- Thermal calculations (Q, UA, LMTD, epsilon-NTU, pressure drop)
- ML-based fouling prediction
- Cleaning schedule optimization
- OPC-UA and CMMS integrations

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
    "duty_tolerance_percent": 0.5,
}


# =============================================================================
# ENUMERATIONS
# =============================================================================

class TEMAType(Enum):
    """TEMA shell-and-tube exchanger types."""
    AES = "AES"  # Floating head with backing device
    BEM = "BEM"  # Bonnet cover, fixed tubesheet
    AEU = "AEU"  # U-tube with floating head
    BEU = "BEU"  # Bonnet cover, U-tube
    AKT = "AKT"  # Kettle reboiler with floating head
    AJW = "AJW"  # Thermosiphon


class FlowArrangement(Enum):
    """Heat exchanger flow arrangements."""
    COUNTERFLOW = "counterflow"
    PARALLEL = "parallel"
    CROSSFLOW_MIXED = "crossflow_mixed"
    CROSSFLOW_UNMIXED = "crossflow_unmixed"
    SHELL_AND_TUBE_1_2 = "shell_tube_1_2"
    SHELL_AND_TUBE_2_4 = "shell_tube_2_4"


class ExchangerConfiguration(Enum):
    """Epsilon-NTU exchanger configurations."""
    COUNTERFLOW = "counterflow"
    PARALLEL_FLOW = "parallel_flow"
    CROSSFLOW_BOTH_UNMIXED = "crossflow_both_unmixed"
    CROSSFLOW_CMIN_MIXED = "crossflow_cmin_mixed"
    CROSSFLOW_CMAX_MIXED = "crossflow_cmax_mixed"
    CROSSFLOW_BOTH_MIXED = "crossflow_both_mixed"
    SHELL_TUBE_ONE_SHELL = "shell_tube_one_shell"
    SHELL_TUBE_N_SHELLS = "shell_tube_n_shells"


class DataQuality(Enum):
    """Data quality levels."""
    GOOD = "good"
    DEGRADED = "degraded"
    BAD = "bad"
    UNCERTAIN = "uncertain"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ExchangerConfig:
    """Heat exchanger configuration."""
    exchanger_id: str
    exchanger_name: str
    tema_type: TEMAType
    flow_arrangement: FlowArrangement
    shell_diameter_m: float
    tube_od_m: float
    tube_id_m: float
    tube_length_m: float
    tube_count: int
    tube_passes: int
    shell_passes: int
    baffle_spacing_m: float
    baffle_cut_percent: float
    design_duty_kW: float
    design_UA_kW_K: float
    design_pressure_drop_shell_kPa: float
    design_pressure_drop_tube_kPa: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OperatingState:
    """Current operating state of a heat exchanger."""
    exchanger_id: str
    timestamp: datetime
    # Hot side temperatures
    T_hot_in_C: float
    T_hot_out_C: float
    # Cold side temperatures
    T_cold_in_C: float
    T_cold_out_C: float
    # Flow rates
    m_dot_hot_kg_s: float
    m_dot_cold_kg_s: float
    # Pressures
    P_hot_in_kPa: float
    P_hot_out_kPa: float
    P_cold_in_kPa: float
    P_cold_out_kPa: float
    # Fluid properties
    Cp_hot_kJ_kgK: float
    Cp_cold_kJ_kgK: float
    rho_hot_kg_m3: float
    rho_cold_kg_m3: float
    mu_hot_Pa_s: float
    mu_cold_Pa_s: float
    k_hot_W_mK: float
    k_cold_W_mK: float
    # Data quality
    data_quality: DataQuality = DataQuality.GOOD
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThermalKPIs:
    """Thermal performance KPIs."""
    exchanger_id: str
    timestamp: datetime
    # Heat duties
    Q_hot_kW: float
    Q_cold_kW: float
    Q_avg_kW: float
    heat_balance_error_percent: float
    # LMTD
    lmtd_C: float
    lmtd_corrected_C: float
    F_factor: float
    # UA
    UA_actual_kW_K: float
    UA_design_kW_K: float
    UA_ratio: float
    # Effectiveness
    epsilon: float
    epsilon_max: float
    NTU: float
    C_ratio: float
    # Pressure drops
    dP_shell_kPa: float
    dP_tube_kPa: float
    dP_ratio_shell: float
    dP_ratio_tube: float
    # Fouling
    Rf_m2K_kW: float
    cleanliness_factor: float
    # Provenance
    provenance_hash: str
    computation_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FoulingState:
    """Current fouling state of an exchanger."""
    exchanger_id: str
    timestamp: datetime
    fouling_resistance_m2K_kW: float
    ua_degradation_percent: float
    predicted_days_to_threshold: int
    confidence_score: float
    trend: str  # "increasing", "stable", "decreasing"
    recommended_action: str
    provenance_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CleaningRecommendation:
    """Cleaning schedule recommendation."""
    exchanger_id: str
    timestamp: datetime
    recommended_cleaning_date: datetime
    urgency: str  # "routine", "scheduled", "urgent", "critical"
    estimated_cost_usd: float
    estimated_energy_savings_kWh: float
    estimated_payback_days: int
    confidence_score: float
    recommendation_id: str
    provenance_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# EXCHANGER CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def sample_exchanger_config() -> ExchangerConfig:
    """Create a sample shell-and-tube exchanger configuration."""
    return ExchangerConfig(
        exchanger_id="HX-001",
        exchanger_name="Crude Preheater",
        tema_type=TEMAType.AES,
        flow_arrangement=FlowArrangement.SHELL_AND_TUBE_1_2,
        shell_diameter_m=0.762,  # 30 inches
        tube_od_m=0.01905,  # 3/4 inch OD
        tube_id_m=0.01483,  # 14 BWG
        tube_length_m=6.096,  # 20 feet
        tube_count=400,
        tube_passes=2,
        shell_passes=1,
        baffle_spacing_m=0.254,  # 10 inches
        baffle_cut_percent=25.0,
        design_duty_kW=5000.0,
        design_UA_kW_K=125.0,
        design_pressure_drop_shell_kPa=35.0,
        design_pressure_drop_tube_kPa=70.0,
        metadata={
            "service": "crude_preheat",
            "installed_date": "2020-06-15",
            "last_cleaning": "2024-06-15",
        },
    )


@pytest.fixture
def counterflow_config() -> ExchangerConfig:
    """Create a simple counterflow exchanger configuration."""
    return ExchangerConfig(
        exchanger_id="HX-CF-001",
        exchanger_name="Simple Counterflow",
        tema_type=TEMAType.BEM,
        flow_arrangement=FlowArrangement.COUNTERFLOW,
        shell_diameter_m=0.5,
        tube_od_m=0.0254,
        tube_id_m=0.0221,
        tube_length_m=3.0,
        tube_count=100,
        tube_passes=1,
        shell_passes=1,
        baffle_spacing_m=0.3,
        baffle_cut_percent=25.0,
        design_duty_kW=1000.0,
        design_UA_kW_K=50.0,
        design_pressure_drop_shell_kPa=20.0,
        design_pressure_drop_tube_kPa=40.0,
        metadata={},
    )


@pytest.fixture
def parallel_flow_config() -> ExchangerConfig:
    """Create a parallel flow exchanger configuration."""
    return ExchangerConfig(
        exchanger_id="HX-PF-001",
        exchanger_name="Parallel Flow Cooler",
        tema_type=TEMAType.BEM,
        flow_arrangement=FlowArrangement.PARALLEL,
        shell_diameter_m=0.4,
        tube_od_m=0.0254,
        tube_id_m=0.0221,
        tube_length_m=2.5,
        tube_count=80,
        tube_passes=1,
        shell_passes=1,
        baffle_spacing_m=0.25,
        baffle_cut_percent=25.0,
        design_duty_kW=500.0,
        design_UA_kW_K=25.0,
        design_pressure_drop_shell_kPa=15.0,
        design_pressure_drop_tube_kPa=30.0,
        metadata={},
    )


# =============================================================================
# OPERATING STATE FIXTURES
# =============================================================================

@pytest.fixture
def sample_operating_state() -> OperatingState:
    """Create a sample operating state with typical refinery conditions."""
    return OperatingState(
        exchanger_id="HX-001",
        timestamp=datetime.now(timezone.utc),
        # Temperatures
        T_hot_in_C=150.0,
        T_hot_out_C=90.0,
        T_cold_in_C=30.0,
        T_cold_out_C=100.0,
        # Flow rates
        m_dot_hot_kg_s=25.0,
        m_dot_cold_kg_s=20.0,
        # Pressures
        P_hot_in_kPa=500.0,
        P_hot_out_kPa=465.0,
        P_cold_in_kPa=700.0,
        P_cold_out_kPa=630.0,
        # Fluid properties (typical crude oil / water)
        Cp_hot_kJ_kgK=2.3,  # Crude oil
        Cp_cold_kJ_kgK=4.18,  # Water
        rho_hot_kg_m3=800.0,
        rho_cold_kg_m3=990.0,
        mu_hot_Pa_s=0.002,
        mu_cold_Pa_s=0.0008,
        k_hot_W_mK=0.13,
        k_cold_W_mK=0.62,
        data_quality=DataQuality.GOOD,
        metadata={},
    )


@pytest.fixture
def operating_state_equal_dt() -> OperatingState:
    """Operating state with equal terminal temperature differences (edge case)."""
    return OperatingState(
        exchanger_id="HX-EQUAL-DT",
        timestamp=datetime.now(timezone.utc),
        T_hot_in_C=100.0,
        T_hot_out_C=50.0,
        T_cold_in_C=20.0,
        T_cold_out_C=70.0,  # dT1 = 100-70 = 30, dT2 = 50-20 = 30
        m_dot_hot_kg_s=10.0,
        m_dot_cold_kg_s=10.0,
        P_hot_in_kPa=300.0,
        P_hot_out_kPa=280.0,
        P_cold_in_kPa=400.0,
        P_cold_out_kPa=370.0,
        Cp_hot_kJ_kgK=4.18,
        Cp_cold_kJ_kgK=4.18,
        rho_hot_kg_m3=980.0,
        rho_cold_kg_m3=990.0,
        mu_hot_Pa_s=0.0005,
        mu_cold_Pa_s=0.0008,
        k_hot_W_mK=0.65,
        k_cold_W_mK=0.62,
        data_quality=DataQuality.GOOD,
        metadata={"test_case": "equal_terminal_differences"},
    )


@pytest.fixture
def operating_state_temperature_cross() -> OperatingState:
    """Operating state with temperature cross (cold outlet > hot outlet)."""
    return OperatingState(
        exchanger_id="HX-TEMP-CROSS",
        timestamp=datetime.now(timezone.utc),
        T_hot_in_C=100.0,
        T_hot_out_C=50.0,
        T_cold_in_C=20.0,
        T_cold_out_C=80.0,  # Cold outlet > hot outlet (temperature cross)
        m_dot_hot_kg_s=10.0,
        m_dot_cold_kg_s=5.0,
        P_hot_in_kPa=300.0,
        P_hot_out_kPa=280.0,
        P_cold_in_kPa=400.0,
        P_cold_out_kPa=370.0,
        Cp_hot_kJ_kgK=4.18,
        Cp_cold_kJ_kgK=4.18,
        rho_hot_kg_m3=980.0,
        rho_cold_kg_m3=990.0,
        mu_hot_Pa_s=0.0005,
        mu_cold_Pa_s=0.0008,
        k_hot_W_mK=0.65,
        k_cold_W_mK=0.62,
        data_quality=DataQuality.GOOD,
        metadata={"test_case": "temperature_cross"},
    )


@pytest.fixture
def operating_state_balanced() -> OperatingState:
    """Operating state with balanced capacity rates (C_ratio = 1)."""
    return OperatingState(
        exchanger_id="HX-BALANCED",
        timestamp=datetime.now(timezone.utc),
        T_hot_in_C=120.0,
        T_hot_out_C=70.0,
        T_cold_in_C=30.0,
        T_cold_out_C=80.0,  # Balanced: same capacity rate
        m_dot_hot_kg_s=10.0,
        m_dot_cold_kg_s=10.0,
        P_hot_in_kPa=300.0,
        P_hot_out_kPa=280.0,
        P_cold_in_kPa=400.0,
        P_cold_out_kPa=370.0,
        Cp_hot_kJ_kgK=4.18,
        Cp_cold_kJ_kgK=4.18,  # Same Cp gives C_ratio = 1
        rho_hot_kg_m3=980.0,
        rho_cold_kg_m3=990.0,
        mu_hot_Pa_s=0.0005,
        mu_cold_Pa_s=0.0008,
        k_hot_W_mK=0.65,
        k_cold_W_mK=0.62,
        data_quality=DataQuality.GOOD,
        metadata={"test_case": "balanced_capacity_rates"},
    )


@pytest.fixture
def operating_state_high_ntu() -> OperatingState:
    """Operating state with very high NTU (asymptotic effectiveness)."""
    return OperatingState(
        exchanger_id="HX-HIGH-NTU",
        timestamp=datetime.now(timezone.utc),
        T_hot_in_C=150.0,
        T_hot_out_C=35.0,  # Very close to T_cold_in (high effectiveness)
        T_cold_in_C=30.0,
        T_cold_out_C=145.0,  # Very close to T_hot_in
        m_dot_hot_kg_s=5.0,
        m_dot_cold_kg_s=5.0,
        P_hot_in_kPa=300.0,
        P_hot_out_kPa=250.0,
        P_cold_in_kPa=400.0,
        P_cold_out_kPa=350.0,
        Cp_hot_kJ_kgK=4.18,
        Cp_cold_kJ_kgK=4.18,
        rho_hot_kg_m3=980.0,
        rho_cold_kg_m3=990.0,
        mu_hot_Pa_s=0.0005,
        mu_cold_Pa_s=0.0008,
        k_hot_W_mK=0.65,
        k_cold_W_mK=0.62,
        data_quality=DataQuality.GOOD,
        metadata={"test_case": "high_ntu"},
    )


@pytest.fixture
def operating_state_low_flow() -> OperatingState:
    """Operating state with very low flow rates (edge case)."""
    return OperatingState(
        exchanger_id="HX-LOW-FLOW",
        timestamp=datetime.now(timezone.utc),
        T_hot_in_C=100.0,
        T_hot_out_C=40.0,
        T_cold_in_C=25.0,
        T_cold_out_C=85.0,
        m_dot_hot_kg_s=0.1,  # Very low flow
        m_dot_cold_kg_s=0.1,
        P_hot_in_kPa=150.0,
        P_hot_out_kPa=148.0,
        P_cold_in_kPa=200.0,
        P_cold_out_kPa=198.0,
        Cp_hot_kJ_kgK=4.18,
        Cp_cold_kJ_kgK=4.18,
        rho_hot_kg_m3=980.0,
        rho_cold_kg_m3=990.0,
        mu_hot_Pa_s=0.0005,
        mu_cold_Pa_s=0.0008,
        k_hot_W_mK=0.65,
        k_cold_W_mK=0.62,
        data_quality=DataQuality.GOOD,
        metadata={"test_case": "low_flow"},
    )


# =============================================================================
# THERMAL KPIs FIXTURES
# =============================================================================

@pytest.fixture
def sample_thermal_kpis() -> ThermalKPIs:
    """Create sample thermal KPIs for testing."""
    provenance_data = "HX-001:2024-01-15:Q=3450:UA=115"
    provenance_hash = hashlib.sha256(provenance_data.encode()).hexdigest()

    return ThermalKPIs(
        exchanger_id="HX-001",
        timestamp=datetime.now(timezone.utc),
        Q_hot_kW=3450.0,
        Q_cold_kW=3465.0,
        Q_avg_kW=3457.5,
        heat_balance_error_percent=0.43,
        lmtd_C=42.5,
        lmtd_corrected_C=38.25,
        F_factor=0.90,
        UA_actual_kW_K=90.4,
        UA_design_kW_K=125.0,
        UA_ratio=0.723,
        epsilon=0.68,
        epsilon_max=0.85,
        NTU=1.8,
        C_ratio=0.7,
        dP_shell_kPa=32.0,
        dP_tube_kPa=65.0,
        dP_ratio_shell=0.914,
        dP_ratio_tube=0.929,
        Rf_m2K_kW=0.00035,
        cleanliness_factor=0.723,
        provenance_hash=provenance_hash,
        computation_time_ms=3.5,
        metadata={},
    )


@pytest.fixture
def clean_exchanger_kpis() -> ThermalKPIs:
    """KPIs for a clean exchanger (high UA ratio)."""
    provenance_data = "HX-CLEAN:clean_state"
    provenance_hash = hashlib.sha256(provenance_data.encode()).hexdigest()

    return ThermalKPIs(
        exchanger_id="HX-CLEAN",
        timestamp=datetime.now(timezone.utc),
        Q_hot_kW=5000.0,
        Q_cold_kW=5010.0,
        Q_avg_kW=5005.0,
        heat_balance_error_percent=0.2,
        lmtd_C=40.0,
        lmtd_corrected_C=38.0,
        F_factor=0.95,
        UA_actual_kW_K=131.7,
        UA_design_kW_K=125.0,
        UA_ratio=1.05,
        epsilon=0.78,
        epsilon_max=0.85,
        NTU=2.5,
        C_ratio=0.65,
        dP_shell_kPa=33.0,
        dP_tube_kPa=68.0,
        dP_ratio_shell=0.943,
        dP_ratio_tube=0.971,
        Rf_m2K_kW=0.00005,
        cleanliness_factor=0.98,
        provenance_hash=provenance_hash,
        computation_time_ms=2.8,
        metadata={"state": "clean"},
    )


@pytest.fixture
def fouled_exchanger_kpis() -> ThermalKPIs:
    """KPIs for a heavily fouled exchanger (low UA ratio)."""
    provenance_data = "HX-FOULED:fouled_state"
    provenance_hash = hashlib.sha256(provenance_data.encode()).hexdigest()

    return ThermalKPIs(
        exchanger_id="HX-FOULED",
        timestamp=datetime.now(timezone.utc),
        Q_hot_kW=2500.0,
        Q_cold_kW=2520.0,
        Q_avg_kW=2510.0,
        heat_balance_error_percent=0.8,
        lmtd_C=50.0,
        lmtd_corrected_C=45.0,
        F_factor=0.90,
        UA_actual_kW_K=55.8,
        UA_design_kW_K=125.0,
        UA_ratio=0.446,
        epsilon=0.45,
        epsilon_max=0.85,
        NTU=0.9,
        C_ratio=0.7,
        dP_shell_kPa=50.0,
        dP_tube_kPa=95.0,
        dP_ratio_shell=1.43,
        dP_ratio_tube=1.36,
        Rf_m2K_kW=0.0012,
        cleanliness_factor=0.446,
        provenance_hash=provenance_hash,
        computation_time_ms=4.2,
        metadata={"state": "fouled"},
    )


# =============================================================================
# MOCK OPC-UA CLIENT FIXTURE
# =============================================================================

@pytest.fixture
def mock_opcua_client():
    """Create a mock OPC-UA client for integration testing."""
    client = Mock()

    # Connection management
    client.connect = AsyncMock(return_value=True)
    client.disconnect = AsyncMock(return_value=True)
    client.is_connected = Mock(return_value=True)

    # Health check
    client.health_check = AsyncMock(return_value={
        "status": "healthy",
        "latency_ms": 25,
        "server_state": "running",
        "endpoint": "opc.tcp://localhost:4840",
    })

    # Tag reading
    async def read_tag(tag_path: str) -> Dict[str, Any]:
        """Simulate reading a tag from OPC-UA server."""
        tag_data = {
            "HX-001/TI_HOT_IN": {"value": 150.0, "quality": "good", "unit": "degC"},
            "HX-001/TI_HOT_OUT": {"value": 90.0, "quality": "good", "unit": "degC"},
            "HX-001/TI_COLD_IN": {"value": 30.0, "quality": "good", "unit": "degC"},
            "HX-001/TI_COLD_OUT": {"value": 100.0, "quality": "good", "unit": "degC"},
            "HX-001/FI_HOT": {"value": 25.0, "quality": "good", "unit": "kg/s"},
            "HX-001/FI_COLD": {"value": 20.0, "quality": "good", "unit": "kg/s"},
            "HX-001/PI_HOT_IN": {"value": 500.0, "quality": "good", "unit": "kPa"},
            "HX-001/PI_HOT_OUT": {"value": 465.0, "quality": "good", "unit": "kPa"},
            "HX-001/PI_COLD_IN": {"value": 700.0, "quality": "good", "unit": "kPa"},
            "HX-001/PI_COLD_OUT": {"value": 630.0, "quality": "good", "unit": "kPa"},
        }
        if tag_path in tag_data:
            return {
                **tag_data[tag_path],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        raise ValueError(f"Unknown tag: {tag_path}")

    client.read_tag = AsyncMock(side_effect=read_tag)

    # Batch reading
    async def read_tags(tag_paths: List[str]) -> Dict[str, Any]:
        """Simulate batch reading tags."""
        results = {}
        for path in tag_paths:
            try:
                results[path] = await read_tag(path)
            except ValueError:
                results[path] = {"value": None, "quality": "bad", "error": "Unknown tag"}
        return results

    client.read_tags = AsyncMock(side_effect=read_tags)

    # Subscription management
    subscriptions = {}

    def create_subscription(tags: List[str], callback: Callable) -> str:
        sub_id = str(uuid.uuid4())
        subscriptions[sub_id] = {"tags": tags, "callback": callback}
        return sub_id

    client.create_subscription = Mock(side_effect=create_subscription)
    client.remove_subscription = Mock(return_value=True)
    client.get_subscriptions = Mock(return_value=subscriptions)

    return client


# =============================================================================
# MOCK CMMS CONNECTOR FIXTURE
# =============================================================================

@pytest.fixture
def mock_cmms_connector():
    """Create a mock CMMS connector for integration testing."""
    connector = Mock()

    # Connection management
    connector.connect = AsyncMock(return_value=True)
    connector.disconnect = AsyncMock(return_value=True)

    # Health check
    connector.health_check = AsyncMock(return_value={
        "status": "healthy",
        "latency_ms": 45,
        "cmms_type": "SAP_PM",
        "version": "7.5",
    })

    # Work order cache for deduplication testing
    work_order_cache = {}

    # Create work order
    async def create_work_order(request: Dict[str, Any]) -> Dict[str, Any]:
        """Create a work order in CMMS."""
        key = f"{request.get('exchanger_id')}:{request.get('recommendation_id')}"
        if key not in work_order_cache:
            wo_id = f"WO-{datetime.now().strftime('%Y%m%d')}-{len(work_order_cache) + 1:04d}"
            work_order_cache[key] = {
                "work_order_id": wo_id,
                "exchanger_id": request.get("exchanger_id"),
                "status": "pending",
                "priority": request.get("priority", "medium"),
                "description": request.get("description", "Exchanger cleaning"),
                "scheduled_date": request.get("scheduled_date"),
                "estimated_cost_usd": request.get("estimated_cost_usd", 5000.0),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "created_by": "GL-014_EXCHANGERPRO",
            }
        return work_order_cache[key]

    connector.create_work_order = AsyncMock(side_effect=create_work_order)

    # Get work order
    async def get_work_order(work_order_id: str) -> Dict[str, Any]:
        """Get work order by ID."""
        for wo in work_order_cache.values():
            if wo["work_order_id"] == work_order_id:
                return wo
        return None

    connector.get_work_order = AsyncMock(side_effect=get_work_order)

    # List work orders
    connector.list_work_orders = AsyncMock(return_value=list(work_order_cache.values()))

    # Get equipment history
    connector.get_equipment_history = AsyncMock(return_value=[
        {
            "event_type": "cleaning",
            "date": "2024-06-15",
            "cost_usd": 8500.0,
            "downtime_hours": 24,
            "notes": "Mechanical cleaning, shell and tube side",
        },
        {
            "event_type": "cleaning",
            "date": "2023-12-10",
            "cost_usd": 7200.0,
            "downtime_hours": 20,
            "notes": "Chemical cleaning, tube side only",
        },
    ])

    return connector


# =============================================================================
# ML SERVICE MOCK FIXTURES
# =============================================================================

@pytest.fixture
def mock_ml_service():
    """Create a mock ML service for fouling prediction."""
    service = Mock()

    # Health check
    service.health_check = AsyncMock(return_value={
        "status": "healthy",
        "model_loaded": True,
        "model_version": "1.2.0",
        "latency_ms": 15,
    })

    # Feature extraction
    def extract_features(operating_state: OperatingState) -> Dict[str, float]:
        """Extract ML features from operating state."""
        return {
            "dt_hot": operating_state.T_hot_in_C - operating_state.T_hot_out_C,
            "dt_cold": operating_state.T_cold_out_C - operating_state.T_cold_in_C,
            "flow_ratio": operating_state.m_dot_hot_kg_s / operating_state.m_dot_cold_kg_s,
            "dp_shell_ratio": (operating_state.P_hot_in_kPa - operating_state.P_hot_out_kPa) / 35.0,
            "dp_tube_ratio": (operating_state.P_cold_in_kPa - operating_state.P_cold_out_kPa) / 70.0,
            "reynolds_hot": (operating_state.rho_hot_kg_m3 * operating_state.m_dot_hot_kg_s) /
                           (operating_state.mu_hot_Pa_s * 0.01),
        }

    service.extract_features = Mock(side_effect=extract_features)

    # Predict fouling
    async def predict_fouling(features: Dict[str, float], horizon_days: int = 7) -> Dict[str, Any]:
        """Predict fouling state."""
        # Simulate prediction based on pressure drop ratios
        dp_ratio = (features.get("dp_shell_ratio", 1.0) + features.get("dp_tube_ratio", 1.0)) / 2
        rf_predicted = max(0.0001, (dp_ratio - 1.0) * 0.001)

        return {
            "fouling_resistance_m2K_kW": rf_predicted,
            "ua_degradation_percent": min(50.0, (dp_ratio - 1.0) * 30),
            "predicted_days_to_threshold": max(7, int(180 * (1 - dp_ratio))),
            "confidence_score": 0.85,
            "prediction_interval": {
                "lower": rf_predicted * 0.8,
                "upper": rf_predicted * 1.2,
            },
        }

    service.predict_fouling = AsyncMock(side_effect=predict_fouling)

    return service


@pytest.fixture
def mock_optimizer_service():
    """Create a mock optimizer service for cleaning scheduling."""
    service = Mock()

    # Health check
    service.health_check = AsyncMock(return_value={
        "status": "healthy",
        "solver_type": "CBC",
        "version": "2.10.5",
    })

    # Optimize cleaning schedule
    async def optimize_schedule(
        exchanger_id: str,
        fouling_state: FoulingState,
        constraints: Dict[str, Any],
    ) -> CleaningRecommendation:
        """Optimize cleaning schedule."""
        days_to_clean = min(
            90,
            max(7, fouling_state.predicted_days_to_threshold - 7)
        )

        if fouling_state.ua_degradation_percent > 40:
            urgency = "urgent"
        elif fouling_state.ua_degradation_percent > 25:
            urgency = "scheduled"
        else:
            urgency = "routine"

        recommendation_id = str(uuid.uuid4())
        provenance_data = f"{exchanger_id}:{recommendation_id}:{days_to_clean}"
        provenance_hash = hashlib.sha256(provenance_data.encode()).hexdigest()

        return CleaningRecommendation(
            exchanger_id=exchanger_id,
            timestamp=datetime.now(timezone.utc),
            recommended_cleaning_date=datetime.now(timezone.utc) + timedelta(days=days_to_clean),
            urgency=urgency,
            estimated_cost_usd=8500.0,
            estimated_energy_savings_kWh=15000.0,
            estimated_payback_days=45,
            confidence_score=0.82,
            recommendation_id=recommendation_id,
            provenance_hash=provenance_hash,
            metadata={
                "optimization_method": "cost_benefit_analysis",
                "constraints_applied": list(constraints.keys()),
            },
        )

    service.optimize_schedule = AsyncMock(side_effect=optimize_schedule)

    return service


# =============================================================================
# REFERENCE DATA FOR GOLDEN TESTS
# =============================================================================

@pytest.fixture
def tema_reference_data() -> Dict[str, Any]:
    """Reference data from TEMA standards for validation."""
    return {
        # Shell-and-tube F-factor correction data
        "f_factors": {
            # (R, P) -> F for 1-2 shell-and-tube
            (0.5, 0.6): 0.88,
            (1.0, 0.5): 0.80,
            (1.5, 0.4): 0.75,
            (2.0, 0.3): 0.70,
        },
        # Standard tube sizes (OD, ID) in inches
        "tube_sizes": {
            "3/4_14BWG": (0.75, 0.584),
            "3/4_16BWG": (0.75, 0.620),
            "1_14BWG": (1.0, 0.834),
            "1_16BWG": (1.0, 0.870),
        },
        # Standard fouling factors (m2-K/kW)
        "fouling_factors": {
            "crude_oil": 0.00035,
            "diesel_fuel": 0.00020,
            "gas_oil": 0.00020,
            "cooling_water_treated": 0.00018,
            "cooling_water_untreated": 0.00035,
            "steam_clean": 0.00009,
            "steam_oil_bearing": 0.00018,
        },
    }


@pytest.fixture
def engineering_reference_cases() -> List[Dict[str, Any]]:
    """Engineering reference cases for validation against known results."""
    return [
        {
            "name": "Counterflow Water-Water",
            "T_hot_in_C": 100.0,
            "T_hot_out_C": 60.0,
            "T_cold_in_C": 20.0,
            "T_cold_out_C": 80.0,
            "expected_lmtd_C": 28.85,  # (20-40)/ln(20/40)
            "expected_epsilon": 0.75,  # (80-20)/(100-20)
        },
        {
            "name": "Parallel Flow Classic",
            "T_hot_in_C": 100.0,
            "T_hot_out_C": 60.0,
            "T_cold_in_C": 20.0,
            "T_cold_out_C": 50.0,
            "expected_lmtd_C": 33.65,  # (80-10)/ln(80/10)
        },
        {
            "name": "Equal Terminal Differences",
            "T_hot_in_C": 100.0,
            "T_hot_out_C": 50.0,
            "T_cold_in_C": 20.0,
            "T_cold_out_C": 70.0,
            "expected_lmtd_C": 30.0,  # dT1 = dT2 = 30
        },
    ]


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
        ):
            self.failure_rate = failure_rate
            self.slow_rate = slow_rate
            self.slow_delay_ms = slow_delay_ms
            self.call_count = 0
            self.failure_count = 0
            self.slow_count = 0
            self.seed = TEST_CONFIG["random_seed"]
            np.random.seed(self.seed)

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
    "TEMAType",
    "FlowArrangement",
    "ExchangerConfiguration",
    "DataQuality",
    # Data classes
    "ExchangerConfig",
    "OperatingState",
    "ThermalKPIs",
    "FoulingState",
    "CleaningRecommendation",
    # Fixtures are auto-discovered by pytest
]
