"""
GL-007 FURNACEPULSE - Test Fixtures

Pytest fixtures for testing furnace performance monitoring.
Provides sample data, mock clients, and test configurations.
"""

import pytest
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, MagicMock
from dataclasses import dataclass, field
import hashlib
import json


# =============================================================================
# Data Classes for Test Fixtures
# =============================================================================

@dataclass
class FurnaceState:
    """Represents the current state of a furnace for testing."""
    furnace_id: str
    name: str
    status: str  # RUNNING, IDLE, STARTUP, SHUTDOWN, MAINTENANCE
    mode: str  # AUTO, MANUAL, SAFE
    operating_hours: float
    last_maintenance_date: datetime
    design_capacity_kw: float
    current_load_percent: float
    zone_count: int


@dataclass
class TMTReading:
    """Tube Metal Temperature reading for testing."""
    tube_id: str
    zone: str  # RADIANT, CONVECTION, SHIELD, CROSSOVER
    temperature_C: float
    rate_of_rise_C_min: float
    design_limit_C: float
    timestamp: datetime
    signal_quality: str  # GOOD, BAD, SUSPECT, MISSING
    position_x: float = 0.0
    position_y: float = 0.0


@dataclass
class TelemetrySignal:
    """Generic telemetry signal for testing."""
    tag_id: str
    value: float
    unit: str
    timestamp: datetime
    quality: str
    source: str


@dataclass
class MaintenanceRecord:
    """Maintenance history record for RUL testing."""
    record_id: str
    component_id: str
    component_type: str
    maintenance_type: str  # PREVENTIVE, CORRECTIVE, PREDICTIVE
    date_performed: datetime
    operating_hours_at_maintenance: float
    notes: str
    cost_usd: float


@dataclass
class ComplianceChecklistItem:
    """NFPA 86 compliance checklist item."""
    item_id: str
    category: str
    description: str
    requirement: str
    status: str  # PASS, FAIL, N/A, PENDING
    evidence_ref: Optional[str] = None
    last_checked: Optional[datetime] = None


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Default test configuration for FurnacePulse."""
    return {
        "agent_id": "GL-007",
        "agent_name": "FurnacePulse",
        "version": "1.0.0",
        "environment": "test",
        # Safety limits
        "tmt_max_C": 950.0,
        "tmt_warning_C": 900.0,
        "tmt_advisory_C": 850.0,
        "tmt_rate_of_rise_max_C_min": 10.0,
        "tmt_rate_of_rise_warning_C_min": 5.0,
        # Draft limits
        "draft_min_Pa": -250.0,
        "draft_max_Pa": 25.0,
        # Efficiency targets
        "efficiency_target_percent": 85.0,
        "excess_air_min_percent": 10.0,
        "excess_air_max_percent": 30.0,
        "stack_temp_max_C": 450.0,
        # Alert configuration
        "alert_debounce_seconds": 60,
        "alert_escalation_minutes": 15,
        # RUL configuration
        "rul_confidence_level": 0.95,
        "rul_min_data_points": 50,
        # Provenance
        "provenance_enabled": True,
        "deterministic_mode": True,
    }


@pytest.fixture
def alert_thresholds() -> Dict[str, Dict[str, float]]:
    """Alert threshold configuration for testing."""
    return {
        "TMT": {
            "ADVISORY": 850.0,
            "WARNING": 900.0,
            "URGENT": 950.0,
        },
        "RATE_OF_RISE": {
            "ADVISORY": 3.0,
            "WARNING": 5.0,
            "URGENT": 10.0,
        },
        "EFFICIENCY_LOSS": {
            "ADVISORY": 2.0,
            "WARNING": 5.0,
            "URGENT": 10.0,
        },
        "DRAFT_DEVIATION": {
            "ADVISORY": 10.0,
            "WARNING": 25.0,
            "URGENT": 50.0,
        },
    }


# =============================================================================
# Furnace State Fixtures
# =============================================================================

@pytest.fixture
def sample_furnace_state() -> FurnaceState:
    """Create a sample furnace state for testing."""
    return FurnaceState(
        furnace_id="FRN-001",
        name="Ethylene Cracker Furnace 1",
        status="RUNNING",
        mode="AUTO",
        operating_hours=45000.0,
        last_maintenance_date=datetime(2025, 1, 15),
        design_capacity_kw=50000.0,
        current_load_percent=85.0,
        zone_count=4,
    )


@pytest.fixture
def multiple_furnace_states() -> List[FurnaceState]:
    """Create multiple furnace states for batch testing."""
    return [
        FurnaceState(
            furnace_id="FRN-001",
            name="Ethylene Cracker Furnace 1",
            status="RUNNING",
            mode="AUTO",
            operating_hours=45000.0,
            last_maintenance_date=datetime(2025, 1, 15),
            design_capacity_kw=50000.0,
            current_load_percent=85.0,
            zone_count=4,
        ),
        FurnaceState(
            furnace_id="FRN-002",
            name="Ethylene Cracker Furnace 2",
            status="RUNNING",
            mode="AUTO",
            operating_hours=38000.0,
            last_maintenance_date=datetime(2025, 2, 1),
            design_capacity_kw=50000.0,
            current_load_percent=78.0,
            zone_count=4,
        ),
        FurnaceState(
            furnace_id="FRN-003",
            name="Reformer Furnace",
            status="MAINTENANCE",
            mode="SAFE",
            operating_hours=62000.0,
            last_maintenance_date=datetime(2024, 12, 1),
            design_capacity_kw=75000.0,
            current_load_percent=0.0,
            zone_count=6,
        ),
    ]


# =============================================================================
# TMT Reading Fixtures
# =============================================================================

@pytest.fixture
def sample_tmt_readings_normal() -> List[TMTReading]:
    """Create normal TMT readings (all within limits)."""
    base_time = datetime.now()
    return [
        TMTReading(
            tube_id="T-R1-01",
            zone="RADIANT",
            temperature_C=820.0,
            rate_of_rise_C_min=0.5,
            design_limit_C=950.0,
            timestamp=base_time,
            signal_quality="GOOD",
            position_x=1.0,
            position_y=1.0,
        ),
        TMTReading(
            tube_id="T-R1-02",
            zone="RADIANT",
            temperature_C=825.0,
            rate_of_rise_C_min=0.3,
            design_limit_C=950.0,
            timestamp=base_time,
            signal_quality="GOOD",
            position_x=2.0,
            position_y=1.0,
        ),
        TMTReading(
            tube_id="T-R1-03",
            zone="RADIANT",
            temperature_C=818.0,
            rate_of_rise_C_min=-0.2,
            design_limit_C=950.0,
            timestamp=base_time,
            signal_quality="GOOD",
            position_x=3.0,
            position_y=1.0,
        ),
        TMTReading(
            tube_id="T-C1-01",
            zone="CONVECTION",
            temperature_C=650.0,
            rate_of_rise_C_min=0.1,
            design_limit_C=850.0,
            timestamp=base_time,
            signal_quality="GOOD",
            position_x=1.0,
            position_y=5.0,
        ),
        TMTReading(
            tube_id="T-C1-02",
            zone="CONVECTION",
            temperature_C=655.0,
            rate_of_rise_C_min=0.2,
            design_limit_C=850.0,
            timestamp=base_time,
            signal_quality="GOOD",
            position_x=2.0,
            position_y=5.0,
        ),
    ]


@pytest.fixture
def sample_tmt_readings_hotspot() -> List[TMTReading]:
    """Create TMT readings with a hotspot scenario."""
    base_time = datetime.now()
    return [
        # Normal tubes
        TMTReading(
            tube_id="T-R1-01",
            zone="RADIANT",
            temperature_C=820.0,
            rate_of_rise_C_min=0.5,
            design_limit_C=950.0,
            timestamp=base_time,
            signal_quality="GOOD",
            position_x=1.0,
            position_y=1.0,
        ),
        TMTReading(
            tube_id="T-R1-02",
            zone="RADIANT",
            temperature_C=825.0,
            rate_of_rise_C_min=0.3,
            design_limit_C=950.0,
            timestamp=base_time,
            signal_quality="GOOD",
            position_x=2.0,
            position_y=1.0,
        ),
        # HOTSPOT - Temperature exceeds warning threshold
        TMTReading(
            tube_id="T-R1-03",
            zone="RADIANT",
            temperature_C=920.0,  # Above 900C warning threshold
            rate_of_rise_C_min=2.5,  # Elevated rate of rise
            design_limit_C=950.0,
            timestamp=base_time,
            signal_quality="GOOD",
            position_x=3.0,
            position_y=1.0,
        ),
        # Adjacent tube also elevated (spatial clustering)
        TMTReading(
            tube_id="T-R1-04",
            zone="RADIANT",
            temperature_C=895.0,  # Near warning threshold
            rate_of_rise_C_min=1.8,
            design_limit_C=950.0,
            timestamp=base_time,
            signal_quality="GOOD",
            position_x=4.0,
            position_y=1.0,
        ),
        # Normal convection tubes
        TMTReading(
            tube_id="T-C1-01",
            zone="CONVECTION",
            temperature_C=650.0,
            rate_of_rise_C_min=0.1,
            design_limit_C=850.0,
            timestamp=base_time,
            signal_quality="GOOD",
            position_x=1.0,
            position_y=5.0,
        ),
    ]


@pytest.fixture
def sample_tmt_readings_critical() -> List[TMTReading]:
    """Create TMT readings with critical hotspot (above urgent threshold)."""
    base_time = datetime.now()
    return [
        TMTReading(
            tube_id="T-R1-01",
            zone="RADIANT",
            temperature_C=820.0,
            rate_of_rise_C_min=0.5,
            design_limit_C=950.0,
            timestamp=base_time,
            signal_quality="GOOD",
            position_x=1.0,
            position_y=1.0,
        ),
        # CRITICAL HOTSPOT - Exceeds design limit
        TMTReading(
            tube_id="T-R1-02",
            zone="RADIANT",
            temperature_C=965.0,  # Above 950C urgent/design limit
            rate_of_rise_C_min=8.5,  # High rate of rise
            design_limit_C=950.0,
            timestamp=base_time,
            signal_quality="GOOD",
            position_x=2.0,
            position_y=1.0,
        ),
        TMTReading(
            tube_id="T-R1-03",
            zone="RADIANT",
            temperature_C=945.0,  # Near design limit
            rate_of_rise_C_min=6.0,
            design_limit_C=950.0,
            timestamp=base_time,
            signal_quality="GOOD",
            position_x=3.0,
            position_y=1.0,
        ),
    ]


@pytest.fixture
def sample_tmt_readings_rate_of_rise_alert() -> List[TMTReading]:
    """Create TMT readings with high rate of rise (temperatures normal)."""
    base_time = datetime.now()
    return [
        TMTReading(
            tube_id="T-R1-01",
            zone="RADIANT",
            temperature_C=800.0,  # Normal temperature
            rate_of_rise_C_min=12.0,  # URGENT: exceeds 10 C/min limit
            design_limit_C=950.0,
            timestamp=base_time,
            signal_quality="GOOD",
            position_x=1.0,
            position_y=1.0,
        ),
        TMTReading(
            tube_id="T-R1-02",
            zone="RADIANT",
            temperature_C=805.0,
            rate_of_rise_C_min=7.0,  # WARNING: exceeds 5 C/min
            design_limit_C=950.0,
            timestamp=base_time,
            signal_quality="GOOD",
            position_x=2.0,
            position_y=1.0,
        ),
    ]


# =============================================================================
# Telemetry Signal Fixtures
# =============================================================================

@pytest.fixture
def sample_telemetry_signals() -> List[TelemetrySignal]:
    """Create sample telemetry signals for testing."""
    base_time = datetime.now()
    return [
        # Fuel flow
        TelemetrySignal(
            tag_id="FRN-001.FUEL.FLOW",
            value=1500.0,
            unit="kg/h",
            timestamp=base_time,
            quality="GOOD",
            source="OPCUA",
        ),
        # Fuel pressure
        TelemetrySignal(
            tag_id="FRN-001.FUEL.PRESSURE",
            value=350.0,
            unit="kPa",
            timestamp=base_time,
            quality="GOOD",
            source="OPCUA",
        ),
        # Stack temperature
        TelemetrySignal(
            tag_id="FRN-001.STACK.TEMP",
            value=380.0,
            unit="C",
            timestamp=base_time,
            quality="GOOD",
            source="OPCUA",
        ),
        # Flue gas O2
        TelemetrySignal(
            tag_id="FRN-001.FLUE.O2",
            value=3.5,
            unit="%",
            timestamp=base_time,
            quality="GOOD",
            source="OPCUA",
        ),
        # Draft pressure (firebox)
        TelemetrySignal(
            tag_id="FRN-001.DRAFT.FIREBOX",
            value=-25.0,
            unit="Pa",
            timestamp=base_time,
            quality="GOOD",
            source="OPCUA",
        ),
        # Draft pressure (stack)
        TelemetrySignal(
            tag_id="FRN-001.DRAFT.STACK",
            value=-150.0,
            unit="Pa",
            timestamp=base_time,
            quality="GOOD",
            source="OPCUA",
        ),
        # Combustion air flow
        TelemetrySignal(
            tag_id="FRN-001.AIR.FLOW",
            value=25000.0,
            unit="kg/h",
            timestamp=base_time,
            quality="GOOD",
            source="OPCUA",
        ),
        # Production throughput
        TelemetrySignal(
            tag_id="FRN-001.PRODUCT.FLOW",
            value=45000.0,
            unit="kg/h",
            timestamp=base_time,
            quality="GOOD",
            source="OPCUA",
        ),
    ]


@pytest.fixture
def sample_efficiency_inputs() -> Dict[str, float]:
    """Sample inputs for efficiency calculation."""
    return {
        "fuel_mass_flow_kg_h": 1500.0,
        "fuel_lhv_MJ_kg": 48.0,  # Natural gas LHV
        "fuel_temperature_C": 25.0,
        "combustion_air_flow_kg_h": 25000.0,
        "combustion_air_temperature_C": 150.0,  # Preheated air
        "flue_gas_temperature_C": 380.0,
        "flue_gas_O2_percent": 3.5,
        "ambient_temperature_C": 25.0,
        "useful_heat_output_kW": 18000.0,
        "production_rate_kg_h": 45000.0,
    }


# =============================================================================
# Maintenance History Fixtures
# =============================================================================

@pytest.fixture
def sample_maintenance_history() -> List[MaintenanceRecord]:
    """Create sample maintenance history for RUL testing."""
    return [
        MaintenanceRecord(
            record_id="MR-001",
            component_id="TUBE-R1-01",
            component_type="RADIANT_TUBE",
            maintenance_type="PREVENTIVE",
            date_performed=datetime(2024, 6, 15),
            operating_hours_at_maintenance=40000.0,
            notes="Tube inspection - no defects found",
            cost_usd=5000.0,
        ),
        MaintenanceRecord(
            record_id="MR-002",
            component_id="TUBE-R1-01",
            component_type="RADIANT_TUBE",
            maintenance_type="CORRECTIVE",
            date_performed=datetime(2023, 1, 20),
            operating_hours_at_maintenance=32000.0,
            notes="Minor crack repair",
            cost_usd=15000.0,
        ),
        MaintenanceRecord(
            record_id="MR-003",
            component_id="TUBE-R1-01",
            component_type="RADIANT_TUBE",
            maintenance_type="PREVENTIVE",
            date_performed=datetime(2022, 3, 10),
            operating_hours_at_maintenance=24000.0,
            notes="Scheduled inspection",
            cost_usd=5000.0,
        ),
        MaintenanceRecord(
            record_id="MR-004",
            component_id="BURNER-01",
            component_type="BURNER",
            maintenance_type="PREVENTIVE",
            date_performed=datetime(2024, 9, 1),
            operating_hours_at_maintenance=43000.0,
            notes="Burner tip replacement",
            cost_usd=8000.0,
        ),
        MaintenanceRecord(
            record_id="MR-005",
            component_id="REFRACTORY-ZONE1",
            component_type="REFRACTORY",
            maintenance_type="CORRECTIVE",
            date_performed=datetime(2024, 3, 15),
            operating_hours_at_maintenance=38000.0,
            notes="Hot spot repair - patched 0.5 sqm",
            cost_usd=25000.0,
        ),
    ]


@pytest.fixture
def sample_failure_history() -> List[Dict[str, Any]]:
    """Sample failure history for Weibull fitting."""
    return [
        {"component_type": "RADIANT_TUBE", "time_to_failure_hours": 45000, "censored": False},
        {"component_type": "RADIANT_TUBE", "time_to_failure_hours": 52000, "censored": False},
        {"component_type": "RADIANT_TUBE", "time_to_failure_hours": 48000, "censored": False},
        {"component_type": "RADIANT_TUBE", "time_to_failure_hours": 55000, "censored": True},  # Still running
        {"component_type": "RADIANT_TUBE", "time_to_failure_hours": 43000, "censored": False},
        {"component_type": "RADIANT_TUBE", "time_to_failure_hours": 60000, "censored": True},
        {"component_type": "RADIANT_TUBE", "time_to_failure_hours": 47000, "censored": False},
        {"component_type": "RADIANT_TUBE", "time_to_failure_hours": 51000, "censored": False},
    ]


# =============================================================================
# NFPA 86 Compliance Fixtures
# =============================================================================

@pytest.fixture
def sample_nfpa86_checklist() -> List[ComplianceChecklistItem]:
    """Create sample NFPA 86 compliance checklist."""
    return [
        ComplianceChecklistItem(
            item_id="NFPA86-4.3.1",
            category="Flame Supervision",
            description="Flame detection system operational",
            requirement="Flame detector installed and functional for each burner",
            status="PASS",
            evidence_ref="EVD-001",
            last_checked=datetime(2025, 1, 15),
        ),
        ComplianceChecklistItem(
            item_id="NFPA86-4.3.2",
            category="Flame Supervision",
            description="Flame failure response time",
            requirement="Fuel shutoff within 4 seconds of flame loss",
            status="PASS",
            evidence_ref="EVD-002",
            last_checked=datetime(2025, 1, 15),
        ),
        ComplianceChecklistItem(
            item_id="NFPA86-5.2.1",
            category="Combustion Air",
            description="Combustion air interlock",
            requirement="Furnace operation interlocked with combustion air supply",
            status="PASS",
            evidence_ref="EVD-003",
            last_checked=datetime(2025, 1, 15),
        ),
        ComplianceChecklistItem(
            item_id="NFPA86-6.1.1",
            category="Fuel Shutoff",
            description="Emergency fuel shutoff",
            requirement="Manual emergency shutoff accessible and tested",
            status="PASS",
            evidence_ref="EVD-004",
            last_checked=datetime(2025, 1, 15),
        ),
        ComplianceChecklistItem(
            item_id="NFPA86-7.1.1",
            category="Purge Cycle",
            description="Pre-ignition purge",
            requirement="Minimum 4 volume changes before ignition",
            status="PASS",
            evidence_ref="EVD-005",
            last_checked=datetime(2025, 1, 15),
        ),
        ComplianceChecklistItem(
            item_id="NFPA86-8.2.1",
            category="Temperature Monitoring",
            description="Over-temperature protection",
            requirement="High-temperature alarm and shutoff functional",
            status="PASS",
            evidence_ref="EVD-006",
            last_checked=datetime(2025, 1, 15),
        ),
    ]


@pytest.fixture
def sample_nfpa86_checklist_with_failures() -> List[ComplianceChecklistItem]:
    """Create NFPA 86 checklist with some failures."""
    return [
        ComplianceChecklistItem(
            item_id="NFPA86-4.3.1",
            category="Flame Supervision",
            description="Flame detection system operational",
            requirement="Flame detector installed and functional for each burner",
            status="PASS",
            evidence_ref="EVD-001",
            last_checked=datetime(2025, 1, 15),
        ),
        ComplianceChecklistItem(
            item_id="NFPA86-4.3.2",
            category="Flame Supervision",
            description="Flame failure response time",
            requirement="Fuel shutoff within 4 seconds of flame loss",
            status="FAIL",  # FAILURE
            evidence_ref="EVD-002",
            last_checked=datetime(2025, 1, 15),
        ),
        ComplianceChecklistItem(
            item_id="NFPA86-7.1.1",
            category="Purge Cycle",
            description="Pre-ignition purge",
            requirement="Minimum 4 volume changes before ignition",
            status="PENDING",  # Not yet verified
            evidence_ref=None,
            last_checked=None,
        ),
    ]


# =============================================================================
# Mock Client Fixtures
# =============================================================================

@pytest.fixture
def mock_opcua_client():
    """Create mock OPC-UA client for testing."""
    mock_client = AsyncMock()

    # Mock connection methods
    mock_client.connect = AsyncMock(return_value=True)
    mock_client.disconnect = AsyncMock(return_value=True)
    mock_client.is_connected = Mock(return_value=True)

    # Mock tag reading
    async def mock_read_tag(tag_id: str):
        tag_values = {
            "FRN-001.FUEL.FLOW": {"value": 1500.0, "quality": "GOOD", "timestamp": datetime.now()},
            "FRN-001.FUEL.PRESSURE": {"value": 350.0, "quality": "GOOD", "timestamp": datetime.now()},
            "FRN-001.STACK.TEMP": {"value": 380.0, "quality": "GOOD", "timestamp": datetime.now()},
            "FRN-001.FLUE.O2": {"value": 3.5, "quality": "GOOD", "timestamp": datetime.now()},
            "FRN-001.DRAFT.FIREBOX": {"value": -25.0, "quality": "GOOD", "timestamp": datetime.now()},
            "FRN-001.TMT.R1.01": {"value": 820.0, "quality": "GOOD", "timestamp": datetime.now()},
            "FRN-001.TMT.R1.02": {"value": 825.0, "quality": "GOOD", "timestamp": datetime.now()},
        }
        return tag_values.get(tag_id, {"value": None, "quality": "BAD", "timestamp": datetime.now()})

    mock_client.read_tag = mock_read_tag

    # Mock batch reading
    async def mock_read_tags(tag_ids: List[str]):
        return {tag_id: await mock_read_tag(tag_id) for tag_id in tag_ids}

    mock_client.read_tags = mock_read_tags

    # Mock subscription
    mock_client.subscribe = AsyncMock(return_value="sub-001")
    mock_client.unsubscribe = AsyncMock(return_value=True)

    return mock_client


@pytest.fixture
def mock_kafka_producer():
    """Create mock Kafka producer for testing."""
    mock_producer = AsyncMock()

    # Track sent messages
    mock_producer.sent_messages = []

    async def mock_send(topic: str, message: Dict[str, Any], key: Optional[str] = None):
        mock_producer.sent_messages.append({
            "topic": topic,
            "message": message,
            "key": key,
            "timestamp": datetime.now(),
        })
        return {"partition": 0, "offset": len(mock_producer.sent_messages) - 1}

    mock_producer.send = mock_send
    mock_producer.flush = AsyncMock(return_value=True)
    mock_producer.close = AsyncMock(return_value=True)

    # Mock start/stop
    mock_producer.start = AsyncMock(return_value=True)
    mock_producer.stop = AsyncMock(return_value=True)

    return mock_producer


@pytest.fixture
def mock_cmms_client():
    """Create mock CMMS client for testing."""
    mock_client = Mock()

    # Mock work order creation
    def mock_create_work_order(work_order: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "work_order_id": f"WO-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "status": "CREATED",
            "created_at": datetime.now().isoformat(),
            **work_order,
        }

    mock_client.create_work_order = Mock(side_effect=mock_create_work_order)

    # Mock asset lookup
    def mock_get_asset(asset_id: str) -> Dict[str, Any]:
        return {
            "asset_id": asset_id,
            "asset_name": f"Asset {asset_id}",
            "asset_type": "FURNACE",
            "location": "Unit 1",
            "criticality": "HIGH",
            "last_maintenance": datetime(2025, 1, 15).isoformat(),
        }

    mock_client.get_asset = Mock(side_effect=mock_get_asset)

    # Mock maintenance schedule
    mock_client.get_maintenance_schedule = Mock(return_value=[
        {
            "schedule_id": "SCH-001",
            "asset_id": "FRN-001",
            "maintenance_type": "PREVENTIVE",
            "next_due": (datetime.now() + timedelta(days=30)).isoformat(),
            "frequency_days": 90,
        },
    ])

    return mock_client


@pytest.fixture
def mock_historian_client():
    """Create mock historian client for testing."""
    mock_client = AsyncMock()

    async def mock_query_time_series(
        tag_id: str,
        start_time: datetime,
        end_time: datetime,
        interval_seconds: int = 60,
    ) -> List[Dict[str, Any]]:
        """Generate mock time series data."""
        data = []
        current = start_time
        base_value = 820.0

        while current <= end_time:
            # Add some variation
            import random
            value = base_value + random.uniform(-10, 10)
            data.append({
                "timestamp": current.isoformat(),
                "value": value,
                "quality": "GOOD",
            })
            current += timedelta(seconds=interval_seconds)

        return data

    mock_client.query_time_series = mock_query_time_series
    mock_client.connect = AsyncMock(return_value=True)
    mock_client.disconnect = AsyncMock(return_value=True)

    return mock_client


# =============================================================================
# API Test Fixtures
# =============================================================================

@pytest.fixture
def sample_telemetry_request() -> Dict[str, Any]:
    """Sample telemetry processing request."""
    return {
        "furnace_id": "FRN-001",
        "timestamp": datetime.now().isoformat(),
        "signals": [
            {"tag_id": "FUEL.FLOW", "value": 1500.0, "unit": "kg/h", "quality": "GOOD"},
            {"tag_id": "STACK.TEMP", "value": 380.0, "unit": "C", "quality": "GOOD"},
            {"tag_id": "FLUE.O2", "value": 3.5, "unit": "%", "quality": "GOOD"},
        ],
        "tmt_readings": [
            {"tube_id": "T-R1-01", "temperature_C": 820.0, "zone": "RADIANT"},
            {"tube_id": "T-R1-02", "temperature_C": 825.0, "zone": "RADIANT"},
        ],
    }


@pytest.fixture
def sample_kpi_request() -> Dict[str, Any]:
    """Sample KPI calculation request."""
    return {
        "furnace_id": "FRN-001",
        "period_start": (datetime.now() - timedelta(hours=24)).isoformat(),
        "period_end": datetime.now().isoformat(),
        "metrics": ["thermal_efficiency", "sfc", "excess_air", "availability"],
    }


@pytest.fixture
def sample_rul_request() -> Dict[str, Any]:
    """Sample RUL prediction request."""
    return {
        "furnace_id": "FRN-001",
        "components": [
            {"component_id": "TUBE-R1-01", "component_type": "RADIANT_TUBE"},
            {"component_id": "BURNER-01", "component_type": "BURNER"},
        ],
        "confidence_level": 0.95,
        "include_uncertainty": True,
    }


# =============================================================================
# Provenance Test Fixtures
# =============================================================================

@pytest.fixture
def known_calculation_inputs() -> Dict[str, Any]:
    """Known inputs for determinism testing."""
    return {
        "fuel_mass_flow_kg_h": 1500.0,
        "fuel_lhv_MJ_kg": 48.0,
        "useful_heat_output_kW": 18000.0,
        "flue_gas_O2_percent": 3.5,
    }


@pytest.fixture
def expected_calculation_outputs() -> Dict[str, Any]:
    """Expected outputs for known inputs (pre-calculated)."""
    # Fuel input = 1500 kg/h * 48 MJ/kg / 3.6 = 20000 kW
    # Thermal efficiency = 18000 / 20000 * 100 = 90%
    # Excess air from O2 = 3.5 / (21 - 3.5) * 100 = 20%
    return {
        "fuel_input_kW": 20000.0,
        "thermal_efficiency_percent": 90.0,
        "excess_air_percent": 20.0,
        "sfc_MJ_kg": 1.2,  # Example SFC
    }


# =============================================================================
# Performance Test Fixtures
# =============================================================================

@pytest.fixture
def large_tmt_dataset() -> List[TMTReading]:
    """Generate large TMT dataset for performance testing."""
    base_time = datetime.now()
    readings = []

    # 100 tubes with 1000 readings each (simulate 1 hour at 1 reading/3.6 sec)
    for tube_num in range(100):
        for reading_num in range(10):  # Reduced for faster tests
            readings.append(
                TMTReading(
                    tube_id=f"T-R1-{tube_num:03d}",
                    zone="RADIANT" if tube_num < 50 else "CONVECTION",
                    temperature_C=800.0 + (tube_num % 50) + (reading_num * 0.1),
                    rate_of_rise_C_min=0.5 + (reading_num * 0.01),
                    design_limit_C=950.0,
                    timestamp=base_time + timedelta(seconds=reading_num * 3.6),
                    signal_quality="GOOD",
                    position_x=float(tube_num % 10),
                    position_y=float(tube_num // 10),
                )
            )

    return readings


# =============================================================================
# User/RBAC Test Fixtures
# =============================================================================

@pytest.fixture
def sample_user_operator() -> Dict[str, Any]:
    """Sample operator user for RBAC testing."""
    return {
        "user_id": "user-001",
        "username": "jsmith",
        "roles": ["OPERATOR"],
        "permissions": ["READ_TELEMETRY", "READ_ALERTS", "ACKNOWLEDGE_ALERTS"],
        "site_access": ["SITE-001"],
    }


@pytest.fixture
def sample_user_engineer() -> Dict[str, Any]:
    """Sample engineer user for RBAC testing."""
    return {
        "user_id": "user-002",
        "username": "mjones",
        "roles": ["ENGINEER", "OPERATOR"],
        "permissions": [
            "READ_TELEMETRY", "READ_ALERTS", "ACKNOWLEDGE_ALERTS",
            "MODIFY_SETPOINTS", "VIEW_RUL", "EXPORT_DATA",
        ],
        "site_access": ["SITE-001", "SITE-002"],
    }


@pytest.fixture
def sample_user_admin() -> Dict[str, Any]:
    """Sample admin user for RBAC testing."""
    return {
        "user_id": "user-003",
        "username": "admin",
        "roles": ["ADMIN", "ENGINEER", "OPERATOR"],
        "permissions": ["*"],  # All permissions
        "site_access": ["*"],  # All sites
    }


# =============================================================================
# Evidence Package Fixtures
# =============================================================================

@pytest.fixture
def sample_evidence_package() -> Dict[str, Any]:
    """Sample evidence package for compliance testing."""
    return {
        "package_id": "EVD-PKG-001",
        "created_at": datetime.now().isoformat(),
        "furnace_id": "FRN-001",
        "event_type": "COMPLIANCE_AUDIT",
        "items": [
            {
                "item_id": "EVD-001",
                "type": "SENSOR_DATA",
                "description": "TMT readings during test",
                "data_hash": hashlib.sha256(b"tmt_data").hexdigest(),
            },
            {
                "item_id": "EVD-002",
                "type": "CALCULATION_RESULT",
                "description": "Efficiency calculation",
                "data_hash": hashlib.sha256(b"efficiency_calc").hexdigest(),
            },
            {
                "item_id": "EVD-003",
                "type": "ALERT_LOG",
                "description": "Alert history",
                "data_hash": hashlib.sha256(b"alert_log").hexdigest(),
            },
        ],
        "package_hash": hashlib.sha256(b"package_content").hexdigest(),
        "signed_by": "system",
        "is_immutable": True,
    }


# =============================================================================
# Deterministic Test Cases
# =============================================================================

DETERMINISTIC_EFFICIENCY_TEST_CASES = [
    {
        "name": "natural_gas_typical",
        "inputs": {
            "fuel_mass_flow_kg_h": 1500.0,
            "fuel_lhv_MJ_kg": 48.0,
            "useful_heat_output_kW": 18000.0,
        },
        "expected": {
            "fuel_input_kW": 20000.0,
            "thermal_efficiency_percent": 90.0,
        },
    },
    {
        "name": "fuel_oil_high_efficiency",
        "inputs": {
            "fuel_mass_flow_kg_h": 1000.0,
            "fuel_lhv_MJ_kg": 42.0,
            "useful_heat_output_kW": 10500.0,
        },
        "expected": {
            "fuel_input_kW": 11666.67,
            "thermal_efficiency_percent": 90.0,
        },
    },
    {
        "name": "low_load_operation",
        "inputs": {
            "fuel_mass_flow_kg_h": 500.0,
            "fuel_lhv_MJ_kg": 48.0,
            "useful_heat_output_kW": 5000.0,
        },
        "expected": {
            "fuel_input_kW": 6666.67,
            "thermal_efficiency_percent": 75.0,
        },
    },
]

DETERMINISTIC_EXCESS_AIR_TEST_CASES = [
    {
        "name": "optimal_combustion",
        "inputs": {"flue_gas_O2_percent": 3.0},
        "expected": {"excess_air_percent": 16.67},
    },
    {
        "name": "high_excess_air",
        "inputs": {"flue_gas_O2_percent": 5.0},
        "expected": {"excess_air_percent": 31.25},
    },
    {
        "name": "low_excess_air",
        "inputs": {"flue_gas_O2_percent": 2.0},
        "expected": {"excess_air_percent": 10.53},
    },
]
