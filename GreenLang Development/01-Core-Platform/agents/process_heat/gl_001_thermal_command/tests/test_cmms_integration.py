"""
Unit tests for GL-001 ThermalCommand Orchestrator CMMS Integration Module

Tests CMMS integration with 85%+ coverage.
Validates work order creation, CMMS adapters, and maintenance workflows.

Author: GreenLang Test Engineering Team
Version: 1.0.0
"""

import pytest
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from greenlang.agents.process_heat.gl_001_thermal_command.cmms_integration import (
    CMMSManager,
    CMMSAdapter,
    SAPPMAdapter,
    SAPPMConfig,
    MaximoAdapter,
    MaximoConfig,
    MockCMMSAdapter,
    WorkOrder,
    WorkOrderPriority,
    WorkOrderType,
    WorkOrderStatus,
    WorkOrderTemplate,
    Equipment,
    EquipmentCriticality,
    SparePart,
    LaborEstimate,
    ProblemCode,
    CMMSResponse,
    CMMSType,
    create_cmms_manager,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_adapter():
    """Create mock CMMS adapter."""
    return MockCMMSAdapter()


@pytest.fixture
def cmms_manager(mock_adapter):
    """Create CMMS manager with mock adapter."""
    return CMMSManager(adapter=mock_adapter, auto_submit=True)


@pytest.fixture
def sap_config():
    """Create SAP PM configuration."""
    return SAPPMConfig(
        base_url="https://sap.test.com/odata",
        username="TEST_USER",
        password="TEST_PASS",
        plant="1000",
    )


@pytest.fixture
def maximo_config():
    """Create Maximo configuration."""
    return MaximoConfig(
        base_url="https://maximo.test.com/api",
        api_key="test-api-key",
        site_id="SITE1",
        org_id="ORG1",
    )


@pytest.fixture
def sample_work_order():
    """Create sample work order."""
    return WorkOrder(
        equipment_id="BLR-001",
        equipment_tag="BLR-001",
        short_description="High stack temperature investigation",
        long_description="Stack temperature trending high - inspect refractory",
        problem_code=ProblemCode.HIGH_STACK_TEMP,
        priority=WorkOrderPriority.HIGH,
        work_order_type=WorkOrderType.CORRECTIVE,
    )


@pytest.fixture
def sample_equipment():
    """Create sample equipment."""
    return Equipment(
        equipment_id="BLR-001",
        tag_number="BLR-001",
        description="Main Process Boiler #1",
        equipment_type="boiler",
        location="Plant Area A",
        criticality=EquipmentCriticality.CRITICAL,
    )


@pytest.fixture
def sample_template():
    """Create sample work order template."""
    return WorkOrderTemplate(
        name="High Temperature Response",
        work_order_type=WorkOrderType.CORRECTIVE,
        short_description="High temperature investigation",
        long_description="Standard response procedure for high temperature alarms",
        estimated_duration_hours=4.0,
        parts=[
            SparePart(
                part_number="GASKET-001",
                description="Inspection gasket",
                quantity=1,
                estimated_cost=50.0,
            ),
        ],
        labor=[
            LaborEstimate(craft="Instrument Tech", hours=2.0),
            LaborEstimate(craft="Operator", hours=1.0),
        ],
    )


# =============================================================================
# CMMS MANAGER TESTS
# =============================================================================

class TestCMMSManager:
    """Test suite for CMMSManager."""

    @pytest.mark.unit
    def test_initialization(self, cmms_manager):
        """Test CMMS manager initialization."""
        assert cmms_manager is not None
        assert cmms_manager._auto_submit is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_create_work_order(self, cmms_manager):
        """Test work order creation."""
        work_order = await cmms_manager.create_work_order(
            equipment_id="BLR-001",
            problem_code=ProblemCode.HIGH_STACK_TEMP,
            short_description="Test work order",
            priority=WorkOrderPriority.HIGH,
        )

        assert work_order is not None
        assert work_order.equipment_id == "BLR-001"
        assert work_order.problem_code == ProblemCode.HIGH_STACK_TEMP

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_create_work_order_with_auto_submit(self, cmms_manager):
        """Test work order auto-submission."""
        work_order = await cmms_manager.create_work_order(
            equipment_id="BLR-001",
            problem_code=ProblemCode.HIGH_PROCESS_TEMP,
            short_description="Auto-submit test",
        )

        # Should have external ID from CMMS
        assert work_order.external_id is not None
        assert work_order.status == WorkOrderStatus.SUBMITTED

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_create_from_condition(self, cmms_manager):
        """Test condition-based work order creation."""
        work_order = await cmms_manager.create_from_condition(
            equipment_id="BLR-001",
            problem_code=ProblemCode.HIGH_STACK_TEMP,
            current_value=550.0,
            threshold=500.0,
            unit="degF",
        )

        assert work_order is not None
        assert work_order.trigger_value == 550.0
        assert work_order.trigger_threshold == 500.0
        assert work_order.work_order_type == WorkOrderType.CORRECTIVE

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_create_from_condition_with_ai(self, cmms_manager):
        """Test AI-triggered predictive work order."""
        work_order = await cmms_manager.create_from_condition(
            equipment_id="BLR-001",
            problem_code=ProblemCode.LOW_EFFICIENCY,
            current_value=0.75,
            threshold=0.80,
            unit="efficiency",
            ai_confidence=0.92,
        )

        assert work_order.ai_confidence == 0.92
        assert work_order.work_order_type == WorkOrderType.PREDICTIVE

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_create_from_sis_event(self, cmms_manager):
        """Test SIS event work order creation."""
        work_order = await cmms_manager.create_from_sis_event(
            equipment_id="BLR-001",
            interlock_name="SI-101 High Temperature",
            trip_value=560.0,
            setpoint=550.0,
            unit="degC",
        )

        assert work_order is not None
        assert work_order.problem_code == ProblemCode.SAFETY_INTERLOCK_TRIP
        assert work_order.priority == WorkOrderPriority.HIGH
        assert work_order.lockout_tagout_required is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_create_proof_test_work_order(self, cmms_manager):
        """Test proof test work order creation."""
        work_order = await cmms_manager.create_proof_test_work_order(
            equipment_id="BLR-001",
            interlock_name="SI-101",
            test_procedure_id="TP-001",
        )

        assert work_order is not None
        assert work_order.problem_code == ProblemCode.PROOF_TEST_DUE
        assert work_order.work_order_type == WorkOrderType.PROOF_TEST

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_update_status(self, cmms_manager):
        """Test work order status update."""
        work_order = await cmms_manager.create_work_order(
            equipment_id="BLR-001",
            problem_code=ProblemCode.PM_DUE,
            short_description="Status update test",
        )

        result = await cmms_manager.update_status(
            work_order.work_order_id,
            WorkOrderStatus.IN_PROGRESS,
            notes="Started work",
        )

        assert result is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_close_work_order(self, cmms_manager):
        """Test work order closure."""
        work_order = await cmms_manager.create_work_order(
            equipment_id="BLR-001",
            problem_code=ProblemCode.CALIBRATION_DUE,
            short_description="Close test",
        )

        result = await cmms_manager.close_work_order(
            work_order.work_order_id,
            completion_notes="Work completed successfully",
            actual_hours=3.5,
            actual_cost=350.0,
        )

        assert result is True

    @pytest.mark.unit
    def test_get_work_order(self, cmms_manager):
        """Test getting work order by ID."""
        # Create work order first
        loop = asyncio.get_event_loop()
        work_order = loop.run_until_complete(
            cmms_manager.create_work_order(
                equipment_id="BLR-001",
                problem_code=ProblemCode.OTHER,
                short_description="Get test",
            )
        )

        retrieved = cmms_manager.get_work_order(work_order.work_order_id)
        assert retrieved is not None
        assert retrieved.work_order_id == work_order.work_order_id

    @pytest.mark.unit
    def test_get_open_work_orders(self, cmms_manager):
        """Test getting open work orders."""
        open_orders = cmms_manager.get_open_work_orders()
        assert isinstance(open_orders, list)

    @pytest.mark.unit
    def test_get_work_orders_by_equipment(self, cmms_manager):
        """Test getting work orders for equipment."""
        # Create some work orders
        loop = asyncio.get_event_loop()
        for i in range(3):
            loop.run_until_complete(
                cmms_manager.create_work_order(
                    equipment_id="BLR-TEST",
                    problem_code=ProblemCode.PM_DUE,
                    short_description=f"Equipment test {i}",
                )
            )

        orders = cmms_manager.get_work_orders_by_equipment("BLR-TEST")
        assert len(orders) == 3


# =============================================================================
# TEMPLATE TESTS
# =============================================================================

class TestWorkOrderTemplate:
    """Test suite for work order templates."""

    @pytest.mark.unit
    def test_add_template(self, cmms_manager, sample_template):
        """Test adding work order template."""
        cmms_manager.add_template(
            sample_template,
            problem_codes=[ProblemCode.HIGH_STACK_TEMP],
        )

        retrieved = cmms_manager.get_template(sample_template.template_id)
        assert retrieved is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_template_applied(self, cmms_manager, sample_template):
        """Test template is applied to work order."""
        cmms_manager.add_template(
            sample_template,
            problem_codes=[ProblemCode.HIGH_STACK_TEMP],
        )

        work_order = await cmms_manager.create_work_order(
            equipment_id="BLR-001",
            problem_code=ProblemCode.HIGH_STACK_TEMP,
            short_description="Template test",
        )

        # Should have template parts and labor
        assert len(work_order.parts) > 0 or len(work_order.labor) > 0


# =============================================================================
# WORK ORDER TESTS
# =============================================================================

class TestWorkOrder:
    """Test suite for WorkOrder model."""

    @pytest.mark.unit
    def test_initialization(self, sample_work_order):
        """Test work order initialization."""
        assert sample_work_order.equipment_id == "BLR-001"
        assert sample_work_order.priority == WorkOrderPriority.HIGH

    @pytest.mark.unit
    def test_work_order_id_generation(self):
        """Test work order ID is auto-generated."""
        work_order = WorkOrder(
            equipment_id="TEST",
            short_description="ID test",
        )

        assert work_order.work_order_id is not None
        assert work_order.work_order_id.startswith("WO-")

    @pytest.mark.unit
    def test_estimated_cost_calculation(self):
        """Test estimated cost calculation."""
        work_order = WorkOrder(
            equipment_id="TEST",
            short_description="Cost test",
            parts=[
                SparePart(part_number="P1", description="Part 1", quantity=2, estimated_cost=50.0),
                SparePart(part_number="P2", description="Part 2", quantity=1, estimated_cost=100.0),
            ],
            labor=[
                LaborEstimate(craft="Tech", hours=4.0, rate_per_hour=75.0),
            ],
        )

        # Parts: 2*50 + 1*100 = 200
        # Labor: 4*75 = 300
        # Total: 500
        assert work_order.estimated_cost == 500.0

    @pytest.mark.unit
    def test_provenance_hash(self, sample_work_order):
        """Test work order provenance hash."""
        assert sample_work_order.provenance_hash is not None
        assert len(sample_work_order.provenance_hash) == 64

    @pytest.mark.unit
    def test_default_status(self, sample_work_order):
        """Test default work order status."""
        assert sample_work_order.status == WorkOrderStatus.DRAFT


# =============================================================================
# SPARE PART TESTS
# =============================================================================

class TestSparePart:
    """Test suite for SparePart model."""

    @pytest.mark.unit
    def test_initialization(self):
        """Test spare part initialization."""
        part = SparePart(
            part_number="GASKET-001",
            description="High temp gasket",
            quantity=2,
            estimated_cost=25.0,
        )

        assert part.part_number == "GASKET-001"
        assert part.quantity == 2

    @pytest.mark.unit
    def test_quantity_validation(self):
        """Test quantity must be non-negative."""
        with pytest.raises(ValueError):
            SparePart(
                part_number="TEST",
                description="Test",
                quantity=-1,
            )


# =============================================================================
# LABOR ESTIMATE TESTS
# =============================================================================

class TestLaborEstimate:
    """Test suite for LaborEstimate model."""

    @pytest.mark.unit
    def test_initialization(self):
        """Test labor estimate initialization."""
        labor = LaborEstimate(
            craft="Electrician",
            hours=4.0,
            headcount=2,
            rate_per_hour=85.0,
        )

        assert labor.craft == "Electrician"
        assert labor.hours == 4.0

    @pytest.mark.unit
    def test_total_cost_calculation(self):
        """Test labor total cost calculation."""
        labor = LaborEstimate(
            craft="Mechanic",
            hours=8.0,
            headcount=2,
            rate_per_hour=75.0,
        )

        # 8 hours * 2 workers * $75/hr = $1200
        assert labor.total_cost == 1200.0


# =============================================================================
# SAP PM ADAPTER TESTS
# =============================================================================

class TestSAPPMAdapter:
    """Test suite for SAP PM adapter."""

    @pytest.mark.unit
    def test_initialization(self, sap_config):
        """Test SAP PM adapter initialization."""
        adapter = SAPPMAdapter(sap_config)

        assert adapter is not None
        assert adapter.config == sap_config

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_create_work_order(self, sap_config, sample_work_order):
        """Test creating work order in SAP PM."""
        adapter = SAPPMAdapter(sap_config)

        response = await adapter.create_work_order(sample_work_order)

        assert response is not None
        assert isinstance(response, CMMSResponse)
        assert response.success is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_update_work_order(self, sap_config, sample_work_order):
        """Test updating work order in SAP PM."""
        adapter = SAPPMAdapter(sap_config)

        # First create
        create_response = await adapter.create_work_order(sample_work_order)
        sample_work_order.external_id = create_response.external_id

        # Then update
        update_response = await adapter.update_work_order(sample_work_order)

        assert update_response.success is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_update_without_external_id(self, sap_config, sample_work_order):
        """Test update fails without external ID."""
        adapter = SAPPMAdapter(sap_config)

        response = await adapter.update_work_order(sample_work_order)

        assert response.success is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_close_work_order(self, sap_config):
        """Test closing work order in SAP PM."""
        adapter = SAPPMAdapter(sap_config)

        response = await adapter.close_work_order(
            external_id="4000001234",
            completion_notes="Work completed",
        )

        assert response.success is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_check_connection(self, sap_config):
        """Test SAP PM connection check."""
        adapter = SAPPMAdapter(sap_config)

        connected = await adapter.check_connection()
        assert connected is True


# =============================================================================
# MAXIMO ADAPTER TESTS
# =============================================================================

class TestMaximoAdapter:
    """Test suite for Maximo adapter."""

    @pytest.mark.unit
    def test_initialization(self, maximo_config):
        """Test Maximo adapter initialization."""
        adapter = MaximoAdapter(maximo_config)

        assert adapter is not None
        assert adapter.config == maximo_config

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_create_work_order(self, maximo_config, sample_work_order):
        """Test creating work order in Maximo."""
        adapter = MaximoAdapter(maximo_config)

        response = await adapter.create_work_order(sample_work_order)

        assert response.success is True
        assert response.external_id is not None


# =============================================================================
# MOCK ADAPTER TESTS
# =============================================================================

class TestMockCMMSAdapter:
    """Test suite for mock CMMS adapter."""

    @pytest.mark.unit
    def test_initialization(self, mock_adapter):
        """Test mock adapter initialization."""
        assert mock_adapter is not None
        assert len(mock_adapter._work_orders) == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_create_work_order(self, mock_adapter, sample_work_order):
        """Test creating work order in mock adapter."""
        response = await mock_adapter.create_work_order(sample_work_order)

        assert response.success is True
        assert response.external_id is not None
        assert response.external_id.startswith("MOCK-")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_work_order(self, mock_adapter, sample_work_order):
        """Test retrieving work order from mock adapter."""
        create_response = await mock_adapter.create_work_order(sample_work_order)

        retrieved = await mock_adapter.get_work_order(create_response.external_id)

        assert retrieved is not None
        assert retrieved.external_id == create_response.external_id

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_close_work_order(self, mock_adapter, sample_work_order):
        """Test closing work order in mock adapter."""
        create_response = await mock_adapter.create_work_order(sample_work_order)

        close_response = await mock_adapter.close_work_order(
            create_response.external_id,
            "Completed",
        )

        assert close_response.success is True

    @pytest.mark.unit
    def test_add_equipment(self, mock_adapter, sample_equipment):
        """Test adding equipment to mock adapter."""
        mock_adapter.add_equipment(sample_equipment)

        assert sample_equipment.equipment_id in mock_adapter._equipment


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================

class TestFactoryFunction:
    """Test suite for factory functions."""

    @pytest.mark.unit
    def test_create_sap_manager(self):
        """Test creating SAP PM manager."""
        manager = create_cmms_manager(
            cmms_type=CMMSType.SAP_PM,
            config={
                "base_url": "https://sap.test.com",
                "username": "test",
                "password": "test",
                "plant": "1000",
            },
        )

        assert manager is not None

    @pytest.mark.unit
    def test_create_maximo_manager(self):
        """Test creating Maximo manager."""
        manager = create_cmms_manager(
            cmms_type=CMMSType.MAXIMO,
            config={
                "base_url": "https://maximo.test.com",
                "api_key": "test-key",
                "site_id": "SITE1",
                "org_id": "ORG1",
            },
        )

        assert manager is not None

    @pytest.mark.unit
    def test_create_mock_manager(self):
        """Test creating mock manager."""
        manager = create_cmms_manager(cmms_type=CMMSType.MOCK)

        assert manager is not None


# =============================================================================
# STATISTICS TESTS
# =============================================================================

class TestStatistics:
    """Test suite for CMMS statistics."""

    @pytest.mark.unit
    def test_get_statistics(self, cmms_manager):
        """Test getting work order statistics."""
        # Create some work orders
        loop = asyncio.get_event_loop()
        for i in range(5):
            loop.run_until_complete(
                cmms_manager.create_work_order(
                    equipment_id=f"EQ-{i}",
                    problem_code=ProblemCode.PM_DUE,
                    short_description=f"Stats test {i}",
                )
            )

        stats = cmms_manager.get_statistics(days=30)

        assert "total_work_orders" in stats
        assert stats["total_work_orders"] == 5


# =============================================================================
# AUDIT LOG TESTS
# =============================================================================

class TestAuditLog:
    """Test suite for audit logging."""

    @pytest.mark.unit
    def test_audit_log_creation(self, cmms_manager):
        """Test audit log entry on work order creation."""
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            cmms_manager.create_work_order(
                equipment_id="AUDIT-001",
                problem_code=ProblemCode.OTHER,
                short_description="Audit test",
            )
        )

        audit_log = cmms_manager.get_audit_log(limit=10)
        assert len(audit_log) > 0

    @pytest.mark.unit
    def test_audit_log_provenance(self, cmms_manager):
        """Test audit log entries have provenance hash."""
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            cmms_manager.create_work_order(
                equipment_id="PROV-001",
                problem_code=ProblemCode.OTHER,
                short_description="Provenance test",
            )
        )

        audit_log = cmms_manager.get_audit_log(limit=1)
        assert audit_log[0].get("provenance_hash") is not None


# =============================================================================
# CONNECTION TESTS
# =============================================================================

class TestConnection:
    """Test suite for CMMS connection."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_check_connection(self, cmms_manager):
        """Test CMMS connection check."""
        connected = await cmms_manager.check_connection()
        assert connected is True
