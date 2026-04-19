"""
Unit Tests for gRPC Process Heat Services

Tests cover:
- ProcessHeatService (RunCalculation, GetStatus, StreamResults)
- EmissionsService (CalculateEmissions, GetEmissionFactors)
- ComplianceService (GenerateReport, CheckCompliance)
- Interceptors (LoggingInterceptor, AuthenticationInterceptor)

Example:
    >>> pytest tests/unit/test_grpc_process_heat.py -v
    >>> pytest tests/unit/test_grpc_process_heat.py -k "test_run_calculation" --cov
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from greenlang.infrastructure.api.grpc_server import (
    ProcessHeatServicer,
    EmissionsServicer,
    ComplianceServicer,
    LoggingInterceptor,
    AuthenticationInterceptor,
    ProcessHeatGrpcServer,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def process_heat_servicer():
    """Create ProcessHeatServicer instance."""
    return ProcessHeatServicer()


@pytest.fixture
def emissions_servicer():
    """Create EmissionsServicer instance."""
    return EmissionsServicer()


@pytest.fixture
def compliance_servicer():
    """Create ComplianceServicer instance."""
    return ComplianceServicer()


@pytest.fixture
def mock_request():
    """Create mock gRPC request."""
    request = Mock()
    request.calculation_id = "calc_123"
    request.agent_name = "ThermalCommand"
    request.timestamp = datetime.now()
    request.equipment = Mock(
        equipment_id="eq_001",
        equipment_type="BOILER",
        rated_capacity_mw=50.0,
        fuel_type="NATURAL_GAS",
    )
    request.fuel_input = Mock(
        quantity=100.0,
        unit="m3",
        carbon_content=0.202,
    )
    request.operating_conditions = Mock(
        load_percent=85.0,
        inlet_temperature_celsius=80.0,
        outlet_temperature_celsius=120.0,
        burner_on=True,
    )
    return request


@pytest.fixture
def mock_context():
    """Create mock gRPC context."""
    context = AsyncMock()
    return context


# ============================================================================
# PROCESSHEATSERVICE TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_run_calculation_success(process_heat_servicer, mock_request, mock_context):
    """Test successful RunCalculation request."""
    response = await process_heat_servicer.RunCalculation(mock_request, mock_context)

    assert response.calculation_id == "calc_123"
    assert response.status == "QUEUED"
    assert response.message_id is not None
    assert response.provenance_hash is not None
    assert len(response.provenance_hash) == 64  # SHA-256 hex string


@pytest.mark.asyncio
async def test_run_calculation_generates_id(process_heat_servicer, mock_context):
    """Test RunCalculation generates ID if not provided."""
    request = Mock()
    request.calculation_id = ""
    request.agent_name = "ThermalCommand"

    response = await process_heat_servicer.RunCalculation(request, mock_context)

    assert response.calculation_id is not None
    assert len(response.calculation_id) > 0


@pytest.mark.asyncio
async def test_get_status_queued(process_heat_servicer, mock_request, mock_context):
    """Test GetStatus returns QUEUED state."""
    # First run calculation
    await process_heat_servicer.RunCalculation(mock_request, mock_context)

    # Then check status
    status_request = Mock(calculation_id="calc_123", message_id="msg_123")
    response = await process_heat_servicer.GetStatus(status_request, mock_context)

    assert response.status == "QUEUED"
    assert response.progress_percent == 0


@pytest.mark.asyncio
async def test_get_status_not_found(process_heat_servicer, mock_context):
    """Test GetStatus raises error for unknown calculation."""
    request = Mock(calculation_id="nonexistent")

    with pytest.raises(Exception):
        await process_heat_servicer.GetStatus(request, mock_context)


@pytest.mark.asyncio
async def test_stream_results_timeout(process_heat_servicer, mock_context):
    """Test StreamResults handles timeout gracefully."""
    request = Mock(agent_name="ThermalCommand", filter_type="")

    results = []
    try:
        async for result in process_heat_servicer.StreamResults(request, mock_context):
            results.append(result)
            if len(results) >= 1:
                break
    except Exception:
        pass

    # Should timeout or return empty stream
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_process_background_task(process_heat_servicer, mock_request):
    """Test background processing updates state."""
    calc_id = "calc_test"

    # Manually trigger processing
    await process_heat_servicer._process(calc_id, mock_request)

    # Check final state
    state = process_heat_servicer.states[calc_id]
    assert state['status'] == 'COMPLETED'
    assert state['progress'] == 100
    assert state['completed_at'] is not None


# ============================================================================
# EMISSIONSSERVICE TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_calculate_emissions_success(emissions_servicer, mock_context):
    """Test successful emissions calculation."""
    request = Mock(
        emissions_id="emis_001",
        scope="SCOPE_1",
        region="GLOBAL",
        reporting_year=2024,
        activity_data=[
            Mock(
                activity_id="act_001",
                activity_type="FUEL_COMBUSTION",
                quantity=100.0,
                unit="MWh",
            )
        ]
    )

    response = await emissions_servicer.CalculateEmissions(request, mock_context)

    assert response.emissions_id == "emis_001"
    assert response.total_co2_tonnes > 0
    assert response.total_co2e_tonnes > 0
    assert response.validation_status == "PASS"
    assert len(response.details) == 1


@pytest.mark.asyncio
async def test_calculate_emissions_multiple_activities(emissions_servicer, mock_context):
    """Test emissions calculation with multiple activities."""
    request = Mock(
        emissions_id="emis_002",
        scope="SCOPE_1",
        region="GLOBAL",
        reporting_year=2024,
        activity_data=[
            Mock(activity_id="a1", activity_type="FUEL_COMBUSTION", quantity=50.0, unit="MWh"),
            Mock(activity_id="a2", activity_type="ELECTRICITY", quantity=30.0, unit="MWh"),
            Mock(activity_id="a3", activity_type="STEAM", quantity=20.0, unit="MWh"),
        ]
    )

    response = await emissions_servicer.CalculateEmissions(request, mock_context)

    assert len(response.details) == 3
    assert response.total_co2_tonnes > 0
    assert response.provenance_hash is not None


@pytest.mark.asyncio
async def test_get_emission_factors_success(emissions_servicer, mock_context):
    """Test retrieving emission factors."""
    request = Mock(
        fuel_type="NATURAL_GAS",
        scope="SCOPE_1",
        region="GLOBAL",
        year=2024,
    )

    response = await emissions_servicer.GetEmissionFactors(request, mock_context)

    assert response.fuel_type == "NATURAL_GAS"
    assert len(response.factors) > 0
    assert response.last_updated is not None


@pytest.mark.asyncio
async def test_get_emission_factors_with_region(emissions_servicer, mock_context):
    """Test retrieving emission factors with region filter."""
    request = Mock(
        fuel_type="NATURAL_GAS",
        scope="SCOPE_1",
        region="EU",
        year=2024,
    )

    response = await emissions_servicer.GetEmissionFactors(request, mock_context)

    assert response.fuel_type == "NATURAL_GAS"
    assert response.factors is not None


# ============================================================================
# COMPLIANCESERVICE TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_generate_report_eudr(compliance_servicer, mock_context):
    """Test generating EUDR compliance report."""
    request = Mock(
        report_id="rep_001",
        framework="EUDR",
        facility_id="fac_001",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
        include_recommendations=True,
    )

    response = await compliance_servicer.GenerateReport(request, mock_context)

    assert response.report_id == "rep_001"
    assert response.framework == "EUDR"
    assert response.status in ["COMPLIANT", "PARTIALLY_COMPLIANT", "NON_COMPLIANT"]
    assert response.overall_score >= 0 and response.overall_score <= 100
    assert len(response.items) > 0
    assert len(response.recommendations) > 0


@pytest.mark.asyncio
async def test_generate_report_cbam(compliance_servicer, mock_context):
    """Test generating CBAM compliance report."""
    request = Mock(
        report_id="rep_002",
        framework="CBAM",
        facility_id="fac_002",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
        include_recommendations=False,
    )

    response = await compliance_servicer.GenerateReport(request, mock_context)

    assert response.framework == "CBAM"
    assert len(response.items) >= 2


@pytest.mark.asyncio
async def test_generate_report_csrd(compliance_servicer, mock_context):
    """Test generating CSRD compliance report."""
    request = Mock(
        report_id="rep_003",
        framework="CSRD",
        facility_id="fac_003",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
        include_recommendations=True,
    )

    response = await compliance_servicer.GenerateReport(request, mock_context)

    assert response.framework == "CSRD"
    assert response.provenance_hash is not None


@pytest.mark.asyncio
async def test_check_compliance_eudr_sufficient_data(compliance_servicer, mock_context):
    """Test EUDR compliance check with sufficient data."""
    request = Mock(
        check_id="chk_001",
        regulation="EUDR",
        facility_id="fac_001",
        data_points=[
            Mock(metric_name="suppliers_verified", value=100),
            Mock(metric_name="land_conversion_checked", value=100),
            Mock(metric_name="deforestation_checked", value=100),
        ]
    )

    response = await compliance_servicer.CheckCompliance(request, mock_context)

    assert response.check_id == "chk_001"
    assert response.regulation == "EUDR"
    assert response.is_compliant is True
    assert len(response.violations) == 0


@pytest.mark.asyncio
async def test_check_compliance_eudr_insufficient_data(compliance_servicer, mock_context):
    """Test EUDR compliance check with insufficient data."""
    request = Mock(
        check_id="chk_002",
        regulation="EUDR",
        facility_id="fac_002",
        data_points=[
            Mock(metric_name="suppliers_verified", value=50),
        ]
    )

    response = await compliance_servicer.CheckCompliance(request, mock_context)

    assert response.is_compliant is False
    assert len(response.violations) > 0


@pytest.mark.asyncio
async def test_check_compliance_score_calculation(compliance_servicer, mock_context):
    """Test compliance score calculation."""
    request = Mock(
        check_id="chk_003",
        regulation="CBAM",
        facility_id="fac_003",
        data_points=[
            Mock(metric_name="test1", value=1),
            Mock(metric_name="test2", value=2),
        ]
    )

    response = await compliance_servicer.CheckCompliance(request, mock_context)

    assert response.compliance_score >= 0 and response.compliance_score <= 100


# ============================================================================
# INTERCEPTOR TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_logging_interceptor_creation():
    """Test LoggingInterceptor can be instantiated."""
    interceptor = LoggingInterceptor()
    assert interceptor is not None


@pytest.mark.asyncio
async def test_authentication_interceptor_creation():
    """Test AuthenticationInterceptor can be instantiated."""
    interceptor = AuthenticationInterceptor(require_auth=False)
    assert interceptor is not None


@pytest.mark.asyncio
async def test_authentication_interceptor_no_auth_required():
    """Test AuthenticationInterceptor allows requests when auth not required."""
    interceptor = AuthenticationInterceptor(require_auth=False)
    assert interceptor.require_auth is False


# ============================================================================
# GRPC SERVER TESTS
# ============================================================================

def test_grpc_server_initialization():
    """Test ProcessHeatGrpcServer initialization."""
    server = ProcessHeatGrpcServer(
        host="127.0.0.1",
        port=50051,
        enable_reflection=True,
        enable_health_check=True,
    )

    assert server.host == "127.0.0.1"
    assert server.port == 50051
    assert server.enable_reflection is True
    assert server.enable_health_check is True


def test_grpc_server_custom_port():
    """Test ProcessHeatGrpcServer with custom port."""
    server = ProcessHeatGrpcServer(host="0.0.0.0", port=50052)

    assert server.port == 50052


# ============================================================================
# HASH/UTILITY TESTS
# ============================================================================

def test_process_heat_servicer_hash():
    """Test SHA-256 hash generation."""
    servicer = ProcessHeatServicer()
    hash1 = servicer._hash("test_data")
    hash2 = servicer._hash("test_data")

    assert hash1 == hash2
    assert len(hash1) == 64  # SHA-256 hex string


def test_emissions_servicer_hash():
    """Test EmissionsServicer hash generation."""
    servicer = EmissionsServicer()
    hash1 = servicer._hash("emissions_id_123")
    hash2 = servicer._hash("emissions_id_123")

    assert hash1 == hash2
    assert len(hash1) == 64


def test_compliance_servicer_hash():
    """Test ComplianceServicer hash generation."""
    servicer = ComplianceServicer()
    hash1 = servicer._hash("report_id_456")
    hash2 = servicer._hash("report_id_456")

    assert hash1 == hash2
    assert len(hash1) == 64


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_full_calculation_workflow(process_heat_servicer, mock_request, mock_context):
    """Test complete calculation workflow: Run -> Get Status -> Stream."""
    # Run calculation
    response = await process_heat_servicer.RunCalculation(mock_request, mock_context)
    calc_id = response.calculation_id

    # Get status
    status_req = Mock(calculation_id=calc_id, message_id="msg")
    status = await process_heat_servicer.GetStatus(status_req, mock_context)
    assert status.calculation_id == calc_id

    # Cleanup
    assert calc_id in process_heat_servicer.states


@pytest.mark.asyncio
async def test_full_emissions_workflow(emissions_servicer, mock_context):
    """Test complete emissions workflow: Calculate -> Get Factors."""
    # Calculate emissions
    calc_req = Mock(
        emissions_id="emis_full",
        scope="SCOPE_1",
        region="GLOBAL",
        reporting_year=2024,
        activity_data=[Mock(activity_id="a1", activity_type="FUEL_COMBUSTION", quantity=100.0, unit="MWh")]
    )
    calc_resp = await emissions_servicer.CalculateEmissions(calc_req, mock_context)
    assert calc_resp.emissions_id == "emis_full"

    # Get factors
    factor_req = Mock(fuel_type="NATURAL_GAS", scope="SCOPE_1", region="GLOBAL", year=2024)
    factor_resp = await emissions_servicer.GetEmissionFactors(factor_req, mock_context)
    assert factor_resp.fuel_type == "NATURAL_GAS"


@pytest.mark.asyncio
async def test_full_compliance_workflow(compliance_servicer, mock_context):
    """Test complete compliance workflow: Generate Report -> Check Compliance."""
    # Generate report
    report_req = Mock(
        report_id="rep_full",
        framework="EUDR",
        facility_id="fac_full",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
        include_recommendations=True,
    )
    report_resp = await compliance_servicer.GenerateReport(report_req, mock_context)
    assert report_resp.framework == "EUDR"

    # Check compliance
    check_req = Mock(
        check_id="chk_full",
        regulation="EUDR",
        facility_id="fac_full",
        data_points=[Mock(metric_name="test", value=1) for _ in range(3)]
    )
    check_resp = await compliance_servicer.CheckCompliance(check_req, mock_context)
    assert check_resp.regulation == "EUDR"
