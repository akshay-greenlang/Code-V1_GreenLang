# -*- coding: utf-8 -*-
"""
Pytest fixtures for AGENT-MRV-016 Fuel & Energy Activities Agent tests.

Provides shared test data, fixtures, and mock objects for all test modules.
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from typing import Dict, Any, List

from greenlang.agents.mrv.fuel_energy_activities.models import (
    # Enums
    FuelType,
    ActivityType,
    CalculationMethod,
    ElectricitySource,
    QualityTier,
    RegulatoryFramework,

    # Input Models
    FuelConsumptionRecord,
    ElectricityConsumptionRecord,
    WTTEmissionFactor,
    UpstreamElectricityFactor,
    TDLossFactor,
    SupplierFuelData,

    # Output Models
    Activity3aResult,
    Activity3bResult,
    Activity3cResult,
    CalculationResult,
    FuelBreakdown,
    ElectricityBreakdown,
    TDLossBreakdown,
    GasBreakdown,

    # Compliance Models
    ComplianceCheckResult,
    FrameworkDisclosure,

    # DQI Models
    DQIAssessment,
    DQIScore,

    # Batch Models
    BatchRequest,
    BatchResult,
)


# ============================================================================
# FUEL CONSUMPTION FIXTURES
# ============================================================================

@pytest.fixture
def sample_fuel_record() -> FuelConsumptionRecord:
    """Create sample fuel consumption record for natural gas."""
    return FuelConsumptionRecord(
        facility_id="FAC-001",
        reporting_period="2024-Q1",
        fuel_type=FuelType.NATURAL_GAS,
        fuel_quantity=Decimal("10000.0"),
        fuel_quantity_unit="m3",
        activity_type=ActivityType.ACTIVITY_3A,
        country="US",
        region="Northeast",
        sector="Manufacturing",
        calculation_method=CalculationMethod.FUEL_BASED,
        has_renewable_content=False,
        renewable_fraction=Decimal("0.0"),
        upstream_location=None,
        data_quality_score=Decimal("0.85"),
        metadata={"source": "supplier_invoice"}
    )


@pytest.fixture
def sample_diesel_record() -> FuelConsumptionRecord:
    """Create sample fuel consumption record for diesel."""
    return FuelConsumptionRecord(
        facility_id="FAC-002",
        reporting_period="2024-Q1",
        fuel_type=FuelType.DIESEL,
        fuel_quantity=Decimal("5000.0"),
        fuel_quantity_unit="L",
        activity_type=ActivityType.ACTIVITY_3A,
        country="GB",
        region="England",
        sector="Transportation",
        calculation_method=CalculationMethod.FUEL_BASED,
        has_renewable_content=True,
        renewable_fraction=Decimal("0.07"),  # 7% biodiesel
        upstream_location="North_Sea",
        data_quality_score=Decimal("0.90"),
        metadata={"source": "fuel_card"}
    )


@pytest.fixture
def sample_coal_record() -> FuelConsumptionRecord:
    """Create sample fuel consumption record for coal."""
    return FuelConsumptionRecord(
        facility_id="FAC-003",
        reporting_period="2024-Q1",
        fuel_type=FuelType.ANTHRACITE_COAL,
        fuel_quantity=Decimal("50.0"),
        fuel_quantity_unit="tonnes",
        activity_type=ActivityType.ACTIVITY_3A,
        country="DE",
        region="North_Rhine_Westphalia",
        sector="Power_Generation",
        calculation_method=CalculationMethod.FUEL_BASED,
        has_renewable_content=False,
        renewable_fraction=Decimal("0.0"),
        upstream_location="Ruhr_Basin",
        data_quality_score=Decimal("0.95"),
        metadata={"source": "weighbridge"}
    )


@pytest.fixture
def sample_lpg_record() -> FuelConsumptionRecord:
    """Create sample fuel consumption record for LPG."""
    return FuelConsumptionRecord(
        facility_id="FAC-004",
        reporting_period="2024-Q1",
        fuel_type=FuelType.LPG,
        fuel_quantity=Decimal("2000.0"),
        fuel_quantity_unit="kg",
        activity_type=ActivityType.ACTIVITY_3A,
        country="FR",
        region="Ile_de_France",
        sector="Commercial",
        calculation_method=CalculationMethod.FUEL_BASED,
        has_renewable_content=False,
        renewable_fraction=Decimal("0.0"),
        upstream_location=None,
        data_quality_score=Decimal("0.80"),
        metadata={"source": "tank_level"}
    )


@pytest.fixture
def sample_biofuel_record() -> FuelConsumptionRecord:
    """Create sample fuel consumption record for biofuel (ethanol)."""
    return FuelConsumptionRecord(
        facility_id="FAC-005",
        reporting_period="2024-Q1",
        fuel_type=FuelType.ETHANOL,
        fuel_quantity=Decimal("3000.0"),
        fuel_quantity_unit="L",
        activity_type=ActivityType.ACTIVITY_3A,
        country="US",
        region="Midwest",
        sector="Transportation",
        calculation_method=CalculationMethod.FUEL_BASED,
        has_renewable_content=True,
        renewable_fraction=Decimal("1.0"),  # 100% renewable
        upstream_location="Iowa",
        data_quality_score=Decimal("0.88"),
        metadata={"source": "supplier_certificate", "feedstock": "corn"}
    )


# ============================================================================
# ELECTRICITY CONSUMPTION FIXTURES
# ============================================================================

@pytest.fixture
def sample_electricity_record() -> ElectricityConsumptionRecord:
    """Create sample electricity consumption record for US."""
    return ElectricityConsumptionRecord(
        facility_id="FAC-006",
        reporting_period="2024-Q1",
        electricity_quantity=Decimal("100000.0"),
        electricity_unit="kWh",
        activity_type=ActivityType.ACTIVITY_3B,
        country="US",
        region="Northeast",
        egrid_subregion="NEWE",  # New England
        electricity_source=ElectricitySource.GRID,
        supplier_name="Regional_Utility",
        calculation_method=CalculationMethod.LOCATION_BASED,
        has_renewable_content=False,
        renewable_fraction=Decimal("0.0"),
        data_quality_score=Decimal("0.92"),
        metadata={"source": "utility_bill"}
    )


@pytest.fixture
def sample_electricity_record_uk() -> ElectricityConsumptionRecord:
    """Create sample electricity consumption record for UK."""
    return ElectricityConsumptionRecord(
        facility_id="FAC-007",
        reporting_period="2024-Q1",
        electricity_quantity=Decimal("50000.0"),
        electricity_unit="kWh",
        activity_type=ActivityType.ACTIVITY_3B,
        country="GB",
        region="England",
        egrid_subregion=None,
        electricity_source=ElectricitySource.GRID,
        supplier_name="UK_Power",
        calculation_method=CalculationMethod.LOCATION_BASED,
        has_renewable_content=True,
        renewable_fraction=Decimal("0.35"),  # 35% renewable
        data_quality_score=Decimal("0.90"),
        metadata={"source": "smart_meter"}
    )


@pytest.fixture
def sample_electricity_record_de() -> ElectricityConsumptionRecord:
    """Create sample electricity consumption record for Germany."""
    return ElectricityConsumptionRecord(
        facility_id="FAC-008",
        reporting_period="2024-Q1",
        electricity_quantity=Decimal("75000.0"),
        electricity_unit="MWh",
        activity_type=ActivityType.ACTIVITY_3B,
        country="DE",
        region="Bavaria",
        egrid_subregion=None,
        electricity_source=ElectricitySource.GRID,
        supplier_name="German_Energy",
        calculation_method=CalculationMethod.LOCATION_BASED,
        has_renewable_content=True,
        renewable_fraction=Decimal("0.46"),  # 46% renewable
        data_quality_score=Decimal("0.93"),
        metadata={"source": "scada_system"}
    )


@pytest.fixture
def sample_steam_record() -> ElectricityConsumptionRecord:
    """Create sample steam/heat consumption record."""
    return ElectricityConsumptionRecord(
        facility_id="FAC-009",
        reporting_period="2024-Q1",
        electricity_quantity=Decimal("25000.0"),
        electricity_unit="GJ",
        activity_type=ActivityType.ACTIVITY_3C,
        country="US",
        region="Midwest",
        egrid_subregion=None,
        electricity_source=ElectricitySource.DISTRICT_HEATING,
        supplier_name="District_Heat_Co",
        calculation_method=CalculationMethod.SUPPLIER_SPECIFIC,
        has_renewable_content=False,
        renewable_fraction=Decimal("0.0"),
        data_quality_score=Decimal("0.88"),
        metadata={"source": "heat_meter"}
    )


# ============================================================================
# EMISSION FACTOR FIXTURES
# ============================================================================

@pytest.fixture
def sample_wtt_factor() -> WTTEmissionFactor:
    """Create sample WTT emission factor for natural gas."""
    return WTTEmissionFactor(
        fuel_type=FuelType.NATURAL_GAS,
        country="US",
        region="Northeast",
        wtt_co2_kg_per_unit=Decimal("0.185"),  # kg CO2e per m3
        wtt_ch4_kg_per_unit=Decimal("0.0025"),
        wtt_n2o_kg_per_unit=Decimal("0.00008"),
        fuel_unit="m3",
        source="GREET_2023",
        year=2023,
        quality_tier=QualityTier.TIER_2,
        metadata={"upstream_location": "shale_gas"}
    )


@pytest.fixture
def sample_upstream_ef() -> UpstreamElectricityFactor:
    """Create sample upstream electricity emission factor for US."""
    return UpstreamElectricityFactor(
        country="US",
        region="Northeast",
        egrid_subregion="NEWE",
        upstream_co2_kg_per_kwh=Decimal("0.082"),
        upstream_ch4_kg_per_kwh=Decimal("0.00012"),
        upstream_n2o_kg_per_kwh=Decimal("0.000008"),
        source="EPA_eGRID_2023",
        year=2023,
        quality_tier=QualityTier.TIER_1,
        metadata={"grid_mix": "natural_gas_coal"}
    )


@pytest.fixture
def sample_td_loss_factor() -> TDLossFactor:
    """Create sample T&D loss factor for US (5%)."""
    return TDLossFactor(
        country="US",
        region="Northeast",
        egrid_subregion="NEWE",
        td_loss_percentage=Decimal("5.0"),
        source="EIA_2023",
        year=2023,
        metadata={"grid_age": "modern"}
    )


# ============================================================================
# SUPPLIER DATA FIXTURES
# ============================================================================

@pytest.fixture
def sample_supplier_data() -> SupplierFuelData:
    """Create sample supplier fuel data with EPD."""
    return SupplierFuelData(
        supplier_name="Green_Fuel_Corp",
        fuel_type=FuelType.DIESEL,
        total_lifecycle_emissions_kg_co2e_per_unit=Decimal("0.285"),  # kg CO2e per liter
        fuel_unit="L",
        has_epd=True,
        epd_reference="EPD-GFC-2023-001",
        verification_status="third_party_verified",
        validity_period="2024-01-01 to 2025-12-31",
        metadata={"certifier": "ISO_14025", "biodiesel_content": "7%"}
    )


# ============================================================================
# RESULT FIXTURES
# ============================================================================

@pytest.fixture
def sample_activity_3a_result() -> Activity3aResult:
    """Create sample Activity 3a calculation result."""
    return Activity3aResult(
        facility_id="FAC-001",
        reporting_period="2024-Q1",
        activity_type=ActivityType.ACTIVITY_3A,
        total_wtt_emissions_tco2e=Decimal("20.5"),
        co2_emissions_tco2e=Decimal("18.5"),
        ch4_emissions_tco2e=Decimal("1.8"),
        n2o_emissions_tco2e=Decimal("0.2"),
        fuel_breakdown=[
            FuelBreakdown(
                fuel_type=FuelType.NATURAL_GAS,
                fuel_quantity=Decimal("10000.0"),
                fuel_unit="m3",
                wtt_emissions_tco2e=Decimal("20.5"),
                co2_tco2e=Decimal("18.5"),
                ch4_tco2e=Decimal("1.8"),
                n2o_tco2e=Decimal("0.2"),
                calculation_method=CalculationMethod.FUEL_BASED,
                emission_factor_source="GREET_2023"
            )
        ],
        calculation_timestamp=datetime.now(timezone.utc),
        data_quality_score=Decimal("0.85"),
        provenance_hash="abc123def456",
        metadata={"country": "US"}
    )


@pytest.fixture
def sample_activity_3b_result() -> Activity3bResult:
    """Create sample Activity 3b calculation result."""
    return Activity3bResult(
        facility_id="FAC-006",
        reporting_period="2024-Q1",
        activity_type=ActivityType.ACTIVITY_3B,
        total_upstream_emissions_tco2e=Decimal("8.2"),
        co2_emissions_tco2e=Decimal("7.8"),
        ch4_emissions_tco2e=Decimal("0.35"),
        n2o_emissions_tco2e=Decimal("0.05"),
        electricity_breakdown=[
            ElectricityBreakdown(
                electricity_quantity=Decimal("100000.0"),
                electricity_unit="kWh",
                country="US",
                egrid_subregion="NEWE",
                upstream_emissions_tco2e=Decimal("8.2"),
                co2_tco2e=Decimal("7.8"),
                ch4_tco2e=Decimal("0.35"),
                n2o_tco2e=Decimal("0.05"),
                calculation_method=CalculationMethod.LOCATION_BASED,
                emission_factor_source="EPA_eGRID_2023"
            )
        ],
        calculation_timestamp=datetime.now(timezone.utc),
        data_quality_score=Decimal("0.92"),
        provenance_hash="xyz789ghi012",
        metadata={"supplier": "Regional_Utility"}
    )


@pytest.fixture
def sample_activity_3c_result() -> Activity3cResult:
    """Create sample Activity 3c calculation result."""
    return Activity3cResult(
        facility_id="FAC-006",
        reporting_period="2024-Q1",
        activity_type=ActivityType.ACTIVITY_3C,
        total_td_loss_emissions_tco2e=Decimal("2.5"),
        co2_emissions_tco2e=Decimal("2.3"),
        ch4_emissions_tco2e=Decimal("0.18"),
        n2o_emissions_tco2e=Decimal("0.02"),
        td_loss_breakdown=[
            TDLossBreakdown(
                electricity_quantity=Decimal("100000.0"),
                electricity_unit="kWh",
                country="US",
                egrid_subregion="NEWE",
                td_loss_percentage=Decimal("5.0"),
                td_loss_quantity_kwh=Decimal("5000.0"),
                td_loss_emissions_tco2e=Decimal("2.5"),
                co2_tco2e=Decimal("2.3"),
                ch4_tco2e=Decimal("0.18"),
                n2o_tco2e=Decimal("0.02"),
                calculation_method=CalculationMethod.LOCATION_BASED,
                emission_factor_source="EPA_eGRID_2023"
            )
        ],
        calculation_timestamp=datetime.now(timezone.utc),
        data_quality_score=Decimal("0.90"),
        provenance_hash="mno345pqr678",
        metadata={"grid_region": "NEWE"}
    )


@pytest.fixture
def sample_calculation_result(
    sample_activity_3a_result: Activity3aResult,
    sample_activity_3b_result: Activity3bResult,
    sample_activity_3c_result: Activity3cResult
) -> CalculationResult:
    """Create sample comprehensive calculation result."""
    return CalculationResult(
        facility_id="FAC-001",
        reporting_period="2024-Q1",
        total_scope3_category3_emissions_tco2e=Decimal("31.2"),
        activity_3a_emissions_tco2e=Decimal("20.5"),
        activity_3b_emissions_tco2e=Decimal("8.2"),
        activity_3c_emissions_tco2e=Decimal("2.5"),
        total_co2_tco2e=Decimal("28.6"),
        total_ch4_tco2e=Decimal("2.33"),
        total_n2o_tco2e=Decimal("0.27"),
        gas_breakdown=GasBreakdown(
            co2_tco2e=Decimal("28.6"),
            ch4_tco2e=Decimal("2.33"),
            n2o_tco2e=Decimal("0.27"),
            ch4_percentage=Decimal("7.47"),
            n2o_percentage=Decimal("0.87")
        ),
        activity_3a_result=sample_activity_3a_result,
        activity_3b_result=sample_activity_3b_result,
        activity_3c_result=sample_activity_3c_result,
        calculation_timestamp=datetime.now(timezone.utc),
        processing_time_ms=Decimal("125.5"),
        data_quality_score=Decimal("0.89"),
        provenance_hash="final_hash_123",
        metadata={"total_records": 3}
    )


# ============================================================================
# COMPLIANCE FIXTURES
# ============================================================================

@pytest.fixture
def sample_compliance_result() -> ComplianceCheckResult:
    """Create sample compliance check result."""
    return ComplianceCheckResult(
        facility_id="FAC-001",
        reporting_period="2024-Q1",
        frameworks_checked=[
            RegulatoryFramework.GHG_PROTOCOL,
            RegulatoryFramework.ISO_14064,
            RegulatoryFramework.CSRD
        ],
        overall_compliant=True,
        compliance_score=Decimal("0.95"),
        framework_results={
            RegulatoryFramework.GHG_PROTOCOL: FrameworkDisclosure(
                framework=RegulatoryFramework.GHG_PROTOCOL,
                compliant=True,
                disclosure_requirements=[
                    "Activity 3a WTT emissions calculated",
                    "Activity 3b upstream electricity calculated",
                    "Activity 3c T&D losses calculated",
                    "Gas-level breakdown provided"
                ],
                missing_requirements=[],
                recommendations=[
                    "Consider supplier-specific data for improved accuracy"
                ],
                compliance_percentage=Decimal("100.0")
            )
        },
        data_quality_sufficient=True,
        min_dqi_score=Decimal("0.85"),
        actual_dqi_score=Decimal("0.89"),
        checks_performed=12,
        checks_passed=12,
        checks_failed=0,
        warnings=[],
        errors=[],
        check_timestamp=datetime.now(timezone.utc),
        metadata={"auditor": "system"}
    )


# ============================================================================
# DQI FIXTURES
# ============================================================================

@pytest.fixture
def sample_dqi_assessment() -> DQIAssessment:
    """Create sample DQI assessment."""
    return DQIAssessment(
        facility_id="FAC-001",
        reporting_period="2024-Q1",
        overall_score=Decimal("0.89"),
        tier=QualityTier.TIER_2,
        dimension_scores={
            "completeness": DQIScore(
                dimension="completeness",
                score=Decimal("0.95"),
                weight=Decimal("0.20"),
                weighted_score=Decimal("0.19"),
                assessment="High - All required fields present",
                issues=[]
            ),
            "accuracy": DQIScore(
                dimension="accuracy",
                score=Decimal("0.88"),
                weight=Decimal("0.25"),
                weighted_score=Decimal("0.22"),
                assessment="Good - Tier 2 emission factors used",
                issues=[]
            ),
            "consistency": DQIScore(
                dimension="consistency",
                score=Decimal("0.90"),
                weight=Decimal("0.15"),
                weighted_score=Decimal("0.135"),
                assessment="Good - No conflicting data",
                issues=[]
            ),
            "timeliness": DQIScore(
                dimension="timeliness",
                score=Decimal("0.92"),
                weight=Decimal("0.15"),
                weighted_score=Decimal("0.138"),
                assessment="Good - Data from current year",
                issues=[]
            ),
            "reliability": DQIScore(
                dimension="reliability",
                score=Decimal("0.85"),
                weight=Decimal("0.25"),
                weighted_score=Decimal("0.2125"),
                assessment="Good - Verified sources",
                issues=[]
            )
        },
        improvement_recommendations=[
            "Consider obtaining supplier-specific emission factors for Tier 3 accuracy",
            "Implement automated data validation to reduce manual errors"
        ],
        assessment_timestamp=datetime.now(timezone.utc),
        metadata={"assessor": "dqi_engine"}
    )


# ============================================================================
# BATCH FIXTURES
# ============================================================================

@pytest.fixture
def sample_batch_request(
    sample_fuel_record: FuelConsumptionRecord,
    sample_electricity_record: ElectricityConsumptionRecord
) -> BatchRequest:
    """Create sample batch calculation request."""
    return BatchRequest(
        batch_id="BATCH-2024-Q1-001",
        fuel_records=[sample_fuel_record],
        electricity_records=[sample_electricity_record],
        calculation_method=CalculationMethod.FUEL_BASED,
        include_uncertainty=True,
        include_dqi=True,
        compliance_frameworks=[
            RegulatoryFramework.GHG_PROTOCOL,
            RegulatoryFramework.ISO_14064
        ],
        metadata={
            "requested_by": "data_team",
            "request_timestamp": datetime.now(timezone.utc).isoformat()
        }
    )


# ============================================================================
# CONFIGURATION FIXTURES
# ============================================================================

@pytest.fixture
def test_database_url() -> str:
    """Return test database URL."""
    return "postgresql://test_user:test_pass@localhost:5432/test_fuel_energy"


@pytest.fixture
def test_config_dict() -> Dict[str, Any]:
    """Return test configuration dictionary."""
    return {
        "database_url": "postgresql://test_user:test_pass@localhost:5432/test_fuel_energy",
        "enable_caching": True,
        "cache_ttl_seconds": 300,
        "batch_size": 100,
        "max_workers": 4,
        "api_timeout_seconds": 30,
        "enable_metrics": True,
        "enable_provenance": True,
        "dqi_min_score": 0.7,
        "compliance_frameworks": ["GHG_PROTOCOL", "ISO_14064"]
    }


# ============================================================================
# MOCK DATA GENERATORS
# ============================================================================

@pytest.fixture
def generate_fuel_records():
    """Factory fixture to generate multiple fuel consumption records."""
    def _generate(count: int, fuel_type: FuelType = FuelType.NATURAL_GAS) -> List[FuelConsumptionRecord]:
        records = []
        for i in range(count):
            record = FuelConsumptionRecord(
                facility_id=f"FAC-{i+1:03d}",
                reporting_period="2024-Q1",
                fuel_type=fuel_type,
                fuel_quantity=Decimal(str(10000.0 + i * 1000)),
                fuel_quantity_unit="m3",
                activity_type=ActivityType.ACTIVITY_3A,
                country="US",
                region="Northeast",
                sector="Manufacturing",
                calculation_method=CalculationMethod.FUEL_BASED,
                has_renewable_content=False,
                renewable_fraction=Decimal("0.0"),
                data_quality_score=Decimal("0.85"),
                metadata={"batch": i}
            )
            records.append(record)
        return records
    return _generate


@pytest.fixture
def generate_electricity_records():
    """Factory fixture to generate multiple electricity consumption records."""
    def _generate(count: int) -> List[ElectricityConsumptionRecord]:
        records = []
        for i in range(count):
            record = ElectricityConsumptionRecord(
                facility_id=f"FAC-{i+1:03d}",
                reporting_period="2024-Q1",
                electricity_quantity=Decimal(str(100000.0 + i * 10000)),
                electricity_unit="kWh",
                activity_type=ActivityType.ACTIVITY_3B,
                country="US",
                region="Northeast",
                egrid_subregion="NEWE",
                electricity_source=ElectricitySource.GRID,
                supplier_name="Regional_Utility",
                calculation_method=CalculationMethod.LOCATION_BASED,
                has_renewable_content=False,
                renewable_fraction=Decimal("0.0"),
                data_quality_score=Decimal("0.92"),
                metadata={"batch": i}
            )
            records.append(record)
        return records
    return _generate
