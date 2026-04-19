"""
PACK-046 Intensity Metrics Pack - Shared Test Fixtures
=======================================================

Provides shared fixtures for all PACK-046 test modules including
engine instances, sample data, configuration objects, and helper
utilities.  All numeric fixtures use Decimal for regulatory precision.

Author: GreenLang QA Team
Date: March 2026
"""

import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List

import pytest

# ---------------------------------------------------------------------------
# Path setup - ensure PACK-046 root is importable
# ---------------------------------------------------------------------------
PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))


# ---------------------------------------------------------------------------
# Engine Imports
# ---------------------------------------------------------------------------
from engines.denominator_registry_engine import (
    BUILT_IN_DENOMINATORS,
    DenominatorCategory,
    DenominatorDefinition,
    DenominatorRegistryEngine,
    DenominatorUnit,
    DenominatorValue,
    RegistryInput,
    RegistryResult,
    UNIT_CONVERSION_FACTORS,
    ValidationFinding,
    ValidationSeverity,
    get_built_in_denominators,
)
from engines.intensity_calculation_engine import (
    ConsolidatedIntensity,
    ConsolidationInput,
    EmissionsData,
    EntityContribution,
    EntityIntensityInput,
    IntensityCalculationEngine,
    IntensityInput,
    IntensityResult,
    IntensityStatus,
    IntensityTimeSeries,
    PeriodIntensity,
    ScopeInclusion,
    TimeSeriesInput,
    calculate_consolidated_intensity,
    calculate_intensity,
    SCOPE_3_CATEGORIES,
)
from config.pack_config import (
    AVAILABLE_PRESETS,
    DATA_QUALITY_UNCERTAINTY,
    SBTI_SECTOR_PATHWAYS,
    SECTOR_INFO,
    STANDARD_DENOMINATORS,
    AuditConfig,
    BenchmarkConfig,
    BenchmarkSource,
    ConsolidationApproach,
    DataQualityLevel,
    DecompositionConfig,
    DecompositionMethod,
    DenominatorConfig,
    DisclosureConfig,
    DisclosureFramework,
    IntensityCalculationConfig,
    IntensityMetricsConfig,
    IntensitySector,
    NotificationChannel,
    NotificationConfig,
    NullHandling,
    OutputFormat,
    PackConfig,
    PerformanceConfig,
    PropagationMethod,
    RegressionModel,
    ReportingConfig,
    ScenarioConfig,
    ScenarioType,
    SecurityConfig,
    TargetConfig,
    TargetPathway,
    TrendConfig,
    UncertaintyConfig,
    WeightedAverageMethod,
    get_default_config,
    get_denominator_info,
    get_sbti_pathway,
    get_sector_info,
    list_available_presets,
    validate_config,
)


# ---------------------------------------------------------------------------
# Engine Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def denominator_engine() -> DenominatorRegistryEngine:
    """Create a fresh DenominatorRegistryEngine instance."""
    return DenominatorRegistryEngine()


@pytest.fixture
def intensity_engine() -> IntensityCalculationEngine:
    """Create a fresh IntensityCalculationEngine instance."""
    return IntensityCalculationEngine()


# ---------------------------------------------------------------------------
# Organisation / Context Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_org_manufacturing() -> Dict[str, Any]:
    """Sample manufacturing organisation context."""
    return {
        "organisation_id": "org-mfg-001",
        "sector": "manufacturing",
        "frameworks": ["SBTi_SDA", "ESRS_E1_6", "CDP_C6_10", "GRI_305_4"],
        "available_data_ids": ["revenue_usd", "production_tonnes", "fte_employees"],
    }


@pytest.fixture
def sample_org_real_estate() -> Dict[str, Any]:
    """Sample real estate organisation context."""
    return {
        "organisation_id": "org-re-001",
        "sector": "real_estate",
        "frameworks": ["ESRS_E1_6", "CDP_C6_10", "CRREM", "GRESB"],
        "available_data_ids": ["revenue_usd", "floor_area_m2", "floor_area_sqft"],
    }


@pytest.fixture
def sample_org_power() -> Dict[str, Any]:
    """Sample power generation organisation context."""
    return {
        "organisation_id": "org-pwr-001",
        "sector": "power",
        "frameworks": ["SBTi_SDA", "CDP_C6_10"],
        "available_data_ids": ["electricity_mwh", "revenue_usd"],
    }


# ---------------------------------------------------------------------------
# Config Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_pack_config() -> IntensityMetricsConfig:
    """Create default IntensityMetricsConfig."""
    return IntensityMetricsConfig()


@pytest.fixture
def manufacturing_config() -> IntensityMetricsConfig:
    """Create manufacturing sector IntensityMetricsConfig."""
    return IntensityMetricsConfig(
        company_name="ACME Manufacturing",
        sector=IntensitySector.MANUFACTURING,
        reporting_year=2025,
        base_year=2020,
        revenue_meur=500.0,
        employees_fte=2000,
        denominator=DenominatorConfig(
            selected_denominators=["tonnes_output", "revenue_meur"],
            primary_denominator="tonnes_output",
        ),
        intensity_calculation=IntensityCalculationConfig(
            scope_inclusion=ScopeInclusion.SCOPE_1_2_LOCATION,
        ),
    )


@pytest.fixture
def sme_config() -> IntensityMetricsConfig:
    """Create SME sector IntensityMetricsConfig (simplified)."""
    return IntensityMetricsConfig(
        company_name="Small Corp",
        sector=IntensitySector.SME,
        reporting_year=2025,
        base_year=2020,
        revenue_meur=10.0,
        employees_fte=50,
    )


# ---------------------------------------------------------------------------
# Denominator Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_denominator_values() -> List[DenominatorValue]:
    """Create sample denominator values for validation testing."""
    return [
        DenominatorValue(
            denominator_id="revenue_usd",
            period="2023",
            value=Decimal("500.0"),
            unit="USD_million",
            data_quality_score=2,
            source="ERP",
        ),
        DenominatorValue(
            denominator_id="revenue_usd",
            period="2024",
            value=Decimal("550.0"),
            unit="USD_million",
            data_quality_score=1,
            source="ERP",
        ),
        DenominatorValue(
            denominator_id="production_tonnes",
            period="2024",
            value=Decimal("100000"),
            unit="tonne",
            data_quality_score=2,
            source="MES",
        ),
    ]


@pytest.fixture
def sample_registry_input(sample_denominator_values) -> RegistryInput:
    """Create sample RegistryInput for engine testing."""
    return RegistryInput(
        organisation_id="org-test-001",
        sector="manufacturing",
        target_frameworks=["SBTi_SDA", "ESRS_E1_6", "CDP_C6_10"],
        denominator_values=sample_denominator_values,
        available_data_ids=["revenue_usd", "production_tonnes", "fte_employees"],
    )


# ---------------------------------------------------------------------------
# Emissions Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_emissions_full() -> EmissionsData:
    """Create full emissions data with all scopes populated."""
    return EmissionsData(
        scope_1_tco2e=Decimal("5000"),
        scope_2_location_tco2e=Decimal("3000"),
        scope_2_market_tco2e=Decimal("2500"),
        scope_3_tco2e=Decimal("15000"),
        scope_3_categories={
            1: Decimal("8000"),
            4: Decimal("3000"),
            6: Decimal("1000"),
            7: Decimal("500"),
            11: Decimal("2500"),
        },
    )


@pytest.fixture
def sample_emissions_scope1_only() -> EmissionsData:
    """Create emissions data with only Scope 1."""
    return EmissionsData(
        scope_1_tco2e=Decimal("5000"),
    )


@pytest.fixture
def sample_emissions_partial() -> EmissionsData:
    """Create emissions data with partial scope data."""
    return EmissionsData(
        scope_1_tco2e=Decimal("5000"),
        scope_2_location_tco2e=Decimal("3000"),
    )


# ---------------------------------------------------------------------------
# Intensity Input Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_intensity_input(sample_emissions_full) -> IntensityInput:
    """Create a standard intensity calculation input."""
    return IntensityInput(
        entity_id="entity-001",
        period="2024",
        emissions=sample_emissions_full,
        denominator_value=Decimal("500"),
        denominator_unit="USD_million",
        denominator_id="revenue_usd",
        scope_inclusion=ScopeInclusion.SCOPE_1_2_LOCATION,
    )


@pytest.fixture
def sample_consolidation_input() -> ConsolidationInput:
    """Create a multi-entity consolidation input."""
    return ConsolidationInput(
        consolidation_id="consol-001",
        period="2024",
        entities=[
            EntityIntensityInput(
                entity_id="bu-north",
                entity_name="North Division",
                emissions_tco2e=Decimal("3000"),
                denominator_value=Decimal("200"),
            ),
            EntityIntensityInput(
                entity_id="bu-south",
                entity_name="South Division",
                emissions_tco2e=Decimal("5000"),
                denominator_value=Decimal("300"),
            ),
            EntityIntensityInput(
                entity_id="bu-west",
                entity_name="West Division",
                emissions_tco2e=Decimal("2000"),
                denominator_value=Decimal("100"),
            ),
        ],
        denominator_unit="USD_million",
        denominator_id="revenue_usd",
    )


@pytest.fixture
def sample_time_series_input(sample_emissions_full) -> TimeSeriesInput:
    """Create a time-series intensity input."""
    periods = []
    for year, s1, s2, denom in [
        ("2021", "6000", "4000", "400"),
        ("2022", "5500", "3500", "450"),
        ("2023", "5200", "3200", "480"),
        ("2024", "5000", "3000", "500"),
    ]:
        periods.append(IntensityInput(
            entity_id="entity-001",
            period=year,
            emissions=EmissionsData(
                scope_1_tco2e=Decimal(s1),
                scope_2_location_tco2e=Decimal(s2),
            ),
            denominator_value=Decimal(denom),
            denominator_unit="USD_million",
            scope_inclusion=ScopeInclusion.SCOPE_1_2_LOCATION,
        ))
    return TimeSeriesInput(entity_id="entity-001", periods=periods)


# ---------------------------------------------------------------------------
# Dashboard / Report Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_dashboard_data() -> Dict[str, Any]:
    """Create sample data for executive dashboard rendering."""
    return {
        "company_name": "ACME Corp",
        "reporting_period": "FY2025",
        "intensity_metrics": [
            {
                "metric_name": "Revenue Intensity",
                "numerator_label": "tCO2e",
                "denominator_label": "M EUR",
                "current_value": 25.50,
                "prior_value": 28.00,
                "direction": "down",
                "status": "green",
                "sparkline": [
                    {"year": 2021, "value": 32.0},
                    {"year": 2022, "value": 30.5},
                    {"year": 2023, "value": 28.0},
                    {"year": 2024, "value": 25.5},
                ],
            },
            {
                "metric_name": "FTE Intensity",
                "numerator_label": "tCO2e",
                "denominator_label": "FTE",
                "current_value": 4.20,
                "prior_value": 4.50,
                "direction": "down",
                "status": "amber",
            },
        ],
        "benchmark_results": [
            {
                "metric_name": "Revenue Intensity",
                "percentile_rank": 35.0,
                "peer_group": "EU Manufacturing",
                "peer_average": 30.0,
                "best_in_class": 15.0,
                "org_value": 25.5,
            },
        ],
        "target_status": [
            {
                "target_name": "SBTi 2030",
                "target_year": 2030,
                "target_value": 15.0,
                "current_value": 25.5,
                "base_value": 40.0,
                "pct_achieved": 58.0,
                "on_track": True,
                "status": "green",
            },
        ],
        "decomposition_summary": {
            "period_start": 2023,
            "period_end": 2024,
            "activity_effect_pct": 5.0,
            "structure_effect_pct": -2.0,
            "intensity_effect_pct": -12.0,
            "total_change_pct": -9.0,
            "key_driver": "Energy efficiency improvements",
        },
        "action_items": [
            {
                "priority": 1,
                "action": "Accelerate heat pump deployment",
                "expected_impact": "5% reduction in Scope 1",
                "owner": "Facilities Manager",
                "timeline": "Q2 2026",
            },
        ],
    }


@pytest.fixture
def sample_detailed_report_data() -> Dict[str, Any]:
    """Create sample data for detailed report rendering."""
    return {
        "company_name": "ACME Corp",
        "reporting_period": "FY2025",
        "methodology_description": (
            "Intensity metrics calculated using GHG Protocol Corporate "
            "Standard methodology. Emissions data sourced from MRV agents."
        ),
        "calculation_approach": "Scope 1+2 location-based / revenue denominator",
        "scope_configuration": {
            "scope_inclusion": "Scope 1 + Scope 2 (location-based)",
            "scope_3_categories": "Not included",
            "consolidation_approach": "Operational control",
        },
        "denominator_details": [
            {
                "denominator_id": "revenue_meur",
                "name": "Revenue",
                "unit": "M EUR",
                "value": 500.0,
                "source": "ERP (SAP)",
                "data_quality": "audited",
            },
        ],
        "intensity_by_scope": [
            {
                "scope": "Scope 1",
                "emissions_tco2e": 5000.0,
                "denominator_value": 500.0,
                "denominator_unit": "M EUR",
                "intensity_value": 10.0,
                "intensity_unit": "tCO2e/M EUR",
            },
            {
                "scope": "Scope 2 (location)",
                "emissions_tco2e": 3000.0,
                "denominator_value": 500.0,
                "denominator_unit": "M EUR",
                "intensity_value": 6.0,
                "intensity_unit": "tCO2e/M EUR",
            },
        ],
        "intensity_by_denominator": [
            {
                "denominator_name": "Revenue",
                "denominator_unit": "M EUR",
                "scope_1": 10.0,
                "scope_2_location": 6.0,
                "scope_2_market": 5.0,
                "total_s1_s2": 16.0,
            },
        ],
        "time_series": [
            {"year": 2022, "scope_1_intensity": 12.0, "scope_2_intensity": 7.0, "total_intensity": 19.0},
            {"year": 2023, "scope_1_intensity": 11.0, "scope_2_intensity": 6.5, "total_intensity": 17.5},
            {"year": 2024, "scope_1_intensity": 10.0, "scope_2_intensity": 6.0, "total_intensity": 16.0},
        ],
        "entity_breakdown": [
            {
                "entity_name": "Plant A",
                "emissions_tco2e": 4000.0,
                "denominator_value": 250.0,
                "intensity_value": 16.0,
                "share_of_total_pct": 50.0,
            },
            {
                "entity_name": "Plant B",
                "emissions_tco2e": 4000.0,
                "denominator_value": 250.0,
                "intensity_value": 16.0,
                "share_of_total_pct": 50.0,
            },
        ],
        "data_sources": [
            {
                "source_name": "SAP ERP",
                "source_type": "ERP",
                "coverage": "Revenue data",
                "last_updated": "2025-03-01",
                "quality_score": 95.0,
            },
        ],
        "limitations": [
            "Scope 3 emissions not included in this reporting period.",
            "Market-based Scope 2 uses residual mix factors.",
        ],
    }


# ---------------------------------------------------------------------------
# Pipeline / Orchestrator Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_pipeline_config() -> Dict[str, Any]:
    """Create sample pipeline configuration."""
    return {
        "company_name": "ACME Corp",
        "reporting_period": "2025",
        "denominator_types": ["revenue", "fte"],
        "scopes_included": ["scope_1", "scope_2"],
        "max_retries": 2,
        "retry_base_delay_s": 0.01,
        "enable_parallel": False,
        "timeout_per_phase_s": 30.0,
    }
