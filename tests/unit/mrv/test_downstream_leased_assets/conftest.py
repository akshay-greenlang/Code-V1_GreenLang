# -*- coding: utf-8 -*-
"""
Pytest fixtures for AGENT-MRV-026: Downstream Leased Assets Agent.

Provides comprehensive test fixtures for Cat 13 (Downstream Leased Assets)
where the reporter OWNS assets and LEASES them TO tenants (reporter is LESSOR).

Key differences from Cat 8 (Upstream Leased Assets):
- Reporter is the LESSOR, not the lessee.
- Tenant data collection is needed (metered energy from tenants).
- Vacancy handling: base-load during vacant periods.
- Multi-tenant allocation: area, headcount, revenue based.
- Operational control boundary determines Cat 13 vs Scope 1/2.

Fixtures:
- 10 asset input fixtures (buildings, vehicles, equipment, IT, mixed)
- 8 emission factor database fixtures
- 7 mock engine fixtures
- Helper functions for building compliance and portfolio results
- Configuration objects and mock services

Usage:
    def test_something(sample_office_building, building_eui_benchmarks):
        result = calculate(sample_office_building, building_eui_benchmarks)
        assert result["total_co2e_kg"] > 0

Author: GL-TestEngineer
Date: February 2026
"""

from datetime import datetime, date
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock
import pytest


# ============================================================================
# GRACEFUL IMPORTS
# ============================================================================

try:
    from greenlang.downstream_leased_assets.downstream_asset_database import (
        DownstreamAssetDatabaseEngine,
    )
    DB_ENGINE_AVAILABLE = True
except ImportError:
    DB_ENGINE_AVAILABLE = False
    DownstreamAssetDatabaseEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.downstream_leased_assets.asset_specific_calculator import (
        AssetSpecificCalculatorEngine,
    )
    ASSET_CALC_AVAILABLE = True
except ImportError:
    ASSET_CALC_AVAILABLE = False
    AssetSpecificCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.downstream_leased_assets.average_data_calculator import (
        AverageDataCalculatorEngine,
    )
    AVG_CALC_AVAILABLE = True
except ImportError:
    AVG_CALC_AVAILABLE = False
    AverageDataCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.downstream_leased_assets.spend_based_calculator import (
        SpendBasedCalculatorEngine,
    )
    SPEND_CALC_AVAILABLE = True
except ImportError:
    SPEND_CALC_AVAILABLE = False
    SpendBasedCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.downstream_leased_assets.hybrid_aggregator import (
        HybridAggregatorEngine,
    )
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False
    HybridAggregatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.downstream_leased_assets.compliance_checker import (
        ComplianceCheckerEngine,
    )
    COMPLIANCE_AVAILABLE = True
except ImportError:
    COMPLIANCE_AVAILABLE = False
    ComplianceCheckerEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.downstream_leased_assets.downstream_leased_assets_pipeline import (
        DownstreamLeasedAssetsPipelineEngine,
    )
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    DownstreamLeasedAssetsPipelineEngine = None  # type: ignore[assignment,misc]

_SKIP_DB = pytest.mark.skipif(not DB_ENGINE_AVAILABLE, reason="DB engine not available")
_SKIP_ASSET = pytest.mark.skipif(not ASSET_CALC_AVAILABLE, reason="Asset calc not available")
_SKIP_AVG = pytest.mark.skipif(not AVG_CALC_AVAILABLE, reason="Average calc not available")
_SKIP_SPEND = pytest.mark.skipif(not SPEND_CALC_AVAILABLE, reason="Spend calc not available")
_SKIP_HYBRID = pytest.mark.skipif(not HYBRID_AVAILABLE, reason="Hybrid engine not available")
_SKIP_COMPLIANCE = pytest.mark.skipif(not COMPLIANCE_AVAILABLE, reason="Compliance not available")
_SKIP_PIPELINE = pytest.mark.skipif(not PIPELINE_AVAILABLE, reason="Pipeline not available")


# ============================================================================
# SINGLETON RESET AUTOUSE FIXTURE
# ============================================================================


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset all singleton engine instances before and after each test."""
    _do_reset()
    yield
    _do_reset()


def _do_reset():
    """Attempt to reset all engine singletons."""
    engines = [
        DownstreamAssetDatabaseEngine,
        AssetSpecificCalculatorEngine,
        AverageDataCalculatorEngine,
        SpendBasedCalculatorEngine,
        HybridAggregatorEngine,
        ComplianceCheckerEngine,
        DownstreamLeasedAssetsPipelineEngine,
    ]
    for eng in engines:
        if eng is not None:
            for attr_name in ("reset_instance", "reset_singleton", "_reset"):
                fn = getattr(eng, attr_name, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:
                        pass
                    break
    # Also reset config singleton
    try:
        from greenlang.downstream_leased_assets.config import _reset_config
        _reset_config()
    except (ImportError, AttributeError):
        pass


# ============================================================================
# BUILDING INPUT FIXTURES (Lessor perspective)
# ============================================================================


@pytest.fixture
def sample_office_building() -> Dict[str, Any]:
    """Office building OWNED by reporter, leased to tenants, 2500m2, temperate."""
    return {
        "asset_type": "building",
        "building_type": "office",
        "floor_area_sqm": Decimal("2500.00"),
        "climate_zone": "temperate",
        "energy_sources": {
            "electricity_kwh": Decimal("450000"),
            "natural_gas_kwh": Decimal("120000"),
        },
        "occupancy_months": 12,
        "allocation_method": "area",
        "num_tenants": 4,
        "tenant_occupied_area_sqm": Decimal("2200.00"),
        "common_area_sqm": Decimal("300.00"),
        "vacancy_rate": Decimal("0.12"),
        "region": "US",
        "lease_type": "operating",
        "operational_control": "tenant",
        "asset_id": "DLA-BLDG-001",
    }


@pytest.fixture
def sample_retail_building() -> Dict[str, Any]:
    """Retail building, 1800m2, tropical zone, single tenant."""
    return {
        "asset_type": "building",
        "building_type": "retail",
        "floor_area_sqm": Decimal("1800.00"),
        "climate_zone": "tropical",
        "energy_sources": {
            "electricity_kwh": Decimal("380000"),
        },
        "occupancy_months": 12,
        "allocation_method": "area",
        "num_tenants": 1,
        "tenant_occupied_area_sqm": Decimal("1800.00"),
        "common_area_sqm": Decimal("0.00"),
        "vacancy_rate": Decimal("0.00"),
        "region": "GB",
        "lease_type": "operating",
        "operational_control": "tenant",
        "asset_id": "DLA-BLDG-002",
    }


@pytest.fixture
def sample_warehouse() -> Dict[str, Any]:
    """Warehouse, 5000m2, continental/cold zone, partial year occupancy."""
    return {
        "asset_type": "building",
        "building_type": "warehouse",
        "floor_area_sqm": Decimal("5000.00"),
        "climate_zone": "cold",
        "energy_sources": {
            "electricity_kwh": Decimal("180000"),
            "natural_gas_kwh": Decimal("250000"),
        },
        "occupancy_months": 9,
        "allocation_method": "area",
        "num_tenants": 2,
        "tenant_occupied_area_sqm": Decimal("4500.00"),
        "common_area_sqm": Decimal("500.00"),
        "vacancy_rate": Decimal("0.25"),
        "region": "CA",
        "lease_type": "operating",
        "operational_control": "tenant",
        "asset_id": "DLA-BLDG-003",
    }


@pytest.fixture
def sample_data_center() -> Dict[str, Any]:
    """Data center, 1000m2, temperate, high-power, PUE 1.40."""
    return {
        "asset_type": "building",
        "building_type": "data_center",
        "floor_area_sqm": Decimal("1000.00"),
        "climate_zone": "temperate",
        "energy_sources": {
            "electricity_kwh": Decimal("3200000"),
        },
        "occupancy_months": 12,
        "pue": Decimal("1.40"),
        "allocation_method": "area",
        "num_tenants": 8,
        "tenant_occupied_area_sqm": Decimal("800.00"),
        "common_area_sqm": Decimal("200.00"),
        "vacancy_rate": Decimal("0.05"),
        "region": "US",
        "lease_type": "operating",
        "operational_control": "tenant",
        "asset_id": "DLA-BLDG-004",
    }


# ============================================================================
# VEHICLE INPUT FIXTURES (Fleet leased to tenants)
# ============================================================================


@pytest.fixture
def sample_vehicle_fleet() -> Dict[str, Any]:
    """Fleet of 10 medium diesel cars leased to tenants."""
    return {
        "asset_type": "vehicle",
        "vehicle_type": "medium_car",
        "fuel_type": "diesel",
        "fleet_size": 10,
        "annual_distance_km_per_vehicle": Decimal("25000"),
        "region": "US",
        "lease_type": "operating",
        "operational_control": "tenant",
        "asset_id": "DLA-VEH-001",
    }


@pytest.fixture
def sample_heavy_truck_fleet() -> Dict[str, Any]:
    """Fleet of 5 heavy diesel trucks leased to logistics tenant."""
    return {
        "asset_type": "vehicle",
        "vehicle_type": "heavy_truck",
        "fuel_type": "diesel",
        "fleet_size": 5,
        "annual_distance_km_per_vehicle": Decimal("80000"),
        "region": "US",
        "lease_type": "operating",
        "operational_control": "tenant",
        "asset_id": "DLA-VEH-002",
    }


# ============================================================================
# EQUIPMENT INPUT FIXTURES
# ============================================================================


@pytest.fixture
def sample_equipment() -> Dict[str, Any]:
    """Construction equipment, 200kW diesel, 2000 hrs/yr, leased out."""
    return {
        "asset_type": "equipment",
        "equipment_type": "construction",
        "rated_power_kw": Decimal("200"),
        "annual_operating_hours": 2000,
        "load_factor": Decimal("0.60"),
        "energy_source": "diesel",
        "region": "US",
        "lease_type": "operating",
        "operational_control": "tenant",
        "asset_id": "DLA-EQ-001",
    }


# ============================================================================
# IT ASSET INPUT FIXTURES
# ============================================================================


@pytest.fixture
def sample_server_rack() -> Dict[str, Any]:
    """20 servers in leased rack space, PUE 1.4."""
    return {
        "asset_type": "it_asset",
        "it_type": "server",
        "quantity": 20,
        "rated_power_w": Decimal("500"),
        "utilization_pct": Decimal("0.85"),
        "pue": Decimal("1.40"),
        "annual_hours": 8760,
        "region": "US",
        "lease_type": "operating",
        "operational_control": "tenant",
        "asset_id": "DLA-IT-001",
    }


# ============================================================================
# MIXED PORTFOLIO FIXTURES
# ============================================================================


@pytest.fixture
def sample_mixed_portfolio(
    sample_office_building,
    sample_vehicle_fleet,
    sample_equipment,
    sample_server_rack,
) -> List[Dict[str, Any]]:
    """Mixed portfolio: buildings + vehicles + equipment + IT assets."""
    return [
        sample_office_building,
        sample_vehicle_fleet,
        sample_equipment,
        sample_server_rack,
    ]


@pytest.fixture
def sample_spend_only_portfolio() -> List[Dict[str, Any]]:
    """Portfolio with only spend-based data (no metered energy)."""
    return [
        {
            "asset_type": "building",
            "method": "spend_based",
            "naics_code": "531120",
            "amount": Decimal("250000.00"),
            "currency": "USD",
            "reporting_year": 2024,
            "description": "Office building lease portfolio",
            "asset_id": "DLA-SPEND-001",
        },
        {
            "asset_type": "vehicle",
            "method": "spend_based",
            "naics_code": "532112",
            "amount": Decimal("180000.00"),
            "currency": "USD",
            "reporting_year": 2024,
            "description": "Vehicle lease portfolio",
            "asset_id": "DLA-SPEND-002",
        },
        {
            "asset_type": "equipment",
            "method": "spend_based",
            "naics_code": "532412",
            "amount": Decimal("95000.00"),
            "currency": "USD",
            "reporting_year": 2024,
            "description": "Equipment lease portfolio",
            "asset_id": "DLA-SPEND-003",
        },
    ]


# ============================================================================
# EMISSION FACTOR DATABASE FIXTURES
# ============================================================================


@pytest.fixture
def building_eui_benchmarks() -> Dict[str, Dict[str, Decimal]]:
    """Building EUI benchmarks by type and climate zone (kWh/sqm/yr)."""
    return {
        "office": {"tropical": Decimal("200"), "arid": Decimal("220"), "temperate": Decimal("180"), "cold": Decimal("230"), "warm": Decimal("210")},
        "retail": {"tropical": Decimal("250"), "arid": Decimal("270"), "temperate": Decimal("220"), "cold": Decimal("280"), "warm": Decimal("260")},
        "warehouse": {"tropical": Decimal("80"), "arid": Decimal("90"), "temperate": Decimal("70"), "cold": Decimal("110"), "warm": Decimal("85")},
        "data_center": {"tropical": Decimal("2500"), "arid": Decimal("2600"), "temperate": Decimal("2400"), "cold": Decimal("2300"), "warm": Decimal("2550")},
        "hotel": {"tropical": Decimal("300"), "arid": Decimal("320"), "temperate": Decimal("280"), "cold": Decimal("340"), "warm": Decimal("310")},
        "healthcare": {"tropical": Decimal("350"), "arid": Decimal("370"), "temperate": Decimal("330"), "cold": Decimal("400"), "warm": Decimal("360")},
        "education": {"tropical": Decimal("150"), "arid": Decimal("165"), "temperate": Decimal("140"), "cold": Decimal("190"), "warm": Decimal("155")},
        "industrial": {"tropical": Decimal("200"), "arid": Decimal("210"), "temperate": Decimal("190"), "cold": Decimal("240"), "warm": Decimal("205")},
    }


@pytest.fixture
def vehicle_emission_factors() -> Dict[str, Dict[str, Decimal]]:
    """Vehicle EFs by type and fuel (kgCO2e/km)."""
    return {
        "small_car": {"petrol": Decimal("0.14"), "diesel": Decimal("0.13"), "bev": Decimal("0.0")},
        "medium_car": {"petrol": Decimal("0.18"), "diesel": Decimal("0.17"), "hybrid": Decimal("0.12"), "bev": Decimal("0.0")},
        "large_car": {"petrol": Decimal("0.24"), "diesel": Decimal("0.22"), "bev": Decimal("0.0")},
        "suv": {"petrol": Decimal("0.26"), "diesel": Decimal("0.24"), "bev": Decimal("0.0")},
        "light_van": {"petrol": Decimal("0.21"), "diesel": Decimal("0.20"), "bev": Decimal("0.0")},
        "heavy_van": {"petrol": Decimal("0.28"), "diesel": Decimal("0.27"), "bev": Decimal("0.0")},
        "light_truck": {"diesel": Decimal("0.50"), "bev": Decimal("0.0")},
        "heavy_truck": {"diesel": Decimal("0.85"), "bev": Decimal("0.0")},
    }


@pytest.fixture
def equipment_factors() -> Dict[str, Dict[str, Decimal]]:
    """Equipment fuel consumption and load factors by type."""
    return {
        "manufacturing": {"default_load_factor": Decimal("0.75"), "fuel_consumption_factor": Decimal("0.28")},
        "construction": {"default_load_factor": Decimal("0.60"), "fuel_consumption_factor": Decimal("0.32")},
        "generator": {"default_load_factor": Decimal("0.80"), "fuel_consumption_factor": Decimal("0.30")},
        "agricultural": {"default_load_factor": Decimal("0.55"), "fuel_consumption_factor": Decimal("0.25")},
        "mining": {"default_load_factor": Decimal("0.70"), "fuel_consumption_factor": Decimal("0.35")},
        "hvac": {"default_load_factor": Decimal("0.65"), "fuel_consumption_factor": Decimal("0.22")},
    }


@pytest.fixture
def it_asset_factors() -> Dict[str, Dict[str, Decimal]]:
    """IT asset power ratings by type (watts)."""
    return {
        "server": {"typical_power_w": Decimal("500"), "standby_power_w": Decimal("100")},
        "network": {"typical_power_w": Decimal("350"), "standby_power_w": Decimal("80")},
        "storage": {"typical_power_w": Decimal("800"), "standby_power_w": Decimal("150")},
        "desktop": {"typical_power_w": Decimal("200"), "standby_power_w": Decimal("15")},
        "laptop": {"typical_power_w": Decimal("65"), "standby_power_w": Decimal("5")},
        "printer": {"typical_power_w": Decimal("120"), "standby_power_w": Decimal("10")},
        "copier": {"typical_power_w": Decimal("250"), "standby_power_w": Decimal("30")},
    }


@pytest.fixture
def grid_emission_factors() -> Dict[str, Decimal]:
    """Grid emission factors by country/region (kgCO2e/kWh)."""
    return {
        "US": Decimal("0.37170"),
        "GB": Decimal("0.21233"),
        "DE": Decimal("0.35810"),
        "FR": Decimal("0.05100"),
        "JP": Decimal("0.45710"),
        "CA": Decimal("0.12000"),
        "AU": Decimal("0.65600"),
        "IN": Decimal("0.70800"),
        "CN": Decimal("0.55500"),
        "BR": Decimal("0.07400"),
        "GLOBAL": Decimal("0.43200"),
        "SRSO": Decimal("0.39210"),
    }


@pytest.fixture
def fuel_emission_factors() -> Dict[str, Decimal]:
    """Fuel emission factors (kgCO2e/litre or kgCO2e/kWh)."""
    return {
        "petrol": Decimal("2.31"),
        "diesel": Decimal("2.68"),
        "natural_gas": Decimal("0.18"),
        "lpg": Decimal("1.51"),
        "cng": Decimal("2.03"),
        "fuel_oil": Decimal("2.96"),
        "electricity": Decimal("0.37"),
        "district_heating": Decimal("0.25"),
    }


@pytest.fixture
def eeio_spend_factors() -> Dict[str, Dict[str, Any]]:
    """EEIO factors by NAICS code (kgCO2e/USD)."""
    return {
        "531120": {"name": "Lessors of buildings (non-residential)", "ef": Decimal("0.42")},
        "531130": {"name": "Lessors of miniwarehouses/self-storage", "ef": Decimal("0.38")},
        "531190": {"name": "Lessors of other real estate", "ef": Decimal("0.40")},
        "532111": {"name": "Passenger car rental", "ef": Decimal("0.55")},
        "532112": {"name": "Passenger car leasing", "ef": Decimal("0.52")},
        "532120": {"name": "Truck leasing", "ef": Decimal("0.58")},
        "532310": {"name": "General rental centers", "ef": Decimal("0.48")},
        "532412": {"name": "Construction equipment rental", "ef": Decimal("0.61")},
        "532490": {"name": "Other commercial equipment rental", "ef": Decimal("0.50")},
        "518210": {"name": "Data processing/hosting", "ef": Decimal("0.35")},
    }


@pytest.fixture
def vacancy_factors() -> Dict[str, Decimal]:
    """Base-load vacancy factors by building type (fraction of normal energy)."""
    return {
        "office": Decimal("0.25"),
        "retail": Decimal("0.20"),
        "warehouse": Decimal("0.15"),
        "data_center": Decimal("0.60"),
        "hotel": Decimal("0.30"),
        "healthcare": Decimal("0.35"),
        "education": Decimal("0.20"),
        "industrial": Decimal("0.18"),
    }


# ============================================================================
# MOCK ENGINE FIXTURES
# ============================================================================


@pytest.fixture
def mock_database_engine():
    """Mock DownstreamAssetDatabaseEngine."""
    mock = MagicMock()
    mock.get_building_eui.return_value = Decimal("180.00")
    mock.get_vehicle_emission_factor.return_value = Decimal("0.17")
    mock.get_grid_emission_factor.return_value = Decimal("0.37170")
    mock.get_fuel_emission_factor.return_value = Decimal("2.68")
    mock.get_equipment_benchmark.return_value = {
        "default_load_factor": Decimal("0.60"),
        "fuel_consumption_factor": Decimal("0.32"),
    }
    mock.get_it_power_rating.return_value = {
        "typical_power_w": Decimal("500"),
        "standby_power_w": Decimal("100"),
    }
    mock.get_eeio_factor.return_value = {"name": "Lessors of buildings", "ef": Decimal("0.42")}
    mock.get_currency_rate.return_value = Decimal("1.0")
    mock.get_cpi_deflator.return_value = Decimal("1.0000")
    mock.get_vacancy_factor.return_value = Decimal("0.25")
    mock.get_allocation_default.return_value = Decimal("1.0")
    mock.get_available_building_types.return_value = [
        "office", "retail", "warehouse", "data_center",
        "hotel", "healthcare", "education", "industrial",
    ]
    mock.get_available_vehicle_types.return_value = [
        "small_car", "medium_car", "large_car", "suv",
        "light_van", "heavy_van", "light_truck", "heavy_truck",
    ]
    mock.get_available_equipment_types.return_value = [
        "manufacturing", "construction", "generator",
        "agricultural", "mining", "hvac",
    ]
    mock.get_available_it_types.return_value = [
        "server", "network", "storage", "desktop",
        "laptop", "printer", "copier",
    ]
    mock.get_available_eeio_codes.return_value = [
        "531120", "531130", "531190", "532111", "532112",
        "532120", "532310", "532412", "532490", "518210",
    ]
    mock.get_available_climate_zones.return_value = [
        "tropical", "arid", "temperate", "cold", "warm",
    ]
    mock.get_summary.return_value = {"building_types": 8, "total_lookups": 0}
    mock.get_lookup_count.return_value = 0
    return mock


@pytest.fixture
def mock_asset_specific_engine():
    """Mock AssetSpecificCalculatorEngine."""
    mock = MagicMock()
    mock.calculate.return_value = {
        "total_co2e_kg": Decimal("167265.00"),
        "co2_kg": Decimal("160000.00"),
        "ch4_kg": Decimal("4265.00"),
        "n2o_kg": Decimal("3000.00"),
        "method": "asset_specific",
        "dqi_tier": "tier_1",
        "dqi_score": Decimal("4.5"),
        "uncertainty_pct": Decimal("0.05"),
        "provenance_hash": "a" * 64,
    }
    return mock


@pytest.fixture
def mock_average_data_engine():
    """Mock AverageDataCalculatorEngine."""
    mock = MagicMock()
    mock.calculate.return_value = {
        "total_co2e_kg": Decimal("145000.00"),
        "method": "average_data",
        "dqi_tier": "tier_2",
        "dqi_score": Decimal("3.0"),
        "uncertainty_pct": Decimal("0.25"),
        "provenance_hash": "b" * 64,
    }
    return mock


@pytest.fixture
def mock_spend_engine():
    """Mock SpendBasedCalculatorEngine."""
    mock = MagicMock()
    mock.calculate.return_value = {
        "total_co2e_kg": Decimal("105000.00"),
        "method": "spend_based",
        "dqi_tier": "tier_3",
        "dqi_score": Decimal("2.0"),
        "uncertainty_pct": Decimal("0.50"),
        "provenance_hash": "c" * 64,
    }
    return mock


@pytest.fixture
def mock_hybrid_engine():
    """Mock HybridAggregatorEngine."""
    mock = MagicMock()
    mock.aggregate.return_value = {
        "total_co2e_kg": Decimal("155000.00"),
        "method": "hybrid",
        "dqi_score": Decimal("3.5"),
        "by_category": {"building": Decimal("120000"), "vehicle": Decimal("35000")},
        "provenance_hash": "d" * 64,
    }
    return mock


@pytest.fixture
def mock_compliance_engine():
    """Mock ComplianceCheckerEngine."""
    mock = MagicMock()
    mock.check.return_value = {
        "overall_status": "pass",
        "score": Decimal("95.0"),
        "frameworks_checked": 7,
        "results": {},
    }
    return mock


@pytest.fixture
def mock_pipeline_engine():
    """Mock DownstreamLeasedAssetsPipelineEngine."""
    mock = MagicMock()
    mock.process.return_value = {
        "total_co2e_kg": Decimal("167265.00"),
        "provenance_hash": "e" * 64,
        "status": "completed",
    }
    return mock


# ============================================================================
# CONFIGURATION AND SERVICE MOCKS
# ============================================================================


@pytest.fixture
def mock_config():
    """Mock DownstreamLeasedConfig."""
    mock = MagicMock()
    mock.general.agent_id = "GL-MRV-S3-013"
    mock.general.agent_component = "AGENT-MRV-026"
    mock.general.version = "1.0.0"
    mock.general.table_prefix = "gl_dla_"
    mock.general.api_prefix = "/api/v1/downstream-leased-assets"
    mock.general.default_gwp = "AR5"
    mock.general.default_ef_source = "DEFRA"
    mock.general.enabled = True
    mock.general.debug = False
    mock.general.log_level = "INFO"
    mock.general.max_batch_size = 1000
    mock.database.cache_enabled = True
    mock.database.cache_ttl_seconds = 3600
    mock.database.quantize_decimals = 8
    mock.building.default_climate_zone = "temperate"
    mock.building.default_allocation_method = "area"
    mock.building.include_wtt = True
    mock.building.default_pue = Decimal("1.40")
    mock.building.max_floor_area_sqm = 500000
    mock.vehicle.include_wtt = True
    mock.vehicle.default_fuel_type = "diesel"
    mock.vehicle.max_annual_distance_km = 500000
    mock.equipment.include_wtt = True
    mock.equipment.max_operating_hours = 8760
    mock.equipment.default_load_factor = Decimal("0.70")
    mock.it_assets.default_pue = Decimal("1.40")
    mock.it_assets.include_cooling = True
    mock.it_assets.default_utilization = Decimal("0.50")
    mock.allocation.default_method = "area"
    mock.allocation.allow_revenue_allocation = True
    mock.compliance.strict_mode = False
    mock.compliance.materiality_threshold = Decimal("0.01")
    mock.compliance.get_frameworks.return_value = [
        "GHG_PROTOCOL", "ISO_14064", "CSRD_ESRS", "CDP", "SBTI", "SB_253", "GRI",
    ]
    mock.ef_source.primary_source = "DEFRA"
    mock.ef_source.fallback_source = "EPA"
    mock.uncertainty.method = "monte_carlo"
    mock.uncertainty.iterations = 10000
    mock.uncertainty.confidence_level = Decimal("0.95")
    mock.cache.enabled = True
    mock.cache.ttl_seconds = 3600
    mock.api.prefix = "/api/v1/downstream-leased-assets"
    mock.api.tags = ["downstream-leased-assets"]
    mock.api.rate_limit_per_minute = 120
    mock.provenance.enabled = True
    mock.provenance.hash_algorithm = "sha256"
    mock.provenance.chain_validation = True
    mock.metrics.enabled = True
    mock.metrics.prefix = "gl_dla_"
    mock.spend.base_currency = "USD"
    mock.spend.cpi_base_year = 2021
    mock.spend.margin_removal_enabled = True
    mock.vacancy.include_vacancy_emissions = True
    return mock


@pytest.fixture
def mock_service():
    """Mock DownstreamLeasedAssetsService."""
    mock = MagicMock()
    mock.agent_id = "GL-MRV-S3-013"
    mock.version = "1.0.0"
    mock.database_engine = MagicMock()
    mock.asset_specific_engine = MagicMock()
    mock.average_data_engine = MagicMock()
    mock.spend_based_engine = MagicMock()
    mock.hybrid_engine = MagicMock()
    mock.compliance_engine = MagicMock()
    mock.pipeline_engine = MagicMock()
    mock.calculate = AsyncMock(return_value={"total_co2e_kg": 42500.0, "provenance_hash": "a" * 64})
    mock.calculate_batch = AsyncMock(return_value={"results": [], "total_co2e_kg": 0.0})
    mock.calculate_building = AsyncMock(return_value={"total_co2e_kg": 42500.0})
    mock.calculate_vehicle = AsyncMock(return_value={"total_co2e_kg": 5250.0})
    mock.calculate_equipment = AsyncMock(return_value={"total_co2e_kg": 32500.0})
    mock.calculate_it_asset = AsyncMock(return_value={"total_co2e_kg": 2050.0})
    mock.check_compliance = AsyncMock(return_value={"overall_status": "pass", "score": 95.0})
    mock.get_emission_factors = AsyncMock(return_value={"building_eui": {}, "grid_efs": {}})
    mock.health_check = AsyncMock(return_value={"status": "healthy", "agent_id": "GL-MRV-S3-013"})
    mock.aggregate = AsyncMock(return_value={"total_co2e_kg": 82300.0})
    mock.get_uncertainty = AsyncMock(return_value={"lower_bound": 70000, "upper_bound": 95000})
    return mock


@pytest.fixture
def mock_metrics():
    """Mock Prometheus metrics collector."""
    mock = MagicMock()
    mock.observe_latency = MagicMock()
    mock.increment_counter = MagicMock()
    mock.set_gauge = MagicMock()
    return mock


@pytest.fixture
def mock_provenance():
    """Mock provenance tracker."""
    mock = MagicMock()
    mock.start_chain.return_value = "chain-001"
    mock.record_stage = MagicMock()
    mock.seal_chain.return_value = "a" * 64
    mock.validate_chain.return_value = True
    mock.get_current_chain.return_value = MagicMock(entries=[], final_hash="a" * 64)
    return mock


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def build_full_compliance_result(**overrides) -> Dict[str, Any]:
    """Build a fully-compliant result dict with all fields present."""
    base = {
        "total_co2e": 85000.0,
        "total_co2e_kg": Decimal("85000"),
        "method": "asset_specific",
        "calculation_method": "asset_specific",
        "ef_sources": ["DEFRA", "EPA"],
        "ef_source": "defra",
        "exclusions": "None - all asset categories included",
        "dqi_score": 4.0,
        "data_quality_score": 4.0,
        "reporting_period": "2024",
        "reporting_year": 2024,
        "uncertainty": {"lower": 76500, "upper": 93500, "confidence": 0.95},
        "asset_breakdown_provided": True,
        "asset_breakdown": {
            "building": {"count": 3, "co2e_kg": 60000},
            "vehicle": {"count": 10, "co2e_kg": 15000},
            "equipment": {"count": 2, "co2e_kg": 8000},
            "it_asset": {"count": 20, "co2e_kg": 2000},
        },
        "lease_classification_disclosed": True,
        "lease_type": "operating",
        "consolidation_approach": "operational_control",
        "operational_control": "tenant",
        "vacancy_handling": "base_load_included",
        "tenant_data_coverage": 0.85,
        "provenance_hash": "a" * 64,
    }
    base.update(overrides)
    return base


def build_portfolio_result(
    total_co2e_kg: Decimal = Decimal("250000"),
    by_category: Optional[Dict[str, Decimal]] = None,
    count: int = 15,
) -> Dict[str, Any]:
    """Build a portfolio aggregation result."""
    if by_category is None:
        by_category = {
            "building": Decimal("180000"),
            "vehicle": Decimal("45000"),
            "equipment": Decimal("15000"),
            "it_asset": Decimal("10000"),
        }
    return {
        "total_co2e_kg": total_co2e_kg,
        "by_category": by_category,
        "count": count,
        "reporting_period": "2024",
        "provenance_hash": "f" * 64,
    }
