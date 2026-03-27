# -*- coding: utf-8 -*-
"""
Pytest fixtures for AGENT-MRV-022: Downstream Transportation & Distribution Agent.

Provides comprehensive test fixtures for:
- Shipment inputs (road, rail, maritime, air, courier, last-mile)
- Spend inputs (EEIO spend-based with CPI deflation, multi-currency)
- Warehouse inputs (ambient DC, cold storage, fulfillment, cross-dock, retail)
- Last-mile inputs (van, cargo bike, drone, e-van, parcel locker, crowd-ship)
- Average data inputs (6 distribution channels)
- Cold chain inputs (chilled, frozen, pharma, fresh, ambient)
- Return logistics inputs (returns, redelivery, restocking)
- Incoterm inputs (11 Incoterms for boundary classification)
- Compliance inputs (7 frameworks)
- Configuration objects (15 frozen dataclass configs)
- Batch inputs (mixed method batches)
- Mock engines (database, distance, spend, average, warehouse, compliance, pipeline)

Usage:
    def test_something(sample_shipment, mock_database_engine):
        result = calculate(sample_shipment, mock_database_engine)
        assert result.emissions_tco2e > 0

Author: GL-TestEngineer
Date: February 2026
"""

from datetime import datetime, date
from decimal import Decimal
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock
import pytest

# ---------------------------------------------------------------------------
# Graceful imports with _AVAILABLE flag and _SKIP marker
# ---------------------------------------------------------------------------

_AVAILABLE = True
_SKIP = None

try:
    import greenlang.agents.mrv.downstream_transportation  # noqa: F401
except ImportError:
    _AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not _AVAILABLE,
    reason="greenlang.agents.mrv.downstream_transportation not installed or import failed",
)


# ---------------------------------------------------------------------------
# Autouse fixture: reset all singletons before and after each test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset all downstream transportation singletons before and after each test."""
    _do_reset()
    yield
    _do_reset()


def _do_reset():
    """Perform singleton reset for config, database, provenance, pipeline."""
    # Reset config singleton
    try:
        from greenlang.agents.mrv.downstream_transportation.config import reset_config
        reset_config()
    except (ImportError, AttributeError):
        pass

    # Reset database singleton
    try:
        from greenlang.agents.mrv.downstream_transportation.downstream_transport_database import (
            DownstreamTransportDatabaseEngine,
        )
        if hasattr(DownstreamTransportDatabaseEngine, "_instance"):
            DownstreamTransportDatabaseEngine._instance = None
        if hasattr(DownstreamTransportDatabaseEngine, "_instances"):
            DownstreamTransportDatabaseEngine._instances = {}
    except (ImportError, AttributeError):
        pass

    # Reset provenance tracker singleton
    try:
        from greenlang.agents.mrv.downstream_transportation.provenance import (
            get_provenance_tracker,
            ProvenanceTracker,
        )
        if hasattr(ProvenanceTracker, "_instance"):
            ProvenanceTracker._instance = None
    except (ImportError, AttributeError):
        pass

    # Reset pipeline singleton
    try:
        from greenlang.agents.mrv.downstream_transportation.downstream_transport_pipeline import (
            DownstreamTransportPipelineEngine,
        )
        if hasattr(DownstreamTransportPipelineEngine, "_instance"):
            DownstreamTransportPipelineEngine._instance = None
    except (ImportError, AttributeError):
        pass

    # Reset distance calculator singleton
    try:
        from greenlang.agents.mrv.downstream_transportation.distance_based_calculator import (
            DistanceBasedCalculatorEngine,
        )
        if hasattr(DistanceBasedCalculatorEngine, "_instance"):
            DistanceBasedCalculatorEngine._instance = None
    except (ImportError, AttributeError):
        pass

    # Reset spend calculator singleton
    try:
        from greenlang.agents.mrv.downstream_transportation.spend_based_calculator import (
            SpendBasedCalculatorEngine,
        )
        if hasattr(SpendBasedCalculatorEngine, "_instance"):
            SpendBasedCalculatorEngine._instance = None
    except (ImportError, AttributeError):
        pass

    # Reset average data calculator singleton
    try:
        from greenlang.agents.mrv.downstream_transportation.average_data_calculator import (
            AverageDataCalculatorEngine,
        )
        if hasattr(AverageDataCalculatorEngine, "_instance"):
            AverageDataCalculatorEngine._instance = None
    except (ImportError, AttributeError):
        pass

    # Reset warehouse engine singleton
    try:
        from greenlang.agents.mrv.downstream_transportation.warehouse_distribution import (
            WarehouseDistributionEngine,
        )
        if hasattr(WarehouseDistributionEngine, "_instance"):
            WarehouseDistributionEngine._instance = None
    except (ImportError, AttributeError):
        pass

    # Reset compliance checker singleton
    try:
        from greenlang.agents.mrv.downstream_transportation.compliance_checker import (
            ComplianceCheckerEngine,
        )
        if hasattr(ComplianceCheckerEngine, "_instance"):
            ComplianceCheckerEngine._instance = None
    except (ImportError, AttributeError):
        pass


# ============================================================================
# SHIPMENT INPUT FIXTURES
# ============================================================================

@pytest.fixture
def sample_shipment() -> Dict[str, Any]:
    """
    Road shipment: Articulated truck 33t, 350 km, 15 tonnes cargo.

    Represents outbound post-sale delivery via road freight:
    - Mode: Road (articulated truck 33t GVW)
    - Distance: 350 km (regional distribution)
    - Cargo: 15 tonnes (partial load)
    - Incoterm: DAP (seller bears transport risk)
    """
    return {
        "shipment_id": "DTO-SHIP-001",
        "mode": "ROAD",
        "vehicle_type": "ARTICULATED_33T",
        "distance_km": Decimal("350.0"),
        "cargo_mass_tonnes": Decimal("15.0"),
        "origin": "Distribution Center, Chicago",
        "destination": "Retail Store, Detroit",
        "laden": True,
        "temperature_controlled": False,
        "incoterm": "DAP",
        "reporting_period": "2025-Q1",
        "tenant_id": "tenant-dto-001",
        "facility_id": "facility-dc-001",
        "gwp_version": "AR5",
        "ef_scope": "WTW",
    }


@pytest.fixture
def sample_shipment_rail() -> Dict[str, Any]:
    """Electric freight rail, 800 km, 100 tonnes cargo."""
    return {
        "shipment_id": "DTO-SHIP-002",
        "mode": "RAIL",
        "vehicle_type": "ELECTRIC_FREIGHT",
        "distance_km": Decimal("800.0"),
        "cargo_mass_tonnes": Decimal("100.0"),
        "origin": "Factory, Stuttgart",
        "destination": "DC, Hamburg",
        "laden": True,
        "temperature_controlled": False,
        "incoterm": "DDP",
        "reporting_period": "2025-Q1",
        "tenant_id": "tenant-dto-001",
        "facility_id": "facility-fac-001",
        "gwp_version": "AR5",
        "ef_scope": "WTW",
        "country": "DE",
    }


@pytest.fixture
def sample_shipment_maritime() -> Dict[str, Any]:
    """Container ship Panamax, 12000 km, 50 tonnes cargo."""
    return {
        "shipment_id": "DTO-SHIP-003",
        "mode": "MARITIME",
        "vessel_type": "CONTAINER_PANAMAX",
        "distance_km": Decimal("12000.0"),
        "cargo_mass_tonnes": Decimal("50.0"),
        "origin": "Shanghai, CN",
        "destination": "Los Angeles, US",
        "laden": True,
        "temperature_controlled": False,
        "incoterm": "CIF",
        "reporting_period": "2025-Q1",
        "tenant_id": "tenant-dto-001",
        "facility_id": "facility-port-001",
        "gwp_version": "AR5",
        "ef_scope": "WTW",
    }


@pytest.fixture
def sample_shipment_air() -> Dict[str, Any]:
    """Widebody freighter, 6000 km, 3 tonnes cargo."""
    return {
        "shipment_id": "DTO-SHIP-004",
        "mode": "AIR",
        "aircraft_type": "WIDEBODY_FREIGHTER",
        "distance_km": Decimal("6000.0"),
        "cargo_mass_tonnes": Decimal("3.0"),
        "origin": "Frankfurt, DE",
        "destination": "New York, US",
        "laden": True,
        "temperature_controlled": True,
        "incoterm": "DAP",
        "reporting_period": "2025-Q1",
        "tenant_id": "tenant-dto-001",
        "facility_id": "facility-air-001",
        "gwp_version": "AR5",
        "ef_scope": "WTW",
    }


@pytest.fixture
def sample_shipment_courier() -> Dict[str, Any]:
    """Courier / express parcel, 150 km, 0.02 tonnes cargo."""
    return {
        "shipment_id": "DTO-SHIP-005",
        "mode": "COURIER",
        "vehicle_type": "VAN_MEDIUM",
        "distance_km": Decimal("150.0"),
        "cargo_mass_tonnes": Decimal("0.020"),
        "origin": "Fulfillment Center, London",
        "destination": "Customer, Birmingham",
        "laden": True,
        "temperature_controlled": False,
        "incoterm": "DDP",
        "reporting_period": "2025-Q1",
        "tenant_id": "tenant-dto-001",
        "facility_id": "facility-fc-001",
        "gwp_version": "AR5",
        "ef_scope": "WTW",
    }


@pytest.fixture
def sample_shipment_last_mile() -> Dict[str, Any]:
    """Last-mile delivery van, 25 km, 0.01 tonnes cargo."""
    return {
        "shipment_id": "DTO-SHIP-006",
        "mode": "LAST_MILE",
        "vehicle_type": "VAN_SMALL",
        "distance_km": Decimal("25.0"),
        "cargo_mass_tonnes": Decimal("0.010"),
        "origin": "Local Depot, Munich",
        "destination": "Customer, Munich suburbs",
        "laden": True,
        "temperature_controlled": False,
        "delivery_area": "URBAN",
        "incoterm": "DDP",
        "reporting_period": "2025-Q1",
        "tenant_id": "tenant-dto-001",
        "facility_id": "facility-depot-001",
        "gwp_version": "AR5",
        "ef_scope": "WTW",
    }


# ============================================================================
# SPEND INPUT FIXTURES
# ============================================================================

@pytest.fixture
def sample_spend() -> Dict[str, Any]:
    """Spend-based: outbound logistics $75,000 USD, NAICS 484110."""
    return {
        "spend_id": "DTO-SPEND-001",
        "calculation_method": "SPEND_BASED",
        "spend_amount": Decimal("75000.00"),
        "currency": "USD",
        "sector_code": "484110",
        "sector_classification": "NAICS",
        "eeio_database": "USEEIO_2.0",
        "reporting_year": 2024,
        "reporting_period": "2025-Q1",
        "tenant_id": "tenant-dto-001",
        "facility_id": "facility-dc-001",
        "gwp_version": "AR5",
        "description": "Outbound freight trucking spend Q1",
    }


@pytest.fixture
def sample_spend_eur() -> Dict[str, Any]:
    """Spend-based: warehousing EUR 45,000, NAICS 493110."""
    return {
        "spend_id": "DTO-SPEND-002",
        "calculation_method": "SPEND_BASED",
        "spend_amount": Decimal("45000.00"),
        "currency": "EUR",
        "sector_code": "493110",
        "sector_classification": "NAICS",
        "eeio_database": "EXIOBASE_3",
        "reporting_year": 2024,
        "reporting_period": "2025-Q1",
        "tenant_id": "tenant-dto-001",
        "facility_id": "facility-dc-002",
        "gwp_version": "AR5",
        "description": "EU warehouse distribution spend Q1",
    }


@pytest.fixture
def sample_spend_gbp() -> Dict[str, Any]:
    """Spend-based: courier services GBP 12,000, NAICS 492110."""
    return {
        "spend_id": "DTO-SPEND-003",
        "calculation_method": "SPEND_BASED",
        "spend_amount": Decimal("12000.00"),
        "currency": "GBP",
        "sector_code": "492110",
        "sector_classification": "NAICS",
        "eeio_database": "USEEIO_2.0",
        "reporting_year": 2024,
        "reporting_period": "2025-Q1",
        "tenant_id": "tenant-dto-001",
        "facility_id": "facility-fc-001",
        "gwp_version": "AR5",
        "description": "UK courier services spend Q1",
    }


# ============================================================================
# WAREHOUSE INPUT FIXTURES
# ============================================================================

@pytest.fixture
def sample_warehouse() -> Dict[str, Any]:
    """Ambient distribution center, 5000 sqm, 72h dwell time."""
    return {
        "warehouse_id": "DTO-WH-001",
        "warehouse_type": "DISTRIBUTION_CENTER",
        "floor_area_m2": Decimal("5000.0"),
        "dwell_time_hours": Decimal("72.0"),
        "energy_intensity_kwh_m2_year": Decimal("85.0"),
        "grid_intensity_kgco2e_kwh": Decimal("0.42"),
        "temperature_controlled": False,
        "region": "US",
        "tenant_id": "tenant-dto-001",
        "reporting_period": "2025-Q1",
    }


@pytest.fixture
def sample_warehouse_cold() -> Dict[str, Any]:
    """Cold storage facility, 2000 sqm, 48h dwell time, R-404A."""
    return {
        "warehouse_id": "DTO-WH-002",
        "warehouse_type": "COLD_STORAGE",
        "floor_area_m2": Decimal("2000.0"),
        "dwell_time_hours": Decimal("48.0"),
        "energy_intensity_kwh_m2_year": Decimal("180.0"),
        "grid_intensity_kgco2e_kwh": Decimal("0.42"),
        "temperature_controlled": True,
        "temperature_range": "CHILLED",
        "refrigerant_type": "R-404A",
        "refrigerant_charge_kg": Decimal("35.0"),
        "annual_leak_rate": Decimal("0.10"),
        "region": "US",
        "tenant_id": "tenant-dto-001",
        "reporting_period": "2025-Q1",
    }


@pytest.fixture
def sample_warehouse_fulfillment() -> Dict[str, Any]:
    """E-commerce fulfillment center, 15000 sqm, high throughput."""
    return {
        "warehouse_id": "DTO-WH-003",
        "warehouse_type": "FULFILLMENT_CENTER",
        "floor_area_m2": Decimal("15000.0"),
        "dwell_time_hours": Decimal("24.0"),
        "energy_intensity_kwh_m2_year": Decimal("110.0"),
        "grid_intensity_kgco2e_kwh": Decimal("0.42"),
        "temperature_controlled": False,
        "region": "US",
        "tenant_id": "tenant-dto-001",
        "reporting_period": "2025-Q1",
    }


@pytest.fixture
def sample_warehouse_retail() -> Dict[str, Any]:
    """Retail store storage, 500 sqm, partial cold."""
    return {
        "warehouse_id": "DTO-WH-004",
        "warehouse_type": "RETAIL_STORAGE",
        "floor_area_m2": Decimal("500.0"),
        "dwell_time_hours": Decimal("168.0"),
        "energy_intensity_kwh_m2_year": Decimal("200.0"),
        "grid_intensity_kgco2e_kwh": Decimal("0.42"),
        "temperature_controlled": True,
        "temperature_range": "CHILLED",
        "region": "US",
        "tenant_id": "tenant-dto-001",
        "reporting_period": "2025-Q1",
    }


@pytest.fixture
def sample_warehouse_cross_dock() -> Dict[str, Any]:
    """Cross-dock hub, 3000 sqm, 4h dwell time."""
    return {
        "warehouse_id": "DTO-WH-005",
        "warehouse_type": "CROSS_DOCK",
        "floor_area_m2": Decimal("3000.0"),
        "dwell_time_hours": Decimal("4.0"),
        "energy_intensity_kwh_m2_year": Decimal("60.0"),
        "grid_intensity_kgco2e_kwh": Decimal("0.42"),
        "temperature_controlled": False,
        "region": "US",
        "tenant_id": "tenant-dto-001",
        "reporting_period": "2025-Q1",
    }


# ============================================================================
# LAST-MILE INPUT FIXTURES
# ============================================================================

@pytest.fixture
def sample_last_mile() -> Dict[str, Any]:
    """Last-mile van delivery, urban, 15 km."""
    return {
        "delivery_id": "DTO-LM-001",
        "vehicle_type": "VAN_DIESEL",
        "delivery_area": "URBAN",
        "distance_km": Decimal("15.0"),
        "parcels_delivered": 25,
        "cargo_mass_kg": Decimal("200.0"),
        "region": "US",
        "tenant_id": "tenant-dto-001",
        "reporting_period": "2025-Q1",
    }


@pytest.fixture
def sample_last_mile_urban() -> Dict[str, Any]:
    """Urban last-mile cargo bike delivery."""
    return {
        "delivery_id": "DTO-LM-002",
        "vehicle_type": "CARGO_BIKE",
        "delivery_area": "URBAN",
        "distance_km": Decimal("8.0"),
        "parcels_delivered": 15,
        "cargo_mass_kg": Decimal("60.0"),
        "region": "DE",
        "tenant_id": "tenant-dto-001",
        "reporting_period": "2025-Q1",
    }


@pytest.fixture
def sample_last_mile_suburban() -> Dict[str, Any]:
    """Suburban last-mile electric van delivery."""
    return {
        "delivery_id": "DTO-LM-003",
        "vehicle_type": "VAN_ELECTRIC",
        "delivery_area": "SUBURBAN",
        "distance_km": Decimal("25.0"),
        "parcels_delivered": 20,
        "cargo_mass_kg": Decimal("350.0"),
        "region": "US",
        "tenant_id": "tenant-dto-001",
        "reporting_period": "2025-Q1",
    }


@pytest.fixture
def sample_last_mile_rural() -> Dict[str, Any]:
    """Rural last-mile diesel van delivery."""
    return {
        "delivery_id": "DTO-LM-004",
        "vehicle_type": "VAN_DIESEL",
        "delivery_area": "RURAL",
        "distance_km": Decimal("55.0"),
        "parcels_delivered": 8,
        "cargo_mass_kg": Decimal("400.0"),
        "region": "US",
        "tenant_id": "tenant-dto-001",
        "reporting_period": "2025-Q1",
    }


# ============================================================================
# AVERAGE DATA INPUT FIXTURES
# ============================================================================

@pytest.fixture
def sample_average_data() -> Dict[str, Any]:
    """Average-data method: e-commerce direct-to-consumer channel."""
    return {
        "channel": "ECOMMERCE_DTC",
        "product_category": "consumer_electronics",
        "annual_revenue_usd": Decimal("5000000.00"),
        "annual_units_sold": 50000,
        "average_weight_kg": Decimal("2.5"),
        "region": "US",
        "reporting_period": "2025",
        "tenant_id": "tenant-dto-001",
    }


@pytest.fixture
def sample_average_data_retail() -> Dict[str, Any]:
    """Average-data method: retail distribution channel."""
    return {
        "channel": "RETAIL_DISTRIBUTION",
        "product_category": "food_beverage",
        "annual_revenue_usd": Decimal("12000000.00"),
        "annual_units_sold": 1000000,
        "average_weight_kg": Decimal("0.75"),
        "region": "US",
        "reporting_period": "2025",
        "tenant_id": "tenant-dto-001",
    }


@pytest.fixture
def sample_average_data_wholesale() -> Dict[str, Any]:
    """Average-data method: wholesale distribution channel."""
    return {
        "channel": "WHOLESALE",
        "product_category": "industrial_equipment",
        "annual_revenue_usd": Decimal("8000000.00"),
        "annual_units_sold": 2000,
        "average_weight_kg": Decimal("150.0"),
        "region": "EU",
        "reporting_period": "2025",
        "tenant_id": "tenant-dto-001",
    }


# ============================================================================
# CALCULATION INPUT FIXTURE
# ============================================================================

@pytest.fixture
def sample_calculation_input(sample_shipment) -> Dict[str, Any]:
    """Complete calculation request for distance-based method."""
    return {
        "request_id": "DTO-REQ-001",
        "calculation_method": "DISTANCE_BASED",
        "input_data": sample_shipment,
        "gwp_version": "AR5",
        "ef_scope": "WTW",
        "uncertainty_analysis": True,
        "monte_carlo_iterations": 1000,
        "include_provenance": True,
        "tenant_id": "tenant-dto-001",
        "user_id": "user-001",
        "requested_at": datetime.utcnow().isoformat(),
    }


# ============================================================================
# CONFIGURATION FIXTURE
# ============================================================================

@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Sample downstream transportation configuration."""
    return {
        "general": {
            "enabled": True,
            "debug": False,
            "log_level": "INFO",
            "agent_id": "GL-MRV-S3-009",
            "version": "1.0.0",
            "table_prefix": "gl_dto_",
            "max_retries": 3,
            "timeout": 300,
        },
        "database": {
            "url": "postgresql://test:test@localhost:5432/greenlang_test",
            "pool_size": 5,
            "max_overflow": 10,
        },
        "distance": {
            "default_ef_scope": "WTW",
            "include_wtt": True,
            "include_return_logistics": True,
            "default_load_factor": Decimal("0.70"),
        },
        "spend": {
            "default_eeio_database": "USEEIO_2.0",
            "cpi_base_year": 2021,
            "enable_cpi_deflation": True,
            "enable_margin_removal": False,
        },
        "warehouse": {
            "default_energy_intensity_kwh_m2_year": Decimal("85.0"),
            "default_grid_intensity_kgco2e_kwh": Decimal("0.42"),
            "cold_storage_uplift": Decimal("2.1"),
        },
        "last_mile": {
            "default_parcels_per_route": 25,
            "urban_stop_density": Decimal("3.5"),
            "suburban_stop_density": Decimal("1.5"),
            "rural_stop_density": Decimal("0.3"),
        },
        "compliance": {
            "enabled_frameworks": [
                "GHG_PROTOCOL", "ISO_14064", "ISO_14083",
                "CSRD", "CDP", "SBTI", "SB_253",
            ],
            "double_counting_check": True,
            "incoterm_boundary_enforcement": True,
        },
        "provenance": {
            "enabled": True,
            "hash_algorithm": "sha256",
            "include_merkle_tree": True,
        },
        "uncertainty": {
            "enabled": True,
            "method": "monte_carlo",
            "iterations": 1000,
        },
        "api": {
            "prefix": "/api/v1/downstream-transportation",
            "max_batch_size": 100,
            "rate_limit": 60,
        },
        "cache": {
            "enabled": True,
            "ttl_seconds": 3600,
            "max_entries": 10000,
        },
        "metrics": {
            "enabled": True,
            "prefix": "gl_dto_",
        },
        "ef_source": {
            "primary": "DEFRA_2024",
            "secondary": "EPA_2024",
            "fallback": "GLEC_2023",
        },
        "cold_chain": {
            "default_reefer_uplift": Decimal("1.20"),
            "default_refrigerant": "R-404A",
            "default_leak_rate": Decimal("0.10"),
        },
        "return_logistics": {
            "default_return_rate": Decimal("0.15"),
            "return_distance_factor": Decimal("1.0"),
            "restocking_emissions_per_unit_kgco2e": Decimal("0.5"),
        },
    }


# ============================================================================
# BATCH INPUT FIXTURES
# ============================================================================

@pytest.fixture
def sample_batch(
    sample_shipment,
    sample_shipment_rail,
    sample_spend,
) -> Dict[str, Any]:
    """Batch of 3 mixed-method calculation requests."""
    return {
        "batch_id": "DTO-BATCH-001",
        "requests": [
            {
                "request_id": "DTO-REQ-001",
                "calculation_method": "DISTANCE_BASED",
                "input_data": sample_shipment,
            },
            {
                "request_id": "DTO-REQ-002",
                "calculation_method": "DISTANCE_BASED",
                "input_data": sample_shipment_rail,
            },
            {
                "request_id": "DTO-REQ-003",
                "calculation_method": "SPEND_BASED",
                "input_data": sample_spend,
            },
        ],
        "parallel_processing": True,
        "max_workers": 4,
        "aggregate_results": True,
        "tenant_id": "tenant-dto-001",
        "user_id": "user-001",
        "requested_at": datetime.utcnow().isoformat(),
    }


# ============================================================================
# COMPLIANCE INPUT FIXTURES
# ============================================================================

@pytest.fixture
def sample_compliance_data() -> Dict[str, Any]:
    """Compliance check for all 7 frameworks."""
    return {
        "frameworks": [
            "GHG_PROTOCOL", "ISO_14064", "ISO_14083",
            "CSRD", "CDP", "SBTI", "SB_253",
        ],
        "calculation_results": [
            {"total_co2e": Decimal("250.0"), "method": "DISTANCE_BASED"},
        ],
        "incoterm": "DAP",
        "boundary_disclosed": True,
        "mode_breakdown_provided": True,
        "wtt_disclosed": True,
        "data_quality_score": Decimal("4.2"),
        "reporting_period": "2025",
        "tenant_id": "tenant-dto-001",
    }


# ============================================================================
# INCOTERM INPUT FIXTURES
# ============================================================================

@pytest.fixture
def sample_incoterm_data() -> Dict[str, Any]:
    """Incoterm boundary classification data for Cat 4 vs Cat 9."""
    return {
        "incoterm": "DAP",
        "seller_pays_transport": True,
        "seller_pays_insurance": False,
        "transport_risk_transfer_point": "destination",
        "classification": "CATEGORY_9",
        "all_incoterms": {
            "EXW": {"cat_9": True, "description": "Ex Works"},
            "FCA": {"cat_9": True, "description": "Free Carrier"},
            "FAS": {"cat_9": True, "description": "Free Alongside Ship"},
            "FOB": {"cat_9": True, "description": "Free on Board"},
            "CPT": {"cat_9": False, "description": "Carriage Paid To"},
            "CIF": {"cat_9": False, "description": "Cost Insurance Freight"},
            "CIP": {"cat_9": False, "description": "Carriage Insurance Paid"},
            "DAP": {"cat_9": False, "description": "Delivered at Place"},
            "DPU": {"cat_9": False, "description": "Delivered at Place Unloaded"},
            "DDP": {"cat_9": False, "description": "Delivered Duty Paid"},
            "CFR": {"cat_9": False, "description": "Cost and Freight"},
        },
    }


# ============================================================================
# COLD CHAIN INPUT FIXTURES
# ============================================================================

@pytest.fixture
def sample_cold_chain_data() -> Dict[str, Any]:
    """Cold chain shipment: chilled pharmaceutical transport."""
    return {
        "cold_chain_regime": "PHARMA",
        "temperature_range_c": {"min": Decimal("2.0"), "max": Decimal("8.0")},
        "refrigerant_type": "R-134A",
        "refrigerant_charge_kg": Decimal("8.0"),
        "annual_leak_rate": Decimal("0.08"),
        "reefer_fuel_uplift": Decimal("1.25"),
        "gwp_100": Decimal("1430"),
        "transport_mode": "ROAD",
        "distance_km": Decimal("500.0"),
        "cargo_mass_tonnes": Decimal("5.0"),
    }


# ============================================================================
# RETURN LOGISTICS INPUT FIXTURES
# ============================================================================

@pytest.fixture
def sample_return_data() -> Dict[str, Any]:
    """Return logistics data for e-commerce returns."""
    return {
        "return_rate": Decimal("0.20"),
        "original_distance_km": Decimal("350.0"),
        "return_distance_factor": Decimal("1.0"),
        "restocking_emissions_per_unit_kgco2e": Decimal("0.5"),
        "units_sold": 1000,
        "redelivery_rate": Decimal("0.05"),
        "redelivery_distance_km": Decimal("25.0"),
        "mode": "ROAD",
        "vehicle_type": "VAN_MEDIUM",
    }


# ============================================================================
# MOCK ENGINE FIXTURES
# ============================================================================

@pytest.fixture
def mock_database_engine() -> Mock:
    """Mock DownstreamTransportDatabaseEngine with all 18 lookup methods."""
    engine = MagicMock()

    # Transport EF lookup
    engine.get_transport_emission_factor = Mock(return_value={
        "ef_kgco2e_per_tonne_km": Decimal("0.10700"),
        "mode": "ROAD",
        "vehicle_type": "ARTICULATED_33T",
        "source": "DEFRA_2024",
        "scope": "WTW",
        "uncertainty": Decimal("0.15"),
    })

    # Cold chain EF lookup
    engine.get_cold_chain_factor = Mock(return_value={
        "reefer_uplift": Decimal("1.20"),
        "regime": "CHILLED",
        "mode": "ROAD",
    })

    # Warehouse EF lookup
    engine.get_warehouse_emission_factor = Mock(return_value={
        "ef_kgco2e_per_m2_year": Decimal("35.70"),
        "warehouse_type": "DISTRIBUTION_CENTER",
        "source": "DEFRA_2024",
    })

    # Last-mile EF lookup
    engine.get_last_mile_factor = Mock(return_value={
        "ef_kgco2e_per_parcel": Decimal("0.22"),
        "vehicle_type": "VAN_DIESEL",
        "area": "URBAN",
        "source": "DEFRA_2024",
    })

    # EEIO factor lookup
    engine.get_eeio_factor = Mock(return_value={
        "ef_kgco2e_per_usd": Decimal("0.4500"),
        "sector_code": "484110",
        "database": "USEEIO_2.0",
    })

    # Currency rate lookup
    engine.get_currency_rate = Mock(return_value=Decimal("1.0850"))

    # CPI deflator lookup
    engine.get_cpi_deflator = Mock(return_value=Decimal("1.1490"))

    # Grid EF lookup
    engine.get_grid_emission_factor = Mock(return_value=Decimal("0.42000"))

    # Channel average data lookup
    engine.get_channel_average = Mock(return_value={
        "ef_kgco2e_per_unit": Decimal("1.25"),
        "channel": "ECOMMERCE_DTC",
    })

    # Incoterm classification
    engine.get_incoterm_classification = Mock(return_value={
        "incoterm": "DAP",
        "category_9_applicable": True,
        "seller_bears_transport": True,
    })

    # Load factor lookup
    engine.get_load_factor = Mock(return_value=Decimal("0.70"))

    # Return factor lookup
    engine.get_return_factor = Mock(return_value={
        "return_rate": Decimal("0.15"),
        "return_distance_factor": Decimal("1.0"),
    })

    # DQI scoring
    engine.get_dqi_scoring = Mock(return_value={
        "dimensions": 5,
        "weights_sum": Decimal("1.0"),
    })

    # Uncertainty ranges
    engine.get_uncertainty_range = Mock(return_value={
        "lower_pct": Decimal("0.85"),
        "upper_pct": Decimal("1.15"),
    })

    # Comparison data
    engine.get_mode_comparison = Mock(return_value={
        "road": Decimal("0.10700"),
        "rail": Decimal("0.02800"),
        "maritime": Decimal("0.01600"),
        "air": Decimal("0.60200"),
    })

    # WTT emission factor
    engine.get_wtt_factor = Mock(return_value=Decimal("0.02500"))

    # Fleet emission factor
    engine.get_fleet_average_factor = Mock(return_value=Decimal("0.09500"))

    # Singleton identity
    engine._instance_id = id(engine)

    return engine


@pytest.fixture
def mock_distance_engine() -> Mock:
    """Mock DistanceBasedCalculatorEngine."""
    engine = MagicMock()
    engine.calculate_shipment = Mock(return_value={
        "emissions_tco2e": Decimal("0.56175"),
        "distance_km": Decimal("350.0"),
        "cargo_mass_tonnes": Decimal("15.0"),
        "ef_kgco2e_per_tonne_km": Decimal("0.10700"),
        "calculation_method": "DISTANCE_BASED",
        "provenance_hash": "a" * 64,
    })
    return engine


@pytest.fixture
def mock_spend_engine() -> Mock:
    """Mock SpendBasedCalculatorEngine."""
    engine = MagicMock()
    engine.calculate_spend = Mock(return_value={
        "emissions_tco2e": Decimal("33.750"),
        "spend_usd": Decimal("75000.00"),
        "ef_kgco2e_per_usd": Decimal("0.4500"),
        "calculation_method": "SPEND_BASED",
        "provenance_hash": "b" * 64,
    })
    return engine


@pytest.fixture
def mock_average_engine() -> Mock:
    """Mock AverageDataCalculatorEngine."""
    engine = MagicMock()
    engine.calculate_channel = Mock(return_value={
        "emissions_tco2e": Decimal("62.500"),
        "channel": "ECOMMERCE_DTC",
        "units_sold": 50000,
        "ef_per_unit": Decimal("1.25"),
        "calculation_method": "AVERAGE_DATA",
        "provenance_hash": "c" * 64,
    })
    return engine


@pytest.fixture
def mock_warehouse_engine() -> Mock:
    """Mock WarehouseDistributionEngine."""
    engine = MagicMock()
    engine.calculate_warehouse = Mock(return_value={
        "emissions_tco2e": Decimal("3.478"),
        "warehouse_type": "DISTRIBUTION_CENTER",
        "floor_area_m2": Decimal("5000.0"),
        "dwell_time_hours": Decimal("72.0"),
        "provenance_hash": "d" * 64,
    })
    engine.calculate_last_mile = Mock(return_value={
        "emissions_tco2e": Decimal("5.500"),
        "deliveries": 25,
        "provenance_hash": "e" * 64,
    })
    return engine


@pytest.fixture
def mock_compliance_engine() -> Mock:
    """Mock ComplianceCheckerEngine."""
    engine = MagicMock()
    engine.check_compliance = Mock(return_value={
        "compliant": True,
        "framework": "GHG_PROTOCOL",
        "category": "SCOPE_3_CATEGORY_9",
        "issues": [],
        "warnings": [],
        "completeness_score": Decimal("0.95"),
        "data_quality_score": Decimal("0.88"),
    })
    return engine


@pytest.fixture
def mock_pipeline_engine() -> Mock:
    """Mock DownstreamTransportPipelineEngine."""
    engine = MagicMock()
    engine.process = Mock(return_value={
        "result": {
            "emissions_tco2e": Decimal("0.56175"),
            "calculation_method": "DISTANCE_BASED",
            "data_quality_score": Decimal("0.88"),
        },
        "provenance_hash": "f" * 64,
        "processing_time_ms": 125.0,
        "validation_status": "PASS",
    })
    return engine
