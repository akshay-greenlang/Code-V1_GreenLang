# -*- coding: utf-8 -*-
"""
Pytest fixtures for AGENT-MRV-017: Upstream Transportation & Distribution Agent.

Provides comprehensive test fixtures for:
- Transport legs (road, maritime, air, rail, pipeline)
- Transport hubs (logistics, warehouse, distribution)
- Transport chains (multi-leg with hubs)
- Input models (shipment, fuel, spend, supplier)
- Configuration objects (allocation, reefer, warehouse)
- Mock engines (database, distance, fuel, spend, compliance)

Usage:
    def test_something(sample_transport_leg, mock_database_engine):
        result = calculate(sample_transport_leg, mock_database_engine)
        assert result.emissions_tco2e > 0
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock
import pytest

# Note: Adjust imports when actual models are implemented
# from greenlang.upstream_transportation.models import (
#     CalculationMethod, TransportMode, RoadVehicleType, MaritimeVesselType,
#     AircraftType, TransportFuelType, Incoterm, HubType, PipelineStage,
#     TransportLeg, TransportHub, TransportChain, ShipmentInput,
#     FuelConsumptionInput, SpendInput, SupplierEmissionInput,
#     AllocationConfig, ReeferConfig, WarehouseConfig, CalculationRequest
# )
# from greenlang.upstream_transportation.config import UpstreamTransportationConfig


# ============================================================================
# TRANSPORT LEG FIXTURES
# ============================================================================

@pytest.fixture
def sample_transport_leg() -> Dict[str, Any]:
    """
    Sample road transport leg - Articulated truck 40-44 tonnes.

    Represents a typical long-haul road freight journey:
    - Vehicle: Articulated truck (DEFRA: HGV Articulated 40-44t)
    - Fuel: Diesel (B7 blend typical in EU)
    - Distance: 500 km (cross-country)
    - Cargo: 20 tonnes (45% capacity utilization)
    """
    return {
        "leg_id": "LEG-ROAD-001",
        "mode": "ROAD",
        "vehicle_type": "ARTICULATED_40_44T",
        "fuel_type": "DIESEL",
        "distance_km": Decimal("500.0"),
        "cargo_mass_tonnes": Decimal("20.0"),
        "origin_location": "Hamburg, DE",
        "destination_location": "Munich, DE",
        "laden": True,
        "temperature_controlled": False,
        "description": "Road freight Hamburg to Munich"
    }


@pytest.fixture
def sample_maritime_leg() -> Dict[str, Any]:
    """
    Sample maritime transport leg - Container ship Panamax.

    Represents a typical trans-Pacific container shipping route:
    - Vessel: Panamax container ship (4000-5000 TEU)
    - Fuel: HFO (Heavy Fuel Oil, typical for deep-sea)
    - Distance: 19,500 km (Asia-Europe route)
    - Cargo: 50 tonnes (2 TEU equivalent)
    """
    return {
        "leg_id": "LEG-SEA-001",
        "mode": "MARITIME",
        "vessel_type": "CONTAINER_PANAMAX",
        "fuel_type": "HFO",
        "distance_km": Decimal("19500.0"),
        "cargo_mass_tonnes": Decimal("50.0"),
        "origin_location": "Shanghai, CN",
        "destination_location": "Rotterdam, NL",
        "laden": True,
        "temperature_controlled": False,
        "description": "Container shipping Shanghai to Rotterdam"
    }


@pytest.fixture
def sample_air_leg() -> Dict[str, Any]:
    """
    Sample air transport leg - Widebody freighter.

    Represents a typical intercontinental air cargo flight:
    - Aircraft: Widebody freighter (B747F, B777F)
    - Fuel: Jet A-1 (standard aviation turbine fuel)
    - Distance: 8,000 km (US-Europe)
    - Cargo: 5 tonnes (high-value, time-sensitive)
    """
    return {
        "leg_id": "LEG-AIR-001",
        "mode": "AIR",
        "aircraft_type": "WIDEBODY_FREIGHTER",
        "fuel_type": "JET_FUEL",
        "distance_km": Decimal("8000.0"),
        "cargo_mass_tonnes": Decimal("5.0"),
        "origin_location": "New York, US",
        "destination_location": "Frankfurt, DE",
        "laden": True,
        "temperature_controlled": False,
        "description": "Air freight New York to Frankfurt"
    }


@pytest.fixture
def sample_rail_leg() -> Dict[str, Any]:
    """
    Sample rail transport leg - Electric freight train.

    Represents a typical European electric rail freight:
    - Train: Electric freight (EU average grid intensity)
    - Fuel: Electricity (grid mix)
    - Distance: 800 km (cross-border)
    - Cargo: 100 tonnes (bulk cargo)
    """
    return {
        "leg_id": "LEG-RAIL-001",
        "mode": "RAIL",
        "vehicle_type": "ELECTRIC_FREIGHT",
        "fuel_type": "ELECTRICITY",
        "distance_km": Decimal("800.0"),
        "cargo_mass_tonnes": Decimal("100.0"),
        "origin_location": "Lyon, FR",
        "destination_location": "Milan, IT",
        "laden": True,
        "temperature_controlled": False,
        "country": "EU",
        "description": "Electric rail freight Lyon to Milan"
    }


@pytest.fixture
def sample_pipeline_leg() -> Dict[str, Any]:
    """
    Sample pipeline transport leg - Refined petroleum products.

    Represents a typical refined products pipeline:
    - Product: Diesel/gasoline (refined products)
    - Distance: 200 km (regional distribution)
    - Cargo: 1,000 tonnes (continuous flow)
    - Stage: Distribution (refined products to terminals)
    """
    return {
        "leg_id": "LEG-PIPE-001",
        "mode": "PIPELINE",
        "pipeline_stage": "DISTRIBUTION_REFINED",
        "fuel_type": "DIESEL",
        "distance_km": Decimal("200.0"),
        "cargo_mass_tonnes": Decimal("1000.0"),
        "origin_location": "Refinery A",
        "destination_location": "Terminal B",
        "laden": True,
        "description": "Pipeline refined products distribution"
    }


# ============================================================================
# TRANSPORT HUB FIXTURES
# ============================================================================

@pytest.fixture
def sample_transport_hub() -> Dict[str, Any]:
    """
    Sample transport hub - Logistics hub (cross-dock).

    Represents a typical logistics hub with minimal storage:
    - Type: Logistics hub (transshipment)
    - Dwell time: 4 hours (cross-dock operation)
    - No refrigeration (ambient goods)
    - Handling: Forklift operations
    """
    return {
        "hub_id": "HUB-LOG-001",
        "hub_type": "LOGISTICS_HUB",
        "location": "Hamburg Logistics Center",
        "dwell_time_hours": Decimal("4.0"),
        "floor_area_m2": None,
        "temperature_controlled": False,
        "refrigerant_type": None,
        "handling_operations": ["forklift", "pallet_jack"],
        "description": "Cross-dock logistics hub"
    }


@pytest.fixture
def sample_warehouse_hub() -> Dict[str, Any]:
    """
    Sample warehouse hub - Cold storage facility.

    Represents a typical refrigerated warehouse:
    - Type: Cold storage warehouse
    - Floor area: 1,000 m² (allocated to shipment)
    - Dwell time: 48 hours (2-day storage)
    - Temperature: Chilled (0-5°C)
    - Refrigerant: R-404A (common in cold storage)
    """
    return {
        "hub_id": "HUB-WARE-001",
        "hub_type": "COLD_STORAGE_WAREHOUSE",
        "location": "Rotterdam Cold Store",
        "dwell_time_hours": Decimal("48.0"),
        "floor_area_m2": Decimal("1000.0"),
        "temperature_controlled": True,
        "refrigerant_type": "R-404A",
        "refrigerant_charge_kg": Decimal("50.0"),
        "annual_leak_rate": Decimal("0.10"),  # 10% annual leak rate
        "temperature_range": "CHILLED",
        "description": "Cold storage warehouse"
    }


# ============================================================================
# TRANSPORT CHAIN FIXTURES
# ============================================================================

@pytest.fixture
def sample_transport_chain(
    sample_transport_leg,
    sample_maritime_leg,
    sample_rail_leg,
    sample_transport_hub,
    sample_warehouse_hub
) -> Dict[str, Any]:
    """
    Sample multi-modal transport chain.

    Represents a complex supply chain journey:
    1. Road: Factory to port (truck, 500 km)
    2. Hub: Port logistics hub (4 hours)
    3. Maritime: Trans-Pacific shipping (19,500 km)
    4. Hub: Destination warehouse (48 hours, cold storage)
    5. Rail: Port to distribution center (800 km)
    6. Road: Final delivery (100 km, truck)

    Total distance: ~20,900 km
    Total hubs: 2
    Total legs: 4 (road/maritime/rail/road)
    """
    # Create final delivery leg
    final_delivery = sample_transport_leg.copy()
    final_delivery["leg_id"] = "LEG-ROAD-002"
    final_delivery["distance_km"] = Decimal("100.0")
    final_delivery["cargo_mass_tonnes"] = Decimal("20.0")
    final_delivery["description"] = "Final delivery to customer"

    return {
        "chain_id": "CHAIN-001",
        "shipment_id": "SHIPMENT-12345",
        "legs": [
            sample_transport_leg,
            sample_maritime_leg,
            sample_rail_leg,
            final_delivery
        ],
        "hubs": [
            sample_transport_hub,
            sample_warehouse_hub
        ],
        "total_distance_km": Decimal("20900.0"),
        "origin": "Hamburg, DE",
        "destination": "New York, US",
        "incoterm": "DDP",
        "description": "Multi-modal supply chain Hamburg to New York"
    }


# ============================================================================
# INPUT MODEL FIXTURES
# ============================================================================

@pytest.fixture
def sample_shipment_input(sample_transport_chain) -> Dict[str, Any]:
    """
    Sample shipment input for distance-based calculation.

    Complete shipment record with:
    - Multi-modal transport chain
    - Incoterm DDP (Category 4)
    - Mass-based allocation
    - Temperature control (reefer)
    """
    return {
        "shipment_id": "SHIPMENT-12345",
        "calculation_method": "DISTANCE_BASED",
        "transport_chain": sample_transport_chain,
        "incoterm": "DDP",
        "cargo_mass_tonnes": Decimal("20.0"),
        "reporting_period": "2024-01",
        "tenant_id": "tenant-abc",
        "facility_id": "facility-123",
        "gwp_version": "AR5",
        "ef_scope": "WTW",
        "description": "January shipment - customer order #12345"
    }


@pytest.fixture
def sample_fuel_input() -> Dict[str, Any]:
    """
    Sample fuel consumption input for fuel-based calculation.

    Represents primary data from carrier:
    - Fuel type: Diesel (B7 blend)
    - Consumption: 500 litres
    - Journey: Hamburg to Munich
    - Allocation: Mass-based (20t of 44t capacity)
    """
    return {
        "consumption_id": "FUEL-001",
        "calculation_method": "FUEL_BASED",
        "fuel_type": "DIESEL",
        "fuel_quantity_litres": Decimal("500.0"),
        "cargo_mass_tonnes": Decimal("20.0"),
        "vehicle_capacity_tonnes": Decimal("44.0"),
        "allocation_method": "MASS",
        "distance_km": Decimal("500.0"),
        "mode": "ROAD",
        "vehicle_type": "ARTICULATED_40_44T",
        "reporting_period": "2024-01",
        "tenant_id": "tenant-abc",
        "facility_id": "facility-123",
        "gwp_version": "AR5",
        "ef_scope": "WTW",
        "description": "Diesel consumption Hamburg to Munich"
    }


@pytest.fixture
def sample_spend_input() -> Dict[str, Any]:
    """
    Sample spend input for EEIO-based calculation.

    Represents spend data when activity data unavailable:
    - Spend: 10,000 USD
    - Sector: NAICS 484110 (General Freight Trucking, Local)
    - EEIO factor: EPA USEEIO or EXIOBASE
    - Data quality: Low (Tier 3)
    """
    return {
        "spend_id": "SPEND-001",
        "calculation_method": "SPEND_BASED",
        "spend_amount": Decimal("10000.0"),
        "currency": "USD",
        "sector_code": "484110",
        "sector_classification": "NAICS",
        "eeio_database": "USEEIO_2.0",
        "reporting_period": "2024-Q1",
        "tenant_id": "tenant-abc",
        "facility_id": "facility-123",
        "gwp_version": "AR5",
        "description": "Q1 trucking spend - local freight"
    }


@pytest.fixture
def sample_supplier_input() -> Dict[str, Any]:
    """
    Sample supplier-specific emission data.

    Represents primary data from carrier:
    - Emissions: 500 kgCO2e (carrier calculated)
    - Methodology: GLEC Framework v3.0
    - Scope: WTW (Well-to-Wheel)
    - Verification: Third-party audited
    - Data quality: High (Tier 1)
    """
    return {
        "supplier_emission_id": "SUPP-001",
        "calculation_method": "SUPPLIER_SPECIFIC",
        "supplier_name": "ABC Logistics GmbH",
        "supplier_id": "SUPP-ABC-001",
        "emissions_tco2e": Decimal("0.500"),  # 500 kgCO2e
        "methodology": "GLEC_FRAMEWORK_V3",
        "scope": "WTW",
        "verification_status": "THIRD_PARTY_VERIFIED",
        "verification_date": "2024-01-15",
        "reporting_period": "2024-01",
        "shipment_id": "SHIPMENT-12345",
        "distance_km": Decimal("500.0"),
        "cargo_mass_tonnes": Decimal("20.0"),
        "tenant_id": "tenant-abc",
        "facility_id": "facility-123",
        "gwp_version": "AR5",
        "data_quality_score": Decimal("0.95"),
        "description": "Carrier-reported emissions - shipment #12345"
    }


# ============================================================================
# CONFIGURATION FIXTURES
# ============================================================================

@pytest.fixture
def sample_allocation_config() -> Dict[str, Any]:
    """
    Sample allocation configuration - Mass-based.

    Allocates emissions based on cargo mass:
    - Method: Mass-based allocation
    - Cargo mass: 20 tonnes
    - Vehicle capacity: 44 tonnes
    - Allocation factor: 20/44 = 0.4545 (45.45%)
    """
    return {
        "allocation_method": "MASS",
        "cargo_mass_tonnes": Decimal("20.0"),
        "vehicle_capacity_tonnes": Decimal("44.0"),
        "cargo_volume_m3": None,
        "vehicle_capacity_m3": None,
        "allocation_factor": Decimal("0.4545"),
        "description": "Mass-based allocation 20t/44t"
    }


@pytest.fixture
def sample_reefer_config() -> Dict[str, Any]:
    """
    Sample reefer (refrigerated transport) configuration.

    Additional emissions for temperature-controlled transport:
    - Temperature: Chilled (0-5°C)
    - Refrigerant: R-134a (HFC-134a, GWP100 AR5 = 1,430)
    - Charge: 5 kg
    - Leak rate: 8% annual (transport average)
    - Uplift factor: 1.15 (15% additional fuel for refrigeration)
    """
    return {
        "temperature_controlled": True,
        "temperature_range": "CHILLED",
        "refrigerant_type": "R-134A",
        "refrigerant_charge_kg": Decimal("5.0"),
        "annual_leak_rate": Decimal("0.08"),  # 8%
        "journey_duration_hours": Decimal("8.0"),
        "reefer_fuel_uplift": Decimal("1.15"),  # 15% uplift
        "description": "Chilled transport with R-134a"
    }


@pytest.fixture
def sample_warehouse_config() -> Dict[str, Any]:
    """
    Sample warehouse emissions configuration.

    Emissions from warehousing activities:
    - Type: Cold storage warehouse
    - Floor area: 500 m² (allocated to shipment)
    - Dwell time: 72 hours (3 days)
    - Energy intensity: 150 kWh/m²/year (cold storage typical)
    - Grid intensity: 0.5 kgCO2e/kWh (EU average)
    """
    return {
        "warehouse_type": "COLD_STORAGE_WAREHOUSE",
        "floor_area_m2": Decimal("500.0"),
        "dwell_time_hours": Decimal("72.0"),
        "energy_intensity_kwh_m2_year": Decimal("150.0"),
        "grid_intensity_kgco2e_kwh": Decimal("0.5"),
        "refrigerant_type": "R-404A",
        "refrigerant_charge_kg": Decimal("20.0"),
        "annual_leak_rate": Decimal("0.10"),  # 10%
        "description": "Cold storage 500m² for 3 days"
    }


@pytest.fixture
def sample_calculation_request(sample_shipment_input) -> Dict[str, Any]:
    """
    Sample calculation request.

    Complete request for emissions calculation:
    - Method: Distance-based (default)
    - Input: Shipment with multi-modal chain
    - Scope: WTW (Well-to-Wheel)
    - GWP: AR5 (IPCC Fifth Assessment Report)
    - Uncertainty: Monte Carlo 1000 iterations
    """
    return {
        "request_id": "REQ-001",
        "calculation_method": "DISTANCE_BASED",
        "input_data": sample_shipment_input,
        "gwp_version": "AR5",
        "ef_scope": "WTW",
        "uncertainty_analysis": True,
        "monte_carlo_iterations": 1000,
        "include_provenance": True,
        "tenant_id": "tenant-abc",
        "user_id": "user-123",
        "requested_at": datetime.utcnow().isoformat()
    }


@pytest.fixture
def sample_batch_request(
    sample_shipment_input,
    sample_fuel_input,
    sample_spend_input
) -> Dict[str, Any]:
    """
    Sample batch calculation request.

    Batch request with multiple shipments:
    - 3 shipments (distance/fuel/spend methods)
    - Parallel processing enabled
    - Aggregation by reporting period
    """
    return {
        "batch_id": "BATCH-001",
        "requests": [
            {
                "request_id": "REQ-001",
                "calculation_method": "DISTANCE_BASED",
                "input_data": sample_shipment_input
            },
            {
                "request_id": "REQ-002",
                "calculation_method": "FUEL_BASED",
                "input_data": sample_fuel_input
            },
            {
                "request_id": "REQ-003",
                "calculation_method": "SPEND_BASED",
                "input_data": sample_spend_input
            }
        ],
        "parallel_processing": True,
        "max_workers": 4,
        "aggregate_results": True,
        "aggregation_key": "reporting_period",
        "tenant_id": "tenant-abc",
        "user_id": "user-123",
        "requested_at": datetime.utcnow().isoformat()
    }


# ============================================================================
# MOCK ENGINE FIXTURES
# ============================================================================

@pytest.fixture
def mock_database_engine():
    """
    Mock UpstreamTransportationDatabaseEngine.

    Provides mock emission factor lookups:
    - Road: 0.8 kgCO2e/tonne-km
    - Maritime: 0.01 kgCO2e/tonne-km
    - Air: 2.5 kgCO2e/tonne-km
    - Rail: 0.05 kgCO2e/tonne-km
    - Fuel: Diesel 2.68 kgCO2e/litre (WTW)
    - EEIO: 0.5 kgCO2e/USD (trucking)
    """
    engine = MagicMock()

    # Mock emission factor lookup
    async def mock_lookup_ef(*args, **kwargs):
        mode = kwargs.get("mode", args[0] if args else None)
        ef_map = {
            "ROAD": Decimal("0.8"),
            "MARITIME": Decimal("0.01"),
            "AIR": Decimal("2.5"),
            "RAIL": Decimal("0.05")
        }
        return {
            "emission_factor_kgco2e_tonne_km": ef_map.get(mode, Decimal("0.5")),
            "gwp_version": "AR5",
            "scope": "WTW",
            "source": "DEFRA_2023",
            "uncertainty": Decimal("0.15")
        }

    engine.lookup_emission_factor = AsyncMock(side_effect=mock_lookup_ef)

    # Mock fuel emission factor lookup
    async def mock_lookup_fuel_ef(*args, **kwargs):
        fuel_type = kwargs.get("fuel_type", args[0] if args else "DIESEL")
        fuel_ef_map = {
            "DIESEL": Decimal("2.68"),  # kgCO2e/litre WTW
            "GASOLINE": Decimal("2.31"),
            "HFO": Decimal("3.11"),
            "JET_FUEL": Decimal("2.52")
        }
        return {
            "emission_factor_kgco2e_litre": fuel_ef_map.get(fuel_type, Decimal("2.5")),
            "gwp_version": "AR5",
            "scope": "WTW",
            "source": "IPCC_2006",
            "uncertainty": Decimal("0.10")
        }

    engine.lookup_fuel_emission_factor = AsyncMock(side_effect=mock_lookup_fuel_ef)

    # Mock EEIO factor lookup
    async def mock_lookup_eeio(*args, **kwargs):
        return {
            "emission_factor_kgco2e_usd": Decimal("0.5"),
            "sector_code": "484110",
            "database": "USEEIO_2.0",
            "uncertainty": Decimal("0.30")
        }

    engine.lookup_eeio_factor = AsyncMock(side_effect=mock_lookup_eeio)

    # Mock hub emission factor lookup
    async def mock_lookup_hub_ef(*args, **kwargs):
        return {
            "emission_factor_kgco2e_m2_hour": Decimal("0.01"),
            "hub_type": "LOGISTICS_HUB",
            "source": "GLEC_2023",
            "uncertainty": Decimal("0.20")
        }

    engine.lookup_hub_emission_factor = AsyncMock(side_effect=mock_lookup_hub_ef)

    return engine


@pytest.fixture
def mock_distance_engine():
    """Mock DistanceCalculatorEngine."""
    engine = MagicMock()

    async def mock_calculate(*args, **kwargs):
        # Return mock calculation result
        return {
            "emissions_tco2e": Decimal("8.0"),  # 500 km * 20 t * 0.8 kgCO2e/t-km / 1000
            "distance_km": Decimal("500.0"),
            "cargo_mass_tonnes": Decimal("20.0"),
            "emission_factor_kgco2e_tonne_km": Decimal("0.8"),
            "calculation_method": "DISTANCE_BASED",
            "provenance_hash": "abc123"
        }

    engine.calculate = AsyncMock(side_effect=mock_calculate)
    return engine


@pytest.fixture
def mock_fuel_engine():
    """Mock FuelBasedCalculatorEngine."""
    engine = MagicMock()

    async def mock_calculate(*args, **kwargs):
        # Return mock calculation result
        # 500 litres * 2.68 kgCO2e/litre * (20/44 allocation) / 1000
        return {
            "emissions_tco2e": Decimal("0.607"),
            "fuel_quantity_litres": Decimal("500.0"),
            "emission_factor_kgco2e_litre": Decimal("2.68"),
            "allocation_factor": Decimal("0.4545"),
            "calculation_method": "FUEL_BASED",
            "provenance_hash": "def456"
        }

    engine.calculate = AsyncMock(side_effect=mock_calculate)
    return engine


@pytest.fixture
def mock_spend_engine():
    """Mock SpendBasedCalculatorEngine."""
    engine = MagicMock()

    async def mock_calculate(*args, **kwargs):
        # Return mock calculation result
        # 10,000 USD * 0.5 kgCO2e/USD / 1000
        return {
            "emissions_tco2e": Decimal("5.0"),
            "spend_amount": Decimal("10000.0"),
            "emission_factor_kgco2e_usd": Decimal("0.5"),
            "sector_code": "484110",
            "calculation_method": "SPEND_BASED",
            "provenance_hash": "ghi789"
        }

    engine.calculate = AsyncMock(side_effect=mock_calculate)
    return engine


@pytest.fixture
def mock_multi_leg_engine():
    """Mock MultiLegOrchestratorEngine."""
    engine = MagicMock()

    async def mock_calculate(*args, **kwargs):
        # Return mock calculation result for multi-leg chain
        return {
            "total_emissions_tco2e": Decimal("15.5"),
            "leg_emissions": [
                {"leg_id": "LEG-ROAD-001", "emissions_tco2e": Decimal("8.0")},
                {"leg_id": "LEG-SEA-001", "emissions_tco2e": Decimal("9.75")},
                {"leg_id": "LEG-RAIL-001", "emissions_tco2e": Decimal("4.0")},
                {"leg_id": "LEG-ROAD-002", "emissions_tco2e": Decimal("1.6")}
            ],
            "hub_emissions": [
                {"hub_id": "HUB-LOG-001", "emissions_tco2e": Decimal("0.04")},
                {"hub_id": "HUB-WARE-001", "emissions_tco2e": Decimal("0.48")}
            ],
            "total_distance_km": Decimal("20900.0"),
            "calculation_method": "DISTANCE_BASED",
            "provenance_hash": "jkl012"
        }

    engine.calculate = AsyncMock(side_effect=mock_calculate)
    return engine


@pytest.fixture
def mock_compliance_engine():
    """Mock ComplianceCheckerEngine."""
    engine = MagicMock()

    async def mock_check(*args, **kwargs):
        # Return mock compliance check result
        return {
            "compliant": True,
            "framework": "GHG_PROTOCOL",
            "category": "SCOPE_3_CATEGORY_4",
            "issues": [],
            "warnings": [],
            "completeness_score": Decimal("0.95"),
            "data_quality_score": Decimal("0.88"),
            "recommendations": [
                "Consider using primary fuel data for higher accuracy"
            ]
        }

    engine.check_compliance = AsyncMock(side_effect=mock_check)
    return engine


@pytest.fixture
def mock_pipeline_engine():
    """Mock UpstreamTransportationPipelineEngine."""
    engine = MagicMock()

    async def mock_process(*args, **kwargs):
        # Return mock pipeline result
        return {
            "result": {
                "emissions_tco2e": Decimal("15.5"),
                "calculation_method": "DISTANCE_BASED",
                "data_quality_score": Decimal("0.88"),
                "uncertainty_range": {
                    "lower": Decimal("13.2"),
                    "upper": Decimal("17.8")
                }
            },
            "provenance_hash": "pipeline-abc123",
            "processing_time_ms": 150.5,
            "validation_status": "PASS"
        }

    engine.process = AsyncMock(side_effect=mock_process)
    return engine


# ============================================================================
# CONFIG FIXTURES
# ============================================================================

@pytest.fixture
def config_fixture(monkeypatch):
    """
    Configuration fixture with environment variable overrides.

    Sets test-specific config values:
    - GL_UTO_ENABLED=true
    - GL_UTO_DATABASE_URL=postgresql://test
    - GL_UTO_DEFAULT_CALCULATION_METHOD=DISTANCE_BASED
    - GL_UTO_DEFAULT_EF_SCOPE=WTW
    - GL_UTO_DECIMAL_PRECISION=6
    - GL_UTO_API_PREFIX=/api/v1/upstream-transportation
    """
    # Set environment variables for testing
    monkeypatch.setenv("GL_UTO_ENABLED", "true")
    monkeypatch.setenv("GL_UTO_DATABASE_URL", "postgresql://test:test@localhost:5432/greenlang_test")
    monkeypatch.setenv("GL_UTO_DEFAULT_CALCULATION_METHOD", "DISTANCE_BASED")
    monkeypatch.setenv("GL_UTO_DEFAULT_EF_SCOPE", "WTW")
    monkeypatch.setenv("GL_UTO_DECIMAL_PRECISION", "6")
    monkeypatch.setenv("GL_UTO_API_PREFIX", "/api/v1/upstream-transportation")
    monkeypatch.setenv("GL_UTO_API_MAX_BATCH_SIZE", "100")

    # Note: Return actual config object when implemented
    # from greenlang.upstream_transportation.config import UpstreamTransportationConfig
    # return UpstreamTransportationConfig()

    return {
        "enabled": True,
        "database_url": "postgresql://test:test@localhost:5432/greenlang_test",
        "default_calculation_method": "DISTANCE_BASED",
        "default_ef_scope": "WTW",
        "decimal_precision": 6,
        "api_prefix": "/api/v1/upstream-transportation",
        "api_max_batch_size": 100
    }
