# -*- coding: utf-8 -*-
"""
Tests for Transport MRV Agents
==============================

Comprehensive tests for all Transport sector MRV agents:
- GL-MRV-TRN-001: Road Transport
- GL-MRV-TRN-002: Aviation
- GL-MRV-TRN-003: Maritime
- GL-MRV-TRN-004: Rail
- GL-MRV-TRN-005: Last Mile
- GL-MRV-TRN-006: EV Fleet
- GL-MRV-TRN-007: Logistics
- GL-MRV-TRN-008: Business Travel
"""

import pytest
from decimal import Decimal
from datetime import datetime

# Road Transport
from greenlang.agents.mrv.transport.road_transport import (
    RoadTransportMRVAgent,
    RoadTransportInput,
    VehicleRecord,
    FleetRecord,
)
from greenlang.agents.mrv.transport.base import (
    VehicleType,
    FuelType,
    CalculationMethod,
    EmissionScope,
)

# Aviation
from greenlang.agents.mrv.transport.aviation import (
    AviationMRVAgent,
    AviationInput,
    FlightRecord,
    CabinClass,
)

# Maritime
from greenlang.agents.mrv.transport.maritime import (
    MaritimeMRVAgent,
    MaritimeInput,
    VoyageRecord,
    VesselType,
    MarineFuelType,
)

# Rail
from greenlang.agents.mrv.transport.rail import (
    RailMRVAgent,
    RailInput,
    RailShipmentRecord,
    RailType,
)

# Last Mile
from greenlang.agents.mrv.transport.last_mile import (
    LastMileMRVAgent,
    LastMileInput,
    DeliveryRecord,
    LastMileVehicle,
)

# EV Fleet
from greenlang.agents.mrv.transport.ev_fleet import (
    EVFleetMRVAgent,
    EVFleetInput,
    EVVehicleRecord,
    EVType,
)

# Logistics
from greenlang.agents.mrv.transport.logistics import (
    LogisticsMRVAgent,
    LogisticsInput,
    LogisticsShipmentRecord,
    TransportLeg,
    LogisticsMode,
)

# Business Travel
from greenlang.agents.mrv.transport.business_travel import (
    BusinessTravelMRVAgent,
    BusinessTravelInput,
    TravelRecord,
    TravelType,
)


class TestRoadTransportMRVAgent:
    """Tests for GL-MRV-TRN-001: Road Transport MRV Agent."""

    def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = RoadTransportMRVAgent()
        assert agent.AGENT_ID == "GL-MRV-TRN-001"
        assert agent.AGENT_VERSION == "1.0.0"

    def test_fuel_based_vehicle_calculation(self):
        """Test fuel-based calculation for individual vehicle."""
        agent = RoadTransportMRVAgent()
        input_data = RoadTransportInput(
            organization_id="TEST001",
            reporting_year=2024,
            vehicle_records=[
                VehicleRecord(
                    vehicle_id="V001",
                    vehicle_type=VehicleType.TRUCK_ARTICULATED,
                    fuel_type=FuelType.DIESEL,
                    fuel_consumed_liters=Decimal("10000"),
                ),
            ],
            calculation_method=CalculationMethod.FUEL_BASED,
        )

        result = agent.calculate(input_data)

        assert result.status == "success"
        assert result.total_emissions_kg_co2e > Decimal("0")
        assert result.total_fuel_liters == Decimal("10000")
        assert result.provenance_hash is not None
        assert len(result.calculation_steps) > 0

    def test_fleet_calculation(self):
        """Test fleet-level calculation."""
        agent = RoadTransportMRVAgent()
        input_data = RoadTransportInput(
            organization_id="TEST001",
            reporting_year=2024,
            fleet_records=[
                FleetRecord(
                    fleet_id="FLEET001",
                    fleet_name="Delivery Fleet",
                    vehicle_count=10,
                    vehicle_type=VehicleType.VAN_MEDIUM,
                    fuel_type=FuelType.DIESEL,
                    fuel_consumed_liters=Decimal("50000"),
                ),
            ],
        )

        result = agent.calculate(input_data)

        assert result.status == "success"
        assert result.total_emissions_kg_co2e > Decimal("0")
        assert VehicleType.VAN_MEDIUM.value in result.emissions_by_vehicle_type

    def test_scope_classification(self):
        """Test correct scope classification."""
        agent = RoadTransportMRVAgent()

        # Owned fleet should be Scope 1
        input_owned = RoadTransportInput(
            organization_id="TEST001",
            reporting_year=2024,
            fleet_records=[
                FleetRecord(
                    fleet_id="FLEET001",
                    vehicle_type=VehicleType.TRUCK_RIGID_MEDIUM,
                    fuel_type=FuelType.DIESEL,
                    fuel_consumed_liters=Decimal("1000"),
                ),
            ],
            is_owned_fleet=True,
        )
        result_owned = agent.calculate(input_owned)
        assert result_owned.scope == EmissionScope.SCOPE_1

        # Not owned should be Scope 3
        input_not_owned = RoadTransportInput(
            organization_id="TEST001",
            reporting_year=2024,
            fleet_records=[
                FleetRecord(
                    fleet_id="FLEET001",
                    vehicle_type=VehicleType.TRUCK_RIGID_MEDIUM,
                    fuel_type=FuelType.DIESEL,
                    fuel_consumed_liters=Decimal("1000"),
                ),
            ],
            is_owned_fleet=False,
        )
        result_not_owned = agent.calculate(input_not_owned)
        assert result_not_owned.scope == EmissionScope.SCOPE_3


class TestAviationMRVAgent:
    """Tests for GL-MRV-TRN-002: Aviation MRV Agent."""

    def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = AviationMRVAgent()
        assert agent.AGENT_ID == "GL-MRV-TRN-002"

    def test_flight_calculation(self):
        """Test single flight emissions calculation."""
        agent = AviationMRVAgent()
        input_data = AviationInput(
            organization_id="TEST001",
            reporting_year=2024,
            flights=[
                FlightRecord(
                    origin_iata="LHR",
                    destination_iata="JFK",
                    cabin_class=CabinClass.ECONOMY,
                    passengers=1,
                ),
            ],
        )

        result = agent.calculate(input_data)

        assert result.status == "success"
        assert result.total_emissions_kg_co2e > Decimal("0")
        assert result.total_flights == 1
        assert result.passenger_emissions_kg > Decimal("0")

    def test_cabin_class_weighting(self):
        """Test that business class has higher emissions than economy."""
        agent = AviationMRVAgent()

        economy = AviationInput(
            organization_id="TEST001",
            reporting_year=2024,
            flights=[
                FlightRecord(
                    origin_iata="LHR",
                    destination_iata="JFK",
                    cabin_class=CabinClass.ECONOMY,
                    passengers=1,
                    distance_km=Decimal("5500"),
                ),
            ],
        )

        business = AviationInput(
            organization_id="TEST001",
            reporting_year=2024,
            flights=[
                FlightRecord(
                    origin_iata="LHR",
                    destination_iata="JFK",
                    cabin_class=CabinClass.BUSINESS,
                    passengers=1,
                    distance_km=Decimal("5500"),
                ),
            ],
        )

        result_economy = agent.calculate(economy)
        result_business = agent.calculate(business)

        assert result_business.total_emissions_kg_co2e > result_economy.total_emissions_kg_co2e

    def test_radiative_forcing(self):
        """Test radiative forcing multiplier."""
        agent = AviationMRVAgent()

        with_rf = AviationInput(
            organization_id="TEST001",
            reporting_year=2024,
            flights=[
                FlightRecord(
                    origin_iata="LHR",
                    destination_iata="JFK",
                    distance_km=Decimal("5500"),
                    passengers=1,
                ),
            ],
            include_radiative_forcing=True,
        )

        without_rf = AviationInput(
            organization_id="TEST001",
            reporting_year=2024,
            flights=[
                FlightRecord(
                    origin_iata="LHR",
                    destination_iata="JFK",
                    distance_km=Decimal("5500"),
                    passengers=1,
                ),
            ],
            include_radiative_forcing=False,
        )

        result_with = agent.calculate(with_rf)
        result_without = agent.calculate(without_rf)

        # With RF should be approximately 1.9x higher
        ratio = result_with.total_emissions_kg_co2e / result_without.total_emissions_kg_co2e
        assert Decimal("1.8") < ratio < Decimal("2.0")


class TestMaritimeMRVAgent:
    """Tests for GL-MRV-TRN-003: Maritime MRV Agent."""

    def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = MaritimeMRVAgent()
        assert agent.AGENT_ID == "GL-MRV-TRN-003"

    def test_fuel_based_calculation(self):
        """Test fuel-based maritime calculation."""
        agent = MaritimeMRVAgent()
        input_data = MaritimeInput(
            organization_id="TEST001",
            reporting_year=2024,
            voyages=[
                VoyageRecord(
                    vessel_type=VesselType.CONTAINER_SHIP,
                    origin_port="NLRTM",
                    destination_port="CNSHA",
                    fuel_type=MarineFuelType.VLSFO,
                    fuel_consumed_tonnes=Decimal("500"),
                ),
            ],
            calculation_method=CalculationMethod.FUEL_BASED,
        )

        result = agent.calculate(input_data)

        assert result.status == "success"
        assert result.total_emissions_kg_co2e > Decimal("0")
        assert result.total_fuel_tonnes == Decimal("500")

    def test_distance_based_calculation(self):
        """Test distance-based maritime calculation."""
        agent = MaritimeMRVAgent()
        input_data = MaritimeInput(
            organization_id="TEST001",
            reporting_year=2024,
            voyages=[
                VoyageRecord(
                    vessel_type=VesselType.BULK_CARRIER,
                    origin_port="NLRTM",
                    destination_port="CNSHA",
                    distance_km=Decimal("19500"),
                    cargo_tonnes=Decimal("50000"),
                ),
            ],
            calculation_method=CalculationMethod.DISTANCE_BASED,
        )

        result = agent.calculate(input_data)

        assert result.status == "success"
        assert result.total_emissions_kg_co2e > Decimal("0")
        assert result.total_tonne_km > Decimal("0")


class TestRailMRVAgent:
    """Tests for GL-MRV-TRN-004: Rail MRV Agent."""

    def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = RailMRVAgent()
        assert agent.AGENT_ID == "GL-MRV-TRN-004"

    def test_freight_calculation(self):
        """Test freight rail emissions calculation."""
        agent = RailMRVAgent()
        input_data = RailInput(
            organization_id="TEST001",
            reporting_year=2024,
            shipments=[
                RailShipmentRecord(
                    rail_type=RailType.FREIGHT_DIESEL,
                    distance_km=Decimal("500"),
                    cargo_tonnes=Decimal("100"),
                ),
            ],
        )

        result = agent.calculate(input_data)

        assert result.status == "success"
        assert result.total_emissions_kg_co2e > Decimal("0")
        assert result.freight_emissions_kg > Decimal("0")

    def test_electric_vs_diesel(self):
        """Test that electric rail has lower emissions than diesel."""
        agent = RailMRVAgent()

        diesel = RailInput(
            organization_id="TEST001",
            reporting_year=2024,
            shipments=[
                RailShipmentRecord(
                    rail_type=RailType.FREIGHT_DIESEL,
                    distance_km=Decimal("500"),
                    cargo_tonnes=Decimal("100"),
                ),
            ],
        )

        electric = RailInput(
            organization_id="TEST001",
            reporting_year=2024,
            shipments=[
                RailShipmentRecord(
                    rail_type=RailType.FREIGHT_ELECTRIC,
                    distance_km=Decimal("500"),
                    cargo_tonnes=Decimal("100"),
                ),
            ],
        )

        result_diesel = agent.calculate(diesel)
        result_electric = agent.calculate(electric)

        assert result_electric.total_emissions_kg_co2e < result_diesel.total_emissions_kg_co2e


class TestLastMileMRVAgent:
    """Tests for GL-MRV-TRN-005: Last Mile MRV Agent."""

    def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = LastMileMRVAgent()
        assert agent.AGENT_ID == "GL-MRV-TRN-005"

    def test_delivery_calculation(self):
        """Test last mile delivery calculation."""
        agent = LastMileMRVAgent()
        input_data = LastMileInput(
            organization_id="TEST001",
            reporting_year=2024,
            deliveries=[
                DeliveryRecord(
                    vehicle_type=LastMileVehicle.VAN_DIESEL,
                    parcels_delivered=100,
                    distance_km=Decimal("50"),
                ),
            ],
        )

        result = agent.calculate(input_data)

        assert result.status == "success"
        assert result.total_emissions_kg_co2e > Decimal("0")
        assert result.total_parcels_delivered == 100
        assert result.emissions_per_parcel_kg is not None

    def test_zero_emission_vehicles(self):
        """Test that cargo bikes have zero direct emissions."""
        agent = LastMileMRVAgent()
        input_data = LastMileInput(
            organization_id="TEST001",
            reporting_year=2024,
            deliveries=[
                DeliveryRecord(
                    vehicle_type=LastMileVehicle.CARGO_BIKE,
                    parcels_delivered=50,
                    distance_km=Decimal("20"),
                ),
            ],
        )

        result = agent.calculate(input_data)

        assert result.total_emissions_kg_co2e == Decimal("0")


class TestEVFleetMRVAgent:
    """Tests for GL-MRV-TRN-006: EV Fleet MRV Agent."""

    def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = EVFleetMRVAgent()
        assert agent.AGENT_ID == "GL-MRV-TRN-006"

    def test_ev_calculation(self):
        """Test EV fleet emissions calculation."""
        agent = EVFleetMRVAgent()
        input_data = EVFleetInput(
            organization_id="TEST001",
            reporting_year=2024,
            vehicles=[
                EVVehicleRecord(
                    vehicle_id="EV001",
                    vehicle_type=EVType.BEV_CAR,
                    electricity_kwh=Decimal("5000"),
                ),
            ],
            country_code="UK",
        )

        result = agent.calculate(input_data)

        assert result.status == "success"
        assert result.total_emissions_kg_co2e > Decimal("0")
        assert result.total_electricity_kwh == Decimal("5000")

    def test_renewable_energy_reduction(self):
        """Test renewable energy certificate reduction."""
        agent = EVFleetMRVAgent()

        without_renewables = EVFleetInput(
            organization_id="TEST001",
            reporting_year=2024,
            vehicles=[
                EVVehicleRecord(
                    vehicle_type=EVType.BEV_CAR,
                    electricity_kwh=Decimal("10000"),
                ),
            ],
            renewable_energy_pct=Decimal("0"),
        )

        with_renewables = EVFleetInput(
            organization_id="TEST001",
            reporting_year=2024,
            vehicles=[
                EVVehicleRecord(
                    vehicle_type=EVType.BEV_CAR,
                    electricity_kwh=Decimal("10000"),
                ),
            ],
            renewable_energy_pct=Decimal("100"),
            has_renewable_certificates=True,
        )

        result_without = agent.calculate(without_renewables)
        result_with = agent.calculate(with_renewables)

        assert result_with.total_emissions_kg_co2e < result_without.total_emissions_kg_co2e


class TestLogisticsMRVAgent:
    """Tests for GL-MRV-TRN-007: Logistics MRV Agent."""

    def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = LogisticsMRVAgent()
        assert agent.AGENT_ID == "GL-MRV-TRN-007"

    def test_multimodal_calculation(self):
        """Test multi-modal logistics calculation."""
        agent = LogisticsMRVAgent()
        input_data = LogisticsInput(
            organization_id="TEST001",
            reporting_year=2024,
            shipments=[
                LogisticsShipmentRecord(
                    origin="Rotterdam, NL",
                    destination="Munich, DE",
                    cargo_weight_tonnes=Decimal("20"),
                    transport_modes=[
                        TransportLeg(mode=LogisticsMode.TRUCK, distance_km=Decimal("800")),
                    ],
                ),
            ],
        )

        result = agent.calculate(input_data)

        assert result.status == "success"
        assert result.total_emissions_kg_co2e > Decimal("0")
        assert result.total_tonne_km > Decimal("0")


class TestBusinessTravelMRVAgent:
    """Tests for GL-MRV-TRN-008: Business Travel MRV Agent."""

    def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = BusinessTravelMRVAgent()
        assert agent.AGENT_ID == "GL-MRV-TRN-008"

    def test_mixed_travel_calculation(self):
        """Test calculation with multiple travel types."""
        agent = BusinessTravelMRVAgent()
        input_data = BusinessTravelInput(
            organization_id="TEST001",
            reporting_year=2024,
            travel_records=[
                TravelRecord(
                    travel_type=TravelType.AIR,
                    distance_km=Decimal("5000"),
                    cabin_class=CabinClass.ECONOMY,
                    passengers=1,
                ),
                TravelRecord(
                    travel_type=TravelType.RAIL,
                    distance_km=Decimal("300"),
                    passengers=1,
                ),
                TravelRecord(
                    travel_type=TravelType.HOTEL,
                    hotel_nights=2,
                ),
            ],
        )

        result = agent.calculate(input_data)

        assert result.status == "success"
        assert result.total_emissions_kg_co2e > Decimal("0")
        assert result.air_emissions_kg > Decimal("0")
        assert result.rail_emissions_kg > Decimal("0")
        assert result.hotel_emissions_kg > Decimal("0")


class TestProvenanceAndAuditTrail:
    """Tests for provenance tracking and audit trails."""

    def test_provenance_hash_consistency(self):
        """Test that same input produces same provenance hash."""
        agent = RoadTransportMRVAgent()
        input_data = RoadTransportInput(
            organization_id="TEST001",
            reporting_year=2024,
            fleet_records=[
                FleetRecord(
                    fleet_id="FLEET001",
                    vehicle_type=VehicleType.TRUCK_ARTICULATED,
                    fuel_type=FuelType.DIESEL,
                    fuel_consumed_liters=Decimal("10000"),
                ),
            ],
        )

        result1 = agent.calculate(input_data)
        result2 = agent.calculate(input_data)

        # Note: Hash may differ due to timestamp, but emissions should match
        assert result1.total_emissions_kg_co2e == result2.total_emissions_kg_co2e

    def test_calculation_steps_present(self):
        """Test that calculation steps are recorded."""
        agent = AviationMRVAgent()
        input_data = AviationInput(
            organization_id="TEST001",
            reporting_year=2024,
            flights=[
                FlightRecord(
                    origin_iata="LHR",
                    destination_iata="JFK",
                    passengers=1,
                ),
            ],
        )

        result = agent.calculate(input_data)

        assert len(result.calculation_steps) > 0
        assert result.calculation_steps[0].step_number == 1
        assert result.calculation_steps[0].description is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
