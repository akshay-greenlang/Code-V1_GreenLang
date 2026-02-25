"""
Test suite for AverageDataCalculatorEngine.

This module tests the average-data calculation method for capital goods emissions,
including material-based calculations, unit conversions, physical emission factors,
embodied carbon, regional adjustments, and comparison across multiple EF sources.

Test Coverage:
- Singleton pattern enforcement
- Full calculation pipeline for single and batch records
- Material-based calculations (weight × EF)
- Unit conversion for various units (kg, t, lb, m3_concrete, etc.)
- Physical emission factor lookup from ICE/ecoinvent/DEFRA
- Area-based emissions for buildings (m2)
- Unit-based emissions for IT equipment (per-unit)
- Transport emissions for delivery/installation
- Gas breakdown (CO2, CH4, N2O)
- Data quality indicator scoring
- Aggregation by material type
- Building, equipment, vehicle, and IT equipment EF lookup
- Embodied carbon calculation
- Regional adjustment factors
- EF source comparison
- Decimal square root precision
- EF hierarchy selection
"""

import pytest
from decimal import Decimal
from datetime import datetime
from typing import Dict, List, Optional
from unittest.mock import Mock, patch, MagicMock

from greenlang.capital_goods.engines.average_data_calculator import (
    AverageDataCalculatorEngine,
    AverageDataRecord,
    AverageDataResult,
    MaterialRecord,
    BuildingRecord,
    EquipmentRecord,
    TransportRecord,
)
from greenlang.capital_goods.models import (
    EmissionFactorSource,
    MaterialType,
    DataQualityDimension,
    DataQualityScore,
)


@pytest.fixture
def calculator_engine():
    """Provide fresh AverageDataCalculatorEngine instance."""
    # Reset singleton
    AverageDataCalculatorEngine._instance = None
    engine = AverageDataCalculatorEngine()
    return engine


@pytest.fixture
def sample_material_record():
    """Provide sample material-based record."""
    return MaterialRecord(
        record_id="MAT-001",
        description="Structural Steel Beam",
        material_type=MaterialType.STRUCTURAL_STEEL,
        quantity=Decimal("5000.00"),
        unit="kg",
        region="GLOBAL",
        year=2023,
    )


@pytest.fixture
def sample_building_record():
    """Provide sample building record."""
    return BuildingRecord(
        record_id="BLD-001",
        description="Office Building",
        building_type="commercial_office",
        floor_area_m2=Decimal("10000.00"),
        region="US",
        year=2023,
    )


@pytest.fixture
def mock_database_engine():
    """Mock CapitalAssetDatabaseEngine."""
    mock_db = MagicMock()
    return mock_db


class TestSingletonPattern:
    """Test singleton pattern enforcement."""

    def test_singleton_returns_same_instance(self):
        """Test that multiple calls return the same instance."""
        # Arrange & Act
        engine1 = AverageDataCalculatorEngine()
        engine2 = AverageDataCalculatorEngine()

        # Assert
        assert engine1 is engine2

    def test_singleton_persists_state(self, calculator_engine):
        """Test that singleton preserves state across references."""
        # Arrange
        calculator_engine._default_region = "EU"

        # Act
        new_ref = AverageDataCalculatorEngine()

        # Assert
        assert new_ref._default_region == "EU"


class TestCalculateMaterialBased:
    """Test calculate() for material-based records."""

    def test_calculate_material_full_pipeline(
        self,
        calculator_engine,
        sample_material_record,
        mock_database_engine
    ):
        """Test complete material calculation pipeline."""
        # Arrange
        mock_database_engine.get_physical_ef.return_value = MagicMock(
            ef_co2e_per_kg=Decimal("2.35"),
            ef_co2_per_kg=Decimal("2.10"),
            ef_ch4_per_kg=Decimal("0.05"),
            ef_n2o_per_kg=Decimal("0.20"),
            source=EmissionFactorSource.ICE,
        )

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            result = calculator_engine.calculate(sample_material_record)

            # Assert
            assert result is not None
            assert result.record_id == "MAT-001"
            assert result.total_emissions_kg_co2e == Decimal("11750.00")  # 5000 * 2.35
            assert result.ef_source == EmissionFactorSource.ICE
            assert result.provenance_hash is not None

    def test_calculate_material_with_unit_conversion(
        self,
        calculator_engine,
        mock_database_engine
    ):
        """Test material calculation with unit conversion."""
        # Arrange
        record = MaterialRecord(
            record_id="MAT-002",
            description="Steel in tonnes",
            material_type=MaterialType.STRUCTURAL_STEEL,
            quantity=Decimal("5.00"),
            unit="t",  # tonnes
            region="GLOBAL",
            year=2023,
        )

        mock_database_engine.get_physical_ef.return_value = MagicMock(
            ef_co2e_per_kg=Decimal("2.35"),
            source=EmissionFactorSource.ICE,
        )

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            result = calculator_engine.calculate(record)

            # Assert
            # 5 tonnes = 5000 kg, emissions = 5000 * 2.35 = 11750
            assert result.total_emissions_kg_co2e == Decimal("11750.00")

    def test_calculate_material_no_ef_found(
        self,
        calculator_engine,
        sample_material_record,
        mock_database_engine
    ):
        """Test calculation when no emission factor found."""
        # Arrange
        mock_database_engine.get_physical_ef.return_value = None

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act & Assert
            with pytest.raises(ValueError, match="No emission factor found"):
                calculator_engine.calculate(sample_material_record)


class TestCalculateBatch:
    """Test calculate_batch() for multiple records."""

    def test_calculate_batch_multiple_materials(self, calculator_engine, mock_database_engine):
        """Test batch calculation for multiple material records."""
        # Arrange
        records = [
            MaterialRecord(
                record_id=f"MAT-{i:03d}",
                description=f"Material {i}",
                material_type=MaterialType.STRUCTURAL_STEEL,
                quantity=Decimal("1000.00"),
                unit="kg",
                region="GLOBAL",
                year=2023,
            )
            for i in range(1, 6)
        ]

        mock_database_engine.get_physical_ef.return_value = MagicMock(
            ef_co2e_per_kg=Decimal("2.35"),
            source=EmissionFactorSource.ICE,
        )

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            results = calculator_engine.calculate_batch(records)

            # Assert
            assert len(results) == 5
            assert all(r.total_emissions_kg_co2e == Decimal("2350.00") for r in results)

    def test_calculate_batch_preserves_order(self, calculator_engine, mock_database_engine):
        """Test batch calculation preserves input order."""
        # Arrange
        records = [
            MaterialRecord(
                record_id=f"MAT-{i:03d}",
                description=f"Material {i}",
                material_type=MaterialType.STRUCTURAL_STEEL,
                quantity=Decimal(f"{i * 1000}.00"),
                unit="kg",
                region="GLOBAL",
                year=2023,
            )
            for i in range(1, 4)
        ]

        mock_database_engine.get_physical_ef.return_value = MagicMock(
            ef_co2e_per_kg=Decimal("2.35"),
            source=EmissionFactorSource.ICE,
        )

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            results = calculator_engine.calculate_batch(records)

            # Assert
            assert results[0].record_id == "MAT-001"
            assert results[1].record_id == "MAT-002"
            assert results[2].record_id == "MAT-003"


class TestUnitConversion:
    """Test convert_to_kg() for various units."""

    @pytest.mark.parametrize("quantity,unit,expected_kg", [
        (Decimal("1000"), "kg", Decimal("1000")),
        (Decimal("1"), "t", Decimal("1000")),
        (Decimal("2204.62"), "lb", Decimal("1000")),
        (Decimal("1"), "mt", Decimal("1000000")),  # metric ton
        (Decimal("1.102"), "ton", Decimal("1000")),  # US ton
    ])
    def test_convert_to_kg_mass_units(
        self,
        calculator_engine,
        quantity,
        unit,
        expected_kg
    ):
        """Test conversion for mass units."""
        # Act
        result = calculator_engine.convert_to_kg(quantity, unit)

        # Assert
        assert abs(result - expected_kg) < Decimal("0.01")

    def test_convert_to_kg_concrete_volume(self, calculator_engine):
        """Test conversion for concrete volume (m3)."""
        # Arrange - 1 m3 concrete ≈ 2400 kg (density)
        quantity = Decimal("10.00")
        unit = "m3_concrete"

        # Act
        result = calculator_engine.convert_to_kg(quantity, unit, material_type="concrete_25mpa")

        # Assert
        expected = Decimal("24000")  # 10 m3 * 2400 kg/m3
        assert abs(result - expected) < Decimal("100")

    def test_convert_to_kg_steel_volume(self, calculator_engine):
        """Test conversion for steel volume (m3)."""
        # Arrange - 1 m3 steel ≈ 7850 kg (density)
        quantity = Decimal("2.00")
        unit = "m3_steel"

        # Act
        result = calculator_engine.convert_to_kg(quantity, unit, material_type="structural_steel")

        # Assert
        expected = Decimal("15700")  # 2 m3 * 7850 kg/m3
        assert abs(result - expected) < Decimal("100")

    def test_convert_to_kg_invalid_unit(self, calculator_engine):
        """Test conversion with invalid unit raises error."""
        # Act & Assert
        with pytest.raises(ValueError, match="Unknown unit"):
            calculator_engine.convert_to_kg(Decimal("100"), "invalid_unit")


class TestPhysicalEmissionFactorLookup:
    """Test lookup_physical_ef() for materials."""

    def test_lookup_physical_ef_ice_source(self, calculator_engine, mock_database_engine):
        """Test lookup from ICE database."""
        # Arrange
        mock_database_engine.get_physical_ef.return_value = MagicMock(
            ef_co2e_per_kg=Decimal("2.35"),
            source=EmissionFactorSource.ICE,
        )

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            result = calculator_engine.lookup_physical_ef(
                MaterialType.STRUCTURAL_STEEL,
                "GLOBAL",
                2023
            )

            # Assert
            assert result is not None
            assert result.ef_co2e_per_kg == Decimal("2.35")
            assert result.source == EmissionFactorSource.ICE

    def test_lookup_physical_ef_ecoinvent_source(self, calculator_engine, mock_database_engine):
        """Test lookup from ecoinvent database."""
        # Arrange
        mock_database_engine.get_physical_ef.return_value = MagicMock(
            ef_co2e_per_kg=Decimal("11.5"),
            source=EmissionFactorSource.ECOINVENT,
        )

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            result = calculator_engine.lookup_physical_ef(
                MaterialType.ALUMINUM_PRIMARY,
                "GLOBAL",
                2023
            )

            # Assert
            assert result.source == EmissionFactorSource.ECOINVENT

    def test_lookup_physical_ef_defra_source(self, calculator_engine, mock_database_engine):
        """Test lookup from DEFRA database."""
        # Arrange
        mock_database_engine.get_physical_ef.return_value = MagicMock(
            ef_co2e_per_kg=Decimal("0.145"),
            source=EmissionFactorSource.DEFRA,
        )

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            result = calculator_engine.lookup_physical_ef(
                MaterialType.CONCRETE_25MPA,
                "UK",
                2023
            )

            # Assert
            assert result.source == EmissionFactorSource.DEFRA


class TestMaterialEmissionsCalculation:
    """Test calculate_material_emissions() formula."""

    def test_calculate_material_emissions_basic_formula(self, calculator_engine):
        """Test basic material emissions: weight × EF."""
        # Arrange
        weight_kg = Decimal("5000.00")
        ef = Decimal("2.35")

        # Act
        result = calculator_engine.calculate_material_emissions(weight_kg, ef)

        # Assert
        assert result == Decimal("11750.00")

    def test_calculate_material_emissions_zero_weight(self, calculator_engine):
        """Test emissions calculation with zero weight."""
        # Act
        result = calculator_engine.calculate_material_emissions(Decimal("0"), Decimal("2.35"))

        # Assert
        assert result == Decimal("0")

    def test_calculate_material_emissions_precision(self, calculator_engine):
        """Test emissions calculation maintains precision."""
        # Arrange
        weight_kg = Decimal("1234.56")
        ef = Decimal("2.3456")

        # Act
        result = calculator_engine.calculate_material_emissions(weight_kg, ef)

        # Assert
        expected = Decimal("2895.698")  # Rounded to 3 decimal places
        assert abs(result - expected) < Decimal("0.001")


class TestAreaBasedEmissions:
    """Test calculate_area_emissions() for buildings."""

    def test_calculate_area_emissions_office_building(
        self,
        calculator_engine,
        mock_database_engine
    ):
        """Test area-based emissions for office building."""
        # Arrange
        area_m2 = Decimal("10000.00")
        building_type = "commercial_office"

        mock_database_engine.get_building_ef.return_value = Decimal("350")  # kg CO2e/m2

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            result = calculator_engine.calculate_area_emissions(area_m2, building_type, "US")

            # Assert
            assert result == Decimal("3500000")  # 10000 * 350

    def test_calculate_area_emissions_residential(
        self,
        calculator_engine,
        mock_database_engine
    ):
        """Test area-based emissions for residential building."""
        # Arrange
        area_m2 = Decimal("5000.00")
        building_type = "residential_multi_family"

        mock_database_engine.get_building_ef.return_value = Decimal("280")  # kg CO2e/m2

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            result = calculator_engine.calculate_area_emissions(area_m2, building_type, "US")

            # Assert
            assert result == Decimal("1400000")


class TestUnitBasedEmissions:
    """Test calculate_unit_emissions() for IT equipment."""

    def test_calculate_unit_emissions_servers(
        self,
        calculator_engine,
        mock_database_engine
    ):
        """Test unit-based emissions for servers."""
        # Arrange
        quantity = Decimal("50")
        equipment_type = "server_rack"

        mock_database_engine.get_equipment_ef.return_value = Decimal("1200")  # kg CO2e/unit

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            result = calculator_engine.calculate_unit_emissions(quantity, equipment_type, "US")

            # Assert
            assert result == Decimal("60000")  # 50 * 1200

    def test_calculate_unit_emissions_laptops(
        self,
        calculator_engine,
        mock_database_engine
    ):
        """Test unit-based emissions for laptops."""
        # Arrange
        quantity = Decimal("100")
        equipment_type = "laptop"

        mock_database_engine.get_equipment_ef.return_value = Decimal("250")  # kg CO2e/unit

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            result = calculator_engine.calculate_unit_emissions(quantity, equipment_type, "US")

            # Assert
            assert result == Decimal("25000")


class TestTransportEmissions:
    """Test calculate_transport_emissions() for delivery/installation."""

    def test_calculate_transport_emissions_truck(self, calculator_engine, mock_database_engine):
        """Test transport emissions for truck delivery."""
        # Arrange
        distance_km = Decimal("500")
        mode = "truck"
        weight_t = Decimal("10")

        mock_database_engine.get_transport_ef.return_value = Decimal("0.062")  # kg CO2e/t-km

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            result = calculator_engine.calculate_transport_emissions(
                distance_km,
                mode,
                weight_t
            )

            # Assert
            assert result == Decimal("310")  # 500 * 10 * 0.062

    def test_calculate_transport_emissions_ship(self, calculator_engine, mock_database_engine):
        """Test transport emissions for ship delivery."""
        # Arrange
        distance_km = Decimal("10000")
        mode = "ship"
        weight_t = Decimal("100")

        mock_database_engine.get_transport_ef.return_value = Decimal("0.008")  # kg CO2e/t-km

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            result = calculator_engine.calculate_transport_emissions(
                distance_km,
                mode,
                weight_t
            )

            # Assert
            assert result == Decimal("8000")

    def test_calculate_transport_emissions_air(self, calculator_engine, mock_database_engine):
        """Test transport emissions for air freight."""
        # Arrange
        distance_km = Decimal("5000")
        mode = "air"
        weight_t = Decimal("5")

        mock_database_engine.get_transport_ef.return_value = Decimal("0.602")  # kg CO2e/t-km

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            result = calculator_engine.calculate_transport_emissions(
                distance_km,
                mode,
                weight_t
            )

            # Assert
            assert result == Decimal("15050")


class TestGasBreakdown:
    """Test split_gas_breakdown() method."""

    def test_split_gas_breakdown_sums_to_total(self, calculator_engine):
        """Test gas breakdown components sum to total."""
        # Arrange
        total_co2e = Decimal("11750.00")
        ef = MagicMock(
            ef_co2e_per_kg=Decimal("2.35"),
            ef_co2_per_kg=Decimal("2.10"),
            ef_ch4_per_kg=Decimal("0.05"),
            ef_n2o_per_kg=Decimal("0.20"),
        )
        weight_kg = Decimal("5000.00")

        # Act
        result = calculator_engine.split_gas_breakdown(total_co2e, ef, weight_kg)

        # Assert
        total = result.co2_kg + result.ch4_kg_co2e + result.n2o_kg_co2e
        assert total == total_co2e

    def test_split_gas_breakdown_individual_gases(self, calculator_engine):
        """Test individual gas calculations."""
        # Arrange
        total_co2e = Decimal("11750.00")
        ef = MagicMock(
            ef_co2e_per_kg=Decimal("2.35"),
            ef_co2_per_kg=Decimal("2.10"),
            ef_ch4_per_kg=Decimal("0.05"),
            ef_n2o_per_kg=Decimal("0.20"),
        )
        weight_kg = Decimal("5000.00")

        # Act
        result = calculator_engine.split_gas_breakdown(total_co2e, ef, weight_kg)

        # Assert
        assert result.co2_kg == Decimal("10500.00")  # 5000 * 2.10
        assert result.ch4_kg_co2e == Decimal("250.00")  # 5000 * 0.05
        assert result.n2o_kg_co2e == Decimal("1000.00")  # 5000 * 0.20


class TestDataQualityScoring:
    """Test score_dqi() method."""

    def test_score_dqi_all_dimensions(self, calculator_engine):
        """Test DQI scoring returns valid scores for all dimensions."""
        # Arrange
        record = MaterialRecord(
            record_id="MAT-001",
            description="Steel",
            material_type=MaterialType.STRUCTURAL_STEEL,
            quantity=Decimal("5000.00"),
            unit="kg",
            region="GLOBAL",
            year=2023,
        )
        ef_source = EmissionFactorSource.ICE

        # Act
        result = calculator_engine.score_dqi(record, ef_source)

        # Assert
        assert result is not None
        assert 1 <= result.technological_representativeness <= 5
        assert 1 <= result.temporal_representativeness <= 5
        assert 1 <= result.geographical_representativeness <= 5
        assert 1 <= result.completeness <= 5
        assert 1 <= result.reliability <= 5

    def test_score_dqi_ice_source_high_reliability(self, calculator_engine):
        """Test ICE source scores high on reliability."""
        # Arrange
        record = MaterialRecord(
            record_id="MAT-001",
            description="Steel",
            material_type=MaterialType.STRUCTURAL_STEEL,
            quantity=Decimal("5000.00"),
            unit="kg",
            region="GLOBAL",
            year=2023,
        )

        # Act
        result = calculator_engine.score_dqi(record, EmissionFactorSource.ICE)

        # Assert
        assert result.reliability >= 4

    def test_score_dqi_ecoinvent_source_high_reliability(self, calculator_engine):
        """Test ecoinvent source scores high on reliability."""
        # Arrange
        record = MaterialRecord(
            record_id="MAT-001",
            description="Aluminum",
            material_type=MaterialType.ALUMINUM_PRIMARY,
            quantity=Decimal("1000.00"),
            unit="kg",
            region="GLOBAL",
            year=2023,
        )

        # Act
        result = calculator_engine.score_dqi(record, EmissionFactorSource.ECOINVENT)

        # Assert
        assert result.reliability >= 4


class TestAggregations:
    """Test aggregation methods."""

    def test_aggregate_by_material(self, calculator_engine):
        """Test aggregation by material type."""
        # Arrange
        results = [
            AverageDataResult(
                record_id="MAT-001",
                material_type=MaterialType.STRUCTURAL_STEEL,
                total_emissions_kg_co2e=Decimal("10000.00"),
            ),
            AverageDataResult(
                record_id="MAT-002",
                material_type=MaterialType.STRUCTURAL_STEEL,
                total_emissions_kg_co2e=Decimal("15000.00"),
            ),
            AverageDataResult(
                record_id="MAT-003",
                material_type=MaterialType.CONCRETE_25MPA,
                total_emissions_kg_co2e=Decimal("8000.00"),
            ),
        ]

        # Act
        aggregated = calculator_engine.aggregate_by_material(results)

        # Assert
        assert len(aggregated) == 2
        assert aggregated[MaterialType.STRUCTURAL_STEEL]["total_emissions"] == Decimal("25000.00")
        assert aggregated[MaterialType.CONCRETE_25MPA]["total_emissions"] == Decimal("8000.00")

    def test_aggregate_by_material_includes_count(self, calculator_engine):
        """Test aggregation includes record count."""
        # Arrange
        results = [
            AverageDataResult(
                record_id=f"MAT-{i:03d}",
                material_type=MaterialType.STRUCTURAL_STEEL,
                total_emissions_kg_co2e=Decimal("1000.00"),
            )
            for i in range(1, 6)
        ]

        # Act
        aggregated = calculator_engine.aggregate_by_material(results)

        # Assert
        assert aggregated[MaterialType.STRUCTURAL_STEEL]["count"] == 5


class TestBuildingEmissionFactor:
    """Test get_building_ef() method."""

    def test_get_building_ef_office(self, calculator_engine, mock_database_engine):
        """Test building EF for office building."""
        # Arrange
        mock_database_engine.get_building_ef.return_value = Decimal("350")

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            result = calculator_engine.get_building_ef("commercial_office", "US")

            # Assert
            assert result == Decimal("350")

    def test_get_building_ef_warehouse(self, calculator_engine, mock_database_engine):
        """Test building EF for warehouse."""
        # Arrange
        mock_database_engine.get_building_ef.return_value = Decimal("180")

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            result = calculator_engine.get_building_ef("warehouse", "US")

            # Assert
            assert result == Decimal("180")


class TestEquipmentEmissionFactor:
    """Test get_equipment_ef() method."""

    def test_get_equipment_ef_cnc_machine(self, calculator_engine, mock_database_engine):
        """Test equipment EF for CNC machine."""
        # Arrange
        mock_database_engine.get_equipment_ef.return_value = Decimal("15000")

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            result = calculator_engine.get_equipment_ef("cnc_machine", "US")

            # Assert
            assert result == Decimal("15000")

    def test_get_equipment_ef_forklift(self, calculator_engine, mock_database_engine):
        """Test equipment EF for forklift."""
        # Arrange
        mock_database_engine.get_equipment_ef.return_value = Decimal("8500")

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            result = calculator_engine.get_equipment_ef("forklift", "US")

            # Assert
            assert result == Decimal("8500")


class TestVehicleEmissionFactor:
    """Test get_vehicle_ef() method."""

    def test_get_vehicle_ef_delivery_truck(self, calculator_engine, mock_database_engine):
        """Test vehicle EF for delivery truck."""
        # Arrange
        mock_database_engine.get_vehicle_ef.return_value = Decimal("25000")

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            result = calculator_engine.get_vehicle_ef("delivery_truck", "US")

            # Assert
            assert result == Decimal("25000")

    def test_get_vehicle_ef_passenger_car(self, calculator_engine, mock_database_engine):
        """Test vehicle EF for passenger car."""
        # Arrange
        mock_database_engine.get_vehicle_ef.return_value = Decimal("7500")

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            result = calculator_engine.get_vehicle_ef("passenger_car", "US")

            # Assert
            assert result == Decimal("7500")


class TestITEquipmentEmissionFactor:
    """Test get_it_equipment_ef() method."""

    def test_get_it_equipment_ef_server(self, calculator_engine, mock_database_engine):
        """Test IT equipment EF for server."""
        # Arrange
        mock_database_engine.get_it_equipment_ef.return_value = Decimal("1200")

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            result = calculator_engine.get_it_equipment_ef("server_rack", "US")

            # Assert
            assert result == Decimal("1200")

    def test_get_it_equipment_ef_laptop(self, calculator_engine, mock_database_engine):
        """Test IT equipment EF for laptop."""
        # Arrange
        mock_database_engine.get_it_equipment_ef.return_value = Decimal("250")

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            result = calculator_engine.get_it_equipment_ef("laptop", "US")

            # Assert
            assert result == Decimal("250")

    def test_get_it_equipment_ef_monitor(self, calculator_engine, mock_database_engine):
        """Test IT equipment EF for monitor."""
        # Arrange
        mock_database_engine.get_it_equipment_ef.return_value = Decimal("180")

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            result = calculator_engine.get_it_equipment_ef("monitor_24inch", "US")

            # Assert
            assert result == Decimal("180")


class TestEmbodiedCarbon:
    """Test calculate_embodied_carbon() method."""

    def test_calculate_embodied_carbon_building(self, calculator_engine, mock_database_engine):
        """Test embodied carbon calculation for building."""
        # Arrange
        materials = {
            MaterialType.STRUCTURAL_STEEL: Decimal("50000"),  # kg
            MaterialType.CONCRETE_25MPA: Decimal("200000"),  # kg
            MaterialType.GLASS: Decimal("10000"),  # kg
        }

        def get_ef_side_effect(material_type, region, year):
            ef_map = {
                MaterialType.STRUCTURAL_STEEL: MagicMock(ef_co2e_per_kg=Decimal("2.35")),
                MaterialType.CONCRETE_25MPA: MagicMock(ef_co2e_per_kg=Decimal("0.145")),
                MaterialType.GLASS: MagicMock(ef_co2e_per_kg=Decimal("0.85")),
            }
            return ef_map.get(material_type)

        mock_database_engine.get_physical_ef.side_effect = get_ef_side_effect

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            result = calculator_engine.calculate_embodied_carbon(materials, "GLOBAL", 2023)

            # Assert
            # Steel: 50000 * 2.35 = 117500
            # Concrete: 200000 * 0.145 = 29000
            # Glass: 10000 * 0.85 = 8500
            # Total: 155000
            expected = Decimal("155000")
            assert abs(result - expected) < Decimal("100")

    def test_calculate_embodied_carbon_single_material(
        self,
        calculator_engine,
        mock_database_engine
    ):
        """Test embodied carbon for single material."""
        # Arrange
        materials = {
            MaterialType.STRUCTURAL_STEEL: Decimal("10000"),
        }

        mock_database_engine.get_physical_ef.return_value = MagicMock(
            ef_co2e_per_kg=Decimal("2.35")
        )

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            result = calculator_engine.calculate_embodied_carbon(materials, "GLOBAL", 2023)

            # Assert
            assert result == Decimal("23500")


class TestRegionalAdjustment:
    """Test apply_regional_adjustment() method."""

    def test_apply_regional_adjustment_china_high_intensity(
        self,
        calculator_engine,
        mock_database_engine
    ):
        """Test regional adjustment for China (higher carbon intensity)."""
        # Arrange
        base_emissions = Decimal("10000.00")
        mock_database_engine.get_regional_adjustment.return_value = Decimal("1.25")  # 25% higher

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            result = calculator_engine.apply_regional_adjustment(
                base_emissions,
                from_region="GLOBAL",
                to_region="CN"
            )

            # Assert
            assert result == Decimal("12500.00")

    def test_apply_regional_adjustment_nordics_low_intensity(
        self,
        calculator_engine,
        mock_database_engine
    ):
        """Test regional adjustment for Nordic countries (lower carbon intensity)."""
        # Arrange
        base_emissions = Decimal("10000.00")
        mock_database_engine.get_regional_adjustment.return_value = Decimal("0.75")  # 25% lower

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            result = calculator_engine.apply_regional_adjustment(
                base_emissions,
                from_region="GLOBAL",
                to_region="SE"
            )

            # Assert
            assert result == Decimal("7500.00")

    def test_apply_regional_adjustment_no_adjustment(
        self,
        calculator_engine,
        mock_database_engine
    ):
        """Test no adjustment when regions match."""
        # Arrange
        base_emissions = Decimal("10000.00")
        mock_database_engine.get_regional_adjustment.return_value = Decimal("1.00")

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            result = calculator_engine.apply_regional_adjustment(
                base_emissions,
                from_region="US",
                to_region="US"
            )

            # Assert
            assert result == Decimal("10000.00")


class TestEFSourceComparison:
    """Test compare_ef_sources() method."""

    def test_compare_ef_sources_multiple_databases(
        self,
        calculator_engine,
        mock_database_engine
    ):
        """Test comparison across multiple EF sources."""
        # Arrange
        material_type = MaterialType.STRUCTURAL_STEEL

        def get_ef_side_effect(mat_type, region, year, source=None):
            source_map = {
                None: MagicMock(ef_co2e_per_kg=Decimal("2.35"), source=EmissionFactorSource.ICE),
                EmissionFactorSource.ECOINVENT: MagicMock(
                    ef_co2e_per_kg=Decimal("2.45"),
                    source=EmissionFactorSource.ECOINVENT
                ),
                EmissionFactorSource.DEFRA: MagicMock(
                    ef_co2e_per_kg=Decimal("2.28"),
                    source=EmissionFactorSource.DEFRA
                ),
            }
            return source_map.get(source)

        mock_database_engine.get_physical_ef.side_effect = get_ef_side_effect

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            result = calculator_engine.compare_ef_sources(material_type, "GLOBAL", 2023)

            # Assert
            assert len(result) >= 3
            assert EmissionFactorSource.ICE in [r["source"] for r in result]
            assert EmissionFactorSource.ECOINVENT in [r["source"] for r in result]
            assert EmissionFactorSource.DEFRA in [r["source"] for r in result]

    def test_compare_ef_sources_includes_variance(self, calculator_engine, mock_database_engine):
        """Test comparison includes variance metrics."""
        # Arrange
        material_type = MaterialType.STRUCTURAL_STEEL

        mock_database_engine.get_physical_ef.side_effect = [
            MagicMock(ef_co2e_per_kg=Decimal("2.35"), source=EmissionFactorSource.ICE),
            MagicMock(ef_co2e_per_kg=Decimal("2.45"), source=EmissionFactorSource.ECOINVENT),
            MagicMock(ef_co2e_per_kg=Decimal("2.28"), source=EmissionFactorSource.DEFRA),
        ]

        with patch.object(calculator_engine, "_database", mock_database_engine):
            # Act
            result = calculator_engine.compare_ef_sources(material_type, "GLOBAL", 2023)

            # Assert
            assert "mean" in result
            assert "std_dev" in result
            assert "min" in result
            assert "max" in result


class TestDecimalSquareRoot:
    """Test _decimal_sqrt() Newton's method precision."""

    def test_decimal_sqrt_perfect_square(self, calculator_engine):
        """Test square root of perfect square."""
        # Act
        result = calculator_engine._decimal_sqrt(Decimal("16"))

        # Assert
        assert result == Decimal("4")

    def test_decimal_sqrt_non_perfect_square(self, calculator_engine):
        """Test square root of non-perfect square."""
        # Act
        result = calculator_engine._decimal_sqrt(Decimal("2"))

        # Assert
        # Should be approximately 1.414213562
        expected = Decimal("1.414213562")
        assert abs(result - expected) < Decimal("0.000001")

    def test_decimal_sqrt_large_number(self, calculator_engine):
        """Test square root of large number."""
        # Act
        result = calculator_engine._decimal_sqrt(Decimal("10000"))

        # Assert
        assert result == Decimal("100")

    def test_decimal_sqrt_small_decimal(self, calculator_engine):
        """Test square root of small decimal."""
        # Act
        result = calculator_engine._decimal_sqrt(Decimal("0.25"))

        # Assert
        assert result == Decimal("0.5")

    def test_decimal_sqrt_precision(self, calculator_engine):
        """Test square root maintains high precision."""
        # Act
        result = calculator_engine._decimal_sqrt(Decimal("3"))

        # Assert
        # Verify precision by squaring result
        squared = result * result
        assert abs(squared - Decimal("3")) < Decimal("0.000001")


class TestEmissionFactorHierarchy:
    """Test select_best_ef() hierarchy."""

    def test_select_best_ef_supplier_priority(self, calculator_engine):
        """Test supplier-specific EF has highest priority."""
        # Arrange
        supplier_ef = Decimal("1.5")
        physical_ef = Decimal("2.0")
        generic_ef = Decimal("2.5")

        # Act
        result = calculator_engine.select_best_ef(
            supplier_ef=supplier_ef,
            physical_ef=physical_ef,
            generic_ef=generic_ef
        )

        # Assert
        assert result == supplier_ef

    def test_select_best_ef_physical_second_priority(self, calculator_engine):
        """Test physical EF has second priority."""
        # Arrange
        supplier_ef = None
        physical_ef = Decimal("2.0")
        generic_ef = Decimal("2.5")

        # Act
        result = calculator_engine.select_best_ef(
            supplier_ef=supplier_ef,
            physical_ef=physical_ef,
            generic_ef=generic_ef
        )

        # Assert
        assert result == physical_ef

    def test_select_best_ef_generic_fallback(self, calculator_engine):
        """Test generic EF is fallback."""
        # Arrange
        supplier_ef = None
        physical_ef = None
        generic_ef = Decimal("2.5")

        # Act
        result = calculator_engine.select_best_ef(
            supplier_ef=supplier_ef,
            physical_ef=physical_ef,
            generic_ef=generic_ef
        )

        # Assert
        assert result == generic_ef

    def test_select_best_ef_none_available(self, calculator_engine):
        """Test returns None when no EF available."""
        # Act
        result = calculator_engine.select_best_ef(
            supplier_ef=None,
            physical_ef=None,
            generic_ef=None
        )

        # Assert
        assert result is None
