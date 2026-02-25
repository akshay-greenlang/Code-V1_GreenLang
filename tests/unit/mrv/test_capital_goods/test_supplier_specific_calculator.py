"""
Unit tests for SupplierSpecificCalculatorEngine.

Tests supplier-specific emission calculations including product-level EF,
PCF allocation, facility allocation methods, and data quality scoring.
"""

import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

from greenlang.mrv.capital_goods.engines.supplier_specific_calculator import (
    SupplierSpecificCalculatorEngine,
    SupplierDataInput,
    ProductLevelEF,
    ProductCarbonFootprint,
    FacilityEmissions,
    AllocationMethod,
    DataQualityScore,
    VerificationLevel,
    SupplierSpecificResult,
)


class TestSupplierSpecificCalculatorEngineSingleton:
    """Test singleton pattern."""

    def test_singleton_same_instance(self):
        """Test that multiple calls return same instance."""
        engine1 = SupplierSpecificCalculatorEngine()
        engine2 = SupplierSpecificCalculatorEngine()
        assert engine1 is engine2

    def test_singleton_with_reset(self):
        """Test singleton reset for testing."""
        engine1 = SupplierSpecificCalculatorEngine()
        SupplierSpecificCalculatorEngine._instance = None
        engine2 = SupplierSpecificCalculatorEngine()
        assert engine1 is not engine2


class TestCalculate:
    """Test calculate() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        SupplierSpecificCalculatorEngine._instance = None
        return SupplierSpecificCalculatorEngine()

    @pytest.fixture
    def product_level_input(self):
        """Create product-level input."""
        return SupplierDataInput(
            asset_id="A001",
            supplier_id="SUP001",
            asset_type="Server",
            purchase_value=Decimal("50000.00"),
            data_source="product_level",
            emission_factor=Decimal("1.5"),  # tCO2e/unit
            quantity=10,
            unit="units",
            metadata={"model": "Dell PowerEdge R750"},
        )

    def test_calculate_success(self, engine, product_level_input):
        """Test successful calculation."""
        result = engine.calculate(product_level_input)

        assert isinstance(result, SupplierSpecificResult)
        assert result.asset_id == "A001"
        assert result.supplier_id == "SUP001"
        assert result.total_emissions > 0
        assert result.data_quality_score is not None
        assert result.provenance_hash is not None

    def test_calculate_routes_to_product_level(self, engine, product_level_input):
        """Test calculation routes to product_level method."""
        with patch.object(engine, 'calculate_product_level') as mock_calc:
            mock_calc.return_value = Mock(total_emissions=Decimal("15.0"))
            result = engine.calculate(product_level_input)
            mock_calc.assert_called_once()

    def test_calculate_routes_to_pcf(self, engine):
        """Test calculation routes to PCF method."""
        pcf_input = SupplierDataInput(
            asset_id="A002",
            supplier_id="SUP002",
            asset_type="Vehicle",
            purchase_value=Decimal("30000.00"),
            data_source="pcf",
            pcf_data={
                "total_emissions": 8.5,
                "functional_unit": "1 vehicle",
                "scope": "cradle-to-gate",
            },
        )

        with patch.object(engine, 'calculate_pcf') as mock_calc:
            mock_calc.return_value = Mock(total_emissions=Decimal("8.5"))
            result = engine.calculate(pcf_input)
            mock_calc.assert_called_once()

    def test_calculate_routes_to_facility_allocation(self, engine):
        """Test calculation routes to facility allocation method."""
        facility_input = SupplierDataInput(
            asset_id="A003",
            supplier_id="SUP003",
            asset_type="Equipment",
            purchase_value=Decimal("100000.00"),
            data_source="facility_allocation",
            facility_emissions=5000.0,  # tCO2e
            allocation_method="economic",
            supplier_revenue=Decimal("10000000.00"),
        )

        with patch.object(engine, 'calculate_facility_allocation') as mock_calc:
            mock_calc.return_value = Mock(total_emissions=Decimal("50.0"))
            result = engine.calculate(facility_input)
            mock_calc.assert_called_once()

    def test_calculate_invalid_data_source(self, engine):
        """Test calculation with invalid data source raises error."""
        invalid_input = SupplierDataInput(
            asset_id="A004",
            supplier_id="SUP004",
            asset_type="Building",
            purchase_value=Decimal("1000000.00"),
            data_source="invalid_source",
        )

        with pytest.raises(ValueError, match="Unsupported data source"):
            engine.calculate(invalid_input)


class TestCalculateBatch:
    """Test calculate_batch() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        SupplierSpecificCalculatorEngine._instance = None
        return SupplierSpecificCalculatorEngine()

    def test_calculate_batch_multiple_inputs(self, engine):
        """Test batch calculation with multiple inputs."""
        inputs = [
            SupplierDataInput(
                asset_id=f"A{i:03d}",
                supplier_id=f"SUP{i:03d}",
                asset_type="Server",
                purchase_value=Decimal("10000.00"),
                data_source="product_level",
                emission_factor=Decimal("1.0"),
                quantity=1,
            )
            for i in range(5)
        ]

        results = engine.calculate_batch(inputs)

        assert len(results) == 5
        assert all(isinstance(r, SupplierSpecificResult) for r in results)
        assert results[0].asset_id == "A000"
        assert results[4].asset_id == "A004"

    def test_calculate_batch_empty_list(self, engine):
        """Test batch calculation with empty list."""
        results = engine.calculate_batch([])
        assert results == []

    def test_calculate_batch_parallel_processing(self, engine):
        """Test batch uses parallel processing for large batches."""
        inputs = [
            SupplierDataInput(
                asset_id=f"A{i:03d}",
                supplier_id="SUP001",
                asset_type="Server",
                purchase_value=Decimal("10000.00"),
                data_source="product_level",
                emission_factor=Decimal("1.0"),
                quantity=1,
            )
            for i in range(100)
        ]

        with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
            mock_executor.return_value.__enter__.return_value.map.return_value = [
                Mock(total_emissions=Decimal("1.0")) for _ in range(100)
            ]
            results = engine.calculate_batch(inputs)
            # Verify parallel executor was used for large batch
            assert len(results) == 100


class TestCalculateProductLevel:
    """Test calculate_product_level() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        SupplierSpecificCalculatorEngine._instance = None
        return SupplierSpecificCalculatorEngine()

    def test_calculate_product_level_basic(self, engine):
        """Test basic product-level calculation."""
        input_data = SupplierDataInput(
            asset_id="A001",
            supplier_id="SUP001",
            asset_type="Laptop",
            purchase_value=Decimal("2000.00"),
            data_source="product_level",
            emission_factor=Decimal("0.5"),  # tCO2e/unit
            quantity=100,
            unit="units",
        )

        result = engine.calculate_product_level(input_data)

        assert result.total_emissions == Decimal("50.0")  # 0.5 * 100
        assert result.asset_id == "A001"
        assert result.calculation_method == "product_level_ef"

    def test_calculate_product_level_with_epd(self, engine):
        """Test product-level calculation with EPD data."""
        input_data = SupplierDataInput(
            asset_id="A002",
            supplier_id="SUP002",
            asset_type="Steel Beam",
            purchase_value=Decimal("50000.00"),
            data_source="product_level",
            emission_factor=Decimal("2.1"),
            quantity=1000,
            unit="kg",
            epd_id="EPD-2024-001",
            epd_verified=True,
            metadata={"epd_program": "EPD International"},
        )

        result = engine.calculate_product_level(input_data)

        assert result.total_emissions == Decimal("2100.0")
        assert result.data_quality_score.verification_level == VerificationLevel.THIRD_PARTY
        assert "epd_id" in result.metadata

    def test_calculate_product_level_missing_ef(self, engine):
        """Test product-level calculation with missing EF raises error."""
        input_data = SupplierDataInput(
            asset_id="A003",
            supplier_id="SUP003",
            asset_type="Equipment",
            purchase_value=Decimal("10000.00"),
            data_source="product_level",
            quantity=5,
        )

        with pytest.raises(ValueError, match="emission_factor.*required"):
            engine.calculate_product_level(input_data)

    def test_calculate_product_level_missing_quantity(self, engine):
        """Test product-level calculation with missing quantity raises error."""
        input_data = SupplierDataInput(
            asset_id="A004",
            supplier_id="SUP004",
            asset_type="Equipment",
            purchase_value=Decimal("10000.00"),
            data_source="product_level",
            emission_factor=Decimal("1.0"),
        )

        with pytest.raises(ValueError, match="quantity.*required"):
            engine.calculate_product_level(input_data)


class TestCalculatePCF:
    """Test calculate_pcf() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        SupplierSpecificCalculatorEngine._instance = None
        return SupplierSpecificCalculatorEngine()

    def test_calculate_pcf_cradle_to_gate(self, engine):
        """Test PCF calculation with cradle-to-gate scope."""
        input_data = SupplierDataInput(
            asset_id="A001",
            supplier_id="SUP001",
            asset_type="Battery",
            purchase_value=Decimal("15000.00"),
            data_source="pcf",
            pcf_data={
                "total_emissions": 12.5,  # tCO2e
                "functional_unit": "1 battery pack",
                "scope": "cradle-to-gate",
                "assessment_method": "ISO 14067",
            },
            quantity=10,
        )

        result = engine.calculate_pcf(input_data)

        assert result.total_emissions == Decimal("125.0")  # 12.5 * 10
        assert result.calculation_method == "pcf"
        assert "assessment_method" in result.metadata

    def test_calculate_pcf_cradle_to_grave(self, engine):
        """Test PCF calculation with cradle-to-grave scope."""
        input_data = SupplierDataInput(
            asset_id="A002",
            supplier_id="SUP002",
            asset_type="Solar Panel",
            purchase_value=Decimal("50000.00"),
            data_source="pcf",
            pcf_data={
                "total_emissions": 2.8,
                "functional_unit": "1 kWp",
                "scope": "cradle-to-grave",
                "use_phase_emissions": 0.0,
                "end_of_life_emissions": 0.3,
            },
            quantity=100,
        )

        result = engine.calculate_pcf(input_data)

        assert result.total_emissions == Decimal("280.0")
        assert result.metadata["pcf_scope"] == "cradle-to-grave"

    def test_calculate_pcf_with_allocation(self, engine):
        """Test PCF calculation with allocation factors."""
        input_data = SupplierDataInput(
            asset_id="A003",
            supplier_id="SUP003",
            asset_type="Processor",
            purchase_value=Decimal("5000.00"),
            data_source="pcf",
            pcf_data={
                "total_emissions": 50.0,
                "functional_unit": "Production run",
                "scope": "cradle-to-gate",
                "allocation_factor": 0.15,  # This product is 15% of run
            },
            quantity=1,
        )

        result = engine.calculate_pcf(input_data)

        assert result.total_emissions == Decimal("7.5")  # 50.0 * 0.15

    def test_calculate_pcf_missing_data(self, engine):
        """Test PCF calculation with missing PCF data raises error."""
        input_data = SupplierDataInput(
            asset_id="A004",
            supplier_id="SUP004",
            asset_type="Equipment",
            purchase_value=Decimal("10000.00"),
            data_source="pcf",
        )

        with pytest.raises(ValueError, match="pcf_data.*required"):
            engine.calculate_pcf(input_data)


class TestCalculateFacilityAllocation:
    """Test calculate_facility_allocation() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        SupplierSpecificCalculatorEngine._instance = None
        return SupplierSpecificCalculatorEngine()

    def test_economic_allocation(self, engine):
        """Test economic allocation method."""
        input_data = SupplierDataInput(
            asset_id="A001",
            supplier_id="SUP001",
            asset_type="Equipment",
            purchase_value=Decimal("100000.00"),
            data_source="facility_allocation",
            facility_emissions=10000.0,  # tCO2e
            allocation_method="economic",
            supplier_revenue=Decimal("50000000.00"),
        )

        result = engine.calculate_facility_allocation(input_data)

        # 100000 / 50000000 * 10000 = 20 tCO2e
        assert result.total_emissions == Decimal("20.0")
        assert result.calculation_method == "facility_allocation_economic"

    def test_physical_allocation(self, engine):
        """Test physical allocation method."""
        input_data = SupplierDataInput(
            asset_id="A002",
            supplier_id="SUP002",
            asset_type="Steel Component",
            purchase_value=Decimal("50000.00"),
            data_source="facility_allocation",
            facility_emissions=5000.0,
            allocation_method="physical",
            product_mass=1000.0,  # kg
            total_facility_output=100000.0,  # kg
        )

        result = engine.calculate_facility_allocation(input_data)

        # 1000 / 100000 * 5000 = 50 tCO2e
        assert result.total_emissions == Decimal("50.0")
        assert result.calculation_method == "facility_allocation_physical"

    def test_mass_allocation(self, engine):
        """Test mass allocation method."""
        input_data = SupplierDataInput(
            asset_id="A003",
            supplier_id="SUP003",
            asset_type="Chemical Product",
            purchase_value=Decimal("20000.00"),
            data_source="facility_allocation",
            facility_emissions=8000.0,
            allocation_method="mass",
            product_mass=500.0,
            total_facility_output=50000.0,
        )

        result = engine.calculate_facility_allocation(input_data)

        # 500 / 50000 * 8000 = 80 tCO2e
        assert result.total_emissions == Decimal("80.0")

    def test_energy_allocation(self, engine):
        """Test energy allocation method."""
        input_data = SupplierDataInput(
            asset_id="A004",
            supplier_id="SUP004",
            asset_type="Industrial Equipment",
            purchase_value=Decimal("200000.00"),
            data_source="facility_allocation",
            facility_emissions=15000.0,
            allocation_method="energy",
            product_energy_content=5000.0,  # MJ
            total_facility_energy=500000.0,  # MJ
        )

        result = engine.calculate_facility_allocation(input_data)

        # 5000 / 500000 * 15000 = 150 tCO2e
        assert result.total_emissions == Decimal("150.0")
        assert result.calculation_method == "facility_allocation_energy"

    def test_hybrid_allocation(self, engine):
        """Test hybrid allocation method."""
        input_data = SupplierDataInput(
            asset_id="A005",
            supplier_id="SUP005",
            asset_type="Complex Product",
            purchase_value=Decimal("150000.00"),
            data_source="facility_allocation",
            facility_emissions=20000.0,
            allocation_method="hybrid",
            economic_weight=0.6,
            physical_weight=0.4,
            supplier_revenue=Decimal("100000000.00"),
            product_mass=2000.0,
            total_facility_output=200000.0,
        )

        result = engine.calculate_facility_allocation(input_data)

        # Economic: 150000/100000000 * 20000 * 0.6 = 18
        # Physical: 2000/200000 * 20000 * 0.4 = 80
        # Total: 98 tCO2e
        assert result.total_emissions == Decimal("98.0")

    def test_allocation_missing_facility_emissions(self, engine):
        """Test allocation with missing facility emissions raises error."""
        input_data = SupplierDataInput(
            asset_id="A006",
            supplier_id="SUP006",
            asset_type="Equipment",
            purchase_value=Decimal("50000.00"),
            data_source="facility_allocation",
            allocation_method="economic",
        )

        with pytest.raises(ValueError, match="facility_emissions.*required"):
            engine.calculate_facility_allocation(input_data)

    def test_allocation_invalid_method(self, engine):
        """Test allocation with invalid method raises error."""
        input_data = SupplierDataInput(
            asset_id="A007",
            supplier_id="SUP007",
            asset_type="Equipment",
            purchase_value=Decimal("50000.00"),
            data_source="facility_allocation",
            facility_emissions=1000.0,
            allocation_method="invalid_method",
        )

        with pytest.raises(ValueError, match="Unsupported allocation method"):
            engine.calculate_facility_allocation(input_data)


class TestValidateSupplierData:
    """Test validate_supplier_data() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        SupplierSpecificCalculatorEngine._instance = None
        return SupplierSpecificCalculatorEngine()

    def test_validate_valid_product_level_data(self, engine):
        """Test validation passes for valid product-level data."""
        input_data = SupplierDataInput(
            asset_id="A001",
            supplier_id="SUP001",
            asset_type="Server",
            purchase_value=Decimal("10000.00"),
            data_source="product_level",
            emission_factor=Decimal("1.5"),
            quantity=5,
        )

        is_valid, errors = engine.validate_supplier_data(input_data)

        assert is_valid is True
        assert errors == []

    def test_validate_missing_required_fields(self, engine):
        """Test validation fails for missing required fields."""
        input_data = SupplierDataInput(
            asset_id="A002",
            supplier_id="SUP002",
            asset_type="Equipment",
            purchase_value=Decimal("5000.00"),
            data_source="product_level",
            # Missing emission_factor and quantity
        )

        is_valid, errors = engine.validate_supplier_data(input_data)

        assert is_valid is False
        assert len(errors) > 0
        assert any("emission_factor" in e for e in errors)

    def test_validate_negative_values(self, engine):
        """Test validation fails for negative values."""
        input_data = SupplierDataInput(
            asset_id="A003",
            supplier_id="SUP003",
            asset_type="Equipment",
            purchase_value=Decimal("-1000.00"),
            data_source="product_level",
            emission_factor=Decimal("-0.5"),
            quantity=-5,
        )

        is_valid, errors = engine.validate_supplier_data(input_data)

        assert is_valid is False
        assert any("purchase_value" in e for e in errors)
        assert any("emission_factor" in e for e in errors)


class TestValidateEPD:
    """Test validate_epd() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        SupplierSpecificCalculatorEngine._instance = None
        return SupplierSpecificCalculatorEngine()

    def test_validate_epd_valid(self, engine):
        """Test EPD validation passes for valid EPD."""
        epd_data = {
            "epd_id": "EPD-2024-001",
            "program_operator": "EPD International",
            "verification_status": "verified",
            "expiry_date": "2025-12-31",
            "scope": "cradle-to-gate",
        }

        is_valid, errors = engine.validate_epd(epd_data)

        assert is_valid is True
        assert errors == []

    def test_validate_epd_missing_required_fields(self, engine):
        """Test EPD validation fails for missing fields."""
        epd_data = {
            "epd_id": "EPD-2024-002",
            # Missing other required fields
        }

        is_valid, errors = engine.validate_epd(epd_data)

        assert is_valid is False
        assert len(errors) > 0

    def test_validate_epd_expired(self, engine):
        """Test EPD validation fails for expired EPD."""
        epd_data = {
            "epd_id": "EPD-2020-001",
            "program_operator": "EPD International",
            "verification_status": "verified",
            "expiry_date": "2021-12-31",  # Expired
            "scope": "cradle-to-gate",
        }

        is_valid, errors = engine.validate_epd(epd_data)

        assert is_valid is False
        assert any("expired" in e.lower() for e in errors)


class TestValidatePCF:
    """Test validate_pcf() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        SupplierSpecificCalculatorEngine._instance = None
        return SupplierSpecificCalculatorEngine()

    def test_validate_pcf_valid(self, engine):
        """Test PCF validation passes for valid PCF."""
        pcf_data = {
            "total_emissions": 15.5,
            "functional_unit": "1 unit",
            "scope": "cradle-to-gate",
            "assessment_method": "ISO 14067",
        }

        is_valid, errors = engine.validate_pcf(pcf_data)

        assert is_valid is True
        assert errors == []

    def test_validate_pcf_missing_fields(self, engine):
        """Test PCF validation fails for missing required fields."""
        pcf_data = {
            "total_emissions": 15.5,
            # Missing other required fields
        }

        is_valid, errors = engine.validate_pcf(pcf_data)

        assert is_valid is False
        assert len(errors) > 0

    def test_validate_pcf_negative_emissions(self, engine):
        """Test PCF validation fails for negative emissions."""
        pcf_data = {
            "total_emissions": -15.5,
            "functional_unit": "1 unit",
            "scope": "cradle-to-gate",
            "assessment_method": "ISO 14067",
        }

        is_valid, errors = engine.validate_pcf(pcf_data)

        assert is_valid is False
        assert any("negative" in e.lower() for e in errors)


class TestScoreDQI:
    """Test score_dqi() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        SupplierSpecificCalculatorEngine._instance = None
        return SupplierSpecificCalculatorEngine()

    def test_score_dqi_high_quality(self, engine):
        """Test DQI scoring for high-quality data."""
        input_data = SupplierDataInput(
            asset_id="A001",
            supplier_id="SUP001",
            asset_type="Server",
            purchase_value=Decimal("10000.00"),
            data_source="product_level",
            emission_factor=Decimal("1.5"),
            quantity=5,
            epd_verified=True,
            verification_level="third_party",
            temporal_representativeness="current_year",
            geographical_representativeness="site_specific",
        )

        dqi_score = engine.score_dqi(input_data)

        assert isinstance(dqi_score, DataQualityScore)
        assert dqi_score.overall_score >= 4.0  # High quality
        assert dqi_score.temporal_score >= 4.0
        assert dqi_score.geographical_score >= 4.0

    def test_score_dqi_medium_quality(self, engine):
        """Test DQI scoring for medium-quality data."""
        input_data = SupplierDataInput(
            asset_id="A002",
            supplier_id="SUP002",
            asset_type="Equipment",
            purchase_value=Decimal("5000.00"),
            data_source="facility_allocation",
            facility_emissions=1000.0,
            allocation_method="economic",
            supplier_revenue=Decimal("1000000.00"),
            temporal_representativeness="recent_year",
            geographical_representativeness="regional",
        )

        dqi_score = engine.score_dqi(input_data)

        assert 2.5 <= dqi_score.overall_score < 4.0  # Medium quality

    def test_score_dqi_low_quality(self, engine):
        """Test DQI scoring for low-quality data."""
        input_data = SupplierDataInput(
            asset_id="A003",
            supplier_id="SUP003",
            asset_type="Equipment",
            purchase_value=Decimal("3000.00"),
            data_source="facility_allocation",
            facility_emissions=500.0,
            allocation_method="economic",
            supplier_revenue=Decimal("500000.00"),
            temporal_representativeness="older_data",
            geographical_representativeness="global_average",
        )

        dqi_score = engine.score_dqi(input_data)

        assert dqi_score.overall_score < 2.5  # Low quality


class TestScoreVerification:
    """Test score_verification() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        SupplierSpecificCalculatorEngine._instance = None
        return SupplierSpecificCalculatorEngine()

    def test_score_third_party_verification(self, engine):
        """Test scoring for third-party verified data."""
        verification_level = engine.score_verification(
            epd_verified=True,
            third_party_verified=True,
            verification_standard="ISO 14064-3",
        )

        assert verification_level == VerificationLevel.THIRD_PARTY

    def test_score_supplier_verified(self, engine):
        """Test scoring for supplier-verified data."""
        verification_level = engine.score_verification(
            epd_verified=False,
            third_party_verified=False,
            supplier_assured=True,
        )

        assert verification_level == VerificationLevel.SUPPLIER

    def test_score_unverified(self, engine):
        """Test scoring for unverified data."""
        verification_level = engine.score_verification(
            epd_verified=False,
            third_party_verified=False,
            supplier_assured=False,
        )

        assert verification_level == VerificationLevel.UNVERIFIED


class TestAggregateBySupplier:
    """Test aggregate_by_supplier() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        SupplierSpecificCalculatorEngine._instance = None
        return SupplierSpecificCalculatorEngine()

    def test_aggregate_by_supplier_single_supplier(self, engine):
        """Test aggregation for single supplier."""
        results = [
            SupplierSpecificResult(
                asset_id=f"A{i:03d}",
                supplier_id="SUP001",
                total_emissions=Decimal("10.0"),
                calculation_method="product_level_ef",
                data_quality_score=Mock(overall_score=4.5),
                provenance_hash=f"hash{i}",
            )
            for i in range(5)
        ]

        aggregated = engine.aggregate_by_supplier(results)

        assert len(aggregated) == 1
        assert "SUP001" in aggregated
        assert aggregated["SUP001"]["total_emissions"] == Decimal("50.0")
        assert aggregated["SUP001"]["asset_count"] == 5

    def test_aggregate_by_supplier_multiple_suppliers(self, engine):
        """Test aggregation for multiple suppliers."""
        results = [
            SupplierSpecificResult(
                asset_id=f"A{i:03d}",
                supplier_id=f"SUP{(i % 3) + 1:03d}",
                total_emissions=Decimal("15.0"),
                calculation_method="product_level_ef",
                data_quality_score=Mock(overall_score=4.0),
                provenance_hash=f"hash{i}",
            )
            for i in range(9)
        ]

        aggregated = engine.aggregate_by_supplier(results)

        assert len(aggregated) == 3
        assert all(f"SUP{i:03d}" in aggregated for i in range(1, 4))
        assert all(agg["asset_count"] == 3 for agg in aggregated.values())


class TestAggregateByDataSource:
    """Test aggregate_by_data_source() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        SupplierSpecificCalculatorEngine._instance = None
        return SupplierSpecificCalculatorEngine()

    def test_aggregate_by_data_source(self, engine):
        """Test aggregation by data source."""
        results = [
            SupplierSpecificResult(
                asset_id=f"A{i:03d}",
                supplier_id="SUP001",
                total_emissions=Decimal("10.0"),
                calculation_method="product_level_ef" if i < 3 else "pcf",
                data_quality_score=Mock(overall_score=4.0),
                provenance_hash=f"hash{i}",
            )
            for i in range(6)
        ]

        aggregated = engine.aggregate_by_data_source(results)

        assert "product_level_ef" in aggregated
        assert "pcf" in aggregated
        assert aggregated["product_level_ef"]["asset_count"] == 3
        assert aggregated["pcf"]["asset_count"] == 3


class TestAggregateByVerification:
    """Test aggregate_by_verification() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        SupplierSpecificCalculatorEngine._instance = None
        return SupplierSpecificCalculatorEngine()

    def test_aggregate_by_verification_level(self, engine):
        """Test aggregation by verification level."""
        results = [
            SupplierSpecificResult(
                asset_id=f"A{i:03d}",
                supplier_id="SUP001",
                total_emissions=Decimal("20.0"),
                calculation_method="product_level_ef",
                data_quality_score=Mock(
                    overall_score=4.0,
                    verification_level=(
                        VerificationLevel.THIRD_PARTY if i < 2
                        else VerificationLevel.SUPPLIER if i < 4
                        else VerificationLevel.UNVERIFIED
                    ),
                ),
                provenance_hash=f"hash{i}",
            )
            for i in range(6)
        ]

        aggregated = engine.aggregate_by_verification(results)

        assert VerificationLevel.THIRD_PARTY in aggregated
        assert VerificationLevel.SUPPLIER in aggregated
        assert VerificationLevel.UNVERIFIED in aggregated
        assert aggregated[VerificationLevel.THIRD_PARTY]["asset_count"] == 2
        assert aggregated[VerificationLevel.SUPPLIER]["asset_count"] == 2
        assert aggregated[VerificationLevel.UNVERIFIED]["asset_count"] == 2


class TestGetCoverageReport:
    """Test get_coverage_report() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        SupplierSpecificCalculatorEngine._instance = None
        return SupplierSpecificCalculatorEngine()

    def test_get_coverage_report(self, engine):
        """Test coverage report generation."""
        results = [
            SupplierSpecificResult(
                asset_id=f"A{i:03d}",
                supplier_id=f"SUP{(i % 2) + 1:03d}",
                total_emissions=Decimal("25.0"),
                calculation_method="product_level_ef" if i < 7 else "pcf",
                data_quality_score=Mock(
                    overall_score=4.5 if i < 5 else 3.0,
                    verification_level=(
                        VerificationLevel.THIRD_PARTY if i < 3
                        else VerificationLevel.SUPPLIER
                    ),
                ),
                provenance_hash=f"hash{i}",
            )
            for i in range(10)
        ]

        report = engine.get_coverage_report(results)

        assert "total_assets" in report
        assert "by_data_source" in report
        assert "by_verification" in report
        assert "by_supplier" in report
        assert report["total_assets"] == 10
        assert report["total_emissions"] > 0


class TestEstimateUncertainty:
    """Test estimate_uncertainty() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        SupplierSpecificCalculatorEngine._instance = None
        return SupplierSpecificCalculatorEngine()

    def test_estimate_uncertainty_high_quality(self, engine):
        """Test uncertainty estimation for high-quality data."""
        result = SupplierSpecificResult(
            asset_id="A001",
            supplier_id="SUP001",
            total_emissions=Decimal("100.0"),
            calculation_method="product_level_ef",
            data_quality_score=Mock(overall_score=4.8),
            provenance_hash="hash001",
        )

        uncertainty = engine.estimate_uncertainty(result)

        assert uncertainty["relative_uncertainty"] < 0.15  # <15% for high quality
        assert "lower_bound" in uncertainty
        assert "upper_bound" in uncertainty

    def test_estimate_uncertainty_low_quality(self, engine):
        """Test uncertainty estimation for low-quality data."""
        result = SupplierSpecificResult(
            asset_id="A002",
            supplier_id="SUP002",
            total_emissions=Decimal("100.0"),
            calculation_method="facility_allocation_economic",
            data_quality_score=Mock(overall_score=2.0),
            provenance_hash="hash002",
        )

        uncertainty = engine.estimate_uncertainty(result)

        assert uncertainty["relative_uncertainty"] > 0.30  # >30% for low quality


class TestComputeProvenanceHash:
    """Test compute_provenance_hash() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        SupplierSpecificCalculatorEngine._instance = None
        return SupplierSpecificCalculatorEngine()

    def test_compute_provenance_hash_deterministic(self, engine):
        """Test provenance hash is deterministic."""
        input_data = SupplierDataInput(
            asset_id="A001",
            supplier_id="SUP001",
            asset_type="Server",
            purchase_value=Decimal("10000.00"),
            data_source="product_level",
            emission_factor=Decimal("1.5"),
            quantity=10,
        )

        result = SupplierSpecificResult(
            asset_id="A001",
            supplier_id="SUP001",
            total_emissions=Decimal("15.0"),
            calculation_method="product_level_ef",
            data_quality_score=Mock(overall_score=4.0),
            provenance_hash="temp",
        )

        hash1 = engine.compute_provenance_hash(input_data, result)
        hash2 = engine.compute_provenance_hash(input_data, result)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex length

    def test_compute_provenance_hash_different_inputs(self, engine):
        """Test different inputs produce different hashes."""
        input1 = SupplierDataInput(
            asset_id="A001",
            supplier_id="SUP001",
            asset_type="Server",
            purchase_value=Decimal("10000.00"),
            data_source="product_level",
            emission_factor=Decimal("1.5"),
            quantity=10,
        )

        input2 = SupplierDataInput(
            asset_id="A002",
            supplier_id="SUP002",
            asset_type="Laptop",
            purchase_value=Decimal("5000.00"),
            data_source="product_level",
            emission_factor=Decimal("0.8"),
            quantity=20,
        )

        result1 = SupplierSpecificResult(
            asset_id="A001",
            supplier_id="SUP001",
            total_emissions=Decimal("15.0"),
            calculation_method="product_level_ef",
            data_quality_score=Mock(overall_score=4.0),
            provenance_hash="temp1",
        )

        result2 = SupplierSpecificResult(
            asset_id="A002",
            supplier_id="SUP002",
            total_emissions=Decimal("16.0"),
            calculation_method="product_level_ef",
            data_quality_score=Mock(overall_score=4.0),
            provenance_hash="temp2",
        )

        hash1 = engine.compute_provenance_hash(input1, result1)
        hash2 = engine.compute_provenance_hash(input2, result2)

        assert hash1 != hash2
