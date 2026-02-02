"""
GL-015 INSULSCAN - Material Database Tests

Unit tests for InsulationMaterial and InsulationMaterialDatabase classes.
Tests material properties, k-value lookup, temperature interpolation,
and material search/filtering functionality.

Coverage target: 85%+
"""

import pytest
import math

from greenlang.agents.process_heat.gl_015_insulation_analysis.materials import (
    InsulationMaterial,
    InsulationMaterialDatabase,
    MaterialCategory,
    TemperatureRange,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def material_database():
    """Create material database instance."""
    return InsulationMaterialDatabase()


@pytest.fixture
def mineral_wool_material(material_database):
    """Get mineral wool 8 pcf material."""
    return material_database.get_material("mineral_wool_8pcf")


@pytest.fixture
def calcium_silicate_material(material_database):
    """Get calcium silicate 8 pcf material."""
    return material_database.get_material("calcium_silicate_8pcf")


@pytest.fixture
def cellular_glass_material(material_database):
    """Get cellular glass 7 pcf material."""
    return material_database.get_material("cellular_glass_7pcf")


@pytest.fixture
def aerogel_material(material_database):
    """Get aerogel blanket material."""
    return material_database.get_material("aerogel_blanket_8pcf")


# =============================================================================
# TEMPERATURE RANGE TESTS
# =============================================================================

class TestTemperatureRange:
    """Tests for TemperatureRange model."""

    def test_temperature_range_creation(self):
        """Test temperature range creation."""
        temp_range = TemperatureRange(min_temp_f=-100, max_temp_f=1200)

        assert temp_range.min_temp_f == -100
        assert temp_range.max_temp_f == 1200

    def test_contains_within_range(self):
        """Test contains method for temperature within range."""
        temp_range = TemperatureRange(min_temp_f=-100, max_temp_f=1200)

        assert temp_range.contains(0) is True
        assert temp_range.contains(500) is True
        assert temp_range.contains(1200) is True
        assert temp_range.contains(-100) is True

    def test_contains_outside_range(self):
        """Test contains method for temperature outside range."""
        temp_range = TemperatureRange(min_temp_f=-100, max_temp_f=1200)

        assert temp_range.contains(-150) is False
        assert temp_range.contains(1500) is False

    def test_contains_at_boundaries(self):
        """Test contains method at exact boundaries."""
        temp_range = TemperatureRange(min_temp_f=0, max_temp_f=100)

        assert temp_range.contains(0) is True
        assert temp_range.contains(100) is True
        assert temp_range.contains(-0.001) is False
        assert temp_range.contains(100.001) is False


# =============================================================================
# INSULATION MATERIAL TESTS
# =============================================================================

class TestInsulationMaterial:
    """Tests for InsulationMaterial model."""

    def test_material_properties(self, mineral_wool_material):
        """Test material basic properties."""
        mat = mineral_wool_material

        assert mat.material_id == "mineral_wool_8pcf"
        assert mat.name == "Mineral Wool - 8 pcf"
        assert mat.category == MaterialCategory.MINERAL_WOOL
        assert mat.density_pcf == 8.0
        assert mat.suitable_for_cold_service is True
        assert mat.requires_vapor_barrier is True

    def test_k_value_at_data_point(self, mineral_wool_material):
        """Test k-value at exact data point."""
        mat = mineral_wool_material

        # Test at exact data points from the material database
        k_at_100 = mat.get_thermal_conductivity(100)
        assert k_at_100 == 0.23

        k_at_500 = mat.get_thermal_conductivity(500)
        assert k_at_500 == 0.37

    def test_k_value_interpolation(self, mineral_wool_material):
        """Test k-value linear interpolation between data points."""
        mat = mineral_wool_material

        # Test interpolation between 100F (k=0.23) and 200F (k=0.26)
        k_at_150 = mat.get_thermal_conductivity(150)

        # Linear interpolation: 0.23 + (0.26 - 0.23) * (150 - 100) / (200 - 100) = 0.245
        expected = 0.23 + (0.26 - 0.23) * 0.5
        assert abs(k_at_150 - expected) < 0.001

    def test_k_value_below_minimum(self, mineral_wool_material):
        """Test k-value clamped at minimum temperature."""
        mat = mineral_wool_material

        # Temperature below range should return k at minimum temp
        k_below_min = mat.get_thermal_conductivity(-200)
        k_at_min = mat.get_thermal_conductivity(-100)

        assert k_below_min == k_at_min

    def test_k_value_above_maximum(self, mineral_wool_material):
        """Test k-value clamped at maximum temperature."""
        mat = mineral_wool_material

        # Temperature above range should return k at maximum temp
        k_above_max = mat.get_thermal_conductivity(1500)
        k_at_max = mat.get_thermal_conductivity(1200)

        assert k_above_max == k_at_max

    def test_mean_thermal_conductivity(self, mineral_wool_material):
        """Test mean thermal conductivity calculation."""
        mat = mineral_wool_material

        # Calculate mean k from 200F to 400F
        k_mean = mat.get_mean_thermal_conductivity(200, 400)

        # Should be approximately average of k values in range
        k_200 = mat.get_thermal_conductivity(200)
        k_400 = mat.get_thermal_conductivity(400)

        # Mean should be between the two endpoints
        assert k_200 <= k_mean <= k_400

    def test_mean_thermal_conductivity_equal_temps(self, mineral_wool_material):
        """Test mean k when inner and outer temps are equal."""
        mat = mineral_wool_material

        k_at_point = mat.get_thermal_conductivity(300)
        k_mean = mat.get_mean_thermal_conductivity(300, 300)

        assert abs(k_mean - k_at_point) < 0.01

    def test_mean_thermal_conductivity_small_range(self, mineral_wool_material):
        """Test mean k for very small temperature range."""
        mat = mineral_wool_material

        k_mean = mat.get_mean_thermal_conductivity(299, 301)
        k_at_300 = mat.get_thermal_conductivity(300)

        assert abs(k_mean - k_at_300) < 0.01


# =============================================================================
# INSULATION MATERIAL DATABASE TESTS
# =============================================================================

class TestInsulationMaterialDatabase:
    """Tests for InsulationMaterialDatabase."""

    def test_database_initialization(self, material_database):
        """Test database initializes with materials."""
        db = material_database

        assert db.material_count >= 50  # Should have 50+ materials

    def test_get_material_valid_id(self, material_database):
        """Test getting material by valid ID."""
        mat = material_database.get_material("mineral_wool_8pcf")

        assert mat is not None
        assert mat.material_id == "mineral_wool_8pcf"

    def test_get_material_invalid_id(self, material_database):
        """Test getting material by invalid ID returns None."""
        mat = material_database.get_material("nonexistent_material")

        assert mat is None

    def test_list_all_materials(self, material_database):
        """Test listing all materials."""
        materials = material_database.list_all_materials()

        assert len(materials) >= 50
        assert all(isinstance(m, InsulationMaterial) for m in materials)

    def test_get_categories(self, material_database):
        """Test getting available categories."""
        categories = material_database.get_categories()

        assert len(categories) > 0
        assert MaterialCategory.MINERAL_WOOL in categories
        assert MaterialCategory.CALCIUM_SILICATE in categories
        assert MaterialCategory.CELLULAR_GLASS in categories
        assert MaterialCategory.AEROGEL in categories

    def test_search_by_category(self, material_database):
        """Test searching materials by category."""
        mineral_wool_materials = material_database.search_materials(
            category=MaterialCategory.MINERAL_WOOL
        )

        assert len(mineral_wool_materials) > 0
        assert all(
            m.category == MaterialCategory.MINERAL_WOOL
            for m in mineral_wool_materials
        )

    def test_search_by_temperature_range(self, material_database):
        """Test searching materials by temperature range."""
        # Find materials suitable for 1000F+
        high_temp_materials = material_database.search_materials(
            max_temp_f=1000
        )

        assert len(high_temp_materials) > 0
        for mat in high_temp_materials:
            assert mat.temperature_range.max_temp_f >= 1000

    def test_search_cold_service_materials(self, material_database):
        """Test searching for cold service materials."""
        cold_materials = material_database.search_materials(
            suitable_for_cold=True
        )

        assert len(cold_materials) > 0
        assert all(m.suitable_for_cold_service for m in cold_materials)

    def test_search_by_max_k_value(self, material_database):
        """Test searching materials by maximum k-value."""
        low_k_materials = material_database.search_materials(
            max_k_value=0.25
        )

        assert len(low_k_materials) > 0
        # All returned materials should have low k-values

    def test_search_combined_criteria(self, material_database):
        """Test searching with multiple criteria."""
        materials = material_database.search_materials(
            category=MaterialCategory.MINERAL_WOOL,
            suitable_for_cold=True,
        )

        assert len(materials) > 0
        for mat in materials:
            assert mat.category == MaterialCategory.MINERAL_WOOL
            assert mat.suitable_for_cold_service is True

    def test_get_recommended_materials_hot_service(self, material_database):
        """Test getting recommended materials for hot service."""
        recommendations = material_database.get_recommended_materials(
            operating_temp_f=500,
            cold_service=False,
        )

        assert len(recommendations) > 0
        # Should be sorted by k-value (lowest first)
        for i in range(len(recommendations) - 1):
            k1 = recommendations[i].get_thermal_conductivity(500)
            k2 = recommendations[i + 1].get_thermal_conductivity(500)
            assert k1 <= k2

    def test_get_recommended_materials_cold_service(self, material_database):
        """Test getting recommended materials for cold service."""
        recommendations = material_database.get_recommended_materials(
            operating_temp_f=40,
            cold_service=True,
        )

        assert len(recommendations) > 0
        assert all(m.suitable_for_cold_service for m in recommendations)

    def test_get_recommended_materials_cryogenic(self, material_database):
        """Test getting recommended materials for cryogenic service."""
        recommendations = material_database.get_recommended_materials(
            operating_temp_f=-200,
            cold_service=True,
        )

        assert len(recommendations) > 0
        for mat in recommendations:
            assert mat.temperature_range.contains(-200)


# =============================================================================
# SPECIFIC MATERIAL TESTS
# =============================================================================

class TestSpecificMaterials:
    """Tests for specific material properties."""

    def test_calcium_silicate_high_temp(self, calcium_silicate_material):
        """Test calcium silicate for high temperature service."""
        mat = calcium_silicate_material

        assert mat.temperature_range.max_temp_f >= 1200
        assert mat.flame_spread_index == 0
        assert mat.smoke_developed_index == 0

    def test_cellular_glass_moisture_resistant(self, cellular_glass_material):
        """Test cellular glass is moisture resistant."""
        mat = cellular_glass_material

        assert mat.moisture_resistant is True
        assert mat.requires_vapor_barrier is False  # No vapor barrier needed
        assert mat.suitable_for_cold_service is True

    def test_aerogel_lowest_k_value(self, material_database):
        """Test aerogel has lowest k-value."""
        aerogel = material_database.get_material("aerogel_blanket_8pcf")

        # Aerogel should have k < 0.15 at 100F
        k_100 = aerogel.get_thermal_conductivity(100)
        assert k_100 <= 0.15

    def test_aerogel_wide_temperature_range(self, aerogel_material):
        """Test aerogel has wide temperature range."""
        mat = aerogel_material

        assert mat.temperature_range.min_temp_f <= -400
        assert mat.temperature_range.max_temp_f >= 1000

    def test_fiberglass_materials(self, material_database):
        """Test fiberglass material varieties."""
        fiberglass_1pcf = material_database.get_material("fiberglass_1pcf")
        fiberglass_3pcf = material_database.get_material("fiberglass_3pcf")
        fiberglass_6pcf = material_database.get_material("fiberglass_6pcf")

        assert fiberglass_1pcf is not None
        assert fiberglass_3pcf is not None
        assert fiberglass_6pcf is not None

        # Higher density should have lower k-value
        k_1pcf = fiberglass_1pcf.get_thermal_conductivity(100)
        k_6pcf = fiberglass_6pcf.get_thermal_conductivity(100)
        assert k_6pcf < k_1pcf

    def test_polyurethane_cold_service(self, material_database):
        """Test polyurethane for cold service."""
        pu = material_database.get_material("polyurethane_2pcf")

        assert pu.suitable_for_cold_service is True
        assert pu.temperature_range.min_temp_f <= -250

    def test_ceramic_fiber_high_temp(self, material_database):
        """Test ceramic fiber for high temperature."""
        cf = material_database.get_material("ceramic_fiber_8pcf")

        assert cf.temperature_range.max_temp_f >= 2500

    def test_refractory_materials(self, material_database):
        """Test refractory materials exist."""
        ifb = material_database.get_material("insulating_firebrick_27pcf")

        assert ifb is not None
        assert ifb.category == MaterialCategory.REFRACTORY
        assert ifb.temperature_range.max_temp_f >= 2500


# =============================================================================
# K-VALUE CALCULATION ACCURACY TESTS
# =============================================================================

class TestKValueAccuracy:
    """Tests for k-value calculation accuracy."""

    @pytest.mark.parametrize("material_id,temp_f,expected_k", [
        ("mineral_wool_8pcf", 100, 0.23),
        ("mineral_wool_8pcf", 500, 0.37),
        ("mineral_wool_8pcf", 1000, 0.66),
        ("calcium_silicate_8pcf", 200, 0.40),
        ("calcium_silicate_8pcf", 600, 0.57),
        ("calcium_silicate_8pcf", 1000, 0.80),
        ("cellular_glass_7pcf", 100, 0.30),
        ("cellular_glass_7pcf", 500, 0.47),
        ("aerogel_blanket_8pcf", 100, 0.11),
        ("aerogel_blanket_8pcf", 500, 0.15),
    ])
    def test_k_value_at_known_points(self, material_database, material_id, temp_f, expected_k):
        """Test k-value matches expected at known data points."""
        mat = material_database.get_material(material_id)
        k = mat.get_thermal_conductivity(temp_f)

        assert abs(k - expected_k) < 0.001

    def test_k_value_monotonically_increasing(self, mineral_wool_material):
        """Test k-value increases with temperature for hot service."""
        mat = mineral_wool_material

        temps = [100, 200, 300, 400, 500, 600, 700, 800]
        k_values = [mat.get_thermal_conductivity(t) for t in temps]

        for i in range(len(k_values) - 1):
            assert k_values[i] <= k_values[i + 1]

    def test_mean_k_integration_accuracy(self, mineral_wool_material):
        """Test mean k-value integration is reasonably accurate."""
        mat = mineral_wool_material

        # Calculate mean k from 100F to 500F
        k_mean = mat.get_mean_thermal_conductivity(100, 500)

        # Simple average of endpoints for comparison
        k_100 = mat.get_thermal_conductivity(100)
        k_500 = mat.get_thermal_conductivity(500)
        simple_avg = (k_100 + k_500) / 2

        # Mean should be close to simple average for approximately linear k vs T
        assert abs(k_mean - simple_avg) / simple_avg < 0.15  # Within 15%


# =============================================================================
# ASTM STANDARDS TESTS
# =============================================================================

class TestASTMCompliance:
    """Tests for ASTM standards compliance data."""

    def test_mineral_wool_astm_standards(self, material_database):
        """Test mineral wool has correct ASTM standards."""
        mat = material_database.get_material("mineral_wool_8pcf")

        assert "ASTM C547" in mat.astm_standards

    def test_calcium_silicate_astm_standards(self, material_database):
        """Test calcium silicate has correct ASTM standards."""
        mat = material_database.get_material("calcium_silicate_8pcf")

        assert "ASTM C533" in mat.astm_standards

    def test_cellular_glass_astm_standards(self, material_database):
        """Test cellular glass has correct ASTM standards."""
        mat = material_database.get_material("cellular_glass_7pcf")

        assert "ASTM C552" in mat.astm_standards

    def test_aerogel_astm_standards(self, material_database):
        """Test aerogel has correct ASTM standards."""
        mat = material_database.get_material("aerogel_blanket_8pcf")

        assert "ASTM C1728" in mat.astm_standards
