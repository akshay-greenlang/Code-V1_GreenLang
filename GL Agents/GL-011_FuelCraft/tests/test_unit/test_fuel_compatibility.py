# -*- coding: utf-8 -*-
"""
Unit Tests for FuelCompatibilityMatrix

Tests all methods of FuelCompatibilityMatrix with comprehensive coverage.
Validates:
- Fuel type registration and lookup
- Compatibility rule matching
- Fail-closed behavior for unknown combinations
- Contamination risk scoring
- Segregation requirements
- Provenance hash generation

Author: GL-TestEngineer
Date: 2025-01-01
"""

import pytest
from datetime import datetime, timezone
from decimal import Decimal

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from safety.fuel_compatibility import (
    FuelCompatibilityMatrix,
    FuelType,
    FuelCategory,
    CompatibilityRule,
    CompatibilityLevel,
    ContaminationRisk,
    SegregationType,
    CompatibilityCheckResult,
    DEFAULT_FUEL_TYPES,
    DEFAULT_COMPATIBILITY_RULES,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def fuel_matrix():
    """Create FuelCompatibilityMatrix instance with defaults."""
    return FuelCompatibilityMatrix()


@pytest.fixture
def custom_fuel_type():
    """Create a custom fuel type for testing."""
    return FuelType(
        fuel_id="CUSTOM_FUEL",
        fuel_name="Custom Test Fuel",
        category=FuelCategory.DISTILLATE,
        flash_point_min_c=60,
        flash_point_max_c=100,
        density_min_kg_m3=850,
        density_max_kg_m3=900,
        viscosity_min_cst=2,
        viscosity_max_cst=10,
        sulfur_max_pct=0.1,
        water_sensitive=True,
        astm_standard="ASTM TEST"
    )


@pytest.fixture
def custom_compatibility_rule():
    """Create a custom compatibility rule for testing."""
    return CompatibilityRule(
        rule_id="CUSTOM_R001",
        fuel_a="CUSTOM_FUEL",
        fuel_b="ULSD",
        compatibility=CompatibilityLevel.FULLY_COMPATIBLE,
        contamination_risk=ContaminationRisk.LOW,
        technical_reason="Test rule for custom fuel"
    )


# =============================================================================
# Initialization Tests
# =============================================================================

@pytest.mark.unit
class TestFuelCompatibilityMatrixInit:
    """Tests for FuelCompatibilityMatrix initialization."""

    def test_default_initialization(self, fuel_matrix):
        """Test matrix initializes with default fuel types and rules."""
        fuel_types = fuel_matrix.list_fuel_types()

        # Check default fuel types are loaded
        assert "HFO" in fuel_types
        assert "VLSFO" in fuel_types
        assert "MGO" in fuel_types
        assert "ULSD" in fuel_types
        assert "FAME" in fuel_types
        assert "MOGAS" in fuel_types
        assert "JET_A" in fuel_types

    def test_custom_fuel_types_initialization(self, custom_fuel_type):
        """Test matrix initializes with custom fuel types."""
        custom_types = {"CUSTOM": custom_fuel_type}
        matrix = FuelCompatibilityMatrix(fuel_types=custom_types)

        fuel_types = matrix.list_fuel_types()

        assert "CUSTOM" in fuel_types
        # Default types should not be present
        assert "HFO" not in fuel_types

    def test_custom_rules_initialization(self, custom_compatibility_rule):
        """Test matrix initializes with custom rules."""
        # First need fuel types for the rule
        custom_types = {
            "CUSTOM_FUEL": FuelType(
                fuel_id="CUSTOM_FUEL",
                fuel_name="Custom",
                category=FuelCategory.DISTILLATE,
            ),
            "ULSD": DEFAULT_FUEL_TYPES["ULSD"],
        }
        matrix = FuelCompatibilityMatrix(
            fuel_types=custom_types,
            compatibility_rules=[custom_compatibility_rule]
        )

        result = matrix.check_compatibility("CUSTOM_FUEL", "ULSD")

        assert result.compatibility_level == CompatibilityLevel.FULLY_COMPATIBLE


# =============================================================================
# Compatibility Check Tests
# =============================================================================

@pytest.mark.unit
class TestCompatibilityCheck:
    """Tests for compatibility checking."""

    def test_same_fuel_always_compatible(self, fuel_matrix):
        """Test that same fuel type is always fully compatible."""
        result = fuel_matrix.check_compatibility("HFO", "HFO")

        assert result.can_co_mingle is True
        assert result.compatibility_level == CompatibilityLevel.FULLY_COMPATIBLE
        assert result.contamination_risk == ContaminationRisk.NONE
        assert result.contamination_score == 0.0
        assert result.segregation_required == SegregationType.NONE

    def test_fully_compatible_fuels(self, fuel_matrix):
        """Test fully compatible fuel combination."""
        result = fuel_matrix.check_compatibility("MGO", "ULSD")

        assert result.can_co_mingle is True
        assert result.compatibility_level == CompatibilityLevel.FULLY_COMPATIBLE
        assert result.contamination_risk == ContaminationRisk.NONE

    def test_conditionally_compatible_fuels(self, fuel_matrix):
        """Test conditionally compatible fuel combination."""
        result = fuel_matrix.check_compatibility("HFO", "VLSFO")

        assert result.can_co_mingle is True
        assert result.compatibility_level == CompatibilityLevel.CONDITIONALLY_COMPATIBLE
        assert len(result.conditions) > 0
        assert result.max_blend_ratio is not None

    def test_conditionally_compatible_with_ratio_exceeded(self, fuel_matrix):
        """Test conditional compatibility fails when ratio is exceeded."""
        # HFO/VLSFO has max_blend_ratio of 0.3 (30%)
        result = fuel_matrix.check_compatibility("HFO", "VLSFO", proposed_ratio=0.5)

        assert result.can_co_mingle is False
        assert "exceeds" in result.reason.lower()

    def test_conditionally_compatible_within_ratio(self, fuel_matrix):
        """Test conditional compatibility passes within ratio."""
        result = fuel_matrix.check_compatibility("HFO", "VLSFO", proposed_ratio=0.2)

        assert result.can_co_mingle is True

    def test_incompatible_fuels(self, fuel_matrix):
        """Test incompatible fuel combination."""
        result = fuel_matrix.check_compatibility("MOGAS", "ULSD")

        assert result.can_co_mingle is False
        assert result.compatibility_level == CompatibilityLevel.INCOMPATIBLE
        assert result.contamination_risk == ContaminationRisk.CRITICAL
        assert result.segregation_required == SegregationType.DEDICATED_TANK

    def test_unknown_fuel_raises_error(self, fuel_matrix):
        """Test that unknown fuel type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown fuel type"):
            fuel_matrix.check_compatibility("UNKNOWN_FUEL", "HFO")

    def test_second_unknown_fuel_raises_error(self, fuel_matrix):
        """Test that unknown second fuel type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown fuel type"):
            fuel_matrix.check_compatibility("HFO", "UNKNOWN_FUEL")


# =============================================================================
# Fail-Closed Behavior Tests
# =============================================================================

@pytest.mark.unit
@pytest.mark.safety
class TestFailClosedBehavior:
    """Tests for fail-closed safety behavior."""

    def test_unknown_combination_blocked(self):
        """Test that unknown combinations are blocked (fail-closed)."""
        # Create matrix with fuel types but no rules between them
        custom_types = {
            "FUEL_A": FuelType(
                fuel_id="FUEL_A",
                fuel_name="Fuel A",
                category=FuelCategory.DISTILLATE,
            ),
            "FUEL_B": FuelType(
                fuel_id="FUEL_B",
                fuel_name="Fuel B",
                category=FuelCategory.RESIDUAL,
            ),
        }
        matrix = FuelCompatibilityMatrix(
            fuel_types=custom_types,
            compatibility_rules=[]  # No rules
        )

        result = matrix.check_compatibility("FUEL_A", "FUEL_B")

        # Should block unknown combinations
        assert result.can_co_mingle is False
        assert result.compatibility_level == CompatibilityLevel.UNKNOWN
        assert result.contamination_risk == ContaminationRisk.HIGH
        assert result.segregation_required == SegregationType.DEDICATED_TANK
        assert "fail-closed" in result.reason.lower()

    def test_unknown_combination_high_contamination_score(self):
        """Test unknown combinations have high contamination score."""
        custom_types = {
            "FUEL_X": FuelType(
                fuel_id="FUEL_X",
                fuel_name="Fuel X",
                category=FuelCategory.BIOFUEL,
            ),
            "FUEL_Y": FuelType(
                fuel_id="FUEL_Y",
                fuel_name="Fuel Y",
                category=FuelCategory.CRUDE,
            ),
        }
        matrix = FuelCompatibilityMatrix(
            fuel_types=custom_types,
            compatibility_rules=[]
        )

        result = matrix.check_compatibility("FUEL_X", "FUEL_Y")

        assert result.contamination_score >= 80.0


# =============================================================================
# Contamination Risk Tests
# =============================================================================

@pytest.mark.unit
class TestContaminationRisk:
    """Tests for contamination risk scoring."""

    def test_contamination_score_none(self, fuel_matrix):
        """Test contamination score is 0 for no risk."""
        # Same fuel has no risk
        result = fuel_matrix.check_compatibility("ULSD", "ULSD")
        assert result.contamination_score == 0.0

    def test_contamination_score_critical(self, fuel_matrix):
        """Test contamination score is 100 for critical risk."""
        result = fuel_matrix.check_compatibility("MOGAS", "ULSD")
        assert result.contamination_score == 100.0

    def test_get_contamination_risk_score_same_fuel(self, fuel_matrix):
        """Test contamination risk score for same fuel is 0."""
        score = fuel_matrix.get_contamination_risk_score("HFO", "HFO")
        assert score == 0.0

    def test_get_contamination_risk_score_unknown(self):
        """Test contamination risk score for unknown combination."""
        custom_types = {
            "FUEL_1": FuelType(fuel_id="FUEL_1", fuel_name="F1", category=FuelCategory.DISTILLATE),
            "FUEL_2": FuelType(fuel_id="FUEL_2", fuel_name="F2", category=FuelCategory.RESIDUAL),
        }
        matrix = FuelCompatibilityMatrix(fuel_types=custom_types, compatibility_rules=[])

        score = matrix.get_contamination_risk_score("FUEL_1", "FUEL_2")
        assert score == 80.0  # Default for unknown


# =============================================================================
# Segregation Requirements Tests
# =============================================================================

@pytest.mark.unit
class TestSegregationRequirements:
    """Tests for segregation requirements."""

    def test_get_segregation_requirements(self, fuel_matrix):
        """Test getting segregation requirements for a fuel."""
        requirements = fuel_matrix.get_segregation_requirements("MOGAS")

        # MOGAS should require dedicated tanks for most fuels
        assert requirements["ULSD"] == SegregationType.DEDICATED_TANK
        assert requirements["JET_A"] == SegregationType.DEDICATED_TANK

    def test_segregation_requirements_unknown_fuel_error(self, fuel_matrix):
        """Test segregation requirements raises error for unknown fuel."""
        with pytest.raises(ValueError, match="Unknown fuel type"):
            fuel_matrix.get_segregation_requirements("UNKNOWN_FUEL")


# =============================================================================
# Compatible Fuels List Tests
# =============================================================================

@pytest.mark.unit
class TestCompatibleFuelsList:
    """Tests for listing compatible fuels."""

    def test_get_compatible_fuels_includes_self(self, fuel_matrix):
        """Test compatible fuels list includes the fuel itself."""
        compatible = fuel_matrix.get_compatible_fuels("ULSD")
        assert "ULSD" in compatible

    def test_get_compatible_fuels_fully_compatible(self, fuel_matrix):
        """Test compatible fuels list includes fully compatible fuels."""
        compatible = fuel_matrix.get_compatible_fuels("MGO")
        assert "ULSD" in compatible  # MGO and ULSD are fully compatible

    def test_get_compatible_fuels_includes_conditional(self, fuel_matrix):
        """Test compatible fuels list includes conditional when flag set."""
        compatible = fuel_matrix.get_compatible_fuels("ULSD", include_conditional=True)
        assert "FAME" in compatible  # ULSD/FAME is conditional

    def test_get_compatible_fuels_excludes_conditional(self, fuel_matrix):
        """Test compatible fuels list excludes conditional when flag not set."""
        compatible = fuel_matrix.get_compatible_fuels("ULSD", include_conditional=False)
        assert "FAME" not in compatible  # ULSD/FAME is conditional

    def test_get_compatible_fuels_unknown_fuel_error(self, fuel_matrix):
        """Test compatible fuels raises error for unknown fuel."""
        with pytest.raises(ValueError, match="Unknown fuel type"):
            fuel_matrix.get_compatible_fuels("UNKNOWN_FUEL")


# =============================================================================
# Fuel Type Registration Tests
# =============================================================================

@pytest.mark.unit
class TestFuelTypeRegistration:
    """Tests for fuel type registration."""

    def test_register_fuel_type(self, fuel_matrix, custom_fuel_type):
        """Test registering a new fuel type."""
        fuel_matrix.register_fuel_type(custom_fuel_type)

        fuel_types = fuel_matrix.list_fuel_types()
        assert "CUSTOM_FUEL" in fuel_types

    def test_get_fuel_type(self, fuel_matrix):
        """Test getting a fuel type by ID."""
        fuel = fuel_matrix.get_fuel_type("HFO")

        assert fuel is not None
        assert fuel.fuel_id == "HFO"
        assert fuel.fuel_name == "Heavy Fuel Oil"
        assert fuel.category == FuelCategory.RESIDUAL

    def test_get_fuel_type_unknown(self, fuel_matrix):
        """Test getting unknown fuel type returns None."""
        fuel = fuel_matrix.get_fuel_type("UNKNOWN")
        assert fuel is None


# =============================================================================
# Compatibility Rule Registration Tests
# =============================================================================

@pytest.mark.unit
class TestCompatibilityRuleRegistration:
    """Tests for compatibility rule registration."""

    def test_register_compatibility_rule(self, fuel_matrix, custom_fuel_type, custom_compatibility_rule):
        """Test registering a new compatibility rule."""
        # First register the fuel type
        fuel_matrix.register_fuel_type(custom_fuel_type)

        # Then register the rule
        fuel_matrix.register_compatibility_rule(custom_compatibility_rule)

        # Check rule is applied
        result = fuel_matrix.check_compatibility("CUSTOM_FUEL", "ULSD")
        assert result.compatibility_level == CompatibilityLevel.FULLY_COMPATIBLE
        assert result.rule_id == "CUSTOM_R001"

    def test_rule_applies_bidirectionally(self, fuel_matrix, custom_fuel_type, custom_compatibility_rule):
        """Test that rules apply in both directions."""
        fuel_matrix.register_fuel_type(custom_fuel_type)
        fuel_matrix.register_compatibility_rule(custom_compatibility_rule)

        # Check both directions
        result_ab = fuel_matrix.check_compatibility("CUSTOM_FUEL", "ULSD")
        result_ba = fuel_matrix.check_compatibility("ULSD", "CUSTOM_FUEL")

        assert result_ab.compatibility_level == result_ba.compatibility_level
        assert result_ab.rule_id == result_ba.rule_id


# =============================================================================
# Provenance Hash Tests
# =============================================================================

@pytest.mark.unit
class TestProvenanceHash:
    """Tests for provenance hash generation."""

    def test_provenance_hash_generated(self, fuel_matrix):
        """Test that provenance hash is generated for each check."""
        result = fuel_matrix.check_compatibility("HFO", "VLSFO")

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256 hex length

    def test_provenance_hash_unique_per_check(self, fuel_matrix):
        """Test that provenance hash is unique per check (includes timestamp)."""
        result1 = fuel_matrix.check_compatibility("HFO", "VLSFO")
        result2 = fuel_matrix.check_compatibility("HFO", "VLSFO")

        # Hashes should be different due to timestamp in check_id
        assert result1.check_id != result2.check_id
        assert result1.provenance_hash != result2.provenance_hash

    def test_check_id_format(self, fuel_matrix):
        """Test check ID has correct format."""
        result = fuel_matrix.check_compatibility("MGO", "ULSD")

        assert result.check_id is not None
        assert len(result.check_id) == 16  # First 16 chars of SHA-256


# =============================================================================
# FuelType Model Tests
# =============================================================================

@pytest.mark.unit
class TestFuelTypeModel:
    """Tests for FuelType Pydantic model."""

    def test_fuel_type_creation(self):
        """Test FuelType creation with all fields."""
        fuel = FuelType(
            fuel_id="TEST",
            fuel_name="Test Fuel",
            category=FuelCategory.DISTILLATE,
            flash_point_min_c=50,
            flash_point_max_c=100,
            density_min_kg_m3=850,
            density_max_kg_m3=900,
            viscosity_min_cst=2,
            viscosity_max_cst=10,
            sulfur_max_pct=0.5,
            water_sensitive=True,
            oxidation_sensitive=False,
            wax_content=True,
            asphaltene_content=False,
            astm_standard="ASTM TEST",
            iso_standard="ISO TEST"
        )

        assert fuel.fuel_id == "TEST"
        assert fuel.category == FuelCategory.DISTILLATE
        assert fuel.water_sensitive is True

    def test_fuel_type_frozen(self):
        """Test FuelType is immutable (frozen)."""
        fuel = FuelType(
            fuel_id="FROZEN",
            fuel_name="Frozen Fuel",
            category=FuelCategory.GASOLINE
        )

        with pytest.raises(Exception):  # ValidationError or similar
            fuel.fuel_id = "MODIFIED"


# =============================================================================
# CompatibilityRule Model Tests
# =============================================================================

@pytest.mark.unit
class TestCompatibilityRuleModel:
    """Tests for CompatibilityRule Pydantic model."""

    def test_compatibility_rule_creation(self):
        """Test CompatibilityRule creation."""
        rule = CompatibilityRule(
            rule_id="TEST_R001",
            fuel_a="HFO",
            fuel_b="MGO",
            compatibility=CompatibilityLevel.CONDITIONALLY_COMPATIBLE,
            conditions=["Test condition 1", "Test condition 2"],
            max_blend_ratio=0.25,
            contamination_risk=ContaminationRisk.MEDIUM,
            segregation_type=SegregationType.TIME_SEPARATION,
            technical_reason="Test technical reason",
            reference_standard="Test Standard"
        )

        assert rule.rule_id == "TEST_R001"
        assert rule.compatibility == CompatibilityLevel.CONDITIONALLY_COMPATIBLE
        assert len(rule.conditions) == 2
        assert rule.max_blend_ratio == 0.25

    def test_compatibility_rule_defaults(self):
        """Test CompatibilityRule default values."""
        rule = CompatibilityRule(
            rule_id="MINIMAL",
            fuel_a="A",
            fuel_b="B",
            compatibility=CompatibilityLevel.FULLY_COMPATIBLE
        )

        assert rule.conditions == []
        assert rule.max_blend_ratio is None
        assert rule.contamination_risk == ContaminationRisk.NONE
        assert rule.segregation_type == SegregationType.NONE
        assert rule.technical_reason == ""


# =============================================================================
# CompatibilityCheckResult Model Tests
# =============================================================================

@pytest.mark.unit
class TestCompatibilityCheckResultModel:
    """Tests for CompatibilityCheckResult model."""

    def test_result_contains_all_fields(self, fuel_matrix):
        """Test result contains all expected fields."""
        result = fuel_matrix.check_compatibility("HFO", "VLSFO")

        # Check all required fields are present
        assert hasattr(result, "check_id")
        assert hasattr(result, "timestamp")
        assert hasattr(result, "fuel_a")
        assert hasattr(result, "fuel_b")
        assert hasattr(result, "can_co_mingle")
        assert hasattr(result, "compatibility_level")
        assert hasattr(result, "contamination_risk")
        assert hasattr(result, "contamination_score")
        assert hasattr(result, "conditions")
        assert hasattr(result, "segregation_required")
        assert hasattr(result, "max_blend_ratio")
        assert hasattr(result, "reason")
        assert hasattr(result, "technical_details")
        assert hasattr(result, "rule_id")
        assert hasattr(result, "provenance_hash")

    def test_result_timestamp_is_utc(self, fuel_matrix):
        """Test result timestamp is in UTC."""
        result = fuel_matrix.check_compatibility("MGO", "ULSD")

        assert result.timestamp.tzinfo is not None
        assert result.timestamp.tzinfo == timezone.utc


# =============================================================================
# Enum Tests
# =============================================================================

@pytest.mark.unit
class TestEnums:
    """Tests for enum values."""

    def test_fuel_category_values(self):
        """Test FuelCategory enum values."""
        assert FuelCategory.DISTILLATE.value == "distillate"
        assert FuelCategory.RESIDUAL.value == "residual"
        assert FuelCategory.GASOLINE.value == "gasoline"
        assert FuelCategory.BIOFUEL.value == "biofuel"
        assert FuelCategory.LNG.value == "lng"

    def test_compatibility_level_values(self):
        """Test CompatibilityLevel enum values."""
        assert CompatibilityLevel.FULLY_COMPATIBLE.value == "fully_compatible"
        assert CompatibilityLevel.CONDITIONALLY_COMPATIBLE.value == "conditional"
        assert CompatibilityLevel.INCOMPATIBLE.value == "incompatible"
        assert CompatibilityLevel.UNKNOWN.value == "unknown"

    def test_contamination_risk_values(self):
        """Test ContaminationRisk enum values."""
        assert ContaminationRisk.NONE.value == "none"
        assert ContaminationRisk.LOW.value == "low"
        assert ContaminationRisk.MEDIUM.value == "medium"
        assert ContaminationRisk.HIGH.value == "high"
        assert ContaminationRisk.CRITICAL.value == "critical"

    def test_segregation_type_values(self):
        """Test SegregationType enum values."""
        assert SegregationType.NONE.value == "none"
        assert SegregationType.DEDICATED_TANK.value == "dedicated"
        assert SegregationType.PHYSICAL_BARRIER.value == "barrier"
        assert SegregationType.TIME_SEPARATION.value == "time"
        assert SegregationType.FLUSH_REQUIRED.value == "flush"
