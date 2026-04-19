# -*- coding: utf-8 -*-
"""
GL-008 TRAPCATCHER - Trap Types Module Tests

Unit tests for trap_types.py module including trap type classification,
selection criteria, and application guidance.

Target Coverage: 85%+
"""

import pytest
from typing import List

from greenlang.agents.process_heat.gl_008_steam_trap_monitor.config import (
    TrapType,
    TrapApplication,
)
from greenlang.agents.process_heat.gl_008_steam_trap_monitor.trap_types import (
    TrapCharacteristics,
    TrapSelectionCriteria,
    TrapRecommendation,
    TrapApplicationGuide,
    TrapTypeClassifier,
    TRAP_CHARACTERISTICS,
    APPLICATION_GUIDES,
)


class TestTrapCharacteristics:
    """Tests for TrapCharacteristics dataclass."""

    def test_float_thermostatic_characteristics(self):
        """Test float thermostatic trap characteristics."""
        char = TRAP_CHARACTERISTICS[TrapType.FLOAT_THERMOSTATIC]

        assert char.trap_type == TrapType.FLOAT_THERMOSTATIC
        assert char.discharge_pattern == "continuous"
        assert char.subcooling_f == 0.0  # Immediate discharge
        assert char.air_venting == "excellent"
        assert char.max_pressure_psig == 465
        assert "continuous" in char.advantages[0].lower()

    def test_inverted_bucket_characteristics(self):
        """Test inverted bucket trap characteristics."""
        char = TRAP_CHARACTERISTICS[TrapType.INVERTED_BUCKET]

        assert char.trap_type == TrapType.INVERTED_BUCKET
        assert char.discharge_pattern == "intermittent"
        assert char.subcooling_f == 3.0  # Slight subcooling
        assert char.typical_service_life_years == 15  # Longest
        assert any("robust" in adv.lower() for adv in char.advantages)

    def test_thermostatic_characteristics(self):
        """Test balanced pressure thermostatic characteristics."""
        char = TRAP_CHARACTERISTICS[TrapType.THERMOSTATIC]

        assert char.trap_type == TrapType.THERMOSTATIC
        assert char.discharge_pattern == "thermostatic"
        assert char.subcooling_f == 20.0  # Significant subcooling
        assert char.air_venting == "excellent"

    def test_thermodynamic_characteristics(self):
        """Test thermodynamic disc trap characteristics."""
        char = TRAP_CHARACTERISTICS[TrapType.THERMODYNAMIC]

        assert char.trap_type == TrapType.THERMODYNAMIC
        assert char.discharge_pattern == "intermittent"
        assert char.air_venting == "fair"  # Poor air venting
        assert any("noisy" in dis.lower() for dis in char.disadvantages)

    def test_bimetallic_characteristics(self):
        """Test bimetallic trap characteristics."""
        char = TRAP_CHARACTERISTICS[TrapType.BIMETALLIC]

        assert char.trap_type == TrapType.BIMETALLIC
        assert char.subcooling_f >= 30  # High subcooling
        assert any("freez" in adv.lower() for adv in char.advantages)

    def test_all_trap_types_have_characteristics(self):
        """Verify all trap types have defined characteristics."""
        major_types = [
            TrapType.FLOAT_THERMOSTATIC,
            TrapType.INVERTED_BUCKET,
            TrapType.THERMOSTATIC,
            TrapType.THERMODYNAMIC,
            TrapType.BIMETALLIC,
        ]

        for trap_type in major_types:
            assert trap_type in TRAP_CHARACTERISTICS, f"Missing: {trap_type}"


class TestApplicationGuides:
    """Tests for application guides."""

    def test_drip_leg_guide(self):
        """Test drip leg application guide."""
        guide = APPLICATION_GUIDES[TrapApplication.DRIP_LEG]

        assert guide.application == TrapApplication.DRIP_LEG
        assert TrapType.INVERTED_BUCKET in guide.recommended_trap_types
        assert TrapType.FLOAT_THERMOSTATIC in guide.avoid_trap_types
        assert guide.typical_sizing_factor == 3.0  # Higher for startup

    def test_heat_exchanger_guide(self):
        """Test heat exchanger application guide."""
        guide = APPLICATION_GUIDES[TrapApplication.HEAT_EXCHANGER]

        assert guide.application == TrapApplication.HEAT_EXCHANGER
        assert TrapType.FLOAT_THERMOSTATIC in guide.recommended_trap_types
        assert TrapType.THERMOSTATIC in guide.avoid_trap_types
        assert "immediate discharge" in " ".join(guide.primary_requirements).lower()

    def test_process_guide(self):
        """Test process application guide."""
        guide = APPLICATION_GUIDES[TrapApplication.PROCESS]

        assert TrapType.FLOAT_THERMOSTATIC in guide.recommended_trap_types
        assert TrapType.INVERTED_BUCKET in guide.recommended_trap_types

    def test_tracer_guide(self):
        """Test steam tracer application guide."""
        guide = APPLICATION_GUIDES[TrapApplication.TRACER]

        assert TrapType.THERMOSTATIC in guide.recommended_trap_types
        assert TrapType.THERMODYNAMIC in guide.recommended_trap_types
        assert "compact" in " ".join(guide.primary_requirements).lower()

    def test_unit_heater_guide(self):
        """Test unit heater application guide."""
        guide = APPLICATION_GUIDES[TrapApplication.UNIT_HEATER]

        assert TrapType.THERMOSTATIC in guide.recommended_trap_types
        assert TrapType.THERMODYNAMIC in guide.avoid_trap_types
        assert "air venting" in " ".join(guide.primary_requirements).lower()

    def test_autoclave_guide(self):
        """Test autoclave application guide."""
        guide = APPLICATION_GUIDES[TrapApplication.AUTOCLAVE]

        assert guide.typical_sizing_factor == 3.0  # High startup loads
        assert "air removal" in " ".join(guide.primary_requirements).lower()


class TestTrapSelectionCriteria:
    """Tests for TrapSelectionCriteria model."""

    def test_minimal_criteria(self):
        """Test creating with minimal required fields."""
        criteria = TrapSelectionCriteria(
            application=TrapApplication.DRIP_LEG,
            steam_pressure_psig=150.0,
            condensate_load_lb_hr=100.0,
        )

        assert criteria.application == TrapApplication.DRIP_LEG
        assert criteria.steam_pressure_psig == 150.0
        assert criteria.condensate_load_lb_hr == 100.0

    def test_full_criteria(self):
        """Test creating with all fields."""
        criteria = TrapSelectionCriteria(
            application=TrapApplication.HEAT_EXCHANGER,
            steam_pressure_psig=200.0,
            back_pressure_psig=15.0,
            condensate_load_lb_hr=500.0,
            require_immediate_discharge=True,
            require_air_venting=True,
            waterhammer_risk=False,
            superheated_steam=False,
            modulating_load=True,
            outdoor_installation=False,
            dirt_in_system=False,
            max_subcooling_acceptable_f=10.0,
            prefer_low_maintenance=True,
            budget_constraint="premium",
        )

        assert criteria.require_immediate_discharge is True
        assert criteria.modulating_load is True
        assert criteria.max_subcooling_acceptable_f == 10.0

    def test_default_values(self):
        """Test default values are set correctly."""
        criteria = TrapSelectionCriteria(
            application=TrapApplication.PROCESS,
            steam_pressure_psig=100.0,
            condensate_load_lb_hr=200.0,
        )

        assert criteria.back_pressure_psig == 0.0
        assert criteria.require_immediate_discharge is False
        assert criteria.require_air_venting is True
        assert criteria.max_subcooling_acceptable_f == 50.0

    def test_invalid_pressure_rejected(self):
        """Test negative pressure is rejected."""
        with pytest.raises(ValueError):
            TrapSelectionCriteria(
                application=TrapApplication.DRIP_LEG,
                steam_pressure_psig=-10.0,
                condensate_load_lb_hr=100.0,
            )

    def test_invalid_load_rejected(self):
        """Test zero/negative load is rejected."""
        with pytest.raises(ValueError):
            TrapSelectionCriteria(
                application=TrapApplication.DRIP_LEG,
                steam_pressure_psig=150.0,
                condensate_load_lb_hr=0.0,
            )


class TestTrapTypeClassifier:
    """Tests for TrapTypeClassifier."""

    @pytest.fixture
    def classifier(self) -> TrapTypeClassifier:
        """Create classifier instance."""
        return TrapTypeClassifier()

    def test_initialization(self, classifier: TrapTypeClassifier):
        """Test classifier initializes correctly."""
        assert classifier._characteristics is not None
        assert classifier._application_guides is not None
        assert classifier._selection_count == 0

    def test_get_trap_characteristics(self, classifier: TrapTypeClassifier):
        """Test getting trap characteristics."""
        char = classifier.get_trap_characteristics(TrapType.FLOAT_THERMOSTATIC)

        assert char is not None
        assert char.trap_type == TrapType.FLOAT_THERMOSTATIC

    def test_get_trap_characteristics_invalid(self, classifier: TrapTypeClassifier):
        """Test getting characteristics for invalid type returns None."""
        # Create a mock invalid type by using None lookup
        result = classifier.get_trap_characteristics(None)
        assert result is None

    def test_get_application_guide(self, classifier: TrapTypeClassifier):
        """Test getting application guide."""
        guide = classifier.get_application_guide(TrapApplication.HEAT_EXCHANGER)

        assert guide is not None
        assert guide.application == TrapApplication.HEAT_EXCHANGER

    def test_select_trap_for_drip_leg(self, classifier: TrapTypeClassifier):
        """Test trap selection for drip leg application."""
        criteria = TrapSelectionCriteria(
            application=TrapApplication.DRIP_LEG,
            steam_pressure_psig=150.0,
            condensate_load_lb_hr=50.0,
            waterhammer_risk=True,
        )

        recommendations = classifier.select_trap_type(criteria)

        assert len(recommendations) > 0
        # Inverted bucket should be top for drip leg
        assert recommendations[0].trap_type == TrapType.INVERTED_BUCKET
        assert recommendations[0].suitability_score > 80

    def test_select_trap_for_heat_exchanger(self, classifier: TrapTypeClassifier):
        """Test trap selection for heat exchanger application."""
        criteria = TrapSelectionCriteria(
            application=TrapApplication.HEAT_EXCHANGER,
            steam_pressure_psig=100.0,
            condensate_load_lb_hr=500.0,
            require_immediate_discharge=True,
            modulating_load=True,
        )

        recommendations = classifier.select_trap_type(criteria)

        assert len(recommendations) > 0
        # Float thermostatic should be top for heat exchanger
        assert recommendations[0].trap_type == TrapType.FLOAT_THERMOSTATIC
        assert "Recommended for" in " ".join(recommendations[0].reasons_for)

    def test_select_trap_for_tracer(self, classifier: TrapTypeClassifier):
        """Test trap selection for steam tracer application."""
        criteria = TrapSelectionCriteria(
            application=TrapApplication.TRACER,
            steam_pressure_psig=100.0,
            condensate_load_lb_hr=10.0,
            outdoor_installation=True,
        )

        recommendations = classifier.select_trap_type(criteria)

        assert len(recommendations) > 0
        # Should prefer thermostatic or thermodynamic for tracers
        top_types = [r.trap_type for r in recommendations[:3]]
        assert TrapType.THERMOSTATIC in top_types or TrapType.THERMODYNAMIC in top_types

    def test_select_trap_pressure_constraint(self, classifier: TrapTypeClassifier):
        """Test trap selection excludes over-pressure traps."""
        criteria = TrapSelectionCriteria(
            application=TrapApplication.PROCESS,
            steam_pressure_psig=550.0,  # Very high pressure
            condensate_load_lb_hr=200.0,
        )

        recommendations = classifier.select_trap_type(criteria)

        # Float thermostatic (465 psig max) should be excluded
        ft_rec = next(
            (r for r in recommendations if r.trap_type == TrapType.FLOAT_THERMOSTATIC),
            None
        )
        # Either not in list or score is 0
        assert ft_rec is None or ft_rec.suitability_score == 0

    def test_select_trap_capacity_constraint(self, classifier: TrapTypeClassifier):
        """Test trap selection considers capacity constraints."""
        criteria = TrapSelectionCriteria(
            application=TrapApplication.REBOILER,
            steam_pressure_psig=150.0,
            condensate_load_lb_hr=60000.0,  # Very high load
        )

        recommendations = classifier.select_trap_type(criteria)

        # Thermodynamic (2500 lb/hr max) should be excluded
        td_rec = next(
            (r for r in recommendations if r.trap_type == TrapType.THERMODYNAMIC),
            None
        )
        assert td_rec is None or td_rec.suitability_score == 0

    def test_select_trap_superheated_steam(self, classifier: TrapTypeClassifier):
        """Test trap selection for superheated steam."""
        criteria = TrapSelectionCriteria(
            application=TrapApplication.DRIP_LEG,
            steam_pressure_psig=150.0,
            condensate_load_lb_hr=100.0,
            superheated_steam=True,
        )

        recommendations = classifier.select_trap_type(criteria)

        # Inverted bucket should score well for superheated
        ib_rec = next(
            r for r in recommendations if r.trap_type == TrapType.INVERTED_BUCKET
        )
        assert "superheated" in " ".join(ib_rec.reasons_for).lower()

    def test_select_trap_dirty_system(self, classifier: TrapTypeClassifier):
        """Test trap selection for dirty system."""
        criteria = TrapSelectionCriteria(
            application=TrapApplication.PROCESS,
            steam_pressure_psig=150.0,
            condensate_load_lb_hr=200.0,
            dirt_in_system=True,
        )

        recommendations = classifier.select_trap_type(criteria)

        # Inverted bucket is dirt tolerant
        ib_rec = next(
            r for r in recommendations if r.trap_type == TrapType.INVERTED_BUCKET
        )
        assert "dirt" in " ".join(ib_rec.reasons_for).lower()

    def test_select_trap_air_venting_requirement(self, classifier: TrapTypeClassifier):
        """Test trap selection with air venting requirement."""
        criteria = TrapSelectionCriteria(
            application=TrapApplication.AUTOCLAVE,
            steam_pressure_psig=50.0,
            condensate_load_lb_hr=100.0,
            require_air_venting=True,
        )

        recommendations = classifier.select_trap_type(criteria)

        # Check that excellent air venters score well
        ft_rec = next(
            r for r in recommendations if r.trap_type == TrapType.FLOAT_THERMOSTATIC
        )
        assert "air venting" in " ".join(ft_rec.reasons_for).lower()

    def test_select_trap_subcooling_constraint(self, classifier: TrapTypeClassifier):
        """Test trap selection with subcooling constraint."""
        criteria = TrapSelectionCriteria(
            application=TrapApplication.PROCESS,
            steam_pressure_psig=150.0,
            condensate_load_lb_hr=200.0,
            max_subcooling_acceptable_f=10.0,  # Tight constraint
        )

        recommendations = classifier.select_trap_type(criteria)

        # Bimetallic (40F subcooling) should score poorly
        bi_rec = next(
            r for r in recommendations if r.trap_type == TrapType.BIMETALLIC
        )
        assert "subcooling" in " ".join(bi_rec.reasons_against).lower()

    def test_recommendations_ranked_correctly(self, classifier: TrapTypeClassifier):
        """Test recommendations are ranked by score."""
        criteria = TrapSelectionCriteria(
            application=TrapApplication.DRIP_LEG,
            steam_pressure_psig=150.0,
            condensate_load_lb_hr=100.0,
        )

        recommendations = classifier.select_trap_type(criteria)

        # Check scores are descending
        for i in range(len(recommendations) - 1):
            assert recommendations[i].suitability_score >= recommendations[i + 1].suitability_score

        # Check rankings
        for i, rec in enumerate(recommendations):
            assert rec.ranking == i + 1

    def test_sizing_guidance_generated(self, classifier: TrapTypeClassifier):
        """Test sizing guidance is generated."""
        criteria = TrapSelectionCriteria(
            application=TrapApplication.HEAT_EXCHANGER,
            steam_pressure_psig=150.0,
            condensate_load_lb_hr=500.0,
        )

        recommendations = classifier.select_trap_type(criteria)

        assert recommendations[0].sizing_recommendation is not None
        assert "lb/hr" in recommendations[0].sizing_recommendation

    def test_selection_count_increments(self, classifier: TrapTypeClassifier):
        """Test selection count increments correctly."""
        assert classifier.selection_count == 0

        criteria = TrapSelectionCriteria(
            application=TrapApplication.DRIP_LEG,
            steam_pressure_psig=150.0,
            condensate_load_lb_hr=100.0,
        )

        classifier.select_trap_type(criteria)
        assert classifier.selection_count == 1

        classifier.select_trap_type(criteria)
        assert classifier.selection_count == 2


class TestValidateTrapForApplication:
    """Tests for trap validation."""

    @pytest.fixture
    def classifier(self) -> TrapTypeClassifier:
        """Create classifier instance."""
        return TrapTypeClassifier()

    def test_valid_trap_application(self, classifier: TrapTypeClassifier):
        """Test valid trap for application."""
        is_valid, issues = classifier.validate_trap_for_application(
            trap_type=TrapType.INVERTED_BUCKET,
            application=TrapApplication.DRIP_LEG,
            pressure_psig=150.0,
            load_lb_hr=100.0,
        )

        assert is_valid is True
        assert len(issues) == 0

    def test_invalid_pressure_rating(self, classifier: TrapTypeClassifier):
        """Test invalid pressure rating detected."""
        is_valid, issues = classifier.validate_trap_for_application(
            trap_type=TrapType.FLOAT_THERMOSTATIC,
            application=TrapApplication.PROCESS,
            pressure_psig=500.0,  # Exceeds 465 max
            load_lb_hr=100.0,
        )

        assert is_valid is False
        assert any("pressure" in issue.lower() for issue in issues)

    def test_invalid_capacity(self, classifier: TrapTypeClassifier):
        """Test invalid capacity detected."""
        is_valid, issues = classifier.validate_trap_for_application(
            trap_type=TrapType.THERMODYNAMIC,
            application=TrapApplication.PROCESS,
            pressure_psig=150.0,
            load_lb_hr=5000.0,  # Exceeds 2500 max
        )

        assert is_valid is False
        assert any("capacity" in issue.lower() for issue in issues)

    def test_contraindicated_application(self, classifier: TrapTypeClassifier):
        """Test contraindicated application detected."""
        is_valid, issues = classifier.validate_trap_for_application(
            trap_type=TrapType.FLOAT_THERMOSTATIC,
            application=TrapApplication.DRIP_LEG,  # Contraindicated
            pressure_psig=150.0,
            load_lb_hr=100.0,
        )

        assert is_valid is False
        assert any("not recommended" in issue.lower() for issue in issues)


class TestASMEB1634Compliance:
    """Tests for ASME B16.34 pressure rating compliance."""

    @pytest.fixture
    def classifier(self) -> TrapTypeClassifier:
        """Create classifier instance."""
        return TrapTypeClassifier()

    def test_class_150_compliant(self, classifier: TrapTypeClassifier):
        """Test Class 150 compliance at 100F."""
        is_compliant, message = classifier.check_asme_b16_34_compliance(
            pressure_psig=250.0,
            temperature_f=100.0,
            trap_rating_class=150,
        )

        assert is_compliant is True
        assert "COMPLIANT" in message

    def test_class_150_non_compliant(self, classifier: TrapTypeClassifier):
        """Test Class 150 non-compliance at high pressure."""
        is_compliant, message = classifier.check_asme_b16_34_compliance(
            pressure_psig=300.0,  # Exceeds rating
            temperature_f=100.0,
            trap_rating_class=150,
        )

        assert is_compliant is False
        assert "EXCEEDS" in message

    def test_class_300_compliant(self, classifier: TrapTypeClassifier):
        """Test Class 300 compliance."""
        is_compliant, message = classifier.check_asme_b16_34_compliance(
            pressure_psig=500.0,
            temperature_f=300.0,
            trap_rating_class=300,
        )

        assert is_compliant is True

    def test_class_600_compliant(self, classifier: TrapTypeClassifier):
        """Test Class 600 compliance at high pressure."""
        is_compliant, message = classifier.check_asme_b16_34_compliance(
            pressure_psig=1000.0,
            temperature_f=400.0,
            trap_rating_class=600,
        )

        assert is_compliant is True

    def test_unknown_class(self, classifier: TrapTypeClassifier):
        """Test unknown class rating."""
        is_compliant, message = classifier.check_asme_b16_34_compliance(
            pressure_psig=100.0,
            temperature_f=100.0,
            trap_rating_class=999,  # Invalid
        )

        assert is_compliant is False
        assert "Unknown" in message

    @pytest.mark.parametrize("temperature,expected_rating", [
        (100, 285),
        (200, 260),
        (400, 200),
        (600, 140),
    ])
    def test_class_150_temperature_derating(
        self,
        classifier: TrapTypeClassifier,
        temperature: float,
        expected_rating: float,
    ):
        """Test Class 150 derating at various temperatures."""
        # Test at 5 psig below rating - should pass
        is_compliant, _ = classifier.check_asme_b16_34_compliance(
            pressure_psig=expected_rating - 5,
            temperature_f=temperature,
            trap_rating_class=150,
        )
        assert is_compliant is True

        # Test at 5 psig above rating - should fail
        is_compliant, _ = classifier.check_asme_b16_34_compliance(
            pressure_psig=expected_rating + 5,
            temperature_f=temperature,
            trap_rating_class=150,
        )
        assert is_compliant is False


class TestTrapTypeClassifierEdgeCases:
    """Edge case tests for TrapTypeClassifier."""

    @pytest.fixture
    def classifier(self) -> TrapTypeClassifier:
        """Create classifier instance."""
        return TrapTypeClassifier()

    def test_very_low_pressure(self, classifier: TrapTypeClassifier):
        """Test selection at very low pressure."""
        criteria = TrapSelectionCriteria(
            application=TrapApplication.UNIT_HEATER,
            steam_pressure_psig=5.0,  # Very low
            condensate_load_lb_hr=10.0,
        )

        recommendations = classifier.select_trap_type(criteria)
        assert len(recommendations) > 0

    def test_very_high_load(self, classifier: TrapTypeClassifier):
        """Test selection at very high load."""
        criteria = TrapSelectionCriteria(
            application=TrapApplication.REBOILER,
            steam_pressure_psig=150.0,
            condensate_load_lb_hr=40000.0,  # Very high
        )

        recommendations = classifier.select_trap_type(criteria)
        # Should recommend inverted bucket (highest capacity)
        ib_rec = next(
            (r for r in recommendations if r.trap_type == TrapType.INVERTED_BUCKET),
            None
        )
        assert ib_rec is not None
        assert ib_rec.suitability_score > 0

    def test_all_constraints_enabled(self, classifier: TrapTypeClassifier):
        """Test selection with all constraints enabled."""
        criteria = TrapSelectionCriteria(
            application=TrapApplication.PROCESS,
            steam_pressure_psig=100.0,
            condensate_load_lb_hr=200.0,
            require_immediate_discharge=True,
            require_air_venting=True,
            waterhammer_risk=True,
            superheated_steam=True,
            modulating_load=True,
            outdoor_installation=True,
            dirt_in_system=True,
            max_subcooling_acceptable_f=5.0,
            prefer_low_maintenance=True,
        )

        recommendations = classifier.select_trap_type(criteria)
        # Should still produce recommendations
        assert len(recommendations) > 0

    def test_conflicting_requirements(self, classifier: TrapTypeClassifier):
        """Test selection with conflicting requirements."""
        # Immediate discharge conflicts with subcooling acceptance
        criteria = TrapSelectionCriteria(
            application=TrapApplication.TRACER,
            steam_pressure_psig=100.0,
            condensate_load_lb_hr=20.0,
            require_immediate_discharge=True,  # Conflicts with tracer
            max_subcooling_acceptable_f=5.0,   # Very tight
        )

        recommendations = classifier.select_trap_type(criteria)
        # Should still produce recommendations
        assert len(recommendations) > 0
        # Float thermostatic should score well for immediate discharge
        ft_rec = next(
            r for r in recommendations if r.trap_type == TrapType.FLOAT_THERMOSTATIC
        )
        assert ft_rec.suitability_score > 50
