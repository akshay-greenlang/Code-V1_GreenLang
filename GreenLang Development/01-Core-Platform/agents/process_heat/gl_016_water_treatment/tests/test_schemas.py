"""
GL-016 WATERGUARD Agent - Schema Tests

Unit tests for Pydantic schema validation covering:
- Input validation (bounds, types, constraints)
- Output model integrity
- Enum handling
- Edge cases and error conditions

Author: GL-TestEngineer
Target Coverage: 85%+
"""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError
import uuid

from greenlang.agents.process_heat.gl_016_water_treatment.schemas import (
    # Enums
    WaterQualityStatus,
    TreatmentProgram,
    BoilerPressureClass,
    BlowdownType,
    ChemicalType,
    CorrosionMechanism,
    # Base models
    WaterSampleInput,
    WaterQualityResult,
    # Boiler water
    BoilerWaterInput,
    BoilerWaterLimits,
    BoilerWaterOutput,
    # Feedwater
    FeedwaterInput,
    FeedwaterLimits,
    FeedwaterOutput,
    # Condensate
    CondensateInput,
    CondensateLimits,
    CondensateOutput,
    # Blowdown
    BlowdownInput,
    BlowdownOutput,
    # Chemical dosing
    ChemicalDosingInput,
    ChemicalDosingOutput,
    # Deaeration
    DeaerationInput,
    DeaerationOutput,
    # Main
    WaterTreatmentInput,
    WaterTreatmentOutput,
)


class TestEnumerations:
    """Test enum definitions and values."""

    def test_water_quality_status_values(self):
        """Test WaterQualityStatus enum has all expected values."""
        assert WaterQualityStatus.EXCELLENT.value == "excellent"
        assert WaterQualityStatus.GOOD.value == "good"
        assert WaterQualityStatus.ACCEPTABLE.value == "acceptable"
        assert WaterQualityStatus.WARNING.value == "warning"
        assert WaterQualityStatus.CRITICAL.value == "critical"
        assert WaterQualityStatus.OUT_OF_SPEC.value == "out_of_spec"

    def test_treatment_program_values(self):
        """Test TreatmentProgram enum has all expected values."""
        assert TreatmentProgram.PHOSPHATE_PRECIPITATE.value == "phosphate_precipitate"
        assert TreatmentProgram.COORDINATED_PHOSPHATE.value == "coordinated_phosphate"
        assert TreatmentProgram.CONGRUENT_PHOSPHATE.value == "congruent_phosphate"
        assert TreatmentProgram.ALL_VOLATILE.value == "all_volatile"
        assert TreatmentProgram.OXYGENATED_TREATMENT.value == "oxygenated_treatment"

    def test_boiler_pressure_class_values(self):
        """Test BoilerPressureClass enum values."""
        assert BoilerPressureClass.LOW_PRESSURE.value == "low_pressure"
        assert BoilerPressureClass.MEDIUM_PRESSURE.value == "medium_pressure"
        assert BoilerPressureClass.HIGH_PRESSURE.value == "high_pressure"
        assert BoilerPressureClass.SUPERCRITICAL.value == "supercritical"

    def test_blowdown_type_values(self):
        """Test BlowdownType enum values."""
        assert BlowdownType.CONTINUOUS.value == "continuous"
        assert BlowdownType.INTERMITTENT.value == "intermittent"
        assert BlowdownType.SURFACE.value == "surface"
        assert BlowdownType.BOTTOM.value == "bottom"
        assert BlowdownType.COMBINED.value == "combined"

    def test_chemical_type_values(self):
        """Test ChemicalType enum has all treatment chemicals."""
        chemicals = [
            ChemicalType.PHOSPHATE,
            ChemicalType.OXYGEN_SCAVENGER,
            ChemicalType.AMINE,
            ChemicalType.SULFITE,
            ChemicalType.HYDRAZINE,
            ChemicalType.CARBOHYDRAZIDE,
            ChemicalType.MORPHOLINE,
            ChemicalType.CYCLOHEXYLAMINE,
        ]
        for chem in chemicals:
            assert chem.value is not None

    def test_corrosion_mechanism_values(self):
        """Test CorrosionMechanism enum values."""
        mechanisms = [
            CorrosionMechanism.OXYGEN_PITTING,
            CorrosionMechanism.CAUSTIC_EMBRITTLEMENT,
            CorrosionMechanism.CAUSTIC_GOUGING,
            CorrosionMechanism.HYDROGEN_DAMAGE,
            CorrosionMechanism.ACID_PHOSPHATE_CORROSION,
            CorrosionMechanism.FLOW_ACCELERATED,
            CorrosionMechanism.UNDER_DEPOSIT,
            CorrosionMechanism.CARBONIC_ACID,
        ]
        for mech in mechanisms:
            assert mech.value is not None


class TestWaterSampleInput:
    """Test WaterSampleInput base model."""

    def test_valid_sample_input(self):
        """Test valid sample input creation."""
        sample = WaterSampleInput(
            sample_point="boiler_drum",
            temperature_f=350.0,
        )
        assert sample.sample_point == "boiler_drum"
        assert sample.temperature_f == 350.0
        assert sample.sample_id is not None
        assert sample.timestamp is not None

    def test_sample_id_auto_generation(self):
        """Test sample_id is auto-generated if not provided."""
        sample1 = WaterSampleInput(sample_point="test")
        sample2 = WaterSampleInput(sample_point="test")
        # Each should have unique ID
        assert sample1.sample_id != sample2.sample_id

    def test_temperature_bounds(self):
        """Test temperature validation bounds (32-700 F)."""
        # Valid temperature
        sample = WaterSampleInput(sample_point="test", temperature_f=400.0)
        assert sample.temperature_f == 400.0

        # Below minimum
        with pytest.raises(ValidationError):
            WaterSampleInput(sample_point="test", temperature_f=20.0)

        # Above maximum
        with pytest.raises(ValidationError):
            WaterSampleInput(sample_point="test", temperature_f=800.0)


class TestWaterQualityResult:
    """Test WaterQualityResult model."""

    def test_valid_quality_result(self):
        """Test valid quality result creation."""
        result = WaterQualityResult(
            parameter="pH",
            value=10.5,
            unit="pH units",
            min_limit=9.5,
            max_limit=11.5,
            target_value=10.5,
            status=WaterQualityStatus.EXCELLENT,
            deviation_pct=0.0,
        )
        assert result.parameter == "pH"
        assert result.value == 10.5
        assert result.status == WaterQualityStatus.EXCELLENT

    def test_quality_result_optional_fields(self):
        """Test quality result with only required fields."""
        result = WaterQualityResult(
            parameter="Test",
            value=100.0,
            unit="ppm",
            status=WaterQualityStatus.GOOD,
        )
        assert result.min_limit is None
        assert result.max_limit is None
        assert result.target_value is None
        assert result.deviation_pct is None


class TestBoilerWaterInput:
    """Test BoilerWaterInput schema validation."""

    def test_valid_boiler_water_input(self, boiler_water_input_excellent):
        """Test valid boiler water input."""
        assert boiler_water_input_excellent.ph == 10.5
        assert boiler_water_input_excellent.operating_pressure_psig == 450.0

    def test_ph_bounds(self):
        """Test pH validation (0-14)."""
        # Valid pH
        bw = BoilerWaterInput(
            sample_point="drum",
            ph=10.5,
            specific_conductivity_umho=2000.0,
            operating_pressure_psig=450.0,
        )
        assert bw.ph == 10.5

        # pH below 0
        with pytest.raises(ValidationError):
            BoilerWaterInput(
                sample_point="drum",
                ph=-0.5,
                specific_conductivity_umho=2000.0,
                operating_pressure_psig=450.0,
            )

        # pH above 14
        with pytest.raises(ValidationError):
            BoilerWaterInput(
                sample_point="drum",
                ph=15.0,
                specific_conductivity_umho=2000.0,
                operating_pressure_psig=450.0,
            )

    def test_phosphate_bounds(self):
        """Test phosphate validation (0-200 ppm)."""
        # Valid phosphate
        bw = BoilerWaterInput(
            sample_point="drum",
            ph=10.5,
            phosphate_ppm=50.0,
            specific_conductivity_umho=2000.0,
            operating_pressure_psig=450.0,
        )
        assert bw.phosphate_ppm == 50.0

        # Phosphate above limit
        with pytest.raises(ValidationError):
            BoilerWaterInput(
                sample_point="drum",
                ph=10.5,
                phosphate_ppm=250.0,  # > 200
                specific_conductivity_umho=2000.0,
                operating_pressure_psig=450.0,
            )

    def test_conductivity_non_negative(self):
        """Test conductivity must be non-negative."""
        with pytest.raises(ValidationError):
            BoilerWaterInput(
                sample_point="drum",
                ph=10.5,
                specific_conductivity_umho=-100.0,
                operating_pressure_psig=450.0,
            )

    def test_operating_pressure_non_negative(self):
        """Test operating pressure must be non-negative."""
        with pytest.raises(ValidationError):
            BoilerWaterInput(
                sample_point="drum",
                ph=10.5,
                specific_conductivity_umho=2000.0,
                operating_pressure_psig=-10.0,
            )


class TestBoilerWaterLimits:
    """Test BoilerWaterLimits schema."""

    def test_valid_limits(self):
        """Test valid boiler water limits creation."""
        limits = BoilerWaterLimits(
            pressure_class=BoilerPressureClass.MEDIUM_PRESSURE,
            treatment_program=TreatmentProgram.COORDINATED_PHOSPHATE,
            ph_min=9.5,
            ph_max=10.5,
            phosphate_min_ppm=2.0,
            phosphate_max_ppm=12.0,
            conductivity_max_umho=5000.0,
            silica_max_ppm=30.0,
        )
        assert limits.ph_min == 9.5
        assert limits.ph_max == 10.5

    def test_limits_ph_validation(self):
        """Test pH limits bounds."""
        with pytest.raises(ValidationError):
            BoilerWaterLimits(
                pressure_class=BoilerPressureClass.MEDIUM_PRESSURE,
                treatment_program=TreatmentProgram.COORDINATED_PHOSPHATE,
                ph_min=15.0,  # > 14
                ph_max=10.5,
                phosphate_max_ppm=12.0,
                conductivity_max_umho=5000.0,
                silica_max_ppm=30.0,
            )


class TestBoilerWaterOutput:
    """Test BoilerWaterOutput schema."""

    def test_valid_output(self):
        """Test valid boiler water output creation."""
        output = BoilerWaterOutput(
            sample_id="BW-001",
            overall_status=WaterQualityStatus.GOOD,
            status_message="Good water chemistry",
            parameter_results=[],
            corrosion_risk_score=25.0,
            scaling_risk_score=15.0,
            deposition_risk_score=10.0,
        )
        assert output.overall_status == WaterQualityStatus.GOOD
        assert output.corrosion_risk_score == 25.0

    def test_output_risk_score_bounds(self):
        """Test risk score bounds (0-100)."""
        # Valid scores
        output = BoilerWaterOutput(
            sample_id="BW-001",
            overall_status=WaterQualityStatus.GOOD,
            status_message="Test",
            corrosion_risk_score=50.0,
            scaling_risk_score=50.0,
            deposition_risk_score=50.0,
        )
        assert output.corrosion_risk_score == 50.0

        # Above maximum
        with pytest.raises(ValidationError):
            BoilerWaterOutput(
                sample_id="BW-001",
                overall_status=WaterQualityStatus.GOOD,
                status_message="Test",
                corrosion_risk_score=150.0,  # > 100
            )


class TestFeedwaterInput:
    """Test FeedwaterInput schema validation."""

    def test_valid_feedwater_input(self, feedwater_input_excellent):
        """Test valid feedwater input."""
        assert feedwater_input_excellent.ph == 9.0
        assert feedwater_input_excellent.dissolved_oxygen_ppb == 3.0

    def test_dissolved_oxygen_non_negative(self):
        """Test dissolved oxygen must be non-negative."""
        with pytest.raises(ValidationError):
            FeedwaterInput(
                sample_point="da_outlet",
                ph=9.0,
                specific_conductivity_umho=0.5,
                dissolved_oxygen_ppb=-5.0,
            )

    def test_feedwater_temperature_bounds(self):
        """Test feedwater temperature bounds (100-500 F)."""
        # Valid temperature
        fw = FeedwaterInput(
            sample_point="da_outlet",
            ph=9.0,
            specific_conductivity_umho=0.5,
            dissolved_oxygen_ppb=5.0,
            temperature_f=227.0,
        )
        assert fw.temperature_f == 227.0

        # Below minimum
        with pytest.raises(ValidationError):
            FeedwaterInput(
                sample_point="da_outlet",
                ph=9.0,
                specific_conductivity_umho=0.5,
                dissolved_oxygen_ppb=5.0,
                temperature_f=50.0,
            )


class TestCondensateInput:
    """Test CondensateInput schema validation."""

    def test_valid_condensate_input(self, condensate_input_excellent):
        """Test valid condensate input."""
        assert condensate_input_excellent.ph == 8.7
        assert condensate_input_excellent.iron_ppb == 15.0

    def test_condensate_return_percentage_bounds(self):
        """Test condensate return percentage (0-100)."""
        # Valid percentage
        cond = CondensateInput(
            sample_point="main_return",
            ph=8.5,
            specific_conductivity_umho=1.0,
            iron_ppb=50.0,
            condensate_return_pct=80.0,
        )
        assert cond.condensate_return_pct == 80.0

        # Above 100%
        with pytest.raises(ValidationError):
            CondensateInput(
                sample_point="main_return",
                ph=8.5,
                specific_conductivity_umho=1.0,
                iron_ppb=50.0,
                condensate_return_pct=110.0,
            )

    def test_iron_required_field(self):
        """Test iron is a required field."""
        with pytest.raises(ValidationError):
            CondensateInput(
                sample_point="main_return",
                ph=8.5,
                specific_conductivity_umho=1.0,
                # Missing iron_ppb
            )


class TestBlowdownInput:
    """Test BlowdownInput schema validation."""

    def test_valid_blowdown_input(self, blowdown_input_optimize):
        """Test valid blowdown input."""
        assert blowdown_input_optimize.continuous_blowdown_rate_pct == 5.0
        assert blowdown_input_optimize.steam_flow_rate_lb_hr == 50000.0

    def test_blowdown_rate_bounds(self):
        """Test blowdown rate bounds (0-20%)."""
        # Valid rate
        bd = BlowdownInput(
            continuous_blowdown_rate_pct=5.0,
            boiler_tds_ppm=2000.0,
            feedwater_tds_ppm=50.0,
            tds_max_ppm=2500.0,
            steam_flow_rate_lb_hr=50000.0,
            operating_pressure_psig=450.0,
        )
        assert bd.continuous_blowdown_rate_pct == 5.0

        # Above maximum
        with pytest.raises(ValidationError):
            BlowdownInput(
                continuous_blowdown_rate_pct=25.0,  # > 20
                boiler_tds_ppm=2000.0,
                feedwater_tds_ppm=50.0,
                tds_max_ppm=2500.0,
                steam_flow_rate_lb_hr=50000.0,
                operating_pressure_psig=450.0,
            )

    def test_steam_flow_positive(self):
        """Test steam flow must be positive."""
        with pytest.raises(ValidationError):
            BlowdownInput(
                continuous_blowdown_rate_pct=5.0,
                boiler_tds_ppm=2000.0,
                feedwater_tds_ppm=50.0,
                tds_max_ppm=2500.0,
                steam_flow_rate_lb_hr=0.0,  # Must be > 0
                operating_pressure_psig=450.0,
            )


class TestBlowdownOutput:
    """Test BlowdownOutput schema."""

    def test_valid_blowdown_output(self):
        """Test valid blowdown output creation."""
        output = BlowdownOutput(
            current_cycles_of_concentration=10.0,
            current_blowdown_rate_pct=10.0,
            current_blowdown_flow_lb_hr=5000.0,
            optimal_cycles_of_concentration=10.0,
            optimal_blowdown_rate_pct=10.0,
            optimal_blowdown_flow_lb_hr=5000.0,
            energy_savings_mmbtu_yr=100.0,
            water_savings_kgal_yr=500.0,
            total_savings_usd_yr=5000.0,
        )
        assert output.current_cycles_of_concentration == 10.0

    def test_cycles_minimum(self):
        """Test cycles must be >= 1."""
        with pytest.raises(ValidationError):
            BlowdownOutput(
                current_cycles_of_concentration=0.5,  # < 1
                current_blowdown_rate_pct=10.0,
                current_blowdown_flow_lb_hr=5000.0,
                optimal_cycles_of_concentration=10.0,
                optimal_blowdown_rate_pct=10.0,
                optimal_blowdown_flow_lb_hr=5000.0,
            )


class TestChemicalDosingInput:
    """Test ChemicalDosingInput schema validation."""

    def test_valid_dosing_input(self, chemical_dosing_input_standard):
        """Test valid chemical dosing input."""
        assert chemical_dosing_input_standard.feedwater_flow_lb_hr == 55000.0
        assert chemical_dosing_input_standard.feedwater_do_ppb == 5.0

    def test_feedwater_flow_positive(self):
        """Test feedwater flow must be positive."""
        with pytest.raises(ValidationError):
            ChemicalDosingInput(
                feedwater_flow_lb_hr=0.0,  # Must be > 0
                feedwater_do_ppb=5.0,
                operating_pressure_psig=450.0,
            )

    def test_target_ph_bounds(self):
        """Test target condensate pH bounds (7-10)."""
        # Valid pH
        dosing = ChemicalDosingInput(
            feedwater_flow_lb_hr=50000.0,
            feedwater_do_ppb=5.0,
            operating_pressure_psig=450.0,
            target_condensate_ph=8.5,
        )
        assert dosing.target_condensate_ph == 8.5

        # Below minimum
        with pytest.raises(ValidationError):
            ChemicalDosingInput(
                feedwater_flow_lb_hr=50000.0,
                feedwater_do_ppb=5.0,
                operating_pressure_psig=450.0,
                target_condensate_ph=6.0,  # < 7
            )


class TestDeaerationInput:
    """Test DeaerationInput schema validation."""

    def test_valid_deaeration_input(self, deaeration_input_excellent):
        """Test valid deaeration input."""
        assert deaeration_input_excellent.deaerator_pressure_psig == 5.0
        assert deaeration_input_excellent.outlet_dissolved_oxygen_ppb == 3.0

    def test_deaerator_pressure_bounds(self):
        """Test deaerator pressure bounds (0-30 psig)."""
        # Valid pressure
        da = DeaerationInput(
            deaerator_pressure_psig=5.0,
            inlet_water_temperature_f=150.0,
            inlet_dissolved_oxygen_ppb=8000.0,
            outlet_dissolved_oxygen_ppb=5.0,
            total_flow_lb_hr=50000.0,
        )
        assert da.deaerator_pressure_psig == 5.0

        # Above maximum
        with pytest.raises(ValidationError):
            DeaerationInput(
                deaerator_pressure_psig=35.0,  # > 30
                inlet_water_temperature_f=150.0,
                inlet_dissolved_oxygen_ppb=8000.0,
                outlet_dissolved_oxygen_ppb=5.0,
                total_flow_lb_hr=50000.0,
            )

    def test_inlet_temperature_bounds(self):
        """Test inlet water temperature bounds (32-300 F)."""
        # Above maximum
        with pytest.raises(ValidationError):
            DeaerationInput(
                deaerator_pressure_psig=5.0,
                inlet_water_temperature_f=350.0,  # > 300
                inlet_dissolved_oxygen_ppb=8000.0,
                outlet_dissolved_oxygen_ppb=5.0,
                total_flow_lb_hr=50000.0,
            )


class TestWaterTreatmentInput:
    """Test WaterTreatmentInput schema validation."""

    def test_valid_treatment_input(self, water_treatment_input_complete):
        """Test valid water treatment input with all components."""
        assert water_treatment_input_complete.system_id == "TEST-WT-001"
        assert water_treatment_input_complete.boiler_water is not None
        assert water_treatment_input_complete.feedwater is not None

    def test_minimal_treatment_input(self, water_treatment_input_minimal):
        """Test minimal water treatment input."""
        assert water_treatment_input_minimal.system_id == "TEST-WT-MIN"
        assert water_treatment_input_minimal.boiler_water is None
        assert water_treatment_input_minimal.feedwater is None

    def test_system_id_required(self):
        """Test system_id is required."""
        with pytest.raises(ValidationError):
            WaterTreatmentInput(
                boiler_operating_pressure_psig=450.0,
                steam_flow_rate_lb_hr=50000.0,
                # Missing system_id
            )

    def test_steam_flow_positive(self):
        """Test steam flow must be positive."""
        with pytest.raises(ValidationError):
            WaterTreatmentInput(
                system_id="TEST",
                boiler_operating_pressure_psig=450.0,
                steam_flow_rate_lb_hr=0.0,  # Must be > 0
            )


class TestWaterTreatmentOutput:
    """Test WaterTreatmentOutput schema."""

    def test_valid_treatment_output(self):
        """Test valid water treatment output creation."""
        output = WaterTreatmentOutput(
            system_id="TEST-001",
            overall_status=WaterQualityStatus.GOOD,
            overall_score=85.0,
            corrosion_risk_score=25.0,
            scaling_risk_score=15.0,
            deposition_risk_score=10.0,
            carryover_risk_score=5.0,
        )
        assert output.overall_score == 85.0
        assert output.overall_status == WaterQualityStatus.GOOD

    def test_overall_score_bounds(self):
        """Test overall score bounds (0-100)."""
        # Valid score
        output = WaterTreatmentOutput(
            system_id="TEST-001",
            overall_status=WaterQualityStatus.GOOD,
            overall_score=85.0,
        )
        assert output.overall_score == 85.0

        # Above maximum
        with pytest.raises(ValidationError):
            WaterTreatmentOutput(
                system_id="TEST-001",
                overall_status=WaterQualityStatus.GOOD,
                overall_score=150.0,  # > 100
            )

        # Below minimum
        with pytest.raises(ValidationError):
            WaterTreatmentOutput(
                system_id="TEST-001",
                overall_status=WaterQualityStatus.GOOD,
                overall_score=-10.0,  # < 0
            )

    def test_output_with_all_analyses(self, boiler_water_input_excellent):
        """Test output can include all analysis types."""
        output = WaterTreatmentOutput(
            system_id="TEST-001",
            overall_status=WaterQualityStatus.EXCELLENT,
            overall_score=95.0,
            kpis={"cycles_of_concentration": 10.0},
            alerts=[{"level": "info", "message": "Test"}],
            recommendations=["Continue monitoring"],
            metadata={"treatment_program": "coordinated_phosphate"},
        )
        assert output.kpis["cycles_of_concentration"] == 10.0
        assert len(output.alerts) == 1
        assert len(output.recommendations) == 1


class TestSchemaSerializationDeserialization:
    """Test schema JSON serialization and deserialization."""

    def test_boiler_water_input_serialization(self, boiler_water_input_excellent):
        """Test BoilerWaterInput serialization to dict."""
        data = boiler_water_input_excellent.dict()
        assert data["ph"] == 10.5
        assert data["sample_point"] == "boiler_drum"

    def test_boiler_water_input_json(self, boiler_water_input_excellent):
        """Test BoilerWaterInput JSON serialization."""
        json_str = boiler_water_input_excellent.json()
        assert "ph" in json_str
        assert "10.5" in json_str

    def test_boiler_water_input_from_dict(self):
        """Test BoilerWaterInput creation from dict."""
        data = {
            "sample_point": "drum",
            "ph": 10.5,
            "specific_conductivity_umho": 2000.0,
            "operating_pressure_psig": 450.0,
        }
        bw = BoilerWaterInput(**data)
        assert bw.ph == 10.5

    def test_enum_serialization(self):
        """Test enum values are serialized correctly."""
        result = WaterQualityResult(
            parameter="pH",
            value=10.5,
            unit="pH units",
            status=WaterQualityStatus.EXCELLENT,
        )
        data = result.dict()
        # With use_enum_values=True, should be string
        assert data["status"] == "excellent"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_values_allowed(self):
        """Test zero is allowed where non-negative is required."""
        bw = BoilerWaterInput(
            sample_point="drum",
            ph=7.0,  # Neutral pH
            specific_conductivity_umho=0.0,  # Zero conductivity (pure water)
            operating_pressure_psig=0.0,  # Atmospheric
        )
        assert bw.specific_conductivity_umho == 0.0

    def test_maximum_allowed_values(self):
        """Test maximum allowed values are accepted."""
        bw = BoilerWaterInput(
            sample_point="drum",
            ph=14.0,  # Maximum pH
            phosphate_ppm=200.0,  # Maximum phosphate
            specific_conductivity_umho=10000.0,
            operating_pressure_psig=2000.0,
        )
        assert bw.ph == 14.0
        assert bw.phosphate_ppm == 200.0

    def test_optional_fields_none(self):
        """Test optional fields can be None."""
        bw = BoilerWaterInput(
            sample_point="drum",
            ph=10.5,
            specific_conductivity_umho=2000.0,
            operating_pressure_psig=450.0,
            # All optional fields are None by default
        )
        assert bw.phosphate_ppm is None
        assert bw.silica_ppm is None
        assert bw.iron_ppb is None
        assert bw.copper_ppb is None
        assert bw.dissolved_oxygen_ppb is None

    def test_timestamp_auto_generation(self):
        """Test timestamp is auto-generated with timezone."""
        bw = BoilerWaterInput(
            sample_point="drum",
            ph=10.5,
            specific_conductivity_umho=2000.0,
            operating_pressure_psig=450.0,
        )
        assert bw.timestamp is not None
        assert bw.timestamp.tzinfo is not None


class TestComplianceWithASME:
    """Test schema compliance with ASME guidelines."""

    @pytest.mark.compliance
    def test_ph_range_covers_asme_limits(self):
        """Test pH range (0-14) covers all ASME operating ranges."""
        # ASME typically specifies pH 9.0-11.5 depending on pressure class
        # Schema should accept this full range
        for ph in [9.0, 9.5, 10.0, 10.5, 11.0, 11.5]:
            bw = BoilerWaterInput(
                sample_point="drum",
                ph=ph,
                specific_conductivity_umho=2000.0,
                operating_pressure_psig=450.0,
            )
            assert bw.ph == ph

    @pytest.mark.compliance
    def test_pressure_class_definitions(self):
        """Test pressure class definitions match ASME."""
        # Low pressure: < 300 psig
        # Medium pressure: 300-900 psig
        # High pressure: 900-1500 psig
        # Supercritical: > 1500 psig
        assert BoilerPressureClass.LOW_PRESSURE.value == "low_pressure"
        assert BoilerPressureClass.MEDIUM_PRESSURE.value == "medium_pressure"
        assert BoilerPressureClass.HIGH_PRESSURE.value == "high_pressure"
        assert BoilerPressureClass.SUPERCRITICAL.value == "supercritical"

    @pytest.mark.compliance
    def test_treatment_program_types(self):
        """Test treatment program types cover ASME/EPRI programs."""
        programs = [
            TreatmentProgram.PHOSPHATE_PRECIPITATE,
            TreatmentProgram.PHOSPHATE_POLYMER,
            TreatmentProgram.COORDINATED_PHOSPHATE,
            TreatmentProgram.CONGRUENT_PHOSPHATE,
            TreatmentProgram.ALL_VOLATILE,
            TreatmentProgram.OXYGENATED_TREATMENT,
            TreatmentProgram.CAUSTIC_TREATMENT,
        ]
        assert len(programs) == 7
