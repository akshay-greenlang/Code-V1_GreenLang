"""
GL-015 INSULSCAN - Schema Tests

Unit tests for schema models including geometry specifications,
insulation layers, jacketing, input/output models, and result schemas.

Coverage target: 85%+
"""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from greenlang.agents.process_heat.gl_015_insulation_analysis.schemas import (
    GeometryType,
    InsulationCondition,
    ServiceType,
    JacketingType,
    PipeGeometry,
    VesselGeometry,
    FlatSurfaceGeometry,
    InsulationLayer,
    JacketingSpec,
    InsulationInput,
    HeatLossResult,
    EconomicThicknessResult,
    SurfaceTemperatureResult,
    CondensationAnalysisResult,
    IRHotSpot,
    IRSurveyResult,
    InsulationRecommendation,
    InsulationOutput,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def valid_pipe_geometry():
    """Create valid pipe geometry."""
    return PipeGeometry(
        nominal_pipe_size_in=4.0,
        pipe_length_ft=100.0,
        pipe_schedule="40",
        orientation="horizontal",
    )


@pytest.fixture
def valid_vessel_geometry():
    """Create valid vessel geometry."""
    return VesselGeometry(
        vessel_diameter_ft=8.0,
        vessel_length_ft=20.0,
        vessel_type="horizontal_cylinder",
        head_type="2:1_elliptical",
        include_heads=True,
    )


@pytest.fixture
def valid_flat_geometry():
    """Create valid flat surface geometry."""
    return FlatSurfaceGeometry(
        length_ft=10.0,
        width_ft=8.0,
        orientation="vertical",
    )


@pytest.fixture
def valid_insulation_layer():
    """Create valid insulation layer."""
    return InsulationLayer(
        layer_number=1,
        material_id="mineral_wool_8pcf",
        thickness_in=2.0,
    )


@pytest.fixture
def valid_jacketing_spec():
    """Create valid jacketing specification."""
    return JacketingSpec(
        jacketing_type=JacketingType.ALUMINUM,
        thickness_in=0.016,
        emissivity=0.10,
    )


@pytest.fixture
def valid_pipe_input(valid_pipe_geometry, valid_insulation_layer, valid_jacketing_spec):
    """Create valid pipe insulation input."""
    return InsulationInput(
        item_name="Test Pipe",
        operating_temperature_f=350.0,
        ambient_temperature_f=77.0,
        geometry_type=GeometryType.PIPE,
        pipe_geometry=valid_pipe_geometry,
        insulation_layers=[valid_insulation_layer],
        jacketing=valid_jacketing_spec,
    )


# =============================================================================
# GEOMETRY ENUM TESTS
# =============================================================================

class TestGeometryEnums:
    """Tests for geometry enums."""

    def test_geometry_type_values(self):
        """Test geometry type enum values."""
        assert GeometryType.PIPE.value == "pipe"
        assert GeometryType.VESSEL.value == "vessel"
        assert GeometryType.FLAT_SURFACE.value == "flat_surface"
        assert GeometryType.TANK.value == "tank"
        assert GeometryType.DUCT.value == "duct"
        assert GeometryType.EQUIPMENT.value == "equipment"

    def test_insulation_condition_values(self):
        """Test insulation condition enum values."""
        assert InsulationCondition.NEW.value == "new"
        assert InsulationCondition.GOOD.value == "good"
        assert InsulationCondition.FAIR.value == "fair"
        assert InsulationCondition.POOR.value == "poor"
        assert InsulationCondition.DAMAGED.value == "damaged"
        assert InsulationCondition.MISSING.value == "missing"
        assert InsulationCondition.SATURATED.value == "saturated"

    def test_service_type_values(self):
        """Test service type enum values."""
        assert ServiceType.HOT.value == "hot"
        assert ServiceType.COLD.value == "cold"
        assert ServiceType.CRYOGENIC.value == "cryogenic"
        assert ServiceType.DUAL_TEMPERATURE.value == "dual_temperature"

    def test_jacketing_type_values(self):
        """Test jacketing type enum values."""
        assert JacketingType.ALUMINUM.value == "aluminum"
        assert JacketingType.STAINLESS_STEEL.value == "stainless_steel"
        assert JacketingType.GALVANIZED.value == "galvanized"
        assert JacketingType.PVC.value == "pvc"
        assert JacketingType.NONE.value == "none"


# =============================================================================
# PIPE GEOMETRY TESTS
# =============================================================================

class TestPipeGeometry:
    """Tests for PipeGeometry schema."""

    def test_valid_pipe_geometry(self, valid_pipe_geometry):
        """Test valid pipe geometry creation."""
        geom = valid_pipe_geometry

        assert geom.nominal_pipe_size_in == 4.0
        assert geom.pipe_length_ft == 100.0
        assert geom.pipe_schedule == "40"
        assert geom.orientation == "horizontal"

    def test_outer_diameter_auto_calculation(self):
        """Test outer diameter is auto-calculated from NPS."""
        geom = PipeGeometry(
            nominal_pipe_size_in=4.0,
            pipe_length_ft=100.0,
        )

        # 4" pipe has OD of 4.500"
        assert geom.outer_diameter_in == 4.500

    def test_outer_diameter_lookup_table(self):
        """Test NPS to OD lookup for common sizes."""
        test_cases = [
            (0.5, 0.840),
            (1.0, 1.315),
            (2.0, 2.375),
            (6.0, 6.625),
            (12.0, 12.750),
            (24.0, 24.000),
        ]

        for nps, expected_od in test_cases:
            geom = PipeGeometry(
                nominal_pipe_size_in=nps,
                pipe_length_ft=10.0,
            )
            assert geom.outer_diameter_in == expected_od

    def test_explicit_outer_diameter(self):
        """Test explicit outer diameter overrides lookup."""
        geom = PipeGeometry(
            nominal_pipe_size_in=4.0,
            outer_diameter_in=4.625,  # Non-standard OD
            pipe_length_ft=100.0,
        )

        assert geom.outer_diameter_in == 4.625

    def test_pipe_size_validation(self):
        """Test pipe size validation."""
        with pytest.raises(ValidationError):
            PipeGeometry(
                nominal_pipe_size_in=0,  # Must be > 0
                pipe_length_ft=100.0,
            )

        with pytest.raises(ValidationError):
            PipeGeometry(
                nominal_pipe_size_in=150,  # Above 120
                pipe_length_ft=100.0,
            )

    def test_pipe_length_validation(self):
        """Test pipe length must be positive."""
        with pytest.raises(ValidationError):
            PipeGeometry(
                nominal_pipe_size_in=4.0,
                pipe_length_ft=0,
            )


# =============================================================================
# VESSEL GEOMETRY TESTS
# =============================================================================

class TestVesselGeometry:
    """Tests for VesselGeometry schema."""

    def test_valid_vessel_geometry(self, valid_vessel_geometry):
        """Test valid vessel geometry creation."""
        geom = valid_vessel_geometry

        assert geom.vessel_diameter_ft == 8.0
        assert geom.vessel_length_ft == 20.0
        assert geom.vessel_type == "horizontal_cylinder"
        assert geom.head_type == "2:1_elliptical"
        assert geom.include_heads is True
        assert geom.shell_thickness_in == 0.5

    def test_vessel_diameter_validation(self):
        """Test vessel diameter must be positive."""
        with pytest.raises(ValidationError):
            VesselGeometry(
                vessel_diameter_ft=0,
                vessel_length_ft=20.0,
            )

    def test_vessel_length_validation(self):
        """Test vessel length must be positive."""
        with pytest.raises(ValidationError):
            VesselGeometry(
                vessel_diameter_ft=8.0,
                vessel_length_ft=-5.0,
            )

    def test_head_types(self):
        """Test different head type options."""
        head_types = ["hemispherical", "2:1_elliptical", "flat", "torispherical"]

        for head_type in head_types:
            geom = VesselGeometry(
                vessel_diameter_ft=8.0,
                vessel_length_ft=20.0,
                head_type=head_type,
            )
            assert geom.head_type == head_type


# =============================================================================
# FLAT SURFACE GEOMETRY TESTS
# =============================================================================

class TestFlatSurfaceGeometry:
    """Tests for FlatSurfaceGeometry schema."""

    def test_valid_flat_geometry(self, valid_flat_geometry):
        """Test valid flat surface geometry creation."""
        geom = valid_flat_geometry

        assert geom.length_ft == 10.0
        assert geom.width_ft == 8.0
        assert geom.orientation == "vertical"

    def test_surface_area_auto_calculation(self):
        """Test surface area is auto-calculated."""
        geom = FlatSurfaceGeometry(
            length_ft=10.0,
            width_ft=8.0,
        )

        assert geom.surface_area_sqft == 80.0

    def test_explicit_surface_area(self):
        """Test explicit surface area overrides calculation."""
        geom = FlatSurfaceGeometry(
            length_ft=10.0,
            width_ft=8.0,
            surface_area_sqft=100.0,  # Override
        )

        assert geom.surface_area_sqft == 100.0

    def test_orientation_options(self):
        """Test orientation options."""
        orientations = ["horizontal_up", "horizontal_down", "vertical"]

        for orientation in orientations:
            geom = FlatSurfaceGeometry(
                length_ft=10.0,
                width_ft=8.0,
                orientation=orientation,
            )
            assert geom.orientation == orientation


# =============================================================================
# INSULATION LAYER TESTS
# =============================================================================

class TestInsulationLayer:
    """Tests for InsulationLayer schema."""

    def test_valid_insulation_layer(self, valid_insulation_layer):
        """Test valid insulation layer creation."""
        layer = valid_insulation_layer

        assert layer.layer_number == 1
        assert layer.material_id == "mineral_wool_8pcf"
        assert layer.thickness_in == 2.0
        assert layer.condition == InsulationCondition.GOOD.value
        assert layer.condition_factor == 1.0

    def test_layer_number_validation(self):
        """Test layer number range validation."""
        with pytest.raises(ValidationError):
            InsulationLayer(
                layer_number=0,  # Must be >= 1
                material_id="mineral_wool_8pcf",
                thickness_in=2.0,
            )

        with pytest.raises(ValidationError):
            InsulationLayer(
                layer_number=6,  # Must be <= 5
                material_id="mineral_wool_8pcf",
                thickness_in=2.0,
            )

    def test_thickness_validation(self):
        """Test thickness validation."""
        with pytest.raises(ValidationError):
            InsulationLayer(
                layer_number=1,
                material_id="mineral_wool_8pcf",
                thickness_in=0,  # Must be > 0
            )

        with pytest.raises(ValidationError):
            InsulationLayer(
                layer_number=1,
                material_id="mineral_wool_8pcf",
                thickness_in=30,  # Must be <= 24
            )

    def test_condition_factor_validation(self):
        """Test condition factor range."""
        with pytest.raises(ValidationError):
            InsulationLayer(
                layer_number=1,
                material_id="mineral_wool_8pcf",
                thickness_in=2.0,
                condition_factor=0.3,  # Must be >= 0.5
            )

        with pytest.raises(ValidationError):
            InsulationLayer(
                layer_number=1,
                material_id="mineral_wool_8pcf",
                thickness_in=2.0,
                condition_factor=2.5,  # Must be <= 2.0
            )

    def test_damaged_condition_factor(self):
        """Test that damaged insulation has higher condition factor."""
        layer = InsulationLayer(
            layer_number=1,
            material_id="mineral_wool_8pcf",
            thickness_in=2.0,
            condition=InsulationCondition.DAMAGED,
            condition_factor=1.5,  # Damaged = higher k-value
        )

        assert layer.condition_factor == 1.5


# =============================================================================
# JACKETING SPEC TESTS
# =============================================================================

class TestJacketingSpec:
    """Tests for JacketingSpec schema."""

    def test_valid_jacketing_spec(self, valid_jacketing_spec):
        """Test valid jacketing specification."""
        spec = valid_jacketing_spec

        assert spec.jacketing_type == JacketingType.ALUMINUM.value
        assert spec.thickness_in == 0.016
        assert spec.emissivity == 0.10
        assert spec.corroded is False

    def test_emissivity_validation(self):
        """Test emissivity range validation."""
        with pytest.raises(ValidationError):
            JacketingSpec(
                jacketing_type=JacketingType.ALUMINUM,
                emissivity=0.02,  # Below 0.03
            )

        with pytest.raises(ValidationError):
            JacketingSpec(
                jacketing_type=JacketingType.ALUMINUM,
                emissivity=1.0,  # Above 0.95
            )

    def test_thickness_validation(self):
        """Test jacketing thickness validation."""
        with pytest.raises(ValidationError):
            JacketingSpec(
                jacketing_type=JacketingType.ALUMINUM,
                thickness_in=0,  # Must be > 0
            )

        with pytest.raises(ValidationError):
            JacketingSpec(
                jacketing_type=JacketingType.ALUMINUM,
                thickness_in=0.2,  # Above 0.1
            )

    def test_corroded_jacketing(self):
        """Test corroded jacketing flag."""
        spec = JacketingSpec(
            jacketing_type=JacketingType.GALVANIZED,
            corroded=True,
        )

        assert spec.corroded is True


# =============================================================================
# INSULATION INPUT TESTS
# =============================================================================

class TestInsulationInput:
    """Tests for InsulationInput schema."""

    def test_valid_pipe_input(self, valid_pipe_input):
        """Test valid pipe insulation input."""
        input_data = valid_pipe_input

        assert input_data.item_name == "Test Pipe"
        assert input_data.operating_temperature_f == 350.0
        assert input_data.ambient_temperature_f == 77.0
        assert input_data.geometry_type == GeometryType.PIPE.value
        assert input_data.pipe_geometry is not None
        assert len(input_data.insulation_layers) == 1
        assert input_data.jacketing is not None

    def test_auto_item_id(self):
        """Test item_id is auto-generated."""
        input_data = InsulationInput(
            operating_temperature_f=350.0,
            geometry_type=GeometryType.PIPE,
            pipe_geometry=PipeGeometry(
                nominal_pipe_size_in=4.0,
                pipe_length_ft=100.0,
            ),
        )

        assert input_data.item_id is not None
        assert len(input_data.item_id) == 8

    def test_operating_temperature_validation(self):
        """Test operating temperature validation."""
        with pytest.raises(ValidationError):
            InsulationInput(
                operating_temperature_f=-500.0,  # Below absolute zero (-459.67F)
                geometry_type=GeometryType.PIPE,
                pipe_geometry=PipeGeometry(
                    nominal_pipe_size_in=4.0,
                    pipe_length_ft=100.0,
                ),
            )

    def test_ambient_temperature_validation(self):
        """Test ambient temperature range."""
        with pytest.raises(ValidationError):
            InsulationInput(
                operating_temperature_f=350.0,
                ambient_temperature_f=-150.0,  # Below -100
                geometry_type=GeometryType.PIPE,
                pipe_geometry=PipeGeometry(
                    nominal_pipe_size_in=4.0,
                    pipe_length_ft=100.0,
                ),
            )

    def test_service_type_auto_detection(self):
        """Test service type auto-detection from temperature."""
        # Cryogenic
        input_data = InsulationInput(
            operating_temperature_f=-200.0,
            geometry_type=GeometryType.PIPE,
            pipe_geometry=PipeGeometry(
                nominal_pipe_size_in=4.0,
                pipe_length_ft=100.0,
            ),
        )
        assert input_data.service_type == ServiceType.CRYOGENIC.value

        # Cold
        input_data = InsulationInput(
            operating_temperature_f=40.0,
            geometry_type=GeometryType.PIPE,
            pipe_geometry=PipeGeometry(
                nominal_pipe_size_in=4.0,
                pipe_length_ft=100.0,
            ),
        )
        assert input_data.service_type == ServiceType.COLD.value

        # Hot
        input_data = InsulationInput(
            operating_temperature_f=350.0,
            geometry_type=GeometryType.PIPE,
            pipe_geometry=PipeGeometry(
                nominal_pipe_size_in=4.0,
                pipe_length_ft=100.0,
            ),
        )
        assert input_data.service_type == ServiceType.HOT.value

    def test_multiple_insulation_layers(self):
        """Test multiple insulation layers."""
        input_data = InsulationInput(
            operating_temperature_f=350.0,
            geometry_type=GeometryType.PIPE,
            pipe_geometry=PipeGeometry(
                nominal_pipe_size_in=4.0,
                pipe_length_ft=100.0,
            ),
            insulation_layers=[
                InsulationLayer(
                    layer_number=1,
                    material_id="mineral_wool_8pcf",
                    thickness_in=2.0,
                ),
                InsulationLayer(
                    layer_number=2,
                    material_id="calcium_silicate_8pcf",
                    thickness_in=1.5,
                ),
            ],
        )

        assert len(input_data.insulation_layers) == 2

    def test_bare_surface_input(self):
        """Test bare surface (no insulation) input."""
        input_data = InsulationInput(
            operating_temperature_f=350.0,
            geometry_type=GeometryType.PIPE,
            pipe_geometry=PipeGeometry(
                nominal_pipe_size_in=4.0,
                pipe_length_ft=100.0,
            ),
            insulation_layers=[],  # No insulation
        )

        assert len(input_data.insulation_layers) == 0


# =============================================================================
# HEAT LOSS RESULT TESTS
# =============================================================================

class TestHeatLossResult:
    """Tests for HeatLossResult schema."""

    def test_valid_heat_loss_result(self):
        """Test valid heat loss result creation."""
        result = HeatLossResult(
            heat_loss_btu_hr=5000.0,
            outer_surface_temperature_f=95.0,
            convection_heat_transfer_btu_hr=3000.0,
            radiation_heat_transfer_btu_hr=2000.0,
            total_thermal_resistance_hr_f_btu=0.05,
        )

        assert result.heat_loss_btu_hr == 5000.0
        assert result.outer_surface_temperature_f == 95.0
        assert result.convection_heat_transfer_btu_hr == 3000.0
        assert result.radiation_heat_transfer_btu_hr == 2000.0

    def test_optional_fields(self):
        """Test optional fields have default values."""
        result = HeatLossResult(
            heat_loss_btu_hr=5000.0,
            outer_surface_temperature_f=95.0,
            convection_heat_transfer_btu_hr=3000.0,
            radiation_heat_transfer_btu_hr=2000.0,
            total_thermal_resistance_hr_f_btu=0.05,
        )

        assert result.heat_loss_btu_hr_ft is None
        assert result.heat_loss_btu_hr_sqft is None
        assert result.bare_surface_heat_loss_btu_hr is None
        assert result.calculation_method == "ASTM_C680"


# =============================================================================
# ECONOMIC THICKNESS RESULT TESTS
# =============================================================================

class TestEconomicThicknessResult:
    """Tests for EconomicThicknessResult schema."""

    def test_valid_economic_result(self):
        """Test valid economic thickness result."""
        result = EconomicThicknessResult(
            optimal_thickness_in=3.0,
            recommended_material="Mineral Wool - 8 pcf",
            current_heat_loss_btu_hr=10000.0,
            optimal_heat_loss_btu_hr=3000.0,
            heat_loss_savings_btu_hr=7000.0,
            annual_energy_cost_current_usd=5000.0,
            annual_energy_cost_optimal_usd=1500.0,
            annual_savings_usd=3500.0,
            insulation_cost_usd=2000.0,
            installation_cost_usd=1000.0,
            total_project_cost_usd=3000.0,
            simple_payback_years=0.86,
            npv_usd=15000.0,
            roi_pct=450.0,
        )

        assert result.optimal_thickness_in == 3.0
        assert result.annual_savings_usd == 3500.0
        assert result.simple_payback_years == 0.86


# =============================================================================
# IR HOT SPOT AND SURVEY RESULT TESTS
# =============================================================================

class TestIRSurveySchemas:
    """Tests for IR survey related schemas."""

    def test_valid_hot_spot(self):
        """Test valid hot spot creation."""
        hot_spot = IRHotSpot(
            location_description="Pipe elbow at unit 3",
            measured_temperature_f=180.0,
            expected_temperature_f=95.0,
            delta_t_f=85.0,
            severity="high",
            recommended_action="Replace damaged insulation",
        )

        assert hot_spot.measured_temperature_f == 180.0
        assert hot_spot.delta_t_f == 85.0
        assert hot_spot.severity == "high"

    def test_ir_survey_result(self):
        """Test IR survey result creation."""
        result = IRSurveyResult(
            ambient_temperature_f=77.0,
            items_surveyed=50,
            total_anomalies=5,
        )

        assert result.items_surveyed == 50
        assert result.total_anomalies == 5
        assert result.hot_spots_identified == []


# =============================================================================
# INSULATION OUTPUT TESTS
# =============================================================================

class TestInsulationOutput:
    """Tests for InsulationOutput schema."""

    def test_valid_output(self):
        """Test valid output creation."""
        heat_loss = HeatLossResult(
            heat_loss_btu_hr=5000.0,
            outer_surface_temperature_f=95.0,
            convection_heat_transfer_btu_hr=3000.0,
            radiation_heat_transfer_btu_hr=2000.0,
            total_thermal_resistance_hr_f_btu=0.05,
        )

        output = InsulationOutput(
            item_id="TEST-001",
            heat_loss=heat_loss,
        )

        assert output.item_id == "TEST-001"
        assert output.status == "success"
        assert output.heat_loss.heat_loss_btu_hr == 5000.0

    def test_output_with_recommendations(self):
        """Test output with recommendations."""
        heat_loss = HeatLossResult(
            heat_loss_btu_hr=5000.0,
            outer_surface_temperature_f=95.0,
            convection_heat_transfer_btu_hr=3000.0,
            radiation_heat_transfer_btu_hr=2000.0,
            total_thermal_resistance_hr_f_btu=0.05,
        )

        recommendation = InsulationRecommendation(
            category="economic",
            title="Add Insulation",
            description="Add 2 inches of insulation",
            current_state="1 inch insulation",
            recommended_action="Add 1 more inch",
        )

        output = InsulationOutput(
            item_id="TEST-001",
            heat_loss=heat_loss,
            recommendations=[recommendation],
        )

        assert len(output.recommendations) == 1
        assert output.recommendations[0].category == "economic"
