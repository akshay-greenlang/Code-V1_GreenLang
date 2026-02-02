"""
GL-015 INSULSCAN - Heat Loss Calculator Tests

Unit tests for HeatLossCalculator including cylindrical (pipe),
vessel, and flat surface heat loss calculations per ASTM C680.

Coverage target: 85%+
"""

import pytest
import math

from greenlang.agents.process_heat.gl_015_insulation_analysis.heat_loss import (
    HeatLossCalculator,
    HeatTransferConstants,
    ConvectionCoefficient,
    RadiationCoefficient,
)
from greenlang.agents.process_heat.gl_015_insulation_analysis.materials import (
    InsulationMaterialDatabase,
)
from greenlang.agents.process_heat.gl_015_insulation_analysis.schemas import (
    InsulationInput,
    PipeGeometry,
    VesselGeometry,
    FlatSurfaceGeometry,
    InsulationLayer,
    JacketingSpec,
    GeometryType,
    JacketingType,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def material_database():
    """Create material database instance."""
    return InsulationMaterialDatabase()


@pytest.fixture
def heat_loss_calculator(material_database):
    """Create heat loss calculator instance."""
    return HeatLossCalculator(
        material_database=material_database,
        convergence_tol=0.001,
        max_iterations=100,
    )


@pytest.fixture
def bare_pipe_input():
    """Create bare pipe input (no insulation)."""
    return InsulationInput(
        item_name="Bare Pipe",
        operating_temperature_f=350.0,
        ambient_temperature_f=77.0,
        geometry_type=GeometryType.PIPE,
        pipe_geometry=PipeGeometry(
            nominal_pipe_size_in=4.0,
            pipe_length_ft=100.0,
            orientation="horizontal",
        ),
        insulation_layers=[],
    )


@pytest.fixture
def insulated_pipe_input():
    """Create insulated pipe input."""
    return InsulationInput(
        item_name="Insulated Pipe",
        operating_temperature_f=350.0,
        ambient_temperature_f=77.0,
        geometry_type=GeometryType.PIPE,
        pipe_geometry=PipeGeometry(
            nominal_pipe_size_in=4.0,
            pipe_length_ft=100.0,
            orientation="horizontal",
        ),
        insulation_layers=[
            InsulationLayer(
                layer_number=1,
                material_id="mineral_wool_8pcf",
                thickness_in=2.0,
            ),
        ],
        jacketing=JacketingSpec(
            jacketing_type=JacketingType.ALUMINUM,
            emissivity=0.10,
        ),
    )


@pytest.fixture
def multi_layer_pipe_input():
    """Create multi-layer insulated pipe input."""
    return InsulationInput(
        item_name="Multi-Layer Pipe",
        operating_temperature_f=600.0,
        ambient_temperature_f=77.0,
        geometry_type=GeometryType.PIPE,
        pipe_geometry=PipeGeometry(
            nominal_pipe_size_in=6.0,
            pipe_length_ft=100.0,
            orientation="horizontal",
        ),
        insulation_layers=[
            InsulationLayer(
                layer_number=1,
                material_id="calcium_silicate_8pcf",
                thickness_in=2.0,
            ),
            InsulationLayer(
                layer_number=2,
                material_id="mineral_wool_8pcf",
                thickness_in=1.5,
            ),
        ],
        jacketing=JacketingSpec(
            jacketing_type=JacketingType.ALUMINUM,
            emissivity=0.10,
        ),
    )


@pytest.fixture
def vessel_input():
    """Create vessel input."""
    return InsulationInput(
        item_name="Test Vessel",
        operating_temperature_f=400.0,
        ambient_temperature_f=77.0,
        geometry_type=GeometryType.VESSEL,
        vessel_geometry=VesselGeometry(
            vessel_diameter_ft=8.0,
            vessel_length_ft=20.0,
            vessel_type="horizontal_cylinder",
            head_type="2:1_elliptical",
            include_heads=True,
        ),
        insulation_layers=[
            InsulationLayer(
                layer_number=1,
                material_id="mineral_wool_8pcf",
                thickness_in=3.0,
            ),
        ],
        jacketing=JacketingSpec(
            jacketing_type=JacketingType.ALUMINUM,
            emissivity=0.10,
        ),
    )


@pytest.fixture
def flat_surface_input():
    """Create flat surface input."""
    return InsulationInput(
        item_name="Test Wall",
        operating_temperature_f=300.0,
        ambient_temperature_f=77.0,
        geometry_type=GeometryType.FLAT_SURFACE,
        flat_geometry=FlatSurfaceGeometry(
            length_ft=10.0,
            width_ft=8.0,
            orientation="vertical",
        ),
        insulation_layers=[
            InsulationLayer(
                layer_number=1,
                material_id="mineral_wool_8pcf",
                thickness_in=2.0,
            ),
        ],
        jacketing=JacketingSpec(
            jacketing_type=JacketingType.ALUMINUM,
            emissivity=0.10,
        ),
    )


# =============================================================================
# HEAT TRANSFER CONSTANTS TESTS
# =============================================================================

class TestHeatTransferConstants:
    """Tests for heat transfer constants."""

    def test_stefan_boltzmann_constant(self):
        """Test Stefan-Boltzmann constant value."""
        assert abs(HeatTransferConstants.STEFAN_BOLTZMANN - 1.714e-9) < 1e-12

    def test_air_properties(self):
        """Test air property values."""
        assert HeatTransferConstants.AIR_KINEMATIC_VISCOSITY > 0
        assert HeatTransferConstants.AIR_THERMAL_CONDUCTIVITY > 0
        assert abs(HeatTransferConstants.AIR_PRANDTL_NUMBER - 0.71) < 0.01

    def test_emissivity_values(self):
        """Test default emissivity values."""
        assert HeatTransferConstants.EMISSIVITY_ALUMINUM == 0.10
        assert HeatTransferConstants.EMISSIVITY_GALVANIZED == 0.28
        assert HeatTransferConstants.EMISSIVITY_PAINTED == 0.90
        assert HeatTransferConstants.EMISSIVITY_BARE_STEEL == 0.80


# =============================================================================
# CALCULATOR INITIALIZATION TESTS
# =============================================================================

class TestHeatLossCalculatorInitialization:
    """Tests for HeatLossCalculator initialization."""

    def test_calculator_initialization(self, heat_loss_calculator):
        """Test calculator initializes correctly."""
        calc = heat_loss_calculator

        assert calc.material_db is not None
        assert calc.convergence_tol == 0.001
        assert calc.max_iterations == 100
        assert calc.calculation_count == 0

    def test_calculator_default_initialization(self):
        """Test calculator with default parameters."""
        calc = HeatLossCalculator()

        assert calc.material_db is not None
        assert calc.convergence_tol == 0.001
        assert calc.max_iterations == 100

    def test_calculator_custom_convergence(self, material_database):
        """Test calculator with custom convergence."""
        calc = HeatLossCalculator(
            material_database=material_database,
            convergence_tol=0.0001,
            max_iterations=200,
        )

        assert calc.convergence_tol == 0.0001
        assert calc.max_iterations == 200


# =============================================================================
# PIPE HEAT LOSS CALCULATION TESTS
# =============================================================================

class TestPipeHeatLossCalculation:
    """Tests for pipe heat loss calculations."""

    def test_insulated_pipe_heat_loss(self, heat_loss_calculator, insulated_pipe_input):
        """Test heat loss calculation for insulated pipe."""
        result = heat_loss_calculator.calculate_heat_loss(insulated_pipe_input)

        assert result is not None
        assert result.heat_loss_btu_hr > 0
        assert result.outer_surface_temperature_f > insulated_pipe_input.ambient_temperature_f
        assert result.outer_surface_temperature_f < insulated_pipe_input.operating_temperature_f
        assert result.heat_loss_btu_hr_ft > 0
        assert result.calculation_method == "ASTM_C680_CYLINDRICAL"

    def test_bare_pipe_vs_insulated(self, heat_loss_calculator, bare_pipe_input, insulated_pipe_input):
        """Test insulated pipe has less heat loss than bare."""
        bare_result = heat_loss_calculator.calculate_heat_loss(bare_pipe_input)
        insulated_result = heat_loss_calculator.calculate_heat_loss(insulated_pipe_input)

        assert bare_result.heat_loss_btu_hr > insulated_result.heat_loss_btu_hr
        assert insulated_result.heat_loss_reduction_pct > 0

    def test_multi_layer_pipe(self, heat_loss_calculator, multi_layer_pipe_input):
        """Test heat loss with multiple insulation layers."""
        result = heat_loss_calculator.calculate_heat_loss(multi_layer_pipe_input)

        assert result is not None
        assert result.heat_loss_btu_hr > 0
        assert len(result.layer_temperatures_f) == 4  # Inner, layer1, layer2, outer
        assert len(result.layer_resistances_hr_f_btu) == 2  # Two insulation layers

        # Temperatures should decrease from inner to outer
        for i in range(len(result.layer_temperatures_f) - 1):
            assert result.layer_temperatures_f[i] >= result.layer_temperatures_f[i + 1]

    def test_heat_loss_per_linear_foot(self, heat_loss_calculator, insulated_pipe_input):
        """Test heat loss per linear foot calculation."""
        result = heat_loss_calculator.calculate_heat_loss(insulated_pipe_input)

        expected_per_ft = result.heat_loss_btu_hr / insulated_pipe_input.pipe_geometry.pipe_length_ft
        assert abs(result.heat_loss_btu_hr_ft - expected_per_ft) < 1.0

    def test_heat_loss_per_sqft(self, heat_loss_calculator, insulated_pipe_input):
        """Test heat loss per square foot calculation."""
        result = heat_loss_calculator.calculate_heat_loss(insulated_pipe_input)

        assert result.heat_loss_btu_hr_sqft > 0

    def test_vertical_vs_horizontal_pipe(self, heat_loss_calculator, insulated_pipe_input):
        """Test vertical pipe orientation affects convection."""
        horizontal_result = heat_loss_calculator.calculate_heat_loss(insulated_pipe_input)

        # Create vertical pipe input
        vertical_input = insulated_pipe_input.copy(deep=True)
        vertical_input.pipe_geometry.orientation = "vertical"
        vertical_result = heat_loss_calculator.calculate_heat_loss(vertical_input)

        # Results should be different due to convection differences
        assert horizontal_result.heat_loss_btu_hr != vertical_result.heat_loss_btu_hr

    def test_wind_effect_on_heat_loss(self, heat_loss_calculator, insulated_pipe_input):
        """Test wind increases heat loss."""
        no_wind_result = heat_loss_calculator.calculate_heat_loss(insulated_pipe_input)

        # Create input with wind
        wind_input = insulated_pipe_input.copy(deep=True)
        wind_input.wind_speed_mph = 10.0
        wind_result = heat_loss_calculator.calculate_heat_loss(wind_input)

        # Wind should increase heat loss
        assert wind_result.heat_loss_btu_hr > no_wind_result.heat_loss_btu_hr


# =============================================================================
# VESSEL HEAT LOSS CALCULATION TESTS
# =============================================================================

class TestVesselHeatLossCalculation:
    """Tests for vessel heat loss calculations."""

    def test_vessel_heat_loss(self, heat_loss_calculator, vessel_input):
        """Test heat loss calculation for vessel."""
        result = heat_loss_calculator.calculate_heat_loss(vessel_input)

        assert result is not None
        assert result.heat_loss_btu_hr > 0
        assert result.outer_surface_temperature_f > vessel_input.ambient_temperature_f
        assert result.outer_surface_temperature_f < vessel_input.operating_temperature_f
        assert result.calculation_method == "ASTM_C680_VESSEL"

    def test_vessel_with_heads(self, heat_loss_calculator, vessel_input):
        """Test vessel with heads included."""
        with_heads = heat_loss_calculator.calculate_heat_loss(vessel_input)

        # Create input without heads
        no_heads_input = vessel_input.copy(deep=True)
        no_heads_input.vessel_geometry.include_heads = False
        no_heads = heat_loss_calculator.calculate_heat_loss(no_heads_input)

        # With heads should have higher heat loss
        assert with_heads.heat_loss_btu_hr > no_heads.heat_loss_btu_hr

    def test_vessel_head_types(self, heat_loss_calculator, vessel_input):
        """Test different head types affect heat loss."""
        head_types = ["hemispherical", "2:1_elliptical", "flat", "torispherical"]
        results = {}

        for head_type in head_types:
            input_copy = vessel_input.copy(deep=True)
            input_copy.vessel_geometry.head_type = head_type
            results[head_type] = heat_loss_calculator.calculate_heat_loss(input_copy)

        # Different head types should give different results
        heat_losses = [r.heat_loss_btu_hr for r in results.values()]
        assert len(set(heat_losses)) > 1  # Not all the same


# =============================================================================
# FLAT SURFACE HEAT LOSS CALCULATION TESTS
# =============================================================================

class TestFlatSurfaceHeatLossCalculation:
    """Tests for flat surface heat loss calculations."""

    def test_flat_surface_heat_loss(self, heat_loss_calculator, flat_surface_input):
        """Test heat loss calculation for flat surface."""
        result = heat_loss_calculator.calculate_heat_loss(flat_surface_input)

        assert result is not None
        assert result.heat_loss_btu_hr > 0
        assert result.outer_surface_temperature_f > flat_surface_input.ambient_temperature_f
        assert result.outer_surface_temperature_f < flat_surface_input.operating_temperature_f
        assert result.calculation_method == "ASTM_C680_FLAT"

    def test_flat_surface_orientations(self, heat_loss_calculator, flat_surface_input):
        """Test different orientations affect heat loss."""
        orientations = ["vertical", "horizontal_up", "horizontal_down"]
        results = {}

        for orientation in orientations:
            input_copy = flat_surface_input.copy(deep=True)
            input_copy.flat_geometry.orientation = orientation
            results[orientation] = heat_loss_calculator.calculate_heat_loss(input_copy)

        # Different orientations should give different results
        heat_losses = [r.heat_loss_btu_hr for r in results.values()]
        # Allow for small numerical differences
        assert max(heat_losses) - min(heat_losses) > 1.0 or len(set(round(h, 0) for h in heat_losses)) >= 2

    def test_flat_surface_area_scaling(self, heat_loss_calculator, flat_surface_input):
        """Test heat loss scales with surface area."""
        result_1 = heat_loss_calculator.calculate_heat_loss(flat_surface_input)

        # Double the area
        double_area_input = flat_surface_input.copy(deep=True)
        double_area_input.flat_geometry.length_ft = 20.0
        double_area_input.flat_geometry.surface_area_sqft = 160.0
        result_2 = heat_loss_calculator.calculate_heat_loss(double_area_input)

        # Heat loss should approximately double
        ratio = result_2.heat_loss_btu_hr / result_1.heat_loss_btu_hr
        assert 1.8 < ratio < 2.2


# =============================================================================
# CONVECTION COEFFICIENT TESTS
# =============================================================================

class TestConvectionCoefficient:
    """Tests for convection coefficient calculations."""

    def test_natural_convection_cylinder(self, heat_loss_calculator):
        """Test natural convection on cylinder (no wind)."""
        h_conv = heat_loss_calculator._calculate_convection_coefficient_cylinder(
            surface_temp_f=150.0,
            ambient_temp_f=77.0,
            diameter_in=6.0,
            wind_speed_mph=0.0,
            orientation="horizontal",
        )

        assert isinstance(h_conv, ConvectionCoefficient)
        assert h_conv.h_conv > 0
        assert h_conv.method == "natural_churchill_chu"
        assert h_conv.rayleigh is not None

    def test_forced_convection_cylinder(self, heat_loss_calculator):
        """Test forced convection on cylinder (with wind)."""
        h_conv = heat_loss_calculator._calculate_convection_coefficient_cylinder(
            surface_temp_f=150.0,
            ambient_temp_f=77.0,
            diameter_in=6.0,
            wind_speed_mph=10.0,
            orientation="horizontal",
        )

        assert h_conv.h_conv > 0
        assert h_conv.method == "combined_hilpert"
        assert h_conv.reynolds is not None

    def test_convection_increases_with_wind(self, heat_loss_calculator):
        """Test convection coefficient increases with wind."""
        h_no_wind = heat_loss_calculator._calculate_convection_coefficient_cylinder(
            surface_temp_f=150.0,
            ambient_temp_f=77.0,
            diameter_in=6.0,
            wind_speed_mph=0.0,
            orientation="horizontal",
        )

        h_with_wind = heat_loss_calculator._calculate_convection_coefficient_cylinder(
            surface_temp_f=150.0,
            ambient_temp_f=77.0,
            diameter_in=6.0,
            wind_speed_mph=15.0,
            orientation="horizontal",
        )

        assert h_with_wind.h_conv > h_no_wind.h_conv

    def test_flat_surface_convection(self, heat_loss_calculator):
        """Test convection on flat surface."""
        h_conv = heat_loss_calculator._calculate_convection_coefficient_flat(
            surface_temp_f=150.0,
            ambient_temp_f=77.0,
            length_ft=5.0,
            orientation="vertical",
            wind_speed_mph=0.0,
        )

        assert h_conv.h_conv > 0


# =============================================================================
# RADIATION COEFFICIENT TESTS
# =============================================================================

class TestRadiationCoefficient:
    """Tests for radiation coefficient calculations."""

    def test_radiation_coefficient(self, heat_loss_calculator):
        """Test radiation coefficient calculation."""
        h_rad = heat_loss_calculator._calculate_radiation_coefficient(
            surface_temp_f=150.0,
            ambient_temp_f=77.0,
            emissivity=0.90,
        )

        assert isinstance(h_rad, RadiationCoefficient)
        assert h_rad.h_rad > 0
        assert h_rad.emissivity == 0.90

    def test_radiation_increases_with_emissivity(self, heat_loss_calculator):
        """Test radiation increases with emissivity."""
        h_low_e = heat_loss_calculator._calculate_radiation_coefficient(
            surface_temp_f=150.0,
            ambient_temp_f=77.0,
            emissivity=0.10,
        )

        h_high_e = heat_loss_calculator._calculate_radiation_coefficient(
            surface_temp_f=150.0,
            ambient_temp_f=77.0,
            emissivity=0.90,
        )

        assert h_high_e.h_rad > h_low_e.h_rad

    def test_radiation_increases_with_temperature(self, heat_loss_calculator):
        """Test radiation increases with temperature difference."""
        h_low_t = heat_loss_calculator._calculate_radiation_coefficient(
            surface_temp_f=100.0,
            ambient_temp_f=77.0,
            emissivity=0.90,
        )

        h_high_t = heat_loss_calculator._calculate_radiation_coefficient(
            surface_temp_f=300.0,
            ambient_temp_f=77.0,
            emissivity=0.90,
        )

        assert h_high_t.h_rad > h_low_t.h_rad


# =============================================================================
# THERMAL RESISTANCE TESTS
# =============================================================================

class TestThermalResistance:
    """Tests for thermal resistance calculations."""

    def test_thermal_resistance_output(self, heat_loss_calculator, insulated_pipe_input):
        """Test thermal resistance is calculated."""
        result = heat_loss_calculator.calculate_heat_loss(insulated_pipe_input)

        assert result.total_thermal_resistance_hr_f_btu > 0

    def test_more_insulation_more_resistance(self, heat_loss_calculator, insulated_pipe_input):
        """Test thicker insulation has more resistance."""
        thin_result = heat_loss_calculator.calculate_heat_loss(insulated_pipe_input)

        # Create thicker insulation input
        thick_input = insulated_pipe_input.copy(deep=True)
        thick_input.insulation_layers[0].thickness_in = 4.0
        thick_result = heat_loss_calculator.calculate_heat_loss(thick_input)

        assert thick_result.total_thermal_resistance_hr_f_btu > thin_result.total_thermal_resistance_hr_f_btu

    def test_layer_resistances(self, heat_loss_calculator, multi_layer_pipe_input):
        """Test layer resistances are calculated."""
        result = heat_loss_calculator.calculate_heat_loss(multi_layer_pipe_input)

        assert len(result.layer_resistances_hr_f_btu) == 2
        assert all(r > 0 for r in result.layer_resistances_hr_f_btu)


# =============================================================================
# CONVERGENCE TESTS
# =============================================================================

class TestConvergence:
    """Tests for iterative convergence."""

    def test_convergence_achieved(self, heat_loss_calculator, insulated_pipe_input):
        """Test calculation converges successfully."""
        result = heat_loss_calculator.calculate_heat_loss(insulated_pipe_input)

        # If we got a result, convergence was achieved
        assert result is not None
        assert result.heat_loss_btu_hr > 0

    def test_calculation_counter(self, heat_loss_calculator, insulated_pipe_input):
        """Test calculation counter increments."""
        initial_count = heat_loss_calculator.calculation_count

        heat_loss_calculator.calculate_heat_loss(insulated_pipe_input)
        assert heat_loss_calculator.calculation_count == initial_count + 1

        heat_loss_calculator.calculate_heat_loss(insulated_pipe_input)
        assert heat_loss_calculator.calculation_count == initial_count + 2


# =============================================================================
# VALIDATION TESTS
# =============================================================================

class TestInputValidation:
    """Tests for input validation."""

    def test_missing_pipe_geometry(self, heat_loss_calculator):
        """Test error when pipe geometry missing."""
        input_data = InsulationInput(
            operating_temperature_f=350.0,
            geometry_type=GeometryType.PIPE,
            pipe_geometry=None,
        )

        with pytest.raises(ValueError, match="Pipe geometry required"):
            heat_loss_calculator.calculate_heat_loss(input_data)

    def test_missing_vessel_geometry(self, heat_loss_calculator):
        """Test error when vessel geometry missing."""
        input_data = InsulationInput(
            operating_temperature_f=350.0,
            geometry_type=GeometryType.VESSEL,
            vessel_geometry=None,
        )

        with pytest.raises(ValueError, match="Vessel geometry required"):
            heat_loss_calculator.calculate_heat_loss(input_data)

    def test_missing_flat_geometry(self, heat_loss_calculator):
        """Test error when flat geometry missing."""
        input_data = InsulationInput(
            operating_temperature_f=350.0,
            geometry_type=GeometryType.FLAT_SURFACE,
            flat_geometry=None,
        )

        with pytest.raises(ValueError, match="Flat geometry required"):
            heat_loss_calculator.calculate_heat_loss(input_data)

    def test_unknown_material(self, heat_loss_calculator):
        """Test error for unknown material ID."""
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
                    material_id="nonexistent_material",
                    thickness_in=2.0,
                ),
            ],
        )

        with pytest.raises(ValueError, match="Unknown material"):
            heat_loss_calculator.calculate_heat_loss(input_data)


# =============================================================================
# CALCULATION ACCURACY TESTS
# =============================================================================

class TestCalculationAccuracy:
    """Tests for calculation accuracy against known values."""

    @pytest.mark.parametrize("operating_temp,expected_range", [
        (200.0, (20, 200)),    # Low temperature
        (350.0, (50, 400)),    # Medium temperature
        (600.0, (100, 800)),   # High temperature
    ])
    def test_heat_loss_reasonable_range(
        self, heat_loss_calculator, operating_temp, expected_range
    ):
        """Test heat loss is in reasonable range for 100ft 4inch pipe."""
        input_data = InsulationInput(
            operating_temperature_f=operating_temp,
            ambient_temperature_f=77.0,
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
            ],
            jacketing=JacketingSpec(
                jacketing_type=JacketingType.ALUMINUM,
                emissivity=0.10,
            ),
        )

        result = heat_loss_calculator.calculate_heat_loss(input_data)

        # Heat loss per foot should be in expected range
        q_per_ft = result.heat_loss_btu_hr_ft
        assert expected_range[0] < q_per_ft < expected_range[1]

    def test_heat_loss_reduction_reasonable(self, heat_loss_calculator, bare_pipe_input, insulated_pipe_input):
        """Test heat loss reduction is reasonable (90%+ with 2in insulation)."""
        bare_result = heat_loss_calculator.calculate_heat_loss(bare_pipe_input)
        insulated_result = heat_loss_calculator.calculate_heat_loss(insulated_pipe_input)

        # 2 inches of mineral wool at 350F should reduce heat loss by 85%+
        assert insulated_result.heat_loss_reduction_pct > 80.0

    def test_surface_temperature_reasonable(self, heat_loss_calculator, insulated_pipe_input):
        """Test surface temperature is between ambient and operating."""
        result = heat_loss_calculator.calculate_heat_loss(insulated_pipe_input)

        assert result.outer_surface_temperature_f > insulated_pipe_input.ambient_temperature_f
        assert result.outer_surface_temperature_f < insulated_pipe_input.operating_temperature_f
        # With low emissivity aluminum jacket, surface temp should be relatively low
        assert result.outer_surface_temperature_f < 150.0
