"""
GL-015 INSULSCAN - Economic Thickness Optimizer Tests

Unit tests for EconomicThicknessOptimizer including NAIMA 3E Plus
methodology, NPV calculations, ROI analysis, and material comparison.

Coverage target: 85%+
"""

import pytest
import math

from greenlang.agents.process_heat.gl_015_insulation_analysis.economic_thickness import (
    EconomicThicknessOptimizer,
    ThicknessPoint,
)
from greenlang.agents.process_heat.gl_015_insulation_analysis.config import (
    InsulationAnalysisConfig,
    EconomicConfig,
)
from greenlang.agents.process_heat.gl_015_insulation_analysis.materials import (
    InsulationMaterialDatabase,
)
from greenlang.agents.process_heat.gl_015_insulation_analysis.heat_loss import (
    HeatLossCalculator,
)
from greenlang.agents.process_heat.gl_015_insulation_analysis.schemas import (
    InsulationInput,
    PipeGeometry,
    InsulationLayer,
    JacketingSpec,
    GeometryType,
    JacketingType,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def analysis_config():
    """Create analysis configuration."""
    return InsulationAnalysisConfig(
        facility_id="TEST-FACILITY",
        economic=EconomicConfig(
            energy_cost_per_mmbtu=10.00,
            operating_hours_per_year=8760,
            plant_lifetime_years=20,
            discount_rate_pct=10.0,
        ),
    )


@pytest.fixture
def material_database():
    """Create material database."""
    return InsulationMaterialDatabase()


@pytest.fixture
def optimizer(analysis_config, material_database):
    """Create economic thickness optimizer."""
    return EconomicThicknessOptimizer(
        config=analysis_config,
        material_database=material_database,
    )


@pytest.fixture
def bare_pipe_input():
    """Create bare pipe input."""
    return InsulationInput(
        item_name="Bare Pipe",
        operating_temperature_f=400.0,
        ambient_temperature_f=77.0,
        geometry_type=GeometryType.PIPE,
        pipe_geometry=PipeGeometry(
            nominal_pipe_size_in=6.0,
            pipe_length_ft=100.0,
        ),
        insulation_layers=[],
    )


@pytest.fixture
def partially_insulated_input():
    """Create partially insulated pipe input."""
    return InsulationInput(
        item_name="Partial Insulation",
        operating_temperature_f=400.0,
        ambient_temperature_f=77.0,
        geometry_type=GeometryType.PIPE,
        pipe_geometry=PipeGeometry(
            nominal_pipe_size_in=6.0,
            pipe_length_ft=100.0,
        ),
        insulation_layers=[
            InsulationLayer(
                layer_number=1,
                material_id="mineral_wool_8pcf",
                thickness_in=1.0,  # Less than optimal
            ),
        ],
        jacketing=JacketingSpec(
            jacketing_type=JacketingType.ALUMINUM,
            emissivity=0.10,
        ),
    )


# =============================================================================
# OPTIMIZER INITIALIZATION TESTS
# =============================================================================

class TestEconomicThicknessOptimizerInit:
    """Tests for optimizer initialization."""

    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initializes correctly."""
        assert optimizer.config is not None
        assert optimizer.economic is not None
        assert optimizer.material_db is not None
        assert optimizer.heat_loss_calc is not None
        assert optimizer.calculation_count == 0

    def test_optimizer_with_custom_calculator(self, analysis_config, material_database):
        """Test optimizer with custom heat loss calculator."""
        custom_calc = HeatLossCalculator(
            material_database=material_database,
            convergence_tol=0.0001,
        )

        optimizer = EconomicThicknessOptimizer(
            config=analysis_config,
            material_database=material_database,
            heat_loss_calculator=custom_calc,
        )

        assert optimizer.heat_loss_calc is custom_calc


# =============================================================================
# ECONOMIC THICKNESS CALCULATION TESTS
# =============================================================================

class TestEconomicThicknessCalculation:
    """Tests for economic thickness calculation."""

    def test_calculate_economic_thickness(self, optimizer, bare_pipe_input):
        """Test basic economic thickness calculation."""
        result = optimizer.calculate_economic_thickness(bare_pipe_input)

        assert result is not None
        assert result.optimal_thickness_in > 0
        assert result.recommended_material is not None
        assert result.calculation_method == "NAIMA_3E_PLUS"

    def test_optimal_thickness_reduces_cost(self, optimizer, bare_pipe_input):
        """Test optimal thickness reduces total annual cost."""
        result = optimizer.calculate_economic_thickness(bare_pipe_input)

        # Current (bare) energy cost should be higher than optimal
        assert result.annual_energy_cost_current_usd > result.annual_energy_cost_optimal_usd

    def test_optimal_thickness_positive_savings(self, optimizer, bare_pipe_input):
        """Test optimal thickness provides positive savings."""
        result = optimizer.calculate_economic_thickness(bare_pipe_input)

        assert result.annual_savings_usd > 0
        assert result.heat_loss_savings_btu_hr > 0

    def test_optimal_thickness_reasonable_payback(self, optimizer, bare_pipe_input):
        """Test optimal thickness has reasonable payback."""
        result = optimizer.calculate_economic_thickness(bare_pipe_input)

        # Payback should be reasonable (< 10 years)
        assert result.simple_payback_years < 10

    def test_optimal_thickness_positive_npv(self, optimizer, bare_pipe_input):
        """Test optimal thickness has positive NPV."""
        result = optimizer.calculate_economic_thickness(bare_pipe_input)

        # NPV should be positive for worthwhile investment
        # Note: At very low energy costs, NPV might be negative
        # For this test with reasonable energy costs, expect positive
        assert result.npv_usd > 0

    def test_specific_material(self, optimizer, bare_pipe_input):
        """Test calculation with specific material."""
        result = optimizer.calculate_economic_thickness(
            bare_pipe_input,
            material_id="calcium_silicate_8pcf",
        )

        assert result.recommended_material == "Calcium Silicate - 8 pcf"

    def test_thickness_range_parameters(self, optimizer, bare_pipe_input):
        """Test thickness range parameters."""
        result = optimizer.calculate_economic_thickness(
            bare_pipe_input,
            min_thickness_in=1.0,
            max_thickness_in=4.0,
            thickness_step_in=0.5,
        )

        # Optimal should be within specified range
        assert 1.0 <= result.optimal_thickness_in <= 4.0


# =============================================================================
# COST CALCULATION TESTS
# =============================================================================

class TestCostCalculations:
    """Tests for cost calculations."""

    def test_annual_energy_cost_calculation(self, optimizer):
        """Test annual energy cost calculation."""
        # 1000 BTU/hr for 8760 hours at $10/MMBTU
        heat_loss = 1000  # BTU/hr
        annual_cost = optimizer._calculate_annual_energy_cost(heat_loss)

        # Expected: 1000 * 8760 / 1,000,000 * 10 = $87.60
        expected = 1000 * 8760 / 1_000_000 * 10.0
        assert abs(annual_cost - expected) < 0.01

    def test_insulation_material_cost(self, optimizer):
        """Test insulation material cost calculation."""
        cost = optimizer._calculate_insulation_material_cost(
            material_id="mineral_wool_8pcf",
            thickness_in=2.0,
            surface_area_sqft=100.0,
        )

        assert cost > 0

    def test_insulation_cost_scales_with_area(self, optimizer):
        """Test insulation cost scales with area."""
        cost_100 = optimizer._calculate_insulation_material_cost(
            material_id="mineral_wool_8pcf",
            thickness_in=2.0,
            surface_area_sqft=100.0,
        )

        cost_200 = optimizer._calculate_insulation_material_cost(
            material_id="mineral_wool_8pcf",
            thickness_in=2.0,
            surface_area_sqft=200.0,
        )

        # Should approximately double
        assert 1.8 < (cost_200 / cost_100) < 2.2

    def test_insulation_cost_scales_with_thickness(self, optimizer):
        """Test insulation cost increases with thickness."""
        cost_1in = optimizer._calculate_insulation_material_cost(
            material_id="mineral_wool_8pcf",
            thickness_in=1.0,
            surface_area_sqft=100.0,
        )

        cost_3in = optimizer._calculate_insulation_material_cost(
            material_id="mineral_wool_8pcf",
            thickness_in=3.0,
            surface_area_sqft=100.0,
        )

        assert cost_3in > cost_1in

    def test_installation_cost(self, optimizer):
        """Test installation cost calculation."""
        cost = optimizer._calculate_installation_cost(
            surface_area_sqft=100.0,
            elevated=False,
        )

        # At 20 sqft/hr and $85/hr, 100 sqft = 5 hours = $425
        expected = (100.0 / 20.0) * 85.0
        assert abs(cost - expected) < 1.0

    def test_installation_cost_elevated(self, optimizer):
        """Test elevated installation cost includes scaffolding."""
        normal = optimizer._calculate_installation_cost(
            surface_area_sqft=100.0,
            elevated=False,
        )

        elevated = optimizer._calculate_installation_cost(
            surface_area_sqft=100.0,
            elevated=True,
        )

        # Elevated should be higher due to scaffolding multiplier
        assert elevated > normal


# =============================================================================
# FINANCIAL CALCULATION TESTS
# =============================================================================

class TestFinancialCalculations:
    """Tests for financial calculations."""

    def test_capital_recovery_factor(self, optimizer):
        """Test capital recovery factor calculation."""
        crf = optimizer._calculate_capital_recovery_factor(
            discount_rate=0.10,
            years=20,
        )

        # CRF for 10% over 20 years is approximately 0.1175
        expected = 0.10 * (1.10 ** 20) / ((1.10 ** 20) - 1)
        assert abs(crf - expected) < 0.0001

    def test_capital_recovery_factor_zero_rate(self, optimizer):
        """Test CRF with zero discount rate."""
        crf = optimizer._calculate_capital_recovery_factor(
            discount_rate=0.0,
            years=20,
        )

        # With 0% discount rate, CRF = 1/years
        assert abs(crf - 0.05) < 0.001

    def test_npv_calculation(self, optimizer):
        """Test NPV calculation."""
        npv = optimizer._calculate_npv(
            initial_investment=1000.0,
            annual_savings=200.0,
            years=10,
            discount_rate=0.10,
        )

        # NPV = -1000 + sum(200/(1.1)^t) for t=1 to 10
        # = -1000 + 200 * (1 - 1.1^-10) / 0.10
        # = -1000 + 200 * 6.1446
        # = -1000 + 1228.92
        # = 228.92
        assert abs(npv - 228.91) < 1.0

    def test_npv_negative_for_poor_investment(self, optimizer):
        """Test NPV is negative for poor investment."""
        npv = optimizer._calculate_npv(
            initial_investment=10000.0,
            annual_savings=100.0,  # Very low savings
            years=10,
            discount_rate=0.10,
        )

        assert npv < 0


# =============================================================================
# THICKNESS POINT EVALUATION TESTS
# =============================================================================

class TestThicknessPointEvaluation:
    """Tests for thickness point evaluation."""

    def test_evaluate_thickness(self, optimizer, bare_pipe_input):
        """Test thickness point evaluation."""
        material = optimizer.material_db.get_material("mineral_wool_8pcf")

        point = optimizer._evaluate_thickness(
            input_data=bare_pipe_input,
            material=material,
            thickness_in=2.0,
        )

        assert isinstance(point, ThicknessPoint)
        assert point.thickness_in == 2.0
        assert point.heat_loss_btu_hr > 0
        assert point.annual_energy_cost_usd > 0
        assert point.insulation_cost_usd > 0
        assert point.total_annual_cost_usd > 0

    def test_thicker_insulation_less_heat_loss(self, optimizer, bare_pipe_input):
        """Test thicker insulation reduces heat loss."""
        material = optimizer.material_db.get_material("mineral_wool_8pcf")

        point_1in = optimizer._evaluate_thickness(
            input_data=bare_pipe_input,
            material=material,
            thickness_in=1.0,
        )

        point_3in = optimizer._evaluate_thickness(
            input_data=bare_pipe_input,
            material=material,
            thickness_in=3.0,
        )

        assert point_3in.heat_loss_btu_hr < point_1in.heat_loss_btu_hr

    def test_thicker_insulation_higher_material_cost(self, optimizer, bare_pipe_input):
        """Test thicker insulation has higher material cost."""
        material = optimizer.material_db.get_material("mineral_wool_8pcf")

        point_1in = optimizer._evaluate_thickness(
            input_data=bare_pipe_input,
            material=material,
            thickness_in=1.0,
        )

        point_3in = optimizer._evaluate_thickness(
            input_data=bare_pipe_input,
            material=material,
            thickness_in=3.0,
        )

        assert point_3in.insulation_cost_usd > point_1in.insulation_cost_usd


# =============================================================================
# MATERIAL SELECTION TESTS
# =============================================================================

class TestMaterialSelection:
    """Tests for material selection."""

    def test_auto_select_material_hot_service(self, optimizer, bare_pipe_input):
        """Test automatic material selection for hot service."""
        material = optimizer._select_optimal_material(bare_pipe_input)

        assert material is not None
        assert material.temperature_range.contains(bare_pipe_input.operating_temperature_f)

    def test_auto_select_material_cold_service(self, optimizer):
        """Test automatic material selection for cold service."""
        cold_input = InsulationInput(
            operating_temperature_f=40.0,
            ambient_temperature_f=77.0,
            geometry_type=GeometryType.PIPE,
            pipe_geometry=PipeGeometry(
                nominal_pipe_size_in=4.0,
                pipe_length_ft=100.0,
            ),
            service_type="cold",
        )

        material = optimizer._select_optimal_material(cold_input)

        assert material is not None
        assert material.suitable_for_cold_service is True


# =============================================================================
# MATERIAL COMPARISON TESTS
# =============================================================================

class TestMaterialComparison:
    """Tests for material comparison functionality."""

    def test_compare_materials(self, optimizer, bare_pipe_input):
        """Test material comparison."""
        materials_to_compare = [
            "mineral_wool_8pcf",
            "calcium_silicate_8pcf",
            "fiberglass_3pcf",
        ]

        results = optimizer.compare_materials(
            input_data=bare_pipe_input,
            material_ids=materials_to_compare,
            target_thickness_in=2.5,
        )

        assert len(results) == 3
        for material_id, result in results.items():
            assert result.optimal_thickness_in >= 2.5

    def test_compare_materials_different_costs(self, optimizer, bare_pipe_input):
        """Test materials have different costs."""
        results = optimizer.compare_materials(
            input_data=bare_pipe_input,
            material_ids=["mineral_wool_8pcf", "aerogel_blanket_8pcf"],
            target_thickness_in=2.0,
        )

        # Aerogel should be more expensive
        mw_cost = results["mineral_wool_8pcf"].total_project_cost_usd
        aerogel_cost = results["aerogel_blanket_8pcf"].total_project_cost_usd

        assert aerogel_cost > mw_cost


# =============================================================================
# SURFACE AREA CALCULATION TESTS
# =============================================================================

class TestSurfaceAreaCalculation:
    """Tests for surface area calculation."""

    def test_pipe_surface_area(self, optimizer):
        """Test pipe surface area calculation."""
        input_data = InsulationInput(
            operating_temperature_f=350.0,
            geometry_type=GeometryType.PIPE,
            pipe_geometry=PipeGeometry(
                nominal_pipe_size_in=4.0,  # OD = 4.5"
                pipe_length_ft=100.0,
            ),
        )

        area = optimizer._get_surface_area(input_data)

        # Area = pi * (4.5/12) * 100 = 117.8 sqft
        expected = math.pi * (4.5 / 12) * 100
        assert abs(area - expected) < 1.0

    def test_flat_surface_area(self, optimizer):
        """Test flat surface area calculation."""
        input_data = InsulationInput(
            operating_temperature_f=350.0,
            geometry_type=GeometryType.FLAT_SURFACE,
            flat_geometry={
                "length_ft": 10.0,
                "width_ft": 8.0,
            },
        )

        area = optimizer._get_surface_area(input_data)

        assert area == 80.0


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_already_optimal_thickness(self, optimizer):
        """Test behavior when already at optimal thickness."""
        # Create input with good insulation
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
                    thickness_in=4.0,  # Already thick
                ),
            ],
        )

        result = optimizer.calculate_economic_thickness(input_data)

        # Additional needed should be small or zero
        assert result.additional_thickness_needed_in <= 2.0

    def test_invalid_material(self, optimizer, bare_pipe_input):
        """Test error for invalid material ID."""
        with pytest.raises(ValueError, match="Unknown material"):
            optimizer.calculate_economic_thickness(
                bare_pipe_input,
                material_id="nonexistent_material",
            )

    def test_calculation_counter(self, optimizer, bare_pipe_input):
        """Test calculation counter increments."""
        initial_count = optimizer.calculation_count

        optimizer.calculate_economic_thickness(bare_pipe_input)
        assert optimizer.calculation_count == initial_count + 1


# =============================================================================
# RESULT CONSISTENCY TESTS
# =============================================================================

class TestResultConsistency:
    """Tests for result consistency."""

    def test_cost_components_sum(self, optimizer, bare_pipe_input):
        """Test cost components sum correctly."""
        result = optimizer.calculate_economic_thickness(bare_pipe_input)

        # Total project cost = insulation + installation
        expected_total = result.insulation_cost_usd + result.installation_cost_usd
        assert abs(result.total_project_cost_usd - expected_total) < 0.01

    def test_savings_consistency(self, optimizer, bare_pipe_input):
        """Test savings are consistent with costs."""
        result = optimizer.calculate_economic_thickness(bare_pipe_input)

        # Annual savings = current cost - optimal cost
        expected_savings = (
            result.annual_energy_cost_current_usd -
            result.annual_energy_cost_optimal_usd
        )
        assert abs(result.annual_savings_usd - expected_savings) < 0.01

    def test_payback_consistency(self, optimizer, bare_pipe_input):
        """Test payback is consistent with costs and savings."""
        result = optimizer.calculate_economic_thickness(bare_pipe_input)

        if result.annual_savings_usd > 0:
            expected_payback = result.total_project_cost_usd / result.annual_savings_usd
            assert abs(result.simple_payback_years - expected_payback) < 0.1
