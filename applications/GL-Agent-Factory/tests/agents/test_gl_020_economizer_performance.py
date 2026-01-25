"""
Unit Tests for GL-020: Economizer Performance Analysis Agent (ECONOPULSE)

Comprehensive test suite covering:
- Heat exchanger effectiveness (NTU-epsilon method)
- Acid dew point calculation (Verhoff-Banchero correlation)
- Steaming risk detection (IAPWS-IF97)
- Cold-end corrosion risk assessment
- LMTD calculations
- Provenance hash generation
- Input validation

Target: 85%+ code coverage

Reference:
- IAPWS-IF97: Industrial Formulation 1997 for Water/Steam
- Verhoff-Banchero (1974): Acid Dew Point Correlation
- Kays & London: Compact Heat Exchangers (NTU-epsilon)
- ASME PTC 4: Fired Steam Generators Performance

Run with:
    pytest tests/agents/test_gl_020_economizer_performance.py -v --cov=backend/agents/gl_020_economizer_performance
"""

import math
import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))

from agents.gl_020_economizer_performance.agent import (
    EconomizerPerformanceAgent,
    EconomizerInput,
    EconomizerOutput,
    FlueGasComposition,
    WaterSideConditions,
    HeatExchangerGeometry,
    OperatingConditions,
    FlowArrangement,
    RiskLevel,
)

from agents.gl_020_economizer_performance.calculators.acid_dew_point import (
    verhoff_banchero_acid_dew_point,
    calculate_partial_pressures,
    PartialPressures,
)


# =============================================================================
# Test Class: Agent Initialization
# =============================================================================


class TestEconomizerAgentInitialization:
    """Tests for EconomizerPerformanceAgent initialization."""

    @pytest.mark.unit
    def test_agent_initializes_with_defaults(self):
        """Test agent initializes correctly with default config."""
        agent = EconomizerPerformanceAgent()

        assert agent is not None
        assert agent.AGENT_ID == "GL-020"
        assert agent.AGENT_NAME == "ECONOPULSE"
        assert agent.VERSION == "1.0.0"

    @pytest.mark.unit
    def test_agent_initializes_with_custom_config(self):
        """Test agent initializes with custom configuration."""
        config = {"debug_mode": True, "max_iterations": 100}
        agent = EconomizerPerformanceAgent(config=config)

        assert agent.config == config


# =============================================================================
# Test Class: Input Validation
# =============================================================================


class TestEconomizerInputValidation:
    """Tests for EconomizerInput Pydantic model validation."""

    @pytest.mark.unit
    def test_valid_flue_gas_input(self):
        """Test valid flue gas composition passes validation."""
        flue_gas = FlueGasComposition(
            temperature_in_celsius=350.0,
            temperature_out_celsius=150.0,
            mass_flow_kg_s=50.0,
            H2O_percent=8.0,
            SO3_ppmv=15.0,
        )

        assert flue_gas.temperature_in_celsius == 350.0
        assert flue_gas.temperature_out_celsius == 150.0

    @pytest.mark.unit
    def test_flue_gas_outlet_less_than_inlet(self):
        """Test flue gas outlet must be less than inlet temperature."""
        with pytest.raises(ValueError, match="outlet temperature.*must be less than"):
            FlueGasComposition(
                temperature_in_celsius=150.0,  # Lower than outlet!
                temperature_out_celsius=200.0,
                mass_flow_kg_s=50.0,
                H2O_percent=8.0,
            )

    @pytest.mark.unit
    def test_valid_water_side_input(self):
        """Test valid water side conditions pass validation."""
        water = WaterSideConditions(
            inlet_temperature_celsius=105.0,
            outlet_temperature_celsius=180.0,
            mass_flow_kg_s=20.0,
            drum_pressure_MPa=4.0,
        )

        assert water.inlet_temperature_celsius == 105.0
        assert water.outlet_temperature_celsius == 180.0

    @pytest.mark.unit
    def test_water_outlet_greater_than_inlet(self):
        """Test water outlet must be greater than inlet temperature."""
        with pytest.raises(ValueError, match="outlet temperature.*must be greater than"):
            WaterSideConditions(
                inlet_temperature_celsius=180.0,  # Higher than outlet!
                outlet_temperature_celsius=105.0,
                mass_flow_kg_s=20.0,
                drum_pressure_MPa=4.0,
            )

    @pytest.mark.unit
    def test_h2o_percent_range(self):
        """Test H2O percent must be 0-30%."""
        with pytest.raises(ValueError):
            FlueGasComposition(
                temperature_in_celsius=350.0,
                temperature_out_celsius=150.0,
                mass_flow_kg_s=50.0,
                H2O_percent=50.0,  # Over 30%
            )

    @pytest.mark.unit
    def test_so3_ppmv_range(self):
        """Test SO3 concentration must be 0-200 ppmv."""
        with pytest.raises(ValueError):
            FlueGasComposition(
                temperature_in_celsius=350.0,
                temperature_out_celsius=150.0,
                mass_flow_kg_s=50.0,
                H2O_percent=8.0,
                SO3_ppmv=500.0,  # Over 200
            )

    @pytest.mark.unit
    def test_drum_pressure_range(self):
        """Test drum pressure must be positive and <= 22 MPa."""
        with pytest.raises(ValueError):
            WaterSideConditions(
                inlet_temperature_celsius=105.0,
                outlet_temperature_celsius=180.0,
                mass_flow_kg_s=20.0,
                drum_pressure_MPa=25.0,  # Over 22 MPa (supercritical)
            )

    @pytest.mark.unit
    def test_flow_arrangement_enum(self):
        """Test flow arrangement accepts valid enum values."""
        geometry = HeatExchangerGeometry(
            flow_arrangement=FlowArrangement.COUNTER_FLOW,
        )
        assert geometry.flow_arrangement == FlowArrangement.COUNTER_FLOW

        geometry2 = HeatExchangerGeometry(
            flow_arrangement=FlowArrangement.PARALLEL_FLOW,
        )
        assert geometry2.flow_arrangement == FlowArrangement.PARALLEL_FLOW


# =============================================================================
# Test Class: Acid Dew Point Calculation (Verhoff-Banchero)
# =============================================================================


class TestAcidDewPointCalculation:
    """Tests for Verhoff-Banchero acid dew point correlation."""

    @pytest.mark.unit
    @pytest.mark.compliance
    def test_verhoff_banchero_known_values(self):
        """
        Test Verhoff-Banchero correlation against known values.

        Reference conditions:
        - P_H2O = 0.08 atm (8% moisture)
        - P_SO3 = 15e-6 atm (15 ppmv)
        - Expected T_dew ~ 127-130 deg C
        """
        T_dew = verhoff_banchero_acid_dew_point(
            P_H2O_atm=0.08,
            P_SO3_atm=15e-6,
        )

        # Verhoff-Banchero typically gives values in 120-140 C range
        assert 120.0 <= T_dew <= 140.0, f"Acid dew point {T_dew} outside expected range"

    @pytest.mark.unit
    @pytest.mark.compliance
    def test_partial_pressure_calculation(self):
        """
        Test partial pressure calculation.

        At 101.325 kPa (1 atm):
        - 8% H2O -> P_H2O = 0.08 atm
        - 15 ppmv SO3 -> P_SO3 = 15e-6 atm
        """
        pressures = calculate_partial_pressures(
            total_pressure_kPa=101.325,
            H2O_percent=8.0,
            SO3_ppmv=15.0,
        )

        assert pressures.P_H2O_atm == pytest.approx(0.08, rel=1e-6)
        assert pressures.P_SO3_atm == pytest.approx(15e-6, rel=1e-6)

    @pytest.mark.unit
    def test_higher_so3_increases_dew_point(self):
        """Test higher SO3 concentration increases acid dew point."""
        T_dew_low = verhoff_banchero_acid_dew_point(P_H2O_atm=0.08, P_SO3_atm=5e-6)
        T_dew_high = verhoff_banchero_acid_dew_point(P_H2O_atm=0.08, P_SO3_atm=50e-6)

        assert T_dew_high > T_dew_low, "Higher SO3 should increase dew point"

    @pytest.mark.unit
    def test_higher_h2o_increases_dew_point(self):
        """Test higher moisture content increases acid dew point."""
        T_dew_low = verhoff_banchero_acid_dew_point(P_H2O_atm=0.05, P_SO3_atm=15e-6)
        T_dew_high = verhoff_banchero_acid_dew_point(P_H2O_atm=0.15, P_SO3_atm=15e-6)

        assert T_dew_high > T_dew_low, "Higher moisture should increase dew point"

    @pytest.mark.unit
    def test_zero_partial_pressure_raises(self):
        """Test zero partial pressure raises ValueError."""
        with pytest.raises(ValueError):
            verhoff_banchero_acid_dew_point(P_H2O_atm=0.0, P_SO3_atm=15e-6)

        with pytest.raises(ValueError):
            verhoff_banchero_acid_dew_point(P_H2O_atm=0.08, P_SO3_atm=0.0)

    @pytest.mark.unit
    @pytest.mark.parametrize("h2o_pct,so3_ppmv,expected_range", [
        (5.0, 10.0, (115, 135)),  # Low moisture, low SO3
        (8.0, 15.0, (120, 140)),  # Typical conditions
        (12.0, 30.0, (135, 155)),  # High moisture, high SO3
        (15.0, 50.0, (145, 165)),  # Very high moisture and SO3
    ])
    def test_acid_dew_point_ranges(self, h2o_pct, so3_ppmv, expected_range):
        """Test acid dew point falls within expected ranges."""
        pressures = calculate_partial_pressures(
            total_pressure_kPa=101.325,
            H2O_percent=h2o_pct,
            SO3_ppmv=so3_ppmv,
        )

        T_dew = verhoff_banchero_acid_dew_point(
            P_H2O_atm=pressures.P_H2O_atm,
            P_SO3_atm=pressures.P_SO3_atm,
        )

        assert expected_range[0] <= T_dew <= expected_range[1], (
            f"T_dew={T_dew:.1f}C outside expected range {expected_range}"
        )


# =============================================================================
# Test Class: Heat Exchanger Effectiveness
# =============================================================================


class TestHeatExchangerEffectiveness:
    """Tests for NTU-epsilon method effectiveness calculations."""

    @pytest.mark.unit
    def test_effectiveness_in_valid_range(self, economizer_agent, economizer_valid_input):
        """Test effectiveness is between 0 and 1."""
        result = economizer_agent.run(economizer_valid_input)

        assert 0.0 <= result.effectiveness <= 1.0
        assert 0.0 <= result.thermal_performance.effectiveness <= 1.0

    @pytest.mark.unit
    @pytest.mark.compliance
    def test_effectiveness_calculation_counter_flow(self, economizer_agent):
        """
        Test counter-flow effectiveness calculation.

        For counter-flow with known temperatures:
        T_hot_in = 350 C, T_hot_out = 150 C
        T_cold_in = 105 C, T_cold_out = 180 C

        epsilon = Q_actual / Q_max
        """
        input_data = EconomizerInput(
            flue_gas=FlueGasComposition(
                temperature_in_celsius=350.0,
                temperature_out_celsius=150.0,
                mass_flow_kg_s=50.0,
                H2O_percent=8.0,
                SO3_ppmv=15.0,
            ),
            water_side=WaterSideConditions(
                inlet_temperature_celsius=105.0,
                outlet_temperature_celsius=180.0,
                mass_flow_kg_s=20.0,
                drum_pressure_MPa=4.0,
            ),
            heat_exchanger=HeatExchangerGeometry(
                flow_arrangement=FlowArrangement.COUNTER_FLOW,
            ),
        )

        result = economizer_agent.run(input_data)

        # Effectiveness should be reasonably high for counter-flow
        assert result.effectiveness > 0.5

    @pytest.mark.unit
    def test_heat_transfer_calculation(self, economizer_agent, economizer_valid_input):
        """Test heat transfer rate calculation."""
        result = economizer_agent.run(economizer_valid_input)

        # Q = m_dot * cp * delta_T
        # For flue gas: 50 kg/s * 1100 J/kg-K * (350-150) K = 11 MW
        assert result.thermal_performance.heat_transfer_kW > 0
        assert result.thermal_performance.heat_transfer_MW > 0

    @pytest.mark.unit
    def test_lmtd_calculation(self, economizer_agent, economizer_valid_input):
        """Test Log Mean Temperature Difference calculation."""
        result = economizer_agent.run(economizer_valid_input)

        # LMTD should be positive
        assert result.thermal_performance.LMTD_celsius > 0

        # For counter-flow:
        # dT1 = T_hot_in - T_cold_out = 350 - 180 = 170
        # dT2 = T_hot_out - T_cold_in = 150 - 105 = 45
        # LMTD = (170-45)/ln(170/45) ~ 95 C
        assert 50 <= result.thermal_performance.LMTD_celsius <= 150


# =============================================================================
# Test Class: Steaming Risk Assessment
# =============================================================================


class TestSteamingRiskAssessment:
    """Tests for steaming risk detection."""

    @pytest.mark.unit
    def test_no_steaming_risk_normal_operation(self, economizer_agent, economizer_valid_input):
        """Test no steaming risk under normal operation."""
        result = economizer_agent.run(economizer_valid_input)

        # Normal operation should have low steaming risk
        # T_water_out (180C) is well below T_sat at 4 MPa (~250C)
        assert result.steaming_analysis.approach_to_saturation_celsius > 20

    @pytest.mark.unit
    @pytest.mark.compliance
    def test_high_steaming_risk_near_saturation(self, economizer_agent, economizer_steaming_risk_input):
        """Test high steaming risk when near saturation."""
        result = economizer_agent.run(economizer_steaming_risk_input)

        # Water outlet at 245C is close to saturation at 4 MPa (~250C)
        assert result.steaming_analysis.approach_to_saturation_celsius < 20
        assert result.steaming_analysis.risk_level in ["HIGH", "SEVERE", "CRITICAL"]

    @pytest.mark.unit
    def test_saturation_temperature_at_4mpa(self, economizer_agent):
        """
        Test saturation temperature at 4 MPa.

        IAPWS-IF97: T_sat(4 MPa) = 250.33 C
        """
        input_data = EconomizerInput(
            flue_gas=FlueGasComposition(
                temperature_in_celsius=300.0,
                temperature_out_celsius=150.0,
                mass_flow_kg_s=50.0,
                H2O_percent=8.0,
            ),
            water_side=WaterSideConditions(
                inlet_temperature_celsius=100.0,
                outlet_temperature_celsius=200.0,
                mass_flow_kg_s=20.0,
                drum_pressure_MPa=4.0,
            ),
        )

        result = economizer_agent.run(input_data)

        # Saturation temp at 4 MPa should be ~250 C
        assert 249 <= result.steaming_analysis.saturation_temperature_celsius <= 252


# =============================================================================
# Test Class: Corrosion Risk Assessment
# =============================================================================


class TestCorrosionRiskAssessment:
    """Tests for cold-end corrosion risk assessment."""

    @pytest.mark.unit
    def test_no_corrosion_risk_high_water_temp(self, economizer_agent, economizer_valid_input):
        """Test no corrosion risk with adequate water temperature."""
        result = economizer_agent.run(economizer_valid_input)

        # With 105C water inlet, metal temp should be above dew point
        assert result.corrosion_analysis.margin_above_dew_point_celsius > 0

    @pytest.mark.unit
    @pytest.mark.compliance
    def test_high_corrosion_risk_cold_operation(self, economizer_agent, economizer_corrosion_risk_input):
        """Test high corrosion risk with cold-end conditions."""
        result = economizer_agent.run(economizer_corrosion_risk_input)

        # Low water inlet (80C) with high SO3 (50 ppmv) should cause risk
        # Tube metal temp could be below acid dew point
        assert result.corrosion_analysis.risk_level in ["MODERATE", "HIGH", "SEVERE"]

    @pytest.mark.unit
    def test_tube_metal_temperature_estimation(self, economizer_agent, economizer_valid_input):
        """Test tube metal temperature is estimated."""
        result = economizer_agent.run(economizer_valid_input)

        # Metal temp should be between water and gas temperatures
        water_inlet = economizer_valid_input.water_side.inlet_temperature_celsius
        gas_outlet = economizer_valid_input.flue_gas.temperature_out_celsius

        metal_temp = result.corrosion_analysis.tube_metal_temperature_celsius

        assert metal_temp >= water_inlet
        assert metal_temp <= gas_outlet


# =============================================================================
# Test Class: Provenance Hash
# =============================================================================


class TestEconomizerProvenanceHash:
    """Tests for economizer provenance hash generation."""

    @pytest.mark.unit
    def test_provenance_hash_exists(self, economizer_agent, economizer_valid_input):
        """Test output includes provenance hash."""
        result = economizer_agent.run(economizer_valid_input)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    @pytest.mark.unit
    def test_provenance_hash_valid_format(self, economizer_agent, economizer_valid_input):
        """Test provenance hash is valid SHA-256."""
        result = economizer_agent.run(economizer_valid_input)

        assert all(c in "0123456789abcdef" for c in result.provenance_hash.lower())

    @pytest.mark.unit
    def test_provenance_chain_captured(self, economizer_agent, economizer_valid_input):
        """Test provenance chain captures all calculation steps."""
        result = economizer_agent.run(economizer_valid_input)

        assert len(result.provenance_chain) >= 4  # At least 4 major steps

        # Check step names
        operations = [step.operation for step in result.provenance_chain]
        assert "thermal_performance_calculation" in operations
        assert "acid_dew_point_calculation" in operations
        assert "steaming_risk_assessment" in operations
        assert "corrosion_risk_assessment" in operations


# =============================================================================
# Test Class: Recommendations
# =============================================================================


class TestEconomizerRecommendations:
    """Tests for operational recommendations."""

    @pytest.mark.unit
    def test_recommendations_provided(self, economizer_agent, economizer_valid_input):
        """Test recommendations are provided in output."""
        result = economizer_agent.run(economizer_valid_input)

        assert isinstance(result.recommendations, list)
        assert len(result.recommendations) >= 1

    @pytest.mark.unit
    def test_steaming_recommendation_when_risk(self, economizer_agent, economizer_steaming_risk_input):
        """Test steaming recommendation when risk is high."""
        result = economizer_agent.run(economizer_steaming_risk_input)

        # Should have recommendation about steaming risk
        steaming_recs = [r for r in result.recommendations if "steam" in r.lower() or "temperature" in r.lower()]
        assert len(steaming_recs) > 0

    @pytest.mark.unit
    def test_corrosion_recommendation_when_risk(self, economizer_agent, economizer_corrosion_risk_input):
        """Test corrosion recommendation when risk is high."""
        result = economizer_agent.run(economizer_corrosion_risk_input)

        # Should have recommendation about corrosion/dew point
        corrosion_recs = [r for r in result.recommendations if "corrosion" in r.lower() or "dew point" in r.lower() or "feedwater" in r.lower()]
        assert len(corrosion_recs) > 0


# =============================================================================
# Test Class: Overall Risk Level
# =============================================================================


class TestOverallRiskLevel:
    """Tests for overall risk level determination."""

    @pytest.mark.unit
    def test_overall_risk_is_worst_case(self, economizer_agent, economizer_steaming_risk_input):
        """Test overall risk is the worst of individual risks."""
        result = economizer_agent.run(economizer_steaming_risk_input)

        steaming_risk = result.steaming_analysis.risk_level
        corrosion_risk = result.corrosion_analysis.risk_level

        # Overall should be at least as high as individual risks
        risk_order = ["NONE", "LOW", "MODERATE", "HIGH", "SEVERE", "CRITICAL"]

        steaming_idx = risk_order.index(steaming_risk) if steaming_risk in risk_order else 0
        corrosion_idx = risk_order.index(corrosion_risk) if corrosion_risk in risk_order else 0
        overall_idx = risk_order.index(result.overall_risk_level) if result.overall_risk_level in risk_order else 0

        assert overall_idx >= max(steaming_idx, corrosion_idx)


# =============================================================================
# Test Class: Edge Cases
# =============================================================================


class TestEconomizerEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.unit
    def test_low_mass_flow(self, economizer_agent):
        """Test handling of very low mass flow rates."""
        input_data = EconomizerInput(
            flue_gas=FlueGasComposition(
                temperature_in_celsius=300.0,
                temperature_out_celsius=150.0,
                mass_flow_kg_s=1.0,  # Very low
                H2O_percent=8.0,
            ),
            water_side=WaterSideConditions(
                inlet_temperature_celsius=100.0,
                outlet_temperature_celsius=150.0,
                mass_flow_kg_s=0.5,  # Very low
                drum_pressure_MPa=2.0,
            ),
        )

        result = economizer_agent.run(input_data)

        assert result.validation_status == "PASS"
        assert result.effectiveness >= 0

    @pytest.mark.unit
    def test_high_pressure_operation(self, economizer_agent):
        """Test high pressure (HRSG) operation."""
        input_data = EconomizerInput(
            flue_gas=FlueGasComposition(
                temperature_in_celsius=500.0,
                temperature_out_celsius=200.0,
                mass_flow_kg_s=100.0,
                H2O_percent=10.0,
            ),
            water_side=WaterSideConditions(
                inlet_temperature_celsius=150.0,
                outlet_temperature_celsius=280.0,
                mass_flow_kg_s=50.0,
                drum_pressure_MPa=15.0,  # High pressure HRSG
            ),
        )

        result = economizer_agent.run(input_data)

        assert result.validation_status == "PASS"

    @pytest.mark.unit
    def test_various_flow_arrangements(self, economizer_agent):
        """Test various heat exchanger flow arrangements."""
        arrangements = [
            FlowArrangement.COUNTER_FLOW,
            FlowArrangement.PARALLEL_FLOW,
            FlowArrangement.CROSS_FLOW_BOTH_UNMIXED,
        ]

        for arrangement in arrangements:
            input_data = EconomizerInput(
                flue_gas=FlueGasComposition(
                    temperature_in_celsius=350.0,
                    temperature_out_celsius=150.0,
                    mass_flow_kg_s=50.0,
                    H2O_percent=8.0,
                ),
                water_side=WaterSideConditions(
                    inlet_temperature_celsius=105.0,
                    outlet_temperature_celsius=180.0,
                    mass_flow_kg_s=20.0,
                    drum_pressure_MPa=4.0,
                ),
                heat_exchanger=HeatExchangerGeometry(
                    flow_arrangement=arrangement,
                ),
            )

            result = economizer_agent.run(input_data)

            assert result.validation_status == "PASS"
            assert 0 <= result.effectiveness <= 1


# =============================================================================
# Test Class: Output Model
# =============================================================================


class TestEconomizerOutput:
    """Tests for EconomizerOutput model."""

    @pytest.mark.unit
    def test_output_has_all_required_fields(self, economizer_agent, economizer_valid_input):
        """Test output includes all required fields."""
        result = economizer_agent.run(economizer_valid_input)

        # Core fields
        assert hasattr(result, "analysis_id")
        assert hasattr(result, "timestamp")
        assert hasattr(result, "effectiveness")
        assert hasattr(result, "acid_dew_point_celsius")

        # Analysis sections
        assert hasattr(result, "thermal_performance")
        assert hasattr(result, "acid_dew_point_analysis")
        assert hasattr(result, "steaming_analysis")
        assert hasattr(result, "corrosion_analysis")

        # Provenance
        assert hasattr(result, "provenance_hash")
        assert hasattr(result, "provenance_chain")

        # Processing metadata
        assert hasattr(result, "processing_time_ms")
        assert hasattr(result, "validation_status")

    @pytest.mark.unit
    def test_output_analysis_id_format(self, economizer_agent, economizer_valid_input):
        """Test analysis ID has correct format."""
        result = economizer_agent.run(economizer_valid_input)

        assert result.analysis_id.startswith("ECON-")

    @pytest.mark.unit
    def test_output_processing_time_recorded(self, economizer_agent, economizer_valid_input):
        """Test processing time is recorded."""
        result = economizer_agent.run(economizer_valid_input)

        assert result.processing_time_ms > 0
        assert result.processing_time_ms < 1000  # Should be well under 1 second


# =============================================================================
# Test Class: Performance
# =============================================================================


class TestEconomizerPerformance:
    """Performance tests for EconomizerPerformanceAgent."""

    @pytest.mark.unit
    @pytest.mark.performance
    def test_single_analysis_under_100ms(self, economizer_agent, economizer_valid_input, performance_timer):
        """Test single analysis completes in under 100ms."""
        performance_timer.start()
        result = economizer_agent.run(economizer_valid_input)
        performance_timer.stop()

        assert performance_timer.elapsed_ms < 100.0

    @pytest.mark.unit
    @pytest.mark.performance
    def test_batch_analysis(self, economizer_agent, performance_timer):
        """Test batch analysis throughput."""
        num_analyses = 100
        inputs = [
            EconomizerInput(
                flue_gas=FlueGasComposition(
                    temperature_in_celsius=300.0 + i * 0.5,
                    temperature_out_celsius=150.0,
                    mass_flow_kg_s=50.0,
                    H2O_percent=8.0,
                    SO3_ppmv=15.0,
                ),
                water_side=WaterSideConditions(
                    inlet_temperature_celsius=100.0 + i * 0.1,
                    outlet_temperature_celsius=180.0,
                    mass_flow_kg_s=20.0,
                    drum_pressure_MPa=4.0,
                ),
            )
            for i in range(num_analyses)
        ]

        performance_timer.start()
        results = [economizer_agent.run(inp) for inp in inputs]
        performance_timer.stop()

        assert len(results) == num_analyses
        throughput = num_analyses / (performance_timer.elapsed_ms / 1000)
        assert throughput >= 20, f"Throughput {throughput:.0f} rec/sec below target"
