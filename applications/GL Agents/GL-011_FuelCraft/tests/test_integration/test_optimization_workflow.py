# -*- coding: utf-8 -*-
"""
Integration Tests for Optimization Workflow

Tests complete optimization workflow from data ingestion to result output.
Validates end-to-end functionality including:
- Data validation pipeline
- Model construction
- Solver execution
- Result extraction
- Provenance bundle creation

Author: GL-TestEngineer
Date: 2025-01-01
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.mark.integration
class TestFullOptimizationPipeline:
    """Tests for complete optimization pipeline."""

    def test_pipeline_data_to_result(
        self,
        fuel_data_objects,
        tank_data_objects,
        demand_data_objects,
        model_config
    ):
        """Test complete data-to-result pipeline."""
        from optimization.model_builder import FuelOptimizationModel, ModelConfig

        config = ModelConfig(**model_config)

        # Build model
        model = FuelOptimizationModel(
            config=config,
            fuels=fuel_data_objects,
            tanks=tank_data_objects,
            demands=demand_data_objects,
        )

        # Verify model is constructed
        assert len(model.variables) > 0
        assert len(model.constraints) > 0
        assert len(model.objective_terms) > 0

        # Get statistics
        stats = model.get_model_statistics()
        assert stats["num_fuels"] == len(fuel_data_objects)
        assert stats["num_periods"] == config.time_periods

    def test_pipeline_with_contracts(
        self,
        fuel_data_objects,
        tank_data_objects,
        demand_data_objects,
        contract_data_objects,
        model_config
    ):
        """Test pipeline with contract constraints."""
        from optimization.model_builder import FuelOptimizationModel, ModelConfig

        config = ModelConfig(**model_config)

        model = FuelOptimizationModel(
            config=config,
            fuels=fuel_data_objects,
            tanks=tank_data_objects,
            demands=demand_data_objects,
            contracts=contract_data_objects,
        )

        stats = model.get_model_statistics()
        assert stats["num_contracts"] == len(contract_data_objects)
        assert stats["num_binary"] > 0  # Contract decision variables


@pytest.mark.integration
class TestCalculatorIntegration:
    """Tests for calculator integration."""

    def test_blend_and_carbon_calculators(
        self,
        blend_calculator,
        carbon_calculator,
        blend_component_diesel
    ):
        """Test blend calculator result feeds into carbon calculator."""
        from calculators.blend_calculator import BlendInput
        from calculators.carbon_calculator import CarbonInput, EmissionBoundary
        from datetime import date

        # Calculate blend
        blend_input = BlendInput(
            components=[blend_component_diesel],
            blend_fractions=[Decimal("1.0")],
        )
        blend_result = blend_calculator.calculate(blend_input)

        # Use blend result for carbon calculation
        carbon_input = CarbonInput(
            fuel_type="diesel",
            energy_mj=blend_result.total_energy_mj,
            boundary=EmissionBoundary.WTW,
            reference_date=date(2024, 1, 1),
        )
        carbon_result = carbon_calculator.calculate(carbon_input)

        # Verify integration
        assert carbon_result.wtw_emissions_kg_co2e > Decimal("0")

    def test_heating_value_and_blend_calculators(
        self,
        heating_value_calculator,
        blend_calculator,
        sample_diesel_properties
    ):
        """Test heating value feeds into blend calculator."""
        from calculators.heating_value_calculator import HeatingValueInput
        from calculators.blend_calculator import BlendComponent, BlendInput

        # Calculate heating value
        hv_input = HeatingValueInput(
            fuel_type="diesel",
            quantity=Decimal("1000"),
            quantity_unit="kg",
        )
        hv_result = heating_value_calculator.calculate(hv_input)

        # Create blend component with calculated values
        component = BlendComponent(
            component_id="DIESEL-001",
            fuel_type="diesel",
            mass_kg=hv_result.mass_kg,
            lhv_mj_kg=sample_diesel_properties["lhv_mj_kg"],
            hhv_mj_kg=sample_diesel_properties["hhv_mj_kg"],
            density_kg_m3=sample_diesel_properties["density_kg_m3"],
            sulfur_wt_pct=sample_diesel_properties["sulfur_wt_pct"],
            ash_wt_pct=sample_diesel_properties["ash_wt_pct"],
            water_vol_pct=sample_diesel_properties["water_vol_pct"],
            viscosity_50c_cst=sample_diesel_properties["viscosity_50c_cst"],
            flash_point_c=sample_diesel_properties["flash_point_c"],
            vapor_pressure_kpa=sample_diesel_properties["vapor_pressure_kpa"],
            carbon_intensity_kg_co2e_mj=sample_diesel_properties["carbon_intensity_kg_co2e_mj"],
        )

        blend_input = BlendInput(
            components=[component],
            blend_fractions=[Decimal("1.0")],
        )
        blend_result = blend_calculator.calculate(blend_input)

        # Verify energy matches
        expected_energy = hv_result.mass_kg * sample_diesel_properties["lhv_mj_kg"]
        assert blend_result.total_energy_mj == expected_energy


@pytest.mark.integration
class TestProvenanceIntegration:
    """Tests for provenance tracking integration."""

    def test_calculation_creates_provenance(
        self,
        blend_calculator,
        blend_component_diesel,
        run_bundle_builder
    ):
        """Test calculation results create provenance records."""
        from calculators.blend_calculator import BlendInput

        blend_input = BlendInput(
            components=[blend_component_diesel],
            blend_fractions=[Decimal("1.0")],
        )
        result = blend_calculator.calculate(blend_input)

        # Add to bundle
        run_bundle_builder.add_input_snapshot(
            name="blend_input",
            data=blend_input.to_dict() if hasattr(blend_input, 'to_dict') else {
                "components": [blend_component_diesel.to_dict()],
                "fractions": ["1.0"],
            }
        )
        run_bundle_builder.add_output(
            name="blend_result",
            data=result.to_dict()
        )

        # Seal bundle
        manifest = run_bundle_builder.seal()

        # Verify provenance
        assert manifest.bundle_hash is not None
        assert len(manifest.components) == 2

    def test_bundle_contains_all_steps(
        self,
        blend_calculator,
        carbon_calculator,
        blend_component_diesel,
        run_bundle_builder
    ):
        """Test bundle captures all calculation steps."""
        from calculators.blend_calculator import BlendInput
        from calculators.carbon_calculator import CarbonInput, EmissionBoundary
        from datetime import date

        # Calculate blend
        blend_input = BlendInput(
            components=[blend_component_diesel],
            blend_fractions=[Decimal("1.0")],
        )
        blend_result = blend_calculator.calculate(blend_input)

        # Calculate carbon
        carbon_input = CarbonInput(
            fuel_type="diesel",
            energy_mj=blend_result.total_energy_mj,
        )
        carbon_result = carbon_calculator.calculate(carbon_input)

        # Add all to bundle
        run_bundle_builder.add_input_snapshot(
            name="blend_input",
            data={"components": 1, "fractions": [1.0]}
        )
        run_bundle_builder.add_output(
            name="blend_result",
            data=blend_result.to_dict()
        )
        run_bundle_builder.add_output(
            name="carbon_result",
            data=carbon_result.to_dict()
        )

        manifest = run_bundle_builder.seal()

        # Verify all components captured
        assert len(manifest.components) == 3


@pytest.mark.integration
class TestCircuitBreakerIntegration:
    """Tests for circuit breaker integration with calculations."""

    def test_circuit_breaker_protects_calculation(
        self,
        circuit_breaker,
        blend_calculator,
        blend_component_diesel
    ):
        """Test circuit breaker protects calculation execution."""
        from calculators.blend_calculator import BlendInput

        def execute_blend():
            blend_input = BlendInput(
                components=[blend_component_diesel],
                blend_fractions=[Decimal("1.0")],
            )
            return blend_calculator.calculate(blend_input)

        # Execute through circuit breaker
        result = circuit_breaker.execute(execute_blend)

        assert result is not None
        assert result.blend_lhv_mj_kg > Decimal("0")

    def test_circuit_breaker_records_failures(self, circuit_breaker):
        """Test circuit breaker records calculation failures."""
        def failing_calculation():
            raise ValueError("Calculation failed")

        with pytest.raises(ValueError):
            circuit_breaker.execute(failing_calculation)

        assert circuit_breaker.failure_count == 1


@pytest.mark.integration
class TestModelExportIntegration:
    """Tests for model export and solver integration."""

    def test_model_export_to_mps(
        self,
        fuel_data_objects,
        tank_data_objects,
        demand_data_objects,
        model_config,
        tmp_path
    ):
        """Test model can be exported to MPS format."""
        from optimization.model_builder import FuelOptimizationModel, ModelConfig

        config = ModelConfig(**model_config)

        model = FuelOptimizationModel(
            config=config,
            fuels=fuel_data_objects,
            tanks=tank_data_objects,
            demands=demand_data_objects,
        )

        mps_file = tmp_path / "model.mps"
        mps_content = model.export_mps(str(mps_file))

        assert mps_file.exists()
        assert "NAME" in mps_content
        assert "ENDATA" in mps_content

    def test_model_export_to_dict_complete(
        self,
        fuel_data_objects,
        tank_data_objects,
        demand_data_objects,
        contract_data_objects,
        model_config
    ):
        """Test model dictionary export is complete."""
        from optimization.model_builder import FuelOptimizationModel, ModelConfig

        config = ModelConfig(**model_config)

        model = FuelOptimizationModel(
            config=config,
            fuels=fuel_data_objects,
            tanks=tank_data_objects,
            demands=demand_data_objects,
            contracts=contract_data_objects,
        )

        data = model.export_to_dict()

        assert "config" in data
        assert "variables" in data
        assert "constraints" in data
        assert "objective_terms" in data
        assert "provenance_hash" in data

        # Verify completeness
        assert len(data["variables"]) == len(model.variables)
        assert len(data["constraints"]) == len(model.constraints)


@pytest.mark.integration
class TestUnitConverterIntegration:
    """Tests for unit converter integration with calculations."""

    def test_unit_conversion_in_blend(self, unit_converter):
        """Test unit conversions used in blend calculations."""
        # Convert energy
        energy_result = unit_converter.convert_energy(
            value="100",
            from_unit="MMBtu",
            to_unit="MJ"
        )

        # Convert mass
        mass_result = unit_converter.convert_mass(
            value="1000",
            from_unit="kg",
            to_unit="lb"
        )

        # Verify conversions
        assert energy_result.output_value > Decimal("0")
        assert mass_result.output_value > Decimal("0")
        assert energy_result.provenance_hash is not None

    def test_volume_conversion_with_temperature(self, unit_converter):
        """Test temperature-corrected volume conversion."""
        result = unit_converter.convert_volume(
            value="1000",
            from_unit="L",
            to_unit="gal",
            temperature_c=Decimal("30.0"),
            api_gravity=Decimal("35.0"),
        )

        assert result.output_value > Decimal("0")
        assert result.temperature_c == Decimal("30.0")


@pytest.mark.integration
class TestEndToEndWorkflow:
    """Tests for complete end-to-end workflow."""

    def test_full_workflow_with_provenance(
        self,
        fuel_data_objects,
        tank_data_objects,
        demand_data_objects,
        model_config,
        run_bundle_builder
    ):
        """Test complete workflow from input to auditable bundle."""
        from optimization.model_builder import FuelOptimizationModel, ModelConfig

        # 1. Build model
        config = ModelConfig(**model_config)
        model = FuelOptimizationModel(
            config=config,
            fuels=fuel_data_objects,
            tanks=tank_data_objects,
            demands=demand_data_objects,
        )

        # 2. Capture inputs
        run_bundle_builder.add_input_snapshot(
            name="fuels",
            data=[f.to_dict() for f in fuel_data_objects],
            version="1.0.0"
        )
        run_bundle_builder.add_input_snapshot(
            name="tanks",
            data=[t.to_dict() for t in tank_data_objects],
            version="1.0.0"
        )
        run_bundle_builder.add_input_snapshot(
            name="demands",
            data=[d.to_dict() for d in demand_data_objects],
            version="1.0.0"
        )

        # 3. Capture solver config
        run_bundle_builder.add_solver_config(
            solver_name="highs",
            config=config.to_dict(),
            tolerances={"mip_gap": 0.01}
        )

        # 4. Capture output (model statistics as placeholder for solver result)
        run_bundle_builder.add_output(
            name="model_statistics",
            data=model.get_model_statistics()
        )

        # 5. Seal bundle
        manifest = run_bundle_builder.seal()

        # Verify complete workflow
        assert manifest.status.value == "sealed"
        assert len(manifest.components) == 5
        assert manifest.bundle_hash is not None
        assert manifest.retention_expires is not None
