# -*- coding: utf-8 -*-
"""
Reproducibility Tests for GL-011 FuelCraft

Validates that calculations can be reproduced exactly:
- Load saved bundle and replay with same inputs
- Assert identical outputs
- Verify provenance chain integrity

Author: GL-TestEngineer
Date: 2025-01-01
"""

import pytest
from decimal import Decimal
from datetime import datetime, date, timezone
import json
import hashlib

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.mark.golden
class TestBundleReplayReproducibility:
    """Tests for bundle replay reproducibility."""

    def test_replay_produces_identical_output(self, tmp_path):
        """Test replaying a bundle produces identical output."""
        from audit.run_bundle import (
            RunBundleBuilder,
            ImmutableStorage,
            BundleReplayValidator,
        )

        # Create original bundle with calculation result
        original_output = {
            "total_cost": Decimal("50000.00"),
            "emissions_kg": Decimal("1500.00"),
            "fuel_mix": {"natural_gas": 0.7, "diesel": 0.3},
        }

        # Convert Decimals to strings for JSON compatibility
        original_output_json = {
            "total_cost": str(original_output["total_cost"]),
            "emissions_kg": str(original_output["emissions_kg"]),
            "fuel_mix": original_output["fuel_mix"],
        }

        builder = RunBundleBuilder(
            run_id="RUN-REPRO-001",
            agent_version="1.0.0",
            environment="test",
        )
        builder.add_output(name="result", data=original_output_json)
        original_manifest = builder.seal()

        # Store bundle
        storage = ImmutableStorage(str(tmp_path / "storage"))
        storage.store_bundle(original_manifest, builder._components_data)

        # Retrieve and replay
        retrieved_manifest = storage.retrieve_bundle(original_manifest.bundle_hash)
        assert retrieved_manifest is not None

        # Simulate replay with same inputs -> same outputs
        replay_output = original_output_json  # Same calculation

        validator = BundleReplayValidator()
        result = validator.validate_replay(
            original_bundle=retrieved_manifest,
            replay_outputs={"result": replay_output}
        )

        assert result.is_identical is True

    def test_modified_input_detected(self, tmp_path):
        """Test that modified inputs are detected during replay."""
        from audit.run_bundle import (
            RunBundleBuilder,
            ImmutableStorage,
            BundleReplayValidator,
        )

        # Original calculation
        original_output = {"value": 100}

        builder = RunBundleBuilder(
            run_id="RUN-REPRO-002",
            agent_version="1.0.0",
            environment="test",
        )
        builder.add_output(name="result", data=original_output)
        original_manifest = builder.seal()

        storage = ImmutableStorage(str(tmp_path / "storage"))
        storage.store_bundle(original_manifest, builder._components_data)

        # Replay with different output (simulating changed inputs)
        modified_output = {"value": 101}

        validator = BundleReplayValidator()
        result = validator.validate_replay(
            original_bundle=original_manifest,
            replay_outputs={"result": modified_output}
        )

        assert result.is_identical is False
        assert len(result.mismatches) > 0


@pytest.mark.golden
class TestCalculationReproducibility:
    """Tests for calculation reproducibility."""

    def test_blend_calculation_reproducible(
        self,
        blend_calculator,
        blend_component_diesel
    ):
        """Test blend calculation is reproducible across sessions."""
        from calculators.blend_calculator import BlendInput

        blend_input = BlendInput(
            components=[blend_component_diesel],
            blend_fractions=[Decimal("1.0")],
        )

        # First "session"
        result1 = blend_calculator.calculate(blend_input)
        result1_dict = result1.to_dict()

        # Simulate new session by creating new calculator
        from calculators.blend_calculator import BlendCalculator
        new_calculator = BlendCalculator()

        result2 = new_calculator.calculate(blend_input)
        result2_dict = result2.to_dict()

        # Core calculation values should be identical
        assert result1_dict["total_mass_kg"] == result2_dict["total_mass_kg"]
        assert result1_dict["blend_lhv_mj_kg"] == result2_dict["blend_lhv_mj_kg"]
        assert result1_dict["total_energy_mj"] == result2_dict["total_energy_mj"]

    def test_carbon_calculation_reproducible(self, carbon_calculator):
        """Test carbon calculation is reproducible."""
        from calculators.carbon_calculator import CarbonInput, EmissionBoundary

        carbon_input = CarbonInput(
            fuel_type="diesel",
            energy_mj=Decimal("1000000"),
            boundary=EmissionBoundary.WTW,
            reference_date=date(2024, 1, 1),
        )

        # Multiple calculations
        results = [carbon_calculator.calculate(carbon_input) for _ in range(5)]

        # All should be identical
        for i in range(1, len(results)):
            assert results[i].ttw_emissions_kg_co2e == results[0].ttw_emissions_kg_co2e
            assert results[i].wtt_emissions_kg_co2e == results[0].wtt_emissions_kg_co2e


@pytest.mark.golden
class TestProvenanceChainIntegrity:
    """Tests for provenance chain integrity."""

    def test_provenance_chain_verifiable(self, tmp_path):
        """Test provenance chain can be verified."""
        from audit.run_bundle import RunBundleBuilder, ImmutableStorage

        # Build bundle with multiple steps
        builder = RunBundleBuilder(
            run_id="RUN-CHAIN-001",
            agent_version="1.0.0",
            environment="test",
        )

        # Add sequential components
        input_data = {"fuel_id": "NG-001", "quantity_mj": 1000000}
        calc_data = {"lhv_mj_kg": 50.0, "blend_fraction": 1.0}
        output_data = {"total_cost": 5000, "emissions_kg": 1500}

        builder.add_input_snapshot(name="inputs", data=input_data)
        builder.add_output(name="calculation", data=calc_data)
        builder.add_output(name="result", data=output_data)

        manifest = builder.seal()

        # Verify chain integrity
        storage = ImmutableStorage(str(tmp_path / "storage"))
        storage.store_bundle(manifest, builder._components_data)

        is_valid = storage.verify_integrity(manifest.bundle_hash)
        assert is_valid is True

        # Verify component count
        assert len(manifest.components) == 3

    def test_provenance_hash_chains(self):
        """Test that provenance hashes form a valid chain."""
        from audit.run_bundle import RunBundleBuilder

        builder = RunBundleBuilder(
            run_id="RUN-CHAIN-002",
            agent_version="1.0.0",
            environment="test",
        )

        # Add components and track their hashes
        builder.add_input_snapshot(name="input1", data={"step": 1})
        builder.add_input_snapshot(name="input2", data={"step": 2})
        builder.add_output(name="output", data={"step": 3})

        manifest = builder.seal()

        # Bundle hash should depend on all component hashes
        component_hashes = sorted([c.content_hash for c in manifest.components])

        # Verify bundle hash is computed from components
        expected_content = "|".join(component_hashes) + "|" + manifest.manifest_hash
        expected_bundle_hash = hashlib.sha256(expected_content.encode()).hexdigest()

        assert manifest.bundle_hash == expected_bundle_hash


@pytest.mark.golden
class TestCrossVersionReproducibility:
    """Tests for reproducibility across software versions."""

    def test_calculation_with_version_metadata(self, blend_calculator):
        """Test calculations include version metadata for reproducibility."""
        from calculators.blend_calculator import BlendComponent, BlendInput

        component = BlendComponent(
            component_id="TEST-001",
            fuel_type="diesel",
            mass_kg=Decimal("1000"),
            lhv_mj_kg=Decimal("43.00"),
            hhv_mj_kg=Decimal("45.80"),
            density_kg_m3=Decimal("840"),
            sulfur_wt_pct=Decimal("0.05"),
            ash_wt_pct=Decimal("0.01"),
            water_vol_pct=Decimal("0.0"),
            viscosity_50c_cst=Decimal("3.5"),
            flash_point_c=Decimal("65.0"),
            vapor_pressure_kpa=Decimal("0.5"),
            carbon_intensity_kg_co2e_mj=Decimal("0.0741"),
        )

        blend_input = BlendInput(
            components=[component],
            blend_fractions=[Decimal("1.0")],
        )

        result = blend_calculator.calculate(blend_input)
        result_dict = result.to_dict()

        # Should include calculator version for reproducibility
        assert blend_calculator.NAME == "BlendCalculator"
        assert blend_calculator.VERSION == "1.0.0"


@pytest.mark.golden
class TestDataRoundtrip:
    """Tests for data serialization roundtrip."""

    def test_result_serialization_roundtrip(
        self,
        blend_calculator,
        blend_component_diesel
    ):
        """Test result can be serialized and deserialized."""
        from calculators.blend_calculator import BlendInput

        blend_input = BlendInput(
            components=[blend_component_diesel],
            blend_fractions=[Decimal("1.0")],
        )

        original_result = blend_calculator.calculate(blend_input)
        original_dict = original_result.to_dict()

        # Serialize to JSON
        json_str = json.dumps(original_dict, default=str)

        # Deserialize
        restored_dict = json.loads(json_str)

        # Key values should match
        assert restored_dict["total_mass_kg"] == original_dict["total_mass_kg"]
        assert restored_dict["blend_lhv_mj_kg"] == original_dict["blend_lhv_mj_kg"]

    def test_bundle_serialization_roundtrip(self, tmp_path):
        """Test bundle can be serialized and deserialized."""
        from audit.run_bundle import RunBundleBuilder, ImmutableStorage

        original_data = {
            "complex": {
                "nested": {"value": 123},
                "list": [1, 2, 3],
            }
        }

        builder = RunBundleBuilder(
            run_id="RUN-SERIAL-001",
            agent_version="1.0.0",
            environment="test",
        )
        builder.add_input_snapshot(name="complex_data", data=original_data)
        manifest = builder.seal()

        # Store and retrieve
        storage = ImmutableStorage(str(tmp_path / "storage"))
        storage.store_bundle(manifest, builder._components_data)

        retrieved = storage.retrieve_bundle(manifest.bundle_hash)

        # Verify integrity
        assert storage.verify_integrity(manifest.bundle_hash) is True
        assert retrieved.bundle_id == manifest.bundle_id


@pytest.mark.golden
class TestEmissionFactorReproducibility:
    """Tests for emission factor reproducibility."""

    def test_same_factor_used_for_same_date(self, carbon_calculator):
        """Test same emission factor is used for same reference date."""
        from calculators.carbon_calculator import CarbonInput, EmissionBoundary

        reference_date = date(2024, 6, 15)

        results = []
        for _ in range(10):
            carbon_input = CarbonInput(
                fuel_type="diesel",
                energy_mj=Decimal("1000000"),
                boundary=EmissionBoundary.TTW,
                reference_date=reference_date,
            )
            result = carbon_calculator.calculate(carbon_input)
            results.append(result)

        # All should use same emission factor
        first_factors = results[0].factors_used
        for result in results[1:]:
            for i, factor in enumerate(result.factors_used):
                assert factor.factor_id == first_factors[i].factor_id
                assert factor.factor_value == first_factors[i].factor_value


@pytest.mark.golden
class TestSolverDeterminism:
    """Tests for solver determinism."""

    def test_model_structure_reproducible(
        self,
        fuel_data_objects,
        tank_data_objects,
        demand_data_objects,
        model_config
    ):
        """Test model structure is reproducible."""
        from optimization.model_builder import FuelOptimizationModel, ModelConfig

        config = ModelConfig(**model_config)

        # Build same model twice
        model1 = FuelOptimizationModel(
            config=config,
            fuels=fuel_data_objects,
            tanks=tank_data_objects,
            demands=demand_data_objects,
        )

        model2 = FuelOptimizationModel(
            config=config,
            fuels=fuel_data_objects,
            tanks=tank_data_objects,
            demands=demand_data_objects,
        )

        # Structure should be identical
        assert len(model1.variables) == len(model2.variables)
        assert len(model1.constraints) == len(model2.constraints)
        assert model1._compute_hash() == model2._compute_hash()
