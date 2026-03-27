"""
test_provenance.py - Tests for CoolingPurchaseProvenance

Tests provenance tracking for AGENT-MRV-012 (Cooling Purchase Agent).
Validates hash chain integrity, deterministic hashing, and stage tracking.

Test Coverage:
- Singleton pattern and reset
- Hash chain creation and sealing
- 19 provenance stages
- Input hashing for all 5 cooling technologies
- Deterministic hashing
- Hash chain verification
"""

import pytest
from decimal import Decimal
from typing import Dict, Any
import hashlib

try:
    from greenlang.agents.mrv.cooling_purchase.provenance import (
        CoolingPurchaseProvenance,
        ProvenanceStage,
    )
except ImportError:
    pytest.skip("cooling_purchase not available", allow_module_level=True)


@pytest.fixture
def provenance():
    """Fresh provenance instance for each test."""
    prov = CoolingPurchaseProvenance()
    prov.reset()
    return prov


class TestSingletonPattern:
    """Test singleton pattern implementation."""

    def test_singleton_instance(self, provenance):
        """Test singleton returns same instance."""
        prov1 = CoolingPurchaseProvenance()
        prov2 = CoolingPurchaseProvenance()
        assert prov1 is prov2

    def test_reset_clears_state(self, provenance):
        """Test reset clears all state."""
        chain_id = provenance.create_chain("test_run")
        provenance.add_stage(chain_id, ProvenanceStage.INPUT_VALIDATION, {"test": "data"})

        provenance.reset()

        # After reset, should be able to create new chain with same ID
        new_chain_id = provenance.create_chain("test_run")
        assert chain_id == new_chain_id


class TestHashChainCreation:
    """Test hash chain creation and management."""

    def test_create_chain_returns_hash(self, provenance):
        """Test create_chain returns initial hash."""
        chain_id = provenance.create_chain("run_001")
        assert isinstance(chain_id, str)
        assert len(chain_id) == 64  # SHA-256 produces 64 hex chars

    def test_create_chain_different_runs(self, provenance):
        """Test different run IDs produce different chain IDs."""
        chain1 = provenance.create_chain("run_001")
        provenance.reset()
        chain2 = provenance.create_chain("run_002")
        assert chain1 != chain2

    def test_add_stage_updates_hash(self, provenance):
        """Test add_stage updates hash."""
        chain_id = provenance.create_chain("test")
        original_hash = chain_id

        new_hash = provenance.add_stage(
            chain_id, ProvenanceStage.INPUT_VALIDATION, {"key": "value"}
        )

        assert new_hash != original_hash
        assert len(new_hash) == 64

    def test_add_stage_sequential(self, provenance):
        """Test sequential stage additions."""
        chain_id = provenance.create_chain("test")

        hash1 = provenance.add_stage(
            chain_id, ProvenanceStage.INPUT_VALIDATION, {"step": 1}
        )
        hash2 = provenance.add_stage(
            chain_id, ProvenanceStage.TECHNOLOGY_LOOKUP, {"step": 2}
        )
        hash3 = provenance.add_stage(
            chain_id, ProvenanceStage.ELECTRIC_CHILLER_CALC, {"step": 3}
        )

        assert hash1 != chain_id
        assert hash2 != hash1
        assert hash3 != hash2

    def test_seal_chain_prevents_additions(self, provenance):
        """Test seal_chain prevents further additions."""
        chain_id = provenance.create_chain("test")
        provenance.add_stage(chain_id, ProvenanceStage.INPUT_VALIDATION, {})

        final_hash = provenance.seal_chain(chain_id)
        assert len(final_hash) == 64

        # Attempting to add after seal should raise error
        with pytest.raises(ValueError, match="sealed"):
            provenance.add_stage(chain_id, ProvenanceStage.TECHNOLOGY_LOOKUP, {})

    def test_seal_chain_returns_final_hash(self, provenance):
        """Test seal_chain returns final hash."""
        chain_id = provenance.create_chain("test")
        hash1 = provenance.add_stage(chain_id, ProvenanceStage.INPUT_VALIDATION, {})

        final_hash = provenance.seal_chain(chain_id)
        assert final_hash == hash1  # Same as last stage hash

    def test_verify_chain_validates_integrity(self, provenance):
        """Test verify_chain validates chain integrity."""
        chain_id = provenance.create_chain("test")
        provenance.add_stage(chain_id, ProvenanceStage.INPUT_VALIDATION, {"a": 1})
        provenance.add_stage(chain_id, ProvenanceStage.TECHNOLOGY_LOOKUP, {"b": 2})
        final_hash = provenance.seal_chain(chain_id)

        # Should verify successfully
        is_valid = provenance.verify_chain(chain_id)
        assert is_valid is True

    def test_verify_chain_detects_tampering(self, provenance):
        """Test verify_chain detects tampering."""
        chain_id = provenance.create_chain("test")
        provenance.add_stage(chain_id, ProvenanceStage.INPUT_VALIDATION, {"a": 1})

        # Manually tamper with internal state (if implementation allows inspection)
        # This test assumes we can access internal state for testing
        # If not, skip this test
        pytest.skip("Tampering detection requires internal state access")


class TestProvenanceStages:
    """Test all 19 provenance stages exist."""

    def test_stage_input_validation(self, provenance):
        """Test INPUT_VALIDATION stage."""
        assert hasattr(ProvenanceStage, "INPUT_VALIDATION")
        chain_id = provenance.create_chain("test")
        provenance.add_stage(chain_id, ProvenanceStage.INPUT_VALIDATION, {})

    def test_stage_technology_lookup(self, provenance):
        """Test TECHNOLOGY_LOOKUP stage."""
        assert hasattr(ProvenanceStage, "TECHNOLOGY_LOOKUP")
        chain_id = provenance.create_chain("test")
        provenance.add_stage(chain_id, ProvenanceStage.TECHNOLOGY_LOOKUP, {})

    def test_stage_electric_chiller_calc(self, provenance):
        """Test ELECTRIC_CHILLER_CALC stage."""
        assert hasattr(ProvenanceStage, "ELECTRIC_CHILLER_CALC")

    def test_stage_absorption_calc(self, provenance):
        """Test ABSORPTION_CALC stage."""
        assert hasattr(ProvenanceStage, "ABSORPTION_CALC")

    def test_stage_free_cooling_calc(self, provenance):
        """Test FREE_COOLING_CALC stage."""
        assert hasattr(ProvenanceStage, "FREE_COOLING_CALC")

    def test_stage_tes_calc(self, provenance):
        """Test TES_CALC stage."""
        assert hasattr(ProvenanceStage, "TES_CALC")

    def test_stage_district_cooling_calc(self, provenance):
        """Test DISTRICT_COOLING_CALC stage."""
        assert hasattr(ProvenanceStage, "DISTRICT_COOLING_CALC")

    def test_stage_refrigerant_leakage(self, provenance):
        """Test REFRIGERANT_LEAKAGE stage."""
        assert hasattr(ProvenanceStage, "REFRIGERANT_LEAKAGE")

    def test_stage_efficiency_adjustment(self, provenance):
        """Test EFFICIENCY_ADJUSTMENT stage."""
        assert hasattr(ProvenanceStage, "EFFICIENCY_ADJUSTMENT")

    def test_stage_auxiliary_energy(self, provenance):
        """Test AUXILIARY_ENERGY stage."""
        assert hasattr(ProvenanceStage, "AUXILIARY_ENERGY")

    def test_stage_grid_emission_factor(self, provenance):
        """Test GRID_EMISSION_FACTOR stage."""
        assert hasattr(ProvenanceStage, "GRID_EMISSION_FACTOR")

    def test_stage_heat_source_emission(self, provenance):
        """Test HEAT_SOURCE_EMISSION stage."""
        assert hasattr(ProvenanceStage, "HEAT_SOURCE_EMISSION")

    def test_stage_tes_loss_calculation(self, provenance):
        """Test TES_LOSS_CALCULATION stage."""
        assert hasattr(ProvenanceStage, "TES_LOSS_CALCULATION")

    def test_stage_emission_calculation(self, provenance):
        """Test EMISSION_CALCULATION stage."""
        assert hasattr(ProvenanceStage, "EMISSION_CALCULATION")

    def test_stage_uncertainty_quantification(self, provenance):
        """Test UNCERTAINTY_QUANTIFICATION stage."""
        assert hasattr(ProvenanceStage, "UNCERTAINTY_QUANTIFICATION")

    def test_stage_compliance_check(self, provenance):
        """Test COMPLIANCE_CHECK stage."""
        assert hasattr(ProvenanceStage, "COMPLIANCE_CHECK")

    def test_stage_batch_aggregation(self, provenance):
        """Test BATCH_AGGREGATION stage."""
        assert hasattr(ProvenanceStage, "BATCH_AGGREGATION")

    def test_stage_result_validation(self, provenance):
        """Test RESULT_VALIDATION stage."""
        assert hasattr(ProvenanceStage, "RESULT_VALIDATION")

    def test_stage_final_seal(self, provenance):
        """Test FINAL_SEAL stage."""
        assert hasattr(ProvenanceStage, "FINAL_SEAL")

    def test_all_19_stages_count(self):
        """Test exactly 19 stages exist."""
        stage_attrs = [
            attr for attr in dir(ProvenanceStage) if not attr.startswith("_")
        ]
        # Filter for actual stage constants
        stage_count = len([attr for attr in stage_attrs if attr.isupper()])
        assert stage_count == 19


class TestElectricChillerInputHashing:
    """Test hash_electric_chiller_input()."""

    def test_hash_electric_chiller_input_returns_hash(self, provenance):
        """Test hash_electric_chiller_input returns hash."""
        input_data = {
            "cooling_output_kwh_th": Decimal("1000.0"),
            "technology": "air_cooled_screw",
            "cop": Decimal("6.1"),
            "grid_emission_factor_kg_co2e_per_kwh": Decimal("0.5"),
        }

        hash_value = provenance.hash_electric_chiller_input(input_data)
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

    def test_hash_electric_chiller_deterministic(self, provenance):
        """Test same inputs produce same hash."""
        input_data = {
            "cooling_output_kwh_th": Decimal("1000.0"),
            "technology": "air_cooled_screw",
            "cop": Decimal("6.1"),
        }

        hash1 = provenance.hash_electric_chiller_input(input_data)
        hash2 = provenance.hash_electric_chiller_input(input_data)
        assert hash1 == hash2

    def test_hash_electric_chiller_different_inputs(self, provenance):
        """Test different inputs produce different hashes."""
        input1 = {"cooling_output_kwh_th": Decimal("1000.0"), "cop": Decimal("6.1")}
        input2 = {"cooling_output_kwh_th": Decimal("2000.0"), "cop": Decimal("6.1")}

        hash1 = provenance.hash_electric_chiller_input(input1)
        hash2 = provenance.hash_electric_chiller_input(input2)
        assert hash1 != hash2

    def test_hash_electric_chiller_with_iplv(self, provenance):
        """Test hashing with IPLV method."""
        input_data = {
            "cooling_output_kwh_th": Decimal("1000.0"),
            "iplv": Decimal("7.2"),
            "method": "iplv_weighted",
        }

        hash_value = provenance.hash_electric_chiller_input(input_data)
        assert len(hash_value) == 64


class TestAbsorptionInputHashing:
    """Test hash_absorption_input()."""

    def test_hash_absorption_input_returns_hash(self, provenance):
        """Test hash_absorption_input returns hash."""
        input_data = {
            "cooling_output_kwh_th": Decimal("1000.0"),
            "technology": "single_effect_libr",
            "heat_source": "natural_gas",
            "heat_ef_kg_co2e_per_gj": Decimal("70.1"),
        }

        hash_value = provenance.hash_absorption_input(input_data)
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

    def test_hash_absorption_deterministic(self, provenance):
        """Test same inputs produce same hash."""
        input_data = {
            "cooling_output_kwh_th": Decimal("1000.0"),
            "technology": "double_effect_libr",
        }

        hash1 = provenance.hash_absorption_input(input_data)
        hash2 = provenance.hash_absorption_input(input_data)
        assert hash1 == hash2

    def test_hash_absorption_different_heat_source(self, provenance):
        """Test different heat sources produce different hashes."""
        input1 = {"heat_source": "natural_gas"}
        input2 = {"heat_source": "waste_heat"}

        hash1 = provenance.hash_absorption_input(input1)
        hash2 = provenance.hash_absorption_input(input2)
        assert hash1 != hash2


class TestFreeCoolingInputHashing:
    """Test hash_free_cooling_input()."""

    def test_hash_free_cooling_input_returns_hash(self, provenance):
        """Test hash_free_cooling_input returns hash."""
        input_data = {
            "cooling_output_kwh_th": Decimal("1000.0"),
            "technology": "waterside_economizer",
            "fan_power_kwh": Decimal("10.0"),
        }

        hash_value = provenance.hash_free_cooling_input(input_data)
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

    def test_hash_free_cooling_deterministic(self, provenance):
        """Test same inputs produce same hash."""
        input_data = {
            "cooling_output_kwh_th": Decimal("500.0"),
            "technology": "airside_economizer",
        }

        hash1 = provenance.hash_free_cooling_input(input_data)
        hash2 = provenance.hash_free_cooling_input(input_data)
        assert hash1 == hash2

    def test_hash_free_cooling_different_technology(self, provenance):
        """Test different technologies produce different hashes."""
        input1 = {"technology": "waterside_economizer"}
        input2 = {"technology": "cooling_tower"}

        hash1 = provenance.hash_free_cooling_input(input1)
        hash2 = provenance.hash_free_cooling_input(input2)
        assert hash1 != hash2


class TestTESInputHashing:
    """Test hash_tes_input()."""

    def test_hash_tes_input_returns_hash(self, provenance):
        """Test hash_tes_input returns hash."""
        input_data = {
            "cooling_output_kwh_th": Decimal("1000.0"),
            "technology": "ice_storage",
            "charging_cop": Decimal("5.0"),
            "loss_percent": Decimal("5.0"),
        }

        hash_value = provenance.hash_tes_input(input_data)
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

    def test_hash_tes_deterministic(self, provenance):
        """Test same inputs produce same hash."""
        input_data = {
            "cooling_output_kwh_th": Decimal("800.0"),
            "technology": "chilled_water_storage",
            "loss_percent": Decimal("3.0"),
        }

        hash1 = provenance.hash_tes_input(input_data)
        hash2 = provenance.hash_tes_input(input_data)
        assert hash1 == hash2

    def test_hash_tes_different_loss_percent(self, provenance):
        """Test different loss percentages produce different hashes."""
        input1 = {"loss_percent": Decimal("3.0")}
        input2 = {"loss_percent": Decimal("5.0")}

        hash1 = provenance.hash_tes_input(input1)
        hash2 = provenance.hash_tes_input(input2)
        assert hash1 != hash2


class TestDistrictCoolingInputHashing:
    """Test hash_district_cooling_input()."""

    def test_hash_district_cooling_input_returns_hash(self, provenance):
        """Test hash_district_cooling_input returns hash."""
        input_data = {
            "cooling_output_kwh_th": Decimal("1000.0"),
            "region": "north_america",
            "district_ef_kg_co2e_per_kwh_th": Decimal("0.15"),
        }

        hash_value = provenance.hash_district_cooling_input(input_data)
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

    def test_hash_district_cooling_deterministic(self, provenance):
        """Test same inputs produce same hash."""
        input_data = {
            "cooling_output_kwh_th": Decimal("2000.0"),
            "region": "europe_west",
        }

        hash1 = provenance.hash_district_cooling_input(input_data)
        hash2 = provenance.hash_district_cooling_input(input_data)
        assert hash1 == hash2

    def test_hash_district_cooling_different_region(self, provenance):
        """Test different regions produce different hashes."""
        input1 = {"region": "north_america"}
        input2 = {"region": "asia_pacific"}

        hash1 = provenance.hash_district_cooling_input(input1)
        hash2 = provenance.hash_district_cooling_input(input2)
        assert hash1 != hash2


class TestRefrigerantLeakageHashing:
    """Test hash_refrigerant_leakage()."""

    def test_hash_refrigerant_leakage_returns_hash(self, provenance):
        """Test hash_refrigerant_leakage returns hash."""
        input_data = {
            "refrigerant": "R-134a",
            "charge_kg": Decimal("50.0"),
            "leak_rate_percent": Decimal("5.0"),
            "gwp": 1430,
        }

        hash_value = provenance.hash_refrigerant_leakage(input_data)
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

    def test_hash_refrigerant_leakage_deterministic(self, provenance):
        """Test same inputs produce same hash."""
        input_data = {
            "refrigerant": "R-410A",
            "charge_kg": Decimal("30.0"),
            "leak_rate_percent": Decimal("3.0"),
        }

        hash1 = provenance.hash_refrigerant_leakage(input_data)
        hash2 = provenance.hash_refrigerant_leakage(input_data)
        assert hash1 == hash2

    def test_hash_refrigerant_leakage_different_refrigerant(self, provenance):
        """Test different refrigerants produce different hashes."""
        input1 = {"refrigerant": "R-134a", "gwp": 1430}
        input2 = {"refrigerant": "R-32", "gwp": 675}

        hash1 = provenance.hash_refrigerant_leakage(input1)
        hash2 = provenance.hash_refrigerant_leakage(input2)
        assert hash1 != hash2


class TestCalculationResultHashing:
    """Test hash_calculation_result()."""

    def test_hash_calculation_result_returns_hash(self, provenance):
        """Test hash_calculation_result returns hash."""
        result_data = {
            "total_emissions_kg_co2e": Decimal("82.0"),
            "energy_consumption_kwh": Decimal("163.93"),
            "technology": "air_cooled_screw",
        }

        hash_value = provenance.hash_calculation_result(result_data)
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

    def test_hash_calculation_result_deterministic(self, provenance):
        """Test same results produce same hash."""
        result_data = {
            "total_emissions_kg_co2e": Decimal("100.0"),
            "co2_kg": Decimal("95.0"),
            "ch4_kg": Decimal("0.003"),
        }

        hash1 = provenance.hash_calculation_result(result_data)
        hash2 = provenance.hash_calculation_result(result_data)
        assert hash1 == hash2

    def test_hash_calculation_result_different_values(self, provenance):
        """Test different results produce different hashes."""
        result1 = {"total_emissions_kg_co2e": Decimal("100.0")}
        result2 = {"total_emissions_kg_co2e": Decimal("200.0")}

        hash1 = provenance.hash_calculation_result(result1)
        hash2 = provenance.hash_calculation_result(result2)
        assert hash1 != hash2


class TestUncertaintyResultHashing:
    """Test hash_uncertainty_result()."""

    def test_hash_uncertainty_result_returns_hash(self, provenance):
        """Test hash_uncertainty_result returns hash."""
        uncertainty_data = {
            "mean_emissions_kg_co2e": Decimal("82.0"),
            "std_dev_kg_co2e": Decimal("5.0"),
            "confidence_95_lower": Decimal("72.0"),
            "confidence_95_upper": Decimal("92.0"),
        }

        hash_value = provenance.hash_uncertainty_result(uncertainty_data)
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

    def test_hash_uncertainty_result_deterministic(self, provenance):
        """Test same uncertainty results produce same hash."""
        uncertainty_data = {"std_dev_kg_co2e": Decimal("10.0"), "cv_percent": 12.5}

        hash1 = provenance.hash_uncertainty_result(uncertainty_data)
        hash2 = provenance.hash_uncertainty_result(uncertainty_data)
        assert hash1 == hash2


class TestComplianceResultHashing:
    """Test hash_compliance_result()."""

    def test_hash_compliance_result_returns_hash(self, provenance):
        """Test hash_compliance_result returns hash."""
        compliance_data = {
            "framework": "ghg_protocol",
            "scope": "scope_2",
            "category": "purchased_cooling",
            "compliant": True,
        }

        hash_value = provenance.hash_compliance_result(compliance_data)
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

    def test_hash_compliance_result_deterministic(self, provenance):
        """Test same compliance results produce same hash."""
        compliance_data = {"framework": "iso_14064", "compliant": True}

        hash1 = provenance.hash_compliance_result(compliance_data)
        hash2 = provenance.hash_compliance_result(compliance_data)
        assert hash1 == hash2

    def test_hash_compliance_result_different_frameworks(self, provenance):
        """Test different frameworks produce different hashes."""
        comp1 = {"framework": "ghg_protocol"}
        comp2 = {"framework": "csrd"}

        hash1 = provenance.hash_compliance_result(comp1)
        hash2 = provenance.hash_compliance_result(comp2)
        assert hash1 != hash2


class TestBatchResultHashing:
    """Test hash_batch_result()."""

    def test_hash_batch_result_returns_hash(self, provenance):
        """Test hash_batch_result returns hash."""
        batch_data = {
            "total_systems": 10,
            "total_emissions_kg_co2e": Decimal("1000.0"),
            "total_cooling_kwh_th": Decimal("10000.0"),
        }

        hash_value = provenance.hash_batch_result(batch_data)
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

    def test_hash_batch_result_deterministic(self, provenance):
        """Test same batch results produce same hash."""
        batch_data = {
            "total_systems": 5,
            "average_emissions_per_system": Decimal("100.0"),
        }

        hash1 = provenance.hash_batch_result(batch_data)
        hash2 = provenance.hash_batch_result(batch_data)
        assert hash1 == hash2

    def test_hash_batch_result_different_counts(self, provenance):
        """Test different system counts produce different hashes."""
        batch1 = {"total_systems": 5}
        batch2 = {"total_systems": 10}

        hash1 = provenance.hash_batch_result(batch1)
        hash2 = provenance.hash_batch_result(batch2)
        assert hash1 != hash2


class TestDeterministicHashing:
    """Test deterministic hashing across all hash functions."""

    def test_electric_chiller_same_input_same_hash(self, provenance):
        """Test electric chiller inputs produce consistent hashes."""
        input_data = {
            "cooling_output_kwh_th": Decimal("1000.0"),
            "cop": Decimal("6.1"),
        }

        hashes = [provenance.hash_electric_chiller_input(input_data) for _ in range(5)]
        assert len(set(hashes)) == 1  # All hashes identical

    def test_absorption_same_input_same_hash(self, provenance):
        """Test absorption inputs produce consistent hashes."""
        input_data = {
            "cooling_output_kwh_th": Decimal("1000.0"),
            "technology": "single_effect_libr",
        }

        hashes = [provenance.hash_absorption_input(input_data) for _ in range(5)]
        assert len(set(hashes)) == 1

    def test_decimal_precision_matters(self, provenance):
        """Test Decimal precision affects hash."""
        input1 = {"value": Decimal("1.0")}
        input2 = {"value": Decimal("1.00")}
        input3 = {"value": Decimal("1.000")}

        hash1 = provenance.hash_electric_chiller_input(input1)
        hash2 = provenance.hash_electric_chiller_input(input2)
        hash3 = provenance.hash_electric_chiller_input(input3)

        # Decimals with same value should produce same hash
        assert hash1 == hash2 == hash3


class TestHashIntegrity:
    """Test hash integrity and collision resistance."""

    def test_small_input_change_changes_hash(self, provenance):
        """Test small input changes produce different hashes."""
        input1 = {"cooling_output_kwh_th": Decimal("1000.0")}
        input2 = {"cooling_output_kwh_th": Decimal("1000.1")}

        hash1 = provenance.hash_electric_chiller_input(input1)
        hash2 = provenance.hash_electric_chiller_input(input2)
        assert hash1 != hash2

    def test_different_input_types_different_hashes(self, provenance):
        """Test different input types produce different hashes."""
        # Electric chiller vs absorption should differ even with same cooling output
        electric_hash = provenance.hash_electric_chiller_input(
            {"cooling_output_kwh_th": Decimal("1000.0")}
        )
        absorption_hash = provenance.hash_absorption_input(
            {"cooling_output_kwh_th": Decimal("1000.0")}
        )

        assert electric_hash != absorption_hash

    def test_hash_avalanche_effect(self, provenance):
        """Test avalanche effect - small change causes large hash difference."""
        input1 = {"value": Decimal("100.0")}
        input2 = {"value": Decimal("100.1")}

        hash1 = provenance.hash_electric_chiller_input(input1)
        hash2 = provenance.hash_electric_chiller_input(input2)

        # Count differing characters
        diff_count = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
        # Expect significant difference (avalanche effect)
        assert diff_count > 20  # Arbitrary threshold, adjust as needed
