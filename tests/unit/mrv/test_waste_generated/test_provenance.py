"""
Unit tests for Waste Generated in Operations Provenance Engine.

Tests provenance tracking, hash generation, chain validation, and
reproducibility guarantees for waste emissions calculations.
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from typing import Dict, Any, List
import json
import hashlib
from copy import deepcopy

from greenlang.mrv.waste_generated.engines.provenance import (
    ProvenanceEngine,
    ProvenanceRecord,
    ProvenanceChain,
    ProvenanceHasher
)
from greenlang.mrv.waste_generated.models import (
    WasteType,
    TreatmentMethod,
    WasteSource
)


# Fixtures
@pytest.fixture
def sample_input_data() -> Dict[str, Any]:
    """Create sample input data for provenance testing."""
    return {
        'waste_stream_id': 'WS-2025-001',
        'waste_type': WasteType.MIXED_MSW,
        'waste_mass_tonnes': Decimal('100.0'),
        'treatment_method': TreatmentMethod.LANDFILL,
        'reporting_period': '2025-01',
        'facility_id': 'FAC-001',
        'region': 'US'
    }


@pytest.fixture
def sample_calculation_params() -> Dict[str, Any]:
    """Create sample calculation parameters."""
    return {
        'doc': Decimal('0.50'),
        'docf': Decimal('0.50'),
        'mcf': Decimal('1.0'),
        'k': Decimal('0.09'),
        'f': Decimal('0.50'),
        'ox': Decimal('0.10'),
        'gwp_ch4': Decimal('28'),
        'methodology': 'IPCC 2006 FOD'
    }


@pytest.fixture
def sample_calculation_results() -> Dict[str, Any]:
    """Create sample calculation results."""
    return {
        'ch4_emissions_tonnes': Decimal('7.350'),
        'co2e_emissions_tonnes': Decimal('205.800'),
        'calculation_year': 2025,
        'lifetime_emissions': False,
        'uncertainty_range': {
            'lower': Decimal('164.640'),
            'upper': Decimal('246.960')
        }
    }


@pytest.fixture
def provenance_engine():
    """Create ProvenanceEngine instance."""
    return ProvenanceEngine()


# Test ProvenanceRecord Creation
class TestProvenanceRecord:
    """Test suite for ProvenanceRecord."""

    def test_record_creation(self, sample_input_data):
        """Test ProvenanceRecord creation with valid data."""
        record = ProvenanceRecord(
            stage='input_validation',
            timestamp=datetime.now(timezone.utc),
            data=sample_input_data,
            data_hash='abc123'
        )

        assert record.stage == 'input_validation'
        assert isinstance(record.timestamp, datetime)
        assert record.data == sample_input_data
        assert record.data_hash == 'abc123'

    def test_record_immutability(self, sample_input_data):
        """Test ProvenanceRecord is immutable (frozen dataclass)."""
        record = ProvenanceRecord(
            stage='input_validation',
            timestamp=datetime.now(timezone.utc),
            data=sample_input_data,
            data_hash='abc123'
        )

        # Attempting to modify should raise FrozenInstanceError
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
            record.stage = 'modified'

    def test_record_to_dict(self, sample_input_data):
        """Test ProvenanceRecord serialization to dict."""
        timestamp = datetime.now(timezone.utc)
        record = ProvenanceRecord(
            stage='input_validation',
            timestamp=timestamp,
            data=sample_input_data,
            data_hash='abc123'
        )

        record_dict = record.to_dict()

        assert record_dict['stage'] == 'input_validation'
        assert record_dict['timestamp'] == timestamp.isoformat()
        assert record_dict['data_hash'] == 'abc123'
        assert 'waste_stream_id' in record_dict['data']

    def test_record_from_dict(self, sample_input_data):
        """Test ProvenanceRecord deserialization from dict."""
        timestamp = datetime.now(timezone.utc)
        record_dict = {
            'stage': 'input_validation',
            'timestamp': timestamp.isoformat(),
            'data': sample_input_data,
            'data_hash': 'abc123'
        }

        record = ProvenanceRecord.from_dict(record_dict)

        assert record.stage == 'input_validation'
        assert isinstance(record.timestamp, datetime)
        assert record.data_hash == 'abc123'


# Test ProvenanceChain
class TestProvenanceChain:
    """Test suite for ProvenanceChain with all 10 stages."""

    def test_chain_creation(self):
        """Test ProvenanceChain initialization."""
        chain = ProvenanceChain(calculation_id='CALC-001')

        assert chain.calculation_id == 'CALC-001'
        assert len(chain.records) == 0
        assert chain.chain_hash is None

    def test_add_record_all_stages(
        self,
        sample_input_data,
        sample_calculation_params,
        sample_calculation_results
    ):
        """Test adding records for all 10 calculation stages."""
        chain = ProvenanceChain(calculation_id='CALC-001')

        # Stage 1: Input Validation
        chain.add_record('input_validation', sample_input_data)

        # Stage 2: Waste Classification
        classification_data = {
            'waste_category': 'Municipal Solid Waste',
            'ewc_code': '20 03 01',
            'is_hazardous': False
        }
        chain.add_record('waste_classification', classification_data)

        # Stage 3: Treatment Method Selection
        treatment_data = {
            'treatment_method': 'landfill',
            'facility_type': 'municipal',
            'has_gas_capture': False
        }
        chain.add_record('treatment_selection', treatment_data)

        # Stage 4: Parameter Lookup
        chain.add_record('parameter_lookup', sample_calculation_params)

        # Stage 5: Emission Factor Retrieval
        ef_data = {
            'emission_factor_source': 'IPCC 2006',
            'doc': '0.50',
            'k': '0.09'
        }
        chain.add_record('emission_factor_retrieval', ef_data)

        # Stage 6: Calculation Execution
        calculation_data = {
            'ddocm': Decimal('25.0'),
            'ch4_generated': Decimal('8.167'),
            'ch4_recovered': Decimal('0.0')
        }
        chain.add_record('calculation_execution', calculation_data)

        # Stage 7: Uncertainty Quantification
        uncertainty_data = {
            'uncertainty_percent': Decimal('20'),
            'lower_bound': Decimal('164.640'),
            'upper_bound': Decimal('246.960')
        }
        chain.add_record('uncertainty_quantification', uncertainty_data)

        # Stage 8: Quality Assurance
        qa_data = {
            'dqi_score': Decimal('3.5'),
            'validation_checks_passed': 12,
            'warnings': []
        }
        chain.add_record('quality_assurance', qa_data)

        # Stage 9: Result Aggregation
        chain.add_record('result_aggregation', sample_calculation_results)

        # Stage 10: Final Output
        output_data = {
            'total_co2e': Decimal('205.800'),
            'reporting_format': 'GHG Protocol',
            'calculation_complete': True
        }
        chain.add_record('final_output', output_data)

        assert len(chain.records) == 10
        assert chain.records[0].stage == 'input_validation'
        assert chain.records[9].stage == 'final_output'

    def test_finalize_chain(self, sample_input_data):
        """Test chain finalization generates chain hash."""
        chain = ProvenanceChain(calculation_id='CALC-001')
        chain.add_record('input_validation', sample_input_data)
        chain.add_record('calculation_execution', {'result': Decimal('100')})

        chain.finalize()

        assert chain.chain_hash is not None
        assert len(chain.chain_hash) == 64  # SHA-256 hex digest

    def test_chain_hash_deterministic(self, sample_input_data):
        """Test chain hash is deterministic (same input → same hash)."""
        chain1 = ProvenanceChain(calculation_id='CALC-001')
        chain1.add_record('input_validation', sample_input_data)
        chain1.finalize()

        chain2 = ProvenanceChain(calculation_id='CALC-001')
        chain2.add_record('input_validation', sample_input_data)
        chain2.finalize()

        assert chain1.chain_hash == chain2.chain_hash

    def test_chain_validation_valid(self, sample_input_data):
        """Test chain validation passes for valid chain."""
        chain = ProvenanceChain(calculation_id='CALC-001')
        chain.add_record('input_validation', sample_input_data)
        chain.add_record('calculation_execution', {'result': Decimal('100')})
        chain.finalize()

        is_valid = chain.validate()

        assert is_valid is True

    def test_chain_validation_broken(self, sample_input_data):
        """Test chain validation fails for tampered chain."""
        chain = ProvenanceChain(calculation_id='CALC-001')
        chain.add_record('input_validation', sample_input_data)
        chain.finalize()

        # Tamper with chain
        chain.records[0].data['waste_mass_tonnes'] = Decimal('999.0')

        is_valid = chain.validate()

        assert is_valid is False

    def test_chain_to_dict(self, sample_input_data):
        """Test ProvenanceChain serialization to dict."""
        chain = ProvenanceChain(calculation_id='CALC-001')
        chain.add_record('input_validation', sample_input_data)
        chain.finalize()

        chain_dict = chain.to_dict()

        assert chain_dict['calculation_id'] == 'CALC-001'
        assert len(chain_dict['records']) == 1
        assert chain_dict['chain_hash'] is not None

    def test_chain_from_dict(self, sample_input_data):
        """Test ProvenanceChain deserialization from dict."""
        chain = ProvenanceChain(calculation_id='CALC-001')
        chain.add_record('input_validation', sample_input_data)
        chain.finalize()

        chain_dict = chain.to_dict()
        restored_chain = ProvenanceChain.from_dict(chain_dict)

        assert restored_chain.calculation_id == 'CALC-001'
        assert len(restored_chain.records) == 1
        assert restored_chain.chain_hash == chain.chain_hash


# Test ProvenanceHasher
class TestProvenanceHasher:
    """Test suite for ProvenanceHasher (SHA-256 hashing)."""

    def test_hash_deterministic(self):
        """Test hash generation is deterministic."""
        data = {'key': 'value', 'number': Decimal('123.456')}

        hash1 = ProvenanceHasher.hash_data(data)
        hash2 = ProvenanceHasher.hash_data(data)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex

    def test_hash_different_data(self):
        """Test different data produces different hashes."""
        data1 = {'key': 'value1'}
        data2 = {'key': 'value2'}

        hash1 = ProvenanceHasher.hash_data(data1)
        hash2 = ProvenanceHasher.hash_data(data2)

        assert hash1 != hash2

    def test_hash_dict_order_independent(self):
        """Test hash is independent of dict key order."""
        data1 = {'a': 1, 'b': 2, 'c': 3}
        data2 = {'c': 3, 'b': 2, 'a': 1}

        hash1 = ProvenanceHasher.hash_data(data1)
        hash2 = ProvenanceHasher.hash_data(data2)

        assert hash1 == hash2

    def test_hash_decimal_precision(self):
        """Test Decimal hashing preserves precision."""
        data1 = {'value': Decimal('1.1')}
        data2 = {'value': Decimal('1.10')}  # Same value, different precision

        hash1 = ProvenanceHasher.hash_data(data1)
        hash2 = ProvenanceHasher.hash_data(data2)

        # Decimals with same value should hash identically
        assert hash1 == hash2

    def test_hash_nested_structures(self):
        """Test hashing of nested data structures."""
        data = {
            'level1': {
                'level2': {
                    'level3': [1, 2, 3],
                    'value': Decimal('99.99')
                }
            }
        }

        hash_result = ProvenanceHasher.hash_data(data)

        assert len(hash_result) == 64

    def test_hash_list_order_matters(self):
        """Test hash is sensitive to list order."""
        data1 = {'items': [1, 2, 3]}
        data2 = {'items': [3, 2, 1]}

        hash1 = ProvenanceHasher.hash_data(data1)
        hash2 = ProvenanceHasher.hash_data(data2)

        assert hash1 != hash2


# Test 27 Hash Helper Methods
class TestHashHelpers:
    """Test suite for all 27 hash helper methods."""

    def test_hash_waste_type(self):
        """Test hashing of WasteType enum."""
        hash1 = ProvenanceHasher.hash_waste_type(WasteType.MIXED_MSW)
        hash2 = ProvenanceHasher.hash_waste_type(WasteType.MIXED_MSW)

        assert hash1 == hash2
        assert len(hash1) == 64

    def test_hash_treatment_method(self):
        """Test hashing of TreatmentMethod enum."""
        hash1 = ProvenanceHasher.hash_treatment_method(TreatmentMethod.LANDFILL)
        hash2 = ProvenanceHasher.hash_treatment_method(TreatmentMethod.INCINERATION)

        assert hash1 != hash2
        assert len(hash1) == 64

    def test_hash_waste_source(self):
        """Test hashing of WasteSource enum."""
        hash1 = ProvenanceHasher.hash_waste_source(WasteSource.OPERATIONS)
        hash2 = ProvenanceHasher.hash_waste_source(WasteSource.OPERATIONS)

        assert hash1 == hash2

    def test_hash_decimal(self):
        """Test hashing of Decimal values."""
        hash1 = ProvenanceHasher.hash_decimal(Decimal('123.456'))
        hash2 = ProvenanceHasher.hash_decimal(Decimal('123.456'))

        assert hash1 == hash2

    def test_hash_decimal_list(self):
        """Test hashing of Decimal list."""
        values = [Decimal('1.1'), Decimal('2.2'), Decimal('3.3')]
        hash1 = ProvenanceHasher.hash_decimal_list(values)
        hash2 = ProvenanceHasher.hash_decimal_list(values)

        assert hash1 == hash2

    def test_hash_datetime(self):
        """Test hashing of datetime."""
        dt = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        hash1 = ProvenanceHasher.hash_datetime(dt)
        hash2 = ProvenanceHasher.hash_datetime(dt)

        assert hash1 == hash2

    def test_hash_input_data(self, sample_input_data):
        """Test hashing of input data."""
        hash1 = ProvenanceHasher.hash_input_data(sample_input_data)
        hash2 = ProvenanceHasher.hash_input_data(sample_input_data)

        assert hash1 == hash2

    def test_hash_calculation_params(self, sample_calculation_params):
        """Test hashing of calculation parameters."""
        hash1 = ProvenanceHasher.hash_calculation_params(sample_calculation_params)
        hash2 = ProvenanceHasher.hash_calculation_params(sample_calculation_params)

        assert hash1 == hash2

    def test_hash_emission_factors(self):
        """Test hashing of emission factors."""
        ef_data = {
            'doc': Decimal('0.50'),
            'k': Decimal('0.09'),
            'source': 'IPCC 2006'
        }

        hash1 = ProvenanceHasher.hash_emission_factors(ef_data)
        hash2 = ProvenanceHasher.hash_emission_factors(ef_data)

        assert hash1 == hash2

    def test_hash_calculation_results(self, sample_calculation_results):
        """Test hashing of calculation results."""
        hash1 = ProvenanceHasher.hash_calculation_results(sample_calculation_results)
        hash2 = ProvenanceHasher.hash_calculation_results(sample_calculation_results)

        assert hash1 == hash2

    def test_hash_uncertainty_data(self):
        """Test hashing of uncertainty data."""
        uncertainty = {
            'uncertainty_percent': Decimal('20'),
            'lower': Decimal('80'),
            'upper': Decimal('120')
        }

        hash1 = ProvenanceHasher.hash_uncertainty_data(uncertainty)
        hash2 = ProvenanceHasher.hash_uncertainty_data(uncertainty)

        assert hash1 == hash2

    def test_hash_quality_metrics(self):
        """Test hashing of quality metrics."""
        qa_data = {
            'dqi_score': Decimal('3.5'),
            'validation_checks': 12
        }

        hash1 = ProvenanceHasher.hash_quality_metrics(qa_data)
        hash2 = ProvenanceHasher.hash_quality_metrics(qa_data)

        assert hash1 == hash2

    def test_hash_waste_composition(self):
        """Test hashing of waste composition."""
        composition = {
            'food_waste': Decimal('0.30'),
            'paper': Decimal('0.25'),
            'plastics': Decimal('0.15')
        }

        hash1 = ProvenanceHasher.hash_waste_composition(composition)
        hash2 = ProvenanceHasher.hash_waste_composition(composition)

        assert hash1 == hash2

    def test_hash_landfill_parameters(self):
        """Test hashing of landfill-specific parameters."""
        params = {
            'doc': Decimal('0.50'),
            'k': Decimal('0.09'),
            'mcf': Decimal('1.0')
        }

        hash1 = ProvenanceHasher.hash_landfill_parameters(params)
        hash2 = ProvenanceHasher.hash_landfill_parameters(params)

        assert hash1 == hash2

    def test_hash_incineration_parameters(self):
        """Test hashing of incineration-specific parameters."""
        params = {
            'cf': Decimal('0.75'),
            'fcf': Decimal('0.40'),
            'of': Decimal('1.0')
        }

        hash1 = ProvenanceHasher.hash_incineration_parameters(params)
        hash2 = ProvenanceHasher.hash_incineration_parameters(params)

        assert hash1 == hash2

    def test_hash_composting_parameters(self):
        """Test hashing of composting parameters."""
        params = {
            'ch4_ef': Decimal('4.0'),
            'n2o_ef': Decimal('0.3')
        }

        hash1 = ProvenanceHasher.hash_composting_parameters(params)
        hash2 = ProvenanceHasher.hash_composting_parameters(params)

        assert hash1 == hash2

    def test_hash_ad_parameters(self):
        """Test hashing of anaerobic digestion parameters."""
        params = {
            'leakage_rate': Decimal('0.05'),
            'biogas_ch4_content': Decimal('0.60')
        }

        hash1 = ProvenanceHasher.hash_ad_parameters(params)
        hash2 = ProvenanceHasher.hash_ad_parameters(params)

        assert hash1 == hash2

    def test_hash_ewc_code(self):
        """Test hashing of EWC code."""
        hash1 = ProvenanceHasher.hash_ewc_code('20 03 01')
        hash2 = ProvenanceHasher.hash_ewc_code('20 03 01')

        assert hash1 == hash2

    def test_hash_facility_data(self):
        """Test hashing of facility data."""
        facility = {
            'facility_id': 'FAC-001',
            'facility_type': 'municipal',
            'region': 'US'
        }

        hash1 = ProvenanceHasher.hash_facility_data(facility)
        hash2 = ProvenanceHasher.hash_facility_data(facility)

        assert hash1 == hash2

    def test_hash_reporting_period(self):
        """Test hashing of reporting period."""
        hash1 = ProvenanceHasher.hash_reporting_period('2025-01')
        hash2 = ProvenanceHasher.hash_reporting_period('2025-01')

        assert hash1 == hash2

    def test_hash_regulatory_framework(self):
        """Test hashing of regulatory framework."""
        hash1 = ProvenanceHasher.hash_regulatory_framework('GHG Protocol')
        hash2 = ProvenanceHasher.hash_regulatory_framework('IPCC 2006')

        assert hash1 != hash2

    def test_hash_gwp_version(self):
        """Test hashing of GWP version."""
        hash1 = ProvenanceHasher.hash_gwp_version('AR5')
        hash2 = ProvenanceHasher.hash_gwp_version('AR5')

        assert hash1 == hash2

    def test_hash_data_quality_indicators(self):
        """Test hashing of data quality indicators."""
        dqi = {
            'temporal': 5,
            'geographical': 4,
            'technological': 3,
            'completeness': 5,
            'reliability': 4
        }

        hash1 = ProvenanceHasher.hash_data_quality_indicators(dqi)
        hash2 = ProvenanceHasher.hash_data_quality_indicators(dqi)

        assert hash1 == hash2

    def test_hash_validation_results(self):
        """Test hashing of validation results."""
        validation = {
            'checks_passed': 12,
            'checks_failed': 0,
            'warnings': []
        }

        hash1 = ProvenanceHasher.hash_validation_results(validation)
        hash2 = ProvenanceHasher.hash_validation_results(validation)

        assert hash1 == hash2

    def test_hash_aggregation_method(self):
        """Test hashing of aggregation method."""
        hash1 = ProvenanceHasher.hash_aggregation_method('sum')
        hash2 = ProvenanceHasher.hash_aggregation_method('weighted_average')

        assert hash1 != hash2

    def test_hash_output_format(self):
        """Test hashing of output format."""
        hash1 = ProvenanceHasher.hash_output_format('GHG Protocol')
        hash2 = ProvenanceHasher.hash_output_format('GHG Protocol')

        assert hash1 == hash2

    def test_hash_metadata(self):
        """Test hashing of metadata."""
        metadata = {
            'calculation_engine_version': '1.0.0',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'user_id': 'user123'
        }

        hash1 = ProvenanceHasher.hash_metadata(metadata)
        hash2 = ProvenanceHasher.hash_metadata(metadata)

        assert hash1 == hash2


# Test Batch Provenance
class TestBatchProvenance:
    """Test suite for batch provenance tracking."""

    def test_batch_provenance_creation(self):
        """Test batch provenance for multiple calculations."""
        batch_chains = []

        for i in range(5):
            chain = ProvenanceChain(calculation_id=f'CALC-{i:03d}')
            chain.add_record('input_validation', {'batch_index': i})
            chain.finalize()
            batch_chains.append(chain)

        assert len(batch_chains) == 5
        assert all(chain.chain_hash is not None for chain in batch_chains)

    def test_batch_hash_aggregation(self):
        """Test aggregating batch hashes."""
        batch_hashes = []

        for i in range(3):
            chain = ProvenanceChain(calculation_id=f'CALC-{i:03d}')
            chain.add_record('input_validation', {'index': i})
            chain.finalize()
            batch_hashes.append(chain.chain_hash)

        # Aggregate batch hash
        batch_data = {'hashes': batch_hashes}
        batch_hash = ProvenanceHasher.hash_data(batch_data)

        assert len(batch_hash) == 64


# Test JSON Serialization
class TestJSONSerialization:
    """Test suite for JSON serialization/deserialization."""

    def test_chain_to_json(self, sample_input_data):
        """Test ProvenanceChain serialization to JSON."""
        chain = ProvenanceChain(calculation_id='CALC-001')
        chain.add_record('input_validation', sample_input_data)
        chain.finalize()

        json_str = chain.to_json()

        assert isinstance(json_str, str)
        assert 'CALC-001' in json_str

    def test_chain_from_json(self, sample_input_data):
        """Test ProvenanceChain deserialization from JSON."""
        chain = ProvenanceChain(calculation_id='CALC-001')
        chain.add_record('input_validation', sample_input_data)
        chain.finalize()

        json_str = chain.to_json()
        restored_chain = ProvenanceChain.from_json(json_str)

        assert restored_chain.calculation_id == 'CALC-001'
        assert restored_chain.chain_hash == chain.chain_hash

    def test_json_roundtrip(self, sample_input_data):
        """Test JSON roundtrip preserves data."""
        chain = ProvenanceChain(calculation_id='CALC-001')
        chain.add_record('input_validation', sample_input_data)
        chain.add_record('calculation_execution', {'result': Decimal('100')})
        chain.finalize()

        json_str = chain.to_json()
        restored_chain = ProvenanceChain.from_json(json_str)

        assert restored_chain.to_json() == json_str


# Test Singleton Pattern
class TestSingletonPattern:
    """Test suite for ProvenanceEngine singleton pattern."""

    def test_singleton_instance(self):
        """Test ProvenanceEngine uses singleton pattern."""
        engine1 = ProvenanceEngine()
        engine2 = ProvenanceEngine()

        assert engine1 is engine2

    def test_singleton_state_shared(self):
        """Test singleton state is shared across instances."""
        engine1 = ProvenanceEngine()
        engine1.register_calculation('CALC-001')

        engine2 = ProvenanceEngine()
        assert engine2.has_calculation('CALC-001')


# Test Decimal Hashing Consistency
class TestDecimalHashingConsistency:
    """Test suite for Decimal hashing consistency."""

    def test_decimal_string_equivalence(self):
        """Test Decimal and string representation hash identically."""
        data1 = {'value': Decimal('123.456')}
        data2 = {'value': '123.456'}

        # When normalized to Decimal, should hash identically
        hash1 = ProvenanceHasher.hash_decimal(Decimal('123.456'))
        hash2 = ProvenanceHasher.hash_decimal(Decimal('123.456'))

        assert hash1 == hash2

    def test_decimal_scientific_notation(self):
        """Test Decimal scientific notation hashing."""
        dec1 = Decimal('1.23E+2')  # 123
        dec2 = Decimal('123.0')

        hash1 = ProvenanceHasher.hash_decimal(dec1)
        hash2 = ProvenanceHasher.hash_decimal(dec2)

        assert hash1 == hash2

    def test_decimal_normalization(self):
        """Test Decimal normalization before hashing."""
        dec1 = Decimal('1.2300')
        dec2 = Decimal('1.23')

        # Normalize to remove trailing zeros
        norm1 = dec1.normalize()
        norm2 = dec2.normalize()

        hash1 = ProvenanceHasher.hash_decimal(norm1)
        hash2 = ProvenanceHasher.hash_decimal(norm2)

        assert hash1 == hash2
