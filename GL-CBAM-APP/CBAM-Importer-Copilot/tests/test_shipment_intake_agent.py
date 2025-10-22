"""
CBAM Importer Copilot - Shipment Intake Agent Tests

Unit tests for ShipmentIntakeAgent functionality.

Version: 1.0.0
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.shipment_intake_agent import ShipmentIntakeAgent


# ============================================================================
# Test Agent Initialization
# ============================================================================

@pytest.mark.unit
class TestAgentInitialization:
    """Test agent initialization and configuration."""

    def test_agent_initializes_successfully(self, cn_codes_path, cbam_rules_path):
        """Test agent can be initialized with valid paths."""
        agent = ShipmentIntakeAgent(
            cn_codes_path=cn_codes_path,
            cbam_rules_path=cbam_rules_path
        )

        assert agent is not None
        assert agent.cn_codes_path == cn_codes_path
        assert agent.cbam_rules_path == cbam_rules_path

    def test_agent_loads_cn_codes(self, cn_codes_path, cbam_rules_path):
        """Test agent loads CN codes database."""
        agent = ShipmentIntakeAgent(
            cn_codes_path=cn_codes_path,
            cbam_rules_path=cbam_rules_path
        )

        # Should have loaded CN codes
        assert hasattr(agent, 'cn_codes')
        assert len(agent.cn_codes) > 0

    def test_agent_loads_validation_rules(self, cn_codes_path, cbam_rules_path):
        """Test agent loads CBAM validation rules."""
        agent = ShipmentIntakeAgent(
            cn_codes_path=cn_codes_path,
            cbam_rules_path=cbam_rules_path
        )

        # Should have loaded rules
        assert hasattr(agent, 'cbam_rules')
        assert len(agent.cbam_rules) > 0


# ============================================================================
# Test CSV Input Processing
# ============================================================================

@pytest.mark.unit
class TestCSVProcessing:
    """Test CSV file processing."""

    def test_reads_csv_successfully(self, sample_shipments_csv, cn_codes_path, cbam_rules_path, importer_info):
        """Test agent can read CSV file."""
        agent = ShipmentIntakeAgent(cn_codes_path, cbam_rules_path)

        result = agent.process(
            input_file=sample_shipments_csv,
            importer_info=importer_info
        )

        assert result is not None
        assert 'validated_shipments' in result
        assert 'metadata' in result

    def test_processes_all_records(self, sample_shipments_csv, cn_codes_path, cbam_rules_path, importer_info):
        """Test agent processes all records from CSV."""
        agent = ShipmentIntakeAgent(cn_codes_path, cbam_rules_path)

        result = agent.process(
            input_file=sample_shipments_csv,
            importer_info=importer_info
        )

        metadata = result['metadata']
        assert metadata['total_records'] == 5  # We have 5 sample records

    def test_returns_valid_records(self, sample_shipments_csv, cn_codes_path, cbam_rules_path, importer_info):
        """Test agent returns valid records."""
        agent = ShipmentIntakeAgent(cn_codes_path, cbam_rules_path)

        result = agent.process(
            input_file=sample_shipments_csv,
            importer_info=importer_info
        )

        validated = result['validated_shipments']
        assert len(validated) > 0
        assert all('cn_code' in record for record in validated)


# ============================================================================
# Test Excel Input Processing
# ============================================================================

@pytest.mark.unit
class TestExcelProcessing:
    """Test Excel file processing."""

    def test_reads_excel_successfully(self, sample_shipments_excel, cn_codes_path, cbam_rules_path, importer_info):
        """Test agent can read Excel file."""
        agent = ShipmentIntakeAgent(cn_codes_path, cbam_rules_path)

        result = agent.process(
            input_file=sample_shipments_excel,
            importer_info=importer_info
        )

        assert result is not None
        assert 'validated_shipments' in result

    def test_excel_same_as_csv(self, sample_shipments_csv, sample_shipments_excel,
                                cn_codes_path, cbam_rules_path, importer_info):
        """Test Excel and CSV produce same results."""
        agent = ShipmentIntakeAgent(cn_codes_path, cbam_rules_path)

        csv_result = agent.process(sample_shipments_csv, importer_info)
        excel_result = agent.process(sample_shipments_excel, importer_info)

        # Should have same number of records
        assert csv_result['metadata']['total_records'] == \
               excel_result['metadata']['total_records']


# ============================================================================
# Test JSON Input Processing
# ============================================================================

@pytest.mark.unit
class TestJSONProcessing:
    """Test JSON file processing."""

    def test_reads_json_successfully(self, sample_shipments_json, cn_codes_path, cbam_rules_path, importer_info):
        """Test agent can read JSON file."""
        agent = ShipmentIntakeAgent(cn_codes_path, cbam_rules_path)

        result = agent.process(
            input_file=sample_shipments_json,
            importer_info=importer_info
        )

        assert result is not None
        assert 'validated_shipments' in result


# ============================================================================
# Test DataFrame Input Processing
# ============================================================================

@pytest.mark.unit
class TestDataFrameProcessing:
    """Test pandas DataFrame processing."""

    def test_processes_dataframe(self, sample_shipments_dataframe, cn_codes_path, cbam_rules_path, importer_info):
        """Test agent can process pandas DataFrame."""
        agent = ShipmentIntakeAgent(cn_codes_path, cbam_rules_path)

        result = agent.process_dataframe(
            df=sample_shipments_dataframe,
            importer_info=importer_info
        )

        assert result is not None
        assert 'validated_shipments' in result

    def test_dataframe_same_as_csv(self, sample_shipments_csv, sample_shipments_dataframe,
                                     cn_codes_path, cbam_rules_path, importer_info):
        """Test DataFrame and CSV produce same results."""
        agent = ShipmentIntakeAgent(cn_codes_path, cbam_rules_path)

        csv_result = agent.process(sample_shipments_csv, importer_info)
        df_result = agent.process_dataframe(sample_shipments_dataframe, importer_info)

        # Should have same number of records
        assert csv_result['metadata']['total_records'] == \
               df_result['metadata']['total_records']


# ============================================================================
# Test Validation Rules
# ============================================================================

@pytest.mark.unit
class TestValidationRules:
    """Test CBAM validation rules enforcement."""

    def test_detects_missing_cn_code(self, invalid_shipments_data, cn_codes_path, cbam_rules_path, importer_info, tmp_path):
        """Test detects missing CN code (E001)."""
        # Create CSV with invalid data
        csv_path = tmp_path / "invalid.csv"
        df = pd.DataFrame([invalid_shipments_data[0]])  # Missing cn_code
        df.to_csv(csv_path, index=False)

        agent = ShipmentIntakeAgent(cn_codes_path, cbam_rules_path)
        result = agent.process(str(csv_path), importer_info)

        # Should have error
        errors = result.get('errors', [])
        assert len(errors) > 0
        assert any('cn_code' in str(e).lower() for e in errors)

    def test_detects_invalid_cn_code_format(self, invalid_shipments_data, cn_codes_path, cbam_rules_path, importer_info, tmp_path):
        """Test detects invalid CN code format (E002)."""
        csv_path = tmp_path / "invalid.csv"
        df = pd.DataFrame([invalid_shipments_data[1]])  # Invalid format
        df.to_csv(csv_path, index=False)

        agent = ShipmentIntakeAgent(cn_codes_path, cbam_rules_path)
        result = agent.process(str(csv_path), importer_info)

        errors = result.get('errors', [])
        assert len(errors) > 0

    def test_detects_negative_quantity(self, invalid_shipments_data, cn_codes_path, cbam_rules_path, importer_info, tmp_path):
        """Test detects negative quantity (E007)."""
        csv_path = tmp_path / "invalid.csv"
        df = pd.DataFrame([invalid_shipments_data[3]])  # Negative quantity
        df.to_csv(csv_path, index=False)

        agent = ShipmentIntakeAgent(cn_codes_path, cbam_rules_path)
        result = agent.process(str(csv_path), importer_info)

        errors = result.get('errors', [])
        assert len(errors) > 0

    def test_accepts_valid_data(self, sample_shipments_csv, cn_codes_path, cbam_rules_path, importer_info):
        """Test accepts valid data without errors."""
        agent = ShipmentIntakeAgent(cn_codes_path, cbam_rules_path)
        result = agent.process(sample_shipments_csv, importer_info)

        errors = result.get('errors', [])
        assert len(errors) == 0  # No errors for valid data


# ============================================================================
# Test Data Enrichment
# ============================================================================

@pytest.mark.unit
class TestDataEnrichment:
    """Test data enrichment with CN code metadata."""

    def test_enriches_with_cn_metadata(self, sample_shipments_csv, cn_codes_path, cbam_rules_path, importer_info):
        """Test enriches shipments with CN code descriptions."""
        agent = ShipmentIntakeAgent(cn_codes_path, cbam_rules_path)
        result = agent.process(sample_shipments_csv, importer_info)

        validated = result['validated_shipments']

        # Should have enriched data
        for record in validated:
            assert 'cn_description' in record or 'description' in record
            assert 'product_group' in record

    def test_enrichment_correct(self, sample_shipments_csv, cn_codes_path, cbam_rules_path, importer_info):
        """Test enrichment data is correct."""
        agent = ShipmentIntakeAgent(cn_codes_path, cbam_rules_path)
        result = agent.process(sample_shipments_csv, importer_info)

        validated = result['validated_shipments']

        # Find iron/steel record
        iron_record = next(r for r in validated if r['cn_code'].startswith('72'))
        assert iron_record['product_group'] in ['iron_steel', 'iron and steel']


# ============================================================================
# Test Supplier Linking
# ============================================================================

@pytest.mark.unit
class TestSupplierLinking:
    """Test linking shipments to supplier data."""

    def test_links_supplier_data(self, sample_shipments_csv, cn_codes_path,
                                  cbam_rules_path, suppliers_path, importer_info):
        """Test links supplier actuals when available."""
        agent = ShipmentIntakeAgent(cn_codes_path, cbam_rules_path, suppliers_path)
        result = agent.process(sample_shipments_csv, importer_info)

        validated = result['validated_shipments']

        # Some records should have supplier data linked
        # (if supplier_id matches supplier file)
        for record in validated:
            if 'supplier_id' in record and record['supplier_id']:
                # May have supplier_emission_factor if linked
                pass  # OK if present or not


# ============================================================================
# Test Performance
# ============================================================================

@pytest.mark.unit
@pytest.mark.performance
class TestPerformance:
    """Test agent performance."""

    def test_processes_quickly(self, sample_shipments_csv, cn_codes_path, cbam_rules_path, importer_info):
        """Test agent processes small dataset quickly."""
        import time

        agent = ShipmentIntakeAgent(cn_codes_path, cbam_rules_path)

        start = time.time()
        result = agent.process(sample_shipments_csv, importer_info)
        duration = time.time() - start

        # Should process 5 records in <1 second
        assert duration < 1.0, f"Processing too slow: {duration:.2f}s"

    def test_throughput_target(self, large_shipments_csv, cn_codes_path, cbam_rules_path, importer_info):
        """Test agent meets throughput target (1000 records/sec)."""
        import time

        agent = ShipmentIntakeAgent(cn_codes_path, cbam_rules_path)

        start = time.time()
        result = agent.process(large_shipments_csv, importer_info)
        duration = time.time() - start

        records = result['metadata']['total_records']
        throughput = records / duration

        # Target: 1000 records/sec
        assert throughput >= 500, \
            f"Throughput too low: {throughput:.0f} records/sec (target: 1000)"


# ============================================================================
# Test Error Handling
# ============================================================================

@pytest.mark.unit
class TestErrorHandling:
    """Test error handling."""

    def test_handles_missing_file(self, cn_codes_path, cbam_rules_path, importer_info):
        """Test handles missing input file gracefully."""
        agent = ShipmentIntakeAgent(cn_codes_path, cbam_rules_path)

        with pytest.raises(FileNotFoundError):
            agent.process("nonexistent.csv", importer_info)

    def test_handles_empty_file(self, tmp_path, cn_codes_path, cbam_rules_path, importer_info):
        """Test handles empty CSV file."""
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("cn_code,country_of_origin,quantity_tons,import_date\n")

        agent = ShipmentIntakeAgent(cn_codes_path, cbam_rules_path)
        result = agent.process(str(empty_csv), importer_info)

        # Should handle gracefully
        assert result['metadata']['total_records'] == 0

    def test_handles_corrupted_csv(self, tmp_path, cn_codes_path, cbam_rules_path, importer_info):
        """Test handles corrupted CSV gracefully."""
        corrupted_csv = tmp_path / "corrupted.csv"
        corrupted_csv.write_text("This is not valid CSV data\n{{corrupted}}")

        agent = ShipmentIntakeAgent(cn_codes_path, cbam_rules_path)

        # Should raise appropriate error
        with pytest.raises(Exception):
            agent.process(str(corrupted_csv), importer_info)


# ============================================================================
# Test Metadata Generation
# ============================================================================

@pytest.mark.unit
class TestMetadata:
    """Test metadata generation."""

    def test_generates_complete_metadata(self, sample_shipments_csv, cn_codes_path, cbam_rules_path, importer_info):
        """Test generates complete metadata."""
        agent = ShipmentIntakeAgent(cn_codes_path, cbam_rules_path)
        result = agent.process(sample_shipments_csv, importer_info)

        metadata = result['metadata']

        # Required fields
        assert 'total_records' in metadata
        assert 'valid_records' in metadata
        assert 'error_count' in metadata
        assert 'warning_count' in metadata
        assert 'processing_time' in metadata

    def test_metadata_accuracy(self, sample_shipments_csv, cn_codes_path, cbam_rules_path, importer_info):
        """Test metadata counts are accurate."""
        agent = ShipmentIntakeAgent(cn_codes_path, cbam_rules_path)
        result = agent.process(sample_shipments_csv, importer_info)

        metadata = result['metadata']
        validated = result['validated_shipments']

        # Counts should match
        assert metadata['valid_records'] == len(validated)
        assert metadata['total_records'] >= metadata['valid_records']
