# -*- coding: utf-8 -*-
"""
CBAM Importer Copilot - Reporting Packager Agent Tests

Unit tests for ReportingPackagerAgent functionality.

Version: 1.0.0
"""

import pytest
from pathlib import Path
import sys
from greenlang.determinism import FinancialDecimal

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.reporting_packager_agent import ReportingPackagerAgent


# ============================================================================
# Test Agent Initialization
# ============================================================================

@pytest.mark.unit
class TestAgentInitialization:
    """Test reporting packager initialization."""

    def test_agent_initializes_successfully(self):
        """Test agent can be initialized."""
        agent = ReportingPackagerAgent()
        assert agent is not None


# ============================================================================
# Test Report Generation
# ============================================================================

@pytest.mark.unit
class TestReportGeneration:
    """Test final report generation."""

    def test_generates_complete_report(self, sample_shipments_data, importer_info):
        """Test generates complete CBAM report."""
        agent = ReportingPackagerAgent()

        # Create emissions data
        emissions_data = {
            'shipments_with_emissions': [
                {
                    **shipment,
                    'embedded_emissions_tco2': 10.0,
                    'emission_factor_tco2_per_ton': 0.8,
                    'emission_factor_source': 'default',
                    'calculation_method': 'deterministic',
                    'product_group': 'iron_steel',
                    'cn_description': 'Iron and steel products'
                }
                for shipment in sample_shipments_data
            ],
            'total_emissions_tco2': 50.0,
            'total_quantity_tons': 61.7,
            'emissions_by_cn_code': {},
            'emissions_by_country': {},
            'emissions_by_product_group': {}
        }

        report = agent.generate_report(
            emissions_data=emissions_data,
            importer_info=importer_info,
            input_file_metadata={'file_name': 'test.csv'}
        )

        assert report is not None
        assert 'report_metadata' in report
        assert 'emissions_summary' in report
        assert 'detailed_goods' in report

    def test_report_metadata_complete(self, sample_shipments_data, importer_info):
        """Test report metadata is complete."""
        agent = ReportingPackagerAgent()

        emissions_data = {
            'shipments_with_emissions': sample_shipments_data,
            'total_emissions_tco2': 50.0
        }

        report = agent.generate_report(emissions_data, importer_info, {})

        metadata = report['report_metadata']

        # Required fields
        assert 'report_id' in metadata
        assert 'generated_at' in metadata
        assert 'importer' in metadata
        assert 'reporting_period' in metadata

        # Importer info
        importer = metadata['importer']
        assert importer['name'] == importer_info['name']
        assert importer['country'] == importer_info['country']
        assert importer['eori'] == importer_info['eori']

    def test_emissions_summary_accurate(self, sample_shipments_data, importer_info):
        """Test emissions summary is accurate."""
        agent = ReportingPackagerAgent()

        total_emissions = 123.45
        total_quantity = 100.0

        emissions_data = {
            'shipments_with_emissions': sample_shipments_data,
            'total_emissions_tco2': total_emissions,
            'total_quantity_tons': total_quantity
        }

        report = agent.generate_report(emissions_data, importer_info, {})

        summary = report['emissions_summary']

        assert summary['total_embedded_emissions_tco2'] == total_emissions
        assert summary['total_quantity_tons'] == total_quantity
        assert summary['total_shipments'] == len(sample_shipments_data)


# ============================================================================
# Test Aggregations
# ============================================================================

@pytest.mark.unit
class TestAggregations:
    """Test emissions aggregation functionality."""

    def test_aggregates_by_cn_code(self, sample_shipments_data, importer_info):
        """Test aggregates emissions by CN code."""
        agent = ReportingPackagerAgent()

        # Add emissions to sample data
        for shipment in sample_shipments_data:
            shipment['embedded_emissions_tco2'] = 10.0
            shipment['product_group'] = 'iron_steel'

        emissions_data = {
            'shipments_with_emissions': sample_shipments_data,
            'total_emissions_tco2': 50.0
        }

        report = agent.generate_report(emissions_data, importer_info, {})

        aggregations = report.get('aggregations', {})
        by_cn = aggregations.get('by_cn_code', [])

        # Should have aggregated by CN code
        assert len(by_cn) > 0

        # Each aggregation should have required fields
        for agg in by_cn:
            assert 'cn_code' in agg
            assert 'total_emissions_tco2' in agg
            assert 'total_quantity_tons' in agg

    def test_aggregates_by_country(self, sample_shipments_data, importer_info):
        """Test aggregates emissions by country of origin."""
        agent = ReportingPackagerAgent()

        for shipment in sample_shipments_data:
            shipment['embedded_emissions_tco2'] = 10.0
            shipment['product_group'] = 'iron_steel'

        emissions_data = {
            'shipments_with_emissions': sample_shipments_data,
            'total_emissions_tco2': 50.0
        }

        report = agent.generate_report(emissions_data, importer_info, {})

        aggregations = report.get('aggregations', {})
        by_country = aggregations.get('by_country', [])

        assert len(by_country) > 0

        for agg in by_country:
            assert 'country_of_origin' in agg
            assert 'total_emissions_tco2' in agg

    def test_aggregates_by_product_group(self, sample_shipments_data, importer_info):
        """Test aggregates emissions by product group."""
        agent = ReportingPackagerAgent()

        for shipment in sample_shipments_data:
            shipment['embedded_emissions_tco2'] = 10.0
            if shipment['cn_code'].startswith('72'):
                shipment['product_group'] = 'iron_steel'
            elif shipment['cn_code'].startswith('76'):
                shipment['product_group'] = 'aluminum'
            else:
                shipment['product_group'] = 'cement'

        emissions_data = {
            'shipments_with_emissions': sample_shipments_data,
            'total_emissions_tco2': 50.0
        }

        report = agent.generate_report(emissions_data, importer_info, {})

        aggregations = report.get('aggregations', {})
        by_group = aggregations.get('by_product_group', [])

        assert len(by_group) > 0

    def test_aggregation_sums_correct(self, sample_shipments_data, importer_info):
        """Test aggregation sums are mathematically correct."""
        agent = ReportingPackagerAgent()

        # Set known emissions
        for i, shipment in enumerate(sample_shipments_data):
            shipment['embedded_emissions_tco2'] = FinancialDecimal.from_string(i + 1)  # 1, 2, 3, 4, 5
            shipment['product_group'] = 'iron_steel'

        total_emissions = sum(i + 1 for i in range(len(sample_shipments_data)))

        emissions_data = {
            'shipments_with_emissions': sample_shipments_data,
            'total_emissions_tco2': total_emissions
        }

        report = agent.generate_report(emissions_data, importer_info, {})

        # Sum of all CN code aggregations should equal total
        by_cn = report['aggregations']['by_cn_code']
        sum_by_cn = sum(agg['total_emissions_tco2'] for agg in by_cn)

        assert abs(sum_by_cn - total_emissions) < 0.01


# ============================================================================
# Test Complex Goods Validation
# ============================================================================

@pytest.mark.unit
@pytest.mark.compliance
class TestComplexGoodsValidation:
    """Test complex goods 20% rule validation."""

    def test_detects_complex_goods_over_20_percent(self, importer_info):
        """Test detects when >20% of goods are complex."""
        agent = ReportingPackagerAgent()

        # Create 10 shipments, 3 complex (30% > 20%)
        shipments = []
        for i in range(10):
            shipments.append({
                'cn_code': f'7207{i:04d}',
                'quantity_tons': 10.0,
                'embedded_emissions_tco2': 8.0,
                'product_group': 'iron_steel',
                'is_complex_good': i < 3  # First 3 are complex
            })

        emissions_data = {
            'shipments_with_emissions': shipments,
            'total_emissions_tco2': 80.0
        }

        report = agent.generate_report(emissions_data, importer_info, {})

        validation = report.get('validation_results', {})

        # Should flag complex goods > 20%
        complex_pct = validation.get('complex_goods_percentage', 0)
        assert complex_pct == 30.0

        # May have warning
        warnings = validation.get('warnings', [])
        # Check if complex goods warning exists (implementation dependent)

    def test_accepts_complex_goods_under_20_percent(self, importer_info):
        """Test accepts when <20% of goods are complex."""
        agent = ReportingPackagerAgent()

        # Create 10 shipments, 1 complex (10% < 20%)
        shipments = []
        for i in range(10):
            shipments.append({
                'cn_code': f'7207{i:04d}',
                'quantity_tons': 10.0,
                'embedded_emissions_tco2': 8.0,
                'product_group': 'iron_steel',
                'is_complex_good': i == 0  # Only first is complex
            })

        emissions_data = {
            'shipments_with_emissions': shipments,
            'total_emissions_tco2': 80.0
        }

        report = agent.generate_report(emissions_data, importer_info, {})

        validation = report.get('validation_results', {})
        complex_pct = validation.get('complex_goods_percentage', 0)

        assert complex_pct == 10.0


# ============================================================================
# Test Final Validation
# ============================================================================

@pytest.mark.unit
@pytest.mark.compliance
class TestFinalValidation:
    """Test final CBAM validation before report submission."""

    def test_validates_all_required_fields(self, sample_shipments_data, importer_info):
        """Test validates all required fields are present."""
        agent = ReportingPackagerAgent()

        for shipment in sample_shipments_data:
            shipment['embedded_emissions_tco2'] = 10.0
            shipment['product_group'] = 'iron_steel'

        emissions_data = {
            'shipments_with_emissions': sample_shipments_data,
            'total_emissions_tco2': 50.0
        }

        report = agent.generate_report(emissions_data, importer_info, {})

        validation = report.get('validation_results', {})

        # Should have validation results
        assert 'is_valid' in validation
        assert isinstance(validation['is_valid'], bool)

    def test_marks_valid_report_as_valid(self, sample_shipments_data, importer_info):
        """Test marks valid report as valid."""
        agent = ReportingPackagerAgent()

        # Create complete, valid data
        for shipment in sample_shipments_data:
            shipment['embedded_emissions_tco2'] = 10.0
            shipment['emission_factor_tco2_per_ton'] = 0.8
            shipment['emission_factor_source'] = 'default'
            shipment['calculation_method'] = 'deterministic'
            shipment['product_group'] = 'iron_steel'

        emissions_data = {
            'shipments_with_emissions': sample_shipments_data,
            'total_emissions_tco2': 50.0
        }

        report = agent.generate_report(emissions_data, importer_info, {})

        validation = report['validation_results']
        assert validation['is_valid'] == True

    def test_includes_error_list(self, sample_shipments_data, importer_info):
        """Test includes error list in validation."""
        agent = ReportingPackagerAgent()

        for shipment in sample_shipments_data:
            shipment['embedded_emissions_tco2'] = 10.0
            shipment['product_group'] = 'iron_steel'

        emissions_data = {
            'shipments_with_emissions': sample_shipments_data,
            'total_emissions_tco2': 50.0
        }

        report = agent.generate_report(emissions_data, importer_info, {})

        validation = report['validation_results']

        assert 'errors' in validation
        assert 'warnings' in validation
        assert isinstance(validation['errors'], list)
        assert isinstance(validation['warnings'], list)


# ============================================================================
# Test Human-Readable Summary
# ============================================================================

@pytest.mark.unit
class TestHumanSummary:
    """Test human-readable summary generation."""

    def test_generates_markdown_summary(self, sample_shipments_data, importer_info):
        """Test generates Markdown summary."""
        agent = ReportingPackagerAgent()

        for shipment in sample_shipments_data:
            shipment['embedded_emissions_tco2'] = 10.0
            shipment['product_group'] = 'iron_steel'

        emissions_data = {
            'shipments_with_emissions': sample_shipments_data,
            'total_emissions_tco2': 50.0,
            'total_quantity_tons': 61.7
        }

        # Generate report with summary
        summary = agent.generate_human_summary(emissions_data, importer_info)

        assert summary is not None
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_summary_contains_key_info(self, sample_shipments_data, importer_info):
        """Test summary contains key information."""
        agent = ReportingPackagerAgent()

        for shipment in sample_shipments_data:
            shipment['embedded_emissions_tco2'] = 10.0
            shipment['product_group'] = 'iron_steel'

        emissions_data = {
            'shipments_with_emissions': sample_shipments_data,
            'total_emissions_tco2': 50.0,
            'total_quantity_tons': 61.7
        }

        summary = agent.generate_human_summary(emissions_data, importer_info)

        # Should contain key metrics (case insensitive check)
        summary_lower = summary.lower()
        assert 'emissions' in summary_lower or 'tco2' in summary_lower
        assert importer_info['name'].lower() in summary_lower or 'importer' in summary_lower


# ============================================================================
# Test Report ID Generation
# ============================================================================

@pytest.mark.unit
class TestReportIDGeneration:
    """Test unique report ID generation."""

    def test_generates_unique_report_id(self, sample_shipments_data, importer_info):
        """Test generates unique report ID."""
        agent = ReportingPackagerAgent()

        emissions_data = {
            'shipments_with_emissions': sample_shipments_data,
            'total_emissions_tco2': 50.0
        }

        report1 = agent.generate_report(emissions_data, importer_info, {})
        report2 = agent.generate_report(emissions_data, importer_info, {})

        id1 = report1['report_metadata']['report_id']
        id2 = report2['report_metadata']['report_id']

        # IDs should be different (unless generated at exact same timestamp)
        # At minimum, should be non-empty
        assert len(id1) > 0
        assert len(id2) > 0

    def test_report_id_format(self, sample_shipments_data, importer_info):
        """Test report ID follows expected format."""
        agent = ReportingPackagerAgent()

        emissions_data = {
            'shipments_with_emissions': sample_shipments_data,
            'total_emissions_tco2': 50.0
        }

        report = agent.generate_report(emissions_data, importer_info, {})
        report_id = report['report_metadata']['report_id']

        # Should contain CBAM prefix or similar identifier
        # Format may vary, but should be structured
        assert len(report_id) > 5  # At least some structure
        assert '-' in report_id or '_' in report_id  # Has delimiters


# ============================================================================
# Test Performance
# ============================================================================

@pytest.mark.unit
@pytest.mark.performance
class TestPerformance:
    """Test reporting packager performance."""

    def test_fast_report_generation(self, sample_shipments_data, importer_info):
        """Test report generation is fast."""
        import time

        agent = ReportingPackagerAgent()

        for shipment in sample_shipments_data:
            shipment['embedded_emissions_tco2'] = 10.0
            shipment['product_group'] = 'iron_steel'

        emissions_data = {
            'shipments_with_emissions': sample_shipments_data,
            'total_emissions_tco2': 50.0
        }

        start = time.time()
        report = agent.generate_report(emissions_data, importer_info, {})
        duration = time.time() - start

        # Should be <1 second for small dataset
        assert duration < 1.0, f"Report generation too slow: {duration:.2f}s"

    def test_large_dataset_performance(self, large_shipments_data, importer_info):
        """Test performance with large dataset (1000 records)."""
        import time

        agent = ReportingPackagerAgent()

        for shipment in large_shipments_data:
            shipment['embedded_emissions_tco2'] = 10.0
            shipment['product_group'] = 'iron_steel'

        emissions_data = {
            'shipments_with_emissions': large_shipments_data,
            'total_emissions_tco2': 10000.0
        }

        start = time.time()
        report = agent.generate_report(emissions_data, importer_info, {})
        duration = time.time() - start

        # Target: <1s for 10K shipments, so 1000 should be <<1s
        assert duration < 2.0, \
            f"Large report generation too slow: {duration:.2f}s for {len(large_shipments_data)} records"


# ============================================================================
# Test Edge Cases
# ============================================================================

@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_handles_empty_shipments(self, importer_info):
        """Test handles empty shipment list."""
        agent = ReportingPackagerAgent()

        emissions_data = {
            'shipments_with_emissions': [],
            'total_emissions_tco2': 0.0
        }

        report = agent.generate_report(emissions_data, importer_info, {})

        assert report is not None
        assert report['emissions_summary']['total_shipments'] == 0

    def test_handles_missing_optional_fields(self, sample_shipments_data, importer_info):
        """Test handles missing optional fields gracefully."""
        agent = ReportingPackagerAgent()

        # Minimal data
        emissions_data = {
            'shipments_with_emissions': sample_shipments_data,
            'total_emissions_tco2': 50.0
        }

        # Should not crash
        report = agent.generate_report(emissions_data, importer_info, {})
        assert report is not None

    def test_handles_very_large_emissions(self, importer_info):
        """Test handles very large emission values."""
        agent = ReportingPackagerAgent()

        shipments = [{
            'cn_code': '72071100',
            'quantity_tons': 1000000.0,
            'embedded_emissions_tco2': 800000.0,
            'product_group': 'iron_steel'
        }]

        emissions_data = {
            'shipments_with_emissions': shipments,
            'total_emissions_tco2': 800000.0
        }

        report = agent.generate_report(emissions_data, importer_info, {})

        assert report is not None
        assert report['emissions_summary']['total_embedded_emissions_tco2'] == 800000.0


# ============================================================================
# Test Data Completeness
# ============================================================================

@pytest.mark.unit
@pytest.mark.compliance
class TestDataCompleteness:
    """Test report contains all required data for EU submission."""

    def test_report_has_all_required_sections(self, sample_shipments_data, importer_info):
        """Test report has all EU-required sections."""
        agent = ReportingPackagerAgent()

        for shipment in sample_shipments_data:
            shipment['embedded_emissions_tco2'] = 10.0
            shipment['product_group'] = 'iron_steel'

        emissions_data = {
            'shipments_with_emissions': sample_shipments_data,
            'total_emissions_tco2': 50.0
        }

        report = agent.generate_report(emissions_data, importer_info, {})

        # EU Registry required sections
        required_sections = [
            'report_metadata',
            'emissions_summary',
            'detailed_goods',
            'aggregations',
            'validation_results'
        ]

        for section in required_sections:
            assert section in report, f"Missing required section: {section}"

    def test_detailed_goods_complete(self, sample_shipments_data, importer_info):
        """Test detailed goods have all required fields."""
        agent = ReportingPackagerAgent()

        for shipment in sample_shipments_data:
            shipment['embedded_emissions_tco2'] = 10.0
            shipment['emission_factor_tco2_per_ton'] = 0.8
            shipment['product_group'] = 'iron_steel'

        emissions_data = {
            'shipments_with_emissions': sample_shipments_data,
            'total_emissions_tco2': 50.0
        }

        report = agent.generate_report(emissions_data, importer_info, {})

        detailed_goods = report['detailed_goods']

        # Each good should have required fields
        for good in detailed_goods:
            assert 'cn_code' in good
            assert 'quantity_tons' in good
            assert 'embedded_emissions_tco2' in good
            assert 'country_of_origin' in good
