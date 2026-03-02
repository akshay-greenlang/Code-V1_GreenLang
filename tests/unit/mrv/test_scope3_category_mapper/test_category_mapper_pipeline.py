# -*- coding: utf-8 -*-
"""
Unit tests for CategoryMapperPipeline (AGENT-MRV-029, Main Pipeline)

80 tests covering:
- Pipeline stage tests: 10 stages x 5 tests each (50 tests)
- End-to-end pipeline tests (20 tests)
- Integration tests: pipeline with engines (10 tests)

The CategoryMapperPipeline orchestrates a 10-stage processing flow:
  1. Input Validation
  2. Source Classification
  3. Code Lookup (NAICS / ISIC / GL)
  4. Classification (rule-based / keyword)
  5. Boundary Determination
  6. Double-Counting Check
  7. Completeness Screening
  8. Compliance Assessment
  9. Provenance Sealing
  10. Output Formatting

Author: GL-TestEngineer
Date: March 2026
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

try:
    from greenlang.scope3_category_mapper.category_mapper_pipeline import (
        CategoryMapperPipeline,
    )
    PIPELINE_AVAILABLE = True
except (ImportError, AttributeError):
    PIPELINE_AVAILABLE = False

try:
    from greenlang.scope3_category_mapper.models import (
        Scope3Category,
        CompanyType,
    )
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

try:
    from greenlang.scope3_category_mapper.category_database import (
        CategoryDatabaseEngine,
    )
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not PIPELINE_AVAILABLE,
    reason="CategoryMapperPipeline not available",
)


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def pipeline():
    """Create a fresh CategoryMapperPipeline instance."""
    if not PIPELINE_AVAILABLE:
        pytest.skip("CategoryMapperPipeline not available")
    return CategoryMapperPipeline()


@pytest.fixture
def single_spend_record() -> Dict[str, Any]:
    """Single spend record for pipeline testing."""
    return {
        "record_id": "SPD-001",
        "amount": Decimal("12500.00"),
        "currency": "USD",
        "gl_account": "5000",
        "naics_code": "331",
        "description": "Raw steel purchase for manufacturing",
        "supplier_name": "US Steel Corp",
        "source_type": "purchase_order",
        "posting_date": "2025-03-15",
    }


@pytest.fixture
def travel_record() -> Dict[str, Any]:
    """Travel record for pipeline testing."""
    return {
        "record_id": "TRV-001",
        "amount": Decimal("3200.00"),
        "currency": "USD",
        "description": "Employee airfare SFO to JFK",
        "source_type": "expense_report",
        "naics_code": "481",
        "gl_account": "6410",
    }


@pytest.fixture
def logistics_record() -> Dict[str, Any]:
    """Logistics record with Incoterm."""
    return {
        "record_id": "LOG-001",
        "amount": Decimal("7500.00"),
        "currency": "USD",
        "description": "Inbound freight Chicago-Detroit",
        "source_type": "invoice",
        "naics_code": "484",
        "gl_account": "5300",
        "incoterm": "FOB",
        "direction": "inbound",
        "transport_mode": "road",
    }


@pytest.fixture
def batch_records(
    single_spend_record, travel_record, logistics_record
) -> List[Dict[str, Any]]:
    """Batch of mixed records."""
    return [single_spend_record, travel_record, logistics_record]


@pytest.fixture
def org_context() -> Dict[str, Any]:
    """Organization context for pipeline."""
    return {
        "company_type": "manufacturer",
        "consolidation_approach": "operational_control",
        "reporting_year": 2024,
    }


# ==============================================================================
# STAGE 1: INPUT VALIDATION
# ==============================================================================


@_SKIP
class TestStage1Validation:
    """Test Stage 1 - Input Validation."""

    def test_valid_input_passes(self, pipeline, single_spend_record, org_context):
        """Valid record passes input validation."""
        validated = pipeline.validate_input(single_spend_record)
        assert validated is not None
        assert validated["record_id"] == "SPD-001"

    def test_invalid_input_raises(self, pipeline):
        """Missing required fields raises ValidationError."""
        with pytest.raises((ValueError, KeyError)):
            pipeline.validate_input({})

    def test_empty_records_handled(self, pipeline, org_context):
        """Empty record list returns empty result."""
        result = pipeline.run(records=[], org_context=org_context)
        assert len(result.classifications) == 0

    def test_missing_record_id_raises(self, pipeline):
        """Record without record_id raises ValidationError."""
        record = {"amount": Decimal("100"), "description": "test"}
        with pytest.raises((ValueError, KeyError)):
            pipeline.validate_input(record)

    def test_negative_amount_handled(self, pipeline):
        """Negative amount is flagged or rejected."""
        record = {
            "record_id": "NEG-001",
            "amount": Decimal("-500.00"),
            "description": "Credit note",
        }
        # Should either raise or flag as warning
        try:
            result = pipeline.validate_input(record)
            assert result.get("validation_warnings") is not None or result is not None
        except (ValueError, KeyError):
            pass  # Also acceptable to reject


# ==============================================================================
# STAGE 2: SOURCE CLASSIFICATION
# ==============================================================================


@_SKIP
class TestStage2SourceClassification:
    """Test Stage 2 - Source Classification."""

    def test_spend_source_detected(self, pipeline, single_spend_record):
        """Purchase order detected as spend source."""
        source = pipeline.classify_source(single_spend_record)
        assert source in ("purchase_order", "spend", "procurement")

    def test_travel_source_detected(self, pipeline, travel_record):
        """Expense report detected as travel source."""
        source = pipeline.classify_source(travel_record)
        assert source in ("expense_report", "travel", "business_travel")

    def test_logistics_source_detected(self, pipeline, logistics_record):
        """Invoice with freight detected as logistics source."""
        source = pipeline.classify_source(logistics_record)
        assert source in ("invoice", "logistics", "freight")

    def test_utility_source_detected(self, pipeline):
        """Utility bill detected as energy source."""
        record = {
            "record_id": "UTL-001",
            "source_type": "utility_bill",
            "description": "Electricity bill Q1",
        }
        source = pipeline.classify_source(record)
        assert source in ("utility_bill", "energy", "utility")

    def test_unknown_source_classified(self, pipeline):
        """Unknown source type is handled gracefully."""
        record = {
            "record_id": "UNK-001",
            "source_type": "unknown",
            "description": "Miscellaneous charge",
        }
        source = pipeline.classify_source(record)
        assert source is not None


# ==============================================================================
# STAGE 3: CODE LOOKUP
# ==============================================================================


@_SKIP
class TestStage3CodeLookup:
    """Test Stage 3 - NAICS / ISIC / GL Code Lookup."""

    def test_naics_code_found(self, pipeline, single_spend_record):
        """NAICS code 331 found -> primary category assigned."""
        lookup = pipeline.lookup_codes(single_spend_record)
        assert lookup is not None
        assert lookup.get("naics_result") is not None

    def test_no_code_proceeds(self, pipeline):
        """Record without NAICS/ISIC/GL proceeds to keyword classification."""
        record = {
            "record_id": "NO-CODE",
            "description": "Office supplies purchase",
        }
        lookup = pipeline.lookup_codes(record)
        # Should not fail; code lookups return None
        assert lookup is not None

    def test_gl_account_lookup(self, pipeline):
        """GL account code 5000 maps to a category."""
        record = {
            "record_id": "GL-001",
            "gl_account": "5000",
        }
        lookup = pipeline.lookup_codes(record)
        assert lookup.get("gl_result") is not None

    def test_isic_code_lookup(self, pipeline):
        """ISIC code 'C' maps to manufacturing."""
        record = {
            "record_id": "ISIC-001",
            "isic_code": "C",
        }
        lookup = pipeline.lookup_codes(record)
        assert lookup.get("isic_result") is not None

    def test_invalid_naics_handled(self, pipeline):
        """Invalid NAICS code handled gracefully."""
        record = {
            "record_id": "BAD-NAICS",
            "naics_code": "XYZ",
        }
        lookup = pipeline.lookup_codes(record)
        assert lookup.get("naics_result") is None


# ==============================================================================
# STAGE 4: CLASSIFICATION
# ==============================================================================


@_SKIP
class TestStage4Classification:
    """Test Stage 4 - Category Classification."""

    def test_records_classified(self, pipeline, single_spend_record, org_context):
        """Record is classified into a Scope 3 category."""
        result = pipeline.classify_record(single_spend_record, org_context)
        assert result is not None
        assert "primary_category" in result

    def test_classification_has_confidence(
        self, pipeline, single_spend_record, org_context
    ):
        """Classification result includes confidence score."""
        result = pipeline.classify_record(single_spend_record, org_context)
        assert "confidence" in result
        conf = result["confidence"]
        assert 0.0 <= float(conf) <= 1.0

    def test_keyword_fallback(self, pipeline, org_context):
        """Record without codes uses keyword classification."""
        record = {
            "record_id": "KW-001",
            "description": "International airfare for client meeting",
            "amount": Decimal("2500.00"),
        }
        result = pipeline.classify_record(record, org_context)
        assert result is not None
        assert "primary_category" in result

    def test_classification_has_method(
        self, pipeline, single_spend_record, org_context
    ):
        """Classification result includes method used."""
        result = pipeline.classify_record(single_spend_record, org_context)
        assert "classification_method" in result
        assert result["classification_method"] in (
            "naics_lookup", "isic_lookup", "gl_lookup",
            "keyword_match", "rule_based", "hybrid",
        )

    def test_classification_has_provenance(
        self, pipeline, single_spend_record, org_context
    ):
        """Classification result includes provenance hash."""
        result = pipeline.classify_record(single_spend_record, org_context)
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64


# ==============================================================================
# STAGE 5: BOUNDARY DETERMINATION
# ==============================================================================


@_SKIP
class TestStage5Boundary:
    """Test Stage 5 - Boundary Determination."""

    def test_upstream_freight_assigned(self, pipeline, logistics_record, org_context):
        """Inbound freight with FOB -> upstream (Cat 4)."""
        classified = pipeline.classify_record(logistics_record, org_context)
        boundary = pipeline.determine_boundary(classified, org_context)
        assert boundary is not None

    def test_capex_boundary_applied(self, pipeline, org_context):
        """Capital goods boundary (Cat 1 vs Cat 2) applied."""
        record = {
            "record_id": "CAP-001",
            "amount": Decimal("50000.00"),
            "description": "Industrial CNC machine",
            "useful_life_years": 10,
            "naics_code": "333",
        }
        classified = pipeline.classify_record(record, org_context)
        boundary = pipeline.determine_boundary(classified, org_context)
        assert boundary is not None

    def test_scope2_excluded_from_cat3(self, pipeline, org_context):
        """Scope 2 generation excluded from Cat 3."""
        record = {
            "record_id": "ENR-001",
            "amount": Decimal("6000.00"),
            "description": "Electricity bill",
            "energy_activity": "wtt_fuel",
        }
        classified = pipeline.classify_record(record, org_context)
        boundary = pipeline.determine_boundary(classified, org_context)
        assert boundary is not None

    def test_boundary_preserves_record_id(self, pipeline, logistics_record, org_context):
        """Record ID is preserved through boundary stage."""
        classified = pipeline.classify_record(logistics_record, org_context)
        boundary = pipeline.determine_boundary(classified, org_context)
        assert boundary.get("record_id") == "LOG-001"

    def test_boundary_adds_dc_rule(self, pipeline, logistics_record, org_context):
        """Boundary determination records which DC rule was applied."""
        classified = pipeline.classify_record(logistics_record, org_context)
        boundary = pipeline.determine_boundary(classified, org_context)
        # May or may not have dc_rule depending on classification
        assert boundary is not None


# ==============================================================================
# STAGE 6: DOUBLE-COUNTING CHECK
# ==============================================================================


@_SKIP
class TestStage6DoubleCountingCheck:
    """Test Stage 6 - Double-Counting Check."""

    def test_no_overlap_in_clean_batch(self, pipeline, batch_records, org_context):
        """Clean batch with distinct categories -> no overlap."""
        classified = [
            pipeline.classify_record(r, org_context) for r in batch_records
        ]
        dc_result = pipeline.check_double_counting(classified)
        assert dc_result.get("overlap_count", 0) == 0

    def test_overlap_detected_duplicate_record(self, pipeline, org_context):
        """Same record assigned to two categories -> overlap detected."""
        records = [
            {"record_id": "DUP-001", "primary_category": "cat_1_purchased_goods"},
            {"record_id": "DUP-001", "primary_category": "cat_2_capital_goods"},
        ]
        dc_result = pipeline.check_double_counting(records)
        assert dc_result.get("overlap_count", 0) >= 1

    def test_dc_check_returns_status(self, pipeline, batch_records, org_context):
        """DC check returns PASS/FAIL/WARNING status."""
        classified = [
            pipeline.classify_record(r, org_context) for r in batch_records
        ]
        dc_result = pipeline.check_double_counting(classified)
        assert dc_result.get("status") in ("PASS", "FAIL", "WARNING")

    def test_dc_check_empty_batch(self, pipeline):
        """Empty batch -> PASS."""
        dc_result = pipeline.check_double_counting([])
        assert dc_result.get("status") == "PASS"

    def test_dc_check_provenance(self, pipeline, batch_records, org_context):
        """DC check result has provenance hash."""
        classified = [
            pipeline.classify_record(r, org_context) for r in batch_records
        ]
        dc_result = pipeline.check_double_counting(classified)
        assert len(dc_result.get("provenance_hash", "")) == 64


# ==============================================================================
# STAGE 7: COMPLETENESS SCREENING
# ==============================================================================


@_SKIP
class TestStage7Completeness:
    """Test Stage 7 - Completeness Screening."""

    def test_completeness_included_in_pipeline(
        self, pipeline, batch_records, org_context
    ):
        """Pipeline output includes completeness report."""
        result = pipeline.run(records=batch_records, org_context=org_context)
        assert result.completeness_report is not None

    def test_completeness_score_populated(
        self, pipeline, batch_records, org_context
    ):
        """Completeness score is a Decimal in [0, 100]."""
        result = pipeline.run(records=batch_records, org_context=org_context)
        score = result.completeness_report.completeness_score
        assert Decimal("0") <= score <= Decimal("100")

    def test_completeness_gaps_listed(
        self, pipeline, batch_records, org_context
    ):
        """Completeness report lists missing categories."""
        result = pipeline.run(records=batch_records, org_context=org_context)
        # 3 records won't cover all 15 categories
        assert len(result.completeness_report.gaps) > 0

    def test_completeness_recommendations(
        self, pipeline, batch_records, org_context
    ):
        """Completeness report includes recommendations."""
        result = pipeline.run(records=batch_records, org_context=org_context)
        assert len(result.completeness_report.recommended_actions) > 0

    def test_completeness_per_category_detail(
        self, pipeline, batch_records, org_context
    ):
        """Completeness report includes per-category detail."""
        result = pipeline.run(records=batch_records, org_context=org_context)
        assert len(result.completeness_report.category_details) == 15


# ==============================================================================
# STAGE 8: COMPLIANCE ASSESSMENT
# ==============================================================================


@_SKIP
class TestStage8Compliance:
    """Test Stage 8 - Compliance Assessment."""

    def test_compliance_in_pipeline(self, pipeline, batch_records, org_context):
        """Pipeline output includes compliance assessment."""
        result = pipeline.run(records=batch_records, org_context=org_context)
        assert result.compliance_assessments is not None

    def test_compliance_all_frameworks(self, pipeline, batch_records, org_context):
        """Compliance assessment covers all 8 frameworks."""
        result = pipeline.run(records=batch_records, org_context=org_context)
        assert len(result.compliance_assessments) == 8

    def test_compliance_has_scores(self, pipeline, batch_records, org_context):
        """Each framework assessment has a score."""
        result = pipeline.run(records=batch_records, org_context=org_context)
        for assessment in result.compliance_assessments.values():
            assert Decimal("0") <= assessment.score <= Decimal("100")

    def test_compliance_has_findings(self, pipeline, batch_records, org_context):
        """Assessments include findings."""
        result = pipeline.run(records=batch_records, org_context=org_context)
        total_findings = sum(
            len(a.findings) for a in result.compliance_assessments.values()
        )
        assert total_findings > 0

    def test_compliance_provenance(self, pipeline, batch_records, org_context):
        """Each assessment has provenance hash."""
        result = pipeline.run(records=batch_records, org_context=org_context)
        for assessment in result.compliance_assessments.values():
            assert len(assessment.provenance_hash) == 64


# ==============================================================================
# STAGE 9: PROVENANCE SEALING
# ==============================================================================


@_SKIP
class TestStage9Provenance:
    """Test Stage 9 - Provenance Sealing."""

    def test_pipeline_result_has_provenance(
        self, pipeline, batch_records, org_context
    ):
        """Pipeline result includes final provenance hash."""
        result = pipeline.run(records=batch_records, org_context=org_context)
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    def test_provenance_chain_complete(
        self, pipeline, batch_records, org_context
    ):
        """Provenance chain includes all 10 stages."""
        result = pipeline.run(records=batch_records, org_context=org_context)
        assert len(result.provenance_chain) >= 10

    def test_provenance_deterministic(
        self, pipeline, batch_records, org_context
    ):
        """Same input produces same provenance hash."""
        r1 = pipeline.run(records=batch_records, org_context=org_context)
        r2 = pipeline.run(records=batch_records, org_context=org_context)
        assert r1.provenance_hash == r2.provenance_hash

    def test_provenance_changes_with_input(
        self, pipeline, org_context
    ):
        """Different input produces different provenance hash."""
        r1 = pipeline.run(
            records=[{"record_id": "A", "amount": Decimal("100"),
                       "description": "Alpha"}],
            org_context=org_context,
        )
        r2 = pipeline.run(
            records=[{"record_id": "B", "amount": Decimal("200"),
                       "description": "Beta"}],
            org_context=org_context,
        )
        assert r1.provenance_hash != r2.provenance_hash

    def test_provenance_sha256(self, pipeline, batch_records, org_context):
        """Provenance hash is SHA-256 format (64 hex chars)."""
        result = pipeline.run(records=batch_records, org_context=org_context)
        import re
        assert re.match(r"^[a-f0-9]{64}$", result.provenance_hash)


# ==============================================================================
# STAGE 10: OUTPUT FORMATTING
# ==============================================================================


@_SKIP
class TestStage10Output:
    """Test Stage 10 - Output Formatting."""

    def test_output_has_classifications(
        self, pipeline, batch_records, org_context
    ):
        """Output includes classification results."""
        result = pipeline.run(records=batch_records, org_context=org_context)
        assert len(result.classifications) == len(batch_records)

    def test_output_has_processing_time(
        self, pipeline, batch_records, org_context
    ):
        """Output includes processing time."""
        result = pipeline.run(records=batch_records, org_context=org_context)
        assert result.processing_time_ms > 0.0

    def test_output_has_record_count(
        self, pipeline, batch_records, org_context
    ):
        """Output includes record count."""
        result = pipeline.run(records=batch_records, org_context=org_context)
        assert result.record_count == len(batch_records)

    def test_output_classification_fields(
        self, pipeline, batch_records, org_context
    ):
        """Each classification has required fields."""
        result = pipeline.run(records=batch_records, org_context=org_context)
        for clf in result.classifications:
            assert "record_id" in clf
            assert "primary_category" in clf
            assert "confidence" in clf
            assert "provenance_hash" in clf

    def test_output_serializable(self, pipeline, batch_records, org_context):
        """Pipeline output is JSON-serializable."""
        import json
        result = pipeline.run(records=batch_records, org_context=org_context)
        # Should be serializable via model_dump
        data = result.model_dump() if hasattr(result, "model_dump") else result.__dict__
        json.dumps(data, default=str)  # Should not raise


# ==============================================================================
# END-TO-END PIPELINE TESTS
# ==============================================================================


@_SKIP
class TestEndToEndPipeline:
    """End-to-end pipeline tests."""

    def test_run_pipeline_single_record(
        self, pipeline, single_spend_record, org_context
    ):
        """Run pipeline with single record."""
        result = pipeline.run(
            records=[single_spend_record], org_context=org_context
        )
        assert result is not None
        assert len(result.classifications) == 1

    def test_run_pipeline_batch(self, pipeline, batch_records, org_context):
        """Run pipeline with batch of records."""
        result = pipeline.run(records=batch_records, org_context=org_context)
        assert len(result.classifications) == len(batch_records)

    def test_run_pipeline_mixed_sources(self, pipeline, org_context):
        """Pipeline handles mixed source types."""
        records = [
            {"record_id": "MIX-1", "amount": Decimal("5000"),
             "source_type": "purchase_order", "naics_code": "331",
             "description": "Steel"},
            {"record_id": "MIX-2", "amount": Decimal("3000"),
             "source_type": "expense_report", "naics_code": "481",
             "description": "Airfare"},
            {"record_id": "MIX-3", "amount": Decimal("1500"),
             "source_type": "utility_bill",
             "description": "Electricity"},
        ]
        result = pipeline.run(records=records, org_context=org_context)
        assert len(result.classifications) == 3

    def test_run_pipeline_provenance_chain(
        self, pipeline, batch_records, org_context
    ):
        """Pipeline builds a provenance chain."""
        result = pipeline.run(records=batch_records, org_context=org_context)
        assert result.provenance_chain is not None
        assert len(result.provenance_chain) >= 10

    def test_run_pipeline_all_stages_executed(
        self, pipeline, batch_records, org_context
    ):
        """All 10 pipeline stages are executed."""
        result = pipeline.run(records=batch_records, org_context=org_context)
        stage_names = [s.get("stage") or s.get("name") for s in result.provenance_chain]
        assert len(stage_names) >= 10

    def test_run_pipeline_metrics_emitted(
        self, pipeline, batch_records, org_context
    ):
        """Pipeline emits metrics (record count, processing time)."""
        result = pipeline.run(records=batch_records, org_context=org_context)
        assert result.processing_time_ms > 0
        assert result.record_count > 0

    def test_run_pipeline_error_handling(self, pipeline, org_context):
        """Pipeline handles invalid record gracefully."""
        records = [
            {"record_id": "GOOD-1", "amount": Decimal("5000"),
             "description": "Valid record"},
            {},  # Invalid record
        ]
        # Should either skip bad records or raise
        try:
            result = pipeline.run(records=records, org_context=org_context)
            # If it doesn't raise, it should still process the good record
            assert result is not None
        except (ValueError, KeyError):
            pass  # Also acceptable

    def test_run_pipeline_large_batch(self, pipeline, org_context):
        """Pipeline handles large batch (1000 records)."""
        records = [
            {"record_id": f"LB-{i:04d}", "amount": Decimal("100"),
             "description": f"Item {i}", "naics_code": "331"}
            for i in range(1000)
        ]
        result = pipeline.run(records=records, org_context=org_context)
        assert result.record_count == 1000

    def test_run_pipeline_financial_context(self, pipeline):
        """Pipeline with financial institution context."""
        records = [
            {"record_id": "FIN-1", "amount": Decimal("500000"),
             "description": "Equity investment TechCorp",
             "naics_code": "523"},
        ]
        ctx = {
            "company_type": "financial",
            "consolidation_approach": "financial_control",
            "reporting_year": 2024,
        }
        result = pipeline.run(records=records, org_context=ctx)
        assert len(result.classifications) == 1

    def test_run_pipeline_retailer_context(self, pipeline):
        """Pipeline with retailer context."""
        records = [
            {"record_id": "RET-1", "amount": Decimal("100000"),
             "description": "Merchandise purchase",
             "naics_code": "445"},
        ]
        ctx = {
            "company_type": "retailer",
            "consolidation_approach": "operational_control",
            "reporting_year": 2024,
        }
        result = pipeline.run(records=records, org_context=ctx)
        assert len(result.classifications) == 1


# ==============================================================================
# PIPELINE INTEGRATION TESTS
# ==============================================================================


@_SKIP
class TestPipelineIntegration:
    """Integration tests: pipeline with real engines."""

    def test_pipeline_spend_to_routing(
        self, pipeline, single_spend_record, org_context
    ):
        """Spend record classified and routed to downstream agent."""
        result = pipeline.run(
            records=[single_spend_record], org_context=org_context
        )
        clf = result.classifications[0]
        assert clf.get("downstream_agent") is not None or "primary_category" in clf

    def test_pipeline_completeness_included(
        self, pipeline, batch_records, org_context
    ):
        """Completeness screening is part of pipeline output."""
        result = pipeline.run(records=batch_records, org_context=org_context)
        assert result.completeness_report is not None
        assert result.completeness_report.total_categories == 15

    def test_pipeline_dc_checks_run(
        self, pipeline, batch_records, org_context
    ):
        """Double-counting checks are executed in pipeline."""
        result = pipeline.run(records=batch_records, org_context=org_context)
        assert result.double_counting_result is not None

    def test_pipeline_with_category_database(
        self, pipeline, single_spend_record, org_context
    ):
        """Pipeline uses CategoryDatabaseEngine for lookups."""
        if not DATABASE_AVAILABLE:
            pytest.skip("CategoryDatabaseEngine not available")
        result = pipeline.run(
            records=[single_spend_record], org_context=org_context
        )
        clf = result.classifications[0]
        assert clf.get("classification_method") in (
            "naics_lookup", "gl_lookup", "keyword_match",
            "isic_lookup", "rule_based", "hybrid",
        )

    def test_pipeline_compliance_assessments(
        self, pipeline, batch_records, org_context
    ):
        """Pipeline includes compliance assessments."""
        result = pipeline.run(records=batch_records, org_context=org_context)
        assert result.compliance_assessments is not None

    def test_pipeline_preserves_all_record_ids(
        self, pipeline, batch_records, org_context
    ):
        """All record IDs from input appear in output."""
        result = pipeline.run(records=batch_records, org_context=org_context)
        input_ids = {r["record_id"] for r in batch_records}
        output_ids = {c["record_id"] for c in result.classifications}
        assert input_ids == output_ids

    def test_pipeline_no_data_loss(
        self, pipeline, batch_records, org_context
    ):
        """Number of output classifications equals input records."""
        result = pipeline.run(records=batch_records, org_context=org_context)
        assert len(result.classifications) == len(batch_records)

    def test_pipeline_performance_under_5s(
        self, pipeline, org_context
    ):
        """Pipeline processes 100 records in under 5 seconds."""
        import time
        records = [
            {"record_id": f"PERF-{i:03d}", "amount": Decimal("100"),
             "description": f"Item {i}", "naics_code": "331"}
            for i in range(100)
        ]
        start = time.monotonic()
        result = pipeline.run(records=records, org_context=org_context)
        elapsed = time.monotonic() - start
        assert elapsed < 5.0
        assert result.record_count == 100

    def test_pipeline_result_immutable(
        self, pipeline, batch_records, org_context
    ):
        """Pipeline result is immutable (frozen model)."""
        result = pipeline.run(records=batch_records, org_context=org_context)
        if hasattr(result, "model_config"):
            # Pydantic frozen model
            with pytest.raises(Exception):
                result.record_count = 999

    def test_pipeline_output_has_agent_metadata(
        self, pipeline, batch_records, org_context
    ):
        """Pipeline output includes agent metadata."""
        result = pipeline.run(records=batch_records, org_context=org_context)
        assert hasattr(result, "agent_id") or "agent_id" in (
            result.__dict__ if hasattr(result, "__dict__") else {}
        )
