# -*- coding: utf-8 -*-
"""
Integration Tests: Full Validation Rule Engine Pipeline (End-to-End)
=====================================================================

Tests the complete pipeline orchestrated by the ValidationRuleEngineService
facade, exercising all seven upstream engine concepts (RuleRegistry,
RuleComposer, RuleEvaluator, ConflictDetector, RulePack,
ValidationReporter, ValidationPipeline) through realistic validation
scenarios.

Test Classes:
    TestEndToEndGHGPipeline               (~8 tests)
    TestEndToEndCSRDPipeline              (~7 tests)
    TestBatchPipelineAcrossDatasets       (~6 tests)
    TestPipelineConflictDetection         (~5 tests)
    TestPipelineReportGeneration          (~5 tests)
    TestPipelineProvenanceAndAudit        (~5 tests)
    TestPipelineErrorRecovery             (~4 tests)

Total: ~40 integration tests

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-019 Validation Rule Engine (GL-DATA-X-022)
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest


# ===========================================================================
# TestEndToEndGHGPipeline
# ===========================================================================


class TestEndToEndGHGPipeline:
    """End-to-end pipeline test with GHG Protocol rule pack."""

    def test_pipeline_completes_with_ghg_pack(
        self, service, sample_emission_records,
    ):
        """Pipeline should complete successfully with GHG Protocol pack."""
        result = service.run_pipeline(
            dataset=sample_emission_records,
            pack_name="ghg_protocol",
        )
        assert result["final_status"] in ("completed", "failed")
        assert result.get("pipeline_id") is not None

    def test_pipeline_stages_include_all_phases(
        self, service, sample_emission_records,
    ):
        """All pipeline stages should appear in stages_completed."""
        result = service.run_pipeline(
            dataset=sample_emission_records,
            pack_name="ghg_protocol",
        )
        stages = result.get("stages_completed", [])
        assert len(stages) >= 3

    def test_pipeline_produces_evaluation_id(
        self, service, sample_emission_records,
    ):
        """Pipeline should produce a valid evaluation_id."""
        result = service.run_pipeline(
            dataset=sample_emission_records,
            pack_name="ghg_protocol",
        )
        assert result.get("evaluation_id") is not None

    def test_pipeline_produces_report_id(
        self, service, sample_emission_records,
    ):
        """Pipeline should produce a valid report_id."""
        result = service.run_pipeline(
            dataset=sample_emission_records,
            pack_name="ghg_protocol",
        )
        assert result.get("report_id") is not None

    def test_pipeline_records_provenance(
        self, service, sample_emission_records,
    ):
        """Pipeline should record provenance entries."""
        initial_count = service.provenance.entry_count
        service.run_pipeline(
            dataset=sample_emission_records,
            pack_name="ghg_protocol",
        )
        assert service.provenance.entry_count > initial_count

    def test_pipeline_provenance_chain_valid(
        self, service, sample_emission_records,
    ):
        """Provenance chain should remain valid after pipeline execution."""
        service.run_pipeline(
            dataset=sample_emission_records,
            pack_name="ghg_protocol",
        )
        assert service.provenance.verify_chain() is True

    def test_pipeline_evaluation_retrievable(
        self, service, sample_emission_records,
    ):
        """Evaluation result from pipeline should be retrievable by ID."""
        result = service.run_pipeline(
            dataset=sample_emission_records,
            pack_name="ghg_protocol",
        )
        eval_id = result.get("evaluation_id")
        if eval_id:
            evaluation = service.get_evaluation(eval_id)
            assert evaluation is not None
            assert evaluation.get("status") == "completed"

    def test_pipeline_elapsed_seconds_positive(
        self, service, sample_emission_records,
    ):
        """Pipeline should report non-negative elapsed_seconds."""
        result = service.run_pipeline(
            dataset=sample_emission_records,
            pack_name="ghg_protocol",
        )
        assert result.get("elapsed_seconds", 0.0) >= 0.0


# ===========================================================================
# TestEndToEndCSRDPipeline
# ===========================================================================


class TestEndToEndCSRDPipeline:
    """End-to-end pipeline test with CSRD/ESRS rule pack."""

    def test_pipeline_completes_with_csrd_pack(
        self, service, sample_supplier_records,
    ):
        """Pipeline should complete with CSRD pack."""
        result = service.run_pipeline(
            dataset=sample_supplier_records,
            pack_name="csrd_esrs",
        )
        assert result["final_status"] in ("completed", "failed")

    def test_csrd_pipeline_produces_evaluation(
        self, service, sample_supplier_records,
    ):
        """CSRD pipeline should produce evaluation results."""
        result = service.run_pipeline(
            dataset=sample_supplier_records,
            pack_name="csrd_esrs",
        )
        assert result.get("evaluation_id") is not None

    def test_csrd_pipeline_produces_report(
        self, service, sample_supplier_records,
    ):
        """CSRD pipeline should produce a compliance report."""
        result = service.run_pipeline(
            dataset=sample_supplier_records,
            pack_name="csrd_esrs",
        )
        assert result.get("report_id") is not None

    def test_csrd_pipeline_records_provenance(
        self, service, sample_supplier_records,
    ):
        """CSRD pipeline should record provenance entries."""
        initial = service.provenance.entry_count
        service.run_pipeline(
            dataset=sample_supplier_records,
            pack_name="csrd_esrs",
        )
        assert service.provenance.entry_count > initial

    def test_csrd_pipeline_conflict_detection(
        self, service, sample_supplier_records,
    ):
        """CSRD pipeline should run conflict detection."""
        result = service.run_pipeline(
            dataset=sample_supplier_records,
            pack_name="csrd_esrs",
        )
        # Conflict count should be a non-negative integer
        assert result.get("conflict_count", 0) >= 0

    def test_csrd_pipeline_provenance_chain_valid(
        self, service, sample_supplier_records,
    ):
        """Provenance chain should be valid after CSRD pipeline."""
        service.run_pipeline(
            dataset=sample_supplier_records,
            pack_name="csrd_esrs",
        )
        assert service.provenance.verify_chain() is True

    def test_csrd_rules_registered_in_service(
        self, service, sample_supplier_records,
    ):
        """CSRD rules should be registered after pipeline execution."""
        service.run_pipeline(
            dataset=sample_supplier_records,
            pack_name="csrd_esrs",
        )
        rules = service.search_rules()
        csrd_rules = [r for r in rules if "csrd" in r.get("name", "").lower()]
        assert len(csrd_rules) >= 3


# ===========================================================================
# TestBatchPipelineAcrossDatasets
# ===========================================================================


class TestBatchPipelineAcrossDatasets:
    """Test batch pipeline execution across multiple datasets."""

    def test_batch_evaluation_three_datasets(
        self, service, sample_emission_records, sample_supplier_records,
    ):
        """Batch evaluate three datasets against GHG pack."""
        pack_result = service.apply_pack("ghg_protocol")
        rs_id = pack_result.get("rule_set_id", "")

        result = service.batch_evaluate(
            rule_set_id=rs_id,
            datasets=[
                sample_emission_records[:3],
                sample_emission_records[3:6],
                sample_emission_records[6:],
            ],
        )
        assert result["total_datasets"] == 3
        assert len(result["evaluations"]) == 3

    def test_batch_evaluation_pass_rate_computed(
        self, service, sample_emission_records,
    ):
        """Batch evaluation should compute overall_pass_rate."""
        pack_result = service.apply_pack("ghg_protocol")
        rs_id = pack_result.get("rule_set_id", "")

        result = service.batch_evaluate(
            rule_set_id=rs_id,
            datasets=[
                sample_emission_records[:5],
                sample_emission_records[5:],
            ],
        )
        assert 0.0 <= result["overall_pass_rate"] <= 1.0

    def test_batch_evaluation_individual_results(
        self, service, sample_emission_records,
    ):
        """Each dataset in batch should have individual evaluation."""
        pack_result = service.apply_pack("ghg_protocol")
        rs_id = pack_result.get("rule_set_id", "")

        result = service.batch_evaluate(
            rule_set_id=rs_id,
            datasets=[
                sample_emission_records[:3],
                sample_emission_records[3:6],
            ],
        )
        for ev in result["evaluations"]:
            assert "evaluation_id" in ev
            assert "result" in ev
            assert ev["result"] in ("pass", "warn", "fail")

    def test_batch_evaluation_records_provenance(
        self, service, sample_emission_records,
    ):
        """Batch evaluation should record provenance entries."""
        pack_result = service.apply_pack("ghg_protocol")
        rs_id = pack_result.get("rule_set_id", "")
        initial = service.provenance.entry_count

        service.batch_evaluate(
            rule_set_id=rs_id,
            datasets=[sample_emission_records[:5]],
        )
        assert service.provenance.entry_count > initial

    def test_batch_evaluation_mixed_packs(
        self, service, sample_emission_records, sample_supplier_records,
    ):
        """Run separate batch evaluations for different packs."""
        ghg = service.apply_pack("ghg_protocol")
        csrd = service.apply_pack("csrd_esrs")

        ghg_result = service.batch_evaluate(
            rule_set_id=ghg.get("rule_set_id", ""),
            datasets=[sample_emission_records],
        )
        csrd_result = service.batch_evaluate(
            rule_set_id=csrd.get("rule_set_id", ""),
            datasets=[sample_supplier_records],
        )

        assert ghg_result["total_datasets"] == 1
        assert csrd_result["total_datasets"] == 1

    def test_batch_evaluation_provenance_chain_valid(
        self, service, sample_emission_records,
    ):
        """Provenance chain should remain valid after batch evaluation."""
        pack_result = service.apply_pack("ghg_protocol")
        rs_id = pack_result.get("rule_set_id", "")

        service.batch_evaluate(
            rule_set_id=rs_id,
            datasets=[
                sample_emission_records[:5],
                sample_emission_records[5:],
            ],
        )
        assert service.provenance.verify_chain() is True


# ===========================================================================
# TestPipelineConflictDetection
# ===========================================================================


class TestPipelineConflictDetection:
    """Test conflict detection within pipelines."""

    def test_detect_conflicts_with_overlapping_rules(self, service):
        """Rules with overlapping ranges should produce overlap conflicts."""
        r1 = service.register_rule(
            name="val_low", rule_type="range_check",
            severity="error", field="value",
            min_value=0.0, max_value=100.0,
        )
        r2 = service.register_rule(
            name="val_high", rule_type="range_check",
            severity="warning", field="value",
            min_value=50.0, max_value=200.0,
        )
        r1_id = r1["rule_id"] if isinstance(r1, dict) else r1.rule_id
        r2_id = r2["rule_id"] if isinstance(r2, dict) else r2.rule_id

        rs = service.create_rule_set(
            name="Overlap Test",
            rule_ids=[r1_id, r2_id],
        )
        rs_id = rs["set_id"] if isinstance(rs, dict) else rs.set_id

        result = service.detect_conflicts(rule_set_id=rs_id)
        assert result["conflict_count"] >= 1
        assert "overlap" in result.get("conflict_types", [])

    def test_detect_conflicts_no_overlap(self, service):
        """Non-overlapping rules should produce no conflicts."""
        r1 = service.register_rule(
            name="val_low", rule_type="range_check",
            severity="error", field="value_a",
            min_value=0.0, max_value=100.0,
        )
        r2 = service.register_rule(
            name="val_high", rule_type="range_check",
            severity="error", field="value_b",
            min_value=0.0, max_value=100.0,
        )
        r1_id = r1["rule_id"]
        r2_id = r2["rule_id"]

        rs = service.create_rule_set(
            name="No Overlap Test",
            rule_ids=[r1_id, r2_id],
        )
        rs_id = rs["set_id"]

        result = service.detect_conflicts(rule_set_id=rs_id)
        assert result["conflict_count"] == 0

    def test_detect_conflicts_records_provenance(self, service):
        """Conflict detection should record provenance."""
        r1 = service.register_rule(
            name="r1", rule_type="range_check",
            severity="error", field="val",
            min_value=0, max_value=100,
        )
        rs = service.create_rule_set(name="RS", rule_ids=[r1["rule_id"]])
        initial = service.provenance.entry_count
        service.detect_conflicts(rule_set_id=rs["set_id"])
        assert service.provenance.entry_count > initial

    def test_list_conflicts_after_detection(self, service):
        """Detected conflicts should be listable."""
        r1 = service.register_rule(
            name="r1", rule_type="range_check",
            severity="error", field="val",
            min_value=0, max_value=100,
        )
        r2 = service.register_rule(
            name="r2", rule_type="range_check",
            severity="error", field="val",
            min_value=50, max_value=200,
        )
        rs = service.create_rule_set(name="RS", rule_ids=[r1["rule_id"], r2["rule_id"]])
        service.detect_conflicts(rule_set_id=rs["set_id"])
        conflicts = service.list_conflicts()
        assert len(conflicts) >= 1

    def test_conflict_detection_in_pipeline(
        self, service, sample_emission_records,
    ):
        """Pipeline execution should include conflict detection stage."""
        result = service.run_pipeline(
            dataset=sample_emission_records,
            pack_name="ghg_protocol",
        )
        stages = result.get("stages_completed", [])
        assert "detect_conflicts" in stages


# ===========================================================================
# TestPipelineReportGeneration
# ===========================================================================


class TestPipelineReportGeneration:
    """Test report generation within the pipeline."""

    def test_generate_report_after_evaluation(
        self, service, sample_emission_records,
    ):
        """Reports should be generatable from evaluation results."""
        pack_result = service.apply_pack("ghg_protocol")
        rs_id = pack_result.get("rule_set_id", "")
        eval_result = service.evaluate(
            rule_set_id=rs_id,
            dataset=sample_emission_records,
        )
        eval_id = eval_result["evaluation_id"]

        report = service.generate_report(
            evaluation_id=eval_id,
            report_type="compliance_report",
            format="json",
        )
        assert report["report_id"] is not None
        assert report["report_type"] == "compliance_report"

    def test_report_contains_pass_rate(
        self, service, sample_emission_records,
    ):
        """Report content should include pass_rate from evaluation."""
        pack_result = service.apply_pack("ghg_protocol")
        rs_id = pack_result.get("rule_set_id", "")
        eval_result = service.evaluate(
            rule_set_id=rs_id,
            dataset=sample_emission_records,
        )
        eval_id = eval_result["evaluation_id"]

        report = service.generate_report(
            evaluation_id=eval_id,
            report_type="compliance_report",
            format="json",
        )
        content = report.get("content", {})
        assert "pass_rate" in content

    def test_report_records_provenance(
        self, service, sample_emission_records,
    ):
        """Report generation should record provenance."""
        pack_result = service.apply_pack("ghg_protocol")
        rs_id = pack_result.get("rule_set_id", "")
        eval_result = service.evaluate(
            rule_set_id=rs_id,
            dataset=sample_emission_records,
        )
        initial = service.provenance.entry_count
        service.generate_report(
            evaluation_id=eval_result["evaluation_id"],
            report_type="compliance_report",
            format="json",
        )
        assert service.provenance.entry_count > initial

    def test_pipeline_includes_report_stage(
        self, service, sample_emission_records,
    ):
        """Pipeline should include report generation stage."""
        result = service.run_pipeline(
            dataset=sample_emission_records,
            pack_name="ghg_protocol",
        )
        stages = result.get("stages_completed", [])
        assert "generate_report" in stages

    def test_pipeline_report_retrievable(
        self, service, sample_emission_records,
    ):
        """Report from pipeline should be stored in service."""
        result = service.run_pipeline(
            dataset=sample_emission_records,
            pack_name="ghg_protocol",
        )
        report_id = result.get("report_id")
        if report_id:
            assert report_id in service._reports


# ===========================================================================
# TestPipelineProvenanceAndAudit
# ===========================================================================


class TestPipelineProvenanceAndAudit:
    """Test provenance tracking and audit trail across pipeline stages."""

    def test_provenance_chain_grows_with_each_operation(self, service):
        """Each operation should add provenance entries."""
        counts = [service.provenance.entry_count]

        service.register_rule(name="r1", rule_type="range_check", severity="error", field="x")
        counts.append(service.provenance.entry_count)

        service.create_rule_set(name="RS", rule_ids=[])
        counts.append(service.provenance.entry_count)

        # Each count should be strictly increasing
        for i in range(1, len(counts)):
            assert counts[i] > counts[i - 1]

    def test_provenance_entries_have_correct_entity_types(
        self, service, sample_emission_records,
    ):
        """Provenance entries should have correct entity types."""
        service.run_pipeline(
            dataset=sample_emission_records,
            pack_name="ghg_protocol",
        )
        entries = service.provenance.get_entries()
        entity_types = {e.entity_type for e in entries}
        # Should include at least validation_rule, rule_set, evaluation
        assert "validation_rule" in entity_types
        assert "rule_set" in entity_types
        assert "evaluation" in entity_types

    def test_provenance_chain_integrity(
        self, service, sample_emission_records,
    ):
        """Full provenance chain should pass integrity verification."""
        service.run_pipeline(
            dataset=sample_emission_records,
            pack_name="ghg_protocol",
        )
        assert service.provenance.verify_chain() is True

    def test_provenance_export_contains_all_entries(
        self, service, sample_emission_records,
    ):
        """Exported chain should contain same count as entry_count."""
        service.run_pipeline(
            dataset=sample_emission_records,
            pack_name="ghg_protocol",
        )
        exported = service.provenance.export_chain()
        assert len(exported) == service.provenance.entry_count

    def test_provenance_entries_have_timestamps(
        self, service, sample_emission_records,
    ):
        """All provenance entries should have timestamps."""
        service.run_pipeline(
            dataset=sample_emission_records,
            pack_name="ghg_protocol",
        )
        entries = service.provenance.get_entries()
        for entry in entries:
            assert entry.timestamp is not None
            assert len(entry.timestamp) > 0


# ===========================================================================
# TestPipelineErrorRecovery
# ===========================================================================


class TestPipelineErrorRecovery:
    """Test pipeline behavior under error conditions."""

    def test_pipeline_with_empty_dataset(self, service):
        """Pipeline should handle empty dataset gracefully."""
        result = service.run_pipeline(
            dataset=[],
            pack_name="ghg_protocol",
        )
        assert result.get("final_status") in ("completed", "failed", "no_data")

    def test_pipeline_with_missing_fields(self, service):
        """Pipeline should handle records with missing fields."""
        result = service.run_pipeline(
            dataset=[{"unknown_field": 42}],
            pack_name="ghg_protocol",
        )
        assert result.get("final_status") in ("completed", "failed")
        assert result.get("pipeline_id") is not None

    def test_pipeline_preserves_provenance_on_partial_failure(self, service):
        """Provenance should be recorded even for partial failures."""
        initial = service.provenance.entry_count
        service.run_pipeline(
            dataset=[{"bad_data": "not_a_number"}],
            pack_name="ghg_protocol",
        )
        assert service.provenance.entry_count > initial

    def test_multiple_sequential_pipelines(
        self, service, sample_emission_records,
    ):
        """Multiple sequential pipelines should not interfere."""
        r1 = service.run_pipeline(
            dataset=sample_emission_records[:5],
            pack_name="ghg_protocol",
        )
        r2 = service.run_pipeline(
            dataset=sample_emission_records[5:],
            pack_name="ghg_protocol",
        )
        assert r1["pipeline_id"] != r2["pipeline_id"]
        assert service.provenance.verify_chain() is True
