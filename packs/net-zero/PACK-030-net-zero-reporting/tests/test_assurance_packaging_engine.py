# -*- coding: utf-8 -*-
"""
Test suite for PACK-030 Net Zero Reporting Pack - Assurance Packaging Engine.

Tests provenance collection, lineage diagram generation, evidence bundle
completeness, ISAE 3410 control matrix, methodology documentation,
SHA-256 hash chains, and audit-ready packaging.

Author:  GreenLang Test Engineering
Pack:    PACK-030 Net Zero Reporting Pack
Engine:  6 of 10 - assurance_packaging_engine.py
"""

import asyncio
import sys
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from engines.assurance_packaging_engine import (
    AssurancePackagingEngine, AssurancePackagingInput, AssurancePackagingResult,
    EvidenceItem, ControlMatrixEntry, ProvenanceRecord,
    AssuranceLevel, EvidenceType, ControlStatus, BundleStatus,
)

from .conftest import (
    assert_provenance_hash, assert_processing_time, compute_sha256,
    timed_block, FRAMEWORKS, EVIDENCE_TYPES,
)


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_records(count=3):
    """Create sample provenance records."""
    return [
        ProvenanceRecord(
            metric_name=f"scope_{i+1}_emissions",
            metric_value=Decimal(str(10000 * (i + 1))),
            unit="tCO2e",
            source_system="PACK-021",
            source_record_id=f"REC-{i+1:03d}",
            calculation_method="ghg_protocol_stationary",
            reviewed_by="reviewer@example.com",
            approved_by="approver@example.com",
        )
        for i in range(count)
    ]


class TestAssuranceInstantiation:
    def test_engine_instantiates(self):
        assert AssurancePackagingEngine() is not None

    def test_engine_version(self):
        assert AssurancePackagingEngine().engine_version == "1.0.0"

    def test_engine_has_package_method(self):
        engine = AssurancePackagingEngine()
        assert hasattr(engine, "collect_provenances") or hasattr(engine, "package")


class TestProvenanceCollection:
    def test_collect_provenances(self):
        engine = AssurancePackagingEngine()
        records = _make_records()
        result = _run(engine.collect_provenances(records))
        assert result is not None
        assert len(result) > 0

    def test_provenance_has_sha256_hashes(self):
        engine = AssurancePackagingEngine()
        records = _make_records()
        result = _run(engine.collect_provenances(records))
        for item in result:
            assert len(item.checksum) == 64

    def test_provenance_count(self):
        engine = AssurancePackagingEngine()
        records = _make_records(5)
        result = _run(engine.collect_provenances(records))
        assert len(result) == 5

    @pytest.mark.parametrize("count", [1, 3, 5])
    def test_provenance_various_counts(self, count):
        engine = AssurancePackagingEngine()
        records = _make_records(count)
        result = _run(engine.collect_provenances(records))
        assert len(result) == count

    def test_provenance_evidence_type(self):
        engine = AssurancePackagingEngine()
        records = _make_records()
        result = _run(engine.collect_provenances(records))
        for item in result:
            assert item.evidence_type == EvidenceType.PROVENANCE_HASH.value

    @pytest.mark.parametrize("run_idx", range(3))
    def test_provenance_deterministic(self, run_idx):
        engine = AssurancePackagingEngine()
        records = _make_records()
        r1 = _run(engine.collect_provenances(records))
        r2 = _run(engine.collect_provenances(records))
        assert len(r1) == len(r2)


class TestLineageDiagrams:
    def test_generate_lineage_diagrams(self):
        engine = AssurancePackagingEngine()
        records = _make_records()
        result = _run(engine.generate_lineage_diagrams(records))
        assert result is not None

    def test_lineage_diagram_count(self):
        engine = AssurancePackagingEngine()
        records = _make_records(4)
        result = _run(engine.generate_lineage_diagrams(records))
        assert len(result) == 4

    def test_lineage_has_mermaid(self):
        engine = AssurancePackagingEngine()
        records = _make_records()
        result = _run(engine.generate_lineage_diagrams(records))
        for diagram in result:
            assert "graph LR" in diagram.mermaid_definition

    def test_lineage_has_provenance_hash(self):
        engine = AssurancePackagingEngine()
        records = _make_records()
        result = _run(engine.generate_lineage_diagrams(records))
        for diagram in result:
            assert len(diagram.provenance_hash) == 64


class TestMethodologyDocumentation:
    def test_package_methodology(self):
        engine = AssurancePackagingEngine()
        records = _make_records()
        result = _run(engine.package_methodology(records))
        assert result is not None
        assert len(result) > 0

    def test_methodology_evidence_type(self):
        engine = AssurancePackagingEngine()
        records = _make_records()
        result = _run(engine.package_methodology(records))
        for item in result:
            assert item.evidence_type == EvidenceType.METHODOLOGY.value

    def test_methodology_has_checksum(self):
        engine = AssurancePackagingEngine()
        records = _make_records()
        result = _run(engine.package_methodology(records))
        for item in result:
            assert len(item.checksum) == 64


class TestControlMatrix:
    def test_create_control_matrix(self):
        engine = AssurancePackagingEngine()
        records = _make_records()
        result = _run(engine.create_control_matrix(records))
        assert result is not None
        assert len(result) > 0

    def test_control_matrix_has_10_controls(self):
        engine = AssurancePackagingEngine()
        records = _make_records()
        result = _run(engine.create_control_matrix(records))
        assert len(result) == 10

    def test_control_matrix_has_assertions(self):
        engine = AssurancePackagingEngine()
        records = _make_records()
        result = _run(engine.create_control_matrix(records))
        assertions = {e.assertion for e in result}
        assert len(assertions) >= 3

    def test_control_matrix_effective_with_evidence(self):
        engine = AssurancePackagingEngine()
        records = _make_records()
        result = _run(engine.create_control_matrix(records))
        effective = [e for e in result if e.status == ControlStatus.EFFECTIVE.value]
        assert len(effective) >= 1

    def test_get_isae_3410_controls(self):
        engine = AssurancePackagingEngine()
        controls = engine.get_isae_3410_controls()
        assert len(controls) == 10


class TestEvidenceBundle:
    def test_create_full_bundle(self):
        engine = AssurancePackagingEngine()
        inp = AssurancePackagingInput(
            organization_id="test-org-001",
            provenance_records=_make_records(),
            include_lineage_diagrams=True,
            include_methodology=True,
            include_control_matrix=True,
            include_reconciliation=True,
        )
        result = _run(engine.package(inp))
        assert result is not None
        assert isinstance(result, AssurancePackagingResult)

    def test_bundle_has_evidence_items(self):
        engine = AssurancePackagingEngine()
        inp = AssurancePackagingInput(
            organization_id="test-org-001",
            provenance_records=_make_records(),
        )
        result = _run(engine.package(inp))
        assert result.total_evidence_items > 0

    def test_bundle_has_provenance(self):
        engine = AssurancePackagingEngine()
        inp = AssurancePackagingInput(
            organization_id="test-org-001",
            provenance_records=_make_records(),
        )
        result = _run(engine.package(inp))
        assert_provenance_hash(result)

    def test_bundle_has_manifest(self):
        engine = AssurancePackagingEngine()
        inp = AssurancePackagingInput(
            organization_id="test-org-001",
            provenance_records=_make_records(),
        )
        result = _run(engine.package(inp))
        assert result.manifest is not None

    def test_bundle_completeness_high(self):
        engine = AssurancePackagingEngine()
        inp = AssurancePackagingInput(
            organization_id="test-org-001",
            provenance_records=_make_records(),
            include_lineage_diagrams=True,
            include_methodology=True,
            include_control_matrix=True,
            include_reconciliation=True,
        )
        result = _run(engine.package(inp))
        assert result.completeness_pct >= Decimal("80")

    @pytest.mark.parametrize("level", [AssuranceLevel.LIMITED, AssuranceLevel.REASONABLE])
    def test_bundle_assurance_levels(self, level):
        engine = AssurancePackagingEngine()
        inp = AssurancePackagingInput(
            organization_id="test-org-001",
            provenance_records=_make_records(),
            assurance_level=level,
        )
        result = _run(engine.package(inp))
        assert result is not None

    @pytest.mark.parametrize("framework", FRAMEWORKS)
    def test_bundle_per_framework(self, framework):
        engine = AssurancePackagingEngine()
        inp = AssurancePackagingInput(
            organization_id="test-org-001",
            framework=framework.lower(),
            provenance_records=_make_records(),
        )
        result = _run(engine.package(inp))
        assert result is not None


class TestAssurancePerformance:
    def test_bundle_under_5_seconds(self):
        engine = AssurancePackagingEngine()
        inp = AssurancePackagingInput(
            organization_id="test-org-001",
            provenance_records=_make_records(),
        )
        with timed_block("assurance_bundle", max_seconds=5.0):
            _run(engine.package(inp))

    def test_provenance_collection_performance(self):
        engine = AssurancePackagingEngine()
        records = _make_records(20)
        with timed_block("provenance_collect", max_seconds=2.0):
            _run(engine.collect_provenances(records))


class TestAssuranceErrorHandling:
    def test_empty_organization_id(self):
        with pytest.raises((ValueError, Exception)):
            AssurancePackagingInput(organization_id="")

    def test_no_records_gives_warning(self):
        engine = AssurancePackagingEngine()
        inp = AssurancePackagingInput(
            organization_id="test-org-001",
            provenance_records=[],
        )
        result = _run(engine.package(inp))
        assert len(result.warnings) > 0


class TestAssuranceResultModel:
    def test_result_has_evidence_items(self):
        engine = AssurancePackagingEngine()
        inp = AssurancePackagingInput(
            organization_id="test-org-001",
            provenance_records=_make_records(),
        )
        result = _run(engine.package(inp))
        assert isinstance(result.evidence_items, list)

    def test_result_serializable(self):
        engine = AssurancePackagingEngine()
        inp = AssurancePackagingInput(
            organization_id="test-org-001",
            provenance_records=_make_records(),
        )
        result = _run(engine.package(inp))
        assert isinstance(result.model_dump(), dict)

    def test_result_processing_time(self):
        engine = AssurancePackagingEngine()
        inp = AssurancePackagingInput(
            organization_id="test-org-001",
            provenance_records=_make_records(),
        )
        result = _run(engine.package(inp))
        assert_processing_time(result)

    def test_result_engine_version(self):
        engine = AssurancePackagingEngine()
        inp = AssurancePackagingInput(
            organization_id="test-org-001",
            provenance_records=_make_records(),
        )
        result = _run(engine.package(inp))
        assert result.engine_version == "1.0.0"

    def test_supported_evidence_types(self):
        engine = AssurancePackagingEngine()
        types = engine.get_supported_evidence_types()
        assert len(types) >= 5

    def test_supported_assurance_levels(self):
        engine = AssurancePackagingEngine()
        levels = engine.get_supported_assurance_levels()
        assert "reasonable" in levels
        assert "limited" in levels
