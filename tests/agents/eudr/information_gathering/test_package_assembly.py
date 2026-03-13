# -*- coding: utf-8 -*-
"""
Unit tests for PackageAssemblyEngine - AGENT-EUDR-027

Tests information package assembly, evidence artifact creation, package
hash computation (deterministic), package validation, package diffing,
assembly statistics, versioning, and provenance chain construction.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-027 (Engine 7: Package Assembly)
"""
from __future__ import annotations

from decimal import Decimal
from typing import Dict, List

import pytest

from greenlang.agents.eudr.information_gathering.package_assembly_engine import (
    PackageAssemblyEngine,
)
from greenlang.agents.eudr.information_gathering.models import (
    Article9ElementName,
    Article9ElementStatus,
    CompletenessClassification,
    CompletenessReport,
    ElementStatus,
    EUDRCommodity,
    EvidenceArtifact,
    GapReport,
    InformationPackage,
    NormalizationRecord,
    NormalizationType,
    PackageDiff,
    QueryResult,
    SupplierProfile,
)


@pytest.fixture
def engine(config):
    return PackageAssemblyEngine(config)


@pytest.fixture
def sample_completeness_report(sample_article9_elements):
    """Create a CompletenessReport for use in assembly calls."""
    return CompletenessReport(
        operation_id="OP-TEST-001",
        commodity=EUDRCommodity.COFFEE,
        elements=list(sample_article9_elements.values()),
        completeness_score=Decimal("95.00"),
        completeness_classification=CompletenessClassification.COMPLETE,
        gap_report=GapReport(total_gaps=0, critical_gaps=0),
        provenance_hash="e" * 64,
    )


@pytest.fixture
def sample_normalization_log():
    """Create a normalization log for assembly."""
    return [
        NormalizationRecord(
            field_name="country",
            source_value="Brazil",
            normalized_value="BR",
            normalization_type=NormalizationType.COUNTRY_CODE,
            confidence=Decimal("1.0"),
        ),
    ]


def _assemble(engine, sample_article9_elements, sample_completeness_report,
              sample_normalization_log, operator_id="OP-DE-001",
              commodity=EUDRCommodity.COFFEE, supplier_profiles=None,
              cert_results=None, query_results=None, public_data=None):
    """Helper to call assemble_package with all required arguments."""
    return engine.assemble_package(
        operation_id="op_test_001",
        operator_id=operator_id,
        commodity=commodity,
        article_9_elements=sample_article9_elements,
        completeness_report=sample_completeness_report,
        supplier_profiles=supplier_profiles or [],
        query_results=query_results or {},
        cert_results=cert_results or [],
        public_data=public_data or {},
        normalization_log=sample_normalization_log,
    )


# ---------------------------------------------------------------------------
# Package Assembly
# ---------------------------------------------------------------------------


class TestAssemblePackage:
    """Test package assembly operations."""

    def test_assemble_package_basic(
        self, engine, sample_article9_elements,
        sample_completeness_report, sample_normalization_log,
    ):
        package = _assemble(
            engine, sample_article9_elements,
            sample_completeness_report, sample_normalization_log,
        )
        assert isinstance(package, InformationPackage)
        assert package.operator_id == "OP-DE-001"
        assert package.commodity == EUDRCommodity.COFFEE
        assert len(package.package_hash) == 64
        assert package.version == 1

    def test_assemble_package_with_suppliers(
        self, engine, sample_article9_elements,
        sample_completeness_report, sample_normalization_log,
        sample_supplier_profile,
    ):
        package = _assemble(
            engine, sample_article9_elements,
            sample_completeness_report, sample_normalization_log,
            supplier_profiles=[sample_supplier_profile],
        )
        assert len(package.supplier_profiles) == 1
        assert package.supplier_profiles[0].supplier_id == "SUP-001"

    def test_assemble_package_with_certifications(
        self, engine, sample_article9_elements,
        sample_completeness_report, sample_normalization_log,
        sample_cert_result,
    ):
        package = _assemble(
            engine, sample_article9_elements,
            sample_completeness_report, sample_normalization_log,
            commodity=EUDRCommodity.WOOD,
            cert_results=[sample_cert_result],
        )
        assert len(package.certification_results) == 1

    def test_assemble_package_completeness_score(
        self, engine, sample_article9_elements,
        sample_completeness_report, sample_normalization_log,
    ):
        package = _assemble(
            engine, sample_article9_elements,
            sample_completeness_report, sample_normalization_log,
        )
        assert package.completeness_score == Decimal("95.00")
        assert package.completeness_classification == "complete"

    def test_assemble_package_valid_until(
        self, engine, sample_article9_elements,
        sample_completeness_report, sample_normalization_log,
    ):
        package = _assemble(
            engine, sample_article9_elements,
            sample_completeness_report, sample_normalization_log,
        )
        assert package.valid_until is not None
        assert package.valid_until > package.assembled_at


# ---------------------------------------------------------------------------
# Evidence Artifact Creation
# ---------------------------------------------------------------------------


class TestCreateEvidenceArtifact:
    """Test evidence artifact creation."""

    def test_create_evidence_artifact(self, engine):
        artifact = engine.create_evidence_artifact(
            article_9_element="geolocation",
            source="satellite_imagery",
            data={"lat": 1.234, "lon": -5.678},
            format="json",
        )
        assert isinstance(artifact, EvidenceArtifact)
        assert artifact.article_9_element == "geolocation"
        assert len(artifact.content_hash) == 64
        assert artifact.s3_path is not None
        assert artifact.s3_path.startswith("s3://")

    def test_create_evidence_artifact_non_dict_data(self, engine):
        artifact = engine.create_evidence_artifact(
            article_9_element="quantity",
            source="erp_system",
            data="1000 kg coffee beans",
            format="csv",
        )
        assert len(artifact.content_hash) == 64

    def test_evidence_artifacts_in_package(
        self, engine, sample_article9_elements,
        sample_completeness_report, sample_normalization_log,
    ):
        package = _assemble(
            engine, sample_article9_elements,
            sample_completeness_report, sample_normalization_log,
        )
        # Should have at least one artifact per Article 9 element
        assert len(package.evidence_artifacts) >= 10


# ---------------------------------------------------------------------------
# Package Hash Computation
# ---------------------------------------------------------------------------


class TestComputePackageHash:
    """Test package hash computation."""

    def test_compute_package_hash_length(
        self, engine, sample_article9_elements,
        sample_completeness_report, sample_normalization_log,
    ):
        package = _assemble(
            engine, sample_article9_elements,
            sample_completeness_report, sample_normalization_log,
        )
        assert len(package.package_hash) == 64

    def test_compute_package_hash_recompute_matches(
        self, engine, sample_article9_elements,
        sample_completeness_report, sample_normalization_log,
    ):
        package = _assemble(
            engine, sample_article9_elements,
            sample_completeness_report, sample_normalization_log,
        )
        recomputed = engine.compute_package_hash(package)
        assert recomputed == package.package_hash


# ---------------------------------------------------------------------------
# Package Validation
# ---------------------------------------------------------------------------


class TestValidatePackage:
    """Test package validation."""

    def test_validate_package_valid(
        self, engine, sample_article9_elements,
        sample_completeness_report, sample_normalization_log,
    ):
        package = _assemble(
            engine, sample_article9_elements,
            sample_completeness_report, sample_normalization_log,
        )
        result = engine.validate_package(package)
        assert isinstance(result, dict)
        assert result["is_valid"] is True
        assert result["checks"]["hash_integrity"] is True
        assert result["checks"]["elements_complete"] is True

    def test_validate_package_tampered_hash(
        self, engine, sample_article9_elements,
        sample_completeness_report, sample_normalization_log,
    ):
        package = _assemble(
            engine, sample_article9_elements,
            sample_completeness_report, sample_normalization_log,
        )
        package.package_hash = "tampered_hash_value_" + "0" * 44
        result = engine.validate_package(package)
        assert result["is_valid"] is False
        assert result["checks"]["hash_integrity"] is False
        assert len(result["errors"]) >= 1

    def test_validate_package_missing_elements(self, engine, sample_completeness_report, sample_normalization_log):
        partial_elements = {}
        for elem in list(Article9ElementName)[:5]:
            partial_elements[elem.value] = Article9ElementStatus(
                element_name=elem.value,
                status=ElementStatus.COMPLETE,
                confidence=Decimal("0.95"),
            )
        package = engine.assemble_package(
            operation_id="op_val_missing",
            operator_id="OP-VAL-003",
            commodity=EUDRCommodity.COFFEE,
            article_9_elements=partial_elements,
            completeness_report=sample_completeness_report,
            supplier_profiles=[],
            query_results={},
            cert_results=[],
            public_data={},
            normalization_log=sample_normalization_log,
        )
        result = engine.validate_package(package)
        assert result["checks"]["elements_complete"] is False


# ---------------------------------------------------------------------------
# Package Diffing
# ---------------------------------------------------------------------------


class TestDiffPackages:
    """Test package diffing."""

    def test_diff_packages_identical(
        self, engine, sample_article9_elements,
        sample_completeness_report, sample_normalization_log,
    ):
        pkg1 = _assemble(
            engine, sample_article9_elements,
            sample_completeness_report, sample_normalization_log,
            operator_id="OP-DIFF-001",
        )
        pkg2 = _assemble(
            engine, sample_article9_elements,
            sample_completeness_report, sample_normalization_log,
            operator_id="OP-DIFF-001",
        )
        diff = engine.diff_packages(pkg1, pkg2)
        assert isinstance(diff, PackageDiff)
        assert diff.added_elements == []
        assert diff.removed_elements == []

    def test_diff_packages_removed_elements(
        self, engine, sample_article9_elements,
        sample_completeness_report, sample_normalization_log,
    ):
        pkg1 = _assemble(
            engine, sample_article9_elements,
            sample_completeness_report, sample_normalization_log,
            operator_id="OP-DIFF-002",
        )
        partial_elements = {
            k: v for k, v in sample_article9_elements.items()
            if k not in ("geolocation", "deforestation_free_evidence")
        }
        pkg2 = engine.assemble_package(
            operation_id="op_diff_002b",
            operator_id="OP-DIFF-002",
            commodity=EUDRCommodity.COFFEE,
            article_9_elements=partial_elements,
            completeness_report=sample_completeness_report,
            supplier_profiles=[],
            query_results={},
            cert_results=[],
            public_data={},
            normalization_log=sample_normalization_log,
        )
        diff = engine.diff_packages(pkg1, pkg2)
        assert "geolocation" in diff.removed_elements
        assert "deforestation_free_evidence" in diff.removed_elements

    def test_diff_packages_score_delta(
        self, engine, sample_article9_elements,
        sample_completeness_report, sample_normalization_log,
    ):
        pkg1 = _assemble(
            engine, sample_article9_elements,
            sample_completeness_report, sample_normalization_log,
            operator_id="OP-DIFF-003",
        )
        pkg2 = _assemble(
            engine, sample_article9_elements,
            sample_completeness_report, sample_normalization_log,
            operator_id="OP-DIFF-003",
        )
        diff = engine.diff_packages(pkg1, pkg2)
        assert diff.score_delta == Decimal("0.00")


# ---------------------------------------------------------------------------
# Assembly Statistics
# ---------------------------------------------------------------------------


class TestAssemblyStats:
    """Test assembly statistics."""

    def test_assembly_stats_after_one(
        self, engine, sample_article9_elements,
        sample_completeness_report, sample_normalization_log,
    ):
        _assemble(
            engine, sample_article9_elements,
            sample_completeness_report, sample_normalization_log,
        )
        stats = engine.get_assembly_stats()
        assert stats["total_assembled"] == 1
        assert stats["packages_stored"] == 1
        assert "coffee" in stats["commodity_breakdown"]
        assert stats["average_completeness"] > 0

    def test_assembly_stats_empty(self, engine):
        stats = engine.get_assembly_stats()
        assert stats["total_assembled"] == 0
        assert stats["packages_stored"] == 0

    def test_clear_store(
        self, engine, sample_article9_elements,
        sample_completeness_report, sample_normalization_log,
    ):
        _assemble(
            engine, sample_article9_elements,
            sample_completeness_report, sample_normalization_log,
        )
        engine.clear_store()
        stats = engine.get_assembly_stats()
        assert stats["total_assembled"] == 0


# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------


class TestVersioning:
    """Test package version auto-increment."""

    def test_version_increments(
        self, engine, sample_article9_elements,
        sample_completeness_report, sample_normalization_log,
    ):
        pkg1 = _assemble(
            engine, sample_article9_elements,
            sample_completeness_report, sample_normalization_log,
            operator_id="OP-VER-001",
        )
        pkg2 = _assemble(
            engine, sample_article9_elements,
            sample_completeness_report, sample_normalization_log,
            operator_id="OP-VER-001",
        )
        assert pkg1.version == 1
        assert pkg2.version == 2


# ---------------------------------------------------------------------------
# Provenance Chain
# ---------------------------------------------------------------------------


class TestProvenanceChain:
    """Test provenance chain built during assembly."""

    def test_provenance_chain_built(
        self, engine, sample_article9_elements,
        sample_completeness_report, sample_normalization_log,
    ):
        package = _assemble(
            engine, sample_article9_elements,
            sample_completeness_report, sample_normalization_log,
        )
        assert len(package.provenance_chain) >= 4  # collect, normalize, validate, assemble
        assert len(package.package_hash) == 64

    def test_provenance_chain_steps(
        self, engine, sample_article9_elements,
        sample_completeness_report, sample_normalization_log,
    ):
        package = _assemble(
            engine, sample_article9_elements,
            sample_completeness_report, sample_normalization_log,
        )
        step_names = [entry.step for entry in package.provenance_chain]
        assert "collect" in step_names
        assert "normalize" in step_names
        assert "validate" in step_names
        assert "assemble" in step_names


# ---------------------------------------------------------------------------
# Package Retrieval
# ---------------------------------------------------------------------------


class TestPackageRetrieval:
    """Test package get/store operations."""

    def test_get_package(
        self, engine, sample_article9_elements,
        sample_completeness_report, sample_normalization_log,
    ):
        package = _assemble(
            engine, sample_article9_elements,
            sample_completeness_report, sample_normalization_log,
        )
        retrieved = engine.get_package(package.package_id)
        assert retrieved is not None
        assert retrieved.package_id == package.package_id

    def test_get_package_not_found(self, engine):
        assert engine.get_package("NONEXISTENT") is None
