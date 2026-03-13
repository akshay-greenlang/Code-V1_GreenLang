# -*- coding: utf-8 -*-
"""
Unit tests for PackageAssembler engine - AGENT-EUDR-036

Tests document package assembly, ordering, size validation,
completeness checking, and manifest generation.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import pytest

from greenlang.agents.eudr.eu_information_system_interface.config import (
    EUInformationSystemInterfaceConfig,
)
from greenlang.agents.eudr.eu_information_system_interface.package_assembler import (
    PackageAssembler,
)
from greenlang.agents.eudr.eu_information_system_interface.provenance import (
    ProvenanceTracker,
)


@pytest.fixture
def assembler() -> PackageAssembler:
    """Create a PackageAssembler instance."""
    config = EUInformationSystemInterfaceConfig()
    return PackageAssembler(config=config, provenance=ProvenanceTracker())


class TestAssemblePackage:
    """Test PackageAssembler.assemble_package()."""

    @pytest.mark.asyncio
    async def test_assemble_basic(self, assembler, sample_documents):
        package = await assembler.assemble_package(
            dds_id="dds-001",
            documents=sample_documents,
        )
        assert package.package_id.startswith("pkg-")
        assert package.dds_id == "dds-001"
        assert package.document_count == 3

    @pytest.mark.asyncio
    async def test_assemble_provenance_hash(self, assembler, sample_documents):
        package = await assembler.assemble_package(
            dds_id="dds-001",
            documents=sample_documents,
        )
        assert len(package.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_documents_ordered(self, assembler, sample_documents):
        package = await assembler.assemble_package(
            dds_id="dds-001",
            documents=sample_documents,
        )
        # DDS form should come first per EU IS ordering
        types = [d.get("type") for d in package.documents]
        assert types[0] == "dds_form"

    @pytest.mark.asyncio
    async def test_document_hashes_added(self, assembler, sample_documents):
        package = await assembler.assemble_package(
            dds_id="dds-001",
            documents=sample_documents,
        )
        for doc in package.documents:
            if doc.get("content"):
                assert "hash" in doc
                assert len(doc["hash"]) == 64

    @pytest.mark.asyncio
    async def test_too_many_documents_raises(self, assembler):
        config = EUInformationSystemInterfaceConfig(max_attachments_per_package=2)
        asm = PackageAssembler(config=config, provenance=ProvenanceTracker())
        docs = [
            {"type": "dds_form", "content": {"a": 1}, "size_bytes": 100},
            {"type": "geolocation_data", "content": {"b": 2}, "size_bytes": 100},
            {"type": "risk_assessment", "content": {"c": 3}, "size_bytes": 100},
        ]
        with pytest.raises(ValueError, match="exceeds maximum"):
            await asm.assemble_package(dds_id="dds-001", documents=docs)

    @pytest.mark.asyncio
    async def test_document_too_large_raises(self, assembler):
        config = EUInformationSystemInterfaceConfig(max_attachment_size_bytes=100)
        asm = PackageAssembler(config=config, provenance=ProvenanceTracker())
        docs = [
            {"type": "dds_form", "content": {}, "size_bytes": 200},
        ]
        with pytest.raises(ValueError, match="exceeds max"):
            await asm.assemble_package(dds_id="dds-001", documents=docs)

    @pytest.mark.asyncio
    async def test_total_size_tracked(self, assembler, sample_documents):
        package = await assembler.assemble_package(
            dds_id="dds-001",
            documents=sample_documents,
        )
        assert package.total_size_bytes > 0

    @pytest.mark.asyncio
    async def test_compression_flag(self, assembler, sample_documents):
        package = await assembler.assemble_package(
            dds_id="dds-001",
            documents=sample_documents,
        )
        # Default config has compress=True
        assert package.compressed is True

    @pytest.mark.asyncio
    async def test_empty_documents(self, assembler):
        package = await assembler.assemble_package(
            dds_id="dds-001",
            documents=[],
        )
        assert package.document_count == 0


class TestValidatePackageCompleteness:
    """Test PackageAssembler.validate_package_completeness()."""

    @pytest.mark.asyncio
    async def test_complete_package(self, assembler, sample_package):
        result = await assembler.validate_package_completeness(sample_package)
        assert result["complete"] is True
        assert len(result["missing_types"]) == 0

    @pytest.mark.asyncio
    async def test_missing_required_type(self, assembler):
        from greenlang.agents.eudr.eu_information_system_interface.models import DocumentPackage
        pkg = DocumentPackage(
            package_id="pkg-001",
            dds_id="dds-001",
            documents=[{"type": "risk_assessment"}],
            document_count=1,
        )
        result = await assembler.validate_package_completeness(pkg)
        assert result["complete"] is False
        assert "dds_form" in result["missing_types"]

    @pytest.mark.asyncio
    async def test_custom_required_types(self, assembler, sample_package):
        result = await assembler.validate_package_completeness(
            sample_package,
            required_types=["dds_form", "certificate"],
        )
        assert result["complete"] is False
        assert "certificate" in result["missing_types"]


class TestGenerateManifest:
    """Test PackageAssembler.generate_manifest()."""

    @pytest.mark.asyncio
    async def test_manifest_basic(self, assembler, sample_package):
        manifest = await assembler.generate_manifest(sample_package)
        assert manifest["package_id"] == "pkg-test-001"
        assert manifest["dds_id"] == "dds-test-001"
        assert manifest["document_count"] == 2

    @pytest.mark.asyncio
    async def test_manifest_has_entries(self, assembler, sample_package):
        manifest = await assembler.generate_manifest(sample_package)
        assert len(manifest["entries"]) == 2

    @pytest.mark.asyncio
    async def test_manifest_has_timestamp(self, assembler, sample_package):
        manifest = await assembler.generate_manifest(sample_package)
        assert "generated_at" in manifest


class TestHealthCheck:
    """Test PackageAssembler.health_check()."""

    @pytest.mark.asyncio
    async def test_health_check(self, assembler):
        health = await assembler.health_check()
        assert health["engine"] == "PackageAssembler"
        assert health["status"] == "available"
