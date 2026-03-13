# -*- coding: utf-8 -*-
"""
Unit tests for DocumentPackager - AGENT-EUDR-037

Tests document addition, submission package creation, validation,
manifests, and health checks.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import pytest

from greenlang.agents.eudr.due_diligence_statement_creator.config import DDSCreatorConfig
from greenlang.agents.eudr.due_diligence_statement_creator.document_packager import DocumentPackager
from greenlang.agents.eudr.due_diligence_statement_creator.models import (
    DocumentPackage, DocumentType, SubmissionPackage, SubmissionStatus,
)


@pytest.fixture
def config():
    return DDSCreatorConfig()


@pytest.fixture
def packager(config):
    return DocumentPackager(config=config)


class TestAddDocument:
    @pytest.mark.asyncio
    async def test_returns_document_package(self, packager):
        doc = await packager.add_document(
            document_type="certificate_of_origin",
            filename="cert.pdf")
        assert isinstance(doc, DocumentPackage)

    @pytest.mark.asyncio
    async def test_document_id_prefix(self, packager):
        doc = await packager.add_document(
            document_type="certificate_of_origin",
            filename="cert.pdf")
        assert doc.document_id.startswith("DOC-")

    @pytest.mark.asyncio
    async def test_document_type_parsed(self, packager):
        doc = await packager.add_document(
            document_type="satellite_imagery",
            filename="image.tiff")
        assert doc.document_type == DocumentType.SATELLITE_IMAGERY

    @pytest.mark.asyncio
    async def test_invalid_type_defaults_other(self, packager):
        doc = await packager.add_document(
            document_type="invalid_xyz",
            filename="test.pdf")
        assert doc.document_type == DocumentType.OTHER

    @pytest.mark.asyncio
    async def test_filename_set(self, packager):
        doc = await packager.add_document(
            document_type="certificate_of_origin",
            filename="my_cert.pdf")
        assert doc.filename == "my_cert.pdf"

    @pytest.mark.asyncio
    async def test_size_bytes_set(self, packager):
        doc = await packager.add_document(
            document_type="certificate_of_origin",
            filename="cert.pdf", size_bytes=1024000)
        assert doc.size_bytes == 1024000

    @pytest.mark.asyncio
    async def test_mime_type_set(self, packager):
        doc = await packager.add_document(
            document_type="certificate_of_origin",
            filename="cert.pdf", mime_type="application/pdf")
        assert doc.mime_type == "application/pdf"

    @pytest.mark.asyncio
    async def test_hash_sha256_set(self, packager):
        doc = await packager.add_document(
            document_type="certificate_of_origin",
            filename="cert.pdf", hash_sha256="a" * 64)
        assert doc.hash_sha256 == "a" * 64

    @pytest.mark.asyncio
    async def test_provenance_hash_set(self, packager):
        doc = await packager.add_document(
            document_type="certificate_of_origin",
            filename="cert.pdf")
        assert len(doc.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_description_set(self, packager):
        doc = await packager.add_document(
            document_type="certificate_of_origin",
            filename="cert.pdf", description="Test description")
        assert doc.description == "Test description"

    @pytest.mark.asyncio
    async def test_language_set(self, packager):
        doc = await packager.add_document(
            document_type="certificate_of_origin",
            filename="cert.pdf", language="fr")
        assert doc.language == "fr"

    @pytest.mark.asyncio
    async def test_documents_added_count(self, packager):
        await packager.add_document(
            document_type="certificate_of_origin", filename="a.pdf")
        await packager.add_document(
            document_type="satellite_imagery", filename="b.tiff")
        health = await packager.health_check()
        assert health["documents_added"] == 2


class TestCreateSubmissionPackage:
    @pytest.mark.asyncio
    async def test_returns_submission_package(self, packager, sample_statement):
        pkg = await packager.create_submission_package(sample_statement)
        assert isinstance(pkg, SubmissionPackage)

    @pytest.mark.asyncio
    async def test_package_id_prefix(self, packager, sample_statement):
        pkg = await packager.create_submission_package(sample_statement)
        assert pkg.package_id.startswith("PKG-")

    @pytest.mark.asyncio
    async def test_statement_id_linked(self, packager, sample_statement):
        pkg = await packager.create_submission_package(sample_statement)
        assert pkg.statement_id == sample_statement.statement_id

    @pytest.mark.asyncio
    async def test_operator_id_linked(self, packager, sample_statement):
        pkg = await packager.create_submission_package(sample_statement)
        assert pkg.operator_id == sample_statement.operator_id

    @pytest.mark.asyncio
    async def test_pending_status(self, packager, sample_statement):
        pkg = await packager.create_submission_package(sample_statement)
        assert pkg.submission_status == SubmissionStatus.PENDING

    @pytest.mark.asyncio
    async def test_with_documents(self, packager, sample_statement, sample_document):
        pkg = await packager.create_submission_package(
            sample_statement, documents=[sample_document])
        assert pkg.document_count >= 1
        assert pkg.total_size_bytes >= sample_document.size_bytes

    @pytest.mark.asyncio
    async def test_no_documents(self, packager, sample_statement):
        pkg = await packager.create_submission_package(sample_statement)
        assert pkg.document_count == 0

    @pytest.mark.asyncio
    async def test_provenance_hash_set(self, packager, sample_statement):
        pkg = await packager.create_submission_package(sample_statement)
        assert len(pkg.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_packages_created_count(self, packager, sample_statement):
        await packager.create_submission_package(sample_statement)
        await packager.create_submission_package(sample_statement)
        health = await packager.health_check()
        assert health["packages_created"] == 2


class TestValidatePackage:
    @pytest.mark.asyncio
    async def test_valid_package(self, packager, sample_statement, sample_document):
        pkg = await packager.create_submission_package(
            sample_statement, documents=[sample_document])
        result = await packager.validate_package(pkg)
        assert result["valid"] is True

    @pytest.mark.asyncio
    async def test_empty_package_fails(self, packager, sample_statement):
        pkg = await packager.create_submission_package(sample_statement)
        result = await packager.validate_package(pkg)
        assert result["valid"] is False
        assert any("no documents" in i.lower() for i in result["issues"])

    @pytest.mark.asyncio
    async def test_oversized_package_fails(self, packager):
        pkg = SubmissionPackage(
            package_id="PKG-X", statement_id="DDS-X",
            operator_id="OP-X",
            document_count=1,
            total_size_bytes=600 * 1024 * 1024)  # 600MB exceeds 500MB
        result = await packager.validate_package(pkg)
        assert result["valid"] is False
        assert any("size" in i.lower() for i in result["issues"])


class TestGetManifest:
    @pytest.mark.asyncio
    async def test_manifest_returns_dict(self, packager, sample_document):
        manifest = await packager.get_manifest([sample_document])
        assert isinstance(manifest, dict)
        assert manifest["total_documents"] == 1

    @pytest.mark.asyncio
    async def test_manifest_empty(self, packager):
        manifest = await packager.get_manifest([])
        assert manifest["total_documents"] == 0
        assert manifest["total_size_bytes"] == 0

    @pytest.mark.asyncio
    async def test_manifest_document_details(self, packager, sample_document):
        manifest = await packager.get_manifest([sample_document])
        assert len(manifest["documents"]) == 1
        assert manifest["documents"][0]["filename"] == sample_document.filename


class TestDocumentPackagerHealth:
    @pytest.mark.asyncio
    async def test_health_check(self, packager):
        health = await packager.health_check()
        assert health["engine"] == "DocumentPackager"
        assert health["status"] == "healthy"
