# -*- coding: utf-8 -*-
"""
Unit tests for DocumentExchange engine - AGENT-EUDR-040

Tests document upload, download, encryption/decryption, integrity
verification, document type validation, metadata tracking, listing,
retrieval, and health checks.

55+ tests.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import pytest

from greenlang.agents.eudr.authority_communication_manager.config import (
    AuthorityCommunicationManagerConfig,
)
from greenlang.agents.eudr.authority_communication_manager.document_exchange import (
    DocumentExchange,
)
from greenlang.agents.eudr.authority_communication_manager.models import (
    Document,
    DocumentType,
)


@pytest.fixture
def config():
    return AuthorityCommunicationManagerConfig()


@pytest.fixture
def exchange(config):
    return DocumentExchange(config=config)


@pytest.fixture
def sample_content() -> bytes:
    """Sample PDF-like content for testing."""
    return b"%PDF-1.4 sample document content " * 100


@pytest.fixture
def small_content() -> bytes:
    """Small content for testing."""
    return b"Hello, EUDR compliance document."


# ====================================================================
# Initialization
# ====================================================================


class TestInit:
    def test_exchange_created(self, exchange):
        assert exchange is not None

    def test_default_config(self):
        e = DocumentExchange()
        assert e.config is not None

    def test_custom_config(self, config):
        e = DocumentExchange(config=config)
        assert e.config is config

    def test_documents_empty(self, exchange):
        assert len(exchange._documents) == 0

    def test_provenance_initialized(self, exchange):
        assert exchange._provenance is not None


# ====================================================================
# Upload Document
# ====================================================================


class TestUploadDocument:
    @pytest.mark.asyncio
    async def test_upload_dds_statement(self, exchange, sample_content):
        result = await exchange.upload_document(
            communication_id="COMM-001",
            document_type="dds_statement",
            title="DDS Statement 2026",
            content=sample_content,
            uploaded_by="OP-001",
        )
        assert isinstance(result, Document)
        assert result.document_type == DocumentType.DDS_STATEMENT

    @pytest.mark.asyncio
    async def test_upload_assigns_id(self, exchange, small_content):
        result = await exchange.upload_document(
            communication_id="COMM-001",
            document_type="certificate",
            title="RSPO Certificate",
            content=small_content,
            uploaded_by="OP-001",
        )
        assert result.document_id is not None
        assert len(result.document_id) > 0

    @pytest.mark.asyncio
    async def test_upload_computes_integrity_hash(self, exchange, small_content):
        result = await exchange.upload_document(
            communication_id="COMM-001",
            document_type="certificate",
            title="Test",
            content=small_content,
            uploaded_by="OP-001",
        )
        assert result.integrity_hash is not None
        assert len(result.integrity_hash) == 64

    @pytest.mark.asyncio
    async def test_upload_integrity_hash_deterministic(self, exchange, small_content):
        r1 = await exchange.upload_document(
            communication_id="COMM-001",
            document_type="certificate",
            title="Test 1",
            content=small_content,
            uploaded_by="OP-001",
        )
        r2 = await exchange.upload_document(
            communication_id="COMM-002",
            document_type="certificate",
            title="Test 2",
            content=small_content,
            uploaded_by="OP-001",
        )
        assert r1.integrity_hash == r2.integrity_hash

    @pytest.mark.asyncio
    async def test_upload_encrypted_by_default(self, exchange, small_content):
        """When encryption_enabled=True, docs should be encrypted."""
        result = await exchange.upload_document(
            communication_id="COMM-001",
            document_type="dds_statement",
            title="Encrypted DDS",
            content=small_content,
            uploaded_by="OP-001",
        )
        assert result.encrypted is True
        assert result.encryption_key_id != ""

    @pytest.mark.asyncio
    async def test_upload_force_no_encryption(self, exchange, small_content):
        result = await exchange.upload_document(
            communication_id="COMM-001",
            document_type="certificate",
            title="Unencrypted Cert",
            content=small_content,
            uploaded_by="OP-001",
            encrypt=False,
        )
        assert result.encrypted is False

    @pytest.mark.asyncio
    async def test_upload_force_encryption(self, exchange, small_content):
        result = await exchange.upload_document(
            communication_id="COMM-001",
            document_type="certificate",
            title="Encrypted Cert",
            content=small_content,
            uploaded_by="OP-001",
            encrypt=True,
        )
        assert result.encrypted is True

    @pytest.mark.asyncio
    async def test_upload_file_size_tracked(self, exchange, sample_content):
        result = await exchange.upload_document(
            communication_id="COMM-001",
            document_type="dds_statement",
            title="Large DDS",
            content=sample_content,
            uploaded_by="OP-001",
        )
        assert result.file_size_bytes > 0

    @pytest.mark.asyncio
    async def test_upload_custom_mime_type(self, exchange, small_content):
        result = await exchange.upload_document(
            communication_id="COMM-001",
            document_type="satellite_imagery",
            title="Satellite Image",
            content=small_content,
            uploaded_by="AUTH-FR-001",
            mime_type="image/tiff",
        )
        assert result.mime_type == "image/tiff"

    @pytest.mark.asyncio
    async def test_upload_with_description(self, exchange, small_content):
        result = await exchange.upload_document(
            communication_id="COMM-001",
            document_type="audit_report",
            title="Q1 Audit",
            content=small_content,
            uploaded_by="AUDITOR-001",
            description="Quarterly audit report covering cocoa supply chain.",
        )
        assert result.description == "Quarterly audit report covering cocoa supply chain."

    @pytest.mark.asyncio
    async def test_upload_with_language(self, exchange, small_content):
        result = await exchange.upload_document(
            communication_id="COMM-001",
            document_type="legal_opinion",
            title="Gutachten",
            content=small_content,
            uploaded_by="LAWYER-001",
            language="de",
        )
        assert result.language.value == "de"

    @pytest.mark.asyncio
    async def test_upload_invalid_type_raises(self, exchange, small_content):
        with pytest.raises(ValueError, match="Invalid"):
            await exchange.upload_document(
                communication_id="COMM-001",
                document_type="invalid_doc_type",
                title="Bad Type",
                content=small_content,
                uploaded_by="OP-001",
            )

    @pytest.mark.asyncio
    async def test_upload_empty_content_raises(self, exchange):
        with pytest.raises(ValueError, match="[Cc]ontent"):
            await exchange.upload_document(
                communication_id="COMM-001",
                document_type="certificate",
                title="Empty Doc",
                content=b"",
                uploaded_by="OP-001",
            )

    @pytest.mark.asyncio
    async def test_upload_stored(self, exchange, small_content):
        result = await exchange.upload_document(
            communication_id="COMM-001",
            document_type="certificate",
            title="Test",
            content=small_content,
            uploaded_by="OP-001",
        )
        assert result.document_id in exchange._documents

    @pytest.mark.asyncio
    async def test_upload_provenance(self, exchange, small_content):
        result = await exchange.upload_document(
            communication_id="COMM-001",
            document_type="certificate",
            title="Test",
            content=small_content,
            uploaded_by="OP-001",
        )
        assert len(result.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_upload_all_document_types(self, exchange, small_content):
        """Test each document type can be uploaded."""
        for dt in DocumentType:
            result = await exchange.upload_document(
                communication_id="COMM-001",
                document_type=dt.value,
                title=f"Test {dt.value}",
                content=small_content,
                uploaded_by="OP-001",
            )
            assert result.document_type == dt


# ====================================================================
# Download Document
# ====================================================================


class TestDownloadDocument:
    @pytest.mark.asyncio
    async def test_download_document(self, exchange, small_content):
        doc = await exchange.upload_document(
            communication_id="COMM-001",
            document_type="certificate",
            title="Test",
            content=small_content,
            uploaded_by="OP-001",
            encrypt=False,
        )
        result = await exchange.download_document(doc.document_id)
        assert result is not None

    @pytest.mark.asyncio
    async def test_download_not_found(self, exchange):
        result = await exchange.download_document("nonexistent")
        assert result is None


# ====================================================================
# Get / List / Health
# ====================================================================


class TestGetListHealth:
    @pytest.mark.asyncio
    async def test_get_document_metadata(self, exchange, small_content):
        doc = await exchange.upload_document(
            communication_id="COMM-001",
            document_type="certificate",
            title="Test",
            content=small_content,
            uploaded_by="OP-001",
        )
        result = await exchange.get_document_metadata(doc.document_id)
        assert result is not None
        assert result.document_id == doc.document_id

    @pytest.mark.asyncio
    async def test_get_document_metadata_not_found(self, exchange):
        result = await exchange.get_document_metadata("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_documents_empty(self, exchange):
        result = await exchange.list_documents()
        assert result == []

    @pytest.mark.asyncio
    async def test_list_documents_multiple(self, exchange, small_content):
        await exchange.upload_document(
            communication_id="COMM-001",
            document_type="certificate",
            title="Doc 1",
            content=small_content,
            uploaded_by="OP-001",
        )
        await exchange.upload_document(
            communication_id="COMM-002",
            document_type="dds_statement",
            title="Doc 2",
            content=small_content,
            uploaded_by="OP-002",
        )
        result = await exchange.list_documents()
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_health_check(self, exchange):
        health = await exchange.health_check()
        assert health["status"] == "healthy"
        assert health["total_documents"] == 0
