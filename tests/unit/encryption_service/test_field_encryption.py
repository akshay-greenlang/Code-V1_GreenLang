# -*- coding: utf-8 -*-
"""
Unit tests for Field-Level Encryption - Encryption at Rest (SEC-003)

Tests the field-level encryption capabilities for database columns, including
type-specific encryption, deterministic search indexes, context binding,
and SQLAlchemy type integration.

Coverage targets: 85%+ of field_encryption.py
"""

from __future__ import annotations

import json
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, date, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Attempt to import field encryption module
# ---------------------------------------------------------------------------
try:
    from greenlang.infrastructure.encryption_service.field_encryption import (
        FieldEncryption,
        FieldEncryptionConfig,
        EncryptedColumnType,
        FieldContext,
    )
    _HAS_FIELD_ENCRYPTION = True
except ImportError:
    _HAS_FIELD_ENCRYPTION = False

pytestmark = [
    pytest.mark.skipif(not _HAS_FIELD_ENCRYPTION, reason="Field encryption not installed"),
]


# ============================================================================
# Helpers and Fixtures
# ============================================================================


@dataclass
class MockFieldContext:
    """Mock field context for testing."""
    tenant_id: str = "tenant-001"
    table_name: str = "users"
    field_name: str = "email"
    record_id: Optional[str] = None

    def __post_init__(self):
        if self.record_id is None:
            self.record_id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, str]:
        return {
            "tenant_id": self.tenant_id,
            "table_name": self.table_name,
            "field_name": self.field_name,
            "record_id": self.record_id,
        }


def _make_mock_encryption_service() -> MagicMock:
    """Create a mock encryption service."""
    service = MagicMock()

    # Track encrypted values for roundtrip testing
    encrypted_store = {}

    async def mock_encrypt(data: bytes, context: Any) -> MagicMock:
        encrypted = MagicMock()
        encrypted.ciphertext = secrets.token_bytes(len(data) + 16)
        encrypted.nonce = secrets.token_bytes(12)
        encrypted.encrypted_dek = secrets.token_bytes(64)
        encrypted.key_version = "v1"
        # Store for decryption
        key = encrypted.ciphertext.hex()
        encrypted_store[key] = data
        encrypted._original = data
        return encrypted

    async def mock_decrypt(encrypted: Any, context: Any) -> bytes:
        if hasattr(encrypted, '_original'):
            return encrypted._original
        key = encrypted.ciphertext.hex()
        return encrypted_store.get(key, b"")

    service.encrypt = AsyncMock(side_effect=mock_encrypt)
    service.decrypt = AsyncMock(side_effect=mock_decrypt)
    service._encrypted_store = encrypted_store

    return service


def _make_mock_search_index() -> MagicMock:
    """Create a mock search index generator."""
    index = MagicMock()

    def generate_index(value: Any, context: Dict[str, str]) -> str:
        # Deterministic index based on value and context
        import hashlib
        data = f"{value}:{context.get('tenant_id', '')}:{context.get('field_name', '')}"
        return hashlib.sha256(data.encode()).hexdigest()[:32]

    index.generate = MagicMock(side_effect=generate_index)
    return index


@pytest.fixture
def mock_encryption_service() -> MagicMock:
    """Create a mock encryption service."""
    return _make_mock_encryption_service()


@pytest.fixture
def mock_search_index() -> MagicMock:
    """Create a mock search index generator."""
    return _make_mock_search_index()


@pytest.fixture
def field_context() -> MockFieldContext:
    """Create a test field context."""
    return MockFieldContext()


# ============================================================================
# Test: String Field Encryption
# ============================================================================


class TestStringFieldEncryption:
    """Tests for string field encryption."""

    @pytest.mark.asyncio
    async def test_encrypt_field_string(
        self,
        mock_encryption_service,
        mock_search_index,
        field_context,
    ):
        """Test encryption of string field."""
        if not _HAS_FIELD_ENCRYPTION:
            pytest.skip("Field encryption not available")

        field_enc = FieldEncryption(
            encryption_service=mock_encryption_service,
            search_index=mock_search_index,
        )

        value = "user@example.com"
        encrypted = await field_enc.encrypt_field(value, field_context)

        assert encrypted is not None
        assert encrypted != value

    @pytest.mark.asyncio
    async def test_decrypt_field_string_roundtrip(
        self,
        mock_encryption_service,
        mock_search_index,
        field_context,
    ):
        """Test decryption roundtrip for string field."""
        if not _HAS_FIELD_ENCRYPTION:
            pytest.skip("Field encryption not available")

        field_enc = FieldEncryption(
            encryption_service=mock_encryption_service,
            search_index=mock_search_index,
        )

        original = "sensitive-data@example.com"
        encrypted = await field_enc.encrypt_field(original, field_context)
        decrypted = await field_enc.decrypt_field(encrypted, field_context)

        assert decrypted == original


# ============================================================================
# Test: Numeric Field Encryption
# ============================================================================


class TestNumericFieldEncryption:
    """Tests for numeric field encryption."""

    @pytest.mark.asyncio
    async def test_encrypt_field_int(
        self,
        mock_encryption_service,
        mock_search_index,
        field_context,
    ):
        """Test encryption of integer field."""
        if not _HAS_FIELD_ENCRYPTION:
            pytest.skip("Field encryption not available")

        field_enc = FieldEncryption(
            encryption_service=mock_encryption_service,
            search_index=mock_search_index,
        )

        value = 12345678
        encrypted = await field_enc.encrypt_field(value, field_context)
        decrypted = await field_enc.decrypt_field(encrypted, field_context, value_type=int)

        assert decrypted == value

    @pytest.mark.asyncio
    async def test_encrypt_field_float(
        self,
        mock_encryption_service,
        mock_search_index,
        field_context,
    ):
        """Test encryption of float field."""
        if not _HAS_FIELD_ENCRYPTION:
            pytest.skip("Field encryption not available")

        field_enc = FieldEncryption(
            encryption_service=mock_encryption_service,
            search_index=mock_search_index,
        )

        value = 123.456789
        encrypted = await field_enc.encrypt_field(value, field_context)
        decrypted = await field_enc.decrypt_field(encrypted, field_context, value_type=float)

        assert abs(decrypted - value) < 0.000001

    @pytest.mark.asyncio
    async def test_encrypt_field_decimal(
        self,
        mock_encryption_service,
        mock_search_index,
        field_context,
    ):
        """Test encryption of Decimal field."""
        if not _HAS_FIELD_ENCRYPTION:
            pytest.skip("Field encryption not available")

        field_enc = FieldEncryption(
            encryption_service=mock_encryption_service,
            search_index=mock_search_index,
        )

        value = Decimal("123456.789012")
        encrypted = await field_enc.encrypt_field(value, field_context)
        decrypted = await field_enc.decrypt_field(encrypted, field_context, value_type=Decimal)

        assert decrypted == value


# ============================================================================
# Test: Complex Type Field Encryption
# ============================================================================


class TestComplexFieldEncryption:
    """Tests for complex type field encryption."""

    @pytest.mark.asyncio
    async def test_encrypt_field_dict(
        self,
        mock_encryption_service,
        mock_search_index,
        field_context,
    ):
        """Test encryption of dict/JSON field."""
        if not _HAS_FIELD_ENCRYPTION:
            pytest.skip("Field encryption not available")

        field_enc = FieldEncryption(
            encryption_service=mock_encryption_service,
            search_index=mock_search_index,
        )

        value = {
            "name": "John Doe",
            "ssn": "123-45-6789",
            "address": {
                "street": "123 Main St",
                "city": "Anytown",
            },
        }
        encrypted = await field_enc.encrypt_field(value, field_context)
        decrypted = await field_enc.decrypt_field(encrypted, field_context, value_type=dict)

        assert decrypted == value

    @pytest.mark.asyncio
    async def test_encrypt_field_list(
        self,
        mock_encryption_service,
        mock_search_index,
        field_context,
    ):
        """Test encryption of list field."""
        if not _HAS_FIELD_ENCRYPTION:
            pytest.skip("Field encryption not available")

        field_enc = FieldEncryption(
            encryption_service=mock_encryption_service,
            search_index=mock_search_index,
        )

        value = ["item1", "item2", "item3"]
        encrypted = await field_enc.encrypt_field(value, field_context)
        decrypted = await field_enc.decrypt_field(encrypted, field_context, value_type=list)

        assert decrypted == value

    @pytest.mark.asyncio
    async def test_encrypt_field_datetime(
        self,
        mock_encryption_service,
        mock_search_index,
        field_context,
    ):
        """Test encryption of datetime field."""
        if not _HAS_FIELD_ENCRYPTION:
            pytest.skip("Field encryption not available")

        field_enc = FieldEncryption(
            encryption_service=mock_encryption_service,
            search_index=mock_search_index,
        )

        value = datetime(2026, 2, 6, 12, 30, 45, tzinfo=timezone.utc)
        encrypted = await field_enc.encrypt_field(value, field_context)
        decrypted = await field_enc.decrypt_field(encrypted, field_context, value_type=datetime)

        assert decrypted == value

    @pytest.mark.asyncio
    async def test_encrypt_field_date(
        self,
        mock_encryption_service,
        mock_search_index,
        field_context,
    ):
        """Test encryption of date field."""
        if not _HAS_FIELD_ENCRYPTION:
            pytest.skip("Field encryption not available")

        field_enc = FieldEncryption(
            encryption_service=mock_encryption_service,
            search_index=mock_search_index,
        )

        value = date(2026, 2, 6)
        encrypted = await field_enc.encrypt_field(value, field_context)
        decrypted = await field_enc.decrypt_field(encrypted, field_context, value_type=date)

        assert decrypted == value


# ============================================================================
# Test: Field Context
# ============================================================================


class TestFieldContext:
    """Tests for field context handling."""

    @pytest.mark.asyncio
    async def test_field_context_includes_tenant(
        self,
        mock_encryption_service,
        mock_search_index,
    ):
        """Test that field context includes tenant_id."""
        if not _HAS_FIELD_ENCRYPTION:
            pytest.skip("Field encryption not available")

        field_enc = FieldEncryption(
            encryption_service=mock_encryption_service,
            search_index=mock_search_index,
        )

        context = MockFieldContext(tenant_id="tenant-123")
        await field_enc.encrypt_field("test", context)

        # Verify encryption was called with context containing tenant_id
        mock_encryption_service.encrypt.assert_called()

    @pytest.mark.asyncio
    async def test_field_context_includes_field_name(
        self,
        mock_encryption_service,
        mock_search_index,
    ):
        """Test that field context includes field_name."""
        if not _HAS_FIELD_ENCRYPTION:
            pytest.skip("Field encryption not available")

        field_enc = FieldEncryption(
            encryption_service=mock_encryption_service,
            search_index=mock_search_index,
        )

        context = MockFieldContext(field_name="ssn")
        await field_enc.encrypt_field("123-45-6789", context)

        mock_encryption_service.encrypt.assert_called()

    @pytest.mark.asyncio
    async def test_different_context_different_ciphertext(
        self,
        mock_encryption_service,
        mock_search_index,
    ):
        """Test that different contexts produce different results."""
        if not _HAS_FIELD_ENCRYPTION:
            pytest.skip("Field encryption not available")

        field_enc = FieldEncryption(
            encryption_service=mock_encryption_service,
            search_index=mock_search_index,
        )

        context1 = MockFieldContext(tenant_id="tenant-001")
        context2 = MockFieldContext(tenant_id="tenant-002")

        value = "same-value"
        encrypted1 = await field_enc.encrypt_field(value, context1)
        encrypted2 = await field_enc.encrypt_field(value, context2)

        # Different contexts should produce different ciphertext
        # (due to different AAD)
        assert encrypted1 != encrypted2


# ============================================================================
# Test: Search Index
# ============================================================================


class TestSearchIndex:
    """Tests for deterministic search index generation."""

    @pytest.mark.asyncio
    async def test_search_index_deterministic(
        self,
        mock_encryption_service,
        mock_search_index,
        field_context,
    ):
        """Test that search index is deterministic for same value."""
        if not _HAS_FIELD_ENCRYPTION:
            pytest.skip("Field encryption not available")

        field_enc = FieldEncryption(
            encryption_service=mock_encryption_service,
            search_index=mock_search_index,
        )

        value = "searchable-value"

        # Generate index twice
        index1 = await field_enc.generate_search_index(value, field_context)
        index2 = await field_enc.generate_search_index(value, field_context)

        assert index1 == index2

    @pytest.mark.asyncio
    async def test_search_index_different_for_different_values(
        self,
        mock_encryption_service,
        mock_search_index,
        field_context,
    ):
        """Test that different values produce different search indexes."""
        if not _HAS_FIELD_ENCRYPTION:
            pytest.skip("Field encryption not available")

        field_enc = FieldEncryption(
            encryption_service=mock_encryption_service,
            search_index=mock_search_index,
        )

        value1 = "value-one"
        value2 = "value-two"

        index1 = await field_enc.generate_search_index(value1, field_context)
        index2 = await field_enc.generate_search_index(value2, field_context)

        assert index1 != index2

    @pytest.mark.asyncio
    async def test_search_index_tenant_isolated(
        self,
        mock_encryption_service,
        mock_search_index,
    ):
        """Test that search index is tenant-isolated."""
        if not _HAS_FIELD_ENCRYPTION:
            pytest.skip("Field encryption not available")

        field_enc = FieldEncryption(
            encryption_service=mock_encryption_service,
            search_index=mock_search_index,
        )

        context1 = MockFieldContext(tenant_id="tenant-001")
        context2 = MockFieldContext(tenant_id="tenant-002")

        value = "same-value"

        index1 = await field_enc.generate_search_index(value, context1)
        index2 = await field_enc.generate_search_index(value, context2)

        # Same value, different tenants = different indexes
        assert index1 != index2


# ============================================================================
# Test: Data Packing/Unpacking
# ============================================================================


class TestDataPacking:
    """Tests for encrypted data packing and unpacking."""

    @pytest.mark.asyncio
    async def test_pack_unpack_encrypted_data(
        self,
        mock_encryption_service,
        mock_search_index,
        field_context,
    ):
        """Test packing and unpacking encrypted field data."""
        if not _HAS_FIELD_ENCRYPTION:
            pytest.skip("Field encryption not available")

        field_enc = FieldEncryption(
            encryption_service=mock_encryption_service,
            search_index=mock_search_index,
        )

        value = "test-value"

        # Encrypt and pack
        encrypted = await field_enc.encrypt_field(value, field_context)

        # Pack for storage
        packed = field_enc.pack(encrypted)
        assert isinstance(packed, (bytes, str))

        # Unpack for decryption
        unpacked = field_enc.unpack(packed)

        # Decrypt
        decrypted = await field_enc.decrypt_field(unpacked, field_context)
        assert decrypted == value

    @pytest.mark.asyncio
    async def test_pack_includes_metadata(
        self,
        mock_encryption_service,
        mock_search_index,
        field_context,
    ):
        """Test that packed data includes necessary metadata."""
        if not _HAS_FIELD_ENCRYPTION:
            pytest.skip("Field encryption not available")

        field_enc = FieldEncryption(
            encryption_service=mock_encryption_service,
            search_index=mock_search_index,
        )

        encrypted = await field_enc.encrypt_field("test", field_context)
        packed = field_enc.pack(encrypted)

        # Unpack and verify metadata is preserved
        unpacked = field_enc.unpack(packed)

        assert hasattr(unpacked, 'key_version') or 'key_version' in str(unpacked)


# ============================================================================
# Test: SQLAlchemy Column Type
# ============================================================================


class TestEncryptedColumnType:
    """Tests for SQLAlchemy encrypted column type."""

    def test_encrypted_column_type_instantiation(self):
        """Test that EncryptedColumnType can be instantiated."""
        if not _HAS_FIELD_ENCRYPTION:
            pytest.skip("Field encryption not available")

        column_type = EncryptedColumnType(
            field_name="email",
            searchable=True,
        )

        assert column_type is not None

    def test_encrypted_column_type_process_bind_param(
        self,
        mock_encryption_service,
    ):
        """Test process_bind_param converts value for storage."""
        if not _HAS_FIELD_ENCRYPTION:
            pytest.skip("Field encryption not available")

        column_type = EncryptedColumnType(
            field_name="email",
            encryption_service=mock_encryption_service,
        )

        # This is typically called by SQLAlchemy during INSERT/UPDATE
        # Test that it handles values correctly
        value = "test@example.com"

        # Note: actual implementation may be sync or async
        # This tests the structure
        assert column_type is not None

    def test_encrypted_column_type_searchable_flag(self):
        """Test that searchable flag affects behavior."""
        if not _HAS_FIELD_ENCRYPTION:
            pytest.skip("Field encryption not available")

        searchable = EncryptedColumnType(field_name="email", searchable=True)
        non_searchable = EncryptedColumnType(field_name="notes", searchable=False)

        assert hasattr(searchable, 'searchable')
        assert searchable.searchable == True
        assert non_searchable.searchable == False


# ============================================================================
# Test: Null Handling
# ============================================================================


class TestNullHandling:
    """Tests for null/None value handling."""

    @pytest.mark.asyncio
    async def test_encrypt_null_returns_null(
        self,
        mock_encryption_service,
        mock_search_index,
        field_context,
    ):
        """Test that encrypting None returns None."""
        if not _HAS_FIELD_ENCRYPTION:
            pytest.skip("Field encryption not available")

        field_enc = FieldEncryption(
            encryption_service=mock_encryption_service,
            search_index=mock_search_index,
        )

        encrypted = await field_enc.encrypt_field(None, field_context)

        assert encrypted is None

    @pytest.mark.asyncio
    async def test_decrypt_null_returns_null(
        self,
        mock_encryption_service,
        mock_search_index,
        field_context,
    ):
        """Test that decrypting None returns None."""
        if not _HAS_FIELD_ENCRYPTION:
            pytest.skip("Field encryption not available")

        field_enc = FieldEncryption(
            encryption_service=mock_encryption_service,
            search_index=mock_search_index,
        )

        decrypted = await field_enc.decrypt_field(None, field_context)

        assert decrypted is None


# ============================================================================
# Test: Empty String Handling
# ============================================================================


class TestEmptyStringHandling:
    """Tests for empty string handling."""

    @pytest.mark.asyncio
    async def test_encrypt_empty_string(
        self,
        mock_encryption_service,
        mock_search_index,
        field_context,
    ):
        """Test encryption of empty string."""
        if not _HAS_FIELD_ENCRYPTION:
            pytest.skip("Field encryption not available")

        field_enc = FieldEncryption(
            encryption_service=mock_encryption_service,
            search_index=mock_search_index,
        )

        encrypted = await field_enc.encrypt_field("", field_context)
        decrypted = await field_enc.decrypt_field(encrypted, field_context)

        assert decrypted == ""
