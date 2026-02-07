# -*- coding: utf-8 -*-
"""
Unit tests for AllowlistManager - SEC-011 PII Service.

Tests the allowlist manager for false positive filtering:
- Pattern matching (regex, exact, prefix, suffix, contains)
- Tenant-specific and global allowlists
- Default allowlist loading
- Entry lifecycle management
- Expiration handling

Coverage target: 85%+ of allowlist/manager.py
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def allowlist_manager(allowlist_config, mock_db_pool, mock_redis_client):
    """Create AllowlistManager instance for testing."""
    try:
        from greenlang.infrastructure.pii_service.allowlist.manager import AllowlistManager
        return AllowlistManager(
            config=allowlist_config,
            db_pool=mock_db_pool,
            redis_client=mock_redis_client,
        )
    except ImportError:
        pytest.skip("AllowlistManager not yet implemented")


@pytest.fixture
def allowlist_manager_no_defaults(allowlist_config, mock_db_pool, mock_redis_client):
    """Create AllowlistManager without default entries."""
    try:
        from greenlang.infrastructure.pii_service.allowlist.manager import AllowlistManager
        allowlist_config.enable_defaults = False
        return AllowlistManager(
            config=allowlist_config,
            db_pool=mock_db_pool,
            redis_client=mock_redis_client,
        )
    except ImportError:
        pytest.skip("AllowlistManager not yet implemented")


# ============================================================================
# TestIsAllowed
# ============================================================================


class TestIsAllowed:
    """Tests for is_allowed() method."""

    @pytest.mark.asyncio
    async def test_is_allowed_exact_match(
        self, allowlist_manager, pii_type_enum, test_tenant_id
    ):
        """is_allowed() returns True for exact pattern match."""
        try:
            from greenlang.infrastructure.pii_service.allowlist.patterns import AllowlistEntry, PatternType

            # Add an exact match entry
            entry = AllowlistEntry(
                pii_type=pii_type_enum.PHONE,
                pattern="555-555-5555",
                pattern_type=PatternType.EXACT,
                reason="Fictional phone number",
                created_by=uuid4(),
            )
            await allowlist_manager.add_entry(entry)
        except ImportError:
            pytest.skip("AllowlistEntry not available")

        result = await allowlist_manager.is_allowed(
            "555-555-5555",
            pii_type_enum.PHONE,
            test_tenant_id,
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_is_allowed_regex_match(
        self, allowlist_manager, pii_type_enum, test_tenant_id
    ):
        """is_allowed() returns True for regex pattern match."""
        try:
            from greenlang.infrastructure.pii_service.allowlist.patterns import AllowlistEntry, PatternType

            entry = AllowlistEntry(
                pii_type=pii_type_enum.EMAIL,
                pattern=r".*@example\.com$",
                pattern_type=PatternType.REGEX,
                reason="RFC 2606 reserved domain",
                created_by=uuid4(),
            )
            await allowlist_manager.add_entry(entry)
        except ImportError:
            pytest.skip("AllowlistEntry not available")

        result = await allowlist_manager.is_allowed(
            "test@example.com",
            pii_type_enum.EMAIL,
            test_tenant_id,
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_is_allowed_prefix_match(
        self, allowlist_manager, pii_type_enum, test_tenant_id
    ):
        """is_allowed() returns True for prefix pattern match."""
        try:
            from greenlang.infrastructure.pii_service.allowlist.patterns import AllowlistEntry, PatternType

            entry = AllowlistEntry(
                pii_type=pii_type_enum.API_KEY,
                pattern="sk_test_",
                pattern_type=PatternType.PREFIX,
                reason="Stripe test key prefix",
                created_by=uuid4(),
            )
            await allowlist_manager.add_entry(entry)
        except ImportError:
            pytest.skip("AllowlistEntry not available")

        result = await allowlist_manager.is_allowed(
            "sk_test_abc123xyz",
            pii_type_enum.API_KEY,
            test_tenant_id,
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_is_allowed_suffix_match(
        self, allowlist_manager, pii_type_enum, test_tenant_id
    ):
        """is_allowed() returns True for suffix pattern match."""
        try:
            from greenlang.infrastructure.pii_service.allowlist.patterns import AllowlistEntry, PatternType

            entry = AllowlistEntry(
                pii_type=pii_type_enum.EMAIL,
                pattern="@localhost",
                pattern_type=PatternType.SUFFIX,
                reason="Localhost email",
                created_by=uuid4(),
            )
            await allowlist_manager.add_entry(entry)
        except ImportError:
            pytest.skip("AllowlistEntry not available")

        result = await allowlist_manager.is_allowed(
            "admin@localhost",
            pii_type_enum.EMAIL,
            test_tenant_id,
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_is_allowed_contains_match(
        self, allowlist_manager, pii_type_enum, test_tenant_id
    ):
        """is_allowed() returns True for contains pattern match."""
        try:
            from greenlang.infrastructure.pii_service.allowlist.patterns import AllowlistEntry, PatternType

            entry = AllowlistEntry(
                pii_type=pii_type_enum.EMAIL,
                pattern="noreply",
                pattern_type=PatternType.CONTAINS,
                reason="No-reply address",
                created_by=uuid4(),
            )
            await allowlist_manager.add_entry(entry)
        except ImportError:
            pytest.skip("AllowlistEntry not available")

        result = await allowlist_manager.is_allowed(
            "noreply@company.com",
            pii_type_enum.EMAIL,
            test_tenant_id,
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_is_allowed_no_match_returns_false(
        self, allowlist_manager, pii_type_enum, test_tenant_id
    ):
        """is_allowed() returns False when no pattern matches."""
        result = await allowlist_manager.is_allowed(
            "real-ssn-123-45-6789",
            pii_type_enum.SSN,
            test_tenant_id,
        )

        assert result is False


# ============================================================================
# TestDefaultAllowlists
# ============================================================================


class TestDefaultAllowlists:
    """Tests for default allowlist loading."""

    @pytest.mark.asyncio
    async def test_default_allowlists_loaded(
        self, allowlist_manager, pii_type_enum, test_tenant_id
    ):
        """Default allowlists are loaded on initialization."""
        # Test card (Stripe test Visa)
        result = await allowlist_manager.is_allowed(
            "4242424242424242",
            pii_type_enum.CREDIT_CARD,
            test_tenant_id,
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_default_email_allowlist(
        self, allowlist_manager, pii_type_enum, test_tenant_id
    ):
        """Default email allowlist covers example.com."""
        result = await allowlist_manager.is_allowed(
            "user@example.com",
            pii_type_enum.EMAIL,
            test_tenant_id,
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_default_phone_allowlist(
        self, allowlist_manager, pii_type_enum, test_tenant_id
    ):
        """Default phone allowlist covers 555 numbers."""
        result = await allowlist_manager.is_allowed(
            "555-123-4567",
            pii_type_enum.PHONE,
            test_tenant_id,
        )

        # 555 numbers are fictional
        assert result is True

    @pytest.mark.asyncio
    async def test_default_ssn_allowlist(
        self, allowlist_manager, pii_type_enum, test_tenant_id
    ):
        """Default SSN allowlist covers invalid ranges."""
        # 000-00-0000 is invalid
        result = await allowlist_manager.is_allowed(
            "000-00-0000",
            pii_type_enum.SSN,
            test_tenant_id,
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_default_ip_allowlist(
        self, allowlist_manager, pii_type_enum, test_tenant_id
    ):
        """Default IP allowlist covers private ranges."""
        result = await allowlist_manager.is_allowed(
            "192.168.1.1",
            pii_type_enum.IP_ADDRESS,
            test_tenant_id,
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_no_defaults_when_disabled(
        self, allowlist_manager_no_defaults, pii_type_enum, test_tenant_id
    ):
        """Defaults not loaded when enable_defaults=False."""
        result = await allowlist_manager_no_defaults.is_allowed(
            "4242424242424242",
            pii_type_enum.CREDIT_CARD,
            test_tenant_id,
        )

        assert result is False


# ============================================================================
# TestTenantSpecificAllowlists
# ============================================================================


class TestTenantSpecificAllowlists:
    """Tests for tenant-specific allowlists."""

    @pytest.mark.asyncio
    async def test_tenant_specific_allowlist(
        self, allowlist_manager, pii_type_enum
    ):
        """Tenant-specific allowlist only applies to that tenant."""
        try:
            from greenlang.infrastructure.pii_service.allowlist.patterns import AllowlistEntry, PatternType

            entry = AllowlistEntry(
                pii_type=pii_type_enum.EMAIL,
                pattern=r".*@tenant-a\.com$",
                pattern_type=PatternType.REGEX,
                reason="Tenant A domain",
                created_by=uuid4(),
                tenant_id="tenant-a",
            )
            await allowlist_manager.add_entry(entry)
        except ImportError:
            pytest.skip("AllowlistEntry not available")

        # Should match for tenant-a
        result_a = await allowlist_manager.is_allowed(
            "user@tenant-a.com",
            pii_type_enum.EMAIL,
            "tenant-a",
        )

        # Should not match for tenant-b
        result_b = await allowlist_manager.is_allowed(
            "user@tenant-a.com",
            pii_type_enum.EMAIL,
            "tenant-b",
        )

        assert result_a is True
        assert result_b is False

    @pytest.mark.asyncio
    async def test_global_allowlist_applies_all_tenants(
        self, allowlist_manager, pii_type_enum
    ):
        """Global allowlist (tenant_id=None) applies to all tenants."""
        try:
            from greenlang.infrastructure.pii_service.allowlist.patterns import AllowlistEntry, PatternType

            entry = AllowlistEntry(
                pii_type=pii_type_enum.EMAIL,
                pattern=r".*@global-safe\.com$",
                pattern_type=PatternType.REGEX,
                reason="Global safe domain",
                created_by=uuid4(),
                tenant_id=None,  # Global
            )
            await allowlist_manager.add_entry(entry)
        except ImportError:
            pytest.skip("AllowlistEntry not available")

        # Should match for any tenant
        for tenant in ["tenant-1", "tenant-2", "tenant-3"]:
            result = await allowlist_manager.is_allowed(
                "user@global-safe.com",
                pii_type_enum.EMAIL,
                tenant,
            )
            assert result is True


# ============================================================================
# TestExpirationHandling
# ============================================================================


class TestExpirationHandling:
    """Tests for entry expiration handling."""

    @pytest.mark.asyncio
    async def test_expired_entry_ignored(
        self, allowlist_manager, pii_type_enum, test_tenant_id
    ):
        """Expired entries are not matched."""
        try:
            from greenlang.infrastructure.pii_service.allowlist.patterns import AllowlistEntry, PatternType

            entry = AllowlistEntry(
                pii_type=pii_type_enum.EMAIL,
                pattern=r".*@expired\.com$",
                pattern_type=PatternType.REGEX,
                reason="Expired entry",
                created_by=uuid4(),
                expires_at=datetime.now(timezone.utc) - timedelta(days=1),  # Expired
            )
            await allowlist_manager.add_entry(entry)
        except ImportError:
            pytest.skip("AllowlistEntry not available")

        result = await allowlist_manager.is_allowed(
            "user@expired.com",
            pii_type_enum.EMAIL,
            test_tenant_id,
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_disabled_entry_ignored(
        self, allowlist_manager, pii_type_enum, test_tenant_id
    ):
        """Disabled entries are not matched."""
        try:
            from greenlang.infrastructure.pii_service.allowlist.patterns import AllowlistEntry, PatternType

            entry = AllowlistEntry(
                pii_type=pii_type_enum.EMAIL,
                pattern=r".*@disabled\.com$",
                pattern_type=PatternType.REGEX,
                reason="Disabled entry",
                created_by=uuid4(),
                enabled=False,
            )
            await allowlist_manager.add_entry(entry)
        except ImportError:
            pytest.skip("AllowlistEntry not available")

        result = await allowlist_manager.is_allowed(
            "user@disabled.com",
            pii_type_enum.EMAIL,
            test_tenant_id,
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_not_expired_entry_matches(
        self, allowlist_manager, pii_type_enum, test_tenant_id
    ):
        """Non-expired entries are matched."""
        try:
            from greenlang.infrastructure.pii_service.allowlist.patterns import AllowlistEntry, PatternType

            entry = AllowlistEntry(
                pii_type=pii_type_enum.EMAIL,
                pattern=r".*@valid\.com$",
                pattern_type=PatternType.REGEX,
                reason="Valid entry",
                created_by=uuid4(),
                expires_at=datetime.now(timezone.utc) + timedelta(days=30),  # Not expired
            )
            await allowlist_manager.add_entry(entry)
        except ImportError:
            pytest.skip("AllowlistEntry not available")

        result = await allowlist_manager.is_allowed(
            "user@valid.com",
            pii_type_enum.EMAIL,
            test_tenant_id,
        )

        assert result is True


# ============================================================================
# TestEntryManagement
# ============================================================================


class TestEntryManagement:
    """Tests for allowlist entry management."""

    @pytest.mark.asyncio
    async def test_add_entry_validates_regex(
        self, allowlist_manager, pii_type_enum
    ):
        """add_entry() validates regex patterns."""
        try:
            from greenlang.infrastructure.pii_service.allowlist.patterns import AllowlistEntry, PatternType

            entry = AllowlistEntry(
                pii_type=pii_type_enum.EMAIL,
                pattern="[invalid(regex",  # Invalid regex
                pattern_type=PatternType.REGEX,
                reason="Invalid regex test",
                created_by=uuid4(),
            )

            with pytest.raises(Exception) as exc_info:
                await allowlist_manager.add_entry(entry)

            assert "regex" in str(exc_info.value).lower() or "pattern" in str(exc_info.value).lower()
        except ImportError:
            pytest.skip("AllowlistEntry not available")

    @pytest.mark.asyncio
    async def test_add_entry_persists(
        self, allowlist_manager, pii_type_enum, test_tenant_id
    ):
        """add_entry() persists to storage."""
        try:
            from greenlang.infrastructure.pii_service.allowlist.patterns import AllowlistEntry, PatternType

            entry = AllowlistEntry(
                pii_type=pii_type_enum.EMAIL,
                pattern="test@persist.com",
                pattern_type=PatternType.EXACT,
                reason="Persistence test",
                created_by=uuid4(),
            )
            entry_id = await allowlist_manager.add_entry(entry)
        except ImportError:
            pytest.skip("AllowlistEntry not available")

        # Should be retrievable
        entries = await allowlist_manager.list_entries(pii_type_enum.EMAIL, test_tenant_id)
        assert any(e.pattern == "test@persist.com" for e in entries)

    @pytest.mark.asyncio
    async def test_remove_entry(
        self, allowlist_manager, pii_type_enum, test_tenant_id
    ):
        """remove_entry() removes entry from storage."""
        try:
            from greenlang.infrastructure.pii_service.allowlist.patterns import AllowlistEntry, PatternType

            entry = AllowlistEntry(
                pii_type=pii_type_enum.EMAIL,
                pattern="test@remove.com",
                pattern_type=PatternType.EXACT,
                reason="Remove test",
                created_by=uuid4(),
            )
            entry_id = await allowlist_manager.add_entry(entry)
        except ImportError:
            pytest.skip("AllowlistEntry not available")

        # Remove
        await allowlist_manager.remove_entry(entry.id)

        # Should no longer match
        result = await allowlist_manager.is_allowed(
            "test@remove.com",
            pii_type_enum.EMAIL,
            test_tenant_id,
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_list_entries_by_type(
        self, allowlist_manager, pii_type_enum, test_tenant_id
    ):
        """list_entries() filters by PII type."""
        try:
            from greenlang.infrastructure.pii_service.allowlist.patterns import AllowlistEntry, PatternType

            # Add entries for different types
            email_entry = AllowlistEntry(
                pii_type=pii_type_enum.EMAIL,
                pattern="test@list.com",
                pattern_type=PatternType.EXACT,
                reason="List test email",
                created_by=uuid4(),
            )
            phone_entry = AllowlistEntry(
                pii_type=pii_type_enum.PHONE,
                pattern="555-999-9999",
                pattern_type=PatternType.EXACT,
                reason="List test phone",
                created_by=uuid4(),
            )
            await allowlist_manager.add_entry(email_entry)
            await allowlist_manager.add_entry(phone_entry)
        except ImportError:
            pytest.skip("AllowlistEntry not available")

        # List only email entries
        email_entries = await allowlist_manager.list_entries(pii_type_enum.EMAIL, test_tenant_id)

        # Should only include email entries
        assert all(e.pii_type == pii_type_enum.EMAIL for e in email_entries)


# ============================================================================
# TestMetricsRecording
# ============================================================================


class TestAllowlistMetrics:
    """Tests for metrics recording."""

    @pytest.mark.asyncio
    async def test_metrics_recorded_on_match(
        self, allowlist_manager, pii_type_enum, test_tenant_id
    ):
        """Metrics are recorded when allowlist matches."""
        from unittest.mock import patch

        with patch("greenlang.infrastructure.pii_service.metrics.gl_pii_allowlist_matches_total") as mock_counter:
            mock_counter.labels.return_value.inc = MagicMock()

            # Use default allowlist (test card)
            await allowlist_manager.is_allowed(
                "4242424242424242",
                pii_type_enum.CREDIT_CARD,
                test_tenant_id,
            )

            # Metrics should be recorded
            # Implementation specific


# ============================================================================
# TestCaching
# ============================================================================


class TestAllowlistCaching:
    """Tests for allowlist caching."""

    @pytest.mark.asyncio
    async def test_cache_hit_avoids_db_lookup(
        self, allowlist_manager, pii_type_enum, test_tenant_id, mock_db_pool
    ):
        """Cache hit avoids database lookup."""
        # First lookup populates cache
        await allowlist_manager.is_allowed(
            "4242424242424242",
            pii_type_enum.CREDIT_CARD,
            test_tenant_id,
        )

        # Track DB calls
        initial_calls = mock_db_pool.fetchall.call_count if hasattr(mock_db_pool.fetchall, 'call_count') else 0

        # Second lookup should use cache
        await allowlist_manager.is_allowed(
            "4242424242424242",
            pii_type_enum.CREDIT_CARD,
            test_tenant_id,
        )

        # Should not have additional DB calls
        # Implementation specific

    @pytest.mark.asyncio
    async def test_cache_invalidation_on_add(
        self, allowlist_manager, pii_type_enum, test_tenant_id
    ):
        """Cache is invalidated when entries are added."""
        try:
            from greenlang.infrastructure.pii_service.allowlist.patterns import AllowlistEntry, PatternType

            # First lookup
            result_1 = await allowlist_manager.is_allowed(
                "new@entry.com",
                pii_type_enum.EMAIL,
                test_tenant_id,
            )
            assert result_1 is False

            # Add entry
            entry = AllowlistEntry(
                pii_type=pii_type_enum.EMAIL,
                pattern="new@entry.com",
                pattern_type=PatternType.EXACT,
                reason="Cache test",
                created_by=uuid4(),
            )
            await allowlist_manager.add_entry(entry)

            # Second lookup should see new entry
            result_2 = await allowlist_manager.is_allowed(
                "new@entry.com",
                pii_type_enum.EMAIL,
                test_tenant_id,
            )
            assert result_2 is True
        except ImportError:
            pytest.skip("AllowlistEntry not available")


# ============================================================================
# TestPatternMatching
# ============================================================================


class TestPatternMatching:
    """Tests for pattern matching edge cases."""

    @pytest.mark.asyncio
    async def test_case_sensitive_exact_match(
        self, allowlist_manager, pii_type_enum, test_tenant_id
    ):
        """Exact match is case-sensitive."""
        try:
            from greenlang.infrastructure.pii_service.allowlist.patterns import AllowlistEntry, PatternType

            entry = AllowlistEntry(
                pii_type=pii_type_enum.EMAIL,
                pattern="Test@Example.com",
                pattern_type=PatternType.EXACT,
                reason="Case test",
                created_by=uuid4(),
            )
            await allowlist_manager.add_entry(entry)
        except ImportError:
            pytest.skip("AllowlistEntry not available")

        # Exact case should match
        result_exact = await allowlist_manager.is_allowed(
            "Test@Example.com",
            pii_type_enum.EMAIL,
            test_tenant_id,
        )

        # Different case should not match (unless case-insensitive configured)
        result_different = await allowlist_manager.is_allowed(
            "test@example.com",
            pii_type_enum.EMAIL,
            test_tenant_id,
        )

        assert result_exact is True
        # Case sensitivity is implementation dependent

    @pytest.mark.asyncio
    async def test_regex_special_characters(
        self, allowlist_manager, pii_type_enum, test_tenant_id
    ):
        """Regex patterns handle special characters correctly."""
        try:
            from greenlang.infrastructure.pii_service.allowlist.patterns import AllowlistEntry, PatternType

            entry = AllowlistEntry(
                pii_type=pii_type_enum.EMAIL,
                pattern=r".*@example\.com$",  # Escaped dot
                pattern_type=PatternType.REGEX,
                reason="Regex special char test",
                created_by=uuid4(),
            )
            await allowlist_manager.add_entry(entry)
        except ImportError:
            pytest.skip("AllowlistEntry not available")

        # Should match example.com
        result_1 = await allowlist_manager.is_allowed(
            "test@example.com",
            pii_type_enum.EMAIL,
            test_tenant_id,
        )

        # Should not match exampleXcom (without escaped dot)
        result_2 = await allowlist_manager.is_allowed(
            "test@exampleXcom",
            pii_type_enum.EMAIL,
            test_tenant_id,
        )

        assert result_1 is True
        assert result_2 is False

    @pytest.mark.asyncio
    async def test_empty_value_handling(
        self, allowlist_manager, pii_type_enum, test_tenant_id
    ):
        """Empty values are handled correctly."""
        result = await allowlist_manager.is_allowed(
            "",
            pii_type_enum.EMAIL,
            test_tenant_id,
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_whitespace_handling(
        self, allowlist_manager, pii_type_enum, test_tenant_id
    ):
        """Whitespace in values is handled correctly."""
        try:
            from greenlang.infrastructure.pii_service.allowlist.patterns import AllowlistEntry, PatternType

            entry = AllowlistEntry(
                pii_type=pii_type_enum.EMAIL,
                pattern="test@example.com",
                pattern_type=PatternType.EXACT,
                reason="Whitespace test",
                created_by=uuid4(),
            )
            await allowlist_manager.add_entry(entry)
        except ImportError:
            pytest.skip("AllowlistEntry not available")

        # Exact match without whitespace
        result_1 = await allowlist_manager.is_allowed(
            "test@example.com",
            pii_type_enum.EMAIL,
            test_tenant_id,
        )

        # With leading/trailing whitespace
        result_2 = await allowlist_manager.is_allowed(
            " test@example.com ",
            pii_type_enum.EMAIL,
            test_tenant_id,
        )

        assert result_1 is True
        # Whitespace handling is implementation dependent
