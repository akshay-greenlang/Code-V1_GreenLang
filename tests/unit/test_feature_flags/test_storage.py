# -*- coding: utf-8 -*-
"""
Unit Tests for InMemoryFlagStorage - INFRA-008

Tests the in-memory storage backend including flag CRUD, rules, overrides,
variants, audit log, LRU eviction, TTL expiration, and concurrent access.
"""

import asyncio
import time
from unittest.mock import patch

import pytest

from greenlang.infrastructure.feature_flags.models import (
    AuditLogEntry,
    FeatureFlag,
    FlagOverride,
    FlagRule,
    FlagStatus,
    FlagType,
    FlagVariant,
)
from greenlang.infrastructure.feature_flags.storage.memory import (
    InMemoryFlagStorage,
    _CacheEntry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def storage():
    """Create a fresh InMemoryFlagStorage instance with no TTL."""
    return InMemoryFlagStorage(max_size=100, default_ttl=0.0)


@pytest.fixture
def sample_flag():
    """Create a sample FeatureFlag for testing."""
    return FeatureFlag(
        key="test-flag",
        name="Test Flag",
        flag_type=FlagType.BOOLEAN,
        status=FlagStatus.ACTIVE,
        default_value=False,
        tags=["platform"],
        owner="team-alpha",
    )


@pytest.fixture
def sample_rule():
    """Create a sample FlagRule for testing."""
    return FlagRule(
        rule_id="rule-001",
        flag_key="test-flag",
        rule_type="user_list",
        priority=10,
        conditions={"users": ["user-42"]},
    )


@pytest.fixture
def sample_override():
    """Create a sample FlagOverride for testing."""
    return FlagOverride(
        flag_key="test-flag",
        scope_type="user",
        scope_value="user-42",
        enabled=True,
    )


@pytest.fixture
def sample_variant():
    """Create a sample FlagVariant for testing."""
    return FlagVariant(
        variant_key="control",
        flag_key="test-flag",
        variant_value={"color": "blue"},
        weight=50.0,
    )


# ---------------------------------------------------------------------------
# Flag CRUD Tests
# ---------------------------------------------------------------------------


class TestFlagCRUD:
    """Test flag create, read, update, delete operations."""

    @pytest.mark.asyncio
    async def test_save_and_get_flag_roundtrip(self, storage, sample_flag):
        """Saving and then getting a flag returns the same data."""
        await storage.save_flag(sample_flag)
        retrieved = await storage.get_flag("test-flag")

        assert retrieved is not None
        assert retrieved.key == "test-flag"
        assert retrieved.name == "Test Flag"
        assert retrieved.flag_type == FlagType.BOOLEAN
        assert retrieved.status == FlagStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_get_flag_nonexistent_returns_none(self, storage):
        """Getting a flag that does not exist returns None."""
        result = await storage.get_flag("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_save_flag_upsert_overwrites(self, storage, sample_flag):
        """Saving a flag with the same key overwrites the existing entry."""
        await storage.save_flag(sample_flag)
        updated = sample_flag.model_copy(update={"name": "Updated Name"})
        await storage.save_flag(updated)

        retrieved = await storage.get_flag("test-flag")
        assert retrieved.name == "Updated Name"

    @pytest.mark.asyncio
    async def test_delete_flag_removes_flag(self, storage, sample_flag):
        """Deleting a flag removes it from storage."""
        await storage.save_flag(sample_flag)
        deleted = await storage.delete_flag("test-flag")

        assert deleted is True
        assert await storage.get_flag("test-flag") is None

    @pytest.mark.asyncio
    async def test_delete_flag_nonexistent_returns_false(self, storage):
        """Deleting a nonexistent flag returns False."""
        result = await storage.delete_flag("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_flag_cascades_to_sub_entities(
        self, storage, sample_flag, sample_rule, sample_override, sample_variant
    ):
        """Deleting a flag also removes its rules, overrides, and variants."""
        await storage.save_flag(sample_flag)
        await storage.save_rule(sample_rule)
        await storage.save_override(sample_override)
        await storage.save_variant(sample_variant)

        await storage.delete_flag("test-flag")

        assert await storage.get_rules("test-flag") == []
        assert await storage.get_overrides("test-flag") == []
        assert await storage.get_variants("test-flag") == []


# ---------------------------------------------------------------------------
# Get All Flags with Filters
# ---------------------------------------------------------------------------


class TestGetAllFlags:
    """Test get_all_flags with status and tag filtering."""

    @pytest.mark.asyncio
    async def test_get_all_flags_no_filter(self, storage):
        """get_all_flags with no filter returns all flags."""
        f1 = FeatureFlag(key="flag-a", name="A", status=FlagStatus.ACTIVE)
        f2 = FeatureFlag(key="flag-b", name="B", status=FlagStatus.DRAFT)
        await storage.save_flag(f1)
        await storage.save_flag(f2)

        results = await storage.get_all_flags()
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_get_all_flags_status_filter(self, storage):
        """get_all_flags with status_filter returns only matching flags."""
        f1 = FeatureFlag(key="flag-a", name="A", status=FlagStatus.ACTIVE)
        f2 = FeatureFlag(key="flag-b", name="B", status=FlagStatus.DRAFT)
        await storage.save_flag(f1)
        await storage.save_flag(f2)

        results = await storage.get_all_flags(status_filter=FlagStatus.ACTIVE)
        assert len(results) == 1
        assert results[0].key == "flag-a"

    @pytest.mark.asyncio
    async def test_get_all_flags_tag_filter(self, storage):
        """get_all_flags with tag_filter returns only flags containing the tag."""
        f1 = FeatureFlag(key="flag-a", name="A", tags=["platform", "scope3"])
        f2 = FeatureFlag(key="flag-b", name="B", tags=["cbam"])
        await storage.save_flag(f1)
        await storage.save_flag(f2)

        results = await storage.get_all_flags(tag_filter="platform")
        assert len(results) == 1
        assert results[0].key == "flag-a"


# ---------------------------------------------------------------------------
# Rules Tests
# ---------------------------------------------------------------------------


class TestRulesStorage:
    """Test rule save and retrieval."""

    @pytest.mark.asyncio
    async def test_save_and_get_rules(self, storage, sample_rule):
        """Saving and getting rules returns the saved rules sorted by priority."""
        rule2 = FlagRule(
            rule_id="rule-002",
            flag_key="test-flag",
            rule_type="environment",
            priority=5,
            conditions={"environments": ["staging"]},
        )
        await storage.save_rule(sample_rule)
        await storage.save_rule(rule2)

        rules = await storage.get_rules("test-flag")
        assert len(rules) == 2
        # Should be sorted by priority ascending
        assert rules[0].rule_id == "rule-002"  # priority 5
        assert rules[1].rule_id == "rule-001"  # priority 10

    @pytest.mark.asyncio
    async def test_get_rules_empty(self, storage):
        """Getting rules for a flag with no rules returns empty list."""
        rules = await storage.get_rules("no-rules-flag")
        assert rules == []

    @pytest.mark.asyncio
    async def test_save_rule_upsert_by_rule_id(self, storage, sample_rule):
        """Saving a rule with the same rule_id replaces the existing one."""
        await storage.save_rule(sample_rule)
        updated = FlagRule(
            rule_id="rule-001",
            flag_key="test-flag",
            rule_type="environment",
            priority=1,
            conditions={"environments": ["prod"]},
        )
        await storage.save_rule(updated)

        rules = await storage.get_rules("test-flag")
        assert len(rules) == 1
        assert rules[0].rule_type == "environment"


# ---------------------------------------------------------------------------
# Overrides Tests
# ---------------------------------------------------------------------------


class TestOverridesStorage:
    """Test override save and retrieval."""

    @pytest.mark.asyncio
    async def test_save_and_get_overrides(self, storage, sample_override):
        """Saving and getting overrides returns the saved overrides."""
        await storage.save_override(sample_override)
        overrides = await storage.get_overrides("test-flag")
        assert len(overrides) == 1
        assert overrides[0].scope_value == "user-42"

    @pytest.mark.asyncio
    async def test_get_overrides_empty(self, storage):
        """Getting overrides for a flag with none returns empty list."""
        overrides = await storage.get_overrides("no-overrides-flag")
        assert overrides == []

    @pytest.mark.asyncio
    async def test_save_override_upsert_by_scope(self, storage, sample_override):
        """Saving an override with same (scope_type, scope_value) replaces it."""
        await storage.save_override(sample_override)
        updated = FlagOverride(
            flag_key="test-flag",
            scope_type="user",
            scope_value="user-42",
            enabled=False,  # Changed
        )
        await storage.save_override(updated)

        overrides = await storage.get_overrides("test-flag")
        assert len(overrides) == 1
        assert overrides[0].enabled is False


# ---------------------------------------------------------------------------
# Variants Tests
# ---------------------------------------------------------------------------


class TestVariantsStorage:
    """Test variant save and retrieval."""

    @pytest.mark.asyncio
    async def test_save_and_get_variants(self, storage, sample_variant):
        """Saving and getting variants returns the saved variants."""
        await storage.save_variant(sample_variant)
        variants = await storage.get_variants("test-flag")
        assert len(variants) == 1
        assert variants[0].variant_key == "control"

    @pytest.mark.asyncio
    async def test_get_variants_empty(self, storage):
        """Getting variants for a flag with none returns empty list."""
        variants = await storage.get_variants("no-variants-flag")
        assert variants == []

    @pytest.mark.asyncio
    async def test_save_variant_upsert_by_key(self, storage, sample_variant):
        """Saving a variant with the same variant_key replaces it."""
        await storage.save_variant(sample_variant)
        updated = FlagVariant(
            variant_key="control",
            flag_key="test-flag",
            variant_value={"color": "red"},
            weight=60.0,
        )
        await storage.save_variant(updated)

        variants = await storage.get_variants("test-flag")
        assert len(variants) == 1
        assert variants[0].weight == 60.0


# ---------------------------------------------------------------------------
# Audit Log Tests
# ---------------------------------------------------------------------------


class TestAuditLogStorage:
    """Test audit log append and retrieval."""

    @pytest.mark.asyncio
    async def test_log_audit_and_retrieve(self, storage):
        """Logging an audit entry and retrieving it works correctly."""
        entry = AuditLogEntry(
            flag_key="test-flag",
            action="created",
            changed_by="admin",
        )
        await storage.log_audit(entry)

        log = await storage.get_audit_log("test-flag")
        assert len(log) == 1
        assert log[0].action == "created"
        assert log[0].changed_by == "admin"

    @pytest.mark.asyncio
    async def test_get_audit_log_empty(self, storage):
        """Getting audit log for a flag with no entries returns empty list."""
        log = await storage.get_audit_log("no-audit-flag")
        assert log == []

    @pytest.mark.asyncio
    async def test_audit_log_ordered_newest_first(self, storage):
        """Audit log entries are returned in reverse chronological order."""
        import time as _time

        for action in ["created", "updated", "killed"]:
            entry = AuditLogEntry(
                flag_key="test-flag",
                action=action,
            )
            await storage.log_audit(entry)
            _time.sleep(0.01)  # Ensure distinct timestamps

        log = await storage.get_audit_log("test-flag")
        assert len(log) == 3
        # Newest first
        assert log[0].action == "killed"
        assert log[2].action == "created"


# ---------------------------------------------------------------------------
# LRU Eviction Tests
# ---------------------------------------------------------------------------


class TestLRUEviction:
    """Test LRU eviction when max_size is exceeded."""

    @pytest.mark.asyncio
    async def test_lru_eviction_oldest_removed(self):
        """When max_size is exceeded, the least recently used entry is evicted."""
        small_storage = InMemoryFlagStorage(max_size=3, default_ttl=0.0)

        for i in range(4):
            flag = FeatureFlag(key=f"flag-{i}", name=f"F{i}")
            await small_storage.save_flag(flag)

        # flag-0 was the first in and was never touched again, so it should be evicted
        assert await small_storage.get_flag("flag-0") is None
        # Newer flags should still be present
        assert await small_storage.get_flag("flag-1") is not None
        assert await small_storage.get_flag("flag-2") is not None
        assert await small_storage.get_flag("flag-3") is not None

    @pytest.mark.asyncio
    async def test_lru_touch_prevents_eviction(self):
        """Accessing a flag promotes it to most-recently-used, preventing eviction."""
        small_storage = InMemoryFlagStorage(max_size=3, default_ttl=0.0)

        for i in range(3):
            flag = FeatureFlag(key=f"flag-{i}", name=f"F{i}")
            await small_storage.save_flag(flag)

        # Touch flag-0 to make it recently used
        await small_storage.get_flag("flag-0")

        # Add one more to trigger eviction
        flag_new = FeatureFlag(key="flag-new", name="New")
        await small_storage.save_flag(flag_new)

        # flag-0 was touched, so flag-1 should be evicted instead
        assert await small_storage.get_flag("flag-0") is not None
        assert await small_storage.get_flag("flag-1") is None


# ---------------------------------------------------------------------------
# TTL Expiration Tests
# ---------------------------------------------------------------------------


class TestTTLExpiration:
    """Test TTL-based lazy expiration."""

    @pytest.mark.asyncio
    async def test_expired_entry_returns_none(self):
        """A flag whose TTL has expired returns None on get."""
        ttl_storage = InMemoryFlagStorage(max_size=100, default_ttl=0.1)

        flag = FeatureFlag(key="ttl-flag", name="TTL")
        await ttl_storage.save_flag(flag)

        # Flag should be retrievable immediately
        assert await ttl_storage.get_flag("ttl-flag") is not None

        # Wait for TTL to expire
        await asyncio.sleep(0.2)

        # Flag should now be expired and return None
        assert await ttl_storage.get_flag("ttl-flag") is None

    @pytest.mark.asyncio
    async def test_no_ttl_never_expires(self, storage, sample_flag):
        """A flag with TTL=0 (no expiry) never expires."""
        await storage.save_flag(sample_flag)

        # Even after a brief wait, the flag should be present
        await asyncio.sleep(0.05)
        assert await storage.get_flag("test-flag") is not None


# ---------------------------------------------------------------------------
# Concurrent Access Tests
# ---------------------------------------------------------------------------


class TestConcurrentAccess:
    """Test thread-safety under concurrent asyncio access."""

    @pytest.mark.asyncio
    async def test_concurrent_save_and_get(self, storage):
        """Multiple concurrent save and get operations do not corrupt data."""
        num_operations = 50

        async def save_flag(idx: int):
            flag = FeatureFlag(key=f"concurrent-{idx}", name=f"C{idx}")
            await storage.save_flag(flag)

        async def get_flag(idx: int):
            await storage.get_flag(f"concurrent-{idx}")

        # Run saves concurrently
        await asyncio.gather(
            *[save_flag(i) for i in range(num_operations)]
        )

        # Run gets concurrently
        results = await asyncio.gather(
            *[storage.get_flag(f"concurrent-{i}") for i in range(num_operations)]
        )

        for i, result in enumerate(results):
            assert result is not None, f"Flag concurrent-{i} should exist"
            assert result.key == f"concurrent-{i}"

    @pytest.mark.asyncio
    async def test_concurrent_mixed_operations(self, storage):
        """Mixed concurrent operations (save, get, delete) are safe."""
        flag = FeatureFlag(key="mixed-flag", name="Mixed")
        await storage.save_flag(flag)

        async def read():
            return await storage.get_flag("mixed-flag")

        async def write():
            f = FeatureFlag(key="mixed-flag", name="Updated")
            await storage.save_flag(f)

        # Run mixed operations concurrently -- should not raise
        tasks = [read() if i % 2 == 0 else write() for i in range(20)]
        await asyncio.gather(*tasks)

        # Final state should be consistent
        final = await storage.get_flag("mixed-flag")
        assert final is not None


# ---------------------------------------------------------------------------
# Health Check and Size Tests
# ---------------------------------------------------------------------------


class TestHealthAndSize:
    """Test health_check and size methods."""

    @pytest.mark.asyncio
    async def test_health_check_returns_healthy(self, storage, sample_flag):
        """health_check returns a healthy status with correct counts."""
        await storage.save_flag(sample_flag)
        health = await storage.health_check()
        assert health["healthy"] is True
        assert health["flags_count"] == 1
        assert health["backend"] == "InMemoryFlagStorage"

    @pytest.mark.asyncio
    async def test_size_returns_current_count(self, storage):
        """size() returns the current number of non-expired entries."""
        for i in range(5):
            flag = FeatureFlag(key=f"size-{i}", name=f"S{i}")
            await storage.save_flag(flag)

        count = await storage.size()
        assert count == 5
