# -*- coding: utf-8 -*-
"""
Unit tests for deterministic LLM caching

Tests DeterministicLLM functionality including:
- Record mode stores responses
- Replay mode uses cached responses
- Golden mode loads pre-recorded
- Cache key includes all factors
- Cache hit/miss statistics
- Export/import golden datasets
- Thread safety
"""

import pytest
import asyncio
import json
from pathlib import Path
from greenlang.intelligence.determinism import (
    DeterministicLLM,
    CacheMode,
    CacheEntry,
    CacheStats,
    JSONCacheBackend,
    SQLiteCacheBackend,
    create_deterministic_provider,
    create_golden_provider,
)
from greenlang.intelligence.schemas.messages import ChatMessage, Role
from greenlang.intelligence.schemas.tools import ToolDef
from greenlang.intelligence.schemas.responses import ChatResponse, Usage, FinishReason, ProviderInfo
from greenlang.intelligence.runtime.budget import Budget

from tests.intelligence.fakes import FakeProvider, make_text_response


class TestRecordMode:
    """Test record mode - calls real LLM and caches responses"""

    @pytest.mark.asyncio
    async def test_record_mode_calls_provider(self, tmp_path):
        """Record mode should call underlying provider"""
        fake = FakeProvider([make_text_response("Test response", cost_usd=0.05)])
        cache_path = tmp_path / "cache.db"

        deterministic = DeterministicLLM.wrap(
            provider=fake,
            mode="record",
            cache_path=cache_path
        )

        budget = Budget(max_usd=0.50)
        response = await deterministic.chat(
            messages=[ChatMessage(role=Role.user, content="Test")],
            budget=budget
        )

        # Should have called provider
        assert fake.get_call_count() == 1
        assert response.text == "Test response"

    @pytest.mark.asyncio
    async def test_record_mode_caches_response(self, tmp_path):
        """Record mode should cache response for future use"""
        fake = FakeProvider([make_text_response("Cached response", cost_usd=0.05)])
        cache_path = tmp_path / "cache.db"

        deterministic = DeterministicLLM.wrap(
            provider=fake,
            mode="record",
            cache_path=cache_path
        )

        budget = Budget(max_usd=0.50)
        messages = [ChatMessage(role=Role.user, content="Test")]

        # First call
        response1 = await deterministic.chat(messages=messages, budget=budget)

        # Second call with same messages
        response2 = await deterministic.chat(messages=messages, budget=budget)

        # First call should miss cache (call provider)
        # Second call should hit cache (not call provider again)
        assert fake.get_call_count() == 1  # Only called once
        assert response1.text == response2.text

    @pytest.mark.asyncio
    async def test_record_mode_updates_stats(self, tmp_path):
        """Record mode should update cache statistics"""
        fake = FakeProvider([make_text_response("Response", cost_usd=0.05)])
        cache_path = tmp_path / "cache.db"

        deterministic = DeterministicLLM.wrap(
            provider=fake,
            mode="record",
            cache_path=cache_path
        )

        budget = Budget(max_usd=0.50)
        messages = [ChatMessage(role=Role.user, content="Test")]

        # First call - cache miss
        await deterministic.chat(messages=messages, budget=budget)
        stats1 = deterministic.stats()
        assert stats1.misses == 1
        assert stats1.hits == 0

        # Second call - cache hit
        await deterministic.chat(messages=messages, budget=budget)
        stats2 = deterministic.stats()
        assert stats2.misses == 1
        assert stats2.hits == 1


class TestReplayMode:
    """Test replay mode - uses cached responses only"""

    @pytest.mark.asyncio
    async def test_replay_mode_uses_cache(self, tmp_path):
        """Replay mode should use cached responses"""
        fake = FakeProvider([make_text_response("Cached", cost_usd=0.05)])
        cache_path = tmp_path / "cache.db"

        # Record first
        deterministic_record = DeterministicLLM.wrap(
            provider=fake,
            mode="record",
            cache_path=cache_path
        )

        budget = Budget(max_usd=0.50)
        messages = [ChatMessage(role=Role.user, content="Test")]

        await deterministic_record.chat(messages=messages, budget=budget)

        # Now replay with different provider (won't be called)
        fake_replay = FakeProvider([make_text_response("Should not see this")])
        deterministic_replay = DeterministicLLM.wrap(
            provider=fake_replay,
            mode="replay",
            cache_path=cache_path
        )

        response = await deterministic_replay.chat(messages=messages, budget=budget)

        # Should use cached response
        assert response.text == "Cached"
        # Should not call provider
        assert fake_replay.get_call_count() == 0

    @pytest.mark.asyncio
    async def test_replay_mode_raises_on_cache_miss(self, tmp_path):
        """Replay mode should raise error on cache miss"""
        fake = FakeProvider([make_text_response("Response")])
        cache_path = tmp_path / "cache.db"

        deterministic = DeterministicLLM.wrap(
            provider=fake,
            mode="replay",
            cache_path=cache_path
        )

        budget = Budget(max_usd=0.50)
        messages = [ChatMessage(role=Role.user, content="Not in cache")]

        with pytest.raises(ValueError, match="Cache miss in replay mode"):
            await deterministic.chat(messages=messages, budget=budget)

    @pytest.mark.asyncio
    async def test_replay_mode_deterministic(self, tmp_path):
        """Replay mode should return same response every time"""
        fake = FakeProvider([make_text_response("Original", cost_usd=0.05)])
        cache_path = tmp_path / "cache.db"

        # Record
        deterministic_record = DeterministicLLM.wrap(
            provider=fake,
            mode="record",
            cache_path=cache_path
        )

        budget = Budget(max_usd=0.50)
        messages = [ChatMessage(role=Role.user, content="Test")]

        await deterministic_record.chat(messages=messages, budget=budget)

        # Replay multiple times
        deterministic_replay = DeterministicLLM.wrap(
            provider=FakeProvider([make_text_response("Different")]),
            mode="replay",
            cache_path=cache_path
        )

        responses = []
        for _ in range(5):
            budget_new = Budget(max_usd=0.50)
            response = await deterministic_replay.chat(messages=messages, budget=budget_new)
            responses.append(response.text)

        # All should be identical
        assert all(r == "Original" for r in responses)


class TestGoldenMode:
    """Test golden mode - uses pre-recorded golden responses"""

    @pytest.mark.asyncio
    async def test_golden_mode_loads_from_file(self, tmp_path):
        """Golden mode should load responses from JSON file"""
        golden_path = tmp_path / "golden.json"

        # Create a golden response manually
        fake_for_key = FakeProvider([make_text_response("Golden")])
        deterministic_temp = DeterministicLLM.wrap(
            provider=fake_for_key,
            mode="record",
            cache_path=golden_path
        )

        budget = Budget(max_usd=0.50)
        messages = [ChatMessage(role=Role.user, content="Test")]

        # Record golden response
        await deterministic_temp.chat(messages=messages, budget=budget)

        # Now use golden mode
        fake_golden = FakeProvider([make_text_response("Should not use")])
        deterministic_golden = DeterministicLLM.wrap(
            provider=fake_golden,
            mode="golden",
            cache_path=golden_path
        )

        response = await deterministic_golden.chat(messages=messages, budget=budget)

        # Should use golden response
        assert response.text == "Golden"
        assert fake_golden.get_call_count() == 0

    @pytest.mark.asyncio
    async def test_golden_mode_raises_on_miss(self, tmp_path):
        """Golden mode should raise on cache miss"""
        golden_path = tmp_path / "golden.json"
        golden_path.write_text("{}")  # Empty cache

        fake = FakeProvider([make_text_response("Response")])
        deterministic = DeterministicLLM.wrap(
            provider=fake,
            mode="golden",
            cache_path=golden_path
        )

        budget = Budget(max_usd=0.50)
        messages = [ChatMessage(role=Role.user, content="Not in golden")]

        with pytest.raises(ValueError, match="Cache miss in golden mode"):
            await deterministic.chat(messages=messages, budget=budget)


class TestCacheKeyComputation:
    """Test cache key includes all relevant factors"""

    @pytest.mark.asyncio
    async def test_cache_key_includes_messages(self, tmp_path):
        """Cache key should include message content"""
        fake = FakeProvider([
            make_text_response("Response 1"),
            make_text_response("Response 2"),
        ])
        cache_path = tmp_path / "cache.db"

        deterministic = DeterministicLLM.wrap(
            provider=fake,
            mode="record",
            cache_path=cache_path
        )

        budget1 = Budget(max_usd=0.50)
        budget2 = Budget(max_usd=0.50)

        # Different messages should have different cache keys
        response1 = await deterministic.chat(
            messages=[ChatMessage(role=Role.user, content="Message 1")],
            budget=budget1
        )

        response2 = await deterministic.chat(
            messages=[ChatMessage(role=Role.user, content="Message 2")],
            budget=budget2
        )

        # Should call provider twice (different cache keys)
        assert fake.get_call_count() == 2
        assert response1.text != response2.text

    @pytest.mark.asyncio
    async def test_cache_key_includes_temperature(self, tmp_path):
        """Cache key should include temperature parameter"""
        fake = FakeProvider([
            make_text_response("Temp 0"),
            make_text_response("Temp 0.5"),
        ])
        cache_path = tmp_path / "cache.db"

        deterministic = DeterministicLLM.wrap(
            provider=fake,
            mode="record",
            cache_path=cache_path
        )

        messages = [ChatMessage(role=Role.user, content="Test")]
        budget1 = Budget(max_usd=0.50)
        budget2 = Budget(max_usd=0.50)

        # Different temperatures should have different cache keys
        response1 = await deterministic.chat(
            messages=messages, budget=budget1, temperature=0.0
        )

        response2 = await deterministic.chat(
            messages=messages, budget=budget2, temperature=0.5
        )

        # Should call provider twice
        assert fake.get_call_count() == 2

    @pytest.mark.asyncio
    async def test_cache_key_includes_tools(self, tmp_path):
        """Cache key should include tools parameter"""
        fake = FakeProvider([
            make_text_response("No tools"),
            make_text_response("With tools"),
        ])
        cache_path = tmp_path / "cache.db"

        deterministic = DeterministicLLM.wrap(
            provider=fake,
            mode="record",
            cache_path=cache_path
        )

        messages = [ChatMessage(role=Role.user, content="Test")]
        budget1 = Budget(max_usd=0.50)
        budget2 = Budget(max_usd=0.50)

        tools = [ToolDef(name="test_tool", description="", parameters={})]

        # Without tools
        response1 = await deterministic.chat(
            messages=messages, budget=budget1, tools=None
        )

        # With tools
        response2 = await deterministic.chat(
            messages=messages, budget=budget2, tools=tools
        )

        # Should call provider twice (different cache keys)
        assert fake.get_call_count() == 2

    @pytest.mark.asyncio
    async def test_cache_key_includes_seed(self, tmp_path):
        """Cache key should include seed parameter"""
        fake = FakeProvider([
            make_text_response("Seed 42"),
            make_text_response("Seed 123"),
        ])
        cache_path = tmp_path / "cache.db"

        deterministic = DeterministicLLM.wrap(
            provider=fake,
            mode="record",
            cache_path=cache_path
        )

        messages = [ChatMessage(role=Role.user, content="Test")]
        budget1 = Budget(max_usd=0.50)
        budget2 = Budget(max_usd=0.50)

        # Different seeds
        response1 = await deterministic.chat(
            messages=messages, budget=budget1, seed=42
        )

        response2 = await deterministic.chat(
            messages=messages, budget=budget2, seed=123
        )

        # Should call provider twice
        assert fake.get_call_count() == 2


class TestCacheStatistics:
    """Test cache hit/miss statistics"""

    @pytest.mark.asyncio
    async def test_stats_tracks_hits(self, tmp_path):
        """Stats should track cache hits"""
        fake = FakeProvider([make_text_response("Response", cost_usd=0.05)])
        cache_path = tmp_path / "cache.db"

        deterministic = DeterministicLLM.wrap(
            provider=fake,
            mode="record",
            cache_path=cache_path
        )

        budget = Budget(max_usd=0.50)
        messages = [ChatMessage(role=Role.user, content="Test")]

        # First call - miss
        await deterministic.chat(messages=messages, budget=budget)

        # Second call - hit
        await deterministic.chat(messages=messages, budget=budget)

        # Third call - hit
        await deterministic.chat(messages=messages, budget=budget)

        stats = deterministic.stats()
        assert stats.hits == 2
        assert stats.misses == 1
        assert stats.total_requests == 3

    @pytest.mark.asyncio
    async def test_stats_calculates_hit_rate(self, tmp_path):
        """Stats should calculate hit rate percentage"""
        fake = FakeProvider([
            make_text_response("R1", cost_usd=0.05),
            make_text_response("R2", cost_usd=0.05),
        ])
        cache_path = tmp_path / "cache.db"

        deterministic = DeterministicLLM.wrap(
            provider=fake,
            mode="record",
            cache_path=cache_path
        )

        messages1 = [ChatMessage(role=Role.user, content="Test 1")]
        messages2 = [ChatMessage(role=Role.user, content="Test 2")]

        # 2 misses
        await deterministic.chat(messages=messages1, budget=Budget(max_usd=0.50))
        await deterministic.chat(messages=messages2, budget=Budget(max_usd=0.50))

        # 2 hits
        await deterministic.chat(messages=messages1, budget=Budget(max_usd=0.50))
        await deterministic.chat(messages=messages2, budget=Budget(max_usd=0.50))

        stats = deterministic.stats()
        assert stats.total_requests == 4
        assert stats.hit_rate == 50.0  # 2/4 = 50%

    @pytest.mark.asyncio
    async def test_stats_tracks_cost_saved(self, tmp_path):
        """Stats should track USD saved by cache hits"""
        fake = FakeProvider([make_text_response("Response", cost_usd=0.10)])
        cache_path = tmp_path / "cache.db"

        deterministic = DeterministicLLM.wrap(
            provider=fake,
            mode="record",
            cache_path=cache_path
        )

        messages = [ChatMessage(role=Role.user, content="Test")]

        # First call - miss (costs $0.10)
        await deterministic.chat(messages=messages, budget=Budget(max_usd=0.50))

        # Second call - hit (saves $0.10)
        await deterministic.chat(messages=messages, budget=Budget(max_usd=0.50))

        # Third call - hit (saves $0.10)
        await deterministic.chat(messages=messages, budget=Budget(max_usd=0.50))

        stats = deterministic.stats()
        assert stats.saved_usd == 0.20  # Saved 2 * $0.10


class TestExportImportGolden:
    """Test export/import of golden datasets"""

    @pytest.mark.asyncio
    async def test_export_golden_creates_file(self, tmp_path):
        """export_golden() should create JSON file"""
        fake = FakeProvider([make_text_response("Response", cost_usd=0.05)])
        cache_path = tmp_path / "cache.db"
        golden_path = tmp_path / "golden.json"

        deterministic = DeterministicLLM.wrap(
            provider=fake,
            mode="record",
            cache_path=cache_path
        )

        # Record some responses
        await deterministic.chat(
            messages=[ChatMessage(role=Role.user, content="Test")],
            budget=Budget(max_usd=0.50)
        )

        # Export golden
        deterministic.export_golden(golden_path)

        # File should exist
        assert golden_path.exists()

        # Should be valid JSON
        data = json.loads(golden_path.read_text())
        assert len(data) > 0

    @pytest.mark.asyncio
    async def test_import_golden_loads_responses(self, tmp_path):
        """import_golden() should load responses into cache"""
        fake = FakeProvider([make_text_response("Original", cost_usd=0.05)])
        cache_path1 = tmp_path / "cache1.db"
        cache_path2 = tmp_path / "cache2.db"
        golden_path = tmp_path / "golden.json"

        # Record and export
        deterministic1 = DeterministicLLM.wrap(
            provider=fake,
            mode="record",
            cache_path=cache_path1
        )

        messages = [ChatMessage(role=Role.user, content="Test")]
        await deterministic1.chat(messages=messages, budget=Budget(max_usd=0.50))
        deterministic1.export_golden(golden_path)

        # Import into new cache
        deterministic2 = DeterministicLLM.wrap(
            provider=FakeProvider([make_text_response("Should not use")]),
            mode="replay",
            cache_path=cache_path2
        )

        deterministic2.import_golden(golden_path)

        # Should be able to replay
        response = await deterministic2.chat(messages=messages, budget=Budget(max_usd=0.50))
        assert response.text == "Original"

    @pytest.mark.asyncio
    async def test_export_import_round_trip(self, tmp_path):
        """Export and import should preserve all data"""
        fake = FakeProvider([make_text_response("Test", cost_usd=0.05, tokens=100)])
        cache_path = tmp_path / "cache.db"
        golden_path = tmp_path / "golden.json"

        deterministic = DeterministicLLM.wrap(
            provider=fake,
            mode="record",
            cache_path=cache_path
        )

        messages = [ChatMessage(role=Role.user, content="Test")]
        original_response = await deterministic.chat(
            messages=messages,
            budget=Budget(max_usd=0.50),
            temperature=0.7,
            seed=42
        )

        # Export
        deterministic.export_golden(golden_path)

        # Clear and re-import
        deterministic.clear_cache()
        deterministic.import_golden(golden_path)

        # Should get same response
        restored_response = await deterministic.chat(
            messages=messages,
            budget=Budget(max_usd=0.50),
            temperature=0.7,
            seed=42
        )

        assert restored_response.text == original_response.text
        assert restored_response.usage.cost_usd == original_response.usage.cost_usd


class TestCacheBackends:
    """Test different cache backend implementations"""

    @pytest.mark.asyncio
    async def test_json_backend(self, tmp_path):
        """JSONCacheBackend should work correctly"""
        cache_path = tmp_path / "cache.json"
        backend = JSONCacheBackend(cache_path)

        # Create test entry
        entry = CacheEntry(
            cache_key="test_key",
            prompt_hash="prompt_hash",
            model="test-model",
            temperature=0.0,
            seed=42,
            timestamp="2025-10-01T12:00:00Z",
            response=ChatResponse(
                text="Test",
                tool_calls=[],
                usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30, cost_usd=0.01),
                finish_reason=FinishReason.stop,
                provider_info=ProviderInfo(provider="test", model="test-model"),
                raw=None
            )
        )

        # Store
        backend.set(entry)

        # Retrieve
        retrieved = backend.get("test_key")

        assert retrieved is not None
        assert retrieved.cache_key == "test_key"
        assert retrieved.response.text == "Test"

    @pytest.mark.asyncio
    async def test_sqlite_backend(self, tmp_path):
        """SQLiteCacheBackend should work correctly"""
        cache_path = tmp_path / "cache.db"
        backend = SQLiteCacheBackend(cache_path)

        # Create test entry
        entry = CacheEntry(
            cache_key="test_key",
            prompt_hash="prompt_hash",
            model="test-model",
            temperature=0.0,
            seed=42,
            timestamp="2025-10-01T12:00:00Z",
            response=ChatResponse(
                text="Test",
                tool_calls=[],
                usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30, cost_usd=0.01),
                finish_reason=FinishReason.stop,
                provider_info=ProviderInfo(provider="test", model="test-model"),
                raw=None
            )
        )

        # Store
        backend.set(entry)

        # Retrieve
        retrieved = backend.get("test_key")

        assert retrieved is not None
        assert retrieved.cache_key == "test_key"
        assert retrieved.response.text == "Test"

    @pytest.mark.asyncio
    async def test_backend_clear(self, tmp_path):
        """Backend clear() should remove all entries"""
        cache_path = tmp_path / "cache.db"
        backend = SQLiteCacheBackend(cache_path)

        # Add entry
        entry = CacheEntry(
            cache_key="test_key",
            prompt_hash="hash",
            model="model",
            temperature=0.0,
            seed=None,
            timestamp="2025-10-01T12:00:00Z",
            response=ChatResponse(
                text="Test",
                tool_calls=[],
                usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30, cost_usd=0.01),
                finish_reason=FinishReason.stop,
                provider_info=ProviderInfo(provider="test", model="test"),
                raw=None
            )
        )

        backend.set(entry)

        # Clear
        backend.clear()

        # Should not find entry
        assert backend.get("test_key") is None


class TestThreadSafety:
    """Test thread safety of cache operations"""

    @pytest.mark.asyncio
    async def test_concurrent_cache_access(self, tmp_path):
        """Should handle concurrent cache access"""
        fake = FakeProvider([make_text_response(f"Response {i}", cost_usd=0.05) for i in range(10)])
        cache_path = tmp_path / "cache.db"

        deterministic = DeterministicLLM.wrap(
            provider=fake,
            mode="record",
            cache_path=cache_path
        )

        async def make_call(i):
            messages = [ChatMessage(role=Role.user, content=f"Test {i}")]
            return await deterministic.chat(messages=messages, budget=Budget(max_usd=0.50))

        # Make multiple concurrent calls
        tasks = [make_call(i) for i in range(5)]
        responses = await asyncio.gather(*tasks)

        # All should succeed
        assert len(responses) == 5
        assert all(r is not None for r in responses)

    @pytest.mark.asyncio
    async def test_concurrent_cache_hits(self, tmp_path):
        """Should handle concurrent cache hits"""
        fake = FakeProvider([make_text_response("Response", cost_usd=0.05)])
        cache_path = tmp_path / "cache.db"

        deterministic = DeterministicLLM.wrap(
            provider=fake,
            mode="record",
            cache_path=cache_path
        )

        messages = [ChatMessage(role=Role.user, content="Test")]

        # Record first
        await deterministic.chat(messages=messages, budget=Budget(max_usd=0.50))

        # Now make concurrent cache hits
        async def get_cached():
            return await deterministic.chat(messages=messages, budget=Budget(max_usd=0.50))

        tasks = [get_cached() for _ in range(10)]
        responses = await asyncio.gather(*tasks)

        # All should return same response
        assert len(responses) == 10
        assert all(r.text == "Response" for r in responses)

        # Should only have called provider once
        assert fake.get_call_count() == 1


class TestConvenienceFunctions:
    """Test convenience functions"""

    @pytest.mark.asyncio
    async def test_create_deterministic_provider(self, tmp_path):
        """create_deterministic_provider() should work with defaults"""
        fake = FakeProvider([make_text_response("Test", cost_usd=0.05)])

        deterministic = create_deterministic_provider(
            provider=fake,
            mode="record",
            cache_dir=str(tmp_path / "cache"),
            cache_name="test.db"
        )

        response = await deterministic.chat(
            messages=[ChatMessage(role=Role.user, content="Test")],
            budget=Budget(max_usd=0.50)
        )

        assert response.text == "Test"

    @pytest.mark.asyncio
    async def test_create_golden_provider(self, tmp_path):
        """create_golden_provider() should create golden mode provider"""
        # First create golden file
        golden_path = tmp_path / "golden.json"
        fake_record = FakeProvider([make_text_response("Golden", cost_usd=0.05)])

        deterministic_record = DeterministicLLM.wrap(
            provider=fake_record,
            mode="record",
            cache_path=golden_path
        )

        messages = [ChatMessage(role=Role.user, content="Test")]
        await deterministic_record.chat(messages=messages, budget=Budget(max_usd=0.50))

        # Now use golden provider
        fake_golden = FakeProvider([make_text_response("Should not use")])
        golden_provider = create_golden_provider(
            provider=fake_golden,
            golden_path=golden_path
        )

        response = await golden_provider.chat(messages=messages, budget=Budget(max_usd=0.50))

        assert response.text == "Golden"
        assert fake_golden.get_call_count() == 0


class TestCacheClearReset:
    """Test cache clearing and statistics reset"""

    @pytest.mark.asyncio
    async def test_clear_cache_removes_entries(self, tmp_path):
        """clear_cache() should remove all cached responses"""
        fake = FakeProvider([
            make_text_response("First", cost_usd=0.05),
            make_text_response("Second", cost_usd=0.05),
        ])
        cache_path = tmp_path / "cache.db"

        deterministic = DeterministicLLM.wrap(
            provider=fake,
            mode="record",
            cache_path=cache_path
        )

        messages = [ChatMessage(role=Role.user, content="Test")]

        # Cache a response
        await deterministic.chat(messages=messages, budget=Budget(max_usd=0.50))

        # Clear cache
        deterministic.clear_cache()

        # Next call should miss cache (call provider again)
        await deterministic.chat(messages=messages, budget=Budget(max_usd=0.50))

        # Should have called provider twice
        assert fake.get_call_count() == 2

    @pytest.mark.asyncio
    async def test_clear_cache_resets_stats(self, tmp_path):
        """clear_cache() should reset statistics"""
        fake = FakeProvider([make_text_response("Response", cost_usd=0.05)])
        cache_path = tmp_path / "cache.db"

        deterministic = DeterministicLLM.wrap(
            provider=fake,
            mode="record",
            cache_path=cache_path
        )

        messages = [ChatMessage(role=Role.user, content="Test")]

        # Make some calls
        await deterministic.chat(messages=messages, budget=Budget(max_usd=0.50))
        await deterministic.chat(messages=messages, budget=Budget(max_usd=0.50))

        # Clear
        deterministic.clear_cache()

        # Stats should be reset
        stats = deterministic.stats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.total_requests == 0
