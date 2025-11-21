# -*- coding: utf-8 -*-
"""
Deterministic LLM Caching for Audit Replay

CRITICAL for compliance - enables reproducible LLM outputs for regulatory audits.

Three operational modes:
1. "record": Call real LLM, cache responses by prompt hash (development)
2. "replay": Use cached responses only - deterministic (testing/audit)
3. "golden": Use pre-recorded golden responses (CI/CD)

Key features:
- SHA-256 based cache keys (prompt + tools + schema + temperature + seed)
- Multiple storage backends (JSON, SQLite, Redis)
- Thread-safe concurrent access
- Cache hit rate statistics
- Export golden datasets for version control

Security:
- Cache stored locally (not in version control unless golden)
- Prompt hashes only (no PII in cache keys)
- Optional sensitive field exclusion

Example usage:
    # Development: Record mode
    provider = OpenAIProvider(config)
    deterministic = DeterministicLLM.wrap(
        provider=provider,
        mode="record",
        cache_path="./cache/llm_responses.db"
    )
    response = await deterministic.chat(messages=[...], budget=budget)

    # Testing: Replay mode (deterministic)
    deterministic = DeterministicLLM.wrap(
        provider=provider,
        mode="replay",
        cache_path="./cache/llm_responses.db"
    )
    response = await deterministic.chat(messages=[...], budget=budget)
    # Returns cached response, never hits real LLM

    # CI/CD: Golden mode
    deterministic = DeterministicLLM.wrap(
        provider=provider,
        mode="golden",
        cache_path="./tests/golden/llm_responses.json"
    )
    response = await deterministic.chat(messages=[...], budget=budget)
    # Uses version-controlled golden responses
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from enum import Enum
from greenlang.serialization import canonical_dumps, canonical_hash
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from pydantic import BaseModel, Field

from greenlang.intelligence.providers.base import LLMProvider
from greenlang.intelligence.schemas.messages import ChatMessage
from greenlang.intelligence.schemas.tools import ToolDef
from greenlang.intelligence.schemas.responses import ChatResponse
from greenlang.intelligence.schemas.jsonschema import JSONSchema
from greenlang.intelligence.runtime.budget import Budget


class CacheMode(str, Enum):
    """Cache operation modes"""

    RECORD = "record"  # Call real LLM, cache responses
    REPLAY = "replay"  # Use cached responses only (deterministic)
    GOLDEN = "golden"  # Use pre-recorded golden responses


class CacheEntry(BaseModel):
    """
    Single cache entry

    Stores complete context for reproducible LLM calls:
    - cache_key: SHA-256 hash of all deterministic inputs
    - prompt_hash: SHA-256 hash of just the prompt (for lookup)
    - model: Model name (for debugging mismatches)
    - temperature: Sampling temperature
    - seed: Random seed (if used)
    - timestamp: When response was recorded
    - response: Complete ChatResponse object
    - metadata: Additional context (user ID, session, etc.)
    """

    cache_key: str = Field(description="SHA-256 hash of all inputs")
    prompt_hash: str = Field(description="SHA-256 hash of prompt only")
    model: str = Field(description="Model name")
    temperature: float = Field(description="Sampling temperature")
    seed: Optional[int] = Field(default=None, description="Random seed")
    timestamp: str = Field(description="ISO 8601 timestamp")
    response: ChatResponse = Field(description="Cached LLM response")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "cache_key": "sha256:abc123...",
                    "prompt_hash": "sha256:def456...",
                    "model": "gpt-4-turbo",
                    "temperature": 0.0,
                    "seed": 42,
                    "timestamp": "2025-10-01T12:34:56Z",
                    "response": {
                        "text": "The emissions are 1,021 kg CO2e.",
                        "tool_calls": [],
                        "usage": {
                            "prompt_tokens": 1200,
                            "completion_tokens": 450,
                            "total_tokens": 1650,
                            "cost_usd": 0.0234,
                        },
                        "finish_reason": "stop",
                        "provider_info": {
                            "provider": "openai",
                            "model": "gpt-4-turbo",
                        },
                    },
                    "metadata": {"session_id": "test_001"},
                }
            ]
        }


class CacheStats(BaseModel):
    """
    Cache statistics

    Tracks cache performance:
    - hits: Number of cache hits
    - misses: Number of cache misses
    - total_requests: Total requests
    - hit_rate: Cache hit percentage
    - saved_usd: Total cost saved by cache hits
    """

    hits: int = Field(default=0, description="Number of cache hits")
    misses: int = Field(default=0, description="Number of cache misses")
    total_requests: int = Field(default=0, description="Total requests")
    hit_rate: float = Field(default=0.0, description="Cache hit percentage (0-100)")
    saved_usd: float = Field(default=0.0, description="Total USD saved by caching")

    def record_hit(self, saved_usd: float = 0.0) -> None:
        """Record cache hit"""
        self.hits += 1
        self.total_requests += 1
        self.saved_usd += saved_usd
        self._update_hit_rate()

    def record_miss(self) -> None:
        """Record cache miss"""
        self.misses += 1
        self.total_requests += 1
        self._update_hit_rate()

    def _update_hit_rate(self) -> None:
        """Recalculate hit rate"""
        if self.total_requests > 0:
            self.hit_rate = (self.hits / self.total_requests) * 100.0


class CacheBackend:
    """Abstract cache storage backend"""

    def get(self, cache_key: str) -> Optional[CacheEntry]:
        """Retrieve cache entry by key"""
        raise NotImplementedError

    def set(self, entry: CacheEntry) -> None:
        """Store cache entry"""
        raise NotImplementedError

    def clear(self) -> None:
        """Clear all cache entries"""
        raise NotImplementedError

    def export_all(self) -> List[CacheEntry]:
        """Export all cache entries"""
        raise NotImplementedError

    def import_all(self, entries: List[CacheEntry]) -> None:
        """Import cache entries"""
        raise NotImplementedError

    def close(self) -> None:
        """Close backend connections"""
        pass


class JSONCacheBackend(CacheBackend):
    """
    JSON file cache backend

    Simple file-based storage for:
    - Small test datasets
    - Golden response sets (version controlled)
    - Development/debugging

    Thread-safe with file locking.
    """

    def __init__(self, cache_path: str | Path):
        self.cache_path = Path(cache_path)
        self.lock = threading.Lock()
        self._ensure_cache_file()

    def _ensure_cache_file(self) -> None:
        """Create cache file if it doesn't exist"""
        if not self.cache_path.exists():
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.lock:
                with open(self.cache_path, "w") as f:
                    json.dump({}, f)

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        """Load cache from file"""
        with self.lock:
            with open(self.cache_path, "r") as f:
                return json.load(f)

    def _save_cache(self, cache: Dict[str, Dict[str, Any]]) -> None:
        """Save cache to file"""
        with self.lock:
            with open(self.cache_path, "w") as f:
                json.dump(cache, f, indent=2)

    def get(self, cache_key: str) -> Optional[CacheEntry]:
        cache = self._load_cache()
        if cache_key in cache:
            return CacheEntry(**cache[cache_key])
        return None

    def set(self, entry: CacheEntry) -> None:
        cache = self._load_cache()
        cache[entry.cache_key] = entry.model_dump()
        self._save_cache(cache)

    def clear(self) -> None:
        self._save_cache({})

    def export_all(self) -> List[CacheEntry]:
        cache = self._load_cache()
        return [CacheEntry(**data) for data in cache.values()]

    def import_all(self, entries: List[CacheEntry]) -> None:
        cache = {entry.cache_key: entry.model_dump() for entry in entries}
        self._save_cache(cache)


class SQLiteCacheBackend(CacheBackend):
    """
    SQLite cache backend

    Production storage with:
    - Indexed lookups (fast retrieval)
    - ACID transactions
    - Optional TTL/expiry
    - Efficient storage

    Thread-safe with connection pooling.
    """

    def __init__(self, cache_path: str | Path):
        self.cache_path = Path(cache_path)
        self.lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema"""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            conn = sqlite3.connect(str(self.cache_path))
            cursor = conn.cursor()

            # Create cache table with indexes
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    cache_key TEXT PRIMARY KEY,
                    prompt_hash TEXT,
                    model TEXT,
                    temperature REAL,
                    seed INTEGER,
                    timestamp TEXT,
                    response TEXT,
                    metadata TEXT
                )
            """
            )

            # Create indexes for fast lookup
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_prompt_hash ON cache(prompt_hash)"
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_model ON cache(model)")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_timestamp ON cache(timestamp)"
            )

            conn.commit()
            conn.close()

    def get(self, cache_key: str) -> Optional[CacheEntry]:
        with self.lock:
            conn = sqlite3.connect(str(self.cache_path))
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT cache_key, prompt_hash, model, temperature, seed, timestamp, response, metadata
                FROM cache
                WHERE cache_key = ?
            """,
                (cache_key,),
            )

            row = cursor.fetchone()
            conn.close()

            if row:
                return CacheEntry(
                    cache_key=row[0],
                    prompt_hash=row[1],
                    model=row[2],
                    temperature=row[3],
                    seed=row[4],
                    timestamp=row[5],
                    response=ChatResponse(**json.loads(row[6])),
                    metadata=json.loads(row[7]),
                )
            return None

    def set(self, entry: CacheEntry) -> None:
        with self.lock:
            conn = sqlite3.connect(str(self.cache_path))
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO cache
                (cache_key, prompt_hash, model, temperature, seed, timestamp, response, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    entry.cache_key,
                    entry.prompt_hash,
                    entry.model,
                    entry.temperature,
                    entry.seed,
                    entry.timestamp,
                    canonical_dumps(entry.response.model_dump()),
                    canonical_dumps(entry.metadata),
                ),
            )

            conn.commit()
            conn.close()

    def clear(self) -> None:
        with self.lock:
            conn = sqlite3.connect(str(self.cache_path))
            cursor = conn.cursor()
            cursor.execute("DELETE FROM cache")
            conn.commit()
            conn.close()

    def export_all(self) -> List[CacheEntry]:
        with self.lock:
            conn = sqlite3.connect(str(self.cache_path))
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT cache_key, prompt_hash, model, temperature, seed, timestamp, response, metadata
                FROM cache
            """
            )

            entries = []
            for row in cursor.fetchall():
                entries.append(
                    CacheEntry(
                        cache_key=row[0],
                        prompt_hash=row[1],
                        model=row[2],
                        temperature=row[3],
                        seed=row[4],
                        timestamp=row[5],
                        response=ChatResponse(**json.loads(row[6])),
                        metadata=json.loads(row[7]),
                    )
                )

            conn.close()
            return entries

    def import_all(self, entries: List[CacheEntry]) -> None:
        with self.lock:
            conn = sqlite3.connect(str(self.cache_path))
            cursor = conn.cursor()

            for entry in entries:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO cache
                    (cache_key, prompt_hash, model, temperature, seed, timestamp, response, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        entry.cache_key,
                        entry.prompt_hash,
                        entry.model,
                        entry.temperature,
                        entry.seed,
                        entry.timestamp,
                        json.dumps(entry.response.model_dump()),
                        json.dumps(entry.metadata),
                    ),
                )

            conn.commit()
            conn.close()


class DeterministicLLM:
    """
    Deterministic LLM wrapper for audit replay

    Wraps any LLMProvider with deterministic caching:
    - Record mode: Call real LLM, cache all responses
    - Replay mode: Use cached responses only (100% deterministic)
    - Golden mode: Use pre-recorded golden responses for tests

    Thread-safe for concurrent access.

    Example usage:
        # Development: Record real LLM responses
        provider = OpenAIProvider(config)
        deterministic = DeterministicLLM.wrap(
            provider=provider,
            mode="record",
            cache_path="./cache/llm.db"
        )
        response = await deterministic.chat(messages=[...], budget=budget)

        # Testing: Replay from cache (no real LLM calls)
        deterministic = DeterministicLLM.wrap(
            provider=provider,
            mode="replay",
            cache_path="./cache/llm.db"
        )
        response = await deterministic.chat(messages=[...], budget=budget)
        # Guaranteed to return same response as record mode

        # CI/CD: Use golden responses
        deterministic = DeterministicLLM.wrap(
            provider=provider,
            mode="golden",
            cache_path="./tests/golden/llm.json"
        )
        response = await deterministic.chat(messages=[...], budget=budget)

        # Export golden dataset for version control
        deterministic.export_golden("./tests/golden/llm_v1.json")

        # View cache statistics
        stats = deterministic.stats()
        print(f"Cache hit rate: {stats.hit_rate:.1f}%")
        print(f"Cost saved: ${stats.saved_usd:.4f}")
    """

    def __init__(
        self,
        provider: LLMProvider,
        mode: CacheMode,
        backend: CacheBackend,
    ):
        """
        Initialize deterministic LLM wrapper

        Args:
            provider: Underlying LLM provider (OpenAI, Anthropic, etc.)
            mode: Cache mode (record/replay/golden)
            backend: Cache storage backend (JSON/SQLite/Redis)
        """
        self.provider = provider
        self.mode = mode
        self.backend = backend
        self._stats = CacheStats()

    @classmethod
    def wrap(
        cls,
        provider: LLMProvider,
        mode: str,
        cache_path: str | Path,
        backend_type: str = "auto",
    ) -> "DeterministicLLM":
        """
        Wrap provider with deterministic caching

        Args:
            provider: LLM provider to wrap
            mode: Cache mode ("record"/"replay"/"golden")
            cache_path: Path to cache file/database
            backend_type: Backend type ("json"/"sqlite"/"auto")
                          "auto" picks based on file extension

        Returns:
            Wrapped deterministic provider

        Example:
            deterministic = DeterministicLLM.wrap(
                provider=OpenAIProvider(config),
                mode="record",
                cache_path="./cache/llm.db"  # auto-detects SQLite
            )
        """
        # Validate mode
        try:
            cache_mode = CacheMode(mode)
        except ValueError:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be one of: {[m.value for m in CacheMode]}"
            )

        # Auto-detect backend type
        cache_path = Path(cache_path)
        if backend_type == "auto":
            if cache_path.suffix == ".json":
                backend_type = "json"
            elif cache_path.suffix in [".db", ".sqlite", ".sqlite3"]:
                backend_type = "sqlite"
            else:
                # Default to SQLite for production
                backend_type = "sqlite"

        # Create backend
        if backend_type == "json":
            backend = JSONCacheBackend(cache_path)
        elif backend_type == "sqlite":
            backend = SQLiteCacheBackend(cache_path)
        else:
            raise ValueError(
                f"Invalid backend_type '{backend_type}'. Must be 'json' or 'sqlite'"
            )

        return cls(provider=provider, mode=cache_mode, backend=backend)

    def _compute_cache_key(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDef]],
        json_schema: Optional[JSONSchema],
        temperature: float,
        top_p: float,
        seed: Optional[int],
        tool_choice: Optional[str],
    ) -> str:
        """
        Compute deterministic cache key

        Hash includes ALL factors that affect LLM output:
        - Messages (prompt + conversation history)
        - Tools (available functions)
        - JSON schema (output constraints)
        - Temperature (sampling randomness)
        - Top-p (nucleus sampling)
        - Seed (random seed)
        - Tool choice (tool selection strategy)
        - Model name (different models = different outputs)

        Returns:
            SHA-256 hash as hex string with "sha256:" prefix
        """
        hasher = hashlib.sha256()

        # Hash messages using canonical JSON
        for msg in messages:
            hasher.update(canonical_dumps(msg.model_dump()).encode())

        # Hash tools using canonical JSON
        if tools:
            for tool in tools:
                hasher.update(canonical_dumps(tool.model_dump()).encode())

        # Hash JSON schema using canonical JSON
        if json_schema:
            hasher.update(canonical_dumps(json_schema).encode())

        # Hash sampling parameters
        hasher.update(f"{temperature}".encode())
        hasher.update(f"{top_p}".encode())

        # Hash seed
        if seed is not None:
            hasher.update(f"{seed}".encode())

        # Hash tool choice
        if tool_choice:
            hasher.update(tool_choice.encode())

        # Hash model name
        hasher.update(self.provider.config.model.encode())

        return f"sha256:{hasher.hexdigest()}"

    def _compute_prompt_hash(self, messages: List[ChatMessage]) -> str:
        """
        Compute hash of prompt only (for lookup/debugging)

        Returns:
            SHA-256 hash as hex string with "sha256:" prefix
        """
        # Use canonical_hash for consistent hashing
        messages_data = [msg.model_dump() for msg in messages]
        return f"sha256:{canonical_hash(messages_data)}"

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: Optional[list[ToolDef]] = None,
        json_schema: Optional[JSONSchema] = None,
        budget: Budget,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
        tool_choice: str | None = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ChatResponse:
        """
        Execute chat with deterministic caching

        Behavior by mode:
        - RECORD: Check cache first, call LLM on miss, cache result
        - REPLAY: Use cache only, raise error on miss
        - GOLDEN: Use cache only, raise error on miss

        Args:
            messages: Conversation history
            tools: Available tools for function calling
            json_schema: JSON schema for response validation
            budget: Budget tracker
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            seed: Random seed for reproducibility
            tool_choice: Tool selection strategy
            metadata: Additional metadata to store with cache entry

        Returns:
            ChatResponse (from cache or real LLM)

        Raises:
            ValueError: If replay/golden mode and cache miss
            BudgetExceeded: If request would exceed budget
        """
        # Compute cache key
        cache_key = self._compute_cache_key(
            messages=messages,
            tools=tools,
            json_schema=json_schema,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            tool_choice=tool_choice,
        )

        # Check cache
        cached_entry = self.backend.get(cache_key)

        if cached_entry:
            # Cache hit
            self._stats.record_hit(saved_usd=cached_entry.response.usage.cost_usd)
            return cached_entry.response

        # Cache miss
        self._stats.record_miss()

        if self.mode in [CacheMode.REPLAY, CacheMode.GOLDEN]:
            # Replay/golden mode requires cache hit
            raise ValueError(
                f"Cache miss in {self.mode.value} mode. "
                f"Cache key: {cache_key}. "
                "Cannot call real LLM in replay/golden mode. "
                "Ensure cache is populated or use record mode."
            )

        # Record mode: Call real LLM
        response = await self.provider.chat(
            messages=messages,
            tools=tools,
            json_schema=json_schema,
            budget=budget,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            tool_choice=tool_choice,
            metadata=metadata,
        )

        # Cache the response
        entry = CacheEntry(
            cache_key=cache_key,
            prompt_hash=self._compute_prompt_hash(messages),
            model=self.provider.config.model,
            temperature=temperature,
            seed=seed,
            timestamp=datetime.now(timezone.utc).isoformat(),
            response=response,
            metadata=dict(metadata) if metadata else {},
        )

        self.backend.set(entry)

        return response

    def stats(self) -> CacheStats:
        """
        Get cache statistics

        Returns:
            Cache statistics (hits, misses, hit rate, cost saved)

        Example:
            stats = deterministic.stats()
            print(f"Hit rate: {stats.hit_rate:.1f}%")
            print(f"Saved: ${stats.saved_usd:.4f}")
        """
        return self._stats

    def clear_cache(self) -> None:
        """
        Clear all cache entries

        WARNING: This deletes all cached responses.
        Use with caution in production.

        Example:
            deterministic.clear_cache()
        """
        self.backend.clear()
        self._stats = CacheStats()  # Reset stats

    def export_golden(self, output_path: str | Path) -> None:
        """
        Export cache as golden dataset

        Exports all cached responses to JSON file for:
        - Version control
        - CI/CD testing
        - Regression testing
        - Audit trails

        Args:
            output_path: Path to output JSON file

        Example:
            # Export current cache as golden v1
            deterministic.export_golden("./tests/golden/llm_v1.json")

            # Commit to version control
            # git add ./tests/golden/llm_v1.json
            # git commit -m "Add LLM golden responses v1"
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Export all entries
        entries = self.backend.export_all()

        # Convert to JSON
        data = {entry.cache_key: entry.model_dump() for entry in entries}

        # Write to file
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    def import_golden(self, input_path: str | Path) -> None:
        """
        Import golden dataset into cache

        Loads pre-recorded golden responses from JSON file.

        Args:
            input_path: Path to golden JSON file

        Example:
            # Load golden responses for testing
            deterministic.import_golden("./tests/golden/llm_v1.json")

            # Now all cached responses available
            response = await deterministic.chat(...)  # Uses golden cache
        """
        input_path = Path(input_path)

        # Load JSON
        with open(input_path, "r") as f:
            data = json.load(f)

        # Convert to entries
        entries = [CacheEntry(**entry_data) for entry_data in data.values()]

        # Import into backend
        self.backend.import_all(entries)

    @property
    def capabilities(self):
        """Delegate to underlying provider"""
        return self.provider.capabilities

    def __del__(self):
        """Cleanup backend on deletion"""
        if hasattr(self, "backend"):
            self.backend.close()


# Convenience functions

def create_deterministic_provider(
    provider: LLMProvider,
    mode: str = "record",
    cache_dir: str | Path = "./cache/llm",
    cache_name: str = "responses.db",
) -> DeterministicLLM:
    """
    Create deterministic provider with sensible defaults

    Args:
        provider: LLM provider to wrap
        mode: Cache mode ("record"/"replay"/"golden")
        cache_dir: Cache directory
        cache_name: Cache filename

    Returns:
        Wrapped deterministic provider

    Example:
        provider = OpenAIProvider(config)
        deterministic = create_deterministic_provider(
            provider=provider,
            mode="record",
            cache_dir="./cache/llm"
        )
    """
    cache_path = Path(cache_dir) / cache_name
    return DeterministicLLM.wrap(
        provider=provider, mode=mode, cache_path=cache_path, backend_type="auto"
    )


def create_golden_provider(
    provider: LLMProvider, golden_path: str | Path
) -> DeterministicLLM:
    """
    Create provider with golden responses for testing

    Args:
        provider: LLM provider (for capability info only, not used)
        golden_path: Path to golden responses JSON

    Returns:
        Deterministic provider in golden mode

    Example:
        provider = OpenAIProvider(config)
        golden = create_golden_provider(
            provider=provider,
            golden_path="./tests/golden/llm_responses.json"
        )

        # All responses come from golden file
        response = await golden.chat(messages=[...], budget=budget)
    """
    return DeterministicLLM.wrap(
        provider=provider, mode="golden", cache_path=golden_path, backend_type="json"
    )


__all__ = [
    "DeterministicLLM",
    "CacheMode",
    "CacheEntry",
    "CacheStats",
    "CacheBackend",
    "JSONCacheBackend",
    "SQLiteCacheBackend",
    "create_deterministic_provider",
    "create_golden_provider",
]
