"""
Deterministic RAG wrapper for replay mode and caching.

CRITICAL: This module enforces deterministic behavior by:
1. Caching retrieval results keyed by query hash
2. Network isolation in replay mode
3. Snapshot-based vector stores for frozen indices
4. Hash verification for cache integrity

Modes:
- replay: Return cached results, block network, enforce determinism
- record: Perform live searches and cache results for future replay
- live: Normal operation without caching
"""

import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime

from greenlang.intelligence.rag.models import QueryResult, Chunk, RAGCitation
from greenlang.intelligence.rag.hashing import query_hash, canonicalize_text
from greenlang.intelligence.rag.config import RAGConfig


class DeterministicRAG:
    """
    Deterministic RAG wrapper with replay mode and caching.

    This class ensures that queries return identical results across runs
    by caching query results and enforcing network isolation in replay mode.

    Cache Format:
        {
            "version": "1.0",
            "created_at": "2025-10-03T14:00:00Z",
            "queries": {
                "query_hash_1": {
                    "query": "original query string",
                    "params": {"k": 6, "collections": [...]},
                    "result": QueryResult.dict(),
                    "cached_at": "2025-10-03T14:00:00Z"
                },
                ...
            }
        }
    """

    def __init__(
        self,
        mode: Literal["replay", "record", "live"] = "replay",
        cache_path: Optional[Path] = None,
        config: Optional[RAGConfig] = None,
    ):
        """
        Initialize deterministic RAG wrapper.

        Args:
            mode: Execution mode
                - replay: Use cached results, block network
                - record: Perform searches and cache results
                - live: Normal operation without caching
            cache_path: Path to cache file (default: .rag_cache.json)
            config: RAG configuration
        """
        self.mode = mode
        self.config = config
        self.cache_path = cache_path or Path(".rag_cache.json")
        self.cache: Dict[str, Any] = {
            "version": "1.0",
            "created_at": datetime.utcnow().isoformat(),
            "queries": {},
        }

        # Load cache in replay/record mode
        if self.mode in ("replay", "record"):
            self._load_cache()

        # Enforce network isolation in replay mode
        if self.mode == "replay":
            self._enforce_network_isolation()

    def _load_cache(self) -> None:
        """
        Load cache from disk.

        Raises:
            FileNotFoundError: If cache file not found in replay mode
            ValueError: If cache format is invalid
        """
        if not self.cache_path.exists():
            if self.mode == "replay":
                raise FileNotFoundError(
                    f"Cache file not found: {self.cache_path}. "
                    "Run in 'record' mode first to build cache."
                )
            # In record mode, start with empty cache
            return

        # Check if file is empty
        if self.cache_path.stat().st_size == 0:
            if self.mode == "replay":
                raise ValueError("Cache file is empty - cannot use replay mode")
            # In record mode, start with empty cache
            return

        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            # Validate cache format
            if "version" not in loaded or "queries" not in loaded:
                raise ValueError("Invalid cache format: missing 'version' or 'queries'")

            # Migrate cache if needed (future compatibility)
            if loaded["version"] != "1.0":
                loaded = self._migrate_cache(loaded)

            self.cache = loaded

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse cache file: {e}")

    def _save_cache(self) -> None:
        """Save cache to disk."""
        # Update timestamp
        self.cache["updated_at"] = datetime.utcnow().isoformat()

        # Ensure directory exists
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Write atomically (write to temp file, then rename)
        temp_path = self.cache_path.with_suffix(".tmp")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2, default=str)

            # Atomic rename
            temp_path.replace(self.cache_path)
        except Exception as e:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise RuntimeError(f"Failed to save cache: {e}")

    def _enforce_network_isolation(self) -> None:
        """
        Enforce network isolation in replay mode.

        Sets environment variables to block network access for:
        - HuggingFace transformers
        - HuggingFace datasets
        - Python random seed (for numpy/torch)

        CRITICAL: This prevents non-determinism from network-fetched models or data.
        """
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["PYTHONHASHSEED"] = "42"

        # Set numpy random seed if available
        try:
            import numpy as np
            np.random.seed(42)
        except ImportError:
            pass

        # Set torch random seed if available
        try:
            import torch
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)
        except ImportError:
            pass

    def _migrate_cache(self, old_cache: Dict) -> Dict:
        """
        Migrate cache from old version to current version.

        Args:
            old_cache: Cache in old format

        Returns:
            Cache in current format
        """
        # For now, just return as-is (no migrations needed yet)
        # In the future, handle version-specific migrations
        return old_cache

    def search(
        self,
        query: str,
        k: int,
        collections: List[str],
        fetch_k: int = 30,
        mmr_lambda: float = 0.5,
        engine: Optional[Any] = None,
    ) -> QueryResult:
        """
        Search with deterministic caching.

        Args:
            query: Query string
            k: Number of results to return (top_k)
            collections: Collections to search
            fetch_k: Number of candidates for MMR
            mmr_lambda: MMR lambda parameter
            engine: RAG engine instance (required for live/record mode)

        Returns:
            QueryResult with chunks and citations

        Raises:
            RuntimeError: If no cached result in replay mode
            ValueError: If engine not provided in live/record mode
        """
        # Compute query hash
        params = {
            "k": k,
            "collections": sorted(collections),  # Sort for determinism
            "fetch_k": fetch_k,
            "mmr_lambda": mmr_lambda,
        }
        qhash = query_hash(query, params)

        # Replay mode: return cached result
        if self.mode == "replay":
            if qhash not in self.cache["queries"]:
                raise RuntimeError(
                    f"No cached result for query hash: {qhash}\n"
                    f"Query: {query}\n"
                    f"Params: {params}\n"
                    "Run in 'record' mode first to cache this query."
                )

            cached_entry = self.cache["queries"][qhash]

            # Reconstruct QueryResult from cached dict
            result = QueryResult(**cached_entry["result"])

            return result

        # Live/record mode: perform actual search
        if engine is None:
            raise ValueError("Engine must be provided for live/record mode")

        # Delegate to actual RAG engine
        result = engine._real_search(
            query=query,
            top_k=k,
            collections=collections,
            fetch_k=fetch_k,
            mmr_lambda=mmr_lambda,
        )

        # Record mode: cache result
        if self.mode == "record":
            self.cache["queries"][qhash] = {
                "query": query,
                "params": params,
                "result": result.dict(),
                "cached_at": datetime.utcnow().isoformat(),
            }
            self._save_cache()

        return result

    def clear_cache(self) -> None:
        """Clear all cached queries."""
        self.cache["queries"] = {}
        self.cache["cleared_at"] = datetime.utcnow().isoformat()
        if self.mode in ("replay", "record"):
            self._save_cache()

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache stats (num_queries, cache_size_bytes, etc.)
        """
        cache_size = 0
        if self.cache_path.exists():
            cache_size = self.cache_path.stat().st_size

        return {
            "mode": self.mode,
            "cache_path": str(self.cache_path),
            "num_queries": len(self.cache["queries"]),
            "cache_size_bytes": cache_size,
            "created_at": self.cache.get("created_at"),
            "updated_at": self.cache.get("updated_at"),
        }

    def export_cache(self, output_path: Path) -> None:
        """
        Export cache to a different file.

        Args:
            output_path: Path to export cache
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2, default=str)

    def import_cache(self, input_path: Path, merge: bool = False) -> None:
        """
        Import cache from a file.

        Args:
            input_path: Path to cache file
            merge: If True, merge with existing cache; else replace
        """
        with open(input_path, "r", encoding="utf-8") as f:
            imported = json.load(f)

        if merge:
            # Merge queries (imported queries override existing)
            self.cache["queries"].update(imported["queries"])
        else:
            # Replace entire cache
            self.cache = imported

        if self.mode in ("replay", "record"):
            self._save_cache()

    def verify_cache_integrity(self) -> Dict[str, Any]:
        """
        Verify cache integrity by checking query hashes.

        Returns:
            Dict with verification results
        """
        errors = []
        warnings = []

        for qhash, entry in self.cache["queries"].items():
            # Recompute hash and verify
            recomputed_hash = query_hash(entry["query"], entry["params"])

            if recomputed_hash != qhash:
                errors.append({
                    "query": entry["query"],
                    "stored_hash": qhash,
                    "recomputed_hash": recomputed_hash,
                    "error": "Hash mismatch",
                })

            # Verify result structure
            try:
                QueryResult(**entry["result"])
            except Exception as e:
                errors.append({
                    "query": entry["query"],
                    "hash": qhash,
                    "error": f"Invalid result structure: {e}",
                })

        return {
            "valid": len(errors) == 0,
            "num_queries": len(self.cache["queries"]),
            "num_errors": len(errors),
            "num_warnings": len(warnings),
            "errors": errors,
            "warnings": warnings,
        }
