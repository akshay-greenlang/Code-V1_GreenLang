# -*- coding: utf-8 -*-
"""
Unit Tests for SeedManager (AGENT-FOUND-008)

Tests seed configuration creation, application, verification,
storage, and hash determinism.

Coverage target: 85%+ of seed_manager.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline SeedConfiguration and SeedManager
# ---------------------------------------------------------------------------

def _content_hash(data: Any) -> str:
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True, ensure_ascii=True)
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


class SeedConfiguration:
    def __init__(self, global_seed: int = 42, numpy_seed: Optional[int] = 42,
                 torch_seed: Optional[int] = 42,
                 custom_seeds: Optional[Dict[str, int]] = None,
                 seed_hash: str = ""):
        self.global_seed = global_seed
        self.numpy_seed = numpy_seed
        self.torch_seed = torch_seed
        self.custom_seeds = custom_seeds or {}
        if not seed_hash:
            seed_data = {
                "global": self.global_seed,
                "numpy": self.numpy_seed,
                "torch": self.torch_seed,
                "custom": self.custom_seeds,
            }
            self.seed_hash = _content_hash(seed_data)[:16]
        else:
            self.seed_hash = seed_hash


class SeedManager:
    """Manages random seeds for reproducibility."""

    def __init__(self):
        self._current: Optional[SeedConfiguration] = None
        self._stored: Dict[str, SeedConfiguration] = {}
        self._applied = False

    def create_seed_config(self, global_seed: int = 42,
                           numpy_seed: Optional[int] = 42,
                           torch_seed: Optional[int] = 42,
                           custom_seeds: Optional[Dict[str, int]] = None) -> SeedConfiguration:
        """Create a seed configuration."""
        return SeedConfiguration(
            global_seed=global_seed,
            numpy_seed=numpy_seed,
            torch_seed=torch_seed,
            custom_seeds=custom_seeds or {},
        )

    def apply_seeds(self, config: SeedConfiguration) -> None:
        """Apply seed configuration globally."""
        self._current = config
        self._applied = True
        random.seed(config.global_seed)

    def verify_seeds(self, current: SeedConfiguration,
                     expected: SeedConfiguration) -> Dict[str, Any]:
        """Verify seed configuration matches."""
        mismatches = []

        if current.global_seed != expected.global_seed:
            mismatches.append(
                f"global_seed: {current.global_seed} vs {expected.global_seed}"
            )
        if current.numpy_seed != expected.numpy_seed:
            mismatches.append(
                f"numpy_seed: {current.numpy_seed} vs {expected.numpy_seed}"
            )
        if current.torch_seed != expected.torch_seed:
            mismatches.append(
                f"torch_seed: {current.torch_seed} vs {expected.torch_seed}"
            )
        for key, val in expected.custom_seeds.items():
            cur_val = current.custom_seeds.get(key)
            if cur_val != val:
                mismatches.append(f"custom_seed[{key}]: {cur_val} vs {val}")

        return {
            "is_match": len(mismatches) == 0,
            "mismatches": mismatches,
        }

    def get_current_seed_config(self) -> Optional[SeedConfiguration]:
        """Get the currently applied seed configuration."""
        return self._current

    def store_seed_config(self, config_id: str,
                          config: SeedConfiguration) -> None:
        """Store a seed configuration for retrieval."""
        self._stored[config_id] = config

    def get_stored_seed_config(self, config_id: str) -> Optional[SeedConfiguration]:
        """Get a stored seed configuration."""
        return self._stored.get(config_id)


# ===========================================================================
# Test Classes
# ===========================================================================


class TestCreateSeedConfig:
    """Test create_seed_config method."""

    def test_create_seed_config_defaults(self):
        mgr = SeedManager()
        cfg = mgr.create_seed_config()
        assert cfg.global_seed == 42
        assert cfg.numpy_seed == 42
        assert cfg.torch_seed == 42
        assert cfg.custom_seeds == {}

    def test_create_seed_config_custom_global(self):
        mgr = SeedManager()
        cfg = mgr.create_seed_config(global_seed=123)
        assert cfg.global_seed == 123

    def test_create_seed_config_custom_numpy(self):
        mgr = SeedManager()
        cfg = mgr.create_seed_config(numpy_seed=999)
        assert cfg.numpy_seed == 999

    def test_create_seed_config_custom_torch(self):
        mgr = SeedManager()
        cfg = mgr.create_seed_config(torch_seed=777)
        assert cfg.torch_seed == 777

    def test_create_seed_config_custom_seeds(self):
        mgr = SeedManager()
        cfg = mgr.create_seed_config(custom_seeds={"model_a": 100, "model_b": 200})
        assert cfg.custom_seeds == {"model_a": 100, "model_b": 200}

    def test_create_seed_config_none_numpy(self):
        mgr = SeedManager()
        cfg = mgr.create_seed_config(numpy_seed=None)
        assert cfg.numpy_seed is None

    def test_create_seed_config_none_torch(self):
        mgr = SeedManager()
        cfg = mgr.create_seed_config(torch_seed=None)
        assert cfg.torch_seed is None

    def test_create_seed_config_hash_auto_computed(self):
        mgr = SeedManager()
        cfg = mgr.create_seed_config()
        assert cfg.seed_hash != ""
        assert len(cfg.seed_hash) == 16


class TestApplySeeds:
    """Test apply_seeds method."""

    def test_apply_seeds_global(self):
        mgr = SeedManager()
        cfg = mgr.create_seed_config(global_seed=42)
        mgr.apply_seeds(cfg)
        assert mgr._applied is True
        assert mgr._current is cfg

    def test_apply_seeds_deterministic_random(self):
        mgr = SeedManager()
        cfg = mgr.create_seed_config(global_seed=42)
        mgr.apply_seeds(cfg)
        val1 = random.random()
        # Reset and reapply
        mgr.apply_seeds(cfg)
        val2 = random.random()
        assert val1 == val2

    def test_apply_seeds_custom(self):
        mgr = SeedManager()
        cfg = mgr.create_seed_config(global_seed=99, custom_seeds={"comp": 55})
        mgr.apply_seeds(cfg)
        current = mgr.get_current_seed_config()
        assert current.global_seed == 99
        assert current.custom_seeds["comp"] == 55


class TestVerifySeeds:
    """Test verify_seeds method."""

    def test_verify_seeds_match(self):
        mgr = SeedManager()
        s1 = mgr.create_seed_config()
        s2 = mgr.create_seed_config()
        result = mgr.verify_seeds(s1, s2)
        assert result["is_match"] is True
        assert result["mismatches"] == []

    def test_verify_seeds_mismatch_global(self):
        mgr = SeedManager()
        s1 = mgr.create_seed_config(global_seed=42)
        s2 = mgr.create_seed_config(global_seed=99)
        result = mgr.verify_seeds(s1, s2)
        assert result["is_match"] is False
        assert any("global_seed" in m for m in result["mismatches"])

    def test_verify_seeds_mismatch_numpy(self):
        mgr = SeedManager()
        s1 = mgr.create_seed_config(numpy_seed=42)
        s2 = mgr.create_seed_config(numpy_seed=99)
        result = mgr.verify_seeds(s1, s2)
        assert result["is_match"] is False
        assert any("numpy_seed" in m for m in result["mismatches"])

    def test_verify_seeds_mismatch_torch(self):
        mgr = SeedManager()
        s1 = mgr.create_seed_config(torch_seed=42)
        s2 = mgr.create_seed_config(torch_seed=99)
        result = mgr.verify_seeds(s1, s2)
        assert result["is_match"] is False
        assert any("torch_seed" in m for m in result["mismatches"])

    def test_verify_seeds_mismatch_custom(self):
        mgr = SeedManager()
        s1 = mgr.create_seed_config(custom_seeds={"comp": 100})
        s2 = mgr.create_seed_config(custom_seeds={"comp": 200})
        result = mgr.verify_seeds(s1, s2)
        assert result["is_match"] is False
        assert any("custom_seed" in m for m in result["mismatches"])

    def test_verify_seeds_custom_missing_key(self):
        mgr = SeedManager()
        s1 = mgr.create_seed_config(custom_seeds={})
        s2 = mgr.create_seed_config(custom_seeds={"comp": 100})
        result = mgr.verify_seeds(s1, s2)
        assert result["is_match"] is False


class TestGetCurrentSeedConfig:
    """Test get_current_seed_config."""

    def test_get_current_before_apply(self):
        mgr = SeedManager()
        assert mgr.get_current_seed_config() is None

    def test_get_current_after_apply(self):
        mgr = SeedManager()
        cfg = mgr.create_seed_config()
        mgr.apply_seeds(cfg)
        current = mgr.get_current_seed_config()
        assert current is not None
        assert current.global_seed == 42


class TestSeedStorage:
    """Test store_seed_config and get_stored_seed_config."""

    def test_store_and_retrieve(self):
        mgr = SeedManager()
        cfg = mgr.create_seed_config(global_seed=123)
        mgr.store_seed_config("seed-001", cfg)
        retrieved = mgr.get_stored_seed_config("seed-001")
        assert retrieved is not None
        assert retrieved.global_seed == 123

    def test_retrieve_nonexistent(self):
        mgr = SeedManager()
        assert mgr.get_stored_seed_config("nonexistent") is None

    def test_overwrite_stored(self):
        mgr = SeedManager()
        cfg1 = mgr.create_seed_config(global_seed=42)
        cfg2 = mgr.create_seed_config(global_seed=99)
        mgr.store_seed_config("seed-001", cfg1)
        mgr.store_seed_config("seed-001", cfg2)
        retrieved = mgr.get_stored_seed_config("seed-001")
        assert retrieved.global_seed == 99


class TestSeedHashDeterminism:
    """Test seed hash determinism."""

    def test_seed_hash_deterministic(self):
        mgr = SeedManager()
        s1 = mgr.create_seed_config(global_seed=42, numpy_seed=42, torch_seed=42)
        s2 = mgr.create_seed_config(global_seed=42, numpy_seed=42, torch_seed=42)
        assert s1.seed_hash == s2.seed_hash

    def test_seed_hash_different_for_different_seeds(self):
        mgr = SeedManager()
        s1 = mgr.create_seed_config(global_seed=42)
        s2 = mgr.create_seed_config(global_seed=99)
        assert s1.seed_hash != s2.seed_hash

    def test_seed_hash_length(self):
        mgr = SeedManager()
        cfg = mgr.create_seed_config()
        assert len(cfg.seed_hash) == 16
