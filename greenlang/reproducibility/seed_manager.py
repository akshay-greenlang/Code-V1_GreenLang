# -*- coding: utf-8 -*-
"""
Seed Management for Reproducible Execution - AGENT-FOUND-008: Reproducibility Agent

Manages random seed configuration for deterministic execution including
global Python random, NumPy random, PyTorch manual seed, and custom
component seeds.

Zero-Hallucination Guarantees:
    - Seeds are applied deterministically via standard APIs
    - Seed verification uses exact integer comparison
    - No probabilistic seed selection

Example:
    >>> from greenlang.reproducibility.seed_manager import SeedManager
    >>> from greenlang.reproducibility.config import ReproducibilityConfig
    >>> mgr = SeedManager(ReproducibilityConfig())
    >>> cfg = mgr.create_seed_config(global_seed=42)
    >>> mgr.apply_seeds(cfg)
    >>> print(mgr.get_current_seed_config().global_seed)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-008 Reproducibility Agent
Status: Production Ready
"""

from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional

from greenlang.reproducibility.config import ReproducibilityConfig
from greenlang.reproducibility.models import (
    SeedConfiguration,
    VerificationStatus,
    VerificationCheck,
)
from greenlang.reproducibility.metrics import record_non_determinism

logger = logging.getLogger(__name__)


class SeedManager:
    """Seed management for reproducible execution.

    Creates, applies, verifies, and stores seed configurations to
    ensure deterministic random number generation across execution runs.

    Attributes:
        _config: Reproducibility configuration.
        _current_seeds: Currently active seed configuration.
        _seed_history: In-memory store of seed configs by execution ID.

    Example:
        >>> mgr = SeedManager(config)
        >>> cfg = mgr.create_seed_config(global_seed=123)
        >>> mgr.apply_seeds(cfg)
        >>> check = mgr.verify_seeds(mgr.get_current_seed_config(), cfg)
        >>> assert check.status == VerificationStatus.PASS
    """

    def __init__(self, config: ReproducibilityConfig) -> None:
        """Initialize SeedManager.

        Args:
            config: Reproducibility configuration instance.
        """
        self._config = config
        self._current_seeds: Optional[SeedConfiguration] = None
        self._seed_history: Dict[str, SeedConfiguration] = {}
        logger.info("SeedManager initialized")

    def create_seed_config(
        self,
        global_seed: int = 42,
        numpy_seed: Optional[int] = 42,
        torch_seed: Optional[int] = 42,
        custom_seeds: Optional[Dict[str, int]] = None,
    ) -> SeedConfiguration:
        """Create a seed configuration.

        Args:
            global_seed: Global Python random seed.
            numpy_seed: NumPy random seed (None to skip).
            torch_seed: PyTorch manual seed (None to skip).
            custom_seeds: Custom seeds for specific components.

        Returns:
            SeedConfiguration instance.
        """
        seed_config = SeedConfiguration(
            global_seed=global_seed,
            numpy_seed=numpy_seed,
            torch_seed=torch_seed,
            custom_seeds=custom_seeds or {},
        )
        logger.debug(
            "Created seed config: global=%d, numpy=%s, torch=%s, custom=%d",
            global_seed, numpy_seed, torch_seed, len(custom_seeds or {}),
        )
        return seed_config

    def apply_seeds(self, seed_config: SeedConfiguration) -> None:
        """Apply a seed configuration to the current runtime.

        Sets global Python random, NumPy, PyTorch, and custom seeds.

        Args:
            seed_config: Seed configuration to apply.
        """
        self._apply_global_seed(seed_config.global_seed)

        if seed_config.numpy_seed is not None:
            self._apply_numpy_seed(seed_config.numpy_seed)

        if seed_config.torch_seed is not None:
            self._apply_torch_seed(seed_config.torch_seed)

        if seed_config.custom_seeds:
            self._apply_custom_seeds(seed_config.custom_seeds)

        self._current_seeds = seed_config
        logger.info("Applied seed configuration: global=%d", seed_config.global_seed)

    def _apply_global_seed(self, seed: int) -> None:
        """Apply the global Python random seed.

        Args:
            seed: Random seed value.
        """
        random.seed(seed)

        # Also try to use the GreenLang deterministic random utility
        try:
            from greenlang.utilities.determinism.random import set_global_random_seed
            set_global_random_seed(seed)
        except ImportError:
            pass

    def _apply_numpy_seed(self, seed: int) -> None:
        """Apply the NumPy random seed.

        Args:
            seed: NumPy random seed value.
        """
        try:
            import numpy as np
            np.random.seed(seed)
            logger.debug("Applied NumPy seed: %d", seed)
        except ImportError:
            logger.debug("NumPy not available; skipping numpy seed")

    def _apply_torch_seed(self, seed: int) -> None:
        """Apply the PyTorch manual seed.

        Args:
            seed: PyTorch seed value.
        """
        try:
            import torch
            torch.manual_seed(seed)
            logger.debug("Applied PyTorch seed: %d", seed)
        except ImportError:
            logger.debug("PyTorch not available; skipping torch seed")

    def _apply_custom_seeds(self, seeds: Dict[str, int]) -> None:
        """Apply custom seeds for specific components.

        Custom seeds are stored for reference but application is
        component-specific. Components should query the seed manager
        for their seed value.

        Args:
            seeds: Dictionary mapping component name to seed value.
        """
        for component, seed in seeds.items():
            logger.debug("Custom seed for %s: %d", component, seed)

    def verify_seeds(
        self,
        current: SeedConfiguration,
        expected: SeedConfiguration,
    ) -> VerificationCheck:
        """Verify current seed configuration matches expected.

        Args:
            current: Current seed configuration.
            expected: Expected seed configuration.

        Returns:
            VerificationCheck result.
        """
        mismatches: List[str] = []

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

        # Check custom seeds
        all_custom_keys = set(expected.custom_seeds.keys()) | set(current.custom_seeds.keys())
        for key in sorted(all_custom_keys):
            exp_val = expected.custom_seeds.get(key)
            cur_val = current.custom_seeds.get(key)
            if exp_val != cur_val:
                mismatches.append(f"custom_seed[{key}]: {cur_val} vs {exp_val}")

        if not mismatches:
            return VerificationCheck(
                check_name="seed_verification",
                status=VerificationStatus.PASS,
                expected_value=expected.seed_hash,
                actual_value=current.seed_hash,
                message="Seed configuration matches",
            )

        record_non_determinism("random_seed")
        return VerificationCheck(
            check_name="seed_verification",
            status=VerificationStatus.FAIL,
            expected_value=expected.seed_hash,
            actual_value=current.seed_hash,
            message=f"Seed mismatches: {'; '.join(mismatches)}",
        )

    def get_current_seed_config(self) -> SeedConfiguration:
        """Get the currently active seed configuration.

        Returns:
            Current SeedConfiguration, or a default if none applied.
        """
        if self._current_seeds is not None:
            return self._current_seeds
        return SeedConfiguration()

    def store_seed_config(
        self,
        seed_config: SeedConfiguration,
        execution_id: str,
    ) -> str:
        """Store a seed configuration for an execution.

        Args:
            seed_config: Seed configuration to store.
            execution_id: Execution ID to associate with.

        Returns:
            The seed_hash as the storage key.
        """
        self._seed_history[execution_id] = seed_config
        logger.debug(
            "Stored seed config for execution %s: hash=%s",
            execution_id, seed_config.seed_hash,
        )
        return seed_config.seed_hash

    def get_seed_config_for_execution(
        self, execution_id: str,
    ) -> Optional[SeedConfiguration]:
        """Retrieve stored seed config for an execution.

        Args:
            execution_id: Execution ID to look up.

        Returns:
            SeedConfiguration or None if not found.
        """
        return self._seed_history.get(execution_id)


__all__ = [
    "SeedManager",
]
