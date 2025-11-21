# -*- coding: utf-8 -*-
"""Determinism Testing Framework for GreenLang Phase 3.

This module provides comprehensive determinism testing infrastructure for all agents,
ensuring reproducible results across runs, platforms, and environments.

Components:
- DeterminismTester: Hash-based reproducibility verification
- SnapshotManager: Golden file testing for agent outputs
- Property-based tests: Invariant testing with Hypothesis

Author: GreenLang Framework Team
Phase: Phase 3 - Production Hardening
Date: November 2024
"""

from .test_framework import DeterminismTester, DeterminismResult
from .snapshot_manager import SnapshotManager, SnapshotDiff

__all__ = [
    "DeterminismTester",
    "DeterminismResult",
    "SnapshotManager",
    "SnapshotDiff",
]
