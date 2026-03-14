"""
PACK-001 CSRD Starter Pack - Configuration Module

This module provides configuration management for the CSRD Starter Pack,
including pack manifest loading, preset resolution, sector-specific
configuration, and environment-based overrides.

Usage:
    >>> from packs.eu_compliance.pack_001_csrd_starter.config import PackConfig
    >>> config = PackConfig.load()
    >>> config = PackConfig.load(size_preset="mid_market", sector_preset="manufacturing")
"""

from config.pack_config import (
    AgentComponentConfig,
    CSRDPackConfig,
    PackConfig,
    PerformanceTargets,
    PresetConfig,
    WorkflowConfig,
    WorkflowPhaseConfig,
)

__all__ = [
    "PackConfig",
    "CSRDPackConfig",
    "WorkflowConfig",
    "WorkflowPhaseConfig",
    "AgentComponentConfig",
    "PresetConfig",
    "PerformanceTargets",
]
