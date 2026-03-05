"""
Target Setting Engine -- Facade for TargetConfigurationEngine

Re-exports TargetConfigurationEngine as TargetSettingEngine for the
unified __init__.py and setup module naming convention.

The underlying engine lives in target_configuration_engine.py and
implements full target lifecycle CRUD, coverage validation, annual
reduction rate calculations, Scope 3 requirement checks, target
summary generation, and SBTi submission form generation.

Example:
    >>> engine = TargetSettingEngine(config)
    >>> target = engine.create_target("org-1", request)
"""

from .target_configuration_engine import TargetConfigurationEngine as TargetSettingEngine

__all__ = ["TargetSettingEngine"]
