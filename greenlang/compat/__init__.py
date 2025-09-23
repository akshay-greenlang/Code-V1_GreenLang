"""
GreenLang Compatibility Shims
============================

This module provides compatibility shims to protect imports during the transition
to the new architecture without moving core packages. These shims ensure that
existing imports continue to work while providing deprecation warnings and
guidance for future-proofing code.

This is a safety net for PR #6 - adding compatibility without risky moves.
"""

import warnings
from typing import Dict, List, Optional

# Track deprecated imports for reporting
_deprecated_imports: Dict[str, List[str]] = {}


def _track_deprecated_import(old_path: str, new_path: str, stacklevel: int = 3) -> None:
    """Track a deprecated import for reporting purposes."""
    if old_path not in _deprecated_imports:
        _deprecated_imports[old_path] = []
    _deprecated_imports[old_path].append(new_path)

    warnings.warn(
        f"Import from '{old_path}' is deprecated. "
        f"Use '{new_path}' instead. "
        f"This import path may be removed in v2.0.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def get_deprecated_imports() -> Dict[str, List[str]]:
    """Get a summary of all deprecated imports that have been used."""
    return _deprecated_imports.copy()


# Common re-exports for modules that might be moved in the future
# This protects against import breakage without actually moving anything


class CompatibilityImporter:
    """
    A helper class to provide compatibility imports with deprecation warnings.
    """

    def __init__(
        self, old_module: str, new_module: str, items: Optional[List[str]] = None
    ):
        self.old_module = old_module
        self.new_module = new_module
        self.items = items or []

    def get_module(self):
        """Import and return the module, with deprecation warning."""
        _track_deprecated_import(self.old_module, self.new_module)
        try:
            # Try new location first
            return __import__(self.new_module, fromlist=self.items)
        except ImportError:
            # Fall back to old location if new one doesn't exist
            return __import__(self.old_module, fromlist=self.items)

    def get_item(self, item_name: str):
        """Import and return a specific item from the module."""
        _track_deprecated_import(
            f"{self.old_module}.{item_name}", f"{self.new_module}.{item_name}"
        )
        module = self.get_module()
        return getattr(module, item_name)


# Pre-define compatibility mappings for common modules
COMPAT_MAPPINGS = {
    # Core modules that might be moved
    "greenlang.core": "core.greenlang",
    "greenlang.packs": "core.greenlang.packs",
    "greenlang.policy": "core.greenlang.policy",
    "greenlang.runtime": "core.greenlang.runtime",
    "greenlang.sdk": "core.greenlang.sdk",
    "greenlang.cli": "core.greenlang.cli",
    "greenlang.hub": "core.greenlang.hub",
    "greenlang.utils": "core.greenlang.utils",
    # Test modules that might be reorganized
    "greenlang.test_utils": "tests.utils",
    "greenlang.testing": "tests.framework",
}


def create_compat_module(old_path: str, new_path: str):
    """
    Create a compatibility module that re-exports everything from the new location.
    """
    try:
        # Import the new module
        new_module = __import__(new_path, fromlist=[""])

        # Create a module-like object that warns on access
        class CompatModule:
            def __getattr__(self, name):
                _track_deprecated_import(f"{old_path}.{name}", f"{new_path}.{name}")
                return getattr(new_module, name)

        return CompatModule()
    except ImportError:
        # If new location doesn't exist, return None
        return None


# Commonly used classes and functions that might be moved
# These can be imported directly from greenlang.compat

try:
    # Try to import from new core structure
    from core.greenlang.sdk.base import Agent, Pipeline
    from core.greenlang.packs.registry import PackRegistry
    from core.greenlang.packs.loader import PackLoader

    # Re-export with deprecation tracking
    def get_agent_class():
        _track_deprecated_import(
            "greenlang.compat.Agent", "core.greenlang.sdk.base.Agent"
        )
        return Agent

    def get_pipeline_class():
        _track_deprecated_import(
            "greenlang.compat.Pipeline", "core.greenlang.sdk.base.Pipeline"
        )
        return Pipeline

    def get_pack_registry():
        _track_deprecated_import(
            "greenlang.compat.PackRegistry",
            "core.greenlang.packs.registry.PackRegistry",
        )
        return PackRegistry

    def get_pack_loader():
        _track_deprecated_import(
            "greenlang.compat.PackLoader", "core.greenlang.packs.loader.PackLoader"
        )
        return PackLoader

except ImportError:
    # Fallback to legacy imports if core structure doesn't exist
    def get_agent_class():
        _track_deprecated_import(
            "greenlang.compat.Agent", "greenlang.agents.base.BaseAgent"
        )
        from greenlang.agents.base import BaseAgent

        return BaseAgent

    def get_pipeline_class():
        _track_deprecated_import(
            "greenlang.compat.Pipeline", "greenlang.core.workflow.Workflow"
        )
        from greenlang.core.workflow import Workflow

        return Workflow

    def get_pack_registry():
        _track_deprecated_import(
            "greenlang.compat.PackRegistry", "greenlang.registry.PackRegistry"
        )
        try:
            from greenlang.registry import PackRegistry

            return PackRegistry
        except ImportError:
            return None

    def get_pack_loader():
        _track_deprecated_import(
            "greenlang.compat.PackLoader", "greenlang.packs.loader.PackLoader"
        )
        try:
            from greenlang.packs.loader import PackLoader

            return PackLoader
        except ImportError:
            return None


# Additional compatibility for commonly used classes
def get_orchestrator_class():
    """Get Orchestrator class with deprecation warning."""
    _track_deprecated_import(
        "greenlang.compat.Orchestrator", "greenlang.core.orchestrator.Orchestrator"
    )
    try:
        from greenlang.core.orchestrator import Orchestrator

        return Orchestrator
    except ImportError:
        # Return a placeholder if not available
        return None


def get_workflow_class():
    """Get Workflow class with deprecation warning."""
    _track_deprecated_import(
        "greenlang.compat.Workflow", "greenlang.core.workflow.Workflow"
    )
    try:
        from greenlang.core.workflow import Workflow

        return Workflow
    except ImportError:
        # Return a placeholder if not available
        return None


# Export commonly accessed items
Agent = get_agent_class()
Pipeline = get_pipeline_class()
PackRegistry = get_pack_registry()
PackLoader = get_pack_loader()
Orchestrator = get_orchestrator_class()
Workflow = get_workflow_class()

# Import testing utilities
from . import testing

__all__ = [
    "Agent",
    "Pipeline",
    "PackRegistry",
    "PackLoader",
    "Orchestrator",
    "Workflow",
    "CompatibilityImporter",
    "create_compat_module",
    "get_deprecated_imports",
    "COMPAT_MAPPINGS",
    "testing",
]
