# -*- coding: utf-8 -*-
"""
GreenLang v2.0: Infrastructure for Climate Intelligence
========================================================

GreenLang is now pure infrastructure. Domain logic lives in packs.
Platform = SDK/CLI/Runtime + Hub + Policy/Provenance

Success = Developer Love + Trust + Distribution
"""

__author__ = "GreenLang Team"
__email__ = "team@greenlang.in"
__license__ = "MIT"

# Import version
from ._version import __version__

# Core infrastructure exports only
# Updated imports for consolidated structure v2.0
try:
    from .integration.sdk.base import Agent, Pipeline, Connector, Dataset, Report
    from .integration.sdk.context import Context, Artifact
except ImportError:
    # Fallback for development
    Agent = Pipeline = Connector = Dataset = Report = None
    Context = Artifact = None

try:
    from .ecosystem.packs.registry import PackRegistry
    from .ecosystem.packs.loader import PackLoader
except ImportError:
    PackRegistry = PackLoader = None

try:
    from .execution.runtime.executor import Executor
except ImportError:
    Executor = None

try:
    from .governance.policy.enforcer import PolicyEnforcer
except ImportError:
    PolicyEnforcer = None

__all__ = [
    # Core SDK abstractions
    "Agent",
    "Pipeline",
    "Connector",
    "Dataset",
    "Report",
    "Context",
    "Artifact",
    # Pack system
    "PackRegistry",
    "PackLoader",
    # Runtime
    "Executor",
    # Policy
    "PolicyEnforcer",
]
