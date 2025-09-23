"""
GreenLang v0.1: Infrastructure for Climate Intelligence
========================================================

GreenLang is now pure infrastructure. Domain logic lives in packs.
Platform = SDK/CLI/Runtime + Hub + Policy/Provenance

Success = Developer Love + Trust + Distribution
"""

__author__ = "GreenLang Team"
__email__ = "team@greenlang.io"
__license__ = "MIT"

# Core infrastructure exports only
from .sdk.base import Agent, Pipeline, Connector, Dataset, Report
from .sdk.context import Context, Artifact
from .packs.registry import PackRegistry
from .packs.loader import PackLoader
from .runtime.executor import Executor
from .policy.enforcer import PolicyEnforcer

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
