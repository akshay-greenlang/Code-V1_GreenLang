"""
GreenLang SDK - Core abstractions for packs
"""

from .base import Agent, Pipeline, Connector, Dataset, Report
from .context import Context, Artifact
from .pipeline import Pipeline as PipelineRunner
# Lazy import client to avoid loading agents at import time
# from .client import GreenLangClient
from .builder import AgentBuilder, WorkflowBuilder

# Lazy import for GreenLangClient
def __getattr__(name):
    if name == "GreenLangClient":
        from .client import GreenLangClient
        return GreenLangClient
    raise AttributeError(f"module 'greenlang.sdk' has no attribute '{name}'")

__all__ = [
    "Agent",
    "Pipeline",
    "Connector",
    "Dataset",
    "Report",
    "Context",
    "Artifact",
    "PipelineRunner",
    "GreenLangClient",
    "AgentBuilder",
    "WorkflowBuilder",
]
