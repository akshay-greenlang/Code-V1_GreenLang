"""Comply-Hub business services: adapter registry, orchestrator, applicability, normalizer."""

# Trigger adapter registration on first import of services package
from services import adapters as _adapters  # noqa: F401
