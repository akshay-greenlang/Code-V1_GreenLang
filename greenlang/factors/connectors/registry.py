# -*- coding: utf-8 -*-
"""
Connector registry: dynamic lookup by source_id (F060).

Thread-safe singleton registry for connector instances.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional, Type

from greenlang.factors.connectors.base import BaseConnector

logger = logging.getLogger(__name__)


class ConnectorRegistry:
    """
    Registry for connector classes and instances.

    Connectors are registered by source_id. The registry supports
    both class-level registration (for lazy instantiation) and
    instance-level registration (for pre-configured connectors).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._classes: Dict[str, Type[BaseConnector]] = {}
        self._instances: Dict[str, BaseConnector] = {}

    def register_class(self, source_id: str, cls: Type[BaseConnector]) -> None:
        """Register a connector class for a source_id."""
        with self._lock:
            if source_id in self._classes:
                logger.warning("Overwriting connector class for source_id=%s", source_id)
            self._classes[source_id] = cls
            logger.debug("Registered connector class %s for %s", cls.__name__, source_id)

    def register_instance(self, connector: BaseConnector) -> None:
        """Register a pre-configured connector instance."""
        with self._lock:
            sid = connector.source_id
            if sid in self._instances:
                logger.warning("Overwriting connector instance for source_id=%s", sid)
            self._instances[sid] = connector
            logger.debug("Registered connector instance for %s", sid)

    def get(
        self,
        source_id: str,
        *,
        license_key: Optional[str] = None,
    ) -> Optional[BaseConnector]:
        """
        Retrieve a connector by source_id.

        Checks instances first, then tries to instantiate from registered class.
        """
        with self._lock:
            if source_id in self._instances:
                return self._instances[source_id]

            cls = self._classes.get(source_id)
            if cls is not None:
                instance = cls(license_key=license_key)
                self._instances[source_id] = instance
                logger.info("Instantiated connector %s for %s", cls.__name__, source_id)
                return instance

        return None

    def list_source_ids(self) -> List[str]:
        """Return all registered source_ids (classes + instances)."""
        with self._lock:
            ids = set(self._classes.keys()) | set(self._instances.keys())
        return sorted(ids)

    def list_instances(self) -> List[BaseConnector]:
        """Return all active connector instances."""
        with self._lock:
            return list(self._instances.values())

    def unregister(self, source_id: str) -> bool:
        """Remove a connector by source_id. Returns True if found."""
        with self._lock:
            removed = False
            if source_id in self._instances:
                del self._instances[source_id]
                removed = True
            if source_id in self._classes:
                del self._classes[source_id]
                removed = True
            return removed

    def clear(self) -> None:
        """Remove all registrations."""
        with self._lock:
            self._classes.clear()
            self._instances.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(set(self._classes.keys()) | set(self._instances.keys()))

    def __contains__(self, source_id: str) -> bool:
        with self._lock:
            return source_id in self._classes or source_id in self._instances


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_global_registry: Optional[ConnectorRegistry] = None
_global_lock = threading.Lock()


def get_global_registry() -> ConnectorRegistry:
    """Return the global connector registry (lazy singleton)."""
    global _global_registry
    if _global_registry is None:
        with _global_lock:
            if _global_registry is None:
                _global_registry = ConnectorRegistry()
    return _global_registry
