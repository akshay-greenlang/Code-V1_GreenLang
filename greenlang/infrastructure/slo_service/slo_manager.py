# -*- coding: utf-8 -*-
"""
SLO Manager - OBS-005: SLO/SLI Definitions & Error Budget Management

Registry and lifecycle manager for SLO definitions.  Supports CRUD
operations, YAML import/export, versioned history, and thread-safe
concurrent access.

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-005 SLO/SLI Definitions & Error Budget Management
Status: Production Ready
"""

from __future__ import annotations

import copy
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None  # type: ignore[assignment]

from greenlang.infrastructure.slo_service.models import SLO, SLI


# ---------------------------------------------------------------------------
# SLOManager
# ---------------------------------------------------------------------------


class SLOManager:
    """Registry and lifecycle manager for SLO definitions.

    Provides CRUD operations, YAML import/export, version history,
    and thread-safe concurrent access to the SLO registry.

    Attributes:
        _registry: Mapping of slo_id to SLO.
        _history: Mapping of slo_id to list of previous SLO versions.
        _lock: Threading lock for concurrent access.
    """

    def __init__(self) -> None:
        """Initialize an empty SLO registry."""
        self._registry: Dict[str, SLO] = {}
        self._history: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create(self, slo: SLO) -> SLO:
        """Register a new SLO in the registry.

        Args:
            slo: SLO definition to register.

        Returns:
            The registered SLO.

        Raises:
            ValueError: If an SLO with the same slo_id already exists,
                or if an SLO with the same name already exists.
        """
        with self._lock:
            if slo.slo_id in self._registry:
                raise ValueError(
                    f"SLO with id '{slo.slo_id}' already exists"
                )

            for existing in self._registry.values():
                if existing.name == slo.name and not existing.deleted:
                    raise ValueError(
                        f"SLO with name '{slo.name}' already exists"
                    )

            self._registry[slo.slo_id] = slo
            self._history.setdefault(slo.slo_id, [])
            logger.info("SLO created: %s (%s)", slo.slo_id, slo.name)
            return slo

    def get(self, slo_id: str) -> Optional[SLO]:
        """Retrieve an SLO by its identifier.

        Args:
            slo_id: SLO identifier.

        Returns:
            SLO instance or None if not found.
        """
        with self._lock:
            slo = self._registry.get(slo_id)
            if slo and slo.deleted:
                return None
            return slo

    def update(self, slo_id: str, updates: Dict[str, Any]) -> SLO:
        """Update an SLO and increment its version.

        The previous version is saved in the history.

        Args:
            slo_id: SLO identifier.
            updates: Dictionary of fields to update.

        Returns:
            Updated SLO.

        Raises:
            KeyError: If the SLO is not found.
        """
        with self._lock:
            slo = self._registry.get(slo_id)
            if slo is None or slo.deleted:
                raise KeyError(f"SLO '{slo_id}' not found")

            self._history.setdefault(slo_id, []).append(slo.to_dict())

            for key, value in updates.items():
                if hasattr(slo, key) and key not in ("slo_id", "created_at"):
                    setattr(slo, key, value)

            slo.version += 1
            slo.updated_at = datetime.now(timezone.utc)
            logger.info("SLO updated: %s (v%d)", slo_id, slo.version)
            return slo

    def delete(self, slo_id: str) -> bool:
        """Soft-delete an SLO.

        Args:
            slo_id: SLO identifier.

        Returns:
            True if the SLO was deleted, False if not found.
        """
        with self._lock:
            slo = self._registry.get(slo_id)
            if slo is None:
                return False
            slo.deleted = True
            slo.updated_at = datetime.now(timezone.utc)
            logger.info("SLO soft-deleted: %s", slo_id)
            return True

    def list_all(
        self,
        service: Optional[str] = None,
        team: Optional[str] = None,
        include_deleted: bool = False,
    ) -> List[SLO]:
        """List SLOs with optional filtering.

        Args:
            service: Filter by service name.
            team: Filter by team name.
            include_deleted: Include soft-deleted SLOs.

        Returns:
            List of matching SLOs.
        """
        with self._lock:
            results = []
            for slo in self._registry.values():
                if not include_deleted and slo.deleted:
                    continue
                if service and slo.service != service:
                    continue
                if team and slo.team != team:
                    continue
                results.append(slo)
            return results

    def get_history(self, slo_id: str) -> List[Dict[str, Any]]:
        """Get version history for an SLO.

        Args:
            slo_id: SLO identifier.

        Returns:
            List of historical SLO dictionaries.
        """
        with self._lock:
            return list(self._history.get(slo_id, []))

    # ------------------------------------------------------------------
    # YAML import / export
    # ------------------------------------------------------------------

    def load_from_yaml(self, path: str) -> List[SLO]:
        """Load SLO definitions from a YAML file.

        Args:
            path: Path to the YAML file.

        Returns:
            List of loaded SLOs.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the YAML format is invalid.
        """
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is required for YAML import")

        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"SLO definitions file not found: {path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict) or "slos" not in data:
            raise ValueError(
                "Invalid SLO definitions format: expected 'slos' key"
            )

        loaded = []
        for slo_data in data["slos"]:
            slo = SLO.from_dict(slo_data)
            try:
                self.create(slo)
                loaded.append(slo)
            except ValueError:
                logger.warning("Skipping duplicate SLO: %s", slo.slo_id)

        logger.info("Loaded %d SLOs from %s", len(loaded), path)
        return loaded

    def save_to_yaml(self, path: str) -> None:
        """Export all SLOs to a YAML file.

        Args:
            path: Output file path.
        """
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is required for YAML export")

        slos = self.list_all()
        data = {"slos": [slo.to_dict() for slo in slos]}

        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        logger.info("Exported %d SLOs to %s", len(slos), path)

    def reload(self, path: str) -> List[SLO]:
        """Clear registry and reload from YAML.

        Args:
            path: Path to the YAML file.

        Returns:
            List of reloaded SLOs.
        """
        with self._lock:
            self._registry.clear()
            self._history.clear()

        return self.load_from_yaml(path)
