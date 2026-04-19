# -*- coding: utf-8 -*-
"""
Climate Ledger - Core Ledger Facade
=====================================

Wraps ``greenlang.data_commons.provenance.ProvenanceTracker`` and
``greenlang.utilities.provenance.ledger.write_run_ledger`` behind a clean
v3 product API for immutable audit trails.

The ``ClimateLedger`` class is the primary entry point for recording,
verifying, and exporting provenance chains within the Climate Ledger
product module.

Example::

    >>> from greenlang.climate_ledger.ledger import ClimateLedger
    >>> ledger = ClimateLedger(agent_name="scope1-calc")
    >>> chain_hash = ledger.record_entry("emission", "e-001", "calculate", "abc123")
    >>> valid, chain = ledger.verify("e-001")
    >>> assert valid is True

Author: GreenLang Platform Team
Date: April 2026
Status: Production Ready
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from greenlang.data_commons.provenance import ProvenanceTracker
from greenlang.utilities.provenance.ledger import write_run_ledger

logger = logging.getLogger(__name__)


class ClimateLedger:
    """Unified Climate Ledger for immutable provenance tracking.

    Provides a product-grade API over the lower-level
    ``ProvenanceTracker`` (chain-hashing) and ``write_run_ledger``
    (JSONL run records) infrastructure.

    Attributes:
        agent_name: Identifier for the agent using this ledger instance.
        storage_backend: Storage mode -- currently ``"memory"`` (default).
        tracker: The underlying ``ProvenanceTracker`` performing
            SHA-256 chain hashing.

    Example::

        >>> ledger = ClimateLedger(agent_name="ghg-inventory")
        >>> h = ledger.record_entry("facility", "f-042", "ingest", "deadbeef")
        >>> ok, chain = ledger.verify("f-042")
        >>> assert ok
    """

    def __init__(
        self,
        agent_name: str,
        storage_backend: str = "memory",
    ) -> None:
        """Initialize a ClimateLedger instance.

        Args:
            agent_name: Short kebab-case identifier for the owning agent.
            storage_backend: Storage mode. Currently only ``"memory"``
                is supported; future versions will add ``"sqlite"``
                and ``"postgres"``.
        """
        self.agent_name = agent_name
        self.storage_backend = storage_backend
        self.tracker = ProvenanceTracker(agent_name=agent_name)
        logger.info(
            "ClimateLedger initialized (agent=%s, backend=%s)",
            agent_name,
            storage_backend,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_entry(
        self,
        entity_type: str,
        entity_id: str,
        operation: str,
        content_hash: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Record a provenance entry in the ledger.

        Delegates to ``ProvenanceTracker.record()`` and returns the
        resulting chain hash.

        Args:
            entity_type: Category of the entity (e.g. ``"emission"``,
                ``"facility"``, ``"factor"``).
            entity_id: Unique identifier for the entity within its type.
            operation: Action being recorded (e.g. ``"ingest"``,
                ``"calculate"``, ``"validate"``).
            content_hash: SHA-256 hex digest of the operation payload.
            metadata: Optional dictionary of extra context.  Stored as
                the ``user_id`` field on the underlying tracker when
                serialized to JSON; primarily for forward compatibility.

        Returns:
            The chain hash (SHA-256 hex string) linking this entry to
            the previous one in the global chain.

        Example::

            >>> h = ledger.record_entry(
            ...     "emission", "e-001", "calculate", "abc123",
            ...     metadata={"framework": "GHG Protocol"},
            ... )
        """
        user_id = "system"
        if metadata:
            user_id = json.dumps(metadata, sort_keys=True, default=str)

        chain_hash = self.tracker.record(
            entity_type=entity_type,
            entity_id=entity_id,
            action=operation,
            data_hash=content_hash,
            user_id=user_id,
        )

        logger.debug(
            "Ledger entry recorded: %s/%s op=%s chain=%s",
            entity_type,
            entity_id,
            operation,
            chain_hash[:16],
        )
        return chain_hash

    def verify(
        self,
        entity_id: str,
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """Verify the integrity of an entity's provenance chain.

        Delegates to ``ProvenanceTracker.verify_chain()``.

        Args:
            entity_id: The entity whose chain should be verified.

        Returns:
            A two-element tuple of ``(is_valid, chain_entries)`` where
            ``is_valid`` is ``True`` when the chain is intact and
            ``chain_entries`` is the ordered list of provenance dicts.

        Example::

            >>> ok, entries = ledger.verify("e-001")
            >>> assert ok
            >>> assert len(entries) >= 1
        """
        return self.tracker.verify_chain(entity_id)

    def export(
        self,
        entity_id: Optional[str] = None,
        format: str = "json",
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Export provenance chain data.

        When *entity_id* is provided, returns that entity's chain as a
        list of entry dicts.  When omitted, returns the full global
        chain wrapped in a summary dict.

        Args:
            entity_id: Restrict export to a single entity.  ``None``
                exports the entire global chain.
            format: Output format hint.  Currently only ``"json"``
                (Python dicts/lists) is supported.

        Returns:
            A list of entry dicts (single entity) or a summary dict
            (global export).

        Raises:
            ValueError: If *format* is not ``"json"``.

        Example::

            >>> chain = ledger.export("e-001")
            >>> all_data = ledger.export()
        """
        if format != "json":
            raise ValueError(
                "Unsupported export format %r; only 'json' is supported" % format
            )

        if entity_id is not None:
            return self.tracker.get_chain(entity_id)

        return {
            "agent_name": self.agent_name,
            "entry_count": self.tracker.entry_count,
            "entity_count": self.tracker.entity_count,
            "entries": self.tracker.get_global_chain(limit=10_000),
        }

    def write_run_record(
        self,
        result: Any,
        ctx: Any,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Write a deterministic JSONL run record to disk.

        Delegates to ``greenlang.utilities.provenance.ledger.write_run_ledger``.

        Args:
            result: Execution result object (exposes ``.success``,
                ``.outputs``, ``.metrics``, etc.).
            ctx: Execution context (exposes ``.pipeline_spec``,
                ``.inputs``, ``.config``, etc.).
            output_path: Destination file path.  Defaults to
                ``out/run.json`` when ``None``.

        Returns:
            ``pathlib.Path`` pointing to the written ledger file.

        Example::

            >>> path = ledger.write_run_record(result, ctx, "out/my_run.json")
        """
        path_arg: Optional[Path] = None
        if output_path is not None:
            path_arg = Path(output_path)

        written = write_run_ledger(result, ctx, output_path=path_arg)
        logger.info("Run record written to %s", written)
        return written

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def entry_count(self) -> int:
        """Total number of provenance entries across all entities."""
        return self.tracker.entry_count

    @property
    def entity_count(self) -> int:
        """Number of unique entities tracked in this ledger."""
        return self.tracker.entity_count

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            "ClimateLedger(agent_name=%r, entries=%d, entities=%d)"
            % (self.agent_name, self.entry_count, self.entity_count)
        )
