# -*- coding: utf-8 -*-
"""
Evidence Manager - AGENT-FOUND-005: Citations & Evidence

Manages evidence packages with CRUD operations, item management,
finalization, and tamper-evident hash chains.

Evidence packages bundle citations, calculation evidence, and supporting
data into audit-ready documentation for regulatory compliance.

Zero-Hallucination Guarantees:
    - All packages finalized with SHA-256 package hashes
    - Tamper-evidence via hash integrity checks
    - Complete audit trail for package mutations

Example:
    >>> from greenlang.citations.evidence import EvidenceManager
    >>> manager = EvidenceManager()
    >>> package = manager.create_package("Scope 1 Emissions Q4 2025")
    >>> manager.add_item(package.package_id, evidence_item)
    >>> pkg_hash = manager.finalize_package(package.package_id)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-005 Citations & Evidence
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.citations.config import CitationsConfig, get_config
from greenlang.citations.models import (
    ChangeType,
    EvidenceItem,
    EvidencePackage,
    EvidenceType,
    RegulatoryFramework,
)
from greenlang.citations.provenance import ProvenanceTracker
from greenlang.citations.metrics import (
    record_evidence_item,
    record_evidence_package,
    record_operation,
    update_packages_count,
)

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


class EvidenceManager:
    """Manages evidence packages for audit-ready documentation.

    Provides creation, item management, finalization, and retrieval
    of evidence packages with SHA-256 integrity guarantees.

    Attributes:
        config: CitationsConfig instance.
        provenance: ProvenanceTracker instance.
        _packages: Internal storage of packages by ID.

    Example:
        >>> manager = EvidenceManager()
        >>> pkg = manager.create_package("Test Evidence")
        >>> item = EvidenceItem(
        ...     evidence_type="calculation",
        ...     description="CO2 calculation",
        ...     data={"co2_kg": 150.0},
        ... )
        >>> manager.add_item(pkg.package_id, item)
        >>> pkg_hash = manager.finalize_package(pkg.package_id)
        >>> assert pkg_hash is not None
    """

    def __init__(
        self,
        config: Optional[CitationsConfig] = None,
        provenance: Optional[ProvenanceTracker] = None,
    ) -> None:
        """Initialize EvidenceManager.

        Args:
            config: Optional config. Uses global config if None.
            provenance: Optional provenance tracker. Creates new one if None.
        """
        self.config = config or get_config()
        self.provenance = provenance or ProvenanceTracker()
        self._packages: Dict[str, EvidencePackage] = {}
        logger.info("EvidenceManager initialized")

    def create_package(
        self,
        name: str,
        description: str = "",
        user_id: str = "system",
        calculation_context: Optional[Dict[str, Any]] = None,
        calculation_result: Optional[Dict[str, Any]] = None,
        regulatory_frameworks: Optional[List[str]] = None,
        compliance_notes: Optional[str] = None,
    ) -> EvidencePackage:
        """Create a new evidence package.

        Args:
            name: Name of the evidence package.
            description: Description of what the package documents.
            user_id: User creating the package.
            calculation_context: Optional calculation context data.
            calculation_result: Optional calculation result data.
            regulatory_frameworks: Optional list of framework strings.
            compliance_notes: Optional compliance notes.

        Returns:
            Created EvidencePackage object.

        Raises:
            ValueError: If packages capacity is exceeded.
        """
        start = time.monotonic()

        # Check capacity
        if len(self._packages) >= self.config.max_packages:
            raise ValueError(
                f"Evidence manager at capacity "
                f"({self.config.max_packages} packages)"
            )

        # Build frameworks list
        frameworks = [
            RegulatoryFramework(fw)
            for fw in (regulatory_frameworks or [])
        ]

        # Create package
        package = EvidencePackage(
            name=name,
            description=description,
            calculation_context=calculation_context or {},
            calculation_result=calculation_result or {},
            regulatory_frameworks=frameworks,
            compliance_notes=compliance_notes,
            created_by=user_id,
        )

        # Store
        self._packages[package.package_id] = package

        # Record provenance
        if self.config.enable_change_logging:
            self.provenance.record(
                entity_type="package",
                entity_id=package.package_id,
                action="create",
                data_hash=self._compute_hash({"name": name}),
                user_id=user_id,
                metadata={"name": name},
            )

        # Update metrics
        record_evidence_package()
        update_packages_count(len(self._packages))

        duration = time.monotonic() - start
        record_operation("create_package", "success", duration)
        logger.info("Created evidence package: %s", package.package_id)

        return package

    def get_package(self, package_id: str) -> Optional[EvidencePackage]:
        """Get an evidence package by ID.

        Args:
            package_id: The package identifier.

        Returns:
            EvidencePackage or None if not found.
        """
        start = time.monotonic()
        result = self._packages.get(package_id)
        duration = time.monotonic() - start
        record_operation(
            "get_package",
            "success" if result else "not_found",
            duration,
        )
        return result

    def add_item(
        self,
        package_id: str,
        evidence_item: EvidenceItem,
    ) -> EvidencePackage:
        """Add an evidence item to a package.

        Args:
            package_id: The package to add the item to.
            evidence_item: The evidence item to add.

        Returns:
            Updated EvidencePackage object.

        Raises:
            ValueError: If package not found, finalized, or at item capacity.
        """
        start = time.monotonic()

        package = self._packages.get(package_id)
        if package is None:
            raise ValueError(f"Package {package_id} not found")

        if package.is_finalized:
            raise ValueError(
                f"Package {package_id} is finalized and cannot be modified"
            )

        # Check item capacity
        max_items = self.config.max_evidence_items_per_package
        if len(package.evidence_items) >= max_items:
            raise ValueError(
                f"Package at item capacity ({max_items} items)"
            )

        # Calculate content hash for the item
        evidence_item.content_hash = evidence_item.calculate_content_hash()

        # Add item and invalidate package hash
        package.evidence_items.append(evidence_item)
        package.package_hash = None

        # Record provenance
        if self.config.enable_change_logging:
            self.provenance.record(
                entity_type="evidence_item",
                entity_id=evidence_item.evidence_id,
                action="add",
                data_hash=evidence_item.content_hash,
                metadata={
                    "package_id": package_id,
                    "evidence_type": evidence_item.evidence_type.value,
                },
            )

        # Update metrics
        record_evidence_item()

        duration = time.monotonic() - start
        record_operation("add_item", "success", duration)
        logger.info(
            "Added evidence item %s to package %s",
            evidence_item.evidence_id,
            package_id,
        )

        return package

    def add_citation(
        self,
        package_id: str,
        citation_id: str,
    ) -> EvidencePackage:
        """Add a citation reference to a package.

        Args:
            package_id: The package to add the citation to.
            citation_id: The citation ID to link.

        Returns:
            Updated EvidencePackage object.

        Raises:
            ValueError: If package not found or finalized.
        """
        start = time.monotonic()

        package = self._packages.get(package_id)
        if package is None:
            raise ValueError(f"Package {package_id} not found")

        if package.is_finalized:
            raise ValueError(
                f"Package {package_id} is finalized and cannot be modified"
            )

        # Avoid duplicates
        if citation_id not in package.citation_ids:
            package.citation_ids.append(citation_id)
            package.package_hash = None  # Invalidate

        # Record provenance
        if self.config.enable_change_logging:
            self.provenance.record(
                entity_type="package",
                entity_id=package_id,
                action="add_citation",
                data_hash=self._compute_hash({
                    "package_id": package_id,
                    "citation_id": citation_id,
                }),
                metadata={"citation_id": citation_id},
            )

        duration = time.monotonic() - start
        record_operation("add_citation", "success", duration)
        logger.info(
            "Added citation %s to package %s", citation_id, package_id,
        )

        return package

    def finalize_package(self, package_id: str) -> str:
        """Finalize an evidence package and compute its tamper-evident hash.

        Once finalized, no further modifications are allowed.

        Args:
            package_id: The package to finalize.

        Returns:
            SHA-256 package hash string.

        Raises:
            ValueError: If package not found or already finalized.
        """
        start = time.monotonic()

        package = self._packages.get(package_id)
        if package is None:
            raise ValueError(f"Package {package_id} not found")

        if package.is_finalized:
            raise ValueError(
                f"Package {package_id} is already finalized"
            )

        # Calculate package hash
        package_hash = package.calculate_package_hash()
        package.package_hash = package_hash
        package.finalized_at = _utcnow()

        # Record provenance
        if self.config.enable_change_logging:
            self.provenance.record(
                entity_type="package",
                entity_id=package_id,
                action="finalize",
                data_hash=package_hash,
                metadata={
                    "items_count": len(package.evidence_items),
                    "citations_count": len(package.citation_ids),
                },
            )

        duration = time.monotonic() - start
        record_operation("finalize_package", "success", duration)
        logger.info(
            "Finalized package %s with hash %s",
            package_id, package_hash[:16],
        )

        return package_hash

    def list_packages(
        self,
        created_by: Optional[str] = None,
        finalized_only: bool = False,
        search: Optional[str] = None,
    ) -> List[EvidencePackage]:
        """List evidence packages with optional filtering.

        Args:
            created_by: Optional filter by creator user ID.
            finalized_only: If True, only return finalized packages.
            search: Optional search term for name/description.

        Returns:
            Filtered list of evidence packages.
        """
        start = time.monotonic()
        packages = list(self._packages.values())

        if created_by is not None:
            packages = [
                p for p in packages if p.created_by == created_by
            ]

        if finalized_only:
            packages = [p for p in packages if p.is_finalized]

        if search is not None:
            term = search.lower()
            packages = [
                p for p in packages
                if (
                    term in p.name.lower()
                    or term in p.description.lower()
                )
            ]

        duration = time.monotonic() - start
        record_operation("list_packages", "success", duration)
        return packages

    def delete_package(
        self,
        package_id: str,
        user_id: str = "system",
        reason: str = "Deletion",
    ) -> bool:
        """Delete an evidence package.

        Args:
            package_id: The package to delete.
            user_id: User performing the deletion.
            reason: Reason for deletion.

        Returns:
            True if deleted, False if not found.

        Raises:
            ValueError: If package is finalized (cannot delete finalized packages).
        """
        start = time.monotonic()

        package = self._packages.get(package_id)
        if package is None:
            record_operation(
                "delete_package", "not_found", time.monotonic() - start,
            )
            return False

        if package.is_finalized:
            raise ValueError(
                f"Cannot delete finalized package {package_id}"
            )

        del self._packages[package_id]

        # Record provenance
        if self.config.enable_change_logging:
            self.provenance.record(
                entity_type="package",
                entity_id=package_id,
                action="delete",
                data_hash=self._compute_hash({"package_id": package_id}),
                user_id=user_id,
                metadata={"reason": reason},
            )

        # Update metrics
        update_packages_count(len(self._packages))

        duration = time.monotonic() - start
        record_operation("delete_package", "success", duration)
        logger.info("Deleted evidence package: %s", package_id)

        return True

    def verify_package_integrity(self, package_id: str) -> bool:
        """Verify that a finalized package has not been tampered with.

        Args:
            package_id: The package to verify.

        Returns:
            True if integrity check passes.

        Raises:
            ValueError: If package not found or not finalized.
        """
        package = self._packages.get(package_id)
        if package is None:
            raise ValueError(f"Package {package_id} not found")

        if not package.is_finalized:
            raise ValueError(
                f"Package {package_id} is not finalized"
            )

        current_hash = package.calculate_package_hash()
        return current_hash == package.package_hash

    @property
    def count(self) -> int:
        """Return the number of evidence packages."""
        return len(self._packages)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_hash(data: Any) -> str:
        """Compute SHA-256 hash for provenance tracking.

        Args:
            data: Data to hash.

        Returns:
            Hex-encoded SHA-256 hash.
        """
        try:
            json_str = json.dumps(data, sort_keys=True, default=str)
            return hashlib.sha256(json_str.encode()).hexdigest()
        except Exception:
            return hashlib.sha256(str(data).encode()).hexdigest()


__all__ = [
    "EvidenceManager",
]
