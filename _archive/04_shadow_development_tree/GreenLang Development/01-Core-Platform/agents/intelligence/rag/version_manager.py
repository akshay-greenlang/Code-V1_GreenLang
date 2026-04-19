# -*- coding: utf-8 -*-
"""
Document version management system for climate/GHG accounting standards.

CRITICAL: This module tracks multiple versions of the same standard and provides
date-based version retrieval for historical compliance (e.g., 2019 reports must
use standards that were active in 2019).

Key features:
- Track multiple versions of same standard (e.g., GHG Protocol v1.00 vs v1.05)
- Date-based version retrieval (return correct version for historical compliance)
- Deprecation warnings
- Errata application tracking
- Version conflict detection
"""

from datetime import date, datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import logging

from greenlang.agents.intelligence.rag.models import DocMeta
from greenlang.agents.intelligence.rag.hashing import file_hash
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class VersionConflict(Exception):
    """Raised when version conflict is detected (same version, different checksums)."""
    pass


class DocumentVersionManager:
    """
    Manages document versions for regulatory compliance.

    Tracks multiple versions of the same standard and provides date-based
    retrieval for historical compliance reporting.

    Example:
        >>> vm = DocumentVersionManager()
        >>> # Register GHG Protocol v1.00 (2004)
        >>> vm.register_version(doc_meta_v1_00)
        >>> # Register GHG Protocol v1.05 (2015 revision)
        >>> vm.register_version(doc_meta_v1_05)
        >>> # Retrieve version for 2010 report (should return v1.00)
        >>> doc = vm.retrieve_by_date("ghg_protocol_corp", date(2010, 1, 1))
        >>> assert doc.version == "1.00"
        >>> # Retrieve version for 2020 report (should return v1.05)
        >>> doc = vm.retrieve_by_date("ghg_protocol_corp", date(2020, 1, 1))
        >>> assert doc.version == "1.05"
    """

    def __init__(self):
        """Initialize version manager."""
        # Map: standard_id → [versions sorted by publication_date]
        self.versions: Dict[str, List[DocMeta]] = defaultdict(list)

        # Map: (standard_id, version) → DocMeta (for fast lookup)
        self.version_index: Dict[Tuple[str, str], DocMeta] = {}

        # Map: doc_id → List of errata applications
        self.errata: Dict[str, List[Dict]] = defaultdict(list)

        # Map: standard_id → deprecation info
        self.deprecations: Dict[str, Dict] = {}

    def register_version(
        self,
        doc_meta: DocMeta,
        standard_id: Optional[str] = None
    ) -> None:
        """
        Register a document version.

        Args:
            doc_meta: Document metadata with version, publication_date, content_hash
            standard_id: Standard identifier (e.g., 'ghg_protocol_corp'). If None,
                        uses doc_meta.extra['standard_id'] or doc_meta.collection

        Raises:
            VersionConflict: If same version exists with different checksum
            ValueError: If required fields are missing

        Example:
            >>> vm = DocumentVersionManager()
            >>> vm.register_version(doc_meta, standard_id="ghg_protocol_corp")
        """
        # Determine standard_id
        if standard_id is None:
            standard_id = doc_meta.extra.get('standard_id', doc_meta.collection)

        # Validate required fields
        if not doc_meta.version:
            raise ValueError(f"Document {doc_meta.doc_id} missing version field")
        if not doc_meta.publication_date:
            raise ValueError(f"Document {doc_meta.doc_id} missing publication_date field")
        if not doc_meta.content_hash:
            raise ValueError(f"Document {doc_meta.doc_id} missing content_hash field")

        # Check for version conflicts
        version_key = (standard_id, doc_meta.version)
        if version_key in self.version_index:
            existing = self.version_index[version_key]
            if existing.content_hash != doc_meta.content_hash:
                raise VersionConflict(
                    f"Version conflict for {standard_id} v{doc_meta.version}: "
                    f"existing hash {existing.content_hash[:8]} != "
                    f"new hash {doc_meta.content_hash[:8]}. "
                    f"This indicates document tampering or incorrect version labeling."
                )
            else:
                # Same version, same hash - already registered
                logger.info(f"Version {standard_id} v{doc_meta.version} already registered")
                return

        # Add to versions list (maintain sorted order by publication_date)
        self.versions[standard_id].append(doc_meta)
        self.versions[standard_id].sort(key=lambda d: d.publication_date or date.min)

        # Add to index
        self.version_index[version_key] = doc_meta

        # Store standard_id in metadata for later retrieval
        doc_meta.extra['standard_id'] = standard_id

        logger.info(
            f"Registered {standard_id} v{doc_meta.version} "
            f"(published {doc_meta.publication_date}, hash {doc_meta.content_hash[:8]})"
        )

    def retrieve_by_date(
        self,
        standard_id: str,
        reference_date: date
    ) -> Optional[DocMeta]:
        """
        Retrieve the document version active on a specific date.

        Returns the most recent version published on or before reference_date.
        This is critical for historical compliance (e.g., 2019 report uses 2019 standards).

        Args:
            standard_id: Standard identifier (e.g., 'ghg_protocol_corp')
            reference_date: Reference date for version lookup

        Returns:
            DocMeta for the active version, or None if no version exists before date

        Example:
            >>> vm = DocumentVersionManager()
            >>> # GHG Protocol v1.00 published 2004-09-01
            >>> # GHG Protocol v1.05 published 2015-03-24
            >>> doc = vm.retrieve_by_date("ghg_protocol_corp", date(2010, 1, 1))
            >>> assert doc.version == "1.00"  # v1.05 not yet published
            >>> doc = vm.retrieve_by_date("ghg_protocol_corp", date(2020, 1, 1))
            >>> assert doc.version == "1.05"  # v1.05 is active
        """
        if standard_id not in self.versions:
            logger.warning(f"No versions registered for {standard_id}")
            return None

        # Find most recent version on or before reference_date
        # Versions are sorted by publication_date (earliest first)
        active_version = None
        for doc_meta in self.versions[standard_id]:
            if doc_meta.publication_date and doc_meta.publication_date <= reference_date:
                active_version = doc_meta
            else:
                # Later versions not yet published
                break

        if active_version:
            logger.info(
                f"Retrieved {standard_id} v{active_version.version} "
                f"for reference date {reference_date}"
            )

            # Check for deprecation warnings
            if standard_id in self.deprecations:
                dep_info = self.deprecations[standard_id]
                dep_date = dep_info.get('deprecation_date')
                if dep_date and reference_date >= dep_date:
                    logger.warning(
                        f"WARNING: {standard_id} v{active_version.version} is deprecated "
                        f"as of {dep_date}. Replacement: {dep_info.get('replacement', 'N/A')}"
                    )
        else:
            logger.warning(
                f"No version of {standard_id} published before {reference_date}"
            )

        return active_version

    def retrieve_by_version(
        self,
        standard_id: str,
        version: str
    ) -> Optional[DocMeta]:
        """
        Retrieve a specific version by standard_id and version string.

        Args:
            standard_id: Standard identifier (e.g., 'ghg_protocol_corp')
            version: Version string (e.g., '1.05')

        Returns:
            DocMeta for the specified version, or None if not found

        Example:
            >>> vm = DocumentVersionManager()
            >>> doc = vm.retrieve_by_version("ghg_protocol_corp", "1.05")
            >>> assert doc.version == "1.05"
        """
        version_key = (standard_id, version)
        doc_meta = self.version_index.get(version_key)

        if doc_meta:
            logger.info(f"Retrieved {standard_id} v{version}")
        else:
            logger.warning(f"Version {standard_id} v{version} not found")

        return doc_meta

    def list_versions(self, standard_id: str) -> List[DocMeta]:
        """
        List all versions of a standard.

        Args:
            standard_id: Standard identifier

        Returns:
            List of DocMeta objects sorted by publication_date (earliest first)

        Example:
            >>> vm = DocumentVersionManager()
            >>> versions = vm.list_versions("ghg_protocol_corp")
            >>> for v in versions:
            ...     print(f"v{v.version} published {v.publication_date}")
        """
        return list(self.versions.get(standard_id, []))

    def check_conflicts(self, query: str) -> List[DocMeta]:
        """
        Check if a query could match multiple versions.

        Returns all versions of standards that match the query string.
        Useful for warning users when version ambiguity exists.

        Args:
            query: Query string (e.g., "GHG Protocol Corporate Standard")

        Returns:
            List of matching DocMeta objects with version conflicts

        Example:
            >>> vm = DocumentVersionManager()
            >>> matches = vm.check_conflicts("GHG Protocol")
            >>> if len(matches) > 1:
            ...     print(f"WARNING: Query matches {len(matches)} versions")
            ...     for m in matches:
            ...         print(f"  - v{m.version} ({m.publication_date})")
        """
        query_lower = query.lower()
        matches = []

        for standard_id, version_list in self.versions.items():
            for doc_meta in version_list:
                # Check if query matches title or standard_id
                if (query_lower in doc_meta.title.lower() or
                    query_lower in standard_id.lower()):
                    matches.append(doc_meta)

        if len(matches) > 1:
            logger.warning(
                f"Query '{query}' matches {len(matches)} versions: " +
                ", ".join(f"{m.extra.get('standard_id', m.collection)} v{m.version}"
                         for m in matches)
            )

        return matches

    def apply_errata(
        self,
        doc_id: str,
        errata_date: date,
        description: str,
        sections_affected: Optional[List[str]] = None
    ) -> None:
        """
        Track errata application for a document.

        Errata are corrections/amendments published after the original document.
        This tracks which errata have been applied for audit purposes.

        Args:
            doc_id: Document identifier
            errata_date: Date errata was published/applied
            description: Description of errata (e.g., "Corrected Table 7.3 emission factors")
            sections_affected: List of section paths affected by errata

        Example:
            >>> vm = DocumentVersionManager()
            >>> vm.apply_errata(
            ...     doc_id="ghg_protocol_v1.05",
            ...     errata_date=date(2016, 6, 1),
            ...     description="Corrected emission factor for natural gas",
            ...     sections_affected=["Chapter 7 > Table 7.3"]
            ... )
        """
        errata_record = {
            'errata_date': errata_date,
            'description': description,
            'sections_affected': sections_affected or [],
            'applied_at': DeterministicClock.utcnow()
        }

        self.errata[doc_id].append(errata_record)

        logger.info(
            f"Applied errata to {doc_id}: {description} "
            f"(published {errata_date}, {len(sections_affected or [])} sections affected)"
        )

    def get_errata(self, doc_id: str) -> List[Dict]:
        """
        Get all errata applied to a document.

        Args:
            doc_id: Document identifier

        Returns:
            List of errata records (sorted by errata_date)

        Example:
            >>> vm = DocumentVersionManager()
            >>> errata_list = vm.get_errata("ghg_protocol_v1.05")
            >>> for e in errata_list:
            ...     print(f"{e['errata_date']}: {e['description']}")
        """
        errata_list = self.errata.get(doc_id, [])
        # Sort by errata_date
        errata_list.sort(key=lambda e: e['errata_date'])
        return errata_list

    def mark_deprecated(
        self,
        standard_id: str,
        deprecation_date: date,
        replacement: Optional[str] = None,
        reason: Optional[str] = None
    ) -> None:
        """
        Mark a standard as deprecated.

        Args:
            standard_id: Standard identifier
            deprecation_date: Date when standard was deprecated
            replacement: Replacement standard (e.g., 'ghg_protocol_corp_v2')
            reason: Reason for deprecation

        Example:
            >>> vm = DocumentVersionManager()
            >>> vm.mark_deprecated(
            ...     standard_id="ghg_protocol_corp_v1.00",
            ...     deprecation_date=date(2015, 3, 24),
            ...     replacement="ghg_protocol_corp_v1.05",
            ...     reason="Superseded by revised edition with updated emission factors"
            ... )
        """
        self.deprecations[standard_id] = {
            'deprecation_date': deprecation_date,
            'replacement': replacement,
            'reason': reason
        }

        logger.info(
            f"Marked {standard_id} as deprecated (effective {deprecation_date})" +
            (f", replacement: {replacement}" if replacement else "")
        )

    def is_deprecated(
        self,
        standard_id: str,
        reference_date: Optional[date] = None
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Check if a standard is deprecated.

        Args:
            standard_id: Standard identifier
            reference_date: Reference date for deprecation check (defaults to today)

        Returns:
            Tuple of (is_deprecated, deprecation_info)

        Example:
            >>> vm = DocumentVersionManager()
            >>> is_dep, info = vm.is_deprecated("ghg_protocol_corp_v1.00")
            >>> if is_dep:
            ...     print(f"Deprecated: {info['reason']}")
            ...     print(f"Use instead: {info['replacement']}")
        """
        if reference_date is None:
            reference_date = date.today()

        if standard_id not in self.deprecations:
            return False, None

        dep_info = self.deprecations[standard_id]
        dep_date = dep_info['deprecation_date']

        is_deprecated = reference_date >= dep_date

        return is_deprecated, dep_info if is_deprecated else None
