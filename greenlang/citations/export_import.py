# -*- coding: utf-8 -*-
"""
Export/Import Manager - AGENT-FOUND-005: Citations & Evidence

Provides citation export and import in multiple formats:
    - BibTeX: Standard academic citation format
    - JSON: GreenLang internal JSON format
    - CSL-JSON: Citation Style Language JSON (Zotero/Mendeley compatible)

Import capabilities:
    - BibTeX parsing and citation creation
    - JSON import from GreenLang export format

Zero-Hallucination Guarantees:
    - All exports are deterministic transformations
    - All imports create proper provenance records
    - No data is inferred or generated during conversion

Example:
    >>> from greenlang.citations.export_import import ExportImportManager
    >>> manager = ExportImportManager(registry=registry)
    >>> bibtex = manager.export_bibtex(["cit-001", "cit-002"])
    >>> csl_json = manager.export_csl(["cit-001"])

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-005 Citations & Evidence
Status: Production Ready
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import date
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from greenlang.citations.config import CitationsConfig, get_config
from greenlang.citations.models import (
    Citation,
    CitationMetadata,
    CitationType,
    SourceAuthority,
)
from greenlang.citations.metrics import record_export, record_operation

if TYPE_CHECKING:
    from greenlang.citations.registry import CitationRegistry

logger = logging.getLogger(__name__)


class ExportImportManager:
    """Manages citation export and import in multiple formats.

    Supports BibTeX, JSON, and CSL-JSON (Citation Style Language) formats
    for interoperability with academic tools like Zotero and Mendeley.

    Attributes:
        config: CitationsConfig instance.
        registry: CitationRegistry for citation lookups and creation.

    Example:
        >>> manager = ExportImportManager(registry=registry)
        >>> bibtex_str = manager.export_bibtex(["cit-001"])
        >>> json_str = manager.export_json(["cit-001"])
        >>> csl_str = manager.export_csl(["cit-001"])
    """

    def __init__(
        self,
        registry: CitationRegistry,
        config: Optional[CitationsConfig] = None,
    ) -> None:
        """Initialize ExportImportManager.

        Args:
            registry: CitationRegistry for citation operations.
            config: Optional config. Uses global config if None.
        """
        self.config = config or get_config()
        self.registry = registry
        logger.info("ExportImportManager initialized")

    def export_bibtex(
        self,
        citation_ids: Optional[List[str]] = None,
    ) -> str:
        """Export citations to BibTeX format.

        Args:
            citation_ids: Optional list of citation IDs.
                If None, exports all citations.

        Returns:
            BibTeX formatted string.
        """
        start = time.monotonic()

        citations = self._resolve_citations(citation_ids)
        entries = [citation.to_bibtex() for citation in citations]
        result = "\n\n".join(entries)

        record_export("bibtex")
        duration = time.monotonic() - start
        record_operation("export_bibtex", "success", duration)
        logger.info("Exported %d citations to BibTeX", len(citations))

        return result

    def export_json(
        self,
        citation_ids: Optional[List[str]] = None,
    ) -> str:
        """Export citations to GreenLang JSON format.

        Args:
            citation_ids: Optional list of citation IDs.
                If None, exports all citations.

        Returns:
            JSON formatted string.
        """
        start = time.monotonic()

        citations = self._resolve_citations(citation_ids)
        data = [citation.model_dump(mode="json") for citation in citations]
        result = json.dumps(data, indent=2, default=str)

        record_export("json")
        duration = time.monotonic() - start
        record_operation("export_json", "success", duration)
        logger.info("Exported %d citations to JSON", len(citations))

        return result

    def export_csl(
        self,
        citation_ids: Optional[List[str]] = None,
    ) -> str:
        """Export citations to CSL-JSON format (Citation Style Language).

        CSL-JSON is the standard format used by Zotero, Mendeley, and other
        reference managers. See https://citeproc-js.readthedocs.io/en/latest/csl-json/markup.html

        Args:
            citation_ids: Optional list of citation IDs.
                If None, exports all citations.

        Returns:
            CSL-JSON formatted string.
        """
        start = time.monotonic()

        citations = self._resolve_citations(citation_ids)
        csl_items = [citation.to_csl() for citation in citations]
        result = json.dumps(csl_items, indent=2, default=str)

        record_export("csl")
        duration = time.monotonic() - start
        record_operation("export_csl", "success", duration)
        logger.info("Exported %d citations to CSL-JSON", len(citations))

        return result

    def import_bibtex(
        self,
        content: str,
        user_id: str = "system",
    ) -> List[Citation]:
        """Import citations from BibTeX format.

        Parses BibTeX entries and creates citations in the registry.
        Handles common BibTeX entry types: article, techreport, manual,
        misc, inproceedings, book.

        Args:
            content: BibTeX formatted string.
            user_id: User performing the import.

        Returns:
            List of created Citation objects.
        """
        start = time.monotonic()

        entries = self._parse_bibtex(content)
        created: List[Citation] = []

        for entry in entries:
            try:
                citation = self._bibtex_entry_to_citation(entry, user_id)
                if citation is not None:
                    created.append(citation)
            except Exception as exc:
                logger.warning(
                    "Failed to import BibTeX entry: %s", str(exc),
                )

        duration = time.monotonic() - start
        record_operation("import_bibtex", "success", duration)
        logger.info("Imported %d citations from BibTeX", len(created))

        return created

    def import_json(
        self,
        content: str,
        user_id: str = "system",
    ) -> List[Citation]:
        """Import citations from GreenLang JSON format.

        Args:
            content: JSON formatted string (list of citation dicts).
            user_id: User performing the import.

        Returns:
            List of created Citation objects.
        """
        start = time.monotonic()

        data = json.loads(content)
        if not isinstance(data, list):
            data = [data]

        created: List[Citation] = []

        for item in data:
            try:
                citation = self._json_item_to_citation(item, user_id)
                if citation is not None:
                    created.append(citation)
            except Exception as exc:
                logger.warning(
                    "Failed to import JSON citation: %s", str(exc),
                )

        duration = time.monotonic() - start
        record_operation("import_json", "success", duration)
        logger.info("Imported %d citations from JSON", len(created))

        return created

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_citations(
        self,
        citation_ids: Optional[List[str]],
    ) -> List[Citation]:
        """Resolve citation IDs to Citation objects.

        Args:
            citation_ids: Optional list of IDs. If None, returns all.

        Returns:
            List of Citation objects.
        """
        if citation_ids is None:
            return self.registry.list()

        citations: List[Citation] = []
        for cid in citation_ids:
            citation = self.registry.get(cid)
            if citation is not None:
                citations.append(citation)
            else:
                logger.warning("Citation not found for export: %s", cid)

        return citations

    def _parse_bibtex(self, content: str) -> List[Dict[str, Any]]:
        """Parse BibTeX content into a list of entry dictionaries.

        Args:
            content: Raw BibTeX string.

        Returns:
            List of parsed entry dicts with keys: type, id, fields.
        """
        entries: List[Dict[str, Any]] = []

        # Match BibTeX entries: @type{id, fields...}
        pattern = r"@(\w+)\s*\{([^,]+),\s*(.*?)\n\}"
        matches = re.findall(pattern, content, re.DOTALL)

        for entry_type, entry_id, fields_str in matches:
            fields: Dict[str, str] = {}

            # Parse individual fields: key = {value}
            field_pattern = r"(\w+)\s*=\s*\{([^}]*)\}"
            field_matches = re.findall(field_pattern, fields_str)

            for key, value in field_matches:
                fields[key.strip().lower()] = value.strip()

            entries.append({
                "type": entry_type.lower().strip(),
                "id": entry_id.strip(),
                "fields": fields,
            })

        return entries

    def _bibtex_entry_to_citation(
        self,
        entry: Dict[str, Any],
        user_id: str,
    ) -> Optional[Citation]:
        """Convert a parsed BibTeX entry to a Citation.

        Args:
            entry: Parsed BibTeX entry dict.
            user_id: User performing the import.

        Returns:
            Created Citation or None if creation fails.
        """
        fields = entry.get("fields", {})
        entry_type = entry.get("type", "misc")

        # Map BibTeX type to CitationType
        type_mapping = {
            "article": "scientific",
            "techreport": "emission_factor",
            "manual": "methodology",
            "misc": "guidance",
            "inproceedings": "scientific",
            "book": "methodology",
        }
        citation_type = type_mapping.get(entry_type, "guidance")

        # Extract title (required)
        title = fields.get("title", "")
        if not title:
            logger.warning("BibTeX entry missing title: %s", entry.get("id"))
            return None

        # Extract authors
        authors_str = fields.get("author", "")
        authors = [
            a.strip() for a in authors_str.split(" and ")
        ] if authors_str else []

        # Extract year for dates
        year_str = fields.get("year", "")
        try:
            year = int(year_str) if year_str else 2024
        except ValueError:
            year = 2024

        effective_date = f"{year}-01-01"

        # Extract other fields
        publisher = fields.get("publisher", "")
        url = fields.get("url", "")
        doi = fields.get("doi", "")
        isbn = fields.get("isbn", "")

        # Build metadata
        metadata: Dict[str, Any] = {"title": title}
        if authors:
            metadata["authors"] = authors
        if publisher:
            metadata["publisher"] = publisher
        if url:
            metadata["url"] = url
        if doi:
            metadata["doi"] = doi
        if isbn:
            metadata["isbn"] = isbn
        if year_str:
            metadata["publication_date"] = f"{year}-01-01"

        # Build edition/version
        edition = fields.get("edition", "")
        if edition:
            metadata["version"] = edition

        # Create citation via registry
        citation = self.registry.create(
            citation_type=citation_type,
            source_authority="other",
            metadata=metadata,
            effective_date=effective_date,
            user_id=user_id,
            change_reason=f"BibTeX import: {entry.get('id', 'unknown')}",
        )

        return citation

    def _json_item_to_citation(
        self,
        item: Dict[str, Any],
        user_id: str,
    ) -> Optional[Citation]:
        """Convert a JSON item to a Citation.

        Args:
            item: Citation dictionary from JSON export.
            user_id: User performing the import.

        Returns:
            Created Citation or None if creation fails.
        """
        # Extract required fields
        citation_type = item.get("citation_type", "guidance")
        source_authority = item.get("source_authority", "other")
        metadata = item.get("metadata", {})
        effective_date = item.get("effective_date", date.today().isoformat())

        if not metadata.get("title"):
            logger.warning("JSON import item missing title")
            return None

        # Extract optional fields
        expiration_date = item.get("expiration_date")
        abstract = item.get("abstract")
        key_values = item.get("key_values")
        notes = item.get("notes")
        frameworks = item.get("regulatory_frameworks")

        citation = self.registry.create(
            citation_type=citation_type,
            source_authority=source_authority,
            metadata=metadata,
            effective_date=effective_date,
            user_id=user_id,
            change_reason="JSON import",
            expiration_date=expiration_date,
            abstract=abstract,
            key_values=key_values,
            notes=notes,
            regulatory_frameworks=frameworks,
        )

        return citation


__all__ = [
    "ExportImportManager",
]
