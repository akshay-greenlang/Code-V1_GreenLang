# -*- coding: utf-8 -*-
"""
Verification Engine - AGENT-FOUND-005: Citations & Evidence

Provides citation verification logic including expiration checks,
supersession detection, hash integrity validation, required field
validation by source authority, and DOI format validation.

Zero-Hallucination Guarantees:
    - All verification is deterministic and rule-based
    - No LLM or ML models involved in verification logic
    - Complete audit of which checks were performed

Example:
    >>> from greenlang.citations.verification import VerificationEngine
    >>> engine = VerificationEngine(registry=registry)
    >>> record = engine.verify_citation("cit-001")
    >>> print(record.status)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-005 Citations & Evidence
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from greenlang.citations.config import CitationsConfig, get_config
from greenlang.citations.models import (
    Citation,
    CitationType,
    SourceAuthority,
    VerificationRecord,
    VerificationStatus,
)
from greenlang.citations.metrics import (
    record_operation,
    record_verification,
    record_verification_failure,
)

if TYPE_CHECKING:
    from greenlang.citations.registry import CitationRegistry

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Source authorities that require version field
# ---------------------------------------------------------------------------

_VERSION_REQUIRED_AUTHORITIES = frozenset({
    SourceAuthority.DEFRA,
    SourceAuthority.EPA,
    SourceAuthority.ECOINVENT,
    SourceAuthority.IPCC,
    SourceAuthority.EXIOBASE,
})


class VerificationEngine:
    """Performs citation verification checks.

    Runs a suite of deterministic checks against citations to determine
    their verification status. Checks include expiration, supersession,
    hash integrity, required fields, and DOI format validation.

    Attributes:
        config: CitationsConfig instance.
        registry: CitationRegistry to look up citations.
        _history: Verification history by citation ID.

    Example:
        >>> engine = VerificationEngine(registry=registry)
        >>> record = engine.verify_citation("cit-001")
        >>> assert record.status == VerificationStatus.VERIFIED
    """

    def __init__(
        self,
        registry: CitationRegistry,
        config: Optional[CitationsConfig] = None,
    ) -> None:
        """Initialize VerificationEngine.

        Args:
            registry: CitationRegistry for citation lookups.
            config: Optional config. Uses global config if None.
        """
        self.config = config or get_config()
        self.registry = registry
        self._history: Dict[str, List[VerificationRecord]] = {}
        logger.info("VerificationEngine initialized")

    def verify_citation(
        self,
        citation_id: str,
        user_id: str = "system",
    ) -> VerificationRecord:
        """Verify a single citation.

        Runs all verification checks and returns a VerificationRecord
        with the result. Also updates the citation's verification status.

        Args:
            citation_id: ID of the citation to verify.
            user_id: User performing the verification.

        Returns:
            VerificationRecord with check results.

        Raises:
            ValueError: If citation not found.
        """
        start = time.monotonic()

        citation = self.registry.get(citation_id)
        if citation is None:
            raise ValueError(f"Citation {citation_id} not found")

        checks_performed: List[str] = []
        issues: List[str] = []

        # Run verification checks
        status = self._run_checks(citation, checks_performed, issues)

        # Update citation verification status
        citation.verification_status = status
        citation.verified_at = _utcnow()
        citation.verified_by = user_id

        # Create verification record
        record = VerificationRecord(
            citation_id=citation_id,
            status=status,
            checked_by=user_id,
            checks_performed=checks_performed,
            issues=issues,
        )
        record.provenance_hash = self._compute_hash({
            "citation_id": citation_id,
            "status": status.value,
            "checks": checks_performed,
            "issues": issues,
            "timestamp": record.checked_at.isoformat(),
        })

        # Store in history
        if citation_id not in self._history:
            self._history[citation_id] = []
        self._history[citation_id].append(record)

        # Record metrics
        result_str = "pass" if status == VerificationStatus.VERIFIED else "fail"
        record_verification(result_str)
        if issues:
            for issue in issues:
                record_verification_failure(issue.split(":")[0])

        duration = time.monotonic() - start
        record_operation("verify_citation", "success", duration)
        logger.info(
            "Verified citation %s: %s (%d checks, %d issues)",
            citation_id, status.value,
            len(checks_performed), len(issues),
        )

        return record

    def verify_batch(
        self,
        citation_ids: List[str],
        user_id: str = "system",
    ) -> Dict[str, VerificationRecord]:
        """Verify multiple citations.

        Args:
            citation_ids: List of citation IDs to verify.
            user_id: User performing the verification.

        Returns:
            Dictionary mapping citation_id to VerificationRecord.
        """
        start = time.monotonic()
        results: Dict[str, VerificationRecord] = {}

        for cid in citation_ids:
            try:
                results[cid] = self.verify_citation(cid, user_id)
            except ValueError as exc:
                logger.warning("Batch verify skipped %s: %s", cid, exc)
                results[cid] = VerificationRecord(
                    citation_id=cid,
                    status=VerificationStatus.INVALID,
                    checked_by=user_id,
                    issues=[str(exc)],
                )

        duration = time.monotonic() - start
        record_operation("verify_batch", "success", duration)
        logger.info(
            "Batch verified %d citations in %.3fs",
            len(citation_ids), duration,
        )

        return results

    def check_expiration(self, citation_id: str) -> bool:
        """Check if a citation has expired.

        Args:
            citation_id: The citation to check.

        Returns:
            True if the citation has NOT expired (is still valid).

        Raises:
            ValueError: If citation not found.
        """
        citation = self.registry.get(citation_id)
        if citation is None:
            raise ValueError(f"Citation {citation_id} not found")

        if citation.expiration_date is None:
            return True
        return citation.expiration_date >= date.today()

    def check_supersession(self, citation_id: str) -> Optional[str]:
        """Check if a citation has been superseded.

        Args:
            citation_id: The citation to check.

        Returns:
            The superseding citation ID if superseded, None otherwise.

        Raises:
            ValueError: If citation not found.
        """
        citation = self.registry.get(citation_id)
        if citation is None:
            raise ValueError(f"Citation {citation_id} not found")

        return citation.superseded_by

    def check_hash_integrity(self, citation_id: str) -> bool:
        """Check if a citation's content hash is still valid.

        Args:
            citation_id: The citation to check.

        Returns:
            True if hash is valid or not set.

        Raises:
            ValueError: If citation not found.
        """
        citation = self.registry.get(citation_id)
        if citation is None:
            raise ValueError(f"Citation {citation_id} not found")

        if not citation.content_hash:
            return True

        current_hash = citation.calculate_content_hash()
        return current_hash == citation.content_hash

    def get_verification_history(
        self,
        citation_id: str,
    ) -> List[VerificationRecord]:
        """Get verification history for a citation.

        Args:
            citation_id: The citation to get history for.

        Returns:
            List of VerificationRecord objects, newest first.
        """
        records = self._history.get(citation_id, [])
        return list(reversed(records))

    # ------------------------------------------------------------------
    # Internal check methods
    # ------------------------------------------------------------------

    def _run_checks(
        self,
        citation: Citation,
        checks_performed: List[str],
        issues: List[str],
    ) -> VerificationStatus:
        """Run all verification checks on a citation.

        Args:
            citation: The citation to verify.
            checks_performed: List to append check names to.
            issues: List to append issue descriptions to.

        Returns:
            Final VerificationStatus.
        """
        # 1. Expiration check
        checks_performed.append("expiration_check")
        if not self._check_expiration(citation, issues):
            return VerificationStatus.EXPIRED

        # 2. Supersession check
        checks_performed.append("supersession_check")
        if not self._check_supersession(citation, issues):
            return VerificationStatus.SUPERSEDED

        # 3. Hash integrity check
        if self.config.enable_hash_validation:
            checks_performed.append("hash_integrity_check")
            if not self._check_hash(citation, issues):
                return VerificationStatus.INVALID

        # 4. Required fields check
        checks_performed.append("required_fields_check")
        if not self._check_required_fields(citation, issues):
            return VerificationStatus.UNVERIFIED

        # 5. DOI format validation for scientific citations
        if citation.citation_type == CitationType.SCIENTIFIC:
            checks_performed.append("doi_format_check")
            if not self._check_doi_format(citation, issues):
                return VerificationStatus.UNVERIFIED

        return VerificationStatus.VERIFIED

    def _check_expiration(
        self,
        citation: Citation,
        issues: List[str],
    ) -> bool:
        """Check if citation has expired.

        Args:
            citation: Citation to check.
            issues: List to append issues to.

        Returns:
            True if not expired.
        """
        if citation.expiration_date and citation.expiration_date < date.today():
            issues.append(
                f"expiration: Citation expired on {citation.expiration_date}"
            )
            return False
        return True

    def _check_supersession(
        self,
        citation: Citation,
        issues: List[str],
    ) -> bool:
        """Check if citation has been superseded.

        Args:
            citation: Citation to check.
            issues: List to append issues to.

        Returns:
            True if not superseded.
        """
        if citation.superseded_by:
            issues.append(
                f"supersession: Superseded by {citation.superseded_by}"
            )
            return False
        return True

    def _check_hash(
        self,
        citation: Citation,
        issues: List[str],
    ) -> bool:
        """Check content hash integrity.

        Args:
            citation: Citation to check.
            issues: List to append issues to.

        Returns:
            True if hash is valid or not set.
        """
        if not citation.content_hash:
            return True

        current_hash = citation.calculate_content_hash()
        if current_hash != citation.content_hash:
            issues.append(
                "hash_integrity: Content hash mismatch "
                f"(stored={citation.content_hash[:16]}..., "
                f"current={current_hash[:16]}...)"
            )
            return False
        return True

    def _check_required_fields(
        self,
        citation: Citation,
        issues: List[str],
    ) -> bool:
        """Check required fields based on source authority.

        Args:
            citation: Citation to check.
            issues: List to append issues to.

        Returns:
            True if all required fields are present.
        """
        valid = True

        # Version required for emission factor databases
        if citation.source_authority in _VERSION_REQUIRED_AUTHORITIES:
            if not citation.metadata.version:
                issues.append(
                    f"required_field: Version required for "
                    f"{citation.source_authority.value} citations"
                )
                valid = False

        # Title always required
        if not citation.metadata.title:
            issues.append("required_field: Title is required")
            valid = False

        return valid

    def _check_doi_format(
        self,
        citation: Citation,
        issues: List[str],
    ) -> bool:
        """Validate DOI format for scientific citations.

        Args:
            citation: Citation to check.
            issues: List to append issues to.

        Returns:
            True if DOI is valid or not required.
        """
        if not citation.metadata.doi:
            issues.append(
                "doi_format: DOI required for scientific citations"
            )
            return False

        if not re.match(r"^10\.\d{4,}/[^\s]+$", citation.metadata.doi):
            issues.append(
                f"doi_format: Invalid DOI format: {citation.metadata.doi}"
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_hash(data: Any) -> str:
        """Compute SHA-256 hash for provenance.

        Args:
            data: Data to hash.

        Returns:
            Hex-encoded SHA-256 hash.
        """
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()


__all__ = [
    "VerificationEngine",
]
