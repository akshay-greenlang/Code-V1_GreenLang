# -*- coding: utf-8 -*-
"""
License compliance scanner (F025).

Scans all factors in an edition for license violations:
- connector_only source but redistribution_allowed: true -> ERROR
- Missing citation_text for attribution_required sources -> WARNING
- certified status but source requires legal signoff that is missing -> BLOCK
- Deprecated factors still marked as redistribution_allowed -> WARNING
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from greenlang.factors.source_registry import SourceRegistryEntry, registry_by_id

logger = logging.getLogger(__name__)


class LicenseSeverity:
    ERROR = "error"
    WARNING = "warning"
    BLOCK = "block"
    INFO = "info"


@dataclass
class LicenseIssue:
    """A single license compliance issue."""

    factor_id: str
    source_id: str
    severity: str  # error | warning | block | info
    rule_id: str
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "source_id": self.source_id,
            "severity": self.severity,
            "rule_id": self.rule_id,
            "message": self.message,
        }


@dataclass
class LicenseScanReport:
    """Full license compliance scan report."""

    edition_id: str
    total_factors: int = 0
    total_issues: int = 0
    errors: int = 0
    warnings: int = 0
    blocks: int = 0
    infos: int = 0
    issues: List[LicenseIssue] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edition_id": self.edition_id,
            "total_factors": self.total_factors,
            "total_issues": self.total_issues,
            "errors": self.errors,
            "warnings": self.warnings,
            "blocks": self.blocks,
            "infos": self.infos,
            "compliant": self.compliant,
            "release_blocked": self.release_blocked,
            "issues": [i.to_dict() for i in self.issues],
        }

    @property
    def compliant(self) -> bool:
        """True if no errors or blocks."""
        return self.errors == 0 and self.blocks == 0

    @property
    def release_blocked(self) -> bool:
        """True if any blocking issues exist."""
        return self.blocks > 0

    def _add_issue(self, issue: LicenseIssue) -> None:
        self.issues.append(issue)
        self.total_issues += 1
        if issue.severity == LicenseSeverity.ERROR:
            self.errors += 1
        elif issue.severity == LicenseSeverity.WARNING:
            self.warnings += 1
        elif issue.severity == LicenseSeverity.BLOCK:
            self.blocks += 1
        elif issue.severity == LicenseSeverity.INFO:
            self.infos += 1


def _get_factor_field(factor: Any, key: str, default: Any = "") -> Any:
    """Get a field from a factor (dict or record)."""
    if isinstance(factor, dict):
        return factor.get(key, default)
    return getattr(factor, key, default)


def _get_nested(factor: Any, *keys: str, default: Any = "") -> Any:
    """Get a nested field from a factor."""
    obj = factor
    for k in keys:
        if isinstance(obj, dict):
            obj = obj.get(k, default)
        else:
            obj = getattr(obj, k, default)
        if obj is None or obj == default:
            return default
    return obj


def scan_license_compliance(
    factors: Sequence[Any],
    *,
    edition_id: str = "unknown",
    registry: Optional[Dict[str, SourceRegistryEntry]] = None,
) -> LicenseScanReport:
    """
    Scan all factors for license compliance violations.

    Rules:
        L01: connector_only source has redistribution_allowed on factor -> ERROR
        L02: attribution_required source but factor missing citation_text -> WARNING
        L03: certified status but source requires legal signoff (missing) -> BLOCK
        L04: deprecated factor still marked redistribution_allowed -> WARNING
        L05: unknown source_id (not in registry) -> WARNING
        L06: factor license_info.redistribution_allowed but registry says no -> ERROR

    Args:
        factors: List of factor dicts or EmissionFactorRecord objects.
        edition_id: Edition label for reporting.
        registry: Source registry dict (loaded from YAML if None).

    Returns:
        LicenseScanReport with all issues found.
    """
    report = LicenseScanReport(edition_id=edition_id, total_factors=len(factors))
    reg = registry if registry is not None else registry_by_id()

    for factor in factors:
        fid = str(_get_factor_field(factor, "factor_id", "?"))

        # Determine source_id
        sid = str(_get_factor_field(factor, "source_id", ""))
        if not sid:
            parts = fid.split(":")
            if len(parts) >= 2:
                sid = parts[1].lower()

        factor_status = str(_get_factor_field(factor, "factor_status", "certified")).lower()

        # Get license_info
        if isinstance(factor, dict):
            license_info = factor.get("license_info") or {}
        else:
            li = getattr(factor, "license_info", None)
            if li is not None and not isinstance(li, dict):
                license_info = {
                    "redistribution_allowed": getattr(li, "redistribution_allowed", False),
                    "attribution_required": getattr(li, "attribution_required", False),
                    "citation_text": getattr(li, "citation_text", ""),
                }
            else:
                license_info = li or {}

        redist_allowed = bool(license_info.get("redistribution_allowed", False))

        # Look up registry entry
        entry = reg.get(sid)

        # L05: Unknown source
        if not entry and sid:
            report._add_issue(LicenseIssue(
                factor_id=fid,
                source_id=sid,
                severity=LicenseSeverity.WARNING,
                rule_id="L05",
                message=f"Source '{sid}' not found in registry",
            ))
            continue

        if not entry:
            continue

        # L01: connector_only + redistribution_allowed
        if entry.connector_only and redist_allowed:
            report._add_issue(LicenseIssue(
                factor_id=fid,
                source_id=sid,
                severity=LicenseSeverity.ERROR,
                rule_id="L01",
                message=f"connector_only source '{sid}' has redistribution_allowed=true on factor",
            ))

        # L02: attribution_required but missing citation
        if entry.attribution_required:
            citation = str(license_info.get("citation_text", "")).strip()
            if not citation and not entry.citation_text.strip():
                report._add_issue(LicenseIssue(
                    factor_id=fid,
                    source_id=sid,
                    severity=LicenseSeverity.WARNING,
                    rule_id="L02",
                    message=f"Source '{sid}' requires attribution but citation_text is missing",
                ))

        # L03: certified but pending legal signoff
        if factor_status == "certified" and entry.approval_required_for_certified:
            art = entry.legal_signoff_artifact
            if not art or not str(art).strip():
                report._add_issue(LicenseIssue(
                    factor_id=fid,
                    source_id=sid,
                    severity=LicenseSeverity.BLOCK,
                    rule_id="L03",
                    message=f"Factor certified but source '{sid}' requires legal signoff (missing artifact)",
                ))

        # L04: deprecated + redistribution_allowed
        if factor_status == "deprecated" and redist_allowed:
            report._add_issue(LicenseIssue(
                factor_id=fid,
                source_id=sid,
                severity=LicenseSeverity.WARNING,
                rule_id="L04",
                message="Deprecated factor still has redistribution_allowed=true",
            ))

        # L06: factor says redistribution allowed but registry says no
        if redist_allowed and not entry.redistribution_allowed and not entry.connector_only:
            report._add_issue(LicenseIssue(
                factor_id=fid,
                source_id=sid,
                severity=LicenseSeverity.ERROR,
                rule_id="L06",
                message=f"Factor redistribution_allowed=true but source '{sid}' registry does not allow redistribution",
            ))

    logger.info(
        "License scan: edition=%s factors=%d issues=%d errors=%d warnings=%d blocks=%d compliant=%s",
        edition_id, report.total_factors, report.total_issues,
        report.errors, report.warnings, report.blocks, report.compliant,
    )
    return report
