# -*- coding: utf-8 -*-
"""
SARIF Generator - SEC-007

Generates SARIF 2.1.0 compliant output for GitHub Security tab integration.
Combines results from multiple scanners into a unified SARIF report.

SARIF (Static Analysis Results Interchange Format) is an OASIS standard
for representing static analysis tool results. Version 2.1.0 is supported
by GitHub Code Scanning.

Example:
    >>> from greenlang.infrastructure.security_scanning.sarif_generator import (
    ...     SARIFGenerator,
    ... )
    >>> generator = SARIFGenerator()
    >>> sarif = generator.generate(scan_report)
    >>> generator.save("/path/to/results.sarif", scan_report)

Author: GreenLang Security Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from greenlang.infrastructure.security_scanning.config import Severity
from greenlang.infrastructure.security_scanning.models import (
    ScanFinding,
    ScanReport,
    ScanResult,
)

logger = logging.getLogger(__name__)

# SARIF 2.1.0 Schema URL
SARIF_SCHEMA = "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json"
SARIF_VERSION = "2.1.0"


# ---------------------------------------------------------------------------
# SARIF Generator
# ---------------------------------------------------------------------------


class SARIFGenerator:
    """Generate SARIF 2.1.0 output from scan results.

    Creates unified SARIF reports compatible with GitHub Code Scanning,
    VS Code SARIF Viewer, and other SARIF-compliant tools.

    Attributes:
        tool_name: Name of the aggregating tool.
        tool_version: Version of the aggregating tool.
        organization: Organization name for tool info.
        include_snippets: Whether to include code snippets.
        max_snippet_length: Maximum length of code snippets.

    Example:
        >>> generator = SARIFGenerator(
        ...     tool_name="GreenLang Security Scanner",
        ...     tool_version="1.0.0"
        ... )
        >>> sarif = generator.generate(report)
    """

    def __init__(
        self,
        tool_name: str = "GreenLang Security Scanner",
        tool_version: str = "1.0.0",
        organization: str = "GreenLang",
        include_snippets: bool = True,
        max_snippet_length: int = 500,
    ) -> None:
        """Initialize SARIF generator.

        Args:
            tool_name: Name of the scanning tool.
            tool_version: Version of the tool.
            organization: Organization name.
            include_snippets: Include code snippets in results.
            max_snippet_length: Maximum snippet length.
        """
        self.tool_name = tool_name
        self.tool_version = tool_version
        self.organization = organization
        self.include_snippets = include_snippets
        self.max_snippet_length = max_snippet_length

    def generate(self, report: ScanReport) -> Dict[str, Any]:
        """Generate SARIF report from scan results.

        Creates a single SARIF run containing all findings from all scanners,
        with each scanner represented as a tool extension.

        Args:
            report: Aggregated scan report.

        Returns:
            SARIF 2.1.0 compliant dictionary.
        """
        logger.info(
            "Generating SARIF report for %d findings from %d scanners",
            len(report.all_findings),
            len(report.scanners_run),
        )

        # Build runs (one per scanner + one aggregated)
        runs = []

        # Add aggregated run with all findings
        aggregated_run = self._build_aggregated_run(report)
        runs.append(aggregated_run)

        sarif = {
            "$schema": SARIF_SCHEMA,
            "version": SARIF_VERSION,
            "runs": runs,
        }

        return sarif

    def generate_per_scanner(self, report: ScanReport) -> Dict[str, Any]:
        """Generate SARIF with separate runs per scanner.

        Creates a SARIF document with one run per scanner, preserving
        individual scanner metadata.

        Args:
            report: Aggregated scan report.

        Returns:
            SARIF 2.1.0 compliant dictionary.
        """
        runs = []

        for result in report.scan_results:
            if result.findings:
                run = self._build_scanner_run(result)
                runs.append(run)

        return {
            "$schema": SARIF_SCHEMA,
            "version": SARIF_VERSION,
            "runs": runs,
        }

    def _build_aggregated_run(self, report: ScanReport) -> Dict[str, Any]:
        """Build aggregated SARIF run from all findings.

        Args:
            report: Scan report.

        Returns:
            SARIF run dictionary.
        """
        # Collect all rules across scanners
        rules = self._build_rules(report.all_findings)

        # Build tool with extensions for each scanner
        tool = self._build_tool(report.scanners_run)

        # Build results
        results = [
            self._build_result(finding)
            for finding in report.all_findings
        ]

        # Build invocation
        invocation = self._build_invocation(report)

        run: Dict[str, Any] = {
            "tool": tool,
            "results": results,
            "invocations": [invocation],
        }

        # Add version control info if available
        if report.git_commit or report.git_branch:
            run["versionControlProvenance"] = [
                self._build_vcs_info(report)
            ]

        # Add automation details
        run["automationDetails"] = {
            "id": f"greenlang-security-scan/{report.report_id}",
            "guid": str(uuid4()),
        }

        return run

    def _build_scanner_run(self, result: ScanResult) -> Dict[str, Any]:
        """Build SARIF run for a single scanner.

        Args:
            result: Scanner result.

        Returns:
            SARIF run dictionary.
        """
        rules = self._build_rules(result.findings)

        tool = {
            "driver": {
                "name": result.scanner_name,
                "version": "unknown",  # Would need scanner version
                "rules": list(rules.values()),
            }
        }

        results = [
            self._build_result(finding)
            for finding in result.findings
        ]

        invocation = {
            "executionSuccessful": result.status.value == "completed",
            "startTimeUtc": (
                result.started_at.isoformat() if result.started_at else None
            ),
            "endTimeUtc": (
                result.completed_at.isoformat() if result.completed_at else None
            ),
        }

        if result.command:
            invocation["commandLine"] = result.command

        return {
            "tool": tool,
            "results": results,
            "invocations": [invocation],
        }

    def _build_tool(self, scanner_names: List[str]) -> Dict[str, Any]:
        """Build SARIF tool component.

        Args:
            scanner_names: List of scanner names.

        Returns:
            SARIF tool dictionary.
        """
        tool: Dict[str, Any] = {
            "driver": {
                "name": self.tool_name,
                "version": self.tool_version,
                "organization": self.organization,
                "informationUri": "https://github.com/greenlang/security-scanning",
                "rules": [],  # Will be populated per-finding
            }
        }

        # Add extensions for each scanner
        if scanner_names:
            tool["extensions"] = [
                {
                    "name": name,
                    "version": "unknown",
                }
                for name in scanner_names
            ]

        return tool

    def _build_rules(
        self, findings: List[ScanFinding]
    ) -> Dict[str, Dict[str, Any]]:
        """Build SARIF rules from findings.

        Args:
            findings: List of findings.

        Returns:
            Dictionary of rule_id to SARIF rule.
        """
        rules: Dict[str, Dict[str, Any]] = {}

        for finding in findings:
            rule_id = finding.rule_id or finding.finding_id

            if rule_id not in rules:
                rule = self._build_rule(finding)
                rules[rule_id] = rule

        return rules

    def _build_rule(self, finding: ScanFinding) -> Dict[str, Any]:
        """Build SARIF rule from finding.

        Args:
            finding: Source finding.

        Returns:
            SARIF rule dictionary.
        """
        rule: Dict[str, Any] = {
            "id": finding.rule_id or finding.finding_id,
            "name": self._sanitize_text(finding.title, 100),
            "shortDescription": {
                "text": self._sanitize_text(finding.title, 100)
            },
            "fullDescription": {
                "text": self._sanitize_text(finding.description, 1000)
            },
        }

        # Add help URI if available
        if finding.vulnerability_info and finding.vulnerability_info.references:
            rule["helpUri"] = finding.vulnerability_info.references[0]
        elif finding.vulnerability_info and finding.vulnerability_info.cve_id:
            rule["helpUri"] = (
                f"https://nvd.nist.gov/vuln/detail/{finding.vulnerability_info.cve_id}"
            )

        # Add properties
        properties: Dict[str, Any] = {
            "security-severity": str(finding.get_risk_score()),
        }

        # Add tags
        if finding.tags:
            properties["tags"] = list(finding.tags)

        # Add CWE if available
        if finding.vulnerability_info and finding.vulnerability_info.cwe_id:
            properties["cwe"] = finding.vulnerability_info.cwe_id

        rule["properties"] = properties

        return rule

    def _build_result(self, finding: ScanFinding) -> Dict[str, Any]:
        """Build SARIF result from finding.

        Args:
            finding: Source finding.

        Returns:
            SARIF result dictionary.
        """
        result: Dict[str, Any] = {
            "ruleId": finding.rule_id or finding.finding_id,
            "level": self._severity_to_level(finding.severity),
            "message": {
                "text": self._sanitize_text(finding.description, 2000)
            },
        }

        # Add locations
        if finding.location:
            result["locations"] = [finding.location.to_sarif_location()]

        # Add fingerprint for tracking
        if finding.fingerprint:
            result["fingerprints"] = {
                "greenlang/v1": finding.fingerprint,
            }

        # Add partial fingerprints for GitHub
            result["partialFingerprints"] = {
                "primaryLocationLineHash": finding.fingerprint[:16],
            }

        # Add related locations (if container location exists)
        if finding.container_location:
            result["properties"] = result.get("properties", {})
            result["properties"]["containerImage"] = finding.container_location.image_ref
            if finding.container_location.layer_digest:
                result["properties"]["layerDigest"] = finding.container_location.layer_digest

        # Add taxa (CVE reference)
        if finding.vulnerability_info and finding.vulnerability_info.cve_id:
            result["taxa"] = [
                {
                    "id": finding.vulnerability_info.cve_id,
                    "toolComponent": {"name": "CVE"},
                }
            ]

        # Add fixes if remediation available
        if finding.remediation_info and finding.remediation_info.description:
            result["fixes"] = [
                {
                    "description": {
                        "text": self._sanitize_text(
                            finding.remediation_info.description, 500
                        )
                    },
                }
            ]

        return result

    def _build_invocation(self, report: ScanReport) -> Dict[str, Any]:
        """Build SARIF invocation from report.

        Args:
            report: Scan report.

        Returns:
            SARIF invocation dictionary.
        """
        invocation: Dict[str, Any] = {
            "executionSuccessful": len(report.scanners_failed) == 0,
        }

        if report.started_at:
            invocation["startTimeUtc"] = report.started_at.isoformat()
        if report.completed_at:
            invocation["endTimeUtc"] = report.completed_at.isoformat()

        if report.scan_path:
            invocation["workingDirectory"] = {
                "uri": report.scan_path,
            }

        # Add notifications for failed scanners
        if report.scanners_failed:
            invocation["toolExecutionNotifications"] = [
                {
                    "level": "error",
                    "message": {"text": f"Scanner '{name}' failed to execute"},
                    "descriptor": {"id": f"scanner-failed-{name}"},
                }
                for name in report.scanners_failed
            ]

        return invocation

    def _build_vcs_info(self, report: ScanReport) -> Dict[str, Any]:
        """Build version control provenance.

        Args:
            report: Scan report.

        Returns:
            SARIF versionControlProvenance dictionary.
        """
        vcs: Dict[str, Any] = {}

        if report.git_commit:
            vcs["revisionId"] = report.git_commit
        if report.git_branch:
            vcs["branch"] = report.git_branch

        return vcs

    def _severity_to_level(self, severity: Severity) -> str:
        """Convert severity to SARIF level.

        SARIF levels: error, warning, note, none

        Args:
            severity: Finding severity.

        Returns:
            SARIF level string.
        """
        level_map = {
            Severity.CRITICAL: "error",
            Severity.HIGH: "error",
            Severity.MEDIUM: "warning",
            Severity.LOW: "note",
            Severity.INFO: "none",
        }
        return level_map.get(severity, "warning")

    def _sanitize_text(self, text: str, max_length: int = 1000) -> str:
        """Sanitize text for SARIF output.

        Removes control characters and truncates to max length.

        Args:
            text: Input text.
            max_length: Maximum length.

        Returns:
            Sanitized text.
        """
        if not text:
            return ""

        # Remove control characters except newlines and tabs
        import re
        sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # Truncate if needed
        if len(sanitized) > max_length:
            sanitized = sanitized[: max_length - 3] + "..."

        return sanitized

    def save(
        self,
        path: str,
        report: ScanReport,
        indent: int = 2,
    ) -> None:
        """Save SARIF report to file.

        Args:
            path: Output file path.
            report: Scan report to save.
            indent: JSON indentation level.
        """
        sarif = self.generate(report)

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sarif, f, indent=indent)

        logger.info("SARIF report saved to %s", path)

    def to_json(self, report: ScanReport, indent: int = 2) -> str:
        """Convert report to SARIF JSON string.

        Args:
            report: Scan report.
            indent: JSON indentation.

        Returns:
            SARIF JSON string.
        """
        sarif = self.generate(report)
        return json.dumps(sarif, indent=indent)


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def merge_sarif_reports(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge multiple SARIF reports into one.

    Combines runs from multiple SARIF documents.

    Args:
        reports: List of SARIF dictionaries.

    Returns:
        Merged SARIF dictionary.
    """
    if not reports:
        return {
            "$schema": SARIF_SCHEMA,
            "version": SARIF_VERSION,
            "runs": [],
        }

    all_runs = []
    for report in reports:
        all_runs.extend(report.get("runs", []))

    return {
        "$schema": SARIF_SCHEMA,
        "version": SARIF_VERSION,
        "runs": all_runs,
    }


def validate_sarif(sarif: Dict[str, Any]) -> List[str]:
    """Validate SARIF structure.

    Performs basic validation of SARIF document structure.

    Args:
        sarif: SARIF dictionary.

    Returns:
        List of validation error messages (empty if valid).
    """
    errors = []

    if not isinstance(sarif, dict):
        errors.append("SARIF must be a dictionary")
        return errors

    if sarif.get("version") != SARIF_VERSION:
        errors.append(f"Expected SARIF version {SARIF_VERSION}")

    if "runs" not in sarif:
        errors.append("Missing 'runs' array")
    elif not isinstance(sarif["runs"], list):
        errors.append("'runs' must be an array")
    else:
        for i, run in enumerate(sarif["runs"]):
            if "tool" not in run:
                errors.append(f"Run {i} missing 'tool'")
            if "results" not in run:
                errors.append(f"Run {i} missing 'results'")

    return errors
