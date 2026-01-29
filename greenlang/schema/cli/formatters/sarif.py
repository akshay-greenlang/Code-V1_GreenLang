# -*- coding: utf-8 -*-
"""
SARIF Output Formatter for GL-FOUND-X-002.

This module provides a SARIF (Static Analysis Results Interchange Format)
output formatter for validation reports. SARIF 2.1.0 is compatible with:
- VS Code (via SARIF Viewer extension)
- GitHub Code Scanning
- Azure DevOps
- Other IDE and CI/CD tools

SARIF Specification: https://sarifweb.azurewebsites.net/

Features:
- Full SARIF 2.1.0 compliance
- Rule definitions with help URLs
- Location information with region details
- Fix suggestions as SARIF fixes
- Configurable artifact URIs

Example:
    >>> from greenlang.schema.cli.formatters.sarif import SARIFFormatter
    >>> formatter = SARIFFormatter(tool_name="greenlang-validator")
    >>> output = formatter.format(validation_report, source_file="data.yaml")
    >>> print(output)

Author: GreenLang Framework Team
Version: 0.1.0
GL-FOUND-X-002: Schema Compiler & Validator
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from greenlang.schema.models.report import ValidationReport, BatchValidationReport
from greenlang.schema.models.finding import Finding, Severity
from greenlang.schema.errors import ERROR_REGISTRY, ErrorCode, get_error_by_code


# SARIF severity levels mapping
SARIF_SEVERITY_MAP: Dict[Severity, str] = {
    Severity.ERROR: "error",
    Severity.WARNING: "warning",
    Severity.INFO: "note",
}

# SARIF level for different severities
SARIF_LEVEL_MAP: Dict[Severity, str] = {
    Severity.ERROR: "error",
    Severity.WARNING: "warning",
    Severity.INFO: "note",
}

# Tool information
TOOL_NAME = "greenlang-schema-validator"
TOOL_VERSION = "0.1.0"
TOOL_INFORMATION_URI = "https://docs.greenlang.dev/schema-validator"
SARIF_VERSION = "2.1.0"
SARIF_SCHEMA = "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json"


class SARIFFormatter:
    """
    SARIF 2.1.0 output formatter for validation reports.

    Produces SARIF-compliant JSON output suitable for integration with
    VS Code, GitHub Code Scanning, and other static analysis tools.

    Attributes:
        tool_name: Name of the analysis tool.
        tool_version: Version of the analysis tool.
        include_fixes: Whether to include fix suggestions.
        base_uri: Base URI for artifact locations.

    Example:
        >>> formatter = SARIFFormatter()
        >>> output = formatter.format(report, source_file="data.yaml")
        >>> # Output is SARIF 2.1.0 compliant JSON
    """

    def __init__(
        self,
        tool_name: str = TOOL_NAME,
        tool_version: str = TOOL_VERSION,
        include_fixes: bool = True,
        base_uri: Optional[str] = None,
        information_uri: str = TOOL_INFORMATION_URI,
    ) -> None:
        """
        Initialize SARIFFormatter.

        Args:
            tool_name: Name of the analysis tool.
            tool_version: Version of the analysis tool.
            include_fixes: Whether to include fix suggestions as SARIF fixes.
            base_uri: Base URI for artifact locations. If None, uses relative paths.
            information_uri: URL for tool documentation.
        """
        self.tool_name = tool_name
        self.tool_version = tool_version
        self.include_fixes = include_fixes
        self.base_uri = base_uri
        self.information_uri = information_uri

    def format(
        self,
        report: Union[ValidationReport, BatchValidationReport],
        source_file: Optional[str] = None,
        indent: Optional[int] = 2,
    ) -> str:
        """
        Format validation report as SARIF JSON.

        Args:
            report: ValidationReport or BatchValidationReport to format.
            source_file: Path to the source file being validated.
            indent: JSON indentation (None for compact).

        Returns:
            SARIF 2.1.0 compliant JSON string.

        Example:
            >>> output = formatter.format(report, source_file="data.yaml")
            >>> sarif_data = json.loads(output)
        """
        sarif_log = self._build_sarif_log(report, source_file)
        return json.dumps(sarif_log, indent=indent, ensure_ascii=False)

    def _build_sarif_log(
        self,
        report: Union[ValidationReport, BatchValidationReport],
        source_file: Optional[str]
    ) -> Dict[str, Any]:
        """
        Build SARIF log structure.

        Args:
            report: Validation report to convert.
            source_file: Source file path.

        Returns:
            SARIF log dictionary.
        """
        # Handle batch reports by combining findings
        if isinstance(report, BatchValidationReport):
            findings = []
            for item in report.results:
                for finding in item.findings:
                    findings.append((finding, item.id or f"item_{item.index}"))
            schema_ref = report.schema_ref
        else:
            findings = [(f, source_file) for f in report.findings]
            schema_ref = report.schema_ref

        # Build the SARIF log
        sarif_log = {
            "$schema": SARIF_SCHEMA,
            "version": SARIF_VERSION,
            "runs": [
                {
                    "tool": self._build_tool_component(schema_ref),
                    "results": [
                        self._build_result(finding, file_ref)
                        for finding, file_ref in findings
                    ],
                    "artifacts": self._build_artifacts(source_file, report),
                    "invocations": [
                        self._build_invocation(report)
                    ],
                }
            ],
        }

        return sarif_log

    def _build_tool_component(self, schema_ref: Any) -> Dict[str, Any]:
        """
        Build SARIF tool component.

        Args:
            schema_ref: Schema reference for context.

        Returns:
            SARIF tool dictionary.
        """
        return {
            "driver": {
                "name": self.tool_name,
                "version": self.tool_version,
                "informationUri": self.information_uri,
                "rules": self._build_rules(),
                "properties": {
                    "schema": str(schema_ref) if schema_ref else None,
                }
            }
        }

    def _build_rules(self) -> List[Dict[str, Any]]:
        """
        Build SARIF rule definitions from error registry.

        Returns:
            List of SARIF rule definitions.
        """
        rules: List[Dict[str, Any]] = []

        for error_code, error_info in ERROR_REGISTRY.items():
            rule = {
                "id": error_info.code,
                "name": error_info.name,
                "shortDescription": {
                    "text": error_info.message_template.split("{")[0].strip()
                },
                "fullDescription": {
                    "text": error_info.message_template
                },
                "defaultConfiguration": {
                    "level": SARIF_LEVEL_MAP.get(
                        error_info.severity,
                        "warning"
                    )
                },
                "properties": {
                    "category": error_info.category.value,
                }
            }

            # Add help URL if available
            if error_info.documentation_url:
                rule["helpUri"] = error_info.documentation_url
            else:
                # Generate default help URL
                rule["helpUri"] = (
                    f"{self.information_uri}/errors/{error_info.code}"
                )

            # Add hint as help text
            if error_info.hint_template:
                rule["help"] = {
                    "text": error_info.hint_template
                }

            rules.append(rule)

        return rules

    def _build_result(
        self,
        finding: Finding,
        file_ref: Optional[str]
    ) -> Dict[str, Any]:
        """
        Build SARIF result for a finding.

        Args:
            finding: Finding to convert.
            file_ref: File reference or identifier.

        Returns:
            SARIF result dictionary.
        """
        result: Dict[str, Any] = {
            "ruleId": finding.code,
            "level": SARIF_LEVEL_MAP.get(finding.severity, "warning"),
            "message": {
                "text": finding.message
            },
            "locations": [
                self._build_location(finding, file_ref)
            ],
        }

        # Add kind based on error category
        if finding.is_error():
            result["kind"] = "fail"
        else:
            result["kind"] = "informational"

        # Add properties with expected/actual values
        properties: Dict[str, Any] = {
            "path": finding.path,
        }
        if finding.expected is not None:
            properties["expected"] = finding.expected
        if finding.actual is not None:
            properties["actual"] = finding.actual

        result["properties"] = properties

        # Add related locations for hints
        if finding.hint and finding.hint.suggested_values:
            result["relatedLocations"] = [
                {
                    "id": 0,
                    "message": {
                        "text": f"Suggested values: {', '.join(str(v) for v in finding.hint.suggested_values[:5])}"
                    },
                    "physicalLocation": self._build_physical_location(file_ref)
                }
            ]

        return result

    def _build_location(
        self,
        finding: Finding,
        file_ref: Optional[str]
    ) -> Dict[str, Any]:
        """
        Build SARIF location for a finding.

        Args:
            finding: Finding with path information.
            file_ref: File reference.

        Returns:
            SARIF location dictionary.
        """
        location: Dict[str, Any] = {
            "physicalLocation": self._build_physical_location(file_ref),
        }

        # Add logical location (JSON path)
        if finding.path:
            location["logicalLocations"] = [
                {
                    "fullyQualifiedName": finding.path,
                    "kind": "jsonPointer",
                }
            ]

        return location

    def _build_physical_location(
        self,
        file_ref: Optional[str]
    ) -> Dict[str, Any]:
        """
        Build SARIF physical location.

        Args:
            file_ref: File reference or path.

        Returns:
            SARIF physical location dictionary.
        """
        physical_location: Dict[str, Any] = {}

        if file_ref:
            artifact_location: Dict[str, Any] = {
                "uri": file_ref
            }
            if self.base_uri:
                artifact_location["uriBaseId"] = "%SRCROOT%"

            physical_location["artifactLocation"] = artifact_location

        return physical_location

    def _build_artifacts(
        self,
        source_file: Optional[str],
        report: Union[ValidationReport, BatchValidationReport]
    ) -> List[Dict[str, Any]]:
        """
        Build SARIF artifacts list.

        Args:
            source_file: Source file path.
            report: Validation report.

        Returns:
            List of SARIF artifact dictionaries.
        """
        artifacts: List[Dict[str, Any]] = []

        if source_file:
            artifact: Dict[str, Any] = {
                "location": {
                    "uri": source_file
                },
                "roles": ["analysisTarget"],
            }

            # Detect MIME type from extension
            if source_file.endswith(".yaml") or source_file.endswith(".yml"):
                artifact["mimeType"] = "application/x-yaml"
            elif source_file.endswith(".json"):
                artifact["mimeType"] = "application/json"

            artifacts.append(artifact)

        return artifacts

    def _build_invocation(
        self,
        report: Union[ValidationReport, BatchValidationReport]
    ) -> Dict[str, Any]:
        """
        Build SARIF invocation information.

        Args:
            report: Validation report.

        Returns:
            SARIF invocation dictionary.
        """
        # Determine execution success
        if isinstance(report, BatchValidationReport):
            execution_successful = report.summary.error_count == 0
        else:
            execution_successful = report.valid

        invocation: Dict[str, Any] = {
            "executionSuccessful": execution_successful,
            "endTimeUtc": datetime.now(timezone.utc).isoformat(),
        }

        # Add timing information
        if isinstance(report, ValidationReport) and report.timings:
            invocation["properties"] = {
                "totalTimeMs": report.timings.total_ms
            }

        return invocation


class SARIFFixFormatter:
    """
    Extended SARIF formatter that includes fix suggestions.

    Converts GreenLang fix suggestions to SARIF fix objects that can
    be applied automatically by supporting tools.

    Example:
        >>> formatter = SARIFFixFormatter()
        >>> output = formatter.format(report, source_file="data.yaml")
        >>> # Fixes can be applied via VS Code Quick Fix
    """

    def __init__(
        self,
        tool_name: str = TOOL_NAME,
        tool_version: str = TOOL_VERSION,
        max_fixes_per_result: int = 5,
    ) -> None:
        """
        Initialize SARIFFixFormatter.

        Args:
            tool_name: Name of the analysis tool.
            tool_version: Version of the analysis tool.
            max_fixes_per_result: Maximum fixes to include per result.
        """
        self.base_formatter = SARIFFormatter(
            tool_name=tool_name,
            tool_version=tool_version,
            include_fixes=True,
        )
        self.max_fixes_per_result = max_fixes_per_result

    def format(
        self,
        report: ValidationReport,
        source_file: Optional[str] = None,
        indent: Optional[int] = 2,
    ) -> str:
        """
        Format validation report as SARIF JSON with fixes.

        Args:
            report: ValidationReport to format.
            source_file: Path to the source file being validated.
            indent: JSON indentation (None for compact).

        Returns:
            SARIF 2.1.0 compliant JSON string with fixes.
        """
        sarif_log = self.base_formatter._build_sarif_log(report, source_file)

        # Add fixes to results if available
        if report.fix_suggestions:
            self._add_fixes_to_results(sarif_log, report, source_file)

        return json.dumps(sarif_log, indent=indent, ensure_ascii=False)

    def _add_fixes_to_results(
        self,
        sarif_log: Dict[str, Any],
        report: ValidationReport,
        source_file: Optional[str]
    ) -> None:
        """
        Add fix suggestions to SARIF results.

        Args:
            sarif_log: SARIF log to modify.
            report: Validation report with fix suggestions.
            source_file: Source file path.
        """
        if not report.fix_suggestions:
            return

        results = sarif_log.get("runs", [{}])[0].get("results", [])

        # Create mapping from path to fix suggestions
        path_to_fixes: Dict[str, List[Any]] = {}
        for suggestion in report.fix_suggestions:
            for path in suggestion.affected_paths():
                if path not in path_to_fixes:
                    path_to_fixes[path] = []
                path_to_fixes[path].append(suggestion)

        # Add fixes to matching results
        for result in results:
            result_path = result.get("properties", {}).get("path", "")
            if result_path in path_to_fixes:
                fixes = self._build_fixes(
                    path_to_fixes[result_path][:self.max_fixes_per_result],
                    source_file
                )
                if fixes:
                    result["fixes"] = fixes

    def _build_fixes(
        self,
        suggestions: List[Any],
        source_file: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Build SARIF fixes from fix suggestions.

        Args:
            suggestions: List of FixSuggestion objects.
            source_file: Source file path.

        Returns:
            List of SARIF fix dictionaries.
        """
        fixes: List[Dict[str, Any]] = []

        for suggestion in suggestions:
            fix: Dict[str, Any] = {
                "description": {
                    "text": suggestion.rationale
                },
                "artifactChanges": [],
            }

            # Convert patches to artifact changes
            for patch_op in suggestion.patch:
                change = self._build_artifact_change(patch_op, source_file)
                if change:
                    fix["artifactChanges"].append(change)

            if fix["artifactChanges"]:
                fixes.append(fix)

        return fixes

    def _build_artifact_change(
        self,
        patch_op: Any,
        source_file: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Build SARIF artifact change from JSON patch operation.

        Args:
            patch_op: JSONPatchOp object.
            source_file: Source file path.

        Returns:
            SARIF artifact change dictionary, or None if not applicable.
        """
        if not source_file:
            return None

        change: Dict[str, Any] = {
            "artifactLocation": {
                "uri": source_file
            },
            "replacements": [],
        }

        # For SARIF, we describe the change as a replacement
        # Note: Full JSON patch to SARIF conversion is complex;
        # this is a simplified version
        replacement: Dict[str, Any] = {
            "deletedRegion": {
                "startLine": 1,  # Would need source mapping for accurate location
            },
            "insertedContent": {
                "text": f"[{patch_op.op}] {patch_op.path}"
            }
        }

        change["replacements"].append(replacement)

        return change


def format_sarif(
    report: Union[ValidationReport, BatchValidationReport],
    source_file: Optional[str] = None,
    source_name: Optional[str] = None,
    indent: Optional[int] = 2,
    include_fixes: bool = False,
) -> str:
    """
    Format validation report as SARIF.

    Convenience function for creating a SARIFFormatter and formatting
    a report in one call.

    Args:
        report: ValidationReport or BatchValidationReport to format.
        source_file: Path to the source file being validated.
        source_name: Alternative name for source (alias for source_file).
        indent: JSON indentation (None for compact).
        include_fixes: Whether to include fix suggestions.

    Returns:
        SARIF 2.1.0 compliant JSON string.

    Example:
        >>> output = format_sarif(report, source_file="data.yaml")
        >>> # Can be used with VS Code SARIF Viewer
    """
    # Use source_name as fallback for source_file
    effective_source = source_file or source_name

    if include_fixes and isinstance(report, ValidationReport):
        formatter = SARIFFixFormatter()
        return formatter.format(report, source_file=effective_source, indent=indent)

    formatter = SARIFFormatter()
    return formatter.format(report, source_file=effective_source, indent=indent)


__all__ = [
    "SARIFFormatter",
    "SARIFFixFormatter",
    "format_sarif",
    "SARIF_VERSION",
    "SARIF_SCHEMA",
]
