# -*- coding: utf-8 -*-
"""
JSON Output Formatter for GL-FOUND-X-002.

This module provides a JSON output formatter for validation reports.
The output is suitable for machine processing, CI/CD integration, and
structured logging.

Features:
- Clean JSON output of ValidationReport
- Configurable indentation (pretty or compact)
- ISO 8601 timestamp formatting
- Pydantic model serialization

Example:
    >>> from greenlang.schema.cli.formatters.json_fmt import JSONFormatter
    >>> formatter = JSONFormatter(indent=2)
    >>> output = formatter.format(validation_report)
    >>> print(output)

Author: GreenLang Framework Team
Version: 0.1.0
GL-FOUND-X-002: Schema Compiler & Validator
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from greenlang.schema.models.report import ValidationReport, BatchValidationReport
from greenlang.schema.models.finding import Finding


class JSONFormatter:
    """
    JSON output formatter for validation reports.

    Produces machine-readable JSON output suitable for CI/CD pipelines,
    structured logging, and programmatic processing.

    Attributes:
        indent: Number of spaces for indentation (None for compact).
        include_timings: Whether to include timing information.
        include_normalized: Whether to include normalized payload.
        include_suggestions: Whether to include fix suggestions.

    Example:
        >>> formatter = JSONFormatter(indent=2)
        >>> print(formatter.format(report))
        {
          "valid": false,
          "schema_ref": {...},
          ...
        }
    """

    def __init__(
        self,
        indent: Optional[int] = 2,
        include_timings: bool = True,
        include_normalized: bool = True,
        include_suggestions: bool = True,
        sort_keys: bool = False,
    ) -> None:
        """
        Initialize JSONFormatter.

        Args:
            indent: Number of spaces for indentation. Use None for compact
                single-line output.
            include_timings: Whether to include timing information in output.
            include_normalized: Whether to include normalized payload in output.
            include_suggestions: Whether to include fix suggestions in output.
            sort_keys: Whether to sort keys alphabetically in output.
        """
        self.indent = indent
        self.include_timings = include_timings
        self.include_normalized = include_normalized
        self.include_suggestions = include_suggestions
        self.sort_keys = sort_keys

    def format(self, report: Union[ValidationReport, BatchValidationReport]) -> str:
        """
        Format validation report as JSON.

        Args:
            report: ValidationReport or BatchValidationReport to format.

        Returns:
            JSON string representation of the report.

        Example:
            >>> output = formatter.format(report)
            >>> data = json.loads(output)
            >>> print(data["valid"])
        """
        if isinstance(report, BatchValidationReport):
            return self._format_batch_report(report)
        return self._format_single_report(report)

    def _format_single_report(self, report: ValidationReport) -> str:
        """Format a single validation report as JSON."""
        data = self._build_report_dict(report)
        return self._serialize(data)

    def _format_batch_report(self, report: BatchValidationReport) -> str:
        """Format a batch validation report as JSON."""
        data = self._build_batch_dict(report)
        return self._serialize(data)

    def _build_report_dict(self, report: ValidationReport) -> Dict[str, Any]:
        """
        Build dictionary representation of ValidationReport.

        Args:
            report: ValidationReport to convert.

        Returns:
            Dictionary suitable for JSON serialization.
        """
        result: Dict[str, Any] = {
            "valid": report.valid,
            "schema_ref": self._serialize_model(report.schema_ref),
            "schema_hash": report.schema_hash,
            "summary": {
                "valid": report.summary.valid,
                "error_count": report.summary.error_count,
                "warning_count": report.summary.warning_count,
                "info_count": report.summary.info_count,
                "total_findings": report.summary.total_findings(),
            },
            "findings": [
                self._serialize_finding(f) for f in report.findings
            ],
        }

        # Optional fields
        if self.include_normalized and report.normalized_payload is not None:
            result["normalized_payload"] = report.normalized_payload

        if self.include_suggestions and report.fix_suggestions is not None:
            result["fix_suggestions"] = [
                self._serialize_model(s) for s in report.fix_suggestions
            ]

        if self.include_timings:
            result["timings_ms"] = report.timings.to_dict()

        return result

    def _build_batch_dict(self, report: BatchValidationReport) -> Dict[str, Any]:
        """
        Build dictionary representation of BatchValidationReport.

        Args:
            report: BatchValidationReport to convert.

        Returns:
            Dictionary suitable for JSON serialization.
        """
        result: Dict[str, Any] = {
            "schema_ref": self._serialize_model(report.schema_ref),
            "schema_hash": report.schema_hash,
            "summary": {
                "total_items": report.summary.total_items,
                "valid_count": report.summary.valid_count,
                "error_count": report.summary.error_count,
                "warning_count": report.summary.warning_count,
                "success_rate": round(report.summary.success_rate(), 2),
            },
            "results": [
                self._serialize_item_result(r) for r in report.results
            ],
        }

        return result

    def _serialize_finding(self, finding: Finding) -> Dict[str, Any]:
        """
        Serialize a Finding to dictionary.

        Args:
            finding: Finding to serialize.

        Returns:
            Dictionary representation of the finding.
        """
        result: Dict[str, Any] = {
            "code": finding.code,
            "severity": finding.severity.value,
            "path": finding.path,
            "message": finding.message,
        }

        # Only include non-None optional fields
        if finding.expected is not None:
            result["expected"] = finding.expected

        if finding.actual is not None:
            result["actual"] = finding.actual

        if finding.hint is not None:
            result["hint"] = {
                "category": finding.hint.category,
            }
            if finding.hint.suggested_values:
                result["hint"]["suggested_values"] = finding.hint.suggested_values
            if finding.hint.docs_url:
                result["hint"]["docs_url"] = finding.hint.docs_url

        return result

    def _serialize_item_result(self, item: Any) -> Dict[str, Any]:
        """
        Serialize an ItemResult to dictionary.

        Args:
            item: ItemResult to serialize.

        Returns:
            Dictionary representation of the item result.
        """
        result: Dict[str, Any] = {
            "index": item.index,
            "valid": item.valid,
            "findings": [
                self._serialize_finding(f) for f in item.findings
            ],
        }

        if item.id is not None:
            result["id"] = item.id

        if self.include_normalized and item.normalized_payload is not None:
            result["normalized_payload"] = item.normalized_payload

        if self.include_suggestions and item.fix_suggestions is not None:
            result["fix_suggestions"] = [
                self._serialize_model(s) for s in item.fix_suggestions
            ]

        return result

    def _serialize_model(self, model: BaseModel) -> Dict[str, Any]:
        """
        Serialize a Pydantic model to dictionary.

        Args:
            model: Pydantic model to serialize.

        Returns:
            Dictionary representation of the model.
        """
        return model.model_dump(mode="json", exclude_none=True)

    def _serialize(self, data: Dict[str, Any]) -> str:
        """
        Serialize dictionary to JSON string.

        Args:
            data: Dictionary to serialize.

        Returns:
            JSON string.
        """
        return json.dumps(
            data,
            indent=self.indent,
            sort_keys=self.sort_keys,
            ensure_ascii=False,
            default=self._json_serializer,
        )

    def _json_serializer(self, obj: Any) -> Any:
        """
        Custom JSON serializer for non-standard types.

        Args:
            obj: Object to serialize.

        Returns:
            Serializable representation.

        Raises:
            TypeError: If object is not serializable.
        """
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, BaseModel):
            return obj.model_dump(mode="json", exclude_none=True)
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class CompactJSONFormatter(JSONFormatter):
    """
    Compact JSON formatter with no indentation.

    Suitable for log ingestion systems and environments where
    file size is a concern.

    Example:
        >>> formatter = CompactJSONFormatter()
        >>> output = formatter.format(report)
        >>> # Output is single line
    """

    def __init__(
        self,
        include_timings: bool = True,
        include_normalized: bool = False,
        include_suggestions: bool = False,
    ) -> None:
        """
        Initialize CompactJSONFormatter.

        Args:
            include_timings: Whether to include timing information.
            include_normalized: Whether to include normalized payload.
            include_suggestions: Whether to include fix suggestions.
        """
        super().__init__(
            indent=None,
            include_timings=include_timings,
            include_normalized=include_normalized,
            include_suggestions=include_suggestions,
            sort_keys=False,
        )


class NDJSONFormatter:
    """
    Newline-delimited JSON (NDJSON) formatter for streaming output.

    Each finding is output as a separate JSON line, suitable for
    streaming processing and log aggregation systems.

    Example:
        >>> formatter = NDJSONFormatter()
        >>> for line in formatter.format_lines(report):
        ...     print(line)
    """

    def __init__(self, include_metadata: bool = True) -> None:
        """
        Initialize NDJSONFormatter.

        Args:
            include_metadata: Whether to include metadata line at start.
        """
        self.include_metadata = include_metadata

    def format(self, report: Union[ValidationReport, BatchValidationReport]) -> str:
        """
        Format validation report as NDJSON.

        Args:
            report: ValidationReport or BatchValidationReport to format.

        Returns:
            NDJSON string with one JSON object per line.
        """
        return "\n".join(self.format_lines(report))

    def format_lines(
        self,
        report: Union[ValidationReport, BatchValidationReport]
    ) -> List[str]:
        """
        Generate NDJSON lines for a report.

        Args:
            report: ValidationReport or BatchValidationReport to format.

        Returns:
            List of JSON strings, one per line.
        """
        lines: List[str] = []

        if isinstance(report, BatchValidationReport):
            return self._format_batch_lines(report)

        # Metadata line (if enabled)
        if self.include_metadata:
            metadata = {
                "type": "validation_report",
                "valid": report.valid,
                "schema_ref": str(report.schema_ref),
                "schema_hash": report.schema_hash,
                "error_count": report.summary.error_count,
                "warning_count": report.summary.warning_count,
                "total_ms": report.timings.total_ms,
            }
            lines.append(json.dumps(metadata, ensure_ascii=False))

        # One line per finding
        for finding in report.findings:
            finding_dict = {
                "type": "finding",
                "code": finding.code,
                "severity": finding.severity.value,
                "path": finding.path,
                "message": finding.message,
            }
            if finding.expected is not None:
                finding_dict["expected"] = finding.expected
            if finding.actual is not None:
                finding_dict["actual"] = finding.actual

            lines.append(json.dumps(finding_dict, ensure_ascii=False))

        return lines

    def _format_batch_lines(self, report: BatchValidationReport) -> List[str]:
        """Format batch report as NDJSON lines."""
        lines: List[str] = []

        # Metadata line
        if self.include_metadata:
            metadata = {
                "type": "batch_validation_report",
                "schema_ref": str(report.schema_ref),
                "schema_hash": report.schema_hash,
                "total_items": report.summary.total_items,
                "valid_count": report.summary.valid_count,
                "error_count": report.summary.error_count,
            }
            lines.append(json.dumps(metadata, ensure_ascii=False))

        # One line per item result
        for item in report.results:
            item_dict = {
                "type": "item_result",
                "index": item.index,
                "valid": item.valid,
                "error_count": item.error_count(),
                "warning_count": item.warning_count(),
            }
            if item.id is not None:
                item_dict["id"] = item.id

            lines.append(json.dumps(item_dict, ensure_ascii=False))

            # Add findings for invalid items
            if not item.valid:
                for finding in item.findings:
                    finding_dict = {
                        "type": "finding",
                        "item_index": item.index,
                        "code": finding.code,
                        "severity": finding.severity.value,
                        "path": finding.path,
                        "message": finding.message,
                    }
                    lines.append(json.dumps(finding_dict, ensure_ascii=False))

        return lines


def format_json(
    report: Union[ValidationReport, BatchValidationReport],
    indent: Optional[int] = 2,
    include_timings: bool = True,
    include_normalized: bool = True,
    include_suggestions: bool = True,
    max_findings: int = 100,
) -> str:
    """
    Format validation report as JSON.

    Convenience function for creating a JSONFormatter and formatting
    a report in one call.

    Args:
        report: ValidationReport or BatchValidationReport to format.
        indent: Number of spaces for indentation (None for compact).
        include_timings: Whether to include timing information.
        include_normalized: Whether to include normalized payload.
        include_suggestions: Whether to include fix suggestions.
        max_findings: Maximum number of findings to include.

    Returns:
        JSON string representation of the report.

    Example:
        >>> output = format_json(report, indent=2)
        >>> data = json.loads(output)
    """
    formatter = JSONFormatter(
        indent=indent,
        include_timings=include_timings,
        include_normalized=include_normalized,
        include_suggestions=include_suggestions,
    )
    return formatter.format(report)


__all__ = [
    "JSONFormatter",
    "CompactJSONFormatter",
    "NDJSONFormatter",
    "format_json",
]
