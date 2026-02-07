# -*- coding: utf-8 -*-
"""
Audit Export Formatters - SEC-005

Export formatters for different output formats:
- CSV: Comma-separated values with streaming support
- JSON: JSON Lines (JSONL) format for large datasets
- Parquet: Columnar format with compression (requires PyArrow)

Author: GreenLang Framework Team
Date: February 2026
"""

from __future__ import annotations

import abc
import csv
import gzip
import io
import json
import logging
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional

logger = logging.getLogger(__name__)


class BaseExporter(abc.ABC):
    """Abstract base class for audit event exporters."""

    def __init__(
        self,
        compress: bool = False,
        include_metadata: bool = True,
    ):
        """Initialize the exporter.

        Args:
            compress: Whether to gzip compress output.
            include_metadata: Whether to include event metadata fields.
        """
        self.compress = compress
        self.include_metadata = include_metadata

    @property
    @abc.abstractmethod
    def content_type(self) -> str:
        """Get the MIME content type for this format."""
        pass

    @property
    @abc.abstractmethod
    def file_extension(self) -> str:
        """Get the file extension for this format."""
        pass

    @abc.abstractmethod
    async def export(
        self,
        events: AsyncIterator[Dict[str, Any]],
        output: io.BytesIO,
    ) -> int:
        """Export events to the output buffer.

        Args:
            events: Async iterator of audit events.
            output: Output buffer to write to.

        Returns:
            Number of events exported.
        """
        pass

    def _serialize_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize an event for export.

        Args:
            event: Raw event data.

        Returns:
            Serialized event data.
        """
        serialized = {}

        for key, value in event.items():
            # Skip metadata if not included
            if key == "metadata" and not self.include_metadata:
                continue

            # Convert datetime objects
            if isinstance(value, datetime):
                serialized[key] = value.isoformat()
            # Convert UUIDs
            elif hasattr(value, "hex"):
                serialized[key] = str(value)
            # Convert enums
            elif hasattr(value, "value"):
                serialized[key] = value.value
            # Convert nested dicts
            elif isinstance(value, dict):
                serialized[key] = json.dumps(value)
            # Convert lists
            elif isinstance(value, list):
                serialized[key] = json.dumps(value)
            else:
                serialized[key] = value

        return serialized


class CSVExporter(BaseExporter):
    """Export audit events to CSV format."""

    def __init__(
        self,
        compress: bool = False,
        include_metadata: bool = True,
        delimiter: str = ",",
        quoting: int = csv.QUOTE_MINIMAL,
    ):
        """Initialize the CSV exporter.

        Args:
            compress: Whether to gzip compress output.
            include_metadata: Whether to include metadata fields.
            delimiter: Field delimiter character.
            quoting: CSV quoting style.
        """
        super().__init__(compress, include_metadata)
        self.delimiter = delimiter
        self.quoting = quoting

    @property
    def content_type(self) -> str:
        """Get the MIME content type."""
        if self.compress:
            return "application/gzip"
        return "text/csv"

    @property
    def file_extension(self) -> str:
        """Get the file extension."""
        if self.compress:
            return "csv.gz"
        return "csv"

    async def export(
        self,
        events: AsyncIterator[Dict[str, Any]],
        output: io.BytesIO,
    ) -> int:
        """Export events to CSV.

        Args:
            events: Async iterator of audit events.
            output: Output buffer.

        Returns:
            Number of events exported.
        """
        # Define column order
        columns = [
            "id", "performed_at", "category", "severity", "event_type",
            "operation", "outcome", "user_id", "user_email", "organization_id",
            "resource_type", "resource_path", "action", "ip_address",
            "change_summary", "error_message", "tags",
        ]

        if self.include_metadata:
            columns.append("metadata")

        # Create string buffer for CSV writing
        string_buffer = io.StringIO()
        writer = csv.DictWriter(
            string_buffer,
            fieldnames=columns,
            delimiter=self.delimiter,
            quoting=self.quoting,
            extrasaction="ignore",
        )

        # Write header
        writer.writeheader()

        count = 0
        async for event in events:
            serialized = self._serialize_event(event)
            writer.writerow(serialized)
            count += 1

            # Flush periodically
            if count % 1000 == 0:
                pass  # StringIO doesn't need explicit flush

        # Get CSV content
        csv_content = string_buffer.getvalue().encode("utf-8")

        # Compress if requested
        if self.compress:
            compressed = gzip.compress(csv_content)
            output.write(compressed)
        else:
            output.write(csv_content)

        return count


class JSONExporter(BaseExporter):
    """Export audit events to JSON Lines (JSONL) format."""

    def __init__(
        self,
        compress: bool = False,
        include_metadata: bool = True,
        pretty: bool = False,
    ):
        """Initialize the JSON exporter.

        Args:
            compress: Whether to gzip compress output.
            include_metadata: Whether to include metadata fields.
            pretty: Whether to pretty-print JSON (increases file size).
        """
        super().__init__(compress, include_metadata)
        self.pretty = pretty

    @property
    def content_type(self) -> str:
        """Get the MIME content type."""
        if self.compress:
            return "application/gzip"
        return "application/x-ndjson"

    @property
    def file_extension(self) -> str:
        """Get the file extension."""
        if self.compress:
            return "jsonl.gz"
        return "jsonl"

    async def export(
        self,
        events: AsyncIterator[Dict[str, Any]],
        output: io.BytesIO,
    ) -> int:
        """Export events to JSONL.

        Args:
            events: Async iterator of audit events.
            output: Output buffer.

        Returns:
            Number of events exported.
        """
        lines: List[bytes] = []
        count = 0

        async for event in events:
            serialized = self._serialize_event(event)

            if self.pretty:
                line = json.dumps(serialized, indent=2, default=str)
            else:
                line = json.dumps(serialized, default=str)

            lines.append(line.encode("utf-8"))
            lines.append(b"\n")
            count += 1

        content = b"".join(lines)

        if self.compress:
            compressed = gzip.compress(content)
            output.write(compressed)
        else:
            output.write(content)

        return count


class ParquetExporter(BaseExporter):
    """Export audit events to Parquet format.

    Requires PyArrow. Falls back to JSON if not available.
    """

    def __init__(
        self,
        compress: bool = True,
        include_metadata: bool = True,
        compression: str = "snappy",
    ):
        """Initialize the Parquet exporter.

        Args:
            compress: Whether to compress (always True for Parquet).
            include_metadata: Whether to include metadata fields.
            compression: Compression algorithm (snappy, gzip, zstd, none).
        """
        super().__init__(compress=True, include_metadata=include_metadata)
        self.compression = compression
        self._pyarrow_available = self._check_pyarrow()

    def _check_pyarrow(self) -> bool:
        """Check if PyArrow is available."""
        try:
            import pyarrow
            return True
        except ImportError:
            logger.warning("PyArrow not available, Parquet export will fall back to JSON")
            return False

    @property
    def content_type(self) -> str:
        """Get the MIME content type."""
        if self._pyarrow_available:
            return "application/vnd.apache.parquet"
        return "application/x-ndjson"

    @property
    def file_extension(self) -> str:
        """Get the file extension."""
        if self._pyarrow_available:
            return "parquet"
        return "jsonl"

    async def export(
        self,
        events: AsyncIterator[Dict[str, Any]],
        output: io.BytesIO,
    ) -> int:
        """Export events to Parquet.

        Args:
            events: Async iterator of audit events.
            output: Output buffer.

        Returns:
            Number of events exported.
        """
        if not self._pyarrow_available:
            # Fall back to JSON
            json_exporter = JSONExporter(
                compress=True,
                include_metadata=self.include_metadata,
            )
            return await json_exporter.export(events, output)

        import pyarrow as pa
        import pyarrow.parquet as pq

        # Collect all events
        records: List[Dict[str, Any]] = []
        async for event in events:
            serialized = self._serialize_event(event)
            records.append(serialized)

        if not records:
            return 0

        # Define schema
        schema = pa.schema([
            ("id", pa.string()),
            ("performed_at", pa.string()),
            ("category", pa.string()),
            ("severity", pa.string()),
            ("event_type", pa.string()),
            ("operation", pa.string()),
            ("outcome", pa.string()),
            ("user_id", pa.string()),
            ("user_email", pa.string()),
            ("organization_id", pa.string()),
            ("resource_type", pa.string()),
            ("resource_path", pa.string()),
            ("action", pa.string()),
            ("ip_address", pa.string()),
            ("change_summary", pa.string()),
            ("error_message", pa.string()),
            ("tags", pa.string()),
            ("metadata", pa.string()),
        ])

        # Create table
        table = pa.Table.from_pylist(records, schema=schema)

        # Write to buffer
        pq.write_table(
            table,
            output,
            compression=self.compression if self.compression != "none" else None,
        )

        return len(records)


def get_exporter(
    format: str,
    compress: bool = False,
    include_metadata: bool = True,
) -> BaseExporter:
    """Get an exporter for the specified format.

    Args:
        format: Export format (csv, json, parquet).
        compress: Whether to compress output.
        include_metadata: Whether to include metadata fields.

    Returns:
        Appropriate exporter instance.

    Raises:
        ValueError: If format is not supported.
    """
    format_lower = format.lower()

    if format_lower == "csv":
        return CSVExporter(compress=compress, include_metadata=include_metadata)
    elif format_lower in ("json", "jsonl"):
        return JSONExporter(compress=compress, include_metadata=include_metadata)
    elif format_lower == "parquet":
        return ParquetExporter(include_metadata=include_metadata)
    else:
        raise ValueError(f"Unsupported export format: {format}")
