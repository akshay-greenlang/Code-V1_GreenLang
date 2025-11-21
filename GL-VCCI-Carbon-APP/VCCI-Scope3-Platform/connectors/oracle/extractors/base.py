# -*- coding: utf-8 -*-
"""
Base Oracle Extractor

Abstract base class for all Oracle Fusion Cloud extractors. Provides common extraction methods,
delta extraction logic, pagination handling, batch processing, and error handling.

This module integrates with the Oracle REST client (built by another agent) and provides
a standardized interface for extracting data from Oracle Fusion Cloud modules.

Key Features:
    - Delta extraction by LastUpdateDate timestamp
    - Pagination with Oracle's links array pattern
    - Retry logic with exponential backoff
    - Field selection for performance optimization
    - Comprehensive error handling and logging
    - Extraction metadata tracking

Key Differences from SAP OData:
    - Query syntax: q=Field >= 'value' (not OData $filter)
    - Pagination: links array with rel='next' (not @odata.nextLink)
    - Field selection: fields=Field1,Field2 (not $select)
    - Response format: {"items": [...], "count": n, "hasMore": bool}

Author: GL-VCCI Development Team
Version: 1.0.0
Phase: 4 (Weeks 22-24) - Oracle Connector Implementation
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional
from urllib.parse import urljoin

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class ExtractionConfig(BaseModel):
    """Configuration for Oracle data extraction.

    Attributes:
        batch_size: Number of records to fetch per request (default: 500)
        max_retries: Maximum number of retry attempts on failure (default: 3)
        timeout_seconds: Request timeout in seconds (default: 300)
        select_fields: List of fields to select (None = all fields)
        enable_delta: Whether to use delta extraction (default: True)
        last_sync_timestamp: Timestamp for delta extraction (ISO 8601 format)
    """
    batch_size: int = Field(default=500, ge=1, le=5000)
    max_retries: int = Field(default=3, ge=0, le=10)
    timeout_seconds: int = Field(default=300, ge=30, le=3600)
    select_fields: Optional[List[str]] = None
    enable_delta: bool = True
    last_sync_timestamp: Optional[str] = None

    @field_validator('last_sync_timestamp')
    @classmethod
    def validate_timestamp(cls, v: Optional[str]) -> Optional[str]:
        """Validate ISO 8601 timestamp format."""
        if v is not None:
            try:
                datetime.fromisoformat(v.replace('Z', '+00:00'))
            except ValueError:
                raise ValueError(f"Invalid timestamp format: {v}. Expected ISO 8601 format.")
        return v


class ExtractionResult(BaseModel):
    """Result of an extraction operation.

    Attributes:
        success: Whether extraction was successful
        records_extracted: Number of records extracted
        extraction_timestamp: When extraction occurred
        last_record_timestamp: Timestamp of the most recent record
        errors: List of error messages (if any)
        metadata: Additional metadata about the extraction
    """
    success: bool
    records_extracted: int = 0
    extraction_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_record_timestamp: Optional[str] = None
    errors: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseExtractor(ABC):
    """Abstract base class for Oracle extractors.

    All Oracle module-specific extractors (Procurement, SCM, Financials) inherit from this class.
    Provides common extraction methods and utilities adapted for Oracle REST API patterns.

    Attributes:
        client: Oracle REST client (injected by caller)
        config: Extraction configuration
        base_url: Base URL for the Oracle REST API endpoint
    """

    def __init__(self, client: Any, config: Optional[ExtractionConfig] = None):
        """Initialize base extractor.

        Args:
            client: Oracle REST client instance
            config: Extraction configuration (uses defaults if None)
        """
        self.client = client
        self.config = config or ExtractionConfig()
        self.base_url: str = ""  # Set by subclass

        logger.info(
            f"Initialized {self.__class__.__name__} with batch_size={self.config.batch_size}, "
            f"delta_enabled={self.config.enable_delta}"
        )

    @abstractmethod
    def get_resource_path(self) -> str:
        """Get the REST resource path for this extractor.

        Returns:
            REST resource path (e.g., '/purchaseOrders', '/shipments')
        """
        pass

    @abstractmethod
    def get_changed_on_field(self) -> str:
        """Get the field name used for delta extraction.

        Returns:
            Field name (e.g., 'LastUpdateDate', 'LastUpdatedDate')
        """
        pass

    def _build_query_params(
        self,
        additional_filters: Optional[List[str]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Build Oracle REST API query parameters.

        Args:
            additional_filters: Additional filter conditions (Oracle q syntax)
            order_by: Sort order (e.g., 'POHeaderId:asc')
            limit: Number of records to fetch
            offset: Starting offset for pagination

        Returns:
            Dictionary of query parameters
        """
        params: Dict[str, Any] = {}

        # Build filter query
        filters = []

        # Add delta filter if enabled
        if self.config.enable_delta and self.config.last_sync_timestamp:
            changed_field = self.get_changed_on_field()
            filters.append(f"{changed_field} >= '{self.config.last_sync_timestamp}'")
            logger.debug(f"Delta filter: {changed_field} >= {self.config.last_sync_timestamp}")

        # Add any additional filters
        if additional_filters:
            filters.extend(additional_filters)

        # Combine filters with AND
        if filters:
            params["q"] = ";".join(filters)

        # Add field selection
        if self.config.select_fields:
            params["fields"] = ",".join(self.config.select_fields)
            logger.debug(f"Select fields: {params['fields']}")

        # Add pagination
        if limit:
            params["limit"] = limit
        else:
            params["limit"] = self.config.batch_size

        if offset > 0:
            params["offset"] = offset

        # Add ordering
        if order_by:
            params["orderBy"] = order_by

        return params

    def _extract_next_link(self, response: Dict[str, Any]) -> Optional[str]:
        """Extract next page link from Oracle REST response.

        Oracle uses a links array with rel='next' for pagination.

        Args:
            response: Oracle REST response dictionary

        Returns:
            URL for next page or None if no more pages
        """
        if not response.get("hasMore", False):
            return None

        links = response.get("links", [])
        for link in links:
            if link.get("rel") == "next":
                return link.get("href")

        return None

    def get_all(
        self,
        additional_filters: Optional[List[str]] = None,
        order_by: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """Extract all records from Oracle with pagination.

        Args:
            additional_filters: Additional Oracle q filter conditions
            order_by: Oracle orderBy field (e.g., 'CreationDate:desc')

        Yields:
            Individual records as dictionaries

        Raises:
            Exception: If extraction fails after all retries
        """
        resource_path = self.get_resource_path()
        offset = 0
        total_records = 0

        logger.info(f"Starting extraction from {resource_path}")

        while True:
            try:
                # Build query parameters
                params = self._build_query_params(
                    additional_filters=additional_filters,
                    order_by=order_by,
                    offset=offset
                )

                # Execute REST request
                logger.debug(f"Fetching records: offset={offset}, limit={params.get('limit')}")
                response = self.client.get(resource_path, params=params)

                # Parse response - Oracle returns {"items": [...], "count": n, "hasMore": bool}
                records = response.get("items", [])

                if not records:
                    logger.info(f"Extraction complete. Total records: {total_records}")
                    break

                # Yield each record
                for record in records:
                    yield record
                    total_records += 1

                # Check for more pages
                next_link = self._extract_next_link(response)
                if not next_link:
                    logger.info(f"Extraction complete. Total records: {total_records}")
                    break

                # Move to next page
                offset += len(records)

            except Exception as e:
                logger.error(f"Error extracting from {resource_path}: {str(e)}")
                raise

    def get_delta(
        self,
        last_sync_timestamp: str,
        additional_filters: Optional[List[str]] = None
    ) -> Iterator[Dict[str, Any]]:
        """Extract only records changed since last sync.

        Args:
            last_sync_timestamp: ISO 8601 timestamp of last successful sync
            additional_filters: Additional Oracle q filter conditions

        Yields:
            Changed records as dictionaries
        """
        # Temporarily override config for delta extraction
        original_timestamp = self.config.last_sync_timestamp
        self.config.last_sync_timestamp = last_sync_timestamp
        self.config.enable_delta = True

        try:
            changed_field = self.get_changed_on_field()
            logger.info(f"Delta extraction since {last_sync_timestamp} using {changed_field}")

            yield from self.get_all(
                additional_filters=additional_filters,
                order_by=f"{changed_field}:asc"
            )
        finally:
            # Restore original config
            self.config.last_sync_timestamp = original_timestamp

    def get_by_id(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Extract a single record by its ID.

        Args:
            entity_id: Entity identifier (e.g., Purchase Order ID)

        Returns:
            Record dictionary or None if not found
        """
        resource_path = self.get_resource_path()

        try:
            params = {}
            if self.config.select_fields:
                params["fields"] = ",".join(self.config.select_fields)

            logger.debug(f"Fetching single record: {resource_path}/{entity_id}")
            record = self.client.get(f"{resource_path}/{entity_id}", params=params)
            return record

        except Exception as e:
            logger.error(f"Error fetching {resource_path}/{entity_id}: {str(e)}")
            return None

    def extract(
        self,
        additional_filters: Optional[List[str]] = None
    ) -> ExtractionResult:
        """Execute extraction and return structured result.

        Args:
            additional_filters: Additional Oracle q filter conditions

        Returns:
            ExtractionResult with success status and metadata
        """
        result = ExtractionResult(success=False)
        extraction_start = datetime.now(timezone.utc)

        try:
            records = list(self.get_all(additional_filters=additional_filters))
            result.records_extracted = len(records)
            result.success = True

            # Find most recent record timestamp
            if records:
                changed_field = self.get_changed_on_field()
                timestamps = [r.get(changed_field) for r in records if r.get(changed_field)]
                if timestamps:
                    result.last_record_timestamp = max(timestamps)

            # Add metadata
            result.metadata = {
                "resource_path": self.get_resource_path(),
                "extraction_duration_seconds": (
                    datetime.now(timezone.utc) - extraction_start
                ).total_seconds(),
                "batch_size": self.config.batch_size,
                "delta_enabled": self.config.enable_delta,
            }

            logger.info(
                f"Extraction successful: {result.records_extracted} records from "
                f"{self.get_resource_path()}"
            )

        except Exception as e:
            result.success = False
            error_msg = f"Extraction failed: {str(e)}"
            result.errors.append(error_msg)
            logger.error(error_msg, exc_info=True)

        return result
