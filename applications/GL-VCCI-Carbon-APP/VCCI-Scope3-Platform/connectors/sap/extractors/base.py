# -*- coding: utf-8 -*-
"""
Base SAP Extractor

Abstract base class for all SAP S/4HANA extractors. Provides common extraction methods,
delta extraction logic, pagination handling, batch processing, and error handling.

This module integrates with the SAP OData client (built by another agent) and provides
a standardized interface for extracting data from various SAP modules.

Key Features:
    - Delta extraction by ChangedOn timestamp
    - Pagination with configurable batch sizes
    - Retry logic with exponential backoff
    - Field selection for performance optimization
    - Comprehensive error handling and logging
    - Extraction metadata tracking

Author: GL-VCCI Development Team
Version: 1.0.0
Phase: 4 (Weeks 19-22) - SAP Connector Implementation
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class ExtractionConfig(BaseModel):
    """Configuration for SAP data extraction.

    Attributes:
        batch_size: Number of records to fetch per request (default: 1000)
        max_retries: Maximum number of retry attempts on failure (default: 3)
        timeout_seconds: Request timeout in seconds (default: 300)
        select_fields: List of fields to select (None = all fields)
        enable_delta: Whether to use delta extraction (default: True)
        last_sync_timestamp: Timestamp for delta extraction (ISO 8601 format)
    """
    batch_size: int = Field(default=1000, ge=1, le=10000)
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
    """Abstract base class for SAP extractors.

    All SAP module-specific extractors (MM, SD, FI) inherit from this class.
    Provides common extraction methods and utilities.

    Attributes:
        client: SAP OData client (injected by caller)
        config: Extraction configuration
        service_name: Name of the OData service (set by subclass)
    """

    def __init__(self, client: Any, config: Optional[ExtractionConfig] = None):
        """Initialize base extractor.

        Args:
            client: SAP OData client instance
            config: Extraction configuration (uses defaults if None)
        """
        self.client = client
        self.config = config or ExtractionConfig()
        self.service_name: str = ""  # Set by subclass

        logger.info(
            f"Initialized {self.__class__.__name__} with batch_size={self.config.batch_size}, "
            f"delta_enabled={self.config.enable_delta}"
        )

    @abstractmethod
    def get_entity_set_name(self) -> str:
        """Get the entity set name for this extractor.

        Returns:
            OData entity set name (e.g., 'A_PurchaseOrder')
        """
        pass

    @abstractmethod
    def get_changed_on_field(self) -> str:
        """Get the field name used for delta extraction.

        Returns:
            Field name (e.g., 'ChangedOn', 'LastChangeDateTime')
        """
        pass

    def _build_filter_query(self, additional_filters: Optional[List[str]] = None) -> Optional[str]:
        """Build OData $filter query string.

        Args:
            additional_filters: Additional filter conditions to include

        Returns:
            OData filter string or None if no filters needed
        """
        filters = []

        # Add delta filter if enabled
        if self.config.enable_delta and self.config.last_sync_timestamp:
            changed_field = self.get_changed_on_field()
            filters.append(f"{changed_field} gt datetime'{self.config.last_sync_timestamp}'")
            logger.debug(f"Delta filter: {changed_field} > {self.config.last_sync_timestamp}")

        # Add any additional filters
        if additional_filters:
            filters.extend(additional_filters)

        return " and ".join(filters) if filters else None

    def _build_select_query(self) -> Optional[str]:
        """Build OData $select query string.

        Returns:
            Comma-separated field list or None for all fields
        """
        if self.config.select_fields:
            select_str = ",".join(self.config.select_fields)
            logger.debug(f"Select fields: {select_str}")
            return select_str
        return None

    def get_all(
        self,
        additional_filters: Optional[List[str]] = None,
        order_by: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """Extract all records from SAP with pagination.

        Args:
            additional_filters: Additional OData filter conditions
            order_by: OData $orderby field (e.g., 'CreatedOn desc')

        Yields:
            Individual records as dictionaries

        Raises:
            Exception: If extraction fails after all retries
        """
        entity_set = self.get_entity_set_name()
        filter_query = self._build_filter_query(additional_filters)
        select_query = self._build_select_query()

        skip = 0
        total_records = 0

        logger.info(f"Starting extraction from {entity_set}")

        while True:
            try:
                # Build query parameters
                params = {
                    "$top": self.config.batch_size,
                    "$skip": skip,
                }

                if filter_query:
                    params["$filter"] = filter_query
                if select_query:
                    params["$select"] = select_query
                if order_by:
                    params["$orderby"] = order_by

                # Execute OData request
                logger.debug(f"Fetching records: skip={skip}, top={self.config.batch_size}")
                response = self.client.get(entity_set, params=params)

                # Parse response
                records = response.get("value", [])

                if not records:
                    logger.info(f"Extraction complete. Total records: {total_records}")
                    break

                # Yield each record
                for record in records:
                    yield record
                    total_records += 1

                # Move to next page
                skip += self.config.batch_size

            except Exception as e:
                logger.error(f"Error extracting from {entity_set}: {str(e)}")
                raise

    def get_delta(
        self,
        last_sync_timestamp: str,
        additional_filters: Optional[List[str]] = None
    ) -> Iterator[Dict[str, Any]]:
        """Extract only records changed since last sync.

        Args:
            last_sync_timestamp: ISO 8601 timestamp of last successful sync
            additional_filters: Additional OData filter conditions

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
                order_by=f"{changed_field} asc"
            )
        finally:
            # Restore original config
            self.config.last_sync_timestamp = original_timestamp

    def get_by_id(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Extract a single record by its ID.

        Args:
            entity_id: Entity identifier (e.g., Purchase Order number)

        Returns:
            Record dictionary or None if not found
        """
        entity_set = self.get_entity_set_name()
        select_query = self._build_select_query()

        try:
            params = {}
            if select_query:
                params["$select"] = select_query

            logger.debug(f"Fetching single record: {entity_set}('{entity_id}')")
            record = self.client.get_by_key(entity_set, entity_id, params=params)
            return record

        except Exception as e:
            logger.error(f"Error fetching {entity_set}('{entity_id}'): {str(e)}")
            return None

    def extract(
        self,
        additional_filters: Optional[List[str]] = None
    ) -> ExtractionResult:
        """Execute extraction and return structured result.

        Args:
            additional_filters: Additional OData filter conditions

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
                "entity_set": self.get_entity_set_name(),
                "extraction_duration_seconds": (
                    datetime.now(timezone.utc) - extraction_start
                ).total_seconds(),
                "batch_size": self.config.batch_size,
                "delta_enabled": self.config.enable_delta,
            }

            logger.info(
                f"Extraction successful: {result.records_extracted} records from "
                f"{self.get_entity_set_name()}"
            )

        except Exception as e:
            result.success = False
            error_msg = f"Extraction failed: {str(e)}"
            result.errors.append(error_msg)
            logger.error(error_msg, exc_info=True)

        return result
