"""
Workday Base Extractor
GL-VCCI Scope 3 Platform

Abstract base class for Workday data extractors with delta sync support.

Version: 1.0.0
Phase: 4 (Weeks 24-26)
Date: 2025-11-06
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Generator
from datetime import date, datetime, timedelta

from ..client import WorkdayRaaSClient
from ..config import WorkdayConnectorConfig
from ..exceptions import WorkdayDataError

logger = logging.getLogger(__name__)


class BaseExtractor(ABC):
    """
    Abstract base class for Workday data extractors.

    Provides common functionality for delta extraction, pagination,
    and error handling.
    """

    def __init__(self, client: WorkdayRaaSClient, config: WorkdayConnectorConfig):
        """
        Initialize base extractor.

        Args:
            client: Workday RaaS client
            config: Workday connector configuration
        """
        self.client = client
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def extract(
        self,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Extract data from Workday.

        Args:
            from_date: Start date for extraction
            to_date: End date for extraction
            **kwargs: Additional extraction parameters

        Returns:
            List of extracted records

        Raises:
            WorkdayDataError: If extraction fails
        """
        pass

    @abstractmethod
    def extract_paginated(
        self,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        **kwargs
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Extract data with pagination.

        Args:
            from_date: Start date for extraction
            to_date: End date for extraction
            **kwargs: Additional extraction parameters

        Yields:
            Batches of extracted records

        Raises:
            WorkdayDataError: If extraction fails
        """
        pass

    def extract_delta(
        self,
        last_sync_date: date,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Extract data since last sync (delta extraction).

        Args:
            last_sync_date: Date of last successful sync
            **kwargs: Additional extraction parameters

        Returns:
            List of records changed since last_sync_date

        Raises:
            WorkdayDataError: If extraction fails
        """
        # Extract from last_sync_date to today
        to_date = date.today()

        self.logger.info(
            f"Extracting delta: from {last_sync_date} to {to_date}"
        )

        return self.extract(
            from_date=last_sync_date,
            to_date=to_date,
            **kwargs
        )

    def validate_date_range(
        self,
        from_date: Optional[date],
        to_date: Optional[date]
    ) -> tuple[date, date]:
        """
        Validate and normalize date range.

        Args:
            from_date: Start date
            to_date: End date

        Returns:
            Tuple of (validated_from_date, validated_to_date)

        Raises:
            ValueError: If date range is invalid
        """
        # Default to last 30 days if not specified
        if not from_date:
            from_date = date.today() - timedelta(days=30)

        if not to_date:
            to_date = date.today()

        # Validate date range
        if from_date > to_date:
            raise ValueError(
                f"from_date ({from_date}) must be before to_date ({to_date})"
            )

        # Warn if date range is very large
        days_diff = (to_date - from_date).days
        if days_diff > 365:
            self.logger.warning(
                f"Large date range: {days_diff} days. "
                "Consider breaking into smaller chunks."
            )

        return from_date, to_date

    def log_extraction_summary(
        self,
        record_count: int,
        from_date: Optional[date],
        to_date: Optional[date],
        elapsed_seconds: float
    ):
        """
        Log extraction summary statistics.

        Args:
            record_count: Number of records extracted
            from_date: Start date
            to_date: End date
            elapsed_seconds: Extraction time in seconds
        """
        self.logger.info(
            f"Extraction complete: {record_count} records in {elapsed_seconds:.2f}s "
            f"({record_count / elapsed_seconds:.2f} records/sec) "
            f"[{from_date} to {to_date}]"
        )
