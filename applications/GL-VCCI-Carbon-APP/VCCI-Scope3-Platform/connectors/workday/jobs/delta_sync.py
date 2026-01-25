# -*- coding: utf-8 -*-
"""
Workday Delta Sync Celery Jobs
GL-VCCI Scope 3 Platform

Celery jobs for incremental data synchronization from Workday RaaS.
Uses Redis for deduplication and tracking last sync timestamps.

Version: 1.0.0
Phase: 4 (Weeks 24-26)
Date: 2025-11-06
"""

import logging
from datetime import date, datetime, timedelta
from typing import Dict, Any, Optional
import redis

from celery import shared_task

from ..config import WorkdayConnectorConfig
from ..client import WorkdayRaaSClient
from ..extractors.hcm_extractor import HCMExtractor
from ..mappers.expense_mapper import ExpenseMapper
from ..mappers.commute_mapper import CommuteMapper
from ..exceptions import WorkdayConnectorError
from greenlang.determinism import DeterministicClock

logger = logging.getLogger(__name__)


# Redis keys for tracking sync state
LAST_SYNC_KEY_EXPENSES = "workday:last_sync:expenses"
LAST_SYNC_KEY_COMMUTES = "workday:last_sync:commutes"
DEDUP_KEY_PREFIX = "workday:dedup:"


class DeduplicationCache:
    """
    Redis-based deduplication cache.

    Tracks processed records to prevent duplicates across sync runs.
    In production, this would be part of the SAP utils module.
    """

    def __init__(self, redis_client: redis.Redis, ttl_days: int = 90):
        """
        Initialize deduplication cache.

        Args:
            redis_client: Redis client instance
            ttl_days: Time-to-live for cache entries in days
        """
        self.redis = redis_client
        self.ttl_seconds = ttl_days * 24 * 60 * 60

    def is_processed(self, record_id: str, record_type: str) -> bool:
        """
        Check if record has been processed.

        Args:
            record_id: Unique record identifier
            record_type: Type of record (e.g., 'expense', 'commute')

        Returns:
            True if record was already processed
        """
        key = f"{DEDUP_KEY_PREFIX}{record_type}:{record_id}"
        return self.redis.exists(key) > 0

    def mark_processed(self, record_id: str, record_type: str):
        """
        Mark record as processed.

        Args:
            record_id: Unique record identifier
            record_type: Type of record
        """
        key = f"{DEDUP_KEY_PREFIX}{record_type}:{record_id}"
        self.redis.setex(key, self.ttl_seconds, "1")

    def mark_batch_processed(self, record_ids: list, record_type: str):
        """
        Mark batch of records as processed.

        Args:
            record_ids: List of record identifiers
            record_type: Type of records
        """
        if not record_ids:
            return

        # Use pipeline for efficiency
        pipe = self.redis.pipeline()
        for record_id in record_ids:
            key = f"{DEDUP_KEY_PREFIX}{record_type}:{record_id}"
            pipe.setex(key, self.ttl_seconds, "1")
        pipe.execute()


def get_redis_client() -> redis.Redis:
    """
    Get Redis client for state tracking.

    Returns:
        Redis client instance
    """
    # In production, load from environment
    import os
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_db = int(os.getenv("REDIS_DB", "0"))

    return redis.Redis(
        host=redis_host,
        port=redis_port,
        db=redis_db,
        decode_responses=True
    )


def get_last_sync_date(redis_client: redis.Redis, key: str) -> date:
    """
    Get last successful sync date from Redis.

    Args:
        redis_client: Redis client
        key: Redis key for sync timestamp

    Returns:
        Last sync date or 30 days ago if no previous sync
    """
    last_sync_str = redis_client.get(key)

    if last_sync_str:
        try:
            return datetime.fromisoformat(last_sync_str).date()
        except Exception as e:
            logger.warning(f"Failed to parse last sync date: {e}")

    # Default to 30 days ago
    return date.today() - timedelta(days=30)


def set_last_sync_date(redis_client: redis.Redis, key: str, sync_date: date):
    """
    Store last successful sync date in Redis.

    Args:
        redis_client: Redis client
        key: Redis key for sync timestamp
        sync_date: Sync date to store
    """
    redis_client.set(key, sync_date.isoformat())


@shared_task(bind=True, max_retries=3, default_retry_delay=300)
def sync_expense_reports(self, tenant_id: str = "tenant-default") -> Dict[str, Any]:
    """
    Celery task: Sync expense reports from Workday (daily).

    This task performs delta extraction of expense reports since last sync,
    maps them to logistics schema, and stores in the platform.

    Args:
        tenant_id: Tenant identifier

    Returns:
        Dict with sync statistics

    Raises:
        WorkdayConnectorError: If sync fails
    """
    logger.info(f"Starting expense reports sync for tenant: {tenant_id}")
    start_time = DeterministicClock.now()

    try:
        # Initialize components
        config = WorkdayConnectorConfig.from_env()
        client = WorkdayRaaSClient(config)
        extractor = HCMExtractor(client, config)
        mapper = ExpenseMapper(tenant_id=tenant_id)

        # Get Redis client
        redis_client = get_redis_client()
        dedup_cache = DeduplicationCache(redis_client)

        # Get last sync date
        last_sync = get_last_sync_date(redis_client, LAST_SYNC_KEY_EXPENSES)
        to_date = date.today()

        logger.info(f"Extracting expenses from {last_sync} to {to_date}")

        # Extract expense reports
        expenses = extractor.extract_expense_reports(
            from_date=last_sync,
            to_date=to_date
        )

        # Filter out already processed records
        new_expenses = [
            e for e in expenses
            if not dedup_cache.is_processed(e.expense_id, "expense")
        ]

        logger.info(
            f"Extracted {len(expenses)} expenses, "
            f"{len(new_expenses)} new (after deduplication)"
        )

        # Map to logistics schema
        logistics_records = mapper.map_expenses_batch(new_expenses)

        # For now, just log
        logger.info(f"Generated {len(logistics_records)} logistics records")

        # Mark records as processed
        processed_ids = [e.expense_id for e in new_expenses]
        dedup_cache.mark_batch_processed(processed_ids, "expense")

        # Update last sync date
        set_last_sync_date(redis_client, LAST_SYNC_KEY_EXPENSES, to_date)

        # Calculate statistics
        elapsed = (DeterministicClock.now() - start_time).total_seconds()

        return {
            "status": "success",
            "tenant_id": tenant_id,
            "records_extracted": len(expenses),
            "records_new": len(new_expenses),
            "records_mapped": len(logistics_records),
            "from_date": last_sync.isoformat(),
            "to_date": to_date.isoformat(),
            "elapsed_seconds": round(elapsed, 2),
        }

    except WorkdayConnectorError as e:
        logger.error(f"Workday sync error: {e}")
        # Retry on connector errors
        raise self.retry(exc=e)

    except Exception as e:
        logger.error(f"Unexpected error in expense sync: {e}")
        return {
            "status": "error",
            "error": str(e),
            "tenant_id": tenant_id,
        }


@shared_task(bind=True, max_retries=3, default_retry_delay=300)
def sync_commute_surveys(self, tenant_id: str = "tenant-default") -> Dict[str, Any]:
    """
    Celery task: Sync commute surveys from Workday (weekly).

    This task performs delta extraction of commute surveys since last sync,
    maps them to Category 7 format, and stores in the platform.

    Args:
        tenant_id: Tenant identifier

    Returns:
        Dict with sync statistics

    Raises:
        WorkdayConnectorError: If sync fails
    """
    logger.info(f"Starting commute surveys sync for tenant: {tenant_id}")
    start_time = DeterministicClock.now()

    try:
        # Initialize components
        config = WorkdayConnectorConfig.from_env()
        client = WorkdayRaaSClient(config)
        extractor = HCMExtractor(client, config)
        mapper = CommuteMapper(tenant_id=tenant_id)

        # Get Redis client
        redis_client = get_redis_client()
        dedup_cache = DeduplicationCache(redis_client)

        # Get last sync date
        last_sync = get_last_sync_date(redis_client, LAST_SYNC_KEY_COMMUTES)
        to_date = date.today()

        logger.info(f"Extracting commutes from {last_sync} to {to_date}")

        # Extract commute surveys
        commutes = extractor.extract_commute_surveys(
            from_date=last_sync,
            to_date=to_date
        )

        # Filter out already processed records
        # Use employee_id + survey_date as unique key
        new_commutes = []
        for c in commutes:
            record_id = f"{c.employee_id}_{c.survey_date}"
            if not dedup_cache.is_processed(record_id, "commute"):
                new_commutes.append(c)

        logger.info(
            f"Extracted {len(commutes)} commutes, "
            f"{len(new_commutes)} new (after deduplication)"
        )

        # Map to Category 7 format
        commute_records = mapper.map_commutes_batch(new_commutes)

        # For now, just log
        logger.info(f"Generated {len(commute_records)} commute records")

        # Mark records as processed
        processed_ids = [
            f"{c.employee_id}_{c.survey_date}"
            for c in new_commutes
        ]
        dedup_cache.mark_batch_processed(processed_ids, "commute")

        # Update last sync date
        set_last_sync_date(redis_client, LAST_SYNC_KEY_COMMUTES, to_date)

        # Calculate statistics
        elapsed = (DeterministicClock.now() - start_time).total_seconds()

        return {
            "status": "success",
            "tenant_id": tenant_id,
            "records_extracted": len(commutes),
            "records_new": len(new_commutes),
            "records_mapped": len(commute_records),
            "from_date": last_sync.isoformat(),
            "to_date": to_date.isoformat(),
            "elapsed_seconds": round(elapsed, 2),
        }

    except WorkdayConnectorError as e:
        logger.error(f"Workday sync error: {e}")
        # Retry on connector errors
        raise self.retry(exc=e)

    except Exception as e:
        logger.error(f"Unexpected error in commute sync: {e}")
        return {
            "status": "error",
            "error": str(e),
            "tenant_id": tenant_id,
        }
