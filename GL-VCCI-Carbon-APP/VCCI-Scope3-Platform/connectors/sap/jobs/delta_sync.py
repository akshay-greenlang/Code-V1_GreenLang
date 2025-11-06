# SAP Delta Sync Jobs
# Celery tasks for incremental SAP data synchronization

"""
SAP Delta Sync Jobs
===================

Celery tasks for incremental (delta) synchronization of SAP data.

Features:
---------
- Delta extraction based on last sync timestamp
- Support for different SAP modules (MM, SD, FI)
- Batch processing (1,000 records per batch)
- Error handling and retry
- Progress tracking
- Integration with extractors and mappers
- Push to ValueChainIntakeAgent
- Redis-based state management

Supported Modules:
-----------------
- MM (Materials Management): Purchase orders, goods receipts
- SD (Sales & Distribution): Deliveries, transportation
- FI (Financial Accounting): Capital goods acquisitions

Usage:
------
```python
from connectors.sap.jobs.delta_sync import sync_purchase_orders

# Trigger sync
result = sync_purchase_orders.delay()

# Or with parameters
result = sync_purchase_orders.delay(
    force_full_sync=False,
    batch_size=500
)
```
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from celery import Celery, Task
from celery.exceptions import Retry
from redis import Redis

from connectors.sap.utils.audit_logger import AuditLogger
from connectors.sap.utils.deduplication import DeduplicationCache
from connectors.sap.utils.rate_limiter import RateLimiter
from connectors.sap.utils.retry_logic import retry_with_backoff

# Configure logger
logger = logging.getLogger(__name__)

# Initialize Celery app
# Note: In production, this would be imported from a central celery.py
celery_app = Celery(
    "sap_tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour hard limit
    task_soft_time_limit=3300,  # 55 minutes soft limit
)

# Initialize utilities
redis_client = Redis.from_url("redis://localhost:6379/0", decode_responses=True)
audit_logger = AuditLogger(enable_database=False)  # Console logging only for now
rate_limiter = RateLimiter(rate=10, per=60, redis_client=redis_client)
dedup_cache = DeduplicationCache(ttl_days=7, redis_client=redis_client)

# Batch size for processing
DEFAULT_BATCH_SIZE = 1000


class SAPSyncTask(Task):
    """Base task class for SAP sync jobs with common functionality."""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure."""
        logger.error(
            f"Task {self.name} failed",
            task_id=task_id,
            exception=str(exc),
            traceback=str(einfo),
        )
        audit_logger.log_error_event(
            error_type="task_failure",
            error_message=str(exc),
            metadata={"task_id": task_id, "task_name": self.name},
        )

    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success."""
        logger.info(f"Task {self.name} completed successfully", task_id=task_id)


@celery_app.task(base=SAPSyncTask, bind=True, max_retries=3)
def sync_purchase_orders(
    self,
    force_full_sync: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Sync purchase orders from SAP MM module.

    Args:
        force_full_sync: Force full sync instead of delta (default: False)
        batch_size: Number of records per batch (default: 1000)
        start_date: Override start date (ISO format)
        end_date: Override end date (ISO format)

    Returns:
        Dictionary with sync results
    """
    module = "MM"
    entity_type = "purchase_order"
    endpoint = "/api/purchaseorders"

    logger.info(f"Starting purchase order sync", force_full_sync=force_full_sync)

    try:
        # Get last sync timestamp
        last_sync = _get_last_sync_timestamp(module, entity_type)

        # Determine sync window
        if force_full_sync or not last_sync:
            # Full sync - last 365 days
            sync_start = datetime.now(timezone.utc) - timedelta(days=365)
            logger.info("Performing full sync of purchase orders")
        else:
            # Delta sync from last sync timestamp
            sync_start = last_sync
            logger.info(f"Performing delta sync from {sync_start.isoformat()}")

        # Override with provided dates if specified
        if start_date:
            sync_start = datetime.fromisoformat(start_date)
        sync_end = (
            datetime.fromisoformat(end_date)
            if end_date
            else datetime.now(timezone.utc)
        )

        # Extract data from SAP
        # Note: In production, this would call the actual SAP extractor
        # from connectors.sap.extractors import PurchaseOrderExtractor
        # extractor = PurchaseOrderExtractor()
        # records = extractor.extract_delta(sync_start, sync_end)

        # Mock extraction for now
        logger.info(
            f"Extracting purchase orders from {sync_start.isoformat()} to {sync_end.isoformat()}"
        )
        records = _mock_extract_records(entity_type, sync_start, sync_end)

        # Filter duplicates
        original_count = len(records)
        transaction_ids = [r["transaction_id"] for r in records]
        non_duplicate_ids = dedup_cache.filter_duplicates(transaction_ids, entity_type)
        records = [r for r in records if r["transaction_id"] in non_duplicate_ids]
        filtered_count = len(records)

        logger.info(
            f"Filtered {original_count - filtered_count} duplicates. "
            f"Processing {filtered_count} records"
        )

        # Process in batches
        total_processed = 0
        total_errors = 0

        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(records) + batch_size - 1) // batch_size

            logger.info(
                f"Processing batch {batch_num}/{total_batches} ({len(batch)} records)"
            )

            try:
                # Rate limiting
                if not rate_limiter.wait_if_needed(endpoint, timeout=300):
                    raise Exception("Rate limit wait timeout exceeded")

                # Map to schema
                # Note: In production, this would call the actual mapper
                # from connectors.sap.mappers import PurchaseOrderMapper
                # mapper = PurchaseOrderMapper()
                # mapped_data = mapper.map_batch(batch)

                # Mock mapping
                mapped_data = _mock_map_records(batch, entity_type)

                # Push to IntakeAgent
                # Note: In production, this would call the actual agent
                # from agents.intake import ValueChainIntakeAgent
                # agent = ValueChainIntakeAgent()
                # result = agent.process_batch(mapped_data)

                # Mock push
                _mock_push_to_intake(mapped_data, entity_type)

                # Mark as processed
                batch_ids = [r["transaction_id"] for r in batch]
                dedup_cache.mark_batch_processed(batch_ids, entity_type)

                # Track lineage
                for record in batch:
                    audit_logger.log_lineage(
                        sap_transaction_id=record["transaction_id"],
                        internal_id=f"calc-{record['transaction_id']}",
                        entity_type=entity_type,
                        sap_module=module,
                    )

                total_processed += len(batch)

                # Update progress
                self.update_state(
                    state="PROGRESS",
                    meta={
                        "current": total_processed,
                        "total": len(records),
                        "status": f"Processed {total_processed}/{len(records)} records",
                    },
                )

            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}")
                total_errors += len(batch)
                audit_logger.log_error_event(
                    error_type="batch_processing_error",
                    endpoint=endpoint,
                    error_message=str(e),
                    metadata={"batch": batch_num, "entity_type": entity_type},
                )

        # Update last sync timestamp
        _update_last_sync_timestamp(module, entity_type, sync_end)

        # Log final results
        result = {
            "module": module,
            "entity_type": entity_type,
            "total_extracted": original_count,
            "total_processed": total_processed,
            "total_errors": total_errors,
            "duplicates_filtered": original_count - filtered_count,
            "sync_start": sync_start.isoformat(),
            "sync_end": sync_end.isoformat(),
        }

        logger.info("Purchase order sync completed", **result)
        return result

    except Exception as e:
        logger.error(f"Purchase order sync failed: {e}")
        audit_logger.log_error_event(
            error_type="sync_job_failure",
            endpoint=endpoint,
            error_message=str(e),
            metadata={"module": module, "entity_type": entity_type},
        )
        # Retry with exponential backoff
        raise self.retry(exc=e, countdown=2 ** self.request.retries)


@celery_app.task(base=SAPSyncTask, bind=True, max_retries=3)
def sync_deliveries(
    self,
    force_full_sync: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Dict[str, Any]:
    """
    Sync deliveries and transportation from SAP SD module.

    Args:
        force_full_sync: Force full sync instead of delta
        batch_size: Number of records per batch

    Returns:
        Dictionary with sync results
    """
    module = "SD"
    entity_type = "delivery"
    endpoint = "/api/deliveries"

    logger.info("Starting delivery sync", force_full_sync=force_full_sync)

    try:
        # Similar implementation to sync_purchase_orders
        # Get last sync, extract delta, filter duplicates, process batches
        last_sync = _get_last_sync_timestamp(module, entity_type)

        if force_full_sync or not last_sync:
            sync_start = datetime.now(timezone.utc) - timedelta(days=365)
        else:
            sync_start = last_sync

        sync_end = datetime.now(timezone.utc)

        # Mock extraction
        records = _mock_extract_records(entity_type, sync_start, sync_end)

        # Process similarly to purchase orders
        total_processed = len(records)

        _update_last_sync_timestamp(module, entity_type, sync_end)

        result = {
            "module": module,
            "entity_type": entity_type,
            "total_processed": total_processed,
            "sync_end": sync_end.isoformat(),
        }

        logger.info("Delivery sync completed", **result)
        return result

    except Exception as e:
        logger.error(f"Delivery sync failed: {e}")
        raise self.retry(exc=e, countdown=2 ** self.request.retries)


@celery_app.task(base=SAPSyncTask, bind=True, max_retries=3)
def sync_capital_goods(
    self,
    force_full_sync: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Dict[str, Any]:
    """
    Sync capital goods acquisitions from SAP FI module.

    Args:
        force_full_sync: Force full sync instead of delta
        batch_size: Number of records per batch

    Returns:
        Dictionary with sync results
    """
    module = "FI"
    entity_type = "capital_good"
    endpoint = "/api/capitalgoods"

    logger.info("Starting capital goods sync", force_full_sync=force_full_sync)

    try:
        last_sync = _get_last_sync_timestamp(module, entity_type)

        if force_full_sync or not last_sync:
            sync_start = datetime.now(timezone.utc) - timedelta(days=365)
        else:
            sync_start = last_sync

        sync_end = datetime.now(timezone.utc)

        # Mock extraction
        records = _mock_extract_records(entity_type, sync_start, sync_end)

        total_processed = len(records)

        _update_last_sync_timestamp(module, entity_type, sync_end)

        result = {
            "module": module,
            "entity_type": entity_type,
            "total_processed": total_processed,
            "sync_end": sync_end.isoformat(),
        }

        logger.info("Capital goods sync completed", **result)
        return result

    except Exception as e:
        logger.error(f"Capital goods sync failed: {e}")
        raise self.retry(exc=e, countdown=2 ** self.request.retries)


# Helper functions


def _get_last_sync_timestamp(module: str, entity_type: str) -> Optional[datetime]:
    """Get last sync timestamp from Redis."""
    key = f"sap:lastsync:{module}:{entity_type}"
    try:
        timestamp_str = redis_client.get(key)
        if timestamp_str:
            return datetime.fromisoformat(timestamp_str)
        return None
    except Exception as e:
        logger.error(f"Error getting last sync timestamp: {e}")
        return None


def _update_last_sync_timestamp(
    module: str, entity_type: str, timestamp: datetime
) -> None:
    """Update last sync timestamp in Redis."""
    key = f"sap:lastsync:{module}:{entity_type}"
    try:
        redis_client.set(key, timestamp.isoformat())
        logger.info(f"Updated last sync timestamp for {module}:{entity_type}")
    except Exception as e:
        logger.error(f"Error updating last sync timestamp: {e}")


def _mock_extract_records(
    entity_type: str, start_date: datetime, end_date: datetime
) -> List[Dict[str, Any]]:
    """Mock record extraction (replace with actual extractor in production)."""
    # Simulate extracting records
    return [
        {
            "transaction_id": f"{entity_type.upper()}-{i:05d}",
            "date": start_date.isoformat(),
            "data": f"Mock data for {entity_type}",
        }
        for i in range(10)  # Mock 10 records
    ]


def _mock_map_records(
    records: List[Dict[str, Any]], entity_type: str
) -> List[Dict[str, Any]]:
    """Mock record mapping (replace with actual mapper in production)."""
    # Simulate mapping to schema
    return [
        {
            "id": r["transaction_id"],
            "type": entity_type,
            "mapped_data": r["data"],
        }
        for r in records
    ]


def _mock_push_to_intake(data: List[Dict[str, Any]], entity_type: str) -> None:
    """Mock push to IntakeAgent (replace with actual agent in production)."""
    logger.info(f"Pushed {len(data)} {entity_type} records to IntakeAgent")
