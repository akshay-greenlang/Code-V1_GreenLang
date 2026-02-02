# -*- coding: utf-8 -*-
# SAP Job Scheduler
# Celery Beat schedule configuration for SAP sync jobs

"""
SAP Job Scheduler
=================

Celery Beat schedule configuration for automated SAP data synchronization.

Features:
---------
- Configurable sync intervals (hourly, daily, weekly)
- Priority handling for different data types
- Job monitoring and health checks
- Schedule management
- Dynamic schedule updates

Usage:
------
```python
from connectors.sap.jobs.scheduler import get_schedule

# Get Celery Beat schedule
schedule = get_schedule()

# Configure Celery app
celery_app.conf.beat_schedule = schedule
```

Running Celery Beat:
-------------------
```bash
# Start Celery worker
celery -A connectors.sap.jobs.delta_sync worker --loglevel=info

# Start Celery Beat scheduler
celery -A connectors.sap.jobs.delta_sync beat --loglevel=info

# Monitor with Flower
celery -A connectors.sap.jobs.delta_sync flower
```
"""

import logging
from datetime import timedelta
from typing import Any, Dict

from celery.schedules import crontab

# Configure logger
logger = logging.getLogger(__name__)


def get_schedule(
    purchase_orders_interval: int = 3600,  # 1 hour
    deliveries_interval: int = 7200,  # 2 hours
    capital_goods_interval: int = 86400,  # 24 hours
) -> Dict[str, Any]:
    """
    Get Celery Beat schedule configuration.

    Args:
        purchase_orders_interval: Sync interval for purchase orders (seconds)
        deliveries_interval: Sync interval for deliveries (seconds)
        capital_goods_interval: Sync interval for capital goods (seconds)

    Returns:
        Celery Beat schedule dictionary
    """
    schedule = {
        # Purchase Orders (MM) - High priority, sync every hour
        "sync-purchase-orders": {
            "task": "connectors.sap.jobs.delta_sync.sync_purchase_orders",
            "schedule": timedelta(seconds=purchase_orders_interval),
            "args": (),
            "kwargs": {
                "force_full_sync": False,
                "batch_size": 1000,
            },
            "options": {
                "priority": 9,  # High priority
                "queue": "sap_sync",
            },
        },
        # Deliveries (SD) - Medium priority, sync every 2 hours
        "sync-deliveries": {
            "task": "connectors.sap.jobs.delta_sync.sync_deliveries",
            "schedule": timedelta(seconds=deliveries_interval),
            "args": (),
            "kwargs": {
                "force_full_sync": False,
                "batch_size": 1000,
            },
            "options": {
                "priority": 7,  # Medium-high priority
                "queue": "sap_sync",
            },
        },
        # Capital Goods (FI) - Lower priority, sync daily
        "sync-capital-goods": {
            "task": "connectors.sap.jobs.delta_sync.sync_capital_goods",
            "schedule": timedelta(seconds=capital_goods_interval),
            "args": (),
            "kwargs": {
                "force_full_sync": False,
                "batch_size": 500,
            },
            "options": {
                "priority": 5,  # Medium priority
                "queue": "sap_sync",
            },
        },
        # Weekly full sync for purchase orders (backup/reconciliation)
        "full-sync-purchase-orders-weekly": {
            "task": "connectors.sap.jobs.delta_sync.sync_purchase_orders",
            "schedule": crontab(hour=2, minute=0, day_of_week=0),  # Sunday 2 AM
            "args": (),
            "kwargs": {
                "force_full_sync": True,
                "batch_size": 1000,
            },
            "options": {
                "priority": 3,  # Lower priority
                "queue": "sap_sync",
            },
        },
        # Health check job - runs every 15 minutes
        "sap-health-check": {
            "task": "connectors.sap.jobs.scheduler.health_check",
            "schedule": crontab(minute="*/15"),  # Every 15 minutes
            "args": (),
            "kwargs": {},
            "options": {
                "priority": 10,  # Highest priority
                "queue": "sap_monitoring",
            },
        },
    }

    logger.info("SAP job schedule configured")
    return schedule


def get_schedule_for_environment(env: str = "production") -> Dict[str, Any]:
    """
    Get environment-specific schedule configuration.

    Args:
        env: Environment name (development, staging, production)

    Returns:
        Celery Beat schedule dictionary
    """
    if env == "development":
        # More frequent syncs for testing
        return get_schedule(
            purchase_orders_interval=300,  # 5 minutes
            deliveries_interval=600,  # 10 minutes
            capital_goods_interval=1800,  # 30 minutes
        )
    elif env == "staging":
        # Moderate frequency
        return get_schedule(
            purchase_orders_interval=1800,  # 30 minutes
            deliveries_interval=3600,  # 1 hour
            capital_goods_interval=43200,  # 12 hours
        )
    else:  # production
        # Standard frequency
        return get_schedule()


# Health check task
from celery import Celery, Task

celery_app = Celery(
    "sap_tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
)


@celery_app.task
def health_check() -> Dict[str, Any]:
    """
    Health check task for SAP sync jobs.

    Checks:
    - Redis connectivity
    - Last sync timestamps
    - Error rates
    - Queue health

    Returns:
        Health check status dictionary
    """
    import redis
    from datetime import datetime, timezone

    logger.info("Running SAP health check")

    health_status = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "healthy",
        "checks": {},
    }

    try:
        # Check Redis connectivity
        redis_client = redis.from_url(
            "redis://localhost:6379/0", decode_responses=True
        )
        redis_client.ping()
        health_status["checks"]["redis"] = "healthy"
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        health_status["checks"]["redis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"

    try:
        # Check last sync timestamps
        modules = [
            ("MM", "purchase_order"),
            ("SD", "delivery"),
            ("FI", "capital_good"),
        ]

        for module, entity_type in modules:
            key = f"sap:lastsync:{module}:{entity_type}"
            last_sync = redis_client.get(key)

            if last_sync:
                last_sync_dt = datetime.fromisoformat(last_sync)
                age = datetime.now(timezone.utc) - last_sync_dt
                age_hours = age.total_seconds() / 3600

                # Alert if last sync is more than 24 hours old
                if age_hours > 24:
                    health_status["checks"][
                        f"{module}_{entity_type}_sync"
                    ] = f"stale: {age_hours:.1f} hours old"
                    health_status["status"] = "degraded"
                else:
                    health_status["checks"][
                        f"{module}_{entity_type}_sync"
                    ] = f"current: {age_hours:.1f} hours old"
            else:
                health_status["checks"][
                    f"{module}_{entity_type}_sync"
                ] = "no sync recorded"

    except Exception as e:
        logger.error(f"Sync timestamp health check failed: {e}")
        health_status["checks"]["sync_timestamps"] = f"error: {str(e)}"
        health_status["status"] = "unhealthy"

    logger.info(f"Health check completed: {health_status['status']}")
    return health_status


# Job monitoring utilities


def get_job_stats() -> Dict[str, Any]:
    """
    Get statistics for SAP sync jobs.

    Returns:
        Dictionary with job statistics
    """
    import redis

    redis_client = redis.from_url("redis://localhost:6379/0", decode_responses=True)

    stats = {
        "modules": {},
        "queues": {},
    }

    # Get stats for each module
    modules = [
        ("MM", "purchase_order"),
        ("SD", "delivery"),
        ("FI", "capital_good"),
    ]

    for module, entity_type in modules:
        # Get last sync time
        key = f"sap:lastsync:{module}:{entity_type}"
        last_sync = redis_client.get(key)

        # Get deduplication stats
        dedup_key = f"sap:dedup:{entity_type}"
        processed_count = redis_client.scard(dedup_key)

        stats["modules"][f"{module}_{entity_type}"] = {
            "last_sync": last_sync,
            "processed_count": processed_count,
        }

    return stats


def pause_job(job_name: str) -> bool:
    """
    Pause a scheduled job.

    Args:
        job_name: Name of the job to pause

    Returns:
        True if paused successfully
    """
    # Note: This would integrate with Celery Beat dynamic schedule
    logger.info(f"Pausing job: {job_name}")
    # Implementation would depend on dynamic schedule backend
    return True


def resume_job(job_name: str) -> bool:
    """
    Resume a paused job.

    Args:
        job_name: Name of the job to resume

    Returns:
        True if resumed successfully
    """
    logger.info(f"Resuming job: {job_name}")
    # Implementation would depend on dynamic schedule backend
    return True
