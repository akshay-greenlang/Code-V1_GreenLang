# -*- coding: utf-8 -*-
"""
Celery Worker Queue Configuration
GL-VCCI Scope 3 Platform - Distributed Task Processing

This module provides Celery configuration for distributed batch processing:
- Task queue for emissions calculations
- Worker pool management
- Task retry and error handling
- Result backend with Redis
- Task prioritization
- Monitoring and metrics

Performance Targets:
- Throughput: 100,000 suppliers/hour
- Worker concurrency: 10 workers per node
- Task latency: P95 <500ms
- Zero task loss

Version: 1.0.0
Team: Performance & Batch Processing (Team 5)
Date: 2025-11-09
"""

import os
import logging
from typing import Dict, Any, Optional
from datetime import timedelta

from celery import Celery, Task
from celery.schedules import crontab
from kombu import Exchange, Queue

logger = logging.getLogger(__name__)


# ============================================================================
# CELERY CONFIGURATION
# ============================================================================

# Broker and backend URLs
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
BROKER_URL = os.getenv('CELERY_BROKER_URL', REDIS_URL)
RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', REDIS_URL)

# Create Celery app
celery_app = Celery(
    'vcci_scope3_workers',
    broker=BROKER_URL,
    backend=RESULT_BACKEND
)


# Celery Configuration
celery_app.conf.update(
    # Task settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,

    # Task execution
    task_acks_late=True,  # Acknowledge after task completes
    task_reject_on_worker_lost=True,  # Requeue if worker dies
    task_track_started=True,  # Track when task starts
    task_time_limit=3600,  # 1 hour hard limit
    task_soft_time_limit=3000,  # 50 minute soft limit

    # Worker settings
    worker_prefetch_multiplier=4,  # Prefetch 4 tasks per worker
    worker_max_tasks_per_child=1000,  # Restart worker after 1000 tasks
    worker_disable_rate_limits=False,
    worker_log_format='[%(asctime)s: %(levelname)s/%(processName)s] %(message)s',

    # Result backend
    result_expires=3600,  # Results expire after 1 hour
    result_backend_transport_options={
        'master_name': 'mymaster',
        'retry_on_timeout': True
    },

    # Retry settings
    task_autoretry_for=(Exception,),
    task_retry_kwargs={'max_retries': 3},
    task_retry_backoff=True,
    task_retry_backoff_max=600,  # Max 10 minutes
    task_retry_jitter=True,

    # Performance
    broker_connection_retry_on_startup=True,
    broker_pool_limit=10,
    broker_transport_options={
        'visibility_timeout': 3600,
        'max_connections': 20
    },

    # Monitoring
    task_send_sent_event=True,
    worker_send_task_events=True,

    # Priority
    task_default_priority=5,
    task_inherit_parent_priority=True,
)


# ============================================================================
# TASK QUEUES AND ROUTING
# ============================================================================

# Define task queues with priorities
celery_app.conf.task_queues = (
    # High priority queue for urgent calculations
    Queue(
        'high_priority',
        Exchange('high_priority'),
        routing_key='high_priority',
        queue_arguments={'x-max-priority': 10}
    ),

    # Default queue for normal calculations
    Queue(
        'default',
        Exchange('default'),
        routing_key='default',
        queue_arguments={'x-max-priority': 5}
    ),

    # Low priority queue for background tasks
    Queue(
        'low_priority',
        Exchange('low_priority'),
        routing_key='low_priority',
        queue_arguments={'x-max-priority': 1}
    ),

    # Batch processing queue
    Queue(
        'batch',
        Exchange('batch'),
        routing_key='batch',
        queue_arguments={'x-max-priority': 7}
    ),

    # Report generation queue
    Queue(
        'reports',
        Exchange('reports'),
        routing_key='reports',
        queue_arguments={'x-max-priority': 3}
    ),
)


# Task routing
celery_app.conf.task_routes = {
    # Emissions calculations
    'workers.tasks.calculate_emission': {'queue': 'default'},
    'workers.tasks.calculate_batch': {'queue': 'batch'},

    # Data processing
    'workers.tasks.process_supplier_data': {'queue': 'default'},
    'workers.tasks.import_csv': {'queue': 'low_priority'},

    # Reporting
    'workers.tasks.generate_report': {'queue': 'reports'},

    # Admin tasks
    'workers.tasks.cleanup_old_data': {'queue': 'low_priority'},
}


# ============================================================================
# BASE TASK CLASS
# ============================================================================

class BaseTask(Task):
    """
    Base task class with error handling and logging.

    Features:
    - Automatic retry on failure
    - Comprehensive logging
    - Performance tracking
    - Error reporting
    """

    autoretry_for = (Exception,)
    retry_kwargs = {'max_retries': 3, 'countdown': 60}
    retry_backoff = True

    def on_success(self, retval, task_id, args, kwargs):
        """Called when task succeeds"""
        logger.info(f"Task {self.name}[{task_id}] succeeded")

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when task fails"""
        logger.error(
            f"Task {self.name}[{task_id}] failed: {exc}",
            exc_info=einfo
        )

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called when task is retried"""
        logger.warning(
            f"Task {self.name}[{task_id}] retry: {exc}"
        )


# ============================================================================
# PERIODIC TASKS
# ============================================================================

# Schedule periodic tasks
celery_app.conf.beat_schedule = {
    # Cleanup old results every day at 2 AM
    'cleanup-old-results': {
        'task': 'workers.tasks.cleanup_old_results',
        'schedule': crontab(hour=2, minute=0),
        'options': {'queue': 'low_priority'}
    },

    # Generate daily summary reports
    'daily-summary-report': {
        'task': 'workers.tasks.generate_daily_summary',
        'schedule': crontab(hour=1, minute=0),
        'options': {'queue': 'reports'}
    },

    # Health check every 5 minutes
    'health-check': {
        'task': 'workers.tasks.health_check',
        'schedule': timedelta(minutes=5),
        'options': {'queue': 'default'}
    },
}


# ============================================================================
# TASK EXECUTION HELPERS
# ============================================================================

def apply_task_async(
    task_name: str,
    args: tuple = (),
    kwargs: dict = None,
    priority: int = 5,
    queue: Optional[str] = None,
    countdown: Optional[int] = None,
    eta: Optional[Any] = None
):
    """
    Helper to apply task asynchronously with options.

    Args:
        task_name: Full task name (e.g., 'workers.tasks.calculate_emission')
        args: Task positional arguments
        kwargs: Task keyword arguments
        priority: Task priority (0-10, higher = more priority)
        queue: Queue name
        countdown: Delay in seconds before execution
        eta: Specific datetime for execution

    Returns:
        AsyncResult
    """
    kwargs = kwargs or {}

    task = celery_app.signature(task_name)

    return task.apply_async(
        args=args,
        kwargs=kwargs,
        priority=priority,
        queue=queue,
        countdown=countdown,
        eta=eta
    )


def apply_batch_tasks(
    task_name: str,
    batch_data: list,
    chunk_size: int = 100,
    priority: int = 7
):
    """
    Apply task to batch of data in chunks.

    Args:
        task_name: Full task name
        batch_data: List of data items
        chunk_size: Items per task
        priority: Task priority

    Returns:
        List of AsyncResults
    """
    from celery import group

    # Split into chunks
    chunks = [
        batch_data[i:i + chunk_size]
        for i in range(0, len(batch_data), chunk_size)
    ]

    # Create task group
    task = celery_app.signature(task_name)

    job = group(
        task.s(chunk).set(priority=priority, queue='batch')
        for chunk in chunks
    )

    # Execute
    result = job.apply_async()

    logger.info(
        f"Submitted batch job: {len(chunks)} chunks, "
        f"{len(batch_data)} total items"
    )

    return result


# ============================================================================
# MONITORING AND METRICS
# ============================================================================

def get_queue_stats() -> Dict[str, Any]:
    """
    Get statistics for all queues.

    Returns:
        Dictionary with queue statistics
    """
    from celery import current_app

    stats = {}
    inspector = current_app.control.inspect()

    # Get active tasks
    active = inspector.active()
    if active:
        stats['active_tasks'] = sum(len(tasks) for tasks in active.values())
    else:
        stats['active_tasks'] = 0

    # Get scheduled tasks
    scheduled = inspector.scheduled()
    if scheduled:
        stats['scheduled_tasks'] = sum(len(tasks) for tasks in scheduled.values())
    else:
        stats['scheduled_tasks'] = 0

    # Get reserved tasks
    reserved = inspector.reserved()
    if reserved:
        stats['reserved_tasks'] = sum(len(tasks) for tasks in reserved.values())
    else:
        stats['reserved_tasks'] = 0

    # Get registered tasks
    registered = inspector.registered()
    if registered:
        stats['registered_tasks'] = list(registered.values())[0] if registered else []
    else:
        stats['registered_tasks'] = []

    return stats


def get_worker_stats() -> Dict[str, Any]:
    """
    Get statistics for all workers.

    Returns:
        Dictionary with worker statistics
    """
    from celery import current_app

    inspector = current_app.control.inspect()

    stats = {
        'active_workers': 0,
        'stats': {}
    }

    # Get worker stats
    worker_stats = inspector.stats()
    if worker_stats:
        stats['active_workers'] = len(worker_stats)
        stats['stats'] = worker_stats

    return stats


# ============================================================================
# WORKER MANAGEMENT
# ============================================================================

def scale_workers(num_workers: int, queue: str = 'default'):
    """
    Scale worker pool for specific queue.

    Args:
        num_workers: Number of workers
        queue: Queue name
    """
    from celery import current_app

    current_app.control.pool_grow(n=num_workers)

    logger.info(f"Scaled {queue} queue to {num_workers} workers")


def purge_queue(queue: str):
    """
    Purge all tasks from queue.

    Args:
        queue: Queue name
    """
    from celery import current_app

    current_app.control.purge()

    logger.warning(f"Purged queue: {queue}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

EXAMPLE_USAGE = """
# ============================================================================
# Celery Worker Queue Usage Examples
# ============================================================================

# Example 1: Start Celery Worker
# ----------------------------------------------------------------------------
# Terminal command:
celery -A workers.celery_config worker \\
    --loglevel=info \\
    --concurrency=10 \\
    --pool=prefork \\
    --queues=default,batch,high_priority


# Example 2: Submit Single Task
# ----------------------------------------------------------------------------
from workers.celery_config import apply_task_async

# Submit calculation task
result = apply_task_async(
    'workers.tasks.calculate_emission',
    args=(supplier_id, category),
    priority=8,
    queue='default'
)

# Wait for result
emissions = result.get(timeout=30)


# Example 3: Submit Batch Tasks
# ----------------------------------------------------------------------------
from workers.celery_config import apply_batch_tasks

# Process 100K suppliers in chunks
supplier_ids = [f"SUP-{i}" for i in range(100000)]

result = apply_batch_tasks(
    'workers.tasks.calculate_batch',
    batch_data=supplier_ids,
    chunk_size=1000,
    priority=7
)

# Wait for all to complete
results = result.get(timeout=3600)


# Example 4: Monitor Queue Stats
# ----------------------------------------------------------------------------
from workers.celery_config import get_queue_stats, get_worker_stats

# Get queue statistics
queue_stats = get_queue_stats()
print(f"Active tasks: {queue_stats['active_tasks']}")
print(f"Scheduled tasks: {queue_stats['scheduled_tasks']}")

# Get worker statistics
worker_stats = get_worker_stats()
print(f"Active workers: {worker_stats['active_workers']}")


# Example 5: Start Flower (Monitoring Dashboard)
# ----------------------------------------------------------------------------
# Terminal command:
celery -A workers.celery_config flower \\
    --port=5555 \\
    --broker=redis://localhost:6379/0

# Access at: http://localhost:5555


# Example 6: Celery Beat (Periodic Tasks)
# ----------------------------------------------------------------------------
# Terminal command:
celery -A workers.celery_config beat \\
    --loglevel=info


# Example 7: Priority Queues
# ----------------------------------------------------------------------------
# High priority task
result = apply_task_async(
    'workers.tasks.calculate_emission',
    args=(supplier_id,),
    priority=10,  # Highest priority
    queue='high_priority'
)

# Low priority task
result = apply_task_async(
    'workers.tasks.cleanup_old_data',
    priority=1,  # Lowest priority
    queue='low_priority'
)
"""


__all__ = [
    'celery_app',
    'BaseTask',
    'apply_task_async',
    'apply_batch_tasks',
    'get_queue_stats',
    'get_worker_stats',
    'scale_workers',
    'purge_queue',
]
