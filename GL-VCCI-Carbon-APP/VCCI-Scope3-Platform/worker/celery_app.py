"""
GL-VCCI Scope 3 Platform - Celery Worker Application
Handles background tasks and ML processing

Version: 2.0.0
"""

import os
import sys
import logging
from celery import Celery
from celery.signals import worker_ready, worker_shutdown

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_settings
from config.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()

# Initialize Celery application
app = Celery(
    "gl-vcci-worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "services.agents.intake.tasks",
        "services.agents.calculator.tasks",
        "services.agents.hotspot.tasks",
        "services.agents.engagement.tasks",
        "services.agents.reporting.tasks",
        "utils.ml.tasks",
    ],
)

# Celery configuration
app.conf.update(
    # Task configuration
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Task execution
    task_track_started=True,
    task_time_limit=3600,  # 1 hour hard limit
    task_soft_time_limit=3000,  # 50 minutes soft limit
    task_acks_late=True,
    worker_prefetch_multiplier=1,

    # Task routing
    task_routes={
        "services.agents.intake.*": {"queue": "intake"},
        "services.agents.calculator.*": {"queue": "calculator"},
        "services.agents.hotspot.*": {"queue": "hotspot"},
        "services.agents.engagement.*": {"queue": "engagement"},
        "services.agents.reporting.*": {"queue": "reporting"},
        "utils.ml.*": {"queue": "ml"},
    },

    # Task priority
    task_default_priority=5,

    # Result backend
    result_expires=3600,  # 1 hour
    result_persistent=True,

    # Worker configuration
    worker_max_tasks_per_child=100,
    worker_disable_rate_limits=False,

    # Beat schedule (for periodic tasks)
    beat_schedule={
        "cleanup-old-results": {
            "task": "utils.maintenance.cleanup_old_results",
            "schedule": 3600.0,  # Every hour
        },
        "sync-emission-factors": {
            "task": "services.factor_broker.tasks.sync_factors",
            "schedule": 86400.0,  # Daily
        },
        "generate-daily-reports": {
            "task": "services.agents.reporting.tasks.generate_daily_reports",
            "schedule": "0 6 * * *",  # 6 AM daily
        },
    },
)


# ==============================================================================
# Celery Signals
# ==============================================================================

@worker_ready.connect
def on_worker_ready(**kwargs):
    """Called when worker is ready to accept tasks."""
    logger.info("🚀 GL-VCCI Celery Worker ready!")
    logger.info(f"Environment: {settings.APP_ENV}")
    logger.info(f"Broker: {settings.CELERY_BROKER_URL}")
    logger.info("📋 Registered task queues: intake, calculator, hotspot, engagement, reporting, ml")


@worker_shutdown.connect
def on_worker_shutdown(**kwargs):
    """Called when worker is shutting down."""
    logger.info("🛑 GL-VCCI Celery Worker shutting down...")


# ==============================================================================
# Task Discovery
# ==============================================================================

@app.task(bind=True)
def debug_task(self):
    """Debug task to test worker functionality."""
    logger.info(f"Request: {self.request!r}")
    return {"status": "success", "message": "Debug task completed"}


if __name__ == "__main__":
    app.start()
