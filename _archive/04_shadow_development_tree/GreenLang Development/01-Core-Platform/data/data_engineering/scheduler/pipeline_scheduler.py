"""
Pipeline Scheduler
==================

Comprehensive scheduling system for emission factor ETL pipelines.
Includes Airflow DAG generation, dependency management, alerting, and SLA monitoring.

Author: GL-DataIntegrationEngineer
Version: 1.0.0
Created: 2025-12-04
"""

from typing import Dict, List, Optional, Any, Callable, Set, Union
from enum import Enum
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field
import logging
import json
import hashlib
from pathlib import Path
from croniter import croniter

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class PipelineStatus(str, Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRY = "retry"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"
    SLA_MISS = "sla_miss"


class ScheduleFrequency(str, Enum):
    """Common schedule frequencies."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    CUSTOM = "custom"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(str, Enum):
    """Alert delivery channels."""
    EMAIL = "email"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"
    LOG = "log"


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================

class AlertConfig(BaseModel):
    """Alert configuration for pipeline monitoring."""
    enabled: bool = True
    channels: List[AlertChannel] = Field(default_factory=lambda: [AlertChannel.EMAIL])
    recipients: List[str] = Field(default_factory=list)
    on_failure: bool = True
    on_success: bool = False
    on_retry: bool = True
    on_sla_miss: bool = True
    slack_webhook: Optional[str] = None
    pagerduty_key: Optional[str] = None
    custom_webhook_url: Optional[str] = None
    min_severity: AlertSeverity = AlertSeverity.WARNING


class SLAConfig(BaseModel):
    """SLA configuration for pipeline monitoring."""
    enabled: bool = True
    max_duration_minutes: int = Field(default=60, description="Maximum pipeline duration")
    max_latency_hours: int = Field(default=24, description="Maximum data latency")
    min_success_rate: float = Field(default=95.0, description="Minimum success rate %")
    alert_on_miss: bool = True
    escalation_after_minutes: int = Field(default=30, description="Escalate after N minutes")


class RetryConfig(BaseModel):
    """Retry configuration for pipeline failures."""
    enabled: bool = True
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay_seconds: int = Field(default=300)  # 5 minutes
    exponential_backoff: bool = True
    max_retry_delay_seconds: int = Field(default=3600)  # 1 hour
    retry_on: List[str] = Field(
        default_factory=lambda: ["ConnectionError", "TimeoutError", "APIError"]
    )


class ScheduleConfig(BaseModel):
    """Schedule configuration for a pipeline."""
    frequency: ScheduleFrequency = ScheduleFrequency.DAILY
    cron_expression: Optional[str] = None  # Custom cron if frequency is CUSTOM
    timezone: str = "UTC"
    start_date: datetime = Field(default_factory=datetime.utcnow)
    end_date: Optional[datetime] = None
    catchup: bool = False  # Run missed schedules
    max_active_runs: int = Field(default=1, ge=1, le=10)
    depends_on: List[str] = Field(default_factory=list)  # Pipeline dependencies
    tags: List[str] = Field(default_factory=list)

    def get_cron_expression(self) -> str:
        """Get cron expression for the schedule."""
        if self.cron_expression:
            return self.cron_expression

        cron_mapping = {
            ScheduleFrequency.HOURLY: "0 * * * *",
            ScheduleFrequency.DAILY: "0 2 * * *",  # 2 AM
            ScheduleFrequency.WEEKLY: "0 2 * * 0",  # Sunday 2 AM
            ScheduleFrequency.MONTHLY: "0 2 1 * *",  # 1st of month 2 AM
            ScheduleFrequency.QUARTERLY: "0 2 1 1,4,7,10 *",  # 1st of quarter
            ScheduleFrequency.ANNUAL: "0 2 1 1 *",  # Jan 1st 2 AM
        }

        return cron_mapping.get(self.frequency, "0 2 * * *")

    def get_next_run(self, after: datetime = None) -> datetime:
        """Get next scheduled run time."""
        base_time = after or datetime.utcnow()
        cron = croniter(self.get_cron_expression(), base_time)
        return cron.get_next(datetime)


# =============================================================================
# SCHEDULED PIPELINE
# =============================================================================

@dataclass
class ScheduledPipeline:
    """Represents a scheduled pipeline."""
    pipeline_id: str
    pipeline_name: str
    pipeline_class: str  # Python class name
    schedule: ScheduleConfig
    config: Dict[str, Any] = field(default_factory=dict)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    sla: SLAConfig = field(default_factory=SLAConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    enabled: bool = True
    description: str = ""
    owner: str = "data-engineering"

    # Runtime state
    last_run: Optional[datetime] = None
    last_status: PipelineStatus = PipelineStatus.PENDING
    next_run: Optional[datetime] = None
    run_count: int = 0
    success_count: int = 0
    failure_count: int = 0

    def __post_init__(self):
        """Initialize next run time."""
        if self.enabled and self.next_run is None:
            self.next_run = self.schedule.get_next_run()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'pipeline_id': self.pipeline_id,
            'pipeline_name': self.pipeline_name,
            'pipeline_class': self.pipeline_class,
            'schedule': {
                'frequency': self.schedule.frequency.value,
                'cron': self.schedule.get_cron_expression(),
                'timezone': self.schedule.timezone,
            },
            'enabled': self.enabled,
            'description': self.description,
            'owner': self.owner,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'last_status': self.last_status.value,
            'next_run': self.next_run.isoformat() if self.next_run else None,
            'statistics': {
                'run_count': self.run_count,
                'success_count': self.success_count,
                'failure_count': self.failure_count,
                'success_rate': (
                    round(self.success_count / self.run_count * 100, 2)
                    if self.run_count > 0 else 0
                ),
            },
        }


# =============================================================================
# PIPELINE SCHEDULER
# =============================================================================

class PipelineScheduler:
    """
    Emission Factor Pipeline Scheduler.

    Features:
    - Schedule pipeline executions (cron-based)
    - Dependency management
    - Alert on failures
    - SLA monitoring
    - Airflow DAG generation
    - Run history tracking
    """

    def __init__(
        self,
        default_alerts: Optional[AlertConfig] = None,
        default_sla: Optional[SLAConfig] = None,
        state_file: Optional[str] = None,
    ):
        """
        Initialize scheduler.

        Args:
            default_alerts: Default alert configuration
            default_sla: Default SLA configuration
            state_file: Path to persist scheduler state
        """
        self.pipelines: Dict[str, ScheduledPipeline] = {}
        self.default_alerts = default_alerts or AlertConfig()
        self.default_sla = default_sla or SLAConfig()
        self.state_file = Path(state_file) if state_file else None
        self.run_history: List[Dict[str, Any]] = []

        if self.state_file and self.state_file.exists():
            self._load_state()

    def register_pipeline(
        self,
        pipeline_name: str,
        pipeline_class: str,
        schedule: ScheduleConfig,
        config: Optional[Dict[str, Any]] = None,
        alerts: Optional[AlertConfig] = None,
        sla: Optional[SLAConfig] = None,
        description: str = "",
        owner: str = "data-engineering",
        enabled: bool = True,
    ) -> ScheduledPipeline:
        """
        Register a pipeline for scheduling.

        Args:
            pipeline_name: Human-readable pipeline name
            pipeline_class: Python class path
            schedule: Schedule configuration
            config: Pipeline-specific configuration
            alerts: Alert configuration (uses default if not provided)
            sla: SLA configuration (uses default if not provided)
            description: Pipeline description
            owner: Pipeline owner
            enabled: Whether pipeline is enabled

        Returns:
            Registered ScheduledPipeline
        """
        pipeline_id = self._generate_pipeline_id(pipeline_name)

        pipeline = ScheduledPipeline(
            pipeline_id=pipeline_id,
            pipeline_name=pipeline_name,
            pipeline_class=pipeline_class,
            schedule=schedule,
            config=config or {},
            alerts=alerts or self.default_alerts,
            sla=sla or self.default_sla,
            enabled=enabled,
            description=description,
            owner=owner,
        )

        self.pipelines[pipeline_id] = pipeline
        logger.info(f"Registered pipeline: {pipeline_name} ({pipeline_id})")

        self._save_state()

        return pipeline

    def _generate_pipeline_id(self, name: str) -> str:
        """Generate pipeline ID from name."""
        normalized = name.lower().replace(' ', '_').replace('-', '_')
        return f"gl_{normalized}"

    def unregister_pipeline(self, pipeline_id: str) -> bool:
        """Unregister a pipeline."""
        if pipeline_id in self.pipelines:
            del self.pipelines[pipeline_id]
            self._save_state()
            return True
        return False

    def get_pipeline(self, pipeline_id: str) -> Optional[ScheduledPipeline]:
        """Get pipeline by ID."""
        return self.pipelines.get(pipeline_id)

    def list_pipelines(
        self,
        enabled_only: bool = False,
        tags: Optional[List[str]] = None
    ) -> List[ScheduledPipeline]:
        """List registered pipelines."""
        pipelines = list(self.pipelines.values())

        if enabled_only:
            pipelines = [p for p in pipelines if p.enabled]

        if tags:
            pipelines = [
                p for p in pipelines
                if any(t in p.schedule.tags for t in tags)
            ]

        return pipelines

    def get_due_pipelines(self, as_of: datetime = None) -> List[ScheduledPipeline]:
        """Get pipelines that are due to run."""
        check_time = as_of or datetime.utcnow()

        due = []
        for pipeline in self.pipelines.values():
            if not pipeline.enabled:
                continue
            if pipeline.next_run and pipeline.next_run <= check_time:
                due.append(pipeline)

        # Sort by next_run (soonest first)
        due.sort(key=lambda p: p.next_run or datetime.max)

        return due

    def get_pipelines_with_dependencies(
        self,
        pipeline_id: str
    ) -> List[ScheduledPipeline]:
        """Get pipeline execution order including dependencies."""
        pipeline = self.pipelines.get(pipeline_id)
        if not pipeline:
            return []

        # Build dependency graph
        visited: Set[str] = set()
        order: List[ScheduledPipeline] = []

        def visit(pid: str):
            if pid in visited:
                return
            visited.add(pid)

            p = self.pipelines.get(pid)
            if p:
                for dep_name in p.schedule.depends_on:
                    dep_id = self._generate_pipeline_id(dep_name)
                    visit(dep_id)
                order.append(p)

        visit(pipeline_id)

        return order

    async def run_pipeline(
        self,
        pipeline_id: str,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a scheduled pipeline.

        Args:
            pipeline_id: Pipeline to run
            force: Run even if not due

        Returns:
            Execution result
        """
        pipeline = self.pipelines.get(pipeline_id)
        if not pipeline:
            return {'error': f'Pipeline not found: {pipeline_id}'}

        if not pipeline.enabled and not force:
            return {'error': 'Pipeline is disabled', 'pipeline_id': pipeline_id}

        started_at = datetime.utcnow()
        result = {
            'pipeline_id': pipeline_id,
            'pipeline_name': pipeline.pipeline_name,
            'started_at': started_at.isoformat(),
            'status': PipelineStatus.RUNNING.value,
        }

        try:
            logger.info(f"Starting pipeline: {pipeline.pipeline_name}")

            # Import and instantiate pipeline
            pipeline_instance = self._instantiate_pipeline(pipeline)

            # Run the pipeline
            pipeline_result = await pipeline_instance.run()

            # Update status
            completed_at = datetime.utcnow()
            duration = (completed_at - started_at).total_seconds()

            result.update({
                'status': PipelineStatus.SUCCESS.value,
                'completed_at': completed_at.isoformat(),
                'duration_seconds': duration,
                'metrics': pipeline_result.metrics if hasattr(pipeline_result, 'metrics') else {},
            })

            # Update pipeline state
            pipeline.last_run = completed_at
            pipeline.last_status = PipelineStatus.SUCCESS
            pipeline.run_count += 1
            pipeline.success_count += 1
            pipeline.next_run = pipeline.schedule.get_next_run(completed_at)

            # Check SLA
            if pipeline.sla.enabled:
                if duration > pipeline.sla.max_duration_minutes * 60:
                    await self._send_sla_alert(pipeline, 'duration_exceeded', duration)
                    pipeline.last_status = PipelineStatus.SLA_MISS

            # Send success alert if configured
            if pipeline.alerts.on_success:
                await self._send_alert(
                    pipeline,
                    AlertSeverity.INFO,
                    f"Pipeline {pipeline.pipeline_name} completed successfully",
                    result
                )

        except Exception as e:
            completed_at = datetime.utcnow()
            duration = (completed_at - started_at).total_seconds()

            result.update({
                'status': PipelineStatus.FAILED.value,
                'completed_at': completed_at.isoformat(),
                'duration_seconds': duration,
                'error': str(e),
            })

            # Update pipeline state
            pipeline.last_run = completed_at
            pipeline.last_status = PipelineStatus.FAILED
            pipeline.run_count += 1
            pipeline.failure_count += 1

            # Handle retry
            if pipeline.retry.enabled and self._should_retry(e, pipeline.retry):
                result['status'] = PipelineStatus.RETRY.value
                # Schedule retry
                retry_delay = self._calculate_retry_delay(pipeline)
                pipeline.next_run = completed_at + timedelta(seconds=retry_delay)

                if pipeline.alerts.on_retry:
                    await self._send_alert(
                        pipeline,
                        AlertSeverity.WARNING,
                        f"Pipeline {pipeline.pipeline_name} failed, retry scheduled",
                        result
                    )
            else:
                pipeline.next_run = pipeline.schedule.get_next_run(completed_at)

                if pipeline.alerts.on_failure:
                    await self._send_alert(
                        pipeline,
                        AlertSeverity.ERROR,
                        f"Pipeline {pipeline.pipeline_name} failed: {str(e)}",
                        result
                    )

            logger.error(f"Pipeline {pipeline.pipeline_name} failed: {e}")

        # Record history
        self.run_history.append(result)
        self._save_state()

        return result

    def _instantiate_pipeline(self, scheduled: ScheduledPipeline):
        """Dynamically instantiate pipeline class."""
        # Parse class path
        parts = scheduled.pipeline_class.rsplit('.', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid pipeline class: {scheduled.pipeline_class}")

        module_path, class_name = parts

        # Import module
        import importlib
        module = importlib.import_module(module_path)
        pipeline_class = getattr(module, class_name)

        # Get config class if exists
        config_class = getattr(module, f'{class_name}Config', None)

        if config_class:
            config = config_class(**scheduled.config)
            return pipeline_class(config)
        else:
            return pipeline_class(scheduled.config)

    def _should_retry(self, error: Exception, retry_config: RetryConfig) -> bool:
        """Determine if error should trigger retry."""
        error_type = type(error).__name__
        return error_type in retry_config.retry_on

    def _calculate_retry_delay(self, pipeline: ScheduledPipeline) -> int:
        """Calculate retry delay with exponential backoff."""
        retry = pipeline.retry
        base_delay = retry.retry_delay_seconds

        if retry.exponential_backoff:
            # Exponential backoff based on failure count
            multiplier = 2 ** min(pipeline.failure_count - 1, 5)
            delay = base_delay * multiplier
            return min(delay, retry.max_retry_delay_seconds)

        return base_delay

    async def _send_alert(
        self,
        pipeline: ScheduledPipeline,
        severity: AlertSeverity,
        message: str,
        details: Dict[str, Any]
    ) -> None:
        """Send alert through configured channels."""
        if not pipeline.alerts.enabled:
            return

        if severity.value < pipeline.alerts.min_severity.value:
            return

        alert_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'pipeline_id': pipeline.pipeline_id,
            'pipeline_name': pipeline.pipeline_name,
            'severity': severity.value,
            'message': message,
            'details': details,
        }

        for channel in pipeline.alerts.channels:
            try:
                if channel == AlertChannel.EMAIL:
                    await self._send_email_alert(pipeline.alerts, alert_data)
                elif channel == AlertChannel.SLACK:
                    await self._send_slack_alert(pipeline.alerts, alert_data)
                elif channel == AlertChannel.PAGERDUTY:
                    await self._send_pagerduty_alert(pipeline.alerts, alert_data)
                elif channel == AlertChannel.WEBHOOK:
                    await self._send_webhook_alert(pipeline.alerts, alert_data)
                elif channel == AlertChannel.LOG:
                    logger.warning(f"Alert: {json.dumps(alert_data)}")
            except Exception as e:
                logger.error(f"Failed to send {channel} alert: {e}")

    async def _send_email_alert(self, config: AlertConfig, data: Dict) -> None:
        """Send email alert."""
        # Implementation would use email service
        logger.info(f"Email alert to {config.recipients}: {data['message']}")

    async def _send_slack_alert(self, config: AlertConfig, data: Dict) -> None:
        """Send Slack alert."""
        if not config.slack_webhook:
            return

        try:
            import httpx

            payload = {
                'text': f"*{data['severity'].upper()}*: {data['message']}",
                'attachments': [{
                    'color': {'error': 'danger', 'warning': 'warning'}.get(data['severity'], 'good'),
                    'fields': [
                        {'title': 'Pipeline', 'value': data['pipeline_name'], 'short': True},
                        {'title': 'Time', 'value': data['timestamp'], 'short': True},
                    ]
                }]
            }

            async with httpx.AsyncClient() as client:
                await client.post(config.slack_webhook, json=payload)

        except ImportError:
            logger.debug("httpx not available for Slack alerts")

    async def _send_pagerduty_alert(self, config: AlertConfig, data: Dict) -> None:
        """Send PagerDuty alert."""
        if not config.pagerduty_key:
            return

        # Implementation would use PagerDuty API
        logger.info(f"PagerDuty alert: {data['message']}")

    async def _send_webhook_alert(self, config: AlertConfig, data: Dict) -> None:
        """Send webhook alert."""
        if not config.custom_webhook_url:
            return

        try:
            import httpx

            async with httpx.AsyncClient() as client:
                await client.post(config.custom_webhook_url, json=data)

        except ImportError:
            logger.debug("httpx not available for webhook alerts")

    async def _send_sla_alert(
        self,
        pipeline: ScheduledPipeline,
        reason: str,
        actual_value: Any
    ) -> None:
        """Send SLA miss alert."""
        if pipeline.alerts.on_sla_miss:
            await self._send_alert(
                pipeline,
                AlertSeverity.WARNING,
                f"SLA miss for {pipeline.pipeline_name}: {reason}",
                {'reason': reason, 'actual_value': actual_value}
            )

    def _save_state(self) -> None:
        """Persist scheduler state."""
        if not self.state_file:
            return

        state = {
            'pipelines': {
                pid: p.to_dict()
                for pid, p in self.pipelines.items()
            },
            'run_history': self.run_history[-100:],  # Keep last 100
            'saved_at': datetime.utcnow().isoformat(),
        }

        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)

    def _load_state(self) -> None:
        """Load scheduler state."""
        if not self.state_file or not self.state_file.exists():
            return

        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)

            self.run_history = state.get('run_history', [])
            logger.info(f"Loaded scheduler state from {self.state_file}")

        except Exception as e:
            logger.warning(f"Failed to load scheduler state: {e}")

    # =========================================================================
    # AIRFLOW DAG GENERATION
    # =========================================================================

    def generate_airflow_dag(
        self,
        pipeline_id: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate Airflow DAG code for a pipeline.

        Args:
            pipeline_id: Pipeline to generate DAG for
            output_path: Optional path to write DAG file

        Returns:
            Generated DAG code
        """
        pipeline = self.pipelines.get(pipeline_id)
        if not pipeline:
            raise ValueError(f"Pipeline not found: {pipeline_id}")

        dag_code = self._generate_dag_code(pipeline)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(dag_code)
            logger.info(f"Generated Airflow DAG: {output_path}")

        return dag_code

    def _generate_dag_code(self, pipeline: ScheduledPipeline) -> str:
        """Generate Airflow DAG Python code."""
        dependencies = '\n    '.join([
            f"'{dep}'," for dep in pipeline.schedule.depends_on
        ]) if pipeline.schedule.depends_on else ''

        return f'''"""
Airflow DAG: {pipeline.pipeline_name}
=====================================

Auto-generated by GreenLang Pipeline Scheduler.
Pipeline ID: {pipeline.pipeline_id}
Description: {pipeline.description}
Owner: {pipeline.owner}

Generated: {datetime.utcnow().isoformat()}
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
from datetime import datetime, timedelta

# Import the pipeline
from {pipeline.pipeline_class.rsplit(".", 1)[0]} import {pipeline.pipeline_class.rsplit(".", 1)[1]}

default_args = {{
    'owner': '{pipeline.owner}',
    'depends_on_past': False,
    'email': {pipeline.alerts.recipients},
    'email_on_failure': {pipeline.alerts.on_failure},
    'email_on_retry': {pipeline.alerts.on_retry},
    'retries': {pipeline.retry.max_retries},
    'retry_delay': timedelta(seconds={pipeline.retry.retry_delay_seconds}),
    'retry_exponential_backoff': {pipeline.retry.exponential_backoff},
    'max_retry_delay': timedelta(seconds={pipeline.retry.max_retry_delay_seconds}),
    'execution_timeout': timedelta(minutes={pipeline.sla.max_duration_minutes}),
}}


async def run_pipeline(**context):
    """Execute the ETL pipeline."""
    config = {json.dumps(pipeline.config, indent=8)}

    pipeline_instance = {pipeline.pipeline_class.rsplit(".", 1)[1]}(config)
    result = await pipeline_instance.run()

    # Push metrics to XCom
    context['ti'].xcom_push(key='metrics', value=result.metrics)
    context['ti'].xcom_push(key='status', value=result.status)

    return result.status


with DAG(
    '{pipeline.pipeline_id}',
    default_args=default_args,
    description='{pipeline.description}',
    schedule_interval='{pipeline.schedule.get_cron_expression()}',
    start_date=datetime({pipeline.schedule.start_date.year}, {pipeline.schedule.start_date.month}, {pipeline.schedule.start_date.day}),
    catchup={pipeline.schedule.catchup},
    max_active_runs={pipeline.schedule.max_active_runs},
    tags={pipeline.schedule.tags},
) as dag:

    run_etl = PythonOperator(
        task_id='run_{pipeline.pipeline_id}',
        python_callable=run_pipeline,
        provide_context=True,
    )

    # Dependencies
    {self._generate_dependency_code(pipeline) if pipeline.schedule.depends_on else '# No dependencies'}

    run_etl
'''

    def _generate_dependency_code(self, pipeline: ScheduledPipeline) -> str:
        """Generate dependency sensor code."""
        sensors = []
        for dep in pipeline.schedule.depends_on:
            dep_id = self._generate_pipeline_id(dep)
            sensors.append(f'''
    wait_for_{dep_id} = ExternalTaskSensor(
        task_id='wait_for_{dep_id}',
        external_dag_id='{dep_id}',
        external_task_id='run_{dep_id}',
        timeout=3600,
    )
    wait_for_{dep_id} >> run_etl''')

        return '\n'.join(sensors)

    def generate_all_dags(self, output_dir: str) -> List[str]:
        """Generate DAGs for all registered pipelines."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        generated = []
        for pipeline_id in self.pipelines:
            dag_file = output_path / f"{pipeline_id}.py"
            self.generate_airflow_dag(pipeline_id, str(dag_file))
            generated.append(str(dag_file))

        return generated


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_default_scheduler(state_file: str = None) -> PipelineScheduler:
    """Create scheduler with default configuration."""
    alerts = AlertConfig(
        enabled=True,
        channels=[AlertChannel.EMAIL, AlertChannel.LOG],
        recipients=['data-alerts@greenlang.io'],
    )

    sla = SLAConfig(
        enabled=True,
        max_duration_minutes=120,
        max_latency_hours=24,
    )

    return PipelineScheduler(
        default_alerts=alerts,
        default_sla=sla,
        state_file=state_file,
    )


def register_emission_factor_pipelines(scheduler: PipelineScheduler) -> None:
    """Register all standard emission factor pipelines."""

    # DEFRA Pipeline - Annual
    scheduler.register_pipeline(
        pipeline_name="DEFRA Annual Refresh",
        pipeline_class="greenlang.data_engineering.etl.defra_pipeline.DEFRAPipeline",
        schedule=ScheduleConfig(
            frequency=ScheduleFrequency.ANNUAL,
            cron_expression="0 6 1 6 *",  # June 1st, 6 AM
            tags=["emission-factors", "defra", "annual"],
        ),
        config={
            "pipeline_name": "defra_annual_refresh",
            "source_name": "DEFRA UK Government",
            "defra_year": datetime.now().year,
        },
        description="Annual refresh of UK DEFRA emission factors",
    )

    # EPA eGRID Pipeline - Quarterly
    scheduler.register_pipeline(
        pipeline_name="EPA eGRID Quarterly",
        pipeline_class="greenlang.data_engineering.etl.epa_egrid_pipeline.EPAeGRIDPipeline",
        schedule=ScheduleConfig(
            frequency=ScheduleFrequency.QUARTERLY,
            cron_expression="0 8 1 1,4,7,10 *",  # Quarterly, 8 AM
            tags=["emission-factors", "egrid", "quarterly", "grid-factors"],
        ),
        description="Quarterly refresh of EPA eGRID grid emission factors",
    )

    # EPA Hub Pipeline - Annual
    scheduler.register_pipeline(
        pipeline_name="EPA Hub Annual",
        pipeline_class="greenlang.data_engineering.etl.epa_hub_pipeline.EPAHubPipeline",
        schedule=ScheduleConfig(
            frequency=ScheduleFrequency.ANNUAL,
            cron_expression="0 7 1 4 *",  # April 1st, 7 AM
            tags=["emission-factors", "epa-hub", "annual"],
        ),
        description="Annual refresh of EPA Emission Factor Hub",
    )

    # IEA Pipeline - Annual
    scheduler.register_pipeline(
        pipeline_name="IEA Annual",
        pipeline_class="greenlang.data_engineering.etl.iea_pipeline.IEAPipeline",
        schedule=ScheduleConfig(
            frequency=ScheduleFrequency.ANNUAL,
            cron_expression="0 9 1 2 *",  # February 1st, 9 AM
            tags=["emission-factors", "iea", "annual", "grid-factors"],
        ),
        description="Annual refresh of IEA international grid factors",
    )

    # Ecoinvent Pipeline - Quarterly (for license subscribers)
    scheduler.register_pipeline(
        pipeline_name="Ecoinvent Quarterly",
        pipeline_class="greenlang.data_engineering.etl.ecoinvent_pipeline.EcoinventPipeline",
        schedule=ScheduleConfig(
            frequency=ScheduleFrequency.QUARTERLY,
            tags=["emission-factors", "ecoinvent", "quarterly", "lci"],
        ),
        enabled=False,  # Requires license
        description="Quarterly refresh of Ecoinvent LCI factors (requires license)",
    )

    logger.info("Registered all standard emission factor pipelines")
