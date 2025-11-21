# -*- coding: utf-8 -*-
"""
Alert rule evaluation engine.

This module evaluates alert rules against metrics and triggers
notifications when conditions are met.
"""

import asyncio
import hashlib
import json
import logging
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field

import aiohttp
from pydantic import BaseModel, Field, validator
import redis.asyncio as aioredis
from greenlang.determinism import DeterministicClock

# Configure logging
logger = logging.getLogger(__name__)


class AlertState(str, Enum):
    """Alert states."""
    OK = "ok"
    PENDING = "pending"
    FIRING = "firing"
    RESOLVED = "resolved"


class AlertSeverity(str, Enum):
    """Alert severities."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class RuleType(str, Enum):
    """Alert rule types."""
    THRESHOLD = "threshold"
    RATE_OF_CHANGE = "rate_of_change"
    ABSENCE = "absence"
    ANOMALY = "anomaly"


class NotificationChannel(str, Enum):
    """Notification delivery channels."""
    EMAIL = "email"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"


@dataclass
class AlertCondition:
    """Alert condition configuration."""
    metric: str
    operator: str  # >, <, >=, <=, ==, !=
    threshold: float
    duration: Optional[int] = None  # Duration in seconds
    aggregation: str = "avg"  # avg, sum, min, max, count


@dataclass
class NotificationConfig:
    """Notification configuration."""
    channel: NotificationChannel
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


class AlertRule(BaseModel):
    """Alert rule configuration."""

    id: str
    name: str
    description: Optional[str] = None
    rule_type: RuleType
    condition: Dict[str, Any]  # Condition configuration
    notifications: List[Dict[str, Any]] = Field(default_factory=list)
    severity: AlertSeverity = AlertSeverity.WARNING
    enabled: bool = True
    tags: Dict[str, str] = Field(default_factory=dict)

    # Grouping and deduplication
    group_by: List[str] = Field(default_factory=list)
    group_wait: int = 0  # Seconds to wait before sending first notification
    group_interval: int = 300  # Seconds between notifications for same group

    # Evaluation settings
    evaluation_interval: int = 60  # Seconds between evaluations
    for_duration: int = 0  # Seconds condition must be true before firing

    # Silence settings
    silenced_until: Optional[datetime] = None

    @validator('condition')
    def validate_condition(cls, v, values):
        """Validate condition based on rule type."""
        rule_type = values.get('rule_type')

        if rule_type == RuleType.THRESHOLD:
            required = ['metric', 'operator', 'threshold']
            for field in required:
                if field not in v:
                    raise ValueError(f"Threshold rule requires '{field}' in condition")

        elif rule_type == RuleType.RATE_OF_CHANGE:
            required = ['metric', 'operator', 'threshold', 'timeframe']
            for field in required:
                if field not in v:
                    raise ValueError(f"Rate of change rule requires '{field}' in condition")

        elif rule_type == RuleType.ABSENCE:
            required = ['metric', 'duration']
            for field in required:
                if field not in v:
                    raise ValueError(f"Absence rule requires '{field}' in condition")

        return v


@dataclass
class AlertInstance:
    """Active alert instance."""

    rule_id: str
    state: AlertState
    value: Optional[float]
    labels: Dict[str, str]
    started_at: datetime
    last_evaluated: datetime
    last_notified: Optional[datetime] = None
    notification_count: int = 0
    fingerprint: str = ""

    def __post_init__(self):
        """Generate fingerprint for deduplication."""
        if not self.fingerprint:
            data = {
                'rule_id': self.rule_id,
                'labels': sorted(self.labels.items())
            }
            self.fingerprint = hashlib.md5(
                json.dumps(data, sort_keys=True).encode()
            ).hexdigest()


class AlertEngine:
    """Alert rule evaluation and notification engine."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        smtp_config: Optional[Dict[str, Any]] = None,
        slack_webhook: Optional[str] = None,
        pagerduty_key: Optional[str] = None
    ):
        """Initialize alert engine.

        Args:
            redis_url: Redis connection URL
            smtp_config: SMTP configuration for email notifications
            slack_webhook: Slack webhook URL
            pagerduty_key: PagerDuty integration key
        """
        self.redis_url = redis_url
        self.smtp_config = smtp_config or {}
        self.slack_webhook = slack_webhook
        self.pagerduty_key = pagerduty_key

        self.redis_client: Optional[aioredis.Redis] = None
        self.rules: Dict[str, AlertRule] = {}
        self.alert_instances: Dict[str, AlertInstance] = {}
        self.running = False
        self._tasks: List[asyncio.Task] = []

    async def start(self) -> None:
        """Start alert engine."""
        logger.info("Starting alert engine")

        # Connect to Redis
        self.redis_client = await aioredis.from_url(
            self.redis_url,
            decode_responses=True
        )

        # Load alert rules from Redis
        await self._load_rules()

        self.running = True

        # Start evaluation task
        self._tasks.append(asyncio.create_task(self._evaluate_rules()))
        self._tasks.append(asyncio.create_task(self._cleanup_resolved_alerts()))

        logger.info("Alert engine started")

    async def stop(self) -> None:
        """Stop alert engine."""
        logger.info("Stopping alert engine")

        self.running = False

        # Cancel tasks
        for task in self._tasks:
            task.cancel()

        await asyncio.gather(*self._tasks, return_exceptions=True)

        # Close Redis
        if self.redis_client:
            await self.redis_client.close()

        logger.info("Alert engine stopped")

    async def add_rule(self, rule: AlertRule) -> None:
        """Add alert rule.

        Args:
            rule: Alert rule to add
        """
        self.rules[rule.id] = rule

        # Save to Redis
        await self.redis_client.set(
            f"alert:rule:{rule.id}",
            rule.json()
        )

        logger.info(f"Added alert rule: {rule.id} ({rule.name})")

    async def remove_rule(self, rule_id: str) -> None:
        """Remove alert rule.

        Args:
            rule_id: Rule ID to remove
        """
        if rule_id in self.rules:
            del self.rules[rule_id]

        # Remove from Redis
        await self.redis_client.delete(f"alert:rule:{rule_id}")

        logger.info(f"Removed alert rule: {rule_id}")

    async def silence_rule(self, rule_id: str, duration: int) -> None:
        """Silence alert rule for a duration.

        Args:
            rule_id: Rule ID to silence
            duration: Duration in seconds
        """
        if rule_id not in self.rules:
            return

        rule = self.rules[rule_id]
        rule.silenced_until = DeterministicClock.utcnow() + timedelta(seconds=duration)

        # Update in Redis
        await self.redis_client.set(
            f"alert:rule:{rule_id}",
            rule.json()
        )

        logger.info(f"Silenced rule {rule_id} until {rule.silenced_until}")

    async def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts.

        Returns:
            List of active alert instances
        """
        alerts = []
        for instance in self.alert_instances.values():
            if instance.state in [AlertState.PENDING, AlertState.FIRING]:
                alerts.append({
                    'rule_id': instance.rule_id,
                    'state': instance.state.value,
                    'value': instance.value,
                    'labels': instance.labels,
                    'started_at': instance.started_at.isoformat(),
                    'last_evaluated': instance.last_evaluated.isoformat(),
                    'fingerprint': instance.fingerprint
                })
        return alerts

    async def _load_rules(self) -> None:
        """Load alert rules from Redis."""
        cursor = b'0'
        while cursor:
            cursor, keys = await self.redis_client.scan(
                cursor=cursor,
                match='alert:rule:*',
                count=100
            )

            for key in keys:
                rule_data = await self.redis_client.get(key)
                if rule_data:
                    try:
                        rule = AlertRule.parse_raw(rule_data)
                        self.rules[rule.id] = rule
                    except Exception as e:
                        logger.error(f"Error loading rule from {key}: {e}")

        logger.info(f"Loaded {len(self.rules)} alert rules")

    async def _evaluate_rules(self) -> None:
        """Evaluate all alert rules periodically."""
        while self.running:
            for rule in list(self.rules.values()):
                if not rule.enabled:
                    continue

                # Check if silenced
                if rule.silenced_until and DeterministicClock.utcnow() < rule.silenced_until:
                    continue

                try:
                    await self._evaluate_rule(rule)
                except Exception as e:
                    logger.error(f"Error evaluating rule {rule.id}: {e}")

            await asyncio.sleep(1)

    async def _evaluate_rule(self, rule: AlertRule) -> None:
        """Evaluate a single alert rule.

        Args:
            rule: Alert rule to evaluate
        """
        # Check evaluation interval
        last_eval_key = f"alert:last_eval:{rule.id}"
        last_eval = await self.redis_client.get(last_eval_key)

        if last_eval:
            last_eval_time = datetime.fromisoformat(last_eval)
            if (DeterministicClock.utcnow() - last_eval_time).total_seconds() < rule.evaluation_interval:
                return

        # Update last evaluation time
        await self.redis_client.set(
            last_eval_key,
            DeterministicClock.utcnow().isoformat()
        )

        # Evaluate based on rule type
        if rule.rule_type == RuleType.THRESHOLD:
            await self._evaluate_threshold_rule(rule)
        elif rule.rule_type == RuleType.RATE_OF_CHANGE:
            await self._evaluate_rate_of_change_rule(rule)
        elif rule.rule_type == RuleType.ABSENCE:
            await self._evaluate_absence_rule(rule)
        elif rule.rule_type == RuleType.ANOMALY:
            await self._evaluate_anomaly_rule(rule)

    async def _evaluate_threshold_rule(self, rule: AlertRule) -> None:
        """Evaluate threshold alert rule.

        Args:
            rule: Alert rule to evaluate
        """
        condition = rule.condition
        metric_name = condition['metric']
        operator = condition['operator']
        threshold = condition['threshold']

        # Get recent metric values from Redis
        # This would query actual metric data
        metric_value = await self._get_metric_value(metric_name, condition.get('aggregation', 'avg'))

        if metric_value is None:
            return

        # Evaluate condition
        triggered = self._evaluate_condition(metric_value, operator, threshold)

        # Get or create alert instance
        instance = await self._get_or_create_instance(rule, metric_value)

        # Update alert state
        if triggered:
            if instance.state == AlertState.OK:
                instance.state = AlertState.PENDING
                instance.started_at = DeterministicClock.utcnow()

            elif instance.state == AlertState.PENDING:
                # Check if condition has been true for required duration
                if rule.for_duration > 0:
                    duration = (DeterministicClock.utcnow() - instance.started_at).total_seconds()
                    if duration >= rule.for_duration:
                        instance.state = AlertState.FIRING
                        await self._send_notifications(rule, instance)
                else:
                    instance.state = AlertState.FIRING
                    await self._send_notifications(rule, instance)

        else:
            if instance.state in [AlertState.PENDING, AlertState.FIRING]:
                instance.state = AlertState.RESOLVED
                await self._send_notifications(rule, instance)

        instance.last_evaluated = DeterministicClock.utcnow()
        instance.value = metric_value

        # Save instance
        self.alert_instances[instance.fingerprint] = instance

    async def _evaluate_rate_of_change_rule(self, rule: AlertRule) -> None:
        """Evaluate rate of change alert rule.

        Args:
            rule: Alert rule to evaluate
        """
        # Get metric values over timeframe
        # Calculate rate of change
        # Compare to threshold
        pass  # Implementation similar to threshold

    async def _evaluate_absence_rule(self, rule: AlertRule) -> None:
        """Evaluate absence alert rule.

        Args:
            rule: Alert rule to evaluate
        """
        condition = rule.condition
        metric_name = condition['metric']
        duration = condition['duration']

        # Check last metric timestamp
        last_seen = await self._get_metric_last_seen(metric_name)

        if last_seen is None:
            return

        # Check if metric is absent
        absent_duration = (DeterministicClock.utcnow() - last_seen).total_seconds()
        triggered = absent_duration >= duration

        instance = await self._get_or_create_instance(rule, absent_duration)

        if triggered and instance.state == AlertState.OK:
            instance.state = AlertState.FIRING
            await self._send_notifications(rule, instance)
        elif not triggered and instance.state == AlertState.FIRING:
            instance.state = AlertState.RESOLVED
            await self._send_notifications(rule, instance)

        instance.last_evaluated = DeterministicClock.utcnow()
        self.alert_instances[instance.fingerprint] = instance

    async def _evaluate_anomaly_rule(self, rule: AlertRule) -> None:
        """Evaluate anomaly detection alert rule.

        Args:
            rule: Alert rule to evaluate
        """
        # Implement ML-based anomaly detection
        pass

    def _evaluate_condition(self, value: float, operator: str, threshold: float) -> bool:
        """Evaluate condition operator.

        Args:
            value: Metric value
            operator: Comparison operator
            threshold: Threshold value

        Returns:
            True if condition is met, False otherwise
        """
        if operator == '>':
            return value > threshold
        elif operator == '<':
            return value < threshold
        elif operator == '>=':
            return value >= threshold
        elif operator == '<=':
            return value <= threshold
        elif operator == '==':
            return value == threshold
        elif operator == '!=':
            return value != threshold
        else:
            return False

    async def _get_metric_value(self, metric_name: str, aggregation: str = 'avg') -> Optional[float]:
        """Get aggregated metric value.

        Args:
            metric_name: Metric name
            aggregation: Aggregation function

        Returns:
            Aggregated metric value or None
        """
        # Query metric values from Redis
        # This would integrate with the metric storage system
        return None

    async def _get_metric_last_seen(self, metric_name: str) -> Optional[datetime]:
        """Get timestamp of last seen metric.

        Args:
            metric_name: Metric name

        Returns:
            Timestamp of last metric or None
        """
        # Query last metric timestamp from Redis
        return None

    async def _get_or_create_instance(
        self,
        rule: AlertRule,
        value: Optional[float] = None
    ) -> AlertInstance:
        """Get or create alert instance.

        Args:
            rule: Alert rule
            value: Current metric value

        Returns:
            Alert instance
        """
        # Generate fingerprint
        labels = {**rule.tags}
        instance_data = {
            'rule_id': rule.id,
            'labels': sorted(labels.items())
        }
        fingerprint = hashlib.md5(
            json.dumps(instance_data, sort_keys=True).encode()
        ).hexdigest()

        # Get or create instance
        if fingerprint not in self.alert_instances:
            instance = AlertInstance(
                rule_id=rule.id,
                state=AlertState.OK,
                value=value,
                labels=labels,
                started_at=DeterministicClock.utcnow(),
                last_evaluated=DeterministicClock.utcnow(),
                fingerprint=fingerprint
            )
            self.alert_instances[fingerprint] = instance

        return self.alert_instances[fingerprint]

    async def _send_notifications(self, rule: AlertRule, instance: AlertInstance) -> None:
        """Send notifications for alert.

        Args:
            rule: Alert rule
            instance: Alert instance
        """
        # Check group interval
        if instance.last_notified:
            interval = (DeterministicClock.utcnow() - instance.last_notified).total_seconds()
            if interval < rule.group_interval:
                return

        # Send to all enabled notification channels
        for notif_config in rule.notifications:
            channel = notif_config.get('channel')
            config = notif_config.get('config', {})

            try:
                if channel == NotificationChannel.EMAIL.value:
                    await self._send_email_notification(rule, instance, config)
                elif channel == NotificationChannel.SLACK.value:
                    await self._send_slack_notification(rule, instance, config)
                elif channel == NotificationChannel.PAGERDUTY.value:
                    await self._send_pagerduty_notification(rule, instance, config)
                elif channel == NotificationChannel.WEBHOOK.value:
                    await self._send_webhook_notification(rule, instance, config)
            except Exception as e:
                logger.error(f"Error sending notification via {channel}: {e}")

        instance.last_notified = DeterministicClock.utcnow()
        instance.notification_count += 1

        # Save alert history
        await self._save_alert_history(rule, instance)

    async def _send_email_notification(
        self,
        rule: AlertRule,
        instance: AlertInstance,
        config: Dict[str, Any]
    ) -> None:
        """Send email notification.

        Args:
            rule: Alert rule
            instance: Alert instance
            config: Email configuration
        """
        if not self.smtp_config:
            logger.warning("SMTP not configured, skipping email notification")
            return

        recipients = config.get('recipients', [])
        if not recipients:
            return

        # Create message
        msg = MIMEMultipart()
        msg['From'] = self.smtp_config.get('from', 'alerts@greenlang.com')
        msg['To'] = ', '.join(recipients)
        msg['Subject'] = f"[{rule.severity.upper()}] {rule.name}"

        body = f"""
        Alert: {rule.name}
        State: {instance.state.value}
        Value: {instance.value}
        Started: {instance.started_at}

        Description: {rule.description or 'N/A'}

        Labels:
        {json.dumps(instance.labels, indent=2)}
        """

        msg.attach(MIMEText(body, 'plain'))

        # Send email
        try:
            with smtplib.SMTP(
                self.smtp_config.get('host', 'localhost'),
                self.smtp_config.get('port', 587)
            ) as server:
                if self.smtp_config.get('use_tls', True):
                    server.starttls()

                if 'username' in self.smtp_config:
                    server.login(
                        self.smtp_config['username'],
                        self.smtp_config['password']
                    )

                server.send_message(msg)

            logger.info(f"Sent email notification for alert {rule.id}")
        except Exception as e:
            logger.error(f"Failed to send email: {e}")

    async def _send_slack_notification(
        self,
        rule: AlertRule,
        instance: AlertInstance,
        config: Dict[str, Any]
    ) -> None:
        """Send Slack notification.

        Args:
            rule: Alert rule
            instance: Alert instance
            config: Slack configuration
        """
        webhook_url = config.get('webhook_url', self.slack_webhook)
        if not webhook_url:
            logger.warning("Slack webhook not configured")
            return

        # Create Slack message
        color = {
            AlertState.FIRING: '#f44336',
            AlertState.RESOLVED: '#4caf50',
            AlertState.PENDING: '#ff9800'
        }.get(instance.state, '#999')

        payload = {
            'attachments': [
                {
                    'color': color,
                    'title': rule.name,
                    'text': rule.description or '',
                    'fields': [
                        {'title': 'State', 'value': instance.state.value, 'short': True},
                        {'title': 'Severity', 'value': rule.severity.value, 'short': True},
                        {'title': 'Value', 'value': str(instance.value), 'short': True},
                        {'title': 'Started', 'value': instance.started_at.isoformat(), 'short': True}
                    ],
                    'footer': 'GreenLang Alerts',
                    'ts': int(DeterministicClock.utcnow().timestamp())
                }
            ]
        }

        # Send to Slack
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload) as response:
                if response.status != 200:
                    logger.error(f"Failed to send Slack notification: {await response.text()}")
                else:
                    logger.info(f"Sent Slack notification for alert {rule.id}")

    async def _send_pagerduty_notification(
        self,
        rule: AlertRule,
        instance: AlertInstance,
        config: Dict[str, Any]
    ) -> None:
        """Send PagerDuty notification.

        Args:
            rule: Alert rule
            instance: Alert instance
            config: PagerDuty configuration
        """
        integration_key = config.get('integration_key', self.pagerduty_key)
        if not integration_key:
            logger.warning("PagerDuty integration key not configured")
            return

        # Create PagerDuty event
        event = {
            'routing_key': integration_key,
            'event_action': 'trigger' if instance.state == AlertState.FIRING else 'resolve',
            'dedup_key': instance.fingerprint,
            'payload': {
                'summary': f"{rule.name}: {instance.state.value}",
                'severity': rule.severity.value,
                'source': 'greenlang-alerts',
                'custom_details': {
                    'rule_id': rule.id,
                    'value': instance.value,
                    'labels': instance.labels
                }
            }
        }

        # Send to PagerDuty
        url = 'https://events.pagerduty.com/v2/enqueue'
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=event) as response:
                if response.status != 202:
                    logger.error(f"Failed to send PagerDuty event: {await response.text()}")
                else:
                    logger.info(f"Sent PagerDuty event for alert {rule.id}")

    async def _send_webhook_notification(
        self,
        rule: AlertRule,
        instance: AlertInstance,
        config: Dict[str, Any]
    ) -> None:
        """Send webhook notification.

        Args:
            rule: Alert rule
            instance: Alert instance
            config: Webhook configuration
        """
        webhook_url = config.get('url')
        if not webhook_url:
            return

        # Create webhook payload
        payload = {
            'rule': {
                'id': rule.id,
                'name': rule.name,
                'severity': rule.severity.value
            },
            'alert': {
                'state': instance.state.value,
                'value': instance.value,
                'labels': instance.labels,
                'started_at': instance.started_at.isoformat(),
                'fingerprint': instance.fingerprint
            }
        }

        # Send webhook
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload) as response:
                if response.status not in [200, 201, 202, 204]:
                    logger.error(f"Webhook failed: {await response.text()}")
                else:
                    logger.info(f"Sent webhook for alert {rule.id}")

    async def _save_alert_history(self, rule: AlertRule, instance: AlertInstance) -> None:
        """Save alert to history.

        Args:
            rule: Alert rule
            instance: Alert instance
        """
        history_entry = {
            'rule_id': rule.id,
            'rule_name': rule.name,
            'state': instance.state.value,
            'value': instance.value,
            'labels': instance.labels,
            'timestamp': DeterministicClock.utcnow().isoformat(),
            'fingerprint': instance.fingerprint
        }

        # Save to Redis sorted set
        await self.redis_client.zadd(
            f"alert:history:{rule.id}",
            {json.dumps(history_entry): DeterministicClock.utcnow().timestamp()}
        )

        # Trim old history (keep last 1000 entries)
        await self.redis_client.zremrangebyrank(
            f"alert:history:{rule.id}",
            0,
            -1001
        )

    async def _cleanup_resolved_alerts(self) -> None:
        """Clean up resolved alert instances."""
        while self.running:
            await asyncio.sleep(3600)  # Run every hour

            current_time = DeterministicClock.utcnow()
            to_remove = []

            for fingerprint, instance in self.alert_instances.items():
                if instance.state == AlertState.RESOLVED:
                    # Remove if resolved for more than 1 hour
                    if (current_time - instance.last_evaluated).total_seconds() > 3600:
                        to_remove.append(fingerprint)

            for fingerprint in to_remove:
                del self.alert_instances[fingerprint]

            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} resolved alerts")
