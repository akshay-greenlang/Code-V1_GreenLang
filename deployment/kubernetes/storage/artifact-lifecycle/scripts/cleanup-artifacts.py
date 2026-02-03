#!/usr/bin/env python3
"""
Artifact Cleanup Script for GreenLang Storage Lifecycle Management

This script provides automated cleanup of old build artifacts and temporary files
based on configurable retention policies. It supports dry-run mode, detailed logging,
and Prometheus metrics export.

Features:
- Boto3-based S3 cleanup operations
- Configurable retention policies per artifact type
- Dry-run mode for safe testing
- Detailed logging with structured output
- Prometheus metrics export via pushgateway
- Slack/webhook notifications
- Batch operations for efficiency
- Legal hold awareness

Usage:
    python cleanup-artifacts.py --config /config/cleanup-config.yaml [--dry-run]

Author: GreenLang DevOps Team
Version: 1.0.0
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('artifact-cleanup')


@dataclass
class CleanupStats:
    """Statistics for cleanup operations."""
    objects_scanned: int = 0
    objects_deleted: int = 0
    objects_skipped: int = 0
    objects_failed: int = 0
    bytes_freed: int = 0
    bytes_scanned: int = 0
    duration_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            'objects_scanned': self.objects_scanned,
            'objects_deleted': self.objects_deleted,
            'objects_skipped': self.objects_skipped,
            'objects_failed': self.objects_failed,
            'bytes_freed': self.bytes_freed,
            'bytes_scanned': self.bytes_scanned,
            'duration_seconds': round(self.duration_seconds, 2),
            'errors': self.errors[:100]  # Limit error list
        }


@dataclass
class CleanupTarget:
    """Configuration for a cleanup target."""
    name: str
    bucket: str
    prefixes: List[str]
    retention_days: int
    exclude_patterns: List[str] = field(default_factory=list)
    filters: List[Dict[str, Any]] = field(default_factory=list)


class PrometheusMetrics:
    """Prometheus metrics exporter."""

    def __init__(self, pushgateway_url: Optional[str] = None, job_name: str = 'artifact_cleanup'):
        self.pushgateway_url = pushgateway_url
        self.job_name = job_name
        self.metrics: Dict[str, Any] = {}

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric."""
        key = (name, frozenset((labels or {}).items()))
        self.metrics[key] = ('gauge', name, value, labels or {})

    def set_counter(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a counter metric."""
        key = (name, frozenset((labels or {}).items()))
        self.metrics[key] = ('counter', name, value, labels or {})

    def push(self):
        """Push metrics to Prometheus Pushgateway."""
        if not self.pushgateway_url:
            logger.debug("Pushgateway URL not configured, skipping metrics push")
            return

        try:
            # Format metrics in Prometheus exposition format
            lines = []
            for _, (metric_type, name, value, labels) in self.metrics.items():
                label_str = ','.join(f'{k}="{v}"' for k, v in labels.items())
                if label_str:
                    lines.append(f'{name}{{{label_str}}} {value}')
                else:
                    lines.append(f'{name} {value}')

            payload = '\n'.join(lines)

            import urllib.request
            url = f"{self.pushgateway_url}/metrics/job/{self.job_name}"
            req = urllib.request.Request(url, data=payload.encode('utf-8'), method='POST')
            req.add_header('Content-Type', 'text/plain')
            urllib.request.urlopen(req, timeout=30)
            logger.info(f"Pushed {len(self.metrics)} metrics to Pushgateway")
        except Exception as e:
            logger.warning(f"Failed to push metrics to Pushgateway: {e}")


class NotificationService:
    """Service for sending notifications."""

    def __init__(self, slack_webhook_url: Optional[str] = None,
                 generic_webhook_url: Optional[str] = None):
        self.slack_webhook_url = slack_webhook_url
        self.generic_webhook_url = generic_webhook_url

    def send_slack(self, message: str, attachments: Optional[List[Dict]] = None):
        """Send Slack notification."""
        if not self.slack_webhook_url:
            logger.debug("Slack webhook URL not configured")
            return

        try:
            import urllib.request
            payload = {'text': message}
            if attachments:
                payload['attachments'] = attachments

            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                self.slack_webhook_url,
                data=data,
                headers={'Content-Type': 'application/json'}
            )
            urllib.request.urlopen(req, timeout=30)
            logger.info("Sent Slack notification")
        except Exception as e:
            logger.warning(f"Failed to send Slack notification: {e}")

    def send_webhook(self, payload: Dict[str, Any]):
        """Send generic webhook notification."""
        if not self.generic_webhook_url:
            logger.debug("Generic webhook URL not configured")
            return

        try:
            import urllib.request
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                self.generic_webhook_url,
                data=data,
                headers={'Content-Type': 'application/json'}
            )
            urllib.request.urlopen(req, timeout=30)
            logger.info("Sent webhook notification")
        except Exception as e:
            logger.warning(f"Failed to send webhook notification: {e}")

    def send_completion_notification(self, stats: Dict[str, CleanupStats],
                                     success: bool, duration: float):
        """Send completion notification with summary."""
        total_deleted = sum(s.objects_deleted for s in stats.values())
        total_bytes = sum(s.bytes_freed for s in stats.values())
        total_failed = sum(s.objects_failed for s in stats.values())

        # Format bytes for readability
        def format_bytes(size: int) -> str:
            for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                if abs(size) < 1024.0:
                    return f"{size:.2f} {unit}"
                size /= 1024.0
            return f"{size:.2f} PB"

        status_emoji = ":white_check_mark:" if success else ":x:"
        status_text = "Completed" if success else "Failed"

        message = (
            f"{status_emoji} *Artifact Cleanup {status_text}*\n"
            f"*Duration:* {duration:.1f} seconds\n"
            f"*Objects Deleted:* {total_deleted:,}\n"
            f"*Space Freed:* {format_bytes(total_bytes)}\n"
        )

        if total_failed > 0:
            message += f"*Failed Objects:* {total_failed:,}\n"

        # Add per-target breakdown
        attachments = []
        for target_name, target_stats in stats.items():
            attachment = {
                'color': 'good' if target_stats.objects_failed == 0 else 'warning',
                'title': target_name,
                'fields': [
                    {'title': 'Scanned', 'value': str(target_stats.objects_scanned), 'short': True},
                    {'title': 'Deleted', 'value': str(target_stats.objects_deleted), 'short': True},
                    {'title': 'Space Freed', 'value': format_bytes(target_stats.bytes_freed), 'short': True},
                    {'title': 'Skipped', 'value': str(target_stats.objects_skipped), 'short': True},
                ]
            }
            attachments.append(attachment)

        self.send_slack(message, attachments)

        # Also send webhook
        webhook_payload = {
            'event': 'artifact_cleanup_completed',
            'success': success,
            'duration_seconds': duration,
            'total_deleted': total_deleted,
            'total_bytes_freed': total_bytes,
            'total_failed': total_failed,
            'targets': {name: s.to_dict() for name, s in stats.items()},
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        self.send_webhook(webhook_payload)


class ArtifactCleaner:
    """Main artifact cleanup class."""

    def __init__(self, config: Dict[str, Any], dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.global_config = config.get('global', {})

        # Initialize S3 client with retry configuration
        boto_config = Config(
            retries={'max_attempts': 3, 'mode': 'adaptive'},
            max_pool_connections=50
        )
        self.s3_client = boto3.client('s3', config=boto_config)

        # Initialize services
        self.metrics = PrometheusMetrics(
            pushgateway_url=os.environ.get('PUSHGATEWAY_URL'),
            job_name='artifact_cleanup'
        )
        self.notifications = NotificationService(
            slack_webhook_url=os.environ.get('SLACK_WEBHOOK_URL'),
            generic_webhook_url=os.environ.get('WEBHOOK_URL')
        )

        # Parse targets
        self.targets = self._parse_targets()

        # Batch settings
        self.batch_size = self.global_config.get('batch_size', 1000)
        self.max_workers = self.global_config.get('max_workers', 10)

        logger.info(f"Initialized ArtifactCleaner with {len(self.targets)} targets")
        logger.info(f"Dry run mode: {self.dry_run}")

    def _parse_targets(self) -> List[CleanupTarget]:
        """Parse cleanup targets from configuration."""
        targets = []
        for target_config in self.config.get('targets', []):
            # Expand environment variables in bucket name
            bucket = os.path.expandvars(target_config['bucket'])
            target = CleanupTarget(
                name=target_config['name'],
                bucket=bucket,
                prefixes=target_config.get('prefixes', ['']),
                retention_days=target_config['retention_days'],
                exclude_patterns=target_config.get('exclude_patterns', []),
                filters=target_config.get('filters', [])
            )
            targets.append(target)
        return targets

    def _matches_exclude_pattern(self, key: str, patterns: List[str]) -> bool:
        """Check if key matches any exclude pattern."""
        for pattern in patterns:
            # Convert glob pattern to regex
            regex = pattern.replace('.', r'\.').replace('*', '.*')
            if re.match(regex, key):
                return True
        return False

    def _passes_tag_filter(self, bucket: str, key: str,
                          filter_config: Dict[str, Any]) -> bool:
        """Check if object passes tag-based filter."""
        try:
            response = self.s3_client.get_object_tagging(Bucket=bucket, Key=key)
            tags = {tag['Key']: tag['Value'] for tag in response.get('TagSet', [])}

            filter_key = filter_config['key']
            filter_value = filter_config['value']
            condition = filter_config.get('condition', 'equals')

            actual_value = tags.get(filter_key, '')

            if condition == 'equals':
                return actual_value == filter_value
            elif condition == 'not_equals':
                return actual_value != filter_value
            elif condition == 'contains':
                return filter_value in actual_value
            elif condition == 'exists':
                return filter_key in tags
            elif condition == 'not_exists':
                return filter_key not in tags
            else:
                logger.warning(f"Unknown filter condition: {condition}")
                return True
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return True  # Object doesn't exist, skip
            logger.warning(f"Failed to get tags for {key}: {e}")
            return True  # Allow on error

    def _check_legal_hold(self, bucket: str, key: str) -> bool:
        """Check if object has legal hold."""
        try:
            response = self.s3_client.get_object_legal_hold(Bucket=bucket, Key=key)
            return response.get('LegalHold', {}).get('Status') == 'ON'
        except ClientError as e:
            # Legal hold not enabled on bucket or object
            if e.response['Error']['Code'] in ['ObjectLockConfigurationNotFoundError',
                                                  'NoSuchKey']:
                return False
            logger.debug(f"Legal hold check for {key}: {e}")
            return False

    def _should_delete(self, bucket: str, key: str, last_modified: datetime,
                       target: CleanupTarget) -> Tuple[bool, str]:
        """Determine if an object should be deleted."""
        # Check retention period
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=target.retention_days)
        if last_modified > cutoff_date:
            return False, f"Object not old enough (modified {last_modified})"

        # Check exclude patterns
        if self._matches_exclude_pattern(key, target.exclude_patterns):
            return False, f"Matches exclude pattern"

        # Check legal hold
        if self._check_legal_hold(bucket, key):
            return False, "Object has legal hold"

        # Check tag filters
        for filter_config in target.filters:
            if filter_config.get('type') == 'tag':
                if not self._passes_tag_filter(bucket, key, filter_config):
                    return False, f"Failed tag filter: {filter_config['key']}"

        return True, "Eligible for deletion"

    def _delete_objects_batch(self, bucket: str, keys: List[str]) -> Tuple[int, int, List[str]]:
        """Delete a batch of objects."""
        if not keys:
            return 0, 0, []

        if self.dry_run:
            logger.info(f"[DRY RUN] Would delete {len(keys)} objects from {bucket}")
            return len(keys), 0, []

        try:
            delete_request = {
                'Objects': [{'Key': key} for key in keys],
                'Quiet': False
            }
            response = self.s3_client.delete_objects(Bucket=bucket, Delete=delete_request)

            deleted = len(response.get('Deleted', []))
            errors = response.get('Errors', [])
            failed = len(errors)
            error_messages = [f"{e['Key']}: {e.get('Message', 'Unknown error')}" for e in errors]

            return deleted, failed, error_messages
        except ClientError as e:
            logger.error(f"Failed to delete batch from {bucket}: {e}")
            return 0, len(keys), [str(e)]

    def _cleanup_target(self, target: CleanupTarget) -> CleanupStats:
        """Clean up a single target."""
        stats = CleanupStats()
        start_time = time.time()

        logger.info(f"Processing target: {target.name}")
        logger.info(f"  Bucket: {target.bucket}")
        logger.info(f"  Prefixes: {target.prefixes}")
        logger.info(f"  Retention: {target.retention_days} days")

        objects_to_delete: List[Tuple[str, int]] = []  # (key, size)

        for prefix in target.prefixes:
            logger.info(f"  Scanning prefix: {prefix}")

            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=target.bucket,
                Prefix=prefix
            )

            for page in page_iterator:
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    size = obj['Size']
                    last_modified = obj['LastModified']

                    stats.objects_scanned += 1
                    stats.bytes_scanned += size

                    should_delete, reason = self._should_delete(
                        target.bucket, key, last_modified, target
                    )

                    if should_delete:
                        objects_to_delete.append((key, size))
                    else:
                        stats.objects_skipped += 1
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"    Skipping {key}: {reason}")

                    # Process batch when threshold reached
                    if len(objects_to_delete) >= self.batch_size:
                        keys = [k for k, _ in objects_to_delete]
                        sizes = sum(s for _, s in objects_to_delete)

                        deleted, failed, errors = self._delete_objects_batch(
                            target.bucket, keys
                        )

                        stats.objects_deleted += deleted
                        stats.objects_failed += failed
                        stats.bytes_freed += sizes if deleted == len(keys) else 0
                        stats.errors.extend(errors)

                        objects_to_delete = []

        # Process remaining objects
        if objects_to_delete:
            keys = [k for k, _ in objects_to_delete]
            sizes = sum(s for _, s in objects_to_delete)

            deleted, failed, errors = self._delete_objects_batch(
                target.bucket, keys
            )

            stats.objects_deleted += deleted
            stats.objects_failed += failed
            stats.bytes_freed += sizes if deleted == len(keys) else 0
            stats.errors.extend(errors)

        stats.duration_seconds = time.time() - start_time

        logger.info(f"  Target {target.name} completed:")
        logger.info(f"    Scanned: {stats.objects_scanned:,}")
        logger.info(f"    Deleted: {stats.objects_deleted:,}")
        logger.info(f"    Skipped: {stats.objects_skipped:,}")
        logger.info(f"    Failed: {stats.objects_failed:,}")
        logger.info(f"    Bytes freed: {stats.bytes_freed:,}")
        logger.info(f"    Duration: {stats.duration_seconds:.2f}s")

        return stats

    def run(self) -> Dict[str, CleanupStats]:
        """Run cleanup for all targets."""
        start_time = time.time()
        all_stats: Dict[str, CleanupStats] = {}
        success = True

        logger.info("=" * 60)
        logger.info("Starting artifact cleanup")
        logger.info(f"Dry run: {self.dry_run}")
        logger.info(f"Targets: {len(self.targets)}")
        logger.info("=" * 60)

        try:
            # Process targets (can be parallelized if needed)
            for target in self.targets:
                try:
                    stats = self._cleanup_target(target)
                    all_stats[target.name] = stats

                    # Update metrics
                    labels = {'target': target.name, 'bucket': target.bucket}
                    self.metrics.set_gauge('artifact_cleanup_objects_scanned',
                                          stats.objects_scanned, labels)
                    self.metrics.set_gauge('artifact_cleanup_objects_deleted',
                                          stats.objects_deleted, labels)
                    self.metrics.set_gauge('artifact_cleanup_bytes_freed',
                                          stats.bytes_freed, labels)
                    self.metrics.set_gauge('artifact_cleanup_objects_failed',
                                          stats.objects_failed, labels)

                    if stats.objects_failed > 0:
                        success = False
                except Exception as e:
                    logger.error(f"Failed to process target {target.name}: {e}")
                    all_stats[target.name] = CleanupStats(errors=[str(e)])
                    success = False

            # Calculate totals
            total_duration = time.time() - start_time
            total_deleted = sum(s.objects_deleted for s in all_stats.values())
            total_bytes = sum(s.bytes_freed for s in all_stats.values())
            total_failed = sum(s.objects_failed for s in all_stats.values())

            # Set total metrics
            self.metrics.set_gauge('artifact_cleanup_total_duration_seconds', total_duration)
            self.metrics.set_gauge('artifact_cleanup_total_objects_deleted', total_deleted)
            self.metrics.set_gauge('artifact_cleanup_total_bytes_freed', total_bytes)
            self.metrics.set_gauge('artifact_cleanup_success', 1 if success else 0)

            # Push metrics
            if self.global_config.get('metrics_enabled', True):
                self.metrics.push()

            # Send notifications
            self.notifications.send_completion_notification(all_stats, success, total_duration)

            # Export report to S3 if configured
            self._export_report(all_stats, total_duration, success)

            logger.info("=" * 60)
            logger.info("Cleanup Summary")
            logger.info("=" * 60)
            logger.info(f"Total duration: {total_duration:.2f} seconds")
            logger.info(f"Total objects deleted: {total_deleted:,}")
            logger.info(f"Total bytes freed: {total_bytes:,}")
            logger.info(f"Total failed: {total_failed:,}")
            logger.info(f"Success: {success}")
            logger.info("=" * 60)

            return all_stats

        except Exception as e:
            logger.error(f"Cleanup failed with error: {e}")
            raise

    def _export_report(self, stats: Dict[str, CleanupStats],
                       duration: float, success: bool):
        """Export cleanup report to S3."""
        reporting_config = self.config.get('reporting', {})
        export_config = reporting_config.get('export_to_s3', {})

        if not export_config.get('enabled', False):
            return

        try:
            bucket = os.path.expandvars(export_config.get('bucket', ''))
            prefix = export_config.get('prefix', 'reports/cleanup/')

            if not bucket:
                logger.warning("Export bucket not configured")
                return

            timestamp = datetime.now(timezone.utc)
            report_key = f"{prefix}{timestamp.strftime('%Y/%m/%d')}/cleanup-report-{timestamp.strftime('%H%M%S')}.json"

            report = {
                'timestamp': timestamp.isoformat(),
                'duration_seconds': duration,
                'success': success,
                'dry_run': self.dry_run,
                'targets': {name: s.to_dict() for name, s in stats.items()},
                'totals': {
                    'objects_deleted': sum(s.objects_deleted for s in stats.values()),
                    'bytes_freed': sum(s.bytes_freed for s in stats.values()),
                    'objects_failed': sum(s.objects_failed for s in stats.values()),
                }
            }

            if self.dry_run:
                logger.info(f"[DRY RUN] Would export report to s3://{bucket}/{report_key}")
                return

            self.s3_client.put_object(
                Bucket=bucket,
                Key=report_key,
                Body=json.dumps(report, indent=2),
                ContentType='application/json',
                Metadata={
                    'report_type': 'cleanup',
                    'success': str(success).lower()
                }
            )
            logger.info(f"Exported report to s3://{bucket}/{report_key}")

        except Exception as e:
            logger.warning(f"Failed to export report: {e}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Artifact cleanup script for GreenLang storage lifecycle management'
    )
    parser.add_argument(
        '--config',
        required=True,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=False,
        help='Run in dry-run mode (no actual deletions)'
    )
    parser.add_argument(
        '--metrics-port',
        type=int,
        default=9090,
        help='Port for metrics server'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )

    args = parser.parse_args()

    # Set log level
    logger.setLevel(getattr(logging, args.log_level))

    # Check for dry run override from environment
    dry_run = args.dry_run or os.environ.get('DRY_RUN', '').lower() == 'true'

    try:
        # Load configuration
        config = load_config(args.config)

        # Override dry_run from config if not set via CLI
        if not dry_run:
            dry_run = config.get('global', {}).get('dry_run', False)

        # Initialize and run cleaner
        cleaner = ArtifactCleaner(config, dry_run=dry_run)
        stats = cleaner.run()

        # Exit with error if any target failed
        total_failed = sum(s.objects_failed for s in stats.values())
        if total_failed > 0:
            logger.warning(f"Cleanup completed with {total_failed} failures")
            sys.exit(1)

        logger.info("Cleanup completed successfully")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
