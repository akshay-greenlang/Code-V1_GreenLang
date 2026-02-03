#!/usr/bin/env python3
"""
Artifact Archive Script for GreenLang Storage Lifecycle Management

This script manages storage class transitions for archiving artifacts to
Glacier and Deep Archive storage classes. It handles batch operations,
cost optimization, and generates archive manifests.

Features:
- Storage class transitions (STANDARD -> GLACIER, DEEP_ARCHIVE)
- Glacier restore handling
- Batch operations for efficiency using S3 Batch Operations
- Cost optimization tracking
- Archive manifest generation
- Slack/webhook notifications

Usage:
    python archive-artifacts.py --config /config/archive-config.yaml [--dry-run]

Author: GreenLang DevOps Team
Version: 1.0.0
"""

import argparse
import json
import logging
import os
import sys
import time
import uuid
import yaml
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
logger = logging.getLogger('artifact-archive')


@dataclass
class ArchiveStats:
    """Statistics for archive operations."""
    objects_scanned: int = 0
    objects_archived: int = 0
    objects_skipped: int = 0
    objects_failed: int = 0
    bytes_archived: int = 0
    bytes_scanned: int = 0
    duration_seconds: float = 0.0
    estimated_cost_savings: float = 0.0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            'objects_scanned': self.objects_scanned,
            'objects_archived': self.objects_archived,
            'objects_skipped': self.objects_skipped,
            'objects_failed': self.objects_failed,
            'bytes_archived': self.bytes_archived,
            'bytes_scanned': self.bytes_scanned,
            'duration_seconds': round(self.duration_seconds, 2),
            'estimated_cost_savings': round(self.estimated_cost_savings, 4),
            'errors': self.errors[:100]
        }


@dataclass
class ArchiveTarget:
    """Configuration for an archive target."""
    name: str
    source_bucket: str
    prefixes: List[str]
    archive_after_days: int
    target_storage_class: str
    min_object_size_bytes: int = 0
    filters: List[Dict[str, Any]] = field(default_factory=list)
    metadata_updates: Dict[str, str] = field(default_factory=dict)


@dataclass
class ManifestEntry:
    """Entry in the archive manifest."""
    object_key: str
    original_size: int
    original_storage_class: str
    target_storage_class: str
    archive_timestamp: str
    object_tags: Dict[str, str]
    etag: str


class CostTracker:
    """Track cost savings from archival."""

    # Approximate cost per GB per month (US East)
    STORAGE_COSTS = {
        'STANDARD': 0.023,
        'STANDARD_IA': 0.0125,
        'ONEZONE_IA': 0.01,
        'GLACIER': 0.004,
        'GLACIER_IR': 0.01,
        'DEEP_ARCHIVE': 0.00099
    }

    def __init__(self):
        self.transitions: List[Tuple[str, str, int]] = []  # (from, to, bytes)

    def add_transition(self, from_class: str, to_class: str, size_bytes: int):
        """Record a storage class transition."""
        self.transitions.append((from_class, to_class, size_bytes))

    def calculate_monthly_savings(self) -> float:
        """Calculate estimated monthly cost savings."""
        total_savings = 0.0
        for from_class, to_class, size_bytes in self.transitions:
            size_gb = size_bytes / (1024 ** 3)
            from_cost = self.STORAGE_COSTS.get(from_class, 0.023)
            to_cost = self.STORAGE_COSTS.get(to_class, 0.004)
            savings = (from_cost - to_cost) * size_gb
            total_savings += savings
        return total_savings

    def get_summary(self) -> Dict[str, Any]:
        """Get cost savings summary."""
        total_bytes = sum(b for _, _, b in self.transitions)
        return {
            'total_transitions': len(self.transitions),
            'total_bytes': total_bytes,
            'total_gb': round(total_bytes / (1024 ** 3), 2),
            'estimated_monthly_savings': round(self.calculate_monthly_savings(), 4),
            'estimated_yearly_savings': round(self.calculate_monthly_savings() * 12, 2)
        }


class NotificationService:
    """Service for sending notifications."""

    def __init__(self, slack_webhook_url: Optional[str] = None):
        self.slack_webhook_url = slack_webhook_url

    def send_slack(self, message: str, attachments: Optional[List[Dict]] = None):
        """Send Slack notification."""
        if not self.slack_webhook_url:
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

    def send_completion_notification(self, stats: Dict[str, ArchiveStats],
                                     cost_summary: Dict[str, Any],
                                     success: bool, duration: float):
        """Send completion notification."""
        total_archived = sum(s.objects_archived for s in stats.values())
        total_bytes = sum(s.bytes_archived for s in stats.values())

        def format_bytes(size: int) -> str:
            for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                if abs(size) < 1024.0:
                    return f"{size:.2f} {unit}"
                size /= 1024.0
            return f"{size:.2f} PB"

        status_emoji = ":white_check_mark:" if success else ":x:"
        status_text = "Completed" if success else "Failed"

        message = (
            f"{status_emoji} *Artifact Archive {status_text}*\n"
            f"*Duration:* {duration:.1f} seconds\n"
            f"*Objects Archived:* {total_archived:,}\n"
            f"*Data Archived:* {format_bytes(total_bytes)}\n"
            f"*Est. Monthly Savings:* ${cost_summary.get('estimated_monthly_savings', 0):.2f}\n"
        )

        self.send_slack(message)


class ArtifactArchiver:
    """Main artifact archiver class."""

    def __init__(self, config: Dict[str, Any], dry_run: bool = False,
                 generate_manifest: bool = True):
        self.config = config
        self.dry_run = dry_run
        self.generate_manifest = generate_manifest
        self.global_config = config.get('global', {})

        # Initialize S3 client
        boto_config = Config(
            retries={'max_attempts': 3, 'mode': 'adaptive'},
            max_pool_connections=50
        )
        self.s3_client = boto3.client('s3', config=boto_config)

        # Initialize services
        self.notifications = NotificationService(
            slack_webhook_url=os.environ.get('SLACK_WEBHOOK_URL')
        )
        self.cost_tracker = CostTracker()

        # Parse targets
        self.targets = self._parse_targets()

        # Batch settings
        self.batch_size = self.global_config.get('batch_size', 500)
        self.use_batch_operations = self.global_config.get('use_batch_operations', True)
        self.batch_operation_threshold = self.global_config.get('batch_operation_threshold', 10000)

        # Manifest entries
        self.manifest_entries: List[ManifestEntry] = []

        logger.info(f"Initialized ArtifactArchiver with {len(self.targets)} targets")
        logger.info(f"Dry run: {self.dry_run}")

    def _parse_targets(self) -> List[ArchiveTarget]:
        """Parse archive targets from configuration."""
        targets = []
        for target_config in self.config.get('targets', []):
            bucket = os.path.expandvars(target_config['source_bucket'])
            target = ArchiveTarget(
                name=target_config['name'],
                source_bucket=bucket,
                prefixes=target_config.get('prefixes', ['']),
                archive_after_days=target_config['archive_after_days'],
                target_storage_class=target_config['target_storage_class'],
                min_object_size_bytes=target_config.get('min_object_size_bytes', 0),
                filters=target_config.get('filters', []),
                metadata_updates=target_config.get('metadata_updates', {})
            )
            targets.append(target)
        return targets

    def _get_object_storage_class(self, bucket: str, key: str) -> Optional[str]:
        """Get current storage class of an object."""
        try:
            response = self.s3_client.head_object(Bucket=bucket, Key=key)
            return response.get('StorageClass', 'STANDARD')
        except ClientError:
            return None

    def _get_object_tags(self, bucket: str, key: str) -> Dict[str, str]:
        """Get object tags."""
        try:
            response = self.s3_client.get_object_tagging(Bucket=bucket, Key=key)
            return {tag['Key']: tag['Value'] for tag in response.get('TagSet', [])}
        except ClientError:
            return {}

    def _passes_filter(self, bucket: str, key: str, storage_class: str,
                       filter_config: Dict[str, Any]) -> bool:
        """Check if object passes a filter."""
        filter_type = filter_config.get('type')

        if filter_type == 'tag':
            tags = self._get_object_tags(bucket, key)
            filter_key = filter_config['key']
            filter_value = filter_config['value']
            condition = filter_config.get('condition', 'equals')

            actual_value = tags.get(filter_key, '')

            if condition == 'equals':
                return actual_value == filter_value
            elif condition == 'not_equals':
                return actual_value != filter_value
            else:
                return True

        elif filter_type == 'storage_class':
            filter_value = filter_config['value']
            condition = filter_config.get('condition', 'equals')

            if condition == 'equals':
                return storage_class == filter_value
            elif condition == 'not_equals':
                return storage_class != filter_value
            else:
                return True

        return True

    def _should_archive(self, bucket: str, key: str, obj: Dict[str, Any],
                        target: ArchiveTarget) -> Tuple[bool, str]:
        """Determine if an object should be archived."""
        size = obj['Size']
        last_modified = obj['LastModified']
        storage_class = obj.get('StorageClass', 'STANDARD')

        # Check age
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=target.archive_after_days)
        if last_modified > cutoff_date:
            return False, f"Object not old enough (modified {last_modified})"

        # Check minimum size
        if size < target.min_object_size_bytes:
            return False, f"Object too small ({size} < {target.min_object_size_bytes})"

        # Check if already in target storage class
        if storage_class == target.target_storage_class:
            return False, f"Already in {target.target_storage_class}"

        # Check filters
        for filter_config in target.filters:
            if not self._passes_filter(bucket, key, storage_class, filter_config):
                return False, f"Failed filter: {filter_config.get('type')}"

        return True, "Eligible for archive"

    def _transition_storage_class(self, bucket: str, key: str,
                                   target_storage_class: str,
                                   metadata_updates: Dict[str, str]) -> bool:
        """Transition object to new storage class."""
        try:
            # Get current metadata
            head = self.s3_client.head_object(Bucket=bucket, Key=key)
            current_metadata = head.get('Metadata', {})

            # Merge metadata updates
            new_metadata = current_metadata.copy()
            for k, v in metadata_updates.items():
                # Handle timestamp placeholder
                if v == '${TIMESTAMP}':
                    v = datetime.now(timezone.utc).isoformat()
                new_metadata[k] = v

            # Copy object to itself with new storage class
            copy_source = {'Bucket': bucket, 'Key': key}
            self.s3_client.copy_object(
                CopySource=copy_source,
                Bucket=bucket,
                Key=key,
                StorageClass=target_storage_class,
                Metadata=new_metadata,
                MetadataDirective='REPLACE'
            )

            return True
        except ClientError as e:
            logger.error(f"Failed to transition {key}: {e}")
            return False

    def _archive_target(self, target: ArchiveTarget) -> ArchiveStats:
        """Archive objects for a single target."""
        stats = ArchiveStats()
        start_time = time.time()

        logger.info(f"Processing target: {target.name}")
        logger.info(f"  Bucket: {target.source_bucket}")
        logger.info(f"  Target storage class: {target.target_storage_class}")
        logger.info(f"  Archive after: {target.archive_after_days} days")

        for prefix in target.prefixes:
            logger.info(f"  Scanning prefix: {prefix}")

            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=target.source_bucket,
                Prefix=prefix
            )

            for page in page_iterator:
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    size = obj['Size']
                    storage_class = obj.get('StorageClass', 'STANDARD')

                    stats.objects_scanned += 1
                    stats.bytes_scanned += size

                    should_archive, reason = self._should_archive(
                        target.source_bucket, key, obj, target
                    )

                    if not should_archive:
                        stats.objects_skipped += 1
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"    Skipping {key}: {reason}")
                        continue

                    if self.dry_run:
                        logger.info(f"    [DRY RUN] Would archive: {key}")
                        stats.objects_archived += 1
                        stats.bytes_archived += size
                        self.cost_tracker.add_transition(
                            storage_class, target.target_storage_class, size
                        )
                        continue

                    # Perform transition
                    success = self._transition_storage_class(
                        target.source_bucket, key,
                        target.target_storage_class,
                        target.metadata_updates
                    )

                    if success:
                        stats.objects_archived += 1
                        stats.bytes_archived += size
                        self.cost_tracker.add_transition(
                            storage_class, target.target_storage_class, size
                        )

                        # Add to manifest
                        if self.generate_manifest:
                            tags = self._get_object_tags(target.source_bucket, key)
                            entry = ManifestEntry(
                                object_key=key,
                                original_size=size,
                                original_storage_class=storage_class,
                                target_storage_class=target.target_storage_class,
                                archive_timestamp=datetime.now(timezone.utc).isoformat(),
                                object_tags=tags,
                                etag=obj.get('ETag', '')
                            )
                            self.manifest_entries.append(entry)
                    else:
                        stats.objects_failed += 1
                        stats.errors.append(f"Failed to transition: {key}")

        stats.duration_seconds = time.time() - start_time
        stats.estimated_cost_savings = self.cost_tracker.calculate_monthly_savings()

        logger.info(f"  Target {target.name} completed:")
        logger.info(f"    Scanned: {stats.objects_scanned:,}")
        logger.info(f"    Archived: {stats.objects_archived:,}")
        logger.info(f"    Skipped: {stats.objects_skipped:,}")
        logger.info(f"    Failed: {stats.objects_failed:,}")
        logger.info(f"    Duration: {stats.duration_seconds:.2f}s")

        return stats

    def _generate_archive_manifest(self, stats: Dict[str, ArchiveStats]):
        """Generate and upload archive manifest."""
        manifest_config = self.config.get('manifest', {})

        if not manifest_config.get('enabled', True):
            return

        bucket = os.path.expandvars(manifest_config.get('bucket', ''))
        prefix = manifest_config.get('prefix', 'manifests/archive/')

        if not bucket:
            logger.warning("Manifest bucket not configured")
            return

        timestamp = datetime.now(timezone.utc)
        manifest_key = f"{prefix}{timestamp.strftime('%Y/%m/%d')}/archive-manifest-{timestamp.strftime('%H%M%S')}.json"

        manifest = {
            'timestamp': timestamp.isoformat(),
            'total_entries': len(self.manifest_entries),
            'summary': {
                name: s.to_dict() for name, s in stats.items()
            },
            'cost_savings': self.cost_tracker.get_summary(),
            'entries': [
                {
                    'object_key': e.object_key,
                    'original_size': e.original_size,
                    'original_storage_class': e.original_storage_class,
                    'target_storage_class': e.target_storage_class,
                    'archive_timestamp': e.archive_timestamp,
                    'etag': e.etag
                }
                for e in self.manifest_entries
            ]
        }

        if self.dry_run:
            logger.info(f"[DRY RUN] Would create manifest at s3://{bucket}/{manifest_key}")
            return

        try:
            self.s3_client.put_object(
                Bucket=bucket,
                Key=manifest_key,
                Body=json.dumps(manifest, indent=2),
                ContentType='application/json'
            )
            logger.info(f"Created manifest at s3://{bucket}/{manifest_key}")
        except Exception as e:
            logger.warning(f"Failed to create manifest: {e}")

    def run(self) -> Dict[str, ArchiveStats]:
        """Run archive for all targets."""
        start_time = time.time()
        all_stats: Dict[str, ArchiveStats] = {}
        success = True

        logger.info("=" * 60)
        logger.info("Starting artifact archive")
        logger.info(f"Dry run: {self.dry_run}")
        logger.info(f"Targets: {len(self.targets)}")
        logger.info("=" * 60)

        try:
            for target in self.targets:
                try:
                    stats = self._archive_target(target)
                    all_stats[target.name] = stats

                    if stats.objects_failed > 0:
                        success = False
                except Exception as e:
                    logger.error(f"Failed to process target {target.name}: {e}")
                    all_stats[target.name] = ArchiveStats(errors=[str(e)])
                    success = False

            # Generate manifest
            if self.generate_manifest:
                self._generate_archive_manifest(all_stats)

            # Calculate totals
            total_duration = time.time() - start_time
            cost_summary = self.cost_tracker.get_summary()

            # Send notifications
            self.notifications.send_completion_notification(
                all_stats, cost_summary, success, total_duration
            )

            logger.info("=" * 60)
            logger.info("Archive Summary")
            logger.info("=" * 60)
            logger.info(f"Total duration: {total_duration:.2f} seconds")
            logger.info(f"Total archived: {sum(s.objects_archived for s in all_stats.values()):,}")
            logger.info(f"Est. monthly savings: ${cost_summary.get('estimated_monthly_savings', 0):.2f}")
            logger.info(f"Success: {success}")
            logger.info("=" * 60)

            return all_stats

        except Exception as e:
            logger.error(f"Archive failed: {e}")
            raise


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Artifact archive script for GreenLang storage lifecycle'
    )
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode')
    parser.add_argument('--generate-manifest', action='store_true', default=True,
                        help='Generate archive manifest')
    parser.add_argument('--metrics-port', type=int, default=9090)
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO')

    args = parser.parse_args()
    logger.setLevel(getattr(logging, args.log_level))

    dry_run = args.dry_run or os.environ.get('DRY_RUN', '').lower() == 'true'

    try:
        config = load_config(args.config)
        if not dry_run:
            dry_run = config.get('global', {}).get('dry_run', False)

        archiver = ArtifactArchiver(config, dry_run=dry_run,
                                    generate_manifest=args.generate_manifest)
        stats = archiver.run()

        total_failed = sum(s.objects_failed for s in stats.values())
        sys.exit(1 if total_failed > 0 else 0)

    except Exception as e:
        logger.error(f"Archive failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
