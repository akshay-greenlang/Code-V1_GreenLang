#!/usr/bin/env python3
"""
Retention Enforcement Script for GreenLang Storage Lifecycle Management

This script enforces retention policies per artifact type, handles legal holds,
generates compliance reports, and alerts on policy violations.

Features:
- Policy evaluation per artifact type
- Legal hold management via DynamoDB
- Compliance reporting (CSRD, SOX)
- Alert generation for violations
- Immutability enforcement

Usage:
    python enforce-retention.py --config /config/retention-config.yaml [--dry-run]

Author: GreenLang DevOps Team
Version: 1.0.0
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import boto3
import yaml
from botocore.config import Config
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('retention-enforcement')


@dataclass
class RetentionStats:
    """Statistics for retention enforcement."""
    objects_evaluated: int = 0
    objects_compliant: int = 0
    objects_non_compliant: int = 0
    objects_with_legal_hold: int = 0
    objects_pending_deletion: int = 0
    objects_pending_archive: int = 0
    violations: List[Dict[str, Any]] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'objects_evaluated': self.objects_evaluated,
            'objects_compliant': self.objects_compliant,
            'objects_non_compliant': self.objects_non_compliant,
            'objects_with_legal_hold': self.objects_with_legal_hold,
            'objects_pending_deletion': self.objects_pending_deletion,
            'objects_pending_archive': self.objects_pending_archive,
            'violations_count': len(self.violations),
            'duration_seconds': round(self.duration_seconds, 2)
        }


@dataclass
class RetentionPolicy:
    """Retention policy configuration."""
    name: str
    artifact_type: str
    bucket: str
    prefixes: List[str]
    active_days: int
    archive_days: Optional[int]
    delete_days: Optional[int]
    archive_storage_class: Optional[str]
    compliance_required: bool
    compliance_framework: Optional[str]
    immutable: bool
    legal_hold_enabled: bool


@dataclass
class LegalHold:
    """Legal hold record."""
    hold_id: str
    artifact_key: str
    bucket: str
    hold_type: str
    created_at: str
    created_by: str
    reason: str
    expires_at: Optional[str] = None


class LegalHoldManager:
    """Manager for legal holds via DynamoDB."""

    def __init__(self, table_name: str):
        self.table_name = table_name
        self.dynamodb = boto3.resource('dynamodb')
        self.table = self.dynamodb.Table(table_name)

    def get_holds_for_object(self, bucket: str, key: str) -> List[LegalHold]:
        """Get all active legal holds for an object."""
        try:
            response = self.table.query(
                KeyConditionExpression='artifact_key = :key',
                FilterExpression='bucket_name = :bucket AND (expires_at = :none OR expires_at > :now)',
                ExpressionAttributeValues={
                    ':key': key,
                    ':bucket': bucket,
                    ':none': None,
                    ':now': datetime.now(timezone.utc).isoformat()
                }
            )
            holds = []
            for item in response.get('Items', []):
                hold = LegalHold(
                    hold_id=item.get('hold_id', ''),
                    artifact_key=item.get('artifact_key', ''),
                    bucket=item.get('bucket_name', ''),
                    hold_type=item.get('hold_type', ''),
                    created_at=item.get('created_at', ''),
                    created_by=item.get('created_by', ''),
                    reason=item.get('reason', ''),
                    expires_at=item.get('expires_at')
                )
                holds.append(hold)
            return holds
        except ClientError as e:
            logger.warning(f"Failed to query legal holds: {e}")
            return []

    def has_active_hold(self, bucket: str, key: str) -> bool:
        """Check if object has any active legal hold."""
        holds = self.get_holds_for_object(bucket, key)
        return len(holds) > 0


class AlertService:
    """Service for sending alerts."""

    def __init__(self, slack_webhook: Optional[str] = None,
                 pagerduty_key: Optional[str] = None):
        self.slack_webhook = slack_webhook
        self.pagerduty_key = pagerduty_key

    def send_slack_alert(self, message: str, severity: str = 'warning'):
        """Send Slack alert."""
        if not self.slack_webhook:
            return

        try:
            import urllib.request
            color = {'critical': 'danger', 'warning': 'warning', 'info': 'good'}.get(severity, 'warning')
            payload = {
                'attachments': [{
                    'color': color,
                    'text': message,
                    'footer': 'GreenLang Retention Enforcement',
                    'ts': int(time.time())
                }]
            }
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(self.slack_webhook, data=data,
                                         headers={'Content-Type': 'application/json'})
            urllib.request.urlopen(req, timeout=30)
        except Exception as e:
            logger.warning(f"Failed to send Slack alert: {e}")

    def send_pagerduty_alert(self, summary: str, severity: str = 'warning'):
        """Send PagerDuty alert."""
        if not self.pagerduty_key:
            return

        try:
            import urllib.request
            pd_severity = {'critical': 'critical', 'warning': 'warning', 'info': 'info'}.get(severity, 'warning')
            payload = {
                'routing_key': self.pagerduty_key,
                'event_action': 'trigger',
                'payload': {
                    'summary': summary,
                    'severity': pd_severity,
                    'source': 'greenlang-retention-enforcement'
                }
            }
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request('https://events.pagerduty.com/v2/enqueue',
                                         data=data,
                                         headers={'Content-Type': 'application/json'})
            urllib.request.urlopen(req, timeout=30)
        except Exception as e:
            logger.warning(f"Failed to send PagerDuty alert: {e}")

    def alert_on_violation(self, violation: Dict[str, Any], severity: str = 'warning'):
        """Send alert for policy violation."""
        message = (
            f":warning: *Retention Policy Violation*\n"
            f"*Artifact:* `{violation.get('key')}`\n"
            f"*Bucket:* {violation.get('bucket')}\n"
            f"*Policy:* {violation.get('policy_name')}\n"
            f"*Violation:* {violation.get('violation_type')}\n"
            f"*Details:* {violation.get('details')}"
        )
        self.send_slack_alert(message, severity)
        if severity == 'critical':
            self.send_pagerduty_alert(f"Retention violation: {violation.get('violation_type')}", severity)


class RetentionEnforcer:
    """Main retention enforcement class."""

    def __init__(self, config: Dict[str, Any], dry_run: bool = False,
                 generate_report: bool = True):
        self.config = config
        self.dry_run = dry_run
        self.generate_report = generate_report
        self.global_config = config.get('global', {})

        # Initialize AWS clients
        boto_config = Config(retries={'max_attempts': 3, 'mode': 'adaptive'})
        self.s3_client = boto3.client('s3', config=boto_config)

        # Initialize services
        legal_hold_table = os.environ.get('LEGAL_HOLD_TABLE', 'greenlang-legal-holds')
        self.legal_hold_manager = LegalHoldManager(legal_hold_table)

        self.alert_service = AlertService(
            slack_webhook=os.environ.get('SLACK_WEBHOOK_URL'),
            pagerduty_key=os.environ.get('PAGERDUTY_ROUTING_KEY')
        )

        # Parse policies
        self.policies = self._parse_policies()
        logger.info(f"Initialized RetentionEnforcer with {len(self.policies)} policies")

    def _parse_policies(self) -> List[RetentionPolicy]:
        """Parse retention policies from configuration."""
        policies = []
        for policy_config in self.config.get('policies', []):
            bucket = os.path.expandvars(policy_config['bucket'])
            retention = policy_config.get('retention', {})
            compliance = policy_config.get('compliance', {})
            legal_hold = policy_config.get('legal_hold', {})

            policy = RetentionPolicy(
                name=policy_config['name'],
                artifact_type=policy_config['artifact_type'],
                bucket=bucket,
                prefixes=policy_config.get('prefixes', ['']),
                active_days=retention.get('active_days', 30),
                archive_days=retention.get('archive_days'),
                delete_days=retention.get('delete_days'),
                archive_storage_class=retention.get('archive_storage_class'),
                compliance_required=compliance.get('required', False),
                compliance_framework=compliance.get('framework'),
                immutable=compliance.get('immutable', False),
                legal_hold_enabled=legal_hold.get('check_enabled', True)
            )
            policies.append(policy)
        return policies

    def _evaluate_object_compliance(self, bucket: str, key: str, obj: Dict[str, Any],
                                    policy: RetentionPolicy) -> Tuple[bool, List[Dict[str, Any]]]:
        """Evaluate if an object is compliant with its retention policy."""
        violations = []
        last_modified = obj['LastModified']
        storage_class = obj.get('StorageClass', 'STANDARD')
        age_days = (datetime.now(timezone.utc) - last_modified).days

        # Check if should be archived
        if policy.archive_days and age_days > policy.active_days:
            if policy.archive_storage_class and storage_class != policy.archive_storage_class:
                # Should be in archive storage class but isn't
                if age_days > policy.archive_days:
                    violations.append({
                        'key': key,
                        'bucket': bucket,
                        'policy_name': policy.name,
                        'violation_type': 'archive_overdue',
                        'details': f"Object is {age_days} days old, should be in {policy.archive_storage_class}"
                    })

        # Check if should be deleted (if delete policy exists and no compliance hold)
        if policy.delete_days and age_days > policy.delete_days:
            if not policy.compliance_required:
                violations.append({
                    'key': key,
                    'bucket': bucket,
                    'policy_name': policy.name,
                    'violation_type': 'deletion_overdue',
                    'details': f"Object is {age_days} days old, exceeds retention of {policy.delete_days} days"
                })

        # Check immutability violations
        if policy.immutable:
            # Check object lock status
            try:
                response = self.s3_client.get_object_retention(Bucket=bucket, Key=key)
                retention = response.get('Retention', {})
                if retention.get('Mode') != 'COMPLIANCE':
                    violations.append({
                        'key': key,
                        'bucket': bucket,
                        'policy_name': policy.name,
                        'violation_type': 'immutability_not_set',
                        'details': 'Object should be in COMPLIANCE retention mode'
                    })
            except ClientError as e:
                if 'ObjectLockConfigurationNotFoundError' not in str(e):
                    violations.append({
                        'key': key,
                        'bucket': bucket,
                        'policy_name': policy.name,
                        'violation_type': 'immutability_check_failed',
                        'details': str(e)
                    })

        return len(violations) == 0, violations

    def _enforce_policy(self, policy: RetentionPolicy) -> RetentionStats:
        """Enforce retention policy on all objects."""
        stats = RetentionStats()
        start_time = time.time()

        logger.info(f"Enforcing policy: {policy.name}")
        logger.info(f"  Bucket: {policy.bucket}")
        logger.info(f"  Artifact type: {policy.artifact_type}")

        for prefix in policy.prefixes:
            logger.info(f"  Scanning prefix: {prefix}")

            paginator = self.s3_client.get_paginator('list_objects_v2')
            try:
                page_iterator = paginator.paginate(Bucket=policy.bucket, Prefix=prefix)
            except ClientError as e:
                logger.error(f"Failed to list objects in {policy.bucket}/{prefix}: {e}")
                continue

            for page in page_iterator:
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    stats.objects_evaluated += 1

                    # Check legal hold
                    if policy.legal_hold_enabled:
                        has_hold = self.legal_hold_manager.has_active_hold(policy.bucket, key)
                        if has_hold:
                            stats.objects_with_legal_hold += 1
                            logger.debug(f"  {key}: Has active legal hold")
                            continue

                    # Evaluate compliance
                    compliant, violations = self._evaluate_object_compliance(
                        policy.bucket, key, obj, policy
                    )

                    if compliant:
                        stats.objects_compliant += 1
                    else:
                        stats.objects_non_compliant += 1
                        stats.violations.extend(violations)

                        # Send alerts for violations
                        for violation in violations:
                            severity = 'critical' if violation['violation_type'] == 'immutability_not_set' else 'warning'
                            self.alert_service.alert_on_violation(violation, severity)

        stats.duration_seconds = time.time() - start_time

        logger.info(f"  Policy {policy.name} completed:")
        logger.info(f"    Evaluated: {stats.objects_evaluated}")
        logger.info(f"    Compliant: {stats.objects_compliant}")
        logger.info(f"    Non-compliant: {stats.objects_non_compliant}")
        logger.info(f"    Legal holds: {stats.objects_with_legal_hold}")

        return stats

    def _generate_compliance_report(self, all_stats: Dict[str, RetentionStats]):
        """Generate compliance report."""
        reporting_config = self.config.get('compliance_reporting', {})
        if not reporting_config.get('enabled', True):
            return

        bucket = os.path.expandvars(reporting_config.get('bucket', ''))
        prefix = reporting_config.get('prefix', 'compliance/retention/')

        if not bucket:
            logger.warning("Compliance report bucket not configured")
            return

        timestamp = datetime.now(timezone.utc)
        report_key = f"{prefix}{timestamp.strftime('%Y/%m/%d')}/retention-report-{timestamp.strftime('%H%M%S')}.json"

        # Aggregate violations
        all_violations = []
        for stats in all_stats.values():
            all_violations.extend(stats.violations)

        report = {
            'timestamp': timestamp.isoformat(),
            'report_type': 'retention_compliance',
            'summary': {
                'total_evaluated': sum(s.objects_evaluated for s in all_stats.values()),
                'total_compliant': sum(s.objects_compliant for s in all_stats.values()),
                'total_non_compliant': sum(s.objects_non_compliant for s in all_stats.values()),
                'total_legal_holds': sum(s.objects_with_legal_hold for s in all_stats.values()),
                'total_violations': len(all_violations)
            },
            'policies': {name: s.to_dict() for name, s in all_stats.items()},
            'violations': all_violations[:1000],  # Limit to 1000 violations
            'compliance_frameworks': list(set(
                p.compliance_framework for p in self.policies
                if p.compliance_framework
            ))
        }

        if self.dry_run:
            logger.info(f"[DRY RUN] Would create report at s3://{bucket}/{report_key}")
            return

        try:
            self.s3_client.put_object(
                Bucket=bucket,
                Key=report_key,
                Body=json.dumps(report, indent=2),
                ContentType='application/json'
            )
            logger.info(f"Created compliance report at s3://{bucket}/{report_key}")
        except Exception as e:
            logger.error(f"Failed to create compliance report: {e}")

    def run(self) -> Dict[str, RetentionStats]:
        """Run retention enforcement for all policies."""
        start_time = time.time()
        all_stats: Dict[str, RetentionStats] = {}

        logger.info("=" * 60)
        logger.info("Starting retention enforcement")
        logger.info(f"Dry run: {self.dry_run}")
        logger.info(f"Policies: {len(self.policies)}")
        logger.info("=" * 60)

        for policy in self.policies:
            try:
                stats = self._enforce_policy(policy)
                all_stats[policy.name] = stats
            except Exception as e:
                logger.error(f"Failed to enforce policy {policy.name}: {e}")
                all_stats[policy.name] = RetentionStats()

        # Generate compliance report
        if self.generate_report:
            self._generate_compliance_report(all_stats)

        total_duration = time.time() - start_time
        total_violations = sum(len(s.violations) for s in all_stats.values())

        logger.info("=" * 60)
        logger.info("Retention Enforcement Summary")
        logger.info("=" * 60)
        logger.info(f"Total duration: {total_duration:.2f} seconds")
        logger.info(f"Total evaluated: {sum(s.objects_evaluated for s in all_stats.values())}")
        logger.info(f"Total violations: {total_violations}")
        logger.info("=" * 60)

        return all_stats


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Retention enforcement for GreenLang storage lifecycle'
    )
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode')
    parser.add_argument('--generate-compliance-report', action='store_true', default=True)
    parser.add_argument('--metrics-port', type=int, default=9090)
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO')

    args = parser.parse_args()
    logger.setLevel(getattr(logging, args.log_level))

    dry_run = args.dry_run or os.environ.get('DRY_RUN', '').lower() == 'true'

    try:
        config = load_config(args.config)
        if not dry_run:
            dry_run = config.get('global', {}).get('dry_run', False)

        enforcer = RetentionEnforcer(config, dry_run=dry_run,
                                     generate_report=args.generate_compliance_report)
        stats = enforcer.run()

        total_violations = sum(len(s.violations) for s in stats.values())
        sys.exit(1 if total_violations > 0 else 0)

    except Exception as e:
        logger.error(f"Retention enforcement failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
