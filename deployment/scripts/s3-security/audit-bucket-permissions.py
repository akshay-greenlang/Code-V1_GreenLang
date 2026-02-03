#!/usr/bin/env python3
"""
GreenLang S3 Security Audit Script - Bucket Permissions Auditor

This script performs comprehensive security audits on S3 buckets including:
- Public access settings
- Encryption configuration
- Bucket policies
- ACL settings
- Versioning status
- Logging configuration
- Object lock settings

Author: GreenLang DevOps Team
Version: 1.0.0
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class S3SecurityAuditor:
    """Comprehensive S3 security auditor for GreenLang buckets."""

    def __init__(
        self,
        region: str = None,
        profile: str = None,
        bucket_prefix: str = "greenlang"
    ):
        """Initialize the S3 security auditor.

        Args:
            region: AWS region to audit
            profile: AWS CLI profile to use
            bucket_prefix: Only audit buckets with this prefix
        """
        session_kwargs = {}
        if profile:
            session_kwargs['profile_name'] = profile
        if region:
            session_kwargs['region_name'] = region

        self.session = boto3.Session(**session_kwargs)
        self.s3_client = self.session.client('s3')
        self.s3_control = self.session.client('s3control')
        self.sts_client = self.session.client('sts')
        self.bucket_prefix = bucket_prefix

        # Get account ID
        self.account_id = self.sts_client.get_caller_identity()['Account']

        # Compliance status tracking
        self.compliance_results = {
            'audit_timestamp': datetime.utcnow().isoformat(),
            'account_id': self.account_id,
            'buckets': [],
            'summary': {
                'total_buckets': 0,
                'compliant_buckets': 0,
                'non_compliant_buckets': 0,
                'critical_issues': 0,
                'high_issues': 0,
                'medium_issues': 0,
                'low_issues': 0
            }
        }

    def list_buckets(self) -> List[str]:
        """List all S3 buckets, optionally filtered by prefix.

        Returns:
            List of bucket names
        """
        try:
            response = self.s3_client.list_buckets()
            buckets = [
                b['Name'] for b in response['Buckets']
                if self.bucket_prefix is None or b['Name'].startswith(self.bucket_prefix)
            ]
            logger.info(f"Found {len(buckets)} buckets to audit")
            return buckets
        except ClientError as e:
            logger.error(f"Failed to list buckets: {e}")
            raise

    def check_public_access_block(self, bucket_name: str) -> Dict[str, Any]:
        """Check public access block settings for a bucket.

        Args:
            bucket_name: Name of the S3 bucket

        Returns:
            Dictionary with public access block settings and compliance status
        """
        result = {
            'check_name': 'public_access_block',
            'status': 'UNKNOWN',
            'severity': 'CRITICAL',
            'details': {}
        }

        try:
            response = self.s3_client.get_public_access_block(Bucket=bucket_name)
            config = response['PublicAccessBlockConfiguration']

            result['details'] = {
                'block_public_acls': config.get('BlockPublicAcls', False),
                'ignore_public_acls': config.get('IgnorePublicAcls', False),
                'block_public_policy': config.get('BlockPublicPolicy', False),
                'restrict_public_buckets': config.get('RestrictPublicBuckets', False)
            }

            # All settings must be True for compliance
            all_blocked = all(result['details'].values())
            result['status'] = 'COMPLIANT' if all_blocked else 'NON_COMPLIANT'

            if not all_blocked:
                result['recommendation'] = (
                    "Enable all public access block settings to prevent "
                    "accidental public exposure of bucket contents."
                )

        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchPublicAccessBlockConfiguration':
                result['status'] = 'NON_COMPLIANT'
                result['details'] = {
                    'block_public_acls': False,
                    'ignore_public_acls': False,
                    'block_public_policy': False,
                    'restrict_public_buckets': False
                }
                result['recommendation'] = (
                    "No public access block configuration found. "
                    "Configure all public access block settings immediately."
                )
            else:
                result['status'] = 'ERROR'
                result['error'] = str(e)

        return result

    def check_encryption(self, bucket_name: str) -> Dict[str, Any]:
        """Check encryption configuration for a bucket.

        Args:
            bucket_name: Name of the S3 bucket

        Returns:
            Dictionary with encryption settings and compliance status
        """
        result = {
            'check_name': 'encryption',
            'status': 'UNKNOWN',
            'severity': 'HIGH',
            'details': {}
        }

        try:
            response = self.s3_client.get_bucket_encryption(Bucket=bucket_name)
            rules = response['ServerSideEncryptionConfiguration']['Rules']

            for rule in rules:
                default_encryption = rule.get('ApplyServerSideEncryptionByDefault', {})
                result['details'] = {
                    'sse_algorithm': default_encryption.get('SSEAlgorithm', 'NONE'),
                    'kms_key_id': default_encryption.get('KMSMasterKeyID', None),
                    'bucket_key_enabled': rule.get('BucketKeyEnabled', False)
                }

            # Check for KMS encryption (preferred) or at least AES256
            if result['details'].get('sse_algorithm') == 'aws:kms':
                result['status'] = 'COMPLIANT'
            elif result['details'].get('sse_algorithm') == 'AES256':
                result['status'] = 'COMPLIANT'
                result['recommendation'] = (
                    "Consider upgrading to KMS encryption for better "
                    "key management and audit capabilities."
                )
            else:
                result['status'] = 'NON_COMPLIANT'
                result['recommendation'] = (
                    "Enable server-side encryption with KMS or AES256."
                )

        except ClientError as e:
            if e.response['Error']['Code'] == 'ServerSideEncryptionConfigurationNotFoundError':
                result['status'] = 'NON_COMPLIANT'
                result['details'] = {'sse_algorithm': 'NONE'}
                result['recommendation'] = (
                    "No encryption configuration found. "
                    "Enable KMS encryption immediately."
                )
            else:
                result['status'] = 'ERROR'
                result['error'] = str(e)

        return result

    def check_versioning(self, bucket_name: str) -> Dict[str, Any]:
        """Check versioning configuration for a bucket.

        Args:
            bucket_name: Name of the S3 bucket

        Returns:
            Dictionary with versioning settings and compliance status
        """
        result = {
            'check_name': 'versioning',
            'status': 'UNKNOWN',
            'severity': 'MEDIUM',
            'details': {}
        }

        try:
            response = self.s3_client.get_bucket_versioning(Bucket=bucket_name)
            result['details'] = {
                'status': response.get('Status', 'Disabled'),
                'mfa_delete': response.get('MFADelete', 'Disabled')
            }

            if result['details']['status'] == 'Enabled':
                result['status'] = 'COMPLIANT'
                if result['details']['mfa_delete'] != 'Enabled':
                    result['recommendation'] = (
                        "Consider enabling MFA Delete for additional "
                        "protection against accidental deletions."
                    )
            else:
                result['status'] = 'NON_COMPLIANT'
                result['recommendation'] = (
                    "Enable versioning to protect against accidental "
                    "deletions and enable data recovery."
                )

        except ClientError as e:
            result['status'] = 'ERROR'
            result['error'] = str(e)

        return result

    def check_logging(self, bucket_name: str) -> Dict[str, Any]:
        """Check access logging configuration for a bucket.

        Args:
            bucket_name: Name of the S3 bucket

        Returns:
            Dictionary with logging settings and compliance status
        """
        result = {
            'check_name': 'logging',
            'status': 'UNKNOWN',
            'severity': 'MEDIUM',
            'details': {}
        }

        try:
            response = self.s3_client.get_bucket_logging(Bucket=bucket_name)
            logging_enabled = response.get('LoggingEnabled')

            if logging_enabled:
                result['details'] = {
                    'enabled': True,
                    'target_bucket': logging_enabled.get('TargetBucket'),
                    'target_prefix': logging_enabled.get('TargetPrefix', '')
                }
                result['status'] = 'COMPLIANT'
            else:
                result['details'] = {'enabled': False}
                result['status'] = 'NON_COMPLIANT'
                result['recommendation'] = (
                    "Enable access logging to track all requests "
                    "made to the bucket for security auditing."
                )

        except ClientError as e:
            result['status'] = 'ERROR'
            result['error'] = str(e)

        return result

    def check_bucket_policy(self, bucket_name: str) -> Dict[str, Any]:
        """Check bucket policy for security issues.

        Args:
            bucket_name: Name of the S3 bucket

        Returns:
            Dictionary with policy analysis and compliance status
        """
        result = {
            'check_name': 'bucket_policy',
            'status': 'UNKNOWN',
            'severity': 'HIGH',
            'details': {}
        }

        try:
            response = self.s3_client.get_bucket_policy(Bucket=bucket_name)
            policy = json.loads(response['Policy'])

            issues = []
            has_ssl_enforcement = False
            has_encryption_enforcement = False

            for statement in policy.get('Statement', []):
                # Check for wildcard principals
                principal = statement.get('Principal', {})
                if principal == '*' or principal == {'AWS': '*'}:
                    effect = statement.get('Effect', 'Allow')
                    if effect == 'Allow':
                        issues.append({
                            'issue': 'wildcard_principal_allow',
                            'severity': 'CRITICAL',
                            'statement_sid': statement.get('Sid', 'Unknown')
                        })

                # Check for SSL enforcement
                condition = statement.get('Condition', {})
                if condition.get('Bool', {}).get('aws:SecureTransport') == 'false':
                    if statement.get('Effect') == 'Deny':
                        has_ssl_enforcement = True

                # Check for encryption enforcement
                if condition.get('StringNotEquals', {}).get('s3:x-amz-server-side-encryption'):
                    if statement.get('Effect') == 'Deny':
                        has_encryption_enforcement = True

            result['details'] = {
                'has_policy': True,
                'has_ssl_enforcement': has_ssl_enforcement,
                'has_encryption_enforcement': has_encryption_enforcement,
                'issues': issues
            }

            if issues:
                result['status'] = 'NON_COMPLIANT'
                result['recommendation'] = (
                    "Review and fix bucket policy issues. "
                    "Remove wildcard principals and add SSL/encryption enforcement."
                )
            elif not has_ssl_enforcement or not has_encryption_enforcement:
                result['status'] = 'NON_COMPLIANT'
                result['recommendation'] = (
                    "Add policy statements to enforce SSL-only access "
                    "and require server-side encryption on uploads."
                )
            else:
                result['status'] = 'COMPLIANT'

        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchBucketPolicy':
                result['details'] = {'has_policy': False}
                result['status'] = 'NON_COMPLIANT'
                result['recommendation'] = (
                    "No bucket policy found. Add a policy that enforces "
                    "SSL-only access and encryption requirements."
                )
            else:
                result['status'] = 'ERROR'
                result['error'] = str(e)

        return result

    def check_acl(self, bucket_name: str) -> Dict[str, Any]:
        """Check bucket ACL for public access.

        Args:
            bucket_name: Name of the S3 bucket

        Returns:
            Dictionary with ACL analysis and compliance status
        """
        result = {
            'check_name': 'acl',
            'status': 'UNKNOWN',
            'severity': 'CRITICAL',
            'details': {}
        }

        try:
            response = self.s3_client.get_bucket_acl(Bucket=bucket_name)

            public_grants = []
            owner = response.get('Owner', {}).get('ID', 'Unknown')

            for grant in response.get('Grants', []):
                grantee = grant.get('Grantee', {})
                grantee_type = grantee.get('Type', '')
                uri = grantee.get('URI', '')

                # Check for public access via ACL
                if grantee_type == 'Group':
                    if 'AllUsers' in uri or 'AuthenticatedUsers' in uri:
                        public_grants.append({
                            'grantee': uri,
                            'permission': grant.get('Permission')
                        })

            result['details'] = {
                'owner': owner,
                'public_grants': public_grants,
                'total_grants': len(response.get('Grants', []))
            }

            if public_grants:
                result['status'] = 'NON_COMPLIANT'
                result['recommendation'] = (
                    "Remove public ACL grants immediately. "
                    "Use bucket policies for access control instead."
                )
            else:
                result['status'] = 'COMPLIANT'

        except ClientError as e:
            result['status'] = 'ERROR'
            result['error'] = str(e)

        return result

    def check_object_lock(self, bucket_name: str) -> Dict[str, Any]:
        """Check object lock configuration for a bucket.

        Args:
            bucket_name: Name of the S3 bucket

        Returns:
            Dictionary with object lock settings and compliance status
        """
        result = {
            'check_name': 'object_lock',
            'status': 'UNKNOWN',
            'severity': 'LOW',
            'details': {}
        }

        try:
            response = self.s3_client.get_object_lock_configuration(Bucket=bucket_name)
            config = response.get('ObjectLockConfiguration', {})

            result['details'] = {
                'enabled': config.get('ObjectLockEnabled') == 'Enabled',
                'rule': config.get('Rule', {})
            }

            if result['details']['enabled']:
                result['status'] = 'COMPLIANT'
                rule = result['details']['rule']
                if rule:
                    retention = rule.get('DefaultRetention', {})
                    result['details']['retention_mode'] = retention.get('Mode')
                    result['details']['retention_days'] = retention.get('Days')
                    result['details']['retention_years'] = retention.get('Years')
            else:
                result['status'] = 'INFO'
                result['recommendation'] = (
                    "Consider enabling Object Lock for critical data "
                    "that requires WORM protection."
                )

        except ClientError as e:
            if e.response['Error']['Code'] == 'ObjectLockConfigurationNotFoundError':
                result['details'] = {'enabled': False}
                result['status'] = 'INFO'
                result['recommendation'] = (
                    "Object Lock is not enabled. Consider enabling for "
                    "compliance-critical data."
                )
            else:
                result['status'] = 'ERROR'
                result['error'] = str(e)

        return result

    def audit_bucket(self, bucket_name: str) -> Dict[str, Any]:
        """Perform comprehensive security audit on a single bucket.

        Args:
            bucket_name: Name of the S3 bucket to audit

        Returns:
            Dictionary with all audit results for the bucket
        """
        logger.info(f"Auditing bucket: {bucket_name}")

        bucket_result = {
            'bucket_name': bucket_name,
            'audit_timestamp': datetime.utcnow().isoformat(),
            'checks': [],
            'overall_status': 'COMPLIANT',
            'issue_counts': {
                'critical': 0,
                'high': 0,
                'medium': 0,
                'low': 0
            }
        }

        # Run all security checks
        checks = [
            self.check_public_access_block(bucket_name),
            self.check_encryption(bucket_name),
            self.check_versioning(bucket_name),
            self.check_logging(bucket_name),
            self.check_bucket_policy(bucket_name),
            self.check_acl(bucket_name),
            self.check_object_lock(bucket_name)
        ]

        for check in checks:
            bucket_result['checks'].append(check)

            # Update issue counts
            if check['status'] == 'NON_COMPLIANT':
                severity = check.get('severity', 'MEDIUM').lower()
                bucket_result['issue_counts'][severity] = \
                    bucket_result['issue_counts'].get(severity, 0) + 1
                bucket_result['overall_status'] = 'NON_COMPLIANT'

        return bucket_result

    def audit_all_buckets(self) -> Dict[str, Any]:
        """Audit all buckets matching the configured prefix.

        Returns:
            Dictionary with all audit results and summary
        """
        buckets = self.list_buckets()
        self.compliance_results['summary']['total_buckets'] = len(buckets)

        for bucket_name in buckets:
            try:
                bucket_result = self.audit_bucket(bucket_name)
                self.compliance_results['buckets'].append(bucket_result)

                # Update summary
                if bucket_result['overall_status'] == 'COMPLIANT':
                    self.compliance_results['summary']['compliant_buckets'] += 1
                else:
                    self.compliance_results['summary']['non_compliant_buckets'] += 1

                # Aggregate issue counts
                for severity, count in bucket_result['issue_counts'].items():
                    key = f"{severity}_issues"
                    self.compliance_results['summary'][key] = \
                        self.compliance_results['summary'].get(key, 0) + count

            except Exception as e:
                logger.error(f"Failed to audit bucket {bucket_name}: {e}")
                self.compliance_results['buckets'].append({
                    'bucket_name': bucket_name,
                    'error': str(e),
                    'overall_status': 'ERROR'
                })

        return self.compliance_results

    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate JSON report of audit results.

        Args:
            output_file: Optional file path to write report

        Returns:
            JSON string of the report
        """
        report = json.dumps(self.compliance_results, indent=2, default=str)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            logger.info(f"Report written to {output_file}")

        return report

    def print_summary(self):
        """Print a human-readable summary of the audit results."""
        summary = self.compliance_results['summary']

        print("\n" + "=" * 60)
        print("S3 SECURITY AUDIT SUMMARY")
        print("=" * 60)
        print(f"Account ID: {self.compliance_results['account_id']}")
        print(f"Audit Time: {self.compliance_results['audit_timestamp']}")
        print("-" * 60)
        print(f"Total Buckets Audited: {summary['total_buckets']}")
        print(f"Compliant Buckets:     {summary['compliant_buckets']}")
        print(f"Non-Compliant Buckets: {summary['non_compliant_buckets']}")
        print("-" * 60)
        print("Issue Breakdown:")
        print(f"  Critical: {summary.get('critical_issues', 0)}")
        print(f"  High:     {summary.get('high_issues', 0)}")
        print(f"  Medium:   {summary.get('medium_issues', 0)}")
        print(f"  Low:      {summary.get('low_issues', 0)}")
        print("=" * 60)

        # Print non-compliant buckets
        non_compliant = [
            b for b in self.compliance_results['buckets']
            if b.get('overall_status') == 'NON_COMPLIANT'
        ]

        if non_compliant:
            print("\nNON-COMPLIANT BUCKETS:")
            print("-" * 60)
            for bucket in non_compliant:
                print(f"\n  Bucket: {bucket['bucket_name']}")
                for check in bucket['checks']:
                    if check['status'] == 'NON_COMPLIANT':
                        print(f"    - {check['check_name']}: {check.get('recommendation', 'N/A')}")


def main():
    """Main entry point for the S3 security auditor."""
    parser = argparse.ArgumentParser(
        description='GreenLang S3 Security Auditor'
    )
    parser.add_argument(
        '--region',
        help='AWS region to audit'
    )
    parser.add_argument(
        '--profile',
        help='AWS CLI profile to use'
    )
    parser.add_argument(
        '--bucket-prefix',
        default='greenlang',
        help='Only audit buckets with this prefix (default: greenlang)'
    )
    parser.add_argument(
        '--output',
        help='Output file for JSON report'
    )
    parser.add_argument(
        '--all-buckets',
        action='store_true',
        help='Audit all buckets regardless of prefix'
    )
    parser.add_argument(
        '--json-only',
        action='store_true',
        help='Output JSON only (no summary)'
    )

    args = parser.parse_args()

    bucket_prefix = None if args.all_buckets else args.bucket_prefix

    auditor = S3SecurityAuditor(
        region=args.region,
        profile=args.profile,
        bucket_prefix=bucket_prefix
    )

    try:
        auditor.audit_all_buckets()

        if args.json_only:
            print(auditor.generate_report(args.output))
        else:
            auditor.print_summary()
            if args.output:
                auditor.generate_report(args.output)
                print(f"\nFull report written to: {args.output}")

        # Exit with non-zero if there are critical issues
        if auditor.compliance_results['summary'].get('critical_issues', 0) > 0:
            sys.exit(1)

    except Exception as e:
        logger.error(f"Audit failed: {e}")
        sys.exit(2)


if __name__ == '__main__':
    main()
