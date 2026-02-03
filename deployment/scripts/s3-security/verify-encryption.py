#!/usr/bin/env python3
"""
GreenLang S3 Security - Encryption Verification Script

This script verifies KMS encryption on all objects in S3 buckets:
- Scans all objects for encryption status
- Reports unencrypted objects
- Optional remediation mode to encrypt unencrypted objects
- Generates detailed compliance reports

Author: GreenLang DevOps Team
Version: 1.0.0
"""

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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


class S3EncryptionVerifier:
    """Verifies and optionally remediates S3 object encryption."""

    def __init__(
        self,
        region: str = None,
        profile: str = None,
        kms_key_id: str = None,
        max_workers: int = 10
    ):
        """Initialize the encryption verifier.

        Args:
            region: AWS region
            profile: AWS CLI profile to use
            kms_key_id: KMS key ID for remediation
            max_workers: Maximum number of parallel workers
        """
        session_kwargs = {}
        if profile:
            session_kwargs['profile_name'] = profile
        if region:
            session_kwargs['region_name'] = region

        self.session = boto3.Session(**session_kwargs)
        self.s3_client = self.session.client('s3')
        self.kms_key_id = kms_key_id
        self.max_workers = max_workers

        # Results tracking
        self.results = {
            'scan_timestamp': datetime.utcnow().isoformat(),
            'buckets': [],
            'summary': {
                'total_objects': 0,
                'encrypted_objects': 0,
                'unencrypted_objects': 0,
                'kms_encrypted': 0,
                'aes256_encrypted': 0,
                'remediated_objects': 0,
                'remediation_errors': 0
            }
        }

    def get_object_encryption(self, bucket: str, key: str) -> Dict[str, Any]:
        """Get encryption information for a single object.

        Args:
            bucket: Bucket name
            key: Object key

        Returns:
            Dictionary with encryption details
        """
        try:
            response = self.s3_client.head_object(Bucket=bucket, Key=key)

            encryption = response.get('ServerSideEncryption')
            kms_key = response.get('SSEKMSKeyId')

            return {
                'key': key,
                'size': response.get('ContentLength', 0),
                'last_modified': response.get('LastModified'),
                'encryption': encryption,
                'kms_key_id': kms_key,
                'is_encrypted': encryption is not None,
                'is_kms_encrypted': encryption == 'aws:kms'
            }
        except ClientError as e:
            logger.warning(f"Failed to get encryption info for {key}: {e}")
            return {
                'key': key,
                'error': str(e),
                'is_encrypted': None
            }

    def scan_bucket(
        self,
        bucket_name: str,
        prefix: str = '',
        max_objects: int = None
    ) -> Dict[str, Any]:
        """Scan all objects in a bucket for encryption status.

        Args:
            bucket_name: Name of the bucket to scan
            prefix: Optional prefix to filter objects
            max_objects: Maximum number of objects to scan

        Returns:
            Dictionary with scan results
        """
        logger.info(f"Scanning bucket: {bucket_name}")

        bucket_result = {
            'bucket_name': bucket_name,
            'prefix': prefix,
            'scan_timestamp': datetime.utcnow().isoformat(),
            'objects': {
                'encrypted': [],
                'unencrypted': [],
                'errors': []
            },
            'counts': {
                'total': 0,
                'encrypted': 0,
                'unencrypted': 0,
                'kms_encrypted': 0,
                'aes256_encrypted': 0,
                'errors': 0
            }
        }

        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=bucket_name,
                Prefix=prefix
            )

            objects_to_check = []
            for page in page_iterator:
                for obj in page.get('Contents', []):
                    objects_to_check.append(obj['Key'])

                    if max_objects and len(objects_to_check) >= max_objects:
                        break

                if max_objects and len(objects_to_check) >= max_objects:
                    break

            logger.info(f"Found {len(objects_to_check)} objects to verify")

            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self.get_object_encryption, bucket_name, key
                    ): key for key in objects_to_check
                }

                for future in as_completed(futures):
                    result = future.result()
                    bucket_result['counts']['total'] += 1

                    if 'error' in result:
                        bucket_result['objects']['errors'].append(result)
                        bucket_result['counts']['errors'] += 1
                    elif result['is_encrypted']:
                        bucket_result['objects']['encrypted'].append(result)
                        bucket_result['counts']['encrypted'] += 1
                        if result['is_kms_encrypted']:
                            bucket_result['counts']['kms_encrypted'] += 1
                        else:
                            bucket_result['counts']['aes256_encrypted'] += 1
                    else:
                        bucket_result['objects']['unencrypted'].append(result)
                        bucket_result['counts']['unencrypted'] += 1

        except ClientError as e:
            logger.error(f"Failed to scan bucket {bucket_name}: {e}")
            bucket_result['error'] = str(e)

        return bucket_result

    def remediate_object(self, bucket: str, key: str) -> Dict[str, Any]:
        """Encrypt an unencrypted object using copy-in-place.

        Args:
            bucket: Bucket name
            key: Object key

        Returns:
            Dictionary with remediation result
        """
        try:
            copy_source = {'Bucket': bucket, 'Key': key}

            if self.kms_key_id:
                self.s3_client.copy_object(
                    Bucket=bucket,
                    Key=key,
                    CopySource=copy_source,
                    ServerSideEncryption='aws:kms',
                    SSEKMSKeyId=self.kms_key_id,
                    MetadataDirective='COPY'
                )
            else:
                self.s3_client.copy_object(
                    Bucket=bucket,
                    Key=key,
                    CopySource=copy_source,
                    ServerSideEncryption='AES256',
                    MetadataDirective='COPY'
                )

            logger.info(f"Encrypted object: {key}")
            return {
                'key': key,
                'status': 'SUCCESS',
                'encryption': 'aws:kms' if self.kms_key_id else 'AES256'
            }

        except ClientError as e:
            logger.error(f"Failed to encrypt {key}: {e}")
            return {
                'key': key,
                'status': 'FAILED',
                'error': str(e)
            }

    def remediate_bucket(
        self,
        bucket_name: str,
        unencrypted_objects: List[Dict],
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """Remediate all unencrypted objects in a bucket.

        Args:
            bucket_name: Name of the bucket
            unencrypted_objects: List of unencrypted object info
            dry_run: If True, only report what would be done

        Returns:
            Dictionary with remediation results
        """
        remediation_result = {
            'bucket_name': bucket_name,
            'dry_run': dry_run,
            'objects_to_remediate': len(unencrypted_objects),
            'remediated': [],
            'failed': []
        }

        if dry_run:
            logger.info(
                f"DRY RUN: Would encrypt {len(unencrypted_objects)} "
                f"objects in {bucket_name}"
            )
            for obj in unencrypted_objects:
                remediation_result['remediated'].append({
                    'key': obj['key'],
                    'status': 'DRY_RUN'
                })
            return remediation_result

        logger.info(
            f"Remediating {len(unencrypted_objects)} objects in {bucket_name}"
        )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self.remediate_object, bucket_name, obj['key']
                ): obj['key'] for obj in unencrypted_objects
            }

            for future in as_completed(futures):
                result = future.result()
                if result['status'] == 'SUCCESS':
                    remediation_result['remediated'].append(result)
                    self.results['summary']['remediated_objects'] += 1
                else:
                    remediation_result['failed'].append(result)
                    self.results['summary']['remediation_errors'] += 1

        return remediation_result

    def verify_bucket(
        self,
        bucket_name: str,
        prefix: str = '',
        max_objects: int = None,
        remediate: bool = False,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """Verify and optionally remediate encryption for a bucket.

        Args:
            bucket_name: Name of the bucket
            prefix: Optional prefix filter
            max_objects: Maximum objects to scan
            remediate: Whether to remediate unencrypted objects
            dry_run: If True with remediate, only report what would be done

        Returns:
            Dictionary with verification and remediation results
        """
        # Scan the bucket
        scan_result = self.scan_bucket(bucket_name, prefix, max_objects)

        # Update summary
        self.results['summary']['total_objects'] += scan_result['counts']['total']
        self.results['summary']['encrypted_objects'] += scan_result['counts']['encrypted']
        self.results['summary']['unencrypted_objects'] += scan_result['counts']['unencrypted']
        self.results['summary']['kms_encrypted'] += scan_result['counts']['kms_encrypted']
        self.results['summary']['aes256_encrypted'] += scan_result['counts']['aes256_encrypted']

        # Remediate if requested and there are unencrypted objects
        if remediate and scan_result['objects']['unencrypted']:
            scan_result['remediation'] = self.remediate_bucket(
                bucket_name,
                scan_result['objects']['unencrypted'],
                dry_run
            )

        self.results['buckets'].append(scan_result)
        return scan_result

    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate JSON report of verification results.

        Args:
            output_file: Optional file path to write report

        Returns:
            JSON string of the report
        """
        report = json.dumps(self.results, indent=2, default=str)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            logger.info(f"Report written to {output_file}")

        return report

    def print_summary(self):
        """Print a human-readable summary of the verification results."""
        summary = self.results['summary']

        print("\n" + "=" * 60)
        print("S3 ENCRYPTION VERIFICATION SUMMARY")
        print("=" * 60)
        print(f"Scan Time: {self.results['scan_timestamp']}")
        print("-" * 60)
        print(f"Total Objects Scanned:    {summary['total_objects']}")
        print(f"Encrypted Objects:        {summary['encrypted_objects']}")
        print(f"  - KMS Encrypted:        {summary['kms_encrypted']}")
        print(f"  - AES256 Encrypted:     {summary['aes256_encrypted']}")
        print(f"Unencrypted Objects:      {summary['unencrypted_objects']}")
        print("-" * 60)

        if summary['remediated_objects'] > 0 or summary['remediation_errors'] > 0:
            print("Remediation Results:")
            print(f"  Successfully Encrypted: {summary['remediated_objects']}")
            print(f"  Failed:                 {summary['remediation_errors']}")
            print("-" * 60)

        # Compliance percentage
        if summary['total_objects'] > 0:
            compliance_pct = (
                summary['encrypted_objects'] / summary['total_objects'] * 100
            )
            print(f"Encryption Compliance:    {compliance_pct:.2f}%")

        print("=" * 60)

        # Print buckets with unencrypted objects
        buckets_with_issues = [
            b for b in self.results['buckets']
            if b['counts']['unencrypted'] > 0
        ]

        if buckets_with_issues:
            print("\nBUCKETS WITH UNENCRYPTED OBJECTS:")
            print("-" * 60)
            for bucket in buckets_with_issues:
                print(f"\n  Bucket: {bucket['bucket_name']}")
                print(f"  Unencrypted: {bucket['counts']['unencrypted']}")
                print("  Sample objects:")
                for obj in bucket['objects']['unencrypted'][:5]:
                    print(f"    - {obj['key']}")
                if bucket['counts']['unencrypted'] > 5:
                    print(f"    ... and {bucket['counts']['unencrypted'] - 5} more")


def main():
    """Main entry point for the encryption verifier."""
    parser = argparse.ArgumentParser(
        description='GreenLang S3 Encryption Verifier'
    )
    parser.add_argument(
        '--bucket',
        required=True,
        help='Bucket name to verify (or comma-separated list)'
    )
    parser.add_argument(
        '--prefix',
        default='',
        help='Object prefix filter'
    )
    parser.add_argument(
        '--region',
        help='AWS region'
    )
    parser.add_argument(
        '--profile',
        help='AWS CLI profile to use'
    )
    parser.add_argument(
        '--max-objects',
        type=int,
        help='Maximum number of objects to scan'
    )
    parser.add_argument(
        '--remediate',
        action='store_true',
        help='Encrypt unencrypted objects'
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Actually perform remediation (default is dry-run)'
    )
    parser.add_argument(
        '--kms-key-id',
        help='KMS key ID to use for encryption (default: AES256)'
    )
    parser.add_argument(
        '--output',
        help='Output file for JSON report'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=10,
        help='Number of parallel workers (default: 10)'
    )
    parser.add_argument(
        '--json-only',
        action='store_true',
        help='Output JSON only (no summary)'
    )

    args = parser.parse_args()

    verifier = S3EncryptionVerifier(
        region=args.region,
        profile=args.profile,
        kms_key_id=args.kms_key_id,
        max_workers=args.workers
    )

    buckets = [b.strip() for b in args.bucket.split(',')]
    dry_run = not args.execute

    try:
        for bucket in buckets:
            verifier.verify_bucket(
                bucket_name=bucket,
                prefix=args.prefix,
                max_objects=args.max_objects,
                remediate=args.remediate,
                dry_run=dry_run
            )

        if args.json_only:
            print(verifier.generate_report(args.output))
        else:
            verifier.print_summary()
            if args.output:
                verifier.generate_report(args.output)
                print(f"\nFull report written to: {args.output}")

        # Exit with non-zero if there are unencrypted objects
        if verifier.results['summary']['unencrypted_objects'] > 0:
            if not args.remediate:
                sys.exit(1)
            elif verifier.results['summary']['remediation_errors'] > 0:
                sys.exit(1)

    except Exception as e:
        logger.error(f"Verification failed: {e}")
        sys.exit(2)


if __name__ == '__main__':
    main()
