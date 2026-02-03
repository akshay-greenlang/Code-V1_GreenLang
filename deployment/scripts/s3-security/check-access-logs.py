#!/usr/bin/env python3
"""
GreenLang S3 Security - Access Log Analyzer

This script analyzes S3 access logs for security anomalies:
- Detects unauthorized access attempts
- Identifies unusual access patterns
- Monitors for suspicious IP addresses
- Generates security alerts

Author: GreenLang DevOps Team
Version: 1.0.0
"""

import argparse
import gzip
import json
import logging
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from io import BytesIO
from typing import Any, Dict, List, Optional, Set

import boto3
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# S3 Access Log Format Pattern
ACCESS_LOG_PATTERN = re.compile(
    r'(?P<bucket_owner>\S+) '
    r'(?P<bucket>\S+) '
    r'\[(?P<time>[^\]]+)\] '
    r'(?P<remote_ip>\S+) '
    r'(?P<requester>\S+) '
    r'(?P<request_id>\S+) '
    r'(?P<operation>\S+) '
    r'(?P<key>\S+) '
    r'"(?P<request_uri>[^"]*)" '
    r'(?P<http_status>\S+) '
    r'(?P<error_code>\S+) '
    r'(?P<bytes_sent>\S+) '
    r'(?P<object_size>\S+) '
    r'(?P<total_time>\S+) '
    r'(?P<turn_around_time>\S+) '
    r'"(?P<referrer>[^"]*)" '
    r'"(?P<user_agent>[^"]*)"'
)


class S3AccessLogAnalyzer:
    """Analyzes S3 access logs for security anomalies."""

    # Suspicious HTTP status codes
    SUSPICIOUS_STATUS_CODES = ['403', '401', '404']

    # Suspicious operations
    SUSPICIOUS_OPERATIONS = [
        'REST.DELETE.BUCKET',
        'REST.PUT.BUCKETPOLICY',
        'REST.DELETE.BUCKETPOLICY',
        'REST.PUT.ACL',
        'REST.PUT.BUCKETVERSIONING'
    ]

    def __init__(
        self,
        region: str = None,
        profile: str = None,
        alert_threshold: int = 10,
        known_ips: Set[str] = None
    ):
        """Initialize the access log analyzer.

        Args:
            region: AWS region
            profile: AWS CLI profile to use
            alert_threshold: Number of suspicious events to trigger alert
            known_ips: Set of known/allowed IP addresses
        """
        session_kwargs = {}
        if profile:
            session_kwargs['profile_name'] = profile
        if region:
            session_kwargs['region_name'] = region

        self.session = boto3.Session(**session_kwargs)
        self.s3_client = self.session.client('s3')
        self.sns_client = self.session.client('sns')

        self.alert_threshold = alert_threshold
        self.known_ips = known_ips or set()

        # Analysis results
        self.results = {
            'analysis_timestamp': datetime.utcnow().isoformat(),
            'logs_analyzed': 0,
            'events_processed': 0,
            'alerts': [],
            'statistics': {
                'by_operation': defaultdict(int),
                'by_status_code': defaultdict(int),
                'by_requester': defaultdict(int),
                'by_ip': defaultdict(int),
                'by_bucket': defaultdict(int)
            },
            'suspicious_activity': {
                'unauthorized_attempts': [],
                'unknown_ips': [],
                'sensitive_operations': [],
                'high_frequency_access': []
            }
        }

    def parse_log_entry(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a single S3 access log entry.

        Args:
            line: Raw log line

        Returns:
            Dictionary with parsed fields or None if parsing fails
        """
        match = ACCESS_LOG_PATTERN.match(line)
        if not match:
            return None

        entry = match.groupdict()

        # Parse timestamp
        try:
            entry['timestamp'] = datetime.strptime(
                entry['time'],
                '%d/%b/%Y:%H:%M:%S %z'
            )
        except ValueError:
            entry['timestamp'] = None

        # Convert numeric fields
        for field in ['http_status', 'bytes_sent', 'object_size', 'total_time']:
            if entry.get(field) and entry[field] != '-':
                try:
                    entry[field] = int(entry[field])
                except ValueError:
                    pass

        return entry

    def download_log_file(self, bucket: str, key: str) -> List[str]:
        """Download and decompress a log file.

        Args:
            bucket: Log bucket name
            key: Log file key

        Returns:
            List of log lines
        """
        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            content = response['Body'].read()

            # Decompress if gzipped
            if key.endswith('.gz'):
                content = gzip.decompress(content)

            return content.decode('utf-8').strip().split('\n')

        except ClientError as e:
            logger.error(f"Failed to download log file {key}: {e}")
            return []

    def analyze_entry(self, entry: Dict[str, Any]):
        """Analyze a single log entry for suspicious activity.

        Args:
            entry: Parsed log entry
        """
        self.results['events_processed'] += 1

        # Update statistics
        self.results['statistics']['by_operation'][entry['operation']] += 1
        self.results['statistics']['by_status_code'][str(entry['http_status'])] += 1
        self.results['statistics']['by_requester'][entry['requester']] += 1
        self.results['statistics']['by_ip'][entry['remote_ip']] += 1
        self.results['statistics']['by_bucket'][entry['bucket']] += 1

        # Check for unauthorized attempts (4xx status codes)
        if str(entry['http_status']) in self.SUSPICIOUS_STATUS_CODES:
            self.results['suspicious_activity']['unauthorized_attempts'].append({
                'timestamp': str(entry.get('timestamp', '')),
                'ip': entry['remote_ip'],
                'operation': entry['operation'],
                'key': entry['key'],
                'status': entry['http_status'],
                'error_code': entry['error_code']
            })

        # Check for unknown IPs
        if self.known_ips and entry['remote_ip'] not in self.known_ips:
            if entry['remote_ip'] != '-':
                self.results['suspicious_activity']['unknown_ips'].append({
                    'timestamp': str(entry.get('timestamp', '')),
                    'ip': entry['remote_ip'],
                    'operation': entry['operation'],
                    'requester': entry['requester']
                })

        # Check for sensitive operations
        if entry['operation'] in self.SUSPICIOUS_OPERATIONS:
            self.results['suspicious_activity']['sensitive_operations'].append({
                'timestamp': str(entry.get('timestamp', '')),
                'ip': entry['remote_ip'],
                'operation': entry['operation'],
                'requester': entry['requester'],
                'bucket': entry['bucket']
            })

    def detect_high_frequency_access(self, window_minutes: int = 5, threshold: int = 100):
        """Detect IPs with unusually high access frequency.

        Args:
            window_minutes: Time window for frequency analysis
            threshold: Number of requests that triggers alert
        """
        ip_counts = self.results['statistics']['by_ip']

        for ip, count in ip_counts.items():
            if count >= threshold:
                self.results['suspicious_activity']['high_frequency_access'].append({
                    'ip': ip,
                    'request_count': count,
                    'threshold': threshold
                })

    def generate_alerts(self) -> List[Dict[str, Any]]:
        """Generate security alerts based on analysis.

        Returns:
            List of alert dictionaries
        """
        alerts = []

        # Alert for high number of unauthorized attempts
        unauth_count = len(self.results['suspicious_activity']['unauthorized_attempts'])
        if unauth_count >= self.alert_threshold:
            alerts.append({
                'severity': 'HIGH',
                'type': 'UNAUTHORIZED_ACCESS_ATTEMPTS',
                'message': f"Detected {unauth_count} unauthorized access attempts",
                'count': unauth_count,
                'details': self.results['suspicious_activity']['unauthorized_attempts'][:10]
            })

        # Alert for unknown IPs
        unknown_ip_count = len(self.results['suspicious_activity']['unknown_ips'])
        if unknown_ip_count >= self.alert_threshold:
            unique_ips = set(
                e['ip'] for e in self.results['suspicious_activity']['unknown_ips']
            )
            alerts.append({
                'severity': 'MEDIUM',
                'type': 'UNKNOWN_IP_ACCESS',
                'message': f"Detected access from {len(unique_ips)} unknown IP addresses",
                'count': unknown_ip_count,
                'unique_ips': list(unique_ips)[:20]
            })

        # Alert for sensitive operations
        sensitive_count = len(self.results['suspicious_activity']['sensitive_operations'])
        if sensitive_count > 0:
            alerts.append({
                'severity': 'CRITICAL',
                'type': 'SENSITIVE_OPERATION_DETECTED',
                'message': f"Detected {sensitive_count} sensitive S3 operations",
                'count': sensitive_count,
                'details': self.results['suspicious_activity']['sensitive_operations']
            })

        # Alert for high frequency access
        high_freq_count = len(self.results['suspicious_activity']['high_frequency_access'])
        if high_freq_count > 0:
            alerts.append({
                'severity': 'MEDIUM',
                'type': 'HIGH_FREQUENCY_ACCESS',
                'message': f"Detected {high_freq_count} IPs with unusually high access frequency",
                'count': high_freq_count,
                'details': self.results['suspicious_activity']['high_frequency_access']
            })

        self.results['alerts'] = alerts
        return alerts

    def analyze_logs(
        self,
        log_bucket: str,
        log_prefix: str = '',
        hours_back: int = 24,
        max_files: int = None
    ) -> Dict[str, Any]:
        """Analyze access logs for a specified time period.

        Args:
            log_bucket: Bucket containing access logs
            log_prefix: Prefix for log files
            hours_back: Number of hours to look back
            max_files: Maximum number of log files to process

        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Analyzing logs from {log_bucket}/{log_prefix}")

        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours_back)

        try:
            # List log files
            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=log_bucket,
                Prefix=log_prefix
            )

            log_files = []
            for page in page_iterator:
                for obj in page.get('Contents', []):
                    # Filter by time (approximate based on last modified)
                    if obj['LastModified'].replace(tzinfo=None) >= start_time:
                        log_files.append(obj['Key'])

            if max_files:
                log_files = log_files[:max_files]

            logger.info(f"Found {len(log_files)} log files to analyze")

            # Process each log file
            for log_file in log_files:
                self.results['logs_analyzed'] += 1
                lines = self.download_log_file(log_bucket, log_file)

                for line in lines:
                    entry = self.parse_log_entry(line)
                    if entry:
                        self.analyze_entry(entry)

            # Detect high frequency access
            self.detect_high_frequency_access()

            # Generate alerts
            self.generate_alerts()

        except ClientError as e:
            logger.error(f"Failed to analyze logs: {e}")
            self.results['error'] = str(e)

        return self.results

    def send_sns_alerts(self, topic_arn: str):
        """Send alerts to SNS topic.

        Args:
            topic_arn: ARN of the SNS topic
        """
        if not self.results['alerts']:
            logger.info("No alerts to send")
            return

        for alert in self.results['alerts']:
            try:
                message = json.dumps({
                    'source': 'GreenLang S3 Access Log Analyzer',
                    'timestamp': datetime.utcnow().isoformat(),
                    'alert': alert
                }, indent=2)

                self.sns_client.publish(
                    TopicArn=topic_arn,
                    Subject=f"[{alert['severity']}] S3 Security Alert: {alert['type']}",
                    Message=message
                )
                logger.info(f"Sent alert to SNS: {alert['type']}")

            except ClientError as e:
                logger.error(f"Failed to send SNS alert: {e}")

    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate JSON report of analysis results.

        Args:
            output_file: Optional file path to write report

        Returns:
            JSON string of the report
        """
        # Convert defaultdicts to regular dicts for JSON serialization
        report_results = dict(self.results)
        report_results['statistics'] = {
            k: dict(v) for k, v in self.results['statistics'].items()
        }

        report = json.dumps(report_results, indent=2, default=str)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            logger.info(f"Report written to {output_file}")

        return report

    def print_summary(self):
        """Print a human-readable summary of the analysis results."""
        print("\n" + "=" * 60)
        print("S3 ACCESS LOG ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Analysis Time:    {self.results['analysis_timestamp']}")
        print(f"Logs Analyzed:    {self.results['logs_analyzed']}")
        print(f"Events Processed: {self.results['events_processed']}")
        print("-" * 60)

        # Operation statistics
        print("\nTop Operations:")
        ops = sorted(
            self.results['statistics']['by_operation'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        for op, count in ops:
            print(f"  {op}: {count}")

        # Status code statistics
        print("\nHTTP Status Codes:")
        for status, count in sorted(self.results['statistics']['by_status_code'].items()):
            print(f"  {status}: {count}")

        # Suspicious activity summary
        print("\n" + "-" * 60)
        print("SUSPICIOUS ACTIVITY DETECTED:")
        print(f"  Unauthorized Attempts: {len(self.results['suspicious_activity']['unauthorized_attempts'])}")
        print(f"  Unknown IPs:           {len(self.results['suspicious_activity']['unknown_ips'])}")
        print(f"  Sensitive Operations:  {len(self.results['suspicious_activity']['sensitive_operations'])}")
        print(f"  High Frequency Access: {len(self.results['suspicious_activity']['high_frequency_access'])}")

        # Alerts
        if self.results['alerts']:
            print("\n" + "=" * 60)
            print("SECURITY ALERTS:")
            print("=" * 60)
            for alert in self.results['alerts']:
                print(f"\n[{alert['severity']}] {alert['type']}")
                print(f"  {alert['message']}")

        print("\n" + "=" * 60)


def main():
    """Main entry point for the access log analyzer."""
    parser = argparse.ArgumentParser(
        description='GreenLang S3 Access Log Analyzer'
    )
    parser.add_argument(
        '--log-bucket',
        required=True,
        help='Bucket containing access logs'
    )
    parser.add_argument(
        '--log-prefix',
        default='',
        help='Prefix for log files'
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
        '--hours',
        type=int,
        default=24,
        help='Number of hours to look back (default: 24)'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        help='Maximum number of log files to process'
    )
    parser.add_argument(
        '--alert-threshold',
        type=int,
        default=10,
        help='Number of suspicious events to trigger alert (default: 10)'
    )
    parser.add_argument(
        '--known-ips',
        help='Comma-separated list of known/allowed IP addresses'
    )
    parser.add_argument(
        '--sns-topic',
        help='SNS topic ARN for alerts'
    )
    parser.add_argument(
        '--output',
        help='Output file for JSON report'
    )
    parser.add_argument(
        '--json-only',
        action='store_true',
        help='Output JSON only (no summary)'
    )

    args = parser.parse_args()

    known_ips = None
    if args.known_ips:
        known_ips = set(ip.strip() for ip in args.known_ips.split(','))

    analyzer = S3AccessLogAnalyzer(
        region=args.region,
        profile=args.profile,
        alert_threshold=args.alert_threshold,
        known_ips=known_ips
    )

    try:
        analyzer.analyze_logs(
            log_bucket=args.log_bucket,
            log_prefix=args.log_prefix,
            hours_back=args.hours,
            max_files=args.max_files
        )

        # Send SNS alerts if configured
        if args.sns_topic and analyzer.results['alerts']:
            analyzer.send_sns_alerts(args.sns_topic)

        if args.json_only:
            print(analyzer.generate_report(args.output))
        else:
            analyzer.print_summary()
            if args.output:
                analyzer.generate_report(args.output)
                print(f"\nFull report written to: {args.output}")

        # Exit with non-zero if there are critical alerts
        critical_alerts = [
            a for a in analyzer.results['alerts']
            if a['severity'] == 'CRITICAL'
        ]
        if critical_alerts:
            sys.exit(1)

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(2)


if __name__ == '__main__':
    main()
