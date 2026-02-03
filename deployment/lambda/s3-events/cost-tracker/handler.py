"""
GreenLang Cost Tracker Lambda Function
Tracks and reports S3 storage costs for monitored buckets.

This function runs on a schedule and performs:
- Retrieves storage metrics for monitored S3 buckets
- Calculates estimated costs by storage class
- Publishes custom CloudWatch metrics
- Sends alerts when cost thresholds are exceeded
- Generates daily/weekly cost reports
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from decimal import Decimal

import boto3
from botocore.exceptions import ClientError

# Configure logging
logger = logging.getLogger()
log_level = os.environ.get('LOG_LEVEL', 'INFO')
logger.setLevel(getattr(logging, log_level))

# Environment variables
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'dev')
COST_ALLOCATION_TAG = os.environ.get('COST_ALLOCATION_TAG', 'greenlang-storage')
ALERT_THRESHOLD_USD = float(os.environ.get('ALERT_THRESHOLD_USD', '1000'))
ALERT_SNS_TOPIC_ARN = os.environ.get('ALERT_SNS_TOPIC_ARN', '')
MONITORED_BUCKETS = os.environ.get('MONITORED_BUCKETS', '').split(',')

# AWS clients
s3_client = boto3.client('s3')
cloudwatch_client = boto3.client('cloudwatch')
ce_client = boto3.client('ce')
sns_client = boto3.client('sns')

# S3 storage pricing (US East region, approximate)
# https://aws.amazon.com/s3/pricing/
STORAGE_PRICING_PER_GB_MONTH = {
    'STANDARD': 0.023,
    'INTELLIGENT_TIERING': 0.023,  # Frequent access tier
    'STANDARD_IA': 0.0125,
    'ONEZONE_IA': 0.01,
    'GLACIER_IR': 0.004,
    'GLACIER': 0.004,
    'DEEP_ARCHIVE': 0.00099
}

# Request pricing (per 1000 requests)
REQUEST_PRICING = {
    'PUT': 0.005,
    'COPY': 0.005,
    'POST': 0.005,
    'LIST': 0.005,
    'GET': 0.0004,
    'SELECT': 0.0004
}


class BucketMetrics:
    """Collects and calculates metrics for a single S3 bucket."""

    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        self.metrics: Dict[str, Any] = {}

    def collect(self) -> Dict[str, Any]:
        """Collect all metrics for the bucket."""
        try:
            # Get bucket location
            location = self._get_bucket_location()

            # Get storage metrics from CloudWatch
            storage_metrics = self._get_storage_metrics()

            # Get request metrics
            request_metrics = self._get_request_metrics()

            # Calculate costs
            cost_estimate = self._calculate_costs(storage_metrics, request_metrics)

            self.metrics = {
                'bucket_name': self.bucket_name,
                'region': location,
                'collection_timestamp': datetime.utcnow().isoformat(),
                'storage': storage_metrics,
                'requests': request_metrics,
                'cost_estimate': cost_estimate
            }

            return self.metrics

        except ClientError as e:
            logger.error(f"Error collecting metrics for {self.bucket_name}: {e}")
            return {
                'bucket_name': self.bucket_name,
                'error': str(e),
                'collection_timestamp': datetime.utcnow().isoformat()
            }

    def _get_bucket_location(self) -> str:
        """Get bucket region."""
        try:
            response = s3_client.get_bucket_location(Bucket=self.bucket_name)
            location = response.get('LocationConstraint', 'us-east-1')
            return location or 'us-east-1'  # None means us-east-1
        except ClientError:
            return 'unknown'

    def _get_storage_metrics(self) -> Dict[str, Any]:
        """Get storage metrics from S3/CloudWatch."""
        metrics = {
            'total_size_bytes': 0,
            'total_objects': 0,
            'by_storage_class': {}
        }

        # Query CloudWatch for S3 storage metrics
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=1)

        storage_types = [
            ('StandardStorage', 'STANDARD'),
            ('IntelligentTieringStorage', 'INTELLIGENT_TIERING'),
            ('StandardIAStorage', 'STANDARD_IA'),
            ('OneZoneIAStorage', 'ONEZONE_IA'),
            ('GlacierStorage', 'GLACIER'),
            ('DeepArchiveStorage', 'DEEP_ARCHIVE')
        ]

        for metric_name, storage_class in storage_types:
            try:
                response = cloudwatch_client.get_metric_statistics(
                    Namespace='AWS/S3',
                    MetricName='BucketSizeBytes',
                    Dimensions=[
                        {'Name': 'BucketName', 'Value': self.bucket_name},
                        {'Name': 'StorageType', 'Value': metric_name}
                    ],
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=86400,  # 1 day
                    Statistics=['Average']
                )

                datapoints = response.get('Datapoints', [])
                if datapoints:
                    size_bytes = int(datapoints[-1].get('Average', 0))
                    if size_bytes > 0:
                        metrics['by_storage_class'][storage_class] = {
                            'size_bytes': size_bytes,
                            'size_gb': size_bytes / (1024 ** 3)
                        }
                        metrics['total_size_bytes'] += size_bytes

            except ClientError as e:
                logger.warning(f"Failed to get {metric_name} for {self.bucket_name}: {e}")

        # Get object count
        try:
            response = cloudwatch_client.get_metric_statistics(
                Namespace='AWS/S3',
                MetricName='NumberOfObjects',
                Dimensions=[
                    {'Name': 'BucketName', 'Value': self.bucket_name},
                    {'Name': 'StorageType', 'Value': 'AllStorageTypes'}
                ],
                StartTime=start_time,
                EndTime=end_time,
                Period=86400,
                Statistics=['Average']
            )

            datapoints = response.get('Datapoints', [])
            if datapoints:
                metrics['total_objects'] = int(datapoints[-1].get('Average', 0))

        except ClientError as e:
            logger.warning(f"Failed to get object count for {self.bucket_name}: {e}")

        metrics['total_size_gb'] = metrics['total_size_bytes'] / (1024 ** 3)

        return metrics

    def _get_request_metrics(self) -> Dict[str, Any]:
        """Get request metrics from CloudWatch."""
        metrics = {
            'total_requests': 0,
            'by_type': {}
        }

        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=1)

        request_types = ['GetRequests', 'PutRequests', 'DeleteRequests', 'ListRequests']

        for request_type in request_types:
            try:
                response = cloudwatch_client.get_metric_statistics(
                    Namespace='AWS/S3',
                    MetricName=request_type,
                    Dimensions=[
                        {'Name': 'BucketName', 'Value': self.bucket_name}
                    ],
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=86400,
                    Statistics=['Sum']
                )

                datapoints = response.get('Datapoints', [])
                if datapoints:
                    count = int(datapoints[-1].get('Sum', 0))
                    metrics['by_type'][request_type.replace('Requests', '')] = count
                    metrics['total_requests'] += count

            except ClientError as e:
                logger.debug(f"Failed to get {request_type} for {self.bucket_name}: {e}")

        return metrics

    def _calculate_costs(self, storage: Dict, requests: Dict) -> Dict[str, Any]:
        """Calculate estimated costs based on metrics."""
        costs = {
            'storage_cost_usd': 0.0,
            'request_cost_usd': 0.0,
            'total_cost_usd': 0.0,
            'by_storage_class': {},
            'by_request_type': {}
        }

        # Calculate storage costs
        for storage_class, data in storage.get('by_storage_class', {}).items():
            price_per_gb = STORAGE_PRICING_PER_GB_MONTH.get(storage_class, 0.023)
            size_gb = data.get('size_gb', 0)
            cost = size_gb * price_per_gb

            costs['by_storage_class'][storage_class] = {
                'size_gb': round(size_gb, 2),
                'monthly_cost_usd': round(cost, 4)
            }
            costs['storage_cost_usd'] += cost

        # Calculate request costs
        for request_type, count in requests.get('by_type', {}).items():
            # Map request type to pricing
            if request_type in ('Get', 'SELECT'):
                price = REQUEST_PRICING['GET']
            elif request_type in ('Put', 'Post', 'Copy', 'Delete'):
                price = REQUEST_PRICING['PUT']
            elif request_type == 'List':
                price = REQUEST_PRICING['LIST']
            else:
                price = REQUEST_PRICING['GET']

            cost = (count / 1000) * price

            costs['by_request_type'][request_type] = {
                'count': count,
                'daily_cost_usd': round(cost, 4)
            }
            costs['request_cost_usd'] += cost

        # Total monthly estimate
        costs['storage_cost_usd'] = round(costs['storage_cost_usd'], 2)
        costs['request_cost_usd'] = round(costs['request_cost_usd'] * 30, 2)  # Project to monthly
        costs['total_cost_usd'] = round(costs['storage_cost_usd'] + costs['request_cost_usd'], 2)

        return costs


def get_actual_costs_from_cost_explorer() -> Optional[Dict[str, Any]]:
    """Get actual S3 costs from AWS Cost Explorer."""
    try:
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=30)

        response = ce_client.get_cost_and_usage(
            TimePeriod={
                'Start': start_date.isoformat(),
                'End': end_date.isoformat()
            },
            Granularity='MONTHLY',
            Metrics=['BlendedCost', 'UnblendedCost', 'UsageQuantity'],
            Filter={
                'Dimensions': {
                    'Key': 'SERVICE',
                    'Values': ['Amazon Simple Storage Service']
                }
            },
            GroupBy=[
                {'Type': 'DIMENSION', 'Key': 'USAGE_TYPE'}
            ]
        )

        costs = {
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'total_cost': 0.0,
            'by_usage_type': {}
        }

        for result in response.get('ResultsByTime', []):
            for group in result.get('Groups', []):
                usage_type = group['Keys'][0]
                amount = float(group['Metrics']['BlendedCost']['Amount'])

                costs['by_usage_type'][usage_type] = round(amount, 4)
                costs['total_cost'] += amount

        costs['total_cost'] = round(costs['total_cost'], 2)

        return costs

    except ClientError as e:
        logger.warning(f"Failed to get Cost Explorer data: {e}")
        return None


def publish_metrics(metrics: List[Dict[str, Any]]) -> None:
    """Publish custom CloudWatch metrics."""
    metric_data = []

    for bucket_metrics in metrics:
        if 'error' in bucket_metrics:
            continue

        bucket_name = bucket_metrics['bucket_name']
        cost_estimate = bucket_metrics.get('cost_estimate', {})
        storage = bucket_metrics.get('storage', {})

        # Total cost metric
        metric_data.append({
            'MetricName': 'EstimatedMonthlyCost',
            'Dimensions': [
                {'Name': 'BucketName', 'Value': bucket_name},
                {'Name': 'Environment', 'Value': ENVIRONMENT}
            ],
            'Value': cost_estimate.get('total_cost_usd', 0),
            'Unit': 'None',
            'Timestamp': datetime.utcnow()
        })

        # Storage size metric
        metric_data.append({
            'MetricName': 'TotalStorageGB',
            'Dimensions': [
                {'Name': 'BucketName', 'Value': bucket_name},
                {'Name': 'Environment', 'Value': ENVIRONMENT}
            ],
            'Value': storage.get('total_size_gb', 0),
            'Unit': 'Gigabytes',
            'Timestamp': datetime.utcnow()
        })

        # Object count metric
        metric_data.append({
            'MetricName': 'TotalObjects',
            'Dimensions': [
                {'Name': 'BucketName', 'Value': bucket_name},
                {'Name': 'Environment', 'Value': ENVIRONMENT}
            ],
            'Value': storage.get('total_objects', 0),
            'Unit': 'Count',
            'Timestamp': datetime.utcnow()
        })

    # Publish in batches of 20
    for i in range(0, len(metric_data), 20):
        batch = metric_data[i:i + 20]
        try:
            cloudwatch_client.put_metric_data(
                Namespace='GreenLang/S3/Costs',
                MetricData=batch
            )
            logger.info(f"Published {len(batch)} metrics to CloudWatch")
        except ClientError as e:
            logger.error(f"Failed to publish metrics: {e}")


def check_cost_alerts(metrics: List[Dict[str, Any]], actual_costs: Optional[Dict]) -> List[Dict]:
    """Check if costs exceed thresholds and generate alerts."""
    alerts = []

    # Calculate total estimated cost
    total_estimated = sum(
        m.get('cost_estimate', {}).get('total_cost_usd', 0)
        for m in metrics if 'error' not in m
    )

    # Check against threshold
    if total_estimated > ALERT_THRESHOLD_USD:
        alerts.append({
            'type': 'COST_THRESHOLD_EXCEEDED',
            'severity': 'WARNING',
            'message': f"Estimated monthly S3 cost ${total_estimated:.2f} exceeds threshold ${ALERT_THRESHOLD_USD:.2f}",
            'current_cost': total_estimated,
            'threshold': ALERT_THRESHOLD_USD,
            'timestamp': datetime.utcnow().isoformat()
        })

    # Check actual costs if available
    if actual_costs and actual_costs.get('total_cost', 0) > ALERT_THRESHOLD_USD:
        alerts.append({
            'type': 'ACTUAL_COST_THRESHOLD_EXCEEDED',
            'severity': 'CRITICAL',
            'message': f"Actual S3 cost ${actual_costs['total_cost']:.2f} exceeds threshold ${ALERT_THRESHOLD_USD:.2f}",
            'current_cost': actual_costs['total_cost'],
            'threshold': ALERT_THRESHOLD_USD,
            'timestamp': datetime.utcnow().isoformat()
        })

    # Check individual bucket costs
    for bucket_metrics in metrics:
        if 'error' in bucket_metrics:
            continue

        bucket_cost = bucket_metrics.get('cost_estimate', {}).get('total_cost_usd', 0)
        bucket_threshold = ALERT_THRESHOLD_USD / len(metrics) * 2  # Per-bucket threshold

        if bucket_cost > bucket_threshold:
            alerts.append({
                'type': 'BUCKET_COST_HIGH',
                'severity': 'INFO',
                'message': f"Bucket {bucket_metrics['bucket_name']} cost ${bucket_cost:.2f} is high",
                'bucket': bucket_metrics['bucket_name'],
                'current_cost': bucket_cost,
                'timestamp': datetime.utcnow().isoformat()
            })

    return alerts


def send_alerts(alerts: List[Dict]) -> None:
    """Send alerts via SNS."""
    if not ALERT_SNS_TOPIC_ARN or not alerts:
        return

    for alert in alerts:
        try:
            subject = f"[{alert['severity']}] GreenLang S3 Cost Alert: {alert['type']}"

            message = json.dumps({
                'default': alert['message'],
                'sms': f"S3 Alert: {alert['message'][:100]}",
                'email': json.dumps(alert, indent=2)
            })

            sns_client.publish(
                TopicArn=ALERT_SNS_TOPIC_ARN,
                Subject=subject[:100],
                Message=message,
                MessageStructure='json'
            )

            logger.info(f"Sent alert: {alert['type']}")

        except ClientError as e:
            logger.error(f"Failed to send alert: {e}")


def generate_cost_report(metrics: List[Dict], actual_costs: Optional[Dict]) -> Dict[str, Any]:
    """Generate a comprehensive cost report."""
    report = {
        'report_id': f"cost-report-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
        'generated_at': datetime.utcnow().isoformat(),
        'environment': ENVIRONMENT,
        'period': 'monthly_estimate',
        'summary': {
            'total_buckets': len(metrics),
            'buckets_with_errors': sum(1 for m in metrics if 'error' in m),
            'total_storage_gb': 0,
            'total_objects': 0,
            'estimated_monthly_cost_usd': 0,
            'actual_monthly_cost_usd': actual_costs.get('total_cost') if actual_costs else None
        },
        'bucket_details': [],
        'cost_breakdown': {
            'by_bucket': {},
            'by_storage_class': {}
        }
    }

    # Aggregate data
    storage_class_totals = {}

    for bucket_metrics in metrics:
        if 'error' in bucket_metrics:
            report['bucket_details'].append({
                'bucket': bucket_metrics['bucket_name'],
                'status': 'error',
                'error': bucket_metrics['error']
            })
            continue

        storage = bucket_metrics.get('storage', {})
        cost = bucket_metrics.get('cost_estimate', {})

        # Update summary
        report['summary']['total_storage_gb'] += storage.get('total_size_gb', 0)
        report['summary']['total_objects'] += storage.get('total_objects', 0)
        report['summary']['estimated_monthly_cost_usd'] += cost.get('total_cost_usd', 0)

        # Bucket details
        report['bucket_details'].append({
            'bucket': bucket_metrics['bucket_name'],
            'status': 'ok',
            'storage_gb': round(storage.get('total_size_gb', 0), 2),
            'objects': storage.get('total_objects', 0),
            'monthly_cost_usd': cost.get('total_cost_usd', 0)
        })

        # Cost by bucket
        report['cost_breakdown']['by_bucket'][bucket_metrics['bucket_name']] = cost.get('total_cost_usd', 0)

        # Aggregate storage class costs
        for sc, sc_data in cost.get('by_storage_class', {}).items():
            if sc not in storage_class_totals:
                storage_class_totals[sc] = {'size_gb': 0, 'cost_usd': 0}
            storage_class_totals[sc]['size_gb'] += sc_data.get('size_gb', 0)
            storage_class_totals[sc]['cost_usd'] += sc_data.get('monthly_cost_usd', 0)

    report['cost_breakdown']['by_storage_class'] = {
        sc: {'size_gb': round(data['size_gb'], 2), 'monthly_cost_usd': round(data['cost_usd'], 2)}
        for sc, data in storage_class_totals.items()
    }

    # Round summary values
    report['summary']['total_storage_gb'] = round(report['summary']['total_storage_gb'], 2)
    report['summary']['estimated_monthly_cost_usd'] = round(report['summary']['estimated_monthly_cost_usd'], 2)

    return report


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for S3 cost tracking.

    Typically invoked on a schedule (e.g., daily).
    """
    logger.info(f"Starting cost tracking for {len(MONITORED_BUCKETS)} buckets")

    # Filter out empty bucket names
    buckets_to_process = [b.strip() for b in MONITORED_BUCKETS if b.strip()]

    if not buckets_to_process:
        # If no specific buckets, try to list all buckets
        try:
            response = s3_client.list_buckets()
            buckets_to_process = [b['Name'] for b in response.get('Buckets', [])]
            # Filter to greenlang buckets
            buckets_to_process = [b for b in buckets_to_process if 'greenlang' in b.lower()]
        except ClientError as e:
            logger.error(f"Failed to list buckets: {e}")
            buckets_to_process = []

    # Collect metrics for each bucket
    all_metrics = []
    for bucket_name in buckets_to_process:
        logger.info(f"Collecting metrics for bucket: {bucket_name}")
        collector = BucketMetrics(bucket_name)
        metrics = collector.collect()
        all_metrics.append(metrics)

    # Get actual costs from Cost Explorer
    actual_costs = get_actual_costs_from_cost_explorer()

    # Publish metrics to CloudWatch
    publish_metrics(all_metrics)

    # Check for cost alerts
    alerts = check_cost_alerts(all_metrics, actual_costs)

    # Send alerts if any
    if alerts:
        send_alerts(alerts)

    # Generate report
    report = generate_cost_report(all_metrics, actual_costs)

    logger.info(f"Cost tracking complete. Summary: {report['summary']}")

    return {
        'statusCode': 200,
        'body': {
            'buckets_processed': len(all_metrics),
            'alerts_generated': len(alerts),
            'report': report
        }
    }
