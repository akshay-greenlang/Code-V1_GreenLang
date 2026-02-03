"""
GreenLang Audit Logger Lambda Function
Records S3 events for compliance auditing and tracking.

This function is triggered by S3 events (via SNS/SQS) and performs:
- Recording all S3 events to compliance audit log
- Enriching events with user context and identity information
- Storing audit records in a compliance S3 bucket
- Generating daily audit summaries
"""

import os
import json
import logging
import gzip
from typing import Dict, Any, List, Optional
from datetime import datetime, date
from urllib.parse import unquote_plus
from io import BytesIO

import boto3
from botocore.exceptions import ClientError

# Configure logging
logger = logging.getLogger()
log_level = os.environ.get('LOG_LEVEL', 'INFO')
logger.setLevel(getattr(logging, log_level))

# Environment variables
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'dev')
AUDIT_BUCKET = os.environ.get('AUDIT_BUCKET', '')
AUDIT_PREFIX = os.environ.get('AUDIT_PREFIX', 'audit-logs/')
ENABLE_DAILY_SUMMARY = os.environ.get('ENABLE_DAILY_SUMMARY', 'true').lower() == 'true'
COMPLIANCE_MODE = os.environ.get('COMPLIANCE_MODE', 'standard')  # standard, strict, hipaa, pci
DLQ_URL = os.environ.get('DLQ_URL', '')

# AWS clients
s3_client = boto3.client('s3')
sqs_client = boto3.client('sqs')
sts_client = boto3.client('sts')

# Cache for caller identity
_caller_identity = None


def get_caller_identity() -> Dict[str, str]:
    """Get the current caller identity (cached)."""
    global _caller_identity

    if _caller_identity is None:
        try:
            response = sts_client.get_caller_identity()
            _caller_identity = {
                'account_id': response['Account'],
                'arn': response['Arn'],
                'user_id': response['UserId']
            }
        except ClientError:
            _caller_identity = {}

    return _caller_identity


class AuditRecord:
    """Represents a single audit record for an S3 event."""

    def __init__(self, event: Dict[str, Any]):
        self.event = event
        self.timestamp = datetime.utcnow()
        self.record_id = f"{self.timestamp.strftime('%Y%m%d%H%M%S%f')}-{hash(json.dumps(event, default=str)) % 10000:04d}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert audit record to dictionary format."""
        s3_info = self.event.get('s3', {})
        user_identity = self.event.get('userIdentity', {})
        request_params = self.event.get('requestParameters', {})

        # Build comprehensive audit record
        record = {
            # Record metadata
            'record_id': self.record_id,
            'timestamp': self.timestamp.isoformat(),
            'environment': ENVIRONMENT,
            'compliance_mode': COMPLIANCE_MODE,

            # Event details
            'event_type': self.event.get('eventName', 'Unknown'),
            'event_source': self.event.get('eventSource', 's3.amazonaws.com'),
            'event_time': self.event.get('eventTime', self.timestamp.isoformat()),
            'event_id': self.event.get('eventID', ''),

            # S3 resource information
            's3': {
                'bucket': s3_info.get('bucket', {}).get('name', ''),
                'bucket_arn': s3_info.get('bucket', {}).get('arn', ''),
                'object_key': unquote_plus(s3_info.get('object', {}).get('key', '')),
                'object_size': s3_info.get('object', {}).get('size'),
                'object_etag': s3_info.get('object', {}).get('eTag', ''),
                'object_version_id': s3_info.get('object', {}).get('versionId'),
                'object_sequencer': s3_info.get('object', {}).get('sequencer', '')
            },

            # User identity information
            'user_identity': {
                'type': user_identity.get('type', ''),
                'principal_id': user_identity.get('principalId', ''),
                'arn': user_identity.get('arn', ''),
                'account_id': user_identity.get('accountId', ''),
                'access_key_id': user_identity.get('accessKeyId', ''),
                'user_name': user_identity.get('userName', ''),
                'invoked_by': user_identity.get('invokedBy', '')
            },

            # Session context (if assumed role)
            'session_context': self._extract_session_context(user_identity),

            # Request information
            'request': {
                'source_ip': self.event.get('sourceIPAddress', ''),
                'user_agent': self.event.get('userAgent', ''),
                'request_id': self.event.get('requestID', ''),
                'parameters': self._sanitize_parameters(request_params)
            },

            # Response information
            'response': {
                'elements': self._sanitize_response(self.event.get('responseElements', {})),
                'error_code': self.event.get('errorCode'),
                'error_message': self.event.get('errorMessage')
            },

            # Additional context
            'context': {
                'aws_region': self.event.get('awsRegion', os.environ.get('AWS_REGION', '')),
                'vpc_endpoint_id': self.event.get('vpcEndpointId'),
                'read_only': self.event.get('readOnly', False),
                'resources': self.event.get('resources', [])
            },

            # Audit metadata
            'audit_metadata': {
                'processor_arn': get_caller_identity().get('arn', ''),
                'processor_account': get_caller_identity().get('account_id', ''),
                'processing_timestamp': self.timestamp.isoformat(),
                'audit_version': '1.0.0'
            }
        }

        # Add compliance-specific fields based on mode
        if COMPLIANCE_MODE in ('strict', 'hipaa', 'pci'):
            record['compliance'] = {
                'mode': COMPLIANCE_MODE,
                'data_classification': self._classify_data(s3_info),
                'retention_required': True,
                'encryption_verified': self._verify_encryption(self.event)
            }

        return record

    def _extract_session_context(self, user_identity: Dict) -> Optional[Dict]:
        """Extract session context from assumed role."""
        session_context = user_identity.get('sessionContext', {})

        if not session_context:
            return None

        session_issuer = session_context.get('sessionIssuer', {})
        attributes = session_context.get('attributes', {})

        return {
            'session_issuer': {
                'type': session_issuer.get('type', ''),
                'principal_id': session_issuer.get('principalId', ''),
                'arn': session_issuer.get('arn', ''),
                'account_id': session_issuer.get('accountId', ''),
                'user_name': session_issuer.get('userName', '')
            },
            'attributes': {
                'mfa_authenticated': attributes.get('mfaAuthenticated', 'false'),
                'creation_date': attributes.get('creationDate', '')
            }
        }

    def _sanitize_parameters(self, params: Dict) -> Dict:
        """Sanitize request parameters, removing sensitive data."""
        if not params:
            return {}

        # Fields that should be redacted
        sensitive_fields = {'authorization', 'x-amz-security-token', 'password', 'secret'}

        sanitized = {}
        for key, value in params.items():
            if key.lower() in sensitive_fields:
                sanitized[key] = '[REDACTED]'
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_parameters(value)
            else:
                sanitized[key] = value

        return sanitized

    def _sanitize_response(self, response: Any) -> Any:
        """Sanitize response elements."""
        if not response:
            return None

        if isinstance(response, dict):
            return self._sanitize_parameters(response)

        return str(response)[:1000]  # Limit response size

    def _classify_data(self, s3_info: Dict) -> str:
        """Classify data sensitivity based on bucket/key patterns."""
        bucket = s3_info.get('bucket', {}).get('name', '').lower()
        key = s3_info.get('object', {}).get('key', '').lower()

        # Classification rules
        if any(x in bucket or x in key for x in ['pii', 'personal', 'sensitive', 'confidential']):
            return 'HIGHLY_SENSITIVE'
        elif any(x in bucket or x in key for x in ['internal', 'private']):
            return 'INTERNAL'
        elif any(x in bucket or x in key for x in ['public', 'static']):
            return 'PUBLIC'
        else:
            return 'STANDARD'

    def _verify_encryption(self, event: Dict) -> bool:
        """Verify if the request/response indicates encryption."""
        response = event.get('responseElements', {}) or {}

        # Check for server-side encryption headers
        sse_headers = ['x-amz-server-side-encryption', 'SSEKMSKeyId', 'x-amz-server-side-encryption-aws-kms-key-id']

        return any(header in str(response) for header in sse_headers)


def write_audit_record(record: Dict[str, Any]) -> bool:
    """Write a single audit record to the audit bucket."""
    if not AUDIT_BUCKET:
        logger.warning("No audit bucket configured - record not persisted")
        return False

    try:
        timestamp = datetime.fromisoformat(record['timestamp'].replace('Z', '+00:00'))

        # Partition by date for efficient querying
        partition_path = timestamp.strftime('%Y/%m/%d/%H')
        key = f"{AUDIT_PREFIX}{partition_path}/{record['record_id']}.json.gz"

        # Compress the record
        json_bytes = json.dumps(record, default=str).encode('utf-8')
        compressed = BytesIO()
        with gzip.GzipFile(fileobj=compressed, mode='wb') as gz:
            gz.write(json_bytes)
        compressed.seek(0)

        # Write to S3
        s3_client.put_object(
            Bucket=AUDIT_BUCKET,
            Key=key,
            Body=compressed.read(),
            ContentType='application/json',
            ContentEncoding='gzip',
            Metadata={
                'record-id': record['record_id'],
                'event-type': record['event_type'],
                'compliance-mode': COMPLIANCE_MODE
            }
        )

        logger.info(f"Wrote audit record {record['record_id']} to s3://{AUDIT_BUCKET}/{key}")
        return True

    except ClientError as e:
        logger.error(f"Failed to write audit record: {e}")
        return False


def write_batch_records(records: List[Dict[str, Any]]) -> int:
    """Write multiple audit records efficiently."""
    if not AUDIT_BUCKET or not records:
        return 0

    try:
        timestamp = datetime.utcnow()
        partition_path = timestamp.strftime('%Y/%m/%d/%H')
        batch_id = timestamp.strftime('%Y%m%d%H%M%S%f')
        key = f"{AUDIT_PREFIX}{partition_path}/batch-{batch_id}.json.gz"

        # Combine records into newline-delimited JSON
        json_lines = '\n'.join(json.dumps(r, default=str) for r in records)

        # Compress
        compressed = BytesIO()
        with gzip.GzipFile(fileobj=compressed, mode='wb') as gz:
            gz.write(json_lines.encode('utf-8'))
        compressed.seek(0)

        # Write to S3
        s3_client.put_object(
            Bucket=AUDIT_BUCKET,
            Key=key,
            Body=compressed.read(),
            ContentType='application/x-ndjson',
            ContentEncoding='gzip',
            Metadata={
                'batch-id': batch_id,
                'record-count': str(len(records)),
                'compliance-mode': COMPLIANCE_MODE
            }
        )

        logger.info(f"Wrote batch of {len(records)} audit records to s3://{AUDIT_BUCKET}/{key}")
        return len(records)

    except ClientError as e:
        logger.error(f"Failed to write batch audit records: {e}")
        return 0


def generate_daily_summary(records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Generate a summary of daily audit events."""
    if not records:
        return None

    today = date.today().isoformat()

    # Aggregate statistics
    event_types = {}
    buckets = {}
    users = {}
    errors = []

    for record in records:
        # Count event types
        event_type = record.get('event_type', 'Unknown')
        event_types[event_type] = event_types.get(event_type, 0) + 1

        # Count bucket access
        bucket = record.get('s3', {}).get('bucket', 'Unknown')
        buckets[bucket] = buckets.get(bucket, 0) + 1

        # Count user activity
        user = record.get('user_identity', {}).get('arn', 'Unknown')
        users[user] = users.get(user, 0) + 1

        # Track errors
        if record.get('response', {}).get('error_code'):
            errors.append({
                'error_code': record['response']['error_code'],
                'error_message': record['response'].get('error_message', ''),
                'event_type': event_type,
                'timestamp': record.get('timestamp')
            })

    summary = {
        'summary_id': f"summary-{today}",
        'date': today,
        'generated_at': datetime.utcnow().isoformat(),
        'environment': ENVIRONMENT,
        'compliance_mode': COMPLIANCE_MODE,
        'statistics': {
            'total_events': len(records),
            'unique_buckets': len(buckets),
            'unique_users': len(users),
            'error_count': len(errors)
        },
        'event_type_breakdown': event_types,
        'bucket_access_breakdown': dict(sorted(buckets.items(), key=lambda x: x[1], reverse=True)[:20]),
        'top_users': dict(sorted(users.items(), key=lambda x: x[1], reverse=True)[:10]),
        'recent_errors': errors[-20:] if errors else []
    }

    return summary


def send_to_dlq(record: Dict, error: str) -> None:
    """Send failed message to dead letter queue."""
    if not DLQ_URL:
        return

    try:
        sqs_client.send_message(
            QueueUrl=DLQ_URL,
            MessageBody=json.dumps({
                'original_record': record,
                'error': error,
                'timestamp': datetime.utcnow().isoformat()
            })
        )
    except ClientError as e:
        logger.error(f"Failed to send to DLQ: {e}")


def process_event(event_data: Dict) -> Dict[str, Any]:
    """Process a single S3/CloudTrail event into an audit record."""
    audit_record = AuditRecord(event_data)
    record_dict = audit_record.to_dict()

    # Write individual record
    success = write_audit_record(record_dict)

    return {
        'record_id': record_dict['record_id'],
        'event_type': record_dict['event_type'],
        'bucket': record_dict['s3']['bucket'],
        'key': record_dict['s3']['object_key'],
        'persisted': success
    }


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for audit logging.

    Supports:
    - Direct S3 events
    - SNS-wrapped S3 events
    - SQS-wrapped events
    - EventBridge CloudTrail events
    """
    logger.info(f"Received event: {json.dumps(event)[:1000]}")

    results = []
    audit_records = []
    batch_item_failures = []

    # Handle EventBridge events (CloudTrail)
    if event.get('source') == 'aws.s3' and event.get('detail-type') == 'AWS API Call via CloudTrail':
        detail = event.get('detail', {})
        result = process_event(detail)
        results.append(result)
        audit_records.append(AuditRecord(detail).to_dict())

    # Handle SQS events
    elif 'Records' in event and event['Records'][0].get('eventSource') == 'aws:sqs':
        for sqs_record in event['Records']:
            message_id = sqs_record.get('messageId', '')
            try:
                body = json.loads(sqs_record.get('body', '{}'))

                # Handle SNS-wrapped S3 events
                if 'Records' in body:
                    s3_records = body['Records']
                elif 'Message' in body:
                    s3_records = json.loads(body['Message']).get('Records', [])
                else:
                    s3_records = [body]

                for s3_record in s3_records:
                    result = process_event(s3_record)
                    results.append(result)
                    audit_records.append(AuditRecord(s3_record).to_dict())

            except Exception as e:
                logger.error(f"Failed to process message {message_id}: {str(e)}")
                batch_item_failures.append({'itemIdentifier': message_id})
                send_to_dlq(sqs_record, str(e))

    # Handle direct S3 events
    elif 'Records' in event:
        for record in event['Records']:
            try:
                result = process_event(record)
                results.append(result)
                audit_records.append(AuditRecord(record).to_dict())
            except Exception as e:
                logger.error(f"Failed to process event: {str(e)}")
                results.append({'status': 'error', 'message': str(e)})

    # Generate daily summary if enabled
    summary = None
    if ENABLE_DAILY_SUMMARY and audit_records:
        summary = generate_daily_summary(audit_records)
        if summary:
            logger.info(f"Generated daily summary: {summary['statistics']}")

    # Build response
    response = {
        'statusCode': 200,
        'body': {
            'processed': len(results),
            'persisted': sum(1 for r in results if r.get('persisted')),
            'errors': sum(1 for r in results if not r.get('persisted')),
            'results': results,
            'summary': summary
        }
    }

    if batch_item_failures:
        response['batchItemFailures'] = batch_item_failures

    logger.info(f"Audit logging complete: {response['body']['processed']} events processed")
    return response
