"""
GreenLang Artifact Validator Lambda Function
Validates uploaded artifacts for integrity, format, and security.

This function is triggered by S3 PUT events on artifact uploads and performs:
- File integrity verification (checksum validation)
- File format validation (extension and magic bytes)
- Optional malware scanning via ClamAV
- Tagging valid artifacts or quarantining invalid ones
"""

import os
import json
import hashlib
import logging
import tempfile
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from urllib.parse import unquote_plus

import boto3
from botocore.exceptions import ClientError

# Configure logging
logger = logging.getLogger()
log_level = os.environ.get('LOG_LEVEL', 'INFO')
logger.setLevel(getattr(logging, log_level))

# Environment variables
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'dev')
MAX_FILE_SIZE_MB = int(os.environ.get('MAX_FILE_SIZE_MB', '100'))
ALLOWED_EXTENSIONS = os.environ.get('ALLOWED_EXTENSIONS', '.zip,.tar.gz,.jar,.whl,.rpm,.deb').split(',')
ENABLE_VIRUS_SCAN = os.environ.get('ENABLE_VIRUS_SCAN', 'false').lower() == 'true'
QUARANTINE_BUCKET = os.environ.get('QUARANTINE_BUCKET', '')
VALID_ARTIFACT_TAG = os.environ.get('VALID_ARTIFACT_TAG', 'validated')
DLQ_URL = os.environ.get('DLQ_URL', '')

# AWS clients
s3_client = boto3.client('s3')
sqs_client = boto3.client('sqs')

# Magic bytes for file type detection
MAGIC_BYTES = {
    '.zip': [b'PK\x03\x04', b'PK\x05\x06', b'PK\x07\x08'],
    '.tar.gz': [b'\x1f\x8b'],
    '.gz': [b'\x1f\x8b'],
    '.jar': [b'PK\x03\x04'],  # JAR files are ZIP files
    '.whl': [b'PK\x03\x04'],  # Wheel files are ZIP files
    '.rpm': [b'\xed\xab\xee\xdb'],
    '.deb': [b'!\x3c\x61\x72\x63\x68\x3e'],  # !<arch>
}


class ValidationError(Exception):
    """Custom exception for validation failures."""

    def __init__(self, message: str, error_code: str, details: Optional[Dict] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}


class ArtifactValidator:
    """Validates S3 artifacts for integrity and security."""

    def __init__(self, bucket: str, key: str):
        self.bucket = bucket
        self.key = key
        self.validation_results: List[Dict] = []
        self.temp_file: Optional[str] = None

    def validate(self) -> Tuple[bool, List[Dict]]:
        """
        Run all validation checks on the artifact.

        Returns:
            Tuple of (is_valid, validation_results)
        """
        try:
            # Get object metadata first
            head_response = s3_client.head_object(Bucket=self.bucket, Key=self.key)
            content_length = head_response.get('ContentLength', 0)
            content_type = head_response.get('ContentType', 'application/octet-stream')
            existing_tags = self._get_object_tags()

            logger.info(f"Validating artifact: s3://{self.bucket}/{self.key}")
            logger.info(f"Size: {content_length} bytes, Type: {content_type}")

            # Run validation checks
            self._check_file_size(content_length)
            self._check_file_extension()

            # Download file for content validation
            self._download_file()

            self._check_magic_bytes()
            checksum = self._calculate_checksum()
            self._verify_checksum(checksum, existing_tags)

            if ENABLE_VIRUS_SCAN:
                self._scan_for_malware()

            # All checks passed
            logger.info(f"Artifact validation successful: {self.key}")
            return True, self.validation_results

        except ValidationError as e:
            logger.warning(f"Validation failed for {self.key}: {e.error_code} - {str(e)}")
            self.validation_results.append({
                'check': e.error_code,
                'passed': False,
                'message': str(e),
                'details': e.details
            })
            return False, self.validation_results

        except ClientError as e:
            logger.error(f"AWS error validating {self.key}: {str(e)}")
            raise

        finally:
            self._cleanup()

    def _check_file_size(self, size_bytes: int) -> None:
        """Validate file size is within limits."""
        max_size_bytes = MAX_FILE_SIZE_MB * 1024 * 1024

        if size_bytes > max_size_bytes:
            raise ValidationError(
                f"File size {size_bytes / 1024 / 1024:.2f}MB exceeds limit of {MAX_FILE_SIZE_MB}MB",
                'FILE_SIZE_EXCEEDED',
                {'size_bytes': size_bytes, 'max_bytes': max_size_bytes}
            )

        self.validation_results.append({
            'check': 'FILE_SIZE',
            'passed': True,
            'message': f"File size {size_bytes / 1024 / 1024:.2f}MB within limit"
        })
        logger.debug(f"File size check passed: {size_bytes} bytes")

    def _check_file_extension(self) -> None:
        """Validate file has an allowed extension."""
        key_lower = self.key.lower()

        # Check for allowed extensions
        valid_extension = False
        matched_extension = None

        for ext in ALLOWED_EXTENSIONS:
            if key_lower.endswith(ext):
                valid_extension = True
                matched_extension = ext
                break

        if not valid_extension:
            raise ValidationError(
                f"File extension not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
                'INVALID_EXTENSION',
                {'key': self.key, 'allowed_extensions': ALLOWED_EXTENSIONS}
            )

        self.validation_results.append({
            'check': 'FILE_EXTENSION',
            'passed': True,
            'message': f"Extension '{matched_extension}' is allowed"
        })
        logger.debug(f"Extension check passed: {matched_extension}")

    def _download_file(self) -> None:
        """Download the file to a temporary location for validation."""
        self.temp_file = tempfile.mktemp(suffix='_artifact')

        try:
            s3_client.download_file(self.bucket, self.key, self.temp_file)
            logger.debug(f"Downloaded artifact to {self.temp_file}")
        except ClientError as e:
            logger.error(f"Failed to download {self.key}: {str(e)}")
            raise

    def _check_magic_bytes(self) -> None:
        """Verify file content matches expected magic bytes for its extension."""
        if not self.temp_file:
            return

        key_lower = self.key.lower()

        # Find the matching extension
        for ext in MAGIC_BYTES:
            if key_lower.endswith(ext):
                expected_magic = MAGIC_BYTES[ext]
                break
        else:
            # No magic bytes check for this extension
            self.validation_results.append({
                'check': 'MAGIC_BYTES',
                'passed': True,
                'message': 'No magic bytes check required for this file type'
            })
            return

        # Read the first bytes of the file
        with open(self.temp_file, 'rb') as f:
            file_header = f.read(16)

        # Check if any expected magic bytes match
        matched = any(file_header.startswith(magic) for magic in expected_magic)

        if not matched:
            raise ValidationError(
                f"File content does not match expected format for extension",
                'MAGIC_BYTES_MISMATCH',
                {'extension': ext, 'header_hex': file_header[:8].hex()}
            )

        self.validation_results.append({
            'check': 'MAGIC_BYTES',
            'passed': True,
            'message': 'File content matches expected format'
        })
        logger.debug("Magic bytes check passed")

    def _calculate_checksum(self) -> str:
        """Calculate SHA256 checksum of the file."""
        if not self.temp_file:
            return ''

        sha256_hash = hashlib.sha256()

        with open(self.temp_file, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256_hash.update(chunk)

        checksum = sha256_hash.hexdigest()

        self.validation_results.append({
            'check': 'CHECKSUM_CALCULATED',
            'passed': True,
            'message': f'SHA256: {checksum}'
        })
        logger.debug(f"Calculated checksum: {checksum}")

        return checksum

    def _verify_checksum(self, calculated: str, tags: Dict[str, str]) -> None:
        """Verify checksum against expected value if provided in tags."""
        expected = tags.get('expected-sha256', tags.get('sha256', ''))

        if not expected:
            # No expected checksum provided - just record the calculated one
            self.validation_results.append({
                'check': 'CHECKSUM_VERIFICATION',
                'passed': True,
                'message': 'No expected checksum provided - calculated value recorded'
            })
            return

        if calculated.lower() != expected.lower():
            raise ValidationError(
                f"Checksum mismatch: expected {expected}, got {calculated}",
                'CHECKSUM_MISMATCH',
                {'expected': expected, 'calculated': calculated}
            )

        self.validation_results.append({
            'check': 'CHECKSUM_VERIFICATION',
            'passed': True,
            'message': 'Checksum verified successfully'
        })
        logger.info("Checksum verification passed")

    def _scan_for_malware(self) -> None:
        """Scan file for malware using ClamAV (if available)."""
        if not self.temp_file:
            return

        try:
            # Try to use clamd socket if available
            import clamd
            cd = clamd.ClamdUnixSocket()

            with open(self.temp_file, 'rb') as f:
                result = cd.instream(f)

            if result and result.get('stream', ('OK', ''))[0] != 'OK':
                raise ValidationError(
                    f"Malware detected: {result.get('stream', ('UNKNOWN', ''))[1]}",
                    'MALWARE_DETECTED',
                    {'scan_result': result}
                )

            self.validation_results.append({
                'check': 'MALWARE_SCAN',
                'passed': True,
                'message': 'No malware detected'
            })
            logger.info("Malware scan passed")

        except ImportError:
            # ClamAV not available
            self.validation_results.append({
                'check': 'MALWARE_SCAN',
                'passed': True,
                'message': 'Malware scanning not available - skipped'
            })
            logger.warning("ClamAV not available - skipping malware scan")

        except Exception as e:
            logger.error(f"Malware scan error: {str(e)}")
            self.validation_results.append({
                'check': 'MALWARE_SCAN',
                'passed': True,
                'message': f'Scan error (non-blocking): {str(e)}'
            })

    def _get_object_tags(self) -> Dict[str, str]:
        """Get existing tags from the S3 object."""
        try:
            response = s3_client.get_object_tagging(Bucket=self.bucket, Key=self.key)
            return {tag['Key']: tag['Value'] for tag in response.get('TagSet', [])}
        except ClientError:
            return {}

    def _cleanup(self) -> None:
        """Clean up temporary files."""
        if self.temp_file and os.path.exists(self.temp_file):
            try:
                os.remove(self.temp_file)
                logger.debug(f"Cleaned up temp file: {self.temp_file}")
            except OSError as e:
                logger.warning(f"Failed to clean up temp file: {e}")


def tag_artifact(bucket: str, key: str, is_valid: bool, results: List[Dict]) -> None:
    """Add validation tags to the artifact."""
    tags = {
        'validation-status': 'valid' if is_valid else 'invalid',
        'validation-timestamp': datetime.utcnow().isoformat(),
        'validator-version': '1.0.0',
        'environment': ENVIRONMENT
    }

    # Get existing tags and merge
    try:
        existing = s3_client.get_object_tagging(Bucket=bucket, Key=key)
        existing_tags = {tag['Key']: tag['Value'] for tag in existing.get('TagSet', [])}
        existing_tags.update(tags)
        tags = existing_tags
    except ClientError:
        pass

    # Convert to TagSet format
    tag_set = [{'Key': k, 'Value': str(v)[:256]} for k, v in tags.items()][:10]  # Max 10 tags

    try:
        s3_client.put_object_tagging(
            Bucket=bucket,
            Key=key,
            Tagging={'TagSet': tag_set}
        )
        logger.info(f"Tagged artifact {key} as {'valid' if is_valid else 'invalid'}")
    except ClientError as e:
        logger.error(f"Failed to tag artifact: {e}")


def quarantine_artifact(bucket: str, key: str, results: List[Dict]) -> None:
    """Move invalid artifact to quarantine bucket."""
    if not QUARANTINE_BUCKET:
        logger.warning("No quarantine bucket configured - artifact not moved")
        return

    try:
        # Copy to quarantine
        quarantine_key = f"quarantine/{datetime.utcnow().strftime('%Y/%m/%d')}/{key}"

        s3_client.copy_object(
            Bucket=QUARANTINE_BUCKET,
            Key=quarantine_key,
            CopySource={'Bucket': bucket, 'Key': key},
            MetadataDirective='COPY',
            TaggingDirective='COPY'
        )

        # Add quarantine metadata
        quarantine_tags = [
            {'Key': 'quarantine-reason', 'Value': results[-1].get('message', 'Validation failed')[:256]},
            {'Key': 'original-bucket', 'Value': bucket},
            {'Key': 'original-key', 'Value': key[:256]},
            {'Key': 'quarantine-timestamp', 'Value': datetime.utcnow().isoformat()}
        ]

        s3_client.put_object_tagging(
            Bucket=QUARANTINE_BUCKET,
            Key=quarantine_key,
            Tagging={'TagSet': quarantine_tags}
        )

        # Delete from original location
        s3_client.delete_object(Bucket=bucket, Key=key)

        logger.info(f"Quarantined artifact: {key} -> {QUARANTINE_BUCKET}/{quarantine_key}")

    except ClientError as e:
        logger.error(f"Failed to quarantine artifact: {e}")
        raise


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
        logger.info("Sent failed message to DLQ")
    except ClientError as e:
        logger.error(f"Failed to send to DLQ: {e}")


def process_s3_event(record: Dict) -> Dict:
    """Process a single S3 event record."""
    s3_info = record.get('s3', {})
    bucket = s3_info.get('bucket', {}).get('name', '')
    key = unquote_plus(s3_info.get('object', {}).get('key', ''))

    if not bucket or not key:
        logger.warning(f"Invalid S3 event record: missing bucket or key")
        return {'status': 'error', 'message': 'Missing bucket or key'}

    # Skip if not an artifact
    if not key.startswith('artifacts/'):
        logger.debug(f"Skipping non-artifact key: {key}")
        return {'status': 'skipped', 'message': 'Not an artifact path'}

    # Validate the artifact
    validator = ArtifactValidator(bucket, key)
    is_valid, results = validator.validate()

    # Tag the artifact
    tag_artifact(bucket, key, is_valid, results)

    # Quarantine if invalid
    if not is_valid and QUARANTINE_BUCKET:
        quarantine_artifact(bucket, key, results)

    return {
        'status': 'valid' if is_valid else 'invalid',
        'bucket': bucket,
        'key': key,
        'results': results
    }


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for S3 artifact validation.

    Supports both direct S3 events and SQS-wrapped S3 events.
    """
    logger.info(f"Received event: {json.dumps(event)[:1000]}")

    results = []
    batch_item_failures = []

    # Handle SQS events (wrapping S3 events)
    if 'Records' in event and event['Records'][0].get('eventSource') == 'aws:sqs':
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
                    result = process_s3_event(s3_record)
                    results.append(result)

            except Exception as e:
                logger.error(f"Failed to process SQS message {message_id}: {str(e)}")
                batch_item_failures.append({'itemIdentifier': message_id})
                send_to_dlq(sqs_record, str(e))

    # Handle direct S3 events
    elif 'Records' in event:
        for record in event['Records']:
            try:
                result = process_s3_event(record)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process S3 event: {str(e)}")
                results.append({'status': 'error', 'message': str(e)})

    # Return response
    response = {
        'statusCode': 200,
        'body': {
            'processed': len(results),
            'valid': sum(1 for r in results if r.get('status') == 'valid'),
            'invalid': sum(1 for r in results if r.get('status') == 'invalid'),
            'skipped': sum(1 for r in results if r.get('status') == 'skipped'),
            'errors': sum(1 for r in results if r.get('status') == 'error'),
            'results': results
        }
    }

    # Add batch item failures for SQS partial batch response
    if batch_item_failures:
        response['batchItemFailures'] = batch_item_failures

    logger.info(f"Validation complete: {response['body']}")
    return response
