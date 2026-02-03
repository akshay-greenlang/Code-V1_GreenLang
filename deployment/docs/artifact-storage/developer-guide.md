# GreenLang Artifact Storage Developer Integration Guide

## Document Information

| Field | Value |
|-------|-------|
| Version | 1.0 |
| Last Updated | 2026-02-03 |
| Owner | Platform Engineering |
| Classification | Internal |

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Authentication and Configuration](#authentication-and-configuration)
3. [Python SDK Examples](#python-sdk-examples)
4. [JavaScript/TypeScript Examples](#javascripttypescript-examples)
5. [Presigned URL Generation](#presigned-url-generation)
6. [Multipart Upload Handling](#multipart-upload-handling)
7. [Error Handling Best Practices](#error-handling-best-practices)
8. [Testing with LocalStack/MinIO](#testing-with-localstackminio)
9. [Common Patterns](#common-patterns)

---

## Getting Started

### Prerequisites

Before integrating with GreenLang artifact storage, ensure you have:

1. **AWS SDK installed** (boto3 for Python, @aws-sdk for JavaScript)
2. **IAM role configured** with appropriate S3 permissions
3. **Service account** with IRSA (IAM Roles for Service Accounts) in Kubernetes
4. **Environment variables** configured for bucket names

### Quick Start

**Python**:
```bash
pip install boto3 aioboto3 botocore
```

**JavaScript/TypeScript**:
```bash
npm install @aws-sdk/client-s3 @aws-sdk/s3-request-presigner
```

### Environment Variables

Set the following environment variables in your application:

```bash
# Bucket configuration
export GREENLANG_DATA_LAKE_BUCKET="greenlang-prod-eu-west-1-data-lake-confidential"
export GREENLANG_REPORTS_BUCKET="greenlang-prod-eu-west-1-reports-confidential"
export GREENLANG_MODELS_BUCKET="greenlang-prod-eu-west-1-models-internal"
export GREENLANG_TEMP_BUCKET="greenlang-prod-eu-west-1-temp-internal"

# AWS configuration
export AWS_REGION="eu-west-1"
export AWS_DEFAULT_REGION="eu-west-1"

# Optional: Enable S3 Transfer Acceleration
export S3_USE_ACCELERATE="true"
```

---

## Authentication and Configuration

### IAM Roles for Service Accounts (IRSA)

In Kubernetes, your pod automatically receives AWS credentials through IRSA:

```yaml
# Pod annotation for IRSA
apiVersion: v1
kind: Pod
metadata:
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::123456789012:role/greenlang-cbam-storage-role
spec:
  serviceAccountName: cbam-service-account
```

### Boto3 Client Configuration

```python
import boto3
from botocore.config import Config

def create_s3_client():
    """Create configured S3 client with retry and timeout settings."""
    config = Config(
        region_name='eu-west-1',
        signature_version='v4',
        retries={
            'max_attempts': 3,
            'mode': 'adaptive'
        },
        connect_timeout=5,
        read_timeout=60,
        max_pool_connections=50
    )

    return boto3.client('s3', config=config)

# For Transfer Acceleration
def create_accelerated_s3_client():
    """Create S3 client with Transfer Acceleration enabled."""
    config = Config(
        region_name='eu-west-1',
        s3={'use_accelerate_endpoint': True}
    )

    return boto3.client('s3', config=config)
```

### Async Client Configuration (aioboto3)

```python
import aioboto3
from botocore.config import Config

async def create_async_s3_client():
    """Create async S3 client for high-concurrency workloads."""
    config = Config(
        region_name='eu-west-1',
        retries={'max_attempts': 3, 'mode': 'adaptive'},
        max_pool_connections=100
    )

    session = aioboto3.Session()
    async with session.client('s3', config=config) as client:
        yield client
```

---

## Python SDK Examples

### Basic Upload

```python
import boto3
import os
from datetime import datetime

def upload_file(file_path: str, bucket: str, key: str, metadata: dict = None) -> dict:
    """
    Upload a file to S3 with metadata.

    Args:
        file_path: Local file path
        bucket: S3 bucket name
        key: S3 object key
        metadata: Optional metadata dict

    Returns:
        S3 response dict
    """
    s3_client = boto3.client('s3')

    extra_args = {
        'Metadata': metadata or {},
        'ServerSideEncryption': 'aws:kms',
        'SSEKMSKeyId': os.environ.get('KMS_KEY_ID', 'alias/greenlang-s3-key')
    }

    # Add content type based on file extension
    content_type_map = {
        '.csv': 'text/csv',
        '.json': 'application/json',
        '.parquet': 'application/x-parquet',
        '.pdf': 'application/pdf',
        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    }

    ext = os.path.splitext(file_path)[1].lower()
    if ext in content_type_map:
        extra_args['ContentType'] = content_type_map[ext]

    response = s3_client.upload_file(
        file_path,
        bucket,
        key,
        ExtraArgs=extra_args
    )

    return response

# Example usage
upload_file(
    file_path='/tmp/emissions_report.pdf',
    bucket=os.environ['GREENLANG_REPORTS_BUCKET'],
    key=f'reports/cbam/2026/02/{datetime.now().isoformat()}_emissions.pdf',
    metadata={
        'tenant-id': 'tenant-123',
        'report-type': 'emissions',
        'generated-by': 'cbam-agent'
    }
)
```

### Basic Download

```python
import boto3
import os

def download_file(bucket: str, key: str, local_path: str) -> str:
    """
    Download a file from S3.

    Args:
        bucket: S3 bucket name
        key: S3 object key
        local_path: Local destination path

    Returns:
        Local file path
    """
    s3_client = boto3.client('s3')

    # Ensure directory exists
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    s3_client.download_file(bucket, key, local_path)

    return local_path

# Example usage
local_file = download_file(
    bucket=os.environ['GREENLANG_REPORTS_BUCKET'],
    key='reports/cbam/2026/02/emissions.pdf',
    local_path='/tmp/downloaded_report.pdf'
)
```

### Streaming Upload

```python
import boto3
from io import BytesIO

def upload_stream(data: bytes, bucket: str, key: str, content_type: str = None) -> dict:
    """
    Upload data directly from memory.

    Args:
        data: Bytes data to upload
        bucket: S3 bucket name
        key: S3 object key
        content_type: Optional content type

    Returns:
        S3 response dict
    """
    s3_client = boto3.client('s3')

    extra_args = {
        'ServerSideEncryption': 'aws:kms'
    }

    if content_type:
        extra_args['ContentType'] = content_type

    response = s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=data,
        **extra_args
    )

    return response

# Example usage - upload JSON
import json

data = {'emissions': 1234.56, 'unit': 'tCO2e'}
upload_stream(
    data=json.dumps(data).encode('utf-8'),
    bucket=os.environ['GREENLANG_DATA_LAKE_BUCKET'],
    key='bronze/cbam/2026/02/03/calculation_result.json',
    content_type='application/json'
)
```

### Streaming Download

```python
import boto3
from io import BytesIO

def download_stream(bucket: str, key: str) -> bytes:
    """
    Download file content directly to memory.

    Args:
        bucket: S3 bucket name
        key: S3 object key

    Returns:
        File content as bytes
    """
    s3_client = boto3.client('s3')

    response = s3_client.get_object(Bucket=bucket, Key=key)

    return response['Body'].read()

# Example usage
content = download_stream(
    bucket=os.environ['GREENLANG_DATA_LAKE_BUCKET'],
    key='bronze/cbam/2026/02/03/calculation_result.json'
)
data = json.loads(content.decode('utf-8'))
```

### Async Upload with aioboto3

```python
import aioboto3
import asyncio
from botocore.config import Config

async def async_upload_file(file_path: str, bucket: str, key: str) -> dict:
    """
    Asynchronously upload a file to S3.

    Args:
        file_path: Local file path
        bucket: S3 bucket name
        key: S3 object key

    Returns:
        S3 response dict
    """
    session = aioboto3.Session()
    config = Config(retries={'max_attempts': 3, 'mode': 'adaptive'})

    async with session.client('s3', config=config) as s3_client:
        with open(file_path, 'rb') as f:
            response = await s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=f.read(),
                ServerSideEncryption='aws:kms'
            )

    return response

# Batch upload multiple files
async def batch_upload(files: list[tuple[str, str, str]]) -> list[dict]:
    """
    Upload multiple files concurrently.

    Args:
        files: List of (file_path, bucket, key) tuples

    Returns:
        List of S3 responses
    """
    tasks = [async_upload_file(fp, b, k) for fp, b, k in files]
    return await asyncio.gather(*tasks)

# Example usage
asyncio.run(batch_upload([
    ('/tmp/file1.csv', 'my-bucket', 'path/file1.csv'),
    ('/tmp/file2.csv', 'my-bucket', 'path/file2.csv'),
    ('/tmp/file3.csv', 'my-bucket', 'path/file3.csv'),
]))
```

### Listing Objects with Pagination

```python
import boto3
from typing import Generator

def list_objects(bucket: str, prefix: str) -> Generator[dict, None, None]:
    """
    List all objects in a bucket with a given prefix.

    Args:
        bucket: S3 bucket name
        prefix: Object key prefix

    Yields:
        Object metadata dicts
    """
    s3_client = boto3.client('s3')
    paginator = s3_client.get_paginator('list_objects_v2')

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            yield obj

# Example usage
for obj in list_objects(
    bucket=os.environ['GREENLANG_REPORTS_BUCKET'],
    prefix='reports/cbam/2026/'
):
    print(f"Key: {obj['Key']}, Size: {obj['Size']}, Modified: {obj['LastModified']}")
```

---

## JavaScript/TypeScript Examples

### Basic Upload

```typescript
import { S3Client, PutObjectCommand } from '@aws-sdk/client-s3';
import { readFileSync } from 'fs';

const s3Client = new S3Client({
  region: 'eu-west-1',
  maxAttempts: 3
});

interface UploadOptions {
  bucket: string;
  key: string;
  filePath: string;
  contentType?: string;
  metadata?: Record<string, string>;
}

async function uploadFile(options: UploadOptions): Promise<void> {
  const fileContent = readFileSync(options.filePath);

  const command = new PutObjectCommand({
    Bucket: options.bucket,
    Key: options.key,
    Body: fileContent,
    ContentType: options.contentType,
    Metadata: options.metadata,
    ServerSideEncryption: 'aws:kms'
  });

  await s3Client.send(command);
}

// Example usage
await uploadFile({
  bucket: process.env.GREENLANG_REPORTS_BUCKET!,
  key: `reports/cbam/2026/02/${new Date().toISOString()}_emissions.pdf`,
  filePath: '/tmp/emissions_report.pdf',
  contentType: 'application/pdf',
  metadata: {
    'tenant-id': 'tenant-123',
    'report-type': 'emissions'
  }
});
```

### Basic Download

```typescript
import { S3Client, GetObjectCommand } from '@aws-sdk/client-s3';
import { writeFileSync } from 'fs';
import { Readable } from 'stream';

const s3Client = new S3Client({ region: 'eu-west-1' });

async function downloadFile(bucket: string, key: string, localPath: string): Promise<void> {
  const command = new GetObjectCommand({
    Bucket: bucket,
    Key: key
  });

  const response = await s3Client.send(command);

  // Convert stream to buffer
  const stream = response.Body as Readable;
  const chunks: Buffer[] = [];

  for await (const chunk of stream) {
    chunks.push(Buffer.from(chunk));
  }

  const buffer = Buffer.concat(chunks);
  writeFileSync(localPath, buffer);
}

// Example usage
await downloadFile(
  process.env.GREENLANG_REPORTS_BUCKET!,
  'reports/cbam/2026/02/emissions.pdf',
  '/tmp/downloaded_report.pdf'
);
```

### Streaming Upload

```typescript
import { S3Client, PutObjectCommand } from '@aws-sdk/client-s3';

const s3Client = new S3Client({ region: 'eu-west-1' });

async function uploadJSON(
  bucket: string,
  key: string,
  data: object
): Promise<void> {
  const command = new PutObjectCommand({
    Bucket: bucket,
    Key: key,
    Body: JSON.stringify(data),
    ContentType: 'application/json',
    ServerSideEncryption: 'aws:kms'
  });

  await s3Client.send(command);
}

// Example usage
await uploadJSON(
  process.env.GREENLANG_DATA_LAKE_BUCKET!,
  'bronze/cbam/2026/02/03/calculation_result.json',
  { emissions: 1234.56, unit: 'tCO2e' }
);
```

### Listing Objects

```typescript
import { S3Client, ListObjectsV2Command, ListObjectsV2CommandOutput } from '@aws-sdk/client-s3';

const s3Client = new S3Client({ region: 'eu-west-1' });

interface S3Object {
  key: string;
  size: number;
  lastModified: Date;
}

async function* listObjects(bucket: string, prefix: string): AsyncGenerator<S3Object> {
  let continuationToken: string | undefined;

  do {
    const command = new ListObjectsV2Command({
      Bucket: bucket,
      Prefix: prefix,
      ContinuationToken: continuationToken
    });

    const response: ListObjectsV2CommandOutput = await s3Client.send(command);

    for (const obj of response.Contents || []) {
      yield {
        key: obj.Key!,
        size: obj.Size!,
        lastModified: obj.LastModified!
      };
    }

    continuationToken = response.NextContinuationToken;
  } while (continuationToken);
}

// Example usage
for await (const obj of listObjects(
  process.env.GREENLANG_REPORTS_BUCKET!,
  'reports/cbam/2026/'
)) {
  console.log(`Key: ${obj.key}, Size: ${obj.size}`);
}
```

---

## Presigned URL Generation

### Python - Generate Upload URL

```python
import boto3
from datetime import datetime, timedelta

def generate_upload_url(
    bucket: str,
    key: str,
    content_type: str,
    expiration_seconds: int = 3600,
    max_size_mb: int = 100
) -> dict:
    """
    Generate a presigned URL for uploading a file.

    Args:
        bucket: S3 bucket name
        key: S3 object key
        content_type: Expected content type
        expiration_seconds: URL expiration time in seconds
        max_size_mb: Maximum file size in MB

    Returns:
        Dict with presigned URL and fields
    """
    s3_client = boto3.client('s3')

    # Generate presigned POST for more control
    conditions = [
        ['content-length-range', 1, max_size_mb * 1024 * 1024],
        {'Content-Type': content_type},
        {'x-amz-server-side-encryption': 'aws:kms'}
    ]

    fields = {
        'Content-Type': content_type,
        'x-amz-server-side-encryption': 'aws:kms'
    }

    presigned_post = s3_client.generate_presigned_post(
        Bucket=bucket,
        Key=key,
        Fields=fields,
        Conditions=conditions,
        ExpiresIn=expiration_seconds
    )

    return presigned_post

# Example usage
upload_url = generate_upload_url(
    bucket=os.environ['GREENLANG_TEMP_BUCKET'],
    key=f'uploads/{datetime.now().strftime("%Y/%m/%d")}/user-upload.csv',
    content_type='text/csv',
    max_size_mb=50
)

print(f"Upload URL: {upload_url['url']}")
print(f"Fields: {upload_url['fields']}")
```

### Python - Generate Download URL

```python
import boto3

def generate_download_url(
    bucket: str,
    key: str,
    expiration_seconds: int = 3600,
    filename: str = None
) -> str:
    """
    Generate a presigned URL for downloading a file.

    Args:
        bucket: S3 bucket name
        key: S3 object key
        expiration_seconds: URL expiration time in seconds
        filename: Optional filename for Content-Disposition

    Returns:
        Presigned download URL
    """
    s3_client = boto3.client('s3')

    params = {
        'Bucket': bucket,
        'Key': key
    }

    if filename:
        params['ResponseContentDisposition'] = f'attachment; filename="{filename}"'

    url = s3_client.generate_presigned_url(
        'get_object',
        Params=params,
        ExpiresIn=expiration_seconds
    )

    return url

# Example usage
download_url = generate_download_url(
    bucket=os.environ['GREENLANG_REPORTS_BUCKET'],
    key='reports/cbam/2026/02/emissions.pdf',
    filename='CBAM_Emissions_Report_2026_02.pdf'
)
```

### TypeScript - Generate Upload URL

```typescript
import { S3Client, PutObjectCommand } from '@aws-sdk/client-s3';
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';

const s3Client = new S3Client({ region: 'eu-west-1' });

interface PresignedUrlOptions {
  bucket: string;
  key: string;
  contentType: string;
  expirationSeconds?: number;
}

async function generateUploadUrl(options: PresignedUrlOptions): Promise<string> {
  const command = new PutObjectCommand({
    Bucket: options.bucket,
    Key: options.key,
    ContentType: options.contentType,
    ServerSideEncryption: 'aws:kms'
  });

  const url = await getSignedUrl(s3Client, command, {
    expiresIn: options.expirationSeconds || 3600
  });

  return url;
}

// Example usage
const uploadUrl = await generateUploadUrl({
  bucket: process.env.GREENLANG_TEMP_BUCKET!,
  key: `uploads/${new Date().toISOString().split('T')[0]}/user-upload.csv`,
  contentType: 'text/csv'
});
```

---

## Multipart Upload Handling

### Python - Multipart Upload

```python
import boto3
from boto3.s3.transfer import TransferConfig
import os

def upload_large_file(
    file_path: str,
    bucket: str,
    key: str,
    part_size_mb: int = 10,
    max_concurrency: int = 10
) -> dict:
    """
    Upload a large file using multipart upload.

    Args:
        file_path: Local file path
        bucket: S3 bucket name
        key: S3 object key
        part_size_mb: Size of each part in MB
        max_concurrency: Maximum concurrent uploads

    Returns:
        Upload result
    """
    s3_client = boto3.client('s3')

    config = TransferConfig(
        multipart_threshold=part_size_mb * 1024 * 1024,
        multipart_chunksize=part_size_mb * 1024 * 1024,
        max_concurrency=max_concurrency,
        use_threads=True
    )

    # Progress callback
    file_size = os.path.getsize(file_path)
    uploaded = 0

    def progress_callback(bytes_transferred):
        nonlocal uploaded
        uploaded += bytes_transferred
        percentage = (uploaded / file_size) * 100
        print(f'Upload progress: {percentage:.1f}%')

    s3_client.upload_file(
        file_path,
        bucket,
        key,
        Config=config,
        Callback=progress_callback,
        ExtraArgs={
            'ServerSideEncryption': 'aws:kms',
            'Metadata': {'upload-type': 'multipart'}
        }
    )

    return {'status': 'completed', 'key': key}

# Example usage
upload_large_file(
    file_path='/data/large_dataset.parquet',
    bucket=os.environ['GREENLANG_DATA_LAKE_BUCKET'],
    key='bronze/cbam/2026/02/03/large_dataset.parquet',
    part_size_mb=50,
    max_concurrency=20
)
```

### Python - Manual Multipart Upload

```python
import boto3
from typing import BinaryIO

def manual_multipart_upload(
    file_obj: BinaryIO,
    bucket: str,
    key: str,
    part_size_bytes: int = 10 * 1024 * 1024
) -> dict:
    """
    Manually manage multipart upload for streaming data.

    Args:
        file_obj: File-like object
        bucket: S3 bucket name
        key: S3 object key
        part_size_bytes: Size of each part in bytes

    Returns:
        Upload completion result
    """
    s3_client = boto3.client('s3')

    # Initiate multipart upload
    response = s3_client.create_multipart_upload(
        Bucket=bucket,
        Key=key,
        ServerSideEncryption='aws:kms'
    )
    upload_id = response['UploadId']

    parts = []
    part_number = 1

    try:
        while True:
            data = file_obj.read(part_size_bytes)
            if not data:
                break

            # Upload part
            part_response = s3_client.upload_part(
                Bucket=bucket,
                Key=key,
                UploadId=upload_id,
                PartNumber=part_number,
                Body=data
            )

            parts.append({
                'ETag': part_response['ETag'],
                'PartNumber': part_number
            })

            part_number += 1

        # Complete multipart upload
        result = s3_client.complete_multipart_upload(
            Bucket=bucket,
            Key=key,
            UploadId=upload_id,
            MultipartUpload={'Parts': parts}
        )

        return result

    except Exception as e:
        # Abort on failure
        s3_client.abort_multipart_upload(
            Bucket=bucket,
            Key=key,
            UploadId=upload_id
        )
        raise e
```

---

## Error Handling Best Practices

### Python - Comprehensive Error Handling

```python
import boto3
from botocore.exceptions import ClientError, ConnectionError, HTTPClientError
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class S3Error(Exception):
    """Base exception for S3 operations."""
    pass

class S3NotFoundError(S3Error):
    """Object not found in S3."""
    pass

class S3AccessDeniedError(S3Error):
    """Access denied to S3 resource."""
    pass

class S3ThrottlingError(S3Error):
    """S3 request throttled."""
    pass

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True
)
def safe_s3_operation(func, *args, **kwargs):
    """
    Execute S3 operation with comprehensive error handling.

    Args:
        func: S3 client method to call
        *args, **kwargs: Arguments for the method

    Returns:
        Operation result

    Raises:
        S3NotFoundError: Object not found
        S3AccessDeniedError: Access denied
        S3ThrottlingError: Request throttled
        S3Error: Other S3 errors
    """
    try:
        return func(*args, **kwargs)

    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']

        if error_code == '404' or error_code == 'NoSuchKey':
            logger.warning(f'S3 object not found: {error_message}')
            raise S3NotFoundError(error_message) from e

        elif error_code == '403' or error_code == 'AccessDenied':
            logger.error(f'S3 access denied: {error_message}')
            raise S3AccessDeniedError(error_message) from e

        elif error_code == 'SlowDown' or error_code == '503':
            logger.warning(f'S3 throttling: {error_message}')
            raise S3ThrottlingError(error_message) from e

        else:
            logger.error(f'S3 client error: {error_code} - {error_message}')
            raise S3Error(error_message) from e

    except ConnectionError as e:
        logger.error(f'S3 connection error: {e}')
        raise S3Error(f'Connection failed: {e}') from e

    except HTTPClientError as e:
        logger.error(f'S3 HTTP error: {e}')
        raise S3Error(f'HTTP error: {e}') from e

# Example usage
def get_object_safe(bucket: str, key: str) -> bytes:
    """Safely get an object from S3."""
    s3_client = boto3.client('s3')

    response = safe_s3_operation(
        s3_client.get_object,
        Bucket=bucket,
        Key=key
    )

    return response['Body'].read()

# Usage with error handling
try:
    content = get_object_safe('my-bucket', 'my-key')
except S3NotFoundError:
    print('File not found, using default')
    content = b'{}'
except S3AccessDeniedError:
    print('Access denied, check permissions')
    raise
```

### TypeScript - Error Handling

```typescript
import { S3Client, GetObjectCommand, S3ServiceException } from '@aws-sdk/client-s3';

class S3Error extends Error {
  constructor(message: string, public readonly code?: string) {
    super(message);
    this.name = 'S3Error';
  }
}

class S3NotFoundError extends S3Error {
  constructor(message: string) {
    super(message, '404');
    this.name = 'S3NotFoundError';
  }
}

class S3AccessDeniedError extends S3Error {
  constructor(message: string) {
    super(message, '403');
    this.name = 'S3AccessDeniedError';
  }
}

async function safeGetObject(bucket: string, key: string): Promise<Buffer> {
  const s3Client = new S3Client({ region: 'eu-west-1' });

  try {
    const command = new GetObjectCommand({ Bucket: bucket, Key: key });
    const response = await s3Client.send(command);

    const chunks: Buffer[] = [];
    for await (const chunk of response.Body as any) {
      chunks.push(Buffer.from(chunk));
    }

    return Buffer.concat(chunks);

  } catch (error) {
    if (error instanceof S3ServiceException) {
      if (error.name === 'NoSuchKey' || error.$metadata.httpStatusCode === 404) {
        throw new S3NotFoundError(`Object not found: ${key}`);
      }

      if (error.name === 'AccessDenied' || error.$metadata.httpStatusCode === 403) {
        throw new S3AccessDeniedError(`Access denied: ${key}`);
      }
    }

    throw new S3Error(`S3 operation failed: ${error}`);
  }
}
```

---

## Testing with LocalStack/MinIO

### LocalStack Setup

```yaml
# docker-compose.yml
version: '3.8'

services:
  localstack:
    image: localstack/localstack:latest
    ports:
      - "4566:4566"
    environment:
      - SERVICES=s3
      - DEBUG=1
      - DATA_DIR=/var/lib/localstack/data
    volumes:
      - localstack_data:/var/lib/localstack

volumes:
  localstack_data:
```

### Python - LocalStack Configuration

```python
import boto3
import os

def create_test_s3_client():
    """Create S3 client configured for LocalStack."""
    if os.environ.get('USE_LOCALSTACK', 'false').lower() == 'true':
        return boto3.client(
            's3',
            endpoint_url='http://localhost:4566',
            aws_access_key_id='test',
            aws_secret_access_key='test',
            region_name='us-east-1'
        )
    else:
        return boto3.client('s3')

# Create test bucket
def setup_test_bucket(bucket_name: str):
    """Create bucket in LocalStack for testing."""
    s3_client = create_test_s3_client()

    try:
        s3_client.create_bucket(Bucket=bucket_name)
    except s3_client.exceptions.BucketAlreadyExists:
        pass

# Test fixture example
import pytest

@pytest.fixture
def s3_client():
    """Pytest fixture for S3 client."""
    os.environ['USE_LOCALSTACK'] = 'true'
    client = create_test_s3_client()

    # Setup test bucket
    setup_test_bucket('test-bucket')

    yield client

    # Cleanup
    paginator = client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket='test-bucket'):
        for obj in page.get('Contents', []):
            client.delete_object(Bucket='test-bucket', Key=obj['Key'])

def test_upload_download(s3_client):
    """Test upload and download operations."""
    test_data = b'Hello, GreenLang!'

    # Upload
    s3_client.put_object(
        Bucket='test-bucket',
        Key='test/hello.txt',
        Body=test_data
    )

    # Download
    response = s3_client.get_object(
        Bucket='test-bucket',
        Key='test/hello.txt'
    )

    assert response['Body'].read() == test_data
```

### MinIO Setup

```yaml
# docker-compose.yml
version: '3.8'

services:
  minio:
    image: minio/minio:latest
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin
    command: server /data --console-address ":9001"
    volumes:
      - minio_data:/data

volumes:
  minio_data:
```

### Python - MinIO Configuration

```python
import boto3

def create_minio_client():
    """Create S3 client configured for MinIO."""
    return boto3.client(
        's3',
        endpoint_url='http://localhost:9000',
        aws_access_key_id='minioadmin',
        aws_secret_access_key='minioadmin',
        region_name='us-east-1'
    )
```

---

## Common Patterns

### Pattern 1: Uploading Calculation Results

```python
import json
from datetime import datetime
from typing import Any

async def upload_calculation_result(
    tenant_id: str,
    calculation_id: str,
    result: dict[str, Any],
    application: str = 'cbam'
) -> str:
    """
    Upload calculation result to the data lake.

    Args:
        tenant_id: Tenant identifier
        calculation_id: Calculation job ID
        result: Calculation result dict
        application: Application name

    Returns:
        S3 object key
    """
    session = aioboto3.Session()

    now = datetime.utcnow()
    key = f'silver/{application}/calculations/{now.year}/{now.month:02d}/{now.day:02d}/{tenant_id}/{calculation_id}.json'

    metadata = {
        'tenant-id': tenant_id,
        'calculation-id': calculation_id,
        'application': application,
        'calculated-at': now.isoformat()
    }

    async with session.client('s3') as s3_client:
        await s3_client.put_object(
            Bucket=os.environ['GREENLANG_DATA_LAKE_BUCKET'],
            Key=key,
            Body=json.dumps(result, default=str).encode('utf-8'),
            ContentType='application/json',
            Metadata=metadata,
            ServerSideEncryption='aws:kms'
        )

    return key
```

### Pattern 2: Storing Reports

```python
from datetime import datetime
import hashlib

async def store_report(
    tenant_id: str,
    report_type: str,
    report_data: bytes,
    format: str = 'pdf',
    filename: str = None
) -> dict:
    """
    Store a generated report.

    Args:
        tenant_id: Tenant identifier
        report_type: Type of report (emissions, compliance, audit)
        report_data: Report file content
        format: File format (pdf, xlsx, csv)
        filename: Optional original filename

    Returns:
        Dict with storage details
    """
    session = aioboto3.Session()

    now = datetime.utcnow()
    content_hash = hashlib.sha256(report_data).hexdigest()[:8]

    key = f'reports/{report_type}/{now.year}/{now.month:02d}/{tenant_id}/{now.isoformat()}_{content_hash}.{format}'

    content_types = {
        'pdf': 'application/pdf',
        'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'csv': 'text/csv',
        'json': 'application/json'
    }

    metadata = {
        'tenant-id': tenant_id,
        'report-type': report_type,
        'generated-at': now.isoformat(),
        'content-hash': content_hash
    }

    if filename:
        metadata['original-filename'] = filename

    async with session.client('s3') as s3_client:
        await s3_client.put_object(
            Bucket=os.environ['GREENLANG_REPORTS_BUCKET'],
            Key=key,
            Body=report_data,
            ContentType=content_types.get(format, 'application/octet-stream'),
            Metadata=metadata,
            ServerSideEncryption='aws:kms'
        )

    return {
        'bucket': os.environ['GREENLANG_REPORTS_BUCKET'],
        'key': key,
        'content_hash': content_hash,
        'size': len(report_data)
    }
```

### Pattern 3: Managing ML Models

```python
import pickle
from datetime import datetime
from typing import Any

async def upload_ml_model(
    model_name: str,
    model_version: str,
    model: Any,
    metrics: dict,
    application: str = 'common'
) -> str:
    """
    Upload a trained ML model.

    Args:
        model_name: Name of the model
        model_version: Semantic version
        model: Model object (must be picklable)
        metrics: Model performance metrics
        application: Application name

    Returns:
        S3 object key
    """
    session = aioboto3.Session()

    # Serialize model
    model_data = pickle.dumps(model)

    now = datetime.utcnow()
    key = f'models/{application}/{model_name}/{model_version}/model.pkl'
    metrics_key = f'models/{application}/{model_name}/{model_version}/metrics.json'

    metadata = {
        'model-name': model_name,
        'model-version': model_version,
        'trained-at': now.isoformat(),
        'framework': type(model).__module__.split('.')[0]
    }

    async with session.client('s3') as s3_client:
        # Upload model
        await s3_client.put_object(
            Bucket=os.environ['GREENLANG_MODELS_BUCKET'],
            Key=key,
            Body=model_data,
            Metadata=metadata,
            ServerSideEncryption='aws:kms'
        )

        # Upload metrics
        await s3_client.put_object(
            Bucket=os.environ['GREENLANG_MODELS_BUCKET'],
            Key=metrics_key,
            Body=json.dumps(metrics).encode('utf-8'),
            ContentType='application/json'
        )

    return key

async def download_ml_model(
    model_name: str,
    model_version: str,
    application: str = 'common'
) -> Any:
    """
    Download a trained ML model.

    Args:
        model_name: Name of the model
        model_version: Semantic version
        application: Application name

    Returns:
        Deserialized model object
    """
    session = aioboto3.Session()

    key = f'models/{application}/{model_name}/{model_version}/model.pkl'

    async with session.client('s3') as s3_client:
        response = await s3_client.get_object(
            Bucket=os.environ['GREENLANG_MODELS_BUCKET'],
            Key=key
        )

        model_data = await response['Body'].read()

    return pickle.loads(model_data)
```

### Pattern 4: Caching Emission Factors

```python
import json
from datetime import datetime, timedelta
from typing import Optional

CACHE_TTL_HOURS = 24

async def cache_emission_factors(
    source: str,
    factors: dict,
    version: str = None
) -> str:
    """
    Cache emission factors from external sources.

    Args:
        source: Data source identifier (e.g., 'ecoinvent', 'ipcc')
        factors: Emission factor data
        version: Optional version identifier

    Returns:
        S3 object key
    """
    session = aioboto3.Session()

    now = datetime.utcnow()
    version = version or now.strftime('%Y%m%d%H%M%S')

    key = f'cache/emission-factors/{source}/{version}/factors.json'

    metadata = {
        'source': source,
        'version': version,
        'cached-at': now.isoformat(),
        'expires-at': (now + timedelta(hours=CACHE_TTL_HOURS)).isoformat()
    }

    async with session.client('s3') as s3_client:
        await s3_client.put_object(
            Bucket=os.environ['GREENLANG_CACHE_BUCKET'],
            Key=key,
            Body=json.dumps(factors).encode('utf-8'),
            ContentType='application/json',
            Metadata=metadata,
            CacheControl=f'max-age={CACHE_TTL_HOURS * 3600}'
        )

    return key

async def get_cached_emission_factors(
    source: str,
    version: str = 'latest'
) -> Optional[dict]:
    """
    Get cached emission factors.

    Args:
        source: Data source identifier
        version: Version or 'latest'

    Returns:
        Emission factor data or None if not cached
    """
    session = aioboto3.Session()

    if version == 'latest':
        # List versions and get most recent
        async with session.client('s3') as s3_client:
            paginator = s3_client.get_paginator('list_objects_v2')

            versions = []
            async for page in paginator.paginate(
                Bucket=os.environ['GREENLANG_CACHE_BUCKET'],
                Prefix=f'cache/emission-factors/{source}/',
                Delimiter='/'
            ):
                for prefix in page.get('CommonPrefixes', []):
                    versions.append(prefix['Prefix'].rstrip('/').split('/')[-1])

            if not versions:
                return None

            version = sorted(versions)[-1]

    key = f'cache/emission-factors/{source}/{version}/factors.json'

    try:
        async with session.client('s3') as s3_client:
            response = await s3_client.get_object(
                Bucket=os.environ['GREENLANG_CACHE_BUCKET'],
                Key=key
            )

            # Check expiration
            expires_at = response['Metadata'].get('expires-at')
            if expires_at and datetime.fromisoformat(expires_at) < datetime.utcnow():
                return None

            data = await response['Body'].read()
            return json.loads(data.decode('utf-8'))

    except s3_client.exceptions.NoSuchKey:
        return None
```

---

## Related Documents

- [Architecture Guide](architecture-guide.md)
- [Operations Runbook](operations-runbook.md)
- [Naming Conventions](naming-conventions.md)
- [Access Procedures](access-procedures.md)

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-03 | Platform Engineering | Initial release |
