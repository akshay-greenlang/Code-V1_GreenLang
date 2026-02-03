# GreenLang Artifact Storage Naming Conventions

## Document Information

| Field | Value |
|-------|-------|
| Version | 1.0 |
| Last Updated | 2026-02-03 |
| Owner | Platform Engineering |
| Classification | Internal |

---

## Table of Contents

1. [Overview](#overview)
2. [Bucket Naming Pattern](#bucket-naming-pattern)
3. [Object Key Structure](#object-key-structure)
4. [Prefix Conventions](#prefix-conventions)
5. [Tag Requirements](#tag-requirements)
6. [Metadata Standards](#metadata-standards)

---

## Overview

### Purpose

This document defines the naming conventions for GreenLang's artifact storage infrastructure. Consistent naming ensures:

- Easy identification of resources
- Automated policy application
- Cost allocation by tag
- Security boundary enforcement
- Operational efficiency

### Naming Principles

1. **Predictable**: Names follow a consistent pattern
2. **Descriptive**: Names convey purpose and scope
3. **Sortable**: Names allow logical grouping
4. **Unique**: Names prevent collisions
5. **Compliant**: Names meet AWS requirements

---

## Bucket Naming Pattern

### Standard Format

```
greenlang-{environment}-{region}-{purpose}-{classification}
```

### Components

| Component | Description | Valid Values | Required |
|-----------|-------------|--------------|----------|
| greenlang | Organization prefix | `greenlang` | Yes |
| environment | Deployment environment | `dev`, `staging`, `prod` | Yes |
| region | AWS region code | `eu-west-1`, `eu-central-1`, `us-east-1` | Yes |
| purpose | Functional purpose | See Purpose Values | Yes |
| classification | Data classification | `public`, `internal`, `confidential` | Yes |

### Purpose Values

| Purpose | Description | Example |
|---------|-------------|---------|
| `data-lake` | Data Lake storage | `greenlang-prod-eu-west-1-data-lake-confidential` |
| `reports` | Generated reports | `greenlang-prod-eu-west-1-reports-confidential` |
| `models` | ML models | `greenlang-prod-eu-west-1-models-internal` |
| `audit-logs` | Audit trail logs | `greenlang-prod-eu-west-1-audit-logs-confidential` |
| `temp` | Temporary storage | `greenlang-prod-eu-west-1-temp-internal` |
| `cache` | Cache storage | `greenlang-prod-eu-west-1-cache-internal` |
| `static` | Static assets | `greenlang-prod-eu-west-1-static-public` |
| `backup` | Backup storage | `greenlang-prod-eu-west-1-backup-confidential` |

### Replica Bucket Naming

For cross-region replication buckets, append `-replica`:

```
greenlang-{environment}-{dr-region}-{purpose}-{classification}-replica
```

**Example**: `greenlang-prod-eu-central-1-data-lake-confidential-replica`

### Bucket Name Examples

| Environment | Region | Purpose | Classification | Bucket Name |
|-------------|--------|---------|----------------|-------------|
| Production | EU West 1 | Data Lake | Confidential | `greenlang-prod-eu-west-1-data-lake-confidential` |
| Production | EU Central 1 | Data Lake Replica | Confidential | `greenlang-prod-eu-central-1-data-lake-confidential-replica` |
| Production | EU West 1 | Reports | Confidential | `greenlang-prod-eu-west-1-reports-confidential` |
| Production | EU West 1 | ML Models | Internal | `greenlang-prod-eu-west-1-models-internal` |
| Staging | EU West 1 | Data Lake | Internal | `greenlang-staging-eu-west-1-data-lake-internal` |
| Development | EU West 1 | Data Lake | Internal | `greenlang-dev-eu-west-1-data-lake-internal` |

### Validation Rules

```python
import re

BUCKET_NAME_PATTERN = re.compile(
    r'^greenlang-(dev|staging|prod)-'
    r'(eu-west-1|eu-central-1|us-east-1|us-west-2)-'
    r'(data-lake|reports|models|audit-logs|temp|cache|static|backup)-'
    r'(public|internal|confidential)(-replica)?$'
)

def validate_bucket_name(name: str) -> bool:
    """Validate bucket name follows GreenLang conventions."""
    if not BUCKET_NAME_PATTERN.match(name):
        return False

    # AWS constraints
    if len(name) < 3 or len(name) > 63:
        return False

    if not name[0].isalnum():
        return False

    return True
```

---

## Object Key Structure

### Standard Key Format

```
{zone}/{application}/{year}/{month}/{day}/{tenant_id}/{object_id}.{extension}
```

### Key Components

| Component | Description | Format | Required |
|-----------|-------------|--------|----------|
| zone | Data Lake zone | `landing`, `bronze`, `silver`, `gold` | Yes |
| application | Application identifier | `cbam`, `csrd`, `sf6`, `common` | Yes |
| year | Year (4 digits) | `YYYY` | Yes |
| month | Month (2 digits) | `MM` | Yes |
| day | Day (2 digits) | `DD` | Yes |
| tenant_id | Tenant identifier | `tenant-{uuid}` | Yes |
| object_id | Unique object identifier | UUID or descriptive name | Yes |
| extension | File extension | `parquet`, `json`, `csv`, `pdf` | Yes |

### Key Examples by Zone

#### Landing Zone

```
landing/{tenant_id}/{upload_id}/{original_filename}
```

**Examples**:
```
landing/tenant-abc123/upload-xyz789/shipments_2026_02.csv
landing/tenant-abc123/upload-xyz789/emissions_data.xlsx
```

#### Bronze Zone

```
bronze/{application}/{year}/{month}/{day}/{tenant_id}/{object_id}.parquet
```

**Examples**:
```
bronze/cbam/2026/02/03/tenant-abc123/shipments_raw_001.parquet
bronze/csrd/2026/02/03/tenant-abc123/energy_consumption_raw.parquet
```

#### Silver Zone

```
silver/{application}/{domain}/{tenant_id}/{object_id}.parquet
```

**Examples**:
```
silver/cbam/emissions/tenant-abc123/calculated_emissions_2026_q1.parquet
silver/csrd/sustainability/tenant-abc123/scope1_emissions.parquet
```

#### Gold Zone

```
gold/{domain}/{metric}/{aggregation_level}.parquet
```

**Examples**:
```
gold/sustainability/emissions_by_scope/tenant_monthly.parquet
gold/compliance/cbam_summary/eu_quarterly.parquet
```

### Reports Key Structure

```
reports/{report_type}/{year}/{month}/{tenant_id}/{report_id}_{timestamp}.{format}
```

**Examples**:
```
reports/cbam/2026/02/tenant-abc123/declaration_20260203T120000Z.pdf
reports/csrd/2026/02/tenant-abc123/sustainability_report_20260203T120000Z.xlsx
reports/audit/2026/02/tenant-abc123/audit_trail_20260203T120000Z.json
```

### Models Key Structure

```
models/{application}/{model_name}/{version}/model.{format}
```

**Examples**:
```
models/cbam/emission_predictor/v1.2.3/model.pkl
models/common/data_classifier/v2.0.0/model.onnx
```

---

## Prefix Conventions

### Standard Prefixes

| Prefix | Purpose | Lifecycle | Access |
|--------|---------|-----------|--------|
| `landing/` | Raw uploads | 24 hours | Write: Upload service, Read: Validation |
| `bronze/` | Validated raw data | 7 years | Write: ETL, Read: Analytics |
| `silver/` | Cleaned/enriched data | 7 years | Write: ETL, Read: Applications |
| `gold/` | Aggregated analytics | 7 years | Write: Analytics, Read: Dashboards |
| `reports/` | Generated reports | 10 years | Write: Report service, Read: Users |
| `models/` | ML models | 2 years | Write: ML pipeline, Read: Inference |
| `cache/` | Cached data | 30 days | Write/Read: Applications |
| `shared/` | Cross-application data | Varies | Read: Multiple applications |
| `archive/` | Archived data | 10 years | Read: Compliance |

### Application Prefixes

Within zones, use application-specific prefixes:

| Application | Prefix | Description |
|-------------|--------|-------------|
| CBAM | `cbam/` | Carbon Border Adjustment Mechanism |
| CSRD | `csrd/` | Corporate Sustainability Reporting |
| SF6 | `sf6/` | SF6 Gas Management |
| EU ETS | `euets/` | EU Emissions Trading System |
| Common | `common/` | Shared resources |

### Prefix Hierarchy Example

```
greenlang-prod-eu-west-1-data-lake-confidential/
|
+-- landing/
|   +-- tenant-abc123/
|       +-- upload-xyz789/
|           +-- file1.csv
|           +-- file2.xlsx
|
+-- bronze/
|   +-- cbam/
|   |   +-- 2026/
|   |       +-- 02/
|   |           +-- 03/
|   |               +-- tenant-abc123/
|   |                   +-- shipments.parquet
|   +-- csrd/
|       +-- 2026/
|           +-- ...
|
+-- silver/
|   +-- cbam/
|   |   +-- emissions/
|   |   |   +-- tenant-abc123/
|   |   |       +-- calculated.parquet
|   |   +-- products/
|   |       +-- ...
|   +-- csrd/
|       +-- ...
|
+-- gold/
|   +-- sustainability/
|   |   +-- emissions_by_scope/
|   |       +-- monthly.parquet
|   +-- compliance/
|       +-- ...
|
+-- shared/
    +-- emission-factors/
    |   +-- v2026.01/
    |       +-- factors.parquet
    +-- reference-data/
        +-- ...
```

---

## Tag Requirements

### Mandatory Tags

All objects must have the following tags:

| Tag Key | Description | Example Values | Required For |
|---------|-------------|----------------|--------------|
| `greenlang:tenant-id` | Tenant identifier | `tenant-abc123` | All tenant data |
| `greenlang:application` | Application owner | `cbam`, `csrd`, `sf6` | All objects |
| `greenlang:environment` | Environment | `dev`, `staging`, `prod` | All objects |
| `greenlang:data-classification` | Classification level | `public`, `internal`, `confidential` | All objects |
| `greenlang:created-at` | Creation timestamp | `2026-02-03T12:00:00Z` | All objects |

### Optional Tags

| Tag Key | Description | Example Values | Use Case |
|---------|-------------|----------------|----------|
| `greenlang:cost-center` | Finance allocation | `eng-100`, `sales-200` | Cost tracking |
| `greenlang:retention-until` | Retention date | `2033-02-03` | Compliance |
| `greenlang:contains-pii` | Contains personal data | `true`, `false` | GDPR compliance |
| `greenlang:regulation` | Regulatory requirement | `csrd`, `cbam`, `sox` | Compliance |
| `greenlang:replication` | Replication enabled | `enabled`, `disabled` | DR planning |

### Tagging Implementation

```python
def apply_standard_tags(
    bucket: str,
    key: str,
    tenant_id: str,
    application: str,
    classification: str = 'confidential',
    extra_tags: dict = None
) -> None:
    """Apply standard tags to an S3 object."""

    tags = {
        'greenlang:tenant-id': tenant_id,
        'greenlang:application': application,
        'greenlang:environment': os.environ.get('ENVIRONMENT', 'prod'),
        'greenlang:data-classification': classification,
        'greenlang:created-at': datetime.utcnow().isoformat() + 'Z'
    }

    if extra_tags:
        tags.update(extra_tags)

    tag_set = [{'Key': k, 'Value': v} for k, v in tags.items()]

    s3_client.put_object_tagging(
        Bucket=bucket,
        Key=key,
        Tagging={'TagSet': tag_set}
    )
```

### Tag Validation

```python
REQUIRED_TAGS = [
    'greenlang:tenant-id',
    'greenlang:application',
    'greenlang:environment',
    'greenlang:data-classification',
    'greenlang:created-at'
]

VALID_APPLICATIONS = ['cbam', 'csrd', 'sf6', 'euets', 'common', 'platform']
VALID_CLASSIFICATIONS = ['public', 'internal', 'confidential']
VALID_ENVIRONMENTS = ['dev', 'staging', 'prod']

def validate_object_tags(bucket: str, key: str) -> list[str]:
    """Validate object tags meet requirements."""

    response = s3_client.get_object_tagging(Bucket=bucket, Key=key)
    tags = {t['Key']: t['Value'] for t in response['TagSet']}

    errors = []

    # Check required tags
    for required in REQUIRED_TAGS:
        if required not in tags:
            errors.append(f"Missing required tag: {required}")

    # Validate values
    if tags.get('greenlang:application') not in VALID_APPLICATIONS:
        errors.append(f"Invalid application: {tags.get('greenlang:application')}")

    if tags.get('greenlang:data-classification') not in VALID_CLASSIFICATIONS:
        errors.append(f"Invalid classification: {tags.get('greenlang:data-classification')}")

    if tags.get('greenlang:environment') not in VALID_ENVIRONMENTS:
        errors.append(f"Invalid environment: {tags.get('greenlang:environment')}")

    return errors
```

---

## Metadata Standards

### Standard Metadata Fields

S3 user metadata (x-amz-meta-*) fields:

| Metadata Key | Description | Format | Required |
|--------------|-------------|--------|----------|
| `x-amz-meta-content-hash` | SHA256 of content | Hex string | Recommended |
| `x-amz-meta-original-filename` | Original upload filename | String | For uploads |
| `x-amz-meta-upload-id` | Upload job identifier | UUID | For uploads |
| `x-amz-meta-processing-status` | Processing state | `pending`, `complete`, `failed` | For pipeline data |
| `x-amz-meta-schema-version` | Data schema version | Semver | For structured data |
| `x-amz-meta-record-count` | Number of records | Integer | For data files |

### Metadata Implementation

```python
def upload_with_metadata(
    file_data: bytes,
    bucket: str,
    key: str,
    original_filename: str = None,
    schema_version: str = None,
    record_count: int = None
) -> dict:
    """Upload file with standard metadata."""

    import hashlib

    metadata = {
        'content-hash': hashlib.sha256(file_data).hexdigest(),
        'uploaded-at': datetime.utcnow().isoformat() + 'Z',
        'uploaded-by': os.environ.get('SERVICE_NAME', 'unknown')
    }

    if original_filename:
        metadata['original-filename'] = original_filename

    if schema_version:
        metadata['schema-version'] = schema_version

    if record_count is not None:
        metadata['record-count'] = str(record_count)

    response = s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=file_data,
        Metadata=metadata,
        ServerSideEncryption='aws:kms'
    )

    return response
```

### Metadata for Different Data Types

#### Parquet Files

```python
parquet_metadata = {
    'schema-version': '1.2.0',
    'record-count': '10000',
    'partition-keys': 'year,month,tenant_id',
    'compression': 'snappy'
}
```

#### Report Files

```python
report_metadata = {
    'report-type': 'cbam-declaration',
    'report-period': '2026-Q1',
    'generated-by': 'report-service',
    'template-version': '2.1.0'
}
```

#### Model Files

```python
model_metadata = {
    'model-name': 'emission-predictor',
    'model-version': '1.2.3',
    'framework': 'scikit-learn',
    'training-date': '2026-02-01T00:00:00Z',
    'accuracy-score': '0.95'
}
```

---

## Related Documents

- [Architecture Guide](architecture-guide.md)
- [Developer Guide](developer-guide.md)
- [Compliance Guide](compliance-guide.md)

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-03 | Platform Engineering | Initial release |
