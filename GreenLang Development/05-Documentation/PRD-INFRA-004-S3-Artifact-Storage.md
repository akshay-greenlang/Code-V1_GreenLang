# PRD: INFRA-004 - S3/Object Storage for Artifacts

**Document Version:** 1.0
**Date:** February 3, 2026
**Status:** READY FOR EXECUTION
**Priority:** P0 - CRITICAL
**Owner:** Infrastructure Team
**Ralphy Task ID:** INFRA-004
**Depends On:** INFRA-001 (EKS Cluster), INFRA-002 (PostgreSQL), INFRA-003 (Redis)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Architecture Overview](#3-architecture-overview)
4. [Technical Requirements](#4-technical-requirements)
5. [Data Lake Architecture](#5-data-lake-architecture)
6. [Use Cases](#6-use-cases)
7. [Security Requirements](#7-security-requirements)
8. [Monitoring and Alerting](#8-monitoring-and-alerting)
9. [Backup and Recovery](#9-backup-and-recovery)
10. [Cost Estimation](#10-cost-estimation)
11. [Implementation Phases](#11-implementation-phases)
12. [Acceptance Criteria](#12-acceptance-criteria)
13. [Dependencies](#13-dependencies)
14. [Risks and Mitigations](#14-risks-and-mitigations)

---

## 1. Executive Summary

### 1.1 Overview

Deploy a production-ready S3-based artifact storage and data lake infrastructure for the GreenLang Climate OS platform. This infrastructure provides:

- **Centralized artifact storage** for build outputs, ML models, and pack bundles
- **Data Lake architecture** with Raw, Bronze, Silver, and Gold zones
- **Compliance-ready storage** for CSRD/CBAM reports with 7-year retention
- **Immutable audit logs** for SOX and regulatory compliance
- **High-performance access** via VPC endpoints and caching integration

### 1.2 Business Justification

| Requirement | Without S3 Infrastructure | With S3 Infrastructure |
|-------------|--------------------------|------------------------|
| **Artifact Storage** | Local disks, inconsistent | Centralized, versioned, encrypted |
| **Report Archival** | Manual, scattered | Automated, compliant, searchable |
| **Audit Trail** | Database logs, deletable | Immutable S3 Object Lock |
| **Data Analytics** | Ad-hoc queries | Data Lake with Athena/Glue |
| **Disaster Recovery** | Manual backups | Cross-region replication |
| **Cost Efficiency** | Single tier storage | Intelligent Tiering, lifecycle |

### 1.3 Expected Benefits and ROI

| Benefit | Impact | Annual Value |
|---------|--------|--------------|
| **70% Storage Cost Reduction** | Intelligent Tiering + Lifecycle | $50K savings |
| **Compliance Automation** | 7-year retention, immutable logs | Risk mitigation |
| **Faster Analytics** | Athena queries on data lake | $30K productivity |
| **Reduced DB Load** | Offload reports/artifacts to S3 | $20K infrastructure |
| **DR Capability** | Cross-region replication | Business continuity |
| **Total Annual Value** | | **~$100K+** |

**Investment:** ~$6-12K/year (S3 + Data Lake services)
**ROI:** 8-15x return on infrastructure investment

---

## 2. Problem Statement

### 2.1 Current Storage Challenges

GreenLang Climate OS generates substantial data requiring persistent, compliant storage:

| Data Type | Current State | Required State | Volume |
|-----------|--------------|----------------|--------|
| **Emission Factors** | Database only | Versioned S3 + cache | 50K+ factors |
| **Calculation Results** | Ephemeral | 7-year archival | 1M+ results/year |
| **CSRD/CBAM Reports** | Local storage | Compliant archive | 10K+ reports/year |
| **Audit Logs** | Database, deletable | Immutable 7-year | 100GB+/year |
| **ML Models** | Ad-hoc storage | Versioned registry | 100+ models |
| **Build Artifacts** | CI/CD temporary | Retained artifacts | 500GB+/year |

### 2.2 Compliance Requirements

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Regulatory Retention Requirements                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  CSRD (Corporate Sustainability Reporting Directive)                        │
│  ├── Sustainability reports: 7 years                                        │
│  ├── Supporting calculations: 7 years                                       │
│  └── Audit trail: 7 years, immutable                                        │
│                                                                              │
│  CBAM (Carbon Border Adjustment Mechanism)                                  │
│  ├── Emissions declarations: 5 years minimum                                │
│  ├── Verification reports: 5 years minimum                                  │
│  └── Transaction records: 5 years minimum                                   │
│                                                                              │
│  SOX (Sarbanes-Oxley)                                                       │
│  ├── Financial audit trails: 7 years                                        │
│  ├── Change logs: 7 years                                                   │
│  └── Access logs: 7 years, tamper-proof                                     │
│                                                                              │
│  GDPR (General Data Protection Regulation)                                  │
│  ├── Data processing records: Duration of processing                        │
│  ├── Consent records: Duration + 3 years                                    │
│  └── Data subject requests: 3 years                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Data Growth Projections

```
Year 1 (2026):
├── Emission Factors: 10 GB (versioned)
├── Calculation Results: 500 GB
├── Reports: 100 GB
├── Audit Logs: 100 GB
├── ML Models: 50 GB
├── Build Artifacts: 500 GB
└── Total: ~1.3 TB

Year 3 (2028):
├── Emission Factors: 30 GB
├── Calculation Results: 2 TB
├── Reports: 500 GB
├── Audit Logs: 500 GB
├── ML Models: 200 GB
├── Build Artifacts: 2 TB
└── Total: ~5.2 TB

Year 5 (2030):
├── Historical Data: 10 TB (archived)
├── Active Data: 3 TB
└── Total: ~13 TB
```

---

## 3. Architecture Overview

### 3.1 Multi-Tier Storage Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     GreenLang S3 Storage Architecture                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        HOT TIER (STANDARD)                           │    │
│  │                        0-30 days, Frequent Access                    │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐│    │
│  │  │  Artifacts  │  │   Reports   │  │  ML Models  │  │ Temp Files  ││    │
│  │  │  (Active)   │  │  (Recent)   │  │  (Active)   │  │ (Ephemeral) ││    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘│    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼ 30 days                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      WARM TIER (STANDARD-IA)                         │    │
│  │                      30-90 days, Infrequent Access                   │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │    │
│  │  │  Artifacts  │  │   Reports   │  │ Calculation │                  │    │
│  │  │  (Older)    │  │  (Archive)  │  │   Results   │                  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼ 90 days                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                       COLD TIER (GLACIER-IR)                         │    │
│  │                       90-365 days, Rare Access                       │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │    │
│  │  │  Historical │  │   Annual    │  │   Backups   │                  │    │
│  │  │   Reports   │  │   Archives  │  │  (Monthly)  │                  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼ 365 days                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                   ARCHIVE TIER (GLACIER DEEP ARCHIVE)                │    │
│  │                   1-7 years, Compliance Retention                    │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │    │
│  │  │  Compliance │  │   Audit     │  │  Historical │                  │    │
│  │  │   Archives  │  │    Logs     │  │    Data     │                  │    │
│  │  │  (7 years)  │  │  (7 years)  │  │  (7 years)  │                  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 S3 Bucket Topology

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         S3 Bucket Architecture                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PRIMARY REGION: us-east-1                                                   │
│  ════════════════════════════════════════════════════════════════════════   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  APPLICATION BUCKETS                                                 │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │                                                                      │    │
│  │  greenlang-prod-artifacts                                           │    │
│  │  ├── builds/           Build outputs, CI/CD artifacts               │    │
│  │  ├── models/           ML models, trained weights                   │    │
│  │  ├── packs/            GreenLang pack bundles                       │    │
│  │  └── uploads/          User uploads (temporary)                     │    │
│  │                                                                      │    │
│  │  greenlang-prod-reports                                             │    │
│  │  ├── csrd/             CSRD sustainability reports                  │    │
│  │  ├── cbam/             CBAM declarations                            │    │
│  │  ├── eudr/             EUDR compliance reports                      │    │
│  │  └── custom/           Custom report outputs                        │    │
│  │                                                                      │    │
│  │  greenlang-prod-audit-logs                                          │    │
│  │  ├── access/           API access logs                              │    │
│  │  ├── changes/          Data change audit trail                      │    │
│  │  ├── calculations/     Calculation provenance                       │    │
│  │  └── compliance/       Compliance event logs                        │    │
│  │  [Object Lock: COMPLIANCE mode, 7 years]                            │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  DATA LAKE BUCKETS                                                   │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │                                                                      │    │
│  │  greenlang-prod-data-lake-raw                                       │    │
│  │  ├── emissions/        Raw emission data ingestion                  │    │
│  │  ├── factors/          Emission factor sources                      │    │
│  │  ├── erp/              ERP data exports                             │    │
│  │  └── external/         External API data                            │    │
│  │                                                                      │    │
│  │  greenlang-prod-data-lake-processed                                 │    │
│  │  ├── bronze/           Cleansed, validated data                     │    │
│  │  ├── silver/           Conformed, enriched data                     │    │
│  │  └── gold/             Aggregated, analytics-ready                  │    │
│  │  [Format: Parquet with Snappy compression]                          │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  INFRASTRUCTURE BUCKETS                                              │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │                                                                      │    │
│  │  greenlang-prod-backups                                             │    │
│  │  ├── database/         PostgreSQL backups (pgBackRest)              │    │
│  │  ├── redis/            Redis RDB/AOF backups                        │    │
│  │  ├── velero/           Kubernetes state backups                     │    │
│  │  └── config/           Configuration snapshots                      │    │
│  │  [Object Lock: GOVERNANCE mode, 90 days]                            │    │
│  │                                                                      │    │
│  │  greenlang-prod-logs                                                │    │
│  │  ├── cloudtrail/       AWS API audit logs                           │    │
│  │  ├── s3-access/        S3 access logs                               │    │
│  │  ├── alb/              Application load balancer logs               │    │
│  │  └── vpc-flow/         VPC flow logs                                │    │
│  │                                                                      │    │
│  │  greenlang-prod-static                                              │    │
│  │  ├── frontend/         React app static assets                      │    │
│  │  ├── docs/             Documentation site                           │    │
│  │  └── public/           Public assets (CORS enabled)                 │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  DR REGION: us-west-2                                                        │
│  ════════════════════════════════════════════════════════════════════════   │
│                                                                              │
│  greenlang-dr-*  (Cross-region replication targets)                         │
│  ├── greenlang-dr-artifacts                                                 │
│  ├── greenlang-dr-reports                                                   │
│  ├── greenlang-dr-audit-logs                                                │
│  └── greenlang-dr-backups                                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Data Flow Architecture                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INGEST FLOW                                                                 │
│  ═══════════                                                                 │
│                                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  API     │───▶│  Lambda  │───▶│  S3 Raw  │───▶│  Glue    │              │
│  │ Request  │    │ Validate │    │  Bucket  │    │ Crawler  │              │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘              │
│                                                        │                     │
│                                                        ▼                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  Glue    │◀───│  Bronze  │◀───│   ETL    │◀───│  Data    │              │
│  │ Catalog  │    │  Zone    │    │   Job    │    │ Catalog  │              │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘              │
│                                                                              │
│  QUERY FLOW                                                                  │
│  ══════════                                                                  │
│                                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  Athena  │───▶│   Glue   │───▶│  S3 Gold │───▶│  Result  │              │
│  │  Query   │    │ Catalog  │    │  Zone    │    │  Cache   │              │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘              │
│        │                                                │                    │
│        └─────────────────▶ Redis Cache ◀────────────────┘                   │
│                                                                              │
│  ARTIFACT FLOW                                                               │
│  ═════════════                                                               │
│                                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  CI/CD   │───▶│ Presigned│───▶│    S3    │───▶│  Event   │              │
│  │ Pipeline │    │   URL    │    │ Artifacts│    │ Trigger  │              │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘              │
│                                                        │                     │
│                                    ┌───────────────────┼───────────────────┐│
│                                    │                   │                   ││
│                                    ▼                   ▼                   ▼│
│                              ┌──────────┐       ┌──────────┐       ┌──────────┐
│                              │ Validate │       │  Index   │       │  Audit   │
│                              │  Lambda  │       │  Lambda  │       │  Lambda  │
│                              └──────────┘       └──────────┘       └──────────┘
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Technical Requirements

### 4.1 S3 Bucket Configuration

| Bucket | Storage Class | Versioning | Encryption | Object Lock | Replication |
|--------|--------------|------------|------------|-------------|-------------|
| artifacts | Intelligent-Tiering | Enabled | SSE-KMS | None | Enabled |
| reports | Standard → Glacier | Enabled | SSE-KMS | None | Enabled |
| audit-logs | Standard | Enabled | SSE-KMS | COMPLIANCE 7y | Enabled |
| data-lake-raw | Standard | Enabled | SSE-KMS | None | None |
| data-lake-processed | Standard | Enabled | SSE-KMS | None | None |
| backups | Standard → Glacier | Enabled | SSE-KMS | GOVERNANCE 90d | Enabled |
| logs | Standard → Glacier | Disabled | SSE-S3 | None | None |
| static | Standard | Disabled | SSE-S3 | None | None |

### 4.2 Lifecycle Policies

```yaml
# Artifact Lifecycle Policy
artifacts:
  transitions:
    - days: 30
      storage_class: INTELLIGENT_TIERING
    - days: 90
      storage_class: GLACIER_IR
    - days: 365
      storage_class: GLACIER
  expiration:
    days: 730  # 2 years
  noncurrent_expiration:
    days: 90

# Report Lifecycle Policy
reports:
  transitions:
    - days: 30
      storage_class: STANDARD_IA
    - days: 90
      storage_class: GLACIER_IR
    - days: 365
      storage_class: GLACIER
    - days: 730
      storage_class: DEEP_ARCHIVE
  expiration:
    days: 2557  # 7 years

# Audit Log Lifecycle Policy
audit_logs:
  transitions:
    - days: 90
      storage_class: GLACIER_IR
    - days: 365
      storage_class: DEEP_ARCHIVE
  # No expiration - Object Lock enforces 7-year retention
```

### 4.3 Encryption Configuration

```hcl
# KMS Key for S3 Encryption
resource "aws_kms_key" "s3_key" {
  description             = "KMS key for GreenLang S3 encryption"
  deletion_window_in_days = 30
  enable_key_rotation     = true

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "Enable IAM policies"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      },
      {
        Sid    = "Allow S3 service"
        Effect = "Allow"
        Principal = {
          Service = "s3.amazonaws.com"
        }
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:GenerateDataKey*"
        ]
        Resource = "*"
      }
    ]
  })

  tags = local.tags
}
```

### 4.4 Object Key Structure

```
# Artifacts Bucket
s3://greenlang-prod-artifacts/
├── builds/
│   └── {pipeline_id}/{run_id}/{artifact_name}
├── models/
│   └── {model_name}/{version}/{artifact_name}
├── packs/
│   └── {pack_name}/{version}/{pack_name}-{version}.tar.gz
└── uploads/
    └── {tenant_id}/{date}/{upload_id}/{filename}

# Reports Bucket
s3://greenlang-prod-reports/
├── csrd/
│   └── {tenant_id}/{year}/{report_id}/
│       ├── report.pdf
│       ├── report.html
│       ├── supporting_data.xlsx
│       └── metadata.json
├── cbam/
│   └── {tenant_id}/{quarter}/{declaration_id}/
└── eudr/
    └── {tenant_id}/{year}/{statement_id}/

# Data Lake Buckets
s3://greenlang-prod-data-lake-processed/
├── bronze/
│   └── {source}/{year}/{month}/{day}/
│       └── data.parquet
├── silver/
│   └── {domain}/{entity}/{year}/{month}/{day}/
│       └── data.parquet
└── gold/
    └── {domain}/{metric}/{year}/{month}/
        └── aggregates.parquet
```

---

## 5. Data Lake Architecture

### 5.1 Zone Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Data Lake Zone Architecture                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  RAW ZONE (Landing)                                                          │
│  ══════════════════                                                          │
│  Purpose: Raw data ingestion, source-of-record                               │
│  Format: Original format (JSON, CSV, XML, API responses)                     │
│  Retention: 90 days (then archived)                                          │
│  Access: Write by ingestion pipelines, Read by ETL jobs                      │
│                                                                              │
│       ┌─────────────────────────────────────────────────────────────┐       │
│       │  Raw Data Sources                                            │       │
│       │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │       │
│       │  │   ERP   │  │External │  │  User   │  │   API   │        │       │
│       │  │ Exports │  │  APIs   │  │Uploads  │  │ Responses│       │       │
│       │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘        │       │
│       │       └────────────┴────────────┴────────────┘              │       │
│       │                           │                                  │       │
│       │                           ▼                                  │       │
│       │              ┌─────────────────────────┐                    │       │
│       │              │    s3://raw-zone/       │                    │       │
│       │              │    {source}/{date}/     │                    │       │
│       │              └─────────────────────────┘                    │       │
│       └─────────────────────────────────────────────────────────────┘       │
│                                    │                                         │
│                                    │ Glue ETL: Validate, Cleanse             │
│                                    ▼                                         │
│  BRONZE ZONE (Cleansed)                                                      │
│  ══════════════════════                                                      │
│  Purpose: Validated, cleansed data with schema enforcement                   │
│  Format: Parquet (Snappy compression)                                        │
│  Retention: 1 year                                                           │
│  Access: ETL jobs, Data engineers                                            │
│                                                                              │
│       ┌─────────────────────────────────────────────────────────────┐       │
│       │              ┌─────────────────────────┐                    │       │
│       │              │   s3://bronze-zone/     │                    │       │
│       │              │   {source}/{date}/      │                    │       │
│       │              │   ├── data.parquet      │                    │       │
│       │              │   └── _metadata/        │                    │       │
│       │              └─────────────────────────┘                    │       │
│       │                                                              │       │
│       │  Transformations Applied:                                    │       │
│       │  • Schema validation and enforcement                        │       │
│       │  • Data type casting                                        │       │
│       │  • Null handling and defaults                               │       │
│       │  • Deduplication                                            │       │
│       │  • PII masking (if applicable)                              │       │
│       └─────────────────────────────────────────────────────────────┘       │
│                                    │                                         │
│                                    │ Glue ETL: Conform, Enrich               │
│                                    ▼                                         │
│  SILVER ZONE (Conformed)                                                     │
│  ═══════════════════════                                                     │
│  Purpose: Business-aligned entities, enriched data                           │
│  Format: Parquet (Snappy compression), Partitioned                           │
│  Retention: 3 years                                                          │
│  Access: Analysts, BI tools, ML pipelines                                    │
│                                                                              │
│       ┌─────────────────────────────────────────────────────────────┐       │
│       │              ┌─────────────────────────┐                    │       │
│       │              │   s3://silver-zone/     │                    │       │
│       │              │   {domain}/{entity}/    │                    │       │
│       │              │   year={y}/month={m}/   │                    │       │
│       │              │   └── data.parquet      │                    │       │
│       │              └─────────────────────────┘                    │       │
│       │                                                              │       │
│       │  Transformations Applied:                                    │       │
│       │  • Entity resolution and matching                           │       │
│       │  • Reference data enrichment                                │       │
│       │  • Calculated fields (emission factors applied)             │       │
│       │  • Partitioning by date                                     │       │
│       └─────────────────────────────────────────────────────────────┘       │
│                                    │                                         │
│                                    │ Glue ETL: Aggregate                     │
│                                    ▼                                         │
│  GOLD ZONE (Aggregated)                                                      │
│  ══════════════════════                                                      │
│  Purpose: Analytics-ready aggregates, KPIs, metrics                          │
│  Format: Parquet (optimized for queries)                                     │
│  Retention: 7 years (compliance)                                             │
│  Access: Dashboards, Reports, Athena queries                                 │
│                                                                              │
│       ┌─────────────────────────────────────────────────────────────┐       │
│       │              ┌─────────────────────────┐                    │       │
│       │              │   s3://gold-zone/       │                    │       │
│       │              │   {domain}/{metric}/    │                    │       │
│       │              │   year={y}/month={m}/   │                    │       │
│       │              │   └── aggregates.parquet│                    │       │
│       │              └─────────────────────────┘                    │       │
│       │                                                              │       │
│       │  Pre-computed Aggregates:                                    │       │
│       │  • Total emissions by scope (1, 2, 3)                       │       │
│       │  • Emissions by category and activity                       │       │
│       │  • Year-over-year comparisons                               │       │
│       │  • Benchmark comparisons                                    │       │
│       └─────────────────────────────────────────────────────────────┘       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 AWS Glue Configuration

```python
# Glue ETL Job: Raw to Bronze
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame

args = getResolvedOptions(sys.argv, ['JOB_NAME', 'source_bucket', 'target_bucket'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Read raw data
raw_df = glueContext.create_dynamic_frame.from_options(
    connection_type="s3",
    connection_options={
        "paths": [f"s3://{args['source_bucket']}/emissions/"],
        "recurse": True
    },
    format="json"
)

# Apply schema validation
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType

schema = StructType([
    StructField("tenant_id", StringType(), False),
    StructField("activity_id", StringType(), False),
    StructField("emission_factor_id", StringType(), False),
    StructField("quantity", DoubleType(), False),
    StructField("unit", StringType(), False),
    StructField("timestamp", TimestampType(), False),
    StructField("source", StringType(), True),
])

# Convert to DataFrame and apply schema
df = raw_df.toDF()
validated_df = spark.createDataFrame(df.rdd, schema)

# Deduplicate
deduplicated_df = validated_df.dropDuplicates(["tenant_id", "activity_id", "timestamp"])

# Write to Bronze zone as Parquet
deduplicated_df.write \
    .mode("append") \
    .partitionBy("tenant_id", "year", "month") \
    .parquet(f"s3://{args['target_bucket']}/bronze/emissions/")

job.commit()
```

### 5.3 Athena Query Configuration

```sql
-- Create Athena table with partition projection
CREATE EXTERNAL TABLE gold.emissions_summary (
    tenant_id STRING,
    scope INT,
    category STRING,
    total_emissions DOUBLE,
    unit STRING,
    record_count BIGINT
)
PARTITIONED BY (year INT, month INT)
ROW FORMAT SERDE 'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe'
STORED AS PARQUET
LOCATION 's3://greenlang-prod-data-lake-processed/gold/emissions/summary/'
TBLPROPERTIES (
    'projection.enabled' = 'true',
    'projection.year.type' = 'integer',
    'projection.year.range' = '2020,2030',
    'projection.month.type' = 'integer',
    'projection.month.range' = '1,12',
    'storage.location.template' = 's3://greenlang-prod-data-lake-processed/gold/emissions/summary/year=${year}/month=${month}'
);

-- Example query: Monthly emissions by tenant
SELECT
    tenant_id,
    scope,
    SUM(total_emissions) as emissions,
    unit
FROM gold.emissions_summary
WHERE year = 2026 AND month BETWEEN 1 AND 6
GROUP BY tenant_id, scope, unit
ORDER BY emissions DESC;
```

---

## 6. Use Cases

### 6.1 Emission Factor Versioning

```python
# emission_factor_storage.py
import boto3
import json
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any

class EmissionFactorStorage:
    """
    Store and retrieve versioned emission factors in S3.
    """

    def __init__(self, bucket: str, prefix: str = "factors"):
        self.s3 = boto3.client('s3')
        self.bucket = bucket
        self.prefix = prefix

    def store_factor(
        self,
        factor_id: str,
        factor_data: Dict[str, Any],
        source: str,
        effective_date: str
    ) -> str:
        """Store a new emission factor version."""

        # Generate version hash
        content = json.dumps(factor_data, sort_keys=True)
        version_hash = hashlib.sha256(content.encode()).hexdigest()[:12]

        # Build S3 key with versioning
        key = f"{self.prefix}/{factor_id}/{effective_date}/{version_hash}.json"

        # Add metadata
        metadata = {
            "factor_id": factor_id,
            "source": source,
            "effective_date": effective_date,
            "version_hash": version_hash,
            "created_at": datetime.utcnow().isoformat(),
        }

        # Store with metadata
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps({
                "metadata": metadata,
                "data": factor_data
            }),
            ContentType="application/json",
            Metadata={
                "factor-id": factor_id,
                "version-hash": version_hash,
                "source": source
            }
        )

        # Update latest pointer
        self._update_latest(factor_id, key)

        return version_hash

    def get_factor(
        self,
        factor_id: str,
        version: Optional[str] = None,
        effective_date: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Retrieve emission factor by ID, optionally at specific version."""

        if version:
            # Get specific version
            prefix = f"{self.prefix}/{factor_id}/"
            response = self.s3.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix
            )

            for obj in response.get('Contents', []):
                if version in obj['Key']:
                    result = self.s3.get_object(
                        Bucket=self.bucket,
                        Key=obj['Key']
                    )
                    return json.loads(result['Body'].read())
        else:
            # Get latest
            key = f"{self.prefix}/{factor_id}/latest.json"
            try:
                result = self.s3.get_object(
                    Bucket=self.bucket,
                    Key=key
                )
                pointer = json.loads(result['Body'].read())

                result = self.s3.get_object(
                    Bucket=self.bucket,
                    Key=pointer['key']
                )
                return json.loads(result['Body'].read())
            except self.s3.exceptions.NoSuchKey:
                return None

        return None

    def _update_latest(self, factor_id: str, key: str):
        """Update the latest pointer for a factor."""
        pointer_key = f"{self.prefix}/{factor_id}/latest.json"
        self.s3.put_object(
            Bucket=self.bucket,
            Key=pointer_key,
            Body=json.dumps({"key": key}),
            ContentType="application/json"
        )
```

### 6.2 Report Archival

```python
# report_archival.py
import boto3
from botocore.config import Config
import json
from datetime import datetime
from typing import Dict, Any, Optional
import uuid

class ReportArchival:
    """
    Archive compliance reports with regulatory retention.
    """

    RETENTION_DAYS = {
        "csrd": 2557,  # 7 years
        "cbam": 1825,  # 5 years
        "eudr": 1825,  # 5 years
        "custom": 365  # 1 year
    }

    def __init__(self, bucket: str):
        self.s3 = boto3.client('s3')
        self.bucket = bucket

    def archive_report(
        self,
        report_type: str,
        tenant_id: str,
        report_content: bytes,
        content_type: str,
        metadata: Dict[str, Any]
    ) -> str:
        """Archive a report with compliance metadata."""

        report_id = str(uuid.uuid4())
        year = datetime.utcnow().year

        # Build S3 key
        key = f"{report_type}/{tenant_id}/{year}/{report_id}/report"
        if content_type == "application/pdf":
            key += ".pdf"
        elif content_type == "text/html":
            key += ".html"
        else:
            key += ".bin"

        # Calculate retention
        retention_days = self.RETENTION_DAYS.get(report_type, 365)

        # Store report
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=report_content,
            ContentType=content_type,
            Metadata={
                "report-id": report_id,
                "report-type": report_type,
                "tenant-id": tenant_id,
                "retention-days": str(retention_days),
                "archived-at": datetime.utcnow().isoformat()
            },
            Tagging=f"report_type={report_type}&tenant_id={tenant_id}&compliance=true"
        )

        # Store metadata
        metadata_key = f"{report_type}/{tenant_id}/{year}/{report_id}/metadata.json"
        self.s3.put_object(
            Bucket=self.bucket,
            Key=metadata_key,
            Body=json.dumps({
                "report_id": report_id,
                "report_type": report_type,
                "tenant_id": tenant_id,
                "archived_at": datetime.utcnow().isoformat(),
                "retention_days": retention_days,
                "content_type": content_type,
                "custom_metadata": metadata
            }),
            ContentType="application/json"
        )

        return report_id

    def get_report(
        self,
        report_type: str,
        tenant_id: str,
        report_id: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve archived report."""

        # Find the report
        prefix = f"{report_type}/{tenant_id}/"
        response = self.s3.list_objects_v2(
            Bucket=self.bucket,
            Prefix=prefix
        )

        for obj in response.get('Contents', []):
            if report_id in obj['Key'] and 'metadata.json' not in obj['Key']:
                # Get report content
                result = self.s3.get_object(
                    Bucket=self.bucket,
                    Key=obj['Key']
                )

                return {
                    "content": result['Body'].read(),
                    "content_type": result['ContentType'],
                    "metadata": result['Metadata']
                }

        return None
```

### 6.3 Immutable Audit Logging

```python
# audit_logger.py
import boto3
import json
from datetime import datetime
import hashlib
from typing import Dict, Any

class ImmutableAuditLogger:
    """
    Log audit events to S3 with Object Lock for compliance.
    """

    def __init__(self, bucket: str):
        self.s3 = boto3.client('s3')
        self.bucket = bucket
        self._last_hash = None

    def log_event(
        self,
        event_type: str,
        actor: str,
        resource: str,
        action: str,
        details: Dict[str, Any]
    ) -> str:
        """Log an immutable audit event."""

        timestamp = datetime.utcnow()

        # Build audit record
        record = {
            "timestamp": timestamp.isoformat(),
            "event_type": event_type,
            "actor": actor,
            "resource": resource,
            "action": action,
            "details": details,
            "previous_hash": self._last_hash
        }

        # Calculate record hash (chain link)
        record_json = json.dumps(record, sort_keys=True)
        record_hash = hashlib.sha256(record_json.encode()).hexdigest()
        record["record_hash"] = record_hash

        # Build S3 key
        date_prefix = timestamp.strftime("%Y/%m/%d/%H")
        key = f"audit/{event_type}/{date_prefix}/{record_hash}.json"

        # Store with Object Lock (retention set at bucket level)
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(record, indent=2),
            ContentType="application/json",
            Metadata={
                "event-type": event_type,
                "actor": actor,
                "record-hash": record_hash
            }
        )

        # Update hash chain
        self._last_hash = record_hash

        return record_hash

    def verify_chain(self, start_key: str, end_key: str) -> bool:
        """Verify audit log chain integrity."""

        # List objects in range
        response = self.s3.list_objects_v2(
            Bucket=self.bucket,
            Prefix="audit/",
            StartAfter=start_key
        )

        previous_hash = None
        for obj in response.get('Contents', []):
            if obj['Key'] > end_key:
                break

            result = self.s3.get_object(
                Bucket=self.bucket,
                Key=obj['Key']
            )
            record = json.loads(result['Body'].read())

            # Verify hash chain
            if previous_hash and record.get('previous_hash') != previous_hash:
                return False

            # Verify record hash
            expected_hash = record.pop('record_hash')
            calculated_hash = hashlib.sha256(
                json.dumps(record, sort_keys=True).encode()
            ).hexdigest()

            if calculated_hash != expected_hash:
                return False

            previous_hash = expected_hash

        return True
```

---

## 7. Security Requirements

### 7.1 Encryption Configuration

| Layer | Method | Key Management |
|-------|--------|----------------|
| At Rest | SSE-KMS | Customer-managed CMK with rotation |
| In Transit | TLS 1.2+ | AWS-managed certificates |
| Client-side | Optional | Application-managed keys |

### 7.2 Access Control Matrix

| Role | Artifacts | Reports | Audit Logs | Data Lake | Backups |
|------|-----------|---------|------------|-----------|---------|
| **Platform Admin** | Full | Full | Read | Full | Full |
| **Developer** | Read/Write | Read | None | Read | None |
| **Data Engineer** | Read | Read | None | Full | Read |
| **Analyst** | Read | Read | None | Read (Gold) | None |
| **Auditor** | Read | Read | Read | Read | Read |
| **Application (IRSA)** | Read/Write | Write | Write | Read/Write | None |
| **CI/CD** | Write | None | None | None | None |

### 7.3 Bucket Policy Template

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "DenyNonSSLAccess",
            "Effect": "Deny",
            "Principal": "*",
            "Action": "s3:*",
            "Resource": [
                "arn:aws:s3:::greenlang-prod-artifacts",
                "arn:aws:s3:::greenlang-prod-artifacts/*"
            ],
            "Condition": {
                "Bool": {
                    "aws:SecureTransport": "false"
                }
            }
        },
        {
            "Sid": "DenyIncorrectEncryption",
            "Effect": "Deny",
            "Principal": "*",
            "Action": "s3:PutObject",
            "Resource": "arn:aws:s3:::greenlang-prod-artifacts/*",
            "Condition": {
                "StringNotEquals": {
                    "s3:x-amz-server-side-encryption": "aws:kms"
                }
            }
        },
        {
            "Sid": "RestrictToVPCEndpoint",
            "Effect": "Deny",
            "Principal": "*",
            "Action": "s3:*",
            "Resource": [
                "arn:aws:s3:::greenlang-prod-artifacts",
                "arn:aws:s3:::greenlang-prod-artifacts/*"
            ],
            "Condition": {
                "StringNotEquals": {
                    "aws:SourceVpce": "${vpc_endpoint_id}"
                }
            }
        }
    ]
}
```

### 7.4 IRSA Configuration

```yaml
# Kubernetes ServiceAccount with IRSA
apiVersion: v1
kind: ServiceAccount
metadata:
  name: greenlang-s3-access
  namespace: greenlang
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::${ACCOUNT_ID}:role/greenlang-prod-s3-access
---
# IAM Policy for the role
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::greenlang-prod-artifacts",
                "arn:aws:s3:::greenlang-prod-artifacts/*",
                "arn:aws:s3:::greenlang-prod-reports",
                "arn:aws:s3:::greenlang-prod-reports/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "kms:Decrypt",
                "kms:GenerateDataKey"
            ],
            "Resource": "arn:aws:kms:us-east-1:${ACCOUNT_ID}:key/${KMS_KEY_ID}"
        }
    ]
}
```

---

## 8. Monitoring and Alerting

### 8.1 CloudWatch Metrics

| Metric | Description | Threshold | Severity |
|--------|-------------|-----------|----------|
| BucketSizeBytes | Total storage size | > 1TB | Warning |
| NumberOfObjects | Object count | > 10M | Warning |
| 4xxErrors | Client errors | > 1% | Warning |
| 5xxErrors | Server errors | > 0.1% | Critical |
| FirstByteLatency | Time to first byte | p99 > 200ms | Warning |
| TotalRequestLatency | Total request time | p99 > 500ms | Warning |

### 8.2 Alert Rules

```yaml
# S3 Prometheus Alerts
groups:
  - name: s3_alerts
    rules:
      - alert: S3High5xxErrorRate
        expr: |
          sum(rate(aws_s3_5xx_errors_total[5m])) by (bucket)
          / sum(rate(aws_s3_all_requests_total[5m])) by (bucket)
          > 0.001
        for: 5m
        labels:
          severity: critical
          service: s3
        annotations:
          summary: "S3 bucket {{ $labels.bucket }} has high 5xx error rate"
          description: "Error rate is {{ $value | humanizePercentage }}"

      - alert: S3StorageQuotaWarning
        expr: |
          aws_s3_bucket_size_bytes / aws_s3_bucket_quota_bytes > 0.8
        for: 1h
        labels:
          severity: warning
          service: s3
        annotations:
          summary: "S3 bucket {{ $labels.bucket }} is at 80% capacity"

      - alert: S3ReplicationLagHigh
        expr: |
          aws_s3_replication_latency_seconds > 3600
        for: 15m
        labels:
          severity: warning
          service: s3
        annotations:
          summary: "S3 replication lag exceeds 1 hour"
```

### 8.3 Grafana Dashboard Panels

| Panel | Visualization | Data Source |
|-------|--------------|-------------|
| Total Storage | Stat | CloudWatch |
| Storage by Bucket | Pie Chart | CloudWatch |
| Request Rate | Time Series | CloudWatch |
| Error Rate | Gauge | CloudWatch |
| Latency Percentiles | Heatmap | CloudWatch |
| Cost Trend | Bar Chart | Cost Explorer |
| Replication Status | Status | CloudWatch |

---

## 9. Backup and Recovery

### 9.1 Cross-Region Replication

```hcl
# Terraform: Cross-region replication
resource "aws_s3_bucket_replication_configuration" "artifacts_replication" {
  bucket = aws_s3_bucket.artifacts.id
  role   = aws_iam_role.replication.arn

  rule {
    id     = "replicate-all"
    status = "Enabled"

    filter {
      prefix = ""
    }

    destination {
      bucket        = "arn:aws:s3:::greenlang-dr-artifacts"
      storage_class = "STANDARD_IA"

      encryption_configuration {
        replica_kms_key_id = aws_kms_key.dr_key.arn
      }

      replication_time {
        status = "Enabled"
        time {
          minutes = 15
        }
      }

      metrics {
        status = "Enabled"
        event_threshold {
          minutes = 15
        }
      }
    }

    delete_marker_replication {
      status = "Enabled"
    }
  }
}
```

### 9.2 RTO/RPO Targets

| Data Type | RTO | RPO | Backup Method |
|-----------|-----|-----|---------------|
| Artifacts | 4 hours | 15 minutes | Cross-region replication |
| Reports | 4 hours | 15 minutes | Cross-region replication |
| Audit Logs | 1 hour | 0 (synchronous) | Cross-region replication |
| Data Lake | 24 hours | 1 hour | Daily snapshots |
| Backups | 4 hours | 0 | Cross-region replication |

### 9.3 Recovery Procedures

```bash
#!/bin/bash
# S3 DR Failover Script

# 1. Verify DR region buckets are accessible
aws s3 ls s3://greenlang-dr-artifacts --region us-west-2

# 2. Update DNS/Route53 to point to DR region
aws route53 change-resource-record-sets \
  --hosted-zone-id ${HOSTED_ZONE_ID} \
  --change-batch file://dr-dns-changes.json

# 3. Update application configuration
kubectl set env deployment/greenlang-api \
  S3_BUCKET_ARTIFACTS=greenlang-dr-artifacts \
  S3_REGION=us-west-2 \
  -n greenlang

# 4. Verify application connectivity
kubectl exec -it deployment/greenlang-api -n greenlang -- \
  aws s3 ls s3://greenlang-dr-artifacts
```

---

## 10. Cost Estimation

### 10.1 Monthly Cost Breakdown

| Component | Quantity | Unit Cost | Monthly Cost |
|-----------|----------|-----------|--------------|
| **S3 Standard** | 500 GB | $0.023/GB | $11.50 |
| **S3 Standard-IA** | 300 GB | $0.0125/GB | $3.75 |
| **S3 Glacier IR** | 200 GB | $0.004/GB | $0.80 |
| **S3 Glacier** | 500 GB | $0.0036/GB | $1.80 |
| **S3 Deep Archive** | 1 TB | $0.00099/GB | $1.01 |
| **PUT/POST Requests** | 10M | $0.005/1K | $50.00 |
| **GET Requests** | 50M | $0.0004/1K | $20.00 |
| **Data Transfer** | 500 GB | $0.09/GB | $45.00 |
| **Cross-Region Replication** | 500 GB | $0.02/GB | $10.00 |
| **KMS** | 10K requests | $0.03/10K | $0.30 |
| **Glue Crawlers** | 10 DPU-hours | $0.44/DPU-hr | $4.40 |
| **Glue ETL** | 50 DPU-hours | $0.44/DPU-hr | $22.00 |
| **Athena Queries** | 100 GB scanned | $5/TB | $0.50 |
| **CloudWatch** | Metrics/Logs | Estimate | $20.00 |
| **Total** | | | **~$191/month** |

### 10.2 Cost by Environment

| Environment | Storage | Requests | Replication | Total |
|-------------|---------|----------|-------------|-------|
| **Development** | $20 | $10 | $0 | **$30** |
| **Staging** | $50 | $30 | $0 | **$80** |
| **Production** | $120 | $100 | $50 | **$270** |
| **DR Region** | $60 | $10 | $0 | **$70** |
| **Total** | | | | **~$450/month** |

### 10.3 Cost Optimization Strategies

1. **Intelligent Tiering**: Automatic cost optimization for unpredictable access
2. **Lifecycle Policies**: Move data to cheaper tiers automatically
3. **S3 Select**: Query only needed columns (90% cost reduction)
4. **Compression**: Parquet with Snappy (60-80% size reduction)
5. **Partitioning**: Reduce Athena scan costs by 90%+

---

## 11. Implementation Phases

### Phase 1: Core Infrastructure (Week 1)

| Task | Owner | Duration | Dependencies |
|------|-------|----------|--------------|
| Create KMS keys | DevOps | 2 hours | None |
| Create S3 buckets (all) | DevOps | 4 hours | KMS keys |
| Configure bucket policies | Security | 4 hours | Buckets |
| Set up VPC endpoints | DevOps | 2 hours | VPC |
| Configure lifecycle policies | DevOps | 2 hours | Buckets |
| Set up cross-region replication | DevOps | 4 hours | DR buckets |

### Phase 2: Data Lake Setup (Week 2)

| Task | Owner | Duration | Dependencies |
|------|-------|----------|--------------|
| Create Glue databases | Data Eng | 2 hours | Buckets |
| Create Glue crawlers | Data Eng | 4 hours | Databases |
| Create ETL jobs | Data Eng | 8 hours | Crawlers |
| Configure Athena workgroups | Data Eng | 2 hours | None |
| Create named queries | Data Eng | 4 hours | Workgroups |
| Test ETL pipeline | Data Eng | 4 hours | ETL jobs |

### Phase 3: Security Hardening (Week 3)

| Task | Owner | Duration | Dependencies |
|------|-------|----------|--------------|
| Configure Object Lock | Security | 4 hours | Audit bucket |
| Set up IRSA roles | Security | 4 hours | EKS |
| Configure AWS Config rules | Security | 4 hours | Buckets |
| Set up CloudTrail | Security | 4 hours | Log bucket |
| Security audit | Security | 8 hours | All above |

### Phase 4: Monitoring & Documentation (Week 4)

| Task | Owner | Duration | Dependencies |
|------|-------|----------|--------------|
| Create Grafana dashboards | DevOps | 8 hours | CloudWatch |
| Configure Prometheus alerts | DevOps | 4 hours | Dashboards |
| Write developer documentation | Tech Writer | 8 hours | All |
| Write operations runbook | DevOps | 8 hours | All |
| DR testing | DevOps | 8 hours | Replication |

---

## 12. Acceptance Criteria

### 12.1 Infrastructure Criteria

- [ ] All 8 S3 buckets created with correct configuration
- [ ] KMS encryption enabled on all buckets
- [ ] Versioning enabled on required buckets
- [ ] Object Lock configured on audit-logs bucket (COMPLIANCE, 7 years)
- [ ] Object Lock configured on backups bucket (GOVERNANCE, 90 days)
- [ ] Lifecycle policies active and verified
- [ ] Cross-region replication working with < 15 minute lag
- [ ] VPC endpoint configured and bucket policies enforced

### 12.2 Data Lake Criteria

- [ ] Glue databases created for all zones (raw, bronze, silver, gold)
- [ ] Crawlers running on schedule and discovering schemas
- [ ] ETL jobs processing data successfully
- [ ] Athena queries returning results correctly
- [ ] Partition projection working for performance

### 12.3 Security Criteria

- [ ] IRSA configured for all application service accounts
- [ ] Bucket policies enforcing SSL and encryption
- [ ] AWS Config rules passing compliance checks
- [ ] CloudTrail logging S3 data events
- [ ] No public access on any bucket

### 12.4 Operational Criteria

- [ ] Grafana dashboards showing all metrics
- [ ] Prometheus alerts configured and tested
- [ ] Documentation complete and reviewed
- [ ] DR failover tested successfully
- [ ] Cost monitoring in place

---

## 13. Dependencies

### 13.1 Infrastructure Dependencies

| Dependency | Type | Status | Required For |
|------------|------|--------|--------------|
| INFRA-001 EKS Cluster | Hard | Complete | IRSA, VPC endpoints |
| INFRA-002 PostgreSQL | Soft | Complete | Glue metadata (optional) |
| INFRA-003 Redis | Soft | Complete | S3 metadata caching |
| AWS Account | Hard | Complete | All resources |
| VPC with private subnets | Hard | Complete | VPC endpoints |

### 13.2 Application Dependencies

| Component | Dependency | Integration Point |
|-----------|------------|-------------------|
| CI/CD Pipeline | Artifact bucket | Build output storage |
| Report Generator | Reports bucket | PDF/HTML storage |
| Audit Service | Audit-logs bucket | Compliance logging |
| ML Pipeline | Artifacts bucket | Model storage |
| Analytics | Data Lake | Athena queries |

---

## 14. Risks and Mitigations

### 14.1 Risk Matrix

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Data loss** | Low | Critical | Cross-region replication, versioning |
| **Security breach** | Low | Critical | Encryption, VPC endpoints, bucket policies |
| **Cost overrun** | Medium | Medium | Lifecycle policies, alerts, budgets |
| **Compliance failure** | Low | Critical | Object Lock, audit logging, retention |
| **Performance issues** | Medium | Medium | CloudFront, caching, partitioning |
| **Replication lag** | Medium | Low | Monitoring, alerts, SLA tracking |

### 14.2 Contingency Plans

**Data Loss Scenario:**
1. Identify affected data from CloudTrail logs
2. Initiate recovery from DR region
3. Use S3 versioning to restore deleted objects
4. Conduct post-incident review

**Security Incident:**
1. Isolate affected bucket (remove all access)
2. Analyze CloudTrail and access logs
3. Rotate KMS keys if compromised
4. Restore from clean backup in DR region

**Cost Spike:**
1. Review CloudWatch cost metrics
2. Identify unexpected usage patterns
3. Apply emergency lifecycle policies
4. Implement request throttling if needed

---

## Appendix A: Terraform Module Reference

```hcl
# Example usage of S3 module
module "s3_storage" {
  source = "./modules/s3"

  environment = "prod"
  project     = "greenlang"

  # Bucket configuration
  create_artifacts_bucket  = true
  create_reports_bucket    = true
  create_audit_logs_bucket = true
  create_data_lake_buckets = true
  create_backups_bucket    = true
  create_logs_bucket       = true
  create_static_bucket     = true

  # Encryption
  kms_key_arn = module.kms.key_arn

  # Replication
  enable_replication     = true
  replication_region     = "us-west-2"
  replication_bucket_arn = module.s3_dr.bucket_arns

  # Object Lock
  audit_logs_retention_days = 2557  # 7 years
  backups_retention_days    = 90

  # Lifecycle
  artifact_expiration_days = 730
  logs_expiration_days     = 365

  # VPC
  vpc_endpoint_id = module.vpc.s3_endpoint_id

  tags = local.common_tags
}
```

---

## Appendix B: File Locations

| Component | Path |
|-----------|------|
| Terraform S3 Module | `deployment/terraform/modules/s3/` |
| Terraform Data Lake | `deployment/terraform/modules/data-lake/` |
| Terraform EFS | `deployment/terraform/modules/efs/` |
| S3 Events Module | `deployment/terraform/modules/s3-events/` |
| S3 Security Module | `deployment/terraform/modules/s3-security/` |
| S3 Compliance | `deployment/terraform/modules/s3-compliance/` |
| S3 Tagging | `deployment/terraform/modules/s3-tagging/` |
| CloudTrail Module | `deployment/terraform/modules/s3-cloudtrail/` |
| Grafana Dashboards | `deployment/monitoring/dashboards/s3-*.json` |
| Prometheus Alerts | `deployment/monitoring/alerts/s3-alerts.yaml` |
| K8s Lifecycle Jobs | `deployment/kubernetes/storage/artifact-lifecycle/` |
| K8s Promotion | `deployment/kubernetes/storage/artifact-promotion/` |
| Lambda Functions | `deployment/lambda/s3-events/` |
| Documentation | `deployment/docs/artifact-storage/` |

---

**End of PRD-INFRA-004 S3/Object Storage for Artifacts**
