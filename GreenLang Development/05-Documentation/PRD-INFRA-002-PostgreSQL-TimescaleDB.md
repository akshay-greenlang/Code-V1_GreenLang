# PRD: INFRA-002 - PostgreSQL + TimescaleDB Primary/Replica Configuration

**Document Version:** 1.0
**Date:** February 3, 2026
**Status:** READY FOR EXECUTION
**Priority:** P0 - CRITICAL
**Owner:** Infrastructure Team
**Ralphy Task ID:** INFRA-002
**Depends On:** INFRA-001 (EKS Cluster Deployment)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Architecture Overview](#3-architecture-overview)
4. [Technical Requirements](#4-technical-requirements)
5. [Security Requirements](#5-security-requirements)
6. [Monitoring and Alerting](#6-monitoring-and-alerting)
7. [Backup and Recovery](#7-backup-and-recovery)
8. [Cost Estimation](#8-cost-estimation)
9. [Implementation Phases](#9-implementation-phases)
10. [Acceptance Criteria](#10-acceptance-criteria)
11. [Dependencies](#11-dependencies)
12. [Risks and Mitigations](#12-risks-and-mitigations)

---

## 1. Executive Summary

### 1.1 Overview

Deploy a production-ready PostgreSQL database cluster with TimescaleDB extension for the GreenLang Climate OS platform. This infrastructure provides:

- **Time-series optimized storage** for billions of climate data points
- **High availability** with Primary/Replica topology
- **99.99% uptime SLA** for mission-critical regulatory reporting
- **Audit-grade data retention** for compliance requirements

### 1.2 Business Justification for Time-Series Database

| Requirement | Traditional PostgreSQL | PostgreSQL + TimescaleDB |
|-------------|----------------------|--------------------------|
| **Time-series queries** | Full table scans, slow | Hypertable partitioning, 10-100x faster |
| **Data compression** | Limited options | 10-40x compression ratio |
| **Continuous aggregates** | Manual materialized views | Automatic, real-time refresh |
| **Data retention** | Manual deletion jobs | Automatic chunk management |
| **Emissions data ingestion** | ~10K rows/sec | ~100K+ rows/sec |

### 1.3 Expected Benefits and ROI

| Benefit | Impact | Annual Value |
|---------|--------|--------------|
| **Query Performance** | 10-100x faster time-series queries | $150K (developer productivity) |
| **Storage Efficiency** | 10-40x compression | $50K (reduced storage costs) |
| **Operational Efficiency** | Automated retention, aggregation | $100K (reduced DBA overhead) |
| **Compliance Readiness** | Audit trails, point-in-time recovery | Risk mitigation (priceless) |
| **Total Annual Value** | | **~$300K** |

**Investment:** ~$30-42K/year (infrastructure costs)
**ROI:** 7-10x return on infrastructure investment

---

## 2. Problem Statement

### 2.1 Climate Data Storage Requirements

GreenLang Climate OS must store and query massive volumes of time-series climate data:

| Data Type | Volume (Year 1) | Volume (Year 3) | Retention |
|-----------|-----------------|-----------------|-----------|
| **Emission Measurements** | 100M rows | 5B rows | 7 years |
| **Sensor Readings** | 500M rows | 25B rows | 2 years |
| **Calculation Results** | 50M rows | 2.5B rows | 7 years |
| **Audit Logs** | 200M rows | 10B rows | 10 years |
| **User Activity** | 20M rows | 1B rows | 5 years |
| **Total Data Points** | **870M** | **43.5B** | - |

### 2.2 Need for Time-Series Optimized Queries

Climate reporting requires efficient time-series queries:

```sql
-- Example: Calculate quarterly emissions for CSRD reporting
-- Without TimescaleDB: 45 seconds (full table scan)
-- With TimescaleDB: 0.3 seconds (chunk pruning + continuous aggregates)

SELECT
    facility_id,
    time_bucket('1 quarter', measurement_time) AS quarter,
    SUM(co2_equivalent_kg) AS total_emissions,
    AVG(uncertainty_pct) AS avg_uncertainty
FROM emission_measurements
WHERE measurement_time BETWEEN '2025-01-01' AND '2025-12-31'
  AND scope = 'scope_1'
GROUP BY facility_id, quarter
ORDER BY quarter;
```

### 2.3 High Availability Requirements

| Requirement | Target | Justification |
|-------------|--------|---------------|
| **Uptime SLA** | 99.99% | Regulatory reporting deadlines are non-negotiable |
| **RTO (Recovery Time Objective)** | < 30 seconds | Business continuity for 24/7 operations |
| **RPO (Recovery Point Objective)** | 0 (zero data loss) | Audit-grade data integrity |
| **Failover Automation** | Fully automated | No human intervention during incidents |

### 2.4 Compliance Requirements

| Regulation | Requirement | Database Impact |
|------------|-------------|-----------------|
| **CSRD** | 7-year data retention | Long-term storage with immutable audit trail |
| **CBAM** | Quarterly reporting with lineage | Full calculation provenance |
| **SOC 2 Type II** | Access logging, encryption | Comprehensive audit logs |
| **GDPR** | Data subject rights, encryption | Row-level access control, encryption |
| **ISO 27001** | Information security controls | Network isolation, encryption |

---

## 3. Architecture Overview

### 3.1 Primary/Replica Topology Diagram

```
                                    ┌─────────────────────────────────────────────┐
                                    │           AWS REGION: us-east-1             │
                                    └─────────────────────────────────────────────┘
                                                         │
            ┌────────────────────────────────────────────┼────────────────────────────────────────────┐
            │                                            │                                            │
            ▼                                            ▼                                            ▼
┌───────────────────────┐              ┌───────────────────────┐              ┌───────────────────────┐
│   Availability Zone   │              │   Availability Zone   │              │   Availability Zone   │
│       us-east-1a      │              │       us-east-1b      │              │       us-east-1c      │
│                       │              │                       │              │                       │
│  ┌─────────────────┐  │              │  ┌─────────────────┐  │              │  ┌─────────────────┐  │
│  │   PRIMARY DB    │  │              │  │  READ REPLICA   │  │              │  │  READ REPLICA   │  │
│  │   (Writer)      │  │  ──────────► │  │  (Hot Standby)  │  │  ◄───────── │  │  (Async)        │  │
│  │                 │  │   Sync       │  │                 │  │    Async    │  │                 │  │
│  │ r6g.xlarge     │  │   Repl.      │  │ r6g.xlarge     │  │    Repl.    │  │ r6g.large      │  │
│  │ + TimescaleDB  │  │              │  │ + TimescaleDB  │  │              │  │ + TimescaleDB  │  │
│  └────────┬────────┘  │              │  └────────┬────────┘  │              │  └────────┬────────┘  │
│           │           │              │           │           │              │           │           │
│  ┌────────┴────────┐  │              │  ┌────────┴────────┐  │              │  ┌────────┴────────┐  │
│  │  EBS gp3 500GB  │  │              │  │  EBS gp3 500GB  │  │              │  │  EBS gp3 500GB  │  │
│  │  16,000 IOPS    │  │              │  │  16,000 IOPS    │  │              │  │  8,000 IOPS     │  │
│  │  Encrypted      │  │              │  │  Encrypted      │  │              │  │  Encrypted      │  │
│  └─────────────────┘  │              │  └─────────────────┘  │              │  └─────────────────┘  │
│                       │              │                       │              │                       │
│  ┌─────────────────┐  │              │  ┌─────────────────┐  │              │                       │
│  │  PgBouncer      │  │              │  │  PgBouncer      │  │              │                       │
│  │  (Connection    │  │              │  │  (Connection    │  │              │                       │
│  │   Pooler)       │  │              │  │   Pooler)       │  │              │                       │
│  └────────┬────────┘  │              │  └────────┬────────┘  │              │                       │
│           │           │              │           │           │              │                       │
└───────────┼───────────┘              └───────────┼───────────┘              └───────────────────────┘
            │                                      │
            └──────────────┬───────────────────────┘
                           │
                           ▼
            ┌───────────────────────────────────────────────────────────────────────────────────────┐
            │                              KUBERNETES CLUSTER (EKS)                                 │
            │                                                                                       │
            │  ┌─────────────────────────────────────────────────────────────────────────────────┐ │
            │  │                         APPLICATION LAYER                                        │ │
            │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │ │
            │  │  │ GL-CSRD-APP  │  │ GL-CBAM-APP  │  │ GL-EUDR-APP  │  │ GL-VCCI-APP  │        │ │
            │  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘        │ │
            │  │         └──────────────────┴─────────────────┴─────────────────┘                │ │
            │  │                                   │                                              │ │
            │  │                      ┌────────────┴────────────┐                                │ │
            │  │                      │     CONNECTION ROUTER    │                                │ │
            │  │                      │    (Read/Write Split)    │                                │ │
            │  │                      │  Writes → Primary        │                                │ │
            │  │                      │  Reads → Replicas        │                                │ │
            │  │                      └──────────────────────────┘                                │ │
            │  └─────────────────────────────────────────────────────────────────────────────────┘ │
            │                                                                                       │
            │  ┌─────────────────────────────────────────────────────────────────────────────────┐ │
            │  │                          BACKUP INFRASTRUCTURE                                   │ │
            │  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐              │ │
            │  │  │  pgBackRest      │  │  S3 Backup       │  │  Cross-Region    │              │ │
            │  │  │  Agent           │  │  Repository      │  │  S3 (us-west-2)  │              │ │
            │  │  └──────────────────┘  └──────────────────┘  └──────────────────┘              │ │
            │  └─────────────────────────────────────────────────────────────────────────────────┘ │
            └───────────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Aurora PostgreSQL vs RDS PostgreSQL Comparison

| Feature | Aurora PostgreSQL | RDS PostgreSQL | Recommendation |
|---------|------------------|----------------|----------------|
| **Performance** | 3x standard PostgreSQL | Standard | Aurora (for scale) |
| **Storage** | Auto-scaling to 128TB | Manual provisioning | Aurora |
| **Replication** | < 10ms lag | Potentially higher | Aurora |
| **Failover Time** | < 30 seconds | 60-120 seconds | Aurora |
| **TimescaleDB Support** | Supported (managed) | Supported (managed) | Both viable |
| **Multi-AZ Cost** | Built-in (no extra) | 2x instance cost | Aurora |
| **Serverless Option** | Aurora Serverless v2 | Not available | Aurora |
| **Cost (Production)** | ~$1,500-2,000/month | ~$800-1,200/month | RDS (budget) |
| **Backtrack** | Up to 72 hours | Not available | Aurora |

**Recommendation:** **Aurora PostgreSQL** for production due to superior HA, automatic failover, and built-in replication. Use **RDS PostgreSQL** for development/staging environments.

### 3.3 TimescaleDB Extension Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TimescaleDB Architecture                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                       HYPERTABLES (Time-Partitioned)                  │   │
│  │                                                                       │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐│   │
│  │  │  emission_measurements (hypertable)                              ││   │
│  │  │  Partitioned by: measurement_time (7-day chunks)                 ││   │
│  │  │  Retention: 7 years (365 chunks + compression)                   ││   │
│  │  │  Compression: After 30 days (10-40x ratio)                       ││   │
│  │  │                                                                   ││   │
│  │  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐        ││   │
│  │  │  │ Chunk 1   │ │ Chunk 2   │ │ Chunk 3   │ │ Chunk N   │  ...   ││   │
│  │  │  │ Jan 1-7   │ │ Jan 8-14  │ │ Jan 15-21 │ │ Latest    │        ││   │
│  │  │  │Compressed │ │Compressed │ │Compressed │ │Uncompressed│        ││   │
│  │  │  └───────────┘ └───────────┘ └───────────┘ └───────────┘        ││   │
│  │  └─────────────────────────────────────────────────────────────────┘│   │
│  │                                                                       │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐│   │
│  │  │  sensor_readings (hypertable)                                    ││   │
│  │  │  Partitioned by: reading_time (1-day chunks)                     ││   │
│  │  │  Retention: 2 years (730 chunks + compression)                   ││   │
│  │  │  Compression: After 7 days                                       ││   │
│  │  └─────────────────────────────────────────────────────────────────┘│   │
│  │                                                                       │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐│   │
│  │  │  calculation_results (hypertable)                                ││   │
│  │  │  Partitioned by: calculation_time (7-day chunks)                 ││   │
│  │  │  Retention: 7 years                                              ││   │
│  │  │  Compression: After 30 days                                      ││   │
│  │  └─────────────────────────────────────────────────────────────────┘│   │
│  │                                                                       │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐│   │
│  │  │  audit_logs (hypertable)                                         ││   │
│  │  │  Partitioned by: event_time (1-day chunks)                       ││   │
│  │  │  Retention: 10 years (immutable, append-only)                    ││   │
│  │  │  Compression: After 7 days                                       ││   │
│  │  └─────────────────────────────────────────────────────────────────┘│   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     CONTINUOUS AGGREGATES (Materialized)              │   │
│  │                                                                       │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐│   │
│  │  │  emissions_hourly_summary                                        ││   │
│  │  │  Refresh: Real-time (streaming)                                  ││   │
│  │  │  Aggregates: SUM, AVG, MIN, MAX, COUNT                           ││   │
│  │  └─────────────────────────────────────────────────────────────────┘│   │
│  │                                                                       │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐│   │
│  │  │  emissions_daily_summary                                         ││   │
│  │  │  Refresh: Every 15 minutes                                       ││   │
│  │  │  Aggregates: SUM, AVG, PERCENTILE                                ││   │
│  │  └─────────────────────────────────────────────────────────────────┘│   │
│  │                                                                       │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐│   │
│  │  │  emissions_quarterly_report (for CSRD/CBAM)                      ││   │
│  │  │  Refresh: Every hour                                             ││   │
│  │  │  Pre-computed for regulatory reporting                           ││   │
│  │  └─────────────────────────────────────────────────────────────────┘│   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.4 Connection Pooling with PgBouncer

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PgBouncer Connection Pooling                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Application Pods (100+ concurrent)                                          │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐           │
│  │App 1│ │App 2│ │App 3│ │App 4│ │App 5│ │App 6│ │App 7│ │ ... │           │
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘           │
│     └───────┴───────┴───────┴───────┼───────┴───────┴───────┘               │
│                                     │                                        │
│                                     ▼                                        │
│                    ┌────────────────────────────────┐                        │
│                    │         PgBouncer Pool         │                        │
│                    │                                │                        │
│                    │  Mode: transaction             │                        │
│                    │  Max Connections: 400          │                        │
│                    │  Default Pool Size: 50         │                        │
│                    │  Reserve Pool: 20              │                        │
│                    │  Max Client Conn: 5000         │                        │
│                    │  Server Lifetime: 3600s        │                        │
│                    │                                │                        │
│                    │  ┌──────────────────────────┐  │                        │
│                    │  │   Connection Pool        │  │                        │
│                    │  │   [################----] │  │                        │
│                    │  │   Active: 40 / Max: 50   │  │                        │
│                    │  └──────────────────────────┘  │                        │
│                    └────────────────┬───────────────┘                        │
│                                     │                                        │
│                    ┌────────────────┴───────────────┐                        │
│                    │         Write/Read Router      │                        │
│                    └────────────────┬───────────────┘                        │
│                                     │                                        │
│              ┌──────────────────────┼──────────────────────┐                 │
│              │                      │                      │                 │
│              ▼                      ▼                      ▼                 │
│     ┌────────────────┐     ┌────────────────┐     ┌────────────────┐        │
│     │   PRIMARY      │     │  REPLICA 1     │     │  REPLICA 2     │        │
│     │   (Writes)     │     │  (Reads)       │     │  (Reads)       │        │
│     │   max_conn=100 │     │  max_conn=200  │     │  max_conn=200  │        │
│     └────────────────┘     └────────────────┘     └────────────────┘        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**PgBouncer Configuration:**

```ini
# pgbouncer.ini
[databases]
greenlang_prod = host=primary.greenlang.internal port=5432 dbname=greenlang
greenlang_prod_ro = host=replica.greenlang.internal port=5432 dbname=greenlang

[pgbouncer]
listen_port = 6432
listen_addr = *
auth_type = scram-sha-256
auth_file = /etc/pgbouncer/userlist.txt

pool_mode = transaction
max_client_conn = 5000
default_pool_size = 50
reserve_pool_size = 20
reserve_pool_timeout = 5
max_db_connections = 400

server_lifetime = 3600
server_idle_timeout = 600
server_connect_timeout = 15
server_login_retry = 3

client_idle_timeout = 300
query_timeout = 300

# Health checks
server_check_query = SELECT 1
server_check_delay = 30

# Logging
log_connections = 1
log_disconnections = 1
log_pooler_errors = 1
stats_period = 60
```

### 3.5 Backup Strategy with pgBackRest

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        pgBackRest Backup Architecture                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    PRIMARY DATABASE SERVER                            │    │
│  │                                                                       │    │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐            │    │
│  │  │  Data Files   │  │  WAL Files    │  │  pgBackRest   │            │    │
│  │  │  /data/base   │  │  /data/pg_wal │  │  Agent        │            │    │
│  │  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘            │    │
│  │          └──────────────────┴──────────────────┘                      │    │
│  │                              │                                         │    │
│  └──────────────────────────────┼─────────────────────────────────────────┘    │
│                                 │                                              │
│                                 ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                    S3 BACKUP REPOSITORY (Primary)                        │  │
│  │                    s3://greenlang-prod-backups-us-east-1                 │  │
│  │                                                                          │  │
│  │  ┌─────────────┬─────────────┬─────────────┬─────────────┐              │  │
│  │  │   Full      │   Full      │   Full      │   Full      │ ...          │  │
│  │  │  Backup 1   │  Backup 2   │  Backup 3   │  Backup N   │              │  │
│  │  │  Weekly     │  Weekly     │  Weekly     │  Weekly     │              │  │
│  │  └──────┬──────┘└──────┬──────┘└──────┬──────┘└──────┬──────┘              │  │
│  │         │              │              │              │                    │  │
│  │  ┌──────┴──────┐┌──────┴──────┐┌──────┴──────┐┌──────┴──────┐            │  │
│  │  │ Incremental ││ Incremental ││ Incremental ││ Incremental │            │  │
│  │  │   Daily     ││   Daily     ││   Daily     ││   Daily     │            │  │
│  │  └─────────────┘└─────────────┘└─────────────┘└─────────────┘            │  │
│  │                                                                          │  │
│  │  ┌───────────────────────────────────────────────────────────────────┐  │  │
│  │  │  WAL Archive (Continuous)                                          │  │  │
│  │  │  Point-in-Time Recovery to any moment within retention             │  │  │
│  │  └───────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                          │  │
│  │  Retention: 35 days (5 full weekly + daily incrementals)                │  │
│  │  Encryption: AES-256                                                     │  │
│  │  Compression: zstd (level 3)                                             │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                 │                                              │
│                                 │ Cross-Region Replication                    │
│                                 ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                    S3 BACKUP REPOSITORY (DR)                             │  │
│  │                    s3://greenlang-prod-backups-us-west-2                 │  │
│  │                                                                          │  │
│  │  Retention: 90 days (quarterly compliance)                               │  │
│  │  Purpose: Disaster recovery, compliance, geographic redundancy           │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Technical Requirements

### 4.1 Database Sizing

| Resource | Primary | Replica 1 (Sync) | Replica 2 (Async) | Justification |
|----------|---------|------------------|-------------------|---------------|
| **Instance Type** | r6g.xlarge | r6g.xlarge | r6g.large | Memory-optimized for DB |
| **vCPU** | 4 | 4 | 2 | Write workload needs |
| **Memory** | 32 GB | 32 GB | 16 GB | Buffer pool, connections |
| **Storage** | 500 GB gp3 | 500 GB gp3 | 500 GB gp3 | Year 1 data + indexes |
| **Storage Max** | 2 TB (auto-scale) | 2 TB | 2 TB | Growth headroom |
| **IOPS** | 16,000 | 16,000 | 8,000 | Peak write throughput |
| **Throughput** | 500 MB/s | 500 MB/s | 250 MB/s | Bulk load operations |

### 4.2 PostgreSQL Configuration

```ini
# postgresql.conf (Production Optimized)

# Memory Configuration
shared_buffers = 8GB                 # 25% of RAM
effective_cache_size = 24GB          # 75% of RAM
work_mem = 256MB                     # Per-operation memory
maintenance_work_mem = 2GB           # For VACUUM, CREATE INDEX

# Write Ahead Log (WAL)
wal_level = replica                  # Required for replication
max_wal_size = 4GB                   # Maximum WAL size
min_wal_size = 1GB                   # Minimum WAL size
wal_compression = on                 # Compress WAL
archive_mode = on                    # Enable WAL archiving
archive_command = 'pgbackrest --stanza=greenlang archive-push %p'

# Replication
max_wal_senders = 10                 # Max replication connections
wal_keep_size = 2GB                  # WAL retention for replicas
synchronous_commit = remote_apply    # Synchronous replication
synchronous_standby_names = 'replica1'

# Connections
max_connections = 500                # Direct connections (use PgBouncer)
superuser_reserved_connections = 5

# Query Planning
random_page_cost = 1.1               # SSD storage
effective_io_concurrency = 200       # gp3 SSD
default_statistics_target = 500      # Better query plans

# Background Writer
bgwriter_lru_maxpages = 400
bgwriter_lru_multiplier = 4.0

# Autovacuum
autovacuum = on
autovacuum_max_workers = 4
autovacuum_naptime = 30s
autovacuum_vacuum_cost_delay = 2ms

# Logging
log_destination = 'csvlog'
logging_collector = on
log_directory = '/var/log/postgresql'
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
log_min_duration_statement = 1000    # Log queries > 1 second
log_checkpoints = on
log_connections = on
log_disconnections = on
log_lock_waits = on
log_temp_files = 0

# TimescaleDB
shared_preload_libraries = 'timescaledb'
timescaledb.max_background_workers = 8
timescaledb.last_tuned = '2026-02-03'
```

### 4.3 Replication Configuration

| Parameter | Sync Replica (AZ-1b) | Async Replica (AZ-1c) |
|-----------|---------------------|----------------------|
| **Mode** | Synchronous | Asynchronous |
| **Purpose** | Zero data loss failover | Read scaling, DR |
| **Max Lag** | 0 bytes | < 1 MB |
| **Failover Priority** | 1 (First) | 2 (Second) |
| **Promotion Time** | Automatic (< 30s) | Manual |

### 4.4 Failover Requirements

| Metric | Target | Implementation |
|--------|--------|----------------|
| **RTO** | < 30 seconds | Automatic failover with Patroni |
| **RPO** | 0 (zero data loss) | Synchronous replication to standby |
| **Detection Time** | < 10 seconds | Patroni health checks every 5s |
| **DNS Propagation** | < 5 seconds | Route53 health checks + failover |
| **Application Reconnect** | < 10 seconds | PgBouncer connection retry |

**Failover Architecture:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Automatic Failover with Patroni                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        etcd Cluster (DCS)                            │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                           │    │
│  │  │  etcd-1  │  │  etcd-2  │  │  etcd-3  │  (Leader election)        │    │
│  │  │  (AZ-1a) │  │  (AZ-1b) │  │  (AZ-1c) │                           │    │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘                           │    │
│  │       └─────────────┴─────────────┘                                  │    │
│  └─────────────────────────┬───────────────────────────────────────────┘    │
│                            │ Leadership info                                 │
│                            ▼                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        Patroni Cluster                               │    │
│  │                                                                      │    │
│  │  ┌────────────────────┐      ┌────────────────────┐                 │    │
│  │  │  Primary (Leader)  │      │  Replica (Standby) │                 │    │
│  │  │  ┌──────────────┐  │      │  ┌──────────────┐  │                 │    │
│  │  │  │   Patroni    │  │      │  │   Patroni    │  │                 │    │
│  │  │  │   Agent      │  │      │  │   Agent      │  │                 │    │
│  │  │  └──────┬───────┘  │      │  └──────┬───────┘  │                 │    │
│  │  │         │          │      │         │          │                 │    │
│  │  │  ┌──────▼───────┐  │      │  ┌──────▼───────┐  │                 │    │
│  │  │  │  PostgreSQL  │──┼──────┼─▶│  PostgreSQL  │  │                 │    │
│  │  │  │  (Primary)   │  │ Sync │  │  (Replica)   │  │                 │    │
│  │  │  └──────────────┘  │ Repl │  └──────────────┘  │                 │    │
│  │  └────────────────────┘      └────────────────────┘                 │    │
│  │                                                                      │    │
│  │  Failover Sequence:                                                  │    │
│  │  1. Primary health check fails (5s timeout)                         │    │
│  │  2. Patroni detects failure (< 10s)                                 │    │
│  │  3. etcd leader lock released                                       │    │
│  │  4. Replica promoted to Primary (< 5s)                              │    │
│  │  5. DNS updated via Route53 (< 5s)                                  │    │
│  │  6. PgBouncer reconnects (< 10s)                                    │    │
│  │  ──────────────────────────────────────                             │    │
│  │  Total RTO: < 30 seconds                                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.5 TimescaleDB Hypertable Definitions

```sql
-- ============================================================
-- HYPERTABLE 1: Emission Measurements
-- Purpose: Store all emission data from facilities, processes
-- ============================================================

CREATE TABLE emission_measurements (
    id                    BIGSERIAL,
    measurement_time      TIMESTAMPTZ NOT NULL,
    organization_id       UUID NOT NULL,
    facility_id           UUID NOT NULL,
    source_id             UUID,

    -- Emission data
    scope                 VARCHAR(20) NOT NULL,  -- scope_1, scope_2, scope_3
    category              VARCHAR(50),           -- Scope 3 category (1-15)
    activity_type         VARCHAR(100) NOT NULL,
    activity_value        DECIMAL(20, 6) NOT NULL,
    activity_unit         VARCHAR(50) NOT NULL,

    -- Calculated emissions
    co2_kg                DECIMAL(20, 6),
    ch4_kg                DECIMAL(20, 6),
    n2o_kg                DECIMAL(20, 6),
    hfc_kg                DECIMAL(20, 6),
    pfc_kg                DECIMAL(20, 6),
    sf6_kg                DECIMAL(20, 6),
    nf3_kg                DECIMAL(20, 6),
    co2_equivalent_kg     DECIMAL(20, 6) NOT NULL,

    -- Calculation metadata
    emission_factor_id    UUID NOT NULL,
    emission_factor_value DECIMAL(20, 10) NOT NULL,
    emission_factor_unit  VARCHAR(50) NOT NULL,
    calculation_method    VARCHAR(50) NOT NULL,
    uncertainty_pct       DECIMAL(5, 2),
    data_quality_score    SMALLINT,

    -- Audit fields
    created_at            TIMESTAMPTZ DEFAULT NOW(),
    created_by            UUID,
    source_document_id    UUID,
    lineage_hash          VARCHAR(64) NOT NULL,

    PRIMARY KEY (measurement_time, id)
);

-- Convert to hypertable with 7-day chunks
SELECT create_hypertable(
    'emission_measurements',
    'measurement_time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Enable compression after 30 days
ALTER TABLE emission_measurements SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'organization_id, facility_id, scope',
    timescaledb.compress_orderby = 'measurement_time DESC'
);

SELECT add_compression_policy('emission_measurements', INTERVAL '30 days');

-- Retention policy: 7 years
SELECT add_retention_policy('emission_measurements', INTERVAL '7 years');

-- Indexes for common queries
CREATE INDEX idx_emissions_org_facility_time
    ON emission_measurements (organization_id, facility_id, measurement_time DESC);
CREATE INDEX idx_emissions_scope_time
    ON emission_measurements (scope, measurement_time DESC);


-- ============================================================
-- HYPERTABLE 2: Sensor Readings
-- Purpose: High-frequency IoT data from meters, sensors
-- ============================================================

CREATE TABLE sensor_readings (
    id                    BIGSERIAL,
    reading_time          TIMESTAMPTZ NOT NULL,
    organization_id       UUID NOT NULL,
    facility_id           UUID NOT NULL,
    device_id             UUID NOT NULL,

    -- Reading data
    metric_type           VARCHAR(50) NOT NULL,  -- electricity, gas, water, etc.
    value                 DECIMAL(20, 6) NOT NULL,
    unit                  VARCHAR(30) NOT NULL,
    quality_flag          SMALLINT DEFAULT 0,

    -- Device metadata
    device_type           VARCHAR(50),
    meter_serial          VARCHAR(100),

    PRIMARY KEY (reading_time, id)
);

SELECT create_hypertable(
    'sensor_readings',
    'reading_time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

ALTER TABLE sensor_readings SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'organization_id, facility_id, device_id',
    timescaledb.compress_orderby = 'reading_time DESC'
);

SELECT add_compression_policy('sensor_readings', INTERVAL '7 days');
SELECT add_retention_policy('sensor_readings', INTERVAL '2 years');


-- ============================================================
-- HYPERTABLE 3: Calculation Results
-- Purpose: Store emissions calculation outputs with full lineage
-- ============================================================

CREATE TABLE calculation_results (
    id                    BIGSERIAL,
    calculation_time      TIMESTAMPTZ NOT NULL,
    organization_id       UUID NOT NULL,

    -- Calculation context
    report_type           VARCHAR(50) NOT NULL,  -- CSRD, CBAM, GHG_INVENTORY
    reporting_period_start TIMESTAMPTZ NOT NULL,
    reporting_period_end   TIMESTAMPTZ NOT NULL,

    -- Results
    scope_1_total_kg      DECIMAL(20, 6),
    scope_2_location_kg   DECIMAL(20, 6),
    scope_2_market_kg     DECIMAL(20, 6),
    scope_3_total_kg      DECIMAL(20, 6),
    total_emissions_kg    DECIMAL(20, 6) NOT NULL,

    -- Per-category breakdowns (JSONB for flexibility)
    category_breakdown    JSONB,
    facility_breakdown    JSONB,

    -- Metadata
    calculation_version   VARCHAR(20) NOT NULL,
    assumptions_set_id    UUID NOT NULL,
    emission_factor_set_id UUID NOT NULL,

    -- Audit
    created_by            UUID,
    approved_by           UUID,
    approved_at           TIMESTAMPTZ,
    lineage_hash          VARCHAR(64) NOT NULL,

    PRIMARY KEY (calculation_time, id)
);

SELECT create_hypertable(
    'calculation_results',
    'calculation_time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

ALTER TABLE calculation_results SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'organization_id, report_type',
    timescaledb.compress_orderby = 'calculation_time DESC'
);

SELECT add_compression_policy('calculation_results', INTERVAL '30 days');
SELECT add_retention_policy('calculation_results', INTERVAL '7 years');


-- ============================================================
-- HYPERTABLE 4: Audit Logs
-- Purpose: Immutable audit trail for compliance
-- ============================================================

CREATE TABLE audit_logs (
    id                    BIGSERIAL,
    event_time            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    organization_id       UUID NOT NULL,

    -- Event details
    event_type            VARCHAR(50) NOT NULL,
    event_category        VARCHAR(50) NOT NULL,
    resource_type         VARCHAR(50) NOT NULL,
    resource_id           UUID,

    -- Actor
    actor_id              UUID,
    actor_type            VARCHAR(30) NOT NULL,  -- user, system, api_key
    actor_ip              INET,

    -- Change details
    action                VARCHAR(30) NOT NULL,  -- CREATE, UPDATE, DELETE, VIEW
    old_values            JSONB,
    new_values            JSONB,

    -- Context
    request_id            UUID,
    session_id            UUID,
    user_agent            TEXT,

    -- Integrity
    hash                  VARCHAR(64) NOT NULL,
    previous_hash         VARCHAR(64),

    PRIMARY KEY (event_time, id)
);

SELECT create_hypertable(
    'audit_logs',
    'event_time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Append-only: No updates or deletes allowed
CREATE OR REPLACE FUNCTION prevent_audit_modification()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'Audit logs are immutable and cannot be modified or deleted';
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER audit_logs_immutable
    BEFORE UPDATE OR DELETE ON audit_logs
    FOR EACH ROW EXECUTE FUNCTION prevent_audit_modification();

ALTER TABLE audit_logs SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'organization_id, event_category',
    timescaledb.compress_orderby = 'event_time DESC'
);

SELECT add_compression_policy('audit_logs', INTERVAL '7 days');
SELECT add_retention_policy('audit_logs', INTERVAL '10 years');


-- ============================================================
-- CONTINUOUS AGGREGATES: Pre-computed summaries
-- ============================================================

-- Hourly emissions summary
CREATE MATERIALIZED VIEW emissions_hourly_summary
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', measurement_time) AS bucket,
    organization_id,
    facility_id,
    scope,
    SUM(co2_equivalent_kg) AS total_co2e_kg,
    COUNT(*) AS measurement_count,
    AVG(uncertainty_pct) AS avg_uncertainty,
    MIN(data_quality_score) AS min_quality,
    MAX(data_quality_score) AS max_quality
FROM emission_measurements
GROUP BY bucket, organization_id, facility_id, scope
WITH NO DATA;

SELECT add_continuous_aggregate_policy('emissions_hourly_summary',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');


-- Daily emissions summary
CREATE MATERIALIZED VIEW emissions_daily_summary
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', measurement_time) AS bucket,
    organization_id,
    facility_id,
    scope,
    category,
    SUM(co2_equivalent_kg) AS total_co2e_kg,
    COUNT(*) AS measurement_count,
    AVG(uncertainty_pct) AS avg_uncertainty,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY co2_equivalent_kg) AS p95_co2e
FROM emission_measurements
GROUP BY bucket, organization_id, facility_id, scope, category
WITH NO DATA;

SELECT add_continuous_aggregate_policy('emissions_daily_summary',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '15 minutes');


-- Quarterly emissions for regulatory reporting (CSRD, CBAM)
CREATE MATERIALIZED VIEW emissions_quarterly_report
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('3 months', measurement_time) AS quarter,
    organization_id,
    scope,
    category,
    SUM(co2_equivalent_kg) AS total_co2e_kg,
    COUNT(DISTINCT facility_id) AS facility_count,
    COUNT(*) AS measurement_count,
    AVG(data_quality_score) AS avg_quality
FROM emission_measurements
GROUP BY quarter, organization_id, scope, category
WITH NO DATA;

SELECT add_continuous_aggregate_policy('emissions_quarterly_report',
    start_offset => INTERVAL '6 months',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 hour');
```

---

## 5. Security Requirements

### 5.1 Encryption at Rest (KMS)

| Component | Encryption | Key Management |
|-----------|------------|----------------|
| **EBS Volumes** | AES-256 | AWS KMS CMK |
| **S3 Backups** | AES-256-GCM | AWS KMS CMK |
| **RDS Storage** | AES-256 | AWS KMS CMK |
| **Backup Archive** | AES-256 | Customer-managed key |

**KMS Key Policy:**

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "Enable IAM User Permissions",
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::ACCOUNT_ID:root"
            },
            "Action": "kms:*",
            "Resource": "*"
        },
        {
            "Sid": "Allow RDS Service",
            "Effect": "Allow",
            "Principal": {
                "Service": "rds.amazonaws.com"
            },
            "Action": [
                "kms:Encrypt",
                "kms:Decrypt",
                "kms:GenerateDataKey*",
                "kms:DescribeKey"
            ],
            "Resource": "*"
        },
        {
            "Sid": "Allow Database Admins",
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::ACCOUNT_ID:role/DatabaseAdminRole"
            },
            "Action": [
                "kms:Encrypt",
                "kms:Decrypt",
                "kms:GenerateDataKey*",
                "kms:DescribeKey",
                "kms:CreateGrant"
            ],
            "Resource": "*"
        }
    ]
}
```

### 5.2 Encryption in Transit (TLS 1.3)

```yaml
# PostgreSQL SSL Configuration
ssl: on
ssl_min_protocol_version: TLSv1.3
ssl_ciphers: 'TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256:TLS_AES_128_GCM_SHA256'
ssl_cert_file: '/etc/ssl/certs/server.crt'
ssl_key_file: '/etc/ssl/private/server.key'
ssl_ca_file: '/etc/ssl/certs/ca.crt'

# Client connection string (enforce SSL)
connection_string: "host=db.greenlang.io dbname=greenlang sslmode=verify-full sslrootcert=/etc/ssl/certs/rds-ca-2019-root.pem"
```

### 5.3 IAM Authentication

```yaml
# IAM Database Authentication
iam_authentication:
  enabled: true
  users:
    - name: app_service
      roles: [app_read, app_write]
      iam_role: arn:aws:iam::ACCOUNT_ID:role/EKS-Pod-DatabaseAccess

    - name: readonly_analytics
      roles: [analytics_read]
      iam_role: arn:aws:iam::ACCOUNT_ID:role/Analytics-DatabaseAccess

    - name: admin
      roles: [rds_superuser]
      iam_role: arn:aws:iam::ACCOUNT_ID:role/DatabaseAdminRole

# IAM Policy for EKS Pod Database Access
iam_policy:
  Version: "2012-10-17"
  Statement:
    - Effect: Allow
      Action:
        - rds-db:connect
      Resource:
        - "arn:aws:rds-db:us-east-1:ACCOUNT_ID:dbuser:cluster-*/app_service"
```

### 5.4 Secrets Rotation

```yaml
# AWS Secrets Manager Configuration
secrets:
  greenlang/prod/database/master:
    description: "Database master credentials"
    rotation:
      enabled: true
      schedule: "rate(30 days)"
      lambda: arn:aws:lambda:us-east-1:ACCOUNT_ID:function:SecretsRotation

  greenlang/prod/database/app:
    description: "Application database credentials"
    rotation:
      enabled: true
      schedule: "rate(7 days)"
      automatic_rotation: true

  greenlang/prod/database/readonly:
    description: "Read-only database credentials"
    rotation:
      enabled: true
      schedule: "rate(7 days)"

# Rotation Lambda Function
rotation_lambda:
  function_name: SecretsRotation-PostgreSQL
  handler: lambda_function.lambda_handler
  runtime: python3.11
  timeout: 30
  environment:
    SECRETS_MANAGER_ENDPOINT: https://secretsmanager.us-east-1.amazonaws.com
```

### 5.5 Network Isolation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Network Security Architecture                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         VPC (10.0.0.0/16)                            │    │
│  │                                                                      │    │
│  │  ┌────────────────────────────────────────────────────────────────┐ │    │
│  │  │  Public Subnets (10.0.1.0/24, 10.0.2.0/24, 10.0.3.0/24)       │ │    │
│  │  │  - NAT Gateways                                                 │ │    │
│  │  │  - Load Balancers                                               │ │    │
│  │  └────────────────────────────────────────────────────────────────┘ │    │
│  │                                                                      │    │
│  │  ┌────────────────────────────────────────────────────────────────┐ │    │
│  │  │  Private Subnets (10.0.11.0/24, 10.0.12.0/24, 10.0.13.0/24)   │ │    │
│  │  │  - EKS Worker Nodes                                             │ │    │
│  │  │  - Application Pods                                             │ │    │
│  │  │                                                                  │ │    │
│  │  │  Security Group: sg-eks-nodes                                   │ │    │
│  │  │  Outbound: Allow 5432 to sg-database                           │ │    │
│  │  └────────────────────────────────────────────────────────────────┘ │    │
│  │                                    │                                 │    │
│  │                                    │ Port 5432 (TLS only)           │    │
│  │                                    ▼                                 │    │
│  │  ┌────────────────────────────────────────────────────────────────┐ │    │
│  │  │  Database Subnets (10.0.21.0/24, 10.0.22.0/24, 10.0.23.0/24)  │ │    │
│  │  │  - PostgreSQL Primary                                          │ │    │
│  │  │  - PostgreSQL Replicas                                         │ │    │
│  │  │  - PgBouncer                                                   │ │    │
│  │  │                                                                 │ │    │
│  │  │  Security Group: sg-database                                   │ │    │
│  │  │  Inbound:                                                      │ │    │
│  │  │    - Port 5432 from sg-eks-nodes                              │ │    │
│  │  │    - Port 5432 from sg-database (replication)                 │ │    │
│  │  │  Outbound:                                                     │ │    │
│  │  │    - Port 443 to S3 VPC Endpoint (backups)                    │ │    │
│  │  │    - Port 5432 to sg-database (replication)                   │ │    │
│  │  │                                                                 │ │    │
│  │  │  NO PUBLIC IP ADDRESSES                                        │ │    │
│  │  │  NO INTERNET GATEWAY ACCESS                                    │ │    │
│  │  └────────────────────────────────────────────────────────────────┘ │    │
│  │                                                                      │    │
│  │  ┌────────────────────────────────────────────────────────────────┐ │    │
│  │  │  VPC Endpoints (Private Link)                                  │ │    │
│  │  │  - com.amazonaws.us-east-1.s3 (Gateway)                       │ │    │
│  │  │  - com.amazonaws.us-east-1.secretsmanager (Interface)         │ │    │
│  │  │  - com.amazonaws.us-east-1.kms (Interface)                    │ │    │
│  │  │  - com.amazonaws.us-east-1.monitoring (Interface)             │ │    │
│  │  └────────────────────────────────────────────────────────────────┘ │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Security Group Rules:**

```hcl
# Database Security Group
resource "aws_security_group" "database" {
  name_prefix = "greenlang-database-"
  vpc_id      = aws_vpc.main.id

  # Inbound: PostgreSQL from EKS nodes only
  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_nodes.id]
    description     = "PostgreSQL from EKS nodes"
  }

  # Inbound: Replication between database nodes
  ingress {
    from_port = 5432
    to_port   = 5432
    protocol  = "tcp"
    self      = true
    description = "PostgreSQL replication"
  }

  # Outbound: S3 for backups (via VPC endpoint)
  egress {
    from_port       = 443
    to_port         = 443
    protocol        = "tcp"
    prefix_list_ids = [aws_vpc_endpoint.s3.prefix_list_id]
    description     = "S3 for backups"
  }

  # Outbound: Replication
  egress {
    from_port = 5432
    to_port   = 5432
    protocol  = "tcp"
    self      = true
    description = "PostgreSQL replication"
  }

  tags = {
    Name = "greenlang-database"
  }
}
```

---

## 6. Monitoring and Alerting

### 6.1 Key Metrics

| Metric | Source | Threshold (Warning) | Threshold (Critical) |
|--------|--------|---------------------|----------------------|
| **Connection Count** | pg_stat_activity | > 350 | > 450 |
| **Connection Wait Time** | PgBouncer | > 100ms | > 500ms |
| **Replication Lag (bytes)** | pg_stat_replication | > 100KB | > 1MB |
| **Replication Lag (seconds)** | Calculated | > 1s | > 5s |
| **Query Latency (p99)** | pg_stat_statements | > 500ms | > 2000ms |
| **Transaction Rate** | pg_stat_database | < 100/s (drop) | < 50/s |
| **Buffer Cache Hit Ratio** | pg_stat_bgwriter | < 95% | < 90% |
| **Disk Usage** | EBS Metrics | > 70% | > 85% |
| **IOPS Utilization** | EBS Metrics | > 70% | > 90% |
| **CPU Utilization** | CloudWatch | > 70% | > 85% |
| **Memory Utilization** | CloudWatch | > 80% | > 90% |
| **WAL Generation Rate** | pg_stat_wal | > 100MB/min | > 500MB/min |
| **Vacuum Lag** | pg_stat_user_tables | > 10M dead tuples | > 50M dead tuples |
| **Lock Waits** | pg_stat_activity | > 5 | > 20 |
| **Long Running Queries** | pg_stat_activity | > 30s | > 120s |

### 6.2 Alert Configuration

```yaml
# Prometheus Alerting Rules
groups:
  - name: postgresql_alerts
    rules:
      # Connection Alerts
      - alert: PostgreSQLHighConnections
        expr: pg_stat_activity_count > 350
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High connection count on {{ $labels.instance }}"
          description: "Connection count is {{ $value }} (threshold: 350)"

      - alert: PostgreSQLConnectionsCritical
        expr: pg_stat_activity_count > 450
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Critical connection count on {{ $labels.instance }}"
          description: "Connection count is {{ $value }} (max: 500)"

      # Replication Alerts
      - alert: PostgreSQLReplicationLag
        expr: pg_replication_lag_bytes > 1048576
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Replication lag critical on {{ $labels.instance }}"
          description: "Replication lag is {{ $value | humanize1024 }}B"

      - alert: PostgreSQLReplicaDown
        expr: pg_replication_is_replica == 0 AND pg_up == 1
        for: 30s
        labels:
          severity: critical
        annotations:
          summary: "PostgreSQL replica stopped replicating"
          description: "Replica {{ $labels.instance }} is no longer replicating"

      # Performance Alerts
      - alert: PostgreSQLSlowQueries
        expr: rate(pg_stat_statements_seconds_total[5m]) / rate(pg_stat_statements_calls_total[5m]) > 0.5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Slow queries detected"
          description: "Average query time is {{ $value }}s"

      - alert: PostgreSQLLowCacheHitRatio
        expr: pg_stat_database_blks_hit / (pg_stat_database_blks_hit + pg_stat_database_blks_read) < 0.95
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Low buffer cache hit ratio"
          description: "Cache hit ratio is {{ $value | humanizePercentage }}"

      # Disk Alerts
      - alert: PostgreSQLDiskSpaceWarning
        expr: (node_filesystem_avail_bytes{mountpoint="/data"} / node_filesystem_size_bytes{mountpoint="/data"}) < 0.30
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Database disk space running low"
          description: "Only {{ $value | humanizePercentage }} disk space remaining"

      - alert: PostgreSQLDiskSpaceCritical
        expr: (node_filesystem_avail_bytes{mountpoint="/data"} / node_filesystem_size_bytes{mountpoint="/data"}) < 0.15
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Database disk space critical"
          description: "Only {{ $value | humanizePercentage }} disk space remaining"

      # TimescaleDB Alerts
      - alert: TimescaleDBChunkCompressionFailing
        expr: increase(timescaledb_compression_errors_total[1h]) > 0
        labels:
          severity: warning
        annotations:
          summary: "TimescaleDB chunk compression failing"
          description: "{{ $value }} compression errors in the last hour"

      - alert: TimescaleDBContinuousAggregateLag
        expr: timescaledb_continuous_aggregate_refresh_lag_seconds > 3600
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Continuous aggregate refresh lagging"
          description: "Aggregate {{ $labels.view_name }} is {{ $value }}s behind"
```

### 6.3 Grafana Dashboards

```yaml
# Dashboard: PostgreSQL Overview
dashboards:
  - name: "PostgreSQL Overview"
    uid: "pg-overview"
    panels:
      - title: "Connection Pool Status"
        type: gauge
        queries:
          - expr: pg_stat_activity_count
            legendFormat: "Active Connections"
        thresholds:
          - color: green
            value: 0
          - color: yellow
            value: 350
          - color: red
            value: 450

      - title: "Replication Lag"
        type: timeseries
        queries:
          - expr: pg_replication_lag_bytes
            legendFormat: "Lag (bytes)"
          - expr: pg_replication_lag_seconds
            legendFormat: "Lag (seconds)"

      - title: "Query Latency Distribution"
        type: heatmap
        queries:
          - expr: histogram_quantile(0.99, rate(pg_stat_statements_seconds_bucket[5m]))

      - title: "Transaction Rate"
        type: timeseries
        queries:
          - expr: rate(pg_stat_database_xact_commit[5m]) + rate(pg_stat_database_xact_rollback[5m])
            legendFormat: "Transactions/sec"

  - name: "TimescaleDB Performance"
    uid: "timescale-perf"
    panels:
      - title: "Hypertable Sizes"
        type: bargauge
        queries:
          - expr: timescaledb_hypertable_size_bytes
            legendFormat: "{{ hypertable_name }}"

      - title: "Compression Ratio"
        type: stat
        queries:
          - expr: timescaledb_compression_ratio

      - title: "Chunk Count by Hypertable"
        type: piechart
        queries:
          - expr: timescaledb_hypertable_chunk_count

      - title: "Continuous Aggregate Refresh Status"
        type: table
        queries:
          - expr: timescaledb_continuous_aggregate_last_refresh_time
```

### 6.4 PagerDuty Integration

```yaml
# PagerDuty Integration
pagerduty:
  integration_key: "${PAGERDUTY_INTEGRATION_KEY}"

  routing_rules:
    - severity: critical
      service: database-critical
      urgency: high
      escalation_policy: database-oncall

    - severity: warning
      service: database-warnings
      urgency: low
      escalation_policy: database-oncall-delayed

  oncall_schedule:
    name: "Database On-Call"
    rotation: weekly
    teams:
      - infrastructure
      - platform

  escalation_policy:
    - name: database-oncall
      rules:
        - delay: 0
          targets:
            - type: schedule
              id: database-primary
        - delay: 15
          targets:
            - type: user
              id: database-manager
        - delay: 30
          targets:
            - type: user
              id: infrastructure-lead
```

---

## 7. Backup and Recovery

### 7.1 Backup Schedule

| Backup Type | Frequency | Retention | Storage | Encryption |
|-------------|-----------|-----------|---------|------------|
| **Full Backup** | Weekly (Sunday 02:00 UTC) | 5 weeks | S3 (us-east-1) | AES-256 |
| **Incremental Backup** | Daily (02:00 UTC) | 35 days | S3 (us-east-1) | AES-256 |
| **WAL Archive** | Continuous (every segment) | 35 days | S3 (us-east-1) | AES-256 |
| **Cross-Region Backup** | Weekly | 90 days | S3 (us-west-2) | AES-256 |
| **Long-Term Archive** | Monthly | 7 years | S3 Glacier | AES-256 |

### 7.2 pgBackRest Configuration

```ini
# /etc/pgbackrest/pgbackrest.conf

[global]
repo1-type=s3
repo1-s3-bucket=greenlang-prod-backups-us-east-1
repo1-s3-region=us-east-1
repo1-s3-endpoint=s3.us-east-1.amazonaws.com
repo1-path=/pgbackrest
repo1-retention-full=5
repo1-retention-diff=35

repo2-type=s3
repo2-s3-bucket=greenlang-prod-backups-us-west-2
repo2-s3-region=us-west-2
repo2-s3-endpoint=s3.us-west-2.amazonaws.com
repo2-path=/pgbackrest
repo2-retention-full=12
repo2-retention-diff=90

# Encryption
repo1-cipher-type=aes-256-cbc
repo2-cipher-type=aes-256-cbc

# Compression
compress-type=zstd
compress-level=3

# Parallelism
process-max=4
protocol-timeout=3600

# Logging
log-level-console=info
log-level-file=detail
log-path=/var/log/pgbackrest

[greenlang]
pg1-path=/var/lib/postgresql/15/data
pg1-port=5432
pg1-user=postgres
pg1-database=greenlang

# Stanza options
archive-push-queue-max=2GB
archive-async=y
```

**Backup Cron Schedule:**

```bash
# /etc/cron.d/pgbackrest

# Full backup every Sunday at 02:00 UTC
0 2 * * 0 postgres pgbackrest --stanza=greenlang --type=full backup

# Incremental backup daily at 02:00 UTC (except Sunday)
0 2 * * 1-6 postgres pgbackrest --stanza=greenlang --type=incr backup

# Verify backup integrity weekly
0 6 * * 0 postgres pgbackrest --stanza=greenlang verify

# Cross-region copy weekly
0 8 * * 0 postgres pgbackrest --stanza=greenlang --repo=2 backup --type=full

# Archive WAL cleanup
0 4 * * * postgres pgbackrest --stanza=greenlang expire
```

### 7.3 Point-in-Time Recovery

```bash
#!/bin/bash
# restore_pitr.sh - Point-in-Time Recovery Script

set -euo pipefail

# Configuration
RESTORE_TARGET="${1:-}"
RECOVERY_TARGET_TIME="${2:-}"
STANZA="greenlang"
DATA_DIR="/var/lib/postgresql/15/data"
BACKUP_DIR="/var/lib/postgresql/15/backup"

if [ -z "$RESTORE_TARGET" ]; then
    echo "Usage: $0 <restore_target> [recovery_target_time]"
    echo "  restore_target: primary|replica"
    echo "  recovery_target_time: YYYY-MM-DD HH:MM:SS (optional, for PITR)"
    exit 1
fi

echo "=== Point-in-Time Recovery ==="
echo "Target: $RESTORE_TARGET"
echo "Recovery Time: ${RECOVERY_TARGET_TIME:-latest}"

# Stop PostgreSQL
echo "Stopping PostgreSQL..."
systemctl stop postgresql

# Backup current data directory
echo "Backing up current data directory..."
mv "$DATA_DIR" "${BACKUP_DIR}/data_$(date +%Y%m%d_%H%M%S)"

# Perform restore
echo "Starting pgBackRest restore..."
if [ -n "$RECOVERY_TARGET_TIME" ]; then
    pgbackrest --stanza=$STANZA \
        --delta \
        --type=time \
        --target="$RECOVERY_TARGET_TIME" \
        --target-action=promote \
        restore
else
    pgbackrest --stanza=$STANZA \
        --delta \
        restore
fi

# Start PostgreSQL
echo "Starting PostgreSQL..."
systemctl start postgresql

# Wait for recovery
echo "Waiting for recovery to complete..."
until pg_isready -q; do
    sleep 1
done

echo "=== Recovery Complete ==="
echo "Database is now available."
```

### 7.4 Cross-Region Backup Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Cross-Region Backup Architecture                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        PRIMARY REGION (us-east-1)                       │ │
│  │                                                                         │ │
│  │  ┌─────────────────┐                    ┌─────────────────────────────┐│ │
│  │  │  PostgreSQL     │  Continuous WAL    │  S3 Backup Repository       ││ │
│  │  │  Primary        │ ─────────────────► │  greenlang-backups-east     ││ │
│  │  │                 │  Archive           │                             ││ │
│  │  └─────────────────┘                    │  - Full backups (weekly)    ││ │
│  │                                          │  - Incremental (daily)      ││ │
│  │                                          │  - WAL archive (continuous) ││ │
│  │                                          │  - Retention: 35 days       ││ │
│  │                                          └──────────────┬──────────────┘│ │
│  └─────────────────────────────────────────────────────────┼────────────────┘ │
│                                                            │                  │
│                                                            │ S3 Cross-Region │
│                                                            │ Replication     │
│                                                            │ (Async)         │
│                                                            ▼                  │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                        DR REGION (us-west-2)                            │  │
│  │                                                                         │  │
│  │  ┌─────────────────────────────┐   ┌─────────────────────────────────┐ │  │
│  │  │  S3 Backup Repository       │   │  Standby PostgreSQL (Cold)      │ │  │
│  │  │  greenlang-backups-west     │   │  (Provisioned only during DR)   │ │  │
│  │  │                             │   │                                  │ │  │
│  │  │  - Full backups (weekly)    │   │  RTO: 2-4 hours                  │ │  │
│  │  │  - Retention: 90 days       │   │  RPO: < 15 minutes               │ │  │
│  │  │  - Glacier (7 year archive) │   │  (based on WAL archive delay)   │ │  │
│  │  └─────────────────────────────┘   └─────────────────────────────────┘ │  │
│  │                                                                         │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

### 7.5 Disaster Recovery Procedures

```yaml
# DR Runbook: Complete Regional Failure

disaster_recovery:
  scenario: "Primary Region (us-east-1) Complete Failure"
  rto: "4 hours"
  rpo: "15 minutes"

  steps:
    - id: 1
      name: "Declare Disaster"
      owner: "Incident Commander"
      duration: "5 minutes"
      actions:
        - Confirm us-east-1 region is unavailable
        - Declare DR event in PagerDuty
        - Notify stakeholders

    - id: 2
      name: "Provision DR Infrastructure"
      owner: "Infrastructure Team"
      duration: "30 minutes"
      actions:
        - Run Terraform in us-west-2
        - Provision RDS instance (r6g.xlarge)
        - Configure security groups
        - Set up network connectivity

    - id: 3
      name: "Restore Database from Backup"
      owner: "Database Team"
      duration: "2-3 hours"
      actions:
        - Identify most recent backup in us-west-2 S3
        - Run pgBackRest restore
        - Apply WAL archive to latest point
        - Verify data integrity
      commands:
        - "pgbackrest --stanza=greenlang --repo=2 --type=time --target='2026-02-03 12:00:00+00' restore"

    - id: 4
      name: "Update DNS and Connectivity"
      owner: "Infrastructure Team"
      duration: "15 minutes"
      actions:
        - Update Route53 records
        - Update application configuration
        - Verify connectivity from applications

    - id: 5
      name: "Validate and Announce Recovery"
      owner: "QA Team"
      duration: "30 minutes"
      actions:
        - Run smoke tests
        - Verify critical workflows
        - Announce recovery to stakeholders

  rollback_to_primary:
    trigger: "Primary region restored"
    steps:
      - Sync data from DR to primary
      - Verify primary database
      - Gradual traffic shift (10% -> 50% -> 100%)
      - Decommission DR resources
```

---

## 8. Cost Estimation

### 8.1 Aurora PostgreSQL Costs (Production)

| Resource | Specification | Monthly Cost |
|----------|---------------|--------------|
| **Aurora Writer Instance** | r6g.xlarge (4 vCPU, 32 GB) | $550 |
| **Aurora Reader Instance (Sync)** | r6g.xlarge (4 vCPU, 32 GB) | $550 |
| **Aurora Reader Instance (Async)** | r6g.large (2 vCPU, 16 GB) | $275 |
| **Aurora Storage** | 500 GB @ $0.10/GB | $50 |
| **Aurora I/O** | ~50M requests @ $0.20/M | $10 |
| **Aurora Backtrack** | 24 hours @ $0.012/M changes | $50 |
| **Subtotal (Aurora)** | | **$1,485** |

### 8.2 Alternative: RDS PostgreSQL Costs

| Resource | Specification | Monthly Cost |
|----------|---------------|--------------|
| **RDS Primary Instance** | r6g.xlarge (4 vCPU, 32 GB) | $460 |
| **RDS Read Replica 1** | r6g.xlarge (4 vCPU, 32 GB) | $460 |
| **RDS Read Replica 2** | r6g.large (2 vCPU, 16 GB) | $230 |
| **RDS Multi-AZ** | Included in instance | $0 |
| **Subtotal (RDS Instances)** | | **$1,150** |

### 8.3 Storage Costs

| Resource | Specification | Monthly Cost |
|----------|---------------|--------------|
| **EBS gp3 Storage (Primary)** | 500 GB @ $0.08/GB | $40 |
| **EBS gp3 Storage (Replicas)** | 1,000 GB @ $0.08/GB | $80 |
| **EBS IOPS (Primary)** | 16,000 IOPS @ $0.005/IOPS | $80 |
| **EBS IOPS (Replicas)** | 24,000 IOPS @ $0.005/IOPS | $120 |
| **EBS Throughput** | 1,250 MB/s @ $0.04/MB/s | $50 |
| **Subtotal (Storage)** | | **$370** |

### 8.4 Backup Storage Costs

| Resource | Specification | Monthly Cost |
|----------|---------------|--------------|
| **S3 Primary Backups** | 1 TB @ $0.023/GB | $23 |
| **S3 Cross-Region** | 1 TB @ $0.023/GB | $23 |
| **S3 Data Transfer (Cross-Region)** | 100 GB @ $0.02/GB | $2 |
| **S3 Glacier (Long-term)** | 500 GB @ $0.004/GB | $2 |
| **Subtotal (Backups)** | | **$50** |

### 8.5 Additional Infrastructure Costs

| Resource | Specification | Monthly Cost |
|----------|---------------|--------------|
| **PgBouncer (2x t3.medium)** | 2 instances | $60 |
| **etcd Cluster (Patroni DCS)** | 3x t3.small | $45 |
| **CloudWatch Logs** | 50 GB @ $0.50/GB | $25 |
| **CloudWatch Metrics** | Custom metrics | $20 |
| **KMS Key Usage** | ~10K requests | $10 |
| **Secrets Manager** | 10 secrets | $5 |
| **VPC Endpoints** | 4 endpoints | $30 |
| **Subtotal (Additional)** | | **$195** |

### 8.6 Data Transfer Costs

| Resource | Specification | Monthly Cost |
|----------|---------------|--------------|
| **Inter-AZ Replication** | 500 GB @ $0.01/GB | $5 |
| **VPC to Internet (Egress)** | 100 GB @ $0.09/GB | $9 |
| **Subtotal (Data Transfer)** | | **$14** |

### 8.7 Total Monthly Cost Summary

| Configuration | Monthly Cost | Annual Cost |
|---------------|--------------|-------------|
| **Aurora PostgreSQL (Recommended)** | $2,114 | $25,368 |
| **RDS PostgreSQL (Budget)** | $1,779 | $21,348 |

**Cost Range:** **$2,100 - $2,500/month** (Aurora) or **$1,800 - $2,200/month** (RDS)

### 8.8 Cost Optimization Opportunities

| Optimization | Savings | Implementation |
|--------------|---------|----------------|
| **Reserved Instances (1-year)** | 30% (~$500/mo) | Commit to 1-year term |
| **Reserved Instances (3-year)** | 50% (~$800/mo) | Commit to 3-year term |
| **Graviton Instances** | 20% (~$300/mo) | Already using r6g |
| **gp3 vs gp2** | 20% (~$60/mo) | Already using gp3 |
| **S3 Intelligent Tiering** | 10% (~$5/mo) | Enable for backups |

**Optimized Annual Cost (1-year RI):** ~$17,750

---

## 9. Implementation Phases

### 9.1 Phase 1: Core Infrastructure (Week 1-2)

| Task | Duration | Owner | Dependencies |
|------|----------|-------|--------------|
| Create Terraform modules for RDS/Aurora | 2 days | DevOps | INFRA-001 complete |
| Configure VPC subnets and security groups | 1 day | DevOps | VPC deployed |
| Provision primary database instance | 2 hours | DevOps | Modules ready |
| Install and configure TimescaleDB extension | 4 hours | DBA | Primary running |
| Create database schemas and roles | 4 hours | DBA | TimescaleDB installed |
| Configure PgBouncer connection pooling | 4 hours | DevOps | Primary accessible |
| Set up AWS Secrets Manager | 2 hours | DevOps | - |
| Configure KMS encryption | 2 hours | Security | - |
| **Phase 1 Total** | **10 days** | | |

**Phase 1 Deliverables:**
- [ ] Primary database instance running
- [ ] TimescaleDB extension installed and configured
- [ ] Basic schemas created
- [ ] PgBouncer operational
- [ ] Encryption enabled (at-rest and in-transit)

### 9.2 Phase 2: Replication Setup (Week 3)

| Task | Duration | Owner | Dependencies |
|------|----------|-------|--------------|
| Deploy synchronous replica (AZ-1b) | 4 hours | DevOps | Primary stable |
| Configure streaming replication | 4 hours | DBA | Replica deployed |
| Deploy asynchronous replica (AZ-1c) | 4 hours | DevOps | Sync replica stable |
| Set up Patroni cluster for HA | 8 hours | DevOps | All replicas ready |
| Deploy etcd cluster for DCS | 4 hours | DevOps | - |
| Configure automatic failover | 4 hours | DevOps | Patroni running |
| Test failover scenarios | 8 hours | QA | Failover configured |
| **Phase 2 Total** | **5 days** | | |

**Phase 2 Deliverables:**
- [ ] Synchronous replica operational
- [ ] Asynchronous replica operational
- [ ] Patroni cluster managing failover
- [ ] Failover tested and documented

### 9.3 Phase 3: TimescaleDB Configuration (Week 4)

| Task | Duration | Owner | Dependencies |
|------|----------|-------|--------------|
| Create hypertables for emission_measurements | 4 hours | DBA | Schema ready |
| Create hypertables for sensor_readings | 4 hours | DBA | Schema ready |
| Create hypertables for calculation_results | 4 hours | DBA | Schema ready |
| Create hypertables for audit_logs | 4 hours | DBA | Schema ready |
| Configure compression policies | 4 hours | DBA | Hypertables created |
| Configure retention policies | 4 hours | DBA | Hypertables created |
| Create continuous aggregates | 8 hours | DBA | Hypertables created |
| Performance testing with sample data | 8 hours | QA | All tables ready |
| **Phase 3 Total** | **5 days** | | |

**Phase 3 Deliverables:**
- [ ] All 4 hypertables created and operational
- [ ] Compression policies active
- [ ] Retention policies configured
- [ ] Continuous aggregates refreshing
- [ ] Performance benchmarks documented

### 9.4 Phase 4: Monitoring and Backup (Week 5)

| Task | Duration | Owner | Dependencies |
|------|----------|-------|--------------|
| Deploy PostgreSQL Exporter | 4 hours | DevOps | Prometheus running |
| Configure Prometheus scraping | 2 hours | DevOps | Exporter running |
| Create Grafana dashboards | 8 hours | DevOps | Metrics flowing |
| Configure alerting rules | 4 hours | DevOps | Dashboards created |
| Set up PagerDuty integration | 2 hours | DevOps | Alerts configured |
| Install and configure pgBackRest | 8 hours | DevOps | Primary accessible |
| Configure S3 backup repository | 4 hours | DevOps | pgBackRest installed |
| Set up cross-region replication | 4 hours | DevOps | Primary repo ready |
| Configure backup schedules | 2 hours | DevOps | All repos ready |
| Test backup and restore | 8 hours | QA | Backups running |
| **Phase 4 Total** | **5 days** | | |

**Phase 4 Deliverables:**
- [ ] Full monitoring stack operational
- [ ] All critical alerts configured
- [ ] PagerDuty integration tested
- [ ] pgBackRest operational
- [ ] Full and incremental backups running
- [ ] Point-in-time recovery tested

### 9.5 Phase 5: Testing and Validation (Week 6)

| Task | Duration | Owner | Dependencies |
|------|----------|-------|--------------|
| Load testing (100K rows/sec ingestion) | 8 hours | QA | All components ready |
| Query performance benchmarking | 8 hours | QA | Data loaded |
| Failover testing (planned) | 4 hours | QA | Patroni stable |
| Failover testing (chaos engineering) | 8 hours | QA | Planned tests pass |
| Backup/restore validation | 8 hours | QA | Backups running |
| Security audit | 8 hours | Security | All access configured |
| Documentation and runbooks | 16 hours | DevOps | All tests pass |
| Knowledge transfer | 8 hours | DevOps | Docs complete |
| **Phase 5 Total** | **8 days** | | |

**Phase 5 Deliverables:**
- [ ] Load test results documented
- [ ] Performance benchmarks documented
- [ ] Failover procedures validated
- [ ] Backup/restore procedures validated
- [ ] Security audit passed
- [ ] Runbooks and documentation complete

### 9.6 Implementation Timeline

```
Week 1  Week 2  Week 3  Week 4  Week 5  Week 6
│       │       │       │       │       │
├───────┴───────┤       │       │       │
│ Phase 1:      │       │       │       │
│ Core Infra    │       │       │       │
│               │       │       │       │
│               ├───────┤       │       │
│               │ Phase │       │       │
│               │ 2:    │       │       │
│               │ Repl. │       │       │
│               │       │       │       │
│               │       ├───────┤       │
│               │       │ Phase │       │
│               │       │ 3:    │       │
│               │       │ Tmscl │       │
│               │       │       │       │
│               │       │       ├───────┤
│               │       │       │ Phase │
│               │       │       │ 4:    │
│               │       │       │ Mon/  │
│               │       │       │ Bkup  │
│               │       │       │       │
│               │       │       │       ├───────┐
│               │       │       │       │ Phase │
│               │       │       │       │ 5:    │
│               │       │       │       │ Test/ │
│               │       │       │       │ Valid │
└───────────────┴───────┴───────┴───────┴───────┘

Total Duration: 6 weeks
```

---

## 10. Acceptance Criteria

### 10.1 Functional Acceptance Criteria

| ID | Criterion | Validation Method | Pass/Fail |
|----|-----------|-------------------|-----------|
| FA-01 | Primary database accepts connections from EKS pods | Connection test from application pod | [ ] |
| FA-02 | Synchronous replica is in streaming replication | `SELECT * FROM pg_stat_replication` | [ ] |
| FA-03 | Asynchronous replica is in streaming replication | `SELECT * FROM pg_stat_replication` | [ ] |
| FA-04 | TimescaleDB extension is installed and active | `\dx timescaledb` | [ ] |
| FA-05 | All 4 hypertables created with correct partitioning | `SELECT * FROM timescaledb_information.hypertables` | [ ] |
| FA-06 | Compression policies active | `SELECT * FROM timescaledb_information.compression_settings` | [ ] |
| FA-07 | Retention policies configured | `SELECT * FROM timescaledb_information.jobs` | [ ] |
| FA-08 | Continuous aggregates refreshing | Check aggregate data freshness | [ ] |
| FA-09 | PgBouncer connection pooling working | Test connection reuse | [ ] |
| FA-10 | IAM authentication working | Connect using IAM credentials | [ ] |

### 10.2 Performance Benchmarks

| ID | Benchmark | Target | Actual | Pass/Fail |
|----|-----------|--------|--------|-----------|
| PB-01 | Insert throughput (emission_measurements) | > 50,000 rows/sec | | [ ] |
| PB-02 | Insert throughput (sensor_readings) | > 100,000 rows/sec | | [ ] |
| PB-03 | Time-range query (1 month, single facility) | < 100ms | | [ ] |
| PB-04 | Time-range query (1 year, all facilities) | < 500ms | | [ ] |
| PB-05 | Continuous aggregate query | < 50ms | | [ ] |
| PB-06 | Complex aggregation (quarterly emissions) | < 200ms | | [ ] |
| PB-07 | Concurrent connections (sustained) | > 500 | | [ ] |
| PB-08 | Connection acquisition time (via PgBouncer) | < 5ms | | [ ] |

### 10.3 High Availability Failover Tests

| ID | Test Scenario | Expected Outcome | RTO Target | Pass/Fail |
|----|---------------|------------------|------------|-----------|
| HA-01 | Kill primary PostgreSQL process | Automatic failover to sync replica | < 30s | [ ] |
| HA-02 | Network partition primary | Automatic failover | < 30s | [ ] |
| HA-03 | EBS volume failure (simulated) | Automatic failover | < 60s | [ ] |
| HA-04 | AZ failure (simulated) | Automatic failover to other AZ | < 60s | [ ] |
| HA-05 | PgBouncer failure | Connection retry to other PgBouncer | < 10s | [ ] |
| HA-06 | Patroni leader election | Clean leadership transfer | < 10s | [ ] |

### 10.4 Backup and Restore Tests

| ID | Test | Expected Outcome | Pass/Fail |
|----|------|------------------|-----------|
| BR-01 | Full backup completes successfully | Backup in S3, verifiable | [ ] |
| BR-02 | Incremental backup completes successfully | Backup in S3, linked to full | [ ] |
| BR-03 | WAL archiving continuous | No gaps in WAL sequence | [ ] |
| BR-04 | Point-in-time restore (1 hour ago) | Data restored to exact point | [ ] |
| BR-05 | Point-in-time restore (24 hours ago) | Data restored to exact point | [ ] |
| BR-06 | Full restore from weekly backup | Complete database restored | [ ] |
| BR-07 | Cross-region restore (us-west-2) | Database restored from DR bucket | [ ] |
| BR-08 | Restore time < 2 hours for 500GB | Full database restored | [ ] |

### 10.5 Security Tests

| ID | Test | Expected Outcome | Pass/Fail |
|----|------|------------------|-----------|
| ST-01 | Direct connection without SSL fails | Connection rejected | [ ] |
| ST-02 | Connection with invalid IAM credentials fails | Authentication rejected | [ ] |
| ST-03 | Connection from unauthorized security group fails | Connection timeout | [ ] |
| ST-04 | Encryption at rest verified | EBS volumes encrypted | [ ] |
| ST-05 | No public IP on database instances | Private IP only | [ ] |
| ST-06 | Secrets rotation executes successfully | New credentials work | [ ] |
| ST-07 | Audit logs capture all DDL/DML | Logs contain expected entries | [ ] |

---

## 11. Dependencies

### 11.1 Infrastructure Dependencies

| Dependency | Status | Blocker? | Notes |
|------------|--------|----------|-------|
| **INFRA-001: EKS Cluster** | Pending | Yes | Database needs EKS for application connectivity |
| **VPC with Database Subnets** | Part of INFRA-001 | Yes | 3 AZs with isolated database subnets |
| **NAT Gateways** | Part of INFRA-001 | Yes | For TimescaleDB license validation |
| **VPC Endpoints** | Part of INFRA-001 | Yes | S3, Secrets Manager, KMS |

### 11.2 Service Dependencies

| Dependency | Status | Blocker? | Notes |
|------------|--------|----------|-------|
| **AWS Secrets Manager** | Available | No | For credential storage and rotation |
| **AWS KMS** | Available | No | For encryption key management |
| **S3** | Available | No | For backup storage |
| **Route53** | Available | No | For DNS failover |
| **CloudWatch** | Available | No | For logs and metrics |

### 11.3 Monitoring Stack Dependencies

| Dependency | Status | Blocker? | Notes |
|------------|--------|----------|-------|
| **Prometheus** | Part of INFRA-001 | No | Metrics collection |
| **Grafana** | Part of INFRA-001 | No | Dashboards |
| **AlertManager** | Part of INFRA-001 | No | Alert routing |
| **PagerDuty** | SaaS | No | On-call alerting |

### 11.4 Application Dependencies

| Application | Database Requirement | Notes |
|-------------|---------------------|-------|
| **GL-CSRD-APP** | emission_measurements, calculation_results | Primary user of time-series data |
| **GL-CBAM-APP** | emission_measurements, audit_logs | Quarterly reporting |
| **GL-EUDR-APP** | audit_logs | Traceability requirements |
| **GL-VCCI-APP** | sensor_readings, emission_measurements | High-frequency data |
| **Agent Factory** | calculation_results | Stores agent execution results |

---

## 12. Risks and Mitigations

### 12.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **TimescaleDB version incompatibility** | Medium | High | Pin TimescaleDB version; test upgrade path in staging |
| **Replication lag under high load** | Medium | Medium | Monitor lag; scale sync replica if needed; tune WAL settings |
| **Connection pool exhaustion** | Low | High | Monitor connections; configure PgBouncer limits; implement connection retry logic |
| **Storage IOPS throttling** | Low | Medium | Provision adequate IOPS; enable gp3 burst credits; monitor CloudWatch |
| **Patroni split-brain scenario** | Very Low | Critical | Use quorum-based etcd cluster; configure proper fencing |

### 12.2 Operational Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Backup corruption undetected** | Low | Critical | Weekly backup verification; cross-region redundancy |
| **Runaway queries consuming resources** | Medium | Medium | Statement timeout; pg_stat_statements monitoring; query kill automation |
| **Credential rotation failure** | Low | Medium | Test rotation in staging; alerting on rotation failures |
| **Insufficient monitoring coverage** | Medium | Medium | Comprehensive alert coverage; regular runbook reviews |

### 12.3 Compliance Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Data retention policy violation** | Low | High | Automated retention policies; regular compliance audits |
| **Audit log tampering** | Very Low | Critical | Immutable audit table (trigger protection); log forwarding to SIEM |
| **Encryption key loss** | Very Low | Critical | Multi-region KMS key backup; key rotation testing |
| **GDPR data subject request failure** | Low | Medium | Data mapping; deletion procedures; regular testing |

### 12.4 Cost Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Storage cost overrun** | Medium | Low | Compression policies; data lifecycle management; billing alerts |
| **Data transfer cost spike** | Low | Low | VPC endpoints; regional data locality; transfer monitoring |
| **Reserved instance underutilization** | Low | Medium | Right-size before committing; start with 1-year terms |

### 12.5 Risk Register Summary

```
┌────────────────────────────────────────────────────────────────────────┐
│                         Risk Heat Map                                   │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Impact                                                                 │
│    ▲                                                                    │
│    │                                                                    │
│  Critical │                    │ Split-brain  │ Audit tampering        │
│    │      │                    │              │ Key loss               │
│    │      │                    │              │                        │
│  High     │ Conn exhaustion   │ TimescaleDB  │ Backup corruption      │
│    │      │                    │ compat       │ Retention violation    │
│    │      │                    │              │                        │
│  Medium   │ Storage IOPS      │ Repl lag     │ GDPR request fail      │
│    │      │                    │ Runaway query│ Monitoring gaps        │
│    │      │                    │              │                        │
│  Low      │ Transfer cost     │ Storage cost │ Cred rotation fail     │
│    │      │                    │ RI underutil │                        │
│    │      │                    │              │                        │
│    └──────┼──────────────────┼──────────────┼────────────────────►    │
│           │   Very Low        │    Low       │   Medium      High      │
│           │                   │              │                         │
│           │                              Likelihood                    │
│                                                                         │
│  Legend: Items in RED require immediate mitigation                     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Appendices

### Appendix A: SQL Schema Reference

Complete schema definitions are provided in Section 4.5.

### Appendix B: Terraform Module Reference

```hcl
# deployment/terraform/modules/database/main.tf

module "database" {
  source = "./modules/database"

  environment           = "prod"
  vpc_id               = module.vpc.vpc_id
  database_subnet_ids  = module.vpc.database_subnet_ids

  # Instance configuration
  instance_class       = "r6g.xlarge"
  allocated_storage    = 500
  max_allocated_storage = 2000
  iops                 = 16000

  # Replication
  multi_az             = true
  replica_count        = 2

  # Backup
  backup_retention_period = 35
  backup_window          = "02:00-04:00"

  # Security
  kms_key_id           = module.kms.database_key_id

  # Tags
  tags = {
    Project     = "GreenLang"
    Environment = "Production"
    Component   = "Database"
  }
}
```

### Appendix C: Monitoring Query Reference

```sql
-- Replication status
SELECT
    client_addr,
    state,
    sent_lsn,
    write_lsn,
    flush_lsn,
    replay_lsn,
    pg_wal_lsn_diff(sent_lsn, replay_lsn) AS lag_bytes
FROM pg_stat_replication;

-- Connection statistics
SELECT
    count(*) FILTER (WHERE state = 'active') AS active,
    count(*) FILTER (WHERE state = 'idle') AS idle,
    count(*) FILTER (WHERE state = 'idle in transaction') AS idle_in_txn,
    count(*) AS total
FROM pg_stat_activity
WHERE datname = 'greenlang';

-- TimescaleDB chunk information
SELECT
    hypertable_name,
    chunk_name,
    range_start,
    range_end,
    is_compressed
FROM timescaledb_information.chunks
ORDER BY range_end DESC;

-- Compression statistics
SELECT
    hypertable_name,
    before_compression_total_bytes / 1024 / 1024 AS before_mb,
    after_compression_total_bytes / 1024 / 1024 AS after_mb,
    (1 - after_compression_total_bytes::float / before_compression_total_bytes) * 100 AS compression_pct
FROM timescaledb_information.compression_settings
JOIN timescaledb_information.hypertable_compression_stats USING (hypertable_name);
```

### Appendix D: Runbook Quick Reference

| Scenario | Runbook Location | On-Call Action |
|----------|------------------|----------------|
| Primary failover | `/runbooks/db-failover.md` | Follow automated failover; verify replication |
| Replication lag | `/runbooks/replication-lag.md` | Check WAL sender; verify network; scale if needed |
| Connection exhaustion | `/runbooks/connection-issues.md` | Kill idle connections; scale PgBouncer |
| Backup failure | `/runbooks/backup-failure.md` | Check S3 access; verify pgBackRest config |
| Performance degradation | `/runbooks/performance.md` | Check pg_stat_statements; analyze slow queries |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-03 | Infrastructure Team | Initial PRD |

---

## Approvals Required

| Approver | Role | Approval For | Status |
|----------|------|--------------|--------|
| CTO | Technical Lead | Architecture approval | [ ] Pending |
| Database Lead | Database Admin | Schema and configuration | [ ] Pending |
| Security Lead | Security | Security controls | [ ] Pending |
| Finance | Budget | Cost approval (~$2,500/mo) | [ ] Pending |

---

**END OF PRD**
