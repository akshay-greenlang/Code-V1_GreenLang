# Prometheus Stack Architecture

## Overview

This document describes the architecture of the GreenLang Prometheus observability stack, including component interactions, data flows, scaling considerations, and operational guidelines.

---

## Architecture Diagram

```
                                         ┌─────────────────────┐
                                         │      Grafana        │
                                         │  (Visualization)    │
                                         └──────────┬──────────┘
                                                    │
                              ┌─────────────────────┼─────────────────────┐
                              │                     │                     │
                     ┌────────▼────────┐  ┌────────▼────────┐  ┌────────▼────────┐
                     │  Thanos Query   │  │  Thanos Query   │  │   Alertmanager  │
                     │   (Frontend)    │  │    (Direct)     │  │     (HA x2)     │
                     └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
                              │                    │                     │
                              └────────────────────┼─────────────────────┘
                                                   │
         ┌─────────────────────────────────────────┼─────────────────────────────────────────┐
         │                                         │                                         │
┌────────▼────────┐                     ┌──────────▼──────────┐                   ┌──────────▼──────────┐
│ Thanos Sidecar  │                     │  Thanos Store GW    │                   │   Thanos Ruler      │
│  (per Prom)     │                     │     (HA x2)         │                   │     (HA x2)         │
└────────┬────────┘                     └──────────┬──────────┘                   └───────────┬─────────┘
         │                                         │                                          │
┌────────▼────────┐                     ┌──────────▼──────────┐                               │
│   Prometheus    │─────────────────────│     S3 Bucket       │───────────────────────────────┘
│    (HA x2)      │                     │  (2-year retention) │
└────────┬────────┘                     └──────────┬──────────┘
         │                                         │
         │                              ┌──────────▼──────────┐
         │                              │  Thanos Compactor   │
         │                              │     (Singleton)     │
         │                              └─────────────────────┘
         │
┌────────┴────────────────────────────────────────────────────────┐
│                        Scrape Targets                            │
├─────────────┬─────────────┬─────────────┬─────────────┬─────────┤
│ GreenLang   │   Agents    │    Kong     │  PostgreSQL │  Redis  │
│    API      │  (47+)      │   Gateway   │  Exporter   │Exporter │
└─────────────┴─────────────┴─────────────┴─────────────┴─────────┘
                              │
                    ┌─────────┴─────────┐
                    │    PushGateway    │
                    │      (HA x2)      │
                    └─────────┬─────────┘
                              │
                    ┌─────────┴─────────┐
                    │    Batch Jobs     │
                    │  (CronJobs)       │
                    └───────────────────┘
```

---

## Components

### Prometheus Server

**Purpose:** Metrics collection, short-term storage, and alert evaluation.

| Attribute | Value |
|-----------|-------|
| Version | 2.50+ |
| Replicas | 2 (HA with pod anti-affinity) |
| Local Retention | 7 days / 50GB |
| Scrape Interval | 15s (default) |
| Evaluation Interval | 15s |

**External Labels:**
- `cluster`: greenlang-{environment}
- `region`: {aws_region}
- `replica`: $(POD_NAME)

**Key Features:**
- Service discovery via ServiceMonitors and PodMonitors
- Rule evaluation for alerts and recording rules
- Thanos sidecar for S3 upload
- Remote write support (optional)

### Thanos Sidecar

**Purpose:** Upload Prometheus blocks to S3, provide Store API for queries.

| Attribute | Value |
|-----------|-------|
| Version | 0.34+ |
| Deployment | Per Prometheus replica |
| Upload Interval | 2 hours (block size) |

**Responsibilities:**
- Upload TSDB blocks to S3 every 2 hours
- Provide StoreAPI for real-time data queries
- Inject external labels into blocks

### Thanos Query

**Purpose:** Unified query layer across all data stores.

| Attribute | Value |
|-----------|-------|
| Version | 0.34+ |
| Replicas | 2 |
| Stores | Sidecars + Store Gateway |

**Features:**
- Deduplication of HA data
- Partial response mode
- Query pushdown to stores
- Store API discovery via DNS

### Thanos Store Gateway

**Purpose:** Query historical data from S3.

| Attribute | Value |
|-----------|-------|
| Version | 0.34+ |
| Replicas | 2 |
| Cache | Index + Chunks (in-memory or memcached) |

**Storage:**
- Index cache: 1GB
- Chunk cache: 2GB
- PVC: 50Gi for block caching

### Thanos Compactor

**Purpose:** Compact and downsample blocks in S3.

| Attribute | Value |
|-----------|-------|
| Version | 0.34+ |
| Replicas | 1 (singleton) |
| PVC | 100Gi |

**Retention Policy:**
| Resolution | Retention |
|------------|-----------|
| Raw (15s) | 30 days |
| 5-minute | 120 days |
| 1-hour | 730 days (2 years) |

**Important:** Only one compactor instance should run at a time.

### Thanos Ruler

**Purpose:** Evaluate rules against historical data.

| Attribute | Value |
|-----------|-------|
| Version | 0.34+ |
| Replicas | 2 |
| Rule Sources | ConfigMaps, Thanos Object Store |

**Use Cases:**
- Long-range alerts (>2h lookback)
- Recording rules for historical aggregations
- SLI/SLO calculations

### Alertmanager

**Purpose:** Alert routing, grouping, and notification delivery.

| Attribute | Value |
|-----------|-------|
| Version | 0.27+ |
| Replicas | 2 (HA with gossip) |
| Cluster Port | 9094 |

**Notification Channels:**
- Slack (all environments)
- PagerDuty (production critical)
- Email (weekly summaries)

**Alert Routing:**
```yaml
route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  routes:
    - match: {severity: critical}
      receiver: pagerduty-critical
    - match: {severity: warning}
      receiver: slack-warnings
```

### PushGateway

**Purpose:** Metrics from batch jobs and short-lived processes.

| Attribute | Value |
|-----------|-------|
| Version | 1.7+ |
| Replicas | 2 |
| Persistence | 2Gi |

**Usage Pattern:**
1. Batch job starts, creates metrics
2. Job pushes metrics to PushGateway
3. Prometheus scrapes PushGateway
4. Job optionally deletes metrics on completion

---

## Data Flow

### Scraping Flow

```
┌────────────┐     scrape      ┌─────────────┐     store      ┌─────────────┐
│  Service   │ ─────────────▶  │  Prometheus │ ─────────────▶ │    TSDB     │
│  /metrics  │   (15s interval) │             │  (local disk) │   (7 days)  │
└────────────┘                 └─────────────┘                └─────────────┘
                                      │
                                      │ every 2 hours
                                      ▼
                               ┌─────────────┐
                               │  S3 Bucket  │
                               │  (Thanos)   │
                               └─────────────┘
```

### Query Flow

```
┌────────────┐     query      ┌──────────────┐     query      ┌─────────────────┐
│  Grafana   │ ─────────────▶ │ Thanos Query │ ─────────────▶ │ Thanos Sidecar  │
│            │                │              │                │ (real-time)     │
└────────────┘                │              │ ─────────────▶ ┌─────────────────┐
                              │              │                │ Thanos Store GW │
                              └──────────────┘                │ (historical)    │
                                      │                       └─────────────────┘
                                      │ deduplicate + merge
                                      ▼
                               ┌─────────────┐
                               │   Results   │
                               └─────────────┘
```

### Alert Flow

```
┌─────────────┐  evaluate   ┌─────────────┐    fire     ┌──────────────┐
│ Prometheus  │ ──────────▶ │   Rules     │ ──────────▶ │ Alertmanager │
│             │  (15s)      │  (PromQL)   │             │              │
└─────────────┘             └─────────────┘             └──────┬───────┘
                                                               │
                                          ┌────────────────────┼───────────────┐
                                          │                    │               │
                                   ┌──────▼─────┐      ┌───────▼──────┐ ┌──────▼──────┐
                                   │   Slack    │      │  PagerDuty   │ │   Email     │
                                   └────────────┘      └──────────────┘ └─────────────┘
```

---

## Storage Architecture

### Local Storage (Prometheus)

```
/prometheus/
├── wal/                    # Write-Ahead Log (recent data)
│   ├── 00000001
│   ├── 00000002
│   └── checkpoint.00000001
├── chunks_head/            # In-memory chunks
└── 01EXAMPLE123/           # Completed blocks
    ├── meta.json
    ├── index
    ├── chunks/
    └── tombstones
```

### Object Storage (S3)

```
gl-thanos-metrics-{env}/
├── 01EXAMPLE123/           # Block ULID
│   ├── meta.json           # Block metadata
│   ├── index               # Block index
│   ├── chunks/
│   │   ├── 000001
│   │   └── 000002
│   └── deletion-mark.json  # If marked for deletion
├── 01EXAMPLE456/           # Another block
└── ...
```

### Retention Tiers

| Tier | Resolution | Retention | Storage Class |
|------|------------|-----------|---------------|
| Hot | 15s | 0-7 days | Local SSD |
| Warm | 15s | 7-30 days | S3 Standard |
| Cool | 5m | 30-120 days | S3 Intelligent Tiering |
| Cold | 1h | 120-730 days | S3 Glacier |

---

## Scaling Guidelines

### Horizontal Scaling

| Component | Scale Trigger | Max Replicas |
|-----------|---------------|--------------|
| Prometheus | Series > 500K per replica | 4 |
| Thanos Query | Query rate > 100 QPS | 4 |
| Thanos Store GW | Historical query latency > 10s | 4 |
| Alertmanager | Not typically scaled beyond 3 | 3 |

### Vertical Scaling

| Component | CPU Request | Memory Request | CPU Limit | Memory Limit |
|-----------|-------------|----------------|-----------|--------------|
| Prometheus | 500m | 2Gi | 2000m | 8Gi |
| Thanos Query | 250m | 512Mi | 1000m | 2Gi |
| Thanos Store GW | 250m | 1Gi | 1000m | 4Gi |
| Thanos Compactor | 250m | 1Gi | 1000m | 4Gi |
| Alertmanager | 100m | 256Mi | 500m | 512Mi |
| PushGateway | 100m | 128Mi | 500m | 512Mi |

### Capacity Planning

**Formula for Prometheus memory:**
```
Memory = (number_of_series * bytes_per_series * 2) + buffer
       = (1,000,000 * 3KB * 2) + 1GB
       = 7GB
```

**Formula for storage:**
```
Storage = ingestion_rate * retention * bytes_per_sample * 2
        = 100,000 samples/sec * 7 days * 1.5 bytes * 2
        = ~180GB
```

---

## High Availability

### Prometheus HA

- 2 replicas with identical configuration
- Pod anti-affinity: `requiredDuringSchedulingIgnoredDuringExecution`
- External labels differentiate replicas
- Thanos deduplicates at query time

### Alertmanager HA

- 2-3 replicas in gossip cluster
- Mesh communication on port 9094
- Automatic leader election
- Notification deduplication

### Thanos HA

- Query: 2+ replicas behind load balancer
- Store Gateway: 2 replicas with sharding
- Compactor: Single replica (singleton)
- Ruler: 2 replicas with rule sharding

---

## Network Architecture

### Service Ports

| Service | Port | Protocol | Purpose |
|---------|------|----------|---------|
| Prometheus | 9090 | HTTP | Web UI / API |
| Prometheus (Thanos) | 10901 | gRPC | Store API |
| Thanos Query | 9090 | HTTP | Web UI / API |
| Thanos Query | 10901 | gRPC | Store API |
| Thanos Store GW | 10901 | gRPC | Store API |
| Alertmanager | 9093 | HTTP | Web UI / API |
| Alertmanager | 9094 | TCP | Cluster gossip |
| PushGateway | 9091 | HTTP | Push endpoint |

### Network Policies

```yaml
# Allow Prometheus to scrape all namespaces
ingress:
  - from:
      - namespaceSelector:
          matchLabels:
            kubernetes.io/metadata.name: monitoring
    ports:
      - port: metrics
        protocol: TCP

# Allow Thanos components to communicate
ingress:
  - from:
      - podSelector:
          matchLabels:
            app.kubernetes.io/part-of: thanos
    ports:
      - port: 10901
        protocol: TCP
```

---

## Security

### Authentication

- Grafana: OAuth2 / SAML via identity provider
- Prometheus: Internal only (no auth)
- Alertmanager: Internal only (no auth)
- API Gateway: JWT tokens for external access

### Authorization

- RBAC for Kubernetes resources
- ServiceAccount per component
- IRSA for S3 access (no static credentials)

### Encryption

- TLS for all inter-component communication
- S3 SSE-KMS for stored blocks
- Secrets managed via External Secrets Operator

---

## Related Documentation

- [Prometheus Operations Guide](../operations/prometheus-operations.md)
- [Metrics Developer Guide](../development/metrics-guide.md)
- [Operational Runbooks](../runbooks/README.md)
- [PRD-OBS-001](../../GreenLang%20Development/05-Documentation/PRD-OBS-001-Prometheus-Metrics-Collection.md)
