# PRD: INFRA-003 - Redis Caching Cluster

**Document Version:** 1.0
**Date:** February 3, 2026
**Status:** READY FOR EXECUTION
**Priority:** P0 - CRITICAL
**Owner:** Infrastructure Team
**Ralphy Task ID:** INFRA-003
**Depends On:** INFRA-001 (EKS Cluster Deployment)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Architecture Overview](#3-architecture-overview)
4. [Technical Requirements](#4-technical-requirements)
5. [Use Cases](#5-use-cases)
6. [Security Requirements](#6-security-requirements)
7. [Monitoring and Alerting](#7-monitoring-and-alerting)
8. [Backup and Recovery](#8-backup-and-recovery)
9. [Cost Estimation](#9-cost-estimation)
10. [Implementation Phases](#10-implementation-phases)
11. [Acceptance Criteria](#11-acceptance-criteria)
12. [Dependencies](#12-dependencies)
13. [Risks and Mitigations](#13-risks-and-mitigations)

---

## 1. Executive Summary

### 1.1 Overview

Deploy a production-ready Redis caching cluster with Sentinel high availability for the GreenLang Climate OS platform. This infrastructure provides:

- **High-performance caching** for emission factors, grid intensity data, and API responses
- **Distributed session storage** across multiple application services
- **Rate limiting** to protect APIs from abuse (10K+ requests/second capacity)
- **99.99% uptime SLA** with automatic failover via Redis Sentinel
- **Message queuing** via Redis Streams for async processing

### 1.2 Business Justification

| Requirement | Without Redis | With Redis Caching |
|-------------|---------------|-------------------|
| **API Response Time** | 200-500ms (DB query) | 5-20ms (cache hit) |
| **Database Load** | 100% queries hit DB | 80-90% served from cache |
| **Rate Limiting** | Application-level, inconsistent | Distributed, atomic, 10K+ req/sec |
| **Session Storage** | DB-backed, slow | In-memory, sub-millisecond |
| **Emission Factor Lookups** | 50ms per lookup | 2ms per lookup |

### 1.3 Expected Benefits and ROI

| Benefit | Impact | Annual Value |
|---------|--------|--------------|
| **10x Faster API Responses** | 95th percentile < 50ms | $100K (user productivity) |
| **80% Reduced DB Load** | Fewer read replicas needed | $50K (infrastructure savings) |
| **Rate Limiting Protection** | Prevent abuse, ensure SLA | Risk mitigation (priceless) |
| **Session Performance** | Sub-millisecond auth checks | $30K (developer productivity) |
| **Total Annual Value** | | **~$180K** |

**Investment:** ~$6-10K/year (infrastructure costs)
**ROI:** 18-30x return on infrastructure investment

---

## 2. Problem Statement

### 2.1 Current Caching Requirements

GreenLang Climate OS requires fast access to frequently-used data across multiple services:

| Data Type | Access Pattern | Current Latency | Required Latency | Volume |
|-----------|---------------|-----------------|------------------|--------|
| **Emission Factors** | Read-heavy, rarely changes | 50-100ms (DB) | < 5ms | 10K+ factors |
| **Grid Intensity Data** | Hourly updates, read-heavy | 30-50ms (DB) | < 5ms | 200+ regions |
| **User Sessions** | Every request | 20-30ms (DB) | < 2ms | 100K+ sessions |
| **API Rate Limits** | Every request | N/A (none) | < 1ms | 10K+ req/sec |
| **Calculation Results** | Frequently re-accessed | 100-200ms (DB) | < 10ms | 1M+ results |

### 2.2 Need for Distributed Cache

Multiple GreenLang services require shared caching:

```
                    ┌─────────────────────────────────────────────────────┐
                    │              GreenLang Microservices                 │
                    │                                                      │
                    │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐│
                    │  │GL-CSRD   │ │GL-CBAM   │ │GL-EUDR   │ │GL-VCCI  ││
                    │  │  APP     │ │  APP     │ │  APP     │ │  APP    ││
                    │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬────┘│
                    │       │            │            │            │      │
                    │       └────────────┴─────┬──────┴────────────┘      │
                    │                          │                          │
                    │                          ▼                          │
                    │              ┌─────────────────────┐                │
                    │              │   SHARED REDIS      │                │
                    │              │   CACHE CLUSTER     │                │
                    │              │                     │                │
                    │              │ - Emission Factors  │                │
                    │              │ - Grid Intensity    │                │
                    │              │ - User Sessions     │                │
                    │              │ - Rate Limits       │                │
                    │              │ - API Responses     │                │
                    │              └─────────────────────┘                │
                    └─────────────────────────────────────────────────────┘
```

### 2.3 High Availability Requirements

| Requirement | Target | Justification |
|-------------|--------|---------------|
| **Uptime SLA** | 99.99% | Critical path for all API requests |
| **RTO (Recovery Time Objective)** | < 30 seconds | Automatic failover via Sentinel |
| **RPO (Recovery Point Objective)** | < 1 second | AOF persistence with fsync |
| **Failover Automation** | Fully automated | No human intervention during incidents |
| **Data Durability** | Best effort | Cache is ephemeral; DB is source of truth |

### 2.4 Rate Limiting Requirements

| Endpoint Category | Rate Limit | Window | Algorithm |
|-------------------|------------|--------|-----------|
| **Public API** | 100 req/min | Sliding window | Token bucket |
| **Authenticated API** | 1,000 req/min | Sliding window | Token bucket |
| **Bulk Operations** | 10 req/min | Fixed window | Fixed counter |
| **Webhooks** | 10,000 req/min | Sliding window | Sliding window log |
| **Internal Services** | Unlimited | N/A | No limit |

**Total Capacity Required:** 10,000+ requests/second across all services

---

## 3. Architecture Overview

### 3.1 Redis Sentinel Topology

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AWS REGION: us-east-1                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
│ Availability Zone │   │ Availability Zone │   │ Availability Zone │
│    us-east-1a     │   │    us-east-1b     │   │    us-east-1c     │
│                   │   │                   │   │                   │
│ ┌───────────────┐ │   │ ┌───────────────┐ │   │ ┌───────────────┐ │
│ │  SENTINEL 1   │ │   │ │  SENTINEL 2   │ │   │ │  SENTINEL 3   │ │
│ │  (Quorum)     │ │   │ │  (Quorum)     │ │   │ │  (Quorum)     │ │
│ │  Port: 26379  │ │   │ │  Port: 26379  │ │   │ │  Port: 26379  │ │
│ └───────┬───────┘ │   │ └───────┬───────┘ │   │ └───────┬───────┘ │
│         │         │   │         │         │   │         │         │
│         │ Monitor │   │         │ Monitor │   │         │ Monitor │
│         ▼         │   │         ▼         │   │         ▼         │
│ ┌───────────────┐ │   │ ┌───────────────┐ │   │ ┌───────────────┐ │
│ │ REDIS MASTER  │ │   │ │ REDIS REPLICA │ │   │ │ REDIS REPLICA │ │
│ │               │ │   │ │     #1        │ │   │ │     #2        │ │
│ │ cache.r6g.    │◀┼───┼─┤ cache.r6g.    │◀┼───┼─┤ cache.r6g.    │ │
│ │ xlarge        │ │   │ │ xlarge        │ │   │ │ large         │ │
│ │               │─┼───┼▶│               │─┼───┼▶│               │ │
│ │ Port: 6379    │ │   │ │ Port: 6379    │ │   │ │ Port: 6379    │ │
│ │ Memory: 26GB  │ │Repl│ │ Memory: 26GB  │ │Repl│ │ Memory: 13GB  │ │
│ └───────────────┘ │   │ └───────────────┘ │   │ └───────────────┘ │
│                   │   │                   │   │                   │
└───────────────────┘   └───────────────────┘   └───────────────────┘

                    ┌─────────────────────────────────────┐
                    │         KUBERNETES CLUSTER          │
                    │                                     │
                    │  ┌───────────────────────────────┐  │
                    │  │     Application Pods          │  │
                    │  │  ┌─────┐ ┌─────┐ ┌─────┐     │  │
                    │  │  │App 1│ │App 2│ │App N│     │  │
                    │  │  └──┬──┘ └──┬──┘ └──┬──┘     │  │
                    │  │     └───────┼───────┘        │  │
                    │  │             ▼                │  │
                    │  │  ┌───────────────────────┐   │  │
                    │  │  │  Redis Connection     │   │  │
                    │  │  │  Pool (per pod)       │   │  │
                    │  │  │  - Min: 5 connections │   │  │
                    │  │  │  - Max: 20 connections│   │  │
                    │  │  │  - Sentinel-aware     │   │  │
                    │  │  └───────────┬───────────┘   │  │
                    │  └──────────────┼───────────────┘  │
                    │                 │                  │
                    │                 ▼                  │
                    │       Sentinel Discovery          │
                    │       (redis-sentinel:26379)      │
                    └─────────────────────────────────────┘
```

### 3.2 AWS ElastiCache vs Self-Managed Comparison

| Feature | AWS ElastiCache | Self-Managed (K8s) | Recommendation |
|---------|-----------------|-------------------|----------------|
| **Setup Complexity** | Low (managed) | High (operators) | ElastiCache |
| **Operational Overhead** | Minimal | Significant | ElastiCache |
| **Automatic Failover** | Built-in | Manual config | ElastiCache |
| **Scaling** | One-click | Complex | ElastiCache |
| **Monitoring** | CloudWatch | Prometheus/Grafana | Both viable |
| **Cost (Production)** | ~$500-800/month | ~$300-500/month | Self-managed (budget) |
| **Data Persistence** | Limited AOF | Full control | Self-managed |
| **Redis Version** | 7.0 (latest) | Any version | Both viable |
| **Custom Config** | Limited | Full control | Self-managed |
| **Multi-AZ** | Built-in | Manual config | ElastiCache |

**Recommendation:** **AWS ElastiCache** for production due to managed operations, automatic failover, and reduced operational burden. Consider **Self-Managed** only if specific Redis features or cost optimization are critical.

### 3.3 Connection Pooling Strategy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Redis Connection Pool Architecture                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Application Pod                                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        Redis Client Library                             │ │
│  │                        (redis-py / ioredis)                             │ │
│  │                                                                         │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │ │
│  │  │                   Connection Pool Configuration                  │   │ │
│  │  │                                                                  │   │ │
│  │  │  pool_size: 10          # Connections per pod                   │   │ │
│  │  │  max_connections: 20    # Maximum connections                    │   │ │
│  │  │  timeout: 5             # Connection timeout (seconds)          │   │ │
│  │  │  retry_on_timeout: true # Retry on timeout                      │   │ │
│  │  │  socket_keepalive: true # Keep connections alive                │   │ │
│  │  │  health_check_interval: 30  # Health check frequency            │   │ │
│  │  │                                                                  │   │ │
│  │  │  ┌────────────────────────────────────────────────────────────┐ │   │ │
│  │  │  │              Active Connections [########--]               │ │   │ │
│  │  │  │              Used: 8 / Pool: 10 / Max: 20                  │ │   │ │
│  │  │  └────────────────────────────────────────────────────────────┘ │   │ │
│  │  └─────────────────────────────────────────────────────────────────┘   │ │
│  │                                                                         │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │ │
│  │  │                   Sentinel Configuration                         │   │ │
│  │  │                                                                  │   │ │
│  │  │  sentinel_hosts:                                                │   │ │
│  │  │    - sentinel-1.greenlang.internal:26379                        │   │ │
│  │  │    - sentinel-2.greenlang.internal:26379                        │   │ │
│  │  │    - sentinel-3.greenlang.internal:26379                        │   │ │
│  │  │  master_name: greenlang-cache                                   │   │ │
│  │  │  socket_timeout: 0.1                                            │   │ │
│  │  │  min_other_sentinels: 1                                         │   │ │
│  │  └─────────────────────────────────────────────────────────────────┘   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Connection Flow:                                                            │
│  1. Client queries Sentinel for master address                               │
│  2. Sentinel returns current master (e.g., redis-master-1:6379)             │
│  3. Client establishes pooled connections to master                          │
│  4. On failover, Sentinel notifies clients of new master                    │
│  5. Client transparently reconnects to new master                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.4 Failover Mechanism

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Redis Sentinel Failover Sequence                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  NORMAL OPERATION                                                            │
│  ================                                                            │
│                                                                              │
│  Sentinel 1 ──┐                                                              │
│  Sentinel 2 ──┼──► Monitor Master (health check every 1s)                   │
│  Sentinel 3 ──┘                                                              │
│                                                                              │
│  FAILURE DETECTION (T+0 to T+5s)                                            │
│  ================================                                            │
│                                                                              │
│  1. Master stops responding to PING                                         │
│  2. Sentinel marks master as SDOWN (subjectively down) after 5s             │
│  3. Sentinels communicate, reach quorum (2/3)                               │
│  4. Master marked as ODOWN (objectively down)                               │
│                                                                              │
│  FAILOVER PROCESS (T+5s to T+15s)                                           │
│  ==================================                                          │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Step 1: Leader Election (T+5s)                                      │    │
│  │  - Sentinels elect a leader to coordinate failover                  │    │
│  │  - Requires majority vote (2/3 sentinels)                           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                           │                                                  │
│                           ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Step 2: Replica Selection (T+6s)                                    │    │
│  │  - Leader selects best replica based on:                            │    │
│  │    * Replication offset (most up-to-date)                           │    │
│  │    * Priority (configured)                                          │    │
│  │    * Run ID (tie-breaker)                                           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                           │                                                  │
│                           ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Step 3: Promotion (T+8s)                                            │    │
│  │  - SLAVEOF NO ONE sent to selected replica                          │    │
│  │  - Replica becomes new master                                       │    │
│  │  - Other replicas reconfigured to follow new master                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                           │                                                  │
│                           ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Step 4: Client Notification (T+10s)                                 │    │
│  │  - Sentinels publish +switch-master event                           │    │
│  │  - Clients discover new master via Sentinel                         │    │
│  │  - Connection pools reconnect transparently                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  RECOVERY COMPLETE (T+15s)                                                  │
│  =========================                                                   │
│                                                                              │
│  - New master serving requests                                              │
│  - Old master (when recovered) joins as replica                             │
│  - Total failover time: < 15 seconds                                        │
│  - Data loss: 0-1 seconds (depending on replication lag)                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Technical Requirements

### 4.1 Redis Version and Instance Sizing

| Resource | Master | Replica 1 | Replica 2 | Justification |
|----------|--------|-----------|-----------|---------------|
| **Redis Version** | 7.2+ | 7.2+ | 7.2+ | Latest stable with Streams |
| **Instance Type** | cache.r6g.xlarge | cache.r6g.xlarge | cache.r6g.large | Memory-optimized |
| **vCPU** | 4 | 4 | 2 | Single-threaded + I/O threads |
| **Memory** | 26.32 GB | 26.32 GB | 13.07 GB | Cache capacity |
| **Network** | Up to 10 Gbps | Up to 10 Gbps | Up to 10 Gbps | High throughput |
| **maxmemory** | 6 GB | 6 GB | 6 GB | Configured limit |

### 4.2 Redis Configuration

```ini
# redis.conf (Production Optimized)

# ============================================================
# NETWORK
# ============================================================
bind 0.0.0.0
port 6379
protected-mode yes
tcp-backlog 511
timeout 0
tcp-keepalive 300

# ============================================================
# MEMORY MANAGEMENT
# ============================================================
maxmemory 6gb
maxmemory-policy allkeys-lru
maxmemory-samples 10

# Active memory defragmentation
activedefrag yes
active-defrag-ignore-bytes 100mb
active-defrag-threshold-lower 10
active-defrag-threshold-upper 25
active-defrag-cycle-min 1
active-defrag-cycle-max 25

# ============================================================
# PERSISTENCE (AOF + RDB)
# ============================================================

# RDB Snapshots
save 900 1          # Save if 1 key changed in 900 seconds
save 300 10         # Save if 10 keys changed in 300 seconds
save 60 10000       # Save if 10000 keys changed in 60 seconds
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir /data

# AOF (Append Only File)
appendonly yes
appendfilename "appendonly.aof"
appenddirname "appendonlydir"
appendfsync everysec
no-appendfsync-on-rewrite no
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb
aof-load-truncated yes
aof-use-rdb-preamble yes

# ============================================================
# REPLICATION
# ============================================================
replica-serve-stale-data yes
replica-read-only yes
repl-diskless-sync yes
repl-diskless-sync-delay 5
repl-ping-replica-period 10
repl-timeout 60
repl-disable-tcp-nodelay no
repl-backlog-size 64mb
repl-backlog-ttl 3600
replica-priority 100
min-replicas-to-write 1
min-replicas-max-lag 10

# ============================================================
# SECURITY
# ============================================================
requirepass ${REDIS_PASSWORD}
masterauth ${REDIS_PASSWORD}
# ACL rules loaded from external file
aclfile /etc/redis/users.acl

# ============================================================
# CLIENTS
# ============================================================
maxclients 10000

# ============================================================
# PERFORMANCE TUNING
# ============================================================
# I/O Threads (Redis 6.0+)
io-threads 4
io-threads-do-reads yes

# Lazy freeing
lazyfree-lazy-eviction yes
lazyfree-lazy-expire yes
lazyfree-lazy-server-del yes
replica-lazy-flush yes
lazyfree-lazy-user-del yes
lazyfree-lazy-user-flush yes

# ============================================================
# LOGGING
# ============================================================
loglevel notice
logfile /var/log/redis/redis.log
syslog-enabled yes
syslog-ident redis
syslog-facility local0

# ============================================================
# SLOW LOG
# ============================================================
slowlog-log-slower-than 10000
slowlog-max-len 128

# ============================================================
# LATENCY MONITORING
# ============================================================
latency-monitor-threshold 100
```

### 4.3 Sentinel Configuration

```ini
# sentinel.conf

# ============================================================
# SENTINEL CONFIGURATION
# ============================================================
port 26379
sentinel monitor greenlang-cache redis-master.greenlang.internal 6379 2
sentinel auth-pass greenlang-cache ${REDIS_PASSWORD}

# Failover timing
sentinel down-after-milliseconds greenlang-cache 5000
sentinel failover-timeout greenlang-cache 60000
sentinel parallel-syncs greenlang-cache 1

# Notification scripts
sentinel notification-script greenlang-cache /etc/sentinel/notify.sh
sentinel client-reconfig-script greenlang-cache /etc/sentinel/reconfig.sh

# Announce settings (for K8s/Docker)
sentinel announce-ip ${POD_IP}
sentinel announce-port 26379

# Security
requirepass ${SENTINEL_PASSWORD}
sentinel sentinel-pass ${SENTINEL_PASSWORD}

# Logging
logfile /var/log/sentinel/sentinel.log
loglevel notice
```

### 4.4 Data Structures Used

| Data Structure | Use Case | Example Key Pattern | TTL |
|----------------|----------|---------------------|-----|
| **String** | Simple key-value cache | `cache:emission_factor:{id}` | 24h |
| **Hash** | Object storage | `session:{session_id}` | 24h |
| **Sorted Set** | Rate limiting (sliding window) | `ratelimit:{client_id}:{endpoint}` | 1min |
| **Set** | Unique tracking | `active_users:{date}` | 24h |
| **List** | Recent activity | `recent_calculations:{org_id}` | 1h |
| **Stream** | Message queue | `stream:calculation_jobs` | 7d |
| **HyperLogLog** | Unique counts | `hll:daily_visitors:{date}` | 7d |

### 4.5 Memory Allocation Strategy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Memory Allocation (6GB Total)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Emission Factors Cache                                    2.0 GB      │ │
│  │  ████████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│ │
│  │  ~50,000 factors @ ~40KB each                                          │ │
│  │  TTL: 24 hours                                                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Grid Intensity Cache                                      0.5 GB      │ │
│  │  ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│ │
│  │  ~500 regions @ ~1MB each (hourly data)                               │ │
│  │  TTL: 1 hour                                                           │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Session Storage                                           1.0 GB      │ │
│  │  ████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│ │
│  │  ~100,000 sessions @ ~10KB each                                        │ │
│  │  TTL: 24 hours (sliding)                                               │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  API Response Cache                                        1.5 GB      │ │
│  │  ████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│ │
│  │  ~150,000 responses @ ~10KB each                                       │ │
│  │  TTL: 5 minutes - 1 hour (varies)                                      │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Rate Limiting & Streams                                   0.5 GB      │ │
│  │  ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│ │
│  │  Rate limit counters + message streams                                 │ │
│  │  TTL: 1 minute - 7 days                                                │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Reserved (Overhead + Buffer)                              0.5 GB      │ │
│  │  ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│ │
│  │  Memory fragmentation, replication buffer, COW overhead                │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Total Configured: 6 GB | Eviction Policy: allkeys-lru                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Use Cases

### 5.1 Emission Factor Caching (24-hour TTL)

```python
# emission_factor_cache.py

import json
import hashlib
from redis import Redis
from typing import Optional, Dict, Any

class EmissionFactorCache:
    """
    Cache emission factors with 24-hour TTL.
    Reduces database load by 90%+ for factor lookups.
    """

    CACHE_PREFIX = "ef"  # emission factor
    DEFAULT_TTL = 86400  # 24 hours in seconds

    def __init__(self, redis_client: Redis):
        self.redis = redis_client

    def _make_key(self, factor_id: str) -> str:
        """Generate cache key for emission factor."""
        return f"{self.CACHE_PREFIX}:{factor_id}"

    def get_factor(self, factor_id: str) -> Optional[Dict[str, Any]]:
        """
        Get emission factor from cache.
        Returns None if not found (cache miss).
        """
        key = self._make_key(factor_id)
        cached = self.redis.get(key)

        if cached:
            return json.loads(cached)
        return None

    def set_factor(self, factor_id: str, factor_data: Dict[str, Any],
                   ttl: int = DEFAULT_TTL) -> None:
        """
        Cache emission factor with TTL.
        Includes provenance hash for audit trail.
        """
        key = self._make_key(factor_id)

        # Add cache metadata
        factor_data["_cached_at"] = int(time.time())
        factor_data["_cache_hash"] = self._compute_hash(factor_data)

        self.redis.setex(key, ttl, json.dumps(factor_data))

    def get_or_load(self, factor_id: str,
                    loader_func: callable) -> Dict[str, Any]:
        """
        Get from cache or load from database.
        Implements cache-aside pattern.
        """
        # Try cache first
        cached = self.get_factor(factor_id)
        if cached:
            return cached

        # Cache miss - load from database
        factor_data = loader_func(factor_id)

        # Store in cache for next time
        if factor_data:
            self.set_factor(factor_id, factor_data)

        return factor_data

    def invalidate(self, factor_id: str) -> None:
        """Invalidate cached factor (on update)."""
        key = self._make_key(factor_id)
        self.redis.delete(key)

    def bulk_get(self, factor_ids: list) -> Dict[str, Dict[str, Any]]:
        """
        Bulk get multiple factors (reduces round trips).
        Uses MGET for efficiency.
        """
        keys = [self._make_key(fid) for fid in factor_ids]
        values = self.redis.mget(keys)

        result = {}
        for factor_id, value in zip(factor_ids, values):
            if value:
                result[factor_id] = json.loads(value)

        return result

    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash for audit trail."""
        # Remove cache metadata before hashing
        clean_data = {k: v for k, v in data.items()
                      if not k.startswith("_")}
        content = json.dumps(clean_data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


# Usage example
"""
redis_client = Redis(host='redis-master', port=6379, password='...')
cache = EmissionFactorCache(redis_client)

# Get emission factor (cache-aside pattern)
factor = cache.get_or_load(
    factor_id="ef_electricity_us_2024",
    loader_func=lambda fid: db.query_emission_factor(fid)
)

# Result (from cache on subsequent calls):
{
    "id": "ef_electricity_us_2024",
    "value": 0.417,
    "unit": "kgCO2e/kWh",
    "source": "EPA eGRID 2024",
    "region": "US",
    "year": 2024,
    "_cached_at": 1706918400,
    "_cache_hash": "a3f2b1c4d5e6..."
}
"""
```

### 5.2 Grid Intensity Caching (1-hour TTL)

```python
# grid_intensity_cache.py

import json
from datetime import datetime, timedelta
from redis import Redis
from typing import Optional, List, Dict

class GridIntensityCache:
    """
    Cache real-time grid carbon intensity data.
    Updates hourly from electricity grid APIs.
    """

    CACHE_PREFIX = "grid"
    DEFAULT_TTL = 3600  # 1 hour

    def __init__(self, redis_client: Redis):
        self.redis = redis_client

    def _make_key(self, region: str, timestamp: datetime = None) -> str:
        """Generate cache key for grid intensity."""
        if timestamp:
            hour = timestamp.strftime("%Y%m%d%H")
            return f"{self.CACHE_PREFIX}:{region}:{hour}"
        return f"{self.CACHE_PREFIX}:{region}:current"

    def get_current_intensity(self, region: str) -> Optional[Dict]:
        """Get current grid intensity for region."""
        key = self._make_key(region)
        cached = self.redis.get(key)
        return json.loads(cached) if cached else None

    def set_intensity(self, region: str, intensity_data: Dict,
                      ttl: int = DEFAULT_TTL) -> None:
        """Cache grid intensity data."""
        key = self._make_key(region)
        self.redis.setex(key, ttl, json.dumps(intensity_data))

        # Also store in time-indexed key for historical queries
        ts_key = self._make_key(region, datetime.utcnow())
        self.redis.setex(ts_key, 86400, json.dumps(intensity_data))  # 24h

    def get_historical(self, region: str,
                       hours: int = 24) -> List[Dict]:
        """Get historical intensity for past N hours."""
        results = []
        now = datetime.utcnow()

        for i in range(hours):
            ts = now - timedelta(hours=i)
            key = self._make_key(region, ts)
            cached = self.redis.get(key)
            if cached:
                results.append(json.loads(cached))

        return results

    def get_all_regions(self) -> Dict[str, Dict]:
        """Get current intensity for all cached regions."""
        pattern = f"{self.CACHE_PREFIX}:*:current"
        keys = self.redis.keys(pattern)

        if not keys:
            return {}

        values = self.redis.mget(keys)
        results = {}

        for key, value in zip(keys, values):
            if value:
                region = key.decode().split(":")[1]
                results[region] = json.loads(value)

        return results


# Usage example
"""
cache = GridIntensityCache(redis_client)

# Cache current grid intensity
cache.set_intensity("DE", {
    "region": "DE",
    "intensity": 385,  # gCO2/kWh
    "unit": "gCO2/kWh",
    "source": "ENTSO-E",
    "timestamp": "2026-02-03T14:00:00Z",
    "forecast_24h": [380, 375, 370, ...],
    "generation_mix": {
        "wind": 35,
        "solar": 15,
        "gas": 25,
        "coal": 15,
        "nuclear": 10
    }
})

# Get current intensity
intensity = cache.get_current_intensity("DE")
# Returns: {"region": "DE", "intensity": 385, ...}
"""
```

### 5.3 API Response Caching

```python
# api_response_cache.py

import json
import hashlib
from functools import wraps
from redis import Redis
from typing import Optional, Callable, Any

class APIResponseCache:
    """
    Cache API responses to reduce computation and DB load.
    Implements cache-aside with automatic key generation.
    """

    CACHE_PREFIX = "api"

    def __init__(self, redis_client: Redis):
        self.redis = redis_client

    def _make_key(self, endpoint: str, params: dict) -> str:
        """Generate deterministic cache key from endpoint and params."""
        params_str = json.dumps(params, sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:12]
        return f"{self.CACHE_PREFIX}:{endpoint}:{params_hash}"

    def get(self, endpoint: str, params: dict) -> Optional[dict]:
        """Get cached API response."""
        key = self._make_key(endpoint, params)
        cached = self.redis.get(key)
        return json.loads(cached) if cached else None

    def set(self, endpoint: str, params: dict, response: dict,
            ttl: int = 300) -> None:
        """Cache API response with TTL."""
        key = self._make_key(endpoint, params)
        self.redis.setex(key, ttl, json.dumps(response))

    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern."""
        full_pattern = f"{self.CACHE_PREFIX}:{pattern}*"
        keys = self.redis.keys(full_pattern)
        if keys:
            return self.redis.delete(*keys)
        return 0

    def cached(self, ttl: int = 300,
               key_params: list = None) -> Callable:
        """
        Decorator to cache API endpoint responses.

        Usage:
            @cache.cached(ttl=600, key_params=['org_id', 'report_type'])
            async def get_emissions_report(org_id, report_type):
                ...
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> Any:
                # Build params dict for cache key
                params = kwargs.copy()
                if key_params:
                    params = {k: kwargs.get(k) for k in key_params}

                endpoint = func.__name__

                # Try cache first
                cached = self.get(endpoint, params)
                if cached:
                    return cached

                # Cache miss - call function
                result = await func(*args, **kwargs)

                # Store in cache
                self.set(endpoint, params, result, ttl)

                return result
            return wrapper
        return decorator


# Usage example with FastAPI
"""
from fastapi import FastAPI, Depends

app = FastAPI()
cache = APIResponseCache(redis_client)

@app.get("/api/v1/emissions/{org_id}")
@cache.cached(ttl=300, key_params=['org_id', 'year'])
async def get_emissions(org_id: str, year: int):
    # This response will be cached for 5 minutes
    return await calculate_emissions(org_id, year)
"""
```

### 5.4 Rate Limiting (Token Bucket and Sliding Window)

```python
# rate_limiter.py

import time
from redis import Redis
from typing import Tuple

class RateLimiter:
    """
    Distributed rate limiting using Redis.
    Supports token bucket and sliding window algorithms.
    """

    def __init__(self, redis_client: Redis):
        self.redis = redis_client

    # ============================================================
    # TOKEN BUCKET ALGORITHM
    # ============================================================

    def token_bucket_check(self, key: str, rate: int,
                           capacity: int) -> Tuple[bool, int]:
        """
        Token bucket rate limiter.

        Args:
            key: Unique identifier (e.g., client_id:endpoint)
            rate: Tokens added per second
            capacity: Maximum bucket capacity

        Returns:
            (allowed, remaining_tokens)
        """
        now = time.time()
        bucket_key = f"rl:tb:{key}"

        # Lua script for atomic operation
        lua_script = """
        local key = KEYS[1]
        local rate = tonumber(ARGV[1])
        local capacity = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])

        local bucket = redis.call('HMGET', key, 'tokens', 'last_update')
        local tokens = tonumber(bucket[1]) or capacity
        local last_update = tonumber(bucket[2]) or now

        -- Add tokens based on time elapsed
        local elapsed = now - last_update
        tokens = math.min(capacity, tokens + (elapsed * rate))

        local allowed = 0
        if tokens >= 1 then
            tokens = tokens - 1
            allowed = 1
        end

        redis.call('HMSET', key, 'tokens', tokens, 'last_update', now)
        redis.call('EXPIRE', key, 60)

        return {allowed, math.floor(tokens)}
        """

        result = self.redis.eval(
            lua_script, 1, bucket_key, rate, capacity, now
        )
        return bool(result[0]), result[1]

    # ============================================================
    # SLIDING WINDOW LOG ALGORITHM
    # ============================================================

    def sliding_window_check(self, key: str, limit: int,
                             window_seconds: int) -> Tuple[bool, int]:
        """
        Sliding window rate limiter using sorted sets.
        More accurate than fixed windows, slightly more memory.

        Args:
            key: Unique identifier
            limit: Maximum requests in window
            window_seconds: Window size in seconds

        Returns:
            (allowed, current_count)
        """
        now = time.time()
        window_start = now - window_seconds
        window_key = f"rl:sw:{key}"

        # Use pipeline for atomicity
        pipe = self.redis.pipeline()

        # Remove old entries outside window
        pipe.zremrangebyscore(window_key, 0, window_start)

        # Count current entries
        pipe.zcard(window_key)

        # Add current request (optimistically)
        pipe.zadd(window_key, {f"{now}:{id(now)}": now})

        # Set expiry
        pipe.expire(window_key, window_seconds + 1)

        results = pipe.execute()
        current_count = results[1]

        if current_count >= limit:
            # Over limit - remove the entry we just added
            self.redis.zremrangebyscore(window_key, now, now + 1)
            return False, current_count

        return True, current_count + 1

    # ============================================================
    # FIXED WINDOW COUNTER (Simple, memory efficient)
    # ============================================================

    def fixed_window_check(self, key: str, limit: int,
                           window_seconds: int) -> Tuple[bool, int]:
        """
        Fixed window counter - simple and memory efficient.
        May have edge case issues at window boundaries.

        Args:
            key: Unique identifier
            limit: Maximum requests in window
            window_seconds: Window size in seconds

        Returns:
            (allowed, current_count)
        """
        window = int(time.time() / window_seconds)
        window_key = f"rl:fw:{key}:{window}"

        # Increment and get current count
        current = self.redis.incr(window_key)

        # Set expiry on first request in window
        if current == 1:
            self.redis.expire(window_key, window_seconds + 1)

        allowed = current <= limit
        return allowed, current


# Usage example with FastAPI middleware
"""
from fastapi import FastAPI, Request, HTTPException

app = FastAPI()
limiter = RateLimiter(redis_client)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Get client identifier
    client_id = request.headers.get("X-API-Key") or request.client.host
    endpoint = request.url.path

    # Check rate limit
    key = f"{client_id}:{endpoint}"
    allowed, remaining = limiter.sliding_window_check(
        key=key,
        limit=100,  # 100 requests
        window_seconds=60  # per minute
    )

    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"X-RateLimit-Remaining": "0"}
        )

    response = await call_next(request)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    return response
"""
```

### 5.5 Session Storage

```python
# session_store.py

import json
import secrets
from datetime import datetime
from redis import Redis
from typing import Optional, Dict, Any

class SessionStore:
    """
    Distributed session storage using Redis.
    Sub-millisecond access for authentication checks.
    """

    CACHE_PREFIX = "session"
    DEFAULT_TTL = 86400  # 24 hours

    def __init__(self, redis_client: Redis):
        self.redis = redis_client

    def _make_key(self, session_id: str) -> str:
        return f"{self.CACHE_PREFIX}:{session_id}"

    def create_session(self, user_id: str,
                       metadata: Dict[str, Any] = None) -> str:
        """
        Create new session and return session ID.
        """
        session_id = secrets.token_urlsafe(32)
        key = self._make_key(session_id)

        session_data = {
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }

        # Store as hash for efficient partial updates
        self.redis.hset(key, mapping={
            k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
            for k, v in session_data.items()
        })
        self.redis.expire(key, self.DEFAULT_TTL)

        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session data. Returns None if session doesn't exist.
        """
        key = self._make_key(session_id)
        data = self.redis.hgetall(key)

        if not data:
            return None

        # Decode hash values
        session = {}
        for k, v in data.items():
            k = k.decode() if isinstance(k, bytes) else k
            v = v.decode() if isinstance(v, bytes) else v
            try:
                session[k] = json.loads(v)
            except json.JSONDecodeError:
                session[k] = v

        return session

    def update_activity(self, session_id: str) -> bool:
        """
        Update last activity timestamp (sliding expiration).
        Returns False if session doesn't exist.
        """
        key = self._make_key(session_id)

        if not self.redis.exists(key):
            return False

        pipe = self.redis.pipeline()
        pipe.hset(key, "last_activity", datetime.utcnow().isoformat())
        pipe.expire(key, self.DEFAULT_TTL)  # Reset TTL
        pipe.execute()

        return True

    def destroy_session(self, session_id: str) -> bool:
        """Delete session (logout)."""
        key = self._make_key(session_id)
        return bool(self.redis.delete(key))

    def get_user_sessions(self, user_id: str) -> list:
        """Get all active sessions for a user."""
        pattern = f"{self.CACHE_PREFIX}:*"
        sessions = []

        for key in self.redis.scan_iter(pattern):
            data = self.redis.hget(key, "user_id")
            if data and data.decode() == user_id:
                session_id = key.decode().split(":")[-1]
                sessions.append(session_id)

        return sessions

    def destroy_user_sessions(self, user_id: str) -> int:
        """Destroy all sessions for a user (force logout)."""
        sessions = self.get_user_sessions(user_id)
        if sessions:
            keys = [self._make_key(sid) for sid in sessions]
            return self.redis.delete(*keys)
        return 0


# Usage example
"""
session_store = SessionStore(redis_client)

# Create session on login
session_id = session_store.create_session(
    user_id="user_123",
    metadata={
        "ip": "192.168.1.1",
        "user_agent": "Mozilla/5.0...",
        "org_id": "org_456"
    }
)

# Validate session on each request
session = session_store.get_session(session_id)
if session:
    session_store.update_activity(session_id)  # Extend TTL
    user_id = session["user_id"]
else:
    # Session expired or invalid
    raise UnauthorizedException()

# Logout
session_store.destroy_session(session_id)
"""
```

### 5.6 Redis Streams for Message Queuing

```python
# message_queue.py

import json
import time
from redis import Redis
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass

@dataclass
class Message:
    id: str
    stream: str
    data: Dict[str, Any]
    timestamp: float

class MessageQueue:
    """
    Message queue using Redis Streams.
    Provides reliable, ordered message delivery with consumer groups.
    """

    def __init__(self, redis_client: Redis):
        self.redis = redis_client

    def create_stream(self, stream_name: str,
                      consumer_group: str) -> bool:
        """
        Create stream and consumer group.
        Safe to call multiple times (idempotent).
        """
        try:
            self.redis.xgroup_create(
                stream_name,
                consumer_group,
                id='0',  # Start from beginning
                mkstream=True  # Create stream if not exists
            )
            return True
        except Exception as e:
            if "BUSYGROUP" in str(e):
                return True  # Group already exists
            raise

    def publish(self, stream_name: str,
                data: Dict[str, Any],
                max_len: int = 100000) -> str:
        """
        Publish message to stream.
        Returns message ID.
        """
        # Serialize nested structures
        flat_data = {
            k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
            for k, v in data.items()
        }

        message_id = self.redis.xadd(
            stream_name,
            flat_data,
            maxlen=max_len,  # Cap stream size
            approximate=True  # More efficient trimming
        )

        return message_id.decode() if isinstance(message_id, bytes) else message_id

    def consume(self, stream_name: str, consumer_group: str,
                consumer_name: str, count: int = 10,
                block_ms: int = 5000) -> List[Message]:
        """
        Consume messages from stream (blocking).
        Messages must be acknowledged after processing.
        """
        messages = []

        result = self.redis.xreadgroup(
            consumer_group,
            consumer_name,
            {stream_name: '>'},  # Only new messages
            count=count,
            block=block_ms
        )

        if not result:
            return messages

        for stream, entries in result:
            stream = stream.decode() if isinstance(stream, bytes) else stream

            for entry_id, data in entries:
                entry_id = entry_id.decode() if isinstance(entry_id, bytes) else entry_id

                # Deserialize data
                parsed_data = {}
                for k, v in data.items():
                    k = k.decode() if isinstance(k, bytes) else k
                    v = v.decode() if isinstance(v, bytes) else v
                    try:
                        parsed_data[k] = json.loads(v)
                    except json.JSONDecodeError:
                        parsed_data[k] = v

                messages.append(Message(
                    id=entry_id,
                    stream=stream,
                    data=parsed_data,
                    timestamp=float(entry_id.split('-')[0]) / 1000
                ))

        return messages

    def acknowledge(self, stream_name: str, consumer_group: str,
                    message_id: str) -> bool:
        """Acknowledge message as processed."""
        result = self.redis.xack(stream_name, consumer_group, message_id)
        return result > 0

    def get_pending(self, stream_name: str, consumer_group: str,
                    consumer_name: str = None,
                    count: int = 100) -> List[Dict]:
        """Get pending (unacknowledged) messages."""
        result = self.redis.xpending_range(
            stream_name,
            consumer_group,
            min='-',
            max='+',
            count=count,
            consumername=consumer_name
        )

        return [
            {
                'id': msg['message_id'].decode(),
                'consumer': msg['consumer'].decode(),
                'idle_time_ms': msg['time_since_delivered'],
                'delivery_count': msg['times_delivered']
            }
            for msg in result
        ]

    def claim_stale(self, stream_name: str, consumer_group: str,
                    consumer_name: str, min_idle_ms: int = 60000,
                    count: int = 10) -> List[Message]:
        """
        Claim messages that have been pending too long.
        Used for recovering from failed consumers.
        """
        # First get pending message IDs
        pending = self.get_pending(stream_name, consumer_group)
        stale_ids = [
            p['id'] for p in pending
            if p['idle_time_ms'] >= min_idle_ms
        ][:count]

        if not stale_ids:
            return []

        # Claim the messages
        result = self.redis.xclaim(
            stream_name,
            consumer_group,
            consumer_name,
            min_idle_ms,
            stale_ids
        )

        messages = []
        for entry_id, data in result:
            entry_id = entry_id.decode() if isinstance(entry_id, bytes) else entry_id
            parsed_data = {
                k.decode(): v.decode() for k, v in data.items()
            }
            messages.append(Message(
                id=entry_id,
                stream=stream_name,
                data=parsed_data,
                timestamp=float(entry_id.split('-')[0]) / 1000
            ))

        return messages


# Usage example - Producer
"""
mq = MessageQueue(redis_client)

# Create stream and consumer group
mq.create_stream("calculations", "calc-workers")

# Publish calculation job
message_id = mq.publish("calculations", {
    "job_type": "emission_calculation",
    "org_id": "org_123",
    "facility_id": "fac_456",
    "period_start": "2026-01-01",
    "period_end": "2026-01-31",
    "priority": "high"
})
print(f"Published job: {message_id}")
"""

# Usage example - Consumer
"""
mq = MessageQueue(redis_client)

while True:
    # Consume messages (blocks for 5 seconds if none available)
    messages = mq.consume(
        stream_name="calculations",
        consumer_group="calc-workers",
        consumer_name="worker-1",
        count=10
    )

    for msg in messages:
        try:
            # Process the job
            process_calculation(msg.data)

            # Acknowledge successful processing
            mq.acknowledge("calculations", "calc-workers", msg.id)

        except Exception as e:
            # Don't acknowledge - message will be retried
            log.error(f"Failed to process {msg.id}: {e}")

    # Also check for stale messages (failed workers)
    stale = mq.claim_stale(
        stream_name="calculations",
        consumer_group="calc-workers",
        consumer_name="worker-1",
        min_idle_ms=300000  # 5 minutes
    )
    for msg in stale:
        # Reprocess stale messages
        ...
"""
```

---

## 6. Security Requirements

### 6.1 Authentication and Authorization

| Requirement | Implementation | Status |
|-------------|----------------|--------|
| **AUTH Password** | `requirepass` with strong password | Required |
| **Master Auth** | `masterauth` for replica authentication | Required |
| **ACL Users** | Separate users per service with minimal permissions | Required |
| **Sentinel Auth** | `requirepass` on Sentinel instances | Required |

### 6.2 Redis ACL Configuration

```ini
# users.acl

# Admin user (for operations only)
user admin on >$ADMIN_PASSWORD ~* &* +@all

# Application user (read/write to app keys)
user app_greenlang on >$APP_PASSWORD ~ef:* ~grid:* ~session:* ~api:* ~rl:* &* +@read +@write +@stream -@admin -@dangerous

# Monitoring user (read-only for metrics)
user monitor on >$MONITOR_PASSWORD ~* &* +@read +info +client +slowlog +latency -@write -@admin -@dangerous

# Default user disabled
user default off
```

### 6.3 TLS Encryption Configuration

```yaml
# redis-tls-config.yaml

# Server TLS Configuration
tls-port 6379
port 0  # Disable non-TLS port

tls-cert-file /etc/redis/tls/redis.crt
tls-key-file /etc/redis/tls/redis.key
tls-ca-cert-file /etc/redis/tls/ca.crt

tls-auth-clients yes  # Require client certificates
tls-protocols "TLSv1.2 TLSv1.3"
tls-ciphersuites "TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256"
tls-prefer-server-ciphers yes

# Replication TLS
tls-replication yes

# Sentinel TLS
sentinel tls-cert-file /etc/redis/tls/sentinel.crt
sentinel tls-key-file /etc/redis/tls/sentinel.key
sentinel tls-ca-cert-file /etc/redis/tls/ca.crt
```

### 6.4 Network Security

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Network Security Architecture                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         VPC (10.0.0.0/16)                            │    │
│  │                                                                      │    │
│  │  ┌─────────────────────────────────────────────────────────────┐   │    │
│  │  │              PRIVATE SUBNET (10.0.10.0/24)                   │   │    │
│  │  │              (No Internet Access)                            │   │    │
│  │  │                                                              │   │    │
│  │  │  ┌──────────────────────────────────────────────────────┐   │   │    │
│  │  │  │              Redis Security Group                     │   │   │    │
│  │  │  │                                                       │   │   │    │
│  │  │  │  Inbound Rules:                                      │   │   │    │
│  │  │  │  - Port 6379: From app-sg only                       │   │   │    │
│  │  │  │  - Port 26379: From app-sg only (Sentinel)           │   │   │    │
│  │  │  │  - Port 6379: From redis-sg (replication)            │   │   │    │
│  │  │  │                                                       │   │   │    │
│  │  │  │  Outbound Rules:                                     │   │   │    │
│  │  │  │  - Port 6379: To redis-sg only (replication)         │   │   │    │
│  │  │  │  - Port 26379: To redis-sg only (Sentinel)           │   │   │    │
│  │  │  │                                                       │   │   │    │
│  │  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │   │    │
│  │  │  │  │ Redis Master│  │ Redis Rep 1 │  │ Redis Rep 2 │  │   │   │    │
│  │  │  │  │ 10.0.10.10  │  │ 10.0.10.11  │  │ 10.0.10.12  │  │   │   │    │
│  │  │  │  └─────────────┘  └─────────────┘  └─────────────┘  │   │   │    │
│  │  │  └──────────────────────────────────────────────────────┘   │   │    │
│  │  └─────────────────────────────────────────────────────────────┘   │    │
│  │                                                                      │    │
│  │  ┌─────────────────────────────────────────────────────────────┐   │    │
│  │  │              EKS SUBNET (10.0.20.0/24)                       │   │    │
│  │  │                                                              │   │    │
│  │  │  ┌──────────────────────────────────────────────────────┐   │   │    │
│  │  │  │              Application Security Group               │   │   │    │
│  │  │  │                                                       │   │   │    │
│  │  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │   │    │
│  │  │  │  │ GL-CSRD-APP │  │ GL-CBAM-APP │  │ GL-EUDR-APP │  │   │   │    │
│  │  │  │  └─────────────┘  └─────────────┘  └─────────────┘  │   │   │    │
│  │  │  └──────────────────────────────────────────────────────┘   │   │    │
│  │  └─────────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.5 AWS Secrets Manager Integration

```python
# secrets_manager.py

import boto3
import json
from functools import lru_cache

class RedisSecretsManager:
    """
    Manage Redis credentials via AWS Secrets Manager.
    Supports automatic rotation.
    """

    def __init__(self, region: str = "us-east-1"):
        self.client = boto3.client('secretsmanager', region_name=region)
        self.secret_id = "greenlang/redis/credentials"

    @lru_cache(maxsize=1)
    def get_credentials(self) -> dict:
        """
        Get Redis credentials from Secrets Manager.
        Cached for performance (clear on rotation).
        """
        response = self.client.get_secret_value(SecretId=self.secret_id)
        return json.loads(response['SecretString'])

    def get_redis_password(self) -> str:
        """Get Redis AUTH password."""
        return self.get_credentials()['password']

    def get_redis_url(self, use_tls: bool = True) -> str:
        """Get full Redis connection URL."""
        creds = self.get_credentials()
        protocol = "rediss" if use_tls else "redis"
        return f"{protocol}://:{creds['password']}@{creds['host']}:{creds['port']}"

    def clear_cache(self):
        """Clear cached credentials (call on rotation)."""
        self.get_credentials.cache_clear()


# Secrets Manager secret structure
"""
{
    "password": "super-secure-password-here",
    "host": "redis-master.greenlang.internal",
    "port": 6379,
    "sentinel_password": "sentinel-password-here",
    "acl_users": {
        "app_greenlang": "app-user-password",
        "monitor": "monitor-password"
    }
}
"""

# Terraform for secret creation
"""
resource "aws_secretsmanager_secret" "redis_credentials" {
  name        = "greenlang/redis/credentials"
  description = "Redis cluster credentials for GreenLang"

  tags = {
    Environment = "production"
    Application = "greenlang"
  }
}

resource "aws_secretsmanager_secret_rotation" "redis_rotation" {
  secret_id           = aws_secretsmanager_secret.redis_credentials.id
  rotation_lambda_arn = aws_lambda_function.redis_rotate.arn

  rotation_rules {
    automatically_after_days = 30
  }
}
"""
```

---

## 7. Monitoring and Alerting

### 7.1 Key Metrics

| Metric | Description | Warning | Critical | Action |
|--------|-------------|---------|----------|--------|
| **Memory Usage** | Used memory vs maxmemory | > 75% | > 90% | Scale up or tune eviction |
| **Connected Clients** | Active connections | > 5000 | > 8000 | Check connection leaks |
| **Cache Hit Rate** | Hits / (Hits + Misses) | < 90% | < 80% | Review TTLs, increase memory |
| **Replication Lag** | Bytes behind master | > 1MB | > 10MB | Check network, replica health |
| **Evicted Keys** | Keys evicted due to memory | > 100/min | > 1000/min | Increase memory |
| **Blocked Clients** | Clients waiting on blocking ops | > 10 | > 50 | Check slow operations |
| **Commands/sec** | Operations per second | N/A | N/A | Baseline monitoring |
| **Latency** | Command execution time | > 1ms avg | > 5ms avg | Check slow commands |
| **Keyspace Misses** | Failed key lookups | Trend up | Trend up | Cache warming needed |

### 7.2 Prometheus Metrics (redis_exporter)

```yaml
# prometheus-redis-rules.yaml

groups:
  - name: redis_alerts
    rules:
      # Memory alerts
      - alert: RedisMemoryHigh
        expr: redis_memory_used_bytes / redis_memory_max_bytes > 0.9
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Redis memory usage critical"
          description: "Redis memory usage is {{ $value | humanizePercentage }}"

      - alert: RedisMemoryWarning
        expr: redis_memory_used_bytes / redis_memory_max_bytes > 0.75
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Redis memory usage high"
          description: "Redis memory usage is {{ $value | humanizePercentage }}"

      # Connection alerts
      - alert: RedisConnectionsHigh
        expr: redis_connected_clients > 8000
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Redis connections critical"
          description: "{{ $value }} clients connected"

      # Replication alerts
      - alert: RedisReplicationBroken
        expr: redis_connected_slaves < 1
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Redis replication broken"
          description: "No replicas connected to master"

      - alert: RedisReplicationLag
        expr: redis_replication_lag_bytes > 10000000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Redis replication lag high"
          description: "Replication lag is {{ $value | humanize1024 }}"

      # Cache efficiency
      - alert: RedisCacheHitRateLow
        expr: >
          rate(redis_keyspace_hits_total[5m]) /
          (rate(redis_keyspace_hits_total[5m]) + rate(redis_keyspace_misses_total[5m])) < 0.8
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Redis cache hit rate low"
          description: "Cache hit rate is {{ $value | humanizePercentage }}"

      # Eviction alerts
      - alert: RedisEvictionsHigh
        expr: rate(redis_evicted_keys_total[5m]) > 100
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Redis evictions high"
          description: "{{ $value }} keys evicted per second"

      # Latency alerts
      - alert: RedisLatencyHigh
        expr: redis_commands_duration_seconds_total / redis_commands_processed_total > 0.005
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Redis latency critical"
          description: "Average command latency is {{ $value }}s"

      # Sentinel alerts
      - alert: RedisSentinelDown
        expr: redis_sentinel_master_status != 1
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Redis Sentinel reports master down"
          description: "Sentinel cannot reach master"
```

### 7.3 Grafana Dashboard

```json
{
  "dashboard": {
    "title": "GreenLang Redis Cluster",
    "uid": "redis-cluster",
    "panels": [
      {
        "title": "Memory Usage",
        "type": "gauge",
        "targets": [{
          "expr": "redis_memory_used_bytes / redis_memory_max_bytes * 100",
          "legendFormat": "Memory %"
        }],
        "fieldConfig": {
          "defaults": {
            "max": 100,
            "thresholds": {
              "steps": [
                {"color": "green", "value": 0},
                {"color": "yellow", "value": 75},
                {"color": "red", "value": 90}
              ]
            }
          }
        }
      },
      {
        "title": "Cache Hit Rate",
        "type": "stat",
        "targets": [{
          "expr": "rate(redis_keyspace_hits_total[5m]) / (rate(redis_keyspace_hits_total[5m]) + rate(redis_keyspace_misses_total[5m])) * 100",
          "legendFormat": "Hit Rate %"
        }]
      },
      {
        "title": "Commands/sec",
        "type": "graph",
        "targets": [{
          "expr": "rate(redis_commands_processed_total[1m])",
          "legendFormat": "{{instance}}"
        }]
      },
      {
        "title": "Connected Clients",
        "type": "graph",
        "targets": [{
          "expr": "redis_connected_clients",
          "legendFormat": "{{instance}}"
        }]
      },
      {
        "title": "Replication Lag",
        "type": "graph",
        "targets": [{
          "expr": "redis_replication_lag_bytes",
          "legendFormat": "{{instance}}"
        }]
      },
      {
        "title": "Evicted Keys",
        "type": "graph",
        "targets": [{
          "expr": "rate(redis_evicted_keys_total[5m])",
          "legendFormat": "Evictions/sec"
        }]
      },
      {
        "title": "Command Latency",
        "type": "heatmap",
        "targets": [{
          "expr": "rate(redis_commands_duration_seconds_bucket[5m])",
          "legendFormat": "{{le}}"
        }]
      },
      {
        "title": "Keys by Database",
        "type": "piechart",
        "targets": [{
          "expr": "redis_db_keys",
          "legendFormat": "db{{db}}"
        }]
      }
    ]
  }
}
```

### 7.4 PagerDuty Integration

```yaml
# alertmanager-config.yaml

global:
  resolve_timeout: 5m
  pagerduty_url: 'https://events.pagerduty.com/v2/enqueue'

route:
  receiver: 'default-receiver'
  group_by: ['alertname', 'severity']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  routes:
    - match:
        severity: critical
      receiver: 'pagerduty-critical'
      continue: true
    - match:
        severity: warning
      receiver: 'slack-warnings'

receivers:
  - name: 'default-receiver'
    slack_configs:
      - api_url: '${SLACK_WEBHOOK_URL}'
        channel: '#infrastructure-alerts'
        title: '{{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

  - name: 'pagerduty-critical'
    pagerduty_configs:
      - service_key: '${PAGERDUTY_SERVICE_KEY}'
        severity: critical
        description: '{{ .GroupLabels.alertname }}'
        details:
          summary: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
          description: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

  - name: 'slack-warnings'
    slack_configs:
      - api_url: '${SLACK_WEBHOOK_URL}'
        channel: '#infrastructure-warnings'
        color: 'warning'
        title: '{{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
```

---

## 8. Backup and Recovery

### 8.1 RDB Snapshot Schedule

| Snapshot Type | Frequency | Retention | Storage |
|--------------|-----------|-----------|---------|
| **Automatic** | Every 15 minutes | 24 hours | Local + S3 |
| **Daily** | 00:00 UTC | 7 days | S3 |
| **Weekly** | Sunday 00:00 UTC | 4 weeks | S3 |
| **Monthly** | 1st of month | 12 months | S3 + Glacier |

```ini
# RDB snapshot configuration
save 900 1        # 15 min if 1 change
save 300 10       # 5 min if 10 changes
save 60 10000     # 1 min if 10000 changes

rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir /data/redis
```

### 8.2 AOF Persistence Configuration

```ini
# AOF configuration
appendonly yes
appendfilename "appendonly.aof"
appenddirname "appendonlydir"

# fsync policy (everysec = good balance)
appendfsync everysec

# AOF rewrite settings
no-appendfsync-on-rewrite no
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb

# Use RDB preamble for faster loads
aof-use-rdb-preamble yes

# Truncated AOF handling
aof-load-truncated yes
```

### 8.3 Point-in-Time Recovery

```bash
#!/bin/bash
# redis-pitr-restore.sh
# Restore Redis to a specific point in time

RESTORE_TIME="2026-02-03T14:30:00Z"
BACKUP_BUCKET="s3://greenlang-redis-backups"
RESTORE_DIR="/data/redis-restore"

# 1. Find the RDB snapshot before restore time
SNAPSHOT=$(aws s3 ls ${BACKUP_BUCKET}/rdb/ \
  | awk -v rt="${RESTORE_TIME}" '$1"T"$2 < rt {f=$4} END {print f}')

echo "Using snapshot: ${SNAPSHOT}"

# 2. Download RDB snapshot
aws s3 cp ${BACKUP_BUCKET}/rdb/${SNAPSHOT} ${RESTORE_DIR}/dump.rdb

# 3. Download AOF files from snapshot time to restore time
aws s3 sync ${BACKUP_BUCKET}/aof/ ${RESTORE_DIR}/aof/ \
  --exclude "*" \
  --include "appendonly-*.aof"

# 4. Stop Redis
redis-cli SHUTDOWN NOSAVE

# 5. Replace data files
cp ${RESTORE_DIR}/dump.rdb /data/redis/
cp -r ${RESTORE_DIR}/aof/* /data/redis/appendonlydir/

# 6. Start Redis (will replay AOF)
redis-server /etc/redis/redis.conf

# 7. Verify restore
redis-cli INFO persistence
```

### 8.4 Cross-Region Backup Strategy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Cross-Region Backup Architecture                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  US-EAST-1 (Primary)                       US-WEST-2 (DR)                   │
│  ┌─────────────────────────────┐           ┌─────────────────────────────┐  │
│  │                             │           │                             │  │
│  │  ┌───────────────────────┐  │           │  ┌───────────────────────┐  │  │
│  │  │    Redis Cluster      │  │           │  │    S3 Bucket (DR)     │  │  │
│  │  │    (Production)       │  │           │  │                       │  │  │
│  │  │                       │  │           │  │  RDB snapshots        │  │  │
│  │  │  ┌─────┐ ┌─────┐     │  │           │  │  AOF archives         │  │  │
│  │  │  │Master│ │Repl │     │  │           │  │  90 days retention    │  │  │
│  │  │  └──┬──┘ └─────┘     │  │           │  │                       │  │  │
│  │  │     │                 │  │           │  └───────────────────────┘  │  │
│  │  └─────┼─────────────────┘  │           │              ▲              │  │
│  │        │                    │           │              │              │  │
│  │        ▼                    │           │              │              │  │
│  │  ┌───────────────────────┐  │           │              │              │  │
│  │  │    S3 Bucket          │  │    S3     │              │              │  │
│  │  │    (Primary)          │──┼──Replication──────────────┘              │  │
│  │  │                       │  │           │                             │  │
│  │  │  RDB snapshots        │  │           │                             │  │
│  │  │  AOF archives         │  │           │                             │  │
│  │  │  35 days retention    │  │           │                             │  │
│  │  └───────────────────────┘  │           │                             │  │
│  │                             │           │                             │  │
│  └─────────────────────────────┘           └─────────────────────────────┘  │
│                                                                              │
│  Recovery Options:                                                           │
│  1. Same-region: Restore from us-east-1 S3 (fastest)                        │
│  2. Cross-region: Restore from us-west-2 S3 (DR scenario)                   │
│  3. Point-in-time: Combine RDB + AOF replay                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Cost Estimation

### 9.1 AWS ElastiCache Costs

| Component | Instance Type | Quantity | Monthly Cost |
|-----------|--------------|----------|--------------|
| **Master Node** | cache.r6g.xlarge | 1 | $292.00 |
| **Replica Node 1** | cache.r6g.xlarge | 1 | $292.00 |
| **Replica Node 2** | cache.r6g.large | 1 | $146.00 |
| **Data Transfer** | Inter-AZ | ~500 GB | $10.00 |
| **Backup Storage** | S3 Standard | ~50 GB | $1.15 |
| **Secrets Manager** | 3 secrets | 3 | $1.20 |
| **CloudWatch** | Metrics + Logs | - | $15.00 |
| **Total ElastiCache** | | | **~$757/month** |

### 9.2 Self-Managed (K8s) Costs

| Component | Instance Type | Quantity | Monthly Cost |
|-----------|--------------|----------|--------------|
| **EKS Node (Master)** | r6g.xlarge | 1 | $147.00 |
| **EKS Node (Replica 1)** | r6g.xlarge | 1 | $147.00 |
| **EKS Node (Replica 2)** | r6g.large | 1 | $73.50 |
| **EBS Storage** | gp3 100GB x 3 | 3 | $24.00 |
| **Data Transfer** | Inter-AZ | ~500 GB | $10.00 |
| **S3 Backup** | Standard | ~50 GB | $1.15 |
| **Engineering Overhead** | Operations | ~5 hrs/mo | $500.00 |
| **Total Self-Managed** | | | **~$902/month** |

### 9.3 Cost Comparison Summary

| Approach | Monthly Cost | Pros | Cons |
|----------|-------------|------|------|
| **ElastiCache** | ~$757 | Managed, automatic failover, less ops | Less config flexibility |
| **Self-Managed** | ~$902 | Full control, any Redis version | Higher ops overhead |

**Recommendation:** Use **AWS ElastiCache** for production. The managed service reduces operational burden and provides better reliability guarantees, offsetting the slightly higher raw infrastructure cost.

---

## 10. Implementation Phases

### 10.1 Phase 1: Core Cluster Setup (Week 1)

**Duration:** 5 days
**Owner:** Infrastructure Team

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Provision ElastiCache cluster | Redis cluster running |
| 1 | Configure security groups | Network isolation |
| 2 | Set up Secrets Manager | Credentials stored |
| 2 | Configure Redis settings | Optimized redis.conf |
| 3 | Deploy redis_exporter | Metrics collection |
| 3 | Configure CloudWatch alarms | Basic alerting |
| 4 | Test basic operations | CRUD operations verified |
| 4 | Load test (10K ops/sec) | Performance baseline |
| 5 | Documentation | Runbook created |

**Exit Criteria:**
- [ ] Redis cluster accessible from EKS
- [ ] Authentication working (ACL + TLS)
- [ ] Basic metrics in Prometheus
- [ ] 10K ops/sec sustained

### 10.2 Phase 2: Sentinel High Availability (Days 6-8)

**Duration:** 3 days
**Owner:** Infrastructure Team

| Day | Task | Deliverable |
|-----|------|-------------|
| 6 | Configure Sentinel nodes | 3 Sentinels deployed |
| 6 | Test master detection | Sentinels monitoring master |
| 7 | Configure automatic failover | Failover parameters set |
| 7 | Test failover (kill master) | Automatic promotion |
| 8 | Test client reconnection | Transparent failover |
| 8 | Document failover procedure | Runbook updated |

**Exit Criteria:**
- [ ] 3 Sentinel nodes operational
- [ ] Automatic failover < 30 seconds
- [ ] Clients reconnect transparently
- [ ] Zero data loss during failover

### 10.3 Phase 3: Monitoring & Alerting (Days 9-10)

**Duration:** 2 days
**Owner:** Infrastructure Team + SRE

| Day | Task | Deliverable |
|-----|------|-------------|
| 9 | Create Grafana dashboard | Redis dashboard |
| 9 | Configure Prometheus alerts | Alert rules deployed |
| 10 | Set up PagerDuty integration | Critical alerts routed |
| 10 | Test alert scenarios | Alerts verified |

**Exit Criteria:**
- [ ] Grafana dashboard operational
- [ ] All critical metrics alerting
- [ ] PagerDuty receiving alerts
- [ ] Runbook for each alert

### 10.4 Phase 4: Integration Testing (Days 11-12)

**Duration:** 2 days
**Owner:** Infrastructure + Backend Teams

| Day | Task | Deliverable |
|-----|------|-------------|
| 11 | Integrate emission factor cache | Cache working in dev |
| 11 | Integrate session storage | Sessions in Redis |
| 12 | Integrate rate limiting | Rate limits enforced |
| 12 | End-to-end testing | All use cases verified |

**Exit Criteria:**
- [ ] All caching use cases working
- [ ] Rate limiting protecting APIs
- [ ] Sessions stored in Redis
- [ ] Performance targets met

### 10.5 Implementation Timeline

```
Week 1                              Week 2
┌─────┬─────┬─────┬─────┬─────┐    ┌─────┬─────┬─────┬─────┬─────┐
│ Mon │ Tue │ Wed │ Thu │ Fri │    │ Mon │ Tue │ Wed │ Thu │ Fri │
├─────┴─────┴─────┴─────┴─────┤    ├─────┴─────┴─────┴─────┴─────┤
│  PHASE 1: Core Cluster      │    │P2: HA │P3: Mon│ P4: Test   │
│  Setup (5 days)             │    │(3 days)│(2 days)│ (2 days)  │
│                             │    │        │       │            │
│  - Provision cluster        │    │- Sent. │- Graf.│- EF cache  │
│  - Security groups          │    │- Failov│- Prom │- Sessions  │
│  - Secrets Manager          │    │- Client│- PD   │- Rate lim  │
│  - Metrics                  │    │        │       │- E2E test  │
│  - Load testing             │    │        │       │            │
└─────────────────────────────┘    └─────────────────────────────┘

Total Duration: 12 days (2.5 weeks)
```

---

## 11. Acceptance Criteria

### 11.1 Performance Benchmarks

| Metric | Target | Test Method |
|--------|--------|-------------|
| **GET Latency (p50)** | < 1ms | redis-benchmark |
| **GET Latency (p99)** | < 5ms | redis-benchmark |
| **SET Latency (p50)** | < 1ms | redis-benchmark |
| **SET Latency (p99)** | < 5ms | redis-benchmark |
| **Throughput** | > 100K ops/sec | redis-benchmark |
| **Memory Efficiency** | < 80% fragmentation | INFO memory |
| **Connection Time** | < 10ms | Application metrics |

**Benchmark Command:**
```bash
redis-benchmark -h redis-master -p 6379 -a ${PASSWORD} \
  -t get,set,lpush,lpop,sadd,spop,hset,mset \
  -n 1000000 -c 100 -q --csv
```

### 11.2 Failover Tests

| Test | Expected Outcome | Pass Criteria |
|------|------------------|---------------|
| **Master Crash** | Replica promoted | < 30 seconds |
| **Network Partition** | New master elected | < 30 seconds |
| **Sentinel Failure** | Remaining sentinels quorum | No impact |
| **Rolling Restart** | Zero downtime | 0 failed requests |
| **Client Reconnect** | Transparent failover | < 10 seconds |

**Failover Test Procedure:**
```bash
# 1. Start monitoring
watch -n 1 'redis-cli -h sentinel-1 SENTINEL master greenlang-cache'

# 2. Kill master (simulate crash)
redis-cli -h redis-master DEBUG SEGFAULT

# 3. Verify failover
# - New master elected within 30 seconds
# - Clients reconnected
# - No data loss (check write count)
```

### 11.3 Data Integrity Tests

| Test | Expected Outcome | Pass Criteria |
|------|------------------|---------------|
| **Write + Read** | Data consistent | 100% match |
| **AOF Replay** | Data restored | 100% recovery |
| **RDB Restore** | Data restored | 100% recovery |
| **Replication** | Replicas in sync | 0 bytes lag |
| **Eviction** | LRU policy | Oldest keys evicted |

### 11.4 Security Tests

| Test | Expected Outcome | Pass Criteria |
|------|------------------|---------------|
| **No Auth** | Connection refused | AUTH required |
| **Invalid Auth** | Connection refused | Error returned |
| **TLS Required** | Non-TLS rejected | TLS enforced |
| **ACL Enforcement** | Unauthorized ops blocked | Permission denied |
| **Network Isolation** | External access blocked | Timeout |

---

## 12. Dependencies

### 12.1 Infrastructure Dependencies

| Dependency | ID | Status | Owner | Notes |
|------------|-----|--------|-------|-------|
| **EKS Cluster** | INFRA-001 | Complete | Infra Team | Required for K8s deployment |
| **VPC Networking** | INFRA-001 | Complete | Infra Team | Private subnets for Redis |
| **Secrets Manager** | - | Available | AWS | For credential storage |
| **S3 Buckets** | - | To Create | Infra Team | For backups |

### 12.2 Application Dependencies

| Dependency | Owner | Integration Point |
|------------|-------|-------------------|
| **GL-CSRD-APP** | Backend Team | Emission factor caching |
| **GL-CBAM-APP** | Backend Team | Grid intensity caching |
| **GL-Auth-Service** | Backend Team | Session storage |
| **API Gateway** | Backend Team | Rate limiting |

### 12.3 Dependency Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Dependency Graph                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                     ┌───────────────────────────┐                           │
│                     │      INFRA-001            │                           │
│                     │   EKS Cluster + VPC       │                           │
│                     │        (Complete)          │                           │
│                     └─────────────┬─────────────┘                           │
│                                   │                                          │
│                     ┌─────────────┼─────────────┐                           │
│                     │             │             │                           │
│                     ▼             ▼             ▼                           │
│  ┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐   │
│  │    INFRA-002        │ │    INFRA-003        │ │    INFRA-004        │   │
│  │  PostgreSQL +       │ │  Redis Caching      │ │  (Future)           │   │
│  │  TimescaleDB        │ │  Cluster            │ │                     │   │
│  │    (Complete)       │ │  (This PRD)         │ │                     │   │
│  └─────────────────────┘ └──────────┬──────────┘ └─────────────────────┘   │
│                                     │                                        │
│                     ┌───────────────┼───────────────┐                       │
│                     │               │               │                       │
│                     ▼               ▼               ▼                       │
│  ┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐   │
│  │    GL-CSRD-APP      │ │    GL-CBAM-APP      │ │    GL-Auth-Service  │   │
│  │                     │ │                     │ │                     │   │
│  │ - EF Caching        │ │ - Grid Intensity    │ │ - Session Storage   │   │
│  │ - API Response      │ │ - API Response      │ │ - Rate Limiting     │   │
│  └─────────────────────┘ └─────────────────────┘ └─────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 13. Risks and Mitigations

### 13.1 Risk Matrix

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Memory exhaustion** | Medium | High | Configure maxmemory + LRU eviction; monitor usage |
| **Failover during peak** | Low | High | Test failover under load; tune timeouts |
| **Data loss on crash** | Low | Medium | Enable AOF with everysec; test recovery |
| **Connection storms** | Medium | Medium | Connection pooling; rate limiting |
| **Security breach** | Low | Critical | TLS + AUTH + ACL; network isolation |
| **Replication lag** | Medium | Medium | Monitor lag; alert on > 1MB |
| **Cost overrun** | Low | Low | Reserved instances; right-size |

### 13.2 Mitigation Details

**Memory Exhaustion:**
```ini
# Prevention
maxmemory 6gb
maxmemory-policy allkeys-lru
maxmemory-samples 10

# Monitoring
- Alert at 75% memory usage (warning)
- Alert at 90% memory usage (critical)
- Review TTLs and eviction patterns
```

**Connection Storms:**
```python
# Application-side connection pooling
pool = redis.ConnectionPool(
    host='redis-master',
    port=6379,
    max_connections=20,
    retry_on_timeout=True,
    socket_keepalive=True,
    health_check_interval=30
)

# Circuit breaker pattern
@circuit_breaker(failure_threshold=5, recovery_timeout=30)
def redis_operation():
    ...
```

---

## Appendix A: Quick Reference Commands

### Redis CLI Commands

```bash
# Connect to Redis
redis-cli -h redis-master -p 6379 -a ${PASSWORD} --tls

# Check cluster info
redis-cli INFO replication
redis-cli INFO memory
redis-cli INFO stats

# Monitor commands in real-time
redis-cli MONITOR

# Check slow log
redis-cli SLOWLOG GET 10

# Memory analysis
redis-cli MEMORY DOCTOR
redis-cli MEMORY STATS

# Key statistics
redis-cli DBSIZE
redis-cli INFO keyspace
```

### Sentinel Commands

```bash
# Check Sentinel status
redis-cli -h sentinel-1 -p 26379 SENTINEL master greenlang-cache

# List replicas
redis-cli -h sentinel-1 -p 26379 SENTINEL slaves greenlang-cache

# Force failover
redis-cli -h sentinel-1 -p 26379 SENTINEL failover greenlang-cache
```

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **AOF** | Append Only File - persistence mechanism logging all write operations |
| **RDB** | Redis Database - point-in-time snapshot persistence |
| **Sentinel** | Redis component for monitoring and automatic failover |
| **Eviction** | Process of removing keys when memory limit is reached |
| **LRU** | Least Recently Used - eviction policy removing oldest accessed keys |
| **TTL** | Time To Live - automatic expiration time for keys |
| **Replication Lag** | Delay between master writes and replica receiving them |

---

## Appendix C: References

1. Redis Official Documentation: https://redis.io/docs/
2. Redis Sentinel Documentation: https://redis.io/docs/management/sentinel/
3. AWS ElastiCache Best Practices: https://docs.aws.amazon.com/AmazonElastiCache/latest/red-ug/best-practices.html
4. Redis Security Guidelines: https://redis.io/docs/management/security/

---

**Document Approval:**

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Infrastructure Lead | | | |
| Security Lead | | | |
| Engineering Manager | | | |
| CTO | | | |

---

*End of Document*
