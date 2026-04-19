# GL-001 ThermalCommand - High Availability & Disaster Recovery Architecture

## Document Information

| Field | Value |
|-------|-------|
| Agent ID | GL-001 |
| Agent Name | ThermalCommand |
| Version | 1.0.0 |
| Last Updated | 2025-12-22 |
| Classification | Internal - Operations |
| Owner | GreenLang Process Heat Team |

---

## 1. Executive Summary

This document defines the High Availability (HA) and Disaster Recovery (DR) architecture for GL-001 ThermalCommand, the master orchestrator for all process heat operations. Given the critical nature of this agent ($20B value at stake), the architecture is designed to achieve:

- **Availability Target**: 99.99% (52.6 minutes downtime/year)
- **Recovery Point Objective (RPO)**: 1 minute
- **Recovery Time Objective (RTO)**: 5 minutes

---

## 2. High Availability Architecture

### 2.1 Architecture Diagram

```
                                    +-----------------------+
                                    |    Global Load        |
                                    |    Balancer (GLB)     |
                                    |    (Cloudflare/AWS)   |
                                    +-----------+-----------+
                                                |
                    +---------------------------+---------------------------+
                    |                           |                           |
        +-----------v-----------+   +-----------v-----------+   +-----------v-----------+
        |   Region: US-EAST-1   |   |   Region: US-WEST-2   |   |   Region: EU-WEST-1   |
        |   (Primary)           |   |   (Secondary)         |   |   (DR/Tertiary)       |
        +-----------+-----------+   +-----------+-----------+   +-----------+-----------+
                    |                           |                           |
        +-----------v-----------+   +-----------v-----------+   +-----------v-----------+
        |    Kubernetes        |   |    Kubernetes        |   |    Kubernetes        |
        |    Cluster (EKS)     |   |    Cluster (EKS)     |   |    Cluster (EKS)     |
        +----------------------+   +----------------------+   +----------------------+
        |                      |   |                      |   |                      |
        |  +----------------+  |   |  +----------------+  |   |  +----------------+  |
        |  | ThermalCommand |  |   |  | ThermalCommand |  |   |  | ThermalCommand |  |
        |  | Pod x3         |  |   |  | Pod x3         |  |   |  | Pod x2         |  |
        |  | (Active)       |  |   |  | (Hot Standby)  |  |   |  | (Warm Standby) |  |
        |  +-------+--------+  |   |  +-------+--------+  |   |  +-------+--------+  |
        |          |           |   |          |           |   |          |           |
        |  +-------v--------+  |   |  +-------v--------+  |   |  +-------v--------+  |
        |  | Redis Sentinel |  |   |  | Redis Sentinel |  |   |  | Redis Sentinel |  |
        |  | (3 nodes)      |  |   |  | (3 nodes)      |  |   |  | (3 nodes)      |  |
        |  +-------+--------+  |   |  +-------+--------+  |   |  +-------+--------+  |
        |          |           |   |          |           |   |          |           |
        |  +-------v--------+  |   |  +-------v--------+  |   |  +-------v--------+  |
        |  | PostgreSQL     |  |   |  | PostgreSQL     |  |   |  | PostgreSQL     |  |
        |  | (Primary + 2   |  |   |  | (Streaming     |  |   |  | (Streaming     |  |
        |  |  Replicas)     |  |   |  |  Replica)      |  |   |  |  Replica)      |  |
        |  +-------+--------+  |   |  +-------+--------+  |   |  +-------+--------+  |
        |          |           |   |          |           |   |          |           |
        |  +-------v--------+  |   |  +-------v--------+  |   |  +-------v--------+  |
        |  | Kafka Cluster  |  |   |  | Kafka Cluster  |  |   |  | Kafka Cluster  |  |
        |  | (3 brokers)    |  |   |  | (MirrorMaker2) |  |   |  | (MirrorMaker2) |  |
        |  +----------------+  |   |  +----------------+  |   |  +----------------+  |
        |                      |   |                      |   |                      |
        +----------------------+   +----------------------+   +----------------------+
                    |                           |                           |
                    +---------------------------+---------------------------+
                                                |
                                    +-----------v-----------+
                                    |   Cross-Region        |
                                    |   Replication         |
                                    |   (S3 + DynamoDB)     |
                                    +-----------------------+
```

### 2.2 Component Redundancy

| Component | Primary Region | Secondary Region | DR Region | Total Capacity |
|-----------|----------------|------------------|-----------|----------------|
| Application Pods | 3 | 3 | 2 | 8 pods |
| Redis Sentinel Nodes | 3 | 3 | 3 | 9 nodes |
| PostgreSQL Instances | 3 (1 primary + 2 replicas) | 1 replica | 1 replica | 5 instances |
| Kafka Brokers | 3 | 3 | 3 | 9 brokers |
| Load Balancers | 2 (Active/Passive) | 2 | 2 | 6 LBs |

### 2.3 Pod Anti-Affinity Strategy

```yaml
# Ensures pods are distributed across failure domains
affinity:
  podAntiAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchLabels:
            app: gl-001-thermalcommand
        topologyKey: topology.kubernetes.io/zone
    preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        podAffinityTerm:
          labelSelector:
            matchLabels:
              app: gl-001-thermalcommand
          topologyKey: kubernetes.io/hostname
```

---

## 3. Failure Modes and Mitigation

### 3.1 Application Layer Failures

| Failure Mode | Detection | Mitigation | Recovery Time |
|--------------|-----------|------------|---------------|
| Pod Crash | Kubernetes liveness probe | Auto-restart by kubelet | < 30 seconds |
| Node Failure | Node heartbeat timeout | Pod rescheduling to healthy node | < 2 minutes |
| AZ Failure | Health check failures | Traffic shift to other AZs | < 1 minute |
| Memory Leak | OOM killer, metrics | Restart + HPA scaling | < 1 minute |
| Deadlock | Liveness probe timeout | Pod restart | < 30 seconds |

### 3.2 Database Layer Failures

| Failure Mode | Detection | Mitigation | Recovery Time |
|--------------|-----------|------------|---------------|
| PostgreSQL Primary Failure | Patroni health check | Automatic failover to replica | < 30 seconds |
| PostgreSQL Replica Failure | Streaming lag monitor | Remove from pool, rebuild | < 5 minutes |
| Connection Pool Exhaustion | Metrics alerting | Auto-scaling, connection recycling | < 1 minute |
| Corruption | Checksum validation | Restore from backup | < 15 minutes |

### 3.3 Cache Layer Failures

| Failure Mode | Detection | Mitigation | Recovery Time |
|--------------|-----------|------------|---------------|
| Redis Master Failure | Sentinel detection | Automatic failover | < 10 seconds |
| Redis Replica Failure | Sentinel monitoring | Rebuild replica | < 5 minutes |
| Cache Miss Storm | Hit ratio monitoring | Circuit breaker + gradual refill | < 2 minutes |
| Memory Pressure | Memory metrics | Eviction policy + scaling | < 1 minute |

### 3.4 Messaging Layer Failures

| Failure Mode | Detection | Mitigation | Recovery Time |
|--------------|-----------|------------|---------------|
| Kafka Broker Failure | ISR monitoring | Partition rebalancing | < 2 minutes |
| Consumer Lag | Lag metrics | Auto-scaling consumers | < 5 minutes |
| Topic Partition Loss | Under-replicated partitions | Replica recovery | < 10 minutes |
| Network Partition | Split-brain detection | Fencing, leader election | < 1 minute |

### 3.5 Infrastructure Failures

| Failure Mode | Detection | Mitigation | Recovery Time |
|--------------|-----------|------------|---------------|
| Network Failure | Connectivity probes | Failover to secondary region | < 3 minutes |
| DNS Failure | Health checks | Fallback DNS (Route53 health) | < 1 minute |
| Certificate Expiry | Cert-manager monitoring | Auto-renewal (Let's Encrypt) | Proactive |
| Storage Failure | EBS health checks | Volume replacement, restore | < 10 minutes |

---

## 4. RTO/RPO Targets by Component

### 4.1 Recovery Objectives Matrix

| Component | RPO | RTO | Backup Frequency | Retention |
|-----------|-----|-----|------------------|-----------|
| **Application State** | 0 (stateless) | 30 seconds | N/A | N/A |
| **PostgreSQL Data** | 1 minute | 5 minutes | Continuous (WAL) | 30 days |
| **Redis Cache** | 5 minutes | 1 minute | RDB every 5 min | 7 days |
| **Kafka Messages** | 1 minute | 5 minutes | Topic replication | 7 days |
| **Audit Logs** | 0 (synchronous) | 15 minutes | Continuous | 7 years |
| **Configuration** | 0 (GitOps) | 2 minutes | Version controlled | Unlimited |
| **Secrets** | 0 (Vault sync) | 5 minutes | Replicated | 90 days |

### 4.2 Data Classification

| Data Type | Criticality | Encryption | Backup Priority |
|-----------|-------------|------------|-----------------|
| Heat Optimization Plans | Critical | AES-256 | P0 |
| Safety Interlock State | Critical | AES-256 | P0 |
| Telemetry Data | High | AES-256 | P1 |
| Audit Trail | Regulatory | AES-256 + Immutable | P0 |
| ML Model Artifacts | High | AES-256 | P1 |
| Session Data | Medium | AES-256 | P2 |

---

## 5. Failover Procedures

### 5.1 Automatic Failover (No Human Intervention)

#### 5.1.1 Pod-Level Failover
```bash
# Kubernetes handles automatically via:
# - livenessProbe: Restart unhealthy pods
# - readinessProbe: Remove from service endpoints
# - PodDisruptionBudget: Maintain minimum availability

# PDB Configuration
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: gl-001-thermalcommand-pdb
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: gl-001-thermalcommand
```

#### 5.1.2 Database Failover (Patroni)
```bash
# Patroni handles PostgreSQL failover automatically
# Leader election via DCS (etcd/Consul)
# Streaming replication with synchronous_commit

# Health endpoint: /patroni
# Leader check: /leader
# Replica check: /replica
```

#### 5.1.3 Redis Sentinel Failover
```bash
# Sentinel configuration
sentinel monitor thermalcommand-redis 10.0.1.10 6379 2
sentinel down-after-milliseconds thermalcommand-redis 5000
sentinel failover-timeout thermalcommand-redis 60000
sentinel parallel-syncs thermalcommand-redis 1
```

### 5.2 Manual Failover Procedures

#### 5.2.1 Region Failover (Primary to Secondary)

```bash
#!/bin/bash
# Manual region failover procedure
# Execute only when primary region is completely unavailable

# Step 1: Verify primary region is down
./verify-region-health.sh us-east-1
if [ $? -eq 0 ]; then
    echo "Primary region is healthy. Aborting failover."
    exit 1
fi

# Step 2: Promote secondary region
kubectl config use-context eks-us-west-2
kubectl scale deployment gl-001-thermalcommand --replicas=5 -n greenlang

# Step 3: Promote PostgreSQL replica to primary
kubectl exec -it postgresql-0 -n greenlang -- patronictl failover

# Step 4: Update DNS
aws route53 change-resource-record-sets \
    --hosted-zone-id Z123456789 \
    --change-batch file://failover-dns.json

# Step 5: Verify services
./verify-service-health.sh us-west-2

# Step 6: Notify stakeholders
./send-notification.sh "GL-001 Failover Complete" "Region: US-WEST-2"
```

#### 5.2.2 Database Failover (Manual)

```bash
#!/bin/bash
# Manual PostgreSQL failover when automatic failover fails

# Step 1: Verify replica health
kubectl exec -it postgresql-replica-0 -n greenlang -- \
    pg_isready -h localhost -p 5432

# Step 2: Stop writes on primary (if accessible)
kubectl exec -it postgresql-0 -n greenlang -- \
    psql -c "SELECT pg_switch_wal();"

# Step 3: Promote replica
kubectl exec -it postgresql-replica-0 -n greenlang -- \
    pg_ctl promote -D /var/lib/postgresql/data

# Step 4: Update connection strings
kubectl patch secret gl-001-db-credentials \
    -p '{"stringData":{"host":"postgresql-replica-0.greenlang.svc"}}'

# Step 5: Restart application pods
kubectl rollout restart deployment/gl-001-thermalcommand -n greenlang
```

---

## 6. Health Checks and Monitoring

### 6.1 Health Check Endpoints

| Endpoint | Purpose | Interval | Timeout | Threshold |
|----------|---------|----------|---------|-----------|
| `/api/v1/health` | Liveness | 10s | 5s | 3 failures |
| `/api/v1/ready` | Readiness | 5s | 3s | 1 failure |
| `/api/v1/health/deep` | Dependencies | 30s | 10s | 2 failures |
| `/metrics` | Prometheus | 15s | 5s | N/A |

### 6.2 Monitoring Stack

```yaml
# Prometheus scrape configuration
scrape_configs:
  - job_name: 'gl-001-thermalcommand'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        regex: gl-001-thermalcommand
        action: keep
    metrics_path: /metrics
    scrape_interval: 15s
```

### 6.3 Key Metrics for HA

| Metric | Alert Threshold | Severity |
|--------|-----------------|----------|
| `thermalcommand_request_duration_seconds{quantile="0.99"}` | > 1s | Warning |
| `thermalcommand_request_duration_seconds{quantile="0.99"}` | > 5s | Critical |
| `thermalcommand_error_rate` | > 1% | Warning |
| `thermalcommand_error_rate` | > 5% | Critical |
| `thermalcommand_pod_available` | < 2 | Critical |
| `pg_replication_lag_seconds` | > 30s | Warning |
| `pg_replication_lag_seconds` | > 60s | Critical |
| `redis_master_link_status` | down | Critical |
| `kafka_consumer_lag` | > 10000 | Warning |

---

## 7. Testing and Validation

### 7.1 Chaos Engineering Tests

| Test | Frequency | Duration | Success Criteria |
|------|-----------|----------|------------------|
| Pod Kill | Weekly | 5 minutes | Service degradation < 5% |
| Node Drain | Monthly | 15 minutes | Zero downtime |
| AZ Failure | Quarterly | 30 minutes | Failover < 3 minutes |
| Network Partition | Monthly | 10 minutes | Proper split-brain handling |
| Database Failover | Weekly | 5 minutes | RPO/RTO within targets |

### 7.2 Runbook Testing

| Runbook | Test Frequency | Last Tested | Next Test |
|---------|---------------|-------------|-----------|
| Region Failover | Quarterly | 2025-09-15 | 2025-12-15 |
| Database Recovery | Monthly | 2025-11-22 | 2025-12-22 |
| Cache Rebuild | Monthly | 2025-11-22 | 2025-12-22 |
| Full DR Test | Semi-annually | 2025-06-01 | 2025-12-01 |

---

## 8. Appendices

### 8.1 Contact Information

| Role | Name | Contact | Escalation |
|------|------|---------|------------|
| Primary On-Call | SRE Team | pager-sre@greenlang.ai | 15 min |
| Secondary On-Call | Platform Team | pager-platform@greenlang.ai | 30 min |
| Incident Commander | Director of Ops | ops-director@greenlang.ai | 1 hour |
| Executive Escalation | VP Engineering | vp-eng@greenlang.ai | 2 hours |

### 8.2 Related Documents

- [DISASTER-RECOVERY.md](./DISASTER-RECOVERY.md) - Full DR procedures
- [SLA.md](./SLA.md) - Service Level Agreement
- [../runbooks/](../runbooks/) - Operational runbooks
- [../deployment/ha/](../../deployment/ha/) - HA deployment manifests

### 8.3 Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-22 | GL DevOps | Initial release |
