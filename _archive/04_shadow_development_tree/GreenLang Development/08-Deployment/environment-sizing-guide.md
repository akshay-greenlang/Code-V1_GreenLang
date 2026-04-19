# Production Environment Sizing Guide

**Version:** 1.0.0
**Last Updated:** 2025-11-08

---

## Sizing Tiers Overview

| Tier | Organizations | Users | Daily Transactions | Monthly Cost |
|------|---------------|-------|-------------------|--------------|
| **Development** | 1 | 1-5 | 100 | $150 |
| **Small** | 1-10 | 10-100 | 10,000 | $626 |
| **Medium** | 10-100 | 100-1,000 | 100,000 | $4,629 |
| **Large** | 100-1,000 | 1,000-10,000 | 1,000,000 | $26,917 |
| **Enterprise** | 1,000+ | 10,000+ | 10,000,000+ | Custom |

---

## Development Environment

### Use Case
- Local development
- Testing
- Demos
- POCs

### Infrastructure Specifications

```yaml
Single Server:
  - Instance: m5.2xlarge (or equivalent)
  - vCPUs: 8
  - RAM: 32 GB
  - Storage: 200 GB SSD
  - Network: 1 Gbps

Deployment: Docker Compose
Cost: ~$150/month (single EC2 instance)

Services (all on one server):
  - GL-CBAM-APP: 1 instance
  - GL-CSRD-APP: 1 web + 1 worker
  - GL-VCCI-APP: 1 backend + 1 worker + 1 frontend
  - PostgreSQL: Single instance (no replication)
  - Redis: Single instance
  - Weaviate: Single instance (optional)
```

### Resource Allocation

```
Total: 8 vCPUs, 32 GB RAM

Application Allocation:
- CBAM: 0.5 vCPU, 1 GB RAM
- CSRD Web: 1 vCPU, 4 GB RAM
- CSRD Worker: 1 vCPU, 4 GB RAM
- VCCI Backend: 1 vCPU, 4 GB RAM
- VCCI Worker: 1 vCPU, 4 GB RAM
- VCCI Frontend: 0.5 vCPU, 1 GB RAM
- PostgreSQL: 1 vCPU, 4 GB RAM
- Redis: 0.5 vCPU, 2 GB RAM
- Weaviate: 0.5 vCPU, 4 GB RAM
- NGINX: 0.5 vCPU, 1 GB RAM
- Monitoring: 0.5 vCPU, 3 GB RAM
```

---

## Small Production Environment

### Use Case
- Small businesses
- Single department
- 1-10 organizations
- Up to 100 users

### Infrastructure Specifications

#### AWS Architecture

```yaml
Compute:
  - Application Servers:
      Type: 2x t3.large
      vCPUs: 2 per instance (4 total)
      RAM: 8 GB per instance (16 GB total)
      Purpose: Run all application containers
      Auto-scaling: No

Database:
  - RDS PostgreSQL:
      Instance: db.t3.medium (Multi-AZ)
      vCPUs: 2
      RAM: 4 GB
      Storage: 500 GB (gp3)
      IOPS: 3000
      Backups: 7 days retention

Cache:
  - ElastiCache Redis:
      Instance: cache.t3.medium
      vCPUs: 2
      RAM: 3.09 GB
      No cluster mode
      No replication

Storage:
  - S3 Buckets:
      Data: 100 GB
      Reports: 50 GB
      Backups: 300 GB
      Logs: 50 GB
      Total: 500 GB

Networking:
  - Application Load Balancer: 1
  - VPC: Single region
  - Subnets: 2 AZs
```

### Performance Characteristics

```yaml
Expected Performance:
  - API Response Time (p95): < 300ms
  - Database Query Time (p95): < 100ms
  - Cache Hit Rate: > 70%
  - Uptime: 99.5% (43 hours downtime/year)

Capacity:
  - CBAM Reports: 50/day
  - CSRD Reports: 10/month
  - VCCI Transactions: 10,000/day
  - Concurrent Users: 20
  - LLM Requests: 10,000/day
```

---

## Medium Production Environment

### Use Case
- Mid-size companies
- Multiple departments
- 10-100 organizations
- Up to 1,000 users

### Infrastructure Specifications

#### AWS Architecture

```yaml
Compute:
  - ECS Fargate (or EKS):
      Web Tier:
        - CBAM: 2 tasks (0.5 vCPU, 1 GB each)
        - CSRD: 4 tasks (1 vCPU, 4 GB each)
        - VCCI: 4 tasks (2 vCPU, 8 GB each)
      Worker Tier:
        - CSRD Workers: 4 tasks (2 vCPU, 8 GB each)
        - VCCI Workers: 8 tasks (2 vCPU, 8 GB each)
      Total: ~48 vCPUs, 192 GB RAM
      Auto-scaling: Yes (2x-4x)

Database:
  - RDS PostgreSQL:
      Primary: db.m5.xlarge (Multi-AZ)
        vCPUs: 4
        RAM: 16 GB
        Storage: 2 TB (gp3)
        IOPS: 12,000
      Read Replica 1: db.m5.large
        vCPUs: 2
        RAM: 8 GB
      Backups: 30 days retention
      Connection Pooling: PgBouncer

Cache:
  - ElastiCache Redis:
      Cluster Mode: Enabled
      Node Type: cache.m5.large
      Nodes: 3 shards × 2 replicas = 6 nodes
      Total Memory: ~37 GB
      vCPUs: 12
      RAM: 37 GB

Vector Database:
  - EC2 for Weaviate:
      Type: 2x m5.2xlarge
      vCPUs: 8 per instance (16 total)
      RAM: 32 GB per instance (64 GB total)
      Storage: 1 TB NVMe SSD per instance

Storage:
  - S3 Buckets:
      Data: 2 TB
      Reports: 500 GB
      Backups: 3 TB
      Logs: 500 GB
      Total: 6 TB
  - EBS:
      Application volumes: 500 GB
      Database volumes: 2 TB
      Weaviate volumes: 2 TB

Networking:
  - Application Load Balancers: 2
  - NAT Gateways: 2
  - VPC: Single region
  - Subnets: 3 AZs

Message Queue:
  - Amazon MQ (RabbitMQ):
      Instance: mq.m5.large
      vCPUs: 2
      RAM: 8 GB
      Multi-AZ: Yes
```

### Performance Characteristics

```yaml
Expected Performance:
  - API Response Time (p95): < 200ms
  - Database Query Time (p95): < 50ms
  - Cache Hit Rate: > 85%
  - Uptime: 99.9% (8.76 hours downtime/year)

Capacity:
  - CBAM Reports: 500/day
  - CSRD Reports: 100/month
  - VCCI Transactions: 100,000/day
  - Concurrent Users: 200
  - LLM Requests: 100,000/day

Throughput:
  - HTTP Requests: 10,000 requests/min
  - Database: 5,000 queries/sec
  - Message Queue: 1,000 messages/sec
```

---

## Large Production Environment

### Use Case
- Enterprise companies
- Global deployments
- 100-1,000 organizations
- Up to 10,000 users

### Infrastructure Specifications

#### AWS Architecture

```yaml
Compute:
  - EKS (Kubernetes):
      Node Groups:
        - Web Tier: 4-20 nodes (m5.2xlarge)
          vCPUs: 8 per node
          RAM: 32 GB per node
          Total: 32-160 vCPUs, 128-640 GB RAM
        - Worker Tier: 8-50 nodes (m5.4xlarge)
          vCPUs: 16 per node
          RAM: 64 GB per node
          Total: 128-800 vCPUs, 512-3200 GB RAM
      Pods:
        - CBAM: 4-20 replicas
        - CSRD Web: 8-40 replicas
        - CSRD Worker: 16-100 replicas
        - VCCI Backend: 8-40 replicas
        - VCCI Worker: 32-200 replicas
        - VCCI Frontend: 8-40 replicas
      Auto-scaling: Yes (HPA + Cluster Autoscaler)

Database:
  - RDS PostgreSQL:
      Primary: db.r5.4xlarge (Multi-AZ)
        vCPUs: 16
        RAM: 128 GB
        Storage: 10 TB (io2)
        IOPS: 50,000
      Read Replicas: 3x db.r5.2xlarge
        vCPUs: 8 per replica
        RAM: 64 GB per replica
      Global: Aurora Global Database (if multi-region)
      Backups: 90 days retention
      Connection Pooling: PgBouncer (pool size: 1000)

Cache:
  - ElastiCache Redis:
      Cluster Mode: Enabled
      Node Type: cache.r5.2xlarge
      Nodes: 6 shards × 2 replicas = 12 nodes
      Total vCPUs: 96
      Total Memory: ~380 GB

Vector Database:
  - EC2 for Weaviate:
      Type: 4x r5.4xlarge
      vCPUs: 16 per instance (64 total)
      RAM: 128 GB per instance (512 GB total)
      Storage: 4 TB NVMe SSD per instance
      Cluster: 4-node Weaviate cluster

Storage:
  - S3 Buckets:
      Data: 50 TB
      Reports: 5 TB
      Backups: 100 TB
      Logs: 5 TB
      Total: 160 TB
  - EBS:
      Application volumes: 2 TB
      Database volumes: 10 TB
      Weaviate volumes: 16 TB

Networking:
  - Application Load Balancers: 4
  - Network Load Balancers: 2
  - NAT Gateways: 6 (2 per AZ)
  - VPC: Multi-region
  - Subnets: 3 AZs per region
  - CloudFront: CDN for static assets
  - Direct Connect: 10 Gbps (optional)

Message Queue:
  - Amazon MQ (RabbitMQ):
      Cluster: 3x mq.m5.2xlarge
      vCPUs: 8 per instance (24 total)
      RAM: 32 GB per instance
      Multi-AZ: Yes
      High Throughput: 10,000 msg/sec
```

### Performance Characteristics

```yaml
Expected Performance:
  - API Response Time (p95): < 100ms
  - Database Query Time (p95): < 20ms
  - Cache Hit Rate: > 95%
  - Uptime: 99.99% (52.56 minutes downtime/year)

Capacity:
  - CBAM Reports: 5,000/day
  - CSRD Reports: 1,000/month
  - VCCI Transactions: 1,000,000/day
  - Concurrent Users: 2,000
  - LLM Requests: 1,000,000/day

Throughput:
  - HTTP Requests: 100,000 requests/min
  - Database: 50,000 queries/sec
  - Message Queue: 10,000 messages/sec
```

---

## Sizing Calculator

### Formula

```python
def calculate_infrastructure_size(
    num_orgs: int,
    num_users: int,
    transactions_per_day: int
) -> dict:
    """
    Calculate required infrastructure size
    """

    # Compute resources
    web_vcpus = max(4, num_users / 50)  # 1 vCPU per 50 users
    worker_vcpus = max(8, transactions_per_day / 10000)  # 1 vCPU per 10K txns
    total_vcpus = web_vcpus + worker_vcpus

    web_ram_gb = web_vcpus * 4  # 4 GB per vCPU
    worker_ram_gb = worker_vcpus * 8  # 8 GB per vCPU
    total_ram_gb = web_ram_gb + worker_ram_gb

    # Database
    db_vcpus = max(4, num_orgs / 10)  # 1 vCPU per 10 orgs
    db_ram_gb = db_vcpus * 4
    db_storage_gb = max(500, num_orgs * 50)  # 50 GB per org

    # Cache
    redis_ram_gb = max(8, num_users / 100)  # 100 MB per user

    # Storage
    s3_storage_tb = max(1, num_orgs * 0.5)  # 500 GB per org

    return {
        "compute": {
            "web_vcpus": web_vcpus,
            "worker_vcpus": worker_vcpus,
            "total_vcpus": total_vcpus,
            "web_ram_gb": web_ram_gb,
            "worker_ram_gb": worker_ram_gb,
            "total_ram_gb": total_ram_gb
        },
        "database": {
            "vcpus": db_vcpus,
            "ram_gb": db_ram_gb,
            "storage_gb": db_storage_gb
        },
        "cache": {
            "ram_gb": redis_ram_gb
        },
        "storage": {
            "s3_tb": s3_storage_tb
        }
    }

# Example usage
result = calculate_infrastructure_size(
    num_orgs=50,
    num_users=500,
    transactions_per_day=50000
)
print(result)
```

---

## Growth Planning

### Scaling Triggers

```yaml
Scale Up Compute When:
  - CPU usage > 70% for 10 minutes
  - Memory usage > 80% for 10 minutes
  - Request latency (p95) > 500ms for 15 minutes
  - Queue depth > 1000 messages for 5 minutes

Scale Up Database When:
  - CPU usage > 70% for 15 minutes
  - Connection count > 80% of max_connections
  - Query latency (p95) > 200ms
  - Read replica lag > 5 seconds

Scale Up Cache When:
  - Memory usage > 90%
  - Cache hit rate < 80%
  - Eviction rate > 100/sec

Scale Up Storage When:
  - Disk usage > 80%
  - IOPS usage > 80% of provisioned
```

### Growth Trajectory Example

```
Year 1:
  Q1: Small (10 orgs, 50 users) → $626/month
  Q2: Small (20 orgs, 100 users) → $800/month (20% increase)
  Q3: Medium (40 orgs, 200 users) → $2,000/month
  Q4: Medium (80 orgs, 500 users) → $4,000/month

Year 2:
  Q1: Medium (100 orgs, 800 users) → $5,000/month
  Q2: Large (200 orgs, 1,500 users) → $12,000/month
  Q3: Large (400 orgs, 3,000 users) → $18,000/month
  Q4: Large (800 orgs, 6,000 users) → $25,000/month

Year 3:
  Q1: Enterprise (1,000 orgs, 8,000 users) → $30,000/month
  Q2+: Custom pricing and architecture
```

---

## Regional Deployment Sizing

### Multi-Region Architecture (Large+)

```yaml
Primary Region (us-east-1):
  - Full infrastructure (read + write)
  - 70% of traffic

Secondary Region (eu-west-1):
  - Full infrastructure (read + write)
  - 25% of traffic
  - Disaster recovery standby

Tertiary Region (ap-southeast-1):
  - Read-only replicas
  - 5% of traffic
  - Latency optimization for APAC

Data Replication:
  - PostgreSQL: Aurora Global Database
  - Redis: Active-active replication
  - S3: Cross-region replication
  - Weaviate: Manual backup/restore
```

---

## Recommendations Summary

| Organizations | Users | Recommended Tier | Monthly Cost | Deployment |
|---------------|-------|------------------|--------------|------------|
| 1 | 1-5 | Development | $150 | Single server |
| 1-10 | 10-100 | Small | $626 | Multi-server |
| 10-100 | 100-1,000 | Medium | $4,629 | Kubernetes/ECS |
| 100-1,000 | 1,000-10,000 | Large | $26,917 | Multi-AZ Kubernetes |
| 1,000+ | 10,000+ | Enterprise | Custom | Multi-region Kubernetes |

---

**Document Owner:** Infrastructure Team
**Last Updated:** 2025-11-08
**Next Review:** Quarterly
