# CBAM IMPORTER COPILOT - SCALING GUIDE

**Version:** 1.0.0
**Last Updated:** 2025-11-18
**Owner:** GreenLang CBAM SRE Team

---

## PURPOSE

This guide provides scaling strategies for the CBAM Importer Copilot to handle varying workloads, from normal quarterly reporting (1,000-10,000 shipments) to high-volume year-end reporting (50,000+ shipments), while maintaining performance targets and EU reporting deadlines.

---

## PERFORMANCE BASELINES

### Current Performance (3 replicas, standard resources)

| Metric | Value | SLA Target |
|--------|-------|------------|
| **Throughput** | 16 shipments/sec | >15 shipments/sec |
| **Latency (P95)** | 200ms | <500ms |
| **End-to-End Processing** | 10 minutes for 10K shipments | <15 minutes |
| **Agent 1: Intake** | 1,200 shipments/sec | >1,000 shipments/sec |
| **Agent 2: Calculation** | <3ms per shipment | <5ms |
| **Agent 3: Packaging** | <1 sec for 10K | <2 sec |
| **Memory Usage** | 1.5GB per pod | <2GB |
| **CPU Usage** | 60% average | <80% |

---

## CAPACITY PLANNING

### Shipment Volume Guidelines

| Quarterly Volume | Recommended Replicas | Resources | Expected Processing Time |
|------------------|----------------------|-----------|--------------------------|
| **<1,000** | 3 | Standard | <5 minutes |
| **1,000-5,000** | 3 | Standard | 5-10 minutes |
| **5,000-10,000** | 3-5 | Standard | 10-15 minutes |
| **10,000-20,000** | 5-8 | Increased | 15-30 minutes |
| **20,000-50,000** | 8-12 | Increased | 30-60 minutes |
| **50,000+** | 12-20 | High | 1-2 hours |

**Standard Resources:**
- CPU: 1000m (1 core) request, 2000m limit
- Memory: 1Gi request, 2Gi limit

**Increased Resources:**
- CPU: 2000m (2 cores) request, 4000m limit
- Memory: 2Gi request, 4Gi limit

**High Resources:**
- CPU: 4000m (4 cores) request, 8000m limit
- Memory: 4Gi request, 8Gi limit

---

## HORIZONTAL SCALING

### Manual Horizontal Scaling

Scale based on shipment volume and processing deadlines.

#### Scale Up

```bash
# For moderate volume increase (10K-20K shipments)
kubectl scale deployment/cbam-importer --replicas=5 -n greenlang

# For high volume (20K-50K shipments)
kubectl scale deployment/cbam-importer --replicas=10 -n greenlang

# For extreme volume (50K+ shipments)
kubectl scale deployment/cbam-importer --replicas=15 -n greenlang

# Monitor scaling
kubectl get pods -n greenlang -l app=cbam-importer -w
```

#### Scale Down

```bash
# After peak period, return to normal
kubectl scale deployment/cbam-importer --replicas=3 -n greenlang

# Gradual scale-down
kubectl scale deployment/cbam-importer --replicas=5 -n greenlang
# Wait 10 minutes, monitor
kubectl scale deployment/cbam-importer --replicas=3 -n greenlang
```

### Automatic Horizontal Scaling (HPA)

**Setup Horizontal Pod Autoscaler:**

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: cbam-importer-hpa
  namespace: greenlang
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: cbam-importer
  minReplicas: 3
  maxReplicas: 15
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: cbam_pipeline_active_runs
      target:
        type: AverageValue
        averageValue: "2"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
      - type: Pods
        value: 2
        periodSeconds: 60
      selectPolicy: Max
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 25
        periodSeconds: 60
      - type: Pods
        value: 1
        periodSeconds: 60
      selectPolicy: Min
```

**Deploy HPA:**

```bash
kubectl apply -f k8s/hpa.yaml

# Monitor HPA
kubectl get hpa cbam-importer-hpa -n greenlang -w

# Check current metrics
kubectl describe hpa cbam-importer-hpa -n greenlang
```

**HPA Behavior:**
- **Scale Up:** 50% increase or 2 pods, whichever is greater, every 60 seconds
- **Scale Down:** 25% decrease or 1 pod, whichever is less, every 60 seconds (after 5 min stabilization)
- **CPU Trigger:** >70% utilization
- **Memory Trigger:** >80% utilization
- **Custom Metric:** >2 active pipeline runs per pod

---

## VERTICAL SCALING

### When to Use Vertical Scaling

- Individual pods hitting memory limits (OOMKilled)
- High CPU usage (>80%) even with low replica count
- Large shipment batches per request (>5,000 shipments at once)
- Complex emissions calculations (many product types)

### Increase Pod Resources

```bash
# Method 1: Edit deployment directly
kubectl edit deployment cbam-importer -n greenlang

# Update resources section:
# resources:
#   requests:
#     cpu: 2000m
#     memory: 2Gi
#   limits:
#     cpu: 4000m
#     memory: 4Gi

# Method 2: Patch deployment
kubectl patch deployment cbam-importer -n greenlang --patch '
spec:
  template:
    spec:
      containers:
      - name: cbam-importer
        resources:
          requests:
            cpu: "2000m"
            memory: "2Gi"
          limits:
            cpu: "4000m"
            memory: "4Gi"
'

# Monitor rollout
kubectl rollout status deployment/cbam-importer -n greenlang

# Verify new resources
kubectl get pods -n greenlang -l app=cbam-importer -o jsonpath='{.items[0].spec.containers[0].resources}'
```

### Resource Tiers

**Tier 1: Standard (Default)**
```yaml
resources:
  requests:
    cpu: 1000m
    memory: 1Gi
  limits:
    cpu: 2000m
    memory: 2Gi
```
**Use Case:** <10,000 shipments/quarter

**Tier 2: Increased**
```yaml
resources:
  requests:
    cpu: 2000m
    memory: 2Gi
  limits:
    cpu: 4000m
    memory: 4Gi
```
**Use Case:** 10,000-50,000 shipments/quarter

**Tier 3: High**
```yaml
resources:
  requests:
    cpu: 4000m
    memory: 4Gi
  limits:
    cpu: 8000m
    memory: 8Gi
```
**Use Case:** >50,000 shipments/quarter or year-end peak

---

## DATABASE SCALING

### PostgreSQL Scaling

#### Connection Pool Tuning

```bash
# Increase connection pool for high concurrency
kubectl set env deployment/cbam-importer \
  DB_POOL_SIZE=20 \
  DB_MAX_OVERFLOW=40 \
  DB_POOL_TIMEOUT=30 \
  -n greenlang

# Monitor connection usage
kubectl exec -n greenlang deployment/postgres -- \
  psql -c "SELECT count(*) as connections, state FROM pg_stat_activity GROUP BY state;"
```

**Recommended Settings:**

| Replicas | DB_POOL_SIZE | DB_MAX_OVERFLOW | Total Connections |
|----------|--------------|-----------------|-------------------|
| 3 | 10 | 20 | 90 |
| 5 | 12 | 24 | 180 |
| 10 | 15 | 30 | 450 |
| 15 | 20 | 40 | 900 |

**PostgreSQL max_connections** should be set to at least Total Connections + 100 (for admin)

#### Database Vertical Scaling

```bash
# Increase PostgreSQL resources
kubectl edit statefulset postgres -n greenlang

# Update to:
# resources:
#   requests:
#     cpu: 4000m
#     memory: 8Gi
#   limits:
#     cpu: 8000m
#     memory: 16Gi
```

#### Database Optimization

```bash
# Vacuum and analyze
kubectl exec -n greenlang deployment/postgres -- \
  psql greenlang -c "VACUUM ANALYZE shipments;"

# Create indexes for common queries
kubectl exec -n greenlang deployment/postgres -- \
  psql greenlang -c "
CREATE INDEX IF NOT EXISTS idx_shipments_import_date ON shipments(import_date);
CREATE INDEX IF NOT EXISTS idx_shipments_cn_code ON shipments(cn_code);
CREATE INDEX IF NOT EXISTS idx_shipments_supplier_id ON shipments(supplier_id);
"

# Update statistics
kubectl exec -n greenlang deployment/postgres -- \
  psql greenlang -c "ANALYZE;"
```

#### Read Replicas (Future)

For very high query volume:

```yaml
# postgres-replica.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres-replica
spec:
  replicas: 2
  # Configure as read replicas
  # Route read queries to replicas
  # Keep writes on primary
```

---

## APPLICATION-LEVEL OPTIMIZATION

### Batch Processing

For large volumes, use batch processing mode:

```bash
# Process in batches of 5,000 shipments
python cbam_pipeline.py \
  --input large_shipments.csv \
  --batch-size 5000 \
  --output-dir /batched_output/ \
  --merge-final \
  --parallel-batches 3 \
  --importer-name "Large Importer" \
  --importer-country DE \
  --importer-eori DE123456789012 \
  --declarant-name "John Doe" \
  --declarant-position "Compliance Officer"
```

**Batch Processing Benefits:**
- Lower memory usage (process chunks instead of all at once)
- Better parallelization (multiple batches concurrently)
- Failure isolation (one batch fails, others continue)
- Progress tracking (know which batches completed)

### Performance Tuning Environment Variables

```bash
kubectl set env deployment/cbam-importer \
  CBAM_PARALLEL_WORKERS=8 \
  CBAM_SKIP_INTERMEDIATE_OUTPUTS=true \
  CBAM_SKIP_SUMMARY_MARKDOWN=true \
  CBAM_CACHE_EMISSION_FACTORS=true \
  CBAM_CACHE_CN_CODES=true \
  CBAM_BATCH_INSERT_SIZE=1000 \
  -n greenlang
```

**Performance Variables:**

| Variable | Default | High Performance | Description |
|----------|---------|------------------|-------------|
| CBAM_PARALLEL_WORKERS | 4 | 8-16 | Parallel processing workers |
| CBAM_SKIP_INTERMEDIATE_OUTPUTS | false | true | Skip intermediate file writes |
| CBAM_SKIP_SUMMARY_MARKDOWN | false | true | Skip markdown generation |
| CBAM_CACHE_EMISSION_FACTORS | true | true | Cache emission factors in memory |
| CBAM_CACHE_CN_CODES | true | true | Cache CN codes in memory |
| CBAM_BATCH_INSERT_SIZE | 500 | 1000 | Database batch insert size |

### Caching Strategy

```python
# config.py - Caching configuration
REDIS_ENABLED = True
REDIS_HOST = "redis"
REDIS_PORT = 6379
CACHE_TTL_SECONDS = 3600

# Cache emission factors (rarely change)
# Cache CN code metadata (rarely change)
# Cache supplier profiles (updated weekly)
# Don't cache shipment data (unique per quarter)
```

---

## MONITORING DURING SCALING

### Key Metrics to Watch

```bash
# 1. Pod count and health
kubectl get pods -n greenlang -l app=cbam-importer

# 2. Resource usage
kubectl top pods -n greenlang -l app=cbam-importer

# 3. Application metrics
curl http://cbam-importer:8001/metrics | grep -E "cbam_pipeline_duration|cbam_throughput|cbam_errors"

# 4. Database connections
kubectl exec -n greenlang deployment/postgres -- \
  psql -c "SELECT count(*) FROM pg_stat_activity WHERE state = 'active';"

# 5. Queue depth (if using queues)
curl http://cbam-importer:8001/metrics | grep cbam_queue_depth
```

### Grafana Dashboards

- **CBAM Performance Dashboard:** http://grafana:3000/d/cbam-perf
  - Throughput (shipments/sec)
  - Latency percentiles (P50, P95, P99)
  - Active pipeline runs
  - Pod resource usage

- **CBAM Scaling Dashboard:** http://grafana:3000/d/cbam-scaling
  - Replica count over time
  - HPA metrics and decisions
  - Resource saturation
  - Scaling events

### Alerts

Critical alerts during scaling:

```yaml
# alerts.yml
- alert: CBAMHighThroughputNeeded
  expr: cbam_pending_shipments > 10000 AND cbam_throughput_per_sec < 20
  for: 5m
  annotations:
    summary: "High shipment volume requires scaling"
    description: "{{ $value }} shipments pending, consider scaling up"

- alert: CBAMPodCPUSaturation
  expr: avg(rate(container_cpu_usage_seconds_total{pod=~"cbam-importer.*"}[5m])) > 0.85
  for: 10m
  annotations:
    summary: "CBAM pods CPU saturated"
    description: "Average CPU usage {{ $value }}, scale up or optimize"

- alert: CBAMDatabaseConnectionExhaustion
  expr: pg_stat_activity_count / pg_settings_max_connections > 0.8
  for: 5m
  annotations:
    summary: "PostgreSQL connection pool near limit"
    description: "{{ $value }}% of max connections used"
```

---

## SCALING PLAYBOOKS

### Playbook 1: Year-End High Volume (50,000 shipments)

**Situation:** Annual year-end reporting with 5x normal volume

**Timeline:** 1 week before deadline

**Actions:**

```bash
# Day -7: Pre-scale
# 1. Scale to 8 replicas
kubectl scale deployment/cbam-importer --replicas=8 -n greenlang

# 2. Increase resources
kubectl patch deployment cbam-importer -n greenlang --patch '
spec:
  template:
    spec:
      containers:
      - name: cbam-importer
        resources:
          requests: {cpu: "2000m", memory: "2Gi"}
          limits: {cpu: "4000m", memory: "4Gi"}
'

# 3. Increase database connections
kubectl set env deployment/cbam-importer DB_POOL_SIZE=20 DB_MAX_OVERFLOW=40 -n greenlang

# 4. Enable performance mode
kubectl set env deployment/cbam-importer \
  CBAM_PARALLEL_WORKERS=12 \
  CBAM_SKIP_INTERMEDIATE_OUTPUTS=true \
  -n greenlang

# Day -3: Monitor and adjust
# Check performance metrics daily
# Scale to 12 replicas if needed

# Day 0: Process
# Use batch processing
python cbam_pipeline.py --input year_end_50k.csv --batch-size 5000 --parallel-batches 4 [...]

# Day +1: Scale down
kubectl scale deployment/cbam-importer --replicas=3 -n greenlang
# Revert to standard resources
```

### Playbook 2: Emergency Same-Day Processing

**Situation:** 10,000 shipments need processing in 2 hours (deadline emergency)

**Actions:**

```bash
# Immediate: Max scale
kubectl scale deployment/cbam-importer --replicas=15 -n greenlang

# Increase resources to High tier
kubectl patch deployment cbam-importer -n greenlang --patch '
spec:
  template:
    spec:
      containers:
      - name: cbam-importer
        resources:
          requests: {cpu: "4000m", memory: "4Gi"}
          limits: {cpu: "8000m", memory: "8Gi"}
'

# Max performance mode
kubectl set env deployment/cbam-importer \
  CBAM_PARALLEL_WORKERS=16 \
  CBAM_SKIP_INTERMEDIATE_OUTPUTS=true \
  CBAM_SKIP_SUMMARY_MARKDOWN=true \
  CBAM_BATCH_INSERT_SIZE=2000 \
  -n greenlang

# Process with aggressive batching
python cbam_pipeline.py --input urgent_10k.csv --batch-size 2000 --parallel-batches 5 [...]

# Monitor intensively (every 5 minutes)
watch -n 300 'curl -s http://cbam-importer:8001/metrics | grep cbam_pipeline_progress'
```

---

## COST OPTIMIZATION

### Right-Sizing After Peak

After high-volume periods, reduce resources to save costs:

```bash
# 1. Analyze actual usage during peak
kubectl top pods -n greenlang -l app=cbam-importer --containers

# 2. Scale down to normal
kubectl scale deployment/cbam-importer --replicas=3 -n greenlang

# 3. Reduce resources to standard
kubectl patch deployment cbam-importer -n greenlang --patch '
spec:
  template:
    spec:
      containers:
      - name: cbam-importer
        resources:
          requests: {cpu: "1000m", memory: "1Gi"}
          limits: {cpu: "2000m", memory: "2Gi"}
'

# 4. Disable HPA during low volume
kubectl delete hpa cbam-importer-hpa -n greenlang
```

### Resource Requests Optimization

**Requests** = Guaranteed resources (used for scheduling)
**Limits** = Maximum allowed (prevents resource starvation)

**Best Practices:**
- Set requests to 80% of normal usage
- Set limits to 2x requests (allows bursting)
- Monitor actual usage and adjust quarterly

---

## TROUBLESHOOTING SCALING ISSUES

### Pods Not Starting After Scale Up

```bash
# Check node resources
kubectl describe nodes | grep -A 5 "Allocated resources"

# Check events
kubectl get events -n greenlang --sort-by='.lastTimestamp' | grep cbam-importer

# Common issue: Insufficient node resources
# Solution: Add more nodes or reduce resource requests
```

### Performance Not Improving After Scaling

```bash
# Check if bottleneck is elsewhere
# 1. Database connections maxed?
kubectl exec -n greenlang deployment/postgres -- \
  psql -c "SELECT count(*) FROM pg_stat_activity;"

# 2. Single-threaded bottleneck in code?
# Profile with py-spy
kubectl exec -n greenlang deployment/cbam-importer -- \
  py-spy top --pid 1 --duration 30

# 3. I/O bottleneck?
kubectl exec -n greenlang deployment/cbam-importer -- \
  iostat -x 5 3
```

---

## RELATED RUNBOOKS

- MAINTENANCE.md - Resource planning and capacity reviews
- INCIDENT_RESPONSE.md - Performance incident handling
- TROUBLESHOOTING.md - Performance degradation diagnosis

---

**Document Control:**
- **Version:** 1.0.0
- **Last Updated:** 2025-11-18
- **Next Review:** 2025-12-18 (quarterly)
- **Owner:** CBAM SRE Team

---

*This scaling guide should be reviewed and updated quarterly based on actual usage patterns and performance data.*
