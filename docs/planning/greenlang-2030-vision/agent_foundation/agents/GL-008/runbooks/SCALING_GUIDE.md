# GL-008 SteamTrapInspector - Scaling Guide

**Version:** 1.0.0
**Last Updated:** 2025-11-26
**Owner:** Platform Operations Team

---

## Table of Contents

1. [Scaling Overview](#scaling-overview)
2. [Horizontal Scaling](#horizontal-scaling)
3. [Vertical Scaling](#vertical-scaling)
4. [Auto-Scaling Configuration](#auto-scaling-configuration)
5. [Capacity Planning](#capacity-planning)
6. [Performance Benchmarks](#performance-benchmarks)
7. [Multi-Site Deployment](#multi-site-deployment)
8. [Database Scaling](#database-scaling)
9. [Monitoring & Metrics](#monitoring--metrics)
10. [Cost Optimization](#cost-optimization)

---

## Scaling Overview

### Current Capacity Baseline

**Production Environment (as of 2025-11-26):**

| Component | Replicas | CPU (per pod) | Memory (per pod) | Max Throughput |
|-----------|----------|---------------|------------------|----------------|
| steam-trap-inspector | 4 | 1000m | 2Gi | 400 inspections/min |
| api-gateway | 3 | 500m | 1Gi | 1000 req/sec |
| worker | 5 | 2000m | 4Gi | 250 jobs/min |
| sensor-gateway | 3 | 500m | 1Gi | 5000 sensors |
| ml-service | 2 | 4000m | 8Gi | 100 predictions/sec |

**Resource Totals:**
- Total CPU: 26 cores
- Total Memory: 52 GB
- Total Pods: 17

### Scaling Triggers

**Scale Up When:**
- CPU utilization >70% for >5 minutes
- Memory utilization >80% for >5 minutes
- Request queue depth >100
- API latency P95 >2 seconds
- Error rate >5%
- Inspection backlog >500 jobs

**Scale Down When:**
- CPU utilization <30% for >15 minutes
- Memory utilization <40% for >15 minutes
- Request queue depth <10
- API latency P95 <500ms
- Off-peak hours (configured per site)

### Scaling Limits

**Maximum Scale:**
- steam-trap-inspector: 20 replicas
- api-gateway: 12 replicas
- worker: 30 replicas
- sensor-gateway: 10 replicas
- ml-service: 8 replicas

**Minimum Scale:**
- steam-trap-inspector: 2 replicas (HA requirement)
- api-gateway: 2 replicas (HA requirement)
- worker: 3 replicas
- sensor-gateway: 2 replicas (HA requirement)
- ml-service: 1 replica

---

## Horizontal Scaling

### Manual Horizontal Scaling

```bash
# Scale steam-trap-inspector to 8 replicas
kubectl scale deployment/steam-trap-inspector -n greenlang-gl008 --replicas=8

# Verify scaling
kubectl get pods -n greenlang-gl008 -l app=steam-trap-inspector
kubectl rollout status deployment/steam-trap-inspector -n greenlang-gl008

# Check pod distribution across nodes
kubectl get pods -n greenlang-gl008 -l app=steam-trap-inspector -o wide

# Monitor resource utilization after scaling
kubectl top pods -n greenlang-gl008 -l app=steam-trap-inspector
```

### Scale All Components

```bash
#!/bin/bash
# Scale all GL-008 components for high-load scenario

echo "Scaling GL-008 components for high-load..."

kubectl scale deployment/steam-trap-inspector -n greenlang-gl008 --replicas=12
kubectl scale deployment/api-gateway -n greenlang-gl008 --replicas=6
kubectl scale deployment/worker -n greenlang-gl008 --replicas=15
kubectl scale deployment/sensor-gateway -n greenlang-gl008 --replicas=5
kubectl scale deployment/ml-service -n greenlang-gl008 --replicas=4

echo "Waiting for all components to scale..."
kubectl rollout status deployment/steam-trap-inspector -n greenlang-gl008
kubectl rollout status deployment/api-gateway -n greenlang-gl008
kubectl rollout status deployment/worker -n greenlang-gl008
kubectl rollout status deployment/sensor-gateway -n greenlang-gl008
kubectl rollout status deployment/ml-service -n greenlang-gl008

echo "Scaling complete. Current pod count:"
kubectl get pods -n greenlang-gl008 | grep -E "(steam-trap|api-gateway|worker|sensor-gateway|ml-service)" | wc -l

echo "Resource utilization:"
kubectl top pods -n greenlang-gl008
```

### Gradual Scaling

```bash
#!/bin/bash
# Gradually scale up to avoid overwhelming dependencies

COMPONENT="steam-trap-inspector"
TARGET_REPLICAS=16
STEP_SIZE=2
WAIT_TIME=120  # seconds between steps

CURRENT=$(kubectl get deployment/$COMPONENT -n greenlang-gl008 -o jsonpath='{.spec.replicas}')

echo "Gradually scaling $COMPONENT from $CURRENT to $TARGET_REPLICAS replicas..."

for ((i=$CURRENT+$STEP_SIZE; i<=$TARGET_REPLICAS; i+=$STEP_SIZE)); do
  echo "Scaling to $i replicas..."
  kubectl scale deployment/$COMPONENT -n greenlang-gl008 --replicas=$i

  echo "Waiting for rollout..."
  kubectl rollout status deployment/$COMPONENT -n greenlang-gl008

  echo "Checking health..."
  HEALTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" https://api.greenlang.io/v1/steam-trap/health)

  if [ "$HEALTH_STATUS" -eq 200 ]; then
    echo "✓ Health check passed at $i replicas"
  else
    echo "⚠ Health check failed. Stopping scaling."
    exit 1
  fi

  if [ $i -lt $TARGET_REPLICAS ]; then
    echo "Waiting $WAIT_TIME seconds before next step..."
    sleep $WAIT_TIME
  fi
done

echo "✓ Gradual scaling complete: $TARGET_REPLICAS replicas"
```

---

## Vertical Scaling

### Increase Pod Resources

```bash
# Increase CPU and memory for steam-trap-inspector
kubectl patch deployment steam-trap-inspector -n greenlang-gl008 --type='json' -p='[
  {
    "op": "replace",
    "path": "/spec/template/spec/containers/0/resources/requests/cpu",
    "value": "2000m"
  },
  {
    "op": "replace",
    "path": "/spec/template/spec/containers/0/resources/requests/memory",
    "value": "4Gi"
  },
  {
    "op": "replace",
    "path": "/spec/template/spec/containers/0/resources/limits/cpu",
    "value": "4000m"
  },
  {
    "op": "replace",
    "path": "/spec/template/spec/containers/0/resources/limits/memory",
    "value": "8Gi"
  }
]'

# Restart to apply new resource allocations
kubectl rollout restart deployment/steam-trap-inspector -n greenlang-gl008
kubectl rollout status deployment/steam-trap-inspector -n greenlang-gl008

# Verify new resource allocations
kubectl describe pod -n greenlang-gl008 -l app=steam-trap-inspector | grep -A 5 "Limits:"
```

### ML Service Vertical Scaling

```bash
# ML inference requires significant CPU/memory
# Scale vertically for ML-intensive workloads

kubectl patch deployment ml-service -n greenlang-gl008 --type='json' -p='[
  {
    "op": "replace",
    "path": "/spec/template/spec/containers/0/resources/requests/cpu",
    "value": "8000m"
  },
  {
    "op": "replace",
    "path": "/spec/template/spec/containers/0/resources/requests/memory",
    "value": "16Gi"
  },
  {
    "op": "replace",
    "path": "/spec/template/spec/containers/0/resources/limits/cpu",
    "value": "16000m"
  },
  {
    "op": "replace",
    "path": "/spec/template/spec/containers/0/resources/limits/memory",
    "value": "32Gi"
  }
]'

# Consider GPU for ML inference at high scale
# Requires GPU-enabled node pool
kubectl patch deployment ml-service -n greenlang-gl008 --type='json' -p='[
  {
    "op": "add",
    "path": "/spec/template/spec/containers/0/resources/limits/nvidia.com~1gpu",
    "value": "1"
  }
]'
```

### Resource Right-Sizing

```bash
#!/bin/bash
# Analyze actual resource usage and recommend right-sizing

echo "=== GL-008 Resource Right-Sizing Analysis ==="
echo ""

for deployment in steam-trap-inspector api-gateway worker sensor-gateway ml-service; do
  echo "Component: $deployment"
  echo "----------------------------------------"

  # Get current requests/limits
  CURRENT_CPU_REQUEST=$(kubectl get deployment/$deployment -n greenlang-gl008 -o jsonpath='{.spec.template.spec.containers[0].resources.requests.cpu}')
  CURRENT_MEM_REQUEST=$(kubectl get deployment/$deployment -n greenlang-gl008 -o jsonpath='{.spec.template.spec.containers[0].resources.requests.memory}')

  echo "Current CPU Request: $CURRENT_CPU_REQUEST"
  echo "Current Memory Request: $CURRENT_MEM_REQUEST"

  # Get actual usage (average over last 24 hours)
  echo "Analyzing actual usage..."
  kubectl top pods -n greenlang-gl008 -l app=$deployment

  echo ""
done

echo "Recommendation: Update resource requests based on actual P95 usage + 20% buffer"
```

---

## Auto-Scaling Configuration

### Horizontal Pod Autoscaler (HPA)

```yaml
# File: hpa-steam-trap-inspector.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: steam-trap-inspector-hpa
  namespace: greenlang-gl008
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: steam-trap-inspector
  minReplicas: 4
  maxReplicas: 20
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
        averageUtilization: 75
  - type: Pods
    pods:
      metric:
        name: inspection_queue_depth
      target:
        type: AverageValue
        averageValue: "50"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
      - type: Pods
        value: 2
        periodSeconds: 60
      selectPolicy: Min
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
      - type: Pods
        value: 4
        periodSeconds: 30
      selectPolicy: Max
```

```bash
# Apply HPA configuration
kubectl apply -f hpa-steam-trap-inspector.yaml

# Verify HPA status
kubectl get hpa -n greenlang-gl008

# Watch HPA in action
kubectl get hpa steam-trap-inspector-hpa -n greenlang-gl008 --watch

# Describe HPA for detailed metrics
kubectl describe hpa steam-trap-inspector-hpa -n greenlang-gl008
```

### Custom Metrics for Autoscaling

```yaml
# File: custom-metrics-adapter.yaml
# Configure Prometheus Adapter for custom metrics

apiVersion: v1
kind: ConfigMap
metadata:
  name: adapter-config
  namespace: monitoring
data:
  config.yaml: |
    rules:
    - seriesQuery: 'inspection_queue_depth{namespace="greenlang-gl008"}'
      resources:
        overrides:
          namespace: {resource: "namespace"}
          pod: {resource: "pod"}
      name:
        matches: "^(.*)$"
        as: "inspection_queue_depth"
      metricsQuery: 'avg_over_time(inspection_queue_depth{<<.LabelMatchers>>}[2m])'

    - seriesQuery: 'api_request_latency_p95{namespace="greenlang-gl008"}'
      resources:
        overrides:
          namespace: {resource: "namespace"}
          pod: {resource: "pod"}
      name:
        matches: "^(.*)$"
        as: "api_latency_p95"
      metricsQuery: 'avg_over_time(api_request_latency_p95{<<.LabelMatchers>>}[5m])'

    - seriesQuery: 'sensor_data_ingestion_rate{namespace="greenlang-gl008"}'
      resources:
        overrides:
          namespace: {resource: "namespace"}
          pod: {resource: "pod"}
      name:
        matches: "^(.*)$"
        as: "sensor_ingestion_rate"
      metricsQuery: 'rate(sensor_data_ingestion_total{<<.LabelMatchers>>}[5m])'
```

### Vertical Pod Autoscaler (VPA)

```yaml
# File: vpa-steam-trap-inspector.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: steam-trap-inspector-vpa
  namespace: greenlang-gl008
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: steam-trap-inspector
  updatePolicy:
    updateMode: "Auto"  # Or "Initial" for recommendations only
  resourcePolicy:
    containerPolicies:
    - containerName: steam-trap-inspector
      minAllowed:
        cpu: 500m
        memory: 1Gi
      maxAllowed:
        cpu: 8000m
        memory: 16Gi
      controlledResources:
      - cpu
      - memory
```

```bash
# Apply VPA (requires VPA controller installed)
kubectl apply -f vpa-steam-trap-inspector.yaml

# Get VPA recommendations
kubectl describe vpa steam-trap-inspector-vpa -n greenlang-gl008
```

---

## Capacity Planning

### Capacity Planning Matrix

| Sites | Traps | Inspections/Day | Required Replicas | CPU Cores | Memory (GB) | DB Size (GB) |
|-------|-------|-----------------|-------------------|-----------|-------------|--------------|
| 10 | 1,000 | 10,000 | 4 | 10 | 20 | 50 |
| 50 | 5,000 | 50,000 | 8 | 20 | 40 | 150 |
| 100 | 10,000 | 100,000 | 12 | 30 | 60 | 300 |
| 250 | 25,000 | 250,000 | 20 | 50 | 100 | 750 |
| 500 | 50,000 | 500,000 | 32 | 80 | 160 | 1,500 |
| 1,000 | 100,000 | 1,000,000 | 50 | 125 | 250 | 3,000 |

### Growth Forecasting

```python
#!/usr/bin/env python3
# File: scripts/forecast_capacity.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def forecast_capacity_requirements(current_sites, growth_rate_monthly, months_ahead=12):
    """
    Forecast infrastructure capacity requirements based on growth projections

    Args:
        current_sites: Current number of sites
        growth_rate_monthly: Monthly growth rate (e.g., 0.10 for 10% monthly growth)
        months_ahead: Number of months to forecast
    """

    forecast = []

    for month in range(1, months_ahead + 1):
        projected_sites = int(current_sites * ((1 + growth_rate_monthly) ** month))
        projected_traps = projected_sites * 100  # Avg 100 traps per site
        projected_inspections_daily = projected_traps * 10  # 10 inspections per trap per day

        # Calculate required infrastructure
        required_replicas = max(4, int(projected_inspections_daily / 25000) * 4)
        required_cpu_cores = required_replicas * 2.5  # Avg 2.5 cores per replica
        required_memory_gb = required_replicas * 5    # Avg 5GB per replica
        required_db_size_gb = projected_traps * 3     # 3GB per 1000 traps

        monthly_cost = (
            required_cpu_cores * 30 * 0.04 +      # $0.04 per core-hour
            required_memory_gb * 30 * 0.005 +     # $0.005 per GB-hour
            required_db_size_gb * 0.10            # $0.10 per GB-month
        )

        forecast.append({
            'month': month,
            'date': (datetime.now() + timedelta(days=30*month)).strftime('%Y-%m'),
            'sites': projected_sites,
            'traps': projected_traps,
            'inspections_daily': projected_inspections_daily,
            'required_replicas': required_replicas,
            'required_cpu_cores': int(required_cpu_cores),
            'required_memory_gb': int(required_memory_gb),
            'required_db_size_gb': int(required_db_size_gb),
            'monthly_cost_usd': int(monthly_cost)
        })

    df = pd.DataFrame(forecast)
    print(df.to_string(index=False))

    # Identify capacity milestones
    print("\n=== Capacity Milestones ===")
    for idx, row in df.iterrows():
        if row['required_replicas'] > 12 and df.iloc[idx-1]['required_replicas'] <= 12:
            print(f"⚠️  {row['date']}: Upgrade to medium cluster (>12 replicas)")
        if row['required_replicas'] > 24 and df.iloc[idx-1]['required_replicas'] <= 24:
            print(f"⚠️  {row['date']}: Upgrade to large cluster (>24 replicas)")
        if row['required_db_size_gb'] > 500 and df.iloc[idx-1]['required_db_size_gb'] <= 500:
            print(f"⚠️  {row['date']}: Database scaling required (>500GB)")

if __name__ == "__main__":
    # Current state: 100 sites, 10% monthly growth
    forecast_capacity_requirements(
        current_sites=100,
        growth_rate_monthly=0.10,
        months_ahead=12
    )
```

### Pre-Scaling Checklist

```bash
#!/bin/bash
# Pre-scaling validation checklist

echo "=== GL-008 Pre-Scaling Checklist ==="
echo ""

READY=true

# Check 1: Node capacity
echo "[1/8] Checking node capacity..."
AVAILABLE_NODES=$(kubectl get nodes -l node.kubernetes.io/instance-type=n1-standard-4 | grep Ready | wc -l)
if [ "$AVAILABLE_NODES" -lt 5 ]; then
  echo "⚠️  WARNING: Only $AVAILABLE_NODES nodes available. Consider adding more nodes."
  READY=false
else
  echo "✓ Sufficient nodes available: $AVAILABLE_NODES"
fi

# Check 2: Database connection pool
echo "[2/8] Checking database connection pool..."
DB_POOL_SIZE=$(psql $DB_URL -t -c "SHOW max_connections;")
if [ "$DB_POOL_SIZE" -lt 200 ]; then
  echo "⚠️  WARNING: Database connection pool may be insufficient: $DB_POOL_SIZE"
  READY=false
else
  echo "✓ Database connection pool adequate: $DB_POOL_SIZE"
fi

# Check 3: Load balancer limits
echo "[3/8] Checking load balancer capacity..."
# Add load balancer health check
echo "✓ Load balancer check (manual verification required)"

# Check 4: Disk space
echo "[4/8] Checking disk space..."
DB_DISK_USAGE=$(psql $DB_URL -t -c "SELECT pg_size_pretty(pg_database_size('greenlang'));")
echo "Current database size: $DB_DISK_USAGE"
echo "✓ Disk space check (ensure >30% free)"

# Check 5: Network bandwidth
echo "[5/8] Checking network bandwidth..."
echo "✓ Network bandwidth (manual verification required)"

# Check 6: Rate limits
echo "[6/8] Checking external API rate limits..."
echo "✓ Rate limit check (manual verification required)"

# Check 7: Cost budget
echo "[7/8] Checking cost budget..."
echo "⚠️  Estimated monthly cost increase: Calculate based on scaling plan"

# Check 8: Monitoring alerts
echo "[8/8] Verifying monitoring alerts configured..."
ALERT_COUNT=$(kubectl get prometheusrules -n monitoring -l app=steam-trap-inspector -o name | wc -l)
if [ "$ALERT_COUNT" -gt 0 ]; then
  echo "✓ Monitoring alerts configured: $ALERT_COUNT rules"
else
  echo "⚠️  WARNING: No monitoring alerts found"
  READY=false
fi

echo ""
if [ "$READY" = true ]; then
  echo "✓ Ready to scale"
  exit 0
else
  echo "⚠️  Address warnings before scaling"
  exit 1
fi
```

---

## Performance Benchmarks

### Baseline Performance Metrics

**Single Pod Performance:**
- Inspections/minute: 100
- API requests/second: 250
- ML predictions/second: 50
- Memory usage: 1.8 GB avg
- CPU usage: 0.8 cores avg

**Cluster Performance (4 replicas):**
- Inspections/minute: 400
- API requests/second: 1,000
- ML predictions/second: 200
- End-to-end inspection latency: 12 seconds (P95)
- API latency: 250ms (P95)

### Load Testing

```bash
#!/bin/bash
# Load testing script for GL-008

LOAD_TEST_DURATION=300  # 5 minutes
TARGET_RPS=500

echo "Starting load test..."
echo "Target: $TARGET_RPS req/sec for $LOAD_TEST_DURATION seconds"

# Use k6 for load testing
k6 run --vus 50 --duration ${LOAD_TEST_DURATION}s - <<EOF
import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
  stages: [
    { duration: '60s', target: 100 },   // Ramp up to 100 RPS
    { duration: '60s', target: 300 },   // Ramp up to 300 RPS
    { duration: '60s', target: $TARGET_RPS }, // Reach target
    { duration: '60s', target: $TARGET_RPS }, // Sustain
    { duration: '60s', target: 0 },     // Ramp down
  ],
  thresholds: {
    'http_req_duration': ['p(95)<2000'], // 95% requests < 2s
    'http_req_failed': ['rate<0.05'],    // <5% error rate
  },
};

export default function () {
  const res = http.post(
    'https://api.greenlang.io/v1/steam-trap/inspection',
    JSON.stringify({
      trap_id: 'trap_test_\${__VU}_\${__ITER}',
      site_id: 'site_001',
      sensor_data: {
        acoustic_frequency: 1250,
        thermal_temp: 185.5,
        pressure: 150
      }
    }),
    {
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer \${__ENV.API_TOKEN}'
      }
    }
  );

  check(res, {
    'status is 200': (r) => r.status === 200,
    'response time < 2s': (r) => r.timings.duration < 2000,
  });

  sleep(1);
}
EOF

echo "Load test complete. Check Grafana for detailed metrics:"
echo "https://grafana.greenlang.io/d/gl008-performance"
```

### Stress Testing

```bash
#!/bin/bash
# Stress test to find breaking point

echo "=== GL-008 Stress Test ==="
echo "WARNING: This will push the system to its limits"
read -p "Proceed? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
  exit 0
fi

START_RPS=100
MAX_RPS=2000
STEP=100
DURATION_PER_STEP=120  # 2 minutes per step

for rps in $(seq $START_RPS $STEP $MAX_RPS); do
  echo ""
  echo "Testing at $rps RPS..."

  # Run load test
  k6 run --vus $((rps/10)) --duration ${DURATION_PER_STEP}s load-test.js

  # Check system health
  ERROR_RATE=$(kubectl logs -n greenlang-gl008 -l app=steam-trap-inspector --since=2m | grep ERROR | wc -l)
  LATENCY_P95=$(curl -s https://api.greenlang.io/v1/steam-trap/metrics/latency | jq -r '.p95_latency_ms')

  echo "Error count (2m): $ERROR_RATE"
  echo "Latency P95: ${LATENCY_P95}ms"

  # Check if system is breaking
  if [ "$ERROR_RATE" -gt 100 ] || [ "$LATENCY_P95" -gt 5000 ]; then
    echo ""
    echo "⚠️  System breaking point reached at $rps RPS"
    echo "Error rate: $ERROR_RATE errors/2min"
    echo "Latency P95: ${LATENCY_P95}ms"
    break
  fi

  sleep 30  # Cool down between tests
done

echo ""
echo "Stress test complete"
```

---

## Multi-Site Deployment

### Regional Deployment Architecture

```yaml
# File: regional-deployment-us-east.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: steam-trap-inspector-us-east
  namespace: greenlang-gl008
  labels:
    region: us-east-1
spec:
  replicas: 6
  selector:
    matchLabels:
      app: steam-trap-inspector
      region: us-east-1
  template:
    metadata:
      labels:
        app: steam-trap-inspector
        region: us-east-1
    spec:
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: topology.kubernetes.io/region
                operator: In
                values:
                - us-east-1
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - steam-trap-inspector
              topologyKey: kubernetes.io/hostname
      containers:
      - name: steam-trap-inspector
        image: greenlang/steam-trap-inspector:v2.4.2
        env:
        - name: REGION
          value: "us-east-1"
        - name: DB_URL
          value: "postgresql://db-us-east-1.greenlang.io:5432/greenlang"
        resources:
          requests:
            cpu: 1000m
            memory: 2Gi
          limits:
            cpu: 2000m
            memory: 4Gi
```

### Multi-Region Traffic Distribution

```yaml
# File: global-load-balancer.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: steam-trap-inspector-global
  annotations:
    kubernetes.io/ingress.class: "gce"
    ingress.gcp.kubernetes.io/pre-shared-cert: "greenlang-ssl-cert"
spec:
  rules:
  - host: api.greenlang.io
    http:
      paths:
      - path: /v1/steam-trap/*
        pathType: Prefix
        backend:
          service:
            name: steam-trap-inspector-lb
            port:
              number: 80
---
apiVersion: v1
kind: Service
metadata:
  name: steam-trap-inspector-lb
  annotations:
    cloud.google.com/backend-config: '{"default": "steam-trap-backendconfig"}'
spec:
  type: LoadBalancer
  selector:
    app: steam-trap-inspector
  ports:
  - port: 80
    targetPort: 8080
```

### Cross-Region Failover

```bash
#!/bin/bash
# Failover from primary to secondary region

PRIMARY_REGION="us-east-1"
SECONDARY_REGION="us-west-2"

echo "Initiating failover from $PRIMARY_REGION to $SECONDARY_REGION..."

# Step 1: Scale up secondary region
kubectl scale deployment/steam-trap-inspector-$SECONDARY_REGION \
  -n greenlang-gl008 --replicas=12

# Step 2: Wait for secondary to be ready
kubectl rollout status deployment/steam-trap-inspector-$SECONDARY_REGION -n greenlang-gl008

# Step 3: Update DNS/load balancer to route to secondary
# (Implementation depends on cloud provider)

# Step 4: Drain primary region
kubectl scale deployment/steam-trap-inspector-$PRIMARY_REGION \
  -n greenlang-gl008 --replicas=0

echo "Failover complete. Traffic now routed to $SECONDARY_REGION"
```

---

## Database Scaling

### Read Replicas

```bash
# Add read replica for read-heavy workloads
# Configure application to use read replica for queries

kubectl set env deployment/steam-trap-inspector -n greenlang-gl008 \
  DB_READ_REPLICA_URL="postgresql://db-read-replica.greenlang.io:5432/greenlang" \
  ENABLE_READ_REPLICA=true \
  READ_REPLICA_SPLIT_RATIO=0.8  # 80% reads to replica

kubectl rollout restart deployment/steam-trap-inspector -n greenlang-gl008
```

### Connection Pooling

```bash
# Use PgBouncer for connection pooling
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pgbouncer
  namespace: greenlang-gl008
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pgbouncer
  template:
    metadata:
      labels:
        app: pgbouncer
    spec:
      containers:
      - name: pgbouncer
        image: edoburu/pgbouncer:latest
        env:
        - name: DATABASE_URL
          value: "postgresql://db.greenlang.io:5432/greenlang"
        - name: MAX_CLIENT_CONN
          value: "1000"
        - name: DEFAULT_POOL_SIZE
          value: "25"
        ports:
        - containerPort: 5432
EOF

# Update application to connect via PgBouncer
kubectl set env deployment/steam-trap-inspector -n greenlang-gl008 \
  DB_URL="postgresql://pgbouncer.greenlang-gl008.svc.cluster.local:5432/greenlang"
```

### Database Partitioning

```sql
-- Partition trap_inspections table by date for better performance
-- This should be done during a maintenance window

-- Create partitioned table
CREATE TABLE trap_inspections_partitioned (
    LIKE trap_inspections INCLUDING ALL
) PARTITION BY RANGE (detected_at);

-- Create monthly partitions
CREATE TABLE trap_inspections_2025_11 PARTITION OF trap_inspections_partitioned
    FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');

CREATE TABLE trap_inspections_2025_12 PARTITION OF trap_inspections_partitioned
    FOR VALUES FROM ('2025-12-01') TO ('2026-01-01');

-- Create future partitions automatically
CREATE OR REPLACE FUNCTION create_monthly_partitions()
RETURNS void AS $$
DECLARE
    start_date date;
    end_date date;
    partition_name text;
BEGIN
    FOR i IN 0..11 LOOP
        start_date := date_trunc('month', CURRENT_DATE + interval '1 month' * i);
        end_date := start_date + interval '1 month';
        partition_name := 'trap_inspections_' || to_char(start_date, 'YYYY_MM');

        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS %I PARTITION OF trap_inspections_partitioned FOR VALUES FROM (%L) TO (%L)',
            partition_name, start_date, end_date
        );
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Schedule monthly partition creation
SELECT cron.schedule('create-monthly-partitions', '0 0 1 * *', 'SELECT create_monthly_partitions()');
```

---

## Monitoring & Metrics

### Key Scaling Metrics

```bash
# Create Grafana dashboard for scaling metrics

cat > grafana-scaling-dashboard.json <<EOF
{
  "dashboard": {
    "title": "GL-008 Scaling Metrics",
    "panels": [
      {
        "title": "Pod Count Over Time",
        "targets": [{
          "expr": "count(kube_pod_info{namespace='greenlang-gl008', pod=~'steam-trap-inspector.*'})"
        }]
      },
      {
        "title": "CPU Utilization",
        "targets": [{
          "expr": "avg(rate(container_cpu_usage_seconds_total{namespace='greenlang-gl008'}[5m])) * 100"
        }]
      },
      {
        "title": "Memory Utilization",
        "targets": [{
          "expr": "avg(container_memory_working_set_bytes{namespace='greenlang-gl008'}) / avg(container_spec_memory_limit_bytes{namespace='greenlang-gl008'}) * 100"
        }]
      },
      {
        "title": "Request Throughput",
        "targets": [{
          "expr": "rate(http_requests_total{namespace='greenlang-gl008'}[5m])"
        }]
      }
    ]
  }
}
EOF
```

### Scaling Alerts

```yaml
# File: prometheus-scaling-alerts.yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: gl008-scaling-alerts
  namespace: monitoring
spec:
  groups:
  - name: gl008-scaling
    interval: 30s
    rules:
    - alert: HighCPUUtilization
      expr: avg(rate(container_cpu_usage_seconds_total{namespace="greenlang-gl008"}[5m])) > 0.8
      for: 5m
      annotations:
        summary: "High CPU utilization - consider scaling up"
        description: "CPU utilization is {{ $value }}%, exceeding 80% threshold"

    - alert: HighMemoryUtilization
      expr: avg(container_memory_working_set_bytes{namespace="greenlang-gl008"}) / avg(container_spec_memory_limit_bytes{namespace="greenlang-gl008"}) > 0.85
      for: 5m
      annotations:
        summary: "High memory utilization - consider scaling up"

    - alert: InspectionQueueBacklog
      expr: inspection_queue_depth{namespace="greenlang-gl008"} > 500
      for: 10m
      annotations:
        summary: "Large inspection queue backlog - scale up workers"

    - alert: LowResourceUtilization
      expr: avg(rate(container_cpu_usage_seconds_total{namespace="greenlang-gl008"}[15m])) < 0.2
      for: 30m
      annotations:
        summary: "Low resource utilization - consider scaling down"
```

---

## Cost Optimization

### Cost-Effective Scaling Strategies

```bash
# Use spot instances for non-critical workloads
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: worker-spot
  namespace: greenlang-gl008
spec:
  replicas: 10
  selector:
    matchLabels:
      app: worker
      instance-type: spot
  template:
    metadata:
      labels:
        app: worker
        instance-type: spot
    spec:
      nodeSelector:
        cloud.google.com/gke-preemptible: "true"
      tolerations:
      - key: cloud.google.com/gke-preemptible
        operator: Equal
        value: "true"
        effect: NoSchedule
      containers:
      - name: worker
        image: greenlang/worker:v2.4.2
EOF

# Scale down during off-peak hours
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: CronJob
metadata:
  name: scale-down-off-peak
  namespace: greenlang-gl008
spec:
  schedule: "0 22 * * *"  # 10 PM daily
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: kubectl
            image: bitnami/kubectl:latest
            command:
            - /bin/sh
            - -c
            - |
              kubectl scale deployment/steam-trap-inspector -n greenlang-gl008 --replicas=2
              kubectl scale deployment/worker -n greenlang-gl008 --replicas=3
          restartPolicy: OnFailure
---
apiVersion: batch/v1
kind: CronJob
metadata:
  name: scale-up-peak
  namespace: greenlang-gl008
spec:
  schedule: "0 6 * * *"  # 6 AM daily
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: kubectl
            image: bitnami/kubectl:latest
            command:
            - /bin/sh
            - -c
            - |
              kubectl scale deployment/steam-trap-inspector -n greenlang-gl008 --replicas=8
              kubectl scale deployment/worker -n greenlang-gl008 --replicas=12
          restartPolicy: OnFailure
EOF
```

---

**Document Version:** 1.0.0
**Last Reviewed:** 2025-11-26
**Next Review:** 2026-02-26
**Maintained By:** Platform Operations Team
