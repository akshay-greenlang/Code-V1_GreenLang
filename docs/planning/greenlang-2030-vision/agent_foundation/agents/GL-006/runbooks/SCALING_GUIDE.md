# GL-006 HeatRecoveryMaximizer Scaling Guide

## Document Information

| Field | Value |
|-------|-------|
| Agent ID | GL-006 |
| Codename | HEATRECLAIM |
| Version | 1.0.0 |
| Last Updated | 2024-11-26 |

---

## 1. Overview

This guide provides procedures and best practices for scaling the GL-006 HeatRecoveryMaximizer agent to handle varying workloads. It covers horizontal and vertical scaling, autoscaling configuration, and capacity planning.

---

## 2. Scaling Architecture

### 2.1 Current Configuration

| Environment | Min Replicas | Max Replicas | CPU Request | Memory Request |
|-------------|--------------|--------------|-------------|----------------|
| Development | 1 | 2 | 100m | 128Mi |
| Staging | 2 | 5 | 200m | 256Mi |
| Production | 3 | 10 | 500m | 512Mi |

### 2.2 Scaling Dimensions

1. **Horizontal Scaling** - Add/remove pod replicas
2. **Vertical Scaling** - Increase/decrease pod resources
3. **Database Scaling** - Scale PostgreSQL connections/resources
4. **Cache Scaling** - Scale Redis capacity

---

## 3. Horizontal Scaling

### 3.1 Manual Horizontal Scaling

```bash
# Scale to specific number of replicas
kubectl scale deployment gl-006-heatreclaim --replicas=5 -n greenlang

# Verify scaling
kubectl get pods -n greenlang -l app=gl-006-heatreclaim -w

# Check deployment status
kubectl get deployment gl-006-heatreclaim -n greenlang
```

### 3.2 Horizontal Pod Autoscaler (HPA)

The HPA automatically scales based on metrics:

```yaml
# Current HPA configuration
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gl-006-heatreclaim-hpa
  namespace: greenlang
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gl-006-heatreclaim
  minReplicas: 3
  maxReplicas: 10
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
```

### 3.3 Modifying HPA

```bash
# View current HPA
kubectl get hpa gl-006-heatreclaim-hpa -n greenlang -o yaml

# Edit HPA
kubectl edit hpa gl-006-heatreclaim-hpa -n greenlang

# Or patch specific values
kubectl patch hpa gl-006-heatreclaim-hpa -n greenlang -p '{"spec":{"minReplicas":5,"maxReplicas":15}}'
```

### 3.4 HPA Tuning Parameters

| Parameter | Description | Recommendation |
|-----------|-------------|----------------|
| minReplicas | Minimum pods | >= 3 for HA |
| maxReplicas | Maximum pods | Based on cluster capacity |
| CPU target | CPU utilization threshold | 60-70% |
| Memory target | Memory utilization threshold | 70-80% |
| scaleDown stabilization | Cooldown before scale down | 300s |
| scaleUp stabilization | Cooldown before scale up | 0s |

---

## 4. Vertical Scaling

### 4.1 Resource Recommendations

| Workload Type | CPU Request | CPU Limit | Memory Request | Memory Limit |
|---------------|-------------|-----------|----------------|--------------|
| Light | 250m | 500m | 256Mi | 512Mi |
| Normal | 500m | 1000m | 512Mi | 1Gi |
| Heavy | 1000m | 2000m | 1Gi | 2Gi |
| Intensive | 2000m | 4000m | 2Gi | 4Gi |

### 4.2 Adjusting Resources

```bash
# Update resource limits
kubectl patch deployment gl-006-heatreclaim -n greenlang -p '{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "gl-006-heatreclaim",
          "resources": {
            "requests": {
              "cpu": "500m",
              "memory": "512Mi"
            },
            "limits": {
              "cpu": "1000m",
              "memory": "1Gi"
            }
          }
        }]
      }
    }
  }
}'

# Verify update
kubectl describe deployment gl-006-heatreclaim -n greenlang | grep -A 10 "Limits"
```

### 4.3 Vertical Pod Autoscaler (VPA)

If VPA is installed:

```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: gl-006-heatreclaim-vpa
  namespace: greenlang
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gl-006-heatreclaim
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
      - containerName: gl-006-heatreclaim
        minAllowed:
          cpu: "100m"
          memory: "128Mi"
        maxAllowed:
          cpu: "4000m"
          memory: "4Gi"
```

---

## 5. Database Scaling

### 5.1 Connection Pool Scaling

```bash
# Update connection pool size in ConfigMap
kubectl edit configmap gl-006-heatreclaim-config -n greenlang

# Key parameters:
# DATABASE_POOL_SIZE: 10 -> 20
# DATABASE_MAX_OVERFLOW: 20 -> 40

# Restart to apply
kubectl rollout restart deployment gl-006-heatreclaim -n greenlang
```

### 5.2 PostgreSQL Scaling

```bash
# Check PostgreSQL connections
kubectl exec -it postgresql-0 -n greenlang -- psql -U greenlang -c "SELECT count(*) FROM pg_stat_activity;"

# Scale PostgreSQL if using operator
kubectl patch postgresql gl-006-db -n greenlang -p '{"spec":{"numberOfInstances":3}}'
```

---

## 6. Cache Scaling

### 6.1 Redis Memory Scaling

```bash
# Check Redis memory usage
kubectl exec -it redis-0 -n greenlang -- redis-cli INFO memory

# Increase Redis memory if needed
kubectl patch statefulset redis -n greenlang -p '{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "redis",
          "resources": {
            "limits": {
              "memory": "2Gi"
            }
          }
        }]
      }
    }
  }
}'
```

### 6.2 Redis Cluster Scaling

If using Redis Cluster:

```bash
# Add Redis nodes
kubectl scale statefulset redis -n greenlang --replicas=6

# Rebalance cluster
kubectl exec -it redis-0 -n greenlang -- redis-cli --cluster rebalance
```

---

## 7. Load-Based Scaling

### 7.1 Scaling Triggers

| Metric | Threshold | Action |
|--------|-----------|--------|
| CPU > 70% | 5 minutes | Scale up |
| CPU < 30% | 10 minutes | Scale down |
| Memory > 80% | 5 minutes | Scale up |
| Request latency P99 > 5s | 3 minutes | Scale up |
| Queue size > 100 | Immediate | Scale up |

### 7.2 Proactive Scaling

For expected load increases:

```bash
# Pre-scale before expected load
kubectl scale deployment gl-006-heatreclaim --replicas=10 -n greenlang

# Set minimum replicas higher
kubectl patch hpa gl-006-heatreclaim-hpa -n greenlang -p '{"spec":{"minReplicas":8}}'
```

---

## 8. Capacity Planning

### 8.1 Resource Requirements per Request Type

| Operation | CPU (avg) | Memory (avg) | Duration (avg) |
|-----------|-----------|--------------|----------------|
| Stream Analysis | 50m | 50Mi | 0.5s |
| Pinch Analysis | 200m | 100Mi | 2s |
| Network Synthesis | 500m | 200Mi | 30s |
| ROI Calculation | 100m | 50Mi | 0.5s |
| Full Analysis | 800m | 300Mi | 60s |

### 8.2 Capacity Calculation

```
Required Replicas = (Requests/sec) * (Avg Duration) / (Target CPU Utilization) * (CPU per Request)

Example:
- 100 requests/sec
- 2s avg duration
- 70% target utilization
- 200m avg CPU

Replicas = 100 * 2 / 0.7 / 5 = ~57 replicas @ 1000m each
Or ~6 replicas @ 10000m (10 cores) each
```

### 8.3 Scaling Recommendations by Load

| Daily Requests | Min Replicas | Max Replicas | Resources |
|----------------|--------------|--------------|-----------|
| < 10,000 | 2 | 4 | Light |
| 10,000 - 50,000 | 3 | 6 | Normal |
| 50,000 - 200,000 | 5 | 10 | Normal |
| 200,000 - 500,000 | 8 | 15 | Heavy |
| > 500,000 | 10 | 25 | Intensive |

---

## 9. Scaling Best Practices

### 9.1 General Guidelines

1. **Start Conservative** - Begin with lower limits and increase as needed
2. **Monitor Before Scaling** - Understand current utilization patterns
3. **Scale Gradually** - Avoid sudden large scale changes
4. **Test Scaling** - Verify scaling behavior in staging first
5. **Document Changes** - Record all scaling decisions

### 9.2 Anti-Patterns to Avoid

- Over-provisioning (wasting resources)
- Under-provisioning (poor performance)
- Ignoring scale-down (cost accumulation)
- Scaling without monitoring
- Manual scaling for predictable patterns

### 9.3 Pod Disruption Budget

Ensure PDB is configured correctly:

```bash
# Check PDB
kubectl get pdb gl-006-heatreclaim-pdb -n greenlang

# Ensure minAvailable is appropriate
# For 3 replicas: minAvailable=2
# For 5 replicas: minAvailable=3
# For 10 replicas: minAvailable=7
```

---

## 10. Monitoring Scaling

### 10.1 Key Metrics to Watch

```bash
# HPA status
kubectl get hpa gl-006-heatreclaim-hpa -n greenlang -w

# Pod resource usage
kubectl top pods -n greenlang -l app=gl-006-heatreclaim

# Node resource usage
kubectl top nodes

# Custom metrics
curl -s localhost:9090/metrics | grep gl006_active_calculations
curl -s localhost:9090/metrics | grep gl006_request_queue_size
```

### 10.2 Scaling Events

```bash
# View scaling events
kubectl get events -n greenlang --field-selector reason=SuccessfulRescale

# HPA events
kubectl describe hpa gl-006-heatreclaim-hpa -n greenlang | grep -A 20 "Events"
```

---

## 11. Emergency Scaling Procedures

### 11.1 Emergency Scale-Up

```bash
# Immediate scale up
kubectl scale deployment gl-006-heatreclaim --replicas=15 -n greenlang

# Temporarily increase HPA max
kubectl patch hpa gl-006-heatreclaim-hpa -n greenlang -p '{"spec":{"maxReplicas":20}}'

# Verify pods are starting
kubectl get pods -n greenlang -l app=gl-006-heatreclaim -w
```

### 11.2 Emergency Scale-Down

```bash
# Scale down safely
kubectl scale deployment gl-006-heatreclaim --replicas=3 -n greenlang

# Restore HPA settings
kubectl patch hpa gl-006-heatreclaim-hpa -n greenlang -p '{"spec":{"minReplicas":3,"maxReplicas":10}}'
```

---

## 12. Related Documents

- [INCIDENT_RESPONSE.md](./INCIDENT_RESPONSE.md)
- [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)
- [ROLLBACK_PROCEDURE.md](./ROLLBACK_PROCEDURE.md)
- [MAINTENANCE.md](./MAINTENANCE.md)

---

*This guide is maintained by the Platform Team. For updates, contact platform-team@greenlang.io*
