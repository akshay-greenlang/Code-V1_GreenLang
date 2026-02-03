# High CPU Usage Runbook

## Alert Names
- `NodeHighCpuUsageWarning`
- `NodeHighCpuUsageCritical`
- `PodHighCpuUsage`
- `RDSHighCpuUsage`

## Overview

This runbook provides guidance for responding to high CPU usage alerts across GreenLang infrastructure components including Kubernetes nodes, pods, and RDS databases.

---

## Quick Reference

| Severity | Threshold | Response Time | Escalation |
|----------|-----------|---------------|------------|
| Warning  | >70%      | 30 minutes    | On-call engineer |
| Critical | >85%      | 5 minutes     | Incident commander |

---

## Diagnosis Steps

### 1. Identify the Affected Component

```bash
# Check which nodes have high CPU
kubectl top nodes

# Check which pods are consuming the most CPU
kubectl top pods --all-namespaces --sort-by=cpu

# For a specific namespace
kubectl top pods -n greenlang --sort-by=cpu
```

### 2. Node-Level Investigation

```bash
# Get detailed node information
kubectl describe node <node-name>

# Check node conditions
kubectl get node <node-name> -o jsonpath='{.status.conditions[*]}' | jq

# SSH to the node (if needed) and check processes
ssh <node-ip>
top -c -o %CPU
htop
```

### 3. Pod-Level Investigation

```bash
# Get pod details
kubectl describe pod <pod-name> -n <namespace>

# Check pod resource limits and requests
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.spec.containers[*].resources}'

# Check if pod is being throttled
kubectl exec -it <pod-name> -n <namespace> -- cat /sys/fs/cgroup/cpu/cpu.stat
```

### 4. Application-Level Investigation

```bash
# Check application logs for issues
kubectl logs <pod-name> -n <namespace> --tail=100

# Get into the container for debugging
kubectl exec -it <pod-name> -n <namespace> -- /bin/sh

# Inside container, check CPU-intensive processes
top -c
ps aux --sort=-%cpu | head -20
```

### 5. RDS Investigation

```bash
# Check RDS performance insights (AWS CLI)
aws rds describe-db-instances --db-instance-identifier greenlang-db

# Check slow query log
aws rds download-db-log-file-portion \
  --db-instance-identifier greenlang-db \
  --log-file-name slowquery/mysql-slowquery.log

# Check active connections and queries
# Connect to the database
psql -h <rds-endpoint> -U <username> -d greenlang

# Run diagnostic queries
SELECT * FROM pg_stat_activity WHERE state = 'active';
SELECT * FROM pg_stat_statements ORDER BY total_time DESC LIMIT 10;
```

---

## Remediation Steps

### Immediate Actions

#### For Node High CPU

1. **Identify the cause**
   ```bash
   # Check which pods are using the most CPU on the node
   kubectl get pods --all-namespaces -o wide | grep <node-name>
   kubectl top pods --all-namespaces --sort-by=cpu
   ```

2. **If a runaway pod is identified**
   ```bash
   # Option 1: Restart the pod
   kubectl delete pod <pod-name> -n <namespace>

   # Option 2: Scale down the deployment temporarily
   kubectl scale deployment <deployment-name> -n <namespace> --replicas=0
   ```

3. **If the node is overloaded**
   ```bash
   # Cordon the node to prevent new pods
   kubectl cordon <node-name>

   # Drain non-critical pods (carefully)
   kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data
   ```

#### For Pod High CPU

1. **Check if CPU limits are too low**
   ```bash
   # Increase CPU limits if needed
   kubectl patch deployment <deployment-name> -n <namespace> --type='json' \
     -p='[{"op": "replace", "path": "/spec/template/spec/containers/0/resources/limits/cpu", "value": "1000m"}]'
   ```

2. **Scale horizontally if appropriate**
   ```bash
   kubectl scale deployment <deployment-name> -n <namespace> --replicas=5
   ```

#### For RDS High CPU

1. **Identify expensive queries**
   ```sql
   -- Find queries consuming the most CPU
   SELECT query, calls, total_time, mean_time
   FROM pg_stat_statements
   ORDER BY total_time DESC
   LIMIT 20;

   -- Check for long-running queries
   SELECT pid, now() - pg_stat_activity.query_start AS duration, query
   FROM pg_stat_activity
   WHERE state != 'idle'
   ORDER BY duration DESC;
   ```

2. **Kill expensive queries if needed**
   ```sql
   -- Terminate a specific query
   SELECT pg_terminate_backend(<pid>);
   ```

3. **Scale RDS if needed**
   ```bash
   # Modify RDS instance class (causes brief downtime)
   aws rds modify-db-instance \
     --db-instance-identifier greenlang-db \
     --db-instance-class db.r5.large \
     --apply-immediately
   ```

### Long-term Fixes

1. **Optimize Application Code**
   - Profile CPU-intensive code paths
   - Implement caching for expensive computations
   - Use async/await for I/O-bound operations

2. **Improve Resource Allocation**
   ```yaml
   # Example resource configuration
   resources:
     requests:
       cpu: "500m"
       memory: "512Mi"
     limits:
       cpu: "1000m"
       memory: "1Gi"
   ```

3. **Implement HPA (Horizontal Pod Autoscaler)**
   ```yaml
   apiVersion: autoscaling/v2
   kind: HorizontalPodAutoscaler
   metadata:
     name: greenlang-app-hpa
   spec:
     scaleTargetRef:
       apiVersion: apps/v1
       kind: Deployment
       name: greenlang-app
     minReplicas: 3
     maxReplicas: 10
     metrics:
     - type: Resource
       resource:
         name: cpu
         target:
           type: Utilization
           averageUtilization: 70
   ```

4. **Database Optimization**
   - Add missing indexes
   - Optimize slow queries
   - Implement connection pooling
   - Consider read replicas for read-heavy workloads

---

## Verification

After taking remediation actions, verify the issue is resolved:

```bash
# Check CPU has decreased
kubectl top nodes
kubectl top pods -n greenlang

# Verify pods are healthy
kubectl get pods -n greenlang

# Check Prometheus metrics
curl -s "http://prometheus:9090/api/v1/query?query=node_cpu_seconds_total" | jq
```

---

## Escalation

If the issue persists after following this runbook:

1. **Warning Level (>70%, 30+ minutes)**
   - Page on-call engineer
   - Slack: #greenlang-oncall

2. **Critical Level (>85%, 5+ minutes)**
   - Page incident commander
   - Create incident in PagerDuty
   - Slack: #greenlang-incidents
   - Consider activating disaster recovery procedures

---

## Related Runbooks

- [High Memory Usage Runbook](./high-memory-runbook.md)
- [Pod Crash Loop Runbook](./pod-crash-loop-runbook.md)
- [Database Connection Issues Runbook](./database-connection-runbook.md)

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2024-01-15 | Platform Team | Initial version |
| 1.1 | 2024-02-01 | SRE Team | Added RDS section |
