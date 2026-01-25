# Resource Utilization Runbook

This runbook covers alerts related to CPU, memory, and disk resource utilization in the GreenLang platform.

---

## Table of Contents

- [HighCPUUsage](#highcpuusage)
- [CriticalCPUUsage](#criticalcpuusage)
- [HighMemoryUsage](#highmemoryusage)
- [CriticalMemoryUsage](#criticalmemoryusage)
- [DiskSpaceLow](#diskspacelow)

---

## HighCPUUsage

### Alert Details

| Property | Value |
|----------|-------|
| **Alert Name** | HighCPUUsage |
| **Severity** | Warning |
| **Team** | Platform |
| **Evaluation Interval** | 60s |
| **For Duration** | 10m |
| **Threshold** | 80% |

**PromQL Expression:**

```promql
(
  100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)
) > 80
```

### Description

This alert fires when average CPU utilization on a node exceeds 80% for 10 minutes. This is a warning that the node is under heavy load and may soon become saturated.

### Impact Assessment

| Impact Area | Severity | Description |
|-------------|----------|-------------|
| **User Impact** | Low | No immediate impact, but risk increasing |
| **Data Impact** | None | Data integrity not affected |
| **SLA Impact** | Low | May lead to latency issues |
| **Revenue Impact** | Low | Early warning for capacity |

### Diagnostic Steps

1. **Identify CPU consumers**

   ```bash
   # SSH to the node or use kubectl
   kubectl debug node/{{ $labels.instance }} -it --image=busybox -- top -b -n1 | head -20

   # Or check container-level CPU
   kubectl top pods --all-namespaces --sort-by=cpu | head -20
   ```

2. **Check CPU usage by container**

   ```bash
   # Query container CPU usage
   curl -s "http://prometheus:9090/api/v1/query?query=sum(rate(container_cpu_usage_seconds_total{instance='{{ $labels.instance }}'}[5m]))by(pod,container)*100" | jq '.data.result | sort_by(.value[1] | tonumber) | reverse | .[0:10]'
   ```

3. **Check for CPU throttling**

   ```bash
   # Check throttling metrics
   curl -s "http://prometheus:9090/api/v1/query?query=sum(rate(container_cpu_cfs_throttled_periods_total[5m]))by(pod)" | jq '.data.result | sort_by(.value[1] | tonumber) | reverse | .[0:10]'
   ```

4. **Check node specifications**

   ```bash
   # Get node capacity
   kubectl describe node {{ $labels.instance }} | grep -A5 "Capacity:"

   # Get current allocation
   kubectl describe node {{ $labels.instance }} | grep -A5 "Allocated resources:"
   ```

5. **Review recent deployments**

   ```bash
   # Check for recent deployments that might have increased CPU
   kubectl get deployments --all-namespaces -o wide | head -20
   ```

### Resolution Steps

#### Scenario 1: Single pod consuming excessive CPU

```bash
# 1. Identify the pod
kubectl top pods --all-namespaces --sort-by=cpu | head -5

# 2. Check if it's expected (batch job, calculation spike)
kubectl logs -n {{ namespace }} {{ pod }} --tail=50

# 3. If runaway process, restart the pod
kubectl delete pod -n {{ namespace }} {{ pod }}

# 4. If recurring, investigate code or set CPU limits
kubectl patch deployment -n {{ namespace }} {{ deployment }} \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"app","resources":{"limits":{"cpu":"2"}}}]}}}}'
```

#### Scenario 2: Load spike across multiple pods

```bash
# 1. Check request rate
curl -s "http://prometheus:9090/api/v1/query?query=sum(rate(http_requests_total[5m]))" | jq .

# 2. Scale affected deployments
kubectl scale deployment -n greenlang <deployment> --replicas=5

# 3. If cluster-wide, add nodes
# For AWS:
aws autoscaling set-desired-capacity \
  --auto-scaling-group-name greenlang-nodes \
  --desired-capacity 10

# For GKE:
gcloud container clusters resize greenlang-cluster \
  --num-nodes=10 --zone=us-central1-a
```

#### Scenario 3: Inefficient workload

```bash
# 1. Profile the application
kubectl exec -n greenlang deploy/<deployment> -- \
  python -m cProfile -s cumtime app.py

# 2. Check for CPU-intensive operations
kubectl logs -n greenlang -l app=<app> | \
  grep -E "calculation|processing" | tail -20

# 3. Optimize code or offload to worker nodes
```

#### Scenario 4: Node overcommitted

```bash
# 1. Check node allocation vs capacity
kubectl describe node {{ $labels.instance }} | grep -A10 "Allocated"

# 2. Evict non-critical pods if needed
kubectl cordon {{ $labels.instance }}
kubectl drain {{ $labels.instance }} --ignore-daemonsets --delete-emptydir-data

# 3. Rebalance workloads
kubectl uncordon {{ $labels.instance }}
```

### Escalation Path

| Level | Team | Contact | When to Escalate |
|-------|------|---------|------------------|
| L1 | On-Call Engineer | PagerDuty | Initial response |
| L2 | Platform Team | #platform-oncall Slack | If not resolved in 30 minutes |
| L3 | Cloud Team | Cloud provider support | If infrastructure issue |

---

## CriticalCPUUsage

### Alert Details

| Property | Value |
|----------|-------|
| **Alert Name** | CriticalCPUUsage |
| **Severity** | Critical |
| **Team** | Platform |
| **Evaluation Interval** | 60s |
| **For Duration** | 5m |
| **Threshold** | 95% |

**PromQL Expression:**

```promql
(
  100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)
) > 95
```

### Description

This alert fires when CPU utilization exceeds 95% for 5 minutes. This is a critical condition - the node is saturated and likely causing performance degradation.

### Impact Assessment

| Impact Area | Severity | Description |
|-------------|----------|-------------|
| **User Impact** | High | Significant latency increase likely |
| **Data Impact** | Low | Requests may timeout |
| **SLA Impact** | High | Directly impacts latency SLA |
| **Revenue Impact** | Medium | Degraded user experience |

### Diagnostic Steps

Follow [HighCPUUsage diagnostic steps](#diagnostic-steps) with urgency.

### Resolution Steps

**Immediate Actions Required:**

```bash
# 1. Identify top CPU consumers immediately
kubectl top pods --all-namespaces --sort-by=cpu | head -10

# 2. Check for runaway processes
kubectl exec -n greenlang deploy/<top-consumer> -- top -b -n1

# 3. Kill or restart problematic pods
kubectl delete pod -n <namespace> <pod-name>

# 4. Scale out immediately
kubectl scale deployment -n greenlang <deployment> --replicas=10

# 5. If node is unresponsive, drain and replace
kubectl drain {{ $labels.instance }} --ignore-daemonsets --delete-emptydir-data --force

# 6. Add emergency capacity
# AWS:
aws autoscaling set-desired-capacity \
  --auto-scaling-group-name greenlang-nodes \
  --desired-capacity 15
```

### Escalation Path

| Level | Team | Contact | When to Escalate |
|-------|------|---------|------------------|
| L1 | On-Call Engineer | PagerDuty | Immediate response |
| L2 | Platform Team | Direct page | If not resolved in 5 minutes |
| L3 | Engineering Manager | Phone call | If SLA breach imminent |

---

## HighMemoryUsage

### Alert Details

| Property | Value |
|----------|-------|
| **Alert Name** | HighMemoryUsage |
| **Severity** | Warning |
| **Team** | Platform |
| **Evaluation Interval** | 60s |
| **For Duration** | 10m |
| **Threshold** | 85% |

**PromQL Expression:**

```promql
(
  (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes)
  / node_memory_MemTotal_bytes
) * 100 > 85
```

### Description

This alert fires when memory utilization exceeds 85% for 10 minutes. High memory usage can lead to OOM kills and service instability.

### Impact Assessment

| Impact Area | Severity | Description |
|-------------|----------|-------------|
| **User Impact** | Low | No immediate impact |
| **Data Impact** | Medium | Risk of OOM killing pods |
| **SLA Impact** | Low | Warning indicator |
| **Revenue Impact** | Low | Early warning |

### Diagnostic Steps

1. **Identify memory consumers**

   ```bash
   # Top memory-consuming pods
   kubectl top pods --all-namespaces --sort-by=memory | head -20

   # Node-level memory
   kubectl debug node/{{ $labels.instance }} -it --image=busybox -- free -m
   ```

2. **Check container memory usage**

   ```bash
   # Query container memory
   curl -s "http://prometheus:9090/api/v1/query?query=sum(container_memory_usage_bytes{instance='{{ $labels.instance }}'})by(pod)/1024/1024" | jq '.data.result | sort_by(.value[1] | tonumber) | reverse | .[0:10]'
   ```

3. **Check for memory leaks**

   ```bash
   # Memory usage trend over time
   curl -s "http://prometheus:9090/api/v1/query_range?query=sum(container_memory_usage_bytes{pod=~'.*'})by(pod)&start=$(date -d '24 hours ago' +%s)&end=$(date +%s)&step=3600" | jq .
   ```

4. **Check OOM events**

   ```bash
   # Recent OOM kills
   kubectl get events --all-namespaces --field-selector reason=OOMKilled
   ```

5. **Check node memory breakdown**

   ```bash
   # Detailed memory info
   kubectl debug node/{{ $labels.instance }} -it --image=busybox -- cat /proc/meminfo
   ```

### Resolution Steps

#### Scenario 1: Single pod with high memory

```bash
# 1. Identify the pod
kubectl top pods --all-namespaces --sort-by=memory | head -5

# 2. Check if memory leak
kubectl logs -n {{ namespace }} {{ pod }} | grep -i "memory\|heap\|gc"

# 3. Restart the pod to reclaim memory
kubectl delete pod -n {{ namespace }} {{ pod }}

# 4. Set memory limits to prevent future issues
kubectl patch deployment -n {{ namespace }} {{ deployment }} \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"app","resources":{"limits":{"memory":"4Gi"},"requests":{"memory":"2Gi"}}}]}}}}'
```

#### Scenario 2: Cache consuming too much memory

```bash
# 1. Check Redis/cache memory
kubectl exec -n greenlang deploy/redis -- redis-cli info memory

# 2. If Redis memory high, flush old keys
kubectl exec -n greenlang deploy/redis -- redis-cli --scan --pattern '*:old:*' | \
  xargs redis-cli del

# 3. Set maxmemory policy
kubectl exec -n greenlang deploy/redis -- \
  redis-cli config set maxmemory-policy allkeys-lru
```

#### Scenario 3: Application memory leak

```bash
# 1. Enable memory profiling
kubectl set env deployment/<deployment> -n greenlang \
  PYTHONTRACEMALLOC=1

# 2. Get memory snapshot
kubectl exec -n greenlang deploy/<deployment> -- \
  python -c "import tracemalloc; tracemalloc.start(); # your profiling code"

# 3. Schedule periodic restarts until fix deployed
kubectl patch deployment -n greenlang <deployment> \
  -p '{"spec":{"template":{"metadata":{"annotations":{"kubectl.kubernetes.io/restartedAt":"'$(date -Iseconds)'"}}}}}'
```

#### Scenario 4: Node overprovisioned

```bash
# 1. Check total memory requests vs capacity
kubectl describe node {{ $labels.instance }} | grep -A10 "Allocated"

# 2. Evict lower-priority workloads
kubectl cordon {{ $labels.instance }}
kubectl delete pods -n <namespace> -l priority=low

# 3. Add more nodes to the cluster
```

### Escalation Path

| Level | Team | Contact | When to Escalate |
|-------|------|---------|------------------|
| L1 | On-Call Engineer | PagerDuty | Initial response |
| L2 | Platform Team | #platform-oncall Slack | If approaching 95% |
| L3 | Backend Team | #backend-oncall Slack | If memory leak confirmed |

---

## CriticalMemoryUsage

### Alert Details

| Property | Value |
|----------|-------|
| **Alert Name** | CriticalMemoryUsage |
| **Severity** | Critical |
| **Team** | Platform |
| **Evaluation Interval** | 60s |
| **For Duration** | 5m |
| **Threshold** | 95% |

**PromQL Expression:**

```promql
(
  (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes)
  / node_memory_MemTotal_bytes
) * 100 > 95
```

### Description

This alert fires when memory utilization exceeds 95% for 5 minutes. The node is at risk of OOM kills which can cause cascading failures.

### Impact Assessment

| Impact Area | Severity | Description |
|-------------|----------|-------------|
| **User Impact** | High | OOM kills causing service disruption |
| **Data Impact** | Medium | In-flight requests may be lost |
| **SLA Impact** | High | Availability impact from pod restarts |
| **Revenue Impact** | High | Service instability |

### Resolution Steps

**Immediate Actions Required:**

```bash
# 1. Identify biggest memory consumers
kubectl top pods --all-namespaces --sort-by=memory | head -10

# 2. Kill non-essential pods to free memory
kubectl delete pods -n greenlang -l priority=low

# 3. Evict pods from the node
kubectl drain {{ $labels.instance }} --ignore-daemonsets --delete-emptydir-data

# 4. Clear system caches (if SSH access)
kubectl debug node/{{ $labels.instance }} -it --image=busybox -- \
  sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"

# 5. Scale down non-critical workloads
kubectl scale deployment -n greenlang <non-critical-app> --replicas=0

# 6. Add emergency node capacity
```

### Escalation Path

| Level | Team | Contact | When to Escalate |
|-------|------|---------|------------------|
| L1 | On-Call Engineer | PagerDuty | Immediate response |
| L2 | Platform Team | Direct page | If OOM kills occurring |
| L3 | Engineering Manager | Phone call | If service disruption |

---

## DiskSpaceLow

### Alert Details

| Property | Value |
|----------|-------|
| **Alert Name** | DiskSpaceLow |
| **Severity** | Warning |
| **Team** | Platform |
| **Evaluation Interval** | 60s |
| **For Duration** | 15m |
| **Threshold** | 80% |

**PromQL Expression:**

```promql
(
  (node_filesystem_size_bytes{mountpoint="/"} - node_filesystem_avail_bytes{mountpoint="/"})
  / node_filesystem_size_bytes{mountpoint="/"}
) * 100 > 80
```

### Description

This alert fires when root filesystem usage exceeds 80%. Low disk space can cause service failures, failed deployments, and data loss.

### Impact Assessment

| Impact Area | Severity | Description |
|-------------|----------|-------------|
| **User Impact** | Low | No immediate impact |
| **Data Impact** | High | Risk of data write failures |
| **SLA Impact** | Medium | May prevent deployments |
| **Revenue Impact** | Medium | Risk of service degradation |

### Diagnostic Steps

1. **Check disk usage breakdown**

   ```bash
   # Get disk usage on node
   kubectl debug node/{{ $labels.instance }} -it --image=busybox -- df -h

   # Find large directories
   kubectl debug node/{{ $labels.instance }} -it --image=busybox -- \
     du -sh /* 2>/dev/null | sort -rh | head -20
   ```

2. **Check container disk usage**

   ```bash
   # Container filesystem usage
   curl -s "http://prometheus:9090/api/v1/query?query=container_fs_usage_bytes{instance='{{ $labels.instance }}'}" | jq '.data.result | sort_by(.value[1] | tonumber) | reverse | .[0:10]'
   ```

3. **Check log file sizes**

   ```bash
   # Find large log files
   kubectl debug node/{{ $labels.instance }} -it --image=busybox -- \
     find /var/log -type f -size +100M -exec ls -lh {} \;
   ```

4. **Check Docker/containerd images**

   ```bash
   # Unused images
   kubectl debug node/{{ $labels.instance }} -it --image=docker -- \
     docker system df
   ```

5. **Check PersistentVolumeClaims**

   ```bash
   # PVC usage
   kubectl get pvc --all-namespaces
   kubectl describe pvc -n greenlang <pvc-name>
   ```

### Resolution Steps

#### Scenario 1: Log files consuming space

```bash
# 1. Identify large logs
kubectl debug node/{{ $labels.instance }} -it --image=busybox -- \
  find /var/log -type f -size +100M

# 2. Rotate logs
kubectl debug node/{{ $labels.instance }} -it --image=busybox -- \
  logrotate -f /etc/logrotate.conf

# 3. Clear old logs
kubectl debug node/{{ $labels.instance }} -it --image=busybox -- \
  find /var/log -type f -name "*.gz" -mtime +7 -delete

# 4. Truncate large active logs (careful - may lose logs)
kubectl debug node/{{ $labels.instance }} -it --image=busybox -- \
  truncate -s 0 /var/log/large.log
```

#### Scenario 2: Docker images consuming space

```bash
# 1. Clean up unused Docker resources
kubectl debug node/{{ $labels.instance }} -it --image=docker -- \
  docker system prune -a -f

# 2. Remove old images
kubectl debug node/{{ $labels.instance }} -it --image=docker -- \
  docker image prune -a --filter "until=168h" -f

# 3. Clean up containerd (if using containerd)
kubectl debug node/{{ $labels.instance }} -it --image=busybox -- \
  crictl rmi --prune
```

#### Scenario 3: Application data consuming space

```bash
# 1. Identify application data directories
kubectl debug node/{{ $labels.instance }} -it --image=busybox -- \
  du -sh /var/lib/kubelet/pods/* | sort -rh | head

# 2. Check for failed pods leaving data
kubectl get pods --all-namespaces --field-selector=status.phase=Failed

# 3. Clean up terminated pods
kubectl delete pods --all-namespaces --field-selector=status.phase=Failed
kubectl delete pods --all-namespaces --field-selector=status.phase=Succeeded
```

#### Scenario 4: Persistent Volume full

```bash
# 1. Check PV usage
kubectl exec -n greenlang deploy/<app> -- df -h

# 2. Clean up old data in PV
kubectl exec -n greenlang deploy/<app> -- \
  find /data -type f -mtime +30 -delete

# 3. Resize PV if needed (depends on storage class)
kubectl patch pvc -n greenlang <pvc-name> \
  -p '{"spec":{"resources":{"requests":{"storage":"100Gi"}}}}'
```

### Preventive Measures

```bash
# 1. Set up log rotation
kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: logrotate-config
  namespace: greenlang
data:
  logrotate.conf: |
    /var/log/*.log {
      daily
      rotate 7
      compress
      delaycompress
      missingok
      notifempty
    }
EOF

# 2. Configure container log limits
# Add to kubelet config:
# containerLogMaxSize: "50Mi"
# containerLogMaxFiles: 3

# 3. Set up automated cleanup CronJob
kubectl apply -f k8s/jobs/disk-cleanup-cronjob.yaml
```

### Escalation Path

| Level | Team | Contact | When to Escalate |
|-------|------|---------|------------------|
| L1 | On-Call Engineer | PagerDuty | Initial response |
| L2 | Platform Team | #platform-oncall Slack | If >90% usage |
| L3 | Storage Team | #storage Slack | If PV resize needed |

---

## Quick Reference Card

| Alert | Severity | Threshold | First Check | Quick Fix |
|-------|----------|-----------|-------------|-----------|
| HighCPUUsage | Warning | >80% | `kubectl top pods` | Scale deployment |
| CriticalCPUUsage | Critical | >95% | Top CPU consumers | Kill/restart pods |
| HighMemoryUsage | Warning | >85% | `kubectl top pods` | Restart memory hogs |
| CriticalMemoryUsage | Critical | >95% | Memory consumers | Drain node |
| DiskSpaceLow | Warning | >80% | `df -h` | Clean logs/images |

## Resource Management Best Practices

1. **Set resource requests and limits** for all containers
2. **Configure HPA** for auto-scaling based on CPU/memory
3. **Implement log rotation** and retention policies
4. **Schedule regular cleanup** of old images and data
5. **Monitor trends** to predict capacity needs
6. **Use node affinity** to spread workloads
7. **Implement pod disruption budgets** for graceful degradation
