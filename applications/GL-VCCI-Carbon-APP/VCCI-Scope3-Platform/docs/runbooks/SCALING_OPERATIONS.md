# Scaling Operations Runbook

**Scenario**: Scale platform resources in response to load changes, growth, or performance requirements through horizontal pod autoscaling (HPA), vertical scaling, cluster expansion, and database read replica management.

**Severity**: P2 (Planned) / P1 (Emergency Scaling)

**RTO/RPO**: N/A (Operational procedure)

**Owner**: Platform Team / SRE

## Prerequisites

- kubectl access to EKS cluster
- AWS Console access (EKS, RDS, EC2)
- Grafana/Prometheus access for metrics
- Helm 3.x installed
- Understanding of current capacity baselines

## Detection

### Indicators for Scaling Need

1. **High CPU/Memory Utilization**:
   - Pod CPU > 70% sustained
   - Pod Memory > 80% sustained
   - Node CPU > 75% sustained

2. **Increased Latency**:
   - P95 API response time > 2 seconds
   - Database query time > 500ms
   - Queue processing lag > 5 minutes

3. **Growing Request Volume**:
   - Requests per second increased > 50%
   - Active user count trending upward
   - Data ingestion volume doubled

4. **Resource Constraints**:
   - Pods in "Pending" state due to insufficient resources
   - HPA at max replica count
   - Throttling errors in logs

### Check Current Resource Status

```bash
# Check pod resource utilization
kubectl top pods -n vcci-scope3 --sort-by=memory

# Check node resource utilization
kubectl top nodes

# Check HPA status
kubectl get hpa -n vcci-scope3 -o wide

# Check pod status
kubectl get pods -n vcci-scope3 -o wide | grep -v "Running\|Completed"
```

**Expected Output**:
```
NAME                              CPU(cores)   MEMORY(bytes)
api-gateway-7d9f8b6c5d-abc12      245m         512Mi
calculation-engine-5f6g7h8i9-def  890m         1.2Gi
```

## Step-by-Step Procedure

### Part 1: Horizontal Pod Autoscaling (HPA)

#### Step 1: Review Current HPA Configuration

```bash
# List all HPA resources
kubectl get hpa -n vcci-scope3

# Detailed view of specific HPA
kubectl describe hpa api-gateway-hpa -n vcci-scope3

# Check HPA metrics
kubectl get hpa api-gateway-hpa -n vcci-scope3 -o yaml
```

**Sample HPA Configuration**:
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-gateway-hpa
  namespace: vcci-scope3
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-gateway
  minReplicas: 3
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
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
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

#### Step 2: Adjust HPA Thresholds (If Needed)

```bash
# Option A: Using kubectl patch (quick adjustment)
kubectl patch hpa api-gateway-hpa -n vcci-scope3 --patch '
spec:
  maxReplicas: 30
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 60
'

# Option B: Edit HPA directly
kubectl edit hpa api-gateway-hpa -n vcci-scope3

# Verify changes
kubectl describe hpa api-gateway-hpa -n vcci-scope3 | grep -A 10 "Metrics"
```

**Common Adjustments**:
- Increase `maxReplicas` for higher capacity ceiling
- Lower `averageUtilization` for more aggressive scaling
- Adjust `stabilizationWindowSeconds` to prevent flapping

#### Step 3: Monitor HPA Behavior

```bash
# Watch HPA in real-time
watch -n 5 'kubectl get hpa -n vcci-scope3'

# Check HPA events
kubectl describe hpa api-gateway-hpa -n vcci-scope3 | tail -20

# View scaling decisions
kubectl get events -n vcci-scope3 --field-selector involvedObject.kind=HorizontalPodAutoscaler --sort-by='.lastTimestamp'

# Query Prometheus for metric data
curl -s 'http://prometheus:9090/api/v1/query?query=kube_horizontalpodautoscaler_status_current_replicas{namespace="vcci-scope3"}' | jq
```

#### Step 4: Create Custom Metric HPA (Advanced)

```yaml
# For RPS-based scaling
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-gateway-rps-hpa
  namespace: vcci-scope3
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-gateway
  minReplicas: 3
  maxReplicas: 50
  metrics:
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "1000"
  - type: External
    external:
      metric:
        name: sqs_queue_depth
        selector:
          matchLabels:
            queue_name: scope3-calculations
      target:
        type: AverageValue
        averageValue: "100"
```

```bash
# Apply custom HPA
kubectl apply -f custom-hpa.yaml

# Verify custom metrics are available
kubectl get --raw /apis/custom.metrics.k8s.io/v1beta1 | jq
```

### Part 2: Manual Pod Scaling

#### Step 5: Emergency Manual Scaling

**When to Use**: HPA not responding fast enough, immediate capacity needed

```bash
# Scale deployment immediately
kubectl scale deployment api-gateway -n vcci-scope3 --replicas=15

# Scale multiple deployments
for deployment in api-gateway calculation-engine data-ingestion; do
  kubectl scale deployment/$deployment -n vcci-scope3 --replicas=10
  echo "Scaled $deployment to 10 replicas"
done

# Verify scaling progress
kubectl get deployments -n vcci-scope3 -o wide

# Watch pod rollout
kubectl get pods -n vcci-scope3 -l app=api-gateway -w
```

**Expected Timeline**:
- **T+0s**: Scale command issued
- **T+5-15s**: New pods created
- **T+30-60s**: Pods running and ready
- **T+60-120s**: Pods receiving traffic

#### Step 6: Scale Down Safely

```bash
# Before scaling down, check current load
kubectl top pods -n vcci-scope3 -l app=api-gateway

# Gradual scale down (reduce by 25%)
CURRENT_REPLICAS=$(kubectl get deployment api-gateway -n vcci-scope3 -o jsonpath='{.spec.replicas}')
NEW_REPLICAS=$(( CURRENT_REPLICAS * 75 / 100 ))
kubectl scale deployment api-gateway -n vcci-scope3 --replicas=$NEW_REPLICAS

# Monitor for errors during scale down
kubectl logs -n vcci-scope3 -l app=api-gateway --tail=100 | grep -i "error\|connection refused"

# Check if pods are being terminated gracefully
kubectl get events -n vcci-scope3 | grep -i "killing\|terminated"
```

**Scale Down Best Practices**:
- Scale down during low-traffic periods
- Reduce gradually (25-50% at a time)
- Monitor error rates for 10-15 minutes between reductions
- Ensure graceful shutdown configured (terminationGracePeriodSeconds)

### Part 3: Cluster Autoscaling

#### Step 7: Configure Cluster Autoscaler

```bash
# Check current cluster autoscaler configuration
kubectl describe deployment cluster-autoscaler -n kube-system

# View autoscaler logs
kubectl logs -n kube-system deployment/cluster-autoscaler --tail=50

# Check autoscaler status
kubectl get configmap cluster-autoscaler-status -n kube-system -o yaml
```

**Cluster Autoscaler ConfigMap**:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: cluster-autoscaler-priority-expander
  namespace: kube-system
data:
  priorities: |
    10:
      - .*-spot-.*
    50:
      - .*-on-demand-.*
```

#### Step 8: Add/Resize Node Groups

```bash
# List current node groups
aws eks list-nodegroups --cluster-name vcci-scope3-prod

# Describe node group
aws eks describe-nodegroup \
  --cluster-name vcci-scope3-prod \
  --nodegroup-name vcci-scope3-workers-prod \
  --query 'nodegroup.{DesiredSize:scalingConfig.desiredSize,MinSize:scalingConfig.minSize,MaxSize:scalingConfig.maxSize,InstanceTypes:instanceTypes}' \
  --output table

# Update node group size
aws eks update-nodegroup-config \
  --cluster-name vcci-scope3-prod \
  --nodegroup-name vcci-scope3-workers-prod \
  --scaling-config minSize=3,maxSize=20,desiredSize=6

# Monitor node provisioning
watch -n 10 'kubectl get nodes -o wide'

# Check new nodes are ready
kubectl get nodes -l eks.amazonaws.com/nodegroup=vcci-scope3-workers-prod
```

**Expected Timeline**:
- **T+0**: Update command issued
- **T+2-5 min**: New EC2 instances launching
- **T+5-8 min**: Nodes joining cluster
- **T+8-10 min**: Nodes ready for scheduling

#### Step 9: Add New Node Group (Different Instance Type)

```bash
# Create node group for memory-intensive workloads
aws eks create-nodegroup \
  --cluster-name vcci-scope3-prod \
  --nodegroup-name vcci-scope3-memory-optimized \
  --scaling-config minSize=2,maxSize=10,desiredSize=3 \
  --subnets subnet-abc123 subnet-def456 subnet-ghi789 \
  --instance-types r6i.2xlarge \
  --node-role arn:aws:iam::123456789012:role/VCCINodeInstanceRole \
  --labels workload-type=memory-intensive \
  --taints key=workload-type,value=memory-intensive,effect=NoSchedule \
  --tags "Environment=production,Application=vcci-scope3"

# Wait for node group to be active
aws eks wait nodegroup-active \
  --cluster-name vcci-scope3-prod \
  --nodegroup-name vcci-scope3-memory-optimized

# Verify nodes
kubectl get nodes -l workload-type=memory-intensive
```

#### Step 10: Update Pod Node Affinity

```bash
# Update deployment to use new node group
kubectl patch deployment calculation-engine -n vcci-scope3 --patch '
spec:
  template:
    spec:
      nodeSelector:
        workload-type: memory-intensive
      tolerations:
      - key: workload-type
        operator: Equal
        value: memory-intensive
        effect: NoSchedule
'

# Verify pods are rescheduled
kubectl get pods -n vcci-scope3 -l app=calculation-engine -o wide
```

### Part 4: Vertical Pod Scaling

#### Step 11: Analyze Resource Usage Patterns

```bash
# Check historical resource usage
kubectl top pods -n vcci-scope3 --containers --sort-by=memory

# Query Prometheus for detailed metrics
curl -s 'http://prometheus:9090/api/v1/query?query=max_over_time(container_memory_working_set_bytes{namespace="vcci-scope3",pod=~"api-gateway.*"}[7d])' | jq

# Get resource requests vs actual usage
kubectl describe pod -n vcci-scope3 -l app=api-gateway | grep -A 5 "Limits\|Requests"
```

#### Step 12: Adjust Resource Requests/Limits

```bash
# Update deployment with new resource specifications
kubectl patch deployment api-gateway -n vcci-scope3 --patch '
spec:
  template:
    spec:
      containers:
      - name: api-gateway
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
'

# Verify rollout
kubectl rollout status deployment/api-gateway -n vcci-scope3

# Monitor new pods
kubectl top pods -n vcci-scope3 -l app=api-gateway
```

**Resource Sizing Guidelines**:
- **Requests**: Set to P90 actual usage
- **Limits**: Set to 1.5-2x requests (allow for spikes)
- **CPU**: More conservative (easier to scale horizontally)
- **Memory**: More headroom (OOMKills are disruptive)

#### Step 13: Install Vertical Pod Autoscaler (VPA)

```bash
# Install VPA
git clone https://github.com/kubernetes/autoscaler.git
cd autoscaler/vertical-pod-autoscaler
./hack/vpa-up.sh

# Create VPA for a deployment
cat <<EOF | kubectl apply -f -
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: data-ingestion-vpa
  namespace: vcci-scope3
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: data-ingestion
  updatePolicy:
    updateMode: "Recreate"  # or "Auto" for automatic updates
  resourcePolicy:
    containerPolicies:
    - containerName: data-ingestion
      minAllowed:
        cpu: 100m
        memory: 256Mi
      maxAllowed:
        cpu: 2
        memory: 4Gi
EOF

# Check VPA recommendations
kubectl describe vpa data-ingestion-vpa -n vcci-scope3
```

### Part 5: Database Read Replica Scaling

#### Step 14: Add RDS Read Replicas

```bash
# Create read replica
aws rds create-db-instance-read-replica \
  --db-instance-identifier vcci-scope3-prod-read-replica-1 \
  --source-db-instance-identifier vcci-scope3-prod-postgres \
  --db-instance-class db.r6g.xlarge \
  --availability-zone us-west-2b \
  --publicly-accessible false \
  --tags Key=Environment,Value=production Key=Application,Value=vcci-scope3

# Monitor replica creation
watch -n 30 'aws rds describe-db-instances \
  --db-instance-identifier vcci-scope3-prod-read-replica-1 \
  --query "DBInstances[0].DBInstanceStatus" \
  --output text'

# Wait for replica to be available (typically 10-20 minutes)
aws rds wait db-instance-available \
  --db-instance-identifier vcci-scope3-prod-read-replica-1

# Get replica endpoint
REPLICA_ENDPOINT=$(aws rds describe-db-instances \
  --db-instance-identifier vcci-scope3-prod-read-replica-1 \
  --query 'DBInstances[0].Endpoint.Address' \
  --output text)

echo "Read Replica Endpoint: $REPLICA_ENDPOINT"
```

#### Step 15: Configure Application for Read Replicas

```bash
# Update application ConfigMap with read replica endpoints
kubectl patch configmap database-config -n vcci-scope3 --patch "
data:
  DB_READ_REPLICAS: \"$REPLICA_ENDPOINT,vcci-scope3-prod-read-replica-2.abc.us-west-2.rds.amazonaws.com\"
"

# Restart pods to pick up new configuration
kubectl rollout restart deployment/reporting-service -n vcci-scope3
kubectl rollout restart deployment/api-gateway -n vcci-scope3

# Verify read traffic distribution
kubectl logs -n vcci-scope3 deployment/reporting-service | grep "database_host" | tail -20
```

#### Step 16: Monitor Read Replica Lag

```bash
# Check replication lag
aws cloudwatch get-metric-statistics \
  --namespace AWS/RDS \
  --metric-name ReplicaLag \
  --dimensions Name=DBInstanceIdentifier,Value=vcci-scope3-prod-read-replica-1 \
  --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 300 \
  --statistics Average,Maximum \
  --output table

# Query replica directly
psql -h $REPLICA_ENDPOINT -U vcci_admin -d scope3_platform << EOF
SELECT
  now() - pg_last_xact_replay_timestamp() AS replication_lag,
  pg_is_in_recovery() AS is_replica;
EOF
```

**Acceptable Lag**: < 30 seconds for reporting queries

### Part 6: Redis Cache Scaling

#### Step 17: Scale ElastiCache Cluster

```bash
# Describe current cluster
aws elasticache describe-cache-clusters \
  --cache-cluster-id vcci-scope3-redis-prod \
  --show-cache-node-info \
  --query 'CacheClusters[0].{NodeType:CacheNodeType,NumNodes:NumCacheNodes,Status:CacheClusterStatus}' \
  --output table

# Scale up (add nodes to cluster)
aws elasticache modify-cache-cluster \
  --cache-cluster-id vcci-scope3-redis-prod \
  --num-cache-nodes 5 \
  --apply-immediately

# Or modify node type (requires snapshot/restore)
aws elasticache modify-cache-cluster \
  --cache-cluster-id vcci-scope3-redis-prod \
  --cache-node-type cache.r6g.xlarge \
  --apply-immediately

# Monitor modification progress
watch -n 30 'aws elasticache describe-cache-clusters \
  --cache-cluster-id vcci-scope3-redis-prod \
  --query "CacheClusters[0].CacheClusterStatus" \
  --output text'
```

## Validation

### Post-Scaling Validation Checklist

```bash
# 1. All pods running and ready
kubectl get pods -n vcci-scope3 -o wide | grep -v "Running.*1/1\|Running.*2/2"

# 2. HPA within target ranges
kubectl get hpa -n vcci-scope3 -o custom-columns=NAME:.metadata.name,CURRENT:.status.currentReplicas,MIN:.spec.minReplicas,MAX:.spec.maxReplicas,CPU:.status.currentMetrics[0].resource.current.averageUtilization

# 3. Node capacity available
kubectl describe nodes | grep -A 5 "Allocated resources"

# 4. No pending pods
kubectl get pods --all-namespaces --field-selector status.phase=Pending

# 5. API response times acceptable
curl -w "@curl-format.txt" -o /dev/null -s https://api.vcci-scope3.com/health

# 6. Database connections healthy
kubectl run -it --rm db-check \
  --image=postgres:14 \
  --restart=Never \
  -- psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform -c "\
    SELECT count(*) as active_connections FROM pg_stat_activity WHERE state = 'active';"

# 7. Error rate normal
kubectl logs -n vcci-scope3 -l app=api-gateway --since=10m | grep -c "ERROR"

# 8. Resource utilization in healthy range
kubectl top nodes
kubectl top pods -n vcci-scope3 --sort-by=cpu
```

### Performance Testing After Scaling

```bash
# Load test with increased capacity
kubectl run -it --rm load-test \
  --image=williamyeh/wrk \
  --restart=Never \
  -- -t12 -c400 -d60s --latency https://api.vcci-scope3.com/api/v1/emissions/calculate

# Monitor during load test
watch -n 2 'kubectl top pods -n vcci-scope3'

# Check autoscaling response
kubectl get hpa -n vcci-scope3 -w
```

## Troubleshooting

### Issue 1: Pods Stuck in Pending State

**Symptoms**: New pods not scheduling after scale-up

**Diagnosis**:
```bash
# Check why pods are pending
kubectl describe pod <pending-pod-name> -n vcci-scope3 | grep -A 10 Events

# Check node capacity
kubectl describe nodes | grep -E "Name:|Allocatable:|Allocated"

# Check for taints preventing scheduling
kubectl describe nodes | grep Taints
```

**Resolution**:
```bash
# If insufficient CPU/memory, add nodes
aws eks update-nodegroup-config \
  --cluster-name vcci-scope3-prod \
  --nodegroup-name vcci-scope3-workers-prod \
  --scaling-config desiredSize=8

# If taint issues, update tolerations
kubectl patch deployment api-gateway -n vcci-scope3 --patch '
spec:
  template:
    spec:
      tolerations:
      - key: "node.kubernetes.io/disk-pressure"
        operator: "Exists"
        effect: "NoSchedule"
'
```

### Issue 2: HPA Not Scaling

**Symptoms**: HPA shows "unknown" for metrics or not scaling despite high load

**Diagnosis**:
```bash
# Check metrics server
kubectl get deployment metrics-server -n kube-system

# Check if metrics are available
kubectl get --raw /apis/metrics.k8s.io/v1beta1/nodes
kubectl get --raw /apis/metrics.k8s.io/v1beta1/pods

# Check HPA status
kubectl describe hpa <hpa-name> -n vcci-scope3
```

**Resolution**:
```bash
# Restart metrics server
kubectl rollout restart deployment metrics-server -n kube-system

# Verify resource requests are set (required for HPA)
kubectl get deployment api-gateway -n vcci-scope3 -o jsonpath='{.spec.template.spec.containers[0].resources}'

# Check HPA calculation
kubectl get hpa <hpa-name> -n vcci-scope3 -o yaml | grep -A 20 status
```

### Issue 3: Node Group Not Scaling

**Symptoms**: Cluster autoscaler not adding nodes despite pending pods

**Diagnosis**:
```bash
# Check cluster autoscaler logs
kubectl logs -n kube-system deployment/cluster-autoscaler --tail=100

# Check node group limits
aws eks describe-nodegroup \
  --cluster-name vcci-scope3-prod \
  --nodegroup-name vcci-scope3-workers-prod \
  --query 'nodegroup.scalingConfig'
```

**Resolution**:
```bash
# Increase max size
aws eks update-nodegroup-config \
  --cluster-name vcci-scope3-prod \
  --nodegroup-name vcci-scope3-workers-prod \
  --scaling-config maxSize=30

# Check for AWS service limits
aws service-quotas get-service-quota \
  --service-code ec2 \
  --quota-code L-1216C47A  # Running On-Demand Standard instances

# Review cluster autoscaler config
kubectl edit deployment cluster-autoscaler -n kube-system
# Verify --max-nodes-total and --max-cores-total flags
```

### Issue 4: High Read Replica Lag

**Symptoms**: Replication lag > 60 seconds

**Diagnosis**:
```bash
# Check replica lag metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/RDS \
  --metric-name ReplicaLag \
  --dimensions Name=DBInstanceIdentifier,Value=vcci-scope3-prod-read-replica-1 \
  --start-time $(date -u -d '30 minutes ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 60 \
  --statistics Maximum \
  --output table

# Check for high write activity on primary
aws cloudwatch get-metric-statistics \
  --namespace AWS/RDS \
  --metric-name WriteIOPS \
  --dimensions Name=DBInstanceIdentifier,Value=vcci-scope3-prod-postgres \
  --start-time $(date -u -d '30 minutes ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 300 \
  --statistics Average \
  --output table
```

**Resolution**:
```bash
# Upgrade replica instance class
aws rds modify-db-instance \
  --db-instance-identifier vcci-scope3-prod-read-replica-1 \
  --db-instance-class db.r6g.2xlarge \
  --apply-immediately

# Temporarily reduce read traffic to lagging replica
# Update load balancer weights or remove from rotation
```

## Related Documentation

- [Performance Tuning Runbook](./PERFORMANCE_TUNING.md)
- [Capacity Planning Runbook](./CAPACITY_PLANNING.md)
- [Database Failover Runbook](./DATABASE_FAILOVER.md)
- [Kubernetes HPA Documentation](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- [AWS EKS Best Practices - Autoscaling](https://aws.github.io/aws-eks-best-practices/cluster-autoscaling/)
- [RDS Read Replicas](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/USER_ReadRepl.html)

## Appendix: Scaling Decision Matrix

| Metric | Threshold | Action | Priority |
|--------|-----------|--------|----------|
| Pod CPU > 70% | Sustained 5 min | Increase HPA max or adjust threshold | P2 |
| Pod Memory > 85% | Sustained 5 min | Vertical scaling or add replicas | P1 |
| Node CPU > 80% | Sustained 10 min | Add nodes to group | P2 |
| Pending Pods > 0 | 5 minutes | Scale node group immediately | P1 |
| API P95 > 2s | Sustained 10 min | Scale API pods, check DB | P1 |
| DB Connections > 80% | Any time | Add read replicas or connection pooling | P1 |
| Queue Depth > 1000 | Sustained 15 min | Scale workers | P2 |
| Read Replica Lag > 60s | Sustained 10 min | Upgrade replica or reduce read load | P2 |

## Contact Information

- **Platform Team**: platform-team@company.com
- **SRE On-Call**: PagerDuty escalation
- **Cloud Infrastructure**: infrastructure@company.com
