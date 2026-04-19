# Capacity Planning Runbook

**Scenario**: Proactive capacity planning and resource forecasting to ensure platform can handle growth, seasonal peaks, and prevent resource constraints.

**Severity**: P2 (Proactive) / P1 (Approaching limits)

**RTO/RPO**: N/A (Planning activity)

**Owner**: Platform Team / FinOps / Capacity Planning Team

## Prerequisites

- Access to monitoring systems (Prometheus, Grafana, CloudWatch)
- AWS Cost Explorer access
- Historical usage data (minimum 3 months)
- Business growth projections
- Understanding of application architecture

## Detection

### Capacity Planning Triggers

1. **Scheduled Reviews**:
   - Quarterly capacity planning sessions
   - Monthly resource utilization reviews
   - Pre-peak season assessments

2. **Resource Alerts**:
   - Consistent utilization > 70%
   - Resource requests approaching quotas
   - Storage growth > 20% month-over-month
   - Cost increase > 25% month-over-month

3. **Business Drivers**:
   - New client onboarding (> 100 entities)
   - Product feature launches
   - Anticipated traffic increases
   - Regulatory reporting deadlines

### Current Capacity Assessment

```bash
# Check cluster resource utilization
kubectl top nodes

# Check namespace resource quotas
kubectl describe resourcequota -n vcci-scope3

# Check pod resource requests vs limits
kubectl describe nodes | grep -A 5 "Allocated resources"

# Calculate cluster capacity utilization
cat > /tmp/cluster_capacity.sh << 'EOF'
#!/bin/bash
NODES=$(kubectl get nodes -o json)

TOTAL_CPU=$(echo $NODES | jq -r '.items[].status.capacity.cpu' | awk '{sum+=$1} END {print sum}')
TOTAL_MEM=$(echo $NODES | jq -r '.items[].status.capacity.memory' | sed 's/Ki$//' | awk '{sum+=$1} END {print sum/1024/1024}')

ALLOC_CPU=$(echo $NODES | jq -r '.items[].status.allocatable.cpu' | awk '{sum+=$1} END {print sum}')
ALLOC_MEM=$(echo $NODES | jq -r '.items[].status.allocatable.memory' | sed 's/Ki$//' | awk '{sum+=$1} END {print sum/1024/1024}')

echo "=== Cluster Capacity ==="
echo "Total CPU Cores: $TOTAL_CPU"
echo "Total Memory (GB): $TOTAL_MEM"
echo "Allocatable CPU Cores: $ALLOC_CPU"
echo "Allocatable Memory (GB): $ALLOC_MEM"
EOF

chmod +x /tmp/cluster_capacity.sh
/tmp/cluster_capacity.sh
```

## Step-by-Step Procedure

### Part 1: Historical Analysis

#### Step 1: Gather Historical Metrics

```bash
# Create data collection directory
mkdir -p /tmp/capacity_planning_$(date +%Y%m%d)
cd /tmp/capacity_planning_$(date +%Y%m%d)

# Export Prometheus metrics (last 90 days)
PROM_URL="http://prometheus:9090"

# CPU utilization over time
curl -s "${PROM_URL}/api/v1/query_range?query=avg(rate(container_cpu_usage_seconds_total{namespace='vcci-scope3'}[5m]))&start=$(date -d '90 days ago' +%s)&end=$(date +%s)&step=3600" | \
  jq -r '.data.result[0].values[] | @csv' > cpu_utilization_90d.csv

# Memory utilization over time
curl -s "${PROM_URL}/api/v1/query_range?query=avg(container_memory_working_set_bytes{namespace='vcci-scope3'})&start=$(date -d '90 days ago' +%s)&end=$(date +%s)&step=3600" | \
  jq -r '.data.result[0].values[] | @csv' > memory_utilization_90d.csv

# Request count over time
curl -s "${PROM_URL}/api/v1/query_range?query=sum(rate(http_requests_total{namespace='vcci-scope3'}[5m]))&start=$(date -d '90 days ago' +%s)&end=$(date +%s)&step=3600" | \
  jq -r '.data.result[0].values[] | @csv' > request_rate_90d.csv

# Database size growth
aws cloudwatch get-metric-statistics \
  --namespace AWS/RDS \
  --metric-name FreeStorageSpace \
  --dimensions Name=DBInstanceIdentifier,Value=vcci-scope3-prod-postgres \
  --start-time $(date -u -d '90 days ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 86400 \
  --statistics Average \
  --output json | jq -r '.Datapoints[] | [.Timestamp, .Average] | @csv' > db_storage_90d.csv

# S3 storage growth
aws cloudwatch get-metric-statistics \
  --namespace AWS/S3 \
  --metric-name BucketSizeBytes \
  --dimensions Name=BucketName,Value=vcci-scope3-data-prod Name=StorageType,Value=StandardStorage \
  --start-time $(date -u -d '90 days ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 86400 \
  --statistics Average \
  --output json | jq -r '.Datapoints[] | [.Timestamp, .Average] | @csv' > s3_storage_90d.csv

echo "Historical data collected in $(pwd)"
```

#### Step 2: Analyze Growth Trends

```bash
# Create analysis script
cat > /tmp/analyze_trends.py << 'EOF'
import pandas as pd
import numpy as np
from scipy import stats
import json

# Load CPU data
cpu_df = pd.read_csv('cpu_utilization_90d.csv', names=['timestamp', 'value'])
cpu_df['timestamp'] = pd.to_datetime(cpu_df['timestamp'], unit='s')
cpu_df['value'] = cpu_df['value'].astype(float)

# Calculate trend
slope, intercept, r_value, p_value, std_err = stats.linregress(
    range(len(cpu_df)), cpu_df['value']
)

cpu_growth_rate = (slope / cpu_df['value'].mean()) * 100  # % per day

# Load memory data
mem_df = pd.read_csv('memory_utilization_90d.csv', names=['timestamp', 'value'])
mem_df['timestamp'] = pd.to_datetime(mem_df['timestamp'], unit='s')
mem_df['value'] = mem_df['value'].astype(float) / 1024 / 1024 / 1024  # Convert to GB

slope, intercept, r_value, p_value, std_err = stats.linregress(
    range(len(mem_df)), mem_df['value']
)

mem_growth_rate = (slope / mem_df['value'].mean()) * 100  # % per day

# Load request data
req_df = pd.read_csv('request_rate_90d.csv', names=['timestamp', 'value'])
req_df['timestamp'] = pd.to_datetime(req_df['timestamp'], unit='s')
req_df['value'] = req_df['value'].astype(float)

slope, intercept, r_value, p_value, std_err = stats.linregress(
    range(len(req_df)), req_df['value']
)

req_growth_rate = (slope / req_df['value'].mean()) * 100  # % per day

# Generate report
report = {
    "analysis_date": pd.Timestamp.now().isoformat(),
    "period_days": 90,
    "cpu": {
        "current_avg": round(cpu_df['value'].tail(7).mean() * 100, 2),
        "growth_rate_pct_per_day": round(cpu_growth_rate, 4),
        "growth_rate_pct_per_month": round(cpu_growth_rate * 30, 2),
        "projected_90d": round(cpu_df['value'].tail(1).values[0] * 100 + (cpu_growth_rate * 90), 2)
    },
    "memory": {
        "current_avg_gb": round(mem_df['value'].tail(7).mean(), 2),
        "growth_rate_pct_per_day": round(mem_growth_rate, 4),
        "growth_rate_pct_per_month": round(mem_growth_rate * 30, 2),
        "projected_90d_gb": round(mem_df['value'].tail(1).values[0] + (mem_growth_rate * mem_df['value'].mean() * 90 / 100), 2)
    },
    "requests": {
        "current_avg_rps": round(req_df['value'].tail(7).mean(), 2),
        "growth_rate_pct_per_day": round(req_growth_rate, 4),
        "growth_rate_pct_per_month": round(req_growth_rate * 30, 2),
        "projected_90d_rps": round(req_df['value'].tail(1).values[0] + (req_growth_rate * req_df['value'].mean() * 90 / 100), 2)
    }
}

print(json.dumps(report, indent=2))

with open('capacity_analysis.json', 'w') as f:
    json.dump(report, f, indent=2)
EOF

# Run analysis
python3 /tmp/analyze_trends.py

# View results
cat capacity_analysis.json | jq
```

#### Step 3: Identify Seasonal Patterns

```bash
# Analyze day-of-week patterns
curl -s "${PROM_URL}/api/v1/query?query=avg_over_time(rate(http_requests_total{namespace='vcci-scope3'}[24h])[90d:1d])" | \
  jq -r '.data.result[0].values[]' > daily_pattern.txt

# Analyze month-over-month growth
psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform << 'EOF'
SELECT
  DATE_TRUNC('month', created_at) AS month,
  COUNT(*) AS new_entities,
  SUM(COUNT(*)) OVER (ORDER BY DATE_TRUNC('month', created_at)) AS cumulative_entities
FROM entity_master
WHERE created_at > NOW() - INTERVAL '12 months'
GROUP BY DATE_TRUNC('month', created_at)
ORDER BY month;
EOF

# Identify peak usage periods
cat > /tmp/peak_analysis.sh << 'EOF'
#!/bin/bash
# Find peak hours
curl -s "${PROM_URL}/api/v1/query_range?query=sum(rate(http_requests_total{namespace='vcci-scope3'}[5m]))&start=$(date -d '30 days ago' +%s)&end=$(date +%s)&step=3600" | \
  jq -r '.data.result[0].values[] | "\(.[0]) \(.[1])"' | \
  awk '{
    split($1, d, "");
    hour = strftime("%H", $1);
    sum[hour] += $2;
    count[hour]++;
  }
  END {
    for (h in sum) {
      printf "Hour %02d: %.2f avg req/s\n", h, sum[h]/count[h];
    }
  }' | sort
EOF

chmod +x /tmp/peak_analysis.sh
/tmp/peak_analysis.sh
```

### Part 2: Resource Forecasting

#### Step 4: Project Future Resource Needs

```bash
# Create forecasting script
cat > /tmp/forecast_resources.py << 'EOF'
import json
import math

# Load analysis results
with open('capacity_analysis.json') as f:
    analysis = json.load(f)

# Business growth assumptions
business_growth = {
    "new_clients_per_month": 5,
    "entities_per_client": 50,
    "calculations_per_entity_per_month": 12,
    "data_per_calculation_kb": 10
}

# Current capacity
current = {
    "nodes": 6,
    "cpu_per_node": 8,
    "memory_per_node_gb": 32,
    "db_storage_gb": 500,
    "s3_storage_tb": 2
}

# Forecast for 6, 12 months
forecasts = []

for months in [3, 6, 12]:
    # Calculate projected growth
    cpu_growth = analysis['cpu']['growth_rate_pct_per_month'] * months
    mem_growth = analysis['memory']['growth_rate_pct_per_month'] * months
    req_growth = analysis['requests']['growth_rate_pct_per_month'] * months

    # Business-driven growth
    new_entities = business_growth['new_clients_per_month'] * months * business_growth['entities_per_client']
    new_calculations = new_entities * business_growth['calculations_per_entity_per_month']
    new_data_gb = (new_calculations * business_growth['data_per_calculation_kb']) / 1024 / 1024

    # Calculate required resources
    required_cpu_cores = math.ceil(current['nodes'] * current['cpu_per_node'] * (1 + cpu_growth/100))
    required_memory_gb = math.ceil(current['nodes'] * current['memory_per_node_gb'] * (1 + mem_growth/100))
    required_nodes = math.ceil(max(
        required_cpu_cores / current['cpu_per_node'],
        required_memory_gb / current['memory_per_node_gb']
    ))

    required_db_storage_gb = math.ceil(current['db_storage_gb'] * 1.5)  # 50% buffer
    required_s3_storage_tb = current['s3_storage_tb'] + (new_data_gb / 1024)

    forecast = {
        "months_ahead": months,
        "projected_growth": {
            "cpu_pct": round(cpu_growth, 2),
            "memory_pct": round(mem_growth, 2),
            "requests_pct": round(req_growth, 2)
        },
        "required_resources": {
            "nodes": required_nodes,
            "total_cpu_cores": required_cpu_cores,
            "total_memory_gb": required_memory_gb,
            "db_storage_gb": required_db_storage_gb,
            "s3_storage_tb": round(required_s3_storage_tb, 2)
        },
        "additional_resources_needed": {
            "nodes": required_nodes - current['nodes'],
            "cpu_cores": required_cpu_cores - (current['nodes'] * current['cpu_per_node']),
            "memory_gb": required_memory_gb - (current['nodes'] * current['memory_per_node_gb']),
            "db_storage_gb": required_db_storage_gb - current['db_storage_gb'],
            "s3_storage_tb": round(required_s3_storage_tb - current['s3_storage_tb'], 2)
        }
    }

    forecasts.append(forecast)

report = {
    "forecast_date": json.dumps(pd.Timestamp.now().isoformat()),
    "current_capacity": current,
    "business_assumptions": business_growth,
    "forecasts": forecasts
}

print(json.dumps(report, indent=2))

with open('capacity_forecast.json', 'w') as f:
    json.dump(report, f, indent=2)
EOF

python3 /tmp/forecast_resources.py
cat capacity_forecast.json | jq
```

#### Step 5: Calculate Cost Projections

```bash
# Get current costs
aws ce get-cost-and-usage \
  --time-period Start=$(date -d '3 months ago' +%Y-%m-%d),End=$(date +%Y-%m-%d) \
  --granularity MONTHLY \
  --metrics BlendedCost \
  --group-by Type=SERVICE \
  --filter file://cost-filter.json > current_costs.json

# Cost filter for VCCI Scope3 resources
cat > cost-filter.json << 'EOF'
{
  "Tags": {
    "Key": "Application",
    "Values": ["vcci-scope3"]
  }
}
EOF

# Calculate projected costs
cat > /tmp/cost_projection.py << 'EOF'
import json

# Load forecast
with open('capacity_forecast.json') as f:
    forecast = json.load(f)

# Current monthly costs (update with actual values)
current_monthly_costs = {
    "ec2_nodes": 1200,  # 6 nodes * $200/month
    "rds": 800,
    "s3": 100,
    "data_transfer": 300,
    "other": 200
}

total_current = sum(current_monthly_costs.values())

# Cost projections
cost_forecasts = []

for f in forecast['forecasts']:
    months = f['months_ahead']

    # Calculate new costs based on resource growth
    node_ratio = f['required_resources']['nodes'] / forecast['current_capacity']['nodes']
    db_storage_ratio = f['required_resources']['db_storage_gb'] / forecast['current_capacity']['db_storage_gb']
    s3_storage_ratio = f['required_resources']['s3_storage_tb'] / forecast['current_capacity']['s3_storage_tb']

    projected_costs = {
        "ec2_nodes": round(current_monthly_costs['ec2_nodes'] * node_ratio, 2),
        "rds": round(current_monthly_costs['rds'] * 1.2, 2),  # Assume 20% growth
        "s3": round(current_monthly_costs['s3'] * s3_storage_ratio, 2),
        "data_transfer": round(current_monthly_costs['data_transfer'] * 1.3, 2),  # Assume 30% growth
        "other": current_monthly_costs['other']
    }

    total_projected = sum(projected_costs.values())
    increase_pct = ((total_projected - total_current) / total_current) * 100

    cost_forecast = {
        "months_ahead": months,
        "monthly_costs": projected_costs,
        "total_monthly_cost": round(total_projected, 2),
        "increase_from_current": {
            "dollars": round(total_projected - total_current, 2),
            "percent": round(increase_pct, 2)
        },
        "annual_cost_at_this_level": round(total_projected * 12, 2)
    }

    cost_forecasts.append(cost_forecast)

report = {
    "current_monthly_cost": round(total_current, 2),
    "current_annual_cost": round(total_current * 12, 2),
    "projections": cost_forecasts
}

print(json.dumps(report, indent=2))

with open('cost_forecast.json', 'w') as f:
    json.dump(report, f, indent=2)
EOF

python3 /tmp/cost_projection.py
cat cost_forecast.json | jq
```

### Part 3: Quota and Limit Management

#### Step 6: Review AWS Service Quotas

```bash
# Check EC2 quotas
aws service-quotas get-service-quota \
  --service-code ec2 \
  --quota-code L-1216C47A \
  --query 'Quota.{QuotaName:QuotaName,Value:Value,UsageMetric:UsageMetric}' \
  --output table

# Check EBS volume quotas
aws service-quotas get-service-quota \
  --service-code ebs \
  --quota-code L-D18FCD1D \
  --output table

# Check RDS quotas
aws service-quotas get-service-quota \
  --service-code rds \
  --quota-code L-7B6409FD \
  --output table

# Check current usage vs quotas
cat > /tmp/quota_check.sh << 'EOF'
#!/bin/bash

echo "=== AWS Service Quota Check ==="
echo ""

# EC2 instances
RUNNING_INSTANCES=$(aws ec2 describe-instances \
  --filters Name=instance-state-name,Values=running \
  --query 'Reservations[].Instances[].[InstanceId]' \
  --output text | wc -l)

INSTANCE_QUOTA=$(aws service-quotas get-service-quota \
  --service-code ec2 \
  --quota-code L-1216C47A \
  --query 'Quota.Value' \
  --output text)

echo "EC2 Instances: $RUNNING_INSTANCES / $INSTANCE_QUOTA ($(echo "scale=1; $RUNNING_INSTANCES * 100 / $INSTANCE_QUOTA" | bc)%)"

# EBS volumes
EBS_VOLUMES=$(aws ec2 describe-volumes --query 'Volumes[].VolumeId' --output text | wc -w)
EBS_QUOTA=$(aws service-quotas get-service-quota \
  --service-code ebs \
  --quota-code L-D18FCD1D \
  --query 'Quota.Value' \
  --output text)

echo "EBS Volumes: $EBS_VOLUMES / $EBS_QUOTA ($(echo "scale=1; $EBS_VOLUMES * 100 / $EBS_QUOTA" | bc)%)"

# RDS instances
RDS_INSTANCES=$(aws rds describe-db-instances --query 'DBInstances[].DBInstanceIdentifier' --output text | wc -w)
RDS_QUOTA=$(aws service-quotas get-service-quota \
  --service-code rds \
  --quota-code L-7B6409FD \
  --query 'Quota.Value' \
  --output text)

echo "RDS Instances: $RDS_INSTANCES / $RDS_QUOTA ($(echo "scale=1; $RDS_INSTANCES * 100 / $RDS_QUOTA" | bc)%)"

EOF

chmod +x /tmp/quota_check.sh
/tmp/quota_check.sh
```

#### Step 7: Request Quota Increases (If Needed)

```bash
# Request EC2 instance quota increase
aws service-quotas request-service-quota-increase \
  --service-code ec2 \
  --quota-code L-1216C47A \
  --desired-value 100 \
  --output json

# Request EBS storage quota increase
aws service-quotas request-service-quota-increase \
  --service-code ebs \
  --quota-code L-D18FCD1D \
  --desired-value 500 \
  --output json

# Check request status
aws service-quotas list-requested-service-quota-change-history \
  --service-code ec2 \
  --output table
```

#### Step 8: Update Kubernetes Resource Quotas

```bash
# Review current namespace quotas
kubectl describe resourcequota -n vcci-scope3

# Update resource quota based on forecast
cat > /tmp/updated-quota.yaml << EOF
apiVersion: v1
kind: ResourceQuota
metadata:
  name: vcci-scope3-quota
  namespace: vcci-scope3
spec:
  hard:
    requests.cpu: "80"        # Updated from 60
    requests.memory: "160Gi"  # Updated from 120Gi
    limits.cpu: "120"         # Updated from 90
    limits.memory: "240Gi"    # Updated from 180Gi
    pods: "200"               # Updated from 150
    services: "50"
    persistentvolumeclaims: "30"
EOF

kubectl apply -f /tmp/updated-quota.yaml

# Verify quota update
kubectl describe resourcequota vcci-scope3-quota -n vcci-scope3
```

### Part 4: Implementation Planning

#### Step 9: Create Scaling Schedule

```bash
# Create phased scaling plan
cat > /tmp/scaling_schedule.md << 'EOF'
# Capacity Scaling Schedule - Q1 2024

## Phase 1: Immediate (Week 1-2)
**Target Date**: 2024-01-31

### Actions:
- [ ] Increase EKS node group max size from 10 to 15
- [ ] Update HPA max replicas for api-gateway from 10 to 15
- [ ] Increase RDS storage from 500GB to 750GB
- [ ] Request AWS quota increases

### Resources Required:
- Additional nodes: 3 x m5.2xlarge
- Estimated cost increase: $600/month

### Validation:
- [ ] Monitor utilization for 1 week
- [ ] Verify autoscaling triggers working
- [ ] Check cost tracking

## Phase 2: Near-term (Month 2)
**Target Date**: 2024-02-28

### Actions:
- [ ] Add dedicated node group for calculation workloads
- [ ] Implement Redis cluster mode for cache
- [ ] Add 2nd RDS read replica
- [ ] Upgrade primary RDS instance class

### Resources Required:
- New node group: 4 x c5.4xlarge (compute-optimized)
- Redis cluster: 3 nodes
- RDS read replica: db.r6g.xlarge
- Estimated cost increase: $1,800/month

## Phase 3: Long-term (Month 3-6)
**Target Date**: 2024-06-30

### Actions:
- [ ] Multi-region deployment (DR region)
- [ ] Implement data archival to Glacier
- [ ] Database sharding preparation
- [ ] CDN implementation for static assets

### Resources Required:
- DR region infrastructure: ~50% of primary
- Estimated cost increase: $2,500/month
EOF

cat /tmp/scaling_schedule.md
```

#### Step 10: Document Capacity Plan

```bash
# Create comprehensive capacity plan document
cat > /tmp/capacity_plan_$(date +%Y-Q%q).md << EOF
# VCCI Scope 3 Platform - Capacity Plan $(date +%Y-Q%q)

**Prepared**: $(date)
**Planning Horizon**: 12 months
**Review Cycle**: Quarterly

## Executive Summary
Based on historical growth trends and business projections, the platform will require:
- **30% increase** in compute capacity
- **50% increase** in database storage
- **40% increase** in S3 storage
- **Estimated cost increase**: \$4,900/month by end of year

## Current Capacity

### Compute
- **Nodes**: 6 x m5.2xlarge (8 vCPU, 32GB RAM each)
- **Total**: 48 vCPUs, 192GB RAM
- **Utilization**: 62% CPU, 58% Memory

### Database
- **Instance**: db.r6g.2xlarge (8 vCPUs, 64GB RAM)
- **Storage**: 500GB gp3 (currently 340GB used, 68%)
- **Read Replicas**: 1

### Storage
- **S3**: 2TB (currently 1.4TB used, 70%)
- **Backup Retention**: 35 days

## Growth Analysis

### Historical Trends (Last 90 Days)
$(cat capacity_analysis.json | jq .)

### Projected Growth
$(cat capacity_forecast.json | jq '.forecasts')

## Cost Projections
$(cat cost_forecast.json | jq .)

## Recommendations

### Immediate Actions (0-30 days)
1. Increase node group max size to accommodate auto-scaling
2. Expand RDS storage to 750GB
3. Implement more aggressive cache warming

### Short-term Actions (1-3 months)
1. Add compute-optimized node group for calculation workloads
2. Deploy additional RDS read replica
3. Implement data lifecycle policies for S3

### Long-term Actions (3-12 months)
1. Plan for multi-region architecture
2. Investigate database sharding strategy
3. Implement data archival to reduce storage costs

## Risk Assessment

### High Utilization Risks
- **Database Storage**: Will reach 80% in ~90 days
- **S3 Storage**: Will reach 80% in ~120 days
- **Mitigation**: Proactive scaling schedule implemented

### Cost Overrun Risks
- **Growth exceeding forecast**: 20% probability
- **Mitigation**: Monthly cost reviews, automated alerting

## Success Metrics
- Maintain resource utilization between 50-75%
- Zero incidents due to capacity constraints
- Cost growth < revenue growth
- Auto-scaling events handled without manual intervention

## Review Schedule
- **Monthly**: Utilization metrics review
- **Quarterly**: Full capacity planning review
- **Annual**: Architecture review and long-term planning

---
**Approved by**: [Name, Title]
**Next Review**: $(date -d '+3 months' +%Y-%m-%d)
EOF

cat /tmp/capacity_plan_$(date +%Y-Q%q).md
```

### Part 5: Monitoring and Alerting

#### Step 11: Implement Capacity Alerts

```bash
# Create capacity monitoring alerts
cat > /tmp/capacity-alerts.yaml << 'EOF'
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: capacity-planning-alerts
  namespace: monitoring
spec:
  groups:
  - name: capacity
    interval: 5m
    rules:
    - alert: HighNodeCPUUtilization
      expr: avg(instance:node_cpu_utilization:rate1m) > 0.75
      for: 30m
      labels:
        severity: warning
        team: platform
      annotations:
        summary: "Node CPU utilization consistently high"
        description: "Average node CPU utilization is {{ $value | humanizePercentage }} (threshold: 75%)"

    - alert: HighNodeMemoryUtilization
      expr: (1 - avg(node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) > 0.80
      for: 30m
      labels:
        severity: warning
        team: platform
      annotations:
        summary: "Node memory utilization consistently high"
        description: "Average node memory utilization is {{ $value | humanizePercentage }} (threshold: 80%)"

    - alert: ApproachingResourceQuota
      expr: max(namespace_resource_quota:cpu_requests:percent{namespace="vcci-scope3"}) > 0.80
      for: 15m
      labels:
        severity: warning
        team: platform
      annotations:
        summary: "Namespace approaching CPU request quota"
        description: "Namespace vcci-scope3 is using {{ $value | humanizePercentage }} of CPU quota"

    - alert: DatabaseStorageHigh
      expr: aws_rds_free_storage_space_bytes{dbinstance_identifier="vcci-scope3-prod-postgres"} / aws_rds_total_storage_space_bytes < 0.20
      for: 1h
      labels:
        severity: warning
        team: database
      annotations:
        summary: "Database storage running low"
        description: "Database has only {{ $value | humanizePercentage }} free space remaining"

    - alert: S3BucketGrowthRateHigh
      expr: predict_linear(aws_s3_bucket_size_bytes{bucket="vcci-scope3-data-prod"}[7d], 30*24*3600) > 5000000000000  # 5TB
      for: 1h
      labels:
        severity: info
        team: platform
      annotations:
        summary: "S3 bucket projected to exceed 5TB in 30 days"
        description: "Current growth rate will result in {{ $value | humanize }}B in 30 days"
EOF

kubectl apply -f /tmp/capacity-alerts.yaml
```

#### Step 12: Create Capacity Dashboard

```bash
# Create Grafana dashboard for capacity planning
cat > /tmp/capacity-dashboard.json << 'EOF'
{
  "dashboard": {
    "title": "Capacity Planning Dashboard",
    "panels": [
      {
        "title": "Node CPU Utilization Trend",
        "targets": [
          {
            "expr": "avg(instance:node_cpu_utilization:rate5m)"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Node Memory Utilization Trend",
        "targets": [
          {
            "expr": "1 - avg(node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)"
          }
        ]
      },
      {
        "title": "Resource Quota Usage",
        "targets": [
          {
            "expr": "namespace_resource_quota:cpu_requests:percent{namespace='vcci-scope3'}"
          }
        ]
      },
      {
        "title": "Database Storage Forecast",
        "targets": [
          {
            "expr": "predict_linear(aws_rds_free_storage_space_bytes[7d], 90*24*3600)"
          }
        ]
      },
      {
        "title": "Request Rate Growth",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{namespace='vcci-scope3'}[5m]))"
          }
        ]
      }
    ]
  }
}
EOF

# Import dashboard to Grafana
curl -X POST http://grafana:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GRAFANA_API_KEY" \
  -d @/tmp/capacity-dashboard.json
```

## Validation Checklist

- [ ] Historical data collected (minimum 90 days)
- [ ] Growth trends analyzed
- [ ] Resource forecast calculated
- [ ] Cost projections completed
- [ ] AWS quotas reviewed and increased if needed
- [ ] Kubernetes quotas updated
- [ ] Scaling schedule created
- [ ] Capacity plan documented and approved
- [ ] Monitoring alerts configured
- [ ] Capacity dashboard created
- [ ] Stakeholders informed

## Troubleshooting

### Issue 1: Inconsistent Growth Patterns

**Resolution**: Use multiple forecasting methods (linear, exponential), consider seasonality

### Issue 2: Cost Projections Inaccurate

**Resolution**: Review Reserved Instance coverage, check for spot instance opportunities

### Issue 3: Quota Increase Requests Denied

**Resolution**: Provide business justification, engage AWS TAM, consider alternative architectures

## Related Documentation

- [Scaling Operations Runbook](./SCALING_OPERATIONS.md)
- [Performance Tuning Runbook](./PERFORMANCE_TUNING.md)
- [AWS Well-Architected Framework - Cost Optimization](https://docs.aws.amazon.com/wellarchitected/latest/cost-optimization-pillar/welcome.html)

## Contact Information

- **Platform Team**: platform-team@company.com
- **FinOps**: finops@company.com
- **AWS TAM**: tam@aws.amazon.com
- **Capacity Planning**: capacity@company.com
