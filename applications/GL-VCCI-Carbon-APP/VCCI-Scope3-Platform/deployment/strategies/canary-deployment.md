# Canary Deployment Strategy for GL-VCCI

## Overview

Canary deployment is a progressive rollout strategy where a new version is gradually rolled out to a subset of users/servers before being deployed to the entire infrastructure. This allows early detection of issues with minimal user impact.

## Architecture

```
                    ┌─────────────────┐
                    │  Ingress/LB     │
                    │  (Traffic Split)│
                    └────────┬────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
         ┌──────▼──────┐          ┌──────▼──────┐
         │   STABLE    │          │   CANARY    │
         │   (v1.0.0)  │          │   (v2.0.0)  │
         │   95% Traffic│         │  5% Traffic │
         └─────────────┘          └─────────────┘
         Replicas: 19             Replicas: 1
```

## Canary Progression Strategy

```
Initial:  Stable 100% → Canary 0%
Stage 1:  Stable 95%  → Canary 5%   (1 pod)
Stage 2:  Stable 90%  → Canary 10%  (2 pods)
Stage 3:  Stable 75%  → Canary 25%  (5 pods)
Stage 4:  Stable 50%  → Canary 50%  (10 pods)
Stage 5:  Stable 25%  → Canary 75%  (15 pods)
Stage 6:  Stable 0%   → Canary 100% (20 pods, promote)
```

## Benefits

- **Risk Mitigation**: Limited user exposure to new version
- **Early Issue Detection**: Catch bugs before full rollout
- **Gradual Migration**: Smooth transition with monitoring at each stage
- **A/B Testing**: Compare metrics between versions
- **Quick Rollback**: Easy to revert if issues detected

## Implementation Methods

### Method 1: Using Kubernetes Native (Service + Multiple Deployments)

#### Step 1: Deploy Canary Deployment

```yaml
# k8s/canary/canary-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vcci-backend-api-canary
  namespace: vcci-production
spec:
  replicas: 1  # Start with 1 pod (5% of total 20)
  selector:
    matchLabels:
      app: vcci-backend-api
      version: v2.0.0
      track: canary
  template:
    metadata:
      labels:
        app: vcci-backend-api
        version: v2.0.0
        track: canary
    spec:
      containers:
        - name: backend-api
          image: YOUR_REGISTRY/vcci-backend:v2.0.0
          # ... rest of spec
```

#### Step 2: Stable Deployment

```yaml
# k8s/canary/stable-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vcci-backend-api-stable
  namespace: vcci-production
spec:
  replicas: 19  # 95% of total
  selector:
    matchLabels:
      app: vcci-backend-api
      version: v1.0.0
      track: stable
  template:
    metadata:
      labels:
        app: vcci-backend-api
        version: v1.0.0
        track: stable
    spec:
      containers:
        - name: backend-api
          image: YOUR_REGISTRY/vcci-backend:v1.0.0
          # ... rest of spec
```

#### Step 3: Service (Load Balances Both)

```yaml
# Service automatically load balances between stable and canary
apiVersion: v1
kind: Service
metadata:
  name: vcci-backend-api
spec:
  selector:
    app: vcci-backend-api  # Matches both stable and canary
  ports:
    - port: 8000
      targetPort: 8000
```

### Method 2: Using Istio Virtual Service (Recommended for Fine Control)

```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: vcci-backend-api
  namespace: vcci-production
spec:
  hosts:
    - vcci-backend-api
  http:
    - match:
        - headers:
            canary:
              exact: "true"
      route:
        - destination:
            host: vcci-backend-api
            subset: canary
          weight: 100

    - route:
        - destination:
            host: vcci-backend-api
            subset: stable
          weight: 95
        - destination:
            host: vcci-backend-api
            subset: canary
          weight: 5

---
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: vcci-backend-api
spec:
  host: vcci-backend-api
  subsets:
    - name: stable
      labels:
        version: v1.0.0
        track: stable
    - name: canary
      labels:
        version: v2.0.0
        track: canary
```

### Method 3: Using Argo Rollouts (Progressive Delivery)

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: vcci-backend-api
  namespace: vcci-production
spec:
  replicas: 20
  strategy:
    canary:
      steps:
        - setWeight: 5
        - pause: {duration: 10m}
        - setWeight: 10
        - pause: {duration: 10m}
        - setWeight: 25
        - pause: {duration: 10m}
        - setWeight: 50
        - pause: {duration: 10m}
        - setWeight: 75
        - pause: {duration: 10m}

      # Automated analysis
      analysis:
        templates:
          - templateName: success-rate
        startingStep: 2
        args:
          - name: service-name
            value: vcci-backend-api

      # Automated rollback on failure
      trafficRouting:
        istio:
          virtualService:
            name: vcci-backend-api
            routes:
              - primary

  selector:
    matchLabels:
      app: vcci-backend-api

  template:
    metadata:
      labels:
        app: vcci-backend-api
    spec:
      # ... pod spec
```

## Deployment Script

```bash
#!/bin/bash
# canary-deploy.sh - Automated Canary Deployment

set -e

NAMESPACE="vcci-production"
APP="vcci-backend-api"
NEW_VERSION="$1"
CANARY_REPLICAS=1
STABLE_REPLICAS=19

# Validate version
if [ -z "$NEW_VERSION" ]; then
  echo "Usage: $0 <version>"
  exit 1
fi

echo "Starting canary deployment for $APP:$NEW_VERSION"

# Deploy canary
echo "Deploying canary (5% traffic)..."
kubectl apply -f k8s/canary/canary-deployment.yaml
kubectl set image deployment/${APP}-canary \
  ${APP}=YOUR_REGISTRY/${APP}:${NEW_VERSION} -n $NAMESPACE

# Wait for canary to be ready
kubectl rollout status deployment/${APP}-canary -n $NAMESPACE

# Monitor canary for 10 minutes
echo "Monitoring canary (Stage 1: 5%)..."
./scripts/monitor-canary.sh $NAMESPACE $APP 600

# Check metrics
if ./scripts/check-canary-metrics.sh $NAMESPACE $APP; then
  echo "Canary metrics healthy, proceeding to Stage 2"
else
  echo "Canary metrics unhealthy, rolling back"
  kubectl delete deployment/${APP}-canary -n $NAMESPACE
  exit 1
fi

# Stage 2: 10%
echo "Scaling canary to 10%..."
kubectl scale deployment/${APP}-canary --replicas=2 -n $NAMESPACE
kubectl scale deployment/${APP}-stable --replicas=18 -n $NAMESPACE
./scripts/monitor-canary.sh $NAMESPACE $APP 600

# Stage 3: 25%
echo "Scaling canary to 25%..."
kubectl scale deployment/${APP}-canary --replicas=5 -n $NAMESPACE
kubectl scale deployment/${APP}-stable --replicas=15 -n $NAMESPACE
./scripts/monitor-canary.sh $NAMESPACE $APP 600

# Stage 4: 50%
echo "Scaling canary to 50%..."
kubectl scale deployment/${APP}-canary --replicas=10 -n $NAMESPACE
kubectl scale deployment/${APP}-stable --replicas=10 -n $NAMESPACE
./scripts/monitor-canary.sh $NAMESPACE $APP 600

# Stage 5: 75%
echo "Scaling canary to 75%..."
kubectl scale deployment/${APP}-canary --replicas=15 -n $NAMESPACE
kubectl scale deployment/${APP}-stable --replicas=5 -n $NAMESPACE
./scripts/monitor-canary.sh $NAMESPACE $APP 600

# Stage 6: 100% (Promote)
echo "Promoting canary to stable..."
kubectl scale deployment/${APP}-canary --replicas=20 -n $NAMESPACE
kubectl scale deployment/${APP}-stable --replicas=0 -n $NAMESPACE

# Update stable deployment to new version
kubectl set image deployment/${APP}-stable \
  ${APP}=YOUR_REGISTRY/${APP}:${NEW_VERSION} -n $NAMESPACE

# Swap labels (canary becomes stable)
kubectl patch deployment ${APP}-canary -n $NAMESPACE \
  -p '{"spec":{"template":{"metadata":{"labels":{"track":"stable"}}}}}'

kubectl patch deployment ${APP}-stable -n $NAMESPACE \
  -p '{"spec":{"template":{"metadata":{"labels":{"track":"canary"}}}}}'

echo "Canary deployment completed successfully!"
```

## Monitoring Script

```bash
#!/bin/bash
# monitor-canary.sh - Monitor canary deployment metrics

NAMESPACE=$1
APP=$2
DURATION=${3:-300}  # Default 5 minutes

echo "Monitoring $APP in $NAMESPACE for $DURATION seconds..."

START_TIME=$(date +%s)
END_TIME=$((START_TIME + DURATION))

while [ $(date +%s) -lt $END_TIME ]; do
  # Get canary and stable pod counts
  CANARY_PODS=$(kubectl get pods -n $NAMESPACE -l app=$APP,track=canary -o name | wc -l)
  STABLE_PODS=$(kubectl get pods -n $NAMESPACE -l app=$APP,track=stable -o name | wc -l)

  # Get error rates (requires Prometheus)
  CANARY_ERRORS=$(curl -s "http://prometheus:9090/api/v1/query?query=rate(http_requests_total{app=\"$APP\",track=\"canary\",status=~\"5..\"}[1m])" | jq -r '.data.result[0].value[1] // 0')
  STABLE_ERRORS=$(curl -s "http://prometheus:9090/api/v1/query?query=rate(http_requests_total{app=\"$APP\",track=\"stable\",status=~\"5..\"}[1m])" | jq -r '.data.result[0].value[1] // 0')

  echo "$(date): Canary Pods: $CANARY_PODS, Stable Pods: $STABLE_PODS"
  echo "  Canary Error Rate: $CANARY_ERRORS, Stable Error Rate: $STABLE_ERRORS"

  # Check if canary error rate is significantly higher than stable
  if (( $(echo "$CANARY_ERRORS > $STABLE_ERRORS * 2" | bc -l) )); then
    echo "WARNING: Canary error rate is 2x higher than stable!"
  fi

  sleep 30
done

echo "Monitoring complete"
```

## Metrics to Monitor

### Key Performance Indicators (KPIs)

1. **Error Rate**
   ```promql
   rate(http_requests_total{track="canary",status=~"5.."}[5m])
   /
   rate(http_requests_total{track="canary"}[5m])
   ```

2. **Response Time (p99)**
   ```promql
   histogram_quantile(0.99,
     rate(http_request_duration_seconds_bucket{track="canary"}[5m])
   )
   ```

3. **Request Rate**
   ```promql
   rate(http_requests_total{track="canary"}[5m])
   ```

4. **CPU Usage**
   ```promql
   rate(container_cpu_usage_seconds_total{pod=~".*-canary-.*"}[5m])
   ```

5. **Memory Usage**
   ```promql
   container_memory_usage_bytes{pod=~".*-canary-.*"}
   ```

### Comparison Metrics (Canary vs Stable)

```bash
# Error rate comparison
kubectl exec -it prometheus-pod -- promtool query instant \
  'rate(http_requests_total{track="canary",status=~"5.."}[5m]) /
   rate(http_requests_total{track="stable",status=~"5.."}[5m])'

# Response time comparison
kubectl exec -it prometheus-pod -- promtool query instant \
  'histogram_quantile(0.99, rate(http_request_duration_seconds_bucket{track="canary"}[5m])) /
   histogram_quantile(0.99, rate(http_request_duration_seconds_bucket{track="stable"}[5m]))'
```

## Automated Rollback Criteria

Rollback automatically if any of these conditions are met:

1. **Error Rate**: Canary error rate > 2x stable error rate
2. **Response Time**: Canary p99 > 1.5x stable p99
3. **Crash Loop**: Any canary pod crashes more than 3 times
4. **Health Checks**: Readiness probe fails > 50% of checks

```yaml
# Argo Rollouts Analysis Template
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: success-rate
spec:
  args:
    - name: service-name
  metrics:
    - name: success-rate
      interval: 1m
      count: 5
      successCondition: result >= 0.95
      failureLimit: 3
      provider:
        prometheus:
          address: http://prometheus:9090
          query: |
            sum(rate(http_requests_total{app="{{args.service-name}}",track="canary",status!~"5.."}[1m]))
            /
            sum(rate(http_requests_total{app="{{args.service-name}}",track="canary"}[1m]))
```

## Best Practices

1. **Start Small**: Begin with 1-5% traffic
2. **Monitor Closely**: Watch metrics at each stage
3. **Automate Checks**: Use automated analysis to detect issues
4. **Gradual Progression**: Don't rush through stages
5. **User Segmentation**: Route specific users to canary (beta testers)
6. **Database Compatibility**: Ensure canary works with current schema
7. **Rollback Plan**: Test rollback procedure before deployment

## Advanced Techniques

### Header-Based Routing (Beta Users)

```yaml
# Route beta users to canary
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: vcci-backend-api
spec:
  hosts:
    - vcci-backend-api
  http:
    # Beta users always get canary
    - match:
        - headers:
            x-user-group:
              exact: "beta"
      route:
        - destination:
            host: vcci-backend-api
            subset: canary
          weight: 100

    # Regular users get weighted split
    - route:
        - destination:
            host: vcci-backend-api
            subset: stable
          weight: 95
        - destination:
            host: vcci-backend-api
            subset: canary
          weight: 5
```

### Geographic Routing

```yaml
# Route specific region to canary first
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: vcci-backend-api
spec:
  hosts:
    - vcci-backend-api
  http:
    - match:
        - headers:
            x-region:
              exact: "us-west-1"
      route:
        - destination:
            host: vcci-backend-api
            subset: canary
          weight: 100

    - route:
        - destination:
            host: vcci-backend-api
            subset: stable
          weight: 100
```

## Troubleshooting

### Issue: Uneven traffic distribution

```bash
# Check actual traffic distribution
kubectl exec -it prometheus-pod -- promtool query instant \
  'sum(rate(http_requests_total{track="canary"}[5m])) /
   sum(rate(http_requests_total[5m]))'

# Check pod distribution
kubectl get pods -n vcci-production -l app=vcci-backend-api --show-labels

# Verify service endpoints
kubectl get endpoints vcci-backend-api -n vcci-production -o yaml
```

### Issue: Canary pods not receiving traffic

```bash
# Check service selector
kubectl describe service vcci-backend-api -n vcci-production

# Verify pod labels
kubectl get pods -n vcci-production -l track=canary --show-labels

# Check for network policies blocking traffic
kubectl describe networkpolicy -n vcci-production
```

## Comparison: Canary vs Blue-Green

| Aspect | Canary | Blue-Green |
|--------|--------|------------|
| **Traffic Split** | Gradual (5% → 100%) | Instant (0% → 100%) |
| **Risk** | Lower | Medium |
| **Rollback** | Gradual decrease | Instant switch |
| **Resources** | 1x + small overhead | 2x resources |
| **Complexity** | Higher (traffic routing) | Lower |
| **Detection Time** | Faster (limited blast radius) | Slower (full deployment) |
| **Best For** | High-risk changes | Predictable changes |

## Conclusion

Canary deployment provides a safe, progressive rollout strategy that minimizes risk by limiting user exposure to new versions. Combined with automated monitoring and rollback, it's ideal for production deployments where reliability is critical.

For GL-VCCI, canary deployments are recommended for:
- Major version upgrades
- ML model updates
- Database schema changes
- Critical calculation engine changes
