# Blue-Green Deployment Strategy for GL-VCCI

## Overview

Blue-Green deployment is a zero-downtime deployment strategy where two identical production environments (Blue and Green) run simultaneously. Only one environment serves production traffic at any time.

## Architecture

```
                    ┌─────────────────┐
                    │  Load Balancer  │
                    │   (Service)     │
                    └────────┬────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
         ┌──────▼──────┐          ┌──────▼──────┐
         │    BLUE     │          │    GREEN    │
         │ Environment │          │ Environment │
         │ (Active)    │          │  (Standby)  │
         └─────────────┘          └─────────────┘
         v1.0.0 (Production)      v2.0.0 (New Version)
```

## Benefits

- **Zero Downtime**: Instant cutover between versions
- **Easy Rollback**: Switch back to previous version instantly
- **Testing in Production**: Test new version with production data before cutover
- **Reduced Risk**: New version is fully deployed and tested before receiving traffic

## Implementation Steps

### 1. Deploy Green Environment

Deploy the new version (v2.0.0) to the Green environment while Blue (v1.0.0) serves production traffic.

```bash
# Deploy Green environment with new version
kubectl apply -f k8s/blue-green/green-deployment.yaml

# Wait for Green to be ready
kubectl wait --for=condition=available --timeout=300s \
  deployment/vcci-backend-api-green -n vcci-production
```

### 2. Test Green Environment

Run smoke tests and validation against Green environment:

```bash
# Port forward to Green service for testing
kubectl port-forward service/vcci-backend-api-green 8001:8000 -n vcci-production

# Run health checks
curl http://localhost:8001/health/live
curl http://localhost:8001/health/ready

# Run smoke tests
./scripts/smoke-test.sh http://localhost:8001

# Run integration tests
./scripts/integration-test.sh http://localhost:8001
```

### 3. Switch Traffic to Green

Update the main service selector to point to Green deployment:

```bash
# Switch traffic to Green
kubectl patch service vcci-backend-api -n vcci-production \
  -p '{"spec":{"selector":{"version":"v2.0.0","color":"green"}}}'

# Verify traffic is routed to Green
kubectl get service vcci-backend-api -n vcci-production -o yaml
```

### 4. Monitor Green Environment

Monitor the new version for errors, performance issues, and anomalies:

```bash
# Watch pod status
kubectl get pods -n vcci-production -l color=green -w

# Monitor logs
kubectl logs -f -l color=green -n vcci-production

# Check metrics
kubectl top pods -n vcci-production -l color=green

# Monitor error rates in Prometheus/Grafana
# Check custom dashboards for:
# - Request rate
# - Error rate
# - Response time
# - CPU/Memory usage
```

### 5. Rollback (if needed)

If issues are detected, rollback by switching traffic back to Blue:

```bash
# Switch back to Blue
kubectl patch service vcci-backend-api -n vcci-production \
  -p '{"spec":{"selector":{"version":"v1.0.0","color":"blue"}}}'

# Verify rollback
kubectl get service vcci-backend-api -n vcci-production
```

### 6. Decommission Blue

Once Green is stable, decommission the old Blue environment:

```bash
# Scale down Blue deployment
kubectl scale deployment vcci-backend-api-blue --replicas=0 -n vcci-production

# Or delete Blue deployment
kubectl delete deployment vcci-backend-api-blue -n vcci-production
```

## Kubernetes Manifests

### Blue Deployment (v1.0.0)

See: `k8s/blue-green/blue-deployment.yaml`

### Green Deployment (v2.0.0)

See: `k8s/blue-green/green-deployment.yaml`

### Main Service (Traffic Router)

The service selector determines which environment receives traffic:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: vcci-backend-api
spec:
  selector:
    app: vcci-backend-api
    color: blue  # Change to 'green' to switch traffic
  ports:
    - port: 8000
      targetPort: 8000
```

## Automation Script

```bash
#!/bin/bash
# blue-green-deploy.sh - Automated Blue-Green Deployment

set -e

NAMESPACE="vcci-production"
APP="vcci-backend-api"
NEW_VERSION="$1"
CURRENT_COLOR=$(kubectl get service $APP -n $NAMESPACE -o jsonpath='{.spec.selector.color}')

if [ "$CURRENT_COLOR" == "blue" ]; then
  NEW_COLOR="green"
  OLD_COLOR="blue"
else
  NEW_COLOR="blue"
  OLD_COLOR="green"
fi

echo "Current active: $OLD_COLOR"
echo "Deploying to: $NEW_COLOR"

# Deploy new version
kubectl apply -f k8s/blue-green/${NEW_COLOR}-deployment.yaml
kubectl set image deployment/${APP}-${NEW_COLOR} \
  ${APP}=YOUR_REGISTRY/${APP}:${NEW_VERSION} -n $NAMESPACE

# Wait for rollout
kubectl rollout status deployment/${APP}-${NEW_COLOR} -n $NAMESPACE

# Run smoke tests
echo "Running smoke tests..."
kubectl port-forward service/${APP}-${NEW_COLOR} 8001:8000 -n $NAMESPACE &
PF_PID=$!
sleep 5

if curl -f http://localhost:8001/health/live; then
  echo "Health check passed"
else
  echo "Health check failed"
  kill $PF_PID
  exit 1
fi
kill $PF_PID

# Switch traffic
read -p "Switch traffic to $NEW_COLOR? (yes/no): " CONFIRM
if [ "$CONFIRM" == "yes" ]; then
  kubectl patch service $APP -n $NAMESPACE \
    -p "{\"spec\":{\"selector\":{\"color\":\"$NEW_COLOR\"}}}"
  echo "Traffic switched to $NEW_COLOR"
else
  echo "Deployment cancelled"
  exit 1
fi

# Monitor for 5 minutes
echo "Monitoring $NEW_COLOR for 5 minutes..."
sleep 300

# Scale down old environment
read -p "Scale down $OLD_COLOR? (yes/no): " CONFIRM
if [ "$CONFIRM" == "yes" ]; then
  kubectl scale deployment/${APP}-${OLD_COLOR} --replicas=0 -n $NAMESPACE
  echo "Scaled down $OLD_COLOR"
fi
```

## Best Practices

1. **Database Migrations**: Ensure schema changes are backward compatible
2. **Feature Flags**: Use feature flags for gradual feature rollout
3. **Monitoring**: Set up comprehensive monitoring before switching
4. **Automated Testing**: Run automated tests on Green before switching
5. **Communication**: Notify team before and after deployment
6. **Rollback Plan**: Always have a tested rollback procedure
7. **Resource Management**: Ensure cluster has capacity for both environments

## Considerations

### Pros
- Instant rollback capability
- Zero downtime deployments
- Full production testing before cutover
- Clear separation of versions

### Cons
- Requires 2x resources during deployment
- Database migrations need special handling
- Stateful services are more complex
- Cost of running duplicate environment

## Database Migration Strategy

For database migrations in Blue-Green deployments:

1. **Backward Compatible Changes**: Make changes that work with both versions
   ```sql
   -- Add new column with default value
   ALTER TABLE emissions ADD COLUMN new_field VARCHAR(255) DEFAULT 'default';
   ```

2. **Three-Phase Migration**:
   - Phase 1: Deploy schema changes compatible with old code
   - Phase 2: Deploy new application code (Blue-Green)
   - Phase 3: Remove old schema after Blue is decommissioned

3. **Separate Migration Jobs**: Run migrations as Kubernetes Jobs
   ```bash
   kubectl apply -f k8s/migrations/migration-job.yaml
   kubectl wait --for=condition=complete job/db-migration -n vcci-production
   ```

## Monitoring Checklist

During and after cutover, monitor:

- [ ] HTTP error rates (4xx, 5xx)
- [ ] Response time (p50, p95, p99)
- [ ] Database connection pool
- [ ] Cache hit rate
- [ ] Celery queue length
- [ ] CPU and memory usage
- [ ] Application logs for errors
- [ ] Business metrics (calculations/min)

## Rollback Decision Tree

```
Is Green experiencing issues?
├─ Yes
│  ├─ Traffic < 10% on Green → Delete Green, fix in staging
│  ├─ Traffic 10-50% on Green → Immediate rollback to Blue
│  └─ Traffic > 50% on Green → Evaluate severity
│     ├─ Critical → Immediate rollback
│     └─ Non-critical → Apply hotfix to Green
└─ No → Continue monitoring
```

## Integration with CI/CD

```yaml
# .github/workflows/blue-green-deploy.yml
name: Blue-Green Deployment

on:
  workflow_dispatch:
    inputs:
      version:
        description: 'Version to deploy'
        required: true

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Deploy to Green
        run: |
          ./deployment/scripts/blue-green-deploy.sh ${{ github.event.inputs.version }}

      - name: Run Smoke Tests
        run: |
          ./tests/smoke-tests.sh

      - name: Switch Traffic
        run: |
          kubectl patch service vcci-backend-api -n vcci-production \
            -p '{"spec":{"selector":{"color":"green"}}}'
```

## Troubleshooting

### Issue: Service not routing to new version
```bash
# Check service selector
kubectl describe service vcci-backend-api -n vcci-production

# Check pod labels
kubectl get pods -n vcci-production --show-labels

# Verify endpoints
kubectl get endpoints vcci-backend-api -n vcci-production
```

### Issue: Green pods not ready
```bash
# Check pod status
kubectl describe pod <pod-name> -n vcci-production

# Check logs
kubectl logs <pod-name> -n vcci-production

# Check events
kubectl get events -n vcci-production --sort-by='.lastTimestamp'
```

### Issue: Database connection errors
```bash
# Verify database connectivity from Green pod
kubectl exec -it <green-pod> -n vcci-production -- \
  psql $DATABASE_URL -c "SELECT 1"

# Check database migration status
kubectl logs <migration-job-pod> -n vcci-production
```

## Conclusion

Blue-Green deployment provides a robust, zero-downtime deployment strategy with instant rollback capabilities. While it requires additional resources, the safety and reliability it provides make it ideal for production deployments of the GL-VCCI platform.
