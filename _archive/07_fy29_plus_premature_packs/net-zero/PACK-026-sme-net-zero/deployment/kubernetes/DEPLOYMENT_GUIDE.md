# PACK-026 SME Net Zero Pack - Kubernetes Deployment Guide

## Prerequisites

1. **Kubernetes Cluster**: v1.24+
2. **kubectl**: Configured and authenticated
3. **PostgreSQL**: GreenLang database with V158-V165 migrations applied
4. **Ingress Controller**: NGINX Ingress Controller installed
5. **Cert Manager**: For TLS certificates
6. **Prometheus Operator**: For monitoring (optional)

## Deployment Steps

### Step 1: Build Docker Image

```bash
cd C:\Users\aksha\Code-V1_GreenLang\packs\net-zero\PACK-026-sme-net-zero

# Build image
docker build -t greenlang/pack-026-sme-net-zero:1.0.0 -f deployment/Dockerfile .

# Tag for registry
docker tag greenlang/pack-026-sme-net-zero:1.0.0 registry.greenlang.io/pack-026-sme-net-zero:1.0.0

# Push to registry
docker push registry.greenlang.io/pack-026-sme-net-zero:1.0.0
```

### Step 2: Update Secrets

Edit `deployment/kubernetes/pack-026-deployment.yaml` and update the Secret section with your OAuth credentials:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: pack-026-secrets
  namespace: greenlang-packs
type: Opaque
stringData:
  # Xero OAuth (get from https://developer.xero.com/)
  XERO_CLIENT_ID: "your_xero_client_id"
  XERO_CLIENT_SECRET: "your_xero_client_secret"

  # QuickBooks OAuth (get from https://developer.intuit.com/)
  QB_CLIENT_ID: "your_qb_client_id"
  QB_CLIENT_SECRET: "your_qb_client_secret"

  # Sage OAuth (get from https://developer.sage.com/)
  SAGE_CLIENT_ID: "your_sage_client_id"
  SAGE_CLIENT_SECRET: "your_sage_client_secret"

  # Token encryption (generate with: openssl rand -hex 32)
  TOKEN_ENCRYPTION_KEY: "your_256bit_encryption_key"

  # Database connection
  DATABASE_URL: "postgresql://greenlang:PASSWORD@postgres.greenlang.svc.cluster.local:5432/greenlang"
```

**Security Note**: In production, use sealed secrets or external secret management (Vault, AWS Secrets Manager, etc.)

### Step 3: Apply Database Migrations

**IMPORTANT**: Migrations V158-V165 must be applied before deploying the pack.

```bash
# Start Docker Desktop
# Then apply migrations
cd C:\Users\aksha\Code-V1_GreenLang\deployment
docker-compose up -d postgres

# Wait for PostgreSQL to be ready
sleep 10

# Apply migrations
./apply_migrations.sh V158 V165

# Verify migrations
docker-compose exec postgres psql -U greenlang -d greenlang -c "\
SELECT version FROM schema_migrations WHERE version >= 158 AND version <= 165 ORDER BY version;"
```

Expected output:
```
 version
---------
     158
     159
     160
     161
     162
     163
     164
     165
(8 rows)
```

### Step 4: Deploy to Kubernetes

```bash
# Apply deployment manifests
kubectl apply -f deployment/kubernetes/pack-026-deployment.yaml

# Wait for deployment to be ready
kubectl wait --for=condition=available --timeout=300s \
  deployment/pack-026-sme-net-zero -n greenlang-packs

# Check pod status
kubectl get pods -n greenlang-packs -l app.kubernetes.io/name=pack-026-sme-net-zero
```

### Step 5: Verify Deployment

```bash
# Check deployment status
kubectl get deployment -n greenlang-packs pack-026-sme-net-zero

# Check service
kubectl get service -n greenlang-packs pack-026-sme-net-zero

# Check ingress
kubectl get ingress -n greenlang-packs pack-026-sme-net-zero

# View logs
kubectl logs -n greenlang-packs -l app.kubernetes.io/name=pack-026-sme-net-zero --tail=100

# Test health endpoint
kubectl port-forward -n greenlang-packs svc/pack-026-sme-net-zero 8080:80
curl http://localhost:8080/health
```

Expected health response:
```json
{
  "status": "healthy",
  "pack_id": "PACK-026",
  "pack_name": "SME Net Zero Pack",
  "version": "1.0.0",
  "database": "connected",
  "timestamp": "2026-03-18T10:00:00Z"
}
```

### Step 6: Load Grant Database

```bash
# Copy grant data to pod
kubectl cp data/comprehensive_grant_database.json \
  greenlang-packs/pack-026-sme-net-zero-xxxxx:/tmp/grants.json

# Import grants
kubectl exec -n greenlang-packs pack-026-sme-net-zero-xxxxx -- \
  python -c "
from integrations.grant_database_bridge import GrantDatabaseBridge
import json

with open('/tmp/grants.json') as f:
    data = json.load(f)

bridge = GrantDatabaseBridge()
for grant in data['grants']:
    bridge.import_grant(grant)

print(f'Imported {len(data[\"grants\"])} grants')
"
```

### Step 7: Configure Monitoring

```bash
# Verify ServiceMonitor
kubectl get servicemonitor -n greenlang-packs pack-026-sme-net-zero

# Check Prometheus targets
kubectl port-forward -n monitoring svc/prometheus-k8s 9090:9090
# Visit http://localhost:9090/targets and verify pack-026-sme-net-zero target is UP

# View metrics
curl http://localhost:9090/api/v1/query?query=pack026_baseline_calculations_total
```

### Step 8: Test Pack Functionality

```bash
# Test Express Onboarding Workflow
curl -X POST https://api.greenlang.io/packs/sme-net-zero/workflows/express \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "company_name": "Test SME Ltd",
    "headcount": 25,
    "revenue_usd": 2500000,
    "sector": "information_technology",
    "reporting_year": 2025
  }'

# Test Quick Wins Engine
curl -X POST https://api.greenlang.io/packs/sme-net-zero/engines/quick-wins \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "baseline_tco2e": 500,
    "sector": "information_technology",
    "budget_usd": 10000
  }'

# Test Grant Finder
curl -X POST https://api.greenlang.io/packs/sme-net-zero/engines/grant-finder \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "sector": "information_technology",
    "headcount": 25,
    "country": "GB",
    "postcode": "EC2A"
  }'
```

## Scaling

### Manual Scaling

```bash
# Scale up
kubectl scale deployment pack-026-sme-net-zero -n greenlang-packs --replicas=10

# Scale down
kubectl scale deployment pack-026-sme-net-zero -n greenlang-packs --replicas=3
```

### Auto-scaling

Auto-scaling is configured via HorizontalPodAutoscaler:
- **Min replicas**: 3
- **Max replicas**: 20
- **CPU target**: 70%
- **Memory target**: 80%

Monitor auto-scaling:
```bash
kubectl get hpa -n greenlang-packs pack-026-sme-net-zero -w
```

## Monitoring & Alerting

### Key Metrics

```promql
# Request rate
rate(pack026_http_requests_total[5m])

# Error rate
rate(pack026_http_errors_total[5m])

# Baseline calculation duration
histogram_quantile(0.95, rate(pack026_baseline_duration_seconds_bucket[5m]))

# OAuth token refresh success rate
rate(pack026_oauth_refresh_success_total[5m]) / rate(pack026_oauth_refresh_total[5m])

# Grant database sync status
pack026_grant_database_last_sync_timestamp_seconds
```

### Grafana Dashboard

Import dashboard from `deployment/kubernetes/grafana-dashboard.json`:
- Request/error rates
- Latency percentiles (p50, p95, p99)
- Resource utilization (CPU, memory)
- OAuth connection health
- Grant database status

## Troubleshooting

### Pod Not Starting

```bash
# Check pod events
kubectl describe pod -n greenlang-packs pack-026-sme-net-zero-xxxxx

# Check init container logs
kubectl logs -n greenlang-packs pack-026-sme-net-zero-xxxxx -c wait-for-postgres

# Check main container logs
kubectl logs -n greenlang-packs pack-026-sme-net-zero-xxxxx -c pack-026-api
```

### Database Connection Issues

```bash
# Test database connectivity from pod
kubectl exec -n greenlang-packs pack-026-sme-net-zero-xxxxx -- \
  nc -zv postgres.greenlang.svc.cluster.local 5432

# Check database migrations
kubectl exec -n greenlang-packs pack-026-sme-net-zero-xxxxx -- \
  python -c "
from sqlalchemy import create_engine, text
engine = create_engine('postgresql://greenlang:PASSWORD@postgres:5432/greenlang')
with engine.connect() as conn:
    result = conn.execute(text('SELECT version FROM schema_migrations WHERE version >= 158 AND version <= 165'))
    print([row[0] for row in result])
"
```

### OAuth Integration Issues

```bash
# Test Xero connection
kubectl exec -n greenlang-packs pack-026-sme-net-zero-xxxxx -- \
  python -c "
from integrations.xero_connector import XeroConnector, XeroConfig
import os

config = XeroConfig(
    client_id=os.getenv('XERO_CLIENT_ID'),
    client_secret=os.getenv('XERO_CLIENT_SECRET'),
    redirect_uri='https://greenlang.io/oauth/xero/callback'
)
connector = XeroConnector(config=config)
url, state = connector.get_authorization_url()
print(f'Auth URL: {url[:100]}...')
print(f'State: {state}')
"
```

## Rollback

```bash
# Rollback to previous deployment
kubectl rollout undo deployment/pack-026-sme-net-zero -n greenlang-packs

# Check rollout status
kubectl rollout status deployment/pack-026-sme-net-zero -n greenlang-packs

# View rollout history
kubectl rollout history deployment/pack-026-sme-net-zero -n greenlang-packs
```

## Production Checklist

- [ ] Database migrations V158-V165 applied successfully
- [ ] OAuth credentials configured for all 3 platforms
- [ ] TLS certificates provisioned and valid
- [ ] Ingress routing configured and tested
- [ ] ServiceMonitor scraped by Prometheus
- [ ] Grafana dashboard imported
- [ ] Alerting rules configured
- [ ] Grant database loaded (18+ programs)
- [ ] Health endpoints responding
- [ ] All 3 pod replicas running
- [ ] HPA configured and tested
- [ ] PodDisruptionBudget active (minAvailable: 2)
- [ ] Resource requests/limits tuned
- [ ] Logs shipping to Loki
- [ ] Backup strategy defined

## Security Notes

1. **Secrets Management**: Use external secret store (Vault, AWS Secrets Manager, etc.) in production
2. **RBAC**: Minimal permissions granted to service account
3. **Network Policies**: Restrict pod-to-pod communication
4. **Pod Security**: Non-root user, read-only filesystem, no privilege escalation
5. **TLS**: All external communication over HTTPS
6. **OAuth Token Storage**: Encrypted at rest using AES-256-GCM

---

**Deployment Status**: ✅ Ready for Production
**Pack Version**: 1.0.0
**Last Updated**: 2026-03-18
**Author**: GreenLang Platform Team
