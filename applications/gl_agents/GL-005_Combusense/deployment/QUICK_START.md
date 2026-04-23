# GL-005 CombustionControlAgent - Quick Start Guide

**Get deployed in 5 minutes**

---

## Prerequisites

- kubectl installed and configured
- kustomize installed
- Docker access (for building images)

---

## Step 1: Configure Environment

```bash
cd /path/to/GL-005
cp .env.template .env
nano .env  # Edit with your credentials
```

**Minimum Required Variables:**
- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_URL` - Redis connection string
- `ANTHROPIC_API_KEY` - Claude API key

---

## Step 2: Create Secrets

```bash
# Development
kubectl create namespace greenlang-dev
kubectl create secret generic gl-005-secrets \
  --from-literal=database_url='postgresql://user:pass@host/db' \
  --from-literal=redis_url='redis://host:6379/0' \
  --from-literal=anthropic_api_key='sk-ant-xxx' \
  -n greenlang-dev

# Production
kubectl create namespace greenlang
kubectl create secret generic gl-005-secrets \
  --from-literal=database_url='postgresql://user:pass@host/db' \
  --from-literal=redis_url='redis://host:6379/0' \
  --from-literal=anthropic_api_key='sk-ant-xxx' \
  -n greenlang
```

---

## Step 3: Build and Push Docker Image

```bash
# Build
docker build -f Dockerfile \
  --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
  --build-arg VCS_REF=$(git rev-parse --short HEAD) \
  -t gcr.io/greenlang/gl-005-combustion-control:1.0.0 .

# Push
docker push gcr.io/greenlang/gl-005-combustion-control:1.0.0
```

---

## Step 4: Deploy

### Option A: Automated Script (Recommended)

```bash
cd deployment/scripts

# Validate
./validate.sh dev

# Deploy
./deploy.sh dev

# Verify
kubectl get pods -n greenlang-dev -l app=gl-005-combustion-control
```

### Option B: Manual Deployment

```bash
cd deployment

# Build manifests
kustomize build kustomize/overlays/dev > /tmp/gl-005-dev.yaml

# Review
cat /tmp/gl-005-dev.yaml

# Apply
kubectl apply -f /tmp/gl-005-dev.yaml

# Wait for rollout
kubectl rollout status deployment/dev-gl-005-combustion-control -n greenlang-dev
```

---

## Step 5: Verify

```bash
# Check pods
kubectl get pods -n greenlang-dev -l app=gl-005-combustion-control

# Check logs
kubectl logs -n greenlang-dev -l app=gl-005-combustion-control -f

# Port-forward
kubectl port-forward -n greenlang-dev svc/dev-gl-005-combustion-control 8000:80

# Test health
curl http://localhost:8000/api/v1/health
```

---

## Step 6: Access Application

### Local Access (Port Forward)

```bash
kubectl port-forward -n greenlang-dev svc/dev-gl-005-combustion-control 8000:80
```

Then visit: http://localhost:8000

### External Access (Ingress)

Configure DNS:
```
gl-005-dev.greenlang.io -> <ingress-ip>
```

Then visit: https://gl-005-dev.greenlang.io

---

## Troubleshooting

### Pods Not Starting

```bash
kubectl describe pod <pod-name> -n greenlang-dev
kubectl logs <pod-name> -n greenlang-dev
```

### Connection Issues

```bash
# Check secrets
kubectl get secret gl-005-secrets -n greenlang-dev -o yaml

# Test database connectivity
kubectl exec -it <pod-name> -n greenlang-dev -- env | grep DATABASE
```

### Rollback

```bash
cd deployment/scripts
./rollback.sh dev
```

---

## Production Deployment

Once validated in dev/staging:

```bash
# Validate
./validate.sh production

# Deploy
./deploy.sh production

# Monitor
kubectl get pods -n greenlang -l app=gl-005-combustion-control -w
```

---

## Common Commands

```bash
# Scale manually
kubectl scale deployment gl-005-combustion-control -n greenlang --replicas=5

# View logs
kubectl logs -n greenlang -l app=gl-005-combustion-control -f --tail=100

# Check HPA
kubectl get hpa -n greenlang

# Check metrics
kubectl top pods -n greenlang -l app=gl-005-combustion-control

# Rollback
./rollback.sh production
```

---

## Next Steps

1. Read full deployment guide: `DEPLOYMENT_GUIDE.md`
2. Configure monitoring dashboards
3. Set up alerting (PagerDuty, Slack)
4. Review security settings
5. Schedule regular backups

---

**Need Help?**
- Full documentation: `DEPLOYMENT_GUIDE.md`
- Summary: `DEPLOYMENT_SUMMARY.md`
- Support: devops@greenlang.io
