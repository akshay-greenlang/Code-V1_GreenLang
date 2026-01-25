# GreenLang DevOps - Quick Reference Card

## Common Commands

### Development
```bash
make dev              # Install dev dependencies
make test             # Run tests
make lint             # Run linters
make ci-test          # Full CI test suite
```

### Docker
```bash
make docker-build-all      # Build all images
make docker-push-all       # Push all images
make docker-compose-up     # Start stack
make docker-compose-down   # Stop stack
```

### Kubernetes
```bash
make deploy-dev       # Deploy to dev
make k8s-status       # Check status
make k8s-logs         # View logs
make k8s-port-forward # Port forward to localhost:8000
make rollback-dev     # Rollback deployment
```

## File Locations

### CI/CD Workflows
```
.github/workflows/
├── ci-comprehensive.yml   # Main CI pipeline
├── build-docker.yml       # Docker build & push
└── deploy-k8s.yml         # Kubernetes deployment
```

### Docker
```
Dockerfile              # CLI runtime
Dockerfile.api          # FastAPI server
docker-compose.yml      # Local stack
```

### Kubernetes
```
kubernetes/dev/
├── namespace.yaml      # Namespace
├── configmap.yaml      # Configuration
├── secrets.yaml        # Secrets (template)
├── deployment.yaml     # Deployment
├── service.yaml        # Services
├── ingress.yaml        # Ingress
└── hpa.yaml           # Autoscaling + RBAC
```

## Docker Images

```
ghcr.io/greenlang/greenlang:latest
ghcr.io/greenlang/greenlang-api:latest
ghcr.io/greenlang/greenlang-full:latest
ghcr.io/greenlang/greenlang-runner:latest
```

## Endpoints

### Development
```
http://localhost:8000              # API (local)
https://dev.greenlang.io/api       # API (dev cluster)
https://dev.greenlang.io/docs      # Swagger docs
https://dev.greenlang.io/metrics   # Prometheus metrics
```

### Health Checks
```
GET /api/v1/health     # Liveness probe
GET /api/v1/ready      # Readiness probe
GET /metrics           # Prometheus metrics
```

## Secrets Management

### Create Kubernetes Secrets
```bash
kubectl create secret generic greenlang-secrets \
  --from-literal=database-url='postgresql://...' \
  --from-literal=redis-url='redis://...' \
  --from-literal=secret-key='...' \
  --namespace greenlang-dev
```

### View Secrets
```bash
kubectl get secrets -n greenlang-dev
kubectl describe secret greenlang-secrets -n greenlang-dev
```

## Troubleshooting

### Check Pod Status
```bash
kubectl get pods -n greenlang-dev
kubectl describe pod <pod-name> -n greenlang-dev
```

### View Logs
```bash
kubectl logs -f <pod-name> -n greenlang-dev
kubectl logs -f -l app=greenlang-api -n greenlang-dev
```

### Debug Container
```bash
kubectl exec -it <pod-name> -n greenlang-dev -- /bin/bash
```

### Restart Deployment
```bash
kubectl rollout restart deployment/greenlang-api -n greenlang-dev
```

## CI/CD Triggers

### Automatic
- **PR to master/main** → ci-comprehensive.yml
- **Push to master** → ci-comprehensive.yml + build-docker.yml + deploy-k8s.yml (dev)
- **Tag v*.*.*** → build-docker.yml (with signing)

### Manual
- GitHub Actions → Run workflow → Choose environment

## Scaling

### Manual Scaling
```bash
kubectl scale deployment greenlang-api --replicas=5 -n greenlang-dev
```

### HPA Status
```bash
kubectl get hpa -n greenlang-dev
kubectl describe hpa greenlang-api-hpa -n greenlang-dev
```

## Resources

### Documentation
- `DEVOPS_README.md` - Complete guide
- `DEVOPS_DELIVERY_SUMMARY.md` - What was built
- Inline comments in all files

### Support
- GitHub Issues: https://github.com/greenlang/greenlang/issues
- Documentation: https://greenlang.io/docs
- Discord: https://discord.gg/greenlang
