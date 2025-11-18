# GL-002 CI/CD Scripts

Comprehensive automation scripts for deployment, health checking, and quality validation.

## Scripts Overview

| Script | Purpose | Usage |
|--------|---------|-------|
| `quality_gates.py` | Validate all quality gates | `python quality_gates.py` |
| `deploy-staging.sh` | Deploy to staging environment | `./deploy-staging.sh [TAG]` |
| `deploy-production.sh` | Deploy to production (blue-green) | `./deploy-production.sh [TAG]` |
| `rollback.sh` | Rollback deployment | `./rollback.sh [ENVIRONMENT]` |
| `health-check.sh` | Comprehensive health validation | `./health-check.sh [ENV] [URL]` |

## quality_gates.py

Comprehensive quality gate validation for CI/CD pipeline.

### Features

- Code coverage validation (>= 95%)
- Type hint coverage check (>= 100%)
- Security issue detection (0 critical)
- Complexity analysis (<= 10)
- Test results validation
- Documentation completeness

### Usage

```bash
# Run all quality gates
python scripts/quality_gates.py

# In CI pipeline
pytest --cov=. --cov-report=json:coverage.json tests/
python scripts/quality_gates.py
```

### Exit Codes

- `0`: All quality gates passed
- `1`: One or more quality gates failed

## deploy-staging.sh

Automated deployment to staging environment.

### Features

- Docker image verification
- Kubernetes deployment update
- Rollout status monitoring
- Pod health verification
- Smoke test execution
- Automated rollback on failure

### Usage

```bash
# Deploy latest image
./scripts/deploy-staging.sh

# Deploy specific version
./scripts/deploy-staging.sh v1.2.3

# With custom registry
IMAGE_REGISTRY=custom.registry.io ./scripts/deploy-staging.sh v1.2.3
```

### Environment Variables

- `IMAGE_REGISTRY`: Container registry (default: `ghcr.io/greenlang`)
- `NAMESPACE`: Kubernetes namespace (default: `greenlang`)
- `STAGING_URL`: Staging URL for smoke tests

## deploy-production.sh

Blue-green deployment to production with safety checks.

### Features

- Production confirmation prompt
- Blue deployment snapshot
- Green deployment creation
- Comprehensive health validation
- Automated traffic switching
- Automatic rollback on failure
- Deployment notifications

### Usage

```bash
# Deploy to production (with confirmation)
./scripts/deploy-production.sh v1.2.3

# Check prerequisites
./scripts/deploy-production.sh --check

# Dry run
DRY_RUN=true ./scripts/deploy-production.sh v1.2.3
```

### Safety Features

1. **Confirmation Prompt**: Requires explicit confirmation
2. **Cluster Verification**: Validates production cluster context
3. **Blue Snapshot**: Saves current state before deployment
4. **Health Checks**: 5 retries with 10s interval
5. **Automatic Rollback**: Triggers on any failure
6. **Notifications**: Slack/PagerDuty alerts

## rollback.sh

Automated rollback to previous deployment version.

### Features

- Deployment history display
- Automated rollback execution
- Health verification after rollback
- Status notifications

### Usage

```bash
# Rollback production
./scripts/rollback.sh production

# Rollback staging (skip confirmation)
./scripts/rollback.sh staging

# View rollback history
kubectl rollout history deployment/gl-002-boiler-efficiency -n greenlang
```

### Rollback Flow

1. Confirm rollback (production only)
2. Display deployment history
3. Execute rollback
4. Wait for completion
5. Verify all pods healthy
6. Run health checks
7. Send notifications

## health-check.sh

Comprehensive health validation for deployed services.

### Features

- Service availability check
- Health endpoint validation
- Readiness verification
- Liveness check
- Metrics endpoint test
- Response time measurement
- Kubernetes pod status
- Database connectivity
- Cache connectivity

### Usage

```bash
# Check staging health
./scripts/health-check.sh staging

# Check production health
./scripts/health-check.sh production

# Custom URL
./scripts/health-check.sh production https://custom.url

# Local development
./scripts/health-check.sh dev http://localhost:8000
```

### Health Checks

1. **Service Availability**: Basic connectivity
2. **Health Endpoint**: `/api/v1/health`
3. **Readiness**: `/api/v1/ready`
4. **Liveness**: `/api/v1/health/live`
5. **Metrics**: `/metrics`
6. **API Docs**: `/docs`
7. **Response Time**: Latency measurement
8. **K8s Status**: Pod health (if kubectl available)
9. **Database**: Connection check
10. **Cache**: Redis connectivity

### Exit Codes

- `0`: All health checks passed
- `1`: One or more health checks failed

## Script Permissions

Make scripts executable:

```bash
chmod +x scripts/*.sh
```

## Environment Configuration

### Required Variables

```bash
# Kubernetes
export KUBECONFIG=/path/to/kubeconfig

# Container Registry
export IMAGE_REGISTRY=ghcr.io/greenlang

# Deployment
export NAMESPACE=greenlang
export DEPLOYMENT_NAME=gl-002-boiler-efficiency

# URLs
export STAGING_URL=https://api.staging.greenlang.io
export PROD_URL=https://api.boiler.greenlang.io

# Notifications
export SLACK_WEBHOOK_URL=https://hooks.slack.com/...
export PAGERDUTY_API_KEY=your-key
```

### Optional Variables

```bash
# Timeouts
export TIMEOUT=10m
export HEALTH_CHECK_RETRIES=5
export HEALTH_CHECK_INTERVAL=10

# Logging
export DEBUG=true
export VERBOSE=true
```

## CI/CD Integration

### GitHub Actions

```yaml
- name: Run quality gates
  run: python scripts/quality_gates.py

- name: Deploy to staging
  run: ./scripts/deploy-staging.sh ${{ github.sha }}

- name: Health check
  run: ./scripts/health-check.sh staging

- name: Deploy to production
  run: ./scripts/deploy-production.sh v${{ github.run_number }}
```

### GitLab CI

```yaml
quality_gates:
  script:
    - python scripts/quality_gates.py

deploy_staging:
  script:
    - ./scripts/deploy-staging.sh ${CI_COMMIT_SHA}

health_check:
  script:
    - ./scripts/health-check.sh staging

deploy_production:
  script:
    - ./scripts/deploy-production.sh v${CI_PIPELINE_IID}
```

## Troubleshooting

### Script Permission Denied

```bash
chmod +x scripts/*.sh
```

### kubectl Not Found

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/
```

### Health Check Timeout

```bash
# Increase timeout
TIMEOUT=15m ./scripts/health-check.sh production

# Increase retries
HEALTH_CHECK_RETRIES=10 ./scripts/health-check.sh production
```

### Deployment Fails

```bash
# Check deployment status
kubectl get deployment gl-002-boiler-efficiency -n greenlang

# View events
kubectl get events -n greenlang --sort-by='.lastTimestamp'

# Check logs
kubectl logs -l app=gl-002-boiler-efficiency -n greenlang --tail=100

# Manual rollback
./scripts/rollback.sh production
```

## Best Practices

1. **Always test in staging first**
2. **Run health checks before production**
3. **Monitor deployments for 24 hours**
4. **Keep rollback plan ready**
5. **Use semantic versioning for tags**
6. **Document deployment issues**
7. **Communicate with stakeholders**

## Security

- Scripts use non-root user in containers
- Secrets managed via environment variables
- No credentials in code
- All connections use TLS
- Image scanning before deployment

## Support

For issues or questions:
- DevOps Team: devops@greenlang.io
- Slack: #greenlang-devops
- PagerDuty: GL-002 On-Call

---

**Last Updated**: 2025-11-17
**Version**: 1.0.0
