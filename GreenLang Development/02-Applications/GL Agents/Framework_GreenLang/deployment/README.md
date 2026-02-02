# GreenLang Agent Deployment Templates

This directory contains production-ready deployment templates for GreenLang Industrial AI Agents (GL-001 through GL-016).

## Directory Structure

```
deployment/
├── github_workflows/          # GitHub Actions CI/CD templates
│   ├── ci.yml                 # Pull request checks (lint, test, coverage, security)
│   ├── release.yml            # Automated release with semantic versioning
│   ├── security.yml           # Security scanning (bandit, safety, CodeQL)
│   └── deploy.yml             # Kubernetes deployment workflow
│
├── kubernetes/                # Kubernetes manifest templates
│   ├── deployment.yaml        # Deployment with resource limits and health checks
│   ├── service.yaml           # Service and Ingress configuration
│   ├── hpa.yaml               # Horizontal Pod Autoscaler
│   ├── pdb.yaml               # Pod Disruption Budget
│   ├── networkpolicy.yaml     # Network security policies
│   ├── configmap.yaml         # Configuration management
│   └── secret.yaml            # Secret management templates
│
├── helm/                      # Helm chart template
│   ├── Chart.yaml             # Chart metadata
│   ├── values.yaml            # Default values
│   ├── README.md              # Chart documentation
│   └── templates/             # Kubernetes resource templates
│       ├── _helpers.tpl       # Template helpers
│       ├── deployment.yaml    # Deployment template
│       ├── service.yaml       # Service template
│       ├── ingress.yaml       # Ingress template
│       ├── hpa.yaml           # HPA template
│       ├── pdb.yaml           # PDB template
│       ├── networkpolicy.yaml # Network policy template
│       ├── configmap.yaml     # ConfigMap template
│       ├── secret.yaml        # Secret template
│       ├── serviceaccount.yaml# ServiceAccount and RBAC
│       ├── servicemonitor.yaml# Prometheus ServiceMonitor
│       └── tests/             # Helm test templates
│
└── docker/                    # Docker templates
    ├── Dockerfile.template    # Multi-stage Dockerfile
    ├── docker-compose.template.yml  # Docker Compose for local dev
    ├── .env.template          # Environment variables template
    └── .dockerignore.template # Docker ignore patterns
```

## Quick Start

### 1. Set Up GitHub Actions

Copy the workflow files to your agent's `.github/workflows/` directory:

```bash
# For agent GL-001
mkdir -p "GL-001_Thermalcommand/.github/workflows"
cp deployment/github_workflows/*.yml "GL-001_Thermalcommand/.github/workflows/"

# Customize the AGENT_NAME variable in each workflow file
sed -i 's/GL-XXX_AgentName/GL-001_Thermalcommand/g' "GL-001_Thermalcommand/.github/workflows/"*.yml
```

### 2. Set Up Docker

Copy Docker templates to your agent's root directory:

```bash
# For agent GL-001
cp deployment/docker/Dockerfile.template "GL-001_Thermalcommand/Dockerfile"
cp deployment/docker/docker-compose.template.yml "GL-001_Thermalcommand/docker-compose.yml"
cp deployment/docker/.env.template "GL-001_Thermalcommand/.env"
cp deployment/docker/.dockerignore.template "GL-001_Thermalcommand/.dockerignore"

# Customize environment variables
sed -i 's/gl-xxx/gl-001/g' "GL-001_Thermalcommand/.env"
sed -i 's/AgentName/Thermalcommand/g' "GL-001_Thermalcommand/.env"
```

### 3. Deploy with Helm

```bash
# Install the Helm chart
helm install gl-001 deployment/helm/ \
  --set agent.id=gl-001 \
  --set agent.name=Thermalcommand \
  --set image.tag=1.0.0 \
  --namespace greenlang-prod

# Or use a custom values file
helm install gl-001 deployment/helm/ \
  -f values-gl-001.yaml \
  --namespace greenlang-prod
```

### 4. Deploy with Kubernetes Manifests

```bash
# Create namespace
kubectl create namespace greenlang-prod

# Apply templates with substitution
export AGENT_ID=gl-001
export AGENT_NAME=Thermalcommand
export NAMESPACE=greenlang-prod
export VERSION=1.0.0
export IMAGE_REGISTRY=ghcr.io/greenlang

envsubst < deployment/kubernetes/deployment.yaml | kubectl apply -f -
envsubst < deployment/kubernetes/service.yaml | kubectl apply -f -
envsubst < deployment/kubernetes/hpa.yaml | kubectl apply -f -
envsubst < deployment/kubernetes/pdb.yaml | kubectl apply -f -
envsubst < deployment/kubernetes/networkpolicy.yaml | kubectl apply -f -
envsubst < deployment/kubernetes/configmap.yaml | kubectl apply -f -
```

## Agent Deployment Matrix

| Agent ID | Agent Name | Port | Group |
|----------|------------|------|-------|
| GL-001 | Thermalcommand | 8001 | thermal |
| GL-002 | Flameguard | 8002 | thermal |
| GL-003 | UnifiedSteam | 8003 | steam |
| GL-004 | Burnmaster | 8004 | combustion |
| GL-005 | Combusense | 8005 | combustion |
| GL-006 | HEATRECLAIM | 8006 | thermal |
| GL-007 | FurnacePulse | 8007 | thermal |
| GL-008 | Trapcatcher | 8008 | steam |
| GL-009 | ThermalIQ | 8009 | thermal |
| GL-010 | EmissionGuardian | 8010 | emissions |
| GL-011 | FuelCraft | 8011 | fuel |
| GL-012 | SteamQual | 8012 | steam |
| GL-013 | PredictiveMaintenance | 8013 | maintenance |
| GL-014 | Exchangerpro | 8014 | thermal |
| GL-015 | Insulscan | 8015 | thermal |
| GL-016 | Waterguard | 8016 | water |

## CI/CD Workflow

### Pull Request Pipeline (ci.yml)

1. **Lint** - Code quality checks (Black, isort, flake8, pylint, mypy)
2. **Test** - Unit tests with coverage (pytest, codecov)
3. **Security** - Quick security scan (bandit, safety)
4. **Build** - Docker image build validation
5. **Docs** - Documentation validation

### Release Pipeline (release.yml)

1. **Validate** - Version validation
2. **Test** - Full test suite
3. **Security Audit** - Comprehensive security scan
4. **Build & Push** - Multi-platform Docker build
5. **SBOM** - Generate software bill of materials
6. **Release** - Create GitHub release
7. **Deploy Staging** - Automatic staging deployment

### Security Pipeline (security.yml)

- Bandit (Python SAST)
- Safety (Dependency vulnerabilities)
- CodeQL (Advanced SAST)
- Gitleaks (Secret scanning)
- Trivy (Container scanning)
- Checkov (IaC scanning)
- License compliance

### Deploy Pipeline (deploy.yml)

- Rolling deployment
- Blue-green deployment
- Canary deployment
- Automatic rollback

## Kubernetes Resources

### Resource Limits

Default resource configuration:

| Resource | Request | Limit |
|----------|---------|-------|
| CPU | 250m | 1000m |
| Memory | 256Mi | 1Gi |
| Ephemeral Storage | 100Mi | 500Mi |

### Autoscaling

HPA configuration:

| Metric | Target | Min | Max |
|--------|--------|-----|-----|
| CPU | 70% | 3 | 20 |
| Memory | 80% | 3 | 20 |

### Health Checks

| Probe | Path | Initial Delay | Period |
|-------|------|---------------|--------|
| Startup | /api/v1/health | 10s | 5s |
| Liveness | /api/v1/health | 0s | 15s |
| Readiness | /api/v1/ready | 0s | 5s |

## Security Features

- Non-root container execution
- Read-only root filesystem
- Dropped capabilities
- Network policies
- Pod security context
- Secret management via External Secrets
- TLS termination at ingress

## Monitoring

- Prometheus metrics at /metrics
- Grafana dashboards
- Alerting rules
- OpenTelemetry tracing
- Structured JSON logging

## Best Practices

1. **Never commit secrets** - Use External Secrets or Sealed Secrets
2. **Use image digests** - Pin images for reproducibility
3. **Enable PDB** - Ensure availability during disruptions
4. **Apply network policies** - Limit pod-to-pod communication
5. **Monitor deployments** - Use ServiceMonitor for Prometheus
6. **Test Helm charts** - Run `helm test` after deployment
7. **Use semantic versioning** - Follow semver for releases

## Support

For issues or questions, please contact the GreenLang DevOps team:
- Email: devops@greenlang.io
- Slack: #greenlang-devops
