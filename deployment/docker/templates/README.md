# GreenLang Dockerfile Templates

Three parameterized Dockerfile templates that replace 30+ near-identical Dockerfiles across the codebase.

## Templates

| Template | Replaces | Use Case |
|---|---|---|
| `Dockerfile.agent` | 16+ agent Dockerfiles (DATA-011..020, GL-001..017) | Any GL agent microservice |
| `Dockerfile.cli` | Dockerfile.core, Dockerfile.full, Dockerfile.secure, Runner.Dockerfile, Full.Dockerfile | GreenLang CLI images (core/full/dev) |
| `Dockerfile.api` | CSRD, VCCI, CBAM, Registry, Agent Factory API Dockerfiles | FastAPI/ASGI application services |

## Quick Reference

### Dockerfile.agent

Builds any GreenLang agent as a hardened microservice with health checks, metrics, and proper signal handling.

**Key build args:**

| ARG | Default | Description |
|---|---|---|
| `AGENT_ID` | `gl-000` | Agent identifier (e.g., `GL-DATA-X-014`) |
| `AGENT_NAME` | `agent` | Human-readable name (e.g., `duplicate-detector`) |
| `APP_MODULE` | `api.rest_api:app` | Uvicorn import path |
| `APP_PORT` | `8000` | HTTP listen port |
| `METRICS_PORT` | `9090` | Prometheus metrics port |
| `HEALTH_PATH` | `/health` | Health check endpoint path |
| `WORKERS` | `4` | Uvicorn worker count |
| `REQUIREMENTS_FILE` | `requirements.txt` | Path to pip requirements |
| `VERSION` | `1.0.0` | Semantic version |

**Build targets:** `production` (default), `development`, `test`

**Examples:**

```bash
# Duplicate Detection Agent
docker build \
  --build-arg AGENT_ID=GL-DATA-X-014 \
  --build-arg AGENT_NAME=duplicate-detector \
  --build-arg APP_MODULE=greenlang.duplicate_detector.setup:create_app \
  --build-arg HEALTH_PATH=/api/v1/dedup/health \
  -f deployment/docker/templates/Dockerfile.agent \
  -t greenlang/duplicate-detector:1.0.0 .

# Climate Hazard Connector (non-default port)
docker build \
  --build-arg AGENT_ID=GL-DATA-GEO-002 \
  --build-arg AGENT_NAME=climate-hazard \
  --build-arg APP_MODULE=greenlang.climate_hazard.setup:create_app \
  --build-arg HEALTH_PATH=/api/v1/climate-hazard/health \
  --build-arg APP_PORT=8020 \
  -f deployment/docker/templates/Dockerfile.agent \
  -t greenlang/climate-hazard:1.0.0 .

# Validation Rule Engine
docker build \
  --build-arg AGENT_ID=GL-DATA-X-022 \
  --build-arg AGENT_NAME=validation-rule-engine \
  --build-arg APP_MODULE=greenlang.validation_rule_engine.setup:create_app \
  --build-arg HEALTH_PATH=/api/v1/validation-rules/health \
  -f deployment/docker/templates/Dockerfile.agent \
  -t greenlang/validation-rule-engine:1.0.0 .

# Development mode with hot-reload
docker build \
  --target development \
  --build-arg AGENT_ID=GL-DATA-X-014 \
  --build-arg AGENT_NAME=duplicate-detector \
  --build-arg APP_MODULE=greenlang.duplicate_detector.setup:create_app \
  -f deployment/docker/templates/Dockerfile.agent \
  -t greenlang/duplicate-detector:dev .

# CI test runner
docker build \
  --target test \
  --build-arg AGENT_ID=GL-DATA-X-014 \
  -f deployment/docker/templates/Dockerfile.agent \
  -t greenlang/duplicate-detector:test .
```

### Dockerfile.cli

Builds the GreenLang CLI with configurable dependency profiles controlled by `GL_INSTALL_EXTRAS`.

**Key build args:**

| ARG | Default | Description |
|---|---|---|
| `GL_INSTALL_EXTRAS` | `""` (empty = core) | Comma-separated pyproject.toml extras |
| `GL_VERSION` | `0.3.0` | GreenLang version |

**Build targets:** `production` (default), `development`

**Dependency profiles:**

| Profile | `GL_INSTALL_EXTRAS` value | Image size | Use case |
|---|---|---|---|
| Core | `""` | ~150 MB | Minimal CLI |
| Server | `server,security` | ~250 MB | API-capable runtime |
| Full | `full` | ~500 MB | All optional deps |
| Dev | `full,dev,test` | ~650 MB | Development + CI |

**Examples:**

```bash
# Minimal core CLI
docker build \
  --build-arg GL_INSTALL_EXTRAS="" \
  -f deployment/docker/templates/Dockerfile.cli \
  -t greenlang/cli:core .

# Full CLI
docker build \
  --build-arg GL_INSTALL_EXTRAS="full" \
  -f deployment/docker/templates/Dockerfile.cli \
  -t greenlang/cli:full .

# Server-ready with security
docker build \
  --build-arg GL_INSTALL_EXTRAS="server,security" \
  -f deployment/docker/templates/Dockerfile.cli \
  -t greenlang/cli:secure .

# Development environment
docker build \
  --build-arg GL_INSTALL_EXTRAS="full,dev,test" \
  --target development \
  -f deployment/docker/templates/Dockerfile.cli \
  -t greenlang/cli:dev .
```

### Dockerfile.api

Builds any GreenLang FastAPI/ASGI application (CSRD, VCCI, CBAM, Registry, etc.).

**Key build args:**

| ARG | Default | Description |
|---|---|---|
| `APP_NAME` | `app` | Application name |
| `APP_MODULE` | `greenlang.api.main:app` | Uvicorn import path |
| `APP_PORT` | `8000` | HTTP listen port |
| `METRICS_PORT` | `9090` | Prometheus metrics port |
| `HEALTH_PATH` | `/api/v1/health` | Health check endpoint |
| `INSTALL_EXTRAS` | `server,data,security` | pyproject.toml extras |
| `WORKERS` | `4` | Uvicorn worker count |

**Build targets:** `production` (default), `development`, `worker`

**Examples:**

```bash
# CSRD Reporting Platform
docker build \
  --build-arg APP_NAME=csrd-reporting \
  --build-arg APP_MODULE=greenlang.csrd.api:app \
  --build-arg INSTALL_EXTRAS="server,security,analytics" \
  -f deployment/docker/templates/Dockerfile.api \
  -t greenlang/csrd:1.0.0 .

# CBAM Importer Copilot
docker build \
  --build-arg APP_NAME=cbam-importer \
  --build-arg APP_MODULE=greenlang.cbam.api:app \
  --build-arg APP_PORT=8001 \
  -f deployment/docker/templates/Dockerfile.api \
  -t greenlang/cbam:1.0.0 .

# VCCI Scope 3 Backend
docker build \
  --build-arg APP_NAME=vcci-scope3 \
  --build-arg APP_MODULE=greenlang.vcci.api:app \
  -f deployment/docker/templates/Dockerfile.api \
  -t greenlang/vcci:1.0.0 .

# Agent Registry API
docker build \
  --build-arg APP_NAME=agent-registry \
  --build-arg APP_MODULE=greenlang.registry.api:app \
  --build-arg HEALTH_PATH=/health \
  -f deployment/docker/templates/Dockerfile.api \
  -t greenlang/registry:1.0.0 .

# CBAM background worker
docker build \
  --target worker \
  --build-arg APP_NAME=cbam-worker \
  --build-arg CELERY_APP=greenlang.cbam.tasks:app \
  --build-arg CELERY_QUEUES=cbam-imports,cbam-reports \
  -f deployment/docker/templates/Dockerfile.api \
  -t greenlang/cbam-worker:1.0.0 .
```

## .dockerignore Template

Copy `.dockerignore.template` to `.dockerignore` in your build context root:

```bash
cp deployment/docker/templates/.dockerignore.template .dockerignore
```

This prevents secrets, tests, documentation, IDE files, and build artefacts from entering the Docker build context.

## Security Features (all templates)

All three templates implement the following security hardening:

- **Non-root user** with UID 10001 (container best practice range)
- **Multi-stage builds** to exclude build tools from production images
- **tini** as PID 1 for signal forwarding and zombie process reaping
- **Minimal base image** (`python:3.11-slim-bookworm`)
- **No pip cache** in final image
- **Read-only root filesystem** compatible (writable dirs declared as VOLUME)
- **OCI labels** for supply chain metadata
- **Health checks** with configurable endpoints
- **Capability dropping** via `setcap` on Python binary

## Migration from Existing Dockerfiles

To migrate an existing agent Dockerfile to use the template, replace the per-agent file with a build command referencing the template. For CI/CD, update the `docker build` invocation in the pipeline YAML:

```yaml
# Before (per-agent Dockerfile)
- name: Build
  run: docker build -f deployment/docker/Dockerfile.duplicate-detector -t img .

# After (parameterized template)
- name: Build
  run: |
    docker build \
      --build-arg AGENT_ID=GL-DATA-X-014 \
      --build-arg AGENT_NAME=duplicate-detector \
      --build-arg APP_MODULE=greenlang.duplicate_detector.setup:create_app \
      --build-arg HEALTH_PATH=/api/v1/dedup/health \
      -f deployment/docker/templates/Dockerfile.agent \
      -t greenlang/duplicate-detector:${{ github.sha }} .
```
