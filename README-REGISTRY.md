# GreenLang Agent Registry

Production-grade agent registry infrastructure with PostgreSQL, FastAPI, Python SDK, and CLI.

## Quick Start

### 1. Start Services (Docker)

```bash
# Start all services
docker-compose -f docker-compose.dev.yml up -d

# Verify services are running
curl http://localhost:8000/health
```

### 2. Publish Your First Agent

```bash
# Set registry URL
export GL_REGISTRY_URL=http://localhost:8000

# Publish agent
gl agent publish my-agent.glpack \
  --version 1.0.0 \
  --namespace mycompany \
  --author "Your Name"

# List agents
gl agent list

# View agent details
gl agent info mycompany/my-agent
```

### 3. Use Python SDK

```python
from greenlang_sdk.registry import RegistryClient

# Initialize client
client = RegistryClient(base_url="http://localhost:8000")

# Register agent
agent = client.register(
    name="my-agent",
    namespace="mycompany",
    description="My awesome agent"
)

# Publish version
version = client.publish_version(
    agent_id=agent["id"],
    version="1.0.0",
    pack_path="/path/to/my-agent.glpack"
)

print(f"Published: {agent['name']} v{version['version']}")
```

## Architecture

```
┌─────────────────────────────────────────┐
│  CLI: gl agent publish/list/info        │
│            ↓                             │
│  SDK: RegistryClient                    │
│            ↓                             │
│  API: FastAPI + asyncpg                 │
│            ↓                             │
│  Database: PostgreSQL + Redis           │
└─────────────────────────────────────────┘
```

## Components Built

### 1. PostgreSQL Schema (`core/greenlang/registry/schema.sql`)
- `agents` table: Core agent metadata
- `agent_versions` table: Version history
- `agent_certifications` table: GL-CERT certifications
- `agent_tags` table: Searchable tags
- `agent_downloads` table: Usage analytics
- Indexes and views for performance

### 2. Registry API (`core/greenlang/registry/api.py`)
- FastAPI-based REST API
- Async PostgreSQL with asyncpg
- Full CRUD operations for agents, versions, certifications
- Health checks and readiness probes
- Pagination and filtering

**Endpoints:**
- `POST /api/v1/agents` - Register agent
- `GET /api/v1/agents` - List agents
- `GET /api/v1/agents/{id}` - Get agent details
- `POST /api/v1/agents/{id}/versions` - Publish version
- `POST /api/v1/agents/{id}/certify` - Submit certification

### 3. Registry Client (`sdks/python/greenlang_sdk/registry/client.py`)
- `RegistryClient` - Synchronous client
- `AsyncRegistryClient` - Async client for high-performance
- HTTP client with automatic retries
- Error handling and exceptions
- Type-safe responses

**Methods:**
- `register()` - Register new agent
- `publish_version()` - Publish agent version
- `list_agents()` - List agents with filters
- `get()` - Get agent details
- `certify()` - Submit certification

### 4. CLI Commands (`core/greenlang/cli/cmd_registry.py`)
- `gl agent publish <pack>` - Publish agent to registry
- `gl agent list` - List registered agents
- `gl agent info <id>` - Show agent details
- `gl agent certify <id>` - Submit certification
- Rich terminal output with tables and colors

### 5. Docker Compose (`docker-compose.dev.yml`)
- PostgreSQL 14 with automatic schema initialization
- Redis 7 for caching
- Registry API with hot reload
- pgAdmin for database management
- Redis Commander for cache inspection

## Services

| Service | Port | URL |
|---------|------|-----|
| Registry API | 8000 | http://localhost:8000 |
| API Docs | 8000 | http://localhost:8000/docs |
| PostgreSQL | 5432 | localhost:5432 |
| Redis | 6379 | localhost:6379 |
| pgAdmin | 5050 | http://localhost:5050 |
| Redis Commander | 8081 | http://localhost:8081 |

## CLI Usage Examples

```bash
# Publish agent
gl agent publish thermosync.glpack \
  --version 1.0.0 \
  --namespace greenlang \
  --author "GreenLang Team"

# List agents in namespace
gl agent list --namespace greenlang

# Search agents
gl agent list --search "temperature"

# View agent details
gl agent info greenlang/thermosync

# JSON output
gl agent info greenlang/thermosync --json

# Submit certification
gl agent certify <agent-id> \
  --version 1.0.0 \
  --dimension security \
  --status passed \
  --score 95.0
```

## Python SDK Examples

### Synchronous

```python
from greenlang_sdk.registry import RegistryClient

with RegistryClient(base_url="http://localhost:8000") as client:
    # Register agent
    agent = client.register(
        name="thermosync",
        namespace="greenlang",
        description="Temperature monitoring agent"
    )

    # Publish version
    version = client.publish_version(
        agent_id=agent["id"],
        version="1.0.0",
        pack_path="/path/to/thermosync.glpack",
        capabilities=["monitoring", "alerting"],
        dependencies=[{"name": "python", "version": ">=3.8"}]
    )

    # List agents
    agents = client.list_agents(namespace="greenlang", page=1, page_size=20)

    # Submit certification
    cert = client.certify(
        agent_id=agent["id"],
        version="1.0.0",
        dimension="security",
        status="passed",
        score=95.0,
        evidence={"tests_passed": 100, "vulnerabilities": 0}
    )
```

### Asynchronous

```python
from greenlang_sdk.registry import AsyncRegistryClient
import asyncio

async def main():
    async with AsyncRegistryClient(base_url="http://localhost:8000") as client:
        agent = await client.register(name="thermosync", namespace="greenlang")
        version = await client.publish_version(agent["id"], "1.0.0", "/path/to/pack.glpack")
        print(f"Published {agent['name']} v{version['version']}")

asyncio.run(main())
```

## Environment Configuration

```bash
# Database
export DB_HOST=localhost
export DB_PORT=5432
export DB_USER=greenlang
export DB_PASSWORD=greenlang
export DB_NAME=greenlang_registry

# Redis
export REDIS_HOST=localhost
export REDIS_PORT=6379

# Registry
export GL_REGISTRY_URL=http://localhost:8000
export GL_REGISTRY_API_KEY=your-api-key  # Optional
```

## Database Schema

### Core Tables

**agents**
- `id` (UUID) - Primary key
- `name`, `namespace` - Agent identification
- `description`, `author` - Metadata
- `spec_hash` - SHA-256 hash of spec
- `status` - active, deprecated, archived
- `created_at`, `updated_at` - Timestamps

**agent_versions**
- `id` (UUID) - Primary key
- `agent_id` (FK) - References agents
- `version` - Semantic version (1.0.0)
- `pack_path` - Path to .glpack file
- `pack_hash` - SHA-256 hash of pack
- `metadata`, `capabilities`, `dependencies` - JSONB
- `status` - published, yanked, deprecated
- `published_at` - Timestamp

**agent_certifications**
- `id` (UUID) - Primary key
- `agent_id` (FK) - References agents
- `version` - Certified version
- `dimension` - security, performance, reliability
- `status` - passed, failed, pending
- `score` - 0-100
- `evidence` - JSONB test results
- `certified_by` - Certifying authority
- `certification_date` - Timestamp

## Production Deployment

### Kubernetes

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/registry-deployment.yaml

# Scale replicas
kubectl scale deployment greenlang-registry --replicas=5

# Check status
kubectl get pods -l app=greenlang-registry
```

### Security

- Enable API key authentication
- Use SSL/TLS for database connections
- Store secrets in AWS Secrets Manager or Vault
- Deploy in private VPC with security groups
- Implement rate limiting

## Monitoring

```bash
# Health check
curl http://localhost:8000/health

# Readiness check (includes DB)
curl http://localhost:8000/api/v1/ready

# View logs
docker-compose -f docker-compose.dev.yml logs -f registry-api

# Metrics (future)
curl http://localhost:8000/metrics
```

## Development

```bash
# Start services
docker-compose -f docker-compose.dev.yml up -d

# Stop services
docker-compose -f docker-compose.dev.yml down

# Rebuild API
docker-compose -f docker-compose.dev.yml build registry-api

# View database
docker exec -it greenlang-postgres psql -U greenlang -d greenlang_registry

# Clear data and restart
docker-compose -f docker-compose.dev.yml down -v
docker-compose -f docker-compose.dev.yml up -d
```

## Documentation

- Complete Setup Guide: `docs/REGISTRY_SETUP.md`
- API Documentation: http://localhost:8000/docs (Swagger UI)
- Schema Reference: `core/greenlang/registry/schema.sql`

## Support

- Issues: https://github.com/greenlang/core/issues
- Documentation: https://docs.greenlang.io
- Email: support@greenlang.io
