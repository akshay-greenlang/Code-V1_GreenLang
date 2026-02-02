# GreenLang Agent Registry

The centralized metadata repository that treats agents as first-class, versioned, governed assets within the GreenLang Agent Factory.

## Features

- **Agent Publishing**: Register new agents and versions with semantic versioning
- **Agent Discovery**: Search and list available agents with filtering
- **Lifecycle Management**: Promote agents through draft -> experimental -> certified -> deprecated
- **Multi-tenant Support**: Tenant isolation for enterprise deployments
- **Audit Trail**: Complete audit logging of all registry operations
- **Governance Policies**: Configurable policies per tenant

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.10+ (for local development)
- PostgreSQL 14+ (or use Docker)

### Running with Docker

```bash
# Start all services (PostgreSQL, Redis, API)
docker-compose up -d

# Run database migrations
docker-compose --profile migrations run --rm migrations

# View logs
docker-compose logs -f registry-api

# Stop all services
docker-compose down
```

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# Set environment variables
export DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/greenlang_registry"

# Run migrations
alembic upgrade head

# Start development server
uvicorn greenlang_registry.api.app:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/registry/agents` | Publish a new agent version |
| GET | `/api/v1/registry/agents` | List agents (paginated) |
| GET | `/api/v1/registry/agents/{id}` | Get agent details |
| POST | `/api/v1/registry/agents/{id}/promote` | Promote agent state |

### Health Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Basic health check |
| GET | `/health/ready` | Readiness check with DB |
| GET | `/health/live` | Liveness probe |

### API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Usage Examples

### Publish an Agent

```bash
curl -X POST "http://localhost:8000/api/v1/registry/agents" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "gl-cbam-calculator",
    "name": "CBAM Carbon Calculator",
    "description": "Calculates embedded carbon for CBAM shipments",
    "version": "1.0.0",
    "domain": "sustainability.cbam",
    "type": "calculator",
    "team": "cbam-team",
    "container_image": "gcr.io/greenlang/cbam-calculator:1.0.0"
  }'
```

### List Agents

```bash
# List all agents
curl "http://localhost:8000/api/v1/registry/agents"

# Filter by domain
curl "http://localhost:8000/api/v1/registry/agents?domain=sustainability.cbam"

# Search by name
curl "http://localhost:8000/api/v1/registry/agents?search=carbon"
```

### Get Agent Details

```bash
curl "http://localhost:8000/api/v1/registry/agents/gl-cbam-calculator"
```

### Promote Agent

```bash
# Promote to experimental
curl -X POST "http://localhost:8000/api/v1/registry/agents/gl-cbam-calculator/promote" \
  -H "Content-Type: application/json" \
  -d '{
    "target_state": "experimental",
    "reason": "Ready for testing",
    "promoted_by": "qa-team"
  }'
```

## Lifecycle States

| State | Description | Valid Transitions |
|-------|-------------|-------------------|
| `draft` | Initial state for new agents | -> experimental |
| `experimental` | Ready for limited testing | -> certified, deprecated |
| `certified` | Production-ready, fully tested | -> deprecated |
| `deprecated` | Scheduled for removal | (none) |

## Database Schema

The registry uses PostgreSQL with 7 core tables:

1. **agents**: Core agent metadata
2. **agent_versions**: Versioned agent releases
3. **evaluation_results**: Agent evaluation data
4. **state_transitions**: Lifecycle state audit trail
5. **usage_metrics**: Usage analytics
6. **audit_logs**: Comprehensive audit logging
7. **governance_policies**: Tenant governance rules

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+asyncpg://postgres:postgres@localhost:5432/greenlang_registry` |
| `DB_POOL_SIZE` | Connection pool size | `20` |
| `DB_MAX_OVERFLOW` | Max overflow connections | `10` |
| `DB_ECHO` | Enable SQL logging | `false` |
| `CORS_ORIGINS` | Allowed CORS origins | `*` |
| `LOG_LEVEL` | Logging level | `info` |
| `PORT` | API port | `8000` |
| `WORKERS` | Gunicorn workers | `4` |

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=greenlang_registry --cov-report=html

# Run specific test file
pytest tests/test_api.py -v

# Run specific test
pytest tests/test_api.py::TestPublishAgent::test_publish_new_agent_success -v
```

## Project Structure

```
greenlang_registry/
├── api/
│   ├── __init__.py
│   ├── app.py              # FastAPI application
│   └── routes.py           # API route handlers
├── db/
│   ├── __init__.py
│   ├── client.py           # Async database client
│   └── models.py           # SQLAlchemy models
├── migrations/
│   ├── env.py              # Alembic configuration
│   └── versions/           # Migration scripts
├── tests/
│   ├── conftest.py         # Test fixtures
│   ├── test_api.py         # API endpoint tests
│   └── test_models.py      # Pydantic model tests
├── __init__.py
├── models.py               # Pydantic request/response models
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
└── alembic.ini
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
