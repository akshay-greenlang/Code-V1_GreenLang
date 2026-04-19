# Platform/Development Team Charter

**Version:** 1.0
**Date:** 2025-12-03
**Team:** Platform/Development
**Tech Lead:** TBD
**Headcount:** 4-5 engineers

---

## Team Mission

Build the core infrastructure plumbing that powers the Agent Factory ecosystem: SDK foundations, agent registry, CLI tools, and API gateways that enable seamless agent lifecycle management from generation to production deployment.

**Core Principle:** Infrastructure should be invisible to users but indispensable to operations.

---

## Team Mandate

The Platform Team owns the foundational infrastructure layer:

1. **SDK Core Infrastructure:** Authentication, logging, error handling, configuration
2. **Agent Registry:** Centralized repository for all agents (source code, Docker images, metadata)
3. **CLI Tools:** Command-line interface for agent management (generate, deploy, test, monitor)
4. **API Gateway:** RESTful APIs for all platform services with rate limiting and authentication

**Non-Goals:**
- Agent generation logic (AI/Agent Team owns this)
- ML model infrastructure (ML Platform Team owns this)
- Domain validation (Climate Science Team owns this)
- Production deployment infrastructure (DevOps Team owns this)

---

## Team Composition

### Roles & Responsibilities

**Tech Lead (1):**
- Platform architecture and design
- API contracts and versioning
- Cross-team coordination (all teams)
- Performance and scalability

**Backend Engineers (2-3):**
- Agent registry implementation
- API gateway development
- SDK core library
- Database design and optimization

**Full-Stack Engineers (2):**
- CLI tool development
- Registry UI/dashboard
- Developer portal
- Documentation site

---

## Core Responsibilities

### 1. SDK Core Infrastructure (Plumbing)

**Deliverables:**

| Component | Description | Phase |
|-----------|-------------|-------|
| **Authentication Module** | JWT-based auth for all SDK calls | Phase 1 |
| **Logging Framework** | Structured logging (JSON) for all operations | Phase 1 |
| **Error Handling** | Standardized error codes and messages | Phase 1 |
| **Configuration Management** | Environment-based config (dev, staging, prod) | Phase 1 |
| **Rate Limiting** | Token bucket algorithm for API calls | Phase 2 |
| **Caching Layer** | Redis-based caching for performance | Phase 2 |
| **Multi-Tenancy** | Tenant isolation and resource quotas | Phase 3 |

**Technical Specifications:**

**SDK Core Architecture:**
```python
# greenlang_sdk/core/__init__.py

from greenlang_sdk.core.auth import AuthClient, JWTAuth
from greenlang_sdk.core.logging import StructuredLogger
from greenlang_sdk.core.errors import (
    GreenLangError,
    ValidationError,
    AuthenticationError,
    RateLimitError
)
from greenlang_sdk.core.config import Config, Environment

__all__ = [
    "AuthClient",
    "JWTAuth",
    "StructuredLogger",
    "GreenLangError",
    "ValidationError",
    "AuthenticationError",
    "RateLimitError",
    "Config",
    "Environment",
]
```

**Authentication Module:**
```python
class AuthClient:
    """
    Authentication client for GreenLang SDK.

    Supports:
    - JWT tokens (primary)
    - API keys (legacy)
    - OAuth 2.0 (enterprise)
    """

    def __init__(self, auth_url: str, client_id: str, client_secret: str):
        self.auth_url = auth_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_cache = {}

    def authenticate(self) -> str:
        """
        Authenticate and return JWT token.

        Returns:
            JWT token (valid for 1 hour)

        Raises:
            AuthenticationError: If authentication fails
        """
        # Check cache
        if self._is_token_valid():
            return self.token_cache["access_token"]

        # Request new token
        response = requests.post(
            f"{self.auth_url}/oauth/token",
            json={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "scope": "agent:read agent:write"
            }
        )

        if response.status_code != 200:
            raise AuthenticationError("Authentication failed")

        token_data = response.json()
        self.token_cache = {
            "access_token": token_data["access_token"],
            "expires_at": time.time() + token_data["expires_in"]
        }

        return self.token_cache["access_token"]

    def _is_token_valid(self) -> bool:
        """Check if cached token is still valid."""
        if "access_token" not in self.token_cache:
            return False
        return time.time() < self.token_cache["expires_at"] - 60  # 1 min buffer
```

**Structured Logging:**
```python
class StructuredLogger:
    """
    JSON-based structured logging for all SDK operations.

    Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
    """

    def __init__(self, service_name: str, environment: str):
        self.service_name = service_name
        self.environment = environment
        self.logger = structlog.get_logger()

    def info(self, message: str, **kwargs):
        """Log info-level message."""
        self.logger.info(
            message,
            service=self.service_name,
            environment=self.environment,
            timestamp=datetime.utcnow().isoformat(),
            **kwargs
        )

    def error(self, message: str, error: Exception = None, **kwargs):
        """Log error-level message."""
        log_data = {
            "service": self.service_name,
            "environment": self.environment,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        }

        if error:
            log_data["error_type"] = type(error).__name__
            log_data["error_message"] = str(error)
            log_data["error_traceback"] = traceback.format_exc()

        self.logger.error(message, **log_data)
```

**Error Handling:**
```python
# Standardized error codes
class ErrorCode(Enum):
    # Client errors (4xx)
    VALIDATION_ERROR = "VALIDATION_ERROR"
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    AUTHORIZATION_ERROR = "AUTHORIZATION_ERROR"
    RATE_LIMIT_ERROR = "RATE_LIMIT_ERROR"
    NOT_FOUND_ERROR = "NOT_FOUND_ERROR"

    # Server errors (5xx)
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"

class GreenLangError(Exception):
    """Base exception for all GreenLang errors."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        details: dict = None,
        http_status: int = 500
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.http_status = http_status
        super().__init__(self.message)

    def to_dict(self) -> dict:
        """Convert error to JSON-serializable dict."""
        return {
            "error": {
                "code": self.error_code.value,
                "message": self.message,
                "details": self.details,
                "timestamp": datetime.utcnow().isoformat()
            }
        }

class ValidationError(GreenLangError):
    """Validation error (400)."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(
            message=message,
            error_code=ErrorCode.VALIDATION_ERROR,
            details=details,
            http_status=400
        )
```

**Success Metrics:**
- SDK adoption: 100% of agents use SDK core
- Authentication uptime: 99.95%
- Logging coverage: 100% of SDK operations logged
- Error handling: 100% of errors follow standard format

---

### 2. Agent Registry (Centralized Repository)

**Deliverables:**

| Component | Description | Phase |
|-----------|-------------|-------|
| **Registry Database** | PostgreSQL database for agent metadata | Phase 1 |
| **Registry API** | RESTful API for agent CRUD operations | Phase 1 |
| **Agent Storage** | S3-compatible storage for agent artifacts | Phase 1 |
| **Version Management** | Semantic versioning and rollback | Phase 1 |
| **Registry UI** | Web dashboard for browsing agents | Phase 2 |
| **Search & Discovery** | Full-text search for agents | Phase 2 |
| **Analytics Dashboard** | Usage metrics and trends | Phase 3 |

**Technical Specifications:**

**Agent Registry Schema:**
```sql
-- agents table
CREATE TABLE agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(100) UNIQUE NOT NULL,  -- e.g., "GL-CBAM-APP"
    name VARCHAR(255) NOT NULL,
    description TEXT,
    regulation VARCHAR(100),  -- e.g., "CBAM Regulation 2023/956"
    category VARCHAR(50),  -- e.g., "cbam", "eudr", "csrd"
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    created_by VARCHAR(255),
    status VARCHAR(50) DEFAULT 'active',  -- active, deprecated, archived
    CONSTRAINT agent_id_format CHECK (agent_id ~ '^GL-[A-Z0-9-]+$')
);

-- agent_versions table
CREATE TABLE agent_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID REFERENCES agents(id) ON DELETE CASCADE,
    version VARCHAR(20) NOT NULL,  -- e.g., "1.0.0"
    agentspec_url TEXT NOT NULL,
    source_code_url TEXT NOT NULL,
    docker_image_url TEXT NOT NULL,
    certification_status VARCHAR(50),  -- certified, pending, failed
    certification_id VARCHAR(100),
    quality_score DECIMAL(5, 2),  -- 0-100
    test_coverage DECIMAL(5, 2),  -- 0-100
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB,
    UNIQUE(agent_id, version),
    CONSTRAINT version_format CHECK (version ~ '^[0-9]+\.[0-9]+\.[0-9]+$')
);

-- agent_dependencies table
CREATE TABLE agent_dependencies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_version_id UUID REFERENCES agent_versions(id) ON DELETE CASCADE,
    dependency_type VARCHAR(50),  -- python_package, api, database
    dependency_name VARCHAR(255),
    dependency_version VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

-- agent_usage table (metrics)
CREATE TABLE agent_usage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_version_id UUID REFERENCES agent_versions(id) ON DELETE CASCADE,
    user_id VARCHAR(255),
    execution_count BIGINT DEFAULT 0,
    total_latency_ms BIGINT DEFAULT 0,
    error_count BIGINT DEFAULT 0,
    last_used_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_agents_category ON agents(category);
CREATE INDEX idx_agents_status ON agents(status);
CREATE INDEX idx_agent_versions_agent_id ON agent_versions(agent_id);
CREATE INDEX idx_agent_versions_certification ON agent_versions(certification_status);
CREATE INDEX idx_agent_usage_agent_version ON agent_usage(agent_version_id);
```

**Registry API:**
```python
# POST /v1/registry/agents
# Create new agent
{
  "agent_id": "GL-CBAM-APP",
  "name": "CBAM Compliance Agent",
  "description": "Automates CBAM compliance reporting",
  "regulation": "CBAM Regulation 2023/956",
  "category": "cbam",
  "created_by": "user@greenlang.com"
}

# Response (201 Created)
{
  "id": "uuid-123",
  "agent_id": "GL-CBAM-APP",
  "status": "active",
  "created_at": "2025-12-03T10:00:00Z"
}

# POST /v1/registry/agents/{agent_id}/versions
# Create new agent version
{
  "version": "1.0.0",
  "agentspec_url": "https://github.com/greenlang/agents/GL-CBAM-APP/agentspec.yaml",
  "source_code_url": "s3://greenlang-agents/GL-CBAM-APP/v1.0.0/src.tar.gz",
  "docker_image_url": "ghcr.io/greenlang/gl-cbam-app:1.0.0",
  "quality_score": 95.0,
  "test_coverage": 92.5,
  "metadata": {
    "generation_time_ms": 45000,
    "model_used": "claude-sonnet-4-5"
  }
}

# Response (201 Created)
{
  "id": "uuid-456",
  "agent_id": "GL-CBAM-APP",
  "version": "1.0.0",
  "certification_status": "pending",
  "created_at": "2025-12-03T10:30:00Z"
}

# GET /v1/registry/agents?category=cbam&status=active
# List agents
{
  "agents": [
    {
      "agent_id": "GL-CBAM-APP",
      "name": "CBAM Compliance Agent",
      "latest_version": "1.0.0",
      "certification_status": "certified",
      "quality_score": 95.0
    }
  ],
  "total": 1,
  "page": 1,
  "per_page": 20
}

# GET /v1/registry/agents/{agent_id}/versions/{version}
# Get specific version
{
  "agent_id": "GL-CBAM-APP",
  "version": "1.0.0",
  "agentspec_url": "...",
  "source_code_url": "...",
  "docker_image_url": "...",
  "certification_status": "certified",
  "certification_id": "CERT-CBAM-001",
  "quality_score": 95.0,
  "test_coverage": 92.5,
  "created_at": "2025-12-03T10:30:00Z"
}
```

**Success Metrics:**
- Registry uptime: 99.95%
- API response time: <100ms (p95)
- Agent search time: <500ms
- Storage cost: <$0.10 per agent per month

---

### 3. CLI Tools (Command-Line Interface)

**Deliverables:**

| Component | Description | Phase |
|-----------|-------------|-------|
| **greenlang CLI** | Main CLI tool for agent management | Phase 1 |
| **Agent Generation** | `greenlang generate` command | Phase 1 |
| **Agent Deployment** | `greenlang deploy` command | Phase 1 |
| **Agent Testing** | `greenlang test` command | Phase 2 |
| **Agent Monitoring** | `greenlang monitor` command | Phase 2 |
| **Interactive Mode** | REPL for agent exploration | Phase 3 |

**Technical Specifications:**

**CLI Command Structure:**
```bash
# Installation
pip install greenlang-cli

# Authentication
greenlang login
  --client-id <client_id>
  --client-secret <client_secret>
  --environment <dev|staging|prod>

# Generate agent from AgentSpec
greenlang generate
  --agentspec <path_or_url>
  --output <output_directory>
  --validation-level <strict|standard|relaxed>

# Deploy agent to production
greenlang deploy
  --agent-id <agent_id>
  --version <version>
  --environment <dev|staging|prod>
  --dry-run  # Preview without deploying

# Run tests on agent
greenlang test
  --agent-id <agent_id>
  --version <version>
  --test-suite <golden|regression|performance>

# Monitor agent in production
greenlang monitor
  --agent-id <agent_id>
  --metrics <latency|errors|usage>
  --period <1h|24h|7d|30d>

# List agents in registry
greenlang list
  --category <cbam|eudr|csrd>
  --status <active|deprecated|archived>
  --format <table|json|yaml>

# Get agent details
greenlang describe
  --agent-id <agent_id>
  --version <version>

# Certify agent
greenlang certify
  --agent-id <agent_id>
  --version <version>
  --regulation <cbam|eudr|csrd>
```

**CLI Implementation (Python Click):**
```python
import click
from greenlang_sdk import AgentFactory, AgentRegistry, AuthClient

@click.group()
def cli():
    """GreenLang CLI for agent management."""
    pass

@cli.command()
@click.option("--agentspec", required=True, help="Path or URL to AgentSpec file")
@click.option("--output", default="./generated", help="Output directory")
@click.option("--validation-level", default="strict", help="Validation level")
def generate(agentspec: str, output: str, validation_level: str):
    """Generate agent from AgentSpec."""
    click.echo(f"Generating agent from {agentspec}...")

    # Initialize factory
    factory = AgentFactory(
        auth=AuthClient.from_env(),
        validation_level=validation_level
    )

    # Generate
    result = factory.generate(
        agentspec_path=agentspec,
        output_dir=output
    )

    if result.success:
        click.echo(f"✓ Agent generated successfully!")
        click.echo(f"  Agent ID: {result.agent_id}")
        click.echo(f"  Version: {result.version}")
        click.echo(f"  Quality Score: {result.quality_score}")
        click.echo(f"  Output: {output}")
    else:
        click.echo(f"✗ Generation failed: {result.error}", err=True)
        exit(1)

@cli.command()
@click.option("--agent-id", required=True, help="Agent ID")
@click.option("--version", required=True, help="Agent version")
@click.option("--environment", default="staging", help="Environment")
@click.option("--dry-run", is_flag=True, help="Preview without deploying")
def deploy(agent_id: str, version: str, environment: str, dry_run: bool):
    """Deploy agent to environment."""
    click.echo(f"Deploying {agent_id} v{version} to {environment}...")

    if dry_run:
        click.echo("DRY RUN - no changes will be made")

    # Deploy logic here
    click.echo("✓ Deployment successful!")

if __name__ == "__main__":
    cli()
```

**Success Metrics:**
- CLI adoption: >80% of developers use CLI
- Command success rate: >95%
- Documentation coverage: 100% of commands
- User satisfaction (NPS): >60

---

### 4. API Gateway (RESTful APIs)

**Deliverables:**

| Component | Description | Phase |
|-----------|-------------|-------|
| **API Gateway** | Kong or AWS API Gateway for routing | Phase 1 |
| **Rate Limiting** | Per-user/tenant rate limits | Phase 1 |
| **API Documentation** | OpenAPI/Swagger docs | Phase 1 |
| **API Versioning** | v1, v2, etc. with backward compatibility | Phase 2 |
| **API Analytics** | Usage metrics and trends | Phase 2 |
| **Developer Portal** | Self-service API key management | Phase 3 |

**Technical Specifications:**

**API Gateway Routes:**
```yaml
# Agent Factory API
POST /v1/agents/generate
GET /v1/agents/{agent_id}/status

# Agent Registry API
POST /v1/registry/agents
GET /v1/registry/agents
GET /v1/registry/agents/{agent_id}
POST /v1/registry/agents/{agent_id}/versions
GET /v1/registry/agents/{agent_id}/versions/{version}

# Model API (proxied from ML Platform)
POST /v1/models/generate
GET /v1/models/{model_id}

# Validation API (proxied from Climate Science)
POST /v1/validate/cbam
POST /v1/validate/eudr
POST /v1/validate/csrd
```

**Rate Limiting:**
```yaml
rate_limits:
  free_tier:
    requests_per_minute: 10
    requests_per_day: 1000
    burst: 20

  pro_tier:
    requests_per_minute: 100
    requests_per_day: 50000
    burst: 200

  enterprise_tier:
    requests_per_minute: 1000
    requests_per_day: 1000000
    burst: 2000
```

**Success Metrics:**
- API uptime: 99.95%
- API response time: <200ms (p95)
- Rate limit accuracy: 100%
- API documentation coverage: 100%

---

## Deliverables by Phase

### Phase 1: Foundation (Weeks 1-16)

**Week 1-4: SDK Core**
- [ ] Authentication module (JWT)
- [ ] Structured logging framework
- [ ] Error handling and codes
- [ ] Configuration management

**Week 5-8: Agent Registry**
- [ ] PostgreSQL database schema
- [ ] Registry API (CRUD operations)
- [ ] S3 storage integration
- [ ] Version management

**Week 9-12: CLI Tools**
- [ ] CLI framework (Click)
- [ ] `greenlang generate` command
- [ ] `greenlang deploy` command
- [ ] `greenlang list` command

**Week 13-16: API Gateway**
- [ ] Kong API Gateway setup
- [ ] Rate limiting configuration
- [ ] OpenAPI documentation
- [ ] Authentication integration

**Phase 1 Exit Criteria:**
- [ ] SDK core published (PyPI)
- [ ] Agent registry operational (100+ agents)
- [ ] CLI installed by 50+ developers
- [ ] API uptime: 99.9%

---

### Phase 2: Production Scale (Weeks 17-28)

**Week 17-20: Advanced SDK**
- [ ] Caching layer (Redis)
- [ ] Rate limiting client-side
- [ ] Retry logic and circuit breakers
- [ ] Performance optimization

**Week 21-24: Registry Enhancements**
- [ ] Registry UI (web dashboard)
- [ ] Full-text search (Elasticsearch)
- [ ] Analytics dashboard
- [ ] API versioning (v2)

**Week 25-28: CLI & Developer Experience**
- [ ] `greenlang test` command
- [ ] `greenlang monitor` command
- [ ] Interactive mode (REPL)
- [ ] Shell auto-completion

**Phase 2 Exit Criteria:**
- [ ] Registry UI launched
- [ ] Search operational (<500ms)
- [ ] CLI adoption: >80%
- [ ] API v2 released

---

### Phase 3: Enterprise Ready (Weeks 29-40)

**Week 29-32: Multi-Tenancy**
- [ ] Tenant isolation in registry
- [ ] Resource quotas per tenant
- [ ] Tenant analytics dashboard
- [ ] RBAC for multi-user teams

**Week 33-36: Developer Portal**
- [ ] Self-service API key management
- [ ] Usage dashboards per user
- [ ] Billing integration (Stripe)
- [ ] Documentation site (docs.greenlang.com)

**Week 37-40: Scale & Optimization**
- [ ] Multi-region registry (3+ regions)
- [ ] CDN for agent artifacts
- [ ] Advanced caching strategies
- [ ] Cost optimization (<$0.10/agent/month)

**Phase 3 Exit Criteria:**
- [ ] Multi-tenancy operational
- [ ] Developer portal launched
- [ ] Multi-region deployment
- [ ] Cost per agent: <$0.10/month

---

## Success Metrics & KPIs

### North Star Metrics

| Metric | Phase 1 Target | Phase 2 Target | Phase 3 Target | Measurement |
|--------|---------------|---------------|---------------|-------------|
| **Registry Uptime** | 99.9% | 99.95% | 99.99% | Availability over 30 days |
| **API Response Time** | <200ms | <100ms | <50ms | p95 latency |
| **CLI Adoption** | 50 users | 200 users | 1,000 users | Active monthly users |
| **Agents in Registry** | 100 | 500 | 2,000 | Total registered agents |

---

## Interfaces with Other Teams

### AI/Agent Team
- Provides: Agent packages, API specs
- Receives: Registry infrastructure, CLI tools

### ML Platform Team
- Provides: API gateway for model access
- Receives: Model serving API integration

### DevOps Team
- Provides: Registry database, API specs
- Receives: Deployment infrastructure, monitoring

---

## Technical Stack

- **Backend:** Python 3.11+, FastAPI, PostgreSQL
- **Storage:** S3, Redis
- **API Gateway:** Kong
- **CLI:** Python Click
- **Frontend:** React, TypeScript (for registry UI)

---

**Document Control:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-03 | GL Product Manager | Initial Platform Team charter |

---

**Approvals:**

- Platform Tech Lead: ___________________
- Engineering Lead: ___________________
- Product Manager: ___________________
