# GL-001 ProcessHeatOrchestrator - Configuration Files Manifest

**Generated**: 2025-11-17
**Agent**: GL-001 ProcessHeatOrchestrator (Master Orchestrator)
**Purpose**: Production-grade configuration files for multi-plant process heat operations

---

## Overview

This document provides a comprehensive manifest of all configuration files created for GL-001 ProcessHeatOrchestrator, the master orchestrator for multi-plant process heat operations.

---

## Files Created (5 Total)

### 1. requirements.txt (244 lines, 8,374 bytes)

**Purpose**: Python package dependencies for GL-001 master orchestrator

**Key Dependency Categories**:
- **AI/ML**: Anthropic, OpenAI, LangChain for intelligent classification
- **Numeric/Scientific**: NumPy, SciPy, Pandas for calculations
- **Optimization**: PuLP, CVXPY, Pyomo for linear programming heat distribution
- **Databases**: asyncpg (PostgreSQL), psycopg2-binary, Redis, TimescaleDB support
- **Industrial Protocols**:
  - `asyncua==1.1.0` - OPC UA for SCADA integration
  - `pymodbus==3.6.4` - Modbus TCP for PLCs/RTUs
  - `paho-mqtt==1.6.1` - MQTT for IoT sensor data
  - `kafka-python==2.0.2` - Kafka for event streaming
- **Web Framework**: FastAPI, uvicorn for REST API
- **Security**: cryptography, PyJWT, passlib, OAuth2 (authlib)
- **Monitoring**: prometheus-client, OpenTelemetry, structlog
- **Testing**: pytest, pytest-asyncio, pytest-cov, hypothesis
- **Code Quality**: mypy, black, isort, ruff, bandit, safety

**Special Features**:
- Multi-plant SCADA integration dependencies
- ERP system integration (SAP, Oracle, Dynamics)
- Sub-agent coordination libraries
- Time series database support
- Distributed task queue (Celery)

**Total Packages**: ~90+ production and development dependencies

---

### 2. .env.template (360 lines, 13,427 bytes)

**Purpose**: Environment variables template with 80+ configuration parameters

**Configuration Sections**:

#### Agent Identification (4 variables)
- `GL_001_AGENT_ID`, `GL_001_AGENT_NAME`, `GL_001_VERSION`, `GL_001_ENVIRONMENT`

#### Database - PostgreSQL/TimescaleDB (13 variables)
- Primary database connection (host, port, credentials, SSL)
- Connection pooling parameters
- TimescaleDB-specific settings (retention, compression, chunking)
- Read replica configuration

#### Redis - Caching & Coordination (11 variables)
- Redis connection (host, port, password, SSL)
- Redis Sentinel for high availability
- Cache TTL settings for different data types

#### SCADA Integration (27 variables across 3 plants)
- **Plant 1** - OPC UA integration (11 variables)
- **Plant 2** - OPC UA integration (9 variables)
- **Plant 3** - Modbus TCP integration (5 variables)
- TLS certificates, polling intervals, authentication

#### MQTT Broker - IoT Sensor Data (9 variables)
- Broker connection, TLS certificates, QoS settings

#### ERP Integration (24 variables)
- **SAP S/4HANA** (11 variables) - OAuth2, rate limiting, batch sync
- **Oracle ERP Cloud** (7 variables) - REST API integration
- **Microsoft Dynamics 365** (6 variables) - Azure AD authentication

#### Sub-Agent Coordination (14 variables)
- GL-002 Boiler Efficiency Optimizer
- GL-003 Steam System Optimizer
- GL-004 Heat Recovery Optimizer
- GL-005 Emissions Compliance Monitor
- Parallel execution settings

#### AI/LLM Configuration (12 variables)
- Anthropic Claude (primary)
- OpenAI (fallback)
- Zero-hallucination enforcement policies

#### Optimization Parameters (7 variables)
- Linear programming settings
- Solver configuration (GLPK, CBC, CPLEX, GUROBI)
- Convergence tolerance, max iterations

#### Performance Tuning (5 variables)
- Concurrent calculations, timeouts, worker threads

#### Monitoring & Observability (9 variables)
- Prometheus metrics, OpenTelemetry tracing
- Logging configuration (level, format, rotation)

#### Alerting (11 variables)
- Email alerts, Slack webhooks, PagerDuty integration

#### Compliance & Regulations (7 variables)
- Emission limits by region (EU ETS, US EPA, China MEE, India MOEF)
- Compliance reporting, audit trail

#### Security (13 variables)
- API key authentication, JWT tokens
- TLS/SSL certificates
- Rate limiting, encryption at rest

#### Backup & Disaster Recovery (5 variables)
- Backup intervals, retention, encryption

#### Feature Flags (5 variables)
- Multi-plant optimization, predictive maintenance, analytics

#### Development & Testing (5 variables)
- Debug mode, mock integrations, profiling

#### Deployment Metadata (6 variables)
- Deployment ID, region, Kubernetes namespace

---

### 3. .gitignore (535 lines, 9,412 bytes)

**Purpose**: Prevent sensitive data and unnecessary files from version control

**Exclusion Categories**:

#### Secrets & Credentials (26 patterns)
- `.env` files (all variants)
- Certificates: `*.pem`, `*.key`, `*.cert`, `*.crt`
- Credentials: `credentials.json`, `secrets.yaml`, API keys
- SCADA certificates directory
- ERP credentials directory
- OAuth tokens, SSH keys

#### Python (60+ patterns)
- Bytecode: `__pycache__/`, `*.pyc`
- Virtual environments: `venv/`, `.venv/`, `env/`
- Build artifacts: `dist/`, `build/`, `*.egg-info/`
- Test coverage: `htmlcov/`, `.coverage`, `.pytest_cache/`

#### IDEs & Editors (30+ patterns)
- VSCode: `.vscode/`, `*.code-workspace`
- PyCharm: `.idea/`, `*.iml`
- Sublime, Vim, Emacs, Eclipse

#### Operating System (15 patterns)
- macOS: `.DS_Store`
- Windows: `Thumbs.db`, `Desktop.ini`
- Linux: `*~`, `.directory`

#### Logs & Temporary Files (20+ patterns)
- Application logs: `*.log`, `logs/`, `orchestrator.log`
- Security reports: `bandit-report.*`, `safety-report.*`

#### Data Files (30+ patterns)
- Large files: `*.csv`, `*.parquet`, `*.h5`
- SCADA sensor exports: `scada_data/`, `sensor_exports/`
- ERP exports: `erp_exports/`, `sap_exports/`
- Optimization results: `optimization_results/`

#### Backup Files (15 patterns)
- Database backups: `*.sql.gz`, `db_backups/`
- File backups: `*.bak`, `*.backup`

#### Cache Files (15 patterns)
- Application cache: `.cache/`, `calc_cache/`
- Redis dumps: `dump.rdb`

#### Build Artifacts (20 patterns)
- Terraform: `*.tfstate`, `.terraform/`
- Kubernetes: `*.yaml.bak`

#### Monitoring & Metrics (15 patterns)
- Prometheus data, performance profiles, memory dumps

#### Compliance & Audit (10 patterns)
- Audit trails (sensitive operations)
- Compliance reports (proprietary data)

#### SBOM Artifacts (5 patterns)
- `*.spdx`, `*.cdx.json`, `cyclonedx-*.json`

#### Development & Testing (20 patterns)
- Test data, benchmarks, profiling results

#### Explicitly Allowed (10 patterns)
- `.env.template`, `requirements.txt`, security reports

---

### 4. .dockerignore (369 lines, 8,635 bytes)

**Purpose**: Minimize Docker image size by excluding unnecessary files from build context

**Exclusion Categories**:

#### Version Control (5 patterns)
- `.git/`, `.github/`, `.gitlab-ci.yml`

#### Documentation (10 patterns)
- `*.md` (except `DEPLOYMENT_GUIDE.md`, `RUNBOOK.md`)
- `docs/`

#### Development & Testing (20+ patterns)
- Tests: `tests/`, `test_*.py`, `conftest.py`
- Test data: `test_data/`, `mock_data/`, `fixtures/`
- Coverage: `htmlcov/`, `.coverage`

#### IDE Files (15 patterns)
- Same as `.gitignore` (VSCode, PyCharm, Sublime, etc.)

#### Python Development (15 patterns)
- Virtual environments: `venv/`, `.venv/`
- Cache: `__pycache__/`, `.mypy_cache/`
- Build artifacts: `build/`, `dist/`

#### Secrets & Credentials (20 patterns)
- All sensitive files (same as `.gitignore`)
- **Critical**: Never include in Docker images

#### Logs & Data Files (30+ patterns)
- Application logs, SCADA sensor data, ERP exports
- Optimization results, plant operational data

#### Monitoring Data (10 patterns)
- Prometheus data, Grafana dashboards, performance profiles

#### Backup Files (10 patterns)
- Database backups, file backups

#### SBOM (5 patterns)
- Built separately from main image

#### Compliance & Audit (10 patterns)
- Sensitive operational data excluded

#### Infrastructure as Code (15 patterns)
- Terraform, Kubernetes dev files, Docker Compose

#### CI/CD (10 patterns)
- GitHub Actions, GitLab CI, Travis, CircleCI

#### Code Quality Tools (15 patterns)
- `.pre-commit-config.yaml`, `.flake8`, security reports

#### Deployment Scripts (5 patterns)
- Handled separately from runtime image

#### Development Requirements (3 patterns)
- `requirements-dev.txt` excluded
- **Included**: `requirements.txt` (production only)

#### Explicitly Included (5 patterns)
- Application code, `.env.template`, production deployment files

**Result**: Minimal production Docker image (no tests, docs, dev tools)

---

### 5. .pre-commit-config.yaml (316 lines, 10,478 bytes)

**Purpose**: Automated code quality enforcement before every commit

**Hook Categories (15 repositories)**:

#### Code Formatting (2 hooks)
- **Black** (rev: 24.1.1) - Code formatter (120 char line length)
- **isort** (rev: 5.13.2) - Import sorting

#### Linting (3 hooks)
- **Ruff** (rev: v0.2.1) - Fast Python linter with auto-fix
- **Flake8** (rev: 7.0.0) - Style guide enforcement
  - Additional: flake8-docstrings, flake8-bugbear, flake8-comprehensions

#### Type Checking (1 hook)
- **MyPy** (rev: v1.8.0) - Static type checker
  - Strict mode: `--disallow-untyped-defs`, `--check-untyped-defs`

#### Security Scanning (2 hooks)
- **Bandit** (rev: 1.7.6) - Security issue scanner
- **Detect-secrets** (rev: v1.4.0) - Credential leak prevention

#### Dependency Security (1 hook)
- **Safety** (rev: v1.3.3) - Vulnerability scanner for dependencies

#### General File Checks (15 hooks)
- Trailing whitespace, end-of-file fixer
- YAML/JSON/TOML syntax validation
- Large file detection (max 1MB)
- Merge conflict detection
- Debug statement detection
- Mixed line ending fixes

#### YAML Linting (1 hook)
- **yamllint** (rev: v1.33.0) - YAML style checker

#### Dockerfile Linting (1 hook)
- **Hadolint** (rev: v2.12.0) - Dockerfile best practices

#### Markdown Linting (1 hook)
- **markdownlint** (rev: v0.39.0) - Markdown formatting

#### Docstring Validation (1 hook)
- **pydocstyle** (rev: 6.3.0) - Google-style docstrings

#### Requirements Management (1 hook)
- **pip-compile** (rev: 7.4.0) - Requirements pinning

#### Commit Message Validation (1 hook)
- **Commitizen** (rev: 3.13.0) - Conventional commit messages

#### Complexity Check (1 hook)
- **McCabe** (rev: 0.7.0) - Cyclomatic complexity (max 10)

#### Shell Script Linting (1 hook)
- **ShellCheck** (rev: v0.9.0.6) - Shell script analyzer

**Global Settings**:
- Language: Python 3.11
- Default stage: commit
- Fail-fast: false (run all hooks)
- Minimum version: 2.20.0
- CI auto-fix enabled (weekly updates)

**Total Hooks**: 32 quality checks across 15 repositories

---

## Installation Instructions

### 1. Install Python Dependencies

```bash
cd GreenLang_2030/agent_foundation/agents/GL-001
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
# Copy template to .env
cp .env.template .env

# Edit .env with actual values
nano .env  # or vim, code, etc.
```

**Critical Variables to Set**:
- Database credentials (`DATABASE_PASSWORD`)
- Redis password (`REDIS_PASSWORD`)
- SCADA endpoints and certificates (per plant)
- ERP credentials (SAP, Oracle, Dynamics)
- Sub-agent API keys (`AGENT_GL002_API_KEY`, etc.)
- AI/LLM API keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`)
- Security keys (`JWT_SECRET_KEY`, `API_KEY_VALUE`)

### 3. Install Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually (optional)
pre-commit run --all-files
```

### 4. Build Docker Image

```bash
# Build production image
docker build -f Dockerfile.production -t greenlang/gl-001:latest .

# Verify image size (should be optimized due to .dockerignore)
docker images greenlang/gl-001:latest
```

---

## File Locations

All files are located in:
```
C:\Users\aksha\Code-V1_GreenLang\GreenLang_2030\agent_foundation\agents\GL-001\
```

**Files**:
1. `requirements.txt` - Python dependencies
2. `.env.template` - Environment variables template
3. `.gitignore` - Git exclusion patterns
4. `.dockerignore` - Docker build exclusions
5. `.pre-commit-config.yaml` - Pre-commit hooks

---

## Security Considerations

### Secrets Management

1. **NEVER commit .env to version control**
   - `.gitignore` prevents this
   - Use `.env.template` for documentation

2. **Certificate Management**
   - Store certificates outside repository
   - Use Kubernetes secrets or HashiCorp Vault
   - Reference via environment variables

3. **API Keys**
   - Rotate regularly (every 90 days)
   - Use different keys per environment (dev/staging/prod)
   - Store in secrets manager (AWS Secrets Manager, Azure Key Vault)

4. **Database Passwords**
   - Use strong passwords (32+ characters)
   - Enable SSL/TLS for all database connections
   - Rotate credentials quarterly

### Pre-commit Security

The `.pre-commit-config.yaml` enforces:
- **Bandit**: Detects common security issues in Python code
- **Detect-secrets**: Prevents credential leaks
- **Safety**: Scans for vulnerable dependencies

Run before every commit to prevent security issues.

---

## Multi-Plant Configuration

GL-001 supports multiple plant configurations:

### Plant 1 - OPC UA (Primary)
- Protocol: OPC UA over TLS
- Endpoint: `opc.tcp://opc.plant1.example.com:4840`
- Sensors: Temperature, pressure, flow rate, energy meters
- Polling: 5-second intervals

### Plant 2 - OPC UA (Secondary)
- Protocol: OPC UA over TLS
- Endpoint: `opc.tcp://opc.plant2.example.com:4840`
- Similar configuration to Plant 1

### Plant 3 - Modbus TCP
- Protocol: Modbus TCP
- Endpoint: `modbus.plant3.example.com:502`
- Legacy PLC integration

Each plant requires:
- Unique SCADA endpoint
- TLS certificates (for OPC UA)
- Sensor mappings in `.env`
- Sub-agent coordination configuration

---

## ERP Integration

Supports multiple ERP systems simultaneously:

### SAP S/4HANA (Primary)
- OAuth 2.0 authentication
- OData API integration
- Batch sync every 60 minutes
- Rate limit: 100 requests/minute

### Oracle ERP Cloud (Optional)
- REST API integration
- Basic authentication
- Custom retry logic

### Microsoft Dynamics 365 (Optional)
- Azure AD authentication
- REST API v9.2
- OAuth 2.0 with tenant-specific credentials

Configure via `.env`:
- Enable/disable per system (`ERP_SAP_ENABLED=true`)
- Set credentials and endpoints
- Configure sync intervals and batch sizes

---

## Sub-Agent Coordination

GL-001 orchestrates 4+ sub-agents:

### GL-002 - Boiler Efficiency Optimizer
- Endpoint: `https://gl002.greenlang.io/api/v1`
- Timeout: 120 seconds
- Focus: Boiler combustion optimization

### GL-003 - Steam System Optimizer
- Endpoint: `https://gl003.greenlang.io/api/v1`
- Timeout: 120 seconds
- Focus: Steam distribution and condensate return

### GL-004 - Heat Recovery Optimizer
- Endpoint: `https://gl004.greenlang.io/api/v1`
- Timeout: 120 seconds
- Focus: Waste heat recovery systems

### GL-005 - Emissions Compliance Monitor
- Endpoint: `https://gl005.greenlang.io/api/v1`
- Timeout: 90 seconds
- Focus: Real-time emissions monitoring

Configuration in `.env`:
- `AGENT_GL00X_ENABLED` - Enable/disable sub-agent
- `AGENT_GL00X_ENDPOINT` - Sub-agent API endpoint
- `AGENT_GL00X_API_KEY` - Authentication key
- `SUBAGENT_MAX_PARALLEL=10` - Max concurrent sub-agent requests

---

## Performance Optimization

### Database Connection Pooling
```env
DATABASE_POOL_MIN_SIZE=10
DATABASE_POOL_MAX_SIZE=50
DATABASE_POOL_TIMEOUT_SECONDS=30
```

### Redis Connection Pool
```env
REDIS_MAX_CONNECTIONS=100
```

### Calculation Concurrency
```env
MAX_CONCURRENT_CALCULATIONS=50
WORKER_THREADS=8
ASYNC_IO_THREADS=16
```

### Cache TTL Settings
```env
CACHE_TTL_SENSOR_DATA_SECONDS=60       # Sensor data (1 min)
CACHE_TTL_CALCULATIONS_SECONDS=300     # Calculations (5 min)
CACHE_TTL_OPTIMIZATION_SECONDS=600     # Optimization (10 min)
CACHE_TTL_ERP_DATA_SECONDS=3600        # ERP data (1 hour)
```

---

## Monitoring & Observability

### Prometheus Metrics
- Endpoint: `http://localhost:9090/metrics`
- Metrics: HTTP requests, calculation latency, SCADA health, sub-agent status

### OpenTelemetry Tracing
- Exporter: OTLP
- Endpoint: `https://otel-collector.greenlang.io:4317`
- Sampling: 100% (production)

### Logging
- Format: JSON (structured logging)
- Output: stdout + file
- Rotation: 100MB per file
- Retention: 30 days

---

## Compliance & Audit

### Emission Limits by Region
```env
EMISSION_LIMIT_EU_ETS_KG_MWH=200      # EU Emissions Trading System
EMISSION_LIMIT_US_EPA_KG_MWH=220      # US Environmental Protection Agency
EMISSION_LIMIT_CHINA_MEE_KG_MWH=210   # China Ministry of Ecology
EMISSION_LIMIT_INDIA_MOEF_KG_MWH=230  # India Ministry of Environment
```

### Audit Trail
- Retention: 365 days
- Storage: `/var/lib/greenlang/audit`
- Encryption: Enabled

### Compliance Reporting
- Frequency: Daily
- Recipients: compliance@greenlang.io, legal@greenlang.io

---

## Disaster Recovery

### Backup Configuration
```env
BACKUP_ENABLED=true
BACKUP_INTERVAL_HOURS=6
BACKUP_RETENTION_DAYS=30
BACKUP_STORAGE_PATH=/var/backups/greenlang/gl001
BACKUP_ENCRYPTION_ENABLED=true
```

### High Availability

**Redis Sentinel** (optional):
```env
REDIS_SENTINEL_ENABLED=true
REDIS_SENTINEL_HOSTS=sentinel1:26379,sentinel2:26379,sentinel3:26379
REDIS_SENTINEL_MASTER=greenlang-master
```

**Database Replica** (optional):
```env
DATABASE_REPLICA_HOST=localhost
DATABASE_REPLICA_PORT=5432
DATABASE_REPLICA_NAME=greenlang_process_heat_replica
```

---

## Next Steps

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   cp .env.template .env
   # Edit .env with actual values
   ```

3. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

4. **Run Tests**
   ```bash
   pytest tests/ -v --cov=.
   ```

5. **Build Docker Image**
   ```bash
   docker build -f Dockerfile.production -t greenlang/gl-001:latest .
   ```

6. **Deploy to Kubernetes**
   ```bash
   kubectl apply -f deployment/
   ```

---

## Related Documentation

- `DEPLOYMENT_GUIDE.md` - Kubernetes deployment instructions
- `RUNBOOK.md` - Operational procedures
- `SECURITY_AUDIT_REPORT.md` - Security assessment
- `MONITORING.md` - Observability setup

---

## Support & Contacts

- **Engineering**: engineering@greenlang.io
- **Operations**: ops@greenlang.io
- **Security**: security@greenlang.io
- **Compliance**: compliance@greenlang.io

---

**Document Version**: 1.0.0
**Last Updated**: 2025-11-17
**Maintained By**: GL-DevOpsEngineer
