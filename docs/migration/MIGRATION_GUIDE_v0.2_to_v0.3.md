# GreenLang Migration Guide: v0.2 to v0.3

**Document Version:** 1.0.0
**Last Updated:** 2025-11-08
**Target Audience:** Developers, DevOps Engineers, System Administrators
**Estimated Migration Time:** 2-4 hours for small deployments, 1-2 days for enterprise deployments

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Breaking Changes](#breaking-changes)
3. [Migration Steps](#migration-steps)
4. [Code Examples (Before/After)](#code-examples-beforeafter)
5. [Troubleshooting Guide](#troubleshooting-guide)
6. [Rollback Procedures](#rollback-procedures)
7. [Support Contacts](#support-contacts)

---

## Executive Summary

### What's New in v0.3

GreenLang v0.3 represents a major architectural upgrade with the following key improvements:

- **AgentSpec v2**: Enhanced agent specification with improved type safety and validation
- **Async Agents**: Full support for asynchronous agent execution
- **GraphQL API (Beta)**: Modern GraphQL API alongside REST
- **Enhanced Security**: MFA, encryption at rest, and improved audit logging
- **Performance**: 40% faster workflow execution through async processing
- **Breaking Changes**: 12 breaking changes requiring code updates

### Migration Complexity Matrix

| Component | Complexity | Est. Time | Risk Level |
|-----------|-----------|-----------|------------|
| Agent Packs (AgentSpec v1 → v2) | HIGH | 2-4 hours | MEDIUM |
| API Endpoints | MEDIUM | 1-2 hours | LOW |
| Configuration Files | LOW | 30 minutes | LOW |
| Database Schema | MEDIUM | 1 hour | MEDIUM |
| Workflows | MEDIUM | 1-2 hours | LOW |
| Custom Agents | HIGH | 4-8 hours | HIGH |

### Critical Decision Points

1. **Backward Compatibility**: v0.3 supports AgentSpec v1 in deprecated mode until v0.4
2. **Database Migration**: Requires downtime (estimated 5-15 minutes)
3. **API Compatibility**: REST API v2 endpoints change (v1 endpoints remain available)
4. **Python Version**: Minimum Python 3.9 (up from 3.8)

---

## Breaking Changes

### BC-001: AgentSpec v2 Schema Changes

**Impact:** HIGH - Affects all custom agent packs

**What Changed:**
- `pack.yaml` schema now requires `spec_version: "2.0"`
- `inputs` and `outputs` now require explicit type definitions
- `capabilities` field moved from root to `metadata.capabilities`
- `dependencies` format changed to support version constraints

**Migration Required:**
```yaml
# OLD (v0.2 - AgentSpec v1)
spec_version: "1.0"
name: my-agent
inputs:
  - fuel_type
  - consumption

# NEW (v0.3 - AgentSpec v2)
spec_version: "2.0"
name: my-agent
inputs:
  fuel_type:
    type: string
    required: true
  consumption:
    type: number
    required: true
```

**Timeline:**
- v0.3: AgentSpec v1 supported with deprecation warnings
- v0.4: AgentSpec v1 removed (Q2 2026)

---

### BC-002: API Endpoint Changes

**Impact:** MEDIUM - Affects API consumers

**What Changed:**
- `/api/v1/agents/register` → `/api/v2/agents`
- `/api/v1/workflows/execute` → `/api/v2/workflows/execute`
- Response format now includes `metadata` envelope
- Authentication now requires JWT tokens (API keys deprecated)

**Migration Required:**
```python
# OLD (v0.2)
response = requests.post(
    "http://localhost:8000/api/v1/agents/register",
    json={"name": "my-agent"},
    headers={"X-API-Key": api_key}
)

# NEW (v0.3)
response = requests.post(
    "http://localhost:8000/api/v2/agents",
    json={
        "data": {"name": "my-agent"},
        "metadata": {"client_version": "0.3.0"}
    },
    headers={"Authorization": f"Bearer {jwt_token}"}
)
```

**Timeline:**
- v0.3: `/api/v1/` endpoints available with deprecation headers
- v0.4: `/api/v1/` endpoints removed (Q2 2026)

---

### BC-003: Configuration Format Changes

**Impact:** LOW - Affects `greenlang.yaml` configuration

**What Changed:**
- `database.url` → `database.connection_string`
- `security.encryption` now required (was optional)
- `logging.level` must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL

**Migration Required:**
```yaml
# OLD (v0.2)
database:
  url: postgresql://localhost/greenlang
security:
  enabled: true
logging:
  level: info

# NEW (v0.3)
database:
  connection_string: postgresql://localhost/greenlang
  pool_size: 10
security:
  enabled: true
  encryption:
    algorithm: AES-256-GCM
    key_rotation_days: 90
logging:
  level: INFO  # Must be uppercase
  format: json
```

---

### BC-004: Database Schema Changes

**Impact:** MEDIUM - Requires Alembic migration

**What Changed:**
- New tables: `audit_logs`, `user_sessions`, `agent_versions`
- Modified tables: `agents` (added `spec_version` column), `workflows` (added `async_enabled` column)
- Dropped columns: `agents.legacy_format`, `workflows.deprecated_config`

**Migration Required:**
```bash
# Automatic via CLI tool
greenlang migrate execute

# Manual via Alembic
alembic upgrade head
```

**Schema Changes:**
```sql
-- Added columns
ALTER TABLE agents ADD COLUMN spec_version VARCHAR(10) DEFAULT '1.0';
ALTER TABLE workflows ADD COLUMN async_enabled BOOLEAN DEFAULT FALSE;

-- New tables
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    action VARCHAR(255) NOT NULL,
    resource_type VARCHAR(100),
    resource_id UUID,
    timestamp TIMESTAMP DEFAULT NOW(),
    ip_address VARCHAR(45),
    metadata JSONB
);

CREATE TABLE user_sessions (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id),
    token_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP NOT NULL,
    ip_address VARCHAR(45),
    user_agent TEXT
);
```

---

### BC-005: Workflow Execution Context

**Impact:** MEDIUM - Affects custom workflow executors

**What Changed:**
- `ExecutionContext` now requires `async_mode` parameter
- `context.data` is now immutable (use `context.set_data()`)
- `context.agent_results` changed from dict to ordered list

**Migration Required:**
```python
# OLD (v0.2)
context = ExecutionContext(data=input_data)
context.data["new_field"] = value

# NEW (v0.3)
context = ExecutionContext(data=input_data, async_mode=True)
context.set_data("new_field", value)
```

---

### BC-006: Agent Registration

**Impact:** MEDIUM - Affects programmatic agent registration

**What Changed:**
- `Orchestrator.register_agent()` now validates AgentSpec v2
- Agent ID must be unique across all tenants
- Agent metadata now required (name, version, description)

**Migration Required:**
```python
# OLD (v0.2)
orchestrator.register_agent("my_agent", MyAgent())

# NEW (v0.3)
orchestrator.register_agent(
    agent_id="my_agent",
    agent_instance=MyAgent(),
    metadata={
        "name": "My Custom Agent",
        "version": "1.0.0",
        "description": "Custom agent for processing",
        "spec_version": "2.0"
    }
)
```

---

### BC-007: Environment Variables

**Impact:** LOW - Affects deployment configuration

**What Changed:**
- `GREENLANG_ENV` → `GL_ENV`
- `GREENLANG_DB_URL` → `GL_DATABASE_URL`
- New required: `GL_SECRET_KEY` (for encryption)

**Migration Required:**
```bash
# OLD (v0.2)
export GREENLANG_ENV=production
export GREENLANG_DB_URL=postgresql://localhost/greenlang

# NEW (v0.3)
export GL_ENV=production
export GL_DATABASE_URL=postgresql://localhost/greenlang
export GL_SECRET_KEY=$(openssl rand -hex 32)
export GL_ENCRYPTION_KEY=$(openssl rand -base64 32)
```

---

### BC-008: Python Package Dependencies

**Impact:** LOW - Affects requirements.txt

**What Changed:**
- Minimum Python version: 3.9 (was 3.8)
- `aiohttp>=3.9.0` now required (for async agents)
- `cryptography>=41.0.0` now required (for encryption)
- `pydantic>=2.0.0` (was 1.10.x)

**Migration Required:**
```txt
# OLD (v0.2 requirements.txt)
greenlang==0.2.0
pydantic==1.10.12
python>=3.8

# NEW (v0.3 requirements.txt)
greenlang==0.3.0
pydantic>=2.0.0,<3.0.0
aiohttp>=3.9.0
cryptography>=41.0.0
python>=3.9
```

---

### BC-009: Agent Result Format

**Impact:** MEDIUM - Affects agent output parsing

**What Changed:**
- Agent results now wrapped in `AgentResult` dataclass
- `.success` is now `.status` (enum: SUCCESS, FAILURE, PARTIAL)
- `.data` and `.errors` are now `.output` and `.error_details`

**Migration Required:**
```python
# OLD (v0.2)
result = agent.run(input_data)
if result.success:
    data = result.data
else:
    errors = result.errors

# NEW (v0.3)
result = agent.run(input_data)
if result.status == AgentStatus.SUCCESS:
    data = result.output
else:
    errors = result.error_details
```

---

### BC-010: Logging Format

**Impact:** LOW - Affects log parsing

**What Changed:**
- Default log format changed to JSON (was plain text)
- Log levels must be uppercase
- Structured logging with correlation IDs

**Migration Required:**
```python
# OLD (v0.2) - Plain text logs
2025-11-08 10:30:45 INFO Agent executed successfully

# NEW (v0.3) - JSON logs
{
  "timestamp": "2025-11-08T10:30:45.123Z",
  "level": "INFO",
  "message": "Agent executed successfully",
  "correlation_id": "abc-123-def-456",
  "agent_id": "fuel_agent",
  "duration_ms": 45
}
```

---

### BC-011: CLI Command Changes

**Impact:** LOW - Affects CLI users

**What Changed:**
- `gl run` now requires `--backend` flag (default: local)
- `gl agent register` → `gl agents register`
- `gl workflow execute` → `gl workflows execute`

**Migration Required:**
```bash
# OLD (v0.2)
gl run workflow.yaml --input data.json
gl agent register pack.yaml

# NEW (v0.3)
gl run workflow.yaml --input data.json --backend local
gl agents register pack.yaml
```

---

### BC-012: Async/Sync Agent Execution

**Impact:** HIGH - Affects custom agent development

**What Changed:**
- Agents can now be `async` or `sync`
- Default changed from sync-only to async-first
- Must explicitly opt-out of async with `@sync_only` decorator

**Migration Required:**
```python
# OLD (v0.2) - All agents were sync
class MyAgent(BaseAgent):
    def run(self, input_data):
        return self.process(input_data)

# NEW (v0.3) - Async by default
class MyAgent(BaseAgent):
    async def run(self, input_data):
        return await self.process(input_data)

# OR opt-out of async
from greenlang.agents import sync_only

@sync_only
class MyLegacyAgent(BaseAgent):
    def run(self, input_data):
        return self.process(input_data)
```

---

## Migration Steps

### Pre-Migration Checklist

Before starting the migration, ensure:

- [ ] Backup all databases (see [Backup Procedure](#backup-procedure))
- [ ] Review all breaking changes relevant to your deployment
- [ ] Test migration on staging environment first
- [ ] Schedule maintenance window (15-30 minutes recommended)
- [ ] Notify users of planned downtime
- [ ] Verify Python 3.9+ is installed
- [ ] Ensure all dependencies are compatible
- [ ] Review custom agent code for async compatibility

---

### Step 1: Backup Existing Data

**Duration:** 5-10 minutes
**Risk:** LOW
**Rollback Impact:** CRITICAL (required for rollback)

#### Database Backup

```bash
# PostgreSQL backup
pg_dump -h localhost -U greenlang_user greenlang > backup_v0.2_$(date +%Y%m%d_%H%M%S).sql

# MySQL backup
mysqldump -u greenlang_user -p greenlang > backup_v0.2_$(date +%Y%m%d_%H%M%S).sql

# SQLite backup
cp greenlang.db greenlang_v0.2_backup_$(date +%Y%m%d_%H%M%S).db
```

#### Configuration Backup

```bash
# Backup configuration files
cp greenlang.yaml greenlang_v0.2.yaml.backup
cp .env .env.v0.2.backup

# Backup agent packs
tar -czf agent_packs_v0.2_backup.tar.gz ~/.greenlang/packs/

# Backup workflows
tar -czf workflows_v0.2_backup.tar.gz ./workflows/
```

#### Verification

```bash
# Verify backups exist
ls -lh backup_v0.2_*.sql
ls -lh greenlang_v0.2.yaml.backup
ls -lh agent_packs_v0.2_backup.tar.gz
```

---

### Step 2: Update Dependencies

**Duration:** 10-15 minutes
**Risk:** LOW
**Rollback Impact:** MEDIUM

#### Update Python Version (if needed)

```bash
# Check current Python version
python --version

# If < 3.9, install Python 3.9+
# Ubuntu/Debian
sudo apt update
sudo apt install python3.9 python3.9-venv

# macOS (Homebrew)
brew install python@3.9

# Windows (download from python.org)
# https://www.python.org/downloads/
```

#### Update GreenLang Package

```bash
# Create new virtual environment with Python 3.9+
python3.9 -m venv venv_v0.3
source venv_v0.3/bin/activate  # Linux/macOS
# OR
venv_v0.3\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip

# Install GreenLang v0.3
pip install greenlang==0.3.0

# Verify installation
greenlang --version
# Expected output: greenlang v0.3.0
```

#### Update Dependencies

```bash
# If using requirements.txt
pip install -r requirements_v0.3.txt

# If using pip directly
pip install \
  "greenlang==0.3.0" \
  "pydantic>=2.0.0,<3.0.0" \
  "aiohttp>=3.9.0" \
  "cryptography>=41.0.0" \
  "alembic>=1.12.0"

# Verify dependencies
pip check
```

---

### Step 3: Run Migration Scripts

**Duration:** 5-15 minutes
**Risk:** MEDIUM
**Rollback Impact:** HIGH (requires database restore)

#### Automated Migration (Recommended)

```bash
# Analyze current installation
greenlang migrate analyze
# Output: Detected version: 0.2.0, target: 0.3.0

# Generate migration plan
greenlang migrate plan --output migration_plan.json
# Review migration_plan.json

# Execute migration (dry-run first)
greenlang migrate execute --dry-run
# Review output for issues

# Execute actual migration
greenlang migrate execute
# Output:
# [1/7] Backing up database... ✓
# [2/7] Updating database schema... ✓
# [3/7] Migrating configuration files... ✓
# [4/7] Converting agent packs... ✓
# [5/7] Updating workflows... ✓
# [6/7] Running validation tests... ✓
# [7/7] Migration complete! ✓

# Verify migration success
greenlang migrate verify
```

#### Manual Migration

If automated migration fails or is unavailable:

```bash
# 1. Run Alembic migrations
alembic upgrade head

# 2. Update configuration manually
# Edit greenlang.yaml (see BC-003)

# 3. Convert agent packs
python -m greenlang.tools.convert_agentspec_v1_to_v2 \
  --input ~/.greenlang/packs/ \
  --output ~/.greenlang/packs_v2/

# 4. Update environment variables
# Edit .env file (see BC-007)

# 5. Restart services
systemctl restart greenlang  # Linux
# OR
docker-compose restart greenlang  # Docker
```

---

### Step 4: Update Agent pack.yaml Files

**Duration:** 1-2 hours (depending on number of custom agents)
**Risk:** MEDIUM
**Rollback Impact:** LOW (can revert files)

#### Automated Conversion

```bash
# Convert all agent packs to AgentSpec v2
greenlang agents convert \
  --input ~/.greenlang/packs/ \
  --output ~/.greenlang/packs_v2/ \
  --backup

# Validate converted packs
greenlang agents validate ~/.greenlang/packs_v2/*/pack.yaml
```

#### Manual Conversion

For each `pack.yaml` file:

```yaml
# BEFORE (AgentSpec v1)
spec_version: "1.0"
name: fuel-calculator
version: "1.0.0"
description: Calculate emissions from fuel consumption
author: GreenLang Team
license: MIT

inputs:
  - fuel_type
  - consumption
  - unit

outputs:
  - co2e_emissions_kg
  - fuel_type
  - consumption

capabilities:
  - network_egress
  - file_read

dependencies:
  - numpy
  - pandas

# AFTER (AgentSpec v2)
spec_version: "2.0"
metadata:
  name: fuel-calculator
  version: "1.0.0"
  description: Calculate emissions from fuel consumption
  author: GreenLang Team
  license: MIT
  capabilities:
    - network_egress
    - file_read
  tags:
    - emissions
    - fuel
    - calculator

inputs:
  fuel_type:
    type: string
    description: Type of fuel consumed
    required: true
    enum:
      - electricity
      - natural_gas
      - diesel
      - gasoline
  consumption:
    type: number
    description: Amount of fuel consumed
    required: true
    minimum: 0
  unit:
    type: string
    description: Unit of measurement
    required: true
    enum:
      - kWh
      - therms
      - liters
      - gallons

outputs:
  co2e_emissions_kg:
    type: number
    description: CO2 equivalent emissions in kilograms
  fuel_type:
    type: string
    description: Fuel type (echoed from input)
  consumption:
    type: number
    description: Consumption amount (echoed from input)

dependencies:
  - name: numpy
    version: ">=1.24.0,<2.0.0"
  - name: pandas
    version: ">=2.0.0,<3.0.0"
```

#### Validation

```bash
# Validate each converted pack
greenlang agents validate pack.yaml

# Expected output:
# ✓ Schema validation passed
# ✓ All required fields present
# ✓ Input/output types valid
# ✓ Dependencies resolved
```

---

### Step 5: Update Configuration Files

**Duration:** 15-30 minutes
**Risk:** LOW
**Rollback Impact:** LOW

#### Update greenlang.yaml

```yaml
# BEFORE (v0.2)
version: "0.2.0"

database:
  url: postgresql://user:pass@localhost/greenlang
  echo: false

security:
  enabled: true
  api_keys_enabled: true

logging:
  level: info
  file: greenlang.log

# AFTER (v0.3)
version: "0.3.0"

database:
  connection_string: postgresql://user:pass@localhost/greenlang
  pool_size: 10
  max_overflow: 20
  pool_timeout: 30
  pool_recycle: 3600

security:
  enabled: true
  mfa_required: false  # Enable for production
  jwt_secret: ${GL_JWT_SECRET}
  jwt_expiry_hours: 24
  encryption:
    algorithm: AES-256-GCM
    key: ${GL_ENCRYPTION_KEY}
    key_rotation_days: 90
  session_timeout_minutes: 30

logging:
  level: INFO  # Must be uppercase
  format: json  # Changed from plain text
  file: logs/greenlang.log
  max_size_mb: 100
  backup_count: 10
  structured: true
  correlation_id_enabled: true

async:
  enabled: true
  max_workers: 10
  timeout_seconds: 300
```

#### Update .env File

```bash
# BEFORE (v0.2)
GREENLANG_ENV=production
GREENLANG_DB_URL=postgresql://localhost/greenlang
API_KEY=your-api-key-here

# AFTER (v0.3)
GL_ENV=production
GL_DATABASE_URL=postgresql://localhost/greenlang
GL_SECRET_KEY=<generate with: openssl rand -hex 32>
GL_ENCRYPTION_KEY=<generate with: openssl rand -base64 32>
GL_JWT_SECRET=<generate with: openssl rand -hex 32>
GL_API_KEYS_ENABLED=false  # Deprecated in favor of JWT
GL_ASYNC_ENABLED=true
GL_LOG_LEVEL=INFO
```

---

### Step 6: Test Workflows

**Duration:** 30-60 minutes
**Risk:** LOW
**Rollback Impact:** LOW

#### Run Test Workflows

```bash
# 1. Test simple workflow
greenlang run tests/workflows/simple.yaml \
  --input tests/data/simple_input.json \
  --backend local

# Expected output:
# ✓ Workflow completed successfully
# ✓ All agents executed
# ✓ Results validated

# 2. Test async workflow
greenlang run tests/workflows/async_test.yaml \
  --input tests/data/async_input.json \
  --backend local

# 3. Test API endpoints
curl -X POST http://localhost:8000/api/v2/workflows/execute \
  -H "Authorization: Bearer ${JWT_TOKEN}" \
  -H "Content-Type: application/json" \
  -d @tests/data/workflow_request.json

# 4. Test agent registration
greenlang agents register tests/packs/test-pack/pack.yaml

# 5. Run integration tests
pytest tests/integration/test_migration.py -v
```

#### Validate Results

```bash
# Compare v0.2 vs v0.3 outputs
diff -u \
  <(cat baseline_v0.2_output.json | jq -S .) \
  <(cat current_v0.3_output.json | jq -S .)

# Check for regressions
greenlang migrate verify --compare-baseline baseline_v0.2/
```

---

### Step 7: Deploy to Production

**Duration:** 15-30 minutes
**Risk:** HIGH
**Rollback Impact:** CRITICAL

#### Pre-Deployment Checklist

- [ ] All tests passing on staging
- [ ] Backup verified and accessible
- [ ] Rollback plan documented
- [ ] Monitoring alerts configured
- [ ] Team notified of deployment
- [ ] Maintenance window scheduled

#### Deployment Steps

```bash
# 1. Enable maintenance mode
greenlang admin maintenance enable \
  --message "Upgrading to v0.3.0. Expected downtime: 15 minutes"

# 2. Stop current services
systemctl stop greenlang-api
systemctl stop greenlang-workers

# OR for Docker
docker-compose stop

# 3. Pull new version
git pull origin v0.3.0
# OR
docker pull greenlang/greenlang:0.3.0

# 4. Run migration
greenlang migrate execute --production

# 5. Start services
systemctl start greenlang-api
systemctl start greenlang-workers

# OR for Docker
docker-compose up -d

# 6. Verify health
greenlang admin health-check

# Expected output:
# ✓ Database: Connected
# ✓ API: Responding
# ✓ Workers: Active (10/10)
# ✓ Agents: Loaded (25/25)

# 7. Disable maintenance mode
greenlang admin maintenance disable

# 8. Monitor logs
tail -f logs/greenlang.log | jq .
```

#### Post-Deployment Validation

```bash
# Run smoke tests
pytest tests/smoke/ -v

# Check API health
curl http://localhost:8000/health

# Verify agents loaded
greenlang agents list

# Check workflow execution
greenlang run tests/workflows/smoke_test.yaml

# Monitor metrics
greenlang admin metrics --watch
```

---

## Code Examples (Before/After)

### Example 1: Agent Registration

```python
# BEFORE (v0.2)
from greenlang.core.orchestrator import Orchestrator
from greenlang.agents import FuelAgent

orchestrator = Orchestrator()
orchestrator.register_agent("fuel", FuelAgent())

# AFTER (v0.3)
from greenlang.core.orchestrator import Orchestrator
from greenlang.agents import FuelAgent

orchestrator = Orchestrator()
orchestrator.register_agent(
    agent_id="fuel",
    agent_instance=FuelAgent(),
    metadata={
        "name": "Fuel Emissions Agent",
        "version": "2.0.0",
        "spec_version": "2.0",
        "description": "Calculates emissions from fuel consumption"
    }
)
```

---

### Example 2: Workflow Execution

```python
# BEFORE (v0.2)
from greenlang.core.workflow import Workflow

workflow = Workflow.from_yaml("workflow.yaml")
result = workflow.execute(input_data)

if result["success"]:
    print(f"Total emissions: {result['data']['total_emissions']}")

# AFTER (v0.3)
from greenlang.core.workflow import Workflow
from greenlang.core.context import ExecutionContext
from greenlang.core.types import AgentStatus

workflow = Workflow.from_yaml("workflow.yaml")
context = ExecutionContext(
    data=input_data,
    async_mode=True,
    timeout_seconds=300
)

result = await workflow.execute(context)  # Note: async

if result.status == AgentStatus.SUCCESS:
    print(f"Total emissions: {result.output['total_emissions']}")
else:
    print(f"Errors: {result.error_details}")
```

---

### Example 3: API Calls

```python
# BEFORE (v0.2)
import requests

response = requests.post(
    "http://localhost:8000/api/v1/workflows/execute",
    json={
        "workflow_id": "emissions-calc",
        "input": {"fuel_type": "electricity", "consumption": 1000}
    },
    headers={"X-API-Key": "your-api-key"}
)

data = response.json()

# AFTER (v0.3)
import requests

# 1. Get JWT token first
auth_response = requests.post(
    "http://localhost:8000/api/v2/auth/token",
    json={"username": "user", "password": "pass"}
)
jwt_token = auth_response.json()["data"]["access_token"]

# 2. Execute workflow
response = requests.post(
    "http://localhost:8000/api/v2/workflows/execute",
    json={
        "data": {
            "workflow_id": "emissions-calc",
            "input": {"fuel_type": "electricity", "consumption": 1000}
        },
        "metadata": {
            "client_version": "0.3.0",
            "correlation_id": "abc-123"
        }
    },
    headers={"Authorization": f"Bearer {jwt_token}"}
)

# Response is now wrapped in envelope
data = response.json()["data"]
metadata = response.json()["metadata"]
```

---

### Example 4: Configuration Loading

```python
# BEFORE (v0.2)
from greenlang.config import load_config

config = load_config("greenlang.yaml")
db_url = config["database"]["url"]

# AFTER (v0.3)
from greenlang.config import load_config

config = load_config("greenlang.yaml")
db_url = config["database"]["connection_string"]

# Access encryption settings (new in v0.3)
encryption_key = config["security"]["encryption"]["key"]
key_rotation_days = config["security"]["encryption"]["key_rotation_days"]
```

---

### Example 5: Custom Agent Development

```python
# BEFORE (v0.2)
from greenlang.agents.base import BaseAgent

class MyCustomAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_id="my_custom_agent",
            name="My Custom Agent",
            version="1.0.0"
        )

    def run(self, input_data):
        # Synchronous processing
        result = self.process_data(input_data)
        return {
            "success": True,
            "data": result
        }

# AFTER (v0.3)
from greenlang.agents.base import BaseAgent
from greenlang.core.types import AgentResult, AgentStatus
from typing import Dict, Any

class MyCustomAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_id="my_custom_agent",
            metadata={
                "name": "My Custom Agent",
                "version": "1.0.0",
                "spec_version": "2.0",
                "description": "Custom agent for data processing"
            }
        )

    async def run(self, input_data: Dict[str, Any]) -> AgentResult:
        # Asynchronous processing
        try:
            result = await self.process_data_async(input_data)
            return AgentResult(
                status=AgentStatus.SUCCESS,
                output=result,
                metadata={
                    "execution_time_ms": 123,
                    "agent_version": "1.0.0"
                }
            )
        except Exception as e:
            return AgentResult(
                status=AgentStatus.FAILURE,
                output={},
                error_details={"error": str(e)},
                metadata={}
            )

    async def process_data_async(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Async processing logic
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.example.com/data") as response:
                return await response.json()
```

---

### Example 6: Database Queries

```python
# BEFORE (v0.2)
from greenlang.db import get_database

db = get_database()
agents = db.query("SELECT * FROM agents WHERE status = 'active'")

# AFTER (v0.3)
from greenlang.db import get_database_session
from greenlang.models import Agent
from sqlalchemy import select

async with get_database_session() as session:
    stmt = select(Agent).where(Agent.status == "active")
    result = await session.execute(stmt)
    agents = result.scalars().all()
```

---

## Troubleshooting Guide

### Issue 1: Migration Script Fails

**Symptoms:**
```
Error: Migration failed at step 3/7
Database schema update failed: column 'spec_version' already exists
```

**Cause:** Partial migration from previous attempt

**Solution:**
```bash
# 1. Restore from backup
pg_restore -d greenlang backup_v0.2_20251108.sql

# 2. Clean migration state
greenlang migrate reset

# 3. Re-run migration
greenlang migrate execute --force
```

---

### Issue 2: AgentSpec v1 Validation Errors

**Symptoms:**
```
Error: Invalid AgentSpec v2: Missing required field 'metadata'
File: ~/.greenlang/packs/custom-agent/pack.yaml
```

**Cause:** Agent pack not properly converted to v2

**Solution:**
```bash
# Auto-convert the pack
greenlang agents convert \
  --input ~/.greenlang/packs/custom-agent/ \
  --output ~/.greenlang/packs/custom-agent-v2/

# Validate conversion
greenlang agents validate ~/.greenlang/packs/custom-agent-v2/pack.yaml
```

---

### Issue 3: API Authentication Failures

**Symptoms:**
```
401 Unauthorized: Invalid or expired token
```

**Cause:** Using deprecated API key instead of JWT token

**Solution:**
```python
# Generate JWT token
import requests

response = requests.post(
    "http://localhost:8000/api/v2/auth/token",
    json={
        "username": "your_username",
        "password": "your_password"
    }
)

token = response.json()["data"]["access_token"]

# Use token in subsequent requests
headers = {"Authorization": f"Bearer {token}"}
```

---

### Issue 4: Async Agent Execution Errors

**Symptoms:**
```
RuntimeError: This agent requires async execution but was called synchronously
```

**Cause:** Agent was marked async but called without `await`

**Solution:**
```python
# WRONG
result = agent.run(input_data)

# CORRECT
result = await agent.run(input_data)

# OR wrap in sync context
import asyncio
result = asyncio.run(agent.run(input_data))
```

---

### Issue 5: Configuration Not Loading

**Symptoms:**
```
KeyError: 'database.url'
Config file: greenlang.yaml
```

**Cause:** Using old configuration key names

**Solution:**
```yaml
# Update greenlang.yaml
database:
  connection_string: postgresql://localhost/greenlang  # Changed from 'url'
```

---

### Issue 6: Missing Encryption Key

**Symptoms:**
```
ValueError: GL_ENCRYPTION_KEY environment variable not set
```

**Cause:** New required environment variable not configured

**Solution:**
```bash
# Generate encryption key
export GL_ENCRYPTION_KEY=$(openssl rand -base64 32)

# Add to .env file
echo "GL_ENCRYPTION_KEY=$(openssl rand -base64 32)" >> .env

# Verify
echo $GL_ENCRYPTION_KEY
```

---

### Issue 7: Database Connection Pool Exhausted

**Symptoms:**
```
sqlalchemy.exc.TimeoutError: QueuePool limit of size 5 overflow 10 reached
```

**Cause:** Default pool size too small for async workload

**Solution:**
```yaml
# Update greenlang.yaml
database:
  pool_size: 20  # Increased from default 10
  max_overflow: 30  # Increased from default 20
```

---

### Issue 8: Import Errors After Upgrade

**Symptoms:**
```
ImportError: cannot import name 'ExecutionContext' from 'greenlang.core'
```

**Cause:** Module reorganization in v0.3

**Solution:**
```python
# OLD import
from greenlang.core import ExecutionContext

# NEW import
from greenlang.core.context import ExecutionContext

# See full import mapping
greenlang migrate imports --show-mapping
```

---

## Rollback Procedures

### When to Rollback

Rollback if:
- Migration fails after 3 retry attempts
- Critical functionality broken in production
- Data integrity issues detected
- Performance degradation > 50%
- Cannot resolve errors within maintenance window

### Rollback Steps

**Duration:** 10-15 minutes
**Risk:** MEDIUM

```bash
# 1. Enable maintenance mode
greenlang admin maintenance enable --message "Rolling back to v0.2"

# 2. Stop v0.3 services
systemctl stop greenlang-api greenlang-workers
# OR
docker-compose stop

# 3. Restore database
pg_restore -c -d greenlang backup_v0.2_20251108.sql

# 4. Restore configuration
cp greenlang_v0.2.yaml.backup greenlang.yaml
cp .env.v0.2.backup .env

# 5. Restore agent packs
rm -rf ~/.greenlang/packs/
tar -xzf agent_packs_v0.2_backup.tar.gz -C ~/

# 6. Downgrade Python package
pip install greenlang==0.2.0

# 7. Start v0.2 services
systemctl start greenlang-api greenlang-workers
# OR
docker-compose up -d

# 8. Verify rollback
greenlang --version  # Should show 0.2.0
greenlang admin health-check

# 9. Disable maintenance mode
greenlang admin maintenance disable

# 10. Document rollback reason
echo "Rollback reason: [DESCRIBE ISSUE]" >> rollback_log.txt
```

### Post-Rollback

- Analyze root cause of migration failure
- Document issues encountered
- Create issue tickets for fixes
- Plan retry migration with fixes
- Communicate status to stakeholders

---

## Support Contacts

### Community Support

- **GitHub Issues**: https://github.com/greenlang/greenlang/issues
- **Discord**: https://discord.gg/greenlang
- **Forum**: https://forum.greenlang.io
- **Stack Overflow**: Tag `greenlang`

### Enterprise Support

- **Email**: enterprise@greenlang.io
- **Phone**: +1 (555) 123-4567
- **Support Portal**: https://support.greenlang.io
- **SLA**: 4-hour response for critical issues

### Migration Assistance

- **Migration Consulting**: migration@greenlang.io
- **Professional Services**: https://greenlang.io/services
- **Training**: https://greenlang.io/training

### Emergency Contacts

- **24/7 Hotline**: +1 (555) 999-0000 (Enterprise only)
- **Critical Bug**: critical@greenlang.io
- **Security Issues**: security@greenlang.io

---

## Additional Resources

### Documentation

- [AgentSpec v2 Specification](../specs/agentspec_v2.md)
- [API v2 Reference](../API_REFERENCE.md)
- [Async Agents Guide](../guides/async_agents.md)
- [Security Best Practices](../SECURITY.md)

### Tools

- [Migration CLI Tool](../../greenlang/cli/migrate.py)
- [AgentSpec Converter](../../tools/convert_agentspec.py)
- [Config Validator](../../tools/validate_config.py)

### Videos

- [Migration Walkthrough](https://www.youtube.com/watch?v=...)
- [What's New in v0.3](https://www.youtube.com/watch?v=...)
- [Async Agents Tutorial](https://www.youtube.com/watch?v=...)

---

**Document Revision History:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-11-08 | GreenLang Team | Initial migration guide for v0.2 to v0.3 |

---

**End of Migration Guide**
