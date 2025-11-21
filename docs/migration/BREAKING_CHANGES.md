# GreenLang Breaking Changes Documentation

**Version:** v0.2 to v0.3
**Last Updated:** 2025-11-08
**Document Version:** 1.0.0

---

## Overview

This document lists all breaking changes introduced in GreenLang v0.3, along with migration paths, code examples, and timelines for deprecation and removal.

---

## Summary Table

| ID | Change | Impact | Migration Effort | Deprecation Timeline |
|----|--------|--------|-----------------|---------------------|
| BC-001 | AgentSpec v2 Schema | HIGH | 2-4 hours | Deprecated v0.3 → Removed v0.4 |
| BC-002 | API Endpoints | MEDIUM | 1-2 hours | Deprecated v0.3 → Removed v0.4 |
| BC-003 | Configuration Format | LOW | 30 min | Deprecated v0.3 → Removed v0.4 |
| BC-004 | Database Schema | MEDIUM | 15 min (automated) | v0.3 only |
| BC-005 | Execution Context | MEDIUM | 1-2 hours | v0.3 only |
| BC-006 | Agent Registration | MEDIUM | 1 hour | v0.3 only |
| BC-007 | Environment Variables | LOW | 15 min | Deprecated v0.3 → Removed v0.4 |
| BC-008 | Python Dependencies | LOW | 15 min | v0.3 only |
| BC-009 | Agent Result Format | MEDIUM | 1-2 hours | v0.3 only |
| BC-010 | Logging Format | LOW | 30 min | v0.3 only |
| BC-011 | CLI Commands | LOW | 15 min | v0.3 only |
| BC-012 | Async/Sync Execution | HIGH | 4-8 hours | Sync-only deprecated v0.4 |

---

## BC-001: AgentSpec v2 Schema Changes

### Impact: HIGH
**Affected:** All custom agent packs

### What Changed

Agent pack manifests (`pack.yaml`) now require AgentSpec v2 format with explicit type definitions for inputs and outputs.

### Before (v0.2 - AgentSpec v1)

```yaml
spec_version: "1.0"
name: emissions-calculator
version: "1.0.0"
description: Calculate carbon emissions
author: GreenLang Team
license: MIT

inputs:
  - fuel_type
  - consumption
  - unit

outputs:
  - co2e_emissions_kg
  - fuel_type

capabilities:
  - network_egress

dependencies:
  - numpy
  - pandas
```

### After (v0.3 - AgentSpec v2)

```yaml
spec_version: "2.0"

metadata:
  name: emissions-calculator
  version: "1.0.0"
  description: Calculate carbon emissions
  author: GreenLang Team
  license: MIT
  capabilities:
    - network_egress
  tags:
    - emissions
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

dependencies:
  - name: numpy
    version: ">=1.24.0,<2.0.0"
  - name: pandas
    version: ">=2.0.0,<3.0.0"
```

### Migration Steps

1. **Automated Conversion:**
   ```bash
   greenlang migrate execute
   # Automatically converts all agent packs
   ```

2. **Manual Conversion:**
   ```bash
   # For each pack.yaml file:
   # 1. Change spec_version to "2.0"
   # 2. Wrap metadata fields in "metadata" section
   # 3. Convert inputs to typed objects
   # 4. Convert outputs to typed objects
   # 5. Move capabilities to metadata
   # 6. Add version constraints to dependencies
   ```

3. **Validate Conversion:**
   ```bash
   greenlang agents validate pack.yaml
   ```

### Deprecation Timeline

- **v0.3:** AgentSpec v1 supported with deprecation warnings
- **v0.4 (Q2 2026):** AgentSpec v1 removed entirely

### Testing

```python
# Test that v2 pack loads correctly
from greenlang.packs import PackManifest

manifest = PackManifest.from_file("pack.yaml")
assert manifest.spec_version == "2.0"
assert "fuel_type" in manifest.inputs
assert manifest.inputs["fuel_type"]["type"] == "string"
```

---

## BC-002: API Endpoint Changes

### Impact: MEDIUM
**Affected:** API consumers using v1 endpoints

### What Changed

API endpoints moved from `/api/v1/` to `/api/v2/` with improved request/response format and JWT authentication.

### Before (v0.2)

```python
import requests

# Using API key authentication
response = requests.post(
    "http://localhost:8000/api/v1/agents/register",
    json={
        "name": "my-agent",
        "version": "1.0.0"
    },
    headers={"X-API-Key": "your-api-key-here"}
)

data = response.json()  # Direct response
```

### After (v0.3)

```python
import requests

# 1. Get JWT token
auth_response = requests.post(
    "http://localhost:8000/api/v2/auth/token",
    json={
        "username": "user",
        "password": "password"
    }
)
jwt_token = auth_response.json()["data"]["access_token"]

# 2. Use JWT authentication
response = requests.post(
    "http://localhost:8000/api/v2/agents",
    json={
        "data": {
            "name": "my-agent",
            "version": "1.0.0"
        },
        "metadata": {
            "client_version": "0.3.0",
            "correlation_id": "unique-request-id"
        }
    },
    headers={"Authorization": f"Bearer {jwt_token}"}
)

# Response is wrapped in envelope
data = response.json()["data"]
metadata = response.json()["metadata"]
```

### Endpoint Mapping

| v0.2 (Deprecated) | v0.3 (Current) |
|------------------|----------------|
| `POST /api/v1/agents/register` | `POST /api/v2/agents` |
| `GET /api/v1/agents/list` | `GET /api/v2/agents` |
| `POST /api/v1/workflows/execute` | `POST /api/v2/workflows/execute` |
| `GET /api/v1/workflows/list` | `GET /api/v2/workflows` |
| `GET /api/v1/health` | `GET /api/v2/health` |

### Migration Steps

1. **Update Base URL:** Change `/api/v1/` to `/api/v2/`
2. **Implement JWT Auth:** Replace API key with JWT token
3. **Wrap Requests:** Wrap request data in `{"data": {...}, "metadata": {...}}`
4. **Unwrap Responses:** Extract data from `response.json()["data"]`

### Deprecation Timeline

- **v0.3:** `/api/v1/` endpoints available with `Deprecated: true` header
- **v0.4 (Q2 2026):** `/api/v1/` endpoints removed

### Testing

```python
def test_api_v2_authentication():
    # Test JWT token generation
    response = requests.post(
        "http://localhost:8000/api/v2/auth/token",
        json={"username": "test", "password": "test"}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()["data"]

def test_api_v2_endpoint():
    token = get_jwt_token()
    response = requests.get(
        "http://localhost:8000/api/v2/agents",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert "data" in response.json()
    assert "metadata" in response.json()
```

---

## BC-003: Configuration Format Changes

### Impact: LOW
**Affected:** `greenlang.yaml` and `.env` files

### What Changed

Configuration keys renamed for consistency, new required fields added.

### Before (v0.2)

```yaml
# greenlang.yaml
version: "0.2.0"

database:
  url: postgresql://user:pass@localhost/greenlang
  echo: false

security:
  enabled: true
  api_keys_enabled: true

logging:
  level: info  # lowercase
  file: greenlang.log
```

```bash
# .env
GREENLANG_ENV=production
GREENLANG_DB_URL=postgresql://localhost/greenlang
API_KEY=your-api-key
```

### After (v0.3)

```yaml
# greenlang.yaml
version: "0.3.0"

database:
  connection_string: postgresql://user:pass@localhost/greenlang  # renamed
  pool_size: 10
  max_overflow: 20

security:
  enabled: true
  mfa_required: false
  jwt_secret: ${GL_JWT_SECRET}
  encryption:  # NEW REQUIRED
    algorithm: AES-256-GCM
    key: ${GL_ENCRYPTION_KEY}
    key_rotation_days: 90

logging:
  level: INFO  # uppercase
  format: json  # NEW
  file: logs/greenlang.log
  structured: true  # NEW
```

```bash
# .env
GL_ENV=production  # renamed
GL_DATABASE_URL=postgresql://localhost/greenlang  # renamed
GL_SECRET_KEY=<generate-with-openssl>  # NEW REQUIRED
GL_ENCRYPTION_KEY=<generate-with-openssl>  # NEW REQUIRED
GL_JWT_SECRET=<generate-with-openssl>  # NEW REQUIRED
```

### Migration Steps

1. **Update greenlang.yaml:**
   ```bash
   # Automated
   greenlang migrate execute

   # Manual
   # 1. Change 'database.url' to 'database.connection_string'
   # 2. Add 'security.encryption' section
   # 3. Change logging.level to uppercase
   # 4. Add logging.format: json
   ```

2. **Update .env:**
   ```bash
   # Generate new keys
   export GL_SECRET_KEY=$(openssl rand -hex 32)
   export GL_ENCRYPTION_KEY=$(openssl rand -base64 32)
   export GL_JWT_SECRET=$(openssl rand -hex 32)

   # Rename variables
   GREENLANG_ENV → GL_ENV
   GREENLANG_DB_URL → GL_DATABASE_URL
   ```

### Deprecation Timeline

- **v0.3:** Old keys supported with deprecation warnings
- **v0.4 (Q2 2026):** Old keys removed

### Testing

```python
def test_configuration_loading():
    from greenlang.config import load_config

    config = load_config("greenlang.yaml")
    assert config["version"] == "0.3.0"
    assert "connection_string" in config["database"]
    assert "encryption" in config["security"]
    assert config["logging"]["level"] == "INFO"
```

---

## BC-004: Database Schema Changes

### Impact: MEDIUM
**Affected:** Database requires Alembic migration

### What Changed

New tables added, existing tables modified with additional columns.

### Schema Changes

```sql
-- New tables
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    action VARCHAR(255) NOT NULL,
    resource_type VARCHAR(100),
    resource_id UUID,
    timestamp TIMESTAMP DEFAULT NOW(),
    ip_address VARCHAR(45),
    metadata JSONB,
    INDEX idx_audit_logs_user_id (user_id),
    INDEX idx_audit_logs_timestamp (timestamp)
);

CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id),
    token_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP NOT NULL,
    ip_address VARCHAR(45),
    user_agent TEXT,
    INDEX idx_user_sessions_token (token_hash),
    INDEX idx_user_sessions_user_id (user_id)
);

CREATE TABLE agent_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES agents(id),
    version VARCHAR(50) NOT NULL,
    spec_version VARCHAR(10) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    manifest JSONB NOT NULL,
    INDEX idx_agent_versions_agent_id (agent_id)
);

-- Modified tables
ALTER TABLE agents
    ADD COLUMN spec_version VARCHAR(10) DEFAULT '1.0',
    ADD COLUMN created_at TIMESTAMP DEFAULT NOW(),
    ADD COLUMN updated_at TIMESTAMP DEFAULT NOW();

ALTER TABLE workflows
    ADD COLUMN async_enabled BOOLEAN DEFAULT FALSE,
    ADD COLUMN created_at TIMESTAMP DEFAULT NOW(),
    ADD COLUMN updated_at TIMESTAMP DEFAULT NOW();

-- Dropped columns (deprecated)
ALTER TABLE agents DROP COLUMN IF EXISTS legacy_format;
ALTER TABLE workflows DROP COLUMN IF EXISTS deprecated_config;
```

### Migration Steps

1. **Automated (Recommended):**
   ```bash
   greenlang migrate execute
   # Automatically runs Alembic migrations
   ```

2. **Manual:**
   ```bash
   # Backup database first
   pg_dump greenlang > backup_before_migration.sql

   # Run Alembic migrations
   alembic upgrade head

   # Verify migration
   greenlang migrate verify
   ```

### Rollback

```bash
# If migration fails
alembic downgrade -1  # Rollback one migration

# Or restore from backup
psql greenlang < backup_before_migration.sql
```

### Testing

```python
def test_database_schema():
    from sqlalchemy import inspect
    from greenlang.db import get_engine

    engine = get_engine()
    inspector = inspect(engine)

    # Verify new tables exist
    tables = inspector.get_table_names()
    assert "audit_logs" in tables
    assert "user_sessions" in tables
    assert "agent_versions" in tables

    # Verify new columns
    agents_columns = [col["name"] for col in inspector.get_columns("agents")]
    assert "spec_version" in agents_columns
```

---

## BC-005: Workflow Execution Context

### Impact: MEDIUM
**Affected:** Custom workflow executors

### What Changed

`ExecutionContext` now requires `async_mode` parameter and data is immutable.

### Before (v0.2)

```python
from greenlang.core import ExecutionContext

context = ExecutionContext(data=input_data)
context.data["new_field"] = value  # Direct mutation
results = context.agent_results  # Dict
```

### After (v0.3)

```python
from greenlang.core.context import ExecutionContext

context = ExecutionContext(
    data=input_data,
    async_mode=True,  # NEW REQUIRED
    timeout_seconds=300
)
context.set_data("new_field", value)  # Use setter
results = context.get_agent_results()  # Ordered list
```

### Migration Steps

```python
# Update all ExecutionContext instantiations
# OLD
context = ExecutionContext(data=input_data)

# NEW
context = ExecutionContext(
    data=input_data,
    async_mode=True  # or False for sync
)

# Update data mutations
# OLD
context.data["key"] = value

# NEW
context.set_data("key", value)

# Update result access
# OLD
result = context.agent_results["agent_id"]

# NEW
results = context.get_agent_results()
result = next(r for r in results if r.agent_id == "agent_id")
```

### Testing

```python
def test_execution_context():
    context = ExecutionContext(
        data={"test": "data"},
        async_mode=True
    )

    # Test immutability
    with pytest.raises(AttributeError):
        context.data["new"] = "value"

    # Test setter
    context.set_data("new", "value")
    assert context.get_data("new") == "value"
```

---

## BC-006: Agent Registration

### Impact: MEDIUM
**Affected:** Programmatic agent registration

### What Changed

`Orchestrator.register_agent()` now requires metadata and validates AgentSpec v2.

### Before (v0.2)

```python
from greenlang.core.orchestrator import Orchestrator
from greenlang.agents import FuelAgent

orchestrator = Orchestrator()
orchestrator.register_agent("fuel", FuelAgent())
```

### After (v0.3)

```python
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
        "description": "Calculates emissions from fuel consumption",
        "author": "GreenLang Team"
    }
)
```

### Migration Steps

Add metadata to all `register_agent()` calls:

```python
# Template
orchestrator.register_agent(
    agent_id="your_agent_id",
    agent_instance=YourAgent(),
    metadata={
        "name": "Human-readable name",
        "version": "1.0.0",
        "spec_version": "2.0",
        "description": "Agent description",
        "author": "Your name/team"
    }
)
```

### Testing

```python
def test_agent_registration():
    orchestrator = Orchestrator()

    # Should require metadata
    with pytest.raises(ValueError):
        orchestrator.register_agent("test", TestAgent())

    # Should work with metadata
    orchestrator.register_agent(
        "test",
        TestAgent(),
        metadata={
            "name": "Test",
            "version": "1.0.0",
            "spec_version": "2.0",
            "description": "Test agent"
        }
    )
```

---

## BC-007: Environment Variables

### Impact: LOW
**Affected:** Deployment configurations

### What Changed

Environment variables renamed with `GL_` prefix, new required variables added.

### Migration Table

| Old (v0.2) | New (v0.3) | Required |
|------------|------------|----------|
| `GREENLANG_ENV` | `GL_ENV` | Yes |
| `GREENLANG_DB_URL` | `GL_DATABASE_URL` | Yes |
| `API_KEY` | N/A (deprecated) | No |
| - | `GL_SECRET_KEY` | Yes |
| - | `GL_ENCRYPTION_KEY` | Yes |
| - | `GL_JWT_SECRET` | Yes |

### Migration Steps

```bash
# Generate new secret keys
export GL_SECRET_KEY=$(openssl rand -hex 32)
export GL_ENCRYPTION_KEY=$(openssl rand -base64 32)
export GL_JWT_SECRET=$(openssl rand -hex 32)

# Rename existing variables
export GL_ENV=$GREENLANG_ENV
export GL_DATABASE_URL=$GREENLANG_DB_URL

# Update .env file
sed -i 's/GREENLANG_ENV/GL_ENV/g' .env
sed -i 's/GREENLANG_DB_URL/GL_DATABASE_URL/g' .env
echo "GL_SECRET_KEY=$(openssl rand -hex 32)" >> .env
echo "GL_ENCRYPTION_KEY=$(openssl rand -base64 32)" >> .env
echo "GL_JWT_SECRET=$(openssl rand -hex 32)" >> .env
```

### Testing

```python
def test_environment_variables():
    import os

    # Required variables must be set
    assert os.environ.get("GL_ENV") is not None
    assert os.environ.get("GL_DATABASE_URL") is not None
    assert os.environ.get("GL_SECRET_KEY") is not None
    assert os.environ.get("GL_ENCRYPTION_KEY") is not None
```

---

## BC-008: Python Package Dependencies

### Impact: LOW
**Affected:** Python version and package requirements

### What Changed

- Minimum Python version: 3.9 (was 3.8)
- Pydantic upgraded to v2 (was v1)
- New dependencies: `aiohttp`, `cryptography`

### Before (v0.2)

```txt
# requirements.txt
greenlang==0.2.0
pydantic==1.10.12
python>=3.8
```

### After (v0.3)

```txt
# requirements.txt
greenlang==0.3.0
pydantic>=2.0.0,<3.0.0
aiohttp>=3.9.0
cryptography>=41.0.0
alembic>=1.12.0
python>=3.9
```

### Migration Steps

```bash
# 1. Upgrade Python if needed
python --version  # Must be 3.9+

# 2. Create new virtual environment
python3.9 -m venv venv
source venv/bin/activate

# 3. Upgrade packages
pip install --upgrade pip
pip install greenlang==0.3.0

# 4. Update requirements.txt
pip freeze > requirements.txt
```

### Pydantic v2 Migration

```python
# OLD (Pydantic v1)
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

    class Config:
        orm_mode = True

# NEW (Pydantic v2)
from pydantic import BaseModel, ConfigDict

class User(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    age: int
```

### Testing

```python
def test_python_version():
    import sys
    assert sys.version_info >= (3, 9)

def test_dependencies():
    import pydantic
    import aiohttp
    import cryptography

    # Pydantic v2
    assert pydantic.__version__.startswith("2.")
```

---

## BC-009: Agent Result Format

### Impact: MEDIUM
**Affected:** Agent output parsing

### What Changed

Agent results now wrapped in `AgentResult` dataclass with status enum.

### Before (v0.2)

```python
result = agent.run(input_data)

if result.success:  # Boolean
    data = result.data
else:
    errors = result.errors
```

### After (v0.3)

```python
from greenlang.core.types import AgentStatus

result = agent.run(input_data)

if result.status == AgentStatus.SUCCESS:  # Enum
    data = result.output  # renamed from 'data'
elif result.status == AgentStatus.FAILURE:
    errors = result.error_details  # renamed from 'errors'
elif result.status == AgentStatus.PARTIAL:
    data = result.output
    warnings = result.warnings
```

### Migration Steps

```python
# Update all agent result handling
# OLD
if result.success:
    process(result.data)
else:
    log_errors(result.errors)

# NEW
from greenlang.core.types import AgentStatus

if result.status == AgentStatus.SUCCESS:
    process(result.output)
elif result.status == AgentStatus.FAILURE:
    log_errors(result.error_details)
```

### Testing

```python
def test_agent_result():
    from greenlang.core.types import AgentResult, AgentStatus

    result = AgentResult(
        status=AgentStatus.SUCCESS,
        output={"result": "data"},
        metadata={"duration_ms": 100}
    )

    assert result.status == AgentStatus.SUCCESS
    assert "result" in result.output
```

---

## BC-010: Logging Format

### Impact: LOW
**Affected:** Log parsing and monitoring

### What Changed

Default log format changed from plain text to structured JSON.

### Before (v0.2)

```
2025-11-08 10:30:45 INFO Agent executed successfully
2025-11-08 10:30:46 ERROR Database connection failed
```

### After (v0.3)

```json
{"timestamp": "2025-11-08T10:30:45.123Z", "level": "INFO", "message": "Agent executed successfully", "correlation_id": "abc-123", "agent_id": "fuel_agent", "duration_ms": 45}
{"timestamp": "2025-11-08T10:30:46.456Z", "level": "ERROR", "message": "Database connection failed", "correlation_id": "def-456", "error": "Connection timeout"}
```

### Migration Steps

1. **Update Log Parsers:**
   ```python
   # OLD
   pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (\w+) (.+)"

   # NEW
   import json
   log_entry = json.loads(log_line)
   timestamp = log_entry["timestamp"]
   level = log_entry["level"]
   message = log_entry["message"]
   ```

2. **Update Monitoring:**
   ```bash
   # OLD (grep)
   grep "ERROR" greenlang.log

   # NEW (jq)
   cat greenlang.log | jq 'select(.level == "ERROR")'
   ```

3. **Configure JSON Logging:**
   ```yaml
   # greenlang.yaml
   logging:
     format: json
     structured: true
   ```

### Testing

```python
def test_json_logging():
    import json
    import logging

    # Configure JSON logging
    from greenlang.logging import configure_logging
    configure_logging(format="json")

    # Read log output
    with open("test.log") as f:
        log_entry = json.loads(f.readline())

    assert "timestamp" in log_entry
    assert "level" in log_entry
    assert "message" in log_entry
```

---

## BC-011: CLI Command Changes

### Impact: LOW
**Affected:** CLI scripts and automation

### What Changed

CLI commands reorganized and some renamed for consistency.

### Command Changes

| Old (v0.2) | New (v0.3) |
|------------|------------|
| `gl run workflow.yaml` | `gl run workflow.yaml --backend local` |
| `gl agent register` | `gl agents register` |
| `gl workflow execute` | `gl workflows execute` |
| `gl pack validate` | `gl packs validate` |

### Migration Steps

```bash
# Update all scripts
# OLD
gl run workflow.yaml --input data.json
gl agent register pack.yaml

# NEW
gl run workflow.yaml --input data.json --backend local
gl agents register pack.yaml

# Find and replace in scripts
sed -i 's/gl agent /gl agents /g' *.sh
sed -i 's/gl workflow /gl workflows /g' *.sh
sed -i 's/gl pack /gl packs /g' *.sh
```

### Testing

```bash
# Test new commands
gl run --help
gl agents --help
gl workflows --help
gl packs --help
```

---

## BC-012: Async/Sync Agent Execution

### Impact: HIGH
**Affected:** Custom agent development

### What Changed

Default agent execution changed from sync-only to async-first. Agents must explicitly opt-out of async.

### Before (v0.2)

```python
from greenlang.agents.base import BaseAgent

class MyAgent(BaseAgent):
    def run(self, input_data):  # Always sync
        result = self.process(input_data)
        return {"success": True, "data": result}
```

### After (v0.3)

```python
from greenlang.agents.base import BaseAgent
from greenlang.core.types import AgentResult, AgentStatus

# Option 1: Async agent (recommended)
class MyAsyncAgent(BaseAgent):
    async def run(self, input_data):
        result = await self.process_async(input_data)
        return AgentResult(
            status=AgentStatus.SUCCESS,
            output=result
        )

    async def process_async(self, data):
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.example.com") as resp:
                return await resp.json()

# Option 2: Sync agent (opt-out)
from greenlang.agents import sync_only

@sync_only
class MySyncAgent(BaseAgent):
    def run(self, input_data):
        result = self.process(input_data)
        return AgentResult(
            status=AgentStatus.SUCCESS,
            output=result
        )
```

### Migration Steps

1. **For async agents:**
   ```python
   # Add 'async' keyword to run method
   # Use 'await' for async operations
   # Update imports (aiohttp, asyncio)
   ```

2. **For sync agents:**
   ```python
   # Add @sync_only decorator
   from greenlang.agents import sync_only

   @sync_only
   class MyAgent(BaseAgent):
       def run(self, input_data):
           # Sync code remains unchanged
   ```

### Testing

```python
import pytest

@pytest.mark.asyncio
async def test_async_agent():
    agent = MyAsyncAgent()
    result = await agent.run({"test": "data"})
    assert result.status == AgentStatus.SUCCESS

def test_sync_agent():
    agent = MySyncAgent()
    result = agent.run({"test": "data"})
    assert result.status == AgentStatus.SUCCESS
```

---

## Quick Reference

### All Breaking Changes Summary

```
✅ BC-001: AgentSpec v2 (HIGH) - Convert pack.yaml files
✅ BC-002: API Endpoints (MEDIUM) - Update to /api/v2/
✅ BC-003: Config Format (LOW) - Rename database.url
✅ BC-004: Database Schema (MEDIUM) - Run migrations
✅ BC-005: Execution Context (MEDIUM) - Add async_mode
✅ BC-006: Agent Registration (MEDIUM) - Add metadata
✅ BC-007: Environment Vars (LOW) - Rename to GL_*
✅ BC-008: Dependencies (LOW) - Upgrade Python 3.10+
✅ BC-009: Result Format (MEDIUM) - Use AgentStatus
✅ BC-010: Logging (LOW) - Switch to JSON
✅ BC-011: CLI Commands (LOW) - Update command names
✅ BC-012: Async/Sync (HIGH) - Add async support
```

### Migration Priority

1. **Critical (Do First):**
   - BC-004: Database Schema
   - BC-007: Environment Variables
   - BC-008: Python Dependencies

2. **High Priority:**
   - BC-001: AgentSpec v2
   - BC-012: Async/Sync

3. **Medium Priority:**
   - BC-002: API Endpoints
   - BC-005: Execution Context
   - BC-006: Agent Registration
   - BC-009: Result Format

4. **Low Priority:**
   - BC-003: Config Format
   - BC-010: Logging
   - BC-011: CLI Commands

---

## Support

For migration assistance:
- Email: migration@greenlang.io
- Documentation: https://docs.greenlang.io/migration
- Migration Tool: `greenlang migrate --help`

---

**Last Updated:** 2025-11-08
**Document Version:** 1.0.0
**Maintained By:** GreenLang Team
