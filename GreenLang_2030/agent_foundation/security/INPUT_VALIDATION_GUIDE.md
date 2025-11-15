# Input Validation Framework - Developer Guide

## Overview

This guide provides comprehensive instructions for using the GreenLang Input Validation Framework to prevent injection attacks and ensure data integrity throughout the application.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Validation Types](#validation-types)
3. [Common Patterns](#common-patterns)
4. [Security Best Practices](#security-best-practices)
5. [Examples](#examples)
6. [Testing](#testing)

---

## Quick Start

### Installation

The input validation framework is part of the `security` module:

```python
from security.input_validation import (
    InputValidator,
    TenantIdModel,
    UserIdModel,
    SafeQueryInput,
)
```

### Basic Usage

```python
# Validate alphanumeric input
tenant_id = InputValidator.validate_alphanumeric(
    user_input,
    "tenant_id",
    min_length=3,
    max_length=255
)

# Validate UUID
user_id = InputValidator.validate_uuid(user_input, "user_id")

# Validate email
email = InputValidator.validate_email(user_input)
```

---

## Validation Types

### 1. Alphanumeric Validation

**Use for:** IDs, names, resource identifiers

```python
# Basic validation
tenant_id = InputValidator.validate_alphanumeric(
    value="tenant-123",
    field_name="tenant_id"
)

# With length constraints
agent_name = InputValidator.validate_alphanumeric(
    value=user_input,
    field_name="agent_name",
    min_length=3,
    max_length=100
)
```

**Allows:** `a-z`, `A-Z`, `0-9`, `_`, `-`
**Blocks:** Special characters, spaces, SQL/command injection

---

### 2. UUID Validation

**Use for:** User IDs, execution IDs, unique identifiers

```python
user_id = InputValidator.validate_uuid(
    value="123e4567-e89b-12d3-a456-426614174000",
    field_name="user_id"
)
```

**Format:** RFC 4122 UUID
**Returns:** Lowercase UUID string

---

### 3. Email Validation

**Use for:** Email addresses

```python
email = InputValidator.validate_email("user@example.com")
```

**Returns:** Lowercase email string
**Max Length:** 255 characters

---

### 4. SQL Injection Prevention

**Use for:** ANY string that will be used in SQL queries

```python
# Validate individual value
description = InputValidator.validate_no_sql_injection(
    value=user_input,
    field_name="description"
)

# Use SafeQueryInput for complete query safety
query_filter = SafeQueryInput(
    field="tenant_id",  # Validated against whitelist
    value="tenant-123",  # Validated for SQL injection
    operator="="  # Validated against operator whitelist
)
```

**Detects:**
- SQL keywords (SELECT, INSERT, UPDATE, DELETE, DROP, etc.)
- Comment indicators (`--`, `/*`, `#`)
- Quote characters (`'`, `"`)
- Semicolons (`;`)

---

### 5. Command Injection Prevention

**Use for:** System commands, subprocess arguments

```python
# Validate command argument
arg = InputValidator.validate_no_command_injection(
    value=user_input,
    field_name="command_arg"
)

# Use SafeCommandInput for complete command safety
cmd = SafeCommandInput(
    command="kubectl",  # Validated against whitelist
    args=["get", "pods"],  # Each validated for injection
    allowed_commands=["kubectl", "docker", "helm"]
)
```

**Detects:**
- Shell metacharacters (`;`, `|`, `&`, `` ` ``, `$`, `(`, `)`, etc.)
- Command substitution
- Pipe operators
- Redirection operators

---

### 6. Path Traversal Prevention

**Use for:** File paths, directory paths

```python
# Basic path validation
file_path = InputValidator.validate_path(
    value="C:/data/config.json",
    must_exist=False,
    allow_relative=False
)

# With extension validation
yaml_path = InputValidator.validate_path(
    value=user_input,
    must_exist=True,
    allowed_extensions=['.yaml', '.yml']
)

# Use SafePathInput model
path_input = SafePathInput(
    path="C:/data/config.json",
    must_exist=True,
    allowed_extensions=['.json']
)
```

**Prevents:**
- `../` traversal
- Absolute path attacks (`/etc/passwd`, `C:\Windows\System32`)
- Null byte injection (`%00`)

---

### 7. SSRF Prevention

**Use for:** URLs, IP addresses

```python
# Validate URL
api_url = InputValidator.validate_url(
    value="https://api.example.com/data",
    allowed_schemes=['https'],
    allow_private_ips=False
)

# Validate IP address
ip_address = InputValidator.validate_ip_address(
    value="8.8.8.8",
    allow_private=False,
    allow_loopback=False
)
```

**Prevents:**
- Private IP ranges (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)
- Loopback addresses (127.0.0.0/8)
- Link-local addresses
- Dangerous URL schemes (file://, gopher://, etc.)

---

### 8. Field Name Whitelisting

**Use for:** Dynamic SQL query building

```python
# Validate field name against whitelist
field_name = InputValidator.validate_field_name("tenant_id")

# Use in SafeQueryInput
query = SafeQueryInput(
    field="tenant_id",  # Must be in ALLOWED_FIELDS whitelist
    value="tenant-123",
    operator="="
)
```

**Whitelist:**
```python
ALLOWED_FIELDS = {
    'tenant_id', 'user_id', 'agent_id', 'execution_id',
    'name', 'email', 'status', 'type', 'tier',
    'created_at', 'updated_at', 'created_by', 'updated_by',
    # ... (see input_validation.py for complete list)
}
```

**To add a field:** Update `ALLOWED_FIELDS` in `input_validation.py`

---

### 9. XSS Prevention

**Use for:** User-generated content displayed in web pages

```python
# Detect XSS patterns
safe_text = InputValidator.validate_no_xss(
    value=user_input,
    field_name="description"
)

# Sanitize HTML
safe_html = InputValidator.sanitize_html(
    "<script>alert('xss')</script>"
)
# Returns: "&lt;script&gt;alert('xss')&lt;/script&gt;"
```

---

## Common Patterns

### Pattern 1: Database Query with User Input

```python
from security.input_validation import SafeQueryInput
from database.postgres_manager_secure import SecureQueryBuilder

# Build safe query
builder = SecureQueryBuilder("agents")

filters = [
    SafeQueryInput(
        field="tenant_id",
        value=user_provided_tenant_id,
        operator="="
    ),
    SafeQueryInput(
        field="status",
        value=user_provided_status,
        operator="="
    )
]

# Generate parameterized query (SQL injection safe)
query, params = builder.build_select(
    filters=filters,
    limit=100,
    offset=0,
    sort_by="created_at",
    sort_direction="DESC"
)

# Execute with asyncpg (parameters bound safely)
results = await conn.fetch(query, *params)
```

**Why this is safe:**
- Field names validated against whitelist
- Operators validated against whitelist
- Values checked for SQL injection patterns
- Query uses parameterized placeholders ($1, $2, etc.)
- No string interpolation

---

### Pattern 2: Kubernetes Deployment

```python
from factory.deployment_secure import SecureDeploymentManager

manager = SecureDeploymentManager()

# Deploy agent (all inputs validated)
result = manager.deploy_agent(
    agent_name=user_provided_name,  # Validated: alphanumeric only
    image_tag=user_provided_tag,     # Validated: alphanumeric only
    namespace=user_provided_ns,      # Validated: alphanumeric only
    replicas=user_provided_replicas  # Validated: integer, 1-100
)
```

**Why this is safe:**
- All inputs validated before building command
- Command executed with `shell=False`
- Arguments passed as list (no shell parsing)
- Command whitelist enforced

---

### Pattern 3: API Request Validation

```python
from fastapi import FastAPI, Request
from api.middleware.validation import RequestValidationMiddleware

app = FastAPI()

# Add validation middleware
app.add_middleware(RequestValidationMiddleware)

@app.post("/api/agents")
async def create_agent(request: Request, agent_data: dict):
    # Request already validated by middleware:
    # - Content-Type checked
    # - Content-Length checked
    # - Rate limited
    # - Headers validated

    # Validate request body
    tenant_id = InputValidator.validate_alphanumeric(
        agent_data.get("tenant_id"),
        "tenant_id"
    )

    # Safe to process
    ...
```

---

### Pattern 4: Pydantic Models for Input Validation

```python
from security.input_validation import TenantIdModel, EmailModel

# Use Pydantic models for automatic validation
class CreateUserRequest(BaseModel):
    tenant_id: str
    email: str

    @validator('tenant_id')
    def validate_tenant(cls, v):
        return InputValidator.validate_alphanumeric(v, 'tenant_id')

    @validator('email')
    def validate_email(cls, v):
        return InputValidator.validate_email(v)

# FastAPI automatically validates
@app.post("/users")
async def create_user(request: CreateUserRequest):
    # request.tenant_id and request.email are validated
    ...
```

---

## Security Best Practices

### 1. Defense in Depth

**Always use multiple layers of validation:**

```python
# Layer 1: Pydantic model validation
class AgentQuery(BaseModel):
    tenant_id: str

    @validator('tenant_id')
    def validate_tenant(cls, v):
        return InputValidator.validate_alphanumeric(v, 'tenant_id')

# Layer 2: SQL injection check
InputValidator.validate_no_sql_injection(query.tenant_id, 'tenant_id')

# Layer 3: Parameterized query
query, params = builder.build_select(...)  # Uses $1, $2 placeholders

# Layer 4: Database-level permissions
# Ensure database user has minimal required permissions
```

---

### 2. Whitelist Over Blacklist

**Always prefer whitelisting:**

```python
# GOOD: Whitelist approach
ALLOWED_FIELDS = {'tenant_id', 'user_id', 'status'}

if field not in ALLOWED_FIELDS:
    raise ValueError(f"Field not allowed: {field}")

# BAD: Blacklist approach (can be bypassed)
BLOCKED_FIELDS = {'password', 'secret'}

if field not in BLOCKED_FIELDS:
    # Allow
    pass
```

---

### 3. Fail Securely

**When validation fails, deny access:**

```python
# GOOD: Explicit validation with clear error
try:
    tenant_id = InputValidator.validate_alphanumeric(
        user_input,
        "tenant_id"
    )
except ValueError as e:
    logger.warning(f"Validation failed: {e}")
    raise HTTPException(400, "Invalid tenant ID")

# BAD: Silently accepting invalid input
tenant_id = user_input or "default"  # NEVER DO THIS
```

---

### 4. Log Suspicious Activity

```python
# Log validation failures for security monitoring
try:
    InputValidator.validate_no_sql_injection(value, "field")
except ValueError as e:
    logger.warning(
        "Potential SQL injection detected",
        extra={
            "field": "field",
            "value_preview": value[:100],
            "client_ip": request.client.host
        }
    )
    raise
```

---

### 5. Rate Limiting

```python
from api.middleware.validation import RateLimiter

# Protect against brute force attacks
rate_limiter = RateLimiter(requests=100, window=60)

if not rate_limiter.allow_request(client_id):
    raise HTTPException(429, "Rate limit exceeded")
```

---

## Examples

### Example 1: Safe User Registration

```python
from security.input_validation import InputValidator, EmailModel
from database.postgres_manager_secure import SecurePostgresOperations

async def register_user(
    tenant_id: str,
    email: str,
    name: str
) -> dict:
    """Register new user with validation."""

    # Validate inputs
    validated_tenant = InputValidator.validate_alphanumeric(
        tenant_id, "tenant_id", min_length=3, max_length=255
    )

    validated_email = InputValidator.validate_email(email)

    validated_name = InputValidator.validate_alphanumeric(
        name, "name", min_length=1, max_length=255
    )

    # Create user safely
    ops = SecurePostgresOperations("users")

    user = await ops.create(
        conn=db_connection,
        data={
            "tenant_id": validated_tenant,
            "email": validated_email,
            "name": validated_name,
            "created_at": datetime.utcnow(),
        }
    )

    return user
```

---

### Example 2: Safe File Upload

```python
from security.input_validation import InputValidator

async def upload_config_file(
    tenant_id: str,
    file_path: str,
    file_content: bytes
) -> dict:
    """Upload configuration file with validation."""

    # Validate tenant ID
    validated_tenant = InputValidator.validate_alphanumeric(
        tenant_id, "tenant_id"
    )

    # Validate file path
    validated_path = InputValidator.validate_path(
        file_path,
        must_exist=False,
        allow_relative=False,
        allowed_extensions=['.yaml', '.yml', '.json']
    )

    # Validate file size
    if len(file_content) > 10_000_000:  # 10MB
        raise ValueError("File too large")

    # Validate file content is valid JSON/YAML
    if validated_path.suffix == '.json':
        config = InputValidator.validate_json(file_content.decode())
    elif validated_path.suffix in ['.yaml', '.yml']:
        import yaml
        config = yaml.safe_load(file_content)

    # Safe to save file
    with open(validated_path, 'wb') as f:
        f.write(file_content)

    return {"path": str(validated_path), "size": len(file_content)}
```

---

### Example 3: Safe Kubernetes Deployment

```python
from factory.deployment_secure import SecureDeploymentManager

async def deploy_production_agent(
    agent_name: str,
    version: str,
    replicas: int
) -> dict:
    """Deploy agent to production with validation."""

    # Validate inputs
    validated_name = InputValidator.validate_alphanumeric(
        agent_name, "agent_name", max_length=253
    )

    validated_version = InputValidator.validate_alphanumeric(
        version, "version", max_length=128
    )

    validated_replicas = InputValidator.validate_integer(
        replicas, "replicas", min_value=1, max_value=10
    )

    # Deploy safely
    manager = SecureDeploymentManager()

    result = manager.deploy_agent(
        agent_name=validated_name,
        image_tag=validated_version,
        namespace="production",
        replicas=validated_replicas
    )

    return result
```

---

## Testing

### Running Security Tests

```bash
# Run all security tests
pytest testing/security_tests/ -v

# Run specific test file
pytest testing/security_tests/test_input_validation.py -v

# Run with coverage
pytest testing/security_tests/ --cov=security --cov-report=html
```

### Writing New Tests

```python
import pytest
from security.input_validation import InputValidator

def test_my_validation():
    """Test custom validation logic."""

    # Test valid input
    result = InputValidator.validate_alphanumeric("valid-123", "field")
    assert result == "valid-123"

    # Test invalid input
    with pytest.raises(ValueError, match="alphanumeric"):
        InputValidator.validate_alphanumeric("invalid@123", "field")
```

---

## Troubleshooting

### Common Issues

**Issue:** `ValueError: Field 'my_field' not in whitelist`

**Solution:** Add field to `ALLOWED_FIELDS` in `input_validation.py`:

```python
ALLOWED_FIELDS = {
    'tenant_id', 'user_id', 'agent_id',
    'my_field',  # Add your field here
    # ...
}
```

---

**Issue:** `ValueError: contains potential SQL injection pattern`

**Solution:**
- Ensure you're not including SQL keywords in user input
- Use parameterized queries instead of string concatenation
- If the input is legitimate (e.g., "SELECT as a word in description"), consider sanitizing

```python
# Instead of rejecting, sanitize
safe_value = InputValidator.sanitize_html(user_input)
```

---

**Issue:** `ValueError: Path contains dangerous traversal patterns`

**Solution:**
- Ensure paths are absolute
- Use `Path.resolve()` to normalize paths
- Avoid user-controlled path components

```python
from pathlib import Path

# Construct safe path
base_dir = Path("/app/data")
file_name = InputValidator.validate_alphanumeric(user_input, "file_name")
safe_path = base_dir / f"{file_name}.json"
```

---

## Performance Considerations

### Caching Validation Results

For expensive validations (e.g., UUID format), consider caching:

```python
from functools import lru_cache

@lru_cache(maxsize=10000)
def validate_and_cache_uuid(uuid_str: str) -> str:
    """Cache UUID validation results."""
    return InputValidator.validate_uuid(uuid_str, "uuid")
```

### Batch Validation

For large datasets, validate in batches:

```python
def validate_batch(values: List[str], validator_func) -> List[str]:
    """Validate list of values."""
    validated = []
    errors = []

    for i, value in enumerate(values):
        try:
            validated.append(validator_func(value, f"item[{i}]"))
        except ValueError as e:
            errors.append((i, str(e)))

    if errors:
        raise ValueError(f"Validation failed for {len(errors)} items: {errors[:5]}")

    return validated
```

---

## Additional Resources

- **OWASP Top 10:** https://owasp.org/www-project-top-ten/
- **SQL Injection Prevention:** https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html
- **Command Injection Prevention:** https://cheatsheetseries.owasp.org/cheatsheets/OS_Command_Injection_Defense_Cheat_Sheet.html
- **Input Validation Best Practices:** https://cheatsheetseries.owasp.org/cheatsheets/Input_Validation_Cheat_Sheet.html

---

## Support

For questions or issues:
1. Check this guide
2. Review test cases in `testing/security_tests/`
3. Contact security team
4. Open an issue in the repository

---

**Remember: Input validation is your first line of defense. Always validate, never trust user input.**
