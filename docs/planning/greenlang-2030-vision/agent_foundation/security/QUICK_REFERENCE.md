# Input Validation - Quick Reference Card

## Import

```python
from security.input_validation import InputValidator
```

## Basic Validation

| Method | Use For | Example |
|--------|---------|---------|
| `validate_alphanumeric()` | IDs, names | `InputValidator.validate_alphanumeric(value, "field")` |
| `validate_uuid()` | User IDs | `InputValidator.validate_uuid(value, "user_id")` |
| `validate_email()` | Email addresses | `InputValidator.validate_email(value)` |
| `validate_integer()` | Numbers | `InputValidator.validate_integer(value, "count", min_value=0, max_value=100)` |

## Security Validation

| Method | Prevents | Example |
|--------|----------|---------|
| `validate_no_sql_injection()` | SQL injection | `InputValidator.validate_no_sql_injection(value, "field")` |
| `validate_no_command_injection()` | Command injection | `InputValidator.validate_no_command_injection(value, "arg")` |
| `validate_no_xss()` | XSS attacks | `InputValidator.validate_no_xss(value, "field")` |
| `validate_path()` | Path traversal | `InputValidator.validate_path(value, must_exist=False)` |
| `validate_url()` | SSRF attacks | `InputValidator.validate_url(value, allowed_schemes=['https'])` |
| `validate_ip_address()` | SSRF attacks | `InputValidator.validate_ip_address(value, allow_private=False)` |

## Whitelist Validation

| Method | Use For | Example |
|--------|---------|---------|
| `validate_field_name()` | SQL field names | `InputValidator.validate_field_name("tenant_id")` |
| `validate_operator()` | SQL operators | `InputValidator.validate_operator("=")` |
| `validate_command()` | Shell commands | `InputValidator.validate_command("kubectl", ["kubectl", "docker"])` |

## Pydantic Models

```python
from security.input_validation import (
    TenantIdModel,
    UserIdModel,
    EmailModel,
    SafeQueryInput,
    SafePathInput,
    SafeUrlInput,
    SafeCommandInput,
)

# Use in FastAPI
class CreateUserRequest(BaseModel):
    tenant_id: str
    email: str

    @validator('tenant_id')
    def validate_tenant(cls, v):
        return InputValidator.validate_alphanumeric(v, 'tenant_id')
```

## Database (SQL Injection Prevention)

```python
from database.postgres_manager_secure import SecureQueryBuilder
from security.input_validation import SafeQueryInput

# Build safe query
builder = SecureQueryBuilder("agents")

filters = [
    SafeQueryInput(field="tenant_id", value="tenant-123", operator="=")
]

query, params = builder.build_select(filters=filters, limit=100, offset=0)

# Execute
results = await conn.fetch(query, *params)
```

## Commands (Command Injection Prevention)

```python
from factory.deployment_secure import SecureCommandExecutor

executor = SecureCommandExecutor()

# Execute kubectl
result = executor.execute_kubectl(
    command="get",
    resource_type="pods",
    namespace="default"
)

# Execute docker
result = executor.execute_docker(
    command="build",
    image="myapp",
    tag="v1.0"
)

# Execute helm
result = executor.execute_helm(
    command="install",
    release_name="myapp",
    chart="stable/nginx"
)
```

## API Middleware

```python
from fastapi import FastAPI
from api.middleware.validation import RequestValidationMiddleware

app = FastAPI()

# Add validation middleware
app.add_middleware(RequestValidationMiddleware)
```

## Common Patterns

### Pattern 1: Validate API Request

```python
def create_agent(data: dict):
    # Validate inputs
    tenant_id = InputValidator.validate_alphanumeric(
        data.get("tenant_id"), "tenant_id"
    )
    name = InputValidator.validate_alphanumeric(
        data.get("name"), "name"
    )

    # Process validated data
    ...
```

### Pattern 2: Safe Database Insert

```python
from database.postgres_manager_secure import SecureQueryBuilder

builder = SecureQueryBuilder("users")

data = {
    "tenant_id": validated_tenant_id,
    "email": validated_email,
    "name": validated_name,
}

query, params = builder.build_insert(data)
result = await conn.fetchrow(query, *params)
```

### Pattern 3: Safe File Upload

```python
# Validate filename
filename = InputValidator.validate_alphanumeric(
    user_filename, "filename"
)

# Validate path
base_dir = Path("/app/data")
full_path = base_dir / f"{filename}.json"

validated_path = InputValidator.validate_path(
    str(full_path),
    allowed_extensions=['.json']
)

# Safe to use
with open(validated_path, 'w') as f:
    f.write(data)
```

### Pattern 4: Safe URL Fetch

```python
# Validate URL
url = InputValidator.validate_url(
    user_url,
    allowed_schemes=['https'],
    allow_private_ips=False
)

# Safe to fetch
response = requests.get(url, timeout=10)
```

## Error Handling

```python
try:
    tenant_id = InputValidator.validate_alphanumeric(
        user_input, "tenant_id"
    )
except ValueError as e:
    # Log security event
    logger.warning(f"Validation failed: {e}")

    # Return error to user
    raise HTTPException(400, "Invalid tenant ID format")
```

## Testing

```bash
# Run all security tests
pytest testing/security_tests/ -v

# Run specific test
pytest testing/security_tests/test_input_validation.py -v

# Run with coverage
pytest testing/security_tests/ --cov=security
```

## Whitelists

### Allowed Fields
```python
'tenant_id', 'user_id', 'agent_id', 'execution_id',
'name', 'email', 'status', 'type', 'tier',
'created_at', 'updated_at'
```

### Allowed Operators
```python
'=', '!=', '>', '<', '>=', '<=', 'IN', 'LIKE'
```

### Allowed kubectl Commands
```python
'get', 'describe', 'logs', 'apply', 'delete', 'create',
'rollout', 'scale'
```

### Allowed docker Commands
```python
'build', 'push', 'pull', 'tag', 'images', 'ps', 'inspect'
```

### Allowed helm Commands
```python
'install', 'upgrade', 'uninstall', 'list', 'status', 'rollback'
```

## Performance

- Validation: <1ms per field
- Query building: <5ms
- Request validation: <2ms

## Security Checklist

- [ ] Validate ALL user inputs
- [ ] Use parameterized queries (never string concatenation)
- [ ] Use shell=False for subprocess
- [ ] Whitelist field names
- [ ] Whitelist operators
- [ ] Whitelist commands
- [ ] Validate file paths
- [ ] Validate URLs
- [ ] Rate limit API endpoints
- [ ] Log security events

## Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| "Field not in whitelist" | Field name not allowed | Add to ALLOWED_FIELDS |
| "SQL injection detected" | Dangerous SQL pattern | Use parameterized queries |
| "Command not allowed" | Command not whitelisted | Use allowed command or add to whitelist |
| "Path traversal detected" | Dangerous path pattern | Use absolute paths only |

## Resources

- **Guide:** `security/INPUT_VALIDATION_GUIDE.md`
- **Examples:** `security/examples.py`
- **Tests:** `testing/security_tests/`
- **Summary:** `security/IMPLEMENTATION_SUMMARY.md`

---

**Print this card and keep it handy while coding!**
