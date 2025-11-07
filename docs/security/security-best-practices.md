# GreenLang Security Best Practices

## For Developers

### 1. Input Validation

**Always validate and sanitize user input:**

```python
from greenlang.security import (
    ValidationError,
    XSSValidator,
    PathTraversalValidator,
    validate_username,
)

# Validate usernames
try:
    safe_username = validate_username(user_input)
except ValidationError as e:
    # Handle validation error
    logger.warning(f"Invalid username: {e}")
    return {"error": "Invalid username"}

# Prevent XSS
try:
    safe_html = XSSValidator.sanitize_html(user_content)
except ValidationError as e:
    return {"error": "Invalid content"}

# Prevent path traversal
try:
    safe_path = PathTraversalValidator.validate_path(
        file_path,
        base_dir="/safe/directory",
        must_exist=True
    )
except ValidationError as e:
    return {"error": "Invalid file path"}
```

### 2. Audit Logging

**Log all security-sensitive operations:**

```python
from greenlang.security import get_audit_logger

audit = get_audit_logger()

# Log authentication
audit.log_auth_success(
    user_id="user123",
    username="john.doe",
    ip_address=request.remote_addr
)

# Log authorization decisions
audit.log_authz_decision(
    user_id="user123",
    resource_type="agent",
    resource_id="fuel_agent",
    action="execute",
    allowed=True,
    reason="User has required permissions"
)

# Log agent execution
audit.log_agent_execution(
    agent_name="FuelAgent",
    user_id="user123",
    result="success",
    execution_time_ms=1234.56,
    details={"emissions_kg_co2": 123.45}
)

# Log security violations
audit.log_security_violation(
    violation_type="rate_limit_exceeded",
    description="User exceeded rate limit",
    user_id="user123",
    ip_address=request.remote_addr,
    details={"requests_per_minute": 150}
)
```

### 3. Secrets Management

**Never hardcode secrets:**

```python
import os
from pathlib import Path
from dotenv import load_dotenv

# Load from environment
load_dotenv()

# Good: From environment variable
API_KEY = os.getenv("GREENLANG_API_KEY")
if not API_KEY:
    raise ValueError("GREENLANG_API_KEY environment variable not set")

# Bad: Hardcoded
# API_KEY = "gl_1234567890abcdef"  # DON'T DO THIS!

# Good: From secure file with restricted permissions
def load_api_key():
    key_file = Path.home() / ".greenlang" / "api_key"
    if not key_file.exists():
        raise FileNotFoundError("API key file not found")

    # Check file permissions (Unix-like systems)
    if key_file.stat().st_mode & 0o077:
        raise PermissionError("API key file has insecure permissions")

    return key_file.read_text().strip()
```

### 4. Secure API Key Generation

```python
import secrets
import string

def generate_api_key(prefix: str = "gl_", length: int = 32) -> str:
    """Generate cryptographically secure API key."""
    # Use secrets module for cryptographic randomness
    alphabet = string.ascii_letters + string.digits
    key_part = ''.join(secrets.choice(alphabet) for _ in range(length))
    return f"{prefix}{key_part}"

# Usage
api_key = generate_api_key()
print(f"Generated API key: {api_key}")
```

### 5. Secure File Operations

```python
from greenlang.security import PathTraversalValidator

def safe_file_write(filename: str, content: str, base_dir: str):
    """Safely write file preventing path traversal."""
    try:
        # Validate and sanitize filename
        safe_filename = PathTraversalValidator.sanitize_filename(filename)

        # Validate path is within base directory
        full_path = PathTraversalValidator.validate_path(
            Path(base_dir) / safe_filename,
            base_dir=base_dir
        )

        # Write file
        full_path.write_text(content)

        # Audit log the operation
        audit = get_audit_logger()
        audit.log_data_access(
            user_id=current_user.id,
            data_type="file",
            data_id=str(full_path),
            operation="write",
            result="success"
        )

    except ValidationError as e:
        logger.error(f"File validation failed: {e}")
        raise
```

### 6. Secure Command Execution

```python
import subprocess
from greenlang.security import CommandInjectionValidator

def safe_execute_command(cmd_list: list[str]):
    """Safely execute command preventing injection."""
    try:
        # Validate command list
        safe_cmd = CommandInjectionValidator.validate_command_list(cmd_list)

        # Execute with shell=False (safer)
        result = subprocess.run(
            safe_cmd,
            shell=False,  # IMPORTANT: Never use shell=True with user input
            capture_output=True,
            text=True,
            timeout=30,
            check=True
        )

        return result.stdout

    except ValidationError as e:
        logger.error(f"Command validation failed: {e}")
        raise
    except subprocess.TimeoutExpired:
        logger.error("Command timeout")
        raise
```

## For System Administrators

### 1. Environment Configuration

Create `.env` file with secure settings:

```bash
# Application
GREENLANG_ENV=production
GREENLANG_DEBUG=false

# Security
GREENLANG_SECRET_KEY=<generate-strong-secret>
GREENLANG_API_KEY_ROTATION_DAYS=90

# Database (use encrypted connection)
DATABASE_URL=postgresql://user:pass@localhost:5432/greenlang?sslmode=require

# Redis (use TLS)
REDIS_URL=rediss://localhost:6379

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=60

# Audit Logging
AUDIT_LOG_ENABLED=true
AUDIT_LOG_PATH=/var/log/greenlang/audit.jsonl

# CORS
CORS_ALLOWED_ORIGINS=https://app.greenlang.io,https://admin.greenlang.io
```

### 2. File Permissions

```bash
# Secure configuration files
chmod 600 ~/.greenlang/config.yaml
chmod 600 ~/.greenlang/api_key
chmod 700 ~/.greenlang

# Secure log files
chmod 640 /var/log/greenlang/*.log
chown greenlang:greenlang /var/log/greenlang/

# Secure application directory
chmod 755 /opt/greenlang
chown greenlang:greenlang /opt/greenlang
```

### 3. Firewall Configuration

```bash
# Allow only necessary ports
ufw default deny incoming
ufw default allow outgoing
ufw allow 443/tcp  # HTTPS
ufw allow 22/tcp   # SSH (consider changing port)
ufw enable

# Rate limiting for SSH
ufw limit 22/tcp
```

### 4. Regular Security Scans

```bash
# Run daily via cron
0 2 * * * cd /opt/greenlang && bash security/dependency-scan.sh
0 3 * * * cd /opt/greenlang && bandit -c .bandit -r greenlang core -o /var/log/greenlang/bandit-$(date +\%Y\%m\%d).txt
```

### 5. TLS/SSL Configuration

Use strong TLS configuration:

```nginx
# Nginx example
ssl_protocols TLSv1.3 TLSv1.2;
ssl_ciphers 'TLS_AES_128_GCM_SHA256:TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256';
ssl_prefer_server_ciphers off;
ssl_session_cache shared:SSL:10m;
ssl_session_timeout 10m;
ssl_stapling on;
ssl_stapling_verify on;

# Security headers
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
add_header X-Frame-Options "DENY" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
```

## For Security Teams

### 1. Regular Security Audits

Monthly checklist:
- [ ] Review audit logs for anomalies
- [ ] Check for failed authentication attempts
- [ ] Review API key usage
- [ ] Scan for vulnerabilities
- [ ] Update dependencies
- [ ] Review firewall rules
- [ ] Test backup restoration
- [ ] Review access controls

### 2. Vulnerability Management

```bash
# Weekly vulnerability scanning
pip-audit --desc
bandit -c .bandit -r greenlang core

# Check for outdated packages
pip list --outdated

# Review CVE databases
# - https://nvd.nist.gov/
# - https://www.cvedetails.com/
```

### 3. Incident Response Testing

Quarterly drills:
1. Simulate data breach
2. Test communication procedures
3. Verify backup restoration
4. Test failover procedures
5. Document lessons learned

### 4. Security Metrics

Track these metrics:
- Failed authentication attempts per day
- Rate limit violations per day
- Security scan findings over time
- Mean time to patch vulnerabilities
- Audit log volume and alerts
- API key age distribution

## Common Security Anti-Patterns to Avoid

### ❌ Don't: Trust User Input
```python
# Bad
def delete_file(filename):
    os.remove(filename)  # Path traversal risk!
```

### ✅ Do: Validate Everything
```python
# Good
def delete_file(filename):
    safe_path = PathTraversalValidator.validate_path(
        filename,
        base_dir="/safe/directory"
    )
    os.remove(safe_path)
```

### ❌ Don't: Use exec() or eval()
```python
# Bad
user_code = request.json["code"]
exec(user_code)  # Code injection!
```

### ✅ Do: Use Sandboxed Execution
```python
# Good
from greenlang.sandbox import Sandbox
sandbox = Sandbox()
result = sandbox.execute(user_code, timeout=5)
```

### ❌ Don't: Hardcode Secrets
```python
# Bad
API_KEY = "gl_1234567890abcdef"
```

### ✅ Do: Use Environment Variables
```python
# Good
API_KEY = os.getenv("GREENLANG_API_KEY")
```

### ❌ Don't: Use Weak Randomness
```python
# Bad
import random
token = str(random.randint(1000000, 9999999))
```

### ✅ Do: Use Cryptographic Randomness
```python
# Good
import secrets
token = secrets.token_urlsafe(32)
```

## Security Training Resources

- OWASP Top 10: https://owasp.org/www-project-top-ten/
- SANS Security Training: https://www.sans.org/
- Python Security: https://python.readthedocs.io/en/stable/library/security.html
- Cloud Security Alliance: https://cloudsecurityalliance.org/

## Questions?

Contact the security team: security@greenlang.io
