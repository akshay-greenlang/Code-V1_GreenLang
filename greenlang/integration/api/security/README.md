# GreenLang API Security

Comprehensive security features for the GreenLang API including CSRF protection, rate limiting, and security headers.

## Features

### 1. CSRF Protection
- **Double-submit cookie pattern** with HMAC signatures
- **Token generation and validation**
- **Configurable token expiry**
- **Exempt paths** for read-only endpoints
- **FastAPI middleware integration**

### 2. Rate Limiting
- **Token bucket algorithm** for smooth rate limiting
- **Redis-backed** for distributed systems
- **Per-IP and per-user limits**
- **Configurable limits by endpoint**
- **Retry-After headers** for client guidance
- **Local fallback** when Redis is unavailable

### 3. Security Headers
- **X-Content-Type-Options**: Prevent MIME type sniffing
- **X-Frame-Options**: Prevent clickjacking
- **X-XSS-Protection**: Enable browser XSS protection
- **Strict-Transport-Security**: Force HTTPS connections
- **Content-Security-Policy**: Control resource loading
- **Referrer-Policy**: Control referrer information
- **Permissions-Policy**: Control browser features

## Quick Start

### Basic Setup

```python
from fastapi import FastAPI
from greenlang.api.security import setup_security

app = FastAPI(title="GreenLang API")

# Setup with defaults
setup_security(app)
```

### Custom Configuration

```python
from greenlang.api.security import setup_security, create_security_config

# Create configuration
config = create_security_config(
    csrf_secret_key="your-secret-key",
    redis_url="redis://localhost:6379",
    default_rate_limit="100/minute",
    security_preset="strict"
)

# Apply to app
setup_security(
    app,
    csrf_config=config["csrf"],
    rate_limit_config=config["rate_limit"],
    headers_config=config["headers"]
)
```

## Configuration Examples

### CSRF Protection

```python
from greenlang.api.security import CSRFConfig, CSRFProtect

csrf_config = CSRFConfig(
    secret_key="your-secret-key",
    token_length=32,
    token_expiry_seconds=3600,
    cookie_name="csrf_token",
    header_name="X-CSRF-Token",
    safe_methods={"GET", "HEAD", "OPTIONS"},
    exempt_paths={"/health", "/metrics"},
    cookie_secure=True,
    cookie_samesite="strict"
)

csrf_protect = CSRFProtect(csrf_config)
```

### Rate Limiting

```python
from greenlang.api.security import RateLimitConfig, RateLimiter

rate_limit_config = RateLimitConfig(
    redis_url="redis://localhost:6379",
    default_limit=100,
    default_period="minute",
    endpoint_limits={
        "/api/auth/login": "5/minute",
        "/api/auth/register": "3/minute",
        "/api/calculate": "20/minute"
    },
    burst_multiplier=1.5,
    include_retry_after=True
)

limiter = RateLimiter(rate_limit_config)

# Use as decorator
@app.post("/api/data")
@limiter.limit("10/minute")
async def create_data(data: dict):
    return {"created": True}
```

### Security Headers

```python
from greenlang.api.security import SecurityHeadersConfig, SECURITY_PRESETS

# Use preset
headers_config = SECURITY_PRESETS["strict"]

# Or customize
headers_config = SecurityHeadersConfig(
    x_content_type_options="nosniff",
    x_frame_options="DENY",
    x_xss_protection="1; mode=block",
    strict_transport_security="max-age=31536000; includeSubDomains",
    enable_hsts=True,
    enable_csp=True,
    content_security_policy={
        "default_src": ["'self'"],
        "script_src": ["'self'", "'unsafe-inline'"],
        "style_src": ["'self'", "'unsafe-inline'"],
        "img_src": ["'self'", "data:", "https:"]
    },
    referrer_policy="strict-origin-when-cross-origin"
)
```

## Environment-Based Configuration

```python
import os

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

if ENVIRONMENT == "production":
    config = create_security_config(
        csrf_secret_key=os.getenv("CSRF_SECRET"),
        redis_url=os.getenv("REDIS_URL"),
        security_preset="strict",
        enable_hsts=True,
        enable_csp=True
    )
elif ENVIRONMENT == "staging":
    config = create_security_config(
        security_preset="balanced",
        enable_hsts=True
    )
else:  # development
    config = create_security_config(
        security_preset="relaxed",
        enable_hsts=False,
        enable_csp=False
    )
```

## Security Presets

### Strict (Production)
- Denies all framing
- Strict CSP with no inline scripts/styles
- No referrer information sent
- All security headers enabled

### Balanced (Default)
- Allows same-origin framing
- CSP allows unsafe-inline for compatibility
- Strict origin referrer policy
- Most security headers enabled

### Relaxed (Development)
- Allows same-origin framing
- CSP disabled for easier debugging
- Origin referrer policy
- Basic security headers only

## API Endpoint Protection

### Protected Endpoint (State-Changing)

```python
@app.post("/api/data")
async def create_data(request: Request, data: DataModel):
    # CSRF token required
    csrf_token = request.headers.get("X-CSRF-Token")
    if not csrf_token:
        raise HTTPException(403, "CSRF token required")

    # Validate token
    if not csrf_protect.validate_token(csrf_token):
        raise HTTPException(403, "Invalid CSRF token")

    # Process request
    return {"created": True}
```

### Read-Only Endpoint

```python
@app.get("/api/data/{id}")
async def get_data(id: str):
    # No CSRF required for GET requests
    return {"id": id, "data": "..."}
```

## Testing

### Test CSRF Protection

```bash
# Get CSRF token
curl -X GET http://localhost:8000/api/csrf-token

# Use token in POST request
curl -X POST http://localhost:8000/api/data \
  -H "X-CSRF-Token: <token>" \
  -H "Content-Type: application/json" \
  -d '{"name": "test"}'
```

### Test Rate Limiting

```bash
# Make multiple requests to test rate limit
for i in {1..15}; do
  curl -X GET http://localhost:8000/api/data/test
  echo
done

# Check headers for rate limit info
curl -I http://localhost:8000/api/data/test
# X-RateLimit-Limit: 100
# X-RateLimit-Remaining: 99
# X-RateLimit-Reset: 1234567890
```

### Test Security Headers

```bash
# Check response headers
curl -I http://localhost:8000/api/data/test

# Expected headers:
# X-Content-Type-Options: nosniff
# X-Frame-Options: DENY
# X-XSS-Protection: 1; mode=block
# Strict-Transport-Security: max-age=31536000
# Content-Security-Policy: default-src 'self'
```

## Best Practices

### CSRF Protection
1. **Always validate tokens** for state-changing operations
2. **Use HTTPS** in production for secure cookie transmission
3. **Rotate secret keys** periodically
4. **Exempt only safe methods** (GET, HEAD, OPTIONS)
5. **Log validation failures** for security monitoring

### Rate Limiting
1. **Use Redis** for distributed systems
2. **Set appropriate limits** based on endpoint complexity
3. **Include Retry-After headers** for client guidance
4. **Monitor rate limit violations** for abuse detection
5. **Implement user-specific limits** for authenticated endpoints

### Security Headers
1. **Use strict CSP** in production
2. **Enable HSTS** with preload for HTTPS sites
3. **Test CSP policies** before enforcing
4. **Monitor CSP violations** via report-uri
5. **Update headers** based on security requirements

## Monitoring

### Metrics to Track
- CSRF validation failures
- Rate limit violations per endpoint
- CSP violation reports
- Security header compliance
- Response times with security middleware

### Logging

```python
import logging

# Configure security logging
logging.basicConfig(level=logging.INFO)
security_logger = logging.getLogger("greenlang.api.security")
security_logger.setLevel(logging.DEBUG)

# Log security events
security_logger.info("CSRF token validated successfully")
security_logger.warning("Rate limit exceeded for IP: 192.168.1.1")
security_logger.error("CSP violation detected")
```

## Troubleshooting

### CSRF Token Issues
- **Token missing**: Ensure client sends token in header or cookie
- **Token expired**: Refresh token before expiry
- **Signature mismatch**: Check secret key configuration

### Rate Limiting Issues
- **Redis connection failed**: Falls back to local limiting
- **Limits not applied**: Check endpoint pattern matching
- **Too restrictive**: Adjust limits based on usage patterns

### Security Header Issues
- **CSP blocking resources**: Adjust CSP directives
- **HSTS issues**: Only enabled on HTTPS
- **Frame blocking**: Adjust X-Frame-Options if embedding needed

## Performance Considerations

### CSRF
- Token validation: ~0.1ms per request
- Token generation: ~0.2ms
- Memory usage: ~100 bytes per active token

### Rate Limiting
- Redis check: ~1-2ms per request
- Local check: ~0.01ms per request
- Memory usage: ~200 bytes per bucket

### Security Headers
- Header addition: ~0.01ms per response
- CSP parsing: ~0.1ms
- Negligible memory impact

## Production Deployment

### Environment Variables

```bash
# Required
CSRF_SECRET_KEY="strong-random-secret"
REDIS_URL="redis://redis:6379/0"

# Optional
ENVIRONMENT="production"
RATE_LIMIT_DEFAULT="100/minute"
CSP_REPORT_URI="https://api.example.com/csp-report"
```

### Docker Configuration

```dockerfile
FROM python:3.9

# Install dependencies
RUN pip install redis[asyncio]

# Security configuration
ENV CSRF_SECRET_KEY=${CSRF_SECRET_KEY}
ENV REDIS_URL=${REDIS_URL}
ENV ENVIRONMENT=production

# Run with security enabled
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes Configuration

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: security-config
data:
  ENVIRONMENT: "production"
  REDIS_URL: "redis://redis-service:6379"
  RATE_LIMIT_DEFAULT: "100/minute"
---
apiVersion: v1
kind: Secret
metadata:
  name: security-secrets
type: Opaque
data:
  CSRF_SECRET_KEY: <base64-encoded-secret>
```

## License

Part of the GreenLang Framework - Enterprise Sustainability Platform