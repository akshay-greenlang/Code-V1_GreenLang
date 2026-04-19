"""
FastAPI Integration Example for GreenLang Security

This module demonstrates how to integrate all security features
into a FastAPI application.
"""

import os
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, Depends, Request, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from greenlang.api.security import (
    setup_security,
    create_security_config,
    CSRFProtect,
    CSRFConfig,
    RateLimiter,
    generate_csrf_token
)

# Environment-based configuration
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CSRF_SECRET = os.getenv("CSRF_SECRET_KEY", "your-secret-key-change-in-production")

# Create FastAPI app
app = FastAPI(
    title="GreenLang API",
    description="Secure API with CSRF, Rate Limiting, and Security Headers",
    version="1.0.0"
)

# Create security configuration based on environment
if ENVIRONMENT == "production":
    security_config = create_security_config(
        csrf_secret_key=CSRF_SECRET,
        redis_url=REDIS_URL,
        default_rate_limit="100/minute",
        security_preset="strict",
        enable_hsts=True,
        enable_csp=True,
        csp_report_uri="/api/csp-report"
    )
elif ENVIRONMENT == "staging":
    security_config = create_security_config(
        csrf_secret_key=CSRF_SECRET,
        redis_url=REDIS_URL,
        default_rate_limit="200/minute",
        security_preset="balanced",
        enable_hsts=True,
        enable_csp=True
    )
else:  # development
    security_config = create_security_config(
        csrf_secret_key=CSRF_SECRET,
        redis_url=None,  # Use local rate limiting in development
        default_rate_limit="1000/minute",
        security_preset="relaxed",
        enable_hsts=False,
        enable_csp=False
    )

# Setup all security middleware
setup_security(
    app,
    csrf_config=security_config["csrf"],
    rate_limit_config=security_config["rate_limit"],
    headers_config=security_config["headers"]
)

# Initialize security components for use in endpoints
csrf_protect = CSRFProtect(security_config["csrf"])
rate_limiter = RateLimiter(security_config["rate_limit"])


# Request/Response models
class LoginRequest(BaseModel):
    """Login request model."""
    username: str
    password: str
    csrf_token: Optional[str] = None


class DataRequest(BaseModel):
    """Data creation request."""
    name: str
    value: str
    csrf_token: Optional[str] = None


class TokenResponse(BaseModel):
    """CSRF token response."""
    csrf_token: str
    expires_in: int


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint - no security restrictions."""
    return {
        "message": "GreenLang Secure API",
        "environment": ENVIRONMENT,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    """Health check endpoint - exempt from security middleware."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/csrf-token")
async def get_csrf_token(request: Request):
    """
    Get CSRF token for state-changing operations.

    This endpoint is read-only and doesn't require CSRF protection itself.
    """
    token = csrf_protect.generate_csrf_token()

    response = JSONResponse(
        content={
            "csrf_token": f"{token.token}:{token.signature}",
            "expires_in": security_config["csrf"].token_expiry_seconds
        }
    )

    # Also set token in cookie for double-submit pattern
    response.set_cookie(
        key=security_config["csrf"].cookie_name,
        value=f"{token.token}:{token.signature}",
        secure=security_config["csrf"].cookie_secure,
        httponly=security_config["csrf"].cookie_httponly,
        samesite=security_config["csrf"].cookie_samesite,
        max_age=security_config["csrf"].token_expiry_seconds
    )

    return response


@app.post("/api/auth/login")
@rate_limiter.limit("5/minute")  # Strict rate limit for login
async def login(request: Request, login_data: LoginRequest):
    """
    Login endpoint with rate limiting and CSRF protection.

    Rate limited to 5 requests per minute per IP.
    """
    # Validate CSRF token for state-changing operation
    if login_data.csrf_token:
        token_parts = login_data.csrf_token.split(":")
        if len(token_parts) == 2:
            token_value, signature = token_parts
            if not csrf_protect.validate_token(token_value, signature):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid CSRF token"
                )

    # Simulate authentication (replace with real logic)
    if login_data.username == "admin" and login_data.password == "password":
        return {
            "access_token": "fake-jwt-token",
            "token_type": "bearer",
            "expires_in": 3600
        }

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials"
    )


@app.post("/api/data")
@rate_limiter.limit("20/minute")  # Moderate rate limit for data creation
async def create_data(request: Request, data: DataRequest):
    """
    Create data endpoint with CSRF protection and rate limiting.

    Rate limited to 20 requests per minute per IP.
    """
    # Validate CSRF token
    if data.csrf_token:
        token_parts = data.csrf_token.split(":")
        if len(token_parts) == 2:
            token_value, signature = token_parts
            if not csrf_protect.validate_token(token_value, signature):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid CSRF token"
                )
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF token required"
        )

    # Simulate data creation
    return {
        "id": "12345",
        "name": data.name,
        "value": data.value,
        "created_at": datetime.now().isoformat()
    }


@app.get("/api/data/{data_id}")
@rate_limiter.limit("100/minute")  # Standard rate limit for reads
async def get_data(request: Request, data_id: str):
    """
    Get data endpoint - read-only, no CSRF required.

    Rate limited to 100 requests per minute per IP.
    """
    # Simulate data retrieval
    return {
        "id": data_id,
        "name": "Sample Data",
        "value": "Sample Value",
        "created_at": datetime.now().isoformat()
    }


@app.post("/api/calculate")
@rate_limiter.limit("10/minute")  # CPU-intensive operation
async def calculate(request: Request, data: dict):
    """
    Calculation endpoint with strict rate limiting.

    Rate limited to 10 requests per minute to prevent resource exhaustion.
    """
    # Validate CSRF token from headers
    csrf_token = request.headers.get("X-CSRF-Token")
    if not csrf_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF token required in header"
        )

    token_parts = csrf_token.split(":")
    if len(token_parts) != 2:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid CSRF token format"
        )

    token_value, signature = token_parts
    if not csrf_protect.validate_token(token_value, signature):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid CSRF token"
        )

    # Simulate calculation
    return {
        "result": sum(data.get("values", [])),
        "calculation_id": "calc_12345",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/csp-report")
async def csp_report(request: Request):
    """
    Endpoint to receive CSP violation reports.

    This should log violations for security monitoring.
    """
    report = await request.json()

    # Log CSP violation (in production, send to monitoring system)
    print(f"CSP Violation Report: {report}")

    return {"status": "reported"}


# Error handlers

@app.exception_handler(429)
async def rate_limit_handler(request: Request, exc):
    """Custom handler for rate limit exceeded."""
    return JSONResponse(
        status_code=429,
        content={
            "detail": "Rate limit exceeded. Please try again later.",
            "retry_after": request.headers.get("Retry-After", "60")
        },
        headers={
            "Retry-After": request.headers.get("Retry-After", "60")
        }
    )


@app.exception_handler(403)
async def csrf_error_handler(request: Request, exc):
    """Custom handler for CSRF errors."""
    return JSONResponse(
        status_code=403,
        content={
            "detail": "CSRF validation failed. Please refresh and try again.",
            "error": "csrf_validation_failed"
        }
    )


if __name__ == "__main__":
    import uvicorn

    # Run with appropriate settings based on environment
    if ENVIRONMENT == "production":
        uvicorn.run(
            "integration_example:app",
            host="0.0.0.0",
            port=8000,
            ssl_keyfile="path/to/key.pem",
            ssl_certfile="path/to/cert.pem",
            log_level="info"
        )
    else:
        uvicorn.run(
            "integration_example:app",
            host="127.0.0.1",
            port=8000,
            reload=True,
            log_level="debug"
        )