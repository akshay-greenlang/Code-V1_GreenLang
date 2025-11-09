"""
Request Signing & Verification System
GL-VCCI Scope 3 Platform

HMAC-based request signing for critical operations:
- Batch upload requests
- Report generation requests
- Data export requests
- Configuration changes

Provides:
- HMAC-SHA256 signature generation
- Timestamp validation (prevent replay attacks)
- Nonce tracking (prevent duplicate requests)
- Signature verification middleware

Version: 1.0.0
Security Enhancement: 2025-11-09
"""

import os
import hmac
import hashlib
import secrets
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from fastapi import Request, HTTPException, status, Depends
import redis.asyncio as redis

logger = logging.getLogger(__name__)

# Configuration
SIGNING_SECRET = os.getenv("REQUEST_SIGNING_SECRET")
SIGNATURE_ALGORITHM = "hmac-sha256"
TIMESTAMP_TOLERANCE_SECONDS = int(os.getenv("REQUEST_TIMESTAMP_TOLERANCE", "300"))  # 5 minutes
NONCE_TTL_SECONDS = int(os.getenv("REQUEST_NONCE_TTL", "600"))  # 10 minutes

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

# Redis client (lazy initialization)
_redis_client: Optional[redis.Redis] = None


class SignatureError(HTTPException):
    """Request signature validation error."""

    def __init__(self, detail: str = "Invalid request signature"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Signature"},
        )


async def get_redis_client() -> redis.Redis:
    """
    Get or create Redis client for nonce tracking.

    Returns:
        Redis client instance
    """
    global _redis_client

    if _redis_client is None:
        _redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASSWORD,
            decode_responses=True,
        )

        # Test connection
        try:
            await _redis_client.ping()
            logger.info("Connected to Redis for nonce tracking")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            raise

    return _redis_client


def generate_nonce() -> str:
    """
    Generate a cryptographically secure nonce.

    Returns:
        Random nonce string

    Example:
        >>> nonce = generate_nonce()
    """
    return secrets.token_urlsafe(32)


def generate_timestamp() -> str:
    """
    Generate current UTC timestamp in ISO format.

    Returns:
        ISO format timestamp string

    Example:
        >>> timestamp = generate_timestamp()
    """
    return datetime.utcnow().isoformat()


def compute_signature(
    method: str,
    path: str,
    timestamp: str,
    nonce: str,
    body: str = "",
    secret: Optional[str] = None,
) -> str:
    """
    Compute HMAC-SHA256 signature for a request.

    The signature is computed over:
    - HTTP method (GET, POST, etc.)
    - Request path
    - Timestamp
    - Nonce
    - Request body (if present)

    Args:
        method: HTTP method (uppercase)
        path: Request path (including query params)
        timestamp: ISO format timestamp
        nonce: Unique request nonce
        body: Request body (JSON string)
        secret: Signing secret (defaults to SIGNING_SECRET)

    Returns:
        Hex-encoded HMAC signature

    Example:
        >>> signature = compute_signature(
        ...     "POST",
        ...     "/api/batch-upload",
        ...     timestamp,
        ...     nonce,
        ...     json.dumps(body_data)
        ... )
    """
    if secret is None:
        secret = SIGNING_SECRET

    if not secret:
        raise ValueError("Request signing secret not configured")

    # Construct signature payload
    # Format: METHOD\nPATH\nTIMESTAMP\nNONCE\nBODY
    payload_parts = [
        method.upper(),
        path,
        timestamp,
        nonce,
        body,
    ]

    payload = "\n".join(payload_parts)

    # Compute HMAC-SHA256
    signature = hmac.new(
        secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    return signature


def verify_signature(
    method: str,
    path: str,
    timestamp: str,
    nonce: str,
    provided_signature: str,
    body: str = "",
    secret: Optional[str] = None,
) -> bool:
    """
    Verify HMAC signature for a request.

    Args:
        method: HTTP method
        path: Request path
        timestamp: Request timestamp
        nonce: Request nonce
        provided_signature: Signature from request header
        body: Request body
        secret: Signing secret (defaults to SIGNING_SECRET)

    Returns:
        True if signature is valid, False otherwise

    Example:
        >>> if verify_signature(method, path, timestamp, nonce, signature, body):
        ...     print("Valid signature")
    """
    try:
        expected_signature = compute_signature(
            method, path, timestamp, nonce, body, secret
        )

        # Constant-time comparison to prevent timing attacks
        return hmac.compare_digest(expected_signature, provided_signature)

    except Exception as e:
        logger.error(f"Error verifying signature: {str(e)}")
        return False


def verify_timestamp(timestamp: str, tolerance_seconds: int = TIMESTAMP_TOLERANCE_SECONDS) -> bool:
    """
    Verify request timestamp is within acceptable range.

    This prevents replay attacks by rejecting old requests.

    Args:
        timestamp: ISO format timestamp from request
        tolerance_seconds: Maximum age of request in seconds

    Returns:
        True if timestamp is valid, False otherwise

    Example:
        >>> if verify_timestamp(request_timestamp):
        ...     print("Timestamp valid")
    """
    try:
        request_time = datetime.fromisoformat(timestamp)
        now = datetime.utcnow()

        # Check if timestamp is too old
        age = now - request_time
        if age > timedelta(seconds=tolerance_seconds):
            logger.warning(f"Request timestamp too old: {age.total_seconds()}s")
            return False

        # Check if timestamp is in the future (clock skew)
        if request_time > now + timedelta(seconds=60):  # Allow 1 min clock skew
            logger.warning("Request timestamp is in the future")
            return False

        return True

    except (ValueError, TypeError) as e:
        logger.error(f"Invalid timestamp format: {str(e)}")
        return False


async def verify_nonce(nonce: str) -> bool:
    """
    Verify nonce has not been used before.

    This prevents replay attacks by tracking used nonces.

    Args:
        nonce: Request nonce

    Returns:
        True if nonce is valid (not seen before), False otherwise

    Example:
        >>> if await verify_nonce(request_nonce):
        ...     print("Nonce valid")
    """
    try:
        redis_client = await get_redis_client()

        # Check if nonce exists
        key = f"nonce:{nonce}"
        exists = await redis_client.exists(key)

        if exists:
            logger.warning(f"Nonce reuse detected: {nonce}")
            return False

        # Store nonce with TTL
        await redis_client.setex(key, NONCE_TTL_SECONDS, "1")

        return True

    except Exception as e:
        logger.error(f"Error verifying nonce: {str(e)}")
        # Fail secure: treat as invalid if we can't verify
        return False


async def verify_signed_request(
    request: Request,
    require_body_signature: bool = True,
) -> Dict[str, Any]:
    """
    Verify a signed request.

    Checks:
    1. Required headers present
    2. Timestamp validity
    3. Nonce uniqueness
    4. HMAC signature

    Args:
        request: FastAPI request object
        require_body_signature: Whether to include body in signature

    Returns:
        Dictionary with signature metadata

    Raises:
        SignatureError: If signature verification fails

    Example:
        ```python
        @app.post("/api/critical-operation")
        async def critical_op(
            signature_data: dict = Depends(verify_signed_request)
        ):
            return {"status": "ok"}
        ```
    """
    # Extract signature headers
    timestamp = request.headers.get("X-Request-Timestamp")
    nonce = request.headers.get("X-Request-Nonce")
    signature = request.headers.get("X-Request-Signature")

    if not timestamp or not nonce or not signature:
        raise SignatureError(
            "Missing required signature headers: "
            "X-Request-Timestamp, X-Request-Nonce, X-Request-Signature"
        )

    # Verify timestamp
    if not verify_timestamp(timestamp):
        raise SignatureError("Invalid or expired timestamp")

    # Verify nonce (prevent replay)
    if not await verify_nonce(nonce):
        raise SignatureError("Invalid or reused nonce")

    # Get request body if needed
    body = ""
    if require_body_signature and request.method in ["POST", "PUT", "PATCH"]:
        body_bytes = await request.body()
        body = body_bytes.decode("utf-8")

    # Verify signature
    method = request.method
    path = str(request.url.path)

    if request.url.query:
        path = f"{path}?{request.url.query}"

    if not verify_signature(method, path, timestamp, nonce, signature, body):
        raise SignatureError("Invalid signature")

    logger.info(
        f"Verified signed request: {method} {path}, nonce: {nonce}"
    )

    return {
        "timestamp": timestamp,
        "nonce": nonce,
        "method": method,
        "path": path,
    }


# Decorator for signed endpoints
def require_signature(require_body: bool = True):
    """
    Decorator to require signature verification on an endpoint.

    Args:
        require_body: Whether to include body in signature

    Returns:
        FastAPI dependency

    Example:
        ```python
        @app.post("/api/batch-upload")
        async def batch_upload(
            data: dict,
            signature: dict = Depends(require_signature())
        ):
            return {"status": "uploaded"}
        ```
    """

    async def _verify_signature(request: Request):
        return await verify_signed_request(request, require_body)

    return Depends(_verify_signature)


# Client-side signing helper
class RequestSigner:
    """
    Client-side helper for signing requests.

    Example:
        ```python
        signer = RequestSigner(secret)

        headers = signer.sign_request(
            method="POST",
            path="/api/batch-upload",
            body=json.dumps(data)
        )

        response = requests.post(
            url,
            json=data,
            headers=headers
        )
        ```
    """

    def __init__(self, secret: str):
        """
        Initialize request signer.

        Args:
            secret: Signing secret
        """
        self.secret = secret

    def sign_request(
        self,
        method: str,
        path: str,
        body: str = "",
    ) -> Dict[str, str]:
        """
        Sign a request and return headers.

        Args:
            method: HTTP method
            path: Request path
            body: Request body (JSON string)

        Returns:
            Dictionary of signature headers to add to request

        Example:
            >>> headers = signer.sign_request("POST", "/api/upload", body_json)
        """
        timestamp = generate_timestamp()
        nonce = generate_nonce()

        signature = compute_signature(
            method, path, timestamp, nonce, body, self.secret
        )

        return {
            "X-Request-Timestamp": timestamp,
            "X-Request-Nonce": nonce,
            "X-Request-Signature": signature,
        }


# Validation utilities
def validate_signing_config():
    """Validate request signing configuration."""
    if not SIGNING_SECRET:
        logger.warning(
            "REQUEST_SIGNING_SECRET not set. Request signing will fail. "
            "Generate a strong secret: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
        )
    elif len(SIGNING_SECRET) < 32:
        logger.warning(
            f"REQUEST_SIGNING_SECRET is too short ({len(SIGNING_SECRET)} chars). "
            "Recommended: at least 32 characters"
        )
    else:
        logger.info(
            f"Request signing configured: "
            f"algorithm={SIGNATURE_ALGORITHM}, "
            f"timestamp_tolerance={TIMESTAMP_TOLERANCE_SECONDS}s"
        )


# Audit logging for signed requests
async def log_signed_request(
    signature_data: Dict[str, Any],
    user_id: Optional[str] = None,
    ip_address: Optional[str] = None,
):
    """
    Log signed request for audit trail.

    Args:
        signature_data: Signature metadata from verification
        user_id: Optional user identifier
        ip_address: Optional client IP

    Example:
        >>> await log_signed_request(
        ...     signature_data,
        ...     user_id="admin@example.com",
        ...     ip_address="192.168.1.1"
        ... )
    """
    logger.info(
        f"Signed request executed: "
        f"method={signature_data['method']}, "
        f"path={signature_data['path']}, "
        f"user={user_id}, "
        f"ip={ip_address}, "
        f"nonce={signature_data['nonce']}"
    )

    # In production, send to audit log storage
    # await store_audit_log({
    #     "event_type": "signed_request",
    #     "signature_data": signature_data,
    #     "user_id": user_id,
    #     "ip_address": ip_address,
    #     "timestamp": datetime.utcnow().isoformat(),
    # })


# Statistics and monitoring
async def get_signature_stats() -> Dict[str, Any]:
    """
    Get statistics about signature usage.

    Returns:
        Dictionary with statistics

    Example:
        >>> stats = await get_signature_stats()
        >>> print(f"Active nonces: {stats['active_nonces']}")
    """
    redis_client = await get_redis_client()

    # Count active nonces
    nonce_count = 0
    pattern = "nonce:*"

    async for _ in redis_client.scan_iter(match=pattern):
        nonce_count += 1

    return {
        "active_nonces": nonce_count,
        "nonce_ttl_seconds": NONCE_TTL_SECONDS,
        "timestamp_tolerance_seconds": TIMESTAMP_TOLERANCE_SECONDS,
    }


# Call validation on import
validate_signing_config()


__all__ = [
    "SignatureError",
    "generate_nonce",
    "generate_timestamp",
    "compute_signature",
    "verify_signature",
    "verify_timestamp",
    "verify_nonce",
    "verify_signed_request",
    "require_signature",
    "RequestSigner",
    "log_signed_request",
    "get_signature_stats",
    "get_redis_client",
]
