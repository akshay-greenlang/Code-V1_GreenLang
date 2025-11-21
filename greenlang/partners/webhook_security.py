# -*- coding: utf-8 -*-
"""
Webhook Security Module for GreenLang

This module provides security features for webhooks including:
- HMAC signature generation and verification
- Timestamp-based replay attack prevention
- Rate limiting for webhook deliveries
- IP whitelisting
- Webhook payload validation
"""

import hashlib
import hmac
import time
import ipaddress
from datetime import datetime, timedelta
from typing import Optional, List, Set
import logging

from fastapi import HTTPException, Request, Header
from pydantic import BaseModel
import redis
from greenlang.determinism import DeterministicClock

logger = logging.getLogger(__name__)

# Constants
SIGNATURE_ALGORITHM = "sha256"
REPLAY_WINDOW_SECONDS = 300  # 5 minutes
MAX_WEBHOOKS_PER_MINUTE = 100
MAX_PAYLOAD_SIZE_BYTES = 1024 * 1024  # 1 MB


class SignatureVerificationError(Exception):
    """Raised when signature verification fails"""
    pass


class ReplayAttackError(Exception):
    """Raised when timestamp is outside allowed window"""
    pass


class RateLimitExceededError(Exception):
    """Raised when rate limit is exceeded"""
    pass


class WebhookSignature:
    """
    Handles HMAC signature generation and verification for webhooks
    """

    @staticmethod
    def generate_signature(payload: bytes, secret: str, timestamp: Optional[int] = None) -> str:
        """
        Generate HMAC-SHA256 signature for webhook payload

        Args:
            payload: JSON payload as bytes
            secret: Webhook secret key
            timestamp: Unix timestamp (optional, uses current time if not provided)

        Returns:
            Signature in format "sha256=<hex_digest>"

        Example:
            >>> payload = b'{"event": "workflow.completed"}'
            >>> secret = "my-webhook-secret"
            >>> sig = WebhookSignature.generate_signature(payload, secret)
            >>> print(sig)
            sha256=a1b2c3d4...
        """
        if timestamp is None:
            timestamp = int(time.time())

        # Combine timestamp and payload for signature
        signed_payload = f"{timestamp}.{payload.decode('utf-8')}".encode('utf-8')

        signature = hmac.new(
            secret.encode('utf-8'),
            signed_payload,
            hashlib.sha256
        ).hexdigest()

        return f"{SIGNATURE_ALGORITHM}={signature}"

    @staticmethod
    def verify_signature(
        payload: bytes,
        signature: str,
        secret: str,
        timestamp: int,
        tolerance_seconds: int = REPLAY_WINDOW_SECONDS
    ) -> bool:
        """
        Verify HMAC signature for webhook payload

        Args:
            payload: JSON payload as bytes
            signature: Signature from X-GreenLang-Signature header
            secret: Webhook secret key
            timestamp: Unix timestamp from X-GreenLang-Timestamp header
            tolerance_seconds: Maximum age of request in seconds

        Returns:
            True if signature is valid and timestamp is within tolerance

        Raises:
            SignatureVerificationError: If signature is invalid
            ReplayAttackError: If timestamp is outside tolerance window

        Example:
            >>> payload = b'{"event": "workflow.completed"}'
            >>> secret = "my-webhook-secret"
            >>> signature = "sha256=a1b2c3d4..."
            >>> timestamp = 1699459200
            >>> WebhookSignature.verify_signature(payload, signature, secret, timestamp)
            True
        """
        # Check timestamp to prevent replay attacks
        current_time = int(time.time())
        time_diff = abs(current_time - timestamp)

        if time_diff > tolerance_seconds:
            raise ReplayAttackError(
                f"Timestamp {timestamp} is outside tolerance window "
                f"({tolerance_seconds}s). Time difference: {time_diff}s"
            )

        # Generate expected signature
        expected_signature = WebhookSignature.generate_signature(payload, secret, timestamp)

        # Compare signatures using constant-time comparison
        if not hmac.compare_digest(signature, expected_signature):
            raise SignatureVerificationError("Invalid signature")

        return True

    @staticmethod
    def extract_signature_parts(signature: str) -> tuple[str, str]:
        """
        Extract algorithm and digest from signature

        Args:
            signature: Signature in format "algorithm=digest"

        Returns:
            Tuple of (algorithm, digest)

        Raises:
            ValueError: If signature format is invalid
        """
        if '=' not in signature:
            raise ValueError("Invalid signature format. Expected 'algorithm=digest'")

        parts = signature.split('=', 1)
        if len(parts) != 2:
            raise ValueError("Invalid signature format. Expected 'algorithm=digest'")

        algorithm, digest = parts

        if algorithm != SIGNATURE_ALGORITHM:
            raise ValueError(f"Unsupported algorithm: {algorithm}. Expected {SIGNATURE_ALGORITHM}")

        return algorithm, digest


class WebhookRateLimiter:
    """
    Rate limiter for webhook deliveries to prevent abuse
    """

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    def check_rate_limit(
        self,
        partner_id: str,
        limit: int = MAX_WEBHOOKS_PER_MINUTE,
        window_seconds: int = 60
    ) -> bool:
        """
        Check if partner is within webhook rate limit

        Args:
            partner_id: Partner ID
            limit: Maximum webhooks per window
            window_seconds: Time window in seconds

        Returns:
            True if within limit

        Raises:
            RateLimitExceededError: If rate limit exceeded
        """
        key = f"webhook_rate_limit:{partner_id}:{int(time.time() / window_seconds)}"

        try:
            current = self.redis.get(key)

            if current is None:
                # First webhook in this window
                self.redis.setex(key, window_seconds, 1)
                return True

            current = int(current)

            if current >= limit:
                raise RateLimitExceededError(
                    f"Webhook rate limit exceeded: {limit} webhooks per {window_seconds}s"
                )

            self.redis.incr(key)
            return True

        except redis.RedisError as e:
            logger.error(f"Redis error in rate limiter: {e}")
            # Fail open - allow webhook if Redis is down
            return True

    def get_remaining_quota(
        self,
        partner_id: str,
        limit: int = MAX_WEBHOOKS_PER_MINUTE,
        window_seconds: int = 60
    ) -> int:
        """
        Get remaining webhook quota for partner

        Args:
            partner_id: Partner ID
            limit: Maximum webhooks per window
            window_seconds: Time window in seconds

        Returns:
            Number of remaining webhooks in current window
        """
        key = f"webhook_rate_limit:{partner_id}:{int(time.time() / window_seconds)}"

        try:
            current = self.redis.get(key)
            if current is None:
                return limit

            return max(0, limit - int(current))

        except redis.RedisError as e:
            logger.error(f"Redis error getting quota: {e}")
            return limit


class IPWhitelist:
    """
    IP whitelisting for webhook deliveries
    """

    def __init__(self):
        self.whitelists: dict[str, Set[ipaddress.IPv4Network | ipaddress.IPv6Network]] = {}

    def add_ip_range(self, partner_id: str, ip_range: str):
        """
        Add IP range to whitelist for partner

        Args:
            partner_id: Partner ID
            ip_range: IP address or CIDR range (e.g., "192.168.1.0/24")
        """
        if partner_id not in self.whitelists:
            self.whitelists[partner_id] = set()

        try:
            network = ipaddress.ip_network(ip_range, strict=False)
            self.whitelists[partner_id].add(network)
            logger.info(f"Added IP range {ip_range} to whitelist for partner {partner_id}")
        except ValueError as e:
            raise ValueError(f"Invalid IP range: {ip_range}") from e

    def remove_ip_range(self, partner_id: str, ip_range: str):
        """
        Remove IP range from whitelist for partner

        Args:
            partner_id: Partner ID
            ip_range: IP address or CIDR range
        """
        if partner_id not in self.whitelists:
            return

        try:
            network = ipaddress.ip_network(ip_range, strict=False)
            self.whitelists[partner_id].discard(network)
            logger.info(f"Removed IP range {ip_range} from whitelist for partner {partner_id}")
        except ValueError as e:
            logger.warning(f"Invalid IP range for removal: {ip_range}")

    def is_allowed(self, partner_id: str, ip_address: str) -> bool:
        """
        Check if IP address is allowed for partner

        Args:
            partner_id: Partner ID
            ip_address: IP address to check

        Returns:
            True if IP is allowed (or no whitelist configured)
        """
        # If no whitelist configured, allow all IPs
        if partner_id not in self.whitelists or not self.whitelists[partner_id]:
            return True

        try:
            ip = ipaddress.ip_address(ip_address)
            for network in self.whitelists[partner_id]:
                if ip in network:
                    return True
            return False
        except ValueError:
            logger.warning(f"Invalid IP address: {ip_address}")
            return False

    def get_whitelist(self, partner_id: str) -> List[str]:
        """
        Get whitelist for partner

        Args:
            partner_id: Partner ID

        Returns:
            List of IP ranges in CIDR notation
        """
        if partner_id not in self.whitelists:
            return []

        return [str(network) for network in self.whitelists[partner_id]]


class WebhookValidator:
    """
    Validates webhook payloads and headers
    """

    @staticmethod
    def validate_payload_size(payload: bytes, max_size: int = MAX_PAYLOAD_SIZE_BYTES) -> bool:
        """
        Validate payload size

        Args:
            payload: Payload bytes
            max_size: Maximum allowed size in bytes

        Returns:
            True if valid

        Raises:
            ValueError: If payload exceeds max size
        """
        size = len(payload)
        if size > max_size:
            raise ValueError(
                f"Payload size ({size} bytes) exceeds maximum ({max_size} bytes)"
            )
        return True

    @staticmethod
    def validate_headers(request: Request) -> dict:
        """
        Validate required webhook headers

        Args:
            request: FastAPI request object

        Returns:
            Dictionary with validated header values

        Raises:
            HTTPException: If required headers are missing or invalid
        """
        # Check for required headers
        signature = request.headers.get("X-GreenLang-Signature")
        if not signature:
            raise HTTPException(status_code=400, detail="Missing X-GreenLang-Signature header")

        timestamp_str = request.headers.get("X-GreenLang-Timestamp")
        if not timestamp_str:
            raise HTTPException(status_code=400, detail="Missing X-GreenLang-Timestamp header")

        event_type = request.headers.get("X-GreenLang-Event")
        if not event_type:
            raise HTTPException(status_code=400, detail="Missing X-GreenLang-Event header")

        event_id = request.headers.get("X-GreenLang-Event-ID")
        if not event_id:
            raise HTTPException(status_code=400, detail="Missing X-GreenLang-Event-ID header")

        # Validate timestamp format
        try:
            timestamp = int(timestamp_str)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid timestamp format")

        return {
            "signature": signature,
            "timestamp": timestamp,
            "event_type": event_type,
            "event_id": event_id
        }


# Middleware for webhook verification
async def verify_webhook_request(
    request: Request,
    secret: str,
    x_greenlang_signature: str = Header(...),
    x_greenlang_timestamp: int = Header(...)
):
    """
    FastAPI dependency for verifying webhook requests

    Args:
        request: FastAPI request object
        secret: Webhook secret for signature verification
        x_greenlang_signature: Signature from header
        x_greenlang_timestamp: Timestamp from header

    Raises:
        HTTPException: If verification fails
    """
    # Read and validate payload
    payload = await request.body()

    try:
        WebhookValidator.validate_payload_size(payload)
    except ValueError as e:
        raise HTTPException(status_code=413, detail=str(e))

    # Verify signature
    try:
        WebhookSignature.verify_signature(
            payload,
            x_greenlang_signature,
            secret,
            x_greenlang_timestamp
        )
    except SignatureVerificationError:
        raise HTTPException(status_code=401, detail="Invalid signature")
    except ReplayAttackError as e:
        raise HTTPException(status_code=401, detail=str(e))


# Example usage for partners receiving webhooks
class WebhookReceiver:
    """
    Example webhook receiver implementation for partners

    Partners should use this class to verify incoming webhooks from GreenLang.
    """

    def __init__(self, webhook_secret: str):
        """
        Initialize webhook receiver

        Args:
            webhook_secret: Secret provided when creating webhook
        """
        self.secret = webhook_secret

    def verify_request(self, payload: bytes, signature: str, timestamp: int) -> bool:
        """
        Verify incoming webhook request

        Args:
            payload: Request body as bytes
            signature: Value from X-GreenLang-Signature header
            timestamp: Value from X-GreenLang-Timestamp header

        Returns:
            True if verification successful

        Example:
            >>> receiver = WebhookReceiver("my-secret")
            >>> payload = request.body
            >>> signature = request.headers["X-GreenLang-Signature"]
            >>> timestamp = int(request.headers["X-GreenLang-Timestamp"])
            >>> if receiver.verify_request(payload, signature, timestamp):
            >>>     # Process webhook
            >>>     pass
        """
        try:
            return WebhookSignature.verify_signature(
                payload,
                signature,
                self.secret,
                timestamp
            )
        except (SignatureVerificationError, ReplayAttackError) as e:
            logger.error(f"Webhook verification failed: {e}")
            return False


# Utility functions
def generate_webhook_secret() -> str:
    """
    Generate a secure webhook secret

    Returns:
        Hex-encoded random secret (64 characters)
    """
    import secrets
    return secrets.token_hex(32)


def compute_signature_for_test(payload: dict, secret: str) -> tuple[str, int]:
    """
    Compute signature for testing webhooks

    Args:
        payload: Webhook payload dictionary
        secret: Webhook secret

    Returns:
        Tuple of (signature, timestamp)

    Example:
        >>> payload = {"event": "workflow.completed", "data": {...}}
        >>> secret = "my-webhook-secret"
        >>> signature, timestamp = compute_signature_for_test(payload, secret)
        >>> # Use these values in test request headers
    """
    import json

    timestamp = int(time.time())
    payload_bytes = json.dumps(payload).encode('utf-8')
    signature = WebhookSignature.generate_signature(payload_bytes, secret, timestamp)

    return signature, timestamp


if __name__ == "__main__":
    # Example usage
    import json

    # Sender (GreenLang) side
    webhook_secret = generate_webhook_secret()
    print(f"Generated webhook secret: {webhook_secret}")

    payload = {
        "event": "workflow.completed",
        "event_id": "evt_123",
        "timestamp": DeterministicClock.utcnow().isoformat(),
        "partner_id": "partner_456",
        "data": {
            "workflow_id": "wf_789",
            "status": "success"
        }
    }

    payload_bytes = json.dumps(payload).encode('utf-8')
    timestamp = int(time.time())
    signature = WebhookSignature.generate_signature(payload_bytes, webhook_secret, timestamp)

    print(f"Signature: {signature}")
    print(f"Timestamp: {timestamp}")

    # Receiver (Partner) side
    receiver = WebhookReceiver(webhook_secret)
    is_valid = receiver.verify_request(payload_bytes, signature, timestamp)
    print(f"Signature valid: {is_valid}")
