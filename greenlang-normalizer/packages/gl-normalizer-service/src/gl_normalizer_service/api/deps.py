"""
Dependency injection utilities for GL Normalizer Service.

This module provides FastAPI dependencies for authentication, authorization,
rate limiting, and service access. Dependencies are designed for reusability
and testability.

Dependencies:
    get_settings: Application settings
    get_current_user: Authenticated user from JWT/API key
    get_normalizer: Normalizer service instance
    get_job_store: Async job storage
    get_vocabulary_service: Vocabulary management service

Usage:
    >>> from fastapi import Depends
    >>> from gl_normalizer_service.api.deps import get_current_user
    >>>
    >>> @app.get("/protected")
    >>> async def protected_endpoint(user: User = Depends(get_current_user)):
    ...     return {"user": user.email}
"""

from datetime import datetime
from functools import lru_cache
from typing import Annotated, Optional
from uuid import uuid4

from fastapi import Depends, Header, HTTPException, Request, status
from jose import JWTError, jwt
from pydantic import BaseModel

from gl_normalizer_service.config import Settings, get_settings


# ==============================================================================
# User Models
# ==============================================================================


class User(BaseModel):
    """
    Authenticated user model.

    Attributes:
        id: Unique user identifier
        email: User email address
        tenant_id: Tenant/organization identifier
        roles: User roles for RBAC
        api_key_id: API key identifier (if authenticated via API key)
    """

    id: str
    email: str
    tenant_id: str
    roles: list[str] = []
    api_key_id: Optional[str] = None


class ServiceAccount(BaseModel):
    """
    Service account for machine-to-machine authentication.

    Attributes:
        id: Service account identifier
        name: Service account name
        tenant_id: Tenant/organization identifier
        scopes: Authorized scopes
    """

    id: str
    name: str
    tenant_id: str
    scopes: list[str] = []


# ==============================================================================
# Authentication Dependencies
# ==============================================================================


async def get_api_key(
    request: Request,
    x_api_key: Annotated[Optional[str], Header(alias="X-API-Key")] = None,
    settings: Settings = Depends(get_settings),
) -> Optional[str]:
    """
    Extract API key from request header.

    Args:
        request: FastAPI request
        x_api_key: API key from header
        settings: Application settings

    Returns:
        API key string if present, None otherwise
    """
    # Check custom header name from settings
    if x_api_key:
        return x_api_key

    # Check authorization header for API key
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("ApiKey "):
        return auth_header[7:]

    return None


async def get_bearer_token(
    request: Request,
) -> Optional[str]:
    """
    Extract JWT bearer token from Authorization header.

    Args:
        request: FastAPI request

    Returns:
        JWT token string if present, None otherwise
    """
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:]
    return None


async def validate_api_key(api_key: str, settings: Settings) -> Optional[User]:
    """
    Validate API key and return associated user.

    In production, this would query a database or cache. For development,
    we accept a test API key.

    Args:
        api_key: API key to validate
        settings: Application settings

    Returns:
        User associated with API key, or None if invalid

    Note:
        Production implementation should:
        - Query API key store (Redis/database)
        - Check key expiration
        - Validate key scopes
        - Rate limit by key
    """
    # Development mode: accept test keys
    if settings.env == "development" and api_key.startswith("dev_"):
        return User(
            id="dev_user_001",
            email="developer@greenlang.io",
            tenant_id="dev_tenant",
            roles=["developer", "normalizer:read", "normalizer:write"],
            api_key_id=api_key[:20],
        )

    # Production: validate against key store
    # TODO: Implement production API key validation
    # Example:
    # key_data = await redis.get(f"api_key:{hash(api_key)}")
    # if key_data and not key_data.expired:
    #     return User(**key_data.user)

    return None


async def validate_jwt_token(token: str, settings: Settings) -> Optional[User]:
    """
    Validate JWT token and return user claims.

    Args:
        token: JWT token string
        settings: Application settings

    Returns:
        User from token claims, or None if invalid

    Raises:
        HTTPException: If token is expired or malformed
    """
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.algorithm],
        )

        user_id: str = payload.get("sub")
        email: str = payload.get("email", "")
        tenant_id: str = payload.get("tenant_id", "")
        roles: list[str] = payload.get("roles", [])

        if not user_id:
            return None

        return User(
            id=user_id,
            email=email,
            tenant_id=tenant_id,
            roles=roles,
        )

    except JWTError:
        return None


async def get_current_user(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    api_key: Annotated[Optional[str], Depends(get_api_key)] = None,
    bearer_token: Annotated[Optional[str], Depends(get_bearer_token)] = None,
) -> User:
    """
    Get current authenticated user from request.

    Supports both API key and JWT authentication. API key takes precedence
    if both are provided.

    Args:
        request: FastAPI request
        settings: Application settings
        api_key: API key from header
        bearer_token: JWT bearer token

    Returns:
        Authenticated user

    Raises:
        HTTPException: 401 if authentication fails

    Example:
        >>> @app.get("/me")
        >>> async def get_me(user: User = Depends(get_current_user)):
        ...     return {"email": user.email}
    """
    user: Optional[User] = None

    # Try API key authentication first
    if api_key:
        user = await validate_api_key(api_key, settings)
        if user:
            # Store authentication method in request state
            request.state.auth_method = "api_key"
            return user

    # Try JWT authentication
    if bearer_token:
        user = await validate_jwt_token(bearer_token, settings)
        if user:
            request.state.auth_method = "jwt"
            return user

    # Authentication failed
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={
            "code": "GLNORM-007",
            "message": "Authentication required. Provide API key or Bearer token.",
        },
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_user_optional(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    api_key: Annotated[Optional[str], Depends(get_api_key)] = None,
    bearer_token: Annotated[Optional[str], Depends(get_bearer_token)] = None,
) -> Optional[User]:
    """
    Get current user if authenticated, otherwise None.

    Use this for endpoints that support both authenticated and anonymous access.

    Args:
        request: FastAPI request
        settings: Application settings
        api_key: API key from header
        bearer_token: JWT bearer token

    Returns:
        Authenticated user or None
    """
    try:
        return await get_current_user(request, settings, api_key, bearer_token)
    except HTTPException:
        return None


def require_role(required_role: str):
    """
    Dependency factory for role-based access control.

    Args:
        required_role: Role required to access the endpoint

    Returns:
        Dependency function that validates user role

    Example:
        >>> @app.delete("/admin/resource")
        >>> async def admin_delete(
        ...     user: User = Depends(require_role("admin"))
        ... ):
        ...     return {"deleted": True}
    """

    async def role_checker(
        user: Annotated[User, Depends(get_current_user)],
    ) -> User:
        if required_role not in user.roles and "admin" not in user.roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "code": "GLNORM-007",
                    "message": f"Required role: {required_role}",
                },
            )
        return user

    return role_checker


def require_scope(required_scope: str):
    """
    Dependency factory for scope-based access control.

    Args:
        required_scope: Scope required to access the endpoint

    Returns:
        Dependency function that validates user scope

    Example:
        >>> @app.post("/normalize")
        >>> async def normalize(
        ...     user: User = Depends(require_scope("normalizer:write"))
        ... ):
        ...     return {"normalized": True}
    """

    async def scope_checker(
        user: Annotated[User, Depends(get_current_user)],
    ) -> User:
        # Check if user has required scope in roles
        if required_scope not in user.roles and "admin" not in user.roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "code": "GLNORM-007",
                    "message": f"Required scope: {required_scope}",
                },
            )
        return user

    return scope_checker


# ==============================================================================
# Service Dependencies
# ==============================================================================


class NormalizerService:
    """
    Normalizer service wrapper.

    Wraps the gl_normalizer_core package for use in API endpoints.
    Provides async methods for normalization operations.
    """

    def __init__(self, settings: Settings):
        """
        Initialize normalizer service.

        Args:
            settings: Application settings
        """
        self.settings = settings
        # In production, initialize gl_normalizer_core here
        # from gl_normalizer_core import Normalizer
        # self._normalizer = Normalizer(config=settings.normalizer_config)

    async def normalize(
        self,
        value: str | float | int,
        unit: str,
        target_unit: Optional[str] = None,
        entity: Optional[str] = None,
        context: Optional[dict] = None,
    ) -> dict:
        """
        Normalize a single value.

        Args:
            value: Value to normalize
            unit: Source unit
            target_unit: Target unit (optional)
            entity: Entity context (optional)
            context: Additional context (optional)

        Returns:
            Normalization result dictionary
        """
        # Generate audit ID
        audit_id = f"aud_{uuid4().hex[:12]}"

        # TODO: Call gl_normalizer_core
        # result = await self._normalizer.normalize(
        #     value=value,
        #     unit=unit,
        #     target_unit=target_unit,
        #     entity=entity,
        #     context=context,
        # )

        # Mock implementation for development
        try:
            numeric_value = float(value) if isinstance(value, str) else value
        except ValueError:
            numeric_value = 0.0

        # Simulate unit conversion
        conversion_factor = 0.001 if "kg" in unit.lower() else 1.0
        canonical_value = numeric_value * conversion_factor

        return {
            "canonical_value": canonical_value,
            "canonical_unit": target_unit or "metric_ton_co2e",
            "confidence": 0.95,
            "needs_review": False,
            "review_reasons": [],
            "audit_id": audit_id,
            "source_value": value,
            "source_unit": unit,
            "conversion_factor": conversion_factor,
            "metadata": {
                "vocabulary_version": "2026.1",
                "normalization_rule": "unit_conversion",
            },
        }

    async def normalize_batch(
        self,
        items: list[dict],
        batch_mode: str = "PARTIAL",
        threshold: float = 0.1,
    ) -> tuple[list[dict], dict]:
        """
        Normalize a batch of values.

        Args:
            items: List of items to normalize
            batch_mode: How to handle failures
            threshold: Failure threshold for THRESHOLD mode

        Returns:
            Tuple of (results list, summary dict)
        """
        import time

        start_time = time.time()
        results = []
        success_count = 0
        failed_count = 0
        review_count = 0

        for item in items:
            try:
                result = await self.normalize(
                    value=item["value"],
                    unit=item["unit"],
                    target_unit=item.get("target_unit"),
                    entity=item.get("entity"),
                    context=item.get("context"),
                )
                results.append({
                    "id": item["id"],
                    "success": True,
                    "result": result,
                })
                success_count += 1
                if result.get("needs_review"):
                    review_count += 1

                # Check threshold mode
                if batch_mode == "THRESHOLD":
                    failure_rate = failed_count / len(results)
                    if failure_rate > threshold:
                        break

            except Exception as e:
                failed_count += 1
                results.append({
                    "id": item["id"],
                    "success": False,
                    "error": {
                        "code": "GLNORM-009",
                        "message": str(e),
                    },
                })

                # Check fail fast mode
                if batch_mode == "FAIL_FAST":
                    break

        processing_time_ms = int((time.time() - start_time) * 1000)

        summary = {
            "total": len(items),
            "success": success_count,
            "failed": failed_count,
            "needs_review": review_count,
            "processing_time_ms": processing_time_ms,
        }

        return results, summary


class JobStore:
    """
    Async job storage and management.

    Manages job lifecycle for async normalization requests.
    In production, backed by Redis or a database.
    """

    def __init__(self, settings: Settings):
        """
        Initialize job store.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self._jobs: dict[str, dict] = {}  # In-memory store for development

    async def create_job(
        self,
        items: list[dict],
        batch_mode: str,
        callback_url: Optional[str] = None,
        priority: int = 5,
    ) -> dict:
        """
        Create a new async job.

        Args:
            items: Items to process
            batch_mode: Batch processing mode
            callback_url: Webhook URL for completion
            priority: Job priority

        Returns:
            Job metadata
        """
        job_id = f"job_{uuid4()}"
        now = datetime.utcnow()

        job = {
            "job_id": job_id,
            "status": "PENDING",
            "progress": {
                "processed": 0,
                "total": len(items),
                "percent_complete": 0.0,
                "current_rate": None,
            },
            "items": items,
            "batch_mode": batch_mode,
            "callback_url": callback_url,
            "priority": priority,
            "created_at": now.isoformat() + "Z",
            "started_at": None,
            "completed_at": None,
            "summary": None,
            "result_url": None,
            "error": None,
        }

        self._jobs[job_id] = job
        return job

    async def get_job(self, job_id: str) -> Optional[dict]:
        """
        Get job by ID.

        Args:
            job_id: Job identifier

        Returns:
            Job metadata or None if not found
        """
        return self._jobs.get(job_id)

    async def update_job(self, job_id: str, updates: dict) -> Optional[dict]:
        """
        Update job metadata.

        Args:
            job_id: Job identifier
            updates: Fields to update

        Returns:
            Updated job or None if not found
        """
        if job_id not in self._jobs:
            return None

        self._jobs[job_id].update(updates)
        return self._jobs[job_id]


class VocabularyService:
    """
    Vocabulary management service.

    Provides access to normalization vocabularies and lookup tables.
    """

    def __init__(self, settings: Settings):
        """
        Initialize vocabulary service.

        Args:
            settings: Application settings
        """
        self.settings = settings

    async def list_vocabularies(self) -> list[dict]:
        """
        List available vocabularies.

        Returns:
            List of vocabulary metadata
        """
        # TODO: Load from gl_normalizer_core
        return [
            {
                "id": "ghg_units",
                "name": "GHG Emission Units",
                "description": "Standard units for greenhouse gas emissions (CO2, CH4, N2O, etc.)",
                "version": "2026.1",
                "entry_count": 245,
                "last_updated": "2026-01-15T00:00:00Z",
                "categories": ["emissions", "carbon", "methane"],
            },
            {
                "id": "energy_units",
                "name": "Energy Units",
                "description": "Energy measurement units (kWh, MWh, BTU, joules, etc.)",
                "version": "2026.1",
                "entry_count": 128,
                "last_updated": "2026-01-15T00:00:00Z",
                "categories": ["energy", "electricity", "fuel"],
            },
            {
                "id": "water_units",
                "name": "Water Units",
                "description": "Water volume and flow units (gallons, liters, cubic meters, etc.)",
                "version": "2026.1",
                "entry_count": 86,
                "last_updated": "2026-01-15T00:00:00Z",
                "categories": ["water", "volume", "flow"],
            },
            {
                "id": "waste_units",
                "name": "Waste Units",
                "description": "Waste measurement units (tons, kg, cubic yards, etc.)",
                "version": "2026.1",
                "entry_count": 64,
                "last_updated": "2026-01-15T00:00:00Z",
                "categories": ["waste", "recycling", "disposal"],
            },
            {
                "id": "entity_names",
                "name": "Entity Normalization",
                "description": "Company and organization name normalization vocabulary",
                "version": "2026.1",
                "entry_count": 15420,
                "last_updated": "2026-01-20T00:00:00Z",
                "categories": ["companies", "organizations", "facilities"],
            },
        ]

    async def get_vocabulary(self, vocabulary_id: str) -> Optional[dict]:
        """
        Get vocabulary by ID.

        Args:
            vocabulary_id: Vocabulary identifier

        Returns:
            Vocabulary metadata or None if not found
        """
        vocabularies = await self.list_vocabularies()
        for vocab in vocabularies:
            if vocab["id"] == vocabulary_id:
                return vocab
        return None


# ==============================================================================
# Service Instance Dependencies
# ==============================================================================


@lru_cache
def get_normalizer_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> NormalizerService:
    """
    Get cached normalizer service instance.

    Args:
        settings: Application settings

    Returns:
        NormalizerService instance
    """
    return NormalizerService(settings)


@lru_cache
def get_job_store(
    settings: Annotated[Settings, Depends(get_settings)],
) -> JobStore:
    """
    Get cached job store instance.

    Args:
        settings: Application settings

    Returns:
        JobStore instance
    """
    return JobStore(settings)


@lru_cache
def get_vocabulary_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> VocabularyService:
    """
    Get cached vocabulary service instance.

    Args:
        settings: Application settings

    Returns:
        VocabularyService instance
    """
    return VocabularyService(settings)


# ==============================================================================
# Request Context Dependencies
# ==============================================================================


async def get_request_id(request: Request) -> str:
    """
    Get or generate request ID for tracing.

    Args:
        request: FastAPI request

    Returns:
        Request ID string
    """
    # Check for existing request ID from headers
    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        request_id = f"req_{uuid4().hex[:12]}"

    # Store in request state
    request.state.request_id = request_id
    return request_id


async def get_tenant_id(
    user: Annotated[User, Depends(get_current_user)],
) -> str:
    """
    Get tenant ID from authenticated user.

    Args:
        user: Authenticated user

    Returns:
        Tenant ID string
    """
    return user.tenant_id
