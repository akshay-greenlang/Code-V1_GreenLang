"""
Model API - FastAPI endpoints for LLM model invocation and evaluation.

This module implements the RESTful API for:
- Model invocation with provenance tracking
- Response evaluation against golden tests
- Performance metrics collection
- JWT authentication and rate limiting

Example:
    >>> from fastapi import FastAPI
    >>> from greenlang.ml_platform.model_api import app
    >>> # uvicorn greenlang.ml_platform.model_api:app --reload
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
import hashlib
import logging
import time
import uuid
from enum import Enum

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import jwt

from greenlang.registry.model_registry import (
    model_registry,
    ModelProvider,
    ModelCapability,
    ModelMetadata
)

logger = logging.getLogger(__name__)

# JWT Configuration
JWT_SECRET = "greenlang-ml-platform-secret-change-in-production"
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Rate limiting (in-memory, will be Redis in production)
request_counts: Dict[str, List[float]] = {}
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW_SECONDS = 60


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ModelInvokeRequest(BaseModel):
    """Request model for model invocation."""

    model_id: str = Field(..., description="Model ID from registry")
    prompt: str = Field(..., description="Input prompt", min_length=1, max_length=100000)
    system_prompt: Optional[str] = Field(None, description="System prompt")
    max_tokens: Optional[int] = Field(None, ge=1, le=8192, description="Max tokens to generate")
    temperature: Optional[float] = Field(0.0, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Nucleus sampling")
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="Available tools")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Request metadata")

    @validator('temperature')
    def validate_temperature(cls, v):
        """Validate temperature for zero-hallucination mode."""
        if v != 0.0:
            logger.warning(
                f"Non-zero temperature ({v}) may introduce non-determinism. "
                "For zero-hallucination guarantees, use temperature=0.0"
            )
        return v


class ModelInvokeResponse(BaseModel):
    """Response model for model invocation."""

    request_id: str = Field(..., description="Unique request ID")
    model_id: str = Field(..., description="Model ID used")
    response_text: str = Field(..., description="Model response")
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")

    # Performance metrics
    latency_ms: float = Field(..., description="End-to-end latency")
    input_tokens: int = Field(..., description="Input token count")
    output_tokens: int = Field(..., description="Output token count")
    total_tokens: int = Field(..., description="Total token count")

    # Cost tracking
    cost_usd: float = Field(..., description="Request cost in USD")

    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelEvaluateRequest(BaseModel):
    """Request model for response evaluation."""

    model_id: str = Field(..., description="Model ID")
    prompt: str = Field(..., description="Input prompt")
    response: str = Field(..., description="Model response")
    expected_response: Optional[str] = Field(None, description="Expected response for golden test")
    evaluation_criteria: List[str] = Field(
        default_factory=list,
        description="Evaluation criteria (e.g., 'accuracy', 'completeness')"
    )


class EvaluationResult(BaseModel):
    """Result of response evaluation."""

    passed: bool = Field(..., description="Whether evaluation passed")
    score: float = Field(..., ge=0.0, le=1.0, description="Evaluation score (0-1)")
    criteria_results: Dict[str, bool] = Field(..., description="Per-criterion results")
    feedback: str = Field(..., description="Evaluation feedback")
    determinism_check: Optional[bool] = Field(None, description="Bit-perfect reproducibility")


class ModelMetricsResponse(BaseModel):
    """Response model for model metrics."""

    model_id: str
    total_requests: int
    total_tokens: int
    total_cost_usd: float
    avg_latency_ms: Optional[float]
    success_rate: float = Field(..., ge=0.0, le=1.0)
    last_updated: datetime


class AuthToken(BaseModel):
    """JWT authentication token."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int = JWT_EXPIRATION_HOURS * 3600


# ============================================================================
# AUTHENTICATION & RATE LIMITING
# ============================================================================

security = HTTPBearer()


def create_access_token(data: dict) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Verify JWT token."""
    try:
        payload = jwt.decode(
            credentials.credentials,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM]
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication token")


async def check_rate_limit(request: Request, token_data: dict = Depends(verify_token)):
    """Check rate limit for user."""
    user_id = token_data.get("sub", "unknown")
    current_time = time.time()

    # Initialize user's request history
    if user_id not in request_counts:
        request_counts[user_id] = []

    # Remove old requests outside the window
    request_counts[user_id] = [
        req_time for req_time in request_counts[user_id]
        if current_time - req_time < RATE_LIMIT_WINDOW_SECONDS
    ]

    # Check rate limit
    if len(request_counts[user_id]) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW_SECONDS}s"
        )

    # Add current request
    request_counts[user_id].append(current_time)

    return token_data


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="GreenLang Model API",
    description="Production-grade LLM model invocation and evaluation API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware - SECURITY: Configure specific origins in production
# Set CORS_ALLOWED_ORIGINS environment variable (comma-separated list)
import os
_cors_origins = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000")
_allowed_origins = [origin.strip() for origin in _cors_origins.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-Request-ID"],
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_provenance_hash(request: ModelInvokeRequest, response: str) -> str:
    """Calculate SHA-256 hash for provenance tracking."""
    provenance_str = f"{request.model_id}|{request.prompt}|{response}|{datetime.utcnow().isoformat()}"
    return hashlib.sha256(provenance_str.encode()).hexdigest()


def calculate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate request cost based on token usage."""
    model = model_registry.get_model(model_id)
    if not model or not model.avg_cost_per_1k_tokens:
        return 0.0

    total_tokens = input_tokens + output_tokens
    return (total_tokens / 1000.0) * model.avg_cost_per_1k_tokens


def mock_model_invocation(request: ModelInvokeRequest) -> tuple[str, int, int]:
    """
    Mock model invocation (replace with actual LLM SDK calls).

    Returns:
        Tuple of (response_text, input_tokens, output_tokens)
    """
    # TODO: Replace with actual Anthropic/OpenAI SDK calls
    model = model_registry.get_model(request.model_id)
    if not model:
        raise ValueError(f"Model not found: {request.model_id}")

    # Mock response
    response_text = f"[Mock response from {model.name}] Processed: {request.prompt[:50]}..."

    # Mock token counts (rough estimate: 1 token ≈ 4 chars)
    input_tokens = len(request.prompt) // 4
    output_tokens = len(response_text) // 4

    return response_text, input_tokens, output_tokens


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "service": "GreenLang Model API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "invoke": "POST /v1/models/invoke",
            "evaluate": "POST /v1/models/evaluate",
            "metrics": "GET /v1/models/{model_id}/metrics"
        }
    }


@app.post("/v1/auth/token", response_model=AuthToken)
async def get_token(api_key: str = Header(...)):
    """
    Get JWT access token.

    In production, validate API key against database.
    """
    # TODO: Validate API key against database
    if not api_key.startswith("gl_"):
        raise HTTPException(status_code=401, detail="Invalid API key")

    token = create_access_token(data={"sub": api_key})
    return AuthToken(access_token=token)


@app.post("/v1/models/invoke", response_model=ModelInvokeResponse)
async def invoke_model(
    request: ModelInvokeRequest,
    user: dict = Depends(check_rate_limit)
):
    """
    Invoke an LLM model with provenance tracking.

    This endpoint:
    1. Validates model exists and is available
    2. Invokes the model with the provided prompt
    3. Tracks performance metrics (latency, tokens, cost)
    4. Calculates provenance hash for audit trail
    5. Updates model registry metrics

    Returns:
        ModelInvokeResponse with complete telemetry
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())

    try:
        # Validate model exists
        model = model_registry.get_model(request.model_id)
        if not model:
            raise HTTPException(
                status_code=404,
                detail=f"Model not found: {request.model_id}"
            )

        logger.info(f"Invoking model {request.model_id} (request_id={request_id})")

        # Invoke model (mock for now)
        response_text, input_tokens, output_tokens = mock_model_invocation(request)

        # Calculate metrics
        latency_ms = (time.time() - start_time) * 1000
        total_tokens = input_tokens + output_tokens
        cost_usd = calculate_cost(request.model_id, input_tokens, output_tokens)

        # Calculate provenance hash
        provenance_hash = calculate_provenance_hash(request, response_text)

        # Update model registry metrics
        model_registry.update_metrics(
            model_id=request.model_id,
            requests=1,
            tokens=total_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms
        )

        logger.info(
            f"Model invocation complete: {latency_ms:.2f}ms, "
            f"{total_tokens} tokens, ${cost_usd:.6f}"
        )

        return ModelInvokeResponse(
            request_id=request_id,
            model_id=request.model_id,
            response_text=response_text,
            provenance_hash=provenance_hash,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            metadata=request.metadata
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model invocation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Model invocation failed: {str(e)}")


@app.post("/v1/models/evaluate", response_model=EvaluationResult)
async def evaluate_response(
    request: ModelEvaluateRequest,
    user: dict = Depends(check_rate_limit)
):
    """
    Evaluate a model response against criteria.

    This endpoint:
    1. Compares response to expected output (if provided)
    2. Evaluates against specified criteria
    3. Checks determinism (bit-perfect reproducibility)
    4. Returns detailed evaluation results

    Returns:
        EvaluationResult with pass/fail and detailed feedback
    """
    try:
        criteria_results = {}
        score = 0.0

        # Check against expected response (exact match for determinism)
        if request.expected_response:
            exact_match = request.response == request.expected_response
            criteria_results["exact_match"] = exact_match
            if exact_match:
                score += 0.5

        # Evaluate against criteria
        for criterion in request.evaluation_criteria:
            # Mock evaluation logic (replace with actual evaluation)
            result = len(request.response) > 10  # Simple check
            criteria_results[criterion] = result
            if result:
                score += 0.5 / len(request.evaluation_criteria)

        passed = all(criteria_results.values())

        # Determinism check (requires multiple invocations)
        determinism_check = request.expected_response is not None and criteria_results.get("exact_match", False)

        feedback = "Evaluation complete. "
        if passed:
            feedback += "All criteria passed."
        else:
            failed = [k for k, v in criteria_results.items() if not v]
            feedback += f"Failed criteria: {', '.join(failed)}"

        return EvaluationResult(
            passed=passed,
            score=min(score, 1.0),
            criteria_results=criteria_results,
            feedback=feedback,
            determinism_check=determinism_check
        )

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.get("/v1/models/{model_id}/metrics", response_model=ModelMetricsResponse)
async def get_model_metrics(
    model_id: str,
    user: dict = Depends(verify_token)
):
    """
    Get performance metrics for a model.

    Returns:
        Aggregated metrics including requests, tokens, cost, latency
    """
    try:
        model = model_registry.get_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

        return ModelMetricsResponse(
            model_id=model.id,
            total_requests=model.total_requests,
            total_tokens=model.total_tokens,
            total_cost_usd=model.total_cost_usd,
            avg_latency_ms=model.avg_latency_ms,
            success_rate=1.0,  # TODO: Track failures
            last_updated=model.updated_at
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get metrics: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@app.get("/v1/models", response_model=List[ModelMetadata])
async def list_models(
    provider: Optional[ModelProvider] = None,
    capability: Optional[ModelCapability] = None,
    certified_only: bool = False,
    user: dict = Depends(verify_token)
):
    """
    List available models with optional filtering.

    Query parameters:
        provider: Filter by provider (anthropic, openai, local)
        capability: Filter by capability (text_generation, code_generation, etc.)
        certified_only: Only return zero-hallucination certified models
    """
    try:
        models = model_registry.list_models(
            provider=provider,
            capability=capability,
            certified_only=certified_only
        )
        return models
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "models_registered": len(model_registry.models)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
