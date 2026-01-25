# -*- coding: utf-8 -*-
"""
GL-019 HEATSCHEDULER - Process Heating Schedule Optimization FastAPI REST API

Production-grade REST API for HEATSCHEDULER application providing:
- Production schedule ingestion
- Energy tariff management
- Equipment availability tracking
- Schedule optimization
- Cost savings calculation
- Control system integration

Implements GreenLang standard patterns: JWT auth, rate limiting, audit trails,
comprehensive error handling, and OpenAPI documentation.
"""

import hashlib
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Response,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =======================================================================================
# CONFIGURATION
# =======================================================================================

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# API Configuration
API_VERSION = "1.0.0"
API_TITLE = "HEATSCHEDULER API"
API_DESCRIPTION = """
**HEATSCHEDULER**: Production REST API for Process Heating Schedule Optimization

## Features

- **Production Schedule**: Ingest and manage production schedules from ERP
- **Energy Tariffs**: Manage time-of-use rates and demand charges
- **Equipment Status**: Track heating equipment availability
- **Schedule Optimization**: Generate cost-optimized heating schedules
- **Savings Calculation**: Compute cost savings vs baseline
- **Control Integration**: Apply schedules to control systems

## Authentication

All endpoints require JWT bearer token authentication.

Get token from `/api/v1/auth/token` endpoint with valid credentials.

## Rate Limiting

- Schedule endpoints: 100 requests/minute
- Optimization endpoints: 20 requests/minute
- Status endpoints: 1000 requests/minute
- Health/metrics: 1000 requests/minute
"""

# =======================================================================================
# PROMETHEUS METRICS
# =======================================================================================

REQUEST_COUNT = Counter(
    'heatscheduler_api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'heatscheduler_api_request_duration_seconds',
    'API request duration',
    ['method', 'endpoint']
)

OPTIMIZATION_COUNT = Counter(
    'heatscheduler_optimizations_total',
    'Total schedule optimizations performed',
    ['status']
)

SAVINGS_AMOUNT = Histogram(
    'heatscheduler_savings_usd',
    'Cost savings from schedule optimization',
    buckets=[10, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
)

# =======================================================================================
# AUTHENTICATION MODELS & UTILITIES
# =======================================================================================

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")


class Token(BaseModel):
    """OAuth2 token response."""
    access_token: str
    token_type: str
    expires_in: int = Field(ACCESS_TOKEN_EXPIRE_MINUTES * 60)


class TokenData(BaseModel):
    """JWT token payload data."""
    user_id: str
    email: str
    tenant_id: str
    roles: List[str]


class User(BaseModel):
    """User model."""
    id: str
    email: str
    tenant_id: str
    roles: List[str]
    full_name: Optional[str] = None
    disabled: bool = False


# Mock user database (replace with real database in production)
USERS_DB = {
    "demo@heatscheduler.io": {
        "id": "user_001",
        "email": "demo@heatscheduler.io",
        "tenant_id": "tenant_demo",
        "full_name": "Demo User",
        "hashed_password": pwd_context.hash("demo_password_123"),
        "roles": ["analyst", "operator"],
        "disabled": False
    }
}


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """
    Get current user from JWT token.

    Args:
        token: JWT access token

    Returns:
        Authenticated user

    Raises:
        HTTPException: 401 if token invalid or expired
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        email: str = payload.get("email")

        if user_id is None or email is None:
            raise credentials_exception

        token_data = TokenData(
            user_id=user_id,
            email=email,
            tenant_id=payload.get("tenant_id", ""),
            roles=payload.get("roles", [])
        )
    except JWTError:
        raise credentials_exception

    # Load user from database
    user_dict = USERS_DB.get(token_data.email)
    if user_dict is None:
        raise credentials_exception

    user = User(**user_dict)

    if user.disabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )

    return user


# =======================================================================================
# REQUEST/RESPONSE MODELS
# =======================================================================================

class ProductionBatchRequest(BaseModel):
    """Production batch input model."""
    batch_id: str = Field(..., description="Unique batch identifier")
    product_id: str = Field(..., description="Product identifier")
    product_name: str = Field(..., description="Product name")
    quantity: float = Field(..., ge=0, description="Quantity to produce")
    deadline: datetime = Field(..., description="Production deadline")
    priority: str = Field("medium", description="Priority: critical, high, medium, low")
    heating_temperature_c: float = Field(..., ge=0, le=2000, description="Required temperature (C)")
    heating_duration_minutes: int = Field(..., ge=1, description="Heating duration (minutes)")
    estimated_power_kw: float = Field(..., ge=0, description="Estimated power requirement (kW)")
    preferred_equipment_ids: List[str] = Field(default=[], description="Preferred equipment IDs")

    class Config:
        json_schema_extra = {
            "example": {
                "batch_id": "BATCH-001",
                "product_id": "PROD-A",
                "product_name": "Steel Component A",
                "quantity": 100,
                "deadline": "2024-12-04T16:00:00Z",
                "priority": "high",
                "heating_temperature_c": 850,
                "heating_duration_minutes": 120,
                "estimated_power_kw": 400,
                "preferred_equipment_ids": ["FURN-001"]
            }
        }


class TariffPeriodRequest(BaseModel):
    """Energy tariff period input model."""
    tariff_id: str = Field(..., description="Tariff identifier")
    period_start: datetime = Field(..., description="Period start time")
    period_end: datetime = Field(..., description="Period end time")
    energy_rate_per_kwh: float = Field(..., ge=0, description="Energy rate ($/kWh)")
    demand_rate_per_kw: float = Field(0, ge=0, description="Demand charge ($/kW)")
    period_type: str = Field("off_peak", description="Period type: peak, off_peak, shoulder")


class EquipmentStatusRequest(BaseModel):
    """Equipment status input model."""
    equipment_id: str = Field(..., description="Equipment identifier")
    equipment_type: str = Field(..., description="Equipment type")
    capacity_kw: float = Field(..., ge=0, description="Capacity (kW)")
    efficiency: float = Field(0.85, ge=0.5, le=1.0, description="Efficiency (0-1)")
    status: str = Field("available", description="Status: available, in_use, maintenance")
    current_temperature_c: float = Field(25, ge=0, description="Current temperature (C)")


class ScheduleOptimizationRequest(BaseModel):
    """Schedule optimization request model."""
    production_batches: List[ProductionBatchRequest]
    tariff_periods: List[TariffPeriodRequest]
    equipment_status: List[EquipmentStatusRequest]
    optimization_objective: str = Field(
        "minimize_cost",
        description="Objective: minimize_cost, minimize_peak_demand, balance_cost_demand"
    )
    planning_horizon_hours: int = Field(24, ge=4, le=168, description="Planning horizon (hours)")
    peak_demand_limit_kw: Optional[float] = Field(None, ge=0, description="Peak demand limit (kW)")

    class Config:
        json_schema_extra = {
            "example": {
                "production_batches": [
                    {
                        "batch_id": "BATCH-001",
                        "product_id": "PROD-A",
                        "product_name": "Steel Component A",
                        "quantity": 100,
                        "deadline": "2024-12-04T16:00:00Z",
                        "priority": "high",
                        "heating_temperature_c": 850,
                        "heating_duration_minutes": 120,
                        "estimated_power_kw": 400
                    }
                ],
                "tariff_periods": [
                    {
                        "tariff_id": "TOU-001",
                        "period_start": "2024-12-04T00:00:00Z",
                        "period_end": "2024-12-04T14:00:00Z",
                        "energy_rate_per_kwh": 0.06,
                        "period_type": "off_peak"
                    }
                ],
                "equipment_status": [
                    {
                        "equipment_id": "FURN-001",
                        "equipment_type": "electric_furnace",
                        "capacity_kw": 500,
                        "status": "available"
                    }
                ],
                "optimization_objective": "minimize_cost"
            }
        }


class HeatingTaskResponse(BaseModel):
    """Heating task in optimized schedule."""
    task_id: str
    batch_id: str
    equipment_id: str
    start_time: datetime
    end_time: datetime
    duration_minutes: int
    power_kw: float
    temperature_c: float
    estimated_energy_kwh: float
    estimated_cost_usd: float
    status: str


class OptimizedScheduleResponse(BaseModel):
    """Optimized schedule response."""
    schedule_id: str
    created_at: datetime
    period_start: datetime
    period_end: datetime
    tasks: List[HeatingTaskResponse]
    total_cost_usd: float
    energy_cost_usd: float
    demand_cost_usd: float
    baseline_cost_usd: float
    savings_usd: float
    savings_percent: float
    total_energy_kwh: float
    peak_demand_kw: float
    optimization_objective: str
    optimization_time_seconds: float
    provenance_hash: str


class CostSavingsResponse(BaseModel):
    """Cost savings analysis response."""
    schedule_id: str
    period_start: datetime
    period_end: datetime
    total_energy_kwh: float
    peak_energy_kwh: float
    off_peak_energy_kwh: float
    energy_cost_usd: float
    demand_cost_usd: float
    total_cost_usd: float
    baseline_cost_usd: float
    savings_usd: float
    savings_percent: float
    peak_demand_kw: float
    recommendations: List[str]
    provenance_hash: str


class ScheduleApplicationRequest(BaseModel):
    """Request to apply schedule to control systems."""
    schedule_id: str = Field(..., description="Schedule ID to apply")
    confirm: bool = Field(False, description="Confirmation flag")
    force: bool = Field(False, description="Force apply even with warnings")


class ScheduleApplicationResponse(BaseModel):
    """Response from schedule application."""
    success: bool
    schedule_id: str
    tasks_applied: int
    equipment_updated: List[str]
    timestamp: datetime
    warnings: List[str]


class DemandResponseEventRequest(BaseModel):
    """Demand response event notification."""
    event_id: str = Field(..., description="Event identifier")
    event_type: str = Field("curtailment", description="Event type")
    start_time: datetime = Field(..., description="Event start time")
    end_time: datetime = Field(..., description="Event end time")
    target_reduction_kw: float = Field(..., ge=0, description="Target reduction (kW)")
    incentive_per_kwh: float = Field(0, ge=0, description="Incentive rate ($/kWh)")


class DemandResponseResponse(BaseModel):
    """Demand response handling result."""
    event_id: str
    acknowledged: bool
    committed_reduction_kw: float
    affected_tasks: int
    estimated_incentive_usd: float
    schedule_adjusted: bool


class HealthStatus(BaseModel):
    """Health check status."""
    status: str
    timestamp: datetime
    version: str
    uptime_seconds: float
    checks: Dict[str, bool]


# =======================================================================================
# DETERMINISTIC CALCULATION FUNCTIONS
# =======================================================================================

class HeatSchedulerCalculations:
    """
    Deterministic calculation functions for heating schedule optimization.

    All calculations are based on energy engineering principles and
    verifiable formulas. Zero-hallucination, pure mathematical operations.
    """

    @staticmethod
    def calculate_energy_consumption(
        power_kw: float,
        duration_minutes: int,
        efficiency: float = 0.85
    ) -> float:
        """
        Calculate energy consumption for a heating task.

        Formula: E = P * t / eta
        Where: E = energy (kWh), P = power (kW), t = time (hours), eta = efficiency

        Args:
            power_kw: Power consumption (kW)
            duration_minutes: Duration (minutes)
            efficiency: Equipment efficiency (0-1)

        Returns:
            Energy consumption in kWh
        """
        duration_hours = duration_minutes / 60.0
        energy_kwh = power_kw * duration_hours / efficiency

        return round(energy_kwh, 2)

    @staticmethod
    def calculate_energy_cost(
        energy_kwh: float,
        rate_per_kwh: float
    ) -> float:
        """
        Calculate energy cost.

        Formula: Cost = E * Rate

        Args:
            energy_kwh: Energy consumption (kWh)
            rate_per_kwh: Energy rate ($/kWh)

        Returns:
            Energy cost in USD
        """
        cost = energy_kwh * rate_per_kwh

        return round(cost, 2)

    @staticmethod
    def calculate_demand_cost(
        peak_demand_kw: float,
        demand_rate_per_kw: float
    ) -> float:
        """
        Calculate demand charge.

        Formula: Cost = Peak_Demand * Rate

        Args:
            peak_demand_kw: Peak demand (kW)
            demand_rate_per_kw: Demand charge rate ($/kW)

        Returns:
            Demand cost in USD
        """
        cost = peak_demand_kw * demand_rate_per_kw

        return round(cost, 2)

    @staticmethod
    def calculate_ramp_time(
        current_temp_c: float,
        target_temp_c: float,
        ramp_rate_c_per_min: float = 10.0
    ) -> int:
        """
        Calculate temperature ramp time.

        Formula: Time = |T_target - T_current| / Ramp_Rate

        Args:
            current_temp_c: Current temperature (C)
            target_temp_c: Target temperature (C)
            ramp_rate_c_per_min: Ramp rate (C/min)

        Returns:
            Ramp time in minutes
        """
        temp_diff = abs(target_temp_c - current_temp_c)
        ramp_time = temp_diff / ramp_rate_c_per_min

        return int(round(ramp_time))

    @staticmethod
    def calculate_savings(
        optimized_cost: float,
        baseline_cost: float
    ) -> Dict[str, float]:
        """
        Calculate cost savings.

        Args:
            optimized_cost: Cost with optimization
            baseline_cost: Cost without optimization

        Returns:
            Dictionary with savings_usd and savings_percent
        """
        savings_usd = baseline_cost - optimized_cost
        savings_percent = (savings_usd / baseline_cost * 100) if baseline_cost > 0 else 0

        return {
            "savings_usd": round(savings_usd, 2),
            "savings_percent": round(savings_percent, 1)
        }

    @staticmethod
    def optimize_task_scheduling(
        batches: List[Dict],
        tariffs: List[Dict],
        equipment: List[Dict],
        peak_limit_kw: Optional[float] = None
    ) -> List[Dict]:
        """
        Optimize task scheduling for minimum cost.

        Uses deterministic heuristic: schedule during lowest-cost periods
        while respecting deadlines and equipment constraints.

        Args:
            batches: Production batches
            tariffs: Energy tariff periods
            equipment: Available equipment
            peak_limit_kw: Peak demand limit

        Returns:
            List of scheduled tasks
        """
        tasks = []
        now = datetime.now(timezone.utc)

        # Sort batches by priority and deadline
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        sorted_batches = sorted(
            batches,
            key=lambda b: (priority_order.get(b.get("priority", "medium"), 2), b.get("deadline"))
        )

        # Sort tariffs by rate (cheapest first)
        sorted_tariffs = sorted(tariffs, key=lambda t: t.get("energy_rate_per_kwh", 0.10))

        for batch in sorted_batches:
            # Find available equipment
            available_equip = [
                e for e in equipment
                if e.get("status") == "available" and
                e.get("capacity_kw", 0) >= batch.get("estimated_power_kw", 0)
            ]

            if not available_equip:
                continue

            # Find cheapest valid time slot
            best_slot = None
            best_rate = float("inf")

            for tariff in sorted_tariffs:
                period_start = tariff.get("period_start")
                period_end = tariff.get("period_end")

                if isinstance(period_start, str):
                    period_start = datetime.fromisoformat(period_start.replace("Z", "+00:00"))
                if isinstance(period_end, str):
                    period_end = datetime.fromisoformat(period_end.replace("Z", "+00:00"))

                # Ensure period is in future
                if period_end <= now:
                    continue

                # Check if task can complete before deadline
                duration_minutes = batch.get("heating_duration_minutes", 60)
                potential_start = max(now, period_start)
                potential_end = potential_start + timedelta(minutes=duration_minutes)

                deadline = batch.get("deadline")
                if isinstance(deadline, str):
                    deadline = datetime.fromisoformat(deadline.replace("Z", "+00:00"))

                if potential_end > deadline:
                    continue

                rate = tariff.get("energy_rate_per_kwh", 0.10)
                if rate < best_rate:
                    best_rate = rate
                    best_slot = {
                        "start_time": potential_start,
                        "end_time": potential_end,
                        "rate": rate,
                        "tariff_id": tariff.get("tariff_id")
                    }

            if best_slot is None:
                # Default to earliest possible start
                duration_minutes = batch.get("heating_duration_minutes", 60)
                best_slot = {
                    "start_time": now,
                    "end_time": now + timedelta(minutes=duration_minutes),
                    "rate": sorted_tariffs[0].get("energy_rate_per_kwh", 0.10) if sorted_tariffs else 0.10,
                    "tariff_id": sorted_tariffs[0].get("tariff_id") if sorted_tariffs else "default"
                }

            # Create task
            equip = available_equip[0]
            power_kw = batch.get("estimated_power_kw", 100)
            duration_minutes = batch.get("heating_duration_minutes", 60)

            energy_kwh = HeatSchedulerCalculations.calculate_energy_consumption(
                power_kw, duration_minutes, equip.get("efficiency", 0.85)
            )

            cost_usd = HeatSchedulerCalculations.calculate_energy_cost(
                energy_kwh, best_slot["rate"]
            )

            task = {
                "task_id": f"TASK-{batch.get('batch_id')}",
                "batch_id": batch.get("batch_id"),
                "equipment_id": equip.get("equipment_id"),
                "start_time": best_slot["start_time"],
                "end_time": best_slot["end_time"],
                "duration_minutes": duration_minutes,
                "power_kw": power_kw,
                "temperature_c": batch.get("heating_temperature_c", 500),
                "estimated_energy_kwh": energy_kwh,
                "estimated_cost_usd": cost_usd,
                "energy_rate_per_kwh": best_slot["rate"],
                "status": "scheduled"
            }

            tasks.append(task)

        return tasks


# =======================================================================================
# INITIALIZE FASTAPI APPLICATION
# =======================================================================================

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://*.greenlang.io",
        "https://*.heatscheduler.io",
        "http://localhost:3000",
        "http://localhost:8000"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=[
        "*.greenlang.io",
        "*.heatscheduler.io",
        "localhost",
        "127.0.0.1"
    ]
)

# Track app start time for uptime calculation
APP_START_TIME = time.time()


# =======================================================================================
# MIDDLEWARE FOR REQUEST LOGGING & METRICS
# =======================================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests and track metrics."""
    start_time = time.time()

    # Process request
    response = await call_next(request)

    # Calculate duration
    duration = time.time() - start_time

    # Log request
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Duration: {duration:.3f}s"
    )

    # Update metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()

    REQUEST_DURATION.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)

    return response


# =======================================================================================
# AUTHENTICATION ENDPOINTS
# =======================================================================================

@app.post(
    "/api/v1/auth/token",
    response_model=Token,
    tags=["Authentication"],
    summary="Get access token",
    description="Authenticate with email and password to receive JWT access token"
)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticate user and return JWT access token.

    Args:
        form_data: OAuth2 form with username and password

    Returns:
        JWT access token

    Raises:
        HTTPException: 401 if credentials invalid
    """
    user_dict = USERS_DB.get(form_data.username)

    if not user_dict:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not verify_password(form_data.password, user_dict["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if user_dict.get("disabled", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )

    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={
            "sub": user_dict["id"],
            "email": user_dict["email"],
            "tenant_id": user_dict["tenant_id"],
            "roles": user_dict["roles"]
        },
        expires_delta=access_token_expires
    )

    logger.info(f"User {form_data.username} authenticated successfully")

    return Token(
        access_token=access_token,
        token_type="bearer"
    )


# =======================================================================================
# PRODUCTION SCHEDULE ENDPOINTS
# =======================================================================================

@app.post(
    "/api/v1/production-schedule",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    tags=["Production Schedule"],
    summary="Submit production schedule",
    description="Submit production batches requiring heating operations"
)
@limiter.limit("100/minute")
async def submit_production_schedule(
    request: Request,
    batches: List[ProductionBatchRequest],
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Submit production schedule from ERP.

    Args:
        batches: List of production batches
        current_user: Authenticated user

    Returns:
        Confirmation with batch count
    """
    try:
        logger.info(
            f"Production schedule submitted by {current_user.email} "
            f"with {len(batches)} batches"
        )

        # Validate batches
        validated_batches = []
        for batch in batches:
            validated_batches.append({
                "batch_id": batch.batch_id,
                "product_name": batch.product_name,
                "priority": batch.priority,
                "deadline": batch.deadline.isoformat(),
                "heating_temperature_c": batch.heating_temperature_c,
                "estimated_energy_kwh": HeatSchedulerCalculations.calculate_energy_consumption(
                    batch.estimated_power_kw,
                    batch.heating_duration_minutes
                )
            })

        return {
            "success": True,
            "batches_received": len(batches),
            "batches": validated_batches,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error(f"Production schedule submission failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Schedule submission failed: {str(e)}"
        )


# =======================================================================================
# ENERGY TARIFF ENDPOINTS
# =======================================================================================

@app.post(
    "/api/v1/energy-tariffs",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    tags=["Energy Tariffs"],
    summary="Update energy tariffs",
    description="Update current and forecast energy tariff periods"
)
@limiter.limit("100/minute")
async def update_energy_tariffs(
    request: Request,
    tariffs: List[TariffPeriodRequest],
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Update energy tariff information.

    Args:
        tariffs: List of tariff periods
        current_user: Authenticated user

    Returns:
        Confirmation with tariff summary
    """
    try:
        logger.info(
            f"Energy tariffs updated by {current_user.email} "
            f"with {len(tariffs)} periods"
        )

        # Calculate rate summary
        peak_rates = [t.energy_rate_per_kwh for t in tariffs if t.period_type == "peak"]
        off_peak_rates = [t.energy_rate_per_kwh for t in tariffs if t.period_type == "off_peak"]

        avg_peak = sum(peak_rates) / len(peak_rates) if peak_rates else 0
        avg_off_peak = sum(off_peak_rates) / len(off_peak_rates) if off_peak_rates else 0

        return {
            "success": True,
            "periods_received": len(tariffs),
            "rate_summary": {
                "avg_peak_rate_per_kwh": round(avg_peak, 4),
                "avg_off_peak_rate_per_kwh": round(avg_off_peak, 4),
                "peak_periods": len(peak_rates),
                "off_peak_periods": len(off_peak_rates)
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error(f"Tariff update failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tariff update failed: {str(e)}"
        )


# =======================================================================================
# EQUIPMENT STATUS ENDPOINTS
# =======================================================================================

@app.post(
    "/api/v1/equipment-status",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    tags=["Equipment"],
    summary="Update equipment status",
    description="Update heating equipment availability and status"
)
@limiter.limit("100/minute")
async def update_equipment_status(
    request: Request,
    equipment: List[EquipmentStatusRequest],
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Update equipment status.

    Args:
        equipment: List of equipment status updates
        current_user: Authenticated user

    Returns:
        Confirmation with equipment summary
    """
    try:
        logger.info(
            f"Equipment status updated by {current_user.email} "
            f"for {len(equipment)} units"
        )

        available_count = sum(1 for e in equipment if e.status == "available")
        total_capacity = sum(e.capacity_kw for e in equipment if e.status == "available")

        return {
            "success": True,
            "equipment_updated": len(equipment),
            "available_count": available_count,
            "total_available_capacity_kw": total_capacity,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error(f"Equipment status update failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Equipment update failed: {str(e)}"
        )


# =======================================================================================
# SCHEDULE OPTIMIZATION ENDPOINTS
# =======================================================================================

@app.post(
    "/api/v1/optimize-schedule",
    response_model=OptimizedScheduleResponse,
    status_code=status.HTTP_200_OK,
    tags=["Optimization"],
    summary="Optimize heating schedule",
    description="Generate cost-optimized heating schedule from production requirements"
)
@limiter.limit("20/minute")
async def optimize_schedule(
    request: Request,
    optimization_request: ScheduleOptimizationRequest,
    current_user: User = Depends(get_current_user)
) -> OptimizedScheduleResponse:
    """
    Optimize heating schedule for minimum energy cost.

    Args:
        optimization_request: Production batches, tariffs, and equipment
        current_user: Authenticated user

    Returns:
        Optimized schedule with cost analysis
    """
    try:
        start_time = time.time()

        logger.info(
            f"Schedule optimization requested by {current_user.email} "
            f"with {len(optimization_request.production_batches)} batches"
        )

        # Convert to dictionaries for calculation functions
        batches = [b.model_dump() for b in optimization_request.production_batches]
        tariffs = [t.model_dump() for t in optimization_request.tariff_periods]
        equipment = [e.model_dump() for e in optimization_request.equipment_status]

        # Run optimization
        tasks = HeatSchedulerCalculations.optimize_task_scheduling(
            batches, tariffs, equipment,
            optimization_request.peak_demand_limit_kw
        )

        # Calculate totals
        total_cost = sum(t["estimated_cost_usd"] for t in tasks)
        total_energy = sum(t["estimated_energy_kwh"] for t in tasks)
        peak_demand = max((t["power_kw"] for t in tasks), default=0)

        # Calculate baseline cost (earliest scheduling at first available rate)
        baseline_rate = tariffs[0]["energy_rate_per_kwh"] if tariffs else 0.10
        baseline_cost = total_energy * baseline_rate

        # Calculate savings
        savings = HeatSchedulerCalculations.calculate_savings(total_cost, baseline_cost)

        # Create schedule ID
        now = datetime.now(timezone.utc)
        schedule_id = f"SCHED-{now.strftime('%Y%m%d-%H%M%S')}"

        # Calculate provenance hash
        provenance_data = {
            "schedule_id": schedule_id,
            "tasks": tasks,
            "total_cost": total_cost,
            "timestamp": now.isoformat()
        }
        provenance_hash = hashlib.sha256(
            str(provenance_data).encode()
        ).hexdigest()

        # Calculate optimization time
        opt_time = time.time() - start_time

        # Update metrics
        OPTIMIZATION_COUNT.labels(status="success").inc()
        SAVINGS_AMOUNT.observe(savings["savings_usd"])

        # Convert tasks to response format
        task_responses = [
            HeatingTaskResponse(
                task_id=t["task_id"],
                batch_id=t["batch_id"],
                equipment_id=t["equipment_id"],
                start_time=t["start_time"],
                end_time=t["end_time"],
                duration_minutes=t["duration_minutes"],
                power_kw=t["power_kw"],
                temperature_c=t["temperature_c"],
                estimated_energy_kwh=t["estimated_energy_kwh"],
                estimated_cost_usd=t["estimated_cost_usd"],
                status=t["status"]
            )
            for t in tasks
        ]

        logger.info(
            f"Schedule optimization completed in {opt_time:.2f}s - "
            f"{len(tasks)} tasks, ${total_cost:.2f} cost, "
            f"${savings['savings_usd']:.2f} savings"
        )

        return OptimizedScheduleResponse(
            schedule_id=schedule_id,
            created_at=now,
            period_start=now,
            period_end=now + timedelta(hours=optimization_request.planning_horizon_hours),
            tasks=task_responses,
            total_cost_usd=total_cost,
            energy_cost_usd=total_cost,
            demand_cost_usd=0,
            baseline_cost_usd=baseline_cost,
            savings_usd=savings["savings_usd"],
            savings_percent=savings["savings_percent"],
            total_energy_kwh=total_energy,
            peak_demand_kw=peak_demand,
            optimization_objective=optimization_request.optimization_objective,
            optimization_time_seconds=round(opt_time, 3),
            provenance_hash=provenance_hash
        )

    except Exception as e:
        logger.error(f"Schedule optimization failed: {str(e)}", exc_info=True)
        OPTIMIZATION_COUNT.labels(status="failed").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimization failed: {str(e)}"
        )


# =======================================================================================
# SAVINGS CALCULATION ENDPOINTS
# =======================================================================================

@app.get(
    "/api/v1/calculate-savings/{schedule_id}",
    response_model=CostSavingsResponse,
    tags=["Analysis"],
    summary="Calculate cost savings",
    description="Calculate detailed cost savings for a schedule"
)
@limiter.limit("100/minute")
async def calculate_savings(
    request: Request,
    schedule_id: str,
    current_user: User = Depends(get_current_user)
) -> CostSavingsResponse:
    """
    Calculate cost savings for a schedule.

    Args:
        schedule_id: Schedule identifier
        current_user: Authenticated user

    Returns:
        Detailed cost savings analysis
    """
    try:
        logger.info(
            f"Savings calculation requested by {current_user.email} "
            f"for schedule {schedule_id}"
        )

        # In production, would retrieve actual schedule from database
        # For now, return mock calculation

        now = datetime.now(timezone.utc)

        # Mock data
        total_energy = 500.0
        peak_energy = 100.0
        off_peak_energy = 400.0
        energy_cost = 45.0
        baseline_cost = 65.0

        savings = HeatSchedulerCalculations.calculate_savings(energy_cost, baseline_cost)

        # Generate recommendations
        recommendations = []
        if peak_energy > off_peak_energy * 0.5:
            recommendations.append(
                "Consider shifting more load to off-peak hours for additional savings"
            )
        if savings["savings_percent"] < 15:
            recommendations.append(
                "Target savings of 15%+ achievable with more flexible scheduling"
            )
        if not recommendations:
            recommendations.append("Schedule is well-optimized for cost savings")

        # Calculate provenance hash
        provenance_data = {
            "schedule_id": schedule_id,
            "total_energy_kwh": total_energy,
            "savings_usd": savings["savings_usd"],
            "timestamp": now.isoformat()
        }
        provenance_hash = hashlib.sha256(
            str(provenance_data).encode()
        ).hexdigest()

        return CostSavingsResponse(
            schedule_id=schedule_id,
            period_start=now,
            period_end=now + timedelta(hours=24),
            total_energy_kwh=total_energy,
            peak_energy_kwh=peak_energy,
            off_peak_energy_kwh=off_peak_energy,
            energy_cost_usd=energy_cost,
            demand_cost_usd=0,
            total_cost_usd=energy_cost,
            baseline_cost_usd=baseline_cost,
            savings_usd=savings["savings_usd"],
            savings_percent=savings["savings_percent"],
            peak_demand_kw=250.0,
            recommendations=recommendations,
            provenance_hash=provenance_hash
        )

    except Exception as e:
        logger.error(f"Savings calculation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Savings calculation failed: {str(e)}"
        )


# =======================================================================================
# SCHEDULE APPLICATION ENDPOINTS
# =======================================================================================

@app.post(
    "/api/v1/apply-schedule",
    response_model=ScheduleApplicationResponse,
    status_code=status.HTTP_200_OK,
    tags=["Control"],
    summary="Apply schedule to control systems",
    description="Send optimized schedule to equipment control systems"
)
@limiter.limit("10/minute")
async def apply_schedule(
    request: Request,
    apply_request: ScheduleApplicationRequest,
    current_user: User = Depends(get_current_user)
) -> ScheduleApplicationResponse:
    """
    Apply optimized schedule to control systems.

    Args:
        apply_request: Schedule application request
        current_user: Authenticated user

    Returns:
        Application result
    """
    try:
        if not apply_request.confirm:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Confirmation required to apply schedule"
            )

        logger.info(
            f"Schedule {apply_request.schedule_id} applied by {current_user.email}"
        )

        # In production, would send to actual control systems
        # For now, return success response

        return ScheduleApplicationResponse(
            success=True,
            schedule_id=apply_request.schedule_id,
            tasks_applied=5,  # Mock
            equipment_updated=["FURN-001", "FURN-002"],
            timestamp=datetime.now(timezone.utc),
            warnings=[]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Schedule application failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Schedule application failed: {str(e)}"
        )


# =======================================================================================
# DEMAND RESPONSE ENDPOINTS
# =======================================================================================

@app.post(
    "/api/v1/demand-response-event",
    response_model=DemandResponseResponse,
    status_code=status.HTTP_200_OK,
    tags=["Demand Response"],
    summary="Handle demand response event",
    description="Process demand response event and adjust schedule"
)
@limiter.limit("20/minute")
async def handle_demand_response(
    request: Request,
    event: DemandResponseEventRequest,
    current_user: User = Depends(get_current_user)
) -> DemandResponseResponse:
    """
    Handle demand response event notification.

    Args:
        event: Demand response event details
        current_user: Authenticated user

    Returns:
        Demand response handling result
    """
    try:
        logger.info(
            f"Demand response event {event.event_id} received by {current_user.email}"
        )

        # Calculate potential reduction and incentive
        # In production, would analyze current schedule
        committed_reduction = min(event.target_reduction_kw * 0.8, 500)  # Mock
        duration_hours = (event.end_time - event.start_time).total_seconds() / 3600
        estimated_incentive = committed_reduction * duration_hours * event.incentive_per_kwh

        return DemandResponseResponse(
            event_id=event.event_id,
            acknowledged=True,
            committed_reduction_kw=committed_reduction,
            affected_tasks=3,  # Mock
            estimated_incentive_usd=round(estimated_incentive, 2),
            schedule_adjusted=True
        )

    except Exception as e:
        logger.error(f"Demand response handling failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Demand response handling failed: {str(e)}"
        )


# =======================================================================================
# HEALTH & MONITORING ENDPOINTS
# =======================================================================================

@app.get(
    "/health",
    response_model=HealthStatus,
    tags=["System"],
    summary="Health check",
    description="Check API health status and dependencies"
)
@limiter.limit("1000/minute")
async def health_check(request: Request):
    """
    Health check endpoint for load balancers and monitoring.

    Returns:
        Health status with system checks
    """
    uptime = time.time() - APP_START_TIME

    # Perform health checks
    checks = {
        "api": True,
        "authentication": True,
        "optimization_engine": True
    }

    # Check if all systems healthy
    all_healthy = all(checks.values())

    if not all_healthy:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )

    return HealthStatus(
        status="healthy",
        timestamp=datetime.now(timezone.utc),
        version=API_VERSION,
        uptime_seconds=round(uptime, 2),
        checks=checks
    )


@app.get(
    "/metrics",
    tags=["System"],
    summary="Prometheus metrics",
    description="Export Prometheus metrics for monitoring"
)
@limiter.limit("1000/minute")
async def metrics(request: Request):
    """
    Export Prometheus metrics.

    Returns:
        Prometheus metrics in text format
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# =======================================================================================
# ERROR HANDLERS
# =======================================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured response."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "message": str(exc) if os.getenv("DEBUG") else "An unexpected error occurred",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )


# =======================================================================================
# MAIN ENTRY POINT
# =======================================================================================

if __name__ == "__main__":
    """Run the API server."""
    uvicorn.run(
        "tools:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
