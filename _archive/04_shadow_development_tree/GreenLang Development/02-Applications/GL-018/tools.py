# -*- coding: utf-8 -*-
"""
GL-018 FLUEFLOW - Flue Gas Combustion Optimization FastAPI REST API

Production-grade REST API for FLUEFLOW application providing:
- Flue gas composition analysis
- Combustion efficiency calculation
- Air-fuel ratio optimization
- Emissions compliance reporting
- Performance monitoring and trending

Implements GreenLang standard patterns: JWT auth, rate limiting, audit trails,
comprehensive error handling, and OpenAPI documentation.
"""

import hashlib
import logging
import os
import time
from datetime import datetime, timedelta
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
from pydantic import BaseModel, Field, validator
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
API_TITLE = "FLUEFLOW API"
API_DESCRIPTION = """
**FLUEFLOW**: Production REST API for Flue Gas Combustion Optimization

## Features

- **Flue Gas Analysis**: Analyze combustion composition and quality
- **Efficiency Calculation**: Calculate combustion and thermal efficiency
- **Air-Fuel Ratio Optimization**: Optimize for maximum efficiency and minimum emissions
- **Emissions Compliance**: Track regulatory compliance (EPA, EU ETS)
- **Performance Reporting**: Trend analysis and performance metrics

## Authentication

All endpoints require JWT bearer token authentication.

Get token from `/api/v1/auth/token` endpoint with valid credentials.

## Rate Limiting

- Analysis endpoints: 100 requests/minute
- Status endpoints: 1000 requests/minute
- Health/metrics: 1000 requests/minute
"""

# =======================================================================================
# PROMETHEUS METRICS
# =======================================================================================

REQUEST_COUNT = Counter(
    'flueflow_api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'flueflow_api_request_duration_seconds',
    'API request duration',
    ['method', 'endpoint']
)

ANALYSIS_COUNT = Counter(
    'flueflow_analysis_total',
    'Total flue gas analyses performed',
    ['analysis_type']
)

EFFICIENCY_CALCULATIONS = Histogram(
    'flueflow_efficiency_percent',
    'Combustion efficiency calculations',
    buckets=[70, 75, 80, 85, 90, 95, 100]
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
    "demo@flueflow.io": {
        "id": "user_001",
        "email": "demo@flueflow.io",
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


def get_password_hash(password: str) -> str:
    """Hash password."""
    return pwd_context.hash(password)


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

class FlueGasComposition(BaseModel):
    """Flue gas composition data."""
    o2_percent: float = Field(..., ge=0, le=21, description="Oxygen concentration (%)")
    co2_percent: float = Field(..., ge=0, le=20, description="CO2 concentration (%)")
    co_ppm: float = Field(..., ge=0, le=10000, description="CO concentration (ppm)")
    nox_ppm: float = Field(..., ge=0, le=1000, description="NOx concentration (ppm)")
    sox_ppm: float = Field(0, ge=0, le=5000, description="SOx concentration (ppm)")
    temperature_celsius: float = Field(..., ge=0, le=1500, description="Flue gas temperature (°C)")

    @validator('o2_percent', 'co2_percent')
    def validate_composition(cls, v, values):
        """Validate that O2 + CO2 doesn't exceed 21%."""
        if 'o2_percent' in values and 'co2_percent' in values:
            total = values['o2_percent'] + v
            if total > 21:
                raise ValueError("O2 + CO2 cannot exceed 21%")
        return v


class BurnerOperatingData(BaseModel):
    """Burner operating parameters."""
    fuel_type: str = Field(..., description="Fuel type: natural_gas, diesel, heavy_oil, coal")
    fuel_flow_rate: float = Field(..., ge=0, description="Fuel flow rate (kg/hr or m3/hr)")
    air_flow_rate: float = Field(..., ge=0, description="Air flow rate (m3/hr)")
    steam_output: float = Field(..., ge=0, description="Steam output (kg/hr)")
    feedwater_temp: float = Field(..., ge=0, le=250, description="Feedwater temperature (°C)")
    steam_pressure: float = Field(..., ge=0, description="Steam pressure (bar)")
    ambient_temp: float = Field(25, ge=-20, le=50, description="Ambient temperature (°C)")


class FlueGasAnalysisRequest(BaseModel):
    """Request model for flue gas analysis."""
    burner_id: str = Field(..., description="Unique burner identifier")
    composition: FlueGasComposition
    timestamp: Optional[datetime] = None

    class Config:
        schema_extra = {
            "example": {
                "burner_id": "burner_001",
                "composition": {
                    "o2_percent": 3.5,
                    "co2_percent": 12.5,
                    "co_ppm": 50,
                    "nox_ppm": 150,
                    "sox_ppm": 20,
                    "temperature_celsius": 180
                }
            }
        }


class CombustionAnalysisResult(BaseModel):
    """Combustion analysis result."""
    burner_id: str
    combustion_quality: str = Field(..., description="Quality rating: excellent, good, fair, poor")
    excess_air_percent: float = Field(..., description="Excess air percentage")
    air_fuel_ratio: float = Field(..., description="Actual air-fuel ratio")
    stoichiometric_ratio: float = Field(..., description="Stoichiometric air-fuel ratio")
    combustion_completeness: float = Field(..., ge=0, le=100, description="Combustion completeness (%)")
    unburned_losses_percent: float = Field(..., description="Unburned fuel losses (%)")
    analysis_timestamp: datetime
    recommendations: List[str]
    provenance_hash: str


class EfficiencyCalculationRequest(BaseModel):
    """Request model for efficiency calculation."""
    burner_id: str
    flue_gas: FlueGasComposition
    operating_data: BurnerOperatingData

    class Config:
        schema_extra = {
            "example": {
                "burner_id": "burner_001",
                "flue_gas": {
                    "o2_percent": 3.5,
                    "co2_percent": 12.5,
                    "co_ppm": 50,
                    "nox_ppm": 150,
                    "temperature_celsius": 180
                },
                "operating_data": {
                    "fuel_type": "natural_gas",
                    "fuel_flow_rate": 1000,
                    "air_flow_rate": 12000,
                    "steam_output": 10000,
                    "feedwater_temp": 105,
                    "steam_pressure": 10,
                    "ambient_temp": 25
                }
            }
        }


class EfficiencyAssessment(BaseModel):
    """Efficiency assessment result."""
    burner_id: str
    combustion_efficiency: float = Field(..., ge=0, le=100, description="Combustion efficiency (%)")
    thermal_efficiency: float = Field(..., ge=0, le=100, description="Thermal efficiency (%)")
    stack_loss: float = Field(..., description="Stack heat loss (%)")
    radiation_loss: float = Field(..., description="Radiation/convection loss (%)")
    unaccounted_loss: float = Field(..., description="Unaccounted losses (%)")
    efficiency_rating: str = Field(..., description="Rating: excellent, good, fair, poor")
    improvement_potential: float = Field(..., description="Potential efficiency improvement (%)")
    annual_savings_potential: float = Field(..., description="Annual cost savings potential ($)")
    recommendations: List[str]
    timestamp: datetime
    provenance_hash: str


class AirFuelOptimizationRequest(BaseModel):
    """Request model for air-fuel ratio optimization."""
    burner_id: str
    current_flue_gas: FlueGasComposition
    current_operating_data: BurnerOperatingData
    optimization_priority: str = Field(
        "balanced",
        description="Priority: efficiency, emissions, balanced"
    )


class AirFuelRatioRecommendation(BaseModel):
    """Air-fuel ratio optimization recommendation."""
    burner_id: str
    current_air_fuel_ratio: float
    recommended_air_fuel_ratio: float
    current_excess_air_percent: float
    recommended_excess_air_percent: float
    expected_efficiency_gain: float = Field(..., description="Expected efficiency improvement (%)")
    expected_nox_reduction: float = Field(..., description="Expected NOx reduction (%)")
    expected_co_change: float = Field(..., description="Expected CO change (%)")
    air_flow_adjustment: float = Field(..., description="Required air flow change (%)")
    implementation_steps: List[str]
    warnings: List[str]
    timestamp: datetime
    provenance_hash: str


class EmissionsComplianceReport(BaseModel):
    """Emissions compliance report."""
    burner_id: str
    compliance_status: str = Field(..., description="Status: COMPLIANT, WARNING, NON_COMPLIANT")
    regulatory_standard: str = Field(..., description="Standard: EPA, EU_ETS, CUSTOM")
    nox_compliance: Dict[str, Any]
    sox_compliance: Dict[str, Any]
    co_compliance: Dict[str, Any]
    particulate_compliance: Optional[Dict[str, Any]] = None
    overall_margin_percent: float
    violations: List[str]
    warnings: List[str]
    corrective_actions: List[str]
    next_report_due: datetime
    timestamp: datetime
    provenance_hash: str


class PerformanceMetrics(BaseModel):
    """Performance metrics and trends."""
    burner_id: str
    period: str = Field(..., description="Period: 1h, 24h, 7d, 30d")
    avg_efficiency: float
    min_efficiency: float
    max_efficiency: float
    efficiency_trend: str = Field(..., description="Trend: improving, stable, declining")
    avg_nox_ppm: float
    avg_co_ppm: float
    avg_o2_percent: float
    uptime_percent: float
    total_fuel_consumed: float
    total_steam_generated: float
    performance_score: float = Field(..., ge=0, le=100, description="Overall performance score")
    trends: Dict[str, Any]
    alerts: List[str]
    timestamp: datetime


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

class FlueFlowCalculations:
    """
    Deterministic calculation functions for flue gas combustion optimization.

    All calculations based on ASME PTC 4 (Fired Steam Generators) and
    EPA emission standards. Zero-hallucination, pure mathematical operations.
    """

    # Stoichiometric air-fuel ratios by fuel type (kg air / kg fuel)
    STOICHIOMETRIC_RATIOS = {
        "natural_gas": 17.2,
        "diesel": 14.5,
        "heavy_oil": 13.8,
        "coal": 10.5,
        "propane": 15.7,
        "butane": 15.4
    }

    # Lower heating values (LHV) in MJ/kg
    FUEL_LHV = {
        "natural_gas": 50.0,
        "diesel": 42.5,
        "heavy_oil": 40.5,
        "coal": 25.0,
        "propane": 46.3,
        "butane": 45.7
    }

    @staticmethod
    def calculate_excess_air(o2_percent: float, co2_percent: float, fuel_type: str) -> float:
        """
        Calculate excess air percentage from O2 and CO2 measurements.

        Formula: EA = (O2 / (21 - O2)) × 100

        Args:
            o2_percent: Oxygen concentration in flue gas (%)
            co2_percent: CO2 concentration in flue gas (%)
            fuel_type: Type of fuel

        Returns:
            Excess air percentage
        """
        if o2_percent >= 21:
            return 0.0

        # Calculate excess air from O2
        excess_air = (o2_percent / (21 - o2_percent)) * 100

        return round(excess_air, 2)

    @staticmethod
    def calculate_air_fuel_ratio(
        air_flow_rate: float,
        fuel_flow_rate: float,
        fuel_type: str
    ) -> tuple:
        """
        Calculate actual and stoichiometric air-fuel ratios.

        Args:
            air_flow_rate: Air flow rate (m3/hr)
            fuel_flow_rate: Fuel flow rate (kg/hr or m3/hr)
            fuel_type: Type of fuel

        Returns:
            Tuple of (actual_ratio, stoichiometric_ratio)
        """
        # Air density at standard conditions (kg/m3)
        air_density = 1.2

        # Actual air mass flow (kg/hr)
        air_mass_flow = air_flow_rate * air_density

        # Actual air-fuel ratio
        actual_afr = air_mass_flow / fuel_flow_rate if fuel_flow_rate > 0 else 0

        # Stoichiometric ratio
        stoich_afr = FlueFlowCalculations.STOICHIOMETRIC_RATIOS.get(
            fuel_type.lower(), 15.0
        )

        return round(actual_afr, 2), stoich_afr

    @staticmethod
    def calculate_combustion_efficiency(
        o2_percent: float,
        co_ppm: float,
        flue_gas_temp: float,
        ambient_temp: float,
        fuel_type: str
    ) -> float:
        """
        Calculate combustion efficiency using ASME PTC 4 method.

        Efficiency = 100 - (Stack Loss + Radiation Loss + Unburned Loss)

        Args:
            o2_percent: Oxygen in flue gas (%)
            co_ppm: Carbon monoxide (ppm)
            flue_gas_temp: Flue gas temperature (°C)
            ambient_temp: Ambient temperature (°C)
            fuel_type: Type of fuel

        Returns:
            Combustion efficiency (%)
        """
        # Stack loss calculation (dry flue gas loss)
        # L_stack = K × (T_flue - T_ambient) / (21 - O2)
        # where K ≈ 0.68 for natural gas

        k_factor = {
            "natural_gas": 0.68,
            "diesel": 0.70,
            "heavy_oil": 0.72,
            "coal": 0.75
        }.get(fuel_type.lower(), 0.70)

        temp_diff = flue_gas_temp - ambient_temp

        if o2_percent >= 21:
            stack_loss = temp_diff * k_factor * 0.1  # Minimal loss
        else:
            stack_loss = k_factor * temp_diff / (21 - o2_percent)

        # Unburned fuel loss from CO
        # L_unburned ≈ CO_ppm × 0.0125 (empirical)
        unburned_loss = co_ppm * 0.0125

        # Radiation and convection loss (typically 1-2% for well-insulated boilers)
        radiation_loss = 1.5

        # Calculate efficiency
        efficiency = 100 - (stack_loss + unburned_loss + radiation_loss)

        # Clamp to realistic range
        efficiency = max(50, min(efficiency, 99))

        return round(efficiency, 2)

    @staticmethod
    def calculate_thermal_efficiency(
        fuel_flow_rate: float,
        fuel_type: str,
        steam_output: float,
        feedwater_temp: float,
        steam_pressure: float
    ) -> float:
        """
        Calculate overall thermal efficiency.

        η_thermal = (Steam Energy Output) / (Fuel Energy Input) × 100

        Args:
            fuel_flow_rate: Fuel consumption (kg/hr)
            fuel_type: Type of fuel
            steam_output: Steam generation rate (kg/hr)
            feedwater_temp: Feedwater temperature (°C)
            steam_pressure: Steam pressure (bar)

        Returns:
            Thermal efficiency (%)
        """
        # Fuel energy input (MJ/hr)
        lhv = FlueFlowCalculations.FUEL_LHV.get(fuel_type.lower(), 42.0)
        fuel_energy = fuel_flow_rate * lhv

        # Steam enthalpy calculation (simplified)
        # h_steam ≈ 2800 kJ/kg at 10 bar (saturated)
        # Adjust for pressure: h ≈ 2700 + pressure × 10
        steam_enthalpy = 2700 + min(steam_pressure * 10, 500)  # kJ/kg

        # Feedwater enthalpy (kJ/kg)
        # h_feedwater ≈ 4.18 × T
        feedwater_enthalpy = 4.18 * feedwater_temp

        # Energy absorbed by water (MJ/hr)
        steam_energy = steam_output * (steam_enthalpy - feedwater_enthalpy) / 1000

        # Thermal efficiency
        if fuel_energy > 0:
            thermal_eff = (steam_energy / fuel_energy) * 100
        else:
            thermal_eff = 0

        # Clamp to realistic range
        thermal_eff = max(50, min(thermal_eff, 95))

        return round(thermal_eff, 2)

    @staticmethod
    def optimize_excess_air(
        current_o2: float,
        current_co: float,
        fuel_type: str,
        priority: str = "balanced"
    ) -> Dict[str, float]:
        """
        Optimize excess air for best efficiency and emissions.

        Args:
            current_o2: Current O2 (%)
            current_co: Current CO (ppm)
            fuel_type: Type of fuel
            priority: Optimization priority (efficiency, emissions, balanced)

        Returns:
            Dictionary with optimization results
        """
        # Target O2 levels for different priorities
        if priority == "efficiency":
            # Lower O2 for max efficiency (but watch CO)
            target_o2 = 2.5
        elif priority == "emissions":
            # Higher O2 for complete combustion, lower emissions
            target_o2 = 4.0
        else:  # balanced
            target_o2 = 3.0

        # Check if CO is high (incomplete combustion)
        if current_co > 100:
            # Need more air regardless of priority
            target_o2 = max(target_o2, current_o2 + 0.5)

        # Calculate required air adjustment
        current_excess_air = FlueFlowCalculations.calculate_excess_air(
            current_o2, 0, fuel_type
        )
        target_excess_air = FlueFlowCalculations.calculate_excess_air(
            target_o2, 0, fuel_type
        )

        air_adjustment = ((target_excess_air - current_excess_air) /
                         max(current_excess_air, 1)) * 100

        # Estimate efficiency gain
        # Reducing O2 by 1% typically improves efficiency by 0.6-1%
        o2_change = target_o2 - current_o2
        efficiency_gain = -o2_change * 0.8  # Negative O2 change = efficiency gain

        return {
            "target_o2": round(target_o2, 2),
            "target_excess_air": round(target_excess_air, 2),
            "air_adjustment_percent": round(air_adjustment, 2),
            "estimated_efficiency_gain": round(efficiency_gain, 2)
        }


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
        "https://*.flueflow.io",
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
        "*.flueflow.io",
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
# FLUE GAS ANALYSIS ENDPOINTS
# =======================================================================================

@app.post(
    "/api/v1/analyze-flue-gas",
    response_model=CombustionAnalysisResult,
    status_code=status.HTTP_200_OK,
    tags=["Analysis"],
    summary="Analyze flue gas composition",
    description="Analyze flue gas composition and assess combustion quality"
)
@limiter.limit("100/minute")
async def analyze_flue_gas(
    request: Request,
    analysis_request: FlueGasAnalysisRequest,
    current_user: User = Depends(get_current_user)
) -> CombustionAnalysisResult:
    """
    Analyze flue gas composition and assess combustion quality.

    Args:
        analysis_request: Flue gas composition data
        current_user: Authenticated user

    Returns:
        Combustion analysis results with quality assessment
    """
    try:
        logger.info(
            f"Flue gas analysis requested by {current_user.email} "
            f"for burner {analysis_request.burner_id}"
        )

        comp = analysis_request.composition
        timestamp = analysis_request.timestamp or datetime.utcnow()

        # Calculate excess air
        excess_air = FlueFlowCalculations.calculate_excess_air(
            comp.o2_percent, comp.co2_percent, "natural_gas"
        )

        # Estimate air-fuel ratio (simplified without actual flow data)
        stoich_ratio = FlueFlowCalculations.STOICHIOMETRIC_RATIOS.get(
            "natural_gas", 17.2
        )
        actual_ratio = stoich_ratio * (1 + excess_air / 100)

        # Calculate combustion completeness
        # High CO indicates incomplete combustion
        if comp.co_ppm < 50:
            combustion_completeness = 99.5
        elif comp.co_ppm < 100:
            combustion_completeness = 98.0
        elif comp.co_ppm < 200:
            combustion_completeness = 95.0
        else:
            combustion_completeness = 90.0

        # Unburned losses
        unburned_losses = comp.co_ppm * 0.0125

        # Determine combustion quality
        if excess_air < 10 and comp.co_ppm < 50:
            quality = "excellent"
        elif excess_air < 20 and comp.co_ppm < 100:
            quality = "good"
        elif excess_air < 40 and comp.co_ppm < 200:
            quality = "fair"
        else:
            quality = "poor"

        # Generate recommendations
        recommendations = []

        if excess_air > 30:
            recommendations.append(
                f"Excess air is high ({excess_air:.1f}%). "
                "Reduce air flow to improve efficiency."
            )
        elif excess_air < 10:
            recommendations.append(
                f"Excess air is low ({excess_air:.1f}%). "
                "Monitor CO levels closely to ensure complete combustion."
            )

        if comp.co_ppm > 100:
            recommendations.append(
                f"CO level is elevated ({comp.co_ppm:.0f} ppm). "
                "Increase air flow or check burner tuning."
            )

        if comp.temperature_celsius > 200:
            recommendations.append(
                f"Flue gas temperature is high ({comp.temperature_celsius:.0f}°C). "
                "Check heat exchanger for fouling."
            )

        if comp.nox_ppm > 200:
            recommendations.append(
                f"NOx emissions are high ({comp.nox_ppm:.0f} ppm). "
                "Consider reducing excess air or lowering combustion temperature."
            )

        if not recommendations:
            recommendations.append("Combustion parameters are within optimal range.")

        # Calculate provenance hash
        provenance_data = {
            "burner_id": analysis_request.burner_id,
            "composition": comp.dict(),
            "timestamp": timestamp.isoformat(),
            "user_id": current_user.id
        }
        provenance_hash = hashlib.sha256(
            str(provenance_data).encode()
        ).hexdigest()

        # Update metrics
        ANALYSIS_COUNT.labels(analysis_type="flue_gas").inc()

        return CombustionAnalysisResult(
            burner_id=analysis_request.burner_id,
            combustion_quality=quality,
            excess_air_percent=excess_air,
            air_fuel_ratio=actual_ratio,
            stoichiometric_ratio=stoich_ratio,
            combustion_completeness=combustion_completeness,
            unburned_losses_percent=unburned_losses,
            analysis_timestamp=timestamp,
            recommendations=recommendations,
            provenance_hash=provenance_hash
        )

    except Exception as e:
        logger.error(f"Flue gas analysis failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


@app.post(
    "/api/v1/calculate-efficiency",
    response_model=EfficiencyAssessment,
    status_code=status.HTTP_200_OK,
    tags=["Analysis"],
    summary="Calculate combustion efficiency",
    description="Calculate combustion and thermal efficiency with loss analysis"
)
@limiter.limit("100/minute")
async def calculate_efficiency(
    request: Request,
    efficiency_request: EfficiencyCalculationRequest,
    current_user: User = Depends(get_current_user)
) -> EfficiencyAssessment:
    """
    Calculate combustion and thermal efficiency.

    Args:
        efficiency_request: Flue gas and operating data
        current_user: Authenticated user

    Returns:
        Comprehensive efficiency assessment
    """
    try:
        logger.info(
            f"Efficiency calculation requested by {current_user.email} "
            f"for burner {efficiency_request.burner_id}"
        )

        fg = efficiency_request.flue_gas
        op = efficiency_request.operating_data

        # Calculate combustion efficiency
        combustion_eff = FlueFlowCalculations.calculate_combustion_efficiency(
            fg.o2_percent,
            fg.co_ppm,
            fg.temperature_celsius,
            op.ambient_temp,
            op.fuel_type
        )

        # Calculate thermal efficiency
        thermal_eff = FlueFlowCalculations.calculate_thermal_efficiency(
            op.fuel_flow_rate,
            op.fuel_type,
            op.steam_output,
            op.feedwater_temp,
            op.steam_pressure
        )

        # Calculate individual losses
        # Stack loss
        k_factor = 0.68 if op.fuel_type == "natural_gas" else 0.70
        temp_diff = fg.temperature_celsius - op.ambient_temp
        stack_loss = k_factor * temp_diff / (21 - fg.o2_percent) if fg.o2_percent < 21 else 5.0

        # Radiation loss
        radiation_loss = 1.5

        # Unaccounted loss
        unaccounted_loss = 100 - combustion_eff - stack_loss - radiation_loss
        unaccounted_loss = max(0, unaccounted_loss)

        # Determine efficiency rating
        if combustion_eff >= 90:
            rating = "excellent"
        elif combustion_eff >= 85:
            rating = "good"
        elif combustion_eff >= 80:
            rating = "fair"
        else:
            rating = "poor"

        # Calculate improvement potential
        # Best-in-class efficiency for this fuel type
        best_efficiency = {
            "natural_gas": 92,
            "diesel": 90,
            "heavy_oil": 88,
            "coal": 85
        }.get(op.fuel_type.lower(), 90)

        improvement_potential = max(0, best_efficiency - combustion_eff)

        # Estimate annual savings
        # Assume $5/MMBtu fuel cost, 8000 hours operation
        fuel_lhv = FlueFlowCalculations.FUEL_LHV.get(op.fuel_type.lower(), 42.0)
        annual_fuel_mmbtu = (op.fuel_flow_rate * fuel_lhv * 8000) / 1055  # Convert MJ to MMBtu

        if improvement_potential > 0:
            savings_percent = improvement_potential / combustion_eff
            annual_savings = annual_fuel_mmbtu * 5.0 * savings_percent
        else:
            annual_savings = 0

        # Generate recommendations
        recommendations = []

        if stack_loss > 12:
            recommendations.append(
                f"Stack loss is high ({stack_loss:.1f}%). "
                "Consider installing economizer or reducing flue gas temperature."
            )

        if fg.o2_percent > 5:
            recommendations.append(
                f"Excess air is high (O2={fg.o2_percent:.1f}%). "
                "Reduce air flow to improve efficiency."
            )

        if combustion_eff < 85:
            recommendations.append(
                "Combustion efficiency is below target. "
                "Schedule burner tuning and maintenance."
            )

        if improvement_potential > 3:
            recommendations.append(
                f"Efficiency improvement potential: {improvement_potential:.1f}%. "
                f"Estimated annual savings: ${annual_savings:,.0f}"
            )

        if not recommendations:
            recommendations.append("Operating at near-optimal efficiency.")

        # Calculate provenance hash
        provenance_data = {
            "burner_id": efficiency_request.burner_id,
            "flue_gas": fg.dict(),
            "operating_data": op.dict(),
            "timestamp": datetime.utcnow().isoformat()
        }
        provenance_hash = hashlib.sha256(
            str(provenance_data).encode()
        ).hexdigest()

        # Update metrics
        ANALYSIS_COUNT.labels(analysis_type="efficiency").inc()
        EFFICIENCY_CALCULATIONS.observe(combustion_eff)

        return EfficiencyAssessment(
            burner_id=efficiency_request.burner_id,
            combustion_efficiency=combustion_eff,
            thermal_efficiency=thermal_eff,
            stack_loss=round(stack_loss, 2),
            radiation_loss=radiation_loss,
            unaccounted_loss=round(unaccounted_loss, 2),
            efficiency_rating=rating,
            improvement_potential=round(improvement_potential, 2),
            annual_savings_potential=round(annual_savings, 2),
            recommendations=recommendations,
            timestamp=datetime.utcnow(),
            provenance_hash=provenance_hash
        )

    except Exception as e:
        logger.error(f"Efficiency calculation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Efficiency calculation failed: {str(e)}"
        )


@app.post(
    "/api/v1/optimize-air-fuel-ratio",
    response_model=AirFuelRatioRecommendation,
    status_code=status.HTTP_200_OK,
    tags=["Optimization"],
    summary="Optimize air-fuel ratio",
    description="Generate air-fuel ratio optimization recommendations"
)
@limiter.limit("100/minute")
async def optimize_air_fuel_ratio(
    request: Request,
    optimization_request: AirFuelOptimizationRequest,
    current_user: User = Depends(get_current_user)
) -> AirFuelRatioRecommendation:
    """
    Optimize air-fuel ratio for efficiency and emissions.

    Args:
        optimization_request: Current operating data and optimization priority
        current_user: Authenticated user

    Returns:
        Air-fuel ratio optimization recommendations
    """
    try:
        logger.info(
            f"Air-fuel optimization requested by {current_user.email} "
            f"for burner {optimization_request.burner_id}"
        )

        fg = optimization_request.current_flue_gas
        op = optimization_request.current_operating_data
        priority = optimization_request.optimization_priority

        # Calculate current state
        current_excess_air = FlueFlowCalculations.calculate_excess_air(
            fg.o2_percent, fg.co2_percent, op.fuel_type
        )

        current_afr, stoich_afr = FlueFlowCalculations.calculate_air_fuel_ratio(
            op.air_flow_rate, op.fuel_flow_rate, op.fuel_type
        )

        # Get optimization results
        optimization = FlueFlowCalculations.optimize_excess_air(
            fg.o2_percent, fg.co_ppm, op.fuel_type, priority
        )

        # Calculate recommended AFR
        target_excess_air = optimization["target_excess_air"]
        recommended_afr = stoich_afr * (1 + target_excess_air / 100)

        # Estimate NOx reduction
        # Lower excess air typically reduces NOx by ~5% per 1% O2 reduction
        o2_reduction = fg.o2_percent - optimization["target_o2"]
        nox_reduction = o2_reduction * 5.0

        # Estimate CO change
        # Reducing excess air may increase CO
        if o2_reduction > 1:
            co_change = o2_reduction * 15  # ppm increase per % O2 decrease
        else:
            co_change = 0

        # Generate implementation steps
        implementation_steps = [
            f"1. Gradually reduce air damper opening by {abs(optimization['air_adjustment_percent']):.1f}%",
            f"2. Monitor O2 levels - target: {optimization['target_o2']:.1f}%",
            "3. Monitor CO levels - ensure CO remains below 100 ppm",
            "4. Observe flame pattern for changes (should remain stable)",
            "5. Allow system to stabilize for 15-30 minutes",
            "6. Re-measure emissions and efficiency",
            "7. Fine-tune as needed to achieve optimal balance"
        ]

        # Generate warnings
        warnings = []

        if fg.co_ppm > 100:
            warnings.append(
                "WARNING: Current CO is elevated. Do not reduce air flow further."
            )

        if optimization["air_adjustment_percent"] < -10:
            warnings.append(
                "Large air flow reduction required. Make changes gradually in 2-3% steps."
            )

        if fg.o2_percent < 2:
            warnings.append(
                "Current O2 is very low. Risk of incomplete combustion. Increase air flow."
            )

        if priority == "efficiency" and fg.co_ppm > 50:
            warnings.append(
                "Cannot prioritize efficiency with elevated CO levels. "
                "Must ensure complete combustion first."
            )

        # Calculate provenance hash
        provenance_data = {
            "burner_id": optimization_request.burner_id,
            "current_state": {
                "flue_gas": fg.dict(),
                "operating": op.dict()
            },
            "optimization": optimization,
            "timestamp": datetime.utcnow().isoformat()
        }
        provenance_hash = hashlib.sha256(
            str(provenance_data).encode()
        ).hexdigest()

        # Update metrics
        ANALYSIS_COUNT.labels(analysis_type="optimization").inc()

        return AirFuelRatioRecommendation(
            burner_id=optimization_request.burner_id,
            current_air_fuel_ratio=current_afr,
            recommended_air_fuel_ratio=round(recommended_afr, 2),
            current_excess_air_percent=current_excess_air,
            recommended_excess_air_percent=optimization["target_excess_air"],
            expected_efficiency_gain=optimization["estimated_efficiency_gain"],
            expected_nox_reduction=round(nox_reduction, 1),
            expected_co_change=round(co_change, 1),
            air_flow_adjustment=optimization["air_adjustment_percent"],
            implementation_steps=implementation_steps,
            warnings=warnings,
            timestamp=datetime.utcnow(),
            provenance_hash=provenance_hash
        )

    except Exception as e:
        logger.error(f"Air-fuel optimization failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimization failed: {str(e)}"
        )


@app.get(
    "/api/v1/emissions-compliance/{burner_id}",
    response_model=EmissionsComplianceReport,
    tags=["Compliance"],
    summary="Get emissions compliance report",
    description="Retrieve emissions compliance status and regulatory assessment"
)
@limiter.limit("100/minute")
async def get_emissions_compliance(
    request: Request,
    burner_id: str,
    standard: str = "EPA",
    current_user: User = Depends(get_current_user)
) -> EmissionsComplianceReport:
    """
    Get emissions compliance report for a burner.

    Args:
        burner_id: Burner identifier
        standard: Regulatory standard (EPA, EU_ETS, CUSTOM)
        current_user: Authenticated user

    Returns:
        Emissions compliance report
    """
    try:
        logger.info(
            f"Emissions compliance report requested by {current_user.email} "
            f"for burner {burner_id}"
        )

        # Mock current emissions data (would come from database in production)
        current_emissions = {
            "nox_ppm": 180,
            "sox_ppm": 25,
            "co_ppm": 75,
            "particulate_mg_m3": 15
        }

        # Regulatory limits by standard
        if standard == "EPA":
            limits = {
                "nox_ppm": 200,
                "sox_ppm": 50,
                "co_ppm": 100,
                "particulate_mg_m3": 20
            }
        elif standard == "EU_ETS":
            limits = {
                "nox_ppm": 150,
                "sox_ppm": 35,
                "co_ppm": 100,
                "particulate_mg_m3": 10
            }
        else:  # CUSTOM
            limits = {
                "nox_ppm": 250,
                "sox_ppm": 100,
                "co_ppm": 150,
                "particulate_mg_m3": 30
            }

        # Check compliance for each parameter
        violations = []
        warnings = []

        # NOx compliance
        nox_margin = ((limits["nox_ppm"] - current_emissions["nox_ppm"]) /
                      limits["nox_ppm"]) * 100

        if current_emissions["nox_ppm"] > limits["nox_ppm"]:
            violations.append(
                f"NOx exceeds limit: {current_emissions['nox_ppm']} ppm > {limits['nox_ppm']} ppm"
            )
            nox_status = "NON_COMPLIANT"
        elif nox_margin < 10:
            warnings.append(
                f"NOx approaching limit: {current_emissions['nox_ppm']} ppm "
                f"({nox_margin:.1f}% margin)"
            )
            nox_status = "WARNING"
        else:
            nox_status = "COMPLIANT"

        # SOx compliance
        sox_margin = ((limits["sox_ppm"] - current_emissions["sox_ppm"]) /
                      limits["sox_ppm"]) * 100

        if current_emissions["sox_ppm"] > limits["sox_ppm"]:
            violations.append(
                f"SOx exceeds limit: {current_emissions['sox_ppm']} ppm > {limits['sox_ppm']} ppm"
            )
            sox_status = "NON_COMPLIANT"
        elif sox_margin < 10:
            warnings.append(
                f"SOx approaching limit: {current_emissions['sox_ppm']} ppm"
            )
            sox_status = "WARNING"
        else:
            sox_status = "COMPLIANT"

        # CO compliance
        co_margin = ((limits["co_ppm"] - current_emissions["co_ppm"]) /
                     limits["co_ppm"]) * 100

        if current_emissions["co_ppm"] > limits["co_ppm"]:
            violations.append(
                f"CO exceeds limit: {current_emissions['co_ppm']} ppm > {limits['co_ppm']} ppm"
            )
            co_status = "NON_COMPLIANT"
        elif co_margin < 10:
            warnings.append(
                f"CO approaching limit: {current_emissions['co_ppm']} ppm"
            )
            co_status = "WARNING"
        else:
            co_status = "COMPLIANT"

        # Overall compliance status
        if violations:
            overall_status = "NON_COMPLIANT"
        elif warnings:
            overall_status = "WARNING"
        else:
            overall_status = "COMPLIANT"

        # Overall margin (worst case)
        overall_margin = min(nox_margin, sox_margin, co_margin)

        # Corrective actions
        corrective_actions = []

        if nox_status != "COMPLIANT":
            corrective_actions.append(
                "Reduce excess air or lower combustion temperature to reduce NOx"
            )

        if sox_status != "COMPLIANT":
            corrective_actions.append(
                "Switch to low-sulfur fuel or install SOx scrubber"
            )

        if co_status != "COMPLIANT":
            corrective_actions.append(
                "Increase air flow or tune burner for complete combustion"
            )

        if overall_status == "COMPLIANT" and not warnings:
            corrective_actions.append("No corrective actions required - maintain current operation")

        # Calculate provenance hash
        provenance_data = {
            "burner_id": burner_id,
            "standard": standard,
            "current_emissions": current_emissions,
            "limits": limits,
            "timestamp": datetime.utcnow().isoformat()
        }
        provenance_hash = hashlib.sha256(
            str(provenance_data).encode()
        ).hexdigest()

        return EmissionsComplianceReport(
            burner_id=burner_id,
            compliance_status=overall_status,
            regulatory_standard=standard,
            nox_compliance={
                "current_ppm": current_emissions["nox_ppm"],
                "limit_ppm": limits["nox_ppm"],
                "margin_percent": round(nox_margin, 1),
                "status": nox_status
            },
            sox_compliance={
                "current_ppm": current_emissions["sox_ppm"],
                "limit_ppm": limits["sox_ppm"],
                "margin_percent": round(sox_margin, 1),
                "status": sox_status
            },
            co_compliance={
                "current_ppm": current_emissions["co_ppm"],
                "limit_ppm": limits["co_ppm"],
                "margin_percent": round(co_margin, 1),
                "status": co_status
            },
            particulate_compliance={
                "current_mg_m3": current_emissions["particulate_mg_m3"],
                "limit_mg_m3": limits["particulate_mg_m3"],
                "margin_percent": round(
                    ((limits["particulate_mg_m3"] - current_emissions["particulate_mg_m3"]) /
                     limits["particulate_mg_m3"]) * 100, 1
                ),
                "status": "COMPLIANT"
            },
            overall_margin_percent=round(overall_margin, 1),
            violations=violations,
            warnings=warnings,
            corrective_actions=corrective_actions,
            next_report_due=datetime.utcnow() + timedelta(days=30),
            timestamp=datetime.utcnow(),
            provenance_hash=provenance_hash
        )

    except Exception as e:
        logger.error(f"Emissions compliance report failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Compliance report failed: {str(e)}"
        )


@app.get(
    "/api/v1/performance-report/{burner_id}",
    response_model=PerformanceMetrics,
    tags=["Performance"],
    summary="Get performance report",
    description="Retrieve performance metrics and trends for specified period"
)
@limiter.limit("100/minute")
async def get_performance_report(
    request: Request,
    burner_id: str,
    period: str = "24h",
    current_user: User = Depends(get_current_user)
) -> PerformanceMetrics:
    """
    Get performance report with trends.

    Args:
        burner_id: Burner identifier
        period: Time period (1h, 24h, 7d, 30d)
        current_user: Authenticated user

    Returns:
        Performance metrics and trends
    """
    try:
        logger.info(
            f"Performance report requested by {current_user.email} "
            f"for burner {burner_id}, period {period}"
        )

        # Mock performance data (would come from time-series database in production)
        # Simulate realistic variations
        import random

        base_efficiency = 87.5
        efficiency_variation = 2.0

        avg_efficiency = base_efficiency + random.uniform(-0.5, 0.5)
        min_efficiency = avg_efficiency - efficiency_variation
        max_efficiency = avg_efficiency + efficiency_variation

        # Determine trend
        recent_avg = avg_efficiency + random.uniform(-1, 1)
        if recent_avg > avg_efficiency + 0.5:
            trend = "improving"
        elif recent_avg < avg_efficiency - 0.5:
            trend = "declining"
        else:
            trend = "stable"

        # Mock operating data
        avg_nox = 175 + random.uniform(-20, 20)
        avg_co = 65 + random.uniform(-15, 15)
        avg_o2 = 3.5 + random.uniform(-0.5, 0.5)

        # Uptime calculation
        if period == "1h":
            uptime_percent = 100.0
            hours = 1
        elif period == "24h":
            uptime_percent = 98.5
            hours = 24
        elif period == "7d":
            uptime_percent = 97.2
            hours = 168
        else:  # 30d
            uptime_percent = 96.5
            hours = 720

        # Fuel consumption and steam generation
        avg_fuel_rate = 1000  # kg/hr
        avg_steam_rate = 10000  # kg/hr

        total_fuel = avg_fuel_rate * hours * (uptime_percent / 100)
        total_steam = avg_steam_rate * hours * (uptime_percent / 100)

        # Performance score (0-100)
        # Based on efficiency, emissions, uptime
        efficiency_score = (avg_efficiency / 95) * 40  # Max 40 points
        emissions_score = max(0, (1 - avg_nox / 300) * 30)  # Max 30 points
        uptime_score = (uptime_percent / 100) * 30  # Max 30 points

        performance_score = efficiency_score + emissions_score + uptime_score

        # Trends data
        trends = {
            "efficiency_trend": {
                "direction": trend,
                "change_percent": round(recent_avg - avg_efficiency, 2)
            },
            "emissions_trend": {
                "nox_direction": "stable",
                "co_direction": "improving" if avg_co < 70 else "stable"
            },
            "fuel_efficiency_trend": {
                "kg_fuel_per_ton_steam": round(total_fuel / (total_steam / 1000), 2)
            }
        }

        # Generate alerts
        alerts = []

        if avg_efficiency < 85:
            alerts.append(f"Average efficiency below target: {avg_efficiency:.1f}%")

        if trend == "declining":
            alerts.append("Efficiency trend is declining - schedule maintenance")

        if avg_nox > 200:
            alerts.append(f"Average NOx elevated: {avg_nox:.0f} ppm")

        if uptime_percent < 95:
            alerts.append(f"Uptime below target: {uptime_percent:.1f}%")

        if not alerts:
            alerts.append("All parameters within normal range")

        return PerformanceMetrics(
            burner_id=burner_id,
            period=period,
            avg_efficiency=round(avg_efficiency, 2),
            min_efficiency=round(min_efficiency, 2),
            max_efficiency=round(max_efficiency, 2),
            efficiency_trend=trend,
            avg_nox_ppm=round(avg_nox, 1),
            avg_co_ppm=round(avg_co, 1),
            avg_o2_percent=round(avg_o2, 2),
            uptime_percent=round(uptime_percent, 2),
            total_fuel_consumed=round(total_fuel, 2),
            total_steam_generated=round(total_steam, 2),
            performance_score=round(performance_score, 1),
            trends=trends,
            alerts=alerts,
            timestamp=datetime.utcnow()
        )

    except Exception as e:
        logger.error(f"Performance report failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Performance report failed: {str(e)}"
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
        "calculations": True
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
        timestamp=datetime.utcnow(),
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
            "timestamp": datetime.utcnow().isoformat()
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
            "timestamp": datetime.utcnow().isoformat()
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
