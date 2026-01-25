# -*- coding: utf-8 -*-
"""
Partner API Module for GreenLang

This module provides comprehensive API endpoints for partner management,
including authentication, partner CRUD operations, API key management,
usage tracking, and billing integration.

Features:
- API key-based authentication
- OAuth 2.0 for partner apps
- Rate limiting (1000 requests/hour per partner)
- Partner registration and management
- API key lifecycle management
- Usage statistics and billing
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import secrets
import hmac
import json
import time
from functools import wraps
import asyncio
import logging

# Third-party imports
from fastapi import FastAPI, HTTPException, Depends, Request, Header, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr, HttpUrl, validator
import jwt
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Boolean, JSON, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker, relationship
import redis
from greenlang.utilities.determinism import DeterministicClock

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
Base = declarative_base()
engine = create_engine("postgresql://localhost/greenlang_partners")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Redis for rate limiting
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# Constants
JWT_SECRET = "your-secret-key-change-in-production"
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours
API_KEY_LENGTH = 32


class PartnerTier(str, Enum):
    """Partner tier levels with different quotas and features"""
    FREE = "FREE"
    BASIC = "BASIC"
    PRO = "PRO"
    ENTERPRISE = "ENTERPRISE"


class PartnerStatus(str, Enum):
    """Partner account status"""
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    SUSPENDED = "SUSPENDED"
    DELETED = "DELETED"


class APIKeyStatus(str, Enum):
    """API key status"""
    ACTIVE = "ACTIVE"
    REVOKED = "REVOKED"
    EXPIRED = "EXPIRED"


# Tier configurations
TIER_CONFIGS = {
    PartnerTier.FREE: {
        "api_quota": 100,  # requests per hour
        "max_api_keys": 1,
        "webhook_support": False,
        "white_label": False,
        "analytics_retention_days": 7,
        "support_level": "community",
    },
    PartnerTier.BASIC: {
        "api_quota": 1000,
        "max_api_keys": 3,
        "webhook_support": True,
        "white_label": False,
        "analytics_retention_days": 30,
        "support_level": "email",
    },
    PartnerTier.PRO: {
        "api_quota": 10000,
        "max_api_keys": 10,
        "webhook_support": True,
        "white_label": True,
        "analytics_retention_days": 90,
        "support_level": "priority",
    },
    PartnerTier.ENTERPRISE: {
        "api_quota": 100000,
        "max_api_keys": 50,
        "webhook_support": True,
        "white_label": True,
        "analytics_retention_days": 365,
        "support_level": "dedicated",
    },
}


# Database Models
class PartnerModel(Base):
    """Partner database model"""
    __tablename__ = "partners"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    website = Column(String)
    tier = Column(String, default=PartnerTier.FREE)
    status = Column(String, default=PartnerStatus.PENDING)
    api_quota = Column(Integer, default=100)
    webhook_url = Column(String, nullable=True)
    webhook_secret = Column(String, nullable=True)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    # Relationships
    api_keys = relationship("APIKeyModel", back_populates="partner", cascade="all, delete-orphan")
    usage_records = relationship("UsageRecordModel", back_populates="partner", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index('idx_partner_email', 'email'),
        Index('idx_partner_status', 'status'),
    )


class APIKeyModel(Base):
    """API Key database model"""
    __tablename__ = "api_keys"

    id = Column(String, primary_key=True)
    partner_id = Column(String, ForeignKey("partners.id"), nullable=False)
    key_hash = Column(String, nullable=False, unique=True)
    key_prefix = Column(String, nullable=False)  # First 8 chars for display
    name = Column(String, nullable=False)
    status = Column(String, default=APIKeyStatus.ACTIVE)
    scopes = Column(JSON, default=list)  # ["read", "write", "admin"]
    last_used_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    partner = relationship("PartnerModel", back_populates="api_keys")

    # Indexes
    __table_args__ = (
        Index('idx_api_key_hash', 'key_hash'),
        Index('idx_api_key_partner', 'partner_id'),
    )


class UsageRecordModel(Base):
    """Usage tracking model"""
    __tablename__ = "usage_records"

    id = Column(String, primary_key=True)
    partner_id = Column(String, ForeignKey("partners.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    endpoint = Column(String, nullable=False)
    method = Column(String, nullable=False)
    status_code = Column(Integer, nullable=False)
    response_time_ms = Column(Integer, nullable=False)
    request_size_bytes = Column(Integer, default=0)
    response_size_bytes = Column(Integer, default=0)
    metadata = Column(JSON, default=dict)

    # Relationships
    partner = relationship("PartnerModel", back_populates="usage_records")

    # Indexes
    __table_args__ = (
        Index('idx_usage_partner_timestamp', 'partner_id', 'timestamp'),
        Index('idx_usage_timestamp', 'timestamp'),
    )


class BillingRecordModel(Base):
    """Billing record model"""
    __tablename__ = "billing_records"

    id = Column(String, primary_key=True)
    partner_id = Column(String, ForeignKey("partners.id"), nullable=False)
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    total_requests = Column(Integer, default=0)
    total_cost = Column(Integer, default=0)  # in cents
    currency = Column(String, default="USD")
    status = Column(String, default="PENDING")  # PENDING, PAID, FAILED
    invoice_url = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    paid_at = Column(DateTime, nullable=True)


# Pydantic Models for API
class PartnerBase(BaseModel):
    """Base partner schema"""
    name: str
    email: EmailStr
    website: Optional[HttpUrl] = None


class PartnerCreate(PartnerBase):
    """Schema for creating a partner"""
    password: str
    tier: PartnerTier = PartnerTier.FREE

    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        return v


class PartnerUpdate(BaseModel):
    """Schema for updating a partner"""
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    website: Optional[HttpUrl] = None
    webhook_url: Optional[HttpUrl] = None


class PartnerResponse(PartnerBase):
    """Schema for partner response"""
    id: str
    tier: PartnerTier
    status: PartnerStatus
    api_quota: int
    webhook_url: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class APIKeyCreate(BaseModel):
    """Schema for creating an API key"""
    name: str
    scopes: List[str] = ["read", "write"]
    expires_in_days: Optional[int] = None


class APIKeyResponse(BaseModel):
    """Schema for API key response"""
    id: str
    key_prefix: str
    name: str
    status: APIKeyStatus
    scopes: List[str]
    last_used_at: Optional[datetime]
    expires_at: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True


class APIKeyWithSecret(APIKeyResponse):
    """Schema for API key response with secret (only on creation)"""
    key: str


class LoginRequest(BaseModel):
    """Schema for login request"""
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """Schema for token response"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class UsageStats(BaseModel):
    """Schema for usage statistics"""
    partner_id: str
    period_start: datetime
    period_end: datetime
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time_ms: float
    total_data_transferred_mb: float
    requests_by_endpoint: Dict[str, int]
    requests_by_day: Dict[str, int]


class BillingInfo(BaseModel):
    """Schema for billing information"""
    partner_id: str
    current_period_start: datetime
    current_period_end: datetime
    current_usage: int
    quota_limit: int
    estimated_cost: float
    currency: str
    invoices: List[Dict[str, Any]]


# Dataclasses for internal use
@dataclass
class Partner:
    """Partner dataclass"""
    id: str
    name: str
    email: str
    website: str
    tier: PartnerTier
    status: PartnerStatus
    api_quota: int
    webhook_url: Optional[str]
    created_at: datetime
    updated_at: datetime


@dataclass
class APIKey:
    """API Key dataclass"""
    id: str
    partner_id: str
    key_hash: str
    key_prefix: str
    name: str
    status: APIKeyStatus
    scopes: List[str]
    last_used_at: Optional[datetime]
    expires_at: Optional[datetime]
    created_at: datetime


# Utility Functions
def generate_id(prefix: str = "partner") -> str:
    """Generate a unique ID"""
    random_part = secrets.token_hex(8)
    return f"{prefix}_{random_part}"


def generate_api_key() -> str:
    """Generate a secure API key"""
    return f"gl_{secrets.token_urlsafe(API_KEY_LENGTH)}"


def hash_password(password: str) -> str:
    """Hash a password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash"""
    return hash_password(password) == password_hash


def hash_api_key(api_key: str) -> str:
    """Hash an API key for storage"""
    return hashlib.sha256(api_key.encode()).hexdigest()


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = DeterministicClock.utcnow() + expires_delta
    else:
        expire = DeterministicClock.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> dict:
    """Decode and validate a JWT access token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


# Rate Limiting
class RateLimiter:
    """Rate limiter using Redis"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    def check_rate_limit(self, partner_id: str, quota: int, window_seconds: int = 3600) -> Tuple[bool, int]:
        """
        Check if partner is within rate limit

        Returns:
            Tuple of (is_allowed, remaining_requests)
        """
        key = f"rate_limit:{partner_id}:{int(time.time() / window_seconds)}"

        try:
            current = self.redis.get(key)
            if current is None:
                # First request in this window
                self.redis.setex(key, window_seconds, 1)
                return True, quota - 1

            current = int(current)
            if current >= quota:
                return False, 0

            self.redis.incr(key)
            return True, quota - current - 1
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            # Fail open - allow request if Redis is down
            return True, quota

    def reset_rate_limit(self, partner_id: str):
        """Reset rate limit for a partner"""
        pattern = f"rate_limit:{partner_id}:*"
        keys = self.redis.keys(pattern)
        if keys:
            self.redis.delete(*keys)


rate_limiter = RateLimiter(redis_client)


# Dependency injection
def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_current_partner_from_token(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
) -> PartnerModel:
    """Get current partner from JWT token"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")

    token = authorization.split(" ")[1]
    payload = decode_access_token(token)
    partner_id = payload.get("sub")

    if not partner_id:
        raise HTTPException(status_code=401, detail="Invalid token")

    partner = db.query(PartnerModel).filter(PartnerModel.id == partner_id).first()
    if not partner:
        raise HTTPException(status_code=401, detail="Partner not found")

    if partner.status != PartnerStatus.ACTIVE:
        raise HTTPException(status_code=403, detail="Partner account is not active")

    return partner


async def get_current_partner_from_api_key(
    x_api_key: Optional[str] = Header(None),
    db: Session = Depends(get_db)
) -> PartnerModel:
    """Get current partner from API key"""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing API key")

    key_hash = hash_api_key(x_api_key)
    api_key = db.query(APIKeyModel).filter(APIKeyModel.key_hash == key_hash).first()

    if not api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if api_key.status != APIKeyStatus.ACTIVE:
        raise HTTPException(status_code=401, detail="API key is not active")

    if api_key.expires_at and api_key.expires_at < DeterministicClock.utcnow():
        api_key.status = APIKeyStatus.EXPIRED
        db.commit()
        raise HTTPException(status_code=401, detail="API key has expired")

    partner = db.query(PartnerModel).filter(PartnerModel.id == api_key.partner_id).first()
    if not partner:
        raise HTTPException(status_code=401, detail="Partner not found")

    if partner.status != PartnerStatus.ACTIVE:
        raise HTTPException(status_code=403, detail="Partner account is not active")

    # Update last used timestamp
    api_key.last_used_at = DeterministicClock.utcnow()
    db.commit()

    return partner


def check_rate_limit_middleware(partner: PartnerModel):
    """Middleware to check rate limits"""
    allowed, remaining = rate_limiter.check_rate_limit(partner.id, partner.api_quota)

    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Quota: {partner.api_quota} requests/hour",
            headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": str(3600)}
        )

    return {"remaining": remaining}


# FastAPI app
app = FastAPI(title="GreenLang Partner API", version="1.0.0")


# Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000

    logger.info(
        f"{request.method} {request.url.path} - {response.status_code} - {process_time:.2f}ms"
    )

    return response


# Authentication Endpoints
@app.post("/api/partners/register", response_model=PartnerResponse, status_code=status.HTTP_201_CREATED)
async def register_partner(partner_data: PartnerCreate, db: Session = Depends(get_db)):
    """
    Register a new partner

    Creates a new partner account with the specified tier.
    Default tier is FREE with 100 requests/hour quota.
    """
    # Check if email already exists
    existing = db.query(PartnerModel).filter(PartnerModel.email == partner_data.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Get tier configuration
    tier_config = TIER_CONFIGS[partner_data.tier]

    # Create partner
    partner = PartnerModel(
        id=generate_id("partner"),
        name=partner_data.name,
        email=partner_data.email,
        website=str(partner_data.website) if partner_data.website else None,
        tier=partner_data.tier,
        status=PartnerStatus.PENDING,
        api_quota=tier_config["api_quota"],
        password_hash=hash_password(partner_data.password),
        webhook_secret=secrets.token_hex(32)
    )

    db.add(partner)
    db.commit()
    db.refresh(partner)

    logger.info(f"New partner registered: {partner.id} ({partner.email})")

    return partner


@app.post("/api/partners/login", response_model=TokenResponse)
async def login(credentials: LoginRequest, db: Session = Depends(get_db)):
    """
    Login to get access token

    Returns a JWT token that can be used for authentication.
    """
    partner = db.query(PartnerModel).filter(PartnerModel.email == credentials.email).first()

    if not partner or not verify_password(credentials.password, partner.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if partner.status != PartnerStatus.ACTIVE:
        raise HTTPException(status_code=403, detail="Account is not active")

    # Update last login
    partner.last_login = DeterministicClock.utcnow()
    db.commit()

    # Create access token
    access_token = create_access_token(data={"sub": partner.id, "email": partner.email})

    return TokenResponse(
        access_token=access_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


# Partner Management Endpoints
@app.get("/api/partners/me", response_model=PartnerResponse)
async def get_current_partner(partner: PartnerModel = Depends(get_current_partner_from_token)):
    """Get current partner details"""
    return partner


@app.get("/api/partners/{partner_id}", response_model=PartnerResponse)
async def get_partner(
    partner_id: str,
    current_partner: PartnerModel = Depends(get_current_partner_from_token),
    db: Session = Depends(get_db)
):
    """
    Get partner details by ID

    Partners can only access their own details unless they have admin privileges.
    """
    if partner_id != current_partner.id:
        raise HTTPException(status_code=403, detail="Access denied")

    return current_partner


@app.put("/api/partners/{partner_id}", response_model=PartnerResponse)
async def update_partner(
    partner_id: str,
    partner_update: PartnerUpdate,
    current_partner: PartnerModel = Depends(get_current_partner_from_token),
    db: Session = Depends(get_db)
):
    """
    Update partner details

    Partners can update their name, email, website, and webhook URL.
    """
    if partner_id != current_partner.id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Update fields
    if partner_update.name:
        current_partner.name = partner_update.name
    if partner_update.email:
        # Check if email is already used
        existing = db.query(PartnerModel).filter(
            PartnerModel.email == partner_update.email,
            PartnerModel.id != partner_id
        ).first()
        if existing:
            raise HTTPException(status_code=400, detail="Email already in use")
        current_partner.email = partner_update.email
    if partner_update.website:
        current_partner.website = str(partner_update.website)
    if partner_update.webhook_url:
        current_partner.webhook_url = str(partner_update.webhook_url)

    current_partner.updated_at = DeterministicClock.utcnow()
    db.commit()
    db.refresh(current_partner)

    return current_partner


@app.delete("/api/partners/{partner_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_partner(
    partner_id: str,
    current_partner: PartnerModel = Depends(get_current_partner_from_token),
    db: Session = Depends(get_db)
):
    """
    Delete partner account

    This will revoke all API keys and mark the account as deleted.
    """
    if partner_id != current_partner.id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Revoke all API keys
    db.query(APIKeyModel).filter(APIKeyModel.partner_id == partner_id).update(
        {"status": APIKeyStatus.REVOKED}
    )

    # Mark as deleted
    current_partner.status = PartnerStatus.DELETED
    db.commit()

    return None


# API Key Management Endpoints
@app.post("/api/partners/{partner_id}/api-keys", response_model=APIKeyWithSecret, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    partner_id: str,
    key_data: APIKeyCreate,
    current_partner: PartnerModel = Depends(get_current_partner_from_token),
    db: Session = Depends(get_db)
):
    """
    Create a new API key

    The API key is only shown once during creation. Make sure to save it securely.
    """
    if partner_id != current_partner.id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Check tier limits
    tier_config = TIER_CONFIGS[current_partner.tier]
    existing_keys = db.query(APIKeyModel).filter(
        APIKeyModel.partner_id == partner_id,
        APIKeyModel.status == APIKeyStatus.ACTIVE
    ).count()

    if existing_keys >= tier_config["max_api_keys"]:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum API keys ({tier_config['max_api_keys']}) reached for {current_partner.tier} tier"
        )

    # Generate API key
    api_key = generate_api_key()
    key_hash = hash_api_key(api_key)
    key_prefix = api_key[:12]  # First 12 chars for display

    # Set expiration
    expires_at = None
    if key_data.expires_in_days:
        expires_at = DeterministicClock.utcnow() + timedelta(days=key_data.expires_in_days)

    # Create API key record
    api_key_model = APIKeyModel(
        id=generate_id("key"),
        partner_id=partner_id,
        key_hash=key_hash,
        key_prefix=key_prefix,
        name=key_data.name,
        scopes=key_data.scopes,
        expires_at=expires_at
    )

    db.add(api_key_model)
    db.commit()
    db.refresh(api_key_model)

    logger.info(f"API key created for partner {partner_id}: {api_key_model.id}")

    # Return key with secret (only time it's shown)
    return APIKeyWithSecret(
        id=api_key_model.id,
        key_prefix=api_key_model.key_prefix,
        name=api_key_model.name,
        status=api_key_model.status,
        scopes=api_key_model.scopes,
        last_used_at=api_key_model.last_used_at,
        expires_at=api_key_model.expires_at,
        created_at=api_key_model.created_at,
        key=api_key  # Full key only on creation
    )


@app.get("/api/partners/{partner_id}/api-keys", response_model=List[APIKeyResponse])
async def list_api_keys(
    partner_id: str,
    current_partner: PartnerModel = Depends(get_current_partner_from_token),
    db: Session = Depends(get_db)
):
    """
    List all API keys for a partner

    Returns a list of API keys with metadata (not the actual keys).
    """
    if partner_id != current_partner.id:
        raise HTTPException(status_code=403, detail="Access denied")

    api_keys = db.query(APIKeyModel).filter(
        APIKeyModel.partner_id == partner_id
    ).order_by(APIKeyModel.created_at.desc()).all()

    return api_keys


@app.delete("/api/partners/{partner_id}/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_api_key(
    partner_id: str,
    key_id: str,
    current_partner: PartnerModel = Depends(get_current_partner_from_token),
    db: Session = Depends(get_db)
):
    """
    Revoke an API key

    Revoked keys cannot be reactivated.
    """
    if partner_id != current_partner.id:
        raise HTTPException(status_code=403, detail="Access denied")

    api_key = db.query(APIKeyModel).filter(
        APIKeyModel.id == key_id,
        APIKeyModel.partner_id == partner_id
    ).first()

    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")

    api_key.status = APIKeyStatus.REVOKED
    db.commit()

    logger.info(f"API key revoked: {key_id}")

    return None


# Usage and Billing Endpoints
@app.get("/api/partners/{partner_id}/usage", response_model=UsageStats)
async def get_usage_statistics(
    partner_id: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_partner: PartnerModel = Depends(get_current_partner_from_token),
    db: Session = Depends(get_db)
):
    """
    Get usage statistics for a partner

    Returns detailed usage metrics for the specified time period.
    """
    if partner_id != current_partner.id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Default to last 30 days
    if not end_date:
        end_date = DeterministicClock.utcnow()
    if not start_date:
        start_date = end_date - timedelta(days=30)

    # Query usage records
    usage_records = db.query(UsageRecordModel).filter(
        UsageRecordModel.partner_id == partner_id,
        UsageRecordModel.timestamp >= start_date,
        UsageRecordModel.timestamp <= end_date
    ).all()

    # Calculate statistics
    total_requests = len(usage_records)
    successful_requests = sum(1 for r in usage_records if 200 <= r.status_code < 300)
    failed_requests = total_requests - successful_requests

    avg_response_time = sum(r.response_time_ms for r in usage_records) / total_requests if total_requests > 0 else 0

    total_data_mb = sum(
        (r.request_size_bytes + r.response_size_bytes) for r in usage_records
    ) / (1024 * 1024)

    # Group by endpoint
    requests_by_endpoint = {}
    for record in usage_records:
        endpoint = record.endpoint
        requests_by_endpoint[endpoint] = requests_by_endpoint.get(endpoint, 0) + 1

    # Group by day
    requests_by_day = {}
    for record in usage_records:
        day = record.timestamp.date().isoformat()
        requests_by_day[day] = requests_by_day.get(day, 0) + 1

    return UsageStats(
        partner_id=partner_id,
        period_start=start_date,
        period_end=end_date,
        total_requests=total_requests,
        successful_requests=successful_requests,
        failed_requests=failed_requests,
        average_response_time_ms=avg_response_time,
        total_data_transferred_mb=total_data_mb,
        requests_by_endpoint=requests_by_endpoint,
        requests_by_day=requests_by_day
    )


@app.get("/api/partners/{partner_id}/billing", response_model=BillingInfo)
async def get_billing_information(
    partner_id: str,
    current_partner: PartnerModel = Depends(get_current_partner_from_token),
    db: Session = Depends(get_db)
):
    """
    Get billing information for a partner

    Returns current usage, quota, and invoice history.
    """
    if partner_id != current_partner.id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Current billing period (month)
    now = DeterministicClock.utcnow()
    period_start = datetime(now.year, now.month, 1)
    if now.month == 12:
        period_end = datetime(now.year + 1, 1, 1)
    else:
        period_end = datetime(now.year, now.month + 1, 1)

    # Get current usage
    current_usage = db.query(UsageRecordModel).filter(
        UsageRecordModel.partner_id == partner_id,
        UsageRecordModel.timestamp >= period_start,
        UsageRecordModel.timestamp < period_end
    ).count()

    # Get invoices
    invoices = db.query(BillingRecordModel).filter(
        BillingRecordModel.partner_id == partner_id
    ).order_by(BillingRecordModel.created_at.desc()).limit(12).all()

    invoice_list = [
        {
            "id": inv.id,
            "period_start": inv.period_start.isoformat(),
            "period_end": inv.period_end.isoformat(),
            "total_requests": inv.total_requests,
            "total_cost": inv.total_cost / 100,  # Convert cents to dollars
            "status": inv.status,
            "invoice_url": inv.invoice_url
        }
        for inv in invoices
    ]

    # Calculate estimated cost (example: $0.01 per request over quota)
    overage = max(0, current_usage - current_partner.api_quota * 24 * 30)  # Monthly quota
    estimated_cost = overage * 0.01

    return BillingInfo(
        partner_id=partner_id,
        current_period_start=period_start,
        current_period_end=period_end,
        current_usage=current_usage,
        quota_limit=current_partner.api_quota * 24 * 30,
        estimated_cost=estimated_cost,
        currency="USD",
        invoices=invoice_list
    )


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": DeterministicClock.utcnow().isoformat()}


# Create tables
def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    import uvicorn
    init_db()
    uvicorn.run(app, host="0.0.0.0", port=8000)
