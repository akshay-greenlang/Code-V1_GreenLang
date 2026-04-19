"""
GreenLang Agent Registry API
FastAPI-based REST API for agent registration, versioning, and certification
"""

import hashlib
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import asyncpg
from asyncpg.pool import Pool
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models
# ============================================================================

class AgentCreate(BaseModel):
    """Request model for creating a new agent"""
    name: str = Field(..., min_length=1, max_length=255)
    namespace: str = Field(default="default", max_length=255)
    description: Optional[str] = None
    author: Optional[str] = None
    repository_url: Optional[str] = None
    homepage_url: Optional[str] = None
    spec_hash: str = Field(..., min_length=64, max_length=64)

    @validator('name')
    def validate_name(cls, v):
        if not v.replace('-', '').replace('_', '').isalnum():
            raise ValueError('Name must be alphanumeric with hyphens/underscores')
        return v.lower()


class AgentVersionCreate(BaseModel):
    """Request model for publishing a new agent version"""
    version: str = Field(..., pattern=r'^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?$')
    pack_path: str
    pack_hash: str = Field(..., min_length=64, max_length=64)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    capabilities: List[str] = Field(default_factory=list)
    dependencies: List[Dict[str, str]] = Field(default_factory=list)
    size_bytes: Optional[int] = None
    published_by: Optional[str] = None


class CertificationCreate(BaseModel):
    """Request model for submitting agent certification"""
    version: str
    dimension: str = Field(..., description="Certification dimension: security, performance, reliability, etc.")
    status: str = Field(..., pattern='^(passed|failed|pending)$')
    score: Optional[float] = Field(None, ge=0, le=100)
    evidence: Dict[str, Any] = Field(default_factory=dict)
    certified_by: str


class AgentResponse(BaseModel):
    """Response model for agent data"""
    id: str
    name: str
    namespace: str
    description: Optional[str]
    author: Optional[str]
    repository_url: Optional[str]
    homepage_url: Optional[str]
    spec_hash: str
    status: str
    created_at: datetime
    updated_at: datetime


class AgentVersionResponse(BaseModel):
    """Response model for agent version data"""
    id: str
    agent_id: str
    version: str
    pack_path: str
    pack_hash: str
    metadata: Dict[str, Any]
    capabilities: List[str]
    dependencies: List[Dict[str, str]]
    size_bytes: Optional[int]
    status: str
    published_by: Optional[str]
    published_at: datetime


class CertificationResponse(BaseModel):
    """Response model for certification data"""
    id: str
    agent_id: str
    version: str
    dimension: str
    status: str
    score: Optional[float]
    evidence: Dict[str, Any]
    certified_by: str
    certification_date: datetime
    expiry_date: Optional[datetime]


class AgentListResponse(BaseModel):
    """Response model for agent listing"""
    agents: List[AgentResponse]
    total: int
    page: int
    page_size: int


# ============================================================================
# Database Connection Pool
# ============================================================================

class Database:
    """Database connection pool manager"""
    pool: Optional[Pool] = None

    @classmethod
    async def connect(cls):
        """Initialize connection pool"""
        try:
            cls.pool = await asyncpg.create_pool(
                host=os.getenv("DB_HOST", "localhost"),
                port=int(os.getenv("DB_PORT", "5432")),
                user=os.getenv("DB_USER", "greenlang"),
                password=os.getenv("DB_PASSWORD", "greenlang"),
                database=os.getenv("DB_NAME", "greenlang_registry"),
                min_size=5,
                max_size=20,
                command_timeout=60,
            )
            logger.info("Database connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    @classmethod
    async def disconnect(cls):
        """Close connection pool"""
        if cls.pool:
            await cls.pool.close()
            logger.info("Database connection pool closed")


async def get_db() -> Pool:
    """Dependency for database access"""
    if not Database.pool:
        raise HTTPException(status_code=500, detail="Database not connected")
    return Database.pool


# ============================================================================
# FastAPI Application
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    # Startup
    await Database.connect()
    yield
    # Shutdown
    await Database.disconnect()


app = FastAPI(
    title="GreenLang Agent Registry API",
    description="API for agent registration, versioning, and certification",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware - SECURITY: Configure specific origins in production
# Set CORS_ALLOWED_ORIGINS environment variable (comma-separated list)
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
# API Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/api/v1/ready")
async def ready_check(db: Pool = Depends(get_db)):
    """Readiness check endpoint"""
    try:
        async with db.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return {"status": "ready", "database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database not ready: {str(e)}")


# ----------------------------------------------------------------------------
# Agent Endpoints
# ----------------------------------------------------------------------------

@app.post("/api/v1/agents", response_model=AgentResponse, status_code=201)
async def create_agent(agent: AgentCreate, db: Pool = Depends(get_db)):
    """Register a new agent"""
    async with db.acquire() as conn:
        # Check if agent already exists
        existing = await conn.fetchrow(
            "SELECT id FROM agents WHERE namespace = $1 AND name = $2",
            agent.namespace, agent.name
        )
        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"Agent {agent.namespace}/{agent.name} already exists"
            )

        # Insert new agent
        row = await conn.fetchrow("""
            INSERT INTO agents (name, namespace, description, author, repository_url, homepage_url, spec_hash)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING id, name, namespace, description, author, repository_url, homepage_url, spec_hash, status, created_at, updated_at
        """, agent.name, agent.namespace, agent.description, agent.author,
            agent.repository_url, agent.homepage_url, agent.spec_hash)

        logger.info(f"Created agent: {agent.namespace}/{agent.name} (id={row['id']})")
        return dict(row)


@app.get("/api/v1/agents", response_model=AgentListResponse)
async def list_agents(
    namespace: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Pool = Depends(get_db)
):
    """List agents with filtering and pagination"""
    async with db.acquire() as conn:
        # Build query
        conditions = []
        params = []
        param_idx = 1

        if namespace:
            conditions.append(f"namespace = ${param_idx}")
            params.append(namespace)
            param_idx += 1

        if status:
            conditions.append(f"status = ${param_idx}")
            params.append(status)
            param_idx += 1

        if search:
            conditions.append(f"(name ILIKE ${param_idx} OR description ILIKE ${param_idx})")
            params.append(f"%{search}%")
            param_idx += 1

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        # Get total count
        total = await conn.fetchval(f"SELECT COUNT(*) FROM agents {where_clause}", *params)

        # Get paginated results
        offset = (page - 1) * page_size
        params.extend([page_size, offset])

        rows = await conn.fetch(f"""
            SELECT id, name, namespace, description, author, repository_url, homepage_url, spec_hash, status, created_at, updated_at
            FROM agents
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """, *params)

        agents = [dict(row) for row in rows]
        return {
            "agents": agents,
            "total": total,
            "page": page,
            "page_size": page_size
        }


@app.get("/api/v1/agents/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: str, db: Pool = Depends(get_db)):
    """Get agent details by ID"""
    async with db.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT id, name, namespace, description, author, repository_url, homepage_url, spec_hash, status, created_at, updated_at
            FROM agents
            WHERE id = $1
        """, uuid.UUID(agent_id))

        if not row:
            raise HTTPException(status_code=404, detail="Agent not found")

        return dict(row)


# ----------------------------------------------------------------------------
# Agent Version Endpoints
# ----------------------------------------------------------------------------

@app.post("/api/v1/agents/{agent_id}/versions", response_model=AgentVersionResponse, status_code=201)
async def publish_version(agent_id: str, version: AgentVersionCreate, db: Pool = Depends(get_db)):
    """Publish a new version of an agent"""
    async with db.acquire() as conn:
        # Check if agent exists
        agent = await conn.fetchrow("SELECT id, name FROM agents WHERE id = $1", uuid.UUID(agent_id))
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        # Check if version already exists
        existing = await conn.fetchrow(
            "SELECT id FROM agent_versions WHERE agent_id = $1 AND version = $2",
            uuid.UUID(agent_id), version.version
        )
        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"Version {version.version} already exists for this agent"
            )

        # Insert new version
        row = await conn.fetchrow("""
            INSERT INTO agent_versions (agent_id, version, pack_path, pack_hash, metadata, capabilities, dependencies, size_bytes, published_by)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING id, agent_id, version, pack_path, pack_hash, metadata, capabilities, dependencies, size_bytes, status, published_by, published_at
        """, uuid.UUID(agent_id), version.version, version.pack_path, version.pack_hash,
            version.metadata, version.capabilities, version.dependencies, version.size_bytes, version.published_by)

        logger.info(f"Published version {version.version} for agent {agent['name']} (id={row['id']})")
        return dict(row)


@app.get("/api/v1/agents/{agent_id}/versions", response_model=List[AgentVersionResponse])
async def list_versions(agent_id: str, db: Pool = Depends(get_db)):
    """List all versions of an agent"""
    async with db.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, agent_id, version, pack_path, pack_hash, metadata, capabilities, dependencies, size_bytes, status, published_by, published_at
            FROM agent_versions
            WHERE agent_id = $1
            ORDER BY published_at DESC
        """, uuid.UUID(agent_id))

        return [dict(row) for row in rows]


@app.get("/api/v1/agents/{agent_id}/versions/{version}", response_model=AgentVersionResponse)
async def get_version(agent_id: str, version: str, db: Pool = Depends(get_db)):
    """Get specific version details"""
    async with db.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT id, agent_id, version, pack_path, pack_hash, metadata, capabilities, dependencies, size_bytes, status, published_by, published_at
            FROM agent_versions
            WHERE agent_id = $1 AND version = $2
        """, uuid.UUID(agent_id), version)

        if not row:
            raise HTTPException(status_code=404, detail="Version not found")

        return dict(row)


# ----------------------------------------------------------------------------
# Certification Endpoints
# ----------------------------------------------------------------------------

@app.post("/api/v1/agents/{agent_id}/certify", response_model=CertificationResponse, status_code=201)
async def submit_certification(agent_id: str, cert: CertificationCreate, db: Pool = Depends(get_db)):
    """Submit certification for an agent version"""
    async with db.acquire() as conn:
        # Verify agent and version exist
        version_exists = await conn.fetchrow(
            "SELECT id FROM agent_versions WHERE agent_id = $1 AND version = $2",
            uuid.UUID(agent_id), cert.version
        )
        if not version_exists:
            raise HTTPException(status_code=404, detail="Agent version not found")

        # Upsert certification
        row = await conn.fetchrow("""
            INSERT INTO agent_certifications (agent_id, version, dimension, status, score, evidence, certified_by)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (agent_id, version, dimension)
            DO UPDATE SET status = EXCLUDED.status, score = EXCLUDED.score, evidence = EXCLUDED.evidence, certified_by = EXCLUDED.certified_by, certification_date = NOW()
            RETURNING id, agent_id, version, dimension, status, score, evidence, certified_by, certification_date, expiry_date
        """, uuid.UUID(agent_id), cert.version, cert.dimension, cert.status, cert.score, cert.evidence, cert.certified_by)

        logger.info(f"Submitted {cert.dimension} certification for agent {agent_id} version {cert.version}: {cert.status}")
        return dict(row)


@app.get("/api/v1/agents/{agent_id}/certifications", response_model=List[CertificationResponse])
async def list_certifications(
    agent_id: str,
    version: Optional[str] = Query(None),
    dimension: Optional[str] = Query(None),
    db: Pool = Depends(get_db)
):
    """List certifications for an agent"""
    async with db.acquire() as conn:
        conditions = ["agent_id = $1"]
        params = [uuid.UUID(agent_id)]
        param_idx = 2

        if version:
            conditions.append(f"version = ${param_idx}")
            params.append(version)
            param_idx += 1

        if dimension:
            conditions.append(f"dimension = ${param_idx}")
            params.append(dimension)
            param_idx += 1

        where_clause = " AND ".join(conditions)

        rows = await conn.fetch(f"""
            SELECT id, agent_id, version, dimension, status, score, evidence, certified_by, certification_date, expiry_date
            FROM agent_certifications
            WHERE {where_clause}
            ORDER BY certification_date DESC
        """, *params)

        return [dict(row) for row in rows]


# ----------------------------------------------------------------------------
# Download Tracking
# ----------------------------------------------------------------------------

@app.post("/api/v1/agents/{agent_id}/versions/{version}/download")
async def track_download(
    agent_id: str,
    version: str,
    downloaded_by: Optional[str] = None,
    db: Pool = Depends(get_db)
):
    """Track agent download for analytics"""
    async with db.acquire() as conn:
        await conn.execute("""
            INSERT INTO agent_downloads (agent_id, version, downloaded_by)
            VALUES ($1, $2, $3)
        """, uuid.UUID(agent_id), version, downloaded_by)

        logger.info(f"Tracked download of agent {agent_id} version {version}")
        return {"status": "recorded"}


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
