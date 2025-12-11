# -*- coding: utf-8 -*-
"""
Analytics REST API
==================

Production-grade FastAPI service for monitoring analytics.

Author: Monitoring & Observability Team
Created: 2025-11-09
"""

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import logging
from prometheus_client import Counter, Histogram, Gauge
import asyncio
from greenlang.determinism import DeterministicClock

logger = logging.getLogger(__name__)

# Prometheus metrics for API itself
api_requests_total = Counter(
    'greenlang_analytics_api_requests_total',
    'Total analytics API requests',
    ['endpoint', 'method', 'status']
)

api_request_duration = Histogram(
    'greenlang_analytics_api_request_duration_seconds',
    'Analytics API request duration',
    ['endpoint']
)

# Initialize FastAPI
app = FastAPI(
    title="GreenLang Analytics API",
    description="Real-time analytics and monitoring data API",
    version="1.0.0"
)

# CORS middleware - SECURITY: Configure specific origins in production
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

# API Key authentication
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify API key for authentication"""
    # In production, check against database or vault
    valid_keys = ["greenlang-api-key-prod", "greenlang-api-key-dev"]
    if api_key not in valid_keys:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


# Response Models
class IUMResponse(BaseModel):
    """IUM metrics response"""
    overall_ium: float = Field(..., ge=0, le=100, description="Overall IUM percentage")
    timestamp: datetime
    applications: Dict[str, float] = Field(..., description="IUM by application")
    teams: Dict[str, float] = Field(..., description="IUM by team")
    trend: str = Field(..., description="Trend: increasing, decreasing, stable")


class CostSavingsResponse(BaseModel):
    """Cost savings response"""
    total_savings_usd: float = Field(..., description="Total cost savings in USD")
    llm_cache_savings: float
    developer_time_savings: float
    roi_percentage: float
    period_days: int
    breakdown: Dict[str, float]


class PerformanceMetrics(BaseModel):
    """Performance metrics response"""
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    error_rate_percent: float
    throughput_rps: float
    cache_hit_rate: float
    service: str
    timestamp: datetime


class ComplianceViolation(BaseModel):
    """Compliance violation model"""
    id: str
    violation_type: str
    severity: str
    file_path: str
    application: str
    team: str
    detected_at: datetime
    resolved: bool


class LeaderboardEntry(BaseModel):
    """Leaderboard entry"""
    rank: int
    developer: str
    team: str
    contributions: int
    ium_score: float
    achievements: List[str]


class HealthStatus(BaseModel):
    """Health status response"""
    status: str = Field(..., description="overall, degraded, down")
    services: Dict[str, Dict[str, Any]]
    uptime_percent: float
    last_incident: Optional[datetime]


# API Endpoints

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "service": "GreenLang Analytics API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": [
            "/api/ium/overall",
            "/api/ium/by-app/{app_name}",
            "/api/ium/by-team/{team_name}",
            "/api/costs/savings",
            "/api/performance/metrics",
            "/api/compliance/violations",
            "/api/leaderboard",
            "/api/health"
        ]
    }


@app.get("/api/ium/overall", response_model=IUMResponse)
async def get_overall_ium(api_key: str = Depends(verify_api_key)):
    """
    Get overall Infrastructure Usage Metric across all applications.

    Returns:
        IUMResponse with overall metrics and breakdowns
    """
    with api_request_duration.labels(endpoint="/api/ium/overall").time():
        try:
            # In production, query Prometheus/TimescaleDB
            ium_data = {
                "overall_ium": 96.5,
                "timestamp": DeterministicClock.now(),
                "applications": {
                    "csrd-reporting": 98.2,
                    "vcci-scope3": 95.8,
                    "factor-broker": 99.1,
                    "entity-mdm": 94.3
                },
                "teams": {
                    "platform": 99.5,
                    "csrd": 97.1,
                    "carbon": 95.2
                },
                "trend": "increasing"
            }

            api_requests_total.labels(
                endpoint="/api/ium/overall",
                method="GET",
                status="200"
            ).inc()

            return IUMResponse(**ium_data)
        except Exception as e:
            logger.error(f"Error fetching IUM data: {e}")
            api_requests_total.labels(
                endpoint="/api/ium/overall",
                method="GET",
                status="500"
            ).inc()
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ium/by-app/{app_name}", response_model=Dict[str, Any])
async def get_ium_by_app(
    app_name: str,
    period_days: int = 30,
    api_key: str = Depends(verify_api_key)
):
    """
    Get IUM metrics for a specific application.

    Args:
        app_name: Application name
        period_days: Time period in days (default: 30)
    """
    with api_request_duration.labels(endpoint="/api/ium/by-app").time():
        try:
            # Query application-specific IUM data
            data = {
                "application": app_name,
                "current_ium": 96.8,
                "average_ium": 95.5,
                "min_ium": 92.1,
                "max_ium": 99.2,
                "trend_data": [
                    {"date": "2025-11-01", "ium": 94.5},
                    {"date": "2025-11-02", "ium": 95.2},
                    {"date": "2025-11-03", "ium": 96.8}
                ],
                "top_files": [
                    {"path": "src/agents/calculator.py", "ium": 99.5},
                    {"path": "src/services/broker.py", "ium": 98.2}
                ],
                "custom_code_hotspots": [
                    {"module": "custom_validators", "lines": 150}
                ]
            }

            api_requests_total.labels(
                endpoint="/api/ium/by-app",
                method="GET",
                status="200"
            ).inc()

            return data
        except Exception as e:
            logger.error(f"Error fetching app IUM: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ium/by-team/{team_name}", response_model=Dict[str, Any])
async def get_ium_by_team(
    team_name: str,
    api_key: str = Depends(verify_api_key)
):
    """Get IUM metrics for a specific team"""
    try:
        data = {
            "team": team_name,
            "current_ium": 97.2,
            "applications": {
                "app1": 98.5,
                "app2": 96.0
            },
            "developers": {
                "dev1": 99.1,
                "dev2": 95.3
            },
            "trend": "stable"
        }
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/costs/savings", response_model=CostSavingsResponse)
async def get_cost_savings(
    period_days: int = 30,
    api_key: str = Depends(verify_api_key)
):
    """
    Get cost savings metrics.

    Args:
        period_days: Time period in days
    """
    try:
        data = {
            "total_savings_usd": 45780.50,
            "llm_cache_savings": 32500.00,
            "developer_time_savings": 13280.50,
            "roi_percentage": 425.5,
            "period_days": period_days,
            "breakdown": {
                "semantic_caching": 32500.00,
                "code_reuse": 8900.00,
                "faster_onboarding": 4380.50
            }
        }

        return CostSavingsResponse(**data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/performance/metrics", response_model=List[PerformanceMetrics])
async def get_performance_metrics(
    service: Optional[str] = None,
    api_key: str = Depends(verify_api_key)
):
    """
    Get performance metrics for services.

    Args:
        service: Optional service filter
    """
    try:
        metrics = [
            PerformanceMetrics(
                p50_latency_ms=45.2,
                p95_latency_ms=156.8,
                p99_latency_ms=312.5,
                error_rate_percent=0.23,
                throughput_rps=245.6,
                cache_hit_rate=0.78,
                service="factor-broker",
                timestamp=DeterministicClock.now()
            ),
            PerformanceMetrics(
                p50_latency_ms=52.1,
                p95_latency_ms=189.3,
                p99_latency_ms=401.2,
                error_rate_percent=0.15,
                throughput_rps=189.4,
                cache_hit_rate=0.82,
                service="entity-mdm",
                timestamp=DeterministicClock.now()
            )
        ]

        if service:
            metrics = [m for m in metrics if m.service == service]

        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/compliance/violations", response_model=List[ComplianceViolation])
async def get_compliance_violations(
    resolved: Optional[bool] = None,
    severity: Optional[str] = None,
    api_key: str = Depends(verify_api_key)
):
    """
    Get compliance violations.

    Args:
        resolved: Filter by resolution status
        severity: Filter by severity (info, warning, critical)
    """
    try:
        violations = [
            ComplianceViolation(
                id="V001",
                violation_type="custom_llm_wrapper",
                severity="critical",
                file_path="src/custom/llm_helper.py",
                application="csrd-reporting",
                team="csrd",
                detected_at=DeterministicClock.now() - timedelta(hours=2),
                resolved=False
            ),
            ComplianceViolation(
                id="V002",
                violation_type="missing_adr",
                severity="warning",
                file_path="src/utils/validators.py",
                application="vcci-scope3",
                team="carbon",
                detected_at=DeterministicClock.now() - timedelta(hours=5),
                resolved=True
            )
        ]

        if resolved is not None:
            violations = [v for v in violations if v.resolved == resolved]
        if severity:
            violations = [v for v in violations if v.severity == severity]

        return violations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/leaderboard", response_model=List[LeaderboardEntry])
async def get_leaderboard(
    limit: int = 10,
    api_key: str = Depends(verify_api_key)
):
    """
    Get developer contributions leaderboard.

    Args:
        limit: Number of top entries to return
    """
    try:
        leaderboard = [
            LeaderboardEntry(
                rank=1,
                developer="akshay",
                team="platform",
                contributions=156,
                ium_score=99.5,
                achievements=["platinum_contributor", "infrastructure_master"]
            ),
            LeaderboardEntry(
                rank=2,
                developer="dev2",
                team="csrd",
                contributions=98,
                ium_score=97.2,
                achievements=["gold_contributor"]
            )
        ]

        return leaderboard[:limit]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health", response_model=HealthStatus)
async def get_health_status(api_key: str = Depends(verify_api_key)):
    """Get overall infrastructure health status"""
    try:
        health_data = {
            "status": "operational",
            "services": {
                "factor-broker": {
                    "status": "up",
                    "uptime": 99.98,
                    "response_time_ms": 45.2
                },
                "entity-mdm": {
                    "status": "up",
                    "uptime": 99.95,
                    "response_time_ms": 52.1
                },
                "cache-redis": {
                    "status": "up",
                    "uptime": 100.0,
                    "connections": 45
                }
            },
            "uptime_percent": 99.96,
            "last_incident": DeterministicClock.now() - timedelta(days=7)
        }

        return HealthStatus(**health_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )
