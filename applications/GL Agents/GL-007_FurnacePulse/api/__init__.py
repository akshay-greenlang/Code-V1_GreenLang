"""
GL-007 FurnacePulse - API Module

REST and GraphQL API endpoints for industrial furnace monitoring,
predictive maintenance, and NFPA 86 compliance management.

Provides:
- REST endpoints for furnace KPIs, TMT monitoring, alerts, and compliance
- GraphQL schema for flexible queries on furnace telemetry and predictions
- Real-time subscriptions for telemetry and alert streams
- RBAC middleware with role-based access control
- Audit logging for compliance and regulatory requirements
- Prometheus metrics endpoint for observability

Endpoints:
- GET /health - Service health check
- GET /status - Agent status with KPIs
- GET /furnaces/{furnace_id}/kpis - Current efficiency KPIs
- GET /furnaces/{furnace_id}/hotspots - Active hotspot alerts
- GET /furnaces/{furnace_id}/tmt - Current TMT readings
- GET /furnaces/{furnace_id}/compliance - NFPA 86 compliance status
- GET /furnaces/{furnace_id}/rul - RUL predictions for components
- POST /furnaces/{furnace_id}/evidence - Generate evidence package
- GET /alerts - List active alerts with filtering
- POST /alerts/{alert_id}/acknowledge - Acknowledge alert
- GET /explain/{prediction_id} - Get SHAP/LIME explanation
- GET /metrics - Prometheus metrics endpoint

Author: GreenLang API Team
Version: 1.0.0
"""

from .rest_api import create_app, router, app
from .graphql_schema import (
    schema,
    FurnacePulseQuery,
    FurnacePulseMutation,
    FurnacePulseSubscription,
)
from .middleware import (
    RBACMiddleware,
    AuditLoggingMiddleware,
    RequestIDMiddleware,
    RateLimitMiddleware,
    ErrorHandlingMiddleware,
    ProvenanceMiddleware,
    UserRole,
)

__all__ = [
    # REST API
    "create_app",
    "router",
    "app",
    # GraphQL
    "schema",
    "FurnacePulseQuery",
    "FurnacePulseMutation",
    "FurnacePulseSubscription",
    # Middleware
    "RBACMiddleware",
    "AuditLoggingMiddleware",
    "RequestIDMiddleware",
    "RateLimitMiddleware",
    "ErrorHandlingMiddleware",
    "ProvenanceMiddleware",
    "UserRole",
]

__version__ = "1.0.0"
__agent__ = "GL-007_FurnacePulse"
