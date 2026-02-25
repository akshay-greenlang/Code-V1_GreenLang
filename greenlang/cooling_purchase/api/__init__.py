"""
AGENT-MRV-012 Cooling Purchase Agent REST API.

Provides 20 FastAPI endpoints at /api/v1/cooling-purchase.
"""

try:
    from greenlang.cooling_purchase.api.router import router, create_router
except ImportError:
    router = None
    create_router = None

__all__ = ["router", "create_router"]
