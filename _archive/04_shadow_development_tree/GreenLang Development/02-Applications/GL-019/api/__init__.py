"""
GL-019 HEATSCHEDULER REST API Module

ProcessHeatingScheduler API - Schedules process heating operations to minimize energy costs.

This module provides a production-grade FastAPI REST API for:
- Schedule optimization and management
- Production batch integration
- Energy tariff management
- Equipment monitoring and control
- Cost analytics and forecasting
- Demand response event handling

Author: GL-APIDeveloper
Version: 1.0.0
"""

from api.main import app, create_application
from api.routes import router

__all__ = ["app", "create_application", "router"]
__version__ = "1.0.0"
__agent_id__ = "GL-019"
__codename__ = "HEATSCHEDULER"