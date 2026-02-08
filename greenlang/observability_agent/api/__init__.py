# -*- coding: utf-8 -*-
"""
Observability Agent Service REST API Package - AGENT-FOUND-010: Observability Agent

Provides the FastAPI router for the Observability & Telemetry Agent SDK endpoints.

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-010 Observability & Telemetry Agent
Status: Production Ready
"""

from greenlang.observability_agent.api.router import router

__all__ = ["router"]
