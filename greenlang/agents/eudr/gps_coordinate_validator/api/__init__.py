# -*- coding: utf-8 -*-
"""
GPS Coordinate Validator REST API - AGENT-EUDR-007

FastAPI router package providing 33 REST endpoints for EUDR GPS coordinate
validation operations including multi-format parsing, coordinate validation,
plausibility analysis, accuracy assessment, compliance reporting, datum
transformation, reverse geocoding, and batch processing.

Route Modules:
    - parsing_routes: Coordinate parsing (single, batch, detect, normalize)
    - validation_routes: Validation (single, batch, range, swap, duplicates)
    - plausibility_routes: Plausibility (full, land/ocean, country, commodity, elevation)
    - assessment_routes: Assessment (full, batch, retrieve, precision)
    - report_routes: Reporting (compliance cert, summary, remediation, retrieve, download)
    - batch_routes: Batch ops (reverse geocode, country, datum transform, batch jobs)
    - router: Main router registration with /v1/eudr-gcv prefix

Auth Integration:
    - JWT authentication via SEC-001 dependency injection
    - RBAC via SEC-002 with eudr-gcv:* permissions
    - Rate limiting via middleware decorator

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-007 GPS Coordinate Validator (GL-EUDR-GPS-007)
Status: Production Ready
"""

from greenlang.agents.eudr.gps_coordinate_validator.api.router import (
    router,
    get_router,
)

__all__ = [
    "router",
    "get_router",
]
