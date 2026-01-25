# -*- coding: utf-8 -*-
"""
GreenLang Emission Factor REST API

Production-grade FastAPI service for emission factor queries and calculations.
Supports 1000+ factors and 1M+ calculations per day.
"""

__version__ = "1.0.0"

from .main import app
from .models import (
    CalculationRequest,
    CalculationResponse,
    EmissionFactorResponse,
    FactorListResponse,
)

__all__ = [
    "app",
    "CalculationRequest",
    "CalculationResponse",
    "EmissionFactorResponse",
    "FactorListResponse",
]
