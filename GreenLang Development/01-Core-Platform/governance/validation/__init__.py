# -*- coding: utf-8 -*-
"""
GreenLang Validation Framework
Comprehensive validation system with JSON Schema, business rules, data quality checks,
and EUDR geolocation validation.
"""

from .framework import ValidationFramework, ValidationResult, ValidationError
from .schema import SchemaValidator, SchemaValidationError
from .rules import RulesEngine, Rule, RuleSet
from .quality import DataQualityValidator, QualityCheck
from .decorators import validate, validate_schema, validate_rules

# Geolocation validation submodule
from . import geolocation

__all__ = [
    "ValidationFramework",
    "ValidationResult",
    "ValidationError",
    "SchemaValidator",
    "SchemaValidationError",
    "RulesEngine",
    "Rule",
    "RuleSet",
    "DataQualityValidator",
    "QualityCheck",
    "validate",
    "validate_schema",
    "validate_rules",
    # Geolocation submodule
    "geolocation",
]
