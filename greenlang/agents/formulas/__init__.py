"""
GreenLang Formula Versioning System

This package provides centralized formula management with version control,
audit trails, rollback capabilities, and A/B testing support.

Features:
    - Version-controlled formula storage (SQLite/PostgreSQL)
    - Rollback to previous formula versions
    - A/B testing between formula variants
    - Dependency resolution and topological sorting
    - Execution logging and provenance tracking
    - Migration from YAML and Python modules

Example:
    >>> from greenlang.formulas import FormulaManager
    >>> manager = FormulaManager("formulas.db")
    >>> result = manager.execute_formula("E1-1", {"scope1": 100, "scope2": 50})
    >>> print(result)
"""

from greenlang.formulas.manager import FormulaManager
from greenlang.formulas.models import (
    FormulaVersion,
    FormulaMetadata,
    FormulaExecutionResult,
)
from greenlang.formulas.repository import FormulaRepository
from greenlang.formulas.engine import FormulaExecutionEngine

__all__ = [
    "FormulaManager",
    "FormulaVersion",
    "FormulaMetadata",
    "FormulaExecutionResult",
    "FormulaRepository",
    "FormulaExecutionEngine",
]

__version__ = "1.0.0"
