# -*- coding: utf-8 -*-
"""
CLI Commands for GL-FOUND-X-002 (GreenLang Schema Compiler & Validator).

This module exports all CLI commands:
    - validate: Validate payloads against schemas
    - compile_schema: Compile schemas to IR
    - lint: Lint schemas for best practices
    - migrate: Schema migration tools (analyze, report, convert)

Author: GreenLang Framework Team
Version: 1.0.0
GL-FOUND-X-002: Schema Compiler & Validator - Task 5.1
"""

from greenlang.schema.cli.commands.validate import validate
from greenlang.schema.cli.commands.compile import compile_schema
from greenlang.schema.cli.commands.lint import lint
from greenlang.schema.cli.commands.migrate import (
    migrate,
    MigrationAnalyzer,
    SchemaConverter,
    ValidationPattern,
    MigrationReport,
    SchemaConversion,
)


__all__ = [
    "validate",
    "compile_schema",
    "lint",
    "migrate",
    "MigrationAnalyzer",
    "SchemaConverter",
    "ValidationPattern",
    "MigrationReport",
    "SchemaConversion",
]
