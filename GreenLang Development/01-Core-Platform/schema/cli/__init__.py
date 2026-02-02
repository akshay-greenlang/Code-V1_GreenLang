# -*- coding: utf-8 -*-
"""
CLI Module for GL-FOUND-X-002 (GreenLang Schema Compiler & Validator).

This module provides the command-line interface for the schema validator.

Commands:
    - greenlang schema validate: Validate payloads against schemas
    - greenlang schema compile: Compile schemas to IR
    - greenlang schema lint: Lint schemas for best practices
    - greenlang validate: Alias for 'greenlang schema validate'

Exit Codes:
    0 - Success (valid payload or successful operation)
    1 - Invalid (validation failed)
    2 - Error (system error)

Example:
    $ greenlang schema validate data.yaml --schema emissions/activity@1.3.0
    $ greenlang schema compile schema.yaml --out ir.json
    $ greenlang schema lint schema.yaml

Author: GreenLang Framework Team
Version: 1.0.0
GL-FOUND-X-002: Schema Compiler & Validator - Task 5.1
"""

from greenlang.schema.cli.main import cli, main, schema

# Exit codes
EXIT_SUCCESS = 0
EXIT_INVALID = 1
EXIT_ERROR = 2


__all__ = [
    "cli",
    "main",
    "schema",
    "EXIT_SUCCESS",
    "EXIT_INVALID",
    "EXIT_ERROR",
]
