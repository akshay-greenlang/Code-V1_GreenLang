# -*- coding: utf-8 -*-
"""
GreenLang Policy Package

This module provides OPA (Open Policy Agent) integration for policy enforcement
at install and runtime.
"""

from .enforcer import check_install, check_run
from .opa import evaluate

__all__ = [
    "check_install",
    "check_run",
    "evaluate",
]
