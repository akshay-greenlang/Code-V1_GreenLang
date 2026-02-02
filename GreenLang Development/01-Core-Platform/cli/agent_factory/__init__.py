# -*- coding: utf-8 -*-
"""
GreenLang Agent Factory CLI
===========================

Enhanced CLI tools for rapid agent creation, validation, testing, and deployment.
Supports GreenLang's 1-agent-per-hour productivity goal.

Commands:
    - gl agent create: Generate agents from templates
    - gl agent validate: Comprehensive spec validation
    - gl agent test: Run golden, integration, and e2e tests
    - gl agent certify: 12-dimension certification
    - gl agent deploy: Kubernetes deployment automation
    - gl agent list: List agents with filtering
    - gl agent diff: Compare agent versions
"""

__version__ = "1.0.0"
__author__ = "GreenLang Team"

from .cli_main import app, main
from .create_command import create_app
from .validate_command import validate_app
from .test_command import test_app
from .certify_command import certify_app
from .deploy_command import deploy_app
from .template_command import template_app
from .productivity_helpers import productivity_app

__all__ = [
    "app",
    "main",
    "create_app",
    "validate_app",
    "test_app",
    "certify_app",
    "deploy_app",
    "template_app",
    "productivity_app",
]
