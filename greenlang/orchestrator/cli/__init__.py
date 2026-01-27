# -*- coding: utf-8 -*-
"""
GreenLang Orchestrator CLI
===========================

Command-line interface for the GL-FOUND-X-001 GreenLang Orchestrator.

Usage:
    gl run pipeline.yaml                    # Submit a pipeline run
    gl status <run-id>                      # Check run status
    gl logs <run-id>                        # View run logs
    gl cancel <run-id>                      # Cancel a run
    gl list                                 # List recent runs
    gl validate pipeline.yaml               # Validate a pipeline file
    gl agents                               # List registered agents

Author: GreenLang Team
Version: 1.0.0
"""

from greenlang.orchestrator.cli.main import create_app, main

__all__ = ["create_app", "main"]
