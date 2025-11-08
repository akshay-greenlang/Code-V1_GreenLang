"""
GL-VCCI CLI Commands Module
Sub-commands for the VCCI CLI

Commands:
- intake: Data ingestion and validation
- calculate: Emissions calculation
- analyze: Analysis and insights
- engage: Supplier engagement
- report: Report generation
- pipeline: End-to-end workflows
- status: System status
- config: Configuration management

Version: 1.0.0
Date: 2025-11-08
"""

# Import command modules
from .intake import intake_app
from .engage import engage_app
from .pipeline import pipeline_app

__all__ = [
    "intake_app",
    "engage_app",
    "pipeline_app",
]
