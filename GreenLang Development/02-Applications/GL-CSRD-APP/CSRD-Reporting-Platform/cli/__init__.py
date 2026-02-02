# -*- coding: utf-8 -*-
"""
CSRD CLI Module
===============

Command-line interface for CSRD reporting.

Commands:
- csrd run: Execute full pipeline
- csrd validate: Validate ESG data only
- csrd audit: Check compliance
- csrd materialize: Run materiality assessment only
"""

from .csrd_commands import csrd

__all__ = ["csrd"]
