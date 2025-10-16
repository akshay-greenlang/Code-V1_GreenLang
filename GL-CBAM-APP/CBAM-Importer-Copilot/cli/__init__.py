"""
CBAM Importer Copilot - CLI Commands

GreenLang CLI integration for CBAM reporting.

Usage:
    gl cbam report [OPTIONS]

Version: 1.0.0
Author: GreenLang CBAM Team
"""

from .cbam_commands import cbam_report

__version__ = "1.0.0"
__all__ = ["cbam_report"]
