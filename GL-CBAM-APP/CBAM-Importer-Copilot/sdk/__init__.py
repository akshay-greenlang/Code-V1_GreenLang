# -*- coding: utf-8 -*-
"""
CBAM Importer Copilot - Python SDK

Simple Python API for CBAM reporting in applications and scripts.

Quick Start:
    from cbam_copilot import cbam_build_report

    report = cbam_build_report(
        input_file="shipments.csv",
        importer_name="Acme Steel EU BV",
        importer_country="NL",
        importer_eori="NL123456789012",
        declarant_name="John Smith",
        declarant_position="Compliance Officer"
    )

Version: 1.0.0
Author: GreenLang CBAM Team
"""

from .cbam_sdk import (
    cbam_build_report,
    cbam_validate_shipments,
    cbam_calculate_emissions,
    CBAMReport,
    CBAMConfig
)

__version__ = "1.0.0"
__all__ = [
    "cbam_build_report",
    "cbam_validate_shipments",
    "cbam_calculate_emissions",
    "CBAMReport",
    "CBAMConfig"
]
