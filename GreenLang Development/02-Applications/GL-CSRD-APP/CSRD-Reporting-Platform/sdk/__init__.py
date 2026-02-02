# -*- coding: utf-8 -*-
"""
CSRD SDK Module
===============

Python SDK for programmatic CSRD reporting.

Main API:
- csrd_build_report(): One-function API
- CSRDConfig: Reusable configuration
- CSRDReport: Report output dataclass

Example:
    from csrd_sdk import csrd_build_report, CSRDConfig

    config = CSRDConfig(
        company_name="Green Manufacturing GmbH",
        lei_code="529900DEMO00000000001",
        reporting_year=2024
    )

    report = csrd_build_report(
        esg_data="data.csv",
        company_profile="profile.json",
        config=config
    )
"""

from .csrd_sdk import (
    csrd_build_report,
    CSRDConfig,
    CSRDReport,
)

__all__ = [
    "csrd_build_report",
    "CSRDConfig",
    "CSRDReport",
]
