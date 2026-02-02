# -*- coding: utf-8 -*-
"""
GreenLang Public Sector Reporting Agents
========================================

Agents for public sector climate reporting including municipal inventories,
progress tracking, citizen dashboards, and regional/national reporting.

Agents:
    GL-REP-PUB-001: Municipal GHG Inventory - City-level inventories
    GL-REP-PUB-002: Climate Action Progress Report - Progress tracking
    GL-REP-PUB-003: Citizen Climate Dashboard - Public dashboards
    GL-REP-PUB-004: Regional Climate Report - Regional reporting
    GL-REP-PUB-005: National Climate Report - National reporting
"""

from greenlang.agents.reporting.public.municipal_ghg_inventory import (
    MunicipalGHGInventoryAgent,
    MunicipalGHGInventoryInput,
    MunicipalGHGInventoryOutput,
    GPCInventory,
    GPCSector,
)

from greenlang.agents.reporting.public.climate_progress_report import (
    ClimateActionProgressReportAgent,
    ProgressReportInput,
    ProgressReportOutput,
    ProgressReport,
)

from greenlang.agents.reporting.public.citizen_dashboard import (
    CitizenClimateDashboardAgent,
    CitizenDashboardInput,
    CitizenDashboardOutput,
    DashboardData,
)

from greenlang.agents.reporting.public.regional_climate_report import (
    RegionalClimateReportAgent,
    RegionalReportInput,
    RegionalReportOutput,
    RegionalReport,
)

from greenlang.agents.reporting.public.national_climate_report import (
    NationalClimateReportAgent,
    NationalReportInput,
    NationalReportOutput,
    NationalReport,
)

__all__ = [
    "MunicipalGHGInventoryAgent",
    "MunicipalGHGInventoryInput",
    "MunicipalGHGInventoryOutput",
    "GPCInventory",
    "GPCSector",
    "ClimateActionProgressReportAgent",
    "ProgressReportInput",
    "ProgressReportOutput",
    "ProgressReport",
    "CitizenClimateDashboardAgent",
    "CitizenDashboardInput",
    "CitizenDashboardOutput",
    "DashboardData",
    "RegionalClimateReportAgent",
    "RegionalReportInput",
    "RegionalReportOutput",
    "RegionalReport",
    "NationalClimateReportAgent",
    "NationalReportInput",
    "NationalReportOutput",
    "NationalReport",
]
