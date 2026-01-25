# -*- coding: utf-8 -*-
"""
GL-010 EmissionsGuardian Compliance Module

This module provides the compliance rules engine for emissions monitoring
and regulatory compliance management.
"""

from .schemas import (
    AveragingPeriod,
    OperatingState,
    ExceedanceSeverity,
    CorrectiveActionState,
    RegulatoryProgram,
    RuleVersion,
    OperatingCondition,
    EffectiveDateRange,
    PermitRule,
    ComplianceSchedule,
    ExceedanceEvent,
    CorrectiveAction,
    ComplianceStatus,
    SubstitutionDataRecord,
)

from .rules_engine import RulesEngine
from .permit_parser import PermitParser
from .exceedance_handler import ExceedanceHandler
from .substitution_handler import SubstitutionHandler
from .reporting_calendar import ReportingCalendar

__all__ = [
    # Enums
    "AveragingPeriod",
    "OperatingState",
    "ExceedanceSeverity",
    "CorrectiveActionState",
    "RegulatoryProgram",
    # Models
    "RuleVersion",
    "OperatingCondition",
    "EffectiveDateRange",
    "PermitRule",
    "ComplianceSchedule",
    "ExceedanceEvent",
    "CorrectiveAction",
    "ComplianceStatus",
    "SubstitutionDataRecord",
    # Classes
    "RulesEngine",
    "PermitParser",
    "ExceedanceHandler",
    "SubstitutionHandler",
    "ReportingCalendar",
]
