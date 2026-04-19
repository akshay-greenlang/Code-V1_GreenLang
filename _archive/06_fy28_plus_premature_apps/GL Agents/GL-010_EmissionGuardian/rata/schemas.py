# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional
import hashlib
import json
from pydantic import BaseModel, Field


class Pollutant(str, Enum):
    NOX = 'nox'
    SO2 = 'so2'
    CO2 = 'co2'
    CO = 'co'
    O2 = 'o2'
    FLOW = 'flow'
    MOISTURE = 'moisture'


class RATATestType(str, Enum):
    STANDARD = 'standard'
    ABBREVIATED = 'abbreviated'
    SINGLE_LOAD = 'single_load'
    THREE_LOAD = 'three_load'


class RATAStatus(str, Enum):
    SCHEDULED = 'scheduled'
    CONTRACTOR_NOTIFIED = 'contractor_notified'
    PROTOCOL_APPROVED = 'protocol_approved'
    IN_PROGRESS = 'in_progress'
    DATA_REVIEW = 'data_review'
    CALCULATIONS_COMPLETE = 'calculations_complete'
    REPORT_GENERATED = 'report_generated'
    SUBMITTED = 'submitted'
    APPROVED = 'approved'
    FAILED = 'failed'
    CANCELLED = 'cancelled'
    POSTPONED = 'postponed'


class ReferenceMethod(str, Enum):
    METHOD_3A = 'method_3a'
    METHOD_6C = 'method_6c'
    METHOD_7E = 'method_7e'
    METHOD_10 = 'method_10'
    METHOD_2 = 'method_2'
    METHOD_2F = 'method_2f'
    METHOD_2G = 'method_2g'
    METHOD_2H = 'method_2h'
    METHOD_4 = 'method_4'


class LoadLevel(str, Enum):
    LOW = 'low'
    MID = 'mid'
    HIGH = 'high'


class PassFailStatus(str, Enum):
    PASS = 'pass'
    FAIL = 'fail'
    CONDITIONAL = 'conditional'
    PENDING = 'pending'


class CalibrationGasLevel(str, Enum):
    ZERO = 'zero'
    LOW = 'low'
    MID = 'mid'
    HIGH = 'high'


@dataclass
class RATATest:
    test_id: str
    monitor_id: str
    pollutant: Pollutant
    test_type: RATATestType = RATATestType.STANDARD
    scheduled_date: Optional[date] = None
    actual_date: Optional[date] = None
    status: RATAStatus = RATAStatus.SCHEDULED
    contractor: Optional[str] = None
    reference_method: Optional[ReferenceMethod] = None


@dataclass  
class RATARun:
    run_number: int
    start_time: datetime
    end_time: datetime
    reference_value: Decimal
    cems_value: Decimal
    load_percent: Decimal
    operating_conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RATAResult:
    test_id: str
    relative_accuracy_percent: Decimal
    bias_percent: Decimal
    mean_difference: Decimal
    confidence_interval: Decimal
    pass_fail: PassFailStatus
    calculation_trace: Dict[str, Any] = field(default_factory=dict)
    provenance_hash: str = ''


@dataclass
class BiasTestResult:
    t_statistic: Decimal
    critical_value: Decimal
    is_biased: bool
    bias_adjustment_factor: Decimal
    needs_adjustment: bool


@dataclass
class CylinderGasAudit:
    audit_id: str
    date: date
    cylinder_id: str
    certified_concentration: Decimal
    measured_concentration: Decimal
    percent_error: Decimal
    pass_fail: PassFailStatus
