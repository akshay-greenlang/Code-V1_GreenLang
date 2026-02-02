"""
Emergency Shutdown (ESD) Framework - IEC 61511 Compliant ESD System

This module provides Emergency Shutdown system components and interfaces
per IEC 61511 for Safety Instrumented Systems.

Core Components:
- ESDInterface: ESD system interface specification
- PriorityManager: ESD > DCS > Agent priority management
- ResponseValidator: <1s response time validation
- BypassManager: Bypass management per IEC 61511
- ESDSimulator: ESD simulation mode for testing

Extended Components (TASK-182 through TASK-190):
- HardwiredInterlockManager: Hardwired interlock status monitoring
- ResponseTimeValidator: Enhanced response time validation framework
- ESDTestManager: Automated ESD test procedures
- BypassWorkflowManager: Bypass request workflow with authorization
- BypassAuditLogger: Comprehensive bypass event logging
- ESDSimulationEngine: Enhanced simulation with what-if analysis
- ESDReportGenerator: ESD audit reports generation

Reference: IEC 61511-1 Clause 11, IEC 61508-2

Example:
    >>> from greenlang.safety.esd import ESDInterface, PriorityManager
    >>> esd = ESDInterface(system_id="ESD-001")
    >>> priority = PriorityManager()

    >>> from greenlang.safety.esd import HardwiredInterlockManager
    >>> interlock_mgr = HardwiredInterlockManager(system_id="ESD-001")

    >>> from greenlang.safety.esd import ESDTestManager
    >>> test_mgr = ESDTestManager(system_id="ESD-001")
"""

# Core ESD Interface
from greenlang.safety.esd.esd_interface import (
    ESDInterface,
    ESDCommand,
    ESDStatus,
    ESDLevel,
    ESDState,
    ESDCommandType,
)

# Priority Manager
from greenlang.safety.esd.priority_manager import (
    PriorityManager,
    CommandPriority,
    PriorityResult,
    Command,
    CommandSource,
    CommandDecision,
)

# Legacy Response Validator
from greenlang.safety.esd.response_validator import (
    ResponseValidator,
    ResponseTest,
    ResponseResult,
)

# Bypass Manager
from greenlang.safety.esd.bypass_manager import (
    BypassManager,
    BypassRequest,
    BypassRecord,
    BypassType,
    BypassStatus,
)

# ESD Simulator
from greenlang.safety.esd.esd_simulator import (
    ESDSimulator,
    SimulationConfig,
    SimulationResult,
)

# TASK-182: Hardwired Interlock Integration
from greenlang.safety.esd.hardwired_interlock import (
    HardwiredInterlockManager,
    InterlockDefinition,
    InterlockStatus,
    InterlockBypass,
    InterlockTestResult,
    InterlockState,
    InterlockType,
    InterlockSignal,
    SignalType,
)

# TASK-185: Response Time Validation (<1s)
from greenlang.safety.esd.response_validation import (
    ResponseTimeValidator,
    ResponseMeasurement,
    ValidationSchedule,
    DegradationAnalysis,
    ComplianceReport as ResponseComplianceReport,
    ValidationResult,
    ComponentType,
)

# TASK-186: ESD Test Procedures
from greenlang.safety.esd.test_procedures import (
    ESDTestManager,
    TestProcedure,
    TestResult,
    TestStep,
    TestSchedule,
    PartialStrokeResult,
    TestType,
    TestStatus,
)

# TASK-187: Bypass Management Workflow
from greenlang.safety.esd.bypass_workflow import (
    BypassWorkflowManager,
    WorkflowRequest,
    BypassRequestData,
    BypassAlarm,
    AuthorizationLevel,
    WorkflowState,
    AlarmPriority,
)

# TASK-188: Bypass Audit Logging
from greenlang.safety.esd.bypass_audit import (
    BypassAuditLogger,
    BypassAuditEvent,
    BypassDurationRecord,
    ComplianceViolation,
    AuditEventType,
    AuditSeverity,
)

# TASK-189: Enhanced ESD Simulation
from greenlang.safety.esd.simulation_enhanced import (
    ESDSimulationEngine,
    SimulationScenario,
    SimulationResult as EnhancedSimulationResult,
    ScenarioStep,
    WhatIfAnalysis,
    TrainingSession,
    ComponentState,
    SimulationMode,
    SimulationIsolation,
    ScenarioType,
)

# TASK-190: ESD Audit Reports
from greenlang.safety.esd.audit_reports import (
    ESDReportGenerator,
    TestHistoryReport,
    BypassHistoryReport,
    ResponseTimeTrendReport,
    ComplianceReport,
    ReportMetadata,
    ReportType,
    ReportFormat,
    ComplianceStatus,
)


__all__ = [
    # Core ESD Interface
    "ESDInterface",
    "ESDCommand",
    "ESDStatus",
    "ESDLevel",
    "ESDState",
    "ESDCommandType",

    # Priority Manager
    "PriorityManager",
    "CommandPriority",
    "PriorityResult",
    "Command",
    "CommandSource",
    "CommandDecision",

    # Legacy Response Validator
    "ResponseValidator",
    "ResponseTest",
    "ResponseResult",

    # Bypass Manager
    "BypassManager",
    "BypassRequest",
    "BypassRecord",
    "BypassType",
    "BypassStatus",

    # ESD Simulator
    "ESDSimulator",
    "SimulationConfig",
    "SimulationResult",

    # TASK-182: Hardwired Interlock Integration
    "HardwiredInterlockManager",
    "InterlockDefinition",
    "InterlockStatus",
    "InterlockBypass",
    "InterlockTestResult",
    "InterlockState",
    "InterlockType",
    "InterlockSignal",
    "SignalType",

    # TASK-185: Response Time Validation
    "ResponseTimeValidator",
    "ResponseMeasurement",
    "ValidationSchedule",
    "DegradationAnalysis",
    "ResponseComplianceReport",
    "ValidationResult",
    "ComponentType",

    # TASK-186: ESD Test Procedures
    "ESDTestManager",
    "TestProcedure",
    "TestResult",
    "TestStep",
    "TestSchedule",
    "PartialStrokeResult",
    "TestType",
    "TestStatus",

    # TASK-187: Bypass Workflow Management
    "BypassWorkflowManager",
    "WorkflowRequest",
    "BypassRequestData",
    "BypassAlarm",
    "AuthorizationLevel",
    "WorkflowState",
    "AlarmPriority",

    # TASK-188: Bypass Audit Logging
    "BypassAuditLogger",
    "BypassAuditEvent",
    "BypassDurationRecord",
    "ComplianceViolation",
    "AuditEventType",
    "AuditSeverity",

    # TASK-189: Enhanced ESD Simulation
    "ESDSimulationEngine",
    "SimulationScenario",
    "EnhancedSimulationResult",
    "ScenarioStep",
    "WhatIfAnalysis",
    "TrainingSession",
    "ComponentState",
    "SimulationMode",
    "SimulationIsolation",
    "ScenarioType",

    # TASK-190: ESD Audit Reports
    "ESDReportGenerator",
    "TestHistoryReport",
    "BypassHistoryReport",
    "ResponseTimeTrendReport",
    "ComplianceReport",
    "ReportMetadata",
    "ReportType",
    "ReportFormat",
    "ComplianceStatus",
]

__version__ = "2.0.0"
