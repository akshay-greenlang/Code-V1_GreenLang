# -*- coding: utf-8 -*-
"""
Mass Balance Calculator Agent - AGENT-EUDR-011

Production-grade mass balance accounting engine for EUDR compliance
covering double-entry ledger management, credit period lifecycle,
conversion factor validation, overdraft detection and enforcement,
loss/waste tracking, carry-forward with expiry, reconciliation with
anomaly detection, and multi-facility consolidation reporting.

This package provides a comprehensive mass balance calculator for
EUDR supply chain traceability supporting mass balance chain of
custody per ISO 22095:2020:

    Capabilities:
        - Double-entry ledger management with real-time balance tracking,
          entry recording, voiding, and utilization rate calculation
        - Credit period lifecycle management with standard-specific
          durations (RSPO 90d, FSC 365d, ISCC 365d, EUDR default 365d),
          grace periods, and carry-forward rules
        - Conversion factor validation against reference yield ratios
          with configurable warn (5%) and reject (15%) deviation thresholds
        - Overdraft detection and enforcement with three modes
          (zero_tolerance, percentage, absolute) and resolution deadlines
        - Loss and waste tracking with six loss types, three waste types,
          and per-commodity tolerance validation
        - Carry-forward with expiry management including partial
          utilization tracking
        - Period-end reconciliation with variance analysis, anomaly
          detection (Z-score), trend analysis, and sign-off workflow
        - Multi-facility consolidation with regional, country, commodity,
          and custom facility grouping

    Foundational modules:
        - config: MassBalanceCalculatorConfig with GL_EUDR_MBC_ env var
          support (35+ settings)
        - models: Pydantic v2 data models with 15 enumerations, 11 core
          models, 18 request models, and 18 response models
        - provenance: SHA-256 chain-hashed audit trail tracking with
          12 entity types and 14 actions
        - metrics: 18 Prometheus self-monitoring metrics (gl_eudr_mbc_)

PRD: PRD-AGENT-EUDR-011
Agent ID: GL-EUDR-MBC-011
Regulation: EU 2023/1115 (EUDR) Articles 4, 10(2)(f), 14
Standard: ISO 22095:2020 Chain of Custody - Mass Balance
Enforcement: December 30, 2025 (large operators); June 30, 2026 (SMEs)

Example:
    >>> from greenlang.agents.eudr.mass_balance_calculator import (
    ...     Ledger,
    ...     LedgerEntry,
    ...     LedgerEntryType,
    ...     OverdraftMode,
    ...     StandardType,
    ...     MassBalanceCalculatorConfig,
    ...     get_config,
    ... )
    >>> ledger = Ledger(
    ...     facility_id="facility-001",
    ...     commodity="cocoa",
    ...     standard=StandardType.RSPO,
    ... )
    >>> entry = LedgerEntry(
    ...     ledger_id=ledger.ledger_id,
    ...     entry_type=LedgerEntryType.INPUT,
    ...     quantity_kg=5000,
    ... )

Author: GreenLang Platform Team
Date: March 2026
Status: Production Ready
"""

from __future__ import annotations

__version__ = "1.0.0"
__agent_id__ = "GL-EUDR-MBC-011"

# ---- Foundational: config ----
try:
    from greenlang.agents.eudr.mass_balance_calculator.config import (
        MassBalanceCalculatorConfig,
        get_config,
        set_config,
        reset_config,
    )
except ImportError:
    MassBalanceCalculatorConfig = None  # type: ignore[assignment,misc]
    get_config = None  # type: ignore[assignment]
    set_config = None  # type: ignore[assignment]
    reset_config = None  # type: ignore[assignment]

# ---- Foundational: models ----
try:
    from greenlang.agents.eudr.mass_balance_calculator.models import (
        # Constants
        VERSION,
        EUDR_DEFORESTATION_CUTOFF,
        MAX_BATCH_SIZE,
        DEFAULT_CREDIT_PERIOD_DAYS,
        RSPO_CREDIT_PERIOD_DAYS,
        FSC_CREDIT_PERIOD_DAYS,
        ISCC_CREDIT_PERIOD_DAYS,
        DEFAULT_GRACE_PERIOD_DAYS,
        DEFAULT_OVERDRAFT_RESOLUTION_HOURS,
        DEFAULT_VARIANCE_ACCEPTABLE_PCT,
        DEFAULT_VARIANCE_WARNING_PCT,
        DEFAULT_CF_WARN_DEVIATION,
        DEFAULT_CF_REJECT_DEVIATION,
        EUDR_RETENTION_YEARS,
        PRIMARY_COMMODITIES,
        DERIVED_TO_PRIMARY,
        # Enumerations
        LedgerEntryType,
        PeriodStatus,
        OverdraftSeverity,
        OverdraftMode,
        LossType,
        WasteType,
        ConversionStatus,
        VarianceClassification,
        ReconciliationStatus,
        CarryForwardStatus,
        ReportFormat,
        ReportType,
        FacilityGroupType,
        ComplianceStatus,
        StandardType,
        # Core Models
        Ledger,
        LedgerEntry,
        CreditPeriod,
        ConversionFactor,
        OverdraftEvent,
        LossRecord,
        CarryForward,
        Reconciliation,
        FacilityGroup,
        ConsolidationReport,
        BatchJob,
        # Request Models
        CreateLedgerRequest,
        RecordEntryRequest,
        BulkEntryRequest,
        SearchLedgerRequest,
        CreatePeriodRequest,
        ExtendPeriodRequest,
        RolloverPeriodRequest,
        ValidateFactorRequest,
        RegisterCustomFactorRequest,
        CheckOverdraftRequest,
        ForecastOutputRequest,
        RequestExemptionRequest,
        RecordLossRequest,
        ValidateLossRequest,
        RunReconciliationRequest,
        SignOffReconciliationRequest,
        GenerateConsolidationRequest,
        CreateFacilityGroupRequest,
        # Response Models
        LedgerResponse,
        LedgerBalanceResponse,
        EntryHistoryResponse,
        PeriodResponse,
        ActivePeriodsResponse,
        FactorValidationResponse,
        ReferenceFactorsResponse,
        OverdraftCheckResponse,
        OverdraftAlertResponse,
        ForecastResponse,
        LossValidationResponse,
        LossTrendsResponse,
        ReconciliationResponse,
        ReconciliationHistoryResponse,
        ConsolidationDashboardResponse,
        ConsolidationReportResponse,
        BatchJobResponse,
        HealthResponse,
    )
except ImportError:
    pass

# ---- Foundational: provenance ----
try:
    from greenlang.agents.eudr.mass_balance_calculator.provenance import (
        ProvenanceRecord,
        ProvenanceTracker,
        VALID_ENTITY_TYPES,
        VALID_ACTIONS,
        get_provenance_tracker,
        set_provenance_tracker,
        reset_provenance_tracker,
    )
except ImportError:
    ProvenanceRecord = None  # type: ignore[assignment,misc]
    ProvenanceTracker = None  # type: ignore[assignment,misc]
    VALID_ENTITY_TYPES = frozenset()  # type: ignore[assignment]
    VALID_ACTIONS = frozenset()  # type: ignore[assignment]
    get_provenance_tracker = None  # type: ignore[assignment]
    set_provenance_tracker = None  # type: ignore[assignment]
    reset_provenance_tracker = None  # type: ignore[assignment]

# ---- Foundational: metrics ----
try:
    from greenlang.agents.eudr.mass_balance_calculator.metrics import (
        PROMETHEUS_AVAILABLE,
        # Metric objects
        mbc_ledger_entries_total,
        mbc_input_entries_total,
        mbc_output_entries_total,
        mbc_overdrafts_detected_total,
        mbc_overdrafts_critical_total,
        mbc_conversion_validations_total,
        mbc_conversion_rejections_total,
        mbc_losses_recorded_total,
        mbc_credits_expired_total,
        mbc_reconciliations_total,
        mbc_reports_generated_total,
        mbc_batch_jobs_total,
        mbc_api_errors_total,
        mbc_entry_recording_duration_seconds,
        mbc_reconciliation_duration_seconds,
        mbc_overdraft_check_duration_seconds,
        mbc_active_ledgers,
        mbc_total_balance_kg,
        # Helper functions
        record_ledger_entry,
        record_input_entry,
        record_output_entry,
        record_overdraft_detected,
        record_overdraft_critical,
        record_conversion_validation,
        record_conversion_rejection,
        record_loss_recorded,
        record_credit_expired,
        record_reconciliation,
        record_report_generated,
        record_batch_job,
        record_api_error,
        observe_entry_recording_duration,
        observe_reconciliation_duration,
        observe_overdraft_check_duration,
        set_active_ledgers,
        set_total_balance_kg,
    )
except ImportError:
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]


__all__ = [
    # -- Version --
    "__version__",
    "__agent_id__",
    "VERSION",
    # -- Config --
    "MassBalanceCalculatorConfig",
    "get_config",
    "set_config",
    "reset_config",
    # -- Constants --
    "EUDR_DEFORESTATION_CUTOFF",
    "MAX_BATCH_SIZE",
    "DEFAULT_CREDIT_PERIOD_DAYS",
    "RSPO_CREDIT_PERIOD_DAYS",
    "FSC_CREDIT_PERIOD_DAYS",
    "ISCC_CREDIT_PERIOD_DAYS",
    "DEFAULT_GRACE_PERIOD_DAYS",
    "DEFAULT_OVERDRAFT_RESOLUTION_HOURS",
    "DEFAULT_VARIANCE_ACCEPTABLE_PCT",
    "DEFAULT_VARIANCE_WARNING_PCT",
    "DEFAULT_CF_WARN_DEVIATION",
    "DEFAULT_CF_REJECT_DEVIATION",
    "EUDR_RETENTION_YEARS",
    "PRIMARY_COMMODITIES",
    "DERIVED_TO_PRIMARY",
    # -- Enumerations --
    "LedgerEntryType",
    "PeriodStatus",
    "OverdraftSeverity",
    "OverdraftMode",
    "LossType",
    "WasteType",
    "ConversionStatus",
    "VarianceClassification",
    "ReconciliationStatus",
    "CarryForwardStatus",
    "ReportFormat",
    "ReportType",
    "FacilityGroupType",
    "ComplianceStatus",
    "StandardType",
    # -- Core Models --
    "Ledger",
    "LedgerEntry",
    "CreditPeriod",
    "ConversionFactor",
    "OverdraftEvent",
    "LossRecord",
    "CarryForward",
    "Reconciliation",
    "FacilityGroup",
    "ConsolidationReport",
    "BatchJob",
    # -- Request Models --
    "CreateLedgerRequest",
    "RecordEntryRequest",
    "BulkEntryRequest",
    "SearchLedgerRequest",
    "CreatePeriodRequest",
    "ExtendPeriodRequest",
    "RolloverPeriodRequest",
    "ValidateFactorRequest",
    "RegisterCustomFactorRequest",
    "CheckOverdraftRequest",
    "ForecastOutputRequest",
    "RequestExemptionRequest",
    "RecordLossRequest",
    "ValidateLossRequest",
    "RunReconciliationRequest",
    "SignOffReconciliationRequest",
    "GenerateConsolidationRequest",
    "CreateFacilityGroupRequest",
    # -- Response Models --
    "LedgerResponse",
    "LedgerBalanceResponse",
    "EntryHistoryResponse",
    "PeriodResponse",
    "ActivePeriodsResponse",
    "FactorValidationResponse",
    "ReferenceFactorsResponse",
    "OverdraftCheckResponse",
    "OverdraftAlertResponse",
    "ForecastResponse",
    "LossValidationResponse",
    "LossTrendsResponse",
    "ReconciliationResponse",
    "ReconciliationHistoryResponse",
    "ConsolidationDashboardResponse",
    "ConsolidationReportResponse",
    "BatchJobResponse",
    "HealthResponse",
    # -- Provenance --
    "ProvenanceRecord",
    "ProvenanceTracker",
    "VALID_ENTITY_TYPES",
    "VALID_ACTIONS",
    "get_provenance_tracker",
    "set_provenance_tracker",
    "reset_provenance_tracker",
    # -- Metrics --
    "PROMETHEUS_AVAILABLE",
    "mbc_ledger_entries_total",
    "mbc_input_entries_total",
    "mbc_output_entries_total",
    "mbc_overdrafts_detected_total",
    "mbc_overdrafts_critical_total",
    "mbc_conversion_validations_total",
    "mbc_conversion_rejections_total",
    "mbc_losses_recorded_total",
    "mbc_credits_expired_total",
    "mbc_reconciliations_total",
    "mbc_reports_generated_total",
    "mbc_batch_jobs_total",
    "mbc_api_errors_total",
    "mbc_entry_recording_duration_seconds",
    "mbc_reconciliation_duration_seconds",
    "mbc_overdraft_check_duration_seconds",
    "mbc_active_ledgers",
    "mbc_total_balance_kg",
    "record_ledger_entry",
    "record_input_entry",
    "record_output_entry",
    "record_overdraft_detected",
    "record_overdraft_critical",
    "record_conversion_validation",
    "record_conversion_rejection",
    "record_loss_recorded",
    "record_credit_expired",
    "record_reconciliation",
    "record_report_generated",
    "record_batch_job",
    "record_api_error",
    "observe_entry_recording_duration",
    "observe_reconciliation_duration",
    "observe_overdraft_check_duration",
    "set_active_ledgers",
    "set_total_balance_kg",
]
