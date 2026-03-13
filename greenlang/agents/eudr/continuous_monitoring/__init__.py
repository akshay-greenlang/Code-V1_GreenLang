# -*- coding: utf-8 -*-
"""
AGENT-EUDR-033: Continuous Monitoring Agent

Real-time supply chain surveillance, deforestation alert correlation,
automated compliance verification, change detection, risk score trending,
data freshness validation, and regulatory change tracking for EUDR
compliance continuous monitoring.

Core capabilities:
    1. SupplyChainMonitor            -- Monitor supplier status,
       certification expiry, geolocation stability, and ownership changes
    2. DeforestationMonitor          -- Correlate EUDR-020 deforestation
       alerts with supply chain entities, assess impact, trigger
       investigations
    3. ComplianceChecker             -- Automated EUDR compliance audits
       for Article 8, risk assessment validity, DDS currency
    4. ChangeDetector                -- Detect and classify entity changes
       with weighted impact scoring and action recommendations
    5. RiskScoreMonitor              -- Track risk score trends via linear
       regression, detect degradation, correlate with incidents
    6. DataFreshnessValidator        -- Validate data currency, identify
       stale entities, schedule refresh operations
    7. RegulatoryTracker             -- Monitor regulatory changes,
       assess impact, map to entities, notify stakeholders

Foundational modules:
    - config.py       -- ContinuousMonitoringConfig with GL_EUDR_CM_
      env var support (60+ settings)
    - models.py       -- Pydantic v2 data models with 14 enumerations,
      15+ core models, and health/status types
    - provenance.py   -- SHA-256 chain-hashed audit trail tracking
    - metrics.py      -- 40 Prometheus self-monitoring metrics (gl_eudr_cm_)

Agent ID: GL-EUDR-CM-033
Module: greenlang.agents.eudr.continuous_monitoring
PRD: PRD-AGENT-EUDR-033
Regulation: EU 2023/1115 (EUDR) Articles 4, 8, 10, 11, 12, 14, 29, 31

Example:
    >>> from greenlang.agents.eudr.continuous_monitoring import (
    ...     ContinuousMonitoringConfig,
    ...     get_config,
    ...     MonitoringScope,
    ...     AlertSeverity,
    ... )
    >>> cfg = get_config()
    >>> print(cfg.supply_chain_scan_interval_minutes)
    60

    >>> from greenlang.agents.eudr.continuous_monitoring import (
    ...     ContinuousMonitoringService,
    ...     get_service,
    ... )
    >>> service = get_service()
    >>> health = await service.health_check()
    >>> assert health["status"] in ("healthy", "degraded")

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-033 Continuous Monitoring Agent (GL-EUDR-CM-033)
Status: Production Ready
"""

from __future__ import annotations

__version__ = "1.0.0"
__agent_id__ = "GL-EUDR-CM-033"

# ---------------------------------------------------------------------------
# Public API listing
# ---------------------------------------------------------------------------

__all__: list[str] = [
    # -- Metadata --
    "__version__",
    "__agent_id__",
    # -- Configuration --
    "ContinuousMonitoringConfig",
    "get_config",
    "reset_config",
    # -- Enumerations (14) --
    "MonitoringScope",
    "ScanStatus",
    "AlertSeverity",
    "AlertStatus",
    "CertificationStatus",
    "ChangeType",
    "ChangeImpact",
    "ComplianceStatus",
    "TrendDirection",
    "FreshnessStatus",
    "RegulatoryImpact",
    "RiskLevel",
    "InvestigationStatus",
    "AuditAction",
    # -- Core Models (15+) --
    "SupplyChainScanRecord",
    "DeforestationMonitorRecord",
    "ComplianceAuditRecord",
    "ChangeDetectionRecord",
    "RiskScoreMonitorRecord",
    "DataFreshnessRecord",
    "RegulatoryTrackingRecord",
    "MonitoringAlert",
    "InvestigationRecord",
    "MonitoringSummary",
    "AuditEntry",
    "HealthStatus",
    # -- Sub-models --
    "SupplierChange",
    "CertificationCheck",
    "GeolocationShift",
    "DeforestationCorrelation",
    "ComplianceCheckItem",
    "RiskScoreSnapshot",
    "StaleEntity",
    "RegulatoryUpdate",
    "ActionRecommendation",
    # -- Constants --
    "AGENT_ID",
    "AGENT_VERSION",
    "EUDR_ARTICLES_MONITORED",
    # -- Provenance --
    "ProvenanceTracker",
    "GENESIS_HASH",
    # -- Metrics (40) --
    "record_supply_chain_scan",
    "record_supplier_change_detected",
    "record_certification_expiry",
    "record_geolocation_drift",
    "record_deforestation_check",
    "record_deforestation_correlation",
    "record_investigation_triggered",
    "record_compliance_audit",
    "record_compliance_check_passed",
    "record_compliance_check_failed",
    "record_change_detected",
    "record_risk_monitor",
    "record_risk_degradation",
    "record_freshness_check",
    "record_regulatory_update",
    "record_alert_generated",
    "observe_supply_chain_scan_duration",
    "observe_deforestation_check_duration",
    "observe_compliance_audit_duration",
    "observe_change_detection_duration",
    "observe_risk_monitor_duration",
    "observe_freshness_check_duration",
    "observe_regulatory_check_duration",
    "observe_impact_assessment_duration",
    "observe_correlation_duration",
    "observe_investigation_trigger_duration",
    "observe_trend_analysis_duration",
    "observe_freshness_report_duration",
    "observe_notification_duration",
    "observe_entity_mapping_duration",
    "set_active_scans",
    "set_pending_investigations",
    "set_expiring_certifications",
    "set_stale_entities",
    "set_open_alerts",
    "set_critical_alerts",
    "set_high_risk_entities",
    "set_compliance_score",
    "set_freshness_percentage",
    "set_monitored_suppliers",
    # -- Engines (7) --
    "SupplyChainMonitor",
    "DeforestationMonitor",
    "ComplianceChecker",
    "ChangeDetector",
    "RiskScoreMonitor",
    "DataFreshnessValidator",
    "RegulatoryTracker",
    # -- Service Facade --
    "ContinuousMonitoringService",
    "get_service",
    "reset_service",
    "lifespan",
]


# ---------------------------------------------------------------------------
# Lazy import machinery
# ---------------------------------------------------------------------------

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Configuration
    "ContinuousMonitoringConfig": ("config", "ContinuousMonitoringConfig"),
    "get_config": ("config", "get_config"),
    "reset_config": ("config", "reset_config"),
    # Enumerations (14)
    "MonitoringScope": ("models", "MonitoringScope"),
    "ScanStatus": ("models", "ScanStatus"),
    "AlertSeverity": ("models", "AlertSeverity"),
    "AlertStatus": ("models", "AlertStatus"),
    "CertificationStatus": ("models", "CertificationStatus"),
    "ChangeType": ("models", "ChangeType"),
    "ChangeImpact": ("models", "ChangeImpact"),
    "ComplianceStatus": ("models", "ComplianceStatus"),
    "TrendDirection": ("models", "TrendDirection"),
    "FreshnessStatus": ("models", "FreshnessStatus"),
    "RegulatoryImpact": ("models", "RegulatoryImpact"),
    "RiskLevel": ("models", "RiskLevel"),
    "InvestigationStatus": ("models", "InvestigationStatus"),
    "AuditAction": ("models", "AuditAction"),
    # Core Models (15+)
    "SupplyChainScanRecord": ("models", "SupplyChainScanRecord"),
    "DeforestationMonitorRecord": ("models", "DeforestationMonitorRecord"),
    "ComplianceAuditRecord": ("models", "ComplianceAuditRecord"),
    "ChangeDetectionRecord": ("models", "ChangeDetectionRecord"),
    "RiskScoreMonitorRecord": ("models", "RiskScoreMonitorRecord"),
    "DataFreshnessRecord": ("models", "DataFreshnessRecord"),
    "RegulatoryTrackingRecord": ("models", "RegulatoryTrackingRecord"),
    "MonitoringAlert": ("models", "MonitoringAlert"),
    "InvestigationRecord": ("models", "InvestigationRecord"),
    "MonitoringSummary": ("models", "MonitoringSummary"),
    "AuditEntry": ("models", "AuditEntry"),
    "HealthStatus": ("models", "HealthStatus"),
    # Sub-models
    "SupplierChange": ("models", "SupplierChange"),
    "CertificationCheck": ("models", "CertificationCheck"),
    "GeolocationShift": ("models", "GeolocationShift"),
    "DeforestationCorrelation": ("models", "DeforestationCorrelation"),
    "ComplianceCheckItem": ("models", "ComplianceCheckItem"),
    "RiskScoreSnapshot": ("models", "RiskScoreSnapshot"),
    "StaleEntity": ("models", "StaleEntity"),
    "RegulatoryUpdate": ("models", "RegulatoryUpdate"),
    "ActionRecommendation": ("models", "ActionRecommendation"),
    # Constants
    "AGENT_ID": ("models", "AGENT_ID"),
    "AGENT_VERSION": ("models", "AGENT_VERSION"),
    "EUDR_ARTICLES_MONITORED": ("models", "EUDR_ARTICLES_MONITORED"),
    # Provenance
    "ProvenanceTracker": ("provenance", "ProvenanceTracker"),
    "GENESIS_HASH": ("provenance", "GENESIS_HASH"),
    # Metrics (counters - 16)
    "record_supply_chain_scan": ("metrics", "record_supply_chain_scan"),
    "record_supplier_change_detected": ("metrics", "record_supplier_change_detected"),
    "record_certification_expiry": ("metrics", "record_certification_expiry"),
    "record_geolocation_drift": ("metrics", "record_geolocation_drift"),
    "record_deforestation_check": ("metrics", "record_deforestation_check"),
    "record_deforestation_correlation": ("metrics", "record_deforestation_correlation"),
    "record_investigation_triggered": ("metrics", "record_investigation_triggered"),
    "record_compliance_audit": ("metrics", "record_compliance_audit"),
    "record_compliance_check_passed": ("metrics", "record_compliance_check_passed"),
    "record_compliance_check_failed": ("metrics", "record_compliance_check_failed"),
    "record_change_detected": ("metrics", "record_change_detected"),
    "record_risk_monitor": ("metrics", "record_risk_monitor"),
    "record_risk_degradation": ("metrics", "record_risk_degradation"),
    "record_freshness_check": ("metrics", "record_freshness_check"),
    "record_regulatory_update": ("metrics", "record_regulatory_update"),
    "record_alert_generated": ("metrics", "record_alert_generated"),
    # Metrics (histograms - 14)
    "observe_supply_chain_scan_duration": ("metrics", "observe_supply_chain_scan_duration"),
    "observe_deforestation_check_duration": ("metrics", "observe_deforestation_check_duration"),
    "observe_compliance_audit_duration": ("metrics", "observe_compliance_audit_duration"),
    "observe_change_detection_duration": ("metrics", "observe_change_detection_duration"),
    "observe_risk_monitor_duration": ("metrics", "observe_risk_monitor_duration"),
    "observe_freshness_check_duration": ("metrics", "observe_freshness_check_duration"),
    "observe_regulatory_check_duration": ("metrics", "observe_regulatory_check_duration"),
    "observe_impact_assessment_duration": ("metrics", "observe_impact_assessment_duration"),
    "observe_correlation_duration": ("metrics", "observe_correlation_duration"),
    "observe_investigation_trigger_duration": ("metrics", "observe_investigation_trigger_duration"),
    "observe_trend_analysis_duration": ("metrics", "observe_trend_analysis_duration"),
    "observe_freshness_report_duration": ("metrics", "observe_freshness_report_duration"),
    "observe_notification_duration": ("metrics", "observe_notification_duration"),
    "observe_entity_mapping_duration": ("metrics", "observe_entity_mapping_duration"),
    # Metrics (gauges - 10)
    "set_active_scans": ("metrics", "set_active_scans"),
    "set_pending_investigations": ("metrics", "set_pending_investigations"),
    "set_expiring_certifications": ("metrics", "set_expiring_certifications"),
    "set_stale_entities": ("metrics", "set_stale_entities"),
    "set_open_alerts": ("metrics", "set_open_alerts"),
    "set_critical_alerts": ("metrics", "set_critical_alerts"),
    "set_high_risk_entities": ("metrics", "set_high_risk_entities"),
    "set_compliance_score": ("metrics", "set_compliance_score"),
    "set_freshness_percentage": ("metrics", "set_freshness_percentage"),
    "set_monitored_suppliers": ("metrics", "set_monitored_suppliers"),
    # Engines (7)
    "SupplyChainMonitor": ("supply_chain_monitor", "SupplyChainMonitor"),
    "DeforestationMonitor": ("deforestation_monitor", "DeforestationMonitor"),
    "ComplianceChecker": ("compliance_checker", "ComplianceChecker"),
    "ChangeDetector": ("change_detector", "ChangeDetector"),
    "RiskScoreMonitor": ("risk_score_monitor", "RiskScoreMonitor"),
    "DataFreshnessValidator": ("data_freshness_validator", "DataFreshnessValidator"),
    "RegulatoryTracker": ("regulatory_tracker", "RegulatoryTracker"),
    # Service Facade
    "ContinuousMonitoringService": ("setup", "ContinuousMonitoringService"),
    "get_service": ("setup", "get_service"),
    "reset_service": ("setup", "reset_service"),
    "lifespan": ("setup", "lifespan"),
}


def __getattr__(name: str) -> object:
    """Module-level __getattr__ for lazy imports."""
    if name in _LAZY_IMPORTS:
        module_suffix, attr_name = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(
            f"greenlang.agents.eudr.continuous_monitoring.{module_suffix}"
        )
        return getattr(mod, attr_name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def get_version() -> str:
    """Return the current module version string."""
    return __version__


def get_agent_info() -> dict:
    """Return agent identification and capability metadata."""
    return {
        "agent_id": __agent_id__,
        "version": __version__,
        "name": "Continuous Monitoring Agent",
        "prd": "PRD-AGENT-EUDR-033",
        "regulation": "EU 2023/1115 (EUDR)",
        "articles": ["4", "8", "10", "11", "12", "14", "29", "31"],
        "upstream_dependencies": [
            "AGENT-EUDR-020 (Deforestation Alert System)",
            "AGENT-EUDR-028 (Risk Assessment Engine)",
            "AGENT-EUDR-026 (Due Diligence Orchestrator)",
            "AGENT-EUDR-023 (Legal Compliance Verifier)",
        ],
        "engines": [
            "SupplyChainMonitor",
            "DeforestationMonitor",
            "ComplianceChecker",
            "ChangeDetector",
            "RiskScoreMonitor",
            "DataFreshnessValidator",
            "RegulatoryTracker",
        ],
        "engine_count": 7,
        "enum_count": 14,
        "core_model_count": 15,
        "sub_model_count": 9,
        "metrics_count": 40,
        "db_prefix": "gl_eudr_cm_",
        "metrics_prefix": "gl_eudr_cm_",
        "env_prefix": "GL_EUDR_CM_",
    }
