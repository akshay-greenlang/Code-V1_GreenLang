# -*- coding: utf-8 -*-
"""
Third-Party Audit Manager Setup Facade - AGENT-EUDR-024

Provides initialization, database table creation, and lifecycle management
for the Third-Party Audit Manager Agent. Serves as the single entry point
for agent setup, combining all 7 engines into a unified facade.

Engines Orchestrated:
    1. AuditPlanningSchedulingEngine - Risk-based audit scheduling
    2. AuditorRegistryQualificationEngine - Auditor competence management
    3. AuditExecutionEngine - Checklist and evidence management
    4. NonConformanceDetectionEngine - NC severity classification
    5. CARManagementEngine - CAR lifecycle management
    6. CertificationIntegrationEngine - Scheme integration
    7. AuditAnalyticsEngine - Analytics and authority liaison

Database Tables (11):
    - gl_eudr_tam_audits
    - gl_eudr_tam_auditors
    - gl_eudr_tam_checklists
    - gl_eudr_tam_evidence
    - gl_eudr_tam_non_conformances
    - gl_eudr_tam_root_cause_analyses
    - gl_eudr_tam_cars
    - gl_eudr_tam_certificates
    - gl_eudr_tam_authority_interactions
    - gl_eudr_tam_audit_reports
    - gl_eudr_tam_provenance

Example:
    >>> from greenlang.agents.eudr.third_party_audit_manager.setup import (
    ...     ThirdPartyAuditManagerSetup,
    ... )
    >>> setup = ThirdPartyAuditManagerSetup()
    >>> setup.initialize()
    >>> planning_engine = setup.get_planning_engine()

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-024 Third-Party Audit Manager (GL-EUDR-TAM-024)
Status: Production Ready
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

from greenlang.agents.eudr.third_party_audit_manager.config import (
    ThirdPartyAuditManagerConfig,
    get_config,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database table definitions (for SQL migration reference)
# ---------------------------------------------------------------------------

DATABASE_TABLES: List[Dict[str, str]] = [
    {
        "name": "gl_eudr_tam_audits",
        "description": "Audit lifecycle records",
    },
    {
        "name": "gl_eudr_tam_auditors",
        "description": "Auditor registry profiles",
    },
    {
        "name": "gl_eudr_tam_checklists",
        "description": "Audit checklists with criteria",
    },
    {
        "name": "gl_eudr_tam_evidence",
        "description": "Audit evidence items",
    },
    {
        "name": "gl_eudr_tam_non_conformances",
        "description": "Non-conformance findings",
    },
    {
        "name": "gl_eudr_tam_root_cause_analyses",
        "description": "Root cause analysis records",
    },
    {
        "name": "gl_eudr_tam_cars",
        "description": "Corrective action requests",
    },
    {
        "name": "gl_eudr_tam_certificates",
        "description": "Certification scheme certificates",
    },
    {
        "name": "gl_eudr_tam_authority_interactions",
        "description": "Competent authority interactions",
    },
    {
        "name": "gl_eudr_tam_audit_reports",
        "description": "ISO 19011 audit reports",
    },
    {
        "name": "gl_eudr_tam_provenance",
        "description": "Provenance chain records",
    },
]


class ThirdPartyAuditManagerSetup:
    """Facade for Third-Party Audit Manager agent initialization.

    Provides a unified entry point for initializing all 7 processing
    engines, database tables, and runtime configuration.

    Attributes:
        config: Agent configuration.
        _planning_engine: Lazy-initialized planning engine.
        _auditor_engine: Lazy-initialized auditor engine.
        _execution_engine: Lazy-initialized execution engine.
        _nc_engine: Lazy-initialized NC detection engine.
        _car_engine: Lazy-initialized CAR management engine.
        _cert_engine: Lazy-initialized certification engine.
        _analytics_engine: Lazy-initialized analytics engine.
    """

    def __init__(
        self,
        config: Optional[ThirdPartyAuditManagerConfig] = None,
    ) -> None:
        """Initialize the setup facade.

        Args:
            config: Optional configuration override.
        """
        self.config = config or get_config()
        self._initialized = False

        # Lazy engine references
        self._planning_engine = None
        self._auditor_engine = None
        self._execution_engine = None
        self._nc_engine = None
        self._car_engine = None
        self._cert_engine = None
        self._analytics_engine = None

        logger.info("ThirdPartyAuditManagerSetup created")

    def initialize(self) -> None:
        """Initialize all engines and verify configuration.

        Creates engine instances and validates that all dependencies
        are available. This method is idempotent.
        """
        if self._initialized:
            logger.debug("ThirdPartyAuditManagerSetup already initialized")
            return

        logger.info("Initializing Third-Party Audit Manager agent...")

        # Initialize all engines
        self._planning_engine = self._create_planning_engine()
        self._auditor_engine = self._create_auditor_engine()
        self._execution_engine = self._create_execution_engine()
        self._nc_engine = self._create_nc_engine()
        self._car_engine = self._create_car_engine()
        self._cert_engine = self._create_cert_engine()
        self._analytics_engine = self._create_analytics_engine()

        self._initialized = True

        logger.info(
            "Third-Party Audit Manager agent initialized: "
            f"7 engines, {len(DATABASE_TABLES)} tables, "
            f"schemes={len(self.config.enabled_schemes)}"
        )

    def get_planning_engine(self):
        """Get the audit planning and scheduling engine.

        Returns:
            AuditPlanningSchedulingEngine instance.
        """
        if self._planning_engine is None:
            self._planning_engine = self._create_planning_engine()
        return self._planning_engine

    def get_auditor_engine(self):
        """Get the auditor registry and qualification engine.

        Returns:
            AuditorRegistryQualificationEngine instance.
        """
        if self._auditor_engine is None:
            self._auditor_engine = self._create_auditor_engine()
        return self._auditor_engine

    def get_execution_engine(self):
        """Get the audit execution engine.

        Returns:
            AuditExecutionEngine instance.
        """
        if self._execution_engine is None:
            self._execution_engine = self._create_execution_engine()
        return self._execution_engine

    def get_nc_engine(self):
        """Get the non-conformance detection engine.

        Returns:
            NonConformanceDetectionEngine instance.
        """
        if self._nc_engine is None:
            self._nc_engine = self._create_nc_engine()
        return self._nc_engine

    def get_car_engine(self):
        """Get the CAR management engine.

        Returns:
            CARManagementEngine instance.
        """
        if self._car_engine is None:
            self._car_engine = self._create_car_engine()
        return self._car_engine

    def get_cert_engine(self):
        """Get the certification integration engine.

        Returns:
            CertificationIntegrationEngine instance.
        """
        if self._cert_engine is None:
            self._cert_engine = self._create_cert_engine()
        return self._cert_engine

    def get_analytics_engine(self):
        """Get the audit analytics engine.

        Returns:
            AuditAnalyticsEngine instance.
        """
        if self._analytics_engine is None:
            self._analytics_engine = self._create_analytics_engine()
        return self._analytics_engine

    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent metadata information.

        Returns:
            Dictionary with agent metadata.
        """
        return {
            "agent_id": "GL-EUDR-TAM-024",
            "agent_name": "Third-Party Audit Manager",
            "version": "1.0.0",
            "prd": "AGENT-EUDR-024",
            "regulation": "EU 2023/1115 (EUDR)",
            "iso_standards": [
                "ISO 19011:2018",
                "ISO/IEC 17065:2012",
                "ISO/IEC 17021-1:2015",
            ],
            "engines": [
                "AuditPlanningSchedulingEngine",
                "AuditorRegistryQualificationEngine",
                "AuditExecutionEngine",
                "NonConformanceDetectionEngine",
                "CARManagementEngine",
                "CertificationIntegrationEngine",
                "AuditAnalyticsEngine",
            ],
            "features": [
                "F1: Audit Planning and Scheduling",
                "F2: Auditor Registry and Qualification",
                "F3: Audit Execution and Monitoring",
                "F4: Non-Conformance Detection and Classification",
                "F5: Corrective Action Request Management",
                "F6: Certification Scheme Integration",
                "F7: ISO 19011 Report Generation",
                "F8: Competent Authority Liaison",
                "F9: Audit Analytics and Dashboards",
            ],
            "certification_schemes": self.config.enabled_schemes,
            "database_tables": len(DATABASE_TABLES),
            "initialized": self._initialized,
        }

    def get_table_definitions(self) -> List[Dict[str, str]]:
        """Get database table definitions for this agent.

        Returns:
            List of table definition dictionaries.
        """
        return DATABASE_TABLES.copy()

    def _create_planning_engine(self):
        """Create the audit planning and scheduling engine."""
        from greenlang.agents.eudr.third_party_audit_manager.audit_planning_scheduling_engine import (
            AuditPlanningSchedulingEngine,
        )
        return AuditPlanningSchedulingEngine(config=self.config)

    def _create_auditor_engine(self):
        """Create the auditor registry and qualification engine."""
        from greenlang.agents.eudr.third_party_audit_manager.auditor_registry_qualification_engine import (
            AuditorRegistryQualificationEngine,
        )
        return AuditorRegistryQualificationEngine(config=self.config)

    def _create_execution_engine(self):
        """Create the audit execution engine."""
        from greenlang.agents.eudr.third_party_audit_manager.audit_execution_engine import (
            AuditExecutionEngine,
        )
        return AuditExecutionEngine(config=self.config)

    def _create_nc_engine(self):
        """Create the non-conformance detection engine."""
        from greenlang.agents.eudr.third_party_audit_manager.non_conformance_detection_engine import (
            NonConformanceDetectionEngine,
        )
        return NonConformanceDetectionEngine(config=self.config)

    def _create_car_engine(self):
        """Create the CAR management engine."""
        from greenlang.agents.eudr.third_party_audit_manager.car_management_engine import (
            CARManagementEngine,
        )
        return CARManagementEngine(config=self.config)

    def _create_cert_engine(self):
        """Create the certification integration engine."""
        from greenlang.agents.eudr.third_party_audit_manager.certification_integration_engine import (
            CertificationIntegrationEngine,
        )
        return CertificationIntegrationEngine(config=self.config)

    def _create_analytics_engine(self):
        """Create the audit analytics engine."""
        from greenlang.agents.eudr.third_party_audit_manager.audit_analytics_engine import (
            AuditAnalyticsEngine,
        )
        return AuditAnalyticsEngine(config=self.config)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_setup_lock = threading.Lock()
_global_setup: Optional[ThirdPartyAuditManagerSetup] = None


def get_setup() -> ThirdPartyAuditManagerSetup:
    """Get the global ThirdPartyAuditManagerSetup singleton.

    Returns:
        ThirdPartyAuditManagerSetup singleton instance.
    """
    global _global_setup
    if _global_setup is None:
        with _setup_lock:
            if _global_setup is None:
                _global_setup = ThirdPartyAuditManagerSetup()
    return _global_setup


def reset_setup() -> None:
    """Reset the global setup singleton (for testing only)."""
    global _global_setup
    with _setup_lock:
        _global_setup = None
        logger.warning("ThirdPartyAuditManagerSetup reset (testing only)")
