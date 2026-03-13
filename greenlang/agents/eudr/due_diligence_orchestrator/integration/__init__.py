# -*- coding: utf-8 -*-
"""
Integration Package for Due Diligence Orchestrator - AGENT-EUDR-026

Provides HTTP client wrappers for invoking all 25 upstream EUDR agents
(EUDR-001 through EUDR-025) and event bus integration for real-time
workflow notifications.

Modules:
    - agent_client: Generic async HTTP client for calling any EUDR agent
    - supply_chain_clients: Convenience wrappers for Phase 1 agents (EUDR-001 to 015)
    - risk_assessment_clients: Convenience wrappers for Phase 2 agents (EUDR-016 to 025)
    - event_bus: Event bus integration for workflow state change notifications

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-026 Due Diligence Orchestrator (GL-EUDR-DDO-026)
Status: Production Ready
"""

from __future__ import annotations

__all__: list[str] = []
