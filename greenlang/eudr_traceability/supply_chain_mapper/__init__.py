# -*- coding: utf-8 -*-
"""
AGENT-EUDR-001: Supply Chain Mapping Master -- Sub-package
==========================================================

This sub-package implements the Supply Chain Mapping Master Agent
for EUDR (Regulation (EU) 2023/1115) compliance. It provides graph-native
supply chain modeling, multi-tier recursive mapping, batch traceability,
risk propagation, gap analysis, and visualization capabilities.

Modules:
    batch_traceability: Many-to-many batch traceability engine (Feature 4)

Agent ID: GL-EUDR-SCM-001
PRD: PRD-AGENT-EUDR-001
"""

__version__ = "1.0.0"
__agent_id__ = "GL-EUDR-SCM-001"
__agent_name__ = "EUDR Supply Chain Mapping Master Agent"

from greenlang.eudr_traceability.supply_chain_mapper.batch_traceability import (
    BatchTraceabilityEngine,
    BatchOperation,
    BatchOperationType,
    TraceResult,
    MassBalanceResult,
    ComplianceAlert,
    TraceabilityScore,
)

__all__ = [
    "__version__",
    "__agent_id__",
    "__agent_name__",
    "BatchTraceabilityEngine",
    "BatchOperation",
    "BatchOperationType",
    "TraceResult",
    "MassBalanceResult",
    "ComplianceAlert",
    "TraceabilityScore",
]
