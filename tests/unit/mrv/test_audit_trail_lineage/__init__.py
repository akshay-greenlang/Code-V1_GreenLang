# -*- coding: utf-8 -*-
"""
Test suite for AGENT-MRV-030: Audit Trail & Lineage Agent (GL-MRV-X-042).

Comprehensive tests covering:
- Models (25 enums, 20+ Pydantic/dataclass models)
- Configuration (thread-safe singleton, GL_ATL_ env prefix, 6 sections)
- AuditEventEngine (Engine 1: immutable SHA-256 chain event recording)
- LineageGraphEngine (Engine 2: DAG lineage construction/traversal)
- EvidencePackagerEngine (Engine 3: audit evidence bundling)
- ComplianceTracerEngine (Engine 4: regulatory requirement traceability)
- ChangeDetectorEngine (Engine 5: change tracking and recalculation)
- ComplianceCheckerEngine (Engine 6: multi-framework compliance validation)
- AuditTrailPipelineEngine (Engine 7: 10-stage pipeline orchestration)
- Provenance (SHA-256 chain provenance)
- Setup (service facade)
- API (25+ FastAPI endpoints)

Target: ~900 tests, 85%+ coverage.

Agent ID: GL-MRV-X-042
Component: AGENT-MRV-030
Version: 1.0.0
Table Prefix: gl_atl_

Author: GL-TestEngineer
Date: March 2026
"""
