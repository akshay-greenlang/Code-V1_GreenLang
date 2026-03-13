# -*- coding: utf-8 -*-
"""
Route Protector - JWT Authentication Service (SEC-001)

Provides utilities to apply authentication and authorization requirements
to existing FastAPI routers without modifying route handler source code.
Works by wrapping route handlers with dependency injection for auth
context validation, permission checking, and tenant isolation.

Features:
    - AuthDependency: extracts and validates JWT / API-key credentials.
    - PermissionDependency: enforces fine-grained permission strings.
    - TenantDependency: ensures tenant isolation across requests.
    - protect_router(): bulk-wraps every route on an existing APIRouter.
    - Decorators: require_auth, require_permissions, require_roles,
      require_tenant for use on individual route handlers.
    - Wildcard permission matching (e.g. ``agents:*`` grants ``agents:execute``).
    - Public-path allow-list for health, docs, and auth endpoints.

Security Compliance:
    - SOC 2 CC6.1 (Logical Access)
    - ISO 27001 A.9.4 (System and Application Access Control)

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import fnmatch
import logging
import re
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.routing import APIRoute

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public endpoints that never require authentication
# ---------------------------------------------------------------------------

PUBLIC_PATHS: Set[str] = {
    "/health",
    "/healthz",
    "/readyz",
    "/livez",
    "/metrics",
    "/auth/login",
    "/auth/token",
    "/auth/refresh",
    "/auth/jwks",
    "/auth/password/reset",
    "/docs",
    "/openapi.json",
    "/redoc",
}

# ---------------------------------------------------------------------------
# Endpoint -> permission map
# ---------------------------------------------------------------------------

PERMISSION_MAP: Dict[str, str] = {
    # agents routes (/api/v1/agents)
    "GET:/api/v1/agents": "agents:list",
    "GET:/api/v1/agents/{agent_id}": "agents:read",
    "POST:/api/v1/agents/{agent_id}/execute": "agents:execute",
    "PATCH:/api/v1/agents/{agent_id}/config": "agents:configure",
    # emissions routes (/api/v1/emissions)
    "GET:/api/v1/emissions": "emissions:list",
    "POST:/api/v1/emissions/calculate": "emissions:calculate",
    "GET:/api/v1/emissions/{emission_id}": "emissions:read",
    # jobs routes (/api/v1/jobs)
    "GET:/api/v1/jobs": "jobs:list",
    "GET:/api/v1/jobs/{job_id}": "jobs:read",
    "POST:/api/v1/jobs/{job_id}/cancel": "jobs:cancel",
    # compliance routes (/api/v1/compliance)
    "GET:/api/v1/compliance/reports": "compliance:list",
    "POST:/api/v1/compliance/reports": "compliance:create",
    "GET:/api/v1/compliance/reports/{report_id}": "compliance:read",
    # factory routes (/api/v1/factory)
    "GET:/api/v1/factory/agents": "factory:list",
    "POST:/api/v1/factory/agents": "factory:create",
    "GET:/api/v1/factory/agents/{key}": "factory:read",
    "PUT:/api/v1/factory/agents/{key}": "factory:update",
    "DELETE:/api/v1/factory/agents/{key}": "factory:delete",
    "POST:/api/v1/factory/agents/{key}/execute": "factory:execute",
    "GET:/api/v1/factory/agents/{key}/metrics": "factory:metrics",
    "POST:/api/v1/factory/agents/batch-execute": "factory:execute",
    "POST:/api/v1/factory/agents/{key}/deploy": "factory:deploy",
    "POST:/api/v1/factory/agents/{key}/rollback": "factory:rollback",
    # flag routes (/api/v1/flags)
    "GET:/api/v1/flags": "flags:list",
    "POST:/api/v1/flags": "flags:create",
    "GET:/api/v1/flags/{flag_key}": "flags:read",
    "PUT:/api/v1/flags/{flag_key}": "flags:update",
    "DELETE:/api/v1/flags/{flag_key}": "flags:delete",
    "POST:/api/v1/flags/{flag_key}/evaluate": "flags:evaluate",
    "POST:/api/v1/flags/evaluate-batch": "flags:evaluate",
    "PUT:/api/v1/flags/{flag_key}/rollout": "flags:rollout",
    "POST:/api/v1/flags/{flag_key}/kill": "flags:kill",
    "POST:/api/v1/flags/{flag_key}/restore": "flags:restore",
    # auth admin routes (/auth/admin)
    "GET:/auth/admin/users": "admin:users:list",
    "GET:/auth/admin/users/{user_id}": "admin:users:read",
    "POST:/auth/admin/users/{user_id}/unlock": "admin:users:unlock",
    "POST:/auth/admin/users/{user_id}/revoke-tokens": "admin:users:revoke",
    "POST:/auth/admin/users/{user_id}/force-password-reset": "admin:users:reset",
    "POST:/auth/admin/users/{user_id}/disable-mfa": "admin:users:mfa",
    "GET:/auth/admin/sessions": "admin:sessions:list",
    "DELETE:/auth/admin/sessions/{session_id}": "admin:sessions:terminate",
    "GET:/auth/admin/audit-log": "admin:audit:read",
    "GET:/auth/admin/lockouts": "admin:lockouts:list",
    # rbac routes (/api/v1/rbac) - SEC-002
    "GET:/api/v1/rbac/roles": "rbac:roles:list",
    "POST:/api/v1/rbac/roles": "rbac:roles:create",
    "GET:/api/v1/rbac/roles/{role_id}": "rbac:roles:read",
    "PUT:/api/v1/rbac/roles/{role_id}": "rbac:roles:update",
    "DELETE:/api/v1/rbac/roles/{role_id}": "rbac:roles:delete",
    "GET:/api/v1/rbac/roles/{role_id}/permissions": "rbac:roles:read",
    "POST:/api/v1/rbac/roles/{role_id}/permissions": "rbac:permissions:grant",
    "DELETE:/api/v1/rbac/roles/{role_id}/permissions/{perm_id}": "rbac:permissions:revoke",
    "GET:/api/v1/rbac/permissions": "rbac:permissions:list",
    "GET:/api/v1/rbac/assignments": "rbac:assignments:list",
    "POST:/api/v1/rbac/assignments": "rbac:assignments:create",
    "DELETE:/api/v1/rbac/assignments/{assignment_id}": "rbac:assignments:revoke",
    "GET:/api/v1/rbac/users/{user_id}/roles": "rbac:assignments:list",
    "GET:/api/v1/rbac/users/{user_id}/permissions": "rbac:check",
    "POST:/api/v1/rbac/check": "rbac:check",
    # encryption routes (/api/v1/encryption) - SEC-003
    "POST:/api/v1/encryption/encrypt": "encryption:encrypt",
    "POST:/api/v1/encryption/decrypt": "encryption:decrypt",
    "GET:/api/v1/encryption/keys": "encryption:admin",
    "POST:/api/v1/encryption/keys/rotate": "encryption:admin",
    "GET:/api/v1/encryption/audit": "encryption:audit",
    "GET:/api/v1/encryption/status": "encryption:read",
    # audit routes (/api/v1/audit) - SEC-005
    "GET:/api/v1/audit/events": "audit:read",
    "GET:/api/v1/audit/events/{event_id}": "audit:read",
    "POST:/api/v1/audit/search": "audit:search",
    "GET:/api/v1/audit/stats": "audit:read",
    "GET:/api/v1/audit/timeline": "audit:read",
    "GET:/api/v1/audit/hotspots": "audit:read",
    "POST:/api/v1/audit/export": "audit:export",
    "GET:/api/v1/audit/export/{job_id}": "audit:export",
    "GET:/api/v1/audit/export/{job_id}/download": "audit:export",
    "POST:/api/v1/audit/reports/soc2": "audit:admin",
    "POST:/api/v1/audit/reports/iso27001": "audit:admin",
    "POST:/api/v1/audit/reports/gdpr": "audit:admin",
    "GET:/api/v1/audit/reports/{job_id}": "audit:admin",
    "GET:/api/v1/audit/reports/{job_id}/download": "audit:admin",
    # secrets routes (/api/v1/secrets) - SEC-006
    "GET:/api/v1/secrets": "secrets:list",
    "GET:/api/v1/secrets/{path:path}": "secrets:read",
    "POST:/api/v1/secrets/{path:path}": "secrets:write",
    "PUT:/api/v1/secrets/{path:path}": "secrets:write",
    "DELETE:/api/v1/secrets/{path:path}": "secrets:admin",
    "GET:/api/v1/secrets/{path:path}/versions": "secrets:read",
    "POST:/api/v1/secrets/{path:path}/undelete": "secrets:admin",
    "POST:/api/v1/secrets/rotate/{path:path}": "secrets:rotate",
    "GET:/api/v1/secrets/rotation/status": "secrets:read",
    "GET:/api/v1/secrets/rotation/schedule": "secrets:read",
    "POST:/api/v1/secrets/rotation/schedule": "secrets:admin",
    "GET:/api/v1/secrets/health": "secrets:read",
    "GET:/api/v1/secrets/status": "secrets:read",
    "GET:/api/v1/secrets/stats": "secrets:read",
    # security scanning routes (/api/v1/security) - SEC-007
    "GET:/api/v1/security/vulnerabilities": "security:read",
    "GET:/api/v1/security/vulnerabilities/{id}": "security:read",
    "POST:/api/v1/security/vulnerabilities/{id}/accept": "security:admin",
    "POST:/api/v1/security/vulnerabilities/{id}/remediate": "security:write",
    "GET:/api/v1/security/vulnerabilities/stats": "security:read",
    "POST:/api/v1/security/scans": "security:scan",
    "GET:/api/v1/security/scans": "security:read",
    "GET:/api/v1/security/scans/{id}": "security:read",
    "GET:/api/v1/security/scans/{id}/findings": "security:read",
    "GET:/api/v1/security/dashboard": "security:read",
    "GET:/api/v1/security/dashboard/trends": "security:read",
    "GET:/api/v1/security/dashboard/coverage": "security:read",
    "GET:/api/v1/security/dashboard/sla": "security:read",
    "GET:/api/v1/security/compliance": "security:read",
    "POST:/api/v1/security/compliance/reports": "security:admin",
    "GET:/api/v1/security/compliance/evidence": "security:read",
    # SOC 2 preparation routes (/api/v1/soc2) - SEC-009
    # Assessment routes
    "GET:/api/v1/soc2/assessment": "soc2:assessment:read",
    "POST:/api/v1/soc2/assessment/run": "soc2:assessment:write",
    "GET:/api/v1/soc2/assessment/score": "soc2:assessment:read",
    "GET:/api/v1/soc2/assessment/gaps": "soc2:assessment:read",
    "GET:/api/v1/soc2/assessment/criteria": "soc2:assessment:read",
    "PUT:/api/v1/soc2/assessment/criteria/{criterion_id}": "soc2:assessment:write",
    # Evidence routes
    "GET:/api/v1/soc2/evidence": "soc2:evidence:read",
    "GET:/api/v1/soc2/evidence/sources": "soc2:evidence:read",
    "GET:/api/v1/soc2/evidence/{criterion}": "soc2:evidence:read",
    "POST:/api/v1/soc2/evidence/collect": "soc2:evidence:write",
    "POST:/api/v1/soc2/evidence/package": "soc2:evidence:package",
    "GET:/api/v1/soc2/evidence/package/{package_id}": "soc2:evidence:read",
    # Testing routes
    "GET:/api/v1/soc2/tests": "soc2:tests:read",
    "GET:/api/v1/soc2/tests/runs": "soc2:tests:read",
    "GET:/api/v1/soc2/tests/runs/{run_id}": "soc2:tests:read",
    "GET:/api/v1/soc2/tests/report": "soc2:tests:read",
    "POST:/api/v1/soc2/tests/run": "soc2:tests:execute",
    "GET:/api/v1/soc2/tests/{test_id}/result": "soc2:tests:read",
    # Portal routes (auditor access)
    "GET:/api/v1/soc2/portal/evidence": "soc2:portal:access",
    "GET:/api/v1/soc2/portal/requests": "soc2:portal:access",
    "POST:/api/v1/soc2/portal/requests": "soc2:portal:access",
    "GET:/api/v1/soc2/portal/requests/{request_id}": "soc2:portal:access",
    "GET:/api/v1/soc2/portal/download/{evidence_id}": "soc2:portal:access",
    "GET:/api/v1/soc2/portal/activity": "soc2:portal:access",
    "GET:/api/v1/soc2/portal/session": "soc2:portal:access",
    # Findings routes
    "GET:/api/v1/soc2/findings": "soc2:findings:read",
    "GET:/api/v1/soc2/findings/summary": "soc2:findings:read",
    "POST:/api/v1/soc2/findings": "soc2:findings:manage",
    "GET:/api/v1/soc2/findings/{finding_id}": "soc2:findings:read",
    "PUT:/api/v1/soc2/findings/{finding_id}": "soc2:findings:manage",
    "POST:/api/v1/soc2/findings/{finding_id}/remediation": "soc2:findings:manage",
    "GET:/api/v1/soc2/findings/{finding_id}/remediation": "soc2:findings:read",
    "PUT:/api/v1/soc2/findings/{finding_id}/close": "soc2:findings:manage",
    # Attestation routes
    "GET:/api/v1/soc2/attestations": "soc2:attestations:read",
    "POST:/api/v1/soc2/attestations": "soc2:attestations:read",
    "GET:/api/v1/soc2/attestations/{attestation_id}": "soc2:attestations:read",
    "POST:/api/v1/soc2/attestations/{attestation_id}/submit": "soc2:attestations:read",
    "POST:/api/v1/soc2/attestations/{attestation_id}/sign": "soc2:attestations:sign",
    "GET:/api/v1/soc2/attestations/{attestation_id}/status": "soc2:attestations:read",
    "POST:/api/v1/soc2/attestations/{attestation_id}/remind": "soc2:attestations:sign",
    # Project routes
    "GET:/api/v1/soc2/project": "soc2:project:read",
    "POST:/api/v1/soc2/project": "soc2:project:manage",
    "GET:/api/v1/soc2/project/summary": "soc2:project:read",
    "GET:/api/v1/soc2/project/timeline": "soc2:project:read",
    "GET:/api/v1/soc2/project/milestones": "soc2:project:read",
    "POST:/api/v1/soc2/project/milestones": "soc2:project:manage",
    "PUT:/api/v1/soc2/project/milestones/{milestone_id}": "soc2:project:manage",
    # Dashboard routes
    "GET:/api/v1/soc2/dashboard/summary": "soc2:dashboard:view",
    "GET:/api/v1/soc2/dashboard/timeline": "soc2:dashboard:view",
    "GET:/api/v1/soc2/dashboard/metrics": "soc2:dashboard:view",
    "GET:/api/v1/soc2/dashboard/health": "soc2:dashboard:view",
    # ==========================================================================
    # Security Operations routes (/api/v1/secops) - SEC-010
    # ==========================================================================
    # Incident Response routes
    "GET:/api/v1/secops/incidents": "secops:incidents:read",
    "GET:/api/v1/secops/incidents/{id}": "secops:incidents:read",
    "POST:/api/v1/secops/incidents/{id}/acknowledge": "secops:incidents:manage",
    "POST:/api/v1/secops/incidents/{id}/assign": "secops:incidents:manage",
    "POST:/api/v1/secops/incidents/{id}/execute-playbook": "secops:playbooks:execute",
    "PUT:/api/v1/secops/incidents/{id}/resolve": "secops:incidents:manage",
    "PUT:/api/v1/secops/incidents/{id}/close": "secops:incidents:manage",
    "GET:/api/v1/secops/incidents/{id}/timeline": "secops:incidents:read",
    "GET:/api/v1/secops/incidents/metrics": "secops:incidents:read",
    # Threat Modeling routes
    "GET:/api/v1/secops/threats": "secops:threats:read",
    "POST:/api/v1/secops/threats": "secops:threats:write",
    "GET:/api/v1/secops/threats/{id}": "secops:threats:read",
    "PUT:/api/v1/secops/threats/{id}": "secops:threats:write",
    "DELETE:/api/v1/secops/threats/{id}": "secops:threats:write",
    "POST:/api/v1/secops/threats/{id}/analyze": "secops:threats:write",
    "POST:/api/v1/secops/threats/{id}/components": "secops:threats:write",
    "POST:/api/v1/secops/threats/{id}/data-flows": "secops:threats:write",
    "PUT:/api/v1/secops/threats/{id}/approve": "secops:threats:write",
    "GET:/api/v1/secops/threats/{id}/report": "secops:threats:read",
    # WAF Management routes
    "GET:/api/v1/secops/waf/rules": "secops:waf:read",
    "POST:/api/v1/secops/waf/rules": "secops:waf:manage",
    "GET:/api/v1/secops/waf/rules/{id}": "secops:waf:read",
    "PUT:/api/v1/secops/waf/rules/{id}": "secops:waf:manage",
    "DELETE:/api/v1/secops/waf/rules/{id}": "secops:waf:manage",
    "POST:/api/v1/secops/waf/rules/{id}/test": "secops:waf:manage",
    "POST:/api/v1/secops/waf/rules/{id}/deploy": "secops:waf:manage",
    "GET:/api/v1/secops/waf/attacks": "secops:waf:read",
    "POST:/api/v1/secops/waf/attacks/{id}/mitigate": "secops:waf:manage",
    "GET:/api/v1/secops/waf/metrics": "secops:waf:read",
    # Vulnerability Disclosure Program routes
    "POST:/api/v1/secops/vdp/submit": "secops:vdp:read",
    "GET:/api/v1/secops/vdp/submissions": "secops:vdp:read",
    "GET:/api/v1/secops/vdp/submissions/{id}": "secops:vdp:read",
    "PUT:/api/v1/secops/vdp/submissions/{id}/triage": "secops:vdp:manage",
    "PUT:/api/v1/secops/vdp/submissions/{id}/confirm": "secops:vdp:manage",
    "PUT:/api/v1/secops/vdp/submissions/{id}/close": "secops:vdp:manage",
    "POST:/api/v1/secops/vdp/submissions/{id}/bounty": "secops:vdp:manage",
    "GET:/api/v1/secops/vdp/hall-of-fame": "secops:vdp:read",
    # Compliance Automation routes
    "GET:/api/v1/secops/compliance/status": "secops:compliance:read",
    "GET:/api/v1/secops/compliance/iso27001": "secops:compliance:read",
    "GET:/api/v1/secops/compliance/iso27001/soa": "secops:compliance:read",
    "GET:/api/v1/secops/compliance/gdpr": "secops:compliance:read",
    "GET:/api/v1/secops/compliance/pci-dss": "secops:compliance:read",
    "POST:/api/v1/secops/compliance/assess": "secops:compliance:manage",
    # DSAR routes
    "POST:/api/v1/secops/dsar": "secops:dsar:read",
    "GET:/api/v1/secops/dsar": "secops:dsar:read",
    "GET:/api/v1/secops/dsar/{id}": "secops:dsar:read",
    "POST:/api/v1/secops/dsar/{id}/verify": "secops:dsar:process",
    "POST:/api/v1/secops/dsar/{id}/execute": "secops:dsar:process",
    "GET:/api/v1/secops/dsar/{id}/download": "secops:dsar:read",
    # Consent routes
    "POST:/api/v1/secops/consent": "secops:compliance:manage",
    "DELETE:/api/v1/secops/consent/{id}": "secops:compliance:manage",
    "GET:/api/v1/secops/consent/user/{user_id}": "secops:compliance:read",
    # Security Training routes
    "GET:/api/v1/secops/training/courses": "secops:training:read",
    "GET:/api/v1/secops/training/courses/{id}": "secops:training:read",
    "GET:/api/v1/secops/training/my-progress": "secops:training:read",
    "POST:/api/v1/secops/training/courses/{id}/start": "secops:training:read",
    "POST:/api/v1/secops/training/courses/{id}/complete": "secops:training:read",
    "POST:/api/v1/secops/training/courses/{id}/assessment": "secops:training:read",
    "GET:/api/v1/secops/training/certificates": "secops:training:read",
    "GET:/api/v1/secops/training/team-compliance": "secops:training:manage",
    # Phishing Simulation routes
    "POST:/api/v1/secops/phishing/campaigns": "secops:phishing:manage",
    "GET:/api/v1/secops/phishing/campaigns": "secops:phishing:manage",
    "GET:/api/v1/secops/phishing/campaigns/{id}": "secops:phishing:manage",
    "POST:/api/v1/secops/phishing/campaigns/{id}/send": "secops:phishing:manage",
    "GET:/api/v1/secops/phishing/campaigns/{id}/metrics": "secops:phishing:manage",
    # Security Score routes
    "GET:/api/v1/secops/security-score": "secops:training:read",
    "GET:/api/v1/secops/security-score/leaderboard": "secops:training:read",
    # ==========================================================================
    # PII Service routes (/api/v1/pii) - SEC-011
    # ==========================================================================
    # Detection and redaction routes
    "POST:/api/v1/pii/detect": "pii:detect",
    "POST:/api/v1/pii/redact": "pii:redact",
    # Tokenization routes
    "POST:/api/v1/pii/tokenize": "pii:tokenize",
    "POST:/api/v1/pii/detokenize": "pii:detokenize",
    # Policy management routes
    "GET:/api/v1/pii/policies": "pii:policies:read",
    "PUT:/api/v1/pii/policies/{pii_type}": "pii:policies:write",
    # Allowlist management routes
    "GET:/api/v1/pii/allowlist": "pii:allowlist:read",
    "POST:/api/v1/pii/allowlist": "pii:allowlist:write",
    "DELETE:/api/v1/pii/allowlist/{id}": "pii:allowlist:write",
    # Quarantine management routes
    "GET:/api/v1/pii/quarantine": "pii:quarantine:read",
    "POST:/api/v1/pii/quarantine/{id}/release": "pii:quarantine:manage",
    "POST:/api/v1/pii/quarantine/{id}/delete": "pii:quarantine:manage",
    # Metrics route
    "GET:/api/v1/pii/metrics": "pii:audit:read",
    # ==========================================================================
    # Duplicate Detection Agent routes (/api/v1/dedup) - AGENT-DATA-011
    # ==========================================================================
    "POST:/api/v1/dedup/jobs": "dedup:jobs:create",
    "GET:/api/v1/dedup/jobs": "dedup:jobs:read",
    "GET:/api/v1/dedup/jobs/{id}": "dedup:jobs:read",
    "DELETE:/api/v1/dedup/jobs/{id}": "dedup:jobs:delete",
    "POST:/api/v1/dedup/fingerprint": "dedup:fingerprint:execute",
    "POST:/api/v1/dedup/block": "dedup:block:execute",
    "POST:/api/v1/dedup/compare": "dedup:compare:execute",
    "POST:/api/v1/dedup/classify": "dedup:classify:execute",
    "GET:/api/v1/dedup/matches": "dedup:matches:read",
    "GET:/api/v1/dedup/matches/{id}": "dedup:matches:read",
    "POST:/api/v1/dedup/clusters": "dedup:clusters:create",
    "GET:/api/v1/dedup/clusters": "dedup:clusters:read",
    "GET:/api/v1/dedup/clusters/{id}": "dedup:clusters:read",
    "POST:/api/v1/dedup/merge": "dedup:merge:execute",
    "GET:/api/v1/dedup/merge/{id}": "dedup:merge:read",
    "POST:/api/v1/dedup/pipeline": "dedup:pipeline:execute",
    "POST:/api/v1/dedup/rules": "dedup:rules:manage",
    "GET:/api/v1/dedup/rules": "dedup:rules:read",
    "GET:/api/v1/dedup/health": "dedup:health:read",
    "GET:/api/v1/dedup/stats": "dedup:stats:read",
    # ==========================================================================
    # Missing Value Imputer Agent routes (/api/v1/imputer) - AGENT-DATA-012
    # ==========================================================================
    "POST:/api/v1/imputer/jobs": "imputer:jobs:create",
    "GET:/api/v1/imputer/jobs": "imputer:jobs:read",
    "GET:/api/v1/imputer/jobs/{id}": "imputer:jobs:read",
    "DELETE:/api/v1/imputer/jobs/{id}": "imputer:jobs:delete",
    "POST:/api/v1/imputer/analyze": "imputer:analyze:execute",
    "GET:/api/v1/imputer/analyze/{id}": "imputer:analyze:read",
    "POST:/api/v1/imputer/impute": "imputer:impute:execute",
    "POST:/api/v1/imputer/impute/batch": "imputer:impute:execute",
    "GET:/api/v1/imputer/results/{id}": "imputer:results:read",
    "POST:/api/v1/imputer/validate": "imputer:validate:execute",
    "GET:/api/v1/imputer/validate/{id}": "imputer:validate:read",
    "POST:/api/v1/imputer/rules": "imputer:rules:manage",
    "GET:/api/v1/imputer/rules": "imputer:rules:read",
    "PUT:/api/v1/imputer/rules/{id}": "imputer:rules:manage",
    "DELETE:/api/v1/imputer/rules/{id}": "imputer:rules:manage",
    "POST:/api/v1/imputer/templates": "imputer:templates:manage",
    "GET:/api/v1/imputer/templates": "imputer:templates:read",
    "POST:/api/v1/imputer/pipeline": "imputer:pipeline:execute",
    "GET:/api/v1/imputer/health": "imputer:health:read",
    "GET:/api/v1/imputer/stats": "imputer:stats:read",
    # ==========================================================================
    # Outlier Detection Agent routes (/api/v1/outlier) - AGENT-DATA-013
    # ==========================================================================
    "POST:/api/v1/outlier/jobs": "outlier:jobs:create",
    "GET:/api/v1/outlier/jobs": "outlier:jobs:read",
    "GET:/api/v1/outlier/jobs/{id}": "outlier:jobs:read",
    "DELETE:/api/v1/outlier/jobs/{id}": "outlier:jobs:delete",
    "POST:/api/v1/outlier/detect": "outlier:detect:execute",
    "POST:/api/v1/outlier/detect/batch": "outlier:detect:execute",
    "GET:/api/v1/outlier/detections": "outlier:detections:read",
    "GET:/api/v1/outlier/detections/{id}": "outlier:detections:read",
    "POST:/api/v1/outlier/classify": "outlier:classify:execute",
    "GET:/api/v1/outlier/classifications/{id}": "outlier:classifications:read",
    "POST:/api/v1/outlier/treat": "outlier:treat:execute",
    "GET:/api/v1/outlier/treatments/{id}": "outlier:treatments:read",
    "POST:/api/v1/outlier/treatments/{id}/undo": "outlier:treat:execute",
    "POST:/api/v1/outlier/thresholds": "outlier:thresholds:manage",
    "GET:/api/v1/outlier/thresholds": "outlier:thresholds:read",
    "POST:/api/v1/outlier/feedback": "outlier:feedback:create",
    "POST:/api/v1/outlier/impact": "outlier:impact:execute",
    "POST:/api/v1/outlier/pipeline": "outlier:pipeline:execute",
    "GET:/api/v1/outlier/health": "outlier:health:read",
    "GET:/api/v1/outlier/stats": "outlier:stats:read",
    # ==========================================================================
    # Time Series Gap Filler Agent routes (/api/v1/gap-filler) - AGENT-DATA-014
    # ==========================================================================
    "POST:/api/v1/gap-filler/jobs": "gap_filler:jobs:create",
    "GET:/api/v1/gap-filler/jobs": "gap_filler:jobs:read",
    "GET:/api/v1/gap-filler/jobs/{job_id}": "gap_filler:jobs:read",
    "DELETE:/api/v1/gap-filler/jobs/{job_id}": "gap_filler:jobs:delete",
    "POST:/api/v1/gap-filler/detect": "gap_filler:detect:execute",
    "POST:/api/v1/gap-filler/detect/batch": "gap_filler:detect:execute",
    "GET:/api/v1/gap-filler/detections": "gap_filler:detections:read",
    "GET:/api/v1/gap-filler/detections/{detection_id}": "gap_filler:detections:read",
    "POST:/api/v1/gap-filler/frequency": "gap_filler:frequency:execute",
    "GET:/api/v1/gap-filler/frequency/{analysis_id}": "gap_filler:frequency:read",
    "POST:/api/v1/gap-filler/fill": "gap_filler:fill:execute",
    "GET:/api/v1/gap-filler/fills/{fill_id}": "gap_filler:fill:read",
    "POST:/api/v1/gap-filler/validate": "gap_filler:validate:execute",
    "GET:/api/v1/gap-filler/validations/{validation_id}": "gap_filler:validate:read",
    "POST:/api/v1/gap-filler/correlations": "gap_filler:correlations:execute",
    "GET:/api/v1/gap-filler/correlations": "gap_filler:correlations:read",
    "POST:/api/v1/gap-filler/calendars": "gap_filler:calendars:manage",
    "GET:/api/v1/gap-filler/calendars": "gap_filler:calendars:read",
    "GET:/api/v1/gap-filler/health": "gap_filler:health:read",
    "GET:/api/v1/gap-filler/stats": "gap_filler:stats:read",
    # ==========================================================================
    # Cross-Source Reconciliation Agent routes (/api/v1/reconciliation) - AGENT-DATA-015
    # ==========================================================================
    "POST:/api/v1/reconciliation/jobs": "reconciliation:jobs:create",
    "GET:/api/v1/reconciliation/jobs": "reconciliation:jobs:read",
    "GET:/api/v1/reconciliation/jobs/{id}": "reconciliation:jobs:read",
    "DELETE:/api/v1/reconciliation/jobs/{id}": "reconciliation:jobs:delete",
    "POST:/api/v1/reconciliation/sources": "reconciliation:sources:create",
    "GET:/api/v1/reconciliation/sources": "reconciliation:sources:read",
    "GET:/api/v1/reconciliation/sources/{id}": "reconciliation:sources:read",
    "PUT:/api/v1/reconciliation/sources/{id}": "reconciliation:sources:update",
    "POST:/api/v1/reconciliation/match": "reconciliation:match:execute",
    "GET:/api/v1/reconciliation/matches": "reconciliation:matches:read",
    "GET:/api/v1/reconciliation/matches/{id}": "reconciliation:matches:read",
    "POST:/api/v1/reconciliation/compare": "reconciliation:compare:execute",
    "GET:/api/v1/reconciliation/discrepancies": "reconciliation:discrepancies:read",
    "GET:/api/v1/reconciliation/discrepancies/{id}": "reconciliation:discrepancies:read",
    "POST:/api/v1/reconciliation/resolve": "reconciliation:resolve:execute",
    "GET:/api/v1/reconciliation/golden-records": "reconciliation:golden_records:read",
    "GET:/api/v1/reconciliation/golden-records/{id}": "reconciliation:golden_records:read",
    "POST:/api/v1/reconciliation/pipeline": "reconciliation:pipeline:execute",
    "GET:/api/v1/reconciliation/health": "reconciliation:health:read",
    "GET:/api/v1/reconciliation/stats": "reconciliation:stats:read",
    # Data Freshness Monitor Agent routes (/api/v1/freshness) - AGENT-DATA-016
    "POST:/api/v1/freshness/datasets": "freshness:datasets:create",
    "GET:/api/v1/freshness/datasets": "freshness:datasets:read",
    "GET:/api/v1/freshness/datasets/{dataset_id}": "freshness:datasets:read",
    "PUT:/api/v1/freshness/datasets/{dataset_id}": "freshness:datasets:update",
    "DELETE:/api/v1/freshness/datasets/{dataset_id}": "freshness:datasets:delete",
    "POST:/api/v1/freshness/sla": "freshness:sla:create",
    "GET:/api/v1/freshness/sla": "freshness:sla:read",
    "GET:/api/v1/freshness/sla/{sla_id}": "freshness:sla:read",
    "PUT:/api/v1/freshness/sla/{sla_id}": "freshness:sla:update",
    "POST:/api/v1/freshness/check": "freshness:check:execute",
    "POST:/api/v1/freshness/check/batch": "freshness:check:execute",
    "GET:/api/v1/freshness/checks": "freshness:checks:read",
    "GET:/api/v1/freshness/breaches": "freshness:breaches:read",
    "GET:/api/v1/freshness/breaches/{breach_id}": "freshness:breaches:read",
    "PUT:/api/v1/freshness/breaches/{breach_id}": "freshness:breaches:update",
    "GET:/api/v1/freshness/alerts": "freshness:alerts:read",
    "GET:/api/v1/freshness/predictions": "freshness:predictions:read",
    "POST:/api/v1/freshness/pipeline": "freshness:pipeline:execute",
    "GET:/api/v1/freshness/health": "freshness:health:read",
    "GET:/api/v1/freshness/stats": "freshness:stats:read",

    # Schema Migration Agent routes (/api/v1/schema-migration) - AGENT-DATA-017
    "POST:/api/v1/schema-migration/schemas": "schema-migration:schemas:create",
    "GET:/api/v1/schema-migration/schemas": "schema-migration:schemas:read",
    "GET:/api/v1/schema-migration/schemas/{schema_id}": "schema-migration:schemas:read",
    "PUT:/api/v1/schema-migration/schemas/{schema_id}": "schema-migration:schemas:update",
    "DELETE:/api/v1/schema-migration/schemas/{schema_id}": "schema-migration:schemas:delete",
    "POST:/api/v1/schema-migration/versions": "schema-migration:versions:create",
    "GET:/api/v1/schema-migration/versions": "schema-migration:versions:read",
    "GET:/api/v1/schema-migration/versions/{version_id}": "schema-migration:versions:read",
    "POST:/api/v1/schema-migration/changes/detect": "schema-migration:changes:create",
    "GET:/api/v1/schema-migration/changes": "schema-migration:changes:read",
    "POST:/api/v1/schema-migration/compatibility/check": "schema-migration:compatibility:create",
    "GET:/api/v1/schema-migration/compatibility": "schema-migration:compatibility:read",
    "POST:/api/v1/schema-migration/plans": "schema-migration:plans:create",
    "GET:/api/v1/schema-migration/plans/{plan_id}": "schema-migration:plans:read",
    "POST:/api/v1/schema-migration/execute": "schema-migration:execute:create",
    "GET:/api/v1/schema-migration/executions/{execution_id}": "schema-migration:executions:read",
    "POST:/api/v1/schema-migration/rollback/{execution_id}": "schema-migration:rollback:create",
    "POST:/api/v1/schema-migration/pipeline": "schema-migration:pipeline:create",
    "GET:/api/v1/schema-migration/health": "schema-migration:health:read",
    "GET:/api/v1/schema-migration/stats": "schema-migration:stats:read",
    # ==========================================================================
    # Data Lineage Tracker Agent routes (/api/v1/data-lineage) - AGENT-DATA-018
    # ==========================================================================
    "POST:/api/v1/data-lineage/assets": "lineage:write",
    "GET:/api/v1/data-lineage/assets": "lineage:read",
    "GET:/api/v1/data-lineage/assets/{id}": "lineage:read",
    "PUT:/api/v1/data-lineage/assets/{id}": "lineage:write",
    "DELETE:/api/v1/data-lineage/assets/{id}": "lineage:admin",
    "POST:/api/v1/data-lineage/transformations": "lineage:write",
    "GET:/api/v1/data-lineage/transformations": "lineage:read",
    "POST:/api/v1/data-lineage/edges": "lineage:write",
    "GET:/api/v1/data-lineage/edges": "lineage:read",
    "GET:/api/v1/data-lineage/graph": "lineage:read",
    "GET:/api/v1/data-lineage/graph/subgraph/{asset_id}": "lineage:read",
    "GET:/api/v1/data-lineage/backward/{asset_id}": "lineage:read",
    "GET:/api/v1/data-lineage/forward/{asset_id}": "lineage:read",
    "POST:/api/v1/data-lineage/impact": "lineage:read",
    "POST:/api/v1/data-lineage/validate": "lineage:validate",
    "GET:/api/v1/data-lineage/validate/{id}": "lineage:read",
    "POST:/api/v1/data-lineage/reports": "lineage:report",
    "POST:/api/v1/data-lineage/pipeline": "lineage:admin",
    "GET:/api/v1/data-lineage/health": "lineage:read",
    "GET:/api/v1/data-lineage/stats": "lineage:read",
    # ==========================================================================
    # Validation Rule Engine routes (/api/v1/validation-rules) - AGENT-DATA-019
    # ==========================================================================
    "POST:/api/v1/validation-rules/rules": "validation-rules:rules:create",
    "GET:/api/v1/validation-rules/rules": "validation-rules:rules:read",
    "GET:/api/v1/validation-rules/rules/{rule_id}": "validation-rules:rules:read",
    "PUT:/api/v1/validation-rules/rules/{rule_id}": "validation-rules:rules:update",
    "DELETE:/api/v1/validation-rules/rules/{rule_id}": "validation-rules:rules:delete",
    "POST:/api/v1/validation-rules/rule-sets": "validation-rules:rule-sets:create",
    "GET:/api/v1/validation-rules/rule-sets": "validation-rules:rule-sets:read",
    "GET:/api/v1/validation-rules/rule-sets/{set_id}": "validation-rules:rule-sets:read",
    "PUT:/api/v1/validation-rules/rule-sets/{set_id}": "validation-rules:rule-sets:update",
    "DELETE:/api/v1/validation-rules/rule-sets/{set_id}": "validation-rules:rule-sets:delete",
    "POST:/api/v1/validation-rules/evaluate": "validation-rules:evaluate:create",
    "POST:/api/v1/validation-rules/evaluate/batch": "validation-rules:evaluate:create",
    "GET:/api/v1/validation-rules/evaluations/{eval_id}": "validation-rules:evaluations:read",
    "POST:/api/v1/validation-rules/conflicts/detect": "validation-rules:conflicts:create",
    "GET:/api/v1/validation-rules/conflicts": "validation-rules:conflicts:read",
    "POST:/api/v1/validation-rules/packs/{pack_name}/apply": "validation-rules:packs:create",
    "GET:/api/v1/validation-rules/packs": "validation-rules:packs:read",
    "POST:/api/v1/validation-rules/reports": "validation-rules:reports:create",
    "POST:/api/v1/validation-rules/pipeline": "validation-rules:pipeline:create",
    "GET:/api/v1/validation-rules/health": "validation-rules:health:read",
    # ==========================================================================
    # Climate Hazard Connector routes (/api/v1/climate-hazard) - AGENT-DATA-020
    # ==========================================================================
    "POST:/api/v1/climate-hazard/sources": "climate-hazard:write",
    "GET:/api/v1/climate-hazard/sources": "climate-hazard:read",
    "GET:/api/v1/climate-hazard/sources/{source_id}": "climate-hazard:read",
    "POST:/api/v1/climate-hazard/hazard-data/ingest": "climate-hazard:write",
    "GET:/api/v1/climate-hazard/hazard-data": "climate-hazard:read",
    "GET:/api/v1/climate-hazard/hazard-data/events": "climate-hazard:read",
    "POST:/api/v1/climate-hazard/risk-index/calculate": "climate-hazard:write",
    "POST:/api/v1/climate-hazard/risk-index/multi-hazard": "climate-hazard:write",
    "POST:/api/v1/climate-hazard/risk-index/compare": "climate-hazard:read",
    "POST:/api/v1/climate-hazard/scenarios/project": "climate-hazard:write",
    "GET:/api/v1/climate-hazard/scenarios": "climate-hazard:read",
    "POST:/api/v1/climate-hazard/assets": "climate-hazard:write",
    "GET:/api/v1/climate-hazard/assets": "climate-hazard:read",
    "POST:/api/v1/climate-hazard/exposure/assess": "climate-hazard:write",
    "POST:/api/v1/climate-hazard/exposure/portfolio": "climate-hazard:write",
    "POST:/api/v1/climate-hazard/vulnerability/score": "climate-hazard:write",
    "POST:/api/v1/climate-hazard/reports/generate": "climate-hazard:write",
    "GET:/api/v1/climate-hazard/reports/{report_id}": "climate-hazard:read",
    "POST:/api/v1/climate-hazard/pipeline/run": "climate-hazard:execute",
    "GET:/api/v1/climate-hazard/health": "climate-hazard:read",
    # ------------------------------------------------------------------
    # Stationary Combustion (AGENT-MRV-001)
    # ------------------------------------------------------------------
    "POST:/api/v1/stationary-combustion/calculate": "stationary-combustion:execute",
    "POST:/api/v1/stationary-combustion/calculate/batch": "stationary-combustion:execute",
    "GET:/api/v1/stationary-combustion/calculations": "stationary-combustion:read",
    "GET:/api/v1/stationary-combustion/calculations/{calc_id}": "stationary-combustion:read",
    "POST:/api/v1/stationary-combustion/fuels": "stationary-combustion:write",
    "GET:/api/v1/stationary-combustion/fuels": "stationary-combustion:read",
    "GET:/api/v1/stationary-combustion/fuels/{fuel_id}": "stationary-combustion:read",
    "POST:/api/v1/stationary-combustion/factors": "stationary-combustion:write",
    "GET:/api/v1/stationary-combustion/factors": "stationary-combustion:read",
    "GET:/api/v1/stationary-combustion/factors/{factor_id}": "stationary-combustion:read",
    "POST:/api/v1/stationary-combustion/equipment": "stationary-combustion:write",
    "GET:/api/v1/stationary-combustion/equipment": "stationary-combustion:read",
    "GET:/api/v1/stationary-combustion/equipment/{equip_id}": "stationary-combustion:read",
    "POST:/api/v1/stationary-combustion/aggregate": "stationary-combustion:execute",
    "GET:/api/v1/stationary-combustion/aggregations": "stationary-combustion:read",
    "POST:/api/v1/stationary-combustion/uncertainty": "stationary-combustion:execute",
    "GET:/api/v1/stationary-combustion/audit/{calc_id}": "stationary-combustion:read",
    "POST:/api/v1/stationary-combustion/validate": "stationary-combustion:execute",
    "GET:/api/v1/stationary-combustion/health": "stationary-combustion:read",
    "GET:/api/v1/stationary-combustion/stats": "stationary-combustion:read",
    # ------------------------------------------------------------------
    # Refrigerants & F-Gas (AGENT-MRV-002)
    # ------------------------------------------------------------------
    "POST:/api/v1/refrigerants-fgas/calculate": "refrigerants-fgas:execute",
    "POST:/api/v1/refrigerants-fgas/calculate/batch": "refrigerants-fgas:execute",
    "GET:/api/v1/refrigerants-fgas/calculations": "refrigerants-fgas:read",
    "GET:/api/v1/refrigerants-fgas/calculations/{calc_id}": "refrigerants-fgas:read",
    "POST:/api/v1/refrigerants-fgas/refrigerants": "refrigerants-fgas:write",
    "GET:/api/v1/refrigerants-fgas/refrigerants": "refrigerants-fgas:read",
    "GET:/api/v1/refrigerants-fgas/refrigerants/{ref_id}": "refrigerants-fgas:read",
    "POST:/api/v1/refrigerants-fgas/equipment": "refrigerants-fgas:write",
    "GET:/api/v1/refrigerants-fgas/equipment": "refrigerants-fgas:read",
    "GET:/api/v1/refrigerants-fgas/equipment/{equip_id}": "refrigerants-fgas:read",
    "POST:/api/v1/refrigerants-fgas/service-events": "refrigerants-fgas:write",
    "GET:/api/v1/refrigerants-fgas/service-events": "refrigerants-fgas:read",
    "POST:/api/v1/refrigerants-fgas/leak-rates": "refrigerants-fgas:write",
    "GET:/api/v1/refrigerants-fgas/leak-rates": "refrigerants-fgas:read",
    "POST:/api/v1/refrigerants-fgas/compliance/check": "refrigerants-fgas:execute",
    "GET:/api/v1/refrigerants-fgas/compliance": "refrigerants-fgas:read",
    "POST:/api/v1/refrigerants-fgas/uncertainty": "refrigerants-fgas:execute",
    "GET:/api/v1/refrigerants-fgas/audit/{calc_id}": "refrigerants-fgas:read",
    "GET:/api/v1/refrigerants-fgas/health": "refrigerants-fgas:read",
    "GET:/api/v1/refrigerants-fgas/stats": "refrigerants-fgas:read",
    # ------------------------------------------------------------------
    # Mobile Combustion (AGENT-MRV-003)
    # ------------------------------------------------------------------
    "POST:/api/v1/mobile-combustion/calculate": "mobile-combustion:execute",
    "POST:/api/v1/mobile-combustion/calculate/batch": "mobile-combustion:execute",
    "GET:/api/v1/mobile-combustion/calculations": "mobile-combustion:read",
    "GET:/api/v1/mobile-combustion/calculations/{calc_id}": "mobile-combustion:read",
    "POST:/api/v1/mobile-combustion/vehicles": "mobile-combustion:write",
    "GET:/api/v1/mobile-combustion/vehicles": "mobile-combustion:read",
    "GET:/api/v1/mobile-combustion/vehicles/{vehicle_id}": "mobile-combustion:read",
    "POST:/api/v1/mobile-combustion/trips": "mobile-combustion:write",
    "GET:/api/v1/mobile-combustion/trips": "mobile-combustion:read",
    "GET:/api/v1/mobile-combustion/trips/{trip_id}": "mobile-combustion:read",
    "POST:/api/v1/mobile-combustion/fuels": "mobile-combustion:write",
    "GET:/api/v1/mobile-combustion/fuels": "mobile-combustion:read",
    "POST:/api/v1/mobile-combustion/factors": "mobile-combustion:write",
    "GET:/api/v1/mobile-combustion/factors": "mobile-combustion:read",
    "POST:/api/v1/mobile-combustion/aggregate": "mobile-combustion:execute",
    "GET:/api/v1/mobile-combustion/aggregations": "mobile-combustion:read",
    "POST:/api/v1/mobile-combustion/uncertainty": "mobile-combustion:execute",
    "POST:/api/v1/mobile-combustion/compliance/check": "mobile-combustion:execute",
    "GET:/api/v1/mobile-combustion/health": "mobile-combustion:read",
    "GET:/api/v1/mobile-combustion/stats": "mobile-combustion:read",
    # ------------------------------------------------------------------
    # Process Emissions (AGENT-MRV-004)
    # ------------------------------------------------------------------
    "POST:/api/v1/process-emissions/calculate": "process-emissions:execute",
    "POST:/api/v1/process-emissions/calculate/batch": "process-emissions:execute",
    "GET:/api/v1/process-emissions/calculations": "process-emissions:read",
    "GET:/api/v1/process-emissions/calculations/{calc_id}": "process-emissions:read",
    "POST:/api/v1/process-emissions/processes": "process-emissions:write",
    "GET:/api/v1/process-emissions/processes": "process-emissions:read",
    "GET:/api/v1/process-emissions/processes/{process_id}": "process-emissions:read",
    "POST:/api/v1/process-emissions/materials": "process-emissions:write",
    "GET:/api/v1/process-emissions/materials": "process-emissions:read",
    "GET:/api/v1/process-emissions/materials/{material_id}": "process-emissions:read",
    "POST:/api/v1/process-emissions/units": "process-emissions:write",
    "GET:/api/v1/process-emissions/units": "process-emissions:read",
    "POST:/api/v1/process-emissions/factors": "process-emissions:write",
    "GET:/api/v1/process-emissions/factors": "process-emissions:read",
    "POST:/api/v1/process-emissions/abatement": "process-emissions:write",
    "GET:/api/v1/process-emissions/abatement": "process-emissions:read",
    "POST:/api/v1/process-emissions/uncertainty": "process-emissions:execute",
    "POST:/api/v1/process-emissions/compliance/check": "process-emissions:execute",
    "GET:/api/v1/process-emissions/health": "process-emissions:read",
    "GET:/api/v1/process-emissions/stats": "process-emissions:read",
    # ------------------------------------------------------------------
    # Fugitive Emissions (AGENT-MRV-005)
    # ------------------------------------------------------------------
    "POST:/api/v1/fugitive-emissions/calculate": "fugitive-emissions:execute",
    "POST:/api/v1/fugitive-emissions/calculate/batch": "fugitive-emissions:execute",
    "GET:/api/v1/fugitive-emissions/calculations": "fugitive-emissions:read",
    "GET:/api/v1/fugitive-emissions/calculations/{calc_id}": "fugitive-emissions:read",
    "POST:/api/v1/fugitive-emissions/sources": "fugitive-emissions:write",
    "GET:/api/v1/fugitive-emissions/sources": "fugitive-emissions:read",
    "GET:/api/v1/fugitive-emissions/sources/{source_id}": "fugitive-emissions:read",
    "POST:/api/v1/fugitive-emissions/components": "fugitive-emissions:write",
    "GET:/api/v1/fugitive-emissions/components": "fugitive-emissions:read",
    "GET:/api/v1/fugitive-emissions/components/{component_id}": "fugitive-emissions:read",
    "POST:/api/v1/fugitive-emissions/surveys": "fugitive-emissions:write",
    "GET:/api/v1/fugitive-emissions/surveys": "fugitive-emissions:read",
    "POST:/api/v1/fugitive-emissions/factors": "fugitive-emissions:write",
    "GET:/api/v1/fugitive-emissions/factors": "fugitive-emissions:read",
    "POST:/api/v1/fugitive-emissions/repairs": "fugitive-emissions:write",
    "GET:/api/v1/fugitive-emissions/repairs": "fugitive-emissions:read",
    "POST:/api/v1/fugitive-emissions/uncertainty": "fugitive-emissions:execute",
    "POST:/api/v1/fugitive-emissions/compliance/check": "fugitive-emissions:execute",
    "GET:/api/v1/fugitive-emissions/health": "fugitive-emissions:read",
    "GET:/api/v1/fugitive-emissions/stats": "fugitive-emissions:read",
    # ── Land Use Emissions (AGENT-MRV-006) ─────────────────────────
    "POST:/api/v1/land-use-emissions/calculations": "land-use:calculate",
    "POST:/api/v1/land-use-emissions/calculations/batch": "land-use:calculate",
    "GET:/api/v1/land-use-emissions/calculations": "land-use:read",
    "GET:/api/v1/land-use-emissions/calculations/{calc_id}": "land-use:read",
    "DELETE:/api/v1/land-use-emissions/calculations/{calc_id}": "land-use:delete",
    "POST:/api/v1/land-use-emissions/carbon-stocks": "land-use:carbon-stocks:write",
    "GET:/api/v1/land-use-emissions/carbon-stocks/{parcel_id}": "land-use:carbon-stocks:read",
    "GET:/api/v1/land-use-emissions/carbon-stocks/{parcel_id}/summary": "land-use:carbon-stocks:read",
    "POST:/api/v1/land-use-emissions/land-parcels": "land-use:parcels:write",
    "GET:/api/v1/land-use-emissions/land-parcels": "land-use:parcels:read",
    "PUT:/api/v1/land-use-emissions/land-parcels/{parcel_id}": "land-use:parcels:write",
    "POST:/api/v1/land-use-emissions/transitions": "land-use:transitions:write",
    "GET:/api/v1/land-use-emissions/transitions": "land-use:transitions:read",
    "GET:/api/v1/land-use-emissions/transitions/matrix": "land-use:transitions:read",
    "POST:/api/v1/land-use-emissions/soc-assessments": "land-use:soc:write",
    "GET:/api/v1/land-use-emissions/soc-assessments/{parcel_id}": "land-use:soc:read",
    "POST:/api/v1/land-use-emissions/compliance/check": "land-use:compliance:check",
    "GET:/api/v1/land-use-emissions/compliance/{check_id}": "land-use:compliance:read",
    "POST:/api/v1/land-use-emissions/uncertainty": "land-use:uncertainty:run",
    "GET:/api/v1/land-use-emissions/aggregations": "land-use:read",
    # ── Waste Treatment Emissions (AGENT-MRV-007) ────────────────
    "POST:/api/v1/waste-treatment-emissions/calculations": "waste-treatment:calculate",
    "POST:/api/v1/waste-treatment-emissions/calculations/batch": "waste-treatment:calculate",
    "GET:/api/v1/waste-treatment-emissions/calculations": "waste-treatment:read",
    "GET:/api/v1/waste-treatment-emissions/calculations/{calc_id}": "waste-treatment:read",
    "DELETE:/api/v1/waste-treatment-emissions/calculations/{calc_id}": "waste-treatment:delete",
    "POST:/api/v1/waste-treatment-emissions/facilities": "waste-treatment:facilities:write",
    "GET:/api/v1/waste-treatment-emissions/facilities": "waste-treatment:facilities:read",
    "PUT:/api/v1/waste-treatment-emissions/facilities/{facility_id}": "waste-treatment:facilities:write",
    "POST:/api/v1/waste-treatment-emissions/waste-streams": "waste-treatment:streams:write",
    "GET:/api/v1/waste-treatment-emissions/waste-streams": "waste-treatment:streams:read",
    "PUT:/api/v1/waste-treatment-emissions/waste-streams/{stream_id}": "waste-treatment:streams:write",
    "POST:/api/v1/waste-treatment-emissions/treatment-events": "waste-treatment:events:write",
    "GET:/api/v1/waste-treatment-emissions/treatment-events": "waste-treatment:events:read",
    "POST:/api/v1/waste-treatment-emissions/methane-recovery": "waste-treatment:recovery:write",
    "GET:/api/v1/waste-treatment-emissions/methane-recovery/{facility_id}": "waste-treatment:recovery:read",
    "POST:/api/v1/waste-treatment-emissions/compliance/check": "waste-treatment:compliance:check",
    "GET:/api/v1/waste-treatment-emissions/compliance/{check_id}": "waste-treatment:compliance:read",
    "POST:/api/v1/waste-treatment-emissions/uncertainty": "waste-treatment:uncertainty:run",
    "GET:/api/v1/waste-treatment-emissions/aggregations": "waste-treatment:read",
    # ── Agricultural Emissions (AGENT-MRV-008) ────────────────
    "POST:/api/v1/agricultural-emissions/calculations": "agricultural:calculate",
    "POST:/api/v1/agricultural-emissions/calculations/batch": "agricultural:calculate",
    "GET:/api/v1/agricultural-emissions/calculations": "agricultural:read",
    "GET:/api/v1/agricultural-emissions/calculations/{calc_id}": "agricultural:read",
    "DELETE:/api/v1/agricultural-emissions/calculations/{calc_id}": "agricultural:delete",
    "POST:/api/v1/agricultural-emissions/farms": "agricultural:farms:write",
    "GET:/api/v1/agricultural-emissions/farms": "agricultural:farms:read",
    "PUT:/api/v1/agricultural-emissions/farms/{farm_id}": "agricultural:farms:write",
    "POST:/api/v1/agricultural-emissions/livestock": "agricultural:livestock:write",
    "GET:/api/v1/agricultural-emissions/livestock": "agricultural:livestock:read",
    "PUT:/api/v1/agricultural-emissions/livestock/{herd_id}": "agricultural:livestock:write",
    "POST:/api/v1/agricultural-emissions/cropland-inputs": "agricultural:cropland:write",
    "GET:/api/v1/agricultural-emissions/cropland-inputs": "agricultural:cropland:read",
    "POST:/api/v1/agricultural-emissions/rice-fields": "agricultural:rice:write",
    "GET:/api/v1/agricultural-emissions/rice-fields": "agricultural:rice:read",
    "POST:/api/v1/agricultural-emissions/compliance/check": "agricultural:compliance:check",
    "GET:/api/v1/agricultural-emissions/compliance/{check_id}": "agricultural:compliance:read",
    "POST:/api/v1/agricultural-emissions/uncertainty": "agricultural:uncertainty:run",
    "GET:/api/v1/agricultural-emissions/aggregations": "agricultural:read",
    # ── Scope 2 Location-Based Emissions (AGENT-MRV-009) ────────────────
    "POST:/api/v1/scope2-location/calculations": "scope2-location:calculate",
    "POST:/api/v1/scope2-location/calculations/batch": "scope2-location:calculate",
    "GET:/api/v1/scope2-location/calculations": "scope2-location:read",
    "GET:/api/v1/scope2-location/calculations/{calc_id}": "scope2-location:read",
    "DELETE:/api/v1/scope2-location/calculations/{calc_id}": "scope2-location:delete",
    "POST:/api/v1/scope2-location/facilities": "scope2-location:facilities:write",
    "GET:/api/v1/scope2-location/facilities": "scope2-location:facilities:read",
    "PUT:/api/v1/scope2-location/facilities/{facility_id}": "scope2-location:facilities:write",
    "POST:/api/v1/scope2-location/consumption": "scope2-location:consumption:write",
    "GET:/api/v1/scope2-location/consumption": "scope2-location:consumption:read",
    "GET:/api/v1/scope2-location/grid-factors": "scope2-location:factors:read",
    "GET:/api/v1/scope2-location/grid-factors/{region}": "scope2-location:factors:read",
    "POST:/api/v1/scope2-location/grid-factors/custom": "scope2-location:factors:write",
    "GET:/api/v1/scope2-location/td-losses": "scope2-location:factors:read",
    "POST:/api/v1/scope2-location/compliance/check": "scope2-location:compliance:check",
    "GET:/api/v1/scope2-location/compliance/{check_id}": "scope2-location:compliance:read",
    "POST:/api/v1/scope2-location/uncertainty": "scope2-location:uncertainty:run",
    "GET:/api/v1/scope2-location/aggregations": "scope2-location:read",
    # ── Scope 2 Market-Based Emissions (AGENT-MRV-010) ─────────────────
    "POST:/api/v1/scope2-market/calculations": "scope2-market:calculate",
    "POST:/api/v1/scope2-market/calculations/batch": "scope2-market:calculate",
    "GET:/api/v1/scope2-market/calculations": "scope2-market:read",
    "GET:/api/v1/scope2-market/calculations/{calc_id}": "scope2-market:read",
    "DELETE:/api/v1/scope2-market/calculations/{calc_id}": "scope2-market:delete",
    "POST:/api/v1/scope2-market/facilities": "scope2-market:facilities:write",
    "GET:/api/v1/scope2-market/facilities": "scope2-market:facilities:read",
    "PUT:/api/v1/scope2-market/facilities/{facility_id}": "scope2-market:facilities:write",
    "POST:/api/v1/scope2-market/instruments": "scope2-market:instruments:write",
    "GET:/api/v1/scope2-market/instruments": "scope2-market:instruments:read",
    "POST:/api/v1/scope2-market/instruments/{instrument_id}/retire": "scope2-market:instruments:write",
    "POST:/api/v1/scope2-market/compliance/check": "scope2-market:compliance:check",
    "GET:/api/v1/scope2-market/compliance/{check_id}": "scope2-market:compliance:read",
    "POST:/api/v1/scope2-market/uncertainty": "scope2-market:uncertainty:run",
    "POST:/api/v1/scope2-market/dual-report": "scope2-market:dual-report:write",
    "GET:/api/v1/scope2-market/aggregations": "scope2-market:read",
    "GET:/api/v1/scope2-market/coverage/{facility_id}": "scope2-market:coverage:read",
    # ── Steam/Heat Purchase (AGENT-MRV-011) ──────────────────────────
    "POST:/api/v1/steam-heat-purchase/calculate/steam": "steam-heat-purchase:calculate",
    "POST:/api/v1/steam-heat-purchase/calculate/heating": "steam-heat-purchase:calculate",
    "POST:/api/v1/steam-heat-purchase/calculate/cooling": "steam-heat-purchase:calculate",
    "POST:/api/v1/steam-heat-purchase/calculate/chp": "steam-heat-purchase:calculate",
    "POST:/api/v1/steam-heat-purchase/calculate/batch": "steam-heat-purchase:calculate",
    "GET:/api/v1/steam-heat-purchase/factors/fuels": "steam-heat-purchase:factors:read",
    "GET:/api/v1/steam-heat-purchase/factors/fuels/{fuel_type}": "steam-heat-purchase:factors:read",
    "GET:/api/v1/steam-heat-purchase/factors/heating/{region}": "steam-heat-purchase:factors:read",
    "GET:/api/v1/steam-heat-purchase/factors/cooling/{technology}": "steam-heat-purchase:factors:read",
    "GET:/api/v1/steam-heat-purchase/factors/chp-defaults": "steam-heat-purchase:factors:read",
    "POST:/api/v1/steam-heat-purchase/facilities": "steam-heat-purchase:facilities:write",
    "GET:/api/v1/steam-heat-purchase/facilities/{facility_id}": "steam-heat-purchase:facilities:read",
    "POST:/api/v1/steam-heat-purchase/suppliers": "steam-heat-purchase:suppliers:write",
    "GET:/api/v1/steam-heat-purchase/suppliers/{supplier_id}": "steam-heat-purchase:suppliers:read",
    "POST:/api/v1/steam-heat-purchase/uncertainty": "steam-heat-purchase:uncertainty:run",
    "POST:/api/v1/steam-heat-purchase/compliance/check": "steam-heat-purchase:compliance:check",
    "GET:/api/v1/steam-heat-purchase/compliance/frameworks": "steam-heat-purchase:compliance:read",
    "POST:/api/v1/steam-heat-purchase/aggregate": "steam-heat-purchase:read",
    "GET:/api/v1/steam-heat-purchase/calculations/{calc_id}": "steam-heat-purchase:read",
    "GET:/api/v1/steam-heat-purchase/health": "steam-heat-purchase:read",

    # ── Cooling Purchase (AGENT-MRV-012) ──────────────────────────────
    "POST:/api/v1/cooling-purchase/calculate/electric": "cooling-purchase:calculate",
    "POST:/api/v1/cooling-purchase/calculate/absorption": "cooling-purchase:calculate",
    "POST:/api/v1/cooling-purchase/calculate/district": "cooling-purchase:calculate",
    "POST:/api/v1/cooling-purchase/calculate/free-cooling": "cooling-purchase:calculate",
    "POST:/api/v1/cooling-purchase/calculate/tes": "cooling-purchase:calculate",
    "POST:/api/v1/cooling-purchase/calculate/batch": "cooling-purchase:calculate",
    "GET:/api/v1/cooling-purchase/technologies": "cooling-purchase:factors:read",
    "GET:/api/v1/cooling-purchase/technologies/{tech_id}": "cooling-purchase:factors:read",
    "GET:/api/v1/cooling-purchase/factors/district/{region}": "cooling-purchase:factors:read",
    "GET:/api/v1/cooling-purchase/factors/heat-source/{source}": "cooling-purchase:factors:read",
    "GET:/api/v1/cooling-purchase/factors/refrigerants": "cooling-purchase:factors:read",
    "POST:/api/v1/cooling-purchase/facilities": "cooling-purchase:facilities:write",
    "GET:/api/v1/cooling-purchase/facilities/{facility_id}": "cooling-purchase:facilities:read",
    "POST:/api/v1/cooling-purchase/suppliers": "cooling-purchase:suppliers:write",
    "GET:/api/v1/cooling-purchase/suppliers/{supplier_id}": "cooling-purchase:suppliers:read",
    "POST:/api/v1/cooling-purchase/uncertainty": "cooling-purchase:uncertainty:run",
    "POST:/api/v1/cooling-purchase/compliance/check": "cooling-purchase:compliance:check",
    "GET:/api/v1/cooling-purchase/compliance/frameworks": "cooling-purchase:compliance:read",
    "POST:/api/v1/cooling-purchase/aggregate": "cooling-purchase:read",
    "GET:/api/v1/cooling-purchase/health": "cooling-purchase:read",

    # -- Dual Reporting Reconciliation (AGENT-MRV-013) --------------------
    "POST:/api/v1/dual-reporting/reconciliations": "dual-reporting:reconcile",
    "POST:/api/v1/dual-reporting/reconciliations/batch": "dual-reporting:reconcile",
    "GET:/api/v1/dual-reporting/reconciliations": "dual-reporting:read",
    "GET:/api/v1/dual-reporting/reconciliations/{recon_id}": "dual-reporting:read",
    "DELETE:/api/v1/dual-reporting/reconciliations/{recon_id}": "dual-reporting:delete",
    "GET:/api/v1/dual-reporting/reconciliations/{recon_id}/discrepancies": "dual-reporting:read",
    "GET:/api/v1/dual-reporting/reconciliations/{recon_id}/waterfall": "dual-reporting:read",
    "GET:/api/v1/dual-reporting/reconciliations/{recon_id}/quality": "dual-reporting:read",
    "GET:/api/v1/dual-reporting/reconciliations/{recon_id}/tables": "dual-reporting:read",
    "GET:/api/v1/dual-reporting/reconciliations/{recon_id}/trends": "dual-reporting:read",
    "POST:/api/v1/dual-reporting/reconciliations/{recon_id}/compliance": "dual-reporting:compliance",
    "GET:/api/v1/dual-reporting/compliance/{compliance_id}": "dual-reporting:compliance:read",
    "GET:/api/v1/dual-reporting/aggregations": "dual-reporting:read",
    "POST:/api/v1/dual-reporting/export": "dual-reporting:export",
    "GET:/api/v1/dual-reporting/health": "dual-reporting:read",
    "GET:/api/v1/dual-reporting/stats": "dual-reporting:read",
    # Purchased Goods & Services (AGENT-MRV-014)
    "POST:/api/v1/purchased-goods/calculate/spend-based": "purchased-goods:calculate",
    "POST:/api/v1/purchased-goods/calculate/average-data": "purchased-goods:calculate",
    "POST:/api/v1/purchased-goods/calculate/supplier-specific": "purchased-goods:calculate",
    "POST:/api/v1/purchased-goods/calculate/hybrid": "purchased-goods:calculate",
    "POST:/api/v1/purchased-goods/calculate/batch": "purchased-goods:calculate",
    "GET:/api/v1/purchased-goods/calculations/{calculation_id}": "purchased-goods:read",
    "GET:/api/v1/purchased-goods/calculations/{calculation_id}/details": "purchased-goods:read",
    "POST:/api/v1/purchased-goods/procurement/upload": "purchased-goods:procurement:write",
    "GET:/api/v1/purchased-goods/procurement/summary": "purchased-goods:procurement:read",
    "POST:/api/v1/purchased-goods/suppliers": "purchased-goods:suppliers:write",
    "GET:/api/v1/purchased-goods/suppliers/{supplier_id}": "purchased-goods:suppliers:read",
    "GET:/api/v1/purchased-goods/suppliers/{supplier_id}/emissions": "purchased-goods:suppliers:read",
    "GET:/api/v1/purchased-goods/emission-factors/eeio": "purchased-goods:ef:read",
    "GET:/api/v1/purchased-goods/emission-factors/physical": "purchased-goods:ef:read",
    "POST:/api/v1/purchased-goods/dqi/score": "purchased-goods:dqi:score",
    "GET:/api/v1/purchased-goods/dqi/{calculation_id}": "purchased-goods:dqi:read",
    "POST:/api/v1/purchased-goods/compliance/check": "purchased-goods:compliance:check",
    "GET:/api/v1/purchased-goods/compliance/frameworks": "purchased-goods:compliance:read",
    "POST:/api/v1/purchased-goods/export": "purchased-goods:export",
    "GET:/api/v1/purchased-goods/health": "purchased-goods:read",
    # Capital Goods (AGENT-MRV-015 / GL-MRV-S3-002)
    "POST:/api/v1/capital-goods/calculate": "capital-goods:calculate",
    "POST:/api/v1/capital-goods/calculate/batch": "capital-goods:calculate",
    "GET:/api/v1/capital-goods/calculations": "capital-goods:read",
    "GET:/api/v1/capital-goods/calculations/{calc_id}": "capital-goods:read",
    "DELETE:/api/v1/capital-goods/calculations/{calc_id}": "capital-goods:delete",
    "POST:/api/v1/capital-goods/assets": "capital-goods:assets:write",
    "GET:/api/v1/capital-goods/assets": "capital-goods:assets:read",
    "PUT:/api/v1/capital-goods/assets/{asset_id}": "capital-goods:assets:write",
    "GET:/api/v1/capital-goods/emission-factors": "capital-goods:factors:read",
    "GET:/api/v1/capital-goods/emission-factors/{factor_id}": "capital-goods:factors:read",
    "POST:/api/v1/capital-goods/emission-factors/custom": "capital-goods:factors:write",
    "POST:/api/v1/capital-goods/classify": "capital-goods:classify",
    "POST:/api/v1/capital-goods/compliance/check": "capital-goods:compliance:check",
    "GET:/api/v1/capital-goods/compliance/{check_id}": "capital-goods:compliance:read",
    "POST:/api/v1/capital-goods/uncertainty": "capital-goods:uncertainty:run",
    "GET:/api/v1/capital-goods/aggregations": "capital-goods:aggregations:read",
    "GET:/api/v1/capital-goods/hot-spots": "capital-goods:read",
    "POST:/api/v1/capital-goods/export": "capital-goods:export",
    "GET:/api/v1/capital-goods/health": "capital-goods:read",
    "GET:/api/v1/capital-goods/stats": "capital-goods:read",
    # Fuel & Energy Activities (AGENT-MRV-016)
    "POST:/api/v1/fuel-energy-activities/calculate": "fuel-energy-activities:calculate",
    "POST:/api/v1/fuel-energy-activities/calculate/batch": "fuel-energy-activities:calculate",
    "GET:/api/v1/fuel-energy-activities/calculations": "fuel-energy-activities:read",
    "GET:/api/v1/fuel-energy-activities/calculations/{calc_id}": "fuel-energy-activities:read",
    "DELETE:/api/v1/fuel-energy-activities/calculations/{calc_id}": "fuel-energy-activities:delete",
    "POST:/api/v1/fuel-energy-activities/fuel-consumption": "fuel-energy-activities:fuel-consumption:write",
    "GET:/api/v1/fuel-energy-activities/fuel-consumption": "fuel-energy-activities:fuel-consumption:read",
    "PUT:/api/v1/fuel-energy-activities/fuel-consumption/{record_id}": "fuel-energy-activities:fuel-consumption:write",
    "POST:/api/v1/fuel-energy-activities/electricity-consumption": "fuel-energy-activities:electricity-consumption:write",
    "GET:/api/v1/fuel-energy-activities/electricity-consumption": "fuel-energy-activities:electricity-consumption:read",
    "GET:/api/v1/fuel-energy-activities/emission-factors": "fuel-energy-activities:factors:read",
    "GET:/api/v1/fuel-energy-activities/emission-factors/{factor_id}": "fuel-energy-activities:factors:read",
    "POST:/api/v1/fuel-energy-activities/emission-factors/custom": "fuel-energy-activities:factors:write",
    "GET:/api/v1/fuel-energy-activities/td-loss-factors": "fuel-energy-activities:td-loss:read",
    "GET:/api/v1/fuel-energy-activities/td-loss-factors/{country_code}": "fuel-energy-activities:td-loss:read",
    "POST:/api/v1/fuel-energy-activities/compliance/check": "fuel-energy-activities:compliance:check",
    "GET:/api/v1/fuel-energy-activities/compliance/{check_id}": "fuel-energy-activities:compliance:read",
    "POST:/api/v1/fuel-energy-activities/uncertainty": "fuel-energy-activities:uncertainty:run",
    "GET:/api/v1/fuel-energy-activities/aggregations": "fuel-energy-activities:read",
    "GET:/api/v1/fuel-energy-activities/health": "fuel-energy-activities:read",
    # Upstream Transportation & Distribution (AGENT-MRV-017)
    "POST:/api/v1/upstream-transportation/calculate": "upstream-transportation:calculate",
    "POST:/api/v1/upstream-transportation/calculate/batch": "upstream-transportation:calculate",
    "GET:/api/v1/upstream-transportation/calculations": "upstream-transportation:read",
    "GET:/api/v1/upstream-transportation/calculations/{calculation_id}": "upstream-transportation:read",
    "DELETE:/api/v1/upstream-transportation/calculations/{calculation_id}/delete": "upstream-transportation:delete",
    "POST:/api/v1/upstream-transportation/transport-chains": "upstream-transportation:chains:write",
    "GET:/api/v1/upstream-transportation/transport-chains/list": "upstream-transportation:chains:read",
    "GET:/api/v1/upstream-transportation/transport-chains/{chain_id}": "upstream-transportation:chains:read",
    "GET:/api/v1/upstream-transportation/emission-factors": "upstream-transportation:factors:read",
    "GET:/api/v1/upstream-transportation/emission-factors/{factor_id}": "upstream-transportation:factors:read",
    "POST:/api/v1/upstream-transportation/emission-factors/custom": "upstream-transportation:factors:write",
    "POST:/api/v1/upstream-transportation/classify": "upstream-transportation:classify",
    "POST:/api/v1/upstream-transportation/compliance/check": "upstream-transportation:compliance:write",
    "GET:/api/v1/upstream-transportation/compliance/{check_id}": "upstream-transportation:compliance:read",
    "POST:/api/v1/upstream-transportation/uncertainty": "upstream-transportation:uncertainty:write",
    "GET:/api/v1/upstream-transportation/aggregations": "upstream-transportation:aggregations:read",
    "GET:/api/v1/upstream-transportation/hot-spots": "upstream-transportation:aggregations:read",
    "POST:/api/v1/upstream-transportation/export": "upstream-transportation:export",
    "GET:/api/v1/upstream-transportation/health": "upstream-transportation:health",
    "GET:/api/v1/upstream-transportation/stats": "upstream-transportation:stats",
    # --- Waste Generated (AGENT-MRV-018) ---
    "POST:/api/v1/waste-generated/calculate": "waste-generated:calculate",
    "POST:/api/v1/waste-generated/calculate/batch": "waste-generated:calculate",
    "POST:/api/v1/waste-generated/calculate/landfill": "waste-generated:calculate",
    "POST:/api/v1/waste-generated/calculate/incineration": "waste-generated:calculate",
    "POST:/api/v1/waste-generated/calculate/recycling": "waste-generated:calculate",
    "POST:/api/v1/waste-generated/calculate/composting": "waste-generated:calculate",
    "POST:/api/v1/waste-generated/calculate/anaerobic-digestion": "waste-generated:calculate",
    "POST:/api/v1/waste-generated/calculate/wastewater": "waste-generated:calculate",
    "GET:/api/v1/waste-generated/calculations/{id}": "waste-generated:read",
    "GET:/api/v1/waste-generated/calculations": "waste-generated:read",
    "DELETE:/api/v1/waste-generated/calculations/{id}": "waste-generated:delete",
    "GET:/api/v1/waste-generated/emission-factors": "waste-generated:read",
    "GET:/api/v1/waste-generated/emission-factors/{waste_type}": "waste-generated:read",
    "GET:/api/v1/waste-generated/waste-types": "waste-generated:read",
    "GET:/api/v1/waste-generated/treatment-methods": "waste-generated:read",
    "POST:/api/v1/waste-generated/compliance/check": "waste-generated:compliance",
    "POST:/api/v1/waste-generated/uncertainty/analyze": "waste-generated:analyze",
    "GET:/api/v1/waste-generated/aggregations/{period}": "waste-generated:read",
    "POST:/api/v1/waste-generated/diversion/analyze": "waste-generated:analyze",
    "GET:/api/v1/waste-generated/provenance/{id}": "waste-generated:read",
    # --- Business Travel (AGENT-MRV-019) ---
    "POST:/api/v1/business-travel/calculate": "business-travel:calculate",
    "POST:/api/v1/business-travel/calculate/batch": "business-travel:calculate",
    "POST:/api/v1/business-travel/calculate/flight": "business-travel:calculate",
    "POST:/api/v1/business-travel/calculate/rail": "business-travel:calculate",
    "POST:/api/v1/business-travel/calculate/road": "business-travel:calculate",
    "POST:/api/v1/business-travel/calculate/hotel": "business-travel:calculate",
    "POST:/api/v1/business-travel/calculate/spend": "business-travel:calculate",
    "GET:/api/v1/business-travel/calculations/{id}": "business-travel:read",
    "GET:/api/v1/business-travel/calculations": "business-travel:read",
    "DELETE:/api/v1/business-travel/calculations/{id}": "business-travel:delete",
    "GET:/api/v1/business-travel/emission-factors": "business-travel:read",
    "GET:/api/v1/business-travel/emission-factors/{mode}": "business-travel:read",
    "GET:/api/v1/business-travel/airports": "business-travel:read",
    "GET:/api/v1/business-travel/transport-modes": "business-travel:read",
    "GET:/api/v1/business-travel/cabin-classes": "business-travel:read",
    "POST:/api/v1/business-travel/compliance/check": "business-travel:compliance",
    "POST:/api/v1/business-travel/uncertainty/analyze": "business-travel:analyze",
    "GET:/api/v1/business-travel/aggregations/{period}": "business-travel:read",
    "POST:/api/v1/business-travel/hot-spots/analyze": "business-travel:analyze",
    "GET:/api/v1/business-travel/provenance/{id}": "business-travel:read",
    # ── Employee Commuting (AGENT-MRV-020) ──────────────────────────────
    "POST:/api/v1/employee-commuting/calculate": "employee-commuting:calculate",
    "POST:/api/v1/employee-commuting/calculate/batch": "employee-commuting:calculate",
    "POST:/api/v1/employee-commuting/calculate/vehicle": "employee-commuting:calculate",
    "POST:/api/v1/employee-commuting/calculate/transit": "employee-commuting:calculate",
    "POST:/api/v1/employee-commuting/calculate/telework": "employee-commuting:calculate",
    "POST:/api/v1/employee-commuting/calculate/carpool": "employee-commuting:calculate",
    "POST:/api/v1/employee-commuting/calculate/active": "employee-commuting:calculate",
    "POST:/api/v1/employee-commuting/calculate/survey": "employee-commuting:calculate",
    "POST:/api/v1/employee-commuting/calculate/average-data": "employee-commuting:calculate",
    "POST:/api/v1/employee-commuting/calculate/spend": "employee-commuting:calculate",
    "POST:/api/v1/employee-commuting/calculate/multi-modal": "employee-commuting:calculate",
    "GET:/api/v1/employee-commuting/calculations/{id}": "employee-commuting:read",
    "GET:/api/v1/employee-commuting/calculations": "employee-commuting:read",
    "DELETE:/api/v1/employee-commuting/calculations/{id}": "employee-commuting:delete",
    "GET:/api/v1/employee-commuting/emission-factors": "employee-commuting:read",
    "GET:/api/v1/employee-commuting/emission-factors/{mode}": "employee-commuting:read",
    "GET:/api/v1/employee-commuting/commute-modes": "employee-commuting:read",
    "GET:/api/v1/employee-commuting/working-days/{region}": "employee-commuting:read",
    "GET:/api/v1/employee-commuting/commute-averages": "employee-commuting:read",
    "GET:/api/v1/employee-commuting/grid-factors/{country}": "employee-commuting:read",
    "GET:/api/v1/employee-commuting/grid-factors": "employee-commuting:read",
    "POST:/api/v1/employee-commuting/compliance/check": "employee-commuting:compliance",
    "POST:/api/v1/employee-commuting/uncertainty/analyze": "employee-commuting:analyze",
    "GET:/api/v1/employee-commuting/aggregations/{period}": "employee-commuting:read",
    "POST:/api/v1/employee-commuting/mode-share/analyze": "employee-commuting:analyze",
    "GET:/api/v1/employee-commuting/provenance/{id}": "employee-commuting:read",

    # ── Upstream Leased Assets (AGENT-MRV-021 / Scope 3 Cat 8) ──
    "POST:/api/v1/upstream-leased-assets/calculate": "upstream-leased-assets:calculate",
    "POST:/api/v1/upstream-leased-assets/calculate/building": "upstream-leased-assets:calculate",
    "POST:/api/v1/upstream-leased-assets/calculate/vehicle": "upstream-leased-assets:calculate",
    "POST:/api/v1/upstream-leased-assets/calculate/equipment": "upstream-leased-assets:calculate",
    "POST:/api/v1/upstream-leased-assets/calculate/it-asset": "upstream-leased-assets:calculate",
    "POST:/api/v1/upstream-leased-assets/calculate/lessor": "upstream-leased-assets:calculate",
    "POST:/api/v1/upstream-leased-assets/calculate/spend": "upstream-leased-assets:calculate",
    "POST:/api/v1/upstream-leased-assets/calculate/batch": "upstream-leased-assets:calculate",
    "POST:/api/v1/upstream-leased-assets/calculate/portfolio": "upstream-leased-assets:calculate",
    "POST:/api/v1/upstream-leased-assets/compliance/check": "upstream-leased-assets:compliance",
    "GET:/api/v1/upstream-leased-assets/calculations/{id}": "upstream-leased-assets:read",
    "GET:/api/v1/upstream-leased-assets/calculations": "upstream-leased-assets:read",
    "DELETE:/api/v1/upstream-leased-assets/calculations/{id}": "upstream-leased-assets:delete",
    "GET:/api/v1/upstream-leased-assets/emission-factors/{type}": "upstream-leased-assets:read",
    "GET:/api/v1/upstream-leased-assets/building-benchmarks": "upstream-leased-assets:read",
    "GET:/api/v1/upstream-leased-assets/grid-factors/{country}": "upstream-leased-assets:read",
    "GET:/api/v1/upstream-leased-assets/lease-classification": "upstream-leased-assets:read",
    "GET:/api/v1/upstream-leased-assets/aggregations": "upstream-leased-assets:read",
    "GET:/api/v1/upstream-leased-assets/provenance/{id}": "upstream-leased-assets:read",
    "GET:/api/v1/upstream-leased-assets/health": "upstream-leased-assets:read",
    "POST:/api/v1/upstream-leased-assets/uncertainty/analyze": "upstream-leased-assets:analyze",
    "POST:/api/v1/upstream-leased-assets/portfolio/analyze": "upstream-leased-assets:analyze",

    # ── Downstream Transportation (AGENT-MRV-022 / Scope 3 Cat 9) ──
    "POST:/api/v1/downstream-transportation/calculate": "downstream-transportation:calculate",
    "POST:/api/v1/downstream-transportation/calculate/distance": "downstream-transportation:calculate",
    "POST:/api/v1/downstream-transportation/calculate/spend": "downstream-transportation:calculate",
    "POST:/api/v1/downstream-transportation/calculate/average-data": "downstream-transportation:calculate",
    "POST:/api/v1/downstream-transportation/calculate/warehouse": "downstream-transportation:calculate",
    "POST:/api/v1/downstream-transportation/calculate/last-mile": "downstream-transportation:calculate",
    "POST:/api/v1/downstream-transportation/calculate/supplier-specific": "downstream-transportation:calculate",
    "POST:/api/v1/downstream-transportation/calculate/batch": "downstream-transportation:calculate",
    "POST:/api/v1/downstream-transportation/calculate/portfolio": "downstream-transportation:calculate",
    "POST:/api/v1/downstream-transportation/compliance/check": "downstream-transportation:compliance",
    "GET:/api/v1/downstream-transportation/calculations/{id}": "downstream-transportation:read",
    "GET:/api/v1/downstream-transportation/calculations": "downstream-transportation:read",
    "DELETE:/api/v1/downstream-transportation/calculations/{id}": "downstream-transportation:delete",
    "GET:/api/v1/downstream-transportation/emission-factors/{mode}": "downstream-transportation:read",
    "GET:/api/v1/downstream-transportation/warehouse-benchmarks": "downstream-transportation:read",
    "GET:/api/v1/downstream-transportation/last-mile-factors": "downstream-transportation:read",
    "GET:/api/v1/downstream-transportation/incoterm-classification": "downstream-transportation:read",
    "GET:/api/v1/downstream-transportation/aggregations": "downstream-transportation:read",
    "GET:/api/v1/downstream-transportation/provenance/{id}": "downstream-transportation:read",
    "GET:/api/v1/downstream-transportation/health": "downstream-transportation:read",
    "POST:/api/v1/downstream-transportation/uncertainty/analyze": "downstream-transportation:analyze",
    "POST:/api/v1/downstream-transportation/portfolio/analyze": "downstream-transportation:analyze",

    # ── Processing of Sold Products (AGENT-MRV-023 / Scope 3 Cat 10) ──
    "POST:/api/v1/processing-sold-products/calculate": "processing-sold-products:calculate",
    "POST:/api/v1/processing-sold-products/calculate/site-specific": "processing-sold-products:calculate",
    "POST:/api/v1/processing-sold-products/calculate/site-specific/energy": "processing-sold-products:calculate",
    "POST:/api/v1/processing-sold-products/calculate/site-specific/fuel": "processing-sold-products:calculate",
    "POST:/api/v1/processing-sold-products/calculate/average-data": "processing-sold-products:calculate",
    "POST:/api/v1/processing-sold-products/calculate/average-data/energy-intensity": "processing-sold-products:calculate",
    "POST:/api/v1/processing-sold-products/calculate/spend": "processing-sold-products:calculate",
    "POST:/api/v1/processing-sold-products/calculate/hybrid": "processing-sold-products:calculate",
    "POST:/api/v1/processing-sold-products/calculate/batch": "processing-sold-products:calculate",
    "POST:/api/v1/processing-sold-products/calculate/portfolio": "processing-sold-products:calculate",
    "POST:/api/v1/processing-sold-products/compliance/check": "processing-sold-products:compliance",
    "GET:/api/v1/processing-sold-products/calculations/{id}": "processing-sold-products:read",
    "GET:/api/v1/processing-sold-products/calculations": "processing-sold-products:read",
    "DELETE:/api/v1/processing-sold-products/calculations/{id}": "processing-sold-products:delete",
    "GET:/api/v1/processing-sold-products/emission-factors/{category}": "processing-sold-products:read",
    "GET:/api/v1/processing-sold-products/processing-types": "processing-sold-products:read",
    "GET:/api/v1/processing-sold-products/processing-chains": "processing-sold-products:read",
    "GET:/api/v1/processing-sold-products/aggregations": "processing-sold-products:read",
    "GET:/api/v1/processing-sold-products/provenance/{id}": "processing-sold-products:read",
    "GET:/api/v1/processing-sold-products/health": "processing-sold-products:read",

    # ── Use of Sold Products (AGENT-MRV-024 / Scope 3 Cat 11) ──
    "POST:/api/v1/use-of-sold-products/calculate": "use-of-sold-products:calculate",
    "POST:/api/v1/use-of-sold-products/calculate/direct/fuel": "use-of-sold-products:calculate",
    "POST:/api/v1/use-of-sold-products/calculate/direct/refrigerant": "use-of-sold-products:calculate",
    "POST:/api/v1/use-of-sold-products/calculate/direct/chemical": "use-of-sold-products:calculate",
    "POST:/api/v1/use-of-sold-products/calculate/indirect/electricity": "use-of-sold-products:calculate",
    "POST:/api/v1/use-of-sold-products/calculate/indirect/heating": "use-of-sold-products:calculate",
    "POST:/api/v1/use-of-sold-products/calculate/indirect/steam": "use-of-sold-products:calculate",
    "POST:/api/v1/use-of-sold-products/calculate/fuels": "use-of-sold-products:calculate",
    "POST:/api/v1/use-of-sold-products/calculate/batch": "use-of-sold-products:calculate",
    "POST:/api/v1/use-of-sold-products/calculate/portfolio": "use-of-sold-products:calculate",
    "POST:/api/v1/use-of-sold-products/compliance/check": "use-of-sold-products:compliance",
    "GET:/api/v1/use-of-sold-products/calculations/{id}": "use-of-sold-products:read",
    "GET:/api/v1/use-of-sold-products/calculations": "use-of-sold-products:read",
    "DELETE:/api/v1/use-of-sold-products/calculations/{id}": "use-of-sold-products:delete",
    "GET:/api/v1/use-of-sold-products/emission-factors/{category}": "use-of-sold-products:read",
    "GET:/api/v1/use-of-sold-products/energy-profiles": "use-of-sold-products:read",
    "GET:/api/v1/use-of-sold-products/refrigerant-gwps": "use-of-sold-products:read",
    "GET:/api/v1/use-of-sold-products/fuel-factors": "use-of-sold-products:read",
    "GET:/api/v1/use-of-sold-products/lifetime-estimates": "use-of-sold-products:read",
    "GET:/api/v1/use-of-sold-products/aggregations": "use-of-sold-products:read",
    "GET:/api/v1/use-of-sold-products/provenance/{id}": "use-of-sold-products:read",
    "GET:/api/v1/use-of-sold-products/health": "use-of-sold-products:read",

    # ── End-of-Life Treatment (AGENT-MRV-025 / Scope 3 Cat 12) ──
    "POST:/api/v1/end-of-life-treatment/calculate": "end-of-life-treatment:calculate",
    "POST:/api/v1/end-of-life-treatment/calculate/waste-type-specific": "end-of-life-treatment:calculate",
    "POST:/api/v1/end-of-life-treatment/calculate/waste-type-specific/landfill": "end-of-life-treatment:calculate",
    "POST:/api/v1/end-of-life-treatment/calculate/waste-type-specific/incineration": "end-of-life-treatment:calculate",
    "POST:/api/v1/end-of-life-treatment/calculate/waste-type-specific/recycling": "end-of-life-treatment:calculate",
    "POST:/api/v1/end-of-life-treatment/calculate/average-data": "end-of-life-treatment:calculate",
    "POST:/api/v1/end-of-life-treatment/calculate/producer-specific": "end-of-life-treatment:calculate",
    "POST:/api/v1/end-of-life-treatment/calculate/hybrid": "end-of-life-treatment:calculate",
    "POST:/api/v1/end-of-life-treatment/calculate/batch": "end-of-life-treatment:calculate",
    "POST:/api/v1/end-of-life-treatment/calculate/portfolio": "end-of-life-treatment:calculate",
    "POST:/api/v1/end-of-life-treatment/compliance/check": "end-of-life-treatment:compliance",
    "GET:/api/v1/end-of-life-treatment/calculations/{id}": "end-of-life-treatment:read",
    "GET:/api/v1/end-of-life-treatment/calculations": "end-of-life-treatment:read",
    "DELETE:/api/v1/end-of-life-treatment/calculations/{id}": "end-of-life-treatment:delete",
    "GET:/api/v1/end-of-life-treatment/emission-factors/{material}": "end-of-life-treatment:read",
    "GET:/api/v1/end-of-life-treatment/product-compositions": "end-of-life-treatment:read",
    "GET:/api/v1/end-of-life-treatment/treatment-mixes": "end-of-life-treatment:read",
    "GET:/api/v1/end-of-life-treatment/avoided-emissions/{id}": "end-of-life-treatment:read",
    "GET:/api/v1/end-of-life-treatment/circularity-score/{id}": "end-of-life-treatment:read",
    "GET:/api/v1/end-of-life-treatment/aggregations": "end-of-life-treatment:read",
    "GET:/api/v1/end-of-life-treatment/provenance/{id}": "end-of-life-treatment:read",
    "GET:/api/v1/end-of-life-treatment/health": "end-of-life-treatment:read",

    # ── Downstream Leased Assets (AGENT-MRV-026 / Scope 3 Cat 13) ──
    "POST:/api/v1/downstream-leased-assets/calculate": "downstream-leased-assets:calculate",
    "POST:/api/v1/downstream-leased-assets/calculate/asset-specific": "downstream-leased-assets:calculate",
    "POST:/api/v1/downstream-leased-assets/calculate/asset-specific/building": "downstream-leased-assets:calculate",
    "POST:/api/v1/downstream-leased-assets/calculate/asset-specific/vehicle": "downstream-leased-assets:calculate",
    "POST:/api/v1/downstream-leased-assets/calculate/asset-specific/equipment": "downstream-leased-assets:calculate",
    "POST:/api/v1/downstream-leased-assets/calculate/asset-specific/it-asset": "downstream-leased-assets:calculate",
    "POST:/api/v1/downstream-leased-assets/calculate/average-data": "downstream-leased-assets:calculate",
    "POST:/api/v1/downstream-leased-assets/calculate/spend-based": "downstream-leased-assets:calculate",
    "POST:/api/v1/downstream-leased-assets/calculate/hybrid": "downstream-leased-assets:calculate",
    "POST:/api/v1/downstream-leased-assets/calculate/batch": "downstream-leased-assets:calculate",
    "POST:/api/v1/downstream-leased-assets/calculate/portfolio": "downstream-leased-assets:calculate",
    "POST:/api/v1/downstream-leased-assets/compliance/check": "downstream-leased-assets:compliance",
    "GET:/api/v1/downstream-leased-assets/calculations/{id}": "downstream-leased-assets:read",
    "GET:/api/v1/downstream-leased-assets/calculations": "downstream-leased-assets:read",
    "DELETE:/api/v1/downstream-leased-assets/calculations/{id}": "downstream-leased-assets:delete",
    "GET:/api/v1/downstream-leased-assets/emission-factors/{asset_type}": "downstream-leased-assets:read",
    "GET:/api/v1/downstream-leased-assets/building-benchmarks": "downstream-leased-assets:read",
    "GET:/api/v1/downstream-leased-assets/grid-factors": "downstream-leased-assets:read",
    "GET:/api/v1/downstream-leased-assets/allocation-methods": "downstream-leased-assets:read",
    "GET:/api/v1/downstream-leased-assets/aggregations": "downstream-leased-assets:read",
    "GET:/api/v1/downstream-leased-assets/provenance/{id}": "downstream-leased-assets:read",
    "GET:/api/v1/downstream-leased-assets/health": "downstream-leased-assets:read",

    # ── Franchises (AGENT-MRV-027 / Scope 3 Cat 14) ──
    "POST:/api/v1/franchises/calculate": "franchises:calculate",
    "POST:/api/v1/franchises/calculate/franchise-specific": "franchises:calculate",
    "POST:/api/v1/franchises/calculate/franchise-specific/qsr": "franchises:calculate",
    "POST:/api/v1/franchises/calculate/franchise-specific/hotel": "franchises:calculate",
    "POST:/api/v1/franchises/calculate/franchise-specific/convenience": "franchises:calculate",
    "POST:/api/v1/franchises/calculate/franchise-specific/retail": "franchises:calculate",
    "POST:/api/v1/franchises/calculate/franchise-specific/fitness": "franchises:calculate",
    "POST:/api/v1/franchises/calculate/average-data": "franchises:calculate",
    "POST:/api/v1/franchises/calculate/spend-based": "franchises:calculate",
    "POST:/api/v1/franchises/calculate/hybrid": "franchises:calculate",
    "POST:/api/v1/franchises/calculate/batch": "franchises:calculate",
    "POST:/api/v1/franchises/calculate/network": "franchises:calculate",
    "POST:/api/v1/franchises/compliance/check": "franchises:compliance",
    "GET:/api/v1/franchises/calculations/{id}": "franchises:read",
    "GET:/api/v1/franchises/calculations": "franchises:read",
    "DELETE:/api/v1/franchises/calculations/{id}": "franchises:delete",
    "GET:/api/v1/franchises/emission-factors/{franchise_type}": "franchises:read",
    "GET:/api/v1/franchises/franchise-benchmarks": "franchises:read",
    "GET:/api/v1/franchises/grid-factors": "franchises:read",
    "GET:/api/v1/franchises/franchise-types": "franchises:read",
    "GET:/api/v1/franchises/aggregations": "franchises:read",
    "GET:/api/v1/franchises/provenance/{id}": "franchises:read",
    "GET:/api/v1/franchises/health": "franchises:read",

    # ── Investments (AGENT-MRV-028 / Scope 3 Cat 15) ──
    "POST:/api/v1/investments/calculate": "investments:calculate",
    "POST:/api/v1/investments/calculate/equity": "investments:calculate",
    "POST:/api/v1/investments/calculate/private-equity": "investments:calculate",
    "POST:/api/v1/investments/calculate/corporate-bond": "investments:calculate",
    "POST:/api/v1/investments/calculate/project-finance": "investments:calculate",
    "POST:/api/v1/investments/calculate/commercial-real-estate": "investments:calculate",
    "POST:/api/v1/investments/calculate/mortgage": "investments:calculate",
    "POST:/api/v1/investments/calculate/motor-vehicle-loan": "investments:calculate",
    "POST:/api/v1/investments/calculate/sovereign-bond": "investments:calculate",
    "POST:/api/v1/investments/calculate/batch": "investments:calculate",
    "POST:/api/v1/investments/calculate/portfolio": "investments:calculate",
    "POST:/api/v1/investments/compliance/check": "investments:compliance",
    "GET:/api/v1/investments/calculations/{id}": "investments:read",
    "GET:/api/v1/investments/calculations": "investments:read",
    "DELETE:/api/v1/investments/calculations/{id}": "investments:delete",
    "GET:/api/v1/investments/emission-factors/{asset_class}": "investments:read",
    "GET:/api/v1/investments/sector-factors": "investments:read",
    "GET:/api/v1/investments/country-factors": "investments:read",
    "GET:/api/v1/investments/pcaf-quality": "investments:read",
    "GET:/api/v1/investments/carbon-intensity": "investments:read",
    "GET:/api/v1/investments/portfolio-alignment": "investments:read",
    "GET:/api/v1/investments/aggregations": "investments:read",
    "GET:/api/v1/investments/provenance/{id}": "investments:read",
    "GET:/api/v1/investments/health": "investments:read",
    # --- Scope 3 Category Mapper (AGENT-MRV-029) ---
    "POST:/api/v1/scope3-category-mapper/classify": "scope3-category-mapper:classify",
    "POST:/api/v1/scope3-category-mapper/classify/batch": "scope3-category-mapper:classify",
    "POST:/api/v1/scope3-category-mapper/classify/spend": "scope3-category-mapper:classify",
    "POST:/api/v1/scope3-category-mapper/classify/purchase-orders": "scope3-category-mapper:classify",
    "POST:/api/v1/scope3-category-mapper/classify/bom": "scope3-category-mapper:classify",
    "POST:/api/v1/scope3-category-mapper/route": "scope3-category-mapper:route",
    "POST:/api/v1/scope3-category-mapper/route/dry-run": "scope3-category-mapper:route",
    "POST:/api/v1/scope3-category-mapper/boundary/determine": "scope3-category-mapper:boundary",
    "POST:/api/v1/scope3-category-mapper/completeness/screen": "scope3-category-mapper:completeness",
    "POST:/api/v1/scope3-category-mapper/completeness/gap-analysis": "scope3-category-mapper:completeness",
    "POST:/api/v1/scope3-category-mapper/double-counting/check": "scope3-category-mapper:double-counting",
    "POST:/api/v1/scope3-category-mapper/compliance/assess": "scope3-category-mapper:compliance",
    "GET:/api/v1/scope3-category-mapper/categories": "scope3-category-mapper:read",
    "GET:/api/v1/scope3-category-mapper/categories/{number}": "scope3-category-mapper:read",
    "GET:/api/v1/scope3-category-mapper/codes/naics/{code}": "scope3-category-mapper:read",
    "GET:/api/v1/scope3-category-mapper/codes/isic/{code}": "scope3-category-mapper:read",
    "GET:/api/v1/scope3-category-mapper/codes/unspsc/{code}": "scope3-category-mapper:read",
    "GET:/api/v1/scope3-category-mapper/health": "scope3-category-mapper:read",
    "GET:/api/v1/scope3-category-mapper/metrics": "scope3-category-mapper:read",
    # ── Audit Trail & Lineage (AGENT-MRV-030 / Cross-Cutting) ──
    "POST:/api/v1/audit-trail-lineage/events": "audit-trail-lineage:record",
    "POST:/api/v1/audit-trail-lineage/events/batch": "audit-trail-lineage:record",
    "GET:/api/v1/audit-trail-lineage/events/{event_id}": "audit-trail-lineage:read",
    "GET:/api/v1/audit-trail-lineage/events": "audit-trail-lineage:read",
    "DELETE:/api/v1/audit-trail-lineage/events/{event_id}": "audit-trail-lineage:delete",
    "POST:/api/v1/audit-trail-lineage/chain/verify": "audit-trail-lineage:verify",
    "GET:/api/v1/audit-trail-lineage/chain/{org_id}/{year}": "audit-trail-lineage:read",
    "POST:/api/v1/audit-trail-lineage/lineage/nodes": "audit-trail-lineage:record",
    "POST:/api/v1/audit-trail-lineage/lineage/edges": "audit-trail-lineage:record",
    "GET:/api/v1/audit-trail-lineage/lineage/graph/{calc_id}": "audit-trail-lineage:read",
    "POST:/api/v1/audit-trail-lineage/lineage/trace": "audit-trail-lineage:trace",
    "GET:/api/v1/audit-trail-lineage/lineage/visualize/{calc_id}": "audit-trail-lineage:read",
    "POST:/api/v1/audit-trail-lineage/evidence/package": "audit-trail-lineage:package",
    "GET:/api/v1/audit-trail-lineage/evidence/{package_id}": "audit-trail-lineage:read",
    "POST:/api/v1/audit-trail-lineage/evidence/{package_id}/sign": "audit-trail-lineage:sign",
    "POST:/api/v1/audit-trail-lineage/evidence/{package_id}/verify": "audit-trail-lineage:verify",
    "POST:/api/v1/audit-trail-lineage/compliance/trace": "audit-trail-lineage:compliance",
    "GET:/api/v1/audit-trail-lineage/compliance/coverage/{org_id}": "audit-trail-lineage:read",
    "POST:/api/v1/audit-trail-lineage/changes/detect": "audit-trail-lineage:detect",
    "GET:/api/v1/audit-trail-lineage/changes/{change_id}": "audit-trail-lineage:read",
    "GET:/api/v1/audit-trail-lineage/changes/{change_id}/impact": "audit-trail-lineage:read",
    "POST:/api/v1/audit-trail-lineage/pipeline/execute": "audit-trail-lineage:execute",
    "POST:/api/v1/audit-trail-lineage/pipeline/execute/batch": "audit-trail-lineage:execute",
    "GET:/api/v1/audit-trail-lineage/summary/{org_id}/{year}": "audit-trail-lineage:read",
    "GET:/api/v1/audit-trail-lineage/health": "audit-trail-lineage:read",
    # ── ISO 14064 Compliance Platform (GL-ISO14064-APP) ──────────────────
    # Organizations
    "GET:/api/v1/iso14064/organizations": "iso14064-organizations:read",
    "GET:/api/v1/iso14064/organizations/{org_id}": "iso14064-organizations:read",
    "POST:/api/v1/iso14064/organizations": "iso14064-organizations:write",
    "PUT:/api/v1/iso14064/organizations/{org_id}": "iso14064-organizations:write",
    "DELETE:/api/v1/iso14064/organizations/{org_id}": "iso14064-organizations:write",
    # Entities
    "GET:/api/v1/iso14064/entities": "iso14064-entities:read",
    "GET:/api/v1/iso14064/entities/{entity_id}": "iso14064-entities:read",
    "GET:/api/v1/iso14064/organizations/{org_id}/entities": "iso14064-entities:read",
    "POST:/api/v1/iso14064/entities": "iso14064-entities:write",
    "PUT:/api/v1/iso14064/entities/{entity_id}": "iso14064-entities:write",
    "DELETE:/api/v1/iso14064/entities/{entity_id}": "iso14064-entities:write",
    # Boundaries (organizational + operational)
    "GET:/api/v1/iso14064/boundaries/organizational/{org_id}": "iso14064-boundaries:read",
    "POST:/api/v1/iso14064/boundaries/organizational": "iso14064-boundaries:write",
    "PUT:/api/v1/iso14064/boundaries/organizational/{boundary_id}": "iso14064-boundaries:write",
    "GET:/api/v1/iso14064/boundaries/operational/{org_id}": "iso14064-boundaries:read",
    "POST:/api/v1/iso14064/boundaries/operational": "iso14064-boundaries:write",
    "PUT:/api/v1/iso14064/boundaries/operational/{boundary_id}": "iso14064-boundaries:write",
    # Inventories
    "GET:/api/v1/iso14064/inventories": "iso14064-inventories:read",
    "GET:/api/v1/iso14064/inventories/{inventory_id}": "iso14064-inventories:read",
    "GET:/api/v1/iso14064/organizations/{org_id}/inventories": "iso14064-inventories:read",
    "POST:/api/v1/iso14064/inventories": "iso14064-inventories:write",
    "PUT:/api/v1/iso14064/inventories/{inventory_id}": "iso14064-inventories:write",
    "DELETE:/api/v1/iso14064/inventories/{inventory_id}": "iso14064-inventories:write",
    "POST:/api/v1/iso14064/inventories/{inventory_id}/finalize": "iso14064-inventories:write",
    # Emission sources
    "GET:/api/v1/iso14064/emissions": "iso14064-emissions:read",
    "GET:/api/v1/iso14064/emissions/{source_id}": "iso14064-emissions:read",
    "GET:/api/v1/iso14064/inventories/{inventory_id}/emissions": "iso14064-emissions:read",
    "POST:/api/v1/iso14064/emissions": "iso14064-emissions:write",
    "POST:/api/v1/iso14064/emissions/batch": "iso14064-emissions:write",
    "PUT:/api/v1/iso14064/emissions/{source_id}": "iso14064-emissions:write",
    "DELETE:/api/v1/iso14064/emissions/{source_id}": "iso14064-emissions:write",
    "POST:/api/v1/iso14064/emissions/calculate": "iso14064-emissions:write",
    # Removal sources
    "GET:/api/v1/iso14064/removals": "iso14064-removals:read",
    "GET:/api/v1/iso14064/removals/{removal_id}": "iso14064-removals:read",
    "GET:/api/v1/iso14064/inventories/{inventory_id}/removals": "iso14064-removals:read",
    "POST:/api/v1/iso14064/removals": "iso14064-removals:write",
    "POST:/api/v1/iso14064/removals/batch": "iso14064-removals:write",
    "PUT:/api/v1/iso14064/removals/{removal_id}": "iso14064-removals:write",
    "DELETE:/api/v1/iso14064/removals/{removal_id}": "iso14064-removals:write",
    # Significance assessments
    "GET:/api/v1/iso14064/significance": "iso14064-significance:read",
    "GET:/api/v1/iso14064/significance/{assessment_id}": "iso14064-significance:read",
    "GET:/api/v1/iso14064/inventories/{inventory_id}/significance": "iso14064-significance:read",
    "POST:/api/v1/iso14064/significance": "iso14064-significance:write",
    "POST:/api/v1/iso14064/significance/assess": "iso14064-significance:write",
    "PUT:/api/v1/iso14064/significance/{assessment_id}": "iso14064-significance:write",
    # Verification
    "GET:/api/v1/iso14064/verification": "iso14064-verification:read",
    "GET:/api/v1/iso14064/verification/{verification_id}": "iso14064-verification:read",
    "GET:/api/v1/iso14064/inventories/{inventory_id}/verification": "iso14064-verification:read",
    "POST:/api/v1/iso14064/verification": "iso14064-verification:write",
    "PUT:/api/v1/iso14064/verification/{verification_id}": "iso14064-verification:write",
    "GET:/api/v1/iso14064/verification/{verification_id}/findings": "iso14064-verification:read",
    "POST:/api/v1/iso14064/verification/{verification_id}/findings": "iso14064-verification:write",
    "PUT:/api/v1/iso14064/verification/findings/{finding_id}": "iso14064-verification:write",
    # Reports
    "GET:/api/v1/iso14064/reports": "iso14064-reports:read",
    "GET:/api/v1/iso14064/reports/{report_id}": "iso14064-reports:read",
    "GET:/api/v1/iso14064/inventories/{inventory_id}/reports": "iso14064-reports:read",
    "POST:/api/v1/iso14064/reports": "iso14064-reports:write",
    "POST:/api/v1/iso14064/reports/generate": "iso14064-reports:generate",
    "POST:/api/v1/iso14064/reports/generate/{inventory_id}": "iso14064-reports:generate",
    "DELETE:/api/v1/iso14064/reports/{report_id}": "iso14064-reports:write",
    # Management actions
    "GET:/api/v1/iso14064/management-actions": "iso14064-management:read",
    "GET:/api/v1/iso14064/management-actions/{action_id}": "iso14064-management:read",
    "GET:/api/v1/iso14064/organizations/{org_id}/management-actions": "iso14064-management:read",
    "POST:/api/v1/iso14064/management-actions": "iso14064-management:write",
    "PUT:/api/v1/iso14064/management-actions/{action_id}": "iso14064-management:write",
    "DELETE:/api/v1/iso14064/management-actions/{action_id}": "iso14064-management:write",
    # Crosswalk (framework mapping)
    "GET:/api/v1/iso14064/crosswalk": "iso14064-crosswalk:read",
    "GET:/api/v1/iso14064/crosswalk/{framework}": "iso14064-crosswalk:read",
    "POST:/api/v1/iso14064/crosswalk/generate": "iso14064-crosswalk:generate",
    "POST:/api/v1/iso14064/crosswalk/generate/{inventory_id}": "iso14064-crosswalk:generate",
    # Dashboard
    "GET:/api/v1/iso14064/dashboard": "iso14064-dashboard:read",
    "GET:/api/v1/iso14064/dashboard/summary/{org_id}": "iso14064-dashboard:read",
    "GET:/api/v1/iso14064/dashboard/trends/{org_id}": "iso14064-dashboard:read",
    "GET:/api/v1/iso14064/dashboard/categories/{inventory_id}": "iso14064-dashboard:read",
    "GET:/api/v1/iso14064/dashboard/verification-status/{org_id}": "iso14064-dashboard:read",
    # Base year
    "GET:/api/v1/iso14064/base-year": "iso14064-inventories:read",
    "GET:/api/v1/iso14064/base-year/{record_id}": "iso14064-inventories:read",
    "GET:/api/v1/iso14064/organizations/{org_id}/base-year": "iso14064-inventories:read",
    "POST:/api/v1/iso14064/base-year": "iso14064-inventories:write",
    "PUT:/api/v1/iso14064/base-year/{record_id}": "iso14064-inventories:write",
    "GET:/api/v1/iso14064/base-year/triggers/{org_id}": "iso14064-inventories:read",
    "POST:/api/v1/iso14064/base-year/triggers": "iso14064-inventories:write",
    # Health
    "GET:/api/v1/iso14064/health": "iso14064-dashboard:read",
    # ==========================================================================
    # CDP Disclosure Platform (GL-CDP-APP) - APP-007
    # ==========================================================================
    # Questionnaires
    "GET:/api/cdp/questionnaires": "cdp-questionnaires:read",
    "GET:/api/cdp/questionnaires/{id}": "cdp-questionnaires:read",
    "POST:/api/cdp/questionnaires": "cdp-questionnaires:write",
    "GET:/api/cdp/questionnaires/{id}/modules": "cdp-questionnaires:read",
    "GET:/api/cdp/questionnaires/{id}/modules/{module_id}": "cdp-questionnaires:read",
    "GET:/api/cdp/questionnaires/{id}/questions": "cdp-questionnaires:read",
    "GET:/api/cdp/questionnaires/{id}/progress": "cdp-questionnaires:read",
    "PUT:/api/cdp/questionnaires/{id}/sector": "cdp-questionnaires:write",
    "GET:/api/cdp/questionnaires/{id}/conditional-questions": "cdp-questionnaires:read",
    # Responses
    "GET:/api/cdp/responses": "cdp-responses:read",
    "GET:/api/cdp/responses/{id}": "cdp-responses:read",
    "POST:/api/cdp/responses": "cdp-responses:write",
    "PUT:/api/cdp/responses/{id}": "cdp-responses:write",
    "PATCH:/api/cdp/responses/{id}/status": "cdp-responses:write",
    "GET:/api/cdp/responses/{id}/versions": "cdp-responses:read",
    "POST:/api/cdp/responses/{id}/evidence": "cdp-responses:write",
    "DELETE:/api/cdp/responses/{id}/evidence/{evidence_id}": "cdp-responses:write",
    "POST:/api/cdp/responses/{id}/assign": "cdp-responses:write",
    "POST:/api/cdp/responses/{id}/review": "cdp-responses:write",
    "POST:/api/cdp/responses/bulk-import": "cdp-responses:write",
    "POST:/api/cdp/responses/bulk-approve": "cdp-responses:write",
    # Scoring
    "GET:/api/cdp/scoring/{questionnaire_id}": "cdp-scoring:read",
    "GET:/api/cdp/scoring/{questionnaire_id}/categories": "cdp-scoring:read",
    "GET:/api/cdp/scoring/{questionnaire_id}/what-if": "cdp-scoring:read",
    "POST:/api/cdp/scoring/{questionnaire_id}/simulate": "cdp-scoring:write",
    "GET:/api/cdp/scoring/{questionnaire_id}/a-level-check": "cdp-scoring:read",
    "GET:/api/cdp/scoring/{questionnaire_id}/trajectory": "cdp-scoring:read",
    "GET:/api/cdp/scoring/{questionnaire_id}/comparison": "cdp-scoring:read",
    "GET:/api/cdp/scoring/{questionnaire_id}/confidence": "cdp-scoring:read",
    # Gap Analysis
    "GET:/api/cdp/gaps/{questionnaire_id}": "cdp-gaps:read",
    "GET:/api/cdp/gaps/{questionnaire_id}/summary": "cdp-gaps:read",
    "GET:/api/cdp/gaps/{questionnaire_id}/recommendations": "cdp-gaps:read",
    "GET:/api/cdp/gaps/{questionnaire_id}/priority": "cdp-gaps:read",
    "GET:/api/cdp/gaps/{questionnaire_id}/uplift": "cdp-gaps:read",
    "POST:/api/cdp/gaps/{questionnaire_id}/analyze": "cdp-gaps:write",
    "GET:/api/cdp/gaps/{questionnaire_id}/progress": "cdp-gaps:read",
    # Benchmarking
    "GET:/api/cdp/benchmarks/sectors": "cdp-benchmarks:read",
    "GET:/api/cdp/benchmarks/{questionnaire_id}/sector": "cdp-benchmarks:read",
    "GET:/api/cdp/benchmarks/{questionnaire_id}/regional": "cdp-benchmarks:read",
    "GET:/api/cdp/benchmarks/{questionnaire_id}/distribution": "cdp-benchmarks:read",
    "GET:/api/cdp/benchmarks/{questionnaire_id}/categories": "cdp-benchmarks:read",
    "POST:/api/cdp/benchmarks/{questionnaire_id}/custom-peers": "cdp-benchmarks:write",
    # Supply Chain
    "GET:/api/cdp/supply-chain/suppliers": "cdp-supply-chain:read",
    "POST:/api/cdp/supply-chain/suppliers": "cdp-supply-chain:write",
    "POST:/api/cdp/supply-chain/invite": "cdp-supply-chain:write",
    "GET:/api/cdp/supply-chain/responses": "cdp-supply-chain:read",
    "GET:/api/cdp/supply-chain/dashboard": "cdp-supply-chain:read",
    "GET:/api/cdp/supply-chain/emissions": "cdp-supply-chain:read",
    "GET:/api/cdp/supply-chain/hotspots": "cdp-supply-chain:read",
    "POST:/api/cdp/supply-chain/cascade": "cdp-supply-chain:write",
    # Transition Plan
    "GET:/api/cdp/transition-plan/{org_id}": "cdp-transition:read",
    "POST:/api/cdp/transition-plan": "cdp-transition:write",
    "PUT:/api/cdp/transition-plan/{id}": "cdp-transition:write",
    "GET:/api/cdp/transition-plan/{id}/milestones": "cdp-transition:read",
    "POST:/api/cdp/transition-plan/{id}/milestones": "cdp-transition:write",
    "GET:/api/cdp/transition-plan/{id}/sbti-check": "cdp-transition:read",
    "GET:/api/cdp/transition-plan/{id}/progress": "cdp-transition:read",
    "GET:/api/cdp/transition-plan/{id}/investment": "cdp-transition:read",
    # Reporting
    "GET:/api/cdp/reports/{questionnaire_id}/checklist": "cdp-reports:read",
    "POST:/api/cdp/reports/{questionnaire_id}/generate/pdf": "cdp-reports:generate",
    "POST:/api/cdp/reports/{questionnaire_id}/generate/excel": "cdp-reports:generate",
    "POST:/api/cdp/reports/{questionnaire_id}/generate/xml": "cdp-reports:generate",
    "GET:/api/cdp/reports/{questionnaire_id}/summary": "cdp-reports:read",
    "POST:/api/cdp/reports/{questionnaire_id}/submit": "cdp-reports:submit",
    "GET:/api/cdp/reports/history": "cdp-reports:read",
    # Dashboard
    "GET:/api/cdp/dashboard/{org_id}": "cdp-dashboard:read",
    "GET:/api/cdp/dashboard/{org_id}/score": "cdp-dashboard:read",
    "GET:/api/cdp/dashboard/{org_id}/progress": "cdp-dashboard:read",
    "GET:/api/cdp/dashboard/{org_id}/timeline": "cdp-dashboard:read",
    "GET:/api/cdp/dashboard/{org_id}/activity": "cdp-dashboard:read",
    "GET:/api/cdp/dashboard/{org_id}/readiness": "cdp-dashboard:read",
    # Settings
    "GET:/api/cdp/settings/{org_id}": "cdp-settings:read",
    "PUT:/api/cdp/settings/{org_id}": "cdp-settings:write",
    "GET:/api/cdp/settings/{org_id}/team": "cdp-settings:read",
    "POST:/api/cdp/settings/{org_id}/team": "cdp-settings:write",
    "PUT:/api/cdp/settings/{org_id}/integrations": "cdp-settings:write",
    # Health
    "GET:/api/cdp/health": "cdp-dashboard:read",

    # =========================================================================
    # GL-TCFD-APP v1.0 -- TCFD Climate Disclosure Platform (APP-008)
    # =========================================================================

    # Governance
    "POST:/api/v1/tcfd/governance/assessments": "tcfd-governance:write",
    "GET:/api/v1/tcfd/governance/assessments": "tcfd-governance:read",
    "GET:/api/v1/tcfd/governance/assessments/{id}": "tcfd-governance:read",
    "PUT:/api/v1/tcfd/governance/assessments/{id}/board": "tcfd-governance:write",
    "PUT:/api/v1/tcfd/governance/assessments/{id}/management": "tcfd-governance:write",
    "GET:/api/v1/tcfd/governance/assessments/{id}/maturity": "tcfd-governance:read",
    "PUT:/api/v1/tcfd/governance/assessments/{id}/competency": "tcfd-governance:write",
    "PUT:/api/v1/tcfd/governance/assessments/{id}/incentives": "tcfd-governance:write",
    "GET:/api/v1/tcfd/governance/assessments/{id}/disclosure": "tcfd-governance:read",
    "GET:/api/v1/tcfd/governance/assessments/{id}/summary": "tcfd-governance:read",

    # Strategy
    "POST:/api/v1/tcfd/strategy/risks": "tcfd-strategy:write",
    "GET:/api/v1/tcfd/strategy/risks": "tcfd-strategy:read",
    "GET:/api/v1/tcfd/strategy/risks/{id}": "tcfd-strategy:read",
    "PUT:/api/v1/tcfd/strategy/risks/{id}": "tcfd-strategy:write",
    "DELETE:/api/v1/tcfd/strategy/risks/{id}": "tcfd-strategy:delete",
    "POST:/api/v1/tcfd/strategy/opportunities": "tcfd-strategy:write",
    "GET:/api/v1/tcfd/strategy/opportunities": "tcfd-strategy:read",
    "GET:/api/v1/tcfd/strategy/opportunities/{id}": "tcfd-strategy:read",
    "PUT:/api/v1/tcfd/strategy/opportunities/{id}": "tcfd-strategy:write",
    "GET:/api/v1/tcfd/strategy/business-impact/{org_id}": "tcfd-strategy:read",
    "GET:/api/v1/tcfd/strategy/value-chain/{org_id}": "tcfd-strategy:read",
    "POST:/api/v1/tcfd/strategy/responses": "tcfd-strategy:write",
    "GET:/api/v1/tcfd/strategy/disclosure/{org_id}": "tcfd-strategy:read",

    # Scenarios
    "GET:/api/v1/tcfd/scenarios/prebuilt": "tcfd-scenarios:read",
    "POST:/api/v1/tcfd/scenarios": "tcfd-scenarios:write",
    "GET:/api/v1/tcfd/scenarios/{id}": "tcfd-scenarios:read",
    "PUT:/api/v1/tcfd/scenarios/{id}": "tcfd-scenarios:write",
    "DELETE:/api/v1/tcfd/scenarios/{id}": "tcfd-scenarios:delete",
    "POST:/api/v1/tcfd/scenarios/analyze": "tcfd-scenarios:execute",
    "GET:/api/v1/tcfd/scenarios/results/{org_id}": "tcfd-scenarios:read",
    "GET:/api/v1/tcfd/scenarios/compare": "tcfd-scenarios:read",
    "POST:/api/v1/tcfd/scenarios/sensitivity": "tcfd-scenarios:execute",
    "GET:/api/v1/tcfd/scenarios/resilience/{org_id}": "tcfd-scenarios:read",
    "POST:/api/v1/tcfd/scenarios/weighted-impact": "tcfd-scenarios:execute",
    "GET:/api/v1/tcfd/scenarios/{id}/narrative": "tcfd-scenarios:read",

    # Physical Risk
    "POST:/api/v1/tcfd/physical-risk/assets": "tcfd-physical-risk:write",
    "GET:/api/v1/tcfd/physical-risk/assets": "tcfd-physical-risk:read",
    "GET:/api/v1/tcfd/physical-risk/assets/{id}": "tcfd-physical-risk:read",
    "PUT:/api/v1/tcfd/physical-risk/assets/{id}": "tcfd-physical-risk:write",
    "POST:/api/v1/tcfd/physical-risk/assess/{asset_id}": "tcfd-physical-risk:execute",
    "GET:/api/v1/tcfd/physical-risk/assess/{asset_id}": "tcfd-physical-risk:read",
    "GET:/api/v1/tcfd/physical-risk/portfolio/{org_id}": "tcfd-physical-risk:read",
    "GET:/api/v1/tcfd/physical-risk/supply-chain/{org_id}": "tcfd-physical-risk:read",
    "POST:/api/v1/tcfd/physical-risk/project/{asset_id}": "tcfd-physical-risk:execute",
    "GET:/api/v1/tcfd/physical-risk/damage/{asset_id}": "tcfd-physical-risk:read",
    "GET:/api/v1/tcfd/physical-risk/insurance/{asset_id}": "tcfd-physical-risk:read",

    # Transition Risk
    "POST:/api/v1/tcfd/transition-risk/assess/{org_id}": "tcfd-transition-risk:execute",
    "GET:/api/v1/tcfd/transition-risk/policy/{org_id}": "tcfd-transition-risk:read",
    "GET:/api/v1/tcfd/transition-risk/technology/{org_id}": "tcfd-transition-risk:read",
    "GET:/api/v1/tcfd/transition-risk/market/{org_id}": "tcfd-transition-risk:read",
    "GET:/api/v1/tcfd/transition-risk/reputation/{org_id}": "tcfd-transition-risk:read",
    "GET:/api/v1/tcfd/transition-risk/composite/{org_id}": "tcfd-transition-risk:read",
    "GET:/api/v1/tcfd/transition-risk/carbon-impact/{org_id}": "tcfd-transition-risk:read",
    "GET:/api/v1/tcfd/transition-risk/stranding/{org_id}": "tcfd-transition-risk:read",
    "GET:/api/v1/tcfd/transition-risk/sector-profile/{sector}": "tcfd-transition-risk:read",
    "GET:/api/v1/tcfd/transition-risk/maturity/{org_id}": "tcfd-transition-risk:read",

    # Opportunities
    "POST:/api/v1/tcfd/opportunities/assess/{org_id}": "tcfd-opportunities:execute",
    "GET:/api/v1/tcfd/opportunities/sizing/{opp_id}": "tcfd-opportunities:read",
    "GET:/api/v1/tcfd/opportunities/cost-savings/{opp_id}": "tcfd-opportunities:read",
    "GET:/api/v1/tcfd/opportunities/investment/{opp_id}": "tcfd-opportunities:read",
    "GET:/api/v1/tcfd/opportunities/pipeline/{org_id}": "tcfd-opportunities:read",
    "PUT:/api/v1/tcfd/opportunities/pipeline/{opp_id}/stage": "tcfd-opportunities:write",
    "GET:/api/v1/tcfd/opportunities/priorities/{org_id}": "tcfd-opportunities:read",
    "GET:/api/v1/tcfd/opportunities/green-revenue/{org_id}": "tcfd-opportunities:read",
    "GET:/api/v1/tcfd/opportunities/summary/{org_id}": "tcfd-opportunities:read",

    # Financial Impact
    "POST:/api/v1/tcfd/financial/impact/{org_id}": "tcfd-financial:execute",
    "GET:/api/v1/tcfd/financial/income-statement/{org_id}": "tcfd-financial:read",
    "GET:/api/v1/tcfd/financial/balance-sheet/{org_id}": "tcfd-financial:read",
    "GET:/api/v1/tcfd/financial/cash-flow/{org_id}": "tcfd-financial:read",
    "GET:/api/v1/tcfd/financial/total/{org_id}": "tcfd-financial:read",
    "POST:/api/v1/tcfd/financial/npv": "tcfd-financial:execute",
    "POST:/api/v1/tcfd/financial/macc/{org_id}": "tcfd-financial:execute",
    "POST:/api/v1/tcfd/financial/carbon-sensitivity/{org_id}": "tcfd-financial:execute",
    "POST:/api/v1/tcfd/financial/monte-carlo/{org_id}": "tcfd-financial:execute",
    "GET:/api/v1/tcfd/financial/climate-var/{org_id}": "tcfd-financial:read",
    "GET:/api/v1/tcfd/financial/projections/{org_id}": "tcfd-financial:read",

    # Risk Management
    "POST:/api/v1/tcfd/risk-management/register/{org_id}": "tcfd-risk-mgmt:write",
    "GET:/api/v1/tcfd/risk-management/register/{org_id}": "tcfd-risk-mgmt:read",
    "POST:/api/v1/tcfd/risk-management/register/{register_id}/entry": "tcfd-risk-mgmt:write",
    "PUT:/api/v1/tcfd/risk-management/entry/{entry_id}": "tcfd-risk-mgmt:write",
    "PUT:/api/v1/tcfd/risk-management/entry/{entry_id}/assess": "tcfd-risk-mgmt:write",
    "POST:/api/v1/tcfd/risk-management/entry/{entry_id}/response": "tcfd-risk-mgmt:write",
    "GET:/api/v1/tcfd/risk-management/heat-map/{register_id}": "tcfd-risk-mgmt:read",
    "GET:/api/v1/tcfd/risk-management/indicators/{entry_id}": "tcfd-risk-mgmt:read",
    "PUT:/api/v1/tcfd/risk-management/erm/{org_id}": "tcfd-risk-mgmt:write",
    "GET:/api/v1/tcfd/risk-management/summary/{org_id}": "tcfd-risk-mgmt:read",
    "GET:/api/v1/tcfd/risk-management/disclosure/{org_id}": "tcfd-risk-mgmt:read",

    # Metrics & Targets
    "GET:/api/v1/tcfd/metrics/emissions/{org_id}": "tcfd-metrics:read",
    "GET:/api/v1/tcfd/metrics/intensity/{org_id}": "tcfd-metrics:read",
    "GET:/api/v1/tcfd/metrics/cross-industry/{org_id}": "tcfd-metrics:read",
    "GET:/api/v1/tcfd/metrics/industry/{org_id}": "tcfd-metrics:read",
    "POST:/api/v1/tcfd/metrics/custom/{org_id}": "tcfd-metrics:write",
    "POST:/api/v1/tcfd/metrics/values/{metric_id}": "tcfd-metrics:write",
    "POST:/api/v1/tcfd/metrics/targets/{org_id}": "tcfd-metrics:write",
    "GET:/api/v1/tcfd/metrics/targets/{org_id}": "tcfd-metrics:read",
    "PUT:/api/v1/tcfd/metrics/targets/{target_id}/progress": "tcfd-metrics:write",
    "GET:/api/v1/tcfd/metrics/sbti/{org_id}": "tcfd-metrics:read",
    "GET:/api/v1/tcfd/metrics/benchmark/{org_id}": "tcfd-metrics:read",
    "GET:/api/v1/tcfd/metrics/temperature/{org_id}": "tcfd-metrics:read",
    "GET:/api/v1/tcfd/metrics/summary/{org_id}": "tcfd-metrics:read",
    "GET:/api/v1/tcfd/metrics/disclosure/{org_id}": "tcfd-metrics:read",

    # Disclosures
    "POST:/api/v1/tcfd/disclosures": "tcfd-disclosures:write",
    "GET:/api/v1/tcfd/disclosures": "tcfd-disclosures:read",
    "GET:/api/v1/tcfd/disclosures/{id}": "tcfd-disclosures:read",
    "PUT:/api/v1/tcfd/disclosures/{id}/section/{code}": "tcfd-disclosures:write",
    "GET:/api/v1/tcfd/disclosures/{id}/compliance": "tcfd-disclosures:read",
    "POST:/api/v1/tcfd/disclosures/{id}/evidence": "tcfd-disclosures:write",
    "GET:/api/v1/tcfd/disclosures/{id}/report": "tcfd-disclosures:read",
    "GET:/api/v1/tcfd/disclosures/{id}/export/pdf": "tcfd-disclosures:export",
    "GET:/api/v1/tcfd/disclosures/{id}/export/excel": "tcfd-disclosures:export",
    "GET:/api/v1/tcfd/disclosures/{id}/export/json": "tcfd-disclosures:export",
    "GET:/api/v1/tcfd/disclosures/{id}/export/xbrl": "tcfd-disclosures:export",
    "PUT:/api/v1/tcfd/disclosures/{id}/approve": "tcfd-disclosures:approve",
    "PUT:/api/v1/tcfd/disclosures/{id}/publish": "tcfd-disclosures:publish",
    "GET:/api/v1/tcfd/disclosures/compare": "tcfd-disclosures:read",

    # Dashboard
    "GET:/api/v1/tcfd/dashboard/summary/{org_id}": "tcfd-dashboard:read",
    "GET:/api/v1/tcfd/dashboard/risk-exposure/{org_id}": "tcfd-dashboard:read",
    "GET:/api/v1/tcfd/dashboard/opportunity-value/{org_id}": "tcfd-dashboard:read",
    "GET:/api/v1/tcfd/dashboard/scenario-impact/{org_id}": "tcfd-dashboard:read",
    "GET:/api/v1/tcfd/dashboard/disclosure-maturity/{org_id}": "tcfd-dashboard:read",
    "GET:/api/v1/tcfd/dashboard/metrics-summary/{org_id}": "tcfd-dashboard:read",
    "GET:/api/v1/tcfd/dashboard/trend/{org_id}": "tcfd-dashboard:read",

    # Gap Analysis
    "POST:/api/v1/tcfd/gap-analysis/assess/{org_id}": "tcfd-gap-analysis:execute",
    "GET:/api/v1/tcfd/gap-analysis/maturity/{org_id}": "tcfd-gap-analysis:read",
    "GET:/api/v1/tcfd/gap-analysis/gaps/{org_id}": "tcfd-gap-analysis:read",
    "GET:/api/v1/tcfd/gap-analysis/benchmark/{org_id}": "tcfd-gap-analysis:read",
    "POST:/api/v1/tcfd/gap-analysis/action-plan/{org_id}": "tcfd-gap-analysis:write",
    "GET:/api/v1/tcfd/gap-analysis/progress/{org_id}": "tcfd-gap-analysis:read",
    "GET:/api/v1/tcfd/gap-analysis/recommendations/{org_id}": "tcfd-gap-analysis:read",

    # ISSB Cross-Walk
    "GET:/api/v1/tcfd/issb/mapping": "tcfd-issb:read",
    "GET:/api/v1/tcfd/issb/compliance/{org_id}": "tcfd-issb:read",
    "GET:/api/v1/tcfd/issb/gaps/{org_id}": "tcfd-issb:read",
    "GET:/api/v1/tcfd/issb/additional-requirements": "tcfd-issb:read",
    "GET:/api/v1/tcfd/issb/migration/{org_id}": "tcfd-issb:read",
    "GET:/api/v1/tcfd/issb/scorecard/{org_id}": "tcfd-issb:read",

    # Settings
    "GET:/api/v1/tcfd/settings/{org_id}": "tcfd-settings:read",
    "PUT:/api/v1/tcfd/settings/{org_id}": "tcfd-settings:write",
    "GET:/api/v1/tcfd/settings/scenarios/defaults": "tcfd-settings:read",
    "PUT:/api/v1/tcfd/settings/{org_id}/scenarios": "tcfd-settings:write",
    "GET:/api/v1/tcfd/settings/supported-jurisdictions": "tcfd-settings:read",

    # ── GL-SBTi-APP v1.0 (APP-009) ──────────────────────────────────────
    # Target Configuration
    "GET:/api/v1/sbti/targets": "sbti-targets:read",
    "GET:/api/v1/sbti/targets/{target_id}": "sbti-targets:read",
    "POST:/api/v1/sbti/targets": "sbti-targets:write",
    "PUT:/api/v1/sbti/targets/{target_id}": "sbti-targets:write",
    "DELETE:/api/v1/sbti/targets/{target_id}": "sbti-targets:delete",
    "PUT:/api/v1/sbti/targets/{target_id}/status": "sbti-targets:write",
    "GET:/api/v1/sbti/targets/{target_id}/summary": "sbti-targets:read",
    "POST:/api/v1/sbti/targets/{target_id}/submission": "sbti-targets:write",
    "GET:/api/v1/sbti/targets/org/{org_id}/scope3-requirement": "sbti-targets:read",
    "POST:/api/v1/sbti/targets/org/{org_id}/coverage-check": "sbti-targets:read",

    # Pathway Calculator
    "POST:/api/v1/sbti/pathways/aca": "sbti-pathways:execute",
    "POST:/api/v1/sbti/pathways/sda": "sbti-pathways:execute",
    "POST:/api/v1/sbti/pathways/economic-intensity": "sbti-pathways:execute",
    "POST:/api/v1/sbti/pathways/physical-intensity": "sbti-pathways:execute",
    "POST:/api/v1/sbti/pathways/supplier-engagement": "sbti-pathways:execute",
    "POST:/api/v1/sbti/pathways/flag-commodity": "sbti-pathways:execute",
    "POST:/api/v1/sbti/pathways/flag-sector": "sbti-pathways:execute",
    "POST:/api/v1/sbti/pathways/compare": "sbti-pathways:read",
    "GET:/api/v1/sbti/pathways/{pathway_id}/milestones": "sbti-pathways:read",

    # Validation
    "POST:/api/v1/sbti/validation/validate": "sbti-validation:execute",
    "GET:/api/v1/sbti/validation/{target_id}/results": "sbti-validation:read",
    "GET:/api/v1/sbti/validation/{target_id}/checklist": "sbti-validation:read",
    "GET:/api/v1/sbti/validation/{target_id}/readiness": "sbti-validation:read",
    "POST:/api/v1/sbti/validation/criteria/{criterion_id}/check": "sbti-validation:execute",
    "GET:/api/v1/sbti/validation/criteria": "sbti-validation:read",
    "POST:/api/v1/sbti/validation/net-zero/validate": "sbti-validation:execute",

    # Scope 3 Screening
    "POST:/api/v1/sbti/scope3/trigger-assessment": "sbti-scope3:execute",
    "GET:/api/v1/sbti/scope3/org/{org_id}/category-breakdown": "sbti-scope3:read",
    "POST:/api/v1/sbti/scope3/hotspot-analysis": "sbti-scope3:execute",
    "POST:/api/v1/sbti/scope3/coverage-calculator": "sbti-scope3:execute",
    "GET:/api/v1/sbti/scope3/org/{org_id}/recommendations": "sbti-scope3:read",
    "GET:/api/v1/sbti/scope3/categories": "sbti-scope3:read",
    "GET:/api/v1/sbti/scope3/org/{org_id}/data-quality": "sbti-scope3:read",

    # FLAG Assessment
    "POST:/api/v1/sbti/flag/trigger-assessment": "sbti-flag:execute",
    "GET:/api/v1/sbti/flag/org/{org_id}/classification": "sbti-flag:read",
    "POST:/api/v1/sbti/flag/commodity-pathway": "sbti-flag:execute",
    "POST:/api/v1/sbti/flag/sector-pathway": "sbti-flag:execute",
    "GET:/api/v1/sbti/flag/commodities": "sbti-flag:read",
    "PUT:/api/v1/sbti/flag/org/{org_id}/deforestation-commitment": "sbti-flag:write",
    "GET:/api/v1/sbti/flag/org/{org_id}/emissions-split": "sbti-flag:read",
    "POST:/api/v1/sbti/flag/removals-eligibility": "sbti-flag:execute",

    # Sector Pathways
    "GET:/api/v1/sbti/sectors": "sbti-sectors:read",
    "GET:/api/v1/sbti/sectors/{sector}/pathway": "sbti-sectors:read",
    "POST:/api/v1/sbti/sectors/{sector}/calculate": "sbti-sectors:execute",
    "POST:/api/v1/sbti/sectors/detect": "sbti-sectors:execute",
    "POST:/api/v1/sbti/sectors/blend": "sbti-sectors:execute",
    "GET:/api/v1/sbti/sectors/{sector}/benchmarks": "sbti-sectors:read",

    # Progress Tracking
    "POST:/api/v1/sbti/progress": "sbti-progress:write",
    "GET:/api/v1/sbti/progress/target/{target_id}/history": "sbti-progress:read",
    "GET:/api/v1/sbti/progress/target/{target_id}/variance/{year}": "sbti-progress:read",
    "GET:/api/v1/sbti/progress/target/{target_id}/status": "sbti-progress:read",
    "GET:/api/v1/sbti/progress/target/{target_id}/cumulative": "sbti-progress:read",
    "GET:/api/v1/sbti/progress/target/{target_id}/projection": "sbti-progress:read",
    "GET:/api/v1/sbti/progress/org/{org_id}/dashboard": "sbti-progress:read",
    "GET:/api/v1/sbti/progress/target/{target_id}/scope-breakdown/{year}": "sbti-progress:read",

    # Temperature Scoring
    "GET:/api/v1/sbti/temperature/org/{org_id}/score": "sbti-temperature:read",
    "GET:/api/v1/sbti/temperature/org/{org_id}/time-series": "sbti-temperature:read",
    "GET:/api/v1/sbti/temperature/org/{org_id}/peer-ranking": "sbti-temperature:read",
    "POST:/api/v1/sbti/temperature/portfolio": "sbti-temperature:execute",
    "GET:/api/v1/sbti/temperature/org/{org_id}/report": "sbti-temperature:read",

    # Recalculation
    "POST:/api/v1/sbti/recalculation/threshold-check": "sbti-recalculation:execute",
    "POST:/api/v1/sbti/recalculation": "sbti-recalculation:write",
    "GET:/api/v1/sbti/recalculation/org/{org_id}/history": "sbti-recalculation:read",
    "GET:/api/v1/sbti/recalculation/{recalculation_id}/revalidation": "sbti-recalculation:read",
    "POST:/api/v1/sbti/recalculation/ma-impact": "sbti-recalculation:execute",
    "GET:/api/v1/sbti/recalculation/{recalculation_id}/audit": "sbti-recalculation:read",

    # Five-Year Review
    "POST:/api/v1/sbti/reviews": "sbti-reviews:write",
    "GET:/api/v1/sbti/reviews/{review_id}": "sbti-reviews:read",
    "GET:/api/v1/sbti/reviews/org/{org_id}/upcoming": "sbti-reviews:read",
    "GET:/api/v1/sbti/reviews/{review_id}/readiness": "sbti-reviews:read",
    "PUT:/api/v1/sbti/reviews/{review_id}/outcome": "sbti-reviews:write",
    "GET:/api/v1/sbti/reviews/org/{org_id}/history": "sbti-reviews:read",
    "GET:/api/v1/sbti/reviews/deadlines/alerts": "sbti-reviews:read",

    # Financial Institutions
    "POST:/api/v1/sbti/financial-institutions/portfolios": "sbti-fi:write",
    "GET:/api/v1/sbti/financial-institutions/portfolios/{portfolio_id}": "sbti-fi:read",
    "POST:/api/v1/sbti/financial-institutions/portfolios/{portfolio_id}/holdings": "sbti-fi:write",
    "GET:/api/v1/sbti/financial-institutions/portfolios/{portfolio_id}/coverage": "sbti-fi:read",
    "GET:/api/v1/sbti/financial-institutions/portfolios/{portfolio_id}/financed-emissions": "sbti-fi:read",
    "GET:/api/v1/sbti/financial-institutions/portfolios/{portfolio_id}/waci": "sbti-fi:read",
    "GET:/api/v1/sbti/financial-institutions/portfolios/{portfolio_id}/temperature": "sbti-fi:read",
    "GET:/api/v1/sbti/financial-institutions/portfolios/{portfolio_id}/pcaf": "sbti-fi:read",
    "POST:/api/v1/sbti/financial-institutions/portfolios/{portfolio_id}/engagement": "sbti-fi:write",
    "GET:/api/v1/sbti/financial-institutions/portfolios/{portfolio_id}/coverage-pathway": "sbti-fi:read",
    "POST:/api/v1/sbti/financial-institutions/portfolios/{portfolio_id}/finz-validation": "sbti-fi:execute",
    "GET:/api/v1/sbti/financial-institutions/portfolios/{portfolio_id}/asset-class-breakdown": "sbti-fi:read",

    # Framework Alignment
    "GET:/api/v1/sbti/frameworks/org/{org_id}/alignment": "sbti-frameworks:read",
    "GET:/api/v1/sbti/frameworks/org/{org_id}/{framework}/mapping": "sbti-frameworks:read",
    "GET:/api/v1/sbti/frameworks/org/{org_id}/gaps": "sbti-frameworks:read",
    "POST:/api/v1/sbti/frameworks/org/{org_id}/unified-report": "sbti-frameworks:execute",

    # Reporting
    "POST:/api/v1/sbti/reports/submission-form": "sbti-reports:write",
    "POST:/api/v1/sbti/reports/progress": "sbti-reports:write",
    "POST:/api/v1/sbti/reports/validation-readiness": "sbti-reports:write",
    "POST:/api/v1/sbti/reports/temperature": "sbti-reports:write",
    "POST:/api/v1/sbti/reports/executive-summary": "sbti-reports:write",
    "GET:/api/v1/sbti/reports/org/{org_id}/history": "sbti-reports:read",
    "GET:/api/v1/sbti/reports/{report_id}/export/{format}": "sbti-reports:read",

    # Dashboard
    "GET:/api/v1/sbti/dashboard/org/{org_id}/readiness": "sbti-dashboard:read",
    "GET:/api/v1/sbti/dashboard/org/{org_id}/target-summary": "sbti-dashboard:read",
    "GET:/api/v1/sbti/dashboard/org/{org_id}/pathway-overview": "sbti-dashboard:read",
    "GET:/api/v1/sbti/dashboard/org/{org_id}/temperature": "sbti-dashboard:read",
    "GET:/api/v1/sbti/dashboard/org/{org_id}/review-countdown": "sbti-dashboard:read",
    "GET:/api/v1/sbti/dashboard/org/{org_id}/milestones": "sbti-dashboard:read",

    # Gap Analysis
    "POST:/api/v1/sbti/gap-analysis/org/{org_id}/run": "sbti-gap-analysis:execute",
    "GET:/api/v1/sbti/gap-analysis/org/{org_id}/results": "sbti-gap-analysis:read",
    "GET:/api/v1/sbti/gap-analysis/org/{org_id}/data-gaps": "sbti-gap-analysis:read",
    "GET:/api/v1/sbti/gap-analysis/org/{org_id}/ambition-gaps": "sbti-gap-analysis:read",
    "GET:/api/v1/sbti/gap-analysis/org/{org_id}/action-plan": "sbti-gap-analysis:read",
    "GET:/api/v1/sbti/gap-analysis/org/{org_id}/readiness-score": "sbti-gap-analysis:read",
    "GET:/api/v1/sbti/gap-analysis/org/{org_id}/benchmarks": "sbti-gap-analysis:read",

    # Settings
    "GET:/api/v1/sbti/settings/org/{org_id}": "sbti-settings:read",
    "PUT:/api/v1/sbti/settings/org/{org_id}": "sbti-settings:write",
    "GET:/api/v1/sbti/settings/org/{org_id}/sector": "sbti-settings:read",
    "PUT:/api/v1/sbti/settings/org/{org_id}/sector": "sbti-settings:write",
    "GET:/api/v1/sbti/settings/org/{org_id}/frameworks": "sbti-settings:read",
    "PUT:/api/v1/sbti/settings/org/{org_id}/frameworks": "sbti-settings:write",
    "GET:/api/v1/sbti/settings/org/{org_id}/mrv-connection": "sbti-settings:read",
    "PUT:/api/v1/sbti/settings/org/{org_id}/mrv-connection": "sbti-settings:write",
    "GET:/api/v1/sbti/settings/org/{org_id}/notifications": "sbti-settings:read",
    "PUT:/api/v1/sbti/settings/org/{org_id}/notifications": "sbti-settings:write",
    "GET:/api/v1/sbti/settings/criteria/definitions": "sbti-settings:read",
    "GET:/api/v1/sbti/settings/sectors/list": "sbti-settings:read",
    "GET:/api/v1/sbti/settings/flag-commodities/list": "sbti-settings:read",

    # ── GL-Taxonomy-APP v1.0 (APP-010) ─────────────────────────────────────
    # Activity Catalog
    "GET:/api/v1/taxonomy/activities": "taxonomy-activities:read",
    "GET:/api/v1/taxonomy/activities/statistics": "taxonomy-activities:read",
    "GET:/api/v1/taxonomy/activities/sectors": "taxonomy-activities:read",
    "GET:/api/v1/taxonomy/activities/search": "taxonomy-activities:read",
    "GET:/api/v1/taxonomy/activities/sector/{sector}": "taxonomy-activities:read",
    "GET:/api/v1/taxonomy/activities/objective/{objective}": "taxonomy-activities:read",
    "GET:/api/v1/taxonomy/activities/nace/{nace_code}": "taxonomy-activities:read",
    "GET:/api/v1/taxonomy/activities/{activity_code}": "taxonomy-activities:read",

    # Eligibility Screening
    "POST:/api/v1/taxonomy/screening/eligibility": "taxonomy-screening:execute",
    "POST:/api/v1/taxonomy/screening/batch": "taxonomy-screening:execute",
    "GET:/api/v1/taxonomy/screening/{org_id}/results": "taxonomy-screening:read",
    "GET:/api/v1/taxonomy/screening/{org_id}/summary": "taxonomy-screening:read",
    "POST:/api/v1/taxonomy/screening/{org_id}/de-minimis": "taxonomy-screening:execute",
    "GET:/api/v1/taxonomy/screening/{org_id}/sector-breakdown": "taxonomy-screening:read",
    "DELETE:/api/v1/taxonomy/screening/{screening_id}": "taxonomy-screening:delete",

    # Substantial Contribution
    "POST:/api/v1/taxonomy/substantial-contribution/assess": "taxonomy-sc:execute",
    "POST:/api/v1/taxonomy/substantial-contribution/batch": "taxonomy-sc:execute",
    "GET:/api/v1/taxonomy/substantial-contribution/{org_id}/results": "taxonomy-sc:read",
    "GET:/api/v1/taxonomy/substantial-contribution/{activity_code}/criteria": "taxonomy-sc:read",
    "GET:/api/v1/taxonomy/substantial-contribution/{activity_code}/profile": "taxonomy-sc:read",
    "POST:/api/v1/taxonomy/substantial-contribution/threshold-check": "taxonomy-sc:execute",
    "POST:/api/v1/taxonomy/substantial-contribution/{assessment_id}/evidence": "taxonomy-sc:write",
    "GET:/api/v1/taxonomy/substantial-contribution/{org_id}/summary": "taxonomy-sc:read",

    # DNSH Assessment
    "POST:/api/v1/taxonomy/dnsh/assess": "taxonomy-dnsh:execute",
    "POST:/api/v1/taxonomy/dnsh/assess/{objective}": "taxonomy-dnsh:execute",
    "POST:/api/v1/taxonomy/dnsh/climate-risk": "taxonomy-dnsh:execute",
    "POST:/api/v1/taxonomy/dnsh/water": "taxonomy-dnsh:execute",
    "POST:/api/v1/taxonomy/dnsh/circular-economy": "taxonomy-dnsh:execute",
    "POST:/api/v1/taxonomy/dnsh/pollution": "taxonomy-dnsh:execute",
    "POST:/api/v1/taxonomy/dnsh/biodiversity": "taxonomy-dnsh:execute",
    "GET:/api/v1/taxonomy/dnsh/{activity_code}/matrix": "taxonomy-dnsh:read",
    "GET:/api/v1/taxonomy/dnsh/{org_id}/summary": "taxonomy-dnsh:read",

    # Minimum Safeguards
    "POST:/api/v1/taxonomy/safeguards/assess": "taxonomy-safeguards:execute",
    "POST:/api/v1/taxonomy/safeguards/assess/{topic}": "taxonomy-safeguards:execute",
    "POST:/api/v1/taxonomy/safeguards/{org_id}/procedural": "taxonomy-safeguards:write",
    "POST:/api/v1/taxonomy/safeguards/{org_id}/outcome": "taxonomy-safeguards:write",
    "POST:/api/v1/taxonomy/safeguards/{org_id}/adverse-finding": "taxonomy-safeguards:write",
    "GET:/api/v1/taxonomy/safeguards/{org_id}/results": "taxonomy-safeguards:read",
    "GET:/api/v1/taxonomy/safeguards/{org_id}/summary": "taxonomy-safeguards:read",

    # KPI Calculation
    "POST:/api/v1/taxonomy/kpi/calculate": "taxonomy-kpi:execute",
    "POST:/api/v1/taxonomy/kpi/turnover": "taxonomy-kpi:execute",
    "POST:/api/v1/taxonomy/kpi/capex": "taxonomy-kpi:execute",
    "POST:/api/v1/taxonomy/kpi/opex": "taxonomy-kpi:execute",
    "POST:/api/v1/taxonomy/kpi/capex-plan": "taxonomy-kpi:execute",
    "GET:/api/v1/taxonomy/kpi/{org_id}/dashboard": "taxonomy-kpi:read",
    "GET:/api/v1/taxonomy/kpi/{org_id}/objective-breakdown": "taxonomy-kpi:read",
    "GET:/api/v1/taxonomy/kpi/{org_id}/compare": "taxonomy-kpi:read",
    "POST:/api/v1/taxonomy/kpi/validate-denominators": "taxonomy-kpi:execute",

    # GAR / BTAR
    "POST:/api/v1/taxonomy/gar/stock": "taxonomy-gar:execute",
    "POST:/api/v1/taxonomy/gar/flow": "taxonomy-gar:execute",
    "POST:/api/v1/taxonomy/gar/btar": "taxonomy-gar:execute",
    "POST:/api/v1/taxonomy/gar/classify-exposure": "taxonomy-gar:execute",
    "GET:/api/v1/taxonomy/gar/{institution_id}/sector-breakdown": "taxonomy-gar:read",
    "GET:/api/v1/taxonomy/gar/{institution_id}/trends": "taxonomy-gar:read",
    "GET:/api/v1/taxonomy/gar/{institution_id}/compare": "taxonomy-gar:read",
    "POST:/api/v1/taxonomy/gar/eba-template": "taxonomy-gar:execute",
    "GET:/api/v1/taxonomy/gar/{institution_id}/asset-class-summary": "taxonomy-gar:read",
    "POST:/api/v1/taxonomy/gar/mortgage-check": "taxonomy-gar:execute",

    # Alignment Workflow
    "POST:/api/v1/taxonomy/alignment/full": "taxonomy-alignment:execute",
    "POST:/api/v1/taxonomy/alignment/portfolio": "taxonomy-alignment:execute",
    "POST:/api/v1/taxonomy/alignment/batch": "taxonomy-alignment:execute",
    "GET:/api/v1/taxonomy/alignment/{org_id}/status/{activity_code}": "taxonomy-alignment:read",
    "GET:/api/v1/taxonomy/alignment/{org_id}/progress": "taxonomy-alignment:read",
    "GET:/api/v1/taxonomy/alignment/{org_id}/dashboard": "taxonomy-alignment:read",
    "GET:/api/v1/taxonomy/alignment/{org_id}/eligible-vs-aligned": "taxonomy-alignment:read",
    "GET:/api/v1/taxonomy/alignment/{org_id}/compare": "taxonomy-alignment:read",

    # Reporting
    "POST:/api/v1/taxonomy/reports/article-8": "taxonomy-reports:write",
    "POST:/api/v1/taxonomy/reports/eba": "taxonomy-reports:write",
    "POST:/api/v1/taxonomy/reports/export": "taxonomy-reports:export",
    "POST:/api/v1/taxonomy/reports/xbrl": "taxonomy-reports:export",
    "GET:/api/v1/taxonomy/reports/{org_id}/history": "taxonomy-reports:read",
    "GET:/api/v1/taxonomy/reports/{org_id}/compare": "taxonomy-reports:read",
    "POST:/api/v1/taxonomy/reports/{report_id}/qualitative": "taxonomy-reports:write",
    "GET:/api/v1/taxonomy/reports/{org_id}/disclosure-summary": "taxonomy-reports:read",

    # Portfolio Management
    "POST:/api/v1/taxonomy/portfolios": "taxonomy-portfolios:write",
    "GET:/api/v1/taxonomy/portfolios/{portfolio_id}": "taxonomy-portfolios:read",
    "PUT:/api/v1/taxonomy/portfolios/{portfolio_id}": "taxonomy-portfolios:write",
    "DELETE:/api/v1/taxonomy/portfolios/{portfolio_id}": "taxonomy-portfolios:delete",
    "POST:/api/v1/taxonomy/portfolios/{portfolio_id}/holdings": "taxonomy-portfolios:write",
    "GET:/api/v1/taxonomy/portfolios/{portfolio_id}/holdings": "taxonomy-portfolios:read",
    "POST:/api/v1/taxonomy/portfolios/upload": "taxonomy-portfolios:write",
    "GET:/api/v1/taxonomy/portfolios/{org_id}/list": "taxonomy-portfolios:read",

    # Dashboard
    "GET:/api/v1/taxonomy/dashboard/{org_id}/overview": "taxonomy-dashboard:read",
    "GET:/api/v1/taxonomy/dashboard/{org_id}/alignment-summary": "taxonomy-dashboard:read",
    "GET:/api/v1/taxonomy/dashboard/{org_id}/kpi-cards": "taxonomy-dashboard:read",
    "GET:/api/v1/taxonomy/dashboard/{org_id}/sector-breakdown": "taxonomy-dashboard:read",
    "GET:/api/v1/taxonomy/dashboard/{org_id}/trends": "taxonomy-dashboard:read",
    "GET:/api/v1/taxonomy/dashboard/{org_id}/eligible-funnel": "taxonomy-dashboard:read",

    # Data Quality
    "POST:/api/v1/taxonomy/data-quality/assess": "taxonomy-data-quality:execute",
    "GET:/api/v1/taxonomy/data-quality/{org_id}/dashboard": "taxonomy-data-quality:read",
    "GET:/api/v1/taxonomy/data-quality/{org_id}/dimensions": "taxonomy-data-quality:read",
    "POST:/api/v1/taxonomy/data-quality/{org_id}/evidence": "taxonomy-data-quality:write",
    "POST:/api/v1/taxonomy/data-quality/{org_id}/improvement-plan": "taxonomy-data-quality:write",
    "GET:/api/v1/taxonomy/data-quality/{org_id}/trends": "taxonomy-data-quality:read",

    # Regulatory Tracking
    "GET:/api/v1/taxonomy/regulatory/delegated-acts": "taxonomy-regulatory:read",
    "GET:/api/v1/taxonomy/regulatory/updates": "taxonomy-regulatory:read",
    "GET:/api/v1/taxonomy/regulatory/{org_id}/omnibus-impact": "taxonomy-regulatory:read",
    "GET:/api/v1/taxonomy/regulatory/{activity_code}/applicable-version": "taxonomy-regulatory:read",
    "GET:/api/v1/taxonomy/regulatory/transition-plan/{org_id}": "taxonomy-regulatory:read",

    # Gap Analysis
    "POST:/api/v1/taxonomy/gap-analysis/{org_id}": "taxonomy-gap-analysis:execute",
    "GET:/api/v1/taxonomy/gap-analysis/{org_id}/results": "taxonomy-gap-analysis:read",
    "GET:/api/v1/taxonomy/gap-analysis/{org_id}/dnsh-gaps": "taxonomy-gap-analysis:read",
    "GET:/api/v1/taxonomy/gap-analysis/{org_id}/safeguard-gaps": "taxonomy-gap-analysis:read",
    "GET:/api/v1/taxonomy/gap-analysis/{org_id}/data-gaps": "taxonomy-gap-analysis:read",
    "POST:/api/v1/taxonomy/gap-analysis/{org_id}/action-plan": "taxonomy-gap-analysis:write",
    "GET:/api/v1/taxonomy/gap-analysis/{org_id}/priority-matrix": "taxonomy-gap-analysis:read",

    # Settings
    "GET:/api/v1/taxonomy/settings/{org_id}": "taxonomy-settings:read",
    "PUT:/api/v1/taxonomy/settings/{org_id}": "taxonomy-settings:write",
    "GET:/api/v1/taxonomy/settings/{org_id}/reporting-periods": "taxonomy-settings:read",
    "POST:/api/v1/taxonomy/settings/{org_id}/reporting-periods": "taxonomy-settings:write",
    "GET:/api/v1/taxonomy/settings/{org_id}/thresholds": "taxonomy-settings:read",
    "PUT:/api/v1/taxonomy/settings/{org_id}/thresholds": "taxonomy-settings:write",
    "GET:/api/v1/taxonomy/settings/{org_id}/mrv-mapping": "taxonomy-settings:read",
    "PUT:/api/v1/taxonomy/settings/{org_id}/mrv-mapping": "taxonomy-settings:write",
    # ==========================================================================
    # EUDR Supply Chain Mapper routes (/api/v1/eudr-scm) - AGENT-EUDR-001
    # ==========================================================================
    # Graph management (Feature 1)
    "POST:/api/v1/eudr-scm/graphs": "eudr-supply-chain:write",
    "GET:/api/v1/eudr-scm/graphs": "eudr-supply-chain:read",
    "GET:/api/v1/eudr-scm/graphs/{graph_id}": "eudr-supply-chain:read",
    "DELETE:/api/v1/eudr-scm/graphs/{graph_id}": "eudr-supply-chain:delete",
    "GET:/api/v1/eudr-scm/graphs/{graph_id}/export": "eudr-supply-chain:export",
    # Multi-tier discovery (Feature 2)
    "POST:/api/v1/eudr-scm/graphs/{graph_id}/discover": "eudr-supply-chain:map",
    "GET:/api/v1/eudr-scm/graphs/{graph_id}/tiers": "eudr-supply-chain:read",
    # Batch traceability (Feature 4)
    "GET:/api/v1/eudr-scm/graphs/{graph_id}/trace/forward/{node_id}": "eudr-supply-chain:read",
    "GET:/api/v1/eudr-scm/graphs/{graph_id}/trace/backward/{node_id}": "eudr-supply-chain:read",
    "GET:/api/v1/eudr-scm/graphs/{graph_id}/trace/batch/{batch_id}": "eudr-supply-chain:read",
    # Risk propagation (Feature 5)
    "POST:/api/v1/eudr-scm/graphs/{graph_id}/risk/propagate": "eudr-supply-chain:analyze",
    "GET:/api/v1/eudr-scm/graphs/{graph_id}/risk/summary": "eudr-supply-chain:read",
    "GET:/api/v1/eudr-scm/graphs/{graph_id}/risk/heatmap": "eudr-supply-chain:read",
    # Gap analysis (Feature 6)
    "POST:/api/v1/eudr-scm/graphs/{graph_id}/gaps/analyze": "eudr-supply-chain:analyze",
    "GET:/api/v1/eudr-scm/graphs/{graph_id}/gaps": "eudr-supply-chain:read",
    "PUT:/api/v1/eudr-scm/graphs/{graph_id}/gaps/{gap_id}/resolve": "eudr-supply-chain:write",
    # Visualization (Feature 7)
    "GET:/api/v1/eudr-scm/graphs/{graph_id}/layout": "eudr-supply-chain:read",
    "GET:/api/v1/eudr-scm/graphs/{graph_id}/sankey": "eudr-supply-chain:read",
    # Supplier onboarding (Feature 9)
    "POST:/api/v1/eudr-scm/onboarding/invite": "eudr-supply-chain:write",
    "GET:/api/v1/eudr-scm/onboarding/{token}": "eudr-supply-chain:read",
    "POST:/api/v1/eudr-scm/onboarding/{token}/submit": "eudr-supply-chain:write",
    # Health check (no auth required but mapped for completeness)
    "GET:/api/v1/eudr-scm/health": "eudr-supply-chain:read",
    # ---- EUDR Geolocation Verification routes (/api/v1/verify) - AGENT-EUDR-002 ----
    # Coordinate validation
    "POST:/api/v1/verify/coordinates": "eudr-geo:coordinates:verify",
    "POST:/api/v1/verify/coordinates/batch": "eudr-geo:coordinates:verify",
    # Polygon verification
    "POST:/api/v1/verify/polygon": "eudr-geo:polygon:verify",
    "POST:/api/v1/verify/polygon/repair": "eudr-geo:polygon:repair",
    # Protected area screening
    "POST:/api/v1/verify/protected-areas": "eudr-geo:protected-areas:check",
    "GET:/api/v1/verify/protected-areas/nearby": "eudr-geo:protected-areas:check",
    # Deforestation verification
    "POST:/api/v1/verify/deforestation": "eudr-geo:deforestation:verify",
    "GET:/api/v1/verify/deforestation/{plot_id}/evidence": "eudr-geo:deforestation:verify",
    # Full plot verification
    "POST:/api/v1/verify/plot": "eudr-geo:plots:verify",
    "GET:/api/v1/verify/plot/{plot_id}": "eudr-geo:plots:read",
    "GET:/api/v1/verify/plot/{plot_id}/history": "eudr-geo:plots:read",
    # Batch verification
    "POST:/api/v1/verify/batch": "eudr-geo:batch:submit",
    "GET:/api/v1/verify/batch/{batch_id}": "eudr-geo:batch:read",
    "GET:/api/v1/verify/batch/{batch_id}/progress": "eudr-geo:batch:read",
    "DELETE:/api/v1/verify/batch/{batch_id}": "eudr-geo:batch:cancel",
    # Accuracy scoring
    "GET:/api/v1/scores/{plot_id}": "eudr-geo:scores:read",
    "GET:/api/v1/scores/{plot_id}/history": "eudr-geo:scores:read",
    "GET:/api/v1/scores/summary": "eudr-geo:scores:read",
    "PUT:/api/v1/scores/weights": "eudr-geo:scores:configure",
    # Compliance reporting
    "POST:/api/v1/compliance/report": "eudr-geo:compliance:generate",
    "GET:/api/v1/compliance/report/{report_id}": "eudr-geo:compliance:read",
    "GET:/api/v1/compliance/summary": "eudr-geo:compliance:read",
    # Health check
    "GET:/api/v1/eudr-geo/health": "eudr-geo:plots:read",

    # ── AGENT-EUDR-003: Satellite Monitoring ──────────────────────────────
    # Imagery
    "POST:/api/v1/eudr-sat/imagery/search": "eudr-sat:imagery:search",
    "POST:/api/v1/eudr-sat/imagery/download": "eudr-sat:imagery:download",
    "GET:/api/v1/eudr-sat/imagery/{scene_id}": "eudr-sat:imagery:read",
    "GET:/api/v1/eudr-sat/imagery/availability": "eudr-sat:imagery:read",
    # Analysis
    "POST:/api/v1/eudr-sat/analysis/ndvi": "eudr-sat:analysis:create",
    "POST:/api/v1/eudr-sat/analysis/baseline": "eudr-sat:baseline:create",
    "GET:/api/v1/eudr-sat/analysis/baseline/{plot_id}": "eudr-sat:baseline:read",
    "POST:/api/v1/eudr-sat/analysis/change-detect": "eudr-sat:analysis:create",
    "POST:/api/v1/eudr-sat/analysis/fusion": "eudr-sat:analysis:create",
    "GET:/api/v1/eudr-sat/analysis/history/{plot_id}": "eudr-sat:analysis:read",
    # Monitoring
    "POST:/api/v1/eudr-sat/monitoring/schedule": "eudr-sat:monitoring:create",
    "GET:/api/v1/eudr-sat/monitoring/schedule/{schedule_id}": "eudr-sat:monitoring:read",
    "PUT:/api/v1/eudr-sat/monitoring/schedule/{schedule_id}": "eudr-sat:monitoring:update",
    "DELETE:/api/v1/eudr-sat/monitoring/schedule/{schedule_id}": "eudr-sat:monitoring:delete",
    "GET:/api/v1/eudr-sat/monitoring/results/{plot_id}": "eudr-sat:monitoring:read",
    "POST:/api/v1/eudr-sat/monitoring/execute": "eudr-sat:monitoring:create",
    # Alerts
    "GET:/api/v1/eudr-sat/alerts": "eudr-sat:alerts:read",
    "GET:/api/v1/eudr-sat/alerts/{alert_id}": "eudr-sat:alerts:read",
    "PUT:/api/v1/eudr-sat/alerts/{alert_id}/acknowledge": "eudr-sat:alerts:acknowledge",
    "GET:/api/v1/eudr-sat/alerts/summary": "eudr-sat:alerts:read",
    # Evidence
    "POST:/api/v1/eudr-sat/evidence/package": "eudr-sat:evidence:create",
    "GET:/api/v1/eudr-sat/evidence/{package_id}": "eudr-sat:evidence:read",
    "GET:/api/v1/eudr-sat/evidence/{package_id}/download": "eudr-sat:evidence:download",
    # Batch
    "POST:/api/v1/eudr-sat/batch": "eudr-sat:analysis:create",
    "GET:/api/v1/eudr-sat/batch/{batch_id}": "eudr-sat:analysis:read",
    "GET:/api/v1/eudr-sat/batch/{batch_id}/progress": "eudr-sat:analysis:read",
    "DELETE:/api/v1/eudr-sat/batch/{batch_id}": "eudr-sat:analysis:create",
    # Health
    "GET:/api/v1/eudr-sat/health": "eudr-sat:imagery:read",
    # -----------------------------------------------------------------------
    # EUDR Forest Cover Analysis routes (AGENT-EUDR-004)
    # -----------------------------------------------------------------------
    # Canopy density
    "POST:/api/v1/eudr-fca/density/analyze": "eudr-fca:density:analyze",
    "POST:/api/v1/eudr-fca/density/batch": "eudr-fca:density:batch",
    "GET:/api/v1/eudr-fca/density/{plot_id}": "eudr-fca:read",
    "GET:/api/v1/eudr-fca/density/{plot_id}/history": "eudr-fca:read",
    "POST:/api/v1/eudr-fca/density/compare": "eudr-fca:density:analyze",
    # Classification
    "POST:/api/v1/eudr-fca/classify": "eudr-fca:classify:run",
    "POST:/api/v1/eudr-fca/classify/batch": "eudr-fca:classify:batch",
    "GET:/api/v1/eudr-fca/classify/{plot_id}": "eudr-fca:read",
    "GET:/api/v1/eudr-fca/classify/types": "eudr-fca:read",
    # Historical reconstruction
    "POST:/api/v1/eudr-fca/historical/reconstruct": "eudr-fca:historical:reconstruct",
    "POST:/api/v1/eudr-fca/historical/batch": "eudr-fca:historical:reconstruct",
    "GET:/api/v1/eudr-fca/historical/{plot_id}": "eudr-fca:read",
    "POST:/api/v1/eudr-fca/historical/compare": "eudr-fca:historical:compare",
    "GET:/api/v1/eudr-fca/historical/{plot_id}/sources": "eudr-fca:read",
    # Verification
    "POST:/api/v1/eudr-fca/verify": "eudr-fca:verify:single",
    "POST:/api/v1/eudr-fca/verify/batch": "eudr-fca:verify:batch",
    "GET:/api/v1/eudr-fca/verify/{plot_id}": "eudr-fca:read",
    "GET:/api/v1/eudr-fca/verify/{plot_id}/evidence": "eudr-fca:read",
    "POST:/api/v1/eudr-fca/verify/complete": "eudr-fca:verify:single",
    # Analysis (height, fragmentation, biomass)
    "POST:/api/v1/eudr-fca/analysis/height": "eudr-fca:height:estimate",
    "POST:/api/v1/eudr-fca/analysis/fragmentation": "eudr-fca:fragmentation:analyze",
    "POST:/api/v1/eudr-fca/analysis/biomass": "eudr-fca:biomass:estimate",
    "GET:/api/v1/eudr-fca/analysis/{plot_id}/profile": "eudr-fca:read",
    "POST:/api/v1/eudr-fca/analysis/compare": "eudr-fca:read",
    # Reports
    "POST:/api/v1/eudr-fca/reports/generate": "eudr-fca:reports:generate",
    "GET:/api/v1/eudr-fca/reports/{report_id}": "eudr-fca:reports:download",
    "GET:/api/v1/eudr-fca/reports/{report_id}/download": "eudr-fca:reports:download",
    "POST:/api/v1/eudr-fca/reports/batch": "eudr-fca:reports:generate",
    # Batch
    "POST:/api/v1/eudr-fca/batch": "eudr-fca:batch:submit",
    "DELETE:/api/v1/eudr-fca/batch/{batch_id}": "eudr-fca:batch:cancel",
    # Health
    "GET:/api/v1/eudr-fca/health": "eudr-fca:read",

    # EUDR Land Use Change Detector routes (AGENT-EUDR-005)
    # Classification (5 routes)
    "POST:/api/v1/eudr-luc/classify": "eudr-luc:classify:run",
    "POST:/api/v1/eudr-luc/classify/batch": "eudr-luc:classify:batch",
    "GET:/api/v1/eudr-luc/classify/{plot_id}": "eudr-luc:read",
    "GET:/api/v1/eudr-luc/classify/{plot_id}/history": "eudr-luc:read",
    "POST:/api/v1/eudr-luc/classify/compare": "eudr-luc:classify:run",
    # Transitions (5 routes)
    "POST:/api/v1/eudr-luc/transitions/detect": "eudr-luc:transitions:detect",
    "POST:/api/v1/eudr-luc/transitions/batch": "eudr-luc:transitions:batch",
    "GET:/api/v1/eudr-luc/transitions/{plot_id}": "eudr-luc:read",
    "POST:/api/v1/eudr-luc/transitions/matrix": "eudr-luc:transitions:detect",
    "GET:/api/v1/eudr-luc/transitions/types": "eudr-luc:read",
    # Trajectories (3 routes)
    "POST:/api/v1/eudr-luc/trajectory/analyze": "eudr-luc:trajectory:analyze",
    "POST:/api/v1/eudr-luc/trajectory/batch": "eudr-luc:trajectory:batch",
    "GET:/api/v1/eudr-luc/trajectory/{plot_id}": "eudr-luc:read",
    # Verification (5 routes)
    "POST:/api/v1/eudr-luc/verify/cutoff": "eudr-luc:verify:cutoff",
    "POST:/api/v1/eudr-luc/verify/batch": "eudr-luc:verify:batch",
    "GET:/api/v1/eudr-luc/verify/{plot_id}": "eudr-luc:read",
    "GET:/api/v1/eudr-luc/verify/{plot_id}/evidence": "eudr-luc:read",
    "POST:/api/v1/eudr-luc/verify/complete": "eudr-luc:verify:cutoff",
    # Risk & Urban (6 routes)
    "POST:/api/v1/eudr-luc/risk/assess": "eudr-luc:risk:assess",
    "POST:/api/v1/eudr-luc/risk/batch": "eudr-luc:risk:batch",
    "GET:/api/v1/eudr-luc/risk/{plot_id}": "eudr-luc:read",
    "POST:/api/v1/eudr-luc/urban/analyze": "eudr-luc:urban:analyze",
    "POST:/api/v1/eudr-luc/urban/batch": "eudr-luc:urban:batch",
    "GET:/api/v1/eudr-luc/urban/{plot_id}": "eudr-luc:read",
    # Reports (4 routes)
    "POST:/api/v1/eudr-luc/reports/generate": "eudr-luc:reports:generate",
    "GET:/api/v1/eudr-luc/reports/{report_id}": "eudr-luc:reports:download",
    "GET:/api/v1/eudr-luc/reports/{report_id}/download": "eudr-luc:reports:download",
    "POST:/api/v1/eudr-luc/reports/batch": "eudr-luc:reports:generate",
    # Batch
    "POST:/api/v1/eudr-luc/batch": "eudr-luc:batch:submit",
    "DELETE:/api/v1/eudr-luc/batch/{batch_id}": "eudr-luc:batch:cancel",
    # Health
    "GET:/api/v1/eudr-luc/health": "eudr-luc:read",

    # EUDR Plot Boundary Manager routes (AGENT-EUDR-006)
    # Boundary CRUD (6 routes)
    "POST:/api/v1/eudr-pbm/boundaries": "eudr-pbm:write",
    "GET:/api/v1/eudr-pbm/boundaries/{plot_id}": "eudr-pbm:read",
    "PUT:/api/v1/eudr-pbm/boundaries/{plot_id}": "eudr-pbm:write",
    "DELETE:/api/v1/eudr-pbm/boundaries/{plot_id}": "eudr-pbm:delete",
    "POST:/api/v1/eudr-pbm/boundaries/batch": "eudr-pbm:batch:submit",
    "POST:/api/v1/eudr-pbm/boundaries/search": "eudr-pbm:read",
    # Validation (4 routes)
    "POST:/api/v1/eudr-pbm/validate": "eudr-pbm:validate:run",
    "POST:/api/v1/eudr-pbm/validate/batch": "eudr-pbm:validate:batch",
    "POST:/api/v1/eudr-pbm/repair": "eudr-pbm:repair:run",
    "POST:/api/v1/eudr-pbm/repair/batch": "eudr-pbm:repair:batch",
    # Area (3 routes)
    "POST:/api/v1/eudr-pbm/area/calculate": "eudr-pbm:area:calculate",
    "POST:/api/v1/eudr-pbm/area/batch": "eudr-pbm:area:batch",
    "POST:/api/v1/eudr-pbm/area/threshold": "eudr-pbm:area:calculate",
    # Overlaps (4 routes)
    "POST:/api/v1/eudr-pbm/overlaps/detect": "eudr-pbm:overlaps:detect",
    "POST:/api/v1/eudr-pbm/overlaps/scan": "eudr-pbm:overlaps:scan",
    "GET:/api/v1/eudr-pbm/overlaps/{plot_id}": "eudr-pbm:read",
    "POST:/api/v1/eudr-pbm/overlaps/resolve": "eudr-pbm:overlaps:resolve",
    # Versions (4 routes)
    "GET:/api/v1/eudr-pbm/versions/{plot_id}": "eudr-pbm:read",
    "GET:/api/v1/eudr-pbm/versions/{plot_id}/at": "eudr-pbm:read",
    "GET:/api/v1/eudr-pbm/versions/{plot_id}/diff": "eudr-pbm:read",
    "GET:/api/v1/eudr-pbm/versions/{plot_id}/lineage": "eudr-pbm:read",
    # Export (6 routes)
    "POST:/api/v1/eudr-pbm/export/geojson": "eudr-pbm:export:run",
    "POST:/api/v1/eudr-pbm/export/kml": "eudr-pbm:export:run",
    "POST:/api/v1/eudr-pbm/export/shapefile": "eudr-pbm:export:run",
    "POST:/api/v1/eudr-pbm/export/eudr-xml": "eudr-pbm:export:run",
    "POST:/api/v1/eudr-pbm/export/batch": "eudr-pbm:export:batch",
    "GET:/api/v1/eudr-pbm/export/{export_id}": "eudr-pbm:export:download",
    # Split/Merge (3 routes)
    "POST:/api/v1/eudr-pbm/split": "eudr-pbm:split:run",
    "POST:/api/v1/eudr-pbm/merge": "eudr-pbm:merge:run",
    "GET:/api/v1/eudr-pbm/genealogy/{plot_id}": "eudr-pbm:read",
    # Batch
    "POST:/api/v1/eudr-pbm/batch": "eudr-pbm:batch:submit",
    "DELETE:/api/v1/eudr-pbm/batch/{batch_id}": "eudr-pbm:batch:cancel",
    # Health
    "GET:/api/v1/eudr-pbm/health": "eudr-pbm:read",

    # EUDR GPS Coordinate Validator routes (AGENT-EUDR-007)
    # Parsing (4 routes)
    "POST:/api/v1/eudr-gcv/parse": "eudr-gcv:parse:single",
    "POST:/api/v1/eudr-gcv/parse/batch": "eudr-gcv:parse:batch",
    "POST:/api/v1/eudr-gcv/parse/detect-format": "eudr-gcv:parse:detect",
    "POST:/api/v1/eudr-gcv/parse/normalize": "eudr-gcv:parse:normalize",
    # Validation (5 routes)
    "POST:/api/v1/eudr-gcv/validate": "eudr-gcv:validate:single",
    "POST:/api/v1/eudr-gcv/validate/batch": "eudr-gcv:validate:batch",
    "POST:/api/v1/eudr-gcv/validate/range": "eudr-gcv:validate:single",
    "POST:/api/v1/eudr-gcv/validate/swap-detect": "eudr-gcv:validate:swap",
    "POST:/api/v1/eudr-gcv/validate/duplicates": "eudr-gcv:validate:duplicates",
    # Plausibility (5 routes)
    "POST:/api/v1/eudr-gcv/plausibility/check": "eudr-gcv:plausibility:check",
    "POST:/api/v1/eudr-gcv/plausibility/land-ocean": "eudr-gcv:plausibility:land-ocean",
    "POST:/api/v1/eudr-gcv/plausibility/country": "eudr-gcv:plausibility:country",
    "POST:/api/v1/eudr-gcv/plausibility/commodity": "eudr-gcv:plausibility:commodity",
    "POST:/api/v1/eudr-gcv/plausibility/elevation": "eudr-gcv:plausibility:elevation",
    # Assessment (4 routes)
    "POST:/api/v1/eudr-gcv/assess": "eudr-gcv:assess:single",
    "POST:/api/v1/eudr-gcv/assess/batch": "eudr-gcv:assess:batch",
    "GET:/api/v1/eudr-gcv/assess/{coord_id}": "eudr-gcv:read",
    "POST:/api/v1/eudr-gcv/assess/precision": "eudr-gcv:assess:precision",
    # Reporting (5 routes)
    "POST:/api/v1/eudr-gcv/reports/compliance": "eudr-gcv:reports:generate",
    "POST:/api/v1/eudr-gcv/reports/batch-summary": "eudr-gcv:reports:generate",
    "POST:/api/v1/eudr-gcv/reports/remediation": "eudr-gcv:reports:generate",
    "GET:/api/v1/eudr-gcv/reports/{report_id}": "eudr-gcv:reports:read",
    "GET:/api/v1/eudr-gcv/reports/{report_id}/download": "eudr-gcv:reports:download",
    # Reverse Geocoding (3 routes)
    "POST:/api/v1/eudr-gcv/geocode/reverse": "eudr-gcv:geocode:reverse",
    "POST:/api/v1/eudr-gcv/geocode/batch": "eudr-gcv:geocode:batch",
    "POST:/api/v1/eudr-gcv/geocode/country": "eudr-gcv:geocode:country",
    # Datum (3 routes)
    "POST:/api/v1/eudr-gcv/datum/transform": "eudr-gcv:datum:transform",
    "POST:/api/v1/eudr-gcv/datum/batch": "eudr-gcv:datum:batch",
    "GET:/api/v1/eudr-gcv/datum/list": "eudr-gcv:read",
    # Batch (2 routes)
    "POST:/api/v1/eudr-gcv/batch": "eudr-gcv:batch:submit",
    "DELETE:/api/v1/eudr-gcv/batch/{batch_id}": "eudr-gcv:batch:cancel",
    # Health
    "GET:/api/v1/eudr-gcv/health": "eudr-gcv:read",

    # EUDR Multi-Tier Supplier Tracker routes (AGENT-EUDR-008)
    # Discovery (4 routes)
    "POST:/api/v1/eudr-mst/discover": "eudr-mst:discover:single",
    "POST:/api/v1/eudr-mst/discover/batch": "eudr-mst:discover:batch",
    "POST:/api/v1/eudr-mst/discover/from-declaration": "eudr-mst:discover:declaration",
    "POST:/api/v1/eudr-mst/discover/from-questionnaire": "eudr-mst:discover:questionnaire",
    # Profiles (6 routes)
    "POST:/api/v1/eudr-mst/suppliers": "eudr-mst:suppliers:create",
    "GET:/api/v1/eudr-mst/suppliers/{supplier_id}": "eudr-mst:read",
    "PUT:/api/v1/eudr-mst/suppliers/{supplier_id}": "eudr-mst:suppliers:update",
    "DELETE:/api/v1/eudr-mst/suppliers/{supplier_id}": "eudr-mst:suppliers:delete",
    "POST:/api/v1/eudr-mst/suppliers/search": "eudr-mst:read",
    "POST:/api/v1/eudr-mst/suppliers/batch": "eudr-mst:suppliers:batch",
    # Tiers (4 routes)
    "GET:/api/v1/eudr-mst/tiers/{supplier_id}": "eudr-mst:tiers:read",
    "POST:/api/v1/eudr-mst/tiers/assess": "eudr-mst:tiers:assess",
    "GET:/api/v1/eudr-mst/tiers/visibility": "eudr-mst:tiers:read",
    "GET:/api/v1/eudr-mst/tiers/gaps": "eudr-mst:tiers:gaps",
    # Relationships (4 routes)
    "POST:/api/v1/eudr-mst/relationships": "eudr-mst:relationships:create",
    "PUT:/api/v1/eudr-mst/relationships/{rel_id}": "eudr-mst:relationships:update",
    "GET:/api/v1/eudr-mst/relationships/{supplier_id}": "eudr-mst:relationships:read",
    "POST:/api/v1/eudr-mst/relationships/history": "eudr-mst:relationships:read",
    # Risk (4 routes)
    "POST:/api/v1/eudr-mst/risk/assess": "eudr-mst:risk:assess",
    "POST:/api/v1/eudr-mst/risk/propagate": "eudr-mst:risk:propagate",
    "GET:/api/v1/eudr-mst/risk/{supplier_id}": "eudr-mst:risk:read",
    "POST:/api/v1/eudr-mst/risk/batch": "eudr-mst:risk:batch",
    # Compliance (4 routes)
    "POST:/api/v1/eudr-mst/compliance/check": "eudr-mst:compliance:check",
    "GET:/api/v1/eudr-mst/compliance/{supplier_id}": "eudr-mst:compliance:read",
    "POST:/api/v1/eudr-mst/compliance/batch": "eudr-mst:compliance:batch",
    "GET:/api/v1/eudr-mst/compliance/alerts": "eudr-mst:compliance:alerts",
    # Reports (5 routes)
    "POST:/api/v1/eudr-mst/reports/audit": "eudr-mst:reports:generate",
    "POST:/api/v1/eudr-mst/reports/tier-summary": "eudr-mst:reports:generate",
    "POST:/api/v1/eudr-mst/reports/gaps": "eudr-mst:reports:generate",
    "GET:/api/v1/eudr-mst/reports/{report_id}": "eudr-mst:reports:read",
    "GET:/api/v1/eudr-mst/reports/{report_id}/download": "eudr-mst:reports:download",
    # Batch (2 routes)
    "POST:/api/v1/eudr-mst/batch": "eudr-mst:batch:submit",
    "DELETE:/api/v1/eudr-mst/batch/{batch_id}": "eudr-mst:batch:cancel",
    # Health
    "GET:/api/v1/eudr-mst/health": "eudr-mst:read",

    # EUDR Chain of Custody routes (AGENT-EUDR-009)
    # Events (5 routes)
    "POST:/api/v1/eudr-coc/events": "eudr-coc:events:record",
    "POST:/api/v1/eudr-coc/events/batch": "eudr-coc:events:bulk",
    "GET:/api/v1/eudr-coc/events/{event_id}": "eudr-coc:read",
    "GET:/api/v1/eudr-coc/events/chain/{batch_id}": "eudr-coc:events:chain",
    "POST:/api/v1/eudr-coc/events/amend/{event_id}": "eudr-coc:events:amend",
    # Batches (7 routes)
    "POST:/api/v1/eudr-coc/batches": "eudr-coc:batches:create",
    "GET:/api/v1/eudr-coc/batches/{batch_id}": "eudr-coc:read",
    "POST:/api/v1/eudr-coc/batches/split": "eudr-coc:batches:split",
    "POST:/api/v1/eudr-coc/batches/merge": "eudr-coc:batches:merge",
    "POST:/api/v1/eudr-coc/batches/blend": "eudr-coc:batches:blend",
    "GET:/api/v1/eudr-coc/batches/{batch_id}/genealogy": "eudr-coc:batches:genealogy",
    "POST:/api/v1/eudr-coc/batches/search": "eudr-coc:read",
    # Models (4 routes)
    "POST:/api/v1/eudr-coc/models/assign": "eudr-coc:models:assign",
    "GET:/api/v1/eudr-coc/models/{facility_id}": "eudr-coc:models:read",
    "POST:/api/v1/eudr-coc/models/validate": "eudr-coc:models:validate",
    "GET:/api/v1/eudr-coc/models/compliance/{facility_id}": "eudr-coc:models:compliance",
    # Balance (5 routes)
    "POST:/api/v1/eudr-coc/balance/input": "eudr-coc:balance:input",
    "POST:/api/v1/eudr-coc/balance/output": "eudr-coc:balance:output",
    "GET:/api/v1/eudr-coc/balance/{facility_id}": "eudr-coc:balance:read",
    "POST:/api/v1/eudr-coc/balance/reconcile": "eudr-coc:balance:reconcile",
    "GET:/api/v1/eudr-coc/balance/history/{facility_id}": "eudr-coc:balance:read",
    # Transformations (3 routes)
    "POST:/api/v1/eudr-coc/transform": "eudr-coc:transform:record",
    "POST:/api/v1/eudr-coc/transform/batch": "eudr-coc:transform:bulk",
    "GET:/api/v1/eudr-coc/transform/{transform_id}": "eudr-coc:read",
    # Documents (3 routes)
    "POST:/api/v1/eudr-coc/documents": "eudr-coc:documents:link",
    "GET:/api/v1/eudr-coc/documents/{batch_id}": "eudr-coc:documents:read",
    "POST:/api/v1/eudr-coc/documents/validate": "eudr-coc:documents:validate",
    # Verification (3 routes)
    "POST:/api/v1/eudr-coc/verify/chain": "eudr-coc:verify:chain",
    "POST:/api/v1/eudr-coc/verify/batch": "eudr-coc:verify:batch",
    "GET:/api/v1/eudr-coc/verify/{verification_id}": "eudr-coc:read",
    # Reports (4 routes)
    "POST:/api/v1/eudr-coc/reports/traceability": "eudr-coc:reports:generate",
    "POST:/api/v1/eudr-coc/reports/mass-balance": "eudr-coc:reports:generate",
    "GET:/api/v1/eudr-coc/reports/{report_id}": "eudr-coc:reports:read",
    "GET:/api/v1/eudr-coc/reports/{report_id}/download": "eudr-coc:reports:download",
    # Batch jobs (2 routes)
    "POST:/api/v1/eudr-coc/batch": "eudr-coc:batch:submit",
    "DELETE:/api/v1/eudr-coc/batch/{batch_id}": "eudr-coc:batch:cancel",
    # Health
    "GET:/api/v1/eudr-coc/health": "eudr-coc:read",

    # -----------------------------------------------------------------------
    # EUDR Segregation Verifier (AGENT-EUDR-010)  --  37 routes
    # -----------------------------------------------------------------------
    # SCPs (6 routes)
    "POST:/api/v1/eudr-sgv/scp": "eudr-sgv:scp:register",
    "GET:/api/v1/eudr-sgv/scp/{scp_id}": "eudr-sgv:read",
    "PUT:/api/v1/eudr-sgv/scp/{scp_id}": "eudr-sgv:scp:update",
    "POST:/api/v1/eudr-sgv/scp/validate": "eudr-sgv:scp:validate",
    "POST:/api/v1/eudr-sgv/scp/batch-import": "eudr-sgv:scp:import",
    "POST:/api/v1/eudr-sgv/scp/search": "eudr-sgv:read",
    # Storage (5 routes)
    "POST:/api/v1/eudr-sgv/storage/zones": "eudr-sgv:storage:register",
    "GET:/api/v1/eudr-sgv/storage/zones/{facility_id}": "eudr-sgv:storage:read",
    "POST:/api/v1/eudr-sgv/storage/events": "eudr-sgv:storage:record",
    "POST:/api/v1/eudr-sgv/storage/audit": "eudr-sgv:storage:audit",
    "GET:/api/v1/eudr-sgv/storage/score/{facility_id}": "eudr-sgv:storage:read",
    # Transport (5 routes)
    "POST:/api/v1/eudr-sgv/transport/vehicles": "eudr-sgv:transport:register",
    "GET:/api/v1/eudr-sgv/transport/vehicles/{vehicle_id}": "eudr-sgv:transport:read",
    "POST:/api/v1/eudr-sgv/transport/verify": "eudr-sgv:transport:verify",
    "POST:/api/v1/eudr-sgv/transport/cleaning": "eudr-sgv:transport:cleaning",
    "GET:/api/v1/eudr-sgv/transport/history/{vehicle_id}": "eudr-sgv:transport:read",
    # Processing (5 routes)
    "POST:/api/v1/eudr-sgv/processing/lines": "eudr-sgv:processing:register",
    "GET:/api/v1/eudr-sgv/processing/lines/{line_id}": "eudr-sgv:processing:read",
    "POST:/api/v1/eudr-sgv/processing/changeover": "eudr-sgv:processing:changeover",
    "POST:/api/v1/eudr-sgv/processing/verify": "eudr-sgv:processing:verify",
    "GET:/api/v1/eudr-sgv/processing/score/{facility_id}": "eudr-sgv:processing:read",
    # Contamination (5 routes)
    "POST:/api/v1/eudr-sgv/contamination/detect": "eudr-sgv:contamination:detect",
    "POST:/api/v1/eudr-sgv/contamination/events": "eudr-sgv:contamination:record",
    "GET:/api/v1/eudr-sgv/contamination/events/{event_id}": "eudr-sgv:contamination:read",
    "POST:/api/v1/eudr-sgv/contamination/impact": "eudr-sgv:contamination:impact",
    "GET:/api/v1/eudr-sgv/contamination/heatmap/{facility_id}": "eudr-sgv:contamination:read",
    # Labels (3 routes)
    "POST:/api/v1/eudr-sgv/labels": "eudr-sgv:labels:register",
    "GET:/api/v1/eudr-sgv/labels/{scp_id}": "eudr-sgv:labels:read",
    "POST:/api/v1/eudr-sgv/labels/audit": "eudr-sgv:labels:audit",
    # Assessment (3 routes)
    "POST:/api/v1/eudr-sgv/assessment": "eudr-sgv:assessment:run",
    "GET:/api/v1/eudr-sgv/assessment/{facility_id}": "eudr-sgv:assessment:read",
    "GET:/api/v1/eudr-sgv/assessment/history/{facility_id}": "eudr-sgv:assessment:read",
    # Reports (5 routes)
    "POST:/api/v1/eudr-sgv/reports/audit": "eudr-sgv:reports:generate",
    "POST:/api/v1/eudr-sgv/reports/contamination": "eudr-sgv:reports:generate",
    "POST:/api/v1/eudr-sgv/reports/evidence": "eudr-sgv:reports:generate",
    "GET:/api/v1/eudr-sgv/reports/{report_id}": "eudr-sgv:reports:read",
    "GET:/api/v1/eudr-sgv/reports/{report_id}/download": "eudr-sgv:reports:download",
    # Batch jobs (2 routes)
    "POST:/api/v1/eudr-sgv/batch": "eudr-sgv:batch:submit",
    "DELETE:/api/v1/eudr-sgv/batch/{job_id}": "eudr-sgv:batch:cancel",
    # Health
    "GET:/api/v1/eudr-sgv/health": "eudr-sgv:read",

    # ── AGENT-EUDR-011: Mass Balance Calculator (37 routes) ──────────────
    # Ledger management (7 routes)
    "POST:/api/v1/eudr-mbc/ledgers": "eudr-mbc:ledger:create",
    "GET:/api/v1/eudr-mbc/ledgers/{ledger_id}": "eudr-mbc:read",
    "POST:/api/v1/eudr-mbc/ledgers/entries": "eudr-mbc:ledger:record",
    "POST:/api/v1/eudr-mbc/ledgers/entries/bulk": "eudr-mbc:ledger:bulk",
    "GET:/api/v1/eudr-mbc/ledgers/{ledger_id}/balance": "eudr-mbc:read",
    "GET:/api/v1/eudr-mbc/ledgers/{ledger_id}/history": "eudr-mbc:read",
    "POST:/api/v1/eudr-mbc/ledgers/search": "eudr-mbc:read",
    # Credit period management (5 routes)
    "POST:/api/v1/eudr-mbc/periods": "eudr-mbc:period:create",
    "GET:/api/v1/eudr-mbc/periods/{period_id}": "eudr-mbc:read",
    "PUT:/api/v1/eudr-mbc/periods/{period_id}": "eudr-mbc:period:update",
    "POST:/api/v1/eudr-mbc/periods/rollover": "eudr-mbc:period:rollover",
    "GET:/api/v1/eudr-mbc/periods/active/{facility_id}": "eudr-mbc:read",
    # Conversion factor validation (4 routes)
    "POST:/api/v1/eudr-mbc/factors/validate": "eudr-mbc:factor:validate",
    "GET:/api/v1/eudr-mbc/factors/reference/{commodity}": "eudr-mbc:read",
    "POST:/api/v1/eudr-mbc/factors/custom": "eudr-mbc:factor:custom",
    "GET:/api/v1/eudr-mbc/factors/history/{facility_id}": "eudr-mbc:read",
    # Overdraft detection (5 routes)
    "POST:/api/v1/eudr-mbc/overdraft/check": "eudr-mbc:overdraft:check",
    "GET:/api/v1/eudr-mbc/overdraft/alerts/{facility_id}": "eudr-mbc:overdraft:read",
    "POST:/api/v1/eudr-mbc/overdraft/forecast": "eudr-mbc:overdraft:forecast",
    "POST:/api/v1/eudr-mbc/overdraft/exemption": "eudr-mbc:overdraft:exemption",
    "GET:/api/v1/eudr-mbc/overdraft/history/{facility_id}": "eudr-mbc:overdraft:read",
    # Loss and waste tracking (4 routes)
    "POST:/api/v1/eudr-mbc/losses": "eudr-mbc:loss:record",
    "GET:/api/v1/eudr-mbc/losses/{facility_id}": "eudr-mbc:loss:read",
    "POST:/api/v1/eudr-mbc/losses/validate": "eudr-mbc:loss:validate",
    "GET:/api/v1/eudr-mbc/losses/trends/{facility_id}": "eudr-mbc:loss:read",
    # Reconciliation (4 routes)
    "POST:/api/v1/eudr-mbc/reconciliation": "eudr-mbc:reconciliation:run",
    "GET:/api/v1/eudr-mbc/reconciliation/{reconciliation_id}": "eudr-mbc:reconciliation:read",
    "POST:/api/v1/eudr-mbc/reconciliation/sign-off": "eudr-mbc:reconciliation:sign_off",
    "GET:/api/v1/eudr-mbc/reconciliation/history/{facility_id}": "eudr-mbc:reconciliation:read",
    # Consolidation (5 routes)
    "POST:/api/v1/eudr-mbc/consolidation/report": "eudr-mbc:consolidation:generate",
    "POST:/api/v1/eudr-mbc/consolidation/groups": "eudr-mbc:consolidation:groups",
    "GET:/api/v1/eudr-mbc/consolidation/dashboard": "eudr-mbc:consolidation:read",
    "GET:/api/v1/eudr-mbc/consolidation/report/{report_id}": "eudr-mbc:consolidation:read",
    "GET:/api/v1/eudr-mbc/consolidation/report/{report_id}/download": "eudr-mbc:consolidation:download",
    # Batch jobs (2 routes)
    "POST:/api/v1/eudr-mbc/batch": "eudr-mbc:batch:submit",
    "DELETE:/api/v1/eudr-mbc/batch/{job_id}": "eudr-mbc:batch:cancel",
    # Health
    "GET:/api/v1/eudr-mbc/health": "eudr-mbc:read",
    # ── Document Authentication (AGENT-EUDR-012) ─────────────────
    # Classify
    "POST:/api/v1/eudr-dav/classify": "eudr-dav:classify",
    "POST:/api/v1/eudr-dav/classify/batch": "eudr-dav:classify:batch",
    "GET:/api/v1/eudr-dav/classify/{document_id}": "eudr-dav:classify:read",
    "GET:/api/v1/eudr-dav/classify/templates": "eudr-dav:templates:read",
    "POST:/api/v1/eudr-dav/classify/templates": "eudr-dav:templates:register",
    # Signature
    "POST:/api/v1/eudr-dav/signatures/verify": "eudr-dav:signatures:verify",
    "POST:/api/v1/eudr-dav/signatures/verify/batch": "eudr-dav:signatures:verify:batch",
    "GET:/api/v1/eudr-dav/signatures/{verification_id}": "eudr-dav:signatures:read",
    "GET:/api/v1/eudr-dav/signatures/history/{document_id}": "eudr-dav:signatures:read",
    # Hash
    "POST:/api/v1/eudr-dav/hashes/compute": "eudr-dav:hashes:compute",
    "POST:/api/v1/eudr-dav/hashes/verify": "eudr-dav:hashes:verify",
    "GET:/api/v1/eudr-dav/hashes/registry/{hash}": "eudr-dav:hashes:read",
    "GET:/api/v1/eudr-dav/hashes/merkle/{dds_id}": "eudr-dav:hashes:read",
    # Certificate
    "POST:/api/v1/eudr-dav/certificates/validate": "eudr-dav:certificates:validate",
    "GET:/api/v1/eudr-dav/certificates/{validation_id}": "eudr-dav:certificates:read",
    "POST:/api/v1/eudr-dav/certificates/trusted-cas": "eudr-dav:certificates:trusted_cas:write",
    "GET:/api/v1/eudr-dav/certificates/trusted-cas": "eudr-dav:certificates:trusted_cas:read",
    # Metadata
    "POST:/api/v1/eudr-dav/metadata/extract": "eudr-dav:metadata:extract",
    "GET:/api/v1/eudr-dav/metadata/{document_id}": "eudr-dav:metadata:read",
    "POST:/api/v1/eudr-dav/metadata/validate": "eudr-dav:metadata:validate",
    # Fraud
    "POST:/api/v1/eudr-dav/fraud/detect": "eudr-dav:fraud:detect",
    "POST:/api/v1/eudr-dav/fraud/detect/batch": "eudr-dav:fraud:detect:batch",
    "GET:/api/v1/eudr-dav/fraud/alerts/{document_id}": "eudr-dav:fraud:read",
    "GET:/api/v1/eudr-dav/fraud/alerts/summary/{operator_id}": "eudr-dav:fraud:read",
    "GET:/api/v1/eudr-dav/fraud/rules": "eudr-dav:fraud:rules:read",
    # CrossRef
    "POST:/api/v1/eudr-dav/crossref/verify": "eudr-dav:crossref:verify",
    "POST:/api/v1/eudr-dav/crossref/verify/batch": "eudr-dav:crossref:verify:batch",
    "GET:/api/v1/eudr-dav/crossref/{verification_id}": "eudr-dav:crossref:read",
    "GET:/api/v1/eudr-dav/crossref/cache/stats": "eudr-dav:crossref:read",
    # Report
    "POST:/api/v1/eudr-dav/reports/authentication": "eudr-dav:reports:generate",
    "POST:/api/v1/eudr-dav/reports/evidence-package": "eudr-dav:reports:generate",
    "GET:/api/v1/eudr-dav/reports/{report_id}": "eudr-dav:reports:read",
    "GET:/api/v1/eudr-dav/reports/{report_id}/download": "eudr-dav:reports:download",
    "GET:/api/v1/eudr-dav/reports/dashboard/{operator_id}": "eudr-dav:reports:read",
    # Batch
    "POST:/api/v1/eudr-dav/batch": "eudr-dav:batch:submit",
    "DELETE:/api/v1/eudr-dav/batch/{job_id}": "eudr-dav:batch:cancel",
    # Health
    "GET:/api/v1/eudr-dav/health": "eudr-dav:read",
    # ---- AGENT-EUDR-013: Blockchain Integration (eudr-bci) ----
    # Anchor
    "POST:/api/v1/eudr-bci/anchors": "eudr-bci:anchors:create",
    "POST:/api/v1/eudr-bci/anchors/batch": "eudr-bci:anchors:batch",
    "GET:/api/v1/eudr-bci/anchors/{anchor_id}": "eudr-bci:anchors:read",
    "GET:/api/v1/eudr-bci/anchors/status/{tx_hash}": "eudr-bci:anchors:read",
    "GET:/api/v1/eudr-bci/anchors/history/{record_id}": "eudr-bci:anchors:read",
    # Contract
    "POST:/api/v1/eudr-bci/contracts/deploy": "eudr-bci:contracts:deploy",
    "GET:/api/v1/eudr-bci/contracts/{contract_id}": "eudr-bci:contracts:read",
    "POST:/api/v1/eudr-bci/contracts/{contract_id}/call": "eudr-bci:contracts:call",
    "GET:/api/v1/eudr-bci/contracts/{contract_id}/state": "eudr-bci:contracts:read",
    "GET:/api/v1/eudr-bci/contracts": "eudr-bci:contracts:read",
    # Chain
    "POST:/api/v1/eudr-bci/chains/connect": "eudr-bci:chains:connect",
    "GET:/api/v1/eudr-bci/chains/{chain_id}/status": "eudr-bci:chains:read",
    "GET:/api/v1/eudr-bci/chains": "eudr-bci:chains:read",
    "POST:/api/v1/eudr-bci/chains/{chain_id}/estimate-gas": "eudr-bci:chains:estimate_gas",
    # Verify
    "POST:/api/v1/eudr-bci/verify": "eudr-bci:verify",
    "POST:/api/v1/eudr-bci/verify/batch": "eudr-bci:verify:batch",
    "POST:/api/v1/eudr-bci/verify/merkle-proof": "eudr-bci:verify:merkle_proof",
    "GET:/api/v1/eudr-bci/verify/{verification_id}": "eudr-bci:verify:read",
    # Event
    "POST:/api/v1/eudr-bci/events/subscribe": "eudr-bci:events:subscribe",
    "DELETE:/api/v1/eudr-bci/events/subscribe/{subscription_id}": "eudr-bci:events:unsubscribe",
    "GET:/api/v1/eudr-bci/events": "eudr-bci:events:read",
    "GET:/api/v1/eudr-bci/events/{event_id}": "eudr-bci:events:read",
    "POST:/api/v1/eudr-bci/events/replay": "eudr-bci:events:replay",
    # Merkle
    "POST:/api/v1/eudr-bci/merkle/build": "eudr-bci:merkle:build",
    "GET:/api/v1/eudr-bci/merkle/{tree_id}": "eudr-bci:merkle:read",
    "POST:/api/v1/eudr-bci/merkle/{tree_id}/proof": "eudr-bci:merkle:proof",
    "POST:/api/v1/eudr-bci/merkle/verify": "eudr-bci:merkle:verify",
    # Sharing
    "POST:/api/v1/eudr-bci/sharing/grant": "eudr-bci:sharing:grant",
    "DELETE:/api/v1/eudr-bci/sharing/revoke/{grant_id}": "eudr-bci:sharing:revoke",
    "GET:/api/v1/eudr-bci/sharing/grants/{record_id}": "eudr-bci:sharing:read",
    "POST:/api/v1/eudr-bci/sharing/request": "eudr-bci:sharing:request",
    "POST:/api/v1/eudr-bci/sharing/confirm": "eudr-bci:sharing:confirm",
    # Evidence
    "POST:/api/v1/eudr-bci/evidence/package": "eudr-bci:evidence:generate",
    "GET:/api/v1/eudr-bci/evidence/{package_id}": "eudr-bci:evidence:read",
    "GET:/api/v1/eudr-bci/evidence/{package_id}/download": "eudr-bci:evidence:download",
    "POST:/api/v1/eudr-bci/evidence/verify": "eudr-bci:evidence:verify",
    # Batch
    "POST:/api/v1/eudr-bci/batch": "eudr-bci:batch:submit",
    "DELETE:/api/v1/eudr-bci/batch/{job_id}": "eudr-bci:batch:cancel",
    # Health
    "GET:/api/v1/eudr-bci/health": "eudr-bci:read",

    # ---- AGENT-EUDR-014: QR Code Generator (eudr-qrg) ----
    # QR Codes
    "POST:/api/v1/eudr-qrg/codes": "eudr-qrg:codes:create",
    "POST:/api/v1/eudr-qrg/codes/batch": "eudr-qrg:codes:batch",
    "GET:/api/v1/eudr-qrg/codes/{code_id}": "eudr-qrg:codes:read",
    "GET:/api/v1/eudr-qrg/codes/{code_id}/image": "eudr-qrg:codes:download",
    "PUT:/api/v1/eudr-qrg/codes/{code_id}/status": "eudr-qrg:codes:update",
    "GET:/api/v1/eudr-qrg/codes": "eudr-qrg:codes:read",
    # Payloads
    "POST:/api/v1/eudr-qrg/payloads/compose": "eudr-qrg:payloads:compose",
    "GET:/api/v1/eudr-qrg/payloads/{payload_id}": "eudr-qrg:payloads:read",
    "POST:/api/v1/eudr-qrg/payloads/validate": "eudr-qrg:payloads:validate",
    # Labels
    "POST:/api/v1/eudr-qrg/labels/render": "eudr-qrg:labels:render",
    "GET:/api/v1/eudr-qrg/labels/templates": "eudr-qrg:labels:read",
    "GET:/api/v1/eudr-qrg/labels/templates/{template_id}": "eudr-qrg:labels:read",
    "POST:/api/v1/eudr-qrg/labels/print-job": "eudr-qrg:labels:print",
    # Batch generation
    "POST:/api/v1/eudr-qrg/batch/generate": "eudr-qrg:batch:generate",
    "GET:/api/v1/eudr-qrg/batch/{batch_id}": "eudr-qrg:batch:read",
    "GET:/api/v1/eudr-qrg/batch/{batch_id}/download": "eudr-qrg:batch:download",
    "DELETE:/api/v1/eudr-qrg/batch/{batch_id}": "eudr-qrg:batch:cancel",
    # Verification URLs
    "POST:/api/v1/eudr-qrg/verification/urls": "eudr-qrg:verification:create",
    "GET:/api/v1/eudr-qrg/verification/{token}": "eudr-qrg:verification:read",
    "POST:/api/v1/eudr-qrg/verification/validate": "eudr-qrg:verification:validate",
    # Anti-counterfeit
    "POST:/api/v1/eudr-qrg/anti-counterfeit/tokens": "eudr-qrg:anti-counterfeit:create",
    "POST:/api/v1/eudr-qrg/anti-counterfeit/verify": "eudr-qrg:anti-counterfeit:verify",
    "GET:/api/v1/eudr-qrg/anti-counterfeit/stats": "eudr-qrg:anti-counterfeit:read",
    # Bulk pipeline
    "POST:/api/v1/eudr-qrg/bulk/submit": "eudr-qrg:bulk:submit",
    "GET:/api/v1/eudr-qrg/bulk/{job_id}": "eudr-qrg:bulk:read",
    "GET:/api/v1/eudr-qrg/bulk/{job_id}/download": "eudr-qrg:bulk:download",
    "DELETE:/api/v1/eudr-qrg/bulk/{job_id}": "eudr-qrg:bulk:cancel",
    # Lifecycle
    "POST:/api/v1/eudr-qrg/lifecycle/activate": "eudr-qrg:lifecycle:activate",
    "POST:/api/v1/eudr-qrg/lifecycle/revoke": "eudr-qrg:lifecycle:revoke",
    "POST:/api/v1/eudr-qrg/lifecycle/renew": "eudr-qrg:lifecycle:renew",
    "GET:/api/v1/eudr-qrg/lifecycle/{code_id}/history": "eudr-qrg:lifecycle:read",
    # Health
    "GET:/api/v1/eudr-qrg/health": "eudr-qrg:read",

    # ---- AGENT-EUDR-015: Mobile Data Collector (eudr-mdc) ----
    # Forms
    "POST:/api/v1/eudr-mdc/forms": "eudr-mdc:forms:submit",
    "POST:/api/v1/eudr-mdc/forms/validate": "eudr-mdc:forms:validate",
    "GET:/api/v1/eudr-mdc/forms/{form_id}": "eudr-mdc:forms:read",
    "GET:/api/v1/eudr-mdc/forms": "eudr-mdc:forms:read",
    "PUT:/api/v1/eudr-mdc/forms/{form_id}": "eudr-mdc:forms:update",
    # GPS
    "POST:/api/v1/eudr-mdc/gps/capture": "eudr-mdc:gps:capture",
    "POST:/api/v1/eudr-mdc/gps/polygon": "eudr-mdc:gps:capture",
    "GET:/api/v1/eudr-mdc/gps/{capture_id}": "eudr-mdc:gps:read",
    "GET:/api/v1/eudr-mdc/gps/form/{form_id}": "eudr-mdc:gps:read",
    # Photos
    "POST:/api/v1/eudr-mdc/photos/upload": "eudr-mdc:photos:upload",
    "GET:/api/v1/eudr-mdc/photos/{photo_id}": "eudr-mdc:photos:read",
    "GET:/api/v1/eudr-mdc/photos/{photo_id}/download": "eudr-mdc:photos:download",
    "POST:/api/v1/eudr-mdc/photos/validate": "eudr-mdc:photos:validate",
    # Sync
    "POST:/api/v1/eudr-mdc/sync/trigger": "eudr-mdc:sync:trigger",
    "GET:/api/v1/eudr-mdc/sync/status/{device_id}": "eudr-mdc:sync:read",
    "GET:/api/v1/eudr-mdc/sync/conflicts/{device_id}": "eudr-mdc:sync:read",
    "POST:/api/v1/eudr-mdc/sync/conflicts/{conflict_id}/resolve": "eudr-mdc:sync:resolve",
    "GET:/api/v1/eudr-mdc/sync/queue/{device_id}": "eudr-mdc:sync:read",
    # Templates
    "POST:/api/v1/eudr-mdc/templates": "eudr-mdc:templates:create",
    "GET:/api/v1/eudr-mdc/templates": "eudr-mdc:templates:read",
    "GET:/api/v1/eudr-mdc/templates/{template_id}": "eudr-mdc:templates:read",
    "PUT:/api/v1/eudr-mdc/templates/{template_id}": "eudr-mdc:templates:update",
    "DELETE:/api/v1/eudr-mdc/templates/{template_id}": "eudr-mdc:templates:delete",
    # Signatures
    "POST:/api/v1/eudr-mdc/signatures/capture": "eudr-mdc:signatures:capture",
    "POST:/api/v1/eudr-mdc/signatures/verify": "eudr-mdc:signatures:verify",
    "GET:/api/v1/eudr-mdc/signatures/{signature_id}": "eudr-mdc:signatures:read",
    # Packages
    "POST:/api/v1/eudr-mdc/packages/build": "eudr-mdc:packages:build",
    "GET:/api/v1/eudr-mdc/packages/{package_id}": "eudr-mdc:packages:read",
    "GET:/api/v1/eudr-mdc/packages/{package_id}/download": "eudr-mdc:packages:download",
    "POST:/api/v1/eudr-mdc/packages/{package_id}/verify": "eudr-mdc:packages:verify",
    # Devices
    "POST:/api/v1/eudr-mdc/devices/register": "eudr-mdc:devices:register",
    "GET:/api/v1/eudr-mdc/devices": "eudr-mdc:devices:read",
    "GET:/api/v1/eudr-mdc/devices/{device_id}": "eudr-mdc:devices:read",
    "PUT:/api/v1/eudr-mdc/devices/{device_id}": "eudr-mdc:devices:update",
    "GET:/api/v1/eudr-mdc/devices/fleet/status": "eudr-mdc:devices:fleet",
    # Health
    "GET:/api/v1/eudr-mdc/health": "eudr-mdc:read",

    # ── AGENT-EUDR-016: Country Risk Evaluator ──────────────────────────
    # Countries
    "POST:/api/v1/eudr-cre/countries/assess": "eudr-cre:countries:assess",
    "GET:/api/v1/eudr-cre/countries": "eudr-cre:countries:read",
    "GET:/api/v1/eudr-cre/countries/{country_code}": "eudr-cre:countries:read",
    "POST:/api/v1/eudr-cre/countries/compare": "eudr-cre:countries:compare",
    "GET:/api/v1/eudr-cre/countries/{country_code}/trends": "eudr-cre:countries:read",
    # Commodities
    "POST:/api/v1/eudr-cre/commodities/analyze": "eudr-cre:commodities:analyze",
    "GET:/api/v1/eudr-cre/commodities/{commodity_type}": "eudr-cre:commodities:read",
    "GET:/api/v1/eudr-cre/commodities/matrix": "eudr-cre:commodities:read",
    "GET:/api/v1/eudr-cre/commodities/correlations": "eudr-cre:commodities:read",
    # Hotspots
    "POST:/api/v1/eudr-cre/hotspots/detect": "eudr-cre:hotspots:detect",
    "GET:/api/v1/eudr-cre/hotspots/{hotspot_id}": "eudr-cre:hotspots:read",
    "GET:/api/v1/eudr-cre/hotspots": "eudr-cre:hotspots:read",
    "GET:/api/v1/eudr-cre/hotspots/alerts": "eudr-cre:hotspots:read",
    "POST:/api/v1/eudr-cre/hotspots/clustering": "eudr-cre:hotspots:detect",
    # Governance
    "POST:/api/v1/eudr-cre/governance/evaluate": "eudr-cre:governance:evaluate",
    "GET:/api/v1/eudr-cre/governance/{country_code}": "eudr-cre:governance:read",
    "GET:/api/v1/eudr-cre/governance": "eudr-cre:governance:read",
    "POST:/api/v1/eudr-cre/governance/compare": "eudr-cre:governance:compare",
    # Due Diligence
    "POST:/api/v1/eudr-cre/due-diligence/classify": "eudr-cre:due-diligence:classify",
    "GET:/api/v1/eudr-cre/due-diligence/{classification_id}": "eudr-cre:due-diligence:read",
    "GET:/api/v1/eudr-cre/due-diligence": "eudr-cre:due-diligence:read",
    "POST:/api/v1/eudr-cre/due-diligence/cost-estimate": "eudr-cre:due-diligence:classify",
    "GET:/api/v1/eudr-cre/due-diligence/audit-frequency": "eudr-cre:due-diligence:read",
    # Trade Flows
    "POST:/api/v1/eudr-cre/trade-flows/analyze": "eudr-cre:trade-flows:analyze",
    "GET:/api/v1/eudr-cre/trade-flows/{flow_id}": "eudr-cre:trade-flows:read",
    "GET:/api/v1/eudr-cre/trade-flows": "eudr-cre:trade-flows:read",
    "GET:/api/v1/eudr-cre/trade-flows/routes": "eudr-cre:trade-flows:read",
    "POST:/api/v1/eudr-cre/trade-flows/re-export-risk": "eudr-cre:trade-flows:analyze",
    # Reports
    "POST:/api/v1/eudr-cre/reports/generate": "eudr-cre:reports:generate",
    "GET:/api/v1/eudr-cre/reports/{report_id}": "eudr-cre:reports:read",
    "GET:/api/v1/eudr-cre/reports": "eudr-cre:reports:read",
    "GET:/api/v1/eudr-cre/reports/{report_id}/download": "eudr-cre:reports:download",
    "POST:/api/v1/eudr-cre/reports/executive-summary": "eudr-cre:reports:generate",
    # Regulatory
    "POST:/api/v1/eudr-cre/regulatory/track": "eudr-cre:regulatory:track",
    "GET:/api/v1/eudr-cre/regulatory/{update_id}": "eudr-cre:regulatory:read",
    "GET:/api/v1/eudr-cre/regulatory": "eudr-cre:regulatory:read",
    "POST:/api/v1/eudr-cre/regulatory/impact-assessment": "eudr-cre:regulatory:assess",
    # Health
    "GET:/api/v1/eudr-cre/health": "eudr-cre:read",
    # ── AGENT-EUDR-017: Supplier Risk Scorer ──────────────────────────────
    # Suppliers
    "POST:/api/v1/eudr-srs/suppliers/assess": "eudr-srs:suppliers:assess",
    "POST:/api/v1/eudr-srs/suppliers/assess-batch": "eudr-srs:suppliers:assess",
    "GET:/api/v1/eudr-srs/suppliers/{supplier_id}/risk": "eudr-srs:suppliers:read",
    "GET:/api/v1/eudr-srs/suppliers/{supplier_id}/trend": "eudr-srs:suppliers:read",
    "POST:/api/v1/eudr-srs/suppliers/compare": "eudr-srs:suppliers:compare",
    "GET:/api/v1/eudr-srs/suppliers/rankings": "eudr-srs:suppliers:read",
    # Due Diligence
    "GET:/api/v1/eudr-srs/due-diligence/{supplier_id}": "eudr-srs:due-diligence:read",
    "POST:/api/v1/eudr-srs/due-diligence/record": "eudr-srs:due-diligence:write",
    "GET:/api/v1/eudr-srs/due-diligence/{supplier_id}/history": "eudr-srs:due-diligence:read",
    "GET:/api/v1/eudr-srs/due-diligence/{supplier_id}/gaps": "eudr-srs:due-diligence:read",
    "POST:/api/v1/eudr-srs/due-diligence/escalate": "eudr-srs:due-diligence:escalate",
    # Documentation
    "POST:/api/v1/eudr-srs/documentation/analyze": "eudr-srs:documentation:analyze",
    "GET:/api/v1/eudr-srs/documentation/{supplier_id}": "eudr-srs:documentation:read",
    "GET:/api/v1/eudr-srs/documentation/{supplier_id}/gaps": "eudr-srs:documentation:read",
    "POST:/api/v1/eudr-srs/documentation/request": "eudr-srs:documentation:write",
    "GET:/api/v1/eudr-srs/documentation/{supplier_id}/expiry": "eudr-srs:documentation:read",
    # Certification
    "POST:/api/v1/eudr-srs/certification/validate": "eudr-srs:certification:validate",
    "GET:/api/v1/eudr-srs/certification/{supplier_id}": "eudr-srs:certification:read",
    "GET:/api/v1/eudr-srs/certification/{supplier_id}/expiry": "eudr-srs:certification:read",
    "POST:/api/v1/eudr-srs/certification/verify-scope": "eudr-srs:certification:validate",
    "GET:/api/v1/eudr-srs/certification/schemes": "eudr-srs:certification:read",
    # Geographic Sourcing
    "POST:/api/v1/eudr-srs/geographic/analyze": "eudr-srs:geographic:analyze",
    "GET:/api/v1/eudr-srs/geographic/{supplier_id}": "eudr-srs:geographic:read",
    "GET:/api/v1/eudr-srs/geographic/{supplier_id}/risk-zones": "eudr-srs:geographic:read",
    "POST:/api/v1/eudr-srs/geographic/concentration": "eudr-srs:geographic:analyze",
    "GET:/api/v1/eudr-srs/geographic/{supplier_id}/changes": "eudr-srs:geographic:read",
    # Network
    "POST:/api/v1/eudr-srs/network/analyze": "eudr-srs:network:analyze",
    "GET:/api/v1/eudr-srs/network/{supplier_id}": "eudr-srs:network:read",
    "GET:/api/v1/eudr-srs/network/{supplier_id}/sub-suppliers": "eudr-srs:network:read",
    "POST:/api/v1/eudr-srs/network/risk-propagation": "eudr-srs:network:analyze",
    "GET:/api/v1/eudr-srs/network/{supplier_id}/graph": "eudr-srs:network:read",
    # Monitoring
    "POST:/api/v1/eudr-srs/monitoring/configure": "eudr-srs:monitoring:write",
    "GET:/api/v1/eudr-srs/monitoring/{supplier_id}/alerts": "eudr-srs:monitoring:read",
    "GET:/api/v1/eudr-srs/monitoring/watchlist": "eudr-srs:monitoring:read",
    "POST:/api/v1/eudr-srs/monitoring/watchlist": "eudr-srs:monitoring:write",
    "GET:/api/v1/eudr-srs/monitoring/portfolio-risk": "eudr-srs:monitoring:read",
    # Reports
    "POST:/api/v1/eudr-srs/reports/generate": "eudr-srs:reports:generate",
    "POST:/api/v1/eudr-srs/reports/generate-batch": "eudr-srs:reports:generate",
    "GET:/api/v1/eudr-srs/reports/{report_id}": "eudr-srs:reports:read",
    "GET:/api/v1/eudr-srs/reports/{report_id}/download": "eudr-srs:reports:export",
    "POST:/api/v1/eudr-srs/reports/portfolio": "eudr-srs:reports:generate",
    # Health
    "GET:/api/v1/eudr-srs/health": "eudr-srs:read",

    # ── AGENT-EUDR-018: Commodity Risk Analyzer ──────────────────────────
    # Commodity Profiles (6 routes)
    "POST:/api/v1/eudr-cra/commodities/profile": "eudr-cra:commodities:profile",
    "POST:/api/v1/eudr-cra/commodities/profile-batch": "eudr-cra:commodities:profile",
    "GET:/api/v1/eudr-cra/commodities/{commodity_id}/risk": "eudr-cra:commodities:read",
    "GET:/api/v1/eudr-cra/commodities/{commodity_id}/history": "eudr-cra:commodities:read",
    "GET:/api/v1/eudr-cra/commodities/compare": "eudr-cra:commodities:compare",
    "GET:/api/v1/eudr-cra/commodities/summary": "eudr-cra:commodities:read",
    # Derived Products (5 routes)
    "POST:/api/v1/eudr-cra/derived-products/analyze": "eudr-cra:derived-products:analyze",
    "GET:/api/v1/eudr-cra/derived-products/{product_id}/chain": "eudr-cra:derived-products:read",
    "GET:/api/v1/eudr-cra/derived-products/{product_id}/risk": "eudr-cra:derived-products:read",
    "GET:/api/v1/eudr-cra/derived-products/mapping": "eudr-cra:derived-products:read",
    "POST:/api/v1/eudr-cra/derived-products/trace": "eudr-cra:derived-products:analyze",
    # Price & Market (5 routes)
    "GET:/api/v1/eudr-cra/price/{commodity_id}/current": "eudr-cra:price:read",
    "GET:/api/v1/eudr-cra/price/{commodity_id}/history": "eudr-cra:price:read",
    "GET:/api/v1/eudr-cra/price/{commodity_id}/volatility": "eudr-cra:price:analyze",
    "GET:/api/v1/eudr-cra/price/market-disruptions": "eudr-cra:price:read",
    "POST:/api/v1/eudr-cra/price/forecast": "eudr-cra:price:analyze",
    # Production Forecasts (5 routes)
    "POST:/api/v1/eudr-cra/production/forecast": "eudr-cra:production:forecast",
    "GET:/api/v1/eudr-cra/production/{commodity_id}/yield": "eudr-cra:production:read",
    "GET:/api/v1/eudr-cra/production/{commodity_id}/climate-impact": "eudr-cra:production:read",
    "GET:/api/v1/eudr-cra/production/{commodity_id}/seasonal": "eudr-cra:production:read",
    "GET:/api/v1/eudr-cra/production/summary": "eudr-cra:production:read",
    # Substitution Risk (5 routes)
    "POST:/api/v1/eudr-cra/substitution/detect": "eudr-cra:substitution:analyze",
    "GET:/api/v1/eudr-cra/substitution/{supplier_id}/history": "eudr-cra:substitution:read",
    "GET:/api/v1/eudr-cra/substitution/alerts": "eudr-cra:substitution:read",
    "POST:/api/v1/eudr-cra/substitution/verify": "eudr-cra:substitution:analyze",
    "GET:/api/v1/eudr-cra/substitution/patterns": "eudr-cra:substitution:read",
    # Regulatory Compliance (5 routes)
    "GET:/api/v1/eudr-cra/regulatory/{commodity_id}/requirements": "eudr-cra:regulatory:read",
    "POST:/api/v1/eudr-cra/regulatory/check-compliance": "eudr-cra:regulatory:analyze",
    "GET:/api/v1/eudr-cra/regulatory/penalty-risk": "eudr-cra:regulatory:read",
    "GET:/api/v1/eudr-cra/regulatory/updates": "eudr-cra:regulatory:read",
    "GET:/api/v1/eudr-cra/regulatory/documentation-requirements": "eudr-cra:regulatory:read",
    # Due Diligence (5 routes)
    "POST:/api/v1/eudr-cra/due-diligence/initiate": "eudr-cra:due-diligence:write",
    "GET:/api/v1/eudr-cra/due-diligence/{workflow_id}/status": "eudr-cra:due-diligence:read",
    "POST:/api/v1/eudr-cra/due-diligence/{workflow_id}/evidence": "eudr-cra:due-diligence:write",
    "GET:/api/v1/eudr-cra/due-diligence/pending": "eudr-cra:due-diligence:read",
    "POST:/api/v1/eudr-cra/due-diligence/{workflow_id}/complete": "eudr-cra:due-diligence:write",
    # Portfolio Aggregation (4 routes)
    "POST:/api/v1/eudr-cra/portfolio/analyze": "eudr-cra:portfolio:analyze",
    "GET:/api/v1/eudr-cra/portfolio/concentration": "eudr-cra:portfolio:read",
    "GET:/api/v1/eudr-cra/portfolio/diversification": "eudr-cra:portfolio:read",
    "GET:/api/v1/eudr-cra/portfolio/summary": "eudr-cra:portfolio:read",
    # Health (1 route)
    "GET:/api/v1/eudr-cra/health": "eudr-cra:read",

    # ── AGENT-EUDR-019: Corruption Index Monitor ─────────────────────────
    # CPI Monitoring (6 routes)
    "GET:/api/v1/eudr-cim/cpi/{country_code}/score": "eudr-cim:cpi:read",
    "GET:/api/v1/eudr-cim/cpi/{country_code}/history": "eudr-cim:cpi:read",
    "GET:/api/v1/eudr-cim/cpi/rankings": "eudr-cim:cpi:read",
    "GET:/api/v1/eudr-cim/cpi/regional": "eudr-cim:cpi:read",
    "POST:/api/v1/eudr-cim/cpi/batch-query": "eudr-cim:cpi:read",
    "GET:/api/v1/eudr-cim/cpi/summary": "eudr-cim:cpi:read",
    # WGI Analysis (5 routes)
    "GET:/api/v1/eudr-cim/wgi/{country_code}/indicators": "eudr-cim:wgi:read",
    "GET:/api/v1/eudr-cim/wgi/{country_code}/history": "eudr-cim:wgi:read",
    "GET:/api/v1/eudr-cim/wgi/{country_code}/dimension/{dimension}": "eudr-cim:wgi:read",
    "POST:/api/v1/eudr-cim/wgi/compare": "eudr-cim:wgi:analyze",
    "GET:/api/v1/eudr-cim/wgi/rankings": "eudr-cim:wgi:read",
    # Bribery Risk (5 routes)
    "POST:/api/v1/eudr-cim/bribery/assess": "eudr-cim:bribery:analyze",
    "GET:/api/v1/eudr-cim/bribery/{country_code}/risk": "eudr-cim:bribery:read",
    "GET:/api/v1/eudr-cim/bribery/{country_code}/sectors": "eudr-cim:bribery:read",
    "GET:/api/v1/eudr-cim/bribery/high-risk-countries": "eudr-cim:bribery:read",
    "POST:/api/v1/eudr-cim/bribery/sector-analysis": "eudr-cim:bribery:analyze",
    # Institutional Quality (5 routes)
    "GET:/api/v1/eudr-cim/institutional/{country_code}/quality": "eudr-cim:institutional:read",
    "GET:/api/v1/eudr-cim/institutional/{country_code}/governance": "eudr-cim:institutional:read",
    "POST:/api/v1/eudr-cim/institutional/assess": "eudr-cim:institutional:analyze",
    "GET:/api/v1/eudr-cim/institutional/forest-governance": "eudr-cim:institutional:read",
    "POST:/api/v1/eudr-cim/institutional/compare": "eudr-cim:institutional:analyze",
    # Trend Analysis (5 routes)
    "POST:/api/v1/eudr-cim/trends/analyze": "eudr-cim:trends:analyze",
    "GET:/api/v1/eudr-cim/trends/{country_code}/trajectory": "eudr-cim:trends:read",
    "GET:/api/v1/eudr-cim/trends/{country_code}/prediction": "eudr-cim:trends:read",
    "GET:/api/v1/eudr-cim/trends/improving": "eudr-cim:trends:read",
    "GET:/api/v1/eudr-cim/trends/deteriorating": "eudr-cim:trends:read",
    # Deforestation Correlation (5 routes)
    "POST:/api/v1/eudr-cim/correlation/analyze": "eudr-cim:correlation:analyze",
    "GET:/api/v1/eudr-cim/correlation/{country_code}/deforestation": "eudr-cim:correlation:read",
    "GET:/api/v1/eudr-cim/correlation/regression": "eudr-cim:correlation:read",
    "GET:/api/v1/eudr-cim/correlation/heatmap": "eudr-cim:correlation:read",
    "GET:/api/v1/eudr-cim/correlation/causal-pathways": "eudr-cim:correlation:read",
    # Alert Management (5 routes)
    "GET:/api/v1/eudr-cim/alerts": "eudr-cim:alerts:read",
    "GET:/api/v1/eudr-cim/alerts/{alert_id}": "eudr-cim:alerts:read",
    "POST:/api/v1/eudr-cim/alerts/configure": "eudr-cim:alerts:write",
    "POST:/api/v1/eudr-cim/alerts/{alert_id}/acknowledge": "eudr-cim:alerts:write",
    "GET:/api/v1/eudr-cim/alerts/summary": "eudr-cim:alerts:read",
    # Compliance Impact (4 routes)
    "POST:/api/v1/eudr-cim/compliance/assess-impact": "eudr-cim:compliance:analyze",
    "GET:/api/v1/eudr-cim/compliance/{country_code}/impact": "eudr-cim:compliance:read",
    "GET:/api/v1/eudr-cim/compliance/dd-recommendations": "eudr-cim:compliance:read",
    "GET:/api/v1/eudr-cim/compliance/country-classifications": "eudr-cim:compliance:read",
    # Health (1 route)
    "GET:/api/v1/eudr-cim/health": "eudr-cim:read",

    # =========================================================================
    # AGENT-EUDR-020: Deforestation Alert System
    # =========================================================================

    # Satellite Change Detection (4 routes)
    "POST:/api/v1/eudr-das/satellite/detect": "eudr-das:satellite:detect",
    "POST:/api/v1/eudr-das/satellite/scan": "eudr-das:satellite:detect",
    "GET:/api/v1/eudr-das/satellite/sources": "eudr-das:satellite:read",
    "GET:/api/v1/eudr-das/satellite/{detection_id}/imagery": "eudr-das:satellite:read",

    # Alert Management (6 routes)
    "GET:/api/v1/eudr-das/alerts": "eudr-das:alerts:read",
    "GET:/api/v1/eudr-das/alerts/{alert_id}": "eudr-das:alerts:read",
    "POST:/api/v1/eudr-das/alerts": "eudr-das:alerts:write",
    "POST:/api/v1/eudr-das/alerts/batch": "eudr-das:alerts:write",
    "GET:/api/v1/eudr-das/alerts/summary": "eudr-das:alerts:read",
    "GET:/api/v1/eudr-das/alerts/statistics": "eudr-das:alerts:read",

    # Severity Classification (4 routes)
    "POST:/api/v1/eudr-das/severity/classify": "eudr-das:severity:analyze",
    "POST:/api/v1/eudr-das/severity/reclassify": "eudr-das:severity:analyze",
    "GET:/api/v1/eudr-das/severity/thresholds": "eudr-das:severity:read",
    "GET:/api/v1/eudr-das/severity/distribution": "eudr-das:severity:read",

    # Spatial Buffer Management (5 routes)
    "POST:/api/v1/eudr-das/buffer/create": "eudr-das:buffer:write",
    "PUT:/api/v1/eudr-das/buffer/{buffer_id}": "eudr-das:buffer:write",
    "POST:/api/v1/eudr-das/buffer/check": "eudr-das:buffer:read",
    "GET:/api/v1/eudr-das/buffer/violations": "eudr-das:buffer:read",
    "GET:/api/v1/eudr-das/buffer/zones": "eudr-das:buffer:read",

    # Cutoff Date Verification (4 routes)
    "POST:/api/v1/eudr-das/cutoff/verify": "eudr-das:cutoff:analyze",
    "POST:/api/v1/eudr-das/cutoff/batch-verify": "eudr-das:cutoff:analyze",
    "GET:/api/v1/eudr-das/cutoff/{detection_id}/evidence": "eudr-das:cutoff:read",
    "GET:/api/v1/eudr-das/cutoff/{detection_id}/timeline": "eudr-das:cutoff:read",

    # Historical Baseline (4 routes)
    "POST:/api/v1/eudr-das/baseline/establish": "eudr-das:baseline:write",
    "POST:/api/v1/eudr-das/baseline/compare": "eudr-das:baseline:analyze",
    "PUT:/api/v1/eudr-das/baseline/{baseline_id}": "eudr-das:baseline:write",
    "GET:/api/v1/eudr-das/baseline/coverage": "eudr-das:baseline:read",

    # Alert Workflow (6 routes)
    "POST:/api/v1/eudr-das/workflow/triage": "eudr-das:workflow:write",
    "POST:/api/v1/eudr-das/workflow/assign": "eudr-das:workflow:write",
    "POST:/api/v1/eudr-das/workflow/investigate": "eudr-das:workflow:write",
    "POST:/api/v1/eudr-das/workflow/resolve": "eudr-das:workflow:write",
    "POST:/api/v1/eudr-das/workflow/escalate": "eudr-das:workflow:write",
    "GET:/api/v1/eudr-das/workflow/sla": "eudr-das:workflow:read",

    # Compliance Impact (4 routes)
    "POST:/api/v1/eudr-das/compliance/assess": "eudr-das:compliance:analyze",
    "GET:/api/v1/eudr-das/compliance/{alert_id}/affected-products": "eudr-das:compliance:read",
    "GET:/api/v1/eudr-das/compliance/recommendations": "eudr-das:compliance:read",
    "POST:/api/v1/eudr-das/compliance/remediation": "eudr-das:compliance:write",

    # Health (1 route)
    "GET:/api/v1/eudr-das/health": "eudr-das:read",

    # ── AGENT-EUDR-021: Indigenous Rights Checker ─────────────────────────
    # Territory Management (5 routes)
    "POST:/api/v1/eudr-irc/territories": "eudr-irc:territories:create",
    "GET:/api/v1/eudr-irc/territories": "eudr-irc:territories:read",
    "GET:/api/v1/eudr-irc/territories/{territory_id}": "eudr-irc:territories:read",
    "PUT:/api/v1/eudr-irc/territories/{territory_id}": "eudr-irc:territories:update",
    "DELETE:/api/v1/eudr-irc/territories/{territory_id}": "eudr-irc:territories:delete",

    # FPIC Verification (4 routes)
    "POST:/api/v1/eudr-irc/fpic/verify": "eudr-irc:fpic:verify",
    "GET:/api/v1/eudr-irc/fpic/documents": "eudr-irc:fpic:read",
    "GET:/api/v1/eudr-irc/fpic/documents/{doc_id}": "eudr-irc:fpic:read",
    "POST:/api/v1/eudr-irc/fpic/score": "eudr-irc:fpic:verify",

    # Land Rights Overlap (4 routes)
    "POST:/api/v1/eudr-irc/overlap/analyze": "eudr-irc:overlap:analyze",
    "GET:/api/v1/eudr-irc/overlap/by-plot/{plot_id}": "eudr-irc:overlap:read",
    "GET:/api/v1/eudr-irc/overlap/by-territory/{territory_id}": "eudr-irc:overlap:read",
    "POST:/api/v1/eudr-irc/overlap/bulk": "eudr-irc:overlap:analyze",

    # Community Consultations (3 routes)
    "POST:/api/v1/eudr-irc/consultations": "eudr-irc:consultations:create",
    "GET:/api/v1/eudr-irc/consultations": "eudr-irc:consultations:read",
    "GET:/api/v1/eudr-irc/consultations/{consultation_id}": "eudr-irc:consultations:read",

    # Rights Violations (4 routes)
    "POST:/api/v1/eudr-irc/violations/detect": "eudr-irc:violations:detect",
    "GET:/api/v1/eudr-irc/violations": "eudr-irc:violations:read",
    "GET:/api/v1/eudr-irc/violations/{violation_id}": "eudr-irc:violations:read",
    "PUT:/api/v1/eudr-irc/violations/{violation_id}/resolve": "eudr-irc:violations:resolve",

    # Indigenous Community Registry (3 routes)
    "POST:/api/v1/eudr-irc/communities": "eudr-irc:consultations:create",
    "GET:/api/v1/eudr-irc/communities": "eudr-irc:consultations:read",
    "GET:/api/v1/eudr-irc/communities/{community_id}": "eudr-irc:consultations:read",

    # Compliance Reporting (2 routes)
    "GET:/api/v1/eudr-irc/compliance/report/{plot_id}": "eudr-irc:compliance:read",
    "POST:/api/v1/eudr-irc/compliance/assess": "eudr-irc:compliance:read",

    # Health (1 route)
    "GET:/api/v1/eudr-irc/health": "eudr-irc:compliance:read",

    # ── AGENT-EUDR-022: Protected Area Validator ──────────────────────────
    # Protected Area Management (6 routes)
    "POST:/api/v1/eudr-pav/protected-areas": "eudr-pav:protected-areas:create",
    "GET:/api/v1/eudr-pav/protected-areas": "eudr-pav:protected-areas:read",
    "GET:/api/v1/eudr-pav/protected-areas/{area_id}": "eudr-pav:protected-areas:read",
    "PUT:/api/v1/eudr-pav/protected-areas/{area_id}": "eudr-pav:protected-areas:update",
    "DELETE:/api/v1/eudr-pav/protected-areas/{area_id}": "eudr-pav:protected-areas:delete",
    "POST:/api/v1/eudr-pav/protected-areas/search": "eudr-pav:protected-areas:read",

    # Spatial Overlap Detection (5 routes)
    "POST:/api/v1/eudr-pav/overlap/detect": "eudr-pav:overlap:analyze",
    "POST:/api/v1/eudr-pav/overlap/analyze": "eudr-pav:overlap:analyze",
    "POST:/api/v1/eudr-pav/overlap/bulk": "eudr-pav:overlap:analyze",
    "GET:/api/v1/eudr-pav/overlap/by-plot/{plot_id}": "eudr-pav:overlap:read",
    "GET:/api/v1/eudr-pav/overlap/by-area/{area_id}": "eudr-pav:overlap:read",

    # Buffer Zone Monitoring (4 routes)
    "POST:/api/v1/eudr-pav/buffer-zones/monitor": "eudr-pav:buffer-zones:monitor",
    "GET:/api/v1/eudr-pav/buffer-zones/violations": "eudr-pav:buffer-zones:read",
    "POST:/api/v1/eudr-pav/buffer-zones/analyze": "eudr-pav:buffer-zones:monitor",
    "POST:/api/v1/eudr-pav/buffer-zones/bulk": "eudr-pav:buffer-zones:monitor",

    # Designation Validation (3 routes)
    "POST:/api/v1/eudr-pav/designation/validate": "eudr-pav:designation:validate",
    "GET:/api/v1/eudr-pav/designation/status/{area_id}": "eudr-pav:designation:read",
    "GET:/api/v1/eudr-pav/designation/history/{area_id}": "eudr-pav:designation:read",

    # Risk Scoring (4 routes)
    "POST:/api/v1/eudr-pav/risk/score": "eudr-pav:risk:score",
    "GET:/api/v1/eudr-pav/risk/heatmap": "eudr-pav:risk:read",
    "GET:/api/v1/eudr-pav/risk/summary": "eudr-pav:risk:read",
    "GET:/api/v1/eudr-pav/risk/proximity-alerts": "eudr-pav:risk:read",

    # Violation Management (4 routes)
    "POST:/api/v1/eudr-pav/violations/detect": "eudr-pav:violations:detect",
    "GET:/api/v1/eudr-pav/violations": "eudr-pav:violations:read",
    "PUT:/api/v1/eudr-pav/violations/{violation_id}/resolve": "eudr-pav:violations:resolve",
    "PUT:/api/v1/eudr-pav/violations/{violation_id}/escalate": "eudr-pav:violations:resolve",

    # Compliance Reporting (3 routes)
    "POST:/api/v1/eudr-pav/compliance/assess": "eudr-pav:compliance:read",
    "GET:/api/v1/eudr-pav/compliance/report/{plot_id}": "eudr-pav:compliance:read",
    "GET:/api/v1/eudr-pav/compliance/audit-trail/{plot_id}": "eudr-pav:compliance:read",

    # PADDD Monitoring (3 routes)
    "POST:/api/v1/eudr-pav/paddd/monitor": "eudr-pav:paddd:monitor",
    "GET:/api/v1/eudr-pav/paddd/events": "eudr-pav:paddd:read",
    "POST:/api/v1/eudr-pav/paddd/impact-assessment": "eudr-pav:paddd:monitor",

    # Health (1 route)
    "GET:/api/v1/eudr-pav/health": "eudr-pav:compliance:read",

    # ── AGENT-EUDR-023: Legal Compliance Verifier ──────────────────────────
    # Legal Framework Management (6 routes)
    "POST:/api/v1/eudr-lcv/frameworks": "eudr-lcv:framework:create",
    "GET:/api/v1/eudr-lcv/frameworks": "eudr-lcv:framework:read",
    "GET:/api/v1/eudr-lcv/frameworks/{framework_id}": "eudr-lcv:framework:read",
    "PUT:/api/v1/eudr-lcv/frameworks/{framework_id}": "eudr-lcv:framework:update",
    "POST:/api/v1/eudr-lcv/frameworks/search": "eudr-lcv:framework:read",
    "GET:/api/v1/eudr-lcv/frameworks/by-country/{country_code}": "eudr-lcv:framework:read",

    # Document Compliance Validation (5 routes)
    "POST:/api/v1/eudr-lcv/documents/validate": "eudr-lcv:document:create",
    "GET:/api/v1/eudr-lcv/documents": "eudr-lcv:document:read",
    "GET:/api/v1/eudr-lcv/documents/{document_id}": "eudr-lcv:document:read",
    "POST:/api/v1/eudr-lcv/documents/bulk-validate": "eudr-lcv:document:create",
    "GET:/api/v1/eudr-lcv/documents/validation-status/{document_id}": "eudr-lcv:document:read",

    # Certification Verification (5 routes)
    "POST:/api/v1/eudr-lcv/certifications/verify": "eudr-lcv:certification:create",
    "GET:/api/v1/eudr-lcv/certifications": "eudr-lcv:certification:read",
    "GET:/api/v1/eudr-lcv/certifications/{certification_id}": "eudr-lcv:certification:read",
    "POST:/api/v1/eudr-lcv/certifications/bulk-verify": "eudr-lcv:certification:create",
    "GET:/api/v1/eudr-lcv/certifications/by-supplier/{supplier_id}": "eudr-lcv:certification:read",

    # Red Flag Detection (5 routes)
    "POST:/api/v1/eudr-lcv/red-flags/detect": "eudr-lcv:red-flag:create",
    "GET:/api/v1/eudr-lcv/red-flags": "eudr-lcv:red-flag:read",
    "GET:/api/v1/eudr-lcv/red-flags/{red_flag_id}": "eudr-lcv:red-flag:read",
    "PUT:/api/v1/eudr-lcv/red-flags/{red_flag_id}/suppress": "eudr-lcv:red-flag:update",
    "POST:/api/v1/eudr-lcv/red-flags/bulk-detect": "eudr-lcv:red-flag:create",

    # Compliance Assessment (4 routes)
    "POST:/api/v1/eudr-lcv/compliance/assess": "eudr-lcv:compliance:create",
    "GET:/api/v1/eudr-lcv/compliance/assessments": "eudr-lcv:compliance:read",
    "GET:/api/v1/eudr-lcv/compliance/assessments/{assessment_id}": "eudr-lcv:compliance:read",
    "GET:/api/v1/eudr-lcv/compliance/summary/{supplier_id}": "eudr-lcv:compliance:read",

    # Audit Report Processing (5 routes)
    "POST:/api/v1/eudr-lcv/audits/ingest": "eudr-lcv:audit:create",
    "GET:/api/v1/eudr-lcv/audits": "eudr-lcv:audit:read",
    "GET:/api/v1/eudr-lcv/audits/{audit_id}": "eudr-lcv:audit:read",
    "PUT:/api/v1/eudr-lcv/audits/{audit_id}/corrective-actions": "eudr-lcv:audit:update",
    "GET:/api/v1/eudr-lcv/audits/by-supplier/{supplier_id}": "eudr-lcv:audit:read",

    # Compliance Reporting (3 routes)
    "POST:/api/v1/eudr-lcv/reports/generate": "eudr-lcv:report:create",
    "GET:/api/v1/eudr-lcv/reports": "eudr-lcv:report:read",
    "GET:/api/v1/eudr-lcv/reports/{report_id}": "eudr-lcv:report:read",

    # Batch Operations (2 routes)
    "POST:/api/v1/eudr-lcv/batch/process": "eudr-lcv:batch:create",
    "GET:/api/v1/eudr-lcv/batch/{batch_id}/status": "eudr-lcv:batch:read",

    # Health (1 route)
    "GET:/api/v1/eudr-lcv/health": "eudr-lcv:compliance:read",

    # ── AGENT-EUDR-024: Third-Party Audit Manager ────────────────────────
    # Audit Management (6 routes)
    "POST:/api/v1/eudr-tam/audits": "eudr-tam:audit:create",
    "GET:/api/v1/eudr-tam/audits": "eudr-tam:audit:read",
    "GET:/api/v1/eudr-tam/audits/{audit_id}": "eudr-tam:audit:read",
    "POST:/api/v1/eudr-tam/audits/schedule": "eudr-tam:audit:schedule",
    "POST:/api/v1/eudr-tam/audits/{audit_id}/start": "eudr-tam:audit:execute",
    "POST:/api/v1/eudr-tam/audits/{audit_id}/complete": "eudr-tam:audit:execute",

    # Auditor Registry (5 routes)
    "POST:/api/v1/eudr-tam/auditors": "eudr-tam:auditor:create",
    "GET:/api/v1/eudr-tam/auditors": "eudr-tam:auditor:read",
    "GET:/api/v1/eudr-tam/auditors/{auditor_id}": "eudr-tam:auditor:read",
    "POST:/api/v1/eudr-tam/auditors/match": "eudr-tam:auditor:match",
    "POST:/api/v1/eudr-tam/auditors/{auditor_id}/qualification": "eudr-tam:auditor:update",

    # Checklist Management (3 routes)
    "GET:/api/v1/eudr-tam/checklists": "eudr-tam:checklist:read",
    "POST:/api/v1/eudr-tam/checklists/custom": "eudr-tam:checklist:create",
    "POST:/api/v1/eudr-tam/audits/{audit_id}/checklist-progress": "eudr-tam:checklist:update",

    # Evidence Collection (3 routes)
    "POST:/api/v1/eudr-tam/audits/{audit_id}/evidence": "eudr-tam:evidence:create",
    "GET:/api/v1/eudr-tam/audits/{audit_id}/evidence": "eudr-tam:evidence:read",
    "DELETE:/api/v1/eudr-tam/evidence/{evidence_id}": "eudr-tam:evidence:delete",

    # Non-Conformance Management (5 routes)
    "POST:/api/v1/eudr-tam/audits/{audit_id}/ncs": "eudr-tam:nc:create",
    "GET:/api/v1/eudr-tam/ncs": "eudr-tam:nc:read",
    "GET:/api/v1/eudr-tam/ncs/{nc_id}": "eudr-tam:nc:read",
    "POST:/api/v1/eudr-tam/ncs/{nc_id}/classify": "eudr-tam:nc:classify",
    "POST:/api/v1/eudr-tam/ncs/{nc_id}/root-cause": "eudr-tam:nc:create",

    # CAR Management (6 routes)
    "POST:/api/v1/eudr-tam/ncs/{nc_id}/car": "eudr-tam:car:create",
    "GET:/api/v1/eudr-tam/cars": "eudr-tam:car:read",
    "GET:/api/v1/eudr-tam/cars/{car_id}": "eudr-tam:car:read",
    "POST:/api/v1/eudr-tam/cars/{car_id}/submit-plan": "eudr-tam:car:update",
    "POST:/api/v1/eudr-tam/cars/{car_id}/verify": "eudr-tam:car:verify",
    "POST:/api/v1/eudr-tam/cars/{car_id}/close": "eudr-tam:car:close",

    # Certification Integration (4 routes)
    "POST:/api/v1/eudr-tam/certificates": "eudr-tam:certificate:create",
    "GET:/api/v1/eudr-tam/certificates": "eudr-tam:certificate:read",
    "GET:/api/v1/eudr-tam/suppliers/{supplier_id}/certificates": "eudr-tam:certificate:read",
    "POST:/api/v1/eudr-tam/certificates/validate-eudr": "eudr-tam:certificate:validate",

    # Report Generation (4 routes)
    "POST:/api/v1/eudr-tam/reports/generate": "eudr-tam:report:create",
    "GET:/api/v1/eudr-tam/reports": "eudr-tam:report:read",
    "GET:/api/v1/eudr-tam/reports/{report_id}": "eudr-tam:report:read",
    "GET:/api/v1/eudr-tam/reports/{report_id}/download": "eudr-tam:report:download",

    # Authority Liaison (3 routes)
    "POST:/api/v1/eudr-tam/authority-interactions": "eudr-tam:authority:create",
    "GET:/api/v1/eudr-tam/authority-interactions": "eudr-tam:authority:read",
    "POST:/api/v1/eudr-tam/authority-inspections/respond": "eudr-tam:authority:respond",

    # Analytics (2 routes)
    "GET:/api/v1/eudr-tam/analytics/compliance-rate": "eudr-tam:analytics:read",
    "GET:/api/v1/eudr-tam/analytics/nc-trends": "eudr-tam:analytics:read",

    # System (2 routes)
    "GET:/api/v1/eudr-tam/health": "eudr-tam:analytics:read",
    "GET:/api/v1/eudr-tam/stats": "eudr-tam:analytics:read",

    # ── AGENT-EUDR-025: Risk Mitigation Advisor ────────────────────────
    # Strategy Generation (5 routes)
    "POST:/api/v1/eudr-rma/strategies/generate": "eudr-rma:strategy:create",
    "GET:/api/v1/eudr-rma/strategies": "eudr-rma:strategy:read",
    "GET:/api/v1/eudr-rma/strategies/{strategy_id}": "eudr-rma:strategy:read",
    "POST:/api/v1/eudr-rma/strategies/{strategy_id}/select": "eudr-rma:strategy:select",
    "POST:/api/v1/eudr-rma/strategies/{strategy_id}/approve": "eudr-rma:strategy:select",

    # Remediation Plans (6 routes)
    "POST:/api/v1/eudr-rma/plans": "eudr-rma:plan:create",
    "GET:/api/v1/eudr-rma/plans": "eudr-rma:plan:read",
    "GET:/api/v1/eudr-rma/plans/{plan_id}": "eudr-rma:plan:read",
    "PUT:/api/v1/eudr-rma/plans/{plan_id}": "eudr-rma:plan:update",
    "PATCH:/api/v1/eudr-rma/plans/{plan_id}/status": "eudr-rma:plan:update",
    "POST:/api/v1/eudr-rma/plans/{plan_id}/clone": "eudr-rma:plan:clone",

    # Capacity Building (5 routes)
    "POST:/api/v1/eudr-rma/capacity/enroll": "eudr-rma:capacity:create",
    "GET:/api/v1/eudr-rma/capacity/programs": "eudr-rma:capacity:read",
    "GET:/api/v1/eudr-rma/capacity/programs/{program_id}": "eudr-rma:capacity:read",
    "PUT:/api/v1/eudr-rma/capacity/suppliers/{supplier_id}/progress": "eudr-rma:capacity:update",
    "GET:/api/v1/eudr-rma/capacity/suppliers/{supplier_id}": "eudr-rma:capacity:read",

    # Mitigation Library (3 routes)
    "GET:/api/v1/eudr-rma/library/measures": "eudr-rma:library:read",
    "GET:/api/v1/eudr-rma/library/measures/{measure_id}": "eudr-rma:library:read",
    "GET:/api/v1/eudr-rma/library/search": "eudr-rma:library:read",

    # Effectiveness Tracking (3 routes)
    "GET:/api/v1/eudr-rma/effectiveness": "eudr-rma:effectiveness:read",
    "GET:/api/v1/eudr-rma/effectiveness/{plan_id}": "eudr-rma:effectiveness:read",
    "GET:/api/v1/eudr-rma/effectiveness/trends": "eudr-rma:effectiveness:read",

    # Continuous Monitoring (4 routes)
    "GET:/api/v1/eudr-rma/monitoring/dashboard": "eudr-rma:monitoring:read",
    "GET:/api/v1/eudr-rma/monitoring/triggers": "eudr-rma:monitoring:read",
    "POST:/api/v1/eudr-rma/monitoring/triggers/{trigger_id}/acknowledge": "eudr-rma:monitoring:update",
    "PUT:/api/v1/eudr-rma/monitoring/triggers/{trigger_id}": "eudr-rma:monitoring:update",

    # Budget Optimization (4 routes)
    "POST:/api/v1/eudr-rma/optimization/run": "eudr-rma:optimization:create",
    "GET:/api/v1/eudr-rma/optimization/results": "eudr-rma:optimization:read",
    "GET:/api/v1/eudr-rma/optimization/results/{run_id}": "eudr-rma:optimization:read",
    "GET:/api/v1/eudr-rma/optimization/scenarios": "eudr-rma:optimization:read",

    # Collaboration Hub (4 routes)
    "POST:/api/v1/eudr-rma/collaboration/messages": "eudr-rma:collaboration:create",
    "GET:/api/v1/eudr-rma/collaboration/messages": "eudr-rma:collaboration:read",
    "POST:/api/v1/eudr-rma/collaboration/tasks": "eudr-rma:collaboration:create",
    "GET:/api/v1/eudr-rma/collaboration/tasks": "eudr-rma:collaboration:read",

    # Reporting (4 routes)
    "POST:/api/v1/eudr-rma/reports/generate": "eudr-rma:report:create",
    "GET:/api/v1/eudr-rma/reports": "eudr-rma:report:read",
    "GET:/api/v1/eudr-rma/reports/{report_id}": "eudr-rma:report:read",
    "GET:/api/v1/eudr-rma/reports/{report_id}/download": "eudr-rma:report:read",

    # Reporting (additional export routes - 2 routes)
    "POST:/api/v1/eudr-rma/reports/{report_id}/export": "eudr-rma:report:create",
    "GET:/api/v1/eudr-rma/reports/summary": "eudr-rma:report:read",

    # Capacity Building (additional batch routes - 2 routes)
    "POST:/api/v1/eudr-rma/capacity/batch-enroll": "eudr-rma:capacity:create",
    "GET:/api/v1/eudr-rma/capacity/statistics": "eudr-rma:capacity:read",

    # System (4 routes)
    "GET:/api/v1/eudr-rma/health": "eudr-rma:monitoring:read",
    "GET:/api/v1/eudr-rma/stats": "eudr-rma:monitoring:read",
    "GET:/api/v1/eudr-rma/config": "eudr-rma:monitoring:read",
    "GET:/api/v1/eudr-rma/version": "eudr-rma:monitoring:read",

    # -----------------------------------------------------------------------
    # EUDR Information Gathering Agent (AGENT-EUDR-027)
    # Regulation: EU 2023/1115 (EUDR) Articles 4, 9, 10, 12, 13, 29, 31
    # -----------------------------------------------------------------------

    # Gathering Operations (4 routes)
    "POST:/api/v1/gathering/operations": "eudr-iga:operations:write",
    "GET:/api/v1/gathering/operations": "eudr-iga:operations:read",
    "GET:/api/v1/gathering/operations/{operation_id}": "eudr-iga:operations:read",
    "POST:/api/v1/gathering/operations/{operation_id}/execute": "eudr-iga:operations:write",

    # External Database Queries (4 routes)
    "POST:/api/v1/gathering/external/query": "eudr-iga:external:query",
    "POST:/api/v1/gathering/external/batch-query": "eudr-iga:external:query",
    "GET:/api/v1/gathering/external/sources": "eudr-iga:external:status",
    "GET:/api/v1/gathering/external/sources/{source}/status": "eudr-iga:external:status",

    # Certification Verification (4 routes)
    "POST:/api/v1/gathering/certifications/verify": "eudr-iga:certifications:verify",
    "POST:/api/v1/gathering/certifications/batch-verify": "eudr-iga:certifications:verify",
    "GET:/api/v1/gathering/certifications/expiring": "eudr-iga:certifications:read",
    "GET:/api/v1/gathering/certifications/supplier/{supplier_id}": "eudr-iga:certifications:read",

    # Public Data (3 routes)
    "POST:/api/v1/gathering/public-data/harvest": "eudr-iga:public-data:harvest",
    "GET:/api/v1/gathering/public-data/freshness": "eudr-iga:public-data:read",
    "GET:/api/v1/gathering/public-data/{source}/latest": "eudr-iga:public-data:read",

    # Supplier Aggregation (4 routes)
    "POST:/api/v1/gathering/suppliers/aggregate": "eudr-iga:suppliers:aggregate",
    "GET:/api/v1/gathering/suppliers/{supplier_id}/profile": "eudr-iga:suppliers:read",
    "GET:/api/v1/gathering/suppliers/discrepancies": "eudr-iga:suppliers:read",
    "POST:/api/v1/gathering/suppliers/resolve/{supplier_id}": "eudr-iga:suppliers:resolve",

    # Completeness Validation (3 routes)
    "POST:/api/v1/gathering/completeness/validate": "eudr-iga:completeness:validate",
    "GET:/api/v1/gathering/completeness/{operation_id}/report": "eudr-iga:completeness:read",
    "GET:/api/v1/gathering/completeness/{operation_id}/gaps": "eudr-iga:completeness:read",

    # Information Packages (4 routes)
    "POST:/api/v1/gathering/packages/assemble": "eudr-iga:packages:assemble",
    "GET:/api/v1/gathering/packages/{package_id}": "eudr-iga:packages:read",
    "GET:/api/v1/gathering/packages/{package_id}/download": "eudr-iga:packages:read",
    "GET:/api/v1/gathering/packages/{package_id}/verify": "eudr-iga:packages:verify",

    # Normalization Audit (1 route)
    "GET:/api/v1/gathering/normalization/log": "eudr-iga:normalization:read",

    # Health (1 route)
    "GET:/api/v1/gathering/health": "eudr-iga:operations:read",

    # -----------------------------------------------------------------------
    # EUDR Due Diligence Orchestrator (AGENT-EUDR-026)
    # Regulation: EU 2023/1115 (EUDR) Articles 4, 8, 9, 10, 11, 12, 13, 31
    # 19 RBAC permissions across 7 categories, 30+ endpoints
    # -----------------------------------------------------------------------

    # Workflow Management (6 routes)
    "POST:/api/v1/eudr-ddo/workflows": "eudr-ddo:workflows:create",
    "GET:/api/v1/eudr-ddo/workflows": "eudr-ddo:workflows:read",
    "GET:/api/v1/eudr-ddo/workflows/{workflow_id}": "eudr-ddo:workflows:read",
    "DELETE:/api/v1/eudr-ddo/workflows/{workflow_id}": "eudr-ddo:workflows:delete",
    "POST:/api/v1/eudr-ddo/workflows/{workflow_id}/validate": "eudr-ddo:workflows:read",
    "POST:/api/v1/eudr-ddo/workflows/{workflow_id}/clone": "eudr-ddo:workflows:create",

    # Execution Control (5 routes)
    "POST:/api/v1/eudr-ddo/workflows/{workflow_id}/start": "eudr-ddo:workflows:manage",
    "POST:/api/v1/eudr-ddo/workflows/{workflow_id}/pause": "eudr-ddo:workflows:manage",
    "POST:/api/v1/eudr-ddo/workflows/{workflow_id}/resume": "eudr-ddo:workflows:manage",
    "POST:/api/v1/eudr-ddo/workflows/{workflow_id}/cancel": "eudr-ddo:workflows:manage",
    "POST:/api/v1/eudr-ddo/workflows/{workflow_id}/rollback": "eudr-ddo:checkpoints:rollback",

    # Status Monitoring (4 routes)
    "GET:/api/v1/eudr-ddo/workflows/{workflow_id}/status": "eudr-ddo:workflows:read",
    "GET:/api/v1/eudr-ddo/workflows/{workflow_id}/progress": "eudr-ddo:workflows:read",
    "GET:/api/v1/eudr-ddo/workflows/{workflow_id}/phase-status": "eudr-ddo:workflows:read",
    "GET:/api/v1/eudr-ddo/workflows/{workflow_id}/eta": "eudr-ddo:workflows:read",

    # Quality Gate Management (3 routes)
    "GET:/api/v1/eudr-ddo/workflows/{workflow_id}/gates": "eudr-ddo:gates:read",
    "POST:/api/v1/eudr-ddo/workflows/{workflow_id}/gates/{gate_id}/override": "eudr-ddo:gates:override",
    "GET:/api/v1/eudr-ddo/workflows/{workflow_id}/gates/{gate_id}/details": "eudr-ddo:gates:read",

    # Checkpoint Management (3 routes)
    "GET:/api/v1/eudr-ddo/workflows/{workflow_id}/checkpoints": "eudr-ddo:checkpoints:read",
    "POST:/api/v1/eudr-ddo/workflows/{workflow_id}/checkpoints": "eudr-ddo:checkpoints:rollback",
    "GET:/api/v1/eudr-ddo/checkpoints/{checkpoint_id}": "eudr-ddo:checkpoints:read",

    # Audit Trail (1 route)
    "GET:/api/v1/eudr-ddo/workflows/{workflow_id}/audit-trail": "eudr-ddo:audit-trail:read",

    # Template Management (4 routes)
    "GET:/api/v1/eudr-ddo/templates": "eudr-ddo:templates:read",
    "GET:/api/v1/eudr-ddo/templates/commodity/{commodity}": "eudr-ddo:templates:read",
    "POST:/api/v1/eudr-ddo/templates": "eudr-ddo:templates:manage",
    "GET:/api/v1/eudr-ddo/templates/{template_id}": "eudr-ddo:templates:read",

    # Package Management (4 routes)
    "POST:/api/v1/eudr-ddo/workflows/{workflow_id}/package": "eudr-ddo:packages:generate",
    "GET:/api/v1/eudr-ddo/packages/{package_id}": "eudr-ddo:packages:read",
    "GET:/api/v1/eudr-ddo/packages/{package_id}/download": "eudr-ddo:packages:download",
    "POST:/api/v1/eudr-ddo/packages/validate": "eudr-ddo:packages:read",

    # Batch Operations (2 routes)
    "POST:/api/v1/eudr-ddo/workflows/batch": "eudr-ddo:batch:manage",
    "GET:/api/v1/eudr-ddo/workflows/batch/{batch_id}": "eudr-ddo:batch:manage",

    # Circuit Breakers (2 routes)
    "GET:/api/v1/eudr-ddo/circuit-breakers": "eudr-ddo:circuit-breakers:read",
    "POST:/api/v1/eudr-ddo/circuit-breakers/{agent_id}/reset": "eudr-ddo:circuit-breakers:manage",

    # Dead Letter Queue (2 routes)
    "GET:/api/v1/eudr-ddo/dead-letter-queue": "eudr-ddo:dlq:read",
    "POST:/api/v1/eudr-ddo/dead-letter-queue/{dlq_id}/resolve": "eudr-ddo:dlq:manage",

    # Health & Monitoring (3 routes)
    "GET:/api/v1/eudr-ddo/health": "eudr-ddo:workflows:read",
    "GET:/api/v1/eudr-ddo/metrics": "eudr-ddo:workflows:read",
    "GET:/api/v1/eudr-ddo/version": "eudr-ddo:workflows:read",

    # -----------------------------------------------------------------------
    # AGENT-EUDR-027: Information Gathering Agent (GL-EUDR-IGA-027)
    # -----------------------------------------------------------------------

    # Information Gathering Operations (2 routes)
    "POST:/api/v1/eudr/information-gathering/gather": "eudr-iga:gather:execute",
    "GET:/api/v1/eudr/information-gathering/gather/{operation_id}": "eudr-iga:gather:read",

    # External Database Queries (1 route)
    "POST:/api/v1/eudr/information-gathering/query/{source}": "eudr-iga:query:execute",

    # Certificate Verification (2 routes)
    "POST:/api/v1/eudr/information-gathering/verify-certificate": "eudr-iga:certificate:verify",
    "POST:/api/v1/eudr/information-gathering/verify-certificates/batch": "eudr-iga:certificate:verify",

    # Public Data Harvest (1 route)
    "POST:/api/v1/eudr/information-gathering/harvest/{source}": "eudr-iga:harvest:execute",

    # Supplier Aggregation (1 route)
    "POST:/api/v1/eudr/information-gathering/aggregate-supplier/{supplier_id}": "eudr-iga:supplier:aggregate",

    # Completeness Validation (1 route)
    "POST:/api/v1/eudr/information-gathering/validate-completeness": "eudr-iga:completeness:validate",

    # Package Assembly (1 route)
    "POST:/api/v1/eudr/information-gathering/assemble-package": "eudr-iga:package:assemble",

    # Health (1 route)
    "GET:/api/v1/eudr/information-gathering/health": "eudr-iga:health:read",

    # -----------------------------------------------------------------------
    # AGENT-EUDR-028: Risk Assessment Engine (GL-EUDR-RAE-028)
    # -----------------------------------------------------------------------

    # Risk Assessment Operations (2 routes)
    "POST:/api/v1/eudr/risk-assessment-engine/assess": "eudr-rae:assess:execute",
    "GET:/api/v1/eudr/risk-assessment-engine/assess/{operation_id}": "eudr-rae:assess:read",

    # Composite Score (1 route)
    "POST:/api/v1/eudr/risk-assessment-engine/composite-score": "eudr-rae:composite:calculate",

    # Article 10 Criteria (1 route)
    "POST:/api/v1/eudr/risk-assessment-engine/evaluate-criteria": "eudr-rae:criteria:evaluate",

    # Country Benchmarks (2 routes)
    "GET:/api/v1/eudr/risk-assessment-engine/benchmarks/{country_code}": "eudr-rae:benchmarks:read",
    "POST:/api/v1/eudr/risk-assessment-engine/benchmarks/batch": "eudr-rae:benchmarks:read",

    # Classification (1 route)
    "POST:/api/v1/eudr/risk-assessment-engine/classify": "eudr-rae:classify:execute",

    # Simplified DD (1 route)
    "POST:/api/v1/eudr/risk-assessment-engine/simplified-dd/check": "eudr-rae:simplified-dd:check",

    # Override (1 route)
    "POST:/api/v1/eudr/risk-assessment-engine/override": "eudr-rae:override:apply",

    # Trend (1 route)
    "GET:/api/v1/eudr/risk-assessment-engine/trend/{operator_id}/{commodity}": "eudr-rae:trend:read",

    # Batch (1 route)
    "POST:/api/v1/eudr/risk-assessment-engine/assess/batch": "eudr-rae:assess:batch",

    # Health (1 route)
    "GET:/api/v1/eudr/risk-assessment-engine/health": "eudr-rae:health:read",

    # -----------------------------------------------------------------------
    # AGENT-EUDR-029: Mitigation Measure Designer (GL-EUDR-MMD-029)
    # -----------------------------------------------------------------------

    # Strategy Design (3 routes)
    "POST:/api/v1/eudr/mitigation-measure-designer/design-strategy": "eudr-mmd:strategies:write",
    "GET:/api/v1/eudr/mitigation-measure-designer/strategies/{strategy_id}": "eudr-mmd:strategies:read",
    "GET:/api/v1/eudr/mitigation-measure-designer/strategies": "eudr-mmd:strategies:read",

    # Measure Lifecycle (3 routes)
    "POST:/api/v1/eudr/mitigation-measure-designer/measures/{measure_id}/approve": "eudr-mmd:measures:approve",
    "POST:/api/v1/eudr/mitigation-measure-designer/measures/{measure_id}/start": "eudr-mmd:measures:write",
    "POST:/api/v1/eudr/mitigation-measure-designer/measures/{measure_id}/complete": "eudr-mmd:measures:write",

    # Verification (1 route)
    "POST:/api/v1/eudr/mitigation-measure-designer/verify/{strategy_id}": "eudr-mmd:verification:execute",

    # Templates (2 routes)
    "GET:/api/v1/eudr/mitigation-measure-designer/templates": "eudr-mmd:templates:read",
    "GET:/api/v1/eudr/mitigation-measure-designer/templates/{template_id}": "eudr-mmd:templates:read",

    # Reports (1 route)
    "GET:/api/v1/eudr/mitigation-measure-designer/generate-report/{strategy_id}": "eudr-mmd:reports:generate",

    # Workflows (1 route)
    "GET:/api/v1/eudr/mitigation-measure-designer/workflows/{workflow_id}/status": "eudr-mmd:workflows:read",

    # Health (1 route - public)
    "GET:/api/v1/eudr/mitigation-measure-designer/health": None,

    # ---------------------------------------------------------------------------
    # AGENT-EUDR-030: Documentation Generator (12 routes)
    # ---------------------------------------------------------------------------

    # DDS Generation (3 routes)
    "POST:/api/v1/eudr/documentation-generator/generate-dds": "eudr-dgn:dds:generate",
    "GET:/api/v1/eudr/documentation-generator/dds/{dds_id}": "eudr-dgn:dds:read",
    "GET:/api/v1/eudr/documentation-generator/dds": "eudr-dgn:dds:list",

    # Article 9 Assembly (1 route)
    "POST:/api/v1/eudr/documentation-generator/assemble-article9/{operator_id}": "eudr-dgn:article9:assemble",

    # Risk & Mitigation Documentation (2 routes)
    "POST:/api/v1/eudr/documentation-generator/document-risk/{assessment_id}": "eudr-dgn:risk:document",
    "POST:/api/v1/eudr/documentation-generator/document-mitigation/{strategy_id}": "eudr-dgn:mitigation:document",

    # Compliance Package (1 route)
    "POST:/api/v1/eudr/documentation-generator/build-package/{dds_id}": "eudr-dgn:package:build",

    # Submission (2 routes)
    "POST:/api/v1/eudr/documentation-generator/submit/{dds_id}": "eudr-dgn:submission:submit",
    "GET:/api/v1/eudr/documentation-generator/submissions/{submission_id}/status": "eudr-dgn:submission:read",

    # Validation (1 route)
    "POST:/api/v1/eudr/documentation-generator/validate/{dds_id}": "eudr-dgn:validation:execute",

    # Version History (1 route)
    "GET:/api/v1/eudr/documentation-generator/versions/{document_id}": "eudr-dgn:versions:read",

    # Health (1 route - public)
    "GET:/api/v1/eudr/documentation-generator/health": None,

    # =========================================================================
    # AGENT-EUDR-031: Stakeholder Engagement Tool (33 routes)
    # =========================================================================

    # Stakeholder Mapping (3 routes)
    "POST:/api/v1/eudr/stakeholder-engagement/map-stakeholder": "eudr-set:stakeholders:write",
    "GET:/api/v1/eudr/stakeholder-engagement/stakeholders": "eudr-set:stakeholders:read",
    "GET:/api/v1/eudr/stakeholder-engagement/stakeholders/{stakeholder_id}": "eudr-set:stakeholders:read",

    # FPIC Workflow (5 routes)
    "POST:/api/v1/eudr/stakeholder-engagement/initiate-fpic": "eudr-set:fpic:write",
    "POST:/api/v1/eudr/stakeholder-engagement/fpic/{fpic_id}/advance-stage": "eudr-set:fpic:write",
    "POST:/api/v1/eudr/stakeholder-engagement/fpic/{fpic_id}/record-consent": "eudr-set:fpic:write",
    "GET:/api/v1/eudr/stakeholder-engagement/fpic/{fpic_id}": "eudr-set:fpic:read",
    "GET:/api/v1/eudr/stakeholder-engagement/fpic": "eudr-set:fpic:read",

    # Grievance Mechanism (7 routes)
    "POST:/api/v1/eudr/stakeholder-engagement/submit-grievance": "eudr-set:grievances:write",
    "POST:/api/v1/eudr/stakeholder-engagement/grievances/{grievance_id}/triage": "eudr-set:grievances:triage",
    "POST:/api/v1/eudr/stakeholder-engagement/grievances/{grievance_id}/investigate": "eudr-set:grievances:investigate",
    "POST:/api/v1/eudr/stakeholder-engagement/grievances/{grievance_id}/resolve": "eudr-set:grievances:resolve",
    "POST:/api/v1/eudr/stakeholder-engagement/grievances/{grievance_id}/appeal": "eudr-set:grievances:appeal",
    "GET:/api/v1/eudr/stakeholder-engagement/grievances/{grievance_id}": "eudr-set:grievances:read",
    "GET:/api/v1/eudr/stakeholder-engagement/grievances": "eudr-set:grievances:read",

    # Consultation Records (7 routes)
    "POST:/api/v1/eudr/stakeholder-engagement/create-consultation": "eudr-set:consultations:write",
    "POST:/api/v1/eudr/stakeholder-engagement/consultations/{consultation_id}/participants": "eudr-set:consultations:write",
    "POST:/api/v1/eudr/stakeholder-engagement/consultations/{consultation_id}/outcomes": "eudr-set:consultations:write",
    "POST:/api/v1/eudr/stakeholder-engagement/consultations/{consultation_id}/evidence": "eudr-set:consultations:write",
    "POST:/api/v1/eudr/stakeholder-engagement/consultations/{consultation_id}/finalize": "eudr-set:consultations:finalize",
    "GET:/api/v1/eudr/stakeholder-engagement/consultations/{consultation_id}": "eudr-set:consultations:read",
    "GET:/api/v1/eudr/stakeholder-engagement/consultations": "eudr-set:consultations:read",

    # Communications (5 routes)
    "POST:/api/v1/eudr/stakeholder-engagement/send-communication": "eudr-set:communications:write",
    "POST:/api/v1/eudr/stakeholder-engagement/schedule-communication": "eudr-set:communications:write",
    "POST:/api/v1/eudr/stakeholder-engagement/send-campaign": "eudr-set:communications:campaign",
    "GET:/api/v1/eudr/stakeholder-engagement/communications/{communication_id}": "eudr-set:communications:read",
    "GET:/api/v1/eudr/stakeholder-engagement/communications": "eudr-set:communications:read",

    # Engagement Assessments (2 routes)
    "POST:/api/v1/eudr/stakeholder-engagement/assess-engagement/{stakeholder_id}": "eudr-set:assessments:write",
    "GET:/api/v1/eudr/stakeholder-engagement/assessments/{assessment_id}": "eudr-set:assessments:read",

    # Compliance Reports (3 routes)
    "POST:/api/v1/eudr/stakeholder-engagement/generate-report": "eudr-set:reports:generate",
    "GET:/api/v1/eudr/stakeholder-engagement/reports/{report_id}": "eudr-set:reports:read",
    "GET:/api/v1/eudr/stakeholder-engagement/reports/{report_id}/export": "eudr-set:reports:export",

    # Health (1 route - public)
    "GET:/api/v1/eudr/stakeholder-engagement/health": None,

    # ===========================================================================
    # AGENT-EUDR-032: Grievance Mechanism Manager (29 routes)
    # ===========================================================================

    # Analytics (3 routes)
    "POST:/api/v1/eudr/grievance-mechanism-manager/analyze-patterns": "eudr-gmm:analytics:run",
    "GET:/api/v1/eudr/grievance-mechanism-manager/analytics": "eudr-gmm:analytics:read",
    "GET:/api/v1/eudr/grievance-mechanism-manager/analytics/{analytics_id}": "eudr-gmm:analytics:read",

    # Root Cause (3 routes)
    "POST:/api/v1/eudr/grievance-mechanism-manager/analyze-root-cause": "eudr-gmm:root-cause:analyze",
    "GET:/api/v1/eudr/grievance-mechanism-manager/root-causes": "eudr-gmm:root-cause:read",
    "GET:/api/v1/eudr/grievance-mechanism-manager/root-causes/{root_cause_id}": "eudr-gmm:root-cause:read",

    # Mediation (7 routes)
    "POST:/api/v1/eudr/grievance-mechanism-manager/initiate-mediation": "eudr-gmm:mediation:initiate",
    "POST:/api/v1/eudr/grievance-mechanism-manager/mediations/{mediation_id}/advance": "eudr-gmm:mediation:advance",
    "POST:/api/v1/eudr/grievance-mechanism-manager/mediations/{mediation_id}/sessions": "eudr-gmm:mediation:write",
    "POST:/api/v1/eudr/grievance-mechanism-manager/mediations/{mediation_id}/agreements": "eudr-gmm:mediation:write",
    "POST:/api/v1/eudr/grievance-mechanism-manager/mediations/{mediation_id}/settlement": "eudr-gmm:mediation:settle",
    "GET:/api/v1/eudr/grievance-mechanism-manager/mediations": "eudr-gmm:mediation:read",
    "GET:/api/v1/eudr/grievance-mechanism-manager/mediations/{mediation_id}": "eudr-gmm:mediation:read",

    # Remediation (6 routes)
    "POST:/api/v1/eudr/grievance-mechanism-manager/create-remediation": "eudr-gmm:remediation:create",
    "POST:/api/v1/eudr/grievance-mechanism-manager/remediations/{remediation_id}/progress": "eudr-gmm:remediation:update",
    "POST:/api/v1/eudr/grievance-mechanism-manager/remediations/{remediation_id}/verify": "eudr-gmm:remediation:verify",
    "POST:/api/v1/eudr/grievance-mechanism-manager/remediations/{remediation_id}/satisfaction": "eudr-gmm:remediation:update",
    "GET:/api/v1/eudr/grievance-mechanism-manager/remediations": "eudr-gmm:remediation:read",
    "GET:/api/v1/eudr/grievance-mechanism-manager/remediations/{remediation_id}": "eudr-gmm:remediation:read",

    # Risk Scoring (3 routes)
    "POST:/api/v1/eudr/grievance-mechanism-manager/compute-risk-score": "eudr-gmm:risk-scoring:compute",
    "GET:/api/v1/eudr/grievance-mechanism-manager/risk-scores": "eudr-gmm:risk-scoring:read",
    "GET:/api/v1/eudr/grievance-mechanism-manager/risk-scores/{risk_score_id}": "eudr-gmm:risk-scoring:read",

    # Collective Grievances (4 routes)
    "POST:/api/v1/eudr/grievance-mechanism-manager/create-collective": "eudr-gmm:collective:create",
    "POST:/api/v1/eudr/grievance-mechanism-manager/collectives/{collective_id}/demands": "eudr-gmm:collective:update",
    "POST:/api/v1/eudr/grievance-mechanism-manager/collectives/{collective_id}/status": "eudr-gmm:collective:update",
    "GET:/api/v1/eudr/grievance-mechanism-manager/collectives": "eudr-gmm:collective:read",
    "GET:/api/v1/eudr/grievance-mechanism-manager/collectives/{collective_id}": "eudr-gmm:collective:read",

    # Regulatory Reporting (3 routes)
    "POST:/api/v1/eudr/grievance-mechanism-manager/generate-report": "eudr-gmm:reports:generate",
    "GET:/api/v1/eudr/grievance-mechanism-manager/reports": "eudr-gmm:reports:read",
    "GET:/api/v1/eudr/grievance-mechanism-manager/reports/{report_id}": "eudr-gmm:reports:read",

    # Health (1 route - public)
    "GET:/api/v1/eudr/grievance-mechanism-manager/health": None,

    # ===========================================================================
    # AGENT-EUDR-033: Continuous Monitoring Agent (30 routes)
    # ===========================================================================

    # Monitoring Jobs (7 routes)
    "POST:/api/v1/eudr/continuous-monitoring/jobs": "eudr-cm:jobs:create",
    "GET:/api/v1/eudr/continuous-monitoring/jobs": "eudr-cm:jobs:read",
    "GET:/api/v1/eudr/continuous-monitoring/jobs/{job_id}": "eudr-cm:jobs:read",
    "PUT:/api/v1/eudr/continuous-monitoring/jobs/{job_id}": "eudr-cm:jobs:update",
    "DELETE:/api/v1/eudr/continuous-monitoring/jobs/{job_id}": "eudr-cm:jobs:delete",
    "POST:/api/v1/eudr/continuous-monitoring/jobs/{job_id}/execute": "eudr-cm:jobs:execute",
    "POST:/api/v1/eudr/continuous-monitoring/jobs/{job_id}/pause": "eudr-cm:jobs:update",

    # Alerts (6 routes)
    "GET:/api/v1/eudr/continuous-monitoring/alerts": "eudr-cm:alerts:read",
    "GET:/api/v1/eudr/continuous-monitoring/alerts/{alert_id}": "eudr-cm:alerts:read",
    "POST:/api/v1/eudr/continuous-monitoring/alerts/{alert_id}/acknowledge": "eudr-cm:alerts:acknowledge",
    "POST:/api/v1/eudr/continuous-monitoring/alerts/{alert_id}/resolve": "eudr-cm:alerts:resolve",
    "DELETE:/api/v1/eudr/continuous-monitoring/alerts/{alert_id}": "eudr-cm:alerts:delete",
    "GET:/api/v1/eudr/continuous-monitoring/alerts/summary": "eudr-cm:alerts:read",

    # Compliance Scanning (4 routes)
    "POST:/api/v1/eudr/continuous-monitoring/compliance/scan": "eudr-cm:compliance:scan",
    "GET:/api/v1/eudr/continuous-monitoring/compliance": "eudr-cm:compliance:read",
    "GET:/api/v1/eudr/continuous-monitoring/compliance/{scan_id}": "eudr-cm:compliance:read",
    "GET:/api/v1/eudr/continuous-monitoring/compliance/{scan_id}/export": "eudr-cm:compliance:export",

    # Change Detection (3 routes)
    "GET:/api/v1/eudr/continuous-monitoring/changes": "eudr-cm:changes:read",
    "GET:/api/v1/eudr/continuous-monitoring/changes/{change_id}": "eudr-cm:changes:read",
    "POST:/api/v1/eudr/continuous-monitoring/changes/{change_id}/analyze": "eudr-cm:changes:analyze",

    # Risk Monitoring (3 routes)
    "POST:/api/v1/eudr/continuous-monitoring/risk/monitor": "eudr-cm:risk:monitor",
    "GET:/api/v1/eudr/continuous-monitoring/risk": "eudr-cm:risk:read",
    "GET:/api/v1/eudr/continuous-monitoring/risk/{assessment_id}": "eudr-cm:risk:read",

    # Data Freshness (3 routes)
    "POST:/api/v1/eudr/continuous-monitoring/freshness/validate": "eudr-cm:freshness:validate",
    "GET:/api/v1/eudr/continuous-monitoring/freshness": "eudr-cm:freshness:read",
    "GET:/api/v1/eudr/continuous-monitoring/freshness/{source_id}": "eudr-cm:freshness:read",

    # Regulatory Tracking (2 routes)
    "GET:/api/v1/eudr/continuous-monitoring/regulatory": "eudr-cm:regulatory:read",
    "POST:/api/v1/eudr/continuous-monitoring/regulatory/import": "eudr-cm:regulatory:import",

    # System Configuration (2 routes)
    "GET:/api/v1/eudr/continuous-monitoring/config": "eudr-cm:config:read",
    "PUT:/api/v1/eudr/continuous-monitoring/config": "eudr-cm:config:update",

    # Health (1 route - public)
    "GET:/api/v1/eudr/continuous-monitoring/health": None,

    # ===========================================================================
    # AGENT-EUDR-034: Annual Review Scheduler (30 routes)
    # ===========================================================================

    # Review Cycles (5 routes)
    "POST:/api/v1/eudr/annual-review-scheduler/cycles": "eudr-ars:cycles:create",
    "GET:/api/v1/eudr/annual-review-scheduler/cycles": "eudr-ars:cycles:read",
    "GET:/api/v1/eudr/annual-review-scheduler/cycles/{cycle_id}": "eudr-ars:cycles:read",
    "PUT:/api/v1/eudr/annual-review-scheduler/cycles/{cycle_id}": "eudr-ars:cycles:update",
    "DELETE:/api/v1/eudr/annual-review-scheduler/cycles/{cycle_id}": "eudr-ars:cycles:delete",
    "POST:/api/v1/eudr/annual-review-scheduler/cycles/{cycle_id}/schedule": "eudr-ars:cycles:schedule",

    # Tasks (6 routes)
    "POST:/api/v1/eudr/annual-review-scheduler/tasks": "eudr-ars:tasks:create",
    "GET:/api/v1/eudr/annual-review-scheduler/tasks": "eudr-ars:tasks:read",
    "GET:/api/v1/eudr/annual-review-scheduler/tasks/{task_id}": "eudr-ars:tasks:read",
    "PUT:/api/v1/eudr/annual-review-scheduler/tasks/{task_id}": "eudr-ars:tasks:update",
    "DELETE:/api/v1/eudr/annual-review-scheduler/tasks/{task_id}": "eudr-ars:tasks:delete",
    "POST:/api/v1/eudr/annual-review-scheduler/tasks/{task_id}/assign": "eudr-ars:tasks:assign",
    "POST:/api/v1/eudr/annual-review-scheduler/tasks/{task_id}/complete": "eudr-ars:tasks:complete",

    # Deadlines (4 routes)
    "POST:/api/v1/eudr/annual-review-scheduler/deadlines": "eudr-ars:deadlines:create",
    "GET:/api/v1/eudr/annual-review-scheduler/deadlines": "eudr-ars:deadlines:read",
    "PUT:/api/v1/eudr/annual-review-scheduler/deadlines/{deadline_id}": "eudr-ars:deadlines:update",
    "POST:/api/v1/eudr/annual-review-scheduler/deadlines/{deadline_id}/submit": "eudr-ars:deadlines:submit",

    # Checklists (4 routes)
    "POST:/api/v1/eudr/annual-review-scheduler/checklists/generate": "eudr-ars:checklists:generate",
    "GET:/api/v1/eudr/annual-review-scheduler/checklists": "eudr-ars:checklists:read",
    "GET:/api/v1/eudr/annual-review-scheduler/checklists/{checklist_id}": "eudr-ars:checklists:read",
    "PUT:/api/v1/eudr/annual-review-scheduler/checklists/{checklist_id}": "eudr-ars:checklists:update",
    "POST:/api/v1/eudr/annual-review-scheduler/checklists/{checklist_id}/complete": "eudr-ars:checklists:complete",

    # Year-over-Year Comparison (3 routes)
    "POST:/api/v1/eudr/annual-review-scheduler/comparison/run": "eudr-ars:comparison:run",
    "GET:/api/v1/eudr/annual-review-scheduler/comparison": "eudr-ars:comparison:read",
    "GET:/api/v1/eudr/annual-review-scheduler/comparison/{comparison_id}/export": "eudr-ars:comparison:export",

    # Calendar Integration (5 routes)
    "GET:/api/v1/eudr/annual-review-scheduler/calendar": "eudr-ars:calendar:read",
    "POST:/api/v1/eudr/annual-review-scheduler/calendar/events": "eudr-ars:calendar:create",
    "PUT:/api/v1/eudr/annual-review-scheduler/calendar/events/{event_id}": "eudr-ars:calendar:update",
    "DELETE:/api/v1/eudr/annual-review-scheduler/calendar/events/{event_id}": "eudr-ars:calendar:delete",
    "POST:/api/v1/eudr/annual-review-scheduler/calendar/sync": "eudr-ars:calendar:sync",

    # Notifications (3 routes)
    "POST:/api/v1/eudr/annual-review-scheduler/notifications/send": "eudr-ars:notifications:send",
    "GET:/api/v1/eudr/annual-review-scheduler/notifications": "eudr-ars:notifications:read",
    "POST:/api/v1/eudr/annual-review-scheduler/notifications/{notification_id}/acknowledge": "eudr-ars:notifications:acknowledge",

    # System Configuration (2 routes)
    "GET:/api/v1/eudr/annual-review-scheduler/config": "eudr-ars:config:read",
    "PUT:/api/v1/eudr/annual-review-scheduler/config": "eudr-ars:config:update",

    # Health (1 route - public)
    "GET:/api/v1/eudr/annual-review-scheduler/health": None,

    # ===========================================================================
    # AGENT-EUDR-035: Improvement Plan Creator (30 routes)
    # ===========================================================================

    # Plans (5 routes)
    "POST:/api/v1/eudr/improvement-plan-creator/plans": "eudr-ipc:plans:create",
    "GET:/api/v1/eudr/improvement-plan-creator/plans": "eudr-ipc:plans:read",
    "GET:/api/v1/eudr/improvement-plan-creator/plans/{plan_id}": "eudr-ipc:plans:read",
    "PUT:/api/v1/eudr/improvement-plan-creator/plans/{plan_id}": "eudr-ipc:plans:update",
    "DELETE:/api/v1/eudr/improvement-plan-creator/plans/{plan_id}": "eudr-ipc:plans:delete",
    "POST:/api/v1/eudr/improvement-plan-creator/plans/{plan_id}/publish": "eudr-ipc:plans:publish",

    # Findings (3 routes)
    "GET:/api/v1/eudr/improvement-plan-creator/findings": "eudr-ipc:findings:read",
    "GET:/api/v1/eudr/improvement-plan-creator/findings/aggregate": "eudr-ipc:findings:aggregate",
    "POST:/api/v1/eudr/improvement-plan-creator/findings/analyze": "eudr-ipc:findings:analyze",

    # Actions (6 routes)
    "POST:/api/v1/eudr/improvement-plan-creator/actions": "eudr-ipc:actions:create",
    "GET:/api/v1/eudr/improvement-plan-creator/actions": "eudr-ipc:actions:read",
    "GET:/api/v1/eudr/improvement-plan-creator/actions/{action_id}": "eudr-ipc:actions:read",
    "PUT:/api/v1/eudr/improvement-plan-creator/actions/{action_id}": "eudr-ipc:actions:update",
    "DELETE:/api/v1/eudr/improvement-plan-creator/actions/{action_id}": "eudr-ipc:actions:delete",
    "POST:/api/v1/eudr/improvement-plan-creator/actions/{action_id}/assign": "eudr-ipc:actions:assign",
    "POST:/api/v1/eudr/improvement-plan-creator/actions/{action_id}/complete": "eudr-ipc:actions:complete",

    # Root Causes (2 routes)
    "POST:/api/v1/eudr/improvement-plan-creator/root-causes/analyze": "eudr-ipc:root-causes:analyze",
    "GET:/api/v1/eudr/improvement-plan-creator/root-causes": "eudr-ipc:root-causes:read",

    # Prioritization (3 routes)
    "POST:/api/v1/eudr/improvement-plan-creator/prioritization/compute": "eudr-ipc:prioritization:compute",
    "GET:/api/v1/eudr/improvement-plan-creator/prioritization": "eudr-ipc:prioritization:read",
    "PUT:/api/v1/eudr/improvement-plan-creator/prioritization/{plan_id}": "eudr-ipc:prioritization:update",

    # Progress (3 routes)
    "GET:/api/v1/eudr/improvement-plan-creator/progress": "eudr-ipc:progress:read",
    "PUT:/api/v1/eudr/improvement-plan-creator/progress/{plan_id}": "eudr-ipc:progress:update",
    "GET:/api/v1/eudr/improvement-plan-creator/progress/{plan_id}/export": "eudr-ipc:progress:export",

    # Stakeholders (3 routes)
    "POST:/api/v1/eudr/improvement-plan-creator/stakeholders/assign": "eudr-ipc:stakeholders:assign",
    "GET:/api/v1/eudr/improvement-plan-creator/stakeholders": "eudr-ipc:stakeholders:read",
    "POST:/api/v1/eudr/improvement-plan-creator/stakeholders/notify": "eudr-ipc:stakeholders:notify",

    # Config (2 routes)
    "GET:/api/v1/eudr/improvement-plan-creator/config": "eudr-ipc:config:read",
    "PUT:/api/v1/eudr/improvement-plan-creator/config": "eudr-ipc:config:update",

    # Health (1 route - public)
    "GET:/api/v1/eudr/improvement-plan-creator/health": None,

    # ===========================================================================
    # AGENT-EUDR-036: EU Information System Interface (30 routes)
    # ===========================================================================

    # DDS Submissions (7 routes)
    "POST:/api/v1/eudr/eu-information-system/submissions": "eudr-euis:submissions:create",
    "GET:/api/v1/eudr/eu-information-system/submissions": "eudr-euis:submissions:read",
    "GET:/api/v1/eudr/eu-information-system/submissions/{submission_id}": "eudr-euis:submissions:read",
    "PUT:/api/v1/eudr/eu-information-system/submissions/{submission_id}": "eudr-euis:submissions:update",
    "DELETE:/api/v1/eudr/eu-information-system/submissions/{submission_id}": "eudr-euis:submissions:delete",
    "POST:/api/v1/eudr/eu-information-system/submissions/{submission_id}/submit": "eudr-euis:submissions:submit",
    "POST:/api/v1/eudr/eu-information-system/submissions/{submission_id}/validate": "eudr-euis:submissions:validate",

    # Operators (5 routes)
    "POST:/api/v1/eudr/eu-information-system/operators": "eudr-euis:operators:create",
    "GET:/api/v1/eudr/eu-information-system/operators": "eudr-euis:operators:read",
    "GET:/api/v1/eudr/eu-information-system/operators/{operator_id}": "eudr-euis:operators:read",
    "PUT:/api/v1/eudr/eu-information-system/operators/{operator_id}": "eudr-euis:operators:update",
    "POST:/api/v1/eudr/eu-information-system/operators/{operator_id}/verify-eori": "eudr-euis:operators:verify",

    # Geolocations (3 routes)
    "POST:/api/v1/eudr/eu-information-system/geolocations/format": "eudr-euis:geolocations:format",
    "POST:/api/v1/eudr/eu-information-system/geolocations/validate": "eudr-euis:geolocations:validate",
    "GET:/api/v1/eudr/eu-information-system/geolocations": "eudr-euis:geolocations:read",

    # Packages (4 routes)
    "POST:/api/v1/eudr/eu-information-system/packages/assemble": "eudr-euis:packages:assemble",
    "GET:/api/v1/eudr/eu-information-system/packages": "eudr-euis:packages:read",
    "GET:/api/v1/eudr/eu-information-system/packages/{package_id}": "eudr-euis:packages:read",
    "GET:/api/v1/eudr/eu-information-system/packages/{package_id}/download": "eudr-euis:packages:download",

    # Statuses (3 routes)
    "GET:/api/v1/eudr/eu-information-system/statuses": "eudr-euis:statuses:read",
    "GET:/api/v1/eudr/eu-information-system/statuses/{submission_id}": "eudr-euis:statuses:read",
    "POST:/api/v1/eudr/eu-information-system/statuses/{submission_id}/poll": "eudr-euis:statuses:poll",

    # Audits (3 routes)
    "GET:/api/v1/eudr/eu-information-system/audits": "eudr-euis:audits:read",
    "GET:/api/v1/eudr/eu-information-system/audits/{audit_id}": "eudr-euis:audits:read",
    "GET:/api/v1/eudr/eu-information-system/audits/export": "eudr-euis:audits:export",

    # Batch (2 routes)
    "POST:/api/v1/eudr/eu-information-system/batch/submit": "eudr-euis:batch:submit",
    "GET:/api/v1/eudr/eu-information-system/batch/{batch_id}": "eudr-euis:batch:read",

    # Config (2 routes)
    "GET:/api/v1/eudr/eu-information-system/config": "eudr-euis:config:read",
    "PUT:/api/v1/eudr/eu-information-system/config": "eudr-euis:config:update",

    # Health (1 route - public)
    "GET:/api/v1/eudr/eu-information-system/health": None,

    # ===========================================================================
    # AGENT-EUDR-037: Due Diligence Statement Creator (32 routes)
    # ===========================================================================

    # Statements CRUD (5 routes)
    "POST:/api/v1/eudr/due-diligence-statement-creator/statements": "eudr-ddsc:create",
    "GET:/api/v1/eudr/due-diligence-statement-creator/statements": "eudr-ddsc:read",
    "GET:/api/v1/eudr/due-diligence-statement-creator/statements/{statement_id}": "eudr-ddsc:read",
    "PUT:/api/v1/eudr/due-diligence-statement-creator/statements/{statement_id}": "eudr-ddsc:update",
    "DELETE:/api/v1/eudr/due-diligence-statement-creator/statements/{statement_id}": "eudr-ddsc:delete",

    # Validation (3 routes)
    "POST:/api/v1/eudr/due-diligence-statement-creator/statements/{statement_id}/validate": "eudr-ddsc:validate",
    "POST:/api/v1/eudr/due-diligence-statement-creator/statements/{statement_id}/validate-geolocation": "eudr-ddsc:validate-geolocation",
    "POST:/api/v1/eudr/due-diligence-statement-creator/statements/{statement_id}/validate-compliance": "eudr-ddsc:validate-compliance",

    # Submission (3 routes)
    "POST:/api/v1/eudr/due-diligence-statement-creator/statements/{statement_id}/submit": "eudr-ddsc:submit",
    "POST:/api/v1/eudr/due-diligence-statement-creator/statements/submit-bulk": "eudr-ddsc:submit-bulk",
    "POST:/api/v1/eudr/due-diligence-statement-creator/statements/{statement_id}/withdraw": "eudr-ddsc:withdraw",

    # Amendment (3 routes)
    "POST:/api/v1/eudr/due-diligence-statement-creator/statements/{statement_id}/amend": "eudr-ddsc:amend",
    "POST:/api/v1/eudr/due-diligence-statement-creator/statements/{statement_id}/amend-approve": "eudr-ddsc:amend-approve",
    "GET:/api/v1/eudr/due-diligence-statement-creator/statements/{statement_id}/version-history": "eudr-ddsc:version-history",

    # Document Packages (3 routes)
    "POST:/api/v1/eudr/due-diligence-statement-creator/statements/{statement_id}/packages": "eudr-ddsc:package-add",
    "DELETE:/api/v1/eudr/due-diligence-statement-creator/statements/{statement_id}/packages/{package_id}": "eudr-ddsc:package-remove",
    "GET:/api/v1/eudr/due-diligence-statement-creator/statements/{statement_id}/packages": "eudr-ddsc:package-list",

    # Signatures (3 routes)
    "POST:/api/v1/eudr/due-diligence-statement-creator/statements/{statement_id}/sign": "eudr-ddsc:sign",
    "GET:/api/v1/eudr/due-diligence-statement-creator/statements/{statement_id}/sign-verify": "eudr-ddsc:sign-verify",
    "POST:/api/v1/eudr/due-diligence-statement-creator/statements/{statement_id}/sign-approve": "eudr-ddsc:sign-approve",

    # Risk Integration (2 routes)
    "GET:/api/v1/eudr/due-diligence-statement-creator/statements/{statement_id}/risk-integrate": "eudr-ddsc:risk-integrate",
    "POST:/api/v1/eudr/due-diligence-statement-creator/statements/{statement_id}/risk-override": "eudr-ddsc:risk-override",

    # Supply Chain (2 routes)
    "GET:/api/v1/eudr/due-diligence-statement-creator/statements/{statement_id}/supply-chain": "eudr-ddsc:supply-chain-view",
    "POST:/api/v1/eudr/due-diligence-statement-creator/statements/{statement_id}/supply-chain-verify": "eudr-ddsc:supply-chain-verify",

    # Reports (2 routes)
    "POST:/api/v1/eudr/due-diligence-statement-creator/reports/generate": "eudr-ddsc:report-generate",
    "GET:/api/v1/eudr/due-diligence-statement-creator/reports/{report_id}/download": "eudr-ddsc:report-download",

    # Admin (2 routes)
    "GET:/api/v1/eudr/due-diligence-statement-creator/admin": "eudr-ddsc:admin",
    "POST:/api/v1/eudr/due-diligence-statement-creator/admin/bulk-operations": "eudr-ddsc:admin-bulk-operations",

    # Config (2 routes)
    "GET:/api/v1/eudr/due-diligence-statement-creator/config": "eudr-ddsc:config:read",
    "PUT:/api/v1/eudr/due-diligence-statement-creator/config": "eudr-ddsc:config:update",

    # Health (1 route - public)
    "GET:/api/v1/eudr/due-diligence-statement-creator/health": None,

    # ===========================================================================
    # AGENT-EUDR-038: Reference Number Generator (30 routes)
    # ===========================================================================

    # Generation (3 routes)
    "POST:/api/v1/eudr/reference-number-generator/references": "eudr-rng:generate",
    "POST:/api/v1/eudr/reference-number-generator/references/batch": "eudr-rng:generate-batch",
    "POST:/api/v1/eudr/reference-number-generator/references/bulk": "eudr-rng:generate-bulk",

    # Validation (3 routes)
    "POST:/api/v1/eudr/reference-number-generator/references/{reference_id}/validate": "eudr-rng:validate",
    "POST:/api/v1/eudr/reference-number-generator/references/{reference_id}/validate-format": "eudr-rng:validate-format",
    "POST:/api/v1/eudr/reference-number-generator/references/{reference_id}/validate-checksum": "eudr-rng:validate-checksum",

    # Read (3 routes)
    "GET:/api/v1/eudr/reference-number-generator/references": "eudr-rng:read",
    "GET:/api/v1/eudr/reference-number-generator/references/{reference_id}": "eudr-rng:read",
    "GET:/api/v1/eudr/reference-number-generator/sequences": "eudr-rng:read-sequences",

    # Lifecycle (3 routes)
    "POST:/api/v1/eudr/reference-number-generator/references/{reference_id}/revoke": "eudr-rng:revoke",
    "POST:/api/v1/eudr/reference-number-generator/references/{reference_id}/transfer": "eudr-rng:transfer",
    "POST:/api/v1/eudr/reference-number-generator/references/{reference_id}/extend-expiry": "eudr-rng:extend-expiry",

    # Batch (3 routes)
    "POST:/api/v1/eudr/reference-number-generator/batches": "eudr-rng:batch-create",
    "GET:/api/v1/eudr/reference-number-generator/batches/{batch_id}": "eudr-rng:batch-status",
    "POST:/api/v1/eudr/reference-number-generator/batches/{batch_id}/cancel": "eudr-rng:batch-cancel",

    # Sequences (3 routes)
    "GET:/api/v1/eudr/reference-number-generator/sequences/{sequence_id}": "eudr-rng:sequence-view",
    "POST:/api/v1/eudr/reference-number-generator/sequences/{sequence_id}/reset": "eudr-rng:sequence-reset",
    "POST:/api/v1/eudr/reference-number-generator/sequences/{sequence_id}/reserve": "eudr-rng:sequence-reserve",

    # Format Rules (2 routes)
    "GET:/api/v1/eudr/reference-number-generator/format-rules": "eudr-rng:format-view",
    "PUT:/api/v1/eudr/reference-number-generator/format-rules": "eudr-rng:format-update",

    # Verification (2 routes)
    "POST:/api/v1/eudr/reference-number-generator/references/{reference_id}/verify": "eudr-rng:verify",
    "POST:/api/v1/eudr/reference-number-generator/references/verify-bulk": "eudr-rng:verify-bulk",

    # Reports (2 routes)
    "POST:/api/v1/eudr/reference-number-generator/reports/generate": "eudr-rng:report-generate",
    "GET:/api/v1/eudr/reference-number-generator/reports/{report_id}/download": "eudr-rng:report-download",

    # Admin (3 routes)
    "GET:/api/v1/eudr/reference-number-generator/admin": "eudr-rng:admin",
    "POST:/api/v1/eudr/reference-number-generator/admin/override-sequence": "eudr-rng:admin-override-sequence",
    "POST:/api/v1/eudr/reference-number-generator/admin/purge-expired": "eudr-rng:admin-purge-expired",

    # Health (1 route - public)
    "GET:/api/v1/eudr/reference-number-generator/health": None,

    # ===========================================================================
    # AGENT-EUDR-039: Customs Declaration Support (31 routes)
    # ===========================================================================

    # Declaration CRUD (4 routes)
    "POST:/api/v1/eudr/customs-declaration-support/declarations": "eudr-cds:create",
    "GET:/api/v1/eudr/customs-declaration-support/declarations": "eudr-cds:read",
    "GET:/api/v1/eudr/customs-declaration-support/declarations/{declaration_id}": "eudr-cds:read",
    "PUT:/api/v1/eudr/customs-declaration-support/declarations/{declaration_id}": "eudr-cds:update",
    "DELETE:/api/v1/eudr/customs-declaration-support/declarations/{declaration_id}": "eudr-cds:delete",

    # Submission (3 routes)
    "POST:/api/v1/eudr/customs-declaration-support/declarations/{declaration_id}/submit": "eudr-cds:submit",
    "POST:/api/v1/eudr/customs-declaration-support/declarations/submit-batch": "eudr-cds:submit-batch",
    "POST:/api/v1/eudr/customs-declaration-support/declarations/{declaration_id}/cancel-submission": "eudr-cds:cancel-submission",

    # Validation (3 routes)
    "POST:/api/v1/eudr/customs-declaration-support/declarations/{declaration_id}/validate": "eudr-cds:validate",
    "POST:/api/v1/eudr/customs-declaration-support/declarations/{declaration_id}/validate-cn-code": "eudr-cds:validate-cn-code",
    "POST:/api/v1/eudr/customs-declaration-support/declarations/{declaration_id}/validate-hs-code": "eudr-cds:validate-hs-code",

    # CN/HS Codes (2 routes)
    "GET:/api/v1/eudr/customs-declaration-support/cn-codes": "eudr-cds:cn-code-map",
    "GET:/api/v1/eudr/customs-declaration-support/hs-codes": "eudr-cds:hs-code-lookup",

    # Origin (2 routes)
    "POST:/api/v1/eudr/customs-declaration-support/declarations/{declaration_id}/origin-verify": "eudr-cds:origin-verify",
    "PUT:/api/v1/eudr/customs-declaration-support/declarations/{declaration_id}/origin": "eudr-cds:origin-update",

    # Value & Tariff (2 routes)
    "POST:/api/v1/eudr/customs-declaration-support/declarations/{declaration_id}/value-calculate": "eudr-cds:value-calculate",
    "POST:/api/v1/eudr/customs-declaration-support/declarations/{declaration_id}/tariff-calculate": "eudr-cds:tariff-calculate",

    # Compliance (2 routes)
    "POST:/api/v1/eudr/customs-declaration-support/declarations/{declaration_id}/compliance-check": "eudr-cds:compliance-check",
    "POST:/api/v1/eudr/customs-declaration-support/declarations/{declaration_id}/compliance-override": "eudr-cds:compliance-override",

    # Customs Systems (3 routes)
    "POST:/api/v1/eudr/customs-declaration-support/declarations/{declaration_id}/ncts-submit": "eudr-cds:ncts-submit",
    "POST:/api/v1/eudr/customs-declaration-support/declarations/{declaration_id}/ais-submit": "eudr-cds:ais-submit",
    "GET:/api/v1/eudr/customs-declaration-support/mrn/{mrn_number}": "eudr-cds:mrn-lookup",

    # Reports (2 routes)
    "POST:/api/v1/eudr/customs-declaration-support/reports/generate": "eudr-cds:report-generate",
    "GET:/api/v1/eudr/customs-declaration-support/reports/{report_id}/download": "eudr-cds:report-download",

    # Admin (3 routes)
    "GET:/api/v1/eudr/customs-declaration-support/admin": "eudr-cds:admin",
    "POST:/api/v1/eudr/customs-declaration-support/admin/bulk-operations": "eudr-cds:admin-bulk-operations",
    "POST:/api/v1/eudr/customs-declaration-support/admin/tariff-update": "eudr-cds:admin-tariff-update",

    # Config (2 routes)
    "GET:/api/v1/eudr/customs-declaration-support/config": "eudr-cds:config:read",
    "PUT:/api/v1/eudr/customs-declaration-support/config": "eudr-cds:config:update",

    # Health (1 route - public)
    "GET:/api/v1/eudr/customs-declaration-support/health": None,

    # ===========================================================================
    # AGENT-EUDR-040: Authority Communication Manager (35 routes)
    # ===========================================================================

    # Communication CRUD (5 routes)
    "POST:/api/v1/eudr/authority-communication-manager/communications": "eudr-acm:create",
    "GET:/api/v1/eudr/authority-communication-manager/communications": "eudr-acm:read",
    "GET:/api/v1/eudr/authority-communication-manager/communications/{communication_id}": "eudr-acm:read",
    "POST:/api/v1/eudr/authority-communication-manager/communications/{communication_id}/respond": "eudr-acm:respond",
    "POST:/api/v1/eudr/authority-communication-manager/communications/{communication_id}/close": "eudr-acm:close",

    # Information Requests (3 routes)
    "GET:/api/v1/eudr/authority-communication-manager/information-requests": "eudr-acm:request-receive",
    "GET:/api/v1/eudr/authority-communication-manager/information-requests/{request_id}": "eudr-acm:request-receive",
    "POST:/api/v1/eudr/authority-communication-manager/information-requests/{request_id}/respond": "eudr-acm:request-respond",

    # Inspections (4 routes)
    "GET:/api/v1/eudr/authority-communication-manager/inspections": "eudr-acm:inspection-schedule",
    "GET:/api/v1/eudr/authority-communication-manager/inspections/{inspection_id}": "eudr-acm:inspection-schedule",
    "POST:/api/v1/eudr/authority-communication-manager/inspections/{inspection_id}/coordinate": "eudr-acm:inspection-coordinate",
    "POST:/api/v1/eudr/authority-communication-manager/inspections/{inspection_id}/report": "eudr-acm:inspection-report",

    # Non-Compliance (4 routes)
    "GET:/api/v1/eudr/authority-communication-manager/non-compliance": "eudr-acm:nc-receive",
    "GET:/api/v1/eudr/authority-communication-manager/non-compliance/{notice_id}": "eudr-acm:nc-receive",
    "POST:/api/v1/eudr/authority-communication-manager/non-compliance/{notice_id}/remediate": "eudr-acm:nc-remediate",
    "POST:/api/v1/eudr/authority-communication-manager/non-compliance/{notice_id}/dispute": "eudr-acm:nc-dispute",

    # Appeals (4 routes)
    "POST:/api/v1/eudr/authority-communication-manager/appeals": "eudr-acm:appeal-file",
    "GET:/api/v1/eudr/authority-communication-manager/appeals": "eudr-acm:appeal-track",
    "GET:/api/v1/eudr/authority-communication-manager/appeals/{appeal_id}": "eudr-acm:appeal-track",
    "POST:/api/v1/eudr/authority-communication-manager/appeals/{appeal_id}/evidence": "eudr-acm:appeal-submit-evidence",

    # Documents (3 routes)
    "POST:/api/v1/eudr/authority-communication-manager/documents/upload": "eudr-acm:document-upload",
    "GET:/api/v1/eudr/authority-communication-manager/documents/{document_id}/download": "eudr-acm:document-download",
    "POST:/api/v1/eudr/authority-communication-manager/documents/{document_id}/encrypt": "eudr-acm:document-encrypt",

    # Notifications (2 routes)
    "POST:/api/v1/eudr/authority-communication-manager/notifications/send": "eudr-acm:notification-send",
    "GET:/api/v1/eudr/authority-communication-manager/notifications": "eudr-acm:notification-read",

    # Authorities (2 routes)
    "GET:/api/v1/eudr/authority-communication-manager/authorities": "eudr-acm:authority-list",
    "GET:/api/v1/eudr/authority-communication-manager/authorities/{authority_id}/contact": "eudr-acm:authority-contact",

    # Templates (2 routes)
    "GET:/api/v1/eudr/authority-communication-manager/templates": "eudr-acm:template-view",
    "POST:/api/v1/eudr/authority-communication-manager/templates/{template_id}/use": "eudr-acm:template-use",

    # Reports (2 routes)
    "POST:/api/v1/eudr/authority-communication-manager/reports/generate": "eudr-acm:report-generate",
    "GET:/api/v1/eudr/authority-communication-manager/reports/{report_id}/download": "eudr-acm:report-download",

    # Admin (3 routes)
    "GET:/api/v1/eudr/authority-communication-manager/admin": "eudr-acm:admin",
    "POST:/api/v1/eudr/authority-communication-manager/admin/manage-authorities": "eudr-acm:admin-manage-authorities",
    "POST:/api/v1/eudr/authority-communication-manager/admin/manage-templates": "eudr-acm:admin-manage-templates",

    # Health (1 route - public)
    "GET:/api/v1/eudr/authority-communication-manager/health": None,
}


# ---------------------------------------------------------------------------
# AuthContext (lightweight re-export / standalone definition)
# ---------------------------------------------------------------------------

try:
    from greenlang.auth.middleware import AuthContext  # noqa: F401
except ImportError:
    # Standalone definition when the middleware module is not available
    @dataclass
    class AuthContext:  # type: ignore[no-redef]
        """Authentication context for the current request."""

        user_id: str
        tenant_id: Optional[str] = None
        email: Optional[str] = None
        name: Optional[str] = None
        roles: List[str] = field(default_factory=list)
        permissions: List[str] = field(default_factory=list)
        scopes: List[str] = field(default_factory=list)
        auth_method: str = "jwt"
        auth_token_id: Optional[str] = None
        session_id: Optional[str] = None
        client_ip: Optional[str] = None
        user_agent: Optional[str] = None
        org_id: Optional[str] = None

        # ---- convenience helpers ----

        def has_role(self, role: str) -> bool:
            """Check if the context includes a role."""
            return role in self.roles

        def has_any_role(self, roles: List[str]) -> bool:
            """Check if the context includes *any* of the roles."""
            return bool(set(self.roles) & set(roles))

        def has_all_roles(self, roles: List[str]) -> bool:
            """Check if the context includes *all* of the roles."""
            return set(roles).issubset(set(self.roles))

        def has_permission(self, permission: str) -> bool:
            """Check if the context includes a permission."""
            return permission in self.permissions

        def has_any_permission(self, permissions: List[str]) -> bool:
            """Check if the context includes *any* of the permissions."""
            return bool(set(self.permissions) & set(permissions))

        def has_scope(self, scope: str) -> bool:
            """Check if the context includes an OAuth2 scope."""
            return scope in self.scopes or "admin" in self.scopes

        def to_dict(self) -> Dict[str, Any]:
            """Serialise to dictionary."""
            return {
                "user_id": self.user_id,
                "tenant_id": self.tenant_id,
                "auth_method": self.auth_method,
                "roles": self.roles,
                "permissions": self.permissions,
                "scopes": self.scopes,
                "email": self.email,
                "name": self.name,
                "org_id": self.org_id,
            }


# ---------------------------------------------------------------------------
# Permission matching helpers
# ---------------------------------------------------------------------------


def _normalise_path(path: str) -> str:
    """Normalise a URL path by stripping trailing slashes."""
    return path.rstrip("/") or "/"


def _is_public_path(path: str) -> bool:
    """Return True if *path* is in the public allow-list.

    Supports exact matches and prefix matching for paths that end with
    trailing segments (e.g. ``/docs/...``).
    """
    normalised = _normalise_path(path)
    if normalised in PUBLIC_PATHS:
        return True
    # Prefix matching for doc sub-paths
    for public in PUBLIC_PATHS:
        if normalised.startswith(public + "/"):
            return True
    return False


def _permission_matches(granted: str, required: str) -> bool:
    """Check whether a *granted* permission satisfies a *required* one.

    Supports wildcard matching:
        - ``agents:*`` matches ``agents:execute``, ``agents:list``, etc.
        - ``admin:*`` matches ``admin:users:list``, ``admin:users:unlock``.
        - ``*`` matches everything.

    Args:
        granted: Permission string the user possesses.
        required: Permission string the endpoint demands.

    Returns:
        True if *granted* satisfies *required*.
    """
    if granted == required:
        return True
    if granted == "*":
        return True
    # Use fnmatch-style: "agents:*" should match "agents:execute"
    if fnmatch.fnmatch(required, granted):
        return True
    # Hierarchical wildcard: "admin:*" should match "admin:users:list"
    if granted.endswith(":*"):
        prefix = granted[:-1]  # "admin:"
        if required.startswith(prefix):
            return True
    return False


def _user_has_permission(auth: AuthContext, required: str) -> bool:
    """Return True if *auth* context satisfies the *required* permission.

    Iterates over all user permissions and checks wildcard matching.

    Args:
        auth: Current authentication context.
        required: The permission string demanded by the endpoint.

    Returns:
        True when at least one granted permission matches.
    """
    for granted in auth.permissions:
        if _permission_matches(granted, required):
            return True
    # super_admin role grants everything
    if "super_admin" in auth.roles:
        return True
    return False


def _lookup_permission_for_route(
    method: str,
    path: str,
    permission_map: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Look up the required permission for a method+path combination.

    Tries an exact match first, then falls back to pattern matching
    against the default ``PERMISSION_MAP``.

    Args:
        method: HTTP method (uppercase).
        path: The route path pattern (may contain ``{param}`` segments).
        permission_map: Optional override map.

    Returns:
        Permission string or None if no mapping exists.
    """
    pmap = permission_map or PERMISSION_MAP
    key = f"{method.upper()}:{_normalise_path(path)}"
    if key in pmap:
        return pmap[key]

    # Try without a trailing path segment (for prefix-based routes)
    for map_key, permission in pmap.items():
        # Convert map key to regex: replace {param} with [^/]+
        pattern_str = map_key.split(":", 1)
        if len(pattern_str) != 2:
            continue
        map_method, map_path = pattern_str
        if map_method != method.upper():
            continue
        regex = re.sub(r"\{[^}]+\}", r"[^/]+", map_path)
        if re.fullmatch(regex, _normalise_path(path)):
            return permission
    return None


# ---------------------------------------------------------------------------
# FastAPI Dependencies
# ---------------------------------------------------------------------------


class AuthDependency:
    """FastAPI dependency that extracts and validates authentication.

    Can operate in two modes:

    1. **With a TokenService** -- validates JWT tokens by calling
       ``token_service.validate_token()``.
    2. **Without a TokenService** -- reads a pre-populated
       ``request.state.auth`` (set by ``AuthenticationMiddleware``).

    Args:
        token_service: Optional ``TokenService`` for direct JWT validation.
        required: When True (default) unauthenticated requests get 401.

    Example:
        >>> auth = AuthDependency(token_service=svc)
        >>> @app.get("/protected")
        ... async def protected(ctx: AuthContext = Depends(auth)):
        ...     return {"user": ctx.user_id}
    """

    def __init__(
        self,
        token_service: Any = None,
        required: bool = True,
    ) -> None:
        self._token_service = token_service
        self._required = required

    async def __call__(self, request: Request) -> Optional[AuthContext]:
        """Extract and validate auth from the incoming request.

        Processing order:
            1. Skip auth if path is in ``PUBLIC_PATHS``.
            2. Return pre-set ``request.state.auth`` if available
               (populated by ``AuthenticationMiddleware``).
            3. Extract ``Authorization: Bearer <token>`` header.
            4. Fall back to ``X-API-Key`` header.
            5. Validate via ``token_service`` if available.
            6. Raise 401 when ``required=True`` and no auth is found.

        Args:
            request: The current FastAPI request.

        Returns:
            ``AuthContext`` when authenticated, or None when
            ``required=False`` and no credentials are present.

        Raises:
            HTTPException: 401 if authentication is required but missing
                or invalid.
        """
        # 1. Public path bypass
        if _is_public_path(request.url.path):
            logger.debug("Public path bypass: %s", request.url.path)
            return None

        # 2. Already authenticated via middleware
        existing_auth: Optional[AuthContext] = getattr(
            request.state, "auth", None
        )
        if existing_auth is not None:
            return existing_auth

        # 3. Try Bearer token
        auth_context = await self._authenticate_bearer(request)

        # 4. Try API key
        if auth_context is None:
            auth_context = await self._authenticate_api_key(request)

        # 5. Store in request state for downstream use
        if auth_context is not None:
            request.state.auth = auth_context
            return auth_context

        # 6. Enforce requirement
        if self._required:
            logger.warning(
                "Authentication required but missing: method=%s path=%s ip=%s",
                request.method,
                request.url.path,
                _extract_client_ip(request),
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return None

    # -- private helpers --

    async def _authenticate_bearer(
        self, request: Request
    ) -> Optional[AuthContext]:
        """Validate a Bearer token from the Authorization header.

        Args:
            request: The current FastAPI request.

        Returns:
            AuthContext on success, None otherwise.
        """
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return None

        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return None

        token = parts[1]

        # If we have a token service, validate through it
        if self._token_service is not None:
            try:
                claims = await self._token_service.validate_token(token)
                return AuthContext(
                    user_id=claims.sub,
                    tenant_id=claims.tenant_id,
                    auth_method="jwt",
                    auth_token_id=getattr(claims, "jti", None),
                    roles=getattr(claims, "roles", []),
                    permissions=getattr(claims, "permissions", []),
                    scopes=getattr(claims, "scopes", []),
                    email=getattr(claims, "email", None),
                    name=getattr(claims, "name", None),
                    client_ip=_extract_client_ip(request),
                    user_agent=request.headers.get("User-Agent"),
                )
            except Exception as exc:
                logger.debug("Bearer token validation failed: %s", exc)
                return None

        # No token service -- cannot validate
        logger.debug("No token_service configured; Bearer token ignored")
        return None

    async def _authenticate_api_key(
        self, request: Request
    ) -> Optional[AuthContext]:
        """Validate an API key from the X-API-Key header.

        API key validation is delegated to the middleware layer. This
        method only checks whether ``request.state.auth`` was populated
        by an upstream ``APIKeyAuthBackend``.

        Args:
            request: The current FastAPI request.

        Returns:
            AuthContext on success, None otherwise.
        """
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            return None
        # API key validation requires the middleware to have run already.
        # We cannot validate raw API keys here without an api_key_manager.
        logger.debug(
            "X-API-Key header present but no API key manager configured"
        )
        return None


class PermissionDependency:
    """FastAPI dependency that enforces a required permission string.

    Reads the ``AuthContext`` from ``request.state.auth`` and checks
    whether the user possesses the required permission, including
    wildcard expansion.

    Args:
        required_permission: The permission string the endpoint demands
            (e.g. ``"agents:execute"``).

    Example:
        >>> @app.post("/agents/{agent_id}/execute")
        ... async def execute(
        ...     request: Request,
        ...     _: None = Depends(PermissionDependency("agents:execute")),
        ... ):
        ...     ...
    """

    def __init__(self, required_permission: str) -> None:
        self._permission = required_permission

    async def __call__(self, request: Request) -> None:
        """Verify the current user has the required permission.

        Args:
            request: The current FastAPI request.

        Raises:
            HTTPException: 401 if not authenticated, 403 if permission
                is denied.
        """
        auth: Optional[AuthContext] = getattr(request.state, "auth", None)
        if auth is None:
            logger.warning(
                "Permission check failed (unauthenticated): "
                "permission=%s path=%s",
                self._permission,
                request.url.path,
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if not _user_has_permission(auth, self._permission):
            logger.warning(
                "Permission denied: user=%s tenant=%s "
                "required=%s granted=%s path=%s",
                auth.user_id,
                auth.tenant_id,
                self._permission,
                auth.permissions,
                request.url.path,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "permission_denied",
                    "required_permission": self._permission,
                    "message": (
                        f"Permission '{self._permission}' is required "
                        f"to access this resource"
                    ),
                },
            )

        logger.debug(
            "Permission granted: user=%s permission=%s path=%s",
            auth.user_id,
            self._permission,
            request.url.path,
        )


class TenantDependency:
    """FastAPI dependency that enforces tenant isolation.

    Ensures the ``tenant_id`` in the auth context matches the tenant
    implied by the request (via path param, query param, or header).
    ``super_admin`` users bypass tenant checks.

    Args:
        param_name: Name of the path/query parameter carrying the tenant
            id.  Defaults to ``"tenant_id"``.
        header_name: Name of the HTTP header carrying the tenant id.
            Defaults to ``"X-Tenant-ID"``.

    Example:
        >>> @app.get("/tenants/{tenant_id}/agents")
        ... async def get_agents(
        ...     tenant_id: str,
        ...     _: str = Depends(TenantDependency()),
        ... ):
        ...     ...
    """

    def __init__(
        self,
        param_name: str = "tenant_id",
        header_name: str = "X-Tenant-ID",
    ) -> None:
        self._param_name = param_name
        self._header_name = header_name

    async def __call__(self, request: Request) -> str:
        """Validate tenant isolation and return the effective tenant_id.

        Args:
            request: The current FastAPI request.

        Returns:
            The validated tenant_id string.

        Raises:
            HTTPException: 401 if not authenticated, 403 if the tenant
                does not match.
        """
        auth: Optional[AuthContext] = getattr(request.state, "auth", None)
        if auth is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required for tenant-scoped access",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Determine the requested tenant from multiple sources
        requested_tenant = self._resolve_tenant(request)

        if requested_tenant is None:
            # No explicit tenant requested; use the auth tenant
            return auth.tenant_id or ""

        if requested_tenant == auth.tenant_id:
            return requested_tenant

        # Cross-tenant access: only super_admin allowed
        if auth.has_role("super_admin") if hasattr(auth, "has_role") else "super_admin" in auth.roles:
            logger.info(
                "Cross-tenant access granted to super_admin: "
                "user=%s own_tenant=%s target_tenant=%s",
                auth.user_id,
                auth.tenant_id,
                requested_tenant,
            )
            return requested_tenant

        logger.warning(
            "Tenant isolation violation: user=%s own_tenant=%s "
            "target_tenant=%s path=%s",
            auth.user_id,
            auth.tenant_id,
            requested_tenant,
            request.url.path,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: tenant isolation violation",
        )

    def _resolve_tenant(self, request: Request) -> Optional[str]:
        """Resolve the target tenant_id from the request.

        Checks in order: path params, query params, HTTP header.

        Args:
            request: The current FastAPI request.

        Returns:
            Tenant id string or None.
        """
        # Path parameter
        tenant = request.path_params.get(self._param_name)
        if tenant:
            return str(tenant)

        # Query parameter
        tenant = request.query_params.get(self._param_name)
        if tenant:
            return str(tenant)

        # Header
        tenant = request.headers.get(self._header_name)
        if tenant:
            return str(tenant)

        return None


# ---------------------------------------------------------------------------
# Router-level protection
# ---------------------------------------------------------------------------


def protect_router(
    router: APIRouter,
    auth_dep: Optional[AuthDependency] = None,
    permission_map: Optional[Dict[str, str]] = None,
    *,
    exclude_paths: Optional[Set[str]] = None,
) -> APIRouter:
    """Apply authentication and permission protection to every route.

    Iterates over all routes on *router* and injects ``AuthDependency``
    and ``PermissionDependency`` as FastAPI dependencies.  The original
    handler code is *not* modified.

    Args:
        router: The ``APIRouter`` whose routes will be protected.
        auth_dep: ``AuthDependency`` instance.  A default instance is
            created when ``None``.
        permission_map: Optional mapping of ``"METHOD:/path"`` to
            permission strings.  Falls back to the module-level
            ``PERMISSION_MAP``.
        exclude_paths: Optional set of path patterns to exclude from
            protection (in addition to ``PUBLIC_PATHS``).

    Returns:
        The *same* ``APIRouter`` (mutated in place) with dependencies
        injected.

    Example:
        >>> from greenlang.infrastructure.auth_service.route_protector import (
        ...     protect_router, AuthDependency,
        ... )
        >>> protect_router(agents_router, auth_dep=AuthDependency(token_service=svc))
    """
    effective_auth = auth_dep or AuthDependency()
    effective_map = permission_map or PERMISSION_MAP
    effective_excludes = exclude_paths or set()

    protected_count = 0
    skipped_count = 0

    for route in router.routes:
        if not isinstance(route, APIRoute):
            continue

        route_path = _normalise_path(route.path)

        # Skip public and excluded paths
        if _is_public_path(route_path) or route_path in effective_excludes:
            skipped_count += 1
            continue

        # Inject auth dependency if not already present
        _inject_auth_dependency(route, effective_auth)

        # Inject permission dependency for each method
        for method in route.methods or {"GET"}:
            perm = _lookup_permission_for_route(
                method, route_path, effective_map
            )
            if perm is not None:
                _inject_permission_dependency(route, perm)

        protected_count += 1

    logger.info(
        "Router protected: %d routes secured, %d skipped (public/excluded)",
        protected_count,
        skipped_count,
    )
    return router


def _inject_auth_dependency(
    route: APIRoute, auth_dep: AuthDependency
) -> None:
    """Add AuthDependency to a route's dependency list.

    Avoids duplicate injection by checking existing dependency types.

    Args:
        route: The FastAPI route to modify.
        auth_dep: The ``AuthDependency`` instance.
    """
    for dep in route.dependencies:
        if isinstance(dep.dependency, AuthDependency):
            return  # already protected
    route.dependencies.append(Depends(auth_dep))


def _inject_permission_dependency(
    route: APIRoute, permission: str
) -> None:
    """Add a PermissionDependency to a route's dependency list.

    Args:
        route: The FastAPI route to modify.
        permission: The permission string to enforce.
    """
    for dep in route.dependencies:
        if (
            isinstance(dep.dependency, PermissionDependency)
            and dep.dependency._permission == permission
        ):
            return  # already has this exact permission check
    route.dependencies.append(Depends(PermissionDependency(permission)))


# ---------------------------------------------------------------------------
# Decorator-style protection for individual route handlers
# ---------------------------------------------------------------------------


def require_auth(func: Optional[Callable] = None) -> Callable:
    """Decorator that requires authentication on a route handler.

    Can be used with or without parentheses::

        @require_auth
        async def handler(request: Request): ...

        @require_auth()
        async def handler(request: Request): ...

    The decorator extracts the ``Request`` from *args* or *kwargs* and
    verifies that ``request.state.auth`` is set.

    Args:
        func: The route handler function to wrap.

    Returns:
        Wrapped handler that raises 401 when unauthenticated.
    """

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            request = _extract_request(args, kwargs)
            auth = getattr(request.state, "auth", None) if request else None
            if auth is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return await fn(*args, **kwargs)

        return wrapper

    if func is not None:
        # Called without parentheses: @require_auth
        return decorator(func)
    # Called with parentheses: @require_auth()
    return decorator


def require_permissions(*permissions: str) -> Callable:
    """Decorator that requires *any* of the specified permissions.

    Uses wildcard-aware matching so ``agents:*`` satisfies
    ``agents:execute``.

    Args:
        *permissions: One or more permission strings.  The user must
            possess at least one.

    Returns:
        Decorator function.

    Example:
        >>> @require_permissions("agents:execute", "agents:admin")
        ... async def execute_agent(request: Request): ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            request = _extract_request(args, kwargs)
            auth = getattr(request.state, "auth", None) if request else None
            if auth is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            # Check each required permission with wildcard matching
            has_perm = any(
                _user_has_permission(auth, perm) for perm in permissions
            )
            if not has_perm:
                logger.warning(
                    "Permission denied (decorator): user=%s "
                    "required=%s granted=%s",
                    auth.user_id,
                    permissions,
                    auth.permissions,
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={
                        "error": "permission_denied",
                        "required_permissions": list(permissions),
                        "message": (
                            f"One of {permissions} is required "
                            f"to access this resource"
                        ),
                    },
                )

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def require_roles(*roles: str) -> Callable:
    """Decorator that requires *any* of the specified roles.

    Args:
        *roles: One or more role names.  The user must possess at least
            one.

    Returns:
        Decorator function.

    Example:
        >>> @require_roles("admin", "super_admin")
        ... async def admin_endpoint(request: Request): ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            request = _extract_request(args, kwargs)
            auth = getattr(request.state, "auth", None) if request else None
            if auth is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            user_roles = set(auth.roles) if hasattr(auth, "roles") else set()
            required_roles = set(roles)
            if not user_roles & required_roles:
                logger.warning(
                    "Role denied (decorator): user=%s required=%s actual=%s",
                    auth.user_id,
                    roles,
                    auth.roles,
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={
                        "error": "role_denied",
                        "required_roles": list(roles),
                        "message": (
                            f"One of roles {roles} is required "
                            f"to access this resource"
                        ),
                    },
                )

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def require_tenant(func: Optional[Callable] = None) -> Callable:
    """Decorator that enforces tenant_id match.

    Reads ``tenant_id`` from the route's kwargs or path params and
    compares it against the auth context.  ``super_admin`` bypasses the
    check.

    Can be used with or without parentheses.

    Args:
        func: The route handler function to wrap.

    Returns:
        Wrapped handler or decorator function.
    """

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            request = _extract_request(args, kwargs)
            auth = getattr(request.state, "auth", None) if request else None
            if auth is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            # Look for tenant_id in kwargs or path params
            requested_tenant = kwargs.get("tenant_id")
            if requested_tenant is None and request is not None:
                requested_tenant = request.path_params.get("tenant_id")

            if (
                requested_tenant is not None
                and requested_tenant != auth.tenant_id
            ):
                is_super = (
                    auth.has_role("super_admin")
                    if hasattr(auth, "has_role")
                    else "super_admin" in auth.roles
                )
                if not is_super:
                    logger.warning(
                        "Tenant isolation violation (decorator): "
                        "user=%s own=%s target=%s",
                        auth.user_id,
                        auth.tenant_id,
                        requested_tenant,
                    )
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Access denied: tenant isolation violation",
                    )

            return await fn(*args, **kwargs)

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_request(
    args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> Optional[Request]:
    """Locate the FastAPI Request object from handler args/kwargs.

    Args:
        args: Positional arguments to the handler.
        kwargs: Keyword arguments to the handler.

    Returns:
        The ``Request`` instance or None.
    """
    request = kwargs.get("request")
    if request is not None:
        return request
    for arg in args:
        if isinstance(arg, Request):
            return arg
    return None


def _extract_client_ip(request: Request) -> str:
    """Extract the client IP address respecting reverse proxies.

    Args:
        request: The current FastAPI request.

    Returns:
        Client IP string.
    """
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    # Constants
    "PUBLIC_PATHS",
    "PERMISSION_MAP",
    # Data classes
    "AuthContext",
    # Dependencies
    "AuthDependency",
    "PermissionDependency",
    "TenantDependency",
    # Router protection
    "protect_router",
    # Decorators
    "require_auth",
    "require_permissions",
    "require_roles",
    "require_tenant",
    # Helpers
    "permission_matches",
]

# Public alias for the internal matching function
permission_matches = _permission_matches
