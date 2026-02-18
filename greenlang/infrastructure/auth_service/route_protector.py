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
