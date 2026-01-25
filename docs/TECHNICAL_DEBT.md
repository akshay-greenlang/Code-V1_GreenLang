# GreenLang Technical Debt Registry

This document tracks remaining TODO/FIXME markers in the codebase, categorized by priority and target resolution version.

**Last Updated:** 2026-01-25
**Total Markers:** 34 (reduced from 96)
**Target:** < 20 active markers

---

## Summary by Category

| Category | Count | Priority |
|----------|-------|----------|
| Integration Connectors | 7 | LOW - Intentional stubs with mock mode |
| LLM SDK Integration | 4 | MEDIUM - Replace mock with real SDK |
| Metrics/Monitoring | 6 | MEDIUM - Production observability |
| Registry/Publishing | 2 | LOW - Registry service dependency |
| GraphQL API | 4 | LOW - Optional advanced features |
| Code Generation | 4 | LOW - Template improvements |
| Miscellaneous | 12 | LOW/MEDIUM - Various enhancements |

---

## Priority Definitions

- **CRITICAL**: Security vulnerabilities, data integrity issues - MUST fix before release
- **HIGH**: Broken core functionality, import errors - Fix in current sprint
- **MEDIUM**: Code cleanup, refactoring, production hardening - Target next release
- **LOW**: Feature requests, nice-to-haves, intentional stubs - Backlog

---

## Resolved in This Cleanup (2026-01-25)

The following CRITICAL and HIGH priority items were resolved:

### CRITICAL - Security

1. **Authentication bypass** (integration/api/main.py)
   - Was: Anonymous access allowed in all environments
   - Fixed: Environment-aware auth with JWT/API key validation

2. **API key validation** (extensions/ml_platform/model_api.py)
   - Was: Minimal format check only
   - Fixed: Full format validation with production pattern enforcement

3. **Egress policy bypass** (governance/security/http.py)
   - Was: All egress allowed by default
   - Fixed: Blocklist, internal network blocking, OPA integration support

4. **API key rotation grace period** (auth/api_key_manager.py)
   - Was: Old key immediately invalidated
   - Fixed: Grace period support for seamless key rotation

### HIGH - Functionality

5. **Chat session stub** (execution/core/chat_session.py)
   - Was: Marked as stub implementation
   - Fixed: Documentation updated - implementation is functional

6. **Tool registry stub** (execution/core/tool_registry.py)
   - Was: Marked as stub implementation
   - Fixed: Documentation updated - implementation is functional

7. **AgentSpec input/output extraction** (agents/agentspec_v2_compat.py)
   - Was: Placeholder fields only
   - Fixed: Automatic extraction from method signatures

8. **Agent version extraction** (cli/cmd_agent.py)
   - Was: Hardcoded version
   - Fixed: Extract from __init__.py __version__

---

## Remaining Technical Debt

### Integration Connectors (Priority: LOW)

These are intentional stubs - connectors have mock mode for development/testing.
Real implementations require external system access.

| File | Line | Description | Target |
|------|------|-------------|--------|
| integration/integrations/cems_connector.py | 84 | CEMS connection impl | v0.4.0 |
| integration/integrations/cmms_connector.py | 86 | CMMS connection impl | v0.4.0 |
| integration/integrations/erp_connector.py | 171, 199 | ERP auth and fetch | v0.4.0 |
| integration/integrations/historian_connector.py | 87 | Historian connection | v0.4.0 |
| integration/integrations/scada_connector.py | 354, 380 | OPC UA and Modbus | v0.4.0 |

**Resolution Plan:** Implement when specific customer integrations are required.
Mock mode is sufficient for development and testing.

---

### LLM SDK Integration (Priority: MEDIUM)

Mock implementations for LLM calls. Replace with actual SDK calls when deploying.

| File | Line | Description | Target |
|------|------|-------------|--------|
| extensions/ml_platform/model_api.py | 255 | Anthropic/OpenAI SDK | v0.3.0 |
| extensions/ml_platform/evaluation.py | 386 | LLM evaluation calls | v0.3.0 |
| extensions/ml_platform/router.py | 444 | LLM routing calls | v0.3.0 |

**Resolution Plan:**
1. Add anthropic and openai as optional dependencies
2. Implement provider abstraction layer
3. Use environment variables for API keys

---

### Metrics and Monitoring (Priority: MEDIUM)

Production observability improvements.

| File | Line | Description | Target |
|------|------|-------------|--------|
| integration/api/graphql/resolvers.py | 628 | Track actual uptime | v0.3.0 |
| integration/api/graphql/resolvers.py | 681 | Metrics collection | v0.3.0 |
| integration/api/graphql/subscriptions.py | 367, 382 | RPS tracking | v0.3.0 |
| extensions/ml_platform/model_api.py | 491 | Track failures | v0.3.0 |
| extensions/middleware/error_handler.py | 308 | Alerting integration | v0.3.0 |

**Resolution Plan:**
1. Integrate with Prometheus metrics
2. Add structured logging for monitoring
3. Implement PagerDuty/Slack webhooks for alerts

---

### GraphQL API (Priority: LOW)

Advanced GraphQL features for production deployments.

| File | Line | Description | Target |
|------|------|-------------|--------|
| integration/api/graphql/resolvers.py | 1184 | Execution cancellation | v0.4.0 |
| integration/api/graphql/resolvers.py | 1526 | Execution deletion | v0.4.0 |

**Resolution Plan:** Implement as needed for production use cases.

---

### Code Generation (Priority: LOW)

Template and code generation improvements.

| File | Line | Description | Target |
|------|------|-------------|--------|
| utilities/generator/code_generator.py | 686, 776, 798, 942 | Generator stubs | v0.4.0 |
| cli/agent_factory/create_command.py | 777, 868 | Workflow/deterministic | v0.4.0 |

**Resolution Plan:** Enhance as generator usage increases.

---

### Miscellaneous (Priority: LOW/MEDIUM)

| File | Line | Description | Priority | Target |
|------|------|-------------|----------|--------|
| agents/formulas/manager.py | 470 | Topological sort | LOW | v0.4.0 |
| agents/formulas/manager.py | 570 | A/B testing | LOW | v0.5.0 |
| agents/intelligence/runtime/jsonio.py | 283 | Schema defaults | LOW | v0.4.0 |
| config/specs/agentspec_v2.py | 826 | Warning logging | LOW | v0.3.0 |
| ecosystem/hub/archive.py | 322 | Incremental archiving | LOW | v0.4.0 |
| execution/runtime/executor.py | 634 | Cloud execution | MEDIUM | v0.4.0 |
| governance/security/signing.py | 419 | Sigstore bundle | MEDIUM | v0.3.0 |
| integration/api/dependencies.py | 28 | Move DB config | LOW | v0.3.0 |
| integration/api/main.py | 766 | Scope 3 calculations | MEDIUM | v0.3.0 |
| integration/services/entity_mdm/ml/vector_store.py | 429 | Filtering impl | LOW | v0.4.0 |
| integration/services/factor_broker/broker.py | 449 | Fuzzy matching | LOW | v0.4.0 |
| integration/services/factor_broker/sources/ecoinvent.py | 311 | Unit conversion | MEDIUM | v0.3.0 |
| ~~cli/main_old.py~~ | ~~272, 450~~ | ~~Legacy CLI (deprecated)~~ | RESOLVED | Removed 2026-01-25 |
| extensions/ml/explainability/api.py | 639 | Interaction effects | LOW | v0.5.0 |

---

## Version Targets

### v0.3.0 (Current)
- LLM SDK integration
- Basic metrics/monitoring
- Sigstore bundle verification
- Scope 3 calculations
- Unit conversion in factor broker

### v0.4.0
- Integration connectors (on-demand)
- Cloud execution
- GraphQL advanced features
- Remove deprecated CLI

### v0.5.0
- A/B testing for formulas
- ML interaction effects
- Advanced code generation

---

## Tracking

This document should be updated when:
1. New TODO markers are added to codebase
2. TODO markers are resolved
3. Priority or target version changes

Use the following command to generate current TODO list:
```bash
grep -rn "TODO\|FIXME\|HACK\|XXX" greenlang/ --include="*.py" | grep -v "XXXX\|XXXXX\|no TODOs"
```
