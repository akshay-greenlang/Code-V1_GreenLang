# GreenLang Agent Exit Bar Criteria Checklist
## Human-Readable Production Readiness Guide

**Version:** 1.0.0
**Date:** 2025-10-16
**Based on:** GL_agent_requirement.md 12-dimension framework
**Purpose:** Validate all GreenLang AI agents before production deployment

---

## How to Use This Checklist

1. **For each agent**: Go through all 12 dimensions systematically
2. **Score each criterion**: Use the point system (total = 100 points)
3. **Identify blockers**: Mark any REQUIRED criteria that fail
4. **Calculate total**: Agent needs ≥95/100 to be production-ready
5. **Generate report**: Use `validate_exit_bar.py` for automated validation

### Scoring Thresholds

- **100-95:** ✅ PRODUCTION READY - Deploy to production
- **94-80:** ⚠️ PRE-PRODUCTION - Minor gaps, nearly ready
- **79-60:** 🟡 DEVELOPMENT - Major work complete, significant gaps
- **59-40:** 🟠 EARLY DEVELOPMENT - Some implementation, many gaps
- **39-20:** 🔴 SPECIFICATION ONLY - Spec exists, minimal code
- **19-0:** ⚫ NOT STARTED - Little to no work

---

## Agent Information

**Agent Name:** ___________________________
**Agent ID:** ___________________________
**Spec Path:** ___________________________
**Evaluator:** ___________________________
**Date:** ___________________________

---

## D1: Specification Completeness (10 points)

**Total Possible:** 10 points
**Minimum Required:** 10 points (all criteria are REQUIRED)

### D1.1: AgentSpec V2.0 YAML File Exists (2 points)

- [ ] **REQUIRED** - Specification file exists at correct path
  - **Expected location:** `specs/{domain}/{subdomain}/agent_{id}_{name}.yaml`
  - **Validation:** File exists and is readable
  - **Points:** 2

**Status:** [ ] PASS  [ ] FAIL
**Notes:** _________________________________

---

### D1.2: All 11 Mandatory Sections Present (2 points)

- [ ] **REQUIRED** - All 11 sections exist in specification

Required sections:
1. [ ] `agent_metadata` - Identity, domain, complexity, priority
2. [ ] `description` - Purpose, strategic context, capabilities
3. [ ] `tools` - Tool-first architecture (4-20 tools)
4. [ ] `ai_integration` - ChatSession config (temp=0.0, seed=42)
5. [ ] `sub_agents` - Coordination patterns (if applicable)
6. [ ] `inputs` - JSON Schema with validation
7. [ ] `outputs` - JSON Schema with provenance
8. [ ] `testing` - Coverage targets (≥80%), test categories
9. [ ] `deployment` - Pack config, dependencies, resources
10. [ ] `documentation` - README, API docs, examples
11. [ ] `compliance` - Security, SBOM, standards
12. [ ] `metadata` - Version control, changelog, reviewers

**Status:** [ ] PASS (11/11)  [ ] FAIL
**Count:** ___/11
**Notes:** _________________________________

---

### D1.3: Specification Validation Passes (2 points)

- [ ] **REQUIRED** - Zero validation errors from validate_agent_specs.py

**Command:** `python scripts/validate_agent_specs.py {spec_path}`
**Expected:** "0 ERRORS" or "VALIDATION PASSED"

**Status:** [ ] PASS  [ ] FAIL
**Error Count:** ___
**Notes:** _________________________________

---

### D1.4: AI Temperature = 0.0 (2 points)

- [ ] **REQUIRED** - temperature=0.0 configured for determinism

**YAML Path:** `ai_integration.temperature`
**Expected Value:** `0.0` (exactly)

**Status:** [ ] PASS  [ ] FAIL
**Actual Value:** ___
**Notes:** _________________________________

---

### D1.5: AI Seed = 42 (2 points)

- [ ] **REQUIRED** - seed=42 configured for reproducibility

**YAML Path:** `ai_integration.seed`
**Expected Value:** `42` (exactly)

**Status:** [ ] PASS  [ ] FAIL
**Actual Value:** ___
**Notes:** _________________________________

---

**D1 TOTAL SCORE:** ___/10

---

## D2: Code Implementation (15 points)

**Total Possible:** 15 points
**Minimum Required:** 15 points (all criteria are REQUIRED)

### D2.1: Implementation File Exists (3 points)

- [ ] **REQUIRED** - Python implementation file exists

**Expected location:** `greenlang/agents/{agent_name}_ai.py`
**Validation:** File exists and is readable

**Status:** [ ] PASS  [ ] FAIL
**Notes:** _________________________________

---

### D2.2: Tool-First Architecture (3 points)

- [ ] **REQUIRED** - At least 3 tool implementations present

**Pattern:** `def _.*_impl(self`
**Minimum Count:** 3 tool implementations

**Status:** [ ] PASS  [ ] FAIL
**Actual Count:** ___
**Notes:** _________________________________

---

### D2.3: ChatSession Integration (3 points)

- [ ] **REQUIRED** - ChatSession integration present

**Pattern:** `ChatSession(self.provider)` or `ChatSession(`
**Minimum Count:** 1 occurrence

**Status:** [ ] PASS  [ ] FAIL
**Notes:** _________________________________

---

### D2.4: Type Hints Complete (3 points)

- [ ] **REQUIRED** - mypy type checking passes

**Command:** `mypy greenlang/agents/{agent_name}_ai.py --strict`
**Expected:** "Success: no issues found" (warnings OK)

**Status:** [ ] PASS  [ ] FAIL
**Error Count:** ___
**Notes:** _________________________________

---

### D2.5: No Hardcoded Secrets (3 points)

- [ ] **REQUIRED** - Secret scanning passes (zero secrets found)

**Command:** `grep -rn 'sk-|api_key.*=|password.*=' {file}`
**Expected:** No matches (CLEAN)

**Status:** [ ] PASS  [ ] FAIL
**Secrets Found:** ___
**Notes:** _________________________________

---

**D2 TOTAL SCORE:** ___/15

---

## D3: Test Coverage (15 points)

**Total Possible:** 15 points
**Minimum Required:** 13 points (D3.1, D3.2, D3.7 are critical)

### D3.1: Test File Exists (2 points)

- [ ] **REQUIRED** - Test file exists at correct location

**Expected location:** `tests/agents/test_{agent_name}_ai.py`
**Validation:** File exists and is readable

**Status:** [ ] PASS  [ ] FAIL
**Notes:** _________________________________

---

### D3.2: Line Coverage ≥80% (5 points)

- [ ] **REQUIRED** - Test coverage meets minimum threshold

**Command:** `pytest tests/agents/test_{agent_name}_ai.py --cov=greenlang.agents.{agent_name}_ai --cov-report=term`
**Expected:** Coverage ≥80%

**Status:** [ ] PASS  [ ] FAIL
**Actual Coverage:** ___%
**Notes:** _________________________________

---

### D3.3: Unit Tests Present (2 points)

- [ ] **REQUIRED** - Sufficient unit tests for tool implementations

**Pattern:** `def test_.*_tool` or `def test_.*_impl`
**Minimum Count:** 10 unit tests

**Status:** [ ] PASS  [ ] FAIL
**Actual Count:** ___
**Notes:** _________________________________

---

### D3.4: Integration Tests Present (2 points)

- [ ] **REQUIRED** - Sufficient integration tests for workflows

**Pattern:** `def test_.*_workflow` or `def test_.*_integration` or `def test_full_`
**Minimum Count:** 5 integration tests

**Status:** [ ] PASS  [ ] FAIL
**Actual Count:** ___
**Notes:** _________________________________

---

### D3.5: Determinism Tests Present (2 points)

- [ ] **REQUIRED** - Sufficient determinism tests

**Pattern:** `def test_.*_determinism` or `def test_.*_reproducible` or `def test_same_input`
**Minimum Count:** 3 determinism tests

**Status:** [ ] PASS  [ ] FAIL
**Actual Count:** ___
**Notes:** _________________________________

---

### D3.6: Boundary Tests Present (1 point)

- [ ] **REQUIRED** - Sufficient boundary/edge case tests

**Pattern:** `def test_.*_empty` or `def test_.*_zero` or `def test_.*_negative` or `def test_.*_boundary`
**Minimum Count:** 5 boundary tests

**Status:** [ ] PASS  [ ] FAIL
**Actual Count:** ___
**Notes:** _________________________________

---

### D3.7: All Tests Passing (1 point)

- [ ] **REQUIRED** - Zero test failures

**Command:** `pytest tests/agents/test_{agent_name}_ai.py -v`
**Expected:** All tests pass, zero failures

**Status:** [ ] PASS  [ ] FAIL
**Failed Tests:** ___
**Notes:** _________________________________

---

**D3 TOTAL SCORE:** ___/15

---

## D4: Deterministic AI Guarantees (10 points)

**Total Possible:** 10 points
**Minimum Required:** 10 points (all criteria are REQUIRED)

### D4.1: Temperature = 0.0 in Code (3 points)

- [ ] **REQUIRED** - temperature=0.0 in ChatSession calls

**Pattern:** `temperature=0.0`
**Minimum Count:** 1 occurrence in code

**Status:** [ ] PASS  [ ] FAIL
**Notes:** _________________________________

---

### D4.2: Seed = 42 in Code (3 points)

- [ ] **REQUIRED** - seed=42 in ChatSession calls

**Pattern:** `seed=42`
**Minimum Count:** 1 occurrence in code

**Status:** [ ] PASS  [ ] FAIL
**Notes:** _________________________________

---

### D4.3: All Tools Deterministic (2 points)

- [ ] **REQUIRED** - No random operations in tool implementations

**Pattern:** `random.` or `np.random.` or `torch.rand`
**Expected:** 0 occurrences (no randomness)

**Status:** [ ] PASS  [ ] FAIL
**Random Operations Found:** ___
**Notes:** _________________________________

---

### D4.4: Provenance Tracking Enabled (2 points)

- [ ] **REQUIRED** - Provenance tracking implemented

**Pattern:** `provenance` or `metadata["provenance"]` or `tool_calls`
**Minimum Count:** 1 occurrence

**Status:** [ ] PASS  [ ] FAIL
**Notes:** _________________________________

---

**D4 TOTAL SCORE:** ___/10

---

## D5: Documentation Completeness (5 points)

**Total Possible:** 5 points
**Minimum Required:** 3 points (D5.1, D5.2, D5.3 are critical)

### D5.1: Module Docstring Present (1 point)

- [ ] **REQUIRED** - Module-level docstring exists

**Pattern:** `""".*AI-powered.*"""`
**Expected:** Comprehensive module docstring

**Status:** [ ] PASS  [ ] FAIL
**Notes:** _________________________________

---

### D5.2: Class Docstring Present (1 point)

- [ ] **REQUIRED** - Class docstring with features section

**Pattern:** `class.*AI.*:.*""".*Features:`
**Expected:** Comprehensive class docstring

**Status:** [ ] PASS  [ ] FAIL
**Notes:** _________________________________

---

### D5.3: Method Docstrings (1 point)

- [ ] **REQUIRED** - 90%+ methods have docstrings

**Expected:** ≥90% docstring coverage

**Status:** [ ] PASS  [ ] FAIL
**Actual Coverage:** ___%
**Notes:** _________________________________

---

### D5.4: README/Documentation File (1 point)

- [ ] Optional - Agent-specific documentation exists

**Locations:** `docs/agents/{agent_name}_ai.md` OR `greenlang/agents/README_{agent_name}.md`

**Status:** [ ] PASS  [ ] FAIL
**Notes:** _________________________________

---

### D5.5: Example Use Cases (1 point)

- [ ] Optional - 3+ example use cases in spec

**YAML Path:** `documentation.example_use_cases`
**Minimum Count:** 3 examples

**Status:** [ ] PASS  [ ] FAIL
**Actual Count:** ___
**Notes:** _________________________________

---

**D5 TOTAL SCORE:** ___/5

---

## D6: Compliance & Security (10 points)

**Total Possible:** 10 points
**Minimum Required:** 10 points (all criteria are REQUIRED)

### D6.1: Zero Secrets Flag (3 points)

- [ ] **REQUIRED** - zero_secrets=true in specification

**YAML Path:** `compliance.zero_secrets`
**Expected Value:** `true`

**Status:** [ ] PASS  [ ] FAIL
**Notes:** _________________________________

---

### D6.2: SBOM Required Flag (2 points)

- [ ] **REQUIRED** - sbom_required=true in specification

**YAML Path:** `compliance.sbom_required`
**Expected Value:** `true`

**Status:** [ ] PASS  [ ] FAIL
**Notes:** _________________________________

---

### D6.3: Digital Signature Flag (1 point)

- [ ] **REQUIRED** - digital_signature=true in specification

**YAML Path:** `compliance.digital_signature`
**Expected Value:** `true`

**Status:** [ ] PASS  [ ] FAIL
**Notes:** _________________________________

---

### D6.4: Standards Declared (2 points)

- [ ] **REQUIRED** - 2+ compliance standards declared

**YAML Path:** `compliance.standards`
**Minimum Count:** 2 standards

**Status:** [ ] PASS  [ ] FAIL
**Actual Count:** ___
**Standards:** _________________________________

---

### D6.5: No Hardcoded Credentials (2 points)

- [ ] **REQUIRED** - No credentials in code

**Pattern:** `API_KEY=|SECRET=|PASSWORD=|TOKEN=`
**Expected:** 0 matches (CLEAN)

**Status:** [ ] PASS  [ ] FAIL
**Notes:** _________________________________

---

**D6 TOTAL SCORE:** ___/10

---

## D7: Deployment Readiness (10 points)

**Total Possible:** 10 points
**Minimum Required:** 9 points (D7.5 is optional)

### D7.1: Pack Configuration Exists (3 points)

- [ ] **REQUIRED** - Complete deployment pack configuration

Required fields:
- [ ] `deployment.pack_id`
- [ ] `deployment.pack_version`
- [ ] `deployment.dependencies`
- [ ] `deployment.resource_requirements`

**Status:** [ ] PASS (4/4)  [ ] FAIL
**Notes:** _________________________________

---

### D7.2: Python Dependencies Declared (2 points)

- [ ] **REQUIRED** - Python packages declared

**YAML Path:** `deployment.dependencies.python_packages`
**Minimum Count:** 1 package

**Status:** [ ] PASS  [ ] FAIL
**Actual Count:** ___
**Notes:** _________________________________

---

### D7.3: GreenLang Dependencies Declared (2 points)

- [ ] **REQUIRED** - GreenLang module dependencies declared

**YAML Path:** `deployment.dependencies.greenlang_modules`
**Minimum Count:** 1 module

**Status:** [ ] PASS  [ ] FAIL
**Actual Count:** ___
**Notes:** _________________________________

---

### D7.4: Resource Requirements (2 points)

- [ ] **REQUIRED** - Resource requirements specified

Required fields:
- [ ] `deployment.resource_requirements.memory_mb`
- [ ] `deployment.resource_requirements.cpu_cores`
- [ ] `deployment.resource_requirements.gpu_required`

**Status:** [ ] PASS (3/3)  [ ] FAIL
**Notes:** _________________________________

---

### D7.5: API Endpoints Defined (1 point)

- [ ] Optional - API endpoints defined

**YAML Path:** `deployment.api_endpoints`
**Minimum Count:** 1 endpoint

**Status:** [ ] PASS  [ ] FAIL
**Actual Count:** ___
**Notes:** _________________________________

---

**D7 TOTAL SCORE:** ___/10

---

## D8: Exit Bar Criteria (10 points)

**Total Possible:** 10 points
**Minimum Required:** 10 points (all criteria are REQUIRED)

### D8.1: All Tests Passing (3 points)

- [ ] **REQUIRED** - Zero test failures

**Command:** `pytest tests/agents/test_{agent_name}_ai.py --tb=short`
**Expected:** All tests pass

**Status:** [ ] PASS  [ ] FAIL
**Failed Count:** ___
**Notes:** _________________________________

---

### D8.2: Test Coverage ≥80% (3 points)

- [ ] **REQUIRED** - Coverage meets threshold

**Command:** `pytest --cov --cov-report=term`
**Expected:** ≥80% coverage

**Status:** [ ] PASS  [ ] FAIL
**Actual Coverage:** ___%
**Notes:** _________________________________

---

### D8.3: No Security Issues (2 points)

- [ ] **REQUIRED** - No critical security issues marked in code

**Pattern:** `TODO.*SECURITY|FIXME.*SECURITY|XXX.*SECURITY`
**Expected:** 0 matches (CLEAN)

**Status:** [ ] PASS  [ ] FAIL
**Issues Found:** ___
**Notes:** _________________________________

---

### D8.4: Specification Validation (2 points)

- [ ] **REQUIRED** - Spec validation passes

**Command:** `python scripts/validate_agent_specs.py {spec_path}`
**Expected:** "0 ERRORS"

**Status:** [ ] PASS  [ ] FAIL
**Notes:** _________________________________

---

**D8 TOTAL SCORE:** ___/10

---

## D9: Integration & Coordination (5 points)

**Total Possible:** 5 points
**Minimum Required:** 5 points (all criteria are REQUIRED)

### D9.1: Dependencies Declared (2 points)

- [ ] **REQUIRED** - Dependencies section present in spec

**YAML Path:** `description.dependencies`
**Validation:** Section exists (can be empty)

**Status:** [ ] PASS  [ ] FAIL
**Notes:** _________________________________

---

### D9.2: BaseAgent Inheritance (2 points)

- [ ] **REQUIRED** - Agent inherits from BaseAgent

**Pattern:** `class.*\(BaseAgent\)`
**Minimum Count:** 1 occurrence

**Status:** [ ] PASS  [ ] FAIL
**Notes:** _________________________________

---

### D9.3: AgentResult Return Type (1 point)

- [ ] **REQUIRED** - Uses AgentResult return type

**Pattern:** `-> AgentResult:` or `return AgentResult(`
**Minimum Count:** 1 occurrence

**Status:** [ ] PASS  [ ] FAIL
**Notes:** _________________________________

---

**D9 TOTAL SCORE:** ___/5

---

## D10: Business Impact & Metrics (5 points)

**Total Possible:** 5 points
**Minimum Required:** 3 points (D10.1, D10.3 are critical)

### D10.1: Strategic Context (2 points)

- [ ] **REQUIRED** - Strategic context documented in spec

**YAML Path:** `description.strategic_context`
**Expected:** Section exists and is complete

**Status:** [ ] PASS  [ ] FAIL
**Notes:** _________________________________

---

### D10.2: Business Impact Section (2 points)

- [ ] Optional - Business impact section present

**YAML Path:** `business_impact`
**Expected:** Section exists with market/carbon/economic data

**Status:** [ ] PASS  [ ] FAIL
**Notes:** _________________________________

---

### D10.3: Performance Requirements (1 point)

- [ ] **REQUIRED** - Performance requirements defined

**YAML Path:** `testing.performance_requirements`
**Expected:** Section exists

**Status:** [ ] PASS  [ ] FAIL
**Notes:** _________________________________

---

**D10 TOTAL SCORE:** ___/5

---

## D11: Operational Excellence (5 points)

**Total Possible:** 5 points
**Minimum Required:** 4 points (D11.3 is optional)

### D11.1: Logging Implementation (2 points)

- [ ] **REQUIRED** - Sufficient logging statements

**Pattern:** `logger.` or `logging.`
**Minimum Count:** 3 log statements

**Status:** [ ] PASS  [ ] FAIL
**Actual Count:** ___
**Notes:** _________________________________

---

### D11.2: Error Handling (2 points)

- [ ] **REQUIRED** - Sufficient error handling

**Pattern:** `try:` or `except.*:`
**Minimum Count:** 2 try-except blocks

**Status:** [ ] PASS  [ ] FAIL
**Actual Count:** ___
**Notes:** _________________________________

---

### D11.3: Performance Tracking (1 point)

- [ ] Optional - Performance metrics tracked

**Pattern:** `_ai_call_count|_tool_call_count|_total_cost_usd|metadata[.*cost`
**Minimum Count:** 1 occurrence

**Status:** [ ] PASS  [ ] FAIL
**Notes:** _________________________________

---

**D11 TOTAL SCORE:** ___/5

---

## D12: Continuous Improvement (5 points)

**Total Possible:** 5 points
**Minimum Required:** 5 points (all criteria are REQUIRED)

### D12.1: Version Control Metadata (2 points)

- [ ] **REQUIRED** - Version control fields present

Required fields:
- [ ] `metadata.version`
- [ ] `metadata.created_date`
- [ ] `metadata.last_modified`

**Status:** [ ] PASS (3/3)  [ ] FAIL
**Notes:** _________________________________

---

### D12.2: Change Log Present (2 points)

- [ ] **REQUIRED** - Change log with entries

**YAML Path:** `metadata.change_log`
**Minimum Count:** 1 entry

**Status:** [ ] PASS  [ ] FAIL
**Actual Count:** ___
**Notes:** _________________________________

---

### D12.3: Review Status (1 point)

- [ ] **REQUIRED** - Review status documented

**YAML Path:** `metadata.review_status`
**Allowed Values:** "Approved", "In Review", "Draft"

**Status:** [ ] PASS  [ ] FAIL
**Actual Value:** ___
**Notes:** _________________________________

---

**D12 TOTAL SCORE:** ___/5

---

## FINAL SCORE SUMMARY

| Dimension | Score | Max | Status |
|-----------|-------|-----|--------|
| D1: Specification | ___/10 | 10 | [ ] PASS [ ] PARTIAL [ ] FAIL |
| D2: Implementation | ___/15 | 15 | [ ] PASS [ ] PARTIAL [ ] FAIL |
| D3: Test Coverage | ___/15 | 15 | [ ] PASS [ ] PARTIAL [ ] FAIL |
| D4: Deterministic AI | ___/10 | 10 | [ ] PASS [ ] PARTIAL [ ] FAIL |
| D5: Documentation | ___/5 | 5 | [ ] PASS [ ] PARTIAL [ ] FAIL |
| D6: Compliance | ___/10 | 10 | [ ] PASS [ ] PARTIAL [ ] FAIL |
| D7: Deployment | ___/10 | 10 | [ ] PASS [ ] PARTIAL [ ] FAIL |
| D8: Exit Bar | ___/10 | 10 | [ ] PASS [ ] PARTIAL [ ] FAIL |
| D9: Integration | ___/5 | 5 | [ ] PASS [ ] PARTIAL [ ] FAIL |
| D10: Business Impact | ___/5 | 5 | [ ] PASS [ ] PARTIAL [ ] FAIL |
| D11: Operations | ___/5 | 5 | [ ] PASS [ ] PARTIAL [ ] FAIL |
| D12: Improvement | ___/5 | 5 | [ ] PASS [ ] PARTIAL [ ] FAIL |
| **TOTAL** | **___/100** | **100** | |

---

## PRODUCTION READINESS ASSESSMENT

**Overall Score:** ___/100

**Readiness Status:**
- [ ] ✅ PRODUCTION READY (≥95)
- [ ] ⚠️ PRE-PRODUCTION (80-94)
- [ ] 🟡 DEVELOPMENT (60-79)
- [ ] 🟠 EARLY DEVELOPMENT (40-59)
- [ ] 🔴 SPECIFICATION ONLY (20-39)
- [ ] ⚫ NOT STARTED (0-19)

**Production Deployment:**
- [ ] APPROVED - Ready for production
- [ ] BLOCKED - See blockers below
- [ ] ON HOLD - Waiting for dependencies

---

## BLOCKERS TO PRODUCTION

List all REQUIRED criteria that failed:

1. ___________________________________
2. ___________________________________
3. ___________________________________
4. ___________________________________
5. ___________________________________

---

## RECOMMENDED ACTIONS

Priority fixes needed to reach production:

1. ___________________________________
2. ___________________________________
3. ___________________________________

---

## TIMELINE ESTIMATE

**Current Score:** ___/100
**Target Score:** 95/100 (production ready)
**Gap:** ___ points

**Estimated Time to Production:**
- [ ] 0 days - Ready now
- [ ] 1-2 weeks - Minor gaps
- [ ] 3-4 weeks - Major gaps
- [ ] 6-8 weeks - Significant work needed
- [ ] 12+ weeks - Early stage

---

## SIGN-OFF

**Evaluator:** ___________________________ **Date:** ___________

**Engineering Lead:** ___________________________ **Date:** ___________

**Security Lead:** ___________________________ **Date:** ___________

**Product Lead:** ___________________________ **Date:** ___________

**Final Approval:** ___________________________ **Date:** ___________

---

## AUTOMATED VALIDATION

**Script:** `python scripts/validate_exit_bar.py --agent {agent_name}`

**Last Run:** ___________________________
**Result:** [ ] PASS [ ] FAIL
**Report:** ___________________________

---

**END OF EXIT BAR CHECKLIST**

*For questions or issues, contact the GreenLang Framework Team*
