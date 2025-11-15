# GL-002 Code Quality Remediation Checklist

**Assessment Date:** 2025-11-15
**Last Updated:** 2025-11-15
**Status:** READY FOR IMPLEMENTATION

---

## Pre-Remediation Setup

### Preparation Tasks
- [ ] Team meeting scheduled to discuss findings
- [ ] All 4 quality reports read by team
- [ ] Development environment set up with quality tools
- [ ] Git branch created for fixes (`feature/code-quality-fixes`)
- [ ] Pre-commit hooks installed
- [ ] CI/CD pipeline updated for quality gates

### Tools Installation
```bash
# Install required tools
pip install mypy pyright pylint flake8 black

# Install pre-commit
pip install pre-commit

# Configure pre-commit hooks
pre-commit install
```

---

## Phase 1: Critical Blocking Issues (4 hours)

### Task 1.1: Fix Broken Relative Imports
**Time:** 15 minutes | **Status:** [ ] TODO | **Owner:** [ ]

Files to fix:
- [ ] `calculators/blowdown_optimizer.py` (line 15)
- [ ] `calculators/combustion_efficiency.py` (line 15)
- [ ] `calculators/control_optimization.py` (line 15)
- [ ] `calculators/economizer_performance.py` (line 15)
- [ ] `calculators/emissions_calculator.py` (line 16)
- [ ] `calculators/fuel_optimization.py` (line 16)
- [ ] `calculators/heat_transfer.py` (line 15)
- [ ] `calculators/steam_generation.py` (line 16)

Changes required:
- [ ] Replace `from provenance import` with `from .provenance import`
- [ ] Test each file imports correctly
- [ ] Run full test suite

Verification:
```bash
cd calculators
python -c "from combustion_efficiency import CombustionEfficiencyCalculator"
# Should import without error
```

---

### Task 1.2: Remove Hardcoded Credentials
**Time:** 30 minutes | **Status:** [ ] TODO | **Owner:** [ ]

Files to fix:
- [ ] `tests/test_integrations.py` (auth_token, access_token)
- [ ] `tests/test_security.py` (password, api_key)

Changes required:
- [ ] Create `.env.test` template with placeholders
- [ ] Update test code to use `os.getenv()` with defaults
- [ ] Add `.env.test` and `.env.*.local` to `.gitignore`
- [ ] Create pre-commit hook to detect credential patterns

Verification:
```bash
grep -r "password\|api_key\|auth_token\|secret" tests/ | grep "=\s*['\"]"
# Should return nothing
```

---

### Task 1.3: Add Thread-Safe Cache with Locking
**Time:** 2-3 hours | **Status:** [ ] TODO | **Owner:** [ ]

File: `boiler_efficiency_orchestrator.py`

Changes required:
- [ ] Import RLock from threading
- [ ] Import OrderedDict from collections
- [ ] Add `self._cache_lock = RLock()` to `__init__`
- [ ] Add `self._metrics_lock = RLock()` to `__init__`
- [ ] Replace `self._results_cache = {}` with `OrderedDict()`
- [ ] Update `_analyze_operational_state_async` to use cache lock
- [ ] Update `_optimize_combustion_async` to use cache lock
- [ ] Update all other cache operations with locking
- [ ] Update `_update_performance_metrics` with metrics lock
- [ ] Update `_store_in_cache` with lock and FIFO eviction

Verification:
```bash
python -m pytest tests/test_concurrency.py -v
# Should pass concurrency tests
```

---

### Task 1.4: Add Constraint Validation
**Time:** 2 hours | **Status:** [ ] TODO | **Owner:** [ ]

File: `config.py` (OperationalConstraints class)

Changes required:
- [ ] Add validator for `max_pressure_bar >= min_pressure_bar`
- [ ] Add validator for `max_temperature_c >= min_temperature_c`
- [ ] Add validator for `max_excess_air_percent >= min_excess_air_percent`
- [ ] Add validator for `max_load_percent >= min_load_percent`
- [ ] Add test cases for invalid configurations

Verification:
```bash
python -m pytest tests/test_validation.py -v -k constraint
# Should pass all constraint validation tests
```

---

## Phase 2: Type Hints (10 hours)

### Task 2.1: Add Type Hints to Main Orchestrator
**Time:** 3 hours | **Status:** [ ] TODO | **Owner:** [ ]

File: `boiler_efficiency_orchestrator.py`

Changes required:
- [ ] Add return type hints to all public methods
- [ ] Add parameter type hints to all public methods
- [ ] Add return type hints to all helper methods (starting with public ones)
- [ ] Add parameter type hints to all helper methods
- [ ] Fix any type inconsistencies

Verification:
```bash
mypy boiler_efficiency_orchestrator.py --strict
# Should show 0 errors
```

---

### Task 2.2: Add Type Hints to Tools Module
**Time:** 3 hours | **Status:** [ ] TODO | **Owner:** [ ]

File: `tools.py`

Changes required:
- [ ] Add return types to all 50+ methods
- [ ] Add parameter types to all method signatures
- [ ] Ensure consistency with dataclass definitions
- [ ] Fix import statements if needed

Verification:
```bash
mypy tools.py --strict
# Should show 0 errors
```

---

### Task 2.3: Add Type Hints to Calculator Modules
**Time:** 2 hours | **Status:** [ ] TODO | **Owner:** [ ]

Files: All `calculators/*.py` (8 files)

Changes required:
- [ ] For each calculator module:
  - [ ] Add return types to calculate methods
  - [ ] Add parameter types to all methods
  - [ ] Update dataclass definitions if needed

Verification:
```bash
mypy calculators/ --strict
# Should show 0 errors for all calculator modules
```

---

### Task 2.4: Add Type Hints to Integration Modules
**Time:** 2 hours | **Status:** [ ] TODO | **Owner:** [ ]

Files: All `integrations/*.py` (6 files)

Changes required:
- [ ] For each integration module:
  - [ ] Add return types to all public methods
  - [ ] Add parameter types to all method signatures
  - [ ] Update dataclass definitions if needed

Verification:
```bash
mypy integrations/ --strict
# Should show 0 errors for all integration modules
```

---

### Task 2.5: Verify All Type Hints with Type Checker
**Time:** 1 hour | **Status:** [ ] TODO | **Owner:** [ ]

Changes required:
- [ ] Run mypy on entire project
- [ ] Resolve all remaining type errors
- [ ] Configure mypy to strict mode in setup.cfg
- [ ] Add type checking to CI/CD pipeline

Verification:
```bash
mypy . --strict --ignore-missing-imports
# Should show 0 errors (except missing imports)

pyright .
# Should show 0 errors
```

---

## Phase 3: Validation & Error Handling (6-8 hours)

### Task 3.1: Add Timeout Enforcement
**Time:** 2 hours | **Status:** [ ] TODO | **Owner:** [ ]

File: `boiler_efficiency_orchestrator.py`

Methods to update:
- [ ] `_analyze_operational_state_async` (lines 356-360)
- [ ] `_optimize_combustion_async` (lines 418-423)
- [ ] `_optimize_steam_generation_async` (lines 448-453)
- [ ] `_minimize_emissions_async` (lines 476-480)
- [ ] `_calculate_parameter_adjustments_async` (lines 505-510)
- [ ] `_coordinate_agents_async` (lines 731-736)

Changes required:
- [ ] Wrap each `asyncio.to_thread` call with `asyncio.wait_for`
- [ ] Use `self.boiler_config.calculation_timeout_seconds` for timeout value
- [ ] Add proper TimeoutError handling
- [ ] Log timeout events

Verification:
```bash
python -m pytest tests/test_timeouts.py -v
# Should pass timeout enforcement tests
```

---

### Task 3.2: Add Input Validation - Constraints
**Time:** 2 hours | **Status:** [ ] TODO | **Owner:** [ ]

Files: Multiple files with constraint usage

Changes required:
- [ ] Add validation in `_apply_safety_constraints` method
- [ ] Verify all constraint fields are used correctly
- [ ] Check for division by zero with constraint values
- [ ] Add bounds checking for constraint violations

Verification:
```bash
python -m pytest tests/test_validation.py -v -k constraints
# Should pass constraint validation tests
```

---

### Task 3.3: Add Null/None Checks
**Time:** 2 hours | **Status:** [ ] TODO | **Owner:** [ ]

File: `integrations/data_transformers.py` (priority)

Changes required:
- [ ] Add None checks in `process_scada_data`
- [ ] Add None checks in `process_dcs_data`
- [ ] Add None checks in `convert_temperature`
- [ ] Add None checks in `convert_pressure`
- [ ] Add None checks in `convert_flow_rate`
- [ ] Add None checks in all data transformation methods

Verification:
```bash
python -m pytest tests/test_data_validation.py -v
# Should pass null check tests
```

---

### Task 3.4: Add Sensor Data Validation
**Time:** 2 hours | **Status:** [ ] TODO | **Owner:** [ ]

File: `tools.py`

Changes required:
- [ ] Add range validation in `calculate_boiler_efficiency`
- [ ] Validate fuel_flow > 0
- [ ] Validate steam_flow >= 0
- [ ] Validate stack_temp > ambient_temp
- [ ] Validate o2_percent in valid range (0-21%)
- [ ] Validate pressure values are positive
- [ ] Add informative error messages

Verification:
```bash
python -m pytest tests/test_sensor_validation.py -v
# Should pass sensor validation tests
```

---

## Phase 4: Testing & Verification (4 hours)

### Task 4.1: Update Unit Tests
**Time:** 1 hour | **Status:** [ ] TODO | **Owner:** [ ]

Changes required:
- [ ] Update tests to use new validation
- [ ] Add tests for invalid constraint combinations
- [ ] Add tests for timeout scenarios
- [ ] Add tests for None/null inputs
- [ ] Add tests for sensor data edge cases

Verification:
```bash
python -m pytest tests/test_boiler_efficiency_orchestrator.py -v
python -m pytest tests/test_tools.py -v
python -m pytest tests/test_calculators.py -v
# All should pass
```

---

### Task 4.2: Add Concurrency Tests
**Time:** 1 hour | **Status:** [ ] TODO | **Owner:** [ ]

New test file: `tests/test_concurrency.py`

Tests to add:
- [ ] Test 100 concurrent cache operations
- [ ] Test cache size limits under concurrent load
- [ ] Test metrics accuracy under concurrent updates
- [ ] Test no data corruption with concurrent access

Verification:
```bash
python -m pytest tests/test_concurrency.py -v
# All concurrency tests should pass
```

---

### Task 4.3: Full Test Suite Run
**Time:** 1 hour | **Status:** [ ] TODO | **Owner:** [ ]

Changes required:
- [ ] Run complete test suite
- [ ] Verify all 9 test files pass
- [ ] Check code coverage
- [ ] Verify no warnings

Verification:
```bash
python -m pytest -v --cov=. --cov-report=html
# All tests pass, coverage >80%
```

---

### Task 4.4: Type Checking Validation
**Time:** 0.5 hour | **Status:** [ ] TODO | **Owner:** [ ]

Changes required:
- [ ] Run mypy with strict mode
- [ ] Run pyright in strict mode
- [ ] Resolve all reported errors
- [ ] Document any necessary type: ignore comments

Verification:
```bash
mypy . --strict --ignore-missing-imports
pyright . --outputjson
# Both should show 0 errors
```

---

## Additional Tasks (High Priority)

### Task A.1: Reduce File Complexity
**Time:** 1-2 days | **Status:** [ ] TODO | **Owner:** [ ]

Files to refactor (>600 lines):
- [ ] `boiler_efficiency_orchestrator.py` (1,123 lines)
  - [ ] Split into orchestrator.py + state_management.py
- [ ] `integrations/data_transformers.py` (1,301 lines)
  - [ ] Split into separate converter modules
- [ ] `integrations/agent_coordinator.py` (1,105 lines)
  - [ ] Split into coordinator.py + task_manager.py

---

### Task A.2: Improve Error Recovery
**Time:** 3 hours | **Status:** [ ] TODO | **Owner:** [ ]

File: `boiler_efficiency_orchestrator.py`

Changes required:
- [ ] Add full traceback to error recovery
- [ ] Add error type classification
- [ ] Add error context preservation
- [ ] Improve error messages with actionable info

---

### Task A.3: Create Constants Module
**Time:** 2 hours | **Status:** [ ] TODO | **Owner:** [ ]

New file: `constants.py`

Constants to extract:
- [ ] Default sensor values
- [ ] Safety limit constants
- [ ] Cache settings
- [ ] Timeout values
- [ ] Conversion factors

---

### Task A.4: Add Pre-commit Hooks
**Time:** 1 hour | **Status:** [ ] TODO | **Owner:** [ ]

Setup required:
- [ ] Create `.pre-commit-config.yaml`
- [ ] Add mypy check
- [ ] Add flake8 check
- [ ] Add credential scanner
- [ ] Add import sort check

---

## Success Criteria Checklist

### Code Quality
- [ ] All type checkers (mypy/pyright) report 0 errors
- [ ] No hardcoded credentials in any file
- [ ] All imports resolve correctly
- [ ] All functions have complete type hints
- [ ] No bare except statements

### Reliability
- [ ] No race conditions in concurrent operations
- [ ] All constraints validated at instantiation
- [ ] All async operations have timeout enforcement
- [ ] All inputs validated (None checks)
- [ ] Error messages include full context

### Testing
- [ ] Full test suite passes (100%)
- [ ] Concurrency tests pass
- [ ] Integration tests pass
- [ ] Type checking validation passes
- [ ] Pre-commit hooks pass
- [ ] Code coverage >80%

### Documentation
- [ ] All code quality reports show PASS
- [ ] No outstanding issues in FIXES_REQUIRED.md
- [ ] Team sign-off obtained
- [ ] Ready for production deployment

---

## Sign-Off

### Developer Sign-Off
- [ ] Developer: _________________ Date: _______
- [ ] Fixes implemented and tested

### Code Review Sign-Off
- [ ] Reviewer: _________________ Date: _______
- [ ] Code review completed
- [ ] All issues resolved

### QA Sign-Off
- [ ] QA Lead: _________________ Date: _______
- [ ] Full testing completed
- [ ] Quality gates passed

### Project Manager Sign-Off
- [ ] PM: _________________ Date: _______
- [ ] Ready for production deployment

---

## Timeline

### Estimated Schedule (assuming full-time commitment)

**Week 1:**
- Monday-Tuesday: Phase 1 (Critical Fixes) - 8 hours
- Wednesday: Phase 2a (Main orchestrator type hints) - 3 hours
- Thursday-Friday: Phase 2b-2c (Tools and calculators) - 6 hours

**Week 2:**
- Monday-Wednesday: Phase 2d-2e (Integrations and final verification) - 6 hours
- Thursday: Phase 3 (Validation & error handling) - 6 hours
- Friday: Phase 4 (Testing) - 4 hours

**Week 3:**
- Monday-Wednesday: Refactoring large files - 6-8 hours
- Thursday-Friday: Final verification and sign-off - 4 hours

**Week 4:**
- Monday: Code review and adjustments - 4 hours
- Tuesday: Production deployment prep - 2 hours
- Wednesday: Production deployment readiness - 2 hours

---

## Notes & Comments

Use this section for tracking issues and insights during remediation:

```
Date: ______
Task: ______________________
Issue: _____________________
Resolution: _________________

---

Date: ______
Task: ______________________
Issue: _____________________
Resolution: _________________
```

---

## References

- CODE_QUALITY_REPORT.md - Detailed analysis (1,053 lines)
- FIXES_REQUIRED.md - Implementation guide (645 lines)
- QUALITY_ASSESSMENT_SUMMARY.txt - Executive summary (435 lines)
- QUALITY_VALIDATION_INDEX.md - Navigation guide (432 lines)

---

**Created:** 2025-11-15
**Last Updated:** 2025-11-15
**Status:** READY FOR IMPLEMENTATION
**Next Review:** After Phase 1 completion
