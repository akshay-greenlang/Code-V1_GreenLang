# PRD: Logging Standardization Validation

**Status**: Draft
**Priority**: High
**Component**: Infrastructure - Logging (INFRA-009)
**Created**: 2026-04-02
**Owner**: Platform Team

## Overview

This PRD defines validation tasks to verify the logging standardization migration across the GreenLang codebase. The migration moves from f-string logging and instance-level `self.logger` to standardized module-level loggers with %-formatting and structured logging support.

## Background

The logging standardization initiative addresses:
- **Performance**: F-string logging evaluates strings even when logs are filtered
- **Consistency**: Mixed usage of `self.logger` vs module-level loggers
- **Integration**: Proper integration with INFRA-009 structured logging system
- **Best Practices**: Alignment with Python logging library recommendations (%-formatting for lazy evaluation)

## Validation Tasks

### Task 1: Verify F-String Logging Elimination

**Objective**: Ensure f-string logging has been eliminated from production code.

**Verification Command**:
```bash
grep -r "logger\.\w\+(f[\"']" greenlang/ --include="*.py" -l | wc -l
```

**Expected Outcome**:
- Count < 5 files remaining
- Remaining files should ONLY be:
  - Template generators (files containing `def generate_` or `_template.py`)
  - Test fixtures/templates (files in `tests/` or containing `test_`)

**Validation Steps**:
1. Run the grep command to count remaining f-string log calls
2. List all files with remaining f-strings:
   ```bash
   grep -r "logger\.\w\+(f[\"']" greenlang/ --include="*.py" -l
   ```
3. Manually inspect each file to verify it's a legitimate exception
4. Report any production code files that still use f-string logging

**Success Criteria**:
- ✅ Count ≤ 5 files
- ✅ All remaining files are templates, generators, or test code
- ✅ Zero production agent/engine/service files with f-string logging

---

### Task 2: Verify self.logger Standardization

**Objective**: Ensure all agent code uses module-level loggers instead of instance-level `self.logger`.

**Verification Command**:
```bash
grep -r "self\.logger\.\w\+(" greenlang/ --include="*.py" -l | wc -l
```

**Expected Outcome**:
- Count < 5 files remaining
- All agent files should use `logger = logging.getLogger(__name__)`

**Validation Steps**:
1. Run the grep command to count remaining self.logger usage
2. List all files with self.logger:
   ```bash
   grep -r "self\.logger\.\w\+(" greenlang/ --include="*.py" -l
   ```
3. Verify remaining files are legacy/deprecated code or intentional exceptions
4. Check agent base classes don't set `self.logger` in `__init__`

**Success Criteria**:
- ✅ Count ≤ 5 files
- ✅ No active agent implementation uses `self.logger`
- ✅ BaseAgent and derived base classes use module-level loggers

---

### Task 3: Verify Module-Level Logger Consistency

**Objective**: Ensure standardized logger pattern is consistently applied across agent code.

**Sampling Strategy**: Random sample of 20 Python files from `greenlang/agents/`

**Verification Pattern**:
Each sampled file should have:
```python
import logging
# ... other imports ...

logger = logging.getLogger(__name__)

# ... rest of code with logger.info(), logger.error(), etc ...
```

**Validation Steps**:
1. Generate random sample:
   ```bash
   find greenlang/agents/ -name "*.py" -type f | shuf -n 20
   ```
2. For each sampled file, verify:
   - Has `import logging` in top-level imports
   - Has `logger = logging.getLogger(__name__)` at module level (before class definitions)
   - No `self.logger` usage in any method
   - Uses `logger.info()`, `logger.error()`, etc. directly

**Success Criteria**:
- ✅ 100% of sampled files follow the pattern
- ✅ No `self.logger` in any sampled file
- ✅ All logging calls use module-level `logger` variable

---

### Task 4: Verify Conversion Quality

**Objective**: Ensure f-string to %-format conversion was done correctly without syntax errors.

**Sampling Strategy**: 10 files that were converted (identified by git history or known conversion list)

**Verification Checks** (per file):

1. **Syntax Correctness**: No Python syntax errors
   ```bash
   python -m py_compile <file_path>
   ```

2. **Placeholder Matching**: Number of `%s`/`%d`/`%r` placeholders matches argument count
   - Example CORRECT: `logger.info("Processing %s items", count)`
   - Example WRONG: `logger.info("Processing %s items from %s")`  # missing second arg

3. **No Double-Conversion**: Check for artifacts like:
   - `%s %s` with single argument
   - `%%s` (escaped percent signs where not intended)
   - Missing arguments after conversion

4. **exc_info Preservation**: Lines with `exc_info=True` before conversion still have it
   - Example: `logger.exception()` calls should still use exception formatting
   - Example: `logger.error("Error: %s", str(e), exc_info=True)` preserved

**Validation Steps**:
1. Identify 10 converted files (check git diff or conversion log)
2. For each file:
   - Run `python -m py_compile` to verify syntax
   - Grep for logging calls: `grep "logger\.\w\+(" <file> -n`
   - Manually inspect each logging call for placeholder/argument match
   - Check for exc_info usage: `grep "exc_info" <file>`
3. Document any issues found

**Success Criteria**:
- ✅ All 10 files compile without syntax errors
- ✅ 100% of logging calls have correct placeholder-to-argument ratio
- ✅ No double-conversion artifacts found
- ✅ All exc_info usage preserved correctly

---

### Task 5: Verify INFRA-009 Integration

**Objective**: Ensure the structured logging infrastructure (INFRA-009) is properly integrated and importable.

**Required Components**:
```
greenlang/infrastructure/logging/
├── __init__.py          # Exports: configure_logging, get_logger, LoggingConfig
├── setup.py             # Functions: configure_logging(), get_logger()
├── config.py            # Class: LoggingConfig (Pydantic model)
├── context.py           # Functions: bind_context(), bind_agent_context()
├── redaction.py         # Class: RedactionProcessor
├── middleware.py        # Class: StructuredLoggingMiddleware (FastAPI/Starlette)
└── formatters.py        # Classes: JsonFormatter, ConsoleFormatter
```

**Validation Steps**:

1. **File Existence Check**:
   ```bash
   ls -la greenlang/infrastructure/logging/
   ```
   Verify all 7 files exist.

2. **Import Check**:
   ```bash
   python -c "from greenlang.infrastructure.logging import configure_logging, get_logger, LoggingConfig"
   ```
   Should succeed with no import errors.

3. **Component Verification**:
   ```bash
   # Check setup.py exports
   python -c "from greenlang.infrastructure.logging.setup import configure_logging, get_logger; print('OK')"

   # Check config.py exports
   python -c "from greenlang.infrastructure.logging.config import LoggingConfig; print('OK')"

   # Check context.py exports
   python -c "from greenlang.infrastructure.logging.context import bind_context, bind_agent_context; print('OK')"

   # Check redaction.py exports
   python -c "from greenlang.infrastructure.logging.redaction import RedactionProcessor; print('OK')"

   # Check middleware.py exports
   python -c "from greenlang.infrastructure.logging.middleware import StructuredLoggingMiddleware; print('OK')"

   # Check formatters.py exports
   python -c "from greenlang.infrastructure.logging.formatters import JsonFormatter, ConsoleFormatter; print('OK')"
   ```

4. **Functional Test**:
   ```python
   from greenlang.infrastructure.logging import configure_logging, get_logger

   configure_logging()
   logger = get_logger(__name__)
   logger.info("Test message: %s", "INFRA-009 integration verified")
   ```

**Success Criteria**:
- ✅ All 7 required files exist
- ✅ All imports succeed without errors
- ✅ configure_logging() and get_logger() are callable
- ✅ Test logging call produces output

---

### Task 6: Verify No Import Breakage

**Objective**: Ensure logging changes didn't introduce import errors in core modules.

**Critical Import Paths**:
```python
# Agent modules
from greenlang.agents import data_processor
from greenlang.agents.base import BaseAgent
from greenlang.agents.foundation import orchestrator
from greenlang.agents.data import pdf_extractor
from greenlang.agents.mrv import stationary_combustion
from greenlang.agents.eudr import traceability_validator

# Infrastructure
from greenlang.infrastructure.logging import configure_logging
from greenlang.infrastructure.database import get_db_connection
from greenlang.infrastructure.cache import RedisCache

# Applications
from greenlang.applications.ghg_app import GHGApplication
from greenlang.applications.eudr_app import EUDRApplication
```

**Validation Steps**:

1. **Agent Imports**:
   ```bash
   python -c "from greenlang.agents import data_processor; print('data_processor OK')"
   python -c "from greenlang.agents.base import BaseAgent; print('BaseAgent OK')"
   python -c "from greenlang.agents.foundation import orchestrator; print('orchestrator OK')"
   python -c "from greenlang.agents.data import pdf_extractor; print('pdf_extractor OK')"
   python -c "from greenlang.agents.mrv import stationary_combustion; print('stationary_combustion OK')"
   python -c "from greenlang.agents.eudr import traceability_validator; print('traceability_validator OK')"
   ```

2. **Infrastructure Imports**:
   ```bash
   python -c "from greenlang.infrastructure.logging import configure_logging; print('logging OK')"
   python -c "from greenlang.infrastructure.database import get_db_connection; print('database OK')"
   python -c "from greenlang.infrastructure.cache import RedisCache; print('cache OK')"
   ```

3. **Application Imports**:
   ```bash
   python -c "from greenlang.applications.ghg_app import GHGApplication; print('GHG OK')"
   python -c "from greenlang.applications.eudr_app import EUDRApplication; print('EUDR OK')"
   ```

4. **Run Smoke Tests** (if available):
   ```bash
   pytest tests/unit/test_base_agent.py -v
   pytest tests/unit/infrastructure/test_logging.py -v
   ```

**Success Criteria**:
- ✅ All agent imports succeed
- ✅ All infrastructure imports succeed
- ✅ All application imports succeed
- ✅ Smoke tests pass (if available)

---

### Task 7: Generate Summary Report

**Objective**: Provide quantitative metrics on the logging standardization migration.

**Metrics to Collect**:

1. **Module-Level Logger Adoption**:
   ```bash
   grep -r "logger = logging.getLogger(__name__)" greenlang/ --include="*.py" -l | wc -l
   ```

2. **Remaining F-String Logging**:
   ```bash
   grep -r "logger\.\w\+(f[\"']" greenlang/ --include="*.py" -l | wc -l
   grep -r "logger\.\w\+(f[\"']" greenlang/ --include="*.py" -l
   ```

3. **Remaining self.logger Usage**:
   ```bash
   grep -r "self\.logger\.\w\+(" greenlang/ --include="*.py" -l | wc -l
   grep -r "self\.logger\.\w\+(" greenlang/ --include="*.py" -l
   ```

4. **Total Python Files in Scope**:
   ```bash
   find greenlang/ -name "*.py" -type f | wc -l
   ```

5. **Logging Coverage**:
   - Files with any logging calls: `grep -r "logger\.\w\+(" greenlang/ --include="*.py" -l | wc -l`
   - Files with standardized logging: Count from metric #1

**Report Format**:

```markdown
# Logging Standardization Migration Summary

**Date**: 2026-04-02
**Scope**: greenlang/ directory

## Metrics

| Metric | Count | Notes |
|--------|-------|-------|
| Total Python Files | {count} | All .py files in greenlang/ |
| Files with Module-Level Loggers | {count} | `logger = logging.getLogger(__name__)` |
| Files with Any Logging | {count} | Any `logger.*()` call |
| **Adoption Rate** | **{percentage}%** | (Module-level / Total with logging) |
| | | |
| **Remaining Issues** | | |
| Files with F-String Logging | {count} | Should be ≤ 5 |
| Files with self.logger | {count} | Should be ≤ 5 |

## F-String Logging Remaining
{list of files or "None - All eliminated"}

## self.logger Usage Remaining
{list of files or "None - All standardized"}

## Validation Status

| Task | Status | Details |
|------|--------|---------|
| Task 1: F-String Elimination | {✅/❌} | {details} |
| Task 2: self.logger Standardization | {✅/❌} | {details} |
| Task 3: Module-Level Consistency | {✅/❌} | {details} |
| Task 4: Conversion Quality | {✅/❌} | {details} |
| Task 5: INFRA-009 Integration | {✅/❌} | {details} |
| Task 6: Import Breakage Check | {✅/❌} | {details} |

## Recommendations

{Any follow-up actions needed based on findings}

## Conclusion

{Overall assessment: PASS/FAIL with reasoning}
```

**Validation Steps**:
1. Run all metric collection commands
2. Populate the report template with actual values
3. Calculate adoption percentage
4. List any remaining issues
5. Provide pass/fail status for each task
6. Generate recommendations for any failures

**Success Criteria**:
- ✅ Report generated with all metrics populated
- ✅ Adoption rate > 95%
- ✅ F-string count ≤ 5
- ✅ self.logger count ≤ 5
- ✅ All 6 validation tasks show ✅ status

---

## Overall Success Criteria

The logging standardization migration is considered **COMPLETE** when:

1. ✅ All 7 tasks pass their individual success criteria
2. ✅ No import breakage in core modules
3. ✅ INFRA-009 integration fully functional
4. ✅ Adoption rate ≥ 95% across greenlang/
5. ✅ Summary report shows < 5 remaining issues total
6. ✅ Code quality checks (syntax, placeholder matching) at 100%

## Timeline

- **Task Execution**: 2-3 hours
- **Issue Resolution**: 1-2 days (if issues found)
- **Final Sign-Off**: Upon all tasks passing

## Dependencies

- Python 3.11+ environment
- Access to greenlang/ codebase
- Git history (for identifying converted files in Task 4)
- pytest (for smoke tests in Task 6)

## Deliverables

1. Completed validation report (Task 7 output)
2. List of any remaining issues with remediation plan
3. Sign-off confirmation from Platform Team

---

**End of PRD**
