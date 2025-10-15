# Coverage Quick Start Guide

## Current Status

**Coverage:** 11.16% (4,265 / 29,809 statements)
**Tests:** 410 tests collected
**Date:** October 13, 2025

---

## View Coverage Reports

### HTML Report (Interactive)

```bash
# Windows
start .coverage_html\index.html

# macOS/Linux
open .coverage_html/index.html

# Or using Python
python -m webbrowser .coverage_html/index.html
```

**Features:**
- Click files to see line-by-line coverage
- Red = uncovered lines
- Green = covered lines
- Navigation: [ ] for prev/next file, ? for help

### Terminal Report

```bash
# Full report
python -m coverage report --skip-empty

# Only show files with gaps
python -m coverage report --skip-covered --sort=cover

# Module-level summary
python analyze_coverage.py
```

---

## Run Tests with Coverage

### Full Suite

```bash
# Using coverage.py (recommended on Windows)
python -m coverage run -m pytest tests/ --maxfail=999 -q
python -m coverage html
python -m coverage report

# Using pytest-cov (may have capture issues)
pytest tests/ --cov=greenlang --cov-report=html --cov-report=term-missing -p no:capture --maxfail=999
```

### By Component

```bash
# Agents only
python -m coverage run -m pytest tests/agents/ -q
python -m coverage report --include="greenlang/agents/*"

# Intelligence only
python -m coverage run -m pytest tests/intelligence/ -q
python -m coverage report --include="greenlang/intelligence/*"

# CLI only
python -m coverage run -m pytest tests/cli/ -q
python -m coverage report --include="greenlang/cli/*"
```

---

## Top Priorities

### 1. CLI Module (6.22% coverage)
**Gap:** 5,235 statements
**Start with:** `tests/cli/test_cmd_doctor.py`

### 2. Intelligence Module (17.03% coverage)
**Gap:** 4,467 statements
**Start with:** Provider tests (OpenAI, Anthropic)

### 3. Agents Module (21.95% coverage)
**Gap:** 2,574 statements
**Start with:** Fuel agent, Intensity agent

### 4. Monitoring (0% coverage)
**Gap:** 718 statements
**Start with:** Health checks

### 5. Telemetry (0% coverage)
**Gap:** 1,408 statements
**Start with:** Basic metrics

---

## Files Needing Tests

### Zero Coverage (Critical)

```
greenlang/cli/main_old.py (328 statements)
greenlang/cli/telemetry.py (334 statements)
greenlang/benchmarks/performance_suite.py (340 statements)
greenlang/monitoring/health.py (360 statements)
greenlang/monitoring/metrics.py (274 statements)
```

### Low Coverage (High Priority)

```
greenlang/intelligence/providers/openai.py (0%)
greenlang/intelligence/providers/anthropic.py (0%)
greenlang/agents/fuel_agent.py (0%)
greenlang/agents/boiler_agent.py (0%)
greenlang/runtime/executor.py (0%)
```

---

## Next Steps

1. **Week 1:** Add CLI tests (doctor, init, pack) → 25% coverage
2. **Week 2:** Add agent tests (fuel, intensity, validator) → 35% coverage
3. **Week 3-4:** Intelligence providers & runtime → 50% coverage
4. **Week 5-10:** Monitoring, telemetry, complete system → 80%+ coverage

---

## Key Files

- **Full Report:** `COVERAGE_BASELINE_REPORT.md`
- **HTML Report:** `.coverage_html/index.html`
- **JSON Data:** `coverage.json`
- **Analysis Script:** `analyze_coverage.py`
- **Coverage Config:** `.coveragerc` (if exists)

---

## Troubleshooting

### Pytest Capture Error

```bash
# Use this workaround
python -m coverage run -m pytest tests/ --maxfail=999 -q

# Or disable capture
pytest tests/ --cov=greenlang -p no:capture
```

### Old Coverage Data

```bash
# Clear old data
python -m coverage erase

# Regenerate
python -m coverage run -m pytest tests/ -q
python -m coverage html
```

### Import Errors

Fixed in this baseline:
- ProviderInfo now exported from intelligence module
- All 410 tests now collect successfully

---

**Last Updated:** October 13, 2025
**Next Review:** After Phase 1 (Week 2)
