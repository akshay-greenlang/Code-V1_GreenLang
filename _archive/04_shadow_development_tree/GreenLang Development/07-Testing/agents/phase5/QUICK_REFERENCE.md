# Phase 5 Compliance Tests - QUICK REFERENCE

## 🚀 Quick Start

### Validate Compliance (No pytest needed)
```bash
cd tests/agents/phase5
python validate_compliance.py
```

### Run All Tests
```bash
pytest tests/agents/phase5/test_critical_path_compliance.py -v
```

## 📊 Test Categories

| Category | Tests | Purpose |
|----------|-------|---------|
| **Determinism** | 9 | Byte-for-byte identical outputs |
| **No LLM** | 7 | Zero AI dependencies |
| **Performance** | 5 | <10ms execution time |
| **Deprecation** | 3 | Warnings for AI versions |
| **Audit Trail** | 7 | Complete provenance |
| **Reproducibility** | 4 | Cross-session consistency |
| **Integration** | 2 | End-to-end workflow |
| **Summary** | 1 | Compliance report |
| **TOTAL** | **38** | **All compliance requirements** |

## 🎯 Critical Path Agents

| Agent | Tests | Status |
|-------|-------|--------|
| `FuelAgent` | 12 | ✅ CRITICAL PATH |
| `GridFactorAgent` | 8 | ✅ CRITICAL PATH |
| `BoilerAgent` | 6 | ✅ CRITICAL PATH |
| `CarbonAgent` | 4 | ✅ CRITICAL PATH |

## 🔍 Key Tests

### Test Determinism
```bash
pytest tests/agents/phase5/ -v -k "determinism"
```
**Purpose**: Verify identical outputs (10 iterations per test)

### Test Performance
```bash
pytest tests/agents/phase5/ -v -k "performance"
```
**Target**: <10ms per calculation (100x faster than AI)

### Test No LLM Dependencies
```bash
pytest tests/agents/phase5/ -v -k "llm"
```
**Purpose**: Verify no ChatSession, RAG, or API calls

### Test Audit Trails
```bash
pytest tests/agents/phase5/ -v -k "audit"
```
**Purpose**: Verify complete provenance for SOC 2 / ISO 14064-1

## 📈 Expected Performance

```
FuelAgent:        ~3ms   (target: <10ms) ✅
GridFactorAgent:  ~2ms   (target: <10ms) ✅
BoilerAgent:      ~5ms   (target: <10ms) ✅
CarbonAgent:      ~1ms   (target: <10ms) ✅

AI Version:     ~1000ms
Deterministic:    ~3ms
Speedup:        300x+ ✅
```

## ✅ Compliance Checklist

- [ ] All 38 tests pass
- [ ] Average execution time <10ms
- [ ] Zero LLM dependencies
- [ ] Complete audit trails present
- [ ] Determinism validated (10 iterations)
- [ ] Deprecation warnings working
- [ ] Integration tests pass
- [ ] Documentation complete

## 🚨 What If Tests Fail?

### Non-Deterministic Output
```
FAILED test_fuel_agent_determinism_natural_gas
```
**Fix**: Check for random numbers, timestamps in calculations, or floating-point issues

### Performance Too Slow
```
FAILED test_fuel_agent_performance_target
AssertionError: 15.23ms (target: <10ms)
```
**Fix**: Profile code, add caching, optimize database lookups

### LLM Dependency Detected
```
FAILED test_fuel_agent_no_chatsession_import
```
**Fix**: Remove ChatSession imports, move AI code to separate AI agent

### Missing Audit Trail
```
FAILED test_fuel_agent_audit_trail_completeness
```
**Fix**: Add complete metadata to results (agent_id, calculation, version)

## 📁 Files

```
tests/agents/phase5/
├── test_critical_path_compliance.py  # 38 tests (1,176 lines)
├── conftest.py                       # 16 fixtures (318 lines)
├── validate_compliance.py            # Quick validation script
├── README.md                         # Complete documentation
├── QUICK_REFERENCE.md                # This file
└── PHASE_5_COMPLIANCE_TEST_DELIVERY.md  # Delivery report
```

## 🎓 Regulatory Standards

- ✅ **ISO 14064-1**: GHG Accounting (deterministic calculations)
- ✅ **GHG Protocol**: Corporate Standard (transparent methodology)
- ✅ **SOC 2 Type II**: Deterministic Controls (audit trails)

## 💡 Pro Tips

1. **Run determinism tests first** - They catch the most critical issues
2. **Use `-v` flag** - Shows detailed test output
3. **Use `-s` flag** - Shows print statements for debugging
4. **Run validate_compliance.py** - Quick check without full pytest
5. **Check compliance summary** - Shows overview of all requirements

## 📞 Support

- See: `README.md` - Full documentation
- See: `PHASE_5_COMPLIANCE_TEST_DELIVERY.md` - Delivery report
- See: `AGENT_CATEGORIZATION_AUDIT.md` - Agent categorization
- See: `AGENT_PATTERNS_GUIDE.md` - Agent patterns

---

**Quick Commands**:
```bash
# Quick validation (no pytest)
python validate_compliance.py

# All tests
pytest tests/agents/phase5/ -v

# Specific category
pytest tests/agents/phase5/ -v -k "determinism"

# With output
pytest tests/agents/phase5/ -v -s

# Stop on first failure
pytest tests/agents/phase5/ -v -x
```
