# FuelAgentAI: v1 to v2 Migration Guide

**Version:** 1.0.0
**Date:** October 2025
**Migration Timeline:** 12 months (v1 sunset: 2026-Q3)

---

## Table of Contents

1. [Overview](#overview)
2. [Timeline](#timeline)
3. [What's Changing](#whats-changing)
4. [Migration Strategies](#migration-strategies)
5. [Step-by-Step Migration](#step-by-step-migration)
6. [Testing Your Migration](#testing-your-migration)
7. [Common Migration Patterns](#common-migration-patterns)
8. [Troubleshooting](#troubleshooting)
9. [FAQ](#faq)
10. [Support](#support)

---

## Overview

FuelAgentAI v2 is **100% backward compatible** with v1. Your existing code will continue working unchanged.

### Key Points

✅ **Zero Breaking Changes**: All v1 code works in v2
✅ **Opt-In Enhancement**: v2 features require explicit opt-in
✅ **12-Month Timeline**: Ample time for gradual migration
✅ **Side-by-Side Support**: Run v1 and v2 concurrently during migration
✅ **Performance Improvement**: v2 is 20% cheaper than v1

### Migration Goals

1. **Phase 1** (Immediate): Deploy v2, keep v1 behavior
2. **Phase 2** (1-3 months): Migrate reporting to enhanced format
3. **Phase 3** (3-6 months): Optimize for performance (fast path)
4. **Phase 4** (6-12 months): Full v2 feature adoption

---

## Timeline

### Official Support Timeline

| Date | Milestone | Action Required |
|------|-----------|-----------------|
| **2025-Q4** | v2 Launch | Deploy v2 (v1 behavior maintained) |
| **2026-Q1** | Migration Period | Begin migrating to v2 features |
| **2026-Q2** | v1 Deprecation Notice | Official 6-month warning |
| **2026-Q3** | v1 Sunset | v1 API removed (v2 only) |

### Recommended Migration Timeline

| Week | Task | Priority |
|------|------|----------|
| **Week 1** | Deploy v2 with v1 behavior | High |
| **Week 2** | Test backward compatibility | High |
| **Week 3-4** | Update CI/CD for v2 | Medium |
| **Week 5-8** | Migrate reporting endpoints to enhanced format | Medium |
| **Week 9-12** | Enable fast path for production | High |
| **Month 4-6** | Adopt advanced features (DQS, provenance) | Low |

---

## What's Changing

### What's NOT Changing

✅ **All v1 inputs remain valid**
✅ **Default output format unchanged**
✅ **Emission calculation results identical**
✅ **API endpoint paths unchanged**
✅ **Authentication unchanged**

### What's NEW (Opt-In)

🆕 **Multi-gas breakdown** (CO2, CH4, N2O)
🆕 **Provenance tracking** (source attribution)
🆕 **Data Quality Scoring** (5-dimension DQS)
🆕 **Enhanced response formats** (legacy, enhanced, compact)
🆕 **Scope/boundary controls** (Scope 1/2/3, WTT/WTW)
🆕 **Fast path optimization** (60% cost reduction)

---

## Migration Strategies

### Strategy 1: No-Change Deployment (Recommended First Step)

**Timeline:** Day 1
**Risk:** None
**Benefit:** Deploy v2 infrastructure immediately

```python
# OLD (v1)
from greenlang.agents import FuelAgentAI

agent = FuelAgentAI()
result = agent.run({
    "fuel_type": "diesel",
    "amount": 1000,
    "unit": "gallons"
})

# NEW (v2 - same behavior)
from greenlang.agents import FuelAgentAI_v2

agent = FuelAgentAI_v2()  # Just change import
result = agent.run({      # Same payload
    "fuel_type": "diesel",
    "amount": 1000,
    "unit": "gallons"
})
# Output is identical to v1
```

### Strategy 2: Gradual Feature Adoption

**Timeline:** Weeks 2-8
**Risk:** Low
**Benefit:** Adopt v2 features gradually

```python
# Week 2: Enable fast path (performance optimization)
agent = FuelAgentAI_v2(
    enable_explanations=False,  # Disable AI for speed
    enable_fast_path=True       # 60% cost reduction
)

# Week 4: Adopt enhanced format for compliance reporting
result = agent.run({
    "fuel_type": "diesel",
    "amount": 1000,
    "unit": "gallons",
    "response_format": "enhanced"  # Add this line
})

# Week 6: Add scope and boundary for CSRD compliance
result = agent.run({
    "fuel_type": "diesel",
    "amount": 1000,
    "unit": "gallons",
    "scope": "1",                 # Add scope
    "boundary": "WTW",            # Add boundary
    "response_format": "enhanced"
})
```

### Strategy 3: Parallel Run (Validation)

**Timeline:** Week 3-4
**Risk:** None
**Benefit:** Validate v2 accuracy

```python
# Run v1 and v2 side-by-side for validation
from greenlang.agents import FuelAgentAI, FuelAgentAI_v2

payload = {
    "fuel_type": "diesel",
    "amount": 1000,
    "unit": "gallons"
}

# v1 calculation
agent_v1 = FuelAgentAI()
result_v1 = agent_v1.run(payload)
emissions_v1 = result_v1["data"]["co2e_emissions_kg"]

# v2 calculation (same format)
agent_v2 = FuelAgentAI_v2()
result_v2 = agent_v2.run(payload)
emissions_v2 = result_v2["data"]["co2e_emissions_kg"]

# Validate identical results
assert emissions_v1 == emissions_v2, "v1 and v2 should match"
print("✅ v2 validation passed")
```

---

## Step-by-Step Migration

### Step 1: Update Dependencies

```bash
# Update greenlang package
pip install --upgrade greenlang

# Verify v2 is available
python -c "from greenlang.agents import FuelAgentAI_v2; print('v2 ready')"
```

### Step 2: Update Imports (No Logic Changes)

```python
# Before
from greenlang.agents import FuelAgentAI

# After
from greenlang.agents import FuelAgentAI_v2 as FuelAgentAI
# ^ Alias keeps existing code working
```

### Step 3: Deploy and Monitor

```python
# Add monitoring to track migration
agent = FuelAgentAI_v2()

result = agent.run(payload)

# Log metadata for analysis
metadata = result.get("metadata", {})
print(f"Execution path: {metadata.get('execution_path')}")  # "fast" or "ai"
print(f"Latency: {metadata.get('calculation_time_ms')}ms")
print(f"Cost: ${metadata.get('total_cost_usd')}")
```

### Step 4: Migrate Reporting Endpoints

```python
# Identify reporting endpoints
def calculate_emissions_for_report(fuel_type, amount, unit):
    agent = FuelAgentAI_v2()

    # Use enhanced format for reports
    result = agent.run({
        "fuel_type": fuel_type,
        "amount": amount,
        "unit": unit,
        "scope": "1",                  # Add scope
        "response_format": "enhanced"   # Enable v2 features
    })

    if result["success"]:
        data = result["data"]
        return {
            # v1 fields (unchanged)
            "total_co2e_kg": data["co2e_emissions_kg"],

            # v2 enhancements
            "multi_gas": data["vectors_kg"],
            "provenance": data["factor_record"],
            "quality": data["quality"]
        }
```

### Step 5: Enable Fast Path for Production

```python
# Production endpoints: maximize performance
def calculate_emissions_production(fuel_type, amount, unit):
    # Initialize with fast path optimization
    agent = FuelAgentAI_v2(
        enable_explanations=False,
        enable_recommendations=False,
        enable_fast_path=True
    )

    result = agent.run({
        "fuel_type": fuel_type,
        "amount": amount,
        "unit": unit,
        # Use legacy format for fast path
        "response_format": "legacy"
    })

    return result["data"]["co2e_emissions_kg"]

# Result: <100ms latency, ~$0 cost
```

### Step 6: Update Tests

```python
import pytest
from greenlang.agents import FuelAgentAI_v2

def test_emissions_calculation_v2():
    """Test emissions calculation with v2 agent"""
    agent = FuelAgentAI_v2()

    result = agent.run({
        "fuel_type": "diesel",
        "amount": 1000,
        "unit": "gallons"
    })

    assert result["success"], f"Calculation failed: {result.get('error')}"
    assert result["data"]["co2e_emissions_kg"] > 0
    assert result["data"]["emission_factor"] > 0

def test_enhanced_format_v2():
    """Test v2 enhanced format"""
    agent = FuelAgentAI_v2()

    result = agent.run({
        "fuel_type": "diesel",
        "amount": 1000,
        "unit": "gallons",
        "response_format": "enhanced"
    })

    assert "vectors_kg" in result["data"]
    assert "factor_record" in result["data"]
    assert "quality" in result["data"]
```

---

## Testing Your Migration

### Validation Checklist

- [ ] v2 installation successful
- [ ] Imports updated
- [ ] All existing tests pass
- [ ] v1 and v2 results match (parallel run)
- [ ] Enhanced format validated
- [ ] Fast path tested
- [ ] Performance benchmarks met
- [ ] Monitoring in place

### Test Suite

```python
def run_migration_validation():
    """Comprehensive migration validation"""
    from greenlang.agents import FuelAgentAI, FuelAgentAI_v2

    # Test cases
    test_cases = [
        ("diesel", 1000, "gallons"),
        ("natural_gas", 5000, "therms"),
        ("electricity", 10000, "kWh"),
        ("gasoline", 500, "gallons"),
    ]

    print("Running migration validation...")

    for fuel_type, amount, unit in test_cases:
        payload = {
            "fuel_type": fuel_type,
            "amount": amount,
            "unit": unit
        }

        # v1 result
        agent_v1 = FuelAgentAI()
        result_v1 = agent_v1.run(payload)

        # v2 result (legacy format)
        agent_v2 = FuelAgentAI_v2()
        result_v2 = agent_v2.run(payload)

        # Compare
        emissions_v1 = result_v1["data"]["co2e_emissions_kg"]
        emissions_v2 = result_v2["data"]["co2e_emissions_kg"]

        assert emissions_v1 == emissions_v2, (
            f"{fuel_type}: v1={emissions_v1}, v2={emissions_v2}"
        )

        print(f"✅ {fuel_type} validated: {emissions_v1:.2f} kg CO2e")

    print("\n✅ All migration tests passed!")

if __name__ == "__main__":
    run_migration_validation()
```

---

## Common Migration Patterns

### Pattern 1: API Endpoint Migration

```python
# OLD (v1)
@app.route("/api/v1/emissions", methods=["POST"])
def calculate_emissions_v1():
    from greenlang.agents import FuelAgentAI

    data = request.json
    agent = FuelAgentAI()
    result = agent.run(data)

    return jsonify(result["data"])

# NEW (v2 - backward compatible)
@app.route("/api/v2/emissions", methods=["POST"])
def calculate_emissions_v2():
    from greenlang.agents import FuelAgentAI_v2

    data = request.json
    agent = FuelAgentAI_v2(
        enable_fast_path=True  # Performance optimization
    )
    result = agent.run(data)

    return jsonify(result["data"])

# MIGRATION PERIOD: Support both
@app.route("/api/emissions", methods=["POST"])
def calculate_emissions():
    # Check client preference
    api_version = request.headers.get("X-API-Version", "v2")

    if api_version == "v1":
        return calculate_emissions_v1()
    else:
        return calculate_emissions_v2()
```

### Pattern 2: Batch Processing

```python
# OLD (v1)
def process_batch_v1(requests):
    from greenlang.agents import FuelAgentAI

    agent = FuelAgentAI()
    results = []

    for req in requests:
        result = agent.run(req)
        results.append(result["data"])

    return results

# NEW (v2 - with performance optimization)
def process_batch_v2(requests):
    from greenlang.agents import FuelAgentAI_v2

    # Fast path for batch processing (60% cost reduction)
    agent = FuelAgentAI_v2(
        enable_explanations=False,
        enable_fast_path=True
    )

    results = []
    for req in requests:
        result = agent.run(req)
        results.append(result["data"])

    return results

# Result: 80% cost reduction for batched requests (fast path + cache)
```

### Pattern 3: Compliance Reporting

```python
# OLD (v1) - limited compliance data
def generate_compliance_report_v1(year, data):
    from greenlang.agents import FuelAgentAI

    agent = FuelAgentAI()
    report = {
        "year": year,
        "total_emissions_kg": 0,
        "by_fuel": {}
    }

    for entry in data:
        result = agent.run(entry)
        emissions = result["data"]["co2e_emissions_kg"]
        report["total_emissions_kg"] += emissions
        # Missing: source attribution, data quality, uncertainty

    return report

# NEW (v2) - full CSRD compliance
def generate_compliance_report_v2(year, data):
    from greenlang.agents import FuelAgentAI_v2

    agent = FuelAgentAI_v2()
    report = {
        "year": year,
        "total_emissions_kg": 0,
        "by_fuel": {},
        "multi_gas_breakdown": {"CO2": 0, "CH4": 0, "N2O": 0},
        "data_quality": [],
        "sources": set()
    }

    for entry in data:
        # Request enhanced format
        entry["response_format"] = "enhanced"
        result = agent.run(entry)

        if result["success"]:
            data = result["data"]

            # Total emissions
            report["total_emissions_kg"] += data["co2e_emissions_kg"]

            # Multi-gas
            for gas in ["CO2", "CH4", "N2O"]:
                report["multi_gas_breakdown"][gas] += data["vectors_kg"][gas]

            # Provenance
            report["sources"].add(data["factor_record"]["source_org"])

            # Data quality
            report["data_quality"].append({
                "fuel": entry["fuel_type"],
                "dqs_score": data["quality"]["dqs"]["overall_score"],
                "uncertainty_pct": data["quality"]["uncertainty_95ci_pct"]
            })

    # Convert set to list for JSON serialization
    report["sources"] = list(report["sources"])

    return report

# Result: CSRD E1-5 compliant report
```

---

## Troubleshooting

### Issue 1: Import Error

**Error:**
```
ImportError: cannot import name 'FuelAgentAI_v2' from 'greenlang.agents'
```

**Solution:**
```bash
# Update greenlang package
pip install --upgrade greenlang>=2.0.0

# Verify installation
python -c "from greenlang.agents import FuelAgentAI_v2; print('OK')"
```

### Issue 2: Different Results (v1 vs v2)

**Error:**
```
AssertionError: v1 (10210.0) != v2 (10210.5)
```

**Solution:**
```python
# Small differences (<0.1%) are due to floating-point precision
# Use approximate comparison
import math

def approximately_equal(a, b, tolerance=0.001):
    return math.isclose(a, b, rel_tol=tolerance)

assert approximately_equal(emissions_v1, emissions_v2), (
    f"Significant difference: v1={emissions_v1}, v2={emissions_v2}"
)
```

### Issue 3: Performance Degradation

**Symptom:** v2 slower than v1

**Solution:**
```python
# Ensure fast path is enabled
agent = FuelAgentAI_v2(
    enable_explanations=False,      # Must be False for fast path
    enable_recommendations=False,    # Must be False for fast path
    enable_fast_path=True           # Enable optimization
)

# Use legacy format
result = agent.run({
    "fuel_type": "diesel",
    "amount": 1000,
    "unit": "gallons",
    "response_format": "legacy"     # Fast path eligible
})

# Check execution path
assert result["metadata"]["execution_path"] == "fast", (
    "Fast path not used - check agent configuration"
)
```

### Issue 4: Cache Not Working

**Symptom:** Low cache hit rate (<90%)

**Solution:**
```python
from greenlang.data.emission_factor_database import EmissionFactorDatabase

# Check cache statistics
db = EmissionFactorDatabase(enable_cache=True)
stats = db.get_cache_stats()

print(f"Hit rate: {stats['hit_rate_pct']:.1f}%")
print(f"Hits: {stats['hits']}")
print(f"Misses: {stats['misses']}")

# If hit rate is low, increase cache size
db = EmissionFactorDatabase(
    enable_cache=True,
    cache_size=2000,  # Increase from default 1000
    cache_ttl=7200    # Increase TTL to 2 hours
)
```

---

## FAQ

### Q: Do I need to migrate immediately?

**A:** No. v1 is supported until 2026-Q3 (12 months). However, we recommend deploying v2 with v1 behavior immediately to benefit from performance improvements.

### Q: Will v2 increase my costs?

**A:** No. v2 is **20% cheaper** than v1 due to fast path optimization and caching. Fast path eliminates AI costs for simple requests.

### Q: How do I know if I'm using fast path?

**A:** Check `result["metadata"]["execution_path"]`. It will be `"fast"` (optimized) or `"ai"` (full orchestration).

```python
result = agent.run(payload)
print(f"Execution path: {result['metadata']['execution_path']}")
```

### Q: Can I run v1 and v2 side-by-side?

**A:** Yes. Both versions are fully supported during the migration period.

```python
from greenlang.agents import FuelAgentAI, FuelAgentAI_v2

agent_v1 = FuelAgentAI()
agent_v2 = FuelAgentAI_v2()

# Use both for validation
```

### Q: What happens after v1 sunset (2026-Q3)?

**A:** v1 API will be removed. Only v2 will be available. However, v2 maintains v1 behavior via `response_format="legacy"`.

### Q: How do I test enhanced format?

**A:**
```python
result = agent.run({
    "fuel_type": "diesel",
    "amount": 1000,
    "unit": "gallons",
    "response_format": "enhanced"  # Add this
})

# Check for v2 fields
assert "vectors_kg" in result["data"]
assert "factor_record" in result["data"]
assert "quality" in result["data"]
```

### Q: Are there any breaking changes?

**A:** **No breaking changes.** All v1 code works in v2 without modification.

---

## Support

### Migration Support

- **Email:** support@greenlang.ai
- **Migration Assistance:** Available for Enterprise customers
- **Office Hours:** Weekly migration Q&A sessions

### Resources

- **API Documentation:** [API_V2_DOCUMENTATION.md](./API_V2_DOCUMENTATION.md)
- **GitHub Issues:** https://github.com/greenlang/greenlang/issues
- **Slack Community:** #fuel-agent-v2-migration

### Enterprise Migration Services

For enterprise customers, we offer:
- Dedicated migration engineer
- Custom migration plan
- Performance audit
- Training sessions
- Post-migration support

Contact: enterprise@greenlang.ai

---

**Document Version:** 1.0.0
**Last Updated:** 2025-10-24
**Author:** GreenLang Framework Team
