# GL-011 FUELCRAFT: Validation Error Examples

This document shows actual error messages you'll see when trying to create non-compliant configurations.

## COMPLIANCE VIOLATION Errors

### Error 1: Non-Deterministic Temperature

```python
>>> from config import FuelManagementConfig
>>> config = FuelManagementConfig(temperature=0.7)

ValidationError: 1 validation error for FuelManagementConfig
temperature
  COMPLIANCE VIOLATION: temperature must be 0.0 for deterministic fuel
  optimization calculations. Got: 0.7. This ensures reproducible results
  for regulatory compliance. [type=value_error]
```

**Fix:**
```python
config = FuelManagementConfig(temperature=0.0)  # ✓ Compliant
```

---

### Error 2: Wrong Random Seed

```python
>>> config = FuelManagementConfig(seed=123)

ValidationError: 1 validation error for FuelManagementConfig
seed
  COMPLIANCE VIOLATION: seed must be 42 for deterministic calculations.
  Got: 123. This ensures bit-perfect reproducibility across runs.
  [type=value_error]
```

**Fix:**
```python
config = FuelManagementConfig(seed=42)  # ✓ Compliant
```

---

### Error 3: Deterministic Mode Disabled

```python
>>> config = FuelManagementConfig(deterministic_mode=False)

ValidationError: 1 validation error for FuelManagementConfig
deterministic_mode
  COMPLIANCE VIOLATION: deterministic_mode must be True for regulatory
  compliance. All fuel cost and emissions calculations must be reproducible
  for audit trails. [type=value_error]
```

**Fix:**
```python
config = FuelManagementConfig(deterministic_mode=True)  # ✓ Compliant
```

---

### Error 4: Invalid Fuel Types

```python
>>> config = FuelManagementConfig(
...     supported_fuels=['natural_gas', 'coal', 'unicorn_tears', 'magic_beans']
... )

ValidationError: 1 validation error for FuelManagementConfig
supported_fuels
  COMPLIANCE VIOLATION: Invalid fuel types: {'magic_beans', 'unicorn_tears'}.
  Only ISO 6976 / ASTM D4809 compliant fuels are allowed: {'coal', 'propane',
  'hydrogen', 'fuel_oil', 'syngas', 'biomass', 'natural_gas', 'diesel'}
  [type=value_error]
```

**Fix:**
```python
config = FuelManagementConfig(
    supported_fuels=['natural_gas', 'coal', 'biomass', 'hydrogen']  # ✓ Compliant
)
```

---

### Error 5: Insufficient Decimal Precision

```python
>>> config = FuelManagementConfig(decimal_precision=6)

ValidationError: 1 validation error for FuelManagementConfig
decimal_precision
  COMPLIANCE VIOLATION: decimal_precision must be >= 10 for financial
  calculations. Got: 6. Required for accurate cost optimization to
  0.0000000001 precision. [type=value_error]
```

**Fix:**
```python
config = FuelManagementConfig(decimal_precision=10)  # ✓ Compliant
```

---

### Error 6: Provenance Tracking Disabled

```python
>>> config = FuelManagementConfig(enable_provenance=False)

ValidationError: 1 validation error for FuelManagementConfig
enable_provenance
  COMPLIANCE VIOLATION: enable_provenance must be True. SHA-256 audit
  trails are required for all optimization decisions. [type=value_error]
```

**Fix:**
```python
config = FuelManagementConfig(enable_provenance=True)  # ✓ Compliant
```

---

### Error 7: Missing Required Alert Thresholds

```python
>>> config = FuelManagementConfig(
...     alert_thresholds={
...         'fuel_shortage': 0.15,
...         'cost_overrun': 0.10
...         # Missing: emissions_violation, integration_failure
...     }
... )

ValidationError: 1 validation error for FuelManagementConfig
alert_thresholds
  COMPLIANCE VIOLATION: Missing required alert thresholds:
  {'integration_failure', 'emissions_violation'}. These alerts are
  mandatory for safe fuel management operations. [type=value_error]
```

**Fix:**
```python
config = FuelManagementConfig(
    alert_thresholds={
        'fuel_shortage': 0.15,
        'cost_overrun': 0.10,
        'emissions_violation': 0.05,      # ✓ Added
        'integration_failure': 0.0        # ✓ Added
    }
)
```

---

### Error 8: Invalid Alert Threshold Range

```python
>>> config = FuelManagementConfig(
...     alert_thresholds={
...         'fuel_shortage': 1.5,  # Invalid: > 1.0
...         'cost_overrun': 0.10,
...         'emissions_violation': 0.05,
...         'integration_failure': 0.0
...     }
... )

ValidationError: 1 validation error for FuelManagementConfig
alert_thresholds
  COMPLIANCE VIOLATION: Alert threshold 'fuel_shortage' must be between
  0 and 1. Got: 1.5 [type=value_error]
```

**Fix:**
```python
config = FuelManagementConfig(
    alert_thresholds={
        'fuel_shortage': 0.15,  # ✓ Valid (0.0 - 1.0)
        'cost_overrun': 0.10,
        'emissions_violation': 0.05,
        'integration_failure': 0.0
    }
)
```

---

## SECURITY VIOLATION Errors

### Error 9: Zero Secrets Policy Disabled

```python
>>> config = FuelManagementConfig(zero_secrets=False)

ValidationError: 1 validation error for FuelManagementConfig
zero_secrets
  SECURITY VIOLATION: zero_secrets must be True. API keys and credentials
  must never be in config.py. Use environment variables or secrets manager.
  [type=value_error]
```

**Fix:**
```python
config = FuelManagementConfig(zero_secrets=True)  # ✓ Compliant
# Store secrets in environment variables:
# export FUEL_API_KEY="sk-..."
# export ERP_PASSWORD="..."
```

---

### Error 10: TLS Disabled

```python
>>> config = FuelManagementConfig(tls_enabled=False)

ValidationError: 1 validation error for FuelManagementConfig
tls_enabled
  SECURITY VIOLATION: tls_enabled must be True for production deployments.
  All API connections must use TLS 1.3 for data protection. [type=value_error]
```

**Fix:**
```python
config = FuelManagementConfig(tls_enabled=True)  # ✓ Compliant
```

---

## PRODUCTION ENVIRONMENT Errors

### Error 11: Debug Mode in Production

```python
>>> config = FuelManagementConfig(
...     environment='production',
...     debug_mode=True
... )

ValidationError: 1 validation error for FuelManagementConfig
  SECURITY VIOLATION: debug_mode must be False in production environment.
  Debug mode exposes sensitive operational data. [type=value_error]
```

**Fix:**
```python
config = FuelManagementConfig(
    environment='production',
    debug_mode=False  # ✓ Compliant
)
```

---

### Error 12: Missing TLS in Production

```python
>>> config = FuelManagementConfig(
...     environment='production',
...     tls_enabled=False
... )

ValidationError: 1 validation error for FuelManagementConfig
tls_enabled
  SECURITY VIOLATION: tls_enabled must be True for production deployments.
  All API connections must use TLS 1.3 for data protection. [type=value_error]
```

**Fix:**
```python
config = FuelManagementConfig(
    environment='production',
    tls_enabled=True  # ✓ Compliant
)
```

---

### Error 13: Non-Deterministic in Production

```python
>>> config = FuelManagementConfig(
...     environment='production',
...     deterministic_mode=False
... )

ValidationError: 1 validation error for FuelManagementConfig
deterministic_mode
  COMPLIANCE VIOLATION: deterministic_mode must be True for regulatory
  compliance. All fuel cost and emissions calculations must be reproducible
  for audit trails. [type=value_error]
```

**Fix:**
```python
config = FuelManagementConfig(
    environment='production',
    deterministic_mode=True  # ✓ Compliant
)
```

---

### Error 14: Missing Provenance in Production

```python
>>> config = FuelManagementConfig(
...     environment='production',
...     enable_provenance=False
... )

ValidationError: 1 validation error for FuelManagementConfig
enable_provenance
  COMPLIANCE VIOLATION: enable_provenance must be True. SHA-256 audit
  trails are required for all optimization decisions. [type=value_error]
```

**Fix:**
```python
config = FuelManagementConfig(
    environment='production',
    enable_provenance=True  # ✓ Compliant
)
```

---

### Error 15: Missing Audit Logging in Production

```python
>>> config = FuelManagementConfig(
...     environment='production',
...     enable_audit_logging=False
... )

ValidationError: 1 validation error for FuelManagementConfig
  COMPLIANCE VIOLATION: Audit logging required in production.
  Set enable_audit_logging=True for compliance. [type=value_error]
```

**Fix:**
```python
config = FuelManagementConfig(
    environment='production',
    enable_audit_logging=True  # ✓ Compliant
)
```

---

## PERFORMANCE VIOLATION Errors

### Error 16: Excessive Calculation Timeout

```python
>>> config = FuelManagementConfig(calculation_timeout_seconds=120)

ValidationError: 1 validation error for FuelManagementConfig
  PERFORMANCE VIOLATION: calculation_timeout_seconds should not exceed
  60 seconds. Got: 120. Long timeouts indicate inefficient calculations.
  [type=value_error]
```

**Fix:**
```python
config = FuelManagementConfig(
    calculation_timeout_seconds=30  # ✓ Compliant (<= 60)
)
```

---

## RUNTIME ASSERTION Errors

### Error 17: Compliance Assertion Failure

```python
>>> config = FuelManagementConfig(
...     temperature=0.0,
...     seed=42,
...     deterministic_mode=True,
...     zero_secrets=True,
...     enable_provenance=True,
...     tls_enabled=True,
...     alert_thresholds={
...         'fuel_shortage': 0.15,
...         'cost_overrun': 0.10
...         # Missing required alerts!
...     }
... )
>>> config.assert_compliance_ready()

AssertionError: Missing required alerts: {'integration_failure', 'emissions_violation'}
```

**Fix:**
```python
config = FuelManagementConfig(
    temperature=0.0,
    seed=42,
    deterministic_mode=True,
    zero_secrets=True,
    enable_provenance=True,
    tls_enabled=True,
    alert_thresholds={
        'fuel_shortage': 0.15,
        'cost_overrun': 0.10,
        'emissions_violation': 0.05,      # ✓ Added
        'integration_failure': 0.0        # ✓ Added
    }
)
config.assert_compliance_ready()  # ✓ PASS
```

---

## MULTIPLE VIOLATIONS

### Error 18: Multiple Compliance Violations

```python
>>> config = FuelManagementConfig(
...     temperature=0.7,
...     seed=999,
...     deterministic_mode=False
... )

ValidationError: 3 validation errors for FuelManagementConfig
temperature
  COMPLIANCE VIOLATION: temperature must be 0.0 for deterministic fuel
  optimization calculations. Got: 0.7. [type=value_error]
seed
  COMPLIANCE VIOLATION: seed must be 42 for deterministic calculations.
  Got: 999. [type=value_error]
deterministic_mode
  COMPLIANCE VIOLATION: deterministic_mode must be True for regulatory
  compliance. [type=value_error]
```

**Fix:**
```python
config = FuelManagementConfig(
    temperature=0.0,              # ✓ Fixed
    seed=42,                      # ✓ Fixed
    deterministic_mode=True       # ✓ Fixed
)
```

---

## VALID CONFIGURATIONS (No Errors)

### Example 1: Development Configuration

```python
>>> config = FuelManagementConfig(
...     environment="development",
...     temperature=0.0,
...     seed=42,
...     deterministic_mode=True,
...     zero_secrets=True,
...     tls_enabled=True,
...     enable_provenance=True,
...     decimal_precision=10,
...     alert_thresholds={
...         'fuel_shortage': 0.15,
...         'cost_overrun': 0.10,
...         'emissions_violation': 0.05,
...         'integration_failure': 0.0
...     }
... )

✓ Configuration created successfully!
```

---

### Example 2: Production Configuration

```python
>>> config = FuelManagementConfig(
...     environment="production",
...     temperature=0.0,
...     seed=42,
...     deterministic_mode=True,
...     zero_secrets=True,
...     tls_enabled=True,
...     enable_provenance=True,
...     enable_audit_logging=True,
...     debug_mode=False,
...     decimal_precision=10,
...     calculation_timeout_seconds=30,
...     alert_thresholds={
...         'fuel_shortage': 0.15,
...         'cost_overrun': 0.10,
...         'emissions_violation': 0.05,
...         'integration_failure': 0.0
...     }
... )
>>> config.assert_compliance_ready()
>>> config.assert_security_ready()
>>> config.assert_determinism_ready()

✓ All validations passed!
✓ Configuration is production-ready!
```

---

## Error Message Components

Each validation error contains:

1. **Error Type:** `COMPLIANCE VIOLATION` or `SECURITY VIOLATION`
2. **Field Name:** The configuration field that failed validation
3. **Problem Description:** What's wrong with the value
4. **Actual Value:** The value that was provided (if applicable)
5. **Required Value/Range:** What's required for compliance
6. **Justification:** Why this requirement exists

**Example Breakdown:**

```
COMPLIANCE VIOLATION: temperature must be 0.0 for deterministic fuel
                      ^^^^^^^^^^^^^^^
                      Error Type

                      temperature must be 0.0 for deterministic fuel
                      ^^^^^^^^^^^
                      Field Name

                      optimization calculations. Got: 0.7.
                                                 ^^^^^^^^
                                                 Actual Value

                      This ensures reproducible results for regulatory compliance.
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                      Justification
```

---

## Quick Troubleshooting Guide

| Error Contains | Problem | Fix |
|----------------|---------|-----|
| "temperature must be 0.0" | Non-deterministic LLM | Set `temperature=0.0` |
| "seed must be 42" | Non-reproducible | Set `seed=42` |
| "deterministic_mode must be True" | Compliance violation | Set `deterministic_mode=True` |
| "zero_secrets must be True" | Security risk | Set `zero_secrets=True` |
| "tls_enabled must be True" | Unencrypted | Set `tls_enabled=True` |
| "Invalid fuel types" | Non-ISO/ASTM fuels | Use only approved fuels |
| "decimal_precision must be >= 10" | Insufficient precision | Set `decimal_precision>=10` |
| "enable_provenance must be True" | Missing audit trail | Set `enable_provenance=True` |
| "Missing required alert thresholds" | Safety risk | Add all 4 alert thresholds |
| "debug_mode must be False in production" | Production risk | Set `debug_mode=False` |

---

**Note:** All error messages are designed to be:
- **Actionable:** Clear what needs to change
- **Informative:** Explains why it's required
- **Specific:** Shows actual vs. expected values
- **Compliant:** References regulatory requirements
