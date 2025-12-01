# GL-011 FUELCRAFT: Compliance Validator Quick Reference

## COMPLIANCE VIOLATION Errors

### Determinism Violations

```python
# ERROR: temperature must be 0.0
config = FuelManagementConfig(temperature=0.7)
# ❌ COMPLIANCE VIOLATION: temperature must be 0.0 for deterministic fuel optimization calculations

# ERROR: seed must be 42
config = FuelManagementConfig(seed=123)
# ❌ COMPLIANCE VIOLATION: seed must be 42 for deterministic calculations

# ERROR: deterministic_mode must be True
config = FuelManagementConfig(deterministic_mode=False)
# ❌ COMPLIANCE VIOLATION: deterministic_mode must be True for regulatory compliance
```

### Security Violations

```python
# ERROR: zero_secrets must be True
config = FuelManagementConfig(zero_secrets=False)
# ❌ SECURITY VIOLATION: zero_secrets must be True. API keys must be in environment variables

# ERROR: tls_enabled must be True
config = FuelManagementConfig(tls_enabled=False)
# ❌ SECURITY VIOLATION: tls_enabled must be True. All API connections must use TLS 1.3
```

### Fuel Quality Violations

```python
# ERROR: Invalid fuel type
config = FuelManagementConfig(supported_fuels=['natural_gas', 'unicorn_tears'])
# ❌ COMPLIANCE VIOLATION: Invalid fuel types: {'unicorn_tears'}
# Only ISO 6976 / ASTM D4809 compliant fuels allowed
```

### Calculation Violations

```python
# ERROR: decimal_precision must be >= 10
config = FuelManagementConfig(decimal_precision=6)
# ❌ COMPLIANCE VIOLATION: decimal_precision must be >= 10 for financial calculations
```

### Provenance Violations

```python
# ERROR: enable_provenance must be True
config = FuelManagementConfig(enable_provenance=False)
# ❌ COMPLIANCE VIOLATION: SHA-256 audit trails required
```

### Alert Violations

```python
# ERROR: Missing required alert thresholds
config = FuelManagementConfig(
    alert_thresholds={'fuel_shortage': 0.15}  # Missing 3 required alerts
)
# ❌ COMPLIANCE VIOLATION: Missing required alert thresholds:
# {'emissions_violation', 'integration_failure', 'cost_overrun'}
```

### Production Environment Violations

```python
# ERROR: debug_mode in production
config = FuelManagementConfig(environment='production', debug_mode=True)
# ❌ SECURITY VIOLATION: debug_mode must be False in production

# ERROR: No TLS in production
config = FuelManagementConfig(environment='production', tls_enabled=False)
# ❌ SECURITY VIOLATION: TLS required in production

# ERROR: Non-deterministic in production
config = FuelManagementConfig(environment='production', deterministic_mode=False)
# ❌ COMPLIANCE VIOLATION: Deterministic mode required in production
```

## Valid Configurations

### Development Configuration

```python
config = FuelManagementConfig(
    environment="development",
    temperature=0.0,              # ✓ Locked to 0.0
    seed=42,                      # ✓ Locked to 42
    deterministic_mode=True,      # ✓ Required
    zero_secrets=True,            # ✓ Required
    tls_enabled=True,             # ✓ Required
    enable_provenance=True,       # ✓ Required
    decimal_precision=10,         # ✓ >= 10
    alert_thresholds={            # ✓ All 4 required
        'fuel_shortage': 0.15,
        'cost_overrun': 0.10,
        'emissions_violation': 0.05,
        'integration_failure': 0.0
    }
)
```

### Production Configuration

```python
config = FuelManagementConfig(
    environment="production",
    temperature=0.0,              # ✓ Locked to 0.0
    seed=42,                      # ✓ Locked to 42
    deterministic_mode=True,      # ✓ Required
    zero_secrets=True,            # ✓ Required
    tls_enabled=True,             # ✓ Required in production
    enable_provenance=True,       # ✓ Required in production
    enable_audit_logging=True,    # ✓ Required in production
    debug_mode=False,             # ✓ Must be False in production
    decimal_precision=10,         # ✓ >= 10
    calculation_timeout_seconds=30,  # ✓ <= 60
    alert_thresholds={            # ✓ All 4 required
        'fuel_shortage': 0.15,
        'cost_overrun': 0.10,
        'emissions_violation': 0.05,
        'integration_failure': 0.0
    }
)
```

## Runtime Assertions

```python
# Check compliance readiness
config.assert_compliance_ready()
# Validates: deterministic_mode, temperature, seed, zero_secrets,
#            enable_provenance, tls_enabled, alert_thresholds

# Check security readiness
config.assert_security_ready()
# Validates: zero_secrets, tls_enabled, no credentials in URLs

# Check determinism readiness
config.assert_determinism_ready()
# Validates: deterministic_mode, temperature=0.0, seed=42,
#            decimal_precision>=10, optimization timeout<=300s
```

## Allowed Fuel Types (ISO 6976 / ASTM D4809)

```python
supported_fuels = [
    'natural_gas',  # ✓ ISO 6976
    'coal',         # ✓ ASTM D4809
    'biomass',      # ✓ ISO 17225
    'fuel_oil',     # ✓ ASTM D4809
    'diesel',       # ✓ ASTM D975
    'hydrogen',     # ✓ ISO 14687
    'propane',      # ✓ ASTM D1835
    'syngas'        # ✓ ISO 6976
]
```

## Required Alert Thresholds

```python
alert_thresholds = {
    'fuel_shortage': 0.15,         # ✓ Required (0.0 - 1.0)
    'cost_overrun': 0.10,          # ✓ Required (0.0 - 1.0)
    'emissions_violation': 0.05,   # ✓ Required (0.0 - 1.0)
    'integration_failure': 0.0     # ✓ Required (0.0 - 1.0)
}
```

## Environment-Specific Requirements

| Requirement | Development | Staging | Production |
|-------------|-------------|---------|------------|
| TLS enabled | Optional | Required | **REQUIRED** |
| Deterministic mode | Optional | Required | **REQUIRED** |
| Debug mode | Allowed | Allowed | **FORBIDDEN** |
| Provenance tracking | Optional | Required | **REQUIRED** |
| Audit logging | Optional | Required | **REQUIRED** |
| Temperature = 0.0 | Required | Required | **REQUIRED** |
| Seed = 42 | Required | Required | **REQUIRED** |

## Validation Execution Flow

```
Configuration Instantiation
    ↓
Field Validators (parallel)
    ├─ validate_temperature()     → temperature = 0.0
    ├─ validate_seed()             → seed = 42
    ├─ validate_deterministic()    → deterministic_mode = True
    ├─ validate_zero_secrets()     → zero_secrets = True
    ├─ validate_tls()              → tls_enabled = True
    ├─ validate_fuels()            → ISO/ASTM compliant
    ├─ validate_precision()        → decimal_precision >= 10
    ├─ validate_provenance()       → enable_provenance = True
    └─ validate_thresholds()       → All 4 alerts present
    ↓
Model Validator (after all fields)
    └─ validate_environment_consistency()
        └─ Production checks (if env='production')
    ↓
Configuration Valid ✓
    ↓
Runtime Assertions (optional)
    ├─ assert_compliance_ready()
    ├─ assert_security_ready()
    └─ assert_determinism_ready()
    ↓
Ready for Production ✓
```

## Error Prevention Checklist

Before deploying GL-011 FuelManagementOrchestrator:

- [ ] `temperature = 0.0` (deterministic calculations)
- [ ] `seed = 42` (reproducible results)
- [ ] `deterministic_mode = True` (regulatory compliance)
- [ ] `zero_secrets = True` (no hardcoded credentials)
- [ ] `tls_enabled = True` (encrypted connections)
- [ ] `enable_provenance = True` (SHA-256 audit trails)
- [ ] `decimal_precision >= 10` (financial accuracy)
- [ ] All 4 alert thresholds configured
- [ ] Only ISO/ASTM compliant fuels
- [ ] Production: `debug_mode = False`
- [ ] Production: `enable_audit_logging = True`

## Quick Test

```bash
cd docs/planning/greenlang-2030-vision/agent_foundation/agents/GL-011
python test_config_validators.py
```

Expected output: **All 14 tests PASS** ✓
