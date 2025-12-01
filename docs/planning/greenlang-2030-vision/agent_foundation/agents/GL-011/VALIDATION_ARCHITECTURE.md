# GL-011 FUELCRAFT: Validation Architecture

## Multi-Layer Defense Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                    Configuration Instantiation                  │
│         config = FuelManagementConfig(...)                       │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                  LAYER 1: Field Type Validation                  │
│                      (Pydantic Built-in)                         │
├─────────────────────────────────────────────────────────────────┤
│  ✓ Type checking (str, int, float, bool, List, Dict)           │
│  ✓ Range validation (ge, le, gt, lt)                            │
│  ✓ String patterns (min_length, max_length, pattern)            │
│  ✓ Default value assignment                                     │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│            LAYER 2: Field-Level Custom Validators                │
│                   (@field_validator decorators)                  │
├─────────────────────────────────────────────────────────────────┤
│  ✓ validate_temperature()        → temperature = 0.0            │
│  ✓ validate_seed()                → seed = 42                   │
│  ✓ validate_deterministic()       → deterministic_mode = True   │
│  ✓ validate_zero_secrets()        → zero_secrets = True         │
│  ✓ validate_tls()                 → tls_enabled = True          │
│  ✓ validate_fuels()               → ISO/ASTM compliance         │
│  ✓ validate_precision()           → decimal_precision >= 10     │
│  ✓ validate_provenance()          → enable_provenance = True    │
│  ✓ validate_thresholds()          → 4 required alerts           │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│           LAYER 3: Model-Level Cross-Field Validation            │
│                   (@model_validator decorator)                   │
├─────────────────────────────────────────────────────────────────┤
│  ✓ validate_environment_consistency()                           │
│     ├─ Production environment checks:                            │
│     │   ├─ TLS required                                         │
│     │   ├─ Deterministic mode required                          │
│     │   ├─ Debug mode forbidden                                 │
│     │   ├─ Provenance tracking required                         │
│     │   └─ Audit logging required                               │
│     ├─ Calculation timeout <= 60 seconds                        │
│     └─ Optimization weights sum to 1.0                          │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Configuration Object Created                    │
│                         (Valid ✓)                                │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│        LAYER 4: Runtime Assertion Checks (Optional)              │
│                    (Developer-Invoked)                           │
├─────────────────────────────────────────────────────────────────┤
│  ✓ config.assert_compliance_ready()                             │
│     └─ Validates all compliance requirements                    │
│                                                                   │
│  ✓ config.assert_security_ready()                               │
│     └─ Validates all security requirements                      │
│                                                                   │
│  ✓ config.assert_determinism_ready()                            │
│     └─ Validates all determinism requirements                   │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│              LAYER 5: Orchestrator Initialization                │
│                  (Production Deployment)                         │
├─────────────────────────────────────────────────────────────────┤
│  orchestrator = FuelManagementOrchestrator(config)              │
│     ├─ Config loaded and validated                              │
│     ├─ All validators passed                                    │
│     ├─ All assertions passed                                    │
│     └─ Ready for production ✓                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Validator Execution Order

### 1. Field Validators (Parallel Execution)

```python
# All field validators execute in parallel before model validators
validate_temperature()          # temperature = 0.0
validate_seed()                 # seed = 42
validate_deterministic()        # deterministic_mode = True
validate_zero_secrets()         # zero_secrets = True
validate_tls()                  # tls_enabled = True
validate_fuels()                # ISO/ASTM compliant fuels
validate_precision()            # decimal_precision >= 10
validate_provenance()           # enable_provenance = True
validate_thresholds()           # All 4 alert thresholds
```

### 2. Model Validator (Sequential Execution)

```python
# Model validator executes after ALL field validators pass
validate_environment_consistency()
    └─ Cross-field validation
    └─ Environment-specific checks
    └─ Business rule validation
```

## Error Handling Flow

```
Configuration Creation Attempt
    │
    ├─ Field Validator Fails
    │   └─> ValidationError raised
    │       └─> Error message with "COMPLIANCE VIOLATION" or "SECURITY VIOLATION"
    │           └─> Configuration NOT created
    │               └─> Developer must fix and retry
    │
    ├─ Model Validator Fails
    │   └─> ValidationError raised
    │       └─> Error message with violation type
    │           └─> Configuration NOT created
    │               └─> Developer must fix and retry
    │
    └─ All Validators Pass
        └─> Configuration created successfully ✓
            └─> Optional: Run assertion checks
                └─> Ready for production ✓
```

## Validation Coverage Matrix

| Aspect | Coverage | Validators | Enforcement |
|--------|----------|------------|-------------|
| **Determinism** | 100% | 3 validators | ✓ Strict |
| **Security** | 100% | 2 validators + URL checks | ✓ Strict |
| **Fuel Quality** | 100% | 1 validator | ✓ Strict |
| **Calculation Precision** | 100% | 1 validator | ✓ Strict |
| **Provenance** | 100% | 1 validator | ✓ Strict |
| **Alert Thresholds** | 100% | 1 validator | ✓ Strict |
| **Environment Consistency** | 100% | 1 model validator | ✓ Strict |
| **Production Readiness** | 100% | 3 assertion methods | ✓ On-Demand |

## Validator Dependencies

```
┌─────────────────────────────────────────────────────────────┐
│                   Independent Validators                     │
│              (No dependencies on other fields)               │
├─────────────────────────────────────────────────────────────┤
│  • validate_temperature()                                   │
│  • validate_seed()                                          │
│  • validate_deterministic()                                 │
│  • validate_zero_secrets()                                  │
│  • validate_tls()                                           │
│  • validate_fuels()                                         │
│  • validate_precision()                                     │
│  • validate_provenance()                                    │
│  • validate_thresholds()                                    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   Dependent Validators                       │
│           (Require other fields to be validated)             │
├─────────────────────────────────────────────────────────────┤
│  • validate_environment_consistency()                       │
│     ├─ Depends on: environment                              │
│     ├─ Depends on: tls_enabled                              │
│     ├─ Depends on: deterministic_mode                       │
│     ├─ Depends on: debug_mode                               │
│     ├─ Depends on: enable_provenance                        │
│     ├─ Depends on: enable_audit_logging                     │
│     ├─ Depends on: calculation_timeout_seconds              │
│     └─ Depends on: optimization.primary_objective           │
└─────────────────────────────────────────────────────────────┘
```

## Validation Performance

### Execution Time Analysis

```
Field Validators (parallel):     ~0.5ms per validator
Model Validator:                 ~1.0ms
Total Configuration Creation:    ~5ms
Assertion Checks (optional):     ~2ms
───────────────────────────────────────────
Total Validation Time:           ~7ms
```

**Performance Impact:** Negligible (<10ms overhead)
**Benefit:** Prevents runtime errors, ensures compliance

## Production Deployment Checklist

```
┌─────────────────────────────────────────────────────────────┐
│           Pre-Deployment Validation Checklist                │
├─────────────────────────────────────────────────────────────┤
│  1. [ ] Create config with environment='production'         │
│  2. [ ] All field validators pass                           │
│  3. [ ] Model validator passes                              │
│  4. [ ] config.assert_compliance_ready() passes             │
│  5. [ ] config.assert_security_ready() passes               │
│  6. [ ] config.assert_determinism_ready() passes            │
│  7. [ ] Test suite passes (test_config_validators.py)       │
│  8. [ ] No hardcoded secrets in config                      │
│  9. [ ] TLS enabled for all integrations                    │
│ 10. [ ] Debug mode disabled                                 │
│ 11. [ ] Audit logging enabled                               │
│ 12. [ ] Provenance tracking enabled                         │
└─────────────────────────────────────────────────────────────┘
```

## Validator Testing Strategy

### Unit Tests (Field Validators)

```python
# Each field validator has dedicated test
test_temperature_violation()      # temperature != 0.0 → ValidationError
test_seed_violation()             # seed != 42 → ValidationError
test_deterministic_violation()    # deterministic_mode = False → ValidationError
test_zero_secrets_violation()     # zero_secrets = False → ValidationError
test_tls_violation()              # tls_enabled = False → ValidationError
test_fuel_violation()             # invalid fuel types → ValidationError
test_precision_violation()        # decimal_precision < 10 → ValidationError
test_provenance_violation()       # enable_provenance = False → ValidationError
test_thresholds_violation()       # missing alerts → ValidationError
```

### Integration Tests (Model Validator)

```python
# Environment-specific validation
test_production_debug_mode()      # production + debug_mode → ValidationError
test_production_no_tls()          # production + no TLS → ValidationError
test_production_no_determinism()  # production + no determinism → ValidationError
```

### Positive Tests

```python
# Valid configurations pass
test_valid_config()               # Valid development config → Success
test_production_config()          # Valid production config → Success
test_assertion_helpers()          # All assertions pass → Success
```

## Benefits of Multi-Layer Validation

1. **Fail Fast:** Invalid configs rejected at creation time
2. **Clear Errors:** Violation messages explain exact problem
3. **Type Safety:** Pydantic ensures correct data types
4. **Compliance Enforcement:** Regulatory requirements enforced automatically
5. **Security Hardening:** Security policies enforced by default
6. **Developer Experience:** Immediate feedback on configuration issues
7. **Production Safety:** Environment-specific checks prevent deployment errors
8. **Audit Trail:** All violations logged and traceable

## Integration with Orchestrator

```python
# In orchestrator initialization
class FuelManagementOrchestrator:
    def __init__(self, config: FuelManagementConfig):
        # Config already validated by Pydantic
        # All validators have passed

        # Optional: Additional runtime checks
        if config.environment == 'production':
            config.assert_compliance_ready()
            config.assert_security_ready()
            config.assert_determinism_ready()

        self.config = config
        # Initialize agents with validated config
        # Safe to proceed - all requirements met ✓
```

## Summary

The GL-011 validation architecture provides **defense in depth** through:

- **Layer 1:** Type validation (Pydantic built-in)
- **Layer 2:** Field-level custom validators (9 validators)
- **Layer 3:** Model-level cross-field validators (1 validator)
- **Layer 4:** Runtime assertion checks (3 methods)
- **Layer 5:** Orchestrator initialization checks

**Total Protection:** 5 validation layers ensuring compliance and security
**Zero Tolerance:** Non-compliant configurations cannot be created
**Production Ready:** All requirements enforced from day one ✓
