# Test Coverage Sprint Plan

## Current State
- **Current Coverage**: 9.43%
- **Target Coverage**: 40% (for v0.2.0 final)
- **Timeline**: 1-2 days

## Priority Areas for Testing

### 1. Core CLI Module (High Priority)
**Files**: `greenlang/cli.py`, `core/greenlang/cli.py`
- [ ] Test all CLI commands
- [ ] Test error handling
- [ ] Test help text generation
- [ ] Test version display

### 2. Pack Management (High Priority)
**Files**: `core/greenlang/pack_registry.py`, `core/greenlang/pack_loader.py`
- [ ] Test pack initialization
- [ ] Test pack validation
- [ ] Test pack listing
- [ ] Test pack installation

### 3. Executor Module (Medium Priority)
**Files**: `core/greenlang/executor.py`
- [ ] Test pipeline execution
- [ ] Test error handling
- [ ] Test output formatting
- [ ] Test async operations

### 4. Configuration (Medium Priority)
**Files**: `core/greenlang/config.py`
- [ ] Test config loading
- [ ] Test default values
- [ ] Test environment variables
- [ ] Test validation

### 5. Utils Module (Low Priority)
**Files**: `core/greenlang/utils.py`
- [ ] Test file operations
- [ ] Test path handling
- [ ] Test logging utilities

## Test Implementation Strategy

### Phase 1: CLI Tests (Day 1 Morning)
Create comprehensive CLI tests covering all commands:
```python
# tests/unit/test_cli.py
- test_version_command()
- test_help_command()
- test_doctor_command()
- test_pack_list_command()
- test_pack_validate_command()
- test_init_command()
```

### Phase 2: Pack Management Tests (Day 1 Afternoon)
```python
# tests/unit/test_pack_manager.py
- test_pack_discovery()
- test_pack_validation()
- test_pack_loading()
- test_pack_metadata()
```

### Phase 3: Integration Tests (Day 2 Morning)
```python
# tests/integration/test_end_to_end.py
- test_full_pack_lifecycle()
- test_pipeline_execution()
- test_error_recovery()
```

## Quick Wins
These areas will give maximum coverage boost:
1. CLI command tests (estimated +10% coverage)
2. Pack validation tests (estimated +8% coverage)
3. Config tests (estimated +5% coverage)
4. Utils tests (estimated +5% coverage)

## Next Steps
1. Start with CLI tests immediately
2. Run coverage after each module
3. Focus on happy path first, then edge cases
4. Document any issues found during testing

## Success Metrics
- [ ] Reach 40% test coverage
- [ ] All critical paths tested
- [ ] No failing tests
- [ ] Coverage report generated

## Commands to Run
```bash
# Run tests with coverage
pytest tests/ --cov=greenlang --cov=core --cov-report=html

# Check coverage
coverage report

# Generate HTML report
coverage html
```