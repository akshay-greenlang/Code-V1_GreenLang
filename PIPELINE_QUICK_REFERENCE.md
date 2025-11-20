# Data Pipeline Quick Reference

## Common Commands

### Import Operations
```bash
# Basic import
python scripts/pipeline_manager.py import

# Import with validation
python scripts/pipeline_manager.py import --validate

# Import without backup (not recommended)
python scripts/pipeline_manager.py import --no-backup
```

### Validation
```bash
# Validate all YAML files
python scripts/pipeline_manager.py validate
```

### Monitoring
```bash
# Quality dashboard
python scripts/pipeline_manager.py dashboard

# Coverage analysis
python scripts/pipeline_manager.py coverage

# Source diversity
python scripts/pipeline_manager.py sources

# Data freshness
python scripts/pipeline_manager.py freshness
```

### Reports
```bash
# Export JSON report
python scripts/pipeline_manager.py export-report --output report.json

# Export HTML report
python scripts/pipeline_manager.py export-report --output report.html --format html
```

### Scheduling
```bash
# Schedule daily import at 2 AM
python scripts/pipeline_manager.py schedule --daily --time 02:00

# Schedule weekly import on Sundays at 3 AM
python scripts/pipeline_manager.py schedule --weekly --day sunday --time 03:00
```

### Workflow Management
```bash
# Submit change request
python scripts/pipeline_manager.py workflow submit \
  --factor-id fuel_diesel \
  --change-type updated \
  --value 2.70 \
  --reason "EPA 2025 update"

# Approve request
python scripts/pipeline_manager.py workflow approve \
  --request-id cr_fuel_diesel_123 \
  --reviewer admin \
  --notes "Verified"

# Reject request
python scripts/pipeline_manager.py workflow reject \
  --request-id cr_fuel_diesel_123 \
  --reviewer admin \
  --notes "Needs more documentation"

# List pending reviews
python scripts/pipeline_manager.py workflow pending --reviewer admin
```

## Python API

### Import
```python
from greenlang.data.pipeline import AutomatedImportPipeline, ImportJob

pipeline = AutomatedImportPipeline("emission_factors.db")
job = ImportJob(
    job_id="import_001",
    job_name="Monthly Update",
    source_files=["data/factors.yaml"],
    target_database="emission_factors.db",
    triggered_by="admin"
)
result = await pipeline.execute_import(job)
```

### Validation
```python
from greenlang.data.pipeline import EmissionFactorValidator

validator = EmissionFactorValidator()
result = await validator.validate_file("data/factors.yaml")
print(f"Quality: {result.quality_score}/100")
```

### Monitoring
```python
from greenlang.data.pipeline import DataQualityMonitor

monitor = DataQualityMonitor("emission_factors.db")
metrics = monitor.calculate_quality_metrics()
print(f"Overall Quality: {metrics.overall_quality_score}/100")
```

### Workflow
```python
from greenlang.data.pipeline import UpdateWorkflow, ApprovalManager, ChangeType

approval_mgr = ApprovalManager("emission_factors.db.versions.db")
workflow = UpdateWorkflow("emission_factors.db", approval_mgr)

request = workflow.submit_change_request(
    factor_id="fuel_diesel",
    change_type=ChangeType.VALUE_CHANGED,
    proposed_changes={"emission_factor_value": 2.70},
    change_reason="EPA 2025 update",
    requested_by="engineer_1"
)
```

## Quality Score Interpretation

| Score | Status | Action |
|-------|--------|--------|
| 90-100 | Excellent | Production ready |
| 75-89 | Good | Minor issues only |
| 60-74 | Fair | Needs improvement |
| 0-59 | Poor | Requires attention |

## Validation Rules

| Rule | Weight | Checks |
|------|--------|--------|
| URI Accessibility | 15% | Source URLs accessible |
| Date Freshness | 20% | Data <3 years old |
| Range Validation | 25% | Reasonable values |
| Unit Validation | 15% | Consistent units |
| Source Validation | 10% | Credible sources |
| Schema Validation | 15% | Complete data |

## File Locations

| Component | File |
|-----------|------|
| Models | `greenlang/data/pipeline/models.py` |
| Validator | `greenlang/data/pipeline/validator.py` |
| Pipeline | `greenlang/data/pipeline/pipeline.py` |
| Monitoring | `greenlang/data/pipeline/monitoring.py` |
| Workflow | `greenlang/data/pipeline/workflow.py` |
| Dashboard | `greenlang/data/pipeline/dashboard.py` |
| CLI Tool | `scripts/pipeline_manager.py` |
| Tests | `tests/test_data_pipeline.py` |

## Database Tables

| Table | Purpose |
|-------|---------|
| emission_factors | Main factors table |
| factor_units | Additional unit variations |
| factor_gas_vectors | Gas decomposition |
| calculation_audit_log | Usage tracking |
| factor_versions | Version history |
| change_log | Audit trail |
| change_requests | Approval workflow |

## Troubleshooting

### Import fails
```bash
# Check validation
python scripts/pipeline_manager.py validate
# Fix issues in YAML files
# Re-run import
```

### Low quality score
```bash
# Check specific metrics
python scripts/pipeline_manager.py dashboard
# Address issues based on scores
```

### Rollback needed
```python
from greenlang.data.pipeline import RollbackManager
rollback_mgr = RollbackManager("emission_factors.db")
rollback_mgr.rollback("backups/backup_xxx.db")
```

## Daily Workflow

```bash
# 1. Morning check
python scripts/pipeline_manager.py dashboard

# 2. Review freshness
python scripts/pipeline_manager.py freshness

# 3. Check coverage
python scripts/pipeline_manager.py coverage

# 4. Export report (weekly)
python scripts/pipeline_manager.py export-report --output weekly.json
```

## Documentation

- Full Guide: `docs/DATA_PIPELINE_GUIDE.md`
- Implementation Summary: `PIPELINE_IMPLEMENTATION_SUMMARY.md`
- Package README: `greenlang/data/pipeline/README.md`
- Quick Reference: `PIPELINE_QUICK_REFERENCE.md` (this file)
