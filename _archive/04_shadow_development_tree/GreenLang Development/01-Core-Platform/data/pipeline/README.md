# GreenLang Enterprise Data Pipeline

Production-grade data pipeline system for managing 500+ emission factors with automated imports, validation, versioning, and monitoring.

## Features

### Core Capabilities

- **Automated Import Pipeline**
  - Scheduled jobs (daily/weekly)
  - Pre-import validation
  - Duplicate detection & resolution
  - Success/failure logging
  - Automatic rollback on failure

- **Multi-Layer Validation**
  - URI accessibility checks
  - Emission factor range validation
  - Source organization verification
  - Date freshness checks (3-year threshold)
  - Unit consistency validation
  - Schema compliance

- **Version Control & Changelog**
  - Complete factor versioning
  - Change tracking (what, when, why)
  - Deprecation warnings
  - Migration path documentation

- **Data Quality Monitoring**
  - Coverage gap analysis
  - Source diversity tracking
  - Geographic coverage heatmap
  - Data freshness dashboard
  - Quality scoring (0-100)

- **Update Workflow**
  - Change request submission
  - Review & approval workflow
  - Testing before deployment
  - User communication

## Quick Start

### Installation

```bash
pip install pydantic httpx pyyaml schedule rich
```

### Basic Usage

```python
from greenlang.data.pipeline import (
    AutomatedImportPipeline,
    EmissionFactorValidator,
    DataQualityMonitor
)

# 1. Run import
pipeline = AutomatedImportPipeline("emission_factors.db")
job = ImportJob(
    job_id="import_001",
    job_name="Monthly Update",
    source_files=["data/factors.yaml"],
    target_database="emission_factors.db",
    triggered_by="admin"
)
result = await pipeline.execute_import(job)

# 2. Validate data
validator = EmissionFactorValidator()
result = await validator.validate_file("data/factors.yaml")
print(f"Quality Score: {result.quality_score}/100")

# 3. Monitor quality
monitor = DataQualityMonitor("emission_factors.db")
metrics = monitor.calculate_quality_metrics()
print(f"Overall Quality: {metrics.overall_quality_score}/100")
```

### CLI Usage

```bash
# Import data
python scripts/pipeline_manager.py import --validate

# Validate files
python scripts/pipeline_manager.py validate

# View dashboard
python scripts/pipeline_manager.py dashboard

# Export report
python scripts/pipeline_manager.py export-report --output report.json

# Schedule daily import
python scripts/pipeline_manager.py schedule --daily --time 02:00
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Sources                             │
│              (YAML, APIs, CSV, Excel)                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Validation Framework                            │
│  ✓ URI Check  ✓ Range  ✓ Freshness  ✓ Schema              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│           Automated Import Pipeline                          │
│  • Pre-validation  • Backup  • Rollback  • Logging          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  SQLite Database                             │
│  • Factors  • Versions  • Changelog  • Audit                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│            Quality Monitoring Dashboard                      │
│  • Metrics  • Coverage  • Freshness  • Diversity            │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Validation Framework

**File:** `validator.py`

Six validation rules with weighted scoring:

| Rule | Weight | Purpose |
|------|--------|---------|
| URI Accessibility | 15% | Verifies source URLs are accessible |
| Date Freshness | 20% | Ensures data is recent (<3 years) |
| Range Validation | 25% | Checks values are reasonable |
| Unit Validation | 15% | Validates unit consistency |
| Source Validation | 10% | Verifies credible sources |
| Schema Validation | 15% | Ensures completeness |

**Example:**
```python
validator = EmissionFactorValidator()
result = await validator.validate_factor(factor_id, data)

if result.is_valid:
    print(f"Quality: {result.quality_score}/100")
else:
    for error in result.errors:
        print(f"Error: {error['message']}")
```

### 2. Import Pipeline

**File:** `pipeline.py`

Enterprise-grade import with:
- Pre-import validation
- Database backup
- Transaction management
- Automatic rollback
- Progress tracking

**Example:**
```python
pipeline = AutomatedImportPipeline(
    db_path="emission_factors.db",
    backup_dir="backups",
    enable_versioning=True
)

result = await pipeline.execute_import(
    job,
    validate_before=True,
    rollback_on_failure=True
)

print(f"Status: {result.status}")
print(f"Success Rate: {result.success_rate}%")
```

### 3. Monitoring System

**File:** `monitoring.py`

Four analysis components:

**Coverage Analyzer**
```python
analyzer = CoverageAnalyzer(db_path)
coverage = analyzer.analyze_category_coverage()
# Identifies gaps in categories, geographies, scopes
```

**Source Diversity Analyzer**
```python
analyzer = SourceDiversityAnalyzer(db_path)
diversity = analyzer.analyze_source_distribution()
# Tracks source diversity (HHI index)
```

**Freshness Tracker**
```python
tracker = FreshnessTracker(db_path)
freshness = tracker.analyze_freshness()
# Identifies stale data (>3 years)
```

**Quality Monitor**
```python
monitor = DataQualityMonitor(db_path)
metrics = monitor.calculate_quality_metrics()
# Overall quality score (0-100)
```

### 4. Update Workflow

**File:** `workflow.py`

Controlled update process:

**Submit Change Request**
```python
workflow = UpdateWorkflow(db_path, approval_manager)
request = workflow.submit_change_request(
    factor_id="fuel_diesel",
    change_type=ChangeType.VALUE_CHANGED,
    proposed_changes={"emission_factor_value": 2.70},
    change_reason="EPA 2025 update",
    requested_by="engineer_1"
)
```

**Approval Levels**
- **Automatic**: Minor metadata changes
- **Peer Review**: Standard updates (<10% change)
- **Admin Approval**: Major changes (>10% change)
- **Committee Review**: New factors or deprecation

**Implement Approved Changes**
```python
# After approval
success = workflow.implement_approved_change(
    request_id=request.request_id,
    implemented_by="engineer_1"
)
```

### 5. Dashboard & CLI

**File:** `dashboard.py`

Interactive monitoring dashboard:

```python
dashboard = DataQualityDashboard(db_path)

# Display overview
dashboard.display_overview()

# Display coverage
dashboard.display_coverage_report()

# Export report
dashboard.export_report("report.html", format="html")
```

**CLI Commands:**
```bash
# View dashboard
python pipeline_manager.py dashboard

# Analyze coverage
python pipeline_manager.py coverage

# Check freshness
python pipeline_manager.py freshness

# Export report
python pipeline_manager.py export-report -o report.json
```

## Data Models

### ImportJob
```python
ImportJob(
    job_id="import_20250119",
    job_name="Monthly Update",
    source_files=["factors.yaml"],
    target_database="factors.db",
    status=ImportStatus.PENDING,
    total_factors_processed=0,
    successful_imports=0,
    failed_imports=0,
    validation_result=None,
    can_rollback=True
)
```

### ValidationResult
```python
ValidationResult(
    validation_id="val_001",
    is_valid=True,
    quality_score=87.5,
    total_records=500,
    valid_records=485,
    invalid_records=15,
    errors=[],
    warnings=[],
    validation_duration_ms=1234.5
)
```

### DataQualityMetrics
```python
DataQualityMetrics(
    overall_quality_score=85.2,
    completeness_score=90.0,
    accuracy_score=95.0,
    freshness_score=75.0,
    consistency_score=88.0,
    total_factors=500,
    unique_sources=12,
    stale_factors_count=25
)
```

### ChangeRequest
```python
ChangeRequest(
    request_id="cr_001",
    change_type=ChangeType.VALUE_CHANGED,
    factor_id="fuel_diesel",
    proposed_changes={"value": 2.70},
    current_values={"value": 2.68},
    change_reason="EPA 2025 update",
    review_status=ReviewStatus.PENDING_REVIEW,
    approved=False
)
```

## Configuration

### Validation Thresholds

```python
# Date freshness
max_age_years = 3
warning_age_years = 2

# Range validation
expected_ranges = {
    'fuels': {'natural_gas': (0.05, 0.3)},
    'electricity': {'grid': (0.1, 1.2)}
}

# Quality score thresholds
EXCELLENT = 90  # Production ready
GOOD = 75       # Minor issues
FAIR = 60       # Needs improvement
POOR = 0        # Requires attention
```

### Scheduled Imports

```python
# Daily import at 2 AM
scheduler.schedule_daily(time_str="02:00")

# Weekly import on Sundays at 3 AM
scheduler.schedule_weekly(day="sunday", time_str="03:00")
```

### Backup Retention

```python
# Keep backups for 90 days
rollback_mgr.cleanup_old_backups(keep_days=90)
```

## Quality Scoring

### Overall Quality Score

```
Overall = (
    Completeness × 0.25 +
    Accuracy × 0.25 +
    Freshness × 0.25 +
    Consistency × 0.15 +
    Source Diversity × 0.10
)
```

### Interpretation

- **90-100**: Excellent - Production ready
- **75-89**: Good - Minor issues only
- **60-74**: Fair - Needs improvement
- **0-59**: Poor - Requires attention

## Best Practices

### 1. Always Validate Before Import
```python
job.validate_before_import = True
```

### 2. Enable Backups
```python
result = await pipeline.execute_import(
    job,
    rollback_on_failure=True
)
```

### 3. Monitor Quality Weekly
```bash
python pipeline_manager.py dashboard
```

### 4. Regular Freshness Audits
```bash
python pipeline_manager.py freshness
```

### 5. Use Approval Workflow
```python
request = workflow.submit_change_request(...)
```

### 6. Document All Changes
```python
change_reason="Updated based on EPA 2025 data release"
supporting_docs=["https://epa.gov/2025-update.pdf"]
```

## Scaling to 10,000+ Factors

### Database Optimization
```sql
-- Add composite indexes
CREATE INDEX idx_composite ON emission_factors(category, source_org);
```

### Batch Processing
```python
BATCH_SIZE = 1000
for batch in chunks(factors, BATCH_SIZE):
    process_batch(batch)
```

### Parallel Validation
```python
tasks = [validator.validate_file(f) for f in files]
results = await asyncio.gather(*tasks)
```

## Testing

```bash
# Run all tests
pytest tests/test_data_pipeline.py -v

# Run specific test
pytest tests/test_data_pipeline.py::TestValidation -v

# Run with coverage
pytest tests/test_data_pipeline.py --cov=greenlang.data.pipeline
```

## Troubleshooting

### Import Failures

**Problem:** Import fails with validation errors

**Solution:**
```bash
# Identify issues
python pipeline_manager.py validate

# Fix YAML files
# Re-run import
python pipeline_manager.py import
```

### Low Quality Score

**Problem:** Overall quality < 70

**Solution:**
```python
metrics = monitor.calculate_quality_metrics()

# Check specific scores
if metrics.freshness_score < 60:
    # Update old factors

if metrics.completeness_score < 60:
    # Add missing fields
```

### Rollback Fails

**Problem:** Cannot restore backup

**Solution:**
```python
# Manual restoration
import shutil
shutil.copy2("backups/backup_xxx.db", "emission_factors.db")
```

## Performance

**Benchmarks** (500 factors):
- Import: ~5 seconds
- Validation: ~10 seconds
- Quality Metrics: ~2 seconds
- Coverage Analysis: ~1 second

**Scalability** (10,000 factors):
- Import: ~60 seconds
- Validation: ~120 seconds
- Quality Metrics: ~10 seconds

## Support

**Documentation:** See `docs/DATA_PIPELINE_GUIDE.md`

**Issues:** Report bugs or feature requests via GitHub Issues

**Contact:** support@greenlang.io

## License

Copyright 2025 GreenLang. All rights reserved.
