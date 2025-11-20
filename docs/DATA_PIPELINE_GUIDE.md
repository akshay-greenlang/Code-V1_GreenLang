# GreenLang Enterprise Data Pipeline Guide

## Overview

The GreenLang Data Pipeline is a production-grade system for managing emission factors with automated imports, comprehensive validation, version control, and data quality monitoring.

**Key Features:**
- Automated scheduled imports (daily/weekly)
- Multi-layer validation framework
- Automatic rollback on failure
- Version control and change tracking
- Data quality monitoring dashboard
- Approval workflow for updates
- Coverage gap analysis
- Source diversity tracking

## Architecture

```
Data Pipeline Architecture
┌─────────────────────────────────────────────────────────────┐
│                      Data Sources                            │
│  (YAML Files, APIs, Manual Submissions)                     │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│              Validation Framework                            │
│  • URI Accessibility    • Range Validation                  │
│  • Date Freshness      • Unit Consistency                   │
│  • Schema Compliance   • Source Verification                │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│           Automated Import Pipeline                          │
│  • Pre-import validation                                     │
│  • Database backup                                           │
│  • Transaction management                                    │
│  • Rollback on failure                                       │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│                SQLite Database                               │
│  • emission_factors    • factor_versions                    │
│  • change_log          • change_requests                    │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│          Data Quality Monitoring                             │
│  • Quality metrics    • Coverage analysis                   │
│  • Source diversity   • Freshness tracking                  │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

```bash
pip install pydantic httpx pyyaml schedule rich
```

### Setup

```python
from greenlang.data.pipeline import (
    AutomatedImportPipeline,
    EmissionFactorValidator,
    DataQualityMonitor,
    UpdateWorkflow
)
```

## Quick Start

### 1. Run Import Pipeline

```bash
# Basic import
python scripts/pipeline_manager.py import

# Import with validation
python scripts/pipeline_manager.py import --validate

# Import without backup (not recommended)
python scripts/pipeline_manager.py import --no-backup
```

### 2. Validate Data

```bash
# Validate all YAML files
python scripts/pipeline_manager.py validate
```

### 3. View Quality Dashboard

```bash
# Show overall quality metrics
python scripts/pipeline_manager.py dashboard

# Show coverage analysis
python scripts/pipeline_manager.py coverage

# Show source diversity
python scripts/pipeline_manager.py sources

# Show data freshness
python scripts/pipeline_manager.py freshness
```

### 4. Export Reports

```bash
# Export JSON report
python scripts/pipeline_manager.py export-report --output report.json

# Export HTML report
python scripts/pipeline_manager.py export-report --output report.html --format html
```

## Automated Import Pipeline

### Configuration

```python
from greenlang.data.pipeline import AutomatedImportPipeline, ImportJob

pipeline = AutomatedImportPipeline(
    db_path="path/to/emission_factors.db",
    backup_dir="path/to/backups",
    enable_versioning=True
)

# Create import job
job = ImportJob(
    job_id="import_20250119",
    job_name="Monthly Update",
    source_files=[
        "data/emission_factors_registry.yaml",
        "data/emission_factors_expansion_phase1.yaml",
    ],
    target_database="emission_factors.db",
    validate_before_import=True,
    triggered_by="admin_user",
    trigger_type="manual"
)

# Execute import
import asyncio
result = asyncio.run(pipeline.execute_import(job))

print(f"Status: {result.status}")
print(f"Imported: {result.successful_imports}/{result.total_factors_processed}")
print(f"Success Rate: {result.success_rate:.1f}%")
```

### Scheduled Imports

```python
from greenlang.data.pipeline import ScheduledImporter

scheduler = ScheduledImporter(
    pipeline=pipeline,
    source_files=yaml_files,
    db_path="emission_factors.db"
)

# Schedule daily import at 2 AM
scheduler.schedule_daily(time_str="02:00")

# Schedule weekly import on Sundays at 3 AM
scheduler.schedule_weekly(day="sunday", time_str="03:00")

# Run scheduler (blocking)
scheduler.run_scheduler()
```

**CLI Usage:**

```bash
# Schedule daily import
python scripts/pipeline_manager.py schedule --daily --time 02:00

# Schedule weekly import
python scripts/pipeline_manager.py schedule --weekly --day sunday --time 03:00
```

## Validation Framework

### Validation Rules

The pipeline includes 6 validation rules:

1. **URI Accessibility** (15% weight)
   - Checks that source URIs are accessible
   - Validates HTTP response codes
   - Tracks response times

2. **Date Freshness** (20% weight)
   - Ensures data is not older than 3 years
   - Warns if data is older than 2 years
   - Calculates age metrics

3. **Range Validation** (25% weight)
   - Validates emission factors are within reasonable ranges
   - Category-specific range checks
   - Detects outliers and anomalies

4. **Unit Validation** (15% weight)
   - Ensures units are valid and consistent
   - Checks unit compatibility with categories

5. **Source Validation** (10% weight)
   - Verifies source organizations are credible
   - Checks against recognized sources list

6. **Schema Validation** (15% weight)
   - Validates required fields are present
   - Checks field types
   - Ensures data completeness

### Running Validation

```python
from greenlang.data.pipeline import EmissionFactorValidator

validator = EmissionFactorValidator()

# Validate single factor
result = await validator.validate_factor(
    factor_id="fuel_diesel",
    factor_data={
        "name": "Diesel Fuel",
        "emission_factor_kg_co2e_per_liter": 2.68,
        "scope": "Scope 1",
        "source": "EPA",
        "uri": "https://www.epa.gov/...",
        "last_updated": "2024-11-01"
    }
)

print(f"Valid: {result.is_valid}")
print(f"Quality Score: {result.quality_score}/100")
print(f"Errors: {len(result.errors)}")
print(f"Warnings: {len(result.warnings)}")

# Validate entire YAML file
result = await validator.validate_file("data/emission_factors_registry.yaml")
```

### Quality Scoring

Quality scores are calculated as weighted averages:

```
Overall Score = (
    URI Accessibility × 0.15 +
    Date Freshness × 0.20 +
    Range Validation × 0.25 +
    Unit Validation × 0.15 +
    Source Validation × 0.10 +
    Schema Validation × 0.15
) × 100
```

**Score Interpretation:**
- **90-100**: Excellent - Production ready
- **75-89**: Good - Minor issues only
- **60-74**: Fair - Needs improvement
- **0-59**: Poor - Requires attention

## Data Quality Monitoring

### Quality Metrics

```python
from greenlang.data.pipeline import DataQualityMonitor

monitor = DataQualityMonitor(db_path="emission_factors.db")

# Calculate comprehensive metrics
metrics = monitor.calculate_quality_metrics()

print(f"Overall Quality: {metrics.overall_quality_score}/100")
print(f"Completeness: {metrics.completeness_score}/100")
print(f"Accuracy: {metrics.accuracy_score}/100")
print(f"Freshness: {metrics.freshness_score}/100")
print(f"Consistency: {metrics.consistency_score}/100")

print(f"\nTotal Factors: {metrics.total_factors}")
print(f"Unique Sources: {metrics.unique_sources}")
print(f"Stale Factors: {metrics.stale_factors_count}")
```

### Coverage Analysis

```python
from greenlang.data.pipeline import CoverageAnalyzer

analyzer = CoverageAnalyzer(db_path="emission_factors.db")

# Analyze category coverage
coverage = analyzer.analyze_category_coverage()

print(f"Total Categories: {coverage['total_categories']}")
print(f"Categories Below Threshold: {coverage['categories_below_threshold']}")

for category, data in coverage['coverage_by_category'].items():
    print(f"{category}: {data['total']} factors")

# Identify gaps
for gap in coverage['subcategory_gaps']:
    print(f"Gap: {gap['category']}/{gap['subcategory']} - {gap['current_count']} factors")
```

### Source Diversity

```python
from greenlang.data.pipeline import SourceDiversityAnalyzer

analyzer = SourceDiversityAnalyzer(db_path="emission_factors.db")

diversity = analyzer.analyze_source_distribution()

print(f"Unique Sources: {diversity['unique_sources']}")
print(f"Diversity Score: {diversity['diversity_score']}/100")
print(f"Top 3 Concentration: {diversity['top_3_concentration_pct']:.1f}%")

# Identify missing sources
missing = analyzer.identify_source_gaps()
print(f"Recommended sources to add: {', '.join(missing)}")
```

### Data Freshness

```python
from greenlang.data.pipeline import FreshnessTracker

tracker = FreshnessTracker(db_path="emission_factors.db")

freshness = tracker.analyze_freshness()

print(f"Freshness Score: {freshness['freshness_score']}/100")
print(f"Average Age: {freshness['average_age_years']:.1f} years")

dist = freshness['distribution']
print(f"Fresh (<1y): {dist['fresh_count']}")
print(f"Recent (1-2y): {dist['recent_count']}")
print(f"Aging (2-3y): {dist['aging_count']}")
print(f"Stale (>3y): {dist['stale_count']}")

# Get oldest factors
for factor in freshness['stale_factors'][:5]:
    print(f"  {factor['factor_id']}: {factor['age_years']} years old")
```

## Update Workflow

### Submitting Change Requests

```python
from greenlang.data.pipeline import UpdateWorkflow, ApprovalManager, ChangeType

approval_manager = ApprovalManager(version_db_path="emission_factors.db.versions.db")
workflow = UpdateWorkflow(
    db_path="emission_factors.db",
    approval_manager=approval_manager
)

# Submit change request
request = workflow.submit_change_request(
    factor_id="fuel_diesel",
    change_type=ChangeType.VALUE_CHANGED,
    proposed_changes={
        "emission_factor_value": 2.70,
        "source_uri": "https://www.epa.gov/new-data"
    },
    change_reason="Updated based on EPA 2025 data release",
    requested_by="data_engineer_1",
    supporting_docs=["https://epa.gov/2025-update.pdf"],
    source_refs=["EPA GHG Inventory 2025"]
)

print(f"Request ID: {request.request_id}")
print(f"Status: {request.review_status}")
print(f"Impact: {request.impact_assessment['impact_level']}")
```

**CLI Usage:**

```bash
# Submit change request
python scripts/pipeline_manager.py workflow submit \
  --factor-id fuel_diesel \
  --change-type updated \
  --value 2.70 \
  --uri "https://www.epa.gov/new-data" \
  --reason "Updated based on EPA 2025 data"
```

### Approval Process

```python
# Approve request
approval_manager.approve_request(
    request_id="cr_fuel_diesel_1234567890",
    reviewed_by="data_admin",
    review_notes="Verified against EPA source. Approved for implementation."
)

# Reject request
approval_manager.reject_request(
    request_id="cr_fuel_diesel_1234567890",
    reviewed_by="data_admin",
    review_notes="Source data not verified. Please provide additional documentation."
)
```

**CLI Usage:**

```bash
# Approve request
python scripts/pipeline_manager.py workflow approve \
  --request-id cr_fuel_diesel_1234567890 \
  --reviewer data_admin \
  --notes "Verified against EPA source"

# Reject request
python scripts/pipeline_manager.py workflow reject \
  --request-id cr_fuel_diesel_1234567890 \
  --reviewer data_admin \
  --notes "Source data not verified"

# List pending reviews
python scripts/pipeline_manager.py workflow pending \
  --reviewer data_admin
```

### Implementing Changes

```python
# Implement approved change
success = workflow.implement_approved_change(
    request_id="cr_fuel_diesel_1234567890",
    implemented_by="data_engineer_1"
)

if success:
    print("Change implemented successfully")
    # Version record created
    # Changelog entry added
    # Database updated
```

## Rollback Capability

### Manual Rollback

```python
from greenlang.data.pipeline import RollbackManager

rollback_mgr = RollbackManager(
    db_path="emission_factors.db",
    backup_dir="backups"
)

# Create backup before risky operation
backup_path = rollback_mgr.create_backup(job_id="manual_backup")

# ... perform operation ...

# Rollback if needed
if something_went_wrong:
    success = rollback_mgr.rollback(backup_path)
```

### Automatic Rollback

The import pipeline automatically creates backups and rolls back on failure:

```python
job = ImportJob(...)
result = await pipeline.execute_import(
    job,
    validate_before=True,
    rollback_on_failure=True  # Automatic rollback enabled
)

if result.status == ImportStatus.ROLLED_BACK:
    print("Import failed and was rolled back")
    print(f"Backup restored from: {result.backup_path}")
```

### Backup Management

```python
# Cleanup old backups (keep 30 days)
deleted_count = rollback_mgr.cleanup_old_backups(keep_days=30)
print(f"Deleted {deleted_count} old backups")
```

## Version Control

### Version History

Every change to emission factors is tracked:

```python
# Version records include:
version = FactorVersion(
    version_id="v_fuel_diesel_2",
    factor_id="fuel_diesel",
    version_number=2,
    factor_data={"emission_factor_value": 2.70},  # New data
    previous_data={"emission_factor_value": 2.68},  # Old data
    change_type=ChangeType.VALUE_CHANGED,
    change_summary="Updated to EPA 2025 data",
    changed_fields=["emission_factor_value"],
    changed_by="data_engineer_1",
    change_reason="EPA released new data",
    version_timestamp=datetime.now(),
    effective_from=datetime.now(),
    validation_passed=True,
    data_hash="sha256_hash"
)
```

### Changelog

```python
# Query changelog
conn = sqlite3.connect("emission_factors.db.versions.db")
cursor = conn.cursor()

cursor.execute("""
    SELECT log_id, timestamp, change_type, factor_id, summary
    FROM change_log
    WHERE factor_id = ?
    ORDER BY timestamp DESC
    LIMIT 10
""", ("fuel_diesel",))

for row in cursor.fetchall():
    print(f"{row[1]}: {row[2]} - {row[4]}")
```

## Monitoring Dashboard

### Interactive Dashboard

```python
from greenlang.data.pipeline import DataQualityDashboard

dashboard = DataQualityDashboard(db_path="emission_factors.db")

# Display overview
dashboard.display_overview()

# Display coverage report
dashboard.display_coverage_report()

# Display source analysis
dashboard.display_source_analysis()

# Display freshness analysis
dashboard.display_freshness_analysis()
```

### Export Reports

```python
# Export JSON report
dashboard.export_report(
    output_path="quality_report.json",
    format="json"
)

# Export HTML report
dashboard.export_report(
    output_path="quality_report.html",
    format="html"
)
```

## Best Practices

### 1. Always Validate Before Import

```python
job.validate_before_import = True
```

### 2. Enable Backups for Production

```python
result = await pipeline.execute_import(
    job,
    rollback_on_failure=True
)
```

### 3. Monitor Quality Metrics

```bash
# Daily quality check
python scripts/pipeline_manager.py dashboard

# Weekly coverage review
python scripts/pipeline_manager.py coverage
```

### 4. Regular Freshness Audits

```bash
# Identify stale factors
python scripts/pipeline_manager.py freshness
```

### 5. Use Approval Workflow for Production

```python
# Submit change requests instead of direct updates
request = workflow.submit_change_request(...)
```

### 6. Document All Changes

```python
change_reason="Updated based on EPA 2025 data release",
supporting_docs=["https://epa.gov/2025-update.pdf"]
```

### 7. Regular Backup Cleanup

```python
# Monthly cleanup of old backups
rollback_mgr.cleanup_old_backups(keep_days=90)
```

## Troubleshooting

### Import Failures

**Problem:** Import job fails with validation errors

**Solution:**
```bash
# Run validation first to identify issues
python scripts/pipeline_manager.py validate

# Check validation results
# Fix YAML files based on errors
# Re-run import
python scripts/pipeline_manager.py import
```

### Rollback Issues

**Problem:** Rollback fails

**Solution:**
```python
# Check backup exists
backup_path = "backups/backup_job123_20250119_020000.db"
if not Path(backup_path).exists():
    print("Backup not found!")

# Manual database restoration
import shutil
shutil.copy2(backup_path, "emission_factors.db")
```

### Quality Score Low

**Problem:** Overall quality score below 70

**Solution:**
```python
# Identify specific issues
metrics = monitor.calculate_quality_metrics()

# Address completeness
if metrics.completeness_score < 60:
    # Add missing fields to YAML files

# Address freshness
if metrics.freshness_score < 60:
    # Update old factors

# Address source diversity
if metrics.unique_sources < 5:
    # Add factors from additional sources
```

## Scaling to 10,000+ Factors

### Database Optimization

```sql
-- Add additional indexes for large datasets
CREATE INDEX idx_composite_category_source ON emission_factors(category, source_org);
CREATE INDEX idx_composite_geo_quality ON emission_factors(country_code, data_quality_tier);
```

### Batch Processing

```python
# For large imports, process in batches
BATCH_SIZE = 1000

for i in range(0, len(factors), BATCH_SIZE):
    batch = factors[i:i + BATCH_SIZE]
    # Process batch
```

### Parallel Validation

```python
import asyncio

# Validate files in parallel
tasks = [validator.validate_file(f) for f in yaml_files]
results = await asyncio.gather(*tasks)
```

### Streaming Large Files

```python
# For very large YAML files (>100MB)
import yaml

with open(large_file, 'r') as f:
    for document in yaml.safe_load_all(f):
        # Process each document
        pass
```

## Support

For issues or questions:
- GitHub Issues: https://github.com/greenlang/greenlang
- Documentation: https://docs.greenlang.io
- Email: support@greenlang.io

## License

Copyright 2025 GreenLang. All rights reserved.
