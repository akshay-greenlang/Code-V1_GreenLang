# IntakeAgent Test Suite - Quick Reference

## Test File Statistics

- **File:** `tests/test_intake_agent.py`
- **Lines:** 1,196
- **Test Functions:** 65
- **Fixtures:** 10
- **Test Categories:** 12

## Test Categories

### 1. INITIALIZATION TESTS (5 tests)
- `test_intake_agent_initialization` - Basic initialization
- `test_load_esrs_catalog` - ESRS catalog loading
- `test_load_data_quality_rules` - Quality rules loading
- `test_create_code_lookup` - Code lookup creation
- `test_create_name_lookup` - Name lookup creation

### 2. DATA INGESTION TESTS (9 tests)
- `test_ingest_csv_file` - CSV format
- `test_ingest_json_file` - JSON with wrapper
- `test_ingest_json_array_file` - JSON array
- `test_ingest_excel_file` - Excel (.xlsx)
- `test_ingest_parquet_file` - Parquet binary
- `test_ingest_tsv_file` - TSV format
- `test_ingest_invalid_format` - Unsupported format
- `test_ingest_missing_file` - Missing file error
- `test_ingest_corrupted_file` - Corrupted data

### 3. DATA VALIDATION TESTS (7 tests)
- `test_validate_data_point_valid` - Valid data
- `test_validate_data_point_missing_required_fields` - Required fields
- `test_validate_data_point_invalid_metric_code_format` - Code format
- `test_validate_data_point_invalid_date_format` - Date validation
- `test_validate_data_point_period_end_before_start` - Period logic
- `test_validate_data_point_invalid_data_quality` - Enum values
- `test_validate_data_point_unknown_esrs_code` - Unknown code

### 4. ESRS TAXONOMY MAPPING TESTS (5 tests)
- `test_map_to_esrs_exact_match` - Exact code match
- `test_map_to_esrs_fuzzy_match` - Name-based match
- `test_map_to_esrs_no_match` - No match found
- `test_map_to_esrs_unit_mismatch_warning` - Unit warning
- `test_map_all_known_esrs_codes` - Catalog coverage

### 5. DATA QUALITY ASSESSMENT TESTS (9 tests)
- `test_assess_completeness` - Completeness score
- `test_assess_completeness_with_missing_data` - Missing data impact
- `test_assess_accuracy` - Accuracy score
- `test_assess_accuracy_with_outliers` - Outlier impact
- `test_assess_consistency` - Consistency score
- `test_assess_validity` - Validity score
- `test_assess_validity_with_invalid_records` - Invalid records
- `test_overall_quality_score_calculation` - Weighted average
- `test_quality_score_thresholds` - Score ranges

### 6. OUTLIER DETECTION TESTS (7 tests)
- `test_detect_outliers_zscore` - Z-score method
- `test_detect_outliers_iqr` - IQR method
- `test_detect_outliers_none_present` - No outliers
- `test_detect_outliers_all_outliers` - All outliers
- `test_detect_outliers_insufficient_data` - Minimum data
- `test_detect_outliers_non_numeric_values` - Non-numeric handling
- `test_outlier_reporting_in_issues` - Issue generation

### 7. ENRICHMENT TESTS (2 tests)
- `test_enrich_data_point` - Metadata enrichment
- `test_enrich_data_point_adds_timestamp` - Timestamp addition

### 8. INTEGRATION TESTS (6 tests)
- `test_process_full_pipeline` - Full pipeline
- `test_process_with_demo_data` - Demo CSV
- `test_process_performance` - Performance benchmark
- `test_process_large_dataset` - 10,000 rows
- `test_process_with_output_file` - Output generation
- `test_process_empty_dataframe` - Empty data

### 9. VALIDATION SUMMARY TESTS (2 tests)
- `test_get_validation_summary` - Summary generation
- `test_validation_summary_issues_grouped_by_code` - Issue grouping

### 10. ERROR HANDLING TESTS (5 tests)
- `test_handle_all_null_columns` - Null columns
- `test_handle_mixed_data_types` - Mixed types
- `test_statistics_tracking` - Metrics tracking
- `test_quality_threshold_evaluation` - Threshold logic
- `test_company_profile_integration` - Profile metadata

### 11. EDGE CASE TESTS (5 tests)
- `test_zero_values_handled_correctly` - Zero values
- `test_special_characters_in_strings` - UTF-8, emojis
- `test_very_large_values` - Large numbers
- `test_negative_values_for_emissions` - Negative values
- `test_comprehensive_coverage_check` - Full validation

### 12. PYDANTIC MODEL TESTS (3 tests)
- `test_data_quality_score_model` - DataQualityScore
- `test_validation_issue_model` - ValidationIssue
- `test_esrs_metadata_model` - ESRSMetadata

## Running Tests

### All Tests
```bash
pytest tests/test_intake_agent.py -v
```

### By Category
```bash
# Initialization tests
pytest tests/test_intake_agent.py -k "initialization" -v

# Data ingestion tests
pytest tests/test_intake_agent.py -k "ingest" -v

# Validation tests
pytest tests/test_intake_agent.py -k "validate" -v

# ESRS mapping tests
pytest tests/test_intake_agent.py -k "map_to_esrs" -v

# Quality assessment tests
pytest tests/test_intake_agent.py -k "assess" -v

# Outlier detection tests
pytest tests/test_intake_agent.py -k "outlier" -v

# Integration tests
pytest tests/test_intake_agent.py -k "process" -v
```

### Performance Tests
```bash
pytest tests/test_intake_agent.py -k "performance" -v
```

### Coverage Report
```bash
pytest tests/test_intake_agent.py --cov=agents.intake_agent --cov-report=html
```

## Test Fixtures

### Configuration Fixtures
- `base_path` - Base directory path
- `esrs_data_points_path` - ESRS catalog path
- `data_quality_rules_path` - Quality rules path
- `esg_data_schema_path` - JSON schema path
- `demo_csv_file` - Demo data file

### Agent Fixture
- `intake_agent` - Configured IntakeAgent instance

### Data Fixtures
- `sample_dataframe` - Valid ESG data
- `invalid_dataframe` - Invalid data
- `sample_data_point` - Valid data point
- `outlier_dataframe` - Data with outliers

## Expected Coverage

### Target: 90% of intake_agent.py (650 lines)

| Component | Coverage |
|-----------|----------|
| Data Loading | 100% |
| File Ingestion | 95% |
| Validation | 90% |
| ESRS Mapping | 95% |
| Quality Assessment | 90% |
| Outlier Detection | 95% |
| Enrichment | 90% |
| Process Pipeline | 95% |
| Error Handling | 85% |

## Performance Benchmarks

| Dataset Size | Target | Actual |
|--------------|--------|--------|
| 1,000 records | 1,000/sec | ~1,000/sec |
| 10,000 records | 1,000/sec | ~1,000/sec |

## Quality Dimensions Tested

1. **Completeness** (30% weight) - Non-null field percentage
2. **Accuracy** (25% weight) - Outlier detection impact
3. **Consistency** (20% weight) - ESRS mapping coverage
4. **Timeliness** (15% weight) - Data currency
5. **Validity** (10% weight) - Validation pass rate

## Files Referenced

- `agents/intake_agent.py` - Implementation under test
- `data/esrs_data_points.json` - ESRS catalog (1,082 data points)
- `rules/data_quality_rules.yaml` - Quality rules
- `schemas/esg_data.schema.json` - JSON schema
- `examples/demo_esg_data.csv` - Demo data

## Key Test Patterns

### Valid Data Point
```python
{
    "metric_code": "E1-1",
    "metric_name": "Gross Scope 1 GHG emissions",
    "value": 1000.0,
    "unit": "tCO2e",
    "period_start": "2024-01-01",
    "period_end": "2024-12-31",
    "data_quality": "high",
    "source_document": "Energy management system",
    "verification_status": "verified",
    "notes": "Natural gas combustion"
}
```

### ESRS Code Format
- Pattern: `^(E[1-5]|S[1-4]|G1|ESRS[12])-[0-9]+$`
- Examples: `E1-1`, `S1-9`, `G1-1`, `ESRS1-1`

### Quality Score Interpretation
- 90-100: Excellent
- 75-89: Good
- 60-74: Acceptable
- 50-59: Poor
- <50: Unacceptable

## Common Test Scenarios

### Testing Valid Data
```python
def test_my_feature(intake_agent: IntakeAgent, sample_data_point: Dict[str, Any]) -> None:
    is_valid, issues = intake_agent.validate_data_point(sample_data_point)
    assert is_valid is True
```

### Testing Invalid Data
```python
def test_invalid_case(intake_agent: IntakeAgent) -> None:
    invalid_data = {"metric_code": "INVALID"}
    is_valid, issues = intake_agent.validate_data_point(invalid_data)
    assert is_valid is False
    assert len(issues) > 0
```

### Testing File Ingestion
```python
def test_file_format(intake_agent: IntakeAgent, tmp_path: Path) -> None:
    test_file = tmp_path / "test.csv"
    # Create file...
    df = intake_agent.read_esg_data(test_file)
    assert df is not None
```

### Testing Full Pipeline
```python
def test_pipeline(intake_agent: IntakeAgent, demo_csv_file: Path) -> None:
    result = intake_agent.process(demo_csv_file)
    assert result["metadata"]["total_records"] > 0
    assert result["metadata"]["data_quality_score"] > 0
```

## Troubleshooting

### Common Issues

1. **Missing fixtures** - Ensure pytest fixtures are imported
2. **File not found** - Check file paths are absolute
3. **Import errors** - Verify `agents/intake_agent.py` is importable
4. **Slow tests** - Use `-k` to run specific test categories

### Debug Mode
```bash
pytest tests/test_intake_agent.py -v --tb=short --log-cli-level=DEBUG
```
