# GreenLang Validation and I/O Testing Guide

This directory contains comprehensive unit tests for GreenLang's validation and I/O modules.

## Test Structure

```
tests/unit/
├── validation/
│   ├── __init__.py
│   ├── conftest.py              # Shared fixtures for validation tests
│   ├── test_framework.py        # ValidationFramework tests (561 lines)
│   ├── test_schema.py           # SchemaValidator tests (464 lines)
│   └── test_rules.py            # RulesEngine tests (730 lines)
└── io/
    ├── __init__.py
    ├── conftest.py              # Shared fixtures for I/O tests
    ├── test_readers.py          # DataReader tests (497 lines)
    └── test_writers.py          # DataWriter tests (542 lines)
```

## Validation Tests

### test_framework.py (561 lines)

Comprehensive tests for the ValidationFramework class covering:

- **ValidationError Tests**
  - Error creation with all fields
  - Severity levels (ERROR, WARNING, INFO)
  - String representation
  - Value and expected field tracking

- **ValidationResult Tests**
  - Result creation and initialization
  - Adding errors, warnings, and info messages
  - Merging multiple results
  - Error/warning counting
  - Summary generation
  - Metadata handling

- **ValidationFramework Tests**
  - Framework initialization
  - Adding/removing validators
  - Validator configuration
  - Single and multiple validator execution
  - Stop-on-error functionality
  - Enabling/disabling validators
  - Exception handling in validators
  - Pre/post validator hooks
  - Batch validation
  - Validation summaries

**Key Features Tested:**
- Pre-validator hooks (run before validation)
- Post-validator hooks (run after validation)
- Batch validation with summary statistics
- Custom validator configurations
- Severity-based filtering
- Metadata tracking

### test_schema.py (464 lines)

Comprehensive tests for JSON Schema validation:

- **Basic Schema Validation**
  - Object and array schemas
  - Required field validation
  - Type checking (string, integer, boolean, etc.)
  - Nested object validation

- **Advanced Constraints**
  - Minimum/maximum values
  - String length constraints
  - Pattern matching (regex)
  - Enum validation
  - Additional properties control
  - Unique items in arrays

- **Schema Features**
  - oneOf, allOf, anyOf validation
  - Custom format validators (email, date, uri)
  - Field path tracking for errors
  - Schema compilation and caching

- **Graceful Degradation**
  - Basic validation when jsonschema unavailable
  - Fallback type checking
  - Required field validation

**Key Features Tested:**
- JSON Schema Draft 7 support
- Complex nested schemas
- Format validators
- Error location tracking
- Loading schemas from files

### test_rules.py (730 lines)

Comprehensive tests for the Business Rules Engine:

- **All 12 Rule Operators**
  - `==` (EQUALS)
  - `!=` (NOT_EQUALS)
  - `>` (GREATER_THAN)
  - `>=` (GREATER_EQUAL)
  - `<` (LESS_THAN)
  - `<=` (LESS_EQUAL)
  - `in` (IN)
  - `not_in` (NOT_IN)
  - `contains` (CONTAINS)
  - `regex` (REGEX)
  - `is_null` (IS_NULL)
  - `not_null` (NOT_NULL)

- **Rule Management**
  - Creating and adding rules
  - Rule sets for organization
  - Enabling/disabling rules
  - Custom error messages
  - Severity levels

- **Advanced Features**
  - Nested field path support (e.g., `address.city`)
  - Conditional rules (execute only if condition met)
  - Rule evaluation order
  - Type mismatch handling

- **Error Handling**
  - Custom error messages
  - Default error messages
  - Error location tracking
  - Metadata about rules evaluated

**Key Features Tested:**
- All comparison operators
- Nested field access with dot notation
- Conditional rule execution
- Rule sets for grouping
- Loading rules from configuration

## I/O Tests

### test_readers.py (497 lines)

Comprehensive tests for multi-format data reading:

- **Supported Formats**
  - JSON (objects and arrays)
  - CSV (with/without headers, custom delimiters)
  - TSV (tab-separated values)
  - Text files
  - YAML (when PyYAML available)
  - Excel (.xlsx with openpyxl, .xls with xlrd)
  - XML (when lxml available)
  - Parquet (when pyarrow available)

- **Reading Features**
  - Automatic format detection from extension
  - Custom encoding support
  - Format-specific options (delimiters, sheet names, etc.)
  - Nested data structures
  - Unicode/special character handling

- **Error Handling**
  - File not found errors
  - Unsupported format errors
  - Invalid file content (malformed JSON, etc.)
  - Graceful degradation when optional dependencies missing

- **Path Handling**
  - String and Path object support
  - Relative and absolute paths
  - Case-insensitive extension matching

**Key Features Tested:**
- Format auto-detection
- Optional dependency handling
- Large file reading
- Special characters and unicode
- Empty file handling

### test_writers.py (542 lines)

Comprehensive tests for multi-format data writing:

- **Supported Formats**
  - JSON (with custom indentation)
  - CSV (with custom delimiters)
  - TSV (tab-separated)
  - Text files
  - YAML (when PyYAML available)
  - Excel (.xlsx when openpyxl available)
  - Parquet (when pyarrow available)

- **Writing Features**
  - Automatic directory creation
  - File overwriting behavior
  - Custom encoding support
  - Format-specific options
  - Pretty-printing (JSON indentation)
  - Auto-width columns (Excel)

- **Error Handling**
  - Unsupported format errors
  - Invalid data format errors (e.g., CSV requires list of dicts)
  - Directory creation failures

- **Special Cases**
  - Writing empty data
  - Large datasets
  - Special characters and unicode
  - Null values
  - Filenames with spaces

**Key Features Tested:**
- Automatic parent directory creation
- File overwriting
- Format-specific options
- Special character handling
- Large data writing

## Shared Fixtures (conftest.py)

### validation/conftest.py

Provides common fixtures for validation tests:
- `sample_valid_data`: Valid test data
- `sample_invalid_data`: Invalid test data
- `temp_schema_file`: Temporary JSON schema file

### io/conftest.py

Provides common fixtures for I/O tests:
- `sample_records`: List of record dictionaries
- `sample_json_object`: Nested JSON object
- `create_test_files`: Factory for creating test files
- `large_dataset`: 1000-record dataset for performance testing

## Running Tests

### Run all validation tests
```bash
pytest tests/unit/validation/ -v
```

### Run all I/O tests
```bash
pytest tests/unit/io/ -v
```

### Run specific test file
```bash
pytest tests/unit/validation/test_framework.py -v
```

### Run specific test class
```bash
pytest tests/unit/validation/test_framework.py::TestValidationFramework -v
```

### Run specific test
```bash
pytest tests/unit/validation/test_framework.py::TestValidationFramework::test_add_validator -v
```

### Run with coverage
```bash
pytest tests/unit/validation/ --cov=greenlang.validation --cov-report=html
pytest tests/unit/io/ --cov=greenlang.io --cov-report=html
```

### Run with markers
```bash
# Run only fast tests (if marked)
pytest tests/unit/ -m "not slow"

# Run only tests that need optional dependencies
pytest tests/unit/ -m "optional_deps"
```

## Test Statistics

| Test File | Lines | Test Classes | Key Areas |
|-----------|-------|--------------|-----------|
| test_framework.py | 561 | 3 | ValidationError, ValidationResult, ValidationFramework |
| test_schema.py | 464 | 12 | JSON Schema, format validators, graceful degradation |
| test_rules.py | 730 | 10 | All operators, rule sets, nested paths, conditions |
| test_readers.py | 497 | 10 | Multi-format reading, auto-detection, error handling |
| test_writers.py | 542 | 10 | Multi-format writing, directory creation, overwriting |
| **Total** | **2794** | **45+** | **Comprehensive coverage** |

## Coverage Goals

These tests aim to achieve:
- **Line Coverage**: >90% for all validation and I/O modules
- **Branch Coverage**: >85% for conditional logic
- **Edge Cases**: All error conditions tested
- **Integration**: Tests work with optional dependencies when available

## Dependencies

### Required
- pytest
- pytest-cov (for coverage reports)

### Optional (for full test coverage)
- jsonschema (schema validation tests)
- PyYAML (YAML I/O tests)
- openpyxl (Excel .xlsx tests)
- xlrd (Excel .xls tests)
- lxml (XML tests)
- pyarrow (Parquet tests)

## Best Practices

1. **Use fixtures**: Leverage conftest.py fixtures for common test data
2. **Test isolation**: Each test is independent and can run alone
3. **Clear names**: Test names describe what they test
4. **Edge cases**: Test empty data, null values, special characters
5. **Error paths**: Test error handling and exceptions
6. **Optional deps**: Tests skip gracefully if dependencies missing

## Contributing

When adding new tests:
1. Follow existing naming conventions
2. Add docstrings to test functions
3. Use appropriate fixtures
4. Test both success and failure cases
5. Update this guide if adding new test categories

## Related Documentation

- [Validation Framework Documentation](../../../docs/validation.md)
- [I/O Module Documentation](../../../docs/io.md)
- [Testing Strategy](../../../docs/testing.md)
