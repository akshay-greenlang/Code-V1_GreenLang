# I/O Tests

Comprehensive test suite for GreenLang I/O modules (readers and writers).

## Quick Start

```bash
# Run all I/O tests
pytest tests/unit/io/ -v

# Run with coverage
pytest tests/unit/io/ --cov=greenlang.io --cov-report=html

# Run specific test file
pytest tests/unit/io/test_readers.py -v
```

## Test Files

### test_readers.py (497 lines)
Tests for DataReader class supporting multiple file formats.

**Supported Formats:**
- ✓ JSON (.json) - objects and arrays
- ✓ CSV (.csv) - with/without headers, custom delimiters
- ✓ TSV (.tsv) - tab-separated values
- ✓ Text (.txt) - plain text
- ✓ YAML (.yaml, .yml) - when PyYAML available
- ✓ Excel (.xlsx, .xls) - when openpyxl/xlrd available
- ✓ XML (.xml) - when lxml available
- ✓ Parquet (.parquet) - when pyarrow available

**Coverage:**
- ✓ Automatic format detection
- ✓ Custom encoding support
- ✓ Format-specific options
- ✓ Nested data structures
- ✓ Unicode and special characters
- ✓ Error handling (file not found, invalid format)
- ✓ Graceful degradation when dependencies missing

**Example:**
```python
from greenlang.io.readers import DataReader

reader = DataReader()

# Automatic format detection
data = reader.read("data.json")
data = reader.read("data.csv")
data = reader.read("data.xlsx")

# With options
data = reader.read("data.csv", csv_delimiter=";", csv_has_header=True)
```

### test_writers.py (542 lines)
Tests for DataWriter class supporting multiple output formats.

**Supported Formats:**
- ✓ JSON (.json) - with custom indentation
- ✓ CSV (.csv) - with custom delimiters
- ✓ TSV (.tsv) - tab-separated
- ✓ Text (.txt) - plain text
- ✓ YAML (.yaml, .yml) - when PyYAML available
- ✓ Excel (.xlsx) - when openpyxl available
- ✓ Parquet (.parquet) - when pyarrow available

**Coverage:**
- ✓ Automatic directory creation
- ✓ File overwriting behavior
- ✓ Format-specific options
- ✓ Special character handling
- ✓ Empty data handling
- ✓ Large dataset writing
- ✓ Error handling

**Example:**
```python
from greenlang.io.writers import DataWriter

writer = DataWriter()

# Write to different formats
writer.write(data, "output.json", indent=2)
writer.write(records, "output.csv")
writer.write(records, "output.xlsx", sheet_name="Data")

# Directories created automatically
writer.write(data, "path/to/nested/output.json")
```

## Test Statistics

- **Total Lines**: 1,039
- **Test Classes**: 20+
- **Test Functions**: 80+
- **Coverage Target**: >90%

## Key Features Tested

### DataReader
- Multi-format support
- Automatic format detection
- Optional dependency handling
- Error handling and validation
- Path handling (string, Path, absolute, relative)
- Edge cases (empty files, large files, special characters)

### DataWriter
- Multi-format output
- Directory auto-creation
- File overwriting
- Format-specific options
- Data validation (e.g., CSV requires list of dicts)
- Path handling

## Fixtures (conftest.py)

Shared fixtures available to all tests:
- `sample_records`: List of record dictionaries
- `sample_json_object`: Nested JSON structure
- `create_test_files`: Factory for creating test files
- `large_dataset`: 1000-record dataset for performance testing

## Dependencies

**Required:**
- pytest
- Standard library (json, csv)

**Optional (for full format support):**
- PyYAML - for YAML reading/writing
- openpyxl - for Excel .xlsx reading/writing
- xlrd - for Excel .xls reading
- lxml - for XML reading
- pyarrow - for Parquet reading/writing

Tests automatically skip when optional dependencies are missing.

## Format-Specific Testing

### JSON
- Objects and arrays
- Nested structures
- Unicode characters
- Pretty-printing (indentation)
- Empty objects/arrays

### CSV/TSV
- With/without headers
- Custom delimiters
- Special characters (commas, quotes)
- Empty files
- Field ordering

### Excel
- Multiple sheets
- Sheet selection
- Auto-width columns
- Header rows

### YAML
- Nested structures
- Lists and dictionaries
- Flow style options

## Error Scenarios Tested

- File not found
- Unsupported formats
- Malformed data (invalid JSON, etc.)
- Missing dependencies
- Invalid data types (e.g., non-dict for CSV)
- Permission errors
- Path issues

## Performance Testing

Large dataset fixtures included for testing:
- 1000+ record datasets
- Memory efficiency
- Streaming capabilities (for future)
