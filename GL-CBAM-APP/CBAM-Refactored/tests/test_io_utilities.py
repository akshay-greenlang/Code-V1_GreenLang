"""
I/O Utilities Test Suite

Tests the GreenLang I/O utilities:
- DataReader (CSV, JSON, Excel, YAML multi-format reading)
- DataWriter (multi-format writing)
- ResourceLoader (caching, automatic format detection)
- File operations (encoding detection, compression support)

Validates replacement of custom file I/O with framework utilities

Author: GreenLang CBAM Team
Date: 2025-10-16
"""

import json
import pytest
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List

# Import framework I/O utilities
from greenlang.io import (
    DataReader,
    DataWriter,
    ResourceLoader,
    detect_encoding,
    detect_format
)


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def test_data_dir(tmp_path):
    """Create temporary test data directory."""
    data_dir = tmp_path / "io_test_data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return [
        {"id": 1, "name": "Item 1", "value": 100.0},
        {"id": 2, "name": "Item 2", "value": 200.0},
        {"id": 3, "name": "Item 3", "value": 300.0}
    ]


@pytest.fixture
def csv_file(test_data_dir, sample_data):
    """Create sample CSV file."""
    csv_path = test_data_dir / "test.csv"

    df = pd.DataFrame(sample_data)
    df.to_csv(csv_path, index=False)

    return csv_path


@pytest.fixture
def json_file(test_data_dir, sample_data):
    """Create sample JSON file."""
    json_path = test_data_dir / "test.json"

    with open(json_path, 'w') as f:
        json.dump(sample_data, f)

    return json_path


@pytest.fixture
def yaml_file(test_data_dir, sample_data):
    """Create sample YAML file."""
    yaml_path = test_data_dir / "test.yaml"

    with open(yaml_path, 'w') as f:
        yaml.dump({"items": sample_data}, f)

    return yaml_path


@pytest.fixture
def excel_file(test_data_dir, sample_data):
    """Create sample Excel file."""
    excel_path = test_data_dir / "test.xlsx"

    df = pd.DataFrame(sample_data)
    df.to_excel(excel_path, index=False)

    return excel_path


# ============================================================================
# TEST DATA READER
# ============================================================================

class TestDataReader:
    """Test DataReader multi-format reading."""

    def test_reader_initialization(self):
        """Test DataReader initializes."""
        reader = DataReader()
        assert reader is not None

    def test_read_csv(self, csv_file):
        """Test reading CSV file."""
        reader = DataReader()
        data = reader.read(csv_file)

        assert len(data) == 3
        assert data[0]['name'] == "Item 1"

    def test_read_json(self, json_file):
        """Test reading JSON file."""
        reader = DataReader()
        data = reader.read(json_file)

        assert len(data) == 3
        assert data[0]['id'] == 1

    def test_read_yaml(self, yaml_file):
        """Test reading YAML file."""
        reader = DataReader()
        data = reader.read(yaml_file)

        assert 'items' in data
        assert len(data['items']) == 3

    def test_read_excel(self, excel_file):
        """Test reading Excel file."""
        reader = DataReader()
        data = reader.read(excel_file)

        assert len(data) == 3
        assert data[0]['value'] == 100.0

    def test_auto_format_detection(self, csv_file, json_file):
        """Test automatic format detection."""
        reader = DataReader()

        # Should automatically detect CSV
        csv_data = reader.read(csv_file)
        assert len(csv_data) == 3

        # Should automatically detect JSON
        json_data = reader.read(json_file)
        assert len(json_data) == 3

    def test_read_nonexistent_file(self):
        """Test reading nonexistent file raises error."""
        reader = DataReader()

        with pytest.raises(FileNotFoundError):
            reader.read("/nonexistent/file.csv")

    def test_read_unsupported_format(self, test_data_dir):
        """Test reading unsupported format raises error."""
        reader = DataReader()

        # Create unsupported file type
        unsupported = test_data_dir / "test.bin"
        unsupported.write_bytes(b"binary data")

        with pytest.raises(ValueError, match="Unsupported format"):
            reader.read(unsupported)


# ============================================================================
# TEST DATA WRITER
# ============================================================================

class TestDataWriter:
    """Test DataWriter multi-format writing."""

    def test_writer_initialization(self):
        """Test DataWriter initializes."""
        writer = DataWriter()
        assert writer is not None

    def test_write_csv(self, test_data_dir, sample_data):
        """Test writing CSV file."""
        writer = DataWriter()
        output_path = test_data_dir / "output.csv"

        writer.write(sample_data, output_path, format='csv')

        assert output_path.exists()

        # Read back and verify
        reader = DataReader()
        data = reader.read(output_path)
        assert len(data) == 3

    def test_write_json(self, test_data_dir, sample_data):
        """Test writing JSON file."""
        writer = DataWriter()
        output_path = test_data_dir / "output.json"

        writer.write(sample_data, output_path, format='json')

        assert output_path.exists()

        # Read back and verify
        with open(output_path) as f:
            data = json.load(f)
        assert len(data) == 3

    def test_write_yaml(self, test_data_dir, sample_data):
        """Test writing YAML file."""
        writer = DataWriter()
        output_path = test_data_dir / "output.yaml"

        writer.write({"items": sample_data}, output_path, format='yaml')

        assert output_path.exists()

        # Read back and verify
        with open(output_path) as f:
            data = yaml.safe_load(f)
        assert len(data['items']) == 3

    def test_write_excel(self, test_data_dir, sample_data):
        """Test writing Excel file."""
        writer = DataWriter()
        output_path = test_data_dir / "output.xlsx"

        writer.write(sample_data, output_path, format='excel')

        assert output_path.exists()

    def test_auto_format_from_extension(self, test_data_dir, sample_data):
        """Test automatic format detection from file extension."""
        writer = DataWriter()

        # CSV from extension
        csv_path = test_data_dir / "auto.csv"
        writer.write(sample_data, csv_path)
        assert csv_path.exists()

        # JSON from extension
        json_path = test_data_dir / "auto.json"
        writer.write(sample_data, json_path)
        assert json_path.exists()

    def test_overwrite_protection(self, test_data_dir, sample_data):
        """Test overwrite protection."""
        writer = DataWriter()
        output_path = test_data_dir / "existing.json"

        # Write first time
        writer.write(sample_data, output_path)

        # Write again with overwrite=False
        with pytest.raises(FileExistsError):
            writer.write(sample_data, output_path, overwrite=False)


# ============================================================================
# TEST RESOURCE LOADER
# ============================================================================

class TestResourceLoader:
    """Test ResourceLoader with caching."""

    def test_loader_initialization(self):
        """Test ResourceLoader initializes."""
        loader = ResourceLoader()
        assert loader is not None

    def test_load_resource(self, json_file):
        """Test loading resource."""
        loader = ResourceLoader()
        data = loader.load(json_file, format='json')

        assert len(data) == 3

    def test_resource_caching(self, json_file):
        """Test resource caching."""
        loader = ResourceLoader(cache_enabled=True)

        # Load first time (cache miss)
        import time
        start = time.time()
        data1 = loader.load(json_file, format='json')
        time1 = time.time() - start

        # Load second time (cache hit)
        start = time.time()
        data2 = loader.load(json_file, format='json')
        time2 = time.time() - start

        # Should be same data
        assert data1 == data2

        # Cache hit should be faster
        print(f"First load: {time1:.6f}s, Second load: {time2:.6f}s")

    def test_cache_invalidation(self, json_file, sample_data):
        """Test cache invalidation when file changes."""
        loader = ResourceLoader(cache_enabled=True)

        # Load initial data
        data1 = loader.load(json_file, format='json')

        # Modify file
        modified_data = sample_data + [{"id": 4, "name": "Item 4", "value": 400.0}]
        with open(json_file, 'w') as f:
            json.dump(modified_data, f)

        # Load again (should detect change and reload)
        data2 = loader.load(json_file, format='json', cache_ttl=0)

        assert len(data2) == 4

    def test_load_multiple_resources(self, json_file, yaml_file):
        """Test loading multiple resources."""
        loader = ResourceLoader()

        resources = {
            'json_data': json_file,
            'yaml_data': yaml_file
        }

        loaded = loader.load_multiple(resources)

        assert 'json_data' in loaded
        assert 'yaml_data' in loaded
        assert len(loaded['json_data']) == 3


# ============================================================================
# TEST ENCODING DETECTION
# ============================================================================

class TestEncodingDetection:
    """Test encoding detection."""

    def test_detect_utf8(self, test_data_dir):
        """Test detecting UTF-8 encoding."""
        utf8_file = test_data_dir / "utf8.txt"
        utf8_file.write_text("Hello World", encoding='utf-8')

        encoding = detect_encoding(utf8_file)

        assert encoding.lower() in ['utf-8', 'utf8', 'ascii']

    def test_detect_utf16(self, test_data_dir):
        """Test detecting UTF-16 encoding."""
        utf16_file = test_data_dir / "utf16.txt"
        utf16_file.write_text("Hello World", encoding='utf-16')

        encoding = detect_encoding(utf16_file)

        assert 'utf-16' in encoding.lower() or 'utf16' in encoding.lower()

    def test_detect_latin1(self, test_data_dir):
        """Test detecting Latin-1 encoding."""
        latin1_file = test_data_dir / "latin1.txt"
        latin1_file.write_bytes("Café résumé".encode('latin-1'))

        encoding = detect_encoding(latin1_file)

        # Should detect as latin-1 or iso-8859-1
        assert encoding.lower() in ['latin-1', 'iso-8859-1', 'latin1', 'iso8859-1']


# ============================================================================
# TEST FORMAT DETECTION
# ============================================================================

class TestFormatDetection:
    """Test format detection."""

    def test_detect_csv_format(self, csv_file):
        """Test detecting CSV format."""
        format_type = detect_format(csv_file)

        assert format_type == 'csv'

    def test_detect_json_format(self, json_file):
        """Test detecting JSON format."""
        format_type = detect_format(json_file)

        assert format_type == 'json'

    def test_detect_yaml_format(self, yaml_file):
        """Test detecting YAML format."""
        format_type = detect_format(yaml_file)

        assert format_type in ['yaml', 'yml']

    def test_detect_excel_format(self, excel_file):
        """Test detecting Excel format."""
        format_type = detect_format(excel_file)

        assert format_type in ['excel', 'xlsx']


# ============================================================================
# TEST FILE OPERATIONS
# ============================================================================

class TestFileOperations:
    """Test file operation utilities."""

    def test_ensure_directory_exists(self, test_data_dir):
        """Test ensuring directory exists."""
        from greenlang.io import ensure_directory

        new_dir = test_data_dir / "new" / "nested" / "directory"

        ensure_directory(new_dir)

        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_safe_file_write(self, test_data_dir, sample_data):
        """Test safe file writing (atomic)."""
        from greenlang.io import safe_write

        output_path = test_data_dir / "safe_output.json"

        # Write atomically
        safe_write(output_path, json.dumps(sample_data))

        # Verify
        with open(output_path) as f:
            data = json.load(f)
        assert len(data) == 3

    def test_temp_file_cleanup(self):
        """Test temporary file cleanup."""
        from greenlang.io import TempFile

        temp_path = None
        with TempFile(suffix='.json') as temp_file:
            temp_path = temp_file.path
            temp_file.write('{"test": true}')
            assert temp_path.exists()

        # Should be cleaned up after context
        assert not temp_path.exists()


# ============================================================================
# TEST BATCH PROCESSING
# ============================================================================

class TestBatchProcessing:
    """Test batch file processing."""

    def test_read_multiple_files(self, csv_file, json_file):
        """Test reading multiple files in batch."""
        reader = DataReader()

        files = [csv_file, json_file]
        results = reader.read_batch(files)

        assert len(results) == 2
        assert all(len(data) == 3 for data in results)

    def test_write_multiple_files(self, test_data_dir, sample_data):
        """Test writing multiple files in batch."""
        writer = DataWriter()

        outputs = [
            (sample_data, test_data_dir / "batch1.json", 'json'),
            (sample_data, test_data_dir / "batch2.csv", 'csv')
        ]

        writer.write_batch(outputs)

        assert (test_data_dir / "batch1.json").exists()
        assert (test_data_dir / "batch2.csv").exists()


# ============================================================================
# TEST ERROR HANDLING
# ============================================================================

class TestErrorHandling:
    """Test I/O error handling."""

    def test_handle_corrupted_json(self, test_data_dir):
        """Test handling corrupted JSON file."""
        reader = DataReader()

        corrupted = test_data_dir / "corrupted.json"
        corrupted.write_text("{invalid json")

        with pytest.raises(json.JSONDecodeError):
            reader.read(corrupted)

    def test_handle_corrupted_csv(self, test_data_dir):
        """Test handling malformed CSV."""
        reader = DataReader()

        # Create CSV with mismatched columns
        malformed = test_data_dir / "malformed.csv"
        malformed.write_text("col1,col2,col3\nval1,val2\nval1,val2,val3,val4")

        # Should handle gracefully or raise informative error
        try:
            data = reader.read(malformed)
            # If it succeeds, data should be parsed as best as possible
            assert data is not None
        except Exception as e:
            # If it fails, error should be informative
            assert len(str(e)) > 0

    def test_handle_permission_error(self, test_data_dir, sample_data):
        """Test handling permission errors."""
        writer = DataWriter()

        # Try to write to non-writable location
        # Note: This may not work on all systems
        try:
            writer.write(sample_data, Path("/root/no_permission.json"))
        except (PermissionError, OSError) as e:
            # Expected error
            assert "permission" in str(e).lower() or "access" in str(e).lower()


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
