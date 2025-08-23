"""
I/O utilities for integration tests.
"""
import json
import csv
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import tempfile
import shutil


class TestIOHelper:
    """Helper class for test I/O operations."""
    
    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize with optional base directory."""
        self.base_dir = base_dir or Path(tempfile.mkdtemp(prefix="greenlang_test_"))
        self.created_files = []
    
    def cleanup(self):
        """Clean up all created files and directories."""
        if self.base_dir.exists():
            shutil.rmtree(self.base_dir)
    
    def write_json(self, filename: str, data: Dict[str, Any]) -> Path:
        """Write JSON data to file."""
        filepath = self.base_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.created_files.append(filepath)
        return filepath
    
    def read_json(self, filename: str) -> Dict[str, Any]:
        """Read JSON data from file."""
        filepath = self.base_dir / filename
        with open(filepath) as f:
            return json.load(f)
    
    def write_yaml(self, filename: str, data: Dict[str, Any]) -> Path:
        """Write YAML data to file."""
        filepath = self.base_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        
        self.created_files.append(filepath)
        return filepath
    
    def read_yaml(self, filename: str) -> Dict[str, Any]:
        """Read YAML data from file."""
        filepath = self.base_dir / filename
        with open(filepath) as f:
            return yaml.safe_load(f)
    
    def write_csv(self, filename: str, rows: List[Dict[str, Any]], 
                  headers: Optional[List[str]] = None) -> Path:
        """Write CSV data to file."""
        filepath = self.base_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if not rows:
            # Write empty CSV with headers only
            with open(filepath, 'w', newline='') as f:
                if headers:
                    writer = csv.writer(f)
                    writer.writerow(headers)
        else:
            # Determine headers from first row if not provided
            if headers is None:
                headers = list(rows[0].keys())
            
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(rows)
        
        self.created_files.append(filepath)
        return filepath
    
    def read_csv(self, filename: str) -> List[Dict[str, str]]:
        """Read CSV data from file."""
        filepath = self.base_dir / filename
        with open(filepath, newline='') as f:
            reader = csv.DictReader(f)
            return list(reader)
    
    def write_text(self, filename: str, content: str) -> Path:
        """Write text content to file."""
        filepath = self.base_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        filepath.write_text(content)
        self.created_files.append(filepath)
        return filepath
    
    def read_text(self, filename: str) -> str:
        """Read text content from file."""
        filepath = self.base_dir / filename
        return filepath.read_text()
    
    def create_temp_dir(self, prefix: str = "test_") -> Path:
        """Create a temporary directory within base_dir."""
        temp_dir = self.base_dir / f"{prefix}{tempfile.mktemp()[-8:]}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir
    
    def file_exists(self, filename: str) -> bool:
        """Check if a file exists."""
        return (self.base_dir / filename).exists()
    
    def list_files(self, pattern: str = "*") -> List[Path]:
        """List files matching pattern."""
        return list(self.base_dir.glob(pattern))


def load_fixture(category: str, name: str) -> Dict[str, Any]:
    """
    Load a fixture file from the fixtures directory.
    
    Args:
        category: Category folder (e.g., 'workflows', 'data')
        name: Fixture filename
    
    Returns:
        Loaded fixture data
    """
    fixture_path = Path(__file__).parent.parent.parent / "fixtures" / category / name
    
    if name.endswith('.json'):
        with open(fixture_path) as f:
            return json.load(f)
    elif name.endswith('.yaml') or name.endswith('.yml'):
        with open(fixture_path) as f:
            return yaml.safe_load(f)
    else:
        # Return as text for other formats
        return fixture_path.read_text()


def save_snapshot(content: str, category: str, name: str):
    """
    Save a snapshot file for golden testing.
    
    Args:
        content: Content to save
        category: Category folder (e.g., 'reports', 'cli')
        name: Snapshot filename
    """
    snapshot_path = Path(__file__).parent.parent / "snapshots" / category / name
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(content)


def load_snapshot(category: str, name: str) -> str:
    """
    Load a snapshot file for comparison.
    
    Args:
        category: Category folder (e.g., 'reports', 'cli')
        name: Snapshot filename
    
    Returns:
        Snapshot content
    """
    snapshot_path = Path(__file__).parent.parent / "snapshots" / category / name
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")
    return snapshot_path.read_text()


def validate_json_schema(data: Dict[str, Any], schema: Dict[str, Any]):
    """
    Validate JSON data against a schema.
    
    Args:
        data: Data to validate
        schema: JSON schema
    
    Raises:
        jsonschema.ValidationError: If validation fails
    """
    try:
        import jsonschema
        jsonschema.validate(data, schema)
    except ImportError:
        # If jsonschema not installed, do basic validation
        _basic_schema_validation(data, schema)


def _basic_schema_validation(data: Dict[str, Any], schema: Dict[str, Any]):
    """Basic schema validation without jsonschema library."""
    if 'required' in schema:
        for field in schema['required']:
            if field not in data:
                raise ValueError(f"Required field missing: {field}")
    
    if 'properties' in schema:
        for field, field_schema in schema['properties'].items():
            if field in data:
                value = data[field]
                expected_type = field_schema.get('type')
                
                if expected_type:
                    type_map = {
                        'string': str,
                        'number': (int, float),
                        'integer': int,
                        'boolean': bool,
                        'object': dict,
                        'array': list
                    }
                    
                    expected_python_type = type_map.get(expected_type)
                    if expected_python_type and not isinstance(value, expected_python_type):
                        raise TypeError(
                            f"Field {field} has wrong type. "
                            f"Expected {expected_type}, got {type(value).__name__}"
                        )


class OutputCapture:
    """Capture output files from a test run."""
    
    def __init__(self):
        self.files = {}
    
    def capture(self, path: Path, alias: str = None):
        """Capture a file's content."""
        if path.exists():
            alias = alias or path.name
            if path.suffix == '.json':
                with open(path) as f:
                    self.files[alias] = json.load(f)
            else:
                self.files[alias] = path.read_text()
    
    def get(self, alias: str):
        """Get captured content by alias."""
        return self.files.get(alias)
    
    def assert_exists(self, alias: str):
        """Assert that a file was captured."""
        if alias not in self.files:
            raise AssertionError(f"Expected file not captured: {alias}")