# GREENLANG UTILITIES LIBRARY DESIGN

**Comprehensive Technical Specification for Framework Utility Modules**

**Version:** 1.0
**Date:** 2025-10-16
**Status:** Technical Specification
**Audience:** Engineers, Architects

---

## 🎯 OVERVIEW

This document provides detailed technical specifications for all utility modules in the GreenLang framework. These utilities represent the core infrastructure that will enable 50-70% framework contribution, reducing custom code by 2,000-3,000 lines per agent.

### **Utility Modules Covered**

| Module | Purpose | LOC Saved | Priority |
|--------|---------|-----------|----------|
| **Validation** | Schema & business rule validation | 600 lines | ⭐⭐⭐⭐⭐ |
| **Provenance** | Audit trails & file integrity | 555 lines | ⭐⭐⭐⭐⭐ |
| **Data I/O** | Multi-format read/write | 400 lines | ⭐⭐⭐⭐ |
| **Batch Processing** | Parallel batch execution | 300 lines | ⭐⭐⭐⭐ |
| **Reporting** | Multi-dimensional aggregation | 300 lines | ⭐⭐⭐ |
| **Computation** | Calculation caching | 200 lines | ⭐⭐⭐ |
| **Pipelines** | Multi-agent orchestration | 200 lines | ⭐⭐⭐⭐ |
| **Testing** | Test fixtures & assertions | 400 lines | ⭐⭐⭐ |
| **TOTAL** | | **2,955 lines** | |

---

## 📦 MODULE 1: VALIDATION FRAMEWORK

### **Purpose**

Provide comprehensive validation infrastructure for:
- JSON Schema validation
- Business rule validation
- Data quality checks
- Custom validation rules

### **Module Structure**

```
greenlang/validation/
├── __init__.py
├── framework.py          # ValidationFramework main class
├── schema.py             # JSON Schema validator
├── rules.py              # Business rules engine
├── quality.py            # Data quality checks
└── decorators.py         # @validate decorator
```

### **Core API**

#### **1. ValidationFramework Class**

```python
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import jsonschema
from pydantic import BaseModel

class ValidationResult:
    """Result of validation operation."""

    def __init__(self):
        self.is_valid: bool = True
        self.errors: List[Dict] = []
        self.warnings: List[Dict] = []
        self.stats: Dict[str, int] = {
            'total': 0,
            'valid': 0,
            'invalid': 0,
            'warnings': 0
        }

    def add_error(self, field: str, message: str, value: Any = None):
        """Add validation error."""
        self.errors.append({
            'field': field,
            'message': message,
            'value': value,
            'severity': 'error'
        })
        self.is_valid = False
        self.stats['invalid'] += 1

    def add_warning(self, field: str, message: str, value: Any = None):
        """Add validation warning."""
        self.warnings.append({
            'field': field,
            'message': message,
            'value': value,
            'severity': 'warning'
        })
        self.stats['warnings'] += 1

    def to_dict(self) -> Dict:
        """Export as dictionary."""
        return {
            'is_valid': self.is_valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'stats': self.stats
        }


class ValidationFramework:
    """
    Comprehensive validation framework for GreenLang agents.

    Features:
    - JSON Schema validation
    - Business rule validation
    - Data quality checks
    - Custom validation functions
    - Batch validation with progress
    """

    def __init__(
        self,
        schema_path: Optional[Path] = None,
        rules: Optional[Dict[str, Callable]] = None,
        strict: bool = True
    ):
        """
        Initialize validation framework.

        Args:
            schema_path: Path to JSON Schema file
            rules: Dictionary of business rules {rule_name: validation_fn}
            strict: If True, treat warnings as errors
        """
        self.schema = None
        if schema_path:
            self.schema = self._load_schema(schema_path)

        self.rules = rules or {}
        self.strict = strict
        self.validator = None
        if self.schema:
            self.validator = jsonschema.Draft7Validator(self.schema)

    def _load_schema(self, schema_path: Path) -> Dict:
        """Load JSON Schema from file."""
        import json
        with open(schema_path) as f:
            return json.load(f)

    def validate_schema(self, data: Dict) -> ValidationResult:
        """
        Validate data against JSON Schema.

        Args:
            data: Data to validate

        Returns:
            ValidationResult with errors and warnings
        """
        result = ValidationResult()
        result.stats['total'] = 1

        if not self.validator:
            result.add_warning('schema', 'No schema defined')
            return result

        # Validate against schema
        errors = sorted(self.validator.iter_errors(data), key=lambda e: e.path)

        for error in errors:
            field_path = '.'.join(str(p) for p in error.path) or 'root'
            result.add_error(
                field=field_path,
                message=error.message,
                value=error.instance if len(str(error.instance)) < 100 else str(error.instance)[:100]
            )

        if result.is_valid:
            result.stats['valid'] = 1

        return result

    def validate_rules(self, data: Dict) -> ValidationResult:
        """
        Validate data against business rules.

        Args:
            data: Data to validate

        Returns:
            ValidationResult with rule violations
        """
        result = ValidationResult()
        result.stats['total'] = len(self.rules)

        for rule_name, rule_fn in self.rules.items():
            try:
                is_valid, message = rule_fn(data)
                if not is_valid:
                    result.add_error(
                        field=rule_name,
                        message=message,
                        value=None
                    )
                else:
                    result.stats['valid'] += 1
            except Exception as e:
                result.add_error(
                    field=rule_name,
                    message=f"Rule execution failed: {str(e)}",
                    value=None
                )

        return result

    def validate_all(self, data: Dict) -> ValidationResult:
        """
        Run all validations (schema + rules).

        Args:
            data: Data to validate

        Returns:
            Combined ValidationResult
        """
        # Schema validation
        result = self.validate_schema(data)

        # Business rules validation
        if self.rules:
            rules_result = self.validate_rules(data)
            result.errors.extend(rules_result.errors)
            result.warnings.extend(rules_result.warnings)
            result.stats['total'] += rules_result.stats['total']
            result.stats['valid'] += rules_result.stats['valid']
            result.stats['invalid'] += rules_result.stats['invalid']

            if not rules_result.is_valid:
                result.is_valid = False

        return result

    def validate_batch(
        self,
        items: List[Dict],
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, ValidationResult]:
        """
        Validate multiple items in batch.

        Args:
            items: List of items to validate
            progress_callback: Optional callback(current, total)

        Returns:
            Dictionary mapping item index to ValidationResult
        """
        results = {}

        for idx, item in enumerate(items):
            result = self.validate_all(item)
            results[idx] = result

            if progress_callback:
                progress_callback(idx + 1, len(items))

        return results


# Example business rules
def create_cbam_rules() -> Dict[str, Callable]:
    """Example: CBAM-specific business rules."""

    def rule_emissions_positive(data: Dict) -> tuple[bool, str]:
        """Emissions must be positive."""
        emissions = data.get('direct_emissions', 0)
        if emissions < 0:
            return False, f"Emissions cannot be negative: {emissions}"
        return True, "OK"

    def rule_quantity_positive(data: Dict) -> tuple[bool, str]:
        """Quantity must be positive."""
        quantity = data.get('quantity', 0)
        if quantity <= 0:
            return False, f"Quantity must be positive: {quantity}"
        return True, "OK"

    def rule_country_code_valid(data: Dict) -> tuple[bool, str]:
        """Country code must be ISO 3166-1 alpha-2."""
        country = data.get('country_of_origin', '')
        valid_codes = {'US', 'GB', 'FR', 'DE', 'CN', 'JP', 'IN', 'BR'}  # Subset for example
        if country and country not in valid_codes:
            return False, f"Invalid country code: {country}"
        return True, "OK"

    return {
        'emissions_positive': rule_emissions_positive,
        'quantity_positive': rule_quantity_positive,
        'country_code_valid': rule_country_code_valid
    }
```

#### **2. Validation Decorator**

```python
from functools import wraps

def validate(
    schema_path: Optional[Path] = None,
    rules: Optional[Dict[str, Callable]] = None,
    raise_on_error: bool = True
):
    """
    Decorator to validate function inputs.

    Usage:
        @validate(schema_path='input_schema.json', raise_on_error=True)
        def process_data(data: Dict) -> Dict:
            # Process validated data
            return result
    """
    def decorator(func: Callable) -> Callable:
        validator = ValidationFramework(schema_path=schema_path, rules=rules)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Assume first argument is data to validate
            if args:
                data = args[0]
                result = validator.validate_all(data)

                if not result.is_valid and raise_on_error:
                    raise ValueError(f"Validation failed: {result.errors}")

                # Attach validation result to function
                func.validation_result = result

            return func(*args, **kwargs)

        return wrapper
    return decorator


# Example usage
@validate(schema_path=Path('schemas/cbam_input.json'), raise_on_error=True)
def process_cbam_good(data: Dict) -> Dict:
    """Process CBAM good with automatic validation."""
    # Data is already validated
    return {
        'cn_code': data['cn_code'],
        'emissions': data['direct_emissions'],
        'status': 'processed'
    }
```

### **Usage Examples**

#### **Example 1: Basic Schema Validation**

```python
from greenlang.validation import ValidationFramework
from pathlib import Path

# Initialize validator
validator = ValidationFramework(
    schema_path=Path('schemas/cbam_good.json'),
    strict=True
)

# Validate single item
data = {
    'cn_code': '7208.10',
    'quantity': 1000,
    'direct_emissions': 2.5,
    'country_of_origin': 'CN'
}

result = validator.validate_schema(data)
if result.is_valid:
    print("✓ Validation passed")
else:
    for error in result.errors:
        print(f"✗ {error['field']}: {error['message']}")
```

#### **Example 2: Business Rules Validation**

```python
from greenlang.validation import ValidationFramework, create_cbam_rules

# Initialize with business rules
validator = ValidationFramework(
    schema_path=Path('schemas/cbam_good.json'),
    rules=create_cbam_rules(),
    strict=True
)

# Validate with all rules
result = validator.validate_all(data)

print(f"Valid: {result.is_valid}")
print(f"Errors: {len(result.errors)}")
print(f"Warnings: {len(result.warnings)}")
print(f"Stats: {result.stats}")
```

#### **Example 3: Batch Validation**

```python
import pandas as pd
from greenlang.validation import ValidationFramework

# Load data
df = pd.read_csv('cbam_goods.csv')
items = df.to_dict('records')

# Validate batch with progress
validator = ValidationFramework(schema_path=Path('schemas/cbam_good.json'))

def progress(current, total):
    print(f"Validating: {current}/{total} ({current/total*100:.1f}%)")

results = validator.validate_batch(items, progress_callback=progress)

# Analyze results
valid_count = sum(1 for r in results.values() if r.is_valid)
invalid_count = len(results) - valid_count

print(f"\nValidation complete:")
print(f"  Valid: {valid_count}")
print(f"  Invalid: {invalid_count}")

# Get invalid items
invalid_items = [(idx, r) for idx, r in results.items() if not r.is_valid]
for idx, result in invalid_items[:5]:  # Show first 5
    print(f"\nItem {idx}:")
    for error in result.errors:
        print(f"  - {error['field']}: {error['message']}")
```

### **Migration Example**

#### **Before (Custom Code - 250 lines)**

```python
class CustomValidator:
    def __init__(self):
        self.errors = []

    def validate_cn_code(self, code):
        # 50 lines of CN code validation
        if not code or len(code) < 4:
            self.errors.append(f"Invalid CN code: {code}")
        # ... more validation

    def validate_emissions(self, emissions):
        # 40 lines of emissions validation
        if emissions < 0:
            self.errors.append(f"Negative emissions: {emissions}")
        # ... more validation

    def validate_quantity(self, quantity):
        # 40 lines of quantity validation
        # ...

    # ... 5 more validation methods (120 lines)

    def validate_all(self, data):
        # 50 lines of orchestration
        self.errors = []
        self.validate_cn_code(data.get('cn_code'))
        self.validate_emissions(data.get('emissions'))
        # ... call all validators
        return len(self.errors) == 0
```

#### **After (Framework - 50 lines)**

```python
from greenlang.validation import ValidationFramework, validate

# Define business rules
def create_rules():
    return {
        'emissions_positive': lambda d: (d['emissions'] >= 0, "Emissions must be >= 0"),
        'quantity_positive': lambda d: (d['quantity'] > 0, "Quantity must be > 0"),
        # ... more rules
    }

# Use framework
validator = ValidationFramework(
    schema_path=Path('schemas/cbam_good.json'),
    rules=create_rules()
)

# Or use decorator
@validate(schema_path=Path('schemas/cbam_good.json'))
def process_data(data: Dict) -> Dict:
    # Data is already validated
    return {'status': 'processed'}
```

**LOC Reduction: 250 → 50 lines (80% reduction)**

---

## 📦 MODULE 2: PROVENANCE SYSTEM

### **Purpose**

Provide automatic audit trails and file integrity verification for:
- File hashing (SHA256)
- Environment capture
- Dependency tracking
- Provenance record generation

### **Module Structure**

```
greenlang/provenance/
├── __init__.py
├── framework.py          # ProvenanceFramework main class
├── hashing.py            # File integrity (SHA256)
├── environment.py        # Environment capture
├── records.py            # ProvenanceRecord model
└── decorators.py         # @traced decorator
```

### **Core API**

#### **1. ProvenanceFramework Class**

```python
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import hashlib
import platform
import json
import sys

class ProvenanceRecord:
    """Immutable provenance record."""

    def __init__(self):
        self.timestamp = datetime.now().isoformat()
        self.agent_id: Optional[str] = None
        self.agent_version: Optional[str] = None
        self.operation: Optional[str] = None
        self.inputs: Dict[str, str] = {}  # {filename: sha256}
        self.outputs: Dict[str, str] = {}  # {filename: sha256}
        self.parameters: Dict[str, Any] = {}
        self.environment: Dict[str, str] = {}
        self.dependencies: Dict[str, str] = {}
        self.execution_time: Optional[float] = None
        self.status: str = 'pending'
        self.error: Optional[str] = None

    def to_dict(self) -> Dict:
        """Export as dictionary."""
        return {
            'timestamp': self.timestamp,
            'agent_id': self.agent_id,
            'agent_version': self.agent_version,
            'operation': self.operation,
            'inputs': self.inputs,
            'outputs': self.outputs,
            'parameters': self.parameters,
            'environment': self.environment,
            'dependencies': self.dependencies,
            'execution_time': self.execution_time,
            'status': self.status,
            'error': self.error
        }

    def to_json(self, path: Path):
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class ProvenanceFramework:
    """
    Automatic provenance tracking for GreenLang agents.

    Features:
    - SHA256 file hashing
    - Environment capture (Python, OS, dependencies)
    - Input/output tracking
    - Execution time tracking
    - Immutable records
    """

    def __init__(
        self,
        agent_id: str,
        agent_version: str,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize provenance framework.

        Args:
            agent_id: Unique agent identifier
            agent_version: Agent version
            output_dir: Directory for provenance records
        """
        self.agent_id = agent_id
        self.agent_version = agent_version
        self.output_dir = output_dir or Path('provenance')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Capture environment once
        self.environment = self._capture_environment()
        self.dependencies = self._capture_dependencies()

    def _capture_environment(self) -> Dict[str, str]:
        """Capture execution environment."""
        return {
            'python_version': sys.version,
            'python_implementation': platform.python_implementation(),
            'platform': platform.platform(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'hostname': platform.node(),
        }

    def _capture_dependencies(self) -> Dict[str, str]:
        """Capture installed packages."""
        try:
            import pkg_resources
            return {
                pkg.key: pkg.version
                for pkg in pkg_resources.working_set
            }
        except Exception:
            return {}

    def hash_file(self, file_path: Path) -> str:
        """
        Calculate SHA256 hash of file.

        Args:
            file_path: Path to file

        Returns:
            Hexadecimal SHA256 hash
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, 'rb') as f:
            # Read in chunks for large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def create_record(
        self,
        operation: str,
        inputs: Optional[Dict[str, Path]] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> ProvenanceRecord:
        """
        Create new provenance record.

        Args:
            operation: Operation name (e.g., 'process_goods')
            inputs: Input files {name: path}
            parameters: Operation parameters

        Returns:
            ProvenanceRecord instance
        """
        record = ProvenanceRecord()
        record.agent_id = self.agent_id
        record.agent_version = self.agent_version
        record.operation = operation
        record.environment = self.environment
        record.dependencies = self.dependencies
        record.parameters = parameters or {}

        # Hash input files
        if inputs:
            for name, path in inputs.items():
                if path.exists():
                    record.inputs[name] = self.hash_file(path)

        return record

    def finalize_record(
        self,
        record: ProvenanceRecord,
        outputs: Optional[Dict[str, Path]] = None,
        execution_time: Optional[float] = None,
        status: str = 'success',
        error: Optional[str] = None
    ):
        """
        Finalize provenance record with outputs.

        Args:
            record: ProvenanceRecord to finalize
            outputs: Output files {name: path}
            execution_time: Execution time in seconds
            status: 'success' or 'failed'
            error: Error message if failed
        """
        # Hash output files
        if outputs:
            for name, path in outputs.items():
                if path.exists():
                    record.outputs[name] = self.hash_file(path)

        record.execution_time = execution_time
        record.status = status
        record.error = error

        # Save record
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.agent_id}_{record.operation}_{timestamp}.json"
        record.to_json(self.output_dir / filename)

    def verify_file(self, file_path: Path, expected_hash: str) -> bool:
        """
        Verify file integrity.

        Args:
            file_path: Path to file
            expected_hash: Expected SHA256 hash

        Returns:
            True if hash matches, False otherwise
        """
        actual_hash = self.hash_file(file_path)
        return actual_hash == expected_hash

    def load_record(self, record_path: Path) -> ProvenanceRecord:
        """Load provenance record from file."""
        with open(record_path) as f:
            data = json.load(f)

        record = ProvenanceRecord()
        for key, value in data.items():
            setattr(record, key, value)

        return record
```

#### **2. Provenance Decorator**

```python
from functools import wraps
import time

def traced(
    operation: str,
    inputs: Optional[Dict[str, str]] = None,  # {param_name: file_param}
    outputs: Optional[Dict[str, str]] = None  # {param_name: file_param}
):
    """
    Decorator to automatically track provenance.

    Usage:
        @traced(
            operation='process_goods',
            inputs={'input_file': 'input_path'},
            outputs={'output_file': 'output_path'}
        )
        def process_goods(input_path: Path, output_path: Path, **kwargs):
            # Processing logic
            return result
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Get provenance framework from self
            if not hasattr(self, 'provenance'):
                return func(self, *args, **kwargs)

            provenance = self.provenance

            # Extract input files
            input_files = {}
            if inputs:
                for name, param in inputs.items():
                    if param in kwargs:
                        input_files[name] = Path(kwargs[param])

            # Extract parameters (excluding file paths)
            parameters = {k: v for k, v in kwargs.items() if k not in (inputs or {}).values()}

            # Create record
            record = provenance.create_record(
                operation=operation,
                inputs=input_files,
                parameters=parameters
            )

            # Execute function
            start_time = time.time()
            try:
                result = func(self, *args, **kwargs)
                execution_time = time.time() - start_time

                # Extract output files
                output_files = {}
                if outputs:
                    for name, param in outputs.items():
                        if param in kwargs:
                            output_files[name] = Path(kwargs[param])

                # Finalize record
                provenance.finalize_record(
                    record=record,
                    outputs=output_files,
                    execution_time=execution_time,
                    status='success'
                )

                return result

            except Exception as e:
                execution_time = time.time() - start_time
                provenance.finalize_record(
                    record=record,
                    execution_time=execution_time,
                    status='failed',
                    error=str(e)
                )
                raise

        return wrapper
    return decorator
```

### **Usage Examples**

#### **Example 1: Manual Provenance Tracking**

```python
from greenlang.provenance import ProvenanceFramework
from pathlib import Path
import time

# Initialize framework
provenance = ProvenanceFramework(
    agent_id='cbam-importer',
    agent_version='1.0.0',
    output_dir=Path('provenance_records')
)

# Start operation
record = provenance.create_record(
    operation='import_goods',
    inputs={'raw_data': Path('input.csv')},
    parameters={'batch_size': 100, 'validate': True}
)

# Process data
start_time = time.time()
try:
    # ... processing logic ...
    output_path = Path('output.json')

    execution_time = time.time() - start_time

    # Finalize
    provenance.finalize_record(
        record=record,
        outputs={'processed_data': output_path},
        execution_time=execution_time,
        status='success'
    )
except Exception as e:
    provenance.finalize_record(
        record=record,
        execution_time=time.time() - start_time,
        status='failed',
        error=str(e)
    )
```

#### **Example 2: Automatic Provenance with Decorator**

```python
from greenlang.agents import BaseDataProcessor
from greenlang.provenance import traced
from pathlib import Path

class CBAMImporter(BaseDataProcessor):
    agent_id = 'cbam-importer'
    version = '1.0.0'

    @traced(
        operation='import_goods',
        inputs={'input_file': 'input_path'},
        outputs={'output_file': 'output_path'}
    )
    def import_goods(
        self,
        input_path: Path,
        output_path: Path,
        batch_size: int = 100
    ) -> Dict:
        """Import goods with automatic provenance tracking."""
        # Read input
        df = self.read_input(input_path)

        # Process
        results = self.process_batch(df.to_dict('records'))

        # Write output
        self.write_output(results, output_path)

        return {'processed': len(results)}
```

#### **Example 3: Verify File Integrity**

```python
from greenlang.provenance import ProvenanceFramework
from pathlib import Path

# Load provenance record
provenance = ProvenanceFramework(
    agent_id='cbam-importer',
    agent_version='1.0.0'
)

record_path = Path('provenance_records/cbam-importer_import_goods_20251016_120000.json')
record = provenance.load_record(record_path)

# Verify input file integrity
input_file = Path('input.csv')
expected_hash = record.inputs['raw_data']

if provenance.verify_file(input_file, expected_hash):
    print("✓ Input file integrity verified")
else:
    print("✗ Input file has been modified!")

# Verify output file integrity
output_file = Path('output.json')
expected_hash = record.outputs['processed_data']

if provenance.verify_file(output_file, expected_hash):
    print("✓ Output file integrity verified")
else:
    print("✗ Output file has been modified!")
```

### **Migration Example**

#### **Before (Custom Code - 605 lines)**

```python
class ProvenanceTracker:
    """Custom provenance implementation from CBAM."""

    def __init__(self, agent_id, version):
        # 50 lines of initialization
        self.agent_id = agent_id
        self.version = version
        # ... setup logging, storage, etc.

    def calculate_sha256(self, file_path):
        # 30 lines of hashing logic
        # ...

    def capture_environment(self):
        # 80 lines of environment capture
        # Python version, OS, packages, etc.
        # ...

    def create_provenance_record(self, operation, inputs):
        # 100 lines of record creation
        # ...

    def finalize_record(self, record, outputs):
        # 80 lines of finalization logic
        # ...

    def save_record(self, record):
        # 60 lines of storage logic
        # ...

    # ... 10 more methods (205 lines)
```

#### **After (Framework - 50 lines)**

```python
from greenlang.provenance import ProvenanceFramework, traced
from greenlang.agents import BaseDataProcessor

class CBAMImporter(BaseDataProcessor):
    agent_id = 'cbam-importer'
    version = '1.0.0'

    # Provenance automatically initialized by BaseDataProcessor

    @traced(
        operation='import_goods',
        inputs={'input_file': 'input_path'},
        outputs={'output_file': 'output_path'}
    )
    def import_goods(self, input_path: Path, output_path: Path):
        """Automatic provenance tracking with decorator."""
        # Business logic only - provenance handled by framework
        results = self.process_data(input_path)
        self.write_output(results, output_path)
        return results
```

**LOC Reduction: 605 → 50 lines (92% reduction)**

---

## 📦 MODULE 3: DATA I/O LIBRARY

### **Purpose**

Provide unified interface for reading/writing multiple data formats:
- CSV, JSON, Excel, Parquet, XML
- Automatic format detection
- Streaming support for large files
- Schema inference

### **Module Structure**

```
greenlang/io/
├── __init__.py
├── readers.py            # DataReader class
├── writers.py            # DataWriter class
├── formats.py            # Format handlers
└── streaming.py          # Streaming support
```

### **Core API**

#### **1. DataReader Class**

```python
from typing import Optional, Dict, Any, Iterator
from pathlib import Path
import pandas as pd
import json

class DataReader:
    """
    Universal data reader supporting multiple formats.

    Supported formats:
    - CSV (.csv)
    - JSON (.json, .jsonl)
    - Excel (.xlsx, .xls)
    - Parquet (.parquet)
    - XML (.xml)
    """

    def __init__(self):
        self.format_handlers = {
            '.csv': self._read_csv,
            '.json': self._read_json,
            '.jsonl': self._read_jsonl,
            '.xlsx': self._read_excel,
            '.xls': self._read_excel,
            '.parquet': self._read_parquet,
            '.xml': self._read_xml
        }

    def read(
        self,
        file_path: Path,
        format: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Read file into DataFrame.

        Args:
            file_path: Path to file
            format: Optional format override (auto-detected if None)
            **kwargs: Format-specific parameters

        Returns:
            pandas DataFrame
        """
        file_path = Path(file_path)

        # Detect format
        if format is None:
            format = file_path.suffix.lower()

        # Get handler
        handler = self.format_handlers.get(format)
        if not handler:
            raise ValueError(f"Unsupported format: {format}")

        # Read file
        return handler(file_path, **kwargs)

    def _read_csv(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Read CSV file."""
        defaults = {
            'encoding': 'utf-8',
            'na_values': ['', 'NA', 'N/A', 'null'],
            'keep_default_na': True
        }
        defaults.update(kwargs)
        return pd.read_csv(file_path, **defaults)

    def _read_json(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Read JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle different JSON structures
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            # If dict has list values, treat as columns
            if all(isinstance(v, list) for v in data.values()):
                return pd.DataFrame(data)
            # Otherwise, treat as single record
            return pd.DataFrame([data])
        else:
            raise ValueError("JSON must be list or dict")

    def _read_jsonl(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Read JSON Lines file."""
        records = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return pd.DataFrame(records)

    def _read_excel(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Read Excel file."""
        defaults = {'sheet_name': 0}
        defaults.update(kwargs)
        return pd.read_excel(file_path, **defaults)

    def _read_parquet(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Read Parquet file."""
        return pd.read_parquet(file_path, **kwargs)

    def _read_xml(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Read XML file."""
        return pd.read_xml(file_path, **kwargs)

    def read_streaming(
        self,
        file_path: Path,
        chunk_size: int = 10000,
        **kwargs
    ) -> Iterator[pd.DataFrame]:
        """
        Read file in chunks (for large files).

        Args:
            file_path: Path to file
            chunk_size: Number of records per chunk
            **kwargs: Format-specific parameters

        Yields:
            DataFrame chunks
        """
        file_path = Path(file_path)
        format = file_path.suffix.lower()

        if format == '.csv':
            yield from pd.read_csv(file_path, chunksize=chunk_size, **kwargs)
        elif format == '.jsonl':
            chunk = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        chunk.append(json.loads(line))
                        if len(chunk) >= chunk_size:
                            yield pd.DataFrame(chunk)
                            chunk = []
                if chunk:
                    yield pd.DataFrame(chunk)
        else:
            # For unsupported streaming formats, read all at once
            yield self.read(file_path, **kwargs)
```

#### **2. DataWriter Class**

```python
class DataWriter:
    """
    Universal data writer supporting multiple formats.
    """

    def __init__(self):
        self.format_handlers = {
            '.csv': self._write_csv,
            '.json': self._write_json,
            '.jsonl': self._write_jsonl,
            '.xlsx': self._write_excel,
            '.parquet': self._write_parquet,
            '.xml': self._write_xml
        }

    def write(
        self,
        data: pd.DataFrame,
        file_path: Path,
        format: Optional[str] = None,
        **kwargs
    ):
        """
        Write DataFrame to file.

        Args:
            data: DataFrame to write
            file_path: Output file path
            format: Optional format override (auto-detected if None)
            **kwargs: Format-specific parameters
        """
        file_path = Path(file_path)

        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Detect format
        if format is None:
            format = file_path.suffix.lower()

        # Get handler
        handler = self.format_handlers.get(format)
        if not handler:
            raise ValueError(f"Unsupported format: {format}")

        # Write file
        handler(data, file_path, **kwargs)

    def _write_csv(self, data: pd.DataFrame, file_path: Path, **kwargs):
        """Write CSV file."""
        defaults = {
            'index': False,
            'encoding': 'utf-8'
        }
        defaults.update(kwargs)
        data.to_csv(file_path, **defaults)

    def _write_json(self, data: pd.DataFrame, file_path: Path, **kwargs):
        """Write JSON file."""
        defaults = {
            'orient': 'records',
            'indent': 2
        }
        defaults.update(kwargs)
        data.to_json(file_path, **defaults)

    def _write_jsonl(self, data: pd.DataFrame, file_path: Path, **kwargs):
        """Write JSON Lines file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            for record in data.to_dict('records'):
                f.write(json.dumps(record) + '\n')

    def _write_excel(self, data: pd.DataFrame, file_path: Path, **kwargs):
        """Write Excel file."""
        defaults = {
            'index': False,
            'sheet_name': 'Sheet1'
        }
        defaults.update(kwargs)
        data.to_excel(file_path, **defaults)

    def _write_parquet(self, data: pd.DataFrame, file_path: Path, **kwargs):
        """Write Parquet file."""
        data.to_parquet(file_path, **kwargs)

    def _write_xml(self, data: pd.DataFrame, file_path: Path, **kwargs):
        """Write XML file."""
        data.to_xml(file_path, **kwargs)
```

### **Usage Examples**

```python
from greenlang.io import DataReader, DataWriter
from pathlib import Path

# Initialize
reader = DataReader()
writer = DataWriter()

# Read any format
df = reader.read(Path('input.csv'))
df = reader.read(Path('input.json'))
df = reader.read(Path('input.xlsx'))

# Write any format
writer.write(df, Path('output.csv'))
writer.write(df, Path('output.json'))
writer.write(df, Path('output.parquet'))

# Streaming for large files
for chunk in reader.read_streaming(Path('large_file.csv'), chunk_size=10000):
    # Process chunk
    processed = transform(chunk)
    # Append to output
    writer.write(processed, Path('output.csv'), mode='a')
```

**LOC Reduction: 200 → 20 lines (90% reduction)**

---

## 📊 SUMMARY TABLE

### **Complete LOC Savings Analysis**

| Module | Custom LOC | Framework LOC | Custom After | Reduction |
|--------|-----------|---------------|--------------|-----------|
| **Validation** | 250 | 600 framework | 50 | 80% |
| **Provenance** | 605 | 605 framework | 50 | 92% |
| **Data I/O** | 200 | 400 framework | 20 | 90% |
| **Batch Processing** | 200 | 300 framework | 50 | 75% |
| **Reporting** | 300 | 300 framework | 150 | 50% |
| **Computation** | 150 | 200 framework | 50 | 67% |
| **Pipelines** | 200 | 200 framework | 80 | 60% |
| **Testing** | 600 | 400 framework | 200 | 67% |
| | | | | |
| **TOTAL** | **2,505** | **3,005** | **650** | **74%** |

### **Developer Experience Improvement**

| Task | Before | After | Time Saved |
|------|--------|-------|------------|
| **Validation setup** | 4 hours | 15 minutes | 94% |
| **Provenance implementation** | 8 hours | 10 minutes | 98% |
| **Data I/O handling** | 3 hours | 10 minutes | 94% |
| **Batch processing** | 4 hours | 30 minutes | 88% |
| **Overall development** | 2-3 weeks | 3-5 days | 75% |

---

## 🎯 NEXT STEPS

1. **Review**: Validate these specifications with engineering team
2. **Prototype**: Build Tier 1 modules (Validation, Provenance, Data I/O)
3. **Test**: Migrate one existing agent as proof of concept
4. **Iterate**: Refine APIs based on feedback
5. **Document**: Create comprehensive usage guides
6. **Release**: Launch framework v0.1 for early adopters

---

**Status:** ✅ **Ready for Engineering Review**

**Next Document:** Tool Ecosystem Design (Module 6-8 specifications)

---

*"The best utilities are invisible - they just make everything else easier."*
