#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo Agents for Weekly Demo
============================

Provides FileConnector and DataProcessor agents for demonstrating
GreenLang's pipeline capabilities.
"""

import json
import csv
import os
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


class FileConnector:
    """
    FileConnector Agent - Handles file I/O operations for pipelines
    """

    def __init__(self):
        """Initialize FileConnector"""
        self.supported_formats = ['.csv', '.json', '.txt', '.jsonl']
        self.encoding = 'utf-8'

    def read(self, file_path: str, format: str = "auto") -> Dict[str, Any]:
        """Read data from file"""
        path = Path(file_path)

        if not path.exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}",
                "data": None
            }

        # Auto-detect format if needed
        if format == "auto":
            format = path.suffix.lower()

        try:
            if format in ['.json', 'json']:
                with open(path, 'r', encoding=self.encoding) as f:
                    data = json.load(f)
                    return {
                        "success": True,
                        "data": data,
                        "format": "json",
                        "rows": len(data) if isinstance(data, list) else 1
                    }

            elif format in ['.csv', 'csv']:
                rows = []
                with open(path, 'r', encoding=self.encoding) as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                return {
                    "success": True,
                    "data": rows,
                    "format": "csv",
                    "rows": len(rows),
                    "columns": list(rows[0].keys()) if rows else []
                }

            elif format in ['.txt', 'txt']:
                with open(path, 'r', encoding=self.encoding) as f:
                    content = f.read()
                return {
                    "success": True,
                    "data": content,
                    "format": "text",
                    "lines": len(content.splitlines())
                }

            else:
                return {
                    "success": False,
                    "error": f"Unsupported format: {format}",
                    "data": None
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to read file: {str(e)}",
                "data": None
            }

    def write(self, data: Any, file_path: str, format: str = "auto") -> Dict[str, Any]:
        """Write data to file"""
        path = Path(file_path)

        # Auto-detect format if needed
        if format == "auto":
            format = path.suffix.lower()

        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if format in ['.json', 'json']:
                with open(path, 'w', encoding=self.encoding) as f:
                    json.dump(data, f, indent=2, default=str)

            elif format in ['.csv', 'csv']:
                if isinstance(data, list) and data and isinstance(data[0], dict):
                    with open(path, 'w', encoding=self.encoding, newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=data[0].keys())
                        writer.writeheader()
                        writer.writerows(data)
                else:
                    return {
                        "success": False,
                        "error": "CSV format requires list of dictionaries"
                    }

            elif format in ['.txt', 'txt']:
                with open(path, 'w', encoding=self.encoding) as f:
                    if isinstance(data, str):
                        f.write(data)
                    else:
                        f.write(str(data))

            else:
                return {
                    "success": False,
                    "error": f"Unsupported format: {format}"
                }

            return {
                "success": True,
                "file_path": str(path),
                "format": format,
                "size_bytes": path.stat().st_size
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to write file: {str(e)}"
            }

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process method for pipeline integration"""
        action = inputs.get('action', 'read')

        if action == 'read':
            return self.read(
                file_path=inputs.get('file_path'),
                format=inputs.get('format', 'auto')
            )
        elif action == 'write':
            return self.write(
                data=inputs.get('data'),
                file_path=inputs.get('file_path'),
                format=inputs.get('format', 'auto')
            )
        else:
            return {
                "success": False,
                "error": f"Unknown action: {action}"
            }


class DataProcessor:
    """
    DataProcessor Agent - Performs data transformations and processing
    """

    def __init__(self):
        """Initialize DataProcessor"""
        self.operations = [
            'filter', 'sort', 'aggregate',
            'transform', 'validate', 'clean'
        ]

    def filter(self, data: List[Dict], conditions: Dict) -> List[Dict]:
        """Filter data based on conditions"""
        filtered = []

        for record in data:
            match = True
            for field, condition in conditions.items():
                if isinstance(condition, dict):
                    # Complex condition
                    op = condition.get('op', 'eq')
                    value = condition.get('value')

                    if op == 'eq' and record.get(field) != value:
                        match = False
                    elif op == 'gt' and not (record.get(field) > value):
                        match = False
                    elif op == 'lt' and not (record.get(field) < value):
                        match = False
                    elif op == 'contains' and value not in str(record.get(field, '')):
                        match = False
                else:
                    # Simple equality
                    if record.get(field) != condition:
                        match = False

            if match:
                filtered.append(record)

        return filtered

    def sort(self, data: List[Dict], key: str, reverse: bool = False) -> List[Dict]:
        """Sort data by key"""
        return sorted(data, key=lambda x: x.get(key, ''), reverse=reverse)

    def aggregate(self, data: List[Dict], group_by: str, operations: Dict) -> Dict:
        """Aggregate data"""
        groups = {}

        # Group data
        for record in data:
            key = record.get(group_by, 'unknown')
            if key not in groups:
                groups[key] = []
            groups[key].append(record)

        # Apply aggregations
        results = {}
        for group_key, group_data in groups.items():
            result = {"group": group_key}

            for field, op in operations.items():
                values = [r.get(field) for r in group_data if r.get(field) is not None]

                if op == 'sum':
                    result[f"{field}_sum"] = sum(values)
                elif op == 'avg':
                    result[f"{field}_avg"] = sum(values) / len(values) if values else 0
                elif op == 'count':
                    result[f"{field}_count"] = len(values)
                elif op == 'min':
                    result[f"{field}_min"] = min(values) if values else None
                elif op == 'max':
                    result[f"{field}_max"] = max(values) if values else None

            results[group_key] = result

        return results

    def transform(self, data: List[Dict], transformations: Dict) -> List[Dict]:
        """Transform data fields"""
        transformed = []

        for record in data:
            new_record = record.copy()

            for field, transform in transformations.items():
                if isinstance(transform, str):
                    # Simple field rename
                    if transform in record:
                        new_record[field] = record[transform]
                elif isinstance(transform, dict):
                    # Complex transformation
                    op = transform.get('op')

                    if op == 'uppercase':
                        source = transform.get('source', field)
                        new_record[field] = str(record.get(source, '')).upper()
                    elif op == 'lowercase':
                        source = transform.get('source', field)
                        new_record[field] = str(record.get(source, '')).lower()
                    elif op == 'multiply':
                        source = transform.get('source', field)
                        factor = transform.get('factor', 1)
                        value = record.get(source, 0)
                        new_record[field] = value * factor if isinstance(value, (int, float)) else value
                    elif op == 'concat':
                        sources = transform.get('sources', [])
                        separator = transform.get('separator', ' ')
                        values = [str(record.get(s, '')) for s in sources]
                        new_record[field] = separator.join(values)

            transformed.append(new_record)

        return transformed

    def validate(self, data: List[Dict], rules: Dict) -> Dict[str, Any]:
        """Validate data against rules"""
        errors = []
        valid_count = 0

        for i, record in enumerate(data):
            record_errors = []

            for field, rule in rules.items():
                value = record.get(field)

                if isinstance(rule, dict):
                    # Check required
                    if rule.get('required') and value is None:
                        record_errors.append(f"Field '{field}' is required")

                    # Check type
                    expected_type = rule.get('type')
                    if expected_type and value is not None:
                        if expected_type == 'number' and not isinstance(value, (int, float)):
                            record_errors.append(f"Field '{field}' must be a number")
                        elif expected_type == 'string' and not isinstance(value, str):
                            record_errors.append(f"Field '{field}' must be a string")

                    # Check min/max
                    if 'min' in rule and isinstance(value, (int, float)) and value < rule['min']:
                        record_errors.append(f"Field '{field}' is below minimum {rule['min']}")
                    if 'max' in rule and isinstance(value, (int, float)) and value > rule['max']:
                        record_errors.append(f"Field '{field}' exceeds maximum {rule['max']}")

            if record_errors:
                errors.append({
                    "record_index": i,
                    "errors": record_errors
                })
            else:
                valid_count += 1

        return {
            "valid": len(errors) == 0,
            "total_records": len(data),
            "valid_records": valid_count,
            "invalid_records": len(errors),
            "errors": errors[:10]  # Limit error details
        }

    def clean(self, data: List[Dict], options: Dict) -> List[Dict]:
        """Clean data"""
        cleaned = []

        for record in data:
            # Remove nulls if specified
            if options.get('remove_nulls'):
                record = {k: v for k, v in record.items() if v is not None}

            # Trim strings
            if options.get('trim_strings'):
                record = {
                    k: v.strip() if isinstance(v, str) else v
                    for k, v in record.items()
                }

            # Remove empty strings
            if options.get('remove_empty'):
                record = {k: v for k, v in record.items() if v != ''}

            # Normalize case
            if options.get('normalize_case'):
                case_fields = options.get('case_fields', [])
                for field in case_fields:
                    if field in record and isinstance(record[field], str):
                        record[field] = record[field].lower()

            cleaned.append(record)

        return cleaned

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process method for pipeline integration"""
        operation = inputs.get('operation', 'transform')
        data = inputs.get('data', [])

        try:
            if operation == 'filter':
                result = self.filter(data, inputs.get('conditions', {}))
                return {
                    "success": True,
                    "data": result,
                    "count": len(result)
                }

            elif operation == 'sort':
                result = self.sort(
                    data,
                    inputs.get('key', 'id'),
                    inputs.get('reverse', False)
                )
                return {
                    "success": True,
                    "data": result,
                    "count": len(result)
                }

            elif operation == 'aggregate':
                result = self.aggregate(
                    data,
                    inputs.get('group_by', 'category'),
                    inputs.get('operations', {'value': 'sum'})
                )
                return {
                    "success": True,
                    "data": result,
                    "groups": len(result)
                }

            elif operation == 'transform':
                result = self.transform(
                    data,
                    inputs.get('transformations', {})
                )
                return {
                    "success": True,
                    "data": result,
                    "count": len(result)
                }

            elif operation == 'validate':
                result = self.validate(
                    data,
                    inputs.get('rules', {})
                )
                return {
                    "success": True,
                    "validation": result,
                    "data": data if result['valid'] else []
                }

            elif operation == 'clean':
                result = self.clean(
                    data,
                    inputs.get('options', {'remove_nulls': True})
                )
                return {
                    "success": True,
                    "data": result,
                    "count": len(result)
                }

            else:
                return {
                    "success": False,
                    "error": f"Unknown operation: {operation}"
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"Processing failed: {str(e)}"
            }


# Register agents for GreenLang discovery
AGENTS = {
    'FileConnector': FileConnector,
    'DataProcessor': DataProcessor
}


def get_agent(name: str):
    """Get agent class by name"""
    return AGENTS.get(name)


if __name__ == "__main__":
    # Demo usage
    print("Demo Agents for GreenLang")
    print("=" * 50)

    # Test FileConnector
    print("\nTesting FileConnector...")
    connector = FileConnector()

    # Create test data
    test_data = [
        {"id": 1, "name": "Alice", "score": 95},
        {"id": 2, "name": "Bob", "score": 87},
        {"id": 3, "name": "Charlie", "score": 92}
    ]

    # Write test data
    result = connector.write(test_data, "test_data.json")
    print(f"Write result: {result}")

    # Read test data
    result = connector.read("test_data.json")
    print(f"Read result: {result}")

    # Test DataProcessor
    print("\nTesting DataProcessor...")
    processor = DataProcessor()

    # Filter data
    filtered = processor.filter(test_data, {"score": {"op": "gt", "value": 90}})
    print(f"Filtered (score > 90): {filtered}")

    # Sort data
    sorted_data = processor.sort(test_data, "score", reverse=True)
    print(f"Sorted by score: {sorted_data}")

    # Clean up
    Path("test_data.json").unlink(missing_ok=True)

    print("\n[OK] Demo agents ready for use!")