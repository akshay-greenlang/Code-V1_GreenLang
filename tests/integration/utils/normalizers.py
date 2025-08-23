"""
Text and data normalization utilities for integration tests.
"""
import re
import json
import math
from typing import Any, Dict, List, Union
from pathlib import Path


def normalize_text(text: str) -> str:
    """
    Normalize text for snapshot comparison.
    
    Strips timestamps, paths, and normalizes whitespace.
    """
    # Remove timestamps (ISO format)
    text = re.sub(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?Z?', '<TIMESTAMP>', text)
    
    # Remove date-only patterns
    text = re.sub(r'\d{4}-\d{2}-\d{2}', '<DATE>', text)
    
    # Remove absolute Windows paths
    text = re.sub(r'[A-Z]:\\[\w\\.-]+', '<PATH>', text)
    
    # Remove absolute Unix paths
    text = re.sub(r'/[\w/.-]+', '<PATH>', text)
    
    # Normalize temp directories
    text = re.sub(r'(tmp|temp|Temp|TEMP)[/\\][\w.-]+', '<TMP>', text)
    
    # Remove UUIDs
    text = re.sub(
        r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
        '<UUID>',
        text,
        flags=re.IGNORECASE
    )
    
    # Remove port numbers
    text = re.sub(r':\d{4,5}\b', ':<PORT>', text)
    
    # Normalize multiple whitespace to single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove trailing whitespace
    text = text.strip()
    
    return text


def normalize_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize JSON data for comparison.
    
    Rounds floats, sorts keys, and normalizes paths.
    """
    def _normalize_value(value: Any) -> Any:
        if isinstance(value, float):
            # Round to 6 decimal places
            return round(value, 6)
        elif isinstance(value, str):
            # Apply text normalization
            return normalize_text(value)
        elif isinstance(value, dict):
            # Recursively normalize dict
            return {k: _normalize_value(v) for k, v in sorted(value.items())}
        elif isinstance(value, list):
            # Recursively normalize list
            return [_normalize_value(item) for item in value]
        else:
            return value
    
    return _normalize_value(data)


def round_floats(data: Union[Dict, List, float], precision: int = 6) -> Union[Dict, List, float]:
    """
    Recursively round all floats in a data structure.
    """
    if isinstance(data, float):
        return round(data, precision)
    elif isinstance(data, dict):
        return {k: round_floats(v, precision) for k, v in data.items()}
    elif isinstance(data, list):
        return [round_floats(item, precision) for item in data]
    else:
        return data


def strip_provenance(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove provenance information for comparison.
    """
    result = data.copy()
    
    # Remove common provenance fields
    provenance_fields = [
        'timestamp', 'created_at', 'updated_at', 'version',
        'source', 'dataset_version', 'last_updated'
    ]
    
    for field in provenance_fields:
        result.pop(field, None)
    
    # Recursively clean nested structures
    for key, value in result.items():
        if isinstance(value, dict):
            result[key] = strip_provenance(value)
    
    return result


def normalize_emissions(emissions: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize emissions data for comparison.
    """
    normalized = emissions.copy()
    
    # Round emission values
    if 'total_co2e_kg' in normalized:
        normalized['total_co2e_kg'] = round(normalized['total_co2e_kg'], 2)
    
    if 'total_co2e_tons' in normalized:
        normalized['total_co2e_tons'] = round(normalized['total_co2e_tons'], 3)
    
    if 'by_fuel' in normalized and isinstance(normalized['by_fuel'], dict):
        normalized['by_fuel'] = {
            fuel: round(value, 2)
            for fuel, value in normalized['by_fuel'].items()
        }
    
    if 'intensity' in normalized and isinstance(normalized['intensity'], dict):
        for key in normalized['intensity']:
            if isinstance(normalized['intensity'][key], float):
                normalized['intensity'][key] = round(normalized['intensity'][key], 4)
    
    return normalized


def assert_numerical_invariants(emissions: Dict[str, Any], tolerance: float = 1e-6):
    """
    Assert numerical invariants in emissions data.
    
    Checks:
    - Sum of by_fuel equals total
    - Percentages sum to 100%
    - All values are non-negative
    """
    errors = []
    
    # Check total equals sum of parts
    if 'by_fuel' in emissions and 'total_co2e_kg' in emissions:
        fuel_sum = sum(emissions['by_fuel'].values())
        total = emissions['total_co2e_kg']
        
        if not math.isclose(fuel_sum, total, rel_tol=tolerance):
            errors.append(
                f"Sum of by_fuel ({fuel_sum}) doesn't match total ({total})"
            )
    
    # Check kg to tons conversion
    if 'total_co2e_kg' in emissions and 'total_co2e_tons' in emissions:
        expected_tons = emissions['total_co2e_kg'] / 1000
        actual_tons = emissions['total_co2e_tons']
        
        if not math.isclose(expected_tons, actual_tons, rel_tol=tolerance):
            errors.append(
                f"Tons conversion error: {actual_tons} != {expected_tons}"
            )
    
    # Check all values are non-negative
    def check_non_negative(data, path=""):
        if isinstance(data, dict):
            for key, value in data.items():
                check_non_negative(value, f"{path}.{key}" if path else key)
        elif isinstance(data, (int, float)):
            if data < 0:
                errors.append(f"Negative value at {path}: {data}")
    
    check_non_negative(emissions)
    
    # Check percentages if present
    if 'by_fuel_percentage' in emissions:
        percentage_sum = sum(emissions['by_fuel_percentage'].values())
        if not math.isclose(percentage_sum, 100.0, abs_tol=tolerance):
            errors.append(
                f"Percentages sum to {percentage_sum}, not 100%"
            )
    
    if errors:
        raise AssertionError("Numerical invariant violations:\n" + "\n".join(errors))


def normalize_report(report_text: str) -> str:
    """
    Normalize a report for snapshot comparison.
    """
    # Apply basic text normalization
    report = normalize_text(report_text)
    
    # Normalize report-specific patterns
    # Remove report ID/version
    report = re.sub(r'Report ID: [\w-]+', 'Report ID: <ID>', report)
    report = re.sub(r'Version: [\d.]+', 'Version: <VERSION>', report)
    
    # Normalize emission values (keep structure but round)
    report = re.sub(r'\d+\.\d{7,}', lambda m: str(round(float(m.group()), 6)), report)
    
    # Normalize percentage values
    report = re.sub(r'(\d+\.\d{3,})%', lambda m: f"{round(float(m.group(1)), 2)}%", report)
    
    return report


def compare_snapshots(actual: str, expected_path: Path, update: bool = False) -> bool:
    """
    Compare actual output with expected snapshot.
    
    Args:
        actual: Actual output text
        expected_path: Path to expected snapshot file
        update: If True, update the snapshot file
    
    Returns:
        True if snapshots match
    """
    normalized_actual = normalize_report(actual)
    
    if update or not expected_path.exists():
        # Update or create snapshot
        expected_path.parent.mkdir(parents=True, exist_ok=True)
        expected_path.write_text(normalized_actual)
        return True
    
    # Compare with existing snapshot
    expected = expected_path.read_text()
    normalized_expected = normalize_report(expected)
    
    return normalized_actual == normalized_expected


def format_diff(actual: str, expected: str) -> str:
    """
    Format a diff between actual and expected text.
    """
    import difflib
    
    actual_lines = actual.splitlines(keepends=True)
    expected_lines = expected.splitlines(keepends=True)
    
    diff = difflib.unified_diff(
        expected_lines,
        actual_lines,
        fromfile='expected',
        tofile='actual',
        lineterm=''
    )
    
    return ''.join(diff)