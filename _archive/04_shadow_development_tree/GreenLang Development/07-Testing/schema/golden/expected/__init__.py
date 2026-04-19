"""Expected validation reports for golden tests.

This package contains JSON files with expected validation report structures:
    - string_valid_001_report.json: Expected report for valid string payload
    - missing_required_001_report.json: Expected report for missing required field
    - type_mismatch_001_report.json: Expected report for type mismatch error
    - range_violation_001_report.json: Expected report for range violation
    - enum_violation_001_report.json: Expected report for enum violation
    - multiple_errors_001_report.json: Expected report with multiple errors

Report Structure:
{
    "valid": boolean,
    "schema_ref": { "schema_id": string, "version": string },
    "summary": {
        "error_count": int,
        "warning_count": int,
        "info_count": int,
        "total_findings": int
    },
    "findings": [
        {
            "code": "GLSCHEMA-Exxx",
            "severity": "error|warning|info",
            "path": "/json/pointer/path",
            "message": "Human-readable message",
            "expected": { ... },
            "actual": { ... }
        }
    ]
}
"""
