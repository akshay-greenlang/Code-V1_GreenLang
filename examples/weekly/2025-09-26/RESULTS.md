# Weekly Demo Results - 2025-09-26

## Execution Summary
- **Date**: 2025-09-26 14:58:06
- **GreenLang Version**: 0.3.0
- **Pipeline**: weekly-demo-2025-09-26
- **Status**: Success
- **Run ID**: weekly-20250926-145806

## Security Capabilities Tested
- Network: Denied (default)
- Filesystem: Limited (read: ./data, write: ./output)
- Subprocess: Denied (default)
- Clock: Allowed

## Pipeline Results
- **Total Records Processed**: 6
- **Locations**: facility-a, facility-c, facility-b

## Performance Metrics
- Execution time: < 60s
- Steps completed: 3/3
- All validations passed

## Output Files
- weekly_report.json: Generated successfully
- Pipeline execution completed within SLA

## Security Validation
- [PASS] Default-deny policy enforced
- [PASS] No network access attempted
- [PASS] Filesystem access within allowlist
- [PASS] No subprocess execution

## Notes
This demo validates the default-deny security model and capability-based access control.
All operations completed within the defined security boundaries.
