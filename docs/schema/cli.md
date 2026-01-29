# GL-FOUND-X-002: CLI Command Reference

## Overview

The GreenLang Schema CLI provides command-line access to the Schema Compiler & Validator (GL-FOUND-X-002). It supports single file validation, batch processing, schema compilation, and multiple output formats.

---

## Installation

```bash
# Install via pip
pip install greenlang-sdk

# Verify installation
greenlang --version
```

---

## Command Structure

```
greenlang schema <command> [options]
greenlang validate <file> [options]  # Alias for 'greenlang schema validate'
```

### Available Commands

| Command | Description |
|---------|-------------|
| `validate` | Validate payloads against schemas |
| `compile` | Compile schemas to intermediate representation (IR) |
| `lint` | Lint schemas for best practices |
| `migrate` | Migrate schemas between versions |

---

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success - validation passed or operation completed |
| `1` | Invalid - payload failed validation |
| `2` | Error - system error, missing file, or configuration issue |

---

## greenlang schema validate

Validate a payload against a GreenLang schema.

### Synopsis

```bash
greenlang schema validate [FILE] [OPTIONS]
greenlang validate [FILE] [OPTIONS]  # Shorthand alias
```

### Arguments

| Argument | Description |
|----------|-------------|
| `FILE` | Path to payload file (YAML/JSON), or `-` for stdin |

### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--schema` | `-s` | Required | Schema reference (e.g., `emissions/activity@1.3.0`) |
| `--profile` | `-p` | `standard` | Validation profile: `strict`, `standard`, `permissive` |
| `--format` | `-f` | `pretty` | Output format: `pretty`, `text`, `table`, `json`, `sarif` |
| `--patch-level` | | `safe` | Fix suggestion level: `safe`, `needs_review`, `unsafe` |
| `--return-normalized` | `-n` | false | Include normalized payload in output |
| `--fail-on-warnings` | `-w` | false | Exit with code 1 on any warnings |
| `--max-errors` | | `100` | Maximum errors to report (0 = unlimited) |
| `--glob` | `-g` | | Glob pattern for batch validation |
| `--verbose` | `-v` | | Increase verbosity (-v, -vv) |
| `--quiet` | `-q` | false | Suppress output except exit code |

### Basic Examples

```bash
# Validate a single YAML file
greenlang schema validate emissions_data.yaml --schema gl-emissions-input@1.0.0

# Validate JSON from stdin
cat data.json | greenlang schema validate - -s emissions/activity@1.3.0

# Use the shorthand alias
greenlang validate data.yaml -s emissions/activity@1.3.0
```

### Validation Profiles

```bash
# Strict mode - all warnings become errors
greenlang schema validate data.yaml -s test@1.0.0 --profile strict

# Standard mode (default) - warnings are reported but don't fail
greenlang schema validate data.yaml -s test@1.0.0 --profile standard

# Permissive mode - only critical errors fail
greenlang schema validate data.yaml -s test@1.0.0 --profile permissive
```

### Output Formats

```bash
# Pretty format (default) - human-readable with colors
greenlang schema validate data.yaml -s test@1.0.0 --format pretty

# Table format - structured for terminal display
greenlang schema validate data.yaml -s test@1.0.0 --format table

# Text format - simple text output
greenlang schema validate data.yaml -s test@1.0.0 --format text

# JSON format - machine-readable for CI/CD
greenlang schema validate data.yaml -s test@1.0.0 --format json

# SARIF format - for GitHub Code Scanning
greenlang schema validate data.yaml -s test@1.0.0 --format sarif
```

### Batch Validation

```bash
# Validate all YAML files in a directory
greenlang schema validate --glob "data/*.yaml" -s emissions/activity@1.3.0

# Recursive glob pattern
greenlang schema validate --glob "**/*.yaml" -s test@1.0.0

# Batch validation with JSON output
greenlang schema validate -g "data/*.yaml" -s test@1.0.0 --format json
```

### CI/CD Integration

```bash
# Quiet mode with JSON output for parsing
greenlang validate data.yaml -s test@1.0.0 --format json --quiet

# Fail on warnings (stricter CI checks)
greenlang validate data.yaml -s test@1.0.0 --fail-on-warnings

# Generate SARIF for GitHub Code Scanning
greenlang schema validate src/**/*.yaml -s test@1.0.0 --format sarif > results.sarif

# Limit errors for large datasets
greenlang validate data.yaml -s test@1.0.0 --max-errors 10
```

### Verbose Output

```bash
# Show all findings (not just first 5)
greenlang schema validate data.yaml -s test@1.0.0 -v

# Debug mode with full details
greenlang schema validate data.yaml -s test@1.0.0 -vv
```

### Normalized Payload Output

```bash
# Return the normalized (coerced) payload
greenlang schema validate data.yaml -s test@1.0.0 --return-normalized

# Combine with JSON format for programmatic access
greenlang validate data.yaml -s test@1.0.0 -n --format json
```

---

## greenlang schema compile

Compile a schema to intermediate representation (IR).

### Synopsis

```bash
greenlang schema compile <SCHEMA_FILE> [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `SCHEMA_FILE` | Path to schema file (YAML/JSON) |

### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--out` | `-o` | stdout | Output file for compiled IR |
| `--format` | `-f` | `json` | Output format: `json`, `yaml` |
| `--validate` | | true | Validate schema before compilation |
| `--verbose` | `-v` | | Show compilation details |

### Examples

```bash
# Compile schema to IR JSON
greenlang schema compile schema.yaml --out ir.json

# Compile to YAML format
greenlang schema compile schema.yaml -o ir.yaml --format yaml

# Compile with validation details
greenlang schema compile schema.yaml -v
```

---

## greenlang schema lint

Lint schemas for best practices and potential issues.

### Synopsis

```bash
greenlang schema lint <SCHEMA_FILE> [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `SCHEMA_FILE` | Path to schema file or glob pattern |

### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--fix` | | false | Auto-fix safe issues |
| `--format` | `-f` | `pretty` | Output format |
| `--config` | `-c` | | Custom lint configuration file |

### Examples

```bash
# Lint a schema file
greenlang schema lint schema.yaml

# Lint with auto-fix
greenlang schema lint schema.yaml --fix

# Lint multiple schemas
greenlang schema lint "schemas/*.yaml"
```

---

## greenlang schema migrate

Migrate schemas or payloads between versions.

### Synopsis

```bash
greenlang schema migrate <FILE> [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `--from-version` | Source schema version |
| `--to-version` | Target schema version |
| `--schema` | Schema ID for the migration |
| `--dry-run` | Show changes without applying |

### Examples

```bash
# Migrate payload to new schema version
greenlang schema migrate data.yaml --schema emissions/activity --from-version 1.2.0 --to-version 1.3.0

# Dry run to preview changes
greenlang schema migrate data.yaml --schema emissions/activity --to-version 2.0.0 --dry-run
```

---

## Schema Reference Formats

The `--schema` option accepts multiple formats:

```bash
# Short form: schema_id@version
greenlang validate data.yaml -s emissions/activity@1.3.0

# GreenLang URI
greenlang validate data.yaml -s gl://schemas/emissions/activity@1.3.0

# Built-in schema ID
greenlang validate data.yaml -s gl-emissions-input@1.0.0
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GL_SCHEMA_REGISTRY_URL` | Schema registry base URL | (none) |
| `GL_SCHEMA_CACHE_DIR` | Local cache directory | `~/.greenlang/cache` |
| `GL_LOG_LEVEL` | Logging level | `WARNING` |
| `GREENLANG_API_KEY` | API key for authenticated access | (none) |

```bash
# Example: Use custom registry
export GL_SCHEMA_REGISTRY_URL=https://schemas.mycompany.com
greenlang validate data.yaml -s custom/schema@1.0.0
```

---

## Output Examples

### Pretty Format (Default)

```
Validation Result: VALID

Schema: emissions/activity@1.3.0
Hash: a1b2c3d4e5f6...

Summary:
  Errors:   0
  Warnings: 0

Processing time: 8.5ms
```

### JSON Format

```json
{
  "valid": true,
  "schema_ref": {
    "schema_id": "emissions/activity",
    "version": "1.3.0"
  },
  "schema_hash": "a1b2c3d4e5f6...",
  "summary": {
    "valid": true,
    "error_count": 0,
    "warning_count": 0
  },
  "findings": [],
  "timings_ms": {
    "total": 8.5,
    "parse": 1.2,
    "validate": 6.8,
    "normalize": 0.5
  }
}
```

### SARIF Format

```json
{
  "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
  "version": "2.1.0",
  "runs": [
    {
      "tool": {
        "driver": {
          "name": "GreenLang Schema Validator",
          "version": "1.0.0",
          "informationUri": "https://docs.greenlang.io/schema"
        }
      },
      "results": []
    }
  ]
}
```

### Error Output Example

```
Validation Result: INVALID

Schema: emissions/activity@1.3.0

Errors (3):
  1. [GLSCHEMA-E100] emissions.0.quantity
     Type mismatch: expected 'number', got 'string'
     Value: "1000"
     Fix: Convert string to number
          data["emissions"][0]["quantity"] = 1000

  2. [GLSCHEMA-E101] emissions.0.scope
     Invalid enum value: 4 is not one of [1, 2, 3]
     Fix: Use valid scope value (1, 2, or 3)

  3. [GLSCHEMA-E200] emissions.1.fuel_type
     Required field missing
     Fix: Add required field 'fuel_type'
          data["emissions"][1]["fuel_type"] = ""

Warnings (1):
  1. [GLSCHEMA-W300] organization_id
     Field 'organization_id' is recommended but missing

Processing time: 12.3ms
```

---

## Scripting Examples

### Bash Script for CI

```bash
#!/bin/bash
set -e

SCHEMA="emissions/activity@1.3.0"
DATA_DIR="./data"

echo "Validating all YAML files..."
greenlang schema validate \
  --glob "${DATA_DIR}/**/*.yaml" \
  --schema "$SCHEMA" \
  --format json \
  --quiet \
  > validation_results.json

# Check exit code
if [ $? -eq 0 ]; then
  echo "All validations passed"
else
  echo "Validation failed - see validation_results.json"
  exit 1
fi
```

### PowerShell Script

```powershell
$schema = "emissions/activity@1.3.0"
$files = Get-ChildItem -Path .\data -Filter *.yaml -Recurse

foreach ($file in $files) {
    Write-Host "Validating: $($file.Name)"
    $result = greenlang validate $file.FullName -s $schema --format json | ConvertFrom-Json

    if (-not $result.valid) {
        Write-Error "Validation failed for $($file.Name)"
        $result.findings | ForEach-Object { Write-Host "  - $($_.message)" }
    }
}
```

### Python Subprocess

```python
import subprocess
import json

def validate_file(file_path: str, schema: str) -> dict:
    """Validate a file using the CLI."""
    result = subprocess.run(
        ["greenlang", "validate", file_path, "-s", schema, "--format", "json"],
        capture_output=True,
        text=True
    )

    if result.returncode == 2:
        raise RuntimeError(f"Validation error: {result.stderr}")

    return json.loads(result.stdout)

# Usage
result = validate_file("data.yaml", "emissions/activity@1.3.0")
print(f"Valid: {result['valid']}")
```

---

## Troubleshooting

### Common Issues

**Schema not found:**
```bash
Error: Schema version not found: emissions/activity@1.3.0
```
Solution: Check schema ID and version, or verify registry URL is configured.

**File not found:**
```bash
Error: File not found: data.yaml
```
Solution: Verify file path is correct and file exists.

**Invalid YAML/JSON:**
```bash
Error: Failed to parse payload: expected ',' or '}' at line 5
```
Solution: Validate your YAML/JSON syntax before running validation.

**Rate limit exceeded:**
```bash
Error: Rate limit exceeded. Try again in 42 seconds.
```
Solution: Wait for the retry period, or use batch validation for multiple files.

---

## See Also

- [REST API Reference](api.md)
- [Python SDK Guide](sdk.md)
- [Error Codes Reference](error_codes.md)
- [Migration Guide](migration.md)
