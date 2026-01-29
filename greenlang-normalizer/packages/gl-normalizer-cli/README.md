# GreenLang Unit & Reference Normalizer CLI

Command-line interface for the GreenLang Unit & Reference Normalizer (GL-FOUND-X-003).

## Installation

```bash
pip install gl-normalizer-cli
```

Or install from source:

```bash
cd greenlang-normalizer/packages/gl-normalizer-cli
pip install -e .
```

## Quick Start

### Initialize Configuration

```bash
# Interactive setup
glnorm config init

# Or set API key directly
glnorm config set api_key YOUR_API_KEY
```

### Normalize a Single Value

```bash
# Basic normalization
glnorm normalize 100 kg --to metric_ton

# Output: 0.1 metric_ton

# With target unit
glnorm normalize 1500 kWh --to MJ

# Output: 5400 MJ

# JSON output
glnorm normalize 100 kg --to metric_ton --format json
```

### Batch Processing

```bash
# Process CSV file
glnorm batch data.csv --output results.json

# Process with specific columns
glnorm batch data.csv --value-col energy --unit-col energy_unit --output results.json

# Stream from stdin
cat input.json | glnorm batch - --output output.json

# Different processing modes
glnorm batch data.csv --mode fail_fast --output results.json
glnorm batch data.csv --mode partial --output results.json
glnorm batch data.csv --mode threshold --error-threshold 0.05 --output results.json
```

### Vocabulary Management

```bash
# List all vocabularies
glnorm vocab list

# Filter by type
glnorm vocab list --type fuel

# Search vocabularies
glnorm vocab search "natural gas"
glnorm vocab search diesel --type fuel --limit 5

# Show entry details
glnorm vocab show FUEL_NAT_GAS_001

# List vocabulary versions
glnorm vocab versions
```

### Configuration

```bash
# Initialize configuration file
glnorm config init

# Set configuration values
glnorm config set api_key YOUR_API_KEY
glnorm config set default_policy_mode STRICT
glnorm config set local_mode true

# Show current configuration
glnorm config show

# Get specific value
glnorm config get api_url

# Reset to defaults
glnorm config reset --yes
```

## Commands Reference

### `glnorm normalize`

Normalize a single value with its unit.

```bash
glnorm normalize <value> <unit> [OPTIONS]

Arguments:
  value    Numeric value to normalize
  unit     Unit string (e.g., kg, kWh, "kg CO2e")

Options:
  --to, -t TEXT           Target unit for conversion
  --context, -c TEXT      JSON context (expected_dimension, reference_conditions)
  --format, -f TEXT       Output format: json, yaml, table
  --api-key, -k TEXT      API key for remote service
  --local, -l             Use local engine
  --strict, -s            Enable strict mode
  --verbose, -v           Show detailed output
```

**Examples:**

```bash
# Basic conversion
glnorm normalize 100 kg --to metric_ton

# With dimension validation
glnorm normalize 1500 kWh --to MJ --context '{"expected_dimension": "energy"}'

# With reference conditions (for Nm3, scf)
glnorm normalize 250 Nm3 --context '{"reference_conditions": {"temperature_C": 0, "pressure_kPa": 101.325}}'

# JSON output
glnorm normalize 100 "kg CO2e" --format json
```

### `glnorm batch`

Process multiple records from a file.

```bash
glnorm batch <input_file> [OPTIONS]

Arguments:
  input_file    Input file (CSV, JSON, JSONL) or '-' for stdin

Options:
  --output, -o TEXT       Output file or '-' for stdout
  --mode, -m TEXT         Processing mode: fail_fast, partial, threshold
  --format, -f TEXT       Output format: json, jsonl, csv, yaml
  --to, -t TEXT           Target unit for all records
  --value-col TEXT        Column name for values (default: value)
  --unit-col TEXT         Column name for units (default: unit)
  --id-col TEXT           Column name for record IDs
  --error-threshold FLOAT Error rate threshold for threshold mode
  --batch-size, -b INT    Records per API batch
  --quiet, -q             Suppress progress output
```

**Input File Formats:**

CSV:
```csv
id,value,unit
1,100,kg
2,1500,kWh
3,250,Nm3
```

JSON:
```json
[
  {"id": "1", "value": 100, "unit": "kg"},
  {"id": "2", "value": 1500, "unit": "kWh"}
]
```

JSONL:
```
{"id": "1", "value": 100, "unit": "kg"}
{"id": "2", "value": 1500, "unit": "kWh"}
```

### `glnorm vocab`

Vocabulary management commands.

```bash
# List vocabularies
glnorm vocab list [OPTIONS]
  --type, -t TEXT         Filter: fuel, material, process, all
  --version, -V TEXT      Vocabulary version
  --format, -f TEXT       Output format
  --include-deprecated    Include deprecated entries

# Search vocabularies
glnorm vocab search <term> [OPTIONS]
  --type, -t TEXT         Filter by entity type
  --limit, -n INT         Max results (default: 10)
  --min-score FLOAT       Minimum similarity score

# Show entry details
glnorm vocab show <vocab_id> [OPTIONS]
  --version, -V TEXT      Vocabulary version

# List versions
glnorm vocab versions [OPTIONS]
```

### `glnorm config`

Configuration management.

```bash
# Initialize config
glnorm config init [OPTIONS]
  --force, -f             Overwrite existing
  --interactive/--no-interactive

# Set value
glnorm config set <key> <value>

# Get value
glnorm config get <key>

# Show all
glnorm config show [OPTIONS]
  --show-secrets          Show API key
  --format, -f TEXT       Output format

# Unset value
glnorm config unset <key>

# Show config path
glnorm config path

# Reset to defaults
glnorm config reset [--yes]
```

**Configuration Keys:**

| Key | Type | Description |
|-----|------|-------------|
| `api_url` | string | API endpoint URL |
| `api_key` | string | API authentication key |
| `default_policy_mode` | string | STRICT or LENIENT |
| `default_output_format` | string | json, yaml, table, csv |
| `vocabulary_version` | string | Version or 'latest' |
| `local_mode` | boolean | Use local engine by default |
| `verbose` | boolean | Verbose output by default |
| `cache_enabled` | boolean | Enable vocabulary caching |
| `cache_ttl_seconds` | integer | Cache TTL |
| `timeout_seconds` | integer | API timeout |

### `glnorm version`

Show version information.

```bash
glnorm version
glnorm --version
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GLNORM_API_KEY` | API key (overrides config file) |
| `GLNORM_API_URL` | API URL (overrides config file) |
| `GLNORM_CONFIG_DIR` | Custom config directory |
| `GLNORM_POLICY_MODE` | Default policy mode |
| `GLNORM_VOCAB_VERSION` | Default vocabulary version |
| `GLNORM_LOCAL_MODE` | Use local mode (true/false) |
| `GLNORM_VERBOSE` | Verbose output (true/false) |

## Configuration File

Default location: `~/.glnorm/config.yaml`

Example configuration:

```yaml
api_url: https://api.greenlang.io/normalizer/v1
api_key: your-api-key-here
default_policy_mode: LENIENT
default_output_format: table
vocabulary_version: latest
local_mode: false
verbose: false
cache_enabled: true
cache_ttl_seconds: 3600
timeout_seconds: 30
```

## Output Formats

### JSON Output

```json
{
  "success": true,
  "original_value": 100.0,
  "original_unit": "kg",
  "canonical_value": 0.1,
  "canonical_unit": "metric_ton",
  "dimension": "SI",
  "conversion_factor": 0.001,
  "provenance_hash": "abc123...",
  "processing_time_ms": 5.23,
  "warnings": [],
  "errors": []
}
```

### YAML Output

```yaml
success: true
original_value: 100.0
original_unit: kg
canonical_value: 0.1
canonical_unit: metric_ton
dimension: SI
```

### Table Output

```
┌──────────────────────────────────────┐
│       Normalization Result           │
├──────────┬───────────────────────────┤
│ Status   │ SUCCESS                   │
│ Original │ 100 kg                    │
│ Canonical│ 0.1 metric_ton            │
│ Dimension│ SI                        │
│ Factor   │ 0.001                     │
└──────────┴───────────────────────────┘
```

## Error Codes

The CLI uses GLNORM error codes for consistent error reporting:

| Code | Description |
|------|-------------|
| GLNORM-E100 | Unit parse failed |
| GLNORM-E200 | Dimension mismatch |
| GLNORM-E300 | Conversion not supported |
| GLNORM-E400 | Reference not found |
| GLNORM-E500 | Vocabulary error |
| GLNORM-E600 | Audit error |

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check src/

# Run type checking
mypy src/
```

## License

Proprietary - GreenLang Team
