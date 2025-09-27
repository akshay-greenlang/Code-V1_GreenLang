# GreenLang CLI Commands Reference

## Overview

The GreenLang CLI (`gl`) provides commands for managing pipelines, packs, and security policies.

## Global Options

These options work with all commands:

```bash
gl [global-options] <command> [command-options]

Global Options:
  --version          Show version information
  --help, -h         Show help message
  --verbose, -v      Enable verbose output
  --debug            Enable debug logging
  --config FILE      Use custom config file
  --no-color         Disable colored output
  --format FORMAT    Output format (json|yaml|table)
```

## Commands

### `gl init` - Initialize Project

Create a new GreenLang project or pack.

```bash
# Initialize a basic project
gl init my-project

# Initialize with template
gl init --template ml-pipeline my-ml-project

# Initialize a pack
gl init pack my-pack

Options:
  --template TYPE    Use project template (basic|ml|etl|api)
  --force           Overwrite existing files
  --minimal         Create minimal structure
```

### `gl run` - Execute Pipelines

Run GreenLang pipelines with security enforcement.

```bash
# Run a pipeline
gl run pipeline.yaml

# Run with parameters
gl run pipeline.yaml --param env=prod --param debug=true

# Run with specific runner
gl run pipeline.yaml --runner docker

# Run with custom output directory
gl run pipeline.yaml --output ./results

Options:
  --runner TYPE      Execution runner (local|docker|k8s)
  --param KEY=VAL    Set pipeline parameter
  --output DIR       Output directory
  --inputs FILE      Input parameters file (JSON/YAML)
  --dry-run         Validate without executing
  --watch           Watch for file changes
  --timeout SEC     Execution timeout in seconds
  --no-cache        Disable caching
```

#### Subcommands

```bash
# List recent runs
gl run list
gl run list --limit 10 --status failed

# Get run information
gl run info <run-id>
gl run info <run-id> --format json

# View run logs
gl run logs <run-id>
gl run logs <run-id> --follow --step process-data

# Export run artifacts
gl run export <run-id> --output ./export
```

### `gl pack` - Manage Packs

Install, create, and manage GreenLang packs.

```bash
# List installed packs
gl pack list
gl pack list --remote  # List available packs

# Search packs
gl pack search "machine learning"
gl pack search --tag analytics

# Install a pack
gl pack install greenlang/weather-forecast
gl pack install org/pack@1.2.3
gl pack install ./local-pack.tar.gz

# Install with verification
gl pack install org/pack --verify-signature

Options:
  --verify-signature    Require signature verification
  --allow-unsigned      Allow unsigned packs (dev only)
  --force              Overwrite existing installation
  --no-deps            Don't install dependencies
```

#### Pack Development

```bash
# Create a new pack
gl pack create my-pack

# Validate pack manifest
gl pack validate ./my-pack
gl pack validate pack.yaml

# Build pack archive
gl pack build ./my-pack
gl pack build --output dist/

# Sign a pack
gl pack sign ./my-pack
gl pack sign my-pack.tar.gz --key ~/.keys/signing.key

# Publish to registry
gl pack publish ./my-pack
gl pack publish --registry https://hub.greenlang.ai
```

#### Pack Information

```bash
# Show pack details
gl pack info greenlang/ml-pipeline
gl pack info --manifest  # Show full manifest

# Show pack dependencies
gl pack deps greenlang/data-processor
gl pack deps --tree  # Tree view

# Verify pack integrity
gl pack verify org/pack
gl pack verify pack.tar.gz --checksum sha256:abc123...
```

### `gl verify` - Verification Tools

Verify signatures, checksums, and integrity.

```bash
# Verify file signature
gl verify file.tar.gz
gl verify file.tar.gz --sig file.sig

# Verify with specific key
gl verify file.tar.gz --key public.key

# Verify SBOM
gl verify sbom.spdx.json
gl verify sbom.spdx.json --policy sbom-policy.rego

# Verify container image
gl verify image:tag
gl verify ghcr.io/greenlang/greenlang:latest

Options:
  --sig FILE         Signature file
  --key FILE         Public key for verification
  --checksum HASH    Expected checksum
  --policy FILE      OPA policy for validation
  --verbose         Show verification details
```

### `gl policy` - Policy Management

Manage and evaluate OPA policies.

```bash
# Check policy compliance
gl policy check ./my-pack
gl policy check pipeline.yaml

# Evaluate policy
gl policy eval policy.rego --data input.json
gl policy eval --bundle policies/ --input request.json

# Test policies
gl policy test policies/
gl policy test --coverage  # With coverage report

# Format policy files
gl policy fmt policies/*.rego
gl policy fmt --write  # Update files in place

Options:
  --bundle DIR       Policy bundle directory
  --data FILE        Data file for evaluation
  --input FILE       Input file for evaluation
  --format FORMAT    Output format (json|pretty|raw)
```

### `gl doctor` - System Diagnostics

Check system health and configuration.

```bash
# Run all diagnostics
gl doctor

# Check specific component
gl doctor --check dependencies
gl doctor --check network
gl doctor --check security

# Fix common issues
gl doctor --fix

Output includes:
  - GreenLang version
  - Python version
  - Installed dependencies
  - Configuration status
  - Security settings
  - Network connectivity
  - Registry access
  - Cache status
```

### `gl config` - Configuration Management

View and manage GreenLang configuration.

```bash
# Show current configuration
gl config show
gl config show --format yaml

# Get specific value
gl config get registry.url
gl config get security.default_deny

# Set configuration value
gl config set registry.url https://hub.greenlang.ai
gl config set telemetry.enabled false

# Reset to defaults
gl config reset
gl config reset registry  # Reset section

Options:
  --global          Use global config (~/.greenlang/config)
  --local           Use local config (./.greenlang/config)
  --format FORMAT   Output format (json|yaml|env)
```

### `gl cache` - Cache Management

Manage GreenLang cache for packs and artifacts.

```bash
# Show cache information
gl cache info
gl cache info --detailed

# Clear cache
gl cache clear
gl cache clear --packs     # Clear only packs
gl cache clear --artifacts  # Clear only artifacts

# Prune old entries
gl cache prune
gl cache prune --days 30  # Remove entries older than 30 days

# Verify cache integrity
gl cache verify
```

### `gl version` - Version Information

Show detailed version information.

```bash
# Show version
gl version

# Show with components
gl version --verbose

# Check for updates
gl version --check-updates

Output:
  GreenLang CLI version 0.3.0
  Python: 3.9.10
  Platform: linux-x86_64
  Config: ~/.greenlang/config.yaml
  Registry: https://hub.greenlang.ai
```

## Environment Variables

Control CLI behavior with environment variables:

```bash
# Development mode (relaxes security)
export GREENLANG_DEV_MODE=true

# Custom config location
export GREENLANG_CONFIG=/path/to/config.yaml

# Log level (DEBUG|INFO|WARNING|ERROR)
export GREENLANG_LOG_LEVEL=DEBUG

# Disable telemetry
export GREENLANG_TELEMETRY_DISABLED=true

# Custom cache directory
export GREENLANG_CACHE_DIR=/tmp/gl-cache

# Registry URL
export GREENLANG_REGISTRY=https://custom.registry.com

# Parallel execution
export GREENLANG_PARALLEL=4
```

## Configuration Files

### Global Configuration

Location: `~/.greenlang/config.yaml`

```yaml
# Registry settings
registry:
  url: https://hub.greenlang.ai
  verify_signatures: true
  timeout: 30

# Security settings
security:
  default_deny: true
  require_signatures: true
  allowed_registries:
    - hub.greenlang.ai
    - ghcr.io

# Runtime settings
runtime:
  default_runner: local
  parallel_steps: 4
  timeout: 3600

# Telemetry (opt-in)
telemetry:
  enabled: false
  endpoint: https://telemetry.greenlang.ai
```

### Project Configuration

Location: `./.greenlang/config.yaml`

```yaml
# Project-specific settings
project:
  name: my-project
  version: 1.0.0

# Default parameters
defaults:
  params:
    environment: development
    debug: true

# Pack sources
packs:
  sources:
    - https://hub.greenlang.ai
    - ./local-packs
```

## Output Formats

Most commands support multiple output formats:

```bash
# Table format (default)
gl pack list

# JSON format
gl pack list --format json

# YAML format
gl run info <run-id> --format yaml

# Plain text
gl version --format plain

# CSV (where applicable)
gl run list --format csv > runs.csv
```

## Exit Codes

GreenLang CLI uses standard exit codes:

- `0` - Success
- `1` - General error
- `2` - Invalid arguments
- `3` - Configuration error
- `4` - Security policy violation
- `5` - Network error
- `6` - Dependency error
- `7` - Verification failed
- `130` - Interrupted (Ctrl+C)

## Examples

### Complete Pipeline Workflow

```bash
# Initialize project
gl init my-analysis

# Install required packs
gl pack install greenlang/data-tools --verify-signature
gl pack install greenlang/ml-tools --verify-signature

# Validate pipeline
gl run pipeline.yaml --dry-run

# Run pipeline with parameters
gl run pipeline.yaml \
  --param dataset=production \
  --param model=rf \
  --runner docker \
  --output ./results

# Check results
gl run list --limit 1
gl run logs $(gl run list --limit 1 --format json | jq -r '.[0].id')
```

### Security Workflow

```bash
# Verify pack before installation
gl verify greenlang/sensitive-pack.tar.gz

# Check policy compliance
gl policy check pipeline.yaml

# Install with strict security
gl pack install org/pack \
  --verify-signature \
  --policy security.rego

# Run with security audit
gl run pipeline.yaml \
  --audit \
  --log-file security.log
```

## Getting Help

```bash
# General help
gl --help
gl help

# Command help
gl run --help
gl help pack install

# Debug issues
gl doctor
gl doctor --verbose

# Report issues
# https://github.com/greenlang/greenlang/issues
```