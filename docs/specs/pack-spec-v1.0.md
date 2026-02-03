# GreenLang Pack Specification v1.0

**Version:** 1.0.0
**Status:** Draft
**Last Updated:** 2026-02-03

## Overview

A GreenLang Pack is a self-contained, versioned unit of climate calculation logic that can be installed, validated, signed, and published. Packs are the fundamental distribution unit for GreenLang's "language" of climate compliance workflows.

**Core Definition:**
> A Pack is a deterministic, auditable bundle containing agents, pipelines, datasets, and policies for climate calculations.

## Pack Structure

```
my-pack/
├── pack.yaml           # Pack manifest (REQUIRED)
├── CARD.md             # Human-readable documentation (REQUIRED)
├── gl.yaml             # Main pipeline definition (OPTIONAL)
├── agents/             # Agent implementations
│   └── *.py
├── pipelines/          # Additional pipeline definitions
│   └── *.yaml
├── datasets/           # Embedded data files
│   └── *.json|*.csv
├── policies/           # OPA/Rego policy files
│   └── *.rego
├── tests/              # Test files
│   └── test_*.py
├── sbom.json           # Software Bill of Materials (generated)
└── signatures/         # Digital signatures (generated)
```

## Pack Manifest (pack.yaml)

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Pack identifier (lowercase, alphanumeric, hyphens) |
| `version` | string | Semantic version (MAJOR.MINOR.PATCH) |
| `kind` | string | Must be `"pack"` |
| `pack_schema_version` | string | Schema version, must be `"1.0"` |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `description` | string | Brief description of the pack |
| `author` | object | Author information |
| `license` | string | SPDX license identifier |
| `contents` | object | Pack contents specification |
| `dependencies` | array | Required dependencies |
| `capabilities` | object | Required runtime capabilities |
| `policy` | object | Policy constraints |
| `security` | object | Security requirements |
| `metadata` | object | Additional metadata |
| `compat` | object | Compatibility requirements |
| `tests` | array | Test file patterns |
| `card` | string | Path to CARD.md |

### Complete Schema

```yaml
# pack.yaml - Full specification

# === REQUIRED FIELDS ===
name: my-pack-name                    # [a-z0-9-]+, max 64 chars
version: 1.0.0                        # semver format
kind: pack                            # literal "pack"
pack_schema_version: '1.0'            # schema version

# === IDENTIFICATION ===
description: |
  Brief description of what this pack does.

author:
  name: Author Name                   # required if author present
  email: author@example.com           # optional
  organization: Organization Name     # optional

license: Apache-2.0                   # SPDX identifier

# === CONTENTS ===
contents:
  agents:
    - name: AgentName                 # display name
      class_path: agents.module:Class # Python import path
      description: What the agent does
      inputs:                         # input schema
        field_name: type              # string|number|boolean|array|object
      outputs:                        # output schema
        field_name: type

  pipelines:
    - name: pipeline-name             # identifier
      file: pipelines/file.yaml       # path to gl.yaml
      description: What the pipeline does

  datasets:
    - name: dataset-name              # identifier
      path: data/file.json            # path to data file
      format: json                    # json|csv|parquet
      card: cards/dataset.md          # optional data card
      size: 45KB                      # optional size hint

  reports: []                         # report templates

# === DEPENDENCIES ===
dependencies:
  - name: greenlang-sdk
    version: '>=0.1.0'
  - pandas>=1.3.0                     # pip-style dependency
  - numpy>=1.20.0

# === CAPABILITIES ===
# Capabilities define what runtime permissions the pack requires
capabilities:
  fs:                                 # filesystem access
    allow: true|false
    read:
      allowlist:
        - ${INPUT_DIR}/**
        - ${PACK_DATA_DIR}/**
      denylist: []
    write:
      allowlist:
        - ${RUN_TMP}/**
      denylist:
        - /**/*.py                    # prevent code modification

  net:                                # network access
    allow: true|false
    outbound:
      allowlist:
        - https://api.example.com/*
      denylist: []

  clock:                              # system clock access
    allow: true|false                 # false for deterministic packs

  subprocess:                         # subprocess spawning
    allow: true|false
    allowlist: []                     # allowed commands
    denylist: []

# === POLICY CONSTRAINTS ===
policy:
  install: policies/install.rego      # install-time policy
  runtime: policies/runtime.rego      # runtime policy

  network:                            # allowed network hosts
    - api.weather.gov
    - solcast.com

  data_residency:                     # allowed data regions
    - US
    - EU
    - APAC

  ef_vintage_min: 2024                # minimum emission factor year

  license_allowlist:                  # allowed dependency licenses
    - Apache-2.0
    - MIT
    - BSD-3-Clause

# === SECURITY ===
security:
  sbom: sbom.json                     # path to SBOM
  signatures: []                      # signature files

# === PROVENANCE ===
provenance:
  sbom: true                          # generate SBOM on publish
  signing: true                       # require signing on publish

# === COMPATIBILITY ===
compat:
  greenlang: '>=0.1.0'                # GreenLang version
  python: '>=3.10'                    # Python version

# === METADATA ===
metadata:
  authors:
    - name: Author Name
      email: author@example.com
  homepage: https://example.com
  repository: https://github.com/org/repo
  publisher: publisher-name
  tags:
    - emissions
    - carbon
    - compliance

# === TESTING ===
tests:
  - tests/*.py
  - tests/**/*.py

test_command: pytest tests/

# === DOCUMENTATION ===
card: CARD.md

# === VERSION CONSTRAINTS ===
min_greenlang_version: 0.1.0
```

## CARD.md Format

The CARD.md (Climate Asset Repository Document) provides human-readable documentation.

### Required Sections

```markdown
# Pack Name

Brief one-line description.

## Description

Detailed description of what this pack does, its purpose,
and intended use cases.

## Usage

### Installation

```bash
gl pack add pack-name
```

### Basic Usage

```bash
gl run pack-name/pipeline-name --input data.json
```

## Inputs

| Name | Type | Required | Description |
|------|------|----------|-------------|
| field | type | yes/no | description |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| field | type | description |

## Examples

### Example 1: Basic Usage

```yaml
# Input
field: value

# Output
result: value
```

## Limitations

- Known limitation 1
- Known limitation 2

## References

- [Reference 1](url)
- [Standard Name](url)

## Changelog

### 1.0.0 (2026-02-03)
- Initial release
```

## Pack Versioning

### Semantic Versioning

Packs MUST use semantic versioning (SemVer):

- **MAJOR** (X.0.0): Breaking changes to inputs/outputs
- **MINOR** (0.X.0): New features, backward compatible
- **PATCH** (0.0.X): Bug fixes, no API changes

### Version Constraints

```yaml
dependencies:
  - name: other-pack
    version: '>=1.0.0,<2.0.0'   # Range constraint
  - name: another-pack
    version: '~=1.4.0'          # Compatible release
  - name: exact-pack
    version: '==1.2.3'          # Exact version
```

### Compatibility Guarantees

| Change Type | Version Bump | Example |
|------------|--------------|---------|
| Remove output field | MAJOR | Remove `co2e_kg` |
| Change output type | MAJOR | `number` -> `string` |
| Add required input | MAJOR | New required field |
| Add optional input | MINOR | New optional field |
| Add output field | MINOR | New output |
| Fix calculation bug | PATCH | Correct formula |

## Pack Validation

### Validation Command

```bash
gl pack validate [path]
```

### Validation Rules

1. **Manifest Validation**
   - All required fields present
   - Valid semver version
   - Valid SPDX license
   - Valid capability declarations

2. **Content Validation**
   - All referenced files exist
   - Agent class paths are importable
   - Pipeline YAML is valid
   - Dataset files are readable

3. **Policy Validation**
   - Rego policies compile without errors
   - Network allowlist is valid
   - License allowlist matches dependencies

4. **Security Validation**
   - No hardcoded secrets
   - No unsafe imports (eval, exec, etc.)
   - Dependencies have acceptable licenses

### Validation Output

```json
{
  "valid": true,
  "pack": "my-pack",
  "version": "1.0.0",
  "warnings": [],
  "errors": [],
  "checks": {
    "manifest": "pass",
    "contents": "pass",
    "policy": "pass",
    "security": "pass"
  }
}
```

## Pack Signing

### Signing Process

1. Generate SBOM (Software Bill of Materials)
2. Compute content hash
3. Sign with pack author's key
4. Attach signature to pack

### Signature Format

```json
{
  "pack": "my-pack",
  "version": "1.0.0",
  "content_hash": "sha256:abc123...",
  "sbom_hash": "sha256:def456...",
  "signer": {
    "name": "Author Name",
    "email": "author@example.com",
    "key_id": "KEY123"
  },
  "timestamp": "2026-02-03T12:00:00Z",
  "signature": "base64-encoded-signature"
}
```

### Verification Command

```bash
gl verify my-pack

# Output:
# Pack: my-pack@1.0.0
# Signer: Author Name <author@example.com>
# Signed: 2026-02-03T12:00:00Z
# SBOM: 47 dependencies, 0 vulnerabilities
# Provenance: Verified
# Status: VALID
```

## SBOM Requirements

### Format

Packs MUST generate CycloneDX or SPDX format SBOMs.

### Required Fields

- All Python dependencies with versions
- Pack metadata (name, version, author)
- License information
- Vulnerability scan results

### Example SBOM (CycloneDX)

```json
{
  "bomFormat": "CycloneDX",
  "specVersion": "1.5",
  "version": 1,
  "metadata": {
    "component": {
      "name": "my-pack",
      "version": "1.0.0",
      "type": "application"
    }
  },
  "components": [
    {
      "name": "pandas",
      "version": "2.0.0",
      "type": "library",
      "licenses": [{"id": "BSD-3-Clause"}]
    }
  ]
}
```

## Pack Lifecycle

### 1. Create

```bash
gl init my-pack
# Creates pack.yaml and CARD.md scaffolding
```

### 2. Develop

```bash
# Add agents, pipelines, datasets
# Run locally
gl run ./my-pack/gl.yaml --input data.json
```

### 3. Validate

```bash
gl pack validate ./my-pack
```

### 4. Test

```bash
cd my-pack && pytest tests/
```

### 5. Publish

```bash
gl pack publish ./my-pack
# Runs: validate -> test -> policy -> sbom -> sign -> push
```

### 6. Install

```bash
gl pack add my-pack@1.0.0
```

### 7. Use

```bash
gl run my-pack/pipeline-name --input data.json
```

### 8. Verify

```bash
gl verify my-pack
```

## Security Model

### Capability-Based Security

Packs declare required capabilities; runtime enforces limits.

### Principle of Least Privilege

- Default: No network, no subprocess, read-only filesystem
- Explicit: Must declare each capability needed
- Auditable: All capability grants logged

### Trust Levels

| Level | Description | Requirements |
|-------|-------------|--------------|
| Unsigned | No verification | None |
| Signed | Author verified | Valid signature |
| Verified | Author + org verified | Signature + publisher verification |
| Certified | Official GreenLang | GreenLang team signature |

## Appendix A: JSON Schema

The complete JSON Schema for pack.yaml is available at:
`greenlang/specs/schemas/pack-v1.0.schema.json`

## Appendix B: Reserved Names

The following pack names are reserved:
- `greenlang-*` (official packs)
- `gl-*` (system packs)
- `core`, `base`, `foundation`

## Appendix C: Variable Substitution

The following variables are available in capability paths:

| Variable | Description |
|----------|-------------|
| `${INPUT_DIR}` | Directory containing input files |
| `${OUTPUT_DIR}` | Directory for output files |
| `${PACK_DATA_DIR}` | Pack's embedded data directory |
| `${RUN_TMP}` | Temporary directory for this run |
| `${HOME}` | User's home directory |
