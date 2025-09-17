# GreenLang Pack Manifest Specification

## Overview

The pack manifest (`pack.yaml` or `manifest.yaml`) is the central configuration file that defines a GreenLang pack's metadata, contents, dependencies, and security capabilities. All packs must include a valid manifest conforming to this specification.

## Manifest Structure

```yaml
# Required fields
name: my-pack-name           # DNS-safe pack name
version: 1.0.0               # Semantic version (MAJOR.MINOR.PATCH)
kind: pack                   # Type: pack, dataset, or connector
license: MIT                 # SPDX license identifier
contents:                    # Pack contents and artifacts
  pipelines:
    - pipeline1.yaml
    - pipeline2.yaml
  agents: []                 # Optional agent list
  datasets: []               # Optional dataset files
  reports: []                # Optional report templates

# Optional fields
compat:                      # Compatibility constraints
  greenlang: ">=0.9.0"       # GreenLang version requirement
  python: ">=3.8"            # Python version requirement

dependencies:                # External dependencies
  - numpy>=1.20.0
  - pandas
  - emissions-core/base@1.0.0

capabilities:                # Security capabilities (deny-by-default)
  net:
    allow: false
    outbound:
      allowlist: []
  fs:
    allow: false
    read:
      allowlist: []
    write:
      allowlist: []
  clock:
    allow: false
  subprocess:
    allow: false
    allowlist: []

card: CARD.md                # Model/Pack card documentation
policy:                      # Policy constraints
  network: []
  ef_vintage_min: 2024

security:                    # Security metadata
  sbom: sbom.spdx.json      # Software Bill of Materials
  signatures:
    - pack.sig
  vulnerabilities:
    max_critical: 0
    max_high: 5

metadata:                    # Additional discovery metadata
  description: "Pack description"
  author: "Author Name"
  tags:
    - emissions
    - calculator
```

## Field Reference

### Required Fields

#### `name` (string, required)
DNS-safe pack identifier.

**Rules:**
- Must be lowercase
- 3-64 characters
- Start with a letter
- End with alphanumeric
- May contain hyphens
- Pattern: `^[a-z][a-z0-9-]{1,62}[a-z0-9]$`

**Examples:**
```yaml
name: emissions-calculator
name: fuel-optimizer-v2
name: grid-carbon-factors
```

#### `version` (string, required)
Semantic version following MAJOR.MINOR.PATCH format.

**Rules:**
- Must follow semantic versioning
- Pattern: `^\d+\.\d+\.\d+$`

**Examples:**
```yaml
version: 1.0.0
version: 2.15.3
version: 0.1.0
```

#### `kind` (string, required)
Type of GreenLang package.

**Values:**
- `pack`: Standard pipeline pack
- `dataset`: Data-only pack
- `connector`: Integration connector

#### `license` (string, required)
SPDX license identifier.

**Common licenses:**
- `MIT`
- `Apache-2.0`
- `GPL-3.0`
- `BSD-3-Clause`
- `Commercial`
- `Proprietary`

#### `contents` (object, required)
Defines pack contents and artifacts.

**Fields:**
- `pipelines` (array, required): List of pipeline YAML files
- `agents` (array): Agent names provided by pack
- `datasets` (array): Dataset file paths
- `reports` (array): Report template paths

**Example:**
```yaml
contents:
  pipelines:
    - emissions/calculate.yaml
    - emissions/report.yaml
  agents:
    - FuelAgent
    - GridFactorAgent
  datasets:
    - data/emission_factors.json
    - data/grid_intensities.csv
  reports:
    - templates/annual_report.md
```

### Optional Fields

#### `compat` (object)
Compatibility constraints for runtime environments.

**Fields:**
- `greenlang`: Version constraint (e.g., `">=0.9.0"`)
- `python`: Python version constraint (e.g., `">=3.8,<3.12"`)

#### `dependencies` (array)
External dependencies required by the pack.

**Formats:**
- Python package: `numpy>=1.20.0`
- Pack dependency: `org/pack-name@version`
- Git repository: `git+https://github.com/org/repo.git`

#### `capabilities` (object)
Security capabilities for pack execution. All capabilities default to deny when not specified.

##### Network Capability (`net`)
Controls network access.

**Fields:**
- `allow` (boolean): Enable network access
- `outbound.allowlist` (array): Allowed domain patterns

**Example:**
```yaml
capabilities:
  net:
    allow: true
    outbound:
      allowlist:
        - https://api.company.com/*
        - https://*.climatenza.com/*
```

##### Filesystem Capability (`fs`)
Controls filesystem access.

**Fields:**
- `allow` (boolean): Enable filesystem access
- `read.allowlist` (array): Paths allowed for reading
- `write.allowlist` (array): Paths allowed for writing

**Path Variables:**
- `${INPUT_DIR}`: Staged input files
- `${PACK_DATA_DIR}`: Pack's bundled data
- `${RUN_TMP}`: Temporary workspace

**Example:**
```yaml
capabilities:
  fs:
    allow: true
    read:
      allowlist:
        - ${INPUT_DIR}/**
        - ${PACK_DATA_DIR}/**
    write:
      allowlist:
        - ${RUN_TMP}/**
```

##### Clock Capability (`clock`)
Controls time access.

**Fields:**
- `allow` (boolean): Enable real-time clock (false = frozen/deterministic)

**Example:**
```yaml
capabilities:
  clock:
    allow: false  # Use deterministic time
```

##### Subprocess Capability (`subprocess`)
Controls subprocess execution.

**Fields:**
- `allow` (boolean): Enable subprocess execution
- `allowlist` (array): Allowed binaries (absolute paths)

**Example:**
```yaml
capabilities:
  subprocess:
    allow: true
    allowlist:
      - /usr/bin/exiftool
      - /usr/local/bin/ffmpeg
```

#### `card` (string)
Path to Model Card or Pack Card documentation.

**Common names:**
- `CARD.md`
- `README.md`
- `MODEL_CARD.md`

#### `policy` (object)
Policy constraints and requirements.

**Common fields:**
- `network`: Allowed network endpoints
- `ef_vintage_min`: Minimum emission factor vintage year
- `data_residency`: Data residency requirements
- `max_runtime_seconds`: Maximum execution time

#### `security` (object)
Security metadata and constraints.

**Fields:**
- `sbom`: Path to Software Bill of Materials
- `signatures`: List of signature files
- `vulnerabilities`: Vulnerability tolerance settings

**Example:**
```yaml
security:
  sbom: sbom.spdx.json
  signatures:
    - pack.sig
    - pack.sig.asc
  vulnerabilities:
    max_critical: 0
    max_high: 5
    max_medium: 20
```

#### `metadata` (object)
Additional metadata for pack discovery.

**Common fields:**
- `description`: Human-readable description
- `author`: Pack author
- `homepage`: Project homepage
- `repository`: Source repository
- `tags`: Categorization tags

## Validation

### Schema Validation

Manifests are validated against the JSON Schema at runtime. Key validation rules:

1. **Name validation**: Must be DNS-safe
2. **Version validation**: Must be semantic version
3. **License validation**: Must be valid SPDX identifier
4. **Path validation**: No path traversal (`../`)
5. **Binary validation**: Must use absolute paths
6. **Domain validation**: Valid URL patterns

### Security Validation

The installer performs additional security checks:

1. **Capability restrictions**:
   - No wildcards in binary paths
   - No dangerous binaries (shells, interpreters)
   - No write access to system directories
   - No read access to sensitive paths

2. **Dependency validation**:
   - Version constraints recommended
   - Known vulnerability checks
   - License compatibility

3. **File validation**:
   - All referenced files must exist
   - Proper file permissions
   - No symlinks to external paths

## Best Practices

### 1. Principle of Least Privilege
Only request capabilities absolutely necessary for functionality.

```yaml
# Good: Specific, minimal capabilities
capabilities:
  net:
    allow: true
    outbound:
      allowlist:
        - https://api.emissions.gov/v1/*

# Bad: Overly broad capabilities
capabilities:
  net:
    allow: true
    outbound:
      allowlist:
        - https://*  # Too permissive!
```

### 2. Version Pinning
Pin dependency versions for reproducibility.

```yaml
# Good: Pinned versions
dependencies:
  - numpy==1.21.0
  - pandas>=1.3.0,<2.0.0

# Bad: Unpinned versions
dependencies:
  - numpy
  - pandas
```

### 3. Documentation
Always include pack card and descriptions.

```yaml
card: CARD.md
metadata:
  description: "Calculates Scope 1 & 2 emissions for commercial buildings"
  author: "GreenLang Team"
  homepage: "https://greenlang.org/packs/emissions"
```

### 4. Supply Chain Security
Include SBOM and signatures.

```yaml
security:
  sbom: sbom.spdx.json
  signatures:
    - pack.sig
```

## CLI Commands

### Validate Manifest
```bash
gl pack validate /path/to/pack
```

### Lint Capabilities
```bash
gl capabilities lint /path/to/pack
```

### Show Capabilities
```bash
gl capabilities show /path/to/pack
```

### Install with Validation
```bash
gl pack install /path/to/pack
```

## Examples

### Minimal Pack
```yaml
name: minimal-pack
version: 1.0.0
kind: pack
license: MIT
contents:
  pipelines:
    - pipeline.yaml
```

### Data Processing Pack
```yaml
name: data-processor
version: 2.0.0
kind: pack
license: Apache-2.0
contents:
  pipelines:
    - process.yaml
  datasets:
    - data/inputs.csv
capabilities:
  fs:
    allow: true
    read:
      allowlist:
        - ${INPUT_DIR}/**
        - ${PACK_DATA_DIR}/**
    write:
      allowlist:
        - ${RUN_TMP}/**
dependencies:
  - pandas>=1.3.0
  - numpy>=1.20.0
```

### API Integration Pack
```yaml
name: api-connector
version: 1.5.0
kind: connector
license: MIT
contents:
  pipelines:
    - fetch.yaml
    - sync.yaml
capabilities:
  net:
    allow: true
    outbound:
      allowlist:
        - https://api.partner.com/v2/*
  fs:
    allow: true
    write:
      allowlist:
        - ${RUN_TMP}/**
  clock:
    allow: true  # Need real-time for API timestamps
compat:
  greenlang: ">=0.9.0"
  python: ">=3.8,<3.12"
metadata:
  description: "Fetches emission data from Partner API"
  tags:
    - connector
    - api
    - emissions
```

## Migration Guide

### From v0.x to v1.0

1. **Add capabilities block** (defaults to deny-all):
```yaml
# Add this section to existing manifests
capabilities:
  net:
    allow: false
  fs:
    allow: false
  clock:
    allow: false
  subprocess:
    allow: false
```

2. **Update dependencies format**:
```yaml
# Old format
dependencies:
  - numpy

# New format
dependencies:
  - numpy>=1.20.0
```

3. **Add security section**:
```yaml
security:
  sbom: sbom.spdx.json
```

## Troubleshooting

### Common Errors

#### "Pack name must be DNS-safe"
- Use lowercase letters, numbers, and hyphens only
- Start with a letter, end with alphanumeric
- 3-64 characters total

#### "Path traversal not allowed"
- Remove `../` from paths
- Use environment variables like `${INPUT_DIR}`

#### "Binary path must be absolute"
- Change `exiftool` to `/usr/bin/exiftool`
- Use `which exiftool` to find absolute path

#### "Capability denied by organization policy"
- Check organization's capability policy
- Request approval from security team
- Reduce requested capabilities

### Debug Mode

Enable debug output for manifest validation:
```bash
GL_DEBUG=1 gl pack validate /path/to/pack
```

View capability enforcement logs:
```bash
GL_AUDIT=1 gl run pipeline.yaml
```