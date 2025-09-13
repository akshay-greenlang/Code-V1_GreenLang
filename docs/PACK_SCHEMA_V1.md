# Pack Manifest — `pack.yaml` (Spec v1.0)

**Status:** Stable (v1.0)  
**Scope:** Applies to all GreenLang packs.  
**SemVer:** Breaking changes bump MAJOR; additive fields bump MINOR.

## Table of Contents
- [Required Fields](#required-fields)
- [Optional Fields](#optional-fields)
- [Constraints](#constraints)
- [Examples](#examples)
- [Validation Rules](#validation-rules)
- [Deprecation & Migration](#deprecation--migration)

## Required Fields

### `name` (string, slug)
- **Format:** DNS-safe; `[a-z0-9-]{3,64}`
- **Description:** Unique identifier for the pack
- **Example:** `"boiler-solar"`, `"ml-pipeline-v2"`

### `version` (semver)
- **Format:** `MAJOR.MINOR.PATCH`
- **Description:** Semantic version of the pack
- **Example:** `"1.0.0"`, `"2.3.1"`

### `kind` (enum)
- **Values:** `pack | dataset | connector`
- **Description:** Type of the GreenLang package
- **Default:** `"pack"`

### `license` (string)
- **Format:** Valid SPDX identifier
- **Description:** Software license for the pack
- **Examples:** `"MIT"`, `"Apache-2.0"`, `"Commercial"`, `"GPL-3.0"`
- **Reference:** [SPDX License List](https://spdx.org/licenses/)

### `contents.pipelines` (array of paths)
- **Format:** Array with at least one pipeline path
- **Description:** List of pipeline files (gl.yaml)
- **Example:** `["gl.yaml"]`, `["pipelines/main.yaml", "pipelines/backup.yaml"]`

## Optional Fields

### `compat` (object)
- **`compat.greenlang`** (string range): GreenLang version compatibility
  - Example: `">=0.3,<0.5"`
- **`compat.python`** (string range): Python version compatibility
  - Example: `">=3.10"`

### `contents` (extended)
- **`contents.agents`** (string[]): List of agent names
  - Example: `["BoilerAgent", "SolarOffsetAgent"]`
- **`contents.datasets`** (string[]): Dataset file paths
  - Example: `["datasets/ef_in_2025.csv"]`
- **`contents.reports`** (string[]): Report template paths
  - Example: `["reports/cfo_brief.html.j2"]`

### `dependencies` (array)
- **Format:** Array of strings or objects
- **Examples:**
  ```yaml
  dependencies:
    - "pandas>=2.1"
    - { name: "ephem", version: ">=4.1" }
  ```

### `card` (path)
- **Description:** Path to Model Card or Pack Card documentation
- **Example:** `"CARD.md"`

### `policy` (object)
- **Description:** Policy constraints and requirements
- **Fields:**
  - `network`: Allowed network endpoints
  - `data_residency`: Data residency regions
  - `ef_vintage_min`: Minimum EF vintage year
- **Example:**
  ```yaml
  policy:
    network: ["era5:*"]
    data_residency: ["IN", "EU"]
    ef_vintage_min: 2024
  ```

### `security` (object)
- **`security.sbom`** (string): Path to SBOM file
  - Example: `"sbom.spdx.json"`
- **`security.signatures`** (array): Digital signatures
  - Example: `["pack.sig", "release.sig"]`

## Constraints

- **`version`**: Must match semver pattern `^\d+\.\d+\.\d+$`
- **`kind`**: Must be one of `pack|dataset|connector`
- **Paths**: Must be relative to pack root; all listed files MUST exist
- **License**: Must be valid SPDX identifier
- **Name**: Must be DNS-safe, lowercase, alphanumeric with hyphens

## Examples

### Minimal Example
```yaml
name: "boiler-solar"
version: "1.0.0"
kind: "pack"
license: "MIT"
contents:
  pipelines: ["gl.yaml"]
```

### Full Example
```yaml
name: "boiler-solar"
version: "1.0.0"
kind: "pack"
license: "MIT"
compat:
  greenlang: ">=0.3,<0.5"
  python: ">=3.10"
contents:
  pipelines: ["gl.yaml"]
  agents: ["BoilerAgent", "SolarOffsetAgent", "CarbonAgent", "ReportAgent"]
  datasets: ["datasets/ef_in_2025.csv"]
  reports: ["reports/cfo_brief.html.j2"]
dependencies:
  - "pandas>=2.1"
  - { name: "ephem", version: ">=4.1" }
card: "CARD.md"
policy:
  network: ["era5:*"]
  data_residency: ["IN", "EU"]
  ef_vintage_min: 2024
security:
  sbom: "sbom.spdx.json"
  signatures: ["pack.sig"]
```

## Validation Rules

### Required Validation (MUST PASS)
1. **Required keys present**: `name`, `version`, `kind`, `license`, `contents.pipelines`
2. **Enum validation**: `kind` must be valid enum value
3. **Semver validation**: `version` must be valid semantic version
4. **File existence**: Every path in `contents.*` and `card` must exist
5. **SPDX license**: License must be valid SPDX identifier

### Recommended Validation (WARNINGS)
1. **Missing recommended fields**: Warn if `card`, `compat.*`, or `security.sbom` missing
2. **Dependency clarity**: Warn if dependencies lack version constraints
3. **Documentation**: Warn if no README.md or CARD.md present

## Deprecation & Migration

### Version Support Timeline
- **v0.3-0.4**: Pre-v1 manifests accepted with deprecation warnings
- **v0.5+**: Only v1.0 manifests accepted (pre-v1 will fail)

### Migration Path
1. Use migration script: `scripts/migrate_pack_yaml_v1.py`
2. Script behavior:
   - Backs up original as `pack.yaml.bak`
   - Adds missing `contents.pipelines` if `gl.yaml` exists
   - Normalizes field ordering
   - Shows diff of changes

### Breaking Changes from Pre-v1
- `contents` is now required (minimum `pipelines` array)
- `kind` must be explicit enum value
- `version` must be strict semver (no `v` prefix)
- `license` must be valid SPDX identifier

## JSON Schema Reference

The formal JSON Schema is available at:
- Local: `schemas/pack.schema.v1.json`
- URL: `https://greenlang.io/schema/pack.v1.json`

## CLI Validation

Validate a pack manifest:
```bash
gl pack validate [path/to/pack.yaml]
gl pack validate --json  # JSON output
gl pack validate --strict  # Fail on warnings
```

## Backward Compatibility

### Reading Old Manifests
The parser will attempt to read pre-v1 manifests with these adaptations:
- Missing `contents`: Auto-add if `gl.yaml` exists
- Legacy `type` field: Map to `kind`
- Version with `v` prefix: Strip prefix

### Forward Compatibility
New optional fields can be added in v1.x without breaking v1.0 parsers:
- Unknown fields are preserved but not validated
- Tools should warn on unknown fields but not fail

## References

- [Semantic Versioning 2.0.0](https://semver.org/)
- [SPDX License List](https://spdx.org/licenses/)
- [JSON Schema Draft 2020-12](https://json-schema.org/)
- [GreenLang CLI Documentation](./CLI.md)