# GreenLang Security Model: Default-Deny Architecture

## Overview

GreenLang implements a **default-deny security model** where all operations are denied unless explicitly allowed by policy. This document describes the security architecture, policy enforcement points, and configuration options.

## Core Principles

### 1. Default Deny Everything
- **No implicit permissions**: Operations fail unless explicitly allowed
- **Policy failures = denial**: Missing or broken policies result in denial, not permission
- **Fail closed**: System denies access when uncertain

### 2. Multiple Enforcement Points
- **Install time**: Packs cannot be installed without passing policy
- **Execution time**: Pipeline steps cannot run without authorization
- **Capability gates**: Resources (network, filesystem, subprocess) are denied by default

### 3. Explicit Allowlists
- **Publisher allowlist**: Only approved publishers can install packs
- **Region allowlist**: Execution limited to approved regions
- **Capability allowlist**: Only declared and approved capabilities are granted

## Policy Enforcement Points

### Installation Policy
```yaml
# Checks performed at pack installation:
- Signature verification (required by default)
- Publisher allowlist check
- License compatibility
- Security metadata (SBOM required)
- Size limits
```

### Execution Policy
```yaml
# Checks performed at runtime:
- User authentication
- Region restrictions
- Resource limits (memory, CPU)
- Rate limiting
- Capability requirements
```

### Capability Gates
```yaml
# Capabilities must be:
1. Declared in pack manifest
2. Allowed by organization policy
3. Explicitly requested at runtime
```

## Configuration

### Organization Allowlists

Create `~/.greenlang/org-config.yaml`:
```yaml
org:
  allowed_publishers:
    - greenlang-official
    - verified
    - partner-1

  allowed_regions:
    - US
    - EU
    - APAC

  allowed_capabilities:
    - fs  # Filesystem access only by default
```

### Pack Manifest Requirements

Packs must declare capabilities in `pack.yaml`:
```yaml
name: my-pack
version: 1.0.0
publisher: greenlang-official  # Must be in allowlist

# Declare required capabilities
capabilities:
  - fs   # Filesystem access
  - net  # Network access

policy:
  # Network targets (if net capability used)
  network:
    - api.weather.gov
    - "*.greenlang.io"

  # Data residency requirements
  data_residency:
    - US
    - EU
```

## CLI Flags and Overrides

### Installation Flags
```bash
# Normal installation (strict)
gl pack add emissions-core@1.0.0

# Allow unsigned packs (development only!)
gl pack add test-pack --allow-unsigned

# Bypass all policies (DANGEROUS - dev only!)
gl pack add test-pack --policy-permissive
```

### Publishing Flags
```bash
# Normal publishing (with policy checks)
gl pack publish ./my-pack

# Skip policy checks (not recommended)
gl pack publish ./my-pack --skip-policy
```

## Policy Testing

### Test Policy Decisions
```bash
# Test if a pack would be allowed
gl policy check pack.yaml

# Test with custom input
gl policy test policies/default/allowlists.rego --input test-input.json

# Validate policy syntax
gl policy validate my-policy.rego
```

### Example Test Input
```json
{
  "pack": {
    "name": "test-pack",
    "version": "1.0.0",
    "publisher": "greenlang-official",
    "signature_verified": true,
    "declared_capabilities": ["fs"]
  },
  "request": {
    "requested_capabilities": ["fs"],
    "reason": "Read emission factors",
    "run_id": "abc123"
  },
  "org": {
    "allowed_publishers": ["greenlang-official"],
    "allowed_regions": ["US"],
    "allowed_capabilities": ["fs"]
  },
  "env": {
    "region": "US",
    "runtime": "docker"
  },
  "user": {
    "authenticated": true,
    "role": "developer"
  }
}
```

## Error Messages

### Standard Error Format
```
POLICY.DENIED_INSTALL: Publisher 'unknown' not in allowed list
  → Add 'unknown' to org.allowed_publishers or contact security team

POLICY.DENIED_CAPABILITY: Capability 'net' not declared in pack manifest
  → Add 'net' to 'capabilities' in manifest.yaml

POLICY.DENIED_EXECUTION: User not authenticated
  → Authenticate with 'gl auth login'
```

## Security Best Practices

### For Organizations

1. **Maintain minimal allowlists**
   - Only add trusted publishers
   - Limit allowed regions to where you operate
   - Grant minimal capabilities (start with 'fs' only)

2. **Regular audits**
   ```bash
   # List all installed packs
   gl pack list --verbose

   # Check policy compliance
   gl policy audit
   ```

3. **Monitor policy decisions**
   - All denials are logged
   - Review `/var/log/greenlang/policy.log` regularly

### For Pack Developers

1. **Declare minimal capabilities**
   ```yaml
   # Bad - requesting unnecessary capabilities
   capabilities: [fs, net, subprocess]

   # Good - only what's needed
   capabilities: [fs]
   ```

2. **Sign your packs**
   ```bash
   gl pack sign ./my-pack --key ~/.ssh/signing-key
   ```

3. **Provide SBOM**
   ```bash
   gl pack generate-sbom ./my-pack > sbom.json
   ```

## Migration Guide

### From Permissive to Strict

#### Week 1: Observation Mode
```bash
# Enable strict mode in CI only
export GL_POLICY_STRICT=1
```

#### Week 2: Warning Mode
```bash
# Log warnings but don't block
export GL_POLICY_WARN_ONLY=1
```

#### Week 3: Enforcement
```bash
# Full enforcement (default)
# No environment variables needed
```

### Rollback Plan
```bash
# Emergency override (temporary only!)
export GL_POLICY_PERMISSIVE=1

# This will:
# - Allow unsigned packs
# - Skip publisher verification
# - Allow all capabilities
# - LOG HUGE WARNINGS
```

## Troubleshooting

### Common Issues

#### "Pack installation denied"
```bash
# Check why it was denied
gl policy check pack.yaml --verbose

# Common solutions:
# 1. Sign the pack
# 2. Add publisher to allowlist
# 3. Ensure SBOM is included
```

#### "Capability denied"
```bash
# Check current capabilities
gl pack info my-pack | grep capabilities

# Add to manifest.yaml:
capabilities:
  - fs
  - net  # If network needed
```

#### "OPA not installed"
```bash
# Install OPA
gl doctor --install-opa

# Or manually:
# Linux/Mac
curl -L -o opa https://openpolicyagent.org/downloads/latest/opa_linux_amd64
chmod +x opa
sudo mv opa /usr/local/bin/

# Windows
# Download from https://openpolicyagent.org/docs/latest/#running-opa
```

## Security Contact

Report security issues to: security@greenlang.io

## Appendix: Policy Rule Reference

### Installation Rules
- `allow_install`: Pack can be installed
- `publisher_allowed`: Publisher in allowlist
- `signature_valid`: Digital signature verified
- `sbom_present`: SBOM provided

### Execution Rules
- `allow_execution`: Pipeline can run
- `authenticated`: User is authenticated
- `region_allowed`: Current region is allowed
- `rate_limit_ok`: Within rate limits

### Capability Rules
- `capability_allowed[cap]`: Specific capability is allowed
- `capability_declared`: Pack declares the capability
- `capability_org_allowed`: Organization allows it
- `capability_requested`: Runtime requests it

## Version History

- v1.0.0 (2024-01): Initial default-deny implementation
- v1.1.0 (planned): Add Sigstore integration
- v1.2.0 (planned): Add supply chain attestations