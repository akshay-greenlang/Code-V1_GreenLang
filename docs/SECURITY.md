# GreenLang Security Documentation

## Overview

GreenLang implements comprehensive security measures to protect against supply chain attacks, data exfiltration, and other security threats. This document describes the security features and how to configure them.

## Default Security Posture

GreenLang follows a **default-deny** security model:

- All network operations require HTTPS by default
- Pack signatures are verified before installation
- Path traversal protection is enforced during extraction
- TLS 1.2+ is required for all connections
- Unsigned packs are blocked unless explicitly allowed

## Security Features

### 1. HTTPS Enforcement

All network operations enforce HTTPS by default:

- Pack downloads from registries
- Hub API communications
- Git repository cloning
- External resource fetching

**HTTP URLs are blocked** unless explicitly allowed in development mode.

### 2. TLS Configuration

- **Minimum TLS version**: 1.2
- **Certificate verification**: Always enabled
- **Hostname verification**: Always enabled
- **Custom CA support**: Via `GL_CA_BUNDLE` environment variable

### 3. Signature Verification

All packs must be signed before installation:

- Signatures are verified using checksums (Sigstore integration coming soon)
- Publisher identity is validated
- Integrity checks prevent tampering

### 4. Path Traversal Protection

Archive extraction is protected against path traversal attacks:

- Absolute paths are blocked
- `../` sequences are detected and blocked
- Symlinks are validated to stay within extraction directory
- All paths are resolved and validated before extraction

### 5. Capability Model

Packs must explicitly declare required capabilities in their manifest:

```yaml
capabilities:
  net: false       # Outbound network access
  fs: false        # Filesystem access beyond workdir
  subprocess: false # Subprocess execution
  clock: false     # Time access (for deterministic runs)
```

## Environment Variables

### Production Settings (Default)

These are the secure defaults for production:

- `GL_CA_BUNDLE`: Path to custom CA certificate bundle (optional)

### Development Settings

⚠️ **WARNING**: These settings reduce security and should NEVER be used in production:

- `GL_ALLOW_INSECURE_FOR_DEV=1`: Allow HTTP URLs (development only)
- `GL_ALLOW_UNSIGNED_FOR_DEV=1`: Allow unsigned packs (development only)
- `GL_ALLOW_PRIVATE_HOSTS_FOR_DEV=1`: Allow localhost/private IPs (development only)

## Security Checks

### Running Security Checks Locally

```bash
python scripts/check_security.py
```

This checks for:
- Insecure SSL/TLS patterns
- HTTP URLs
- Hardcoded secrets
- Path traversal vulnerabilities

### CI Security Checks

The CI pipeline automatically runs security checks on every push and pull request:

- Insecure pattern detection
- Dependency vulnerability scanning
- Environment variable validation
- File permission checks

## Corporate Environments

### Using Corporate CA Certificates

For environments with custom Certificate Authorities:

```bash
export GL_CA_BUNDLE=/path/to/corporate-ca-bundle.pem
gl pack install package-name
```

### Proxy Configuration

GreenLang respects standard proxy environment variables:

```bash
export HTTPS_PROXY=https://proxy.company.com:8080
export NO_PROXY=*.internal.company.com
```

## Security Best Practices

### For Pack Developers

1. **Never disable SSL verification** in your pack code
2. **Always use HTTPS** for external resources
3. **Declare minimal capabilities** in your manifest
4. **Sign your packs** before distribution
5. **Validate all inputs** and sanitize paths

### For GreenLang Users

1. **Keep GreenLang updated** to get latest security patches
2. **Only install trusted packs** from verified publishers
3. **Review pack capabilities** before installation
4. **Use signature verification** (enabled by default)
5. **Run in containers** for additional isolation

### For System Administrators

1. **Disable development flags** in production environments
2. **Configure corporate CA certificates** properly
3. **Monitor audit logs** for suspicious activity
4. **Use network policies** to restrict egress
5. **Implement RBAC** for pack installation permissions

## Reporting Security Issues

If you discover a security vulnerability in GreenLang:

1. **DO NOT** open a public issue
2. Email security@greenlang.ai with details
3. Include steps to reproduce if possible
4. Allow 90 days for disclosure

## Security Roadmap

### Completed (v0.2.0)
- ✅ HTTPS enforcement
- ✅ Path traversal protection
- ✅ Basic signature verification
- ✅ TLS 1.2+ requirement
- ✅ Security audit scripts

### In Progress
- 🚧 Sigstore keyless signing integration
- 🚧 SBOM generation and verification
- 🚧 Runtime sandboxing improvements

### Planned
- 📋 Hardware security module (HSM) support
- 📋 Attestation and provenance tracking
- 📋 Advanced threat detection
- 📋 Security scorecard integration

## Compliance

GreenLang's security features help meet requirements for:

- **Supply Chain Security**: SLSA Level 2 (working toward Level 3)
- **NIST Guidelines**: Following NIST 800-218 practices
- **EU Regulations**: GDPR data residency controls
- **Industry Standards**: CIS benchmarks compliance

## Appendix: Security Module Architecture

```
greenlang/security/
├── network.py       # HTTPS enforcement, TLS config
├── paths.py         # Path traversal protection
├── signatures.py    # Pack signing and verification
└── __init__.py      # Public API exports
```

For more details on implementation, see the source code in `core/greenlang/security/`.