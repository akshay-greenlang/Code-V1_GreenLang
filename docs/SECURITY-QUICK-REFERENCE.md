# Security Quick Reference - GreenLang Docker Images

## TL;DR Security Verification

### Quick Verification (One Command)
```bash
./scripts/verify-security-artifacts.sh [VERSION]
```

### Manual Verification Commands

#### 1. Verify Cosign Signature
```bash
cosign verify ghcr.io/[owner]/greenlang-runner:[version] \
  --certificate-identity-regexp "https://github.com/[owner]/greenlang/.github/workflows/.*" \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
```

#### 2. View SBOM
```bash
cosign download sbom ghcr.io/[owner]/greenlang-runner:[version] | jq
```

#### 3. Check Vulnerabilities
Go to: Repository → Security → Code scanning alerts → Filter "trivy"

## Image URLs

| Image Type | GHCR | Docker Hub |
|------------|------|------------|
| **Runner** | `ghcr.io/[owner]/greenlang-runner:[version]` | `greenlang/core-runner:[version]` |
| **Full** | `ghcr.io/[owner]/greenlang-full:[version]` | `greenlang/core-full:[version]` |

## Security Features ✅

- ✅ **Cosign Signatures**: Keyless OIDC signing
- ✅ **SBOM**: SPDX format attached to images
- ✅ **Vulnerability Scans**: Trivy scanning with SARIF results
- ✅ **Multi-arch**: linux/amd64, linux/arm64
- ✅ **Transparency**: Rekor transparency log

## Prerequisites

```bash
# Install required tools
brew install cosign jq docker

# Or using other package managers:
# Ubuntu: apt-get install jq docker.io && install cosign manually
# Windows: choco install jq docker-desktop && install cosign manually
```

## Common Commands

```bash
# Full verification with JSON output
./scripts/verify-security-artifacts.sh --output-format json [VERSION]

# Skip SBOM check
./scripts/verify-security-artifacts.sh --skip-sbom [VERSION]

# Verify specific registry
./scripts/verify-security-artifacts.sh -r ghcr.io -o myorg [VERSION]

# CI-friendly verification
./scripts/verify-security-artifacts-ci.sh [VERSION]
```

## Production Usage

```dockerfile
# ✅ Good: Use specific version
FROM ghcr.io/[owner]/greenlang-runner:0.2.0

# ❌ Avoid: Using latest tag
FROM ghcr.io/[owner]/greenlang-runner:latest
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Signature verification fails | Check certificate identity pattern |
| SBOM not found | Verify image was built with latest workflows |
| No vulnerability results | Check GitHub Security tab permissions |
| Tools missing | Install cosign, jq, docker |

## Links

- 📖 [Full Documentation](../SECURITY-VERIFICATION.md)
- 🔧 [Verification Script](../scripts/verify-security-artifacts.sh)
- 🛡️ [Security Policy](../SECURITY.md)
- 🚀 [Workflows](.github/workflows/)

---
**Quick Start**: `./scripts/verify-security-artifacts.sh [VERSION]`