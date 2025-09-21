# Security Verification Guide for GreenLang Docker Images

This document provides comprehensive guidance on verifying the security artifacts (cosign signatures, SBOM, and vulnerability scans) for GreenLang Docker images.

## Overview

GreenLang Docker images are secured with multiple layers of protection:

- **Cosign Signatures**: All images are signed using keyless OIDC signing with Sigstore
- **SBOM (Software Bill of Materials)**: Generated in SPDX format and attached to images
- **Vulnerability Scanning**: Performed with Trivy and results uploaded to GitHub Security tab
- **Multi-architecture**: Images support both linux/amd64 and linux/arm64

## Published Images

### Runner Images (Minimal Production)
- **GHCR**: `ghcr.io/[owner]/greenlang-runner:[version]`
- **Docker Hub**: `greenlang/core-runner:[version]`
- **Purpose**: Minimal production runtime for GreenLang pipelines

### Full Images (Developer/CI)
- **GHCR**: `ghcr.io/[owner]/greenlang-full:[version]`
- **Docker Hub**: `greenlang/core-full:[version]`
- **Purpose**: Development and CI image with build tools

## Security Artifacts

### 1. Cosign Signatures

All images are signed using keyless OIDC signing with Sigstore/Cosign. This provides:
- Cryptographic proof of authenticity
- Transparency log entries in Rekor
- No need to manage signing keys

#### Verification Command
```bash
# Install cosign
brew install cosign  # or your package manager

# Verify runner image signature
cosign verify ghcr.io/[owner]/greenlang-runner:[version] \
  --certificate-identity-regexp "https://github.com/[owner]/greenlang/.github/workflows/.*" \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com

# Verify full image signature
cosign verify ghcr.io/[owner]/greenlang-full:[version] \
  --certificate-identity-regexp "https://github.com/[owner]/greenlang/.github/workflows/.*" \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
```

### 2. SBOM (Software Bill of Materials)

SBOMs are generated in SPDX format and attached to each image, providing:
- Complete inventory of software components
- License information
- Vulnerability tracking capabilities

#### Viewing SBOM
```bash
# Download and view SBOM
cosign download sbom ghcr.io/[owner]/greenlang-runner:[version] | jq

# Save SBOM to file
cosign download sbom ghcr.io/[owner]/greenlang-runner:[version] > sbom.spdx.json

# View package count
cosign download sbom ghcr.io/[owner]/greenlang-runner:[version] | jq '.packages | length'
```

### 3. Vulnerability Scanning

Trivy vulnerability scans are performed on all images:
- Scans for OS and library vulnerabilities
- Results uploaded to GitHub Security tab
- SARIF format for integration with security tools

#### Accessing Scan Results
1. **GitHub Security Tab**: Navigate to the repository's Security tab > Code scanning alerts
2. **Workflow Artifacts**: Download scan results from workflow run artifacts
3. **Manual Scanning**: Run Trivy locally on the images

```bash
# Manual Trivy scan
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image ghcr.io/[owner]/greenlang-runner:[version]
```

## Verification Tools

### Automated Verification Script

Use the provided verification script for comprehensive checks:

```bash
# Basic verification
./scripts/verify-security-artifacts.sh 0.2.0

# Verbose output
./scripts/verify-security-artifacts.sh -v 0.2.0

# Custom registry/owner
./scripts/verify-security-artifacts.sh -r ghcr.io -o myorg 0.2.0

# Skip specific checks
./scripts/verify-security-artifacts.sh --skip-sbom 0.2.0

# JSON output
./scripts/verify-security-artifacts.sh --output-format json 0.2.0

# Markdown report
./scripts/verify-security-artifacts.sh --output-format markdown 0.2.0
```

### CI Integration

For CI environments, use the simplified script:

```bash
# Quick verification in CI
./scripts/verify-security-artifacts-ci.sh 0.2.0
```

## Security Workflow Details

### Workflows with Security Features

1. **release-docker.yml**: Complete security implementation
   - Cosign signing with OIDC
   - SBOM generation and attachment
   - Trivy vulnerability scanning
   - SARIF upload to GitHub Security

2. **docker-complete-dod.yml**: DoD-compliant security
   - Multi-architecture builds
   - Security artifact generation
   - Comprehensive verification

3. **docker-publish-public.yml**: Public release with security
   - Same security features as release workflow
   - Public package visibility

4. **trivy.yml**: Dedicated vulnerability scanning
   - Daily scheduled scans
   - SARIF results upload
   - Config and image scanning

### Required Permissions

All security workflows require specific GitHub Actions permissions:

```yaml
permissions:
  contents: read              # Read repository content
  packages: write            # Publish to GHCR
  id-token: write           # OIDC token for keyless signing
  security-events: write    # Upload SARIF to Security tab
```

## Manual Verification Steps

### Step 1: Verify Image Existence
```bash
# Check if images exist
docker manifest inspect ghcr.io/[owner]/greenlang-runner:[version]
docker manifest inspect ghcr.io/[owner]/greenlang-full:[version]
```

### Step 2: Verify Signatures
```bash
# Verify cosign signatures
cosign verify ghcr.io/[owner]/greenlang-runner:[version] \
  --certificate-identity-regexp "https://github.com/[owner]/greenlang/.github/workflows/.*" \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
```

### Step 3: Check SBOM
```bash
# Verify SBOM exists and is valid
sbom_json=$(cosign download sbom ghcr.io/[owner]/greenlang-runner:[version])
echo "$sbom_json" | jq -e '.spdxVersion and .creationInfo and .packages'
```

### Step 4: Review Vulnerability Scans
1. Go to GitHub repository → Security tab → Code scanning alerts
2. Filter by "trivy" to see vulnerability scan results
3. Review any critical or high-severity findings

### Step 5: Test Image Functionality
```bash
# Test runner image
docker run --rm ghcr.io/[owner]/greenlang-runner:[version] --version

# Test full image
docker run --rm ghcr.io/[owner]/greenlang-full:[version] gl --version
```

## Security Best Practices

### For Users

1. **Always verify signatures** before using images in production
2. **Review SBOM** to understand software components
3. **Monitor vulnerability scans** for security updates
4. **Use specific version tags** instead of `latest` in production

### For Maintainers

1. **Keep workflows updated** with latest action versions
2. **Monitor security alerts** and address vulnerabilities promptly
3. **Rotate signing certificates** as needed
4. **Test security verification** regularly

## Troubleshooting

### Common Issues

#### Signature Verification Fails
```bash
# Check if using correct certificate identity
cosign verify ghcr.io/[owner]/greenlang-runner:[version] \
  --certificate-identity-regexp "https://github.com/[owner]/greenlang/.github/workflows/.*" \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
```

#### SBOM Not Found
- Check if the image was built with the latest workflows
- Verify the image tag is correct
- Ensure the workflow completed successfully

#### Vulnerability Scan Results Missing
- Check GitHub Security tab permissions
- Verify SARIF files were uploaded in workflow
- Ensure `security-events: write` permission is set

### Getting Help

1. **GitHub Issues**: Report security verification problems
2. **Workflow Logs**: Check GitHub Actions logs for detailed error messages
3. **Documentation**: Review this guide and workflow comments

## Security Contact

For security-related issues or questions:
- **Security Issues**: Use GitHub's private vulnerability reporting
- **General Questions**: Open a GitHub issue with the `security` label
- **Urgent Issues**: Follow the security policy in SECURITY.md

## Compliance and Standards

### Standards Compliance
- **SLSA Level 3**: Build provenance and verification
- **Supply Chain Security**: SBOM generation and vulnerability scanning
- **DoD Requirements**: Multi-architecture, signed images, security scanning

### Audit Trail
- All security artifacts are logged in Sigstore Rekor transparency log
- Build provenance available through SLSA attestations
- Vulnerability scan history in GitHub Security tab

## Automated Monitoring

### GitHub Security Features
- **Dependabot**: Automated dependency updates
- **Security Advisories**: Vulnerability notifications
- **Code Scanning**: Trivy and other security tools

### External Monitoring
- **Rekor Transparency Log**: Public audit trail
- **CVE Databases**: Vulnerability tracking
- **Container Registries**: Security scan integration

---

**Last Updated**: {{ current_date }}
**Version**: 1.0
**Maintainers**: GreenLang Security Team