# Software Bill of Materials (SBOM) Documentation

## Overview

Starting with **GreenLang v0.2.0**, all released artifacts include comprehensive Software Bills of Materials (SBOMs) that provide complete transparency about dependencies and components. This documentation covers SBOM generation, verification, and usage.

## Table of Contents

- [What is an SBOM?](#what-is-an-sbom)
- [Supported Formats](#supported-formats)
- [SBOM Coverage](#sbom-coverage)
- [Local SBOM Generation](#local-sbom-generation)
- [CI/CD SBOM Generation](#cicd-sbom-generation)
- [Verification Steps](#verification-steps)
- [Using SBOMs](#using-sboms)
- [Security Considerations](#security-considerations)

## What is an SBOM?

A Software Bill of Materials (SBOM) is a comprehensive inventory of all components, dependencies, and metadata associated with a software artifact. It's the software equivalent of a list of ingredients on food packaging, providing:

- **Transparency**: Complete visibility into what's included in our software
- **Security**: Ability to quickly identify vulnerable components
- **Compliance**: Meet regulatory and customer requirements
- **Supply Chain Security**: Track provenance and authenticity of components

## Supported Formats

GreenLang generates SBOMs in two industry-standard formats:

### Primary Format: CycloneDX JSON
- **File pattern**: `*.cdx.json`
- **Standard**: [CycloneDX 1.5](https://cyclonedx.org/)
- **Used for**: Container attestations, primary distribution
- **Benefits**: Rich component metadata, vulnerability tracking, licensing information
- **Tool Version**: Syft v1.0.0

### Secondary Format: SPDX JSON
- **File pattern**: `*.spdx.json`
- **Standard**: [SPDX 2.3](https://spdx.dev/)
- **Used for**: Compliance, interoperability
- **Benefits**: ISO standard, wide tool support, legal focus
- **Tool Version**: Syft v1.0.0

### Signing & Attestation Tools
- **Cosign Version**: v2.2.4
- **Signing Method**: Keyless OIDC (Sigstore)
- **Attestation Type**: In-toto with CycloneDX predicate

## SBOM Coverage

### Python Packages

For each Python package we publish:

| Artifact | CycloneDX | SPDX | Signed | File Pattern |
|----------|-----------|------|--------|--------------|
| Wheel (.whl) | ✅ Required | ✅ Required | ✅ | `sbom-greenlang-{version}-wheel.{format}.json` |
| Source dist (.tar.gz) | ✅ Required | ⚪ Optional | ✅ | `sbom-greenlang-{version}-sdist.{format}.json` |

### Docker Images

For each container image we publish:

| Image | CycloneDX | SPDX | Attested | Registry |
|-------|-----------|------|----------|----------|
| greenlang-runner | ✅ Required | ✅ Required | ✅ | GHCR + Docker Hub |
| greenlang-full | ✅ Required | ✅ Required | ✅ | GHCR + Docker Hub |

## Local SBOM Generation

### Prerequisites

Install required tools:

```bash
# macOS
brew install syft cosign

# Linux
curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sudo sh -s -- -b /usr/local/bin
curl -sSfL https://github.com/sigstore/cosign/releases/latest/download/cosign-linux-amd64 -o cosign
chmod +x cosign && sudo mv cosign /usr/local/bin/

# Windows (PowerShell as Administrator)
# Download from: https://github.com/anchore/syft/releases
# Download from: https://github.com/sigstore/cosign/releases
```

### Quick Start

Use our provided scripts for automated SBOM generation:

```bash
# Generate all SBOMs (Python + Docker)
./scripts/generate-sboms.sh 0.2.0

# Generate Python SBOMs only
./scripts/generate-sboms.sh 0.2.0 --python

# Generate Docker SBOMs only
./scripts/generate-sboms.sh 0.2.0 --docker

# Windows users
scripts\generate-sboms.bat 0.2.0
```

### Manual Generation

#### Python Wheel SBOM

```bash
# Build the wheel
python -m build

# Generate CycloneDX SBOM (PRIMARY)
syft dist/greenlang-0.2.0-py3-none-any.whl \
  -o cyclonedx-json=artifacts/sbom/sbom-greenlang-0.2.0-wheel.cdx.json

# Generate SPDX SBOM (SECONDARY)
syft dist/greenlang-0.2.0-py3-none-any.whl \
  -o spdx-json=artifacts/sbom/sbom-greenlang-0.2.0-wheel.spdx.json
```

#### Python Source Distribution SBOM

```bash
# Generate CycloneDX SBOM
syft dist/greenlang-0.2.0.tar.gz \
  -o cyclonedx-json=artifacts/sbom/sbom-greenlang-0.2.0-sdist.cdx.json
```

#### Docker Image SBOM

```bash
# Build the image
docker build -t greenlang-runner:0.2.0 -f Dockerfile.runner .

# Generate CycloneDX SBOM
syft docker:greenlang-runner:0.2.0 \
  -o cyclonedx-json=artifacts/sbom/sbom-image-greenlang-runner-0.2.0.cdx.json

# Generate SPDX SBOM
syft docker:greenlang-runner:0.2.0 \
  -o spdx-json=artifacts/sbom/sbom-image-greenlang-runner-0.2.0.spdx.json
```

## CI/CD SBOM Generation

Our GitHub Actions workflows automatically generate SBOMs for all releases:

### Workflow: `sbom-generation.yml`

Triggered on:
- Release tags (`v*`)
- Manual workflow dispatch
- Called by other workflows

Features:
- Parallel SBOM generation for all artifacts
- Automatic signing with Cosign (keyless OIDC)
- Attestation attachment to container images
- Validation and verification gates

### Integration Points

```yaml
# In release workflows
jobs:
  generate-sboms:
    uses: ./.github/workflows/sbom-generation.yml
    with:
      version: '0.2.0'
      python_artifacts: true
      docker_images: true
```

## Verification Steps

### 1. Verify SBOM Signatures

All SBOMs are signed using Cosign keyless signing:

```bash
# Verify SBOM signature (requires .sig and .crt files)
cosign verify-blob \
  --certificate sbom-greenlang-0.2.0-wheel.cdx.json.crt \
  --signature sbom-greenlang-0.2.0-wheel.cdx.json.sig \
  --certificate-identity-regexp ".*" \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com \
  sbom-greenlang-0.2.0-wheel.cdx.json
```

### 2. Verify Container Image Attestations

Docker images have CycloneDX SBOMs attached as in-toto attestations:

```bash
# Verify attestation exists and is valid
cosign verify-attestation \
  --type cyclonedx \
  --certificate-identity-regexp "https://github.com/.*" \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com \
  ghcr.io/akshay-greenlang/greenlang-runner:0.2.0

# Download the SBOM from the attestation
cosign download attestation \
  --type cyclonedx \
  ghcr.io/akshay-greenlang/greenlang-runner:0.2.0 | \
  jq -r '.payload' | base64 -d | jq '.predicate'
```

### 3. Verify SBOM Contents

```bash
# Check CycloneDX SBOM structure
jq '.bomFormat, .specVersion, .components | length' sbom-greenlang-0.2.0-wheel.cdx.json

# Check SPDX SBOM structure
jq '.spdxVersion, .packages | length' sbom-greenlang-0.2.0-wheel.spdx.json

# List all components with versions
jq -r '.components[] | "\(.name)@\(.version)"' sbom-greenlang-0.2.0-wheel.cdx.json
```

### 4. Scan for Vulnerabilities

Use the SBOM with vulnerability scanners:

```bash
# Using Grype
grype sbom:sbom-greenlang-0.2.0-wheel.cdx.json

# Using Trivy
trivy sbom sbom-greenlang-0.2.0-wheel.cdx.json
```

## Using SBOMs

### Download from GitHub Releases

Starting with v0.2.0, all SBOMs are attached to GitHub releases:

```bash
# Download all SBOMs for a release
gh release download v0.2.0 --pattern "sbom-*.json"

# Download signatures
gh release download v0.2.0 --pattern "*.sig"
```

### Extract from Container Images

```bash
# Method 1: Using cosign
cosign download attestation --type cyclonedx \
  ghcr.io/akshay-greenlang/greenlang-runner:0.2.0

# Method 2: Using syft (regenerates SBOM)
syft registry:ghcr.io/akshay-greenlang/greenlang-runner:0.2.0 \
  -o cyclonedx-json
```

### Integration with Security Tools

#### SPDX Tools

```bash
# Validate SPDX document
pyspdx validate sbom-greenlang-0.2.0-wheel.spdx.json

# Convert to other formats
pyspdx convert sbom-greenlang-0.2.0-wheel.spdx.json -o sbom.rdf
```

#### CycloneDX Tools

```bash
# Install CycloneDX CLI
npm install -g @cyclonedx/cli

# Validate CycloneDX document
cyclonedx validate --input-file sbom-greenlang-0.2.0-wheel.cdx.json

# Convert between formats
cyclonedx convert --input-file sbom.cdx.json --output-file sbom.xml
```

## Security Considerations

### Supply Chain Security

1. **Always verify signatures** before trusting an SBOM
2. **Check attestations** for container images before deployment
3. **Monitor for vulnerabilities** using the SBOMs with scanning tools
4. **Validate provenance** by checking the certificate identity

### Best Practices

1. **Regular Updates**: Regenerate SBOMs when dependencies change
2. **Automated Scanning**: Integrate SBOM scanning into CI/CD pipelines
3. **Retention**: Keep SBOMs for all deployed versions for incident response
4. **Access Control**: Treat SBOMs as sensitive (they reveal your full dependency tree)

### Incident Response

When a vulnerability is discovered:

```bash
# 1. Check if affected component is in your SBOMs
for sbom in artifacts/sbom/*.cdx.json; do
  echo "Checking $sbom..."
  jq -r '.components[] | select(.name=="vulnerable-package") | .version' "$sbom"
done

# 2. Find all images/packages containing the component
grep -r "vulnerable-package" artifacts/sbom/

# 3. Generate updated SBOMs after patching
./scripts/generate-sboms.sh 0.2.1
```

## Troubleshooting

### Common Issues

#### Syft Installation Failed

```bash
# Manual installation
wget https://github.com/anchore/syft/releases/download/v1.0.0/syft_1.0.0_linux_amd64.tar.gz
tar -xzf syft_1.0.0_linux_amd64.tar.gz
sudo mv syft /usr/local/bin/
```

#### Attestation Verification Failed

```bash
# Check if image has attestations
cosign tree ghcr.io/akshay-greenlang/greenlang-runner:0.2.0

# Verify with debug output
COSIGN_EXPERIMENTAL=1 cosign verify-attestation \
  --type cyclonedx \
  --certificate-identity-regexp ".*" \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com \
  ghcr.io/akshay-greenlang/greenlang-runner:0.2.0 \
  --output text
```

#### SBOM Generation Out of Memory

```bash
# Use streaming mode for large images
syft docker:large-image:latest \
  -o cyclonedx-json \
  --parallelism 1 \
  --file-metadata-cataloger-enabled=false
```

## Compliance & Standards

Our SBOM implementation complies with:

- **Executive Order 14028**: Improving the Nation's Cybersecurity
- **NTIA Minimum Elements**: All required fields included
- **CycloneDX 1.5**: Full specification compliance
- **SPDX 2.3**: ISO/IEC 5962:2021 standard compliance

## Tool Versions

The following tool versions are used in our SBOM generation pipeline:

- **Syft**: v1.0.0 (SBOM generation)
- **Cosign**: v2.2.4 (signing and attestations)
- **GitHub Actions**: ubuntu-latest runner
- **Docker Buildx**: v0.12.4 (multi-arch builds)

## References

- [CycloneDX Specification](https://cyclonedx.org/specification/overview/)
- [SPDX Specification](https://spdx.github.io/spdx-spec/)
- [Syft Documentation](https://github.com/anchore/syft)
- [Cosign Documentation](https://docs.sigstore.dev/cosign/overview/)
- [NTIA SBOM Requirements](https://www.ntia.gov/sbom)

## Support

For issues or questions about SBOMs:

1. Check this documentation
2. Review [GitHub Issues](https://github.com/akshay-greenlang/Code-V1_GreenLang/issues)
3. Contact the security team

---

*Last updated: v0.2.0 | Generated with comprehensive SBOM support*