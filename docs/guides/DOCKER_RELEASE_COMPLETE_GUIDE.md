# Complete Docker Release Workflow Guide

## Overview

The `docker-release-complete.yml` workflow is a comprehensive, production-ready solution that addresses all critical Docker build requirements including:

- **Multi-Architecture Support**: AMD64 + ARM64 platforms
- **Security Artifacts**: Cosign signatures and SBOM generation
- **Dual Registry Publication**: GitHub Container Registry (GHCR) + Docker Hub
- **Vulnerability Scanning**: Trivy security analysis
- **Quality Assurance**: Comprehensive testing and verification

## Features Comparison

| Feature | docker-complete-dod.yml | docker-publish-public.yml | **docker-release-complete.yml** |
|---------|-------------------------|----------------------------|----------------------------------|
| Multi-arch (AMD64+ARM64) | ✅ | ✅ | ✅ |
| Cosign Keyless Signing | ✅ | ✅ | ✅ |
| SBOM Generation | ✅ | ✅ | ✅ Enhanced |
| Trivy Vulnerability Scanning | ✅ | ✅ | ✅ Enhanced |
| Docker Hub Publication | ❌ | ✅ | ✅ Conditional |
| GHCR Publication | ✅ | ✅ | ✅ |
| Build Caching | Basic | ✅ | ✅ Enhanced |
| Conditional Logic | ❌ | Limited | ✅ Advanced |
| Error Handling | Basic | Good | ✅ Comprehensive |
| Testing & Verification | Basic | Good | ✅ Extensive |
| Job Parallelization | ❌ | ❌ | ✅ Matrix Strategy |

## Usage

### 1. Manual Trigger (Recommended)

```bash
# Navigate to Actions tab in GitHub
# Select "Complete Docker Release - Production Ready"
# Click "Run workflow"
# Fill in parameters:
```

**Parameters:**
- `version`: Version to release (e.g., `0.2.0`) - **Required**
- `force_rebuild`: Force rebuild even if images exist - Default: `false`
- `skip_docker_hub`: Skip Docker Hub publication (GHCR only) - Default: `false`
- `publish_latest`: Also tag and publish as latest - Default: `true`

### 2. Automatic Triggers

- **Git Tag Push**: Triggered on tags matching `v*.*.*` (e.g., `v0.2.0`)
- **GitHub Release**: Triggered when a release is published

### 3. GitHub CLI Trigger

```bash
# Install GitHub CLI if not already installed
gh auth login

# Trigger the workflow
gh workflow run "Complete Docker Release - Production Ready" \
  --ref main \
  -f version="0.2.0" \
  -f publish_latest="true"
```

## Required Secrets

### GitHub Container Registry (GHCR)
- Automatically handled via `GITHUB_TOKEN` (no setup required)

### Docker Hub (Optional but Recommended)
```bash
# Set these secrets in your repository settings
DOCKERHUB_USERNAME=your_dockerhub_username
DOCKERHUB_TOKEN=your_dockerhub_access_token
```

**Note**: If Docker Hub secrets are not provided, the workflow will automatically skip Docker Hub publication and only publish to GHCR.

## Workflow Architecture

The workflow uses a multi-job architecture for optimal performance and reliability:

```
┌─────────────┐
│  Preflight  │──┐
│   Checks    │  │
└─────────────┘  │
                 ▼
┌─────────────────────────────┐
│    Build & Publish         │
│  (Matrix: runner, full)     │
└─────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────┐
│  Security & Compliance      │
│  (Matrix: runner, full)     │
└─────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────┐
│   Verification & Testing    │
│ (Matrix: amd64/arm64 × 2)   │
└─────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────┐
│    Final Reporting          │
└─────────────────────────────┘
```

## Published Images

### Runner Image (Minimal Production)
- **GHCR**: `ghcr.io/[owner]/greenlang-runner:[version]`
- **Docker Hub**: `greenlang/core-runner:[version]`
- **Purpose**: Minimal runtime for production deployments
- **Size**: Optimized for smallest footprint

### Full Image (Developer/CI)
- **GHCR**: `ghcr.io/[owner]/greenlang-full:[version]`
- **Docker Hub**: `greenlang/core-full:[version]`
- **Purpose**: Complete development and CI environment
- **Includes**: Build tools, development dependencies

### Tag Strategy
- `[version]`: Specific version (e.g., `0.2.0`)
- `[major.minor]`: Major.minor version (e.g., `0.2`)
- `latest`: Latest stable release (conditional)

## Security Features

### 1. Cosign Signatures
All images are signed using Cosign with keyless OIDC authentication:

```bash
# Verify image signature
cosign verify ghcr.io/[owner]/greenlang-runner:0.2.0 \
  --certificate-identity-regexp 'https://github.com/[repo]/.*' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
```

### 2. Software Bill of Materials (SBOM)
Multiple SBOM formats generated and attached:
- **SPDX JSON**: Industry standard format
- **CycloneDX JSON**: OWASP standard format
- **Syft JSON**: Native Syft format

```bash
# Download SBOM
cosign download sbom ghcr.io/[owner]/greenlang-runner:0.2.0

# View attached SBOM
cosign tree ghcr.io/[owner]/greenlang-runner:0.2.0
```

### 3. Vulnerability Scanning
Comprehensive security scanning with Trivy:
- **SARIF Output**: Uploaded to GitHub Security tab
- **Table Output**: Human-readable console output
- **Severity Levels**: CRITICAL and HIGH vulnerabilities flagged

## Quality Assurance

### Multi-Platform Testing
Every image is tested on both architectures:
- **linux/amd64**: Native x86_64 testing
- **linux/arm64**: ARM64 testing via QEMU emulation

### Functionality Verification
- Basic command execution tests
- Version verification
- Help command validation
- Multi-arch manifest verification

### Performance Optimizations
- **Registry Caching**: Build cache stored in registry
- **Layer Caching**: Optimized Dockerfile layer caching
- **Parallel Builds**: Matrix strategy for concurrent jobs
- **QEMU Optimization**: Latest QEMU version for ARM64 builds

## Monitoring and Troubleshooting

### Workflow Status
- Check the Actions tab for real-time progress
- Each job provides detailed logging
- Matrix jobs show per-image/platform status

### Common Issues

#### 1. Docker Hub Authentication Failed
```
Error: login denied for docker.io
```
**Solution**: Verify `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` secrets are set correctly.

#### 2. ARM64 Build Timeout
```
Error: Job exceeded maximum timeout
```
**Solution**: ARM64 builds via QEMU are slower. This is expected behavior.

#### 3. Cosign Signing Failed
```
Error: failed to sign image
```
**Solution**: Ensure `id-token: write` permission is granted in workflow.

### Debug Mode
Enable debug logging by setting workflow variable:
```yaml
env:
  ACTIONS_STEP_DEBUG: true
```

## Integration Examples

### Docker Compose
```yaml
version: '3.8'
services:
  greenlang-runner:
    image: ghcr.io/[owner]/greenlang-runner:0.2.0
    # or: greenlang/core-runner:0.2.0
    platform: linux/amd64  # or linux/arm64
```

### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: greenlang-app
spec:
  template:
    spec:
      containers:
      - name: greenlang
        image: ghcr.io/[owner]/greenlang-runner:0.2.0
        # Kubernetes will automatically select the correct architecture
```

### CI/CD Pipeline
```yaml
# .github/workflows/test.yml
jobs:
  test:
    runs-on: ubuntu-latest
    container:
      image: ghcr.io/[owner]/greenlang-full:latest
    steps:
      - uses: actions/checkout@v4
      - run: gl test
```

## Migration from Other Workflows

### From docker-complete-dod.yml
- Add Docker Hub secrets (optional)
- Update workflow references
- Benefits: Docker Hub publication, enhanced error handling

### From docker-publish-public.yml
- No breaking changes
- Benefits: Better job parallelization, enhanced testing

### From release-docker.yml
- Update trigger references
- Benefits: Conditional logic, comprehensive reporting

## Best Practices

1. **Version Management**: Use semantic versioning (MAJOR.MINOR.PATCH)
2. **Secret Management**: Rotate Docker Hub tokens regularly
3. **Image Size**: Monitor image sizes and optimize Dockerfiles
4. **Security**: Review vulnerability scan results regularly
5. **Testing**: Validate images in staging before production use

## Support

For issues or questions:
1. Check workflow logs in GitHub Actions
2. Review this documentation
3. Check existing GitHub issues
4. Create new issue with workflow run URL and error details

---

This workflow represents the definitive, production-ready Docker build solution for the GreenLang project, meeting all DoD requirements and industry best practices.