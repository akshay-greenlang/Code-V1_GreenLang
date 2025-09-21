# Docker Hub Publication Setup

This document outlines the setup required to publish GreenLang Docker images to Docker Hub alongside the existing GHCR (GitHub Container Registry) publication.

## Overview

The GitHub Actions workflows have been updated to support dual publication:
- **GHCR**: `ghcr.io/akshay-greenlang/greenlang-runner` and `ghcr.io/akshay-greenlang/greenlang-full`
- **Docker Hub**: `greenlang/core-runner` and `greenlang/core-full`

## Required GitHub Secrets

To enable Docker Hub publication, the following repository secrets must be configured in the GitHub repository settings:

### DOCKERHUB_USERNAME
- **Value**: Your Docker Hub username (should be `greenlang` for the organization account)
- **Description**: Username for authenticating with Docker Hub
- **Location**: Repository Settings → Secrets and variables → Actions → Repository secrets

### DOCKERHUB_TOKEN
- **Value**: Docker Hub access token (not password)
- **Description**: Access token for secure authentication with Docker Hub
- **How to generate**:
  1. Log in to Docker Hub
  2. Go to Account Settings → Security → Access Tokens
  3. Click "New Access Token"
  4. Name: `GreenLang-GitHub-Actions`
  5. Permissions: `Read, Write, Delete` (for pushing images)
  6. Copy the generated token (you won't see it again)

## Setting Up the Secrets

1. Navigate to your GitHub repository
2. Go to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add both secrets:
   - Name: `DOCKERHUB_USERNAME`, Value: `greenlang`
   - Name: `DOCKERHUB_TOKEN`, Value: `[your-docker-hub-access-token]`

## Docker Hub Organization Setup

Ensure the following Docker Hub repositories exist and are properly configured:

### greenlang/core-runner
- **Repository**: `https://hub.docker.com/r/greenlang/core-runner`
- **Description**: "Minimal production runtime for GreenLang pipelines"
- **Visibility**: Public
- **Tags**: `0.2.0`, `0.2`, `latest`

### greenlang/core-full
- **Repository**: `https://hub.docker.com/r/greenlang/core-full`
- **Description**: "Developer/CI image with build tools for GreenLang"
- **Visibility**: Public
- **Tags**: `0.2.0`, `0.2`, `latest`

## Workflow Updates

The following workflows have been updated to support Docker Hub publication:

### 1. `.github/workflows/release-docker.yml`
- **Primary workflow** for releasing Docker images
- Triggers on version tags (`v*.*.*`) and manual dispatch
- Includes full security features (signing, SBOM, vulnerability scanning)
- Publishes to both GHCR and Docker Hub simultaneously

### 2. `.github/workflows/docker-publish-public.yml`
- **Secondary workflow** for manual public releases
- Manual dispatch only
- Simplified workflow focused on public distribution
- Publishes to both GHCR and Docker Hub

## Image Structure

Both workflows build and publish two image variants:

| Image Type | GHCR Location | Docker Hub Location | Purpose |
|------------|---------------|---------------------|---------|
| Runner | `ghcr.io/akshay-greenlang/greenlang-runner` | `greenlang/core-runner` | Minimal production runtime |
| Full | `ghcr.io/akshay-greenlang/greenlang-full` | `greenlang/core-full` | Developer/CI with build tools |

## Tagging Strategy

Each image is published with three tags:
- **Specific version**: `0.2.0` (exact version)
- **Major.minor**: `0.2` (allows patch updates)
- **Latest**: `latest` (always points to newest release)

## Multi-Architecture Support

All images are built and published for:
- `linux/amd64` (Intel/AMD 64-bit)
- `linux/arm64` (ARM 64-bit, including Apple Silicon)

## Security Features

All images include:
- **Digital signatures** using Cosign (keyless signing with OIDC)
- **Software Bill of Materials (SBOM)** in SPDX and CycloneDX formats
- **Vulnerability scanning** with Trivy
- **SLSA Level 3 provenance** attestation

## Testing the Setup

After configuring the secrets, test the setup by:

1. **Manual dispatch of docker-publish-public workflow**:
   ```bash
   gh workflow run docker-publish-public.yml -f version=0.2.0
   ```

2. **Check Docker Hub repositories**:
   - Visit `https://hub.docker.com/r/greenlang/core-runner`
   - Visit `https://hub.docker.com/r/greenlang/core-full`
   - Verify images are present with correct tags

3. **Test pulling images**:
   ```bash
   # Test runner image
   docker pull greenlang/core-runner:0.2.0
   docker run --rm greenlang/core-runner:0.2.0 --version

   # Test full image
   docker pull greenlang/core-full:0.2.0
   docker run --rm greenlang/core-full:0.2.0 gl --version
   ```

## Troubleshooting

### Common Issues

1. **Authentication Failed**:
   - Verify `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` secrets are set correctly
   - Ensure the Docker Hub access token has `Read, Write, Delete` permissions
   - Check that the Docker Hub account has push access to the `greenlang` organization

2. **Repository Not Found**:
   - Ensure the Docker Hub repositories `greenlang/core-runner` and `greenlang/core-full` exist
   - Verify the repositories are public
   - Check organization membership and permissions

3. **Multi-arch Build Failures**:
   - Verify QEMU setup is working correctly
   - Check that Dockerfile supports multi-arch builds
   - Ensure base images support the target architectures

### Monitoring

Monitor the publication process through:
- **GitHub Actions**: Check workflow runs in the Actions tab
- **Docker Hub**: Monitor repository activity and download stats
- **Security**: Review vulnerability scan results in the Security tab

## Next Steps

Once Docker Hub publication is working:

1. **Update documentation** to reference Docker Hub images
2. **Update examples** to use Docker Hub images for easier access
3. **Monitor usage** and adjust retention policies as needed
4. **Consider automated security monitoring** for published images

## Support

For issues with Docker Hub publication:
- Check GitHub Actions workflow logs
- Review Docker Hub repository settings
- Verify secret configuration
- Test with manual workflow dispatch first