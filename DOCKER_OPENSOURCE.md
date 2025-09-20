# GreenLang Docker Images - Open Source Approach

## Container Registry Strategy

As an open-source project, GreenLang uses GitHub Container Registry (GHCR) as the primary distribution method for Docker images. This provides:

- **No credentials required for pulling** - Images are publicly accessible
- **Automatic authentication** - GitHub Actions use GITHUB_TOKEN automatically
- **Security scanning** - GitHub provides Dependabot alerts for container vulnerabilities
- **SBOM generation** - Software Bill of Materials included
- **Cosign signing** - Keyless signing with OIDC (no secrets needed)

## Available Images

### Production Runtime (Runner)
```bash
# Pull the minimal production image
docker pull ghcr.io/akshay-greenlang/greenlang-runner:0.2.0
docker pull ghcr.io/akshay-greenlang/greenlang-runner:latest

# Multi-architecture support
# Automatically pulls the correct architecture (amd64 or arm64)
docker run --rm ghcr.io/akshay-greenlang/greenlang-runner:0.2.0 --version
```

### Development Image (Full)
```bash
# Pull the full development image with build tools
docker pull ghcr.io/akshay-greenlang/greenlang-full:0.2.0
docker pull ghcr.io/akshay-greenlang/greenlang-full:latest
```

## Security Features

All images include:
- Non-root user (UID 10001)
- Read-only root filesystem
- Dropped all Linux capabilities
- Health checks
- Signed with Cosign (keyless)
- SBOM attached
- Regular vulnerability scanning

## Verification

### Verify Image Signature (No Keys Required)
```bash
# Uses keyless verification with OIDC
cosign verify ghcr.io/akshay-greenlang/greenlang-runner:0.2.0 \
  --certificate-identity-regexp "https://github.com/akshay-greenlang/*" \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
```

### Check SBOM
```bash
# Download and inspect the Software Bill of Materials
syft ghcr.io/akshay-greenlang/greenlang-runner:0.2.0 -o spdx-json
```

### Scan for Vulnerabilities
```bash
# Check for security vulnerabilities
trivy image ghcr.io/akshay-greenlang/greenlang-runner:0.2.0
```

## Fork-Friendly Setup

When forking GreenLang:

1. **No secrets needed** - The workflow uses GITHUB_TOKEN which is automatic
2. **Images publish to your namespace** - ghcr.io/YOUR-USERNAME/greenlang-runner
3. **Signing works automatically** - Cosign uses OIDC, no keys needed

### Optional: Docker Hub Mirror

If you want to also push to Docker Hub (optional):

1. Fork the repository
2. Add these secrets to your fork:
   - `DOCKERHUB_USERNAME`: Your Docker Hub username
   - `DOCKERHUB_TOKEN`: Your Docker Hub access token
3. Uncomment the Docker Hub sections in `.github/workflows/release-docker.yml`

## Building Locally

```bash
# Build runner image
docker buildx build -f Dockerfile.runner -t greenlang-runner:local .

# Build full image
docker buildx build -f Dockerfile.full -t greenlang-full:local .

# Multi-architecture build (requires buildx)
docker buildx create --use
docker buildx build --platform linux/amd64,linux/arm64 -f Dockerfile.runner -t greenlang-runner:local .
```

## Community Contributions

We welcome contributions! The open-source approach means:
- No proprietary dependencies
- Public CI/CD logs for transparency
- Community can verify builds
- Fork and customize freely

## Support

- Issues: https://github.com/akshay-greenlang/Code-V1_GreenLang/issues
- Discussions: https://github.com/akshay-greenlang/Code-V1_GreenLang/discussions
- Security: Report vulnerabilities via GitHub Security tab