# GreenLang Docker Multi-Architecture Build System

## Overview

This document describes the GreenLang v0.2.0 Docker multi-architecture build system, which provides secure, signed, and SBOM-attached container images for both production runtime and development/CI environments.

## 🎯 Objectives Achieved

### Security First Approach
- ✅ **Non-root user** (UID 10001) by default
- ✅ **Read-only root filesystem** with explicit volume mounts
- ✅ **Pinned base images** with SHA256 digests
- ✅ **Multi-stage builds** for minimal attack surface
- ✅ **Dropped ALL capabilities**
- ✅ **Security scanning** with Trivy
- ✅ **SBOM generation** with Syft (SPDX and CycloneDX)
- ✅ **Keyless signing** with Cosign (OIDC)
- ✅ **OPA policies** for container admission control

### Multi-Architecture Support
- ✅ **linux/amd64** (x86_64)
- ✅ **linux/arm64** (ARM64/Apple Silicon)
- ✅ Single manifest list for seamless platform selection

### Image Variants

#### 1. Runner Image (Production Minimal)
- **Purpose**: Lightweight runtime for production pipelines
- **Base**: `python:3.11-slim` with security hardening
- **Size**: ~150MB
- **Contents**: GreenLang core only, no build tools
- **Tags**:
  - `ghcr.io/akshay-greenlang/greenlang-runner:0.2.0`
  - `docker.io/greenlang/core-runner:0.2.0`

#### 2. Full Image (Developer/CI)
- **Purpose**: Complete environment for development and CI/CD
- **Base**: `python:3.11-slim` with build tools
- **Size**: ~800MB
- **Contents**: GreenLang with all extras, dev tools, compilers
- **Tags**:
  - `ghcr.io/akshay-greenlang/greenlang-full:0.2.0`
  - `docker.io/greenlang/core-full:0.2.0`

## 🔒 Security Implementation

### Container Security Policies

Located in `/policies/container/`:

1. **admission.rego**: Container admission control
   - Registry allowlisting
   - Signature verification
   - SBOM requirements
   - Vulnerability thresholds

2. **network.rego**: Network segmentation
   - Egress control to approved endpoints
   - Ingress restrictions
   - Rate limiting

3. **resources.rego**: Resource management
   - CPU/Memory limits by container type
   - Namespace quotas
   - Priority classes

### Kubernetes Security

1. **NetworkPolicy**: Strict ingress/egress controls
2. **PodSecurityPolicy**: Enforced security standards
3. **RBAC**: Minimal permissions with separate service accounts
4. **Security Contexts**: Non-root, read-only FS, dropped capabilities

## 📦 Build Process

### Local Build

```bash
# Build runner image
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --file Dockerfile.runner \
  --build-arg GL_VERSION=0.2.0 \
  --tag greenlang/core-runner:local \
  .

# Build full image
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --file Dockerfile.full \
  --build-arg GL_VERSION=0.2.0 \
  --tag greenlang/core-full:local \
  .
```

### CI/CD Pipeline

The GitHub Actions workflow (`release-docker.yml`) automates:

1. **Multi-arch build** with Docker Buildx
2. **Push to registries** (GHCR and Docker Hub)
3. **SBOM generation** with Syft
4. **Vulnerability scanning** with Trivy
5. **Image signing** with Cosign (keyless)
6. **Attestation** with SLSA framework

Trigger with:
```bash
# Via tag push
git tag v0.2.0
git push --tags

# Or manual dispatch
gh workflow run release-docker.yml -f version=0.2.0
```

## 🔍 Verification

### Verify Image Signatures

```bash
# Install cosign
brew install cosign

# Verify runner image
cosign verify ghcr.io/akshay-greenlang/greenlang-runner:0.2.0 \
  --certificate-identity-regexp "https://github.com/.*/release-docker.yml.*" \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com

# Verify full image
cosign verify ghcr.io/akshay-greenlang/greenlang-full:0.2.0 \
  --certificate-identity-regexp "https://github.com/.*/release-docker.yml.*" \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
```

### View SBOM

```bash
# Download SBOM
cosign download sbom ghcr.io/akshay-greenlang/greenlang-runner:0.2.0 > sbom.json

# View with syft
syft ghcr.io/akshay-greenlang/greenlang-runner:0.2.0
```

### Security Scan

```bash
# Scan with Trivy
trivy image ghcr.io/akshay-greenlang/greenlang-runner:0.2.0

# Scan with Grype
grype ghcr.io/akshay-greenlang/greenlang-runner:0.2.0
```

## 🚀 Usage Examples

### Docker Run

```bash
# Run runner image
docker run --rm \
  --user 10001:10001 \
  --read-only \
  --security-opt=no-new-privileges:true \
  --cap-drop=ALL \
  -v $(pwd)/examples:/examples:ro \
  ghcr.io/akshay-greenlang/greenlang-runner:0.2.0 \
  compose /examples/pipeline.yaml

# Interactive development with full image
docker run -it --rm \
  --user 10001:10001 \
  -v $(pwd):/workspace \
  ghcr.io/akshay-greenlang/greenlang-full:0.2.0 \
  bash
```

### Docker Compose

```bash
# Start services
docker-compose up -d

# Run commands
docker-compose exec greenlang-runner gl --help
docker-compose exec greenlang-full pytest

# View logs
docker-compose logs -f greenlang-runner
```

### Kubernetes Deployment

```bash
# Create namespace with security policies
kubectl apply -f kubernetes/manifests/podsecuritypolicy.yaml

# Deploy runner
kubectl apply -f kubernetes/manifests/runner-deployment.yaml

# Apply network policies
kubectl apply -f kubernetes/manifests/networkpolicy.yaml

# Verify deployment
kubectl get pods -n greenlang-system
kubectl logs -n greenlang-system -l app.kubernetes.io/name=greenlang-runner
```

## 🔧 Troubleshooting

### Common Issues

1. **Permission Denied**
   - Ensure volumes are writable by UID 10001
   - Check SecurityContext settings

2. **Read-only filesystem errors**
   - Mount writable volumes for cache/logs/tmp
   - Use emptyDir volumes in Kubernetes

3. **Network connectivity issues**
   - Check NetworkPolicy allows required egress
   - Verify DNS resolution works

4. **Image pull errors**
   - Verify image signatures if policy enforcement enabled
   - Check registry credentials

### Debug Commands

```bash
# Check image details
docker inspect ghcr.io/akshay-greenlang/greenlang-runner:0.2.0

# Run with debug logging
docker run --rm \
  -e GL_LOG_LEVEL=DEBUG \
  ghcr.io/akshay-greenlang/greenlang-runner:0.2.0 \
  --help

# Test OPA policies
opa test policies/container/

# Verify Kubernetes security context
kubectl get pod <pod-name> -o jsonpath='{.spec.securityContext}'
```

## 📊 Performance Considerations

### Image Sizes
- Runner: ~150MB (optimized for fast pulls)
- Full: ~800MB (includes build tools)

### Startup Times
- Runner: <5 seconds
- Full: <10 seconds

### Resource Usage
- Runner: 100m CPU, 128Mi RAM (minimum)
- Full: 500m CPU, 512Mi RAM (minimum)

## 🔄 Maintenance

### Updating Base Images

1. Find latest secure digest:
```bash
docker pull python:3.11-slim
docker inspect python:3.11-slim | grep -i sha256
```

2. Update Dockerfiles with new digest
3. Run security scan
4. Test builds
5. Create PR with changes

### Dependency Updates

```bash
# Update PyPI packages
pip-compile --upgrade requirements.txt

# Rebuild images
./scripts/test-docker-build.sh

# Run security scan
trivy image greenlang/core-runner:local
```

## 📝 Compliance

### Standards Met
- ✅ CIS Docker Benchmark (Level 1)
- ✅ NIST 800-190 Container Security
- ✅ SLSA Level 3 Supply Chain Security
- ✅ OCI Image Specification 1.0

### Security Controls
- SC-1: Non-root execution
- SC-2: Read-only filesystem
- SC-3: Network segmentation
- SC-4: Resource limits
- SC-5: Vulnerability scanning
- SC-6: Image signing
- SC-7: SBOM generation
- SC-8: Policy enforcement

## 🎯 Future Enhancements

### v0.3.0 Roadmap
- [ ] Distroless base images
- [ ] Hardware attestation
- [ ] Runtime security (Falco)
- [ ] Service mesh integration
- [ ] GPU support for ML workloads

### v0.4.0 Roadmap
- [ ] WASM runtime support
- [ ] Confidential computing
- [ ] Zero-trust networking
- [ ] Automated compliance reporting

## 📞 Support

For issues or questions:
- GitHub Issues: https://github.com/akshay-greenlang/Code-V1_GreenLang/issues
- Security: security@greenlang.io
- Documentation: https://docs.greenlang.io/docker

## ✅ Checklist Summary

**Completed:**
- [x] Multi-arch Dockerfiles (runner + full)
- [x] GitHub Actions CI/CD workflow
- [x] SBOM generation and attachment
- [x] Cosign keyless signing
- [x] Trivy vulnerability scanning
- [x] OPA admission policies
- [x] Kubernetes manifests with security
- [x] Docker Compose for local testing
- [x] Test scripts
- [x] Comprehensive documentation

**Security Hardening Applied:**
- [x] User UID 10001 (high UID for containers)
- [x] Read-only root filesystem
- [x] Dropped ALL capabilities
- [x] No new privileges
- [x] Seccomp profiles
- [x] Network policies
- [x] Resource limits
- [x] Health checks

This implementation provides enterprise-grade container security while maintaining developer productivity and operational efficiency.