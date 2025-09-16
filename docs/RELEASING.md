# GreenLang Release Process

## Overview
This document describes the release process for GreenLang, ensuring consistent versioning across all components.

## Version Management

### Single Source of Truth
The version is maintained in a single `VERSION` file at the repository root. All components read from this file:
- Python packages (via `pyproject.toml` and `setup.py`)
- Docker images (via build args)
- CLI tools (via package imports)
- Documentation

### Version Format
We follow Semantic Versioning (SemVer): `MAJOR.MINOR.PATCH`
- **MAJOR**: Breaking API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

## Release Process

### 1. Prepare the Release

#### Update Version
```bash
# Edit the VERSION file
echo "0.2.1" > VERSION
```

#### Update Documentation
1. Edit `VERSION.md` with release notes:
   - Add new version section
   - List major changes
   - Document breaking changes (if any)
   - Update migration instructions

2. Update `CHANGELOG.md` (if exists):
   - Follow Keep a Changelog format
   - Include all changes since last release

#### Run Version Consistency Check
```bash
# On Unix/Linux/Mac
./scripts/check_version_consistency.sh

# On Windows
scripts\check_version_consistency.bat
```

### 2. Commit and Tag

#### Create Release Commit
```bash
git add VERSION VERSION.md
git commit -m "chore(release): bump to 0.2.1"
```

#### Create Git Tag
```bash
# Create annotated tag
git tag -a v0.2.1 -m "Release v0.2.1"

# Push changes and tag
git push origin main
git push origin v0.2.1
```

### 3. Build and Publish

#### PyPI Release
```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info/

# Build distributions
python -m build

# Upload to PyPI (requires credentials)
python -m twine upload dist/*
```

#### Docker Image
```bash
# Read version
GL_VERSION=$(cat VERSION)

# Build image
docker build \
  --build-arg GL_VERSION="$GL_VERSION" \
  -t ghcr.io/greenlang/greenlang:$GL_VERSION \
  -t ghcr.io/greenlang/greenlang:latest \
  .

# Push to registry
docker push ghcr.io/greenlang/greenlang:$GL_VERSION
docker push ghcr.io/greenlang/greenlang:latest
```

### 4. GitHub Release

1. Go to GitHub Releases page
2. Click "Create a new release"
3. Select the tag (e.g., `v0.2.1`)
4. Title: `GreenLang v0.2.1`
5. Add release notes from `VERSION.md`
6. Attach artifacts:
   - Python wheel (`dist/*.whl`)
   - Source distribution (`dist/*.tar.gz`)
   - SBOM file (if generated)
   - Signatures (if signing)

### 5. Post-Release

#### Update Default Version
After release, bump to next development version:
```bash
# Bump to next dev version
echo "0.2.2-dev" > VERSION
git add VERSION
git commit -m "chore: bump to 0.2.2-dev"
git push
```

#### Announce Release
- Discord announcement
- Twitter/Social media
- Email to users list
- Update documentation site

## CI/CD Integration

### GitHub Actions Workflow
The release process can be automated with GitHub Actions:

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Verify tag matches VERSION
        run: |
          TAG="${GITHUB_REF#refs/tags/}"
          VERSION="v$(cat VERSION)"
          if [ "$TAG" != "$VERSION" ]; then
            echo "Tag $TAG doesn't match VERSION $VERSION"
            exit 1
          fi

      - name: Build and publish to PyPI
        # ... build steps

      - name: Build and push Docker image
        # ... docker steps
```

## Rollback Procedure

If a release needs to be rolled back:

1. **PyPI**: Cannot delete, but can yank:
   ```bash
   python -m twine yank greenlang==0.2.1
   ```

2. **Docker**: Remove or retag:
   ```bash
   # Remove tag from registry (if permissions allow)
   # Or push previous version as latest
   docker pull ghcr.io/greenlang/greenlang:0.2.0
   docker tag ghcr.io/greenlang/greenlang:0.2.0 ghcr.io/greenlang/greenlang:latest
   docker push ghcr.io/greenlang/greenlang:latest
   ```

3. **Git**: Create a revert commit:
   ```bash
   git revert <commit-hash>
   git push
   ```

## Security Considerations

### Signing Releases
All releases should be signed:

```bash
# Sign the artifacts
gpg --detach-sign --armor dist/*.whl
gpg --detach-sign --armor dist/*.tar.gz

# Verify signatures
gpg --verify dist/*.whl.asc dist/*.whl
```

### SBOM Generation
Generate Software Bill of Materials:

```bash
# Using syft or similar tool
syft packages dir:. -o spdx-json > sbom.spdx.json
```

### Security Checklist
- [ ] No hardcoded secrets in code
- [ ] Dependencies are up to date
- [ ] Security vulnerabilities scanned
- [ ] SBOM generated
- [ ] Artifacts signed
- [ ] Default-deny policies enforced

## Troubleshooting

### Version Mismatch
If version inconsistency is detected:
1. Run `./scripts/check_version_consistency.sh`
2. Fix any reported issues
3. Ensure VERSION file is the single source of truth

### Build Failures
- Clear all caches: `rm -rf build/ dist/ *.egg-info/`
- Update build tools: `pip install --upgrade build twine`
- Check Python version compatibility

### Tag Issues
- Ensure tag format: `vX.Y.Z` (with 'v' prefix)
- Delete and recreate tag if needed:
  ```bash
  git tag -d v0.2.1
  git push origin :refs/tags/v0.2.1
  git tag -a v0.2.1 -m "Release v0.2.1"
  git push origin v0.2.1
  ```

## Support

For release issues:
- Open issue on GitHub
- Contact team@greenlang.io
- Discord: #releases channel