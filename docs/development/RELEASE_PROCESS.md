# GreenLang Release Process

This document outlines the release process for GreenLang, including both automated and manual approaches.

## Overview

GreenLang follows a semantic versioning approach with automated release workflows that handle:
- Version validation
- Python package building
- Changelog generation
- GitHub release creation
- Optional PyPI publishing

## Release Types

### Patch Release (0.1.0 → 0.1.1)
- Bug fixes
- Minor improvements
- Documentation updates

### Minor Release (0.1.0 → 0.2.0)
- New features
- Backward-compatible API changes
- Significant improvements

### Major Release (0.1.0 → 1.0.0)
- Breaking changes
- Major architecture changes
- API incompatible changes

## Automated Release Process

### Prerequisites

1. **Repository Secrets** (for PyPI publishing):
   - `PYPI_API_TOKEN`: Production PyPI API token
   - `TEST_PYPI_API_TOKEN`: TestPyPI API token

2. **Repository Variables** (optional):
   - `ENABLE_PYPI_PUBLISH`: Set to `'true'` to enable PyPI publishing
   - `ENABLE_TESTPYPI_PUBLISH`: Set to `'true'` to enable TestPyPI publishing

### Release Steps

1. **Create and Push Tag**:
   ```bash
   git tag v0.1.1
   git push origin v0.1.1
   ```

2. **Automated Process**: The release workflow automatically:
   - Validates the tag version matches `pyproject.toml`
   - Builds Python packages (wheel and source distribution)
   - Generates changelog from commit history
   - Creates GitHub release with artifacts
   - Tests package installation across platforms
   - Optionally publishes to PyPI (if enabled)

3. **Verification**: Check the GitHub releases page for the new release.

## Manual Release Process

Use the manual process for more control or when automation fails.

### Using the Release Script

1. **Prepare Release**:
   ```bash
   # Dry run to see what will happen
   python scripts/release.py --version 0.1.1 --dry-run

   # Create the release
   python scripts/release.py --version 0.1.1

   # Create and push to origin
   python scripts/release.py --version 0.1.1 --push
   ```

2. **Script Features**:
   - Updates version in `pyproject.toml` and `greenlang/__init__.py`
   - Generates changelog from git commits using conventional commit format
   - Updates `CHANGELOG.md`
   - Creates git tag
   - Optionally pushes to origin

### Manual Steps

If you need to do everything manually:

1. **Update Version Files**:
   ```bash
   # Update pyproject.toml
   sed -i 's/version = "0.1.0"/version = "0.1.1"/' pyproject.toml

   # Update greenlang/__init__.py
   sed -i 's/__version__ = "0.1.0"/__version__ = "0.1.1"/' greenlang/__init__.py
   ```

2. **Update Changelog**:
   ```bash
   # Edit CHANGELOG.md manually or use:
   python scripts/release.py --version 0.1.1 --dry-run
   ```

3. **Commit and Tag**:
   ```bash
   git add .
   git commit -m "chore: bump version to 0.1.1"
   git tag v0.1.1
   git push origin main
   git push origin v0.1.1
   ```

## Changelog Generation

### Conventional Commits

The automated changelog generation recognizes these commit prefixes:

- `feat:` or `feature:` → **Features**
- `fix:` → **Bug Fixes**
- `docs:` → **Documentation**
- `refactor:` → **Refactoring**
- `test:` → **Testing**
- `chore:` → **Chores**
- Other commits → **Other Changes**

### Manual Changelog Workflow

You can also trigger changelog generation manually:

```bash
# Via GitHub Actions UI
# Go to Actions → Auto-generate Changelog → Run workflow

# Or using GitHub CLI
gh workflow run changelog.yml -f version=0.1.1
```

## Build and Distribution

### Local Build

```bash
# Install build tools
pip install build twine

# Build packages
python -m build

# Check packages
python -m twine check dist/*

# Test installation
pip install dist/*.whl
```

### PyPI Publishing

The automated workflow can publish to PyPI if enabled:

1. **TestPyPI** (for testing):
   - Set `ENABLE_TESTPYPI_PUBLISH` to `'true'`
   - Add `TEST_PYPI_API_TOKEN` secret
   - Package is published to test.pypi.org

2. **Production PyPI**:
   - Set `ENABLE_PYPI_PUBLISH` to `'true'`
   - Add `PYPI_API_TOKEN` secret
   - Package is published to pypi.org

### Manual PyPI Publishing

```bash
# Publish to TestPyPI
python -m twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ greenlang

# Publish to production PyPI
python -m twine upload dist/*
```

## Version Management

### Version Sources

The version is maintained in two places:
1. `pyproject.toml` - Primary source
2. `greenlang/__init__.py` - For runtime access

### Version Validation

The automated workflow validates that:
- Tag version (e.g., `v0.1.1`) matches `pyproject.toml` version (`0.1.1`)
- Version follows semantic versioning format
- Working directory is clean before release

### setuptools_scm Integration

Optional: For automatic version management from git tags:
```toml
[tool.setuptools_scm]
write_to = "greenlang/_version.py"
```

## Release Checklist

### Pre-Release
- [ ] All tests passing
- [ ] Documentation updated
- [ ] Version updated in both files
- [ ] CHANGELOG.md updated
- [ ] No uncommitted changes

### Release
- [ ] Tag created and pushed
- [ ] GitHub release created
- [ ] Packages built successfully
- [ ] Installation tests pass
- [ ] PyPI publication (if applicable)

### Post-Release
- [ ] Verify GitHub release
- [ ] Test installation from PyPI
- [ ] Announce release
- [ ] Update documentation site

## Troubleshooting

### Common Issues

1. **Version Mismatch Error**:
   ```
   Version mismatch: pyproject.toml has X, tag has Y
   ```
   Solution: Ensure both files have the same version.

2. **Build Failures**:
   - Check `pyproject.toml` syntax
   - Verify all dependencies are available
   - Check package structure

3. **PyPI Upload Failures**:
   - Verify API tokens are correct
   - Check if version already exists
   - Ensure package passes `twine check`

4. **Test Failures**:
   - Check import statements
   - Verify CLI entry points
   - Test in clean environment

### Rollback Process

If a release needs to be rolled back:

1. **Delete GitHub Release**:
   ```bash
   gh release delete v0.1.1
   ```

2. **Delete Tag**:
   ```bash
   git tag -d v0.1.1
   git push origin --delete v0.1.1
   ```

3. **Revert Version Changes**:
   ```bash
   git revert <commit-hash>
   ```

## Security Considerations

1. **API Tokens**: Store in GitHub Secrets, never in code
2. **Provenance**: All releases include build provenance
3. **Signing**: Consider adding package signing
4. **SBOM**: Software Bill of Materials generated for packs

## Contact

For questions about the release process:
- Create an issue in the repository
- Contact the GreenLang Team at team@greenlang.io