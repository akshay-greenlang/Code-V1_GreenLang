# GitHub Pre-Release Instructions for v0.2.0b2

This document provides step-by-step instructions for creating the GitHub pre-release for GreenLang v0.2.0b2.

## 🔧 Prerequisites

### 1. GitHub CLI Setup
Ensure GitHub CLI is installed and authenticated:

```bash
# Check if gh CLI is installed
gh --version

# If not installed, install from: https://cli.github.com/
# On Windows: winget install --id GitHub.cli
# On macOS: brew install gh
# On Linux: See https://github.com/cli/cli/blob/trunk/docs/install_linux.md

# Authenticate with GitHub
gh auth login
```

### 2. Verify Build Artifacts
Ensure all required files are present:

```bash
# Check distribution files
ls -la dist/
# Should contain:
# - greenlang-0.2.0b2-py3-none-any.whl
# - greenlang-0.2.0b2.tar.gz

# Check SBOM files
ls -la sbom/
# Should contain:
# - greenlang-full-0.2.0.spdx.json
# - greenlang-dist-0.2.0.spdx.json
# - greenlang-runner-0.2.0.spdx.json

# Check release notes
ls -la RELEASE_NOTES_v0.2.0b2.md
```

### 3. Repository Status
Ensure you're in the correct repository and branch:

```bash
# Check current repository
gh repo view

# Check current branch (should be on master/main)
git branch --show-current

# Ensure working directory is clean
git status
```

## 🚀 Release Execution

### Option 1: Using the Automated Scripts

#### On Linux/macOS/WSL (Bash):
```bash
# Preview what will be executed (dry run)
./scripts/create_github_release_v0.2.0b2.sh

# Execute the actual release
./scripts/create_github_release_v0.2.0b2.sh --execute
```

#### On Windows (Batch):
```cmd
# Preview what will be executed (dry run)
scripts\create_github_release_v0.2.0b2.bat

# Execute the actual release
scripts\create_github_release_v0.2.0b2.bat --execute
```

### Option 2: Manual Commands

If you prefer to run commands manually:

#### 1. Create and Push Git Tag
```bash
# Create the tag
git tag v0.2.0b2 -m "Release v0.2.0b2"

# Push the tag to origin
git push origin v0.2.0b2
```

#### 2. Create GitHub Release
```bash
# Create the pre-release with all artifacts
gh release create v0.2.0b2 \
  --notes-file "RELEASE_NOTES_v0.2.0b2.md" \
  --title "v0.2.0b2 – Infra Seed (Beta 2)" \
  --prerelease \
  "dist/greenlang-0.2.0b2-py3-none-any.whl" \
  "dist/greenlang-0.2.0b2.tar.gz" \
  "sbom/greenlang-full-0.2.0.spdx.json" \
  "sbom/greenlang-dist-0.2.0.spdx.json" \
  "sbom/greenlang-runner-0.2.0.spdx.json"
```

## 📋 Post-Release Checklist

After creating the release:

### 1. Verify Release
```bash
# View the created release
gh release view v0.2.0b2

# Check release URL
gh repo view --web
# Navigate to: Releases > v0.2.0b2
```

### 2. Test Installation
```bash
# Test installation from GitHub release
gh release download v0.2.0b2 --pattern "*.whl"
pip install greenlang-0.2.0b2-py3-none-any.whl

# Verify installation
gl --version
```

### 3. Update Documentation
- [ ] Update main README.md with new version
- [ ] Update installation instructions
- [ ] Announce on community channels

### 4. TestPyPI Upload (if not already done)
```bash
# Upload to TestPyPI for testing
python -m twine upload --repository testpypi dist/greenlang-0.2.0b2*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ greenlang==0.2.0b2
```

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### 1. Tag Already Exists
```bash
# If you need to recreate the tag
git tag -d v0.2.0b2                    # Delete local tag
git push origin :refs/tags/v0.2.0b2    # Delete remote tag
```

#### 2. GitHub CLI Authentication Issues
```bash
# Re-authenticate
gh auth logout
gh auth login
```

#### 3. Missing Artifacts
```bash
# Rebuild distribution files
python -m build

# Regenerate SBOM files (if available)
# Run your SBOM generation script
```

#### 4. Release Creation Fails
```bash
# Delete the release if it was partially created
gh release delete v0.2.0b2

# Try again with the create command
```

## 📞 Support

If you encounter issues:

1. **Check logs**: Review the output of the release commands
2. **Verify prerequisites**: Ensure all requirements are met
3. **GitHub Status**: Check https://www.githubstatus.com/
4. **Community Support**: Reach out on Discord or GitHub Discussions

## 🔐 Security Notes

- All artifacts include SBOM files for transparency
- Consider signing artifacts with cosign (if implemented)
- Verify checksums before and after upload
- Use secure authentication methods for GitHub CLI

---

**Ready to release GreenLang v0.2.0b2!**

Choose your preferred method above and execute the release when ready.