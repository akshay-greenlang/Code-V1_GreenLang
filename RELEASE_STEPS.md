# Complete Release Steps for Week 0 - v0.3.0

## Prerequisites Check

### 1. Install Required Tools (if not already installed)

#### Option A: Install Cosign (for artifact signing)
```powershell
# For Windows using Scoop
scoop install cosign

# Or download directly from GitHub
# Visit: https://github.com/sigstore/cosign/releases
# Download: cosign-windows-amd64.exe
# Rename to: cosign.exe
# Add to PATH
```

#### Option B: Use Docker for signing (alternative)
```bash
docker run --rm -v ${PWD}:/workspace gcr.io/projectsigstore/cosign:latest version
```

### 2. Verify Git Configuration
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Step-by-Step Release Process

### ✅ Step 1: Commit Changes (COMPLETED)
```bash
git add -A
git commit -m "feat: Complete Week 0 requirements with v0.3.0 baseline"
```
**Status**: ✅ Completed - Commit hash: 2102be5

### ✅ Step 2: Create Release Tag (COMPLETED)
```bash
git tag -a v0.3.0-rc.0 -m "Q4'25 Week 0 Release Candidate"
```
**Status**: ✅ Completed - Tag created

### ✅ Step 3: Build Distribution (COMPLETED)
```bash
python -m build
```
**Status**: ✅ Completed - Files created:
- `dist/greenlang_cli-0.3.0-py3-none-any.whl`
- `dist/greenlang_cli-0.3.0.tar.gz`

### 🔄 Step 4: Sign Artifacts (PENDING)

#### Option A: Manual Signing with Cosign
```bash
# Sign the wheel file
cosign sign-blob --yes dist/greenlang_cli-0.3.0-py3-none-any.whl \
  --output-signature dist/greenlang_cli-0.3.0-py3-none-any.whl.sig

# Sign the source distribution
cosign sign-blob --yes dist/greenlang_cli-0.3.0.tar.gz \
  --output-signature dist/greenlang_cli-0.3.0.tar.gz.sig

# Verify signatures
cosign verify-blob dist/greenlang_cli-0.3.0-py3-none-any.whl \
  --signature dist/greenlang_cli-0.3.0-py3-none-any.whl.sig \
  --certificate-identity-regexp ".*" \
  --certificate-oidc-issuer-regexp ".*"
```

#### Option B: Use the Signing Script
```bash
# On Windows, use Git Bash or WSL
bash sign_artifacts.sh

# Or on Windows CMD/PowerShell
wsl bash sign_artifacts.sh
```

#### Option C: GitHub Actions (Automated)
The artifacts will be automatically signed when you push to GitHub if the workflow is configured.

### 🔄 Step 5: Push to GitHub

#### Push commits and tags
```bash
# Push the commits
git push origin master

# Push the tag
git push origin v0.3.0-rc.0

# Or push everything at once
git push origin master --tags
```

### 🔄 Step 6: Create GitHub Release

#### Using GitHub CLI (gh)
```bash
# Install GitHub CLI if needed
# Windows: winget install GitHub.cli

# Create release with artifacts
gh release create v0.3.0-rc.0 \
  --title "v0.3.0-rc.0: Q4'25 Week 0 Complete" \
  --notes "## Q4'25 Week 0 Release Candidate

### ✅ All Requirements Met
- Version normalized to v0.3.0
- Python >=3.10 enforced everywhere
- Sandbox capability gating implemented
- Supply chain security ready
- All DoD requirements satisfied

### 📦 Artifacts
- greenlang_cli-0.3.0-py3-none-any.whl
- greenlang_cli-0.3.0.tar.gz

### 🔏 Security
- Artifacts signed with cosign
- SBOM included
- Provenance documented" \
  dist/greenlang_cli-0.3.0-py3-none-any.whl \
  dist/greenlang_cli-0.3.0.tar.gz \
  provenance.txt

# If you have signatures, add them too
gh release upload v0.3.0-rc.0 \
  dist/greenlang_cli-0.3.0-py3-none-any.whl.sig \
  dist/greenlang_cli-0.3.0.tar.gz.sig
```

#### Using GitHub Web Interface
1. Go to: https://github.com/akshay-greenlang/Code-V1_GreenLang/releases
2. Click "Draft a new release"
3. Choose tag: `v0.3.0-rc.0`
4. Title: "v0.3.0-rc.0: Q4'25 Week 0 Complete"
5. Upload artifacts:
   - `dist/greenlang_cli-0.3.0-py3-none-any.whl`
   - `dist/greenlang_cli-0.3.0.tar.gz`
   - `provenance.txt`
   - Signature files (if created)
6. Mark as "Pre-release"
7. Click "Publish release"

### 🔄 Step 7: Verify CI/CD Status

#### Check GitHub Actions
1. Go to: https://github.com/akshay-greenlang/Code-V1_GreenLang/actions
2. Look for the workflow runs triggered by the tag push
3. Verify all checks pass (especially the matrix builds)

#### Expected CI Results
- ✅ Ubuntu (Python 3.10, 3.11, 3.12)
- ✅ macOS (Python 3.10, 3.11, 3.12)
- ✅ Windows (Python 3.10, 3.11, 3.12)
- ✅ Security scans
- ✅ Artifact signing (if configured in CI)

### 🔄 Step 8: Test Installation from Release

```bash
# Create a test environment
python -m venv test-release
test-release\Scripts\activate  # On Windows
# source test-release/bin/activate  # On Unix/Mac

# Install from the built wheel
pip install dist/greenlang_cli-0.3.0-py3-none-any.whl

# Verify installation
gl --version  # Should show 0.3.0
gl doctor     # Should show all checks passing

# Test basic functionality
gl pack list
gl init pack-basic test-pack
```

## Verification Checklist

### Pre-Push Verification
- [x] All version files show 0.3.0
- [x] Python >=3.10 in all config files
- [x] setup.py exists and works
- [x] Sandbox implementation complete
- [x] Provenance documentation created
- [x] Signing script ready

### Post-Push Verification
- [ ] Git tag v0.3.0-rc.0 visible on GitHub
- [ ] CI/CD runs triggered
- [ ] All CI checks passing
- [ ] GitHub release created
- [ ] Artifacts attached to release
- [ ] Installation test successful

## Troubleshooting

### If CI Fails
1. Check the error logs in GitHub Actions
2. Common issues:
   - Import errors: Check relative imports
   - Test failures: May need to update test expectations for v0.3.0
   - Linting: Run `flake8` locally to fix issues

### If Signing Fails
1. Cosign not installed: Install using the methods above
2. No OIDC token: Run in GitHub Actions or use `cosign sign-blob --key`
3. Network issues: Check firewall/proxy settings

### If Push Fails
```bash
# If you get "rejected" errors
git pull --rebase origin master
git push origin master

# If tag already exists
git tag -d v0.3.0-rc.0  # Delete local
git push --delete origin v0.3.0-rc.0  # Delete remote
git tag -a v0.3.0-rc.0 -m "..."  # Recreate
git push origin v0.3.0-rc.0  # Push again
```

## Final Steps for CTO

After completing all steps above:

1. **Send Completion Report**:
   - Link to GitHub release
   - Link to passing CI runs
   - Summary of all changes

2. **Request Sign-off**:
   - Confirm v0.3.0 as Q4'25 baseline
   - Get approval for PyPI publication

3. **Next Milestone**:
   - Plan Week 1 tasks
   - Schedule team sync

## Summary Commands (Quick Reference)

```bash
# Complete release in one go (after commit)
git tag -a v0.3.0-rc.0 -m "Q4'25 Week 0 Release Candidate"
git push origin master --tags
python -m build

# Create GitHub release (with gh CLI)
gh release create v0.3.0-rc.0 --title "v0.3.0-rc.0: Q4'25 Week 0 Complete" \
  --notes-file WEEK_0_COMPLETION.md \
  dist/*.whl dist/*.tar.gz provenance.txt

# Verify
gl --version  # Should show 0.3.0
```

---

**Status**: Ready for release. Follow the steps above to complete the process.