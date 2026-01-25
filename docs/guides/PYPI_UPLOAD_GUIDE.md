# 📦 Step-by-Step Guide: Publishing GreenLang v0.2.0 to PyPI

## Prerequisites

### 1. Install Twine (if not already installed)
```bash
pip install --upgrade twine
```

### 2. Create PyPI Account (if you don't have one)
1. Go to https://pypi.org/account/register/
2. Create your account
3. Verify your email

### 3. Set Up API Token (Recommended over password)
1. Log in to https://pypi.org/
2. Go to Account Settings → API tokens
3. Click "Add API token"
4. Name: "greenlang-upload" (or any name)
5. Scope: "Entire account" or "Project: greenlang" (if project exists)
6. Copy the token (starts with `pypi-`)
7. **SAVE THIS TOKEN SECURELY** - you won't see it again!

## Step-by-Step Upload Process

### Step 1: Verify Your Build Artifacts
```bash
# Check that files exist and are correct version
ls -la dist/
# Should show:
# greenlang-0.2.0-py3-none-any.whl
# greenlang-0.2.0.tar.gz

# Verify the packages are valid
twine check dist/greenlang-0.2.0*
# Should output: "PASSED" for both files
```

### Step 2: Configure Twine Authentication

#### Option A: Using .pypirc file (Permanent)
Create/edit `~/.pypirc` (or `%USERPROFILE%\.pypirc` on Windows):

```ini
[pypi]
username = __token__
password = pypi-YOUR-TOKEN-HERE
```

#### Option B: Using Environment Variables (Temporary)
```bash
# On Windows (PowerShell)
$env:TWINE_USERNAME="__token__"
$env:TWINE_PASSWORD="pypi-YOUR-TOKEN-HERE"

# On Windows (Command Prompt)
set TWINE_USERNAME=__token__
set TWINE_PASSWORD=pypi-YOUR-TOKEN-HERE

# On Linux/Mac
export TWINE_USERNAME="__token__"
export TWINE_PASSWORD="pypi-YOUR-TOKEN-HERE"
```

#### Option C: Enter Interactively (Most Secure)
Just run the upload command and enter when prompted:
- Username: `__token__`
- Password: `pypi-YOUR-TOKEN-HERE`

### Step 3: Upload to PyPI (PRODUCTION)

```bash
# Upload both wheel and source distribution
twine upload dist/greenlang-0.2.0*

# Or if you want to be explicit:
twine upload dist/greenlang-0.2.0-py3-none-any.whl dist/greenlang-0.2.0.tar.gz
```

Expected output:
```
Uploading distributions to https://upload.pypi.org/legacy/
Uploading greenlang-0.2.0-py3-none-any.whl
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 549.6/549.6 kB
Uploading greenlang-0.2.0.tar.gz
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 580.1/580.1 kB

View at:
https://pypi.org/project/greenlang/0.2.0/
```

### Step 4: Verify the Upload
```bash
# Wait 1-2 minutes for PyPI to process

# Test installation from PyPI
pip install greenlang==0.2.0

# Verify it works
python -c "import greenlang; print(greenlang.__version__)"
# Should output: 0.2.0

# Test the CLI
gl --version
# Should show: GreenLang v0.2.0
```

### Step 5: Create GitHub Final Release

```bash
# 1. Create and push the final tag
git tag v0.2.0 -m "Release v0.2.0 - Infra Seed (Final)"
git push origin v0.2.0

# 2. Create release notes file
# Create RELEASE_NOTES_v0.2.0.md with final release notes

# 3. Create GitHub release using gh CLI
gh release create v0.2.0 \
  --title "v0.2.0 - Infra Seed (Production Release)" \
  --notes-file "RELEASE_NOTES_v0.2.0.md" \
  "dist/greenlang-0.2.0-py3-none-any.whl" \
  "dist/greenlang-0.2.0.tar.gz"
```

### Step 6: Announce General Availability

Create announcement for:
1. GitHub README update
2. Discord/Slack announcement
3. Twitter/Social media
4. Email to beta testers

Example announcement:
```markdown
🎉 GreenLang v0.2.0 is now available on PyPI!

Install: pip install greenlang
With analytics: pip install greenlang[analytics]

Features:
✅ Climate Intelligence Framework
✅ Cross-platform CLI
✅ Pack management system
✅ Default-deny security policies
✅ Python SDK

Docs: https://github.com/greenlang/greenlang
PyPI: https://pypi.org/project/greenlang/
```

## Troubleshooting

### If Upload Fails

1. **403 Forbidden**:
   - Token might be wrong or expired
   - Project name might already be taken
   - You might not have permissions

2. **File already exists**:
   - Version 0.2.0 might already be uploaded
   - You cannot overwrite PyPI releases
   - Bump to 0.2.1 if needed

3. **Invalid distribution**:
   - Run `twine check dist/*` to verify
   - Rebuild if necessary: `python -m build`

### To Delete/Yank a Release (if needed)
1. Log in to https://pypi.org/
2. Go to your project
3. Click on the version
4. Click "Options" → "Yank"
(Note: Yanked versions can still be installed explicitly)

## Security Notes

⚠️ **NEVER**:
- Commit your PyPI token to git
- Share your token publicly
- Use your PyPI password directly (use tokens instead)

✅ **ALWAYS**:
- Use API tokens instead of passwords
- Store tokens securely (password manager)
- Rotate tokens periodically
- Use 2FA on your PyPI account

## Quick Command Summary

```bash
# 1. Check files
twine check dist/greenlang-0.2.0*

# 2. Upload to PyPI
twine upload dist/greenlang-0.2.0*

# 3. Verify
pip install greenlang==0.2.0
gl --version

# 4. Create GitHub release
git tag v0.2.0 -m "Release v0.2.0"
git push origin v0.2.0
gh release create v0.2.0 --title "v0.2.0" --notes-file "RELEASE_NOTES.md" dist/*
```

## Success Checklist

- [ ] Twine installed
- [ ] PyPI account created
- [ ] API token generated and saved
- [ ] Files verified with `twine check`
- [ ] Uploaded to PyPI with `twine upload`
- [ ] Package installable with `pip install greenlang`
- [ ] CLI works with `gl --version`
- [ ] GitHub release created
- [ ] Announcement sent

Once all items are checked, GreenLang v0.2.0 is officially live! 🚀