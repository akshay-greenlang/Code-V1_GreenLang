# Manual Steps to Trigger Docker Release Workflow

Since GitHub CLI is installed but not in PATH, follow these steps:

## Option 1: Fix GitHub CLI PATH (Recommended)

1. **Find where gh.exe is installed:**
   - Check: `C:\Program Files\GitHub CLI\`
   - Or: `C:\Program Files (x86)\GitHub CLI\`
   - Or: `%LOCALAPPDATA%\Programs\GitHub CLI\`

2. **Add to PATH:**
   ```cmd
   setx PATH "%PATH%;C:\Program Files\GitHub CLI"
   ```
   - Close and reopen Command Prompt

3. **Then run:**
   ```cmd
   gh auth login
   gh workflow run release-docker.yml -f version=0.2.0
   ```

## Option 2: Use Full Path to gh.exe

1. **Find gh.exe location first:**
   ```cmd
   dir /s /b C:\gh.exe 2>nul
   dir /s /b "C:\Program Files\gh.exe" 2>nul
   ```

2. **Run with full path (example):**
   ```cmd
   "C:\Program Files\GitHub CLI\gh.exe" auth login
   "C:\Program Files\GitHub CLI\gh.exe" workflow run release-docker.yml -f version=0.2.0
   ```

## Option 3: Web Browser Method (Easiest!)

1. **Go to:** https://github.com/akshay-greenlang/Code-V1_GreenLang/actions/workflows/release-docker.yml

2. **Click** the green "Run workflow" button (top right)

3. **Fill in:**
   - Use workflow from: `Branch: master`
   - Version to release: `0.2.0`

4. **Click** "Run workflow" (green button)

5. **Wait** for workflow to complete (10-15 minutes)

## After Workflow Completes

Run verification:
```cmd
cd "C:\Users\rshar\Desktop\Akshay Makar\Tools\GreenLang\Code V1_GreenLang"
scripts\verify-docker-dod.bat akshay-greenlang 0.2.0
```

## Expected Results

When successful, the workflow will:
- ✅ Build multi-arch images (amd64 + arm64)
- ✅ Push to ghcr.io/akshay-greenlang/
- ✅ Sign with Cosign
- ✅ Generate SBOM
- ✅ Scan with Trivy
- ✅ Create tags: 0.2.0, 0.2, latest

## Monitor Progress

Watch the workflow at:
https://github.com/akshay-greenlang/Code-V1_GreenLang/actions

Green checkmark = Success!
Red X = Check logs for errors