# Step-by-Step Guide: Trigger Docker Release & Verify DoD

## Option A: Using GitHub CLI (Recommended)

### Step 1: Install GitHub CLI
1. Open your browser and go to: https://github.com/cli/cli/releases/latest
2. Download `gh_2.*.0_windows_amd64.msi` (or latest version)
3. Run the installer
4. Restart your command prompt

### Step 2: Authenticate with GitHub
Open cmd and run:
```cmd
gh auth login
```
- Choose: GitHub.com
- Choose: HTTPS
- Choose: Login with web browser
- Copy the code shown and press Enter
- Your browser will open - paste the code
- Authorize the CLI

### Step 3: Trigger the Workflow
```cmd
cd "C:\Users\rshar\Desktop\Akshay Makar\Tools\GreenLang\Code V1_GreenLang"
gh workflow run release-docker.yml -f version=0.2.0
```

### Step 4: Monitor Progress
```cmd
REM Check workflow status
gh run list --workflow=release-docker.yml --limit=1

REM Watch the workflow (updates every 3 seconds)
gh run watch
```

### Step 5: Verify DoD (After Completion)
```cmd
scripts\verify-docker-dod.bat akshay-greenlang 0.2.0
```

---

## Option B: Using Personal Access Token

### Step 1: Create GitHub Token
1. Go to: https://github.com/settings/tokens/new
2. Give it a name: "GreenLang Docker Release"
3. Select scopes:
   - `repo` (Full control of private repositories)
   - `workflow` (Update GitHub Action workflows)
4. Click "Generate token"
5. **COPY THE TOKEN NOW** (you won't see it again!)

### Step 2: Run the Script
```cmd
cd "C:\Users\rshar\Desktop\Akshay Makar\Tools\GreenLang\Code V1_GreenLang"
trigger-workflow-curl.bat
```
When prompted, paste your token and press Enter.

### Step 3: Monitor in Browser
Go to: https://github.com/akshay-greenlang/Code-V1_GreenLang/actions

Look for "Release Docker Images" workflow with status "In progress"

### Step 4: Verify DoD (After Completion)
Wait for the workflow to complete (green checkmark), then run:
```cmd
scripts\verify-docker-dod.bat akshay-greenlang 0.2.0
```

---

## Option C: Quick Script (if gh is installed)

### Just run this single command:
```cmd
cd "C:\Users\rshar\Desktop\Akshay Makar\Tools\GreenLang\Code V1_GreenLang"
trigger-workflow.bat
```

This script will:
1. Check if gh is installed
2. Authenticate if needed
3. Trigger the workflow
4. Show you next steps

---

## Expected Timeline

1. **Workflow trigger**: Immediate
2. **Build time**: 10-15 minutes
   - Multi-arch build (amd64 + arm64)
   - Pushing to registry
   - Signing and SBOM generation
3. **Verification**: 2 minutes

## Success Criteria (DoD)

When you run `scripts\verify-docker-dod.bat akshay-greenlang 0.2.0`, you should see:

✅ PASS for:
- Runner image found on GHCR
- linux/amd64 in manifest
- linux/arm64 in manifest
- Non-root user (UID: 10001)
- Healthcheck configured
- gl version works
- Version label present
- Source label present
- License label present

## Troubleshooting

### If workflow doesn't start:
- Check: https://github.com/akshay-greenlang/Code-V1_GreenLang/actions
- Look for any failed workflows
- Check the logs for errors

### If verification fails:
- Wait a few more minutes (registry propagation)
- Check if images are visible at: https://github.com/akshay-greenlang/Code-V1_GreenLang/pkgs/container/greenlang-runner

### Common Issues:
1. **"gh: command not found"** - Install GitHub CLI first
2. **"Bad credentials"** - Re-authenticate: `gh auth login`
3. **"Workflow not found"** - Ensure you're in the right directory
4. **Docker not installed** - Install Docker Desktop for verification script

## What Happens in the Workflow

1. **Builds** multi-architecture Docker images
2. **Pushes** to ghcr.io/akshay-greenlang/
3. **Signs** with Cosign (keyless OIDC)
4. **Generates** SBOM with Syft
5. **Scans** with Trivy for vulnerabilities
6. **Creates** attestations for supply chain security

## Final Images Available

After successful completion:
- `ghcr.io/akshay-greenlang/greenlang-runner:0.2.0`
- `ghcr.io/akshay-greenlang/greenlang-runner:latest`
- `ghcr.io/akshay-greenlang/greenlang-full:0.2.0`
- `ghcr.io/akshay-greenlang/greenlang-full:latest`

Each supports both `linux/amd64` and `linux/arm64` architectures!