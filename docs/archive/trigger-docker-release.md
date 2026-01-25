# Manually Trigger Docker Release Workflow

Since the automatic tag trigger isn't working, you can manually trigger the workflow:

## Option 1: Via GitHub Web UI

1. Go to: https://github.com/akshay-greenlang/Code-V1_GreenLang/actions/workflows/release-docker.yml
2. Click the "Run workflow" button
3. Enter version: `0.2.0`
4. Click "Run workflow"

## Option 2: Via GitHub CLI (if installed)

```bash
gh workflow run release-docker.yml -f version=0.2.0
```

## Option 3: Via API (using curl)

First, create a personal access token:
1. Go to: https://github.com/settings/tokens
2. Generate new token with `repo` and `workflow` scopes

Then run:
```bash
curl -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  https://api.github.com/repos/akshay-greenlang/Code-V1_GreenLang/actions/workflows/release-docker.yml/dispatches \
  -d '{"ref":"master","inputs":{"version":"0.2.0"}}'
```

## What the Workflow Will Do

Once triggered, the workflow will:

1. ✅ Build multi-architecture images (linux/amd64, linux/arm64)
2. ✅ Push to GitHub Container Registry (ghcr.io)
3. ✅ Generate and attach SBOM
4. ✅ Sign images with Cosign (keyless OIDC)
5. ✅ Scan for vulnerabilities with Trivy
6. ✅ Create attestations

## After Workflow Completes

The images will be available at:
- `ghcr.io/akshay-greenlang/greenlang-runner:0.2.0`
- `ghcr.io/akshay-greenlang/greenlang-runner:latest`
- `ghcr.io/akshay-greenlang/greenlang-full:0.2.0`
- `ghcr.io/akshay-greenlang/greenlang-full:latest`

## Verification

After the workflow completes successfully, run:
```bash
scripts\verify-docker-dod.bat akshay-greenlang 0.2.0
```

This will verify all Definition of Done requirements are met.