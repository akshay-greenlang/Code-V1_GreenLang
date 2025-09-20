# PowerShell script to trigger Docker release
Write-Host "Checking GitHub CLI installation..." -ForegroundColor Cyan
$ghPath = Get-Command gh -ErrorAction SilentlyContinue

if (-not $ghPath) {
    Write-Host "GitHub CLI not found in PATH!" -ForegroundColor Red
    Write-Host "Please ensure gh.exe is installed and in PATH" -ForegroundColor Yellow
    exit 1
}

Write-Host "GitHub CLI found at: $($ghPath.Path)" -ForegroundColor Green

Write-Host "`nChecking authentication..." -ForegroundColor Cyan
gh auth status

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nNeed to authenticate. Running gh auth login..." -ForegroundColor Yellow
    gh auth login
}

Write-Host "`nTriggering Docker Release Workflow for v0.2.0..." -ForegroundColor Cyan
gh workflow run release-docker.yml -f version=0.2.0

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nWorkflow triggered successfully!" -ForegroundColor Green
} else {
    Write-Host "Retrying with full path..." -ForegroundColor Yellow
    gh workflow run .github/workflows/release-docker.yml -f version=0.2.0
}

Write-Host "`nWaiting 5 seconds for workflow to start..." -ForegroundColor Cyan
Start-Sleep -Seconds 5

Write-Host "`nChecking latest workflow run..." -ForegroundColor Cyan
gh run list --workflow=release-docker.yml --limit=1

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "Workflow has been triggered!" -ForegroundColor Green
Write-Host "Monitor progress at:" -ForegroundColor Green
Write-Host "https://github.com/akshay-greenlang/Code-V1_GreenLang/actions" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Green