# GreenLang Docker Fix and Publish Script
# This script will trigger the fixed Docker workflows

param(
    [Parameter(Mandatory=$false)]
    [string]$Version = "0.2.0",

    [Parameter(Mandatory=$false)]
    [switch]$ForceRebuild = $false
)

Write-Host "🐳 GreenLang Docker Fix and Publish" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "📦 Version: $Version" -ForegroundColor Green
Write-Host "🔄 Force Rebuild: $ForceRebuild" -ForegroundColor Green
Write-Host ""

# Check if gh CLI is installed
try {
    $ghVersion = gh --version
    Write-Host "✅ GitHub CLI detected: $($ghVersion[0])" -ForegroundColor Green
} catch {
    Write-Host "❌ GitHub CLI not found. Please install it first:" -ForegroundColor Red
    Write-Host "   https://cli.github.com/" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "🚀 Triggering Docker workflows..." -ForegroundColor Yellow

# Trigger the new public Docker workflow
Write-Host "1. Triggering docker-publish-public workflow..." -ForegroundColor Blue
try {
    $result1 = gh workflow run docker-publish-public.yml -f version=$Version -f force_rebuild=$ForceRebuild
    Write-Host "   ✅ docker-publish-public workflow triggered" -ForegroundColor Green
} catch {
    Write-Host "   ❌ Failed to trigger docker-publish-public workflow" -ForegroundColor Red
    Write-Host "   Error: $_" -ForegroundColor Red
}

# Trigger the docker-complete-dod workflow as backup
Write-Host "2. Triggering docker-complete-dod workflow..." -ForegroundColor Blue
try {
    $result2 = gh workflow run docker-complete-dod.yml -f version=$Version
    Write-Host "   ✅ docker-complete-dod workflow triggered" -ForegroundColor Green
} catch {
    Write-Host "   ❌ Failed to trigger docker-complete-dod workflow" -ForegroundColor Red
    Write-Host "   Error: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "🔍 Checking workflow status..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

try {
    Write-Host "Recent workflow runs:" -ForegroundColor Blue
    gh run list --limit 5
} catch {
    Write-Host "❌ Failed to get workflow status" -ForegroundColor Red
}

Write-Host ""
Write-Host "📋 Next steps:" -ForegroundColor Cyan
Write-Host "1. Monitor the workflows at: https://github.com/akshay-greenlang/Code-V1_GreenLang/actions" -ForegroundColor White
Write-Host "2. Once complete, test the images:" -ForegroundColor White
Write-Host "   docker pull ghcr.io/akshay-greenlang/greenlang-runner:$Version" -ForegroundColor Gray
Write-Host "   docker pull ghcr.io/akshay-greenlang/greenlang-full:$Version" -ForegroundColor Gray
Write-Host "3. Verify public access (should work without login)" -ForegroundColor White
Write-Host ""
Write-Host "🎉 Docker fix workflows initiated!" -ForegroundColor Green