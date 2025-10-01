# GreenLang CLI - Optimized Installation Guide

## Quick Start

### Windows
```bash
# One-time setup (downloads wheels for offline use)
scripts\setup-cache.bat

# Quick install in any new environment
scripts\quick-install.bat
```

### Linux/Mac
```bash
# One-time setup (downloads wheels for offline use)
chmod +x scripts/*.sh
scripts/setup-cache.sh

# Quick install in any new environment
scripts/quick-install.sh
```

## Problem Solved
When installing `greenlang-cli` in new virtual environments, pip normally downloads all 35+ dependencies every time (600MB+), which is slow and bandwidth-consuming. Our solution caches everything locally.

## Installation Methods

### 1. Use Requirements Lock File (Fastest Online)
We've created `requirements-lock.txt` with pinned versions:
```bash
pip install -r requirements-lock.txt
```

### 2. Create Local Wheel Cache (Offline Installation)
Run once to download all wheels:
```bash
setup-pip-cache.bat
```

Then for future installations:
```bash
pip install --no-index --find-links wheels greenlang-cli
```

### 3. Quick Install Script
Use the provided script that automatically chooses the fastest method:
```bash
quick-install.bat
```

### 4. Use Pip's Built-in Cache
Pip caches downloaded packages by default. Check your cache:
```bash
pip cache dir
```

To see cache info:
```bash
pip cache info
```

### 5. Set Custom Cache Directory
```bash
set PIP_CACHE_DIR=C:\your\cache\directory
pip install greenlang-cli
```

### 6. Pre-download for Multiple Environments
Download once, use everywhere:
```bash
# Download wheels without installing
pip download greenlang-cli -d wheels

# Install from downloaded wheels
pip install --no-index --find-links wheels greenlang-cli
```

## Benefits
- **Faster installations**: No need to re-download packages
- **Offline capability**: Install without internet connection
- **Bandwidth saving**: Download once, use many times
- **Version consistency**: Lock file ensures same versions across environments

## File Structure
```
Code V1_GreenLang/
├── requirements-lock.txt    # Pinned dependency versions
├── setup-pip-cache.bat      # Setup script for wheel cache
├── quick-install.bat        # Automated installation script
└── wheels/                  # Directory with cached wheels (after setup)
```