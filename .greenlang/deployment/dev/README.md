# GreenLang-First Development Environment

Complete guide for setting up your local development environment with GreenLang-First enforcement.

## Quick Start

```bash
# Clone repository
git clone https://github.com/greenlang/greenlang.git
cd greenlang

# Run setup script
chmod +x .greenlang/deployment/dev/setup.sh
./.greenlang/deployment/dev/setup.sh

# Verify installation
greenlang --version
pre-commit run --all-files
```

## Prerequisites

### Required
- **Python 3.10+** - Core runtime
- **Node.js 18+** - Frontend and tooling
- **Git 2.30+** - Version control

### Optional but Recommended
- **Docker 20+** - For OPA policy server
- **VS Code** - Recommended IDE with extensions

## Installation Methods

### Method 1: Automated Setup (Recommended)

The setup script handles everything:

```bash
./.greenlang/deployment/dev/setup.sh
```

**What it does:**
- Detects your OS (Windows/Mac/Linux)
- Checks prerequisites
- Installs pre-commit hooks
- Installs linters and code quality tools
- Configures IDE integrations
- Sets up OPA policy engine
- Installs GreenLang CLI tools
- Verifies installation

### Method 2: Manual Setup

If the automated setup fails, follow these steps:

#### 1. Install Pre-commit Hooks

```bash
pip3 install pre-commit
cd /path/to/greenlang
pre-commit install
pre-commit install --hook-type commit-msg
pre-commit install --hook-type pre-push
```

#### 2. Install Linters

**Python:**
```bash
pip3 install pylint flake8 black isort mypy bandit safety
```

**Node.js:**
```bash
npm install -g eslint prettier typescript tslint
```

**Infrastructure:**
```bash
# macOS
brew install terraform tflint shellcheck yamllint hadolint

# Linux
sudo apt install shellcheck yamllint
pip3 install yamllint ansible-lint

# Windows
choco install terraform shellcheck
pip3 install yamllint
```

#### 3. Install OPA

**macOS:**
```bash
brew install opa
```

**Linux:**
```bash
curl -L -o opa https://openpolicyagent.org/downloads/latest/opa_linux_amd64
chmod +x opa
sudo mv opa /usr/local/bin/
```

**Windows:**
```powershell
# Download from https://www.openpolicyagent.org/docs/latest/#running-opa
# Add to PATH
```

#### 4. Install GreenLang CLI

```bash
cd .greenlang/cli
pip3 install -e .
```

## Configuration

### Development Settings

Edit `.greenlang/deployment/config/dev.yaml`:

```yaml
enforcement:
  ium_threshold: 80  # Lower for dev
  require_adr: false  # More lenient
  block_on_violation: false  # Warn only

linting:
  enabled: true
  auto_fix: true

opa:
  server: "http://localhost:8181"
  policies_path: ".greenlang/enforcement/opa-policies"

monitoring:
  enabled: false  # No telemetry in dev
```

### IDE Configuration

#### VS Code

Install recommended extensions:
```bash
code --install-extension ms-python.python
code --install-extension dbaeumer.vscode-eslint
code --install-extension tsandall.opa
code --install-extension greenlang.greenlang-first
```

Settings are auto-configured in `.vscode/settings.json`.

#### PyCharm/IntelliJ

1. Enable external tools: Pre-commit, Pylint, ESLint
2. Configure file watchers for auto-formatting
3. Install GreenLang plugin from marketplace

#### Vim/Neovim

Add to `.vimrc`:
```vim
" GreenLang-First integration
autocmd BufWritePre * :silent !greenlang lint --fix %
```

### Git Hooks Configuration

Customize `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/greenlang/greenlang-hooks
    rev: v1.0.0
    hooks:
      - id: greenlang-lint
        args: ['--fix']
      - id: ium-check
        args: ['--threshold=80']
      - id: adr-check
        stages: [commit-msg]
```

## Usage

### Daily Workflow

```bash
# Start working on a feature
git checkout -b feature/my-feature

# Check current IUM score
greenlang ium calculate

# Make changes...
vim myfile.py

# Hooks run automatically on commit
git add myfile.py
git commit -m "feat: Add new feature"  # Hooks validate

# If hooks fail
greenlang lint --fix .
git add -u
git commit -m "feat: Add new feature"

# Push (pre-push hooks run)
git push origin feature/my-feature
```

### CLI Commands

```bash
# Check IUM score
greenlang ium calculate
greenlang ium calculate --path ./src/
greenlang ium calculate --verbose

# Lint code
greenlang lint .
greenlang lint --fix .
greenlang lint --file myfile.py

# Check ADRs
greenlang adr check
greenlang adr create --title "Use PostgreSQL for data storage"

# Run OPA policies
greenlang policy test
greenlang policy eval --input data.json

# Generate reports
greenlang report --format html
greenlang report --format json --output report.json

# Deployment commands
greenlang deploy validate --env dev
greenlang deploy status
```

### Testing Enforcement

```bash
# Test pre-commit hooks without committing
pre-commit run --all-files

# Test specific hook
pre-commit run greenlang-lint --all-files

# Test OPA policies
cd .greenlang/enforcement/opa-policies
opa test . -v

# Test with violations to see blocking
echo "TODO: fix this" > test.py
git add test.py
git commit -m "test"  # Should fail

# Override (only in dev)
git commit -m "test" --no-verify  # Bypasses hooks
```

## Troubleshooting

### Pre-commit Hooks Not Running

**Symptom:** Commits succeed without running hooks

**Solution:**
```bash
# Reinstall hooks
pre-commit uninstall
pre-commit install --install-hooks

# Verify
ls -la .git/hooks/
cat .git/hooks/pre-commit  # Should reference pre-commit
```

### OPA Server Not Starting

**Symptom:** `Connection refused` when running policies

**Solution:**
```bash
# Check if OPA is running
curl http://localhost:8181/health

# Start OPA manually
opa run --server --addr localhost:8181 .greenlang/enforcement/opa-policies &

# Check logs
tail -f ~/.greenlang/logs/opa.log

# Kill existing OPA processes
killall opa
```

### Python Import Errors

**Symptom:** `ModuleNotFoundError` when running CLI

**Solution:**
```bash
# Reinstall in editable mode
cd .greenlang/cli
pip3 install -e .

# Check Python path
python3 -c "import sys; print(sys.path)"

# Add to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/.greenlang
```

### Linter Conflicts

**Symptom:** Black and Flake8 disagree on formatting

**Solution:**
```bash
# Configure Flake8 to ignore Black's formatting
echo "[flake8]
max-line-length = 88
extend-ignore = E203, E501
" > setup.cfg

# Run Black first, then Flake8
black .
flake8 .
```

### Permission Denied on Windows

**Symptom:** `./setup.sh: Permission denied`

**Solution:**
```bash
# Use Git Bash or WSL
bash .greenlang/deployment/dev/setup.sh

# Or use PowerShell script (coming soon)
# .\setup.ps1
```

### Slow Pre-commit Hooks

**Symptom:** Commits take >30 seconds

**Solution:**
```bash
# Skip slow hooks in dev
SKIP=pylint,mypy git commit -m "message"

# Or configure to run only on changed files
# Edit .pre-commit-config.yaml:
# - id: pylint
#   files: \.py$
#   exclude: ^tests/
```

### Docker Issues on Windows

**Symptom:** OPA Docker container won't start

**Solution:**
```bash
# Use OPA binary instead
opa run --server .greenlang/enforcement/opa-policies

# Or use WSL2 for Docker
wsl --install
# Then run Docker in WSL2
```

## Environment Variables

Set these in your shell profile (`.bashrc`, `.zshrc`, etc.):

```bash
# Required
export GREENLANG_HOME="/path/to/greenlang/.greenlang"
export PATH="$PATH:$GREENLANG_HOME/cli/bin"

# Optional
export GREENLANG_ENV="dev"
export GREENLANG_IUM_THRESHOLD="80"
export GREENLANG_OPA_SERVER="http://localhost:8181"
export GREENLANG_LOG_LEVEL="INFO"
export GREENLANG_CACHE_DIR="$HOME/.cache/greenlang"
```

## Performance Optimization

### Speed Up Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
default_stages: [commit]  # Don't run on push in dev

repos:
  - repo: local
    hooks:
      - id: greenlang-quick
        name: GreenLang Quick Check
        entry: greenlang lint --quick
        language: system
        pass_filenames: true
```

### Reduce OPA Policy Load

```bash
# Only load necessary policies in dev
opa run --server \
  --addr localhost:8181 \
  .greenlang/enforcement/opa-policies/core \
  .greenlang/enforcement/opa-policies/security
```

### Cache Dependencies

```bash
# Cache pip packages
pip3 install --cache-dir ~/.cache/pip -r requirements.txt

# Cache npm packages
npm config set cache ~/.npm --global
```

## Development Best Practices

### 1. Run Checks Before Commit

```bash
# Pre-flight checklist
greenlang ium calculate  # Should be >80%
greenlang lint --fix .   # Auto-fix issues
pre-commit run --all-files  # Verify hooks pass
pytest  # Run tests
```

### 2. Keep IUM Score High

```bash
# Check IUM impact of changes
greenlang ium calculate --baseline main

# Target >80% in dev, >95% in prod
```

### 3. Document Decisions

```bash
# Create ADR for significant changes
greenlang adr create \
  --title "Use Redis for caching" \
  --status accepted \
  --consequences "Faster response times, need Redis server"
```

### 4. Update Dependencies Regularly

```bash
# Weekly dependency updates
pip3 install --upgrade -r requirements.txt
npm update

# Check for security issues
safety check
npm audit
```

## Getting Help

### Resources

- **Documentation:** `.greenlang/deployment/docs/`
- **Examples:** `.greenlang/examples/`
- **Runbooks:** `.greenlang/deployment/docs/runbooks/`

### Support Channels

- **GitHub Issues:** https://github.com/greenlang/greenlang/issues
- **Slack:** #greenlang-support
- **Email:** support@greenlang.io
- **Office Hours:** Tuesdays 2-3 PM EST

### Common Questions

**Q: Can I disable enforcement temporarily?**
A: In dev, yes: `git commit --no-verify`. In prod, no.

**Q: What's the minimum IUM score?**
A: Dev: 80%, Staging: 90%, Prod: 95%

**Q: How do I report false positives?**
A: Create issue with label `false-positive` and example code.

**Q: Can I customize enforcement rules?**
A: Yes, edit `.greenlang/deployment/config/dev.yaml` or create `.greenlangrc` in project root.

## Updates and Maintenance

```bash
# Update GreenLang tools
cd .greenlang/cli
git pull
pip3 install -e . --upgrade

# Update pre-commit hooks
pre-commit autoupdate

# Update OPA policies
cd .greenlang/enforcement/opa-policies
git pull
opa test . -v  # Verify

# Clear caches
rm -rf ~/.cache/greenlang
rm -rf ~/.cache/pip
```

## Next Steps

1. Complete setup: `./.greenlang/deployment/dev/setup.sh`
2. Verify installation: `greenlang --version`
3. Run first check: `greenlang ium calculate`
4. Read enforcement guide: `.greenlang/enforcement/README.md`
5. Review examples: `.greenlang/examples/`

---

**Version:** 1.0.0
**Last Updated:** 2025-11-09
**Maintainer:** DevOps Team
