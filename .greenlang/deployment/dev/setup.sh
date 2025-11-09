#!/bin/bash
# GreenLang-First Development Environment Setup Script
# Supports: Windows (Git Bash/WSL), macOS, Linux
# Version: 1.0.0

set -e

GREENLANG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GREENLANG_VERSION="1.0.0"
BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BOLD}===================================================${NC}"
echo -e "${BOLD}  GreenLang-First Development Environment Setup${NC}"
echo -e "${BOLD}  Version: ${GREENLANG_VERSION}${NC}"
echo -e "${BOLD}===================================================${NC}\n"

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        OS="windows"
    else
        echo -e "${RED}Unsupported OS: $OSTYPE${NC}"
        exit 1
    fi
    echo -e "${GREEN}Detected OS: $OS${NC}"
}

# Check prerequisites
check_prerequisites() {
    echo -e "\n${BOLD}Checking prerequisites...${NC}"

    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Python 3 is not installed. Please install Python 3.8+${NC}"
        exit 1
    fi
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    echo -e "${GREEN}Python ${PYTHON_VERSION} found${NC}"

    # Check Node.js
    if ! command -v node &> /dev/null; then
        echo -e "${YELLOW}Node.js not found. Installing Node.js 18...${NC}"
        if [[ "$OS" == "macos" ]]; then
            brew install node@18
        elif [[ "$OS" == "linux" ]]; then
            curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
            sudo apt-get install -y nodejs
        else
            echo -e "${RED}Please install Node.js 18+ manually from https://nodejs.org${NC}"
            exit 1
        fi
    fi
    NODE_VERSION=$(node --version)
    echo -e "${GREEN}Node.js ${NODE_VERSION} found${NC}"

    # Check Git
    if ! command -v git &> /dev/null; then
        echo -e "${RED}Git is not installed. Please install Git.${NC}"
        exit 1
    fi
    GIT_VERSION=$(git --version | awk '{print $3}')
    echo -e "${GREEN}Git ${GIT_VERSION} found${NC}"

    # Check Docker (optional but recommended)
    if command -v docker &> /dev/null; then
        DOCKER_VERSION=$(docker --version | awk '{print $3}' | sed 's/,//')
        echo -e "${GREEN}Docker ${DOCKER_VERSION} found${NC}"
    else
        echo -e "${YELLOW}Docker not found (optional). OPA policies will run in local mode.${NC}"
    fi
}

# Install pre-commit hooks
install_precommit_hooks() {
    echo -e "\n${BOLD}Installing pre-commit hooks...${NC}"

    # Install pre-commit framework
    pip3 install pre-commit --upgrade

    # Copy pre-commit config if not exists
    if [ ! -f "$GREENLANG_ROOT/../../.pre-commit-config.yaml" ]; then
        cp "$GREENLANG_ROOT/enforcement/pre-commit-config.yaml" "$GREENLANG_ROOT/../../.pre-commit-config.yaml"
    fi

    # Install hooks
    cd "$GREENLANG_ROOT/../.."
    pre-commit install
    pre-commit install --hook-type commit-msg
    pre-commit install --hook-type pre-push

    echo -e "${GREEN}Pre-commit hooks installed successfully${NC}"
}

# Install linters
install_linters() {
    echo -e "\n${BOLD}Installing linters and code quality tools...${NC}"

    # Python linters
    pip3 install --upgrade \
        pylint \
        flake8 \
        black \
        isort \
        mypy \
        bandit \
        safety

    # Node.js linters
    npm install -g \
        eslint \
        prettier \
        jshint \
        tslint \
        typescript

    # Infrastructure linters
    if [[ "$OS" == "macos" ]]; then
        brew install \
            terraform \
            tflint \
            shellcheck \
            yamllint \
            hadolint
    elif [[ "$OS" == "linux" ]]; then
        # Terraform
        wget -O- https://apt.releases.hashicorp.com/gpg | gpg --dearmor | sudo tee /usr/share/keyrings/hashicorp-archive-keyring.gpg
        echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
        sudo apt update && sudo apt install -y terraform

        # Other tools
        sudo apt install -y shellcheck yamllint

        # Hadolint
        wget -O hadolint https://github.com/hadolint/hadolint/releases/latest/download/hadolint-Linux-x86_64
        chmod +x hadolint
        sudo mv hadolint /usr/local/bin/
    else
        echo -e "${YELLOW}Windows detected. Please install linters manually:${NC}"
        echo "  - Terraform: https://www.terraform.io/downloads"
        echo "  - ShellCheck: https://www.shellcheck.net/"
        echo "  - yamllint: pip install yamllint"
        echo "  - hadolint: https://github.com/hadolint/hadolint"
    fi

    # Python infrastructure linters
    pip3 install yamllint ansible-lint

    echo -e "${GREEN}Linters installed successfully${NC}"
}

# Configure IDE integrations
configure_ide() {
    echo -e "\n${BOLD}Configuring IDE integrations...${NC}"

    # VSCode settings
    VSCODE_SETTINGS_DIR="$GREENLANG_ROOT/../../.vscode"
    mkdir -p "$VSCODE_SETTINGS_DIR"

    cat > "$VSCODE_SETTINGS_DIR/settings.json" <<EOF
{
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.mypyEnabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "eslint.enable": true,
  "prettier.enable": true,
  "files.associations": {
    "*.rego": "rego"
  },
  "greenlang.enforcement.enabled": true,
  "greenlang.iumThreshold": 90,
  "greenlang.requireADR": true
}
EOF

    # VSCode extensions recommendations
    cat > "$VSCODE_SETTINGS_DIR/extensions.json" <<EOF
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "dbaeumer.vscode-eslint",
    "esbenp.prettier-vscode",
    "hashicorp.terraform",
    "redhat.vscode-yaml",
    "tsandall.opa",
    "timonwong.shellcheck",
    "greenlang.greenlang-first"
  ]
}
EOF

    echo -e "${GREEN}IDE configuration complete${NC}"
    echo -e "${YELLOW}Recommended VSCode extensions saved to .vscode/extensions.json${NC}"
}

# Setup OPA policy engine
setup_opa() {
    echo -e "\n${BOLD}Setting up OPA policy engine...${NC}"

    # Install OPA
    if ! command -v opa &> /dev/null; then
        if [[ "$OS" == "macos" ]]; then
            brew install opa
        elif [[ "$OS" == "linux" ]]; then
            curl -L -o opa https://openpolicyagent.org/downloads/latest/opa_linux_amd64
            chmod +x opa
            sudo mv opa /usr/local/bin/
        else
            echo -e "${YELLOW}Download OPA from: https://www.openpolicyagent.org/docs/latest/#running-opa${NC}"
        fi
    fi

    if command -v opa &> /dev/null; then
        OPA_VERSION=$(opa version | head -1)
        echo -e "${GREEN}OPA ${OPA_VERSION} installed${NC}"

        # Test OPA policies
        echo -e "\n${BOLD}Testing OPA policies...${NC}"
        cd "$GREENLANG_ROOT/enforcement/opa-policies"
        opa test . -v

        # Start OPA server in background (development mode)
        if [[ "$OS" != "windows" ]]; then
            opa run --server --addr localhost:8181 "$GREENLANG_ROOT/enforcement/opa-policies" > /dev/null 2>&1 &
            OPA_PID=$!
            echo $OPA_PID > /tmp/opa.pid
            echo -e "${GREEN}OPA server started on http://localhost:8181 (PID: $OPA_PID)${NC}"
        fi
    else
        echo -e "${YELLOW}OPA not installed. Skipping policy tests.${NC}"
    fi
}

# Install CLI tools
install_cli_tools() {
    echo -e "\n${BOLD}Installing GreenLang CLI tools...${NC}"

    # Install greenlang-cli
    if [ -f "$GREENLANG_ROOT/cli/setup.py" ]; then
        cd "$GREENLANG_ROOT/cli"
        pip3 install -e .
        echo -e "${GREEN}GreenLang CLI installed${NC}"
    fi

    # Install deployment CLI
    if [ -f "$GREENLANG_ROOT/deployment/requirements.txt" ]; then
        pip3 install -r "$GREENLANG_ROOT/deployment/requirements.txt"
    fi

    # Add to PATH
    if [[ "$OS" != "windows" ]]; then
        PROFILE_FILE="$HOME/.bashrc"
        if [[ "$OS" == "macos" ]]; then
            PROFILE_FILE="$HOME/.zshrc"
        fi

        if ! grep -q "GREENLANG_HOME" "$PROFILE_FILE"; then
            echo "" >> "$PROFILE_FILE"
            echo "# GreenLang-First Environment" >> "$PROFILE_FILE"
            echo "export GREENLANG_HOME=\"$GREENLANG_ROOT\"" >> "$PROFILE_FILE"
            echo "export PATH=\"\$PATH:\$GREENLANG_HOME/cli/bin\"" >> "$PROFILE_FILE"
            echo -e "${GREEN}GreenLang environment variables added to $PROFILE_FILE${NC}"
        fi
    fi
}

# Verify installation
verify_installation() {
    echo -e "\n${BOLD}Verifying installation...${NC}"

    ERRORS=0

    # Check pre-commit
    if pre-commit --version &> /dev/null; then
        echo -e "${GREEN}✓ Pre-commit hooks${NC}"
    else
        echo -e "${RED}✗ Pre-commit hooks${NC}"
        ((ERRORS++))
    fi

    # Check Python linters
    if pylint --version &> /dev/null; then
        echo -e "${GREEN}✓ Pylint${NC}"
    else
        echo -e "${RED}✗ Pylint${NC}"
        ((ERRORS++))
    fi

    # Check Node linters
    if eslint --version &> /dev/null; then
        echo -e "${GREEN}✓ ESLint${NC}"
    else
        echo -e "${RED}✗ ESLint${NC}"
        ((ERRORS++))
    fi

    # Check OPA
    if opa version &> /dev/null; then
        echo -e "${GREEN}✓ OPA${NC}"
    else
        echo -e "${YELLOW}⚠ OPA (optional)${NC}"
    fi

    # Check infrastructure tools
    if terraform version &> /dev/null; then
        echo -e "${GREEN}✓ Terraform${NC}"
    else
        echo -e "${YELLOW}⚠ Terraform (optional)${NC}"
    fi

    if shellcheck --version &> /dev/null; then
        echo -e "${GREEN}✓ ShellCheck${NC}"
    else
        echo -e "${YELLOW}⚠ ShellCheck (optional)${NC}"
    fi

    # Check GreenLang CLI
    if greenlang --version &> /dev/null; then
        echo -e "${GREEN}✓ GreenLang CLI${NC}"
    else
        echo -e "${YELLOW}⚠ GreenLang CLI${NC}"
    fi

    if [ $ERRORS -eq 0 ]; then
        echo -e "\n${GREEN}${BOLD}Installation successful!${NC}"
        return 0
    else
        echo -e "\n${YELLOW}${BOLD}Installation completed with $ERRORS errors${NC}"
        return 1
    fi
}

# Display next steps
display_next_steps() {
    echo -e "\n${BOLD}===================================================${NC}"
    echo -e "${BOLD}  Setup Complete!${NC}"
    echo -e "${BOLD}===================================================${NC}\n"

    echo -e "${BOLD}Next Steps:${NC}"
    echo -e "  1. Restart your terminal or run: ${YELLOW}source ~/.bashrc${NC} (or ~/.zshrc)"
    echo -e "  2. Run: ${YELLOW}greenlang --help${NC} to see available commands"
    echo -e "  3. Test pre-commit hooks: ${YELLOW}pre-commit run --all-files${NC}"
    echo -e "  4. Check IUM score: ${YELLOW}greenlang ium calculate${NC}"
    echo -e "  5. Review configuration: ${YELLOW}cat .greenlang/config/dev.yaml${NC}\n"

    echo -e "${BOLD}Documentation:${NC}"
    echo -e "  - Developer Guide: ${YELLOW}.greenlang/deployment/dev/README.md${NC}"
    echo -e "  - Troubleshooting: ${YELLOW}.greenlang/deployment/docs/troubleshooting.md${NC}"
    echo -e "  - OPA Policies: ${YELLOW}.greenlang/enforcement/opa-policies/${NC}\n"

    echo -e "${BOLD}Support:${NC}"
    echo -e "  - Issues: https://github.com/greenlang/greenlang/issues"
    echo -e "  - Slack: #greenlang-support\n"
}

# Cleanup on error
cleanup() {
    if [ -f /tmp/opa.pid ]; then
        kill $(cat /tmp/opa.pid) 2>/dev/null || true
        rm /tmp/opa.pid
    fi
}

trap cleanup EXIT

# Main execution
main() {
    detect_os
    check_prerequisites
    install_precommit_hooks
    install_linters
    configure_ide
    setup_opa
    install_cli_tools
    verify_installation
    display_next_steps
}

main "$@"
