#!/usr/bin/env bash
set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the root directory (parent of scripts directory)
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "Checking version consistency for GreenLang..."
echo "============================================="

# 1. Read the VERSION file
if [ ! -f "VERSION" ]; then
    echo -e "${RED}ERROR: VERSION file not found in root directory${NC}"
    exit 1
fi

ROOT_VERSION="$(cat VERSION | tr -d ' \n\r')"
echo -e "Root VERSION file: ${GREEN}$ROOT_VERSION${NC}"

# 2. Check pyproject.toml is using dynamic version
echo -n "Checking pyproject.toml... "
if grep -E '^[[:space:]]*version[[:space:]]*=' pyproject.toml >/dev/null 2>&1; then
    echo -e "${RED}FAIL${NC}"
    echo -e "${RED}ERROR: pyproject.toml must not set a hardcoded [project].version${NC}"
    echo "       It should use dynamic = [\"version\"] instead"
    exit 1
fi

# Check that dynamic version is configured
if ! grep -q 'dynamic.*=.*\["version"\]' pyproject.toml 2>/dev/null && \
   ! grep -q 'dynamic.*=.*\[.*"version"' pyproject.toml 2>/dev/null; then
    echo -e "${YELLOW}WARNING${NC}"
    echo -e "${YELLOW}WARNING: pyproject.toml may not have dynamic version configured${NC}"
else
    echo -e "${GREEN}OK${NC}"
fi

# 3. Check setup.py doesn't hardcode version
echo -n "Checking setup.py... "
if [ -f "setup.py" ]; then
    if grep -E "version[[:space:]]*=[[:space:]]*['\"]([0-9]+\.){2}[0-9]+" setup.py >/dev/null 2>&1; then
        echo -e "${RED}FAIL${NC}"
        echo -e "${RED}ERROR: setup.py hardcodes a version. It should read from VERSION file${NC}"
        exit 1
    fi
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${YELLOW}Not found (OK)${NC}"
fi

# 4. Check VERSION.md mentions current version
echo -n "Checking VERSION.md... "
if [ -f "VERSION.md" ]; then
    if grep -q "$ROOT_VERSION" VERSION.md; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${YELLOW}WARNING${NC}"
        echo -e "${YELLOW}WARNING: VERSION.md doesn't mention current version $ROOT_VERSION${NC}"
    fi
else
    echo -e "${YELLOW}Not found${NC}"
fi

# 5. Check Dockerfile uses GL_VERSION build arg
echo -n "Checking Dockerfiles... "
DOCKERFILE_COUNT=0
DOCKERFILE_OK=0
for dockerfile in $(find . -name "Dockerfile*" -type f 2>/dev/null); do
    DOCKERFILE_COUNT=$((DOCKERFILE_COUNT + 1))
    if grep -q "ARG GL_VERSION" "$dockerfile"; then
        DOCKERFILE_OK=$((DOCKERFILE_OK + 1))
    else
        echo -e "${YELLOW}WARNING${NC}"
        echo -e "${YELLOW}WARNING: $dockerfile doesn't use ARG GL_VERSION${NC}"
    fi
done
if [ $DOCKERFILE_COUNT -eq 0 ]; then
    echo -e "${YELLOW}No Dockerfiles found${NC}"
elif [ $DOCKERFILE_OK -eq $DOCKERFILE_COUNT ]; then
    echo -e "${GREEN}OK (${DOCKERFILE_COUNT} files)${NC}"
else
    echo -e "${YELLOW}${DOCKERFILE_OK}/${DOCKERFILE_COUNT} OK${NC}"
fi

# 6. Check if git tag matches VERSION (only if on a tag)
if git describe --exact-match --tags HEAD >/dev/null 2>&1; then
    echo -n "Checking git tag... "
    GIT_TAG="$(git describe --exact-match --tags HEAD)"
    EXPECTED_TAG="v${ROOT_VERSION}"
    if [ "$GIT_TAG" = "$EXPECTED_TAG" ]; then
        echo -e "${GREEN}OK ($GIT_TAG)${NC}"
    else
        echo -e "${RED}FAIL${NC}"
        echo -e "${RED}ERROR: Git tag $GIT_TAG doesn't match VERSION $ROOT_VERSION${NC}"
        echo "       Expected tag: $EXPECTED_TAG"
        exit 1
    fi
fi

# 7. Optional: Check if package can be imported and reports correct version
# This only works after installation
if command -v python >/dev/null 2>&1; then
    echo -n "Checking Python package version... "

    # Try to import and check version
    PYTHON_VERSION=$(python -c "
try:
    import greenlang
    print(greenlang.__version__)
except ImportError:
    print('NOT_INSTALLED')
except Exception as e:
    print(f'ERROR: {e}')
" 2>/dev/null || echo "ERROR")

    if [ "$PYTHON_VERSION" = "NOT_INSTALLED" ]; then
        echo -e "${YELLOW}Not installed (run 'pip install -e .' to test)${NC}"
    elif [ "$PYTHON_VERSION" = "$ROOT_VERSION" ]; then
        echo -e "${GREEN}OK${NC}"
    elif [ "$PYTHON_VERSION" = "ERROR" ]; then
        echo -e "${YELLOW}Error checking version${NC}"
    else
        echo -e "${RED}FAIL${NC}"
        echo -e "${RED}ERROR: Python package reports version $PYTHON_VERSION but VERSION file has $ROOT_VERSION${NC}"
        exit 1
    fi
fi

# Summary
echo ""
echo "============================================="
echo -e "${GREEN}Version consistency check complete!${NC}"
echo -e "Version: ${GREEN}$ROOT_VERSION${NC}"
echo ""
echo "To bump version:"
echo "  1. Edit VERSION file with new version"
echo "  2. Update VERSION.md with release notes"
echo "  3. Commit: git commit -m 'chore(release): bump to X.Y.Z'"
echo "  4. Tag: git tag vX.Y.Z && git push --tags"
echo "  5. CI will build and publish with correct version"