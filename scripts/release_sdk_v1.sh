#!/usr/bin/env bash
# Release the GreenLang Factors SDK v1.0.0 to PyPI + npm.
#
# Defaults to a dry run. Real publishes require:
#   - PYPI_TOKEN (or ~/.pypirc) for `twine upload`
#   - NPM_TOKEN (or `npm login`) for `npm publish`
#
# Usage:
#   ./scripts/release_sdk_v1.sh             # dry-run (build + check, no publish)
#   ./scripts/release_sdk_v1.sh --commit    # real publish
#   ./scripts/release_sdk_v1.sh --commit --skip-npm   # only PyPI
#   ./scripts/release_sdk_v1.sh --commit --skip-pypi  # only npm

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_DIR="$REPO_ROOT/greenlang/factors/sdk/python"
TS_DIR="$REPO_ROOT/greenlang/factors/sdk/ts"

COMMIT=0
SKIP_PYPI=0
SKIP_NPM=0

for arg in "$@"; do
  case "$arg" in
    --commit)    COMMIT=1 ;;
    --skip-pypi) SKIP_PYPI=1 ;;
    --skip-npm)  SKIP_NPM=1 ;;
    --help|-h)
      sed -n '2,17p' "$0"
      exit 0
      ;;
    *) echo "Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

mode="DRY-RUN"
if [[ $COMMIT -eq 1 ]]; then mode="COMMIT"; fi

echo "==> Releasing Factors SDK v1.0.0 ($mode)"
echo

# --- Python ---------------------------------------------------------------
if [[ $SKIP_PYPI -eq 0 ]]; then
  echo "==> Python SDK"
  pushd "$PY_DIR" >/dev/null

  # Verify version pin
  py_version=$(grep -E '^version = ' pyproject.toml | head -1 | cut -d'"' -f2)
  echo "    pyproject version: $py_version"
  if [[ "$py_version" != "1.0.0" ]]; then
    echo "ERROR: pyproject.toml version is $py_version, expected 1.0.0" >&2
    exit 1
  fi

  echo "    Building wheel + sdist..."
  rm -rf dist/ build/
  python -m build

  echo "    twine check..."
  python -m twine check dist/*

  if [[ $COMMIT -eq 1 ]]; then
    echo "    Publishing to PyPI..."
    python -m twine upload dist/*
  else
    echo "    DRY: would run: python -m twine upload dist/*"
  fi

  popd >/dev/null
  echo
fi

# --- TypeScript -----------------------------------------------------------
if [[ $SKIP_NPM -eq 0 ]]; then
  echo "==> TypeScript SDK"
  pushd "$TS_DIR" >/dev/null

  ts_version=$(node -p "require('./package.json').version")
  echo "    package.json version: $ts_version"
  if [[ "$ts_version" != "1.0.0" ]]; then
    echo "ERROR: package.json version is $ts_version, expected 1.0.0" >&2
    exit 1
  fi

  echo "    npm install..."
  npm install --no-fund --no-audit

  echo "    npm run build..."
  npm run build

  echo "    npm test..."
  npm test --silent || { echo "TS tests failed"; exit 1; }

  if [[ $COMMIT -eq 1 ]]; then
    echo "    npm publish --access public..."
    npm publish --access public
  else
    echo "    DRY: would run: npm publish --access public"
  fi

  popd >/dev/null
fi

echo
echo "==> Done. Tag and push:"
echo "    git tag -s factors-sdk/v1.0.0 -m 'Factors SDK v1.0.0'"
echo "    git push origin factors-sdk/v1.0.0"
