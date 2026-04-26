package greenlang

import rego.v1

# Repository-level governance entrypoint used by .github/workflows/code-governance.yml.
# Specific policy suites live under policies/greenlang-first and policies/guardrails.
default allow := true
