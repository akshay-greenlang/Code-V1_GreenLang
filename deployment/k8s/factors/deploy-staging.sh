#!/usr/bin/env bash
# =============================================================================
# GreenLang Factors API — Staging deployer (DEP6)
# =============================================================================
#
# WHAT THIS DOES
#   1. Confirms you're pointed at the staging cluster (NEVER prod).
#   2. Renders the staging kustomize overlay to build/staging-rendered.yaml.
#   3. Diffs the rendered manifest vs the live cluster state.
#   4. Applies with server-side mode and force-conflicts.
#   5. Waits for the factors-api rollout to finish.
#   6. Runs smoke tests against /v1/health and /v1/health/signing-status.
#
# WHAT THIS DOES NOT DO
#   * It never touches production.
#   * It never exports secrets to disk.
#   * It never disables SignedReceiptsMiddleware.
#
# USAGE
#   ./deployment/k8s/factors/deploy-staging.sh                    # full run
#   DRY_RUN=1 ./deployment/k8s/factors/deploy-staging.sh          # skip apply
#   STAGING_HOST=my-host ./deployment/k8s/factors/deploy-staging.sh
#   SMOKE_API_KEY=... ./deployment/k8s/factors/deploy-staging.sh  # for /signing-status
#
# REQUIRES
#   * kubectl >= 1.28
#   * kustomize >= 5.0   (bundled with kubectl; `kubectl kustomize` is used)
#   * jq, curl
#
# EXPECTED KUBE CONTEXT
#   Anything whose name contains "staging" (case-insensitive). The prompt
#   will abort on contexts containing "prod", "production", or "prd".
# =============================================================================

set -euo pipefail

# ----------------------------------------------------------------------------- #
# Config
# ----------------------------------------------------------------------------- #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OVERLAY_PATH="${SCRIPT_DIR}/overlays/staging"
BUILD_DIR="${REPO_ROOT}/build"
RENDERED="${BUILD_DIR}/staging-rendered.yaml"

STAGING_NAMESPACE="${STAGING_NAMESPACE:-factors-staging}"
STAGING_DEPLOYMENT="${STAGING_DEPLOYMENT:-staging-factors-api}"
STAGING_HOST="${STAGING_HOST:-factors-staging.greenlang.com}"
ROLLOUT_TIMEOUT="${ROLLOUT_TIMEOUT:-5m}"
DRY_RUN="${DRY_RUN:-0}"

# ----------------------------------------------------------------------------- #
# Log helpers (stderr so they never pollute rendered YAML on stdout)
# ----------------------------------------------------------------------------- #
_c_reset=$'\033[0m'; _c_red=$'\033[31m'; _c_grn=$'\033[32m'
_c_ylw=$'\033[33m'; _c_blu=$'\033[34m'; _c_bld=$'\033[1m'

log()  { printf '%s[factors-staging]%s %s\n' "$_c_blu" "$_c_reset" "$*" >&2; }
ok()   { printf '%s[factors-staging]%s %s%s\n' "$_c_grn" "$_c_reset" "$*" "$_c_reset" >&2; }
warn() { printf '%s[factors-staging]%s %s%s\n' "$_c_ylw" "$_c_reset" "$*" "$_c_reset" >&2; }
die()  { printf '%s[factors-staging]%s %sERROR:%s %s\n' "$_c_red" "$_c_reset" "$_c_bld" "$_c_reset" "$*" >&2; exit 1; }

# ----------------------------------------------------------------------------- #
# Preflight
# ----------------------------------------------------------------------------- #
require_bin() {
    command -v "$1" >/dev/null 2>&1 || die "missing required binary: $1"
}

preflight() {
    log "preflight: checking required tools"
    require_bin kubectl
    require_bin curl
    require_bin jq

    log "preflight: confirming kubeconfig context is staging"
    local ctx
    ctx="$(kubectl config current-context 2>/dev/null || true)"
    [[ -n "$ctx" ]] || die "no kubeconfig context set — run 'kubectl config use-context <staging-ctx>'"

    local lc_ctx
    lc_ctx="$(printf '%s' "$ctx" | tr '[:upper:]' '[:lower:]')"

    if [[ "$lc_ctx" == *prod* || "$lc_ctx" == *production* || "$lc_ctx" == *prd* ]]; then
        die "kubeconfig context '$ctx' looks like production — refusing to proceed"
    fi
    if [[ "$lc_ctx" != *staging* && "$lc_ctx" != *stg* ]]; then
        die "kubeconfig context '$ctx' does not contain 'staging' or 'stg' — refusing to proceed. If this is intentional, set STAGING_CONTEXT_OVERRIDE=1 and re-run."
    fi
    ok "kubeconfig context: $ctx"

    log "preflight: confirming staging namespace is reachable"
    if ! kubectl get ns "$STAGING_NAMESPACE" >/dev/null 2>&1; then
        warn "namespace '$STAGING_NAMESPACE' does not exist yet — it will be created by the apply step"
    fi

    mkdir -p "$BUILD_DIR"
}

# ----------------------------------------------------------------------------- #
# Render
# ----------------------------------------------------------------------------- #
render() {
    log "render: kustomize -> $RENDERED"
    kubectl kustomize "$OVERLAY_PATH" >"$RENDERED"
    local lines
    lines="$(wc -l <"$RENDERED" | tr -d ' ')"
    ok "rendered $lines lines of YAML"

    log "render: sanity-check no literal secrets leaked into the manifest"
    if grep -E '(BEGIN (RSA|EC|OPENSSH|PRIVATE) KEY|sk_live|whsec_|ed25519-priv:)' "$RENDERED" >/dev/null; then
        die "rendered manifest contains secret-looking content. Aborting. Inspect $RENDERED."
    fi
    ok "no inline secrets detected"
}

# ----------------------------------------------------------------------------- #
# Diff
# ----------------------------------------------------------------------------- #
diff_vs_cluster() {
    log "diff: rendered vs live cluster state"
    # `kubectl diff` exits 1 when there are differences and 0 when there are
    # none. Either is fine for us — we only fail on exit >=2 (connection err).
    set +e
    kubectl diff -f "$RENDERED" >"${BUILD_DIR}/staging-diff.txt" 2>&1
    local rc=$?
    set -e
    case "$rc" in
        0) ok "no differences detected — apply will be a no-op" ;;
        1) ok "differences detected (see $BUILD_DIR/staging-diff.txt)"; head -80 "${BUILD_DIR}/staging-diff.txt" >&2 ;;
        *) cat "${BUILD_DIR}/staging-diff.txt" >&2; die "kubectl diff failed (rc=$rc)" ;;
    esac
}

# ----------------------------------------------------------------------------- #
# Apply
# ----------------------------------------------------------------------------- #
apply_manifest() {
    if [[ "$DRY_RUN" == "1" ]]; then
        warn "DRY_RUN=1 — skipping apply"
        return
    fi
    log "apply: server-side with --force-conflicts"
    kubectl apply \
        -f "$RENDERED" \
        --server-side \
        --force-conflicts \
        --field-manager=factors-staging-deployer
    ok "apply succeeded"
}

# ----------------------------------------------------------------------------- #
# Rollout
# ----------------------------------------------------------------------------- #
wait_rollout() {
    if [[ "$DRY_RUN" == "1" ]]; then return; fi
    log "rollout: waiting for $STAGING_DEPLOYMENT (timeout=$ROLLOUT_TIMEOUT)"
    kubectl -n "$STAGING_NAMESPACE" rollout status \
        "deployment/$STAGING_DEPLOYMENT" \
        --timeout="$ROLLOUT_TIMEOUT"
    ok "rollout complete"
}

# ----------------------------------------------------------------------------- #
# Smoke tests
# ----------------------------------------------------------------------------- #
smoke() {
    if [[ "$DRY_RUN" == "1" ]]; then return; fi

    log "smoke: GET https://$STAGING_HOST/v1/health"
    local health
    health="$(curl -sSf --max-time 10 "https://${STAGING_HOST}/v1/health")" \
        || die "smoke: /v1/health failed"
    printf '%s\n' "$health" | jq . >&2
    local status
    status="$(printf '%s' "$health" | jq -r '.status // empty')"
    [[ "$status" == "ok" ]] || die "smoke: /v1/health did not return status=ok (got '$status')"
    ok "/v1/health OK"

    log "smoke: GET https://$STAGING_HOST/v1/health/signing-status"
    if [[ -z "${SMOKE_API_KEY:-}" ]]; then
        warn "SMOKE_API_KEY not set — skipping signing-status smoke (endpoint requires auth)"
        warn "re-run with SMOKE_API_KEY=<staging-key> to exercise the full path"
        return
    fi
    local signing
    signing="$(curl -sSf --max-time 10 \
        -H "X-API-Key: $SMOKE_API_KEY" \
        "https://${STAGING_HOST}/v1/health/signing-status")" \
        || die "smoke: /v1/health/signing-status failed"
    printf '%s\n' "$signing" | jq . >&2
    local installed alg rot
    installed="$(printf '%s' "$signing" | jq -r '.signing_installed // false')"
    alg="$(printf '%s' "$signing" | jq -r '.alg // empty')"
    rot="$(printf '%s' "$signing" | jq -r '.rotation_status // empty')"
    [[ "$installed" == "true" ]] || die "smoke: signing_installed=false — Vault sync likely failed"
    [[ "$alg" == "ed25519" ]] || die "smoke: unexpected signing alg: $alg"
    case "$rot" in
        current|due) ok "signing rotation_status=$rot — healthy" ;;
        overdue)     warn "signing rotation_status=overdue — schedule rotation ASAP" ;;
        *)           warn "signing rotation_status=$rot — investigate" ;;
    esac
    ok "/v1/health/signing-status OK"
}

# ----------------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------------- #
main() {
    log "=== GreenLang Factors API — staging deploy ==="
    log "overlay : $OVERLAY_PATH"
    log "rendered: $RENDERED"
    log "ns      : $STAGING_NAMESPACE"
    log "deploy  : $STAGING_DEPLOYMENT"
    log "host    : $STAGING_HOST"
    log "dry-run : $DRY_RUN"

    preflight
    render
    diff_vs_cluster
    apply_manifest
    wait_rollout
    smoke

    ok "=== staging deploy complete ==="
}

main "$@"
