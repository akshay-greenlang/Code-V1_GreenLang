# GreenLang v1 Multi-App UX Contract

## Scope

This contract defines shared UX behavior for the multi-app web shell covering:

- `GL-CBAM-APP`
- `GL-CSRD-APP`
- `GL-VCCI-Carbon-APP`

## Shared Run Contract

Every workspace must return a normalized run payload:

- `run_id`: unique run identifier
- `app_id`: `cbam | csrd | vcci`
- `success`: boolean
- `status`: `completed | failed`
- `execution_mode`: `native | fallback | mixed`
- `artifacts`: list of relative artifact paths
- `warnings`: list of warning strings
- `errors`: list of error strings
- `summary`: app-specific object rendered by workspace

## Shared UI States

- `idle`: form ready for input
- `running`: progress indicator visible, submit disabled
- `completed`: status card + artifacts visible
- `failed`: error card visible with actionable text

## Shared Artifact Behavior

- Individual artifact download per row.
- Bundle download (`zip`) where export is allowed.
- If export is blocked, disable bundle action with reason.

## Shared Validation and Policy Cards

Use consistent status chips:

- `PASS`
- `WARN`
- `FAIL`

Use consistent tones:

- PASS: green
- WARN: amber
- FAIL: red

## Security and Error Hygiene

- Never expose local filesystem paths in user errors.
- Validate uploaded filenames and file sizes.
- Reject invalid session/run identifiers.
- Enforce API key checks when configured.

## Non-Goals

- Full cross-app enterprise IAM in this pass.
- Deep workflow redesign for each app backend.
- Replacing existing CLI release gates.
