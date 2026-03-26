# GreenLang v1 Multi-App Information Architecture

## Purpose

Define the minimum enterprise IA for a single frontend entry point that provides access to `GL-CBAM-APP`, `GL-CSRD-APP`, and `GL-VCCI-Carbon-APP` without breaking existing CBAM behavior.

## Product Surface

- `Platform Home`: global overview, app switcher, health summary.
- `App Workspaces`: dedicated workspace per app with consistent run pattern.
- `Run Center`: cross-app run history and status.
- `Artifact Center`: download-ready outputs by run.
- `Policy and Validation`: per-run status cards with consistent terminology.

## Primary Navigation

- `Home`
- `CBAM`
- `CSRD`
- `VCCI`
- `Runs`

## Interaction Model

- User selects an app workspace.
- User submits run input(s) for that app.
- UI shows run status and execution mode (`native` or `fallback`).
- UI renders app-specific summary and common artifact list.
- User downloads zip bundle or individual artifacts.

## Global UX Rules

- Do not hide app limitations. If an app surface is partial, show explicit readiness notes.
- Preserve deterministic evidence semantics from CLI (`manifest`, `checksums`, audit artifacts).
- Keep one error model for all workspaces: `title`, `message`, `details`.
- Keep one status model for all workspaces: `queued`, `running`, `completed`, `failed`.

## Must-Change-Only Constraint

Changes are limited to:

1. frontend shell and routing,
2. adapter endpoints needed for cross-app access,
3. docs/runbooks needed to make usage explicit and testable.

Avoid deep backend rewrites unless required to unblock user-visible flow.
