# GreenLang V2 Enterprise UX Shell

## UX Objective

Provide a single enterprise control surface for 6-8 apps with consistent behavior, policy visibility, and operational traceability.

## Required Product Surfaces

- global app switcher with role-aware visibility
- unified run center with cross-app filtering
- policy and compliance status rail
- artifact explorer and deterministic diff surface
- connector health and incident banner system

## Shared Interaction Contract

- consistent run lifecycle states (`queued`, `running`, `completed`, `failed`)
- consistent status chips (`PASS`, `WARN`, `FAIL`)
- consistent error envelope (`title`, `message`, `details`)
- consistent download semantics and export gating

## V2 UX Quality Gates

1. lint/type/test/build must pass for frontend packages.
2. e2e workflow must pass key operator journeys for `/apps` and app workspaces.
3. visual regression checks must pass baseline shell snapshots.
4. accessibility and keyboard navigation checks for critical workflows.

## V2.2 Implementation Status

- root React shell serves `/apps/*`, `/runs`, `/governance`, and `/admin` from FastAPI SPA fallback.
- command palette (`Ctrl/Cmd+K`) is enabled for app and platform route navigation.
- run center includes an interactive pipeline DAG (keyboard-focusable stages, evidence deep-links), linear stage progress, bundle/artifact links, and checksum-based artifact diffing.
- workspaces use a shared runner with per-app regulatory copy, file hints, and checklist sidecars (`workspaceConfig`).
- run lifecycle exposes server fields `run_state` (`completed`, `failed`, `blocked`, `partial_success`) plus `can_export`, `warnings`, and `errors` on run APIs.
- admin console (`/admin`) surfaces `/health`, session counts, `docs/v2/RELEASE_TRAIN_LOCAL_EVIDENCE.json` (when present), and `applications/connectors/v2_connector_registry.yaml` via `/api/v1/admin/*`.
- shell chrome (`GET /api/v1/shell/chrome-context`) feeds the compliance rail and connector incident banner from pack, agent, policy bundle, and connector registry signals.
- governance center reads live pack tier, agent lifecycle, and policy bundle metadata from `/api/v1/governance/*`.
- design tokens and theme live in the internal package `@greenlang/shell-ui` (`frontend/packages/shell-ui`).
- Playwright + axe-core run in CI (`frontend/e2e`, job `v2-2-shell-quality`); Vitest covers unit smoke and authz; Storybook builds in the same job (`npm run build-storybook`).
- Global API health strip: `/health` polled from shell chrome; warning banner when checks fail (connector-specific drill-down remains on Admin).
- role-aware routing is enforced in both navigation visibility and route guards; role `Select` uses `InputLabel` for accessible naming.
- realtime status uses SSE heartbeat with reconnect/degraded state handling in workspace views; live status is exposed to assistive tech via `aria-live`.

## Implemented vs still evolving

| Area | Status |
|------|--------|
| Platform backend, CI, docs closure pack | Mature |
| Admin + connector + release-train visibility | Implemented (read-only; evidence file optional) |
| Interactive DAG + evidence links | Implemented |
| Per-app workspace differentiation | Implemented (config-driven panels; not separate SPA per app) |
| Full design system / Storybook | Storybook 8 for `@greenlang/shell-ui` primitives (`frontend/.storybook`, `npm run storybook`) |
| Pixel visual regression (frontend) | Playwright `e2e/visual-shell.spec.ts` on Linux CI when PNG baselines are committed; maintainer workflow uploads artifacts (`v2-shell-visual-snapshots.yml`) |
| WCAG audit coverage | Automated axe (incl. color-contrast) on CBAM, runs, governance, admin; manual AA checklist in `docs/v2/WCAG_SHELL_CHECKLIST.md` |
