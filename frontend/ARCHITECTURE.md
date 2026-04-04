# Frontend layout

## Canonical V2.2 enterprise shell

The package at `frontend/` is the **canonical** GreenLang V2.2 enterprise shell: React + MUI, shared chrome in `src/components/ShellLayout.tsx`, workspace pages under `src/pages/`, and design tokens plus reusable chrome primitives in `packages/shell-ui` (`@greenlang/shell-ui`).

Quality gates for this shell live in `.github/workflows/v2-frontend-ux-ci.yml` (`v2-2-shell-quality`).

## Legacy VCCI Scope3 app UI

`applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/frontend/` is a **separate** frontend package (app-local). It is still linted/tested in CI for regression safety but is not the enterprise shell surface. Prefer extending the canonical shell for cross-app UX, compliance rail, and run-center behavior.
